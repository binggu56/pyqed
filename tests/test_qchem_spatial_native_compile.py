import numpy as np
import pytest

from pyqed.mps import cpp_davidson
from pyqed.qchem.dmrg.backends.reduced import (
    build_spatial_complementary_operator_families,
)
from pyqed.qchem.dmrg.spatial_terms import (
    dense_from_spatial_term_map,
    spatial_complementary_family_term_maps,
    spatial_local_ops,
)


def test_cpp_spatial_qchem_family_entries_match_python_reference():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.build_spatial_qchem_family_entries is None
    ):
        pytest.skip("C++ qchem family compiler is unavailable")

    h1 = np.array([[1.0, 0.2], [0.2, -0.3]])
    eri = np.zeros((2, 2, 2, 2), dtype=float)
    eri[0, 0, 0, 0] = 0.7
    eri[0, 1, 1, 0] = 0.11
    eri[1, 0, 0, 1] = -0.04
    eri[1, 1, 1, 1] = 0.5
    eri_spin = np.stack(
        (
            np.stack((eri, eri.copy())),
            np.stack((eri.copy(), eri.copy())),
        )
    )

    reference = build_spatial_complementary_operator_families(
        h1,
        eri_spin,
        cutoff=1.0e-12,
        include_half=True,
    )
    native = cpp_davidson.build_spatial_qchem_family_entries(
        h1,
        eri_spin,
        1.0e-12,
        True,
    )

    for name in ("S", "R", "A", "P", "B", "Q"):
        expected = dict(reference[name].entries)
        actual = dict(native["entries"][name])
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, abs=1.0e-14)


def test_cpp_spatial_qchem_family_term_maps_match_python_reference_dense():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.build_spatial_qchem_family_entries is None
        or cpp_davidson.build_spatial_qchem_family_term_maps is None
    ):
        pytest.skip("C++ qchem family compiler is unavailable")

    h1 = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.2, -0.3, 0.05],
            [-0.1, 0.05, 0.4],
        ]
    )
    eri = np.zeros((3, 3, 3, 3), dtype=float)
    eri[0, 0, 0, 0] = 0.7
    eri[0, 1, 1, 0] = 0.11
    eri[1, 0, 0, 1] = -0.04
    eri[2, 1, 0, 2] = 0.08
    eri[1, 2, 2, 1] = -0.03
    eri[2, 2, 2, 2] = 0.5
    eri_spin = np.stack(
        (
            np.stack((eri, eri.copy())),
            np.stack((eri.copy(), eri.copy())),
        )
    )

    reference = build_spatial_complementary_operator_families(
        h1,
        eri_spin,
        cutoff=1.0e-12,
        include_half=True,
    )
    expected_maps = spatial_complementary_family_term_maps(
        reference,
        cutoff=1.0e-12,
    )
    native_entries = cpp_davidson.build_spatial_qchem_family_entries(
        h1,
        eri_spin,
        1.0e-12,
        True,
    )
    native_maps = cpp_davidson.build_spatial_qchem_family_term_maps(
        native_entries["entries"],
        native_entries["n_sites"],
        1.0e-12,
    )
    spatial_local_ops().update(
        {
            str(name): np.asarray(matrix, dtype=complex)
            for name, matrix in dict(native_maps["local_ops"]).items()
        }
    )

    for name in ("R", "P"):
        expected = dense_from_spatial_term_map(expected_maps[name], 3)
        actual = dense_from_spatial_term_map(
            dict(native_maps["term_maps"][name]),
            3,
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_cpp_spatial_block2_carrier_shape():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.build_spatial_block2_carrier_mpo is None
    ):
        pytest.skip("C++ spatial carrier builder is unavailable")

    native = cpp_davidson.build_spatial_block2_carrier_mpo(2, 4)
    factors = list(native["factors"])
    assert len(factors) == 2
    assert factors[0].shape == (1, 1, 4, 4)
    assert np.allclose(factors[0][0, 0], np.eye(4))
    assert dict(native["info"])["qchem_compile_backend_actual"] == "cpp"


def _sparse_coefficients_from_spatial_term_map(term_map, n_sites, local_ops=None):
    ops = dict(spatial_local_ops())
    if local_ops:
        ops.update({str(name): np.asarray(matrix) for name, matrix in local_ops.items()})
    ident = ops["I"]
    out = {}
    for key, coeff in dict(term_map).items():
        symbol, dofs = key
        site_mats = [ident.copy() for _ in range(n_sites)]
        for token, site in zip(str(symbol).split(), tuple(dofs)):
            site = int(site)
            site_mats[site] = site_mats[site] @ ops[token]
        entries_by_site = []
        for matrix in site_mats:
            flat = np.asarray(matrix).reshape(-1)
            entries_by_site.append(
                [(idx, value) for idx, value in enumerate(flat) if abs(value) > 0.0]
            )

        def visit(site, index, scale):
            if site == n_sites:
                out[index] = out.get(index, 0.0j) + coeff * scale
                return
            for phys, value in entries_by_site[site]:
                visit(site + 1, index * 16 + int(phys), scale * value)

        visit(0, 0, 1.0 + 0.0j)
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def _sparse_coefficients_from_spatial_mpo(factors):
    states = {0: {0: 1.0 + 0.0j}}
    for factor in factors:
        core = np.asarray(factor)
        left_dim, right_dim, _, _ = core.shape
        nxt = {right: {} for right in range(right_dim)}
        for left in range(left_dim):
            left_states = states.get(left, {})
            if not left_states:
                continue
            for right in range(right_dim):
                block = core[left, right].reshape(-1)
                nz = [(idx, value) for idx, value in enumerate(block) if abs(value) > 1.0e-14]
                if not nz:
                    continue
                right_states = nxt[right]
                for prefix, prefix_value in left_states.items():
                    for phys, value in nz:
                        index = prefix * 16 + int(phys)
                        right_states[index] = right_states.get(index, 0.0j) + prefix_value * value
        states = {right: values for right, values in nxt.items() if values}
    assert set(states) == {0}
    return {
        key: value
        for key, value in states[0].items()
        if abs(value) > 1.0e-12
    }


def test_cpp_spatial_family_mpos_use_sparse_gram_tt_svd_for_six_sites():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.build_spatial_qchem_family_mpos is None
        or cpp_davidson.MovingEnvironment is None
    ):
        pytest.skip("C++ spatial family MPO builder is unavailable")

    n_sites = 6
    r_terms = {
        ("cu cdu", (0, 5)): 0.25,
        ("JW nu", (2, 4)): -0.4,
        ("n", (3,)): 0.7,
    }
    p_terms = {
        ("cdu cu cdd cd", (0, 1, 4, 5)): 0.2,
        ("cdd cd cdu cu", (1, 2, 3, 4)): -0.17,
        ("nu nd", (4, 5)): 0.09,
    }
    payload = {
        "term_maps": {
            "R": r_terms,
            "P": p_terms,
        },
        "local_ops": {},
    }

    owner = cpp_davidson.MovingEnvironment()
    descriptor = owner.install_spatial_qchem_family_descriptor(
        "six-site-qchem-families",
        payload,
        n_sites,
        1.0e-12,
        "auto",
        "first_site",
    )
    native = owner.build_spatial_qchem_family_mpos_from_descriptor(
        "six-site-qchem-families"
    )
    native_reused = owner.build_spatial_qchem_family_mpos_from_descriptor(
        "six-site-qchem-families"
    )
    assert descriptor["backend_actual"] == "cpp_spatial_qchem_family_descriptor"
    assert tuple(descriptor["family_names"]) == ("R", "P:g0", "P:g1")
    assert native["backend_actual"] == "cpp_spatial_sparse_tt_svd_family_mpos"
    assert native_reused["backend_actual"] == "cpp_spatial_sparse_tt_svd_family_mpos"
    assert native["descriptor_backend_actual"] == "cpp_spatial_qchem_family_descriptor"
    assert native["descriptor_key"] == "six-site-qchem-families"
    info = {
        str(name): dict(item)
        for name, item in dict(native_reused["family_mpo_info"]).items()
    }
    assert info["R"]["tt_svd_backend"] == "sparse_gram"
    assert info["R"]["tt_svd_gram_solver"] == "zheev"
    assert info["R"]["tt_svd_route_backend"] == "compact_sorted_columns"
    assert info["R"]["tt_svd_route_cache_backend"] == (
        "persistent_cpp_compact_route_cache"
    )
    assert info["R"]["tt_svd_route_cache_owner"] == (
        owner.spatial_route_plan_cache_owner_key()
    )
    assert info["R"]["tt_svd_route_cache_hits"] == n_sites - 1
    assert info["R"]["tt_svd_route_cache_misses"] == 0
    owner_cache_stats = dict(owner.spatial_route_plan_cache_stats())
    assert owner_cache_stats["records"] >= n_sites - 1
    assert owner_cache_stats["owner_key"] == owner.spatial_route_plan_cache_owner_key()
    assert len(info["R"]["tt_svd_route_columns_by_step"]) == n_sites - 1
    assert info["R"].get("compression_fallback_reason") is None
    assert info["R"]["tt_svd_dense_elements"] == 16**n_sites
    owner_stats = dict(owner.stats())
    assert owner_stats["spatial_qchem_family_descriptor_records"] == 1
    assert owner_stats["spatial_qchem_family_descriptor_installs"] == 1
    assert owner_stats["spatial_qchem_family_descriptor_mpo_builds"] == 2

    mpos = {
        str(name): list(factors)
        for name, factors in dict(native_reused["family_mpos"]).items()
    }
    families = {"R": r_terms, "P:g0": {}, "P:g1": {}}
    for key, value in p_terms.items():
        split_site = tuple(key[1])[0]
        group = int(split_site) * 2 // n_sites
        families[f"P:g{group}"][key] = value

    for name, terms in families.items():
        if not terms:
            continue
        expected = _sparse_coefficients_from_spatial_term_map(terms, n_sites)
        actual = _sparse_coefficients_from_spatial_mpo(mpos[name])
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, abs=1.0e-10)
