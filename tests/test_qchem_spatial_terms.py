import numpy as np
import pytest

from pyqed.qchem.dmrg.dmrg import (
    _build_spatial_active_hamiltonian_matrix,
    _normalize_spatial_family_environment_backend,
)
from pyqed.qchem.dmrg.backends.reduced import (
    build_spatial_complementary_operator_families,
    build_spatial_reduced_hamiltonian_mpo,
)
from pyqed.qchem.dmrg.spatial_terms import (
    accumulate_symbolic_term,
    merge_term_maps,
    spatial_local_ops,
    spatial_complementary_local_matrices,
    spatial_complementary_local_matrix,
    spatial_complementary_family_hamiltonian_term_map,
    spatial_complementary_family_term_maps,
    spatial_one_body_term_map,
    spatial_two_generator_family_term_map,
    spatial_two_body_term_map,
    spatial_two_body_spinfree_term_map,
)
from pyqed.mps.nonabelian import (
    AutoMPO,
    FullyReducedSpatialOrbitalSite,
    RankCoupledMPO,
    SpatialSpinFreeERIBuilder,
    physical_leg_from_spatial_orbital,
)
from pyqed.mps.nonabelian.models import (
    _dense_matrix_from_local_mpo,
    add_spatial_one_body_terms,
    add_spatial_two_generator_product_terms,
)


def _kron_all(operators):
    out = np.asarray(operators[0], dtype=complex)
    for operator in operators[1:]:
        out = np.kron(out, np.asarray(operator, dtype=complex))
    return out


def _dense_from_spatial_term_map(term_map, nsites):
    ops = spatial_local_ops()
    ident = ops["I"]
    dense = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        local = [ident.copy() for _ in range(nsites)]
        for piece, site in zip(symbol.split(), dofs):
            local[site] = ops[piece]
        dense += factor * _kron_all(local)
    return dense


def test_spatial_family_environment_backend_block2_aliases_use_family_mpos():
    assert _normalize_spatial_family_environment_backend(None) == "block2"
    assert _normalize_spatial_family_environment_backend("block2") == "block2"
    assert _normalize_spatial_family_environment_backend("autompo") == "block2"
    assert (
        _normalize_spatial_family_environment_backend("native_generators")
        == "block2_native"
    )
    assert (
        _normalize_spatial_family_environment_backend("adaptive_block2")
        == "block2_adaptive"
    )
    assert (
        _normalize_spatial_family_environment_backend("renormalized_generators")
        == "block2"
    )
    assert _normalize_spatial_family_environment_backend("none") == "none"


def test_spatial_complementary_operator_families_group_integrals():
    h1 = np.zeros((3, 3))
    h1[0, 2] = 0.05
    eri = np.zeros((2, 2, 3, 3, 3, 3))
    eri[:, :, 0, 1, 1, 2] = 0.4
    eri[:, :, 2, 0, 1, 1] = -0.2

    families = build_spatial_complementary_operator_families(h1, eri, cutoff=1.0e-12)

    assert families.names == ("S", "R", "A", "P", "B", "Q")
    assert families["P"].entries[(0, 1, 1, 2)] == pytest.approx(0.2)
    assert families["Q"].entries[(0, 2, 1)] == pytest.approx(-0.2)
    assert families["R"].entries[(0, 2)] == pytest.approx(-0.15)
    assert (0, 1) in families["A"].entries
    assert (1, 2) in families["A"].entries
    assert families.as_metadata()["families"]["P"]["n_terms"] == 2


def test_spatial_complementary_local_matrix_matches_two_site_hamiltonian():
    h1 = np.array(
        [
            [0.2, -0.03],
            [-0.03, -0.1],
        ]
    )
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.5
    eri[0, 0, 1, 1] = 0.2
    eri[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))

    families = build_spatial_complementary_operator_families(h1, h2, cutoff=1.0e-12)
    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_local = spatial_complementary_local_matrix(families, 0)
    channel_mats = spatial_complementary_local_matrices(families, 0)

    np.testing.assert_allclose(dense_local, dense_ref, atol=1.0e-12)
    np.testing.assert_allclose(
        sum(channel_mats.values()),
        dense_local,
        atol=1.0e-12,
    )
    assert set(channel_mats) == {"R", "P"}
    assert np.linalg.norm(channel_mats["R"]) > 0.0
    assert np.linalg.norm(channel_mats["P"]) > 0.0


def test_spatial_one_body_term_map_matches_dense_reference():
    h1 = np.array(
        [
            [0.2, -0.03, 0.04],
            [-0.03, -0.1, 0.07],
            [0.04, 0.07, 0.5],
        ]
    )
    h2 = np.zeros((2, 2, 3, 3, 3, 3))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_terms = _dense_from_spatial_term_map(spatial_one_body_term_map(h1), 3)

    np.testing.assert_allclose(dense_terms, dense_ref, atol=1.0e-12)


def test_spatial_two_body_term_map_matches_dense_reference():
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.5
    eri[0, 0, 1, 1] = 0.2
    eri[1, 1, 0, 0] = 0.2
    h1 = np.zeros((2, 2))
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_terms = _dense_from_spatial_term_map(spatial_two_body_term_map(eri), 2)

    np.testing.assert_allclose(dense_terms, dense_ref, atol=1.0e-12)


def test_spatial_two_body_spinfree_term_map_matches_component_reference():
    rng = np.random.default_rng(7)
    eri = rng.normal(size=(3, 3, 3, 3))

    dense_component = _dense_from_spatial_term_map(spatial_two_body_term_map(eri), 3)
    dense_spinfree = _dense_from_spatial_term_map(
        spatial_two_body_spinfree_term_map(eri),
        3,
    )

    np.testing.assert_allclose(dense_spinfree, dense_component, atol=1.0e-12)


def test_spatial_complementary_family_term_maps_reconstruct_hamiltonian_terms():
    rng = np.random.default_rng(19)
    h1 = rng.normal(size=(3, 3))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(3, 3, 3, 3))
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))
    families = build_spatial_complementary_operator_families(
        h1,
        h2,
        cutoff=1.0e-12,
        include_half=True,
    )

    family_terms = spatial_complementary_family_hamiltonian_term_map(families)
    reference_terms = merge_term_maps(
        spatial_one_body_term_map(h1),
        spatial_two_body_spinfree_term_map(eri),
    )

    assert len(family_terms) == len(reference_terms)
    np.testing.assert_allclose(
        _dense_from_spatial_term_map(family_terms, 3),
        _dense_from_spatial_term_map(reference_terms, 3),
        atol=1.0e-12,
    )
    family_counts = {
        name: len(term_map)
        for name, term_map in spatial_complementary_family_term_maps(families).items()
    }
    assert set(family_counts) == {"R", "P"}
    assert family_counts["R"] > 0
    assert family_counts["P"] > 0


def test_native_spatial_generator_family_mpos_match_term_maps():
    h1 = np.array(
        [
            [0.2, -0.03],
            [0.04, -0.1],
        ]
    )
    p_entries = {
        (0, 0, 1, 1): 0.35,
        (0, 1, 1, 0): -0.2,
    }
    leg = physical_leg_from_spatial_orbital()

    r_builder = AutoMPO([leg, leg])
    add_spatial_one_body_terms(r_builder, h1, cutoff=1.0e-12, family="R")
    r_dense = _dense_matrix_from_local_mpo(r_builder.build())
    np.testing.assert_allclose(
        r_dense,
        _dense_from_spatial_term_map(spatial_one_body_term_map(h1), 2),
        atol=1.0e-12,
    )

    p_builder = AutoMPO([leg, leg])
    add_spatial_two_generator_product_terms(
        p_builder,
        p_entries,
        cutoff=1.0e-12,
        family="P",
    )
    p_dense = _dense_matrix_from_local_mpo(p_builder.build())
    np.testing.assert_allclose(
        p_dense,
        _dense_from_spatial_term_map(spatial_two_generator_family_term_map(p_entries), 2),
        atol=1.0e-12,
    )


def test_merge_term_maps_cancels_near_zero_terms():
    first = {}
    second = {}
    accumulate_symbolic_term(first, "n", [0], 0.5)
    accumulate_symbolic_term(second, "n", [0], -0.5)

    assert merge_term_maps(first, second) == {}


def test_fully_reduced_spatial_reduced_hamiltonian_builds_one_body_only():
    h1 = np.diag([0.3, -0.2])

    result = build_spatial_reduced_hamiltonian_mpo(
        h1,
        eri=None,
        fully_reduced=True,
        n_elec=2,
        spin=0,
        ecore=-1.2,
        orb_sym=(1, 1),
    )

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body"] is False
    assert len(result.factors) == 2
    assert result.mpo is result.factors
    assert result.ncas == 2
    assert result.ecore == pytest.approx(-1.2)
    assert result.initialize_system_kwargs() == {
        "n_sites": 2,
        "n_elec": 2,
        "spin": 0,
        "orb_sym": (1, 1),
    }
    assert result.info["block_hamiltonian"] is True


def test_fully_reduced_spatial_reduced_hamiltonian_builds_four_distinct_eri_strings():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 2, 3] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body"] is True
    assert result.info["two_body_reduced_string_terms"] > 0


def test_fully_reduced_spatial_reduced_hamiltonian_builds_diagonal_density_eri_terms():
    h1 = np.zeros((3, 3))
    eri = np.zeros((2, 2, 3, 3, 3, 3))
    eri[:, :, 0, 0, 1, 1] = 0.2
    eri[:, :, 2, 2, 2, 2] = 0.3

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_density_bilinear_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 0, 1, 2] = 0.2
    eri[:, :, 1, 2, 3, 3] = -0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_bilinear_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_bilinear_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_endpoint_density_bilinear_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 0, 1, 0] = 0.1
    eri[:, :, 1, 0, 1, 1] = -0.2

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_bilinear_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_bilinear_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_pair_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 0, 2] = 0.1
    eri[:, :, 0, 1, 2, 1] = -0.2
    eri[:, :, 2, 3, 2, 3] = 0.3

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_pair_terms"] == 3
    assert result.info["two_body_representation"] == "fully_reduced_pair_eri"


def test_fully_reduced_pair_eri_keeps_rank_coupled_chain_with_dense_prefix():
    leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 1, 0, 2] = 0.1
    autompo = AutoMPO([leg] * 3)

    SpatialSpinFreeERIBuilder([leg] * 3, eri).add_to(autompo)
    factors = autompo.build()

    assert factors
    assert all(isinstance(factor, RankCoupledMPO) for factor in factors)


def test_fully_reduced_spatial_reduced_hamiltonian_builds_exchange_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 2, 0] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_exchange_terms"] > 0
    assert result.info["two_body_representation"] == "fully_reduced_exchange_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_exchange_with_one_body_correction():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 1, 2] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_exchange_terms"] > 0
    assert result.info["two_body_one_body_correction_terms"] == 1
