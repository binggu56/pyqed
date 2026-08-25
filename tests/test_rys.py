import math

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem import basis as basis_module
try:
    from pyqed.qchem import _rys_cy
except ImportError:  # pragma: no cover - optional accelerator
    _rys_cy = None
from pyqed.qchem.basis import (
    ContractedGaussian,
    ERI,
    electron_repulsion,
    make_contractions,
    parse_gbs,
    _basis_path,
)
from pyqed.qchem.rys import (
    _contracted_eri_pppp_rys_cached,
    _contracted_eri_ppss_rys_cached,
    boys,
    contracted_eri_cartesian_rys,
    contracted_eri_pppp_rys,
    contracted_eri_ppss_rys,
    contracted_eri_psss_rys,
    contracted_eri_psps_rys,
    contracted_eri_ssss_rys,
    primitive_eri_ppss_block_rys,
    primitive_eri_psss_block_rys,
    primitive_eri_psps_block_rys,
    primitive_eri_ssss_rys,
    rys_roots_weights,
)


def test_rys_single_root_weight_reproduces_boys_f0():
    for T in (0.0, 1.0e-10, 1.0e-4, 0.1, 1.0, 10.0, 50.0):
        roots, weights = rys_roots_weights(1, T)
        assert roots.shape == (1,)
        assert weights.shape == (1,)
        assert 0.0 <= roots[0] <= 1.0
        np.testing.assert_allclose(np.sum(weights), boys(0, T), atol=1e-14, rtol=1e-14)


def test_compiled_low_order_rys_rule_reproduces_required_moments():
    if _rys_cy is None:
        pytest.skip("compiled Rys kernels are unavailable")

    def stable_boys(order, T):
        return sum(
            (-T) ** k / (math.factorial(k) * (2 * order + 2 * k + 1))
            for k in range(80)
        ) if T < 1.0 else boys(order, T)

    for nroots in (1, 2, 3):
        for T in (0.0, 1.0e-10, 1.0e-4, 0.1, 1.0, 10.0, 100.0):
            roots, weights = _rys_cy.rys_roots_weights_low(nroots, T)
            nodes = roots / (1.0 + roots)
            assert np.all(nodes >= 0.0)
            assert np.all(nodes < 1.0)
            assert np.all(weights > 0.0)
            for order in range(2 * nroots):
                moment = np.dot(weights, nodes**order)
                np.testing.assert_allclose(moment, stable_boys(order, T), atol=2.0e-14, rtol=2.0e-12)


def test_native_cpp_low_root_fast_rules_reproduce_required_moments():
    native = basis_module._integrals_cpp
    if native is None or not hasattr(native, "compute_rys_roots_weights"):
        pytest.skip("native C++ integral kernels are unavailable")

    points = np.concatenate(
        (
            np.linspace(0.0, 40.0, 161),
            np.array([1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0]),
        )
    )
    for nroots in (2, 3):
        for T in points:
            roots, weights = native.compute_rys_roots_weights(nroots, float(T))
            roots = np.asarray(roots)
            weights = np.asarray(weights)
            nodes = roots / (1.0 + roots)
            assert np.all(nodes >= 0.0)
            assert np.all(nodes < 1.0)
            assert np.all(weights > 0.0)
            for order in range(2 * nroots):
                reference = boys(order, float(T))
                np.testing.assert_allclose(
                    np.dot(weights, nodes**order),
                    reference,
                    atol=2.0e-14,
                    rtol=6.0e-11,
                )


def test_compiled_rys_recurrence_covers_every_sp_shell_pattern():
    if _rys_cy is None:
        pytest.skip("compiled Rys kernels are unavailable")

    centers = np.asarray(
        ((0.1, -0.2, 0.3), (0.0, 0.2, 1.1), (0.3, -0.1, -0.2), (-0.4, 0.3, 0.2)),
        dtype=float,
    )
    primitive_exponents = (0.5, 0.3, 0.4, 0.2)
    p_shells = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for mask in range(16):
        shells = []
        origins = []
        exponents = []
        weights = []
        nprim = []
        ranges = []
        for center in range(4):
            start = len(shells)
            components = p_shells if mask & (1 << center) else ((0, 0, 0),)
            for component in components:
                shells.append(component)
                origins.append(centers[center])
                exponents.append((primitive_exponents[center],))
                weights.append((1.0,))
                nprim.append(1)
            ranges.append((start, len(shells)))

        args = [item for bounds in ranges for item in bounds]
        shells_array = np.ascontiguousarray(shells, dtype=np.int64)
        origins_array = np.ascontiguousarray(origins, dtype=float)
        exponents_array = np.ascontiguousarray(exponents, dtype=float)
        weights_array = np.ascontiguousarray(weights, dtype=float)
        nprim_array = np.ascontiguousarray(nprim, dtype=np.int64)
        block = np.asarray(
            _rys_cy.compute_cartesian_shell_quartet_block_rys(
                shells_array, origins_array, exponents_array, weights_array, nprim_array, *args,
            )
        )
        derivative_reference = np.asarray(
            _rys_cy.compute_cartesian_shell_quartet_block_rys_derivative_reference(
                shells_array, origins_array, exponents_array, weights_array, nprim_array, *args,
            )
        )
        reference = np.empty_like(block)
        for index in np.ndindex(block.shape):
            ao_indices = tuple(ranges[center][0] + index[center] for center in range(4))
            reference[index] = electron_repulsion(
                primitive_exponents[0], shells[ao_indices[0]], centers[0],
                primitive_exponents[1], shells[ao_indices[1]], centers[1],
                primitive_exponents[2], shells[ao_indices[2]], centers[2],
                primitive_exponents[3], shells[ao_indices[3]], centers[3],
            )
        np.testing.assert_allclose(block, derivative_reference, atol=2.0e-12, rtol=2.0e-11)
        np.testing.assert_allclose(block, reference, atol=2.0e-12, rtol=2.0e-11)


def test_native_rys_shell_blocks_cover_d_and_f_through_seven_roots():
    try:
        from pyqed.qchem import _integrals_cpp
    except ImportError:
        pytest.skip("native integral kernels are unavailable")
    if not hasattr(_integrals_cpp, "compute_shell_quartet_rys_l3"):
        pytest.skip("native d/f Rys shell-block validation helper is unavailable")

    def components(l):
        return [
            (lx, ly, l - lx - ly)
            for lx in range(l, -1, -1)
            for ly in range(l - lx, -1, -1)
        ]

    centers = np.asarray(
        ((0.1, -0.2, 0.3), (0.0, 0.2, 1.1), (0.3, -0.1, -0.2), (-0.4, 0.3, 0.2)),
        dtype=float,
    )
    primitive_exponents = (0.5, 0.3, 0.4, 0.2)
    for angular_momenta in ((3, 3, 2, 0), (3, 3, 3, 0), (3, 3, 3, 3)):
        shells = []
        origins = []
        exponents = []
        weights = []
        nprim = []
        ranges = []
        for center, l in enumerate(angular_momenta):
            start = len(shells)
            for angular in components(l):
                shells.append(angular)
                origins.append(centers[center])
                exponents.append((primitive_exponents[center],))
                weights.append((1.0,))
                nprim.append(1)
            ranges.append((start, len(shells)))
        args = [item for bounds in ranges for item in bounds]
        block = np.asarray(
            _integrals_cpp.compute_shell_quartet_rys_l3(
                np.ascontiguousarray(shells, dtype=np.int64),
                np.ascontiguousarray(origins, dtype=float),
                np.ascontiguousarray(exponents, dtype=float),
                np.ascontiguousarray(weights, dtype=float),
                np.ascontiguousarray(nprim, dtype=np.int64),
                *args,
            )
        )
        sample_indices = {
            (0, 0, 0, 0),
            tuple(size - 1 for size in block.shape),
            tuple(size // 2 for size in block.shape),
            tuple((3 * axis + 1) % size for axis, size in enumerate(block.shape)),
        }
        for index in sample_indices:
            ao_indices = tuple(ranges[center][0] + index[center] for center in range(4))
            reference = electron_repulsion(
                primitive_exponents[0], shells[ao_indices[0]], centers[0],
                primitive_exponents[1], shells[ao_indices[1]], centers[1],
                primitive_exponents[2], shells[ao_indices[2]], centers[2],
                primitive_exponents[3], shells[ao_indices[3]], centers[3],
            )
            np.testing.assert_allclose(block[index], reference, atol=3.0e-11, rtol=3.0e-11)


def test_primitive_ssss_rys_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.0, 0.0, 0.0)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    ref = electron_repulsion(a, (0, 0, 0), A, b, (0, 0, 0), B, c, (0, 0, 0), C, d, (0, 0, 0), D)
    val = primitive_eri_ssss_rys(a, A, b, B, c, C, d, D)
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_contracted_ssss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.0, 0.0],
        shell=(0, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_ssss_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_contracted_ssss_rys_matches_sto3g_h2_basis_functions():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    basis_dict = parse_gbs(_basis_path(mol.basis))
    basis = make_contractions(
        basis_dict,
        mol.atom_symbols(),
        np.asarray(mol.atom_coords(), dtype=float),
        coord_types="p",
    )

    assert len(basis) == 2
    assert all(tuple(fn.shell) == (0, 0, 0) for fn in basis)

    ref = ERI(basis[0], basis[0], basis[1], basis[1])
    val = contracted_eri_ssss_rys(basis[0], basis[0], basis[1], basis[1])
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_primitive_psss_block_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    block = primitive_eri_psss_block_rys(a, A, b, B, c, C, d, D)
    refs = np.asarray(
        [
            electron_repulsion(a, (1, 0, 0), A, b, (0, 0, 0), B, c, (0, 0, 0), C, d, (0, 0, 0), D),
            electron_repulsion(a, (0, 1, 0), A, b, (0, 0, 0), B, c, (0, 0, 0), C, d, (0, 0, 0), D),
            electron_repulsion(a, (0, 0, 1), A, b, (0, 0, 0), B, c, (0, 0, 0), C, d, (0, 0, 0), D),
        ]
    )
    np.testing.assert_allclose(block, refs, atol=1e-12, rtol=1e-12)


def test_primitive_ppss_block_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    shells = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    block = primitive_eri_ppss_block_rys(a, A, b, B, c, C, d, D)
    refs = np.zeros((3, 3), dtype=float)
    for i, sh_a in enumerate(shells):
        for j, sh_b in enumerate(shells):
            refs[i, j] = electron_repulsion(
                a, sh_a, A,
                b, sh_b, B,
                c, (0, 0, 0), C,
                d, (0, 0, 0), D,
            )
    np.testing.assert_allclose(block, refs, atol=1e-12, rtol=1e-12)


def test_contracted_psss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_psss_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_contracted_ppss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 1),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_ppss_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_contracted_ppss_rys_reuses_cache_under_p_center_permutation():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 1),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    _contracted_eri_ppss_rys_cached.cache_clear()

    ref = ERI(a, b, c, d)
    val1 = contracted_eri_ppss_rys(a, b, c, d)
    info1 = _contracted_eri_ppss_rys_cached.cache_info()
    val2 = contracted_eri_ppss_rys(b, a, c, d)
    info2 = _contracted_eri_ppss_rys_cached.cache_info()

    np.testing.assert_allclose(val1, ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(val2, ref, atol=1e-12, rtol=1e-12)
    assert info1.misses == 1
    assert info2.hits == info1.hits + 1


def test_generic_dsss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(2, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_cartesian_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-11, rtol=1e-11)


def test_generic_dpss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 1, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 1),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_cartesian_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-11, rtol=1e-11)


def test_generic_ddss_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(2, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 2, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 0),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_cartesian_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-11, rtol=1e-11)


def test_primitive_psps_block_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    shells = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    block = primitive_eri_psps_block_rys(a, A, b, B, c, C, d, D)
    refs = np.zeros((3, 3), dtype=float)
    for i, sh_a in enumerate(shells):
        for k, sh_c in enumerate(shells):
            refs[i, k] = electron_repulsion(
                a, sh_a, A,
                b, (0, 0, 0), B,
                c, sh_c, C,
                d, (0, 0, 0), D,
            )
    np.testing.assert_allclose(block, refs, atol=1e-12, rtol=1e-12)


def test_contracted_psps_rys_matches_existing_contracted_eri():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 0, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 1),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(0, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    ref = ERI(a, b, c, d)
    val = contracted_eri_psps_rys(a, b, c, d)
    np.testing.assert_allclose(val, ref, atol=1e-12, rtol=1e-12)


def test_contracted_pppp_rys_matches_existing_contracted_eri_and_reuses_cache_under_permutation():
    a = ContractedGaussian(
        origin=[0.0, 0.1, -0.1],
        shell=(1, 0, 0),
        exps=[0.6, 0.2],
        coefs=[0.7, 0.4],
    )
    b = ContractedGaussian(
        origin=[0.0, 0.0, 1.4],
        shell=(0, 1, 0),
        exps=[0.5, 0.15],
        coefs=[0.6, 0.3],
    )
    c = ContractedGaussian(
        origin=[0.2, 0.1, -0.2],
        shell=(0, 0, 1),
        exps=[0.7, 0.25],
        coefs=[0.5, 0.2],
    )
    d = ContractedGaussian(
        origin=[-0.3, 0.2, 0.4],
        shell=(1, 0, 0),
        exps=[0.8, 0.3],
        coefs=[0.4, 0.25],
    )

    _contracted_eri_pppp_rys_cached.cache_clear()

    ref = ERI(a, b, c, d)
    val1 = contracted_eri_pppp_rys(a, b, c, d)
    info1 = _contracted_eri_pppp_rys_cached.cache_info()
    val2 = contracted_eri_pppp_rys(c, d, a, b)
    info2 = _contracted_eri_pppp_rys_cached.cache_info()

    np.testing.assert_allclose(val1, ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(val2, ref, atol=1e-12, rtol=1e-12)
    assert info1.misses == 1
    assert info2.hits == info1.hits + 1
