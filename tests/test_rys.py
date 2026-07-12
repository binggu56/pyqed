import math

import numpy as np

from pyqed.qchem import Molecule
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
    contracted_eri_ppps_rys,
    contracted_eri_ppss_rys,
    contracted_eri_psss_rys,
    contracted_eri_psps_rys,
    contracted_eri_ssss_rys,
    primitive_eri_pppp_block_rys,
    primitive_eri_pppp_rys,
    primitive_eri_ppps_block_rys,
    primitive_eri_ppps_rys,
    primitive_eri_ppss_block_rys,
    primitive_eri_ppss_rys,
    primitive_eri_psss_block_rys,
    primitive_eri_psss_rys,
    primitive_eri_psps_block_rys,
    primitive_eri_psps_rys,
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


def test_primitive_scalar_rys_wrappers_match_block_entries():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    np.testing.assert_allclose(
        primitive_eri_psss_rys((1, 0, 0), a, A, b, B, c, C, d, D),
        primitive_eri_psss_block_rys(a, A, b, B, c, C, d, D)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        primitive_eri_ppss_rys((1, 0, 0), a, A, (0, 0, 1), b, B, c, C, d, D),
        primitive_eri_ppss_block_rys(a, A, b, B, c, C, d, D)[0, 2],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        primitive_eri_psps_rys((1, 0, 0), a, A, b, B, (0, 1, 0), c, C, d, D),
        primitive_eri_psps_block_rys(a, A, b, B, c, C, d, D)[0, 1],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        primitive_eri_ppps_rys((1, 0, 0), a, A, (0, 0, 1), b, B, (0, 1, 0), c, C, d, D),
        primitive_eri_ppps_block_rys(a, A, b, B, c, C, d, D)[0, 2, 1],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        primitive_eri_pppp_rys((1, 0, 0), a, A, (0, 0, 1), b, B, (0, 1, 0), c, C, (1, 0, 0), d, D),
        primitive_eri_pppp_block_rys(a, A, b, B, c, C, d, D)[0, 2, 1, 0],
        atol=1e-12,
        rtol=1e-12,
    )


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


def test_primitive_ppps_block_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    shells = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    block = primitive_eri_ppps_block_rys(a, A, b, B, c, C, d, D)
    refs = np.zeros((3, 3, 3), dtype=float)
    for i, sh_a in enumerate(shells):
        for j, sh_b in enumerate(shells):
            for k, sh_c in enumerate(shells):
                refs[i, j, k] = electron_repulsion(
                    a, sh_a, A,
                    b, sh_b, B,
                    c, sh_c, C,
                    d, (0, 0, 0), D,
                )
    np.testing.assert_allclose(block, refs, atol=1e-12, rtol=1e-12)


def test_primitive_pppp_block_matches_existing_primitive_eri():
    a = 0.5
    b = 0.3
    c = 0.4
    d = 0.2
    A = (0.1, -0.2, 0.3)
    B = (0.0, 0.0, 1.1)
    C = (0.2, -0.1, 0.3)
    D = (0.4, 0.3, -0.2)

    shells = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    block = primitive_eri_pppp_block_rys(a, A, b, B, c, C, d, D)
    refs = np.zeros((3, 3, 3, 3), dtype=float)
    for i, sh_a in enumerate(shells):
        for j, sh_b in enumerate(shells):
            for k, sh_c in enumerate(shells):
                for l, sh_d in enumerate(shells):
                    refs[i, j, k, l] = electron_repulsion(
                        a, sh_a, A,
                        b, sh_b, B,
                        c, sh_c, C,
                        d, sh_d, D,
                    )
    np.testing.assert_allclose(block, refs, atol=1e-12, rtol=1e-12)


def test_contracted_ppps_rys_matches_existing_contracted_eri():
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
        shell=(0, 1, 0),
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
    val = contracted_eri_ppps_rys(a, b, c, d)
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
