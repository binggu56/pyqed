import numpy as np

from pyqed.qchem.gdvr.rhf import (
    GDVRFockBuilder,
    _fock_2e_slice_collocated_reference,
    fock_2e_slice_collocated,
)


def _hermitian_density(size, seed=1):
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return mat + mat.conj().T


def _swap_matrix_indices(m):
    return np.arange(m * m).reshape(m, m).T.reshape(-1)


def test_gdvr_fock_builder_m1_fast_path_matches_reference():
    rng = np.random.default_rng(2)
    nz = 5
    m = 1
    eri_j = [[float(rng.random()) for _ in range(nz)] for _ in range(nz)]
    eri_k = [[float(rng.random()) for _ in range(nz)] for _ in range(nz)]
    p = _hermitian_density(nz, seed=3)

    builder = GDVRFockBuilder(eri_j, eri_k, nz, m)
    packed = fock_2e_slice_collocated(p, builder, None, nz, m, k_scale=0.7)
    reference = _fock_2e_slice_collocated_reference(p, eri_j, eri_k, nz, m, k_scale=0.7)

    np.testing.assert_allclose(packed, reference, atol=1e-12)


def test_gdvr_fock_builder_m_blocks_match_reference_without_mirror_symmetry():
    rng = np.random.default_rng(4)
    nz = 4
    m = 2
    mm = m * m
    eri_j = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    eri_k = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    for i in range(nz):
        for j in range(nz):
            a = rng.normal(size=(mm, mm))
            b = rng.normal(size=(mm, mm))
            eri_j[i][j] = a @ a.T
            eri_k[i][j] = 0.5 * (b + b.T)
    p = _hermitian_density(nz * m, seed=5)

    builder = GDVRFockBuilder(eri_j, eri_k, nz, m)
    assert not builder.exchange_mirror_ok
    packed = fock_2e_slice_collocated(p, builder, None, nz, m)
    reference = _fock_2e_slice_collocated_reference(p, eri_j, eri_k, nz, m)

    np.testing.assert_allclose(packed, reference, atol=1e-12)


def test_gdvr_fock_builder_uses_hermitian_half_exchange_when_valid():
    rng = np.random.default_rng(6)
    nz = 4
    m = 2
    mm = m * m
    swap = _swap_matrix_indices(m)
    eri_j = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    eri_k = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    for i in range(nz):
        for j in range(i, nz):
            a = rng.normal(size=(mm, mm))
            b = rng.normal(size=(mm, mm))
            eri_j[i][j] = a @ a.T
            eri_j[j][i] = eri_j[i][j].T
            block = 0.5 * (b + b.T)
            eri_k[i][j] = block
            eri_k[j][i] = block[np.ix_(swap, swap)]
    p = _hermitian_density(nz * m, seed=7)

    builder = GDVRFockBuilder(eri_j, eri_k, nz, m)
    assert builder.exchange_mirror_ok
    packed = fock_2e_slice_collocated(p, builder, None, nz, m)
    reference = _fock_2e_slice_collocated_reference(p, eri_j, eri_k, nz, m)

    np.testing.assert_allclose(packed, reference, atol=1e-12)


def test_gdvr_fock_builder_low_rank_blocks_match_reference():
    rng = np.random.default_rng(8)
    nz = 5
    m = 3
    mm = m * m
    eri_j = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    eri_k = [[np.zeros((mm, mm)) for _ in range(nz)] for _ in range(nz)]
    for i in range(nz):
        for j in range(nz):
            u = rng.normal(size=(mm, 1))
            v = rng.normal(size=(mm, 1))
            eri_j[i][j] = u @ u.T
            eri_k[i][j] = v @ v.T
    p = _hermitian_density(nz * m, seed=9)

    builder = GDVRFockBuilder(eri_j, eri_k, nz, m, low_rank_tol=1e-13)
    assert builder.uses_low_rank
    packed = fock_2e_slice_collocated(p, builder, None, nz, m)
    reference = _fock_2e_slice_collocated_reference(p, eri_j, eri_k, nz, m)

    np.testing.assert_allclose(packed, reference, atol=1e-11)
