import numpy as np
from scipy.linalg import expm
from tensorly.decomposition import tensor_train_matrix

from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mps import MPS, MPO, expmpo


def _dense_to_mpo(matrix, nsites, phys_dim=2):
    tt = tensor_train_matrix(
        np.asarray(matrix, dtype=complex).reshape((phys_dim,) * nsites + (phys_dim,) * nsites),
        rank=matrix.shape[0],
    )
    return MPO([np.asarray(core).transpose(0, 3, 1, 2) for core in tt.factors])


def _mpo_to_dense(mpo):
    cores = [np.asarray(core).transpose(0, 2, 3, 1) for core in mpo.factors]
    tensor = cores[0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    tensor = np.squeeze(tensor, axis=(0, -1))
    nsites = len(cores)
    perm = list(range(0, 2 * nsites, 2)) + list(range(1, 2 * nsites, 2))
    tensor = np.transpose(tensor, axes=perm)
    dim = int(np.prod(mpo.dims))
    return tensor.reshape((dim, dim))


def _dense_to_mps(vector, nsites, phys_dim=2):
    factors = decompose(
        np.asarray(vector, dtype=complex).reshape((phys_dim,) * nsites),
        rank=[1] + [vector.size] * (nsites - 1) + [1],
    )
    return MPS(factors)


def _mps_to_dense(mps):
    return np.asarray(tt_to_tensor(mps.factors)).reshape(-1)


def test_dense_to_mpo_roundtrip_for_small_real_matrix():
    dense = np.array(
        [
            [0.10, 0.20, 0.00, 0.00],
            [0.00, -0.05, 0.12, 0.00],
            [0.00, 0.00, 0.03, 0.18],
            [0.00, 0.00, 0.00, -0.02],
        ],
        dtype=complex,
    )
    mpo = _dense_to_mpo(dense, nsites=2)
    np.testing.assert_allclose(_mpo_to_dense(mpo), dense, atol=1e-12)


def test_expmpo_matches_dense_expm_for_small_hermitian_mpo():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(4, 4))
    h_dense = 0.1 * (base + base.T)

    h_mpo = _dense_to_mpo(h_dense, nsites=2)
    u_mpo = expmpo(h_mpo, constant=-0.23, D=None, order=8, scale=2)

    np.testing.assert_allclose(_mpo_to_dense(h_mpo), h_dense, atol=1e-12)
    np.testing.assert_allclose(_mpo_to_dense(u_mpo), expm(-0.23 * h_dense), atol=1e-10, rtol=1e-10)


def test_expmpo_matches_dense_expm_for_small_nonhermitian_mpo():
    h_dense = np.array(
        [
            [0.10, 0.20, 0.00, 0.00],
            [0.00, -0.05, 0.12, 0.00],
            [0.00, 0.00, 0.03, 0.18],
            [0.00, 0.00, 0.00, -0.02],
        ],
        dtype=complex,
    )

    h_mpo = _dense_to_mpo(h_dense, nsites=2)
    u_mpo = expmpo(h_mpo, constant=0.31, D=None, order=8, scale=2)

    np.testing.assert_allclose(_mpo_to_dense(h_mpo), h_dense, atol=1e-12)
    np.testing.assert_allclose(_mpo_to_dense(u_mpo), expm(0.31 * h_dense), atol=1e-10, rtol=1e-10)


def test_expmpo_compressed_taylor_path_preserves_coefficients():
    rng = np.random.default_rng(17)
    base = rng.normal(size=(8, 8))
    h_dense = 0.05 * (base + base.T)
    h_mpo = _dense_to_mpo(h_dense, nsites=3)

    u_mpo = expmpo(
        h_mpo,
        constant=-0.03j,
        D=16,
        order=4,
        scale=2,
    )

    np.testing.assert_allclose(
        _mpo_to_dense(u_mpo),
        expm(-0.03j * h_dense),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_expmpo_matches_dense_expm_and_state_action_on_small_three_site_case():
    rng = np.random.default_rng(19)
    base = rng.normal(size=(8, 8))
    h_dense = 0.08 * (base + base.T)
    psi_dense = rng.normal(size=(8,)) + 1j * rng.normal(size=(8,))
    psi_dense /= np.linalg.norm(psi_dense)

    h_mpo = _dense_to_mpo(h_dense, nsites=3)
    psi_mps = _dense_to_mps(psi_dense, nsites=3)
    exact_u = expm(-0.41 * h_dense)

    low_u_mpo = expmpo(h_mpo, constant=-0.41, D=None, order=2, scale=0)
    high_u_mpo = expmpo(h_mpo, constant=-0.41, D=None, order=8, scale=2)

    low_u_dense = _mpo_to_dense(low_u_mpo)
    high_u_dense = _mpo_to_dense(high_u_mpo)
    np.testing.assert_allclose(low_u_dense, exact_u, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(high_u_dense, exact_u, atol=1e-10, rtol=1e-10)

    low_state = low_u_mpo @ psi_mps
    high_state = high_u_mpo @ psi_mps
    exact_state = exact_u @ psi_dense
    np.testing.assert_allclose(_mps_to_dense(low_state), exact_state, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(_mps_to_dense(high_state), exact_state, atol=1e-10, rtol=1e-10)
