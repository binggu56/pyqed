import numpy as np

from pyqed.qchem.gdvr import two_electron_wedge_basis, two_electron_wedge_kinetic


def _full_kinetic(K):
    return np.kron(K, np.eye(K.shape[0])) + np.kron(np.eye(K.shape[0]), K)


def _folded_reference(K, pairs, phase):
    npoints = K.shape[0]
    full = _full_kinetic(K)
    wedge_index = pairs[:, 0] * npoints + pairs[:, 1]
    mirror_index = pairs[:, 1] * npoints + pairs[:, 0]
    return full[np.ix_(wedge_index, wedge_index)] + phase * full[np.ix_(wedge_index, mirror_index)]


def test_two_electron_wedge_basis_is_raw_ordered_grid():
    pairs, phase = two_electron_wedge_basis(5, exchange="singlet")
    assert phase == 1
    assert pairs.shape[0] == 5 * 4 // 2
    assert all(i < j for i, j in pairs)


def test_two_electron_wedge_kinetic_matches_folded_singlet_operator():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(5, 5))
    K = 0.5 * (raw + raw.T)
    Tw, pairs = two_electron_wedge_kinetic(K, exchange="singlet")
    expected = _folded_reference(K, pairs, phase=1)
    np.testing.assert_allclose(Tw, expected, atol=1e-12)
    np.testing.assert_allclose(Tw, Tw.T, atol=1e-12)


def test_two_electron_wedge_kinetic_matches_folded_triplet_operator():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(6, 6))
    K = 0.5 * (raw + raw.T)
    Tw, pairs = two_electron_wedge_kinetic(K, exchange="triplet")
    expected = _folded_reference(K, pairs, phase=-1)
    np.testing.assert_allclose(Tw, expected, atol=1e-12)
    np.testing.assert_allclose(Tw, Tw.T, atol=1e-12)


def test_two_electron_wedge_extension_reproduces_full_action_on_wedge():
    rng = np.random.default_rng(14)
    raw = rng.normal(size=(4, 4))
    K = 0.5 * (raw + raw.T)
    Tw, pairs, extension = two_electron_wedge_kinetic(K, exchange="singlet", return_extension=True)
    psi = rng.normal(size=pairs.shape[0])
    full = _full_kinetic(K)
    wedge_index = pairs[:, 0] * K.shape[0] + pairs[:, 1]
    np.testing.assert_allclose(Tw @ psi, (full @ (extension @ psi))[wedge_index], atol=1e-12)
