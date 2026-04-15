import numpy as np

from pyqed.dvr import SDDVR, simultaneous_diagonalize


def _random_orthogonal(n, seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return q


def test_simultaneous_diagonalize_recovers_commuting_structure():
    u = _random_orthogonal(4, seed=3)
    d1 = np.diag([-1.0, -0.25, 0.5, 1.2])
    d2 = np.diag([0.1, 0.4, 0.9, 1.7])
    ops = np.stack((u.T @ d1 @ u, u.T @ d2 @ u))

    transform, diagonalized, _ = simultaneous_diagonalize(ops)

    for op, ref in zip(diagonalized, (d1, d2)):
        np.testing.assert_allclose(
            np.sort(np.diag(op)),
            np.sort(np.diag(ref)),
            atol=1e-8,
        )
        np.testing.assert_allclose(op - np.diag(np.diag(op)), 0.0, atol=1e-8)

    np.testing.assert_allclose(transform @ transform.T, np.eye(4), atol=1e-8)


def test_sddvr_local_operator_and_roundtrip():
    u = _random_orthogonal(3, seed=7)
    x = np.diag([-1.0, 0.0, 1.5])
    y = np.diag([0.2, 0.8, 1.1])
    sddvr = SDDVR(np.stack((u.T @ x @ u, u.T @ y @ u)), labels=['x', 'y'])

    v_dvr = sddvr.local_operator(lambda xv, yv: xv**2 + 2.0 * yv)
    v_fbr = sddvr.dvr2fbr(v_dvr)

    np.testing.assert_allclose(
        np.sort(sddvr.grid[:, 0]),
        np.sort(np.diag(x)),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.sort(sddvr.grid[:, 1]),
        np.sort(np.diag(y)),
        atol=1e-8,
    )
    np.testing.assert_allclose(sddvr.fbr2dvr(v_fbr), v_dvr, atol=1e-8)
    assert sddvr.diagonal_error() < 1e-8
