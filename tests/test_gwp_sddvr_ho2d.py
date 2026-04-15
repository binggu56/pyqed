import numpy as np
from itertools import product

from pyqed.dvr import GaussianWavepacketFBR


def test_gwp_sddvr_2d_independent_harmonic_oscillator():
    wx, wy = 1.0, 1.5
    centers_1d = np.linspace(-2.5, 2.5, 5)
    centers = np.array(list(product(centers_1d, centers_1d)))

    fbr = GaussianWavepacketFBR(
        centers=centers,
        widths=[wx, wy],
        labels=['x', 'y'],
    )
    h_sd, _ = fbr.harmonic_hamiltonian_sddvr(
        omega=[wx, wy],
        mass=[1.0, 1.0],
    )
    e, _ = np.linalg.eigh(h_sd)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    np.testing.assert_allclose(e[:6], exact[:6], atol=2.9e-1)


def test_gwp_sddvr_projected_quadratic_is_more_accurate_than_default_diagonal():
    wx, wy = 1.0, 1.5
    centers_1d = np.linspace(-2.5, 2.5, 5)
    centers = np.array(list(product(centers_1d, centers_1d)))

    fbr = GaussianWavepacketFBR(
        centers=centers,
        widths=[wx, wy],
        labels=['x', 'y'],
    )
    h_diag, _ = fbr.harmonic_hamiltonian_sddvr(
        omega=[wx, wy],
        mass=[1.0, 1.0],
        approximation='diagonal',
    )
    h_proj, _ = fbr.harmonic_hamiltonian_sddvr(
        omega=[wx, wy],
        mass=[1.0, 1.0],
        approximation='projected',
    )
    e_diag = np.linalg.eigvalsh(h_diag)
    e_proj = np.linalg.eigvalsh(h_proj)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    err_diag = np.max(np.abs(e_diag[:6] - exact[:6]))
    err_proj = np.max(np.abs(e_proj[:6] - exact[:6]))
    assert err_proj < err_diag
