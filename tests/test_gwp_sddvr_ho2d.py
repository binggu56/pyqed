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
    sd = fbr.to_sddvr()

    t_sd = sd.fbr2dvr(fbr.orthonormal_kinetic(mass=[1.0, 1.0]))
    qx, qy = fbr.orthonormal_coordinate_ops()
    v_sd = sd.fbr2dvr(0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy)))
    e, _ = np.linalg.eigh(t_sd + v_sd)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    np.testing.assert_allclose(e[:6], exact[:6], atol=1.2e-1)
