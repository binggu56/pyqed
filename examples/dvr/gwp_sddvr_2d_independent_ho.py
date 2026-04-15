#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D independent harmonic oscillators with Gaussian-FBR SD-DVR.
"""

from itertools import product

import numpy as np

from pyqed.dvr import GaussianWavepacketFBR


def exact_energies(wx, wy, nmax=4):
    vals = []
    for nx in range(nmax):
        for ny in range(nmax):
            vals.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    return np.sort(np.array(vals))


def main():
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
    # For a quadratic oscillator potential we can project V exactly in the
    # orthonormal Gaussian basis before moving to the SD-DVR basis.
    v_sd = sd.fbr2dvr(0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy)))
    h_sd = t_sd + v_sd

    e, _ = np.linalg.eigh(h_sd)
    e_exact = exact_energies(wx, wy, nmax=4)

    print("First six SD-DVR energies:")
    print(e[:6])
    print("\nFirst six exact energies:")
    print(e_exact[:6])
    print("\nEnergy errors:")
    print(e[:6] - e_exact[:6])
    print("\nSD-DVR coordinate diagonalization error:")
    print(sd.diagonal_error())


if __name__ == '__main__':
    main()
