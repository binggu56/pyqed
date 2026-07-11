#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:52:06 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""

from pyqed.dvr import GaussianWavepacketFBR
import numpy as np
import ultraplot as plt

fbr = GaussianWavepacketFBR.random_ho(
    nbasis=20,
    omega=[1.0, 1.5],
    mass=1.0,
    seed=11,
    overlap_cutoff=0.8,
    center_scale=1.2,
    labels=['x', 'y'],
)

sd = fbr.to_sddvr()

centers = np.asarray(fbr.centers)
grid = np.asarray(sd.grid)

ndim = centers.shape[1]
fig, axs = plt.subplots(nrows=2, ncols=ndim, figsize=(4 * ndim, 6), sharex=2)
if ndim == 1:
    axs = np.array(axs).reshape(2, 1)

for i in range(ndim):
    c = centers[:, i]
    g = grid[:, i]

    axs[0, i].scatter(range(len(c)), c, s=20)
    axs[0, i].format(title=f'Gaussian centers dim {i}', ylabel='value')

    axs[1, i].scatter(range(len(g)), g, s=20)
    axs[1, i].format(
        title=f'SD-DVR nodes dim {i}',
        xlabel='node index',
        ylabel='value',
    )

plt.show()