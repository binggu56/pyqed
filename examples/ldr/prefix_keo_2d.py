#!/usr/bin/env python3
"""Validate batched 2D prefix FFTs against the full linked LDR matrix."""

from __future__ import annotations

import json

import numpy as np

from pyqed.dvr import ExponentialDVR
from pyqed.ldr import kinetic


def links_on_grid(shape):
    links = {}
    for axis, axis_size in enumerate(shape):
        for point in np.ndindex(*shape):
            if point[axis] == axis_size - 1:
                continue
            angle = 0.025 * (axis + 1) * (1 + sum(point))
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=complex,
            )
            contraction = np.diag(
                [0.985 - 0.002 * point[axis], 0.96 + 0.003 * point[1 - axis]]
            )
            links[(axis, point)] = rotation @ contraction
    return links


def main():
    shape = (7, 8)
    axes = (
        ExponentialDVR(npts=shape[0], L=7.0, mass=2.0),
        ExponentialDVR(npts=shape[1], L=9.0, mass=3.0),
    )
    descriptors = tuple(axis.kinetic_toeplitz() for axis in axes)
    links = links_on_grid(shape)
    fft = kinetic.PrefixFFTND(descriptors, shape, links)

    matrices = tuple(axis.t() for axis in axes)
    nuclear = np.kron(matrices[0], np.eye(shape[1]))
    nuclear += np.kron(np.eye(shape[0]), matrices[1])
    reference = kinetic.matrix(
        nuclear, shape, 2, links=links, symmetrize=False
    )
    vector = np.random.default_rng(21).normal(size=reference.shape[0])
    scale = max(float(np.linalg.norm(reference @ vector)), np.finfo(float).tiny)
    info = {
        **fft.info,
        "shape": shape,
        "dimension": reference.shape[0],
        "relative_action_error": float(
            np.linalg.norm(fft.matvec(vector) - reference @ vector) / scale
        ),
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
