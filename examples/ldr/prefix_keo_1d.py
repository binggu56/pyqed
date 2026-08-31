#!/usr/bin/env python3
"""Validate prefix-factorized MPOs for a dense periodic one-dimensional KEO."""

from __future__ import annotations

import json

import numpy as np

from pyqed.dvr import ExponentialDVR
from pyqed.ldr import kinetic


def nonunitary_links(ngrid):
    links = {}
    for index in range(ngrid - 1):
        angle = 0.08 * np.sin(2.0 * np.pi * (index + 0.5) / ngrid)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=complex,
        )
        contraction = np.diag(
            [
                0.985 - 0.005 * np.cos(2.0 * np.pi * index / ngrid),
                0.955 + 0.008 * np.sin(2.0 * np.pi * index / ngrid),
            ]
        )
        links[(0, (index,))] = rotation @ contraction
    return links


def main():
    ngrid = 15
    dvr = ExponentialDVR(npts=ngrid, L=12.0, mass=3.0)
    descriptor = dvr.kinetic_toeplitz()
    kinetic_matrix = dvr.t()
    links = nonunitary_links(ngrid)
    components, info = kinetic.prefix_mpos(kinetic_matrix, links)
    fitted = sum(component.to_dense() for component in components)
    reference = kinetic.matrix(
        kinetic_matrix,
        (ngrid,),
        2,
        links=links,
        symmetrize=False,
    )
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    info["relative_matrix_error"] = float(np.linalg.norm(fitted - reference) / scale)
    vector = np.random.default_rng(4).normal(size=ngrid * 2)
    info["relative_action_error"] = float(
        np.linalg.norm((fitted - reference) @ vector)
        / max(float(np.linalg.norm(reference @ vector)), np.finfo(float).tiny)
    )
    info["hermiticity_error"] = float(
        np.linalg.norm(fitted - fitted.conj().T) / scale
    )
    fft = kinetic.PrefixFFT(descriptor, links)
    info["fft_uses_descriptor"] = fft.info["descriptor"]
    info["fft_action_error"] = float(
        np.linalg.norm(fft.matvec(vector) - reference @ vector)
        / max(float(np.linalg.norm(reference @ vector)), np.finfo(float).tiny)
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
