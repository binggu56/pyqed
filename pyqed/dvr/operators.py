"""Grid-based DVR operator helpers."""

from __future__ import annotations

import warnings

import numpy as np

from .dvr_1d import SineDVR


def kinetic(grid, mass=1.0, dvr="sinc"):
    """Return a one-dimensional kinetic matrix on a uniform DVR grid."""

    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points")
    spacing = np.diff(grid)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("grid must be uniform")
    mass = float(mass)
    if mass <= 0.0:
        raise ValueError("mass must be positive")

    kind = str(dvr).lower().replace("_", "-")
    if kind == "sine":
        dx = float(spacing[0])
        return SineDVR(
            grid[0] - dx,
            grid[-1] + dx,
            grid.size,
            mass=mass,
        ).t()

    n = np.arange(grid.size)
    left = n[:, None]
    right = n[None, :]
    dx = float(spacing[0])
    if kind == "sinc":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = 2.0 * (-1.0) ** (left - right) / (left - right) ** 2 / dx**2
        result[n, n] = np.pi**2 / (3.0 * dx**2)
        return result / (2.0 * mass)

    if kind in {"periodic", "sinc-periodic", "sincperiodic"}:
        angle = np.pi * (left - right) / grid.size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if grid.size % 2 == 0:
                result = 2.0 * (-1.0) ** (left - right) / np.sin(angle) ** 2
                result[n, n] = (grid.size**2 + 2.0) / 3.0
            else:
                result = (
                    2.0
                    * (-1.0) ** (left - right)
                    * np.cos(angle)
                    / np.sin(angle) ** 2
                )
                result[n, n] = (grid.size**2 - 1.0) / 3.0
        length = dx * grid.size
        return result * (np.pi / length) ** 2 / (2.0 * mass)

    raise ValueError("dvr must be 'sinc', 'sine', or 'periodic'")
