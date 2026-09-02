"""Nuclear coordinate charts for locally diabatic dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pyqed.dvr import DVR


@dataclass(frozen=True)
class Coord:
    """A grid-independent nuclear-coordinate chart.

    ``to_cartesian(q)`` maps one nuclear coordinate vector to an ``(N, 3)``
    Cartesian geometry. ``bounds`` defines the fitting domain and
    ``periodic_axes`` identifies coordinates whose endpoints are equivalent.
    Electronic sampling and nuclear dynamics discretize this chart independently.
    """

    to_cartesian: Callable | None = None
    bounds: tuple | None = None
    periodic_axes: tuple = ()

    def __post_init__(self):
        if self.to_cartesian is not None and not callable(self.to_cartesian):
            raise TypeError("to_cartesian must be callable")
        bounds = self.bounds
        if bounds is None:
            raise ValueError("coordinate bounds are required")
        bounds = tuple(tuple(float(value) for value in interval) for interval in bounds)
        if not bounds or any(len(interval) != 2 for interval in bounds):
            raise ValueError("bounds must contain one (lower, upper) pair per coordinate")
        if any(
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or upper <= lower
            for lower, upper in bounds
        ):
            raise ValueError("coordinate bounds must be finite and increasing")
        object.__setattr__(self, "bounds", bounds)
        periodic_axes = tuple(
            sorted(set(int(axis) for axis in self.periodic_axes))
        )
        if any(axis < 0 or axis >= len(bounds) for axis in periodic_axes):
            raise ValueError("periodic_axes contains an invalid coordinate")
        object.__setattr__(self, "periodic_axes", periodic_axes)

    def validate_grid(self, grid):
        """Validate a dynamics grid against this chart and return it."""
        if not isinstance(grid, DVR):
            raise TypeError("grid must be a pyqed.dvr.DVR product grid")
        if grid.ndim != self.ndim:
            raise ValueError("grid dimension does not match the coordinate chart")
        for axis, (lower, upper) in zip(grid.x, self.bounds):
            values = np.asarray(axis, dtype=float)
            if np.min(values) < lower or np.max(values) > upper:
                raise ValueError("dynamics grid lies outside the coordinate bounds")
        return grid

    @property
    def ndim(self):
        return len(self.bounds)

    def cartesian(self, q):
        """Evaluate the Cartesian embedding at one coordinate vector."""
        if self.to_cartesian is None:
            raise RuntimeError("this coordinate chart has no Cartesian embedding")
        q = np.asarray(q) if isinstance(q, (list, tuple)) else q
        value = self.to_cartesian(q)
        if isinstance(q, np.ndarray):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                return value
            if array.dtype != object:
                return array
        return value

    __call__ = cartesian


__all__ = ["Coord"]
