"""Small legacy site objects retained for the older DMRG model builders."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class DMRGException(Exception):
    """Base exception raised by the legacy site/block helpers."""


class Site:
    """Single-site Hilbert space with a mutable operator registry."""

    def __init__(self, dim):
        dim = int(dim)
        if dim < 1:
            raise DMRGException("Site dimension must be at least 1.")
        self.dim = dim
        self.operators = {"id": sparse.eye(dim, dim)}

    def add_operator(self, operator_name):
        """Add a zero-filled dense operator under ``operator_name``."""
        name = str(operator_name)
        if name in self.operators:
            raise DMRGException(f"Operator {name!r} already exists.")
        self.operators[name] = np.zeros((self.dim, self.dim))


class Block(Site):
    """Legacy truncated block; behavior is inherited from :class:`Site`."""


class PauliSite(Site):
    """Two-level site containing ``s_z``, ``s_x``, and lowering operators."""

    def __init__(self):
        super().__init__(2)
        for name in ("s_z", "s_x", "s_m"):
            self.add_operator(name)
        self.operators["s_z"][0, 0] = -1.0
        self.operators["s_z"][1, 1] = 1.0
        self.operators["s_x"][0, 1] = 1.0
        self.operators["s_x"][1, 0] = 1.0
        self.operators["s_m"][0, 1] = 1.0


__all__ = ["Block", "DMRGException", "PauliSite", "Site"]
