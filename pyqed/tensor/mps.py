"""Finite tensor-train compatibility import.

The finite-chain implementation lives in :mod:`pyqed.mps`; this module no
longer owns an independent MPS class.
"""

from pyqed.mps.mps import MPS, MPO, apply_mpo

__all__ = ["MPS", "MPO", "apply_mpo"]
