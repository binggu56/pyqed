"""Compatibility shim for the legacy LDR module path.

The canonical implementation now lives in :mod:`pyqed.ldr.core`.
Importing :mod:`pyqed.ldr.solver` continues to work for existing code.
"""

from __future__ import annotations

from .core import LDR

__all__ = ["LDR"]

