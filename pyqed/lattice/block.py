"""Shared mutable renormalized-block state for tensor-network solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """Current renormalized space and data produced while growing it.

    ``qn`` may hold an ``Leg`` or backend-specific sector labels.
    ``tensor`` is the local continuation map needed to grow the block again.
    ``factor`` is the possibly fused tensor emitted into a sequential ansatz.
    Solver-specific state belongs in ``data`` rather than on a physical Site.
    """

    h: Any = None
    qn: Any = None
    tensor: Any = None
    branch_qn: Any = None
    factor: Any = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.factor is None:
            self.factor = self.tensor


__all__ = ["Block"]
