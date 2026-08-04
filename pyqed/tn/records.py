"""Diagnostic records for tensor-network algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TTNSiteUpdate:
    """Diagnostics for one exact one-tensor update."""

    site: int
    raw_dim: int
    energy_before: float
    energy: float
    accepted: bool
    residual_norm: float
