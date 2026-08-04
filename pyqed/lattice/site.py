"""Site metadata containers for LETTA and related tensor-network code paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pyqed import SpinHalfFermionOperators, pauli


@dataclass(frozen=True)
class PhysicalLeg:
    """Metadata for one physical index in a tensor-network ansatz."""

    dim: int
    labels: tuple[str, ...]
    operators: Mapping[str, Any] = field(default_factory=dict)
    fermionic: bool = False
    local_charges: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    jw_metadata: Mapping[str, Any] = field(default_factory=dict)


class Site:
    """A light-weight physical site descriptor with local operators.

    The object is intentionally small and numeric-operator focused:
    - a physical leg (``physical_leg``),
    - basis labels,
    - named operators, and
    - fermionic/JW metadata.
    """

    def __init__(
        self,
        *,
        physical_leg: PhysicalLeg,
        operators: Mapping[str, Any] | None = None,
        labels: Sequence[str] | None = None,
    ):
        if not isinstance(physical_leg, PhysicalLeg):
            raise TypeError("physical_leg must be a PhysicalLeg.")
        self.physical_leg = physical_leg
        self.operators = (
            dict(physical_leg.operators)
            if operators is None
            else dict(operators)
        )
        if labels is None:
            labels = physical_leg.labels
        self.basis_labels = tuple(str(label) for label in labels)
        self.dim = self.d = int(physical_leg.dim)
        if len(self.basis_labels) != self.dim:
            raise ValueError("number of basis labels must match site dimension.")
        # Mirror historical attribute name used in older scripts:
        self.dimensions = (self.dim,)

    @classmethod
    def spinful_fermion(cls, *, include_jw: bool = True) -> "Site":
        """Build a spinful-fermion site (empty/up/down/full)."""
        operators = SpinHalfFermionOperators()
        jw_metadata: dict[str, Any]
        if include_jw:
            jw_metadata = {
                "JW_operator_names": ("JW", "JWu", "JWd"),
                "ordered_states": ("empty", "up", "down", "full"),
                "parity_phase_name": "JW",
            }
        else:
            jw_metadata = {}
        leg = PhysicalLeg(
            dim=4,
            labels=("empty", "up", "down", "full"),
            operators=operators,
            fermionic=True,
            local_charges=((0, 0), (1, 1), (1, -1), (2, 0)),
            jw_metadata=jw_metadata,
        )
        return cls(physical_leg=leg)

    @classmethod
    def spin_half(cls, *, d: int = 2) -> "Site":
        """Spin-1/2 local basis with Pauli operators."""
        if int(d) != 2:
            raise ValueError("spin_half currently supports dim=2 only.")
        I, X, Y, Z = pauli()
        leg = PhysicalLeg(
            dim=int(d),
            labels=("up", "down"),
            operators={"I": I, "X": X, "Y": Y, "Z": Z, "Sz": Z / 2, "Sx": X / 2, "Sy": Y / 2},
            local_charges=((0,), (1,)),
            fermionic=False,
        )
        return cls(physical_leg=leg)

    @classmethod
    def spinless_fermion(cls) -> "Site":
        """Spinless-fermion placeholder with a two-state local Hilbert space."""
        leg = PhysicalLeg(
            dim=2,
            labels=("empty", "occupied"),
            local_charges=((0,), (1,)),
            fermionic=True,
        )
        return cls(physical_leg=leg)

    def add_operator(self, operator_name):
        """Add an operator initialized as an all-zero ``(dim, dim)`` matrix."""
        operator_name = str(operator_name)
        if operator_name in self.operators:
            raise ValueError("operator_name already exists.")
        self.operators[operator_name] = np.zeros((self.dim, self.dim))

    @property
    def local_charges(self):
        return tuple(
            tuple(int(component) for component in charge)
            for charge in self.physical_leg.local_charges
        )

    @property
    def jw_metadata(self):
        return dict(self.physical_leg.jw_metadata)

    @property
    def is_fermionic(self) -> bool:
        return bool(self.physical_leg.fermionic)


class SpinHalfFermionSite(Site):
    """Backward-compatible class name."""

    def __init__(self):
        super().__init__(physical_leg=Site.spinful_fermion().physical_leg)


class SpinHalfSite(Site):
    """Backward-compatible spin-1/2 site."""

    def __init__(self, d=2):
        super().__init__(physical_leg=Site.spin_half(d=int(d)).physical_leg)


class SpinlessFermionSite(Site):
    """Backward-compatible spinless fermion site."""

    def __init__(self):
        super().__init__(physical_leg=Site.spinless_fermion().physical_leg)


__all__ = [
    "Site",
    "PhysicalLeg",
    "SpinHalfFermionSite",
    "SpinHalfSite",
    "SpinlessFermionSite",
]
