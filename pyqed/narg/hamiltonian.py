"""Hamiltonian objects consumed by generic NARG dispatchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def normalize_basis(value: str | None) -> str:
    key = "site" if value is None else str(value).lower().replace("-", "_")
    aliases = {
        "real": "site",
        "real_space": "site",
        "position": "site",
        "k": "momentum",
        "k_space": "momentum",
        "fourier": "momentum",
    }
    return aliases.get(key, key)


def _normalize_symmetry_key(value: str | None) -> str:
    key = "number" if value is None else str(value).lower().replace("-", "_")
    aliases = {
        "bare": "none",
        "dense": "none",
        "no_sym": "none",
        "nosymmetry": "none",
        "nosym": "none",
        "no_symmetry": "none",
        "u1": "number",
        "u1xu1": "number",
        "u1u1": "number",
        "u1_u1": "number",
        "abelian": "number",
        "particle_number": "number",
        "su2": "spin",
        "nonabelian": "spin",
        "non_abelian": "spin",
        "k": "momentum",
        "crystal_momentum": "momentum",
    }
    return aliases.get(key, key)


def normalize_symmetry(value: str | None) -> str:
    """Normalize public symmetry labels used by NARG frontends."""
    if value is None:
        return "none"
    return _normalize_symmetry_key(value)


def normalize_form(value: str | None) -> str:
    key = "integrals" if value is None else str(value).lower().replace("-", "_")
    aliases = {
        "qchem": "integrals",
        "integral": "integrals",
        "eri": "integrals",
    }
    return aliases.get(key, key)


def normalize_orbital_blocks(blocks, *, norb: int | None = None):
    """Return orbital blocks as a tuple of integer tuples."""
    if blocks is None:
        return None
    normalized = tuple(tuple(int(i) for i in block) for block in blocks)
    if not normalized:
        raise ValueError("orbital_blocks must contain at least one block.")
    if any(len(block) < 1 for block in normalized):
        raise ValueError("orbital_blocks cannot contain empty blocks.")
    flat = tuple(i for block in normalized for i in block)
    if len(set(flat)) != len(flat):
        raise ValueError("orbital_blocks cannot contain duplicate orbital indices.")
    if any(i < 0 for i in flat):
        raise ValueError("orbital_blocks cannot contain negative orbital indices.")
    if norb is not None and sorted(flat) != list(range(int(norb))):
        raise ValueError("orbital_blocks must partition all orbital indices exactly once.")
    return normalized


@dataclass(frozen=True)
class HamiltonianSpec:
    """Common metadata for NARG Hamiltonian inputs."""

    basis: str = "site"
    symmetry: str = "number"
    form: str = "integrals"
    orbital_blocks: tuple[tuple[int, ...], ...] | None = None
    target: Any = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class IntegralHamiltonian(HamiltonianSpec):
    """Spatial-orbital one-/two-electron Hamiltonian for qchem NARG."""

    h1e: np.ndarray | None = None
    eri: np.ndarray | None = None
    mol: object | None = None
    mf: object | None = None
    model: object | None = None

    def __post_init__(self):
        object.__setattr__(self, "basis", normalize_basis(self.basis))
        object.__setattr__(self, "symmetry", normalize_symmetry(self.symmetry))
        object.__setattr__(self, "form", "integrals")
        object.__setattr__(self, "h1e", np.asarray(self.h1e, dtype=float))
        object.__setattr__(self, "eri", np.asarray(self.eri, dtype=float))
        object.__setattr__(
            self,
            "orbital_blocks",
            normalize_orbital_blocks(self.orbital_blocks, norb=self.h1e.shape[0]),
        )


@dataclass(frozen=True)
class MPOHamiltonian(HamiltonianSpec):
    """MPO or model-native complementary-environment Hamiltonian."""

    tensors: tuple[Any, ...] = ()
    sites: tuple[Any, ...] = ()
    fermionic: bool = True
    model: object | None = None

    def __post_init__(self):
        object.__setattr__(self, "basis", normalize_basis(self.basis))
        object.__setattr__(self, "symmetry", normalize_symmetry(self.symmetry))
        object.__setattr__(self, "form", "mpo")
        object.__setattr__(self, "orbital_blocks", normalize_orbital_blocks(self.orbital_blocks))
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "sites", tuple(self.sites))


__all__ = [
    "HamiltonianSpec",
    "IntegralHamiltonian",
    "MPOHamiltonian",
    "normalize_basis",
    "normalize_form",
    "normalize_orbital_blocks",
    "normalize_symmetry",
]
