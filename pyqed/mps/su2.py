#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SU(2) irreps and particle-number/spin/spatial sectors for reduced MPS DMRG.

Spatial-orbital sites decompose into empty and doubly occupied spin singlets
and a singly occupied spin doublet. Real Abelian point-group labels may be
attached to these sectors; multidimensional spatial irreps use the general
product-group fusion machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class SU2Irrep:
    """
    SU(2) irrep labelled by ``two_j = 2j``.

    Using ``2j`` keeps all labels integer-valued, which is convenient for
    tensor-sector bookkeeping.
    """

    two_j: int
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if int(self.two_j) != self.two_j or self.two_j < 0:
            raise ValueError(f"SU(2) irrep requires non-negative integer two_j, got {self.two_j!r}.")
        object.__setattr__(self, "_hash", 1000003 + int(self.two_j))

    @property
    def j(self) -> float:
        return 0.5 * self.two_j

    @property
    def dim(self) -> int:
        return self.two_j + 1

    def __str__(self) -> str:
        if self.two_j % 2 == 0:
            return f"S={self.two_j // 2}"
        return f"S={self.two_j}/2"

    def __hash__(self) -> int:
        return self._hash


def fuse_irreps(left: SU2Irrep, right: SU2Irrep) -> tuple[SU2Irrep, ...]:
    """
    Clebsch-Gordan fusion rule for two SU(2) irreps.

    In ``2j`` notation this is the range

        ``|two_j1 - two_j2|, |two_j1 - two_j2| + 2, ..., two_j1 + two_j2``.
    """
    lo = abs(left.two_j - right.two_j)
    hi = left.two_j + right.two_j
    return tuple(SU2Irrep(two_j) for two_j in range(lo, hi + 1, 2))


@dataclass(frozen=True, order=True, init=False)
class SpinChargeSector:
    """
    Product sector for particle number, SU(2) spin, and a real-Abelian irrep.

    ``point_group`` is an XOR character bit label; zero is the identity.
    Spatial labels distinguish multiplicity blocks without changing spin CGs.
    """

    charge: int
    irrep: SU2Irrep
    point_group: int
    _hash: int = field(init=False, repr=False, compare=False)

    def __init__(self, charge: int, irrep: SU2Irrep, multiplicity: int | None = None, *, point_group: int = 0):
        if multiplicity not in (None, 1):
            raise ValueError(
                "SpinChargeSector multiplicity is implicit; repeat the sector "
                "on a leg to represent multiple copies."
            )
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "irrep", irrep)
        if int(point_group) != point_group or not 0 <= point_group < 2**31:
            raise ValueError("The Abelian spatial irrep must be a nonnegative bit label.")
        object.__setattr__(self, "point_group", int(point_group))
        self.__post_init__()
        object.__setattr__(
            self,
            "_hash",
            hash((self.charge, self.irrep.two_j, self.point_group)),
        )

    def __post_init__(self):
        if int(self.charge) != self.charge or self.charge < 0:
            raise ValueError(f"Sector charge must be a non-negative integer, got {self.charge!r}.")

    @property
    def packed_charge(self) -> int:
        """Lossless (N, spatial irrep) label for packed complementary sectors.

        The low 32 bits contain N, the high bits the real-Abelian irrep bit
        label. This is a storage key, never an additive particle number.
        """
        if self.charge >= 2**32:
            raise OverflowError("Particle number exceeds the packed sector range.")
        return int(self.charge) | (self.point_group << 32)

    @classmethod
    def from_packed(cls, charge, two_j):
        return cls(int(charge) & 0xffffffff, SU2Irrep(int(two_j)),
                   point_group=int(charge) >> 32)

    @property
    def two_j(self) -> int:
        return self.irrep.two_j

    @property
    def multiplicity(self) -> int:
        """
        Compatibility property.

        Multiplicity is represented by repeated sector entries on a tensor leg,
        not by storing a copy count inside the sector label itself.
        """
        return 1

    @property
    def dim(self) -> int:
        return self.irrep.dim

    def __hash__(self) -> int:
        return self._hash


def fuse_charge_spin_sectors(
    left: SpinChargeSector, right: SpinChargeSector
) -> tuple[SpinChargeSector, ...]:
    """
    Fuse two particle-number, SU(2), and real-Abelian sectors.

    Charges add, spatial labels combine by XOR, and spins follow the CG rule.
    Multiple copies of the same fused sector are represented implicitly by
    repeated sector entries on tensor legs, not by a field on the sector label.
    """
    charge = left.charge + right.charge
    return tuple(SpinChargeSector(charge, irrep, point_group=left.point_group ^ right.point_group)
                 for irrep in fuse_irreps(left.irrep, right.irrep))


@dataclass(frozen=True)
class SpatialOrbitalSite:
    """
    SU(2)-adapted local basis for one spatial orbital.

    Physical states are ordered as

        0. ``|empty>``
        1. ``|up>``
        2. ``|down>``
        3. ``|double>``

    and grouped into charge+spin sectors as

        - ``N=0, S=0`` : [0]
        - ``N=1, S=1/2`` : [1, 2]
        - ``N=2, S=0`` : [3]
    """

    labels: tuple[str, ...] = ("empty", "up", "down", "double")
    sectors: tuple[SpinChargeSector, ...] = (
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(1, SU2Irrep(1)),
        SpinChargeSector(2, SU2Irrep(0)),
    )
    state_index: tuple[tuple[int, ...], ...] = ((0,), (1, 2), (3,))

    @property
    def d(self) -> int:
        return len(self.labels)

    @property
    def qn(self) -> tuple[SpinChargeSector, ...]:
        return self.sectors

    @property
    def degeneracy(self) -> tuple[int, ...]:
        return tuple(len(idx) for idx in self.state_index)


@dataclass(frozen=True)
class SpinOrbitalSite:
    """
    Abelian local basis for one spin-orbital.

    This matches the current QC-DMRG site picture more closely than
    :class:`SpatialOrbitalSite`: a site is either empty or occupied, and the
    occupied state carries a fixed spin projection set by ``spin``.

    We use doubled spin projection ``2Sz`` so all quantum-number labels remain
    integer-valued:

        - ``spin='up'``   -> occupied state has ``(N=1, 2Sz=+1)``
        - ``spin='down'`` -> occupied state has ``(N=1, 2Sz=-1)``
    """

    spin: str = "up"
    labels: tuple[str, ...] = ("empty", "occupied")
    state_index: tuple[tuple[int, ...], ...] = ((0,), (1,))

    def __post_init__(self):
        if self.spin not in {"up", "down"}:
            raise ValueError(f"SpinOrbitalSite spin must be 'up' or 'down', got {self.spin!r}.")

    @property
    def d(self) -> int:
        return len(self.labels)

    @property
    def qn(self) -> tuple[tuple[int, int], ...]:
        occ_sz2 = 1 if self.spin == "up" else -1
        return ((0, 0), (1, occ_sz2))

    @property
    def degeneracy(self) -> tuple[int, ...]:
        return tuple(len(idx) for idx in self.state_index)
