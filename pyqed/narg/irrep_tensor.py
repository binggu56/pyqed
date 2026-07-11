"""Irrep-organized block tensors for Abelian and non-Abelian symmetries.

This module is intentionally small.  It provides the common data model we want
for both the existing Abelian NARG path and the future SU(2)-adapted path:

* ``Irrep`` labels a basis sector.
* ``OpIrrep`` labels an operator's symmetry type.
* ``IrrepSite`` stores sector dimensions.
* ``IrrepTensor`` stores block matrices between sectors.

The first supported operations are conservative: block validation, adjoint,
scalar block matrix multiplication, and dense assembly for debugging.  Non-scalar
SU(2) tensor recoupling should be added on top of this interface rather than
hidden in ordinary matrix multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Protocol

import numpy as np


def twice_spin(value) -> int:
    """Return ``2*S`` as an integer for an SU(2) spin or tensor rank."""
    if isinstance(value, str):
        frac = Fraction(value)
    else:
        frac = Fraction(value).limit_denominator()
    doubled = 2 * frac
    if doubled.denominator != 1:
        raise ValueError(f"SU(2) labels must be integer or half-integer, got {value!r}")
    return int(doubled)


def spin_value(j2: int) -> Fraction:
    """Convert an internal doubled spin label to the physical spin value."""
    return Fraction(int(j2), 2)


def spin_label(j2: int) -> str:
    """Human-readable SU(2) spin/rank label."""
    value = spin_value(j2)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def u1_su2_irrep(nelec: int, spin) -> "Irrep":
    """Create a ``(Ne, 2S)`` sector label from physical ``S`` notation."""
    return Irrep((int(nelec), twice_spin(spin)))


def u1_su2_op_irrep(dnelec: int, rank) -> "OpIrrep":
    """Create a ``(dNe, 2J)`` operator label from physical tensor rank ``J``."""
    return OpIrrep((int(dnelec), twice_spin(rank)))


@dataclass(frozen=True, order=True)
class Irrep:
    """Basis-sector label.

    ``charge`` can be an int for U(1), a tuple for product symmetries such as
    ``(Ne, j2)`` where ``j2 = 2*S``, or any immutable label understood by
    the symmetry backend.
    ``alpha`` labels multiplicity copies of the same irrep.
    """

    charge: object
    alpha: int = 0


@dataclass(frozen=True)
class OpIrrep:
    """Operator symmetry label.

    ``charge`` is the operator's irrep/charge.  For U(1) this is a charge shift.
    For SU(2), use doubled tensor rank ``j2 = 2*J``.  For product
    symmetries, use a tuple such as ``(dnelec, j2)``.
    """

    charge: object


class Symmetry(Protocol):
    name: str

    def dual(self, irrep: object) -> object:
        ...

    def fuse(self, left: object, op: object) -> tuple[object, ...]:
        ...

    def allows(self, bra: object, op: object, ket: object) -> bool:
        ...


@dataclass(frozen=True)
class U1Symmetry:
    name: str = "U1"

    def dual(self, irrep: int) -> int:
        return -int(irrep)

    def fuse(self, left: int, op: int) -> tuple[int, ...]:
        return (int(left) + int(op),)

    def allows(self, bra: int, op: int, ket: int) -> bool:
        return int(bra) == int(ket) + int(op)


@dataclass(frozen=True)
class SU2Symmetry:
    """SU(2) using doubled angular momenta."""

    name: str = "SU2"

    def dual(self, irrep: int) -> int:
        return int(irrep)

    def fuse(self, left: int, op: int) -> tuple[int, ...]:
        left = int(left)
        op = int(op)
        return tuple(range(abs(left - op), left + op + 1, 2))

    def allows(self, bra: int, op: int, ket: int) -> bool:
        return int(bra) in self.fuse(int(ket), int(op))


@dataclass(frozen=True)
class ProductSymmetry:
    factors: tuple[Symmetry, ...]
    name: str = "Product"

    def dual(self, irrep: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(sym.dual(x) for sym, x in zip(self.factors, irrep))

    def fuse(self, left: tuple[object, ...], op: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
        choices = [sym.fuse(x, y) for sym, x, y in zip(self.factors, left, op)]
        out = [()]
        for values in choices:
            out = [prefix + (value,) for prefix in out for value in values]
        return tuple(out)

    def allows(self, bra: tuple[object, ...], op: tuple[object, ...], ket: tuple[object, ...]) -> bool:
        return all(sym.allows(b, o, k) for sym, b, o, k in zip(self.factors, bra, op, ket))


@dataclass(frozen=True)
class IrrepSite:
    """Sector dimensions for one block/site Hilbert space."""

    symmetry: Symmetry
    dims: Mapping[Irrep, int]

    def __post_init__(self):
        for irrep, dim in self.dims.items():
            if not isinstance(irrep, Irrep):
                raise TypeError(f"sector key must be Irrep, got {type(irrep)!r}")
            if int(dim) < 0:
                raise ValueError(f"negative sector dimension for {irrep}: {dim}")

    @property
    def irreps(self) -> tuple[Irrep, ...]:
        return tuple(self.dims.keys())

    @property
    def dim(self) -> int:
        return int(sum(self.dims.values()))

    def sector_dim(self, irrep: Irrep) -> int:
        return int(self.dims.get(irrep, 0))

    def offsets(self) -> dict[Irrep, slice]:
        offsets = {}
        start = 0
        for irrep, dim in self.dims.items():
            stop = start + int(dim)
            offsets[irrep] = slice(start, stop)
            start = stop
        return offsets


class IrrepTensor:
    """Block matrix between two ``IrrepSite`` spaces."""

    def __init__(
        self,
        bra: IrrepSite,
        ket: IrrepSite,
        op: OpIrrep,
        blocks: Mapping[tuple[Irrep, Irrep], np.ndarray] | None = None,
        *,
        validate: bool = True,
    ):
        self.bra = bra
        self.ket = ket
        self.op = op
        self.blocks = dict(blocks or {})
        if validate:
            self.validate()

    @classmethod
    def zeros(cls, bra: IrrepSite, ket: IrrepSite, op: OpIrrep) -> "IrrepTensor":
        return cls(bra, ket, op, {})

    @classmethod
    def from_dense(
        cls,
        bra: IrrepSite,
        ket: IrrepSite,
        op: OpIrrep,
        dense: np.ndarray,
        *,
        drop_zeros: bool = True,
        atol: float = 0.0,
    ) -> "IrrepTensor":
        dense = np.asarray(dense)
        if dense.shape != (bra.dim, ket.dim):
            raise ValueError(f"dense shape {dense.shape} does not match {(bra.dim, ket.dim)}")
        bra_offsets = bra.offsets()
        ket_offsets = ket.offsets()
        blocks = {}
        for bra_irrep in bra.irreps:
            for ket_irrep in ket.irreps:
                if not bra.symmetry.allows(bra_irrep.charge, op.charge, ket_irrep.charge):
                    continue
                block = dense[bra_offsets[bra_irrep], ket_offsets[ket_irrep]]
                if drop_zeros and not np.any(np.abs(block) > atol):
                    continue
                blocks[(bra_irrep, ket_irrep)] = block.copy()
        return cls(bra, ket, op, blocks)

    @classmethod
    def identity(cls, site: IrrepSite) -> "IrrepTensor":
        blocks = {
            (irrep, irrep): np.eye(dim)
            for irrep, dim in site.dims.items()
            if dim
        }
        return cls(site, site, OpIrrep(0 if isinstance(next(iter(site.dims)).charge, int) else tuple(0 for _ in site.symmetry.factors)), blocks)

    def validate(self) -> None:
        if self.bra.symmetry != self.ket.symmetry:
            raise ValueError("bra and ket sites use different symmetries")
        sym = self.bra.symmetry
        for (bra_irrep, ket_irrep), block in self.blocks.items():
            if bra_irrep not in self.bra.dims:
                raise KeyError(f"bra sector {bra_irrep} not present")
            if ket_irrep not in self.ket.dims:
                raise KeyError(f"ket sector {ket_irrep} not present")
            expected = (self.bra.sector_dim(bra_irrep), self.ket.sector_dim(ket_irrep))
            if np.shape(block) != expected:
                raise ValueError(f"block {(bra_irrep, ket_irrep)} has shape {np.shape(block)}, expected {expected}")
            if not sym.allows(bra_irrep.charge, self.op.charge, ket_irrep.charge):
                raise ValueError(f"operator {self.op} does not allow {ket_irrep} -> {bra_irrep}")

    def block(self, bra: Irrep, ket: Irrep) -> np.ndarray:
        return self.blocks.get((bra, ket), np.zeros((self.bra.sector_dim(bra), self.ket.sector_dim(ket))))

    def adjoint(self) -> "IrrepTensor":
        sym = self.bra.symmetry
        blocks = {
            (ket, bra): block.conj().T
            for (bra, ket), block in self.blocks.items()
        }
        return IrrepTensor(self.ket, self.bra, OpIrrep(sym.dual(self.op.charge)), blocks)

    def scalar_matmul(self, other: "IrrepTensor") -> "IrrepTensor":
        """Compose tensors when both operators are scalar under the symmetry."""
        scalar = self._zero_op_charge()
        if self.op.charge != scalar or other.op.charge != scalar:
            raise NotImplementedError("generic non-scalar tensor recoupling is not implemented")
        if self.ket != other.bra:
            raise ValueError("inner IrrepSite mismatch")

        out: dict[tuple[Irrep, Irrep], np.ndarray] = {}
        for (bra, mid), left in self.blocks.items():
            for (mid2, ket), right in other.blocks.items():
                if mid2 != mid:
                    continue
                key = (bra, ket)
                prod = left @ right
                out[key] = prod if key not in out else out[key] + prod
        return IrrepTensor(self.bra, other.ket, OpIrrep(scalar), out)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.bra.dim, self.ket.dim), dtype=self.dtype)
        bra_offsets = self.bra.offsets()
        ket_offsets = self.ket.offsets()
        for (bra, ket), block in self.blocks.items():
            dense[bra_offsets[bra], ket_offsets[ket]] = block
        return dense

    @property
    def dtype(self):
        if not self.blocks:
            return float
        return np.result_type(*[block.dtype for block in self.blocks.values()])

    def _zero_op_charge(self):
        charge = self.op.charge
        if isinstance(charge, tuple):
            return tuple(0 for _ in charge)
        return 0


def u1_site(charges_and_dims: Iterable[tuple[int, int]]) -> IrrepSite:
    return IrrepSite(U1Symmetry(), {Irrep(int(q)): int(dim) for q, dim in charges_and_dims})


def u1_su2_site(sectors: Iterable[tuple[int, int, int]]) -> IrrepSite:
    sym = ProductSymmetry((U1Symmetry("Ne"), SU2Symmetry("SU2")), name="U1xSU2")
    return IrrepSite(sym, {Irrep((int(ne), int(j2))): int(dim) for ne, j2, dim in sectors})


def u1_su2_site_from_spin(sectors: Iterable[tuple[int, object, int]]) -> IrrepSite:
    """Build a U(1)xSU(2) site from physical ``(Ne, S, dim)`` sector labels."""
    sym = ProductSymmetry((U1Symmetry("Ne"), SU2Symmetry("SU2")), name="U1xSU2")
    return IrrepSite(sym, {u1_su2_irrep(ne, spin): int(dim) for ne, spin, dim in sectors})
