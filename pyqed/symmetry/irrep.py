"""Shared irrep-organized spaces and tensors for Abelian and non-Abelian symmetries.

This module is intentionally small.  It provides the common data model we want
for both the existing Abelian NARG path and the future SU(2)-adapted path:

* ``Irrep`` labels a basis sector.
* ``OpIrrep`` labels an operator's symmetry type.
* ``Leg`` stores symmetry sectors and their multiplicities.
* ``IrrepTensor`` stores block matrices between sectors.

The first supported operations are conservative: block validation, adjoint,
scalar block matrix multiplication, and dense assembly for debugging.  Non-scalar
SU(2) tensor recoupling should be added on top of this interface rather than
hidden in ordinary matrix multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
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


def _site_irrep_layout(site):
    from pyqed.lattice import Site

    if not isinstance(site, Site):
        raise TypeError("site must be a canonical pyqed.lattice.Site.")
    if site.charges is None:
        leg = site.leg
        if leg.symmetry is None:
            raise ValueError("a symmetry-aware operator requires site charges or a symmetry Leg.")
        if leg.dim != site.dim:
            raise ValueError(
                "operators on a full non-Abelian basis require explicit per-state charges."
            )
        irreps = leg.irreps
        state_indices = []
        start = 0
        for irrep in irreps:
            stop = start + leg.sector_dim(irrep)
            state_indices.append(tuple(range(start, stop)))
            start = stop
        return leg.symmetry, irreps, tuple(state_indices)
    if len(site.charge_labels) == 1:
        symmetry = U1Symmetry(site.charge_labels[0])
        state_irreps = tuple(Irrep(charge[0]) for charge in site.charges)
    else:
        symmetry = ProductSymmetry(
            tuple(U1Symmetry(label) for label in site.charge_labels),
            name="x".join(site.charge_labels),
        )
        state_irreps = tuple(Irrep(tuple(charge)) for charge in site.charges)
    irreps = tuple(dict.fromkeys(state_irreps))
    state_indices = tuple(
        tuple(
            index
            for index, state_irrep in enumerate(state_irreps)
            if state_irrep == irrep
        )
        for irrep in irreps
    )
    return symmetry, irreps, state_indices


def _restore_leg(
    sectors,
    dims,
    symmetry,
    direction,
    name,
    labels=(),
    local_charges=(),
):
    return Leg(
        sectors,
        dims,
        symmetry=symmetry,
        direction=direction,
        name=name,
        labels=labels,
        local_charges=local_charges,
    )


@dataclass(frozen=True, init=False, eq=False)
class Leg:
    """Tensor-index space with optional symmetry sectors and orientation.

    ``dims`` stores the packed multiplicity of each sector.  For reduced
    non-Abelian tensors this is the reduced dimension, while ``full_dim``
    includes the dimensions of the irreducible representations.
    """

    sectors: tuple[object, ...]
    dims: Mapping[object, int]
    symmetry: Symmetry | None
    direction: int
    name: str | None
    labels: tuple[str, ...]
    local_charges: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        sectors=None,
        dims=None,
        *,
        dim=None,
        symmetry=None,
        direction=1,
        name=None,
        labels=(),
        local_charges=(),
    ):
        if dim is not None:
            if sectors is not None or dims is not None:
                raise TypeError("dim cannot be combined with sectors or dims.")
            sectors = (None,)
            dims = {None: int(dim)}
        if sectors is None:
            raise TypeError("Leg requires sectors/dims or dim.")
        if isinstance(sectors, Mapping):
            if dims is not None:
                raise TypeError("dims must be omitted when sectors is a mapping.")
            dims = dict(sectors)
            sectors = tuple(dims)
        else:
            sectors = tuple(sectors)
            if dims is None:
                raise TypeError("Leg requires sector dimensions.")
            dims = {sector: int(dims[sector]) for sector in sectors}
        if not sectors:
            raise ValueError("Leg requires at least one sector.")
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("Leg direction must be +1 or -1.")
        if len(set(sectors)) != len(sectors):
            raise ValueError("Leg sectors must be unique.")
        normalized = {}
        for sector in sectors:
            if sector not in dims:
                raise ValueError(f"missing dimension for sector {sector!r}.")
            dim = int(dims[sector])
            if dim <= 0:
                raise ValueError(f"sector dimension for {sector!r} must be positive.")
            if symmetry is not None and not isinstance(sector, Irrep):
                raise TypeError(
                    "symmetry-aware Leg sectors must be Irrep objects, "
                    f"got {type(sector)!r}."
                )
            normalized[sector] = dim
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "dims", MappingProxyType(normalized))
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "name", None if name is None else str(name))
        labels = tuple(str(label) for label in labels)
        local_charges = tuple(
            tuple(int(component) for component in charge)
            for charge in local_charges
        )
        total_dim = int(sum(normalized.values()))
        if labels and len(labels) != total_dim:
            raise ValueError("the number of Leg labels must match its dimension.")
        if local_charges and len(local_charges) != total_dim:
            raise ValueError("a Leg must provide one local charge per basis state.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "local_charges", local_charges)

    def __eq__(self, other):
        if (
            not isinstance(other, Leg)
            or self.symmetry != other.symmetry
            or self.direction != other.direction
        ):
            return False
        if dict(self.dims) != dict(other.dims):
            return False
        if self.symmetry is None:
            return self.sectors == other.sectors
        return True

    __hash__ = None

    def __reduce__(self):
        return _restore_leg, (
            self.sectors,
            dict(self.dims),
            self.symmetry,
            self.direction,
            self.name,
            self.labels,
            self.local_charges,
        )

    @property
    def sector_dims(self):
        return self.dims

    @classmethod
    def from_site(cls, site):
        """Build the Abelian irrep decomposition of a canonical physical site."""
        symmetry, irreps, state_indices = _site_irrep_layout(site)
        return cls(
            {
                irrep: len(indices)
                for irrep, indices in zip(irreps, state_indices)
            },
            symmetry=symmetry,
        )

    @property
    def irreps(self) -> tuple[Irrep, ...]:
        if any(not isinstance(sector, Irrep) for sector in self.sectors):
            raise TypeError("this Leg uses backend sector labels rather than shared Irrep labels.")
        return self.sectors

    @property
    def dim(self) -> int:
        """Packed dimension used by the stored tensor representation."""
        return self.reduced_dim

    @property
    def reduced_dim(self) -> int:
        return int(sum(self.dims[sector] for sector in self.sectors))

    @property
    def total_dim(self) -> int:
        """Alias for the packed dimension used by existing MPO code."""
        return self.reduced_dim

    @staticmethod
    def _irrep_dimension(symmetry, charge):
        if isinstance(symmetry, SU2Symmetry):
            return int(charge) + 1
        if isinstance(symmetry, ProductSymmetry):
            return int(np.prod([
                Leg._irrep_dimension(factor, component)
                for factor, component in zip(symmetry.factors, charge)
            ]))
        return 1

    @property
    def full_dim(self) -> int:
        if self.symmetry is None:
            return self.reduced_dim
        return int(sum(
            self.dims[irrep] * self._irrep_dimension(self.symmetry, irrep.charge)
            for irrep in self.irreps
        ))

    def sector_dim(self, sector) -> int:
        return int(self.dims.get(sector, 0))

    def sector_full_dim(self, sector) -> int:
        """Dimension of a sector including non-Abelian irrep degeneracy."""
        if sector not in self.dims:
            return 0
        if self.symmetry is None:
            return self.sector_dim(sector)
        if not isinstance(sector, Irrep):
            raise TypeError("a symmetry-aware Leg requires Irrep sector labels.")
        return self.sector_dim(sector) * self._irrep_dimension(
            self.symmetry,
            sector.charge,
        )

    def multiplicity(self, sector) -> int:
        return self.sector_dim(sector)

    def slices(self) -> dict[object, slice]:
        offsets = {}
        start = 0
        for sector in self.sectors:
            dim = self.dims[sector]
            stop = start + int(dim)
            offsets[sector] = slice(start, stop)
            start = stop
        return offsets

    def offsets(self) -> dict[object, slice]:
        return self.slices()

    @classmethod
    def from_dims(
        cls,
        sector_dims,
        sectors=None,
        *,
        symmetry=None,
        direction=1,
        name=None,
    ):
        if sectors is None:
            sectors = tuple(sector_dims)
        return cls(
            tuple(sectors),
            sector_dims,
            symmetry=symmetry,
            direction=direction,
            name=name,
        )

    @classmethod
    def trivial(
        cls,
        dim,
        *,
        direction=1,
        name=None,
        labels=(),
        local_charges=(),
    ):
        """Return a dense index represented by one trivial sector."""
        dim = int(dim)
        if dim < 1:
            raise ValueError("a trivial Leg dimension must be positive.")
        return cls(
            (None,),
            {None: dim},
            direction=direction,
            name=name,
            labels=labels,
            local_charges=local_charges,
        )

    @classmethod
    def from_slices(
        cls,
        sector_slices,
        *,
        symmetry=None,
        direction=1,
        name=None,
    ):
        return cls(
            tuple(sector_slices),
            {
                sector: int(slice_.stop - slice_.start)
                for sector, slice_ in sector_slices.items()
            },
            symmetry=symmetry,
            direction=direction,
            name=name,
        )

    @classmethod
    def from_tensor_axis(cls, tensor, axis, *, name=None):
        """Build a canonical leg from one reduced tensor axis."""
        axis = int(axis)
        if axis < 0:
            axis += tensor.rank
        if axis < 0 or axis >= tensor.rank:
            raise ValueError(
                f"Tensor axis {axis} out of range for rank-{tensor.rank} tensor."
            )
        if hasattr(tensor, "legs") and len(tensor.legs) == tensor.rank:
            source = tensor.legs[axis]
            return cls(
                source.sectors,
                source.dims,
                symmetry=source.symmetry,
                direction=source.direction,
                name=source.name if name is None else name,
                labels=source.labels,
                local_charges=source.local_charges,
            )
        dims = {}
        for key, block in tensor.data.items():
            sector = key[axis]
            dim = int(np.asarray(block).shape[axis])
            known = dims.get(sector)
            if known is not None and known != dim:
                raise ValueError(
                    f"Inconsistent dimension for sector {sector!r} on axis "
                    f"{axis}: {known} vs {dim}."
                )
            dims[sector] = dim
        sectors = tuple(dict.fromkeys(tensor.qns[axis]))
        for sector in sectors:
            dims.setdefault(sector, tensor.qns[axis].count(sector))
        return cls(
            sectors,
            dims,
            direction=tensor.dirs[axis],
            name=name,
        )

    def same_blocks(self, other) -> bool:
        return (
            isinstance(other, Leg)
            and self.sectors == other.sectors
            and dict(self.dims) == dict(other.dims)
            and self.symmetry == other.symmetry
        )

    def compatible_with(self, other) -> bool:
        return self.same_blocks(other) and self.direction == other.direction

    def dual_compatible_with(self, other) -> bool:
        return self.same_blocks(other) and self.direction == -other.direction

    def dual(self):
        """Return the oppositely oriented view of the same index space.

        Sector labels identify the shared packed basis and therefore remain
        unchanged.  Charge conjugation, when required for an operator irrep,
        is handled by :meth:`Symmetry.dual` rather than by relabelling an MPS
        bond.
        """
        return Leg(
            self.sectors,
            self.dims,
            symmetry=self.symmetry,
            direction=-self.direction,
            name=self.name,
            labels=self.labels,
            local_charges=self.local_charges,
        )

    def fuse(self, other):
        if not isinstance(other, Leg):
            return NotImplemented
        if self.symmetry is None or self.symmetry != other.symmetry:
            raise TypeError("fusion requires Legs with the same explicit symmetry.")
        dims = {}
        for left in self.irreps:
            for right in other.irreps:
                for charge in self.symmetry.fuse(left.charge, right.charge):
                    irrep = Irrep(charge)
                    dims[irrep] = (
                        dims.get(irrep, 0)
                        + self.dims[left] * other.dims[right]
                    )
        return Leg(
            dims,
            symmetry=self.symmetry,
            direction=self.direction,
        )


class IrrepTensor:
    """Rank-generic reduced tensor stored over canonical :class:`Leg` objects.

    The primary constructor accepts ``(blocks, legs, directions)`` for an
    arbitrary-rank tensor.  For symmetry-operator code, the established
    ``(bra_leg, ket_leg, op_irrep, blocks)`` form remains available.  In both
    cases the numerical payload has one reduced dense block per sector tuple;
    fusion and recoupling metadata are carried alongside the canonical legs.
    """

    def __init__(
        self,
        first=None,
        second=None,
        third=None,
        fourth=None,
        *,
        data=None,
        qns=None,
        dirs=None,
        fusion_legs=None,
        metadata=None,
        fusion_edges=None,
        validate=True,
    ):
        if data is not None or qns is not None or dirs is not None:
            if second is not None or third is not None:
                raise TypeError(
                    "use positional tensor arguments or data/qns/dirs, not both."
                )
            if data is not None and first is not None:
                raise TypeError("supply tensor blocks as either first or data, not both.")
            blocks = first if data is None else data
            if blocks is None or qns is None or dirs is None:
                raise TypeError("data, qns, and dirs must be supplied together.")
            first, second, third = blocks, qns, dirs
        self._operator_mode = isinstance(first, Leg)
        if self._operator_mode:
            self._init_operator(first, second, third, fourth)
        else:
            self._init_tensor(
                first,
                second,
                third,
                fusion_legs=fusion_legs,
                fusion_edges=fusion_edges,
                metadata=metadata,
            )
        if validate:
            self.validate()

    @staticmethod
    def _ordered_unique(items):
        return tuple(dict.fromkeys(items))

    @classmethod
    def _legs_from_layout(cls, data, qns, dirs):
        legs = []
        for axis, (axis_qns, direction) in enumerate(zip(qns, dirs)):
            sectors = cls._ordered_unique(axis_qns)
            dims = {}
            for key, block in data.items():
                sector = key[axis]
                dim = int(np.asarray(block).shape[axis])
                known = dims.get(sector)
                if known is not None and known != dim:
                    raise ValueError(
                        f"Inconsistent reduced dimension for sector {sector!r} "
                        f"on axis {axis}: {known} vs {dim}."
                    )
                dims[sector] = dim
            for sector in sectors:
                dims.setdefault(sector, list(axis_qns).count(sector))
            legs.append(Leg(sectors, dims, direction=direction))
        return tuple(legs)

    def _init_tensor(
        self,
        blocks,
        legs_or_qns,
        directions,
        *,
        fusion_legs,
        fusion_edges,
        metadata,
    ):
        self.data = {
            tuple(key): np.asarray(value)
            for key, value in dict(blocks).items()
        }
        if fusion_legs is not None and fusion_edges is not None:
            raise ValueError("Specify only one of fusion_legs or fusion_edges.")
        if legs_or_qns and all(isinstance(leg, Leg) for leg in legs_or_qns):
            legs = tuple(legs_or_qns)
            dirs = tuple(int(direction) for direction in directions)
            if tuple(leg.direction for leg in legs) != dirs:
                legs = tuple(
                    Leg(
                        leg.sectors,
                        leg.dims,
                        symmetry=leg.symmetry,
                        direction=direction,
                        name=leg.name,
                        labels=leg.labels,
                        local_charges=leg.local_charges,
                    )
                    for leg, direction in zip(legs, dirs)
                )
            qns = [list(leg.sectors) for leg in legs]
        else:
            qns = [list(axis_qns) for axis_qns in legs_or_qns]
            dirs = tuple(int(direction) for direction in directions)
            legs = self._legs_from_layout(self.data, qns, dirs)
        self.legs = legs
        self.qns = qns
        self.dirs = list(dirs)
        self.rank = len(self.legs)
        selected_fusion_legs = fusion_legs if fusion_legs is not None else fusion_edges
        if selected_fusion_legs is None:
            selected_fusion_legs = (None,) * self.rank
        self.fusion_legs = list(selected_fusion_legs)
        if len(self.fusion_legs) != self.rank:
            raise ValueError(
                "fusion_legs/legs length mismatch: "
                f"{len(self.fusion_legs)} vs {self.rank}"
            )
        self.metadata = dict(metadata or {})

    def _init_operator(self, bra, ket, op, blocks):
        if not isinstance(ket, Leg):
            raise TypeError("ket must be a Leg.")
        if not isinstance(op, OpIrrep):
            raise TypeError("op must be an OpIrrep.")
        self.bra = bra
        self.ket = ket
        self.op = op
        self.legs = (bra, ket)
        self.qns = [list(bra.sectors), list(ket.sectors)]
        self.dirs = [bra.direction, ket.direction]
        self.rank = 2
        self.fusion_legs = [None, None]
        self.metadata = {}
        self.data = {
            tuple(key): np.asarray(value)
            for key, value in dict(blocks or {}).items()
        }

    @property
    def blocks(self):
        return self.data

    @property
    def shape(self):
        return tuple(leg.dim for leg in self.legs)

    @property
    def ndim(self):
        return self.rank

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def storage_mode(self):
        """Numerical storage family selected by the tensor's legs."""
        if all(
            leg.symmetry is None and leg.sectors == (None,)
            for leg in self.legs
        ):
            return "dense"
        return "nonabelian" if self.has_nonabelian_symmetry else "abelian"

    @property
    def is_dense(self):
        return self.storage_mode == "dense"

    @property
    def nblocks(self):
        return len(self.data)

    @property
    def has_nonabelian_symmetry(self):
        for leg in self.legs:
            if isinstance(leg.symmetry, (SU2Symmetry, ProductSymmetry)):
                return True
            for sector in leg.sectors:
                if hasattr(sector, "is_abelian") and not sector.is_abelian:
                    return True
                irrep = getattr(sector, "irrep", None)
                if int(getattr(irrep, "dim", 1)) > 1:
                    return True
                if any(
                    int(getattr(component, "dim", 1)) > 1
                    for component in getattr(sector, "components", ())
                ):
                    return True
        return False

    @property
    def fusion_edges(self):
        return self.fusion_legs

    @property
    def dtype(self):
        if not self.data:
            return float
        return np.result_type(*[block.dtype for block in self.data.values()])

    def validate(self):
        if len(self.legs) != self.rank or len(self.dirs) != self.rank:
            raise ValueError("legs/directions rank mismatch.")
        sector_sets = [set(leg.sectors) for leg in self.legs]
        for key, block in self.data.items():
            if len(key) != self.rank:
                raise ValueError(
                    f"Block key rank mismatch: expected {self.rank}, got "
                    f"{len(key)} for key {key!r}."
                )
            for axis, sector in enumerate(key):
                if sector not in sector_sets[axis]:
                    raise ValueError(
                        f"Sector {sector!r} on leg {axis} is not present in "
                        "declared leg sectors."
                    )
            expected = tuple(
                self.legs[axis].sector_dim(sector)
                for axis, sector in enumerate(key)
            )
            if np.shape(block) != expected:
                raise ValueError(
                    f"Block {key!r} has shape {np.shape(block)!r}; expected "
                    f"{expected!r} from its Legs."
                )
        if self._operator_mode:
            if self.bra.symmetry != self.ket.symmetry:
                raise ValueError("bra and ket legs use different symmetries.")
            symmetry = self.bra.symmetry
            if symmetry is None:
                raise ValueError("operator IrrepTensor requires explicit symmetry.")
            for bra_irrep, ket_irrep in self.data:
                if not symmetry.allows(
                    bra_irrep.charge,
                    self.op.charge,
                    ket_irrep.charge,
                ):
                    raise ValueError(
                        f"operator {self.op} does not allow "
                        f"{ket_irrep} -> {bra_irrep}."
                    )

    def copy(self):
        metadata = self.metadata.copy()
        metadata.pop("_cpp_split_site", None)
        if self._operator_mode:
            return IrrepTensor(
                self.bra,
                self.ket,
                self.op,
                {key: block.copy() for key, block in self.data.items()},
            )
        return type(self)(
            {key: block.copy() for key, block in self.data.items()},
            self.legs,
            self.dirs,
            fusion_legs=self.fusion_legs,
            metadata=metadata,
        )

    def _check_compatible(self, other):
        if not isinstance(other, IrrepTensor):
            raise TypeError(f"Expected IrrepTensor, got {type(other).__name__}.")
        if self._operator_mode != other._operator_mode:
            raise ValueError("IrrepTensor metadata mismatch.")
        if not self._operator_mode and not (
            self.has_nonabelian_symmetry or other.has_nonabelian_symmetry
        ):
            if self.rank != other.rank:
                raise ValueError("IrrepTensor metadata mismatch.")
            return
        if self.legs != other.legs or self.fusion_legs != other.fusion_legs:
            raise ValueError("IrrepTensor metadata mismatch.")
        if self._operator_mode and self.op != other.op:
            raise ValueError("IrrepTensor operator irrep mismatch.")

    def _binary(self, other, sign):
        self._check_compatible(other)
        blocks = {key: block.copy() for key, block in self.data.items()}
        for key, block in other.data.items():
            blocks[key] = blocks.get(key, 0) + sign * block
        if self._operator_mode:
            return IrrepTensor(self.bra, self.ket, self.op, blocks)
        return type(self)(
            blocks,
            self.legs,
            self.dirs,
            fusion_legs=self.fusion_legs,
            metadata=self.metadata.copy(),
        )

    def __add__(self, other):
        return self._binary(other, 1)

    def __sub__(self, other):
        return self._binary(other, -1)

    def __mul__(self, scalar):
        blocks = {key: block * scalar for key, block in self.data.items()}
        if self._operator_mode:
            return IrrepTensor(self.bra, self.ket, self.op, blocks)
        return type(self)(
            blocks,
            self.legs,
            self.dirs,
            fusion_legs=self.fusion_legs,
            metadata=self.metadata.copy(),
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1.0 / scalar)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple, np.ndarray)):
            axes = tuple(axes[0])
        if sorted(axes) != list(range(self.rank)):
            raise ValueError(
                f"Invalid transpose axes {axes!r} for rank-{self.rank} tensor."
            )
        blocks = {
            tuple(key[axis] for axis in axes): np.transpose(block, axes)
            for key, block in self.data.items()
        }
        legs = [self.legs[axis] for axis in axes]
        dirs = [self.dirs[axis] for axis in axes]
        fusion_legs = [self.fusion_legs[axis] for axis in axes]
        return type(self)(
            blocks,
            legs,
            dirs,
            fusion_legs=fusion_legs,
            metadata=self.metadata.copy(),
        )

    def conj(self):
        if self._operator_mode:
            return IrrepTensor(
                self.bra,
                self.ket,
                self.op,
                {key: block.conj() for key, block in self.data.items()},
            )
        return type(self)(
            {key: block.conj() for key, block in self.data.items()},
            [leg.dual() for leg in self.legs],
            [-direction for direction in self.dirs],
            fusion_legs=self.fusion_legs,
            metadata=self.metadata.copy(),
        )

    def block(self, *sectors):
        if len(sectors) == 1 and isinstance(sectors[0], tuple):
            sectors = sectors[0]
        key = tuple(sectors)
        if key in self.data:
            return self.data[key]
        shape = tuple(
            self.legs[axis].sector_dim(sector)
            for axis, sector in enumerate(key)
        )
        return np.zeros(shape, dtype=self.dtype)

    def to_dense(self):
        dense = np.zeros(
            tuple(leg.dim for leg in self.legs),
            dtype=self.dtype,
        )
        offsets = [leg.offsets() for leg in self.legs]
        for key, block in self.data.items():
            dense[tuple(offsets[axis][sector] for axis, sector in enumerate(key))] = block
        return dense

    def __array__(self, dtype=None, copy=None):
        dense = self.to_dense()
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)
        if copy is False:
            return dense
        return np.array(dense, copy=True)

    def __getitem__(self, key):
        if self.is_dense:
            return next(iter(self.data.values()))[key]
        return self.to_dense()[key]

    def __iter__(self):
        if not self.is_dense:
            raise TypeError("iteration is only defined for dense-storage IrrepTensor objects.")
        return iter(next(iter(self.data.values())))

    def __setitem__(self, key, value):
        if not self.is_dense:
            raise TypeError(
                "item assignment is only available for dense-storage IrrepTensor objects."
            )
        next(iter(self.data.values()))[key] = value

    def astype(self, dtype, copy=True):
        if self._operator_mode:
            return type(self)(
                self.bra,
                self.ket,
                self.op,
                {
                    key: np.asarray(block).astype(dtype, copy=copy)
                    for key, block in self.data.items()
                },
            )
        return type(self)(
            {
                key: np.asarray(block).astype(dtype, copy=copy)
                for key, block in self.data.items()
            },
            self.legs,
            self.dirs,
            fusion_legs=self.fusion_legs,
            metadata=self.metadata.copy(),
        )

    def ravel(self, order="C"):
        return self.to_dense().ravel(order=order)

    def reshape(self, *shape, order="C"):
        """Dense NumPy view used by backend-agnostic numerical kernels."""
        return self.to_dense().reshape(*shape, order=order)

    def swapaxes(self, axis1, axis2):
        axes = list(range(self.rank))
        axes[int(axis1)], axes[int(axis2)] = axes[int(axis2)], axes[int(axis1)]
        return self.transpose(axes)

    def dot(self, other):
        if isinstance(other, IrrepTensor):
            total = 0.0
            for key, block in self.data.items():
                if key in other.data:
                    total += np.vdot(block, other.data[key])
            return total
        return np.vdot(self.to_dense().reshape(-1), np.asarray(other).reshape(-1))

    def norm(self):
        return float(np.sqrt(max(float(np.real(self.dot(self))), 0.0)))

    @classmethod
    def from_dense_data(cls, array, *, dirs=None, names=None, copy=True):
        """Wrap an arbitrary dense tensor in one trivial block per leg."""
        array = np.asarray(array)
        if array.ndim == 0:
            raise ValueError("IrrepTensor.from_dense_data requires rank >= 1.")
        if any(int(dim) < 1 for dim in array.shape):
            raise ValueError("dense tensor dimensions must be positive.")
        if dirs is None:
            dirs = (1,) * array.ndim
        dirs = tuple(int(direction) for direction in dirs)
        if len(dirs) != array.ndim:
            raise ValueError("dirs length must match the dense tensor rank.")
        if names is None:
            names = (None,) * array.ndim
        names = tuple(names)
        if len(names) != array.ndim:
            raise ValueError("names length must match the dense tensor rank.")
        legs = tuple(
            Leg.trivial(dim, direction=direction, name=name)
            for dim, direction, name in zip(array.shape, dirs, names)
        )
        return cls(
            {(None,) * array.ndim: np.array(array, copy=bool(copy))},
            legs,
            dirs,
        )

    def __repr__(self):
        return (
            f"IrrepTensor(rank={self.rank}, blocks={len(self.data)}, "
            f"nonabelian={self.has_nonabelian_symmetry})"
        )

    @classmethod
    def zeros(cls, bra: Leg, ket: Leg, op: OpIrrep):
        return cls(bra, ket, op, {})

    @classmethod
    def from_dense(
        cls,
        bra,
        ket,
        op,
        dense,
        *,
        drop_zeros=True,
        atol=0.0,
    ):
        dense = np.asarray(dense)
        if dense.shape != (bra.dim, ket.dim):
            raise ValueError(
                f"dense shape {dense.shape} does not match {(bra.dim, ket.dim)}"
            )
        blocks = {}
        bra_offsets = bra.offsets()
        ket_offsets = ket.offsets()
        for bra_irrep in bra.irreps:
            for ket_irrep in ket.irreps:
                if not bra.symmetry.allows(
                    bra_irrep.charge,
                    op.charge,
                    ket_irrep.charge,
                ):
                    continue
                block = dense[bra_offsets[bra_irrep], ket_offsets[ket_irrep]]
                if drop_zeros and not np.any(np.abs(block) > atol):
                    continue
                blocks[(bra_irrep, ket_irrep)] = block.copy()
        return cls(bra, ket, op, blocks)

    @classmethod
    def from_site_operator(
        cls,
        site,
        operator,
        *,
        leg=None,
        op_charge=None,
        atol=0.0,
    ):
        """Block a homogeneous-charge canonical local operator by irreps."""
        derived_leg = site.leg
        _symmetry, irreps, state_indices = _site_irrep_layout(site)
        if leg is None:
            leg = derived_leg
        elif leg != derived_leg:
            raise ValueError("leg is inconsistent with the canonical site.")
        matrix = site.operator(operator) if isinstance(operator, str) else np.asarray(operator)
        if matrix.shape != (site.dim, site.dim):
            raise ValueError(
                f"operator must have shape {(site.dim, site.dim)}, got {matrix.shape}."
            )
        if site.charges is None:
            state_charges = [None] * site.dim
            for irrep, indices in zip(irreps, state_indices):
                charge = irrep.charge if isinstance(irrep.charge, tuple) else (irrep.charge,)
                for index in indices:
                    state_charges[index] = tuple(int(value) for value in charge)
            state_charges = tuple(state_charges)
        else:
            state_charges = site.charges
        transfers = set()
        rows, columns = np.nonzero(np.abs(matrix) > float(atol))
        for row, column in zip(rows, columns):
            transfers.add(
                tuple(
                    int(out_value) - int(in_value)
                    for out_value, in_value in zip(
                        state_charges[int(row)],
                        state_charges[int(column)],
                    )
                )
            )
        if not transfers:
            raise ValueError("operator has no entries above atol.")
        if op_charge is None:
            if len(transfers) != 1:
                raise ValueError(
                    "operator contains multiple charge transfers; split it into "
                    "homogeneous irrep components."
                )
            normalized = next(iter(transfers))
        else:
            normalized = (
                tuple(int(value) for value in op_charge)
                if isinstance(op_charge, Iterable) and not isinstance(op_charge, (str, bytes))
                else (int(op_charge),)
            )
            if transfers != {normalized}:
                raise ValueError(
                    f"operator transfers {sorted(transfers)!r}, not {normalized!r}."
                )
        op_charge = normalized[0] if len(normalized) == 1 else normalized
        blocks = {}
        for bra_irrep, bra_indices in zip(leg.irreps, state_indices):
            for ket_irrep, ket_indices in zip(leg.irreps, state_indices):
                if not leg.symmetry.allows(
                    bra_irrep.charge,
                    op_charge,
                    ket_irrep.charge,
                ):
                    continue
                block = matrix[np.ix_(bra_indices, ket_indices)]
                if np.any(np.abs(block) > float(atol)):
                    blocks[(bra_irrep, ket_irrep)] = block.copy()
        return cls(leg, leg, OpIrrep(op_charge), blocks)

    @classmethod
    def identity(cls, leg):
        blocks = {
            (irrep, irrep): np.eye(dim)
            for irrep, dim in leg.dims.items()
            if dim
        }
        charge = next(iter(leg.dims)).charge
        zero = 0 if isinstance(charge, int) else tuple(0 for _ in charge)
        return cls(leg, leg, OpIrrep(zero), blocks)

    def adjoint(self):
        if not self._operator_mode:
            raise TypeError("adjoint is defined for rank-two operator tensors.")
        symmetry = self.bra.symmetry
        blocks = {
            (ket, bra): block.conj().T
            for (bra, ket), block in self.data.items()
        }
        return IrrepTensor(
            self.ket,
            self.bra,
            OpIrrep(symmetry.dual(self.op.charge)),
            blocks,
        )

    def scalar_matmul(self, other):
        if not self._operator_mode or not other._operator_mode:
            raise TypeError("scalar_matmul requires operator IrrepTensors.")
        scalar = self._zero_op_charge()
        if self.op.charge != scalar or other.op.charge != scalar:
            raise NotImplementedError(
                "generic non-scalar tensor recoupling is not implemented"
            )
        if self.ket != other.bra:
            raise ValueError("inner Leg mismatch")
        blocks = {}
        for (bra, mid), left in self.data.items():
            for (mid2, ket), right in other.data.items():
                if mid2 != mid:
                    continue
                key = (bra, ket)
                product = left @ right
                blocks[key] = product if key not in blocks else blocks[key] + product
        return IrrepTensor(self.bra, other.ket, OpIrrep(scalar), blocks)

    def _zero_op_charge(self):
        charge = self.op.charge
        return tuple(0 for _ in charge) if isinstance(charge, tuple) else 0


def u1_leg(charges_and_dims: Iterable[tuple[int, int]]) -> Leg:
    return Leg(
        {Irrep(int(q)): int(dim) for q, dim in charges_and_dims},
        symmetry=U1Symmetry(),
    )


def u1_su2_leg(sectors: Iterable[tuple[int, int, int]]) -> Leg:
    sym = ProductSymmetry((U1Symmetry("Ne"), SU2Symmetry("SU2")), name="U1xSU2")
    return Leg(
        {Irrep((int(ne), int(j2))): int(dim) for ne, j2, dim in sectors},
        symmetry=sym,
    )


def u1_su2_leg_from_spin(sectors: Iterable[tuple[int, object, int]]) -> Leg:
    """Build a U(1)xSU(2) site from physical ``(Ne, S, dim)`` sector labels."""
    sym = ProductSymmetry((U1Symmetry("Ne"), SU2Symmetry("SU2")), name="U1xSU2")
    return Leg(
        {u1_su2_irrep(ne, spin): int(dim) for ne, spin, dim in sectors},
        symmetry=sym,
    )


__all__ = [
    "Irrep",
    "Leg",
    "IrrepTensor",
    "OpIrrep",
    "ProductSymmetry",
    "SU2Symmetry",
    "Symmetry",
    "U1Symmetry",
    "spin_label",
    "spin_value",
    "twice_spin",
    "u1_leg",
    "u1_su2_irrep",
    "u1_su2_op_irrep",
    "u1_su2_leg",
    "u1_su2_leg_from_spin",
]
