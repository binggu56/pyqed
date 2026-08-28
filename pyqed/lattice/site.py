"""Physical-index and local-site metadata for tensor-network states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Leg:
    """Metadata for one physical tensor index.

    Dense legs need only ``dim``.  Symmetry-reduced legs additionally carry an
    ordered sector decomposition.  Operators deliberately live on :class:`Site`:
    a leg describes an index, not an algebra acting on that index.
    """

    dim: int | None = None
    labels: tuple[str, ...] = ()
    local_charges: tuple[tuple[int, ...], ...] = ()
    sectors: tuple[Any, ...] = ()
    sector_dims: Mapping[Any, int] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        sectors = tuple(self.sectors)
        sector_dims = {
            sector: int(size) for sector, size in dict(self.sector_dims).items()
        }
        if sectors:
            if len(set(sectors)) != len(sectors):
                raise ValueError("Leg sectors must be unique and ordered.")
            missing = [sector for sector in sectors if sector not in sector_dims]
            if missing:
                raise ValueError(f"Missing dimensions for leg sectors {missing!r}.")
            extra = [sector for sector in sector_dims if sector not in sectors]
            if extra:
                raise ValueError(f"Dimensions supplied for undeclared sectors {extra!r}.")
            if any(sector_dims[sector] < 1 for sector in sectors):
                raise ValueError("Every leg sector dimension must be positive.")
            total_dim = sum(sector_dims[sector] for sector in sectors)
            if self.dim is not None and int(self.dim) != total_dim:
                raise ValueError(
                    f"Leg dimension {self.dim} does not match sector total {total_dim}."
                )
            dim = total_dim
        else:
            if sector_dims:
                raise ValueError("sector_dims requires an ordered sectors tuple.")
            if self.dim is None or int(self.dim) < 1:
                raise ValueError("Leg dimension must be positive.")
            dim = int(self.dim)

        labels = tuple(str(label) for label in self.labels)
        if labels and len(labels) != dim:
            raise ValueError("The number of leg labels must match its dimension.")
        local_charges = tuple(
            tuple(int(component) for component in charge)
            for charge in self.local_charges
        )
        if local_charges and len(local_charges) != dim:
            raise ValueError("A leg must provide one local charge per basis state.")
        if local_charges:
            rank = len(local_charges[0])
            if any(len(charge) != rank for charge in local_charges):
                raise ValueError("All local charges on a leg must have the same rank.")

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "local_charges", local_charges)
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "sector_dims", sector_dims)

    @property
    def total_dim(self) -> int:
        return int(self.dim)

    def sector_dim(self, sector) -> int:
        """Return the reduced multiplicity of ``sector``."""
        if not self.sectors:
            raise ValueError("This dense leg has no sector decomposition.")
        return self.sector_dims[sector]

    def slices(self) -> dict[Any, slice]:
        """Return dense slices for the ordered sector decomposition."""
        offset = 0
        result = {}
        for sector in self.sectors:
            size = self.sector_dims[sector]
            result[sector] = slice(offset, offset + size)
            offset += size
        return result

    @classmethod
    def from_slices(cls, sector_slices, *, name=None) -> "Leg":
        return cls.from_dims(
            {
                sector: int(slice_.stop - slice_.start)
                for sector, slice_ in sector_slices.items()
            },
            sectors=tuple(sector_slices),
            name=name,
        )

    @classmethod
    def from_dims(cls, sector_dims, sectors=None, *, name=None) -> "Leg":
        if sectors is None:
            sectors = tuple(sector_dims)
        sectors = tuple(sectors)
        return cls(
            sectors=sectors,
            sector_dims={sector: int(sector_dims[sector]) for sector in sectors},
            name=name,
        )


class Site:
    """A local Hilbert space: one physical :class:`Leg` and its operators."""

    def __init__(
        self,
        leg: Leg,
        *,
        operators: Mapping[str, Any] | None = None,
        fermionic: bool = False,
        jw_metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(leg, Leg):
            raise TypeError("site leg must be a Leg.")
        self.leg = leg
        self.operators = {} if operators is None else dict(operators)
        self.fermionic = bool(fermionic)
        self._jw_metadata = {} if jw_metadata is None else dict(jw_metadata)
        self.dim = self.d = int(leg.dim)
        self.dimensions = (self.dim,)
        self.basis_labels = leg.labels or tuple(str(index) for index in range(self.dim))

    @classmethod
    def generic(cls, dim: int, *, labels: Sequence[str] = ()) -> "Site":
        """Create an anonymous dense site with an identity operator."""
        leg = Leg(dim=int(dim), labels=tuple(labels))
        return cls(leg, operators={"I": np.eye(leg.dim)})

    @classmethod
    def spinful_fermion(cls, *, include_jw: bool = True) -> "Site":
        from pyqed import SpinHalfFermionOperators

        operators = SpinHalfFermionOperators()
        jw_metadata = (
            {
                "JW_operator_names": ("JW", "JWu", "JWd"),
                "ordered_states": ("empty", "up", "down", "full"),
                "parity_phase_name": "JW",
            }
            if include_jw
            else {}
        )
        return cls(
            Leg(
                dim=4,
                labels=("empty", "up", "down", "full"),
                local_charges=((0, 0), (1, 1), (1, -1), (2, 0)),
            ),
            operators=operators,
            fermionic=True,
            jw_metadata=jw_metadata,
        )

    @classmethod
    def spin_half(cls) -> "Site":
        from pyqed import pauli

        identity, x, y, z = pauli()
        return cls(
            Leg(
                dim=2,
                labels=("up", "down"),
                local_charges=((0,), (1,)),
            ),
            operators={
                "I": identity,
                "X": x,
                "Y": y,
                "Z": z,
                "Sx": x / 2,
                "Sy": y / 2,
                "Sz": z / 2,
            },
        )

    @classmethod
    def spinless_fermion(cls) -> "Site":
        return cls(
            Leg(
                dim=2,
                labels=("empty", "occupied"),
                local_charges=((0,), (1,)),
            ),
            operators={"I": np.eye(2)},
            fermionic=True,
        )

    def add_operator(self, operator_name, operator=None):
        """Register a local operator, defaulting to a zero matrix."""
        name = str(operator_name)
        if name in self.operators:
            raise ValueError(f"Operator {name!r} already exists.")
        if operator is None:
            operator = np.zeros((self.dim, self.dim))
        operator = np.asarray(operator)
        if operator.shape != (self.dim, self.dim):
            raise ValueError(
                f"Operator {name!r} has shape {operator.shape}; expected {(self.dim, self.dim)}."
            )
        self.operators[name] = operator
        return self

    @property
    def local_charges(self):
        return self.leg.local_charges

    @property
    def jw_metadata(self):
        return dict(self._jw_metadata)

    @property
    def is_fermionic(self) -> bool:
        return self.fermionic


def normalize_sites(sites, dims) -> tuple[Site, ...]:
    """Return validated ordered sites, inferring anonymous sites when absent."""
    dims = tuple(int(dim) for dim in dims)
    if sites is None:
        return tuple(Site.generic(dim) for dim in dims)
    sites = tuple(Site(site) if isinstance(site, Leg) else site for site in sites)
    if len(sites) != len(dims):
        raise ValueError("The number of sites must match the tensor chain length.")
    if any(not isinstance(site, Site) for site in sites):
        raise TypeError("sites must contain Site or Leg objects.")
    for index, (site, dim) in enumerate(zip(sites, dims)):
        if site.dim != dim:
            raise ValueError(
                f"Site {index} dimension {site.dim} does not match tensor dimension {dim}."
            )
    return sites


def legs_compatible(left: Leg, right: Leg) -> bool:
    """Whether two legs describe compatible bases.

    Empty metadata is treated as unspecified, so anonymous dense legs compose
    with richer site descriptions of the same dimension.
    """
    if left.dim != right.dim:
        return False
    for lhs, rhs in (
        (left.labels, right.labels),
        (left.local_charges, right.local_charges),
        (left.sectors, right.sectors),
    ):
        if lhs and rhs and lhs != rhs:
            return False
    if left.sector_dims and right.sector_dims and left.sector_dims != right.sector_dims:
        return False
    return True


def sites_compatible(left, right) -> bool:
    left = tuple(left)
    right = tuple(right)
    return len(left) == len(right) and all(
        legs_compatible(lhs.leg, rhs.leg) for lhs, rhs in zip(left, right)
    )


def richer_sites(primary, fallback) -> tuple[Site, ...]:
    """Choose the richer compatible site descriptor at each position."""
    if not sites_compatible(primary, fallback):
        raise ValueError("Tensor-network operands use incompatible ordered sites.")
    result = []
    for first, second in zip(primary, fallback):
        def metadata_score(site):
            return (
                4 * bool(site.leg.labels)
                + 4 * bool(site.leg.local_charges)
                + 4 * bool(site.leg.sectors)
                + max(0, len(site.operators) - 1)
                + 2 * bool(site.jw_metadata)
            )

        first_score = metadata_score(first)
        second_score = metadata_score(second)
        result.append(first if first_score >= second_score else second)
    return tuple(result)


__all__ = [
    "Leg",
    "Site",
    "legs_compatible",
    "normalize_sites",
    "richer_sites",
    "sites_compatible",
]
