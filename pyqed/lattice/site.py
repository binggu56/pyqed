"""Physical-index and local-site metadata for tensor-network states."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from pyqed.symmetry import Leg


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
