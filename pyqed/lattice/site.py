"""Canonical local Hilbert-space descriptors.

The classes in this module describe physical sites independently of a tensor
network backend.  A site owns its basis, local operators, and optional
per-basis-state Abelian charges.  :class:`CompositeSite` additionally retains
the factorization of a fused local Hilbert space.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from pyqed.symmetry import Leg


def _readonly_matrix(value, dim, *, name):
    matrix = np.asarray(value)
    if matrix.shape != (dim, dim):
        raise ValueError(
            f"operator {name!r} must have shape {(dim, dim)}, got {matrix.shape}."
        )
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"operator {name!r} must contain only finite values.")
    matrix = np.array(matrix, copy=True)
    matrix.setflags(write=False)
    return matrix


def _charge(value):
    if isinstance(value, Integral):
        return (int(value),)
    try:
        result = tuple(int(component) for component in value)
    except TypeError as error:
        raise TypeError("a charge must be an integer or an iterable of integers.") from error
    if not result:
        raise ValueError("charge tuples cannot be empty.")
    return result


def _fermion_parities_from_leg(leg, dim):
    """Infer basis-state parity from a Leg carrying particle number."""
    from pyqed.symmetry import ProductSymmetry, U1Symmetry

    symmetry = leg.symmetry
    if isinstance(symmetry, U1Symmetry):
        particle_component = 0
    elif isinstance(symmetry, ProductSymmetry):
        particle_component = None
        particle_names = {
            "n",
            "ne",
            "n_e",
            "number",
            "particle_number",
            "electron_number",
        }
        for index, factor in enumerate(symmetry.factors):
            if (
                isinstance(factor, U1Symmetry)
                and str(factor.name).lower() in particle_names
            ):
                particle_component = index
                break
        if particle_component is None:
            return None
    else:
        return None

    reduced = []
    full = []
    for sector in leg.irreps:
        charge = sector.charge
        components = charge if isinstance(charge, tuple) else (charge,)
        parity = int(components[particle_component]) % 2
        reduced.extend((parity,) * leg.sector_dim(sector))
        full.extend((parity,) * leg.sector_full_dim(sector))
    if len(reduced) == dim:
        return tuple(reduced)
    if len(full) == dim:
        return tuple(full)
    return None


class Site:
    """Immutable description of one local physical Hilbert space.

    Parameters
    ----------
    dim
        Local Hilbert-space dimension.  It may be omitted when ``labels`` are
        supplied.
    labels
        Basis-state labels in tensor-index order.
    basis
        Alias for ``labels``.
    leg
        Symmetry-sector decomposition of the physical tensor index.  When it
        is omitted, it is derived from ``charges`` or created as a dense leg.
    operators
        Named local matrices.  An identity named ``"I"`` is inserted when it
        is not supplied.
    charges
        Optional additive Abelian charge of every basis state.  Scalar charges
        are normalized to one-component tuples.
    charge_labels
        Names of the additive charge components, such as ``("n", "2sz")``.
    parities
        Fermion parity of every basis state, encoded as zero or one.  Bosonic
        sites default to all-even parity.
    statistics
        Either ``"bosonic"`` or ``"fermionic"``.  A fermionic site can infer
        parity from a particle-number charge or from ``leg``.
    """

    __slots__ = (
        "_labels",
        "_operators",
        "_charges",
        "_charge_labels",
        "_parities",
        "_leg",
        "_statistics",
        "_name",
    )

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} objects are immutable.")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        dim=None,
        *,
        labels=None,
        basis=None,
        leg=None,
        operators: Mapping[str, np.ndarray] | None = None,
        charges=None,
        charge_labels=None,
        parities=None,
        statistics=None,
        name: str | None = None,
    ):
        if basis is not None:
            if labels is not None:
                raise TypeError("specify basis or labels, not both.")
            labels = basis
        if labels is None:
            if dim is None:
                raise TypeError("Site requires dim or labels.")
            dim = int(dim)
            if dim < 1:
                raise ValueError("site dimension must be positive.")
            labels = tuple(str(index) for index in range(dim))
        else:
            labels = tuple(str(label) for label in labels)
            if not labels:
                raise ValueError("a site must contain at least one basis state.")
            if len(set(labels)) != len(labels):
                raise ValueError("site basis labels must be unique.")
            if dim is not None and int(dim) != len(labels):
                raise ValueError("dim does not match the number of basis labels.")
            dim = len(labels)

        from pyqed.symmetry import Leg

        if leg is not None:
            if not isinstance(leg, Leg):
                raise TypeError("leg must be a pyqed.symmetry.Leg.")
            if leg.full_dim != dim and leg.dim != dim:
                raise ValueError(
                    f"leg dimension ({leg.full_dim}) does not match basis dimension ({dim})."
                )

        matrices = {}
        for operator_name, value in dict(operators or {}).items():
            operator_name = str(operator_name)
            if not operator_name:
                raise ValueError("operator names cannot be empty.")
            matrices[operator_name] = _readonly_matrix(
                value,
                dim,
                name=operator_name,
            )
        if "I" not in matrices:
            matrices["I"] = _readonly_matrix(np.eye(dim), dim, name="I")

        if charges is None:
            if charge_labels is not None:
                raise ValueError("charge_labels require per-basis-state charges.")
            normalized_charges = None
            normalized_charge_labels = None
        else:
            normalized_charges = tuple(_charge(value) for value in charges)
            if len(normalized_charges) != dim:
                raise ValueError("charges must contain one entry per basis state.")
            rank = len(normalized_charges[0])
            if any(len(value) != rank for value in normalized_charges):
                raise ValueError("all site charges must have the same rank.")
            if charge_labels is None:
                normalized_charge_labels = tuple(f"q{index}" for index in range(rank))
            else:
                normalized_charge_labels = tuple(
                    str(label).lower() for label in charge_labels
                )
                if len(normalized_charge_labels) != rank:
                    raise ValueError("charge_labels must match the site charge rank.")
                if len(set(normalized_charge_labels)) != rank:
                    raise ValueError("charge_labels must be unique.")

        if statistics is None:
            normalized_statistics = None
        else:
            normalized_statistics = str(statistics).lower()
            if normalized_statistics not in {"bosonic", "fermionic"}:
                raise ValueError("statistics must be 'bosonic' or 'fermionic'.")

        if parities is None and normalized_statistics == "fermionic":
            parity_component = None
            if normalized_charge_labels is not None:
                for label in ("n", "ne", "number", "particle_number"):
                    if label in normalized_charge_labels:
                        parity_component = normalized_charge_labels.index(label)
                        break
            inferred_parities = (
                _fermion_parities_from_leg(leg, dim)
                if parity_component is None and leg is not None
                else None
            )
            if parity_component is None and inferred_parities is None:
                raise ValueError(
                    "a fermionic Site requires parities or a particle-number charge."
                )
            if inferred_parities is not None:
                normalized_parities = inferred_parities
            else:
                normalized_parities = tuple(
                    int(charge[parity_component]) % 2
                    for charge in normalized_charges
                )
        elif parities is None:
            normalized_parities = (0,) * dim
        else:
            normalized_parities = tuple(int(value) for value in parities)
            if len(normalized_parities) != dim:
                raise ValueError("parities must contain one entry per basis state.")
            if any(value not in (0, 1) for value in normalized_parities):
                raise ValueError("fermion parities must be zero or one.")
        if normalized_statistics is None:
            normalized_statistics = (
                "fermionic" if any(normalized_parities) else "bosonic"
            )
        if normalized_statistics == "bosonic" and any(normalized_parities):
            raise ValueError("a bosonic Site cannot contain odd-parity basis states.")

        self._labels = labels
        self._operators = MappingProxyType(matrices)
        self._charges = normalized_charges
        self._charge_labels = normalized_charge_labels
        self._parities = normalized_parities
        self._statistics = normalized_statistics
        self._name = type(self).__name__ if name is None else str(name)

        if leg is None:
            leg = (
                Leg.from_site(self)
                if normalized_charges is not None
                else Leg.from_dims({None: dim})
            )
        self._leg = leg

    @property
    def name(self):
        return self._name

    @property
    def labels(self):
        return self._labels

    @property
    def basis(self):
        return self._labels

    @property
    def dim(self):
        return len(self._labels)

    @property
    def d(self):
        """Short dimension alias used by existing tensor-network code."""
        return self.dim

    @property
    def operators(self):
        return self._operators

    @property
    def charges(self):
        return self._charges

    @property
    def charge_labels(self):
        return self._charge_labels

    @property
    def parities(self):
        return self._parities

    @property
    def leg(self):
        return self._leg

    @property
    def statistics(self):
        return self._statistics

    def operator(self, name):
        """Return a named read-only local operator."""
        try:
            return self._operators[str(name)]
        except KeyError as error:
            raise KeyError(f"site {self.name!r} has no operator {name!r}.") from error

    def __repr__(self):
        symmetry = "" if self.charges is None else f", charge_rank={len(self.charges[0])}"
        return f"{type(self).__name__}(dim={self.dim}{symmetry})"


class CompositeSite(Site):
    """Tensor product of sites with its microscopic factorization retained."""

    __slots__ = ("_factors", "_factor_dims")

    def __init__(self, factors: Sequence[Site], *, name: str | None = None):
        factors = tuple(factors)
        if not factors:
            raise ValueError("CompositeSite requires at least one factor.")
        if any(not isinstance(site, Site) for site in factors):
            raise TypeError("all CompositeSite factors must be canonical Site objects.")

        factor_dims = tuple(site.dim for site in factors)
        configurations = tuple(product(*(site.labels for site in factors)))
        labels = tuple("|".join(configuration) for configuration in configurations)

        if all(site.charges is not None for site in factors):
            ranks = {len(site.charges[0]) for site in factors}
            if len(ranks) != 1:
                raise ValueError("all composite factors must use the same charge rank.")
            charge_labels = {site.charge_labels for site in factors}
            if len(charge_labels) != 1:
                raise ValueError("all composite factors must use the same charge labels.")
            charges = tuple(
                tuple(sum(parts) for parts in zip(*configuration))
                for configuration in product(*(site.charges for site in factors))
            )
        else:
            charges = None
            charge_labels = {None}

        parities = tuple(
            sum(configuration) % 2
            for configuration in product(*(site.parities for site in factors))
        )
        self._factors = factors
        self._factor_dims = factor_dims
        super().__init__(
            labels=labels,
            charges=charges,
            charge_labels=next(iter(charge_labels)),
            parities=parities,
            name=name or " x ".join(site.name for site in factors),
        )

    @property
    def factors(self):
        return self._factors

    @property
    def factor_dims(self):
        return self._factor_dims

    def flatten(self, states):
        """Map factor-state indices to the composite basis index."""
        states = tuple(int(state) for state in states)
        if len(states) != len(self.factor_dims):
            raise ValueError("states must contain one index per composite factor.")
        if any(state < 0 or state >= dim for state, dim in zip(states, self.factor_dims)):
            raise ValueError("a factor-state index is outside its local basis.")
        return int(np.ravel_multi_index(states, self.factor_dims))

    def unflatten(self, state):
        """Map one composite basis index to factor-state indices."""
        state = int(state)
        if state < 0 or state >= self.dim:
            raise ValueError("composite state index is outside the local basis.")
        return tuple(int(value) for value in np.unravel_index(state, self.factor_dims))

    def product_operator(self, operators):
        """Embed a product of factor-local operators.

        ``operators`` may be a mapping from factor index to an operator name
        or matrix, or a sequence containing one entry per factor.  ``None``
        selects the identity on a factor.
        """
        if isinstance(operators, Mapping):
            entries = tuple(operators.get(index) for index in range(len(self.factors)))
        else:
            entries = tuple(operators)
            if len(entries) != len(self.factors):
                raise ValueError("operators must contain one entry per factor.")

        result = np.ones((1, 1))
        for site, entry in zip(self.factors, entries):
            if entry is None:
                matrix = site.operator("I")
            elif isinstance(entry, str):
                matrix = site.operator(entry)
            else:
                matrix = _readonly_matrix(entry, site.dim, name="<anonymous>")
            result = np.kron(result, matrix)
        result.setflags(write=False)
        return result

    def operator_on(self, factor, operator):
        """Embed one named or explicit operator on a selected factor."""
        factor = int(factor)
        if factor < 0 or factor >= len(self.factors):
            raise IndexError("composite factor index is out of range.")
        return self.product_operator({factor: operator})


class SpinHalfSite(Site):
    """Spin-1/2 site in the ``up, down`` basis with charge ``2*Sz``."""

    __slots__ = ()

    def __init__(self):
        identity = np.eye(2)
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        z = np.diag([1.0, -1.0])
        plus = np.array([[0.0, 1.0], [0.0, 0.0]])
        minus = plus.T
        super().__init__(
            labels=("up", "down"),
            operators={
                "I": identity,
                "X": x,
                "Y": y,
                "Z": z,
                "Sx": 0.5 * x,
                "Sy": 0.5 * y,
                "Sz": 0.5 * z,
                "Sp": plus,
                "Sm": minus,
            },
            charges=(1, -1),
            charge_labels=("2sz",),
            name="spin-1/2",
        )


class SpinlessFermionSite(Site):
    """Empty/occupied spinless-fermion site."""

    __slots__ = ()

    def __init__(self):
        annihilation = np.array([[0.0, 1.0], [0.0, 0.0]])
        number = np.diag([0.0, 1.0])
        super().__init__(
            labels=("empty", "occupied"),
            operators={
                "c": annihilation,
                "cdag": annihilation.T,
                "n": number,
                "parity": np.diag([1.0, -1.0]),
            },
            charges=(0, 1),
            charge_labels=("n",),
            parities=(0, 1),
            name="spinless fermion",
        )


class SpinHalfFermionSite(Site):
    """Four-state spinful-fermion site with ``(N, 2*Sz)`` charges."""

    __slots__ = ()

    def __init__(self):
        n_up_diag = np.array([0.0, 1.0, 0.0, 1.0])
        n_down_diag = np.array([0.0, 0.0, 1.0, 1.0])
        c_up = np.zeros((4, 4))
        c_up[0, 1] = c_up[2, 3] = 1.0
        c_down = np.zeros((4, 4))
        c_down[0, 2] = 1.0
        c_down[1, 3] = -1.0
        create_up = c_up.T
        create_down = c_down.T
        spin_plus = create_up @ c_down
        spin_minus = create_down @ c_up
        super().__init__(
            labels=("empty", "up", "down", "double"),
            operators={
                "Cu": c_up,
                "Cdu": create_up,
                "Cd": c_down,
                "Cdd": create_down,
                "Nu": np.diag(n_up_diag),
                "Nd": np.diag(n_down_diag),
                "N": np.diag(n_up_diag + n_down_diag),
                "Ntot": np.diag(n_up_diag + n_down_diag),
                "double": np.diag(n_up_diag * n_down_diag),
                "NuNd": np.diag(n_up_diag * n_down_diag),
                "parity": np.diag((-1.0) ** (n_up_diag + n_down_diag)),
                "JW": np.diag((-1.0) ** (n_up_diag + n_down_diag)),
                "Sz": np.diag(0.5 * (n_up_diag - n_down_diag)),
                "Sp": spin_plus,
                "Sm": spin_minus,
                "Sx": 0.5 * (spin_plus + spin_minus),
                "Sy": -0.5j * (spin_plus - spin_minus),
            },
            charges=((0, 0), (1, 1), (1, -1), (2, 0)),
            charge_labels=("n", "2sz"),
            parities=(0, 1, 1, 0),
            name="spin-1/2 fermion",
        )


class BosonSite(Site):
    """Truncated bosonic site containing occupations ``0`` through ``nmax``."""

    __slots__ = ()

    def __init__(self, nmax):
        nmax = int(nmax)
        if nmax < 0:
            raise ValueError("nmax must be nonnegative.")
        dim = nmax + 1
        annihilation = np.diag(np.sqrt(np.arange(1, dim, dtype=float)), 1)
        super().__init__(
            labels=tuple(str(number) for number in range(dim)),
            operators={
                "b": annihilation,
                "bdag": annihilation.T,
                "n": np.diag(np.arange(dim, dtype=float)),
            },
            charges=tuple(range(dim)),
            charge_labels=("n",),
            name=f"boson(nmax={nmax})",
        )


__all__ = [
    "BosonSite",
    "CompositeSite",
    "Site",
    "SpinHalfFermionSite",
    "SpinHalfSite",
    "SpinlessFermionSite",
]
