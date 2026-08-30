"""Structured Hamiltonians and analytical finite-state MPO construction."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from pyqed.lattice.site import Site
from .mpo import MPO


@dataclass(frozen=True)
class LocalTerm:
    r"""An operator acting on an ordered tuple of physical sites.

    ``operator`` uses product-basis ordering over ``sites``: its row index is
    the bra multi-index and its column index is the ket multi-index.
    Site tuples are required to be strictly increasing so that terms with the
    same support can be combined unambiguously.
    """

    sites: tuple[int, ...]
    operator: np.ndarray

    def __init__(self, sites, operator, *, coefficient=1.0):
        sites = tuple(int(site) for site in sites)
        if sites != tuple(sorted(set(sites))):
            raise ValueError("term sites must be unique and strictly increasing.")
        operator = np.asarray(operator) * coefficient
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("term operator must be a square matrix.")
        if np.any(~np.isfinite(operator)):
            raise ValueError("term operator must contain only finite values.")
        operator = np.array(operator, copy=True)
        operator.setflags(write=False)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "operator", operator)


def _matrix_signature(matrix):
    value = np.ascontiguousarray(matrix)
    return value.dtype.str, value.shape, value.tobytes()


@dataclass(frozen=True)
class OperatorString:
    """A coefficient times named single-site operators on distinct sites."""

    sites: tuple[int, ...]
    names: tuple[str, ...]
    operators: tuple[np.ndarray, ...]
    coefficient: complex

    @classmethod
    def from_sites(cls, physical_sites, site_operators, *, coefficient=1.0):
        entries = []
        for entry in site_operators:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "operator strings require entries such as (site, 'Sz')."
                )
            site, name = entry
            site = int(site)
            if site < 0 or site >= len(physical_sites):
                raise IndexError(f"operator-string site {site} is out of range.")
            if not isinstance(name, str):
                raise TypeError("operator-string operator names must be strings.")
            entries.append((site, name, physical_sites[site].operator(name)))
        if not entries:
            raise ValueError("an operator string must contain at least one operator.")
        entries.sort(key=lambda item: item[0])
        sites = tuple(entry[0] for entry in entries)
        if len(set(sites)) != len(sites):
            raise ValueError(
                "an operator string cannot contain the same site more than once."
            )

        coefficient = np.asarray(coefficient).item()
        if not np.isfinite(coefficient):
            raise ValueError("operator-string coefficient must be finite.")
        operators = []
        for _site, _name, operator in entries:
            value = np.array(operator, copy=True)
            value.setflags(write=False)
            operators.append(value)
        return cls(
            sites,
            tuple(entry[1] for entry in entries),
            tuple(operators),
            coefficient,
        )

    @property
    def signature(self):
        return tuple(
            (site, _matrix_signature(operator))
            for site, operator in zip(self.sites, self.operators)
        )

    @property
    def adjoint_signature(self):
        return tuple(
            (site, _matrix_signature(operator.T.conj()))
            for site, operator in zip(self.sites, self.operators)
        )

    def adjoint(self):
        operators = []
        for operator in self.operators:
            value = np.array(operator.T.conj(), copy=True)
            value.setflags(write=False)
            operators.append(value)
        return type(self)(
            self.sites,
            tuple(f"{name}†" for name in self.names),
            tuple(operators),
            np.conj(self.coefficient),
        )

    def to_local_term(self):
        operator = np.ones((1, 1), dtype=np.result_type(*self.operators))
        for local_operator in self.operators:
            operator = np.kron(operator, local_operator)
        return LocalTerm(self.sites, operator, coefficient=self.coefficient)


def _operator_string_mpo(physical_sites, products, dtype):
    """Compile analytical product strings into a shared-prefix automaton."""
    nsites = len(physical_sites)
    start = ("start",)
    final = ("final",)
    transitions = [dict() for _ in range(nsites)]

    def add_transition(site, left, right, operator, coefficient=1.0, *, accumulate):
        operator = np.asarray(operator, dtype=dtype)
        key = (left, right, _matrix_signature(operator))
        if key in transitions[site]:
            if accumulate:
                transitions[site][key][1] += coefficient
            return
        transitions[site][key] = [operator, coefficient]

    for product in products:
        current = start
        prefix = ()
        previous_site = None
        for position, (site, operator) in enumerate(
            zip(product.sites, product.operators)
        ):
            if previous_site is not None:
                for gap in range(previous_site + 1, site):
                    add_transition(
                        gap,
                        current,
                        current,
                        np.eye(physical_sites[gap].dim, dtype=dtype),
                        accumulate=False,
                    )

            terminal = position == len(product.sites) - 1
            if terminal:
                following = final
                coefficient = product.coefficient
            else:
                prefix += ((site, _matrix_signature(operator)),)
                following = ("prefix", prefix)
                coefficient = 1.0
            add_transition(
                site,
                current,
                following,
                operator,
                coefficient,
                accumulate=terminal,
            )
            current = following
            previous_site = site

    for site, physical_site in enumerate(physical_sites):
        identity = np.eye(physical_site.dim, dtype=dtype)
        add_transition(site, start, start, identity, accumulate=False)
        add_transition(site, final, final, identity, accumulate=False)

    records = []
    for site_transitions in transitions:
        records.append(
            tuple(
                (key[0], key[1], operator, coefficient)
                for key, (operator, coefficient) in site_transitions.items()
                if coefficient != 0
            )
        )

    reachable = [set() for _ in range(nsites + 1)]
    reachable[0].add(start)
    for site in range(nsites):
        for left, right, _operator, _coefficient in records[site]:
            if left in reachable[site]:
                reachable[site + 1].add(right)

    productive = [set() for _ in range(nsites + 1)]
    productive[-1].add(final)
    for site in range(nsites - 1, -1, -1):
        for left, right, _operator, _coefficient in records[site]:
            if right in productive[site + 1]:
                productive[site].add(left)

    if start not in productive[0]:
        tensors = [
            np.eye(site.dim, dtype=dtype)[None, None, :, :]
            for site in physical_sites
        ]
        tensors[0] = np.zeros_like(tensors[0])
        return MPO(tensors, sites=physical_sites)

    active = []
    for cut in range(nsites + 1):
        states = reachable[cut] & productive[cut]
        ordered = []
        for special in (start, final):
            if special in states:
                ordered.append(special)
        ordered.extend(sorted(states - {start, final}, key=repr))
        active.append(tuple(ordered))
    state_maps = tuple(
        {state: index for index, state in enumerate(states)}
        for states in active
    )

    tensors = []
    for site, physical_site in enumerate(physical_sites):
        tensor = np.zeros(
            (
                len(active[site]),
                len(active[site + 1]),
                physical_site.dim,
                physical_site.dim,
            ),
            dtype=dtype,
        )
        for left, right, operator, coefficient in records[site]:
            if left in state_maps[site] and right in state_maps[site + 1]:
                tensor[
                    state_maps[site][left],
                    state_maps[site + 1][right],
                ] += coefficient * operator
        tensors.append(tensor)
    return MPO(tensors, sites=physical_sites)



def _operator_tt_cores(sites, operator, dims):
    """Factor one finite-support operator into exact-to-roundoff TT cores."""
    support_dims = tuple(dims[site] for site in sites)
    nsites = len(sites)
    tensor = np.asarray(operator).reshape(support_dims + support_dims)
    interleaved = tuple(
        index
        for pair in zip(range(nsites), range(nsites, 2 * nsites))
        for index in pair
    )
    pair_dims = tuple(dim * dim for dim in support_dims)
    work = tensor.transpose(interleaved).reshape(pair_dims)
    cores = []
    previous_rank = 1
    for position, (site, pair_dim) in enumerate(zip(sites[:-1], pair_dims[:-1])):
        matrix = work.reshape(previous_rank * pair_dim, -1)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if singular_values.size:
            threshold = (
                np.finfo(singular_values.dtype).eps
                * max(matrix.shape)
                * singular_values[0]
            )
            rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        else:  # pragma: no cover - a positive physical dimension prevents this
            rank = 1
        left = left[:, :rank]
        singular_values = singular_values[:rank]
        right = right[:rank]
        dim = dims[site]
        cores.append(left.reshape(previous_rank, dim, dim, rank).transpose(0, 3, 1, 2))
        work = (singular_values[:, None] * right).reshape(
            (rank,) + pair_dims[position + 1 :]
        )
        previous_rank = rank
    final_dim = dims[sites[-1]]
    cores.append(work.reshape(previous_rank, 1, final_dim, final_dim))
    return tuple(cores)


class Hamiltonian:
    r"""A sum of finite-support operators without a many-body matrix.

    The representation stores

    .. math::

        H = c I + \sum_t h_t,

    and combines terms that have identical supports.  ``to_dense`` and
    ``to_sparse`` are explicit small-system reference helpers; frontier LETTA
    contractions consume the local terms directly.
    """

    def __init__(self, sites, terms=(), *, products=(), constant=0.0):
        """Build a structured Hamiltonian over canonical physical sites.

        ``sites`` should normally contain :class:`pyqed.lattice.Site`
        instances.  A sequence of integer dimensions remains accepted as a
        narrow migration path and is converted immediately to anonymous
        canonical sites.
        """
        values = tuple(sites)
        if not values:
            raise ValueError("sites must contain at least one physical site.")
        if all(isinstance(value, Site) for value in values):
            self.sites = values
        elif all(isinstance(value, Integral) for value in values):
            self.sites = tuple(Site(int(value)) for value in values)
        else:
            raise TypeError(
                "sites must contain only canonical Site objects or only integer "
                "local dimensions."
            )
        self.dims = tuple(site.dim for site in self.sites)

        constant = np.asarray(constant).item()
        if not np.isfinite(constant):
            raise ValueError("constant must be finite.")
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(constant))
        if abs(np.imag(constant)) > tolerance:
            raise ValueError("a Hermitian Hamiltonian requires a real constant.")
        self.constant = float(np.real(constant))

        grouped: dict[tuple[int, ...], np.ndarray] = {}
        for term in terms:
            if not isinstance(term, LocalTerm):
                try:
                    sites, operator = term
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "terms must contain LocalTerm objects or (sites, operator) pairs."
                    ) from error
                term = LocalTerm(sites, operator)
            if any(site < 0 or site >= len(self.dims) for site in term.sites):
                raise ValueError("term support contains an invalid site.")
            support_dim = int(np.prod([self.dims[site] for site in term.sites]))
            if term.operator.shape != (support_dim, support_dim):
                raise ValueError(
                    f"operator on sites {term.sites} must have shape "
                    f"{(support_dim, support_dim)}."
                )
            if not term.sites:
                self.constant += float(np.real(term.operator[0, 0]))
                if abs(np.imag(term.operator[0, 0])) > tolerance:
                    raise ValueError("an empty-support term must be real.")
                continue
            if term.sites in grouped:
                grouped[term.sites] = grouped[term.sites] + term.operator
            else:
                grouped[term.sites] = np.array(term.operator, copy=True)

        combined = []
        for sites in sorted(grouped):
            operator = grouped[sites]
            scale = max(float(np.linalg.norm(operator, ord=np.inf)), 1.0)
            if not np.allclose(
                operator,
                operator.T.conj(),
                rtol=0.0,
                atol=128.0 * np.finfo(float).eps * scale,
            ):
                raise ValueError(
                    f"combined operator on sites {sites} is not Hermitian."
                )
            combined.append(LocalTerm(sites, 0.5 * (operator + operator.T.conj())))
        self._local_terms = tuple(combined)
        self.products = ()
        self._materialized_terms = None
        self._products_validated = False
        for product in products:
            if not isinstance(product, OperatorString):
                raise TypeError("products must contain OperatorString objects.")
            self._validate_product(product)
            self.products += (product,)
        self._refresh_dtype()
        dimension = int(np.prod(self.dims))
        self.shape = (dimension, dimension)

    def _refresh_dtype(self):
        self.dtype = np.dtype(
            np.result_type(
                self.constant,
                *[term.operator.dtype for term in self._local_terms],
                *[
                    value
                    for product in self.products
                    for value in (
                        product.coefficient,
                        *product.operators,
                    )
                ],
                np.float64,
            )
        )

    def _validate_product(self, product):
        for site, operator in zip(product.sites, product.operators):
            if site < 0 or site >= len(self.sites):
                raise IndexError(f"operator-string site {site} is out of range.")
            expected = (self.dims[site], self.dims[site])
            if operator.shape != expected:
                raise ValueError(
                    f"operator-string matrix on site {site} must have shape {expected}."
                )

    def _validate_products_hermitian(self):
        if self._products_validated:
            return
        if not self.products:
            self._products_validated = True
            return
        totals = {}
        for product in self.products:
            totals[product.signature] = (
                totals.get(product.signature, 0.0) + product.coefficient
            )
        scale = max(
            1.0,
            *[abs(coefficient) for coefficient in totals.values()],
        )
        tolerance = 256.0 * np.finfo(float).eps * scale
        for product in self.products:
            coefficient = totals[product.signature]
            adjoint_coefficient = sum(
                candidate.coefficient
                for candidate in self.products
                if candidate.sites == product.sites
                and all(
                    np.allclose(
                        candidate_operator,
                        operator.T.conj(),
                        rtol=0.0,
                        atol=256.0
                        * np.finfo(float).eps
                        * max(1.0, float(np.linalg.norm(operator, ord=np.inf))),
                    )
                    for operator, candidate_operator in zip(
                        product.operators,
                        candidate.operators,
                    )
                )
            )
            if not np.allclose(
                adjoint_coefficient,
                np.conj(coefficient),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "analytical operator strings must form a Hermitian sum; "
                    "add the missing adjoint product."
                )
        self._products_validated = True

    def add_product(self, coefficient, *site_operators, add_hc=False):
        """Append an analytical product of named ``Site`` operators.

        Examples
        --------
        ``H.add_product(J, (i, "Sx"), (j, "Sx"))`` adds
        ``J * Sx_i * Sx_j`` without constructing a Kronecker-product matrix.
        """
        product = OperatorString.from_sites(
            self.sites,
            site_operators,
            coefficient=coefficient,
        )
        self._validate_product(product)
        additions = (product, product.adjoint()) if add_hc else (product,)
        self.products += additions
        self._materialized_terms = None
        self._products_validated = False
        self._refresh_dtype()
        return self

    @property
    def terms(self) -> tuple[LocalTerm, ...]:
        """Combined dense-support terms, materialized only when requested."""
        if self._materialized_terms is None:
            self._validate_products_hermitian()
            grouped = {
                term.sites: np.array(term.operator, copy=True)
                for term in self._local_terms
            }
            for product in self.products:
                term = product.to_local_term()
                grouped[term.sites] = (
                    grouped.get(term.sites, 0.0) + term.operator
                )
            combined = []
            for sites in sorted(grouped):
                operator = grouped[sites]
                scale = max(float(np.linalg.norm(operator, ord=np.inf)), 1.0)
                if not np.allclose(
                    operator,
                    operator.T.conj(),
                    rtol=0.0,
                    atol=128.0 * np.finfo(float).eps * scale,
                ):
                    raise ValueError(
                        f"combined operator on sites {sites} is not Hermitian."
                    )
                combined.append(
                    LocalTerm(sites, 0.5 * (operator + operator.T.conj()))
                )
            self._materialized_terms = tuple(combined)
        return self._materialized_terms

    @property
    def local_terms(self) -> tuple[LocalTerm, ...]:
        """Dense local kernels without materializing analytical products."""
        return self._local_terms

    @property
    def nterms(self) -> int:
        return len(self.terms)

    @property
    def nproducts(self) -> int:
        return len(self.products)

    @property
    def supports(self) -> tuple[tuple[int, ...], ...]:
        """Distinct term supports without materializing operator strings."""
        return tuple(
            sorted(
                {term.sites for term in self._local_terms}
                | {product.sites for product in self.products}
            )
        )

    def matvec(self, vector) -> np.ndarray:
        """Apply the local-term sum without materializing its full matrix."""
        vector = np.asarray(vector)
        dimension = self.shape[0]
        if vector.shape != (dimension,):
            raise ValueError(f"vector must have shape {(dimension,)}.")
        dtype = np.result_type(vector.dtype, self.dtype)
        state = np.asarray(vector, dtype=dtype).reshape(self.dims)
        result = self.constant * state
        all_sites = tuple(range(len(self.dims)))
        for term in self._local_terms:
            rest = tuple(site for site in all_sites if site not in term.sites)
            permutation = term.sites + rest
            inverse = np.argsort(permutation)
            support_dim = term.operator.shape[0]
            state_matrix = np.transpose(state, permutation).reshape(support_dim, -1)
            contribution = (term.operator @ state_matrix).reshape(
                tuple(self.dims[site] for site in permutation)
            )
            result = result + np.transpose(contribution, inverse)
        self._validate_products_hermitian()
        for product in self.products:
            contribution = state
            for site, operator in zip(product.sites, product.operators):
                contribution = np.tensordot(
                    operator,
                    contribution,
                    axes=(1, site),
                )
                contribution = np.moveaxis(contribution, 0, site)
            result = result + product.coefficient * contribution
        return np.asarray(result).reshape(-1)

    def __matmul__(self, vector):
        return self.matvec(vector)

    def expectation(self, vector, *, normalize=True):
        vector = np.asarray(vector)
        numerator = np.vdot(vector, self.matvec(vector))
        if not normalize:
            return numerator
        denominator = np.vdot(vector, vector)
        if abs(denominator) <= np.finfo(float).tiny:
            raise ValueError("cannot evaluate the energy of a zero vector.")
        return float(np.real(numerator / denominator))

    def to_mpo(
        self,
        *,
        minimize=False,
        max_rank=None,
        rtol=None,
        atol=None,
    ) -> MPO:
        """Build an exact finite-state MPO without a many-body matrix.

        Analytical operator strings are inserted directly into a shared-prefix
        automaton without Kronecker products or SVD. Dense ``LocalTerm`` kernels
        are TT-SVD-factorized only on their finite support and combined with the
        analytical MPO.
        """
        self._validate_products_hermitian()
        minimize = bool(minimize)
        if not minimize and any(
            value is not None for value in (max_rank, rtol, atol)
        ):
            raise ValueError(
                "max_rank/rtol/atol require minimize=True when building an MPO."
            )

        def finalized(mpo):
            return (
                mpo.compress(max_rank=max_rank, rtol=rtol, atol=atol)
                if minimize
                else mpo
            )
        paths = []
        if self.constant != 0.0:
            paths.append(
                (
                    (0,),
                    (
                        (self.constant * np.eye(self.dims[0], dtype=self.dtype))[
                            None, None, :, :
                        ],
                    ),
                )
            )
        for term in self._local_terms:
            if np.linalg.norm(term.operator) == 0.0:
                continue
            paths.append(
                (
                    term.sites,
                    _operator_tt_cores(term.sites, term.operator, self.dims),
                )
            )

        if not paths:
            if self.products:
                return finalized(
                    _operator_string_mpo(
                        self.sites,
                        self.products,
                        self.dtype,
                    )
                )
            tensors = []
            for site, dim in enumerate(self.dims):
                tensor = np.eye(dim, dtype=self.dtype)[None, None, :, :]
                if site == 0:
                    tensor = np.zeros_like(tensor)
                tensors.append(tensor)
            return finalized(MPO(tensors, sites=self.sites))

        nsites = len(self.dims)
        states = [None] * (nsites + 1)
        states[0] = (("idle",),)
        states[-1] = (("done",),)
        for cut in range(1, nsites):
            cut_states = [("idle",), ("done",)]
            for term_index, (sites, cores) in enumerate(paths):
                for stage in range(len(sites) - 1):
                    if sites[stage] < cut <= sites[stage + 1]:
                        rank = cores[stage].shape[1]
                        cut_states.extend(
                            ("active", term_index, stage, channel)
                            for channel in range(rank)
                        )
            states[cut] = tuple(cut_states)
        state_maps = tuple(
            {state: index for index, state in enumerate(cut_states)}
            for cut_states in states
        )

        tensors = []
        for site, dim in enumerate(self.dims):
            left_states = state_maps[site]
            right_states = state_maps[site + 1]
            tensor = np.zeros(
                (len(left_states), len(right_states), dim, dim),
                dtype=self.dtype,
            )
            identity = np.eye(dim, dtype=self.dtype)
            idle = ("idle",)
            done = ("done",)
            if idle in left_states and idle in right_states:
                tensor[left_states[idle], right_states[idle]] += identity
            if done in left_states and done in right_states:
                tensor[left_states[done], right_states[done]] += identity

            for term_index, (sites, cores) in enumerate(paths):
                if len(sites) == 1:
                    if site == sites[0]:
                        tensor[left_states[idle], right_states[done]] += cores[0][0, 0]
                    continue
                if site == sites[0]:
                    core = cores[0]
                    for right_channel in range(core.shape[1]):
                        target = ("active", term_index, 0, right_channel)
                        tensor[left_states[idle], right_states[target]] += core[
                            0, right_channel
                        ]
                    continue
                if site == sites[-1]:
                    core = cores[-1]
                    stage = len(sites) - 2
                    for left_channel in range(core.shape[0]):
                        source = ("active", term_index, stage, left_channel)
                        tensor[left_states[source], right_states[done]] += core[
                            left_channel, 0
                        ]
                    continue
                if site in sites[1:-1]:
                    position = sites.index(site)
                    core = cores[position]
                    for left_channel in range(core.shape[0]):
                        source = (
                            "active",
                            term_index,
                            position - 1,
                            left_channel,
                        )
                        for right_channel in range(core.shape[1]):
                            target = (
                                "active",
                                term_index,
                                position,
                                right_channel,
                            )
                            tensor[left_states[source], right_states[target]] += core[
                                left_channel, right_channel
                            ]
                    continue
                for stage, (left_site, right_site) in enumerate(
                    zip(sites[:-1], sites[1:])
                ):
                    if left_site < site < right_site:
                        rank = cores[stage].shape[1]
                        for channel in range(rank):
                            active = (
                                "active",
                                term_index,
                                stage,
                                channel,
                            )
                            tensor[
                                left_states[active], right_states[active]
                            ] += identity
                        break
            tensors.append(tensor)
        local_mpo = MPO(tensors, sites=self.sites)
        if self.products:
            local_mpo = local_mpo + _operator_string_mpo(
                self.sites,
                self.products,
                self.dtype,
            )
        return finalized(local_mpo)

    def to_dense(self) -> np.ndarray:
        """Materialize a dense reference matrix; exponential in site count."""
        dimension = self.shape[0]
        matrix = np.empty((dimension, dimension), dtype=self.dtype)
        basis_vector = np.zeros(dimension, dtype=self.dtype)
        for column in range(dimension):
            basis_vector[column] = 1
            matrix[:, column] = self.matvec(basis_vector)
            basis_vector[column] = 0
        return matrix

    def to_sparse(self) -> csr_matrix:
        """Materialize a CSR reference matrix; exponential in site count."""
        return csr_matrix(self.to_dense())


# Kept as a module-level migration alias for existing internal callers.  New
# public code should use the shorter canonical name.
LocalHamiltonian = Hamiltonian


__all__ = ["Hamiltonian", "LocalTerm", "OperatorString"]
