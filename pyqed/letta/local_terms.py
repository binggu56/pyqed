"""Structured local Hamiltonians for graph-tied LETTA contractions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


def _is_site_like(value) -> bool:
    return hasattr(value, "dim") or hasattr(value, "d")


def _normalize_sites(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (int, np.integer)):
        return None
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError("sites must be a sequence.") from error

    if not values:
        return ()
    if not all(_is_site_like(item) for item in values):
        return None
    normalized = []
    for site in values:
        if hasattr(site, "dim"):
            dim = int(site.dim)
        else:
            dim = int(site.d)
        if dim < 1:
            raise ValueError("site dimensions must be positive.")
        normalized.append(site)
    return tuple(normalized)


def _normalize_site_charges(site_charges, *, site_index: int, dim: int):
    normalized = tuple(_as_charge_tuple(charge) for charge in site_charges)
    if len(normalized) != dim:
        raise ValueError(
            f"site {site_index} charges must contain one entry per local basis state."
        )
    if normalized:
        rank = len(normalized[0])
        if any(len(charge) != rank for charge in normalized):
            raise ValueError(
                f"site {site_index} local charges must all have the same rank."
            )
    return normalized


def local_charges_from_sites(sites, *, require: bool = False):
    """Return local charges from site metadata when available.

    ``require=False`` returns ``None`` when no complete metadata is present.
    """
    if sites is None:
        if require:
            raise ValueError(
                "site-wise local charges are required but no sites are attached."
            )
        return None
    sites = tuple(sites)
    if not sites:
        if require:
            raise ValueError("site-wise local charges are required for non-empty systems.")
        return ()

    collected = []
    for index, site in enumerate(sites):
        raw_charges = None
        for candidate in (
            getattr(site, "local_charges", None),
            getattr(site, "charges", None),
        ):
            if candidate is not None:
                raw_charges = candidate
                break
        dim = int(getattr(site, "dim", getattr(site, "d")))
        if raw_charges is None:
            if require:
                raise ValueError(
                    f"site {index} has no local charge metadata."
                )
            return None
        collected.append(_normalize_site_charges(raw_charges, site_index=index, dim=dim))
    return tuple(collected)


def _normalized_dims(dims):
    if dims is None:
        raise ValueError("dims is required when sites are not provided.")
    if isinstance(dims, (int, np.integer)):
        dims = (dims,)
    try:
        dims = tuple(int(value) for value in dims)
    except TypeError as error:
        raise TypeError("dims must be a positive integer or a sequence of positive integers.") from error
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive local dimensions.")
    return tuple(dims)


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


@dataclass(frozen=True)
class LocalMPO:
    """An open-boundary matrix-product operator in bra/ket convention."""

    dims: tuple[int, ...]
    tensors: tuple[np.ndarray, ...]

    def __init__(self, dims, tensors):
        dims = tuple(int(dim) for dim in dims)
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        if len(tensors) != len(dims):
            raise ValueError("an MPO must contain one tensor per site.")
        previous = 1
        copied = []
        for site, (dim, tensor) in enumerate(zip(dims, tensors)):
            if tensor.ndim != 4:
                raise ValueError("MPO tensors must have four axes.")
            if tensor.shape[0] != previous:
                raise ValueError(f"MPO bond mismatch before site {site}.")
            if tensor.shape[2:] != (dim, dim):
                raise ValueError(f"MPO tensor {site} has wrong physical shape.")
            previous = tensor.shape[1]
            value = np.array(tensor, copy=True)
            value.setflags(write=False)
            copied.append(value)
        if previous != 1:
            raise ValueError("the final MPO bond dimension must be one.")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "tensors", tuple(copied))

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return (1,) + tuple(tensor.shape[1] for tensor in self.tensors)

    @property
    def dtype(self):
        return np.dtype(np.result_type(*[tensor.dtype for tensor in self.tensors]))

    def to_dense(self) -> np.ndarray:
        """Materialize a dense reference matrix from the MPO."""
        environment = np.ones((1, 1, 1), dtype=self.dtype)
        output_dim = input_dim = 1
        for tensor, dim in zip(self.tensors, self.dims):
            value = np.tensordot(environment, tensor, axes=(0, 0))
            value = value.transpose(2, 0, 3, 1, 4)
            output_dim *= dim
            input_dim *= dim
            environment = value.reshape(tensor.shape[1], output_dim, input_dim)
        return environment[0]

    def compose(self, other: "LocalMPO") -> "LocalMPO":
        r"""Return the exact operator product ``self @ other``.

        The physical ket index of ``self`` is contracted with the physical
        bra index of ``other`` at every site.  The two MPO bond spaces are
        fused without truncation.
        """
        if not isinstance(other, LocalMPO):
            raise TypeError("other must be a LocalMPO.")
        if self.dims != other.dims:
            raise ValueError("MPO dimensions must match for composition.")
        tensors = []
        for left, right in zip(self.tensors, other.tensors):
            product = np.einsum(
                "absi,cdik->acbdsk",
                left,
                right,
                optimize=True,
            )
            tensors.append(
                product.reshape(
                    left.shape[0] * right.shape[0],
                    left.shape[1] * right.shape[1],
                    left.shape[2],
                    right.shape[3],
                )
            )
        return LocalMPO(self.dims, tensors)

    def compress(self, rtol=None) -> "LocalMPO":
        """Return a canonicalized MPO with redundant bond channels removed.

        The tensors are treated as a tensor train with local dimension
        ``bra_dim * ket_dim``.  A left-to-right QR sweep first makes the left
        blocks isometric; a right-to-left SVD sweep then exposes and removes
        linearly dependent channels on every bond.  ``rtol=None`` uses the
        usual matrix-rank threshold at each SVD and is exact to floating-point
        roundoff.  A nonnegative ``rtol`` instead drops singular values no
        larger than ``rtol * largest_singular_value``.
        """
        if rtol is not None:
            rtol = float(rtol)
            if not np.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite nonnegative number or None.")

        cores = [np.array(tensor, copy=True) for tensor in self.tensors]

        # Left-canonicalize.  The transpose is essential because the stored
        # layout is (left, right, bra, ket), while a TT unfolding groups
        # (left, bra, ket) against right.
        for site in range(len(cores) - 1):
            left_dim, right_dim, bra_dim, ket_dim = cores[site].shape
            matrix = (
                cores[site]
                .transpose(0, 2, 3, 1)
                .reshape(
                    left_dim * bra_dim * ket_dim,
                    right_dim,
                )
            )
            left, transfer = np.linalg.qr(matrix, mode="reduced")
            rank = left.shape[1]
            cores[site] = left.reshape(left_dim, bra_dim, ket_dim, rank).transpose(
                0, 3, 1, 2
            )
            cores[site + 1] = np.tensordot(
                transfer,
                cores[site + 1],
                axes=(1, 0),
            )

        # TT rounding.  Following the QR sweep, these are the Schmidt
        # singular values of the represented operator across each cut.
        for site in range(len(cores) - 1, 0, -1):
            left_dim, right_dim, bra_dim, ket_dim = cores[site].shape
            matrix = (
                cores[site]
                .transpose(0, 2, 3, 1)
                .reshape(
                    left_dim,
                    bra_dim * ket_dim * right_dim,
                )
            )
            left, singular_values, right = np.linalg.svd(
                matrix,
                full_matrices=False,
            )
            if singular_values.size and singular_values[0] > 0.0:
                relative_tolerance = (
                    np.finfo(singular_values.dtype).eps * max(matrix.shape)
                    if rtol is None
                    else rtol
                )
                rank = int(
                    np.count_nonzero(
                        singular_values > relative_tolerance * singular_values[0]
                    )
                )
                rank = max(1, rank)
            else:
                rank = 1

            left = left[:, :rank]
            singular_values = singular_values[:rank]
            right = right[:rank]
            cores[site] = right.reshape(rank, bra_dim, ket_dim, right_dim).transpose(
                0, 3, 1, 2
            )
            transfer = left * singular_values[None, :]
            previous = np.tensordot(
                cores[site - 1],
                transfer,
                axes=(1, 0),
            )
            cores[site - 1] = previous.transpose(0, 3, 1, 2)

        return LocalMPO(self.dims, cores)


@dataclass(frozen=True)
class LocalMPOProduct:
    """Lazy exact product of two MPOs with unfused bond channels.

    This stores ``left @ right`` without allocating the dense Kronecker
    product of their MPO bonds.  Identity-aware contractors can consume the
    two factors directly; :meth:`materialize` remains available for small
    reference calculations and backends that require ordinary MPO tensors.
    """

    left: LocalMPO
    right: LocalMPO

    def __init__(self, left, right):
        if not isinstance(left, LocalMPO) or not isinstance(right, LocalMPO):
            raise TypeError("left and right must be LocalMPO objects.")
        if left.dims != right.dims:
            raise ValueError("MPO dimensions must match for composition.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def dims(self) -> tuple[int, ...]:
        return self.left.dims

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(
            left * right
            for left, right in zip(self.left.bond_dims, self.right.bond_dims)
        )

    @property
    def dtype(self):
        return np.dtype(np.result_type(self.left.dtype, self.right.dtype))

    @property
    def materialized_elements(self) -> int:
        return int(
            sum(
                left_bond
                * right_bond
                * next_left
                * next_right
                * dim**2
                for left_bond, right_bond, next_left, next_right, dim in zip(
                    self.left.bond_dims[:-1],
                    self.right.bond_dims[:-1],
                    self.left.bond_dims[1:],
                    self.right.bond_dims[1:],
                    self.dims,
                )
            )
        )

    def materialize(self) -> LocalMPO:
        """Allocate the ordinary fused-bond MPO for ``left @ right``."""
        return self.left.compose(self.right)

    def to_dense(self) -> np.ndarray:
        """Materialize the many-body matrix for small-system validation."""
        return self.left.to_dense() @ self.right.to_dense()


def _as_charge_tuple(value) -> tuple[int, ...]:
    if hasattr(value, "charge"):
        value = value.charge
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return tuple(int(component) for component in value)
    return (int(value),)


def _add_charge_tuples(left, right) -> tuple[int, ...]:
    left = _as_charge_tuple(left)
    right = _as_charge_tuple(right)
    if len(left) != len(right):
        raise ValueError("all Abelian charges must have the same rank.")
    return tuple(a + b for a, b in zip(left, right))


def _normalized_local_charges(dims, local_charges, target):
    dims = tuple(int(dim) for dim in dims)
    if len(local_charges) != len(dims):
        raise ValueError("local_charges must contain one entry per site.")
    charges = tuple(
        tuple(_as_charge_tuple(charge) for charge in site_charges)
        for site_charges in local_charges
    )
    if any(len(site_charges) != dim for site_charges, dim in zip(charges, dims)):
        raise ValueError(
            "each local_charges entry must contain one charge per local state."
        )
    target = _as_charge_tuple(target)
    if any(
        len(charge) != len(target)
        for site_charges in charges
        for charge in site_charges
    ):
        raise ValueError("all local charges must have the target charge rank.")
    return dims, charges, target


def fixed_charge_projector_mpo(
    dims,
    local_charges,
    target,
    *,
    left_boundary=None,
) -> LocalMPO:
    r"""Build the exact diagonal projector onto one total Abelian charge.

    The MPO bond is the running charge.  At each cut, unreachable prefixes and
    prefixes that cannot be completed to ``target`` are removed exactly.  The
    projector acts once on each physical site; repeated tied-leg occurrences
    in a graph LETTA tensor are not separate charge carriers.
    """
    dims, local_charges, target = _normalized_local_charges(
        dims,
        local_charges,
        target,
    )
    rank = len(target)
    if left_boundary is None:
        left_boundary = tuple(0 for _ in range(rank))
    left_boundary = _as_charge_tuple(left_boundary)
    if len(left_boundary) != rank:
        raise ValueError("left_boundary and target must have the same charge rank.")

    nsites = len(dims)
    prefixes = [set() for _ in range(nsites + 1)]
    prefixes[0].add(left_boundary)
    for site, charges in enumerate(local_charges):
        prefixes[site + 1] = {
            _add_charge_tuples(prefix, charge)
            for prefix in prefixes[site]
            for charge in charges
        }

    zero = tuple(0 for _ in range(rank))
    suffixes = [set() for _ in range(nsites + 1)]
    suffixes[-1].add(zero)
    for site in range(nsites - 1, -1, -1):
        suffixes[site] = {
            _add_charge_tuples(charge, suffix)
            for charge in local_charges[site]
            for suffix in suffixes[site + 1]
        }

    bond_charges = []
    for cut in range(nsites + 1):
        completable = {
            prefix
            for prefix in prefixes[cut]
            if any(
                _add_charge_tuples(prefix, suffix) == target
                for suffix in suffixes[cut]
            )
        }
        if cut == 0:
            completable &= {left_boundary}
        if cut == nsites:
            completable &= {target}
        states = tuple(sorted(completable))
        if not states:
            raise ValueError("no product configurations have the requested charge.")
        bond_charges.append(states)

    tensors = []
    for site, (dim, charges) in enumerate(zip(dims, local_charges)):
        left_states = bond_charges[site]
        right_states = bond_charges[site + 1]
        right_lookup = {charge: index for index, charge in enumerate(right_states)}
        tensor = np.zeros(
            (len(left_states), len(right_states), dim, dim),
            dtype=float,
        )
        for left_index, left_charge in enumerate(left_states):
            for physical, charge in enumerate(charges):
                right_charge = _add_charge_tuples(left_charge, charge)
                right_index = right_lookup.get(right_charge)
                if right_index is not None:
                    tensor[left_index, right_index, physical, physical] = 1.0
        tensors.append(tensor)
    return LocalMPO(dims, tensors)


def validate_charge_conservation(
    hamiltonian: "LocalHamiltonian",
    local_charges=None,
    *,
    atol=None,
) -> None:
    r"""Raise unless every local term conserves all supplied Abelian charges."""
    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be a LocalHamiltonian.")
    if local_charges is None:
        local_charges = local_charges_from_sites(
            hamiltonian.sites,
            require=True,
        )
    dims, local_charges, target = _normalized_local_charges(
        hamiltonian.dims,
        local_charges,
        tuple(0 for _ in _as_charge_tuple(local_charges[0][0])),
    )
    del dims, target
    eps = np.finfo(float).eps
    for term in hamiltonian.terms:
        support_dims = tuple(hamiltonian.dims[site] for site in term.sites)
        for component in range(len(local_charges[0][0])):
            diagonal = np.empty(int(np.prod(support_dims)), dtype=float)
            for row, configuration in enumerate(np.ndindex(*support_dims)):
                diagonal[row] = sum(
                    local_charges[site][physical][component]
                    for site, physical in zip(term.sites, configuration)
                )
            commutator = (
                diagonal[:, None] * term.operator
                - term.operator * diagonal[None, :]
            )
            scale = max(float(np.linalg.norm(term.operator, ord=np.inf)), 1.0)
            tolerance = (
                512.0 * eps * max(term.operator.shape) * scale
                if atol is None
                else float(atol)
            )
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("atol must be finite and nonnegative.")
            error = float(np.linalg.norm(commutator, ord=np.inf))
            if error > tolerance:
                raise ValueError(
                    f"Hamiltonian term on sites {term.sites} does not conserve "
                    f"charge component {component}: commutator norm "
                    f"{error:.3e} exceeds {tolerance:.3e}."
                )


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


class LocalHamiltonian:
    r"""A sum of finite-support operators without a many-body matrix.

    The representation stores

    .. math::

        H = c I + \sum_t h_t,

    and combines terms that have identical supports.  ``to_dense`` and
    ``to_sparse`` are explicit small-system reference helpers; frontier LETTA
    contractions consume the local terms directly.
    """

    def __init__(self, dims=None, terms=(), *, constant=0.0, sites=None):
        explicit_sites = _normalize_sites(sites)
        inferred_sites = _normalize_sites(dims)
        if explicit_sites is not None and inferred_sites is not None:
            raise TypeError("provide either dims or sites, not both.")
        if explicit_sites is not None:
            self.sites = explicit_sites
            dims = tuple(int(getattr(site, "dim", site.d)) for site in self.sites)
        elif inferred_sites is not None:
            self.sites = inferred_sites
            dims = tuple(int(getattr(site, "dim", site.d)) for site in self.sites)
        else:
            self.sites = None

        self.dims = _normalized_dims(dims)
        if self.sites is not None and len(self.sites) != len(self.dims):
            raise ValueError("the number of sites must match dims.")
        local_charges = local_charges_from_sites(self.sites)
        if local_charges is not None and len(local_charges) != len(self.dims):
            raise ValueError("local charge metadata must cover every site.")
        self.local_charges = local_charges
        self.legs = (
            tuple(site.leg for site in self.sites)
            if self.sites is not None
            else None
        )

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
        self.terms = tuple(combined)
        self.dtype = np.dtype(
            np.result_type(
                self.constant,
                *[term.operator.dtype for term in self.terms],
                np.float64,
            )
        )
        dimension = int(np.prod(self.dims))
        self.shape = (dimension, dimension)

    @property
    def nterms(self) -> int:
        return len(self.terms)

    @property
    def supports(self) -> tuple[tuple[int, ...], ...]:
        return tuple(term.sites for term in self.terms)

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
        for term in self.terms:
            rest = tuple(site for site in all_sites if site not in term.sites)
            permutation = term.sites + rest
            inverse = np.argsort(permutation)
            support_dim = term.operator.shape[0]
            state_matrix = np.transpose(state, permutation).reshape(support_dim, -1)
            contribution = (term.operator @ state_matrix).reshape(
                tuple(self.dims[site] for site in permutation)
            )
            result = result + np.transpose(contribution, inverse)
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

    def to_mpo(self) -> LocalMPO:
        """Build an exact finite-state MPO without a many-body matrix.

        Each local term is TT-SVD-factorized only on its support.  A shared
        idle/done automaton starts exactly one term and carries its internal
        TT channel across gaps.  The construction is polynomial in the stored
        local-operator data size and never visits the full Hilbert basis.  A
        generic dense many-site term can itself have exponentially large data
        and TT ranks; the intended scalable case has bounded-size local terms.
        """
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
        for term in self.terms:
            if np.linalg.norm(term.operator) == 0.0:
                continue
            paths.append(
                (
                    term.sites,
                    _operator_tt_cores(term.sites, term.operator, self.dims),
                )
            )

        if not paths:
            tensors = []
            for site, dim in enumerate(self.dims):
                tensor = np.eye(dim, dtype=self.dtype)[None, None, :, :]
                if site == 0:
                    tensor = np.zeros_like(tensor)
                tensors.append(tensor)
            return LocalMPO(self.dims, tensors)

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
        return LocalMPO(self.dims, tensors)

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


__all__ = [
    "LocalHamiltonian",
    "LocalMPO",
    "LocalMPOProduct",
    "LocalTerm",
    "local_charges_from_sites",
    "fixed_charge_projector_mpo",
    "validate_charge_conservation",
]
