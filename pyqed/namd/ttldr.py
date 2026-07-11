"""Small tensor-operator layer for TT-LDR experiments.

The classes here are intentionally lightweight.  They do not replace the
current dense/vectorized Triatom propagator; they provide structured operator
pieces that can be tested and then used as building blocks for a future
matrix-free TT-LDR backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def _normalize_rank(max_rank, step, nsteps):
    if max_rank is None:
        return None
    if np.isscalar(max_rank):
        return int(max_rank)
    ranks = list(max_rank)
    if len(ranks) == nsteps + 1:
        return int(ranks[step + 1])
    if len(ranks) == nsteps:
        return int(ranks[step])
    raise ValueError("max_rank must be a scalar or a TT-rank list.")


def _tt_svd(tensor, max_rank=None, atol=0.0):
    """Return tensor-train cores for a dense tensor using sequential SVD."""
    tensor = np.asarray(tensor)
    dims = tuple(int(d) for d in tensor.shape)
    if not dims:
        return [tensor.reshape(1, 1, 1)]

    cores = []
    unfolding = tensor.reshape(1, *dims)
    left_rank = 1
    for step, dim in enumerate(dims[:-1]):
        unfolding = unfolding.reshape(left_rank * dim, -1)
        u, s, vh = np.linalg.svd(unfolding, full_matrices=False)
        keep = s.size
        if atol > 0.0:
            keep = max(1, int(np.count_nonzero(s > atol)))
        rank_cap = _normalize_rank(max_rank, step, len(dims) - 1)
        if rank_cap is not None:
            keep = min(keep, rank_cap)
        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep]
        cores.append(u.reshape(left_rank, dim, keep))
        unfolding = s[:, None] * vh
        left_rank = keep
    cores.append(unfolding.reshape(left_rank, dims[-1], 1))
    return cores


def _tt_to_tensor(cores):
    tensor = np.asarray(cores[0])
    for core in cores[1:]:
        tensor = np.tensordot(tensor, np.asarray(core), axes=([-1], [0]))
    shape = [core.shape[1] for core in cores]
    return tensor.reshape(shape)


@dataclass
class Diagonal:
    """Diagonal operator stored as a tensor train over site dimensions."""

    cores: list[np.ndarray]
    site_dims: tuple[int, ...]
    label: str = "diagonal"

    @classmethod
    def from_values(cls, values, *, max_rank=None, atol=0.0, label="diagonal"):
        values = np.asarray(values)
        cores = _tt_svd(values, max_rank=max_rank, atol=atol)
        return cls(cores=cores, site_dims=tuple(values.shape), label=label)

    def to_tensor(self):
        return _tt_to_tensor(self.cores)

    def apply(self, state):
        state = np.asarray(state)
        if state.shape != self.site_dims:
            raise ValueError(f"state shape {state.shape} != operator shape {self.site_dims}")
        return self.to_tensor() * state


@dataclass
class MPO:
    """Matrix-product operator over a list of local site dimensions."""

    cores: list[np.ndarray]
    site_dims: tuple[int, ...]
    label: str = "mpo"

    @classmethod
    def from_dense_matrix(cls, matrix, site_dims, *, max_rank=None, atol=0.0, label="mpo"):
        site_dims = tuple(int(d) for d in site_dims)
        matrix = np.asarray(matrix)
        dim = int(np.prod(site_dims))
        if matrix.shape != (dim, dim):
            raise ValueError(f"matrix shape {matrix.shape} != {(dim, dim)}")

        nsites = len(site_dims)
        op_tensor = matrix.reshape(*site_dims, *site_dims)
        interleave = []
        for site in range(nsites):
            interleave.extend([site, site + nsites])
        op_tensor = np.transpose(op_tensor, interleave)
        pair_dims = tuple(d * d for d in site_dims)
        tt_cores = _tt_svd(op_tensor.reshape(pair_dims), max_rank=max_rank, atol=atol)
        cores = [
            core.reshape(core.shape[0], site_dims[i], site_dims[i], core.shape[2])
            for i, core in enumerate(tt_cores)
        ]
        return cls(cores=cores, site_dims=site_dims, label=label)

    @classmethod
    def from_product(cls, factors: Iterable[np.ndarray], *, coefficient=1.0, label="product"):
        cores = []
        site_dims = []
        for index, factor in enumerate(factors):
            factor = np.asarray(factor)
            if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
                raise ValueError("product factors must be square matrices")
            if index == 0:
                factor = coefficient * factor
            cores.append(factor.reshape(1, factor.shape[0], factor.shape[1], 1))
            site_dims.append(factor.shape[0])
        return cls(cores=cores, site_dims=tuple(site_dims), label=label)

    def to_dense_matrix(self):
        tensor = np.asarray(self.cores[0])
        for core in self.cores[1:]:
            tensor = np.tensordot(tensor, np.asarray(core), axes=([-1], [0]))
        nsites = len(self.site_dims)
        tensor = np.squeeze(tensor, axis=(0, -1))
        out_axes = [2 * site for site in range(nsites)]
        in_axes = [2 * site + 1 for site in range(nsites)]
        tensor = np.transpose(tensor, out_axes + in_axes)
        dim = int(np.prod(self.site_dims))
        return tensor.reshape(dim, dim)

    def apply(self, state):
        state = np.asarray(state)
        if state.shape != self.site_dims:
            raise ValueError(f"state shape {state.shape} != operator shape {self.site_dims}")
        return (self.to_dense_matrix() @ state.reshape(-1)).reshape(self.site_dims)


@dataclass
class LinkedOverlap:
    """Linked-product LDR overlap descriptor.

    This keeps the nearest-neighbor electronic overlaps as structured data.  It
    can materialize a dense overlap MPO for small test systems, but production
    use should consume the links directly.
    """

    nx: tuple[int, ...]
    nstates: int
    links: dict
    label: str = "linked-overlap"

    @property
    def site_dims(self):
        return (*self.nx, self.nstates)

    def block(self, solver, i, j, bra_idx, ket_idx):
        return solver._linked_overlap_block(
            i,
            j,
            bra_idx,
            ket_idx,
            self.links,
            self.nstates,
        )

    def to_mpo(self, solver, *, max_rank=None, atol=0.0):
        overlap = solver._build_linked_overlap_from_links(self.links, self.nstates)
        matrix = overlap.reshape(int(np.prod(self.site_dims)), int(np.prod(self.site_dims)))
        return MPO.from_dense_matrix(
            matrix,
            self.site_dims,
            max_rank=max_rank,
            atol=atol,
            label=self.label,
        )


@dataclass
class ProductTerm:
    """Rank-1 tensor-product operator term.

    This is the intended representation for separable kinetic pieces such as
    ``D_i(q)`` times a small rotational ``J_a J_b`` matrix.
    """

    factors: tuple[np.ndarray, ...]
    coefficient: complex = 1.0
    label: str = "product-term"

    def to_mpo(self):
        return MPO.from_product(
            self.factors,
            coefficient=self.coefficient,
            label=self.label,
        )


@dataclass
class Bundle:
    """Structured TT-LDR operator pieces for a Triatom model."""

    site_dims: tuple[int, ...]
    potential: Diagonal | None = None
    overlap: MPO | LinkedOverlap | None = None
    rotational_terms: list[ProductTerm] = field(default_factory=list)

    def add_rotational_term(self, factors, *, coefficient=1.0, label="rotational-term"):
        term = ProductTerm(
            factors=tuple(np.asarray(factor) for factor in factors),
            coefficient=coefficient,
            label=label,
        )
        self.rotational_terms.append(term)
        return term


@dataclass
class Action:
    """Matrix-free TT-LDR action on a dense wavepacket tensor.

    The operators are structured, but the wavepacket is intentionally still a
    dense tensor in this first layer.  A compressed MPS state needs truncation
    and error-control policy, so it should be added after this action is
    validated against the current Triatom driver.
    """

    solver: object
    T: object
    bundle: Bundle
    threshold: float = 0.0
    _dense_overlap: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.nx = tuple(int(n) for n in self.solver.nx)
        self.ng = int(np.prod(self.nx))
        self.nstates = int(self.solver.nstates)
        self.shape = tuple(int(n) for n in self.bundle.site_dims)
        self.has_rot = len(self.shape) == len(self.nx) + 2
        self.nrot = self.shape[-2] if self.has_rot else 1
        if self.has_rot:
            expected = (*self.nx, self.nrot, self.nstates)
        else:
            expected = (*self.nx, self.nstates)
        if self.shape != expected:
            raise ValueError(f"bundle site_dims {self.shape} != expected {expected}")

    @property
    def size(self):
        return int(np.prod(self.shape))

    def _state(self, psi):
        psi = np.asarray(psi)
        if psi.shape == (self.size,):
            return psi.reshape(self.shape)
        if psi.shape != self.shape:
            raise ValueError(f"state shape {psi.shape} != expected {self.shape}")
        return psi

    def _overlap_blocks(self):
        if self._dense_overlap is None:
            matrix = self.bundle.overlap.to_dense_matrix()
            self._dense_overlap = matrix.reshape(
                self.ng,
                self.nstates,
                self.ng,
                self.nstates,
            )
        return self._dense_overlap

    def _block(self, i, j, bra_idx, ket_idx):
        overlap = self.bundle.overlap
        if overlap is None:
            return np.eye(self.nstates, dtype=complex)
        if isinstance(overlap, MPO):
            return self._overlap_blocks()[i, :, j, :]
        if isinstance(overlap, LinkedOverlap):
            return overlap.block(self.solver, i, j, bra_idx, ket_idx)
        raise TypeError(f"unsupported overlap type {type(overlap)!r}")

    def v(self, psi):
        """Apply the APES/diagonal potential action."""
        psi = self._state(psi)
        if self.bundle.potential is None:
            return np.zeros_like(psi, dtype=complex)
        return self.bundle.potential.apply(psi)

    def k(self, psi):
        """Apply the LDR kinetic action using the bundle overlap."""
        try:
            import scipy.sparse as sp
        except ModuleNotFoundError:
            sp = None

        psi = self._state(psi)
        if sp is not None and sp.issparse(self.T):
            return self._k_sparse(psi, self.T.tocsr())
        return self._k_dense(psi, np.asarray(self.T))

    def h(self, psi):
        """Apply ``K + V``."""
        return self.k(psi) + self.v(psi)

    def linear(self, part="h"):
        """Return a SciPy ``LinearOperator`` for ``h``, ``k``, or ``v``."""
        from scipy.sparse.linalg import LinearOperator

        actions = {
            "h": self.h,
            "k": self.k,
            "v": self.v,
        }
        if part not in actions:
            raise ValueError("part must be 'h', 'k', or 'v'")
        action = actions[part]
        dtype = np.result_type(getattr(self.T, "dtype", complex), complex)

        def matvec(vec):
            vec = np.asarray(vec).reshape(-1)
            return action(vec).reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        return LinearOperator(
            shape=(self.size, self.size),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=dtype,
        )

    def _k_dense(self, psi, T):
        overlap = self.bundle.overlap
        if self.has_rot:
            T = T.reshape(self.ng, self.nrot, self.ng, self.nrot)
            psi_rs = psi.reshape(self.ng, self.nrot, self.nstates)
            if overlap is None:
                return np.einsum("irjs,jsa->ira", T, psi_rs, optimize=True).reshape(
                    self.shape
                )
            if isinstance(overlap, MPO):
                A = self._overlap_blocks()
                return np.einsum(
                    "irjs,iajb,jsb->ira",
                    T,
                    A,
                    psi_rs,
                    optimize=True,
                ).reshape(self.shape)
            return self._k_linked_dense_rot(psi_rs, T)

        psi_g = psi.reshape(self.ng, self.nstates)
        if overlap is None:
            return (T @ psi_g).reshape(self.shape)
        if isinstance(overlap, MPO):
            A = self._overlap_blocks()
            return np.einsum("ij,iajb,jb->ia", T, A, psi_g, optimize=True).reshape(
                self.shape
            )
        return self._k_linked_dense(psi_g, T).reshape(self.shape)

    def _k_sparse(self, psi, T):
        indices = self.solver._grid_indices()
        rows = np.repeat(np.arange(T.shape[0]), np.diff(T.indptr))
        cols = T.indices.copy()
        data = T.data.copy()
        if self.threshold > 0.0:
            keep = np.abs(data) > self.threshold
            rows = rows[keep]
            cols = cols[keep]
            data = data[keep]

        if self.has_rot:
            psi_rs = psi.reshape(self.ng, self.nrot, self.nstates)
            out = np.zeros_like(psi_rs, dtype=np.result_type(data.dtype, complex))
            for row, col, Tij in zip(rows, cols, data):
                i, r = divmod(int(row), self.nrot)
                j, s = divmod(int(col), self.nrot)
                Aij = self._block(i, j, indices[i], indices[j])
                out[i, r] += Tij * (Aij @ psi_rs[j, s])
            return out.reshape(self.shape)

        psi_g = psi.reshape(self.ng, self.nstates)
        if self.bundle.overlap is None:
            return (T @ psi_g).reshape(self.shape)

        out = np.zeros_like(psi_g, dtype=np.result_type(data.dtype, complex))
        for row, col, Tij in zip(rows, cols, data):
            i = int(row)
            j = int(col)
            Aij = self._block(i, j, indices[i], indices[j])
            out[i] += Tij * (Aij @ psi_g[j])
        return out.reshape(self.shape)

    def _k_linked_dense(self, psi, T):
        indices = self.solver._grid_indices()
        out = np.zeros_like(psi, dtype=np.result_type(T.dtype, complex))
        for i, bra_idx in enumerate(indices):
            for j, ket_idx in enumerate(indices):
                Tij = T[i, j]
                if self.threshold > 0.0 and abs(Tij) <= self.threshold:
                    continue
                Aij = self._block(i, j, bra_idx, ket_idx)
                out[i] += Tij * (Aij @ psi[j])
        return out

    def _k_linked_dense_rot(self, psi, T):
        indices = self.solver._grid_indices()
        out = np.zeros_like(psi, dtype=np.result_type(T.dtype, complex))
        for i, bra_idx in enumerate(indices):
            for j, ket_idx in enumerate(indices):
                block = T[i, :, j, :]
                if self.threshold > 0.0 and np.max(np.abs(block)) <= self.threshold:
                    continue
                Aij = self._block(i, j, bra_idx, ket_idx)
                transported = psi[j] @ Aij.T
                out[i] += np.einsum("rs,sa->ra", block, transported, optimize=True)
        return out.reshape(self.shape)


def build_potential(apes, *, nrot=1, max_rank=None, atol=0.0):
    """Build a diagonal TT operator from APES values."""
    values = np.asarray(apes)
    if nrot > 1:
        values = np.expand_dims(values, axis=-2)
        values = np.broadcast_to(values, (*values.shape[:-2], int(nrot), values.shape[-1]))
    return Diagonal.from_values(
        values,
        max_rank=max_rank,
        atol=atol,
        label="apes-potential",
    )


def build_overlap(solver, *, max_rank=None, atol=0.0, prefer_links=True):
    """Build an overlap operator descriptor from a ``Triatom`` instance."""
    links = getattr(solver, "overlap_links", None)
    if prefer_links and links is not None:
        return LinkedOverlap(
            nx=tuple(solver.nx),
            nstates=int(solver.nstates),
            links=links,
        )

    overlap = getattr(solver, "overlap_matrix", None)
    if overlap is None:
        return None
    site_dims = (*tuple(solver.nx), int(solver.nstates))
    matrix = np.asarray(overlap).reshape(int(np.prod(site_dims)), int(np.prod(site_dims)))
    return MPO.from_dense_matrix(
        matrix,
        site_dims,
        max_rank=max_rank,
        atol=atol,
        label="electronic-overlap",
    )


def build_bundle(solver, *, max_rank=None, atol=0.0, prefer_links=True):
    """Collect APES, overlap, and future kinetic pieces for a ``Triatom``."""
    if solver.apes is None:
        raise RuntimeError("APES not built. Set solver.apes before building TT-LDR pieces.")
    potential = build_potential(
        solver.apes,
        nrot=solver.nrot if getattr(solver, "J", 0) > 0 else 1,
        max_rank=max_rank,
        atol=atol,
    )
    overlap = build_overlap(
        solver,
        max_rank=max_rank,
        atol=atol,
        prefer_links=prefer_links,
    )
    site_dims = potential.site_dims
    return Bundle(site_dims=site_dims, potential=potential, overlap=overlap)


def build_action(
    solver,
    T=None,
    *,
    bundle=None,
    sparse=False,
    threshold=0.0,
    max_rank=None,
    atol=0.0,
    prefer_links=True,
):
    """Build a matrix-free TT-LDR action for a ``Triatom`` solver."""
    if bundle is None:
        bundle = build_bundle(
            solver,
            max_rank=max_rank,
            atol=atol,
            prefer_links=prefer_links,
        )
    if T is None:
        try:
            T = solver.buildK(sparse=sparse)
        except TypeError:
            T = solver.buildK()
    try:
        import scipy.sparse as sp
    except ModuleNotFoundError:
        sp = None
    if sp is not None and sp.issparse(T):
        T = 0.5 * (T + T.getH())
        T = T.tocsr()
    else:
        T = np.asarray(T)
        T = 0.5 * (T + T.conj().T)
    return Action(solver=solver, T=T, bundle=bundle, threshold=threshold)
