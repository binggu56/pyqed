#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational sweeps for NARG basis optimization.

This module is intentionally small and dense-reference based. It implements
the finite two-site optimization loop needed by NARG while keeping the local
effective Hamiltonian construction explicit. For production-size NARG runs,
the dense projectors here should be replaced by cached left/right
environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, eigsh

from ..core import SequentialNARGState, fuse_two_sites, narg_state_vector

try:
    from opt_einsum import contract as _contract
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    _contract = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional accelerator
    njit = None


_DEFAULT_MATRIX_FREE_LOCAL_DIM = 8192
_DEFAULT_MATRIX_FREE_MEMORY_LIMIT = 512 * 1024**2
_DEFAULT_MATRIX_FREE_FALLBACK_DIM = 2048
_DEFAULT_SUPPORT_BATCH_SIZE = 64
_SPARSE_MPO_DENSITY_LIMIT = 0.20
_SPARSE_MPO_MIN_BOND_PRODUCT = 3000
_SPARSE_MPO_SITE_CACHE_MAXSIZE = 128
_SPARSE_MPO_SITE_CACHE = {}
_TWO_SITE_MPO_ENTRY_CACHE_MAXSIZE = 128
_TWO_SITE_MPO_ENTRY_CACHE = {}


if njit is not None:
    @njit(nogil=True, cache=False)
    def _support_heff_sparse_numba(coords, left, right, entry_starts, entry_m, entry_n, entry_values):
        nallowed = coords.shape[0]
        heff = np.empty((nallowed, nallowed), dtype=np.complex128)
        di = left.shape[3]
        dj = right.shape[3]
        for row in range(nallowed):
            bra_left = coords[row, 0]
            bra_i = coords[row, 1]
            bra_j = coords[row, 2]
            bra_right = coords[row, 3]
            for col in range(nallowed):
                ket_left = coords[col, 0]
                ket_i = coords[col, 1]
                ket_j = coords[col, 2]
                ket_right = coords[col, 3]
                block = (((bra_i * di + ket_i) * dj + bra_j) * dj + ket_j)
                value = 0.0 + 0.0j
                for entry in range(entry_starts[block], entry_starts[block + 1]):
                    m = entry_m[entry]
                    n = entry_n[entry]
                    value += (
                        left[bra_left, ket_left, m, bra_i, ket_i]
                        * entry_values[entry]
                        * right[bra_right, ket_right, n, bra_j, ket_j]
                    )
                heff[row, col] = value
        return heff

    @njit(nogil=True, cache=False)
    def _support_heff_indexed_numba(coords, left, right, w_left, w_right):
        nallowed = coords.shape[0]
        heff = np.empty((nallowed, nallowed), dtype=np.complex128)
        wm = w_left.shape[0]
        wp = w_left.shape[1]
        wn = w_right.shape[1]
        for row in range(nallowed):
            bra_left = coords[row, 0]
            bra_i = coords[row, 1]
            bra_j = coords[row, 2]
            bra_right = coords[row, 3]
            for col in range(nallowed):
                ket_left = coords[col, 0]
                ket_i = coords[col, 1]
                ket_j = coords[col, 2]
                ket_right = coords[col, 3]
                value = 0.0 + 0.0j
                for m in range(wm):
                    left_value = left[bra_left, ket_left, m, bra_i, ket_i]
                    if left_value == 0:
                        continue
                    for p in range(wp):
                        w0 = w_left[m, p, bra_i, ket_i]
                        if w0 == 0:
                            continue
                        for n in range(wn):
                            w1 = w_right[p, n, bra_j, ket_j]
                            if w1 == 0:
                                continue
                            value += (
                                left_value
                                * w0
                                * w1
                                * right[bra_right, ket_right, n, bra_j, ket_j]
                            )
                heff[row, col] = value
        return heff
else:
    _support_heff_sparse_numba = None
    _support_heff_indexed_numba = None


def _two_site_mpo_sparse_entries(w_left, w_right):
    w_left = np.asarray(w_left)
    w_right = np.asarray(w_right)
    key = (id(w_left), id(w_right), w_left.shape, w_right.shape)
    if key in _TWO_SITE_MPO_ENTRY_CACHE:
        cached_left, cached_right, cached_result = _TWO_SITE_MPO_ENTRY_CACHE[key]
        if cached_left is w_left and cached_right is w_right:
            return cached_result

    left_density = np.count_nonzero(w_left) / max(1, w_left.size)
    right_density = np.count_nonzero(w_right) / max(1, w_right.size)
    if max(left_density, right_density) > _SPARSE_MPO_DENSITY_LIMIT:
        result = None
        _cache_two_site_mpo_entries(key, w_left, w_right, result)
        return result

    di = w_left.shape[2]
    dj = w_right.shape[2]
    shared_dim = w_left.shape[1]
    blocks = [dict() for _ in range(di * di * dj * dj)]
    for shared in range(shared_dim):
        left_entries = []
        for bra_i in range(di):
            for ket_i in range(di):
                column = w_left[:, shared, bra_i, ket_i]
                rows = np.flatnonzero(column)
                if rows.size:
                    left_entries.append((bra_i, ket_i, rows, column[rows]))
        if not left_entries:
            continue

        right_entries = []
        for bra_j in range(dj):
            for ket_j in range(dj):
                row = w_right[shared, :, bra_j, ket_j]
                cols = np.flatnonzero(row)
                if cols.size:
                    right_entries.append((bra_j, ket_j, cols, row[cols]))
        if not right_entries:
            continue

        for bra_i, ket_i, rows, left_values in left_entries:
            for bra_j, ket_j, cols, right_values in right_entries:
                block = blocks[(((bra_i * di + ket_i) * dj + bra_j) * dj + ket_j)]
                for m, left_value in zip(rows, left_values):
                    for n, right_value in zip(cols, right_values):
                        entry_key = (int(m), int(n))
                        block[entry_key] = block.get(entry_key, 0.0) + left_value * right_value

    starts = np.empty(len(blocks) + 1, dtype=np.int64)
    entry_m = []
    entry_n = []
    entry_values = []
    cursor = 0
    starts[0] = 0
    for block_index, block in enumerate(blocks):
        for (m, n), value in block.items():
            if value != 0:
                entry_m.append(m)
                entry_n.append(n)
                entry_values.append(value)
                cursor += 1
        starts[block_index + 1] = cursor

    result = (
        starts,
        np.asarray(entry_m, dtype=np.int64),
        np.asarray(entry_n, dtype=np.int64),
        np.asarray(entry_values, dtype=np.result_type(w_left.dtype, w_right.dtype, complex)),
    )
    _cache_two_site_mpo_entries(key, w_left, w_right, result)
    return result


def _cache_two_site_mpo_entries(key, w_left, w_right, value):
    if key in _TWO_SITE_MPO_ENTRY_CACHE:
        _TWO_SITE_MPO_ENTRY_CACHE.pop(key)
    elif len(_TWO_SITE_MPO_ENTRY_CACHE) >= _TWO_SITE_MPO_ENTRY_CACHE_MAXSIZE:
        _TWO_SITE_MPO_ENTRY_CACHE.pop(next(iter(_TWO_SITE_MPO_ENTRY_CACHE)))
    _TWO_SITE_MPO_ENTRY_CACHE[key] = (w_left, w_right, value)


def _sparse_mpo_site_package(mpo_site):
    mpo_site = np.asarray(mpo_site)
    key = (id(mpo_site), mpo_site.shape)
    if key in _SPARSE_MPO_SITE_CACHE:
        cached_site, cached_result = _SPARSE_MPO_SITE_CACHE[key]
        if cached_site is mpo_site:
            return cached_result

    if mpo_site.shape[0] * mpo_site.shape[1] < _SPARSE_MPO_MIN_BOND_PRODUCT:
        result = None
        _cache_sparse_mpo_site_package(key, mpo_site, result)
        return result
    density = np.count_nonzero(mpo_site) / max(1, mpo_site.size)
    if density > _SPARSE_MPO_DENSITY_LIMIT:
        result = None
        _cache_sparse_mpo_site_package(key, mpo_site, result)
        return result

    blocks = []
    for bra_state in range(mpo_site.shape[2]):
        row = []
        for ket_state in range(mpo_site.shape[3]):
            block = csr_matrix(mpo_site[:, :, bra_state, ket_state])
            if not block.nnz:
                row.append(None)
                continue
            rows = np.flatnonzero(np.diff(block.indptr))
            cols = np.unique(block.indices)
            row.append((block[:, cols], block[rows, :], rows, cols))
        blocks.append(tuple(row))
    result = tuple(blocks)
    _cache_sparse_mpo_site_package(key, mpo_site, result)
    return result


def _cache_sparse_mpo_site_package(key, mpo_site, value):
    if key in _SPARSE_MPO_SITE_CACHE:
        _SPARSE_MPO_SITE_CACHE.pop(key)
    elif len(_SPARSE_MPO_SITE_CACHE) >= _SPARSE_MPO_SITE_CACHE_MAXSIZE:
        _SPARSE_MPO_SITE_CACHE.pop(next(iter(_SPARSE_MPO_SITE_CACHE)))
    _SPARSE_MPO_SITE_CACHE[key] = (mpo_site, value)


def _advance_left_environment_sparse(env, tensor, package, right_mpo_dim):
    bra_left, ket_left, left_mpo_dim, left_phys, ket_phys = env.shape
    right_bond = tensor.shape[3]
    next_phys = tensor.shape[2]
    dtype = np.result_type(env.dtype, tensor.dtype)
    out = np.zeros(
        (right_bond, right_bond, right_mpo_dim, next_phys, next_phys),
        dtype=dtype,
    )
    tensor_conj = tensor.conj()
    for bra_state in range(left_phys):
        for ket_state in range(ket_phys):
            block = package[bra_state][ket_state]
            if block is None:
                continue
            left_block, _right_block, _rows, cols = block
            q = np.asarray(
                env[:, :, :, bra_state, ket_state].reshape(bra_left * ket_left, left_mpo_dim)
                @ left_block
            ).reshape(bra_left, ket_left, cols.size)
            out[:, :, cols, :, :] += np.einsum(
                "bkn,buc,kvd->cdnuv",
                q,
                tensor_conj[:, bra_state, :, :],
                tensor[:, ket_state, :, :],
                optimize=True,
            )
    return out


def _advance_right_environment_sparse(env, tensor, package, left_mpo_dim):
    bra_right, ket_right, right_mpo_dim, bra_phys, ket_phys = env.shape
    left_bond = tensor.shape[0]
    left_phys = tensor.shape[1]
    dtype = np.result_type(env.dtype, tensor.dtype)
    out = np.zeros(
        (left_bond, left_bond, left_mpo_dim, left_phys, left_phys),
        dtype=dtype,
    )
    tensor_conj = tensor.conj()
    for bra_state in range(bra_phys):
        for ket_state in range(ket_phys):
            block = package[bra_state][ket_state]
            if block is None:
                continue
            _left_block, right_block, rows, _cols = block
            q = np.asarray(
                env[:, :, :, bra_state, ket_state].reshape(bra_right * ket_right, right_mpo_dim)
                @ right_block.T
            ).reshape(bra_right, ket_right, rows.size)
            out[:, :, rows, :, :] += np.einsum(
                "cdm,bxc,kyd->bkmxy",
                q,
                tensor_conj[:, :, bra_state, :],
                tensor[:, :, ket_state, :],
                optimize=True,
            )
    return out


def _as_matrix(operator):
    matrix = operator.toarray() if issparse(operator) else np.asarray(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix.")
    return matrix


def _validate_dims(dims):
    dims = tuple(int(d) for d in dims)
    if not dims or any(d < 1 for d in dims):
        raise ValueError("dims must be a non-empty sequence of positive integers.")
    return dims


def _validate_local_solver(local_solver):
    solver = str(local_solver).lower().replace("-", "_")
    if solver not in {"auto", "dense", "matrix_free"}:
        raise ValueError("local_solver must be 'auto', 'dense', or 'matrix_free'.")
    return solver


def _default_bonds(dims, bond_dim):
    bonds = [1]
    left_dim = 1
    total = int(np.prod(dims))
    for d in dims[:-1]:
        left_dim *= d
        right_dim = total // left_dim
        bonds.append(min(int(bond_dim), left_dim, right_dim))
    bonds.append(1)
    return bonds


def _normalize_with_metric(vector, metric):
    norm2 = np.vdot(vector, metric @ vector)
    norm = np.sqrt(float(np.real(norm2)))
    if norm < 1e-14:
        raise ValueError("Cannot normalize a numerically zero state.")
    return vector / norm


def _metric_basis(metric, *, metric_tol=1e-12, metric_threshold=None):
    metric = 0.5 * (metric + metric.conj().T)
    if metric_threshold is None:
        metric_scale = max(1.0, float(np.linalg.norm(metric, ord=np.inf)))
        metric_threshold = metric_tol * metric_scale
    try:
        metric_vals, metric_vecs = linalg.eigh(
            metric,
            subset_by_value=(metric_threshold, np.inf),
            driver="evx",
            check_finite=False,
        )
    except (ValueError, linalg.LinAlgError):
        metric_vals, metric_vecs = linalg.eigh(metric, check_finite=False)
    keep = metric_vals > metric_threshold
    if not np.any(keep):
        raise ValueError("Effective overlap metric is numerically singular.")
    return metric_vecs[:, keep] / np.sqrt(metric_vals[keep])[None, :]


def _lowest_hermitian_eigenpair(matrix, *, iterative_threshold=256, tol=1e-10):
    matrix = 0.5 * (matrix + matrix.conj().T)
    if matrix.shape[0] > iterative_threshold:
        try:
            evals, evecs = eigsh(
                matrix,
                k=1,
                which="SA",
                tol=tol,
                maxiter=max(1000, 20 * matrix.shape[0]),
            )
            return float(np.real(evals[0])), evecs[:, 0]
        except Exception:
            pass
    evals, evecs = linalg.eigh(
        matrix,
        subset_by_index=[0, 0],
        check_finite=False,
    )
    return float(np.real(evals[0])), evecs[:, 0]


def _lowest_generalized_eigenpair(hamiltonian, metric, *, metric_tol=1e-12):
    """
    Solve the lowest generalized eigenpair in the nonsingular metric range.
    """
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    basis = _metric_basis(metric, metric_tol=metric_tol)
    reduced_h = basis.conj().T @ hamiltonian @ basis
    energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
    vector = basis @ reduced_vector
    vector = _normalize_with_metric(vector, metric)
    return energy, vector


def _metric_blocks_from_support(coords, mleft, mright, *, metric_tol=1e-12):
    left_diag = np.einsum("bkxx->bkx", mleft[:, :, 0], optimize=True)
    right_diag = np.einsum("cduu->cdu", mright[:, :, 0], optimize=True)
    di = mleft.shape[3]
    dj = mright.shape[3]

    raw_blocks = []
    metric_scale = 1.0
    for si in range(di):
        left_block = 0.5 * (left_diag[:, :, si] + left_diag[:, :, si].conj().T)
        for sj in range(dj):
            block_indices = np.flatnonzero((coords[:, 1] == si) & (coords[:, 2] == sj))
            if block_indices.size == 0:
                continue
            right_block = 0.5 * (right_diag[:, :, sj] + right_diag[:, :, sj].conj().T)
            left_sel = coords[block_indices, 0]
            right_sel = coords[block_indices, 3]
            metric = (
                left_block[left_sel[:, None], left_sel[None, :]]
                * right_block[right_sel[:, None], right_sel[None, :]]
            )
            metric = 0.5 * (metric + metric.conj().T)
            metric_scale = max(metric_scale, float(np.linalg.norm(metric, ord=np.inf)))
            raw_blocks.append((block_indices, metric))

    threshold = float(metric_tol) * metric_scale
    blocks = []
    for block_indices, metric in raw_blocks:
        try:
            basis = _metric_basis(metric, metric_threshold=threshold)
        except ValueError:
            continue
        blocks.append((block_indices, basis))
    if not blocks:
        raise ValueError("Effective overlap metric is numerically singular.")
    return blocks


def _lowest_generalized_eigenpair_from_metric_blocks(hamiltonian, blocks):
    offsets = []
    reduced_dim = 0
    dtype = hamiltonian.dtype
    for block_indices, basis in blocks:
        offsets.append((reduced_dim, reduced_dim + basis.shape[1]))
        reduced_dim += basis.shape[1]
        dtype = np.result_type(dtype, basis.dtype)
    if reduced_dim < 1:
        raise ValueError("Effective overlap metric is numerically singular.")

    reduced_h = np.empty((reduced_dim, reduced_dim), dtype=dtype)
    for row_block, (row_indices, row_basis) in enumerate(blocks):
        row_slice = slice(*offsets[row_block])
        row_basis_h = row_basis.conj().T
        for col_block, (col_indices, col_basis) in enumerate(blocks):
            col_slice = slice(*offsets[col_block])
            reduced_h[row_slice, col_slice] = (
                row_basis_h @ hamiltonian[np.ix_(row_indices, col_indices)] @ col_basis
            )
    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
    energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)

    vector = np.zeros(hamiltonian.shape[0], dtype=np.result_type(reduced_vector.dtype, dtype))
    for block, (block_indices, basis) in enumerate(blocks):
        block_slice = slice(*offsets[block])
        vector[block_indices] = basis @ reduced_vector[block_slice]
    return energy, vector


@dataclass
class TensorTrainLETTAResult:
    """
    Result container returned by :meth:`TensorTrainLETTA.run`.
    """

    energy: float
    cores: list
    history: list
    converged: bool
    ncompleted: int


@dataclass
class LETTAResult:
    """
    Result container returned by :meth:`LETTA.run`.
    """

    energy: float
    tensors: list
    history: list
    converged: bool
    ncompleted: int


@dataclass
class LETTAOperatorPackage:
    """Reusable MPO and metric environments for one LETTA sweep."""

    ansatz: object
    mpo: list
    direction: str
    left_envs: list
    right_envs: list
    metric_left: list
    metric_right: list

    @classmethod
    def for_sweep(cls, ansatz, mpo, direction):
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        identity = ansatz.identity_mpo()
        nlocal = ansatz.nlocal_tensors
        if direction == "lr":
            right_envs = ansatz._right_local_environments(mpo)
            metric_right = ansatz._right_metric_environments()
            left_envs = [None] * nlocal
            metric_left = [None] * nlocal
            left_envs[0] = np.ones(
                (1, 1, mpo[0].shape[0], ansatz.dims[0], ansatz.dims[0]),
                dtype=right_envs[0].dtype,
            )
            metric_left[0] = np.ones(
                (1, 1, identity[0].shape[0], ansatz.dims[0], ansatz.dims[0]),
                dtype=metric_right[0].dtype,
            )
        else:
            left_envs = ansatz._left_local_environments(mpo)
            metric_left = ansatz._left_metric_environments()
            right_envs = [None] * nlocal
            metric_right = [None] * nlocal
            if ansatz.has_terminal_tensor:
                right_envs[ansatz.npairs - 1] = ansatz._terminal_right_environment(
                    mpo,
                    dtype=left_envs[-1].dtype,
                )
                metric_right[ansatz.npairs - 1] = ansatz._terminal_right_environment(
                    dtype=metric_left[-1].dtype,
                )
            else:
                right_envs[-1] = np.ones(
                    (1, 1, mpo[-1].shape[1], ansatz.dims[-1], ansatz.dims[-1]),
                    dtype=left_envs[-1].dtype,
                )
                metric_right[-1] = np.ones(
                    (1, 1, identity[-1].shape[1], ansatz.dims[-1], ansatz.dims[-1]),
                    dtype=metric_left[-1].dtype,
                )
        return cls(ansatz, mpo, direction, left_envs, right_envs, metric_left, metric_right)

    def advance_after_update(self, tensor_index):
        tensor_index = int(tensor_index)
        if self.direction == "lr":
            if tensor_index < self.ansatz.npairs and tensor_index + 1 < self.ansatz.nlocal_tensors:
                self.left_envs[tensor_index + 1] = self.ansatz._advance_left_environment(
                    self.left_envs[tensor_index],
                    self.mpo[tensor_index],
                    self.ansatz.tensors[tensor_index],
                )
                self.metric_left[tensor_index + 1] = self.ansatz._advance_left_metric_environment(
                    self.metric_left[tensor_index],
                    self.ansatz.tensors[tensor_index],
                )
        elif self.ansatz.has_terminal_tensor and tensor_index == self.ansatz.npairs:
            self.right_envs[self.ansatz.npairs - 1] = self.ansatz._terminal_right_environment(
                self.mpo,
                dtype=self.left_envs[self.ansatz.npairs].dtype,
            )
            self.metric_right[self.ansatz.npairs - 1] = self.ansatz._terminal_right_environment(
                dtype=self.metric_left[self.ansatz.npairs].dtype,
            )
        elif tensor_index:
            self.right_envs[tensor_index - 1] = self.ansatz._advance_right_environment(
                self.right_envs[tensor_index],
                self.mpo[tensor_index + 1],
                self.ansatz.tensors[tensor_index],
            )
            self.metric_right[tensor_index - 1] = self.ansatz._advance_right_metric_environment(
                self.metric_right[tensor_index],
                self.ansatz.tensors[tensor_index],
            )


class TensorTrainLETTA:
    """
    Local eigensolver tensor-train ansatz for NARG states.

    The state is stored as MPS-like NARG cores with shape
    ``(left, physical, right)``. The physical index may represent a primitive
    coordinate grid, an adiabatic channel, or any retained local NARG basis.

    Parameters
    ----------
    hamiltonian
        Full Hamiltonian in the product basis defined by ``dims``.
    dims
        Local basis dimensions.
    bond_dim
        Maximum number of retained renormalized states on every bond.
    overlap
        Optional full overlap matrix. If provided, local solves use
        ``H_eff c = E S_eff c``. If omitted, the product basis is assumed
        orthonormal, but the local MPS projector metric is still included.
    cores
        Optional initial cores with shape ``(left, physical, right)``.
    seed
        Random seed used when ``cores`` is omitted.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=32,
        overlap=None,
        cores=None,
        seed=None,
    ):
        self.dims = _validate_dims(dims)
        expected = int(np.prod(self.dims))
        if hamiltonian is None:
            self.hamiltonian = None
        else:
            self.hamiltonian = _as_matrix(hamiltonian)
            if self.hamiltonian.shape != (expected, expected):
                raise ValueError(
                    f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {expected}."
                )

        self.overlap = None if overlap is None else _as_matrix(overlap)
        if self.overlap is not None and self.overlap.shape != (expected, expected):
            raise ValueError("overlap shape must match product dimension.")

        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        self.rng = np.random.default_rng(seed)
        self.cores = self._random_cores() if cores is None else self._validate_cores(cores)
        self.history = []
        self.converged = False
        self.energy = None

    def _random_cores(self):
        bonds = _default_bonds(self.dims, self.bond_dim)
        cores = []
        for n, d in enumerate(self.dims):
            shape = (bonds[n], d, bonds[n + 1])
            core = self.rng.normal(size=shape)
            core = core / np.sqrt(np.prod(shape))
            cores.append(core.astype(float))
        return cores

    def _validate_cores(self, cores):
        cores = [np.asarray(core, dtype=complex if np.iscomplexobj(core) else float) for core in cores]
        if len(cores) != len(self.dims):
            raise ValueError("number of cores must match dims.")
        for n, (core, d) in enumerate(zip(cores, self.dims)):
            if core.ndim != 3 or core.shape[1] != d:
                raise ValueError(f"core {n} must have shape (left, {d}, right).")
            if n == 0 and core.shape[0] != 1:
                raise ValueError("first core must have left bond dimension 1.")
            if n == len(cores) - 1 and core.shape[2] != 1:
                raise ValueError("last core must have right bond dimension 1.")
            if n and cores[n - 1].shape[2] != core.shape[0]:
                raise ValueError(f"bond mismatch between cores {n - 1} and {n}.")
        return cores

    @property
    def nsites(self):
        return len(self.dims)

    def copy(self):
        return TensorTrainLETTA(
            self.hamiltonian.copy(),
            self.dims,
            bond_dim=self.bond_dim,
            overlap=None if self.overlap is None else self.overlap.copy(),
            cores=[core.copy() for core in self.cores],
        )

    def state_vector(self):
        """Return the dense product-basis vector represented by the cores."""
        psi = self.cores[0][0]
        for core in self.cores[1:]:
            psi = np.tensordot(psi, core, axes=([-1], [0]))
        return psi.reshape(-1)

    def norm(self):
        if self.overlap is None:
            norm2 = float(np.real(self._mpo_matrix_element(self.identity_mpo())))
            return 0.0 if -1e-12 < norm2 < 0.0 else norm2
        psi = self.state_vector()
        return float(np.real(np.vdot(psi, self.overlap @ psi)))

    def expectation(self):
        psi = self.state_vector()
        denom = np.vdot(psi, psi) if self.overlap is None else np.vdot(psi, self.overlap @ psi)
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(np.vdot(psi, self.hamiltonian @ psi) / denom))

    def _left_basis(self, stop):
        basis = np.ones((1, 1), dtype=self.cores[0].dtype)
        for core in self.cores[:stop]:
            basis = np.tensordot(basis, core, axes=([1], [0]))
            basis = basis.reshape(-1, core.shape[2])
        return basis

    def _right_basis(self, start):
        dtype = self.cores[-1].dtype
        basis = np.ones((1, 1), dtype=dtype)
        for core in reversed(self.cores[start:]):
            basis = np.tensordot(core, basis, axes=([2], [0]))
            basis = basis.reshape(core.shape[0], -1)
        return basis

    def _bond_projector(self, bond):
        left = self._left_basis(bond)
        right = self._right_basis(bond + 2)
        dl = self.cores[bond].shape[0]
        dr = self.cores[bond + 1].shape[2]
        di = self.dims[bond]
        dj = self.dims[bond + 1]
        eye_i = np.eye(di, dtype=left.dtype)
        eye_j = np.eye(dj, dtype=left.dtype)
        projector = np.einsum("xa,it,ju,by->xijyatub", left, eye_i, eye_j, right)
        return projector.reshape(left.shape[0] * di * dj * right.shape[1], dl * di * dj * dr)

    def _solve_local(self, bond):
        projector = self._bond_projector(bond)
        heff = projector.conj().T @ self.hamiltonian @ projector
        if self.overlap is None:
            seff = projector.conj().T @ projector
        else:
            seff = projector.conj().T @ self.overlap @ projector
        seff = 0.5 * (seff + seff.conj().T)
        heff = 0.5 * (heff + heff.conj().T)

        return _lowest_generalized_eigenpair(heff, seff)

    def _split_local_vector(self, bond, vector, direction):
        left_dim = self.cores[bond].shape[0]
        right_dim = self.cores[bond + 1].shape[2]
        di = self.dims[bond]
        dj = self.dims[bond + 1]
        theta = vector.reshape(left_dim, di, dj, right_dim)
        matrix = theta.reshape(left_dim * di, dj * right_dim)
        u, singular_values, vh = linalg.svd(matrix, full_matrices=False)
        keep = min(self.bond_dim, len(singular_values))
        discarded = singular_values[keep:]
        u = u[:, :keep]
        s = singular_values[:keep]
        vh = vh[:keep]

        if direction == "lr":
            left_core = u.reshape(left_dim, di, keep)
            right_core = (s[:, None] * vh).reshape(keep, dj, right_dim)
        else:
            left_core = (u * s[None, :]).reshape(left_dim, di, keep)
            right_core = vh.reshape(keep, dj, right_dim)

        trunc_err = float(np.sum(discarded**2))
        self.cores[bond] = left_core
        self.cores[bond + 1] = right_core
        return keep, trunc_err

    def sweep(self, direction="lr"):
        """
        Perform one two-site sweep and return per-bond diagnostics.
        """
        if self.nsites < 2:
            raise ValueError("At least two sites are required for two-site sweeps.")
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        bonds = range(self.nsites - 1)
        if direction == "rl":
            bonds = reversed(list(bonds))

        updates = []
        for bond in bonds:
            local_energy, vector = self._solve_local(bond)
            kept, trunc_err = self._split_local_vector(bond, vector, direction)
            updates.append(
                {
                    "bond": int(bond),
                    "local_energy": float(local_energy),
                    "kept": int(kept),
                    "trunc_err": trunc_err,
                }
            )
        return updates

    def run(self, *, nsweeps=4, start_direction="lr", alternate=True, tol=1e-10, verbose=0):
        """
        Run finite two-site sweeps.
        """
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep(direction)
            energy = updates[-1]["local_energy"] if updates else self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return TensorTrainLETTAResult(
            energy=self.energy,
            cores=[core.copy() for core in self.cores],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )


class LETTA:
    r"""
    Dense-reference nearest-neighbor leg-tied tensor ansatz.

    The represented wavefunction is

    .. math::

        \Psi(\sigma_0,\ldots,\sigma_{L-1}) =
        \sum_{\alpha_0\ldots\alpha_{L-3}}
        \prod_{i=0}^{L-2}
        A^{[i]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_i},

    with boundary bond dimensions ``alpha[-1] = alpha[L-2] = 1``.  The
    physical index ``sigma_i`` is therefore shared by neighboring tensors,
    unlike in an MPS where each physical leg appears in exactly one tensor.

    This class is a small dense prototype for one-site LETTA optimization.  It
    is intended for validating the variational equations and for seeding from a
    NARG/MPS state before replacing dense projectors by cached environments.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=4,
        overlap=None,
        tensors=None,
        local_masks=None,
        seed=None,
    ):
        self.dims = _validate_dims(dims)
        if len(self.dims) < 2:
            raise ValueError("LETTA needs at least two physical sites.")
        expected = int(np.prod(self.dims))
        if hamiltonian is None:
            self.hamiltonian = None
        else:
            self.hamiltonian = _as_matrix(hamiltonian)
            if self.hamiltonian.shape != (expected, expected):
                raise ValueError(
                    f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {expected}."
                )

        self.overlap = None if overlap is None else _as_matrix(overlap)
        if self.overlap is not None and self.overlap.shape != (expected, expected):
            raise ValueError("overlap shape must match product dimension.")

        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        self.rng = np.random.default_rng(seed)
        self.tensors = self._random_tensors() if tensors is None else self._validate_tensors(tensors)
        self.local_masks = self._validate_local_masks(local_masks)
        self._apply_local_masks()
        self.history = []
        self.converged = False
        self.energy = None
        self.normalize()

    @classmethod
    def from_state_vector(
        cls,
        hamiltonian,
        dims,
        state,
        *,
        bond_dim=4,
        overlap=None,
        seed=None,
        fit_sweeps=4,
        ridge=1e-12,
    ):
        """
        Initialize a leg-tied LETTA by least-squares fitting a dense state.

        This is the practical bridge from a NARG/MPS guess: compute the NARG
        state vector, then fit the tied-leg tensors by alternating one-site
        linear least squares.
        """
        obj = cls(hamiltonian, dims, bond_dim=bond_dim, overlap=overlap, seed=seed)
        obj.fit_state(state, nsweeps=fit_sweeps, ridge=ridge)
        return obj

    @staticmethod
    def _convert_narg_tensors(tensors, coeff, *, dims=None, root=0, append_terminal=False):
        """
        Convert a sequential NARG factorization to leg-tied LETTA tensors.

        The expected NARG tensor convention is
        ``tensor_i[physical_i * left + alpha_left, alpha_right, physical_{i+1}]``.
        The final NARG coefficient vector is absorbed into the last LETTA
        tensor by default.  With ``append_terminal=True`` it is appended as a
        terminal one-site LETTA tensor ``A[-1][sigma_last, alpha_final]``.
        """
        tensors = [np.asarray(tensor) for tensor in tensors]
        if not tensors:
            raise ValueError("at least one NARG tensor is required.")
        coeff = np.asarray(coeff)
        if coeff.ndim == 2:
            coeff = coeff[:, int(root)]
        elif coeff.ndim != 1:
            raise ValueError("coeff must be a one- or two-dimensional array.")

        if dims is None:
            inferred = [tensors[0].shape[0]]
            inferred.extend(tensor.shape[2] for tensor in tensors)
            dims = tuple(inferred)
        dims = _validate_dims(dims)
        if len(dims) != len(tensors) + 1:
            raise ValueError("dims must have length len(tensors)+1.")

        bond_dims = [tensor.shape[1] for tensor in tensors]
        final_dim = bond_dims[-1]
        if coeff.size != dims[-1] * final_dim:
            raise ValueError("coeff size must equal dims[-1] times the final NARG bond dimension.")

        letta_tensors = []
        for i, tensor in enumerate(tensors):
            left_dim = 1 if i == 0 else bond_dims[i - 1]
            right_dim = bond_dims[i] if (append_terminal or i < len(tensors) - 1) else 1
            if tensor.shape != (dims[i] * left_dim, bond_dims[i], dims[i + 1]):
                raise ValueError(
                    f"NARG tensor {i} must have shape "
                    f"({dims[i] * left_dim}, {bond_dims[i]}, {dims[i + 1]})."
                )
            out = np.zeros(
                (left_dim, dims[i], dims[i + 1], right_dim),
                dtype=np.result_type(tensor.dtype, coeff.dtype),
            )
            if append_terminal or i < len(tensors) - 1:
                for left, si, sj, right in np.ndindex(left_dim, dims[i], dims[i + 1], right_dim):
                    row = si * left_dim + left
                    out[left, si, sj, right] = tensor[row, right, sj]
            else:
                for left, si, sj in np.ndindex(left_dim, dims[i], dims[i + 1]):
                    row = si * left_dim + left
                    out[left, si, sj, 0] = sum(
                        tensor[row, alpha, sj] * coeff[sj * final_dim + alpha]
                        for alpha in range(final_dim)
                    )
            letta_tensors.append(out)
        if append_terminal:
            letta_tensors.append(coeff.reshape(dims[-1], final_dim).copy())
        return letta_tensors

    @classmethod
    def from_narg(
        cls,
        narg,
        coeff=None,
        *,
        dims=None,
        root=0,
        hamiltonian=None,
        bond_dim=None,
        overlap=None,
        seed=None,
        local_masks=None,
        preserve_support=False,
        support_tol=1e-12,
        append_terminal=False,
        fit_sweeps=4,
        ridge=1e-12,
    ):
        """
        Initialize LETTA from a NARG result.

        ``narg`` may either be a list of sequential NARG tensors, in which case
        ``coeff`` supplies the final root coefficients, or an object exposing a
        dense ``state_vector()`` method.
        """
        if coeff is None and hasattr(narg, "state_vector"):
            dims = tuple(dims if dims is not None else getattr(narg, "dims"))
            if bond_dim is None:
                bond_dim = getattr(narg, "bond_dim", 4)
            return cls.from_state_vector(
                hamiltonian,
                dims,
                narg.state_vector(),
                bond_dim=bond_dim,
                overlap=overlap,
                seed=seed,
                fit_sweeps=fit_sweeps,
                ridge=ridge,
            )
        if coeff is None:
            raise TypeError("coeff is required when narg is a list of NARG tensors.")
        letta_tensors = cls._convert_narg_tensors(
            narg, coeff, dims=dims, root=root, append_terminal=append_terminal
        )
        if preserve_support and local_masks is None:
            local_masks = [np.abs(tensor) > float(support_tol) for tensor in letta_tensors]
        if dims is None:
            pair_tensors = letta_tensors[:-1] if append_terminal else letta_tensors
            dims = (pair_tensors[0].shape[1],) + tuple(tensor.shape[2] for tensor in pair_tensors)
        if bond_dim is None:
            pair_tensors = letta_tensors[:len(dims) - 1]
            bond_dim = max(max(tensor.shape[0], tensor.shape[3]) for tensor in pair_tensors)
            if len(letta_tensors) == len(dims):
                bond_dim = max(bond_dim, letta_tensors[-1].shape[1])
        return cls(
            hamiltonian,
            dims,
            bond_dim=bond_dim,
            overlap=overlap,
            tensors=letta_tensors,
            local_masks=local_masks,
            seed=seed,
        )

    @property
    def nsites(self):
        return len(self.dims)

    @property
    def nbonds(self):
        return len(self.dims) - 1

    @property
    def npairs(self):
        return len(self.dims) - 1

    @property
    def has_terminal_tensor(self):
        return len(self.tensors) == self.npairs + 1

    @property
    def nlocal_tensors(self):
        return len(self.tensors)

    def _default_letta_bonds(self):
        return [1] + [self.bond_dim] * max(0, self.nsites - 2) + [1]

    def _random_tensors(self):
        bonds = self._default_letta_bonds()
        tensors = []
        for i in range(self.nbonds):
            shape = (bonds[i], self.dims[i], self.dims[i + 1], bonds[i + 1])
            tensor = self.rng.normal(size=shape)
            tensor = tensor / np.sqrt(np.prod(shape))
            tensors.append(tensor.astype(float))
        return tensors

    def _validate_tensors(self, tensors):
        tensors = [np.asarray(tensor, dtype=complex if np.iscomplexobj(tensor) else float) for tensor in tensors]
        if len(tensors) not in {self.npairs, self.npairs + 1}:
            raise ValueError("number of LETTA tensors must be len(dims)-1, optionally plus a terminal tensor.")
        has_terminal = len(tensors) == self.npairs + 1
        pair_tensors = tensors[:self.npairs]
        for i, tensor in enumerate(pair_tensors):
            if tensor.ndim != 4 or tensor.shape[1:3] != self.dims[i:i + 2]:
                raise ValueError(f"tensor {i} must have shape (left, {self.dims[i]}, {self.dims[i + 1]}, right).")
            if i == 0 and tensor.shape[0] != 1:
                raise ValueError("first LETTA tensor must have left bond dimension 1.")
            if not has_terminal and i == self.npairs - 1 and tensor.shape[3] != 1:
                raise ValueError("last LETTA tensor must have right bond dimension 1.")
            if i and pair_tensors[i - 1].shape[3] != tensor.shape[0]:
                raise ValueError(f"bond mismatch between LETTA tensors {i - 1} and {i}.")
        if has_terminal:
            terminal = tensors[-1]
            final_dim = pair_tensors[-1].shape[3]
            if terminal.ndim != 2 or terminal.shape != (self.dims[-1], final_dim):
                raise ValueError(
                    f"terminal LETTA tensor must have shape ({self.dims[-1]}, {final_dim})."
                )
        return tensors

    def _validate_local_masks(self, local_masks):
        if local_masks is None:
            return [None] * self.nlocal_tensors
        local_masks = list(local_masks)
        if len(local_masks) != self.nlocal_tensors:
            raise ValueError("local_masks must have one entry per LETTA tensor.")
        validated = []
        for i, mask in enumerate(local_masks):
            if mask is None:
                validated.append(None)
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.tensors[i].shape:
                if mask.size != self.tensors[i].size:
                    raise ValueError(f"local mask {i} has incompatible shape {mask.shape}.")
                mask = mask.reshape(self.tensors[i].shape)
            if not np.any(mask):
                raise ValueError(f"local mask {i} removes every tensor entry.")
            validated.append(mask.copy())
        return validated

    def _apply_local_masks(self):
        for i, mask in enumerate(self.local_masks):
            if mask is not None:
                self.tensors[i] = np.where(mask, self.tensors[i], 0)

    def local_support_sizes(self):
        """Return allowed/total entry counts for each symmetry/support mask."""
        sizes = []
        for tensor, mask in zip(self.tensors, self.local_masks):
            allowed = tensor.size if mask is None else int(np.count_nonzero(mask))
            sizes.append((allowed, tensor.size))
        return sizes

    def copy(self):
        return LETTA(
            None if self.hamiltonian is None else self.hamiltonian.copy(),
            self.dims,
            bond_dim=self.bond_dim,
            overlap=None if self.overlap is None else self.overlap.copy(),
            tensors=[tensor.copy() for tensor in self.tensors],
            local_masks=[None if mask is None else mask.copy() for mask in self.local_masks],
        )

    def _amplitude(self, config):
        vec = self.tensors[0][0, config[0], config[1], :]
        for i in range(1, self.npairs):
            vec = vec @ self.tensors[i][:, config[i], config[i + 1], :]
        if self.has_terminal_tensor:
            return vec @ self.tensors[-1][config[-1], :]
        return vec[0]

    def state_vector(self):
        """Return the dense product-basis vector represented by tied tensors."""
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        psi = np.empty(int(np.prod(self.dims)), dtype=dtype)
        for flat, config in enumerate(np.ndindex(*self.dims)):
            psi[flat] = self._amplitude(config)
        return psi

    def norm(self):
        if self.overlap is None:
            norm2 = float(np.real(self._mpo_matrix_element(self.identity_mpo())))
            return 0.0 if -1e-12 < norm2 < 0.0 else norm2
        psi = self.state_vector()
        return float(np.real(np.vdot(psi, self.overlap @ psi)))

    def normalize(self):
        norm = np.sqrt(self.norm())
        if norm < 1e-14:
            raise ValueError("Cannot normalize a numerically zero LETTA state.")
        # Rescale a single tensor; this preserves the tied-leg structure.
        self.tensors[0] = self.tensors[0] / norm
        return self

    def _validate_dense_operator(self, operator):
        operator = _as_matrix(operator)
        expected = int(np.prod(self.dims))
        if operator.shape != (expected, expected):
            raise ValueError(f"operator shape {operator.shape} does not match product dimension {expected}.")
        return operator

    def _looks_like_mpo(self, operator):
        return (
            isinstance(operator, (list, tuple))
            and len(operator) == self.nsites
            and all(np.asarray(site).ndim == 4 for site in operator)
        )

    def _looks_like_product_operators(self, operator):
        return (
            isinstance(operator, (list, tuple))
            and len(operator) == self.nsites
            and all(np.asarray(site).ndim == 2 for site in operator)
        )

    def _expectation_dense_operator(self, operator):
        operator = self._validate_dense_operator(operator)
        psi = self.state_vector()
        denom = np.vdot(psi, psi) if self.overlap is None else np.vdot(psi, self.overlap @ psi)
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(np.vdot(psi, operator @ psi) / denom))

    def expectation(self, operator=None):
        """
        Return ``<operator>`` for a dense operator, MPO, or product operator.

        If ``operator`` is omitted, the dense Hamiltonian supplied at
        construction time is used.
        """
        if operator is None:
            if self.hamiltonian is None:
                raise ValueError("dense hamiltonian is not available; pass an MPO or dense operator.")
            return self._expectation_dense_operator(self.hamiltonian)
        if self._looks_like_mpo(operator):
            return self.expectation_mpo(operator)
        if self._looks_like_product_operators(operator):
            return self.expectation_product_operator(operator)
        return self._expectation_dense_operator(operator)

    def expect(self, operator=None):
        """
        Short alias for :meth:`expectation`.
        """
        return self.expectation(operator)

    def _validate_mpo(self, mpo):
        mpo = [np.asarray(site) for site in mpo]
        if len(mpo) != self.nsites:
            raise ValueError("MPO length must match the number of physical sites.")
        for i, site in enumerate(mpo):
            if site.ndim != 4:
                raise ValueError("each MPO tensor must have shape (left, right, bra, ket).")
            if site.shape[2] != self.dims[i] or site.shape[3] != self.dims[i]:
                raise ValueError(f"MPO tensor {i} physical dimensions do not match dims[{i}].")
            if i == 0 and site.shape[0] != 1:
                raise ValueError("first MPO tensor must have left bond dimension 1.")
            if i == self.nsites - 1 and site.shape[1] != 1:
                raise ValueError("last MPO tensor must have right bond dimension 1.")
            if i and mpo[i - 1].shape[1] != site.shape[0]:
                raise ValueError(f"MPO bond mismatch between tensors {i - 1} and {i}.")
        return mpo

    def identity_mpo(self):
        """
        Return the product-basis identity as an MPO.
        """
        return [np.eye(dim, dtype=self.tensors[0].dtype).reshape(1, 1, dim, dim) for dim in self.dims]

    def apply_mpo(self, mpo, vector):
        """
        Apply an MPO to a dense product-basis vector. This is diagnostic; the
        MPO optimizer below does not form dense local projectors.
        """
        mpo = self._validate_mpo(mpo)
        tmp = np.asarray(vector).reshape(self.dims)[None, ...]
        for site, operator in enumerate(mpo):
            nout = site
            rem_after = self.nsites - site - 1
            tmp = np.tensordot(tmp, operator, axes=([0, nout + 1], [0, 3]))
            right_axis = nout + rem_after
            current_output_axis = right_axis + 1
            order = [right_axis] + list(range(nout)) + [current_output_axis] + list(range(nout, nout + rem_after))
            tmp = np.transpose(tmp, order)
        return tmp[0].reshape(-1)

    def expectation_mpo(self, mpo):
        """
        Expectation value with an MPO contracted directly against the LETTA
        double layer.
        """
        value = self._normalized_mpo_expectation(mpo)
        return float(np.real(value))

    def _mpo_matrix_element(self, mpo):
        """
        Contract ``<Psi|MPO|Psi>`` without forming dense state vectors.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        env = np.ones((1, 1, mpo[0].shape[0], self.dims[0], self.dims[0]), dtype=dtype)
        for i, tensor in enumerate(self.tensors[:self.npairs]):
            env = self._advance_left_environment(env, mpo[i], tensor)
        if self.has_terminal_tensor:
            return np.einsum(
                "bkmxy,mnxy,xb,yk->",
                env,
                mpo[-1],
                self.tensors[-1].conj(),
                self.tensors[-1],
                optimize=True,
            )
        return np.einsum("bkmxy,mnxy->", env, mpo[-1], optimize=True)

    def _mpo_matrix_element_direct(self, mpo):
        """
        Reference full-network contraction for ``_mpo_matrix_element``.
        """
        if _contract is None:
            raise ImportError("opt_einsum is required for direct LETTA contractions.")
        mpo = self._validate_mpo(mpo)
        nbonds = self.nbonds
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        next_label = 0

        def labels(count):
            nonlocal next_label
            out = list(range(next_label, next_label + count))
            next_label += count
            return out

        ket_phys = labels(self.nsites)
        bra_phys = labels(self.nsites)
        ket_bonds = labels(nbonds + 1)
        bra_bonds = labels(nbonds + 1)
        mpo_bonds = labels(self.nsites + 1)

        operands = []
        for site, operator in enumerate(mpo):
            operands.extend(
                [operator, [mpo_bonds[site], mpo_bonds[site + 1], bra_phys[site], ket_phys[site]]]
            )
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=dtype), [ket_bonds[0]]])
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=dtype), [bra_bonds[0]]])
        if self.has_terminal_tensor:
            terminal = self.tensors[-1]
            operands.extend([terminal, [ket_phys[-1], ket_bonds[-1]]])
            operands.extend([terminal.conj(), [bra_phys[-1], bra_bonds[-1]]])
        else:
            operands.extend([np.ones(self.tensors[-1].shape[3], dtype=dtype), [ket_bonds[-1]]])
            operands.extend([np.ones(self.tensors[-1].shape[3], dtype=dtype), [bra_bonds[-1]]])
        for i, tensor in enumerate(self.tensors[:self.npairs]):
            operands.extend([tensor, [ket_bonds[i], ket_phys[i], ket_phys[i + 1], ket_bonds[i + 1]]])
            operands.extend([tensor.conj(), [bra_bonds[i], bra_phys[i], bra_phys[i + 1], bra_bonds[i + 1]]])
        return _contract(*operands, [], optimize="auto")

    def _normalized_mpo_expectation(self, mpo):
        value = self._mpo_matrix_element(mpo)
        denom = self._mpo_matrix_element(self.identity_mpo())
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return value / denom

    def product_operator_mpo(self, operators):
        """
        Build a bond-1 MPO from one local operator per site.

        Each local operator uses the ``(bra, ket)`` convention.
        """
        if len(operators) != self.nsites:
            raise ValueError("number of local operators must match the number of physical sites.")
        mpo = []
        for i, operator in enumerate(operators):
            operator = np.asarray(operator)
            if operator.shape != (self.dims[i], self.dims[i]):
                raise ValueError(f"operator {i} must have shape ({self.dims[i]}, {self.dims[i]}).")
            mpo.append(operator.reshape(1, 1, self.dims[i], self.dims[i]))
        return mpo

    def product_mpo(self, operators):
        """Alias for :meth:`product_operator_mpo`."""
        return self.product_operator_mpo(operators)

    def expectation_product_operator(self, operators):
        """
        Expectation value of a product of local operators.
        """
        return self._normalized_mpo_expectation(self.product_operator_mpo(operators))

    def product_expectation(self, operators):
        """Alias for :meth:`expectation_product_operator`."""
        return self.expectation_product_operator(operators)

    def spatial_correlation(self, op_a, op_b=None, *, connected=False, average=False):
        """
        Compute ``<op_a(i) op_b(j)>`` or its connected correlation matrix.

        Parameters
        ----------
        op_a, op_b
            Local operators. If ``op_b`` is omitted, ``op_a`` is used for both
            sites. On-site entries use the ordered product ``op_a @ op_b``.
        connected
            If true, subtract ``<op_a(i)> <op_b(j)>``.
        average
            If true, return the distance-averaged correlation ``C(r)`` instead
            of the full ``C(i,j)`` matrix.
        """
        if len(set(self.dims)) != 1:
            raise ValueError("spatial_correlation currently requires equal local dimensions.")
        dim = self.dims[0]
        op_a = np.asarray(op_a)
        op_b = op_a if op_b is None else np.asarray(op_b)
        if op_a.shape != (dim, dim) or op_b.shape != (dim, dim):
            raise ValueError(f"local operators must have shape ({dim}, {dim}).")

        eye = np.eye(dim, dtype=np.result_type(op_a.dtype, op_b.dtype, self.tensors[0].dtype))
        one_a = np.empty(self.nsites, dtype=complex)
        one_b = np.empty(self.nsites, dtype=complex)
        corr = np.empty((self.nsites, self.nsites), dtype=complex)

        for i in range(self.nsites):
            ops = [eye] * self.nsites
            ops[i] = op_a
            one_a[i] = self.expectation_product_operator(ops)
            ops = [eye] * self.nsites
            ops[i] = op_b
            one_b[i] = self.expectation_product_operator(ops)

        for i in range(self.nsites):
            for j in range(self.nsites):
                ops = [eye] * self.nsites
                if i == j:
                    ops[i] = op_a @ op_b
                else:
                    ops[i] = op_a
                    ops[j] = op_b
                corr[i, j] = self.expectation_product_operator(ops)

        if connected:
            corr = corr - np.outer(one_a, one_b)
        if average:
            return np.array([np.mean([corr[i, i + r] for i in range(self.nsites - r)]) for r in range(self.nsites)])
        return corr

    def correlation(self, op_a, op_b=None, *, connected=False, average=False):
        """Alias for :meth:`spatial_correlation`."""
        return self.spatial_correlation(op_a, op_b, connected=connected, average=average)

    def local_effective_matrix(self, mpo, tensor_index):
        """
        Contract ``<dPsi/dA_i|MPO|dPsi/dA_i>`` without forming a dense
        product-basis projector.

        The output matrix is ordered consistently with
        ``self.tensors[tensor_index].reshape(-1)``.
        """
        mpo = self._validate_mpo(mpo)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        left_envs = self._left_local_environments(mpo)
        right_envs = self._right_local_environments(mpo)
        if self.has_terminal_tensor and tensor_index == self.npairs:
            return self._terminal_effective_from_environment(mpo, left_envs)
        return self._local_effective_from_environments(mpo, tensor_index, left_envs, right_envs)

    def local_effective_matrix_direct(self, mpo, tensor_index):
        """
        Reference full-network contraction for ``local_effective_matrix``.
        """
        if _contract is None:
            raise ImportError("opt_einsum is required for direct LETTA contractions.")
        mpo = self._validate_mpo(mpo)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")

        shape = self.tensors[tensor_index].shape
        nbonds = self.nbonds
        next_label = 0

        def labels(count):
            nonlocal next_label
            out = list(range(next_label, next_label + count))
            next_label += count
            return out

        ket_phys = labels(self.nsites)
        bra_phys = labels(self.nsites)
        ket_bonds = labels(nbonds + 1)
        bra_bonds = labels(nbonds + 1)
        mpo_bonds = labels(self.nsites + 1)

        operands = []
        for site, operator in enumerate(mpo):
            operands.extend(
                [operator, [mpo_bonds[site], mpo_bonds[site + 1], bra_phys[site], ket_phys[site]]]
            )

        operands.extend([np.ones(self.tensors[0].shape[0], dtype=shape and self.tensors[0].dtype), [ket_bonds[0]]])
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=shape and self.tensors[0].dtype), [bra_bonds[0]]])
        if self.has_terminal_tensor:
            terminal = self.tensors[-1]
            if tensor_index != self.npairs:
                operands.extend([terminal, [ket_phys[-1], ket_bonds[-1]]])
                operands.extend([terminal.conj(), [bra_phys[-1], bra_bonds[-1]]])
        else:
            operands.extend([np.ones(self.tensors[-1].shape[3], dtype=self.tensors[-1].dtype), [ket_bonds[-1]]])
            operands.extend([np.ones(self.tensors[-1].shape[3], dtype=self.tensors[-1].dtype), [bra_bonds[-1]]])

        for i, tensor in enumerate(self.tensors[:self.npairs]):
            if i == tensor_index:
                continue
            operands.extend([tensor, [ket_bonds[i], ket_phys[i], ket_phys[i + 1], ket_bonds[i + 1]]])
            operands.extend([tensor.conj(), [bra_bonds[i], bra_phys[i], bra_phys[i + 1], bra_bonds[i + 1]]])

        if self.has_terminal_tensor and tensor_index == self.npairs:
            output = [bra_phys[-1], bra_bonds[-1], ket_phys[-1], ket_bonds[-1]]
        else:
            output = [
                bra_bonds[tensor_index],
                bra_phys[tensor_index],
                bra_phys[tensor_index + 1],
                bra_bonds[tensor_index + 1],
                ket_bonds[tensor_index],
                ket_phys[tensor_index],
                ket_phys[tensor_index + 1],
                ket_bonds[tensor_index + 1],
            ]
        heff = _contract(*operands, output, optimize="auto")
        dim = int(np.prod(shape))
        return heff.reshape(dim, dim)

    def _left_local_environments(self, mpo):
        """
        Prefix contractions for LETTA one-site MPO environments.

        ``left[k]`` leaves ``(bra_alpha_k, ket_alpha_k, mpo_w_k,
        bra_sigma_k, ket_sigma_k)`` open for active tensor ``k``.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        left = []
        env = np.ones((1, 1, mpo[0].shape[0], self.dims[0], self.dims[0]), dtype=dtype)
        left.append(env)
        nprefix = self.npairs if self.has_terminal_tensor else self.npairs - 1
        for i, tensor in enumerate(self.tensors[:nprefix]):
            env = self._advance_left_environment(env, mpo[i], tensor)
            left.append(env)
        return left

    def _advance_left_environment(self, env, mpo_site, tensor):
        package = _sparse_mpo_site_package(mpo_site)
        if package is not None:
            return _advance_left_environment_sparse(
                env,
                tensor,
                package,
                np.asarray(mpo_site).shape[1],
            )
        return np.einsum(
            "bkmxy,mnxy,bxuc,kyvd->cdnuv",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    def _right_local_environments(self, mpo):
        """
        Suffix contractions for LETTA one-site MPO environments.

        ``right[k]`` leaves ``(bra_alpha_{k+1}, ket_alpha_{k+1},
        mpo_w_{k+2}, bra_sigma_{k+1}, ket_sigma_{k+1})`` open for active
        tensor ``k``.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        right = [None] * self.nlocal_tensors
        if self.has_terminal_tensor:
            env = self._terminal_right_environment(mpo, dtype=dtype)
            right[self.npairs - 1] = env
        else:
            env = np.ones((1, 1, mpo[-1].shape[1], self.dims[-1], self.dims[-1]), dtype=dtype)
            right[-1] = env
        for i in range(self.npairs - 1, 0, -1):
            tensor = self.tensors[i]
            env = self._advance_right_environment(env, mpo[i + 1], tensor)
            right[i - 1] = env
        return right

    def _terminal_right_environment(self, mpo=None, *, dtype=None):
        terminal = self.tensors[-1]
        if dtype is None:
            dtype = terminal.dtype
        right_mpo_dim = 1 if mpo is None else mpo[-1].shape[1]
        env = np.zeros(
            (terminal.shape[1], terminal.shape[1], right_mpo_dim, self.dims[-1], self.dims[-1]),
            dtype=dtype,
        )
        env[:, :, 0, :, :] = np.einsum("uc,vd->cduv", terminal.conj(), terminal, optimize=True)
        return env

    def _advance_right_environment(self, env, mpo_site, tensor):
        package = _sparse_mpo_site_package(mpo_site)
        if package is not None:
            return _advance_right_environment_sparse(
                env,
                tensor,
                package,
                np.asarray(mpo_site).shape[0],
            )
        return np.einsum(
            "cdnuv,mnuv,bxuc,kyvd->bkmxy",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    def _left_metric_environments(self):
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        left = []
        env = np.ones((1, 1, 1, self.dims[0], self.dims[0]), dtype=dtype)
        left.append(env)
        for tensor in self.tensors[:-1]:
            env = self._advance_left_metric_environment(env, tensor)
            left.append(env)
        return left

    def _advance_left_metric_environment(self, env, tensor):
        diagonal = np.einsum("bkxx->bkx", env[:, :, 0], optimize=True)
        advanced = np.einsum(
            "bkx,bxuc,kxvd->cduv",
            diagonal,
            tensor.conj(),
            tensor,
            optimize=True,
        )
        return advanced[:, :, None, :, :]

    def _right_metric_environments(self):
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        right = [None] * self.nlocal_tensors
        if self.has_terminal_tensor:
            env = self._terminal_right_environment(dtype=dtype)
            right[self.npairs - 1] = env
        else:
            env = np.ones((1, 1, 1, self.dims[-1], self.dims[-1]), dtype=dtype)
            right[-1] = env
        for i in range(self.npairs - 1, 0, -1):
            env = self._advance_right_metric_environment(env, self.tensors[i])
            right[i - 1] = env
        return right

    def _advance_right_metric_environment(self, env, tensor):
        diagonal = np.einsum("cduu->cdu", env[:, :, 0], optimize=True)
        advanced = np.einsum(
            "cdu,bxuc,kyud->bkxy",
            diagonal,
            tensor.conj(),
            tensor,
            optimize=True,
        )
        return advanced[:, :, None, :, :]

    def _local_effective_from_environments(self, mpo, tensor_index, left_envs, right_envs):
        tensor_index = int(tensor_index)
        shape = self.tensors[tensor_index].shape
        heff = np.einsum(
            "bkmxy,mpxy,pnuv,cdnuv->bxuckyvd",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            optimize=True,
        )
        return heff.reshape(int(np.prod(shape)), int(np.prod(shape)))

    def _terminal_effective_from_environment(self, mpo, left_envs):
        terminal_index = self.npairs
        terminal = self.tensors[terminal_index]
        heff = np.einsum(
            "cdmxy,mnxy->xcyd",
            left_envs[terminal_index],
            mpo[-1],
            optimize=True,
        )
        return heff.reshape(terminal.size, terminal.size)

    def _terminal_metric_from_environment(self, metric_left):
        terminal_index = self.npairs
        terminal = self.tensors[terminal_index]
        left = metric_left[terminal_index]
        dim, bond_dim = terminal.shape
        metric = np.zeros((dim, bond_dim, dim, bond_dim), dtype=left.dtype)
        left = left[:, :, 0]
        for site_state in range(dim):
            metric[site_state, :, site_state, :] = left[:, :, site_state, site_state]
        return metric.reshape(terminal.size, terminal.size)

    def _apply_local_effective_from_environments(self, mpo, tensor_index, left_envs, right_envs, vector):
        tensor_index = int(tensor_index)
        theta = np.asarray(vector).reshape(self.tensors[tensor_index].shape)
        out = np.einsum(
            "bkmxy,mpxy,pnuv,cdnuv,kyvd->bxuc",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            theta,
            optimize=True,
        )
        return out.reshape(-1)

    def _apply_local_effective_batch_from_environments(self, mpo, tensor_index, left_envs, right_envs, vectors):
        tensor_index = int(tensor_index)
        vectors = np.asarray(vectors)
        theta = vectors.reshape((vectors.shape[0],) + self.tensors[tensor_index].shape)
        out = np.einsum(
            "bkmxy,mpxy,pnuv,cdnuv,qkyvd->qbxuc",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            theta,
            optimize=True,
        )
        return out.reshape(vectors.shape[0], -1)

    def _apply_local_metric_from_environments(self, tensor_index, metric_left, metric_right, vector):
        tensor_index = int(tensor_index)
        theta = np.asarray(vector).reshape(self.tensors[tensor_index].shape)
        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        left_bond, left_ket, left_mpo, di, di_ket = left.shape
        right_bond, right_ket, right_mpo, dj, dj_ket = right.shape
        if left_mpo != 1 or right_mpo != 1 or di != di_ket or dj != dj_ket:
            metric = self._local_metric_from_environments(tensor_index, metric_left, metric_right)
            return metric @ np.asarray(vector).reshape(-1)
        left_diag = np.einsum("bkxx->bkx", left[:, :, 0], optimize=True)
        right_diag = np.einsum("cduu->cdu", right[:, :, 0], optimize=True)
        out = np.einsum(
            "bkx,cdu,kxud->bxuc",
            left_diag,
            right_diag,
            theta,
            optimize=True,
        )
        return out.reshape(-1)

    def _apply_local_metric_batch_from_environments(self, tensor_index, metric_left, metric_right, vectors):
        tensor_index = int(tensor_index)
        vectors = np.asarray(vectors)
        theta = vectors.reshape((vectors.shape[0],) + self.tensors[tensor_index].shape)
        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        left_bond, left_ket, left_mpo, di, di_ket = left.shape
        right_bond, right_ket, right_mpo, dj, dj_ket = right.shape
        if left_mpo != 1 or right_mpo != 1 or di != di_ket or dj != dj_ket:
            metric = self._local_metric_from_environments(tensor_index, metric_left, metric_right)
            return vectors @ metric.T
        left_diag = np.einsum("bkxx->bkx", left[:, :, 0], optimize=True)
        right_diag = np.einsum("cduu->cdu", right[:, :, 0], optimize=True)
        out = np.einsum(
            "bkx,cdu,qkxud->qbxuc",
            left_diag,
            right_diag,
            theta,
            optimize=True,
        )
        return out.reshape(vectors.shape[0], -1)

    def _solve_one_site_mpo_in_support(self, mpo, tensor_index, left_envs, right_envs, metric_left, metric_right, mask):
        tensor_index = int(tensor_index)
        local_dim = int(np.prod(self.tensors[tensor_index].shape))
        allowed = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
        if allowed.size == 0:
            raise ValueError("symmetry/support mask removes every local tensor entry.")
        dtype = np.result_type(
            self.tensors[tensor_index].dtype,
            *[site.dtype for site in mpo],
            metric_left[tensor_index].dtype,
            metric_right[tensor_index].dtype,
            complex,
        )
        s_cols = []
        shape = self.tensors[tensor_index].shape
        coords = np.asarray(np.unravel_index(allowed, shape)).T
        left = left_envs[tensor_index]
        right = right_envs[tensor_index]
        w_left = mpo[tensor_index]
        w_right = mpo[tensor_index + 1]
        sparse_entries = _two_site_mpo_sparse_entries(w_left, w_right)
        if _support_heff_sparse_numba is not None and sparse_entries is not None:
            entry_starts, entry_m, entry_n, entry_values = sparse_entries
            heff = _support_heff_sparse_numba(
                coords.astype(np.int64, copy=False),
                np.asarray(left),
                np.asarray(right),
                entry_starts,
                entry_m,
                entry_n,
                entry_values,
            ).astype(dtype, copy=False)
        elif _support_heff_indexed_numba is not None:
            heff = _support_heff_indexed_numba(
                coords.astype(np.int64, copy=False),
                np.asarray(left),
                np.asarray(right),
                np.asarray(w_left),
                np.asarray(w_right),
            ).astype(dtype, copy=False)
        else:
            heff = np.empty((allowed.size, allowed.size), dtype=dtype)
            two_site_mpo = {}
            for x in range(shape[1]):
                for y in range(shape[1]):
                    for u in range(shape[2]):
                        for v in range(shape[2]):
                            block = w_left[:, :, x, y] @ w_right[:, :, u, v]
                            if np.any(np.abs(block) > 0):
                                two_site_mpo[(x, y, u, v)] = block

            for row, (bra_left, bra_i, bra_j, bra_right) in enumerate(coords):
                for col, (ket_left, ket_i, ket_j, ket_right) in enumerate(coords):
                    block = two_site_mpo.get((bra_i, ket_i, bra_j, ket_j))
                    if block is None:
                        heff[row, col] = 0
                        continue
                    heff[row, col] = (
                        left[bra_left, ket_left, :, bra_i, ket_i]
                        @ block
                        @ right[bra_right, ket_right, :, bra_j, ket_j]
                    )

        mleft = metric_left[tensor_index]
        mright = metric_right[tensor_index]
        if (
            mleft.shape[2] == 1
            and mright.shape[2] == 1
            and mleft.shape[3] == mleft.shape[4]
            and mright.shape[3] == mright.shape[4]
        ):
            heff = 0.5 * (heff + heff.conj().T)
            blocks = _metric_blocks_from_support(coords, mleft, mright)
            energy, reduced_vector = _lowest_generalized_eigenpair_from_metric_blocks(heff, blocks)
            vector = np.zeros(local_dim, dtype=np.result_type(reduced_vector.dtype, dtype))
            vector[allowed] = reduced_vector
            return energy, vector
        else:
            batch_size = min(_DEFAULT_SUPPORT_BATCH_SIZE, allowed.size)
            for start in range(0, allowed.size, batch_size):
                batch = allowed[start:start + batch_size]
                vectors = np.zeros((batch.size, local_dim), dtype=dtype)
                vectors[np.arange(batch.size), batch] = 1
                s_cols.append(
                    self._apply_local_metric_batch_from_environments(
                        tensor_index, metric_left, metric_right, vectors
                    )[:, allowed].T
                )
            seff = np.concatenate(s_cols, axis=1)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        energy, reduced_vector = _lowest_generalized_eigenpair(heff, seff)
        vector = np.zeros(local_dim, dtype=np.result_type(reduced_vector.dtype, dtype))
        vector[allowed] = reduced_vector
        return energy, vector

    def _solve_terminal_mpo_with_environments(self, mpo, left_envs, metric_left):
        terminal_index = self.npairs
        local_dim = int(np.prod(self.tensors[terminal_index].shape))
        heff = self._terminal_effective_from_environment(mpo, left_envs)
        seff = self._terminal_metric_from_environment(metric_left)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)

        local_mask = self.local_masks[terminal_index]
        if local_mask is None:
            return _lowest_generalized_eigenpair(heff, seff)

        allowed = np.flatnonzero(np.asarray(local_mask, dtype=bool).reshape(-1))
        if allowed.size == 0:
            raise ValueError("symmetry/support mask removes every terminal tensor entry.")
        energy, reduced_vector = _lowest_generalized_eigenpair(
            heff[np.ix_(allowed, allowed)],
            seff[np.ix_(allowed, allowed)],
        )
        vector = np.zeros(local_dim, dtype=np.result_type(reduced_vector.dtype, heff.dtype))
        vector[allowed] = reduced_vector
        return energy, vector

    def _local_metric_from_environments(self, tensor_index, metric_left, metric_right):
        """
        Build the one-site overlap matrix for the identity MPO.

        The identity MPO enforces diagonal physical bra/ket indices, so the
        metric is an outer product of the left and right identity environments
        instead of a general four-tensor contraction.
        """
        tensor_index = int(tensor_index)
        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        shape = self.tensors[tensor_index].shape
        left_bond, left_ket, left_mpo, di, di_ket = left.shape
        right_bond, right_ket, right_mpo, dj, dj_ket = right.shape
        if left_mpo != 1 or right_mpo != 1 or di != di_ket or dj != dj_ket:
            return self._local_effective_from_environments(
                self.identity_mpo(), tensor_index, metric_left, metric_right
            )

        metric = np.zeros(
            (left_bond, di, dj, right_bond, left_ket, di, dj, right_ket),
            dtype=np.result_type(left.dtype, right.dtype),
        )
        left = left[:, :, 0]
        right = right[:, :, 0]
        for si in range(di):
            left_block = left[:, :, si, si]
            for sj in range(dj):
                right_block = right[:, :, sj, sj]
                metric[:, si, sj, :, :, si, sj, :] = (
                    left_block[:, None, :, None] * right_block[None, :, None, :]
                )
        dim = int(np.prod(shape))
        return metric.reshape(dim, dim)

    def _metric_basis_from_environments(self, tensor_index, metric_left, metric_right, *, metric_tol=1e-12):
        tensor_index = int(tensor_index)
        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        shape = self.tensors[tensor_index].shape
        left_bond, left_ket, left_mpo, di, di_ket = left.shape
        right_bond, right_ket, right_mpo, dj, dj_ket = right.shape
        if left_mpo != 1 or right_mpo != 1 or di != di_ket or dj != dj_ket:
            return _metric_basis(self._local_metric_from_environments(tensor_index, metric_left, metric_right))

        left = left[:, :, 0]
        right = right[:, :, 0]
        left_eigs = []
        right_eigs = []
        max_metric_eval = 0.0
        for si in range(di):
            block = 0.5 * (left[:, :, si, si] + left[:, :, si, si].conj().T)
            vals, vecs = linalg.eigh(block, check_finite=False)
            keep = vals > metric_tol * max(1.0, float(np.max(np.abs(vals))) if vals.size else 0.0)
            vals, vecs = vals[keep], vecs[:, keep]
            left_eigs.append((vals, vecs))
        for sj in range(dj):
            block = 0.5 * (right[:, :, sj, sj] + right[:, :, sj, sj].conj().T)
            vals, vecs = linalg.eigh(block, check_finite=False)
            keep = vals > metric_tol * max(1.0, float(np.max(np.abs(vals))) if vals.size else 0.0)
            vals, vecs = vals[keep], vecs[:, keep]
            right_eigs.append((vals, vecs))
        for vals_l, _ in left_eigs:
            if not vals_l.size:
                continue
            for vals_r, _ in right_eigs:
                if vals_r.size:
                    max_metric_eval = max(max_metric_eval, float(np.max(vals_l) * np.max(vals_r)))

        threshold = metric_tol * max(1.0, max_metric_eval)
        columns = []
        for si, (vals_l, vecs_l) in enumerate(left_eigs):
            for sj, (vals_r, vecs_r) in enumerate(right_eigs):
                for il, val_l in enumerate(vals_l):
                    for ir, val_r in enumerate(vals_r):
                        metric_val = float(val_l * val_r)
                        if metric_val <= threshold:
                            continue
                        column = np.zeros(shape, dtype=np.result_type(vecs_l.dtype, vecs_r.dtype))
                        column[:, si, sj, :] = (
                            vecs_l[:, il, None] * vecs_r[None, :, ir] / np.sqrt(metric_val)
                        )
                        columns.append(column.reshape(-1))
        if not columns:
            raise ValueError("Effective overlap metric is numerically singular.")
        return np.column_stack(columns)

    def _solve_one_site_mpo(self, mpo, tensor_index):
        heff = self.local_effective_matrix(mpo, tensor_index)
        seff = self.local_effective_matrix(self.identity_mpo(), tensor_index)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        return _lowest_generalized_eigenpair(heff, seff)

    def _solve_one_site_mpo_with_environments(
        self,
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
        *,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        tensor_index = int(tensor_index)
        if self.has_terminal_tensor and tensor_index == self.npairs:
            return self._solve_terminal_mpo_with_environments(mpo, left_envs, metric_left)

        local_mask = self.local_masks[tensor_index]
        if local_mask is not None:
            return self._solve_one_site_mpo_in_support(
                mpo,
                tensor_index,
                left_envs,
                right_envs,
                metric_left,
                metric_right,
                local_mask,
            )
        basis = self._metric_basis_from_environments(tensor_index, metric_left, metric_right)
        local_dim, reduced_dim = basis.shape
        dtype = np.result_type(basis.dtype, *[site.dtype for site in mpo])
        solver = _validate_local_solver(local_solver)
        if matrix_free_threshold is None:
            matrix_free_threshold = _DEFAULT_MATRIX_FREE_LOCAL_DIM
        dense_bytes = local_dim * local_dim * np.dtype(dtype).itemsize
        memory_limited = (
            matrix_free_memory_limit is not None
            and dense_bytes > int(matrix_free_memory_limit)
        )
        use_matrix_free = solver == "matrix_free" or (
            solver == "auto"
            and (local_dim > int(matrix_free_threshold) or memory_limited)
        )

        if use_matrix_free:
            current = self.tensors[tensor_index].reshape(-1)
            v0 = basis.conj().T @ current
            v0_norm = np.linalg.norm(v0)
            if v0_norm > 1e-14:
                v0 = v0 / v0_norm
            else:
                v0 = None

            def matvec(coeff):
                ket = basis @ coeff
                applied = self._apply_local_effective_from_environments(
                    mpo, tensor_index, left_envs, right_envs, ket
                )
                return basis.conj().T @ applied

            operator = LinearOperator(
                (reduced_dim, reduced_dim),
                matvec=matvec,
                dtype=dtype,
            )
            try:
                evals, evecs = eigsh(
                    operator,
                    k=1,
                    which="SA",
                    tol=matrix_free_tol,
                    maxiter=matrix_free_maxiter or max(1000, 20 * reduced_dim),
                    v0=v0,
                )
                energy = float(np.real(evals[0]))
                reduced_vector = evecs[:, 0]
            except Exception as exc:
                if reduced_dim > int(matrix_free_fallback_dim):
                    raise RuntimeError(
                        "Matrix-free LETTA local eigensolve failed and the "
                        f"reduced dimension ({reduced_dim}) is too large for "
                        "the dense fallback. Increase matrix_free_maxiter, "
                        "loosen matrix_free_tol, or use local_solver='dense' "
                        "for a smaller bond dimension."
                    ) from exc
                columns = [matvec(np.eye(reduced_dim, dtype=dtype)[:, j]) for j in range(reduced_dim)]
                reduced_h = np.column_stack(columns)
                reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
                energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
        else:
            heff = self._local_effective_from_environments(mpo, tensor_index, left_envs, right_envs)
            heff = 0.5 * (heff + heff.conj().T)
            reduced_h = basis.conj().T @ heff @ basis
            energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
        vector = basis @ reduced_vector
        return energy, vector

    def _partial_amplitude(self, tensor_index, config, left, right):
        if tensor_index == 0:
            left_coeff = 1.0
        else:
            vec = self.tensors[0][0, config[0], config[1], :]
            for i in range(1, tensor_index):
                vec = vec @ self.tensors[i][:, config[i], config[i + 1], :]
            left_coeff = vec[left]

        if tensor_index == self.nbonds - 1:
            if self.has_terminal_tensor:
                right_coeff = self.tensors[-1][config[-1], right]
            else:
                right_coeff = 1.0
        else:
            if self.has_terminal_tensor:
                rvec = self.tensors[-1][config[-1], :]
                last = self.npairs - 1
                for i in range(last, tensor_index, -1):
                    rvec = self.tensors[i][:, config[i], config[i + 1], :] @ rvec
            else:
                last = self.nbonds - 1
                rvec = self.tensors[last][:, config[last], config[last + 1], 0]
                for i in range(last - 1, tensor_index, -1):
                    rvec = self.tensors[i][:, config[i], config[i + 1], :] @ rvec
            right_coeff = rvec[right]

        return left_coeff * right_coeff

    def _one_site_projector(self, tensor_index):
        tensor = self.tensors[tensor_index]
        if self.has_terminal_tensor and tensor_index == self.npairs:
            dim, bond_dim = tensor.shape
            nrow = int(np.prod(self.dims))
            ncol = dim * bond_dim
            projector = np.zeros((nrow, ncol), dtype=np.result_type(*[t.dtype for t in self.tensors], complex))
            for flat, config in enumerate(np.ndindex(*self.dims)):
                vec = self.tensors[0][0, config[0], config[1], :]
                for i in range(1, self.npairs):
                    vec = vec @ self.tensors[i][:, config[i], config[i + 1], :]
                site_state = config[-1]
                start = site_state * bond_dim
                projector[flat, start:start + bond_dim] = vec
            return projector

        left_dim, di, dj, right_dim = tensor.shape
        nrow = int(np.prod(self.dims))
        ncol = left_dim * di * dj * right_dim
        projector = np.zeros((nrow, ncol), dtype=np.result_type(*[t.dtype for t in self.tensors], complex))

        for flat, config in enumerate(np.ndindex(*self.dims)):
            si = config[tensor_index]
            sj = config[tensor_index + 1]
            for left in range(left_dim):
                for right in range(right_dim):
                    col = (((left * di + si) * dj + sj) * right_dim + right)
                    projector[flat, col] = self._partial_amplitude(tensor_index, config, left, right)
        return projector

    def _solve_one_site(self, tensor_index):
        if self.hamiltonian is None:
            raise ValueError("dense hamiltonian is not available; use optimize_tensor_mpo(mpo, tensor_index).")
        projector = self._one_site_projector(tensor_index)
        heff = projector.conj().T @ self.hamiltonian @ projector
        if self.overlap is None:
            seff = projector.conj().T @ projector
        else:
            seff = projector.conj().T @ self.overlap @ projector
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        return _lowest_generalized_eigenpair(heff, seff)

    def optimize_tensor(self, tensor_index):
        """
        Optimize one tied tensor with all other tensors fixed.
        """
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        local_energy, vector = self._solve_one_site(tensor_index)
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks[tensor_index] is not None:
            self.tensors[tensor_index] = np.where(
                self.local_masks[tensor_index],
                self.tensors[tensor_index],
                0,
            )
        self.normalize()
        return {"tensor": tensor_index, "local_energy": float(local_energy)}

    def optimize_tensor_mpo(
        self,
        mpo,
        tensor_index,
        *,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        """
        Optimize one tied tensor using an MPO-contracted local Hamiltonian.
        """
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        left_envs = self._left_local_environments(mpo)
        right_envs = self._right_local_environments(mpo)
        metric_left = self._left_metric_environments()
        metric_right = self._right_metric_environments()
        local_energy, vector = self._solve_one_site_mpo_with_environments(
            mpo,
            tensor_index,
            left_envs,
            right_envs,
            metric_left,
            metric_right,
            local_solver=local_solver,
            matrix_free_threshold=matrix_free_threshold,
            matrix_free_memory_limit=matrix_free_memory_limit,
            matrix_free_tol=matrix_free_tol,
            matrix_free_maxiter=matrix_free_maxiter,
            matrix_free_fallback_dim=matrix_free_fallback_dim,
        )
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks[tensor_index] is not None:
            self.tensors[tensor_index] = np.where(
                self.local_masks[tensor_index],
                self.tensors[tensor_index],
                0,
            )
        return {"tensor": tensor_index, "local_energy": float(local_energy)}

    def _optimize_tensor_mpo_with_environments(
        self,
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        local_energy, vector = self._solve_one_site_mpo_with_environments(
            mpo,
            tensor_index,
            left_envs,
            right_envs,
            metric_left,
            metric_right,
            local_solver=local_solver,
            matrix_free_threshold=matrix_free_threshold,
            matrix_free_memory_limit=matrix_free_memory_limit,
            matrix_free_tol=matrix_free_tol,
            matrix_free_maxiter=matrix_free_maxiter,
            matrix_free_fallback_dim=matrix_free_fallback_dim,
        )
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks[tensor_index] is not None:
            self.tensors[tensor_index] = np.where(
                self.local_masks[tensor_index],
                self.tensors[tensor_index],
                0,
            )
        return {"tensor": int(tensor_index), "local_energy": float(local_energy)}

    def sweep(self, direction="lr", operator=None):
        """
        Perform one one-site variational sweep over tied tensors.
        """
        if operator is not None:
            if self._looks_like_mpo(operator):
                return self.sweep_mpo(operator, direction=direction)
            old_hamiltonian = self.hamiltonian
            self.hamiltonian = self._validate_dense_operator(operator)
            try:
                return self.sweep(direction=direction)
            finally:
                self.hamiltonian = old_hamiltonian

        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        indices = range(self.nlocal_tensors)
        if direction == "rl":
            indices = reversed(list(indices))
        return [self.optimize_tensor(i) for i in indices]

    def sweep_mpo(
        self,
        mpo,
        direction="lr",
        *,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        """
        Perform one one-site sweep using MPO-contracted local Hamiltonians.
        """
        mpo = self._validate_mpo(mpo)
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        updates = []
        package = LETTAOperatorPackage.for_sweep(self, mpo, direction)

        if direction == "lr":
            for i in range(self.nlocal_tensors):
                updates.append(
                    self._optimize_tensor_mpo_with_environments(
                        mpo,
                        i,
                        package.left_envs,
                        package.right_envs,
                        package.metric_left,
                        package.metric_right,
                        local_solver=local_solver,
                        matrix_free_threshold=matrix_free_threshold,
                        matrix_free_memory_limit=matrix_free_memory_limit,
                        matrix_free_tol=matrix_free_tol,
                        matrix_free_maxiter=matrix_free_maxiter,
                        matrix_free_fallback_dim=matrix_free_fallback_dim,
                    )
                )
                package.advance_after_update(i)
        else:
            for i in reversed(range(self.nlocal_tensors)):
                updates.append(
                    self._optimize_tensor_mpo_with_environments(
                        mpo,
                        i,
                        package.left_envs,
                        package.right_envs,
                        package.metric_left,
                        package.metric_right,
                        local_solver=local_solver,
                        matrix_free_threshold=matrix_free_threshold,
                        matrix_free_memory_limit=matrix_free_memory_limit,
                        matrix_free_tol=matrix_free_tol,
                        matrix_free_maxiter=matrix_free_maxiter,
                        matrix_free_fallback_dim=matrix_free_fallback_dim,
                    )
                )
                package.advance_after_update(i)

        return updates

    def run(
        self,
        operator=None,
        *,
        nsweeps=4,
        start_direction="lr",
        alternate=True,
        tol=1e-10,
        verbose=0,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        """
        Run one-site LETTA variational sweeps.

        ``operator`` may be omitted to use the stored dense Hamiltonian, or may
        be supplied as a dense matrix or MPO. Existing ``run_mpo`` callers are
        still supported, but ``run(mpo, nsweeps=...)`` is the preferred form.
        """
        if operator is not None:
            if self._looks_like_mpo(operator):
                return self.run_mpo(
                    operator,
                    nsweeps=nsweeps,
                    start_direction=start_direction,
                    alternate=alternate,
                    tol=tol,
                    verbose=verbose,
                    local_solver=local_solver,
                    matrix_free_threshold=matrix_free_threshold,
                    matrix_free_memory_limit=matrix_free_memory_limit,
                    matrix_free_tol=matrix_free_tol,
                    matrix_free_maxiter=matrix_free_maxiter,
                    matrix_free_fallback_dim=matrix_free_fallback_dim,
                )
            old_hamiltonian = self.hamiltonian
            self.hamiltonian = self._validate_dense_operator(operator)
            try:
                return self.run(
                    nsweeps=nsweeps,
                    start_direction=start_direction,
                    alternate=alternate,
                    tol=tol,
                    verbose=verbose,
                )
            finally:
                self.hamiltonian = old_hamiltonian

        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep(direction)
            energy = updates[-1]["local_energy"] if updates else self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"letta sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return LETTAResult(
            energy=self.energy,
            tensors=[tensor.copy() for tensor in self.tensors],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )

    def run_mpo(
        self,
        mpo,
        *,
        nsweeps=4,
        start_direction="lr",
        alternate=True,
        tol=1e-10,
        verbose=0,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
    ):
        """
        Run one-site LETTA sweeps using MPO-contracted local Hamiltonians.

        ``local_solver="auto"`` materializes the dense local Hamiltonian for
        small problems and switches to a matrix-free ARPACK solve when the
        dense local matrix would exceed ``matrix_free_memory_limit`` or the
        local dimension exceeds ``matrix_free_threshold``. Use
        ``local_solver="matrix_free"`` to force the MPO-action path.
        """
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        mpo = self._validate_mpo(mpo)
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep_mpo(
                mpo,
                direction,
                local_solver=local_solver,
                matrix_free_threshold=matrix_free_threshold,
                matrix_free_memory_limit=matrix_free_memory_limit,
                matrix_free_tol=matrix_free_tol,
                matrix_free_maxiter=matrix_free_maxiter,
                matrix_free_fallback_dim=matrix_free_fallback_dim,
            )
            energy = updates[-1]["local_energy"] if updates else self.expectation_mpo(mpo)
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"letta-mpo sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return LETTAResult(
            energy=self.energy,
            tensors=[tensor.copy() for tensor in self.tensors],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )

    def fit_state(self, state, *, nsweeps=4, ridge=1e-12):
        """
        Alternating least-squares fit to a dense target state.
        """
        target = np.asarray(state).reshape(-1)
        if target.size != int(np.prod(self.dims)):
            raise ValueError("target state size does not match product dimension.")
        for _ in range(int(nsweeps)):
            for direction in ("lr", "rl"):
                indices = range(self.nbonds)
                if direction == "rl":
                    indices = reversed(list(indices))
                for i in indices:
                    projector = self._one_site_projector(i)
                    normal = projector.conj().T @ projector
                    rhs = projector.conj().T @ target
                    if ridge:
                        normal = normal + float(ridge) * np.eye(normal.shape[0], dtype=normal.dtype)
                    try:
                        vector = linalg.solve(normal, rhs, assume_a="pos")
                    except Exception:
                        vector = linalg.lstsq(projector, target)[0]
                    self.tensors[i] = vector.reshape(self.tensors[i].shape)
            self.normalize()
        return self
