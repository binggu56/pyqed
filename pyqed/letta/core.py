#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational LETTA sweeps.

This module is intentionally small and dense-reference based. It keeps the
local effective Hamiltonian construction explicit for validating LETTA
variational updates before replacing dense projectors by cached environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

from ._support_kernels import (
    apply_support_hamiltonian as _support_kernel_apply,
    assemble_support_hamiltonian as _support_kernel_assemble,
    native_available as _support_kernel_native_available,
)

from .conditional_gauge import apply_conditional_gauges as _apply_conditional_gauges

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
_DEFAULT_ABELIAN_SPARSE_SUPPORT_DIM = 1024
_DEFAULT_DIRECT_REDUCED_HEFF_DIM = 2048
_DEFAULT_ACTIVE_WARM_START_DIM = 256
_DEFAULT_SUPPORT_BATCH_SIZE = 64
_DEFAULT_MASKED_DENSE_SLICE_LOCAL_DIM = 256
_SPARSE_MPO_DENSITY_LIMIT = 0.20
_SPARSE_MPO_MIN_BOND_PRODUCT = 3000
_SPARSE_MPO_SITE_CACHE_MAXSIZE = 128
_SPARSE_MPO_SITE_CACHE = {}
_TWO_SITE_MPO_ENTRY_CACHE_MAXSIZE = 128
_TWO_SITE_MPO_ENTRY_CACHE = {}
_TWO_SITE_MPO_TRANSITION_CACHE_MAXSIZE = 32
_TWO_SITE_MPO_TRANSITION_CACHE = {}
_TWO_SITE_MPO_PHYSICAL_BLOCK_LIMIT = 262_144
_TWO_SITE_MPO_TRANSITION_CHUNK = 128
_EINSUM_PATH_CACHE_MAXSIZE = 256
_EINSUM_PATH_CACHE = {}


def _cached_einsum(subscripts, *operands):
    """Contract with a shape-keyed NumPy path cache."""
    key = (
        str(subscripts),
        tuple(tuple(int(size) for size in operand.shape) for operand in operands),
    )
    path = _EINSUM_PATH_CACHE.get(key)
    if path is None:
        path = np.einsum_path(subscripts, *operands, optimize=True)[0]
        if len(_EINSUM_PATH_CACHE) >= _EINSUM_PATH_CACHE_MAXSIZE:
            _EINSUM_PATH_CACHE.pop(next(iter(_EINSUM_PATH_CACHE)))
        _EINSUM_PATH_CACHE[key] = path
    return np.einsum(subscripts, *operands, optimize=path)


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

    @njit(nogil=True, cache=False)
    def _apply_support_heff_transitions_numba(
        coords,
        left,
        right,
        transition_bra_i,
        transition_ket_i,
        transition_bra_j,
        transition_ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
        group_starts,
        group_positions,
        vector,
    ):
        nallowed = coords.shape[0]
        out = np.zeros(nallowed, dtype=np.complex128)
        dj = right.shape[3]
        for transition in range(transition_bra_i.shape[0]):
            bra_i = transition_bra_i[transition]
            ket_i = transition_ket_i[transition]
            bra_j = transition_bra_j[transition]
            ket_j = transition_ket_j[transition]
            row_group = bra_i * dj + bra_j
            col_group = ket_i * dj + ket_j
            row_start = group_starts[row_group]
            row_stop = group_starts[row_group + 1]
            col_start = group_starts[col_group]
            col_stop = group_starts[col_group + 1]
            if row_start == row_stop or col_start == col_stop:
                continue
            entry_start = entry_starts[transition]
            entry_stop = entry_starts[transition + 1]
            for row_ptr in range(row_start, row_stop):
                row = group_positions[row_ptr]
                bra_left = coords[row, 0]
                bra_right = coords[row, 3]
                row_value = 0.0 + 0.0j
                for col_ptr in range(col_start, col_stop):
                    col = group_positions[col_ptr]
                    ket_value = vector[col]
                    if ket_value == 0:
                        continue
                    ket_left = coords[col, 0]
                    ket_right = coords[col, 3]
                    element = 0.0 + 0.0j
                    for entry in range(entry_start, entry_stop):
                        m = entry_m[entry]
                        n = entry_n[entry]
                        element += (
                            left[bra_left, ket_left, m, bra_i, ket_i]
                            * entry_values[entry]
                            * right[bra_right, ket_right, n, bra_j, ket_j]
                        )
                    row_value += element * ket_value
                out[row] += row_value
        return out
else:
    _support_heff_sparse_numba = None
    _support_heff_indexed_numba = None
    _apply_support_heff_transitions_numba = None


def _two_site_mpo_sparse_entries(w_left, w_right):
    w_left = np.asarray(w_left)
    w_right = np.asarray(w_right)
    key = (id(w_left), id(w_right), w_left.shape, w_right.shape)
    if key in _TWO_SITE_MPO_ENTRY_CACHE:
        cached_left, cached_right, cached_result = _TWO_SITE_MPO_ENTRY_CACHE[key]
        if cached_left is w_left and cached_right is w_right:
            return cached_result

    physical_blocks = int(w_left.shape[2]) * int(w_left.shape[3]) * int(w_right.shape[2]) * int(w_right.shape[3])
    if physical_blocks > _TWO_SITE_MPO_PHYSICAL_BLOCK_LIMIT:
        result = None
        _cache_two_site_mpo_entries(key, w_left, w_right, result)
        return result

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


@dataclass(frozen=True)
class _TwoSiteMPOTransitions:
    """Sparse physical transitions for a two-site MPO product."""

    di: int
    dj: int
    bra_i: np.ndarray
    ket_i: np.ndarray
    bra_j: np.ndarray
    ket_j: np.ndarray
    entry_starts: np.ndarray
    entry_m: np.ndarray
    entry_n: np.ndarray
    entry_values: np.ndarray

    @property
    def ntransitions(self):
        return int(self.bra_i.size)


def _two_site_mpo_sparse_transitions(w_left, w_right):
    """Return nonzero two-site MPO transitions without materializing d^4 blocks."""
    w_left = np.asarray(w_left)
    w_right = np.asarray(w_right)
    key = (id(w_left), id(w_right), w_left.shape, w_right.shape)
    if key in _TWO_SITE_MPO_TRANSITION_CACHE:
        cached_left, cached_right, cached_result = _TWO_SITE_MPO_TRANSITION_CACHE[key]
        if cached_left is w_left and cached_right is w_right:
            return cached_result

    if w_left.ndim != 4 or w_right.ndim != 4:
        result = None
        _cache_two_site_mpo_transitions(key, w_left, w_right, result)
        return result
    if w_left.shape[1] != w_right.shape[0]:
        result = None
        _cache_two_site_mpo_transitions(key, w_left, w_right, result)
        return result
    if w_left.shape[2] != w_left.shape[3] or w_right.shape[2] != w_right.shape[3]:
        result = None
        _cache_two_site_mpo_transitions(key, w_left, w_right, result)
        return result

    shared_dim = int(w_left.shape[1])
    di = int(w_left.shape[2])
    dj = int(w_right.shape[2])
    phys_key_parts = []
    entry_m_parts = []
    entry_n_parts = []
    value_parts = []
    dtype = np.result_type(w_left.dtype, w_right.dtype, complex)
    for shared in range(shared_dim):
        left = w_left[:, shared, :, :]
        lm, lbra, lket = np.nonzero(left)
        if lm.size == 0:
            continue
        lvalues = left[lm, lbra, lket]

        right = w_right[shared, :, :, :]
        rn, rbra, rket = np.nonzero(right)
        if rn.size == 0:
            continue
        rvalues = right[rn, rbra, rket]

        right_count = int(rn.size)
        for start in range(0, int(lm.size), _TWO_SITE_MPO_TRANSITION_CHUNK):
            stop = min(start + _TWO_SITE_MPO_TRANSITION_CHUNK, int(lm.size))
            chunk = stop - start
            rep_lbra = np.repeat(lbra[start:stop], right_count)
            rep_lket = np.repeat(lket[start:stop], right_count)
            tile_rbra = np.tile(rbra, chunk)
            tile_rket = np.tile(rket, chunk)
            phys_key_parts.append(
                (
                    (((rep_lbra.astype(np.int64, copy=False) * di + rep_lket) * dj + tile_rbra) * dj)
                    + tile_rket
                )
            )
            entry_m_parts.append(np.repeat(lm[start:stop], right_count).astype(np.int64, copy=False))
            entry_n_parts.append(np.tile(rn, chunk).astype(np.int64, copy=False))
            value_parts.append(
                (
                    np.repeat(lvalues[start:stop], right_count)
                    * np.tile(rvalues, chunk)
                ).astype(dtype, copy=False)
            )

    if not phys_key_parts:
        result = _TwoSiteMPOTransitions(
            di=di,
            dj=dj,
            bra_i=np.empty(0, dtype=np.int64),
            ket_i=np.empty(0, dtype=np.int64),
            bra_j=np.empty(0, dtype=np.int64),
            ket_j=np.empty(0, dtype=np.int64),
            entry_starts=np.zeros(1, dtype=np.int64),
            entry_m=np.empty(0, dtype=np.int64),
            entry_n=np.empty(0, dtype=np.int64),
            entry_values=np.empty(0, dtype=dtype),
        )
        _cache_two_site_mpo_transitions(key, w_left, w_right, result)
        return result

    phys_keys = np.concatenate(phys_key_parts).astype(np.int64, copy=False)
    entry_m = np.concatenate(entry_m_parts).astype(np.int64, copy=False)
    entry_n = np.concatenate(entry_n_parts).astype(np.int64, copy=False)
    entry_values = np.concatenate(value_parts).astype(dtype, copy=False)

    order = np.argsort(phys_keys, kind="stable")
    phys_keys = phys_keys[order]
    entry_m = entry_m[order]
    entry_n = entry_n[order]
    entry_values = entry_values[order]
    unique_keys, starts = np.unique(phys_keys, return_index=True)
    starts = np.concatenate((starts.astype(np.int64, copy=False), np.asarray([phys_keys.size], dtype=np.int64)))

    decode = unique_keys.copy()
    ket_j = (decode % dj).astype(np.int64, copy=False)
    decode //= dj
    bra_j = (decode % dj).astype(np.int64, copy=False)
    decode //= dj
    ket_i = (decode % di).astype(np.int64, copy=False)
    decode //= di
    bra_i = decode.astype(np.int64, copy=False)

    result = _TwoSiteMPOTransitions(
        di=di,
        dj=dj,
        bra_i=bra_i,
        ket_i=ket_i,
        bra_j=bra_j,
        ket_j=ket_j,
        entry_starts=starts,
        entry_m=entry_m,
        entry_n=entry_n,
        entry_values=entry_values,
    )
    _cache_two_site_mpo_transitions(key, w_left, w_right, result)
    return result


def _cache_two_site_mpo_transitions(key, w_left, w_right, value):
    if key in _TWO_SITE_MPO_TRANSITION_CACHE:
        _TWO_SITE_MPO_TRANSITION_CACHE.pop(key)
    elif len(_TWO_SITE_MPO_TRANSITION_CACHE) >= _TWO_SITE_MPO_TRANSITION_CACHE_MAXSIZE:
        _TWO_SITE_MPO_TRANSITION_CACHE.pop(next(iter(_TWO_SITE_MPO_TRANSITION_CACHE)))
    _TWO_SITE_MPO_TRANSITION_CACHE[key] = (w_left, w_right, value)


def _support_heff_sparse_by_physical_blocks(
    coords,
    left,
    right,
    entry_starts,
    entry_m,
    entry_n,
    entry_values,
    *,
    dtype,
    atol=0.0,
    physical_groups=None,
):
    coords = np.asarray(coords, dtype=np.int64)
    nallowed = coords.shape[0]
    if nallowed == 0:
        return csr_matrix((0, 0), dtype=dtype)
    di = left.shape[3]
    dj = right.shape[3]
    if physical_groups is None:
        groups = {}
        for position, (_left, si, sj, _right) in enumerate(coords):
            groups.setdefault((int(si), int(sj)), []).append(position)
        groups = {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}
    else:
        groups = physical_groups

    rows = []
    cols = []
    data = []
    for bra_i in range(di):
        for ket_i in range(di):
            for bra_j in range(dj):
                row_positions = groups.get((bra_i, bra_j))
                if row_positions is None:
                    continue
                row_coords = coords[row_positions]
                row_left = row_coords[:, 0]
                row_right = row_coords[:, 3]
                for ket_j in range(dj):
                    block_index = (((bra_i * di + ket_i) * dj + bra_j) * dj + ket_j)
                    start = int(entry_starts[block_index])
                    stop = int(entry_starts[block_index + 1])
                    if start == stop:
                        continue
                    col_positions = groups.get((ket_i, ket_j))
                    if col_positions is None:
                        continue
                    col_coords = coords[col_positions]
                    col_left = col_coords[:, 0]
                    col_right = col_coords[:, 3]
                    block = np.zeros(
                        (row_positions.size, col_positions.size),
                        dtype=dtype,
                    )
                    for entry in range(start, stop):
                        m = int(entry_m[entry])
                        n = int(entry_n[entry])
                        block += (
                            entry_values[entry]
                            * left[row_left[:, None], col_left[None, :], m, bra_i, ket_i]
                            * right[row_right[:, None], col_right[None, :], n, bra_j, ket_j]
                        )
                    flat = block.reshape(-1)
                    if atol > 0.0:
                        nonzero = np.flatnonzero(np.abs(flat) > float(atol))
                    else:
                        nonzero = np.flatnonzero(flat != 0)
                    if nonzero.size == 0:
                        continue
                    block_rows = nonzero // col_positions.size
                    block_cols = nonzero % col_positions.size
                    rows.append(row_positions[block_rows])
                    cols.append(col_positions[block_cols])
                    data.append(flat[nonzero])

    if not data:
        return csr_matrix((nallowed, nallowed), dtype=dtype)
    return csr_matrix(
        (
            np.concatenate(data).astype(dtype, copy=False),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(nallowed, nallowed),
        dtype=dtype,
    )


def _support_heff_sparse_by_transitions(
    coords,
    left,
    right,
    transitions,
    *,
    dtype,
    atol=0.0,
    physical_groups=None,
):
    coords = np.asarray(coords, dtype=np.int64)
    nallowed = coords.shape[0]
    if nallowed == 0:
        return csr_matrix((0, 0), dtype=dtype)
    if transitions is None or transitions.ntransitions == 0:
        return csr_matrix((nallowed, nallowed), dtype=dtype)
    if physical_groups is None:
        groups = _physical_groups_from_coords(coords)
    else:
        groups = physical_groups

    rows = []
    cols = []
    data = []
    for transition in range(transitions.ntransitions):
        bra_i = int(transitions.bra_i[transition])
        ket_i = int(transitions.ket_i[transition])
        bra_j = int(transitions.bra_j[transition])
        ket_j = int(transitions.ket_j[transition])
        row_positions = groups.get((bra_i, bra_j))
        if row_positions is None:
            continue
        col_positions = groups.get((ket_i, ket_j))
        if col_positions is None:
            continue

        row_coords = coords[row_positions]
        col_coords = coords[col_positions]
        row_left = row_coords[:, 0]
        row_right = row_coords[:, 3]
        col_left = col_coords[:, 0]
        col_right = col_coords[:, 3]
        block = np.zeros((row_positions.size, col_positions.size), dtype=dtype)
        start = int(transitions.entry_starts[transition])
        stop = int(transitions.entry_starts[transition + 1])
        for entry in range(start, stop):
            m = int(transitions.entry_m[entry])
            n = int(transitions.entry_n[entry])
            block += (
                transitions.entry_values[entry]
                * left[row_left[:, None], col_left[None, :], m, bra_i, ket_i]
                * right[row_right[:, None], col_right[None, :], n, bra_j, ket_j]
            )

        flat = block.reshape(-1)
        if atol > 0.0:
            nonzero = np.flatnonzero(np.abs(flat) > float(atol))
        else:
            nonzero = np.flatnonzero(flat != 0)
        if nonzero.size == 0:
            continue
        block_rows = nonzero // col_positions.size
        block_cols = nonzero % col_positions.size
        rows.append(row_positions[block_rows])
        cols.append(col_positions[block_cols])
        data.append(flat[nonzero])

    if not data:
        return csr_matrix((nallowed, nallowed), dtype=dtype)
    return csr_matrix(
        (
            np.concatenate(data).astype(dtype, copy=False),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(nallowed, nallowed),
        dtype=dtype,
    )


def _physical_group_arrays(physical_groups, di, dj):
    starts = np.empty(int(di) * int(dj) + 1, dtype=np.int64)
    positions = []
    cursor = 0
    starts[0] = 0
    for si in range(int(di)):
        for sj in range(int(dj)):
            group = physical_groups.get((si, sj))
            if group is not None:
                group = np.asarray(group, dtype=np.int64)
                positions.append(group)
                cursor += int(group.size)
            starts[si * int(dj) + sj + 1] = cursor
    if positions:
        return starts, np.concatenate(positions).astype(np.int64, copy=False)
    return starts, np.empty(0, dtype=np.int64)


def _apply_support_heff_by_transitions(
    coords,
    left,
    right,
    transitions,
    vector,
    *,
    dtype,
    physical_groups=None,
):
    """Apply the support-space two-site effective Hamiltonian on the fly."""
    coords = np.asarray(coords, dtype=np.int64)
    vector = np.asarray(vector)
    nallowed = coords.shape[0]
    output_shape = (nallowed,) if vector.ndim == 1 else (nallowed, vector.shape[1])
    out = np.zeros(output_shape, dtype=np.result_type(dtype, vector.dtype))
    if nallowed == 0 or transitions is None or transitions.ntransitions == 0:
        return out
    if _support_kernel_native_available() or vector.ndim == 2:
        return _support_kernel_apply(
            coords,
            left,
            right,
            transitions.bra_i,
            transitions.ket_i,
            transitions.bra_j,
            transitions.ket_j,
            transitions.entry_starts,
            transitions.entry_m,
            transitions.entry_n,
            transitions.entry_values,
            vector,
        ).astype(out.dtype, copy=False)
    if physical_groups is None:
        groups = _physical_groups_from_coords(coords)
    else:
        groups = physical_groups
    if _apply_support_heff_transitions_numba is not None:
        group_starts, group_positions = _physical_group_arrays(groups, transitions.di, transitions.dj)
        return _apply_support_heff_transitions_numba(
            coords,
            np.asarray(left),
            np.asarray(right),
            transitions.bra_i,
            transitions.ket_i,
            transitions.bra_j,
            transitions.ket_j,
            transitions.entry_starts,
            transitions.entry_m,
            transitions.entry_n,
            transitions.entry_values,
            group_starts,
            group_positions,
            vector.astype(np.result_type(vector.dtype, complex), copy=False),
        ).astype(out.dtype, copy=False)

    for transition in range(transitions.ntransitions):
        bra_i = int(transitions.bra_i[transition])
        ket_i = int(transitions.ket_i[transition])
        bra_j = int(transitions.bra_j[transition])
        ket_j = int(transitions.ket_j[transition])
        row_positions = groups.get((bra_i, bra_j))
        if row_positions is None:
            continue
        col_positions = groups.get((ket_i, ket_j))
        if col_positions is None:
            continue
        ket_vector = vector[col_positions]
        if not np.any(ket_vector):
            continue

        row_coords = coords[row_positions]
        col_coords = coords[col_positions]
        row_left = row_coords[:, 0]
        row_right = row_coords[:, 3]
        col_left = col_coords[:, 0]
        col_right = col_coords[:, 3]
        block_out = np.zeros(row_positions.size, dtype=out.dtype)
        start = int(transitions.entry_starts[transition])
        stop = int(transitions.entry_starts[transition + 1])
        for entry in range(start, stop):
            m = int(transitions.entry_m[entry])
            n = int(transitions.entry_n[entry])
            block = (
                transitions.entry_values[entry]
                * left[row_left[:, None], col_left[None, :], m, bra_i, ket_i]
                * right[row_right[:, None], col_right[None, :], n, bra_j, ket_j]
            )
            block_out += block @ ket_vector
        out[row_positions] += block_out
    return out


def _metric_block_offsets(blocks):
    offsets = []
    cursor = 0
    dtype = None
    for _block_indices, basis in blocks:
        width = int(basis.shape[1])
        offsets.append((cursor, cursor + width))
        cursor += width
        dtype = basis.dtype if dtype is None else np.result_type(dtype, basis.dtype)
    return offsets, cursor, dtype


def _metric_blocks_by_physical(blocks, coords):
    mapping = {}
    coords = np.asarray(coords, dtype=np.int64)
    for block_id, (block_indices, _basis) in enumerate(blocks):
        block_indices = np.asarray(block_indices, dtype=np.int64)
        if block_indices.size == 0:
            continue
        physical = tuple(int(x) for x in coords[int(block_indices[0]), 1:3])
        mapping.setdefault(physical, []).append(block_id)
    return mapping


def _reduced_heff_sparse_by_transitions(
    coords,
    left,
    right,
    transitions,
    blocks,
    *,
    dtype,
    atol=0.0,
):
    """Build ``B^H H B`` directly from Abelian/metric support blocks."""
    coords = np.asarray(coords, dtype=np.int64)
    if transitions is None:
        return None
    offsets, reduced_dim, basis_dtype = _metric_block_offsets(blocks)
    if reduced_dim == 0:
        return csr_matrix((0, 0), dtype=dtype)
    dtype = np.result_type(dtype, basis_dtype)
    block_map = _metric_blocks_by_physical(blocks, coords)

    rows = []
    cols = []
    data = []
    for transition in range(transitions.ntransitions):
        bra_i = int(transitions.bra_i[transition])
        ket_i = int(transitions.ket_i[transition])
        bra_j = int(transitions.bra_j[transition])
        ket_j = int(transitions.ket_j[transition])
        row_blocks = block_map.get((bra_i, bra_j))
        if not row_blocks:
            continue
        col_blocks = block_map.get((ket_i, ket_j))
        if not col_blocks:
            continue
        entry_start = int(transitions.entry_starts[transition])
        entry_stop = int(transitions.entry_starts[transition + 1])
        if entry_start == entry_stop:
            continue

        for row_block in row_blocks:
            row_indices, row_basis = blocks[row_block]
            row_indices = np.asarray(row_indices, dtype=np.int64)
            row_coords = coords[row_indices]
            row_left = row_coords[:, 0]
            row_right = row_coords[:, 3]
            row_slice = slice(*offsets[row_block])
            row_width = offsets[row_block][1] - offsets[row_block][0]
            if row_width == 0:
                continue
            row_basis_h = row_basis.conj().T
            for col_block in col_blocks:
                col_indices, col_basis = blocks[col_block]
                col_indices = np.asarray(col_indices, dtype=np.int64)
                col_slice = slice(*offsets[col_block])
                col_width = offsets[col_block][1] - offsets[col_block][0]
                if col_width == 0:
                    continue
                col_coords = coords[col_indices]
                col_left = col_coords[:, 0]
                col_right = col_coords[:, 3]
                support_block = np.zeros((row_indices.size, col_indices.size), dtype=dtype)
                for entry in range(entry_start, entry_stop):
                    m = int(transitions.entry_m[entry])
                    n = int(transitions.entry_n[entry])
                    support_block += (
                        transitions.entry_values[entry]
                        * left[row_left[:, None], col_left[None, :], m, bra_i, ket_i]
                        * right[row_right[:, None], col_right[None, :], n, bra_j, ket_j]
                    )
                reduced_block = row_basis_h @ support_block @ col_basis
                flat = reduced_block.reshape(-1)
                if atol > 0.0:
                    nonzero = np.flatnonzero(np.abs(flat) > float(atol))
                else:
                    nonzero = np.flatnonzero(flat != 0)
                if nonzero.size == 0:
                    continue
                block_rows = nonzero // col_width
                block_cols = nonzero % col_width
                rows.append(np.arange(row_slice.start, row_slice.stop, dtype=np.int64)[block_rows])
                cols.append(np.arange(col_slice.start, col_slice.stop, dtype=np.int64)[block_cols])
                data.append(flat[nonzero])

    if not data:
        return csr_matrix((reduced_dim, reduced_dim), dtype=dtype)
    return csr_matrix(
        (
            np.concatenate(data).astype(dtype, copy=False),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(reduced_dim, reduced_dim),
        dtype=dtype,
    )


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
            out[:, :, cols, :, :] += _cached_einsum(
                "bkn,buc,kvd->cdnuv",
                q,
                tensor_conj[:, bra_state, :, :],
                tensor[:, ket_state, :, :],
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
            out[:, :, rows, :, :] += _cached_einsum(
                "cdm,bxc,kyd->bkmxy",
                q,
                tensor_conj[:, :, bra_state, :],
                tensor[:, :, ket_state, :],
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
    if solver not in {"auto", "dense", "matrix_free", "direct"}:
        raise ValueError("local_solver must be 'auto', 'dense', 'matrix_free', or 'direct'.")
    return solver


def _validate_gauge(gauge):
    if gauge is None:
        return None
    mode = str(gauge).lower().replace("-", "_")
    if mode not in {"virtual", "conditional"}:
        raise ValueError("gauge must be None, 'virtual', or 'conditional'.")
    return mode


def _validate_gauge_mode(mode):
    if mode is None:
        return "symmetric"
    mode = str(mode).lower().replace("-", "_")
    if mode not in {"symmetric", "qr"}:
        raise ValueError("mode must be 'symmetric' or 'qr'.")
    return mode


def _normalize_with_metric(vector, metric):
    norm2 = np.vdot(vector, metric @ vector)
    norm = np.sqrt(float(np.real(norm2)))
    if norm < 1e-14:
        raise ValueError("Cannot normalize a numerically zero state.")
    return vector / norm


def _metric_basis(metric, *, metric_tol=1e-12, metric_threshold=None):
    metric = 0.5 * (metric + metric.conj().T)
    if metric_threshold is None:
        metric_scale = float(np.linalg.norm(metric, ord=np.inf))
        metric_threshold = metric_tol * metric_scale
    if metric.shape == (1, 1):
        value = float(np.real(metric[0, 0]))
        if value <= metric_threshold:
            raise ValueError("Effective overlap metric is numerically singular.")
        return np.asarray([[1.0 / np.sqrt(value)]], dtype=metric.dtype)
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
    if matrix.shape == (1, 1):
        vector = np.ones(1, dtype=matrix.dtype)
        return float(np.real(matrix[0, 0])), vector
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


def _lowest_sparse_lobpcg_with_diagonal_preconditioner(
    operator,
    *,
    v0=None,
    tol=1e-9,
    maxiter=None,
):
    n = int(operator.shape[0])
    if n < 2:
        dense = operator.toarray() if hasattr(operator, "toarray") else np.asarray(operator)
        return _lowest_hermitian_eigenpair(dense)
    if v0 is None:
        guess = np.ones((n, 1), dtype=operator.dtype)
    else:
        guess = np.asarray(v0, dtype=operator.dtype).reshape(n, 1)
    norm = np.linalg.norm(guess)
    if norm <= 1.0e-14:
        guess[:, 0] = 1.0 / np.sqrt(n)
    else:
        guess /= norm

    diagonal = operator.diagonal() if hasattr(operator, "diagonal") else None
    if diagonal is None:
        return None
    diagonal = np.asarray(diagonal)
    scale = max(float(np.max(np.abs(diagonal))) if diagonal.size else 0.0, 1.0)
    inverse_diagonal = 1.0 / np.maximum(np.abs(diagonal), 1.0e-10 * scale)

    def apply_preconditioner(vector):
        return inverse_diagonal[:, None] * np.asarray(vector)

    preconditioner = LinearOperator(
        (n, n),
        matmat=apply_preconditioner,
        dtype=operator.dtype,
    )
    evals, evecs = lobpcg(
        operator,
        guess,
        M=preconditioner,
        largest=False,
        tol=tol,
        maxiter=maxiter or max(40, 4 * n),
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
    metric_scale = 0.0
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


def _metric_blocks_reduced_dim(blocks):
    return int(sum(basis.shape[1] for _indices, basis in blocks))


def _reduced_to_support_vector(coeff, blocks, size, *, dtype=None):
    coeff = np.asarray(coeff)
    if dtype is None:
        dtype = coeff.dtype
    vector = np.zeros(int(size), dtype=np.result_type(dtype, coeff.dtype))
    cursor = 0
    for block_indices, basis in blocks:
        width = basis.shape[1]
        vector[block_indices] = basis @ coeff[cursor:cursor + width]
        cursor += width
    return vector


def _support_to_reduced_vector(vector, blocks, *, dtype=None):
    vector = np.asarray(vector)
    if dtype is None:
        dtype = vector.dtype
    pieces = []
    for block_indices, basis in blocks:
        pieces.append(basis.conj().T @ vector[block_indices])
    if not pieces:
        return np.empty(0, dtype=dtype)
    return np.concatenate(pieces).astype(
        np.result_type(dtype, *[piece.dtype for piece in pieces]),
        copy=False,
    )


def _metric_blocks_sparse_basis(blocks, size, *, dtype=None):
    if dtype is None:
        dtype = np.result_type(*[basis.dtype for _indices, basis in blocks])
    rows = []
    cols = []
    data = []
    cursor = 0
    for block_indices, basis in blocks:
        block_indices = np.asarray(block_indices, dtype=np.int64)
        width = basis.shape[1]
        if width == 0 or block_indices.size == 0:
            continue
        block_rows = np.repeat(block_indices, width)
        block_cols = np.tile(np.arange(cursor, cursor + width, dtype=np.int64), block_indices.size)
        rows.append(block_rows)
        cols.append(block_cols)
        data.append(np.asarray(basis).reshape(-1))
        cursor += width
    if not data:
        return csr_matrix((int(size), 0), dtype=dtype)
    return csr_matrix(
        (
            np.concatenate(data).astype(dtype, copy=False),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(int(size), cursor),
        dtype=dtype,
    )


def _active_projected_warm_start(
    matvec,
    v0,
    reduced_dim,
    *,
    dtype,
    max_dim=_DEFAULT_ACTIVE_WARM_START_DIM,
    relative_tol=1.0e-8,
):
    """Refine an initial vector by solving its small active reduced subspace."""
    if v0 is None:
        return None
    v0 = np.asarray(v0).reshape(-1)
    if v0.size != int(reduced_dim):
        return v0
    scale = float(np.max(np.abs(v0))) if v0.size else 0.0
    if scale <= 0.0:
        return v0
    active = np.flatnonzero(np.abs(v0) > float(relative_tol) * scale)
    if active.size <= 1 or active.size >= int(reduced_dim) or active.size > int(max_dim):
        return v0
    sub_h = np.empty((active.size, active.size), dtype=dtype)
    for column, index in enumerate(active):
        basis_vector = np.zeros(int(reduced_dim), dtype=dtype)
        basis_vector[int(index)] = 1.0
        sub_h[:, column] = matvec(basis_vector)[active]
    sub_h = 0.5 * (sub_h + sub_h.conj().T)
    _energy, active_vector = _lowest_hermitian_eigenpair(
        sub_h,
        iterative_threshold=int(max_dim) + 1,
    )
    refined = np.zeros(int(reduced_dim), dtype=np.result_type(dtype, active_vector.dtype))
    refined[active] = active_vector
    norm = np.linalg.norm(refined)
    if norm <= 1.0e-14:
        return v0
    return refined / norm


def _hermitian_sqrt_pair(matrix, *, eps=1.0e-14, rcond=1.0e-12):
    matrix = np.asarray(matrix)
    matrix = 0.5 * (matrix + matrix.conj().T)
    if matrix.shape == (1, 1):
        value = float(np.real(matrix[0, 0]))
        scale = max(abs(value), float(eps))
        value = max(value, float(eps), float(rcond) * scale)
        root = np.sqrt(value)
        dtype = np.result_type(matrix.dtype, root)
        return (
            np.real_if_close(np.asarray([[root]], dtype=dtype), tol=1000),
            np.real_if_close(np.asarray([[1.0 / root]], dtype=dtype), tol=1000),
        )
    vals, vecs = linalg.eigh(matrix, check_finite=False)
    if vals.size == 0:
        return matrix.copy(), matrix.copy()
    scale = max(float(np.max(np.abs(vals))), float(eps))
    floor = max(float(eps), float(rcond) * scale)
    vals = np.maximum(np.real(vals), floor)
    sqrt_vals = np.sqrt(vals)
    sqrt = (vecs * sqrt_vals[None, :]) @ vecs.conj().T
    inv_sqrt = (vecs * (1.0 / sqrt_vals)[None, :]) @ vecs.conj().T
    return np.real_if_close(sqrt, tol=1000), np.real_if_close(inv_sqrt, tol=1000)


def _physical_groups_from_coords(coords):
    groups = {}
    for position, (_left, si, sj, _right) in enumerate(np.asarray(coords, dtype=np.int64)):
        groups.setdefault((int(si), int(sj)), []).append(position)
    return {
        key: np.asarray(value, dtype=np.int64)
        for key, value in groups.items()
    }


@dataclass(frozen=True)
class _AbelianLocalSupportPlan:
    blocks: tuple
    offsets: tuple
    flat_indices: np.ndarray
    coords: np.ndarray
    physical_groups: dict

    @property
    def size(self):
        return int(self.flat_indices.size)


def _metric_blocks_from_abelian_plan(plan, mleft, mright, *, metric_tol=1e-12):
    left_diag = np.einsum("bkxx->bkx", mleft[:, :, 0], optimize=True)
    right_diag = np.einsum("cduu->cdu", mright[:, :, 0], optimize=True)

    raw_blocks = []
    metric_scale = 0.0
    for block_index, (_start, _stop) in enumerate(plan.offsets):
        start, stop = int(_start), int(_stop)
        if start == stop:
            continue
        block = plan.blocks[block_index]
        si, sj = block.physical
        coords = plan.coords[start:stop]
        left_sel = coords[:, 0]
        right_sel = coords[:, 3]
        left_block = 0.5 * (
            left_diag[:, :, si] + left_diag[:, :, si].conj().T
        )
        right_block = 0.5 * (
            right_diag[:, :, sj] + right_diag[:, :, sj].conj().T
        )
        metric = (
            left_block[left_sel[:, None], left_sel[None, :]]
            * right_block[right_sel[:, None], right_sel[None, :]]
        )
        metric = 0.5 * (metric + metric.conj().T)
        metric_scale = max(metric_scale, float(np.linalg.norm(metric, ord=np.inf)))
        raw_blocks.append((np.arange(start, stop, dtype=np.int64), metric))

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
    def for_sweep(cls, ansatz, mpo, direction, *, verbose=0):
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        identity = ansatz.identity_mpo()
        nlocal = ansatz.nlocal_tensors
        if direction == "lr":
            if int(verbose) > 1:
                print("letta-mpo build right environments", flush=True)
            right_envs = ansatz._right_local_environments(mpo, verbose=verbose)
            if int(verbose) > 1:
                print("letta-mpo build right metric environments", flush=True)
            metric_right = ansatz._right_metric_environments(verbose=verbose)
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
            if int(verbose) > 1:
                print("letta-mpo build left environments", flush=True)
            left_envs = ansatz._left_local_environments(mpo, verbose=verbose)
            if int(verbose) > 1:
                print("letta-mpo build left metric environments", flush=True)
            metric_left = ansatz._left_metric_environments(verbose=verbose)
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

    Passing a :class:`~pyqed.letta.LocalHamiltonian` and ``parents`` uses
    the graph/frontier implementation and infers ``dims`` from the
    Hamiltonian.  In that form, ``symmetry="u1"`` keeps every LETTA tensor
    unrestricted and applies an exact fixed-charge projector to the
    represented state.
    """

    def __new__(
        cls,
        hamiltonian,
        dims=None,
        parents=None,
        *,
        symmetry=None,
        charges=None,
        target=0,
        **kwargs,
    ):
        if cls is not LETTA:
            return super().__new__(cls)

        from .local_terms import (
            LocalHamiltonian,
            local_charges_from_sites,
        )

        if isinstance(hamiltonian, LocalHamiltonian) and dims is None:
            dims = hamiltonian.dims
        graph_requested = parents is not None or isinstance(
            hamiltonian, LocalHamiltonian
        )
        if not graph_requested:
            return super().__new__(cls)
        if parents is None:
            raise TypeError("graph LETTA requires parents.")

        options = dict(kwargs)
        reduced_layout_options = tuple(
            name
            for name in ("layout", "symmetry_layout", "abelian_layout")
            if options.pop(name, None) is not None
        )
        if reduced_layout_options:
            names = ", ".join(reduced_layout_options)
            raise TypeError(
                f"{names} selected the removed locally masked U(1) ansatz; "
                "use symmetry='u1' with charges for exact sector projection."
            )

        if symmetry is None:
            normalized_symmetry = None
        else:
            normalized_symmetry = (
                str(symmetry)
                .strip()
                .lower()
                .replace("(", "")
                .replace(")", "")
            )
            normalized_symmetry = normalized_symmetry.replace("_", "").replace(
                "-", ""
            )
            if normalized_symmetry in {"none", "dense", "unrestricted"}:
                normalized_symmetry = None
            elif normalized_symmetry in {
                "u1",
                "abelian",
                "projected",
                "projectedu1",
                "u1projected",
                "exactu1",
                "sectorprojected",
            }:
                normalized_symmetry = "projected_u1"
            else:
                raise ValueError("symmetry must be None or 'u1'.")

        from .frontier_tying import FrontierTiedLETTA

        if normalized_symmetry is None:
            if charges is not None:
                raise TypeError("charges require symmetry='u1'.")
            return FrontierTiedLETTA(hamiltonian, dims, parents, **options)

        if charges is None:
            charges = local_charges_from_sites(
                hamiltonian.sites,
                require=True,
            )
        if "charge_assignment" in options:
            raise TypeError(
                "exact U(1) projection counts every unique physical site once "
                "and does not use charge_assignment."
            )
        from .projected_frontier import SectorProjectedLETTA

        return SectorProjectedLETTA(
            hamiltonian,
            dims,
            parents,
            local_charges=charges,
            target=target,
            **options,
        )

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=4,
        overlap=None,
        tensors=None,
        local_masks=None,
        abelian_layout=None,
        symmetry=None,
        seed=None,
    ):
        if symmetry is not None:
            raise TypeError(
                "symmetry= is available for graph LETTA with a LocalHamiltonian "
                "and parent_sets; dense LETTA accepts abelian_layout instead."
            )
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
        self.abelian_layout = abelian_layout
        if local_masks is None and abelian_layout is not None:
            local_masks = abelian_layout.local_masks()
        self.local_masks = self._validate_local_masks(local_masks)
        self._apply_local_masks()
        self.history = []
        self.converged = False
        self.energy = None
        self.state_metadata = {}
        self._support_plan_cache = {}
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

    @classmethod
    def from_mps(
        cls,
        mps,
        *,
        hamiltonian=None,
        dims=None,
        overlap=None,
        seed=None,
        local_masks=None,
        abelian_layout=None,
    ):
        """
        Embed an open-boundary MPS exactly into the leg-tied LETTA form.

        The LETTA pair tensor at bond ``i`` carries the MPS core for site
        ``i`` and is independent of the shared right physical index.  The last
        LETTA pair tensor absorbs the final two MPS cores.
        """
        factors = mps.factors if hasattr(mps, "factors") else mps
        factors = [np.asarray(factor) for factor in factors]
        if len(factors) < 2:
            raise ValueError("at least two MPS tensors are required.")
        if dims is None:
            dims = tuple(int(factor.shape[1]) for factor in factors)
        dims = _validate_dims(dims)
        if len(dims) != len(factors):
            raise ValueError("dims must have one entry per MPS tensor.")

        for i, (factor, dim) in enumerate(zip(factors, dims)):
            if factor.ndim != 3 or factor.shape[1] != dim:
                raise ValueError(f"MPS tensor {i} must have shape (left, {dim}, right).")
            if i == 0 and factor.shape[0] != 1:
                raise ValueError("first MPS tensor must have left bond dimension 1.")
            if i == len(factors) - 1 and factor.shape[2] != 1:
                raise ValueError("last MPS tensor must have right bond dimension 1.")
            if i and factors[i - 1].shape[2] != factor.shape[0]:
                raise ValueError(f"MPS bond mismatch between tensors {i - 1} and {i}.")

        letta_tensors = []
        for i, factor in enumerate(factors[:-2]):
            tensor = np.repeat(factor[:, :, None, :], dims[i + 1], axis=2)
            letta_tensors.append(tensor.copy())

        penultimate = factors[-2]
        final = factors[-1]
        last = _cached_einsum("asb,btz->astz", penultimate, final)
        letta_tensors.append(last)

        bond_dim = max(max(tensor.shape[0], tensor.shape[3]) for tensor in letta_tensors)
        return cls(
            hamiltonian,
            dims,
            bond_dim=bond_dim,
            overlap=overlap,
            tensors=letta_tensors,
            local_masks=local_masks,
            abelian_layout=abelian_layout,
            seed=seed,
        )

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

    @property
    def ncompleted(self):
        return len(self.history)

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

    def _apply_local_masks(self, tensor_indices=None):
        if tensor_indices is None:
            tensor_indices = range(self.nlocal_tensors)
        for i in tensor_indices:
            mask = self.local_masks[int(i)]
            if mask is not None:
                self.tensors[int(i)] = np.where(mask, self.tensors[int(i)], 0)

    def local_support_sizes(self):
        """Return allowed/total entry counts for each symmetry/support mask."""
        sizes = []
        for tensor, mask in zip(self.tensors, self.local_masks):
            allowed = tensor.size if mask is None else int(np.count_nonzero(mask))
            sizes.append((allowed, tensor.size))
        return sizes

    @staticmethod
    def _expand_repeated_labels(labels, target_size: int):
        labels = list(labels)
        target_size = int(target_size)
        if target_size < len(labels):
            raise ValueError("target bond size cannot be smaller than the current layout.")
        if target_size == len(labels):
            return list(labels)
        if not labels:
            raise ValueError("cannot expand an empty Abelian bond label list.")
        expanded = list(labels)
        cursor = 0
        while len(expanded) < target_size:
            expanded.append(labels[cursor % len(labels)])
            cursor += 1
        return expanded

    def _expanded_abelian_layout(self, bond_dim: int):
        if self.abelian_layout is None:
            return None
        from .abelian import Layout

        bond_qns = getattr(self.abelian_layout, "bond_qns", None)
        if bond_qns is None:
            raise NotImplementedError(
                "bond expansion is not yet implemented for conditional "
                "tied-frontier charge layouts."
            )
        bond_dim = int(bond_dim)
        target_sizes = [1] + [bond_dim] * (len(bond_qns) - 1)
        bond_qns = [
            self._expand_repeated_labels(labels, target)
            for labels, target in zip(bond_qns, target_sizes)
        ]
        return Layout(
            local_qns=self.abelian_layout.local_qns,
            bond_qns=bond_qns,
            target=self.abelian_layout.target,
        )

    @staticmethod
    def _pad_tensor_with_masked_noise(tensor, shape, *, mask=None, noise: float = 0.0, rng=None):
        tensor = np.asarray(tensor)
        shape = tuple(int(dim) for dim in shape)
        out = np.zeros(shape, dtype=tensor.dtype)
        old = tuple(slice(0, dim) for dim in tensor.shape)
        out[old] = tensor
        if float(noise) > 0.0:
            allowed = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
            old_region = np.zeros(shape, dtype=bool)
            old_region[old] = True
            fill = allowed & ~old_region
            if np.any(fill):
                rng = np.random.default_rng() if rng is None else rng
                values = rng.normal(scale=float(noise), size=int(np.count_nonzero(fill)))
                if np.iscomplexobj(out):
                    values = values + 1j * rng.normal(scale=float(noise), size=values.size)
                out[fill] = values
        return out

    def expand_bond_dim(self, bond_dim: int, *, noise: float = 0.0, seed=None):
        """Increase LETTA virtual bond dimensions while preserving the old state.

        With ``noise=0`` this is exact zero-padding.  With small ``noise``,
        newly opened symmetry-allowed tensor entries are seeded while all old
        entries are left untouched.
        """
        if self.has_terminal_tensor:
            raise NotImplementedError("expand_bond_dim does not yet support terminal LETTA tensors.")
        bond_dim = int(bond_dim)
        if bond_dim < self.bond_dim:
            raise ValueError("expand_bond_dim only supports increasing the bond dimension.")
        if bond_dim == self.bond_dim:
            return self

        new_layout = self._expanded_abelian_layout(bond_dim)
        new_masks = None if new_layout is None else new_layout.local_masks()
        bonds = [1] + [bond_dim] * max(0, self.nsites - 2) + [1]
        rng = self.rng if seed is None else np.random.default_rng(seed)
        tensors = []
        for i, tensor in enumerate(self.tensors):
            shape = (bonds[i], self.dims[i], self.dims[i + 1], bonds[i + 1])
            mask = None if new_masks is None else new_masks[i]
            tensors.append(
                self._pad_tensor_with_masked_noise(
                    tensor,
                    shape,
                    mask=mask,
                    noise=float(noise),
                    rng=rng,
                )
            )

        self.bond_dim = bond_dim
        self.abelian_layout = new_layout
        self.tensors = tensors
        self.local_masks = self._validate_local_masks(new_masks)
        self._support_plan_cache = {}
        self._apply_local_masks()
        self.normalize()
        self.converged = False
        return self

    def copy(self):
        tensors = [tensor.copy() for tensor in self.tensors]
        out = LETTA(
            None if self.hamiltonian is None else self.hamiltonian.copy(),
            self.dims,
            bond_dim=self.bond_dim,
            overlap=None if self.overlap is None else self.overlap.copy(),
            tensors=tensors,
            local_masks=[None if mask is None else mask.copy() for mask in self.local_masks],
            abelian_layout=self.abelian_layout,
        )
        # Construction normalizes variational guesses.  A copy must retain the
        # raw scale as well as the represented ray, especially during
        # real-time propagation where norm drift is a diagnostic.
        out.tensors = tensors
        out.history = list(self.history)
        out.converged = bool(self.converged)
        out.energy = self.energy
        out.state_metadata = dict(self.state_metadata)
        return out

    def to_state_dict(self, *, metadata=None, include_operators=False):
        """Return a pickle-friendly state payload for restarting LETTA sweeps."""
        payload = {
            "format": "pyqed.letta.LETTA.state",
            "version": 1,
            "dims": tuple(int(dim) for dim in self.dims),
            "bond_dim": int(self.bond_dim),
            "tensors": [tensor.copy() for tensor in self.tensors],
            "local_masks": [None if mask is None else mask.copy() for mask in self.local_masks],
            "abelian_layout": self.abelian_layout,
            "history": list(self.history),
            "converged": bool(self.converged),
            "energy": self.energy,
            "metadata": dict(self.state_metadata if metadata is None else metadata),
        }
        if include_operators:
            payload["hamiltonian"] = None if self.hamiltonian is None else self.hamiltonian.copy()
            payload["overlap"] = None if self.overlap is None else self.overlap.copy()
        return payload

    @classmethod
    def from_state_dict(cls, payload, *, hamiltonian=None, overlap=None):
        """Reconstruct a LETTA state saved by :meth:`to_state_dict`."""
        if payload.get("format") != "pyqed.letta.LETTA.state":
            raise ValueError("not a pyqed LETTA state payload.")
        if hamiltonian is None and "hamiltonian" in payload:
            hamiltonian = payload["hamiltonian"]
        if overlap is None and "overlap" in payload:
            overlap = payload["overlap"]
        out = cls(
            hamiltonian,
            payload["dims"],
            bond_dim=payload.get("bond_dim", 4),
            overlap=overlap,
            tensors=payload["tensors"],
            local_masks=payload.get("local_masks"),
            abelian_layout=payload.get("abelian_layout"),
        )
        out.tensors = [np.asarray(tensor).copy() for tensor in payload["tensors"]]
        out.history = list(payload.get("history", []))
        out.converged = bool(payload.get("converged", False))
        out.energy = payload.get("energy")
        out.state_metadata = dict(payload.get("metadata", {}))
        return out

    def save(self, path, *, metadata=None, include_operators=False):
        """Save this LETTA state to ``path`` for later continuation."""
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(
                self.to_state_dict(metadata=metadata, include_operators=include_operators),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return output

    @classmethod
    def load(cls, path, *, hamiltonian=None, overlap=None):
        """Load a LETTA state saved by :meth:`save`."""
        source = Path(path).expanduser()
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        return cls.from_state_dict(payload, hamiltonian=hamiltonian, overlap=overlap)

    def _local_support_plan(self, tensor_index, mask):
        tensor_index = int(tensor_index)
        key = (tensor_index, id(mask))
        cached = self._support_plan_cache.get(key)
        if cached is not None:
            return cached
        allowed = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
        coords = np.asarray(np.unravel_index(allowed, self.tensors[tensor_index].shape)).T
        physical_groups = _physical_groups_from_coords(coords)
        plan = (allowed, coords, physical_groups)
        self._support_plan_cache[key] = plan
        return plan

    def _local_abelian_support_plan(self, tensor_index, mask):
        if self.abelian_layout is None or not hasattr(self.abelian_layout, "local_tensor_blocks"):
            return None
        tensor_index = int(tensor_index)
        mask = np.asarray(mask, dtype=bool)
        key = ("abelian", tensor_index, id(mask), id(self.abelian_layout))
        cached = self._support_plan_cache.get(key)
        if cached is not None:
            return cached
        try:
            blocks = tuple(self.abelian_layout.local_tensor_blocks(tensor_index))
        except Exception:
            return None
        if not blocks:
            return None
        flat_indices = []
        coords = []
        offsets = []
        kept_blocks = []
        cursor = 0
        mask_flat = mask.reshape(-1)
        for block in blocks:
            block_flat = np.asarray(block.flat_indices, dtype=np.int64)
            block_coords = np.asarray(block.coords, dtype=np.int64)
            if block_flat.size == 0:
                continue
            if block_coords.shape != (block_flat.size, 4):
                return None
            active = mask_flat[block_flat]
            if not np.any(active):
                continue
            block_flat = block_flat[active]
            block_coords = block_coords[active]
            kept_blocks.append(block)
            flat_indices.append(block_flat)
            coords.append(block_coords)
            offsets.append((cursor, cursor + block_flat.size))
            cursor += block_flat.size
        if not flat_indices:
            return None
        flat_indices = np.concatenate(flat_indices)
        coords = np.concatenate(coords, axis=0)
        if flat_indices.size != int(np.count_nonzero(mask_flat)):
            return None
        if not np.all(mask_flat[flat_indices]):
            return None
        if np.unique(flat_indices).size != flat_indices.size:
            return None
        plan = _AbelianLocalSupportPlan(
            blocks=tuple(kept_blocks),
            offsets=tuple(offsets),
            flat_indices=flat_indices,
            coords=coords,
            physical_groups=_physical_groups_from_coords(coords),
        )
        self._support_plan_cache[key] = plan
        return plan

    def _abelian_bond_labels(self, bond, shared_state=None):
        """Return charge labels for one native LETTA virtual bond."""
        layout = self.abelian_layout
        if layout is None:
            return None
        frontier_labels = getattr(layout, "frontier_labels", None)
        if frontier_labels is not None:
            return tuple(frontier_labels(int(bond), shared_state=shared_state))
        bond_qns = getattr(layout, "bond_qns", None)
        if bond_qns is None or int(bond) + 1 >= len(bond_qns):
            return None
        return tuple(bond_qns[int(bond) + 1])

    def _virtual_bond_groups(self, bond):
        bond = int(bond)
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        if left.ndim != 4:
            return []
        right_axis = 0 if right.ndim == 4 else 1 if right.ndim == 2 else None
        if right_axis is None:
            return []
        shared = min(left.shape[3], right.shape[right_axis])
        if shared < 1:
            return []
        labels = self._abelian_bond_labels(bond)
        if labels is not None:
            labels = labels[:shared]
        if labels is None and self.local_masks[bond] is None and self.local_masks[bond + 1] is None:
            return [np.arange(shared, dtype=np.int64)]
        left_mask = self.local_masks[bond]
        right_mask = self.local_masks[bond + 1]
        groups = {}
        for index in range(shared):
            label = None if labels is None else tuple(labels[index])
            left_signature = (
                None
                if left_mask is None
                else np.ascontiguousarray(left_mask[:, :, :, index]).tobytes()
            )
            if right_mask is None:
                right_signature = None
            elif right.ndim == 4:
                right_signature = np.ascontiguousarray(right_mask[index, :, :, :]).tobytes()
            else:
                right_signature = np.ascontiguousarray(right_mask[:, index]).tobytes()
            groups.setdefault((label, left_signature, right_signature), []).append(index)
        return [np.asarray(indices, dtype=np.int64) for indices in groups.values() if indices]

    def _conditional_virtual_bond_groups(self, bond, shared_state):
        """Return support-compatible gauge groups at one shared physical state."""
        bond = int(bond)
        shared_state = int(shared_state)
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        shared = min(left.shape[3], right.shape[0] if right.ndim == 4 else right.shape[1])
        labels = self._abelian_bond_labels(bond, shared_state=shared_state)
        if labels is not None:
            labels = labels[:shared]
        left_mask = self.local_masks[bond]
        right_mask = self.local_masks[bond + 1]
        if labels is None and left_mask is None and right_mask is None:
            return [np.arange(shared, dtype=np.int64)]
        key = (
            "conditional_bond_groups",
            bond,
            shared_state,
            shared,
            id(left_mask),
            id(right_mask),
            id(self.abelian_layout),
        )
        cached = self._support_plan_cache.get(key)
        if cached is not None:
            return cached

        groups = {}
        for index in range(shared):
            label = None if labels is None else tuple(labels[index])
            left_signature = (
                None
                if left_mask is None
                else np.ascontiguousarray(left_mask[:, :, shared_state, index]).tobytes()
            )
            if right_mask is None:
                right_signature = None
            elif right.ndim == 4:
                right_signature = np.ascontiguousarray(
                    right_mask[index, shared_state, :, :]
                ).tobytes()
            else:
                right_signature = bool(right_mask[shared_state, index])
            groups.setdefault((label, left_signature, right_signature), []).append(index)
        result = tuple(
            np.asarray(indices, dtype=np.int64)
            for indices in groups.values()
            if indices
        )
        self._support_plan_cache[key] = result
        return result

    def compress_conditional_bond(
        self,
        bond,
        *,
        direction="lr",
        rtol=1.0e-12,
        atol=0.0,
        max_bond_dim=None,
    ):
        """Compress one LETTA bond by physical-conditioned SVDs.

        For every value of the shared physical leg, the two neighboring
        tensors define a matrix product ``L_s @ R_s``.  This routine factors
        that product independently inside each symmetry/support-compatible
        virtual group, removes numerical null directions, and stores the
        largest retained conditional rank as the new common bond dimension.
        Smaller conditional ranks are represented by local support masks.

        With ``max_bond_dim=None`` only numerical null directions are removed.
        Supplying a maximum additionally truncates the least-important common
        group channels and reports their local discarded singular-value
        weight.
        """
        if (
            self.abelian_layout is not None
            and getattr(self.abelian_layout, "frontier_qns", None) is not None
        ):
            raise NotImplementedError(
                "conditional compression does not yet update tied-frontier "
                "charge labels."
            )
        bond = int(bond)
        if bond < 0 or bond >= self.nlocal_tensors - 1:
            raise IndexError("bond index out of range.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl", "balanced"}:
            raise ValueError("direction must be 'lr', 'rl', or 'balanced'.")
        rtol = float(rtol)
        atol = float(atol)
        if rtol < 0.0 or atol < 0.0:
            raise ValueError("rtol and atol must be nonnegative.")
        if max_bond_dim is not None:
            max_bond_dim = int(max_bond_dim)
            if max_bond_dim < 1:
                raise ValueError("max_bond_dim must be positive when provided.")

        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        if left.ndim != 4 or right.ndim not in {2, 4}:
            raise ValueError(
                "conditional compression needs a pair tensor followed by "
                "a pair or terminal tensor."
            )
        old_dim = int(left.shape[3])
        right_dim = int(right.shape[0] if right.ndim == 4 else right.shape[1])
        if old_dim != right_dim:
            raise ValueError("neighboring LETTA tensors have incompatible virtual bond dimensions.")
        shared_dim = int(left.shape[2])
        expected_shared = int(right.shape[1] if right.ndim == 4 else right.shape[0])
        if shared_dim != expected_shared:
            raise ValueError("neighboring LETTA tensors have incompatible shared physical legs.")

        groups = self._virtual_bond_groups(bond)
        records = []
        numeric_ranks = []
        total_weight = 0.0
        for group in groups:
            group = np.asarray(group, dtype=np.int64)
            group_records = []
            group_ranks = []
            for shared_state in range(shared_dim):
                left_matrix, right_matrix = self._conditional_bond_matrices(
                    bond,
                    shared_state,
                    group,
                )
                product = left_matrix @ right_matrix
                u, singular_values, vh = linalg.svd(
                    product,
                    full_matrices=False,
                    check_finite=False,
                )
                scale = float(singular_values[0]) if singular_values.size else 0.0
                cutoff = max(atol, rtol * scale)
                rank = int(np.count_nonzero(singular_values > cutoff))
                group_records.append((u, singular_values, vh, rank))
                group_ranks.append(rank)
                total_weight += float(np.sum(np.abs(singular_values) ** 2))
            records.append(group_records)
            numeric_ranks.append(group_ranks)

        capacities = [max(ranks, default=0) for ranks in numeric_ranks]
        exact_dim = int(sum(capacities))
        if max_bond_dim is not None and exact_dim > max_bond_dim:
            capacities = [0] * len(records)
            for _ in range(max_bond_dim):
                gains = []
                for group_index, group_records in enumerate(records):
                    channel = capacities[group_index]
                    gain = sum(
                        float(abs(values[channel]) ** 2)
                        for _u, values, _vh, rank in group_records
                        if channel < rank
                    )
                    gains.append(gain)
                selected = int(np.argmax(gains)) if gains else 0
                if not gains or gains[selected] <= 0.0:
                    break
                capacities[selected] += 1

        new_dim = int(sum(capacities))
        if new_dim < 1:
            raise ValueError("conditional compression removed every virtual channel.")

        dtype = np.result_type(left.dtype, right.dtype)
        new_left = np.zeros(left.shape[:-1] + (new_dim,), dtype=dtype)
        if right.ndim == 4:
            new_right = np.zeros((new_dim,) + right.shape[1:], dtype=dtype)
        else:
            new_right = np.zeros((right.shape[0], new_dim), dtype=dtype)
        new_left_mask = np.zeros(new_left.shape, dtype=bool)
        new_right_mask = np.zeros(new_right.shape, dtype=bool)
        old_left_mask = self.local_masks[bond]
        old_right_mask = self.local_masks[bond + 1]
        retained_ranks = np.zeros(shared_dim, dtype=np.int64)
        discarded_weight = 0.0
        new_labels = []
        cursor = 0
        old_labels = None
        bond_qns = (
            None
            if self.abelian_layout is None
            else getattr(self.abelian_layout, "bond_qns", None)
        )
        if bond_qns is not None and bond + 1 < len(bond_qns):
            old_labels = bond_qns[bond + 1]

        for group_index, (group, group_records) in enumerate(zip(groups, records)):
            group = np.asarray(group, dtype=np.int64)
            capacity = int(capacities[group_index])
            if capacity == 0:
                for _u, values, _vh, _rank in group_records:
                    discarded_weight += float(np.sum(np.abs(values) ** 2))
                continue
            if old_labels is not None:
                new_labels.extend([old_labels[int(group[0])]] * capacity)
            left_allowed = (
                np.ones(left.shape[:-1], dtype=bool)
                if old_left_mask is None
                else np.any(old_left_mask[..., group], axis=-1)
            )
            if old_right_mask is None:
                allowed_shape = right.shape[1:] if right.ndim == 4 else (right.shape[0],)
                right_allowed = np.ones(allowed_shape, dtype=bool)
            elif right.ndim == 4:
                right_allowed = np.any(old_right_mask[group, ...], axis=0)
            else:
                right_allowed = np.any(old_right_mask[:, group], axis=1)

            for shared_state, (u, values, vh, numeric_rank) in enumerate(group_records):
                rank = min(int(numeric_rank), capacity)
                retained_ranks[shared_state] += rank
                discarded_weight += float(np.sum(np.abs(values[rank:]) ** 2))
                if rank == 0:
                    continue
                if direction == "lr":
                    left_factor = u[:, :rank]
                    right_factor = values[:rank, None] * vh[:rank, :]
                elif direction == "rl":
                    left_factor = u[:, :rank] * values[None, :rank]
                    right_factor = vh[:rank, :]
                else:
                    roots = np.sqrt(values[:rank])
                    left_factor = u[:, :rank] * roots[None, :]
                    right_factor = roots[:, None] * vh[:rank, :]

                left_slice = new_left[:, :, shared_state, cursor : cursor + rank]
                left_slice[...] = left_factor.reshape(left_slice.shape)
                if right.ndim == 4:
                    right_slice = new_right[cursor : cursor + rank, shared_state, :, :]
                    right_slice[...] = right_factor.reshape(right_slice.shape)
                    new_right_mask[cursor : cursor + rank, shared_state, :, :] = right_allowed[
                        shared_state, :, :
                    ]
                else:
                    new_right[shared_state, cursor : cursor + rank] = right_factor.reshape(rank)
                    new_right_mask[shared_state, cursor : cursor + rank] = right_allowed[shared_state]
                new_left_mask[:, :, shared_state, cursor : cursor + rank] = left_allowed[
                    :, :, shared_state, None
                ]
            cursor += capacity

        self.tensors[bond] = new_left
        self.tensors[bond + 1] = new_right
        masks = [None if mask is None else mask.copy() for mask in self.local_masks]
        masks[bond] = None if np.all(new_left_mask) else new_left_mask
        masks[bond + 1] = None if np.all(new_right_mask) else new_right_mask

        if old_labels is not None:
            from .abelian import Layout

            bond_qns = [list(labels) for labels in bond_qns]
            bond_qns[bond + 1] = list(new_labels)
            self.abelian_layout = Layout(
                local_qns=self.abelian_layout.local_qns,
                bond_qns=bond_qns,
                target=self.abelian_layout.target,
            )

        self.local_masks = self._validate_local_masks(masks)
        self._support_plan_cache = {}
        self._apply_local_masks()
        internal_dims = [tensor.shape[3] for tensor in self.tensors[:-1] if tensor.ndim == 4]
        if self.has_terminal_tensor:
            internal_dims.append(self.tensors[-1].shape[1])
        self.bond_dim = max([1, *map(int, internal_dims)])
        self.converged = False
        relative_discarded = 0.0 if total_weight <= 0.0 else discarded_weight / total_weight
        return {
            "bond": bond,
            "old_dim": old_dim,
            "new_dim": new_dim,
            "exact_dim": exact_dim,
            "sector_ranks": tuple(int(rank) for rank in retained_ranks),
            "discarded_weight": float(discarded_weight),
            "relative_discarded_weight": float(relative_discarded),
            "truncated": bool(discarded_weight > max(atol * atol, 1.0e-30)),
        }

    def compress_conditional_bonds(
        self,
        *,
        direction="lr",
        rtol=1.0e-12,
        atol=0.0,
        max_bond_dim=None,
    ):
        """Compress every LETTA bond in sweep order."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        bonds = range(self.nlocal_tensors - 1)
        if direction == "rl":
            bonds = reversed(list(bonds))
        return [
            self.compress_conditional_bond(
                bond,
                direction=direction,
                rtol=rtol,
                atol=atol,
                max_bond_dim=max_bond_dim,
            )
            for bond in bonds
        ]

    def _right_bond_matrix(self, tensor, group):
        if tensor.ndim == 4:
            return tensor[group, :, :, :].reshape(len(group), -1)
        if tensor.ndim == 2:
            return tensor[:, group].T
        raise ValueError("right tensor must be a LETTA pair tensor or terminal tensor.")

    def _apply_virtual_gauge(self, bond, group, gauge, gauge_inv):
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        dtype = np.result_type(
            left.dtype,
            right.dtype,
            np.asarray(gauge).dtype,
            np.asarray(gauge_inv).dtype,
        )
        if left.dtype != dtype:
            left = left.astype(dtype)
            self.tensors[bond] = left
        if right.dtype != dtype:
            right = right.astype(dtype)
            self.tensors[bond + 1] = right
        group = np.asarray(group, dtype=np.int64)
        left_block = left[:, :, :, group]
        left[:, :, :, group] = np.tensordot(left_block, gauge, axes=([3], [0]))
        if right.ndim == 4:
            right_block = right[group, :, :, :]
            right[group, :, :, :] = np.tensordot(gauge_inv, right_block, axes=([1], [0]))
        elif right.ndim == 2:
            right_block = right[:, group]
            right[:, group] = right_block @ gauge_inv.T
        else:
            raise ValueError("right tensor must be a LETTA pair tensor or terminal tensor.")

    def _conditional_bond_matrices(self, bond, shared_state, group):
        """Return left/right bond matrices at one shared physical state."""
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        group = np.asarray(group, dtype=np.int64)
        left_slice = left[:, :, int(shared_state), :]
        left_matrix = left_slice[:, :, group].reshape(-1, group.size)
        if right.ndim == 4:
            right_slice = right[:, int(shared_state), :, :]
            right_matrix = right_slice[group, :, :].reshape(group.size, -1)
        elif right.ndim == 2:
            right_matrix = right[int(shared_state), group].reshape(group.size, 1)
        else:
            raise ValueError("right tensor must be a LETTA pair tensor or terminal tensor.")
        return left_matrix, right_matrix

    def _apply_conditional_virtual_gauge(
        self,
        bond,
        shared_state,
        group,
        gauge,
        gauge_inv,
    ):
        """Apply a state-preserving physical-conditioned bond gauge."""
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        dtype = np.result_type(
            left.dtype,
            right.dtype,
            np.asarray(gauge).dtype,
            np.asarray(gauge_inv).dtype,
        )
        if left.dtype != dtype:
            left = left.astype(dtype)
            self.tensors[bond] = left
        if right.dtype != dtype:
            right = right.astype(dtype)
            self.tensors[bond + 1] = right
        shared_state = int(shared_state)
        group = np.asarray(group, dtype=np.int64)

        left_slice = left[:, :, shared_state, :]
        left_block = left_slice[:, :, group]
        left_slice[:, :, group] = np.tensordot(left_block, gauge, axes=([2], [0]))

        if right.ndim == 4:
            right_slice = right[:, shared_state, :, :]
            right_block = right_slice[group, :, :]
            right_slice[group, :, :] = np.tensordot(
                gauge_inv,
                right_block,
                axes=([1], [0]),
            )
        elif right.ndim == 2:
            right_block = right[shared_state, group]
            right[shared_state, group] = gauge_inv @ right_block
        else:
            raise ValueError("right tensor must be a LETTA pair tensor or terminal tensor.")

    def canonicalize_conditional_bond(
        self,
        bond,
        *,
        direction="lr",
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=False,
    ):
        """Whiten one bond with gauges conditioned on its shared physical leg.

        For the bond between pair tensors ``i`` and ``i+1``, LETTA permits a
        separate gauge ``G_i(s)`` for every value of the shared physical state
        ``s``.  The gauge and its inverse are applied to the two tensors, so
        the represented state is unchanged configuration by configuration.
        """
        bond = int(bond)
        if bond < 0 or bond >= self.nlocal_tensors - 1:
            raise IndexError("bond index out of range.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        mode = _validate_gauge_mode(mode)
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        if left.ndim != 4:
            return self
        shared_dim = left.shape[2]
        expected_shared_dim = right.shape[1] if right.ndim == 4 else right.shape[0]
        if shared_dim != expected_shared_dim:
            raise ValueError("neighboring LETTA tensors have incompatible shared physical legs.")

        transforms = []
        for shared_state in range(shared_dim):
            for group in self._conditional_virtual_bond_groups(bond, shared_state):
                if group.size == 0:
                    continue
                left_matrix, right_matrix = self._conditional_bond_matrices(
                    bond,
                    shared_state,
                    group,
                )
                if direction == "lr":
                    q = None
                    if mode == "qr" and left_matrix.shape[0] >= left_matrix.shape[1]:
                        q, r = np.linalg.qr(left_matrix, mode="reduced")
                        try:
                            gauge = np.linalg.inv(r)
                        except linalg.LinAlgError:
                            q = None
                        else:
                            gauge_inv = r
                            left_matrix = q
                            right_matrix = gauge_inv @ right_matrix
                    if q is None:
                        gram = left_matrix.conj().T @ left_matrix
                        sqrt, inv_sqrt = _hermitian_sqrt_pair(
                            gram,
                            eps=eps,
                            rcond=rcond,
                        )
                        gauge = inv_sqrt
                        gauge_inv = sqrt
                else:
                    q = None
                    if mode == "qr" and right_matrix.shape[1] >= right_matrix.shape[0]:
                        q, r = np.linalg.qr(right_matrix.T, mode="reduced")
                        r = r.T
                        try:
                            gauge_inv = np.linalg.inv(r)
                        except linalg.LinAlgError:
                            q = None
                        else:
                            gauge = r
                            left_matrix = left_matrix @ gauge
                            right_matrix = q.T
                    if q is None:
                        gram = right_matrix @ right_matrix.conj().T
                        sqrt, inv_sqrt = _hermitian_sqrt_pair(
                            gram,
                            eps=eps,
                            rcond=rcond,
                        )
                        gauge = sqrt
                        gauge_inv = inv_sqrt
                transforms.append(
                    (
                        shared_state,
                        group,
                        gauge,
                        gauge_inv,
                    )
                )
        left, right = _apply_conditional_gauges(left, right, transforms)
        self.tensors[bond] = left
        self.tensors[bond + 1] = right
        self._apply_local_masks((bond, bond + 1))
        if normalize:
            self.normalize()
        return self

    def canonicalize_conditional_bonds(
        self,
        *,
        direction="lr",
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=True,
    ):
        """Apply physical-conditioned canonicalization across the chain."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        mode = _validate_gauge_mode(mode)
        bonds = range(self.nlocal_tensors - 1)
        if direction == "rl":
            bonds = reversed(list(bonds))
        for bond in bonds:
            self.canonicalize_conditional_bond(
                bond,
                direction=direction,
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if normalize:
            self.normalize()
        return self

    def canonicalize_conditional_center(
        self,
        tensor_index,
        *,
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=True,
    ):
        """Build a physical-conditioned mixed gauge around one tensor."""
        mode = _validate_gauge_mode(mode)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        for bond in range(tensor_index):
            self.canonicalize_conditional_bond(
                bond,
                direction="lr",
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        for bond in reversed(range(tensor_index, self.nlocal_tensors - 1)):
            self.canonicalize_conditional_bond(
                bond,
                direction="rl",
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if normalize:
            self.normalize()
        return self

    def canonicalize_virtual_bond(
        self,
        bond,
        *,
        direction="lr",
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=False,
    ):
        """Whiten one virtual bond with a state-preserving block gauge."""
        bond = int(bond)
        if bond < 0 or bond >= self.nlocal_tensors - 1:
            raise IndexError("bond index out of range.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        mode = _validate_gauge_mode(mode)
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        if left.ndim != 4:
            return self
        for group in self._virtual_bond_groups(bond):
            if group.size == 0:
                continue
            left_matrix = left[:, :, :, group].reshape(-1, group.size)
            right_matrix = self._right_bond_matrix(right, group)
            if direction == "lr":
                if mode == "qr" and left_matrix.shape[0] >= left_matrix.shape[1]:
                    q, r = np.linalg.qr(left_matrix, mode="reduced")
                    try:
                        gauge = np.linalg.inv(r)
                    except linalg.LinAlgError:
                        gram = left_matrix.conj().T @ left_matrix
                        sqrt, inv_sqrt = _hermitian_sqrt_pair(
                            gram,
                            eps=eps,
                            rcond=rcond,
                        )
                        gauge = inv_sqrt
                        gauge_inv = sqrt
                    else:
                        gauge_inv = r
                        left_matrix = q
                        right_matrix = gauge_inv @ right_matrix
                else:
                    gram = left_matrix.conj().T @ left_matrix
                    sqrt, inv_sqrt = _hermitian_sqrt_pair(
                        gram,
                        eps=eps,
                        rcond=rcond,
                    )
                    gauge = inv_sqrt
                    gauge_inv = sqrt
            else:
                if mode == "qr" and right_matrix.shape[1] >= right_matrix.shape[0]:
                    q, r = np.linalg.qr(right_matrix.T, mode="reduced")
                    r = r.T
                    try:
                        gauge_inv = np.linalg.inv(r)
                    except linalg.LinAlgError:
                        gram = right_matrix @ right_matrix.conj().T
                        sqrt, inv_sqrt = _hermitian_sqrt_pair(
                            gram,
                            eps=eps,
                            rcond=rcond,
                        )
                        gauge = sqrt
                        gauge_inv = inv_sqrt
                    else:
                        gauge = r
                        left_matrix = left_matrix @ gauge
                        right_matrix = q.T
                else:
                    gram = right_matrix @ right_matrix.conj().T
                    sqrt, inv_sqrt = _hermitian_sqrt_pair(
                        gram,
                        eps=eps,
                        rcond=rcond,
                    )
                    gauge = sqrt
                    gauge_inv = inv_sqrt
            self._apply_virtual_gauge(bond, group, gauge, gauge_inv)
        self._apply_local_masks((bond, bond + 1))
        if normalize:
            self.normalize()
        return self

    def canonicalize_virtual_bonds(
        self,
        *,
        direction="lr",
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=True,
    ):
        """Apply moving-center virtual-bond canonicalization across the chain."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        mode = _validate_gauge_mode(mode)
        bonds = range(self.nlocal_tensors - 1)
        if direction == "rl":
            bonds = reversed(list(bonds))
        for bond in bonds:
            self.canonicalize_virtual_bond(
                bond,
                direction=direction,
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if normalize:
            self.normalize()
        return self

    def canonicalize_center(
        self,
        tensor_index,
        *,
        mode="symmetric",
        eps=1.0e-14,
        rcond=1.0e-12,
        normalize=True,
    ):
        """Move LETTA into a mixed gauge around one active tensor."""
        mode = _validate_gauge_mode(mode)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        for bond in range(tensor_index):
            self.canonicalize_virtual_bond(
                bond,
                direction="lr",
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        for bond in reversed(range(tensor_index, self.nlocal_tensors - 1)):
            self.canonicalize_virtual_bond(
                bond,
                direction="rl",
                mode=mode,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if normalize:
            self.normalize()
        return self

    def balance_virtual_bonds(self, *, eps=1.0e-14, normalize=True):
        """Compatibility alias for state-preserving virtual-bond canonicalization."""
        return self.canonicalize_virtual_bonds(direction="lr", eps=eps, normalize=normalize)

    def local_irrep_operators(self, tensor_index):
        """Return Abelian ``IrrepTensor`` blocks for one LETTA tensor."""
        if self.abelian_layout is None:
            raise ValueError("No Abelian LETTA layout is attached.")
        return self.abelian_layout.tensor_operator_grid(
            int(tensor_index),
            self.tensors[int(tensor_index)],
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

    @staticmethod
    def _advance_left_overlap_environment(env, bra_tensor, ket_tensor):
        diagonal = np.einsum("bkxx->bkx", env[:, :, 0], optimize=True)
        advanced = _cached_einsum(
            "bkx,bxuc,kxvd->cduv",
            diagonal,
            bra_tensor.conj(),
            ket_tensor,
        )
        return advanced[:, :, None, :, :]

    def state_overlap(self, other):
        """Return the product-basis overlap ``<self|other>`` by tensor contraction."""
        if not isinstance(other, LETTA):
            raise TypeError("other must be a LETTA instance.")
        if self.dims != other.dims:
            raise ValueError("LETTA overlaps require matching physical dimensions.")
        dtype = np.result_type(
            *[tensor.dtype for tensor in self.tensors],
            *[tensor.dtype for tensor in other.tensors],
        )
        env = np.ones((1, 1, 1, self.dims[0], self.dims[0]), dtype=dtype)
        for bra_tensor, ket_tensor in zip(self.tensors[:self.npairs], other.tensors[:other.npairs]):
            env = self._advance_left_overlap_environment(env, bra_tensor, ket_tensor)
        diagonal = env[:, :, 0]
        bra_terminal = self.tensors[-1] if self.has_terminal_tensor else None
        ket_terminal = other.tensors[-1] if other.has_terminal_tensor else None
        if bra_terminal is not None and ket_terminal is not None:
            return _cached_einsum(
                "bkxx,xb,xk->",
                diagonal,
                bra_terminal.conj(),
                ket_terminal,
            )
        if bra_terminal is not None:
            return _cached_einsum("bkxx,xb->", diagonal, bra_terminal.conj())
        if ket_terminal is not None:
            return _cached_einsum("bkxx,xk->", diagonal, ket_terminal)
        return np.einsum("bkxx->", diagonal, optimize=True)

    def fidelity(self, other):
        """Return the normalized product-basis fidelity with another LETTA state."""
        overlap = self.state_overlap(other)
        norm_self = np.real(self.state_overlap(self))
        norm_other = np.real(other.state_overlap(other))
        denom = norm_self * norm_other
        if denom <= 0.0:
            raise ValueError("State norm is numerically zero.")
        value = abs(overlap) ** 2 / denom
        return float(np.real_if_close(value))

    def norm(self):
        if self.overlap is None:
            norm2 = float(np.real(self._identity_matrix_element()))
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
            return _cached_einsum(
                "bkmxy,mnxy,xb,yk->",
                env,
                mpo[-1],
                self.tensors[-1].conj(),
                self.tensors[-1],
            )
        return _cached_einsum("bkmxy,mnxy->", env, mpo[-1])

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
        denom = self._identity_matrix_element()
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
        value = self._product_matrix_element(operators)
        denom = self._identity_matrix_element()
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return value / denom

    def product_expectation(self, operators):
        """Alias for :meth:`expectation_product_operator`."""
        return self.expectation_product_operator(operators)

    def _product_start_environment(self, dtype):
        return np.ones((1, 1, self.dims[0], self.dims[0]), dtype=dtype)

    def _advance_product_environment(self, env, operator, tensor):
        return _cached_einsum(
            "bkxy,xy,bxuc,kyvd->cduv",
            env,
            operator,
            tensor.conj(),
            tensor,
        )

    def _final_product_closure(self, operator, dtype):
        operator = np.asarray(operator)
        if self.has_terminal_tensor:
            terminal = self.tensors[-1]
            return _cached_einsum(
                "xy,xb,yk->bkxy",
                operator,
                terminal.conj(),
                terminal,
            )
        closure = np.zeros((1, 1, operator.shape[0], operator.shape[1]), dtype=dtype)
        closure[0, 0] = operator
        return closure

    def _retreat_product_closure(self, closure, operator, tensor):
        return _cached_einsum(
            "xy,bxuc,kyvd,cduv->bkxy",
            operator,
            tensor.conj(),
            tensor,
            closure,
        )

    def _identity_product_environments(self, dtype):
        identities = [np.eye(dim, dtype=dtype) for dim in self.dims]
        left = []
        env = self._product_start_environment(dtype)
        left.append(env)
        for site, tensor in enumerate(self.tensors[:self.npairs]):
            env = self._advance_product_environment(env, identities[site], tensor)
            left.append(env)

        right = [None] * self.nsites
        right[-1] = self._final_product_closure(identities[-1], dtype)
        for site in range(self.nsites - 2, -1, -1):
            right[site] = self._retreat_product_closure(right[site + 1], identities[site], self.tensors[site])
        return identities, left, right

    def _single_site_product_closures(self, operator, identity_right, dtype):
        closures = [None] * self.nsites
        closures[-1] = self._final_product_closure(operator, dtype)
        for site in range(self.nsites - 2, -1, -1):
            closures[site] = self._retreat_product_closure(identity_right[site + 1], operator, self.tensors[site])
        return closures

    def _contract_product_environment(self, env, closure):
        return _cached_einsum("bkxy,bkxy->", env, closure)

    def _identity_matrix_element(self):
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        env = np.ones((1, 1, 1, self.dims[0], self.dims[0]), dtype=dtype)
        for tensor in self.tensors[:self.npairs]:
            env = self._advance_left_metric_environment(env, tensor)
        if self.has_terminal_tensor:
            terminal = self.tensors[-1]
            return _cached_einsum(
                "bkxx,xb,xk->",
                env[:, :, 0],
                terminal.conj(),
                terminal,
            )
        return np.einsum("bkxx->", env[:, :, 0], optimize=True)

    def _product_matrix_element(self, operators):
        if len(operators) != self.nsites:
            raise ValueError("number of local operators must match the number of physical sites.")
        operators = [np.asarray(operator) for operator in operators]
        for i, operator in enumerate(operators):
            if operator.shape != (self.dims[i], self.dims[i]):
                raise ValueError(f"operator {i} must have shape ({self.dims[i]}, {self.dims[i]}).")
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[operator.dtype for operator in operators])
        env = self._product_start_environment(dtype)
        for site, tensor in enumerate(self.tensors[:self.npairs]):
            env = self._advance_product_environment(env, operators[site].astype(dtype, copy=False), tensor)
        closure = self._final_product_closure(operators[-1].astype(dtype, copy=False), dtype)
        return self._contract_product_environment(env, closure)

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

        dtype = np.result_type(op_a.dtype, op_b.dtype, *[tensor.dtype for tensor in self.tensors])
        op_a = op_a.astype(dtype, copy=False)
        op_b = op_b.astype(dtype, copy=False)
        op_ab = (op_a @ op_b).astype(dtype, copy=False)
        identities, left, identity_right = self._identity_product_environments(dtype)
        norm = self._contract_product_environment(left[0], identity_right[0])
        if abs(norm) < 1e-14:
            raise ValueError("State norm is numerically zero.")

        right_a = self._single_site_product_closures(op_a, identity_right, dtype)
        right_b = self._single_site_product_closures(op_b, identity_right, dtype)
        right_ab = self._single_site_product_closures(op_ab, identity_right, dtype)
        one_a = np.empty(self.nsites, dtype=complex)
        one_b = np.empty(self.nsites, dtype=complex)
        corr = np.empty((self.nsites, self.nsites), dtype=complex)

        for site in range(self.nsites):
            one_a[site] = self._contract_product_environment(left[site], right_a[site]) / norm
            one_b[site] = self._contract_product_environment(left[site], right_b[site]) / norm
            corr[site, site] = self._contract_product_environment(left[site], right_ab[site]) / norm

        for i in range(self.nsites - 1):
            env = self._advance_product_environment(left[i], op_a, self.tensors[i])
            for j in range(i + 1, self.nsites):
                corr[i, j] = self._contract_product_environment(env, right_b[j]) / norm
                if j < self.nsites - 1:
                    env = self._advance_product_environment(env, identities[j], self.tensors[j])

        for j in range(self.nsites - 1):
            env = self._advance_product_environment(left[j], op_b, self.tensors[j])
            for i in range(j + 1, self.nsites):
                corr[i, j] = self._contract_product_environment(env, right_a[i]) / norm
                if i < self.nsites - 1:
                    env = self._advance_product_environment(env, identities[i], self.tensors[i])

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

    def _left_local_environments(self, mpo, *, verbose=0):
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
            if int(verbose) > 1:
                print(f"letta-mpo left env {i + 1}/{nprefix}", flush=True)
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
        return _cached_einsum(
            "bkmxy,mnxy,bxuc,kyvd->cdnuv",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
        )

    def _right_local_environments(self, mpo, *, verbose=0):
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
        total = max(self.npairs - 1, 0)
        for step, i in enumerate(range(self.npairs - 1, 0, -1), start=1):
            tensor = self.tensors[i]
            env = self._advance_right_environment(env, mpo[i + 1], tensor)
            right[i - 1] = env
            if int(verbose) > 1:
                print(f"letta-mpo right env {step}/{total}", flush=True)
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
        env[:, :, 0, :, :] = _cached_einsum(
            "uc,vd->cduv",
            terminal.conj(),
            terminal,
        )
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
        return _cached_einsum(
            "cdnuv,mnuv,bxuc,kyvd->bkmxy",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
        )

    def _left_metric_environments(self, *, verbose=0):
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        left = []
        env = np.ones((1, 1, 1, self.dims[0], self.dims[0]), dtype=dtype)
        left.append(env)
        total = max(len(self.tensors) - 1, 0)
        for i, tensor in enumerate(self.tensors[:-1], start=1):
            env = self._advance_left_metric_environment(env, tensor)
            left.append(env)
            if int(verbose) > 1:
                print(f"letta-mpo left metric {i}/{total}", flush=True)
        return left

    def _advance_left_metric_environment(self, env, tensor):
        diagonal = np.einsum("bkxx->bkx", env[:, :, 0], optimize=True)
        advanced = _cached_einsum(
            "bkx,bxuc,kxvd->cduv",
            diagonal,
            tensor.conj(),
            tensor,
        )
        return advanced[:, :, None, :, :]

    def _right_metric_environments(self, *, verbose=0):
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        right = [None] * self.nlocal_tensors
        if self.has_terminal_tensor:
            env = self._terminal_right_environment(dtype=dtype)
            right[self.npairs - 1] = env
        else:
            env = np.ones((1, 1, 1, self.dims[-1], self.dims[-1]), dtype=dtype)
            right[-1] = env
        total = max(self.npairs - 1, 0)
        for step, i in enumerate(range(self.npairs - 1, 0, -1), start=1):
            env = self._advance_right_metric_environment(env, self.tensors[i])
            right[i - 1] = env
            if int(verbose) > 1:
                print(f"letta-mpo right metric {step}/{total}", flush=True)
        return right

    def _advance_right_metric_environment(self, env, tensor):
        diagonal = np.einsum("cduu->cdu", env[:, :, 0], optimize=True)
        advanced = _cached_einsum(
            "cdu,bxuc,kyud->bkxy",
            diagonal,
            tensor.conj(),
            tensor,
        )
        return advanced[:, :, None, :, :]

    def _local_effective_from_environments(self, mpo, tensor_index, left_envs, right_envs):
        tensor_index = int(tensor_index)
        shape = self.tensors[tensor_index].shape
        heff = _cached_einsum(
            "bkmxy,mpxy,pnuv,cdnuv->bxuckyvd",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
        )
        return heff.reshape(int(np.prod(shape)), int(np.prod(shape)))

    def _terminal_effective_from_environment(self, mpo, left_envs):
        terminal_index = self.npairs
        terminal = self.tensors[terminal_index]
        heff = _cached_einsum(
            "cdmxy,mnxy->xcyd",
            left_envs[terminal_index],
            mpo[-1],
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
        out = _cached_einsum(
            "bkmxy,mpxy,pnuv,cdnuv,kyvd->bxuc",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            theta,
        )
        return out.reshape(-1)

    def _apply_local_effective_batch_from_environments(self, mpo, tensor_index, left_envs, right_envs, vectors):
        tensor_index = int(tensor_index)
        vectors = np.asarray(vectors)
        theta = vectors.reshape((vectors.shape[0],) + self.tensors[tensor_index].shape)
        out = _cached_einsum(
            "bkmxy,mpxy,pnuv,cdnuv,qkyvd->qbxuc",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            theta,
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
        out = _cached_einsum(
            "bkx,cdu,kxud->bxuc",
            left_diag,
            right_diag,
            theta,
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
        out = _cached_einsum(
            "bkx,cdu,qkxud->qbxuc",
            left_diag,
            right_diag,
            theta,
        )
        return out.reshape(vectors.shape[0], -1)

    def _solve_one_site_mpo_in_support(
        self,
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
        mask,
        *,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
        assume_identity_metric=False,
    ):
        tensor_index = int(tensor_index)
        local_dim = int(np.prod(self.tensors[tensor_index].shape))
        abelian_plan = self._local_abelian_support_plan(tensor_index, mask)
        if abelian_plan is None:
            support_indices, coords, physical_groups = self._local_support_plan(tensor_index, mask)
            metric_block_builder = None
        else:
            support_indices = abelian_plan.flat_indices
            coords = abelian_plan.coords
            physical_groups = abelian_plan.physical_groups
            metric_block_builder = abelian_plan
        support_size = int(support_indices.size)
        if support_size == 0:
            raise ValueError("symmetry/support mask removes every local tensor entry.")
        dtype = np.result_type(
            self.tensors[tensor_index].dtype,
            *[site.dtype for site in mpo],
            metric_left[tensor_index].dtype,
            metric_right[tensor_index].dtype,
            complex,
        )
        shape = self.tensors[tensor_index].shape
        left = left_envs[tensor_index]
        right = right_envs[tensor_index]
        w_left = mpo[tensor_index]
        w_right = mpo[tensor_index + 1]

        def build_heff():
            if local_dim <= _DEFAULT_MASKED_DENSE_SLICE_LOCAL_DIM:
                full_heff = self._local_effective_from_environments(
                    mpo,
                    tensor_index,
                    left_envs,
                    right_envs,
                )
                return full_heff[np.ix_(support_indices, support_indices)]
            transitions = _two_site_mpo_sparse_transitions(w_left, w_right)
            if transitions is not None and transitions.ntransitions:
                return _support_kernel_assemble(
                    coords,
                    np.asarray(left),
                    np.asarray(right),
                    transitions.bra_i,
                    transitions.ket_i,
                    transitions.bra_j,
                    transitions.ket_j,
                    transitions.entry_starts,
                    transitions.entry_m,
                    transitions.entry_n,
                    transitions.entry_values,
                ).astype(dtype, copy=False)
            sparse_entries = _two_site_mpo_sparse_entries(w_left, w_right)
            if _support_heff_sparse_numba is not None and sparse_entries is not None:
                entry_starts, entry_m, entry_n, entry_values = sparse_entries
                return _support_heff_sparse_numba(
                    coords.astype(np.int64, copy=False),
                    np.asarray(left),
                    np.asarray(right),
                    entry_starts,
                    entry_m,
                    entry_n,
                    entry_values,
                ).astype(dtype, copy=False)
            if _support_heff_indexed_numba is not None:
                return _support_heff_indexed_numba(
                    coords.astype(np.int64, copy=False),
                    np.asarray(left),
                    np.asarray(right),
                    np.asarray(w_left),
                    np.asarray(w_right),
                ).astype(dtype, copy=False)

            heff = np.empty((support_size, support_size), dtype=dtype)
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
            return heff

        def build_sparse_heff():
            transitions = _two_site_mpo_sparse_transitions(w_left, w_right)
            if transitions is not None:
                return _support_heff_sparse_by_transitions(
                    coords,
                    np.asarray(left),
                    np.asarray(right),
                    transitions,
                    dtype=dtype,
                    physical_groups=physical_groups,
                )
            sparse_entries = _two_site_mpo_sparse_entries(w_left, w_right)
            if sparse_entries is None:
                return None
            entry_starts, entry_m, entry_n, entry_values = sparse_entries
            return _support_heff_sparse_by_physical_blocks(
                coords,
                np.asarray(left),
                np.asarray(right),
                entry_starts,
                entry_m,
                entry_n,
                entry_values,
                dtype=dtype,
                physical_groups=physical_groups,
            )

        if assume_identity_metric:
            if matrix_free_threshold is None:
                matrix_free_threshold = _DEFAULT_MATRIX_FREE_LOCAL_DIM
            solver = _validate_local_solver(local_solver)
            dense_bytes = support_size * support_size * np.dtype(dtype).itemsize
            memory_limited = (
                matrix_free_memory_limit is not None
                and dense_bytes > int(matrix_free_memory_limit)
            )
            use_matrix_free = solver in {"matrix_free", "direct"} or (
                solver == "auto"
                and (support_size > int(matrix_free_threshold) or memory_limited)
            )
            use_abelian_sparse = (
                self.abelian_layout is not None
                and support_size >= _DEFAULT_ABELIAN_SPARSE_SUPPORT_DIM
            )
            use_matrix_free = use_matrix_free or (solver == "auto" and use_abelian_sparse)

            if use_matrix_free:
                current = self.tensors[tensor_index].reshape(-1)[support_indices]
                current_norm = np.linalg.norm(current)
                v0 = current / current_norm if current_norm > 1.0e-14 else None
                direct_transitions = (
                    _two_site_mpo_sparse_transitions(w_left, w_right)
                    if solver == "direct"
                    else None
                )
                sparse_heff = None if direct_transitions is not None else (
                    build_sparse_heff() if self.abelian_layout is not None else None
                )

                def matvec(vector):
                    if direct_transitions is not None:
                        return _apply_support_heff_by_transitions(
                            coords,
                            np.asarray(left),
                            np.asarray(right),
                            direct_transitions,
                            vector,
                            dtype=dtype,
                            physical_groups=physical_groups,
                        )
                    if sparse_heff is not None:
                        return sparse_heff @ vector
                    full_vector = np.zeros(
                        local_dim,
                        dtype=np.result_type(dtype, np.asarray(vector).dtype),
                    )
                    full_vector[support_indices] = vector
                    return self._apply_local_effective_from_environments(
                        mpo,
                        tensor_index,
                        left_envs,
                        right_envs,
                        full_vector,
                    )[support_indices]

                def matmat(vectors):
                    vectors = np.asarray(vectors)
                    if direct_transitions is not None:
                        return _apply_support_heff_by_transitions(
                            coords,
                            np.asarray(left),
                            np.asarray(right),
                            direct_transitions,
                            vectors,
                            dtype=dtype,
                            physical_groups=physical_groups,
                        )
                    if sparse_heff is not None:
                        return sparse_heff @ vectors
                    return np.column_stack(
                        [matvec(vectors[:, column]) for column in range(vectors.shape[1])]
                    )

                operator = LinearOperator(
                    (support_size, support_size),
                    matvec=matvec,
                    matmat=matmat,
                    dtype=dtype,
                )
                v0 = _active_projected_warm_start(
                    matvec,
                    v0,
                    support_size,
                    dtype=dtype,
                )
                try:
                    evals, evecs = eigsh(
                        operator,
                        k=1,
                        which="SA",
                        tol=matrix_free_tol,
                        maxiter=matrix_free_maxiter or max(1000, 20 * support_size),
                        v0=v0,
                    )
                    energy = float(np.real(evals[0]))
                    support_vector = evecs[:, 0]
                except Exception as exc:
                    if support_size > int(matrix_free_fallback_dim):
                        raise RuntimeError(
                            "Matrix-free LETTA support eigensolve failed and the "
                            f"support dimension ({support_size}) is too large for "
                            "the dense fallback. Increase matrix_free_maxiter, "
                            "loosen matrix_free_tol, or use local_solver='dense' "
                            "for a smaller bond dimension."
                        ) from exc
                    columns = [
                        matvec(np.eye(support_size, dtype=dtype)[:, j])
                        for j in range(support_size)
                    ]
                    reduced_h = np.column_stack(columns)
                    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
                    energy, support_vector = _lowest_hermitian_eigenpair(reduced_h)
            else:
                heff = build_heff()
                heff = 0.5 * (heff + heff.conj().T)
                energy, support_vector = _lowest_hermitian_eigenpair(heff)

            vector = np.zeros(local_dim, dtype=np.result_type(support_vector.dtype, dtype))
            vector[support_indices] = support_vector
            return energy, vector

        mleft = metric_left[tensor_index]
        mright = metric_right[tensor_index]
        if (
            mleft.shape[2] == 1
            and mright.shape[2] == 1
            and mleft.shape[3] == mleft.shape[4]
            and mright.shape[3] == mright.shape[4]
        ):
            if metric_block_builder is None:
                blocks = _metric_blocks_from_support(coords, mleft, mright)
            else:
                blocks = _metric_blocks_from_abelian_plan(metric_block_builder, mleft, mright)
            reduced_dim = _metric_blocks_reduced_dim(blocks)
            if matrix_free_threshold is None:
                matrix_free_threshold = _DEFAULT_MATRIX_FREE_LOCAL_DIM
            solver = _validate_local_solver(local_solver)
            dense_bytes = support_size * support_size * np.dtype(dtype).itemsize
            memory_limited = (
                matrix_free_memory_limit is not None
                and dense_bytes > int(matrix_free_memory_limit)
            )
            use_matrix_free = solver in {"matrix_free", "direct"} or (
                solver == "auto"
                and (support_size > int(matrix_free_threshold) or memory_limited)
            )
            use_abelian_sparse = (
                self.abelian_layout is not None
                and support_size >= _DEFAULT_ABELIAN_SPARSE_SUPPORT_DIM
            )
            use_matrix_free = use_matrix_free or (solver == "auto" and use_abelian_sparse)

            if use_matrix_free:
                current_support = self.tensors[tensor_index].reshape(-1)[support_indices]
                direct_reduced_heff = None
                if (
                    solver != "direct"
                    and self.abelian_layout is not None
                    and reduced_dim <= _DEFAULT_DIRECT_REDUCED_HEFF_DIM
                ):
                    transitions = _two_site_mpo_sparse_transitions(w_left, w_right)
                    if transitions is not None:
                        direct_reduced_heff = _reduced_heff_sparse_by_transitions(
                            coords,
                            np.asarray(left),
                            np.asarray(right),
                            transitions,
                            blocks,
                            dtype=dtype,
                        )
                if direct_reduced_heff is not None:
                    direct_reduced_heff = 0.5 * (
                        direct_reduced_heff + direct_reduced_heff.conj().T
                    )
                    v0 = _support_to_reduced_vector(current_support, blocks, dtype=dtype)
                    v0_norm = np.linalg.norm(v0)
                    if v0_norm > 1e-14:
                        v0 = v0 / v0_norm
                    else:
                        v0 = None
                    try:
                        if reduced_dim <= 256:
                            energy, reduced_vector = _lowest_hermitian_eigenpair(
                                direct_reduced_heff.toarray()
                            )
                        else:
                            evals, evecs = eigsh(
                                direct_reduced_heff,
                                k=1,
                                which="SA",
                                tol=matrix_free_tol,
                                maxiter=matrix_free_maxiter or max(1000, 20 * reduced_dim),
                                v0=v0,
                            )
                            energy = float(np.real(evals[0]))
                            reduced_vector = evecs[:, 0]
                        support_vector = _reduced_to_support_vector(
                            reduced_vector,
                            blocks,
                            support_size,
                            dtype=dtype,
                        )
                        vector = np.zeros(local_dim, dtype=np.result_type(support_vector.dtype, dtype))
                        vector[support_indices] = support_vector
                        return energy, vector
                    except Exception:
                        try:
                            energy, reduced_vector = _lowest_sparse_lobpcg_with_diagonal_preconditioner(
                                direct_reduced_heff,
                                v0=v0,
                                tol=matrix_free_tol,
                                maxiter=matrix_free_maxiter,
                            )
                            support_vector = _reduced_to_support_vector(
                                reduced_vector,
                                blocks,
                                support_size,
                                dtype=dtype,
                            )
                            vector = np.zeros(local_dim, dtype=np.result_type(support_vector.dtype, dtype))
                            vector[support_indices] = support_vector
                            return energy, vector
                        except Exception:
                            pass

                basis_operator = _metric_blocks_sparse_basis(blocks, support_size, dtype=dtype)

                def support_to_reduced(vector):
                    return np.asarray(basis_operator.conj().T @ vector).reshape(-1)

                def reduced_to_support(coeff):
                    return np.asarray(basis_operator @ coeff).reshape(-1)

                v0 = support_to_reduced(current_support)
                v0_norm = np.linalg.norm(v0)
                if v0_norm > 1e-14:
                    v0 = v0 / v0_norm
                else:
                    v0 = None
                direct_transitions = None
                if solver == "direct":
                    direct_transitions = _two_site_mpo_sparse_transitions(w_left, w_right)
                sparse_heff = None if direct_transitions is not None else (
                    build_sparse_heff() if self.abelian_layout is not None else None
                )

                def matvec(coeff):
                    support_vector = reduced_to_support(coeff)
                    if direct_transitions is not None:
                        applied = _apply_support_heff_by_transitions(
                            coords,
                            np.asarray(left),
                            np.asarray(right),
                            direct_transitions,
                            support_vector,
                            dtype=dtype,
                            physical_groups=physical_groups,
                        )
                    elif sparse_heff is not None:
                        applied = sparse_heff @ support_vector
                    else:
                        full_vector = np.zeros(local_dim, dtype=np.result_type(dtype, support_vector.dtype))
                        full_vector[support_indices] = support_vector
                        applied = self._apply_local_effective_from_environments(
                            mpo,
                            tensor_index,
                            left_envs,
                            right_envs,
                            full_vector,
                        )[support_indices]
                    return support_to_reduced(applied)

                operator = LinearOperator(
                    (reduced_dim, reduced_dim),
                    matvec=matvec,
                    dtype=dtype,
                )
                v0 = _active_projected_warm_start(
                    matvec,
                    v0,
                    reduced_dim,
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
                            "Matrix-free LETTA support eigensolve failed and the "
                            f"reduced dimension ({reduced_dim}) is too large for "
                            "the dense fallback. Increase matrix_free_maxiter, "
                            "loosen matrix_free_tol, or use local_solver='dense' "
                            "for a smaller bond dimension."
                        ) from exc
                    columns = [
                        matvec(np.eye(reduced_dim, dtype=dtype)[:, j])
                        for j in range(reduced_dim)
                    ]
                    reduced_h = np.column_stack(columns)
                    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
                    energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
                support_vector = reduced_to_support(reduced_vector)
                vector = np.zeros(local_dim, dtype=np.result_type(support_vector.dtype, dtype))
                vector[support_indices] = support_vector
                return energy, vector

            heff = build_heff()
            heff = 0.5 * (heff + heff.conj().T)
            energy, support_vector = _lowest_generalized_eigenpair_from_metric_blocks(heff, blocks)
            vector = np.zeros(local_dim, dtype=np.result_type(support_vector.dtype, dtype))
            vector[support_indices] = support_vector
            return energy, vector

        heff = build_heff()
        s_cols = []
        batch_size = min(_DEFAULT_SUPPORT_BATCH_SIZE, support_size)
        for start in range(0, support_size, batch_size):
            batch = support_indices[start:start + batch_size]
            vectors = np.zeros((batch.size, local_dim), dtype=dtype)
            vectors[np.arange(batch.size), batch] = 1
            s_cols.append(
                self._apply_local_metric_batch_from_environments(
                    tensor_index, metric_left, metric_right, vectors
                )[:, support_indices].T
            )
        seff = np.concatenate(s_cols, axis=1)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        energy, reduced_vector = _lowest_generalized_eigenpair(heff, seff)
        vector = np.zeros(local_dim, dtype=np.result_type(reduced_vector.dtype, dtype))
        vector[support_indices] = reduced_vector
        return energy, vector

    def _solve_terminal_mpo_with_environments(
        self,
        mpo,
        left_envs,
        metric_left,
        *,
        assume_identity_metric=False,
    ):
        terminal_index = self.npairs
        local_dim = int(np.prod(self.tensors[terminal_index].shape))
        heff = self._terminal_effective_from_environment(mpo, left_envs)
        heff = 0.5 * (heff + heff.conj().T)

        local_mask = self.local_masks[terminal_index]
        if assume_identity_metric:
            if local_mask is None:
                return _lowest_hermitian_eigenpair(heff)
            allowed = np.flatnonzero(np.asarray(local_mask, dtype=bool).reshape(-1))
            if allowed.size == 0:
                raise ValueError("symmetry/support mask removes every terminal tensor entry.")
            energy, reduced_vector = _lowest_hermitian_eigenpair(heff[np.ix_(allowed, allowed)])
            vector = np.zeros(local_dim, dtype=np.result_type(reduced_vector.dtype, heff.dtype))
            vector[allowed] = reduced_vector
            return energy, vector

        seff = self._terminal_metric_from_environment(metric_left)
        seff = 0.5 * (seff + seff.conj().T)
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

    def _local_metric_is_identity(
        self,
        tensor_index,
        metric_left,
        metric_right,
        *,
        atol=1.0e-10,
    ):
        """Return whether the current local norm metric is the identity.

        Conditional canonicalization makes every physical-diagonal left and
        right environment block an identity matrix.  Checking those small
        blocks avoids constructing the full local metric solely to decide
        whether the NARG-like standard eigenproblem is valid.
        """
        tensor_index = int(tensor_index)
        mask = self.local_masks[tensor_index]
        if self.has_terminal_tensor and tensor_index == self.npairs:
            left = metric_left[tensor_index]
            if left.shape[2] != 1 or left.shape[3] != left.shape[4]:
                return False
            left = left[:, :, 0]
            if mask is not None:
                for physical in range(left.shape[2]):
                    allowed = np.flatnonzero(mask[physical])
                    block = left[:, :, physical, physical][np.ix_(allowed, allowed)]
                    if not np.allclose(
                        block,
                        np.eye(allowed.size, dtype=block.dtype),
                        rtol=0.0,
                        atol=float(atol),
                    ):
                        return False
                return True
            identity = np.eye(left.shape[0], dtype=left.dtype)
            return all(
                np.allclose(
                    left[:, :, physical, physical],
                    identity,
                    rtol=0.0,
                    atol=float(atol),
                )
                for physical in range(left.shape[2])
            )

        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        if (
            left.shape[2] != 1
            or right.shape[2] != 1
            or left.shape[3] != left.shape[4]
            or right.shape[3] != right.shape[4]
        ):
            return False
        left = left[:, :, 0]
        right = right[:, :, 0]
        if mask is not None:
            for si in range(left.shape[2]):
                left_block = left[:, :, si, si]
                for sj in range(right.shape[2]):
                    coords = np.argwhere(mask[:, si, sj, :])
                    if not coords.size:
                        continue
                    left_indices = coords[:, 0]
                    right_indices = coords[:, 1]
                    block = (
                        left_block[left_indices[:, None], left_indices[None, :]]
                        * right[:, :, sj, sj][right_indices[:, None], right_indices[None, :]]
                    )
                    if not np.allclose(
                        block,
                        np.eye(coords.shape[0], dtype=block.dtype),
                        rtol=0.0,
                        atol=float(atol),
                    ):
                        return False
            return True
        left_identity = np.eye(left.shape[0], dtype=left.dtype)
        right_identity = np.eye(right.shape[0], dtype=right.dtype)
        for physical in range(left.shape[2]):
            if not np.allclose(
                left[:, :, physical, physical],
                left_identity,
                rtol=0.0,
                atol=float(atol),
            ):
                return False
        for physical in range(right.shape[2]):
            if not np.allclose(
                right[:, :, physical, physical],
                right_identity,
                rtol=0.0,
                atol=float(atol),
            ):
                return False
        return True

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
            keep = vals > metric_tol * (float(np.max(np.abs(vals))) if vals.size else 0.0)
            vals, vecs = vals[keep], vecs[:, keep]
            left_eigs.append((vals, vecs))
        for sj in range(dj):
            block = 0.5 * (right[:, :, sj, sj] + right[:, :, sj, sj].conj().T)
            vals, vecs = linalg.eigh(block, check_finite=False)
            keep = vals > metric_tol * (float(np.max(np.abs(vals))) if vals.size else 0.0)
            vals, vecs = vals[keep], vecs[:, keep]
            right_eigs.append((vals, vecs))
        for vals_l, _ in left_eigs:
            if not vals_l.size:
                continue
            for vals_r, _ in right_eigs:
                if vals_r.size:
                    max_metric_eval = max(max_metric_eval, float(np.max(vals_l) * np.max(vals_r)))

        threshold = metric_tol * max_metric_eval
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
        assume_identity_metric=False,
    ):
        tensor_index = int(tensor_index)
        if self.has_terminal_tensor and tensor_index == self.npairs:
            return self._solve_terminal_mpo_with_environments(
                mpo,
                left_envs,
                metric_left,
                assume_identity_metric=assume_identity_metric,
            )

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
                local_solver=local_solver,
                matrix_free_threshold=matrix_free_threshold,
                matrix_free_memory_limit=matrix_free_memory_limit,
                matrix_free_tol=matrix_free_tol,
                matrix_free_maxiter=matrix_free_maxiter,
                matrix_free_fallback_dim=matrix_free_fallback_dim,
                assume_identity_metric=assume_identity_metric,
            )
        local_dim = int(np.prod(self.tensors[tensor_index].shape))
        if assume_identity_metric:
            basis = None
            reduced_dim = local_dim
            dtype = np.result_type(self.tensors[tensor_index].dtype, *[site.dtype for site in mpo])
        else:
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
        use_matrix_free = solver in {"matrix_free", "direct"} or (
            solver == "auto"
            and (local_dim > int(matrix_free_threshold) or memory_limited)
        )

        if use_matrix_free:
            current = self.tensors[tensor_index].reshape(-1)
            v0 = current if basis is None else basis.conj().T @ current
            v0_norm = np.linalg.norm(v0)
            if v0_norm > 1e-14:
                v0 = v0 / v0_norm
            else:
                v0 = None

            def matvec(coeff):
                ket = coeff if basis is None else basis @ coeff
                applied = self._apply_local_effective_from_environments(
                    mpo, tensor_index, left_envs, right_envs, ket
                )
                return applied if basis is None else basis.conj().T @ applied

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
            reduced_h = heff if basis is None else basis.conj().T @ heff @ basis
            energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
        vector = reduced_vector if basis is None else basis @ reduced_vector
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
        assume_identity_metric=False,
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
        if assume_identity_metric and not self._local_metric_is_identity(
            tensor_index,
            metric_left,
            metric_right,
        ):
            raise ValueError(
                "assume_identity_metric requires a conditionally canonical "
                "full-rank local center."
            )
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
            assume_identity_metric=assume_identity_metric,
        )
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks[tensor_index] is not None:
            self.tensors[tensor_index] = np.where(
                self.local_masks[tensor_index],
                self.tensors[tensor_index],
                0,
            )
        return {
            "tensor": tensor_index,
            "local_energy": float(local_energy),
            "identity_metric": bool(assume_identity_metric),
        }

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
        assume_identity_metric=False,
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
            assume_identity_metric=assume_identity_metric,
        )
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks[tensor_index] is not None:
            self.tensors[tensor_index] = np.where(
                self.local_masks[tensor_index],
                self.tensors[tensor_index],
                0,
            )
        return {
            "tensor": int(tensor_index),
            "local_energy": float(local_energy),
            "identity_metric": bool(assume_identity_metric),
        }

    def sweep(
        self,
        direction="lr",
        operator=None,
        *,
        local_solver="auto",
        matrix_free_threshold=None,
        matrix_free_memory_limit=_DEFAULT_MATRIX_FREE_MEMORY_LIMIT,
        matrix_free_tol=1e-9,
        matrix_free_maxiter=None,
        matrix_free_fallback_dim=_DEFAULT_MATRIX_FREE_FALLBACK_DIM,
        gauge=None,
        canonicalize="symmetric",
        identity_metric=None,
        metric_tol=1.0e-10,
        adapt_bonds=None,
        compress_rtol=1.0e-12,
        compress_atol=0.0,
        max_bond_dim=None,
        verbose=0,
        _canonicalize_start=True,
    ):
        """
        Perform one one-site variational sweep over tied tensors.
        """
        if operator is not None:
            if self._looks_like_mpo(operator):
                return self._sweep_mpo(
                    operator,
                    direction,
                    local_solver=local_solver,
                    matrix_free_threshold=matrix_free_threshold,
                    matrix_free_memory_limit=matrix_free_memory_limit,
                    matrix_free_tol=matrix_free_tol,
                    matrix_free_maxiter=matrix_free_maxiter,
                    matrix_free_fallback_dim=matrix_free_fallback_dim,
                    gauge=gauge,
                    canonicalize=canonicalize,
                    identity_metric=identity_metric,
                    metric_tol=metric_tol,
                    adapt_bonds=adapt_bonds,
                    compress_rtol=compress_rtol,
                    compress_atol=compress_atol,
                    max_bond_dim=max_bond_dim,
                    verbose=verbose,
                    _canonicalize_start=_canonicalize_start,
                )
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

    def _sweep_mpo(
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
        gauge=None,
        canonicalize="symmetric",
        identity_metric=None,
        metric_tol=1.0e-10,
        adapt_bonds=None,
        compress_rtol=1.0e-12,
        compress_atol=0.0,
        max_bond_dim=None,
        verbose=0,
        _canonicalize_start=True,
    ):
        """
        Perform one one-site sweep using MPO-contracted local Hamiltonians.

        ``gauge='conditional'`` applies the LETTA-specific
        physical-conditioned canonical gauge while moving the optimization
        center.  ``'virtual'`` selects the physical-independent bond gauge.

        With the conditional gauge, ``identity_metric`` defaults to true.
        The starting center is canonicalized before environments are built,
        and each full-rank center is solved as a standard Hermitian
        eigenproblem with identity norm metric.  Rank-deficient centers fall
        back to the generalized-metric solver.  Adaptive bond compression also
        defaults to true in this mode and removes the redundant directions
        that would otherwise cause that fallback.
        """
        mpo = self._validate_mpo(mpo)
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        gauge_mode = _validate_gauge(gauge)
        canonicalize_mode = _validate_gauge_mode(canonicalize)
        if identity_metric is None:
            identity_metric = gauge_mode == "conditional"
        identity_metric = bool(identity_metric)
        if identity_metric and gauge_mode != "conditional":
            raise ValueError("identity_metric requires gauge='conditional'.")
        if adapt_bonds is None:
            adapt_bonds = identity_metric
        adapt_bonds = bool(adapt_bonds)
        if adapt_bonds and gauge_mode != "conditional":
            raise ValueError("adapt_bonds requires gauge='conditional'.")
        precompression = []
        if adapt_bonds:
            compression_direction = "rl" if direction == "lr" else "lr"
            precompression = self.compress_conditional_bonds(
                direction=compression_direction,
                rtol=compress_rtol,
                atol=compress_atol,
                max_bond_dim=max_bond_dim,
            )
        if identity_metric and bool(_canonicalize_start):
            start = 0 if direction == "lr" else self.nlocal_tensors - 1
            self.canonicalize_conditional_center(
                start,
                mode=canonicalize_mode,
                normalize=False,
            )
        updates = []
        package = LETTAOperatorPackage.for_sweep(self, mpo, direction, verbose=verbose)

        if direction == "lr":
            for i in range(self.nlocal_tensors):
                use_identity_metric = identity_metric and self._local_metric_is_identity(
                    i,
                    package.metric_left,
                    package.metric_right,
                    atol=metric_tol,
                )
                update = self._optimize_tensor_mpo_with_environments(
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
                    assume_identity_metric=use_identity_metric,
                )
                if not updates and precompression:
                    update["precompression"] = precompression
                updates.append(update)
                if int(verbose) > 1:
                    print(f"letta-mpo tensor {i:>2} | dir=lr | E={update['local_energy']:.12g}")
                if gauge_mode is not None and i + 1 < self.nlocal_tensors:
                    if adapt_bonds:
                        update["compression"] = self.compress_conditional_bond(
                            i,
                            direction="lr",
                            rtol=compress_rtol,
                            atol=compress_atol,
                            max_bond_dim=max_bond_dim,
                        )
                    if gauge_mode == "conditional":
                        self.canonicalize_conditional_bond(
                            i,
                            direction="lr",
                            mode=canonicalize_mode,
                            normalize=False,
                        )
                    else:
                        self.canonicalize_virtual_bond(
                            i,
                            direction="lr",
                            mode=canonicalize_mode,
                            normalize=False,
                        )
                package.advance_after_update(i)
        else:
            for i in reversed(range(self.nlocal_tensors)):
                use_identity_metric = identity_metric and self._local_metric_is_identity(
                    i,
                    package.metric_left,
                    package.metric_right,
                    atol=metric_tol,
                )
                update = self._optimize_tensor_mpo_with_environments(
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
                    assume_identity_metric=use_identity_metric,
                )
                if not updates and precompression:
                    update["precompression"] = precompression
                updates.append(update)
                if int(verbose) > 1:
                    print(f"letta-mpo tensor {i:>2} | dir=rl | E={update['local_energy']:.12g}")
                if gauge_mode is not None and i > 0:
                    if adapt_bonds:
                        update["compression"] = self.compress_conditional_bond(
                            i - 1,
                            direction="rl",
                            rtol=compress_rtol,
                            atol=compress_atol,
                            max_bond_dim=max_bond_dim,
                        )
                    if gauge_mode == "conditional":
                        self.canonicalize_conditional_bond(
                            i - 1,
                            direction="rl",
                            mode=canonicalize_mode,
                            normalize=False,
                        )
                    else:
                        self.canonicalize_virtual_bond(
                            i - 1,
                            direction="rl",
                            mode=canonicalize_mode,
                            normalize=False,
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
        gauge="conditional",
        canonicalize="symmetric",
        identity_metric=None,
        metric_tol=1.0e-10,
        adapt_bonds=None,
        compress_rtol=1.0e-12,
        compress_atol=0.0,
        max_bond_dim=None,
    ):
        """
        Run one-site LETTA variational sweeps.

        ``operator`` may be omitted to use the stored dense Hamiltonian, or may
        be supplied as a dense matrix or MPO.

        The default physical-conditioned canonical gauge makes the overlap
        metric of the next local problem identity on its nonsingular support.
        Use ``gauge='virtual'`` for the older bond-only balancing or ``None``
        to disable canonicalization.  For MPO sweeps, ``identity_metric``
        defaults to true with the conditional gauge and uses that identity
        directly instead of constructing a metric-whitening basis.  Exact
        adaptive bond compression is enabled at the same time; set
        ``adapt_bonds=False`` to retain fixed tensor shapes.
        ``max_bond_dim`` enables additional lossy truncation, with
        discarded weights recorded in each local update.
        """
        canonicalize_mode = _validate_gauge_mode(canonicalize)
        mpo = None
        if operator is not None:
            if self._looks_like_mpo(operator):
                mpo = self._validate_mpo(operator)
            else:
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
        previous_sweep_direction = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            reuse_canonical_center = (
                mpo is not None
                and previous_sweep_direction is not None
                and direction != previous_sweep_direction
                and _validate_gauge(gauge) == "conditional"
                and identity_metric is not False
                and adapt_bonds is False
            )
            if mpo is not None:
                updates = self.sweep(
                    direction,
                    mpo,
                    local_solver=local_solver,
                    matrix_free_threshold=matrix_free_threshold,
                    matrix_free_memory_limit=matrix_free_memory_limit,
                    matrix_free_tol=matrix_free_tol,
                    matrix_free_maxiter=matrix_free_maxiter,
                    matrix_free_fallback_dim=matrix_free_fallback_dim,
                    gauge=gauge,
                    canonicalize=canonicalize_mode,
                    identity_metric=identity_metric,
                    metric_tol=metric_tol,
                    adapt_bonds=adapt_bonds,
                    compress_rtol=compress_rtol,
                    compress_atol=compress_atol,
                    max_bond_dim=max_bond_dim,
                    verbose=verbose,
                    _canonicalize_start=not reuse_canonical_center,
                )
                energy = updates[-1]["local_energy"] if updates else self.expectation_mpo(mpo)
            else:
                updates = self.sweep(direction)
                energy = updates[-1]["local_energy"] if updates else self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "gauge": _validate_gauge(gauge) if mpo is not None else None,
                "canonicalize": canonicalize_mode if mpo is not None else None,
                "reused_canonical_center": bool(reuse_canonical_center),
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                label = "letta-mpo" if mpo is not None else "letta"
                print(
                    f"{label} sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            previous_sweep_direction = direction
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return self

    def _lbfgs_parameter_layout(self, *, complex_parameters=False):
        layout = []
        offset = 0
        for tensor_index, (tensor, mask) in enumerate(zip(self.tensors, self.local_masks)):
            indices = (
                np.arange(tensor.size, dtype=np.int64)
                if mask is None
                else np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
            )
            width = int(indices.size)
            layout.append((tensor_index, indices, slice(offset, offset + width)))
            offset += width
        real_size = offset
        total_size = 2 * real_size if complex_parameters else real_size
        return layout, real_size, total_size

    def _pack_lbfgs_parameters(self, layout, real_size, *, complex_parameters=False):
        values = np.empty(real_size, dtype=complex if complex_parameters else float)
        for tensor_index, indices, target in layout:
            values[target] = self.tensors[tensor_index].reshape(-1)[indices]
        if complex_parameters:
            return np.concatenate((values.real, values.imag))
        return np.asarray(values.real, dtype=float)

    def _unpack_lbfgs_parameters(
        self,
        vector,
        layout,
        real_size,
        *,
        complex_parameters=False,
    ):
        vector = np.asarray(vector, dtype=float)
        if complex_parameters:
            values = vector[:real_size] + 1j * vector[real_size:]
        else:
            values = vector[:real_size]
        for tensor_index, indices, source in layout:
            tensor = np.zeros(
                self.tensors[tensor_index].size,
                dtype=complex if complex_parameters else float,
            )
            tensor[indices] = values[source]
            self.tensors[tensor_index] = tensor.reshape(self.tensors[tensor_index].shape)

    def _mpo_energy_gradient(self, mpo):
        """Return normalized MPO energy and derivatives for all LETTA tensors."""
        mpo = self._validate_mpo(mpo)
        norm = self._identity_matrix_element()
        norm_real = float(np.real(norm))
        if norm_real <= 1.0e-28:
            raise ValueError("Cannot evaluate the LETTA gradient for a zero-norm state.")
        numerator = self._mpo_matrix_element(mpo)
        energy = float(np.real(numerator / norm))

        left_envs = self._left_local_environments(mpo)
        right_envs = self._right_local_environments(mpo)
        metric_left = self._left_metric_environments()
        metric_right = self._right_metric_environments()
        gradients = []
        for tensor_index, tensor in enumerate(self.tensors):
            vector = tensor.reshape(-1)
            if self.has_terminal_tensor and tensor_index == self.npairs:
                h_vector = self._terminal_effective_from_environment(
                    mpo,
                    left_envs,
                ) @ vector
                n_vector = self._terminal_metric_from_environment(metric_left) @ vector
            else:
                h_vector = self._apply_local_effective_from_environments(
                    mpo,
                    tensor_index,
                    left_envs,
                    right_envs,
                    vector,
                )
                n_vector = self._apply_local_metric_from_environments(
                    tensor_index,
                    metric_left,
                    metric_right,
                    vector,
                )
            gradient = (h_vector - energy * n_vector) / norm_real
            gradients.append(gradient.reshape(tensor.shape))
        return energy, gradients

    def run_lbfgs(
        self,
        mpo,
        *,
        maxiter=40,
        stages=2,
        gtol=1.0e-7,
        ftol=1.0e-12,
        maxcor=10,
        maxls=20,
        gauge="conditional",
        center=None,
        verbose=0,
    ):
        """Optimize all LETTA tensors with gauge-retracted tangent L-BFGS.

        The analytical gradient is the collection of local projected
        residuals.  Parameters are restricted to the current masks, and each
        stage starts from a mixed canonical gauge to remove poorly conditioned
        vertical gauge directions from the quasi-Newton history.
        """
        mpo = self._validate_mpo(mpo)
        gauge = _validate_gauge(gauge)
        maxiter = int(maxiter)
        stages = int(stages)
        if maxiter < 1 or stages < 1:
            raise ValueError("maxiter and stages must be positive.")
        if center is None:
            center = self.nlocal_tensors // 2
        center = int(center)
        if center < 0 or center >= self.nlocal_tensors:
            raise IndexError("center tensor index out of range.")

        complex_parameters = any(np.iscomplexobj(tensor) for tensor in self.tensors)
        complex_parameters = complex_parameters or any(np.iscomplexobj(site) for site in mpo)
        if complex_parameters:
            self.tensors = [np.asarray(tensor, dtype=complex) for tensor in self.tensors]

        self.history = []
        self.converged = False
        self.success = False
        self.message = ""
        iteration = 0
        last_result = None

        for stage in range(stages):
            if gauge == "conditional":
                self.canonicalize_conditional_center(center)
            elif gauge == "virtual":
                self.canonicalize_center(center)
            else:
                self.normalize()

            layout, real_size, _total_size = self._lbfgs_parameter_layout(
                complex_parameters=complex_parameters,
            )
            x0 = self._pack_lbfgs_parameters(
                layout,
                real_size,
                complex_parameters=complex_parameters,
            )
            cache = {}

            def objective(vector):
                self._unpack_lbfgs_parameters(
                    vector,
                    layout,
                    real_size,
                    complex_parameters=complex_parameters,
                )
                energy, tensor_gradients = self._mpo_energy_gradient(mpo)
                active_gradient = np.empty(
                    real_size,
                    dtype=complex if complex_parameters else float,
                )
                for tensor_index, indices, target in layout:
                    active_gradient[target] = tensor_gradients[tensor_index].reshape(-1)[indices]
                if complex_parameters:
                    gradient = 2.0 * np.concatenate(
                        (active_gradient.real, active_gradient.imag)
                    )
                else:
                    gradient = 2.0 * np.asarray(active_gradient.real, dtype=float)
                cache["x"] = np.asarray(vector).copy()
                cache["energy"] = energy
                cache["gradient_norm"] = float(np.linalg.norm(gradient, ord=np.inf))
                return energy, gradient

            def callback(vector):
                nonlocal iteration
                if "x" not in cache or not np.array_equal(vector, cache["x"]):
                    objective(vector)
                entry = {
                    "iteration": iteration,
                    "stage": stage,
                    "optimizer": "lbfgs",
                    "energy": cache["energy"],
                    "gradient_norm": cache["gradient_norm"],
                    "gauge": gauge,
                }
                self.history.append(entry)
                if int(verbose) > 0:
                    print(
                        f"letta-lbfgs iter {iteration:>3} | stage={stage} | "
                        f"E={entry['energy']:.12g} | |g|inf={entry['gradient_norm']:.3e}",
                        flush=True,
                    )
                iteration += 1

            last_result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=True,
                callback=callback,
                options={
                    "maxiter": maxiter,
                    "gtol": float(gtol),
                    "ftol": float(ftol),
                    "maxcor": int(maxcor),
                    "maxls": int(maxls),
                },
            )
            self._unpack_lbfgs_parameters(
                last_result.x,
                layout,
                real_size,
                complex_parameters=complex_parameters,
            )

        if gauge == "conditional":
            self.canonicalize_conditional_center(center)
        elif gauge == "virtual":
            self.canonicalize_center(center)
        else:
            self.normalize()
        self.energy = self.expectation_mpo(mpo)
        self.success = bool(last_result.success)
        self.converged = bool(last_result.success)
        self.message = str(last_result.message)
        if not self.history or abs(self.history[-1]["energy"] - self.energy) > 1.0e-14:
            self.history.append(
                {
                    "iteration": iteration,
                    "stage": stages - 1,
                    "optimizer": "lbfgs",
                    "energy": self.energy,
                    "gradient_norm": float(np.linalg.norm(last_result.jac, ord=np.inf)),
                    "gauge": gauge,
                }
            )
        return self

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

# Backward-compatible alias for the chain-ordered dense workflow.
SequentialLETTA = LETTA
