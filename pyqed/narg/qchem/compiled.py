"""Compiled kernels for qchem NARG backends.

The high-level NARG growth still lives in Python because it manages sparse
operator tables, symmetry labels, and eigensolver dispatch.  These kernels cover
the dense projection work that is repeated throughout the Abelian backend.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

try:
    from numba import njit
except Exception:  # pragma: no cover - optional accelerator
    njit = None


if njit is not None:

    @njit(cache=False)
    def _rotate_dense_numba(A, B, U):
        n = U.shape[0]
        D = U.shape[1]
        d = U.shape[2]
        width = d * D

        Umat = np.empty((n, width), dtype=np.complex128)
        for row in range(n):
            for local in range(d):
                offset = local * D
                for state in range(D):
                    Umat[row, offset + state] = U[row, state, local]

        AU = A @ Umat
        rotated = np.conjugate(Umat).T @ AU
        out = np.empty((width, width), dtype=np.complex128)

        for bra_local in range(d):
            bra_offset = bra_local * D
            for ket_local in range(d):
                ket_offset = ket_local * D
                local_coeff = B[bra_local, ket_local]
                for bra_state in range(D):
                    row = bra_offset + bra_state
                    for ket_state in range(D):
                        col = ket_offset + ket_state
                        out[row, col] = local_coeff * rotated[row, col]
        return out

    @njit(cache=False)
    def _add_local_kron_blocks_numba(target4, block, rows, cols, values, coeff):
        old_dim = block.shape[0]
        for nz in range(values.shape[0]):
            row = rows[nz]
            col = cols[nz]
            scale = coeff * values[nz]
            if scale == 0.0:
                continue
            for bra in range(old_dim):
                for ket in range(old_dim):
                    target4[bra, row, ket, col] += scale * block[bra, ket]

    @njit(cache=False)
    def _add_local_kron_blocks_hc_numba(target4, block, rows, cols, values, coeff):
        old_dim = block.shape[0]
        for nz in range(values.shape[0]):
            row = rows[nz]
            col = cols[nz]
            scale = coeff * values[nz]
            if scale == 0.0:
                continue
            scale_h = np.conjugate(scale)
            for bra in range(old_dim):
                for ket in range(old_dim):
                        value = scale * block[bra, ket]
                        target4[bra, row, ket, col] += value
                        target4[ket, col, bra, row] += scale_h * np.conjugate(block[bra, ket])

    @njit(cache=False)
    def _add_sparse_local_kron_blocks_numba(
        target4,
        block_rows,
        block_cols,
        block_values,
        local_rows,
        local_cols,
        local_values,
        coeff,
    ):
        for lnz in range(local_values.shape[0]):
            local_row = local_rows[lnz]
            local_col = local_cols[lnz]
            local_scale = coeff * local_values[lnz]
            if local_scale == 0.0:
                continue
            for bnz in range(block_values.shape[0]):
                target4[
                    block_rows[bnz],
                    local_row,
                    block_cols[bnz],
                    local_col,
                ] += local_scale * block_values[bnz]

    @njit(cache=False)
    def _add_sparse_local_kron_blocks_hc_numba(
        target4,
        block_rows,
        block_cols,
        block_values,
        local_rows,
        local_cols,
        local_values,
        coeff,
    ):
        for lnz in range(local_values.shape[0]):
            local_row = local_rows[lnz]
            local_col = local_cols[lnz]
            local_scale = coeff * local_values[lnz]
            if local_scale == 0.0:
                continue
            local_scale_h = np.conjugate(local_scale)
            for bnz in range(block_values.shape[0]):
                block_row = block_rows[bnz]
                block_col = block_cols[bnz]
                value = local_scale * block_values[bnz]
                target4[block_row, local_row, block_col, local_col] += value
                target4[block_col, local_col, block_row, local_row] += (
                    local_scale_h * np.conjugate(block_values[bnz])
                )

else:
    _rotate_dense_numba = None
    _add_local_kron_blocks_numba = None
    _add_local_kron_blocks_hc_numba = None
    _add_sparse_local_kron_blocks_numba = None
    _add_sparse_local_kron_blocks_hc_numba = None


def available() -> bool:
    """Return whether the optimized compiled-numerics backend is usable."""
    return True


def numba_available() -> bool:
    """Return whether Numba dense fallback kernels are available."""
    return _rotate_dense_numba is not None


def local_kron_available() -> bool:
    """Return whether compiled local-Kronecker accumulation is available."""
    return _add_local_kron_blocks_numba is not None


def require_available() -> None:
    """Raise a clear error when the compiled backend cannot be used."""
    if not available():
        raise ImportError("compiled qchem NARG backend is not available.")


def _rotate_dense_numpy(A, B, U):
    n, D, d = U.shape
    Umat = U.transpose(0, 2, 1).reshape(n, d * D)
    rotated = np.asarray(Umat.conj().T @ np.asarray(A @ Umat))
    return (
        rotated.reshape(d, D, d, D)
        * np.asarray(B)[:, None, :, None]
    ).reshape(d * D, d * D)


def _rotate_sparse_local_blas(A, B, U, local_rows, local_cols):
    D = U.shape[1]
    d = U.shape[2]
    dtype = np.result_type(getattr(A, "dtype", np.asarray(A).dtype), B, U, complex)
    out = np.zeros((d * D, d * D), dtype=dtype)
    applied = {}
    for bra_local, ket_local in zip(local_rows, local_cols):
        bra_local = int(bra_local)
        ket_local = int(ket_local)
        bra_slice = slice(bra_local * D, (bra_local + 1) * D)
        ket_slice = slice(ket_local * D, (ket_local + 1) * D)
        right = applied.get(ket_local)
        if right is None:
            right = A @ U[:, :, ket_local]
            applied[ket_local] = right
        out[bra_slice, ket_slice] = (
            B[bra_local, ket_local]
            * (U[:, :, bra_local].conj().T @ right)
        )
    return out


def rotate(A, B, U):
    """Project ``A tensor B`` through a NARG basis using compiled numerical kernels."""
    B = np.asarray(B)
    U = np.asarray(U)
    local_rows, local_cols = np.nonzero(B)
    if local_rows.size == 0:
        width = U.shape[1] * U.shape[2]
        return np.zeros((width, width), dtype=np.result_type(A, B, U, complex))
    if local_rows.size < B.size:
        return _rotate_sparse_local_blas(A, B, U, local_rows, local_cols)
    if issparse(A):
        A = A.toarray()
    if _rotate_dense_numba is None:
        return _rotate_dense_numpy(A, B, U)
    A = np.ascontiguousarray(A, dtype=np.complex128)
    B = np.ascontiguousarray(B, dtype=np.complex128)
    U = np.ascontiguousarray(U, dtype=np.complex128)
    return _rotate_dense_numba(A, B, U)


def add_local_kron_blocks(target4, block, local, coeff=1.0):
    """Add ``coeff * kron(block, local)`` into a 4D Kronecker view in place."""
    if _add_local_kron_blocks_numba is None:
        return False
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return True
    values = np.asarray(local)[rows, cols]
    block = np.ascontiguousarray(block, dtype=np.complex128)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    _add_local_kron_blocks_numba(target4, block, rows, cols, values, np.complex128(coeff))
    return True


def add_local_kron_blocks_hc(target4, block, local, coeff=1.0):
    """Add ``X + X^dagger`` for ``X = coeff * kron(block, local)`` in place."""
    if _add_local_kron_blocks_hc_numba is None:
        return False
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return True
    values = np.asarray(local)[rows, cols]
    block = np.ascontiguousarray(block, dtype=np.complex128)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    _add_local_kron_blocks_hc_numba(target4, block, rows, cols, values, np.complex128(coeff))
    return True


def add_sparse_local_kron_blocks(target4, block, local, coeff=1.0):
    """Add ``coeff * kron(block, local)`` for a sparse block operator."""
    if _add_sparse_local_kron_blocks_numba is None:
        return False
    block = block.tocoo()
    if block.nnz == 0:
        return True
    local_rows, local_cols = np.nonzero(local)
    if local_rows.size == 0:
        return True
    local_values = np.asarray(local)[local_rows, local_cols]
    _add_sparse_local_kron_blocks_numba(
        target4,
        np.ascontiguousarray(block.row, dtype=np.int64),
        np.ascontiguousarray(block.col, dtype=np.int64),
        np.ascontiguousarray(block.data, dtype=np.complex128),
        np.ascontiguousarray(local_rows, dtype=np.int64),
        np.ascontiguousarray(local_cols, dtype=np.int64),
        np.ascontiguousarray(local_values, dtype=np.complex128),
        np.complex128(coeff),
    )
    return True


def add_sparse_local_kron_blocks_hc(target4, block, local, coeff=1.0):
    """Add ``X + X^dagger`` for ``X = coeff * kron(block, local)`` with sparse ``block``."""
    if _add_sparse_local_kron_blocks_hc_numba is None:
        return False
    block = block.tocoo()
    if block.nnz == 0:
        return True
    local_rows, local_cols = np.nonzero(local)
    if local_rows.size == 0:
        return True
    local_values = np.asarray(local)[local_rows, local_cols]
    _add_sparse_local_kron_blocks_hc_numba(
        target4,
        np.ascontiguousarray(block.row, dtype=np.int64),
        np.ascontiguousarray(block.col, dtype=np.int64),
        np.ascontiguousarray(block.data, dtype=np.complex128),
        np.ascontiguousarray(local_rows, dtype=np.int64),
        np.ascontiguousarray(local_cols, dtype=np.int64),
        np.ascontiguousarray(local_values, dtype=np.complex128),
        np.complex128(coeff),
    )
    return True


__all__ = [
    "add_local_kron_blocks",
    "add_local_kron_blocks_hc",
    "add_sparse_local_kron_blocks",
    "add_sparse_local_kron_blocks_hc",
    "available",
    "local_kron_available",
    "numba_available",
    "require_available",
    "rotate",
]
