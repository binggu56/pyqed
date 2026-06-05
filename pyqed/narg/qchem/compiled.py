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

else:
    _rotate_dense_numba = None


def available() -> bool:
    """Return whether the optimized compiled-numerics backend is usable."""
    return True


def numba_available() -> bool:
    """Return whether Numba dense fallback kernels are available."""
    return _rotate_dense_numba is not None


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
    out = np.zeros((d * D, d * D), dtype=np.result_type(A, B, U, complex))
    for bra_local, ket_local in zip(local_rows, local_cols):
        bra_slice = slice(int(bra_local) * D, (int(bra_local) + 1) * D)
        ket_slice = slice(int(ket_local) * D, (int(ket_local) + 1) * D)
        out[bra_slice, ket_slice] = (
            B[bra_local, ket_local]
            * (U[:, :, bra_local].conj().T @ (A @ U[:, :, ket_local]))
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


__all__ = [
    "available",
    "numba_available",
    "require_available",
    "rotate",
]
