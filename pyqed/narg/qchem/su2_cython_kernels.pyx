# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp


def accumulate_bilinear(
    cnp.ndarray[cnp.complex128_t, ndim=2] out,
    cnp.ndarray[cnp.int64_t, ndim=1] rows,
    cnp.ndarray[cnp.int64_t, ndim=1] cols,
    cnp.ndarray[cnp.int64_t, ndim=1] block_rows,
    cnp.ndarray[cnp.int64_t, ndim=1] block_cols,
    cnp.ndarray[cnp.int64_t, ndim=1] local_rows,
    cnp.ndarray[cnp.int64_t, ndim=1] local_cols,
    cnp.ndarray[cnp.complex128_t, ndim=1] coeffs,
    cnp.ndarray[cnp.complex128_t, ndim=2] block,
    cnp.ndarray[cnp.complex128_t, ndim=2] local,
    cnp.complex128_t prefactor,
):
    """Accumulate coeff * block[...] * local[...] into out."""
    cdef Py_ssize_t n
    cdef Py_ssize_t size = rows.shape[0]
    for n in range(size):
        out[rows[n], cols[n]] += (
            prefactor
            * coeffs[n]
            * block[block_rows[n], block_cols[n]]
            * local[local_rows[n], local_cols[n]]
        )
    return out
