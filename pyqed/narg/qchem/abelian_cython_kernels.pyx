# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp


def _empty_integral_terms(Py_ssize_t L):
    cdef Py_ssize_t q
    pair_terms = {}
    triple_terms = {}
    for q in range(L):
        pair_terms[q] = {
            'density': [],
            'exchange': [],
            'v2b': [],
        }
        triple_terms[q] = []
    return pair_terms, triple_terms


def _precompute_integral_terms_real(cnp.ndarray[cnp.float64_t, ndim=4] eri, double cutoff):
    cdef Py_ssize_t L = eri.shape[0]
    cdef Py_ssize_t q, i, j, k
    cdef double coeff
    cdef double cutoff_abs = cutoff
    pair_terms, triple_terms = _empty_integral_terms(L)

    for q in range(L):
        for i in range(q):
            for j in range(q):
                coeff = eri[i, j, q, q]
                if abs(coeff) > cutoff_abs:
                    pair_terms[q]['density'].append((i, j, coeff))

                coeff = eri[i, q, q, j]
                if abs(coeff) > cutoff_abs:
                    pair_terms[q]['exchange'].append((i, j, coeff))

                coeff = eri[q, i, q, j]
                if abs(coeff) > cutoff_abs:
                    pair_terms[q]['v2b'].append((i, j, 0.5 * coeff))

                for k in range(q):
                    coeff = eri[k, q, j, i]
                    if abs(coeff) > cutoff_abs:
                        triple_terms[q].append((i, j, k, coeff))
    return pair_terms, triple_terms


def _precompute_integral_terms_complex(cnp.ndarray[cnp.complex128_t, ndim=4] eri, double cutoff):
    cdef Py_ssize_t L = eri.shape[0]
    cdef Py_ssize_t q, i, j, k
    cdef double cutoff2 = cutoff * cutoff
    cdef cnp.complex128_t coeff
    pair_terms, triple_terms = _empty_integral_terms(L)

    for q in range(L):
        for i in range(q):
            for j in range(q):
                coeff = eri[i, j, q, q]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    pair_terms[q]['density'].append((i, j, coeff))

                coeff = eri[i, q, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    pair_terms[q]['exchange'].append((i, j, coeff))

                coeff = eri[q, i, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    pair_terms[q]['v2b'].append((i, j, 0.5 * coeff))

                for k in range(q):
                    coeff = eri[k, q, j, i]
                    if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                        triple_terms[q].append((i, j, k, coeff))
    return pair_terms, triple_terms


def precompute_integral_terms(object eri, double cutoff):
    arr = np.asarray(eri)
    if arr.dtype.kind == 'c':
        return _precompute_integral_terms_complex(
            np.ascontiguousarray(arr, dtype=np.complex128),
            cutoff,
        )
    return _precompute_integral_terms_real(
        np.ascontiguousarray(arr, dtype=np.float64),
        cutoff,
    )


def collect_integral_terms(object eri, double cutoff):
    """Collect nonzero Abelian qchem NARG integral terms into packed arrays."""
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] eri_c = np.ascontiguousarray(
        eri, dtype=np.complex128
    )
    cdef Py_ssize_t L = eri_c.shape[0]
    cdef Py_ssize_t q, i, j, k
    cdef Py_ssize_t ndensity = 0
    cdef Py_ssize_t nexchange = 0
    cdef Py_ssize_t nv2b = 0
    cdef Py_ssize_t ntriple = 0
    cdef Py_ssize_t idensity, iexchange, iv2b, itriple
    cdef double cutoff2 = cutoff * cutoff
    cdef cnp.complex128_t coeff

    for q in range(L):
        for i in range(q):
            for j in range(q):
                coeff = eri_c[i, j, q, q]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    ndensity += 1
                coeff = eri_c[i, q, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    nexchange += 1
                coeff = eri_c[q, i, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    nv2b += 1
                for k in range(q):
                    coeff = eri_c[k, q, j, i]
                    if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                        ntriple += 1

    cdef cnp.ndarray[cnp.int64_t, ndim=2] density_idx = np.empty((ndensity, 3), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] exchange_idx = np.empty((nexchange, 3), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] v2b_idx = np.empty((nv2b, 3), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] triple_idx = np.empty((ntriple, 4), dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] density_val = np.empty(ndensity, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] exchange_val = np.empty(nexchange, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] v2b_val = np.empty(nv2b, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] triple_val = np.empty(ntriple, dtype=np.complex128)

    idensity = 0
    iexchange = 0
    iv2b = 0
    itriple = 0
    for q in range(L):
        for i in range(q):
            for j in range(q):
                coeff = eri_c[i, j, q, q]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    density_idx[idensity, 0] = q
                    density_idx[idensity, 1] = i
                    density_idx[idensity, 2] = j
                    density_val[idensity] = coeff
                    idensity += 1

                coeff = eri_c[i, q, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    exchange_idx[iexchange, 0] = q
                    exchange_idx[iexchange, 1] = i
                    exchange_idx[iexchange, 2] = j
                    exchange_val[iexchange] = coeff
                    iexchange += 1

                coeff = eri_c[q, i, q, j]
                if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                    v2b_idx[iv2b, 0] = q
                    v2b_idx[iv2b, 1] = i
                    v2b_idx[iv2b, 2] = j
                    v2b_val[iv2b] = 0.5 * coeff
                    iv2b += 1

                for k in range(q):
                    coeff = eri_c[k, q, j, i]
                    if coeff.real * coeff.real + coeff.imag * coeff.imag > cutoff2:
                        triple_idx[itriple, 0] = q
                        triple_idx[itriple, 1] = i
                        triple_idx[itriple, 2] = j
                        triple_idx[itriple, 3] = k
                        triple_val[itriple] = coeff
                        itriple += 1

    return (
        density_idx, density_val,
        exchange_idx, exchange_val,
        v2b_idx, v2b_val,
        triple_idx, triple_val,
    )
