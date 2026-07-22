# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, sqrt


cdef inline double _sqrt_ratio(double numer, double denom) noexcept nogil:
    if numer <= 0.0:
        return 0.0
    return sqrt(numer / denom)


cdef inline double _phase_from_doubled(int exponent2) noexcept nogil:
    cdef int exponent = exponent2 // 2
    return -1.0 if exponent % 2 else 1.0


cdef inline double _cg_right_spin_half(
    int left_j2,
    int left_m2,
    int right_m2,
    int coupled_j2,
) noexcept nogil:
    cdef double j2 = <double>left_j2
    cdef double m2 = <double>left_m2
    if right_m2 == 1:
        if coupled_j2 == left_j2 + 1:
            return _sqrt_ratio(j2 + m2 + 2.0, 2.0 * (j2 + 1.0))
        if coupled_j2 == left_j2 - 1:
            return -_sqrt_ratio(j2 - m2, 2.0 * (j2 + 1.0))
    elif right_m2 == -1:
        if coupled_j2 == left_j2 + 1:
            return _sqrt_ratio(j2 - m2 + 2.0, 2.0 * (j2 + 1.0))
        if coupled_j2 == left_j2 - 1:
            return _sqrt_ratio(j2 + m2, 2.0 * (j2 + 1.0))
    return 0.0


cdef inline double _cg_right_spin_one(
    int left_j2,
    int left_m2,
    int right_m2,
    int coupled_j2,
) noexcept nogil:
    cdef double j2 = <double>left_j2
    cdef double m2 = <double>left_m2
    cdef int q2 = right_m2
    cdef double denom

    if coupled_j2 == left_j2 + 2:
        denom = 4.0 * (j2 + 1.0) * (j2 + 2.0)
        if q2 == 2:
            return _sqrt_ratio((j2 + m2 + 2.0) * (j2 + m2 + 4.0), denom)
        if q2 == 0:
            return _sqrt_ratio((j2 - m2 + 2.0) * (j2 + m2 + 2.0), 0.5 * denom)
        if q2 == -2:
            return _sqrt_ratio((j2 - m2 + 2.0) * (j2 - m2 + 4.0), denom)

    if coupled_j2 == left_j2 and left_j2 > 0:
        denom = 2.0 * j2 * (j2 + 2.0)
        if q2 == 2:
            return -_sqrt_ratio((j2 - m2) * (j2 + m2 + 2.0), denom)
        if q2 == 0:
            return m2 / sqrt(j2 * (j2 + 2.0))
        if q2 == -2:
            return _sqrt_ratio((j2 + m2) * (j2 - m2 + 2.0), denom)

    if coupled_j2 == left_j2 - 2 and left_j2 >= 2:
        denom = 4.0 * j2 * (j2 + 1.0)
        if q2 == 2:
            return _sqrt_ratio((j2 - m2) * (j2 - m2 - 2.0), denom)
        if q2 == 0:
            return -_sqrt_ratio((j2 - m2) * (j2 + m2), 0.5 * denom)
        if q2 == -2:
            return _sqrt_ratio((j2 + m2) * (j2 + m2 - 2.0), denom)

    return 0.0


cdef double _cg(
    int left_j2,
    int left_m2,
    int right_j2,
    int right_m2,
    int coupled_j2,
    int coupled_m2,
) noexcept nogil:
    if left_m2 + right_m2 != coupled_m2:
        return 0.0
    if abs(left_m2) > left_j2 or abs(right_m2) > right_j2 or abs(coupled_m2) > coupled_j2:
        return 0.0
    if (left_j2 - left_m2) % 2 or (right_j2 - right_m2) % 2 or (coupled_j2 - coupled_m2) % 2:
        return 0.0
    if coupled_j2 < abs(left_j2 - right_j2) or coupled_j2 > left_j2 + right_j2:
        return 0.0
    if (left_j2 + right_j2 + coupled_j2) % 2:
        return 0.0

    if right_j2 == 0:
        if coupled_j2 == left_j2 and coupled_m2 == left_m2 and right_m2 == 0:
            return 1.0
        return 0.0
    if right_j2 == 1:
        return _cg_right_spin_half(left_j2, left_m2, right_m2, coupled_j2)
    if right_j2 == 2:
        return _cg_right_spin_one(left_j2, left_m2, right_m2, coupled_j2)

    if left_j2 == 0 or left_j2 == 1 or left_j2 == 2:
        return (
            _phase_from_doubled(left_j2 + right_j2 - coupled_j2)
            * _cg(right_j2, right_m2, left_j2, left_m2, coupled_j2, coupled_m2)
        )

    return 0.0


cdef double _product_tensor_pair_coeff(
    int bra_total_j2,
    int ket_total_j2,
    int total_rank2,
    int total_q2,
    int ket_total_m2,
    int bra_block_j2,
    int bra_local_j2,
    int ket_block_j2,
    int ket_local_j2,
    int block_rank2,
    int local_rank2,
    double atol,
) noexcept nogil:
    cdef int bra_total_m2 = ket_total_m2 + total_q2
    cdef int ket_block_m2
    cdef int ket_local_m2
    cdef int q_block2
    cdef int q_local2
    cdef int bra_block_m2
    cdef int bra_local_m2
    cdef double value = 0.0
    cdef double ket_cg
    cdef double tensor_cg
    cdef double bra_cg
    cdef double block_cg
    cdef double local_cg

    if bra_total_m2 < -bra_total_j2 or bra_total_m2 > bra_total_j2:
        return 0.0

    for ket_block_m2 in range(-ket_block_j2, ket_block_j2 + 1, 2):
        ket_local_m2 = ket_total_m2 - ket_block_m2
        if ket_local_m2 < -ket_local_j2 or ket_local_m2 > ket_local_j2:
            continue
        ket_cg = _cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            ket_total_j2,
            ket_total_m2,
        )
        if fabs(ket_cg) <= atol:
            continue

        for q_block2 in range(-block_rank2, block_rank2 + 1, 2):
            q_local2 = total_q2 - q_block2
            if q_local2 < -local_rank2 or q_local2 > local_rank2:
                continue
            tensor_cg = _cg(block_rank2, q_block2, local_rank2, q_local2, total_rank2, total_q2)
            if fabs(tensor_cg) <= atol:
                continue

            bra_block_m2 = ket_block_m2 + q_block2
            bra_local_m2 = ket_local_m2 + q_local2
            if bra_block_m2 < -bra_block_j2 or bra_block_m2 > bra_block_j2:
                continue
            if bra_local_m2 < -bra_local_j2 or bra_local_m2 > bra_local_j2:
                continue

            bra_cg = _cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                bra_total_j2,
                bra_total_m2,
            )
            block_cg = _cg(
                ket_block_j2,
                ket_block_m2,
                block_rank2,
                q_block2,
                bra_block_j2,
                bra_block_m2,
            )
            local_cg = _cg(
                ket_local_j2,
                ket_local_m2,
                local_rank2,
                q_local2,
                bra_local_j2,
                bra_local_m2,
            )
            if fabs(bra_cg) <= atol or fabs(block_cg) <= atol or fabs(local_cg) <= atol:
                continue

            value += (
                bra_cg
                * ket_cg
                * tensor_cg
                * block_cg
                * local_cg
                / sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            )
    return value


cdef double _scalar_product_pair_coeff(
    int total_j2,
    int bra_block_j2,
    int bra_local_j2,
    int ket_block_j2,
    int ket_local_j2,
    int block_rank2,
    int local_rank2,
    double atol,
) noexcept nogil:
    cdef int ket_total_m2 = total_j2
    cdef int bra_total_m2 = total_j2
    cdef int ket_block_m2
    cdef int ket_local_m2
    cdef int q_block2
    cdef int q_local2
    cdef int bra_block_m2
    cdef int bra_local_m2
    cdef double value = 0.0
    cdef double ket_cg
    cdef double tensor_cg
    cdef double bra_cg
    cdef double block_cg
    cdef double local_cg

    for ket_block_m2 in range(-ket_block_j2, ket_block_j2 + 1, 2):
        ket_local_m2 = ket_total_m2 - ket_block_m2
        if ket_local_m2 < -ket_local_j2 or ket_local_m2 > ket_local_j2:
            continue
        ket_cg = _cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            total_j2,
            ket_total_m2,
        )
        if fabs(ket_cg) <= atol:
            continue

        for q_block2 in range(-block_rank2, block_rank2 + 1, 2):
            q_local2 = -q_block2
            if q_local2 < -local_rank2 or q_local2 > local_rank2:
                continue
            tensor_cg = _cg(block_rank2, q_block2, local_rank2, q_local2, 0, 0)
            if fabs(tensor_cg) <= atol:
                continue

            bra_block_m2 = ket_block_m2 + q_block2
            bra_local_m2 = ket_local_m2 + q_local2
            if bra_block_m2 < -bra_block_j2 or bra_block_m2 > bra_block_j2:
                continue
            if bra_local_m2 < -bra_local_j2 or bra_local_m2 > bra_local_j2:
                continue

            bra_cg = _cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                total_j2,
                bra_total_m2,
            )
            block_cg = _cg(
                ket_block_j2,
                ket_block_m2,
                block_rank2,
                q_block2,
                bra_block_j2,
                bra_block_m2,
            )
            local_cg = _cg(
                ket_local_j2,
                ket_local_m2,
                local_rank2,
                q_local2,
                bra_local_j2,
                bra_local_m2,
            )
            if fabs(bra_cg) <= atol or fabs(block_cg) <= atol or fabs(local_cg) <= atol:
                continue

            value += (
                bra_cg
                * ket_cg
                * tensor_cg
                * block_cg
                * local_cg
                / sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            )
    return value


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


def product_tensor_estimate_entries(
    cnp.ndarray[cnp.int64_t, ndim=2] bra_states,
    cnp.ndarray[cnp.int64_t, ndim=2] ket_states,
    int bra_total_j2,
    int ket_total_j2,
    int total_rank2,
    int total_q2,
    int ket_total_m2,
    int block_dnelec,
    int block_rank2,
    int local_dnelec,
    int local_rank2,
    double atol,
):
    """Build one product-tensor angular estimate as numeric index arrays."""
    cdef Py_ssize_t bra_count = bra_states.shape[0]
    cdef Py_ssize_t ket_count = ket_states.shape[0]
    cdef Py_ssize_t bra_pos
    cdef Py_ssize_t ket_pos
    cdef Py_ssize_t out_pos
    cdef Py_ssize_t count = 0
    cdef int bra_block_nelec
    cdef int bra_block_j2
    cdef int bra_local_nelec
    cdef int bra_local_j2
    cdef int ket_block_nelec
    cdef int ket_block_j2
    cdef int ket_local_nelec
    cdef int ket_local_j2
    cdef double coeff

    for bra_pos in range(bra_count):
        bra_block_nelec = <int>bra_states[bra_pos, 0]
        bra_block_j2 = <int>bra_states[bra_pos, 1]
        bra_local_nelec = <int>bra_states[bra_pos, 3]
        bra_local_j2 = <int>bra_states[bra_pos, 4]
        for ket_pos in range(ket_count):
            ket_block_nelec = <int>ket_states[ket_pos, 0]
            ket_block_j2 = <int>ket_states[ket_pos, 1]
            ket_local_nelec = <int>ket_states[ket_pos, 3]
            ket_local_j2 = <int>ket_states[ket_pos, 4]
            if bra_block_nelec != ket_block_nelec + block_dnelec:
                continue
            if bra_local_nelec != ket_local_nelec + local_dnelec:
                continue
            coeff = _product_tensor_pair_coeff(
                bra_total_j2,
                ket_total_j2,
                total_rank2,
                total_q2,
                ket_total_m2,
                bra_block_j2,
                bra_local_j2,
                ket_block_j2,
                ket_local_j2,
                block_rank2,
                local_rank2,
                atol,
            )
            if fabs(coeff) <= atol:
                continue
            count += 1

    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    block_rows = np.empty(count, dtype=np.int64)
    block_cols = np.empty(count, dtype=np.int64)
    local_rows = np.empty(count, dtype=np.int64)
    local_cols = np.empty(count, dtype=np.int64)
    coeffs = np.empty(count, dtype=np.float64)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows_view = rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols_view = cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_rows_view = block_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_cols_view = block_cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_rows_view = local_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_cols_view = local_cols
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeffs_view = coeffs

    out_pos = 0
    for bra_pos in range(bra_count):
        bra_block_nelec = <int>bra_states[bra_pos, 0]
        bra_block_j2 = <int>bra_states[bra_pos, 1]
        bra_local_nelec = <int>bra_states[bra_pos, 3]
        bra_local_j2 = <int>bra_states[bra_pos, 4]
        for ket_pos in range(ket_count):
            ket_block_nelec = <int>ket_states[ket_pos, 0]
            ket_block_j2 = <int>ket_states[ket_pos, 1]
            ket_local_nelec = <int>ket_states[ket_pos, 3]
            ket_local_j2 = <int>ket_states[ket_pos, 4]
            if bra_block_nelec != ket_block_nelec + block_dnelec:
                continue
            if bra_local_nelec != ket_local_nelec + local_dnelec:
                continue
            coeff = _product_tensor_pair_coeff(
                bra_total_j2,
                ket_total_j2,
                total_rank2,
                total_q2,
                ket_total_m2,
                bra_block_j2,
                bra_local_j2,
                ket_block_j2,
                ket_local_j2,
                block_rank2,
                local_rank2,
                atol,
            )
            if fabs(coeff) <= atol:
                continue
            rows_view[out_pos] = bra_pos
            cols_view[out_pos] = ket_pos
            block_rows_view[out_pos] = bra_states[bra_pos, 2]
            block_cols_view[out_pos] = ket_states[ket_pos, 2]
            local_rows_view[out_pos] = bra_states[bra_pos, 5]
            local_cols_view[out_pos] = ket_states[ket_pos, 5]
            coeffs_view[out_pos] = coeff
            out_pos += 1

    return rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols


def scalar_product_pair_entries(
    cnp.ndarray[cnp.int64_t, ndim=2] states,
    int total_j2,
    int block_dnelec,
    int block_rank2,
    int local_dnelec,
    int local_rank2,
    double atol,
):
    """Build scalar-product angular entries for one total irrep sector."""
    cdef Py_ssize_t state_count = states.shape[0]
    cdef Py_ssize_t bra_pos
    cdef Py_ssize_t ket_pos
    cdef Py_ssize_t out_pos
    cdef Py_ssize_t count = 0
    cdef int bra_block_nelec
    cdef int bra_block_j2
    cdef int bra_local_nelec
    cdef int bra_local_j2
    cdef int ket_block_nelec
    cdef int ket_block_j2
    cdef int ket_local_nelec
    cdef int ket_local_j2
    cdef double coeff
    cdef bint rank0_case = block_rank2 == 0 and local_rank2 == 0

    for bra_pos in range(state_count):
        bra_block_nelec = <int>states[bra_pos, 0]
        bra_block_j2 = <int>states[bra_pos, 1]
        bra_local_nelec = <int>states[bra_pos, 3]
        bra_local_j2 = <int>states[bra_pos, 4]
        for ket_pos in range(state_count):
            ket_block_nelec = <int>states[ket_pos, 0]
            ket_block_j2 = <int>states[ket_pos, 1]
            ket_local_nelec = <int>states[ket_pos, 3]
            ket_local_j2 = <int>states[ket_pos, 4]
            if bra_block_nelec != ket_block_nelec + block_dnelec:
                continue
            if bra_local_nelec != ket_local_nelec + local_dnelec:
                continue
            if rank0_case:
                if bra_block_j2 != ket_block_j2 or bra_local_j2 != ket_local_j2:
                    continue
                coeff = 1.0 / sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            else:
                coeff = _scalar_product_pair_coeff(
                    total_j2,
                    bra_block_j2,
                    bra_local_j2,
                    ket_block_j2,
                    ket_local_j2,
                    block_rank2,
                    local_rank2,
                    atol,
                )
                if fabs(coeff) <= atol:
                    continue
            count += 1

    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    block_rows = np.empty(count, dtype=np.int64)
    block_cols = np.empty(count, dtype=np.int64)
    local_rows = np.empty(count, dtype=np.int64)
    local_cols = np.empty(count, dtype=np.int64)
    coeffs = np.empty(count, dtype=np.float64)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows_view = rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols_view = cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_rows_view = block_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_cols_view = block_cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_rows_view = local_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_cols_view = local_cols
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeffs_view = coeffs

    out_pos = 0
    for bra_pos in range(state_count):
        bra_block_nelec = <int>states[bra_pos, 0]
        bra_block_j2 = <int>states[bra_pos, 1]
        bra_local_nelec = <int>states[bra_pos, 3]
        bra_local_j2 = <int>states[bra_pos, 4]
        for ket_pos in range(state_count):
            ket_block_nelec = <int>states[ket_pos, 0]
            ket_block_j2 = <int>states[ket_pos, 1]
            ket_local_nelec = <int>states[ket_pos, 3]
            ket_local_j2 = <int>states[ket_pos, 4]
            if bra_block_nelec != ket_block_nelec + block_dnelec:
                continue
            if bra_local_nelec != ket_local_nelec + local_dnelec:
                continue
            if rank0_case:
                if bra_block_j2 != ket_block_j2 or bra_local_j2 != ket_local_j2:
                    continue
                coeff = 1.0 / sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            else:
                coeff = _scalar_product_pair_coeff(
                    total_j2,
                    bra_block_j2,
                    bra_local_j2,
                    ket_block_j2,
                    ket_local_j2,
                    block_rank2,
                    local_rank2,
                    atol,
                )
                if fabs(coeff) <= atol:
                    continue
            rows_view[out_pos] = bra_pos
            cols_view[out_pos] = ket_pos
            block_rows_view[out_pos] = states[bra_pos, 2]
            block_cols_view[out_pos] = states[ket_pos, 2]
            local_rows_view[out_pos] = states[bra_pos, 5]
            local_cols_view[out_pos] = states[ket_pos, 5]
            coeffs_view[out_pos] = coeff
            out_pos += 1

    return rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols


def product_tensor_pair_entries(
    cnp.ndarray[cnp.int64_t, ndim=2] bra_states,
    cnp.ndarray[cnp.int64_t, ndim=2] ket_states,
    int bra_total_j2,
    int ket_total_j2,
    int total_rank2,
    int block_dnelec,
    int block_rank2,
    int local_dnelec,
    int local_rank2,
    double atol,
):
    """Build all averaged product-tensor angular entries for one irrep pair."""
    cdef Py_ssize_t bra_count = bra_states.shape[0]
    cdef Py_ssize_t ket_count = ket_states.shape[0]
    cdef Py_ssize_t bra_pos
    cdef Py_ssize_t ket_pos
    cdef Py_ssize_t out_pos
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t estimate_count = 0
    cdef Py_ssize_t local_count
    cdef int total_q2
    cdef int ket_total_m2
    cdef int bra_total_m2
    cdef int bra_block_nelec
    cdef int bra_block_j2
    cdef int bra_local_nelec
    cdef int bra_local_j2
    cdef int ket_block_nelec
    cdef int ket_block_j2
    cdef int ket_local_nelec
    cdef int ket_local_j2
    cdef double out_coeff
    cdef double coeff
    cdef double scale

    for total_q2 in range(-total_rank2, total_rank2 + 1, 2):
        for ket_total_m2 in range(-ket_total_j2, ket_total_j2 + 1, 2):
            bra_total_m2 = ket_total_m2 + total_q2
            if bra_total_m2 < -bra_total_j2 or bra_total_m2 > bra_total_j2:
                continue
            out_coeff = _cg(
                ket_total_j2,
                ket_total_m2,
                total_rank2,
                total_q2,
                bra_total_j2,
                bra_total_m2,
            )
            if fabs(out_coeff) <= atol:
                continue

            local_count = 0
            for bra_pos in range(bra_count):
                bra_block_nelec = <int>bra_states[bra_pos, 0]
                bra_block_j2 = <int>bra_states[bra_pos, 1]
                bra_local_nelec = <int>bra_states[bra_pos, 3]
                bra_local_j2 = <int>bra_states[bra_pos, 4]
                for ket_pos in range(ket_count):
                    ket_block_nelec = <int>ket_states[ket_pos, 0]
                    ket_block_j2 = <int>ket_states[ket_pos, 1]
                    ket_local_nelec = <int>ket_states[ket_pos, 3]
                    ket_local_j2 = <int>ket_states[ket_pos, 4]
                    if bra_block_nelec != ket_block_nelec + block_dnelec:
                        continue
                    if bra_local_nelec != ket_local_nelec + local_dnelec:
                        continue
                    coeff = _product_tensor_pair_coeff(
                        bra_total_j2,
                        ket_total_j2,
                        total_rank2,
                        total_q2,
                        ket_total_m2,
                        bra_block_j2,
                        bra_local_j2,
                        ket_block_j2,
                        ket_local_j2,
                        block_rank2,
                        local_rank2,
                        atol,
                    )
                    if fabs(coeff) <= atol:
                        continue
                    local_count += 1
            if local_count:
                estimate_count += 1
                count += local_count

    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    block_rows = np.empty(count, dtype=np.int64)
    block_cols = np.empty(count, dtype=np.int64)
    local_rows = np.empty(count, dtype=np.int64)
    local_cols = np.empty(count, dtype=np.int64)
    coeffs = np.empty(count, dtype=np.float64)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows_view = rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols_view = cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_rows_view = block_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] block_cols_view = block_cols
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_rows_view = local_rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_cols_view = local_cols
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeffs_view = coeffs

    if count == 0 or estimate_count == 0:
        return rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols

    out_pos = 0
    for total_q2 in range(-total_rank2, total_rank2 + 1, 2):
        for ket_total_m2 in range(-ket_total_j2, ket_total_j2 + 1, 2):
            bra_total_m2 = ket_total_m2 + total_q2
            if bra_total_m2 < -bra_total_j2 or bra_total_m2 > bra_total_j2:
                continue
            out_coeff = _cg(
                ket_total_j2,
                ket_total_m2,
                total_rank2,
                total_q2,
                bra_total_j2,
                bra_total_m2,
            )
            if fabs(out_coeff) <= atol:
                continue

            local_count = 0
            for bra_pos in range(bra_count):
                bra_block_nelec = <int>bra_states[bra_pos, 0]
                bra_block_j2 = <int>bra_states[bra_pos, 1]
                bra_local_nelec = <int>bra_states[bra_pos, 3]
                bra_local_j2 = <int>bra_states[bra_pos, 4]
                for ket_pos in range(ket_count):
                    ket_block_nelec = <int>ket_states[ket_pos, 0]
                    ket_block_j2 = <int>ket_states[ket_pos, 1]
                    ket_local_nelec = <int>ket_states[ket_pos, 3]
                    ket_local_j2 = <int>ket_states[ket_pos, 4]
                    if bra_block_nelec != ket_block_nelec + block_dnelec:
                        continue
                    if bra_local_nelec != ket_local_nelec + local_dnelec:
                        continue
                    coeff = _product_tensor_pair_coeff(
                        bra_total_j2,
                        ket_total_j2,
                        total_rank2,
                        total_q2,
                        ket_total_m2,
                        bra_block_j2,
                        bra_local_j2,
                        ket_block_j2,
                        ket_local_j2,
                        block_rank2,
                        local_rank2,
                        atol,
                    )
                    if fabs(coeff) <= atol:
                        continue
                    local_count += 1
            if local_count == 0:
                continue

            scale = sqrt(bra_total_j2 + 1.0) / (out_coeff * estimate_count)
            for bra_pos in range(bra_count):
                bra_block_nelec = <int>bra_states[bra_pos, 0]
                bra_block_j2 = <int>bra_states[bra_pos, 1]
                bra_local_nelec = <int>bra_states[bra_pos, 3]
                bra_local_j2 = <int>bra_states[bra_pos, 4]
                for ket_pos in range(ket_count):
                    ket_block_nelec = <int>ket_states[ket_pos, 0]
                    ket_block_j2 = <int>ket_states[ket_pos, 1]
                    ket_local_nelec = <int>ket_states[ket_pos, 3]
                    ket_local_j2 = <int>ket_states[ket_pos, 4]
                    if bra_block_nelec != ket_block_nelec + block_dnelec:
                        continue
                    if bra_local_nelec != ket_local_nelec + local_dnelec:
                        continue
                    coeff = _product_tensor_pair_coeff(
                        bra_total_j2,
                        ket_total_j2,
                        total_rank2,
                        total_q2,
                        ket_total_m2,
                        bra_block_j2,
                        bra_local_j2,
                        ket_block_j2,
                        ket_local_j2,
                        block_rank2,
                        local_rank2,
                        atol,
                    )
                    if fabs(coeff) <= atol:
                        continue
                    rows_view[out_pos] = bra_pos
                    cols_view[out_pos] = ket_pos
                    block_rows_view[out_pos] = bra_states[bra_pos, 2]
                    block_cols_view[out_pos] = ket_states[ket_pos, 2]
                    local_rows_view[out_pos] = bra_states[bra_pos, 5]
                    local_cols_view[out_pos] = ket_states[ket_pos, 5]
                    coeffs_view[out_pos] = coeff * scale
                    out_pos += 1

    return rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols


def product_tensor_group_indices(
    cnp.ndarray[cnp.int64_t, ndim=2] bra_states,
    cnp.ndarray[cnp.int64_t, ndim=2] ket_states,
    cnp.ndarray[cnp.int64_t, ndim=1] rows,
    cnp.ndarray[cnp.int64_t, ndim=1] cols,
):
    """Group compiled product entries by block/local irrep integer keys."""
    cdef Py_ssize_t size = rows.shape[0]
    cdef Py_ssize_t entry
    cdef Py_ssize_t group
    cdef Py_ssize_t field
    cdef Py_ssize_t ngroups = 0
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef bint found
    cdef Py_ssize_t pos
    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_starts_view
    cdef cnp.ndarray[cnp.int64_t, ndim=1] order_view

    group_keys = np.empty((size, 8), dtype=np.int64)
    group_counts = np.zeros(size, dtype=np.int64)
    group_ids = np.empty(size, dtype=np.int64)

    cdef cnp.ndarray[cnp.int64_t, ndim=2] group_keys_view = group_keys
    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_counts_view = group_counts
    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_ids_view = group_ids
    cdef long key0
    cdef long key1
    cdef long key2
    cdef long key3
    cdef long key4
    cdef long key5
    cdef long key6
    cdef long key7

    for entry in range(size):
        row = rows[entry]
        col = cols[entry]
        key0 = bra_states[row, 0]
        key1 = bra_states[row, 1]
        key2 = ket_states[col, 0]
        key3 = ket_states[col, 1]
        key4 = bra_states[row, 3]
        key5 = bra_states[row, 4]
        key6 = ket_states[col, 3]
        key7 = ket_states[col, 4]

        found = False
        for group in range(ngroups):
            if (
                group_keys_view[group, 0] == key0
                and group_keys_view[group, 1] == key1
                and group_keys_view[group, 2] == key2
                and group_keys_view[group, 3] == key3
                and group_keys_view[group, 4] == key4
                and group_keys_view[group, 5] == key5
                and group_keys_view[group, 6] == key6
                and group_keys_view[group, 7] == key7
            ):
                found = True
                break
        if not found:
            group = ngroups
            ngroups += 1
            group_keys_view[group, 0] = key0
            group_keys_view[group, 1] = key1
            group_keys_view[group, 2] = key2
            group_keys_view[group, 3] = key3
            group_keys_view[group, 4] = key4
            group_keys_view[group, 5] = key5
            group_keys_view[group, 6] = key6
            group_keys_view[group, 7] = key7
        group_ids_view[entry] = group
        group_counts_view[group] += 1

    group_starts = np.empty(ngroups + 1, dtype=np.int64)
    order = np.empty(size, dtype=np.int64)
    group_starts_view = group_starts
    order_view = order

    group_starts_view[0] = 0
    for group in range(ngroups):
        group_starts_view[group + 1] = group_starts_view[group] + group_counts_view[group]
        group_counts_view[group] = group_starts_view[group]

    for entry in range(size):
        group = group_ids_view[entry]
        pos = group_counts_view[group]
        order_view[pos] = entry
        group_counts_view[group] = pos + 1

    return group_keys[:ngroups].copy(), group_starts, order
