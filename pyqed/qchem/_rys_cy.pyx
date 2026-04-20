# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t f64_t
ctypedef cnp.int64_t i64_t

DEF RYS_FIXED_MAX_FACTORS = 4
DEF RYS_FIXED_MAX_TERMS = 128


cdef int MAX_FACTORS = 8
cdef int MAX_TERMS = 4096


cdef inline double _vec_component(int name_id, int idx, double[:] AB, double[:] CD, double[:] PQ) noexcept nogil:
    if name_id == 0:
        return AB[idx]
    if name_id == 1:
        return CD[idx]
    return PQ[idx]


cdef inline double _vec_diff_coeff(int center_id, int name_id, double lam_a, double lam_b, double lam_c, double lam_d) noexcept nogil:
    if center_id == 0:  # A
        if name_id == 0:
            return 1.0
        if name_id == 2:
            return lam_a
        return 0.0
    if center_id == 1:  # B
        if name_id == 0:
            return -1.0
        if name_id == 2:
            return lam_b
        return 0.0
    if center_id == 2:  # C
        if name_id == 1:
            return 1.0
        if name_id == 2:
            return -lam_c
        return 0.0
    # D
    if name_id == 1:
        return -1.0
    if name_id == 2:
        return -lam_d
    return 0.0


cdef inline double _pref_mu(int center_id, double mu_ab, double mu_cd) noexcept nogil:
    if center_id == 0 or center_id == 1:
        return mu_ab
    return mu_cd


cdef inline int _pref_vec_name(int center_id) noexcept nogil:
    if center_id == 0 or center_id == 1:
        return 0  # AB
    return 1  # CD


cdef inline double _pref_sign(int center_id) noexcept nogil:
    if center_id == 0 or center_id == 2:
        return -2.0
    return 2.0


cdef inline double _dt_coeff(int center_id, double alpha, double lam_a, double lam_b, double lam_c, double lam_d) noexcept nogil:
    if center_id == 0:
        return 2.0 * alpha * lam_a
    if center_id == 1:
        return 2.0 * alpha * lam_b
    if center_id == 2:
        return -2.0 * alpha * lam_c
    return -2.0 * alpha * lam_d


cdef int _build_promoted_terms_fixed(
    int rank,
    int* center_ids,
    double* exponents,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    int* orders,
    double* scalars,
    int* nvec,
    int vec_axes[][4],
    int vec_names[][4],
    int* ndelta,
    int delta_axis1[][4],
    int delta_axis2[][4],
) noexcept nogil:
    cdef int next_orders[RYS_FIXED_MAX_TERMS]
    cdef double next_scalars[RYS_FIXED_MAX_TERMS]
    cdef int next_nvec[RYS_FIXED_MAX_TERMS]
    cdef int next_vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int next_vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int next_ndelta[RYS_FIXED_MAX_TERMS]
    cdef int next_delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int next_delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef int nterms = 1
    cdef int next_terms
    cdef int step, t, m, rep_idx, rep_idx2
    cdef int center_id, new_axis
    cdef double scale, pref_mu, pref_sign, dtc, coeff, scalar
    cdef int pref_vec
    cdef int order, nv, nd

    orders[0] = 0
    scalars[0] = 1.0
    nvec[0] = 0
    ndelta[0] = 0

    for step in range(rank):
        center_id = center_ids[step]
        new_axis = step
        scale = 1.0 / (2.0 * exponents[step])
        pref_mu = _pref_mu(center_id, mu_ab, mu_cd)
        pref_vec = _pref_vec_name(center_id)
        pref_sign = _pref_sign(center_id)
        dtc = _dt_coeff(center_id, alpha, lam_a, lam_b, lam_c, lam_d)
        next_terms = 0

        for t in range(nterms):
            order = orders[t]
            scalar = scalars[t]
            nv = nvec[t]
            nd = ndelta[t]

            if next_terms >= RYS_FIXED_MAX_TERMS:
                return -1
            next_orders[next_terms] = order
            next_scalars[next_terms] = scale * scalar * pref_sign * pref_mu
            next_nvec[next_terms] = nv + 1
            next_ndelta[next_terms] = nd
            for m in range(nv):
                next_vec_axes[next_terms][m] = vec_axes[t][m]
                next_vec_names[next_terms][m] = vec_names[t][m]
            next_vec_axes[next_terms][nv] = new_axis
            next_vec_names[next_terms][nv] = pref_vec
            for m in range(nd):
                next_delta_axis1[next_terms][m] = delta_axis1[t][m]
                next_delta_axis2[next_terms][m] = delta_axis2[t][m]
            next_terms += 1

            for rep_idx in range(nv):
                coeff = _vec_diff_coeff(center_id, vec_names[t][rep_idx], lam_a, lam_b, lam_c, lam_d)
                if coeff == 0.0:
                    continue
                if next_terms >= RYS_FIXED_MAX_TERMS:
                    return -1
                next_orders[next_terms] = order
                next_scalars[next_terms] = scale * scalar * coeff
                next_nvec[next_terms] = nv - 1
                next_ndelta[next_terms] = nd + 1
                m = 0
                for rep_idx2 in range(nv):
                    if rep_idx2 == rep_idx:
                        continue
                    next_vec_axes[next_terms][m] = vec_axes[t][rep_idx2]
                    next_vec_names[next_terms][m] = vec_names[t][rep_idx2]
                    m += 1
                for m in range(nd):
                    next_delta_axis1[next_terms][m] = delta_axis1[t][m]
                    next_delta_axis2[next_terms][m] = delta_axis2[t][m]
                next_delta_axis1[next_terms][nd] = vec_axes[t][rep_idx]
                next_delta_axis2[next_terms][nd] = new_axis
                next_terms += 1

            if next_terms >= RYS_FIXED_MAX_TERMS:
                return -1
            next_orders[next_terms] = order + 1
            next_scalars[next_terms] = scale * scalar * (-dtc)
            next_nvec[next_terms] = nv + 1
            next_ndelta[next_terms] = nd
            for m in range(nv):
                next_vec_axes[next_terms][m] = vec_axes[t][m]
                next_vec_names[next_terms][m] = vec_names[t][m]
            next_vec_axes[next_terms][nv] = new_axis
            next_vec_names[next_terms][nv] = 2
            for m in range(nd):
                next_delta_axis1[next_terms][m] = delta_axis1[t][m]
                next_delta_axis2[next_terms][m] = delta_axis2[t][m]
            next_terms += 1

        for t in range(next_terms):
            orders[t] = next_orders[t]
            scalars[t] = next_scalars[t]
            nvec[t] = next_nvec[t]
            ndelta[t] = next_ndelta[t]
            for m in range(next_nvec[t]):
                vec_axes[t][m] = next_vec_axes[t][m]
                vec_names[t][m] = next_vec_names[t][m]
            for m in range(next_ndelta[t]):
                delta_axis1[t][m] = next_delta_axis1[t][m]
                delta_axis2[t][m] = next_delta_axis2[t][m]
        nterms = next_terms

    return nterms


cpdef object evaluate_block(
    int rank,
    cnp.ndarray[i64_t, ndim=1] orders,
    cnp.ndarray[f64_t, ndim=1] scalars,
    cnp.ndarray[i64_t, ndim=1] nvec,
    cnp.ndarray[i64_t, ndim=2] vec_axes,
    cnp.ndarray[i64_t, ndim=2] vec_names,
    cnp.ndarray[i64_t, ndim=1] ndelta,
    cnp.ndarray[i64_t, ndim=2] delta_axis1,
    cnp.ndarray[i64_t, ndim=2] delta_axis2,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    double pref,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef int nterms = orders.shape[0]
    cdef int t, m
    cdef double coeff
    cdef int ia, ib, ic, id_
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ
    cdef cnp.ndarray[f64_t, ndim=1] out1
    cdef cnp.ndarray[f64_t, ndim=2] out2
    cdef cnp.ndarray[f64_t, ndim=3] out3
    cdef cnp.ndarray[f64_t, ndim=4] out4

    if rank == 1:
        out1 = np.zeros((3,), dtype=np.float64)
        for ia in range(3):
            for t in range(nterms):
                coeff = scalars[t] * boys_values[orders[t]]
                for m in range(nvec[t]):
                    coeff *= _vec_component(vec_names[t, m], ia, ABv, CDv, PQv)
                for m in range(ndelta[t]):
                    if ia != ia:
                        coeff = 0.0
                        break
                out1[ia] += coeff
        out1 *= pref
        return out1

    if rank == 2:
        out2 = np.zeros((3, 3), dtype=np.float64)
        for ia in range(3):
            for ib in range(3):
                for t in range(nterms):
                    coeff = scalars[t] * boys_values[orders[t]]
                    for m in range(nvec[t]):
                        if vec_axes[t, m] == 0:
                            coeff *= _vec_component(vec_names[t, m], ia, ABv, CDv, PQv)
                        else:
                            coeff *= _vec_component(vec_names[t, m], ib, ABv, CDv, PQv)
                    for m in range(ndelta[t]):
                        if delta_axis1[t, m] == 0:
                            if delta_axis2[t, m] == 1 and ia != ib:
                                coeff = 0.0
                                break
                        elif delta_axis1[t, m] == 1:
                            if delta_axis2[t, m] == 0 and ib != ia:
                                coeff = 0.0
                                break
                    out2[ia, ib] += coeff
        out2 *= pref
        return out2

    if rank == 3:
        out3 = np.zeros((3, 3, 3), dtype=np.float64)
        for ia in range(3):
            for ib in range(3):
                for ic in range(3):
                    for t in range(nterms):
                        coeff = scalars[t] * boys_values[orders[t]]
                        for m in range(nvec[t]):
                            if vec_axes[t, m] == 0:
                                coeff *= _vec_component(vec_names[t, m], ia, ABv, CDv, PQv)
                            elif vec_axes[t, m] == 1:
                                coeff *= _vec_component(vec_names[t, m], ib, ABv, CDv, PQv)
                            else:
                                coeff *= _vec_component(vec_names[t, m], ic, ABv, CDv, PQv)
                        for m in range(ndelta[t]):
                            if delta_axis1[t, m] == 0:
                                if delta_axis2[t, m] == 1 and ia != ib:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t, m] == 2 and ia != ic:
                                    coeff = 0.0
                                    break
                            elif delta_axis1[t, m] == 1:
                                if delta_axis2[t, m] == 0 and ib != ia:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t, m] == 2 and ib != ic:
                                    coeff = 0.0
                                    break
                            else:
                                if delta_axis2[t, m] == 0 and ic != ia:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t, m] == 1 and ic != ib:
                                    coeff = 0.0
                                    break
                        out3[ia, ib, ic] += coeff
        out3 *= pref
        return out3

    if rank == 4:
        out4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for ia in range(3):
            for ib in range(3):
                for ic in range(3):
                    for id_ in range(3):
                        for t in range(nterms):
                            coeff = scalars[t] * boys_values[orders[t]]
                            for m in range(nvec[t]):
                                if vec_axes[t, m] == 0:
                                    coeff *= _vec_component(vec_names[t, m], ia, ABv, CDv, PQv)
                                elif vec_axes[t, m] == 1:
                                    coeff *= _vec_component(vec_names[t, m], ib, ABv, CDv, PQv)
                                elif vec_axes[t, m] == 2:
                                    coeff *= _vec_component(vec_names[t, m], ic, ABv, CDv, PQv)
                                else:
                                    coeff *= _vec_component(vec_names[t, m], id_, ABv, CDv, PQv)
                            for m in range(ndelta[t]):
                                if delta_axis1[t, m] == 0:
                                    if delta_axis2[t, m] == 1 and ia != ib:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 2 and ia != ic:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 3 and ia != id_:
                                        coeff = 0.0
                                        break
                                elif delta_axis1[t, m] == 1:
                                    if delta_axis2[t, m] == 0 and ib != ia:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 2 and ib != ic:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 3 and ib != id_:
                                        coeff = 0.0
                                        break
                                elif delta_axis1[t, m] == 2:
                                    if delta_axis2[t, m] == 0 and ic != ia:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 1 and ic != ib:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 3 and ic != id_:
                                        coeff = 0.0
                                        break
                                else:
                                    if delta_axis2[t, m] == 0 and id_ != ia:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 1 and id_ != ib:
                                        coeff = 0.0
                                        break
                                    if delta_axis2[t, m] == 2 and id_ != ic:
                                        coeff = 0.0
                                        break
                            out4[ia, ib, ic, id_] += coeff
        out4 *= pref
        return out4

    raise ValueError("evaluate_block currently supports rank 1..4 only.")


cpdef object evaluate_psss(
    double pref,
    double mu_ab,
    double alpha,
    double lam_a,
    double aexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef cnp.ndarray[f64_t, ndim=1] out = np.zeros((3,), dtype=np.float64)
    cdef int i
    cdef double[:] ABv = AB
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double scale = pref / aexp
    for i in range(3):
        out[i] = scale * (-mu_ab * ABv[i] * f0 - alpha * lam_a * PQv[i] * f1)
    return out


cpdef double evaluate_psss_scalar(
    double pref,
    double mu_ab,
    double alpha,
    double lam_a,
    double aexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] PQ,
    int i,
):
    cdef double[:] ABv = AB
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double scale = pref / aexp
    return scale * (-mu_ab * ABv[i] * f0 - alpha * lam_a * PQv[i] * f1)


cpdef object evaluate_ppss(
    double pref,
    double mu_ab,
    double alpha,
    double lam_a,
    double lam_b,
    double aexp,
    double bexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef cnp.ndarray[f64_t, ndim=2] out = np.zeros((3, 3), dtype=np.float64)
    cdef int i, j
    cdef double[:] ABv = AB
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double f2 = boys_values[2]
    cdef double delta
    cdef double scale = pref / (4.0 * aexp * bexp)
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            out[i, j] = scale * (
                (2.0 * mu_ab * delta - 4.0 * mu_ab * mu_ab * ABv[i] * ABv[j]) * f0
                + (
                    4.0 * mu_ab * alpha * lam_b * ABv[i] * PQv[j]
                    - 4.0 * mu_ab * alpha * lam_a * PQv[i] * ABv[j]
                    - 2.0 * alpha * lam_a * lam_b * delta
                ) * f1
                + 4.0 * alpha * alpha * lam_a * lam_b * PQv[i] * PQv[j] * f2
            )
    return out


cpdef double evaluate_ppss_scalar(
    double pref,
    double mu_ab,
    double alpha,
    double lam_a,
    double lam_b,
    double aexp,
    double bexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] PQ,
    int i,
    int j,
):
    cdef double[:] ABv = AB
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double f2 = boys_values[2]
    cdef double delta = 1.0 if i == j else 0.0
    cdef double scale = pref / (4.0 * aexp * bexp)
    return scale * (
        (2.0 * mu_ab * delta - 4.0 * mu_ab * mu_ab * ABv[i] * ABv[j]) * f0
        + (
            4.0 * mu_ab * alpha * lam_b * ABv[i] * PQv[j]
            - 4.0 * mu_ab * alpha * lam_a * PQv[i] * ABv[j]
            - 2.0 * alpha * lam_a * lam_b * delta
        ) * f1
        + 4.0 * alpha * alpha * lam_a * lam_b * PQv[i] * PQv[j] * f2
    )


cpdef object evaluate_psps(
    double pref,
    double mu_ab,
    double mu_cd,
    double alpha,
    double lam_a,
    double lam_c,
    double aexp,
    double cexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef cnp.ndarray[f64_t, ndim=2] out = np.zeros((3, 3), dtype=np.float64)
    cdef int i, k
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double f2 = boys_values[2]
    cdef double delta
    cdef double scale = pref / (aexp * cexp)
    for i in range(3):
        for k in range(3):
            delta = 1.0 if i == k else 0.0
            out[i, k] = scale * (
                mu_ab * mu_cd * ABv[i] * CDv[k] * f0
                + (
                    alpha * lam_a * mu_cd * PQv[i] * CDv[k]
                    - alpha * lam_c * mu_ab * ABv[i] * PQv[k]
                    + 0.5 * alpha * lam_a * lam_c * delta
                ) * f1
                - alpha * alpha * lam_a * lam_c * PQv[i] * PQv[k] * f2
            )
    return out


cpdef double evaluate_psps_scalar(
    double pref,
    double mu_ab,
    double mu_cd,
    double alpha,
    double lam_a,
    double lam_c,
    double aexp,
    double cexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
    int i,
    int k,
):
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ
    cdef double f0 = boys_values[0]
    cdef double f1 = boys_values[1]
    cdef double f2 = boys_values[2]
    cdef double delta = 1.0 if i == k else 0.0
    cdef double scale = pref / (aexp * cexp)
    return scale * (
        mu_ab * mu_cd * ABv[i] * CDv[k] * f0
        + (
            alpha * lam_a * mu_cd * PQv[i] * CDv[k]
            - alpha * lam_c * mu_ab * ABv[i] * PQv[k]
            + 0.5 * alpha * lam_a * lam_c * delta
        ) * f1
        - alpha * alpha * lam_a * lam_c * PQv[i] * PQv[k] * f2
    )


cpdef object evaluate_rank3_terms(
    cnp.ndarray[i64_t, ndim=1] orders,
    cnp.ndarray[f64_t, ndim=1] scalars,
    cnp.ndarray[i64_t, ndim=1] nvec,
    cnp.ndarray[i64_t, ndim=2] vec_axes,
    cnp.ndarray[i64_t, ndim=2] vec_names,
    cnp.ndarray[i64_t, ndim=1] ndelta,
    cnp.ndarray[i64_t, ndim=2] delta_axis1,
    cnp.ndarray[i64_t, ndim=2] delta_axis2,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    double pref,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    return evaluate_block(3, orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2, boys_values, pref, AB, CD, PQ)


cpdef object evaluate_rank4_terms(
    cnp.ndarray[i64_t, ndim=1] orders,
    cnp.ndarray[f64_t, ndim=1] scalars,
    cnp.ndarray[i64_t, ndim=1] nvec,
    cnp.ndarray[i64_t, ndim=2] vec_axes,
    cnp.ndarray[i64_t, ndim=2] vec_names,
    cnp.ndarray[i64_t, ndim=1] ndelta,
    cnp.ndarray[i64_t, ndim=2] delta_axis1,
    cnp.ndarray[i64_t, ndim=2] delta_axis2,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    double pref,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    return evaluate_block(4, orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2, boys_values, pref, AB, CD, PQ)


cpdef object evaluate_ppps_specialized(
    double pref,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    double aexp,
    double bexp,
    double cexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef int center_ids[3]
    cdef double exponents[3]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef cnp.ndarray[f64_t, ndim=3] out = np.zeros((3, 3, 3), dtype=np.float64)
    cdef int nterms
    cdef int ia, ib, ic, t, m
    cdef double coeff
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ

    center_ids[0] = 0
    center_ids[1] = 1
    center_ids[2] = 2
    exponents[0] = aexp
    exponents[1] = bexp
    exponents[2] = cexp

    nterms = _build_promoted_terms_fixed(
        3, center_ids, exponents, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
        orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2
    )
    if nterms < 0:
        raise MemoryError("evaluate_ppps_specialized exceeded fixed term storage")

    for ia in range(3):
        for ib in range(3):
            for ic in range(3):
                for t in range(nterms):
                    coeff = scalars[t] * boys_values[orders[t]]
                    for m in range(nvec[t]):
                        if vec_axes[t][m] == 0:
                            coeff *= _vec_component(vec_names[t][m], ia, ABv, CDv, PQv)
                        elif vec_axes[t][m] == 1:
                            coeff *= _vec_component(vec_names[t][m], ib, ABv, CDv, PQv)
                        else:
                            coeff *= _vec_component(vec_names[t][m], ic, ABv, CDv, PQv)
                    for m in range(ndelta[t]):
                        if delta_axis1[t][m] == 0:
                            if delta_axis2[t][m] == 1 and ia != ib:
                                coeff = 0.0
                                break
                            if delta_axis2[t][m] == 2 and ia != ic:
                                coeff = 0.0
                                break
                        elif delta_axis1[t][m] == 1:
                            if delta_axis2[t][m] == 0 and ib != ia:
                                coeff = 0.0
                                break
                            if delta_axis2[t][m] == 2 and ib != ic:
                                coeff = 0.0
                                break
                        else:
                            if delta_axis2[t][m] == 0 and ic != ia:
                                coeff = 0.0
                                break
                            if delta_axis2[t][m] == 1 and ic != ib:
                                coeff = 0.0
                                break
                    out[ia, ib, ic] += coeff
    out *= pref
    return out


cpdef double evaluate_ppps_scalar(
    double pref,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    double aexp,
    double bexp,
    double cexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
    int ia,
    int ib,
    int ic,
):
    cdef int center_ids[3]
    cdef double exponents[3]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef int nterms
    cdef int t, m
    cdef double coeff
    cdef double term
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ

    center_ids[0] = 0
    center_ids[1] = 1
    center_ids[2] = 2
    exponents[0] = aexp
    exponents[1] = bexp
    exponents[2] = cexp

    nterms = _build_promoted_terms_fixed(
        3, center_ids, exponents, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
        orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2
    )
    if nterms < 0:
        raise MemoryError("evaluate_ppps_scalar exceeded fixed term storage")

    coeff = 0.0
    for t in range(nterms):
        term = scalars[t] * boys_values[orders[t]]
        for m in range(nvec[t]):
            if vec_axes[t][m] == 0:
                term *= _vec_component(vec_names[t][m], ia, ABv, CDv, PQv)
            elif vec_axes[t][m] == 1:
                term *= _vec_component(vec_names[t][m], ib, ABv, CDv, PQv)
            else:
                term *= _vec_component(vec_names[t][m], ic, ABv, CDv, PQv)
        for m in range(ndelta[t]):
            if delta_axis1[t][m] == 0:
                if delta_axis2[t][m] == 1 and ia != ib:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 2 and ia != ic:
                    term = 0.0
                    break
            elif delta_axis1[t][m] == 1:
                if delta_axis2[t][m] == 0 and ib != ia:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 2 and ib != ic:
                    term = 0.0
                    break
            else:
                if delta_axis2[t][m] == 0 and ic != ia:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 1 and ic != ib:
                    term = 0.0
                    break
        coeff += term
    return pref * coeff


cpdef object evaluate_pppp_specialized(
    double pref,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    double aexp,
    double bexp,
    double cexp,
    double dexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef int center_ids[4]
    cdef double exponents[4]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef cnp.ndarray[f64_t, ndim=4] out = np.zeros((3, 3, 3, 3), dtype=np.float64)
    cdef int nterms
    cdef int ia, ib, ic, id_, t, m
    cdef double coeff
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ

    center_ids[0] = 0
    center_ids[1] = 1
    center_ids[2] = 2
    center_ids[3] = 3
    exponents[0] = aexp
    exponents[1] = bexp
    exponents[2] = cexp
    exponents[3] = dexp

    nterms = _build_promoted_terms_fixed(
        4, center_ids, exponents, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
        orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2
    )
    if nterms < 0:
        raise MemoryError("evaluate_pppp_specialized exceeded fixed term storage")

    for ia in range(3):
        for ib in range(3):
            for ic in range(3):
                for id_ in range(3):
                    for t in range(nterms):
                        coeff = scalars[t] * boys_values[orders[t]]
                        for m in range(nvec[t]):
                            if vec_axes[t][m] == 0:
                                coeff *= _vec_component(vec_names[t][m], ia, ABv, CDv, PQv)
                            elif vec_axes[t][m] == 1:
                                coeff *= _vec_component(vec_names[t][m], ib, ABv, CDv, PQv)
                            elif vec_axes[t][m] == 2:
                                coeff *= _vec_component(vec_names[t][m], ic, ABv, CDv, PQv)
                            else:
                                coeff *= _vec_component(vec_names[t][m], id_, ABv, CDv, PQv)
                        for m in range(ndelta[t]):
                            if delta_axis1[t][m] == 0:
                                if delta_axis2[t][m] == 1 and ia != ib:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 2 and ia != ic:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 3 and ia != id_:
                                    coeff = 0.0
                                    break
                            elif delta_axis1[t][m] == 1:
                                if delta_axis2[t][m] == 0 and ib != ia:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 2 and ib != ic:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 3 and ib != id_:
                                    coeff = 0.0
                                    break
                            elif delta_axis1[t][m] == 2:
                                if delta_axis2[t][m] == 0 and ic != ia:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 1 and ic != ib:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 3 and ic != id_:
                                    coeff = 0.0
                                    break
                            else:
                                if delta_axis2[t][m] == 0 and id_ != ia:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 1 and id_ != ib:
                                    coeff = 0.0
                                    break
                                if delta_axis2[t][m] == 2 and id_ != ic:
                                    coeff = 0.0
                                    break
                        out[ia, ib, ic, id_] += coeff
    out *= pref
    return out


cpdef double evaluate_pppp_scalar(
    double pref,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    double aexp,
    double bexp,
    double cexp,
    double dexp,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
    int ia,
    int ib,
    int ic,
    int id_,
):
    cdef int center_ids[4]
    cdef double exponents[4]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef int nterms
    cdef int t, m
    cdef double coeff
    cdef double term
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ

    center_ids[0] = 0
    center_ids[1] = 1
    center_ids[2] = 2
    center_ids[3] = 3
    exponents[0] = aexp
    exponents[1] = bexp
    exponents[2] = cexp
    exponents[3] = dexp

    nterms = _build_promoted_terms_fixed(
        4, center_ids, exponents, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
        orders, scalars, nvec, vec_axes, vec_names, ndelta, delta_axis1, delta_axis2
    )
    if nterms < 0:
        raise MemoryError("evaluate_pppp_scalar exceeded fixed term storage")

    coeff = 0.0
    for t in range(nterms):
        term = scalars[t] * boys_values[orders[t]]
        for m in range(nvec[t]):
            if vec_axes[t][m] == 0:
                term *= _vec_component(vec_names[t][m], ia, ABv, CDv, PQv)
            elif vec_axes[t][m] == 1:
                term *= _vec_component(vec_names[t][m], ib, ABv, CDv, PQv)
            elif vec_axes[t][m] == 2:
                term *= _vec_component(vec_names[t][m], ic, ABv, CDv, PQv)
            else:
                term *= _vec_component(vec_names[t][m], id_, ABv, CDv, PQv)
        for m in range(ndelta[t]):
            if delta_axis1[t][m] == 0:
                if delta_axis2[t][m] == 1 and ia != ib:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 2 and ia != ic:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 3 and ia != id_:
                    term = 0.0
                    break
            elif delta_axis1[t][m] == 1:
                if delta_axis2[t][m] == 0 and ib != ia:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 2 and ib != ic:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 3 and ib != id_:
                    term = 0.0
                    break
            elif delta_axis1[t][m] == 2:
                if delta_axis2[t][m] == 0 and ic != ia:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 1 and ic != ib:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 3 and ic != id_:
                    term = 0.0
                    break
            else:
                if delta_axis2[t][m] == 0 and id_ != ia:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 1 and id_ != ib:
                    term = 0.0
                    break
                if delta_axis2[t][m] == 2 and id_ != ic:
                    term = 0.0
                    break
        coeff += term
    return pref * coeff


cpdef object evaluate_promoted_block(
    cnp.ndarray[i64_t, ndim=1] center_ids,
    cnp.ndarray[f64_t, ndim=1] exponents,
    double alpha,
    double mu_ab,
    double mu_cd,
    double lam_a,
    double lam_b,
    double lam_c,
    double lam_d,
    cnp.ndarray[f64_t, ndim=1] boys_values,
    double pref,
    cnp.ndarray[f64_t, ndim=1] AB,
    cnp.ndarray[f64_t, ndim=1] CD,
    cnp.ndarray[f64_t, ndim=1] PQ,
):
    cdef int rank = center_ids.shape[0]
    cdef int nterms = 1
    cdef int next_terms
    cdef int step, t, m, rep_idx, rep_idx2
    cdef int center_id, new_axis
    cdef double scale, pref_mu, pref_sign, dtc, coeff, scalar
    cdef int pref_vec
    cdef int order, nv, nd

    cdef cnp.ndarray[i64_t, ndim=1] orders = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[f64_t, ndim=1] scalars = np.zeros(MAX_TERMS, dtype=np.float64)
    cdef cnp.ndarray[i64_t, ndim=1] nvec = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] vec_axes = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] vec_names = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=1] ndelta = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] delta_axis1 = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] delta_axis2 = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)

    cdef cnp.ndarray[i64_t, ndim=1] next_orders = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[f64_t, ndim=1] next_scalars = np.zeros(MAX_TERMS, dtype=np.float64)
    cdef cnp.ndarray[i64_t, ndim=1] next_nvec = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] next_vec_axes = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] next_vec_names = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=1] next_ndelta = np.zeros(MAX_TERMS, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] next_delta_axis1 = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=2] next_delta_axis2 = np.full((MAX_TERMS, MAX_FACTORS), -1, dtype=np.int64)

    scalars[0] = 1.0

    for step in range(rank):
        center_id = center_ids[step]
        new_axis = step
        scale = 1.0 / (2.0 * exponents[step])
        pref_mu = _pref_mu(center_id, mu_ab, mu_cd)
        pref_vec = _pref_vec_name(center_id)
        pref_sign = _pref_sign(center_id)
        dtc = _dt_coeff(center_id, alpha, lam_a, lam_b, lam_c, lam_d)
        next_terms = 0

        for t in range(nterms):
            order = orders[t]
            scalar = scalars[t]
            nv = nvec[t]
            nd = ndelta[t]

            # prefactor derivative term
            if next_terms >= MAX_TERMS:
                raise MemoryError("evaluate_promoted_block exceeded MAX_TERMS")
            next_orders[next_terms] = order
            next_scalars[next_terms] = scale * scalar * pref_sign * pref_mu
            next_nvec[next_terms] = nv + 1
            next_ndelta[next_terms] = nd
            for m in range(nv):
                next_vec_axes[next_terms, m] = vec_axes[t, m]
                next_vec_names[next_terms, m] = vec_names[t, m]
            next_vec_axes[next_terms, nv] = new_axis
            next_vec_names[next_terms, nv] = pref_vec
            for m in range(nd):
                next_delta_axis1[next_terms, m] = delta_axis1[t, m]
                next_delta_axis2[next_terms, m] = delta_axis2[t, m]
            next_terms += 1

            # replacements of existing vector factors by delta terms
            for rep_idx in range(nv):
                coeff = _vec_diff_coeff(center_id, vec_names[t, rep_idx], lam_a, lam_b, lam_c, lam_d)
                if coeff == 0.0:
                    continue
                if next_terms >= MAX_TERMS:
                    raise MemoryError("evaluate_promoted_block exceeded MAX_TERMS")
                next_orders[next_terms] = order
                next_scalars[next_terms] = scale * scalar * coeff
                next_nvec[next_terms] = nv - 1
                next_ndelta[next_terms] = nd + 1
                m = 0
                for rep_idx2 in range(nv):
                    if rep_idx2 == rep_idx:
                        continue
                    next_vec_axes[next_terms, m] = vec_axes[t, rep_idx2]
                    next_vec_names[next_terms, m] = vec_names[t, rep_idx2]
                    m += 1
                for m in range(nd):
                    next_delta_axis1[next_terms, m] = delta_axis1[t, m]
                    next_delta_axis2[next_terms, m] = delta_axis2[t, m]
                next_delta_axis1[next_terms, nd] = vec_axes[t, rep_idx]
                next_delta_axis2[next_terms, nd] = new_axis
                next_terms += 1

            # boys derivative term
            if next_terms >= MAX_TERMS:
                raise MemoryError("evaluate_promoted_block exceeded MAX_TERMS")
            next_orders[next_terms] = order + 1
            next_scalars[next_terms] = scale * scalar * (-dtc)
            next_nvec[next_terms] = nv + 1
            next_ndelta[next_terms] = nd
            for m in range(nv):
                next_vec_axes[next_terms, m] = vec_axes[t, m]
                next_vec_names[next_terms, m] = vec_names[t, m]
            next_vec_axes[next_terms, nv] = new_axis
            next_vec_names[next_terms, nv] = 2  # PQ
            for m in range(nd):
                next_delta_axis1[next_terms, m] = delta_axis1[t, m]
                next_delta_axis2[next_terms, m] = delta_axis2[t, m]
            next_terms += 1

        orders[:next_terms] = next_orders[:next_terms]
        scalars[:next_terms] = next_scalars[:next_terms]
        nvec[:next_terms] = next_nvec[:next_terms]
        vec_axes[:next_terms, :] = next_vec_axes[:next_terms, :]
        vec_names[:next_terms, :] = next_vec_names[:next_terms, :]
        ndelta[:next_terms] = next_ndelta[:next_terms]
        delta_axis1[:next_terms, :] = next_delta_axis1[:next_terms, :]
        delta_axis2[:next_terms, :] = next_delta_axis2[:next_terms, :]
        nterms = next_terms

    return evaluate_block(
        rank,
        orders[:nterms],
        scalars[:nterms],
        nvec[:nterms],
        vec_axes[:nterms, :],
        vec_names[:nterms, :],
        ndelta[:nterms],
        delta_axis1[:nterms, :],
        delta_axis2[:nterms, :],
        boys_values,
        pref,
        AB,
        CD,
        PQ,
    )
