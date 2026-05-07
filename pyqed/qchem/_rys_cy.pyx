# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport erf, exp, fabs, sqrt, pow
from libc.stdlib cimport malloc, free


ctypedef cnp.float64_t f64_t
ctypedef cnp.int64_t i64_t

cdef double PI = 3.141592653589793238462643383279502884
cdef double ERI_PREFAC = 2.0 * pow(PI, 2.5)

DEF RYS_FIXED_MAX_FACTORS = 4
DEF RYS_FIXED_MAX_TERMS = 128


cdef int MAX_FACTORS = 8
cdef int MAX_TERMS = 65536


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


cdef inline double _boys_value(int n, double T) noexcept nogil:
    cdef double value
    cdef double term
    cdef double add
    cdef double sqrt_T
    cdef double eT
    cdef int k
    cdef int m
    if T < 1.0e-14:
        return 1.0 / (2.0 * n + 1.0)

    if T < 1.0e-8:
        value = 1.0 / (2.0 * n + 1.0)
        term = 1.0
        for k in range(1, 80):
            term *= -T / k
            add = term / (2.0 * n + 2.0 * k + 1.0)
            value += add
            if fabs(add) < 1.0e-18:
                break
        return value

    sqrt_T = sqrt(T)
    value = 0.5 * sqrt(PI / T) * erf(sqrt_T)
    if n == 0:
        return value

    eT = exp(-T)
    for m in range(n):
        value = ((2.0 * m + 1.0) * value - eT) / (2.0 * T)
    return value


cdef inline double _dt_coeff(int center_id, double alpha, double lam_a, double lam_b, double lam_c, double lam_d) noexcept nogil:
    if center_id == 0:
        return 2.0 * alpha * lam_a
    if center_id == 1:
        return 2.0 * alpha * lam_b
    if center_id == 2:
        return -2.0 * alpha * lam_c
    return -2.0 * alpha * lam_d


cdef inline int _shell_kind_axis(i64_t[:, ::1] shells_v, int idx):
    cdef int lx = <int>shells_v[idx, 0]
    cdef int ly = <int>shells_v[idx, 1]
    cdef int lz = <int>shells_v[idx, 2]
    if lx == 0 and ly == 0 and lz == 0:
        return 3
    if lx == 1 and ly == 0 and lz == 0:
        return 0
    if lx == 0 and ly == 1 and lz == 0:
        return 1
    if lx == 0 and ly == 0 and lz == 1:
        return 2
    if lx == 2 and ly == 0 and lz == 0:
        return 4
    if lx == 0 and ly == 2 and lz == 0:
        return 5
    if lx == 0 and ly == 0 and lz == 2:
        return 6
    if lx == 1 and ly == 1 and lz == 0:
        return 7
    if lx == 1 and ly == 0 and lz == 1:
        return 8
    if lx == 0 and ly == 1 and lz == 1:
        return 9
    return -1


cdef inline int _shell_kind_term_count(int kind) noexcept nogil:
    if kind == 3:
        return 1
    if 0 <= kind <= 2:
        return 1
    if 4 <= kind <= 6:
        return 2
    if 7 <= kind <= 9:
        return 1
    return 0


cdef inline void _shell_kind_fill_term(
    int kind,
    double exponent,
    int term_idx,
    double* coeff,
    int* naxes,
    int* ax0,
    int* ax1,
) noexcept nogil:
    if kind == 3:
        coeff[0] = 1.0
        naxes[0] = 0
        ax0[0] = -1
        ax1[0] = -1
        return
    if 0 <= kind <= 2:
        coeff[0] = 1.0
        naxes[0] = 1
        ax0[0] = kind
        ax1[0] = -1
        return
    if 4 <= kind <= 6:
        if term_idx == 0:
            coeff[0] = 1.0
            naxes[0] = 2
            ax0[0] = kind - 4
            ax1[0] = kind - 4
        else:
            coeff[0] = 1.0 / (2.0 * exponent)
            naxes[0] = 0
            ax0[0] = -1
            ax1[0] = -1
        return
    coeff[0] = 1.0
    naxes[0] = 2
    if kind == 7:
        ax0[0] = 0
        ax1[0] = 1
    elif kind == 8:
        ax0[0] = 0
        ax1[0] = 2
    else:
        ax0[0] = 1
        ax1[0] = 2


cdef inline double _pick3(int axis, double x, double y, double z) noexcept nogil:
    if axis == 0:
        return x
    if axis == 1:
        return y
    return z


cdef inline int _shell_kind_l(int kind) noexcept nogil:
    if kind == 3:
        return 0
    if 0 <= kind <= 2:
        return 1
    if 4 <= kind <= 9:
        return 2
    return -1


cdef inline double _vec_component_arr(int name_id, int idx, double* AB, double* CD, double* PQ) noexcept nogil:
    if name_id == 0:
        return AB[idx]
    if name_id == 1:
        return CD[idx]
    return PQ[idx]


cdef inline double _evaluate_fixed_scalar(
    int rank,
    int* coord_ids,
    int nterms,
    int* orders,
    double* scalars,
    int* nvec,
    int vec_axes[][4],
    int vec_names[][4],
    int* ndelta,
    int delta_axis1[][4],
    int delta_axis2[][4],
    double* boys_values,
    double pref,
    double* AB,
    double* CD,
    double* PQ,
) noexcept nogil:
    cdef int t, m
    cdef double term
    cdef double result = 0.0
    for t in range(nterms):
        term = scalars[t] * boys_values[orders[t]]
        for m in range(nvec[t]):
            term *= _vec_component_arr(vec_names[t][m], coord_ids[vec_axes[t][m]], AB, CD, PQ)
        for m in range(ndelta[t]):
            if coord_ids[delta_axis1[t][m]] != coord_ids[delta_axis2[t][m]]:
                term = 0.0
                break
        result += term
    return pref * result


cdef inline double _contracted_ssss_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
) noexcept nogil:
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef double value = 0.0
    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    value += (
                        weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                        * pref * _boys_value(0, T)
                    )
    return value


cdef inline double _contracted_psss_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
    int iaxis,
) noexcept nogil:
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, scale
    cdef double f0, f1
    cdef double ABi, PQi
    cdef double value = 0.0
    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    f0 = _boys_value(0, T)
                    f1 = _boys_value(1, T)
                    ABi = _pick3(iaxis, ABx, ABy, ABz)
                    PQi = _pick3(iaxis, PQx, PQy, PQz)
                    scale = pref / aexp
                    value += (
                        weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                        * scale * (-mu_ab * ABi * f0 - alpha * lam_a * PQi * f1)
                    )
    return value


cdef inline double _contracted_ppss_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
    int iaxis,
    int jaxis,
) noexcept nogil:
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, scale
    cdef double f0, f1, f2
    cdef double ABi, ABj, PQi, PQj, delta
    cdef double value = 0.0
    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    f0 = _boys_value(0, T)
                    f1 = _boys_value(1, T)
                    f2 = _boys_value(2, T)
                    ABi = _pick3(iaxis, ABx, ABy, ABz)
                    ABj = _pick3(jaxis, ABx, ABy, ABz)
                    PQi = _pick3(iaxis, PQx, PQy, PQz)
                    PQj = _pick3(jaxis, PQx, PQy, PQz)
                    delta = 1.0 if iaxis == jaxis else 0.0
                    scale = pref / (4.0 * aexp * bexp)
                    value += (
                        weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                        * scale * (
                            (2.0 * mu_ab * delta - 4.0 * mu_ab * mu_ab * ABi * ABj) * f0
                            + (
                                4.0 * mu_ab * alpha * lam_b * ABi * PQj
                                - 4.0 * mu_ab * alpha * lam_a * PQi * ABj
                                - 2.0 * alpha * lam_a * lam_b * delta
                            ) * f1
                            + 4.0 * alpha * alpha * lam_a * lam_b * PQi * PQj * f2
                        )
                    )
    return value


cdef inline double _contracted_psps_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
    int iaxis,
    int kaxis,
) noexcept nogil:
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_c
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, scale
    cdef double f0, f1, f2
    cdef double ABi, CDk, PQi, PQk, delta
    cdef double value = 0.0
    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    f0 = _boys_value(0, T)
                    f1 = _boys_value(1, T)
                    f2 = _boys_value(2, T)
                    ABi = _pick3(iaxis, ABx, ABy, ABz)
                    CDk = _pick3(kaxis, CDx, CDy, CDz)
                    PQi = _pick3(iaxis, PQx, PQy, PQz)
                    PQk = _pick3(kaxis, PQx, PQy, PQz)
                    delta = 1.0 if iaxis == kaxis else 0.0
                    scale = pref / (aexp * cexp)
                    value += (
                        weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                        * scale * (
                            mu_ab * mu_cd * ABi * CDk * f0
                            + (
                                alpha * lam_a * mu_cd * PQi * CDk
                                - alpha * lam_c * mu_ab * ABi * PQk
                                + 0.5 * alpha * lam_a * lam_c * delta
                            ) * f1
                            - alpha * alpha * lam_a * lam_c * PQi * PQk * f2
                        )
                    )
    return value


cdef double _contracted_cartesian_scalar_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
    int kind_a,
    int kind_b,
    int kind_c,
    int kind_d,
):
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, weight_prod
    cdef double coeff_a, coeff_b, coeff_c, coeff_d, coeff_prod
    cdef double term_total, boys0
    cdef int nterm_a, nterm_b, nterm_c, nterm_d
    cdef int na, nb, nc, nd, rank, pos
    cdef int ta, tb, tc, td, m
    cdef int axa0, axa1, axb0, axb1, axc0, axc1, axd0, axd1
    cdef cnp.ndarray[i64_t, ndim=1] center_ids_arr = np.empty((MAX_FACTORS,), dtype=np.int64)
    cdef cnp.ndarray[i64_t, ndim=1] coord_ids_arr = np.empty((MAX_FACTORS,), dtype=np.int64)
    cdef cnp.ndarray[f64_t, ndim=1] exponent_ids_arr = np.empty((MAX_FACTORS,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] boys_values_arr = np.empty((MAX_FACTORS + 1,), dtype=np.float64)
    cdef i64_t[::1] center_ids_v = center_ids_arr
    cdef i64_t[::1] coord_ids_v = coord_ids_arr
    cdef double[::1] exponent_ids_v = exponent_ids_arr
    cdef double value = 0.0

    nterm_a = _shell_kind_term_count(kind_a)
    nterm_b = _shell_kind_term_count(kind_b)
    nterm_c = _shell_kind_term_count(kind_c)
    nterm_d = _shell_kind_term_count(kind_d)

    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    lam_d = dexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys0 = _boys_value(0, T)
                    weight_prod = weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                    term_total = 0.0

                    for ta in range(nterm_a):
                        _shell_kind_fill_term(kind_a, aexp, ta, &coeff_a, &na, &axa0, &axa1)
                        for tb in range(nterm_b):
                            _shell_kind_fill_term(kind_b, bexp, tb, &coeff_b, &nb, &axb0, &axb1)
                            for tc in range(nterm_c):
                                _shell_kind_fill_term(kind_c, cexp, tc, &coeff_c, &nc, &axc0, &axc1)
                                for td in range(nterm_d):
                                    _shell_kind_fill_term(kind_d, dexp, td, &coeff_d, &nd, &axd0, &axd1)
                                    coeff_prod = coeff_a * coeff_b * coeff_c * coeff_d
                                    rank = na + nb + nc + nd
                                    if rank == 0:
                                        term_total += coeff_prod * pref * boys0
                                        continue
                                    for m in range(rank + 1):
                                        boys_values_arr[m] = _boys_value(m, T)
                                    pos = 0
                                    if na > 0:
                                        center_ids_v[pos] = 0
                                        coord_ids_v[pos] = axa0
                                        exponent_ids_v[pos] = aexp
                                        pos += 1
                                        if na > 1:
                                            center_ids_v[pos] = 0
                                            coord_ids_v[pos] = axa1
                                            exponent_ids_v[pos] = aexp
                                            pos += 1
                                    if nb > 0:
                                        center_ids_v[pos] = 1
                                        coord_ids_v[pos] = axb0
                                        exponent_ids_v[pos] = bexp
                                        pos += 1
                                        if nb > 1:
                                            center_ids_v[pos] = 1
                                            coord_ids_v[pos] = axb1
                                            exponent_ids_v[pos] = bexp
                                            pos += 1
                                    if nc > 0:
                                        center_ids_v[pos] = 2
                                        coord_ids_v[pos] = axc0
                                        exponent_ids_v[pos] = cexp
                                        pos += 1
                                        if nc > 1:
                                            center_ids_v[pos] = 2
                                            coord_ids_v[pos] = axc1
                                            exponent_ids_v[pos] = cexp
                                            pos += 1
                                    if nd > 0:
                                        center_ids_v[pos] = 3
                                        coord_ids_v[pos] = axd0
                                        exponent_ids_v[pos] = dexp
                                        pos += 1
                                        if nd > 1:
                                            center_ids_v[pos] = 3
                                            coord_ids_v[pos] = axd1
                                            exponent_ids_v[pos] = dexp
                                            pos += 1
                                    term_total += coeff_prod * evaluate_promoted_scalar(
                                        center_ids_arr,
                                        coord_ids_arr,
                                        exponent_ids_arr,
                                        rank,
                                        alpha,
                                        mu_ab,
                                        mu_cd,
                                        lam_a,
                                        lam_b,
                                        lam_c,
                                        lam_d,
                                        boys_values_arr,
                                        pref,
                                        np.asarray([ABx, ABy, ABz], dtype=np.float64),
                                        np.asarray([CDx, CDy, CDz], dtype=np.float64),
                                        np.asarray([PQx, PQy, PQz], dtype=np.float64),
                                    )
                    value += weight_prod * term_total
    return value


cdef inline double _contracted_cartesian_scalar_fixed_mv(
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int a,
    int b,
    int c,
    int d,
    int kind_a,
    int kind_b,
    int kind_c,
    int kind_d,
) noexcept nogil:
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, weight_prod
    cdef double coeff_a, coeff_b, coeff_c, coeff_d, coeff_prod
    cdef double term_total
    cdef int nterm_a, nterm_b, nterm_c, nterm_d
    cdef int na, nb, nc, nd, rank, pos
    cdef int ta, tb, tc, td, m
    cdef int axa0, axa1, axb0, axb1, axc0, axc1, axd0, axd1
    cdef int center_ids[4]
    cdef int coord_ids[4]
    cdef double exponent_ids[4]
    cdef double boys_values[5]
    cdef double AB[3]
    cdef double CD[3]
    cdef double PQ[3]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef int nterms
    cdef double value = 0.0

    nterm_a = _shell_kind_term_count(kind_a)
    nterm_b = _shell_kind_term_count(kind_b)
    nterm_c = _shell_kind_term_count(kind_c)
    nterm_d = _shell_kind_term_count(kind_d)

    for ia in range(nprim_v[a]):
        aexp = exps_v[a, ia]
        for ib in range(nprim_v[b]):
            bexp = exps_v[b, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * origins_v[a, 0] + bexp * origins_v[b, 0]) / p
            Py = (aexp * origins_v[a, 1] + bexp * origins_v[b, 1]) / p
            Pz = (aexp * origins_v[a, 2] + bexp * origins_v[b, 2]) / p
            ABx = origins_v[a, 0] - origins_v[b, 0]
            ABy = origins_v[a, 1] - origins_v[b, 1]
            ABz = origins_v[a, 2] - origins_v[b, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[c]):
                cexp = exps_v[c, ic]
                for id_ in range(nprim_v[d]):
                    dexp = exps_v[d, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    lam_d = dexp / q
                    Qx = (cexp * origins_v[c, 0] + dexp * origins_v[d, 0]) / q
                    Qy = (cexp * origins_v[c, 1] + dexp * origins_v[d, 1]) / q
                    Qz = (cexp * origins_v[c, 2] + dexp * origins_v[d, 2]) / q
                    CDx = origins_v[c, 0] - origins_v[d, 0]
                    CDy = origins_v[c, 1] - origins_v[d, 1]
                    CDz = origins_v[c, 2] - origins_v[d, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    CD[0] = CDx; CD[1] = CDy; CD[2] = CDz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    weight_prod = weights_v[a, ia] * weights_v[b, ib] * weights_v[c, ic] * weights_v[d, id_]
                    term_total = 0.0

                    for ta in range(nterm_a):
                        _shell_kind_fill_term(kind_a, aexp, ta, &coeff_a, &na, &axa0, &axa1)
                        for tb in range(nterm_b):
                            _shell_kind_fill_term(kind_b, bexp, tb, &coeff_b, &nb, &axb0, &axb1)
                            for tc in range(nterm_c):
                                _shell_kind_fill_term(kind_c, cexp, tc, &coeff_c, &nc, &axc0, &axc1)
                                for td in range(nterm_d):
                                    _shell_kind_fill_term(kind_d, dexp, td, &coeff_d, &nd, &axd0, &axd1)
                                    coeff_prod = coeff_a * coeff_b * coeff_c * coeff_d
                                    rank = na + nb + nc + nd
                                    if rank == 0:
                                        term_total += coeff_prod * pref * _boys_value(0, T)
                                        continue
                                    if rank > 4:
                                        return 1.0e300
                                    for m in range(rank + 1):
                                        boys_values[m] = _boys_value(m, T)
                                    pos = 0
                                    if na > 0:
                                        center_ids[pos] = 0
                                        coord_ids[pos] = axa0
                                        exponent_ids[pos] = aexp
                                        pos += 1
                                        if na > 1:
                                            center_ids[pos] = 0
                                            coord_ids[pos] = axa1
                                            exponent_ids[pos] = aexp
                                            pos += 1
                                    if nb > 0:
                                        center_ids[pos] = 1
                                        coord_ids[pos] = axb0
                                        exponent_ids[pos] = bexp
                                        pos += 1
                                        if nb > 1:
                                            center_ids[pos] = 1
                                            coord_ids[pos] = axb1
                                            exponent_ids[pos] = bexp
                                            pos += 1
                                    if nc > 0:
                                        center_ids[pos] = 2
                                        coord_ids[pos] = axc0
                                        exponent_ids[pos] = cexp
                                        pos += 1
                                        if nc > 1:
                                            center_ids[pos] = 2
                                            coord_ids[pos] = axc1
                                            exponent_ids[pos] = cexp
                                            pos += 1
                                    if nd > 0:
                                        center_ids[pos] = 3
                                        coord_ids[pos] = axd0
                                        exponent_ids[pos] = dexp
                                        pos += 1
                                        if nd > 1:
                                            center_ids[pos] = 3
                                            coord_ids[pos] = axd1
                                            exponent_ids[pos] = dexp
                                            pos += 1

                                    nterms = _build_promoted_terms_fixed(
                                        rank,
                                        center_ids,
                                        exponent_ids,
                                        alpha,
                                        mu_ab,
                                        mu_cd,
                                        lam_a,
                                        lam_b,
                                        lam_c,
                                        lam_d,
                                        orders,
                                        scalars,
                                        nvec,
                                        vec_axes,
                                        vec_names,
                                        ndelta,
                                        delta_axis1,
                                        delta_axis2,
                                    )
                                    if nterms < 0:
                                        return 1.0e300
                                    term_total += coeff_prod * _evaluate_fixed_scalar(
                                        rank,
                                        coord_ids,
                                        nterms,
                                        orders,
                                        scalars,
                                        nvec,
                                        vec_axes,
                                        vec_names,
                                        ndelta,
                                        delta_axis1,
                                        delta_axis2,
                                        boys_values,
                                        pref,
                                        AB,
                                        CD,
                                        PQ,
                                    )
                    value += weight_prod * term_total
    return value


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


cpdef double contracted_ssss(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef double value = 0.0
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * pref * _boys_value(0, T)
    return value


cpdef double contracted_psss(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
    int iaxis,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef double boys0, boys1
    cdef double scale
    cdef double value = 0.0
    cdef cnp.ndarray[f64_t, ndim=1] boys_values = np.empty((2,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] AB = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] PQ = np.empty((3,), dtype=np.float64)
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            alpha = 0.0
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys_values[0] = _boys_value(0, T)
                    boys_values[1] = _boys_value(1, T)
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * evaluate_psss_scalar(
                        pref, mu_ab, alpha, lam_a, aexp, boys_values, AB, PQ, iaxis
                    )
    return value


cpdef double contracted_ppss(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
    int iaxis,
    int jaxis,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef cnp.ndarray[f64_t, ndim=1] boys_values = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] AB = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] PQ = np.empty((3,), dtype=np.float64)
    cdef double value = 0.0
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys_values[0] = _boys_value(0, T)
                    boys_values[1] = _boys_value(1, T)
                    boys_values[2] = _boys_value(2, T)
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * evaluate_ppss_scalar(
                        pref, mu_ab, alpha, lam_a, lam_b, aexp, bexp, boys_values, AB, PQ, iaxis, jaxis
                    )
    return value


cpdef double contracted_psps(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
    int iaxis,
    int kaxis,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_c
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef cnp.ndarray[f64_t, ndim=1] boys_values = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] AB = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] CD = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] PQ = np.empty((3,), dtype=np.float64)
    cdef double value = 0.0
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys_values[0] = _boys_value(0, T)
                    boys_values[1] = _boys_value(1, T)
                    boys_values[2] = _boys_value(2, T)
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    CD[0] = CDx; CD[1] = CDy; CD[2] = CDz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * evaluate_psps_scalar(
                        pref, mu_ab, mu_cd, alpha, lam_a, lam_c, aexp, cexp, boys_values, AB, CD, PQ, iaxis, kaxis
                    )
    return value


cpdef double contracted_ppps(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
    int iaxis,
    int jaxis,
    int kaxis,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef cnp.ndarray[f64_t, ndim=1] boys_values = np.empty((4,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] AB = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] CD = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] PQ = np.empty((3,), dtype=np.float64)
    cdef double value = 0.0
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    lam_d = dexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys_values[0] = _boys_value(0, T)
                    boys_values[1] = _boys_value(1, T)
                    boys_values[2] = _boys_value(2, T)
                    boys_values[3] = _boys_value(3, T)
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    CD[0] = CDx; CD[1] = CDy; CD[2] = CDz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * evaluate_ppps_scalar(
                        pref, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
                        aexp, bexp, cexp, boys_values, AB, CD, PQ, iaxis, jaxis, kaxis
                    )
    return value


cpdef double contracted_pppp(
    cnp.ndarray[f64_t, ndim=1] a_exps,
    cnp.ndarray[f64_t, ndim=1] a_weights,
    cnp.ndarray[f64_t, ndim=1] A,
    cnp.ndarray[f64_t, ndim=1] b_exps,
    cnp.ndarray[f64_t, ndim=1] b_weights,
    cnp.ndarray[f64_t, ndim=1] B,
    cnp.ndarray[f64_t, ndim=1] c_exps,
    cnp.ndarray[f64_t, ndim=1] c_weights,
    cnp.ndarray[f64_t, ndim=1] C,
    cnp.ndarray[f64_t, ndim=1] d_exps,
    cnp.ndarray[f64_t, ndim=1] d_weights,
    cnp.ndarray[f64_t, ndim=1] D,
    int iaxis,
    int jaxis,
    int kaxis,
    int laxis,
):
    cdef double[:] aexpv = a_exps
    cdef double[:] awev = a_weights
    cdef double[:] bexpv = b_exps
    cdef double[:] bwev = b_weights
    cdef double[:] cexpv = c_exps
    cdef double[:] cwev = c_weights
    cdef double[:] dexpv = d_exps
    cdef double[:] dwev = d_weights
    cdef double[:] Av = A
    cdef double[:] Bv = B
    cdef double[:] Cv = C
    cdef double[:] Dv = D
    cdef Py_ssize_t ia, ib, ic, id_
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref
    cdef cnp.ndarray[f64_t, ndim=1] boys_values = np.empty((5,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] AB = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] CD = np.empty((3,), dtype=np.float64)
    cdef cnp.ndarray[f64_t, ndim=1] PQ = np.empty((3,), dtype=np.float64)
    cdef double value = 0.0
    for ia in range(a_exps.shape[0]):
        aexp = aexpv[ia]
        for ib in range(b_exps.shape[0]):
            bexp = bexpv[ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * Av[0] + bexp * Bv[0]) / p
            Py = (aexp * Av[1] + bexp * Bv[1]) / p
            Pz = (aexp * Av[2] + bexp * Bv[2]) / p
            ABx = Av[0] - Bv[0]
            ABy = Av[1] - Bv[1]
            ABz = Av[2] - Bv[2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(c_exps.shape[0]):
                cexp = cexpv[ic]
                for id_ in range(d_exps.shape[0]):
                    dexp = dexpv[id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    lam_d = dexp / q
                    Qx = (cexp * Cv[0] + dexp * Dv[0]) / q
                    Qy = (cexp * Cv[1] + dexp * Dv[1]) / q
                    Qz = (cexp * Cv[2] + dexp * Dv[2]) / q
                    CDx = Cv[0] - Dv[0]
                    CDy = Cv[1] - Dv[1]
                    CDz = Cv[2] - Dv[2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    boys_values[0] = _boys_value(0, T)
                    boys_values[1] = _boys_value(1, T)
                    boys_values[2] = _boys_value(2, T)
                    boys_values[3] = _boys_value(3, T)
                    boys_values[4] = _boys_value(4, T)
                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    CD[0] = CDx; CD[1] = CDy; CD[2] = CDz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz
                    value += awev[ia] * bwev[ib] * cwev[ic] * dwev[id_] * evaluate_pppp_scalar(
                        pref, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d,
                        aexp, bexp, cexp, dexp, boys_values, AB, CD, PQ, iaxis, jaxis, kaxis, laxis
                    )
    return value


cpdef double contracted_eri_indices_rys(
    int p,
    int q,
    int r,
    int s,
    cnp.ndarray[i64_t, ndim=2] shells,
    cnp.ndarray[f64_t, ndim=2] origins,
    cnp.ndarray[f64_t, ndim=2] exps,
    cnp.ndarray[f64_t, ndim=2] weights,
    cnp.ndarray[i64_t, ndim=1] nprim,
):
    cdef i64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef i64_t[::1] nprim_v = nprim
    cdef int kind_p = _shell_kind_axis(shells_v, p)
    cdef int kind_q = _shell_kind_axis(shells_v, q)
    cdef int kind_r = _shell_kind_axis(shells_v, r)
    cdef int kind_s = _shell_kind_axis(shells_v, s)
    cdef int np_shells = 0
    cdef int tmp

    if kind_p < 0 or kind_q < 0 or kind_r < 0 or kind_s < 0:
        return 0.0

    if kind_p > 3 or kind_q > 3 or kind_r > 3 or kind_s > 3:
        return _contracted_cartesian_scalar_mv(
            origins_v, exps_v, weights_v, nprim_v, p, q, r, s, kind_p, kind_q, kind_r, kind_s
        )

    if kind_p != 3: np_shells += 1
    if kind_q != 3: np_shells += 1
    if kind_r != 3: np_shells += 1
    if kind_s != 3: np_shells += 1

    if np_shells == 0:
        return _contracted_ssss_mv(origins_v, exps_v, weights_v, nprim_v, p, q, r, s)

    if np_shells == 1:
        if kind_p != 3:
            return _contracted_psss_mv(origins_v, exps_v, weights_v, nprim_v, p, q, r, s, kind_p)
        if kind_q != 3:
            return _contracted_psss_mv(origins_v, exps_v, weights_v, nprim_v, q, p, r, s, kind_q)
        if kind_r != 3:
            return _contracted_psss_mv(origins_v, exps_v, weights_v, nprim_v, r, s, p, q, kind_r)
        return _contracted_psss_mv(origins_v, exps_v, weights_v, nprim_v, s, r, p, q, kind_s)

    if np_shells == 2:
        if kind_p != 3 and kind_q != 3 and kind_r == 3 and kind_s == 3:
            if p > q:
                p, q = q, p
                tmp = kind_p
                kind_p = kind_q
                kind_q = tmp
            if r > s:
                r, s = s, r
            return _contracted_ppss_mv(origins_v, exps_v, weights_v, nprim_v, p, q, r, s, kind_p, kind_q)
        if kind_p == 3 and kind_q == 3 and kind_r != 3 and kind_s != 3:
            if r > s:
                r, s = s, r
                tmp = kind_r
                kind_r = kind_s
                kind_s = tmp
            if p > q:
                p, q = q, p
            return _contracted_ppss_mv(origins_v, exps_v, weights_v, nprim_v, r, s, p, q, kind_r, kind_s)
        if kind_p != 3 and kind_r != 3 and kind_q == 3 and kind_s == 3:
            return _contracted_psps_mv(origins_v, exps_v, weights_v, nprim_v, p, q, r, s, kind_p, kind_r)
        if kind_p != 3 and kind_s != 3 and kind_q == 3 and kind_r == 3:
            return _contracted_psps_mv(origins_v, exps_v, weights_v, nprim_v, p, q, s, r, kind_p, kind_s)
        if kind_q != 3 and kind_r != 3 and kind_p == 3 and kind_s == 3:
            return _contracted_psps_mv(origins_v, exps_v, weights_v, nprim_v, q, p, r, s, kind_q, kind_r)
        return _contracted_psps_mv(origins_v, exps_v, weights_v, nprim_v, q, p, s, r, kind_q, kind_s)

    if np_shells == 3:
        if kind_p == 3:
            if r > s:
                r, s = s, r
                tmp = kind_r
                kind_r = kind_s
                kind_s = tmp
            return contracted_ppps(
                exps[r, :nprim[r]], weights[r, :nprim[r]], origins[r],
                exps[s, :nprim[s]], weights[s, :nprim[s]], origins[s],
                exps[q, :nprim[q]], weights[q, :nprim[q]], origins[q],
                exps[p, :nprim[p]], weights[p, :nprim[p]], origins[p],
                kind_r, kind_s, kind_q,
            )
        if kind_q == 3:
            if r > s:
                r, s = s, r
                tmp = kind_r
                kind_r = kind_s
                kind_s = tmp
            return contracted_ppps(
                exps[r, :nprim[r]], weights[r, :nprim[r]], origins[r],
                exps[s, :nprim[s]], weights[s, :nprim[s]], origins[s],
                exps[p, :nprim[p]], weights[p, :nprim[p]], origins[p],
                exps[q, :nprim[q]], weights[q, :nprim[q]], origins[q],
                kind_r, kind_s, kind_p,
            )
        if kind_r == 3:
            if p > q:
                p, q = q, p
                tmp = kind_p
                kind_p = kind_q
                kind_q = tmp
            return contracted_ppps(
                exps[p, :nprim[p]], weights[p, :nprim[p]], origins[p],
                exps[q, :nprim[q]], weights[q, :nprim[q]], origins[q],
                exps[s, :nprim[s]], weights[s, :nprim[s]], origins[s],
                exps[r, :nprim[r]], weights[r, :nprim[r]], origins[r],
                kind_p, kind_q, kind_s,
            )
        if p > q:
            p, q = q, p
            tmp = kind_p
            kind_p = kind_q
            kind_q = tmp
        return contracted_ppps(
            exps[p, :nprim[p]], weights[p, :nprim[p]], origins[p],
            exps[q, :nprim[q]], weights[q, :nprim[q]], origins[q],
            exps[r, :nprim[r]], weights[r, :nprim[r]], origins[r],
            exps[s, :nprim[s]], weights[s, :nprim[s]], origins[s],
            kind_p, kind_q, kind_r,
        )

    return contracted_pppp(
        exps[p, :nprim[p]], weights[p, :nprim[p]], origins[p],
        exps[q, :nprim[q]], weights[q, :nprim[q]], origins[q],
        exps[r, :nprim[r]], weights[r, :nprim[r]], origins[r],
        exps[s, :nprim[s]], weights[s, :nprim[s]], origins[s],
        kind_p, kind_q, kind_r, kind_s,
    )


cpdef compute_cartesian_shell_quartet_block_rys(
    cnp.ndarray[i64_t, ndim=2] shells,
    cnp.ndarray[f64_t, ndim=2] origins,
    cnp.ndarray[f64_t, ndim=2] exps,
    cnp.ndarray[f64_t, ndim=2] weights,
    cnp.ndarray[i64_t, ndim=1] nprim,
    int p0,
    int p1,
    int q0,
    int q1,
    int r0,
    int r1,
    int s0,
    int s1,
):
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int nr_ = r1 - r0
    cdef int ns_ = s1 - s0
    cdef cnp.ndarray[f64_t, ndim=4] block = np.zeros((np_, nq_, nr_, ns_), dtype=np.float64)
    cdef int ip, iq, ir, is_
    cdef double value
    for ip in range(np_):
        for iq in range(nq_):
            for ir in range(nr_):
                for is_ in range(ns_):
                    value = contracted_eri_indices_rys(
                        p0 + ip, q0 + iq, r0 + ir, s0 + is_, shells, origins, exps, weights, nprim
                    )
                    block[ip, iq, ir, is_] = value
    return block


cdef inline int _is_sp_shell_block(i64_t[:, ::1] shells_v, int start, int stop, int* is_p):
    cdef int n = stop - start
    cdef int kind
    cdef int idx
    if n == 1:
        kind = _shell_kind_axis(shells_v, start)
        if kind == 3:
            is_p[0] = 0
            return 1
        if 0 <= kind <= 2:
            is_p[0] = 1
            return 1
        return 0
    if n == 3:
        for idx in range(start, stop):
            kind = _shell_kind_axis(shells_v, idx)
            if kind < 0 or kind > 2:
                return 0
        is_p[0] = 1
        return 1
    return 0


cdef int _compute_sp_shell_quartet_direct(
    double[:, :, :, :] eri_v,
    i64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    i64_t[::1] nprim_v,
    int p0,
    int p1,
    int q0,
    int q1,
    int r0,
    int r1,
    int s0,
    int s1,
):
    cdef int p_is_p = 0
    cdef int q_is_p = 0
    cdef int r_is_p = 0
    cdef int s_is_p = 0
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int nr_ = r1 - r0
    cdef int ns_ = s1 - s0
    cdef int rank = 0
    cdef int center_ids[4]
    cdef int coord_ids[4]
    cdef double exponent_ids[4]
    cdef double boys_values[5]
    cdef double AB[3]
    cdef double CD[3]
    cdef double PQ[3]
    cdef int orders[RYS_FIXED_MAX_TERMS]
    cdef double scalars[RYS_FIXED_MAX_TERMS]
    cdef int nvec[RYS_FIXED_MAX_TERMS]
    cdef int vec_axes[RYS_FIXED_MAX_TERMS][4]
    cdef int vec_names[RYS_FIXED_MAX_TERMS][4]
    cdef int ndelta[RYS_FIXED_MAX_TERMS]
    cdef int delta_axis1[RYS_FIXED_MAX_TERMS][4]
    cdef int delta_axis2[RYS_FIXED_MAX_TERMS][4]
    cdef double values[81]
    cdef Py_ssize_t ia, ib, ic, id_
    cdef int ip, iq, ir, is_, idx, m, pos, nterms
    cdef int p_axis, q_axis, r_axis, s_axis
    cdef double aexp, bexp, cexp, dexp
    cdef double p, q, alpha, mu_ab, mu_cd, lam_a, lam_b, lam_c, lam_d
    cdef double Px, Py, Pz, Qx, Qy, Qz
    cdef double ABx, ABy, ABz, CDx, CDy, CDz, PQx, PQy, PQz
    cdef double AB2, CD2, PQ2, T, pref, weight_prod, value

    if not _is_sp_shell_block(shells_v, p0, p1, &p_is_p):
        return 0
    if not _is_sp_shell_block(shells_v, q0, q1, &q_is_p):
        return 0
    if not _is_sp_shell_block(shells_v, r0, r1, &r_is_p):
        return 0
    if not _is_sp_shell_block(shells_v, s0, s1, &s_is_p):
        return 0

    if p_is_p:
        center_ids[rank] = 0
        rank += 1
    if q_is_p:
        center_ids[rank] = 1
        rank += 1
    if r_is_p:
        center_ids[rank] = 2
        rank += 1
    if s_is_p:
        center_ids[rank] = 3
        rank += 1

    for idx in range(81):
        values[idx] = 0.0

    for ia in range(nprim_v[p0]):
        aexp = exps_v[p0, ia]
        for ib in range(nprim_v[q0]):
            bexp = exps_v[q0, ib]
            p = aexp + bexp
            mu_ab = aexp * bexp / p
            lam_a = aexp / p
            lam_b = bexp / p
            Px = (aexp * origins_v[p0, 0] + bexp * origins_v[q0, 0]) / p
            Py = (aexp * origins_v[p0, 1] + bexp * origins_v[q0, 1]) / p
            Pz = (aexp * origins_v[p0, 2] + bexp * origins_v[q0, 2]) / p
            ABx = origins_v[p0, 0] - origins_v[q0, 0]
            ABy = origins_v[p0, 1] - origins_v[q0, 1]
            ABz = origins_v[p0, 2] - origins_v[q0, 2]
            AB2 = ABx * ABx + ABy * ABy + ABz * ABz
            for ic in range(nprim_v[r0]):
                cexp = exps_v[r0, ic]
                for id_ in range(nprim_v[s0]):
                    dexp = exps_v[s0, id_]
                    q = cexp + dexp
                    alpha = p * q / (p + q)
                    mu_cd = cexp * dexp / q
                    lam_c = cexp / q
                    lam_d = dexp / q
                    Qx = (cexp * origins_v[r0, 0] + dexp * origins_v[s0, 0]) / q
                    Qy = (cexp * origins_v[r0, 1] + dexp * origins_v[s0, 1]) / q
                    Qz = (cexp * origins_v[r0, 2] + dexp * origins_v[s0, 2]) / q
                    CDx = origins_v[r0, 0] - origins_v[s0, 0]
                    CDy = origins_v[r0, 1] - origins_v[s0, 1]
                    CDz = origins_v[r0, 2] - origins_v[s0, 2]
                    CD2 = CDx * CDx + CDy * CDy + CDz * CDz
                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                    T = alpha * PQ2
                    pref = ERI_PREFAC * exp(-mu_ab * AB2) * exp(-mu_cd * CD2) / (p * q * sqrt(p + q))
                    weight_prod = (
                        weights_v[p0, ia] * weights_v[q0, ib] * weights_v[r0, ic] * weights_v[s0, id_]
                    )

                    if rank == 0:
                        value = weight_prod * pref * _boys_value(0, T)
                        values[0] += value
                        continue

                    for m in range(rank + 1):
                        boys_values[m] = _boys_value(m, T)
                    pos = 0
                    if p_is_p:
                        exponent_ids[pos] = aexp
                        pos += 1
                    if q_is_p:
                        exponent_ids[pos] = bexp
                        pos += 1
                    if r_is_p:
                        exponent_ids[pos] = cexp
                        pos += 1
                    if s_is_p:
                        exponent_ids[pos] = dexp
                        pos += 1

                    nterms = _build_promoted_terms_fixed(
                        rank,
                        center_ids,
                        exponent_ids,
                        alpha,
                        mu_ab,
                        mu_cd,
                        lam_a,
                        lam_b,
                        lam_c,
                        lam_d,
                        orders,
                        scalars,
                        nvec,
                        vec_axes,
                        vec_names,
                        ndelta,
                        delta_axis1,
                        delta_axis2,
                    )
                    if nterms < 0:
                        return 0

                    AB[0] = ABx; AB[1] = ABy; AB[2] = ABz
                    CD[0] = CDx; CD[1] = CDy; CD[2] = CDz
                    PQ[0] = PQx; PQ[1] = PQy; PQ[2] = PQz

                    for ip in range(np_):
                        p_axis = _shell_kind_axis(shells_v, p0 + ip)
                        for iq in range(nq_):
                            q_axis = _shell_kind_axis(shells_v, q0 + iq)
                            for ir in range(nr_):
                                r_axis = _shell_kind_axis(shells_v, r0 + ir)
                                for is_ in range(ns_):
                                    s_axis = _shell_kind_axis(shells_v, s0 + is_)
                                    pos = 0
                                    if p_is_p:
                                        coord_ids[pos] = p_axis
                                        pos += 1
                                    if q_is_p:
                                        coord_ids[pos] = q_axis
                                        pos += 1
                                    if r_is_p:
                                        coord_ids[pos] = r_axis
                                        pos += 1
                                    if s_is_p:
                                        coord_ids[pos] = s_axis
                                        pos += 1
                                    idx = (((ip * nq_) + iq) * nr_ + ir) * ns_ + is_
                                    values[idx] += weight_prod * _evaluate_fixed_scalar(
                                        rank,
                                        coord_ids,
                                        nterms,
                                        orders,
                                        scalars,
                                        nvec,
                                        vec_axes,
                                        vec_names,
                                        ndelta,
                                        delta_axis1,
                                        delta_axis2,
                                        boys_values,
                                        pref,
                                        AB,
                                        CD,
                                        PQ,
                                    )

    for ip in range(np_):
        for iq in range(nq_):
            for ir in range(nr_):
                for is_ in range(ns_):
                    idx = (((ip * nq_) + iq) * nr_ + ir) * ns_ + is_
                    value = values[idx]
                    eri_v[p0 + ip, q0 + iq, r0 + ir, s0 + is_] = value
                    eri_v[q0 + iq, p0 + ip, r0 + ir, s0 + is_] = value
                    eri_v[p0 + ip, q0 + iq, s0 + is_, r0 + ir] = value
                    eri_v[q0 + iq, p0 + ip, s0 + is_, r0 + ir] = value
                    eri_v[r0 + ir, s0 + is_, p0 + ip, q0 + iq] = value
                    eri_v[s0 + is_, r0 + ir, p0 + ip, q0 + iq] = value
                    eri_v[r0 + ir, s0 + is_, q0 + iq, p0 + ip] = value
                    eri_v[s0 + is_, r0 + ir, q0 + iq, p0 + ip] = value
    return 1


cpdef compute_dense_eri_blocked_rys(
    cnp.ndarray[i64_t, ndim=2] shells,
    cnp.ndarray[f64_t, ndim=2] origins,
    cnp.ndarray[f64_t, ndim=2] exps,
    cnp.ndarray[f64_t, ndim=2] weights,
    cnp.ndarray[i64_t, ndim=1] nprim,
    cnp.ndarray[f64_t, ndim=2] pair_bounds,
    cnp.ndarray[i64_t, ndim=1] shell_starts,
    cnp.ndarray[i64_t, ndim=1] shell_stops,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef cnp.ndarray[f64_t, ndim=4] eri = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    cdef int ish, jsh, ksh, lsh
    cdef int p0, p1, q0, q1, r0, r1, s0, s1
    cdef int ip, iq, ir, is_, np_, nq_, nr_, ns_, lsh_max
    cdef i64_t computed = 0
    cdef i64_t skipped = 0
    cdef i64_t npair_pq, npair_rs
    cdef double bound_pq, bound_rs
    cdef double* shell_pair_bounds
    cdef double value
    cdef i64_t[::1] shell_starts_v = shell_starts
    cdef i64_t[::1] shell_stops_v = shell_stops
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef i64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef i64_t[::1] nprim_v = nprim
    cdef double[:, :, :, :] eri_v = eri

    shell_pair_bounds = <double*>malloc(nshell * nshell * sizeof(double))
    if shell_pair_bounds == NULL:
        return eri, 0, 0

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        for jsh in range(nshell):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            bound_pq = 0.0
            for ip in range(p0, p1):
                for iq in range(q0, q1):
                    if pair_bounds_v[ip, iq] > bound_pq:
                        bound_pq = pair_bounds_v[ip, iq]
            shell_pair_bounds[ish * nshell + jsh] = bound_pq

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        np_ = p1 - p0
        for jsh in range(ish + 1):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            nq_ = q1 - q0
            bound_pq = shell_pair_bounds[ish * nshell + jsh]

            for ksh in range(ish + 1):
                r0 = <int>shell_starts_v[ksh]
                r1 = <int>shell_stops_v[ksh]
                nr_ = r1 - r0
                lsh_max = jsh if ksh == ish else ksh
                for lsh in range(lsh_max + 1):
                    s0 = <int>shell_starts_v[lsh]
                    s1 = <int>shell_stops_v[lsh]
                    ns_ = s1 - s0
                    bound_rs = shell_pair_bounds[ksh * nshell + lsh]

                    if screen_tol > 0.0 and bound_pq * bound_rs < screen_tol:
                        skipped += 1
                        continue

                    if not _compute_sp_shell_quartet_direct(
                        eri_v,
                        shells_v,
                        origins_v,
                        exps_v,
                        weights_v,
                        nprim_v,
                        p0,
                        p1,
                        q0,
                        q1,
                        r0,
                        r1,
                        s0,
                        s1,
                    ):
                        for ip in range(p0, p1):
                            for iq in range(q0, q1):
                                for ir in range(r0, r1):
                                    for is_ in range(s0, s1):
                                        value = contracted_eri_indices_rys(
                                            ip, iq, ir, is_, shells, origins, exps, weights, nprim
                                        )
                                        eri_v[ip, iq, ir, is_] = value
                                        eri_v[iq, ip, ir, is_] = value
                                        eri_v[ip, iq, is_, ir] = value
                                        eri_v[iq, ip, is_, ir] = value
                                        eri_v[ir, is_, ip, iq] = value
                                        eri_v[is_, ir, ip, iq] = value
                                        eri_v[ir, is_, iq, ip] = value
                                        eri_v[is_, ir, iq, ip] = value
                    if ish == jsh:
                        npair_pq = (np_ * (np_ + 1)) // 2
                    else:
                        npair_pq = np_ * nq_
                    if ksh == lsh:
                        npair_rs = (nr_ * (nr_ + 1)) // 2
                    else:
                        npair_rs = nr_ * ns_
                    if ish == ksh and jsh == lsh:
                        computed += (npair_pq * (npair_pq + 1)) // 2
                    else:
                        computed += npair_pq * npair_rs

    free(shell_pair_bounds)
    return eri, int(computed), int(skipped)


cpdef double evaluate_promoted_scalar(
    cnp.ndarray[i64_t, ndim=1] center_ids,
    cnp.ndarray[i64_t, ndim=1] coord_ids,
    cnp.ndarray[f64_t, ndim=1] exponents,
    int rank,
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
    cdef int nterms = 1
    cdef int next_terms
    cdef int step, t, m, rep_idx, rep_idx2
    cdef int center_id, new_axis
    cdef double scale, pref_mu, pref_sign, dtc, coeff, scalar
    cdef int pref_vec
    cdef int order, nv, nd
    cdef double term
    cdef double result = 0.0
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
    cdef i64_t[::1] coord_v = coord_ids
    cdef double[:] ABv = AB
    cdef double[:] CDv = CD
    cdef double[:] PQv = PQ

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

            if next_terms >= MAX_TERMS:
                raise MemoryError("evaluate_promoted_scalar exceeded MAX_TERMS")
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

            for rep_idx in range(nv):
                coeff = _vec_diff_coeff(center_id, vec_names[t, rep_idx], lam_a, lam_b, lam_c, lam_d)
                if coeff == 0.0:
                    continue
                if next_terms >= MAX_TERMS:
                    raise MemoryError("evaluate_promoted_scalar exceeded MAX_TERMS")
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

            if next_terms >= MAX_TERMS:
                raise MemoryError("evaluate_promoted_scalar exceeded MAX_TERMS")
            next_orders[next_terms] = order + 1
            next_scalars[next_terms] = scale * scalar * (-dtc)
            next_nvec[next_terms] = nv + 1
            next_ndelta[next_terms] = nd
            for m in range(nv):
                next_vec_axes[next_terms, m] = vec_axes[t, m]
                next_vec_names[next_terms, m] = vec_names[t, m]
            next_vec_axes[next_terms, nv] = new_axis
            next_vec_names[next_terms, nv] = 2
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

    for t in range(nterms):
        term = scalars[t] * boys_values[orders[t]]
        for m in range(nvec[t]):
            term *= _vec_component(vec_names[t, m], coord_v[vec_axes[t, m]], ABv, CDv, PQv)
        for m in range(ndelta[t]):
            if coord_v[delta_axis1[t, m]] != coord_v[delta_axis2[t, m]]:
                term = 0.0
                break
        result += term
    return pref * result


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
