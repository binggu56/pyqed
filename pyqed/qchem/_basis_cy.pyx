# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt, fabs, floor, log, tgamma, NAN
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint64_t, uint8_t
from libc.string cimport memset
from scipy.special.cython_special cimport hyp1f1


cdef extern from *:
    """
    static inline void phase_axpy_ri(
        int n,
        double alpha,
        const double * __restrict phase_real,
        const double * __restrict phase_imag,
        double * __restrict output_real,
        double * __restrict output_imag
    ) {
        #if defined(__clang__)
        #pragma clang loop vectorize(enable) interleave(enable)
        #endif
        for (int i = 0; i < n; ++i) {
            output_real[i] += alpha * phase_real[i];
            output_imag[i] += alpha * phase_imag[i];
        }
    }
    """
    void phase_axpy_ri(
        int n,
        double alpha,
        const double* phase_real,
        const double* phase_imag,
        double* output_real,
        double* output_imag,
    ) noexcept nogil


cdef double PI = 3.141592653589793238462643383279502884
cdef double ERI_PREFAC = 2.0 * (PI ** 2.5)
DEF OS_MAX_STATES = 65536
DEF OS_HASH_CAP = 16384
DEF OS_VRR_PAIR_MAX_L = 4
DEF OS_VRR_MAX_CART = 15
DEF THREE_CENTER_E_CAP = 128
DEF THREE_CENTER_R_CAP = 2048
DEF PAIR_FT_E_CAP = 1024


cdef inline size_t idx3(int i, int j, int t, int jdim, int tdim) noexcept nogil:
    return ((<size_t>i * <size_t>jdim) + <size_t>j) * <size_t>tdim + <size_t>t


cdef inline size_t idx4(int t, int u, int v, int n, int udim, int vdim, int ndim) noexcept nogil:
    return ((((<size_t>t * <size_t>udim) + <size_t>u) * <size_t>vdim) + <size_t>v) * <size_t>ndim + <size_t>n


cdef inline int pair_index(int i, int j) noexcept nogil:
    cdef int tmp
    if i < j:
        tmp = i
        i = j
        j = tmp
    return i * (i + 1) // 2 + j


cdef inline int pair_pair_index(int ij, int kl) noexcept nogil:
    cdef int tmp
    if ij < kl:
        tmp = ij
        ij = kl
        kl = tmp
    return ij * (ij + 1) // 2 + kl


cdef inline bint direct_perm_seen(
    int* aa, int* bb, int* cc, int* dd, int n,
    int a, int b, int c, int d,
) noexcept nogil:
    cdef int idx
    for idx in range(n):
        if aa[idx] == a and bb[idx] == b and cc[idx] == c and dd[idx] == d:
            return True
    return False


cdef inline void direct_jk_add_perm(
    double[:, ::1] vj_v,
    double[:, ::1] vk_v,
    double[:, ::1] dm_v,
    double value,
    int a,
    int b,
    int c,
    int d,
) noexcept nogil:
    vj_v[a, b] += dm_v[d, c] * value
    vk_v[a, d] += dm_v[b, c] * value


cdef inline int ncart_for_l(int l) noexcept nogil:
    return (l + 1) * (l + 2) // 2


cdef inline void fill_cartesian_components(int l, int* lx, int* ly, int* lz) noexcept nogil:
    cdef int ix, iy, iz, idx = 0, rem
    for ix in range(l, -1, -1):
        rem = l - ix
        for iy in range(rem, -1, -1):
            iz = rem - iy
            lx[idx] = ix
            ly[idx] = iy
            lz[idx] = iz
            idx += 1


cdef inline double boys_fn(int n, double T) noexcept nogil:
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0)


cdef inline double primitive_overlap(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz
) noexcept nogil:
    cdef double Sx, Sy, Sz
    cdef int tdim_x = l1 + l2 + 2
    cdef int tdim_y = m1 + m2 + 2
    cdef int tdim_z = n1 + n2 + 2
    cdef size_t size_x = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_x
    cdef size_t size_y = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_y
    cdef size_t size_z = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_z
    cdef double* memo_x = <double*>malloc(size_x * sizeof(double))
    cdef double* memo_y = <double*>malloc(size_y * sizeof(double))
    cdef double* memo_z = <double*>malloc(size_z * sizeof(double))
    cdef size_t i
    if l1 < 0 or m1 < 0 or n1 < 0 or l2 < 0 or m2 < 0 or n2 < 0:
        return 0.0
    if memo_x == NULL or memo_y == NULL or memo_z == NULL:
        free(memo_x); free(memo_y); free(memo_z)
        return 0.0
    for i in range(size_x): memo_x[i] = NAN
    for i in range(size_y): memo_y[i] = NAN
    for i in range(size_z): memo_z[i] = NAN
    Sx = E_rec(l1, l2, 0, Ax - Bx, a, b, memo_x, l2 + 1, tdim_x)
    Sy = E_rec(m1, m2, 0, Ay - By, a, b, memo_y, m2 + 1, tdim_y)
    Sz = E_rec(n1, n2, 0, Az - Bz, a, b, memo_z, n2 + 1, tdim_z)
    free(memo_x); free(memo_y); free(memo_z)
    return Sx * Sy * Sz * ((PI / (a + b)) ** 1.5)


cdef inline double binomial_int(int n, int k) noexcept nogil:
    cdef int i
    cdef double value = 1.0
    if k < 0 or k > n:
        return 0.0
    if k > n - k:
        k = n - k
    for i in range(1, k + 1):
        value *= <double>(n - k + i) / <double>i
    return value


cdef inline double triple_gaussian_axis(
    double p, double P,
    int la, double A,
    int lb, double B,
    int lc, double C,
) noexcept nogil:
    cdef int ia, ib, ic, degree
    cdef double ca, cab, coefficient, value = 0.0
    for ia in range(la + 1):
        ca = binomial_int(la, ia) * ((P - A) ** (la - ia))
        for ib in range(lb + 1):
            cab = ca * binomial_int(lb, ib) * ((P - B) ** (lb - ib))
            for ic in range(lc + 1):
                degree = ia + ib + ic
                if degree & 1:
                    continue
                coefficient = cab * binomial_int(lc, ic) * ((P - C) ** (lc - ic))
                value += coefficient * tgamma(0.5 * (degree + 1)) / (p ** (0.5 * (degree + 1)))
    return value


cdef inline double primitive_three_gaussian_overlap(
    double a, int lax, int lay, int laz, double Ax, double Ay, double Az,
    double b, int lbx, int lby, int lbz, double Bx, double By, double Bz,
    double c, int lcx, int lcy, int lcz, double Cx, double Cy, double Cz,
) noexcept nogil:
    cdef double p = a + b + c
    cdef double Px = (a * Ax + b * Bx + c * Cx) / p
    cdef double Py = (a * Ay + b * By + c * Cy) / p
    cdef double Pz = (a * Az + b * Bz + c * Cz) / p
    cdef double prefactor = exp(
        -(a * (Ax*Ax + Ay*Ay + Az*Az)
          + b * (Bx*Bx + By*By + Bz*Bz)
          + c * (Cx*Cx + Cy*Cy + Cz*Cz))
        + p * (Px*Px + Py*Py + Pz*Pz)
    )
    return prefactor * (
        triple_gaussian_axis(p, Px, lax, Ax, lbx, Bx, lcx, Cx)
        * triple_gaussian_axis(p, Py, lay, Ay, lby, By, lcy, Cy)
        * triple_gaussian_axis(p, Pz, laz, Az, lbz, Bz, lcz, Cz)
    )


cdef inline double primitive_gth_local_gaussian(
    double a, int lax, int lay, int laz, double Ax, double Ay, double Az,
    double b, int lbx, int lby, int lbz, double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    double radius, double* coefficients, int nlocal,
) noexcept nogil:
    cdef int power, ix, iy, iz
    cdef double exponent_c = 0.5 / (radius * radius)
    cdef double value = 0.0
    cdef double multinomial, scale
    for power in range(nlocal):
        if coefficients[power] == 0.0:
            continue
        scale = coefficients[power] / (radius ** (2 * power))
        for ix in range(power + 1):
            for iy in range(power - ix + 1):
                iz = power - ix - iy
                multinomial = (
                    binomial_int(power, ix)
                    * binomial_int(power - ix, iy)
                )
                value += scale * multinomial * primitive_three_gaussian_overlap(
                    a, lax, lay, laz, Ax, Ay, Az,
                    b, lbx, lby, lbz, Bx, By, Bz,
                    exponent_c, 2*ix, 2*iy, 2*iz, Cx, Cy, Cz,
                )
    return value


cdef inline double primitive_kinetic(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz
) noexcept nogil:
    cdef double term0, term1, term2
    term0 = b * (2.0 * (l2 + m2 + n2) + 3.0) * primitive_overlap(
        a, l1, m1, n1, Ax, Ay, Az,
        b, l2, m2, n2, Bx, By, Bz,
    )
    term1 = -2.0 * b * b * (
        primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2 + 2, m2, n2, Bx, By, Bz)
        + primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2, m2 + 2, n2, Bx, By, Bz)
        + primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2, m2, n2 + 2, Bx, By, Bz)
    )
    term2 = -0.5 * (
        l2 * (l2 - 1) * primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2 - 2, m2, n2, Bx, By, Bz)
        + m2 * (m2 - 1) * primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2, m2 - 2, n2, Bx, By, Bz)
        + n2 * (n2 - 1) * primitive_overlap(a, l1, m1, n1, Ax, Ay, Az, b, l2, m2, n2 - 2, Bx, By, Bz)
    )
    return term0 + term1 + term2


cdef inline double primitive_nuclear_attraction_kernel(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    double radial_p
) noexcept nogil:
    cdef double p = a + b
    cdef double px = (a * Ax + b * Bx) / p
    cdef double py = (a * Ay + b * By) / p
    cdef double pz = (a * Az + b * Bz) / p
    cdef double dx = px - Cx
    cdef double dy = py - Cy
    cdef double dz = pz - Cz
    cdef double rpc = sqrt(dx * dx + dy * dy + dz * dz)
    cdef double abx = Ax - Bx
    cdef double aby = Ay - By
    cdef double abz = Az - Bz
    cdef int t, u, v
    cdef double ex, exy, value = 0.0
    cdef int tdim_x = l1 + l2 + 2
    cdef int tdim_y = m1 + m2 + 2
    cdef int tdim_z = n1 + n2 + 2
    cdef size_t size_x = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_x
    cdef size_t size_y = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_y
    cdef size_t size_z = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_z
    cdef int tx_max = l1 + l2
    cdef int uy_max = m1 + m2
    cdef int vz_max = n1 + n2
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double* memo_x = <double*>malloc(size_x * sizeof(double))
    cdef double* memo_y = <double*>malloc(size_y * sizeof(double))
    cdef double* memo_z = <double*>malloc(size_z * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    if l1 < 0 or m1 < 0 or n1 < 0 or l2 < 0 or m2 < 0 or n2 < 0:
        return 0.0
    if memo_x == NULL or memo_y == NULL or memo_z == NULL or memo_r == NULL:
        free(memo_x); free(memo_y); free(memo_z); free(memo_r)
        return 0.0
    for i in range(size_x): memo_x[i] = NAN
    for i in range(size_y): memo_y[i] = NAN
    for i in range(size_z): memo_z[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + l2 + 1):
        ex = E_rec(l1, l2, t, abx, a, b, memo_x, l2 + 1, tdim_x)
        for u in range(m1 + m2 + 1):
            exy = ex * E_rec(m1, m2, u, aby, a, b, memo_y, m2 + 1, tdim_y)
            for v in range(n1 + n2 + 1):
                value += exy * E_rec(n1, n2, v, abz, a, b, memo_z, n2 + 1, tdim_z) * R_rec(
                    t, u, v, 0, radial_p, dx, dy, dz, rpc, memo_r, uy_max + 1, vz_max + 1, nmax
                )
    free(memo_x); free(memo_y); free(memo_z); free(memo_r)
    return value * (2.0 * PI / p)


cdef inline double primitive_nuclear_attraction(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double Cx, double Cy, double Cz
) noexcept nogil:
    return primitive_nuclear_attraction_kernel(
        a, l1, m1, n1, Ax, Ay, Az,
        b, l2, m2, n2, Bx, By, Bz,
        Cx, Cy, Cz,
        a + b,
    )


cdef double E_rec(int i, int j, int t, double Qx, double a, double b, double* memo, int jdim, int tdim) noexcept nogil:
    cdef size_t pos
    cdef double p, q, value

    if t < 0 or t > (i + j):
        return 0.0

    pos = idx3(i, j, t, jdim, tdim)
    if memo[pos] == memo[pos]:
        return memo[pos]

    p = a + b
    q = a * b / p
    if i == 0 and j == 0 and t == 0:
        value = exp(-q * Qx * Qx)
    elif j == 0:
        value = (
            (1.0 / (2.0 * p)) * E_rec(i - 1, j, t - 1, Qx, a, b, memo, jdim, tdim)
            - (q * Qx / a) * E_rec(i - 1, j, t, Qx, a, b, memo, jdim, tdim)
            + (t + 1.0) * E_rec(i - 1, j, t + 1, Qx, a, b, memo, jdim, tdim)
        )
    else:
        value = (
            (1.0 / (2.0 * p)) * E_rec(i, j - 1, t - 1, Qx, a, b, memo, jdim, tdim)
            + (q * Qx / b) * E_rec(i, j - 1, t, Qx, a, b, memo, jdim, tdim)
            + (t + 1.0) * E_rec(i, j - 1, t + 1, Qx, a, b, memo, jdim, tdim)
        )

    memo[pos] = value
    return value


def compute_periodic_pair_ft_primitive_terms(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=3] right_origins_batch,
    cnp.ndarray[int64_t, ndim=1] pair_p,
    cnp.ndarray[int64_t, ndim=1] pair_q,
    int nleft,
    cnp.ndarray[int64_t, ndim=1] prim_start,
    cnp.ndarray[double, ndim=1] prim_alpha,
    cnp.ndarray[double, ndim=1] prim_beta,
    cnp.ndarray[double, ndim=1] prim_alpha_over_p,
    cnp.ndarray[double, ndim=1] prim_beta_over_p,
    cnp.ndarray[double, ndim=1] prim_inv_4p,
    cnp.ndarray[double, ndim=1] prim_prefactor,
    cnp.ndarray[uint8_t, ndim=2] image_pair_mask,
    double coeff_tol=0.0,
):
    """Pack periodic AO-pair Fourier primitive terms in compiled loops."""

    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, :, ::1] right_origins_v = right_origins_batch
    cdef int64_t[::1] pair_p_v = pair_p
    cdef int64_t[::1] pair_q_v = pair_q
    cdef int64_t[::1] prim_start_v = prim_start
    cdef double[::1] prim_alpha_v = prim_alpha
    cdef double[::1] prim_beta_v = prim_beta
    cdef double[::1] prim_alpha_over_p_v = prim_alpha_over_p
    cdef double[::1] prim_beta_over_p_v = prim_beta_over_p
    cdef double[::1] prim_inv_4p_v = prim_inv_4p
    cdef double[::1] prim_prefactor_v = prim_prefactor
    cdef uint8_t[:, ::1] image_pair_mask_v = image_pair_mask
    cdef Py_ssize_t npair = pair_p.shape[0]
    cdef Py_ssize_t nimage = right_origins_batch.shape[0]
    cdef Py_ssize_t pass_index, pair_idx, image, primitive, idx
    cdef Py_ssize_t term_count, image_group_count, product_group_count
    cdef Py_ssize_t image_term_start, image_product_start, product_term_start
    cdef int pidx, qidx, qlocal
    cdef int l1, m1, n1, l2, m2, n2, t, u, v
    cdef int size_x, size_y, size_z
    cdef double ax, ay, az, bx, by, bz, abx, aby, abz
    cdef double alpha, beta, alpha_over_p, beta_over_p
    cdef double center_x, center_y, center_z, inv_4p, prefactor
    cdef double ex, exy, coeff
    cdef double memo_x[PAIR_FT_E_CAP]
    cdef double memo_y[PAIR_FT_E_CAP]
    cdef double memo_z[PAIR_FT_E_CAP]

    cdef cnp.ndarray[int64_t, ndim=1] pair_term_starts = np.empty(
        (npair + 1,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] pair_image_group_starts = np.empty(
        (npair + 1,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] term_image = np.empty((0,), dtype=np.int64)
    cdef cnp.ndarray[double, ndim=2] term_center = np.empty((0, 3), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] term_inv_4p = np.empty((0,), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] term_coeff = np.empty((0,), dtype=np.float64)
    cdef cnp.ndarray[int64_t, ndim=2] term_power = np.empty((0, 3), dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] image_group_image = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] image_group_term_start = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] image_group_term_stop = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] image_group_product_start = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] image_group_product_stop = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] product_group_term_start = np.empty(
        (0,), dtype=np.int64
    )
    cdef cnp.ndarray[int64_t, ndim=1] product_group_term_stop = np.empty(
        (0,), dtype=np.int64
    )

    cdef int64_t[::1] pair_term_starts_v = pair_term_starts
    cdef int64_t[::1] pair_image_group_starts_v = pair_image_group_starts
    cdef int64_t[::1] term_image_v = term_image
    cdef double[:, ::1] term_center_v = term_center
    cdef double[::1] term_inv_4p_out_v = term_inv_4p
    cdef double[::1] term_coeff_v = term_coeff
    cdef int64_t[:, ::1] term_power_v = term_power
    cdef int64_t[::1] image_group_image_v = image_group_image
    cdef int64_t[::1] image_group_term_start_v = image_group_term_start
    cdef int64_t[::1] image_group_term_stop_v = image_group_term_stop
    cdef int64_t[::1] image_group_product_start_v = image_group_product_start
    cdef int64_t[::1] image_group_product_stop_v = image_group_product_stop
    cdef int64_t[::1] product_group_term_start_v = product_group_term_start
    cdef int64_t[::1] product_group_term_stop_v = product_group_term_stop

    if pair_q.shape[0] != npair or prim_start.shape[0] != npair + 1:
        raise ValueError("pair and primitive-start arrays have inconsistent lengths")
    if image_pair_mask.shape[0] != nimage or image_pair_mask.shape[1] != npair:
        raise ValueError("image_pair_mask has an inconsistent shape")
    if left_origins.shape[0] != nleft or left_origins.shape[1] != 3:
        raise ValueError("left_origins has an inconsistent shape")
    if right_origins_batch.shape[2] != 3:
        raise ValueError("right_origins_batch must have a final dimension of three")
    coeff_tol = max(coeff_tol, 0.0)

    for pass_index in range(2):
        term_count = 0
        image_group_count = 0
        product_group_count = 0
        if pass_index == 1:
            pair_term_starts_v[0] = 0
            pair_image_group_starts_v[0] = 0

        for pair_idx in range(npair):
            pidx = <int>pair_p_v[pair_idx]
            qidx = <int>pair_q_v[pair_idx]
            qlocal = qidx - nleft
            if pidx < 0 or pidx >= nleft or qlocal < 0 or qlocal >= right_origins_batch.shape[1]:
                raise ValueError("pair index is outside the AO basis")
            l1 = <int>shells_v[pidx, 0]
            m1 = <int>shells_v[pidx, 1]
            n1 = <int>shells_v[pidx, 2]
            l2 = <int>shells_v[qidx, 0]
            m2 = <int>shells_v[qidx, 1]
            n2 = <int>shells_v[qidx, 2]
            size_x = (l1 + 1) * (l2 + 1) * (l1 + l2 + 2)
            size_y = (m1 + 1) * (m2 + 1) * (m1 + m2 + 2)
            size_z = (n1 + 1) * (n2 + 1) * (n1 + n2 + 2)
            if size_x > PAIR_FT_E_CAP or size_y > PAIR_FT_E_CAP or size_z > PAIR_FT_E_CAP:
                raise NotImplementedError("angular momentum exceeds compiled pair-FT plan capacity")
            ax = left_origins_v[pidx, 0]
            ay = left_origins_v[pidx, 1]
            az = left_origins_v[pidx, 2]

            for image in range(nimage):
                if image_pair_mask_v[image, pair_idx] == 0:
                    continue
                image_term_start = term_count
                image_product_start = product_group_count
                bx = right_origins_v[image, qlocal, 0]
                by = right_origins_v[image, qlocal, 1]
                bz = right_origins_v[image, qlocal, 2]
                abx = ax - bx
                aby = ay - by
                abz = az - bz

                for primitive in range(prim_start_v[pair_idx], prim_start_v[pair_idx + 1]):
                    alpha = prim_alpha_v[primitive]
                    beta = prim_beta_v[primitive]
                    alpha_over_p = prim_alpha_over_p_v[primitive]
                    beta_over_p = prim_beta_over_p_v[primitive]
                    inv_4p = prim_inv_4p_v[primitive]
                    prefactor = prim_prefactor_v[primitive]
                    center_x = alpha_over_p * ax
                    center_x += beta_over_p * bx
                    center_y = alpha_over_p * ay
                    center_y += beta_over_p * by
                    center_z = alpha_over_p * az
                    center_z += beta_over_p * bz
                    product_term_start = term_count
                    for idx in range(size_x):
                        memo_x[idx] = NAN
                    for idx in range(size_y):
                        memo_y[idx] = NAN
                    for idx in range(size_z):
                        memo_z[idx] = NAN

                    for t in range(l1 + l2 + 1):
                        ex = E_rec(l1, l2, t, abx, alpha, beta, memo_x, l2 + 1, l1 + l2 + 2)
                        if ex == 0.0:
                            continue
                        for u in range(m1 + m2 + 1):
                            exy = ex * E_rec(m1, m2, u, aby, alpha, beta, memo_y, m2 + 1, m1 + m2 + 2)
                            if exy == 0.0:
                                continue
                            for v in range(n1 + n2 + 1):
                                coeff = exy * E_rec(n1, n2, v, abz, alpha, beta, memo_z, n2 + 1, n1 + n2 + 2)
                                coeff *= prefactor
                                if fabs(coeff) <= coeff_tol:
                                    continue
                                if pass_index == 1:
                                    term_image_v[term_count] = image
                                    term_center_v[term_count, 0] = center_x
                                    term_center_v[term_count, 1] = center_y
                                    term_center_v[term_count, 2] = center_z
                                    term_inv_4p_out_v[term_count] = inv_4p
                                    term_coeff_v[term_count] = coeff
                                    term_power_v[term_count, 0] = t
                                    term_power_v[term_count, 1] = u
                                    term_power_v[term_count, 2] = v
                                term_count += 1

                    if term_count > product_term_start:
                        if pass_index == 1:
                            product_group_term_start_v[product_group_count] = product_term_start
                            product_group_term_stop_v[product_group_count] = term_count
                        product_group_count += 1

                if term_count > image_term_start:
                    if pass_index == 1:
                        image_group_image_v[image_group_count] = image
                        image_group_term_start_v[image_group_count] = image_term_start
                        image_group_term_stop_v[image_group_count] = term_count
                        image_group_product_start_v[image_group_count] = image_product_start
                        image_group_product_stop_v[image_group_count] = product_group_count
                    image_group_count += 1

            if pass_index == 1:
                pair_term_starts_v[pair_idx + 1] = term_count
                pair_image_group_starts_v[pair_idx + 1] = image_group_count

        if pass_index == 0:
            term_image = np.empty((term_count,), dtype=np.int64)
            term_center = np.empty((term_count, 3), dtype=np.float64)
            term_inv_4p = np.empty((term_count,), dtype=np.float64)
            term_coeff = np.empty((term_count,), dtype=np.float64)
            term_power = np.empty((term_count, 3), dtype=np.int64)
            image_group_image = np.empty((image_group_count,), dtype=np.int64)
            image_group_term_start = np.empty((image_group_count,), dtype=np.int64)
            image_group_term_stop = np.empty((image_group_count,), dtype=np.int64)
            image_group_product_start = np.empty((image_group_count,), dtype=np.int64)
            image_group_product_stop = np.empty((image_group_count,), dtype=np.int64)
            product_group_term_start = np.empty((product_group_count,), dtype=np.int64)
            product_group_term_stop = np.empty((product_group_count,), dtype=np.int64)
            term_image_v = term_image
            term_center_v = term_center
            term_inv_4p_out_v = term_inv_4p
            term_coeff_v = term_coeff
            term_power_v = term_power
            image_group_image_v = image_group_image
            image_group_term_start_v = image_group_term_start
            image_group_term_stop_v = image_group_term_stop
            image_group_product_start_v = image_group_product_start
            image_group_product_stop_v = image_group_product_stop
            product_group_term_start_v = product_group_term_start
            product_group_term_stop_v = product_group_term_stop

    if product_group_count:
        factor_rows = np.empty((product_group_count, 4), dtype=np.float64)
        factor_rows[:, :3] = term_center[product_group_term_start]
        factor_rows[:, 3] = term_inv_4p[product_group_term_start]
        factor_rows = np.round(factor_rows, decimals=15)
        unique_factors, product_group_factor_id = np.unique(
            factor_rows,
            axis=0,
            return_inverse=True,
        )
        product_group_factor_id = np.ascontiguousarray(
            product_group_factor_id,
            dtype=np.int64,
        )
        product_factor_count = int(len(unique_factors))
    else:
        product_group_factor_id = np.empty((0,), dtype=np.int64)
        product_factor_count = 0

    return {
        "pair_term_starts": pair_term_starts,
        "term_image": term_image,
        "term_center": term_center,
        "term_inv_4p": term_inv_4p,
        "term_coeff": term_coeff,
        "term_power": term_power,
        "pair_image_group_starts": pair_image_group_starts,
        "image_group_image": image_group_image,
        "image_group_term_start": image_group_term_start,
        "image_group_term_stop": image_group_term_stop,
        "image_group_product_start": image_group_product_start,
        "image_group_product_stop": image_group_product_stop,
        "product_group_term_start": product_group_term_start,
        "product_group_term_stop": product_group_term_stop,
        "product_group_factor_id": product_group_factor_id,
        "product_group_factor_count": product_factor_count,
    }


cdef double R_rec(
    int t, int u, int v, int n,
    double p, double PCx, double PCy, double PCz, double RPC,
    double* memo, int udim, int vdim, int ndim
) noexcept nogil:
    cdef size_t pos
    cdef double value

    if t < 0 or u < 0 or v < 0:
        return 0.0

    pos = idx4(t, u, v, n, udim, vdim, ndim)
    if memo[pos] == memo[pos]:
        return memo[pos]

    if t == 0 and u == 0 and v == 0:
        value = ((-2.0 * p) ** n) * boys_fn(n, p * RPC * RPC)
    elif t == 0 and u == 0:
        value = 0.0
        if v > 1:
            value += (v - 1.0) * R_rec(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)
        value += PCz * R_rec(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)
    elif t == 0:
        value = 0.0
        if u > 1:
            value += (u - 1.0) * R_rec(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)
        value += PCy * R_rec(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)
    else:
        value = 0.0
        if t > 1:
            value += (t - 1.0) * R_rec(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)
        value += PCx * R_rec(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim)

    memo[pos] = value
    return value


cdef double primitive_eri(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz,
    double d, int l4, int m4, int n4, double Dx, double Dy, double Dz
) noexcept nogil:
    cdef double p = a + b
    cdef double q = c + d
    cdef double alpha = p * q / (p + q)
    cdef double px = (a * Ax + b * Bx) / p
    cdef double py = (a * Ay + b * By) / p
    cdef double pz = (a * Az + b * Bz) / p
    cdef double qx = (c * Cx + d * Dx) / q
    cdef double qy = (c * Cy + d * Dy) / q
    cdef double qz = (c * Cz + d * Dz) / q
    cdef double dx = px - qx
    cdef double dy = py - qy
    cdef double dz = pz - qz
    cdef double rpq = sqrt(dx * dx + dy * dy + dz * dz)
    cdef double abx = Ax - Bx
    cdef double aby = Ay - By
    cdef double abz = Az - Bz
    cdef double cdx = Cx - Dx
    cdef double cdy = Cy - Dy
    cdef double cdz = Cz - Dz
    cdef int tdim_abx = l1 + l2 + 2
    cdef int tdim_aby = m1 + m2 + 2
    cdef int tdim_abz = n1 + n2 + 2
    cdef int tdim_cdx = l3 + l4 + 2
    cdef int tdim_cdy = m3 + m4 + 2
    cdef int tdim_cdz = n3 + n4 + 2
    cdef size_t size_abx = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_abx
    cdef size_t size_aby = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_aby
    cdef size_t size_abz = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_abz
    cdef size_t size_cdx = <size_t>(l3 + 1) * <size_t>(l4 + 1) * <size_t>tdim_cdx
    cdef size_t size_cdy = <size_t>(m3 + 1) * <size_t>(m4 + 1) * <size_t>tdim_cdy
    cdef size_t size_cdz = <size_t>(n3 + 1) * <size_t>(n4 + 1) * <size_t>tdim_cdz
    cdef int tx_max = l1 + l2 + l3 + l4
    cdef int uy_max = m1 + m2 + m3 + m4
    cdef int vz_max = n1 + n2 + n3 + n4
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double* memo_abx = <double*>malloc(size_abx * sizeof(double))
    cdef double* memo_aby = <double*>malloc(size_aby * sizeof(double))
    cdef double* memo_abz = <double*>malloc(size_abz * sizeof(double))
    cdef double* memo_cdx = <double*>malloc(size_cdx * sizeof(double))
    cdef double* memo_cdy = <double*>malloc(size_cdy * sizeof(double))
    cdef double* memo_cdz = <double*>malloc(size_cdz * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    cdef double value

    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cdx == NULL or memo_cdy == NULL or memo_cdz == NULL or memo_r == NULL:
        free(memo_abx); free(memo_aby); free(memo_abz)
        free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r)
        return 0.0

    for i in range(size_abx): memo_abx[i] = NAN
    for i in range(size_aby): memo_aby[i] = NAN
    for i in range(size_abz): memo_abz[i] = NAN
    for i in range(size_cdx): memo_cdx[i] = NAN
    for i in range(size_cdy): memo_cdy[i] = NAN
    for i in range(size_cdz): memo_cdz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN

    value = primitive_eri_from_memos(
        l1, m1, n1, l2, m2, n2, l3, m3, n3, l4, m4, n4,
        a, b, c, d, abx, aby, abz, cdx, cdy, cdz,
        alpha, dx, dy, dz, rpq,
        memo_abx, memo_aby, memo_abz, memo_cdx, memo_cdy, memo_cdz, memo_r,
        l2 + 1, tdim_abx, m2 + 1, tdim_aby, n2 + 1, tdim_abz,
        l4 + 1, tdim_cdx, m4 + 1, tdim_cdy, n4 + 1, tdim_cdz,
        uy_max + 1, vz_max + 1, nmax,
    )

    free(memo_abx); free(memo_aby); free(memo_abz)
    free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r)
    return value


cdef inline double primitive_two_center_coulomb(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz
) noexcept nogil:
    cdef double p = a
    cdef double q = b
    cdef double alpha = p * q / (p + q)
    cdef double dx = Ax - Bx
    cdef double dy = Ay - By
    cdef double dz = Az - Bz
    cdef double rab = sqrt(dx * dx + dy * dy + dz * dz)
    cdef int tdim_ax = l1 + 2
    cdef int tdim_ay = m1 + 2
    cdef int tdim_az = n1 + 2
    cdef int tdim_bx = l2 + 2
    cdef int tdim_by = m2 + 2
    cdef int tdim_bz = n2 + 2
    cdef size_t size_ax = <size_t>(l1 + 1) * <size_t>tdim_ax
    cdef size_t size_ay = <size_t>(m1 + 1) * <size_t>tdim_ay
    cdef size_t size_az = <size_t>(n1 + 1) * <size_t>tdim_az
    cdef size_t size_bx = <size_t>(l2 + 1) * <size_t>tdim_bx
    cdef size_t size_by = <size_t>(m2 + 1) * <size_t>tdim_by
    cdef size_t size_bz = <size_t>(n2 + 1) * <size_t>tdim_bz
    cdef int tx_max = l1 + l2
    cdef int uy_max = m1 + m2
    cdef int vz_max = n1 + n2
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double* memo_ax = <double*>malloc(size_ax * sizeof(double))
    cdef double* memo_ay = <double*>malloc(size_ay * sizeof(double))
    cdef double* memo_az = <double*>malloc(size_az * sizeof(double))
    cdef double* memo_bx = <double*>malloc(size_bx * sizeof(double))
    cdef double* memo_by = <double*>malloc(size_by * sizeof(double))
    cdef double* memo_bz = <double*>malloc(size_bz * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_a, exy_a, xyz_a, ex_b, exy_b, sign, value = 0.0
    if memo_ax == NULL or memo_ay == NULL or memo_az == NULL or memo_bx == NULL or memo_by == NULL or memo_bz == NULL or memo_r == NULL:
        free(memo_ax); free(memo_ay); free(memo_az); free(memo_bx); free(memo_by); free(memo_bz); free(memo_r)
        return 0.0
    for i in range(size_ax): memo_ax[i] = NAN
    for i in range(size_ay): memo_ay[i] = NAN
    for i in range(size_az): memo_az[i] = NAN
    for i in range(size_bx): memo_bx[i] = NAN
    for i in range(size_by): memo_by[i] = NAN
    for i in range(size_bz): memo_bz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + 1):
        ex_a = E_rec(l1, 0, t, 0.0, a, 0.0, memo_ax, 1, tdim_ax)
        for u in range(m1 + 1):
            exy_a = ex_a * E_rec(m1, 0, u, 0.0, a, 0.0, memo_ay, 1, tdim_ay)
            for v in range(n1 + 1):
                xyz_a = exy_a * E_rec(n1, 0, v, 0.0, a, 0.0, memo_az, 1, tdim_az)
                for tau in range(l2 + 1):
                    ex_b = E_rec(l2, 0, tau, 0.0, b, 0.0, memo_bx, 1, tdim_bx)
                    for nu in range(m2 + 1):
                        exy_b = ex_b * E_rec(m2, 0, nu, 0.0, b, 0.0, memo_by, 1, tdim_by)
                        for phi in range(n2 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += xyz_a * exy_b * E_rec(n2, 0, phi, 0.0, b, 0.0, memo_bz, 1, tdim_bz) * sign * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rab, memo_r, uy_max + 1, vz_max + 1, nmax)
    free(memo_ax); free(memo_ay); free(memo_az); free(memo_bx); free(memo_by); free(memo_bz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline double primitive_three_center_coulomb(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz
) noexcept nogil:
    cdef double p = a + b
    cdef double q = c
    cdef double alpha = p * q / (p + q)
    cdef double px = (a * Ax + b * Bx) / p
    cdef double py = (a * Ay + b * By) / p
    cdef double pz = (a * Az + b * Bz) / p
    cdef double dx = px - Cx
    cdef double dy = py - Cy
    cdef double dz = pz - Cz
    cdef double rpc = sqrt(dx * dx + dy * dy + dz * dz)
    cdef double abx = Ax - Bx
    cdef double aby = Ay - By
    cdef double abz = Az - Bz
    cdef int tdim_abx = l1 + l2 + 2
    cdef int tdim_aby = m1 + m2 + 2
    cdef int tdim_abz = n1 + n2 + 2
    cdef int tdim_cx = l3 + 2
    cdef int tdim_cy = m3 + 2
    cdef int tdim_cz = n3 + 2
    cdef size_t size_abx = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_abx
    cdef size_t size_aby = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_aby
    cdef size_t size_abz = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_abz
    cdef size_t size_cx = <size_t>(l3 + 1) * <size_t>tdim_cx
    cdef size_t size_cy = <size_t>(m3 + 1) * <size_t>tdim_cy
    cdef size_t size_cz = <size_t>(n3 + 1) * <size_t>tdim_cz
    cdef int tx_max = l1 + l2 + l3
    cdef int uy_max = m1 + m2 + m3
    cdef int vz_max = n1 + n2 + n3
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double memo_abx_stack[THREE_CENTER_E_CAP]
    cdef double memo_aby_stack[THREE_CENTER_E_CAP]
    cdef double memo_abz_stack[THREE_CENTER_E_CAP]
    cdef double memo_cx_stack[THREE_CENTER_E_CAP]
    cdef double memo_cy_stack[THREE_CENTER_E_CAP]
    cdef double memo_cz_stack[THREE_CENTER_E_CAP]
    cdef double memo_r_stack[THREE_CENTER_R_CAP]
    cdef double* memo_abx = NULL
    cdef double* memo_aby = NULL
    cdef double* memo_abz = NULL
    cdef double* memo_cx = NULL
    cdef double* memo_cy = NULL
    cdef double* memo_cz = NULL
    cdef double* memo_r = NULL
    cdef bint use_heap = False
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_c, exy_c, sign, value = 0.0

    if (
        size_abx <= THREE_CENTER_E_CAP
        and size_aby <= THREE_CENTER_E_CAP
        and size_abz <= THREE_CENTER_E_CAP
        and size_cx <= THREE_CENTER_E_CAP
        and size_cy <= THREE_CENTER_E_CAP
        and size_cz <= THREE_CENTER_E_CAP
        and size_r <= THREE_CENTER_R_CAP
    ):
        memo_abx = &memo_abx_stack[0]
        memo_aby = &memo_aby_stack[0]
        memo_abz = &memo_abz_stack[0]
        memo_cx = &memo_cx_stack[0]
        memo_cy = &memo_cy_stack[0]
        memo_cz = &memo_cz_stack[0]
        memo_r = &memo_r_stack[0]
    else:
        use_heap = True
        memo_abx = <double*>malloc(size_abx * sizeof(double))
        memo_aby = <double*>malloc(size_aby * sizeof(double))
        memo_abz = <double*>malloc(size_abz * sizeof(double))
        memo_cx = <double*>malloc(size_cx * sizeof(double))
        memo_cy = <double*>malloc(size_cy * sizeof(double))
        memo_cz = <double*>malloc(size_cz * sizeof(double))
        memo_r = <double*>malloc(size_r * sizeof(double))

    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cx == NULL or memo_cy == NULL or memo_cz == NULL or memo_r == NULL:
        if use_heap:
            free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
        return 0.0
    for i in range(size_abx): memo_abx[i] = NAN
    for i in range(size_aby): memo_aby[i] = NAN
    for i in range(size_abz): memo_abz[i] = NAN
    for i in range(size_cx): memo_cx[i] = NAN
    for i in range(size_cy): memo_cy[i] = NAN
    for i in range(size_cz): memo_cz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + l2 + 1):
        ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, l2 + 1, tdim_abx)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, m2 + 1, tdim_aby)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, n2 + 1, tdim_abz)
                for tau in range(l3 + 1):
                    ex_c = E_rec(l3, 0, tau, 0.0, c, 0.0, memo_cx, 1, tdim_cx)
                    for nu in range(m3 + 1):
                        exy_c = ex_c * E_rec(m3, 0, nu, 0.0, c, 0.0, memo_cy, 1, tdim_cy)
                        for phi in range(n3 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += xyz_ab * exy_c * E_rec(n3, 0, phi, 0.0, c, 0.0, memo_cz, 1, tdim_cz) * sign * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpc, memo_r, uy_max + 1, vz_max + 1, nmax)
    if use_heap:
        free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline double primitive_three_center_precomputed(
    double a, double b, double p, double px, double py, double pz, double abx, double aby, double abz,
    int l1, int m1, int n1, int l2, int m2, int n2,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz
) noexcept nogil:
    cdef double q = c
    cdef double alpha = p * q / (p + q)
    cdef double dx = px - Cx
    cdef double dy = py - Cy
    cdef double dz = pz - Cz
    cdef double rpc = sqrt(dx * dx + dy * dy + dz * dz)
    cdef int tdim_abx = l1 + l2 + 2
    cdef int tdim_aby = m1 + m2 + 2
    cdef int tdim_abz = n1 + n2 + 2
    cdef int tdim_cx = l3 + 2
    cdef int tdim_cy = m3 + 2
    cdef int tdim_cz = n3 + 2
    cdef size_t size_abx = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_abx
    cdef size_t size_aby = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_aby
    cdef size_t size_abz = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_abz
    cdef size_t size_cx = <size_t>(l3 + 1) * <size_t>tdim_cx
    cdef size_t size_cy = <size_t>(m3 + 1) * <size_t>tdim_cy
    cdef size_t size_cz = <size_t>(n3 + 1) * <size_t>tdim_cz
    cdef int tx_max = l1 + l2 + l3
    cdef int uy_max = m1 + m2 + m3
    cdef int vz_max = n1 + n2 + n3
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double memo_abx_stack[THREE_CENTER_E_CAP]
    cdef double memo_aby_stack[THREE_CENTER_E_CAP]
    cdef double memo_abz_stack[THREE_CENTER_E_CAP]
    cdef double memo_cx_stack[THREE_CENTER_E_CAP]
    cdef double memo_cy_stack[THREE_CENTER_E_CAP]
    cdef double memo_cz_stack[THREE_CENTER_E_CAP]
    cdef double memo_r_stack[THREE_CENTER_R_CAP]
    cdef double* memo_abx = NULL
    cdef double* memo_aby = NULL
    cdef double* memo_abz = NULL
    cdef double* memo_cx = NULL
    cdef double* memo_cy = NULL
    cdef double* memo_cz = NULL
    cdef double* memo_r = NULL
    cdef bint use_heap = False
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_c, exy_c, sign, value = 0.0

    if (
        size_abx <= THREE_CENTER_E_CAP
        and size_aby <= THREE_CENTER_E_CAP
        and size_abz <= THREE_CENTER_E_CAP
        and size_cx <= THREE_CENTER_E_CAP
        and size_cy <= THREE_CENTER_E_CAP
        and size_cz <= THREE_CENTER_E_CAP
        and size_r <= THREE_CENTER_R_CAP
    ):
        memo_abx = &memo_abx_stack[0]
        memo_aby = &memo_aby_stack[0]
        memo_abz = &memo_abz_stack[0]
        memo_cx = &memo_cx_stack[0]
        memo_cy = &memo_cy_stack[0]
        memo_cz = &memo_cz_stack[0]
        memo_r = &memo_r_stack[0]
    else:
        use_heap = True
        memo_abx = <double*>malloc(size_abx * sizeof(double))
        memo_aby = <double*>malloc(size_aby * sizeof(double))
        memo_abz = <double*>malloc(size_abz * sizeof(double))
        memo_cx = <double*>malloc(size_cx * sizeof(double))
        memo_cy = <double*>malloc(size_cy * sizeof(double))
        memo_cz = <double*>malloc(size_cz * sizeof(double))
        memo_r = <double*>malloc(size_r * sizeof(double))

    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cx == NULL or memo_cy == NULL or memo_cz == NULL or memo_r == NULL:
        if use_heap:
            free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
        return 0.0
    for i in range(size_abx): memo_abx[i] = NAN
    for i in range(size_aby): memo_aby[i] = NAN
    for i in range(size_abz): memo_abz[i] = NAN
    for i in range(size_cx): memo_cx[i] = NAN
    for i in range(size_cy): memo_cy[i] = NAN
    for i in range(size_cz): memo_cz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + l2 + 1):
        ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, l2 + 1, tdim_abx)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, m2 + 1, tdim_aby)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, n2 + 1, tdim_abz)
                for tau in range(l3 + 1):
                    ex_c = E_rec(l3, 0, tau, 0.0, c, 0.0, memo_cx, 1, tdim_cx)
                    for nu in range(m3 + 1):
                        exy_c = ex_c * E_rec(m3, 0, nu, 0.0, c, 0.0, memo_cy, 1, tdim_cy)
                        for phi in range(n3 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += (
                                xyz_ab
                                * exy_c
                                * E_rec(n3, 0, phi, 0.0, c, 0.0, memo_cz, 1, tdim_cz)
                                * sign
                                * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpc, memo_r, uy_max + 1, vz_max + 1, nmax)
                            )
    if use_heap:
        free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline double primitive_two_center_coulomb_kernel(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double alpha_kernel
) noexcept nogil:
    cdef double p = a
    cdef double q = b
    cdef double dx = Ax - Bx
    cdef double dy = Ay - By
    cdef double dz = Az - Bz
    cdef double rab = sqrt(dx * dx + dy * dy + dz * dz)
    cdef int tdim_ax = l1 + 2
    cdef int tdim_ay = m1 + 2
    cdef int tdim_az = n1 + 2
    cdef int tdim_bx = l2 + 2
    cdef int tdim_by = m2 + 2
    cdef int tdim_bz = n2 + 2
    cdef size_t size_ax = <size_t>(l1 + 1) * <size_t>tdim_ax
    cdef size_t size_ay = <size_t>(m1 + 1) * <size_t>tdim_ay
    cdef size_t size_az = <size_t>(n1 + 1) * <size_t>tdim_az
    cdef size_t size_bx = <size_t>(l2 + 1) * <size_t>tdim_bx
    cdef size_t size_by = <size_t>(m2 + 1) * <size_t>tdim_by
    cdef size_t size_bz = <size_t>(n2 + 1) * <size_t>tdim_bz
    cdef int tx_max = l1 + l2
    cdef int uy_max = m1 + m2
    cdef int vz_max = n1 + n2
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double* memo_ax = <double*>malloc(size_ax * sizeof(double))
    cdef double* memo_ay = <double*>malloc(size_ay * sizeof(double))
    cdef double* memo_az = <double*>malloc(size_az * sizeof(double))
    cdef double* memo_bx = <double*>malloc(size_bx * sizeof(double))
    cdef double* memo_by = <double*>malloc(size_by * sizeof(double))
    cdef double* memo_bz = <double*>malloc(size_bz * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_a, exy_a, xyz_a, ex_b, exy_b, sign, value = 0.0
    if memo_ax == NULL or memo_ay == NULL or memo_az == NULL or memo_bx == NULL or memo_by == NULL or memo_bz == NULL or memo_r == NULL:
        free(memo_ax); free(memo_ay); free(memo_az); free(memo_bx); free(memo_by); free(memo_bz); free(memo_r)
        return 0.0
    for i in range(size_ax): memo_ax[i] = NAN
    for i in range(size_ay): memo_ay[i] = NAN
    for i in range(size_az): memo_az[i] = NAN
    for i in range(size_bx): memo_bx[i] = NAN
    for i in range(size_by): memo_by[i] = NAN
    for i in range(size_bz): memo_bz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + 1):
        ex_a = E_rec(l1, 0, t, 0.0, a, 0.0, memo_ax, 1, tdim_ax)
        for u in range(m1 + 1):
            exy_a = ex_a * E_rec(m1, 0, u, 0.0, a, 0.0, memo_ay, 1, tdim_ay)
            for v in range(n1 + 1):
                xyz_a = exy_a * E_rec(n1, 0, v, 0.0, a, 0.0, memo_az, 1, tdim_az)
                for tau in range(l2 + 1):
                    ex_b = E_rec(l2, 0, tau, 0.0, b, 0.0, memo_bx, 1, tdim_bx)
                    for nu in range(m2 + 1):
                        exy_b = ex_b * E_rec(m2, 0, nu, 0.0, b, 0.0, memo_by, 1, tdim_by)
                        for phi in range(n2 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += xyz_a * exy_b * E_rec(n2, 0, phi, 0.0, b, 0.0, memo_bz, 1, tdim_bz) * sign * R_rec(t + tau, u + nu, v + phi, 0, alpha_kernel, dx, dy, dz, rab, memo_r, uy_max + 1, vz_max + 1, nmax)
    free(memo_ax); free(memo_ay); free(memo_az); free(memo_bx); free(memo_by); free(memo_bz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline double primitive_short_range_two_center_coulomb(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double eta
) noexcept nogil:
    cdef double theta, eta2, theta_lr, lr_scale
    if eta <= 0.0:
        return primitive_two_center_coulomb(
            a, l1, m1, n1, Ax, Ay, Az,
            b, l2, m2, n2, Bx, By, Bz,
        )
    theta = a * b / (a + b)
    eta2 = eta * eta
    theta_lr = theta * eta2 / (theta + eta2)
    lr_scale = eta / sqrt(theta + eta2)
    return (
        primitive_two_center_coulomb(
            a, l1, m1, n1, Ax, Ay, Az,
            b, l2, m2, n2, Bx, By, Bz,
        )
        - lr_scale * primitive_two_center_coulomb_kernel(
            a, l1, m1, n1, Ax, Ay, Az,
            b, l2, m2, n2, Bx, By, Bz,
            theta_lr,
        )
    )


cdef inline double primitive_three_center_coulomb_kernel(
    double a, double b, double p, double px, double py, double pz, double abx, double aby, double abz,
    int l1, int m1, int n1, int l2, int m2, int n2,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz,
    double alpha_kernel
) noexcept nogil:
    cdef double q = c
    cdef double dx = px - Cx
    cdef double dy = py - Cy
    cdef double dz = pz - Cz
    cdef double rpc = sqrt(dx * dx + dy * dy + dz * dz)
    cdef int tdim_abx = l1 + l2 + 2
    cdef int tdim_aby = m1 + m2 + 2
    cdef int tdim_abz = n1 + n2 + 2
    cdef int tdim_cx = l3 + 2
    cdef int tdim_cy = m3 + 2
    cdef int tdim_cz = n3 + 2
    cdef size_t size_abx = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_abx
    cdef size_t size_aby = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_aby
    cdef size_t size_abz = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_abz
    cdef size_t size_cx = <size_t>(l3 + 1) * <size_t>tdim_cx
    cdef size_t size_cy = <size_t>(m3 + 1) * <size_t>tdim_cy
    cdef size_t size_cz = <size_t>(n3 + 1) * <size_t>tdim_cz
    cdef int tx_max = l1 + l2 + l3
    cdef int uy_max = m1 + m2 + m3
    cdef int vz_max = n1 + n2 + n3
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double memo_abx_stack[THREE_CENTER_E_CAP]
    cdef double memo_aby_stack[THREE_CENTER_E_CAP]
    cdef double memo_abz_stack[THREE_CENTER_E_CAP]
    cdef double memo_cx_stack[THREE_CENTER_E_CAP]
    cdef double memo_cy_stack[THREE_CENTER_E_CAP]
    cdef double memo_cz_stack[THREE_CENTER_E_CAP]
    cdef double memo_r_stack[THREE_CENTER_R_CAP]
    cdef double* memo_abx = NULL
    cdef double* memo_aby = NULL
    cdef double* memo_abz = NULL
    cdef double* memo_cx = NULL
    cdef double* memo_cy = NULL
    cdef double* memo_cz = NULL
    cdef double* memo_r = NULL
    cdef bint use_heap = False
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_c, exy_c, sign, value = 0.0

    if (
        size_abx <= THREE_CENTER_E_CAP
        and size_aby <= THREE_CENTER_E_CAP
        and size_abz <= THREE_CENTER_E_CAP
        and size_cx <= THREE_CENTER_E_CAP
        and size_cy <= THREE_CENTER_E_CAP
        and size_cz <= THREE_CENTER_E_CAP
        and size_r <= THREE_CENTER_R_CAP
    ):
        memo_abx = &memo_abx_stack[0]
        memo_aby = &memo_aby_stack[0]
        memo_abz = &memo_abz_stack[0]
        memo_cx = &memo_cx_stack[0]
        memo_cy = &memo_cy_stack[0]
        memo_cz = &memo_cz_stack[0]
        memo_r = &memo_r_stack[0]
    else:
        use_heap = True
        memo_abx = <double*>malloc(size_abx * sizeof(double))
        memo_aby = <double*>malloc(size_aby * sizeof(double))
        memo_abz = <double*>malloc(size_abz * sizeof(double))
        memo_cx = <double*>malloc(size_cx * sizeof(double))
        memo_cy = <double*>malloc(size_cy * sizeof(double))
        memo_cz = <double*>malloc(size_cz * sizeof(double))
        memo_r = <double*>malloc(size_r * sizeof(double))

    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cx == NULL or memo_cy == NULL or memo_cz == NULL or memo_r == NULL:
        if use_heap:
            free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
        return 0.0
    for i in range(size_abx): memo_abx[i] = NAN
    for i in range(size_aby): memo_aby[i] = NAN
    for i in range(size_abz): memo_abz[i] = NAN
    for i in range(size_cx): memo_cx[i] = NAN
    for i in range(size_cy): memo_cy[i] = NAN
    for i in range(size_cz): memo_cz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN
    for t in range(l1 + l2 + 1):
        ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, l2 + 1, tdim_abx)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, m2 + 1, tdim_aby)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, n2 + 1, tdim_abz)
                for tau in range(l3 + 1):
                    ex_c = E_rec(l3, 0, tau, 0.0, c, 0.0, memo_cx, 1, tdim_cx)
                    for nu in range(m3 + 1):
                        exy_c = ex_c * E_rec(m3, 0, nu, 0.0, c, 0.0, memo_cy, 1, tdim_cy)
                        for phi in range(n3 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += (
                                xyz_ab
                                * exy_c
                                * E_rec(n3, 0, phi, 0.0, c, 0.0, memo_cz, 1, tdim_cz)
                                * sign
                                * R_rec(t + tau, u + nu, v + phi, 0, alpha_kernel, dx, dy, dz, rpc, memo_r, uy_max + 1, vz_max + 1, nmax)
                            )
    if use_heap:
        free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline double primitive_short_range_three_center_coulomb(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz,
    double eta
) noexcept nogil:
    cdef double p, q, theta, eta2, theta_lr, lr_scale, px, py, pz
    cdef double abx, aby, abz
    if eta <= 0.0:
        return primitive_three_center_coulomb(
            a, l1, m1, n1, Ax, Ay, Az,
            b, l2, m2, n2, Bx, By, Bz,
            c, l3, m3, n3, Cx, Cy, Cz,
        )
    p = a + b
    q = c
    theta = p * q / (p + q)
    eta2 = eta * eta
    theta_lr = theta * eta2 / (theta + eta2)
    lr_scale = eta / sqrt(theta + eta2)
    px = (a * Ax + b * Bx) / p
    py = (a * Ay + b * By) / p
    pz = (a * Az + b * Bz) / p
    abx = Ax - Bx
    aby = Ay - By
    abz = Az - Bz
    return (
        primitive_three_center_precomputed(
            a, b, p, px, py, pz, abx, aby, abz,
            l1, m1, n1, l2, m2, n2,
            c, l3, m3, n3, Cx, Cy, Cz,
        )
        - lr_scale * primitive_three_center_coulomb_kernel(
            a, b, p, px, py, pz, abx, aby, abz,
            l1, m1, n1, l2, m2, n2,
            c, l3, m3, n3, Cx, Cy, Cz,
            theta_lr,
        )
    )


cdef inline double primitive_short_range_three_center_precomputed(
    double a, double b, double p, double px, double py, double pz, double abx, double aby, double abz,
    int l1, int m1, int n1, int l2, int m2, int n2,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz,
    double eta
) noexcept nogil:
    cdef double q, theta, eta2, theta_lr, lr_scale
    if eta <= 0.0:
        return primitive_three_center_precomputed(
            a, b, p, px, py, pz, abx, aby, abz,
            l1, m1, n1, l2, m2, n2,
            c, l3, m3, n3, Cx, Cy, Cz,
        )
    q = c
    theta = p * q / (p + q)
    eta2 = eta * eta
    theta_lr = theta * eta2 / (theta + eta2)
    lr_scale = eta / sqrt(theta + eta2)
    return (
        primitive_three_center_precomputed(
            a, b, p, px, py, pz, abx, aby, abz,
            l1, m1, n1, l2, m2, n2,
            c, l3, m3, n3, Cx, Cy, Cz,
        )
        - lr_scale * primitive_three_center_coulomb_kernel(
            a, b, p, px, py, pz, abx, aby, abz,
            l1, m1, n1, l2, m2, n2,
            c, l3, m3, n3, Cx, Cy, Cz,
            theta_lr,
        )
    )


cdef inline double primitive_eri_from_memos(
    int l1, int m1, int n1, int l2, int m2, int n2,
    int l3, int m3, int n3, int l4, int m4, int n4,
    double a, double b, double c, double d,
    double abx, double aby, double abz,
    double cdx, double cdy, double cdz,
    double alpha, double dx, double dy, double dz, double rpq,
    double* memo_abx, double* memo_aby, double* memo_abz,
    double* memo_cdx, double* memo_cdy, double* memo_cdz,
    double* memo_r,
    int jdim_abx, int tdim_abx, int jdim_aby, int tdim_aby, int jdim_abz, int tdim_abz,
    int jdim_cdx, int tdim_cdx, int jdim_cdy, int tdim_cdy, int jdim_cdz, int tdim_cdz,
    int udim_r, int vdim_r, int ndim_r,
) noexcept nogil:
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_cd, exy_cd, sign, value = 0.0

    for t in range(l1 + l2 + 1):
        ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, jdim_abx, tdim_abx)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, jdim_aby, tdim_aby)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, jdim_abz, tdim_abz)
                for tau in range(l3 + l4 + 1):
                    ex_cd = E_rec(l3, l4, tau, cdx, c, d, memo_cdx, jdim_cdx, tdim_cdx)
                    for nu in range(m3 + m4 + 1):
                        exy_cd = ex_cd * E_rec(m3, m4, nu, cdy, c, d, memo_cdy, jdim_cdy, tdim_cdy)
                        for phi in range(n3 + n4 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += (
                                xyz_ab
                                * exy_cd
                                * E_rec(n3, n4, phi, cdz, c, d, memo_cdz, jdim_cdz, tdim_cdz)
                                * sign
                                * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpq, memo_r, udim_r, vdim_r, ndim_r)
                            )

    return value * (ERI_PREFAC / (a + b) / (c + d) / sqrt(a + b + c + d))


cdef inline double primitive_eri_precomputed(
    double a, double b, double p, double px, double py, double pz, double abx, double aby, double abz,
    int l1, int m1, int n1, int l2, int m2, int n2,
    double c, double d, double q, double qx, double qy, double qz, double cdx, double cdy, double cdz,
    int l3, int m3, int n3, int l4, int m4, int n4,
) noexcept nogil:
    cdef double alpha = p * q / (p + q)
    cdef double dx = px - qx
    cdef double dy = py - qy
    cdef double dz = pz - qz
    cdef double rpq = sqrt(dx * dx + dy * dy + dz * dz)
    cdef int tdim_abx = l1 + l2 + 2
    cdef int tdim_aby = m1 + m2 + 2
    cdef int tdim_abz = n1 + n2 + 2
    cdef int tdim_cdx = l3 + l4 + 2
    cdef int tdim_cdy = m3 + m4 + 2
    cdef int tdim_cdz = n3 + n4 + 2
    cdef size_t size_abx = <size_t>(l1 + 1) * <size_t>(l2 + 1) * <size_t>tdim_abx
    cdef size_t size_aby = <size_t>(m1 + 1) * <size_t>(m2 + 1) * <size_t>tdim_aby
    cdef size_t size_abz = <size_t>(n1 + 1) * <size_t>(n2 + 1) * <size_t>tdim_abz
    cdef size_t size_cdx = <size_t>(l3 + 1) * <size_t>(l4 + 1) * <size_t>tdim_cdx
    cdef size_t size_cdy = <size_t>(m3 + 1) * <size_t>(m4 + 1) * <size_t>tdim_cdy
    cdef size_t size_cdz = <size_t>(n3 + 1) * <size_t>(n4 + 1) * <size_t>tdim_cdz
    cdef int tx_max = l1 + l2 + l3 + l4
    cdef int uy_max = m1 + m2 + m3 + m4
    cdef int vz_max = n1 + n2 + n3 + n4
    cdef int nmax = tx_max + uy_max + vz_max + 2
    cdef size_t size_r = <size_t>(tx_max + 1) * <size_t>(uy_max + 1) * <size_t>(vz_max + 1) * <size_t>nmax
    cdef double* memo_abx = <double*>malloc(size_abx * sizeof(double))
    cdef double* memo_aby = <double*>malloc(size_aby * sizeof(double))
    cdef double* memo_abz = <double*>malloc(size_abz * sizeof(double))
    cdef double* memo_cdx = <double*>malloc(size_cdx * sizeof(double))
    cdef double* memo_cdy = <double*>malloc(size_cdy * sizeof(double))
    cdef double* memo_cdz = <double*>malloc(size_cdz * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_cd, exy_cd, sign, value = 0.0

    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cdx == NULL or memo_cdy == NULL or memo_cdz == NULL or memo_r == NULL:
        free(memo_abx); free(memo_aby); free(memo_abz)
        free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r)
        return 0.0

    for i in range(size_abx): memo_abx[i] = NAN
    for i in range(size_aby): memo_aby[i] = NAN
    for i in range(size_abz): memo_abz[i] = NAN
    for i in range(size_cdx): memo_cdx[i] = NAN
    for i in range(size_cdy): memo_cdy[i] = NAN
    for i in range(size_cdz): memo_cdz[i] = NAN
    for i in range(size_r): memo_r[i] = NAN

    for t in range(l1 + l2 + 1):
        ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, l2 + 1, tdim_abx)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, m2 + 1, tdim_aby)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, n2 + 1, tdim_abz)
                for tau in range(l3 + l4 + 1):
                    ex_cd = E_rec(l3, l4, tau, cdx, c, d, memo_cdx, l4 + 1, tdim_cdx)
                    for nu in range(m3 + m4 + 1):
                        exy_cd = ex_cd * E_rec(m3, m4, nu, cdy, c, d, memo_cdy, m4 + 1, tdim_cdy)
                        for phi in range(n3 + n4 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            value += (
                                xyz_ab
                                * exy_cd
                                * E_rec(n3, n4, phi, cdz, c, d, memo_cdz, n4 + 1, tdim_cdz)
                                * sign
                                * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpq, memo_r, uy_max + 1, vz_max + 1, nmax)
                            )

    free(memo_abx); free(memo_aby); free(memo_abz)
    free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


cdef inline int os_state_matches(
    int* states,
    int idx,
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    int dx, int dy, int dz,
    int m,
) noexcept nogil:
    cdef int off = idx * 13
    return (
        states[off] == ax and states[off + 1] == ay and states[off + 2] == az
        and states[off + 3] == bx and states[off + 4] == by and states[off + 5] == bz
        and states[off + 6] == cx and states[off + 7] == cy and states[off + 8] == cz
        and states[off + 9] == dx and states[off + 10] == dy and states[off + 11] == dz
        and states[off + 12] == m
    )


cdef double os_eri_rec(
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    int dx, int dy, int dz,
    int m,
    double p, double q, double z, double rho, double T, double base_pref,
    double PAx, double PAy, double PAz,
    double PBx, double PBy, double PBz,
    double QCx, double QCy, double QCz,
    double QDx, double QDy, double QDz,
    double PQx, double PQy, double PQz,
    double ABx, double ABy, double ABz,
    double CDx, double CDy, double CDz,
    int* states,
    double* values,
    int* nstates,
) noexcept nogil:
    cdef int idx, off, axis
    cdef int na = ax + ay + az
    cdef int nb = bx + by + bz
    cdef int nc = cx + cy + cz
    cdef int nd = dx + dy + dz
    cdef double pq_axis, center_axis, value

    if ax < 0 or ay < 0 or az < 0 or bx < 0 or by < 0 or bz < 0 or cx < 0 or cy < 0 or cz < 0 or dx < 0 or dy < 0 or dz < 0:
        return 0.0

    for idx in range(nstates[0]):
        if os_state_matches(states, idx, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m):
            return values[idx]

    if na == 0 and nb == 0 and nc == 0 and nd == 0:
        value = base_pref * boys_fn(m, T)
    elif na > 0:
        if ax >= ay and ax >= az:
            axis = 0
            pq_axis = PQx
            center_axis = PAx
            value = center_axis * os_eri_rec(ax - 1, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * pq_axis * os_eri_rec(ax - 1, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if ax - 1 > 0:
                value += (ax - 1) / (2.0 * p) * (os_eri_rec(ax - 2, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax - 2, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if bx > 0:
                value += bx / (2.0 * p) * (os_eri_rec(ax - 1, ay, az, bx - 1, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax - 1, ay, az, bx - 1, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if cx > 0:
                value += cx / (2.0 * z) * os_eri_rec(ax - 1, ay, az, bx, by, bz, cx - 1, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if dx > 0:
                value += dx / (2.0 * z) * os_eri_rec(ax - 1, ay, az, bx, by, bz, cx, cy, cz, dx - 1, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        elif ay >= az:
            value = PAy * os_eri_rec(ax, ay - 1, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * PQy * os_eri_rec(ax, ay - 1, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if ay - 1 > 0:
                value += (ay - 1) / (2.0 * p) * (os_eri_rec(ax, ay - 2, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax, ay - 2, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if by > 0:
                value += by / (2.0 * p) * (os_eri_rec(ax, ay - 1, az, bx, by - 1, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax, ay - 1, az, bx, by - 1, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if cy > 0:
                value += cy / (2.0 * z) * os_eri_rec(ax, ay - 1, az, bx, by, bz, cx, cy - 1, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if dy > 0:
                value += dy / (2.0 * z) * os_eri_rec(ax, ay - 1, az, bx, by, bz, cx, cy, cz, dx, dy - 1, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        else:
            value = PAz * os_eri_rec(ax, ay, az - 1, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * PQz * os_eri_rec(ax, ay, az - 1, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if az - 1 > 0:
                value += (az - 1) / (2.0 * p) * (os_eri_rec(ax, ay, az - 2, bx, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax, ay, az - 2, bx, by, bz, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if bz > 0:
                value += bz / (2.0 * p) * (os_eri_rec(ax, ay, az - 1, bx, by, bz - 1, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - q / z * os_eri_rec(ax, ay, az - 1, bx, by, bz - 1, cx, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if cz > 0:
                value += cz / (2.0 * z) * os_eri_rec(ax, ay, az - 1, bx, by, bz, cx, cy, cz - 1, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if dz > 0:
                value += dz / (2.0 * z) * os_eri_rec(ax, ay, az - 1, bx, by, bz, cx, cy, cz, dx, dy, dz - 1, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
    elif nc > 0:
        # Move ket angular momentum up on C.  This is the A recurrence with P/Q swapped signs.
        if cx >= cy and cx >= cz:
            value = QCx * os_eri_rec(ax, ay, az, bx, by, bz, cx - 1, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + p / z * PQx * os_eri_rec(ax, ay, az, bx, by, bz, cx - 1, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if cx - 1 > 0:
                value += (cx - 1) / (2.0 * q) * (os_eri_rec(ax, ay, az, bx, by, bz, cx - 2, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - p / z * os_eri_rec(ax, ay, az, bx, by, bz, cx - 2, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if dx > 0:
                value += dx / (2.0 * q) * (os_eri_rec(ax, ay, az, bx, by, bz, cx - 1, cy, cz, dx - 1, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) - p / z * os_eri_rec(ax, ay, az, bx, by, bz, cx - 1, cy, cz, dx - 1, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates))
            if ax > 0:
                value += ax / (2.0 * z) * os_eri_rec(ax - 1, ay, az, bx, by, bz, cx - 1, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
            if bx > 0:
                value += bx / (2.0 * z) * os_eri_rec(ax, ay, az, bx - 1, by, bz, cx - 1, cy, cz, dx, dy, dz, m + 1, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        elif cy >= cz:
            value = os_eri_rec(ax, ay, az, bx, by, bz, cy, cx, cz, dy, dx, dz, m, p, q, z, rho, T, base_pref, PAy, PAx, PAz, PBy, PBx, PBz, QCy, QCx, QCz, QDy, QDx, QDz, PQy, PQx, PQz, ABy, ABx, ABz, CDy, CDx, CDz, states, values, nstates)
        else:
            value = os_eri_rec(ax, ay, az, bx, by, bz, cz, cy, cx, dz, dy, dx, m, p, q, z, rho, T, base_pref, PAz, PAy, PAx, PBz, PBy, PBx, QCz, QCy, QCx, QDz, QDy, QDx, PQz, PQy, PQx, ABz, ABy, ABx, CDz, CDy, CDx, states, values, nstates)
    elif nb > 0:
        if bx >= by and bx >= bz:
            value = os_eri_rec(ax + 1, ay, az, bx - 1, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + ABx * os_eri_rec(ax, ay, az, bx - 1, by, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        elif by >= bz:
            value = os_eri_rec(ax, ay + 1, az, bx, by - 1, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + ABy * os_eri_rec(ax, ay, az, bx, by - 1, bz, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        else:
            value = os_eri_rec(ax, ay, az + 1, bx, by, bz - 1, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + ABz * os_eri_rec(ax, ay, az, bx, by, bz - 1, cx, cy, cz, dx, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
    else:
        if dx >= dy and dx >= dz:
            value = os_eri_rec(ax, ay, az, bx, by, bz, cx + 1, cy, cz, dx - 1, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + CDx * os_eri_rec(ax, ay, az, bx, by, bz, cx, cy, cz, dx - 1, dy, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        elif dy >= dz:
            value = os_eri_rec(ax, ay, az, bx, by, bz, cx, cy + 1, cz, dx, dy - 1, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + CDy * os_eri_rec(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy - 1, dz, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)
        else:
            value = os_eri_rec(ax, ay, az, bx, by, bz, cx, cy, cz + 1, dx, dy, dz - 1, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates) + CDz * os_eri_rec(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz - 1, m, p, q, z, rho, T, base_pref, PAx, PAy, PAz, PBx, PBy, PBz, QCx, QCy, QCz, QDx, QDy, QDz, PQx, PQy, PQz, ABx, ABy, ABz, CDx, CDy, CDz, states, values, nstates)

    if nstates[0] < OS_MAX_STATES:
        off = nstates[0] * 13
        states[off] = ax; states[off + 1] = ay; states[off + 2] = az
        states[off + 3] = bx; states[off + 4] = by; states[off + 5] = bz
        states[off + 6] = cx; states[off + 7] = cy; states[off + 8] = cz
        states[off + 9] = dx; states[off + 10] = dy; states[off + 11] = dz
        states[off + 12] = m
        values[nstates[0]] = value
        nstates[0] += 1
    return value


cdef inline int os_state_matches_arr(int* states, int idx, int* ang, int m) noexcept nogil:
    cdef int off = idx * 13
    cdef int i
    for i in range(12):
        if states[off + i] != ang[i]:
            return 0
    return states[off + 12] == m


cdef inline void os_copy_state(int* dst, int* src) noexcept nogil:
    cdef int i
    for i in range(12):
        dst[i] = src[i]


cdef inline int os_axis_of_max3(int a, int b, int c) noexcept nogil:
    if a >= b and a >= c:
        return 0
    if b >= c:
        return 1
    return 2


cdef inline uint64_t os_pack_key(int* ang, int m) noexcept nogil:
    cdef uint64_t key = <uint64_t>(m & 15)
    cdef int i
    for i in range(12):
        key = (key << 4) | <uint64_t>(ang[i] & 15)
    return key + 1


cdef inline int os_hash_slot(uint64_t key) noexcept nogil:
    key ^= key >> 33
    key *= <uint64_t>0xff51afd7ed558ccd
    key ^= key >> 33
    return <int>(key & <uint64_t>(OS_HASH_CAP - 1))


cdef inline void os_seed_boys_hash(
    int max_m,
    double T,
    double base_pref,
    uint64_t* keys,
    double* values,
    int* nstates,
) noexcept nogil:
    cdef int m, slot
    cdef int zero_ang[12]
    cdef uint64_t key
    cdef double value = base_pref * boys_fn(max_m, T)
    cdef double exp_value = base_pref * exp(-T)

    memset(zero_ang, 0, 12 * sizeof(int))
    for m in range(max_m, -1, -1):
        key = os_pack_key(zero_ang, m)
        slot = os_hash_slot(key)
        while keys[slot] != 0:
            slot = (slot + 1) & (OS_HASH_CAP - 1)
        keys[slot] = key
        values[slot] = value
        nstates[0] += 1
        if m > 0:
            value = (2.0 * T * value + exp_value) / (2.0 * m - 1.0)


cdef inline size_t os_vrr_idx(
    int ax, int ay, int az,
    int cx, int cy, int cz,
    int m,
    int adim,
    int cdim,
    int mdim,
) noexcept nogil:
    return (
        (((((<size_t>ax * <size_t>adim + <size_t>ay) * <size_t>adim + <size_t>az)
        * <size_t>cdim + <size_t>cx) * <size_t>cdim + <size_t>cy)
        * <size_t>cdim + <size_t>cz) * <size_t>mdim + <size_t>m
    )


cdef inline double os_vrr_get(
    double* table,
    int ax, int ay, int az,
    int cx, int cy, int cz,
    int m,
    int max_a,
    int max_c,
    int max_m,
    int adim,
    int cdim,
    int mdim,
) noexcept nogil:
    if ax < 0 or ay < 0 or az < 0 or cx < 0 or cy < 0 or cz < 0 or m < 0:
        return 0.0
    if ax + ay + az > max_a or cx + cy + cz > max_c or m > max_m:
        return 0.0
    return table[os_vrr_idx(ax, ay, az, cx, cy, cz, m, adim, cdim, mdim)]


cdef inline double os_vrr_get_raw(
    double* table,
    int ax, int ay, int az,
    int cx, int cy, int cz,
    int m,
    int adim,
    int cdim,
    int mdim,
) noexcept nogil:
    return table[os_vrr_idx(ax, ay, az, cx, cy, cz, m, adim, cdim, mdim)]


cdef inline void os_vrr_set(
    double* table,
    int ax, int ay, int az,
    int cx, int cy, int cz,
    int m,
    int adim,
    int cdim,
    int mdim,
    double value,
) noexcept nogil:
    table[os_vrr_idx(ax, ay, az, cx, cy, cz, m, adim, cdim, mdim)] = value


cdef void os_subtract_valid_vrr_states(
    double* table,
    double* other,
    int max_a,
    int max_c,
    int max_m,
    double scale,
) noexcept nogil:
    cdef int adim = max_a + 1
    cdef int cdim = max_c + 1
    cdef int mdim = max_m + 1
    cdef int ax, ay, az, cx, cy, cz, m, total
    cdef size_t index

    for ax in range(max_a + 1):
        for ay in range(max_a + 1 - ax):
            for az in range(max_a + 1 - ax - ay):
                for cx in range(max_c + 1):
                    for cy in range(max_c + 1 - cx):
                        for cz in range(max_c + 1 - cx - cy):
                            total = ax + ay + az + cx + cy + cz
                            if total > max_m:
                                continue
                            for m in range(max_m - total + 1):
                                index = os_vrr_idx(
                                    ax, ay, az, cx, cy, cz, m,
                                    adim, cdim, mdim,
                                )
                                table[index] -= scale * other[index]


cdef void os_fill_vrr_table(
    double* table,
    int max_a,
    int max_c,
    int max_m,
    double p,
    double q,
    double z,
    double T,
    double base_pref,
    double* PA,
    double* QC,
    double* PQ,
) noexcept nogil:
    cdef int adim = max_a + 1
    cdef int cdim = max_c + 1
    cdef int mdim = max_m + 1
    cdef int m, total, ax, ay, az, cx, cy, cz, asum, csum, axis
    cdef double value, exp_value = base_pref * exp(-T)

    value = base_pref * boys_fn(max_m, T)
    for m in range(max_m, -1, -1):
        os_vrr_set(table, 0, 0, 0, 0, 0, 0, m, adim, cdim, mdim, value)
        if m > 0:
            value = (2.0 * T * value + exp_value) / (2.0 * m - 1.0)

    for total in range(1, max_m + 1):
        for ax in range(max_a + 1):
            for ay in range(max_a + 1 - ax):
                for az in range(max_a + 1 - ax - ay):
                    asum = ax + ay + az
                    for cx in range(max_c + 1):
                        for cy in range(max_c + 1 - cx):
                            for cz in range(max_c + 1 - cx - cy):
                                csum = cx + cy + cz
                                if asum + csum != total:
                                    continue
                                for m in range(max_m - total + 1):
                                    if asum > 0:
                                        axis = os_axis_of_max3(ax, ay, az)
                                        if axis == 0:
                                            value = (
                                                PA[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if ax - 1 > 0:
                                                value += (ax - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cx > 0:
                                                value += cx / (2.0 * z) * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                        elif axis == 1:
                                            value = (
                                                PA[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if ay - 1 > 0:
                                                value += (ay - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cy > 0:
                                                value += cy / (2.0 * z) * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                        else:
                                            value = (
                                                PA[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if az - 1 > 0:
                                                value += (az - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cz > 0:
                                                value += cz / (2.0 * z) * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                    else:
                                        axis = os_axis_of_max3(cx, cy, cz)
                                        if axis == 0:
                                            value = (
                                                QC[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m, adim, cdim, mdim)
                                                + p / z * PQ[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if cx - 1 > 0:
                                                value += (cx - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if ax > 0:
                                                value += ax / (2.0 * z) * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                        elif axis == 1:
                                            value = (
                                                QC[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m, adim, cdim, mdim)
                                                + p / z * PQ[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if cy - 1 > 0:
                                                value += (cy - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if ay > 0:
                                                value += ay / (2.0 * z) * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                        else:
                                            value = (
                                                QC[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m, adim, cdim, mdim)
                                                + p / z * PQ[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                            )
                                            if cz - 1 > 0:
                                                value += (cz - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m + 1, adim, cdim, mdim)
                                                )
                                            if az > 0:
                                                value += az / (2.0 * z) * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                    os_vrr_set(table, ax, ay, az, cx, cy, cz, m, adim, cdim, mdim, value)


cdef void os_fill_vrr_table_kernel(
    double* table,
    int max_a,
    int max_c,
    int max_m,
    double p,
    double q,
    double alpha_kernel,
    double T,
    double base_pref,
    double* PA,
    double* QC,
    double* PQ,
) noexcept nogil:
    cdef int adim = max_a + 1
    cdef int cdim = max_c + 1
    cdef int mdim = max_m + 1
    cdef int m, total, ax, ay, az, cx, cy, cz, asum, csum, axis
    cdef double value, exp_value = base_pref * exp(-T)
    cdef double alpha_over_p = alpha_kernel / p
    cdef double alpha_over_q = alpha_kernel / q
    cdef double cross = alpha_kernel / (2.0 * p * q)

    value = base_pref * boys_fn(max_m, T)
    for m in range(max_m, -1, -1):
        os_vrr_set(table, 0, 0, 0, 0, 0, 0, m, adim, cdim, mdim, value)
        if m > 0:
            value = (2.0 * T * value + exp_value) / (2.0 * m - 1.0)

    for total in range(1, max_m + 1):
        for ax in range(max_a + 1):
            for ay in range(max_a + 1 - ax):
                for az in range(max_a + 1 - ax - ay):
                    asum = ax + ay + az
                    for cx in range(max_c + 1):
                        for cy in range(max_c + 1 - cx):
                            for cz in range(max_c + 1 - cx - cy):
                                csum = cx + cy + cz
                                if asum + csum != total:
                                    continue
                                for m in range(max_m - total + 1):
                                    if asum > 0:
                                        axis = os_axis_of_max3(ax, ay, az)
                                        if axis == 0:
                                            value = (
                                                PA[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - alpha_over_p * PQ[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if ax - 1 > 0:
                                                value += (ax - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - alpha_over_p * os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cx > 0:
                                                value += cx * cross * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                        elif axis == 1:
                                            value = (
                                                PA[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - alpha_over_p * PQ[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if ay - 1 > 0:
                                                value += (ay - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - alpha_over_p * os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cy > 0:
                                                value += cy * cross * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                        else:
                                            value = (
                                                PA[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m, adim, cdim, mdim)
                                                - alpha_over_p * PQ[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if az - 1 > 0:
                                                value += (az - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m, adim, cdim, mdim)
                                                    - alpha_over_p * os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if cz > 0:
                                                value += cz * cross * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                    else:
                                        axis = os_axis_of_max3(cx, cy, cz)
                                        if axis == 0:
                                            value = (
                                                QC[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m, adim, cdim, mdim)
                                                + alpha_over_q * PQ[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if cx - 1 > 0:
                                                value += (cx - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m, adim, cdim, mdim)
                                                    - alpha_over_q * os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if ax > 0:
                                                value += ax * cross * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim)
                                        elif axis == 1:
                                            value = (
                                                QC[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m, adim, cdim, mdim)
                                                + alpha_over_q * PQ[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                            )
                                            if cy - 1 > 0:
                                                value += (cy - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m, adim, cdim, mdim)
                                                    - alpha_over_q * os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m + 1, adim, cdim, mdim)
                                                )
                                            if ay > 0:
                                                value += ay * cross * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim)
                                        else:
                                            value = (
                                                QC[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m, adim, cdim, mdim)
                                                + alpha_over_q * PQ[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                            )
                                            if cz - 1 > 0:
                                                value += (cz - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m, adim, cdim, mdim)
                                                    - alpha_over_q * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m + 1, adim, cdim, mdim)
                                                )
                                            if az > 0:
                                                value += az * cross * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim)
                                    os_vrr_set(table, ax, ay, az, cx, cy, cz, m, adim, cdim, mdim, value)


cdef double os_vrr_hrr_eval(
    double* table,
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    int dxx, int dyy, int dzz,
    int m,
    int max_a,
    int max_c,
    int max_m,
    double* AB,
    double* CD,
) noexcept nogil:
    cdef int axis
    cdef int adim = max_a + 1
    cdef int cdim = max_c + 1
    cdef int mdim = max_m + 1
    if bx + by + bz > 0:
        axis = os_axis_of_max3(bx, by, bz)
        if axis == 0:
            return (
                os_vrr_hrr_eval(table, ax + 1, ay, az, bx - 1, by, bz, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
                + AB[0] * os_vrr_hrr_eval(table, ax, ay, az, bx - 1, by, bz, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
            )
        if axis == 1:
            return (
                os_vrr_hrr_eval(table, ax, ay + 1, az, bx, by - 1, bz, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
                + AB[1] * os_vrr_hrr_eval(table, ax, ay, az, bx, by - 1, bz, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
            )
        return (
            os_vrr_hrr_eval(table, ax, ay, az + 1, bx, by, bz - 1, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
            + AB[2] * os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz - 1, cx, cy, cz, dxx, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
        )
    if dxx + dyy + dzz > 0:
        axis = os_axis_of_max3(dxx, dyy, dzz)
        if axis == 0:
            return (
                os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx + 1, cy, cz, dxx - 1, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
                + CD[0] * os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx, cy, cz, dxx - 1, dyy, dzz, m, max_a, max_c, max_m, AB, CD)
            )
        if axis == 1:
            return (
                os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx, cy + 1, cz, dxx, dyy - 1, dzz, m, max_a, max_c, max_m, AB, CD)
                + CD[1] * os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx, cy, cz, dxx, dyy - 1, dzz, m, max_a, max_c, max_m, AB, CD)
            )
        return (
            os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx, cy, cz + 1, dxx, dyy, dzz - 1, m, max_a, max_c, max_m, AB, CD)
            + CD[2] * os_vrr_hrr_eval(table, ax, ay, az, bx, by, bz, cx, cy, cz, dxx, dyy, dzz - 1, m, max_a, max_c, max_m, AB, CD)
        )
    return os_vrr_get(table, ax, ay, az, cx, cy, cz, m, max_a, max_c, max_m, adim, cdim, mdim)


cdef inline double os_pow_small(double x, int n) noexcept nogil:
    if n == 0:
        return 1.0
    if n == 1:
        return x
    if n == 2:
        return x * x
    if n == 3:
        return x * x * x
    return x * x * x * x


cdef inline double os_binom_small(int n, int k) noexcept nogil:
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    if n == 2:
        return 2.0
    if n == 3:
        if k == 1 or k == 2:
            return 3.0
    if n == 4:
        if k == 1 or k == 3:
            return 4.0
        if k == 2:
            return 6.0
    return 1.0


cdef double os_vrr_hrr_eval_expanded(
    double* table,
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    int dxx, int dyy, int dzz,
    int m,
    int max_a,
    int max_c,
    int max_m,
    double* AB,
    double* CD,
) noexcept nogil:
    cdef int ix, iy, iz, jx, jy, jz
    cdef int adim = max_a + 1
    cdef int cdim = max_c + 1
    cdef int mdim = max_m + 1
    cdef double coeff_b, coeff_d, value = 0.0

    for ix in range(bx + 1):
        for iy in range(by + 1):
            for iz in range(bz + 1):
                coeff_b = (
                    os_binom_small(bx, ix) * os_pow_small(AB[0], bx - ix)
                    * os_binom_small(by, iy) * os_pow_small(AB[1], by - iy)
                    * os_binom_small(bz, iz) * os_pow_small(AB[2], bz - iz)
                )
                for jx in range(dxx + 1):
                    for jy in range(dyy + 1):
                        for jz in range(dzz + 1):
                            coeff_d = (
                                os_binom_small(dxx, jx) * os_pow_small(CD[0], dxx - jx)
                                * os_binom_small(dyy, jy) * os_pow_small(CD[1], dyy - jy)
                                * os_binom_small(dzz, jz) * os_pow_small(CD[2], dzz - jz)
                            )
                            value += coeff_b * coeff_d * os_vrr_get_raw(
                                table,
                                ax + ix,
                                ay + iy,
                                az + iz,
                                cx + jx,
                                cy + jy,
                                cz + jz,
                                m,
                                adim,
                                cdim,
                                mdim,
                            )
    return value


cdef inline void os_add_dep(
    int dep_ang[8][12],
    int* dep_m,
    int* ndep,
    int* src,
    int m,
) noexcept nogil:
    cdef int i
    if ndep[0] >= 8:
        return
    for i in range(12):
        dep_ang[ndep[0]][i] = src[i]
    dep_m[ndep[0]] = m
    ndep[0] += 1


cdef inline int os_state_is_negative(int* ang) noexcept nogil:
    cdef int i
    for i in range(12):
        if ang[i] < 0:
            return 1
    return 0


cdef inline int os_find_value(
    uint64_t* keys,
    double* values,
    uint64_t key,
    double* value,
) noexcept nogil:
    cdef int slot = os_hash_slot(key)
    cdef int probe
    for probe in range(OS_HASH_CAP):
        if keys[slot] == 0:
            return 0
        if keys[slot] == key:
            value[0] = values[slot]
            return 1
        slot = (slot + 1) & (OS_HASH_CAP - 1)
    return 0


cdef inline double os_lookup_value(
    int* ang,
    int m,
    uint64_t* keys,
    double* values,
) noexcept nogil:
    cdef uint64_t key
    cdef double value[1]
    if os_state_is_negative(ang):
        return 0.0
    key = os_pack_key(ang, m)
    if os_find_value(keys, values, key, value):
        return value[0]
    return 0.0


cdef inline void os_store_value(
    int* ang,
    int m,
    double value,
    uint64_t* keys,
    double* values,
    int* nstates,
) noexcept nogil:
    cdef uint64_t key = os_pack_key(ang, m)
    cdef int slot = os_hash_slot(key)
    cdef int probe
    for probe in range(OS_HASH_CAP):
        if keys[slot] == 0:
            keys[slot] = key
            values[slot] = value
            nstates[0] += 1
            return
        if keys[slot] == key:
            values[slot] = value
            return
        slot = (slot + 1) & (OS_HASH_CAP - 1)


cdef inline int os_fill_deps(
    int* ang,
    int m,
    int dep_ang[8][12],
    int* dep_m,
) noexcept nogil:
    cdef int axis
    cdef int ndep[1]
    cdef int na = ang[0] + ang[1] + ang[2]
    cdef int nb = ang[3] + ang[4] + ang[5]
    cdef int nc = ang[6] + ang[7] + ang[8]
    cdef int nd = ang[9] + ang[10] + ang[11]
    cdef int prev[12]
    cdef int tmp[12]

    ndep[0] = 0
    if os_state_is_negative(ang):
        return 0
    if na == 0 and nb == 0 and nc == 0 and nd == 0:
        return 0

    if na > 0:
        axis = os_axis_of_max3(ang[0], ang[1], ang[2])
        os_copy_state(prev, ang)
        prev[axis] -= 1
        os_add_dep(dep_ang, dep_m, ndep, prev, m)
        os_add_dep(dep_ang, dep_m, ndep, prev, m + 1)
        if prev[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m)
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m)
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
    elif nc > 0:
        axis = os_axis_of_max3(ang[6], ang[7], ang[8])
        os_copy_state(prev, ang)
        prev[6 + axis] -= 1
        os_add_dep(dep_ang, dep_m, ndep, prev, m)
        os_add_dep(dep_ang, dep_m, ndep, prev, m + 1)
        if prev[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m)
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m)
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            os_add_dep(dep_ang, dep_m, ndep, tmp, m + 1)
    elif nb > 0:
        axis = os_axis_of_max3(ang[3], ang[4], ang[5])
        os_copy_state(prev, ang)
        prev[3 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[axis] += 1
        os_add_dep(dep_ang, dep_m, ndep, tmp, m)
        os_add_dep(dep_ang, dep_m, ndep, prev, m)
    elif nd > 0:
        axis = os_axis_of_max3(ang[9], ang[10], ang[11])
        os_copy_state(prev, ang)
        prev[9 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[6 + axis] += 1
        os_add_dep(dep_ang, dep_m, ndep, tmp, m)
        os_add_dep(dep_ang, dep_m, ndep, prev, m)

    return ndep[0]


cdef inline double os_compute_from_table(
    int* ang,
    int m,
    double p, double q, double z, double T, double base_pref,
    double* PA,
    double* QC,
    double* PQ,
    double* AB,
    double* CD,
    uint64_t* keys,
    double* values,
) noexcept nogil:
    cdef int axis
    cdef int na = ang[0] + ang[1] + ang[2]
    cdef int nb = ang[3] + ang[4] + ang[5]
    cdef int nc = ang[6] + ang[7] + ang[8]
    cdef int nd = ang[9] + ang[10] + ang[11]
    cdef int prev[12]
    cdef int tmp[12]
    cdef double value

    if os_state_is_negative(ang):
        return 0.0
    if na == 0 and nb == 0 and nc == 0 and nd == 0:
        return base_pref * boys_fn(m, T)

    if na > 0:
        axis = os_axis_of_max3(ang[0], ang[1], ang[2])
        os_copy_state(prev, ang)
        prev[axis] -= 1
        value = (
            PA[axis] * os_lookup_value(prev, m, keys, values)
            - q / z * PQ[axis] * os_lookup_value(prev, m + 1, keys, values)
        )
        if prev[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            value += prev[axis] / (2.0 * p) * (
                os_lookup_value(tmp, m, keys, values)
                - q / z * os_lookup_value(tmp, m + 1, keys, values)
            )
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            value += ang[3 + axis] / (2.0 * p) * (
                os_lookup_value(tmp, m, keys, values)
                - q / z * os_lookup_value(tmp, m + 1, keys, values)
            )
        if ang[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            value += ang[6 + axis] / (2.0 * z) * os_lookup_value(tmp, m + 1, keys, values)
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            value += ang[9 + axis] / (2.0 * z) * os_lookup_value(tmp, m + 1, keys, values)
    elif nc > 0:
        axis = os_axis_of_max3(ang[6], ang[7], ang[8])
        os_copy_state(prev, ang)
        prev[6 + axis] -= 1
        value = (
            QC[axis] * os_lookup_value(prev, m, keys, values)
            + p / z * PQ[axis] * os_lookup_value(prev, m + 1, keys, values)
        )
        if prev[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            value += prev[6 + axis] / (2.0 * q) * (
                os_lookup_value(tmp, m, keys, values)
                - p / z * os_lookup_value(tmp, m + 1, keys, values)
            )
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            value += ang[9 + axis] / (2.0 * q) * (
                os_lookup_value(tmp, m, keys, values)
                - p / z * os_lookup_value(tmp, m + 1, keys, values)
            )
        if ang[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            value += ang[axis] / (2.0 * z) * os_lookup_value(tmp, m + 1, keys, values)
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            value += ang[3 + axis] / (2.0 * z) * os_lookup_value(tmp, m + 1, keys, values)
    elif nb > 0:
        axis = os_axis_of_max3(ang[3], ang[4], ang[5])
        os_copy_state(prev, ang)
        prev[3 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[axis] += 1
        value = os_lookup_value(tmp, m, keys, values) + AB[axis] * os_lookup_value(prev, m, keys, values)
    else:
        axis = os_axis_of_max3(ang[9], ang[10], ang[11])
        os_copy_state(prev, ang)
        prev[9 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[6 + axis] += 1
        value = os_lookup_value(tmp, m, keys, values) + CD[axis] * os_lookup_value(prev, m, keys, values)

    return value


cdef double os_eri_eval_iterative(
    int* target,
    int m,
    double p, double q, double z, double T, double base_pref,
    double* PA,
    double* QC,
    double* PQ,
    double* AB,
    double* CD,
    uint64_t* keys,
    double* values,
    int* nstates,
    int* stack_ang,
    int* stack_m,
) noexcept nogil:
    cdef int sp = 1
    cdef int i, d, ndep, missing
    cdef int* cur
    cdef int* dep
    cdef uint64_t key
    cdef double value[1]
    cdef int dep_ang[8][12]
    cdef int dep_m[8]

    for i in range(12):
        stack_ang[i] = target[i]
    stack_m[0] = m

    while sp > 0:
        cur = stack_ang + (sp - 1) * 12
        if os_state_is_negative(cur):
            sp -= 1
            continue

        key = os_pack_key(cur, stack_m[sp - 1])
        if os_find_value(keys, values, key, value):
            sp -= 1
            continue

        ndep = os_fill_deps(cur, stack_m[sp - 1], dep_ang, dep_m)
        missing = 0
        for d in range(ndep):
            dep = &dep_ang[d][0]
            if os_state_is_negative(dep):
                continue
            key = os_pack_key(dep, dep_m[d])
            if not os_find_value(keys, values, key, value):
                if sp >= OS_HASH_CAP:
                    return os_eri_rec_generic(
                        target, m, p, q, z, T, base_pref,
                        PA, QC, PQ, AB, CD, keys, values, nstates,
                    )
                for i in range(12):
                    stack_ang[sp * 12 + i] = dep[i]
                stack_m[sp] = dep_m[d]
                sp += 1
                missing = 1
                break
        if missing:
            continue

        value[0] = os_compute_from_table(
            cur, stack_m[sp - 1], p, q, z, T, base_pref,
            PA, QC, PQ, AB, CD, keys, values,
        )
        os_store_value(cur, stack_m[sp - 1], value[0], keys, values, nstates)
        sp -= 1

    return os_lookup_value(target, m, keys, values)


cdef double os_eri_rec_generic(
    int* ang,
    int m,
    double p, double q, double z, double T, double base_pref,
    double* PA,
    double* QC,
    double* PQ,
    double* AB,
    double* CD,
    uint64_t* keys,
    double* values,
    int* nstates,
) noexcept nogil:
    cdef int i, axis, slot, probe
    cdef int na = ang[0] + ang[1] + ang[2]
    cdef int nb = ang[3] + ang[4] + ang[5]
    cdef int nc = ang[6] + ang[7] + ang[8]
    cdef int nd = ang[9] + ang[10] + ang[11]
    cdef int prev[12]
    cdef int tmp[12]
    cdef double value
    cdef uint64_t key

    for i in range(12):
        if ang[i] < 0:
            return 0.0

    key = os_pack_key(ang, m)
    slot = os_hash_slot(key)
    for probe in range(OS_HASH_CAP):
        if keys[slot] == 0:
            break
        if keys[slot] == key:
            return values[slot]
        slot = (slot + 1) & (OS_HASH_CAP - 1)

    if na == 0 and nb == 0 and nc == 0 and nd == 0:
        value = base_pref * boys_fn(m, T)
    elif na > 0:
        axis = os_axis_of_max3(ang[0], ang[1], ang[2])
        os_copy_state(prev, ang)
        prev[axis] -= 1
        value = (
            PA[axis] * os_eri_rec_generic(prev, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            - q / z * PQ[axis] * os_eri_rec_generic(prev, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        )
        if prev[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            value += prev[axis] / (2.0 * p) * (
                os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
                - q / z * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            )
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            value += ang[3 + axis] / (2.0 * p) * (
                os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
                - q / z * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            )
        if ang[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            value += ang[6 + axis] / (2.0 * z) * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            value += ang[9 + axis] / (2.0 * z) * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
    elif nc > 0:
        axis = os_axis_of_max3(ang[6], ang[7], ang[8])
        os_copy_state(prev, ang)
        prev[6 + axis] -= 1
        value = (
            QC[axis] * os_eri_rec_generic(prev, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            + p / z * PQ[axis] * os_eri_rec_generic(prev, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        )
        if prev[6 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[6 + axis] -= 1
            value += prev[6 + axis] / (2.0 * q) * (
                os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
                - p / z * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            )
        if ang[9 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[9 + axis] -= 1
            value += ang[9 + axis] / (2.0 * q) * (
                os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
                - p / z * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            )
        if ang[axis] > 0:
            os_copy_state(tmp, prev)
            tmp[axis] -= 1
            value += ang[axis] / (2.0 * z) * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        if ang[3 + axis] > 0:
            os_copy_state(tmp, prev)
            tmp[3 + axis] -= 1
            value += ang[3 + axis] / (2.0 * z) * os_eri_rec_generic(tmp, m + 1, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
    elif nb > 0:
        axis = os_axis_of_max3(ang[3], ang[4], ang[5])
        os_copy_state(prev, ang)
        prev[3 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[axis] += 1
        value = (
            os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            + AB[axis] * os_eri_rec_generic(prev, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        )
    else:
        axis = os_axis_of_max3(ang[9], ang[10], ang[11])
        os_copy_state(prev, ang)
        prev[9 + axis] -= 1
        os_copy_state(tmp, prev)
        tmp[6 + axis] += 1
        value = (
            os_eri_rec_generic(tmp, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
            + CD[axis] * os_eri_rec_generic(prev, m, p, q, z, T, base_pref, PA, QC, PQ, AB, CD, keys, values, nstates)
        )

    slot = os_hash_slot(key)
    for probe in range(OS_HASH_CAP):
        if keys[slot] == 0:
            keys[slot] = key
            values[slot] = value
            nstates[0] += 1
            break
        if keys[slot] == key:
            values[slot] = value
            break
        slot = (slot + 1) & (OS_HASH_CAP - 1)
    return value




cdef inline int precompute_primitive_pair_data(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    double* a_out,
    double* b_out,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out,
    double* w_out,
) noexcept nogil:
    cdef int ip, iq, idx = 0
    cdef double a, b, pexp
    cdef double Ax = origins_v[p, 0]
    cdef double Ay = origins_v[p, 1]
    cdef double Az = origins_v[p, 2]
    cdef double Bx = origins_v[q, 0]
    cdef double By = origins_v[q, 1]
    cdef double Bz = origins_v[q, 2]

    for ip in range(nprim_v[p]):
        a = exps_v[p, ip]
        for iq in range(nprim_v[q]):
            b = exps_v[q, iq]
            pexp = a + b
            a_out[idx] = a
            b_out[idx] = b
            p_out[idx] = pexp
            px_out[idx] = (a * Ax + b * Bx) / pexp
            py_out[idx] = (a * Ay + b * By) / pexp
            pz_out[idx] = (a * Az + b * Bz) / pexp
            w_out[idx] = weights_v[p, ip] * weights_v[q, iq]
            idx += 1
    return idx


cdef inline int precompute_primitive_pair_geom(
    int p, int q,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    int64_t[::1] nprim_v,
    double* a_out,
    double* b_out,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out,
) noexcept nogil:
    cdef int ip, iq, idx = 0
    cdef double a, b, pexp
    cdef double Ax = origins_v[p, 0]
    cdef double Ay = origins_v[p, 1]
    cdef double Az = origins_v[p, 2]
    cdef double Bx = origins_v[q, 0]
    cdef double By = origins_v[q, 1]
    cdef double Bz = origins_v[q, 2]

    for ip in range(nprim_v[p]):
        a = exps_v[p, ip]
        for iq in range(nprim_v[q]):
            b = exps_v[q, iq]
            pexp = a + b
            a_out[idx] = a
            b_out[idx] = b
            p_out[idx] = pexp
            px_out[idx] = (a * Ax + b * Bx) / pexp
            py_out[idx] = (a * Ay + b * By) / pexp
            pz_out[idx] = (a * Az + b * Bz) / pexp
            idx += 1
    return idx


cdef inline int precompute_primitive_pair_geom_asymmetric(
    int p, int q,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] exps_v,
    int64_t[::1] nprim_v,
    double* a_out,
    double* b_out,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out,
) noexcept nogil:
    cdef int ip, iq, idx = 0
    cdef double a, b, pexp
    cdef double Ax = left_origins_v[p, 0]
    cdef double Ay = left_origins_v[p, 1]
    cdef double Az = left_origins_v[p, 2]
    cdef double Bx = right_origins_v[q, 0]
    cdef double By = right_origins_v[q, 1]
    cdef double Bz = right_origins_v[q, 2]

    for ip in range(nprim_v[p]):
        a = exps_v[p, ip]
        for iq in range(nprim_v[q]):
            b = exps_v[q, iq]
            pexp = a + b
            a_out[idx] = a
            b_out[idx] = b
            p_out[idx] = pexp
            px_out[idx] = (a * Ax + b * Bx) / pexp
            py_out[idx] = (a * Ay + b * By) / pexp
            pz_out[idx] = (a * Az + b * Bz) / pexp
            idx += 1
    return idx


cdef inline double contracted_eri_indices(
    int p, int q, int r, int s,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
) noexcept nogil:
    cdef int ip, iq, ir, is_
    cdef double value = 0.0

    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            for ir in range(nprim_v[r]):
                for is_ in range(nprim_v[s]):
                    value += (
                        weights_v[p, ip] * weights_v[q, iq] * weights_v[r, ir] * weights_v[s, is_]
                        * primitive_eri(
                            exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                            exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                            exps_v[r, ir], <int>shells_v[r, 0], <int>shells_v[r, 1], <int>shells_v[r, 2], origins_v[r, 0], origins_v[r, 1], origins_v[r, 2],
                            exps_v[s, is_], <int>shells_v[s, 0], <int>shells_v[s, 1], <int>shells_v[s, 2], origins_v[s, 0], origins_v[s, 1], origins_v[s, 2],
                        )
                    )
    return value


cdef inline double contracted_overlap_indices(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
) noexcept nogil:
    cdef int ip, iq
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            value += (
                weights_v[p, ip] * weights_v[q, iq]
                * primitive_overlap(
                    exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                    exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                )
            )
    return value


cdef inline double contracted_kinetic_indices(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
) noexcept nogil:
    cdef int ip, iq
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            value += (
                weights_v[p, ip] * weights_v[q, iq]
                * primitive_kinetic(
                    exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                    exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                )
            )
    return value


cdef inline double contracted_nuclear_indices(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    double[:, ::1] atcoords_v,
    double[::1] atnums_v,
) noexcept nogil:
    cdef int ip, iq, c
    cdef double value = 0.0
    for c in range(atnums_v.shape[0]):
        for ip in range(nprim_v[p]):
            for iq in range(nprim_v[q]):
                value -= (
                    atnums_v[c]
                    * weights_v[p, ip] * weights_v[q, iq]
                    * primitive_nuclear_attraction(
                        exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                        exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                        atcoords_v[c, 0], atcoords_v[c, 1], atcoords_v[c, 2],
                    )
                )
    return value


cdef inline double primitive_nuclear_from_hermite(
    double* coeff_x, int tx_max,
    double* coeff_y, int uy_max,
    double* coeff_z, int vz_max,
    double pair_p, double radial_p,
    double px, double py, double pz,
    double Cx, double Cy, double Cz,
    double* memo_r, size_t size_r, int nmax,
) noexcept nogil:
    cdef int t, u, v
    cdef size_t idx
    cdef double dx = px - Cx
    cdef double dy = py - Cy
    cdef double dz = pz - Cz
    cdef double rpc = sqrt(dx * dx + dy * dy + dz * dz)
    cdef double value = 0.0
    for idx in range(size_r):
        memo_r[idx] = NAN
    for t in range(tx_max + 1):
        for u in range(uy_max + 1):
            for v in range(vz_max + 1):
                value += coeff_x[t] * coeff_y[u] * coeff_z[v] * R_rec(
                    t, u, v, 0, radial_p,
                    dx, dy, dz, rpc, memo_r,
                    uy_max + 1, vz_max + 1, nmax,
                )
    return value * (2.0 * PI / pair_p)


cdef inline double primitive_periodic_point_charge_precomputed(
    double* coeff_x, int tx_max,
    double* coeff_y, int uy_max,
    double* coeff_z, int vz_max,
    double px, double py, double pz,
    double Cx, double Cy, double Cz,
    double charge, double weight, double pair_p, double radial_p,
    double lr_scale, int angular, double pair_log_prefactor,
    double nuclear_screen_tol, double log_tol,
    double* memo_r, size_t size_r, int nmax,
) noexcept nogil:
    cdef double dx, dy, dz, distance2, distance, log_bound
    cdef double full_value, lr_value
    if nuclear_screen_tol > 0.0 and lr_scale > 0.0:
        dx = px - Cx
        dy = py - Cy
        dz = pz - Cz
        distance2 = dx * dx + dy * dy + dz * dz
        distance = sqrt(distance2)
        log_bound = (
            log(fabs(charge) + 1.0e-300)
            + pair_log_prefactor
            - radial_p * distance2
            + (angular + 2.0)
            * log(2.0 + distance + 1.0 / sqrt(radial_p))
        )
        if log_bound < log_tol:
            return 0.0
    full_value = primitive_nuclear_from_hermite(
        coeff_x, tx_max, coeff_y, uy_max, coeff_z, vz_max,
        pair_p, pair_p, px, py, pz, Cx, Cy, Cz,
        memo_r, size_r, nmax,
    )
    if lr_scale > 0.0:
        lr_value = primitive_nuclear_from_hermite(
            coeff_x, tx_max, coeff_y, uy_max, coeff_z, vz_max,
            pair_p, radial_p, px, py, pz, Cx, Cy, Cz,
            memo_r, size_r, nmax,
        )
    else:
        lr_value = 0.0
    return -charge * weight * (full_value - lr_scale * lr_value)


cdef inline void contracted_periodic_one_electron_indices(
    int p, int q,
    int64_t[:, ::1] left_shells_v,
    int64_t[:, ::1] right_shells_v,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] left_exps_v,
    double[:, ::1] right_exps_v,
    double[:, ::1] left_weights_v,
    double[:, ::1] right_weights_v,
    int64_t[::1] left_nprim_v,
    int64_t[::1] right_nprim_v,
    double[:, ::1] atcoords_v,
    double[::1] atnums_v,
    double[:, ::1] lattice_v,
    double[:, ::1] inverse_lattice_v,
    int64_t[:, ::1] nuclear_image_keys_v,
    bint use_lattice_images,
    double eta,
    double nuclear_screen_tol,
    double* overlap,
    double* kinetic,
    double* vnuc,
) noexcept nogil:
    cdef int ip, iq, c, image, idx, t, u, v
    cdef int angular, l1, m1, n1, l2, m2, n2
    cdef int tx_max, uy_max, vz_max, nmax
    cdef int size_x, size_y, size_z
    cdef size_t size_r
    cdef double alpha, beta, weight, pair_p, radial_p, lr_scale
    cdef double abx, aby, abz, ab2, pair_q, px, py, pz, log_tol
    cdef double pair_log_prefactor, fx, fy, fz, cx, cy, cz
    cdef int64_t nx, ny, nz
    cdef double* memo_x
    cdef double* memo_y
    cdef double* memo_z
    cdef double* memo_r
    cdef double* coeff_x
    cdef double* coeff_y
    cdef double* coeff_z
    overlap[0] = 0.0
    kinetic[0] = 0.0
    vnuc[0] = 0.0
    log_tol = log(nuclear_screen_tol) if nuclear_screen_tol > 0.0 else 0.0
    l1 = <int>left_shells_v[p, 0]
    m1 = <int>left_shells_v[p, 1]
    n1 = <int>left_shells_v[p, 2]
    l2 = <int>right_shells_v[q, 0]
    m2 = <int>right_shells_v[q, 1]
    n2 = <int>right_shells_v[q, 2]
    tx_max = l1 + l2
    uy_max = m1 + m2
    vz_max = n1 + n2
    angular = tx_max + uy_max + vz_max
    nmax = angular + 2
    size_x = (l1 + 1) * (l2 + 1) * (tx_max + 2)
    size_y = (m1 + 1) * (m2 + 1) * (uy_max + 2)
    size_z = (n1 + 1) * (n2 + 1) * (vz_max + 2)
    size_r = (
        <size_t>(tx_max + 1) * <size_t>(uy_max + 1)
        * <size_t>(vz_max + 1) * <size_t>nmax
    )
    memo_x = <double*>malloc(size_x * sizeof(double))
    memo_y = <double*>malloc(size_y * sizeof(double))
    memo_z = <double*>malloc(size_z * sizeof(double))
    memo_r = <double*>malloc(size_r * sizeof(double))
    coeff_x = <double*>malloc((tx_max + 1) * sizeof(double))
    coeff_y = <double*>malloc((uy_max + 1) * sizeof(double))
    coeff_z = <double*>malloc((vz_max + 1) * sizeof(double))
    if (
        memo_x == NULL or memo_y == NULL or memo_z == NULL or memo_r == NULL
        or coeff_x == NULL or coeff_y == NULL or coeff_z == NULL
    ):
        free(memo_x); free(memo_y); free(memo_z); free(memo_r)
        free(coeff_x); free(coeff_y); free(coeff_z)
        return
    abx = left_origins_v[p, 0] - right_origins_v[q, 0]
    aby = left_origins_v[p, 1] - right_origins_v[q, 1]
    abz = left_origins_v[p, 2] - right_origins_v[q, 2]
    ab2 = abx * abx + aby * aby + abz * abz
    for ip in range(left_nprim_v[p]):
        alpha = left_exps_v[p, ip]
        for iq in range(right_nprim_v[q]):
            beta = right_exps_v[q, iq]
            weight = left_weights_v[p, ip] * right_weights_v[q, iq]
            overlap[0] += weight * primitive_overlap(
                alpha,
                <int>left_shells_v[p, 0], <int>left_shells_v[p, 1], <int>left_shells_v[p, 2],
                left_origins_v[p, 0], left_origins_v[p, 1], left_origins_v[p, 2],
                beta,
                <int>right_shells_v[q, 0], <int>right_shells_v[q, 1], <int>right_shells_v[q, 2],
                right_origins_v[q, 0], right_origins_v[q, 1], right_origins_v[q, 2],
            )
            kinetic[0] += weight * primitive_kinetic(
                alpha,
                <int>left_shells_v[p, 0], <int>left_shells_v[p, 1], <int>left_shells_v[p, 2],
                left_origins_v[p, 0], left_origins_v[p, 1], left_origins_v[p, 2],
                beta,
                <int>right_shells_v[q, 0], <int>right_shells_v[q, 1], <int>right_shells_v[q, 2],
                right_origins_v[q, 0], right_origins_v[q, 1], right_origins_v[q, 2],
            )
            pair_p = alpha + beta
            pair_q = alpha * beta / pair_p
            for idx in range(size_x):
                memo_x[idx] = NAN
            for idx in range(size_y):
                memo_y[idx] = NAN
            for idx in range(size_z):
                memo_z[idx] = NAN
            for t in range(tx_max + 1):
                coeff_x[t] = E_rec(
                    l1, l2, t, abx, alpha, beta,
                    memo_x, l2 + 1, tx_max + 2,
                )
            for u in range(uy_max + 1):
                coeff_y[u] = E_rec(
                    m1, m2, u, aby, alpha, beta,
                    memo_y, m2 + 1, uy_max + 2,
                )
            for v in range(vz_max + 1):
                coeff_z[v] = E_rec(
                    n1, n2, v, abz, alpha, beta,
                    memo_z, n2 + 1, vz_max + 2,
                )
            pair_log_prefactor = (
                log(fabs(weight) + 1.0e-300)
                + log(2.0 * PI / pair_p)
                - pair_q * ab2
            )
            px = (
                alpha * left_origins_v[p, 0]
                + beta * right_origins_v[q, 0]
            ) / pair_p
            py = (
                alpha * left_origins_v[p, 1]
                + beta * right_origins_v[q, 1]
            ) / pair_p
            pz = (
                alpha * left_origins_v[p, 2]
                + beta * right_origins_v[q, 2]
            ) / pair_p
            if eta > 0.0:
                radial_p = pair_p * eta * eta / (pair_p + eta * eta)
                lr_scale = eta / sqrt(pair_p + eta * eta)
            else:
                radial_p = pair_p
                lr_scale = 0.0
            if use_lattice_images:
                for c in range(atnums_v.shape[0]):
                    fx = (
                        (px - atcoords_v[c, 0]) * inverse_lattice_v[0, 0]
                        + (py - atcoords_v[c, 1]) * inverse_lattice_v[1, 0]
                        + (pz - atcoords_v[c, 2]) * inverse_lattice_v[2, 0]
                    )
                    fy = (
                        (px - atcoords_v[c, 0]) * inverse_lattice_v[0, 1]
                        + (py - atcoords_v[c, 1]) * inverse_lattice_v[1, 1]
                        + (pz - atcoords_v[c, 2]) * inverse_lattice_v[2, 1]
                    )
                    fz = (
                        (px - atcoords_v[c, 0]) * inverse_lattice_v[0, 2]
                        + (py - atcoords_v[c, 1]) * inverse_lattice_v[1, 2]
                        + (pz - atcoords_v[c, 2]) * inverse_lattice_v[2, 2]
                    )
                    nx = <int64_t>floor(fx + 0.5)
                    ny = <int64_t>floor(fy + 0.5)
                    nz = <int64_t>floor(fz + 0.5)
                    for image in range(nuclear_image_keys_v.shape[0]):
                        cx = atcoords_v[c, 0] + (
                            (nx + nuclear_image_keys_v[image, 0]) * lattice_v[0, 0]
                            + (ny + nuclear_image_keys_v[image, 1]) * lattice_v[1, 0]
                            + (nz + nuclear_image_keys_v[image, 2]) * lattice_v[2, 0]
                        )
                        cy = atcoords_v[c, 1] + (
                            (nx + nuclear_image_keys_v[image, 0]) * lattice_v[0, 1]
                            + (ny + nuclear_image_keys_v[image, 1]) * lattice_v[1, 1]
                            + (nz + nuclear_image_keys_v[image, 2]) * lattice_v[2, 1]
                        )
                        cz = atcoords_v[c, 2] + (
                            (nx + nuclear_image_keys_v[image, 0]) * lattice_v[0, 2]
                            + (ny + nuclear_image_keys_v[image, 1]) * lattice_v[1, 2]
                            + (nz + nuclear_image_keys_v[image, 2]) * lattice_v[2, 2]
                        )
                        vnuc[0] += primitive_periodic_point_charge_precomputed(
                            coeff_x, tx_max, coeff_y, uy_max, coeff_z, vz_max,
                            px, py, pz,
                            cx, cy, cz, atnums_v[c], weight, pair_p, radial_p,
                            lr_scale, angular, pair_log_prefactor,
                            nuclear_screen_tol, log_tol,
                            memo_r, size_r, nmax,
                        )
            else:
                for c in range(atnums_v.shape[0]):
                    vnuc[0] += primitive_periodic_point_charge_precomputed(
                        coeff_x, tx_max, coeff_y, uy_max, coeff_z, vz_max,
                        px, py, pz,
                        atcoords_v[c, 0], atcoords_v[c, 1], atcoords_v[c, 2],
                        atnums_v[c], weight, pair_p, radial_p, lr_scale, angular,
                        pair_log_prefactor, nuclear_screen_tol, log_tol,
                        memo_r, size_r, nmax,
                    )
    free(memo_x); free(memo_y); free(memo_z); free(memo_r)
    free(coeff_x); free(coeff_y); free(coeff_z)


cdef inline double contracted_two_center_coulomb_indices(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
) noexcept nogil:
    cdef int ip, iq
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            value += (
                weights_v[p, ip] * weights_v[q, iq]
                * primitive_two_center_coulomb(
                    exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                    exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                )
            )
    return value


cdef inline double contracted_three_center_indices(
    int p, int q, int a,
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    int64_t[:, ::1] aux_shells_v,
    double[:, ::1] aux_origins_v,
    double[:, ::1] aux_exps_v,
    double[:, ::1] aux_weights_v,
    int64_t[::1] aux_nprim_v,
) noexcept nogil:
    cdef int ip, iq, ia
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            for ia in range(aux_nprim_v[a]):
                value += (
                    weights_v[p, ip] * weights_v[q, iq] * aux_weights_v[a, ia]
                    * primitive_three_center_coulomb(
                        exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], origins_v[p, 0], origins_v[p, 1], origins_v[p, 2],
                        exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], origins_v[q, 0], origins_v[q, 1], origins_v[q, 2],
                        aux_exps_v[a, ia], <int>aux_shells_v[a, 0], <int>aux_shells_v[a, 1], <int>aux_shells_v[a, 2], aux_origins_v[a, 0], aux_origins_v[a, 1], aux_origins_v[a, 2],
                    )
                )
    return value


cdef inline double contracted_short_range_two_center_indices(
    int p, int q,
    int64_t[:, ::1] shells_v,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    double eta,
) noexcept nogil:
    cdef int ip, iq
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            value += (
                weights_v[p, ip] * weights_v[q, iq]
                * primitive_short_range_two_center_coulomb(
                    exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], left_origins_v[p, 0], left_origins_v[p, 1], left_origins_v[p, 2],
                    exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], right_origins_v[q, 0], right_origins_v[q, 1], right_origins_v[q, 2],
                    eta,
                )
            )
    return value


cdef inline double contracted_short_range_three_center_indices(
    int p, int q, int a,
    int64_t[:, ::1] shells_v,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    int64_t[:, ::1] aux_shells_v,
    double[:, ::1] aux_origins_v,
    double[:, ::1] aux_exps_v,
    double[:, ::1] aux_weights_v,
    int64_t[::1] aux_nprim_v,
    double eta,
) noexcept nogil:
    cdef int ip, iq, ia
    cdef double value = 0.0
    for ip in range(nprim_v[p]):
        for iq in range(nprim_v[q]):
            for ia in range(aux_nprim_v[a]):
                value += (
                    weights_v[p, ip] * weights_v[q, iq] * aux_weights_v[a, ia]
                    * primitive_short_range_three_center_coulomb(
                        exps_v[p, ip], <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2], left_origins_v[p, 0], left_origins_v[p, 1], left_origins_v[p, 2],
                        exps_v[q, iq], <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2], right_origins_v[q, 0], right_origins_v[q, 1], right_origins_v[q, 2],
                        aux_exps_v[a, ia], <int>aux_shells_v[a, 0], <int>aux_shells_v[a, 1], <int>aux_shells_v[a, 2], aux_origins_v[a, 0], aux_origins_v[a, 1], aux_origins_v[a, 2],
                        eta,
                    )
                )
    return value


cpdef compute_short_range_aux_metric(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    double eta,
):
    cdef int naux = shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] metric
    cdef int p, q
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] metric_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if right_origins.shape[0] != naux:
        raise ValueError("right_origins must have one row per auxiliary function.")

    metric = np.zeros((naux, naux), dtype=np.float64)
    metric_v = metric
    with nogil:
        for p in range(naux):
            for q in range(naux):
                metric_v[p, q] = contracted_short_range_two_center_indices(
                    p, q,
                    shells_v,
                    left_origins_v,
                    right_origins_v,
                    exps_v,
                    weights_v,
                    nprim_v,
                    eta,
                )
    return metric


cpdef compute_short_range_aux_metric_masked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[uint8_t, ndim=2] pair_mask,
    double eta,
):
    cdef int naux = shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] metric
    cdef int p, q
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef uint8_t[:, ::1] pair_mask_v = pair_mask
    cdef double[:, ::1] metric_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if right_origins.shape[0] != naux:
        raise ValueError("right_origins must have one row per auxiliary function.")
    if pair_mask.shape[0] != naux or pair_mask.shape[1] != naux:
        raise ValueError("pair_mask must have shape (naux, naux).")

    metric = np.zeros((naux, naux), dtype=np.float64)
    metric_v = metric
    with nogil:
        for p in range(naux):
            for q in range(naux):
                if pair_mask_v[p, q] == 0:
                    continue
                metric_v[p, q] = contracted_short_range_two_center_indices(
                    p, q,
                    shells_v,
                    left_origins_v,
                    right_origins_v,
                    exps_v,
                    weights_v,
                    nprim_v,
                    eta,
                )
    return metric


cpdef compute_short_range_three_center_tensor(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    double eta,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef cnp.ndarray[double, ndim=3] tensor
    cdef int p, q, a
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, :, ::1] tensor_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if left_origins.shape[0] != nao or right_origins.shape[0] != nao:
        raise ValueError("left_origins and right_origins must have one row per AO function.")

    tensor = np.zeros((naux, nao, nao), dtype=np.float64)
    tensor_v = tensor
    with nogil:
        for a in range(naux):
            for p in range(nao):
                for q in range(nao):
                    tensor_v[a, p, q] = contracted_short_range_three_center_indices(
                        p, q, a,
                        shells_v,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        eta,
                    )
    return tensor


cpdef compute_short_range_three_center_tensor_masked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    cnp.ndarray[uint8_t, ndim=2] pair_mask,
    cnp.ndarray[uint8_t, ndim=3] aux_pair_mask,
    double eta,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef cnp.ndarray[double, ndim=3] tensor
    cdef int p, q, a
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef uint8_t[:, ::1] pair_mask_v = pair_mask
    cdef uint8_t[:, :, ::1] aux_pair_mask_v = aux_pair_mask
    cdef double[:, :, ::1] tensor_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if left_origins.shape[0] != nao or right_origins.shape[0] != nao:
        raise ValueError("left_origins and right_origins must have one row per AO function.")
    if pair_mask.shape[0] != nao or pair_mask.shape[1] != nao:
        raise ValueError("pair_mask must have shape (nao, nao).")
    if aux_pair_mask.shape[0] != naux or aux_pair_mask.shape[1] != nao or aux_pair_mask.shape[2] != nao:
        raise ValueError("aux_pair_mask must have shape (naux, nao, nao).")

    tensor = np.zeros((naux, nao, nao), dtype=np.float64)
    tensor_v = tensor
    with nogil:
        for a in range(naux):
            for p in range(nao):
                for q in range(nao):
                    if pair_mask_v[p, q] == 0 or aux_pair_mask_v[a, p, q] == 0:
                        continue
                    tensor_v[a, p, q] = contracted_short_range_three_center_indices(
                        p, q, a,
                        shells_v,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        eta,
                    )
    return tensor


cpdef compute_short_range_three_center_tensor_pair_outer_masked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    cnp.ndarray[uint8_t, ndim=2] pair_mask,
    cnp.ndarray[uint8_t, ndim=3] aux_pair_mask,
    double eta,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int max_pair_prims = 0
    cdef int p, q, ip, iq, ia, iaux, ipair, npair
    cdef double ao_a, ao_b, ao_p, value
    cdef double* pair_a = NULL
    cdef double* pair_b = NULL
    cdef double* pair_p = NULL
    cdef double* pair_px = NULL
    cdef double* pair_py = NULL
    cdef double* pair_pz = NULL
    cdef double* pair_abx = NULL
    cdef double* pair_aby = NULL
    cdef double* pair_abz = NULL
    cdef double* pair_weight = NULL
    cdef cnp.ndarray[double, ndim=3] tensor
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef uint8_t[:, ::1] pair_mask_v = pair_mask
    cdef uint8_t[:, :, ::1] aux_pair_mask_v = aux_pair_mask
    cdef double[:, :, ::1] tensor_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if left_origins.shape[0] != nao or right_origins.shape[0] != nao:
        raise ValueError("left_origins and right_origins must have one row per AO function.")
    if pair_mask.shape[0] != nao or pair_mask.shape[1] != nao:
        raise ValueError("pair_mask must have shape (nao, nao).")
    if aux_pair_mask.shape[0] != naux or aux_pair_mask.shape[1] != nao or aux_pair_mask.shape[2] != nao:
        raise ValueError("aux_pair_mask must have shape (naux, nao, nao).")

    for p in range(nao):
        for q in range(nao):
            max_pair_prims = max(max_pair_prims, <int>(nprim[p] * nprim[q]))
    max_pair_prims = max(max_pair_prims, 1)

    pair_a = <double*>malloc(max_pair_prims * sizeof(double))
    pair_b = <double*>malloc(max_pair_prims * sizeof(double))
    pair_p = <double*>malloc(max_pair_prims * sizeof(double))
    pair_px = <double*>malloc(max_pair_prims * sizeof(double))
    pair_py = <double*>malloc(max_pair_prims * sizeof(double))
    pair_pz = <double*>malloc(max_pair_prims * sizeof(double))
    pair_abx = <double*>malloc(max_pair_prims * sizeof(double))
    pair_aby = <double*>malloc(max_pair_prims * sizeof(double))
    pair_abz = <double*>malloc(max_pair_prims * sizeof(double))
    pair_weight = <double*>malloc(max_pair_prims * sizeof(double))
    if (
        pair_a == NULL or pair_b == NULL or pair_p == NULL
        or pair_px == NULL or pair_py == NULL or pair_pz == NULL
        or pair_abx == NULL or pair_aby == NULL or pair_abz == NULL
        or pair_weight == NULL
    ):
        free(pair_a); free(pair_b); free(pair_p); free(pair_px); free(pair_py)
        free(pair_pz); free(pair_abx); free(pair_aby); free(pair_abz); free(pair_weight)
        raise MemoryError("could not allocate AO-pair primitive buffers")

    tensor = np.zeros((naux, nao, nao), dtype=np.float64)
    tensor_v = tensor
    try:
        with nogil:
            for p in range(nao):
                for q in range(nao):
                    if pair_mask_v[p, q] == 0:
                        continue
                    npair = 0
                    for ip in range(nprim_v[p]):
                        for iq in range(nprim_v[q]):
                            ao_a = exps_v[p, ip]
                            ao_b = exps_v[q, iq]
                            ao_p = ao_a + ao_b
                            pair_a[npair] = ao_a
                            pair_b[npair] = ao_b
                            pair_p[npair] = ao_p
                            pair_px[npair] = (
                                ao_a * left_origins_v[p, 0]
                                + ao_b * right_origins_v[q, 0]
                            ) / ao_p
                            pair_py[npair] = (
                                ao_a * left_origins_v[p, 1]
                                + ao_b * right_origins_v[q, 1]
                            ) / ao_p
                            pair_pz[npair] = (
                                ao_a * left_origins_v[p, 2]
                                + ao_b * right_origins_v[q, 2]
                            ) / ao_p
                            pair_abx[npair] = left_origins_v[p, 0] - right_origins_v[q, 0]
                            pair_aby[npair] = left_origins_v[p, 1] - right_origins_v[q, 1]
                            pair_abz[npair] = left_origins_v[p, 2] - right_origins_v[q, 2]
                            pair_weight[npair] = weights_v[p, ip] * weights_v[q, iq]
                            npair += 1

                    for iaux in range(naux):
                        if aux_pair_mask_v[iaux, p, q] == 0:
                            continue
                        value = 0.0
                        for ipair in range(npair):
                            for ia in range(aux_nprim_v[iaux]):
                                value += (
                                    pair_weight[ipair]
                                    * aux_weights_v[iaux, ia]
                                    * primitive_short_range_three_center_precomputed(
                                        pair_a[ipair],
                                        pair_b[ipair],
                                        pair_p[ipair],
                                        pair_px[ipair],
                                        pair_py[ipair],
                                        pair_pz[ipair],
                                        pair_abx[ipair],
                                        pair_aby[ipair],
                                        pair_abz[ipair],
                                        <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2],
                                        <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2],
                                        aux_exps_v[iaux, ia],
                                        <int>aux_shells_v[iaux, 0], <int>aux_shells_v[iaux, 1], <int>aux_shells_v[iaux, 2],
                                        aux_origins_v[iaux, 0], aux_origins_v[iaux, 1], aux_origins_v[iaux, 2],
                                        eta,
                                    )
                                )
                        tensor_v[iaux, p, q] = value
    finally:
        free(pair_a); free(pair_b); free(pair_p); free(pair_px); free(pair_py)
        free(pair_pz); free(pair_abx); free(pair_aby); free(pair_abz); free(pair_weight)
    return tensor


cdef int compute_shell_triplet_short_range_vrr_hrr_into_tensor(
    int64_t[:, ::1] shells_v,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    int64_t[:, ::1] aux_shells_v,
    double[:, ::1] aux_origins_v,
    double[:, ::1] aux_exps_v,
    double[:, ::1] aux_weights_v,
    int64_t[::1] aux_nprim_v,
    uint8_t[:, ::1] pair_mask_v,
    uint8_t[:, :, ::1] aux_pair_mask_v,
    double[:, :, ::1] tensor_v,
    int p0,
    int p1,
    int q0,
    int q1,
    int a0,
    int a1,
    double* pq_a,
    double* pq_b,
    double* pq_p,
    double* pq_px,
    double* pq_py,
    double* pq_pz,
    int npq,
    double eta,
    double* vrr_table,
    double* lr_vrr_table,
    size_t vrr_table_cap,
) noexcept nogil:
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int na_ = a1 - a0
    cdef int lA, lB, lC, max_a_l, max_c_l, max_m_l
    cdef int ia, ib, ic, idx_pq, ip, iq, iap
    cdef int ao_p, ao_q, aux_i
    cdef int ax[OS_VRR_MAX_CART]
    cdef int ay[OS_VRR_MAX_CART]
    cdef int az[OS_VRR_MAX_CART]
    cdef int bx[OS_VRR_MAX_CART]
    cdef int by[OS_VRR_MAX_CART]
    cdef int bz[OS_VRR_MAX_CART]
    cdef int cx[OS_VRR_MAX_CART]
    cdef int cy[OS_VRR_MAX_CART]
    cdef int cz[OS_VRR_MAX_CART]
    cdef size_t vrr_table_size
    cdef double abx, aby, abz, ab2
    cdef double zeta, theta, theta_lr, eta2, lr_scale
    cdef double dx, dy, dz, pc2, base_pref
    cdef double cexp, prefac, full_value, lr_value
    cdef double PA[3]
    cdef double QC[3]
    cdef double PQ[3]
    cdef double AB[3]
    cdef double CD[3]
    cdef bint use_lr = eta > 0.0

    lA = <int>shells_v[p0, 0] + <int>shells_v[p0, 1] + <int>shells_v[p0, 2]
    lB = <int>shells_v[q0, 0] + <int>shells_v[q0, 1] + <int>shells_v[q0, 2]
    lC = <int>aux_shells_v[a0, 0] + <int>aux_shells_v[a0, 1] + <int>aux_shells_v[a0, 2]
    max_a_l = lA + lB
    max_c_l = lC
    max_m_l = max_a_l + max_c_l
    if max_a_l > OS_VRR_PAIR_MAX_L or max_c_l > OS_VRR_PAIR_MAX_L:
        return 0
    if np_ != ncart_for_l(lA) or nq_ != ncart_for_l(lB) or na_ != ncart_for_l(lC):
        return 0

    vrr_table_size = (
        <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1)
        * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1)
        * <size_t>(max_m_l + 1)
    )
    if vrr_table == NULL or vrr_table_size > vrr_table_cap:
        return 0
    if use_lr and lr_vrr_table == NULL:
        return 0

    fill_cartesian_components(lA, ax, ay, az)
    fill_cartesian_components(lB, bx, by, bz)
    fill_cartesian_components(lC, cx, cy, cz)

    abx = left_origins_v[p0, 0] - right_origins_v[q0, 0]
    aby = left_origins_v[p0, 1] - right_origins_v[q0, 1]
    abz = left_origins_v[p0, 2] - right_origins_v[q0, 2]
    ab2 = abx * abx + aby * aby + abz * abz
    AB[0] = abx; AB[1] = aby; AB[2] = abz
    CD[0] = 0.0; CD[1] = 0.0; CD[2] = 0.0
    eta2 = eta * eta

    for idx_pq in range(npq):
        ip = idx_pq // <int>nprim_v[q0]
        iq = idx_pq - ip * <int>nprim_v[q0]
        for iap in range(aux_nprim_v[a0]):
            cexp = aux_exps_v[a0, iap]
            zeta = pq_p[idx_pq] + cexp
            theta = pq_p[idx_pq] * cexp / zeta
            dx = pq_px[idx_pq] - aux_origins_v[a0, 0]
            dy = pq_py[idx_pq] - aux_origins_v[a0, 1]
            dz = pq_pz[idx_pq] - aux_origins_v[a0, 2]
            pc2 = dx * dx + dy * dy + dz * dz
            base_pref = (
                ERI_PREFAC
                * exp(-(pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]) * ab2)
                / (pq_p[idx_pq] * cexp * sqrt(zeta))
            )
            PA[0] = pq_px[idx_pq] - left_origins_v[p0, 0]
            PA[1] = pq_py[idx_pq] - left_origins_v[p0, 1]
            PA[2] = pq_pz[idx_pq] - left_origins_v[p0, 2]
            QC[0] = 0.0; QC[1] = 0.0; QC[2] = 0.0
            PQ[0] = dx; PQ[1] = dy; PQ[2] = dz
            os_fill_vrr_table_kernel(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                cexp,
                theta,
                theta * pc2,
                base_pref,
                PA,
                QC,
                PQ,
            )
            if use_lr:
                theta_lr = theta * eta2 / (theta + eta2)
                lr_scale = eta / sqrt(theta + eta2)
                os_fill_vrr_table_kernel(
                    lr_vrr_table,
                    max_a_l,
                    max_c_l,
                    max_m_l,
                    pq_p[idx_pq],
                    cexp,
                    theta_lr,
                    theta_lr * pc2,
                    base_pref,
                    PA,
                    QC,
                    PQ,
                )
                os_subtract_valid_vrr_states(
                    vrr_table,
                    lr_vrr_table,
                    max_a_l,
                    max_c_l,
                    max_m_l,
                    lr_scale,
                )
            else:
                lr_scale = 0.0
            for ia in range(np_):
                ao_p = p0 + ia
                for ib in range(nq_):
                    ao_q = q0 + ib
                    if pair_mask_v[ao_p, ao_q] == 0:
                        continue
                    for ic in range(na_):
                        aux_i = a0 + ic
                        if aux_pair_mask_v[aux_i, ao_p, ao_q] == 0:
                            continue
                        prefac = (
                            weights_v[ao_p, ip]
                            * weights_v[ao_q, iq]
                            * aux_weights_v[aux_i, iap]
                        )
                        full_value = os_vrr_hrr_eval_expanded(
                            vrr_table,
                            ax[ia], ay[ia], az[ia],
                            bx[ib], by[ib], bz[ib],
                            cx[ic], cy[ic], cz[ic],
                            0, 0, 0,
                            0,
                            max_a_l,
                            max_c_l,
                            max_m_l,
                            AB,
                            CD,
                        )
                        tensor_v[aux_i, ao_p, ao_q] += prefac * full_value

    return 1


cdef inline bint shell_pair_mask_has_work(
    uint8_t[:, ::1] pair_mask_v,
    int p0,
    int p1,
    int q0,
    int q1,
) noexcept nogil:
    cdef int p, q
    for p in range(p0, p1):
        for q in range(q0, q1):
            if pair_mask_v[p, q] != 0:
                return True
    return False


cdef inline bint shell_triplet_mask_has_work(
    uint8_t[:, ::1] pair_mask_v,
    uint8_t[:, :, ::1] aux_pair_mask_v,
    int p0,
    int p1,
    int q0,
    int q1,
    int a0,
    int a1,
) noexcept nogil:
    cdef int p, q, a
    for p in range(p0, p1):
        for q in range(q0, q1):
            if pair_mask_v[p, q] == 0:
                continue
            for a in range(a0, a1):
                if aux_pair_mask_v[a, p, q] != 0:
                    return True
    return False


cpdef compute_short_range_three_center_tensor_shell_blocked_masked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    cnp.ndarray[int64_t, ndim=1] aux_shell_starts,
    cnp.ndarray[int64_t, ndim=1] aux_shell_stops,
    cnp.ndarray[uint8_t, ndim=2] pair_mask,
    cnp.ndarray[uint8_t, ndim=3] aux_pair_mask,
    double eta,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef int naux_shell = aux_shell_starts.shape[0]
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef cnp.ndarray[double, ndim=3] tensor
    cdef int ish, jsh, ash, p0, p1, q0, q1, a0, a1
    cdef int p, q, a, direct_done
    cdef int* shell_pair_n
    cdef double* shell_pair_a
    cdef double* shell_pair_b
    cdef double* shell_pair_p
    cdef double* shell_pair_px
    cdef double* shell_pair_py
    cdef double* shell_pair_pz
    cdef double* direct_vrr_table
    cdef double* direct_lr_vrr_table
    cdef size_t pair_storage_size
    cdef size_t pq_offset
    cdef int pq_pair_idx
    cdef int npq
    cdef double value
    cdef size_t direct_vrr_table_cap = (
        <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(2 * OS_VRR_PAIR_MAX_L + 1)
    )
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int64_t[::1] aux_shell_starts_v = aux_shell_starts
    cdef int64_t[::1] aux_shell_stops_v = aux_shell_stops
    cdef uint8_t[:, ::1] pair_mask_v = pair_mask
    cdef uint8_t[:, :, ::1] aux_pair_mask_v = aux_pair_mask
    cdef double[:, :, ::1] tensor_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if left_origins.shape[0] != nao or right_origins.shape[0] != nao:
        raise ValueError("left_origins and right_origins must have one row per AO function.")
    if pair_mask.shape[0] != nao or pair_mask.shape[1] != nao:
        raise ValueError("pair_mask must have shape (nao, nao).")
    if aux_pair_mask.shape[0] != naux or aux_pair_mask.shape[1] != nao or aux_pair_mask.shape[2] != nao:
        raise ValueError("aux_pair_mask must have shape (naux, nao, nao).")

    pair_storage_size = <size_t>nshell * <size_t>nshell * <size_t>pair_cap
    shell_pair_n = <int*>malloc(nshell * nshell * sizeof(int))
    shell_pair_a = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_b = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_p = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_px = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_py = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_pz = <double*>malloc(pair_storage_size * sizeof(double))
    direct_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    direct_lr_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    if (
        shell_pair_n == NULL or shell_pair_a == NULL or shell_pair_b == NULL
        or shell_pair_p == NULL or shell_pair_px == NULL or shell_pair_py == NULL
        or shell_pair_pz == NULL or direct_vrr_table == NULL
        or direct_lr_vrr_table == NULL
    ):
        free(shell_pair_n)
        free(shell_pair_a); free(shell_pair_b); free(shell_pair_p)
        free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
        free(direct_vrr_table); free(direct_lr_vrr_table)
        raise MemoryError("Could not allocate short-range shell-block scratch arrays.")

    tensor = np.zeros((naux, nao, nao), dtype=np.float64)
    tensor_v = tensor
    with nogil:
        for ish in range(nshell):
            p0 = <int>shell_starts_v[ish]
            p1 = <int>shell_stops_v[ish]
            for jsh in range(nshell):
                q0 = <int>shell_starts_v[jsh]
                q1 = <int>shell_stops_v[jsh]
                pq_pair_idx = ish * nshell + jsh
                pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                if shell_pair_mask_has_work(pair_mask_v, p0, p1, q0, q1):
                    shell_pair_n[pq_pair_idx] = precompute_primitive_pair_geom_asymmetric(
                        p0,
                        q0,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        nprim_v,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                    )
                else:
                    shell_pair_n[pq_pair_idx] = 0

        for ash in range(naux_shell):
            a0 = <int>aux_shell_starts_v[ash]
            a1 = <int>aux_shell_stops_v[ash]
            for ish in range(nshell):
                p0 = <int>shell_starts_v[ish]
                p1 = <int>shell_stops_v[ish]
                for jsh in range(nshell):
                    q0 = <int>shell_starts_v[jsh]
                    q1 = <int>shell_stops_v[jsh]
                    pq_pair_idx = ish * nshell + jsh
                    pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                    npq = shell_pair_n[pq_pair_idx]
                    if npq <= 0:
                        continue
                    if not shell_triplet_mask_has_work(
                        pair_mask_v,
                        aux_pair_mask_v,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                    ):
                        continue
                    direct_done = compute_shell_triplet_short_range_vrr_hrr_into_tensor(
                        shells_v,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        pair_mask_v,
                        aux_pair_mask_v,
                        tensor_v,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                        npq,
                        eta,
                        direct_vrr_table,
                        direct_lr_vrr_table,
                        direct_vrr_table_cap,
                    )
                    if direct_done == 0:
                        for a in range(a0, a1):
                            for p in range(p0, p1):
                                for q in range(q0, q1):
                                    if pair_mask_v[p, q] == 0 or aux_pair_mask_v[a, p, q] == 0:
                                        continue
                                    value = contracted_short_range_three_center_indices(
                                        p, q, a,
                                        shells_v,
                                        left_origins_v,
                                        right_origins_v,
                                        exps_v,
                                        weights_v,
                                        nprim_v,
                                        aux_shells_v,
                                        aux_origins_v,
                                        aux_exps_v,
                                        aux_weights_v,
                                        aux_nprim_v,
                                        eta,
                                    )
                                    tensor_v[a, p, q] = value

    free(shell_pair_n)
    free(shell_pair_a); free(shell_pair_b); free(shell_pair_p)
    free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
    free(direct_vrr_table); free(direct_lr_vrr_table)
    return tensor


cdef int accumulate_shell_triplet_short_range_vrr_hrr_bloch(
    int64_t[:, ::1] shells_v,
    double[:, ::1] left_origins_v,
    double[:, ::1] right_origins_v,
    double[:, ::1] exps_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    int64_t[:, ::1] aux_shells_v,
    double[:, :, ::1] aux_origins_many_v,
    double[:, ::1] aux_exps_v,
    double[:, ::1] aux_weights_v,
    int64_t[::1] aux_nprim_v,
    uint8_t[:, ::1] pair_mask_v,
    uint8_t[:, :, ::1] aux_pair_mask_v,
    double[:, ::1] phase_real_v,
    double[:, ::1] phase_imag_v,
    double[:, ::1] mirror_phase_real_v,
    double[:, ::1] mirror_phase_imag_v,
    uint8_t[::1] mirror_flags_v,
    double[:, :, :, ::1] out_real_v,
    double[:, :, :, ::1] out_imag_v,
    int64_t[::1] direct_bins_v,
    int64_t[::1] mirror_bins_v,
    bint folded_mode,
    int output_aux_offset,
    int p0,
    int p1,
    int q0,
    int q1,
    int a0,
    int a1,
    double* pq_a,
    double* pq_b,
    double* pq_p,
    double* pq_px,
    double* pq_py,
    double* pq_pz,
    int npq,
    double eta,
    double primitive_exp_cutoff,
    int64_t* primitive_candidates,
    int64_t* primitive_skips,
    double* vrr_table,
    double* lr_vrr_table,
    size_t vrr_table_cap,
    int* active_tasks,
    int64_t* shell_task_skips,
) noexcept nogil:
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int na_ = a1 - a0
    cdef int ntask = aux_origins_many_v.shape[0]
    cdef int nbloch = phase_real_v.shape[1]
    cdef int lA, lB, lC, max_a_l, max_c_l, max_m_l
    cdef int ia, ib, ic, idx_pq, ip, iq, iap, task, active_idx, nactive, b
    cdef int ao_p, ao_q, aux_i, output_aux_i
    cdef int ax[OS_VRR_MAX_CART]
    cdef int ay[OS_VRR_MAX_CART]
    cdef int az[OS_VRR_MAX_CART]
    cdef int bx[OS_VRR_MAX_CART]
    cdef int by[OS_VRR_MAX_CART]
    cdef int bz[OS_VRR_MAX_CART]
    cdef int cx[OS_VRR_MAX_CART]
    cdef int cy[OS_VRR_MAX_CART]
    cdef int cz[OS_VRR_MAX_CART]
    cdef size_t vrr_table_size
    cdef double abx, aby, abz, ab2
    cdef double zeta, theta, theta_lr, eta2, lr_scale
    cdef double dx, dy, dz, pc2, base_pref, decay_exponent, pair_decay
    cdef double min_pair_decay, min_decay_exponent, lower_bound
    cdef double pmin[3]
    cdef double pmax[3]
    cdef double cexp, prefac, full_value, lr_value, value
    cdef double PA[3]
    cdef double QC[3]
    cdef double PQ[3]
    cdef double AB[3]
    cdef double CD[3]
    cdef bint use_lr = eta > 0.0

    lA = <int>shells_v[p0, 0] + <int>shells_v[p0, 1] + <int>shells_v[p0, 2]
    lB = <int>shells_v[q0, 0] + <int>shells_v[q0, 1] + <int>shells_v[q0, 2]
    lC = <int>aux_shells_v[a0, 0] + <int>aux_shells_v[a0, 1] + <int>aux_shells_v[a0, 2]
    max_a_l = lA + lB
    max_c_l = lC
    max_m_l = max_a_l + max_c_l
    if max_a_l > OS_VRR_PAIR_MAX_L or max_c_l > OS_VRR_PAIR_MAX_L:
        return 0
    if np_ != ncart_for_l(lA) or nq_ != ncart_for_l(lB) or na_ != ncart_for_l(lC):
        return 0

    vrr_table_size = (
        <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1)
        * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1)
        * <size_t>(max_m_l + 1)
    )
    if vrr_table == NULL or vrr_table_size > vrr_table_cap:
        return 0
    if use_lr and lr_vrr_table == NULL:
        return 0

    fill_cartesian_components(lA, ax, ay, az)
    fill_cartesian_components(lB, bx, by, bz)
    fill_cartesian_components(lC, cx, cy, cz)

    abx = left_origins_v[p0, 0] - right_origins_v[q0, 0]
    aby = left_origins_v[p0, 1] - right_origins_v[q0, 1]
    abz = left_origins_v[p0, 2] - right_origins_v[q0, 2]
    ab2 = abx * abx + aby * aby + abz * abz
    AB[0] = abx; AB[1] = aby; AB[2] = abz
    CD[0] = 0.0; CD[1] = 0.0; CD[2] = 0.0
    QC[0] = 0.0; QC[1] = 0.0; QC[2] = 0.0
    eta2 = eta * eta

    # PySCF/libpbc-style shell screening: bound the primitive product centers
    # by a box and skip image tasks that cannot pass the exact primitive test.
    nactive = ntask
    if primitive_exp_cutoff > 0.0:
        min_pair_decay = 1e300
        min_decay_exponent = 1e300
        pmin[0] = 1e300; pmin[1] = 1e300; pmin[2] = 1e300
        pmax[0] = -1e300; pmax[1] = -1e300; pmax[2] = -1e300
        for idx_pq in range(npq):
            pair_decay = (
                pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]
            ) * ab2
            if pair_decay < min_pair_decay:
                min_pair_decay = pair_decay
            if pq_px[idx_pq] < pmin[0]:
                pmin[0] = pq_px[idx_pq]
            if pq_py[idx_pq] < pmin[1]:
                pmin[1] = pq_py[idx_pq]
            if pq_pz[idx_pq] < pmin[2]:
                pmin[2] = pq_pz[idx_pq]
            if pq_px[idx_pq] > pmax[0]:
                pmax[0] = pq_px[idx_pq]
            if pq_py[idx_pq] > pmax[1]:
                pmax[1] = pq_py[idx_pq]
            if pq_pz[idx_pq] > pmax[2]:
                pmax[2] = pq_pz[idx_pq]
            for iap in range(aux_nprim_v[a0]):
                cexp = aux_exps_v[a0, iap]
                theta = pq_p[idx_pq] * cexp / (pq_p[idx_pq] + cexp)
                if use_lr:
                    decay_exponent = theta * eta2 / (theta + eta2)
                else:
                    decay_exponent = theta
                if decay_exponent < min_decay_exponent:
                    min_decay_exponent = decay_exponent

        nactive = 0
        for task in range(ntask):
            pc2 = 0.0
            if aux_origins_many_v[task, a0, 0] < pmin[0]:
                dx = pmin[0] - aux_origins_many_v[task, a0, 0]
                pc2 += dx * dx
            elif aux_origins_many_v[task, a0, 0] > pmax[0]:
                dx = aux_origins_many_v[task, a0, 0] - pmax[0]
                pc2 += dx * dx
            if aux_origins_many_v[task, a0, 1] < pmin[1]:
                dy = pmin[1] - aux_origins_many_v[task, a0, 1]
                pc2 += dy * dy
            elif aux_origins_many_v[task, a0, 1] > pmax[1]:
                dy = aux_origins_many_v[task, a0, 1] - pmax[1]
                pc2 += dy * dy
            if aux_origins_many_v[task, a0, 2] < pmin[2]:
                dz = pmin[2] - aux_origins_many_v[task, a0, 2]
                pc2 += dz * dz
            elif aux_origins_many_v[task, a0, 2] > pmax[2]:
                dz = aux_origins_many_v[task, a0, 2] - pmax[2]
                pc2 += dz * dz
            lower_bound = min_pair_decay + min_decay_exponent * pc2
            if lower_bound <= (
                primitive_exp_cutoff
                + 1e-12 * (1.0 + primitive_exp_cutoff)
            ):
                active_tasks[nactive] = task
                nactive += 1
            else:
                primitive_candidates[0] += npq * aux_nprim_v[a0]
                primitive_skips[0] += npq * aux_nprim_v[a0]
                shell_task_skips[0] += 1
    else:
        for task in range(ntask):
            active_tasks[task] = task

    # Primitive and shell metadata are invariant across all image tasks in a
    # relative-translation group. Keep them outside the task loop and write
    # each contribution directly into its Bloch-summed destination.
    for idx_pq in range(npq):
        ip = idx_pq // <int>nprim_v[q0]
        iq = idx_pq - ip * <int>nprim_v[q0]
        PA[0] = pq_px[idx_pq] - left_origins_v[p0, 0]
        PA[1] = pq_py[idx_pq] - left_origins_v[p0, 1]
        PA[2] = pq_pz[idx_pq] - left_origins_v[p0, 2]
        pair_decay = (
            pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]
        ) * ab2
        for iap in range(aux_nprim_v[a0]):
            cexp = aux_exps_v[a0, iap]
            zeta = pq_p[idx_pq] + cexp
            theta = pq_p[idx_pq] * cexp / zeta
            base_pref = (
                ERI_PREFAC
                * exp(-pair_decay)
                / (pq_p[idx_pq] * cexp * sqrt(zeta))
            )
            if use_lr:
                theta_lr = theta * eta2 / (theta + eta2)
                lr_scale = eta / sqrt(theta + eta2)
                decay_exponent = theta_lr
            else:
                theta_lr = 0.0
                lr_scale = 0.0
                decay_exponent = theta

            for active_idx in range(nactive):
                task = active_tasks[active_idx]
                dx = pq_px[idx_pq] - aux_origins_many_v[task, a0, 0]
                dy = pq_py[idx_pq] - aux_origins_many_v[task, a0, 1]
                dz = pq_pz[idx_pq] - aux_origins_many_v[task, a0, 2]
                pc2 = dx * dx + dy * dy + dz * dz
                primitive_candidates[0] += 1
                if (
                    primitive_exp_cutoff > 0.0
                    and pair_decay + decay_exponent * pc2
                    > primitive_exp_cutoff
                ):
                    primitive_skips[0] += 1
                    continue
                PQ[0] = dx; PQ[1] = dy; PQ[2] = dz
                os_fill_vrr_table_kernel(
                    vrr_table,
                    max_a_l,
                    max_c_l,
                    max_m_l,
                    pq_p[idx_pq],
                    cexp,
                    theta,
                    theta * pc2,
                    base_pref,
                    PA,
                    QC,
                    PQ,
                )
                if use_lr:
                    os_fill_vrr_table_kernel(
                        lr_vrr_table,
                        max_a_l,
                        max_c_l,
                        max_m_l,
                        pq_p[idx_pq],
                        cexp,
                        theta_lr,
                        theta_lr * pc2,
                        base_pref,
                        PA,
                        QC,
                        PQ,
                    )
                    os_subtract_valid_vrr_states(
                        vrr_table,
                        lr_vrr_table,
                        max_a_l,
                        max_c_l,
                        max_m_l,
                        lr_scale,
                    )
                for ia in range(np_):
                    ao_p = p0 + ia
                    for ib in range(nq_):
                        ao_q = q0 + ib
                        if pair_mask_v[ao_p, ao_q] == 0:
                            continue
                        for ic in range(na_):
                            aux_i = a0 + ic
                            if aux_pair_mask_v[aux_i, ao_p, ao_q] == 0:
                                continue
                            prefac = (
                                weights_v[ao_p, ip]
                                * weights_v[ao_q, iq]
                                * aux_weights_v[aux_i, iap]
                            )
                            full_value = os_vrr_hrr_eval_expanded(
                                vrr_table,
                                ax[ia], ay[ia], az[ia],
                                bx[ib], by[ib], bz[ib],
                                cx[ic], cy[ic], cz[ic],
                                0, 0, 0,
                                0,
                                max_a_l,
                                max_c_l,
                                max_m_l,
                                AB,
                                CD,
                            )
                            value = prefac * full_value
                            if value == 0.0:
                                continue
                            if folded_mode:
                                output_aux_i = aux_i - output_aux_offset
                                out_real_v[
                                    output_aux_i,
                                    ao_p,
                                    ao_q,
                                    direct_bins_v[task],
                                ] += value
                                if mirror_flags_v[task] != 0:
                                    out_real_v[
                                        output_aux_i,
                                        ao_q,
                                        ao_p,
                                        mirror_bins_v[task],
                                    ] += value
                            elif mirror_flags_v[task] != 0:
                                phase_axpy_ri(
                                    nbloch,
                                    value,
                                    &phase_real_v[task, 0],
                                    &phase_imag_v[task, 0],
                                    &out_real_v[aux_i, ao_p, ao_q, 0],
                                    &out_imag_v[aux_i, ao_p, ao_q, 0],
                                )
                                phase_axpy_ri(
                                    nbloch,
                                    value,
                                    &mirror_phase_real_v[task, 0],
                                    &mirror_phase_imag_v[task, 0],
                                    &out_real_v[aux_i, ao_q, ao_p, 0],
                                    &out_imag_v[aux_i, ao_q, ao_p, 0],
                                )
                            else:
                                phase_axpy_ri(
                                    nbloch,
                                    value,
                                    &phase_real_v[task, 0],
                                    &phase_imag_v[task, 0],
                                    &out_real_v[aux_i, ao_p, ao_q, 0],
                                    &out_imag_v[aux_i, ao_p, ao_q, 0],
                                )
    return 1


cpdef compute_short_range_three_center_bloch_shell_blocked_group_masked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=2] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=3] aux_origins_many,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    cnp.ndarray[int64_t, ndim=1] aux_shell_starts,
    cnp.ndarray[int64_t, ndim=1] aux_shell_stops,
    cnp.ndarray[uint8_t, ndim=2] pair_mask,
    cnp.ndarray[uint8_t, ndim=3] aux_pair_mask,
    cnp.ndarray[double, ndim=2] phase_real,
    cnp.ndarray[double, ndim=2] phase_imag,
    cnp.ndarray[double, ndim=2] mirror_phase_real,
    cnp.ndarray[double, ndim=2] mirror_phase_imag,
    cnp.ndarray[uint8_t, ndim=1] mirror_flags,
    double eta,
    double primitive_exp_cutoff=0.0,
    object output_real=None,
    object output_imag=None,
    object direct_bins=None,
    object mirror_bins=None,
    int aux_shell_begin=0,
    int aux_shell_end=-1,
    int output_aux_offset=0,
):
    """Accumulate one relative-translation group into Bloch-sum buffers."""

    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int ntask = aux_origins_many.shape[0]
    cdef int nbloch = phase_real.shape[1]
    cdef int nshell = shell_starts.shape[0]
    cdef int naux_shell = aux_shell_starts.shape[0]
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef int ish, jsh, ash, p0, p1, q0, q1, a0, a1
    cdef int p, q, a, b, task, direct_done
    cdef int* shell_pair_n
    cdef double* shell_pair_a
    cdef double* shell_pair_b
    cdef double* shell_pair_p
    cdef double* shell_pair_px
    cdef double* shell_pair_py
    cdef double* shell_pair_pz
    cdef double* direct_vrr_table
    cdef double* direct_lr_vrr_table
    cdef int* active_tasks
    cdef size_t pair_storage_size
    cdef size_t pq_offset
    cdef int pq_pair_idx
    cdef int npq
    cdef int64_t primitive_candidates = 0
    cdef int64_t primitive_skips = 0
    cdef int64_t shell_task_skips = 0
    cdef bint folded_mode = direct_bins is not None or mirror_bins is not None
    cdef double value
    cdef size_t direct_vrr_table_cap = (
        <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(2 * OS_VRR_PAIR_MAX_L + 1)
    )
    cdef cnp.ndarray[double, ndim=4] out_real
    cdef cnp.ndarray[double, ndim=4] out_imag
    cdef cnp.ndarray[int64_t, ndim=1] direct_bins_array
    cdef cnp.ndarray[int64_t, ndim=1] mirror_bins_array
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, :, ::1] aux_origins_many_v = aux_origins_many
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int64_t[::1] aux_shell_starts_v = aux_shell_starts
    cdef int64_t[::1] aux_shell_stops_v = aux_shell_stops
    cdef uint8_t[:, ::1] pair_mask_v = pair_mask
    cdef uint8_t[:, :, ::1] aux_pair_mask_v = aux_pair_mask
    cdef double[:, ::1] phase_real_v = phase_real
    cdef double[:, ::1] phase_imag_v = phase_imag
    cdef double[:, ::1] mirror_phase_real_v = mirror_phase_real
    cdef double[:, ::1] mirror_phase_imag_v = mirror_phase_imag
    cdef uint8_t[::1] mirror_flags_v = mirror_flags
    cdef double[:, :, :, ::1] out_real_v
    cdef double[:, :, :, ::1] out_imag_v
    cdef int64_t[::1] direct_bins_v
    cdef int64_t[::1] mirror_bins_v

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if primitive_exp_cutoff < 0.0:
        raise ValueError("primitive_exp_cutoff must be non-negative.")
    if left_origins.shape[0] != nao or right_origins.shape[0] != nao:
        raise ValueError("left_origins and right_origins must have one row per AO function.")
    if aux_origins_many.shape[1] != naux or aux_origins_many.shape[2] != 3:
        raise ValueError("aux_origins_many must have shape (ntask, naux, 3).")
    if pair_mask.shape[0] != nao or pair_mask.shape[1] != nao:
        raise ValueError("pair_mask must have shape (nao, nao).")
    if aux_pair_mask.shape[0] != naux or aux_pair_mask.shape[1] != nao or aux_pair_mask.shape[2] != nao:
        raise ValueError("aux_pair_mask must have shape (naux, nao, nao).")
    if mirror_flags.shape[0] != ntask:
        raise ValueError("mirror_flags must have shape (ntask,).")
    if aux_shell_end < 0:
        aux_shell_end = naux_shell
    if (
        aux_shell_begin < 0
        or aux_shell_begin > aux_shell_end
        or aux_shell_end > naux_shell
    ):
        raise ValueError("Invalid auxiliary-shell range.")
    if folded_mode:
        if direct_bins is None or mirror_bins is None:
            raise ValueError("direct_bins and mirror_bins must be provided together.")
        direct_bins_array = np.ascontiguousarray(direct_bins, dtype=np.int64)
        mirror_bins_array = np.ascontiguousarray(mirror_bins, dtype=np.int64)
        if direct_bins_array.shape[0] != ntask or mirror_bins_array.shape[0] != ntask:
            raise ValueError("folded bin arrays must have shape (ntask,).")
    else:
        if phase_real.shape[0] != ntask or phase_imag.shape[0] != ntask or phase_imag.shape[1] != nbloch:
            raise ValueError("phase arrays must have shape (ntask, nbloch).")
        if mirror_phase_real.shape[0] != ntask or mirror_phase_real.shape[1] != nbloch:
            raise ValueError("mirror phase arrays must have shape (ntask, nbloch).")
        if mirror_phase_imag.shape[0] != ntask or mirror_phase_imag.shape[1] != nbloch:
            raise ValueError("mirror phase arrays must have shape (ntask, nbloch).")
        direct_bins_array = np.zeros(ntask, dtype=np.int64)
        mirror_bins_array = np.zeros(ntask, dtype=np.int64)
    direct_bins_v = direct_bins_array
    mirror_bins_v = mirror_bins_array

    pair_storage_size = <size_t>nshell * <size_t>nshell * <size_t>pair_cap
    shell_pair_n = <int*>malloc(nshell * nshell * sizeof(int))
    shell_pair_a = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_b = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_p = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_px = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_py = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_pz = <double*>malloc(pair_storage_size * sizeof(double))
    direct_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    direct_lr_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    active_tasks = <int*>malloc(ntask * sizeof(int))
    if (
        shell_pair_n == NULL or shell_pair_a == NULL or shell_pair_b == NULL
        or shell_pair_p == NULL or shell_pair_px == NULL or shell_pair_py == NULL
        or shell_pair_pz == NULL or direct_vrr_table == NULL
        or direct_lr_vrr_table == NULL or active_tasks == NULL
    ):
        free(shell_pair_n)
        free(shell_pair_a); free(shell_pair_b); free(shell_pair_p)
        free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
        free(direct_vrr_table); free(direct_lr_vrr_table)
        free(active_tasks)
        raise MemoryError("Could not allocate grouped short-range scratch arrays.")

    if folded_mode:
        if output_real is None:
            raise ValueError("output_real is required for folded accumulation.")
        if not isinstance(output_real, np.ndarray):
            raise TypeError("output_real must be a NumPy array.")
        if output_real.dtype != np.float64 or not output_real.flags.c_contiguous:
            raise TypeError("output_real must be a C-contiguous float64 array.")
        if (
            output_real.shape[1] != nao
            or output_real.shape[2] != nao
        ):
            raise ValueError("output_real must have shape (naux_batch, nao, nao, nbins).")
        if output_aux_offset < 0 or output_aux_offset > naux:
            raise ValueError("output_aux_offset is outside the auxiliary basis.")
        if aux_shell_begin < aux_shell_end and (
            aux_shell_starts_v[aux_shell_begin] < output_aux_offset
            or aux_shell_stops_v[aux_shell_end - 1]
            > output_aux_offset + output_real.shape[0]
        ):
            raise ValueError("The auxiliary-shell range is outside output_real.")
        nbloch = output_real.shape[3]
        if (
            np.min(direct_bins_array, initial=0) < 0
            or np.max(direct_bins_array, initial=-1) >= nbloch
            or np.min(mirror_bins_array, initial=0) < 0
            or np.max(mirror_bins_array, initial=-1) >= nbloch
        ):
            raise ValueError("folded bin index is outside the output buffer.")
        out_real = output_real
        out_imag = np.zeros((1, 1, 1, 1), dtype=np.float64)
    elif (output_real is None) != (output_imag is None):
        raise ValueError("output_real and output_imag must be provided together.")
    elif output_real is None:
        out_real = np.zeros((naux, nao, nao, nbloch), dtype=np.float64)
        out_imag = np.zeros((naux, nao, nao, nbloch), dtype=np.float64)
    else:
        if not isinstance(output_real, np.ndarray) or not isinstance(
            output_imag,
            np.ndarray,
        ):
            raise TypeError("output buffers must be NumPy arrays.")
        if output_real.dtype != np.float64 or output_imag.dtype != np.float64:
            raise TypeError("output buffers must have dtype float64.")
        if (
            output_real.shape[0] != naux
            or output_real.shape[1] != nao
            or output_real.shape[2] != nao
            or output_real.shape[3] != nbloch
            or output_imag.shape[0] != naux
            or output_imag.shape[1] != nao
            or output_imag.shape[2] != nao
            or output_imag.shape[3] != nbloch
        ):
            raise ValueError(
                "output buffers must have shape (naux, nao, nao, nbloch)."
            )
        if not output_real.flags.c_contiguous or not output_imag.flags.c_contiguous:
            raise ValueError("output buffers must be C-contiguous.")
        out_real = output_real
        out_imag = output_imag
    out_real_v = out_real
    out_imag_v = out_imag

    with nogil:
        for ish in range(nshell):
            p0 = <int>shell_starts_v[ish]
            p1 = <int>shell_stops_v[ish]
            for jsh in range(nshell):
                q0 = <int>shell_starts_v[jsh]
                q1 = <int>shell_stops_v[jsh]
                pq_pair_idx = ish * nshell + jsh
                pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                if shell_pair_mask_has_work(pair_mask_v, p0, p1, q0, q1):
                    shell_pair_n[pq_pair_idx] = precompute_primitive_pair_geom_asymmetric(
                        p0,
                        q0,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        nprim_v,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                    )
                else:
                    shell_pair_n[pq_pair_idx] = 0

        for ash in range(aux_shell_begin, aux_shell_end):
            a0 = <int>aux_shell_starts_v[ash]
            a1 = <int>aux_shell_stops_v[ash]
            for ish in range(nshell):
                p0 = <int>shell_starts_v[ish]
                p1 = <int>shell_stops_v[ish]
                for jsh in range(nshell):
                    q0 = <int>shell_starts_v[jsh]
                    q1 = <int>shell_stops_v[jsh]
                    pq_pair_idx = ish * nshell + jsh
                    pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                    npq = shell_pair_n[pq_pair_idx]
                    if npq <= 0:
                        continue
                    if not shell_triplet_mask_has_work(
                        pair_mask_v,
                        aux_pair_mask_v,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                    ):
                        continue
                    direct_done = accumulate_shell_triplet_short_range_vrr_hrr_bloch(
                        shells_v,
                        left_origins_v,
                        right_origins_v,
                        exps_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_many_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        pair_mask_v,
                        aux_pair_mask_v,
                        phase_real_v,
                        phase_imag_v,
                        mirror_phase_real_v,
                        mirror_phase_imag_v,
                        mirror_flags_v,
                        out_real_v,
                        out_imag_v,
                        direct_bins_v,
                        mirror_bins_v,
                        folded_mode,
                        output_aux_offset,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                        npq,
                        eta,
                        primitive_exp_cutoff,
                        &primitive_candidates,
                        &primitive_skips,
                        direct_vrr_table,
                        direct_lr_vrr_table,
                        direct_vrr_table_cap,
                        active_tasks,
                        &shell_task_skips,
                    )
                    if direct_done == 0:
                        for task in range(ntask):
                            for a in range(a0, a1):
                                for p in range(p0, p1):
                                    for q in range(q0, q1):
                                        if pair_mask_v[p, q] == 0 or aux_pair_mask_v[a, p, q] == 0:
                                            continue
                                        value = contracted_short_range_three_center_indices(
                                            p,
                                            q,
                                            a,
                                            shells_v,
                                            left_origins_v,
                                            right_origins_v,
                                            exps_v,
                                            weights_v,
                                            nprim_v,
                                            aux_shells_v,
                                            aux_origins_many_v[task],
                                            aux_exps_v,
                                            aux_weights_v,
                                            aux_nprim_v,
                                            eta,
                                        )
                                        if value == 0.0:
                                            continue
                                        if folded_mode:
                                            out_real_v[
                                                a - output_aux_offset,
                                                p,
                                                q,
                                                direct_bins_v[task],
                                            ] += value
                                            if mirror_flags_v[task] != 0:
                                                out_real_v[
                                                    a - output_aux_offset,
                                                    q,
                                                    p,
                                                    mirror_bins_v[task],
                                                ] += value
                                        elif mirror_flags_v[task] != 0:
                                            phase_axpy_ri(
                                                nbloch,
                                                value,
                                                &phase_real_v[task, 0],
                                                &phase_imag_v[task, 0],
                                                &out_real_v[a, p, q, 0],
                                                &out_imag_v[a, p, q, 0],
                                            )
                                            phase_axpy_ri(
                                                nbloch,
                                                value,
                                                &mirror_phase_real_v[task, 0],
                                                &mirror_phase_imag_v[task, 0],
                                                &out_real_v[a, q, p, 0],
                                                &out_imag_v[a, q, p, 0],
                                            )
                                        else:
                                            phase_axpy_ri(
                                                nbloch,
                                                value,
                                                &phase_real_v[task, 0],
                                                &phase_imag_v[task, 0],
                                                &out_real_v[a, p, q, 0],
                                                &out_imag_v[a, p, q, 0],
                                            )

    free(shell_pair_n)
    free(shell_pair_a); free(shell_pair_b); free(shell_pair_p)
    free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
    free(direct_vrr_table); free(direct_lr_vrr_table)
    free(active_tasks)
    return (
        out_real,
        out_imag,
        primitive_candidates,
        primitive_skips,
        shell_task_skips,
    )


cpdef compute_ri_tensors(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] metric = np.zeros((naux, naux), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] j3 = np.zeros((naux, nao, nao), dtype=np.float64)
    cdef int p, q, a
    cdef double value
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] metric_v = metric
    cdef double[:, :, ::1] j3_v = j3

    with nogil:
        for p in range(naux):
            for q in range(p + 1):
                value = contracted_two_center_coulomb_indices(
                    p, q, aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v
                )
                metric_v[p, q] = value
                metric_v[q, p] = value

    for a in range(naux):
        for p in range(nao):
            for q in range(p + 1):
                value = contracted_three_center_indices(
                    p, q, a,
                    shells_v, origins_v, exps_v, weights_v, nprim_v,
                    aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v,
                )
                j3_v[a, p, q] = value
                j3_v[a, q, p] = value

    return metric, j3


cpdef compute_ri_tensors_packed(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef cnp.ndarray[double, ndim=2] metric = np.zeros((naux, naux), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] j3 = np.zeros((naux, npair), dtype=np.float64)
    cdef int p, q, a, pair
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double value, aux_bound
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, ::1] metric_v = metric
    cdef double[:, ::1] j3_v = j3

    for p in range(naux):
        for q in range(p + 1):
            value = contracted_two_center_coulomb_indices(
                p, q, aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v
            )
            metric_v[p, q] = value
            metric_v[q, p] = value

    for a in range(naux):
        aux_bound = sqrt(fabs(metric_v[a, a]))
        pair = 0
        for p in range(nao):
            for q in range(p + 1):
                if screen_tol > 0.0 and pair_bounds_v[p, q] * aux_bound < screen_tol:
                    skipped += 1
                    pair += 1
                    continue
                value = contracted_three_center_indices(
                    p, q, a,
                    shells_v, origins_v, exps_v, weights_v, nprim_v,
                    aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v,
                )
                j3_v[a, pair] = value
                computed += 1
                pair += 1

    return metric, j3, int(computed), int(skipped)


cpdef compute_aux_metric(
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
):
    cdef int naux = aux_shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] metric = np.zeros((naux, naux), dtype=np.float64)
    cdef int p, q
    cdef double value
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] metric_v = metric

    with nogil:
        for p in range(naux):
            for q in range(p + 1):
                value = contracted_two_center_coulomb_indices(
                    p, q, aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v
                )
                metric_v[p, q] = value
                metric_v[q, p] = value

    return metric


cpdef compute_aux_metric_range(
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    int row_start,
    int row_stop,
):
    cdef int naux = aux_shells.shape[0]
    cdef int nblock
    cdef cnp.ndarray[double, ndim=2] block
    cdef int p, q, local_p
    cdef double value
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] block_v

    if row_start < 0 or row_stop < row_start or row_stop > naux:
        raise ValueError("Invalid auxiliary metric row range.")

    nblock = row_stop - row_start
    block = np.zeros((nblock, naux), dtype=np.float64)
    block_v = block

    with nogil:
        for p in range(row_start, row_stop):
            local_p = p - row_start
            for q in range(p + 1):
                value = contracted_two_center_coulomb_indices(
                    p, q, aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v
                )
                block_v[local_p, q] = value

    return block


cpdef compute_ri_j3_packed_range(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    int aux_start,
    int aux_stop,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[double, ndim=1] aux_diag,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef int nblock
    cdef cnp.ndarray[double, ndim=2] j3
    cdef int p, q, a, local_a, pair
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double value, aux_bound
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[::1] aux_diag_v = aux_diag
    cdef double[:, ::1] j3_v

    if aux_start < 0 or aux_stop < aux_start or aux_stop > naux:
        raise ValueError("Invalid auxiliary range for RI tensor build.")

    nblock = aux_stop - aux_start
    j3 = np.zeros((nblock, npair), dtype=np.float64)
    j3_v = j3

    with nogil:
        for a in range(aux_start, aux_stop):
            local_a = a - aux_start
            aux_bound = aux_diag_v[a]
            pair = 0
            for p in range(nao):
                for q in range(p + 1):
                    if screen_tol > 0.0 and pair_bounds_v[p, q] * aux_bound < screen_tol:
                        skipped += 1
                        pair += 1
                        continue
                    value = contracted_three_center_indices(
                        p, q, a,
                        shells_v, origins_v, exps_v, weights_v, nprim_v,
                        aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v,
                    )
                    j3_v[local_a, pair] = value
                    computed += 1
                    pair += 1

    return j3, int(computed), int(skipped)


cdef int compute_shell_triplet_vrr_hrr_into_j3(
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    int64_t[:, ::1] aux_shells_v,
    double[:, ::1] aux_origins_v,
    double[:, ::1] aux_exps_v,
    double[:, ::1] aux_weights_v,
    int64_t[::1] aux_nprim_v,
    double[:, ::1] pair_bounds_v,
    double[::1] aux_diag_v,
    double[:, ::1] j3_v,
    int aux_start,
    int aux_stop,
    double screen_tol,
    int p0,
    int p1,
    int q0,
    int q1,
    int a0,
    int a1,
    double* pq_a,
    double* pq_b,
    double* pq_p,
    double* pq_px,
    double* pq_py,
    double* pq_pz,
    int npq,
    double* vrr_table,
    size_t vrr_table_cap,
) noexcept nogil:
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int na_ = a1 - a0
    cdef int lA, lB, lC, max_a_l, max_c_l, max_m_l
    cdef int ia, ib, ic, idx_pq, ip, iq, iap
    cdef int ao_p, ao_q, aux_i, local_aux, pair
    cdef int ax[OS_VRR_MAX_CART]
    cdef int ay[OS_VRR_MAX_CART]
    cdef int az[OS_VRR_MAX_CART]
    cdef int bx[OS_VRR_MAX_CART]
    cdef int by[OS_VRR_MAX_CART]
    cdef int bz[OS_VRR_MAX_CART]
    cdef int cx[OS_VRR_MAX_CART]
    cdef int cy[OS_VRR_MAX_CART]
    cdef int cz[OS_VRR_MAX_CART]
    cdef size_t vrr_table_size
    cdef double abx, aby, abz, ab2
    cdef double zeta, alpha, dx, dy, dz, pc2, T, base_pref
    cdef double cexp, prefac, value, pair_bound
    cdef double PA[3]
    cdef double QC[3]
    cdef double PQ[3]
    cdef double AB[3]
    cdef double CD[3]

    lA = <int>shells_v[p0, 0] + <int>shells_v[p0, 1] + <int>shells_v[p0, 2]
    lB = <int>shells_v[q0, 0] + <int>shells_v[q0, 1] + <int>shells_v[q0, 2]
    lC = <int>aux_shells_v[a0, 0] + <int>aux_shells_v[a0, 1] + <int>aux_shells_v[a0, 2]
    max_a_l = lA + lB
    max_c_l = lC
    max_m_l = max_a_l + max_c_l
    if max_a_l > OS_VRR_PAIR_MAX_L or max_c_l > OS_VRR_PAIR_MAX_L:
        return 0
    if np_ != ncart_for_l(lA) or nq_ != ncart_for_l(lB) or na_ != ncart_for_l(lC):
        return 0

    vrr_table_size = (
        <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1)
        * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1)
        * <size_t>(max_m_l + 1)
    )
    if vrr_table == NULL or vrr_table_size > vrr_table_cap:
        return 0

    fill_cartesian_components(lA, ax, ay, az)
    fill_cartesian_components(lB, bx, by, bz)
    fill_cartesian_components(lC, cx, cy, cz)

    abx = origins_v[p0, 0] - origins_v[q0, 0]
    aby = origins_v[p0, 1] - origins_v[q0, 1]
    abz = origins_v[p0, 2] - origins_v[q0, 2]
    ab2 = abx * abx + aby * aby + abz * abz
    AB[0] = abx; AB[1] = aby; AB[2] = abz
    CD[0] = 0.0; CD[1] = 0.0; CD[2] = 0.0

    for idx_pq in range(npq):
        ip = idx_pq // <int>nprim_v[q0]
        iq = idx_pq - ip * <int>nprim_v[q0]
        for iap in range(aux_nprim_v[a0]):
            cexp = aux_exps_v[a0, iap]
            zeta = pq_p[idx_pq] + cexp
            alpha = pq_p[idx_pq] * cexp / zeta
            dx = pq_px[idx_pq] - aux_origins_v[a0, 0]
            dy = pq_py[idx_pq] - aux_origins_v[a0, 1]
            dz = pq_pz[idx_pq] - aux_origins_v[a0, 2]
            pc2 = dx * dx + dy * dy + dz * dz
            T = alpha * pc2
            base_pref = (
                ERI_PREFAC
                * exp(-(pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]) * ab2)
                / (pq_p[idx_pq] * cexp * sqrt(zeta))
            )
            PA[0] = pq_px[idx_pq] - origins_v[p0, 0]
            PA[1] = pq_py[idx_pq] - origins_v[p0, 1]
            PA[2] = pq_pz[idx_pq] - origins_v[p0, 2]
            QC[0] = 0.0; QC[1] = 0.0; QC[2] = 0.0
            PQ[0] = dx; PQ[1] = dy; PQ[2] = dz
            os_fill_vrr_table(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                cexp,
                zeta,
                T,
                base_pref,
                PA,
                QC,
                PQ,
            )
            for ia in range(np_):
                ao_p = p0 + ia
                for ib in range(nq_):
                    ao_q = q0 + ib
                    if ao_p < ao_q:
                        continue
                    pair = pair_index(ao_p, ao_q)
                    pair_bound = pair_bounds_v[ao_p, ao_q]
                    for ic in range(na_):
                        aux_i = a0 + ic
                        if aux_i < aux_start or aux_i >= aux_stop:
                            continue
                        if screen_tol > 0.0 and pair_bound * aux_diag_v[aux_i] < screen_tol:
                            continue
                        local_aux = aux_i - aux_start
                        prefac = weights_v[ao_p, ip] * weights_v[ao_q, iq] * aux_weights_v[aux_i, iap]
                        value = prefac * os_vrr_hrr_eval_expanded(
                            vrr_table,
                            ax[ia], ay[ia], az[ia],
                            bx[ib], by[ib], bz[ib],
                            cx[ic], cy[ic], cz[ic],
                            0, 0, 0,
                            0,
                            max_a_l,
                            max_c_l,
                            max_m_l,
                            AB,
                            CD,
                        )
                        j3_v[local_aux, pair] += value

    return 1


cpdef compute_ri_j3_packed_range_shell_blocked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    int aux_start,
    int aux_stop,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[double, ndim=1] aux_diag,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    cnp.ndarray[int64_t, ndim=1] aux_shell_starts,
    cnp.ndarray[int64_t, ndim=1] aux_shell_stops,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef int naux_shell = aux_shell_starts.shape[0]
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef int npair = nao * (nao + 1) // 2
    cdef int nblock
    cdef cnp.ndarray[double, ndim=2] j3
    cdef int ish, jsh, ash, p0, p1, q0, q1, a0, a1
    cdef int p, q, a, local_a, pair, direct_done
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double pair_bound, value
    cdef int* shell_pair_n
    cdef double* shell_pair_a
    cdef double* shell_pair_b
    cdef double* shell_pair_p
    cdef double* shell_pair_px
    cdef double* shell_pair_py
    cdef double* shell_pair_pz
    cdef double* direct_vrr_table
    cdef size_t pair_storage_size
    cdef size_t pq_offset
    cdef int pq_pair_idx
    cdef size_t direct_vrr_table_cap = (
        <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(2 * OS_VRR_PAIR_MAX_L + 1)
    )
    cdef int npq
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[::1] aux_diag_v = aux_diag
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int64_t[::1] aux_shell_starts_v = aux_shell_starts
    cdef int64_t[::1] aux_shell_stops_v = aux_shell_stops
    cdef double[:, ::1] j3_v

    if aux_start < 0 or aux_stop < aux_start or aux_stop > naux:
        raise ValueError("Invalid auxiliary range for RI tensor build.")
    pair_storage_size = <size_t>nshell * <size_t>nshell * <size_t>pair_cap
    shell_pair_n = <int*>malloc(nshell * nshell * sizeof(int))
    shell_pair_a = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_b = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_p = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_px = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_py = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_pz = <double*>malloc(pair_storage_size * sizeof(double))
    if shell_pair_n == NULL or shell_pair_a == NULL or shell_pair_b == NULL or shell_pair_p == NULL or shell_pair_px == NULL or shell_pair_py == NULL or shell_pair_pz == NULL:
        free(shell_pair_n)
        free(shell_pair_a); free(shell_pair_b); free(shell_pair_p); free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
        raise MemoryError("Could not allocate shell-pair scratch arrays.")
    direct_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    if direct_vrr_table == NULL:
        free(shell_pair_n)
        free(shell_pair_a); free(shell_pair_b); free(shell_pair_p); free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
        raise MemoryError("Could not allocate VRR scratch table.")

    nblock = aux_stop - aux_start
    j3 = np.zeros((nblock, npair), dtype=np.float64)
    j3_v = j3

    with nogil:
        for ish in range(nshell):
            p0 = <int>shell_starts_v[ish]
            p1 = <int>shell_stops_v[ish]
            for jsh in range(nshell):
                q0 = <int>shell_starts_v[jsh]
                q1 = <int>shell_stops_v[jsh]
                pq_pair_idx = ish * nshell + jsh
                pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                shell_pair_n[pq_pair_idx] = precompute_primitive_pair_geom(
                    p0,
                    q0,
                    origins_v,
                    exps_v,
                    nprim_v,
                    shell_pair_a + pq_offset,
                    shell_pair_b + pq_offset,
                    shell_pair_p + pq_offset,
                    shell_pair_px + pq_offset,
                    shell_pair_py + pq_offset,
                    shell_pair_pz + pq_offset,
                )

        for ash in range(naux_shell):
            a0 = <int>aux_shell_starts_v[ash]
            a1 = <int>aux_shell_stops_v[ash]
            if a1 <= aux_start or a0 >= aux_stop:
                continue
            for ish in range(nshell):
                p0 = <int>shell_starts_v[ish]
                p1 = <int>shell_stops_v[ish]
                for jsh in range(ish + 1):
                    q0 = <int>shell_starts_v[jsh]
                    q1 = <int>shell_stops_v[jsh]
                    pq_pair_idx = ish * nshell + jsh
                    pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                    npq = shell_pair_n[pq_pair_idx]
                    direct_done = compute_shell_triplet_vrr_hrr_into_j3(
                        shells_v,
                        origins_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        pair_bounds_v,
                        aux_diag_v,
                        j3_v,
                        aux_start,
                        aux_stop,
                        screen_tol,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                        npq,
                        direct_vrr_table,
                        direct_vrr_table_cap,
                    )
                    if direct_done == 0:
                        for a in range(a0, a1):
                            if a < aux_start or a >= aux_stop:
                                continue
                            local_a = a - aux_start
                            for p in range(p0, p1):
                                for q in range(q0, q1):
                                    if p < q:
                                        continue
                                    if screen_tol > 0.0 and pair_bounds_v[p, q] * aux_diag_v[a] < screen_tol:
                                        continue
                                    pair = pair_index(p, q)
                                    value = contracted_three_center_indices(
                                        p, q, a,
                                        shells_v, origins_v, exps_v, weights_v, nprim_v,
                                        aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v,
                                    )
                                    j3_v[local_a, pair] = value

        for a in range(aux_start, aux_stop):
            for p in range(nao):
                for q in range(p + 1):
                    pair_bound = pair_bounds_v[p, q]
                    if screen_tol > 0.0 and pair_bound * aux_diag_v[a] < screen_tol:
                        skipped += 1
                    else:
                        computed += 1

    free(shell_pair_n)
    free(shell_pair_a); free(shell_pair_b); free(shell_pair_p); free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
    free(direct_vrr_table)
    return j3, int(computed), int(skipped)


cpdef compute_ri_j3_packed_range_shell_blocked_cached(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    int aux_start,
    int aux_stop,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[double, ndim=1] aux_diag,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    cnp.ndarray[int64_t, ndim=1] aux_shell_starts,
    cnp.ndarray[int64_t, ndim=1] aux_shell_stops,
    cnp.ndarray[int64_t, ndim=1] shell_pair_n,
    cnp.ndarray[double, ndim=2] shell_pair_a,
    cnp.ndarray[double, ndim=2] shell_pair_b,
    cnp.ndarray[double, ndim=2] shell_pair_p,
    cnp.ndarray[double, ndim=2] shell_pair_px,
    cnp.ndarray[double, ndim=2] shell_pair_py,
    cnp.ndarray[double, ndim=2] shell_pair_pz,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef int naux_shell = aux_shell_starts.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef int nblock
    cdef cnp.ndarray[double, ndim=2] j3
    cdef int ish, jsh, ash, p0, p1, q0, q1, a0, a1
    cdef int p, q, a, local_a, pair, direct_done
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double pair_bound, value
    cdef double* direct_vrr_table
    cdef int pq_pair_idx
    cdef int npq
    cdef size_t direct_vrr_table_cap = (
        <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(2 * OS_VRR_PAIR_MAX_L + 1)
    )
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[::1] aux_diag_v = aux_diag
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int64_t[::1] aux_shell_starts_v = aux_shell_starts
    cdef int64_t[::1] aux_shell_stops_v = aux_shell_stops
    cdef int64_t[::1] shell_pair_n_v = shell_pair_n
    cdef double[:, ::1] shell_pair_a_v = shell_pair_a
    cdef double[:, ::1] shell_pair_b_v = shell_pair_b
    cdef double[:, ::1] shell_pair_p_v = shell_pair_p
    cdef double[:, ::1] shell_pair_px_v = shell_pair_px
    cdef double[:, ::1] shell_pair_py_v = shell_pair_py
    cdef double[:, ::1] shell_pair_pz_v = shell_pair_pz
    cdef double[:, ::1] j3_v

    if aux_start < 0 or aux_stop < aux_start or aux_stop > naux:
        raise ValueError("Invalid auxiliary range for RI tensor build.")
    if shell_pair_n.shape[0] < nshell * nshell:
        raise ValueError("Invalid shell-pair cache size.")

    direct_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    if direct_vrr_table == NULL:
        raise MemoryError("Could not allocate VRR scratch table.")

    nblock = aux_stop - aux_start
    j3 = np.zeros((nblock, npair), dtype=np.float64)
    j3_v = j3

    with nogil:
        for ash in range(naux_shell):
            a0 = <int>aux_shell_starts_v[ash]
            a1 = <int>aux_shell_stops_v[ash]
            if a1 <= aux_start or a0 >= aux_stop:
                continue
            for ish in range(nshell):
                p0 = <int>shell_starts_v[ish]
                p1 = <int>shell_stops_v[ish]
                for jsh in range(ish + 1):
                    q0 = <int>shell_starts_v[jsh]
                    q1 = <int>shell_stops_v[jsh]
                    pq_pair_idx = ish * nshell + jsh
                    npq = <int>shell_pair_n_v[pq_pair_idx]
                    direct_done = compute_shell_triplet_vrr_hrr_into_j3(
                        shells_v,
                        origins_v,
                        weights_v,
                        nprim_v,
                        aux_shells_v,
                        aux_origins_v,
                        aux_exps_v,
                        aux_weights_v,
                        aux_nprim_v,
                        pair_bounds_v,
                        aux_diag_v,
                        j3_v,
                        aux_start,
                        aux_stop,
                        screen_tol,
                        p0,
                        p1,
                        q0,
                        q1,
                        a0,
                        a1,
                        &shell_pair_a_v[pq_pair_idx, 0],
                        &shell_pair_b_v[pq_pair_idx, 0],
                        &shell_pair_p_v[pq_pair_idx, 0],
                        &shell_pair_px_v[pq_pair_idx, 0],
                        &shell_pair_py_v[pq_pair_idx, 0],
                        &shell_pair_pz_v[pq_pair_idx, 0],
                        npq,
                        direct_vrr_table,
                        direct_vrr_table_cap,
                    )
                    if direct_done == 0:
                        for a in range(a0, a1):
                            if a < aux_start or a >= aux_stop:
                                continue
                            local_a = a - aux_start
                            for p in range(p0, p1):
                                for q in range(q0, q1):
                                    if p < q:
                                        continue
                                    if screen_tol > 0.0 and pair_bounds_v[p, q] * aux_diag_v[a] < screen_tol:
                                        continue
                                    pair = pair_index(p, q)
                                    value = contracted_three_center_indices(
                                        p, q, a,
                                        shells_v, origins_v, exps_v, weights_v, nprim_v,
                                        aux_shells_v, aux_origins_v, aux_exps_v, aux_weights_v, aux_nprim_v,
                                    )
                                    j3_v[local_a, pair] = value

        for a in range(aux_start, aux_stop):
            for p in range(nao):
                for q in range(p + 1):
                    pair_bound = pair_bounds_v[p, q]
                    if screen_tol > 0.0 and pair_bound * aux_diag_v[a] < screen_tol:
                        skipped += 1
                    else:
                        computed += 1

    free(direct_vrr_table)
    return j3, int(computed), int(skipped)


cpdef compute_ri_j3_packed_range_pair_outer(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[int64_t, ndim=2] aux_shells,
    cnp.ndarray[double, ndim=2] aux_origins,
    cnp.ndarray[double, ndim=2] aux_exps,
    cnp.ndarray[double, ndim=2] aux_weights,
    cnp.ndarray[int64_t, ndim=1] aux_nprim,
    int aux_start,
    int aux_stop,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[double, ndim=1] aux_diag,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int naux = aux_shells.shape[0]
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef int npair = nao * (nao + 1) // 2
    cdef int nblock
    cdef cnp.ndarray[double, ndim=2] j3
    cdef int p, q, a, local_a, pair, ij, ia, npq
    cdef int l1, m1, n1, l2, m2, n2, l3, m3, n3
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double value, pair_bound
    cdef double Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz
    cdef double abx, aby, abz
    cdef double* pq_a = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_b = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_p = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_px = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_py = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_pz = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_w = <double*>malloc(pair_cap * sizeof(double))
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[:, ::1] aux_shells_v = aux_shells
    cdef double[:, ::1] aux_origins_v = aux_origins
    cdef double[:, ::1] aux_exps_v = aux_exps
    cdef double[:, ::1] aux_weights_v = aux_weights
    cdef int64_t[::1] aux_nprim_v = aux_nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[::1] aux_diag_v = aux_diag
    cdef double[:, ::1] j3_v

    if aux_start < 0 or aux_stop < aux_start or aux_stop > naux:
        raise ValueError("Invalid auxiliary range for RI tensor build.")
    if pq_a == NULL or pq_b == NULL or pq_p == NULL or pq_px == NULL or pq_py == NULL or pq_pz == NULL or pq_w == NULL:
        free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz); free(pq_w)
        raise MemoryError("Could not allocate primitive-pair scratch arrays.")

    nblock = aux_stop - aux_start
    j3 = np.zeros((nblock, npair), dtype=np.float64)
    j3_v = j3

    with nogil:
        pair = 0
        for p in range(nao):
            Ax = origins_v[p, 0]; Ay = origins_v[p, 1]; Az = origins_v[p, 2]
            l1 = <int>shells_v[p, 0]; m1 = <int>shells_v[p, 1]; n1 = <int>shells_v[p, 2]
            for q in range(p + 1):
                pair_bound = pair_bounds_v[p, q]
                Bx = origins_v[q, 0]; By = origins_v[q, 1]; Bz = origins_v[q, 2]
                l2 = <int>shells_v[q, 0]; m2 = <int>shells_v[q, 1]; n2 = <int>shells_v[q, 2]
                abx = Ax - Bx; aby = Ay - By; abz = Az - Bz
                npq = precompute_primitive_pair_data(
                    p, q, shells_v, origins_v, exps_v, weights_v, nprim_v,
                    pq_a, pq_b, pq_p, pq_px, pq_py, pq_pz, pq_w,
                )
                for a in range(aux_start, aux_stop):
                    if screen_tol > 0.0 and pair_bound * aux_diag_v[a] < screen_tol:
                        skipped += 1
                        continue
                    Cx = aux_origins_v[a, 0]; Cy = aux_origins_v[a, 1]; Cz = aux_origins_v[a, 2]
                    l3 = <int>aux_shells_v[a, 0]; m3 = <int>aux_shells_v[a, 1]; n3 = <int>aux_shells_v[a, 2]
                    value = 0.0
                    for ij in range(npq):
                        for ia in range(aux_nprim_v[a]):
                            value += (
                                pq_w[ij]
                                * aux_weights_v[a, ia]
                                * primitive_three_center_precomputed(
                                    pq_a[ij], pq_b[ij], pq_p[ij], pq_px[ij], pq_py[ij], pq_pz[ij], abx, aby, abz,
                                    l1, m1, n1, l2, m2, n2,
                                    aux_exps_v[a, ia], l3, m3, n3, Cx, Cy, Cz,
                                )
                            )
                    local_a = a - aux_start
                    j3_v[local_a, pair] = value
                    computed += 1
                pair += 1

    free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz); free(pq_w)
    return j3, int(computed), int(skipped)


cpdef compute_dense_eri(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef cnp.ndarray[double, ndim=4] eri = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef int64_t npair_pq, npair_rs
    cdef int p, q, r, s, s_max, ij, kl, npq, nrs
    cdef double bound_pq, value
    cdef double Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz
    cdef int l1, m1, n1, l2, m2, n2, l3, m3, n3, l4, m4, n4
    cdef double abx, aby, abz, cdx, cdy, cdz
    cdef double* pq_a = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_b = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_p = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_px = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_py = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_pz = <double*>malloc(pair_cap * sizeof(double))
    cdef double* pq_w = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_a = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_b = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_p = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_px = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_py = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_pz = <double*>malloc(pair_cap * sizeof(double))
    cdef double* rs_w = <double*>malloc(pair_cap * sizeof(double))

    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, :, :, ::1] eri_v = eri

    if pq_a == NULL or pq_b == NULL or pq_p == NULL or pq_px == NULL or pq_py == NULL or pq_pz == NULL or pq_w == NULL or rs_a == NULL or rs_b == NULL or rs_p == NULL or rs_px == NULL or rs_py == NULL or rs_pz == NULL or rs_w == NULL:
        free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz); free(pq_w)
        free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz); free(rs_w)
        return eri, 0, 0

    for p in range(nao):
        for q in range(p + 1):
            bound_pq = pair_bounds_v[p, q]
            npq = precompute_primitive_pair_data(
                p, q, shells_v, origins_v, exps_v, weights_v, nprim_v,
                pq_a, pq_b, pq_p, pq_px, pq_py, pq_pz, pq_w,
            )
            Ax = origins_v[p, 0]; Ay = origins_v[p, 1]; Az = origins_v[p, 2]
            Bx = origins_v[q, 0]; By = origins_v[q, 1]; Bz = origins_v[q, 2]
            l1 = <int>shells_v[p, 0]; m1 = <int>shells_v[p, 1]; n1 = <int>shells_v[p, 2]
            l2 = <int>shells_v[q, 0]; m2 = <int>shells_v[q, 1]; n2 = <int>shells_v[q, 2]
            abx = Ax - Bx; aby = Ay - By; abz = Az - Bz
            for r in range(p + 1):
                s_max = q if r == p else r
                for s in range(s_max + 1):
                    if screen_tol > 0.0 and bound_pq * pair_bounds_v[r, s] < screen_tol:
                        skipped += 1
                        continue

                    nrs = precompute_primitive_pair_data(
                        r, s, shells_v, origins_v, exps_v, weights_v, nprim_v,
                        rs_a, rs_b, rs_p, rs_px, rs_py, rs_pz, rs_w,
                    )
                    Cx = origins_v[r, 0]; Cy = origins_v[r, 1]; Cz = origins_v[r, 2]
                    Dx = origins_v[s, 0]; Dy = origins_v[s, 1]; Dz = origins_v[s, 2]
                    l3 = <int>shells_v[r, 0]; m3 = <int>shells_v[r, 1]; n3 = <int>shells_v[r, 2]
                    l4 = <int>shells_v[s, 0]; m4 = <int>shells_v[s, 1]; n4 = <int>shells_v[s, 2]
                    cdx = Cx - Dx; cdy = Cy - Dy; cdz = Cz - Dz

                    value = 0.0
                    for ij in range(npq):
                        for kl in range(nrs):
                            value += (
                                pq_w[ij] * rs_w[kl]
                                * primitive_eri_precomputed(
                                    pq_a[ij], pq_b[ij], pq_p[ij], pq_px[ij], pq_py[ij], pq_pz[ij], abx, aby, abz,
                                    l1, m1, n1, l2, m2, n2,
                                    rs_a[kl], rs_b[kl], rs_p[kl], rs_px[kl], rs_py[kl], rs_pz[kl], cdx, cdy, cdz,
                                    l3, m3, n3, l4, m4, n4,
                                )
                            )

                    if screen_tol > 0.0 and fabs(value) < screen_tol:
                        skipped += 1
                        continue

                    eri_v[p, q, r, s] = value
                    eri_v[q, p, r, s] = value
                    eri_v[p, q, s, r] = value
                    eri_v[q, p, s, r] = value
                    eri_v[r, s, p, q] = value
                    eri_v[s, r, p, q] = value
                    eri_v[r, s, q, p] = value
                    eri_v[s, r, q, p] = value
                    computed += 1

    free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz); free(pq_w)
    free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz); free(rs_w)
    return eri, int(computed), int(skipped)


cdef inline void write_shell_block_symmetries(
    double[:, :, :, ::1] eri_v,
    double[:, :, :, ::1] block_v,
    int p0,
    int q0,
    int r0,
    int s0,
    int np_,
    int nq_,
    int nr_,
    int ns_,
) noexcept:
    cdef int ia, ib, ic, id_
    cdef int p, q, r, s
    cdef double value

    for ia in range(np_):
        p = p0 + ia
        for ib in range(nq_):
            q = q0 + ib
            for ic in range(nr_):
                r = r0 + ic
                for id_ in range(ns_):
                    s = s0 + id_
                    value = block_v[ia, ib, ic, id_]
                    eri_v[p, q, r, s] = value
                    eri_v[q, p, r, s] = value
                    eri_v[p, q, s, r] = value
                    eri_v[q, p, s, r] = value
                    eri_v[r, s, p, q] = value
                    eri_v[s, r, p, q] = value
                    eri_v[r, s, q, p] = value
                    eri_v[s, r, q, p] = value


cdef inline void write_eri_symmetries(
    double[:, :, :, ::1] eri_v,
    int p,
    int q,
    int r,
    int s,
    double value,
) noexcept:
    eri_v[p, q, r, s] = value
    eri_v[q, p, r, s] = value
    eri_v[p, q, s, r] = value
    eri_v[q, p, s, r] = value
    eri_v[r, s, p, q] = value
    eri_v[s, r, p, q] = value
    eri_v[r, s, q, p] = value
    eri_v[s, r, q, p] = value


cdef inline int eri_same4(
    int a0,
    int a1,
    int a2,
    int a3,
    int b0,
    int b1,
    int b2,
    int b3,
) noexcept nogil:
    return a0 == b0 and a1 == b1 and a2 == b2 and a3 == b3


cdef inline void add_eri_symmetries_unique(
    double[:, :, :, ::1] eri_v,
    int p,
    int q,
    int r,
    int s,
    double value,
) noexcept:
    if p != q and r != s and not (p == r and q == s):
        eri_v[p, q, r, s] += value
        eri_v[q, p, r, s] += value
        eri_v[p, q, s, r] += value
        eri_v[q, p, s, r] += value
        eri_v[r, s, p, q] += value
        eri_v[s, r, p, q] += value
        eri_v[r, s, q, p] += value
        eri_v[s, r, q, p] += value
        return
    if p == q:
        if r == s:
            eri_v[p, p, r, r] += value
            if p != r:
                eri_v[r, r, p, p] += value
            return
        eri_v[p, p, r, s] += value
        eri_v[p, p, s, r] += value
        eri_v[r, s, p, p] += value
        eri_v[s, r, p, p] += value
        return
    if r == s:
        eri_v[p, q, r, r] += value
        eri_v[q, p, r, r] += value
        eri_v[r, r, p, q] += value
        eri_v[r, r, q, p] += value
        return
    if p == r and q == s:
        eri_v[p, q, p, q] += value
        eri_v[q, p, p, q] += value
        eri_v[p, q, q, p] += value
        eri_v[q, p, q, p] += value
        return

    eri_v[p, q, r, s] += value
    if not eri_same4(q, p, r, s, p, q, r, s):
        eri_v[q, p, r, s] += value
    if not eri_same4(p, q, s, r, p, q, r, s) and not eri_same4(p, q, s, r, q, p, r, s):
        eri_v[p, q, s, r] += value
    if not eri_same4(q, p, s, r, p, q, r, s) and not eri_same4(q, p, s, r, q, p, r, s) and not eri_same4(q, p, s, r, p, q, s, r):
        eri_v[q, p, s, r] += value
    if not eri_same4(r, s, p, q, p, q, r, s) and not eri_same4(r, s, p, q, q, p, r, s) and not eri_same4(r, s, p, q, p, q, s, r) and not eri_same4(r, s, p, q, q, p, s, r):
        eri_v[r, s, p, q] += value
    if not eri_same4(s, r, p, q, p, q, r, s) and not eri_same4(s, r, p, q, q, p, r, s) and not eri_same4(s, r, p, q, p, q, s, r) and not eri_same4(s, r, p, q, q, p, s, r) and not eri_same4(s, r, p, q, r, s, p, q):
        eri_v[s, r, p, q] += value
    if not eri_same4(r, s, q, p, p, q, r, s) and not eri_same4(r, s, q, p, q, p, r, s) and not eri_same4(r, s, q, p, p, q, s, r) and not eri_same4(r, s, q, p, q, p, s, r) and not eri_same4(r, s, q, p, r, s, p, q) and not eri_same4(r, s, q, p, s, r, p, q):
        eri_v[r, s, q, p] += value
    if not eri_same4(s, r, q, p, p, q, r, s) and not eri_same4(s, r, q, p, q, p, r, s) and not eri_same4(s, r, q, p, p, q, s, r) and not eri_same4(s, r, q, p, q, p, s, r) and not eri_same4(s, r, q, p, r, s, p, q) and not eri_same4(s, r, q, p, s, r, p, q) and not eri_same4(s, r, q, p, r, s, q, p):
        eri_v[s, r, q, p] += value


cdef int compute_shell_quartet_vrr_hrr_into_eri(
    int64_t[:, ::1] shells_v,
    double[:, ::1] origins_v,
    double[:, ::1] weights_v,
    int64_t[::1] nprim_v,
    double[:, :, :, ::1] eri_v,
    int p0,
    int p1,
    int q0,
    int q1,
    int r0,
    int r1,
    int s0,
    int s1,
    int pair_cap,
    double* pq_a,
    double* pq_b,
    double* pq_p,
    double* pq_px,
    double* pq_py,
    double* pq_pz,
    int npq,
    double* rs_a,
    double* rs_b,
    double* rs_p,
    double* rs_px,
    double* rs_py,
    double* rs_pz,
    int nrs,
    double* vrr_table,
    size_t vrr_table_cap,
) noexcept:
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int nr_ = r1 - r0
    cdef int ns_ = s1 - s0
    cdef int lA, lB, lC, lD, max_a_l, max_c_l, max_m_l
    cdef int ia, ib, ic, id_, idx_pq, idx_rs, ip, iq, ir, is_
    cdef int ao_p, ao_q, ao_r, ao_s
    cdef int64_t pair_pq_ao, pair_rs_ao
    cdef int ax[OS_VRR_MAX_CART]
    cdef int ay[OS_VRR_MAX_CART]
    cdef int az[OS_VRR_MAX_CART]
    cdef int bx[OS_VRR_MAX_CART]
    cdef int by[OS_VRR_MAX_CART]
    cdef int bz[OS_VRR_MAX_CART]
    cdef int cx[OS_VRR_MAX_CART]
    cdef int cy[OS_VRR_MAX_CART]
    cdef int cz[OS_VRR_MAX_CART]
    cdef int dxc[OS_VRR_MAX_CART]
    cdef int dyc[OS_VRR_MAX_CART]
    cdef int dzc[OS_VRR_MAX_CART]
    cdef size_t vrr_table_size
    cdef double abx, aby, abz, cdx, cdy, cdz
    cdef double zeta, alpha, dx, dy, dz, pq2, ab2, cd2, T, base_pref
    cdef double prefac, value
    cdef double PA[3]
    cdef double QC[3]
    cdef double PQ[3]
    cdef double AB[3]
    cdef double CD[3]

    lA = <int>shells_v[p0, 0] + <int>shells_v[p0, 1] + <int>shells_v[p0, 2]
    lB = <int>shells_v[q0, 0] + <int>shells_v[q0, 1] + <int>shells_v[q0, 2]
    lC = <int>shells_v[r0, 0] + <int>shells_v[r0, 1] + <int>shells_v[r0, 2]
    lD = <int>shells_v[s0, 0] + <int>shells_v[s0, 1] + <int>shells_v[s0, 2]
    max_a_l = lA + lB
    max_c_l = lC + lD
    max_m_l = max_a_l + max_c_l
    if max_a_l > OS_VRR_PAIR_MAX_L or max_c_l > OS_VRR_PAIR_MAX_L:
        return 0
    if np_ != ncart_for_l(lA) or nq_ != ncart_for_l(lB) or nr_ != ncart_for_l(lC) or ns_ != ncart_for_l(lD):
        return 0

    vrr_table_size = (
        <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1)
        * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1)
        * <size_t>(max_m_l + 1)
    )
    if vrr_table == NULL or vrr_table_size > vrr_table_cap:
        return 0

    fill_cartesian_components(lA, ax, ay, az)
    fill_cartesian_components(lB, bx, by, bz)
    fill_cartesian_components(lC, cx, cy, cz)
    fill_cartesian_components(lD, dxc, dyc, dzc)

    abx = origins_v[p0, 0] - origins_v[q0, 0]
    aby = origins_v[p0, 1] - origins_v[q0, 1]
    abz = origins_v[p0, 2] - origins_v[q0, 2]
    cdx = origins_v[r0, 0] - origins_v[s0, 0]
    cdy = origins_v[r0, 1] - origins_v[s0, 1]
    cdz = origins_v[r0, 2] - origins_v[s0, 2]
    ab2 = abx * abx + aby * aby + abz * abz
    cd2 = cdx * cdx + cdy * cdy + cdz * cdz
    AB[0] = abx; AB[1] = aby; AB[2] = abz
    CD[0] = cdx; CD[1] = cdy; CD[2] = cdz

    for idx_pq in range(npq):
        ip = idx_pq // <int>nprim_v[q0]
        iq = idx_pq - ip * <int>nprim_v[q0]
        for idx_rs in range(nrs):
            ir = idx_rs // <int>nprim_v[s0]
            is_ = idx_rs - ir * <int>nprim_v[s0]
            zeta = pq_p[idx_pq] + rs_p[idx_rs]
            alpha = pq_p[idx_pq] * rs_p[idx_rs] / zeta
            dx = pq_px[idx_pq] - rs_px[idx_rs]
            dy = pq_py[idx_pq] - rs_py[idx_rs]
            dz = pq_pz[idx_pq] - rs_pz[idx_rs]
            pq2 = dx * dx + dy * dy + dz * dz
            T = alpha * pq2
            base_pref = (
                ERI_PREFAC
                * exp(-(pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]) * ab2)
                * exp(-(rs_a[idx_rs] * rs_b[idx_rs] / rs_p[idx_rs]) * cd2)
                / (pq_p[idx_pq] * rs_p[idx_rs] * sqrt(zeta))
            )
            PA[0] = pq_px[idx_pq] - origins_v[p0, 0]
            PA[1] = pq_py[idx_pq] - origins_v[p0, 1]
            PA[2] = pq_pz[idx_pq] - origins_v[p0, 2]
            QC[0] = rs_px[idx_rs] - origins_v[r0, 0]
            QC[1] = rs_py[idx_rs] - origins_v[r0, 1]
            QC[2] = rs_pz[idx_rs] - origins_v[r0, 2]
            PQ[0] = dx; PQ[1] = dy; PQ[2] = dz
            os_fill_vrr_table(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                rs_p[idx_rs],
                zeta,
                T,
                base_pref,
                PA,
                QC,
                PQ,
            )
            for ia in range(np_):
                ao_p = p0 + ia
                for ib in range(nq_):
                    ao_q = q0 + ib
                    for ic in range(nr_):
                        ao_r = r0 + ic
                        for id_ in range(ns_):
                            ao_s = s0 + id_
                            if ao_p < ao_q:
                                continue
                            if ao_r < ao_s:
                                continue
                            if p0 == r0 and q0 == s0:
                                pair_pq_ao = (<int64_t>ao_p * (<int64_t>ao_p + 1)) // 2 + <int64_t>ao_q
                                pair_rs_ao = (<int64_t>ao_r * (<int64_t>ao_r + 1)) // 2 + <int64_t>ao_s
                                if pair_pq_ao < pair_rs_ao:
                                    continue
                            prefac = (
                                weights_v[ao_p, ip] * weights_v[ao_q, iq]
                                * weights_v[ao_r, ir] * weights_v[ao_s, is_]
                            )
                            value = prefac * os_vrr_hrr_eval_expanded(
                                vrr_table,
                                ax[ia], ay[ia], az[ia],
                                bx[ib], by[ib], bz[ib],
                                cx[ic], cy[ic], cz[ic],
                                dxc[id_], dyc[id_], dzc[id_],
                                0,
                                max_a_l,
                                max_c_l,
                                max_m_l,
                                AB,
                                CD,
                            )
                            add_eri_symmetries_unique(
                                eri_v, ao_p, ao_q, ao_r, ao_s, value
                            )

    return 1


cpdef compute_dense_eri_blocked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef cnp.ndarray[double, ndim=4] eri = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=4] block
    cdef int ish, jsh, ksh, lsh
    cdef int p0, p1, q0, q1, r0, r1, s0, s1
    cdef int ip, iq, ir, is_
    cdef int np_, nq_, nr_, ns_, lsh_max
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef int pair_idx, pq_pair_idx, rs_pair_idx, direct_done
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double bound_pq, bound_rs, value
    cdef double* shell_pair_bounds
    cdef int* shell_pair_n
    cdef double* shell_pair_a
    cdef double* shell_pair_b
    cdef double* shell_pair_p
    cdef double* shell_pair_px
    cdef double* shell_pair_py
    cdef double* shell_pair_pz
    cdef double* direct_vrr_table
    cdef size_t pq_offset, rs_offset, pair_storage_size
    cdef size_t direct_vrr_table_cap = (
        <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(OS_VRR_PAIR_MAX_L + 1) * <size_t>(OS_VRR_PAIR_MAX_L + 1)
        * <size_t>(2 * OS_VRR_PAIR_MAX_L + 1)
    )
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, :, :, ::1] eri_v = eri
    cdef double[:, :, :, ::1] block_v

    shell_pair_bounds = <double*>malloc(nshell * nshell * sizeof(double))
    shell_pair_n = <int*>malloc(nshell * nshell * sizeof(int))
    pair_storage_size = <size_t>nshell * <size_t>nshell * <size_t>pair_cap
    shell_pair_a = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_b = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_p = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_px = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_py = <double*>malloc(pair_storage_size * sizeof(double))
    shell_pair_pz = <double*>malloc(pair_storage_size * sizeof(double))
    direct_vrr_table = <double*>malloc(direct_vrr_table_cap * sizeof(double))
    if shell_pair_bounds == NULL or shell_pair_n == NULL or shell_pair_a == NULL or shell_pair_b == NULL or shell_pair_p == NULL or shell_pair_px == NULL or shell_pair_py == NULL or shell_pair_pz == NULL or direct_vrr_table == NULL:
        free(shell_pair_bounds); free(shell_pair_n)
        free(shell_pair_a); free(shell_pair_b); free(shell_pair_p); free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
        free(direct_vrr_table)
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
            pair_idx = ish * nshell + jsh
            shell_pair_bounds[pair_idx] = bound_pq
            pq_offset = <size_t>pair_idx * <size_t>pair_cap
            shell_pair_n[pair_idx] = precompute_primitive_pair_geom(
                p0,
                q0,
                origins_v,
                exps_v,
                nprim_v,
                shell_pair_a + pq_offset,
                shell_pair_b + pq_offset,
                shell_pair_p + pq_offset,
                shell_pair_px + pq_offset,
                shell_pair_py + pq_offset,
                shell_pair_pz + pq_offset,
            )

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

                    pq_pair_idx = ish * nshell + jsh
                    rs_pair_idx = ksh * nshell + lsh
                    pq_offset = <size_t>pq_pair_idx * <size_t>pair_cap
                    rs_offset = <size_t>rs_pair_idx * <size_t>pair_cap
                    direct_done = compute_shell_quartet_vrr_hrr_into_eri(
                        shells_v,
                        origins_v,
                        weights_v,
                        nprim_v,
                        eri_v,
                        p0,
                        p1,
                        q0,
                        q1,
                        r0,
                        r1,
                        s0,
                        s1,
                        pair_cap,
                        shell_pair_a + pq_offset,
                        shell_pair_b + pq_offset,
                        shell_pair_p + pq_offset,
                        shell_pair_px + pq_offset,
                        shell_pair_py + pq_offset,
                        shell_pair_pz + pq_offset,
                        shell_pair_n[pq_pair_idx],
                        shell_pair_a + rs_offset,
                        shell_pair_b + rs_offset,
                        shell_pair_p + rs_offset,
                        shell_pair_px + rs_offset,
                        shell_pair_py + rs_offset,
                        shell_pair_pz + rs_offset,
                        shell_pair_n[rs_pair_idx],
                        direct_vrr_table,
                        direct_vrr_table_cap,
                    )
                    if direct_done == 0:
                        block = compute_cartesian_shell_quartet_block(
                            shells, origins, exps, weights, nprim,
                            p0, p1, q0, q1, r0, r1, s0, s1,
                        )
                        block_v = block
                        write_shell_block_symmetries(
                            eri_v, block_v,
                            p0, q0, r0, s0,
                            np_, nq_, nr_, ns_,
                        )
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

    free(shell_pair_bounds); free(shell_pair_n)
    free(shell_pair_a); free(shell_pair_b); free(shell_pair_p); free(shell_pair_px); free(shell_pair_py); free(shell_pair_pz)
    free(direct_vrr_table)
    return eri, int(computed), int(skipped)


cpdef compute_eri_s8(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef cnp.ndarray[double, ndim=1] eri_s8 = np.zeros((npair * (npair + 1)) // 2, dtype=np.float64)
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[::1] eri_v = eri_s8
    cdef int p, q, r, s, s_max, ij, kl, pbase, rbase
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double bound_pq

    for p in range(nao):
        pbase = p * (p + 1) // 2
        for q in range(p + 1):
            ij = pbase + q
            bound_pq = pair_bounds_v[p, q]
            for r in range(p + 1):
                rbase = r * (r + 1) // 2
                s_max = q if r == p else r
                for s in range(s_max + 1):
                    kl = rbase + s
                    if screen_tol > 0.0 and bound_pq * pair_bounds_v[r, s] < screen_tol:
                        skipped += 1
                        continue
                    eri_v[ij * (ij + 1) // 2 + kl] = contracted_eri_indices(
                        p, q, r, s,
                        shells_v, origins_v, exps_v, weights_v, nprim_v,
                    )
                    computed += 1

    return eri_s8, int(computed), int(skipped)


cpdef compute_pair_bounds(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
):
    cdef int nao = shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] bounds = np.zeros((nao, nao), dtype=np.float64)
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] bounds_v = bounds
    cdef int p, q
    cdef double value, bound

    with nogil:
        for p in range(nao):
            for q in range(p + 1):
                value = contracted_eri_indices(
                    p, q, p, q,
                    shells_v, origins_v, exps_v, weights_v, nprim_v,
                )
                bound = sqrt(fabs(value))
                bounds_v[p, q] = bound
                bounds_v[q, p] = bound

    return bounds


cpdef compute_eri_s8_blocked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef cnp.ndarray[double, ndim=1] eri_s8 = np.zeros((npair * (npair + 1)) // 2, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=4] block
    cdef double[::1] eri_v = eri_s8
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int ish, jsh, ksh, lsh, lsh_max
    cdef int p0, p1, q0, q1, r0, r1, s0, s1
    cdef int ip, iq, ir, is_, p, q, r, s, ij, kl
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double bound_pq, bound_rs
    cdef double* shell_pair_bounds
    cdef int* shell_pair_min
    cdef int* shell_pair_max
    cdef int pair_pos

    shell_pair_bounds = <double*>malloc(nshell * nshell * sizeof(double))
    shell_pair_min = <int*>malloc(nshell * nshell * sizeof(int))
    shell_pair_max = <int*>malloc(nshell * nshell * sizeof(int))
    if shell_pair_bounds == NULL or shell_pair_min == NULL or shell_pair_max == NULL:
        free(shell_pair_bounds)
        free(shell_pair_min)
        free(shell_pair_max)
        return eri_s8, 0, 0

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        for jsh in range(nshell):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            bound_pq = 0.0
            shell_pair_min[ish * nshell + jsh] = npair
            shell_pair_max[ish * nshell + jsh] = -1
            for ip in range(p0, p1):
                for iq in range(q0, q1):
                    if pair_bounds_v[ip, iq] > bound_pq:
                        bound_pq = pair_bounds_v[ip, iq]
                    if ip >= iq:
                        pair_pos = pair_index(ip, iq)
                        if pair_pos < shell_pair_min[ish * nshell + jsh]:
                            shell_pair_min[ish * nshell + jsh] = pair_pos
                        if pair_pos > shell_pair_max[ish * nshell + jsh]:
                            shell_pair_max[ish * nshell + jsh] = pair_pos
            shell_pair_bounds[ish * nshell + jsh] = bound_pq

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        for jsh in range(ish + 1):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            bound_pq = shell_pair_bounds[ish * nshell + jsh]

            for ksh in range(nshell):
                r0 = <int>shell_starts_v[ksh]
                r1 = <int>shell_stops_v[ksh]
                for lsh in range(ksh + 1):
                    s0 = <int>shell_starts_v[lsh]
                    s1 = <int>shell_stops_v[lsh]
                    bound_rs = shell_pair_bounds[ksh * nshell + lsh]
                    if shell_pair_max[ish * nshell + jsh] < shell_pair_min[ksh * nshell + lsh]:
                        continue
                    if screen_tol > 0.0 and bound_pq * bound_rs < screen_tol:
                        skipped += 1
                        continue

                    block = compute_cartesian_shell_quartet_block(
                        shells, origins, exps, weights, nprim,
                        p0, p1, q0, q1, r0, r1, s0, s1,
                    )
                    for ip in range(p1 - p0):
                        p = p0 + ip
                        for iq in range(q1 - q0):
                            q = q0 + iq
                            if p < q:
                                continue
                            ij = pair_index(p, q)
                            for ir in range(r1 - r0):
                                r = r0 + ir
                                for is_ in range(s1 - s0):
                                    s = s0 + is_
                                    if r < s:
                                        continue
                                    kl = pair_index(r, s)
                                    if ij < kl:
                                        continue
                                    eri_v[pair_pair_index(ij, kl)] = block[ip, iq, ir, is_]
                                    computed += 1

    free(shell_pair_bounds)
    free(shell_pair_min)
    free(shell_pair_max)
    return eri_s8, int(computed), int(skipped)


cpdef compute_cartesian_shell_quartet_block(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    int p0,
    int p1,
    int q0,
    int q1,
    int r0,
    int r1,
    int s0,
    int s1,
    bint use_iterative=False,
):
    cdef int np_ = p1 - p0
    cdef int nq_ = q1 - q0
    cdef int nr_ = r1 - r0
    cdef int ns_ = s1 - s0
    cdef cnp.ndarray[double, ndim=4] block = np.zeros((np_, nq_, nr_, ns_), dtype=np.float64)
    cdef int ia, ib, ic, id_, ij, kl, npq, nrs
    cdef int lA, lB, lC, lD
    cdef int tdim_ab, tdim_cd, tx_max, uy_max, vz_max, nmax
    cdef size_t size_ab, size_cd, size_r, i
    cdef double abx, aby, abz, cdx, cdy, cdz
    cdef double alpha, dx, dy, dz, rpq, prefac, value
    cdef double zeta, T, base_pref, ab2, cd2, pq2
    cdef double* pq_a
    cdef double* pq_b
    cdef double* pq_p
    cdef double* pq_px
    cdef double* pq_py
    cdef double* pq_pz
    cdef int ip, iq, ir, is_, idx_pq, idx_rs
    cdef double* rs_a
    cdef double* rs_b
    cdef double* rs_p
    cdef double* rs_px
    cdef double* rs_py
    cdef double* rs_pz
    cdef uint64_t* os_keys
    cdef double* os_values
    cdef int* os_stack_ang = NULL
    cdef int* os_stack_m = NULL
    cdef double* vrr_table = NULL
    cdef int os_nstates[1]
    cdef int os_ang[12]
    cdef double PA[3]
    cdef double QC[3]
    cdef double PQ[3]
    cdef double AB[3]
    cdef double CD[3]
    cdef int* ax
    cdef int* ay
    cdef int* az
    cdef int* bx
    cdef int* by
    cdef int* bz
    cdef int* cx
    cdef int* cy
    cdef int* cz
    cdef int* dxc
    cdef int* dyc
    cdef int* dzc
    cdef int max_prim = exps.shape[1]
    cdef int pair_cap = max_prim * max_prim
    cdef int ao_p, ao_q, ao_r, ao_s
    cdef int max_a_l, max_c_l, max_m_l, use_vrr_hrr
    cdef size_t vrr_table_size
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, :, :, ::1] block_v = block
    lA = <int>shells_v[p0, 0] + <int>shells_v[p0, 1] + <int>shells_v[p0, 2]
    lB = <int>shells_v[q0, 0] + <int>shells_v[q0, 1] + <int>shells_v[q0, 2]
    lC = <int>shells_v[r0, 0] + <int>shells_v[r0, 1] + <int>shells_v[r0, 2]
    lD = <int>shells_v[s0, 0] + <int>shells_v[s0, 1] + <int>shells_v[s0, 2]
    max_a_l = lA + lB
    max_c_l = lC + lD
    max_m_l = max_a_l + max_c_l
    use_vrr_hrr = (not use_iterative) and max_a_l <= OS_VRR_PAIR_MAX_L and max_c_l <= OS_VRR_PAIR_MAX_L
    if np_ != ncart_for_l(lA) or nq_ != ncart_for_l(lB) or nr_ != ncart_for_l(lC) or ns_ != ncart_for_l(lD):
        return block

    ax = <int*>malloc(np_ * sizeof(int)); ay = <int*>malloc(np_ * sizeof(int)); az = <int*>malloc(np_ * sizeof(int))
    bx = <int*>malloc(nq_ * sizeof(int)); by = <int*>malloc(nq_ * sizeof(int)); bz = <int*>malloc(nq_ * sizeof(int))
    cx = <int*>malloc(nr_ * sizeof(int)); cy = <int*>malloc(nr_ * sizeof(int)); cz = <int*>malloc(nr_ * sizeof(int))
    dxc = <int*>malloc(ns_ * sizeof(int)); dyc = <int*>malloc(ns_ * sizeof(int)); dzc = <int*>malloc(ns_ * sizeof(int))
    pq_a = <double*>malloc(pair_cap * sizeof(double)); pq_b = <double*>malloc(pair_cap * sizeof(double))
    pq_p = <double*>malloc(pair_cap * sizeof(double)); pq_px = <double*>malloc(pair_cap * sizeof(double))
    pq_py = <double*>malloc(pair_cap * sizeof(double)); pq_pz = <double*>malloc(pair_cap * sizeof(double))
    rs_a = <double*>malloc(pair_cap * sizeof(double)); rs_b = <double*>malloc(pair_cap * sizeof(double))
    rs_p = <double*>malloc(pair_cap * sizeof(double)); rs_px = <double*>malloc(pair_cap * sizeof(double))
    rs_py = <double*>malloc(pair_cap * sizeof(double)); rs_pz = <double*>malloc(pair_cap * sizeof(double))
    os_keys = <uint64_t*>malloc(OS_HASH_CAP * sizeof(uint64_t))
    os_values = <double*>malloc(OS_HASH_CAP * sizeof(double))
    if use_iterative:
        os_stack_ang = <int*>malloc(OS_HASH_CAP * 12 * sizeof(int))
        os_stack_m = <int*>malloc(OS_HASH_CAP * sizeof(int))
    if use_vrr_hrr:
        vrr_table_size = (
            <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1) * <size_t>(max_a_l + 1)
            * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1) * <size_t>(max_c_l + 1)
            * <size_t>(max_m_l + 1)
        )
        vrr_table = <double*>malloc(vrr_table_size * sizeof(double))
    if ax == NULL or ay == NULL or az == NULL or bx == NULL or by == NULL or bz == NULL or cx == NULL or cy == NULL or cz == NULL or dxc == NULL or dyc == NULL or dzc == NULL or pq_a == NULL or pq_b == NULL or pq_p == NULL or pq_px == NULL or pq_py == NULL or pq_pz == NULL or rs_a == NULL or rs_b == NULL or rs_p == NULL or rs_px == NULL or rs_py == NULL or rs_pz == NULL or os_keys == NULL or os_values == NULL or (use_iterative and (os_stack_ang == NULL or os_stack_m == NULL)) or (use_vrr_hrr and vrr_table == NULL):
        free(ax); free(ay); free(az); free(bx); free(by); free(bz); free(cx); free(cy); free(cz); free(dxc); free(dyc); free(dzc)
        free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz)
        free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz)
        free(os_keys); free(os_values); free(os_stack_ang); free(os_stack_m); free(vrr_table)
        return block

    fill_cartesian_components(lA, ax, ay, az)
    fill_cartesian_components(lB, bx, by, bz)
    fill_cartesian_components(lC, cx, cy, cz)
    fill_cartesian_components(lD, dxc, dyc, dzc)

    npq = precompute_primitive_pair_geom(p0, q0, origins_v, exps_v, nprim_v, pq_a, pq_b, pq_p, pq_px, pq_py, pq_pz)
    nrs = precompute_primitive_pair_geom(r0, s0, origins_v, exps_v, nprim_v, rs_a, rs_b, rs_p, rs_px, rs_py, rs_pz)

    abx = origins_v[p0, 0] - origins_v[q0, 0]
    aby = origins_v[p0, 1] - origins_v[q0, 1]
    abz = origins_v[p0, 2] - origins_v[q0, 2]
    cdx = origins_v[r0, 0] - origins_v[s0, 0]
    cdy = origins_v[r0, 1] - origins_v[s0, 1]
    cdz = origins_v[r0, 2] - origins_v[s0, 2]
    idx_pq = 0
    for ip in range(nprim_v[p0]):
        for iq in range(nprim_v[q0]):
            idx_rs = 0
            for ir in range(nprim_v[r0]):
                for is_ in range(nprim_v[s0]):
                    zeta = pq_p[idx_pq] + rs_p[idx_rs]
                    alpha = pq_p[idx_pq] * rs_p[idx_rs] / zeta
                    dx = pq_px[idx_pq] - rs_px[idx_rs]
                    dy = pq_py[idx_pq] - rs_py[idx_rs]
                    dz = pq_pz[idx_pq] - rs_pz[idx_rs]
                    pq2 = dx * dx + dy * dy + dz * dz
                    T = alpha * pq2
                    ab2 = abx * abx + aby * aby + abz * abz
                    cd2 = cdx * cdx + cdy * cdy + cdz * cdz
                    base_pref = (
                        ERI_PREFAC
                        * exp(-(pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]) * ab2)
                        * exp(-(rs_a[idx_rs] * rs_b[idx_rs] / rs_p[idx_rs]) * cd2)
                        / (pq_p[idx_pq] * rs_p[idx_rs] * sqrt(zeta))
                    )
                    PA[0] = pq_px[idx_pq] - origins_v[p0, 0]
                    PA[1] = pq_py[idx_pq] - origins_v[p0, 1]
                    PA[2] = pq_pz[idx_pq] - origins_v[p0, 2]
                    QC[0] = rs_px[idx_rs] - origins_v[r0, 0]
                    QC[1] = rs_py[idx_rs] - origins_v[r0, 1]
                    QC[2] = rs_pz[idx_rs] - origins_v[r0, 2]
                    PQ[0] = dx; PQ[1] = dy; PQ[2] = dz
                    AB[0] = abx; AB[1] = aby; AB[2] = abz
                    CD[0] = cdx; CD[1] = cdy; CD[2] = cdz
                    if use_vrr_hrr:
                        os_fill_vrr_table(
                            vrr_table,
                            max_a_l,
                            max_c_l,
                            max_m_l,
                            pq_p[idx_pq],
                            rs_p[idx_rs],
                            zeta,
                            T,
                            base_pref,
                            PA,
                            QC,
                            PQ,
                        )
                    else:
                        os_nstates[0] = 0
                        memset(os_keys, 0, OS_HASH_CAP * sizeof(uint64_t))
                        os_seed_boys_hash(
                            max_m_l, T, base_pref, os_keys, os_values, os_nstates
                        )

                    for ia in range(np_):
                        ao_p = p0 + ia
                        for ib in range(nq_):
                            ao_q = q0 + ib
                            for ic in range(nr_):
                                ao_r = r0 + ic
                                for id_ in range(ns_):
                                    ao_s = s0 + id_
                                    prefac = (
                                        weights_v[ao_p, ip] * weights_v[ao_q, iq]
                                        * weights_v[ao_r, ir] * weights_v[ao_s, is_]
                                    )
                                    os_ang[0] = ax[ia]; os_ang[1] = ay[ia]; os_ang[2] = az[ia]
                                    os_ang[3] = bx[ib]; os_ang[4] = by[ib]; os_ang[5] = bz[ib]
                                    os_ang[6] = cx[ic]; os_ang[7] = cy[ic]; os_ang[8] = cz[ic]
                                    os_ang[9] = dxc[id_]; os_ang[10] = dyc[id_]; os_ang[11] = dzc[id_]
                                    if use_vrr_hrr:
                                        block_v[ia, ib, ic, id_] += prefac * os_vrr_hrr_eval(
                                            vrr_table,
                                            os_ang[0],
                                            os_ang[1],
                                            os_ang[2],
                                            os_ang[3],
                                            os_ang[4],
                                            os_ang[5],
                                            os_ang[6],
                                            os_ang[7],
                                            os_ang[8],
                                            os_ang[9],
                                            os_ang[10],
                                            os_ang[11],
                                            0,
                                            max_a_l,
                                            max_c_l,
                                            max_m_l,
                                            AB,
                                            CD,
                                        )
                                    elif use_iterative:
                                        block_v[ia, ib, ic, id_] += prefac * os_eri_eval_iterative(
                                            os_ang,
                                            0,
                                            pq_p[idx_pq],
                                            rs_p[idx_rs],
                                            zeta,
                                            T,
                                            base_pref,
                                            PA,
                                            QC,
                                            PQ,
                                            AB,
                                            CD,
                                            os_keys,
                                            os_values,
                                            os_nstates,
                                            os_stack_ang,
                                            os_stack_m,
                                        )
                                    else:
                                        block_v[ia, ib, ic, id_] += prefac * os_eri_rec_generic(
                                            os_ang,
                                            0,
                                            pq_p[idx_pq],
                                            rs_p[idx_rs],
                                            zeta,
                                            T,
                                            base_pref,
                                            PA,
                                            QC,
                                            PQ,
                                            AB,
                                            CD,
                                            os_keys,
                                            os_values,
                                            os_nstates,
                                        )
                    idx_rs += 1
            idx_pq += 1

    free(ax); free(ay); free(az); free(bx); free(by); free(bz); free(cx); free(cy); free(cz); free(dxc); free(dyc); free(dzc)
    free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz)
    free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz)
    free(os_keys); free(os_values); free(os_stack_ang); free(os_stack_m); free(vrr_table)

    return block


cpdef compute_one_electron(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] atcoords,
    cnp.ndarray[double, ndim=1] atnums,
):
    cdef int nao = shells.shape[0]
    cdef cnp.ndarray[double, ndim=2] overlap = np.eye(nao, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] kinetic = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vnuc = np.zeros((nao, nao), dtype=np.float64)
    cdef int p, q
    cdef double s, t, v

    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] atcoords_v = atcoords
    cdef double[::1] atnums_v = atnums
    cdef double[:, ::1] overlap_v = overlap
    cdef double[:, ::1] kinetic_v = kinetic
    cdef double[:, ::1] vnuc_v = vnuc

    for p in range(nao):
        kinetic_v[p, p] = contracted_kinetic_indices(p, p, shells_v, origins_v, exps_v, weights_v, nprim_v)
        vnuc_v[p, p] = contracted_nuclear_indices(p, p, shells_v, origins_v, exps_v, weights_v, nprim_v, atcoords_v, atnums_v)
        for q in range(p):
            s = contracted_overlap_indices(p, q, shells_v, origins_v, exps_v, weights_v, nprim_v)
            t = contracted_kinetic_indices(p, q, shells_v, origins_v, exps_v, weights_v, nprim_v)
            v = contracted_nuclear_indices(p, q, shells_v, origins_v, exps_v, weights_v, nprim_v, atcoords_v, atnums_v)
            overlap_v[p, q] = overlap_v[q, p] = s
            kinetic_v[p, q] = kinetic_v[q, p] = t
            vnuc_v[p, q] = vnuc_v[q, p] = v

    return overlap, kinetic, vnuc


cpdef compute_periodic_one_electron(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=3] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] atcoords,
    cnp.ndarray[double, ndim=1] atnums,
    double eta,
    image_pair_mask=None,
    double nuclear_screen_tol=0.0,
    lattice=None,
    nuclear_image_keys=None,
    right_shells=None,
    right_exps=None,
    right_weights=None,
    right_nprim=None,
):
    cdef int nimages = right_origins.shape[0]
    cdef int nleft = shells.shape[0]
    cdef int nright
    cdef cnp.ndarray[int64_t, ndim=2] right_shells_arr
    cdef cnp.ndarray[double, ndim=2] right_exps_arr
    cdef cnp.ndarray[double, ndim=2] right_weights_arr
    cdef cnp.ndarray[int64_t, ndim=1] right_nprim_arr
    cdef cnp.ndarray[double, ndim=3] overlap
    cdef cnp.ndarray[double, ndim=3] kinetic
    cdef cnp.ndarray[double, ndim=3] vnuc
    cdef int image, p, q
    cdef double s, t, v
    cdef int64_t[:, ::1] shells_v = shells
    cdef int64_t[:, ::1] right_shells_v
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, :, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] right_exps_v
    cdef double[:, ::1] weights_v = weights
    cdef double[:, ::1] right_weights_v
    cdef int64_t[::1] nprim_v = nprim
    cdef int64_t[::1] right_nprim_v
    cdef double[:, ::1] atcoords_v = atcoords
    cdef double[::1] atnums_v = atnums
    cdef double[:, :, ::1] overlap_v
    cdef double[:, :, ::1] kinetic_v
    cdef double[:, :, ::1] vnuc_v
    cdef cnp.ndarray[uint8_t, ndim=3] image_pair_mask_arr
    cdef cnp.ndarray[double, ndim=2] lattice_arr
    cdef cnp.ndarray[double, ndim=2] inverse_lattice_arr
    cdef cnp.ndarray[int64_t, ndim=2] nuclear_image_keys_arr
    cdef uint8_t[:, :, ::1] image_pair_mask_v
    cdef double[:, ::1] lattice_v
    cdef double[:, ::1] inverse_lattice_v
    cdef int64_t[:, ::1] nuclear_image_keys_v
    cdef bint has_image_pair_mask = image_pair_mask is not None
    cdef bint use_lattice_images = lattice is not None or nuclear_image_keys is not None
    cdef bint has_right_basis = (
        right_shells is not None
        or right_exps is not None
        or right_weights is not None
        or right_nprim is not None
    )

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if has_right_basis:
        if (
            right_shells is None
            or right_exps is None
            or right_weights is None
            or right_nprim is None
        ):
            raise ValueError("All right-basis arrays must be provided together.")
        right_shells_arr = np.ascontiguousarray(right_shells, dtype=np.int64)
        right_exps_arr = np.ascontiguousarray(right_exps, dtype=np.float64)
        right_weights_arr = np.ascontiguousarray(right_weights, dtype=np.float64)
        right_nprim_arr = np.ascontiguousarray(right_nprim, dtype=np.int64)
    else:
        right_shells_arr = np.ascontiguousarray(shells, dtype=np.int64)
        right_exps_arr = np.ascontiguousarray(exps, dtype=np.float64)
        right_weights_arr = np.ascontiguousarray(weights, dtype=np.float64)
        right_nprim_arr = np.ascontiguousarray(nprim, dtype=np.int64)
    nright = right_shells_arr.shape[0]
    if left_origins.shape[0] != nleft or right_origins.shape[1] != nright:
        raise ValueError("Periodic origin arrays do not match the AO basis.")
    if (
        right_exps_arr.shape[0] != nright
        or right_weights_arr.shape[0] != nright
        or right_nprim_arr.shape[0] != nright
    ):
        raise ValueError("Right-basis arrays have inconsistent lengths.")
    if atcoords.shape[0] != atnums.shape[0]:
        raise ValueError("atcoords and atnums have inconsistent lengths.")
    if nuclear_screen_tol < 0.0:
        raise ValueError("nuclear_screen_tol must be non-negative.")
    overlap = np.zeros((nimages, nleft, nright), dtype=np.float64)
    kinetic = np.zeros((nimages, nleft, nright), dtype=np.float64)
    vnuc = np.zeros((nimages, nleft, nright), dtype=np.float64)
    right_shells_v = right_shells_arr
    right_exps_v = right_exps_arr
    right_weights_v = right_weights_arr
    right_nprim_v = right_nprim_arr
    overlap_v = overlap
    kinetic_v = kinetic
    vnuc_v = vnuc
    if has_image_pair_mask:
        image_pair_mask_arr = np.ascontiguousarray(image_pair_mask, dtype=np.uint8)
        if (
            image_pair_mask_arr.shape[0] != nimages
            or image_pair_mask_arr.shape[1] != nleft
            or image_pair_mask_arr.shape[2] != nright
        ):
            raise ValueError("image_pair_mask has an inconsistent shape.")
        image_pair_mask_v = image_pair_mask_arr
    if use_lattice_images:
        if lattice is None or nuclear_image_keys is None:
            raise ValueError(
                "lattice and nuclear_image_keys must be provided together."
            )
        lattice_arr = np.ascontiguousarray(lattice, dtype=np.float64)
        nuclear_image_keys_arr = np.ascontiguousarray(
            nuclear_image_keys, dtype=np.int64
        )
        if lattice_arr.shape[0] != 3 or lattice_arr.shape[1] != 3:
            raise ValueError("lattice must have shape (3, 3).")
        if (
            nuclear_image_keys_arr.shape[1] != 3
            or nuclear_image_keys_arr.shape[0] == 0
        ):
            raise ValueError("nuclear_image_keys must have shape (nimage, 3).")
        inverse_lattice_arr = np.ascontiguousarray(
            np.linalg.inv(lattice_arr), dtype=np.float64
        )
    else:
        lattice_arr = np.zeros((3, 3), dtype=np.float64)
        inverse_lattice_arr = np.zeros((3, 3), dtype=np.float64)
        nuclear_image_keys_arr = np.zeros((1, 3), dtype=np.int64)
    lattice_v = lattice_arr
    inverse_lattice_v = inverse_lattice_arr
    nuclear_image_keys_v = nuclear_image_keys_arr

    with nogil:
        for image in range(nimages):
            for p in range(nleft):
                for q in range(nright):
                    if has_image_pair_mask and image_pair_mask_v[image, p, q] == 0:
                        continue
                    contracted_periodic_one_electron_indices(
                        p,
                        q,
                        shells_v,
                        right_shells_v,
                        left_origins_v,
                        right_origins_v[image],
                        exps_v,
                        right_exps_v,
                        weights_v,
                        right_weights_v,
                        nprim_v,
                        right_nprim_v,
                        atcoords_v,
                        atnums_v,
                        lattice_v,
                        inverse_lattice_v,
                        nuclear_image_keys_v,
                        use_lattice_images,
                        eta,
                        nuclear_screen_tol,
                        &s,
                        &t,
                        &v,
                    )
                    overlap_v[image, p, q] = s
                    kinetic_v[image, p, q] = t
                    vnuc_v[image, p, q] = v

    return overlap, kinetic, vnuc


cpdef compute_periodic_gth_local_gaussian(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] left_origins,
    cnp.ndarray[double, ndim=3] right_origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pseudo_coords,
    cnp.ndarray[double, ndim=1] pseudo_radii,
    cnp.ndarray[double, ndim=2] local_coefficients,
    cnp.ndarray[int64_t, ndim=1] nlocal,
):
    cdef int nimages = right_origins.shape[0]
    cdef int nao = shells.shape[0]
    cdef int npseudo = pseudo_coords.shape[0]
    cdef cnp.ndarray[double, ndim=3] values = np.zeros(
        (nimages, nao, nao), dtype=np.float64
    )
    cdef int image, p, q, atom, ip, iq
    cdef double value, primitive_weight
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] left_origins_v = left_origins
    cdef double[:, :, ::1] right_origins_v = right_origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pseudo_coords_v = pseudo_coords
    cdef double[::1] pseudo_radii_v = pseudo_radii
    cdef double[:, ::1] local_coefficients_v = local_coefficients
    cdef int64_t[::1] nlocal_v = nlocal
    cdef double[:, :, ::1] values_v = values

    if left_origins.shape[0] != nao or right_origins.shape[1] != nao:
        raise ValueError("Periodic origin arrays do not match the AO basis.")
    if pseudo_radii.shape[0] != npseudo or local_coefficients.shape[0] != npseudo or nlocal.shape[0] != npseudo:
        raise ValueError("Inconsistent GTH local-potential arrays.")
    if local_coefficients.shape[1] < 4:
        raise ValueError("local_coefficients must have at least four columns.")

    with nogil:
        for image in range(nimages):
            for p in range(nao):
                for q in range(nao):
                    value = 0.0
                    for atom in range(npseudo):
                        for ip in range(nprim_v[p]):
                            for iq in range(nprim_v[q]):
                                primitive_weight = weights_v[p, ip] * weights_v[q, iq]
                                value += primitive_weight * primitive_gth_local_gaussian(
                                    exps_v[p, ip],
                                    <int>shells_v[p, 0], <int>shells_v[p, 1], <int>shells_v[p, 2],
                                    left_origins_v[p, 0], left_origins_v[p, 1], left_origins_v[p, 2],
                                    exps_v[q, iq],
                                    <int>shells_v[q, 0], <int>shells_v[q, 1], <int>shells_v[q, 2],
                                    right_origins_v[image, q, 0], right_origins_v[image, q, 1], right_origins_v[image, q, 2],
                                    pseudo_coords_v[atom, 0], pseudo_coords_v[atom, 1], pseudo_coords_v[atom, 2],
                                    pseudo_radii_v[atom], &local_coefficients_v[atom, 0], <int>nlocal_v[atom],
                                )
                    values_v[image, p, q] = value

    return values


cpdef compute_pivoted_cholesky_factors(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    double tol=1e-8,
    object max_rank=None,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef int rank_limit = npair if max_rank is None else min(int(max_rank), npair)
    cdef cnp.ndarray[int64_t, ndim=2] pairs = np.zeros((npair, 2), dtype=np.int64)
    cdef cnp.ndarray[double, ndim=1] diag = np.zeros(npair, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] chol = np.zeros((npair, rank_limit), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] col = np.zeros(npair, dtype=np.float64)
    cdef int p, q, pair_idx, pivot, pi, pj, i, j, k, rank = 0
    cdef double delta, pivot_bound, value, correction, inv_sqrt_delta

    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef int64_t[:, ::1] pairs_v = pairs
    cdef double[::1] diag_v = diag
    cdef double[:, ::1] chol_v = chol
    cdef double[::1] col_v = col

    pair_idx = 0
    for p in range(nao):
        for q in range(p + 1):
            pairs_v[pair_idx, 0] = p
            pairs_v[pair_idx, 1] = q
            value = pair_bounds_v[p, q]
            diag_v[pair_idx] = value * value if value > 0.0 else 0.0
            pair_idx += 1

    while rank < rank_limit:
        pivot = 0
        delta = diag_v[0] if npair > 0 else 0.0
        for pair_idx in range(1, npair):
            if diag_v[pair_idx] > delta:
                delta = diag_v[pair_idx]
                pivot = pair_idx

        if delta <= tol:
            break

        pi = <int>pairs_v[pivot, 0]
        pj = <int>pairs_v[pivot, 1]
        pivot_bound = pair_bounds_v[pi, pj]

        for pair_idx in range(npair):
            col_v[pair_idx] = 0.0
            i = <int>pairs_v[pair_idx, 0]
            j = <int>pairs_v[pair_idx, 1]
            if screen_tol > 0.0 and pivot_bound * pair_bounds_v[i, j] < screen_tol:
                continue
            col_v[pair_idx] = contracted_eri_indices(
                i, j, pi, pj,
                shells_v, origins_v, exps_v, weights_v, nprim_v,
            )

        if rank > 0:
            for pair_idx in range(npair):
                correction = 0.0
                for k in range(rank):
                    correction += chol_v[pair_idx, k] * chol_v[pivot, k]
                col_v[pair_idx] -= correction

        delta = col_v[pivot]
        if delta <= tol:
            diag_v[pivot] = 0.0
            continue

        inv_sqrt_delta = 1.0 / sqrt(delta)
        for pair_idx in range(npair):
            value = col_v[pair_idx] * inv_sqrt_delta
            chol_v[pair_idx, rank] = value
            diag_v[pair_idx] -= value * value
            if diag_v[pair_idx] < 0.0:
                diag_v[pair_idx] = 0.0
        rank += 1

    return np.asarray(chol[:, :rank].T, dtype=np.float64), np.asarray(pairs, dtype=np.int64)


cpdef contract_jk_s4(
    cnp.ndarray[double, ndim=2] eri_s4,
    cnp.ndarray[double, ndim=2] dm,
    int nao,
):
    cdef int npair = nao * (nao + 1) // 2
    if eri_s4.shape[0] != npair or eri_s4.shape[1] != npair:
        raise ValueError("eri_s4 shape is inconsistent with nao.")
    if dm.shape[0] != nao or dm.shape[1] != nao:
        raise ValueError("dm shape is inconsistent with nao.")

    cdef cnp.ndarray[double, ndim=2] vj = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vk = np.zeros((nao, nao), dtype=np.float64)
    cdef double[:, ::1] eri_v = eri_s4
    cdef double[:, ::1] dm_v = dm
    cdef double[:, ::1] vj_v = vj
    cdef double[:, ::1] vk_v = vk
    cdef int i, j, k, l, ij, kl, il, kj
    cdef double total, dkl

    ij = 0
    for i in range(nao):
        for j in range(i + 1):
            total = 0.0
            kl = 0
            for k in range(nao):
                for l in range(k + 1):
                    dkl = dm_v[l, k]
                    if k != l:
                        dkl += dm_v[k, l]
                    total += eri_v[ij, kl] * dkl
                    kl += 1
            vj_v[i, j] = total
            vj_v[j, i] = total
            ij += 1

    for i in range(nao):
        for j in range(nao):
            total = 0.0
            for k in range(nao):
                kj = pair_index(k, j)
                for l in range(nao):
                    il = pair_index(i, l)
                    total += dm_v[l, k] * eri_v[il, kj]
            vk_v[i, j] = total

    return vj, vk


cpdef contract_jk_s8(
    cnp.ndarray[double, ndim=1] eri_s8,
    cnp.ndarray[double, ndim=2] dm,
    int nao,
):
    cdef int npair = nao * (nao + 1) // 2
    if eri_s8.shape[0] != npair * (npair + 1) // 2:
        raise ValueError("eri_s8 shape is inconsistent with nao.")
    if dm.shape[0] != nao or dm.shape[1] != nao:
        raise ValueError("dm shape is inconsistent with nao.")

    cdef cnp.ndarray[double, ndim=2] vj = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vk = np.zeros((nao, nao), dtype=np.float64)
    cdef double[::1] eri_v = eri_s8
    cdef double[:, ::1] dm_v = dm
    cdef double[:, ::1] vj_v = vj
    cdef double[:, ::1] vk_v = vk
    cdef int i, j, k, l, ij, kl, il, kj
    cdef double total, dkl

    ij = 0
    for i in range(nao):
        for j in range(i + 1):
            total = 0.0
            kl = 0
            for k in range(nao):
                for l in range(k + 1):
                    dkl = dm_v[l, k]
                    if k != l:
                        dkl += dm_v[k, l]
                    total += eri_v[pair_pair_index(ij, kl)] * dkl
                    kl += 1
            vj_v[i, j] = total
            vj_v[j, i] = total
            ij += 1

    for i in range(nao):
        for j in range(nao):
            total = 0.0
            for k in range(nao):
                kj = pair_index(k, j)
                for l in range(nao):
                    il = pair_index(i, l)
                    total += dm_v[l, k] * eri_v[pair_pair_index(il, kj)]
            vk_v[i, j] = total

    return vj, vk


cpdef contract_jk_ri_pairs(
    cnp.ndarray[double, ndim=2] factors,
    cnp.ndarray[double, ndim=2] dm,
    int nao,
):
    cdef int naux = factors.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    if factors.shape[1] != npair:
        raise ValueError("RI pair-factor shape is inconsistent with nao.")
    if dm.shape[0] != nao or dm.shape[1] != nao:
        raise ValueError("dm shape is inconsistent with nao.")

    cdef cnp.ndarray[double, ndim=2] vj = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vk = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] coeff = np.zeros((naux,), dtype=np.float64)
    cdef double[:, ::1] fac_v = factors
    cdef double[:, ::1] dm_v = dm
    cdef double[:, ::1] vj_v = vj
    cdef double[:, ::1] vk_v = vk
    cdef double[::1] coeff_v = coeff
    cdef int p, i, j, k, l, ij, il, kj
    cdef double dkl, total

    for p in range(naux):
        ij = 0
        total = 0.0
        for i in range(nao):
            for j in range(i + 1):
                dkl = dm_v[j, i]
                if i != j:
                    dkl += dm_v[i, j]
                total += fac_v[p, ij] * dkl
                ij += 1
        coeff_v[p] = total

    ij = 0
    for i in range(nao):
        for j in range(i + 1):
            total = 0.0
            for p in range(naux):
                total += coeff_v[p] * fac_v[p, ij]
            vj_v[i, j] = total
            vj_v[j, i] = total
            ij += 1

    for i in range(nao):
        for j in range(nao):
            total = 0.0
            for p in range(naux):
                for l in range(nao):
                    il = pair_index(i, l)
                    for k in range(nao):
                        kj = pair_index(k, j)
                        total += fac_v[p, il] * dm_v[l, k] * fac_v[p, kj]
            vk_v[i, j] = total

    return vj, vk


cpdef direct_jk(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[double, ndim=2] dm,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    if dm.shape[0] != nao or dm.shape[1] != nao:
        raise ValueError("dm shape is inconsistent with the AO basis.")
    if pair_bounds.shape[0] != nao or pair_bounds.shape[1] != nao:
        raise ValueError("pair_bounds shape is inconsistent with the AO basis.")

    cdef cnp.ndarray[double, ndim=2] vj = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vk = np.zeros((nao, nao), dtype=np.float64)
    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, ::1] dm_v = dm
    cdef double[:, ::1] vj_v = vj
    cdef double[:, ::1] vk_v = vk
    cdef int p, q, r, s, s_max, nperm
    cdef int aa[8]
    cdef int bb[8]
    cdef int cc[8]
    cdef int dd[8]
    cdef double value, bound_pq

    for p in range(nao):
        for q in range(p + 1):
            bound_pq = pair_bounds_v[p, q]
            for r in range(p + 1):
                s_max = q if r == p else r
                for s in range(s_max + 1):
                    if screen_tol > 0.0 and bound_pq * pair_bounds_v[r, s] < screen_tol:
                        continue
                    value = contracted_eri_indices(
                        p, q, r, s,
                        shells_v, origins_v, exps_v, weights_v, nprim_v,
                    )
                    nperm = 0

                    if not direct_perm_seen(aa, bb, cc, dd, nperm, p, q, r, s):
                        aa[nperm] = p; bb[nperm] = q; cc[nperm] = r; dd[nperm] = s; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, p, q, r, s)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, q, p, r, s):
                        aa[nperm] = q; bb[nperm] = p; cc[nperm] = r; dd[nperm] = s; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, q, p, r, s)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, p, q, s, r):
                        aa[nperm] = p; bb[nperm] = q; cc[nperm] = s; dd[nperm] = r; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, p, q, s, r)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, q, p, s, r):
                        aa[nperm] = q; bb[nperm] = p; cc[nperm] = s; dd[nperm] = r; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, q, p, s, r)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, r, s, p, q):
                        aa[nperm] = r; bb[nperm] = s; cc[nperm] = p; dd[nperm] = q; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, r, s, p, q)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, s, r, p, q):
                        aa[nperm] = s; bb[nperm] = r; cc[nperm] = p; dd[nperm] = q; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, s, r, p, q)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, r, s, q, p):
                        aa[nperm] = r; bb[nperm] = s; cc[nperm] = q; dd[nperm] = p; nperm += 1
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, r, s, q, p)
                    if not direct_perm_seen(aa, bb, cc, dd, nperm, s, r, q, p):
                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, s, r, q, p)

    return vj, vk


cpdef direct_jk_blocked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    cnp.ndarray[double, ndim=2] dm,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    if dm.shape[0] != nao or dm.shape[1] != nao:
        raise ValueError("dm shape is inconsistent with the AO basis.")
    if pair_bounds.shape[0] != nao or pair_bounds.shape[1] != nao:
        raise ValueError("pair_bounds shape is inconsistent with the AO basis.")

    cdef cnp.ndarray[double, ndim=2] vj = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] vk = np.zeros((nao, nao), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=4] block
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, ::1] dm_v = dm
    cdef double[:, ::1] vj_v = vj
    cdef double[:, ::1] vk_v = vk
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int ish, jsh, ksh, lsh, lsh_max
    cdef int p0, p1, q0, q1, r0, r1, s0, s1
    cdef int ip, iq, ir, is_, p, q, r, s, ij, kl, nperm
    cdef int aa[8]
    cdef int bb[8]
    cdef int cc[8]
    cdef int dd[8]
    cdef double value, bound_pq, bound_rs
    cdef double* shell_pair_bounds
    cdef int* shell_pair_min
    cdef int* shell_pair_max
    cdef int pair_pos

    shell_pair_bounds = <double*>malloc(nshell * nshell * sizeof(double))
    shell_pair_min = <int*>malloc(nshell * nshell * sizeof(int))
    shell_pair_max = <int*>malloc(nshell * nshell * sizeof(int))
    if shell_pair_bounds == NULL or shell_pair_min == NULL or shell_pair_max == NULL:
        free(shell_pair_bounds)
        free(shell_pair_min)
        free(shell_pair_max)
        return vj, vk

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        for jsh in range(nshell):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            bound_pq = 0.0
            shell_pair_min[ish * nshell + jsh] = nao * (nao + 1) // 2
            shell_pair_max[ish * nshell + jsh] = -1
            for ip in range(p0, p1):
                for iq in range(q0, q1):
                    if pair_bounds_v[ip, iq] > bound_pq:
                        bound_pq = pair_bounds_v[ip, iq]
                    if ip >= iq:
                        pair_pos = pair_index(ip, iq)
                        if pair_pos < shell_pair_min[ish * nshell + jsh]:
                            shell_pair_min[ish * nshell + jsh] = pair_pos
                        if pair_pos > shell_pair_max[ish * nshell + jsh]:
                            shell_pair_max[ish * nshell + jsh] = pair_pos
            shell_pair_bounds[ish * nshell + jsh] = bound_pq

    for ish in range(nshell):
        p0 = <int>shell_starts_v[ish]
        p1 = <int>shell_stops_v[ish]
        for jsh in range(ish + 1):
            q0 = <int>shell_starts_v[jsh]
            q1 = <int>shell_stops_v[jsh]
            bound_pq = shell_pair_bounds[ish * nshell + jsh]

            for ksh in range(nshell):
                r0 = <int>shell_starts_v[ksh]
                r1 = <int>shell_stops_v[ksh]
                for lsh in range(ksh + 1):
                    s0 = <int>shell_starts_v[lsh]
                    s1 = <int>shell_stops_v[lsh]
                    bound_rs = shell_pair_bounds[ksh * nshell + lsh]
                    if shell_pair_max[ish * nshell + jsh] < shell_pair_min[ksh * nshell + lsh]:
                        continue
                    if screen_tol > 0.0 and bound_pq * bound_rs < screen_tol:
                        continue

                    block = compute_cartesian_shell_quartet_block(
                        shells, origins, exps, weights, nprim,
                        p0, p1, q0, q1, r0, r1, s0, s1,
                    )
                    for ip in range(p1 - p0):
                        p = p0 + ip
                        for iq in range(q1 - q0):
                            q = q0 + iq
                            if p < q:
                                continue
                            ij = pair_index(p, q)
                            for ir in range(r1 - r0):
                                r = r0 + ir
                                for is_ in range(s1 - s0):
                                    s = s0 + is_
                                    if r < s:
                                        continue
                                    kl = pair_index(r, s)
                                    if ij < kl:
                                        continue
                                    value = block[ip, iq, ir, is_]
                                    nperm = 0

                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, p, q, r, s):
                                        aa[nperm] = p; bb[nperm] = q; cc[nperm] = r; dd[nperm] = s; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, p, q, r, s)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, q, p, r, s):
                                        aa[nperm] = q; bb[nperm] = p; cc[nperm] = r; dd[nperm] = s; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, q, p, r, s)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, p, q, s, r):
                                        aa[nperm] = p; bb[nperm] = q; cc[nperm] = s; dd[nperm] = r; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, p, q, s, r)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, q, p, s, r):
                                        aa[nperm] = q; bb[nperm] = p; cc[nperm] = s; dd[nperm] = r; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, q, p, s, r)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, r, s, p, q):
                                        aa[nperm] = r; bb[nperm] = s; cc[nperm] = p; dd[nperm] = q; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, r, s, p, q)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, s, r, p, q):
                                        aa[nperm] = s; bb[nperm] = r; cc[nperm] = p; dd[nperm] = q; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, s, r, p, q)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, r, s, q, p):
                                        aa[nperm] = r; bb[nperm] = s; cc[nperm] = q; dd[nperm] = p; nperm += 1
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, r, s, q, p)
                                    if not direct_perm_seen(aa, bb, cc, dd, nperm, s, r, q, p):
                                        direct_jk_add_perm(vj_v, vk_v, dm_v, value, s, r, q, p)

    free(shell_pair_bounds)
    free(shell_pair_min)
    free(shell_pair_max)
    return vj, vk


cpdef compute_pivoted_cholesky_factors_blocked(
    cnp.ndarray[int64_t, ndim=2] shells,
    cnp.ndarray[double, ndim=2] origins,
    cnp.ndarray[double, ndim=2] exps,
    cnp.ndarray[double, ndim=2] weights,
    cnp.ndarray[int64_t, ndim=1] nprim,
    cnp.ndarray[double, ndim=2] pair_bounds,
    cnp.ndarray[int64_t, ndim=1] shell_starts,
    cnp.ndarray[int64_t, ndim=1] shell_stops,
    double tol=1e-8,
    object max_rank=None,
    double screen_tol=0.0,
):
    cdef int nao = shells.shape[0]
    cdef int nshell = shell_starts.shape[0]
    cdef int npair = nao * (nao + 1) // 2
    cdef int rank_limit = npair if max_rank is None else min(int(max_rank), npair)
    cdef int rank_capacity = min(rank_limit, 64)
    cdef cnp.ndarray[int64_t, ndim=2] pairs = np.zeros((npair, 2), dtype=np.int64)
    cdef cnp.ndarray[double, ndim=1] diag = np.zeros(npair, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] chol = np.zeros((npair, rank_capacity), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] grown_chol
    cdef cnp.ndarray[double, ndim=1] col = np.zeros(npair, dtype=np.float64)
    cdef cnp.ndarray[int64_t, ndim=2] pair_blocks = np.zeros((nshell * (nshell + 1) // 2, 2), dtype=np.int64)
    cdef cnp.ndarray[double, ndim=1] pair_block_max = np.zeros(nshell * (nshell + 1) // 2, dtype=np.float64)
    cdef int p, q, pair_idx, pivot, pi, pj, i, j, k, rank = 0
    cdef int si, sj, block_idx = 0, start_idx, block_start, block_stop
    cdef int p0, p1, q0, q1
    cdef double delta, pivot_bound, value, correction, inv_sqrt_delta, block_bound

    cdef int64_t[:, ::1] shells_v = shells
    cdef double[:, ::1] origins_v = origins
    cdef double[:, ::1] exps_v = exps
    cdef double[:, ::1] weights_v = weights
    cdef int64_t[::1] nprim_v = nprim
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef int64_t[:, ::1] pairs_v = pairs
    cdef double[::1] diag_v = diag
    cdef double[:, ::1] chol_v = chol
    cdef double[::1] col_v = col
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef int64_t[:, ::1] pair_blocks_v = pair_blocks
    cdef double[::1] pair_block_max_v = pair_block_max

    pair_idx = 0
    for si in range(nshell):
        p0 = <int>shell_starts_v[si]
        p1 = <int>shell_stops_v[si]
        for sj in range(si + 1):
            q0 = <int>shell_starts_v[sj]
            q1 = <int>shell_stops_v[sj]
            block_start = pair_idx
            block_bound = 0.0
            for p in range(p0, p1):
                for q in range(q0, q1):
                    if si == sj and q > p:
                        continue
                    pairs_v[pair_idx, 0] = p
                    pairs_v[pair_idx, 1] = q
                    value = pair_bounds_v[p, q]
                    diag_v[pair_idx] = value * value if value > 0.0 else 0.0
                    if value > block_bound:
                        block_bound = value
                    pair_idx += 1
            pair_blocks_v[block_idx, 0] = block_start
            pair_blocks_v[block_idx, 1] = pair_idx
            pair_block_max_v[block_idx] = block_bound
            block_idx += 1

    while rank < rank_limit:
        if rank == rank_capacity:
            rank_capacity = min(rank_limit, max(rank_capacity + 1, 2 * rank_capacity))
            grown_chol = np.zeros((npair, rank_capacity), dtype=np.float64)
            grown_chol[:, :rank] = chol[:, :rank]
            chol = grown_chol
            chol_v = chol
        pivot = 0
        delta = diag_v[0] if npair > 0 else 0.0
        for pair_idx in range(1, npair):
            if diag_v[pair_idx] > delta:
                delta = diag_v[pair_idx]
                pivot = pair_idx

        if delta <= tol:
            break

        pi = <int>pairs_v[pivot, 0]
        pj = <int>pairs_v[pivot, 1]
        pivot_bound = pair_bounds_v[pi, pj]

        for pair_idx in range(npair):
            col_v[pair_idx] = 0.0

        for start_idx in range(block_idx):
            if screen_tol > 0.0 and pivot_bound * pair_block_max_v[start_idx] < screen_tol:
                continue
            block_start = <int>pair_blocks_v[start_idx, 0]
            block_stop = <int>pair_blocks_v[start_idx, 1]
            for pair_idx in range(block_start, block_stop):
                i = <int>pairs_v[pair_idx, 0]
                j = <int>pairs_v[pair_idx, 1]
                if screen_tol > 0.0 and pivot_bound * pair_bounds_v[i, j] < screen_tol:
                    continue
                col_v[pair_idx] = contracted_eri_indices(
                    i, j, pi, pj,
                    shells_v, origins_v, exps_v, weights_v, nprim_v,
                )

        if rank > 0:
            for pair_idx in range(npair):
                correction = 0.0
                for k in range(rank):
                    correction += chol_v[pair_idx, k] * chol_v[pivot, k]
                col_v[pair_idx] -= correction

        delta = col_v[pivot]
        if delta <= tol:
            diag_v[pivot] = 0.0
            continue

        inv_sqrt_delta = 1.0 / sqrt(delta)
        for pair_idx in range(npair):
            value = col_v[pair_idx] * inv_sqrt_delta
            chol_v[pair_idx, rank] = value
            diag_v[pair_idx] -= value * value
            if diag_v[pair_idx] < 0.0:
                diag_v[pair_idx] = 0.0
        rank += 1

    return np.asarray(chol[:, :rank].T, dtype=np.float64), np.asarray(pairs, dtype=np.int64)
