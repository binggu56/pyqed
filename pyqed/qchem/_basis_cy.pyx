# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt, fabs, NAN
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint64_t, uint8_t
from libc.string cimport memset
from scipy.special.cython_special cimport hyp1f1


cdef double PI = 3.141592653589793238462643383279502884
cdef double ERI_PREFAC = 2.0 * (PI ** 2.5)
DEF OS_MAX_STATES = 65536
DEF OS_HASH_CAP = 16384
DEF OS_VRR_PAIR_MAX_L = 4
DEF OS_VRR_MAX_CART = 15
DEF THREE_CENTER_E_CAP = 128
DEF THREE_CENTER_R_CAP = 2048


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


cdef inline double primitive_nuclear_attraction(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double Cx, double Cy, double Cz
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
                    t, u, v, 0, p, dx, dy, dz, rpc, memo_r, uy_max + 1, vz_max + 1, nmax
                )
    free(memo_x); free(memo_y); free(memo_z); free(memo_r)
    return value * (2.0 * PI / p)


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
                        if use_lr:
                            lr_value = os_vrr_hrr_eval_expanded(
                                lr_vrr_table,
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
                            tensor_v[aux_i, ao_p, ao_q] += prefac * (
                                full_value - lr_scale * lr_value
                            )
                        else:
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
    cdef cnp.ndarray[int64_t, ndim=2] pairs = np.zeros((npair, 2), dtype=np.int64)
    cdef cnp.ndarray[double, ndim=1] diag = np.zeros(npair, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] chol = np.zeros((npair, rank_limit), dtype=np.float64)
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
