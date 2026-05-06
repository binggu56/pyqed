# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt, fabs, NAN
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from scipy.special.cython_special cimport hyp1f1


cdef double PI = 3.141592653589793238462643383279502884
cdef double ERI_PREFAC = 2.0 * (PI ** 2.5)


cdef inline size_t idx3(int i, int j, int t, int jdim, int tdim) noexcept nogil:
    return ((<size_t>i * <size_t>jdim) + <size_t>j) * <size_t>tdim + <size_t>t


cdef inline size_t idx4(int t, int u, int v, int n, int udim, int vdim, int ndim) noexcept nogil:
    return ((((<size_t>t * <size_t>udim) + <size_t>u) * <size_t>vdim) + <size_t>v) * <size_t>ndim + <size_t>n


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
    cdef double* memo_abx = <double*>malloc(size_abx * sizeof(double))
    cdef double* memo_aby = <double*>malloc(size_aby * sizeof(double))
    cdef double* memo_abz = <double*>malloc(size_abz * sizeof(double))
    cdef double* memo_cx = <double*>malloc(size_cx * sizeof(double))
    cdef double* memo_cy = <double*>malloc(size_cy * sizeof(double))
    cdef double* memo_cz = <double*>malloc(size_cz * sizeof(double))
    cdef double* memo_r = <double*>malloc(size_r * sizeof(double))
    cdef size_t i
    cdef int t, u, v, tau, nu, phi
    cdef double ex_ab, exy_ab, xyz_ab, ex_c, exy_c, sign, value = 0.0
    if memo_abx == NULL or memo_aby == NULL or memo_abz == NULL or memo_cx == NULL or memo_cy == NULL or memo_cz == NULL or memo_r == NULL:
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
    free(memo_abx); free(memo_aby); free(memo_abz); free(memo_cx); free(memo_cy); free(memo_cz); free(memo_r)
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)))


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
    cdef int64_t computed = 0
    cdef int64_t skipped = 0
    cdef double bound_pq, bound_rs, value
    cdef double* shell_pair_bounds
    cdef int64_t[::1] shell_starts_v = shell_starts
    cdef int64_t[::1] shell_stops_v = shell_stops
    cdef double[:, ::1] pair_bounds_v = pair_bounds
    cdef double[:, :, :, ::1] eri_v = eri
    cdef double[:, :, :, ::1] block_v

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

                    block = compute_cartesian_shell_quartet_block(
                        shells, origins, exps, weights, nprim,
                        p0, p1, q0, q1, r0, r1, s0, s1,
                    )
                    eri[p0:p1, q0:q1, r0:r1, s0:s1] = block
                    eri[q0:q1, p0:p1, r0:r1, s0:s1] = block.transpose(1, 0, 2, 3)
                    eri[p0:p1, q0:q1, s0:s1, r0:r1] = block.transpose(0, 1, 3, 2)
                    eri[q0:q1, p0:p1, s0:s1, r0:r1] = block.transpose(1, 0, 3, 2)
                    eri[r0:r1, s0:s1, p0:p1, q0:q1] = block.transpose(2, 3, 0, 1)
                    eri[s0:s1, r0:r1, p0:p1, q0:q1] = block.transpose(3, 2, 0, 1)
                    eri[r0:r1, s0:s1, q0:q1, p0:p1] = block.transpose(2, 3, 1, 0)
                    eri[s0:s1, r0:r1, q0:q1, p0:p1] = block.transpose(3, 2, 1, 0)
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
    cdef double* memo_abx
    cdef double* memo_aby
    cdef double* memo_abz
    cdef double* memo_cdx
    cdef double* memo_cdy
    cdef double* memo_cdz
    cdef double* memo_r
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
    if ax == NULL or ay == NULL or az == NULL or bx == NULL or by == NULL or bz == NULL or cx == NULL or cy == NULL or cz == NULL or dxc == NULL or dyc == NULL or dzc == NULL or pq_a == NULL or pq_b == NULL or pq_p == NULL or pq_px == NULL or pq_py == NULL or pq_pz == NULL or rs_a == NULL or rs_b == NULL or rs_p == NULL or rs_px == NULL or rs_py == NULL or rs_pz == NULL:
        free(ax); free(ay); free(az); free(bx); free(by); free(bz); free(cx); free(cy); free(cz); free(dxc); free(dyc); free(dzc)
        free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz)
        free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz)
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
    for ia in range(np_):
        ao_p = p0 + ia
        for ib in range(nq_):
            ao_q = q0 + ib
            for ic in range(nr_):
                ao_r = r0 + ic
                for id_ in range(ns_):
                    ao_s = s0 + id_
                    value = 0.0
                    idx_pq = 0
                    for ip in range(nprim_v[p0]):
                        for iq in range(nprim_v[q0]):
                            idx_rs = 0
                            for ir in range(nprim_v[r0]):
                                for is_ in range(nprim_v[s0]):
                                    prefac = (
                                        weights_v[ao_p, ip] * weights_v[ao_q, iq]
                                        * weights_v[ao_r, ir] * weights_v[ao_s, is_]
                                    )
                                    value += prefac * primitive_eri_precomputed(
                                        pq_a[idx_pq], pq_b[idx_pq], pq_p[idx_pq], pq_px[idx_pq], pq_py[idx_pq], pq_pz[idx_pq], abx, aby, abz,
                                        ax[ia], ay[ia], az[ia], bx[ib], by[ib], bz[ib],
                                        rs_a[idx_rs], rs_b[idx_rs], rs_p[idx_rs], rs_px[idx_rs], rs_py[idx_rs], rs_pz[idx_rs], cdx, cdy, cdz,
                                        cx[ic], cy[ic], cz[ic], dxc[id_], dyc[id_], dzc[id_],
                                    )
                                    idx_rs += 1
                            idx_pq += 1
                    block_v[ia, ib, ic, id_] = value

    free(ax); free(ay); free(az); free(bx); free(by); free(bz); free(cx); free(cy); free(cz); free(dxc); free(dyc); free(dzc)
    free(pq_a); free(pq_b); free(pq_p); free(pq_px); free(pq_py); free(pq_pz)
    free(rs_a); free(rs_b); free(rs_p); free(rs_px); free(rs_py); free(rs_pz)

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
