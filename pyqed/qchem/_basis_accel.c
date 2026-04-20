#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define PI 3.141592653589793238462643383279502884
#define ERI_PREFAC (2.0 * pow(PI, 2.5))

static double boys_fn(int n, double T) {
    if (T < 1e-12) {
        return 1.0 / (2.0 * n + 1.0);
    }

    if (T < 30.0) {
        double term = 1.0 / (2.0 * n + 1.0);
        double value = term;
        for (int k = 1; k < 256; ++k) {
            term *= -T / (double)k;
            double add = term / (2.0 * n + 2.0 * k + 1.0);
            value += add;
            if (fabs(add) < 1e-16) {
                break;
            }
        }
        return value;
    }

    double sqrt_T = sqrt(T);
    double value = 0.5 * sqrt(PI / T) * erf(sqrt_T);
    if (n == 0) {
        return value;
    }

    double exp_T = exp(-T);
    for (int m = 0; m < n; ++m) {
        value = ((2.0 * m + 1.0) * value - exp_T) / (2.0 * T);
    }
    return value;
}

static inline size_t idx3(int i, int j, int t, int jdim, int tdim) {
    return ((size_t)i * (size_t)jdim + (size_t)j) * (size_t)tdim + (size_t)t;
}

static inline size_t idx4(int t, int u, int v, int n, int udim, int vdim, int ndim) {
    return ((((size_t)t * (size_t)udim) + (size_t)u) * (size_t)vdim + (size_t)v) * (size_t)ndim + (size_t)n;
}

static double E_rec(int i, int j, int t, double Qx, double a, double b, double *memo, int jdim, int tdim) {
    if (t < 0 || t > (i + j)) {
        return 0.0;
    }

    size_t pos = idx3(i, j, t, jdim, tdim);
    if (!isnan(memo[pos])) {
        return memo[pos];
    }

    double p = a + b;
    double q = a * b / p;
    double value;
    if (i == 0 && j == 0 && t == 0) {
        value = exp(-q * Qx * Qx);
    } else if (j == 0) {
        value =
            (1.0 / (2.0 * p)) * E_rec(i - 1, j, t - 1, Qx, a, b, memo, jdim, tdim)
            - (q * Qx / a) * E_rec(i - 1, j, t, Qx, a, b, memo, jdim, tdim)
            + (t + 1.0) * E_rec(i - 1, j, t + 1, Qx, a, b, memo, jdim, tdim);
    } else {
        value =
            (1.0 / (2.0 * p)) * E_rec(i, j - 1, t - 1, Qx, a, b, memo, jdim, tdim)
            + (q * Qx / b) * E_rec(i, j - 1, t, Qx, a, b, memo, jdim, tdim)
            + (t + 1.0) * E_rec(i, j - 1, t + 1, Qx, a, b, memo, jdim, tdim);
    }

    memo[pos] = value;
    return value;
}

static double R_rec(
    int t, int u, int v, int n,
    double p, double PCx, double PCy, double PCz, double RPC,
    double *memo, int udim, int vdim, int ndim
) {
    if (t < 0 || u < 0 || v < 0) {
        return 0.0;
    }

    size_t pos = idx4(t, u, v, n, udim, vdim, ndim);
    if (!isnan(memo[pos])) {
        return memo[pos];
    }

    double value;
    if (t == 0 && u == 0 && v == 0) {
        value = pow(-2.0 * p, n) * boys_fn(n, p * RPC * RPC);
    } else if (t == 0 && u == 0) {
        value = 0.0;
        if (v > 1) {
            value += (v - 1.0) * R_rec(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
        }
        value += PCz * R_rec(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
    } else if (t == 0) {
        value = 0.0;
        if (u > 1) {
            value += (u - 1.0) * R_rec(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
        }
        value += PCy * R_rec(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
    } else {
        value = 0.0;
        if (t > 1) {
            value += (t - 1.0) * R_rec(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
        }
        value += PCx * R_rec(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo, udim, vdim, ndim);
    }

    memo[pos] = value;
    return value;
}

static double primitive_eri(
    double a, int l1, int m1, int n1, double Ax, double Ay, double Az,
    double b, int l2, int m2, int n2, double Bx, double By, double Bz,
    double c, int l3, int m3, int n3, double Cx, double Cy, double Cz,
    double d, int l4, int m4, int n4, double Dx, double Dy, double Dz
) {
    double p = a + b;
    double q = c + d;
    double alpha = p * q / (p + q);

    double px = (a * Ax + b * Bx) / p;
    double py = (a * Ay + b * By) / p;
    double pz = (a * Az + b * Bz) / p;
    double qx = (c * Cx + d * Dx) / q;
    double qy = (c * Cy + d * Dy) / q;
    double qz = (c * Cz + d * Dz) / q;
    double dx = px - qx;
    double dy = py - qy;
    double dz = pz - qz;
    double rpq = sqrt(dx * dx + dy * dy + dz * dz);

    double abx = Ax - Bx;
    double aby = Ay - By;
    double abz = Az - Bz;
    double cdx = Cx - Dx;
    double cdy = Cy - Dy;
    double cdz = Cz - Dz;

    int tdim_abx = l1 + l2 + 2;
    int tdim_aby = m1 + m2 + 2;
    int tdim_abz = n1 + n2 + 2;
    int tdim_cdx = l3 + l4 + 2;
    int tdim_cdy = m3 + m4 + 2;
    int tdim_cdz = n3 + n4 + 2;

    size_t size_abx = (size_t)(l1 + 1) * (size_t)(l2 + 1) * (size_t)tdim_abx;
    size_t size_aby = (size_t)(m1 + 1) * (size_t)(m2 + 1) * (size_t)tdim_aby;
    size_t size_abz = (size_t)(n1 + 1) * (size_t)(n2 + 1) * (size_t)tdim_abz;
    size_t size_cdx = (size_t)(l3 + 1) * (size_t)(l4 + 1) * (size_t)tdim_cdx;
    size_t size_cdy = (size_t)(m3 + 1) * (size_t)(m4 + 1) * (size_t)tdim_cdy;
    size_t size_cdz = (size_t)(n3 + 1) * (size_t)(n4 + 1) * (size_t)tdim_cdz;

    int tx_max = l1 + l2 + l3 + l4;
    int uy_max = m1 + m2 + m3 + m4;
    int vz_max = n1 + n2 + n3 + n4;
    int nmax = tx_max + uy_max + vz_max + 2;
    size_t size_r = (size_t)(tx_max + 1) * (size_t)(uy_max + 1) * (size_t)(vz_max + 1) * (size_t)nmax;

    double *memo_abx = (double *)malloc(size_abx * sizeof(double));
    double *memo_aby = (double *)malloc(size_aby * sizeof(double));
    double *memo_abz = (double *)malloc(size_abz * sizeof(double));
    double *memo_cdx = (double *)malloc(size_cdx * sizeof(double));
    double *memo_cdy = (double *)malloc(size_cdy * sizeof(double));
    double *memo_cdz = (double *)malloc(size_cdz * sizeof(double));
    double *memo_r = (double *)malloc(size_r * sizeof(double));
    if (!memo_abx || !memo_aby || !memo_abz || !memo_cdx || !memo_cdy || !memo_cdz || !memo_r) {
        free(memo_abx); free(memo_aby); free(memo_abz);
        free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r);
        return NAN;
    }

    for (size_t i = 0; i < size_abx; ++i) memo_abx[i] = NAN;
    for (size_t i = 0; i < size_aby; ++i) memo_aby[i] = NAN;
    for (size_t i = 0; i < size_abz; ++i) memo_abz[i] = NAN;
    for (size_t i = 0; i < size_cdx; ++i) memo_cdx[i] = NAN;
    for (size_t i = 0; i < size_cdy; ++i) memo_cdy[i] = NAN;
    for (size_t i = 0; i < size_cdz; ++i) memo_cdz[i] = NAN;
    for (size_t i = 0; i < size_r; ++i) memo_r[i] = NAN;

    double value = 0.0;
    for (int t = 0; t < l1 + l2 + 1; ++t) {
        double ex_ab = E_rec(l1, l2, t, abx, a, b, memo_abx, l2 + 1, tdim_abx);
        for (int u = 0; u < m1 + m2 + 1; ++u) {
            double exy_ab = ex_ab * E_rec(m1, m2, u, aby, a, b, memo_aby, m2 + 1, tdim_aby);
            for (int v = 0; v < n1 + n2 + 1; ++v) {
                double xyz_ab = exy_ab * E_rec(n1, n2, v, abz, a, b, memo_abz, n2 + 1, tdim_abz);
                for (int tau = 0; tau < l3 + l4 + 1; ++tau) {
                    double ex_cd = E_rec(l3, l4, tau, cdx, c, d, memo_cdx, l4 + 1, tdim_cdx);
                    for (int nu = 0; nu < m3 + m4 + 1; ++nu) {
                        double exy_cd = ex_cd * E_rec(m3, m4, nu, cdy, c, d, memo_cdy, m4 + 1, tdim_cdy);
                        for (int phi = 0; phi < n3 + n4 + 1; ++phi) {
                            double sign = ((tau + nu + phi) & 1) ? -1.0 : 1.0;
                            value +=
                                xyz_ab
                                * exy_cd
                                * E_rec(n3, n4, phi, cdz, c, d, memo_cdz, n4 + 1, tdim_cdz)
                                * sign
                                * R_rec(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpq, memo_r, uy_max + 1, vz_max + 1, nmax);
                        }
                    }
                }
            }
        }
    }

    free(memo_abx); free(memo_aby); free(memo_abz);
    free(memo_cdx); free(memo_cdy); free(memo_cdz); free(memo_r);
    return value * (ERI_PREFAC / (p * q * sqrt(p + q)));
}

static double contracted_eri(
    const int64_t *shells,
    const double *origins,
    const double *exps,
    const double *weights,
    const int64_t *nprim,
    int max_prim,
    int p, int q, int r, int s
) {
    double value = 0.0;
    int np_p = (int)nprim[p];
    int np_q = (int)nprim[q];
    int np_r = (int)nprim[r];
    int np_s = (int)nprim[s];

    const int64_t *sp = shells + 3 * p;
    const int64_t *sq = shells + 3 * q;
    const int64_t *sr = shells + 3 * r;
    const int64_t *ss = shells + 3 * s;
    const double *op = origins + 3 * p;
    const double *oq = origins + 3 * q;
    const double *or_ = origins + 3 * r;
    const double *os = origins + 3 * s;

    for (int ip = 0; ip < np_p; ++ip) {
        double wp = weights[(size_t)p * (size_t)max_prim + (size_t)ip];
        double ap = exps[(size_t)p * (size_t)max_prim + (size_t)ip];
        for (int iq = 0; iq < np_q; ++iq) {
            double wq = weights[(size_t)q * (size_t)max_prim + (size_t)iq];
            double aq = exps[(size_t)q * (size_t)max_prim + (size_t)iq];
            for (int ir = 0; ir < np_r; ++ir) {
                double wr = weights[(size_t)r * (size_t)max_prim + (size_t)ir];
                double ar = exps[(size_t)r * (size_t)max_prim + (size_t)ir];
                for (int is = 0; is < np_s; ++is) {
                    double ws = weights[(size_t)s * (size_t)max_prim + (size_t)is];
                    double as = exps[(size_t)s * (size_t)max_prim + (size_t)is];
                    double prim = primitive_eri(
                        ap, (int)sp[0], (int)sp[1], (int)sp[2], op[0], op[1], op[2],
                        aq, (int)sq[0], (int)sq[1], (int)sq[2], oq[0], oq[1], oq[2],
                        ar, (int)sr[0], (int)sr[1], (int)sr[2], or_[0], or_[1], or_[2],
                        as, (int)ss[0], (int)ss[1], (int)ss[2], os[0], os[1], os[2]
                    );
                    if (isnan(prim)) {
                        return NAN;
                    }
                    value += wp * wq * wr * ws * prim;
                }
            }
        }
    }

    return value;
}

int compute_dense_eri(
    int nao,
    int max_prim,
    const int64_t *shells,
    const double *origins,
    const double *exps,
    const double *weights,
    const int64_t *nprim,
    const double *pair_bounds,
    double screen_tol,
    double *eri,
    int64_t *computed,
    int64_t *skipped
) {
    int64_t ncomputed = 0;
    int64_t nskipped = 0;

    for (int p = 0; p < nao; ++p) {
        for (int q = 0; q <= p; ++q) {
            double bound_pq = pair_bounds[(size_t)p * (size_t)nao + (size_t)q];
            for (int r = 0; r <= p; ++r) {
                int s_max = (r == p) ? q : r;
                for (int s = 0; s <= s_max; ++s) {
                    if (screen_tol > 0.0 && bound_pq * pair_bounds[(size_t)r * (size_t)nao + (size_t)s] < screen_tol) {
                        nskipped += 1;
                        continue;
                    }

                    double value = contracted_eri(shells, origins, exps, weights, nprim, max_prim, p, q, r, s);
                    if (isnan(value)) {
                        return -1;
                    }
                    if (screen_tol > 0.0 && fabs(value) < screen_tol) {
                        nskipped += 1;
                        continue;
                    }

                    size_t i1 = (((size_t)p * (size_t)nao + (size_t)q) * (size_t)nao + (size_t)r) * (size_t)nao + (size_t)s;
                    size_t i2 = (((size_t)q * (size_t)nao + (size_t)p) * (size_t)nao + (size_t)r) * (size_t)nao + (size_t)s;
                    size_t i3 = (((size_t)p * (size_t)nao + (size_t)q) * (size_t)nao + (size_t)s) * (size_t)nao + (size_t)r;
                    size_t i4 = (((size_t)q * (size_t)nao + (size_t)p) * (size_t)nao + (size_t)s) * (size_t)nao + (size_t)r;
                    size_t i5 = (((size_t)r * (size_t)nao + (size_t)s) * (size_t)nao + (size_t)p) * (size_t)nao + (size_t)q;
                    size_t i6 = (((size_t)s * (size_t)nao + (size_t)r) * (size_t)nao + (size_t)p) * (size_t)nao + (size_t)q;
                    size_t i7 = (((size_t)r * (size_t)nao + (size_t)s) * (size_t)nao + (size_t)q) * (size_t)nao + (size_t)p;
                    size_t i8 = (((size_t)s * (size_t)nao + (size_t)r) * (size_t)nao + (size_t)q) * (size_t)nao + (size_t)p;
                    eri[i1] = value; eri[i2] = value; eri[i3] = value; eri[i4] = value;
                    eri[i5] = value; eri[i6] = value; eri[i7] = value; eri[i8] = value;
                    ncomputed += 1;
                }
            }
        }
    }

    *computed = ncomputed;
    *skipped = nskipped;
    return 0;
}
