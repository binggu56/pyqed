import numpy as np
import itertools
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from scipy.special import ive, i0e, j0, erfc, erfcx
from scipy.integrate import dblquad, IntegrationWarning, nquad, quad
# ============================================================
# Basis labels
# ============================================================

@dataclass
class PrimitiveLabel:
    """
    Minimal label for one primitive GTO.
    kind: '2d-s', '2d-px', '2d-py', '2d-dxy', '2d-dx2', '2d-dy2'
    dim : 2
    l   : (lx, ly, lz)
    """
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"


# ============================================================
# Prony parameters and standard primitive exponents
# ============================================================

# prony coefficients are updated with less fitting error
ETAS = np.array([
    0.00059942, 0.00752137, 0.03022179, 0.06319275, 0.09171853,
    0.10647206, 0.10706705, 0.09933569, 0.08699365, 0.07400703,
    0.0616606, 0.05096948, 0.04209444, 0.03509802, 0.02985465,
    0.02612568, 0.02359271, 0.0219556, 0.02098508, 0.02053458
], dtype=np.float64)

XIS = np.array([
    7.31648813e+00, 4.33806312e+00, 2.67573322e+00, 1.67612356e+00,
    1.05902542e+00, 6.73244082e-01, 4.30191779e-01, 2.76155796e-01,
    1.78029854e-01, 1.15213080e-01, 7.47981415e-02, 4.86502800e-02,
    3.16135354e-02, 2.04036868e-02, 1.29288660e-02, 7.87553459e-03,
    4.44302044e-03, 2.15893384e-03, 7.53663310e-04, 8.25048959e-05
], dtype=np.float64)

# Standard Exp for H / He
STO6_EXPS_H = np.array([35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280], dtype=float)
STO6_EXPS_He = np.array([65.98456824, 12.09819836, 3.384639924, 1.162715163, 0.451516322, 0.185959356], dtype=float)
Exp_631g_ss_H = np.array([18.73113696, 2.825394365, 0.6401216923, 0.1612777588], dtype=float)


# ============================================================
# Global dispatch settings
# ============================================================

CENTER_TOL = 1e-14
OFFCENTER_NUMERICAL_CUTOFF = 0.10


# ============================================================
# Shared utilities (2D)
# ============================================================

def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ad = A[:, None, :] - B[None, :, :]
    return np.sum(Ad * Ad, axis=-1)

def _parse_2d_l(lbl: PrimitiveLabel) -> Tuple[int, int]:
    if lbl.dim != 2:
        raise ValueError("Only dim=2 primitives supported.")
    return lbl.l[0], lbl.l[1]

def _same_center(a, b, tol=CENTER_TOL):
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) < tol


def _all_same_center(centers, tol=CENTER_TOL):
    c0 = np.asarray(centers[0], float)
    for c in centers[1:]:
        if np.linalg.norm(np.asarray(c, float) - c0) > tol:
            return False
    return True


def _pair_is_on_axis_same_center(centers, i, j, nuc_xy, tol=CENTER_TOL):
    return (
        _same_center(centers[i], centers[j], tol=tol)
        and _same_center(centers[i], nuc_xy, tol=tol)
    )


def _quartet_is_same_center(centers, i, j, k, l, tol=CENTER_TOL):
    c0 = np.asarray(centers[i], float)
    return (
        np.linalg.norm(np.asarray(centers[j], float) - c0) < tol
        and np.linalg.norm(np.asarray(centers[k], float) - c0) < tol
        and np.linalg.norm(np.asarray(centers[l], float) - c0) < tol
    )


def make_xy_spd_primitive_basis(
    nuclei_tuples: List[Tuple[float, float, float, float]],
    exps_s: np.ndarray,
    exps_p: np.ndarray,
    exps_d: np.ndarray = None,
    decimals: int = 12,
):
    """
    Generates primitive basis including s, p, and d shells.
    d-shell uses Cartesian components: x^2, y^2, xy.
    """
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    if exps_d is None:
        exps_d = np.array([], float)
    else:
        exps_d = np.asarray(exps_d, float).reshape(-1)

    rows = []
    for (Z, x, y, z) in nuclei_tuples:
        x, y = float(x), float(y)
        for a in exps_s:
            rows.append((a, x, y, 0))
        for a in exps_p:
            rows.append((a, x, y, 1))
        for a in exps_p:
            rows.append((a, x, y, 2))
        for a in exps_d:
            rows.append((a, x, y, 3))
        for a in exps_d:
            rows.append((a, x, y, 4))
        for a in exps_d:
            rows.append((a, x, y, 5))

    if not rows:
        return np.zeros((0,), float), np.zeros((0, 2), float), []

    arr = np.asarray(rows, float)
    arr_r = np.round(arr, decimals=decimals)
    _, idx = np.unique(arr_r, axis=0, return_index=True)
    idx = np.sort(idx)
    arr = arr[idx]

    alphas = arr[:, 0].astype(float)
    centers = arr[:, 1:3].astype(float)
    kinds = arr[:, 3].astype(int)

    labels: List[PrimitiveLabel] = []
    for k in kinds:
        if k == 0:
            labels.append(PrimitiveLabel(kind="2d-s", dim=2, l=(0, 0, 0)))
        elif k == 1:
            labels.append(PrimitiveLabel(kind="2d-px", dim=2, l=(1, 0, 0)))
        elif k == 2:
            labels.append(PrimitiveLabel(kind="2d-py", dim=2, l=(0, 1, 0)))
        elif k == 3:
            labels.append(PrimitiveLabel(kind="2d-dx2", dim=2, l=(2, 0, 0)))
        elif k == 4:
            labels.append(PrimitiveLabel(kind="2d-dy2", dim=2, l=(0, 2, 0)))
        elif k == 5:
            labels.append(PrimitiveLabel(kind="2d-dxy", dim=2, l=(1, 1, 0)))

    return alphas, centers, labels

# Alias for backward compatibility
make_xy_sp_primitive_basis = make_xy_spd_primitive_basis


# ============================================================
# Polynomial/Gaussian 1D helper integrals
# ============================================================

def gamma_binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

def _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta):
    gamma = alpha + beta
    P = (alpha * x1 + beta * x2) / gamma
    pre = np.exp(-(alpha * beta / gamma) * (x1 - x2) ** 2) * np.sqrt(np.pi / gamma)

    Q1 = P - x1
    Q2 = P - x2

    val = 0.0
    for i in range(l1 + 1):
        comb1 = gamma_binom(l1, i)
        term1 = Q1 ** (l1 - i)
        for j in range(l2 + 1):
            comb2 = gamma_binom(l2, j)
            term2 = Q2 ** (l2 - j)

            k = i + j
            if k % 2 == 0:
                m = k // 2
                dfact = 1.0
                for x in range(1, 2 * m, 2):
                    dfact *= x
                gauss_int = dfact / ((2 * gamma) ** m)
                val += comb1 * comb2 * term1 * term2 * gauss_int

    return pre * val

def _kinetic_1d_explicit(l1, l2, x1, x2, alpha, beta):
    term1 = 4.0 * beta**2 * _overlap_1d_explicit(l1, l2 + 2, x1, x2, alpha, beta)
    term2 = -2.0 * beta * (2 * l2 + 1) * _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta)
    term3 = 0.0
    if l2 >= 2:
        term3 = l2 * (l2 - 1) * _overlap_1d_explicit(l1, l2 - 2, x1, x2, alpha, beta)

    return -0.5 * (term1 + term2 + term3)

def _erfcx_safe_paper(x):
    """
    Safe evaluation of exp(x^2) * erfc(x).

    For moderate x, use the exact expression.
    For large x, use the paper-style asymptotic approximation:
        erfc(x) ≈ (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) exp(-x^2),
        t = 1 / (1 + k x)

    so that
        exp(x^2) erfc(x) ≈ a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    """
    x = float(abs(x))

    if x <= 9.0:
        return np.exp(x * x) * erfc(x)

    k  = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + k * x)
    poly = a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    return poly
# ============================================================
# Overlap / kinetic production
# ============================================================

def overlap_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    S = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]
        xA, yA = centers[i]
        lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N):
            aB = alphas[j]
            xB, yB = centers[j]
            lxB, lyB = _parse_2d_l(labels[j])

            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            val = Sx * Sy
            S[i, j] = val
            S[j, i] = val
    return S


def kinetic_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    Tmat = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]
        xA, yA = centers[i]
        lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N):
            aB = alphas[j]
            xB, yB = centers[j]
            lxB, lyB = _parse_2d_l(labels[j])

            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            Tx = _kinetic_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Ty = _kinetic_1d_explicit(lyA, lyB, yA, yB, aA, aB)

            val = Tx * Sy + Sx * Ty
            Tmat[i, j] = val
            Tmat[j, i] = val
    return Tmat


# ============================================================
# V_en helpers
# ============================================================

def _exact_ven_integrand(
    y, x,
    xA, yA, lxA, lyA, aA,
    xB, yB, lxB, lyB, aB,
    xN, yN, dz_sq
):
    """
    Unnormalized integrand for V_en:
        phi_A(r) phi_B(r) / sqrt((r-N)^2 + dz^2)
    """
    dist_sq = (x - xN) ** 2 + (y - yN) ** 2 + dz_sq

    valA = (x - xA) ** lxA * (y - yA) ** lyA * np.exp(-aA * ((x - xA) ** 2 + (y - yA) ** 2))
    valB = (x - xB) ** lxB * (y - yB) ** lyB * np.exp(-aB * ((x - xB) ** 2 + (y - yB) ** 2))

    eps = 1e-14
    op = 1.0 / np.sqrt(dist_sq + eps)
    return valA * valB * op


# def _ven_ss_same_center_exact(alpha_i, alpha_j, dz_abs):
#     """
#     POSITIVE Coulomb integral for same-center unnormalized 2D s-s pair:

#         I_ij(dz) = ∫ d^2ρ exp[-(ai+aj)ρ^2] / sqrt(ρ^2 + dz^2)

#     This routine returns ONLY the positive integral.
#     The minus sign from electron-nuclear attraction is applied later in
#     V_en_sp_total_at_z through:
#         V -= Z * V_nuc
#     """
#     gamma = alpha_i + alpha_j
#     x = np.sqrt(gamma) * abs(dz_abs)
#     return float((np.pi ** 1.5) / np.sqrt(gamma) * _erfcx_safe_paper(x))


def _ven_ss_same_center_exact(alpha_i, alpha_j, dz_abs):
    # previous expansion is replaced by stable erfcx
    gamma = alpha_i + alpha_j
    x = np.sqrt(gamma) * abs(dz_abs)
    return float((np.pi ** 1.5) / np.sqrt(gamma) * erfcx(x))

def _ven_single_numerical(alphas, centers, labels, i, j, nuc_xy, dz_abs):
    aA = alphas[i]
    xA, yA = centers[i]
    lxA, lyA = _parse_2d_l(labels[i])

    aB = alphas[j]
    xB, yB = centers[j]
    lxB, lyB = _parse_2d_l(labels[j])

    xN, yN = nuc_xy

    gamma = aA + aB
    Px = (aA * xA + aB * xB) / gamma
    Py = (aA * yA + aB * yB) / gamma

    sigma = 1.0 / np.sqrt(min(aA, aB))
    bound = 8.0 * sigma
    x_min, x_max = Px - bound, Px + bound
    y_min, y_max = Py - bound, Py + bound

    dz_sq = dz_abs * dz_abs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        val, err = dblquad(
            _exact_ven_integrand,
            x_min, x_max,
            lambda x: y_min, lambda x: y_max,
            args=(xA, yA, lxA, lyA, aA, xB, yB, lxB, lyB, aB, xN, yN, dz_sq),
            epsabs=1e-9,
            epsrel=1e-7,
        )
    return float(val)


def _ven_single_prony(alphas, centers, labels, i, j, nuc_xy, dz_abs):
    aA = alphas[i]
    xA, yA = centers[i]
    lxA, lyA = _parse_2d_l(labels[i])

    aB = alphas[j]
    xB, yB = centers[j]
    lxB, lyB = _parse_2d_l(labels[j])

    xn, yn = nuc_xy
    invz = 1.0 / abs(dz_abs)

    gamma = aA + aB
    P_x = (aA * xA + aB * xB) / gamma
    P_y = (aA * yA + aB * yB) / gamma

    def poly_int(l1, l2, q, c1, c2, zeta_val):
        q1 = q - c1
        q2 = q - c2
        res = 0.0
        for ii in range(l1 + 1):
            term1 = gamma_binom(l1, ii) * (q1 ** (l1 - ii))
            for jj in range(l2 + 1):
                term2 = gamma_binom(l2, jj) * (q2 ** (l2 - jj))
                k = ii + jj
                if k % 2 == 0:
                    m = k // 2
                    dfact = 1.0
                    for x in range(1, 2 * m, 2):
                        dfact *= x
                    res += term1 * term2 * dfact / ((2 * zeta_val) ** m)
        return res

    val_sum = 0.0
    for eta, xi in zip(ETAS, XIS):
        gam_p = xi * invz**2
        weight_p = eta * invz

        zeta = gamma + gam_p
        Qx = (gamma * P_x + gam_p * xn) / zeta
        Qy = (gamma * P_y + gam_p * yn) / zeta

        K_AB = np.exp(-aA * aB / gamma * ((xA - xB) ** 2 + (yA - yB) ** 2))
        K_PN = np.exp(-gamma * gam_p / zeta * ((P_x - xn) ** 2 + (P_y - yn) ** 2))

        pref = K_AB * K_PN * (np.pi / zeta) * weight_p
        Ix = poly_int(lxA, lxB, Qx, xA, xB, zeta)
        Iy = poly_int(lyA, lyB, Qy, yA, yB, zeta)

        val_sum += pref * Ix * Iy

    return float(val_sum)


# ============================================================
# V_en production
# ============================================================

def _V_en_prony_general(alphas, centers, labels, nuc_xy, dz_abs):
    """
    Original matrix-form Prony implementation kept for compatibility.
    """
    invz = 1.0 / abs(dz_abs)
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)
    xn, yn = nuc_xy

    for i in range(n):
        aA = alphas[i]
        xA, yA = centers[i]
        lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, n):
            aB = alphas[j]
            xB, yB = centers[j]
            lxB, lyB = _parse_2d_l(labels[j])

            gamma = aA + aB
            P_x = (aA * xA + aB * xB) / gamma
            P_y = (aA * yA + aB * yB) / gamma

            val_sum = 0.0
            for eta, xi in zip(ETAS, XIS):
                gam_p = xi * invz**2
                weight_p = eta * invz

                K_AB_x = np.exp(-aA * aB / gamma * (xA - xB) ** 2)
                K_AB_y = np.exp(-aA * aB / gamma * (yA - yB) ** 2)

                zeta = gamma + gam_p
                Qx = (gamma * P_x + gam_p * xn) / zeta
                Qy = (gamma * P_y + gam_p * yn) / zeta

                K_PN_x = np.exp(-gamma * gam_p / zeta * (P_x - xn) ** 2)
                K_PN_y = np.exp(-gamma * gam_p / zeta * (P_y - yn) ** 2)

                K_total = (K_AB_x * K_AB_y) * (K_PN_x * K_PN_y) * (np.pi / zeta) * weight_p

                def poly_int(l1, l2, q, c1, c2, zeta_val):
                    q1 = q - c1
                    q2 = q - c2
                    res = 0.0
                    for ii in range(l1 + 1):
                        term1 = gamma_binom(l1, ii) * (q1 ** (l1 - ii))
                        for jj in range(l2 + 1):
                            term2 = gamma_binom(l2, jj) * (q2 ** (l2 - jj))
                            k = ii + jj
                            if k % 2 == 0:
                                m = k // 2
                                dfact = 1.0
                                for x in range(1, 2 * m, 2):
                                    dfact *= x
                                res += term1 * term2 * dfact / ((2 * zeta_val) ** m)
                    return res

                Ix = poly_int(lxA, lxB, Qx, xA, xB, zeta)
                Iy = poly_int(lyA, lyB, Qy, yA, yB, zeta)

                val_sum += K_total * Ix * Iy

            out[i, j] = val_sum
            out[j, i] = val_sum
    return out


def V_en_sp_total_at_z(alphas, centers, labels, nuclei_tuples, z):
    """
    Hybrid V_en without spline tabulation.

    Logic per matrix element:
      1. same-center s-s on nuclear axis -> exact analytic
      2. off-center / non-s pair, small dz -> direct numerical
      3. off-center / non-s pair, large dz -> Prony
    """
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = len(alphas)

    V = np.zeros((N, N), dtype=float)

    for (Z, xN, yN, zN) in nuclei_tuples:
        dz = abs(z - zN)
        nuc_xy = np.array([xN, yN], dtype=float)

        V_nuc = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(i, N):
                same_axis_center = _pair_is_on_axis_same_center(centers, i, j, nuc_xy)

                if same_axis_center and labels[i].kind == "2d-s" and labels[j].kind == "2d-s":
                    val = _ven_ss_same_center_exact(alphas[i], alphas[j], dz)
                elif dz < OFFCENTER_NUMERICAL_CUTOFF:
                    val = _ven_single_numerical(alphas, centers, labels, i, j, nuc_xy, dz)
                else:
                    val = _ven_single_prony(alphas, centers, labels, i, j, nuc_xy, dz)

                V_nuc[i, j] = val
                V_nuc[j, i] = val

        V -= Z * V_nuc
    return V


# ============================================================
# ERI helpers
# ============================================================

def _assign_eri_symmetry(T, i, j, k, l, val):
    idxs = {
        (i, j, k, l),
        (j, i, k, l),
        (i, j, l, k),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, i, j),
        (k, l, j, i),
        (l, k, j, i),
    }
    for a, b, c, d in idxs:
        T[a, b, c, d] = val


def _eri_ssss_same_center_exact(alpha_i, alpha_j, alpha_k, alpha_l, dz_abs):
    """
    Same-center all-s ERI closed form for unnormalized primitives.

    Paper notation:
        alpha_ij = alpha_i + alpha_j
        alpha_kl = alpha_k + alpha_l
        Z = sqrt(alpha_ij * alpha_kl * dz^2 / (alpha_ij + alpha_kl))

    Unnormalized result:
        (ij|kl) = pi^(5/2) * erfcx(Z) / sqrt(alpha_ij * alpha_kl * (alpha_ij + alpha_kl))
    """
    aij = alpha_i + alpha_j
    akl = alpha_k + alpha_l

    Z = np.sqrt((aij * akl * dz_abs * dz_abs) / (aij + akl))

    val = (
        (np.pi ** 2.5) * erfcx(Z)
        / np.sqrt(aij * akl * (aij + akl))
    )
    return float(val)

def _eri_integrand_general(
    x1, y1, x2, y2,
    xA, yA, lxA, lyA, aA,
    xB, yB, lxB, lyB, aB,
    xC, yC, lxC, lyC, aC,
    xD, yD, lxD, lyD, aD,
    dz_sq
):
    """
    Real-space ERI integrand for unnormalized 2D Cartesian Gaussian primitives:
        (x-x0)^lx (y-y0)^ly exp(-a r^2)
    """
    gA = (x1 - xA) ** lxA * (y1 - yA) ** lyA * np.exp(-aA * ((x1 - xA) ** 2 + (y1 - yA) ** 2))
    gB = (x1 - xB) ** lxB * (y1 - yB) ** lyB * np.exp(-aB * ((x1 - xB) ** 2 + (y1 - yB) ** 2))
    gC = (x2 - xC) ** lxC * (y2 - yC) ** lyC * np.exp(-aC * ((x2 - xC) ** 2 + (y2 - yC) ** 2))
    gD = (x2 - xD) ** lxD * (y2 - yD) ** lyD * np.exp(-aD * ((x2 - xD) ** 2 + (y2 - yD) ** 2))

    kern = 1.0 / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + dz_sq)
    return gA * gB * gC * gD * kern


def _eri_ssss_hankel_reference(alpha_a, alpha_b, alpha_c, alpha_d, RA, RB, RC, RD, dz_abs):
    """
    Finite-dz numerical reference for the all-s case using the Hankel/Fourier form.
    Valid for arbitrary in-plane centers.
    """
    RA = np.asarray(RA, float)
    RB = np.asarray(RB, float)
    RC = np.asarray(RC, float)
    RD = np.asarray(RD, float)

    p = alpha_a + alpha_b
    q = alpha_c + alpha_d

    mu_ab = alpha_a * alpha_b / p
    mu_cd = alpha_c * alpha_d / q

    P = (alpha_a * RA + alpha_b * RB) / p
    Q = (alpha_c * RC + alpha_d * RD) / q

    Kab = np.exp(-mu_ab * np.sum((RA - RB) ** 2))
    Kcd = np.exp(-mu_cd * np.sum((RC - RD) ** 2))

    RPQ = np.linalg.norm(P - Q)
    pref = Kab * Kcd * (np.pi**2 / (p * q))

    def integrand(k):
        return np.exp(
            -dz_abs * k
            - k * k / (4.0 * p)
            - k * k / (4.0 * q)
        ) * j0(k * RPQ)

    val, err = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-8,
        limit=300,
    )
    return pref * val


def _gaussian_pair_window(alpha_a, alpha_b, Ra, Rb, nsigma=8.0):
    Ra = np.asarray(Ra, float)
    Rb = np.asarray(Rb, float)
    p = alpha_a + alpha_b
    P = (alpha_a * Ra + alpha_b * Rb) / p
    sigma = 1.0 / np.sqrt(p)
    bound = nsigma * sigma
    return P, bound


def _eri_general_numerical_reference(
    alpha_a, center_a, label_a,
    alpha_b, center_b, label_b,
    alpha_c, center_c, label_c,
    alpha_d, center_d, label_d,
    dz_abs
):
    xA, yA = center_a
    xB, yB = center_b
    xC, yC = center_c
    xD, yD = center_d

    lxA, lyA = _parse_2d_l(label_a)
    lxB, lyB = _parse_2d_l(label_b)
    lxC, lyC = _parse_2d_l(label_c)
    lxD, lyD = _parse_2d_l(label_d)

    P1, b1 = _gaussian_pair_window(alpha_a, alpha_b, center_a, center_b, nsigma=8.0)
    P2, b2 = _gaussian_pair_window(alpha_c, alpha_d, center_c, center_d, nsigma=8.0)

    x1_min, x1_max = P1[0] - b1, P1[0] + b1
    y1_min, y1_max = P1[1] - b1, P1[1] + b1
    x2_min, x2_max = P2[0] - b2, P2[0] + b2
    y2_min, y2_max = P2[1] - b2, P2[1] + b2

    dz_sq = dz_abs * dz_abs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        val, err = nquad(
            _eri_integrand_general,
            [
                [x1_min, x1_max],
                [y1_min, y1_max],
                [x2_min, x2_max],
                [y2_min, y2_max],
            ],
            args=(
                xA, yA, lxA, lyA, alpha_a,
                xB, yB, lxB, lyB, alpha_b,
                xC, yC, lxC, lyC, alpha_c,
                xD, yD, lxD, lyD, alpha_d,
                dz_sq,
            ),
            opts={"epsabs": 1e-8, "epsrel": 1e-6},
        )

    return float(val)


def _eri_exact_reference_element(
    alpha_a, center_a, label_a,
    alpha_b, center_b, label_b,
    alpha_c, center_c, label_c,
    alpha_d, center_d, label_d,
    dz_abs
):
    """
    Unified near-field ERI reference element.
    """
    if (
        label_a.kind == "2d-s"
        and label_b.kind == "2d-s"
        and label_c.kind == "2d-s"
        and label_d.kind == "2d-s"
        and _same_center(center_a, center_b)
        and _same_center(center_a, center_c)
        and _same_center(center_a, center_d)
    ):
        return _eri_ssss_same_center_exact(alpha_a, alpha_b, alpha_c, alpha_d, dz_abs)

    if (
        label_a.kind == "2d-s"
        and label_b.kind == "2d-s"
        and label_c.kind == "2d-s"
        and label_d.kind == "2d-s"
    ):
        return _eri_ssss_hankel_reference(
            alpha_a, alpha_b, alpha_c, alpha_d,
            center_a, center_b, center_c, center_d,
            dz_abs,
        )

    return _eri_general_numerical_reference(
        alpha_a, center_a, label_a,
        alpha_b, center_b, label_b,
        alpha_c, center_c, label_c,
        alpha_d, center_d, label_d,
        dz_abs,
    )


def _eri_numerical_tensor(alphas, centers, labels, dz_eff):
    """
    Direct numerical ERI tensor for rare off-center near-field cases.
    """
    n_ao = len(alphas)
    T = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)

    unique_quartets = []
    for i in range(n_ao):
        for j in range(i, n_ao):
            for k in range(n_ao):
                for l in range(k, n_ao):
                    if (i, j) > (k, l):
                        continue
                    unique_quartets.append((i, j, k, l))

    for i, j, k, l in unique_quartets:
        val = _eri_exact_reference_element(
            alphas[i], centers[i], labels[i],
            alphas[j], centers[j], labels[j],
            alphas[k], centers[k], labels[k],
            alphas[l], centers[l], labels[l],
            dz_eff,
        )
        _assign_eri_symmetry(T, i, j, k, l, val)

    return T


# ============================================================
# ERI exact / Prony branch helpers
# ============================================================

def _hermite_coefficients_general(l_a, l_b, alpha_a, alpha_b, A, B):
    pass


def _bessel_recursion_2d(t_max, u_max, Z, Delta_x, Delta_y, p_eff):
    """
    Stable recursion using scipy.special.ive to handle Z=0 singularity.
    """
    N_total = t_max + u_max
    n_batch = Z.shape[0]

    vals_n = {}
    X = -Z
    for n in range(N_total + 1):
        vals_n[n] = ive(n, X)

    I_tensor = np.zeros((t_max + 1, u_max + 1, N_total + 1, n_batch))
    for n in range(N_total + 1):
        I_tensor[0, 0, n] = ((-1.0) ** n) * vals_n[n]

    factor = -0.5 * p_eff

    def get_n_sum(arr_slice, n_target):
        i_n = arr_slice[n_target]
        i_np1 = arr_slice[n_target + 1]
        i_nm1 = arr_slice[1] if n_target == 0 else arr_slice[n_target - 1]
        return i_nm1 + 2 * i_n + i_np1

    for t in range(t_max):
        for n in range(N_total - t):
            sum_prev = get_n_sum(I_tensor[t - 1, 0], n) if t > 0 else 0.0
            sum_curr = get_n_sum(I_tensor[t, 0], n)
            I_tensor[t + 1, 0, n] = factor * (t * sum_prev + Delta_x * sum_curr)

    for t in range(t_max + 1):
        for u in range(u_max):
            for n in range(N_total - t - u):
                sum_prev = get_n_sum(I_tensor[t, u - 1], n) if u > 0 else 0.0
                sum_curr = get_n_sum(I_tensor[t, u], n)
                I_tensor[t, u + 1, n] = factor * (u * sum_prev + Delta_y * sum_curr)

    return I_tensor[:, :, 0]


def _eri_exact_bessel_tensor(alphas, centers, labels, dz_eff):
    n_ao = len(alphas)
    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    A = alphas[:, None] + alphas[None, :]
    Xi = (alphas[:, None] * alphas[None, :]) / A
    P_centers = (
        alphas[:, None, None] * centers[:, None, :]
        + alphas[None, :, None] * centers[None, :, :]
    ) / A[:, :, None]
    R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    K_ij = np.exp(-Xi * R_ij_sq)

    kind_to_L = {}
    for lbl in labels:
        kind_to_L[lbl.kind] = lbl.l

    inds_by_kind = {}
    for i, lbl in enumerate(labels):
        inds_by_kind.setdefault(lbl.kind, []).append(i)

    kinds = sorted(inds_by_kind.keys())

    for ki, kj, kk, kl in itertools.product(kinds, repeat=4):
        ii = inds_by_kind[ki]
        jj = inds_by_kind[kj]
        kk_ = inds_by_kind[kk]
        ll = inds_by_kind[kl]

        Lix, Liy = kind_to_L[ki][0:2]
        Ljx, Ljy = kind_to_L[kj][0:2]
        Lkx, Lky = kind_to_L[kk][0:2]
        Llx, Lly = kind_to_L[kl][0:2]

        A_bra = A[np.ix_(ii, jj)]
        P_bra = P_centers[np.ix_(ii, jj)]

        B_ket = A[np.ix_(kk_, ll)]
        Q_ket = P_centers[np.ix_(kk_, ll)]

        def get_hermites(la, lb, al_a, al_b, cnt_a, cnt_b, P_vec):
            gam = al_a + al_b
            inv_2g = 0.5 / gam
            shape = P_vec.shape
            Lmax = la + lb
            H_tab = np.zeros((la + 1, lb + 1) + shape + (Lmax + 1,))
            H_tab[0, 0, ..., 0] = 1.0
            for i in range(la + 1):
                for j in range(lb + 1):
                    if i == 0 and j == 0:
                        continue
                    if i > 0:
                        prev = H_tab[i - 1, j]
                        X = (P_vec - cnt_a)
                        H_tab[i, j, ..., :] += X[..., None] * prev
                        t_vals = np.arange(1, Lmax + 1)
                        H_tab[i, j, ..., :-1] += t_vals * prev[..., 1:]
                        H_tab[i, j, ..., 1:] += inv_2g[..., None] * prev[..., :-1]
                    else:
                        prev = H_tab[i, j - 1]
                        X = (P_vec - cnt_b)
                        H_tab[i, j, ..., :] += X[..., None] * prev
                        t_vals = np.arange(1, Lmax + 1)
                        H_tab[i, j, ..., :-1] += t_vals * prev[..., 1:]
                        H_tab[i, j, ..., 1:] += inv_2g[..., None] * prev[..., :-1]
            return H_tab[la, lb]

        Ex_bra = get_hermites(
            Lix, Ljx,
            alphas[ii][:, None], alphas[jj][None, :],
            centers[ii, 0][:, None], centers[jj, 0][None, :],
            P_bra[..., 0],
        )
        Ey_bra = get_hermites(
            Liy, Ljy,
            alphas[ii][:, None], alphas[jj][None, :],
            centers[ii, 1][:, None], centers[jj, 1][None, :],
            P_bra[..., 1],
        )

        Ex_ket = get_hermites(
            Lkx, Llx,
            alphas[kk_][:, None], alphas[ll][None, :],
            centers[kk_, 0][:, None], centers[ll, 0][None, :],
            Q_ket[..., 0],
        )
        Ey_ket = get_hermites(
            Lky, Lly,
            alphas[kk_][:, None], alphas[ll][None, :],
            centers[kk_, 1][:, None], centers[ll, 1][None, :],
            Q_ket[..., 1],
        )

        P_bd = P_bra[:, :, None, None, :]
        Q_bd = Q_ket[None, None, :, :, :]
        Delta = Q_bd - P_bd
        A_bd = A_bra[:, :, None, None]
        B_bd = B_ket[None, None, :, :]
        Sigma = (A_bd + B_bd) / (4.0 * A_bd * B_bd)

        dz2 = dz_eff ** 2
        R_xy_sq = np.sum(Delta**2, axis=-1)
        Z_arg = -(R_xy_sq + dz2) / (8.0 * Sigma)
        p_eff = 1.0 / (4.0 * Sigma)

        Tx_max = (Lix + Ljx) + (Lkx + Llx)
        Ty_max = (Liy + Ljy) + (Lky + Lly)

        I_tensor = _bessel_recursion_2d(
            Tx_max, Ty_max,
            Z_arg.ravel(),
            Delta[..., 0].ravel(),
            Delta[..., 1].ravel(),
            p_eff.ravel(),
        )
        I_tensor = I_tensor.reshape((Tx_max + 1, Ty_max + 1) + Z_arg.shape)

        K_bra = K_ij[np.ix_(ii, jj)][:, :, None, None]
        K_ket = K_ij[np.ix_(kk_, ll)][None, None, :, :]
        Pre = (np.pi**2 / (A_bd * B_bd)) * np.sqrt(np.pi / (4.0 * Sigma)) * K_bra * K_ket

        block_res = np.zeros(Z_arg.shape)
        for t in range(Ex_bra.shape[2]):
            for u in range(Ey_bra.shape[2]):
                for tau in range(Ex_ket.shape[2]):
                    for nu in range(Ey_ket.shape[2]):
                        C_bra = Ex_bra[:, :, t][:, :, None, None] * Ey_bra[:, :, u][:, :, None, None]
                        C_ket = Ex_ket[:, :, tau][None, None, :, :] * Ey_ket[:, :, nu][None, None, :, :]
                        block_res += C_bra * C_ket * I_tensor[t + tau, u + nu]

        eri_tensor[np.ix_(ii, jj, kk_, ll)] += Pre * block_res

    return eri_tensor


def _eri_prony_tensor(alphas, centers, labels, dz_eff):
    n_ao = len(alphas)
    A = alphas[:, None] + alphas[None, :]
    Xi = (alphas[:, None] * alphas[None, :]) / A
    P_centers = (
        alphas[:, None, None] * centers[:, None, :]
        + alphas[None, :, None] * centers[None, :, :]
    ) / A[:, :, None]
    R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    Pref = np.exp(-Xi * R_ij_sq)

    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)

    def get_ops(kind):
        if kind == "2d-s":
            return []
        if kind == "2d-px":
            return [(0, 0)]
        if kind == "2d-py":
            return [(1, 0)]
        if kind == "2d-dx2":
            return [(0, 0), (0, 0)]
        if kind == "2d-dy2":
            return [(1, 0), (1, 0)]
        if kind == "2d-dxy":
            return [(0, 0), (1, 0)]
        return []

    invz = 1.0 / dz_eff
    gammas = XIS * (invz**2)
    weights = ETAS * invz

    unique_kinds = sorted(list(set(l.kind for l in labels)))

    for weight_p, gamma_p in zip(weights, gammas):
        for k_i, k_j, k_k, k_l in itertools.product(unique_kinds, repeat=4):
            ii = [i for i, l in enumerate(labels) if l.kind == k_i]
            jj = [i for i, l in enumerate(labels) if l.kind == k_j]
            kk_ = [i for i, l in enumerate(labels) if l.kind == k_k]
            ll = [i for i, l in enumerate(labels) if l.kind == k_l]

            a_slice = A[np.ix_(ii, jj)]
            P_slice = P_centers[np.ix_(ii, jj)]
            Pre_ij = Pref[np.ix_(ii, jj)]
            alphas_i = alphas[ii]
            alphas_j = alphas[jj]
            Cs_i = centers[ii]
            Cs_j = centers[jj]

            b_slice = A[np.ix_(kk_, ll)]
            Q_slice = P_centers[np.ix_(kk_, ll)]
            Pre_kl = Pref[np.ix_(kk_, ll)]
            alphas_k = alphas[kk_]
            alphas_l = alphas[ll]
            Cs_k = centers[kk_]
            Cs_l = centers[ll]

            a_bd = a_slice[:, :, None, None]
            b_bd = b_slice[None, None, :, :]
            D = a_bd * b_bd + gamma_p * (a_bd + b_bd)
            Theta_p = (a_bd * b_bd * gamma_p) / D

            R_vec = P_slice[:, :, None, None, :] - Q_slice[None, None, :, :, :]
            R_sq = np.sum(R_vec**2, axis=-1)

            G_p = weight_p * (Pre_ij[:, :, None, None] * Pre_kl[None, None, :, :]) * (np.pi**2 / D) * np.exp(-Theta_p * R_sq)

            ops = []
            for op in get_ops(k_i):
                ops.append((op[0], 0))
            for op in get_ops(k_j):
                ops.append((op[0], 1))
            for op in get_ops(k_k):
                ops.append((op[0], 2))
            for op in get_ops(k_l):
                ops.append((op[0], 3))

            if not ops:
                eri_tensor[np.ix_(ii, jj, kk_, ll)] += G_p
                continue

            Forces = {}
            for axis in [0, 1]:
                fc = -2 * Xi[np.ix_(ii, jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_i[:, None, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 0)] = (fc + fp) / (2 * alphas_i[:, None, None, None])

                fc = +2 * Xi[np.ix_(ii, jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_j[None, :, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 1)] = (fc + fp) / (2 * alphas_j[None, :, None, None])

                fc = -2 * Xi[np.ix_(kk_, ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_k[None, None, :, None] / b_bd) * R_vec[..., axis]
                Forces[(axis, 2)] = (fc + fp) / (2 * alphas_k[None, None, :, None])

                fc = +2 * Xi[np.ix_(kk_, ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_l[None, None, None, :] / b_bd) * R_vec[..., axis]
                Forces[(axis, 3)] = (fc + fp) / (2 * alphas_l[None, None, None, :])

            def get_coupling_val(op1, op2):
                ax1, c1 = op1
                ax2, c2 = op2
                if ax1 != ax2:
                    return 0.0

                in_a1 = (c1 < 2)
                in_a2 = (c2 < 2)

                if in_a1 and in_a2:
                    return (1.0 / (2 * a_bd)) - (Theta_p / (2 * a_bd**2))
                elif (not in_a1) and (not in_a2):
                    return (1.0 / (2 * b_bd)) - (Theta_p / (2 * b_bd**2))
                else:
                    return Theta_p / (2 * a_bd * b_bd)

            def recursive_wicks_eval(current_ops):
                if not current_ops:
                    return 1.0
                head = current_ops[0]
                tail = current_ops[1:]
                val = Forces[head] * recursive_wicks_eval(tail)
                for i, other in enumerate(tail):
                    coupling = get_coupling_val(head, other)
                    remaining = tail[:i] + tail[i + 1:]
                    val += coupling * recursive_wicks_eval(remaining)
                return val

            factor = recursive_wicks_eval(ops)
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += factor * G_p

    return eri_tensor

def _all_s_same_center_basis(centers, labels, tol=CENTER_TOL):
    return _all_same_center(centers, tol=tol) and all(lbl.kind == "2d-s" for lbl in labels)

def _eri_all_s_same_center_tensor(alphas, centers, labels, dz_eff):
    """
    Exact ERI tensor for the special case:
      - all transverse centers identical
      - all primitives are 2d-s

    Uses the paper formula for unnormalized primitives.
    """
    n_ao = len(alphas)
    T = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)

    unique_quartets = []
    for i in range(n_ao):
        for j in range(i, n_ao):
            for k in range(n_ao):
                for l in range(k, n_ao):
                    if (i, j) > (k, l):
                        continue
                    unique_quartets.append((i, j, k, l))

    for i, j, k, l in unique_quartets:
        val = _eri_ssss_same_center_exact(
            alphas[i], alphas[j], alphas[k], alphas[l], dz_eff
        )
        _assign_eri_symmetry(T, i, j, k, l, val)

    return T

# ============================================================
# ERI production
# ============================================================

def eri_2d_cartesian_with_p(alphas, centers, labels, delta_z, dz_tol=None):
    """
    ERI production dispatch.

    Rules:
      1. If there is no off-center transverse integral involved and all basis
         functions are 2d-s at the same transverse center, use the paper's
         exact analytical formula.
      2. Otherwise, use the previous strategy:
           - small dz  -> numerical
           - large dz  -> Prony

    If dz_tol is not None, force the reference-style evaluation path.
    """
    dz_eff = abs(delta_z)
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    # Forced/reference mode for debugging or validation
    if dz_tol is not None:
        if _all_s_same_center_basis(centers, labels):
            return _eri_all_s_same_center_tensor(alphas, centers, labels, dz_eff)
        return _eri_numerical_tensor(alphas, centers, labels, dz_eff)

    # Production mode
    if _all_s_same_center_basis(centers, labels):
        return _eri_all_s_same_center_tensor(alphas, centers, labels, dz_eff)

    if dz_eff < OFFCENTER_NUMERICAL_CUTOFF:
        return _eri_numerical_tensor(alphas, centers, labels, dz_eff)

    return _eri_prony_tensor(alphas, centers, labels, dz_eff)

# ============================================================
# Misc helpers used elsewhere
# ============================================================

def pair_params(alphas, centers):
    n = len(alphas)
    a_i = alphas[:, None]
    a_j = alphas[None, :]
    A = a_i + a_j
    return A, None, None

def _permute_K_ikjl(K, n):
    return K.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)


def build_h1_nm(Kz, S_prim, T_prim, z_grid, V_en_of_z):
    Nz = int(Kz.shape[0])
    h1_nm = (Kz[:, :, None, None] * S_prim[None, None, :, :]).astype(float)
    for n in range(Nz):
        h1_nm[n, n] += (T_prim + V_en_of_z(float(z_grid[n])))
    for n in range(Nz):
        for m in range(Nz):
            H = h1_nm[n, m]
            h1_nm[n, m] = 0.5 * (H + H.T)
    return h1_nm


__all__ = [
    "PrimitiveLabel",
    "make_xy_spd_primitive_basis",
    "make_xy_sp_primitive_basis",
    "overlap_2d_cartesian",
    "kinetic_2d_cartesian",
    "V_en_sp_total_at_z",
    "eri_2d_cartesian_with_p",
    "build_h1_nm",
    "STO6_EXPS_H",
    "STO6_EXPS_He",
    "Exp_631g_ss_H",
    "ETAS",
    "XIS",
]
