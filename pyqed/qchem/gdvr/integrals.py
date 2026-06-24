import numpy as np
import itertools
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from scipy.special import ive, i0e, j0, erfc, erfcx
from scipy.integrate import dblquad, IntegrationWarning, nquad, quad
import scipy.integrate as integrate
import scipy.special as sp
import functools
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
# fitting parameters and standard primitive exponents
# ============================================================

# fitting coefficients
ETAS = np.array([8.0637411874e-05, 5.9071651716e-05, 2.1631763955e-11, 2.7007117198e-05, 2.921871996e-05, 2.3462024091e-05, 3.3604020319e-05, 2.6541249539e-05, 3.8959960849e-05, 5.9580149184e-05, 3.6220015997e-05, 1.6130285116e-05, 5.5531000607e-05, 1.0779060387e-05, 9.0655337406e-05, 3.5993102717e-05, 9.4026763659e-05, 0.00010977398014, 4.6481673803e-05, 9.8697654448e-05, 5.3449622483e-05, 0.00014574507241, 0.00021814909533, 4.3243421316e-05, 0.00011248770308, 0.00015489776045, 0.00033605480946, 0.00040367434404, 1.4658290397e-05, 4.5934621558e-05, 0.00037490436276, 0.00040014239759, 4.4091999996e-05, 0.00069247705621, 0.0001630362331, 0.00053649507021, 0.00042041269187, 0.00073944728354, 0.00018164522776, 0.0012406332248, 2.6515793023e-05, 0.0014863630369, 0.0003329649343, 0.0013047029239, 0.00089285450756, 0.0012665964508, 0.0012864005621, 0.0014283442432, 0.001539521378, 0.0018472440664, 0.0018962911302, 0.0020862775447, 0.0023436486368, 0.0027243248474, 0.0027510700386, 0.0031420194378, 0.0035487794405, 0.0038783937201, 0.0041560117744, 0.0047141802259, 0.005193470504, 0.0057120413849, 0.0062135482696, 0.0069907367473, 0.0075849249457, 0.0085908598682, 0.0090880547946, 0.010112890683, 0.012002976829, 0.010989212489, 0.016194341559, 0.010242150068, 0.024519056853, 0.0071122202252, 0.032922772694, 0.0070460373938, 0.040935906537, 0.0055191084455, 0.05307753015, 0.0009305156984, 0.067412714306, 0.072800036461, 0.0084424714866, 0.074363767089, 0.015132986704, 0.074058611939, 0.022742563495, 0.070532596778, 0.027336872376, 0.056664831444, 0.036853652824, 0.040842470239, 0.020739027964, 0.041546375037, 0.0061372018158, 0.016449646024, 0.011911936604, 0.00059741084555, 0.0039701340013, 0.00022114356367, 0.00027963196484, 3.9104180475e-05], dtype=np.float64)

XIS = np.array([1.0320393813e-08, 1.2379525379e-08, 1.3007410055e-08, 1.6173844457e-08, 2.1671426219e-08, 3.0819372215e-08, 3.9488488515e-08, 5.239397947e-08, 6.7352191656e-08, 8.199867748e-08, 1.0733871178e-07, 1.3993987549e-07, 1.5045864863e-07, 1.762693193e-07, 2.6022612644e-07, 3.1978827504e-07, 3.7015405181e-07, 4.4180362755e-07, 5.1259964405e-07, 6.4734980457e-07, 7.8859904928e-07, 1.0121060259e-06, 1.2460518649e-06, 1.5071795775e-06, 1.7569593479e-06, 2.1048930748e-06, 2.6798001247e-06, 3.9862598273e-06, 4.0618706894e-06, 4.3309530723e-06, 5.8091056772e-06, 7.2124239367e-06, 8.5351232982e-06, 1.0560606393e-05, 1.3123236965e-05, 1.5671620218e-05, 1.8965053526e-05, 2.3384323733e-05, 2.8679886374e-05, 3.445667567e-05, 4.0771200327e-05, 5.129473618e-05, 6.2422541922e-05, 7.6007428312e-05, 9.2400388351e-05, 0.00011253074895, 0.00013772902868, 0.00016738522086, 0.0002025720688, 0.00024804805106, 0.00030354482599, 0.00036692765669, 0.00044615253763, 0.00054724664794, 0.00066551100377, 0.00080668549248, 0.00098532060387, 0.0012018944792, 0.0014609396098, 0.00177938937, 0.0021671315854, 0.0026403416651, 0.0032190143368, 0.0039117409934, 0.0047632719365, 0.0058213807652, 0.0070739860415, 0.008589902442, 0.010505787482, 0.012808716845, 0.015531570234, 0.018922744994, 0.02311417046, 0.028130027156, 0.034201812286, 0.041679455593, 0.050808170794, 0.061935497033, 0.075274163361, 0.090705572507, 0.11175575826, 0.16571252954, 0.20009998894, 0.24584084036, 0.30181832979, 0.36389247352, 0.4412302909, 0.54292005309, 0.65869324876, 0.79548721709, 0.98645292185, 1.1988636469, 1.3983486011, 1.7784305553, 2.3439756222, 2.4930918729, 3.2727087091, 4.0240975602, 4.7625671235, 6.9486239367, 6.9497504653, 9.7367725097], dtype=np.float64)

# Standard Exp for H / He
STO6_EXPS_H = np.array([35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280], dtype=float)
STO6_EXPS_He = np.array([65.98456824, 12.09819836, 3.384639924, 1.162715163, 0.451516322, 0.185959356], dtype=float)
Exp_631g_ss_H = np.array([18.73113696, 2.825394365, 0.6401216923, 0.1612777588], dtype=float)


# ============================================================
# Global dispatch settings
# ============================================================

CENTER_TOL = 1e-14
OFFCENTER_NUMERICAL_CUTOFF = 0.05


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

def _ven_ss_reference(alphaA, alphaB, centerA, centerB, nuc_xy, dz):
    """
    Wraps existing _ven_single_numerical to evaluate a pure s-s pair. Delete after full benchmark
    """
    class DummyLabel:
        def __init__(self):
            self.kind = '2d-s'
            self.dim = 2
            self.l = (0,0)
            
    dummy_alphas = np.array([alphaA, alphaB], dtype=float)
    dummy_centers = np.array([centerA, centerB], dtype=float)
    dummy_labels = [DummyLabel(), DummyLabel()]
    
    return _ven_single_numerical(dummy_alphas, dummy_centers, dummy_labels, 0, 1, nuc_xy, dz)

@functools.lru_cache(maxsize=None)
def _cached_ven_ss_hankel(alphaA, alphaB, cAx, cAy, cBx, cBy, nuc_x, nuc_y, dz):
    gamma = alphaA + alphaB
    P_x = (alphaA * cAx + alphaB * cBx) / gamma
    P_y = (alphaA * cAy + alphaB * cBy) / gamma
    
    AB_sq = (cAx - cBx)**2 + (cAy - cBy)**2
    K_AB = np.exp(- (alphaA * alphaB / gamma) * AB_sq)
    
    D = np.sqrt((P_x - nuc_x)**2 + (P_y - nuc_y)**2)
    
    def integrand(k):
        return np.exp(-k * dz) * np.exp(-k**2 / (4.0 * gamma)) * sp.jv(0, k * D)
    
    val, _ = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
    return K_AB * val * (np.pi / gamma)

def _ven_ss_hankel(alphaA, alphaB, centerA, centerB, nuc_xy, dz):
    """
    Unwraps the arrays to pass to the cached function.
    """
    return _cached_ven_ss_hankel(
        alphaA, alphaB, 
        centerA[0], centerA[1], 
        centerB[0], centerB[1], 
        nuc_xy[0], nuc_xy[1], 
        dz
    )

# The Ven FD Engine
def _fd_ven_recursive(alphas, centers, kinds, nuc_xy, dz, h=5e-4):
    """
    Recursively evaluates exact Ven via Finite Differences of the s-orbital baseline.
    Correctly implements the Gaussian derivative recurrence relation.
    """
    # fallback to all s orbitals
    if all(k == '2d-s' for k in kinds):
        return _ven_ss_hankel(
            alphas[0], alphas[1], centers[0], centers[1], nuc_xy, dz
        )
        
    # Find the first orbital that has higher angular momentum
    for idx, kind in enumerate(kinds):
        if kind != '2d-s':
            new_kinds = list(kinds)
            alpha = alphas[idx]
            
            # Helper to evaluate the derivative shift
            def eval_shifted(shift_x, shift_y):
                c_shifted = np.copy(centers)
                c_shifted[idx][0] += shift_x
                c_shifted[idx][1] += shift_y
                return _fd_ven_recursive(alphas, c_shifted, new_kinds, nuc_xy, dz, h)

            # Helper to evaluate the lower-order recurrence term
            def eval_lower(lower_kind):
                lk_list = list(kinds) # Fresh copy to avoid mutating the current state
                lk_list[idx] = lower_kind
                return _fd_ven_recursive(alphas, centers, lk_list, nuc_xy, dz, h)
            
            # p-orbitals
            if kind == '2d-px':
                new_kinds[idx] = '2d-s'
                return (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                
            elif kind == '2d-py':
                new_kinds[idx] = '2d-s'
                return (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                
            # d-orbitals
            elif kind == '2d-dxy':
                new_kinds[idx] = '2d-py'
                return (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                
            elif kind == '2d-dx2':
                new_kinds[idx] = '2d-px'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-s') / (2.0 * alpha)
                
            elif kind == '2d-dy2':
                new_kinds[idx] = '2d-py'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-s') / (2.0 * alpha)
                
            # f-orbitals
            elif kind == '2d-fx3':
                new_kinds[idx] = '2d-dx2'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + 2.0 * eval_lower('2d-px') / (2.0 * alpha)
                
            elif kind == '2d-fx2y':
                # Derived from dxy via x-derivative
                new_kinds[idx] = '2d-dxy'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-py') / (2.0 * alpha)
                
            elif kind == '2d-fxy2':
                # Derived from dxy via y-derivative
                new_kinds[idx] = '2d-dxy'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-px') / (2.0 * alpha)
                
            elif kind == '2d-fy3':
                new_kinds[idx] = '2d-dy2'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + 2.0 * eval_lower('2d-py') / (2.0 * alpha)
                
    unsupported = [k for k in kinds if k != '2d-s']
    raise NotImplementedError(
        f"Unsupported angular momentum in FD engine: {unsupported}. "
        f"This engine currently only supports s, p, d, and f orbitals. "
        f"Full input kinds: {kinds}"
    )


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


def _coerce_local_ecp_terms(local_ecp_terms, nnuc):
    if local_ecp_terms is None:
        return [[] for _ in range(nnuc)]
    if len(local_ecp_terms) != nnuc:
        raise ValueError("local_ecp_terms must match the number of nuclei.")

    out = []
    for atom_terms in local_ecp_terms:
        clean_terms = []
        for term in atom_terms:
            if isinstance(term, dict):
                power = term["power"]
                exponent = term["exponent"]
                coeff = term["coeff"]
            else:
                power, exponent, coeff = term[:3]
            exponent = float(exponent)
            if exponent < 0.0:
                raise ValueError("ECP Gaussian exponents must be non-negative.")
            clean_terms.append((int(power), exponent, float(coeff)))
        out.append(tuple(clean_terms))
    return out


def _coerce_semilocal_ecp_terms(semilocal_ecp_terms, nnuc):
    if semilocal_ecp_terms is None:
        return [[] for _ in range(nnuc)]
    if len(semilocal_ecp_terms) != nnuc:
        raise ValueError("semilocal_ecp_terms must match the number of nuclei.")

    out = []
    for atom_terms in semilocal_ecp_terms:
        clean_terms = []
        for term in atom_terms:
            if isinstance(term, dict):
                angular_momentum = term["angular_momentum"]
                power = term["power"]
                exponent = term["exponent"]
                coeff = term["coeff"]
            else:
                angular_momentum, power, exponent, coeff = term[:4]
            exponent = float(exponent)
            if exponent < 0.0:
                raise ValueError("ECP Gaussian exponents must be non-negative.")
            clean_terms.append((int(angular_momentum), int(power), exponent, float(coeff)))
        out.append(tuple(clean_terms))
    return out


@functools.lru_cache(maxsize=64)
def _angle_even_integral(nx, ny):
    nx = int(nx)
    ny = int(ny)
    if nx < 0 or ny < 0:
        raise ValueError("Angular powers must be non-negative.")
    if (nx % 2) or (ny % 2):
        return 0.0
    ax = nx // 2
    ay = ny // 2
    log_val = (
        np.log(2.0)
        + sp.gammaln(ax + 0.5)
        + sp.gammaln(ay + 0.5)
        - sp.gammaln(ax + ay + 1.0)
    )
    return float(np.exp(log_val))


@functools.lru_cache(maxsize=512)
def _genlaguerre_nodes_weights(order, alpha):
    nodes, weights = sp.roots_genlaguerre(int(order), float(alpha))
    return np.asarray(nodes, float), np.asarray(weights, float)


@functools.lru_cache(maxsize=65536)
def _local_ecp_same_axis_radial(total_power, total_exponent, ecp_exponent, dz_abs, ecp_power, order=96):
    total_power = int(total_power)
    total_exponent = float(total_exponent)
    ecp_exponent = float(ecp_exponent)
    dz_abs = float(abs(dz_abs))
    ecp_power = int(ecp_power)
    if total_exponent <= 0.0:
        raise ValueError("Total Gaussian exponent must be positive.")

    d2 = dz_abs * dz_abs
    prefactor = float(np.exp(-ecp_exponent * d2))
    if d2 < 1e-28:
        alpha = 0.5 * (total_power + ecp_power)
        if alpha <= -1.0:
            raise ValueError("Divergent same-axis local ECP radial integral.")
        return float(0.5 * np.exp(sp.gammaln(alpha + 1.0)) / (total_exponent ** (alpha + 1.0)))

    if ecp_power == 0:
        alpha = 0.5 * total_power
        return float(
            0.5
            * prefactor
            * np.exp(sp.gammaln(alpha + 1.0))
            / (total_exponent ** (alpha + 1.0))
        )

    alpha = 0.5 * total_power
    nodes, weights = _genlaguerre_nodes_weights(int(order), float(alpha))
    vals = (nodes / total_exponent + d2) ** (0.5 * ecp_power)
    return float(
        0.5
        * prefactor
        * np.sum(weights * vals)
        / (total_exponent ** (alpha + 1.0))
    )


def _local_ecp_single_same_axis(alphas, centers, labels, i, j, nuc_xy, dz_abs, power, exponent):
    if not _pair_is_on_axis_same_center(centers, i, j, nuc_xy):
        return None

    lxA, lyA = _parse_2d_l(labels[i])
    lxB, lyB = _parse_2d_l(labels[j])
    nx = lxA + lxB
    ny = lyA + lyB
    angle = _angle_even_integral(nx, ny)
    if angle == 0.0:
        return 0.0

    radial = _local_ecp_same_axis_radial(
        nx + ny,
        float(alphas[i] + alphas[j] + exponent),
        float(exponent),
        float(abs(dz_abs)),
        int(power),
    )
    return angle * radial


def _local_ecp_integrand(
    y, x,
    xA, yA, lxA, lyA, aA,
    xB, yB, lxB, lyB, aB,
    xN, yN, dz_sq, power, exponent
):
    dist_sq = (x - xN) ** 2 + (y - yN) ** 2 + dz_sq
    r = np.sqrt(max(dist_sq, 1e-300))
    valA = (x - xA) ** lxA * (y - yA) ** lyA * np.exp(-aA * ((x - xA) ** 2 + (y - yA) ** 2))
    valB = (x - xB) ** lxB * (y - yB) ** lyB * np.exp(-aB * ((x - xB) ** 2 + (y - yB) ** 2))
    return valA * valB * (r ** power) * np.exp(-exponent * dist_sq)


def _local_ecp_single_numerical(alphas, centers, labels, i, j, nuc_xy, dz_abs, power, exponent):
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

    width_exp = min(float(aA), float(aB), max(float(exponent), 1e-12))
    bound = 8.0 / np.sqrt(width_exp)
    x_min = min(Px, xN) - bound
    x_max = max(Px, xN) + bound
    y_min = min(Py, yN) - bound
    y_max = max(Py, yN) + bound

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        val, _err = dblquad(
            _local_ecp_integrand,
            x_min, x_max,
            lambda x: y_min, lambda x: y_max,
            args=(xA, yA, lxA, lyA, aA, xB, yB, lxB, lyB, aB, xN, yN, dz_abs * dz_abs, power, exponent),
            epsabs=1e-9,
            epsrel=1e-7,
        )
    return float(val)


def local_ecp_potential_at_z(alphas, centers, labels, nuc_xy, dz_abs, terms):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = len(alphas)
    V = np.zeros((N, N), dtype=float)
    nuc_xy = np.asarray(nuc_xy, dtype=float)

    for power, exponent, coeff in terms:
        if coeff == 0.0:
            continue
        for i in range(N):
            for j in range(i, N):
                val = _local_ecp_single_same_axis(
                    alphas, centers, labels, i, j, nuc_xy, dz_abs, power, exponent
                )
                if val is None:
                    val = _local_ecp_single_numerical(
                        alphas, centers, labels, i, j, nuc_xy, dz_abs, power, exponent
                    )
                V[i, j] += coeff * val
                if i != j:
                    V[j, i] += coeff * val
    return V


def _fourier_dvr_cardinal_values(x, z_grid, dz):
    z_grid = np.asarray(z_grid, float)
    x = np.asarray(x, float)
    N = int(z_grid.size)
    if N <= 0:
        raise ValueError("Empty z_grid for Fourier DVR cardinal values.")
    box = float(N * dz)
    if box <= 0.0:
        raise ValueError("Fourier DVR spacing must be positive.")
    u = (x[..., None] - z_grid[None, :]) / box
    u = u - np.round(u)
    small = np.abs(u) < 1e-13
    if N % 2 == 0:
        denom = np.tan(np.pi * u)
    else:
        denom = np.sin(np.pi * u)
    numer = np.sin(np.pi * N * u)
    vals = np.empty_like(u, dtype=float)
    np.divide(numer, float(N) * denom, out=vals, where=~small)
    vals[small] = 1.0
    return vals / np.sqrt(float(dz))


def _same_axis_projector_supported(centers, labels, nuc_xy):
    nuc_xy = np.asarray(nuc_xy, float)
    for center in np.asarray(centers, float):
        if not _same_center(center, nuc_xy):
            return False
    for lbl in labels:
        lx, ly = _parse_2d_l(lbl)
        if lx + ly > 2:
            return False
    return True


def _l0_phi_integral_factor(label, rho_sq):
    lx, ly = _parse_2d_l(label)
    if lx == 0 and ly == 0:
        return 2.0 * np.pi * np.ones_like(rho_sq)
    if (lx == 1 and ly == 0) or (lx == 0 and ly == 1) or (lx == 1 and ly == 1):
        return np.zeros_like(rho_sq)
    if (lx == 2 and ly == 0) or (lx == 0 and ly == 2):
        return np.pi * rho_sq
    return np.zeros_like(rho_sq)


def _semilocal_l0_amplitudes_same_axis(
    r,
    z_grid,
    dz,
    z_nuc,
    alphas,
    labels,
    mu_nodes,
    mu_weights,
):
    r = float(r)
    mu = np.asarray(mu_nodes, float)
    z_eval = float(z_nuc) + r * mu
    cardinal = _fourier_dvr_cardinal_values(z_eval, z_grid, dz)
    rho_sq = (r * r) * (1.0 - mu * mu)
    pref_y00 = 1.0 / np.sqrt(4.0 * np.pi)
    amps = np.zeros((len(z_grid), len(alphas)), dtype=float)
    for ip, (alpha, lbl) in enumerate(zip(alphas, labels)):
        phi_factor = _l0_phi_integral_factor(lbl, rho_sq)
        if not np.any(phi_factor):
            continue
        transverse = phi_factor * np.exp(-float(alpha) * rho_sq)
        weighted = np.asarray(mu_weights, float) * transverse
        amps[:, ip] = pref_y00 * (cardinal.T @ weighted)
    return amps


def _semilocal_l0_amplitudes_same_axis_batch(
    r_nodes,
    z_grid,
    dz,
    z_nuc,
    alphas,
    labels,
    mu_nodes,
    mu_weights,
):
    r_nodes = np.asarray(r_nodes, float)
    mu = np.asarray(mu_nodes, float)
    z_eval = float(z_nuc) + r_nodes[:, None] * mu[None, :]
    cardinal = _fourier_dvr_cardinal_values(z_eval.reshape(-1), z_grid, dz)
    cardinal = cardinal.reshape(r_nodes.size, mu.size, len(z_grid))
    rho_sq = (r_nodes[:, None] ** 2) * (1.0 - mu[None, :] ** 2)
    pref_y00 = 1.0 / np.sqrt(4.0 * np.pi)

    factors = np.zeros((len(alphas), r_nodes.size, mu.size), dtype=float)
    for ip, lbl in enumerate(labels):
        factors[ip] = _l0_phi_integral_factor(lbl, rho_sq)
    if not np.any(factors):
        return np.zeros((r_nodes.size, len(z_grid), len(alphas)), dtype=float)

    transverse = factors * np.exp(-np.asarray(alphas, float)[:, None, None] * rho_sq[None, :, :])
    transverse *= np.asarray(mu_weights, float)[None, None, :]
    amps = pref_y00 * np.einsum("ran,pra->rnp", cardinal, transverse, optimize=True)
    return amps


def semilocal_ecp_projector_blocks(
    alphas,
    centers,
    labels,
    nuclei_tuples,
    z_grid,
    dz,
    semilocal_ecp_terms,
    dvr_method="exp",
    radial_order=96,
    angular_order=64,
    radial_cutoff_tol=1.0e-14,
):
    """Return primitive GDVR blocks for same-axis semilocal ECP projectors.

    The current implementation targets the Li/BFD case used by the GDVR Li2
    examples: Fourier DVR along z, all transverse centers on the ECP axis, and
    semilocal l=0 radial channels.  Unsupported geometries fail loudly instead
    of silently falling back to a scalar approximation.
    """
    terms_by_atom = _coerce_semilocal_ecp_terms(semilocal_ecp_terms, len(nuclei_tuples))
    if not any(terms_by_atom):
        return None
    if str(dvr_method) != "exp":
        raise NotImplementedError("Semilocal ECP projectors currently require Fourier DVR (dvr_method='exp').")

    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    z_grid = np.asarray(z_grid, float)
    Nz = int(z_grid.size)
    nprim = int(alphas.size)
    blocks = np.zeros((Nz, Nz, nprim, nprim), dtype=float)
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(angular_order))
    dz = float(dz)

    def _accumulate_terms(target, terms, zN):
        for angular_momentum, power, exponent, coeff in terms:
            if coeff == 0.0:
                continue
            if int(angular_momentum) != 0:
                raise NotImplementedError("Semilocal ECP projectors currently support only l=0 channels.")
            if exponent <= 0.0:
                raise ValueError("Semilocal ECP exponents must be positive.")
            rmax = np.sqrt(-np.log(float(radial_cutoff_tol)) / float(exponent))
            r_nodes, r_weights = np.polynomial.legendre.leggauss(int(radial_order))
            r_nodes = 0.5 * rmax * (r_nodes + 1.0)
            r_weights = 0.5 * rmax * r_weights
            amps = _semilocal_l0_amplitudes_same_axis_batch(
                r_nodes,
                z_grid,
                dz,
                zN,
                alphas,
                labels,
                mu_nodes,
                mu_weights,
            ).reshape(r_nodes.size, Nz * nprim)
            radial_weights = (
                r_weights
                * coeff
                * (r_nodes ** (power + 2))
                * np.exp(-exponent * r_nodes * r_nodes)
            )
            gram = amps.T @ (radial_weights[:, None] * amps)
            target += gram.reshape(Nz, nprim, Nz, nprim).transpose(0, 2, 1, 3)

    active_atoms = [
        (tuple(terms), float(xN), float(yN), float(zN))
        for terms, (_Z, xN, yN, zN) in zip(terms_by_atom, nuclei_tuples)
        if terms
    ]
    if active_atoms:
        ref_terms, ref_x, ref_y, ref_z = active_atoms[0]
        same_translated_projector = (
            dz > 0.0
            and all(terms == ref_terms for terms, _x, _y, _z in active_atoms)
            and all(abs(x - ref_x) < CENTER_TOL and abs(y - ref_y) < CENTER_TOL for _terms, x, y, _z in active_atoms)
        )
        shifts = []
        if same_translated_projector:
            for _terms, _x, _y, zN in active_atoms:
                shift = int(round((zN - ref_z) / dz))
                if abs((zN - ref_z) - shift * dz) > 1.0e-8:
                    same_translated_projector = False
                    break
                shifts.append(shift)
        if same_translated_projector:
            nuc_xy = np.array([ref_x, ref_y], dtype=float)
            if not _same_axis_projector_supported(centers, labels, nuc_xy):
                raise NotImplementedError(
                    "Semilocal ECP projectors currently support only transverse primitives centered on the ECP axis."
                )
            ref_blocks = np.zeros_like(blocks)
            _accumulate_terms(ref_blocks, ref_terms, ref_z)
            for shift in shifts:
                blocks += np.roll(np.roll(ref_blocks, shift, axis=0), shift, axis=1)
            for n in range(Nz):
                for m in range(n, Nz):
                    blk = 0.5 * (blocks[n, m] + blocks[m, n].T)
                    blocks[n, m] = blk
                    blocks[m, n] = blk.T
            return blocks

    for terms, (_Z, xN, yN, zN) in zip(terms_by_atom, nuclei_tuples):
        if not terms:
            continue
        nuc_xy = np.array([xN, yN], dtype=float)
        if not _same_axis_projector_supported(centers, labels, nuc_xy):
            raise NotImplementedError(
                "Semilocal ECP projectors currently support only transverse primitives centered on the ECP axis."
            )
        _accumulate_terms(blocks, terms, zN)

    for n in range(Nz):
        for m in range(n, Nz):
            blk = 0.5 * (blocks[n, m] + blocks[m, n].T)
            blocks[n, m] = blk
            blocks[m, n] = blk.T
    return blocks


def V_en_sp_total_at_z(
    alphas,
    centers,
    labels,
    nuclei_tuples,
    z,
    softcore_radii=None,
    local_ecp_terms=None,
    matrix_cache=None,
):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = len(alphas)
    V = np.zeros((N, N), dtype=float)
    if softcore_radii is None:
        softcore_radii = np.zeros(len(nuclei_tuples), dtype=float)
    else:
        softcore_radii = np.asarray(softcore_radii, dtype=float).reshape(-1)
        if softcore_radii.size != len(nuclei_tuples):
            raise ValueError("softcore_radii must match the number of nuclei.")
        if np.any(softcore_radii < 0.0):
            raise ValueError("softcore_radii must be non-negative.")
    local_ecp_terms = _coerce_local_ecp_terms(local_ecp_terms, len(nuclei_tuples))

    for inucl, (Z, xN, yN, zN) in enumerate(nuclei_tuples):
        dz0 = float(z) - float(zN)
        rc = float(softcore_radii[inucl])
        dz = float(np.sqrt(dz0 * dz0 + rc * rc))
        nuc_xy = np.array([xN, yN], dtype=float)
        cache_key = None
        if matrix_cache is not None:
            cache_key = (
                round(float(Z), 12),
                round(float(xN), 12),
                round(float(yN), 12),
                round(float(dz), 12),
                round(float(abs(dz0)), 12),
                tuple(local_ecp_terms[inucl]),
            )
            cached = matrix_cache.get(cache_key)
            if cached is not None:
                V += cached
                continue
        V_nuc = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(i, N):
                same_axis_center = _pair_is_on_axis_same_center(centers, i, j, nuc_xy)
                has_higher_l = (labels[i].kind != "2d-s" or labels[j].kind != "2d-s")

                if same_axis_center and not has_higher_l:
                    # 1. Exact Analytical Co-centered
                    val = _ven_ss_same_center_exact(alphas[i], alphas[j], dz)
                elif dz < OFFCENTER_NUMERICAL_CUTOFF:
                    pair_alphas = [alphas[i], alphas[j]]
                    pair_centers = [centers[i], centers[j]]
                    pair_kinds = [labels[i].kind, labels[j].kind]
                    # Use the new FD Engine for p/d orbitals!
                    val = _fd_ven_recursive(pair_alphas, pair_centers, pair_kinds, nuc_xy, dz)
                else:
                    # 3. Far-field -> Global Prony Fit
                    val = _ven_single_prony(alphas, centers, labels, i, j, nuc_xy, dz)

                V_nuc[i, j] = val
                V_nuc[j, i] = val

        contribution = -Z * V_nuc
        if local_ecp_terms[inucl]:
            contribution += local_ecp_potential_at_z(
                alphas,
                centers,
                labels,
                nuc_xy,
                abs(dz0),
                local_ecp_terms[inucl],
            )
        V += contribution
        if cache_key is not None:
            matrix_cache[cache_key] = contribution
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

def _eri_exact_fd_reference(alphas, centers, labels, dz_eff, h=5e-4):
    """
    Builds the full ERI tensor using the exact Finite Difference recursive engine.
    This safely bypasses SciPy's 4D nquad entirely.
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
        q_alphas = [alphas[i], alphas[j], alphas[k], alphas[l]]
        q_centers = [centers[i], centers[j], centers[k], centers[l]]
        q_kinds = [labels[i].kind, labels[j].kind, labels[k].kind, labels[l].kind]
        
        val = _fd_eri_recursive(q_alphas, q_centers, q_kinds, dz_eff, h)
        _assign_eri_symmetry(T, i, j, k, l, val)

    return T


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


@functools.lru_cache(maxsize=None)
def _cached_eri_ssss_hankel(alpha_a, alpha_b, alpha_c, alpha_d, 
                            Rax, Ray, Rbx, Rby, Rcx, Rcy, Rdx, Rdy, dz_abs):
    
    RA = np.array([Rax, Ray], float)
    RB = np.array([Rbx, Rby], float)
    RC = np.array([Rcx, Rcy], float)
    RD = np.array([Rdx, Rdy], float)

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
        return np.exp(-dz_abs * k - k * k / (4.0 * p) - k * k / (4.0 * q)) * j0(k * RPQ)

    val, err = quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=300)
    return pref * val

def _eri_ssss_hankel_reference(alpha_a, alpha_b, alpha_c, alpha_d, RA, RB, RC, RD, dz_abs):
    """
    Unwraps the arrays to pass to the cached function.
    """
    return _cached_eri_ssss_hankel(
        alpha_a, alpha_b, alpha_c, alpha_d,
        RA[0], RA[1], RB[0], RB[1], RC[0], RC[1], RD[0], RD[1], dz_abs
    )
def _gaussian_pair_window(alpha_a, alpha_b, Ra, Rb, nsigma=8.0):
    Ra = np.asarray(Ra, float)
    Rb = np.asarray(Rb, float)
    p = alpha_a + alpha_b
    P = (alpha_a * Ra + alpha_b * Rb) / p
    sigma = 1.0 / np.sqrt(p)
    bound = nsigma * sigma
    return P, bound

def _fd_eri_recursive(alphas, centers, kinds, dz, h=5e-4):
    """
    Recursively evaluates exact ERI via Finite Differences of the 1D Hankel transform.
    Correctly implements the Gaussian derivative recurrence relation for p, d, and f orbitals.
    """
    # Base Case: All orbitals are '2d-s'. Evaluate the fast 1D Hankel!
    if all(k == '2d-s' for k in kinds):
        return _eri_ssss_hankel_reference(
            alphas[0], alphas[1], alphas[2], alphas[3],
            centers[0], centers[1], centers[2], centers[3], dz
        )
        
    # Recursive Case: Find the first orbital that has higher angular momentum
    for idx, kind in enumerate(kinds):
        if kind != '2d-s':
            new_kinds = list(kinds)
            alpha = alphas[idx]
            
            #  Helper to evaluate the derivative shift 
            def eval_shifted(shift_x, shift_y):
                c_shifted = np.copy(centers)
                c_shifted[idx][0] += shift_x
                c_shifted[idx][1] += shift_y
                return _fd_eri_recursive(alphas, c_shifted, new_kinds, dz, h)

            #  Helper to evaluate the lower-order recurrence term 
            def eval_lower(lower_kind):
                lk_list = list(kinds)
                lk_list[idx] = lower_kind
                return _fd_eri_recursive(alphas, centers, lk_list, dz, h)
            
            # p-orbitals
            if kind == '2d-px':
                new_kinds[idx] = '2d-s'
                return (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                
            elif kind == '2d-py':
                new_kinds[idx] = '2d-s'
                return (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                
            # d-orbitals
            elif kind == '2d-dxy':
                new_kinds[idx] = '2d-py'
                return (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                
            elif kind == '2d-dx2':
                new_kinds[idx] = '2d-px'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-s') / (2.0 * alpha)
                
            elif kind == '2d-dy2':
                new_kinds[idx] = '2d-py'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-s') / (2.0 * alpha)
                
            # f-orbitals
            elif kind == '2d-fx3':
                new_kinds[idx] = '2d-dx2'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + 2.0 * eval_lower('2d-px') / (2.0 * alpha)
                
            elif kind == '2d-fx2y':
                # Derived from dxy via x-derivative
                new_kinds[idx] = '2d-dxy'
                deriv = (eval_shifted(h, 0) - eval_shifted(-h, 0)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-py') / (2.0 * alpha)
                
            elif kind == '2d-fxy2':
                # Derived from dxy via y-derivative
                new_kinds[idx] = '2d-dxy'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + eval_lower('2d-px') / (2.0 * alpha)
                
            elif kind == '2d-fy3':
                new_kinds[idx] = '2d-dy2'
                deriv = (eval_shifted(0, h) - eval_shifted(0, -h)) / (4.0 * alpha * h)
                return deriv + 2.0 * eval_lower('2d-py') / (2.0 * alpha)
                
    unsupported = [k for k in kinds if k != '2d-s']
    raise NotImplementedError(
        f"Unsupported angular momentum in FD engine: {unsupported}. "
        f"This engine currently only supports s, p, d, and f orbitals. "
        f"Full input kinds: {kinds}"
    )


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

def _same_center_sp_basis(centers, labels, tol=CENTER_TOL):
    allowed = {"2d-s", "2d-px", "2d-py"}
    return _all_same_center(centers, tol=tol) and all(lbl.kind in allowed for lbl in labels)


def _small_comb(n, k):
    n = np.asarray(n, dtype=int)
    if k == 0:
        return np.ones_like(n, dtype=float)
    if k == 1:
        return n.astype(float)
    if k == 2:
        return 0.5 * n * (n - 1)
    return np.zeros_like(n, dtype=float)


def _same_center_u_moments(mu, dz_abs):
    mu = np.asarray(mu, float)
    a = float(abs(dz_abs))
    root_mu = np.sqrt(mu)
    F = erfcx(a * root_mu)
    c = np.pi ** 1.5

    U0 = c * F / root_mu
    U2 = (
        0.5 * c * F / (mu * root_mu)
        - c * a * a * F / root_mu
        + np.pi * a / mu
    )
    U4 = (
        c * (a ** 4) * F / root_mu
        - np.pi * (a ** 3) / mu
        - c * a * a * F / (mu * root_mu)
        + 1.5 * np.pi * a / (mu * mu)
        + 0.75 * c * F / (mu * mu * root_mu)
    )
    return U0, U2, U4


def _even_gaussian_moment_2d(px, py, exponent):
    px = np.asarray(px, dtype=int)
    py = np.asarray(py, dtype=int)
    exponent = np.asarray(exponent, float)
    out = np.zeros(np.broadcast_shapes(px.shape, py.shape, exponent.shape), dtype=float)
    bx = np.broadcast_to(px, out.shape)
    by = np.broadcast_to(py, out.shape)
    be = np.broadcast_to(exponent, out.shape)
    mask = ((bx % 2) == 0) & ((by % 2) == 0)
    if not np.any(mask):
        return out
    ax = bx[mask] // 2
    ay = by[mask] // 2
    out[mask] = np.exp(
        sp.gammaln(ax + 0.5)
        + sp.gammaln(ay + 0.5)
        - (ax + ay + 1.0) * np.log(be[mask])
    )
    return out


def _same_center_radial_moment_2d(px, py, U0, U2, U4):
    px = np.asarray(px, dtype=int)
    py = np.asarray(py, dtype=int)
    out = np.zeros(np.broadcast_shapes(px.shape, py.shape, U0.shape), dtype=float)
    bx = np.broadcast_to(px, out.shape)
    by = np.broadcast_to(py, out.shape)
    mask = ((bx % 2) == 0) & ((by % 2) == 0)
    if not np.any(mask):
        return out

    ax = bx[mask] // 2
    ay = by[mask] // 2
    n = ax + ay
    angular_avg = np.exp(
        sp.gammaln(ax + 0.5)
        + sp.gammaln(ay + 0.5)
        - np.log(np.pi)
        - sp.gammaln(n + 1.0)
    )
    radial = np.zeros_like(angular_avg, dtype=float)
    u_stack = (
        np.broadcast_to(U0, out.shape)[mask],
        np.broadcast_to(U2, out.shape)[mask],
        np.broadcast_to(U4, out.shape)[mask],
    )
    for order in range(3):
        order_mask = n == order
        if np.any(order_mask):
            radial[order_mask] = u_stack[order][order_mask]
    out[mask] = angular_avg * radial
    return out


def _same_center_pair_axis_terms(power_bra, power_ket, coeff_bra, coeff_ket):
    terms = []
    power_bra = np.asarray(power_bra, dtype=int)
    power_ket = np.asarray(power_ket, dtype=int)
    for ib in range(3):
        cb = _small_comb(power_bra, ib)
        valid_b = ib <= power_bra
        for ik in range(3):
            valid = valid_b & (ik <= power_ket)
            if not np.any(valid):
                continue
            ck = _small_comb(power_ket, ik)
            coeff = cb * ck * (coeff_bra ** (power_bra - ib)) * (coeff_ket ** (power_ket - ik))
            coeff = np.where(valid, coeff, 0.0)
            terms.append((ib + ik, power_bra + power_ket - ib - ik, coeff))
    return terms


def _eri_same_center_sp_analytic_tensor(alphas, centers, labels, dz_eff):
    """Exact same-center ERI tensor for 2D s/px/py primitives."""
    alphas = np.asarray(alphas, float)
    n_ao = int(alphas.size)
    lx = np.array([_parse_2d_l(lbl)[0] for lbl in labels], dtype=int)
    ly = np.array([_parse_2d_l(lbl)[1] for lbl in labels], dtype=int)

    pair_exp = alphas[:, None] + alphas[None, :]
    p = pair_exp[:, :, None, None]
    q = pair_exp[None, None, :, :]
    total = p + q
    mu = (p * q) / total
    coeff_bra = q / total
    coeff_ket = -p / total
    U0, U2, U4 = _same_center_u_moments(mu, dz_eff)

    px_bra = (lx[:, None] + lx[None, :])[:, :, None, None]
    px_ket = (lx[:, None] + lx[None, :])[None, None, :, :]
    py_bra = (ly[:, None] + ly[None, :])[:, :, None, None]
    py_ket = (ly[:, None] + ly[None, :])[None, None, :, :]

    x_terms = _same_center_pair_axis_terms(px_bra, px_ket, coeff_bra, coeff_ket)
    y_terms = _same_center_pair_axis_terms(py_bra, py_ket, coeff_bra, coeff_ket)

    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    for vx, ux, cx in x_terms:
        for vy, uy, cy in y_terms:
            v_moment = _even_gaussian_moment_2d(vx, vy, total)
            if not np.any(v_moment):
                continue
            u_moment = _same_center_radial_moment_2d(ux, uy, U0, U2, U4)
            if not np.any(u_moment):
                continue
            eri_tensor += cx * cy * v_moment * u_moment
    return eri_tensor


def _zero_mean_pair_moment(power_a, power_b, c11, c22, c12):
    power_a = np.asarray(power_a, dtype=int)
    power_b = np.asarray(power_b, dtype=int)
    out = np.zeros(np.broadcast_shapes(power_a.shape, power_b.shape, c11.shape), dtype=float)
    pa = np.broadcast_to(power_a, out.shape)
    pb = np.broadcast_to(power_b, out.shape)

    out[(pa == 0) & (pb == 0)] = 1.0
    out[(pa == 2) & (pb == 0)] = c11[(pa == 2) & (pb == 0)]
    out[(pa == 0) & (pb == 2)] = c22[(pa == 0) & (pb == 2)]
    out[(pa == 1) & (pb == 1)] = c12[(pa == 1) & (pb == 1)]
    mask22 = (pa == 2) & (pb == 2)
    out[mask22] = (c11 * c22 + 2.0 * c12 * c12)[mask22]
    return out


def _eri_prony_same_center_sp_tensor(alphas, centers, labels, dz_eff):
    """Fast Prony ERI tensor for same-center 2D s/px/py primitives."""
    alphas = np.asarray(alphas, float)
    n_ao = int(alphas.size)
    lx = np.array([_parse_2d_l(lbl)[0] for lbl in labels], dtype=int)
    ly = np.array([_parse_2d_l(lbl)[1] for lbl in labels], dtype=int)

    A = alphas[:, None] + alphas[None, :]
    px_pair = lx[:, None] + lx[None, :]
    py_pair = ly[:, None] + ly[None, :]

    a_bd = A[:, :, None, None]
    b_bd = A[None, None, :, :]
    px_a = px_pair[:, :, None, None]
    px_b = px_pair[None, None, :, :]
    py_a = py_pair[:, :, None, None]
    py_b = py_pair[None, None, :, :]

    invz = 1.0 / float(dz_eff)
    gammas = XIS * (invz**2)
    weights = ETAS * invz
    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)

    for weight_p, gamma_p in zip(weights, gammas):
        D = a_bd * b_bd + gamma_p * (a_bd + b_bd)
        theta = (a_bd * b_bd * gamma_p) / D
        base = weight_p * (np.pi**2 / D)
        c11 = (1.0 / (2.0 * a_bd)) - (theta / (2.0 * a_bd * a_bd))
        c22 = (1.0 / (2.0 * b_bd)) - (theta / (2.0 * b_bd * b_bd))
        c12 = theta / (2.0 * a_bd * b_bd)
        mx = _zero_mean_pair_moment(px_a, px_b, c11, c22, c12)
        my = _zero_mean_pair_moment(py_a, py_b, c11, c22, c12)
        eri_tensor += base * mx * my
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
    dz_eff = abs(delta_z)
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    has_higher_l = any(lbl.kind != "2d-s" for lbl in labels)

    # 1. Exact Analytical s-s same center
    if _all_s_same_center_basis(centers, labels):
        return _eri_all_s_same_center_tensor(alphas, centers, labels, dz_eff)

    # 1b. Exact same-center s/p Cartesian moments, used by aligned Li chains.
    if _same_center_sp_basis(centers, labels):
        return _eri_same_center_sp_analytic_tensor(alphas, centers, labels, dz_eff)

    # 2. Strictly 2D regime (dz = 0)
    if dz_eff < 1e-9:
        return _eri_exact_bessel_tensor(alphas, centers, labels, dz_eff)
    
    # 3. Twilight zone (< 0.1)
    if dz_eff < OFFCENTER_NUMERICAL_CUTOFF:
        if has_higher_l:
            # Drop-in FD tensor loop for higher angular momentum
            n_ao = len(alphas)
            T = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
            for i in range(n_ao):
                for j in range(i, n_ao):
                    for k in range(n_ao):
                        for l in range(k, n_ao):
                            if (i, j) > (k, l): continue
                            q_alphas = [alphas[i], alphas[j], alphas[k], alphas[l]]
                            q_centers = [centers[i], centers[j], centers[k], centers[l]]
                            q_kinds = [labels[i].kind, labels[j].kind, labels[k].kind, labels[l].kind]
                            
                            val = _fd_eri_recursive(q_alphas, q_centers, q_kinds, dz_eff)
                            _assign_eri_symmetry(T, i, j, k, l, val)
            return T
        else:
            # Pure s-orbitals are safe for numerical/Hankel
            return _eri_numerical_tensor(alphas, centers, labels, dz_eff)

    # 4. Far-field -> Global Prony Fit
    return _eri_prony_tensor(alphas, centers, labels, dz_eff)


def eri_2d_cartesian_with_p_general_test(alphas, centers, labels, delta_z, dz_tol=None):
    """
    STRESS-TEST DISPATCHER: Forces all ERI calculations (even co-centered ones)
    through the general non-cocentered engines (FD/Hankel and Global Prony).
    
    This function explicitly removes the analytical `_eri_all_s_same_center_tensor` 
    shortcut to validate the numerical stability of the general integrals during SCF.
    """
    dz_eff = abs(delta_z)
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    # 1. Strictly 2D regime (dz = 0)
    # We MUST retain this block. Both the 1D Hankel transform and the Prony fit 
    # mathematically divide by dz, so they will physically crash at exactly zero.
    if dz_eff < 1e-9:
        return _eri_exact_bessel_tensor(alphas, centers, labels, dz_eff)
    
    # 2. Twilight Zone (0 < dz < Cutoff)
    # FORCED GENERAL ENGINE: We bypass the analytical shortcut.
    # _eri_exact_fd_reference automatically routes pure s-orbitals to the 1D Hankel
    # transform, and higher angular momentum to the Recursive FD derivatives.
    if dz_eff < OFFCENTER_NUMERICAL_CUTOFF:
        return _eri_exact_fd_reference(alphas, centers, labels, dz_eff)

    # 3. Far-Field (dz >= Cutoff)
    # FORCED GENERAL ENGINE: Evaluates everything via the 102-point minimax Prony fit.
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

def two_electron_wedge_basis(npoints, exchange="symmetric"):
    """Return raw ordered pairs ``(i, j)`` with ``i < j``.

    The returned phase is the exchange continuation used to fold the omitted
    sector back into the ``x1 < x2`` wedge: ``+1`` for singlet/Neumann and
    ``-1`` for triplet/Dirichlet.
    """
    npoints = int(npoints)
    if npoints < 2:
        raise ValueError("npoints must be at least 2 for an ordered two-electron wedge.")
    key = str(exchange).strip().lower()
    if key in {"symmetric", "sym", "singlet", "+", "+1", "even", "neumann"}:
        phase = 1
    elif key in {"antisymmetric", "antisym", "triplet", "-", "-1", "odd", "dirichlet"}:
        phase = -1
    else:
        raise ValueError("exchange must be symmetric/singlet or antisymmetric/triplet.")
    pairs = [(i, j) for i in range(npoints) for j in range(i + 1, npoints)]
    return np.asarray(pairs, dtype=int), phase


def two_electron_wedge_kinetic(K, exchange="symmetric", return_extension=False, return_transform=False):
    """Fold a full product-DVR KEO onto raw wedge values ``psi[i, j]``, ``i < j``.

    The basis is the unsymmetrized ordered grid ``|i j>`` with ``i < j``.  The
    omitted sector is supplied by exchange continuation,

    ``psi[j, i] = +/- psi[i, j]``.

    Thus the effective matrix is

    ``H_wedge[(ij),(kl)] = H_full[(ij),(kl)] +/- H_full[(ij),(lk)]``.

    This is the ordered-coordinate version of the Neumann/singlet or
    Dirichlet/triplet boundary at ``x1 = x2``; it is not the normalized
    exchange-adapted basis ``(|ij> +/- |ji>)/sqrt(2)``.
    """
    K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square one-particle kinetic matrix.")
    npoints = K.shape[0]
    pairs, phase = two_electron_wedge_basis(npoints, exchange)
    eye = np.eye(npoints, dtype=np.result_type(K, float))
    full = np.kron(K, eye) + np.kron(eye, K)
    wedge_index = pairs[:, 0] * npoints + pairs[:, 1]
    mirror_index = pairs[:, 1] * npoints + pairs[:, 0]
    wedge = full[np.ix_(wedge_index, wedge_index)] + phase * full[np.ix_(wedge_index, mirror_index)]
    wedge = 0.5 * (wedge + wedge.T.conj())
    if return_transform:
        return_extension = True
    if return_extension:
        extension = np.zeros((npoints * npoints, pairs.shape[0]), dtype=np.result_type(K, float))
        cols = np.arange(pairs.shape[0])
        extension[wedge_index, cols] = 1.0
        extension[mirror_index, cols] = phase
        return wedge, pairs, extension
    return wedge, pairs


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
    "local_ecp_potential_at_z",
    "semilocal_ecp_projector_blocks",
    "eri_2d_cartesian_with_p",
    "build_h1_nm",
    "two_electron_wedge_basis",
    "two_electron_wedge_kinetic",
    "STO6_EXPS_H",
    "STO6_EXPS_He",
    "Exp_631g_ss_H",
    "ETAS",
    "XIS",
]
