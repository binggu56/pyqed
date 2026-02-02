import numpy as np
import scipy.linalg as la
from scipy.special import i0e as I0e
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import os

from pyqed.qchem.gdvr.newton_helper import CollocatedERIOp, NewtonHelper

from pyqed.qchem.gdvr.gdvr_integrals import (
    make_xy_spd_primitive_basis,
    V_en_sp_total_at_z,
    build_h1_nm,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
)


#  Basic molecule holder
class Molecule:
    def __init__(self, charges, coords, nelec=None):
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        assert self.charges.shape[0] == self.coords.shape[0]

        if nelec is None:
            self.nelec = int(round(float(np.sum(self.charges))))
        else:
            self.nelec = int(nelec)

    def to_tuples(self):
        return [(float(Z), float(x), float(y), float(z))
                for (Z, (x, y, z)) in zip(self.charges, self.coords)]

    def nuclear_repulsion_energy(self):
        E = 0.0
        Z = self.charges
        R = self.coords
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                dR = R[i] - R[j]
                E += Z[i] * Z[j] / float(np.linalg.norm(dR))
        return E



#  Sine/Exponential/Sinc DVR along z
def sine_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    j = np.arange(1, N + 1)
    z = zmin + j * L / (N + 1)
    n = np.arange(1, N + 1)
    U = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * np.outer(j, n) / (N + 1))
    lam = 0.5 * (np.pi * n / L) ** 2
    Tz = (U * lam) @ U.T
    dz = L / (N + 1)
    return z, Tz, dz

def Exponential_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    dz = L / N
    z = zmin + np.arange(N) * dz
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    arg = np.pi * n_diff / N
    Tz = np.zeros((N, N))
    mask = n_diff != 0
    sign = (-1.0)**n_diff[mask]
    sin_sq = np.sin(arg[mask])**2
    if N % 2 == 0:
        Tz[mask] = 2.0 * sign / sin_sq
        diag_val = (N**2 + 2.0) / 3.0
    else:
        cos_val = np.cos(arg[mask])
        Tz[mask] = 2.0 * sign * cos_val / sin_sq
        diag_val = (N**2 - 1.0) / 3.0
    np.fill_diagonal(Tz, diag_val)
    prefactor = 0.5 * (np.pi / L)**2
    Tz *= prefactor
    return z, Tz, dz


def sinc_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    dz = L / (N + 1)
    j = np.arange(1, N + 1)
    z = zmin + j * dz
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    Tz = np.zeros((N, N))
    diag_val = np.pi**2 / (6.0 * dz**2)
    np.fill_diagonal(Tz, diag_val)
    mask = n_diff != 0
    sign = (-1.0)**n_diff[mask]
    denom = (n_diff[mask] * dz)**2
    Tz[mask] = sign / denom
    return z, Tz, dz


#  S-metric (overlap between primitive gaussians) orthonormalization helper
def _s_orthonormalizer(S, eps_rel=1e-12):
    S = 0.5 * (S + S.T)
    w, U = la.eigh(S)
    if w.size == 0:
        raise ValueError("Empty overlap matrix.")
    wmax = float(np.max(w))
    if wmax <= 0.0:
        raise ValueError("S_prim is non-positive (all eigenvalues <= 0).")
    keep = w > (eps_rel * wmax)
    r = int(np.count_nonzero(keep))
    if r == 0:
        raise ValueError("S_prim has zero numerical rank under the given threshold.")
    X = U[:, keep] / np.sqrt(w[keep])
    return X, r, w


def slice_eigens_xy(z_grid, S_prim, T_prim, V_en_of_z, M=1):
    Nz = len(z_grid)
    M  = int(M)
    X, r, wS = _s_orthonormalizer(S_prim, eps_rel=1e-12)
    if M > r:
        warnings.warn(f"M={M} exceeds rank(S)={r}; reducing M to {r}.")
        M = r
    E_slices = np.zeros((Nz, M), float)
    C_list   = [np.zeros((S_prim.shape[0], M), float) for _ in range(Nz)]

    for k, zk in enumerate(z_grid):
        Vz = V_en_of_z(float(zk))
        Hk = T_prim + Vz
        Hk_ = X.T @ Hk @ X
        if M == 1:
            w, V_ = la.eigh(Hk_, subset_by_index=[0, 0])
            Vk = V_[:, :1]
        else:
            w, V_ = la.eigh(Hk_, subset_by_index=[0, M-1])
            Vk = V_[:, :M]
        Uk = X @ Vk
        E_slices[k, :len(w[:M])] = w[:M]
        C_list[k] = Uk

    return E_slices, C_list



#  Helpers for J/K and PSD
def _psd_project_small(M):
    M = 0.5 * (M + M.T)
    w, V = la.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


def precompute_eri_method2_JK_psd(
    alphas, centers, labels, z_grid, C_list, M,
    max_offset=None, auto_cut=False, cut_eps=1e-8, verbose=True,
):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    z_grid = np.asarray(z_grid, float)
    Nz = z_grid.size
    
    if Nz == 0: raise ValueError("precompute_eri_method2_JK_psd: empty z_grid")
    if Nz > 1: dz = float(abs(z_grid[1] - z_grid[0]))
    else: dz = 0.0

    if max_offset is None or max_offset > (Nz - 1):
        max_offset = Nz - 1

    ERI_J = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]

    eri_by_h = {}
    norm0 = None
    h_max = max_offset
    
    for h in range(0, max_offset + 1):
        delta_z = abs(h * dz)
        eri_ao = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z)
        eri_by_h[h] = eri_ao

        if auto_cut:
            nh = float(np.linalg.norm(eri_ao.reshape(-1)))
            if h == 0: norm0 = max(nh, 1e-16)
            else:
                if norm0 is None: norm0 = max(nh, 1e-16)
                ratio = nh / norm0
                if ratio < cut_eps:
                    h_max = h
                    break

    for m in range(Nz):
        C_m = np.asarray(C_list[m], float)
        for n in range(Nz):
            h = abs(n - m)
            if h > h_max: continue

            C_n = np.asarray(C_list[n], float)
            eri_ao = eri_by_h[h]

            Jabcd = np.einsum("pqrs,pa,qb,rc,sd->abcd", eri_ao, C_m, C_m, C_n, C_n, optimize=True)
            EmnJ = Jabcd.reshape(M * M, M * M)
            EmnJ = _psd_project_small(EmnJ)

            Kabcd = np.einsum("pqrs,pa,rc,qb,sd->abcd", eri_ao, C_m, C_n, C_m, C_n, optimize=True)
            EmnK = Kabcd.reshape(M * M, M * M)
            EmnK = 0.5 * (EmnK + EmnK.T)

            ERI_J[m][n] = EmnJ
            ERI_K[m][n] = EmnK
            
    return ERI_J, ERI_K


def fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=1.0):
    N = Nz * M
    P4 = P.reshape(Nz, M, Nz, M)
    Ddiag = [P4[b, :, b, :].copy() for b in range(Nz)]
    F2e = np.zeros((N, N), dtype=float)

    for m in range(Nz):
        J_mm = np.zeros((M, M), float)
        for n in range(Nz):
            EmnJ = ERI_J[m][n]
            if np.ndim(EmnJ)==0: EmnJ = np.atleast_2d(EmnJ)
            jvec = EmnJ @ Ddiag[n].reshape(M * M)
            J_mm += jvec.reshape(M, M)
        im0 = m * M
        im1 = im0 + M
        F2e[im0:im1, im0:im1] += J_mm

    for m in range(Nz):
        im0 = m * M
        im1 = im0 + M
        for n in range(Nz):
            EmnK = ERI_K[m][n]
            if np.ndim(EmnK)==0: EmnK = np.atleast_2d(EmnK)
            BK   = P4[m, :, n, :]
            kvec = EmnK @ BK.reshape(M * M)
            K_mn = kvec.reshape(M, M)
            in0 = n * M
            in1 = in0 + M
            F2e[im0:im1, in0:in1] -= 0.5 * k_scale * K_mn

    return F2e



#  Build Method II (core + ERIs) and SCF TODO: remove the naming of Method II, i used to think and tried Method I as globally share one set of optimized AO but that is a lot less efficient, already removed
def build_method2(
    mol: Molecule,
    Lz=18.0, Nz=121, M=1,
    s_exps=None, p_exps=None, d_exps=None,
    max_offset=None, auto_cut=False, cut_eps=1e-6, verbose=True, dvr_method='sine'
):
    t0 = time.time()
    if dvr_method == 'sine':
        z, Kz, dz = sine_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'exp':
        z, Kz, dz = Exponential_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'sinc':
        z, Kz, dz = sinc_dvr_1d(-Lz, Lz, Nz)
    else:
        raise NotImplementedError('use sine or exp for not')
    nuclei = mol.to_tuples()

    if verbose:
        print(f"\n[DEBUG] Grid Info for Nz={Nz}, Lz={Lz}:")
        print(f"  > Grid Spacing (dz): {dz:.6f}")

    if s_exps is None: raise ValueError("Please provide s_exps.")
    if p_exps is None: p_exps = np.array([], float)
    if d_exps is None: d_exps = np.array([], float)

    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei,
        exps_s=np.asarray(s_exps, float),
        exps_p=np.asarray(p_exps, float),
        exps_d=np.asarray(d_exps, float),
    )
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)

    def V_en_of_z(zk: float) -> np.ndarray:
        return V_en_sp_total_at_z(alphas, centers, labels, nuclei, zk)

    E_slices, C_list = slice_eigens_xy(
        z_grid=z, S_prim=S_prim, T_prim=T_prim, V_en_of_z=V_en_of_z, M=M,
    )

    Nz = len(z)
    S_slice = np.zeros((Nz, Nz, M, M), float)
    for k in range(Nz):
        Ck = C_list[k]
        for m in range(Nz):
            Cm = C_list[m]
            S_slice[k, m] = Ck.T @ (S_prim @ Cm)

    size = Nz * M
    Hcore = np.einsum('km,kmab->kamb', Kz, S_slice, optimize=True).reshape(size, size)
    Hcore += np.diag(E_slices.reshape(-1))

    if verbose:
        print(f"[Method2] Built Hcore {Hcore.shape} in {time.time()-t0:.2f}s")

    ERI_J, ERI_K = precompute_eri_method2_JK_psd(
        alphas, centers, labels, z, C_list,
        M=M, max_offset=max_offset,
        auto_cut=auto_cut, cut_eps=cut_eps, verbose=verbose
    )

    shapes = {"Nz": Nz, "M": M, "n_ao2d": len(alphas), "size": size, "dz": dz}
    return Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes


class PulayCDIIS:
    def __init__(self, space=6, reg=1e-12):
        self.space = int(space)
        self.reg   = float(reg)
        self.F_hist = []
        self.E_hist = []

    def push(self, F, P):
        E = F @ P - P @ F
        self.F_hist.append(F.copy())
        self.E_hist.append(E.ravel().copy())
        if len(self.F_hist) > self.space:
            self.F_hist = self.F_hist[-self.space:]
            self.E_hist = self.E_hist[-self.space:]

    def mix(self):
        m = len(self.F_hist)
        if m < 2: return self.F_hist[-1]
        R = np.vstack(self.E_hist)
        G = R @ R.T
        B = np.empty((m + 1, m + 1), dtype=float)
        B[:m, :m] = G
        B[:m,  m] = -1.0
        B[ m, :m] = -1.0
        B[ m,  m] =  0.0
        rhs = np.zeros(m + 1, dtype=float)
        rhs[m] = -1.0
        B[:m, :m] += self.reg * np.eye(m)
        try:
            coeff = la.solve(B, rhs)[:m]
        except la.LinAlgError:
            return self.F_hist[-1]
        Fmix = np.zeros_like(self.F_hist[0])
        for c, Fi in zip(coeff, self.F_hist):
            Fmix += c * Fi
        return Fmix


def scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz, M, nelec, Enuc=0.0,
                    conv=1e-7, max_iter=50, verbose=True,
                    damp=0.20, diis_start=3, diis_space=8,
                    level_shift=0.5, shift_decay=0.75, k_ramp_iters=8):
    N = Nz * M
    nocc = nelec // 2
    I = np.eye(N)

    eps, Cmo = la.eigh(Hcore)
    Cocc = Cmo[:, :nocc]
    P = 2.0 * (Cocc @ Cocc.T)

    E_last = np.inf
    P_prev = P.copy()
    P_prev2 = None
    diis = PulayCDIIS(space=diis_space)
    beta = float(level_shift)

    def total_energy(Puse, Fuse):
        E_el = 0.5 * np.sum((Hcore + Fuse) * Puse)
        return E_el + Enuc

    for it in range(1, max_iter + 1):
        t0 = time.time()
        k_scale = 1.0 if k_ramp_iters <= 1 else min(1.0, it / float(k_ramp_iters))

        F2e = fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=k_scale)
        F = Hcore + F2e

        diis.push(F, P)
        if it >= diis_start:
            F = diis.mix()
        if beta > 0.0:
            Q = I - 0.5 * P
            F = F + beta * Q

        eps, Cmo = la.eigh(F)
        Cocc = Cmo[:, :nocc]
        P_new = 2.0 * (Cocc @ Cocc.T)

        if damp > 0.0:
            P_new = (1.0 - damp) * P_new + damp * P

        if P_prev2 is not None:
            d2 = la.norm(P_new - P_prev2, ord='fro')
            d1 = la.norm(P_new - P_prev,  ord='fro')
            if d2 < 1e-6 and d1 > 5e-5:
                P_new = 0.5 * (P_new + P_prev)

        Etot = total_energy(P_new, F)
        dE = abs(Etot - E_last)
        R = F @ P_new - P_new @ F
        rnorm = la.norm(R, ord='fro')

        if verbose:
            print(f"SCF {it:3d}: E = {Etot: .10f}  dE={dE:.2e}  ||[F,P]||={rnorm:.2e}  "
                  f"k={k_scale:.2f}  β={beta:.2f}  damp={damp:.2f}  dt={time.time()-t0:.2f}s")

        if dE < conv and rnorm < 1e-5:
            P = P_new
            E_last = Etot
            break

        if it > 1 and rnorm > 1.5 * la.norm(F @ P - P @ F, ord='fro'):
            damp = min(0.5, 1.25 * damp + 0.02)
            beta = min(2.0, beta * 1.25)
        else:
            beta = max(0.0, beta * shift_decay)

        P_prev2 = P_prev
        P_prev  = P
        P       = P_new
        E_last  = Etot

    info = {"iter": it, "dE": dE, "rnorm": rnorm, "damp": damp, "level_shift": beta, "k_scale": k_scale}
    return Etot, eps, Cmo, P, info


def s_norm(v, S):
    return float(np.sqrt(max(1e-32, v.T @ (S @ v))))

def bound_step_S(delta_dict, active, S, max_norm):
    out = {}
    scale = 1.0
    for n in active:
        nrm = s_norm(delta_dict[n], S)
        if nrm > max_norm:
            scale = min(scale, max_norm / nrm)
    if scale < 1.0:
        for n in active:
            out[n] = delta_dict[n] * scale
        return out, scale
    else:
        return delta_dict, 1.0


def g_dot_delta(g_full, delta_dict, active):
    val = 0.0
    for n in active:
        val += float(np.dot(g_full[n].ravel(), delta_dict[n].ravel()))
    return val

def eri_JK_from_kernels_M1(C_list, K_h, Kx_h):
    Nz = len(C_list)
    d = [C_list[m][:, 0] for m in range(Nz)]
    v_mm = [np.kron(dm, dm) for dm in d]
    ERI_J = [[0.0 for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[0.0 for _ in range(Nz)] for _ in range(Nz)]

    for h in range(Nz):
        Nh = Nz - h
        if Nh <= 0: break
        V_right = np.column_stack([v_mm[nn] for nn in range(h, Nz)])
        WJ = K_h[h] @ V_right
        for m in range(Nh):
            nn = m + h
            ERI_J[m][nn] = float(v_mm[m].T @ WJ[:, m])
            ERI_J[nn][m] = ERI_J[m][nn]
            v_mn = np.kron(d[m], d[nn])
            w_mn = Kx_h[h] @ v_mn
            ERI_K[m][nn] = float(v_mn.T @ w_mn)
            ERI_K[nn][m] = ERI_K[m][nn]

    for m in range(Nz):
        for n in range(Nz):
            if ERI_J[m][n] < 0.0: ERI_J[m][n] = 0.0
    return ERI_J, ERI_K


def rebuild_Hcore_from_d(d_stack, z, Kz, S_prim, T_prim, alphas, centers, labels, nuclei):
    Nz, N = d_stack.shape
    S_scalar = np.zeros((Nz, Nz), float)
    for n in range(Nz):
        dn = d_stack[n]
        S_scalar[n, n] = float(dn.T @ (S_prim @ dn))
        for m in range(n + 1, Nz):
            dm = d_stack[m]
            val = float(dn.T @ (S_prim @ dm))
            S_scalar[n, m] = val
            S_scalar[m, n] = val

    e_local = np.zeros(Nz, float)
    for n in range(Nz):
        Vz = V_en_sp_total_at_z(alphas, centers, labels, nuclei, float(z[n]))
        e_local[n] = float(d_stack[n].T @ ((T_prim + Vz) @ d_stack[n]))

    Hcore = (Kz * S_scalar).astype(float)
    Hcore += np.diag(e_local)
    return Hcore


def save_scf_snapshot(run_folder, run_label, Nz, M, cycle, Etot, C_list, Cmo, P, eps, info,
                      alphas, centers, labels, z_grid, Lz):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"scf_res_{run_label}_Nz{Nz}_M{M}_cyc{cycle}_{now_str}.npz"
    full_path = os.path.join(run_folder, fname)
    try: C_list_arr = np.stack(C_list, axis=0)
    except ValueError: C_list_arr = np.array(C_list, dtype=object)

    labels_serialized = [{'kind': l.kind, 'dim': l.dim, 'l': l.l, 'role': l.role} for l in labels]
    np.savez_compressed(
        full_path, Etot=Etot, C_list=C_list_arr, Cmo=Cmo, P=P, eps=eps, info=info,
        cycle=cycle, timestamp=now_str, alphas=alphas, centers=centers,
        labels_serialized=labels_serialized, z_grid=z_grid, Lz=Lz, Nz=Nz, M=M
    )



#  SWEEP NEWTON HELPER (Gauss-Seidel, for post HF reoptimization at AO level)
class SweepNewtonHelper(NewtonHelper):
    """
    Extends NewtonHelper with O(Nz) exact single-slice gradient construction 
    AND O(Nz) exact diagonal Hessian construction.
    """
    def __init__(self, h1_nm, S_prim, eri_op):
        super().__init__(h1_nm, S_prim, eri_op)
        self.eri_op = eri_op # Store explicitly
        
    def get_gradient_slice_onthefly(self, n, d_stack, P_slice):
        """
        Computes gradient g_n using O(Nz) operations.
        """
        Nz = self.Nz
        N  = self.N
        
        g_n = np.zeros(N, dtype=float)
        dn  = d_stack[n]
        
        # 1. Pre-compute Diagonal Coulomb Term for F_nn (Depends on all k)
        # J_nn = Sum_k P_kk * (nn|kk)
        J_nn = np.zeros((N, N), dtype=float)
        for k in range(Nz):
            p_kk = P_slice[k, k]
            if abs(p_kk) > 1e-12:
                dk = d_stack[k]
                val = self.eriop.block_nm__kl(n, n, k, k, dk, dk)
                J_nn += p_kk * val
                
        # 2. Loop over m to sum F_nm @ d_m
        for m in range(Nz):
            dm = d_stack[m]
            p_nm = P_slice[n, m]
            p_mn = P_slice[m, n]
            
            # One-Electron Part
            F_nm = p_nm * self.h1_nm[n, m] + p_mn * self.h1_nm[m, n]
            
            # Two-Electron Part
            p_sym = 0.5 * (p_nm + p_mn)
            
            if abs(p_sym) > 1e-12:
                V_eff = np.zeros((N, N), dtype=float)
                
                if n == m:
                    # Diagonal Exchange: (nn|nn)
                    K_nn = self.eriop.block_nl__km(n, n, n, n, dn, dn)
                    V_eff = J_nn - 0.5 * P_slice[n, n] * K_nn
                else:
                    # Off-diagonal Exchange: (nn|mm)
                    K_val = self.eriop.block_nl__km(n, m, m, n, dm, dn)
                    V_eff = -0.5 * P_slice[m, n] * K_val
                
                F_nm += 2.0 * p_sym * V_eff
            
            g_n += F_nm @ dm
            
        return g_n

    def get_diagonal_hessian_block_sparse(self, n, d_stack, P_slice):
        """
        Computes the diagonal Hessian block H_nn in O(Nz) time 
        by exploiting DVR sparsity rules (k=l constraints).
        """
        Nz = self.Nz
        N = self.N
        dn = d_stack[n]
        
        # 1. One-electron part: P_nn * h_nn
        H_nn = P_slice[n, n] * self.h1_nm[n, n].copy()
        
        # 2. Two-electron parts (Terms 39, 40, 41 from PDF)
        
        # --- Term 39: P_nn * Sum_kl P_kl [ (nn|kl) - 0.5 (nl|kn) ] --- (for formula check the pdf file. TODO: make the notes for formulas better)
        # Sparsity: (nn|kl) -> k=l. (nl|kn) -> l=n, k=n.
        if abs(P_slice[n, n]) > 1e-12:
            acc39 = np.zeros((N, N), dtype=float)
            
            # Coulomb part: Sum_k P_kk (nn|kk)
            for k in range(Nz):
                p_kk = P_slice[k, k]
                if abs(p_kk) > 1e-12:
                    dk = d_stack[k]
                    acc39 += p_kk * self.eriop.block_nm__kl(n, n, k, k, dk, dk)
            
            # Exchange part: -0.5 * P_nn * (nn|nn)
            kn_nn = self.eriop.block_nl__km(n, n, n, n, dn, dn)
            acc39 -= 0.5 * P_slice[n, n] * kn_nn
            
            H_nn += P_slice[n, n] * acc39

        # --- Term 40: Re Sum_kl P_nk P_nl [ (nk|nl) - 0.5 (nl|mk) ] ---
        # Sparsity: (nk|nl) -> k=l. (nl|nk) -> l=n, k=n.
        acc40 = np.zeros((N, N), dtype=float)
        # Sum_k P_nk^2 (nk|nk)
        for k in range(Nz):
            p_nk = P_slice[n, k]
            if abs(p_nk) > 1e-12:
                dk = d_stack[k]
                acc40 += (p_nk * p_nk) * self.eriop.block_nk__ml(n, n, k, k, dk, dk)
        # -0.5 * P_nn^2 * (nn|nn)
        nl_nk_nn = self.eriop.block_nl__mk(n, n, n, n, dn, dn)
        acc40 -= 0.5 * (P_slice[n, n]**2) * nl_nk_nn
        
        H_nn += acc40 

        # --- Term 41: Sum_kl P_nl P_kn [ (nl|km) - 0.5 (nm|kl) ] ---
        # Sparsity: (nl|kn) -> l=n, k=n. (nn|kl) -> k=l.
        acc41 = np.zeros((N, N), dtype=float)
        
        # P_nn^2 * (nn|nn)
        nl_kn_nn = self.eriop.block_nl__km(n, n, n, n, dn, dn)
        acc41 += (P_slice[n, n]**2) * nl_kn_nn
        
        # -0.5 * Sum_k P_nk^2 (nn|kk)
        for k in range(Nz):
            p_nk = P_slice[n, k]
            if abs(p_nk) > 1e-12:
                dk = d_stack[k]
                acc41 -= 0.5 * (p_nk * p_nk) * self.eriop.block_nm__kl(n, n, k, k, dk, dk)
                
        H_nn += acc41
        
        return H_nn

    def kkt_step_slice(self, n, d_stack, P_slice, S_prim, ridge_base=1e-4):
        """
        Solves the local KKT problem with automatic ridge selection for stability.
        """
        # 1. Compute Exact Gradient (O(Nz))
        # This ensures the direction is consistent with the latest orbital stack
        g_n_vec = self.get_gradient_slice_onthefly(n, d_stack, P_slice)
        g_n = g_n_vec.reshape(-1, 1)
        
        # 2. Compute Exact Diagonal Hessian (O(Nz))
        H_nn = self.get_diagonal_hessian_block_sparse(n, d_stack, P_slice)
        
        # 3. Automated Ridge Selection
        # Calculate eigenvalues to check for negative curvature or ill-conditioning
        try:
            w = np.linalg.eigvalsh(H_nn)
            min_eig = w[0]
            # The ridge shift makes the minimum eigenvalue at least ridge_base
            auto_ridge = max(0.0, ridge_base - min_eig)
        except np.linalg.LinAlgError:
            # Fallback for extreme numerical cases
            auto_ridge = ridge_base
            
        H_nn_shifted = H_nn + auto_ridge * np.eye(H_nn.shape[0])
            
        # 4. Build the KKT System
        # The KKT system ensures the update delta_d remains orthogonal to the constraint
        dn = d_stack[n].reshape(-1, 1)
        s_vec = S_prim @ dn
        
        N = H_nn_shifted.shape[0]
        KKT = np.zeros((N + 1, N + 1), dtype=float)
        KKT[:N, :N] = H_nn_shifted
        KKT[:N, N]  = s_vec.flatten()
        KKT[N, :N]  = s_vec.flatten()
        
        # Right-hand side is the negative gradient for descent
        rhs = np.zeros((N + 1, 1), dtype=float)
        rhs[:N] = -g_n
        
        # 5. Solve the Linear System
        try:
            sol = np.linalg.solve(KKT, rhs)
            delta_d = sol[:N].flatten()
            lam = sol[N]
        except np.linalg.LinAlgError:
            # Emergency descent: steep-descent fallback if KKT is singular
            print(f"  [Warning] KKT singular at site {n}, using scaled gradient.")
            delta_d = -0.01 * g_n.flatten()
            lam = 0.0
            
        return delta_d, lam, g_n_vec
    
def sweep_optimize_driver(
    nh: SweepNewtonHelper,
    d_stack,
    P_slice,
    S_prim,
    n_cycles=1,
    ridge=1,          # large default ridge for stability
    trust_step=1.0,     # Full Newton step
    trust_radius=2.0,   # decides how large a step could be, decides convergence speed vs. monotonic decrement stability
    verbose=True
):
    """
    Performs Symmetric Gauss-Seidel optimization (Forward + Backward).
    """
    Nz = d_stack.shape[0]
    d_current = d_stack.copy()
    
    # Create Symmetric Order: 0->N, then N-2->0
    forward = list(range(Nz))
    backward = list(range(Nz-2, -1, -1))
    symmetric_order = forward + backward
    
    for cyc in range(n_cycles):
        max_delta = 0.0
        max_grad  = 0.0
        
        # Run Symmetric Sweep
        for n in symmetric_order:
            # 1. Solve Local KKT
            delta_d, lam, g_n = nh.kkt_step_slice(n, d_current, P_slice, S_prim, ridge)
            
            gnorm = np.linalg.norm(g_n)
            max_grad = max(max_grad, gnorm)
            
            # 2. Descent Check
            if np.dot(g_n.flatten(), delta_d) > 0: 
                delta_d *= -1.0
            
            # 3. Trust Region (Relaxed)
            snorm = np.sqrt(delta_d @ S_prim @ delta_d)
            if snorm > trust_radius: 
                delta_d *= (trust_radius / snorm)
            
            # 4. Update
            d_cand = d_current[n] + trust_step * delta_d
            
            # Normalize
            norm = np.sqrt(d_cand @ S_prim @ d_cand)
            d_cand /= norm
            
            # Track Stats
            diff = d_cand - d_current[n]
            change = np.sqrt(diff @ S_prim @ diff)
            max_delta = max(max_delta, change)
            
            d_current[n] = d_cand
            
        if verbose:
            print(f"  [Sweep {cyc+1}] Max |g|: {max_grad:.4e}, Max d-change: {max_delta:.6f}")
            
    return d_current

if __name__ == "__main__":
    stime = time.time()

    # charges = np.array([1.0, 1.0, 1.0, 1.0], float)
    charges = np.array([1.0, 1.0], float)
    # charges = np.array([1.0, 2.0], float)
    coords  = np.array([
        [0.0, 0.0,  0.7 ],
        [0.0, 0.0,  -0.7],
        # [0.0, 0.0,  3.6 ],
        # [0.0, 0.0,  0.91],
        # [0.0, 0.0, -3.6 ],
        # [0.0, 0.0, -0.91],
    #     # [0.0, 0.7,  0.7 ],
    #     # [0.0, -0.7,  0.7],
    #     # [0.0, 0.7, -0.7 ],
    #     # [0.0, -0.7, -0.7],
    ], float)
    # coords = np.linspace(-49, 49, 20, dtype=float)
    # coords = np.linspace(-19, 19, 20, dtype=float)
    # charges = np.ones_like(coords, dtype=float)
    # coords = np.stack([np.zeros_like(coords), np.zeros_like(coords), coords], axis=1)
    mol = Molecule(charges, coords, nelec=2)
    NELEC = mol.nelec
    Enuc  = mol.nuclear_repulsion_energy()
    print("=== Newton Sweep test (s+p+d 2D basis) ===")
    print(f"nelec = {NELEC}")
    print(f"Enuc  = {Enuc:.10f} Eh")

    # STO-6G like exponents (H)
    s_exps = np.array([18.73113696, 2.825394365, 0.6401216923, 0.1612777588], float)
    p_exps = np.array([], float)  
    d_exps = np.array([], float) 

    # ------------------------
    #  DVR / slice parameters
    # ------------------------
    Nz = 63
    M  = 1
    LZ = 8
    # Nz = 511
    # M  = 1
    # LZ = 12.5

    # -------------------------
    #  Newton / SCF parameters
    # -------------------------
    ALT_CYCLES              = 20
    SWEEP_ITERATIONS = 10    
    TRUST_STEP       = 1.0    # Full Newton step
    NEWTON_RIDGE     = 0.5   # Small ridge for stability
    TRUST_RADIUS     = 2    # Allow larger steps!
    SCF_MONO_TOL            = 1e-8
    VERBOSE                 = True
    DVR_METHOD              = 'sine'

    batch_folder = f"results_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    # 1. Initial Build
    print(f"\n==== Sweep Strategy: Nz={Nz}, M={M}, Lz={LZ} ====")
    Hcore, z, dz, E_slices, C_list, _ERI_J0, _ERI_K0, shapes = build_method2(
        mol, Lz=LZ, Nz=Nz, M=M,
        s_exps=s_exps, p_exps=p_exps, d_exps=d_exps, 
        max_offset=None, auto_cut=False, verbose=VERBOSE, dvr_method=DVR_METHOD,
    )

    nuclei = mol.to_tuples()
    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei, exps_s=s_exps, exps_p=p_exps, exps_d=d_exps,
    )
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    # print(T_prim)
    # import sys
    # sys.exit()
    n_ao = len(alphas)
    
    # DVR grid for helper
    z_chk, Kz, dz_chk = sine_dvr_1d(-LZ, LZ, Nz)

    # Precompute ERI kernels
    print("[Main] Precomputing ERI kernels...")
    K_h = []
    Kx_h = []
    for h in range(Nz):
        dz_val = h * dz
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
        K_mat = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
        K_h.append(K_mat)
        eri_perm = eri_tensor.transpose(0, 2, 1, 3)
        Kx_mat = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
        Kx_h.append(Kx_mat)

    # Initial SCF
    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
    Etot, eps, Cmo, P, info = scf_rhf_method2(
        Hcore, ERI_J, ERI_K, Nz, M,
        nelec=NELEC, Enuc=Enuc,
        conv=1e-6, max_iter=100, verbose=VERBOSE,
    )
    print(f"[SCF 0] E = {Etot:.12f} Eh  (iters={info['iter']})")
    E_history = [Etot]

    # Setup Sweep Helper
    h1_nm = build_h1_nm(
        Kz, S_prim, T_prim, z,
        lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz),
    )
    ERIop = CollocatedERIOp.from_kernels(N=S_prim.shape[0], Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    nh_sweep = SweepNewtonHelper(h1_nm, S_prim, ERIop) # Use subclass
    
    # Prepare stack
    d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz)])
    # Ensure normalization
    for n in range(Nz):
        dn = d_stack[n]
        d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

    # Main Cycle Loop
    for cyc in range(1, ALT_CYCLES + 1):
        print(f"\n--- Alternation Cycle {cyc} ---")
        
        # 1. Extract Density Slice
        P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
        
        # 2. Perform Sweep Optimization
        # This replaces the global active set selection and big KKT solve
        t_sweep = time.time()
        d_stack = sweep_optimize_driver(
            nh_sweep, d_stack, P_slice, S_prim,
            n_cycles=SWEEP_ITERATIONS,
            ridge=NEWTON_RIDGE,
            trust_step=TRUST_STEP,
            trust_radius=TRUST_RADIUS,
            verbose=VERBOSE
        )
        print(f"   Sweep finished in {time.time() - t_sweep:.4f}s")
        
        # 3. Update Basis & Hamiltonian
        C_list = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
        
        # 4. Rebuild Integrals (O(Nz^2))
        Hcore = rebuild_Hcore_from_d(
            d_stack, z, Kz, S_prim, T_prim,
            alphas, centers, labels, nuclei,
        )
        ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
        
        # 5. SCF Re-optimization
        Etot, eps, Cmo, P, info = scf_rhf_method2(
            Hcore, ERI_J, ERI_K, Nz, M,
            nelec=NELEC, Enuc=Enuc,
            conv=1e-7, max_iter=100, verbose=False, # less verbose inside loop
        )
        
        E_history.append(Etot)
        print(f"   [SCF] E = {Etot:.12f} Eh  ΔE={Etot - E_history[-2]:+.3e}")

        # Save snapshot
        save_scf_snapshot(
            run_folder=batch_folder, run_label="sweep",
            Nz=Nz, M=M, cycle=cyc,
            Etot=Etot, C_list=C_list, Cmo=Cmo, P=P, eps=eps, info=info,
            alphas=alphas, centers=centers, labels=labels, z_grid=z, Lz=LZ
        )
        
        if abs(Etot - E_history[-2]) < 1e-7:
            print("Converged.")
            break

    print(f"\nFinal Energy: {E_history[-1]:.12f} Eh")
    print(f"Total time: {time.time() - stime:.2f}s")
