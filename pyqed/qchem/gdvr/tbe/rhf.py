import numpy as np
import scipy.linalg as la
from scipy.special import i0e as I0e
import time
import warnings
import datetime
import os

from .newton import CollocatedERIOp, NewtonHelper
from pyqed.qchem.atomic_data import slater_radii

from .integrals import (
    STO6_EXPS_H,
    STO6_EXPS_He,
    make_xy_spd_primitive_basis,
    V_en_sp_total_at_z,
    build_h1_nm,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
)


DEFAULT_H_S_EXPS = np.asarray(STO6_EXPS_H, float).copy()
DEFAULT_HE_S_EXPS = np.asarray(STO6_EXPS_He, float).copy()
DEFAULT_P_EXPS = np.array([], float)
DEFAULT_D_EXPS = np.array([], float)


_ELEMENT_TO_Z = {
    "H": 1.0,
    "He": 2.0,
    "Li": 3.0,
    "Be": 4.0,
    "B": 5.0,
    "C": 6.0,
    "N": 7.0,
    "O": 8.0,
    "F": 9.0,
    "Ne": 10.0,
    "Na": 11.0,
    "Mg": 12.0,
    "Al": 13.0,
    "Si": 14.0,
    "P": 15.0,
    "S": 16.0,
    "Cl": 17.0,
    "Ar": 18.0,
}


def _canonicalize_element(symbol):
    text = str(symbol).strip()
    if not text:
        raise ValueError("Empty element label is not allowed.")
    return text[0].upper() + text[1:].lower()


def _elements_to_charges(elements):
    charges = []
    normalized = []
    for elem in elements:
        sym = _canonicalize_element(elem)
        if sym not in _ELEMENT_TO_Z:
            raise ValueError(f"Unsupported element label for GDVR AtomicChain: {elem!r}")
        normalized.append(sym)
        charges.append(_ELEMENT_TO_Z[sym])
    return normalized, np.asarray(charges, float)


def _default_transverse_basis(elements):
    if all(sym == "H" for sym in elements):
        return (
            DEFAULT_H_S_EXPS.copy(),
            DEFAULT_P_EXPS.copy(),
            DEFAULT_D_EXPS.copy(),
        )
    if all(sym == "He" for sym in elements):
        return (
            DEFAULT_HE_S_EXPS.copy(),
            DEFAULT_P_EXPS.copy(),
            DEFAULT_D_EXPS.copy(),
        )
    raise ValueError(
        "No default GDVR transverse basis is defined for this chain. "
        "Please provide s_exps explicitly."
    )


#  Basic atomic chain holder
class AtomicChain:
    def __init__(self, elements, coords, nelec=None, spin = None):
        self.elements, self.charges = _elements_to_charges(elements)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        assert self.charges.shape[0] == self.coords.shape[0]

        if nelec is None:
            self.nelec = int(round(float(np.sum(self.charges))))
        else:
            self.nelec = int(nelec)

        self.spin = spin
        self.hcore = None
        self.eri_j = None
        self.eri_k = None
        self.z = None
        self.dz = None
        self.e_slices = None
        self.c_list = None
        self.shapes = None
        self.gdvr_options = None
        self._newton_context = None

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

    def build(
        self,
        Lz=18.0,
        Nz=121,
        M=1,
        s_exps=None,
        p_exps=None,
        d_exps=None,
        verbose=True,
        dvr_method='sine',
    ):
        if s_exps is None:
            s_exps, default_p_exps, default_d_exps = _default_transverse_basis(self.elements)
            if p_exps is None:
                p_exps = default_p_exps
            if d_exps is None:
                d_exps = default_d_exps
        Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes = build_method2(
            self,
            Lz=Lz,
            Nz=Nz,
            M=M,
            s_exps=s_exps,
            p_exps=p_exps,
            d_exps=d_exps,
            verbose=verbose,
            dvr_method=dvr_method,
        )
        self.hcore = Hcore
        self.eri_j = ERI_J
        self.eri_k = ERI_K
        self.z = z
        self.dz = dz
        self.e_slices = E_slices
        self.c_list = C_list
        self.shapes = shapes
        self.gdvr_options = {
            'Lz': float(Lz),
            'Nz': int(Nz),
            'M': int(M),
            's_exps': None if s_exps is None else np.asarray(s_exps, float).copy(),
            'p_exps': None if p_exps is None else np.asarray(p_exps, float).copy(),
            'd_exps': None if d_exps is None else np.asarray(d_exps, float).copy(),
            'verbose': bool(verbose),
            'dvr_method': str(dvr_method),
        }
        self._newton_context = None
        return self

    def RHF(self):
        return RHF(self)



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
    verbose=True,
):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    z_grid = np.asarray(z_grid, float)
    Nz = z_grid.size
    
    if Nz == 0: raise ValueError("precompute_eri_method2_JK_psd: empty z_grid")
    if Nz > 1: dz = float(abs(z_grid[1] - z_grid[0]))
    else: dz = 0.0

    ERI_J = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]

    eri_by_h = {}
    
    for h in range(Nz):
        delta_z = abs(h * dz)
        eri_ao = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z)
        eri_by_h[h] = eri_ao

    for m in range(Nz):
        C_m = np.asarray(C_list[m], float)
        for n in range(Nz):
            h = abs(n - m)

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
    mol: AtomicChain,
    Lz=18.0, Nz=121, M=1,
    s_exps=None, p_exps=None, d_exps=None,
    verbose=True, dvr_method='sine'
):
    t0 = time.time()
    z, Kz, dz = _gdvr_dvr_grid(Lz, Nz, dvr_method)
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
        M=M, verbose=verbose
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


def _initial_density_guess(
    Hcore,
    nelec,
    guess="hcore",
    e_slices=None,
    z_grid=None,
    atom_z=None,
    atom_charges=None,
    atom_symbols=None,
    sad_sigma=None,
):
    N = Hcore.shape[0]
    nocc = nelec // 2
    mode = str(guess).lower()

    if mode == "slice":
        if e_slices is None:
            raise ValueError("e_slices is required for guess='slice'.")
        eps0 = np.asarray(e_slices, float).reshape(-1)
        if eps0.size != N:
            raise ValueError("Slice-energy guess has incompatible size.")
        occ_idx = np.argsort(eps0)[:nocc]
        P = np.zeros((N, N), float)
        P[occ_idx, occ_idx] = 2.0
        return P

    if mode == "sad":
        if e_slices is None or z_grid is None or atom_z is None or atom_charges is None or atom_symbols is None:
            raise ValueError("e_slices, z_grid, atom_z, atom_charges, and atom_symbols are required for guess='sad'.")

        e_slices = np.asarray(e_slices, float)
        z_grid = np.asarray(z_grid, float).reshape(-1)
        atom_z = np.asarray(atom_z, float).reshape(-1)
        atom_charges = np.asarray(atom_charges, float).reshape(-1)
        atom_symbols = [str(sym) for sym in atom_symbols]

        if e_slices.ndim != 2:
            raise ValueError("e_slices must have shape (Nz, M) for guess='sad'.")
        Nz, M = e_slices.shape
        if Nz * M != N:
            raise ValueError("Slice-energy guess has incompatible size.")

        if atom_z.size != atom_charges.size:
            raise ValueError("atom_z and atom_charges must have the same length.")

        if atom_charges.size == 0:
            return np.zeros((N, N), float)

        dz = float(abs(z_grid[1] - z_grid[0])) if z_grid.size > 1 else 1.0
        scale = 0.0 if np.sum(atom_charges) <= 0.0 else float(nelec) / float(np.sum(atom_charges))

        occ = np.zeros((Nz, M), float)
        orbital_order = np.argsort(np.mean(e_slices, axis=0))
        if sad_sigma is None:
            sigma_values = [max(2.0 * dz, 0.35)] * atom_z.size
        elif isinstance(sad_sigma, str):
            if sad_sigma.lower() != "atomic":
                raise ValueError("String sad_sigma must be 'atomic'.")
            sigma_values = []
            for sym in atom_symbols:
                sigma_values.append(max(2.0 * dz, float(slater_radii.get(sym.lower(), 0.7))))
        else:
            sigma_arr = np.asarray(sad_sigma, float)
            if sigma_arr.ndim == 0:
                sigma_values = [float(sigma_arr)] * atom_z.size
            else:
                sigma_values = sigma_arr.reshape(-1).tolist()
                if len(sigma_values) != atom_z.size:
                    raise ValueError("sad_sigma must be a scalar or one value per atom.")

        for zA, ZA, sigma in zip(atom_z, atom_charges, sigma_values):
            electrons = float(ZA) * scale
            if electrons <= 0.0:
                continue
            w = np.exp(-0.5 * ((z_grid - float(zA)) / sigma) ** 2)
            wsum = float(np.sum(w))
            if wsum <= 0.0:
                w = np.zeros_like(z_grid)
                w[int(np.argmin(np.abs(z_grid - float(zA))))] = 1.0
            else:
                w /= wsum

            remaining = electrons
            for a in orbital_order:
                if remaining <= 1e-14:
                    break
                occ_a = min(2.0, remaining)
                occ[:, a] += occ_a * w
                remaining -= occ_a

        return np.diag(occ.reshape(-1))

    if mode != "hcore":
        raise ValueError(f"Unsupported GDVR RHF initial guess: {guess!r}")

    eps, Cmo = la.eigh(Hcore)
    Cocc = Cmo[:, :nocc]
    return 2.0 * (Cocc @ Cocc.T)


def scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz, M, nelec, Enuc=0.0,
                    conv=1e-7, max_iter=50, verbose=True,
                    damp=0.20, diis_start=3, diis_space=8,
                    level_shift=0.5, shift_decay=0.75, k_ramp_iters=8,
                    guess="hcore", e_slices=None, z_grid=None,
                    atom_z=None, atom_charges=None, atom_symbols=None, sad_sigma=None):
    N = Nz * M
    nocc = nelec // 2
    I = np.eye(N)

    P = _initial_density_guess(
        Hcore,
        nelec,
        guess=guess,
        e_slices=e_slices,
        z_grid=z_grid,
        atom_z=atom_z,
        atom_charges=atom_charges,
        atom_symbols=atom_symbols,
        sad_sigma=sad_sigma,
    )

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


def _gdvr_dvr_grid(Lz, Nz, dvr_method):
    method = str(dvr_method).lower()
    if method == 'sine':
        return sine_dvr_1d(-Lz, Lz, Nz)
    if method == 'exp':
        return Exponential_dvr_1d(-Lz, Lz, Nz)
    if method == 'sinc':
        return sinc_dvr_1d(-Lz, Lz, Nz)
    raise NotImplementedError("dvr_method must be 'sine', 'exp', or 'sinc'.")


def _build_eri_kernels(alphas, centers, labels, z_grid):
    z_grid = np.asarray(z_grid, float)
    Nz = int(z_grid.size)
    if Nz < 1:
        raise ValueError("Cannot build GDVR ERI kernels for an empty z grid.")
    dz = float(abs(z_grid[1] - z_grid[0])) if Nz > 1 else 0.0
    n_ao = len(alphas)

    K_h = [np.zeros((n_ao * n_ao, n_ao * n_ao), float) for _ in range(Nz)]
    Kx_h = [np.zeros((n_ao * n_ao, n_ao * n_ao), float) for _ in range(Nz)]

    h_max = Nz - 1
    for h in range(Nz):
        delta_z = abs(h * dz)
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=delta_z)

        K_h[h] = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
        Kx_h[h] = eri_tensor.transpose(0, 2, 1, 3).reshape(n_ao * n_ao, n_ao * n_ao)

    return K_h, Kx_h, h_max


def _build_newton_context(mol):
    opts = mol.gdvr_options
    if opts is None:
        raise ValueError("Build the GDVR molecule first before constructing Newton helpers.")

    Lz = float(opts["Lz"])
    Nz = int(opts["Nz"])
    dvr_method = str(opts["dvr_method"])
    s_exps = np.asarray(opts["s_exps"], float)
    p_exps = np.asarray(opts["p_exps"], float)
    d_exps = np.asarray(opts["d_exps"], float)
    nuclei = mol.to_tuples()

    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei,
        exps_s=s_exps,
        exps_p=p_exps,
        exps_d=d_exps,
    )
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    z, Kz, dz = _gdvr_dvr_grid(Lz, Nz, dvr_method)
    h1_nm = build_h1_nm(
        Kz,
        S_prim,
        T_prim,
        z,
        lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz),
    )

    K_h, Kx_h, h_max = _build_eri_kernels(
        alphas,
        centers,
        labels,
        z,
    )

    eri_op = CollocatedERIOp.from_kernels(N=S_prim.shape[0], Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    nh_sweep = SweepNewtonHelper(h1_nm, S_prim, eri_op)

    return {
        "alphas": alphas,
        "centers": centers,
        "labels": labels,
        "S_prim": S_prim,
        "T_prim": T_prim,
        "z": z,
        "Kz": Kz,
        "dz": dz,
        "K_h": K_h,
        "Kx_h": Kx_h,
        "eri_op": eri_op,
        "nh_sweep": nh_sweep,
        "nuclei": nuclei,
        "h_max": h_max,
    }


def _get_newton_context(mol):
    if getattr(mol, "_newton_context", None) is None:
        mol._newton_context = _build_newton_context(mol)
    return mol._newton_context


class RHF:
    def __init__(self, mol):
        self.mol = mol
        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.dm = None
        self.info = None

    def run(
        self,
        conv=1e-7,
        max_iter=50,
        verbose=True,
        damp=0.20,
        diis_start=3,
        diis_space=8,
        level_shift=0.5,
        shift_decay=0.75,
        k_ramp_iters=8,
        guess="hcore",
        sad_sigma=None,
    ):
        if self.mol.hcore is None or self.mol.eri_j is None or self.mol.eri_k is None or self.mol.shapes is None:
            raise ValueError(
                "Build the GDVR molecule first with mol.build(...) before calling mol.RHF().run()."
            )

        Etot, eps, Cmo, P, info = scf_rhf_method2(
            self.mol.hcore,
            self.mol.eri_j,
            self.mol.eri_k,
            Nz=self.mol.shapes["Nz"],
            M=self.mol.shapes["M"],
            nelec=self.mol.nelec,
            Enuc=self.mol.nuclear_repulsion_energy(),
            conv=conv,
            max_iter=max_iter,
            verbose=verbose,
            damp=damp,
            diis_start=diis_start,
            diis_space=diis_space,
            level_shift=level_shift,
            shift_decay=shift_decay,
            k_ramp_iters=k_ramp_iters,
            guess=guess,
            e_slices=self.mol.e_slices,
            z_grid=self.mol.z,
            atom_z=self.mol.coords[:, 2],
            atom_charges=self.mol.charges,
            atom_symbols=self.mol.elements,
            sad_sigma=sad_sigma,
        )

        self.e_tot = float(Etot)
        self.mo_energy = np.asarray(eps, float)
        self.mo_coeff = np.asarray(Cmo, float)
        self.dm = np.asarray(P, float)
        self.info = dict(info)
        return self

    def make_rdm1(self):
        if self.dm is None:
            raise ValueError("Run GDVR RHF before requesting the density matrix.")
        return np.asarray(self.dm, float)

    @staticmethod
    def _normalize_d_stack(d_stack, S_prim):
        out = np.asarray(d_stack, float).copy()
        for n in range(out.shape[0]):
            dn = out[n]
            out[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))
        return out

    @staticmethod
    def _project_gradient_stack(nh_sweep, d_stack, P_slice, S_prim):
        grads = []
        for n in range(d_stack.shape[0]):
            g_n = np.asarray(nh_sweep.get_gradient_slice_onthefly(n, d_stack, P_slice), float)
            dn = np.asarray(d_stack[n], float)
            coeff = float(dn.T @ (S_prim @ g_n))
            g_proj = g_n - coeff * dn
            grads.append(g_proj)
        return np.vstack(grads)

    @staticmethod
    def _flatten_stack(d_stack):
        return np.asarray(d_stack, float).reshape(-1)

    @staticmethod
    def _reshape_stack(vec, template):
        return np.asarray(vec, float).reshape(template.shape)

    @staticmethod
    def _lbfgs_direction(g_vec, s_hist, y_hist, rho_hist):
        q = np.asarray(g_vec, float).copy()
        if not s_hist:
            return -q

        alpha = []
        for s_vec, y_vec, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a = rho * float(np.dot(s_vec, q))
            alpha.append(a)
            q -= a * y_vec

        y_last = y_hist[-1]
        s_last = s_hist[-1]
        ys = float(np.dot(y_last, s_last))
        yy = float(np.dot(y_last, y_last))
        gamma = 1.0 if yy <= 1e-16 else max(1e-8, ys / yy)
        r = gamma * q

        for s_vec, y_vec, rho, a in zip(s_hist, y_hist, rho_hist, reversed(alpha)):
            beta = rho * float(np.dot(y_vec, r))
            r += s_vec * (a - beta)

        return -r

    def newton(
        self,
        alt_cycles=6,
        sweep_iterations=4,
        ridge=0.5,
        trust_step=1.0,
        trust_radius=2.0,
        scf_conv=1e-8,
        scf_max_iter=80,
        max_backtrack=3,
        verbose=True,
    ):
        if self.mol.hcore is None or self.mol.shapes is None or self.mol.gdvr_options is None:
            raise ValueError(
                "Build the GDVR molecule first with mol.build(...) before calling newton()."
            )
        if self.dm is None:
            self.run(conv=scf_conv, max_iter=scf_max_iter, verbose=verbose)
        if int(self.mol.shapes["M"]) != 1:
            raise NotImplementedError("newton() currently supports only M=1.")

        ctx = _get_newton_context(self.mol)
        S_prim = ctx["S_prim"]
        T_prim = ctx["T_prim"]
        z = ctx["z"]
        Kz = ctx["Kz"]
        dz = ctx["dz"]
        alphas = ctx["alphas"]
        centers = ctx["centers"]
        labels = ctx["labels"]
        K_h = ctx["K_h"]
        Kx_h = ctx["Kx_h"]
        nh_sweep = ctx["nh_sweep"]
        nuclei = ctx["nuclei"]
        Nz = int(self.mol.shapes["Nz"])

        if self.mol.z is not None and len(self.mol.z) == len(z):
            if not np.allclose(self.mol.z, z):
                raise ValueError("Stored GDVR grid does not match the requested Newton sweep grid.")

        d_stack = np.vstack([np.asarray(self.mol.c_list[n][:, 0], float) for n in range(Nz)])
        for n in range(Nz):
            dn = d_stack[n]
            d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

        E_history = [float(self.e_tot)]
        Enuc = self.mol.nuclear_repulsion_energy()

        for cyc in range(1, int(alt_cycles) + 1):
            P_slice = self.dm.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
            d_stack, sweep_stats = sweep_optimize_driver(
                nh_sweep,
                d_stack,
                P_slice,
                S_prim,
                n_cycles=int(sweep_iterations),
                ridge=float(ridge),
                trust_step=float(trust_step),
                trust_radius=float(trust_radius),
                verbose=verbose,
                return_stats=True,
            )

            C_list = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
            Hcore = rebuild_Hcore_from_d(
                d_stack,
                z,
                Kz,
                S_prim,
                T_prim,
                alphas,
                centers,
                labels,
                nuclei,
            )
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
            Etot, eps, Cmo, P, info = scf_rhf_method2(
                Hcore,
                ERI_J,
                ERI_K,
                Nz,
                1,
                nelec=self.mol.nelec,
                Enuc=Enuc,
                conv=scf_conv,
                max_iter=scf_max_iter,
                verbose=False,
            )

            self.mol.hcore = Hcore
            self.mol.eri_j = ERI_J
            self.mol.eri_k = ERI_K
            self.mol.c_list = C_list
            self.mol.z = z
            self.mol.dz = dz

            self.e_tot = float(Etot)
            self.mo_energy = np.asarray(eps, float)
            self.mo_coeff = np.asarray(Cmo, float)
            self.dm = np.asarray(P, float)
            self.info = dict(info)
            E_history.append(self.e_tot)

            if verbose:
                print(
                    f"[Newton {cyc}] E = {self.e_tot:.12f} Ha  "
                    f"dE={E_history[-1] - E_history[-2]:+.3e}  "
                    f"max|g|={0.0 if sweep_stats is None else sweep_stats['max_grad']:.3e}  "
                    f"maxΔ={0.0 if sweep_stats is None else sweep_stats['max_delta']:.3e}"
                )

            if abs(E_history[-1] - E_history[-2]) < scf_conv:
                break

        self.info = dict(self.info)
        self.info["newton_cycles"] = len(E_history) - 1
        self.info["newton_energy_history"] = tuple(float(v) for v in E_history)
        return self

    def run_newton_sweep(self, *args, **kwargs):
        return self.newton(*args, **kwargs)

    def quasinewton(
        self,
        alt_cycles=6,
        micro_cycles=4,
        initial_step=0.5,
        min_step=1e-3,
        scf_conv=1e-8,
        scf_max_iter=80,
        history_reset_curvature=1e-10,
        history_size=6,
        verbose=True,
    ):
        if self.mol.hcore is None or self.mol.shapes is None or self.mol.gdvr_options is None:
            raise ValueError(
                "Build the GDVR molecule first with mol.build(...) before calling quasinewton()."
            )
        if self.dm is None:
            self.run(conv=scf_conv, max_iter=scf_max_iter, verbose=verbose)
        if int(self.mol.shapes["M"]) != 1:
            raise NotImplementedError("quasinewton() currently supports only M=1.")

        ctx = _get_newton_context(self.mol)
        S_prim = ctx["S_prim"]
        T_prim = ctx["T_prim"]
        z = ctx["z"]
        Kz = ctx["Kz"]
        dz = ctx["dz"]
        alphas = ctx["alphas"]
        centers = ctx["centers"]
        labels = ctx["labels"]
        K_h = ctx["K_h"]
        Kx_h = ctx["Kx_h"]
        nh_sweep = ctx["nh_sweep"]
        nuclei = ctx["nuclei"]
        Nz = int(self.mol.shapes["Nz"])

        d_stack = np.vstack([np.asarray(self.mol.c_list[n][:, 0], float) for n in range(Nz)])
        d_stack = self._normalize_d_stack(d_stack, S_prim)
        E_history = [float(self.e_tot)]
        Enuc = self.mol.nuclear_repulsion_energy()
        s_hist = []
        y_hist = []
        rho_hist = []
        step_guess = float(initial_step)
        converged = False

        for cyc in range(1, int(alt_cycles) + 1):
            cycle_energy_start = E_history[-1]
            cycle_accepts = 0
            accepted_stats = None
            cycle_failed = False

            for micro in range(int(micro_cycles)):
                P_slice = self.dm.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
                g_stack = self._project_gradient_stack(nh_sweep, d_stack, P_slice, S_prim)
                g_vec = self._flatten_stack(g_stack)
                gnorm = float(np.linalg.norm(g_vec))
                if verbose:
                    print(f"[QNewton {cyc}.{micro + 1}] |g| = {gnorm:.6e}")
                if gnorm < scf_conv:
                    converged = True
                    break

                direction = self._lbfgs_direction(g_vec, s_hist, y_hist, rho_hist)
                if float(np.dot(direction, g_vec)) >= 0.0:
                    direction = -g_vec.copy()
                    s_hist = []
                    y_hist = []
                    rho_hist = []

                accepted_micro = False
                step = max(float(min_step), float(step_guess))
                while step >= float(min_step):
                    d_trial = self._reshape_stack(self._flatten_stack(d_stack) + step * direction, d_stack)
                    d_trial = self._normalize_d_stack(d_trial, S_prim)

                    C_list = [d_trial[n].reshape(-1, 1) for n in range(Nz)]
                    Hcore = rebuild_Hcore_from_d(
                        d_trial,
                        z,
                        Kz,
                        S_prim,
                        T_prim,
                        alphas,
                        centers,
                        labels,
                        nuclei,
                    )
                    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
                    Etot, eps, Cmo, P, info = scf_rhf_method2(
                        Hcore,
                        ERI_J,
                        ERI_K,
                        Nz,
                        1,
                        nelec=self.mol.nelec,
                        Enuc=Enuc,
                        conv=scf_conv,
                        max_iter=scf_max_iter,
                        verbose=False,
                    )

                    dE = float(Etot - E_history[-1])
                    if dE <= max(10.0 * scf_conv, 1e-10):
                        P_trial_slice = np.asarray(P, float).reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
                        g_trial_stack = self._project_gradient_stack(
                            nh_sweep, d_trial, P_trial_slice, S_prim
                        )
                        s_vec = self._flatten_stack(d_trial - d_stack)
                        y_vec = self._flatten_stack(g_trial_stack - g_stack)
                        ys = float(np.dot(y_vec, s_vec))
                        if ys > float(history_reset_curvature):
                            s_hist.append(s_vec)
                            y_hist.append(y_vec)
                            rho_hist.append(1.0 / ys)
                            if len(s_hist) > int(history_size):
                                s_hist.pop(0)
                                y_hist.pop(0)
                                rho_hist.pop(0)
                        else:
                            s_hist = []
                            y_hist = []
                            rho_hist = []

                        d_stack = d_trial
                        self.mol.hcore = Hcore
                        self.mol.eri_j = ERI_J
                        self.mol.eri_k = ERI_K
                        self.mol.c_list = C_list
                        self.mol.z = z
                        self.mol.dz = dz

                        self.e_tot = float(Etot)
                        self.mo_energy = np.asarray(eps, float)
                        self.mo_coeff = np.asarray(Cmo, float)
                        self.dm = np.asarray(P, float)
                        self.info = dict(info)
                        E_history.append(self.e_tot)
                        accepted_micro = True
                        cycle_accepts += 1
                        step_guess = min(2.0, max(float(min_step), step * 1.35))
                        accepted_stats = {
                            "micro_cycle": micro + 1,
                            "step": step,
                            "dE": dE,
                            "grad_norm": gnorm,
                        }
                        break

                    step *= 0.5

                if accepted_micro:
                    if abs(E_history[-1] - E_history[-2]) < scf_conv:
                        converged = True
                        break
                    continue

                step_guess = max(float(min_step), 0.5 * step_guess)
                cycle_failed = True
                if verbose:
                    print(
                        f"[QNewton {cyc}.{micro + 1}] no acceptable step found; "
                        f"reducing step guess to {step_guess:.3e}."
                    )
                    break

            if verbose:
                if cycle_accepts and accepted_stats is not None:
                    print(
                        f"[QNewton {cyc}] E = {self.e_tot:.12f} Ha  "
                        f"dE_cycle={E_history[-1] - cycle_energy_start:+.3e}  "
                        f"last_step={accepted_stats['step']:.3f}  "
                        f"accepted={cycle_accepts}"
                    )
                else:
                    print(f"[QNewton {cyc}] no acceptable step found; stopping.")

            if converged:
                break
            if cycle_accepts == 0:
                break
            if cycle_failed and cycle_accepts < int(micro_cycles):
                continue
            if abs(E_history[-1] - cycle_energy_start) < scf_conv:
                break

        self.info = dict(self.info)
        self.info["quasinewton_cycles"] = len(E_history) - 1
        self.info["quasinewton_energy_history"] = tuple(float(v) for v in E_history)
        return self


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
            auto_ridge = max(ridge_base, ridge_base - min_eig)
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
    verbose=True,
    return_stats=False,
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
    
    cycle_stats = []
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
        cycle_stats.append({"cycle": cyc + 1, "max_grad": float(max_grad), "max_delta": float(max_delta)})
            
    if return_stats:
        stats = {
            "cycles": tuple(cycle_stats),
            "max_grad": 0.0 if not cycle_stats else max(item["max_grad"] for item in cycle_stats),
            "max_delta": 0.0 if not cycle_stats else max(item["max_delta"] for item in cycle_stats),
        }
        return d_current, stats
    return d_current

if __name__ == "__main__":
    stime = time.time()

    # charges = np.array([1.0, 1.0, 1.0, 1.0], float)
    elements = ["H", "H"]
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
    mol = AtomicChain(elements, coords, nelec=2)
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
        verbose=VERBOSE, dvr_method=DVR_METHOD,
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
