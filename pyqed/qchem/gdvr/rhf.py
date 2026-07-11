import numpy as np
import scipy.linalg as la
from scipy.special import i0e as I0e
import time
import warnings
import datetime
import os

from pyqed.qchem.atomic_data import atomic_number, element_name
from pyqed.qchem.basis import _basis_path, parse_gbs
from pyqed.qchem.gdvr.newton import CollocatedERIOp, NewtonHelper

from pyqed.qchem.gdvr.integrals import (
    STO6_EXPS_H,
    STO6_EXPS_He,
    make_xy_spd_primitive_basis,
    V_en_sp_total_at_z,
    build_h1_nm,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
    semilocal_ecp_projector_blocks,
)


STO6G_H_S_EXPS = np.asarray(STO6_EXPS_H, float).copy()
STO6G_HE_S_EXPS = np.asarray(STO6_EXPS_He, float).copy()
DEFAULT_P_EXPS = np.array([], float)
DEFAULT_D_EXPS = np.array([], float)


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
        try:
            charge = float(atomic_number(sym.lower()))
        except ValueError as exc:
            raise ValueError(f"Unsupported element label for GDVR AtomicChain: {elem!r}") from exc
        z = int(round(charge))
        if abs(charge - z) > 1e-8 or z <= 0:
            raise ValueError(f"Unsupported element label for GDVR AtomicChain: {elem!r}")
        normalized.append(element_name(z).capitalize())
        charges.append(float(z))
    return normalized, np.asarray(charges, float)


def _default_sto6g_s_exps_for_charges(charges):
    symbols = {_canonical_element_symbol_from_charge(charge) for charge in np.asarray(charges, float)}
    exps = []
    for symbol in sorted(symbols):
        if symbol == "H":
            exps.extend(np.asarray(STO6_EXPS_H, float).tolist())
        elif symbol == "He":
            exps.extend(np.asarray(STO6_EXPS_He, float).tolist())
        else:
            raise ValueError(
                "No built-in GDVR STO-6G transverse basis is defined for "
                f"element {symbol!r}. Use a named basis file or explicit s_exps."
            )
    return np.sort(np.unique(np.asarray(exps, float)))[::-1].copy()


def _default_transverse_basis(elements):
    symbols = {_canonicalize_element(elem) for elem in elements}
    if not symbols:
        raise ValueError("AtomicChain requires at least one element.")
    if symbols <= {"H", "He"}:
        charges = [atomic_number(sym.lower()) for sym in symbols]
        return _default_sto6g_s_exps_for_charges(charges), DEFAULT_P_EXPS.copy(), DEFAULT_D_EXPS.copy()
    raise ValueError(
        "No default GDVR transverse basis is defined for this chain. "
        "Please provide transverse_basis or explicit s_exps."
    )


def _canonical_element_symbol_from_charge(charge):
    z = int(round(float(charge)))
    if abs(float(charge) - z) > 1e-8 or z <= 0:
        raise ValueError(f"Unsupported nuclear charge for canonical basis lookup: {charge!r}")
    return element_name(z).capitalize()


def _unique_desc(values):
    if not values:
        return np.array([], float)
    arr = np.unique(np.asarray(values, float))
    return np.sort(arr)[::-1].copy()


def _extract_transverse_exponents_from_pyscf_basis(basis_name, charges):
    try:
        from pyscf import gto
    except Exception as exc:
        raise ValueError(
            f"Basis {basis_name!r} is not available in pyqed and PySCF could not be imported."
        ) from exc

    symbols = sorted({_canonical_element_symbol_from_charge(charge) for charge in np.asarray(charges, float)})

    s_exps = []
    p_exps = []
    d_exps = []
    for symbol in symbols:
        try:
            shells = gto.basis.load(str(basis_name), symbol)
        except Exception as exc:
            raise ValueError(f"Basis {basis_name!r} does not define element {symbol!r}.") from exc
        for shell in shells:
            angmom = int(shell[0])
            exps = [float(row[0]) for row in shell[1:]]
            if angmom == 0:
                s_exps.extend(exps)
            elif angmom == 1:
                p_exps.extend(exps)
            elif angmom == 2:
                d_exps.extend(exps)

    return _unique_desc(s_exps), _unique_desc(p_exps), _unique_desc(d_exps)


def _extract_transverse_exponents_from_basis(basis_name, charges):
    try:
        basis_dict = parse_gbs(_basis_path(basis_name))
    except Exception:
        return _extract_transverse_exponents_from_pyscf_basis(basis_name, charges)

    symbols = sorted({_canonical_element_symbol_from_charge(charge) for charge in np.asarray(charges, float)})

    s_exps = []
    p_exps = []
    d_exps = []
    for symbol in symbols:
        if symbol not in basis_dict:
            raise ValueError(f"Basis {basis_name!r} does not define element {symbol!r}.")
        for angmom, exps, _coeffs in basis_dict[symbol]:
            if angmom == 0:
                s_exps.extend(np.asarray(exps, float).tolist())
            elif angmom == 1:
                p_exps.extend(np.asarray(exps, float).tolist())
            elif angmom == 2:
                d_exps.extend(np.asarray(exps, float).tolist())

    return _unique_desc(s_exps), _unique_desc(p_exps), _unique_desc(d_exps)


def local_ecp_terms_from_pyscf(symbol, ecp_name="bfd", scalarize_nonlocal=False):
    """Return PySCF ECP terms split into local and semilocal radial channels."""
    try:
        from pyscf import gto
    except Exception as exc:
        raise ValueError("PySCF is required to load named ECP data.") from exc

    core_electrons, channels = gto.basis.load_ecp(str(ecp_name), _canonicalize_element(symbol))
    local_terms = []
    semilocal_terms = []
    omitted_channels = []
    scalarized_channels = []
    for angular_momentum, radial_terms in channels:
        if int(angular_momentum) == -1:
            for slot, entries in enumerate(radial_terms):
                power = int(slot) - 2
                for entry in entries:
                    exponent = float(entry[0])
                    coeff = float(entry[1])
                    local_terms.append((power, exponent, coeff))
        elif any(len(entries) > 0 for entries in radial_terms):
            if scalarize_nonlocal:
                scalarized_channels.append(int(angular_momentum))
                for slot, entries in enumerate(radial_terms):
                    power = int(slot) - 2
                    for entry in entries:
                        exponent = float(entry[0])
                        coeff = float(entry[1])
                        local_terms.append((power, exponent, coeff))
            else:
                omitted_channels.append(int(angular_momentum))
                for slot, entries in enumerate(radial_terms):
                    power = int(slot) - 2
                    for entry in entries:
                        exponent = float(entry[0])
                        coeff = float(entry[1])
                        semilocal_terms.append((int(angular_momentum), power, exponent, coeff))
    return {
        "symbol": _canonicalize_element(symbol),
        "ecp_name": str(ecp_name),
        "core_electrons": int(core_electrons),
        "local_terms": tuple(local_terms),
        "semilocal_terms": tuple(semilocal_terms),
        "omitted_nonlocal_channels": tuple(omitted_channels),
        "scalarized_nonlocal_channels": tuple(scalarized_channels),
    }


def _normalize_local_ecp_terms(local_ecp_terms, nnuc):
    if local_ecp_terms is None:
        return None
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
            clean_terms.append((int(power), float(exponent), float(coeff)))
        out.append(tuple(clean_terms))
    return tuple(out)


def _normalize_semilocal_ecp_terms(semilocal_ecp_terms, nnuc):
    if semilocal_ecp_terms is None:
        return None
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
            clean_terms.append((int(angular_momentum), int(power), float(exponent), float(coeff)))
        out.append(tuple(clean_terms))
    return tuple(out)


def _resolve_transverse_basis(charges, transverse_basis=None, s_exps=None, p_exps=None, d_exps=None):
    basis_name = transverse_basis
    if basis_name is not None:
        if s_exps is not None or p_exps is not None or d_exps is not None:
            raise ValueError("Use either named transverse basis or explicit s_exps/p_exps/d_exps, not both.")
        key = str(basis_name).strip().lower()
        if key == "sto6g":
            return _default_sto6g_s_exps_for_charges(charges), DEFAULT_P_EXPS.copy(), DEFAULT_D_EXPS.copy(), "sto6g"
        s_exps, p_exps, d_exps = _extract_transverse_exponents_from_basis(basis_name, charges)
        return s_exps, p_exps, d_exps, str(basis_name)

    if s_exps is None:
        raise ValueError("Please provide s_exps or transverse_basis='sto6g'.")
    if p_exps is None:
        p_exps = np.array([], float)
    if d_exps is None:
        d_exps = np.array([], float)
    return np.asarray(s_exps, float), np.asarray(p_exps, float), np.asarray(d_exps, float), None


#  Basic molecule holder
class Molecule:
    def __init__(
        self,
        charges,
        coords,
        nelec=None,
        spin=None,
        softcore_radii=None,
        basis_charges=None,
        local_ecp_terms=None,
        semilocal_ecp_terms=None,
        slice_local_ecp_terms=None,
        ecp_metadata=None,
    ):
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        assert self.charges.shape[0] == self.coords.shape[0]
        if softcore_radii is None:
            self.softcore_radii = None
        else:
            self.softcore_radii = np.asarray(softcore_radii, float).reshape(-1)
            if self.softcore_radii.shape != self.charges.shape:
                raise ValueError("softcore_radii must match the number of nuclei.")
            if np.any(self.softcore_radii < 0.0):
                raise ValueError("softcore_radii must be non-negative.")
        if basis_charges is None:
            self.basis_charges = None
        else:
            self.basis_charges = np.asarray(basis_charges, float).reshape(-1)
            if self.basis_charges.shape != self.charges.shape:
                raise ValueError("basis_charges must match the number of nuclei.")
        self.local_ecp_terms = _normalize_local_ecp_terms(local_ecp_terms, len(self.charges))
        self.semilocal_ecp_terms = _normalize_semilocal_ecp_terms(semilocal_ecp_terms, len(self.charges))
        self.slice_local_ecp_terms = _normalize_local_ecp_terms(slice_local_ecp_terms, len(self.charges))
        self.ecp_metadata = {} if ecp_metadata is None else dict(ecp_metadata)

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
        self._gdvr_build_context = None

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

    def position_operator(self, axis="z"):
        """Return the one-electron GDVR position operator along an axis."""
        key = str(axis).strip().lower()
        if key != "z":
            raise NotImplementedError("GDVR currently has a built-in position operator only along z.")
        if self.z is None or self.shapes is None:
            raise ValueError("Build the GDVR molecule before requesting a position operator.")
        nz = int(self.shapes["Nz"])
        m = int(self.shapes["M"])
        z = np.asarray(self.z, dtype=float).reshape(nz)
        return np.diag(np.repeat(z, m))

    def dipole_operator(self, axis="z", *, electronic=True):
        """Return the one-electron GDVR dipole operator along an axis."""
        op = self.position_operator(axis)
        return -op if electronic else op

    def build(
        self,
        Lz=18.0,
        Nz=121,
        M=1,
        transverse_basis=None,
        s_exps=None,
        p_exps=None,
        d_exps=None,
        verbose=True,
        dvr_method='sine',
    ):
        Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes = build_method2(
            self,
            Lz=Lz,
            Nz=Nz,
            M=M,
            transverse_basis=transverse_basis,
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
            "Lz": float(Lz),
            "Nz": int(Nz),
            "M": int(M),
            "transverse_basis": None if transverse_basis is None else str(transverse_basis),
            "s_exps": None if s_exps is None else np.asarray(s_exps, float).copy(),
            "p_exps": None if p_exps is None else np.asarray(p_exps, float).copy(),
            "d_exps": None if d_exps is None else np.asarray(d_exps, float).copy(),
            "verbose": bool(verbose),
            "dvr_method": str(dvr_method),
        }
        self._newton_context = None
        return self

    def RHF(self):
        return RHF(self)


class AtomicChain(Molecule):
    def __init__(
        self,
        elements,
        coords,
        nelec=None,
        spin=None,
        softcore_radii=None,
        basis_charges=None,
        local_ecp_terms=None,
        semilocal_ecp_terms=None,
        slice_local_ecp_terms=None,
        ecp_metadata=None,
    ):
        self.elements, charges = _elements_to_charges(elements)
        super().__init__(
            charges,
            coords,
            nelec=nelec,
            spin=spin,
            softcore_radii=softcore_radii,
            basis_charges=basis_charges,
            local_ecp_terms=local_ecp_terms,
            semilocal_ecp_terms=semilocal_ecp_terms,
            slice_local_ecp_terms=slice_local_ecp_terms,
            ecp_metadata=ecp_metadata,
        )

    def build(
        self,
        Lz=18.0,
        Nz=121,
        M=1,
        transverse_basis=None,
        s_exps=None,
        p_exps=None,
        d_exps=None,
        verbose=True,
        dvr_method='sine',
    ):
        if transverse_basis is None and s_exps is None:
            s_exps, default_p_exps, default_d_exps = _default_transverse_basis(self.elements)
            if p_exps is None:
                p_exps = default_p_exps
            if d_exps is None:
                d_exps = default_d_exps
        return super().build(
            Lz=Lz,
            Nz=Nz,
            M=M,
            transverse_basis=transverse_basis,
            s_exps=s_exps,
            p_exps=p_exps,
            d_exps=d_exps,
            verbose=verbose,
            dvr_method=dvr_method,
        )



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
    return_ao_kernels=False,
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
    h_max = Nz - 1
    
    for h in range(Nz):
        delta_z = abs(h * dz)
        eri_ao = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z)
        eri_by_h[h] = eri_ao

    if int(M) == 1:
        K_h, Kx_h = _ao_kernel_matrices_from_eri_by_h(eri_by_h, Nz, len(alphas))
        ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
        if return_ao_kernels:
            return ERI_J, ERI_K, eri_by_h, int(h_max)
        return ERI_J, ERI_K

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
            
    if return_ao_kernels:
        return ERI_J, ERI_K, eri_by_h, int(h_max)
    return ERI_J, ERI_K


def _ao_kernel_matrices_from_eri_by_h(eri_by_h, Nz, n_ao):
    K_h = [None for _ in range(int(Nz))]
    Kx_h = [None for _ in range(int(Nz))]
    for h, eri_tensor in eri_by_h.items():
        h = int(h)
        if h < 0 or h >= Nz:
            continue
        K_h[h] = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
        eri_perm = eri_tensor.transpose(0, 2, 1, 3)
        Kx_h[h] = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
    return K_h, Kx_h


def _as_block_matrix(block, mm):
    arr = np.asarray(block, dtype=float)
    if arr.ndim == 0:
        if mm != 1:
            raise ValueError("Scalar GDVR ERI block is valid only for M=1.")
        return arr.reshape(1, 1)
    return arr.reshape(mm, mm)


class _PackedBlockMatvec:
    def __init__(self, m_idx, n_idx, blocks, low_rank_tol=1e-11):
        self.m_idx = np.asarray(m_idx, dtype=np.intp)
        self.n_idx = np.asarray(n_idx, dtype=np.intp)
        self.blocks = np.asarray(blocks, dtype=float)
        if self.blocks.ndim != 3:
            raise ValueError("Packed GDVR ERI blocks must have shape (npair, mm, mm).")
        if self.blocks.shape[0] != self.m_idx.size or self.blocks.shape[0] != self.n_idx.size:
            raise ValueError("Packed GDVR ERI pair indices do not match block count.")
        self.dim = int(self.blocks.shape[1])
        self.low_rank_groups = self._build_low_rank_groups(low_rank_tol)

    def _build_low_rank_groups(self, low_rank_tol):
        if self.blocks.size == 0:
            return None
        dim = self.dim
        tol = float(low_rank_tol)
        groups = {}
        total_rank = 0
        for pos, block in enumerate(self.blocks):
            sym_block = 0.5 * (block + block.T)
            evals, evecs = la.eigh(sym_block)
            scale = float(np.max(np.abs(evals))) if evals.size else 0.0
            if scale == 0.0:
                keep = np.zeros_like(evals, dtype=bool)
            elif tol <= 0.0:
                keep = np.abs(evals) > 0.0
            else:
                keep = np.abs(evals) > tol * scale
            rank = int(np.count_nonzero(keep))
            total_rank += rank
            if rank == 0:
                left = np.zeros((dim, 0), dtype=float)
                right = np.zeros((dim, 0), dtype=float)
            else:
                vec = evecs[:, keep]
                left = vec * evals[keep][np.newaxis, :]
                right = vec
            groups.setdefault(rank, [[], [], []])
            groups[rank][0].append(pos)
            groups[rank][1].append(left)
            groups[rank][2].append(right)

        dense_work = self.blocks.shape[0] * dim * dim
        low_rank_work = 2 * dim * total_rank
        if low_rank_work >= 0.85 * dense_work:
            return None

        packed = []
        for rank, (positions, left, right) in groups.items():
            packed.append(
                (
                    np.asarray(positions, dtype=np.intp),
                    np.stack(left, axis=0),
                    np.stack(right, axis=0),
                )
            )
        return tuple(packed)

    def dot(self, vectors):
        vectors = np.asarray(vectors)
        if self.blocks.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.result_type(vectors, float))
        if self.low_rank_groups is None:
            return np.einsum("puv,pv->pu", self.blocks, vectors, optimize=True)

        out = np.zeros((self.blocks.shape[0], self.dim), dtype=np.result_type(vectors, float))
        for positions, left, right in self.low_rank_groups:
            if left.shape[2] == 0:
                continue
            vals = vectors[positions]
            tmp = np.einsum("pur,pu->pr", right, vals, optimize=True)
            out[positions] = np.einsum("pur,pr->pu", left, tmp, optimize=True)
        return out

    @property
    def uses_low_rank(self):
        return self.low_rank_groups is not None


class GDVRFockBuilder:
    """Packed/banded two-electron Fock builder for collocated GDVR ERIs."""

    def __init__(
        self,
        ERI_J,
        ERI_K,
        Nz,
        M,
        active_tol=0.0,
        low_rank_tol=1e-11,
    ):
        self.Nz = int(Nz)
        self.M = int(M)
        self.mm = self.M * self.M
        self.active_tol = float(active_tol)
        self.low_rank_tol = float(low_rank_tol)
        if self.Nz <= 0 or self.M <= 0:
            raise ValueError("Nz and M must be positive for a GDVR Fock builder.")

        if self.M == 1:
            self.j_scalar, self.k_scalar = self._pack_scalar(ERI_J, ERI_K)
            self.j_blocks = None
            self.k_blocks = None
            self.k_upper_blocks = None
            self.exchange_mirror_ok = True
        else:
            self.j_scalar = None
            self.k_scalar = None
            self.j_blocks = self._pack_blocks(ERI_J)
            self.k_blocks = self._pack_blocks(ERI_K)
            self.exchange_mirror_ok = self._check_exchange_mirror(ERI_K)
            self.k_upper_blocks = self._pack_blocks(ERI_K, upper_only=True)

    def _is_active(self, block):
        if self.active_tol <= 0.0:
            return np.any(block != 0.0)
        return float(np.max(np.abs(block))) > self.active_tol

    def _pack_scalar(self, ERI_J, ERI_K):
        j = np.zeros((self.Nz, self.Nz), dtype=float)
        k = np.zeros((self.Nz, self.Nz), dtype=float)
        for m in range(self.Nz):
            for n in range(self.Nz):
                j[m, n] = float(np.asarray(ERI_J[m][n]).reshape(-1)[0])
                k[m, n] = float(np.asarray(ERI_K[m][n]).reshape(-1)[0])
        return j, k

    def _pack_blocks(self, ERI, upper_only=False):
        m_idx = []
        n_idx = []
        blocks = []
        for m in range(self.Nz):
            n_start = m if upper_only else 0
            for n in range(n_start, self.Nz):
                block = _as_block_matrix(ERI[m][n], self.mm)
                if not self._is_active(block):
                    continue
                m_idx.append(m)
                n_idx.append(n)
                blocks.append(block)
        if blocks:
            block_arr = np.stack(blocks, axis=0)
        else:
            block_arr = np.zeros((0, self.mm, self.mm), dtype=float)
        return _PackedBlockMatvec(m_idx, n_idx, block_arr, low_rank_tol=self.low_rank_tol)

    def _check_exchange_mirror(self, ERI_K):
        swap = np.arange(self.mm).reshape(self.M, self.M).T.reshape(-1)
        for m in range(self.Nz):
            for n in range(m + 1, self.Nz):
                upper = _as_block_matrix(ERI_K[m][n], self.mm)
                lower = _as_block_matrix(ERI_K[n][m], self.mm)
                mirrored = upper[np.ix_(swap, swap)]
                if not np.allclose(lower, mirrored, rtol=1e-9, atol=1e-11):
                    return False
        return True

    @property
    def uses_low_rank(self):
        if self.M == 1:
            return False
        return (
            self.j_blocks.uses_low_rank
            or self.k_blocks.uses_low_rank
            or self.k_upper_blocks.uses_low_rank
        )

    @property
    def active_pair_count(self):
        if self.M == 1:
            return int(np.count_nonzero(self.j_scalar) + np.count_nonzero(self.k_scalar))
        return int(self.j_blocks.m_idx.size + self.k_blocks.m_idx.size)

    def fock(self, P, k_scale=1.0, hermitian_exchange=True):
        P = np.asarray(P)
        dtype = np.result_type(P, float)
        if self.M == 1:
            P2 = P.reshape(self.Nz, self.Nz)
            rho = np.diag(P2)
            j_diag = self.j_scalar @ rho
            f2e = np.diag(j_diag).astype(dtype, copy=False)
            f2e = np.asarray(f2e, dtype=dtype)
            f2e -= 0.5 * float(k_scale) * self.k_scalar * P2
            return f2e

        P4 = P.reshape(self.Nz, self.M, self.Nz, self.M)
        f4 = np.zeros((self.Nz, self.M, self.Nz, self.M), dtype=dtype)

        diag_blocks = P4[np.arange(self.Nz), :, np.arange(self.Nz), :]
        diag_vec = diag_blocks.reshape(self.Nz, self.mm)
        j_vec = np.zeros((self.Nz, self.mm), dtype=dtype)
        if self.j_blocks.m_idx.size:
            vals = self.j_blocks.dot(diag_vec[self.j_blocks.n_idx])
            np.add.at(j_vec, self.j_blocks.m_idx, vals)
        f4[np.arange(self.Nz), :, np.arange(self.Nz), :] = j_vec.reshape(self.Nz, self.M, self.M)

        use_upper = bool(hermitian_exchange) and self.exchange_mirror_ok and np.allclose(
            P,
            P.conj().T,
            rtol=1e-10,
            atol=1e-12,
        )
        k_builder = self.k_upper_blocks if use_upper else self.k_blocks
        if k_builder.m_idx.size:
            block_vec = P4[k_builder.m_idx, :, k_builder.n_idx, :].reshape(
                k_builder.m_idx.size, self.mm
            )
            vals = k_builder.dot(block_vec).reshape(k_builder.m_idx.size, self.M, self.M)
            k4 = np.zeros_like(f4)
            np.add.at(
                k4,
                (k_builder.m_idx, slice(None), k_builder.n_idx, slice(None)),
                vals,
            )
            if use_upper:
                off = k_builder.m_idx != k_builder.n_idx
                if np.any(off):
                    np.add.at(
                        k4,
                        (
                            k_builder.n_idx[off],
                            slice(None),
                            k_builder.m_idx[off],
                            slice(None),
                        ),
                        vals[off].conj().swapaxes(1, 2),
                    )
            f4 -= 0.5 * float(k_scale) * k4
        return f4.reshape(self.Nz * self.M, self.Nz * self.M)


def prepare_gdvr_fock_builder(ERI_J, ERI_K=None, Nz=None, M=None):
    if isinstance(ERI_J, GDVRFockBuilder):
        return ERI_J
    if isinstance(ERI_K, GDVRFockBuilder):
        return ERI_K
    if Nz is None or M is None:
        raise ValueError("Nz and M are required when packing GDVR ERIs.")
    return GDVRFockBuilder(ERI_J, ERI_K, Nz, M)


def _fock_2e_slice_collocated_reference(P, ERI_J, ERI_K, Nz, M, k_scale=1.0):
    N = Nz * M
    dtype = np.result_type(P, float)
    P4 = P.reshape(Nz, M, Nz, M)
    Ddiag = [P4[b, :, b, :].copy() for b in range(Nz)]
    F2e = np.zeros((N, N), dtype=dtype)

    for m in range(Nz):
        J_mm = np.zeros((M, M), dtype=dtype)
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


def fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=1.0):
    builder = prepare_gdvr_fock_builder(ERI_J, ERI_K, Nz, M)
    return builder.fock(P, k_scale=k_scale)



#  Build Method II (core + ERIs) and SCF TODO: remove the naming of Method II, i used to think and tried Method I as globally share one set of optimized AO but that is a lot less efficient, already removed
def build_method2(
    mol: Molecule,
    Lz=18.0, Nz=121, M=1,
    transverse_basis=None,
    s_exps=None, p_exps=None, d_exps=None,
    verbose=True, dvr_method='sine'
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

    basis_charges = mol.charges if getattr(mol, "basis_charges", None) is None else mol.basis_charges
    s_exps, p_exps, d_exps, _basis_name = _resolve_transverse_basis(
        charges=basis_charges,
        transverse_basis=transverse_basis,
        s_exps=s_exps,
        p_exps=p_exps,
        d_exps=d_exps,
    )

    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei,
        exps_s=s_exps,
        exps_p=p_exps,
        exps_d=d_exps,
    )
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    softcore_radii = getattr(mol, "softcore_radii", None)
    local_ecp_terms = getattr(mol, "local_ecp_terms", None)
    semilocal_ecp_terms = getattr(mol, "semilocal_ecp_terms", None)
    slice_local_ecp_terms = getattr(mol, "slice_local_ecp_terms", None)
    if slice_local_ecp_terms is None:
        slice_local_ecp_terms = local_ecp_terms
    actual_v_cache = {}
    slice_v_cache = {}

    def V_actual_of_z(zk: float) -> np.ndarray:
        return V_en_sp_total_at_z(
            alphas,
            centers,
            labels,
            nuclei,
            zk,
            softcore_radii=softcore_radii,
            local_ecp_terms=local_ecp_terms,
            matrix_cache=actual_v_cache,
        )

    def V_slice_of_z(zk: float) -> np.ndarray:
        return V_en_sp_total_at_z(
            alphas,
            centers,
            labels,
            nuclei,
            zk,
            softcore_radii=softcore_radii,
            local_ecp_terms=slice_local_ecp_terms,
            matrix_cache=slice_v_cache,
        )

    E_slices, C_list = slice_eigens_xy(
        z_grid=z, S_prim=S_prim, T_prim=T_prim, V_en_of_z=V_slice_of_z, M=M,
    )

    Nz = len(z)
    if int(M) == 1:
        d_stack = np.vstack([np.asarray(C_list[k][:, 0], float) for k in range(Nz)])
        S_scalar = d_stack @ S_prim @ d_stack.T
        S_slice = S_scalar[:, :, None, None]
    else:
        S_slice = np.zeros((Nz, Nz, M, M), float)
        for k in range(Nz):
            Ck = C_list[k]
            for m in range(Nz):
                Cm = C_list[m]
                S_slice[k, m] = Ck.T @ (S_prim @ Cm)

    size = Nz * M
    h1_nm = None
    h_local_ops = None
    if semilocal_ecp_terms is None:
        if int(M) == 1:
            Hcore = (Kz * S_scalar).astype(float, copy=False)
        else:
            Hcore = np.einsum('km,kmab->kamb', Kz, S_slice, optimize=True).reshape(size, size)
        Hcore += np.diag(E_slices.reshape(-1))
    else:
        h1_nm = build_h1_nm(Kz, S_prim, T_prim, z, V_actual_of_z)
        semilocal_blocks = semilocal_ecp_projector_blocks(
            alphas,
            centers,
            labels,
            nuclei,
            z,
            dz,
            semilocal_ecp_terms,
            dvr_method=dvr_method,
        )
        if semilocal_blocks is not None:
            h1_nm += semilocal_blocks
        if int(M) == 1:
            Hcore = np.einsum("ni,nmij,mj->nm", d_stack, h1_nm, d_stack, optimize=True)
        else:
            H_blocks = np.zeros((Nz, Nz, M, M), dtype=float)
            for k in range(Nz):
                Ck = np.asarray(C_list[k], float)
                for m in range(Nz):
                    Cm = np.asarray(C_list[m], float)
                    H_blocks[k, m] = Ck.T @ (h1_nm[k, m] @ Cm)
            Hcore = H_blocks.reshape(size, size)
        Hcore = 0.5 * (Hcore + Hcore.T)

    if verbose:
        print(f"[Method2] Built Hcore {Hcore.shape} in {time.time()-t0:.2f}s")

    eri_result = precompute_eri_method2_JK_psd(
        alphas, centers, labels, z, C_list,
        M=M, verbose=verbose,
        return_ao_kernels=(int(M) == 1),
    )
    if int(M) == 1:
        ERI_J, ERI_K, eri_by_h, h_max = eri_result
        K_h, Kx_h = _ao_kernel_matrices_from_eri_by_h(eri_by_h, Nz, len(alphas))
    else:
        ERI_J, ERI_K = eri_result
        K_h = None
        Kx_h = None
        h_max = None

    shapes = {"Nz": Nz, "M": M, "n_ao2d": len(alphas), "size": size, "dz": dz}
    mol._gdvr_build_context = {
        "alphas": np.asarray(alphas, float).copy(),
        "centers": np.asarray(centers, float).copy(),
        "labels": tuple(labels),
        "S_prim": np.asarray(S_prim, float).copy(),
        "T_prim": np.asarray(T_prim, float).copy(),
        "z": np.asarray(z, float).copy(),
        "Kz": np.asarray(Kz, float).copy(),
        "dz": float(dz),
        "h1_nm": None if h1_nm is None else np.asarray(h1_nm, float).copy(),
        "h_local_ops": h_local_ops,
        "K_h": K_h,
        "Kx_h": Kx_h,
        "eri_h_max": None if h_max is None else int(h_max),
    }
    return Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes


class RHF:
    def __init__(self, mol):
        self.mol = mol
        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
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
        dm0=None,
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
            dm0=dm0,
        )

        self.e_tot = float(Etot)
        self.mo_energy = np.asarray(eps, float)
        self.mo_coeff = np.asarray(Cmo, float)
        self.mo_occ = np.zeros_like(self.mo_energy)
        self.mo_occ[: self.mol.nelec // 2] = 2.0
        self.dm = np.asarray(P, float)
        self.info = dict(info)
        return self

    def make_rdm1(self):
        if self.dm is None:
            raise ValueError("Run GDVR RHF before requesting the density matrix.")
        return np.asarray(self.dm, float)

    def RTTDHF(self, *args, **kwargs):
        from pyqed.qchem.gdvr.rttdhf import RTTDHF

        return RTTDHF(self, *args, **kwargs)

    def to_gto(self, orbitals=None):
        """Return a GTO-like qchem mean-field view backed by GDVR integrals."""
        from pyqed.qchem.gdvr.tddmrg import GDVRMeanFieldAdapter

        if self.mo_coeff is None or self.dm is None:
            raise ValueError("Run GDVR RHF before requesting a GTO-like qchem view.")
        mo_coeff = None
        if orbitals is not None:
            orbitals = tuple(int(i) for i in orbitals)
            mo_coeff = np.asarray(self.mo_coeff)[:, orbitals]
        return GDVRMeanFieldAdapter(self, mo_coeff=mo_coeff)

    def TDDMRG(self, *, ncas=None, nelecas=None, orbitals=None, **kwargs):
        """Construct direct GDVR TDDMRG or active-space qchem TDDMRG."""
        if ncas is None:
            if orbitals is not None:
                raise ValueError("orbitals is only meaningful when ncas is set.")
            from pyqed.qchem.gdvr.tddmrg import TDDMRG

            return TDDMRG(self, nelecas=nelecas, **kwargs)

        if nelecas is None:
            nelecas = int(self.mol.nelec)
        from pyqed.qchem.dmrg import TDDMRG

        return TDDMRG(
            self.to_gto(orbitals=orbitals),
            ncas=int(ncas),
            nelecas=nelecas,
            **kwargs,
        )

    def newton(
        self,
        tol=None,
        max_cycles=50,
        sweep_iterations=4,
        ridge=0.5,
        trust_step=1.0,
        trust_radius=2.0,
        scf_conv=1e-7,
        scf_max_iter=100,
        scf_micro_iter=0,
        scf_full_every=0,
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
        if tol is None:
            tol = scf_conv
        tol = float(tol)
        if tol <= 0.0:
            raise ValueError("newton() requires tol > 0.")
        max_cycles = int(max_cycles)
        if max_cycles <= 0:
            raise ValueError("newton() requires max_cycles >= 1.")
        if scf_micro_iter is None:
            scf_micro_iter = 0
        scf_micro_iter = int(scf_micro_iter)
        if scf_full_every is None:
            scf_full_every = 0
        scf_full_every = int(scf_full_every)

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
        h1_nm = ctx["h1_nm"]
        h_local_ops = ctx["h_local_ops"]
        nh_sweep = ctx["nh_sweep"]
        nuclei = ctx["nuclei"]
        Nz = int(self.mol.shapes["Nz"])

        d_stack = np.vstack([np.asarray(self.mol.c_list[n][:, 0], float) for n in range(Nz)])
        for n in range(Nz):
            dn = d_stack[n]
            d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

        E_history = [float(self.e_tot)]
        Enuc = self.mol.nuclear_repulsion_energy()

        def _run_cycle_scf(hcore, eri_j, eri_k, dm_guess, max_iter):
            return scf_rhf_method2(
                hcore,
                eri_j,
                eri_k,
                Nz,
                1,
                nelec=self.mol.nelec,
                Enuc=Enuc,
                conv=scf_conv,
                max_iter=max_iter,
                dm0=dm_guess,
                verbose=False,
            )

        converged = False
        for cyc in range(1, max_cycles + 1):
            if verbose:
                print(f"\n--- Alternation Cycle {cyc} ---")

            P_slice = self.dm.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
            t_sweep = time.time()
            d_stack = sweep_optimize_driver(
                nh_sweep,
                d_stack,
                P_slice,
                S_prim,
                n_cycles=int(sweep_iterations),
                ridge=float(ridge),
                trust_step=float(trust_step),
                trust_radius=float(trust_radius),
                verbose=verbose,
            )
            if verbose:
                print(f"   Sweep finished in {time.time() - t_sweep:.4f}s")

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
                h_local_ops=h_local_ops,
                h1_nm=h1_nm,
            )
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)

            run_full_scf = (
                scf_micro_iter <= 0
                or scf_micro_iter >= int(scf_max_iter)
                or scf_full_every <= 0
                or (cyc % scf_full_every == 0)
            )

            if run_full_scf:
                Etot, eps, Cmo, P, info = _run_cycle_scf(
                    Hcore, ERI_J, ERI_K, self.dm, int(scf_max_iter)
                )
                info = dict(info)
                info["newton_scf_mode"] = "full"
            else:
                Etot, eps, Cmo, P, info = _run_cycle_scf(
                    Hcore, ERI_J, ERI_K, self.dm, scf_micro_iter
                )
                info = dict(info)
                info["newton_scf_mode"] = "micro"

                # Re-anchor with a full SCF when the outer iteration appears close
                # to convergence, so the stop test uses a fully relaxed density.
                if abs(float(Etot) - E_history[-1]) < max(10.0 * tol, 1e-5):
                    Etot, eps, Cmo, P, info_full = _run_cycle_scf(
                        Hcore, ERI_J, ERI_K, P, int(scf_max_iter)
                    )
                    info = dict(info_full)
                    info["newton_scf_mode"] = "micro+full"

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
                print(f"   [SCF] E = {self.e_tot:.12f} Eh  ΔE={E_history[-1] - E_history[-2]:+.3e}")

            if abs(E_history[-1] - E_history[-2]) < tol:
                converged = True
                if verbose:
                    print("Converged.")
                break

        self.info = dict(self.info)
        self.info["newton_cycles"] = len(E_history) - 1
        self.info["newton_energy_history"] = tuple(float(v) for v in E_history)
        self.info["newton_tol"] = tol
        self.info["newton_converged"] = bool(converged)
        return self


def _gdvr_grid_for_options(Lz, Nz, dvr_method):
    if dvr_method == "sine":
        return sine_dvr_1d(-Lz, Lz, Nz)
    if dvr_method == "exp":
        return Exponential_dvr_1d(-Lz, Lz, Nz)
    if dvr_method == "sinc":
        return sinc_dvr_1d(-Lz, Lz, Nz)
    raise NotImplementedError("use sine, exp, or sinc")


def _build_newton_context(mol: Molecule):
    if mol.gdvr_options is None or mol.shapes is None or mol.z is None:
        raise ValueError("Build the GDVR molecule first before preparing Newton context.")

    opts = mol.gdvr_options
    Nz = int(mol.shapes["Nz"])
    Lz = float(opts["Lz"])
    dvr_method = str(opts["dvr_method"])

    nuclei = mol.to_tuples()
    cache = getattr(mol, "_gdvr_build_context", None)
    use_cache = False
    if cache is not None:
        z_cached = cache.get("z")
        use_cache = (
            z_cached is not None
            and np.asarray(z_cached).shape == np.asarray(mol.z).shape
            and np.allclose(z_cached, mol.z)
        )

    if use_cache:
        alphas = np.asarray(cache["alphas"], float)
        centers = np.asarray(cache["centers"], float)
        labels = tuple(cache["labels"])
        S_prim = np.asarray(cache["S_prim"], float)
        T_prim = np.asarray(cache["T_prim"], float)
        z_chk = np.asarray(cache["z"], float)
        Kz = np.asarray(cache["Kz"], float)
        dz_chk = float(cache["dz"])
    else:
        basis_charges = mol.charges if getattr(mol, "basis_charges", None) is None else mol.basis_charges
        s_exps, p_exps, d_exps, _basis_name = _resolve_transverse_basis(
            charges=basis_charges,
            transverse_basis=opts.get("transverse_basis"),
            s_exps=opts["s_exps"],
            p_exps=opts["p_exps"],
            d_exps=opts["d_exps"],
        )

        alphas, centers, labels = make_xy_spd_primitive_basis(
            nuclei,
            exps_s=s_exps,
            exps_p=p_exps,
            exps_d=d_exps,
        )
        S_prim = overlap_2d_cartesian(alphas, centers, labels)
        T_prim = kinetic_2d_cartesian(alphas, centers, labels)
        z_chk, Kz, dz_chk = _gdvr_grid_for_options(Lz, Nz, dvr_method)
    if not np.allclose(z_chk, mol.z):
        raise ValueError("Stored GDVR grid does not match the requested Newton sweep grid.")

    n_ao = len(alphas)
    K_h = None if not use_cache else cache.get("K_h")
    Kx_h = None if not use_cache else cache.get("Kx_h")
    if K_h is None or Kx_h is None:
        h_max = Nz - 1
        K_h = [None for _ in range(Nz)]
        Kx_h = [None for _ in range(Nz)]
        for h in range(Nz):
            eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=h * dz_chk)
            K_h[h] = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
            eri_perm = eri_tensor.transpose(0, 2, 1, 3)
            Kx_h[h] = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
    else:
        K_h = list(K_h)
        Kx_h = list(Kx_h)
        h_max = cache.get("eri_h_max")

    actual_v_cache = {}
    h1_nm = None if not use_cache else cache.get("h1_nm")
    if h1_nm is None:
        h1_nm = build_h1_nm(
            Kz,
            S_prim,
            T_prim,
            mol.z,
            lambda zz: V_en_sp_total_at_z(
                alphas,
                centers,
                labels,
                nuclei,
                zz,
                softcore_radii=getattr(mol, "softcore_radii", None),
                local_ecp_terms=getattr(mol, "local_ecp_terms", None),
                matrix_cache=actual_v_cache,
            ),
        )
        semilocal_ecp_terms = getattr(mol, "semilocal_ecp_terms", None)
        if semilocal_ecp_terms is not None:
            semilocal_blocks = semilocal_ecp_projector_blocks(
                alphas,
                centers,
                labels,
                nuclei,
                mol.z,
                dz_chk,
                semilocal_ecp_terms,
                dvr_method=dvr_method,
            )
            if semilocal_blocks is not None:
                h1_nm += semilocal_blocks
    else:
        h1_nm = np.asarray(h1_nm, float)

    h_local_ops = None if not use_cache else cache.get("h_local_ops")
    if h_local_ops is None and h1_nm is None:
        local_v_cache = {}
        h_local_ops = np.stack(
            [
                T_prim
                + V_en_sp_total_at_z(
                    alphas,
                    centers,
                    labels,
                    nuclei,
                    float(zz),
                    softcore_radii=getattr(mol, "softcore_radii", None),
                    local_ecp_terms=getattr(mol, "local_ecp_terms", None),
                    matrix_cache=local_v_cache,
                )
                for zz in mol.z
            ],
            axis=0,
        )
    eri_op = CollocatedERIOp.from_kernels(
        N=S_prim.shape[0],
        Nz=Nz,
        dz=dz_chk,
        K_h=K_h,
        Kx_h=Kx_h,
    )

    return {
        "alphas": np.asarray(alphas, float),
        "centers": np.asarray(centers, float),
        "labels": labels,
        "S_prim": S_prim,
        "T_prim": T_prim,
        "z": np.asarray(mol.z, float),
        "Kz": Kz,
        "dz": float(dz_chk),
        "K_h": K_h,
        "Kx_h": Kx_h,
        "eri_h_max": None if h_max is None else int(h_max),
        "h1_nm": h1_nm,
        "h_local_ops": h_local_ops,
        "nuclei": nuclei,
        "nh_sweep": SweepNewtonHelper(h1_nm, S_prim, eri_op),
    }


def _get_newton_context(mol: Molecule):
    if mol._newton_context is None:
        mol._newton_context = _build_newton_context(mol)
    return mol._newton_context


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
                    level_shift=0.5, shift_decay=0.75, k_ramp_iters=8,
                    dm0=None):
    N = Nz * M
    nelec = int(nelec)
    if nelec <= 0 or nelec % 2:
        raise ValueError("GDVR RHF requires a positive even number of electrons.")
    nocc = nelec // 2
    if nocc > N:
        raise ValueError(f"GDVR RHF has {nocc} occupied orbitals but only {N} basis functions.")
    jk_builder = prepare_gdvr_fock_builder(ERI_J, ERI_K, Nz, M)
    I = np.eye(N)

    if dm0 is None:
        eps, Cmo = la.eigh(Hcore)
        Cocc = Cmo[:, :nocc]
        P = 2.0 * (Cocc @ Cocc.T)
    else:
        P = np.asarray(dm0, float).copy()
        P = 0.5 * (P + P.T)
        eps, Cmo = la.eigh(Hcore)

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

        F2e = fock_2e_slice_collocated(P, jk_builder, None, Nz, M, k_scale=k_scale)
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
    d_mat = np.stack([np.asarray(C_list[m][:, 0], float) for m in range(Nz)], axis=0)
    pair_diag = np.einsum("ni,nj->nij", d_mat, d_mat, optimize=True).reshape(Nz, -1)
    ERI_J = [[0.0 for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[0.0 for _ in range(Nz)] for _ in range(Nz)]

    for h in range(Nz):
        Nh = Nz - h
        if Nh <= 0:
            break
        if K_h[h] is None or Kx_h[h] is None:
            continue

        V_right = pair_diag[h:].T
        WJ = K_h[h] @ V_right
        j_vals = np.einsum("ij,ij->i", pair_diag[:Nh], WJ.T, optimize=True)

        pair_off = np.einsum("ni,nj->nij", d_mat[:Nh], d_mat[h:], optimize=True).reshape(Nh, -1).T
        WK = Kx_h[h] @ pair_off
        k_vals = np.einsum("ij,ij->j", pair_off, WK, optimize=True)

        for m, (j_val, k_val) in enumerate(zip(j_vals, k_vals)):
            nn = m + h
            ERI_J[m][nn] = float(j_val)
            ERI_J[nn][m] = ERI_J[m][nn]
            ERI_K[m][nn] = float(k_val)
            ERI_K[nn][m] = ERI_K[m][nn]

    for m in range(Nz):
        for n in range(Nz):
            if ERI_J[m][n] < 0.0: ERI_J[m][n] = 0.0
    return ERI_J, ERI_K


def rebuild_Hcore_from_d(
    d_stack,
    z,
    Kz,
    S_prim,
    T_prim,
    alphas,
    centers,
    labels,
    nuclei,
    h_local_ops=None,
    softcore_radii=None,
    local_ecp_terms=None,
    h1_nm=None,
):
    d_stack = np.asarray(d_stack, float)
    if h1_nm is not None:
        h1_nm = np.asarray(h1_nm, float)
        Hcore = np.einsum("ni,nmij,mj->nm", d_stack, h1_nm, d_stack, optimize=True)
        return 0.5 * (Hcore + Hcore.T)

    S_scalar = d_stack @ S_prim @ d_stack.T

    if h_local_ops is None:
        h_local_ops = np.stack(
            [
                T_prim
                + V_en_sp_total_at_z(
                    alphas,
                    centers,
                    labels,
                    nuclei,
                    float(zz),
                    softcore_radii=softcore_radii,
                    local_ecp_terms=local_ecp_terms,
                )
                for zz in z
            ],
            axis=0,
        )
    e_local = np.einsum("ni,nij,nj->n", d_stack, h_local_ops, d_stack, optimize=True)

    Hcore = (Kz * S_scalar).astype(float, copy=False)
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

class SweepNewtonHelper(NewtonHelper):
    """
    Extends NewtonHelper with O(Nz) exact single-slice gradient construction 
    AND O(Nz) exact diagonal Hessian construction.
    """
    def __init__(self, h1_nm, S_prim, eri_op):
        super().__init__(h1_nm, S_prim, eri_op)
        self.eri_op = eri_op # Store explicitly

    def _active_k_range(self, n):
        hmax = getattr(self.eriop, "max_active_h", self.Nz - 1)
        if hmax < 0:
            return range(0, 0)
        n = int(n)
        hmax = int(hmax)
        return range(max(0, n - hmax), min(self.Nz, n + hmax + 1))

    def _offset_is_active(self, n, k):
        if hasattr(self.eriop, "offset_is_active"):
            return self.eriop.offset_is_active(int(n) - int(k))
        return True

    def _gradient_and_hessian_slice(self, n, d_stack, P_slice, pair_diag=None):
        """
        Compute g_n and H_nn together so shared contractions are evaluated once.
        """
        Nz = self.Nz
        N = self.N
        dn = d_stack[n]
        p_nn = P_slice[n, n]
        n2 = N * N
        if pair_diag is None:
            pair_diag = np.einsum("ni,nj->nij", d_stack, d_stack, optimize=True).reshape(Nz, n2)
        K_nm_kl = self.eriop.K_h
        K_nl_km = self.eriop.K_nl_km
        K_nk_ml = self.eriop.K_nk_ml
        K_nl_mk = self.eriop.K_nl_mk

        def _from_pair(mats, h, pair):
            mat = mats[h]
            if mat is None:
                return np.zeros((N, N), dtype=float)
            return (mat @ pair).reshape(N, N)

        def _from_vecs(mats, h, left, right):
            mat = mats[h]
            if mat is None:
                return np.zeros((N, N), dtype=float)
            pair = np.multiply.outer(left, right).reshape(n2)
            return (mat @ pair).reshape(N, N)

        J_nn = np.zeros((N, N), dtype=float)
        acc40 = np.zeros((N, N), dtype=float)
        acc41 = np.zeros((N, N), dtype=float)
        pair_nn = pair_diag[n]
        K_nn = _from_pair(K_nl_km, 0, pair_nn)
        K_alt_nn = _from_pair(K_nl_mk, 0, pair_nn)
        if abs(p_nn) > 1e-12:
            acc41 += (p_nn * p_nn) * K_nn
            acc40 -= 0.5 * (p_nn * p_nn) * K_alt_nn
            acc40 += (p_nn * p_nn) * _from_pair(K_nk_ml, 0, pair_nn)

        for k in self._active_k_range(n):
            h = abs(n - k)
            Jkk = _from_pair(K_nm_kl, h, pair_diag[k])

            p_kk = P_slice[k, k]
            if abs(p_kk) > 1e-12:
                J_nn += p_kk * Jkk

            p_nk = P_slice[n, k]
            if abs(p_nk) > 1e-12:
                acc41 -= 0.5 * (p_nk * p_nk) * Jkk

        g_n = np.zeros(N, dtype=float)
        for m in range(Nz):
            dm = d_stack[m]
            p_nm = P_slice[n, m]
            p_mn = P_slice[m, n]

            F_nm = p_nm * self.h1_nm[n, m] + p_mn * self.h1_nm[m, n]
            p_sym = 0.5 * (p_nm + p_mn)

            if abs(p_sym) > 1e-12:
                if n == m:
                    V_eff = J_nn - 0.5 * p_nn * K_nn
                elif self._offset_is_active(n, m):
                    K_val = _from_vecs(K_nl_km, abs(n - m), dm, dn)
                    V_eff = -0.5 * p_mn * K_val
                else:
                    V_eff = None
                if V_eff is not None:
                    F_nm += 2.0 * p_sym * V_eff

            g_n += F_nm @ dm

        H_nn = p_nn * self.h1_nm[n, n].copy()
        if abs(p_nn) > 1e-12:
            H_nn += p_nn * (J_nn - 0.5 * p_nn * K_nn)
        H_nn += acc40
        H_nn += acc41
        return g_n, H_nn
        
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
        for k in self._active_k_range(n):
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
                elif self._offset_is_active(n, m):
                    # Off-diagonal Exchange: (nn|mm)
                    K_val = self.eriop.block_nl__km(n, m, m, n, dm, dn)
                    V_eff = -0.5 * P_slice[m, n] * K_val
                else:
                    V_eff = None
                
                if V_eff is not None:
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
            for k in self._active_k_range(n):
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
        for k in self._active_k_range(n):
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
        for k in self._active_k_range(n):
            p_nk = P_slice[n, k]
            if abs(p_nk) > 1e-12:
                dk = d_stack[k]
                acc41 -= 0.5 * (p_nk * p_nk) * self.eriop.block_nm__kl(n, n, k, k, dk, dk)
                
        H_nn += acc41
        
        return H_nn

    def kkt_step_slice(self, n, d_stack, P_slice, S_prim, ridge_base=1e-4, pair_diag=None):
        """
        Solves the local KKT problem with automatic ridge selection for stability.
        """
        # 1-2. Compute Exact Gradient and Diagonal Hessian together to reuse contractions.
        g_n_vec, H_nn = self._gradient_and_hessian_slice(n, d_stack, P_slice, pair_diag=pair_diag)
        g_n = g_n_vec.reshape(-1, 1)
        
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
):
    """
    Performs Symmetric Gauss-Seidel optimization (Forward + Backward).
    """
    Nz = d_stack.shape[0]
    d_current = d_stack.copy()
    pair_diag = np.einsum("ni,nj->nij", d_current, d_current, optimize=True).reshape(Nz, -1)
    
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
            delta_d, lam, g_n = nh.kkt_step_slice(n, d_current, P_slice, S_prim, ridge, pair_diag=pair_diag)
            
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
            pair_diag[n] = np.multiply.outer(d_cand, d_cand).reshape(-1)
            
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
