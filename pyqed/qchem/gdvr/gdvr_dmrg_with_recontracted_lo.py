import os
import time
import pickle
import logging
import datetime
import numpy as np

# ============================================================
# GDVR / HF imports
# ============================================================
from pyqed.qchem.gdvr.gdvr_mean_field import (
    Molecule,
    sine_dvr_1d,
    make_xy_spd_primitive_basis,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
    V_en_sp_total_at_z,
    build_h1_nm,
    build_method2,
    scf_rhf_method2,
    eri_JK_from_kernels_M1,
    rebuild_Hcore_from_d,
    CollocatedERIOp,
    SweepNewtonHelper,
    sweep_optimize_driver,
)

# ============================================================
# DMRG / MPO imports
# ============================================================
from pyqed.mps import DMRG as DMRG_SOLVER, dense_to_symmetric_mpo
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
import pyqed.mps.mps as mps_lib

try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, QN
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
    QN = None

# ============================================================
# Logging, Constants / defaults
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ANG_TO_BOHR = 1.8897259886

DEFAULT_S_EXPS = np.array(
    [35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280],
    dtype=float,
)
DEFAULT_P_EXPS = np.array([], dtype=float)
DEFAULT_D_EXPS = np.array([], dtype=float)

DEFAULT_LZ = 36.0
DEFAULT_NZ = 511
DEFAULT_M = 1

DEFAULT_ALT_CYCLES = 20 # HF Transversal orbital optimization 
DEFAULT_SWEEP_ITERATIONS = 10
DEFAULT_TRUST_STEP = 1.0
DEFAULT_NEWTON_RIDGE = 0.5
DEFAULT_TRUST_RADIUS = 2.0
DEFAULT_DVR_METHOD = "sine"

DEFAULT_DMRG_SWEEPS = 30
DEFAULT_INIT_GUESS = "cid"


# ============================================================
# Symmetry helpers
# ============================================================
class SymmetryManager:
    def __init__(self, sym_list):
        if sym_list is True:
            sym_list = ["charge", "sz"]
        if sym_list is False or sym_list is None:
            sym_list = []
        self.sym_types = [s.lower() for s in sym_list]
        self.rank = len(self.sym_types)
        self.enabled = self.rank > 0

    def get_vac_qn(self):
        return QN(*[0] * self.rank)

    def get_phys_qn(self, site_idx, state_str):
        vals = []
        for sym in self.sym_types:
            if sym in ["charge", "n", "particle"]:
                vals.append(0 if state_str == "emp" else 1)
            elif sym in ["sz", "spin", "s_z"]:
                if state_str == "emp":
                    vals.append(0)
                else:
                    vals.append(1 if site_idx % 2 == 0 else -1)
        return QN(*vals)

    def get_target_qn(self, nelec, spin):
        vals = []
        for sym in self.sym_types:
            if sym in ["charge", "n", "particle"]:
                vals.append(int(nelec))
            elif sym in ["sz", "spin", "s_z"]:
                vals.append(int(spin))
        return QN(*vals)


def gen_hf_config(nelec, nsites):
    return [1] * nelec + [0] * (nsites - nelec)


def gen_cid_configs(nelec, nsites, mixing=0.1):
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)]
    if nelec >= 2 and (nsites - nelec) >= 2:
        dbl = list(hf)
        dbl[nelec - 1] = 0
        dbl[nelec - 2] = 0
        dbl[nelec] = 1
        dbl[nelec + 1] = 1
        configs.append((tuple(dbl), mixing))
    return configs


def gen_random_cisd_configs(nelec, nsites, n_states=10, mixing=0.1):
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)]

    occ_alpha = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 0]
    occ_beta = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 1]
    vir_alpha = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 0]
    vir_beta = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 1]

    for _ in range(n_states):
        new_cfg = list(hf)
        exc_types = []
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1:
            exc_types.append("S_alpha")
        if len(occ_beta) >= 1 and len(vir_beta) >= 1:
            exc_types.append("S_beta")
        if len(occ_alpha) >= 2 and len(vir_alpha) >= 2:
            exc_types.append("D_aa")
        if len(occ_beta) >= 2 and len(vir_beta) >= 2:
            exc_types.append("D_bb")
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1 and len(occ_beta) >= 1 and len(vir_beta) >= 1:
            exc_types.append("D_ab")

        if not exc_types:
            break

        choice = np.random.choice(exc_types)
        if choice == "S_alpha":
            i = np.random.choice(occ_alpha)
            a = np.random.choice(vir_alpha)
            new_cfg[i] = 0
            new_cfg[a] = 1
        elif choice == "S_beta":
            i = np.random.choice(occ_beta)
            a = np.random.choice(vir_beta)
            new_cfg[i] = 0
            new_cfg[a] = 1
        elif choice == "D_aa":
            i, j = np.random.choice(occ_alpha, 2, replace=False)
            a, b = np.random.choice(vir_alpha, 2, replace=False)
            new_cfg[i] = 0
            new_cfg[j] = 0
            new_cfg[a] = 1
            new_cfg[b] = 1
        elif choice == "D_bb":
            i, j = np.random.choice(occ_beta, 2, replace=False)
            a, b = np.random.choice(vir_beta, 2, replace=False)
            new_cfg[i] = 0
            new_cfg[j] = 0
            new_cfg[a] = 1
            new_cfg[b] = 1
        elif choice == "D_ab":
            i = np.random.choice(occ_alpha)
            a = np.random.choice(vir_alpha)
            j = np.random.choice(occ_beta)
            b = np.random.choice(vir_beta)
            new_cfg[i] = 0
            new_cfg[j] = 0
            new_cfg[a] = 1
            new_cfg[b] = 1

        configs.append((tuple(new_cfg), mixing))
    return configs


def build_mps_from_configs(configs_with_amps, sym_mgr, nsites, noise_scale=1e-5):
    if not SYMMETRY_AVAILABLE:
        raise RuntimeError("Symmetry module unavailable.")
    trajectories = []
    vac_qn = sym_mgr.get_vac_qn()

    for cfg, _ in configs_with_amps:
        curr_q = vac_qn
        traj = [curr_q]
        for site_i, occ in enumerate(cfg):
            state_str = "occ" if occ > 0 else "emp"
            phys_q = sym_mgr.get_phys_qn(site_i, state_str)
            curr_q = curr_q + phys_q
            traj.append(curr_q)
        trajectories.append(traj)

    from collections import defaultdict
    mps = []
    for i in range(nsites):
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)

        for k, _ in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i + 1]
            left_groups[qL].append(k)
            right_groups[qR].append(k)

        data = {}
        for k, (cfg, amp) in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i + 1]
            state_str = "occ" if cfg[i] > 0 else "emp"
            qP = sym_mgr.get_phys_qn(i, state_str)
            key = (qL, qR, qP)

            row = 0 if i == 0 else left_groups[qL].index(k)
            col = 0 if i == nsites - 1 else right_groups[qR].index(k)

            if key not in data:
                dL = 1 if i == 0 else len(left_groups[qL])
                dR = 1 if i == nsites - 1 else len(right_groups[qR])
                data[key] = np.zeros((dL, dR, 1), dtype=complex)

            val = amp if i == 0 else 1.0
            noise = (np.random.rand() - 0.5) * noise_scale
            data[key][row, col, 0] += val + noise

        final_qns_L = [trajectories[0][0]] if i == 0 else [q for q in sorted(left_groups.keys()) for _ in left_groups[q]]
        final_qns_R = [trajectories[0][-1]] if i == nsites - 1 else [q for q in sorted(right_groups.keys()) for _ in right_groups[q]]
        final_qns_P = [sym_mgr.get_phys_qn(i, "emp"), sym_mgr.get_phys_qn(i, "occ")]

        bt = BlockTensor(data, [final_qns_L, final_qns_R, final_qns_P], [-1, 1, 1])
        nrm = bt.norm()
        if nrm > 1e-12:
            bt = bt * (1.0 / nrm)
        mps.append(bt)
    return mps


def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    d = 2
    mps_guess = []
    filled_count = 0
    for _ in range(n_spin):
        vec = np.zeros((1, d, 1))
        if filled_count < n_elec:
            vec[0, 1, 0] = 1.0
            filled_count += 1
        else:
            vec[0, 0, 0] = 1.0
        vec += (np.random.rand(1, d, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess


def make_initial_guess(init_guess, nelec, ncas, abelian_symmetry):
    init_guess = init_guess.lower()
    nsites = 2 * ncas

    if abelian_symmetry and SYMMETRY_AVAILABLE and init_guess in ["hf", "cid", "cisd", "random"]:
        sym_mgr = SymmetryManager(["charge", "sz"])
        if init_guess == "hf":
            configs = [(tuple(gen_hf_config(nelec, nsites)), 1.0)]
        elif init_guess == "cid":
            configs = gen_cid_configs(nelec, nsites, mixing=0.5)
        else:
            configs = gen_random_cisd_configs(nelec, nsites, n_states=20, mixing=0.1)
        return build_mps_from_configs(configs, sym_mgr, nsites), sym_mgr, True

    return get_noisy_hf_guess(nelec, nsites, noise=1e-3), None, False


# ============================================================
# Fermionic Hamiltonian helpers
# ============================================================
def get_jw_term_robust(op_str_list, indices, factor):
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n - i - 1):
            if chain[j][0] > chain[j + 1][0]:
                chain[j], chain[j + 1] = chain[j + 1], chain[j]
                swaps += 1

    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        if k > 0:
            prev_site = sorted_indices[k - 1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)


def build_general_rotated_ham_terms(h1, g2, thresh=1e-10):
    ncas = h1.shape[0]
    ham_terms = []

    for p in range(ncas):
        for q in range(ncas):
            val = h1[p, q]
            if abs(val) < thresh:
                continue
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2 * p, 2 * q], val))
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2 * p + 1, 2 * q + 1], val))

    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    val = 0.5 * g2[p, q, r, s]
                    if abs(val) < thresh:
                        continue

                    if p != r and s != q:
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2 * p, 2 * r, 2 * s, 2 * q], val
                        ))
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2 * p + 1, 2 * r + 1, 2 * s + 1, 2 * q + 1], val
                        ))

                    ham_terms.append(get_jw_term_robust(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2 * p, 2 * r + 1, 2 * s + 1, 2 * q], val
                    ))
                    ham_terms.append(get_jw_term_robust(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2 * p + 1, 2 * r, 2 * s, 2 * q + 1], val
                    ))
    return ham_terms


# ============================================================
# NO / localization helpers
# ============================================================
def extract_gamma_z_from_density(P, Nz, M=1):
    P = np.asarray(P)

    if P.shape == (Nz, Nz):
        gamma_z = P.copy()
    elif P.ndim == 4 and P.shape == (Nz, M, Nz, M):
        gamma_z = np.einsum("zmz'm->zz'", P)
    elif P.ndim == 4 and P.shape[1] == 1 and P.shape[3] == 1:
        gamma_z = P[:, 0, :, 0].copy()
    else:
        raise ValueError(f"Unsupported density matrix shape for gamma_z extraction: {P.shape}")

    gamma_z = 0.5 * (gamma_z + gamma_z.conj().T)
    return gamma_z


def natural_orbitals_z_from_gamma(gamma_z, z_grid, sort_by_occup=True):
    gamma_z = np.asarray(gamma_z)
    z_grid = np.asarray(z_grid).real

    gamma_z = 0.5 * (gamma_z + gamma_z.conj().T)
    occs, U = np.linalg.eigh(gamma_z)

    if sort_by_occup:
        order = np.argsort(occs)[::-1]
        occs = occs[order]
        U = U[:, order]

    U = U.astype(np.complex128, copy=False)
    for i in range(U.shape[1]):
        idx = np.argmax(np.abs(U[:, i]))
        if np.real(U[idx, i]) < 0:
            U[:, i] *= -1.0

    prob = np.abs(U) ** 2
    centers = np.einsum("zi,z->i", prob, z_grid)
    z2 = np.einsum("zi,z->i", prob, z_grid**2)
    spreads = np.sqrt(np.maximum(z2 - centers**2, 0.0))
    ipr = np.sum(prob**2, axis=0)
    return occs, U, centers, spreads, ipr


def choose_compressed_no_subspace(occs, U, occ_cut=1e-8, max_orbs=None, min_orbs=1, force_exact=False):
    occs = np.asarray(occs)

    if force_exact and max_orbs is not None:
        n_keep = min(max_orbs, len(occs))
        keep = np.arange(n_keep)
    else:
        keep = np.where(occs > occ_cut)[0]
        if len(keep) < min_orbs:
            keep = np.arange(min(min_orbs, len(occs)))
        if max_orbs is not None:
            keep = keep[:max_orbs]

    return occs[keep], U[:, keep], keep


def localize_compressed_subspace_by_z(U_no, z_grid, occs=None):
    z_grid = np.asarray(z_grid).real
    U_no = np.asarray(U_no, dtype=np.complex128)

    Z = np.diag(z_grid)
    Z_proj = U_no.conj().T @ Z @ U_no
    Z_proj = 0.5 * (Z_proj + Z_proj.conj().T)

    z_centers, V = np.linalg.eigh(Z_proj)
    U_loc = U_no @ V

    for i in range(U_loc.shape[1]):
        idx = np.argmax(np.abs(U_loc[:, i]))
        if np.real(U_loc[idx, i]) < 0:
            U_loc[:, i] *= -1.0

    order = np.argsort(z_centers)
    z_centers = z_centers[order]
    U_loc = U_loc[:, order]
    V = V[:, order]

    prob = np.abs(U_loc) ** 2
    z2 = np.einsum("zi,z->i", prob, z_grid**2)
    spreads = np.sqrt(np.maximum(z2 - z_centers**2, 0.0))
    ipr = np.sum(prob**2, axis=0)

    occs_loc = None
    if occs is not None:
        occs = np.asarray(occs)
        nmat_no = np.diag(occs)
        nmat_loc = V.conj().T @ nmat_no @ V
        occs_loc = np.real(np.diag(nmat_loc))

    return U_loc, z_centers, spreads, ipr, occs_loc, Z_proj


def transform_one_body(h_old, W):
    h_new = W.conj().T @ h_old @ W
    h_new = 0.5 * (h_new + h_new.conj().T)
    return np.real_if_close(h_new)


def transform_density_density_kernel(V_old, W):
    g_new = np.einsum("ip,iq,ik,kr,ks->pqrs", W.conj(), W, V_old, W.conj(), W, optimize=True)
    return np.real_if_close(g_new)


# ============================================================
# Stage 1: HF + transverse optimization
# ============================================================
def run_hf_and_optimize(
    mol,
    outdir,
    Lz=DEFAULT_LZ,
    Nz=DEFAULT_NZ,
    M=DEFAULT_M,
    s_exps=DEFAULT_S_EXPS,
    p_exps=DEFAULT_P_EXPS,
    d_exps=DEFAULT_D_EXPS,
    alt_cycles=DEFAULT_ALT_CYCLES,
    sweep_iterations=DEFAULT_SWEEP_ITERATIONS,
    trust_step=DEFAULT_TRUST_STEP,
    newton_ridge=DEFAULT_NEWTON_RIDGE,
    trust_radius=DEFAULT_TRUST_RADIUS,
    dvr_method=DEFAULT_DVR_METHOD,
):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    nuclei = mol.to_tuples()
    enuc = mol.nuclear_repulsion_energy()

    logger.info("=" * 80)
    logger.info("[HF/OPT] Starting single-geometry workflow")
    logger.info(f"[HF/OPT] nelec = {mol.nelec}, spin = {mol.spin}, Enuc = {enuc:.12f}")
    logger.info(f"[HF/OPT] output_dir = {outdir}")
    logger.info("=" * 80)

    Hcore, z_grid, dz, E_slices, C_list, _, _, _ = build_method2(
        mol,
        Lz=Lz,
        Nz=Nz,
        M=M,
        s_exps=s_exps,
        p_exps=p_exps,
        d_exps=d_exps,
        verbose=False,
        dvr_method=dvr_method,
    )

    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei,
        exps_s=s_exps,
        exps_p=p_exps,
        exps_d=d_exps,
    )
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    n_ao = len(alphas)

    K_h = []
    Kx_h = []
    for h in range(Nz):
        dz_val = h * dz
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
        K_h.append(eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao))
        Kx_h.append(eri_tensor.transpose(0, 2, 1, 3).reshape(n_ao * n_ao, n_ao * n_ao))

    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
    Etot, eps, Cmo, P, info = scf_rhf_method2(
        Hcore, ERI_J, ERI_K, Nz, M,
        nelec=mol.nelec,
        Enuc=enuc,
        conv=1e-5,
        max_iter=100,
        verbose=False,
    )
    logger.info(f"[HF/OPT] Initial HF energy = {Etot:.12f}")

    energy_log = {"hf_initial": float(Etot), "hf_pre_opt": []}

    d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz)])

    _, Kz, _ = sine_dvr_1d(-Lz, Lz, Nz)
    h1_nm = build_h1_nm(
        Kz, S_prim, T_prim, z_grid,
        lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz),
    )
    ERIop = CollocatedERIOp.from_kernels(N=S_prim.shape[0], Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    nh_sweep = SweepNewtonHelper(h1_nm, S_prim, ERIop)

    for n in range(Nz):
        dn = d_stack[n]
        d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

    for cyc in range(1, alt_cycles + 1):
        P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()

        d_stack = sweep_optimize_driver(
            nh_sweep,
            d_stack,
            P_slice,
            S_prim,
            n_cycles=sweep_iterations,
            ridge=newton_ridge,
            trust_step=trust_step,
            trust_radius=trust_radius,
            verbose=False,
        )

        C_list = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
        Hcore = rebuild_Hcore_from_d(
            d_stack, z_grid, Kz, S_prim, T_prim,
            alphas, centers, labels, nuclei,
        )
        ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)

        Etot, eps, Cmo, P, info = scf_rhf_method2(
            Hcore, ERI_J, ERI_K, Nz, M,
            nelec=mol.nelec,
            Enuc=enuc,
            conv=1e-8,
            max_iter=60,
            verbose=False,
        )
        energy_log["hf_pre_opt"].append(float(Etot))
        logger.info(f"[HF/OPT] cycle {cyc:02d}: E = {Etot:.12f}")

    np.savez_compressed(
        os.path.join(outdir, "hf_opt_data.npz"),
        Hcore_curr=Hcore,
        V_coul=np.array(ERI_J),
        P=P,
        Cmo=Cmo,
        Enuc=enuc,
        z_grid=z_grid,
        d_stack=d_stack,
        alphas=alphas,
        centers=centers,
        labels_serialized=np.array(
            [{"kind": l.kind, "dim": l.dim, "l": l.l, "role": l.role} for l in labels],
            dtype=object
        ),
        mol_coords=mol.coords,
        mol_charges=mol.charges,
        nelec=mol.nelec,
        spin=mol.spin,
        Lz=Lz,
        Nz=Nz,
        M=M,
        s_exps=s_exps,
        p_exps=p_exps,
        d_exps=d_exps,
    )

    with open(os.path.join(outdir, "hf_opt_summary.txt"), "w") as f:
        f.write(f"E_hf_opt    {Etot:.12f}\n")
        f.write(f"Enuc        {enuc:.12f}\n")
        f.write(f"elapsed_sec {time.time() - t0:.4f}\n")

    logger.info(f"[HF/OPT] finished, E = {Etot:.12f}")

    return {
        "E_hf_opt": float(Etot),
        "Hcore_curr": Hcore,
        "V_coul": np.array(ERI_J),
        "P": P,
        "Cmo": Cmo,
        "Enuc": float(enuc),
        "z_grid": z_grid,
        "d_stack": d_stack,
        "alphas": alphas,
        "centers": centers,
        "labels": labels,
    }


# ============================================================
# Stage 2: recontract + DMRG
# ============================================================

def rhf_energy_in_compressed_basis(h, g, C_occ, Enuc):
    D = 2.0 * (C_occ @ C_occ.T.conj())
    E1 = np.einsum("pq,pq->", h, D).real
    J = np.einsum("pqrs,rs->pq", g, D).real
    K = np.einsum("prqs,rs->pq", g, D).real
    E2 = 0.5 * np.einsum("pq,pq->", D, (J - 0.5 * K)).real
    return E1 + E2 + Enuc

def run_dmrg_from_hf_data(
    hf_data,
    outdir,
    nkeep=14,
    D=200,
    dmrg_sweeps=DEFAULT_DMRG_SWEEPS,
    z_occ_cut=1e-10,
    init_guess=DEFAULT_INIT_GUESS,
):
    os.makedirs(outdir, exist_ok=True)



    Hcore_curr = np.array(hf_data["Hcore_curr"])
    V_coul = np.array(hf_data["V_coul"])
    P = np.array(hf_data["P"])
    Enuc = float(hf_data["Enuc"])
    z_grid = np.array(hf_data["z_grid"])
    nelec = int(np.round(np.trace(P).real))

    gamma_z = extract_gamma_z_from_density(P, len(z_grid), M=1)
    occs_no, U_no, centers_no, spreads_no, ipr_no = natural_orbitals_z_from_gamma(
        gamma_z, z_grid=z_grid, sort_by_occup=True
    )

    occs_sub, U_sub, keep_idx = choose_compressed_no_subspace(
        occs_no, U_no,
        occ_cut=z_occ_cut,
        max_orbs=nkeep,
        min_orbs=1,
        force_exact=True,
    )

    U_loc, centers_loc, spreads_loc, ipr_loc, occs_loc, Z_proj = localize_compressed_subspace_by_z(
        U_sub, z_grid=z_grid, occs=occs_sub
    )

    gamma_loc = U_loc.conj().T @ gamma_z @ U_loc
    gamma_loc = 0.5 * (gamma_loc + gamma_loc.conj().T)

    occs_hf_loc, C_hf_loc = np.linalg.eigh(gamma_loc)
    order = np.argsort(occs_hf_loc)[::-1]
    occs_hf_loc = occs_hf_loc[order]
    C_hf_loc = C_hf_loc[:, order]

    nocc_spatial = nelec // 2
    C_occ_loc = C_hf_loc[:, :nocc_spatial]

    h_loc = transform_one_body(Hcore_curr, U_loc)
    g_loc = transform_density_density_kernel(V_coul, U_loc)

    np.savez_compressed(
        os.path.join(outdir, "prep_keep_data.npz"),
        h_loc=h_loc,
        g_loc=g_loc,
        W_loc=U_loc,
        centers_loc=centers_loc,
        spreads_loc=spreads_loc,
        ipr_loc=ipr_loc,
        occs_loc=occs_loc if occs_loc is not None else np.array([]),
        occs_sub=occs_sub,
        keep_idx=keep_idx,
        gamma_z=gamma_z,
        gamma_loc=gamma_loc,
        occs_hf_loc=occs_hf_loc,
        C_occ_loc=C_occ_loc,
        Enuc=Enuc,
        nelec=nelec,
        spin=0,
        nkeep=nkeep,
    )

    print("h_loc shape =", h_loc.shape)
    print("g_loc shape =", g_loc.shape)
    print("||h_loc|| =", np.linalg.norm(h_loc))
    print("||g_loc|| =", np.linalg.norm(g_loc.reshape(-1)))
    print("max |h_loc| =", np.max(np.abs(h_loc)))
    print("max |g_loc| =", np.max(np.abs(g_loc)))
    print("trace h_loc =", np.trace(h_loc))

    print("lowest 10 occs_no =", occs_no[:10])
    print("kept occs_sub =", occs_sub)
    print("centers_loc =", centers_loc)

    E_hf_rebuilt = rhf_energy_in_compressed_basis(h_loc, g_loc, C_occ_loc, Enuc)
    print("HF energy rebuilt in kept/localized basis =", E_hf_rebuilt)

    ham_terms = build_general_rotated_ham_terms(h_loc, g_loc, thresh=1e-10)
    logger.info(f"[DMRG] Input operator terms: {len(ham_terms)}")

    basis = [BasisSimpleElectron(i) for i in range(2 * nkeep)]
    model = Model(basis=basis, ham_terms=ham_terms)
    mpo = Mpo(model, algo="qr")

    mpo_dmrg = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
    for w in mpo_dmrg:
        w[np.abs(w) < 1e-10] = 0.0

    abelian_symmetry = True
    if abelian_symmetry:
        sym_mgr = SymmetryManager(["charge", "sz"])
        site_qn_maps = []
        for i in range(2 * nkeep):
            q_emp = sym_mgr.get_phys_qn(i, "emp")
            q_occ = sym_mgr.get_phys_qn(i, "occ")
            site_qn_maps.append({0: q_emp, 1: q_occ})
        final_H = dense_to_symmetric_mpo(mpo_dmrg, site_qn_maps)
    else:
        final_H = mpo_dmrg
        sym_mgr = None

    mps0, sym_mgr_guess, _ = make_initial_guess(
        init_guess=init_guess,
        nelec=nelec,
        ncas=nkeep,
        abelian_symmetry=abelian_symmetry,
    )
    if sym_mgr_guess is not None:
        sym_mgr = sym_mgr_guess

    solver = DMRG_SOLVER(
        final_H,
        D=D,
        nsweeps=dmrg_sweeps,
        init_guess=mps0,
        symmetry=abelian_symmetry,
        target_qn=sym_mgr.get_target_qn(nelec, 0) if sym_mgr is not None else None,
        sym_mgr=sym_mgr,
        charge=nelec,
        spin=0,
        not_conv_err=False,
    )
    solver.run()

    try:
        psi_tensors = solver.ground_state.Bs
        e_elec = mps_lib.expect_mps(psi_tensors, solver.H, psi_tensors)
        E_dmrg = np.real(e_elec) + Enuc
    except Exception as e:
        logger.info(f"[DMRG] expect_mps failed, fallback to solver.e_tot. Error: {e}")
        E_dmrg = solver.e_tot + Enuc

    with open(os.path.join(outdir, "dmrg_ground_state.pkl"), "wb") as f:
        pickle.dump(
            {
                "Bs": solver.ground_state.Bs,
                "nkeep": nkeep,
                "charge": nelec,
                "spin": 0,
                "D": D,
                "dmrg_sweeps": dmrg_sweeps,
                "init_guess": init_guess,
                "abelian_symmetry": abelian_symmetry,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    np.savez_compressed(
        os.path.join(outdir, "dmrg_result.npz"),
        E_dmrg=E_dmrg,
        D=D,
        dmrg_sweeps=dmrg_sweeps,
        nkeep=nkeep,
        init_guess=init_guess,
        state_file="dmrg_ground_state.pkl",
        prep_file="prep_keep_data.npz",
    )

    with open(os.path.join(outdir, "dmrg_summary.txt"), "w") as f:
        f.write(f"nkeep       {nkeep}\n")
        f.write(f"D           {D}\n")
        f.write(f"dmrg_sweeps {dmrg_sweeps}\n")
        f.write(f"init_guess  {init_guess}\n")
        f.write(f"E_dmrg      {E_dmrg:.12f}\n")

    logger.info(f"[DMRG] finished, E = {E_dmrg:.12f}")
    return float(E_dmrg)


# ============================================================
# Single-geometry pipeline
# ============================================================
def run_single_geometry_pipeline(
    mol,
    run_root,
    hf_nkeep=14,
    hf_D=200,
    dmrg_sweeps=30,
    init_guess="cid",
    Lz=DEFAULT_LZ,
    Nz=DEFAULT_NZ,
    M=DEFAULT_M,
    s_exps=DEFAULT_S_EXPS,
    p_exps=DEFAULT_P_EXPS,
    d_exps=DEFAULT_D_EXPS,
):
    os.makedirs(run_root, exist_ok=True)

    hf_dir = os.path.join(run_root, "01_hf_opt")
    dmrg_dir = os.path.join(run_root, f"02_dmrg_keep_{hf_nkeep}_D_{hf_D}")

    hf_data = run_hf_and_optimize(
        mol=mol,
        outdir=hf_dir,
        Lz=Lz,
        Nz=Nz,
        M=M,
        s_exps=s_exps,
        p_exps=p_exps,
        d_exps=d_exps,
    )

    E_dmrg = run_dmrg_from_hf_data(
        hf_data=hf_data,
        outdir=dmrg_dir,
        nkeep=hf_nkeep,
        D=hf_D,
        dmrg_sweeps=dmrg_sweeps,
        init_guess=init_guess,
    )

    with open(os.path.join(run_root, "final_summary.txt"), "w") as f:
        f.write(f"E_hf_opt {hf_data['E_hf_opt']:.12f}\n")
        f.write(f"E_dmrg   {E_dmrg:.12f}\n")

    logger.info("=" * 80)
    logger.info(f"[DONE] HF-opt energy = {hf_data['E_hf_opt']:.12f}")
    logger.info(f"[DONE] DMRG energy   = {E_dmrg:.12f}")
    logger.info(f"[DONE] Output folder = {run_root}")
    logger.info("=" * 80)



# edits geometry here in the main
def main():

    # define the molecule to be calculated
    charges = [1.0]*4
    coords = [[[0.0, 0.0, -3.6], [0.0, 0.0, -0.91], [0.0, 0.0, 0.91], [0.0, 0.0, 3.6]]]

    mol = Molecule(charges, coords, nelec=4, spin = 0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = f"single_geom_run_{timestamp}"

    run_single_geometry_pipeline(
        mol=mol,
        run_root=run_root,
        hf_nkeep=4,
        hf_D=200,
        dmrg_sweeps=30,
        init_guess="cid",
        Lz=8.0,
        Nz=255,
        M=1,
        s_exps=DEFAULT_S_EXPS, # default is 6 gaussians in sto-6g
        p_exps=DEFAULT_P_EXPS, # default is None
        d_exps=DEFAULT_D_EXPS, # default is None
    )


if __name__ == "__main__":
    main()