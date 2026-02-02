import numpy as np
import pickle
import os
import logging
import time

from pyqed.qchem.gdvr.gdvr_mean_field import (
    Molecule, build_method2, make_xy_spd_primitive_basis, 
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,
    scf_rhf_method2, sine_dvr_1d, eri_JK_from_kernels_M1,
    build_h1_nm, V_en_sp_total_at_z, CollocatedERIOp, rebuild_Hcore_from_d,
    SweepNewtonHelper, sweep_optimize_driver
)
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
import pyqed.mps.mps as mps_lib

import gdvr_dmrg_scf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



def get_jw_term_robust(op_str_list, indices, factor):
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
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
            prev_site = sorted_indices[k-1]
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

def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    d = 2; mps_guess = []
    filled_count = 0
    for i in range(n_spin):
        vec = np.zeros((d, 1, 1))
        if filled_count < n_elec: 
            vec[1, 0, 0] = 1.0; filled_count += 1
        else: 
            vec[0, 0, 0] = 1.0
        vec += (np.random.rand(d, 1, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess

def align_orbital_phases(d_old, d_new, S_prim):
    Nz = d_old.shape[0]
    min_overlap = 1.0
    for n in range(Nz):
        overlap = float(d_old[n].T @ S_prim @ d_new[n])
        if overlap < 0:
            d_new[n] *= -1.0
            overlap = -overlap
        min_overlap = min(min_overlap, overlap)
    return d_new, min_overlap

# helpers for calculating <HF|DMRG>
def apply_mpo_to_mps(mpo_tensors, mps_tensors):
    """
    Contracts an MPO into an MPS: |Psi_new> = W |Psi_old>
    """
    new_mps = []
    L = len(mps_tensors)
    
    for i in range(L):
        M = mps_tensors[i]         # (d_in, D_ML, D_MR)
        W = mpo_tensors[i]         # (D_WL, D_WR, d_out, d_in)
        
        # 1. Contract Physical Indices: W(d_in) with M(d_in)
        T = np.tensordot(W, M, axes=([3], [0]))
        
        # 2. Transpose to Target MPS Layout: (d_out, D_WL, D_ML, D_WR, D_MR)
        T = T.transpose(2, 0, 3, 1, 4)
        
        # 3. Fuse Bonds
        d_out = T.shape[0]
        new_DL = T.shape[1] * T.shape[2]
        new_DR = T.shape[3] * T.shape[4]
        
        new_M = T.reshape(d_out, new_DL, new_DR)
        new_mps.append(new_M)
        
    return new_mps

def build_annihilation_mpo(coeff_vector):
    """
    Builds MPO for operator O = sum_i c_i a_i.
    Shape: (Left, Right, Phys_Out, Phys_In)
    Bond dim 2.
    """
    L = len(coeff_vector)
    tensors = []
    
    # Matrices
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    a = np.array([[0, 1], [0, 0]]) # Annihilation: |0><1|
    
    for i in range(L):
        c = coeff_vector[i]
        # Shape (Left, Right, Phys_Out, Phys_In)
        W = np.zeros((2, 2, 2, 2))
        
        W[0, 0] = Z
        W[0, 1] = c * a
        W[1, 1] = I
        
        # Boundary Conditions
        if i == 0: W = W[0:1, :, :, :] # Start State 0
        if i == L-1: W = W[:, 1:2, :, :] # End State 1
            
        tensors.append(W)
        
    return tensors

# helper for overlap calculation <HF|DMRG>
def calculate_overlap_with_hf_robust(mps_tensors, C_mo_spatial, occupied_indices, n_spatial):
    """
    S = < Vac | c_N ... c_1 | Psi_DMRG >
    """
    # 1. Convert MOs to Spin-Orbital Annihilation Vectors
    ops = []
    for mo_idx in occupied_indices:
        vec = C_mo_spatial[:, mo_idx]
        
        # Down Spin (Odd sites)
        op_down = np.zeros(2 * n_spatial)
        op_down[1::2] = vec
        ops.append(op_down)
        
        # Up Spin (Even sites)
        op_up = np.zeros(2 * n_spatial)
        op_up[0::2] = vec
        ops.append(op_up)
    
    # 2. Sequential Application: <HF| = <0| c_N ... c_1
    # We apply c_1, then c_2 ... then c_N to |Psi>
    # The list 'ops' is [c_1, c_2, ... c_N]
    
    curr_mps = [t.copy() for t in mps_tensors]
    # print(f"  -> Calculating Overlap (Robust). Initial Bond Dim: {curr_mps[len(curr_mps)//2].shape}")
    
    # Apply operators sequentially
    for k, op_vec in enumerate(ops):
        mpo = build_annihilation_mpo(op_vec)
        curr_mps = apply_mpo_to_mps(mpo, curr_mps)
        
        # Optional: Compress intermediate MPS to avoid exponential blowup
        # But for overlap calculation, we might want to be careful with compression
        # If bond dims get too large (>2000), uncomment the next lines:
        # if k % 2 == 0: 
        #    curr_mps = compress_mps_simple(curr_mps, max_bond=500)
        
    # 3. Contract with Vacuum <0|
    # MPS is now (d, DL, DR). Vacuum is physical index 0.
    # We contract the "0-th" slice of every tensor.
    
    val = np.array([[1.0]]) # Start scalar (1x1)
    
    for M in curr_mps:
        # M is (d, DL, DR).
        # We need the vacuum component: M[0, :, :]
        # Check M shape to be safe
        if M.shape[0] < 1: 
             # Should not happen for standard basis
             val = val * 0.0; break

        mat = M[0, :, :] 
        val = val @ mat 
        
    overlap = val.flatten()[0]
    return overlap

# helpers for saving wavefunction data at checkpoints
def save_checkpoint(stage_name, d_stack, mps_tensors, energy_dict, mol, params):
    folder = "Checkpoints_withou_U1"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{stage_name}"
    print(f"  [Save] Creating Checkpoint: {filename} ...")
    
    np.savez_compressed(f"{filename}_orbitals.npz", d_stack=d_stack)
    if mps_tensors is not None:
        np.savez_compressed(f"{filename}_mps.npz", *mps_tensors)
    
    meta = {
        "mol_coords": mol.coords, "mol_charges": mol.charges, "nelec": mol.nelec,
        "energy_log": energy_dict, "params": params, "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }
    with open(f"{filename}_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  [Save] Done.")

# main loop
def run_gdvr_dmrg_loop(
    mol, Lz, Nz, basis_cfg,
    pre_opt_cycles=10,      
    dmrg_cycles=3,          
    dmrg_bond_dim=20,
    dmrg_sweeps=10,
    post_dmrg_opt_cycles=5
):
    print("="*60)
    print(f"GDVR-DMRG Loop (Robust Overlap)")
    print(f"System: {mol.nelec} electrons, Nz={Nz}, Lz={Lz}")
    print("="*60)
    
    energy_log = {
        "hf_initial": None, "hf_pre_opt": [],
        "dmrg_cycles": [], "final_overlap": None
    }
    run_params = {"Lz": Lz, "Nz": Nz, "basis": basis_cfg, "bond_dim": dmrg_bond_dim}

    # --- Phase A: Initial HF ---
    s_exps = basis_cfg.get('s'); p_exps = basis_cfg.get('p', []); d_exps = basis_cfg.get('d', [])
    Hcore, z, dz, E_slices, C_list, _, _, _ = build_method2(
        mol, Lz=Lz, Nz=Nz, M=1, s_exps=s_exps, p_exps=p_exps, d_exps=d_exps, 
        verbose=False, dvr_method='sine'
    )
    
    nuclei = mol.to_tuples()
    alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, p_exps, d_exps)
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    n_ao_2d = len(alphas)
    
    K_h = []; Kx_h = []
    for h in range(Nz):
        dz_val = h * dz
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
        n2 = n_ao_2d * n_ao_2d
        K_h.append(eri_tensor.reshape(n2, n2))
        Kx_h.append(eri_tensor.transpose(0, 2, 1, 3).reshape(n2, n2))

    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
    Enuc = mol.nuclear_repulsion_energy()
    Etot, _, Cmo, P, _ = scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz, 1, mol.nelec, Enuc, verbose=False)
    
    print(f"  -> Initial HF Energy: {Etot:.8f} Ha")
    energy_log["hf_initial"] = Etot
    
    # Checkpoint 1: HF Only
    d_stack = np.vstack([C_list[n][:, 0] for n in range(Nz)])
    save_checkpoint("01_HF_Initial", d_stack, None, energy_log, mol, run_params)
    
    _, Kz_grid, _ = sine_dvr_1d(-Lz, Lz, Nz)
    ERIop = CollocatedERIOp.from_kernels(N=n_ao_2d, Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    h1_nm_func = build_h1_nm(Kz_grid, S_prim, T_prim, z, 
                             lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz))

    # --- Phase A.5: Pre-Optimization ---
    if pre_opt_cycles > 0:
        print(f"\n[Phase A.5] Pre-optimizing AOs (HF level)...")
        nh_sweep = SweepNewtonHelper(h1_nm_func, S_prim, ERIop)
        for pcyc in range(pre_opt_cycles):
            P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
            d_stack = sweep_optimize_driver(
                nh_sweep, d_stack, P_slice, S_prim,
                n_cycles=5, ridge=0.5, trust_step=1.0, trust_radius=2.0, verbose=False
            )
            Hcore_curr = rebuild_Hcore_from_d(d_stack, z, Kz_grid, S_prim, T_prim, alphas, centers, labels, nuclei)
            C_list_curr = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
            Etot, _, Cmo, P, _ = scf_rhf_method2(Hcore_curr, ERI_J, ERI_K, Nz, 1, mol.nelec, Enuc, verbose=False)
            energy_log["hf_pre_opt"].append(Etot)
            if (pcyc + 1) % 2 == 0: print(f"   Cycle {pcyc+1}: HF Energy = {Etot:.8f} Ha")

    # Checkpoint 2: HF + Newton
    save_checkpoint("02_HF_NewtonOpt", d_stack, None, energy_log, mol, run_params)

    # --- Phase B: Self-Consistent Loop ---
    last_mps_tensors = None 
    d_stack_old = d_stack.copy()
    final_Cmo = None
    
    for cycle in range(dmrg_cycles):
        print(f"\n[Macro Cycle {cycle+1}/{dmrg_cycles}]")
        d_stack, match_quality = align_orbital_phases(d_stack_old, d_stack, S_prim)
        d_stack_old = d_stack.copy()
        
        if match_quality < 0.5:
            print(f"  [Warning] Orbitals changed significantly. Resetting MPS.")
            last_mps_tensors = None
        
        print("  1. Rebuilding Hamiltonian...")
        Hcore_curr = rebuild_Hcore_from_d(d_stack, z, Kz_grid, S_prim, T_prim, alphas, centers, labels, nuclei)
        C_list_curr = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
        V_coul, V_exch = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
        V_coul = np.array(V_coul) 
        
        print("  2. Constructing MPO...")
        ham_terms = []
        n_spin = 2 * Nz
        cutoff = 1e-10
        
        rows, cols = np.nonzero(np.abs(Hcore_curr) > cutoff)
        for i, j in zip(rows, cols):
            val = Hcore_curr[i, j]
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i, 2*j], val))
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i+1, 2*j+1], val))
        
        rows, cols = np.nonzero(np.abs(V_coul) > cutoff)
        for i, k in zip(rows, cols):
            if i == k: 
                val = V_coul[i, k]
                ham_terms.append(Op("n", 2*i) * Op("n", 2*i+1) * val)
            else: 
                val = 0.5 * V_coul[i, k]
                ham_terms.append(Op("n", 2*i) * Op("n", 2*k) * val)     
                ham_terms.append(Op("n", 2*i+1) * Op("n", 2*k+1) * val) 
                ham_terms.append(Op("n", 2*i) * Op("n", 2*k+1) * val)   
                ham_terms.append(Op("n", 2*i+1) * Op("n", 2*k) * val)   
        
        basis = [BasisSimpleElectron(i) for i in range(n_spin)]
        model = Model(basis=basis, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")
        mpo_dmrg = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        
        print(f"  3. Running DMRG (D={dmrg_bond_dim})...")
        if last_mps_tensors is None:
            mps_guess = get_noisy_hf_guess(mol.nelec, n_spin, noise=1e-3)
        else:
            mps_guess = [t.copy() for t in last_mps_tensors]
        
        solver = mps_lib.DMRG(mpo_dmrg, D=dmrg_bond_dim, nsweeps=dmrg_sweeps, init_guess=mps_guess, not_conv_err=False)
        solver.run()
        
        try:
            psi_tensors = solver.ground_state.Bs
            e_elec = mps_lib.expect_mps(psi_tensors, solver.H, psi_tensors)
            e_dmrg = np.real(e_elec) + Enuc
        except:
            e_dmrg = solver.e_tot + Enuc
            
        last_mps_tensors = solver.ground_state.Bs
        print(f"     -> Final Cycle Energy: {e_dmrg:.8f} Ha")
        
        # Checkpoint 3: First DMRG
        if cycle == 0:
            save_checkpoint("03_DMRG_FirstIter", d_stack, last_mps_tensors, energy_log, mol, run_params)

        # 4. Post-DMRG Optimization
        if cycle < dmrg_cycles - 1: 
            print("  4. Re-optimizing AOs using DMRG 1-RDM...")
            d_stack = gdvr_dmrg_scf.dmrg_ao_optimization_step(
                mol, d_stack, None, S_prim, ERIop, h1_nm_func, 
                z, Kz_grid, T_prim, alphas, centers, labels, K_h, Kx_h, 
                solver=solver, Enuc=Enuc, n_cycles=post_dmrg_opt_cycles, verbose=True
            )
            energy_log["dmrg_cycles"].append({"cycle": cycle, "e_dmrg": e_dmrg, "ao_opt": True})
        else:
            energy_log["dmrg_cycles"].append({"cycle": cycle, "e_dmrg": e_dmrg, "ao_opt": False})
            print("  4. Calculating final RHF solution for Overlap analysis...")
            ERI_J_fin, ERI_K_fin = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
            _, _, final_Cmo, _, _ = scf_rhf_method2(Hcore_curr, ERI_J_fin, ERI_K_fin, Nz, 1, mol.nelec, Enuc, verbose=False)

    # --- Robust Overlap Calculation ---
    final_overlap = None
    if final_Cmo is not None and last_mps_tensors is not None:
        print("\n" + "-"*60)
        print("Calculating Final Overlap <Phi_HF | Psi_DMRG> (Robust Method)...")
        print("-" * 60)
        n_occ = mol.nelec // 2
        occ_indices = list(range(n_occ))
        
        final_overlap = calculate_overlap_with_hf_robust(
            last_mps_tensors, final_Cmo, occ_indices, Nz
        )
        print(f"Overlap S       : {final_overlap:.6f}")
        print(f"Overlap |S|^2   : {abs(final_overlap)**2:.6f}")
        energy_log["final_overlap"] = final_overlap

    # Checkpoint 4: Final
    save_checkpoint("04_DMRG_Final", d_stack, last_mps_tensors, energy_log, mol, run_params)

    # --- Summary ---
    print("\n" + "="*60)
    print("Run Complete. Energy Log:")
    print(f"  HF Initial: {energy_log['hf_initial']:.6f}")
    if energy_log['hf_pre_opt']:
        print(f"  HF Pre-Opt Final: {energy_log['hf_pre_opt'][-1]:.6f}")
    for res in energy_log['dmrg_cycles']:
        print(f"  DMRG Cycle {res['cycle']}: {res['e_dmrg']:.6f}")
    if final_overlap is not None:
        print(f"  Final Overlap |S|^2: {abs(final_overlap)**2:.6f}")
    print("="*60)


if __name__ == "__main__":
    charges = [1.0, 1.0, 1.0, 1.0]
    coords = [[0.0, 0.0, 0.91], [0.0, 0.0, -0.91], [0.0, 0.0, -3.6], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4)

    # charges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # coords = [[0.0, 0.0, -2], [0.0, 0.0, -6], [0.0, 0.0, -10], [0.0, 0.0, -14], [0.0, 0.0, 2], [0.0, 0.0, 6], [0.0, 0.0, 10], [0.0, 0.0, 14]]
    # mol = Molecule(charges, coords, nelec=8)
    
    # coords = [[0.0, 2, 2], [0.0, 2, -2], [0.0, -2, -2], [0.0, -2, 2]]
    # coords = [[0.0, 0.7, 0.7], [0.0, 0.7, -0.7], [0.0, -0.7, -0.7], [0.0, -0.7, 0.7]]
    # mol = Molecule(charges, coords, nelec=4)
    
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    
    run_gdvr_dmrg_loop(
        mol, Lz=8.0, Nz=50, basis_cfg=basis_cfg,
        pre_opt_cycles=10,    
        dmrg_cycles=4,         
        dmrg_bond_dim=20,
        dmrg_sweeps=10,
        post_dmrg_opt_cycles=10 
    )
