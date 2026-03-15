import numpy as np
import pickle
import os
import logging
import time
import collections
import argparse

from pyqed.qchem.gdvr.gdvr_mean_field import (
    Molecule, build_method2, make_xy_spd_primitive_basis, 
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,
    scf_rhf_method2, sine_dvr_1d, eri_JK_from_kernels_M1,
    build_h1_nm, V_en_sp_total_at_z, CollocatedERIOp, rebuild_Hcore_from_d,
    SweepNewtonHelper, sweep_optimize_driver
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def save_checkpoint(stage_name, d_stack, mps_tensors, energy_dict, mol, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{stage_name}"
    logger.info(f"  [Save] Checkpoint: {stage_name}")
    np.savez_compressed(f"{filename}_orbitals.npz", d_stack=d_stack)
    if mps_tensors is not None:
        np.savez_compressed(f"{filename}_mps.npz", *mps_tensors)
    meta = {"mol": (mol.coords, mol.charges), "log": energy_dict, "params": params}
    with open(f"{filename}_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def run_hf_ao_opt(
    mol, Lz, Nz, basis_cfg,
    pre_opt_cycles=10,      
    checkpoint_dir = "."
):
    """
    gdvr hartree fock with atomic orbital contraction coefficients reoptimization

    Parameters
    ----------
    mol : _type_
        _description_
    Lz : _type_
        _description_
    Nz : _type_
        _description_
    basis_cfg : _type_
        _description_
    pre_opt_cycles : int, optional
        _description_, by default 10
    checkpoint_dir : str, optional
        _description_, by default "."

    Returns
    -------
    _type_
        _description_
    """
    logger.info("="*60)
    logger.info(f"System: {mol.nelec} e-, Nz={Nz}, Lz={Lz}")
    logger.info("="*60)
    
    energy_log = {"hf_initial": None, "hf_pre_opt": [], "dmrg_cycles": [], "final_overlap": None}
    run_params = {"Lz": Lz, "Nz": Nz, "basis": basis_cfg}

    s_exps = basis_cfg.get('s'); p_exps = basis_cfg.get('p', []); d_exps = basis_cfg.get('d', [])
    Hcore, z, dz, E_slices, C_list, _, _, _ = build_method2(
        mol, Lz=Lz, Nz=Nz, M=1, s_exps=s_exps, p_exps=p_exps, d_exps=d_exps, verbose=False, dvr_method='sine'
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
    
    logger.info(f"  -> Initial HF Energy: {Etot:.8f} Ha")
    energy_log["hf_initial"] = Etot
    d_stack = np.vstack([C_list[n][:, 0] for n in range(Nz)])
    save_checkpoint("01_HF_Initial", d_stack, None, energy_log, mol, run_params, checkpoint_dir)
    
    _, Kz_grid, _ = sine_dvr_1d(-Lz, Lz, Nz)
    ERIop = CollocatedERIOp.from_kernels(N=n_ao_2d, Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    h1_nm_func = build_h1_nm(Kz_grid, S_prim, T_prim, z, lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz))

    if pre_opt_cycles > 0:
        logger.info(f"\n[Phase A.5] Pre-optimization...")
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
            if (pcyc + 1) % 2 == 0: logger.info(f"   Cycle {pcyc+1}: HF Energy = {Etot:.8f} Ha")

    save_checkpoint("02_HF_NewtonOpt", d_stack, None, energy_log, mol, run_params, checkpoint_dir)

    return d_stack, energy_log, run_params #run_params = {"Lz": Lz, "Nz": Nz, "basis": basis_cfg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=32, required=False)
    parser.add_argument('--idx', type=int, default=0, required=False)
    args = parser.parse_args()

    Nz = args.nz
    idx = args.idx

    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    charges = [1.0]*4
    coords = [[0.0, 0.0, -3.6], [0.0, 0.0, -0.91], [0.0, 0.0, 0.91], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4, spin = 0)
    
    master_dir = f"Scan_Results_Nz_{Nz}"
    checkpoint_path = os.path.join(master_dir)
    
    d_stack, energy_log, run_params = run_hf_ao_opt(
        mol, Lz=6.0, Nz=128, basis_cfg=basis_cfg,
        pre_opt_cycles=10, checkpoint_dir=checkpoint_path
    )
    
    # result_file = os.path.join(master_dir, f"result_idx_{idx:02d}.npz")
    # np.savez(result_file, Energy=E, Overlap=S)
    # logger.info(f"Done. Saved to {result_file}")