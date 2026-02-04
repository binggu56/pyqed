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
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
import pyqed.mps.mps as mps_lib
from pyqed.qchem.gdvr.macro_dmrg_scf_sweep import gdvr_dmrg_scf

try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, tensordot
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



# helper functions
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

def get_entangled_guess(n_elec, n_spin):
    if not SYMMETRY_AVAILABLE: return []
    mps = []
    hf_config = [1]*n_elec + [0]*(n_spin - n_elec)
    dbl_config = hf_config.copy()
    if n_spin >= 4 and n_elec >= 2:
        dbl_config[n_elec-1] = 0; dbl_config[n_elec-2] = 0
        dbl_config[n_elec] = 1; dbl_config[n_elec+1] = 1
        
    curr_hf = 0; curr_dbl = 0
    for i in range(n_spin):
        data = {}
        q_l_hf = curr_hf; phys_hf = hf_config[i]; q_r_hf = q_l_hf + phys_hf
        key_hf = (q_l_hf, q_r_hf, phys_hf)
        if key_hf not in data: data[key_hf] = np.zeros((1,1,1))
        data[key_hf][0,0,0] += 0.9 
        
        q_l_dbl = curr_dbl; phys_dbl = dbl_config[i]; q_r_dbl = q_l_dbl + phys_dbl
        key_dbl = (q_l_dbl, q_r_dbl, phys_dbl)
        if key_dbl not in data: data[key_dbl] = np.zeros((1,1,1))
        data[key_dbl][0,0,0] += 0.1 
        
        qns_L = sorted(list(set(k[0] for k in data)))
        qns_R = sorted(list(set(k[1] for k in data)))
        bt = BlockTensor(data, [qns_L, qns_R, [0, 1]], [-1, 1, 1])
        mps.append(bt)
        curr_hf += phys_hf; curr_dbl += phys_dbl
    return mps

def make_u1_random_block_init_guess(L, target_qn, phys_qns=None, max_bond_sectors=6, seed=0):
    if not SYMMETRY_AVAILABLE: raise ImportError("Symmetry required")
    import numpy as np
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    if phys_qns is None: phys_qns = [0, 1]
    phys_qns = list(phys_qns)
    idxs_by_q = defaultdict(list)
    for i, q in enumerate(phys_qns): idxs_by_q[q].append(i)
    bond_qns = [None] * (L + 1)
    bond_qns[0] = [0]; bond_qns[L] = [int(target_qn)]    
    for i in range(1, L):
        qmin = 0; qmax = int(target_qn)
        candidates = list(range(qmin, qmax + 1))
        if len(candidates) <= max_bond_sectors: chosen = candidates
        else: chosen = sorted(rng.choice(candidates, size=max_bond_sectors, replace=False).tolist())
        if 0 not in chosen: chosen[0] = 0; chosen = sorted(set(chosen))
        bond_qns[i] = list(chosen)
    Bs = []
    for i in range(L):
        left_qs = bond_qns[i]; right_qs = bond_qns[i + 1]
        data = {}
        for qL in left_qs:
            for qP, idxs in idxs_by_q.items():
                qR = qL + qP
                if qR not in right_qs: continue
                blk = (rng.standard_normal((1, 1, len(idxs))) + 1j * rng.standard_normal((1, 1, len(idxs))))
                data[(qL, qR, qP)] = blk
        qns = [list(left_qs), list(right_qs), list(phys_qns)]
        B = BlockTensor(data, qns, [-1, 1, 1])
        nrm = B.norm()
        if nrm != 0: B = B * (1.0 / nrm)
        Bs.append(B)
    return Bs


# initial guess for dmrg using HF result (transform from HF state in MO basis to AO basis)

def get_vacuum_mps_symmetric(L):
    """ Creates a U(1) Symmetric Vacuum MPS |00...0>. """
    if not SYMMETRY_AVAILABLE: raise ImportError("Symmetry module required.")
    Bs = []
    for i in range(L):
        data = {(0, 0, 0): np.array([[[1.0]]])} 
        qns = [[0], [0], [0, 1]]
        dirs = [-1, 1, 1] 
        Bs.append(BlockTensor(data, qns, dirs))
    return Bs

def build_creation_mpo_u1(coeff_vector):
    """ Builds U(1) Symmetric MPO for Operator O = sum_i c_i a_i^dag. """
    L = len(coeff_vector)
    tensors = []
    for i in range(L):
        c = coeff_vector[i]
        data = {}
        data[(0, 0, 0, 0)] = np.array([[[[1.0]]]]) 
        data[(0, 0, 1, 1)] = np.array([[[[1.0]]]]) 
        data[(1, 1, 0, 0)] = np.array([[[[1.0]]]])
        data[(1, 1, 1, 1)] = np.array([[[[-1.0]]]])
        
        if abs(c) > 1e-12:
            data[(0, 1, 1, 0)] = np.array([[[[c]]]]) 
            
        if i == 0:   data = {k:v for k,v in data.items() if k[0] == 0}
        if i == L-1: data = {k:v for k,v in data.items() if k[1] == 1}
            
        qns = [[0, 1], [0, 1], [0, 1], [0, 1]]
        tensors.append(BlockTensor(data, qns, [1, -1, 1, -1]))
    return tensors

def _get_qn_dims(bt, leg_idx):
    dims = {}
    for key, block in bt.data.items():
        q = key[leg_idx]; d = block.shape[leg_idx]
        dims[q] = d
    return dims

def apply_mpo_symmetric(W_list, M_list):
    """
    Symmetric application |Psi'> = W |Psi>. 
    Robustly handles block fusion by pre-calculating dimensions.
    """
    new_mps = []
    L = len(M_list)
    # Track basis pairs and dimensions: { q_new: [ ((qw, qm), dim_prod), ... ] }
    last_right_basis_map = {0: [((0, 0), 1)]} 
    
    for i in range(L):
        W = W_list[i]; M = M_list[i]
        
        # 1. Contract Phys Indices: W[In] with M[Phys]
        # (W_L, W_R, W_Out, M_L, M_R)
        T = tensordot(W, M, axes=([3], [2]))
        
        # 2. Determine new Right Basis (QNs and Dimensions)
        # We must find all valid (qw_r, qm_r) pairs that exist in the output.
        current_right_basis_map = collections.defaultdict(list)
        
        # Get dense dimensions of inputs to ensure consistency
        w_dims_l = _get_qn_dims(W, 0); w_dims_r = _get_qn_dims(W, 1)
        m_dims_l = _get_qn_dims(M, 0); m_dims_r = _get_qn_dims(M, 1)
        
        # Scan T to find populated sectors
        for key_T in T.data:
            qw_r, qm_r = key_T[1], key_T[4]
            q_r_new = qw_r + qm_r
            
            # Calculate dimension: Dim(W_R) * Dim(M_R)
            if qw_r in w_dims_r and qm_r in m_dims_r:
                d = w_dims_r[qw_r] * m_dims_r[qm_r]
                pair_info = ((qw_r, qm_r), d)
                if pair_info not in current_right_basis_map[q_r_new]:
                    current_right_basis_map[q_r_new].append(pair_info)
        
        for q in current_right_basis_map:
            # Sort by qn pair values to ensure deterministic ordering of blocks
            current_right_basis_map[q].sort(key=lambda x: x[0])
            
        # 3. Construct Blocks
        new_data = {}
        blocks_by_sector = collections.defaultdict(dict)
        
        for key_T, block in T.data.items():
            qw_l, qw_r, q_p_out, qm_l, qm_r = key_T
            q_l_new = qw_l + qm_l; q_r_new = qw_r + qm_r
            sector = (q_l_new, q_r_new, q_p_out)
            comp_key = ((qw_l, qm_l), (qw_r, qm_r))
            blocks_by_sector[sector][comp_key] = block

        # Iterate over all potential target sectors formed by valid Left and Right basis
        # We need the set of all (q_l_new, q_r_new) that exist in T OR could exist.
        # Here we iterate T sectors because if T is zero, the block is zero.
        
        for sector, comps in blocks_by_sector.items():
            q_l_new, q_r_new, q_p_out = sector
            
            row_info_list = last_right_basis_map.get(q_l_new, [])
            col_info_list = current_right_basis_map.get(q_r_new, [])
            
            if not row_info_list or not col_info_list: continue 
            
            # Calculate total dimensions
            r_dim = sum(d for _, d in row_info_list)
            c_dim = sum(d for _, d in col_info_list)
            
            # Pre-calculate offsets
            row_offsets = {}; current_r = 0
            for pair, d in row_info_list:
                row_offsets[pair] = (current_r, d)
                current_r += d
            
            col_offsets = {}; current_c = 0
            for pair, d in col_info_list:
                col_offsets[pair] = (current_c, d)
                current_c += d

            # BOUNDARY CONDITION: Force Dim=1 at last site
            # If we are at the end, we project to the unique vacuum state.
            if i == L-1:
                # We assume the valid state is index 0.
                if c_dim > 1:
                    c_dim = 1
                    # Update offsets to map everything to 0? No, that merges distinct states.
                    # We usually just slice [0:1]. 
                    # But we must ensure the `col_offsets` logic doesn't write out of bounds.
                    pass 

            new_block = np.zeros((r_dim, c_dim, 1), dtype=complex) 
            
            for ((w_l, m_l), (w_r, m_r)), blk in comps.items():
                if (w_l, m_l) not in row_offsets or (w_r, m_r) not in col_offsets: continue
                
                r_start, r_len = row_offsets[(w_l, m_l)]
                c_start, c_len_full = col_offsets[(w_r, m_r)]
                
                # Transpose: (WL, WR, Out, ML, MR) -> (WL, ML, WR, MR, Out)
                # T indices: 0:WL, 1:WR, 2:Out, 3:ML, 4:MR
                # Perm: (0, 3, 1, 4, 2)
                blk_perm = blk.transpose(0, 3, 1, 4, 2)
                dim_p = blk.shape[2]
                
                if new_block.shape[2] != dim_p: 
                    new_block = np.zeros((r_dim, c_dim, dim_p), dtype=complex)
                
                # Flatten Left and Right
                to_fill = blk_perm.reshape(blk.shape[0]*blk.shape[3], blk.shape[1]*blk.shape[4], dim_p)
                
                # Check bounds (handles boundary condition slicing)
                actual_r = min(r_len, to_fill.shape[0])
                actual_c = min(c_len_full, c_dim - c_start) # Ensure we don't write past c_dim
                actual_c = min(actual_c, to_fill.shape[1])
                
                if actual_r > 0 and actual_c > 0:
                    new_block[r_start:r_start+actual_r, c_start:c_start+actual_c, :] = to_fill[:actual_r, :actual_c, :]

            new_data[sector] = new_block

        qns_L = sorted(list(set(k[0] for k in new_data)))
        qns_R = sorted(list(set(k[1] for k in new_data)))
        qns_P = [0, 1]
        
        new_mps.append(BlockTensor(new_data, [qns_L, qns_R, qns_P], [-1, 1, 1]))
        last_right_basis_map = current_right_basis_map

    return new_mps

def generate_exact_hf_guess(mol, C_mo_spatial, Nz):
    print(f"  [Guess] Generating EXACT HF Slater Determinant (AO Basis, U1)...")
    n_spin = 2 * Nz
    mps = get_vacuum_mps_symmetric(n_spin)
    
    n_occ = mol.nelec // 2
    for mo_idx in range(n_occ):
        vec = C_mo_spatial[:, mo_idx]
        op_up = np.zeros(n_spin); op_up[0::2] = vec
        mpo = build_creation_mpo_u1(op_up)
        mps = apply_mpo_symmetric(mpo, mps)
        
        op_down = np.zeros(n_spin); op_down[1::2] = vec
        mpo = build_creation_mpo_u1(op_down)
        mps = apply_mpo_symmetric(mpo, mps)
        
    dims = [sum([b.shape[0]*b.shape[1] for b in t.data.values()]) for t in mps] 
    print(f"  [Guess] Finished. Exact HF MPS generated. Dims ~ {max(dims)}")
    return mps



def convert_mpo_symmetric(dense_H_list):
    if not SYMMETRY_AVAILABLE: return dense_H_list
    print("  Converting MPO to U(1) Blocks...")
    sym_H = []
    phys_qns = {0: 0, 1: 1} 
    current_nodes = {(0, 0)}
    for W in dense_H_list:
        new_data = {}
        next_nodes = set()
        valid_incoming = {}
        for l, q in current_nodes:
            if l not in valid_incoming: valid_incoming[l] = set()
            valid_incoming[l].add(q)
        idxs = np.nonzero(np.abs(W) > 1e-14)
        for i in range(len(idxs[0])):
            l, r, out_s, in_s = idxs[0][i], idxs[1][i], idxs[2][i], idxs[3][i]
            val = W[l, r, out_s, in_s]
            if l not in valid_incoming: continue
            q_out = phys_qns[out_s]; q_in = phys_qns[in_s]
            flux = q_out - q_in
            for q_l in valid_incoming[l]:
                q_r = q_l - flux
                next_nodes.add((r, q_r))
                key = (q_l, q_r, phys_qns[out_s], phys_qns[in_s])
                if key not in new_data: new_data[key] = []
                new_data[key].append( ((l, q_l), (r, q_r), val) )
        l_map = {q: sorted([x for x in current_nodes if x[1]==q]) for q in set(x[1] for x in current_nodes)}
        r_map = {q: sorted([x for x in next_nodes if x[1]==q]) for q in set(x[1] for x in next_nodes)}
        final_blocks = {}
        for key, elems in new_data.items():
            q_l, q_r, q_o, q_i = key
            if q_l not in l_map or q_r not in r_map: continue
            rows = l_map[q_l]; cols = r_map[q_r]
            row_idx = {x: i for i, x in enumerate(rows)}
            col_idx = {x: i for i, x in enumerate(cols)}
            blk = np.zeros((len(rows), len(cols), 1, 1))
            for (nl, nr, v) in elems:
                blk[row_idx[nl], col_idx[nr], 0, 0] = v
            final_blocks[key] = blk
        qns_L = sorted(list(l_map.keys())); qns_R = sorted(list(r_map.keys()))
        bt = BlockTensor(final_blocks, [qns_L, qns_R, [], []], [-1, 1, 1, -1])
        sym_H.append(bt)
        current_nodes = next_nodes
    return sym_H

def align_orbital_phases(d_old, d_new, S_prim):
    Nz = d_old.shape[0]
    min_overlap = 1.0
    for n in range(Nz):
        overlap = float(d_old[n].T @ S_prim @ d_new[n])
        if overlap < 0:
            d_new[n] *= -1.0; overlap = -overlap
        min_overlap = min(min_overlap, overlap)
    return d_new, min_overlap

def apply_mpo_to_mps(mpo_tensors, mps_tensors):
    new_mps = []
    L = len(mps_tensors)
    for i in range(L):
        M = mps_tensors[i]; W = mpo_tensors[i]         
        T = np.tensordot(W, M, axes=([3], [0]))
        T = T.transpose(2, 0, 3, 1, 4)
        d_out = T.shape[0]
        new_DL = T.shape[1] * T.shape[2]
        new_DR = T.shape[3] * T.shape[4]
        new_M = T.reshape(d_out, new_DL, new_DR)
        new_mps.append(new_M)
    return new_mps

def build_annihilation_mpo(coeff_vector):
    L = len(coeff_vector); tensors = []
    I = np.eye(2); Z = np.array([[1, 0], [0, -1]]); a = np.array([[0, 1], [0, 0]]) 
    for i in range(L):
        c = coeff_vector[i]
        W = np.zeros((2, 2, 2, 2))
        W[0, 0] = Z; W[0, 1] = c * a; W[1, 1] = I
        if i == 0: W = W[0:1, :, :, :] 
        if i == L-1: W = W[:, 1:2, :, :] 
        tensors.append(W)
    return tensors

def _bt_to_dense(bt):
    """ Helper to convert BlockTensor to dense array for overlap check. """
    maps = []
    for qlist in bt.qns:
        m = collections.defaultdict(list)
        for i, q in enumerate(qlist): m[q].append(i)
        maps.append(m)
    shape = tuple(len(q) for q in bt.qns)
    out = np.zeros(shape, dtype=complex)
    for qkey, block in bt.data.items():
        idx_lists = [maps[leg][qkey[leg]] for leg in range(bt.rank)]
        out[np.ix_(*idx_lists)] += block
    return out

def calculate_overlap_with_hf_robust(mps_tensors, C_mo_spatial, occupied_indices, n_spatial):
    curr_mps = []
    for t in mps_tensors:
        if hasattr(t, 'data'): curr_mps.append(_bt_to_dense(t))
        else: curr_mps.append(t.copy())
    ops = []
    for mo_idx in occupied_indices:
        vec = C_mo_spatial[:, mo_idx]
        op_down = np.zeros(2 * n_spatial); op_down[1::2] = vec; ops.append(op_down)
        op_up = np.zeros(2 * n_spatial); op_up[0::2] = vec; ops.append(op_up)
    for k, op_vec in enumerate(ops):
        mpo = build_annihilation_mpo(op_vec)
        curr_mps = apply_mpo_to_mps(mpo, curr_mps)
    val = np.array([[1.0]])
    for M in curr_mps:
        if M.shape[0] < 1: val = val * 0.0; break
        mat = M[0, :, :]; val = val @ mat 
    return val.flatten()[0]

def save_checkpoint(stage_name, d_stack, mps_tensors, energy_dict, mol, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{stage_name}"
    print(f"  [Save] Checkpoint: {stage_name}")
    np.savez_compressed(f"{filename}_orbitals.npz", d_stack=d_stack)
    if mps_tensors is not None:
        np.savez_compressed(f"{filename}_mps.npz", *mps_tensors)
    meta = {"mol": (mol.coords, mol.charges), "log": energy_dict, "params": params}
    with open(f"{filename}_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def run_gdvr_dmrg_loop(
    mol, Lz, Nz, basis_cfg,
    pre_opt_cycles=10,      
    dmrg_cycles=3,          
    dmrg_bond_dim=20,
    dmrg_sweeps=10,
    post_dmrg_opt_cycles=5,
    U1 = True,
    checkpoint_dir = "."
):
    print("="*60)
    print(f"GDVR-DMRG | Exact HF Guess Mode | U1={U1}")
    print(f"System: {mol.nelec} e-, Nz={Nz}, Lz={Lz}")
    print("="*60)
    
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
    
    print(f"  -> Initial HF Energy: {Etot:.8f} Ha")
    energy_log["hf_initial"] = Etot
    d_stack = np.vstack([C_list[n][:, 0] for n in range(Nz)])
    save_checkpoint("01_HF_Initial", d_stack, None, energy_log, mol, run_params, checkpoint_dir)
    
    _, Kz_grid, _ = sine_dvr_1d(-Lz, Lz, Nz)
    ERIop = CollocatedERIOp.from_kernels(N=n_ao_2d, Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    h1_nm_func = build_h1_nm(Kz_grid, S_prim, T_prim, z, lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz))

    if pre_opt_cycles > 0:
        print(f"\n[Phase A.5] Pre-optimization...")
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

    save_checkpoint("02_HF_NewtonOpt", d_stack, None, energy_log, mol, run_params, checkpoint_dir)

    # --- Phase B: Loop ---
    last_mps_tensors = None 
    d_stack_old = d_stack.copy()
    final_dmrg_energy = 0.0
    final_Cmo = None
    
    for cycle in range(dmrg_cycles):
        print(f"\n[Macro Cycle {cycle+1}/{dmrg_cycles}]")
        d_stack, _ = align_orbital_phases(d_stack_old, d_stack, S_prim)
        d_stack_old = d_stack.copy()
        
        Hcore_curr = rebuild_Hcore_from_d(d_stack, z, Kz_grid, S_prim, T_prim, alphas, centers, labels, nuclei)
        C_list_curr = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
        V_coul, V_exch = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
        V_coul = np.array(V_coul) 
        
        # Build MPO
        ham_terms = []
        rows, cols = np.nonzero(np.abs(Hcore_curr) > 1e-10)
        for i, j in zip(rows, cols):
            val = Hcore_curr[i, j]
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i, 2*j], val))
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i+1, 2*j+1], val))
        
        rows, cols = np.nonzero(np.abs(V_coul) > 1e-10)
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
        
        basis = [BasisSimpleElectron(i) for i in range(2*Nz)]
        model = Model(basis=basis, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")
        mpo_dmrg = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        final_H = convert_mpo_symmetric(mpo_dmrg) if U1 else mpo_dmrg
        
        if last_mps_tensors is None:
            # USE EXACT HF GUESS (FIXED)
            mps_guess = generate_exact_hf_guess(mol, Cmo, Nz)
        else:
            mps_guess = [t.copy() for t in last_mps_tensors]
        
        print(f"  3. Running DMRG (D={dmrg_bond_dim})...")
        solver = mps_lib.DMRG(final_H, D=dmrg_bond_dim, nsweeps=dmrg_sweeps, init_guess=mps_guess, U1=U1, target_qn=mol.nelec, not_conv_err=False)
        solver.run()
        
        try:
            psi_tensors = solver.ground_state.Bs
            e_elec = mps_lib.expect_mps(psi_tensors, solver.H, psi_tensors)
            e_dmrg = np.real(e_elec) + Enuc
        except:
            e_dmrg = solver.e_tot + Enuc
            
        last_mps_tensors = solver.ground_state.Bs
        final_dmrg_energy = e_dmrg
        print(f"     -> Final Cycle Energy: {e_dmrg:.8f} Ha")
        
        if cycle == 0:
            save_checkpoint("03_DMRG_FirstIter", d_stack, last_mps_tensors, energy_log, mol, run_params, checkpoint_dir)

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

    final_overlap = 0.0
    try:
        if final_Cmo is not None and last_mps_tensors is not None:
            print("\n" + "-"*30)
            print("Calculating Final Overlap...")
            final_overlap = calculate_overlap_with_hf_robust(last_mps_tensors, final_Cmo, range(mol.nelec // 2), Nz)
            print(f"Overlap |S|^2 : {abs(final_overlap)**2:.6f}")
    except Exception as e:
        print(f"Overlap calculation failed (returning 0.0): {e}")
        final_overlap = 0.0
    
    energy_log["final_overlap"] = final_overlap
    save_checkpoint("04_DMRG_Final", d_stack, last_mps_tensors, energy_log, mol, run_params, checkpoint_dir)
    return final_dmrg_energy, final_overlap

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
    mol = Molecule(charges, coords, nelec=4)
    
    master_dir = f"Scan_Results_Nz_{Nz}"
    checkpoint_path = os.path.join(master_dir)
    
    E, S = run_gdvr_dmrg_loop(
        mol, Lz=6.0, Nz=64, basis_cfg=basis_cfg,
        pre_opt_cycles=10, dmrg_cycles=4, dmrg_bond_dim=40, dmrg_sweeps=10, post_dmrg_opt_cycles=10,
        U1=True, checkpoint_dir=checkpoint_path
    )
    
    result_file = os.path.join(master_dir, f"result_idx_{idx:02d}.npz")
    np.savez(result_file, Energy=E, Overlap=S)
    print(f"Done. Saved to {result_file}")