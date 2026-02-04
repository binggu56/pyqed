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

try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, tensordot
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None

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

def get_entangled_guess(n_elec, n_spin):
    """
    Constructs a superposition |HF> + alpha*|Doubles>.
    This guarantees Schmidt Rank > 1, preventing 'States=1' collapse.
    """
    if not SYMMETRY_AVAILABLE: return []
    mps = []
    
    # HF Config: First n_elec are Occ(1)
    hf_config = [1]*n_elec + [0]*(n_spin - n_elec)
    
    # Double Excitation Config: Move 2e from HOMO to LUMO
    dbl_config = hf_config.copy()
    if n_spin >= 4 and n_elec >= 2:
        dbl_config[n_elec-1] = 0 # Emp
        dbl_config[n_elec-2] = 0 # Emp
        dbl_config[n_elec]   = 1 # Occ
        dbl_config[n_elec+1] = 1 # Occ
        
    print(f"  [Guess] HF: {hf_config}")
    print(f"  [Guess] Dbl: {dbl_config}")
    
    curr_hf = 0; curr_dbl = 0
    
    for i in range(n_spin):
        data = {}
        
        # Path 1: HF
        q_l_hf = curr_hf
        phys_hf = hf_config[i] # 1 or 0
        q_r_hf = q_l_hf + phys_hf
        
        key_hf = (q_l_hf, q_r_hf, phys_hf)
        if key_hf not in data: data[key_hf] = np.zeros((1,1,1))
        data[key_hf][0,0,0] += 0.9 # Weight for HF
        
        # Path 2: Doubles
        q_l_dbl = curr_dbl
        phys_dbl = dbl_config[i]
        q_r_dbl = q_l_dbl + phys_dbl
        
        key_dbl = (q_l_dbl, q_r_dbl, phys_dbl)
        if key_dbl not in data: data[key_dbl] = np.zeros((1,1,1))
        data[key_dbl][0,0,0] += 0.1 # Weight for Doubles
        
        qns_L = sorted(list(set(k[0] for k in data)))
        qns_R = sorted(list(set(k[1] for k in data)))
        
        # [0, 1] means both Emp and Occ sectors are allowed
        bt = BlockTensor(data, [qns_L, qns_R, [0, 1]], [-1, 1, 1])
        mps.append(bt)
        
        curr_hf += phys_hf
        curr_dbl += phys_dbl
        
    return mps

def make_u1_random_block_init_guess(
    L,
    target_qn,
    phys_qns=None,
    max_bond_sectors=6,
    seed=0,
    complex_dtype=True,
):
    """
    Build a random U(1) BlockTensor MPS with bond dimension > 1 (multiple charge sectors),
    while enforcing total charge = target_qn on the *right boundary bond*.

    Site tensor convention in your code: (LeftBond, RightBond, Phys)
    Block key convention: (qL, qR, qP) with charge flow qR = qL + qP.

    Default phys_qns:
      - d=2 spin-orbital occupancy: [0,1]
      - d=4 spatial orbital:        [0,1,1,2]
    """
    if not SYMMETRY_AVAILABLE:
        raise ImportError("Symmetry module not found: cannot build BlockTensor init guess.")

    import numpy as np
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    if phys_qns is None:
        # default to spin-orbital (your current case looks like d=2)
        phys_qns = [0, 1]
    phys_qns = list(phys_qns)

    # map charge -> list of physical basis indices (for degeneracy, e.g. charge 1 has 2 states in d=4)
    idxs_by_q = defaultdict(list)
    for i, q in enumerate(phys_qns):
        idxs_by_q[q].append(i)

    # choose bond charge sectors at each cut i (bond i is between sites i-1 and i) ----
    # bond_qns[i] is the list of charges carried by that bond basis
    bond_qns = [None] * (L + 1)
    bond_qns[0] = [0]                 # left boundary fixed
    bond_qns[L] = [int(target_qn)]    # right boundary fixed

    for i in range(1, L):
        # feasible charge range at cut i:
        #   at least 0, at most target_qn
        # also must be achievable with remaining sites (each site contributes <= max phys charge)
        qmin = 0
        qmax = int(target_qn)

        # keep it simple: choose up to max_bond_sectors charges uniformly from [qmin,qmax]
        # but always include something feasible; for better conditioning include neighbors too
        candidates = list(range(qmin, qmax + 1))
        if len(candidates) <= max_bond_sectors:
            chosen = candidates
        else:
            chosen = sorted(rng.choice(candidates, size=max_bond_sectors, replace=False).tolist())

        # this is important: make sure 0 is included early, and target_qn included late helps connectivity
        if 0 not in chosen:
            chosen[0] = 0
            chosen = sorted(set(chosen))
        bond_qns[i] = list(chosen)

    # build site BlockTensors 
    Bs = []
    dtype = np.complex128 if complex_dtype else np.float64

    for i in range(L):
        left_qs  = bond_qns[i]
        right_qs = bond_qns[i + 1]

        # sector degeneracy: we take 1 basis state per charge sector (you can increase by repeating charges)
        # so dim(q) = 1 for each q in the list.
        data = {}

        for qL in left_qs:
            for qP, idxs in idxs_by_q.items():
                qR = qL + qP
                if qR not in right_qs:
                    continue

                # block shape: (dimL, dimR, dimPhysSector)
                # here dimL=dimR=1; dimPhysSector=len(idxs)
                blk = (rng.standard_normal((1, 1, len(idxs))) +
                       1j * rng.standard_normal((1, 1, len(idxs)))).astype(dtype)
                data[(qL, qR, qP)] = blk

        # If connectivity is too sparse (possible for unlucky random sector choices), fail loudly
        if len(data) == 0:
            raise RuntimeError(
                f"Site {i}: no allowed (qL,qR,qP) blocks. "
                f"left_qs={left_qs}, right_qs={right_qs}, phys_qns={phys_qns}"
            )

        qns  = [list(left_qs), list(right_qs), list(phys_qns)]
        dirs = [-1, 1, 1]  # your convention
        B = BlockTensor(data, qns, dirs)

        # normalize each tensor a bit so Davidson doesn't start with huge norm variations
        nrm = B.norm()
        if nrm != 0:
            B = B * (1.0 / nrm)

        Bs.append(B)

    return Bs





def get_vacuum_mps_symmetric(L):
    """
    Creates a U(1) Symmetric Vacuum MPS |00...0>.
    """
    if not SYMMETRY_AVAILABLE:
        raise ImportError("Symmetry module required for U1 vacuum.")
    
    Bs = []
    # Vacuum: All bonds QN=0. Physical state 0 (Empty) occupied.
    for i in range(L):
        # Key: (Left=0, Right=0, Phys=0)
        # Shape: (1, 1, 1)
        data = {(0, 0, 0): np.array([[[1.0]]])} 
        
        # Directions: [-1, 1, 1] (Left-In, Right-Out, Phys-Out) matches standard MPS
        qns = [[0], [0], [0, 1]]
        dirs = [-1, 1, 1] 
        
        B = BlockTensor(data, qns, dirs)
        Bs.append(B)
    return Bs

def apply_mpo_symmetric(W_list, M_list):
    """
    Symmetric version of applying MPO to MPS: |Psi'> = W |Psi>.
    Manually handles block fusion to increase bond dimension.
    """
    import collections
    new_mps = []
    L = len(M_list)
    
    # We need to track the basis of the fused bonds to ensure consistency 
    # between site i (Right) and site i+1 (Left).
    # Format: dict mapping QN -> list of (q_w, q_m) pairs
    last_right_basis_map = {0: [(0, 0)]} 
    
    for i in range(L):
        W = W_list[i] # (L, R, Out, In)
        M = M_list[i] # (L, R, P)
        
        # 1. Contract Physical Indices: W[In] with M[Phys]
        # W axes: 3 (In), M axes: 2 (Phys)
        # Result T indices: (W_L, W_R, W_Out, M_L, M_R)
        T = tensordot(W, M, axes=([3], [2]))
        
        # 2. Prepare Data for New BlockTensor
        new_data = {}
        # We need to determine the new Right indices/basis for this site
        current_right_basis_map = collections.defaultdict(list)
        
        # Pre-scan to build the Right Basis Map for this site
        # We look at all resulting (Q_WR, Q_MR) combinations
        for key_T in T.data:
            # key_T: (qw_l, qw_r, q_p_out, qm_l, qm_r)
            qw_r, qm_r = key_T[1], key_T[4]
            q_r_new = qw_r + qm_r
            pair = (qw_r, qm_r)
            if pair not in current_right_basis_map[q_r_new]:
                current_right_basis_map[q_r_new].append(pair)
                
        # Sort basis for determinism
        for q in current_right_basis_map:
            current_right_basis_map[q].sort()
            
        # 3. Construct Blocks
        # We iterate over target sectors (Q_L_new, Q_R_new, Q_P_new)
        # Q_L_new comes from last_right_basis_map (consistency check)
        
        # Group T blocks by their target new indices
        blocks_by_sector = collections.defaultdict(dict)
        
        for key_T, block in T.data.items():
            qw_l, qw_r, q_p_out, qm_l, qm_r = key_T
            
            q_l_new = qw_l + qm_l
            q_r_new = qw_r + qm_r
            
            # Record the block and its constituent indices
            sector = (q_l_new, q_r_new, q_p_out)
            # We index components by the pairs (qw, qm)
            comp_key = ((qw_l, qm_l), (qw_r, qm_r))
            blocks_by_sector[sector][comp_key] = block

        # Stitch blocks together
        for sector, comps in blocks_by_sector.items():
            q_l_new, q_r_new, q_p_out = sector
            
            # Basis definitions
            row_pairs = last_right_basis_map.get(q_l_new, [])
            col_pairs = current_right_basis_map.get(q_r_new, [])
            
            if not row_pairs or not col_pairs: continue 
            
            # Calculate offsets
            row_offsets = {}; r_dim = 0
            for pair in row_pairs: # pair is (qw_l, qm_l)
                # Find dimension from ANY block in T that had this left pair
                # We can't easily get it if it's missing from T.
                # Robust way: use M and W dimensions.
                # Dim = dim(W_Left[qw_l]) * dim(M_Left[qm_l])
                # This requires access to W/M structures. 
                # Simpler: infer from the blocks we HAVE in 'comps'.
                # NOTE: This assumes dense connectivity. Sparse MPOs might miss blocks.
                # For Creation Op, it is safe.
                
                # Hack: find a block in comps with this row_pair
                found = False
                for c_key, blk in comps.items():
                    if c_key[0] == pair:
                        # blk shape: (dw_l, dw_r, d_p, dm_l, dm_r)
                        # We need fused L: dw_l * dm_l
                        d = blk.shape[0] * blk.shape[3]
                        row_offsets[pair] = (r_dim, d)
                        r_dim += d
                        found = True
                        break
                if not found:
                    # If this (L) combination didn't appear in T, it means zero weight.
                    # We still need to account for its dimension to match the basis of the previous site!
                    # For "Creation MPO" on Vacuum, this sparsity is tricky.
                    # HOWEVER, for HF construction, we strictly grow.
                    # Let's assume non-zero blocks define the manifold.
                    pass

            col_offsets = {}; c_dim = 0
            for pair in col_pairs:
                for c_key, blk in comps.items():
                    if c_key[1] == pair:
                        # Fused R: dw_r * dm_r
                        d = blk.shape[1] * blk.shape[4]
                        col_offsets[pair] = (c_dim, d)
                        c_dim += d
                        break
            
            if r_dim == 0 or c_dim == 0: continue

            # Allocate new fused block (L_new, R_new, P)
            # T indices: (W_L, W_R, W_Out, M_L, M_R)
            # Target: (Fused_L, Fused_R, W_Out)
            new_block = np.zeros((r_dim, c_dim, 1), dtype=complex) 
            # Note: 3rd dim '1' is placeholder, we actually reshape result
            
            # Fill
            for ((w_l, m_l), (w_r, m_r)), blk in comps.items():
                if (w_l, m_l) not in row_offsets or (w_r, m_r) not in col_offsets: continue
                
                r_start, r_len = row_offsets[(w_l, m_l)]
                c_start, c_len = col_offsets[(w_r, m_r)]
                
                # blk: (dim_wl, dim_wr, dim_p, dim_ml, dim_mr)
                # Transpose to (dim_wl, dim_ml, dim_wr, dim_mr, dim_p)
                blk_perm = blk.transpose(0, 3, 1, 4, 2)
                # Reshape to (FusedL, FusedR, dim_p)
                dim_p = blk.shape[2]
                blk_fused = blk_perm.reshape(r_len, c_len, dim_p)
                
                if new_block.shape[2] != dim_p:
                    new_block = np.zeros((r_dim, c_dim, dim_p), dtype=complex)
                    
                new_block[r_start:r_start+r_len, c_start:c_start+c_len, :] = blk_fused
                
            new_data[sector] = new_block

        # Create BlockTensor
        # QNs: [List of QL, List of QR, List of Phys]
        # We need to reconstruct the full lists from the data keys
        qns_L = sorted(list(set(k[0] for k in new_data)))
        qns_R = sorted(list(set(k[1] for k in new_data)))
        qns_P = [0, 1] # Phys always 0,1
        
        B_new = BlockTensor(new_data, [qns_L, qns_R, qns_P], [-1, 1, 1])
        new_mps.append(B_new)
        
        last_right_basis_map = current_right_basis_map

    return new_mps

def build_creation_mpo_u1(coeff_vector):
    """
    Builds U(1) Symmetric MPO for Operator O = sum_i c_i a_i^dag.
    """
    L = len(coeff_vector)
    tensors = []
    
    # Phys QNs: 0, 1
    # MPO Bond QNs: 0 (Identity), 1 (Creation applied)
    # Fluxes:
    # I: 0->0 (flux 0)
    # Z: 1->1 (flux 0)
    # a^dag: 0->1 (flux 1)
    
    for i in range(L):
        c = coeff_vector[i]
        data = {}
        
        # Block (L=0, R=0): Identity [Out=0, In=0] -> Flux 0
        # Wait, MPO structure: (L, R, Out, In)
        # convert_mpo convention: q_r = q_l - (q_out - q_in)
        # Z: 1->1. q_out=0, q_in=0 (if Z is on vacuum). Z acts on occ?
        # Let's map standard terms:
        # I (0->0): L=0, R=0. q_out=0, q_in=0.
        # Z (1->1): L=1, R=1. q_out=0, q_in=0? No, Z is diagonal.
        # a^dag (0->1): L=0, R=1. q_out=1, q_in=0. (Flux 1).
        
        # 1. Identity (L=0, R=0) -> I
        # Pauli I: [[1,0],[0,1]]
        data[(0, 0, 0, 0)] = np.array([[[[1.0]]]]) 
        data[(0, 0, 1, 1)] = np.array([[[[1.0]]]]) 

        # 2. Pauli Z (L=1, R=1) -> used for string
        # Z: [[1,0],[0,-1]]
        data[(1, 1, 0, 0)] = np.array([[[[1.0]]]])
        data[(1, 1, 1, 1)] = np.array([[[[-1.0]]]])
        
        # 3. Creation (L=0, R=1) -> c * a^dag
        # a^dag: |1><0|
        if abs(c) > 1e-12:
            data[(0, 1, 1, 0)] = np.array([[[[c]]]])
            
        # Boundary conditions
        # Site 0: Start at L=0. Keep only blocks with L=0.
        if i == 0:
            data = {k:v for k,v in data.items() if k[0] == 0}
        # Site L-1: End at R=1 (must create particle). Keep only R=1.
        if i == L-1:
            data = {k:v for k,v in data.items() if k[1] == 1}
            
        # Construct Tensor
        # QNs for bonds 0, 1
        qns = [[0, 1], [0, 1], [0, 1], [0, 1]]
        # If boundaries restricted indices, strictly we should shrink QNs list, 
        # but BlockTensor handles extra QNs fine usually.
        
        tensors.append(BlockTensor(data, qns, [1, -1, 1, -1])) # L, R, Out, In convention?
        # Note: Your convert_mpo uses convention (L, R, Out, In) with dirs [-1, 1, 1, -1]?
        # Let's verify convert_mpo output dirs. 
        # It sets [-1, 1, 1, -1] -> In, Out, Out, In.
        # My build_creation_mpo_u1 should match.
        # Let's assume standard logic: L(in), R(out), P_out, P_in.
        
    return tensors

def generate_exact_hf_guess(mol, C_mo_spatial, Nz):
    print("  [Guess] Generating EXACT HF Slater Determinant (AO Basis, U1)...")
    
    # 1. Vacuum
    n_spin = 2 * Nz
    mps = get_vacuum_mps_symmetric(n_spin)
    
    # 2. Occupied Orbitals
    n_occ = mol.nelec // 2
    
    # 3. Apply Creation MPO for each electron
    # Apply spin-up then spin-down for each MO
    for mo_idx in range(n_occ):
        vec = C_mo_spatial[:, mo_idx]
        
        # Up Spin
        op_up = np.zeros(n_spin); op_up[0::2] = vec
        mpo = build_creation_mpo_u1(op_up)
        mps = apply_mpo_symmetric(mpo, mps)
        
        # Down Spin
        op_down = np.zeros(n_spin); op_down[1::2] = vec
        mpo = build_creation_mpo_u1(op_down)
        mps = apply_mpo_symmetric(mpo, mps)
        
    print(f"  [Guess] Finished. Final MPS bond dimensions: {[len(b.data) for b in mps]}")
    return mps


def convert_mpo_symmetric(dense_H_list):
    """ Standard Mapping: 0=Emp, 1=Occ """
    if not SYMMETRY_AVAILABLE: return dense_H_list
    print("  Converting MPO to U(1) Blocks (Standard: 0=Emp, 1=Occ)...")
    sym_H = []
    phys_qns = {0: 0, 1: 1} # 0=Emp, 1=Occ

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
            
            # Flux = Q_Out - Q_In
            q_out = phys_qns[out_s]
            q_in = phys_qns[in_s]
            flux = q_out - q_in
            
            for q_l in valid_incoming[l]:
                q_r = q_l - flux
                next_nodes.add((r, q_r))
                # Key: (Q_L, Q_R, Q_Phys_Out, Q_Phys_In)
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
            d_new[n] *= -1.0
            overlap = -overlap
        min_overlap = min(min_overlap, overlap)
    return d_new, min_overlap

# ==============================================================================
#  ROBUST (HEAVY) OVERLAP UTILITIES
# ==============================================================================
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

def calculate_overlap_with_hf_robust(mps_tensors, C_mo_spatial, occupied_indices, n_spatial):
    """
    S = < Vac | c_N ... c_1 | Psi_DMRG >
    Robust Method: Applies operators sequentially to the DMRG state.
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

# ==============================================================================
#  CHECKPOINT SAVER
# ==============================================================================
def save_checkpoint(stage_name, d_stack, mps_tensors, energy_dict, mol, params):
    folder = "Checkpoints_nz_48"
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

# ==============================================================================
#  MAIN LOOP
# ==============================================================================
def run_gdvr_dmrg_loop(
    mol, Lz, Nz, basis_cfg,
    pre_opt_cycles=10,      
    dmrg_cycles=3,          
    dmrg_bond_dim=20,
    dmrg_sweeps=10,
    post_dmrg_opt_cycles=5,
    U1 = True
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
        if U1 == True:
            final_H = convert_mpo_symmetric(mpo_dmrg)
        else:
            final_H = mpo_dmrg
        print(f"  3. Running DMRG (D={dmrg_bond_dim})...")
        if last_mps_tensors is None:
            mps_guess = get_entangled_guess(mol.nelec, n_spin)
            # mps_guess = get_noisy_hf_guess(mol.nelec, n_spin, noise=1e-3)
            mps0 = make_u1_random_block_init_guess(L=n_spin, target_qn=mol.nelec, seed=0)
        else:
            mps_guess = [t.copy() for t in last_mps_tensors]
        
        solver = mps_lib.DMRG(final_H, D=dmrg_bond_dim, nsweeps=dmrg_sweeps, init_guess=mps_guess, U1= U1, target_qn=mol.nelec, not_conv_err=False)
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
    # # coords = [[0.0, 2, 2], [0.0, 2, -2], [0.0, -2, -2], [0.0, -2, 2]]
    # # coords = [[0.0, 0.7, 0.7], [0.0, 0.7, -0.7], [0.0, -0.7, -0.7], [0.0, -0.7, 0.7]]
    mol = Molecule(charges, coords, nelec=4)


    # charges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # coords = [[0.0, 0.0, -2], [0.0, 0.0, -6], [0.0, 0.0, -10], [0.0, 0.0, -14], [0.0, 0.0, 2], [0.0, 0.0, 6], [0.0, 0.0, 10], [0.0, 0.0, 14]]
    # mol = Molecule(charges, coords, nelec=8)

    # charges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # coords = [[0.0, 0.0, -2], [0.0, 0.0, -6], [0.0, 0.0, -10], [0.0, 0.0, 2], [0.0, 0.0, 6], [0.0, 0.0, 10]]
    # charges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # coords = [[0.0, 0.0, -2], [0.0, 0.0, -6], [0.0, 0.0, -10], [0.0, 0.0, 2], [0.0, 0.0, 6], [0.0, 0.0, 10]]
    # mol = Molecule(charges, coords, nelec=6)
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    # S_EXPS = [35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280]
    basis_cfg = {'s': S_EXPS}
    
    run_gdvr_dmrg_loop(
        mol, Lz=6, Nz=64, basis_cfg=basis_cfg,
        pre_opt_cycles=10,    
        dmrg_cycles=4,         
        dmrg_bond_dim=40,
        dmrg_sweeps=10,
        post_dmrg_opt_cycles=10,
        U1= True
    )
