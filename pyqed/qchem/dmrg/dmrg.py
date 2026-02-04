#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:48:18 2026

Quantum Chemitry DMRG with U(1) particle number Symmetry Support

@author: Shuoyi Hu (hushuoyi@westlake.edu.cn)


"""


import numpy as np
import scipy.constants as const

from scipy.sparse.linalg import eigsh

import logging
import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson

from pyqed import au2ev, au2angstrom

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB
from pyqed.qchem.mcscf.casci import h1e_for_cas

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators

# from numba import vectorize, float64, jit
import time
from opt_einsum import contract

from collections import namedtuple
from scipy.sparse import identity, kron, csr_matrix, diags

from pyqed import Molecule
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.mps.mps import DMRG
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, tensordot
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

#  Fermionic Logic patch adding JW chain
def get_jw_term_robust(op_str_list, indices, factor):
    """
    Constructs a fermionic term with explicit Jordan-Wigner strings (sigma_z)
    and correct sign handling (parity).
    """
    # 1. Canonical Sort: Sort operators by site index
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

    # 2. Insert sigma_z filling (Jordan-Wigner String)
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        # Fill gap between previous site and current site with Z
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        # 3. Handle Creation/Annihilation Phase
        # If we are applying 'a' and there are an odd number of operators to the right, flip sign
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)


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



# initial guess from hf but with added noise to prevenr stuck in hf product state, it happens sometimes
def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates an MPS guess based on filling the first N_elec spin-orbitals,
    but adds small noise to prevent the solver from getting stuck in the HF state.
    """
    d = 2
    mps_guess = []
    filled_count = 0

    for i in range(n_spin):
        vec = np.zeros((d, 1, 1))
        if filled_count < n_elec:
            vec[1, 0, 0] = 1.0; filled_count += 1
        else:
            vec[0, 0, 0] = 1.0

        # Add Noise
        vec += (np.random.rand(d, 1, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)

    return mps_guess



# get initial guess to be |HF> + alpha*|Doubles>
# this is a better guess for U(1) enabled case since that would preserve particle number in the guess
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




def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

# def infinite_system_algorithm(L, m):

#     initial_block = Block(length=1, basis_size=4, operator_dict={
#         "H": H1,
#         "Cu": ops['Cu'],
#         "Cd": ops['Cd'],
#         "Nu": ops['Nu'],
#         "Nd": ops['Nd']
#     })

#     block = initial_block
#     # Repeatedly enlarge the system by performing a single DMRG step, using a
#     # reflection of the current block as the environment.
#     while 2 * block.length < L:
#         print("L =", block.length * 2 + 2)
#         block, energy = single_dmrg_step(block, block, m=m)
#         print("E/L =", energy / (block.length * 2))



class QCDMRG(CASCI):
    """
    ab initio DRMG quantum chemistry calculation
    """
    def __init__(self, mf, ncas, nelecas, D, init_guess='cid', m_warmup=None,\
                 spin=None, tol=1e-6):
        """
        DMRG sweeping algorithm directly using DVR set (without SCF calculations)

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        D : TYPE, optional
            maximum bond dimension. The default is None.
        tol: float
            tolerance for energy convergence

        Returns
        -------
        None.

        """
        # assert(isinstance(mf, RHF1D))

        self.mf = mf

        self.d = 2 # local dimension for spin orbital
        # self.d = 4 # local dimension for spacial orbital

        self.nsites = self.L = ncas

        # assert(mf.eri.shape == (self.L, self.L))

        self.spin_purification = False


        self.D = self.m = D

        self.tol = tol # tolerance for energy convergence
        self.rigid_shift = 0

        if m_warmup is None:
            m_warmup = D
        self.m_warmup = m_warmup


        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        self.nelec = mf.nelec

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        assert(ncore >= 0)


        self.ncore = ncore

        if ncas > 20:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = None
        self.mo_cas = None

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin
        self.shift = None
        self.ss = None

        self.mf = mf
        # self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None
        self.H_raw = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space

        self.spin_purification = False

        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None

        self.init_guess = init_guess
        

    def get_initial_guess(self, method='cid', noise=1e-3):
        """
        Dispatcher for generating U(1) Symmetric Initial Guesses.
        
        Options:
            'hf'       : Pure Hartree-Fock Product State (Bond Dim = 1)
            'hf_noise' : HF with small random noise in valid sectors (Bond Dim = 1)
            'cid'      : HF + Doubles (Bond Dim > 1, Entangled). 
            'random'   : Fully random MPS in target sector (Bond Dim = D)
        """
        method = method.lower()
        print(f"  [InitGuess] Generating U(1) guess: method='{method}' (Target N={self.nelecas})")

        if method == 'hf':
            return self._guess_hf(noise=0.0)
        
        elif method == 'hf_noise':
            return self._guess_hf(noise=noise)
        
        elif method == 'cid':
            return self._guess_cid(mixing_weight=0.5)
        
        elif method == 'cisd':
            
            return self._guess_random_cisd(n_states=10, mixing_weight=0.1)
            
        elif method == 'random':
            return make_u1_random_block_init_guess(
                L=2*self.ncas, 
                target_qn=self.nelecas,
                max_bond_sectors=8
            )
        else:
            raise ValueError(f"Unknown initial guess method: {method}")

    def _guess_hf(self, noise=0.0):
        """
        Constructs a U(1) BlockTensor MPS representing the Hartree-Fock state.
        If noise > 0, adds small random values to the allowed blocks.
        """
        if not SYMMETRY_AVAILABLE: return []
        
        # 1. Define HF Occupation (Spin-Orbitals)
        # First 'nelecas' orbitals are filled (1), rest are empty (0)
        occ_pattern = [1]*self.nelecas + [0]*(2*self.ncas - self.nelecas)
        
        mps = []
        curr_q = 0 # Current charge from left
        
        for site_idx, occ in enumerate(occ_pattern):
            # Physical QN: 0 or 1
            phys_q = occ
            next_q = curr_q + phys_q
            
            # Key: (qL, qR, qP)
            key = (curr_q, next_q, phys_q)
            
            # Create Block (1x1x1 for product state)
            block = np.zeros((1,1,1), dtype=complex)
            block[0,0,0] = 1.0
            
            # Add noise if requested (keeps it in the same sector)
            if noise > 0:
                block[0,0,0] += (np.random.rand() - 0.5) * noise
                block /= np.linalg.norm(block) # Re-normalize
            
            data = {key: block}
            qns_L = [curr_q]
            qns_R = [next_q]
            
            # Create BlockTensor
            bt = BlockTensor(data, [qns_L, qns_R, [0, 1]], [-1, 1, 1])
            mps.append(bt)
            
            curr_q = next_q
            
        return mps

    def _guess_cid(self, mixing_weight=0.1):
        """
        Constructs |HF> + mixing_weight * |Doubles>
        Moves 2 electrons from HOMO/HOMO-1 to LUMO/LUMO+1
        """
        if not SYMMETRY_AVAILABLE: return []
        
        n_sites = 2 * self.ncas
        mps = []
        
        # 1. Define Configurations
        # HF: [1, 1, ..., 1, 0, 0, ...]
        hf_occ = [1]*self.nelecas + [0]*(n_sites - self.nelecas)
        
        # Doubles: Flip last 2 occupied to 0, first 2 empty to 1
        dbl_occ = hf_occ.copy()
        if self.nelecas >= 2 and (n_sites - self.nelecas) >= 2:
            # Annihilate HOMO, HOMO-1
            dbl_occ[self.nelecas-1] = 0
            dbl_occ[self.nelecas-2] = 0
            # Create LUMO, LUMO+1
            dbl_occ[self.nelecas]   = 1
            dbl_occ[self.nelecas+1] = 1
        else:
            print("  [Warning] Active space too small for Doubles. Reverting to HF.")
            return self._guess_hf()

        # 2. Build MPS
        curr_hf = 0
        curr_dbl = 0
        
        for i in range(n_sites):
            data = {}
            
            # HF part
            q_hf_in = curr_hf
            p_hf    = hf_occ[i]
            q_hf_out = q_hf_in + p_hf
            
            key_hf = (q_hf_in, q_hf_out, p_hf)
            if key_hf not in data: data[key_hf] = np.zeros((1,1,1), dtype=complex)
            data[key_hf][0,0,0] += 1.0 # Main weight
            
            # CID part
            q_dbl_in = curr_dbl
            p_dbl    = dbl_occ[i]
            q_dbl_out = q_dbl_in + p_dbl
            
            key_dbl = (q_dbl_in, q_dbl_out, p_dbl)
            if key_dbl not in data: data[key_dbl] = np.zeros((1,1,1), dtype=complex)
            data[key_dbl][0,0,0] += mixing_weight # Perturbation weight
            
            # Collect QNs for metadata
            qns_L = sorted(list(set(k[0] for k in data)))
            qns_R = sorted(list(set(k[1] for k in data)))
            
            bt = BlockTensor(data, [qns_L, qns_R, [0, 1]], [-1, 1, 1])
            # Normalize the site
            bt = bt * (1.0 / bt.norm())
            
            mps.append(bt)
            
            curr_hf = q_hf_out
            curr_dbl = q_dbl_out
            
        return mps


    def _guess_random_cisd(self, n_states=10, mixing_weight=0.1):
        """
        Constructs a superposition: |HF> + w * sum(|Excitations>)
        Fixes boundaries to ensure Dim=1 at edges (Fan-Out / Fan-In).
        """
        if not SYMMETRY_AVAILABLE: return []
        import numpy as np
        from collections import defaultdict

        # 1. Define HF Configuration
        hf_config = [1]*self.nelecas + [0]*(2*self.ncas - self.nelecas)
        
        # 2. Generate Random Excitation Configs
        configs = [tuple(hf_config)] # State 0 is HF
        
        # Identify Occupied and Virtual indices
        occ_idxs = [i for i, x in enumerate(hf_config) if x == 1]
        vir_idxs = [i for i, x in enumerate(hf_config) if x == 0]
        
        for _ in range(n_states):
            new_cfg = list(hf_config)
            
            # Decide: Single (50%) or Double (50%)
            if np.random.rand() > 0.5 and len(occ_idxs)>=2 and len(vir_idxs)>=2:
                i, j = np.random.choice(occ_idxs, 2, replace=False)
                a, b = np.random.choice(vir_idxs, 2, replace=False)
                new_cfg[i] = 0; new_cfg[j] = 0
                new_cfg[a] = 1; new_cfg[b] = 1
            else:
                i = np.random.choice(occ_idxs)
                a = np.random.choice(vir_idxs)
                new_cfg[i] = 0
                new_cfg[a] = 1
            
            configs.append(tuple(new_cfg))

        # 3. Trace Quantum Numbers
        L = len(hf_config)
        bond_qs = [[0] * len(configs)] # Left boundary 0
        
        for i in range(L):
            curr_qs = bond_qs[-1]
            next_qs = []
            for k, cfg in enumerate(configs):
                next_qs.append(curr_qs[k] + cfg[i])
            bond_qs.append(next_qs)

        # 4. Build MPS Tensors
        mps = []
        for i in range(L):
            # Group states by Charge Sectors
            left_groups = defaultdict(list)
            for k, q in enumerate(bond_qs[i]):
                left_groups[q].append(k)
                
            right_groups = defaultdict(list)
            for k, q in enumerate(bond_qs[i+1]):
                right_groups[q].append(k)

            data = {}
            
            for k, cfg in enumerate(configs):
                qL = bond_qs[i][k]
                qR = bond_qs[i+1][k]
                p  = cfg[i]
                
                # fix boundary rows/cols for Dim=1 at edges as needed in dmrg initialization
                # Site 0: All states share input vacuum (row=0)
                if i == 0:
                    row = 0
                else:
                    row = left_groups[qL].index(k)
                    
                # Site L-1: All states share output vacuum (col=0)
                if i == L-1:
                    col = 0
                else:
                    col = right_groups[qR].index(k)
                
                key = (qL, qR, p)
                
                if key not in data:
                    # Determine Block Dimensions
                    if i == 0:   dL = 1
                    else:        dL = len(left_groups[qL])
                    
                    if i == L-1: dR = 1
                    else:        dR = len(right_groups[qR])
                    
                    data[key] = np.zeros((dL, dR, 1), dtype=complex)
                
                # Amplitude
                amp = 1.0 if k == 0 else mixing_weight
                amp += (np.random.rand() - 0.5) * 1e-4
                
                data[key][row, col, 0] = amp

            # Construct QN infromation
            # Correctly handle boundary dims of 1
            if i == 0:
                qns_L = [0]
            else:
                qns_L = []
                for q in sorted(left_groups.keys()):
                    qns_L.extend([q] * len(left_groups[q]))

            if i == L-1:
                # Use the target charge (should be same for all k)
                qns_R = [bond_qs[i+1][0]] 
            else:
                qns_R = []
                for q in sorted(right_groups.keys()):
                    qns_R.extend([q] * len(right_groups[q]))

            bt = BlockTensor(data, [qns_L, qns_R, [0, 1]], [-1, 1, 1])
            bt = bt * (1.0 / bt.norm())
            mps.append(bt)

        print(f"  [InitGuess] Built Randomized CISD guess with {n_states} excitations.")
        return mps


    def fix_nelec(self, shift):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{N} - N)^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    def fix_spin(self, shift, spin=0):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{S}^2 - S(S+1))^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip

        Returns
        -------
        H1: list of [h1e_a, h1e_b]
        H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
        """
        # from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = h1e_for_cas(mf, ncas=self.ncas, ncore=self.ncore)

        self.e_core = energy_core


        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        # H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        # nmo = Ca.shape[1] # n

        eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

        ### compute SO ERIs (MO)
        eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # physicts notation <pq|rs>
        # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

        # eri_aa -= eri_aa.swapaxes(1,3)

        eri_bb = eri_aa.copy()

        eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
        eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




        # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
        #                         compact=False)).reshape((n,n,n,n), order="C")
        # eri_aa -= eri_aa.swapaxes(1,3)

        # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # eri_bb -= eri_bb.swapaxes(1,3)

        # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
        # compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

        # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
                         # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
        H1 = [H, H]

        if spin_flip:
            raise NotImplementedError('Spin-flip matrix elements not implemented yet')
        #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
        #     return H1, H2, H2_SF
        # else:
        #     return H1, H2
        return H1, H2

    def build(self, mo_coeff=None):

        # 1. Extract Integrals & dims
        # mol = mf.mol
        # mf = self.mf
        # if self.ncore == 0:
        #     h1 = mf.get_hcore_mo()
        #     eri = mf.get_eri_mo(notation='chem') # (pq|rs)
        # else:
        #     h1e, eri = self.get_SO_matrix()
        
        # self.nstates = nstates

        # if method == 'ci':

        ncore = self.ncore
        ncas = self.ncas

        # define the core and active space orbitals
        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff # use HF MOs
        else:
            self.mo_coeff = mo_coeff

        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]


        # effective H for CAS
        h1e, eri = self.get_SO_matrix()
        
        if self.spin_purification:

            logging.info('Purify spin by energy penalty')

            # if self.shift is not None:
            # H1, H2 = self.fix_spin(H1, H2, ss=ss, shift=shift)
            shift = self.shift

            norb = self.ncas
            h1e = [h + 3./4 * shift * np.eye(norb) for h in h1e]

            for p in range(norb):
                for q in range(norb):
                    eri[:, :, p, q, q, p] -=  0.5 * shift * 2
                    eri[:, :, p, p, q, q] -= 0.25 * shift * 2

        # h2e[0,0] -= h2e[0,0].swapaxes(1,3)
        # h2e[1,1] -= h2e[1,1].swapaxes(1,3)
        

        n_spatial = self.ncas
        nso = 2 * n_spatial
        print(f"  System: {n_spatial} spatial orbitals, {nso} spin-orbitals")

        # 2. Build Hamiltonian (Using Robust JW Builder)
        print("  Building Hamiltonian MPO...")
        ham_terms = []
        cutoff = 1e-10
        # --- One-Body Terms: h_pq a+_p a_q ---
        for p in range(ncas):
            for q in range(ncas):
                val = h1e[0][p, q]
                if abs(val) > cutoff:
                    # Spin Up (Indices 2p, 2q)
                    ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))
                    # Spin Down (Indices 2p+1, 2q+1)
                    ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))

        # --- Two-Body Terms: 0.5 * (pq|rs) a+_p a+_r a_s a_q ---
        for p in range(ncas):
            for q in range(ncas):
                for r in range(ncas):
                    for s in range(ncas):
                        val = 0.5 * eri[0, 0, p, q, r, s]
                        if abs(val) < cutoff: continue

                        # p,r creation; s,q annihilation

                        # Same Spin (Pauli Exclusion p!=r)
                        if p != r and s != q:
                            # Up-Up
                            ham_terms.append(get_jw_term_robust(
                                [r"a^\dagger", r"a^\dagger", "a", "a"],
                                [2*p, 2*r, 2*s, 2*q], val
                            ))
                            # Dn-Dn
                            ham_terms.append(get_jw_term_robust(
                                [r"a^\dagger", r"a^\dagger", "a", "a"],
                                [2*p+1, 2*r+1, 2*s+1, 2*q+1], val
                            ))

                        # Mixed Spin (No Pauli restriction on spatial indices)
                        # Up-Dn (p Up, r Dn, s Dn, q Up)
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p, 2*r+1, 2*s+1, 2*q], val
                        ))
                        # Dn-Up (p Dn, r Up, s Up, q Dn)
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p+1, 2*r, 2*s, 2*q+1], val
                        ))

        # 3. Generate MPO
        basis_sites = [BasisSimpleElectron(i) for i in range(nso)]
        model = Model(basis=basis_sites, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")

        # get it transposed for solver in PyQED: (L, R, P, P) -> (L, P, R, P)
        self.H_raw = mpo.matrices
        H = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        self.H = H

        return self

    def run(self, symmetry=False, nsweeps=50):
        """
        Only support Abelian symmetry

        Parameters
        ----------
        symmetry : TYPE, optional
            DESCRIPTION. The default is False.
        nsweeps : TYPE, optional
            DESCRIPTION. The default is 50.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        dmrg : TYPE
            DESCRIPTION.

        """

        if self.H_raw is None:
            self.build()
        final_H = self.H
        
        # Initial_guess_NOISE    = 1e-3

        # get mpo and mps initial guess
        # mpo_dmrg = qc_dmrg_mpo(mf)
        if symmetry:
            print("  Converting MPO to U(1) Blocks (0=Emp, 1=Occ)...")
            H_input = [w.transpose(0, 3, 1, 2) for w in self.H_raw]
            final_H = convert_mpo_symmetric(H_input)

            print("  Generating Initial Guess")

            # if self.target_qn != self.nelec:
                
            #     if self.init_guess == 'hf':
            #         mps0 = make_u1_random_block_init_guess(2*self.ncas, target_qn=self.target_qn)
            # else:
            
            # initial guess 
            # init_guess = self.init_guess.lower()
            # if init_guess == 'hf':
            #     pass
            # elif init_guess == 'cid':
            #     mps0 = get_entangled_guess(self.nelecas, 2*self.ncas)
            # elif init_guess == 'idmrg':
            #     pass
            # else:
            #     raise NotImplementedError('Only CID is supported.')

            init_guess = self.init_guess.lower()
            mps0 = self.get_initial_guess(method=init_guess)
        else:
            print("  Running Dense DMRG...")
            final_H = self.H

            
            mps0 = get_noisy_hf_guess(self.nelecas, 2*self.ncas, noise=1e-3)

        #     mps0 = get_noisy_hf_guess(mol.nelec, 2*self.ncas, noise=Initial_guess_NOISE)


        t0 = time.time()
        # print(x.shape for x in final_H)  ## for debug only
        # run dmrg!
        print(f"  Starting Sweeps (D={self.D})...")
        dmrg = DMRG(final_H, D=self.D, nsweeps=nsweeps, init_guess=mps0, U1=symmetry, target_qn=self.nelecas, not_conv_err=False)


        dmrg.run()
        self.dmrg = dmrg
        if symmetry:
            self.check_u1_symmetry()


        if not dmrg.converged and self.init_guess == 'hf':
            print('Consider using initial guess with bond dimension > 1.')

        # 6. Report result
        e_dmrg_total = dmrg.e_tot + self.e_core

        print('final number of electron is ',dmrg.nelec_dmrg())
        print(f"  RHF Energy:         {self.mf.e_tot:.8f} Ha")
        print(f"  E(DMRG) =  {e_dmrg_total:.8f} Ha")
        print(f"  Correlation Energy = {e_dmrg_total - self.mf.e_tot:.8f} Ha")
        print(f"  Time:               {time.time()-t0:.2f} s")

        return dmrg


    def dump(self):
        pass


    def check_u1_symmetry(self):
        """
        Post-run analysis: Checks particle number conservation and prints 
        the orbital occupancy profile in a Quantum Chemistry context.
        """
        if self.dmrg.ground_state is None:
            print("  [Error] No ground state found. Run DMRG first.")
            return

        print("\n" + "="*45)
        print("  U(1) Symmetry Check & Occupancy Profile")
        print("="*45)

        # 1. Get Occupations from DMRG (returns dict {site_idx: n_val})
        # This automatically handles the trace logic for you.
        total_N_measured, local_occs = self.dmrg.nelec_dmrg(return_local=True)
        
        # 2. Print Formatted Table
        print(f"{'Orb':<5} {'Spin':<6} {'Occ':<10} {'Status'}")
        print("-" * 40)
        
        # Iterate through spatial orbitals
        # self.ncas is the number of spatial orbitals
        for i in range(self.ncas): 
            # Map spatial index 'i' to spin-orbital indices
            idx_up = 2 * i
            idx_dn = 2 * i + 1
            
            n_up = local_occs.get(idx_up, 0.0)
            n_dn = local_occs.get(idx_dn, 0.0)
            
            # Helper to generate status bar
            def get_status(n):
                if n > 0.98: return "Full"
                if n < 0.02: return "."
                return "~" # Entangled

            print(f"{i:<5} {'Up':<6} {n_up:<10.5f} {get_status(n_up)}")
            print(f"{i:<5} {'Down':<6} {n_dn:<10.5f} {get_status(n_dn)}")
            
        print("-" * 40)
        
        # 3. Validation Logic
        print(f"  Target N: {self.nelecas:.5f}")
        print(f"  Actual N: {total_N_measured:.5f}")
        
        err = abs(total_N_measured - self.nelecas)
        if err < 1e-5:
             print("  [PASS] Particle number conserved.")
        else:
             print(f"  [FAIL] Particle conservation broken! Error: {err:.2e}")
             print("         (Check your MPO definitions or Initial Guess)")
        print("="*45 + "\n")



class DMRGSCF(QCDMRG):
    """
    optimize the orbitals
    """
    pass


if __name__=='__main__':

    from pyqed.qchem.mcscf.direct_ci import CASCI

    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


    from pyqed.qchem.mol import atomic_chain
    
    natom = 6
    z = np.linspace(-6, 6, natom)
    mol = atomic_chain(natom, z)

    # mol.basis = 'aug-ccpvdz'
    mol.basis = 'ccpvdz'
    mol.build(driver='pyscf')

    mf = mol.RHF().run()


    dmrg = QCDMRG(mf, ncas=12, nelecas=4, D=60, init_guess='cisd') #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
    dmrg.build().run(symmetry=True)

    # mc = CASCI(mf, ncas=8, nelecas=4)
    # mc.run()

    # conn refers to the connection operator, that is, the operator on the edge of
    # the block, on the interior of the chain.  We need to be able to represent S^z
    # and S^+ on that site in the current basis in order to grow the chain.
    # initial_block = Block(length=1, basis_size=model_d, operator_dict={
    #     "H": H1,
    #     "Cu": ops['Cu'],
    #     "Cd": ops['Cd'],
    #     "Nu": ops['Nu'],
    #     "Nd": ops['Nd']
    # })

    #infinite_system_algorithm(L=100, m=20)
    # finite_system_algorithm(L=nsites, m_warmup=10, m=10)