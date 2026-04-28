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

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators

# from numba import vectorize, float64, jit
import time
from opt_einsum import contract

from collections import namedtuple
from scipy.sparse import identity, kron, csr_matrix, diags

# from pyqed import Molecule
from pyqed.qchem.mcscf.casci import CASCI, h1e_for_cas
from pyqed.mps import DMRG, MPS, dense_to_symmetric_mpo
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, tensordot, QN, SymmetryManager
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
from collections import defaultdict

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


class SymmetryManager:
    def __init__(self, sym_list):
        if sym_list is True: sym_list = ['charge', 'sz']
        if sym_list is False or sym_list is None: sym_list = []
        self.sym_types = [s.lower() for s in sym_list]
        self.rank = len(self.sym_types)
        self.enabled = self.rank > 0

    def get_vac_qn(self):
        return QN(*[0]*self.rank)

    def get_phys_qn(self, site_idx, state_str):
        """Map physical state ('emp', 'occ') to QN based on active symmetries."""
        vals = []
        for sym in self.sym_types:
            if sym in ['charge', 'n', 'particle']:
                if state_str == 'emp': vals.append(0)
                else: vals.append(1) 
            
            elif sym in ['sz', 'spin', 's_z']:
                # Even=Up(+1), Odd=Down(-1) -> Returns 2*Sz integers
                if state_str == 'emp': 
                    vals.append(0)
                elif state_str == 'occ':
                    if site_idx % 2 == 0: vals.append(1)  # Up
                    else: vals.append(-1) # Down
        return QN(*vals)
    
    def get_target_qn(self, nelec, spin):
        vals = []
        for sym in self.sym_types:
            if sym in ['charge', 'n', 'particle']:
                vals.append(int(nelec))
            elif sym in ['sz', 'spin', 's_z']:
                vals.append(int(spin))
        return QN(*vals)

# Configuration generators helpers for initial guess
# non-normalized in those configs is fine. it is handeled in build_mps_from_configs.
def gen_hf_config(nelec, nsites):
    """Returns HF occupation list [1, 1, ..., 0, 0]"""
    return [1]*nelec + [0]*(nsites - nelec)

def gen_cid_configs(nelec, nsites, mixing=0.1):
    """Returns list of (config, amp) for HF + Doubles"""
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)] # HF gets weight 1.0
    # Simple Double: 2 on HOMO -> 2 on LUMO
    if nelec >= 2 and (nsites - nelec) >= 2:
        dbl = list(hf)
        dbl[nelec-1] = 0; dbl[nelec-2] = 0
        dbl[nelec]   = 1; dbl[nelec+1] = 1
        configs.append((tuple(dbl), mixing))
    return configs

def gen_random_cisd_configs(nelec, nsites, n_states=10, mixing=0.1):
    """Returns HF + Random Singles/Doubles that strictly conserve Sz."""
    # Assuming gen_hf_config returns a list like [1, 1, 1, 1, 0, 0, ...]
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)]

    # Segregate occupied and virtual indices by Spin (Alpha=Even, Beta=Odd)
    occ_alpha = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 0]
    occ_beta  = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 1]
    vir_alpha = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 0]
    vir_beta  = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 1]
    
    for _ in range(n_states):
        new_cfg = list(hf)
        
        # Determine physically valid excitations based on available electrons/holes
        exc_types = []
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1: exc_types.append('S_alpha')
        if len(occ_beta) >= 1 and len(vir_beta) >= 1: exc_types.append('S_beta')
        if len(occ_alpha) >= 2 and len(vir_alpha) >= 2: exc_types.append('D_aa')
        if len(occ_beta) >= 2 and len(vir_beta) >= 2: exc_types.append('D_bb')
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1 and len(occ_beta) >= 1 and len(vir_beta) >= 1: exc_types.append('D_ab')
        
        if not exc_types:
            break # Active space too small for further excitations
            
        choice = np.random.choice(exc_types)
        
        if choice == 'S_alpha':
            i = np.random.choice(occ_alpha); a = np.random.choice(vir_alpha)
            new_cfg[i] = 0; new_cfg[a] = 1
            
        elif choice == 'S_beta':
            i = np.random.choice(occ_beta); a = np.random.choice(vir_beta)
            new_cfg[i] = 0; new_cfg[a] = 1
            
        elif choice == 'D_aa':
            i, j = np.random.choice(occ_alpha, 2, replace=False)
            a, b = np.random.choice(vir_alpha, 2, replace=False)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1
            
        elif choice == 'D_bb':
            i, j = np.random.choice(occ_beta, 2, replace=False)
            a, b = np.random.choice(vir_beta, 2, replace=False)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1
            
        elif choice == 'D_ab':
            # The most important correlation for singlet states
            i = np.random.choice(occ_alpha); a = np.random.choice(vir_alpha)
            j = np.random.choice(occ_beta);  b = np.random.choice(vir_beta)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1
            
        configs.append((tuple(new_cfg), mixing))
        
    return configs

def build_mps_from_configs(configs_with_amps, sym_mgr, nsites, noise_scale=1e-5):
    """
    Constructs an entangled U(1) symmetric MPS from a list of determinant configurations.
    
    Args:
        configs_with_amps: List of tuples (occupation_list, amplitude).
        sym_mgr: SymmetryManager instance.
        nsites: Total number of sites.
        noise_scale: Magnitude of random noise to inject for symmetry breaking.
        
    Returns:
        List[BlockTensor]: The resulting MPS in (Left, Right, Phys) convention.
    """
    # Pre-calculate QN Trajectories for all configurations
    # traj[k] is a list of bond QNs [BoundL, Q1, Q2, ..., BoundR] for config k
    trajectories = []
    vac_qn = sym_mgr.get_vac_qn()

    for cfg, _ in configs_with_amps:
        curr_q = vac_qn
        traj = [curr_q]
        for site_i, occ in enumerate(cfg):
            state_str = 'occ' if occ > 0 else 'emp'
            phys_q = sym_mgr.get_phys_qn(site_i, state_str)
            curr_q = curr_q + phys_q
            traj.append(curr_q)
        trajectories.append(traj)
    mps = []
    # 2. Build Tensors Site by Site
    for i in range(nsites):
        # Grouping Logic 
        # Map QN -> List of configuration indices passing through this sector
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)
        
        for k, _ in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            left_groups[qL].append(k)
            right_groups[qR].append(k)
        # Fill Block Data
        data = {}
        for k, (cfg, amp) in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            state_str = 'occ' if cfg[i] > 0 else 'emp'
            qP = sym_mgr.get_phys_qn(i, state_str)
            key = (qL, qR, qP)
            # Determine Matrix Coordinates (Fan-Out / Fan-In boundaries)
            # At i=0 (Left Boundary), all configs share row 0.
            # At i=L-1 (Right Boundary), all configs share col 0.
            row = 0 if i == 0 else left_groups[qL].index(k)
            col = 0 if i == nsites - 1 else right_groups[qR].index(k)
            
            # Initialize Block if missing
            if key not in data:
                dL = 1 if i == 0 else len(left_groups[qL])
                dR = 1 if i == nsites - 1 else len(right_groups[qR])
                # Phys dimension is always 1 per sector for spin-orbitals
                data[key] = np.zeros((dL, dR, 1), dtype=complex)
            
            # value = Amplitude (only applied at first site) + Noise
            val = amp if i == 0 else 1.0
            noise = (np.random.rand() - 0.5) * noise_scale
            data[key][row, col, 0] += val + noise

        # get basis data
        # Construct flat lists of QNs for the BlockTensor axes
        # Left Bond QNs
        if i == 0:
            final_qns_L = [trajectories[0][0]] # [Vacuum]
        else:
            # Repeat QN 'n' times if 'n' configs pass through it
            final_qns_L = [q for q in sorted(left_groups.keys()) for _ in left_groups[q]]
        # Right Bond QNs
        if i == nsites - 1:
            final_qns_R = [trajectories[0][-1]] # [Target]
        else:
            final_qns_R = [q for q in sorted(right_groups.keys()) for _ in right_groups[q]]
        # Physical QNs (Generic from Manager)
        final_qns_P = [sym_mgr.get_phys_qn(i, 'emp'), sym_mgr.get_phys_qn(i, 'occ')]
        # Create BlockTensor 
        bt = BlockTensor(data, [final_qns_L, final_qns_R, final_qns_P], [-1, 1, 1])
        # Normalize 
        nrm = bt.norm()
        if nrm > 1e-12:
            bt = bt * (1.0 / nrm)
        mps.append(bt)
    return mps


def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates an MPS guess based on filling the first N_elec spin-orbitals.
    used in dense branch,
    Corrected Shape: (Left, Phys, Right) -> (1, d, 1)
    """
    d = 2
    mps_guess = []
    filled_count = 0
    for i in range(n_spin):
        # (Left=1, Phys=d, Right=1)
        vec = np.zeros((1, d, 1)) 
        if filled_count < n_elec:
            vec[0, 1, 0] = 1.0 # Occupied
            filled_count += 1
        else:
            vec[0, 0, 0] = 1.0 # Empty
        # noise 
        rand_noise = (np.random.rand(1, d, 1) - 0.5) * noise
        vec += rand_noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess


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
    def __init__(self, mf, ncas, nelecas, D, init_guess='hf', m_warmup=None,\
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
        

    def get_initial_guess_symmetric(self, method='cid'):
        """
        New Robust Initial Guess Dispatcher.
        """
        method = method.lower()
        nsites = 2 * self.ncas
        
        # Ensure Manager exists (created in run())
        if not hasattr(self, 'sym_mgr'):
            self.sym_mgr = SymmetryManager(['charge', 'sz']) # Default fallback
            
        print(f"  [InitGuess] Generating guess: '{method}' with {self.sym_mgr.sym_types}")

        # 1. Generate Configurations (Physics)
        if method == 'hf':
            hf_cfg = gen_hf_config(self.nelecas, nsites)
            configs = [(hf_cfg, 1.0)]
            
        elif method == 'cid':
            configs = gen_cid_configs(self.nelecas, nsites, mixing=0.5)
            
        elif method == 'cisd' or method == 'random':
            configs = gen_random_cisd_configs(self.nelecas, nsites, n_states=20)
            
        else:
            # Fallback to HF
            print(f"  [Warning] Method {method} not found. Defaulting to HF.")
            hf_cfg = gen_hf_config(self.nelecas, nsites)
            configs = [(hf_cfg, 1.0)]

        # 2. Build Tensor (Math)
        mps = build_mps_from_configs(configs, self.sym_mgr, nsites)
        return mps

    def get_initial_guess_dense(self, noise=1e-3):
        return get_noisy_hf_guess(self.nelecas, 2*self.ncas, noise=noise)

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

    # def fix_spin(self, shift, spin=0, ss = 0):
    #     """
    #     fix the number of electrons by energy penalty

    #     .. math::

    #         \mathcal{H} = H + \lambda (\hat{S}^2 - S(S+1))^2

    #     Parameters
    #     ----------
    #     shift : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # self.h1e += ...
    #     # self.eri += ...
    #     return self

    def fix_spin(self, s=None, ss=0, shift=0.2):
        """
        fix the spin by energy penalty

        .. math::

            H = H + \mu (\hat{S}^2 - S(S+1))

        Parameters
        ----------
        s : TYPE, optional
            DESCRIPTION. The default is None.
        ss : TYPE, optional
            DESCRIPTION. The default is 0.
        shift : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        None.
        """
        if s is None:
            s = (np.sqrt(4*ss+1)-1)/2
            if not np.isclose(2*s, round(2*s)):
                raise Warning("s = {} inconsistent spin value".format(s))
        else:
            if ss is None:
                ss = s * (s+1)
            else:
                raise ValueError('s and ss cannot be specified simultaneously.')

        if ss == 0:
            # first-order spin penalty J. Phys. Chem. A 2022, 126, 12, 2050–2060
            # H' = H + J \hat{S}^2

            self.ss = ss
            self.shift = shift
            self.spin_purification = True

            return self


        else:
            # second-order spin penalty
            raise NotImplementedError('Second-order spin panelty not implemented.')

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

        H, energy_core = h1e_for_cas(self, mo_coeff=self.mo_coeff)

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
        if self.spin_purification:
            J = self.shift
            
            # On-site terms (p == q)
            for p in range(ncas):
                # 3/4 J * (n_{p, up} + n_{p, dn})
                ham_terms.append(Op("n", 2*p) * (0.75 * J))
                ham_terms.append(Op("n", 2*p+1) * (0.75 * J))
                # -3/2 J * n_{p, up} n_{p, dn}
                ham_terms.append(Op("n", 2*p) * Op("n", 2*p+1) * (-1.5 * J))
                
            # Cross-site terms (p != q)
            for p in range(ncas):
                for q in range(ncas):
                    if p == q: continue
                    # S_z^2 
                    ham_terms.append(Op("n", 2*p) * Op("n", 2*q) * (0.25 * J))
                    ham_terms.append(Op("n", 2*p+1) * Op("n", 2*q+1) * (0.25 * J))
                    ham_terms.append(Op("n", 2*p) * Op("n", 2*q+1) * (-0.25 * J))
                    ham_terms.append(Op("n", 2*p+1) * Op("n", 2*q) * (-0.25 * J))
                    # S_+ S_- (Spin Flip)
                    # 1.0 * J * a^+_{p, up} a_{p, dn} a^+_{q, dn} a_{q, up}
                    ham_terms.append(get_jw_term_robust(
                        [r"a^\dagger", "a", r"a^\dagger", "a"],
                        [2*p, 2*p+1, 2*q+1, 2*q], J
                    ))
        basis_sites = [BasisSimpleElectron(i) for i in range(nso)]
        model = Model(basis=basis_sites, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")

        # get it transposed for solver in PyQED: (L, R, P, P) -> (L, P, R, P)
        self.H_raw = mpo.matrices
        H = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        self.H = H

        return self

    def calc_spin_square(self):
        """
        Builds the S^2 MPO and evaluates its expectation value.

        Returns
        -------
        _type_
            _description_
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            return 0.0
            
        import pyqed.mps.mps as mps_lib
        
        ncas = self.ncas
        s2_terms = []
        
        # On-site terms
        for p in range(ncas):
            s2_terms.append(Op("n", 2*p) * 0.75)
            s2_terms.append(Op("n", 2*p+1) * 0.75)
            s2_terms.append(Op("n", 2*p) * Op("n", 2*p+1) * (-1.5))
            
        # Cross-site terms
        for p in range(ncas):
            for q in range(ncas):
                if p == q: continue
                s2_terms.append(Op("n", 2*p) * Op("n", 2*q) * 0.25)
                s2_terms.append(Op("n", 2*p+1) * Op("n", 2*q+1) * 0.25)
                s2_terms.append(Op("n", 2*p) * Op("n", 2*q+1) * (-0.25))
                s2_terms.append(Op("n", 2*p+1) * Op("n", 2*q) * (-0.25))
                s2_terms.append(get_jw_term_robust(
                    [r"a^\dagger", "a", r"a^\dagger", "a"],
                    [2*p, 2*p+1, 2*q+1, 2*q], 1.0
                ))
                
        basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
        model = Model(basis=basis_sites, ham_terms=s2_terms)
        mpo = Mpo(model, algo="qr")
        mpo_dense = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        
        states_to_eval = self.dmrg.states 
        if (hasattr(self.dmrg, 'states') and self.dmrg.states is not None):
            states_to_eval = self.dmrg.states
        else:
            states_to_eval = [self.dmrg.ground_state]
        s2_vals = []
        
        for state in states_to_eval:
            if hasattr(state.Bs[0], 'qns'):
                dense_state = mps_lib.symmetric_to_dense(state)
                psi_for_eval = dense_state.Bs
            else:
                psi_for_eval = state.Bs
                
            s2 = mps_lib.expect_mps(psi_for_eval, mpo_dense, psi_for_eval)
            s2_vals.append(float(np.real(s2)))
            
        return np.array(s2_vals) if self.nstates > 1 else s2_vals[0]

    def run(self, nstates=1, weights=None, symmetry_list=None, nsweeps=50, initial_guess=None, mo_coeff = None, **kwargs):
        """
        Parameters
        ----------
        symmetry_list : list of strings or bool
            ['charge', 'sz'] or True/False.
        """
        self.nstates = nstates
        if weights is None:
            self.weights = np.ones(nstates) / nstates
        else:
            self.weights = np.array(weights)
        if symmetry_list is not None:
            self.saved_symmetry_list = symmetry_list
        else:
            symmetry_list = getattr(self, 'saved_symmetry_list', None)
        if initial_guess is not None:
            self.init_guess = initial_guess
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if self.H_raw is None:
            self.build()
        # Initialize Symmetry 
        self.sym_mgr = SymmetryManager(symmetry_list)
        # Setup site QN, We are still assuming spin-orbitals (d=2) here TODO: make it more tobust
        if self.sym_mgr.enabled:
            print(f"  [Symmetry] Enabled: {self.sym_mgr.sym_types}")
            site_qn_maps = []
            for i in range(self.ncas):
                # even (up) sites
                map_up = {
                    0: self.sym_mgr.get_phys_qn(2*i, 'emp'), 
                    1: self.sym_mgr.get_phys_qn(2*i, 'occ')
                }
                site_qn_maps.append(map_up)
                # Odd (Down) Site
                map_dn = {
                    0: self.sym_mgr.get_phys_qn(2*i+1, 'emp'), 
                    1: self.sym_mgr.get_phys_qn(2*i+1, 'occ')
                }
                site_qn_maps.append(map_dn)
            # get MPO in symmetric form with QN index
            print("  Converting MPO to BlockTensors...")
            final_H = dense_to_symmetric_mpo(self.H, site_qn_maps)
            print(f"  MPO Converted. Sites: {len(final_H)}")
            # Calculate Target QN
            target_qn = self.sym_mgr.get_target_qn(self.nelecas, self.mf.mol.spin)
            print(f"  Target QN set to: {target_qn}")
            # get Initial Guess
            print(f"  Generating Initial Guess ({self.init_guess})...")
            mps0 = self.get_initial_guess_symmetric(method=self.init_guess.lower())
            use_symmetry = True
        else: # dense branch without U(1) symmetry
            final_H = self.H
            mps0 = self.get_initial_guess_dense(noise=1e-3)
            target_qn = None
            use_symmetry = False
            self.sym_mgr = None
        t0 = time.time()
        print(f"  Starting Sweeps (D={self.D})...")
        dmrg = DMRG(final_H, D=self.D, nsweeps=nsweeps, init_guess=mps0, symmetry=use_symmetry, target_qn=target_qn, sym_mgr=self.sym_mgr, not_conv_err=False, nstates=self.nstates, weights=self.weights)
        dmrg.run()
        self.dmrg = dmrg
        # Report
        e_dmrg_total = dmrg.e_tot + self.e_core
        s2_val = self.calc_spin_square()
        if self.spin_purification:
            e_dmrg_total -= self.shift * s2_val
        self.e_tot = e_dmrg_total
        print(f"  RHF Energy:         {self.mf.e_tot:.8f} Ha")
        if self.nstates == 1:
            print(f"  E(DMRG) =           {e_dmrg_total:.8f} Ha")
            print(f"  Correlation Energy = {e_dmrg_total - self.mf.e_tot:.8f} Ha")
            print(f"  <S^2> =             {s2_val:.6f}")
        else:
            for i in range(self.nstates):
                print(f"  Root {i} E(DMRG) = {e_dmrg_total[i]:.8f} Ha")
                print(f"  Root {i} E(DMRG) = {e_dmrg_total[i]:.8f} Ha, <S^2> = {s2_val[i]:.6f}")
        print(f"  Time:               {time.time()-t0:.2f} s")
        if use_symmetry:
            self.check_abelian_symmetry()
        return dmrg

    def dump(self):
        pass

    def check_abelian_symmetry(self):
        """
        Post-run analysis: Checks conservation of all active symmetries 
        (Charge, Sz, etc.) by calculating expectation values via 1-RDMs.
        """
        if self.dmrg.ground_state is None:
            print("  [Error] No ground state found. Run DMRG first.")
            return
        print("\n" + "="*60)
        print("  Symmetry Conservation Check")
        print("="*60)
        # Calculate local site RDMs, returns a dict {site_idx: rho_dense (d,d)}
        try:
            rdms = self.dmrg.make_local_site_rdm()
        except Exception as e:
            print(f"  [Error] Failed to calculate RDM: {e}")
            return
        # initialize storage for quantum number
        total_N_calc = 0.0
        total_Sz_calc = 0.0
        
        print(f"{'Orb':<5} {'Spin':<6} {'Occ':<10} {'Sz_local':<10} {'Status'}")
        print("-" * 60)
        # Iterate over Spatial Orbitals (each splits into 2 Spin-Orbitals)
        # now still assuming d=2 (Spin-Orbital) mapping: 2*i = Up, 2*i+1 = Down
        for i in range(self.ncas):
            # Spin Up Site
            idx_up = 2 * i
            rho_up = rdms[idx_up]
            n_up = rho_up[1, 1].real 
            # Spin Down Site
            idx_dn = 2 * i + 1
            rho_dn = rdms[idx_dn]
            n_dn = rho_dn[1, 1].real
            
            # Charge = N_up + N_dn
            n_local = n_up + n_dn
            # Spin = 1/2 * (N_up - N_dn)
            sz_local = 0.5 * (n_up - n_dn)
            
            total_N_calc += n_local
            total_Sz_calc += sz_local
            # get print nice looking 
            def status(n):
                if n > 0.98: return "Full"
                if n < 0.02: return "."
                return "~" # Entangled
            print(f"{i:<5} {'Up':<6} {n_up:<10.5f} {0.5*n_up:<10.5f} {status(n_up)}")
            print(f"{i:<5} {'Down':<6} {n_dn:<10.5f} {-0.5*n_dn:<10.5f} {status(n_dn)}")
        print("-" * 60)
        # compare with Targets 
        target_qns = self.sym_mgr.get_target_qn(self.nelecas, self.mf.mol.spin)
        print(f"\n  Global Conservation Summary:")
        # iterate over the active symmetries in the manager
        for idx, sym_type in enumerate(self.sym_mgr.sym_types):
            target_val = target_qns[idx]
            if sym_type in ['charge', 'n', 'particle']:
                measured = total_N_calc
                diff = abs(measured - target_val)
                label = "Charge (N)"
            elif sym_type in ['sz', 'spin', 's_z']:
                measured = total_Sz_calc * 2.0 
                diff = abs(measured - target_val)
                label = "Spin (2Sz)"
            else:
                measured = 0.0
                diff = 0.0
                label = f"Unknown ({sym_type})"
            print(f"    {label:<12} : Target={target_val:<8.4f} | Measured={measured:<8.4f} | Diff={diff:.2e} ")

    def make_rdm1(self, state_id=0, spatial=False, with_core=False):
        """
        Calculates the 1-RDM. 
        If spatial=True, spin-traces to the spatial MO basis.
        If with_core=True, re-embeds the frozen core electrons on the diagonal.
        \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>, same as CASCI make_rdm1
        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default False
        with_core : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a state.")
        if hasattr(self.dmrg, 'states') and isinstance(self.dmrg.states, list):
            state = self.dmrg.states[state_id]
        else:
            state = self.dmrg.ground_state
        
        # Get Spin-Orbital RDM
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_state = symmetric_to_dense(state)
            dense_state.dim = 2 
            P_raw = dense_state.make_rdm1()
        else:
            P_raw = state.make_rdm1()
            
        # Convert to Spatial MO basis if requested (or if with_core is True)
        if spatial or with_core:
            ncas = self.ncas
            P_spatial = np.zeros((ncas, ncas), dtype=float)
            for p in range(ncas):
                for q in range(ncas):
                    val = P_raw[2*p, 2*q] + P_raw[2*p+1, 2*q+1]
                    P_spatial[q,p] = float(np.real(val))
            P_out = P_spatial
        else:
            P_out = P_raw

        # Embed Frozen Core for CASSCF optimizations
        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            D = np.zeros((norb, norb), dtype=float)
            if ncore > 0:
                np.fill_diagonal(D[:ncore, :ncore], 2.0)
            D[ncore:norb, ncore:norb] = P_out
            return D
            
        return P_out

    def make_rdm2(self, state_id=0, spatial=False, with_core=False, idx_pairs=None):
        """
        Calculates the 2-RDM.
        If spatial=True, spin-traces to the spatial MO basis.

        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default False
        with_core : bool, optional
            _description_, by default False
        idx_pairs : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a state.")
        if hasattr(self.dmrg, 'states') and isinstance(self.dmrg.states, list):
            state = self.dmrg.states[state_id]
        else:
            state = self.dmrg.ground_state
        
        # Get Spin-Orbital RDM
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_state = symmetric_to_dense(state)
            dense_state.dim = 2 
            G_raw = dense_state.make_rdm2()
        else:
            G_raw = state.make_rdm2()
            
        # Convert to Spatial MO basis if requested
        if spatial or with_core:
            ncas = self.ncas
            D_spatial = np.zeros((ncas, ncas, ncas, ncas), dtype=float)
            for p in range(ncas):
                for q in range(ncas):
                    for r in range(ncas):
                        for s in range(ncas):
                            # p^dag r^dag sq Spatial Convention: dm2[p,q,r,s] = sum_{sig, tau} <p_sig^dag r_tau^dag s_tau q_sig>
                            val = G_raw[2*p,   2*r,   2*s,   2*q] + \
                                  G_raw[2*p,   2*r+1, 2*s+1, 2*q] + \
                                  G_raw[2*p+1, 2*r,   2*s,   2*q+1] + \
                                  G_raw[2*p+1, 2*r+1, 2*s+1, 2*q+1]
                            D_spatial[p, q, r, s] = float(np.real(val))
            G_out = D_spatial
        else:
            G_out = G_raw
            
        # Embed Frozen Core 
        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            D2 = np.zeros((norb, norb, norb, norb), dtype=float)
            if ncore > 0:
                I = np.eye(ncore)
                D2[:ncore, :ncore, :ncore, :ncore] = 4 * np.einsum('ij,kl->ijkl', I, I) - 2 * np.einsum('ps,rq->pqrs', I, I)
                
                dm1 = self.make_rdm1(state_id, spatial=True, with_core=False)
                for i in range(ncore):
                    D2[i, i, ncore:norb, ncore:norb] = 2 * dm1
                    D2[ncore:norb, ncore:norb, i, i] = 2 * dm1
                    D2[i, ncore:norb, i, ncore:norb] = -dm1
                    D2[ncore:norb, i, ncore:norb, i] = -dm1
                    
            D2[ncore:norb, ncore:norb, ncore:norb, ncore:norb] = G_out
            return D2
            
        return G_out

    def make_rdm12(self, state_id=0, spatial=True, with_core=False):
        """
        standard rdm calculator used for SCF

        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default True
        with_core : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        return self.make_rdm1(state_id, spatial, with_core), self.make_rdm2(state_id, spatial, with_core)

    def make_local_site_rdm(self, idx=None):
        """
        Calculate the local reduced density matrices for individual, isolated spin-orbitals.

        This method traces out the rest of the chain to isolate the internal 
        quantum state of specific sites.

        Parameters
        ----------
        idx : int or list of int, optional
            The specific site index (or indices) to evaluate. If None, evaluates 
            the local density matrices for all sites in the active space. 
            By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices (int) to their corresponding 
            local density matrices (numpy.ndarray). For spin-orbitals with a physical 
            dimension `d`, the returned matrix shape is `(d, d)`.

        Raises
        ------
        ValueError
            If the DMRG solver has not been run and no ground state is available.
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
        return self.dmrg.make_local_site_rdm(idx=idx)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the diagonal blocks of the 2-site reduced density matrix.

        Extracts the two-site quantum state :math:`\rho_{ij}` needed to compute 
        density-density correlations (e.g., :math:`\langle n_i n_j \rangle`) without 
        evaluating the full :math:`\mathcal{O}(L^4)` global 2-RDM tensor.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the 2-site RDM for. 
            If None, computes RDMs for all possible unique pairs in the active space. 
            By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding 
            dense reduced density matrix (numpy.ndarray). If the physical dimension 
            of a single site is `d`, the returned matrix shape is `(d*d, d*d)`.

        Raises
        ------
        ValueError
            If the DMRG solver has not been run and no ground state is available.
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
        return self.dmrg.make_diagonal_rdm2(idx_pairs=idx_pairs)


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


    dmrg = QCDMRG(mf, ncas=10, nelecas=6, D=40) #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
    dmrg.build().run(symmetry_list=['charge','sz'], initial_guess='cid')

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