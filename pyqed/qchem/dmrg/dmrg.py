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
from pyqed.mps.autompo.basis import BasisSimpleElectron, BasisSpatialOrbital
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

def get_jw_term_spatial(symbols, sites, value):
    """
    Creates an MPO term for spatial orbitals using exact JW strings.
    Automatically handles intra-site multiplication and inter-site Z-strings.
    
    symbols: list of strings (e.g., [r"a^\dagger_u", "a_u"])
    sites: list of spatial orbital indices (e.g., [p, q])
    value: float or complex weight
    """
    if not symbols:
        return None
        
    min_site = min(sites)
    max_site = max(sites)
    
    term = value
    
    for x in range(min_site, max_site + 1):
        local_op = None
        
        # Multiply the contributions of all operators acting on site x
        for sym, s in zip(symbols, sites):
            op_to_multiply = None
            if x < s:
                op_to_multiply = Op("Z", x)
            elif x == s:
                op_to_multiply = Op(sym, x)
            # if x > s, it contributes nothing
            
            if op_to_multiply is not None:
                if local_op is None:
                    local_op = op_to_multiply
                else:
                    # handles intra-site fermion signs
                    local_op = local_op * op_to_multiply
                    
        if local_op is not None:
            term = term * local_op
            
    return term

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

def build_mps_from_configs(configs_with_amps, site_qn_maps, nsites, noise_scale=1e-5):
    """
    Constructs an entangled U(1) symmetric MPS from a list of determinant configurations.
    
    Args:
        configs_with_amps: List of tuples (occupation_list, amplitude).
        site_qn_maps: List of dicts mapping integer states to QNs for each site.
        nsites: Total number of sites (N for spatial, 2N for spin).
        noise_scale: Magnitude of random noise to inject for symmetry breaking.
        
    Returns:
        List[BlockTensor]: The resulting MPS in (Left, Right, Phys) convention.
    """
    trajectories = []
    
    # Extract the Vacuum QN from the first site's '0' (Empty) state
    vac_qn = site_qn_maps[0][0]
    if isinstance(vac_qn, tuple):
        vac_qn = vac_qn * 0 # Ensure it's a zeroed QN object
    else:
        vac_qn = 0

    # 1. Pre-calculate QN Trajectories
    for cfg, _ in configs_with_amps:
        curr_q = vac_qn
        traj = [curr_q]
        for site_i, occ_state in enumerate(cfg):
            # occ_state is an integer (0, 1 for d=2) or (0, 1, 2, 3 for d=4)
            phys_q = site_qn_maps[site_i][occ_state]
            curr_q = curr_q + phys_q
            traj.append(curr_q)
        trajectories.append(traj)

    mps = []
    
    # 2. Build Tensors Site by Site
    for i in range(nsites):
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)
        
        for k, _ in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            left_groups[qL].append(k)
            right_groups[qR].append(k)
            
        data = {}
        for k, (cfg, amp) in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            occ_state = cfg[i]
            qP = site_qn_maps[i][occ_state]
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

        # Construct flat lists of QNs for the BlockTensor axes
        final_qns_L = [trajectories[0][0]] if i == 0 else [q for q in sorted(left_groups.keys()) for _ in left_groups[q]]
        final_qns_R = [trajectories[0][-1]] if i == nsites - 1 else [q for q in sorted(right_groups.keys()) for _ in right_groups[q]]
        final_qns_P = list(site_qn_maps[i].values()) # All possible physical QNs for this site
        
        bt = BlockTensor(data, [final_qns_L, final_qns_R, final_qns_P], [-1, 1, 1])
        
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
        Robust Initial Guess Dispatcher.
        Handles both d=2 (spin-orbital) and d=4 (spatial-orbital) mappings.
        """
        method = method.lower()
        d_local = self.H[0].shape[2] 
        nsites_spin = 2 * self.ncas # Generators always output 2N spin-orbitals
        
        print(f"  [InitGuess] Generating guess '{method}' for d={d_local}")

        # 1. Generate Configurations (Always 2N length initially)
        if method == 'hf':
            configs = [(gen_hf_config(self.nelecas, nsites_spin), 1.0)]
        elif method == 'cid':
            configs = gen_cid_configs(self.nelecas, nsites_spin, mixing=0.5)
        elif method == 'cisd' or method == 'random':
            configs = gen_random_cisd_configs(self.nelecas, nsites_spin, n_states=20)
        else:
            print(f"  [Warning] Method {method} not found. Defaulting to HF.")
            configs = [(gen_hf_config(self.nelecas, nsites_spin), 1.0)]

        # 2. Compress to Spatial Basis if d=4
        if d_local == 4:
            spatial_configs = []
            for spin_cfg, amp in configs:
                spatial_cfg = []
                for i in range(0, len(spin_cfg), 2):
                    occ_up = spin_cfg[i]      # 0 or 1
                    occ_dn = spin_cfg[i+1]    # 0 or 1
                    # Math maps perfectly: 0(Emp), 1(Up), 2(Dn), 3(Docc)
                    spatial_state = occ_up + (2 * occ_dn) 
                    spatial_cfg.append(spatial_state)
                spatial_configs.append((tuple(spatial_cfg), amp))
                
            configs_to_build = spatial_configs
            nsites_build = self.ncas
        else:
            # Leave as d=2
            configs_to_build = configs
            nsites_build = nsites_spin

        # 3. Build Tensor
        # Pass site_qn_maps instead of sym_mgr so it knows how to translate integers to QNs
        mps = build_mps_from_configs(configs_to_build, self.site_qn_maps, nsites_build)
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

    def build(self, mo_coeff=None, orbital_type='spin'):
        """
        Build the Hamiltonian MPO.
        
        Parameters
        ----------
        mo_coeff : ndarray, optional
            Molecular orbital coefficients. Defaults to self.mf.mo_coeff.
        orbital_type : str, optional
            'spin' for 2N spin-orbital basis (d=2).
            'spatial' for N spatial-orbital basis (d=4).
            Defaults to 'spin'.
        """
        self.orbital_type = orbital_type.lower()
        if self.orbital_type not in ['spin', 'spatial']:
            raise ValueError("orbital_type must be either 'spin' or 'spatial'")

        ncore = self.ncore
        ncas = self.ncas

        # Define the core and active space orbitals
        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]

        # Effective H for CAS
        h1e, eri = self.get_SO_matrix()
        
        if self.orbital_type == 'spin':
            print(f"  System: {ncas} spatial orbitals, {2*ncas} spin-orbitals (2-state basis)")
        else:
            print(f"  System: {ncas} spatial orbitals (4-state basis)")
            
        print("  Building Hamiltonian MPO...")
        
        ham_terms = []
        cutoff = 1e-10
        
        # --- One-Body Terms: h_pq a+_p a_q ---
        for p in range(ncas):
            for q in range(ncas):
                val = h1e[0][p, q]
                if abs(val) > cutoff:
                    if self.orbital_type == 'spin':
                        ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))         # Up
                        ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))     # Down
                    else:
                        ham_terms.append(get_jw_term_spatial([r"a^\dagger_u", "a_u"], [p, q], val))        # Up
                        ham_terms.append(get_jw_term_spatial([r"a^\dagger_d", "a_d"], [p, q], val))        # Down

        # --- Two-Body Terms: 0.5 * (pq|rs) a+_p a+_r a_s a_q ---
        for p in range(ncas):
            for q in range(ncas):
                for r in range(ncas):
                    for s in range(ncas):
                        val = 0.5 * eri[0, 0, p, q, r, s]
                        if abs(val) < cutoff: continue

                        # Same Spin (Pauli Exclusion p!=r and s!=q)
                        if p != r and s != q:
                            if self.orbital_type == 'spin':
                                ham_terms.append(get_jw_term_robust([r"a^\dagger", r"a^\dagger", "a", "a"], [2*p, 2*r, 2*s, 2*q], val))         # Up-Up
                                ham_terms.append(get_jw_term_robust([r"a^\dagger", r"a^\dagger", "a", "a"], [2*p+1, 2*r+1, 2*s+1, 2*q+1], val)) # Dn-Dn
                            else:
                                ham_terms.append(get_jw_term_spatial([r"a^\dagger_u", r"a^\dagger_u", "a_u", "a_u"], [p, r, s, q], val))        # Up-Up
                                ham_terms.append(get_jw_term_spatial([r"a^\dagger_d", r"a^\dagger_d", "a_d", "a_d"], [p, r, s, q], val))        # Dn-Dn

                        # Mixed Spin (No Pauli restriction on spatial indices)
                        if self.orbital_type == 'spin':
                            ham_terms.append(get_jw_term_robust([r"a^\dagger", r"a^\dagger", "a", "a"], [2*p, 2*r+1, 2*s+1, 2*q], val))         # Up-Dn
                            ham_terms.append(get_jw_term_robust([r"a^\dagger", r"a^\dagger", "a", "a"], [2*p+1, 2*r, 2*s, 2*q+1], val))         # Dn-Up
                        else:
                            ham_terms.append(get_jw_term_spatial([r"a^\dagger_u", r"a^\dagger_d", "a_d", "a_u"], [p, r, s, q], val))             # Up-Dn
                            ham_terms.append(get_jw_term_spatial([r"a^\dagger_d", r"a^\dagger_u", "a_u", "a_d"], [p, r, s, q], val))             # Dn-Up
                        
        # --- Spin Purification Terms ---
        if getattr(self, 'spin_purification', False):
            J = self.shift
            
            # On-site terms
            for p in range(ncas):
                if self.orbital_type == 'spin':
                    ham_terms.append(Op("n", 2*p) * (0.75 * J))
                    ham_terms.append(Op("n", 2*p+1) * (0.75 * J))
                    ham_terms.append(Op("n", 2*p) * Op("n", 2*p+1) * (-1.5 * J))
                else:
                    ham_terms.append(Op("n_u", p) * (0.75 * J))
                    ham_terms.append(Op("n_d", p) * (0.75 * J))
                    ham_terms.append(Op("n_u", p) * Op("n_d", p) * (-1.5 * J))
                
            # Cross-site terms (p != q)
            for p in range(ncas):
                for q in range(ncas):
                    if p == q: continue
                    
                    if self.orbital_type == 'spin':
                        # S_z^2 
                        ham_terms.append(Op("n", 2*p) * Op("n", 2*q) * (0.25 * J))
                        ham_terms.append(Op("n", 2*p+1) * Op("n", 2*q+1) * (0.25 * J))
                        ham_terms.append(Op("n", 2*p) * Op("n", 2*q+1) * (-0.25 * J))
                        ham_terms.append(Op("n", 2*p+1) * Op("n", 2*q) * (-0.25 * J))
                        # S_+ S_- (Spin Flip)
                        ham_terms.append(get_jw_term_robust([r"a^\dagger", "a", r"a^\dagger", "a"], [2*p, 2*p+1, 2*q+1, 2*q], J))
                    else:
                        # S_z^2 
                        ham_terms.append(Op("n_u", p) * Op("n_u", q) * (0.25 * J))
                        ham_terms.append(Op("n_d", p) * Op("n_d", q) * (0.25 * J))
                        ham_terms.append(Op("n_u", p) * Op("n_d", q) * (-0.25 * J))
                        ham_terms.append(Op("n_d", p) * Op("n_u", q) * (-0.25 * J))
                        # S_+ S_- (Spin Flip)
                        ham_terms.append(get_jw_term_spatial([r"a^\dagger_u", "a_d", r"a^\dagger_d", "a_u"], [p, p, q, q], J))
                    
        # --- Basis Construction & MPO Assembly ---
        if self.orbital_type == 'spin':
            basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
        else:
            basis_sites = [BasisSpatialOrbital(i) for i in range(ncas)]
            
        model = Model(basis=basis_sites, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")

        # Get it transposed for solver in PyQED: (L, R, P, P) -> (L, P, R, P)
        self.H_raw = mpo.matrices
        self.H = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]

        return self

    def calc_spin_square(self):
        """
        Builds the S^2 MPO and evaluates its expectation value.
        Adapts to spin-orbital (d=2) or spatial-orbital (d=4) basis.
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            return 0.0
            
        import pyqed.mps.mps as mps_lib
        
        ncas = self.ncas
        orbital_type = getattr(self, 'orbital_type', 'spin')
        s2_terms = []
        
        if orbital_type == 'spin':
            # --- Spin-Orbital Basis (d=2, 2N sites) ---
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
            
        else:
            # --- Spatial-Orbital Basis (d=4, N sites) ---
            # On-site terms
            for p in range(ncas):
                s2_terms.append(Op("n_u", p) * 0.75)
                s2_terms.append(Op("n_d", p) * 0.75)
                s2_terms.append(Op("n_u", p) * Op("n_d", p) * (-1.5))
                
            # Cross-site terms
            for p in range(ncas):
                for q in range(ncas):
                    if p == q: continue
                    s2_terms.append(Op("n_u", p) * Op("n_u", q) * 0.25)
                    s2_terms.append(Op("n_d", p) * Op("n_d", q) * 0.25)
                    s2_terms.append(Op("n_u", p) * Op("n_d", q) * (-0.25))
                    s2_terms.append(Op("n_d", p) * Op("n_u", q) * (-0.25))
                    s2_terms.append(get_jw_term_spatial(
                        [r"a^\dagger_u", "a_d", r"a^\dagger_d", "a_u"], 
                        [p, p, q, q], 1.0
                    ))
                    
            basis_sites = [BasisSpatialOrbital(i) for i in range(ncas)]

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
            # RESTORED: Your original working dense conversion
            if hasattr(state.Bs[0], 'qns'):
                dense_state = mps_lib.symmetric_to_dense(state)
                psi_for_eval = dense_state.Bs
            else:
                psi_for_eval = state.Bs
                
            s2 = mps_lib.expect_mps(psi_for_eval, mpo_dense, psi_for_eval)
            s2_vals.append(float(np.real(s2)))
            
        return np.array(s2_vals) if self.nstates > 1 else s2_vals[0]    

    def run(self, nstates=1, weights=None, symmetry_list=None, nsweeps=50, initial_guess=None, mo_coeff=None, site_qn_maps=None, target_qn=None, phys_qns=None, **kwargs):
        """
        Run the DMRG optimization.
        
        Parameters
        ----------
        site_qn_maps : list of dicts, optional
            Explicit mapping of physical indices to QN objects for each site. 
            Overrides auto-generation.
        target_qn : QN, optional
            Explicit target quantum number for the right boundary.
        phys_qns : list of QN, optional
            List of all possible physical QN states in the system.
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

        # Auto-detect local physical dimension from the Hamiltonian
        d_local = self.H[0].shape[2]

        # Determine if symmetry is active
        if symmetry_list or (site_qn_maps is not None) or (target_qn is not None):
            use_symmetry = True
            
            # --- 1. Dynamic Symmetry Mapping ---
            # If the user didn't explicitly provide the mapping, we auto-build it.
            if site_qn_maps is None:
                site_qn_maps = []
                sym_types = [s.lower() for s in (symmetry_list or ['charge', 'sz'])]
                
                # Auto-build Target QN
                if target_qn is None:
                    t_vals = []
                    if 'charge' in sym_types: t_vals.append(int(self.nelecas))
                    if 'sz' in sym_types: t_vals.append(int(self.mf.mol.spin))
                    target_qn = QN(*t_vals)
                    
                # Auto-build maps based on local dimension
                if d_local == 2:
                    print(f"  [Symmetry] Auto-detected d=2 (Spin-Orbitals). Symmetries: {sym_types}")
                    for i in range(self.ncas):
                        # Up site
                        up_vals_emp = [0 for _ in sym_types]
                        up_vals_occ = [1 if sym == 'charge' else (1 if sym == 'sz' else 0) for sym in sym_types]
                        site_qn_maps.append({0: QN(*up_vals_emp), 1: QN(*up_vals_occ)})
                        
                        # Down site
                        dn_vals_emp = [0 for _ in sym_types]
                        dn_vals_occ = [1 if sym == 'charge' else (-1 if sym == 'sz' else 0) for sym in sym_types]
                        site_qn_maps.append({0: QN(*dn_vals_emp), 1: QN(*dn_vals_occ)})
                        
                elif d_local == 4:
                    print(f"  [Symmetry] Auto-detected d=4 (Spatial-Orbitals). Symmetries: {sym_types}")
                    for i in range(self.ncas):
                        vals_emp  = [0 for _ in sym_types]
                        vals_up   = [1 if sym == 'charge' else (1 if sym == 'sz' else 0) for sym in sym_types]
                        vals_dn   = [1 if sym == 'charge' else (-1 if sym == 'sz' else 0) for sym in sym_types]
                        vals_docc = [2 if sym == 'charge' else (0 if sym == 'sz' else 0) for sym in sym_types]
                        site_qn_maps.append({
                            0: QN(*vals_emp),
                            1: QN(*vals_up),
                            2: QN(*vals_dn),
                            3: QN(*vals_docc)
                        })
                else:
                    raise ValueError(f"Unsupported local dimension d={d_local}. Please pass site_qn_maps explicitly.")
            
            # Validation: target_qn is required if taking over manually
            if target_qn is None:
                raise ValueError("target_qn must be provided if explicitly passing site_qn_maps.")
                
            # Collect unique physical QNs while preserving sequence order!
            if phys_qns is None:
                phys_qns = []
                for smap in site_qn_maps:
                    for qn in smap.values():
                        if qn not in phys_qns:
                            phys_qns.append(qn)
                            
            # Save the map to self so the guess generator can use it later
            self.site_qn_maps = site_qn_maps
            
            # --- 2. Initialize the New Algebraic Manager ---
            self.sym_mgr = SymmetryManager(phys_qns=phys_qns, target_qn=target_qn)
            print(f"  [Symmetry] Manager Initialized.")
            print(f"             Target QN: {target_qn}")
            print(f"             Available Phys QNs: {phys_qns}")

            # --- 3. Convert MPO ---
            print("  Converting MPO to BlockTensors...")
            final_H = dense_to_symmetric_mpo(self.H, site_qn_maps)
            print(f"  MPO Converted. Sites: {len(final_H)}")

            # --- 4. Initial Guess ---
            print(f"  Generating Initial Guess ({self.init_guess})...")
            mps0 = self.get_initial_guess_symmetric(method=self.init_guess.lower())
            use_symmetry = True
            
        else: 
            # Dense branch without U(1) symmetry
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
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
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

        total_N_calc = 0.0
        total_Sz_calc = 0.0
        orbital_type = getattr(self, 'orbital_type', 'spin')
        
        print(f"{'Orb':<5} {'Spin':<6} {'Occ':<10} {'Sz_local':<10} {'Status'}")
        print("-" * 60)

        for i in range(self.ncas):
            if orbital_type == 'spin':
                # --- Spin-Orbital Basis (d=2) ---
                idx_up = 2 * i
                n_up = rdms[idx_up][1, 1].real 
                
                idx_dn = 2 * i + 1
                n_dn = rdms[idx_dn][1, 1].real
            else:
                # --- Spatial-Orbital Basis (d=4) ---
                # Basis index: 0=Empty, 1=Up, 2=Down, 3=Doubly Occupied
                rho = rdms[i]
                n_up = rho[1, 1].real + rho[3, 3].real
                n_dn = rho[2, 2].real + rho[3, 3].real
            
            # Charge = N_up + N_dn
            n_local = n_up + n_dn
            # Spin = 1/2 * (N_up - N_dn)
            sz_local = 0.5 * (n_up - n_dn)
            
            total_N_calc += n_local
            total_Sz_calc += sz_local
            
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


    dmrg = QCDMRG(mf, ncas=10, nelecas=6, D=100) #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
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