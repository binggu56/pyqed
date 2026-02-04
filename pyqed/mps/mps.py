#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:41:46 2024

#####################################################

main DMRG module using MPS/MPO representations

ground state optimization

time-evolving block decimation

# Ian McCulloch August 2017                         #
#####################################################


@author: Bing Gu
"""



import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.sparse as sparse
import math
from copy import deepcopy
from scipy.sparse.linalg import eigsh #Lanczos diagonalization for hermitian matrices

# from pyqed.mps.mps import LeftCanonical, RightCanonical, ZipperLeft, ZipperRight
from pyqed.mps.decompose import decompose, compress
try:
    from pyqed.mps.symmetry import BlockTensor, tensordot, solve_davidson
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
from scipy.linalg import expm, block_diag
import warnings

def SpinHalfFermionOperators(filling=1.):
    d = 4
    states = ['empty', 'up', 'down', 'full']
    # 0) Build the operators.
    Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
    Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)

    Nu = np.diag(Nu_diag)
    Nd = np.diag(Nd_diag)
    Ntot = np.diag(Nu_diag + Nd_diag)
    dN = np.diag(Nu_diag + Nd_diag - filling)
    NuNd = np.diag(Nu_diag * Nd_diag)
    JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
    JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
    JW = JWu * JWd  # (-1)^{Nu+Nd}


    Cu = np.zeros((d, d))
    Cu[0, 1] = Cu[2, 3] = 1
    Cdu = np.transpose(Cu)
    # For spin-down annihilation operator: include a Jordan-Wigner string JWu
    # this ensures that Cdu.Cd = - Cd.Cdu
    # c.f. the chapter on the Jordan-Wigner trafo in the userguide
    Cd_noJW = np.zeros((d, d))
    Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
    Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
    Cdd = np.transpose(Cd)

    # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
    # where S^gamma is the 2x2 matrix for spin-half
    Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
    Sp = np.dot(Cdu, Cd)
    Sm = np.dot(Cdd, Cu)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    ops = dict(JW=JW, JWu=JWu, JWd=JWd,
               Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
               Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
               Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable
    return ops

def svd_symmetric(AA, cutoff=1e-10, m_max=None):

    AA_perm = AA.transpose(0, 2, 1, 3)

    # blocks_by_q_mid is used to combine different (q_L, q_phys_L) pair that actually share same q_tot_L, which would belong to same sub-block after svd
    blocks_by_q_mid = {}
    row_map = {}
    col_map = {}

    for qn_tuple, block in AA_perm.data.items():
        q_L, q_phys_L, q_R, q_phys_R = qn_tuple

        # charge flow
        q_mid = q_L + q_phys_L

        blocks_by_q_mid.setdefault(q_mid, [])
        row_map.setdefault(q_mid, set())
        col_map.setdefault(q_mid, set())

        blocks_by_q_mid[q_mid].append((qn_tuple, block))
        row_map[q_mid].add((q_L, q_phys_L))
        col_map[q_mid].add((q_R, q_phys_R))

    # Storage
    sv_list = []   # (s, q_mid, local_index)
    U_store = {}
    V_store = {}
    
    # Store S blocks temporarily
    S_store = {} 

    for q_mid, entries in blocks_by_q_mid.items():
        rows = sorted(row_map[q_mid])
        cols = sorted(col_map[q_mid])

        r_starts, c_starts = {}, {}
        r_dim = c_dim = 0

        for r in rows:
            for qn, blk in entries:
                if (qn[0], qn[1]) == r:
                    r_starts[r] = r_dim
                    r_dim += blk.shape[0] * blk.shape[1]
                    break

        for c in cols:
            for qn, blk in entries:
                if (qn[2], qn[3]) == c:
                    c_starts[c] = c_dim
                    c_dim += blk.shape[2] * blk.shape[3]
                    break

        M = np.zeros((r_dim, c_dim), dtype=entries[0][1].dtype)

        for qn, blk in entries:
            r0 = r_starts[(qn[0], qn[1])]
            c0 = c_starts[(qn[2], qn[3])]
            M[r0:r0+blk.shape[0]*blk.shape[1],
              c0:c0+blk.shape[2]*blk.shape[3]] = blk.reshape(
                  blk.shape[0]*blk.shape[1],
                  blk.shape[2]*blk.shape[3]
              )

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        S_store[q_mid] = S 

        for i, s in enumerate(S):
            sv_list.append((s, q_mid, i))

        U_store[q_mid] = (U, rows, r_starts, entries)
        V_store[q_mid] = (Vt, cols, c_starts, entries)

    # GLOBAL truncation with bookkeeping
    sv_list.sort(reverse=True, key=lambda x: x[0])
    
    # Calculate truncation error
    full_sq_norm = sum(s**2 for s, _, _ in sv_list)
    
    if m_max is not None:
        sv_list = sv_list[:m_max]
        
    trunc_err = 0.0
    if full_sq_norm > 1e-12:
        trunc_err = 1.0 - sum(s**2 for s, _, _ in sv_list) / full_sq_norm

    kept = {}
    for s, q_mid, i in sv_list:
        kept.setdefault(q_mid, []).append(i)

    final_U = {}
    final_V = {}
    final_S = {} # Dictionary to store S diagonal matrices

    for q_mid, idxs in kept.items():
        idxs = sorted(idxs) # Ensure indices are sorted
        U, rows, r_starts, entries = U_store[q_mid]
        Vt, cols, c_starts, entries = V_store[q_mid]
        
        S_block = S_store[q_mid][idxs]
        final_S[q_mid] = np.diag(S_block)

        for r in rows:
            for qn, blk in entries:
                if (qn[0], qn[1]) == r:
                    d1, d2 = blk.shape[0], blk.shape[1]
                    break
            r0 = r_starts[r]
            ublk = U[r0:r0+d1*d2, idxs].reshape(d1, d2, len(idxs))
            final_U[(r[0], r[1], q_mid)] = ublk

        for c in cols:
            for qn, blk in entries:
                if (qn[2], qn[3]) == c:
                    d3, d4 = blk.shape[2], blk.shape[3]
                    break
            c0 = c_starts[c]
            vblk = Vt[idxs, c0:c0+d3*d4].reshape(len(idxs), d3, d4)
            final_V[(q_mid, c[0], c[1])] = vblk

    bond_qns = []
    for q_mid, idxs in kept.items():
        bond_qns.extend([q_mid] * len(idxs))

    # Original qns from AA
    qns_L      = AA_perm.qns[0]
    qns_pL     = AA_perm.qns[1]
    qns_R      = AA_perm.qns[2]
    qns_pR     = AA_perm.qns[3]

    # ----------------------------
    # Construct tensors
    # ----------------------------
    U = BlockTensor(
        final_U,
        qns=[qns_L, qns_pL, bond_qns],
        dirs=[AA_perm.dirs[0], AA_perm.dirs[1], 1]
    )

    V = BlockTensor(
        final_V,
        qns=[bond_qns, qns_R, qns_pR],
        dirs=[-1, AA_perm.dirs[2], AA_perm.dirs[3]]
    )
    
    # Return S as a dictionary of arrays {q_mid: diagonal_matrix}
    return U, V, final_S, trunc_err, sum(len(v) for v in kept.values())



class HamiltonianMultiplyU1:
    """
    Symmetric version of HamiltonianMultiply using BlockTensor.
    """
    def __init__(self, E, W, F):
        self.E = E
        self.W = W
        self.F = F
        self.dtype = np.float64 

    def matvec(self, A):
        # A is BlockTensor with indices (Left, Right, Phys_L, Phys_R)
        # E: (MPO_L, MPS_L, MPS_L')
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)
        # F: (MPO_R, MPS_R, MPS_R')
        
        # 1. Contract E with A
        # E indices: (a, i, j) -> (MPO, Bra, Ket)
        # A indices: (j, k, s1, s2) -> (Left, Right, PhysL, PhysR)
        # Contract E[Ket] with A[Left] -> E[2] with A[0]
        # Result R: (a, i, k, s1, s2)
        R = tensordot(self.E, A, axes=([2], [0]))
        
        # 2. Contract R with W1 (Left Site)
        # W1: (a, b, s1', s1) -> (Left, Right, Out, In)
        # R: (a, i, k, s1, s2)
        # Contract R[MPO_L]=R[0] with W1[Left]=W1[0]
        # Contract R[Phys1]=R[3] with W1[In]=W1[3]
        T2 = tensordot(R, self.W[0], axes=([0, 3], [0, 3]))
        # T2: (i, k, s2, b, s1') -> (Bra_L, Right, PhysR, MPO_R, PhysL_Out)
        
        # 3. Contract T2 with W2 (Right Site)
        # W2: (b, c, s2', s2) -> (Left, Right, Out, In)
        # T2: (i, k, s2, b, s1')
        # Contract T2[MPO_R]=T2[3] with W2[Left]=W2[0]
        # Contract T2[PhysR]=T2[2] with W2[In]=W2[3]
        T3 = tensordot(T2, self.W[1], axes=([3, 2], [0, 3]))
        # T3: (i, k, s1', c, s2') -> (Bra_L, Right, PhysL_Out, MPO_R, PhysR_Out)
        
        # 4. Contract T3 with F
        # F: (c, k, l) -> (MPO_R, Bra_R, Ket_R)
        # contract T3[Right]=T3[1] (which corresponds to A's Right/Ket) 
        # with F[Ket]=F[2].
        # And T3[MPO_R]=T3[3] with F[MPO]=F[0].
        T4 = tensordot(T3, self.F, axes=([3, 1], [0, 2])) 
        # Result indices: (i, s1', s2', l) -> (Bra_L, PhysL_Out, PhysR_Out, Bra_R)
        
        # 5. Transpose to match A structure (Left, Right, PhysL, PhysR)
        # Current: (Bra_L, PhysL, PhysR, Bra_R) -> (0, 1, 2, 3)
        # Target: (Bra_L, Bra_R, PhysL, PhysR) -> (0, 3, 1, 2)
        A_new = T4.transpose(0, 3, 1, 2)
        
        return A_new



class MPS:
    def __init__(self, Bs, Ss=None, homogenous=True, bc='finite', form="B"):
        """
        class for matrix product states.

        Parameters
        ----------
        mps : list
            list of 3-tensors.

        Returns
        -------
        None.

        """
        assert bc in ['finite', 'infinite']
        self.Bs = self.factors = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'open' else self.L
        self.gauge = None

        self.data = self.factors = Bs
        
        # --- ROBUST DIM CALCULATION FOR BLOCKTENSORS ---
        if homogenous:
            try:
                self.dim = Bs[0].shape[1]
            except TypeError:
                if hasattr(Bs[0], 'data'):
                    # U(1) tensors in this code are (Left, Right, Phys) -> Index 2
                    phys_dims = {}
                    for key, block in Bs[0].data.items():
                        # key is (qL, qR, qP)
                        q_p = key[2]  
                        if q_p not in phys_dims:
                            phys_dims[q_p] = block.shape[2]
                    self.dim = sum(phys_dims.values())
                else:
                    self.dim = 0 # Should not happen
        else:
            # Similar logic for inhomogeneous
            self.dims = []
            for B in Bs:
                try:
                    self.dims.append(B.shape[1])
                except TypeError:
                    if hasattr(B, 'data'):
                        phys_dims = {}
                        for key, block in B.data.items():
                            q_p = key[1]
                            if q_p not in phys_dims:
                                phys_dims[q_p] = block.shape[1]
                        self.dims.append(sum(phys_dims.values()))
                    else:
                        self.dims.append(0)

        # self._mpo = None

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss], self.bc)

    def get_bond_dimensions(self):
        """
        Return bond dimensions.
        """
        try:
            return [self.Bs[i].shape[2] for i in range(self.nbonds)]
        except TypeError:
             # Fallback for BlockTensor bond dims (Axis 2)
             bonds = []
             for i in range(self.nbonds):
                 B = self.Bs[i]
                 bond_dims = {}
                 for key, block in B.data.items():
                     q_r = key[2]
                     if q_r not in bond_dims:
                         bond_dims[q_r] = block.shape[2]
                 bonds.append(sum(bond_dims.values()))
             return bonds

    # def decompose(self, chi_max):
    #     pass

    def __add__(self, other):
        assert len(self.data) == len(other.data)
        # for different length, we should choose the maximum one
        C = []
        for j in range(self.sites):
            tmp = block_diag(self.data[j], other.data[j])
            C.append(tmp.copy())

        return MPS(C)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=(1, 1))  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        B = self.Bs[0]  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self.Bs[i]
            T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
            # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
        T = np.reshape(T, (chi**2, chi**2))
        # Obtain the 2nd largest eigenvalue
        eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi =  -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i) # vL i vR
        C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
        C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self.Bs[j]  # vL k vR
        C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
        C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
        C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
        return C

    def evolve_v(self, other):
        """
        apply the evolution operator due to V(R) to the wavefunction in the TT format

                    |   |
                ---V---V---
                    |   |
                    |   |
                ---A---A---
            =
                    |   |
                ===B===B===

        .. math::

            U_{\beta_i \beta_{i+1}}^{j_i} A_{\alpha_i \alpha_{i+1}}^{j_i} =
            A^{j_i}_{\beta_i \alpha_i, \beta_{i+1} \alpha_{i+1}}

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        MPS object.

        """
        assert(other.L == self.L)
        assert(other.dims == self.dims)

        As = []
        for n in range(self.L):

            al, d, ar = self.factors[n].shape
            bl, d, br = other.factors[n].shape

            c = np.einsum('aib, cid -> acibd', other.factors[n], self.factors[n])
            c.reshape((al * bl, d, ar * br))
            As.append(c.copy())

        return MPS(As)

    # def __add__(self, other):
    #     pass

    # def evolve_t(self):
    #     pass

    def left_canonicalize(self):
        pass

    def right_canonicalize(self):
        pass

    def left_to_vidal(self):
        pass

    def left_to_right(self):
        pass

    # def build_U_mpo(self):
    #     # build MPO representation of the short-time propagator
    #     pass

    # # def run(self, dt=0.1, Nt=10):
    # #     pass

    # # def obs_local(self, e_op, n):
    # #     pass

    # def apply_mpo(self):
    #     pass

    def compress(self, chi_max):
        return MPS(compress(self.factors, chi_max)[0])
    
    def calc_1site_rdm(self, idx=None):
        """
        Calculate 1-site reduced density matrices.

        Dense (numpy) MPS path: uses the existing implementation assuming tensors
        are ordered as (phys, chi_L, chi_R).

        U(1) (BlockTensor) path: builds left/right overlap environments using the
        same contraction logic as the DMRG sweeps (contract_from_left/right),
        but leaves the physical indices at the target site open.
        Returns *dense* (numpy) d×d matrices for convenience.
        """
        import numpy as np

        if idx is None:
            idx = list(range(self.L))
        elif isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, (list, tuple)):
            idx = list(idx)
        else:
            raise ValueError("idx must be None, int, list, or tuple")

        if self.L == 0:
            return {}

        # U(1) BlockTensor way of 1-rdm calculation
        if SYMMETRY_AVAILABLE and isinstance(self.Bs[0], BlockTensor):

            def _make_id_mpo_from_phys_qns(phys_qns):
                # phys_qns is a list with length d (may contain duplicates for degeneracy)
                from collections import defaultdict
                idxs_by_q = defaultdict(list)
                for k, q in enumerate(list(phys_qns)):
                    idxs_by_q[q].append(k)

                data = {}
                for q, ks in idxs_by_q.items():
                    d = len(ks)
                    # block for (L=0,R=0,Out=q,In=q): shape (1,1,d,d)
                    data[(0, 0, q, q)] = np.eye(d).reshape(1, 1, d, d)

                qns = [[0], [0], list(phys_qns), list(phys_qns)]
                # Conventional MPO directions (bondL, bondR, out, in)
                dirs = [1, -1, 1, -1]
                return BlockTensor(data, qns, dirs)

            def _blockmat_to_dense(rho_bt, phys_qns):
                # rho_bt is a 2-index BlockTensor over physical indices
                d = len(phys_qns)
                out = np.zeros((d, d), dtype=complex)

                # map q -> positions in the physical basis ordering
                from collections import defaultdict
                pos = defaultdict(list)
                for k, q in enumerate(list(phys_qns)):
                    pos[q].append(k)

                for (q0, q1), blk in rho_bt.data.items():
                    rows = pos[q0]
                    cols = pos[q1]
                    # blk is (len(rows), len(cols))
                    for a, ra in enumerate(rows):
                        for b, cb in enumerate(cols):
                            out[ra, cb] = blk[a, b]
                return out

            # Build an identity MPO list (bond dims 1) just to trace out other sites correctly.
            W_id = []
            for s in range(self.L):
                # physical qns live on index 2 for MPS tensors in this code path (L,R,Phys)
                phys_qns = self.Bs[s].qns[2]
                W_id.append(_make_id_mpo_from_phys_qns(phys_qns))

            # Build left overlap environments E[s] for bond left of site s
            E = [None] * self.L
            E[0] = initial_E(W_id[0])
            for s in range(0, self.L - 1):
                E[s + 1] = contract_from_left(W_id[s], self.Bs[s], E[s], self.Bs[s])

            # Build right overlap environments R[s] for bond right of site s
            R = [None] * self.L
            qs = sorted({key[1] for key in self.Bs[-1].data.keys()})
            if len(qs) != 1:
                raise ValueError(f"Ambiguous total charge on last bond: {qs}.")
            target_qn = qs[0]
            R[-1] = initial_F(W_id[-1], target_qn=target_qn)
            for s in range(self.L - 1, 0, -1):
                R[s - 1] = contract_from_right(W_id[s], self.Bs[s], R[s], self.Bs[s])

            rdm = {}
            for s in idx:
                # L: (wL, bra_L, ket_L),  B: (ket_L, ket_R, phys)
                t1 = tensordot(E[s], self.Bs[s], axes=([2], [0]))          # (wL, bra_L, ket_R, phys)
                # R: (wR, bra_R, ket_R)
                t2 = tensordot(t1, R[s], axes=([2], [2]))                  # (wL, bra_L, phys, wR, bra_R)
                # B*: (bra_L, bra_R, phys')
                rho4 = tensordot(t2, self.Bs[s].conj(), axes=([1, 4], [0, 1]))  # (wL, phys, wR, phys')

                # Squeeze singleton MPO bond dims (they are always 1 here)
                data2 = {}
                for (qwL, qP, qwR, qPp), blk in rho4.data.items():
                    # blk shape (1, dP, 1, dPp) -> (dP, dPp)
                    data2[(qP, qPp)] = blk.reshape(blk.shape[1], blk.shape[3])
                phys_qns = self.Bs[s].qns[2]
                rho2 = BlockTensor(data2, [list(phys_qns), list(phys_qns)], [1, -1])

                rho_dense = _blockmat_to_dense(rho2, phys_qns)
                tr = np.trace(rho_dense)
                if abs(tr) > 0:
                    rho_dense = rho_dense / tr   # enforce Tr(rho)=1
                rdm[s] = rho_dense

            return rdm

        # 1-rdm calculation without U(1)
        # 1. Build Left Environments (L_env[i] is contraction of 0...i-1)
        L_env = [np.array([[1.0]])]
        curr_L = L_env[0]
        for i in range(self.L - 1):
            # L(bra_L,ket_L) * B(p,ket_L,ket_R) -> temp(bra_L,p,ket_R)
            temp = np.tensordot(curr_L, self.Bs[i], axes=(1, 1))
            # temp(bra_L,p,ket_R) * B*(p,bra_L,bra_R) -> curr_L(ket_R,bra_R)
            curr_L = np.tensordot(temp, self.Bs[i].conj(), axes=([0, 1], [1, 0]))
            curr_L = curr_L.T
            L_env.append(curr_L)

        # 2. Build Right Environments (R_env[i] is contraction of i+1...L-1)
        R_env = [None] * self.L
        curr_R = np.array([[1.0]])
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            # B(p,chiL,chiR) * R(bra_R,ket_R) -> temp(p,chiL,bra_R)
            temp = np.tensordot(self.Bs[i], curr_R, axes=(2, 1))
            # temp(p,chiL,bra_R) * B*(p,bra_L,bra_R) -> curr_R(chiL,bra_L)
            curr_R = np.tensordot(temp, self.Bs[i].conj(), axes=([0, 2], [0, 2])).T
            R_env[i - 1] = curr_R

        rdm = {}
        for i in idx:
            t1 = np.tensordot(L_env[i], self.Bs[i], axes=(1, 1))
            t2 = np.tensordot(t1, R_env[i], axes=(2, 1))
            rho = np.tensordot(t2, self.Bs[i].conj(), axes=([0, 2], [1, 2]))
            rdm[i] = rho

        return rdm

    def calc_2site_rdm(self, idx_pairs=None):
        """
        Calculate 2-site reduced density matrices.
        """
        import numpy as np
        from collections import defaultdict

        # Helper for build identity on a bond for BlockTensor envs
        def _bond_eye(qns_bond, dir0=1):
            idxs_by_q = defaultdict(list)
            for k, q in enumerate(qns_bond):
                idxs_by_q[q].append(k)
            data = {}
            for q, ks in idxs_by_q.items():
                data[(q, q)] = np.eye(len(ks), dtype=complex)
            return BlockTensor(data, [list(qns_bond), list(qns_bond)], [dir0, -dir0])

        # Helper for densify a BlockTensor
        def _bt_to_dense(bt):
            maps = []
            for qlist in bt.qns:
                m = defaultdict(list)
                for i, q in enumerate(qlist):
                    m[q].append(i)
                maps.append(m)
            shape = tuple(len(q) for q in bt.qns)
            out = np.zeros(shape, dtype=complex)
            for qkey, block in bt.data.items():
                idx_lists = [maps[leg][qkey[leg]] for leg in range(bt.rank)]
                out[np.ix_(*idx_lists)] += block
            return out

        # Normalize idx_pairs
        if idx_pairs is None:
            pairs_by_i = {i: list(range(i + 1, self.L)) for i in range(self.L)}
        else:
            if isinstance(idx_pairs, tuple) and len(idx_pairs) == 2:
                idx_pairs = [idx_pairs]
            pairs_by_i = defaultdict(list)
            for (i, j) in idx_pairs:
                if i == j: continue
                a, b = (i, j) if i < j else (j, i)
                pairs_by_i[a].append(b)
            for i in pairs_by_i:
                pairs_by_i[i] = sorted(set(pairs_by_i[i]))

        # 2-rdm calculation with U(1) off
        if not (SYMMETRY_AVAILABLE and isinstance(self.Bs[0], BlockTensor)):
            # 1) Build overlap environments
            L_env = [np.array([[1.0]])]
            curr_L = L_env[0]
            for i in range(self.L - 1):
                temp = np.tensordot(curr_L, self.Bs[i], axes=(1, 1))
                curr_L = np.tensordot(temp, self.Bs[i].conj(), axes=([0, 1], [1, 0])).T
                L_env.append(curr_L)

            R_env = [None] * self.L
            curr_R = np.array([[1.0]])
            R_env[-1] = curr_R
            for i in range(self.L - 1, 0, -1):
                temp = np.tensordot(self.Bs[i], curr_R, axes=(2, 1))
                curr_R = np.tensordot(temp, self.Bs[i].conj(), axes=([0, 2], [0, 2])).T
                R_env[i - 1] = curr_R

            # 2) Precompute components
            L_components = []
            for i in range(self.L):
                t = np.tensordot(L_env[i], self.Bs[i], axes=(1, 1))
                comp = np.tensordot(t, self.Bs[i].conj(), axes=(0, 1))
                comp = comp.transpose(0, 2, 3, 1)
                L_components.append(comp)

            R_components = []
            for i in range(self.L):
                t = np.tensordot(self.Bs[i], R_env[i], axes=(2, 1))
                comp = np.tensordot(t, self.Bs[i].conj(), axes=(2, 2))
                comp = comp.transpose(0, 2, 3, 1)
                R_components.append(comp)

            # 3) Assemble
            rdm = {}
            for i in range(self.L):
                js = pairs_by_i.get(i, [])
                if not js: continue

                tensor = L_components[i]
                max_j = max(js)
                for j in range(i + 1, max_j + 1):
                    if j > i + 1:
                        k = j - 1
                        tensor = np.tensordot(tensor, self.Bs[k], axes=(3, 1))
                        tensor = np.tensordot(tensor, self.Bs[k].conj(), axes=([2, 3], [1, 0]))
                        tensor = tensor.transpose(0, 1, 3, 2)

                    if j in js:
                        # FIX: Use np.tensordot explicitly here (not the symmetric one)
                        rho_ij = np.tensordot(tensor, R_components[j], axes=([2, 3], [2, 3]))
                        rho_ij = rho_ij.transpose(0, 2, 1, 3)

                        d_i, d_j = rho_ij.shape[0], rho_ij.shape[1]
                        
                        # Normalize
                        rho_mat = rho_ij.reshape(d_i * d_j, d_i * d_j)
                        tr = np.trace(rho_mat)
                        if abs(tr) > 1e-12:
                            rho_mat /= tr
                            
                        rdm[(i, j)] = rho_mat

            return rdm

        # U(1) = True BRANCH (BlockTensors)
        # 1) Build overlap environments
        L_env = []
        curr_L = _bond_eye(self.Bs[0].qns[0], dir0=self.Bs[0].dirs[0])
        L_env.append(curr_L)
        for i in range(self.L - 1):
            temp = tensordot(curr_L, self.Bs[i], axes=([1], [0]))
            curr_L = tensordot(temp, self.Bs[i].conj(), axes=([0, 2], [0, 2]))
            curr_L = curr_L.transpose(1, 0)
            L_env.append(curr_L)

        R_env = [None] * self.L
        curr_R = _bond_eye(self.Bs[-1].qns[1], dir0=self.Bs[-1].dirs[1])
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            temp = tensordot(self.Bs[i], curr_R, axes=([1], [1]))
            curr_R = tensordot(temp, self.Bs[i].conj(), axes=([2, 1], [1, 2]))
            curr_R = curr_R.transpose(1, 0)
            R_env[i - 1] = curr_R

        # 2) Precompute components
        L_components = []
        for i in range(self.L):
            t = tensordot(L_env[i], self.Bs[i], axes=([1], [0]))
            comp = tensordot(t, self.Bs[i].conj(), axes=([0], [0]))
            comp = comp.transpose(1, 3, 2, 0)
            L_components.append(comp)

        R_components = []
        for i in range(self.L):
            t = tensordot(self.Bs[i], R_env[i], axes=([1], [1]))
            comp = tensordot(t, self.Bs[i].conj(), axes=([2], [1]))
            comp = comp.transpose(1, 3, 2, 0)
            R_components.append(comp)

        # 3) Assemble
        rdm = {}
        for i in range(self.L):
            js = pairs_by_i.get(i, [])
            if not js: continue

            tensor = L_components[i]
            max_j = max(js)

            for j in range(i + 1, max_j + 1):
                if j > i + 1:
                    k = j - 1
                    tensor = tensordot(tensor, self.Bs[k], axes=([3], [0]))
                    tensor = tensordot(tensor, self.Bs[k].conj(), axes=([2, 4], [0, 2]))
                    tensor = tensor.transpose(0, 1, 3, 2)

                if j in js:
                    rho_ij = tensordot(tensor, R_components[j], axes=([2, 3], [2, 3]))
                    rho_ij = rho_ij.transpose(0, 2, 1, 3)

                    rho_dense = _bt_to_dense(rho_ij)
                    d_i, d_j = rho_dense.shape[0], rho_dense.shape[1]
                    
                    # Normalize
                    rho_mat = rho_dense.reshape(d_i * d_j, d_i * d_j)
                    tr = np.trace(rho_mat)
                    if abs(tr) > 1e-12:
                        rho_mat /= tr

                    rdm[(i, j)] = rho_mat

        return rdm


class Site(object):
    """A general single site

    You use this class to create a single site. The site comes empty (i.e.
    with no operators included), but for th identity operator. You should
    add operators you need to make you site up.

    Parameters
    ----------
    dim : an int
	Size of the Hilbert space. The dimension must be at least 1. A site of
        dim = 1  represents the vaccum (or something strange like that, it's
        used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the site.

    Examples
    --------
    >>> from dmrg101.core.sites import Site
    >>> brand_new_site = Site(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_site.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_site.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
        """
        Creates an empty site of dimension dim.

        	Raises
        	------
        	DMRGException
        	    if `dim` < 1.

        	Notes
        	-----
        	Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
        	is added to the `self.operators` dictionary.
        """
        if dim < 1:
            raise DMRGException("Site dim must be at least 1")
        # super(Site, self).__init__()
        self.dim = dim
        self.operators = { "id" : scipy.sparse.eye(self.dim, self.dim) }

    def add_operator(self, operator_name):
        """
        Adds an operator to the site.

          Parameters
       	----------
           	operator_name : string
       	    The operator name.

       	Raises
       	------
       	DMRGException
       	    if `operator_name` is already in the dict.

       	Notes
       	-----
       	Postcond:

              - `self.operators` has one item more, and
              - the newly created operator is a (`self.dim`, `self.dim`)
                matrix of full of zeros.

       	Examples
       	--------
       	>>> new_site = Site(2)
       	>>> print new_site.operators.keys()
       	['id']
       	>>> new_site.add_operator('s_z')
       	>>> print new_site.operators.keys()
       	['s_z', 'id']
       	>>> # note that the newly created op has all zeros
       	>>> print new_site.operators['s_z']
       	[[ 0.  0.]
        	 [ 0.  0.]]
        """

        if str(operator_name) in self.operators.keys():
            raise DMRGException("Operator name exists already")
        else:
            self.operators[str(operator_name)] = np.zeros((self.dim, self.dim))

"""Exception class for the DMRG code
"""
class DMRGException(Exception):
    """A base exception for the DMRG code

    Parameters
    ----------
    msg : a string
        A message explaining the error
    """
    def __init__(self, msg):
        super(DMRGException, self).__init__()
        self.msg = msg

    def __srt__(self, msg):
        	return repr(self.msg)

class Block(Site):
    """A block.

    That is the representation of the Hilbert space and operators of a
    direct product of single site's Hilbert space and operators, that have
    been truncated.

    You use this class to create the two blocks (one for the left, one for
    the right) needed in the DMRG algorithm. The block comes empty.

    Parameters
    ----------
    dim : an int.
	Size of the Hilbert space. The dimension must be at least 1. A
	block of dim = 1  represents the vaccum (or something strange like
	that, it's used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the block.

    Examples
    --------
    >>> from dmrg101.core.block import Block
    >>> brand_new_block = Block(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_block.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_block.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
        """Creates an empty block of dimension dim.

        Raises
        ------
        DMRGException
                if `dim` < 1.

        Notes
        -----
        Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
        is added to the `self.operators` dictionary. A full of zeros block
        Hamiltonian operator is added to the list.
        """
        super(Block, self).__init__(dim)

class PauliSite(Site):
    """
    A site for spin 1/2 models.

    You use this site for models where the single sites are spin
    one-half sites. The Hilbert space is ordered such as the first state
    is the spin down, and the second state is the spin up. Therefore e.g.
    you have the following relation between operator matrix elements:

    .. math::

        \langle \downarrow | A | uparrow \rangle = A_{0,1}

    Notes
    -----
    Postcond: The site has already built-in the spin operators for s_z, s_p, s_m.

    Examples
    --------
    >>> from dmrg101.core.sites import PauliSite
    >>> pauli_site = PauliSite()
    >>> # check all it's what you expected
    >>> print pauli_site.dim
    2
    >>> print pauli_site.operators.keys()
    ['s_p', 's_z', 's_m', 'id']
    >>> print pauli_site.operators['s_z']
    [[-1.  0.]
      [ 0.  1.]]
    >>> print pauli_site.operators['s_x']
    [[ 0.  1.]
      [ 1.  0.]]
    """
    def __init__(self):
        """
        Creates the spin one-half site with Pauli matrices.

 	  Notes
 	  -----
 	  Postcond : the dimension is set to 2, and the Pauli matrices
 	  are added as operators.

        """
        super(PauliSite, self).__init__(2)
	# add the operators
        self.add_operator("s_z")
        self.add_operator("s_x")
        self.add_operator("s_m")

	# for clarity
        s_z = self.operators["s_z"]
        s_x = self.operators["s_x"]
        s_m = self.operators["s_m"]

	# set the matrix elements different from zero to the right values
        s_z[0, 0] = -1.0
        s_z[1, 1] = 1.0
        s_x[0, 1] = 1.0
        s_x[1, 0] = 1.0
        s_m[0, 1] = 1.0





def LeftCanonical(M):
    '''
        Function that takes an MPS 'M' as input (order of legs: left-bottom-right) and returns a copy of it that is
            transformed into left canonical form and normalized.

    Src:
        https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
    '''
    Mcopy = M.copy() #create copy of M

    N = len(Mcopy) #nr of sites

    for l in range(N):
        # reshape
        Taux = Mcopy[l]
        Taux = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))

        # SVD
        U,S,Vdag = np.linalg.svd(Taux,full_matrices=False)
        '''
            Note: full_matrices=False leads to a trivial truncation of the matrices (thin SVD).
        '''

        # update M[l]
        Mcopy[l] = np.reshape(U,(np.shape(Mcopy[l])[0],np.shape(Mcopy[l])[1],np.shape(U)[1]))

        # update M[l+1]
        SVdag = np.matmul(np.diag(S),Vdag)
        if l < N-1:
            Mcopy[l+1] = np.einsum('ij,jkl',SVdag,Mcopy[l+1])
        else:
            '''
                Note: in the last site (l=N-1), S*Vdag is a number that determines the normalization of the MPS.
                    We discard this number, which corresponds to normalizing the MPS.
            '''

    return Mcopy


def RightCanonical(M):
    '''
        Function that takes an MPS 'M' as input (order of legs: left-bottom-right) and returns a copy of it that is
            transformed into right canonical form and normalized.
    '''
    Mcopy = M.copy() #create copy of M

    N = len(Mcopy) #nr of sites

    for l in range(N-1,-1,-1):
        # reshape
        Taux = Mcopy[l]
        Taux = np.reshape(Taux,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))

        # SVD
        U,S,Vdag = np.linalg.svd(Taux,full_matrices=False)

        # update M[l]
        Mcopy[l] = np.reshape(Vdag,(np.shape(Vdag)[0],np.shape(Mcopy[l])[1],np.shape(Mcopy[l])[2]))

        # update M[l-1]
        US = np.matmul(U,np.diag(S))
        if l > 0:
            Mcopy[l-1] = np.einsum('ijk,kl',Mcopy[l-1],US)
        else:
            '''
                Note: in the first site (l=0), U*S is a number that determines the normalization of the MPS. We
                    discard this number, which corresponds to normalizing the MPS.
            '''

    return Mcopy

# class MPS:
#     def __init__(self, factors, homogenous=False, form=None):
#         """
#         class for matrix product states.

#         Parameters
#         ----------
#         mps : list
#             list of 3-tensors. [chi1, d, chi2]
#         chi_max:
#             maximum bond order used in compress. Default None.

#         Returns
#         -------
#         None.

#         """
#         self.factors = self.data = factors
#         self.nsites = self.L = len(factors)
#         self.nbonds = self.nsites - 1
#         # self.chi_max = chi_max

#         self.form = form

#         if homogenous:
#             self.dims = [mps[0].shape[1], ] * self.nsites
#         else:
#             self.dims = [t.shape[1] for t in factors] # physical dims of each site

#         # self._mpo = None

#     def bond_orders(self):
#         return [t.shape[2] for t in self.factors] # bond orders


#     def compress(self, chi_max):
#         return MPS(compress(self.factors, chi_max)[0])

#     def __add__(self, other):
#         assert len(self.data) == len(other.data)
#         # for different length, we should choose the maximum one
#         C = []
#         for j in range(self.sites):
#             tmp = block_diag(self.data[j], other.data[j])
#             C.append(tmp.copy())

#         return MPS(C)

    # def build_mpo_list(self):
    #     # build MPO representation of the propagator
    #     pass

    # def copy(self):
    #     return copy.copy(self)

    # def run(self, dt=0.1, Nt=10):
    #     pass

    # def obs_single_site(self, e_op, n):
    #     pass

    # def two_sites(self):
    #     pass

    # # def to_tensor(self):
    # #     return mps_to_tensor(self.factors)

    # # def to_vec(self):
    # #     return mps_to_tensor(self.factors)

    # def left_canonicalize(self):
    #     pass

    # def right_canonicalize(self):
    #     pass

    # def left_to_right(self):
    #     pass

    # def site_canonicalize(self):
    #     pass


class MPO:
    def __init__(self, factors, homogenous=False):
        """
        class for matrix product operators.

        Parameters
        ----------
        factors : list
            list of 4-tensors of dimension. [chi1, d, chi2, d]
        chi_max:
            maximum bond order used in compress. Default None.

        Returns
        -------
        None.

        """
        self.factors = self.data = self.cores = factors
        self.nsites = self.L = len(factors)
        self.nbonds = self.L - 1
        # self.chi_max = chi_max


        if homogenous:
            self.dims = [factors[0].shape[1], ] * self.nsites
        else:
            self.dims = [t.shape[1] for t in factors] # physical dims of each site

        # self._mpo = None

    def bond_orders(self):
        return [t.shape[0] for t in self.factors] # bond orders

    def ground_state(self, algorithm='dmrg'):
        pass

    def dot(self, mps, rank):
        # apply MPO to MPS followed by a compression

        factors = apply_mpo(self.factors, mps.factors, rank)

        return MPS(factors)

    def __matmul__(self, other, compress=False, D=None):
        """
        define product of two MPOs

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(other, MPO):
            return product_MPO(self.factors, other.factors)

        elif isinstance(other, MPS):
            return apply_mpo(self.factors, other.factors)


    def __mul__(self, other):
        """
        Element-wise multiplication of MPO with another MPO or scalar.

        Supports:
        - MPO * MPO: Element-wise (Hadamard) product
        - MPO * scalar: Scalar multiplication

        For MPO×MPO multiplication, virtual bonds are combined via Kronecker product.

        Parameters
        ----------
        other : MPO, int, float, or complex
            Second operand for multiplication

        Returns
        -------
        MPO
            Element-wise product

        Raises
        ------
        ValueError
            If MPOs have incompatible physical dimensions
        """
        # Scalar multiplication
        if isinstance(other, (int, float, complex)):
            factors_new = [f.copy() for f in self.factors]
            factors_new[0] = factors_new[0] * other
            return MPO(factors_new)

        # MPO * MPO element-wise multiplication
        elif isinstance(other, MPO):
            if self.L != other.L:
                raise ValueError(
                    f"MPOs must have same length: {self.L} vs {other.L}")
            
            if self.dims != other.dims:
                raise ValueError(
                    f"Physical dimensions must match: {self.dims} vs {other.dims}")

            factors_new = []
            for i in range(self.L):
                # self.factors[i]: (chi1, d, chi2, d)
                # other.factors[i]: (xi1, d, xi2, d)
                W1 = self.factors[i]
                W2 = other.factors[i]
                
                # Element-wise product: einsum 'aijb,mijn->amijbn'
                # Then reshape to (chi1*xi1, d, chi2*xi2, d)
                core = np.reshape(
                    np.einsum('aijb,mijn->amijbn', W1, W2),
                    [W1.shape[0] * W2.shape[0], W1.shape[1], 
                     W1.shape[2] * W2.shape[2], W1.shape[3]])
                factors_new.append(core)

            return MPO(factors_new)

        else:
            raise ValueError(
                'Second operand must be MPO, int, float, or complex')

    def __add__(self, other):
        """
        Add two MPOs element-wise.

        Implements the direct sum in MPO format by concatenating virtual bonds.
        Both MPOs must have compatible physical dimensions.

        Parameters
        ----------
        other : MPO
            Second MPO to add

        Returns
        -------
        MPO
            Sum of the two MPOs with increased bond dimensions

        Raises
        ------
        TypeError
            If other is not an MPO instance
        ValueError
            If MPOs have incompatible shapes
        """
        if not isinstance(other, MPO):
            raise TypeError("Only support addition of two MPO objects.")

        if self.L != other.L:
            raise ValueError(
                f"MPOs must have same length: {self.L} vs {other.L}")

        if self.dims != other.dims:
            raise ValueError(
                f"Physical dimensions must match: {self.dims} vs {other.dims}")

        sum_factors = []
        for i in range(self.L):
            # factors: (chi1, d, chi2, d)
            W1 = self.factors[i]
            W2 = other.factors[i]
            r1_l, d, r1_r, _ = W1.shape
            r2_l, _, r2_r, _ = W2.shape
            
            if i == 0:
                # First site: concatenate along right bond (axis 2)
                W_sum = np.concatenate([W1, W2], axis=2)
            elif i == self.L - 1:
                # Last site: concatenate along left bond (axis 0)
                W_sum = np.concatenate([W1, W2], axis=0)
            else:
                # Middle sites: block diagonal structure
                W_sum = np.zeros((r1_l + r2_l, d, r1_r + r2_r, d), 
                                dtype=W1.dtype)
                W_sum[:r1_l, :, :r1_r, :] = W1
                W_sum[r1_l:, :, r1_r:, :] = W2

            sum_factors.append(W_sum)
        
        return MPO(sum_factors)



    def exponential(self, constant=1.0, method='taylor', order=4, scale=0):
        """
        Calculate the exponential of an MPO: exp(constant*self).

        Parameters
        ----------
        constant : float or complex
            Constant coefficient in the exponent
            For time propagation (time-evolution operator), use constant = -i*dt (with ℏ=1)
            For thermal density matrix (Boltzmann operator), use constant = -β where β=1/kT
        method : str
            Method to use for calculation. Options:
            - 'taylor': Taylor expansion (default)
        order : int
            Order of the Taylor expansion (only for method='taylor'), default is 4
        scale : int
            Scaling parameter 2^scale for better numerical stability, default is 0
            Details: exp(x) = (exp(x/2^scale))^(2^scale) scale-squaring technique to ensure unitarity

        Returns
        -------
        MPO
            Result of the exponential operation exp(constant*self)
        """
        if method != 'taylor':
            raise ValueError(f"Method '{method}' not implemented. Only 'taylor' is supported.")

        # Scale constant for numerical stability
        scaled_constant = constant / (2 ** scale)

        # Determine the appropriate dtype for the result
        # Need to handle: real MPO with complex constant, complex MPO with real constant, etc.
        constant_dtype = np.array(scaled_constant).dtype
        mpo_dtype = self.factors[0].dtype
        # Use numpy's result_type to determine the appropriate output dtype
        result_dtype = np.result_type(constant_dtype, mpo_dtype)

        # Create identity MPO with the correct dtype
        identity_factors = []
        for i in range(self.L):
            d = self.dims[i]
            # All sites have the same structure for identity: (1, d, 1, d)
            W = np.zeros((1, d, 1, d), dtype=result_dtype)
            for j in range(d):
                W[0, j, 0, j] = 1.0
            identity_factors.append(W)

        result = MPO(identity_factors)
        
        # Taylor expansion: exp(x) = sum_{k=0}^{order} x^k / k!
        # term keeps track of (constant*self)^k
        term = MPO([f.copy() for f in identity_factors])  # k=0 term
        
        factorial = 1
        for k in range(1, order + 1):
            # Compute next term: term = term @ self
            term = term @ self
            factorial = factorial * k
            coefficient = (scaled_constant ** k) / factorial

            # Add this term: result = result + coefficient * term
            result = result + (term * coefficient)

        # Apply scaling: result = result^(2^scale)
        for _ in range(scale):
            result = result @ result

        return result



# apply_mpo_to_mps = apply_mpo

def apply_mpo(w_list, B_list, chi_max):
    """
    Apply the MPO to an MPS.

    MPS in :math:`[\alpha_l, d_l, \alpha_{l+1}]`

    MPO in :math:`[\alpha_l, d_l, \alpha_{l+1}, d_l]`

    Parameters
    ----------
    B_list : TYPE
        DESCRIPTION.
    s_list : TYPE
        DESCRIPTION.
    w_list : TYPE
        DESCRIPTION.
    chi_max : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # d = B_list[0].shape[1] # size of local space
    # D = w_list[0].shape[1]

    L = len(B_list) # nsites

    chi1, d, chi2 = B_list[0].shape # left and right bond dims
    b1, d, b2, d = w_list[0].shape # left and right bond dims

    B = np.tensordot(w_list[0], B_list[0], axes=(3,1))
    B = np.transpose(B,(3,0,1,4,2))

    B = np.reshape(B,(chi1*b1, d, chi2*b2))

    B_list[0] = B

    for i_site in range(1,L-1):
        chi1, d, chi2 = B_list[i_site].shape
        b1, _, b2, _ = w_list[i_site].shape # left and right bond dims

        B = np.tensordot(w_list[i_site], B_list[i_site], axes=(3,1))
        B = np.reshape(np.transpose(B,(3,0,1,4,2)),(chi1*b1, d, chi2*b2))

        B_list[i_site] = B
        # s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)

    # last site
    chi1, d, chi2 = B_list[L-1].shape
    b1, _, b2, _ = w_list[L-1].shape # left and right bond dims

    B = np.tensordot(w_list[L-1], B_list[L-1], axes=(3,1))
    B = np.reshape(np.transpose(B,(3,0,1,4,2)),(chi1*b1, d, chi2*b2))

    # s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B

    return B
    # return compress(B_list, chi_max)

'''
    Function that makes the following contractions (numbers denote leg order):

         /--3--**--1--Mt--3--
         |             |
         |             2
         |             |
         |             *
         |             *
         |             |
         |             4                 /--3--
         |             |                 |
        Tl--2--**--1---O--3--     =     Tf--2--
         |             |                 |
         |             2                 \--1--
         |             |
         |             *
         |             *
         |             |
         |             2
         |             |
         \--1--**--3--Mb--1--
'''
def ZipperLeft(Tl,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mb,Tl)
    Taux = np.einsum('ijkl,kjmn',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mt)

    return Tf

def expect(mpo, mps):
    # <GS| O |GS> , closing the zipper from the left
    Taux = np.ones((1,1,1))
    for l in range(N):
        Taux = ZipperLeft(Taux, mps[l].conj().T, mpo[l], mps[l])
    print('<GS| H |GS> = ', Taux[0,0,0])
    # print('analytical result = ', -2*(N-1)/3)
    return Taux[0, 0, 0]

'''
    Function that makes the following contractions (numbers denote leg order):

         --1--Mt--3--**--1--\
               |            |
               2            |
               |            |
               *            |
               *            |
               |            |
               4            |            --1--\
               |            |                 |
         --1---O--3--**--2--Tr     =     --2--Tf
               |            |                 |
               2            |            --3--/
               |            |
               *            |
               *            |
               |            |
               2            |
               |            |
         --3--Mb--1--**--3--/
'''
def ZipperRight(Tr,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mt,Tr)
    Taux = np.einsum('ijkl,mnkj',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mb)

    return Tf

def expect_zipper_right(mpo, mps):
    # <GS| H |GS> for AKLT model, closing the zipper from the right
    Taux = np.ones((1,1,1))
    for l in range(N-1,-1,-1):
        Taux = ZipperRight(Taux, mps[l].conj().T, mpo[l], mps[l])
    # print('<GS| H |GS> = ', Taux[0,0,0])
    # print('analytical result = ', -2*(N-1)/3)

    return Taux[0,0,0]


# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual vonds

# MPO W-matrix is a 4-index tensor, W[s,t,i,j]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds

## initial E and F matrices for the left and right vacuum states
def initial_E(W):
    if SYMMETRY_AVAILABLE and isinstance(W, BlockTensor):
        # MPO (In), Bra (In), Ket (In) -> Need Out (+1)
        data = {(0, 0, 0): np.ones((1, 1, 1))}
        qns = [[0], [0], [0]]
        dirs = [1, -1, 1] 
        return BlockTensor(data, qns, dirs)
    E = np.zeros((W.shape[0], 1, 1))
    E[0] = 1
    return E

def initial_F(W, target_qn=0):
    """
    Constructs the initial Right Environment (Vacuum).
    """
    if SYMMETRY_AVAILABLE and isinstance(W, BlockTensor):
        # MPO (In), Bra (Out), Ket (In) -> Need [In, Out, In] = [-1, 1, -1]
        data = {(0, target_qn, target_qn): np.ones((1, 1, 1))}
        qns = [[0], [target_qn], [target_qn]]
        dirs = [-1, 1, -1]
        return BlockTensor(data, qns, dirs)
    F = np.zeros((W.shape[1], 1, 1))
    F[-1] = 1
    return F


def dense_to_symmetric(mps_list, phys_qns=None, tol=1e-12):
    """
    Convert a *product-state* dense MPS guess into a true U(1) BlockTensor MPS.

    Supports:
      - spin-orbital sites: d=2, phys_qns=[0,1]
      - spatial-orbital sites: d=4, phys_qns=[0,1,1,2]

    Requirement:
      - product state only (bond dims must be 1 at every site)
      - each site must have support in exactly one charge sector
    """
    if not SYMMETRY_AVAILABLE:
        return mps_list

    import numpy as np

    # Infer phys_qns from d if not given
    if phys_qns is None:
        # peek at first site to infer d
        M0 = np.asarray(mps_list[0])
        if M0.ndim != 3:
            raise ValueError(f"Expected rank-3 tensors, got shape {M0.shape}")
        # extract d from any axis that looks like physical for product tensors
        d_candidates = sorted(set(M0.shape))
        # more robust: just take the axis that isn't 1 if it's product state
        d = next((x for x in M0.shape if x != 1), None)
        if d is None:
            d = M0.shape[-1]

        if d == 2:
            phys_qns = [0, 1]
        elif d == 4:
            phys_qns = [0, 1, 1, 2]
        else:
            raise ValueError(f"Cannot infer phys_qns for local dimension d={d}. Pass phys_qns explicitly.")
    phys_qns = list(phys_qns)
    new_list = []

    qL = 0  # cumulative charge to the left

    for site, M in enumerate(mps_list):
        M = np.asarray(M)
        if M.ndim != 3:
            raise ValueError(f"Site {site}: expected rank-3 tensor, got shape {M.shape}")

        # extract local vector v[d] from product state tensor
        if M.shape[0] == 1 and M.shape[1] == 1:      # (L,R,P)
            v = M[0, 0, :]
        elif M.shape[1] == 1 and M.shape[2] == 1:    # (P,L,R)
            v = M[:, 0, 0]
        elif M.shape[0] == 1 and M.shape[2] == 1:    # (L,P,R)
            v = M[0, :, 0]
        else:
            raise ValueError(
                f"Site {site}: only supports product-state tensors with bond dims 1. Got {M.shape}."
            )

        d = len(v)
        if d != len(phys_qns):
            raise ValueError(f"Site {site}: local dim d={d} but phys_qns length={len(phys_qns)}")

        supp = [k for k, amp in enumerate(v) if abs(amp) > tol]
        if not supp:
            raise ValueError(f"Site {site}: zero local vector.")

        q_support = sorted(set(phys_qns[k] for k in supp))
        if len(q_support) != 1:
            raise ValueError(
                f"Site {site}: local state spans multiple charge sectors {q_support}. "
                "Provide a fixed-charge product guess (or implement a general converter)."
            )

        qP = q_support[0]
        qR = qL + qP

        idxs = [k for k, q in enumerate(phys_qns) if q == qP]
        vec = v[idxs].astype(complex)

        block = vec.reshape(1, 1, len(idxs))
        data = {(qL, qR, qP): block}

        qns = [[qL], [qR], list(phys_qns)]
        dirs = [-1, 1, 1]  # keep your convention

        new_list.append(BlockTensor(data, qns, dirs))
        qL = qR

    return new_list


def contract_from_right(W, A, F, B):
    """
    ## tensor contraction from the right hand side
    ##  -+     -A--+
    ##   |      |  |
    ##  -F' =  -W--F
    ##   |      |  |
    ##  -+     -B--+

    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    #return np.einsum("abst,sij,bjl,tkl->aik",W,A,F,B, optimize=True)

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if SYMMETRY_AVAILABLE and isinstance(A, BlockTensor):
        # F: (MPO, Bra, Ket). A_bra: A.conj().
        # Contract F.Bra(1) with A.conj().Right(1)
        Temp = tensordot(A.conj(), F, axes=([1], [1]))
        
        # Contract with W (L, R, Out, In)
        # Contract Temp.MPO(2) with W.Right(1)
        # Contract Temp.P(1) with W.Out(2)
        Temp = tensordot(Temp, W, axes=([2, 1], [1, 2])) 
        
        # Contract with B(Ket): (L, R, P)
        # Contract Temp.Ket(1) with B.Right(1)
        # Contract Temp.In_W(3) with B.Phys(2)
        Temp = tensordot(Temp, B, axes=([1, 3], [1, 2])) 
        
        return Temp.transpose(1, 0, 2)
    else:
        Temp = np.einsum("sij,bjl->sbil", A, F)
        Temp = np.einsum("sbil,abst->tail", Temp, W)
        return np.einsum("tail,tkl->aik", Temp, B)

def contract_from_left(W, A, E, B):
    """
    ## tensor contraction from the left hand side
    ## +-    +--A-
    ## |     |  |
    ## E' =  E--F-
    ## |     |  |
    ## +-    +--B-

    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    # return np.einsum("abst,sij,aik,tkl->bjl",W,A,E,B, optimize=True)

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if SYMMETRY_AVAILABLE and isinstance(A, BlockTensor):
        # E: (MPO, Bra, Ket). A_bra: A.conj().
        # Contract E.Bra(1) with A.conj().Left(0)
        Temp = tensordot(E, A.conj(), axes=([1], [0])) 
        
        # Contract with W (L, R, Out, In)
        # Contract Temp.MPO(0) with W.Left(0)
        # Contract Temp.P(3) with W.Out(2)
        Temp = tensordot(Temp, W, axes=([0, 3], [0, 2])) 
        
        # Contract with B (L, R, P)
        # Contract Temp.Ket(0) with B.Left(0)
        # Contract Temp.W_In(3) with B.Phys(2)
        Temp = tensordot(Temp, B, axes=([0, 3], [0, 2])) 
        
        return Temp.transpose(1, 0, 2)
    else:
        Temp = np.einsum("sij,aik->sajk", A, E)
        Temp = np.einsum("sajk,abst->tbjk", Temp, W)
        return np.einsum("tbjk,tkl->bjl", Temp, B)


def construct_F(Alist, MPO, Blist, target_qn = None):
    """
    # construct the initial E and F matrices.
    # we choose to start from the left hand side, so the initial E matrix
    # is zero, the initial F matrices cover the complete chain

    Parameters
    ----------
    Alist : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    Blist : TYPE
        DESCRIPTION.

    Returns
    -------
    F : TYPE
        DESCRIPTION.

    """
    if SYMMETRY_AVAILABLE and isinstance(Blist[-1], BlockTensor):
        if target_qn is None:
            # pick the unique right-bond qR from the last site tensor
            # key = (qL, qR, qP) for site tensors in this code
            qs = sorted({key[1] for key in Blist[-1].data.keys()})
            if len(qs) != 1:
                raise ValueError(f"Ambiguous total charge on last bond: {qs}. Pass target_qn explicitly.")
            target_qn = qs[0]

    F = [initial_F(MPO[-1], target_qn=target_qn if target_qn is not None else 0)]
    for i in range(len(MPO)-1, 0, -1):
        F.append(contract_from_right(MPO[i], Alist[i], F[-1], Blist[i]))
    return F

def construct_E(Alist, MPO, Blist):
    return [initial_E(MPO[0])]


def coarse_grain_MPO(W, X):
    """
    # 2-1 coarse-graining of two site MPO into one site
    #  |     |  |
    # -R- = -W--X-
    #  |     |  |

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("abst,bcuv->acsutv",W,X),
                      [W.shape[0], X.shape[1],
                       W.shape[2]*X.shape[2],
                       W.shape[3]*X.shape[3]])


def product_W(W, X):
    """
    # 'vertical' product of MPO W-matrices
    #        |
    #  |    -W-
    # -R- =  |
    #  |    -X-
    #        |

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("abst,cdtu->acbdsu", W, X), [W.shape[0]*X.shape[0],
                                                             W.shape[1]*X.shape[1],
                                                             W.shape[2],X.shape[3]])


def product_MPO(M1, M2):
    assert len(M1) == len(M2)
    Result = []
    for i in range(0, len(M1)):
        Result.append(product_W(M1[i], M2[i]))
    return Result



def coarse_grain_MPS(A,B):
    """
    # 2-1 coarse-graining of two-site MPS into one site
    #   |     |  |
      -R- <= -A--B-

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("sij,tjk->stik",A,B),
                      [A.shape[0]*B.shape[0], A.shape[1], B.shape[2]])

def fine_grain_MPS(A, dims):
    assert A.shape[0] == dims[0] * dims[1]
    Theta = np.transpose(np.reshape(A, dims + [A.shape[1], A.shape[2]]),
                         (0,2,1,3))
    M = np.reshape(Theta, (dims[0]*A.shape[1], dims[1]*A.shape[2]))
    U, S, V = np.linalg.svd(M, full_matrices=0)
    U = np.reshape(U, (dims[0], A.shape[1], -1))
    V = np.transpose(np.reshape(V, (-1, dims[1], A.shape[2])), (1,0,2))
    # assert U is left-orthogonal
    # assert V is right-orthogonal
    #print(np.dot(V[0],np.transpose(V[0])) + np.dot(V[1],np.transpose(V[1])))
    return U, S, V

def truncate_SVD(U, S, V, m):
    """
    # truncate the matrices from an SVD to at most m states

    Parameters
    ----------
    U : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    U : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    trunc : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    """
    m = min(len(S), m)
    trunc = np.sum(S[m:])
    S = S[0:m]
    U = U[:,:,0:m]
    V = V[:,0:m,:]
    return U,S,V,trunc,m

# Functor to evaluate the Hamiltonian matrix-vector multiply
#        +--A--+
#        |  |  |
# -R- =  E--W--F
#  |     |  |  |
#        +-   -+
class HamiltonianMultiply(sparse.linalg.LinearOperator):
    def __init__(self, E, W, F):
        self.E = E
        self.W = W
        self.F = F
        self.dtype = np.dtype('d')
        self.req_shape = [W.shape[2], E.shape[1], F.shape[2]]
        self.size = self.req_shape[0]*self.req_shape[1]*self.req_shape[2]
        self.shape = [self.size, self.size]

    def _matvec(self, A):
        # the einsum function doesn't appear to optimize the contractions properly,
        # so we split it into individual summations in the optimal order
        #R = np.einsum("aij,sik,abst,bkl->tjl",self.E,np.reshape(A, self.req_shape),
        #              self.W,self.F, optimize=True)
        R = np.einsum("aij,sik->ajsk", self.E, np.reshape(A, self.req_shape), optimize=True)
        R = np.einsum("ajsk,abst->bjtk", R, self.W, optimize=True)
        R = np.einsum("bjtk,bkl->tjl", R, self.F, optimize=True)
        return np.reshape(R, -1)

## optimize a single site given the MPO matrix W, and tensors E,F
def optimize_site(A, W, E, F, tol=1E-8):
    H = HamiltonianMultiply(E,W,F)
    # we choose tol=1E-8 here, which is OK for small calculations.
    # to bemore robust, we should take the tol -> 0 towards the end
    # of the calculation.
    E, V = sparse.linalg.eigsh(H,1,v0=A,which='SA', tol=tol)
    return (E[0],np.reshape(V[:,0], H.req_shape))


def optimize_two_sites(A, B, W1, W2, E, F, m, dir, U1=False):
    """
    two-site optimization of MPS A,B with respect to MPO W1,W2 and
    environment tensors E,F
    dir = 'left' or 'right' for a left-moving or right-moving sweep

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    W1 : TYPE
        DESCRIPTION.
    W2 : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    dir : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    trunc : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    """
    if U1:
        if not SYMMETRY_AVAILABLE:
            raise ImportError("Symmetry module not found. Cannot run U1=True.")
            
        # 1. Form Initial Guess (Bond Dimension expansion happens here naturally in SVD)
        # A: (Bond_L, Bond_M, Phys_L)
        # B: (Bond_M, Bond_R, Phys_R)
        # AA = A * B -> (Bond_L, Phys_L, Bond_R, Phys_R)
        # Note on A/B indices in BlockTensor:
        # standard MPS layout: (Left, Right, Phys).
        # Contraction: A[Right] -- B[Left]
        
        # Check rank to be sure
        if A.rank == 3:
            AA = tensordot(A, B, axes=([1], [0])) # this will return as BlockTensor Object
            AA = AA.transpose(0, 2, 1, 3)
            
            # add noise to AA
            # forces Davidson to explore new sectors
            noise_scale = 1e-4
            for k in AA.data:
                # Add random noise to existing blocks
                AA.data[k] += (np.random.rand(*AA.data[k].shape) - 0.5) * noise_scale
        else:
            raise ValueError(f"Unexpected tensor rank {A.rank} in symmetric opt")

        # 2. Define Linear Operator
        H_op = HamiltonianMultiplyU1(E, [W1, W2], F)
        
        # 3. Solve Eigenproblem (Davidson)
        # Normalize guess
        norm = AA.norm()
        AA = AA * (1.0/norm)
        
        energy, AA_new = solve_davidson(H_op, AA, tol=1e-5)
        
        # 4. SVD and Split
        # AA_new is (L, R, Phys_L, Phys_R)
        # We need to return A(L, M, P_L) and B(M, R, P_R)
        
        # Use our symmetric SVD, 3rd argument S_dict is used in normalization
        U, V, S_dict, trunc, m_kept = svd_symmetric(AA_new, m_max=m)
        
        # U is (L, P_L, M).
        # V is (M, R, P_R).
        
        # We need to contract S_dict into either U or V depending on 'dir'. (left or right sweep)
        
        # Helper to contract Diagonal S into U
        # U is (L, P_L, Bond). S is (Bond, Bond).
        # We want U*S -> (L, P_L, Bond)
        def multiply_U_S(U_tensor, S_data):
            # U: data[(qL, qP, qM)] -> shape (dL, dP, dM)
            # S: data[qM] -> shape (dM, dM)
            new_data = {}
            for (qL, qP, qM), block in U_tensor.data.items():
                if qM in S_data:
                    # Contract last index of U with S
                    # block: (dL, dP, dM). S: (dM, dM)
                    # Result: (dL, dP, dM)
                    new_block = np.tensordot(block, S_data[qM], axes=([2], [0]))
                    new_data[(qL, qP, qM)] = new_block
            return BlockTensor(new_data, U_tensor.qns, U_tensor.dirs)

        # Helper to contract S into V
        # S is (Bond, Bond). V is (Bond, R, P_R).
        # We want S*V -> (Bond, R, P_R)
        def multiply_S_V(S_data, V_tensor):
            new_data = {}
            for (qM, qR, qP), block in V_tensor.data.items():
                if qM in S_data:
                    # Contract S with first index of V
                    # S: (dM, dM). block: (dM, dR, dP)
                    # Result: (dM, dR, dP)
                    new_block = np.tensordot(S_data[qM], block, axes=([1], [0]))
                    new_data[(qM, qR, qP)] = new_block
            return BlockTensor(new_data, V_tensor.qns, V_tensor.dirs)

        if dir == 'right':  # Right Sweep: Center moves Right.
            # A = U. B = S * V.
            
            # U is (L, P_L, M). Transpose to standard MPS shape A(L, M, P_L)
            A_new = U.transpose(0, 2, 1)
            
            # Contract S into V, then V is already (M, R, P_R), which is standard B shape
            B_new = multiply_S_V(S_dict, V)
            
        else: # that is dir == 'left'
            # Left Sweep: Center moves Left.
            # A = U * S. B = V.
            
            # Contract U * S first
            A_US = multiply_U_S(U, S_dict)
            
            # Transpose to standard MPS shape A(L, M, P_L)
            A_new = A_US.transpose(0, 2, 1)
            
            # B is just V
            B_new = V

        return energy, A_new, B_new, trunc, m_kept

    else:
        W = coarse_grain_MPO(W1,W2)
        AA = coarse_grain_MPS(A,B)
        H = HamiltonianMultiply(E,W,F)
        E,V = sparse.linalg.eigsh(H,1,v0=AA,which='SA')
        AA = np.reshape(V[:,0], H.req_shape)
        A,S,B = fine_grain_MPS(AA, [A.shape[0], B.shape[0]])
        A,S,B,trunc,m = truncate_SVD(A,S,B,m)
        
        if (dir == 'right'):
            B = np.einsum("ij,sjk->sik", np.diag(S), B)
        else:
            assert dir == 'left'
            A = np.einsum("sij,jk->sik", A, np.diag(S))
        return E[0], A, B, trunc, m

def two_site_dmrg(MPS, MPO, m, sweeps=50, conv=1e-6, U1=False, target_qn = None, not_conv_err = True):
    """
    Driver function to perform sweeps of 2-site DMRG


    Parameters
    ----------
    MPS : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    sweeps : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    MPS : TYPE
        DESCRIPTION.

    """

    E = construct_E(MPS, MPO, MPS)
    F = construct_F(MPS, MPO, MPS, target_qn=target_qn)
    F.pop()
    
    # Skip dense expectation check for U1 to avoid crash
    Eold = 0.0 
    converged = False
    gauge = None
    for sweep in range(0, int(sweeps/2)):
        for i in range(0, len(MPS)-2): 
            Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1=U1
            )
            print("Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2,i,i+1, Energy, states, trunc))

            E.append(contract_from_left(MPO[i], MPS[i], E[-1], MPS[i]))
            F.pop()

        if abs(Energy - Eold) < conv:
            print("DMRG Converged at sweep {}. \n Total energy = {}".format(sweep, Energy))
            converged = True
            gauge = "Left"
            break
        else:
            Eold = Energy

        for i in range(len(MPS)-2, 0, -1): 
            Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1=U1
            )
            print("Sweep {} Sites {},{}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2+1, i, i+1, Energy, states, trunc))
            F.append(contract_from_right(MPO[i+1], MPS[i+1], F[-1], MPS[i+1]))
            E.pop()

        if abs(Energy - Eold) < conv:
            print("DMRG Converged at sweep {}. \n Total energy = {}".format(sweep, Energy))
            converged = True
            gauge = "Right"
            break
        else:
            Eold = Energy
    if not_conv_err == True:
        if converged == False:
            raise ValueError("DMRG did not converge within the given number of sweeps, if you wish to disable this error, set not_conv_err = False. or you should increase the number of sweeps.")
    if not_conv_err == False:
        if converged == False:
            print("DMRG did not converge within the given number of sweeps, returning the last result.")
    if gauge == None:
        gauge = "Right"

    return Energy, MPS, gauge


def expect_mps(bra, MPO, ket=None):
    """
    Evaluate the expectation value of an MPO on a given MPS
    .. math::

         <A|MPO|B>

    Parameters
    ----------
    AList : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    BList : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    AList = bra
    BList = ket

    if ket is None:
        ket = bra

    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]





class DMRG:
    """
    ground state finite DMRG in MPO/MPS framework
    """
    def __init__(self, H, D, nsweeps=None, init_guess=None, opt='2site', U1 = False, target_qn = None, not_conv_err = True):
        """


        Parameters
        ----------
        H : TYPE
            MPO of H.
        D : TYPE
            maximum bond dimension.
        nsweeps : TYPE, optional
            DESCRIPTION. The default is None.
        init_guess : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.H = H
        self.D = D
        self.nsweeps = nsweeps
        self.opt = opt

        self.init_guess = init_guess
        self.mps = None
        self.e_tot = None
        self.U1 = U1
        self.target_qn = target_qn
        self.ground_state = None
        self.ground_state_raw = None
        self.not_conv_err = not_conv_err


    def run(self):

        if self.init_guess is None:
            raise ValueError('Invalid initial guess.')

        if self.U1:
            if isinstance(self.init_guess, list) and not isinstance(self.init_guess[0], BlockTensor):
                self.init_guess = dense_to_symmetric(self.init_guess, phys_qns=None)

            if self.target_qn is not None:
                target_qn = int(self.target_qn)
            else:
                # otherwise infer from last-site right-bond charge
                qs = sorted({key[1] for key in self.init_guess[-1].data.keys()})
                if len(qs) != 1:
                    raise ValueError(f"Ambiguous total charge on last bond: {qs}. Set DMRG(..., target_qn=...) explicitly.")
                target_qn = qs[0]

        if self.opt == '1site':
            fDMRG_1site_GS_OBC(self.H, self.D, self.nsweeps)

        else:
            self.e_tot, self.ground_state_raw, self.gauge = two_site_dmrg(
                self.init_guess, self.H, self.D, self.nsweeps, U1=self.U1, target_qn=self.target_qn, not_conv_err = self.not_conv_err)
            self.ground_state = MPS(self.ground_state_raw)
        return self

    def expect(self, e_ops):
        """
        Compute expectation value of ground states

        Parameters
        ----------
        e_ops : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """

        psi = self.ground_state

        return [expect(psi, e_op) for e_op in e_ops]

    def make_rdm(self, idx=None):
        """
        Calculate 1-site reduced density matrix of the ground state.
        Wrapper for MPS.calc_1site_rdm
        \gamma_{ij} = < 0| c_j^\dagger c_i | 0 >
        """
        if self.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.calc_1site_rdm(idx)

    def make_rdm2(self, idx_pairs=None):
        """
        Calculate 2-site reduced density matrix of the ground state.
        Wrapper for MPS.calc_2site_rdm
        """
        if self.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.calc_2site_rdm(idx_pairs)

    def nelec_dmrg(self, idx=None, weights=None, return_local=False, normalize=True, atol=1e-10):
        """
        Return the total number of electrons deduced from the 1-site RDM(s).

        Parameters
        ----------
        idx : None | int | list[int]
            Which sites to include. None -> all sites.
        weights : None | array-like
            Local occupation eigenvalues for each physical basis state (dense case).
            If None, uses common defaults:
            d=2 -> [0,1]
            d=4 -> [0,1,1,2]
            For U(1) BlockTensor RDM, weights are taken from the block quantum number q.
        return_local : bool
            If True, also return a dict {i: <n_i>}.
        normalize : bool
            If True, divide by Tr(rho_i) when Tr deviates from 1 (robust against gauge / normalization issues).
        atol : float
            Tolerance for trace normalization check.

        Returns
        -------
        float  or  (float, dict)
            Total electron number, optionally with site-resolved occupations.
        """
        import numpy as np

        rdm = self.make_rdm(idx)

        local = {}
        for i, rho in rdm.items():
            # --- U(1) BlockTensor RDM path: use block labels q as particle number ---
            if SYMMETRY_AVAILABLE and (BlockTensor is not None) and isinstance(rho, BlockTensor):
                tr = 0.0 + 0.0j
                n  = 0.0 + 0.0j
                for (q_bra, q_ket), blk in rho.data.items():
                    if q_bra == q_ket:
                        t = np.trace(blk)
                        tr += t
                        n  += float(q_bra) * t

                if normalize and abs(tr - 1.0) > atol and abs(tr) > atol:
                    n = n / tr

                local[i] = n
                continue

            # --- Dense ndarray RDM path ---
            rho = np.asarray(rho)
            d = rho.shape[0]
            tr = np.trace(rho)

            if weights is None:
                if d == 2:
                    w = np.array([0.0, 1.0], dtype=float)
                elif d == 4:
                    w = np.array([0.0, 1.0, 1.0, 2.0], dtype=float)
                else:
                    raise ValueError(
                        f"nelec_dmrg: cannot infer occupation weights for local dim d={d}. "
                        "Pass `weights=` explicitly."
                    )
            else:
                w = np.asarray(weights, dtype=float)
                if w.shape[0] != d:
                    raise ValueError(f"nelec_dmrg: weights length {w.shape[0]} != local dim {d}.")

            n = np.sum(w * np.real(np.diag(rho)))
            if normalize and abs(tr - 1.0) > atol and abs(tr) > atol:
                n = n / np.real(tr)

            local[i] = n + 0.0j  # keep consistent type (will be real in practice)

        total = np.real(np.sum(list(local.values()))).item()
        if return_local:
            # convert local to real python floats when possible
            local_real = {i: np.real(v).item() for i, v in local.items()}
            return total, local_real
        return total



def autoMPO(h1e, eri):
    """
    write the Hamiltonian into the MPO form

    .. math::

        H = \sum_{i,j} h_{ij} E_{ij} + \sum_{i < j} v_{ij} n_i n_j

        E_{ij} = \sum_\sigma c_{i\sigma}^\dagger c_{j\sigma}

    Parameters
    ----------
    h1e : TYPE
        one-electron core Hamiltonian.
    eri : TYPE
        electron-repulsion integral

    Returns
    -------
    None.

    """
    pass


class TEBD(DMRG):

    def run(self, psi0):
        return tebd(psi0, self.U, chi_max=self.D)


def tebd(B_list, s_list, U_list, chi_max):
    """
    Use TEBD to optmize the MPS and to rduce it back to the orginal size.
    """
    d = B_list[0].shape[0]
    L = len(B_list)

    for p in [0,1]:

        for i_bond in np.arange(p,L-1,2):
            i1=i_bond
            i2=i_bond+1

            chi1 = B_list[i1].shape[1]
            chi3 = B_list[i2].shape[2]

            # Construct theta matrix #
            C = np.tensordot(B_list[i1],B_list[i2],axes=(2,1))
            #C = np.einsum('aij, bjk -> aibk', B_list[i1], B_list[i2])
            C = np.tensordot(C,U_list[i_bond],axes=([0,2],[2,3]))
            print(np.shape(C))

            # ? Why not directly SVD the C tensor?

            theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))

            C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))
            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta)
            Z=Z.T

            W = np.dot(C,Z.conj())
            chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

            # Obtain the new values for B and l #
            invsq = np.sqrt(sum(Y[:chi2]**2))
            s_list[i2] = Y[:chi2]/invsq
            B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq
            B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))


class TDVP(DMRG):
    pass



def fDMRG_1site_GS_OBC(H,D,Nsweeps):
    '''
    Function that implements finite-system DMRG (one-site update version) to obtain the ground state of an input
            Hamiltonian MPO (order of legs: left-bottom-right-top), 'H', that represents a system with open boundary
            conditions.

    Notes:
            - the outputs are the ground state energy at every step of the algorithm, 'E_list', and the ground state
                MPS (order of legs: left-bottom-right) at the final step, 'M'.
            - the maximum bond dimension allowed for the ground state MPS is an input, 'D'.
            - the number of sweeps is an input, 'Nsweeps'.
    '''
    N = len(H) #nr of sites

    # random MPS (left-bottom-right)
    M = []
    M.append(np.random.rand(1, np.shape(H[0])[3],D))

    for l in range(1,N-1):
        M.append(np.random.rand(D,np.shape(H[l])[3],D))
    M.append(np.random.rand(D,np.shape(H[N-1])[3],1))

    ## normalized MPS in right canonical form
    # M = LeftCanonical(M)
    M = RightCanonical(M)

    # Hzip
    '''
        Every step of the finite-system DMRG consists in optimizing a local tensor M[l] of an MPS in site
            canonical form. The value of l is sweeped back and forth between 0 and N-1.

        For a given l, we define Hzip as a list with N+2 elements where:

            - Hzip[0] = Hzip[N+1] = np.ones((1,1,1))

            - Hzip[it] =

                /--------------M[it-1]--3--
                |             \|
                |              |
                |              |
                Hzip[it-1]-----H[it-1]--2--          for it = 1, 2, ..., l
                |              |
                |              |
                |             /|
                \--------------M[it-1]^†--1--

            - Hzip[it] =

                --1--M[it-1]-----\
                     |/          |
                     |           |
                     |           |
                --2--H[it-1]-----Hzip[it+1]          for it = l+1, l+2, ..., N
                     |           |
                     |           |
                     |\          |
                --3--M[it-1]^†---/

        Here, we initialize Hzip considering l=0 (note that this is consistent with starting with a random MPS in
            right canonical form). Consistently, we will start the DMRG routine with a right sweep.
    '''
    Hzip = [np.ones((1,1,1)) for it in range(N+2)]
    for l in range(N-1,-1,-1):
        Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    # DMRG routine
    E_list = []
    for itsweeps in range(Nsweeps):
        ## right sweep
        for l in range(N):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                    np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            '''
                Note: for performance purposes, we initialize Lanczos with the previous version of the local
                    tensor M[l].
            '''
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the right sweep, the local tensor M[l] obtained from Lanczos has to be left normalized.
                    This is achieved by SVD. The remaining S*Vdag is contracted with M[l+1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(U,(np.shape(Taux)[0],np.shape(Taux)[1],np.shape(U)[1]))
            if l < N-1:
                M[l+1] = np.einsum('ij,jkl',np.matmul(np.diag(S),Vdag),M[l+1])

            ### update Hzip
            Hzip[l+1] = ZipperLeft(Hzip[l],M[l].conj().T,H[l],M[l])

        ## left sweep
        for l in range(N-1,-1,-1):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                   np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the left sweep, the local tensor M[l] obtained from Lanczos has to be right normalized.
                    This is achieved by SVD. The remaining U*S is contracted with M[l-1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(Vdag,(np.shape(Vdag)[0],np.shape(Taux)[1],np.shape(Taux)[2]))
            if l > 0:
                M[l-1] = np.einsum('ijk,kl',M[l-1],np.matmul(U,np.diag(S)))

            ### update Hzip
            Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    return E_list,M





if __name__ == '__main__':

    ##
    ## Parameters for the DMRG simulation for spin-1/2 chain
    ## To apply to fermions, we only need to change the MPO if H
    ##

    d=2   # local bond dimension, 0=up, 1=down
    N=10 # number of sites

    ## initial state |+-+-+-+-+->

    InitialA1 = np.zeros((d,1,1))
    InitialA1[0,0,0] = 1
    InitialA2 = np.zeros((d,1,1))
    InitialA2[1,0,0] = 1

    initial_mps = [InitialA1, InitialA2] * int(N/2)

    ## Local operators
    I = np.identity(2)
    Z = np.zeros((2,2))
    Sz = np.array([[0.5,  0  ],
                 [0  , -0.5]])
    Sp = np.array([[0, 0],
                 [1, 0]])
    Sm = np.array([[0, 1],
                 [0, 0]])

    ## Hamiltonian MPO
    W = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z],
                  [Z,  Z,      Z,      Z,  Sz],
                  [Z,  Z,      Z,      Z,  Sm],
                  [Z,  Z,      Z,      Z,  Sp],
                  [Z,  Z,      Z,      Z,   I]])

    print(W.shape)

    # left-hand edge is 1x5 matrix
    Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z]])

    # right-hand edge is 5x1 matrix
    Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])

    # the complete MPO
    H = MPO = [Wfirst] + ([W] * (N-2)) + [Wlast]

    dmrg = DMRG(H, D=10, nsweeps=8)
    dmrg.init_guess = initial_mps
    dmrg.run()

    



    # # MPO for H^2, to calculate the variance
    # HamSquared = product_MPO(MPO, MPO)

    # 8 sweeps with m=10 states
    # two_site_dmrg(MPS, MPO, 10, 8)

# # energy and energy squared
# E_10 = Expectation(MPS, MPO, MPS);
# Esq_10 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=20 states
# two_site_dmrg(MPS, MPO, 20, 2)

# # energy and energy squared
# E_20 = Expectation(MPS, MPO, MPS);
# Esq_20 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=30 states
# two_site_dmrg(MPS, MPO, 30, 2)

# # energy and energy squared
# E_30 = Expectation(MPS, MPO, MPS);
# Esq_30 = Expectation(MPS, HamSquared, MPS);

# Energy = Expectation(MPS, MPO, MPS)
# print("Final energy expectation value {}".format(Energy))

# # calculate the variance <(H-E)^2> = <H^2> - E^2

# print("m=10 variance = {:16.12f}".format(Esq_10 - E_10*E_10))
# print("m=20 variance = {:16.12f}".format(Esq_20 - E_20*E_20))
# print("m=30 variance = {:16.12f}".format(Esq_30 - E_30*E_30))