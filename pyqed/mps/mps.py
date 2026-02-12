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
    from pyqed.mps.symmetry import BlockTensor, tensordot, solve_davidson, QN, SymmetryManager
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

# below is some helper functions for the dmrg sweep with U(1) symmetry.
def svd_symmetric(AA, cutoff=1e-10, m_max=None):

    AA_perm = AA.transpose(0, 2, 1, 3)

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

    sv_list.sort(reverse=True, key=lambda x: x[0])
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
    final_S = {} # Capture S

    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
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

def dense_to_symmetric_mpo(dense_mpo_list, site_qn_maps, tol=1e-12):
    """
    General converter from Dense MPO (L, R, Out, In) to Symmetric BlockTensor.
    Includes strict type checking to prevent Integer/QN mismatches.
    """
    if not SYMMETRY_AVAILABLE:
        raise ImportError("Symmetry module required.")
    sym_H = []
    # Determine Zero QN type
    first_val = list(site_qn_maps[0].values())[0]
    # Ensure zero_qn is of the same type as the map values (QN)
    zero_qn = first_val * 0 if isinstance(first_val, tuple) else 0
    # Track allowed Right-Bond QNs. Start with Vacuum (Left=0).
    # Store (Dense_Index, QN_Value)
    current_nodes = {(0, zero_qn)}
    print(f"  [MPO Convert] Start. Sites={len(dense_mpo_list)}, ZeroQN={zero_qn} (Type: {type(zero_qn)})")
    for site_idx, W in enumerate(dense_mpo_list):
        new_data = {}
        next_nodes = set()
        phys_qns = site_qn_maps[site_idx]
        # Metadata for BlockTensor
        all_phys_out = sorted(list(set(phys_qns.values())))
        all_phys_in = sorted(list(set(phys_qns.values())))
        # Optimize lookup
        valid_incoming = {}
        for l_idx, q_l in current_nodes:
            if l_idx not in valid_incoming: valid_incoming[l_idx] = set()
            valid_incoming[l_idx].add(q_l)
        # W shape: (Left, Right, Out, In)
        idxs = np.nonzero(np.abs(W) > tol)
        for i in range(len(idxs[0])):
            l, r, out_s, in_s = idxs[0][i], idxs[1][i], idxs[2][i], idxs[3][i]
            val = W[l, r, out_s, in_s]
            if l not in valid_incoming:
                continue
            # Retrieve Physical QNs
            q_out = phys_qns[out_s]
            q_in = phys_qns[in_s]
            # Q_Right = Q_Left - (Q_Out - Q_In)
            flux = q_out - q_in
            for q_l in valid_incoming[l]:
                # Ensure q_l is a QN (tuple), not an int
                if not isinstance(q_l, tuple):
                    raise TypeError(f"Site {site_idx}: q_l became {type(q_l)} ({q_l})! Expected QN/tuple.")
                q_r = q_l - flux
                next_nodes.add((r, q_r))
                # Construct Key: Must be (QN, QN, QN, QN)
                key = (q_l, q_r, q_out, q_in)
                if key not in new_data: new_data[key] = []
                new_data[key].append( ((l, q_l), (r, q_r), val) )
        # Build BlockTensor Maps
        l_map = {q: sorted([x for x in current_nodes if x[1]==q]) for q in set(x[1] for x in current_nodes)}
        r_map = {q: sorted([x for x in next_nodes if x[1]==q]) for q in set(x[1] for x in next_nodes)}
        final_blocks = {}
        for key, elems in new_data.items():
            q_l, q_r, q_o, q_i = key
            # Validation
            if q_l not in l_map or q_r not in r_map:
                continue
            rows = l_map[q_l]; cols = r_map[q_r]
            row_idx = {x: k for k, x in enumerate(rows)}
            col_idx = {x: k for k, x in enumerate(cols)}
            blk = np.zeros((len(rows), len(cols), 1, 1), dtype=W.dtype)
            for (nl, nr, v) in elems:
                blk[row_idx[nl], col_idx[nr], 0, 0] = v
            final_blocks[key] = blk
        qns_L = sorted(list(l_map.keys()))
        qns_R = sorted(list(r_map.keys()))
        qns_Out = all_phys_out
        qns_In = all_phys_in
        bt = BlockTensor(final_blocks, [qns_L, qns_R, qns_Out, qns_In], [-1, 1, 1, -1])
        sym_H.append(bt)
        # Verify generated keys for first site (debug use)
        if site_idx == 0 and len(final_blocks) > 0:
            sample_key = next(iter(final_blocks.keys()))
            if not isinstance(sample_key[0], tuple):
                 print(f"  [ERROR] Site 0 generated INTEGER keys: {sample_key}. Expected QNs.")
        current_nodes = next_nodes
    return sym_H


class UniformMPS:
    #TODO uniform MPS
    def __init__(self, Bs, labels='lpr'):

        self.labels = labels
        self.p_idx = self.labels.index('p')




class MPS:
    def __init__(self, Bs, Ss=None, bc='finite', \
                 labels=['lv', 'p', 'rv'], homogenous=False, center=None, gauge=None):
        """
        Base class for matrix product states.
        supports flexible tensor layouts via the `labels` argument.

        Parameters
        ----------
        Bs : list of np.ndarray
            The site tensors.
            - Must be a list of rank-3 tensors.
            - Shape depends on `labels`, e.g., ['lv', 'p', 'rv'] -> means(Bond_L, Phys, Bond_R).

        Ss : list of np.ndarray, optional
            The bond singular values (Schmidt coefficients).
            - `Ss[i]` corresponds to the bond between site `i` and `i+1`.
            - Used for calculating entanglement entropy and handling canonical forms.

        homogenous : bool, optional
            If True, assumes all sites share the same physical dimension structure. Default is True.

        bc : str, optional
            Boundary conditions. Options:
            - 'finite': Open Boundary Conditions (OBC).
            - 'infinite': Infinite Boundary Conditions (IBC/PBC).
            default is 'finite'.

        labels : list of str, optional
            Describes the leg index order in tensors `Bs`.
            The default is ['lv', 'p', 'rv'].

            Supported Keys:
            - 'lv': Left-Virtual (Bond to the left)
            - 'rv': Right-Virtual (Bond to the right)
            - 'p':  Physical (Local Hilbert space)

            Common Examples:
            - ['lv', 'p', 'rv']: Standard Dense format (Left, Phys, Right).
            - ['p', 'lv', 'rv']: "Physics" format (Phys, Left, Right).
            - ['lv', 'rv', 'p']: "BlockTensor" format (Left, Right, Phys).

        Attributes
        ----------
        L : int
            Number of sites (length of the chain).
        nbonds : int
            Number of bonds (L-1 for finite, L for infinite).
        dim : int
            Physical dimension (d) of the sites.
        lv_idx, p_idx, rv_idx : int
            Cached integer positions of the axes based on `labels`.
        Center : int
            Canonical center site index. Default is -1 (no specific center).
        """
        assert bc in ['finite', 'periodic']
        self.Bs = self.data = self.factors = Bs
        self.Ss = Ss
        self.gauge = gauge

        # leg sequence
        if labels is None:
            warnings.warn("MPS labels not specified, assuming ['lv', 'p', 'rv'].")
            self.labels = ['lv', 'p', 'rv']
        else:
            self.labels = labels
        try:
            self.lv_idx = self.labels.index('lv')
            self.rv_idx = self.labels.index('rv')
            self.p_idx = self.labels.index('p')
        except ValueError as e:
            missing_label = str(e).split()[-1]
            raise ValueError(f"MPS initialization failed: The label list {self.labels} is missing the required label {missing_label}.")
        if len(self.labels) != 3:
             warnings.warn(f"Warning: You provided {len(self.labels)} labels but MPS tensors are usually Rank-3. Ensure your boundaries have dummy indices.")



        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'finite' else self.L


        if self.gauge == 'mixed':
            if not 0 <= self.center <= self.L:
                raise ValueError(f"Invalid center index {self.center} for MPS with {self.L} sites.")
            if self.center is None:
                warnings.warn("MPS created without a canonical center. Some operations may require canonicalization first.")

        self.homogenous = homogenous
        if homogenous:
            try:
                self.dim = Bs[0].shape[self.p_idx]
            except TypeError:
                if hasattr(Bs[0], 'data'):
                    # U(1) tensors in this code are (Left, Right, Phys) -> Index 2 TODO: get that to Left Phy Right
                    phys_dims = {}
                    for key, block in Bs[0].data.items():
                        # key is (qL, qR, qP)
                        q_p = key[2]
                        if q_p not in phys_dims:
                            phys_dims[q_p] = block.shape[2]
                    self.dim = sum(phys_dims.values())
                else:
                    self.dim = 0
        else:  # inhomogenous

            self.dims = []
            for B in Bs:
                try:
                    self.dims.append(B.shape[1])
                except TypeError:
                    if hasattr(B, 'data'):
                        phys_dims = {}
                        for key, block in B.data.items():
                            q_p = key[2]
                            if q_p not in phys_dims:
                                phys_dims[q_p] = block.shape[2]
                        self.dims.append(sum(phys_dims.values()))
                    else:
                        self.dims.append(0)

    def check_sanity(self):
        # TODO make sure the specified gauge is correct
        pass

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss] if self.Ss is not None else None, self.bc, labels=self.labels)

    def bond_orders(self):
        """Return right bond dimensions for each site."""
        return [t.shape[2] for t in self.factors]

    def norm(self):
        """
        Calculate the MPS norm :math:`N = \sqrt{<\psi|\psi>}` robustly using standard layouts.
        """
        if self.gauge is None:

            val = np.ones((1, 1), dtype=complex)
            for i in range(self.L):
                B = self._get_std_B(i) # (lv, p, rv)
                # Contract Left legs: val(a, b) * B(b, p, r) -> T(a, p, r)
                T = np.tensordot(val, B, axes=(1, 0))
                # Contract with conjugate: T(a, p, r) * B*(a, p, r') -> val(r, r')
                val = np.tensordot(T, B.conj(), axes=([0, 1], [0, 1]))
            return np.abs(val[0, 0])

        elif self.gauge == 'right_canonical':
            B = self.Bs[0]
            return np.einsum('aib, aib ->', B.conj(), B)

        elif self.gauge == 'left_canonical':
            B = self.Bs[-1]
            return np.einsum('aib, aib ->', B.conj(), B)

        elif self.gauge == 'mixed':
            B = self.Bs[self.center]
            return np.einsum('aib, aib ->', B.conj(), B)

    def normalize(self):
        """
        normalize a MPS norm :math:`N = \sqrt{<\psi|\psi>}`
        """
        if self.gauge is None:

            val = np.ones((1, 1), dtype=complex)
            for i in range(self.L):
                B = self._get_std_B(i) # (lv, p, rv)
                # Contract Left legs: val(a, b) * B(b, p, r) -> T(a, p, r)
                T = np.tensordot(val, B, axes=(1, 0))
                # Contract with conjugate: T(a, p, r) * B*(a, p, r') -> val(r, r')
                val = np.tensordot(T, B.conj(), axes=([0, 1], [0, 1]))

            if val < 1e-12: raise warnings.warn('Norm {val} is too small.')

            self.Bs[0] /=  np.sqrt(np.abs(val[0, 0]))

        elif self.gauge == 'right_canonical':
            B = self.Bs[0]
            self.Bs[0] /= np.sqrt(np.einsum('aib, aib ->', B.conj(), B))

        elif self.gauge == 'left_canonical':
            B = self.Bs[-1]
            self.Bs[-1] /= np.einsum('aib, aib ->', B.conj(), B)

        elif self.gauge == 'mixed':
            B = self.Bs[self.center]
            self.Bs[self.center] /= np.einsum('aib, aib ->', B.conj(), B)

        return self


    def set_labels(self, new_labels):
        """
        Allow user to manually assign/correct labels after creation.

        Common examples:
        - ['lv', 'p', 'rv']  (Left-Virtual, Physical, Right-Virtual)
        - ['lv', 'rv', 'p']  (Left-Virtual, Right-Virtual, Physical)
        - ['p', 'lv', 'rv']  (Physical, Left-Virtual, Right-Virtual)
        """
        self.labels = new_labels
        try:
            self.lv_idx = self.labels.index('lv')
            self.rv_idx = self.labels.index('rv')
            self.p_idx = self.labels.index('p')
        except ValueError as e:
            missing_label = str(e).split()[-1]
            raise ValueError(f"MPS initialization failed: The label list {self.labels} is missing the required label {missing_label}.")
        if len(self.labels) != 3:
             warnings.warn(f"Warning: You provided {len(self.labels)} labels but MPS tensors are usually Rank-3. Ensure your boundaries have dummy indices.")


    def to_order(self, target_labels):
        """Returns a new MPS with tensors transposed to target_labels."""
        if self.labels == target_labels:
            return self.copy()

        perm = [self.labels.index(l) for l in target_labels]
        new_Bs = [B.transpose(perm) for B in self.Bs]
        return MPS(new_Bs, self.Ss, self.bc, labels=target_labels)

    def transpose(self, labels):
        """
        transpose ALL tensors to target sequence

        Parameters
        ----------
        labels : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.to_order(labels)

    def _get_std_B(self, i):
        """
        Internal Helper: Returns B[i] transposed to standard [Left, Phys, Right].
        """
        B = self.Bs[i]
        # Check if it has data AND that data is a dict (BlockTensor structure)
        if hasattr(B, 'data') and isinstance(B.data, dict):
            return B
        return B.transpose(self.lv_idx, self.p_idx, self.rv_idx)

    def get_bond_dimensions(self):
        try:
            return [self.Bs[i].shape[self.rv_idx] for i in range(self.nbonds)]
        except (TypeError, AttributeError):
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

    def __add__(self, other):
        """
        Sum of two MPS states: |Result> = |self> + |other>

        Logic:
        - First Site: Concatenate [A, B] horizontally.
        - Middle Sites: Block Diagonal.
        - Last Site: Concatenate [[A], [B]] vertically.

        not using block_diag from scipy since first site need to be row vector and last site need to be column vector.
        """
        assert self.L == other.L

        C = []
        for j in range(self.L):
            A = self._get_std_B(j)
            B = other._get_std_B(j)

            la, d, ra = A.shape
            lb, _, rb = B.shape

            if j == 0:
                # first site is Row Vector [A, B]
                # Left dim stays 1 (assuming La=Lb=1)
                # Right dim sums: Ra + Rb
                new_tensor = np.zeros((la, d, ra + rb), dtype=np.result_type(A, B))
                new_tensor[:, :, :ra] = A
                new_tensor[:, :, ra:] = B

            elif j == self.L - 1:
                # last iste is Column Vector [[A], [B]]
                # Left dim sums: La + Lb
                # Right dim stays 1 (assuming Ra=Rb=1)
                new_tensor = np.zeros((la + lb, d, ra), dtype=np.result_type(A, B))
                new_tensor[:la, :, :] = A
                new_tensor[la:, :, :] = B

            else:
                # middles sites are Block Diagonal
                # Left sums, Right sums
                new_tensor = np.zeros((la + lb, d, ra + rb), dtype=np.result_type(A, B))
                new_tensor[:la, :, :ra] = A
                new_tensor[la:, :, ra:] = B

            C.append(new_tensor)

        return MPS(C, labels=['lv', 'p', 'rv'])

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition
        at any of the bonds.
        """
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i-1].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def get_theta1(self, i):
        """
        Calculate effective single-site wave function on sites i.
        Automatically detects Left/Right canonical forms based on self.center.
        """
        tensor = self._get_std_B(i)
        if self.center == None:
            raise NotImplementedError("need to first do canonicalization to have a center site for get_theta1(), currently have not implemented the functions for that. TODO: maybe we will do self.shift_center(i) later. Buy me a coffee to prioritize this feature.")
        # Right of Center
        if i > self.center:
            if i == 0:
                if self.bc == 'infinite':
                    S_left = self.Ss[-1]
                else:
                    return tensor # Open Boundary, no left weights
            else:
                S_left = self.Ss[i-1]
            # Contract S_left (diag) with Tensor (Left Index 0)
            return np.tensordot(np.diag(S_left), tensor, axes=([1], [0]))
        # Left of Center
        elif i < self.center:
            S_right = self.Ss[i]
            # Contract Tensor (Right Index 2) with S_right (diag)
            return np.tensordot(tensor, np.diag(S_right), axes=([2], [0]))
        # At Center
        else:
            return tensor

    def get_theta2(self, i):
        """
        Calculate effective two-site wave function on sites i, i+1.
        Handles crossing the orthogonality center.
        """
        j = (i + 1) % self.L
        if self.center < 0 or self.center > self.L -1:
            raise NotImplementedError("need to first do canonicalization to have a center site for get_theta2(), currently have not implemented the functions for that. TODO: maybe we will do self.shift_center(i) later. Buy me a coffee to prioritize this feature.")
        # The bond (i, j) is the center
        # i is Left-Canonical (A), j is Right-Canonical (B)
        if i == self.center:
            A_i = self._get_std_B(i) # Pure tensor
            S_mid = self.Ss[i]
            B_j = self._get_std_B(j) # Pure tensor
            # A_i * S_mid
            temp = np.tensordot(A_i, np.diag(S_mid), axes=([2], [0]))
            # (A_i * S_mid) * B_j
            return np.tensordot(temp, B_j, axes=([2], [0]))
        # Entire block is to the Right of Center
        # theta1(i) * B_j
        elif self.center != -1 and i > self.center:
            return np.tensordot(self.get_theta1(i), self._get_std_B(j), axes=([2], [0]))
        # Entire block is to the Left of Center
        # A_i * theta1(j)
        elif self.center != -1 and j < self.center:
            return np.tensordot(self._get_std_B(i), self.get_theta1(j), axes=([2], [0]))

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            # theta: [L, P, R]
            theta = self.get_theta1(i)

            # op: [P_out, P_in]. Contract P_in (1) with theta P (1)
            # op_theta: [P_out, L, R]
            op_theta = np.tensordot(op, theta, axes=(1, 1))

            # Contract with theta*: [L, P, R]
            # Match: L(1)-L(0), R(2)-R(2), P_out(0)-P(1)
            # einsum: 'plr,lpr->'
            val = np.tensordot(op_theta, theta.conj(), axes=([0, 1, 2], [1, 0, 2]))
            result.append(val)
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            # theta: [L, Pi, Pj, R]
            theta = self.get_theta2(i)

            # op[i]: [Pi_out, Pj_out, Pi_in, Pj_in]
            # Contract (Pi_in, Pj_in) [2,3] with theta (Pi, Pj) [1,2]
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))

            # op_theta: [Pi_out, Pj_out, L, R]
            # Contract with theta*: [L, Pi, Pj, R]
            val = np.tensordot(op_theta, theta.conj(), axes=([0, 1, 2, 3], [1, 2, 0, 3]))
            result.append(val)
        return np.real_if_close(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        B = self._get_std_B(0)  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self._get_std_B(i)
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
            B = self._get_std_B(k)  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self._get_std_B(j)  # vL k vR
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

            A = self._get_std_B(n)
            V = other._get_std_B(n)

            al, d, ar = A.shape
            vl, d, vr = V.shape

            c = np.einsum('aib, cid -> acibd', V, A)
            c = c.reshape((al * vl, d, ar * vr))
            As.append(c.copy())

        return MPS(As, labels=['lv', 'p', 'rv'])

    # def __add__(self, other):
    #     pass

    # def evolve_t(self):
    #     pass

    def left_canonicalize(self):
        """
        Sweeps from Left (0) to Right (L-1) to transform the MPS into Left-Canonical Form.
        Effect:
        - Tensors Bs[0]...Bs[L-2] become Left-Isometries (A).
        - Populates self.Ss with bond weights.
        - Moves orthogonality center to the last site (L-1).
        """
        if SYMMETRY_AVAILABLE and isinstance(self.Bs[0], BlockTensor):
            self.Center = self.L - 1
            return
        if self.Ss is None or len(self.Ss) != self.nbonds:
            self.Ss = [None] * self.nbonds
        # Get permutation
        perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])
        # Sweep Left -> Right
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            dl, dp, dr = B.shape
            # Reshape (Left * Phys, Right)
            mat = B.reshape(dl * dp, dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi = len(S)
            # Update Site i and Reshape to (L,P,R) and Transpose back
            self.Bs[i] = U.reshape(dl, dp, chi).transpose(perm_inv)
            # Ss[i] is the bond between i and i+1
            self.Ss[i] = S / np.linalg.norm(S)
            # Pass weights (S * Vh)
            # Matrix M = diag(S) * Vh  (Shape: chi, dr)
            M = np.dot(np.diag(S), Vh)
            # Contract M with B_next on its Left index
            B_next = self._get_std_B(i+1) # [Left_Old, Phys, Right]
            B_next_updated = np.tensordot(M, B_next, axes=([1], [0])) # (New_Bond, Phys, Right)
            self.Bs[i+1] = B_next_updated.transpose(perm_inv)
        # Normalize
        B_last = self._get_std_B(self.L - 1)
        B_last /= np.linalg.norm(B_last)
        self.Bs[self.L - 1] = B_last.transpose(perm_inv)
        # Update Center
        self.Center = self.L - 1

    def right_canonicalize(self):
        """
        Sweeps from Right (L-1) to Left (0) to transform the MPS into Right-Canonical Form.
        Effect:
        - Tensors Bs[1]...Bs[L-1] become Right-Isometries (B).
        - Populates self.Ss with bond weights.
        - Moves orthogonality center to the first site (0).
        """
        if SYMMETRY_AVAILABLE and isinstance(self.Bs[0], BlockTensor):
            self.Center = 0
            return
        if self.Ss is None or len(self.Ss) != self.nbonds:
            self.Ss = [None] * self.nbonds
        # Get permutation
        perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])
        # Sweep Right -> Left
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            dl, dp, dr = B.shape
            # Reshape (Left, Phys * Right)
            mat = B.reshape(dl, dp * dr)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi = len(S)
            # Update Site i (The Isometry Vh)
            # Reshape Vh to (New_Bond, Phys, Right) and Transpose back
            self.Bs[i] = Vh.reshape(chi, dp, dr).transpose(perm_inv)
            # Ss[i-1] is the bond between i-1 and i
            self.Ss[i-1] = S / np.linalg.norm(S)
            # Pass weights (U * S)
            # Matrix M = U * diag(S) (Shape: dl, chi)
            M = np.dot(U, np.diag(S))
            # Contract B_prev with M on its Right index
            B_prev = self._get_std_B(i-1) # [Left, Phys, Right_Old]
            B_prev_updated = np.tensordot(B_prev, M, axes=([2], [0])) # (Left, Phys, New_Bond)
            self.Bs[i-1] = B_prev_updated.transpose(perm_inv)
        # Normalize
        B_first = self._get_std_B(0)
        B_first /= np.linalg.norm(B_first)
        self.Bs[0] = B_first.transpose(perm_inv)
        # Update Center
        self.Center = 0

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
        compressed_factors = compress(self.factors, chi_max)
        if isinstance(compressed_factors, tuple):
            compressed_factors = compressed_factors[0]
        return MPS(compressed_factors, labels=['lv','p','rv'])

    def make_rdm1(self, idx=None):
        return self.calc_1site_rdm(idx)

    def calc_1site_rdm(self, idx=None):
        """
        Calculate 1-site reduced density matrices.

        Dense (numpy) MPS path: uses standard [Left, Phys, Right] layout via _get_std_B.

        U(1) (BlockTensor) path: builds left/right overlap environments using the
        same contraction logic as the DMRG sweeps (contract_from_left/right),
        but leaves the physical indices at the target site open.
        TODO: U(1) branch is now still in Left Phys Right layout for MPS tensors.  Need to standardize.
        Both Branch give numpy d×d matrices for convenience.
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
                # Deduce the correct "Zero" type from physics
                sample_qn = phys_qns[0]
                # Multiply by 0 to get QN(0,0) or 0 depending on type
                zero_qn = sample_qn * 0

                from collections import defaultdict
                idxs_by_q = defaultdict(list)
                for k, q in enumerate(list(phys_qns)):
                    idxs_by_q[q].append(k)

                data = {}
                for q, ks in idxs_by_q.items():
                    d = len(ks)
                    # Use zero_qn for bond indices (L, R)
                    # Shape: (BondL=1, BondR=1, Out=d, In=d)
                    data[(zero_qn, zero_qn, q, q)] = np.eye(d).reshape(1, 1, d, d)

                qns = [[zero_qn], [zero_qn], list(phys_qns), list(phys_qns)]
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
            # this extract the total qn on the last bond
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

        if idx is None:
            idx = list(range(self.L))
        elif isinstance(idx, int):
            idx = [idx]
        rdm = {}
        for i in idx:
            # Get Effective Wavefunction [Left, Phys, Right]
            theta = self.get_theta1(i)
            # Contract Left_env and Right_env
            # Result is rho[Phys, Phys*]
            rho = np.tensordot(theta, theta.conj(), axes=([0, 2], [0, 2]))
            # Normalize
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho /= tr
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
            # 1) Build Left Environments
            L_env = [np.array([[1.0]])]
            curr_L = L_env[0]
            for i in range(self.L - 1):
                B = self._get_std_B(i)
                # Contract L_env(Bra, Ket) with B(Ket, P, R) -> temp(Bra, P, R)
                temp = np.tensordot(L_env[-1], B, axes=(1, 0))
                # Contract temp with B*(Bra, P, R*) -> L_next(R, R*) -> Transpose to (R*, R)
                curr_L = np.tensordot(temp, B.conj(), axes=([0, 1], [0, 1])).T
                L_env.append(curr_L)

            # Build Right Environments
            R_env = [None] * self.L
            curr_R = np.array([[1.0]])
            R_env[-1] = curr_R
            for i in range(self.L - 1, 0, -1):
                B = self._get_std_B(i)
                # Contract B(L, P, R) with R_env(Bra, Ket) -> temp(L, P, Bra)
                temp = np.tensordot(B, R_env[i], axes=(2, 1))
                # Contract temp with B*(L*, P, Bra) -> R_prev(L, L*) -> Transpose to (L*, L)
                curr_R = np.tensordot(temp, B.conj(), axes=([1, 2], [1, 2])).T
                R_env[i - 1] = curr_R

            # 2) Precompute components
            # group the Environment + Site Tensor to expose Physical and Bond indices.

            # L_components[i]: [Pi, Pi*, R*, R]
            L_components = []
            for i in range(self.L):
                B = self._get_std_B(i)
                # L_env(Bra, Ket) * B(Ket, P, R) -> t(Bra, P, R)
                t = np.tensordot(L_env[i], B, axes=(1, 0))
                # t(Bra, P, R) * B*(Bra, P*, R*) -> comp(P, R, P*, R*)
                comp = np.tensordot(t, B.conj(), axes=(0, 0))
                # Reorder to [Pi, Pi*, R*, R]
                comp = comp.transpose(0, 2, 3, 1)
                L_components.append(comp)

            # R_components[j]: [L, L*, Pj*, Pj]
            R_components = []
            for i in range(self.L):
                B = self._get_std_B(i)
                # B(L, P, R) * R_env(Bra, Ket) -> t(L, P, Bra)
                t = np.tensordot(B, R_env[i], axes=(2, 1))
                # t(L, P, Bra) * B*(L*, P*, Bra) -> comp(L, P, L*, P*)
                comp = np.tensordot(t, B.conj(), axes=(2, 2))
                comp = comp.transpose(0, 2, 3, 1)
                R_components.append(comp)

            # 3) Assemble
            rdm = {}
            for i in range(self.L):
                js = pairs_by_i.get(i, [])
                if not js: continue

                # Start with component at i: [Pi, Pi*, R*_i, R_i]
                tensor = L_components[i]
                max_j = max(js)
                for j in range(i + 1, max_j + 1):
                    # If exist sites between i and j, propagate the bond indices by tracing over the physical indices of the intermediate sites (Transfer Matrix).
                    if j > i + 1:
                        k = j - 1
                        B = self._get_std_B(k) # [L, P, R]
                        # Apply Transfer Matrix at site k:
                        # tensor: [Pi, Pi*, L*_k, L_k]
                        # B:      [L_k, P_k, R_k]
                        # B*:     [L*_k, P_k, R*_k]
                        # Contract L with L, L* with L*, trace P_k.
                        # Output: [Pi, Pi*, R*_k, R_k]
                        tensor = np.einsum('abcd, def, geh -> abfh', tensor, B, B.conj(), optimize=True)

                    if j in js:
                        # Contract with component at j: [L_j, L*_j, Pj*, Pj]
                        # Connect Bond indices: R_prev(3) with L_j(0), R*_prev(2) with L*_j(1)
                        rho_raw = np.tensordot(tensor, R_components[j], axes=([3, 2], [0, 1]))

                        # rho_raw: [Pi, Pi*, Pj*, Pj]
                        # Reorder to standard form [Pi, Pj, Pi*, Pj*]
                        rho_ij = rho_raw.transpose(0, 3, 1, 2)

                        d_i, d_j = rho_ij.shape[0], rho_ij.shape[1]
                        rho_mat = rho_ij.reshape(d_i * d_j, d_i * d_j)

                        # Normalize
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
    def __init__(self, factors, target_qn=None, labels='left, right, up, down', homogenous=False):
        """
        class for matrix product operators.

        TODO: switch leg orders to left, up, down, right

        Parameters
        ----------
        factors : list
            list of 4-tensors of dimension. [chi1, chi2, d_up, d_down]
            chi1: left virtual bond
            chi2: right virtual bond
            d_up: physical output (bra)
            d_down: physical input (ket)
        chi_max:
            maximum bond order used in compress. Default None.

        Returns
        -------
        None.

        """
        self.factors = self.data = self.cores = factors
        self.nsites = self.L = len(factors)
        self.nbonds = self.L - 1


        if homogenous:
            self.dims = [factors[0].shape[2], ] * self.nsites
        else:
            self.dims = [t.shape[2] for t in factors]

    def bond_orders(self):
        """Return right bond dimensions for each site."""
        return [t.shape[1] for t in self.factors]

    def ground_state(self, algorithm='dmrg'):
        pass

    def dot(self, mps, D=None):
        if D is None:
            D = max(self.bond_orders()+mps.bond_orders()) if isinstance(mps, MPO) \
                else max(mps.bond_orders())*2

        # apply MPO to MPS followed by a compression
        factors = apply_mpo(self.factors, mps.factors, D)
        return MPS(factors)


    def matmul(self, other, chi_max=None):
        """
        MPO @ MPO -> MPO
        MPO @ MPS -> MPS
        """

        # if self.labels :

        if chi_max is None:
            chi_max = max(self.bond_orders()+other.bond_orders()) if isinstance(other, MPO) else max(self.bond_orders())*2

        if isinstance(other, MPO):
            # 1. Compute raw product
            # Output format of product_MPO is (Left, Right, Up, Down)
            raw_factors = product_MPO(self.factors, other.factors)

            # 2. Prepare for compress
            # decompose.py strictly requires shape: (Left, Physical, Right)
            # But our MPO product produces: (Left, Right, Up, Down)
            mps_factors = []
            phys_dims = []

            for W in raw_factors:
                s = W.shape
                # Store original physical dims (d_up, d_down)
                phys_dims.append((s[2], s[3]))

                # Step A: Merge physical legs -> (Left, Right, Phys_Combined)
                W_flat = W.reshape(s[0], s[1], s[2] * s[3])

                # Step B: Transpose to match decompose.py -> (Left, Phys_Combined, Right)
                W_ready = W_flat.transpose(0, 2, 1)

                mps_factors.append(W_ready)

            # 3. Compress (Input is Left, Phys, Right)
            # The output B will also be (Left, Phys, Right)
            compressed_factors = compress(mps_factors, chi_max)

            # 4. Restore MPO format
            final_factors = []
            for i, B in enumerate(compressed_factors):
                # B shape: (new_chi_L, d_combined, new_chi_R)

                # Step A: Transpose back -> (new_chi_L, new_chi_R, d_combined)
                B_transposed = B.transpose(0, 2, 1)

                # Step B: Split physical legs -> (new_chi_L, new_chi_R, d_up, d_down)
                d_up, d_down = phys_dims[i]
                W_final = B_transposed.reshape(B_transposed.shape[0], B_transposed.shape[1], d_up, d_down)

                final_factors.append(W_final)

            return MPO(final_factors)

        elif isinstance(other, MPS):
            new_factors, _ = apply_mpo(self.factors, other.factors, chi_max)
            return MPS(new_factors)

        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __matmul__(self, other):
        """
        UNCOMPRESSED
        
        MPO @ MPO -> MPO
        MPO @ MPS -> MPS

        Args:
            other: List of MPO tensors. shape: (Left, Right, Phys_Out, Phys_In) 
            or MPS object (left, phys, right)

        Returns:
            list: New tensors in standard (Left, Phys, Right) layout.
        """



        if isinstance(other, MPO):
            # 1. Compute raw product
            # Output format of product_MPO is (Left, Right, Up, Down)
            raw_factors = product_MPO(self.factors, other.factors)

            # 2. Prepare for compress
            # decompose.py strictly requires shape: (Left, Physical, Right)
            # But our MPO product produces: (Left, Right, Up, Down)
            factors = []
            phys_dims = []

            for W in raw_factors:
                s = W.shape
                # Store original physical dims (d_up, d_down)
                phys_dims.append((s[2], s[3]))

                # Step A: Merge physical legs -> (Left, Right, Phys_Combined)
                W_flat = W.reshape(s[0], s[1], s[2] * s[3])

                # Step B: Transpose to match decompose.py -> (Left, Phys_Combined, Right)
                W_ready = W_flat.transpose(0, 2, 1)

                factors.append(W_ready)

            # 4. Restore MPO format
            final_factors = []
            for i, B in enumerate(factors):
                # B shape: (new_chi_L, d_combined, new_chi_R)

                # Step A: Transpose back -> (new_chi_L, new_chi_R, d_combined)
                B_transposed = B.transpose(0, 2, 1)

                # Step B: Split physical legs -> (new_chi_L, new_chi_R, d_up, d_down)
                d_up, d_down = phys_dims[i]
                W_final = B_transposed.reshape(B_transposed.shape[0], B_transposed.shape[1], d_up, d_down)

                final_factors.append(W_final)

            return MPO(final_factors)
        
        elif isinstance(other, MPS): # MPO @ MPS 
        
            L = other.L
            if L != self.L:
                raise ValueError(f"MPO length does not match ({self.L}) and MPS ({L}).")

            factors = []
            for i in range(L):
                W = self.factors[i] # Shape: (wL, wR, pOut, pIn)
                # B = psi_mps._get_std_B(i) # MPS to (Left, Phys, Right)
                B = other.factors[i]
    
                # psi = U @ psi
                # B: (bL, pIn, bR)
                # W: (wL, wR, pOut, pIn)
                # Contract B[Phys] (axis 1) with W[PhysIn] (axis 3)
                T = np.tensordot(B, W, axes=(1, 3))
    
                # Result T: (bL, bR, wL, wR, pOut)
                # rearrange to: (NewLeft, NewPhys, NewRight)
                # NewLeft  = (bL, wL) -> Indices (0, 2)
                # NewPhys  = (pOut)   -> Index   (4)
                # NewRight = (bR, wR) -> Indices (1, 3)
                # Transpose: (0, 2, 4, 1, 3)
                T = T.transpose(0, 2, 4, 1, 3)
    
                # Fuse Bonds
                s = T.shape
                dim_L = s[0] * s[1]
                dim_P = s[2]
                dim_R = s[3] * s[4]
                T_flat = T.reshape(dim_L, dim_P, dim_R)
                factors.append(T_flat)

            return MPS(factors)


    def __mul__(self, other):
        """
        Element-wise multiplication of MPO with another MPO or scalar.
        MPO index order: [chi1, chi2, d_up, d_down]
        """
        # Scalar multiplication
        if isinstance(other, (int, float, complex)):
            factors_new = self.factors
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
                # W1, W2: [chi1, chi2, d_up, d_down] = [a, b, i, j]
                W1 = self.factors[i]
                W2 = other.factors[i]

                # Element-wise product on physical indices, Kronecker on virtual
                # einsum: 'abij,mnij->ambnij' then reshape to [chi1*xi1, chi2*xi2, d_up, d_down]
                core = np.reshape(
                    np.einsum('abij,mnij->ambnij', W1, W2),
                    [W1.shape[0] * W2.shape[0],   # chi1 * xi1
                     W1.shape[1] * W2.shape[1],   # chi2 * xi2
                     W1.shape[2],                  # d_up
                     W1.shape[3]])                 # d_down
                factors_new.append(core)

            return MPO(factors_new)

        else:
            raise ValueError(
                'Second operand must be MPO, int, float, or complex')


    def __add__(self, other):
        """
        Add two MPOs element-wise.
        MPO index order: [chi1, chi2, d_up, d_down]
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
            # factors: [chi1, chi2, d_up, d_down]
            W1 = self.factors[i]
            W2 = other.factors[i]
            r1_l, r1_r, d_up, d_down = W1.shape  # chi1, chi2, d_up, d_down
            r2_l, r2_r, _, _ = W2.shape

            if i == 0:
                # First site: concatenate along right bond (axis 1)
                W_sum = np.concatenate([W1, W2], axis=1)
            elif i == self.L - 1:
                # Last site: concatenate along left bond (axis 0)
                W_sum = np.concatenate([W1, W2], axis=0)
            else:
                # Middle sites: block diagonal structure
                out_dtype = np.result_type(W1.dtype, W2.dtype)
                W_sum = np.zeros((r1_l + r2_l, r1_r + r2_r, d_up, d_down),
                                    dtype=out_dtype)
                W_sum[:r1_l, :r1_r, :, :] = W1
                W_sum[r1_l:, r1_r:, :, :] = W2

            sum_factors.append(W_sum)

        return MPO(sum_factors)

    def __rmul__(self, other):
        """
        Support scalar * MPO and MPO * MPO via reflected multiplication.
        This simply delegates to __mul__ so that multiplication is
        effectively commutative for the supported operand types.
        """
        return self.__mul__(other)

    def exponential(self, constant=1.0, D=None, method='taylor', \
                    order=4, scale=0):
        """
        Calculate the exponential of an MPO: exp(constant*self).
        MPO index order: [chi1, chi2, d_up, d_down]

        Parameters
        ----------
        constant : TYPE, optional
            DESCRIPTION. The default is 1.0.
        D : TYPE, optional
            DESCRIPTION. The default is None.
        method : str, optional
            algorithm for exponentiating an MPO. The default is 'taylor'.
        order : TYPE, optional
            DESCRIPTION. The default is 4.
        scale : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return expmpo(self.H, constant, D=D, method='taylor', order=4, scale=0)


def gwp_mps(coord, nstates=None, inistates=0, a=None, x0=None, p0=0., dx=None, **kwargs):
    """
    Generate a separable Gaussian wave packet (GWP) in matrix product state (MPS) form.

    This routine builds a product MPS where each physical dimension is represented by a rank-3 tensor of shape ``[1, d, 1]``.

    The first tensor can optionally encode a discrete internal state basis of size ``nstates``. The spatial part is a direct product of 1D Gaussians, one per coordinate dimension, with optional momentum phase factors.

    MPS index order: ``[chi1, d, chi2] = [left_bond, physical, right_bond]``.

    Parameters
    ----------
    coord : list or array-like
        Sequence of 1D coordinate arrays. ``coord[i]`` provides the grid for the
        ``i``-th spatial dimension.
    nstates : int, optional
        Number of internal (discrete) states. If provided, a leading state tensor of
        shape ``[1, 1, nstates]`` is prepended to the MPS.
    inistates : int, optional
        Index of the initial internal state set to 1. Default is 0.
    a : array-like, optional
        Diagonal width matrix for the Gaussian. Only the diagonal entries ``a[i,i]``
        are used. If omitted, the identity matrix is used.
    x0 : float or array-like, optional
        Initial position(s). If scalar, the same value is used for all dimensions.
        If array-like, it is broadcast or truncated to match the number of dimensions.
    p0 : float, optional
        Initial momentum (same for all dimensions). Default is 0.
    dx : array-like, optional
        Grid spacings per dimension used for normalization. If omitted, all ones are used.
    **kwargs : dict
        Extra keyword arguments (currently unused; kept for API compatibility).

    Returns
    -------
    mps : list of numpy.ndarray
        List of MPS core tensors (complex dtype). Each tensor is rank-3 with
        shape ``[1, d, 1]``.

    Notes
    -----
    The 1D Gaussian for dimension ``i`` is

    $$
    \psi_i(x) = \left(\frac{a_i}{\pi}\right)^{1/4}
    \exp\left[-\frac{a_i}{2}(x-x_{0,i})^2\right]
    \exp\left[i p_0 (x-x_{0,i})\right],
    $$

    where ``a_i = a[i,i]``. The total wave packet is the product over dimensions.

    Examples
    --------
    Build a 2D Gaussian packet on two grids (no internal state)::

        x = np.linspace(-5.0, 5.0, 101)
        y = np.linspace(-3.0, 3.0, 61)
        mps = gwp_mps([x, y], a=np.diag([1.0, 2.0]), x0=[0.5, -0.2], p0=1.5)

    Include a 3-level internal state with initial state index 1::
        x = np.linspace(-4.0, 4.0, 81)
        mps = gwp_mps([x], nstates=3, inistates=1, a=np.diag([0.8]))
    return mps:
        site0: shape (1, 1, 3)  # internal state, only have vaule at index (:,:,1)
        site1: shape (1, 1, 81) # GWP
        Notes
    -----
    - The first tensor in the MPS represents the quantum state if `nstates` is provided.
    - The Gaussian wave packet is computed as:
      `(a / π)^(1/4) * exp(-a * (x - x0)^2 / 2) * exp(1j * p0 * (x - x0))`
      where `a` is the width parameter, `x` is the coordinate, `x0` is the initial position,
      and `p0` is the momentum.
    """
    ndim = len(coord)
    mps = []

    if nstates is not None:
        s = np.zeros((1, nstates, 1), dtype=complex)
        s[0, inistates, 0] = 1.0
        mps.append(s)

    if a is None:
        a = np.eye(ndim)
    if dx is None:
        dx = np.ones(ndim)
    if x0 is None:
        x0 = [0] * ndim
    else:
        x0 = list(x0)
        if len(x0) < ndim:
            x0 += [0] * (ndim - len(x0))
        else:
            x0 = x0[:ndim]

    for i in range(ndim):
        # GWP tensor: [chi1, d, chi2] = [1, len(coord[i]), 1]
        gwp = np.zeros((1, len(coord[i]), 1), dtype=complex)
        x = coord[i]
        ai = a[i, i]
        psi = (ai / np.pi) ** (1 / 4) * np.exp(-ai * (x - x0[i]) ** 2 / 2.) * np.exp(
            1j * p0 * (x - x0[i])) * np.sqrt(dx[i])
        gwp[0, :, 0] = psi
        mps.append(gwp)
    return mps
    # return MPS(mps, labels=['lv', 'p', 'rv']) # previously not returning MPS object (though we could let it be and actually is better), since currently shuoyi is not using this function, avoiding crashing in other places, so keeping the it unchanged

def show(tt_in):
    """
    Check and display mode sizes and TT-ranks of a TT/MPS/MPO tensor.

    MPS index order: ``[chi1, chi2, d]``
    MPO index order: ``[chi1, chi2, d_up, d_down]``

    Parameters
    ----------
    tt_in : MPS, MPO, or list of numpy.ndarray
        Input tensor network or list of cores. Each core must have shape
        ``[r_{k}, r_{k+1}, d]`` (MPS) or ``[r_{k}, r_{k+1}, d_up, d_down]`` (MPO).


    Examples
    --------
    Display summary for a list of cores (raw TT)::
        cores = [np.random.rand(1, 4, 2, 2), np.random.rand(4, 6,4,4), np.random.rand(6, 1, 8, 8)]
        show(cores)
            TT-tensor     3D : |2| |4|  |8|
            Type  = MPO :        \4/ \6/
    """
    if not isinstance(tt_in, MPS) and not isinstance(tt_in, MPO):
        tt = tt_in
    else:
        tt = tt_in.factors

    d = len(tt)
    n = []
    r = [1]

    for G in tt:
        if len(G.shape) not in (3, 4):
            raise ValueError('Invalid core for TT-tensor')

        if G.shape[0] != r[-1]:
            raise ValueError('Invalid shape of core for TT-tensor')

        if len(G.shape) == 4:
            label = 'MPO'
            n.append(G.shape[2])
        elif len(G.shape) == 3:
            label = 'MPS'
            n.append(G.shape[2])

        r.append(G.shape[1])

    if r[-1] != 1:
        raise ValueError('Invalid shape of core for TT-tensor')

    text1 = f'{label} with {d:-5d}D : '
    text2 = ' '

    for k in range(d):
        text1 += ' ' * max(0, len(text2) - len(text1) - 1)
        text1 += f'|{n[k]}|'

        if k < d - 1:
            text2 += ' ' * (len(text1) - len(text2) - 1)
            text2 += f'\\{r[k + 1]}/'

    print(text1 + '\n' + text2)


def expmpo(H, constant=1.0, D=None, method='taylor', order=4, scale=0):
    """

    Calculate the exponential of an MPO 
    
    .. math::
        U = e^{constant * H } 
        
    MPO index order: [chi1, chi2, d_up, d_down]

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    constant : TYPE, optional
        DESCRIPTION. The default is 1.0.
    D : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is 'taylor'.
    order : TYPE, optional
        DESCRIPTION. The default is 4.
    scale : TYPE, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """

    if method.lower() != 'taylor':
        raise ValueError(f"Method '{method}' not implemented. Only 'taylor' is supported.")

    scaled_constant = constant / (2 ** scale)

    constant_dtype = np.array(scaled_constant).dtype
    mpo_dtype = H.factors[0].dtype
    result_dtype = np.result_type(constant_dtype, mpo_dtype)

    # Create identity MPO with correct index order [chi1, chi2, d_up, d_down]
    identity_factors = []
    for i in range(H.L):
        d = H.dims[i]
        # Identity: [1, 1, d, d] with delta_{ij}
        W = np.zeros((1, 1, d, d), dtype=result_dtype)
        for j in range(d):
            W[0, 0, j, j] = 1.0
        identity_factors.append(W)

    result = MPO(identity_factors)
    term = MPO(identity_factors)

    factorial = 1
    for k in range(1, order + 1):
        term = term.matmul(H, chi_max=D)
        factorial = factorial * k
        coefficient = (scaled_constant ** k) / factorial
        result = result + (term * coefficient)

    for _ in range(scale):
        result = result.matmul(result, chi_max=D)

    return result

def apply_mpo(w_list, B_list, chi_max=None):
    """
    Apply the MPO to an MPS.

    MPS index order: [chi1, chi2, d] = [left_bond, right_bond, physical]
    MPO index order: [chi1, chi2, d_up, d_down] = [left_bond, right_bond, out, in]

    Parameters
    ----------
    w_list : list
        MPO tensors, each with shape [chi1, chi2, d_up, d_down].
    B_list : list
        MPS tensors, each with shape [chi1, chi2, d].
    chi_max : int
        Maximum bond dimension for compression. Default is None.

    Returns
    -------
    tuple
        (compressed_mps, truncation_error)

    Note
    ----
    This function does NOT modify the input B_list.
    """
    if isinstance(w_list, MPO):
        w_list_copy = w_list.factors
    else:
        w_list_copy = w_list
    if isinstance(B_list, MPS):
        B_list_copy = B_list.factors
    else:
        B_list_copy = B_list

    L = len(w_list_copy)
    if L > len(B_list_copy):
        raise ValueError("MPO must have the same or shorter length than MPS.")

    result = [B.copy() for B in B_list_copy]

    for i_site in range(L):
        # B: [chi_L, chi_R, d]
        # W: [b_L, b_R, d_out, d_in]
        chi_L, chi_R, d = result[i_site].shape
        b_L, b_R, d_out, d_in = w_list_copy[i_site].shape
        # Contract W[..., d_in] with B[..., d]
        # W: abij (a=b_L, b=b_R, i=d_out, j=d_in)
        # B: klj (k=chi_L, l=chi_R, j=d)
        # einsum: 'abij,klj->akbli' then reshape to [chi_L*b_L, chi_R*b_R, d_out]
        B_new = np.einsum('abij,klj->akbli', w_list_copy[i_site], result[i_site])
        B_new = np.reshape(B_new, (chi_L * b_L, chi_R * b_R, d_out))
        result[i_site] = B_new

    if chi_max is None:
        return result
    else:
        return compress(result, chi_max)



def product_W(W, X):
    """
    'Vertical' product of MPO W-matrices.

    MPO index order: [chi1, chi2, d_up, d_down]

    Diagram:
           |d_up (from W)
          -W-
           | (W's d_down contracts with X's d_up)
          -X-
           |d_down (from X)

    W acts first (on ket), X acts second.
    Result: [chi1_W * chi1_X, chi2_W * chi2_X, d_up_W, d_down_X]
    """
    # W: [a, b, s, t] = [chi1, chi2, d_up, d_down]
    # X: [c, d, t, u] = [chi1, chi2, d_up, d_down]
    # Contract W's d_down (t) with X's d_up (t)
    # Result indices: a, b, s (from W), c, d, u (from X)
    # Final shape: [a*c, b*d, s, u]
    return np.reshape(
        np.einsum("abst,cdtu->acbdsu", W, X),
        [W.shape[0] * X.shape[0],   # chi1
         W.shape[1] * X.shape[1],   # chi2
         W.shape[2],                 # d_up (from W)
         X.shape[3]]                 # d_down (from X)
    )


def product_MPO(M1, M2):
    """
    Vertical product of two MPOs: M1 @ M2.

    M1 acts first (closer to ket), M2 acts second (closer to bra).

    Note: This function does NOT modify M1 or M2.
    """
    if isinstance(M1, MPO):
        M1_copy = M1.factors
    else:
        M1_copy = M1
    if isinstance(M2, MPO):
        M2_copy = M2.factors
    else:
        M2_copy = M2

    L=min(len(M1_copy), len(M2_copy))

    Result = []
    for i in range(L):
        Result.append(product_W(M1_copy[i], M2_copy[i]))
    if len(M1_copy) > L:
        for i in range(L, len(M1_copy)):
            Result.append(M1_copy[i])
    if len(M2_copy) > L:
        for i in range(L, len(M2_copy)):
            Result.append(M2_copy[i])
    return Result




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

# def expect(mpo, mps):
#     # <GS| O |GS> , closing the zipper from the left
#     Taux = np.ones((1,1,1))
#     for l in range(N):
#         Taux = ZipperLeft(Taux, mps[l].conj().T, mpo[l], mps[l])
#     print('<GS| H |GS> = ', Taux[0,0,0])
#     # print('analytical result = ', -2*(N-1)/3)
#     return Taux[0, 0, 0]


def ZipperRight(Tr,Mb,O,Mt):
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
    Taux = np.einsum('ijk,klm',Mt,Tr, optimize=True)
    Taux = np.einsum('ijkl,mnkj',Taux,O, optimize=True)
    Tf = np.einsum('ijkl,jlm',Taux,Mb, optimize=True)

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

def initial_E(W):
    """
    Construct the initial Left Environment (E) tensor for the vacuum state.

    This represents the contraction of all sites to the left of the chain
    (effectively scalar 1 for vacuum).

    Index Convention:
    -----------------
    [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        The MPO tensor at the first site (index 0).
        Used to determine the MPO bond dimension (chi_MPO) and symmetry sector.

    Returns
    -------
    E : np.ndarray or BlockTensor
        The left environment tensor.
        - Dense Shape: (W.shape[0], 1, 1)
        - U(1) keys: (0, 0, 0) -> 1.0 (Scalar identity block)
    """
    if SYMMETRY_AVAILABLE and isinstance(W, BlockTensor):
        sample_qn = W.qns[0][0] if len(W.qns[0]) > 0 else 0
        if isinstance(sample_qn, tuple):
            zero_qn = sample_qn * 0 # e.g. QN(0,0)
        else:
            zero_qn = 0

        # MPO (In), Bra (In), Ket (In) -> Need Out (+1)
        # Key format: (MPO_Bond, Bra_Bond, Ket_Bond)
        data = {(zero_qn, zero_qn, zero_qn): np.ones((1, 1, 1))}
        qns = [[zero_qn], [zero_qn], [zero_qn]]
        dirs = [1, -1, 1]
        return BlockTensor(data, qns, dirs)

    # Dense branch without U(1)
    # MPO Left Bond dimension is W.shape[0]
    E = np.zeros((W.shape[0], 1, 1))
    E[0] = 1 # vacuum state
    return E

def initial_F(W, target_qn=0):
    """
    Constructs the initial Right Environment (Vacuum).

    represents the contraction of all sites to the right of the chain.
    For U(1) symmetry, this enforces the total target charge of the system
    (Bra and Ket must end at `target_qn`).

    Index Convention:
    -----------------
    [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        The MPO tensor at the last site (index -1).
        Used to determine the MPO bond dimension.
    target_qn : int, optional
        The target quantum number (total charge) of the wavefunction.
        Required for BlockTensor to ensure the bra/ket bonds match the
        target sector at the boundary. Default is 0.

    Returns
    -------
    F : np.ndarray or BlockTensor
        The right environment tensor.
        - Dense Shape: (W.shape[1], 1, 1)
        - U(1) keys: (0, target_qn, target_qn) -> 1.0
    """
    if SYMMETRY_AVAILABLE and isinstance(W, BlockTensor):
        sample_qn = W.qns[1][0] if len(W.qns[1]) > 0 else 0

        if isinstance(sample_qn, tuple):
            zero_qn = sample_qn * 0
        else:
            zero_qn = 0

        # MPO (In), Bra (Out), Ket (In) -> Need [In, Out, In] = [-1, 1, -1]
        # Key format: (MPO_Bond, Bra_Bond, Ket_Bond)
        data = {(zero_qn, target_qn, target_qn): np.ones((1, 1, 1))}
        qns = [[zero_qn], [target_qn], [target_qn]]
        dirs = [-1, 1, -1]
        return BlockTensor(data, qns, dirs)

    # Dense branch without U(1)
    # MPO Right Bond dimension is W.shape[1]
    F = np.zeros((W.shape[1], 1, 1))
    F[-1] = 1
    return F


def dense_to_symmetric(mps_list, phys_qns=None, tol=1e-12):
    """
    Convert a *product-state* dense MPS guess into a true U(1) BlockTensor MPS.
    TODO: this now currently only supports particle number symmetry, add Sz, Lz etc. and also make each symmetry optional.

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

    Index Convention (Dense):
    -------------------------
    Input A, B: [Left, Phys, Right]
    Input W:    [Left, Right, Out, In]
    Input F:    [MPO_Bond, Bra_Bond, Ket_Bond]
    Output F':  [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        MPO tensor at this site.
    A : np.ndarray or MPS or BlockTensor
        Ket MPS tensor.
    F : np.ndarray or BlockTensor
        Right environment tensor.
    B : np.ndarray or MPS or BlockTensor
        Bra MPS tensor.

    Returns
    -------
    F_new : np.ndarray or BlockTensor
        The updated right environment.
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

    #  Dense Branch ---
    if isinstance(A, MPS):
        A_std = A.factors[0].transpose(A.lv_idx, A.p_idx, A.rv_idx)
    elif isinstance(A, np.ndarray) and A.ndim == 3:
        A_std = A
    else:
        raise ValueError(f"Unknown type/shape for A: {type(A)}")

    if isinstance(B, MPS):
        B_std = B.factors[0].transpose(B.lv_idx, B.p_idx, B.rv_idx)
    else:
        B_std = B

    # Contraction
    # F: (MPO, Bra, Ket)
    # A_std: (Left, Phys, Right)

    # Step A: Contract F with A* (Bra)
    # F[Bra] (1) -- A*[Right] (2)
    # T1: (MPO_R, Ket_R, Left_Bra, Phys_Bra)
    T1 = np.tensordot(F, A_std.conj(), axes=(1, 2))

    # Step B: Contract T1 with W
    # T1[MPO_R] (0) -- W[Right] (1)
    # T1[Phys_Bra] (3) -- W[Out] (2)
    # T2: (Ket_R, Left_Bra, Left_MPO, Phys_In)
    T2 = np.tensordot(T1, W, axes=([0, 3], [1, 2]))

    # Step C: Contract T2 with B
    # T2[Ket_R] (0) -- B[Right] (2)
    # T2[Phys_In] (3) -- B[Phys] (1)
    # Result: (Left_Bra, Left_MPO, Left_Ket)
    F_new = np.tensordot(T2, B_std, axes=([0, 3], [2, 1]))

    # 3. Reorder to (MPO, Bra, Ket) -> (1, 0, 2)
    return F_new.transpose(1, 0, 2)

def contract_from_left(W, A, E, B):
    """
    ## tensor contraction from the left hand side
    ## +-    +--A-
    ## |     |  |
    ## E' =  E--F-
    ## |     |  |
    ## +-    +--B-

    Index Convention (Dense):
    -------------------------
    Input A, B: [Left, Phys, Right] (Standard MPS Site)
    Input W:    [Left, Right, Out, In] (Standard MPO)
    Input E:    [MPO_Bond, Bra_Bond, Ket_Bond] (Standard Env)
    Output E':  [MPO_Bond, Bra_Bond, Ket_Bond]

    Parameters
    ----------
    W : np.ndarray or BlockTensor
        MPO tensor at this site.
    A : np.ndarray or MPS or BlockTensor
        Ket MPS tensor (top/bra in diagram) at this site.
    E : np.ndarray or BlockTensor
        Left environment tensor.
    B : np.ndarray or MPS or BlockTensor
        Bra MPS tensor (bottom/ket in diagram) at this site.

    Returns
    -------
    E_new : np.ndarray or BlockTensor
        The updated left environment.
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

    #  Dense Branch ---
    # 1. Standardize Inputs to [Left, Phys, Right]
    if isinstance(A, MPS):
        A_std = A.factors[0].transpose(A.lv_idx, A.p_idx, A.rv_idx)
    elif isinstance(A, np.ndarray) and A.ndim == 3:
        A_std = A
    else:
        raise ValueError(f"Unknown type/shape for A: {type(A)}")

    if isinstance(B, MPS):
        B_std = B.factors[0].transpose(B.lv_idx, B.p_idx, B.rv_idx)
    elif isinstance(B, np.ndarray) and B.ndim == 3:
        B_std = B
    else:
        raise ValueError(f"Unknown type/shape for B: {type(B)}")

    # 2. Perform Contraction
    # E: (a, i, k) -> (MPO_Left, Bra_Left, Ket_Left)
    # A_std: (i, s, j) -> (Bra_Left, Phys, Bra_Right)
    # W: (a, b, s, t) -> (MPO_L, MPO_R, Phys_Out, Phys_In)
    # B_std: (k, t, l) -> (Ket_Left, Phys, Ket_Right)

    # Step A: Contract E with A* (Bra)
    # E[1] (Bra_L) -- A*[0] (Bra_L)
    # T1 shape: (MPO_L, Ket_L, Phys_Bra, Bra_R)
    T1 = np.tensordot(E, A_std.conj(), axes=(1, 0))

    # Step B: Contract T1 with W (MPO)
    # T1[0] (MPO_L) -- W[0] (MPO_L)
    # T1[2] (Phys_Bra) -- W[2] (Phys_Out)
    # T2 shape: (Ket_L, Bra_R, MPO_R, Phys_In)
    T2 = np.tensordot(T1, W, axes=([0, 2], [0, 2]))

    # Step C: Contract T2 with B (Ket)
    # T2[0] (Ket_L) -- B[0] (Ket_L)
    # T2[3] (Phys_In) -- B[1] (Phys_In)
    # Result shape: (Bra_R, MPO_R, Ket_R)
    E_new = np.tensordot(T2, B_std, axes=([0, 3], [0, 1]))

    # 3. Reorder to Standard Environment (MPO, Bra, Ket)
    # Current: (Bra_R, MPO_R, Ket_R) -> (1, 0, 2)
    return E_new.transpose(1, 0, 2)



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
    # if SYMMETRY_AVAILABLE and isinstance(Blist[-1], BlockTensor):
    #     if target_qn is None:
    #         # pick the unique right-bond qR from the last site tensor
    #         # key = (qL, qR, qP) for site tensors in this code
    #         qs = sorted({key[1] for key in Blist[-1].data.keys()})
    #         if len(qs) != 1:
    #             raise ValueError(f"Ambiguous total charge on last bond: {qs}. Pass target_qn explicitly.")
    #         target_qn = qs[0]

    F = [initial_F(MPO[-1], target_qn=target_qn if target_qn is not None else 0)]
    for i in range(len(MPO)-1, 0, -1):
        F.append(contract_from_right(MPO[i], Alist[i], F[-1], Blist[i]))
    return F

def construct_E(Alist, MPO, Blist):
    return [initial_E(MPO[0])]


def coarse_grain_MPO(W, X):
    """
    # 2-to-1 coarse-graining of two site MPO into one site
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




def coarse_grain_MPS(A,B):
    """
    # 2-1 coarse-graining of two-site MPS into one site
    #  |   |      |  |
      -theta- <= -A--B-

    Parameters
    ----------
    Input A, B: [Left, Phys, Right]

    Returns
    -------
    Output: [Left_A, Phys_A, Phys_B, Right_B]

    """
    # A: (L_a, P_a, R_a)
    # B: (L_b, P_b, R_b)  where R_a == L_b
    # Contract A[2] with B[0]
    return np.reshape(np.tensordot(A, B, axes=(2, 0)),[A.shape[1]*B.shape[1], A.shape[0], B.shape[2]]) # Result: (L_a, P_a, P_b, R_b)

def fine_grain_MPS(Theta, dims):
    """
    Split a two-site tensor back into two MPS tensors via SVD.
    Input Theta: [Left, Phys_A, Phys_B, Right]
    #  |   |      |  |
      -theta- => -A--B-
    """
    # Theta shape: (Chi_L, d_A, d_B, Chi_R)

    # 1. Group indices for SVD: (Chi_L * d_A) x (d_B * Chi_R)
    # This corresponds to "Left Canonical" splitting
    chi_L = Theta.shape[0]
    chi_R = Theta.shape[3]
    d_A = dims[0]
    d_B = dims[1]

    # Reshape to Matrix
    Psi = Theta.reshape(chi_L * d_A, d_B * chi_R)

    # SVD
    U, S, V = np.linalg.svd(Psi, full_matrices=False)

    # Reshape U -> A [Left, Phys, Right_Bond]
    # U columns are the new bond index
    A = U.reshape(chi_L, d_A, -1)

    # Reshape V -> B [Left_Bond, Phys, Right]
    # V rows are the new bond index
    B = V.reshape(-1, d_B, chi_R)

    return A, S, B

def truncate_SVD(U, S, V, m):
    """
    # truncate the matrices from an SVD to at most m states
    U shape: (Left, Phys, Right)
    V shape: (Left, Phys, Right)
    """
    m = min(len(S), m)
    trunc = np.sum(S[m:])
    S = S[0:m]
    # U has the bond on the last axis: (Left, Phys, Bond)
    U = U[:, :, 0:m]
    # V has the bond on the first axis: (Bond, Phys, Right)
    V = V[0:m, :, :]
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
        self.W = W # MPO: (Left, Right, Out, In) -> (L, R, P_bra, P_ket)
        self.F = F
        self.dtype = np.dtype('d')

        # Determine shapes
        # E: (MPO, Bra_L, Ket_L)
        # F: (MPO, Bra_R, Ket_R)
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)

        # Required Input Vector Shape (Two-Site Tensor):
        # We expect input vector 'v' to flatten a tensor of shape:
        # (Ket_L, Phys_A, Phys_B, Ket_R)
        # Note: If it's 1-site, it's (Ket_L, Phys, Ket_R)

        self.chi_L = E.shape[2] # Ket_L
        self.chi_R = F.shape[2] # Ket_R

        # W has combined physical dimensions if coarse-grained
        self.d_out = W.shape[2]
        self.d_in = W.shape[3]

        self.req_shape = (self.chi_L, self.d_in, self.chi_R)
        self.size = self.chi_L * self.d_in * self.chi_R
        self.shape = (self.size, self.size)

    def _matvec(self, v):
        # 1. Reshape vector to tensor A [Left, Phys, Right]
        A = v.reshape(self.req_shape)

        # 2. Contract: E(a,i,k) * A(k,s,l) * W(a,b,r,s) * F(b,j,l)
        # E: (MPO_L, Bra_L, Ket_L)
        # A: (Ket_L, Phys_In, Ket_R)
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)
        # F: (MPO_R, Bra_R, Ket_R)

        # Step A: E * A -> Contract Ket_L
        # E[2] with A[0]
        # T1: (MPO_L, Bra_L, Phys_In, Ket_R)
        T1 = np.tensordot(self.E, A, axes=(2, 0))

        # Step B: T1 * W -> Contract MPO_L and Phys_In
        # T1[0] (MPO_L) with W[0] (MPO_L)
        # T1[2] (Phys_In) with W[3] (Phys_In)
        # T2: (Bra_L, Ket_R, MPO_R, Phys_Out)
        T2 = np.tensordot(T1, self.W, axes=([0, 2], [0, 3]))

        # Step C: T2 * F -> Contract MPO_R and Ket_R
        # T2[2] (MPO_R) with F[0] (MPO_R)
        # T2[1] (Ket_R) with F[2] (Ket_R)
        # Result: (Bra_L, Phys_Out, Bra_R)
        R = np.tensordot(T2, self.F, axes=([2, 1], [0, 2]))

        return np.reshape(R, -1)

## optimize a single site given the MPO matrix W, and tensors E,F
def optimize_site(A, W, E, F, tol=1E-8):
    H = HamiltonianMultiply(E,W,F)
    # we choose tol=1E-8 here, which is OK for small calculations.
    # to bemore robust, we should take the tol -> 0 towards the end
    # of the calculation.
    E, V = sparse.linalg.eigsh(H,1,v0=A,which='SA', tol=tol)
    return (E[0],np.reshape(V[:,0], H.req_shape))

def inject_noise_symmetric(AA, sym_mgr, noise_val=1e-4):
    """
    Injects noise into ALL valid symmetry sectors.
    Args:
        AA: The BlockTensor to perturb
        sym_mgr: SymmetryManager instance to get valid physical QNs
        noise_val: Magnitude of noise
    """
    if not hasattr(AA, 'data'):
        return AA
    valid_qL = {}
    valid_qR = {}

    first_blk = next(iter(AA.data.values()))
    is_complex = np.iscomplexobj(first_blk)
    dtype = first_blk.dtype

    for (qL, qR, qP1, qP2), blk in AA.data.items():
        valid_qL[qL] = blk.shape[0]
        valid_qR[qR] = blk.shape[1]

    # get valid physical QNs
    # generate the QNs for standard spin-orbital states: Emp, Occ(Up), Occ(Dn)
    # The manager knows if 'Occ' means QN(1) or QN(1,1) or QN(1,1,-1).
    possible_phys_qns = set()
    # Always include Vacuum
    possible_phys_qns.add(sym_mgr.get_phys_qn(0, 'emp'))
    # Include Occupied states
    # We check both "Even-like" (Up) and "Odd-like" (Down) indices
    # to cover all bases for spin-orbitals.
    possible_phys_qns.add(sym_mgr.get_phys_qn(0, 'occ')) # "Up"
    possible_phys_qns.add(sym_mgr.get_phys_qn(1, 'occ')) # "Down"
    possible_phys_qns = list(possible_phys_qns)
    # Iterate and Inject (Same logic as before)
    for qL, dL in valid_qL.items():
        for qP1 in possible_phys_qns:
            for qP2 in possible_phys_qns:
                # Calculate required Right Sector
                target_qR = qL + qP1 + qP2
                if target_qR in valid_qR:
                    dR = valid_qR[target_qR]
                    key = (qL, target_qR, qP1, qP2)
                    # Dimensions for spin-orbitals are 1
                    dP1, dP2 = 1, 1
                    if key not in AA.data:
                        noise = (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        if is_complex:
                            noise = noise + 1j * (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        AA.data[key] = noise.astype(dtype)
    return AA


def optimize_two_sites(A, B, W1, W2, E, F, m, dir, U1=False, sym_mgr=None):
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
        # U(1) on branch is still using (L,R,P) MPS index convention. TODO:also fix to LPR if have time.
        # A: (Bond_L, Bond_M, Phys_L)
        # B: (Bond_M, Bond_R, Phys_R)
        # AA = A * B -> (Bond_L, Phys_L, Bond_R, Phys_R)
        if A.rank == 3:
            AA = tensordot(A, B, axes=([1], [0])) # this will return as BlockTensor Object
            AA = AA.transpose(0, 2, 1, 3) # standard MPO bond index currently is (L,R,out,in). TODO: reshape to (L, out, in, R) (note that currently dense branch is (L, R, out, in), better also fix that one)

            # # add noise to AA. TODO: maby make noise decay as round goes? that should be done by injecting a smaller noise_val as dmrg sweep goes. fix and test for optimal value if have time
            AA = inject_noise_symmetric(AA, noise_val=1e-4, sym_mgr=sym_mgr)
        else:
            raise ValueError(f"Unexpected tensor rank {A.rank} in symmetric opt")
        H_op = HamiltonianMultiplyU1(E, [W1, W2], F)
        norm = AA.norm()
        AA = AA * (1.0/norm)
        energy, AA_new = solve_davidson(H_op, AA, tol=1e-5)
        # AA_new is (L, R, Phys_L, Phys_R)
        # We need to return A(L, M, P_L) and B(M, R, P_R)
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
    else: # Dense branch ( MPS index standardized to Left, Phys, Right)
        W = coarse_grain_MPO(W1,W2)
        # Returns (Left, Phys_A, Phys_B, Right)
        AA = coarse_grain_MPS(A,B)
        # Optimize
        H = HamiltonianMultiply(E,W,F)
        E, V_flat = sparse.linalg.eigsh(H,1,v0=AA,which='SA')
        # Fine Grain (SVD Split)
        # V_flat is (Left * Phys_A * Phys_B * Right)
        # We need V_tensor: (Left, Phys_A, Phys_B, Right)
        AA = V_flat.reshape(A.shape[0], A.shape[1], B.shape[1], B.shape[2])
        A,S,B = fine_grain_MPS(AA, [A.shape[1], B.shape[1]])
        A,S,B,trunc,m = truncate_SVD(A,S,B,m)
        if (dir == 'right'):
            # B = S * B.  S is (m,), B is (m, d, R).
            # Contract S with B[0] (Left bond of B)
            B = np.tensordot(np.diag(S), B, axes=(1, 0))
        else:
            assert dir == 'left'
            # A = A * S.  A is (L, d, m), S is (m,)
            # Contract A[2] (Right bond) with S
            A = np.tensordot(A, np.diag(S), axes=(2, 0))
        return E[0], A, B, trunc, m

def two_site_dmrg(mps, mpo, m, sweeps=50, conv=1e-6, U1=False, target_qn=None,\
                  not_conv_err=True, sym_mgr=None):
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
    MPS = mps 
    MPO = mpo 
    
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
                MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1=U1, sym_mgr=sym_mgr
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
                MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1=U1, sym_mgr=sym_mgr
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
    else:
        if converged == False:
            print("DMRG did not converge within {sweeps} sweeps, returning the last result.")
    if gauge == None:
        gauge = "Right"

    return Energy, MPS, gauge, converged


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
    if ket is None:
        ket = bra
        
    AList = bra
    BList = ket



    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]






# class TDVP(DMRG):
#     pass



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
    InitialA1 = np.zeros((1, d, 1))
    InitialA1[0, 0, 0] = 1  # Up state
    InitialA2 = np.zeros((1, d, 1))
    InitialA2[0, 1, 0] = 1  # Down state

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
    H = [Wfirst] + ([W] * (N-2)) + [Wlast]

    dmrg = DMRG(H, D=10, nsweeps=8)
    dmrg.init_guess = initial_mps
    dmrg.init_guess = MPS(initial_mps, labels=['lv', 'p', 'rv'])
    dmrg.run()
    # print(dmrg.ground_state.calc_1site_rdm())





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