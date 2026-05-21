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
from collections import defaultdict
# from pyqed.mps.mps import LeftCanonical, RightCanonical, ZipperLeft, ZipperRight
from pyqed.mps.decompose import decompose, compress

import logging
logger = logging.getLogger(__name__)
try:
    from pyqed.mps.symmetry import (
        BlockTensor,
        tensordot,
        solve_davidson,
        solve_davidson_block,
        QN,
        Sector,
        SymmetryManager,
        is_sector_like,
        zero_like_sector,
    )
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
from scipy.linalg import expm, block_diag
import warnings
from tensorly.decomposition import tensor_train_matrix

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




def _make_complementary_boundary_stack(families, nsites):
    if families is None:
        return None, {}
    try:
        from pyqed.mps.nonabelian.renormalized import (
            ComplementaryFamilyRenormalizedOperatorBlock,
            ComplementaryFamilyRenormalizedOperatorTable,
            ComplementaryRenormalizedOperatorStack,
        )
    except Exception:
        return None, {}

    stack = ComplementaryRenormalizedOperatorStack(families=families)
    payloads = {}

    def _attach_family_table(entry):
        blocks = {}
        for channel, name in enumerate(entry.family_names):
            payload = entry.family_payloads.get(str(name))
            n_terms = 0 if payload is None else int(payload.n_terms)
            cross_terms = 0 if payload is None else int(payload.cross_terms)
            channels = (int(channel),) if cross_terms > 0 else ()
            payload_keys = () if payload is None else tuple(key for key, _value in payload.entries)
            blocks[str(name)] = ComplementaryFamilyRenormalizedOperatorBlock(
                family_name=str(name),
                channels=channels,
                symbolic_terms=int(cross_terms),
                payload_keys=payload_keys,
                stored_elements=int(n_terms),
                payload_norm=0.0 if payload is None else float(payload.coefficient_norm),
                coefficient_terms=int(n_terms),
                coefficient_cross_terms=int(cross_terms),
            )
        object.__setattr__(
            entry,
            "family_operator_table",
            ComplementaryFamilyRenormalizedOperatorTable(
                side=entry.side,
                bond=entry.bond,
                family_blocks=blocks,
                source="abelian_complementary_payload_table",
            ),
        )
        return entry

    parent = None
    for bond in range(max(0, int(nsites) - 1)):
        entry = _attach_family_table(stack.put(
            "left",
            bond,
            signature=("abelian_complementary_boundary", "left", int(bond)),
            source="abelian_sweep_boundary",
            parent_key=parent,
        ))
        payloads[("left", int(bond))] = entry
        parent = entry.key

    parent = None
    for bond in range(max(0, int(nsites) - 1), 0, -1):
        entry = _attach_family_table(stack.put(
            "right",
            bond,
            signature=("abelian_complementary_boundary", "right", int(bond)),
            source="abelian_sweep_boundary",
            parent_key=parent,
        ))
        payloads[("right", int(bond))] = entry
        parent = entry.key

    return stack, payloads


class AbelianComplementaryBoundaryActionTable:
    """Dense callable boundary table for one Abelian two-site local layout."""

    def __init__(
        self,
        matrix,
        layout,
        qns,
        dirs,
        *,
        bond=None,
        source="exact_residual_table",
        boundary_family_tables=None,
        channel_matrices=None,
    ):
        self.matrix = np.asarray(matrix, dtype=complex)
        self.layout = tuple(layout)
        self.qns = [list(q) for q in qns]
        self.dirs = list(dirs)
        self.bond = None if bond is None else int(bond)
        self.source = str(source)
        self.boundary_family_tables = tuple(boundary_family_tables or ())
        self.channel_matrices = {
            str(name): np.asarray(value, dtype=complex)
            for name, value in (channel_matrices or {}).items()
        }

    @staticmethod
    def _size(layout):
        return int(sum(np.prod(shape, dtype=int) for _key, shape in layout))

    def flatten(self, A):
        chunks = []
        for key, shape in self.layout:
            blk = A.data.get(key)
            if blk is None:
                chunks.append(np.zeros(int(np.prod(shape, dtype=int)), dtype=complex))
            else:
                chunks.append(np.asarray(blk).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=complex)

    def unflatten(self, vec):
        data = {}
        pos = 0
        for key, shape in self.layout:
            n = int(np.prod(shape, dtype=int))
            data[key] = np.asarray(vec[pos:pos + n]).reshape(shape).copy()
            pos += n
        return BlockTensor(data, [q[:] for q in self.qns], self.dirs[:])

    def apply(self, A):
        return self.unflatten(self.matrix @ self.flatten(A))

    def apply_channels(self, A):
        vec = self.flatten(A)
        return {
            name: self.unflatten(matrix @ vec)
            for name, matrix in self.channel_matrices.items()
        }

    @property
    def dim(self):
        return int(self.matrix.shape[0])

    @property
    def nnz(self):
        return int(np.count_nonzero(np.abs(self.matrix) > 1.0e-14))

    @property
    def stats(self):
        active = []
        table_stats = []
        for table in self.boundary_family_tables:
            if table is None:
                continue
            active.extend(str(name) for name in table.active_family_names)
            table_stats.append({
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            })
        return {
            "kind": "abelian_complementary_boundary_action_table",
            "source": self.source,
            "bond": self.bond,
            "dim": int(self.dim),
            "nnz": int(self.nnz),
            "stored_elements": int(np.asarray(self.matrix).size),
            "layout_blocks": int(len(self.layout)),
            "norm": float(np.linalg.norm(self.matrix)),
            "active_family_names": tuple(sorted(set(active))),
            "boundary_family_tables": tuple(table_stats),
            "boundary_channel_tables": tuple(
                {
                    "name": str(name),
                    "dim": int(matrix.shape[0]),
                    "nnz": int(np.count_nonzero(np.abs(matrix) > 1.0e-14)),
                    "stored_elements": int(matrix.size),
                    "norm": float(np.linalg.norm(matrix)),
                }
                for name, matrix in self.channel_matrices.items()
            ),
        }


class HamiltonianMultiplyU1:
    """
    Symmetric version of HamiltonianMultiply using BlockTensor.
    """
    _GLOBAL_BOUNDARY_EINSUM_EXPR_CACHE = {}
    _GLOBAL_BOUNDARY_BATCHED_EINSUM_EXPR_CACHE = {}
    _GLOBAL_BOUNDARY_DIRECT_EINSUM_EXPR_CACHE = {}
    _GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE = {}
    _GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE = {}

    def __init__(
        self,
        E,
        W,
        F,
        complementary_operator_families=None,
        bond=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
    ):
        self.E = E
        self.W = W
        self.F = F
        self.complementary_operator_families = complementary_operator_families
        self.bond = None if bond is None else int(bond)
        self.complementary_boundary_payloads = complementary_boundary_payloads or {}
        self.complementary_split_stats = complementary_split_stats
        self.complementary_family_environments = complementary_family_environments or {}
        self.complementary_direct_family_environments = (
            complementary_direct_family_environments or {}
        )
        self._prefer_boundary_factorized = bool(
            getattr(
                complementary_operator_families,
                "prefer_complementary_payload_tensor_matvec",
                False,
            )
        )
        self._debug_boundary_channel_matrices = bool(
            getattr(
                complementary_operator_families,
                "debug_boundary_channel_matrices",
                False,
            )
        )
        self._debug_complementary_split_metadata = bool(
            getattr(
                complementary_operator_families,
                "debug_complementary_split_metadata",
                False,
            )
        )
        self._debug_complementary_action_check = bool(
            getattr(
                complementary_operator_families,
                "debug_complementary_action_check",
                False,
            )
        )
        self._debug_complementary_action_check_tol = float(
            getattr(
                complementary_operator_families,
                "debug_complementary_action_check_tol",
                1.0e-10,
            )
        )
        self._debug_complementary_action_check_limit = int(
            getattr(
                complementary_operator_families,
                "debug_complementary_action_check_limit",
                32,
            )
        )
        self._prefer_precontracted_family_environment = bool(
            getattr(
                complementary_operator_families,
                "prefer_precontracted_family_environment",
                False,
            )
        )
        self.dtype = np.float64
        self._dense_cache = {}
        self._action_cache = {}
        self._compiled_action_cap = 4
        self._boundary_factorized_cache = {}
        self._boundary_factorized_action_cap = 8192
        self._w12_cache = None
        self._local_complementary_cache = {}
        self._local_complementary_channel_cache = {}
        self._boundary_table_cache = {}
        self._boundary_table_max_dim = 32
        self._boundary_einsum_path_cache = {}
        self._boundary_einsum_expr_cache = self._GLOBAL_BOUNDARY_EINSUM_EXPR_CACHE
        self._boundary_batched_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_BATCHED_EINSUM_EXPR_CACHE
        )
        self._boundary_direct_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_DIRECT_EINSUM_EXPR_CACHE
        )

    @staticmethod
    def _component_action_token(E, W, F):
        return (
            HamiltonianMultiplyU1._tensor_token(E),
            HamiltonianMultiplyU1._tensor_token(W[0]),
            HamiltonianMultiplyU1._tensor_token(W[1]),
            HamiltonianMultiplyU1._tensor_token(F),
        )

    def _build_direct_operator_plan(self, A, E, W, F, label):
        layout = self._layout(A)
        cache_key = (
            self._component_action_token(E, W, F),
            layout,
            "direct_operator",
            str(label),
        )
        if cache_key in self._boundary_factorized_cache:
            return self._boundary_factorized_cache[cache_key]

        e_by_ket_l = self._block_index(E, (2,))
        w1_by_left_in = self._block_index(W[0], (0, 3))
        w2_by_left_in = self._block_index(W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(F, (0, 2))

        left_operator_cache = {}
        right_operator_cache = {}
        entries = []
        out_shapes = {}
        channels = set()
        dtype_args = []
        left_eq = "aij,abux->bijux"
        right_eq = "bcvy,clk->bvylk"
        action_eq = "bijux,jkxy,bvylk->iluv"

        def _left_operator(e_key, e_blk, w1_key, w1_blk):
            key = (id(e_blk), id(w1_blk))
            cached = left_operator_cache.get(key)
            if cached is not None:
                return cached
            op = np.einsum(left_eq, e_blk, w1_blk, optimize="greedy")
            left_operator_cache[key] = op
            return op

        def _right_operator(w2_key, w2_blk, f_key, f_blk):
            key = (id(w2_blk), id(f_blk))
            cached = right_operator_cache.get(key)
            if cached is not None:
                return cached
            op = np.einsum(right_eq, w2_blk, f_blk, optimize="greedy")
            right_operator_cache[key] = op
            return op

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._boundary_factorized_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._boundary_factorized_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._boundary_factorized_cache[cache_key] = None
                        return None
                    channel = w1_key[1]
                    left_op = _left_operator(e_key, e_blk, w1_key, w1_blk)
                    for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._boundary_factorized_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            right_op = _right_operator(w2_key, w2_blk, f_key, f_blk)
                            if left_op.shape[0] != right_op.shape[0]:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                left_op.shape[1],
                                right_op.shape[3],
                                left_op.shape[3],
                                right_op.shape[1],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            expr_key = (left_op.shape, a_blk.shape, right_op.shape)
                            expr = self._boundary_direct_einsum_expr_cache.get(expr_key)
                            if expr is None:
                                try:
                                    import opt_einsum as oe

                                    expr = oe.contract_expression(
                                        action_eq,
                                        left_op.shape,
                                        a_blk.shape,
                                        right_op.shape,
                                        optimize="greedy",
                                    )
                                except Exception:
                                    try:
                                        expr = np.einsum_path(
                                            action_eq,
                                            left_op,
                                            a_blk,
                                            right_op,
                                            optimize="greedy",
                                        )[0]
                                    except ValueError:
                                        self._boundary_factorized_cache[cache_key] = None
                                        return None
                                self._boundary_direct_einsum_expr_cache[expr_key] = expr
                            dtype_args.extend((left_op.dtype, right_op.dtype))
                            channels.add(channel)
                            entries.append((channel, a_key, out_key, left_op, right_op, expr))

        if not entries:
            plan = ((), (), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._boundary_factorized_cache[cache_key] = plan
            return plan
        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            tuple(entries),
            out_shapes,
            out_qns,
            A.dirs[:],
            dtype,
        )
        self._boundary_factorized_cache[cache_key] = plan
        return plan

    def _build_boundary_direct_operator_plan(self, A):
        return self._build_direct_operator_plan(
            A,
            self.E,
            self.W,
            self.F,
            "boundary_direct_operator",
        )

    def _apply_direct_operator_plan(self, A, plan):
        if plan is None:
            return None
        channels, entries, out_shapes, out_qns, out_dirs, dtype = plan
        action_eq = "bijux,jkxy,bvylk->iluv"
        total_data = {
            key: np.zeros(shape, dtype=dtype)
            for key, shape in out_shapes.items()
        }
        for _channel, a_key, out_key, left_op, right_op, expr in entries:
            if callable(expr):
                contribution = expr(left_op, A.data[a_key], right_op)
            else:
                contribution = np.einsum(
                    action_eq,
                    left_op,
                    A.data[a_key],
                    right_op,
                    optimize=expr,
                )
            total_data[out_key] += contribution
        return BlockTensor(total_data, out_qns, out_dirs), channels, entries

    def _matvec_boundary_direct_operator(self, A, *, local_channels=None):
        plan = self._build_boundary_direct_operator_plan(A)
        applied = self._apply_direct_operator_plan(A, plan)
        if applied is None:
            return None
        total, channels, entries = applied
        local_channels = local_channels or {}
        for tensor in local_channels.values():
            total = total - tensor
        return {
            "total": total,
            "channels": {},
            "stats": {
                "kind": "abelian_complementary_boundary_direct_operator_action",
                "source": "precontracted_left_right_middle_channel_operators",
                "bond": self.bond,
                "n_entries": int(len(entries)),
                "n_mpo_middle_channels": int(len(channels)),
                "n_channels": int(len(channels) + len(local_channels)),
                "channel_names": tuple(
                    f"mpo_middle:{repr(channel)}" for channel in channels
                )
                + tuple(f"subtract_local:{name}" for name in local_channels),
                "channels_materialized": False,
            },
        }

    def _record_complementary_split(
        self,
        mode,
        *,
        local=None,
        local_channels=None,
        boundary=None,
        boundary_table=None,
        boundary_operator=None,
    ):
        stats = self.complementary_split_stats
        if stats is None:
            return
        mode = str(mode)
        stats["calls"] = int(stats.get("calls", 0)) + 1
        modes = stats.setdefault("modes", {})
        modes[mode] = int(modes.get(mode, 0)) + 1
        table_stats = None if boundary_table is None else boundary_table.stats
        entry = {
            "bond": self.bond,
            "mode": mode,
            "local_norm": None if local is None else float(local.norm()),
            "local_channels": {
                str(name): float(value.norm())
                for name, value in (local_channels or {}).items()
            },
            "boundary_norm": None if boundary is None else float(boundary.norm()),
            "boundary_table": table_stats,
            "boundary_operator": boundary_operator,
            "metadata": (
                self._compact_complementary_split_metadata()
                if self._debug_complementary_split_metadata
                else None
            ),
        }
        stats["last"] = entry
        if self.bond is not None:
            bonds = stats.setdefault("bonds", {})
            bond_key = int(self.bond)
            bond_stats = bonds.setdefault(bond_key, {"calls": 0, "modes": {}})
            bond_stats["calls"] = int(bond_stats.get("calls", 0)) + 1
            bond_modes = bond_stats.setdefault("modes", {})
            bond_modes[mode] = int(bond_modes.get(mode, 0)) + 1
            bond_stats["last"] = entry

    def _audit_complementary_action(self, A, candidate, mode):
        if not self._debug_complementary_action_check:
            return None
        stats = self.complementary_split_stats
        if stats is not None:
            audits = stats.setdefault(
                "action_audits",
                {
                    "enabled": True,
                    "calls": 0,
                    "checked": 0,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                    "tol": float(self._debug_complementary_action_check_tol),
                    "failures": 0,
                },
            )
            if int(audits.get("checked", 0)) >= self._debug_complementary_action_check_limit:
                audits["calls"] = int(audits.get("calls", 0)) + 1
                return None
        else:
            audits = None

        exact = self._matvec_generic(A)
        diff = candidate - exact
        abs_diff = float(diff.norm())
        ref_norm = max(float(exact.norm()), 1.0e-30)
        rel_diff = abs_diff / ref_norm
        record = {
            "bond": self.bond,
            "mode": str(mode),
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "tol": float(self._debug_complementary_action_check_tol),
        }
        if audits is not None:
            audits["calls"] = int(audits.get("calls", 0)) + 1
            audits["checked"] = int(audits.get("checked", 0)) + 1
            audits["max_abs_diff"] = max(float(audits.get("max_abs_diff", 0.0)), abs_diff)
            audits["max_rel_diff"] = max(float(audits.get("max_rel_diff", 0.0)), rel_diff)
            audits["last"] = record
            if abs_diff > self._debug_complementary_action_check_tol:
                audits["failures"] = int(audits.get("failures", 0)) + 1
        if abs_diff > self._debug_complementary_action_check_tol:
            raise AssertionError(
                "Complementary local action mismatch on bond {} in mode {}: "
                "abs_diff={:.6e}, rel_diff={:.6e}, tol={:.6e}".format(
                    self.bond,
                    mode,
                    abs_diff,
                    rel_diff,
                    self._debug_complementary_action_check_tol,
                )
            )
        return record

    @staticmethod
    def _layout(A):
        keys = tuple(sorted(A.data))
        return tuple((key, A.data[key].shape) for key in keys)

    @staticmethod
    def _qns_from_layout(layout):
        if not layout:
            return []
        rank = len(layout[0][0])
        return [sorted({key[axis] for key, _shape in layout}) for axis in range(rank)]

    @staticmethod
    def _size(layout):
        return int(sum(np.prod(shape, dtype=int) for _key, shape in layout))

    @staticmethod
    def _dense_policy(max_dim):
        if isinstance(max_dim, str):
            policy = max_dim.strip().lower()
            if policy == "auto":
                return True, 256, 32
            if policy in {"off", "none", "false", "0"}:
                return False, 0, 0
            try:
                max_dim = int(policy)
            except ValueError as exc:
                raise ValueError("local_dense_max_dim must be an integer, 0, or 'auto'.") from exc
        max_dim = int(max_dim)
        return False, max_dim, 0

    @staticmethod
    def _flatten(A, layout):
        chunks = []
        for key, shape in layout:
            blk = A.data.get(key)
            if blk is None:
                chunks.append(np.zeros(int(np.prod(shape, dtype=int)), dtype=complex))
            else:
                chunks.append(np.asarray(blk).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=complex)

    @staticmethod
    def _unflatten(vec, proto, layout):
        data = {}
        pos = 0
        for key, shape in layout:
            n = int(np.prod(shape, dtype=int))
            data[key] = np.asarray(vec[pos:pos + n]).reshape(shape).copy()
            pos += n
        return BlockTensor(data, HamiltonianMultiplyU1._qns_from_layout(layout), proto.dirs[:])

    def _closed_layout(self, proto, max_dim):
        _auto, cap, _min_dim = self._dense_policy(max_dim)
        max_dim = cap
        layout_map = {key: shape for key, shape in self._layout(proto)}
        dirs = proto.dirs[:]
        for _ in range(4):
            layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
            if self._size(layout) > int(max_dim):
                return None
            data = {}
            rng = np.random.default_rng(123)
            for key, shape in layout:
                data[key] = rng.standard_normal(shape).astype(complex)
            qns = self._qns_from_layout(layout)
            y = self._matvec_generic(BlockTensor(data, qns, dirs))
            changed = False
            for key, blk in y.data.items():
                if key not in layout_map:
                    layout_map[key] = blk.shape
                    changed = True
                elif layout_map[key] != blk.shape:
                    return None
            if not changed:
                return tuple((key, layout_map[key]) for key in sorted(layout_map))
        return None

    def dense_matrix(self, proto, max_dim=256):
        auto, cap, min_dim = self._dense_policy(max_dim)
        if cap <= 0:
            return None, None
        max_dim = cap
        layout = self._closed_layout(proto, max_dim)
        if layout is None:
            return None, None
        dim = self._size(layout)
        if dim <= 0 or dim > int(max_dim):
            return None, None
        if auto and dim < int(min_dim):
            return None, None
        cache_key = layout
        cached = self._dense_cache.get(cache_key)
        if cached is not None:
            return cached, layout

        H = np.zeros((dim, dim), dtype=complex)
        offsets = []
        pos = 0
        for key, shape in layout:
            n = int(np.prod(shape, dtype=int))
            offsets.append((key, shape, pos, n))
            pos += n

        qns = proto.qns[:]
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        for col in range(dim):
            data = {}
            for key, shape, start, n in offsets:
                if start <= col < start + n:
                    arr = np.zeros(shape, dtype=dtype)
                    arr.reshape(-1)[col - start] = 1.0
                    data[key] = arr
                    break
            basis = BlockTensor(data, qns, proto.dirs[:])
            H[:, col] = self._flatten(self.matvec(basis), layout)
        H = 0.5 * (H + H.conj().T)
        self._dense_cache[cache_key] = H
        return H, layout

    @staticmethod
    def _tensor_token(T):
        return (id(T.data), len(T.data))

    def _action_token(self):
        return (
            self._tensor_token(self.E),
            self._tensor_token(self.W[0]),
            self._tensor_token(self.W[1]),
            self._tensor_token(self.F),
        )

    @staticmethod
    def _block_index(T, axes):
        idx = defaultdict(list)
        for key, blk in T.data.items():
            idx[tuple(key[ax] for ax in axes)].append((key, blk))
        return idx

    def _build_action_plan(self, A):
        layout = self._layout(A)
        cache_key = (self._action_token(), layout)
        if cache_key in self._action_cache:
            return self._action_cache[cache_key]

        e_by_ket_l = self._block_index(self.E, (2,))
        w1_by_left_in = self._block_index(self.W[0], (0, 3))
        w2_by_left_in = self._block_index(self.W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(self.F, (0, 2))

        entries = []
        out_shapes = {}
        dtype_args = []
        eq = "aij,jkxy,abux,bcvy,clk->iluv"

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._action_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._action_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._action_cache[cache_key] = None
                        return None
                    for w2_key, w2_blk in w2_by_left_in.get((w1_key[1], p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._action_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._action_cache[cache_key] = None
                                return None
                            try:
                                path = np.einsum_path(
                                    eq, e_blk, a_blk, w1_blk, w2_blk, f_blk,
                                    optimize="greedy",
                                )[0]
                            except ValueError:
                                self._action_cache[cache_key] = None
                                return None
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                e_blk.shape[1],
                                f_blk.shape[1],
                                w1_blk.shape[2],
                                w2_blk.shape[2],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._action_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            dtype_args.extend((e_blk.dtype, w1_blk.dtype, w2_blk.dtype, f_blk.dtype))
                            entries.append((a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, path))
                            if len(entries) > self._compiled_action_cap:
                                self._action_cache[cache_key] = None
                                return None

        if not entries:
            plan = ((), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._action_cache[cache_key] = plan
            return plan

        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (entries, out_shapes, out_qns, A.dirs[:], dtype)
        self._action_cache[cache_key] = plan
        return plan

    def _matvec_compiled(self, A):
        plan = self._build_action_plan(A)
        if plan is None:
            return None
        entries, out_shapes, out_qns, out_dirs, dtype = plan
        out = {key: np.zeros(shape, dtype=dtype) for key, shape in out_shapes.items()}
        eq = "aij,jkxy,abux,bcvy,clk->iluv"
        for a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, path in entries:
            out[out_key] += np.einsum(
                eq, e_blk, A.data[a_key], w1_blk, w2_blk, f_blk,
                optimize=path,
            )
        return BlockTensor(out, out_qns, out_dirs)

    def _matvec_generic(self, A):
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

    @staticmethod
    def _matvec_generic_components(E, W, F, A):
        R = tensordot(E, A, axes=([2], [0]))
        T2 = tensordot(R, W[0], axes=([0, 3], [0, 3]))
        T3 = tensordot(T2, W[1], axes=([3, 2], [0, 3]))
        T4 = tensordot(T3, F, axes=([3, 1], [0, 2]))
        return T4.transpose(0, 3, 1, 2)

    def _matvec_named_family_channels(self, A):
        if not self.complementary_family_environments:
            return None
        channels = {}
        stats = {}
        for name, env in self.complementary_family_environments.items():
            try:
                E, W, F = env
            except Exception:
                continue
            if E is None or F is None or W is None or len(W) != 2:
                continue
            if self._prefer_precontracted_family_environment:
                plan = self._build_direct_operator_plan(
                    A,
                    E,
                    W,
                    F,
                    ("named_family", str(name)),
                )
                applied = self._apply_direct_operator_plan(A, plan)
            else:
                applied = None
            if applied is None:
                tensor = self._matvec_generic_components(E, W, F, A)
                n_entries = None
                n_channels = None
                source = "generic_tensordot_family_environment"
            else:
                tensor, family_channels, entries = applied
                n_entries = int(len(entries))
                n_channels = int(len(family_channels))
                source = "precontracted_family_environment"
            if tensor is not None:
                key = str(name)
                channels[key] = tensor
                stats[key] = {
                    "source": source,
                    "n_entries": n_entries,
                    "n_mpo_middle_channels": n_channels,
                }
        if not channels:
            return None
        return channels, stats

    def _matvec_direct_symbolic_family_channels(self, A):
        if not self.complementary_direct_family_environments:
            return None
        channels = {}
        stats = {}
        for name, entries in self.complementary_direct_family_environments.items():
            total = None
            n_entries = 0
            for E, W, F in entries:
                tensor = self._matvec_generic_components(E, W, F, A)
                total = tensor if total is None else total + tensor
                n_entries += 1
            if total is not None:
                key = str(name)
                channels[key] = total
                stats[key] = {
                    "source": "direct_symbolic_term_environments",
                    "n_entries": int(n_entries),
                }
        if not channels:
            return None
        return channels, stats

    def _matvec_middle_mpo_channel(self, A, channel):
        w1_data = {
            key: value
            for key, value in self.W[0].data.items()
            if key[1] == channel
        }
        w2_data = {
            key: value
            for key, value in self.W[1].data.items()
            if key[0] == channel
        }
        if not w1_data or not w2_data:
            return None
        W1 = BlockTensor(w1_data, self.W[0].qns[:], self.W[0].dirs[:])
        W2 = BlockTensor(w2_data, self.W[1].qns[:], self.W[1].dirs[:])
        original = self.W
        try:
            self.W = [W1, W2]
            return self._matvec_generic(A)
        finally:
            self.W = original

    def _middle_mpo_channels(self):
        channels = {
            key[1]
            for key in self.W[0].data
        }.intersection({
            key[0]
            for key in self.W[1].data
        })
        return tuple(sorted(channels, key=lambda item: repr(item)))

    def _build_boundary_factorized_plan(self, A):
        layout = self._layout(A)
        cache_key = (self._action_token(), layout, "boundary_factorized")
        if cache_key in self._boundary_factorized_cache:
            return self._boundary_factorized_cache[cache_key]

        e_by_ket_l = self._block_index(self.E, (2,))
        w1_by_left_in = self._block_index(self.W[0], (0, 3))
        w2_by_left_in = self._block_index(self.W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(self.F, (0, 2))

        entries = []
        out_shapes = {}
        dtype_args = []
        channels = set()
        eq = "aij,jkxy,abux,bcvy,clk->iluv"

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._boundary_factorized_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._boundary_factorized_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._boundary_factorized_cache[cache_key] = None
                        return None
                    channel = w1_key[1]
                    for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._boundary_factorized_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            expr_key = (
                                e_blk.shape,
                                a_blk.shape,
                                w1_blk.shape,
                                w2_blk.shape,
                                f_blk.shape,
                            )
                            expr = self._boundary_einsum_expr_cache.get(expr_key)
                            if expr is None:
                                try:
                                    import opt_einsum as oe

                                    expr = oe.contract_expression(
                                        eq,
                                        e_blk.shape,
                                        a_blk.shape,
                                        w1_blk.shape,
                                        w2_blk.shape,
                                        f_blk.shape,
                                        optimize="greedy",
                                    )
                                except Exception:
                                    try:
                                        expr = np.einsum_path(
                                            eq,
                                            e_blk,
                                            a_blk,
                                            w1_blk,
                                            w2_blk,
                                            f_blk,
                                            optimize="greedy",
                                        )[0]
                                    except ValueError:
                                        self._boundary_factorized_cache[cache_key] = None
                                        return None
                                self._boundary_einsum_expr_cache[expr_key] = expr
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                e_blk.shape[1],
                                f_blk.shape[1],
                                w1_blk.shape[2],
                                w2_blk.shape[2],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            dtype_args.extend((e_blk.dtype, w1_blk.dtype, w2_blk.dtype, f_blk.dtype))
                            channels.add(channel)
                            entries.append((channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, expr))
                            if len(entries) > int(self._boundary_factorized_action_cap):
                                self._boundary_factorized_cache[cache_key] = None
                                return None

        if not entries:
            plan = ((), (), (), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._boundary_factorized_cache[cache_key] = plan
            return plan

        batch_sources = {}
        for channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, _expr in entries:
            batch_key = (
                out_key,
                e_blk.shape,
                A.data[a_key].shape,
                w1_blk.shape,
                w2_blk.shape,
                f_blk.shape,
            )
            batch_sources.setdefault(batch_key, []).append(
                (channel, a_key, e_blk, w1_blk, w2_blk, f_blk)
            )

        batched_groups = []
        batch_eq = "zaij,zjkxy,zabux,zbcvy,zclk->iluv"
        for batch_key, members in batch_sources.items():
            out_key, e_shape, a_shape, w1_shape, w2_shape, f_shape = batch_key
            if len(members) < 2:
                continue
            expr_key = (len(members), e_shape, a_shape, w1_shape, w2_shape, f_shape)
            expr = self._boundary_batched_einsum_expr_cache.get(expr_key)
            if expr is None:
                try:
                    import opt_einsum as oe

                    expr = oe.contract_expression(
                        batch_eq,
                        (len(members),) + tuple(e_shape),
                        (len(members),) + tuple(a_shape),
                        (len(members),) + tuple(w1_shape),
                        (len(members),) + tuple(w2_shape),
                        (len(members),) + tuple(f_shape),
                        optimize="greedy",
                    )
                except Exception:
                    expr = None
                self._boundary_batched_einsum_expr_cache[expr_key] = expr
            if expr is None:
                continue
            try:
                e_stack = np.stack([entry[2] for entry in members], axis=0)
                w1_stack = np.stack([entry[3] for entry in members], axis=0)
                w2_stack = np.stack([entry[4] for entry in members], axis=0)
                f_stack = np.stack([entry[5] for entry in members], axis=0)
            except ValueError:
                continue
            batched_groups.append(
                (
                    out_key,
                    tuple(entry[0] for entry in members),
                    tuple(entry[1] for entry in members),
                    e_stack,
                    w1_stack,
                    w2_stack,
                    f_stack,
                    expr,
                )
            )

        batched_entry_ids = {
            (out_key, a_key, channel)
            for out_key, channels, a_keys, *_rest in batched_groups
            for channel, a_key in zip(channels, a_keys)
        }
        scalar_entries = tuple(
            entry
            for entry in entries
            if (entry[2], entry[1], entry[0]) not in batched_entry_ids
        )

        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            scalar_entries,
            tuple(batched_groups),
            out_shapes,
            out_qns,
            A.dirs[:],
            dtype,
        )
        self._boundary_factorized_cache[cache_key] = plan
        return plan

    def _matvec_boundary_factorized(
        self,
        A,
        *,
        local_channels=None,
        collect_channels=False,
    ):
        plan = self._build_boundary_factorized_plan(A)
        if plan is None:
            return None
        channels, entries, batched_groups, out_shapes, out_qns, out_dirs, dtype = plan
        batched_entries = sum(len(group[2]) for group in batched_groups)
        scalar_entries = len(entries)
        if (
            not collect_channels
            and scalar_entries
            and batched_entries <= scalar_entries
        ):
            return None
        eq = "aij,jkxy,abux,bcvy,clk->iluv"
        total_data = {
            key: np.zeros(shape, dtype=dtype)
            for key, shape in out_shapes.items()
        }
        channel_data = None
        if collect_channels:
            channel_data = {
                channel: {
                    key: np.zeros(shape, dtype=dtype)
                    for key, shape in out_shapes.items()
                }
                for channel in channels
            }
        if not collect_channels:
            for out_key, _channels, a_keys, e_stack, w1_stack, w2_stack, f_stack, expr in batched_groups:
                a_stack = np.stack([A.data[a_key] for a_key in a_keys], axis=0)
                total_data[out_key] += expr(
                    e_stack,
                    a_stack,
                    w1_stack,
                    w2_stack,
                    f_stack,
                )
        for channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, expr in entries:
            if callable(expr):
                contribution = expr(e_blk, A.data[a_key], w1_blk, w2_blk, f_blk)
            else:
                contribution = np.einsum(
                    eq,
                    e_blk,
                    A.data[a_key],
                    w1_blk,
                    w2_blk,
                    f_blk,
                    optimize=expr,
                )
            total_data[out_key] += contribution
            if channel_data is not None:
                channel_data[channel][out_key] += contribution

        total = BlockTensor(total_data, out_qns, out_dirs)
        if not total.data:
            return None

        channel_tensors = {}
        if channel_data is not None:
            channel_tensors = {
                f"mpo_middle:{repr(channel)}": BlockTensor(data, out_qns, out_dirs)
                for channel, data in channel_data.items()
            }

        local_channels = local_channels or {}
        for name, tensor in local_channels.items():
            if collect_channels:
                channel_tensors[f"subtract_local:{name}"] = tensor * -1.0
            total = total - tensor

        return {
            "total": total,
            "channels": channel_tensors,
            "stats": {
                "kind": "abelian_complementary_boundary_factorized_action",
                "source": "middle_mpo_channels_minus_local_RP",
                "bond": self.bond,
                "n_entries": int(
                    len(entries)
                    + sum(len(group[2]) for group in batched_groups)
                ),
                "n_batched_groups": int(len(batched_groups)),
                "n_scalar_entries": int(len(entries)),
                "n_mpo_middle_channels": int(len(channels)),
                "n_channels": int(len(channels) + len(local_channels)),
                "channel_names": tuple(
                    f"mpo_middle:{repr(channel)}" for channel in channels
                )
                + tuple(f"subtract_local:{name}" for name in local_channels),
                "channels_materialized": bool(collect_channels),
            },
        }

    def _two_site_mpo(self):
        if self._w12_cache is not None:
            return self._w12_cache
        W12 = tensordot(self.W[0], self.W[1], axes=([1], [0]))
        W12 = W12.transpose(0, 3, 1, 4, 2, 5)
        self._w12_cache = W12
        return W12

    def _matvec_fused_mpo(self, A):
        W12 = self._two_site_mpo()
        R = tensordot(self.E, A, axes=([2], [0]))
        T = tensordot(R, W12, axes=([0, 3, 4], [0, 4, 5]))
        T = tensordot(T, self.F, axes=([2, 1], [0, 2]))
        return T.transpose(0, 3, 1, 2)

    @staticmethod
    def _sector_components(sector):
        labels = tuple(getattr(sector, "labels", ()))
        comps = tuple(getattr(sector, "components", ()))
        return dict(zip(labels, comps))

    @classmethod
    def _spatial_state_indices_from_qns(cls, qns):
        found = {}
        for idx, qn in enumerate(qns):
            comps = cls._sector_components(qn)
            if "charge" not in comps or "sz" not in comps:
                return None
            key = (int(comps["charge"]), int(comps["sz"]))
            if key in found:
                return None
            found[key] = int(idx)
        required = ((0, 0), (1, 1), (1, -1), (2, 0))
        if any(key not in found for key in required):
            return None
        return tuple(found[key] for key in required)

    @staticmethod
    def _axis_dims(A, axis):
        dims = {}
        for key, block in A.data.items():
            qn = key[axis]
            dim = int(block.shape[axis])
            old = dims.get(qn)
            if old is not None and old != dim:
                return None
            dims[qn] = dim
        return dims

    def local_complementary_matrix(self, A=None):
        if self.complementary_operator_families is None or self.bond is None:
            return None
        cache_key = (id(self.complementary_operator_families), int(self.bond))
        mat = self._local_complementary_cache.get(cache_key)
        if mat is None:
            mat = self._GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE.get(cache_key)
        if mat is None:
            try:
                from pyqed.qchem.dmrg.spatial_terms import (
                    spatial_complementary_local_matrix,
                )
            except Exception:
                return None
            mat = spatial_complementary_local_matrix(
                self.complementary_operator_families,
                int(self.bond),
            )
            self._local_complementary_cache[cache_key] = mat
            self._GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE[cache_key] = mat
        if A is None:
            return mat
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        order = [4 * i + j for i in left_order for j in right_order]
        return mat[np.ix_(order, order)]

    def local_complementary_channel_matrices(self, A=None):
        if self.complementary_operator_families is None or self.bond is None:
            return None
        cache_key = (id(self.complementary_operator_families), int(self.bond))
        mats = self._local_complementary_channel_cache.get(cache_key)
        if mats is None:
            mats = self._GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE.get(cache_key)
        if mats is None:
            try:
                from pyqed.qchem.dmrg.spatial_terms import (
                    spatial_complementary_local_matrices,
                )
            except Exception:
                return None
            mats = spatial_complementary_local_matrices(
                self.complementary_operator_families,
                int(self.bond),
            )
            self._local_complementary_channel_cache[cache_key] = mats
            self._GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE[cache_key] = mats
        if A is None:
            return mats
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        order = [4 * i + j for i in left_order for j in right_order]
        return {
            name: np.asarray(mat, dtype=complex)[np.ix_(order, order)]
            for name, mat in mats.items()
        }

    @property
    def complementary_local_metadata(self):
        mat = self.local_complementary_matrix()
        if mat is None:
            return None
        return {
            "enabled": True,
            "bond": self.bond,
            "matrix_shape": tuple(int(x) for x in mat.shape),
            "nnz": int(np.count_nonzero(np.abs(mat) > 1.0e-14)),
            "norm": float(np.linalg.norm(mat)),
            "channels": {
                str(name): {
                    "matrix_shape": tuple(int(x) for x in np.asarray(channel_mat).shape),
                    "nnz": int(np.count_nonzero(np.abs(channel_mat) > 1.0e-14)),
                    "norm": float(np.linalg.norm(channel_mat)),
                }
                for name, channel_mat in (
                    self.local_complementary_channel_matrices() or {}
                ).items()
            },
        }

    def _apply_local_complementary_matrix(self, A, mat, tol=1.0e-14):
        if mat is None:
            return None
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        p1_dims = self._axis_dims(A, 2)
        p2_dims = self._axis_dims(A, 3)
        if p1_dims is None or p2_dims is None:
            return None
        p1_qns = tuple(A.qns[2])
        p2_qns = tuple(A.qns[3])
        pair_qns = tuple((q1, q2) for q1 in p1_qns for q2 in p2_qns)
        n_pair = len(pair_qns)
        mat = np.asarray(mat, dtype=complex).reshape(n_pair, n_pair)
        out = {}
        for in_key, block in A.data.items():
            ql, qr, p1_in, p2_in = in_key
            try:
                in_col = pair_qns.index((p1_in, p2_in))
            except ValueError:
                continue
            for out_row, (p1_out, p2_out) in enumerate(pair_qns):
                coeff = mat[out_row, in_col]
                if abs(coeff) <= tol:
                    continue
                p1_dim = int(p1_dims.get(p1_out, block.shape[2]))
                p2_dim = int(p2_dims.get(p2_out, block.shape[3]))
                if p1_dim != block.shape[2]:
                    continue
                if p2_dim != block.shape[3]:
                    continue
                out_key = (ql, qr, p1_out, p2_out)
                contrib = coeff * block
                if out_key in out:
                    out[out_key] += contrib
                else:
                    out[out_key] = contrib.copy()
        return BlockTensor(out, A.qns[:], A.dirs[:])

    def _matvec_local_complementary_channels(self, A, tol=1.0e-14):
        mats = self.local_complementary_channel_matrices(A)
        if mats is None:
            return None
        out = {}
        for name, mat in mats.items():
            applied = self._apply_local_complementary_matrix(A, mat, tol=tol)
            if applied is not None and applied.norm() > tol:
                out[str(name)] = applied
        return out

    def _matvec_local_complementary(self, A, tol=1.0e-14):
        return self._apply_local_complementary_matrix(
            A,
            self.local_complementary_matrix(A),
            tol=tol,
        )

    def split_local_action(self, A):
        """
        Return the exact complementary split for this local problem.

        When complementary payload matvecs are enabled, the boundary term is
        applied through the factorized middle-channel plan and local R/P
        channels are subtracted explicitly.  Otherwise, small local spaces may
        use a dense boundary table before falling back to the exact residual
        ``full - local``.
        """
        if self._prefer_boundary_factorized:
            local_channels = self._matvec_local_complementary_channels(A)
            if local_channels is None:
                local = None
            else:
                local = None
                for value in local_channels.values():
                    local = value if local is None else local + value
            direct_symbolic = self._matvec_direct_symbolic_family_channels(A)
            if direct_symbolic is not None:
                direct_channels, direct_stats = direct_symbolic
                total = None
                boundary_channels = {}
                for name, tensor in direct_channels.items():
                    total = tensor if total is None else total + tensor
                    boundary = tensor
                    if local_channels is not None and name in local_channels:
                        boundary = boundary - local_channels[name]
                    boundary_channels[f"{name}:boundary"] = boundary
                if total is not None:
                    mode = (
                        "direct_RP_symbolic_environment"
                        if local is not None
                        else "direct_symbolic_environment"
                    )
                    self._audit_complementary_action(A, total, mode)
                    stats = {
                        "kind": "abelian_direct_symbolic_family_environment_action",
                        "source": "direct_symbolic_term_environments",
                        "bond": self.bond,
                        "family_names": tuple(direct_channels),
                        "n_channels": int(
                            len(direct_channels)
                            + len(local_channels or {})
                        ),
                        "channel_names": tuple(
                            f"{name}:full" for name in direct_channels
                        )
                        + tuple(
                            f"{name}:local" for name in (local_channels or {})
                        ),
                        "channels_materialized": True,
                        "family_channel_stats": direct_stats,
                    }
                    self._record_complementary_split(
                        mode,
                        local=local,
                        local_channels=local_channels,
                        boundary=None if local is None else total - local,
                        boundary_operator=stats,
                    )
                    return {
                        "total": total,
                        "local": local,
                        "local_channels": local_channels,
                        "boundary": None if local is None else total - local,
                        "boundary_channels": boundary_channels,
                        "boundary_operator": stats,
                        "family_channels": direct_channels,
                        "mode": mode,
                    }
            named_family = self._matvec_named_family_channels(A)
            if named_family is not None:
                named_family_channels, named_family_stats = named_family
                total = None
                boundary_channels = {}
                for name, tensor in named_family_channels.items():
                    total = tensor if total is None else total + tensor
                    boundary = tensor
                    if local_channels is not None and name in local_channels:
                        boundary = boundary - local_channels[name]
                    boundary_channels[f"{name}:boundary"] = boundary
                if total is not None:
                    mode = (
                        "named_RP_family_environment"
                        if local is not None
                        else "named_family_environment"
                    )
                    self._audit_complementary_action(A, total, mode)
                    stats = {
                        "kind": "abelian_named_family_environment_action",
                        "source": "family_mpo_environments",
                        "bond": self.bond,
                        "family_names": tuple(named_family_channels),
                        "n_channels": int(
                            len(named_family_channels)
                            + len(local_channels or {})
                        ),
                        "channel_names": tuple(
                            f"{name}:full" for name in named_family_channels
                        )
                        + tuple(
                            f"{name}:local" for name in (local_channels or {})
                        ),
                        "channels_materialized": True,
                        "family_channel_stats": named_family_stats,
                    }
                    self._record_complementary_split(
                        mode,
                        local=local,
                        local_channels=local_channels,
                        boundary=None if local is None else total - local,
                        boundary_operator=stats,
                    )
                    return {
                        "total": total,
                        "local": local,
                        "local_channels": local_channels,
                        "boundary": None if local is None else total - local,
                        "boundary_channels": boundary_channels,
                        "boundary_operator": stats,
                        "family_channels": named_family_channels,
                        "mode": mode,
                    }
        else:
            local_channels = None
            local = self._matvec_local_complementary(A)
        if local is None:
            local = None
        if self._prefer_boundary_factorized:
            boundary_direct = self._matvec_boundary_direct_operator(
                A,
                local_channels=local_channels,
            )
            if boundary_direct is not None:
                boundary = boundary_direct["total"]
                mode = (
                    "local_RP_plus_boundary_direct_operator_table"
                    if local is not None
                    else "boundary_direct_operator_no_local_complementary"
                )
                total = boundary if local is None else boundary + local
                self._audit_complementary_action(A, total, mode)
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_operator=boundary_direct["stats"],
                )
                return {
                    "total": total,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_channels": boundary_direct["channels"],
                    "boundary_operator": boundary_direct["stats"],
                    "mode": mode,
                }
            boundary_factorized = self._matvec_boundary_factorized(
                A,
                local_channels=local_channels,
            )
            if boundary_factorized is not None:
                boundary = boundary_factorized["total"]
                mode = (
                    "local_RP_plus_boundary_factorized_operator_table"
                    if local is not None
                    else "boundary_factorized_no_local_complementary"
                )
                total = boundary if local is None else boundary + local
                self._audit_complementary_action(A, total, mode)
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_operator=boundary_factorized["stats"],
                )
                return {
                    "total": total,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_channels": boundary_factorized["channels"],
                    "boundary_operator": boundary_factorized["stats"],
                    "mode": mode,
                }
        if local is not None:
            boundary_table = self._boundary_table(A)
            if boundary_table is not None:
                boundary = boundary_table.apply(A)
                mode = "local_RP_plus_boundary_operator_table"
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_table=boundary_table,
                )
                return {
                    "total": boundary + local,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_table": boundary_table,
                    "mode": mode,
                }

        full = self._matvec_generic(A)
        if local is None:
            mode = "full_mpo_no_local_complementary"
            self._record_complementary_split(mode, local=None, boundary=full)
            return {
                "total": full,
                "local": None,
                "local_channels": None,
                "boundary": full,
                "mode": mode,
            }
        boundary = full - local
        mode = "local_RP_plus_boundary_residual"
        self._record_complementary_split(
            mode,
            local=local,
            local_channels=local_channels,
            boundary=boundary,
        )
        return {
            "total": boundary + local,
            "local": local,
            "local_channels": local_channels,
            "boundary": boundary,
            "mode": mode,
        }

    def _boundary_table(self, proto):
        cap = int(self._boundary_table_max_dim)
        layout = self._closed_layout(proto, cap)
        if layout is None:
            return None
        dim = self._size(layout)
        if dim <= 0 or dim > cap:
            return None
        cache_key = (int(self.bond or 0), layout)
        cached = self._boundary_table_cache.get(cache_key)
        if cached is not None:
            return cached

        H = np.zeros((dim, dim), dtype=complex)
        channel_matrices = {}
        if self._debug_boundary_channel_matrices:
            middle_channels = self._middle_mpo_channels()
            channel_matrices = {
                f"mpo_middle:{repr(channel)}": np.zeros((dim, dim), dtype=complex)
                for channel in middle_channels
            }
            local_channel_names = tuple(
                (self.local_complementary_channel_matrices(proto) or {}).keys()
            )
            for name in local_channel_names:
                channel_matrices[f"subtract_local:{name}"] = np.zeros((dim, dim), dtype=complex)
        else:
            middle_channels = ()
        offsets = []
        pos = 0
        for key, shape in layout:
            n = int(np.prod(shape, dtype=int))
            offsets.append((key, shape, pos, n))
            pos += n
        qns = proto.qns[:]
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        for col in range(dim):
            data = {}
            for key, shape, start, n in offsets:
                if start <= col < start + n:
                    arr = np.zeros(shape, dtype=dtype)
                    arr.reshape(-1)[col - start] = 1.0
                    data[key] = arr
                    break
            basis = BlockTensor(data, qns, proto.dirs[:])
            full = self._matvec_generic(basis)
            local = self._matvec_local_complementary(basis)
            if local is None:
                return None
            H[:, col] = self._flatten(full - local, layout)
            if self._debug_boundary_channel_matrices:
                for channel in middle_channels:
                    contribution = self._matvec_middle_mpo_channel(basis, channel)
                    if contribution is not None:
                        channel_matrices[f"mpo_middle:{repr(channel)}"][:, col] = (
                            self._flatten(contribution, layout)
                        )
                local_channels = self._matvec_local_complementary_channels(basis) or {}
                for name, contribution in local_channels.items():
                    channel_matrices[f"subtract_local:{name}"][:, col] = -self._flatten(
                        contribution,
                        layout,
                    )
        if channel_matrices:
            channel_sum = sum(channel_matrices.values(), np.zeros_like(H))
            unresolved = H - channel_sum
            if np.linalg.norm(unresolved) > 1.0e-10:
                channel_matrices["unresolved_residual"] = unresolved
        table = AbelianComplementaryBoundaryActionTable(
            H,
            layout,
            qns,
            proto.dirs[:],
            bond=self.bond,
            source="exact_full_mpo_minus_local_RP",
            boundary_family_tables=self._boundary_family_tables(),
            channel_matrices=channel_matrices,
        )
        self._boundary_table_cache[cache_key] = table
        return table

    def _matvec_boundary_table(self, A):
        table = self._boundary_table(A)
        if table is None:
            return None
        return table.apply(A)

    def _boundary_family_tables(self):
        payloads = self.complementary_boundary_payloads or {}
        tables = []
        for side in ("left", "right"):
            entry = payloads.get(side)
            table = None if entry is None else entry.family_operator_table
            if table is not None:
                tables.append(table)
        return tuple(tables)

    def _compact_complementary_split_metadata(self):
        def _table_summary(entry):
            table = None if entry is None else entry.family_operator_table
            if table is None:
                return None
            return {
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            }

        def _payload_summary(entry):
            if entry is None:
                return None
            return {
                "side": str(entry.side),
                "bond": int(entry.bond),
                "numeric_payload_terms": int(
                    sum(payload.n_terms for payload in entry.family_payloads.values())
                ),
                "numeric_payload_cross_terms": int(
                    sum(payload.cross_terms for payload in entry.family_payloads.values())
                ),
            }

        payloads = self.complementary_boundary_payloads or {}
        left = payloads.get("left")
        right = payloads.get("right")
        local = self.complementary_local_metadata
        local_summary = None if local is None else {
            "bond": local.get("bond"),
            "matrix_shape": local.get("matrix_shape"),
            "nnz": local.get("nnz"),
            "norm": local.get("norm"),
            "channels": local.get("channels", {}),
        }
        return {
            "enabled": self.complementary_operator_families is not None,
            "bond": self.bond,
            "local": local_summary,
            "boundary_table_cache_entries": int(len(self._boundary_table_cache)),
            "left_family_operator_table": _table_summary(left),
            "right_family_operator_table": _table_summary(right),
            "left_payload": _payload_summary(left),
            "right_payload": _payload_summary(right),
        }

    @property
    def complementary_split_metadata(self):
        payloads = self.complementary_boundary_payloads or {}
        left = payloads.get("left")
        right = payloads.get("right")
        return {
            "enabled": self.complementary_operator_families is not None,
            "bond": self.bond,
            "local": self.complementary_local_metadata,
            "boundary_source": "cached_boundary_table_or_full_mpo_residual",
            "boundary_table_max_dim": int(self._boundary_table_max_dim),
            "boundary_table_cache_entries": int(len(self._boundary_table_cache)),
            "boundary_action_tables": tuple(
                table.stats for table in self._boundary_table_cache.values()
            ),
            "left_family_operator_table": (
                None
                if left is None or left.family_operator_table is None
                else left.family_operator_table.stats
            ),
            "right_family_operator_table": (
                None
                if right is None or right.family_operator_table is None
                else right.family_operator_table.stats
            ),
            "left_payload": None if left is None else left.stats,
            "right_payload": None if right is None else right.stats,
        }

    def matvec(self, A):
        if self.complementary_operator_families is not None:
            return self.split_local_action(A)["total"]
        A_new = self._matvec_compiled(A)
        if A_new is not None:
            return A_new
        return self._matvec_generic(A)

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
    zero_qn = zero_like_sector(first_val)
    # Track allowed Right-Bond QNs. Start with Vacuum (Left=0).
    # Store (Dense_Index, QN_Value)
    current_nodes = {(0, zero_qn)}
    logger.info(f"  [MPO Convert] Start. Sites={len(dense_mpo_list)}, ZeroQN={zero_qn} (Type: {type(zero_qn)})")
    for site_idx, W in enumerate(dense_mpo_list):
        new_data = {}
        next_nodes = set()
        phys_qns = site_qn_maps[site_idx]
        # Metadata for BlockTensor
        all_phys_out = sorted(list(set(phys_qns.values())))
        all_phys_in = sorted(list(set(phys_qns.values())))
        unique_phys_qns = len(all_phys_out) == len(phys_qns)
        if unique_phys_qns:
            curr_by_q = defaultdict(list)
            for l_idx, q_l in current_nodes:
                if not is_sector_like(q_l):
                    raise TypeError(
                        f"Site {site_idx}: q_l became {type(q_l)} ({q_l})! Expected sector-like symmetry label."
                    )
                curr_by_q[q_l].append(int(l_idx))
            curr_by_q = {q: sorted(rows) for q, rows in curr_by_q.items()}
            phys_items = sorted(phys_qns.items())
            n_right = int(W.shape[1])
            all_right = np.arange(n_right)
            transition_cache = []
            for out_s, q_out in phys_items:
                for in_s, q_in in phys_items:
                    try:
                        flux = q_out - q_in
                    except TypeError as exc:
                        raise TypeError(
                            "dense_to_symmetric_mpo currently requires Abelian sector differences on physical legs. "
                            "The new symmetry layer can host U(1)xSU(2) sectors, but non-Abelian MPO conversion "
                            "still needs a reduced-tensor implementation."
                        ) from exc
                    transition_cache.append((int(out_s), int(in_s), q_out, q_in, flux))

            for q_l, rows in curr_by_q.items():
                row_arr = np.asarray(rows, dtype=int)
                for out_s, in_s, _q_out, _q_in, flux in transition_cache:
                    q_r = q_l - flux
                    sub = W[np.ix_(row_arr, all_right)][:, :, out_s, in_s]
                    if sub.size == 0:
                        continue
                    cols = np.nonzero(np.any(np.abs(sub) > tol, axis=0))[0]
                    for r in cols:
                        next_nodes.add((int(r), q_r))

            l_map = {q: [(int(l), q) for l in rows] for q, rows in curr_by_q.items()}
            r_map = {q: sorted([x for x in next_nodes if x[1] == q]) for q in set(x[1] for x in next_nodes)}
            final_blocks = {}
            for q_l, row_nodes in l_map.items():
                row_idx = np.asarray([node[0] for node in row_nodes], dtype=int)
                for out_s, in_s, q_out, q_in, flux in transition_cache:
                    q_r = q_l - flux
                    col_nodes = r_map.get(q_r)
                    if not col_nodes:
                        continue
                    col_idx = np.asarray([node[0] for node in col_nodes], dtype=int)
                    mat = W[np.ix_(row_idx, col_idx)][:, :, out_s, in_s]
                    if not np.any(np.abs(mat) > tol):
                        continue
                    final_blocks[(q_l, q_r, q_out, q_in)] = mat[:, :, None, None].copy()

            qns_L = sorted(list(l_map.keys()))
            qns_R = sorted(list(r_map.keys()))
            qns_Out = all_phys_out
            qns_In = all_phys_in
            bt = BlockTensor(final_blocks, [qns_L, qns_R, qns_Out, qns_In], [-1, 1, 1, -1])
            sym_H.append(bt)
            if site_idx == 0 and len(final_blocks) > 0:
                sample_key = next(iter(final_blocks.keys()))
                if not is_sector_like(sample_key[0]):
                     print(f"  [ERROR] Site 0 generated invalid sector keys: {sample_key}.")
            current_nodes = next_nodes
            continue

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
            try:
                flux = q_out - q_in
            except TypeError as exc:
                raise TypeError(
                    "dense_to_symmetric_mpo currently requires Abelian sector differences on physical legs. "
                    "The new symmetry layer can host U(1)xSU(2) sectors, but non-Abelian MPO conversion "
                    "still needs a reduced-tensor implementation."
                ) from exc
            for q_l in valid_incoming[l]:
                if not is_sector_like(q_l):
                    raise TypeError(
                        f"Site {site_idx}: q_l became {type(q_l)} ({q_l})! Expected sector-like symmetry label."
                    )
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
            if not is_sector_like(sample_key[0]):
                 print(f"  [ERROR] Site 0 generated invalid sector keys: {sample_key}.")
        current_nodes = next_nodes
    return sym_H


class UniformMPS:
    #TODO uniform MPS
    def __init__(self, Bs, labels='lpr'):

        self.labels = labels
        self.p_idx = self.labels.index('p')




class MPS:
    def __init__(self, Bs, Ss=None, bc='finite', \
                 labels=['lv', 'p', 'rv'], homogenous=False, center=-1, gauge=None):
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
            - 'periodic': periodic Boundary Conditions (IBC/PBC).
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
            Number of bonds (L-1 for finite, L for periodic).
        dim : int
            Physical dimension (d) of the sites.
        lv_idx, p_idx, rv_idx : int
            Cached integer positions of the axes based on `labels`.
        center : int
            Canonical center site index. Default is -1 (no specific center).
        """
        assert bc in ['finite', 'periodic']
        self.bc = bc

        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'finite' else self.L        
        self.Ss = Ss
        
        if (center!= -1) and gauge is None:
            self.center = center 
            self.gauge = None
        elif (center == -1) and gauge is not None:
            # assign canonical center by the canonical form of the assigned MPS state
            gauge = gauge.lower()
            self.gauge = gauge
            
            if gauge in ['left', 'lv', 'l']:
                self.center = self.L - 1
            elif gauge in ['right', 'rv', 'r']:
                self.center = 0
            elif gauge in ['mixed']:
                assert isinstance(center, int)
                if not 0 <= center <= self.L:
                    raise ValueError(f"Invalid center index {center} for MPS with {self.L} sites.")
                self.center = center
            
            else:
                raise ValueError('Unrecognized gauge {gauge} for MPS')
        
        elif (center == -1) and gauge is None:
            # print('You are creating a MPS without a gauge. Suggest calling right_canonicalize() for canonicalization first.")')
            self.gauge = None
            self.center = center
        else:
            raise ValueError('Cannot specify both gauge and center. Use only one.')

        

        # leg sequence
        if labels is None:
            warnings.warn("MPS labels not specified, assuming ['lv', 'p', 'rv'].")
            self.labels = ['lv', 'p', 'rv']
        else:

            if len(labels) != 3:
                 warnings.warn(f"Warning: You provided {len(labels)} labels but MPS tensors are usually Rank-3. Ensure your boundaries have dummy indices.")
            self.labels = labels
            
        try:
            self.lv_idx = self.labels.index('lv')
            self.rv_idx = self.labels.index('rv')
            self.p_idx = self.labels.index('p')
        except ValueError as e:
            missing_label = str(e).split()[-1]
            raise ValueError(f"MPS initialization failed: The label list {self.labels} is missing the required label {missing_label}.")
        
        # order legs to [left, phys, right] 
        if self.lv_idx != 0 and self.p_idx != 1:
            Bs = [B.transpose(self.lv_idx, self.p_idx, self.rv_idx) for B in Bs]
        
        self.Bs = self.data = self.factors = Bs


        self.homogenous = homogenous
        if homogenous:
            try:
                self.dim = Bs[0].shape[1]
            except TypeError:
                if hasattr(Bs[0], 'qns'):
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
                    if hasattr(B, 'qns'):
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

            # if val < 1e-12: raise warnings.warn('Norm {val} is too small.')
            val_scalar = np.abs(np.atleast_1d(val)[0])
            if val_scalar < 1e-12:
                import warnings
                warnings.warn(f'Norm {val_scalar} is too small.')

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
        """Returns a new MPS with tensors transposed to target_labels. 
        DEPRECATED. Use transpose()"""
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
        if hasattr(B, 'qns') and isinstance(B.data, dict):
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
    
    def get_singular_values(self, bond_id):
        pass

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

    def __getitem__(self, i):
        """Allows reading site tensors like a list: tensor = mps[i]"""
        return self.Bs[i]

    def __setitem__(self, i, value):
        """Allows updating site tensors like a list: mps[i] = new_tensor"""
        self.Bs[i] = value
        # Note: Because self.Bs, self.data, and self.factors point to the same list, 
        # updating self.Bs[i] automatically updates the others!

    def __len__(self):
        """Allows getting the number of sites using len(mps)"""
        return self.L
    
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
        if self.center == -1:
            raise NotImplementedError("need to first do canonicalization to have a center site for get_theta1(), currently have not implemented the functions for that. TODO: maybe we will do self.shift_center(i) later. Buy me a coffee to prioritize this feature.")
        # Right of Center
        if i > self.center:
            if i == 0:
                if self.bc == 'periodic':
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
        assert self.bc == 'periodic'  # works only in the periodic case
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
            self.center = self.L - 1
            return self
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
        self.center = self.L - 1
        return self

    def right_canonicalize(self):
        """
        Sweeps from Right (L-1) to Left (0) to transform the MPS into Right-Canonical Form.
        Effect:
        - Tensors Bs[1]...Bs[L-1] become Right-Isometries (B).
        - Populates self.Ss with bond weights.
        - Moves orthogonality center to the first site (0).
        """
        if SYMMETRY_AVAILABLE and isinstance(self.Bs[0], BlockTensor):
            self.center = 0
            return self
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
        self.center = 0
        return self

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

    def _calc_local_site_rdms(self, idx=None):
        """
        Calculate the local reduced density matrix for individual, isolated sites.
        (it is not 1 site rdm getting all <c^\dagger_i c_j>, this function only provides local information, such as the probability of the site being empty, singly occupied, or doubly occupied (<c^\dagger_i c_i>).

        Parameters
        ----------
        idx : int, list of int, tuple of int, or None, optional
            The specific site index (or indices) to calculate the local RDM for. 
            If None, calculates the local RDM for all sites in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices to their corresponding 
            $d \times d$ local density matrices (as numpy arrays), where $d$ is the 
            local physical dimension of the site.

        Raises
        ------
        ValueError
            If `idx` is not an int, list, tuple, or None.
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

        if SYMMETRY_AVAILABLE and hasattr(self.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_self = symmetric_to_dense(self)
            return dense_self._calc_local_site_rdms(idx=idx)

        # 1. Build Left Environments
        L_env = [np.array([[1.0]], dtype=complex)]
        curr_L = L_env[0]
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            tmp = np.tensordot(curr_L, B, axes=(1, 0))
            curr_L = np.tensordot(tmp, B.conj(), axes=([0, 1], [0, 1])).T
            L_env.append(curr_L)
            
        # 2. Build Right Environments
        R_env = [None] * self.L
        curr_R = np.array([[1.0]], dtype=complex)
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            tmp = np.tensordot(B, curr_R, axes=(2, 1))
            curr_R = np.tensordot(tmp, B.conj(), axes=([1, 2], [1, 2])).T
            R_env[i-1] = curr_R

        # 3. Assemble Local RDMs
        rdm = {}
        for i in idx:
            B = self._get_std_B(i)
            # Contract L_env with B -> tmp1(Bra_L, P, R)
            tmp1 = np.tensordot(L_env[i], B, axes=(1, 0))
            # Contract tmp1 with R_env -> tmp2(Bra_L, P, Bra_R)
            tmp2 = np.tensordot(tmp1, R_env[i], axes=(2, 1))
            # Contract tmp2 with B* -> rho(P, P*)
            rho = np.tensordot(tmp2, B.conj(), axes=([0, 2], [0, 2]))
            
            # Normalize to ensure Tr(rho) = 1
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho /= tr
            rdm[i] = rho
        return rdm
    
    def make_local_site_rdm(self, idx=None):
        """
        Wrapper for local one-site reduced density matrices.
        """
        return self._calc_local_site_rdms(idx=idx)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the 2-site density-density correlation <n_i n_j> for specified pairs of sites.
        
        This method computes the exact local two-site probability trace by explicitly 
        building the left and right environments. To maximize memory efficiency for 
        quantum chemistry applications using spin-orbitals (where local dimension d=2), 
        it discards the full 16-element density matrix and strictly returns the 
        joint occupation probability |11><11|.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the correlation for. 
            If None, the function calculates the values for all possible unique 
            pairs `i < j` in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding scalar correlation value `<n_i n_j>` (as a real float).
            
        Notes
        -----
        This method assumes a spin-orbital mapping where the local physical dimension 
        is d=2 (Empty, Occupied). The returned scalar corresponds to the bottom-right 
        diagonal element of the theoretical 4x4 two-site reduced density matrix.
        """
        if SYMMETRY_AVAILABLE and hasattr(self.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_self = symmetric_to_dense(self)
            return dense_self.make_diagonal_rdm2(idx_pairs=idx_pairs)

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

        # 1) Build Left Environments
        L_env = [np.array([[1.0]])]
        curr_L = L_env[0]
        for i in range(self.L - 1):
            B = self._get_std_B(i)
            temp = np.tensordot(L_env[-1], B, axes=(1, 0))
            curr_L = np.tensordot(temp, B.conj(), axes=([0, 1], [0, 1])).T
            L_env.append(curr_L)

        # 2) Build Right Environments
        R_env = [None] * self.L
        curr_R = np.array([[1.0]])
        R_env[-1] = curr_R
        for i in range(self.L - 1, 0, -1):
            B = self._get_std_B(i)
            temp = np.tensordot(B, R_env[i], axes=(2, 1))
            curr_R = np.tensordot(temp, B.conj(), axes=([1, 2], [1, 2])).T
            R_env[i - 1] = curr_R

        # 3) Precompute components
        L_components = []
        for i in range(self.L):
            B = self._get_std_B(i)
            t = np.tensordot(L_env[i], B, axes=(1, 0))
            comp = np.tensordot(t, B.conj(), axes=(0, 0))
            comp = comp.transpose(0, 2, 3, 1)
            L_components.append(comp)

        R_components = []
        for i in range(self.L):
            B = self._get_std_B(i)
            t = np.tensordot(B, R_env[i], axes=(2, 1))
            comp = np.tensordot(t, B.conj(), axes=(2, 2))
            comp = comp.transpose(0, 2, 3, 1)
            R_components.append(comp)

        # 4) Assemble and Extract Scalar
        rdm = {}
        for i in range(self.L):
            js = pairs_by_i.get(i, [])
            if not js: continue

            tensor = L_components[i]
            max_j = max(js)
            for j in range(i + 1, max_j + 1):
                # Propagate transfer matrix for intermediate sites
                if j > i + 1:
                    k = j - 1
                    B = self._get_std_B(k) 
                    tensor = np.einsum('abcd, def, ceh -> abhf', tensor, B, B.conj(), optimize=True)

                if j in js:
                    rho_raw = np.tensordot(tensor, R_components[j], axes=([3, 2], [0, 1]))
                    rho_ij = rho_raw.transpose(0, 3, 1, 2)

                    d_i, d_j = rho_ij.shape[0], rho_ij.shape[1]
                    rho_mat = rho_ij.reshape(d_i * d_j, d_i * d_j)
                    
                    tr = np.trace(rho_mat)
                    if abs(tr) > 1e-12:
                        rho_mat /= tr
                    
                    rdm[(i, j)] = np.real(rho_mat[-1, -1])
        return rdm
    
    def make_rdm1(self, sym_mgr=None):
        """
        Calculate the full global 1-electron reduced density matrix (1-RDM).

        The elements are defined as $\\gamma_{ij} = \\langle \Psi | c_i^\\dagger c_j | \\Psi \\rangle$.
        This method supports both the dense branch (using explicit Jordan-Wigner 
        strings and transfer matrices) and the U(1) symmetric branch (using hole-state 
        overlaps).

        Parameters
        ----------
        sym_mgr : pyqed.mps.symmetry.SymmetryManager, optional
            The symmetry manager containing the physical quantum number definitions. 
            Strictly required if the MPS is utilizing the U(1) symmetric BlockTensor 
            backend. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L)` representing the global 1-RDM, 
            where `L` is the number of sites in the MPS.

        Raises
        ------
        ValueError
            If the MPS uses the U(1) symmetric backend but `sym_mgr` is not provided.
        NotImplementedError
            If the dense branch is called on a system with a local physical dimension 
            other than d=2.
        """
        L = self.L
        
        # 1. Symmetric Branch
        if SYMMETRY_AVAILABLE and hasattr(self.Bs[0], 'qns'):
            if sym_mgr is None:
                raise ValueError("[Error] Symmetric RDM requires sym_mgr.")
            
            d_local = len(sym_mgr.phys_qns)
            norbs_spin = 2 * L if d_local == 4 else L
            P = np.zeros((norbs_spin, norbs_spin), dtype=complex)
            
            # Pre-calculate hole states: |phi_j> = a_j |Psi>
            phis = [None] * norbs_spin
            for spin_idx in range(norbs_spin):
                if d_local == 2:
                    spatial_idx = spin_idx
                    spin = 'up' if spin_idx % 2 == 0 else 'down'
                else:
                    spatial_idx = spin_idx // 2
                    spin = 'up' if spin_idx % 2 == 0 else 'down'
                    
                W_a = build_annihilation_mpo_symmetric(spatial_idx, L, sym_mgr, spin)
                try:
                    phi_data = apply_mpo_symmetric(W_a, self.Bs) 
                    if phi_data:
                        phis[spin_idx] = MPS(phi_data, labels=self.labels, bc=self.bc)
                except Exception:
                    phis[spin_idx] = None

            # Compute Overlaps <phi_i | phi_j>
            for i in range(norbs_spin):
                for j in range(i, norbs_spin):
                    if (i % 2) != (j % 2): continue # Spin conservation
                    if phis[i] is None or phis[j] is None: continue
                    
                    val = self._mps_dot(phis[i], phis[j])
                    P[i, j] = val
                    P[j, i] = val.conjugate()
            return P

        # 2. Dense Branch (Exact O(L^2 D^3) evaluation with JW strings)
        else:
            P = np.zeros((L, L), dtype=complex)
            d = self.Bs[0].shape[1]
            
            if d == 2:
                c_op    = np.array([[0, 1], [0, 0]], dtype=complex)
                cdag_op = np.array([[0, 0], [1, 0]], dtype=complex)
                z_op    = np.array([[1, 0], [0, -1]], dtype=complex)
                n_op    = np.array([[0, 0], [0, 1]], dtype=complex)
            else:
                raise NotImplementedError(f"Dense 1-RDM currently supports d=2 spin-orbitals, got d={d}.")
                
            # A. Build Global Environments (Left and Right)
            L_env = [np.array([[1.0]], dtype=complex)]
            curr_L = L_env[0]
            for i in range(L - 1):
                B = self._get_std_B(i)
                tmp = np.tensordot(curr_L, B, axes=(1, 0)) 
                curr_L = np.tensordot(tmp, B.conj(), axes=([0, 1], [0, 1])).T 
                L_env.append(curr_L)
                
            R_env = [None] * L
            curr_R = np.array([[1.0]], dtype=complex)
            R_env[-1] = curr_R
            for i in range(L - 1, 0, -1):
                B = self._get_std_B(i)
                tmp = np.tensordot(B, curr_R, axes=(2, 1)) 
                curr_R = np.tensordot(tmp, B.conj(), axes=([1, 2], [1, 2])).T 
                R_env[i-1] = curr_R
                
            # B. Compute diagonal and off-diagonal elements
            for i in range(L):
                B_i = self._get_std_B(i)
                
                # 1. Diagonal element: <c_i^\dagger c_i>
                op_ket_n = np.tensordot(n_op, B_i, axes=(1, 1)).transpose(1, 0, 2)
                tmp_n1 = np.tensordot(L_env[i], op_ket_n, axes=(1, 0))
                tmp_n2 = np.tensordot(tmp_n1, B_i.conj(), axes=([0, 1], [0, 1]))
                P[i, i] = np.sum(tmp_n2 * R_env[i].T) # Trace the boundaries
                
                # 2. Off-diagonal elements <c_i^\dagger Z ... Z c_j>
                op_ket_i = np.tensordot(cdag_op, B_i, axes=(1, 1)).transpose(1, 0, 2)
                tmp = np.tensordot(L_env[i], op_ket_i, axes=(1, 0)) 
                T = np.tensordot(tmp, B_i.conj(), axes=([0, 1], [0, 1])).T 
                
                for j in range(i + 1, L):
                    B_j = self._get_std_B(j)
                    op_ket_j = np.tensordot(c_op, B_j, axes=(1, 1)).transpose(1, 0, 2)
                    
                    tmp1 = np.tensordot(T, op_ket_j, axes=(1, 0)) 
                    tmp2 = np.tensordot(tmp1, B_j.conj(), axes=([0, 1], [0, 1])) 
                    val = np.sum(tmp2 * R_env[j].T) 
                    
                    P[i, j] = val
                    P[j, i] = np.conj(val)
                    
                    # Advance JW string (Z)
                    op_ket_z = np.tensordot(z_op, B_j, axes=(1, 1)).transpose(1, 0, 2)
                    tmpz = np.tensordot(T, op_ket_z, axes=(1, 0))
                    T = np.tensordot(tmpz, B_j.conj(), axes=([0, 1], [0, 1])).T
                    
            return P

    def make_rdm2(self, sym_mgr=None):
        """
        Calculate the full global 4-index 2-electron reduced density matrix (2-RDM).

        The elements are defined as $\\Gamma_{pqrs} = \\langle \Psi | c_p^\\dagger c_r^\\dagger c_s c_q | \\Psi \\rangle$.
        This method evaluates the exact overlaps of two-hole states generated by applying 
        annihilation operators (and their corresponding Jordan-Wigner strings) directly 
        to the MPS. Scaling is $\\mathcal{O}(L^4)$.

        Parameters
        ----------
        sym_mgr : pyqed.mps.symmetry.SymmetryManager, optional
            The symmetry manager. Required if the MPS is utilizing the U(1) 
            symmetric BlockTensor backend. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L, L, L)` representing the 
            complete 4-index 2-RDM. Returns an array of zeros if called on a symmetric 
            MPS without providing the `sym_mgr`.
        """
        L = self.L
        G = np.zeros((L, L, L, L), dtype=complex)

        if SYMMETRY_AVAILABLE and hasattr(self.Bs[0], 'qns'):
            if not sym_mgr:
                print("[Warning] Symmetric 2-RDM requires sym_mgr.")
                return G
            vac_qn = sym_mgr.get_vac_qn()

            # 1. Pre-calculate single holes |phi_q> = a_q |Psi>
            phis = [None] * L
            for q in range(L):
                spin = 'up' if q % 2 == 0 else 'down'
                W_q = build_annihilation_mpo_symmetric(q, L, sym_mgr, spin)
                try:
                    d = apply_mpo_symmetric(W_q, self.Bs)
                    if d: phis[q] = MPS(d, labels=self.labels, bc=self.bc)
                except: pass

            # 2. Double loop O(N^4)
            for p in range(L):
                if phis[p] is None: continue
                for r in range(L):
                    # Build |Bra> = a_r |phi_p>
                    spin_r = 'up' if r % 2 == 0 else 'down'
                    W_r = build_annihilation_mpo_symmetric(r, L, sym_mgr, spin_r)
                    try:
                        bra_data = apply_mpo_symmetric(W_r, phis[p].Bs)
                        if not bra_data: continue
                        bra_mps = MPS(bra_data, labels=self.labels, bc=self.bc)
                    except: continue

                    for s in range(L):
                        if phis[s] is None: continue
                        for q in range(L):
                            if phis[q] is None: continue
                            if ((p%2) + (r%2)) != ((s%2) + (q%2)): continue

                            # Build |Ket> = a_s |phi_q>
                            spin_s = 'up' if s % 2 == 0 else 'down'
                            W_s = build_annihilation_mpo_symmetric(s, L, sym_mgr, spin_s)
                            try:
                                ket_data = apply_mpo_symmetric(W_s, phis[q].Bs)
                                if not ket_data: continue
                                ket_mps = MPS(ket_data, labels=self.labels, bc=self.bc)
                            except: continue
                            
                            val = self._mps_dot(bra_mps, ket_mps)
                            G[p, r, s, q] = val
            return G
        else:
            d = self.Bs[0].shape[1]
            if d != 2:
                raise NotImplementedError(f"Dense 2-RDM currently supports d=2 spin-orbitals, got d={d}.")
                
            c_op = np.array([[0, 1], [0, 0]], dtype=complex)
            z_op = np.array([[1, 0], [0, -1]], dtype=complex)
            perm_inv = np.argsort([self.lv_idx, self.p_idx, self.rv_idx])
            
            def apply_annihilation(mps_obj, q):
                """Applies c_q (with JW strings) to return a new MPS."""
                new_Bs = []
                for i in range(L):
                    B_std = mps_obj._get_std_B(i)
                    if i < q:
                        new_B = np.tensordot(z_op, B_std, axes=(1, 1)).transpose(1, 0, 2)
                    elif i == q:
                        new_B = np.tensordot(c_op, B_std, axes=(1, 1)).transpose(1, 0, 2)
                    else:
                        new_B = B_std.copy()
                    # Restore original index order
                    new_Bs.append(new_B.transpose(perm_inv))
                return MPS(new_Bs, labels=mps_obj.labels, bc=mps_obj.bc)

            # Pre-calculate 1-hole states: |phi_q> = c_q |Psi>
            phis = [None] * L
            for q in range(L):
                tmp = apply_annihilation(self, q)
                # Filter out 'dead' states with 0 electrons at site q
                if abs(self._mps_dot(tmp, tmp)) > 1e-14:
                    phis[q] = tmp

            # Double loop O(L^4) for two-hole overlaps
            for p in range(L):
                if phis[p] is None: continue
                for r in range(L):
                    bra_mps = apply_annihilation(phis[p], r)
                    if abs(self._mps_dot(bra_mps, bra_mps)) < 1e-14: continue
                    
                    for s in range(L):
                        for q in range(L):
                            if phis[q] is None: continue
                            # Spin conservation check
                            if ((p%2) + (r%2)) != ((s%2) + (q%2)): continue
                            
                            ket_mps = apply_annihilation(phis[q], s)
                            val = self._mps_dot(bra_mps, ket_mps)
                            G[p, r, s, q] = val
                            
            return G

    def _mps_dot(self, mps1, mps2):
        """
        Calculate the inner product (overlap) between two Matrix Product States.

        Evaluates < mps1 | mps2 >. Handles both dense NumPy tensors 
        and symmetric BlockTensors efficiently by contracting the network from left to right.

        Parameters
        ----------
        mps1 : MPS
            The bra state |mps1>. The tensors of this state will be conjugated.
        mps2 : MPS
            The ket state |mps2>.

        Returns
        -------
        complex
            The scalar inner product evaluated from the complete contraction of the 
            two MPS chains.
        """
        # Symmetric Branch (BlockTensor)
        if SYMMETRY_AVAILABLE and isinstance(mps1.Bs[0], BlockTensor):
            mps1_std = mps1.to_order(['lv', 'rv', 'p'])
            mps2_std = mps2.to_order(['lv', 'rv', 'p'])
            if len(mps1_std.Bs[0].data) == 0 or len(mps2_std.Bs[0].data) == 0:
                return 0.0j
            # E[q_bra_bond, q_ket_bond] = Matrix(dim_bra x dim_ket)
            # Detect Vacuum QN from the first block
            first_key = next(iter(mps1_std.Bs[0].data.keys()))
            vac_qn = first_key[0] # Left Bond QN
            
            # Initialize Environment as 1x1 Identity in Vacuum sector
            # E_blocks maps (QN_Bra, QN_Ket) -> Numpy Array
            E_blocks = { (vac_qn, vac_qn): np.ones((1, 1), dtype=complex) }
            
            for i in range(self.L):
                A = mps1_std.Bs[i] # Bra state (will be conjugated)
                B = mps2_std.Bs[i] # Ket state
                E_next = {}
                
                # Iterate over current Environment sectors
                for (qLa, qLb), mat_E in E_blocks.items():
                    # mat_E shape: (dL_A, dL_B)
                    
                    # 1. Filter Bra Blocks (A) that match Left QN = qLa
                    # A.data keys: (qL, qR, qP)
                    for keyA, blkA in A.data.items():
                        if keyA[0] != qLa: continue
                        qRa, qP = keyA[1], keyA[2]
                        
                        # 2. Filter Ket Blocks (B) that match Left QN = qLb AND Phys QN = qP
                        # B.data keys: (qL, qR, qP)
                        for keyB, blkB in B.data.items():
                            if keyB[0] != qLb or keyB[2] != qP: continue
                            qRb = keyB[1]
                            
                            # Contraction 
                            # A*: (dL_A, dR_A, dP) -> from Bra
                            # B : (dL_B, dR_B, dP) -> from Ket
                            # E : (dL_A, dL_B)
                            
                            # 1: Contract E with A* over Left_Bra (index 0)
                            # T(dL_B, dR_A, dP) = E(dL_A, dL_B) * A*(dL_A, dR_A, dP)
                            # Axes: E[0] with A*[0]
                            T = np.tensordot(mat_E, blkA.conj(), axes=(0, 0))
                            
                            # 2: Contract T with B over Left_Ket (index 0 of B, 0 of T)
                            # and Physical (index 2 of B, 2 of T)
                            # Res(dR_A, dR_B) = T(dL_B, dR_A, dP) * B(dL_B, dR_B, dP)
                            # Axes: T[0, 2] with B[0, 2]
                            block_res = np.tensordot(T, blkB, axes=([0, 2], [0, 2]))
                            
                            # Accumulate into next Environment
                            next_key = (qRa, qRb)
                            if next_key in E_next:
                                E_next[next_key] += block_res
                            else:
                                E_next[next_key] = block_res
                                
                E_blocks = E_next
                
            # Final Result: Trace/Sum of the last environment block(s)
            # For a proper overlap <Psi|Psi>, this should be a scalar 1.0
            total = sum(np.sum(blk) for blk in E_blocks.values())
            return total

        # Dense Branch
        else:
            mps1_std = mps1.to_order(['lv', 'p', 'rv'])
            mps2_std = mps2.to_order(['lv', 'p', 'rv'])
            val = np.array([[1.0]], dtype=complex)
            for i in range(self.L):
                A = mps1_std.Bs[i] # (Left, Phys, Right)
                B = mps2_std.Bs[i] 
                
                # E(la, lb) * A*(la, p, ra) -> T(lb, p, ra)
                T = np.tensordot(val, A.conj(), axes=(0, 0))
                # T(lb, p, ra) * B(lb, p, rb) -> Next_E(ra, rb)
                val = np.tensordot(T, B, axes=([0, 1], [0, 1]))
                
            return val.flatten()[0]

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
    def __init__(self, factors, target_qn=None, labels=['left', 'right', 'up', 'down'], homogenous=False):
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
        # TODO: label treatment
        #if self.labels :
            #error (if not four terms, not including correct name type...)
        #if self.labels not ['left', 'right', 'up', 'down']
            #swap to not ['left', 'right', 'up', 'down']
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

        # if self.labels :  # TODO: add label treatment to label (actual we want to do it in initilization stage)

        if isinstance(other, MPO):
            if chi_max is None:
                # Preserve the exact MPO product by default. Compressing to
                # `max(self.bond_orders()+other.bond_orders())` is generally too
                # aggressive for operator products and breaks routines such as
                # expmpo(..., D=None), which expect an untruncated Taylor build.
                return self.__matmul__(other)

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
            if chi_max is None:
                chi_max = max(self.bond_orders()) * 2
            new_factors = apply_mpo(self.factors, other.factors, chi_max)
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
            factors_new = [W.copy() for W in self.factors]
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

    def exponential(self, constant=1.0, D=None, method='taylor', order=4, scale=0):
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

        return expmpo(self.H, constant, method='taylor', order=4, scale=0)

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


def _mpo_to_dense_operator(mpo):
    """Contract a small MPO into a full dense operator matrix."""
    cores = [np.asarray(core).transpose(0, 2, 3, 1) for core in mpo.factors]
    tensor = cores[0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    tensor = np.squeeze(tensor, axis=(0, -1))
    nsites = len(cores)
    perm = list(range(0, 2 * nsites, 2)) + list(range(1, 2 * nsites, 2))
    tensor = np.transpose(tensor, axes=perm)
    dim = int(np.prod(mpo.dims))
    return tensor.reshape((dim, dim))


def _dense_operator_to_mpo(matrix, dims):
    """Factor a dense operator into an MPO exactly on small Hilbert spaces."""
    matrix = np.asarray(matrix, dtype=complex)
    tensor = matrix.reshape(tuple(dims) + tuple(dims))
    tt = tensor_train_matrix(tensor, rank=matrix.shape[0])
    return MPO([np.asarray(core).transpose(0, 3, 1, 2) for core in tt.factors])


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

    # On small Hilbert spaces, avoid MPO Taylor/compression entirely and build
    # the exact dense exponential. This provides a reliable oracle path for
    # regression tests and avoids uncontrolled bond growth when D is None.
    dense_dim = int(np.prod(H.dims))
    if D is None and dense_dim <= 256:
        dense_h = _mpo_to_dense_operator(H)
        return _dense_operator_to_mpo(expm(constant * dense_h), H.dims)

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

def apply_mpo(w_list, B_list, chi_max):
    """
    Apply the MPO to an MPS.

    MPS index order: [chi_L, d, chi_R] = [Left, Phys, Right]
    MPO index order: [b_L, b_R, d_out, d_in] = [Left, Right, Out, In]

    Parameters
    ----------
    w_list : list
        MPO tensors, each with shape [chi1, chi2, d_up, d_down].
    B_list : list
        MPS tensors, each with shape [chi_L, d, chi_R].
    chi_max : int
        Maximum bond dimension for compression.

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
        # B: [chi_L, d_in, chi_R]
        # W: [b_L, b_R, d_out, d_in]
        chi_L, d_in_mps, chi_R = result[i_site].shape
        b_L, b_R, d_out, d_in_mpo = w_list_copy[i_site].shape
        
        # Contract W's In (j) with B's Phys (j)
        # W: abij (a=b_L, b=b_R, i=d_out, j=d_in)
        # B: kjl (k=chi_L, j=d_in, l=chi_R)
        # einsum: 'abij,kjl->akilb' -> (b_L, chi_L, d_out, chi_R, b_R)
        B_new = np.einsum('abij,kjl->akilb', w_list_copy[i_site], result[i_site])
        
        # Reshape to strictly [Left, Phys, Right] for the compress function
        B_new = np.reshape(B_new, (b_L * chi_L, d_out, chi_R * b_R))
        result[i_site] = B_new
    
    # compress strictly expects [Left, Phys, Right]
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
        zero_qn = zero_like_sector(sample_qn)

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

        zero_qn = zero_like_sector(sample_qn)

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
    Convert a product-state dense MPS guess into a true U(1) BlockTensor MPS.
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

def symmetric_to_dense(mps_obj):
    """
    Converts a U(1) symmetric BlockTensor MPS back to a standard dense NumPy MPS.
    """
    import collections
    from pyqed.mps.mps import MPS
    
    dense_factors = []
    for bt in mps_obj.factors:
        # If it's already dense, return it safely in standard layout
        if not hasattr(bt, 'qns'):
            if mps_obj.labels != ['lv', 'p', 'rv']:
                return mps_obj.to_order(['lv', 'p', 'rv'])
            return mps_obj
            
        # Map QNs to absolute array indices
        maps = []
        for qlist in bt.qns:
            m = collections.defaultdict(list)
            for i, q in enumerate(qlist): 
                m[q].append(i)
            maps.append(m)
            
        # Allocate dense block and fill with symmetry sectors
        shape = tuple(len(q) for q in bt.qns)
        out = np.zeros(shape, dtype=complex)
        for qkey, block in bt.data.items():
            idx_lists = [maps[leg][qkey[leg]] for leg in range(bt.rank)]
            out[np.ix_(*idx_lists)] += block
            
        # BlockTensors in this code are (Left, Right, Phys). 
        # Transpose to standard (Left, Phys, Right)
        out_std = out.transpose(0, 2, 1)
        dense_factors.append(out_std)
        
    return MPS(dense_factors, labels=['lv', 'p', 'rv'])

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

def sa_svd_dense(AA_list, weights, direction, m_max=None):
    """
    State-averaged dense two-site SVD.

    ``AA_list`` contains two-site wavefunctions with shape
    ``(left, phys_left, phys_right, right)``.  The retained basis is obtained
    from the weighted reduced density matrix, while state 0 is projected into
    that basis to keep propagating a single representative MPS.
    """
    weights = np.asarray(weights, dtype=float)
    if len(AA_list) != len(weights):
        raise ValueError("weights must have the same length as AA_list.")
    if np.sum(weights) <= 0:
        raise ValueError("state-average weights must sum to a positive value.")
    weights = weights / np.sum(weights)

    chi_l, d_l, d_r, chi_r = AA_list[0].shape
    mats = [AA.reshape(chi_l * d_l, d_r * chi_r) for AA in AA_list]

    if direction == 'right':
        rho = np.zeros((chi_l * d_l, chi_l * d_l), dtype=np.result_type(*mats, np.complex128))
        for w, mat in zip(weights, mats):
            rho += w * (mat @ mat.conj().T)
        evals, U = np.linalg.eigh(rho)
        idx = np.argsort(evals)[::-1]
        evals, U = evals[idx], U[:, idx]
        all_evals = evals
        nkeep = len(evals) if m_max is None else min(int(m_max), len(evals))
        evals, U = evals[:nkeep], U[:, :nkeep]
        S = np.sqrt(np.clip(evals, 0.0, None))
        Sinv = np.zeros_like(S)
        Sinv[S > 1e-12] = 1.0 / S[S > 1e-12]
        V = (np.diag(Sinv) @ U.conj().T @ mats[0]).reshape(nkeep, d_r, chi_r)
        A = U.reshape(chi_l, d_l, nkeep)
        trunc = float(np.sum(np.clip(all_evals[nkeep:], 0.0, None)))
    else:
        rho = np.zeros((d_r * chi_r, d_r * chi_r), dtype=np.result_type(*mats, np.complex128))
        for w, mat in zip(weights, mats):
            rho += w * (mat.conj().T @ mat)
        evals, Vcols = np.linalg.eigh(rho)
        idx = np.argsort(evals)[::-1]
        evals, Vcols = evals[idx], Vcols[:, idx]
        all_evals = evals
        nkeep = len(evals) if m_max is None else min(int(m_max), len(evals))
        evals, Vcols = evals[:nkeep], Vcols[:, :nkeep]
        S = np.sqrt(np.clip(evals, 0.0, None))
        Sinv = np.zeros_like(S)
        Sinv[S > 1e-12] = 1.0 / S[S > 1e-12]
        Umat = mats[0] @ Vcols @ np.diag(Sinv)
        A = Umat.reshape(chi_l, d_l, nkeep)
        V = Vcols.conj().T.reshape(nkeep, d_r, chi_r)
        trunc = float(np.sum(np.clip(all_evals[nkeep:], 0.0, None)))

    return A, S, V, trunc, nkeep

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
        self.dtype = np.result_type(E, W, F, np.complex128)

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
    """
    if not hasattr(AA, 'qns'):
        return AA
    valid_qL = {}
    valid_qR = {}

    first_blk = next(iter(AA.data.values()))
    is_complex = np.iscomplexobj(first_blk)
    dtype = first_blk.dtype

    for (qL, qR, qP1, qP2), blk in AA.data.items():
        valid_qL[qL] = blk.shape[0]
        valid_qR[qR] = blk.shape[1]

    # Get valid physical QNs directly from the manager (works for d=2 and d=4)
    possible_phys_qns = sym_mgr.phys_qns

    # Iterate and Inject
    for qL, dL in valid_qL.items():
        for qP1 in possible_phys_qns:
            for qP2 in possible_phys_qns:
                target_qR = qL + qP1 + qP2
                if target_qR in valid_qR:
                    dR = valid_qR[target_qR]
                    key = (qL, target_qR, qP1, qP2)
                    dP1, dP2 = 1, 1
                    if key not in AA.data:
                        noise = (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        if is_complex:
                            noise = noise + 1j * (np.random.rand(dL, dR, dP1, dP2) - 0.5) * noise_val
                        AA.data[key] = noise.astype(dtype)
    return AA


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

def sa_svd_symmetric(AA_list, weights, dir, m_max=None):
    """State-Averaged SVD for U(1) BlockTensors."""
    AA_perm_0 = AA_list[0].transpose(0, 2, 1, 3)
    blocks_by_q_mid = {}
    row_map, col_map = {}, {}
    M_dict = {k: {} for k in range(len(AA_list))}
    
    for k, AA in enumerate(AA_list):
        AA_perm = AA.transpose(0, 2, 1, 3)
        for qn_tuple, block in AA_perm.data.items():
            q_L, q_phys_L, q_R, q_phys_R = qn_tuple
            q_mid = q_L + q_phys_L
            if q_mid not in blocks_by_q_mid:
                blocks_by_q_mid[q_mid] = []
                row_map[q_mid] = set()
                col_map[q_mid] = set()
            if k == 0:
                blocks_by_q_mid[q_mid].append(qn_tuple)
                row_map[q_mid].add((q_L, q_phys_L))
                col_map[q_mid].add((q_R, q_phys_R))
            M_dict[k].setdefault(q_mid, []).append((qn_tuple, block))
            
    sv_list = []
    U_store, V_store, S_store = {}, {}, {}
    
    for q_mid in blocks_by_q_mid.keys():
        rows, cols = sorted(row_map[q_mid]), sorted(col_map[q_mid])
        r_starts, c_starts = {}, {}
        r_dim = c_dim = 0
        entries_0 = M_dict[0][q_mid]
        
        for r in rows:
            for qn, blk in entries_0:
                if (qn[0], qn[1]) == r:
                    r_starts[r] = r_dim; r_dim += blk.shape[0] * blk.shape[1]; break
        for c in cols:
            for qn, blk in entries_0:
                if (qn[2], qn[3]) == c:
                    c_starts[c] = c_dim; c_dim += blk.shape[2] * blk.shape[3]; break
                    
        M_matrices = []
        for k in range(len(AA_list)):
            M = np.zeros((r_dim, c_dim), dtype=entries_0[0][1].dtype)
            for qn, blk in M_dict[k][q_mid]:
                r0, c0 = r_starts[(qn[0], qn[1])], c_starts[(qn[2], qn[3])]
                M[r0:r0+blk.shape[0]*blk.shape[1], c0:c0+blk.shape[2]*blk.shape[3]] = blk.reshape(blk.shape[0]*blk.shape[1], blk.shape[2]*blk.shape[3])
            M_matrices.append(M)
            
        if dir == 'right':
            rho = np.zeros((r_dim, r_dim), dtype=M.dtype)
            for k in range(len(AA_list)): rho += weights[k] * (M_matrices[k] @ M_matrices[k].conj().T)
            S2, U = np.linalg.eigh(rho)
            idx = np.argsort(S2)[::-1]; S2, U = S2[idx], U[:, idx]
            S = np.sqrt(np.abs(S2))
            S_inv = np.zeros_like(S); S_inv[S > 1e-12] = 1.0 / S[S > 1e-12]
            Vt = np.diag(S_inv) @ U.conj().T @ M_matrices[0] # Project State 0 to propagate
        else:
            rho = np.zeros((c_dim, c_dim), dtype=M.dtype)
            for k in range(len(AA_list)): rho += weights[k] * (M_matrices[k].conj().T @ M_matrices[k])
            S2, V = np.linalg.eigh(rho)
            idx = np.argsort(S2)[::-1]; S2, V = S2[idx], V[:, idx]
            S = np.sqrt(np.abs(S2))
            Vt = V.conj().T
            S_inv = np.zeros_like(S); S_inv[S > 1e-12] = 1.0 / S[S > 1e-12]
            U = M_matrices[0] @ V @ np.diag(S_inv) # Project State 0 to propagate
            
        S_store[q_mid] = S
        for i, s in enumerate(S): sv_list.append((s, q_mid, i))
        U_store[q_mid] = (U, rows, r_starts, entries_0)
        V_store[q_mid] = (Vt, cols, c_starts, entries_0)
        
    sv_list.sort(reverse=True, key=lambda x: x[0])
    if m_max is not None: sv_list = sv_list[:m_max]
    kept = {}
    for s, q_mid, i in sv_list: kept.setdefault(q_mid, []).append(i)
        
    final_U, final_V, final_S, bond_qns = {}, {}, {}, []
    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
        U, rows, r_starts, entries_0 = U_store[q_mid]
        Vt, cols, c_starts, entries_0 = V_store[q_mid]
        final_S[q_mid] = np.diag(S_store[q_mid][idxs])
        bond_qns.extend([q_mid] * len(idxs))
        
        for r in rows:
            for qn, blk in entries_0:
                if (qn[0], qn[1]) == r: d1, d2 = blk.shape[0], blk.shape[1]; break
            r0 = r_starts[r]
            final_U[(r[0], r[1], q_mid)] = U[r0:r0+d1*d2, idxs].reshape(d1, d2, len(idxs))
        for c in cols:
            for qn, blk in entries_0:
                if (qn[2], qn[3]) == c: d3, d4 = blk.shape[2], blk.shape[3]; break
            c0 = c_starts[c]
            final_V[(q_mid, c[0], c[1])] = Vt[idxs, c0:c0+d3*d4].reshape(len(idxs), d3, d4)
            
    U_t = BlockTensor(final_U, [AA_perm_0.qns[0], AA_perm_0.qns[1], bond_qns], [AA_perm_0.dirs[0], AA_perm_0.dirs[1], 1])
    V_t = BlockTensor(final_V, [bond_qns, AA_perm_0.qns[2], AA_perm_0.qns[3]], [-1, AA_perm_0.dirs[2], AA_perm_0.dirs[3]])
    return U_t, V_t, final_S, 0.0, sum(len(v) for v in kept.values())

def compatible_blocktensor_structure(x, y):
    if x.rank != y.rank:
        return False
    if x.dirs != y.dirs:
        return False
    if len(x.qns) != len(y.qns):
        return False
    for qx, qy in zip(x.qns, y.qns):
        if qx != qy:
            return False
    for k, bx in x.data.items():
        if k in y.data and bx.shape != y.data[k].shape:
            return False
    return True

def optimize_two_sites(
    A,
    B,
    W1,
    W2,
    E,
    F,
    m,
    dir,
    U1=False,
    sym_mgr=None,
    nstates=1,
    weights=None,
    init_vecs=None,
    davidson_tol=1e-5,
    davidson_max_iter=30,
    noise=1e-4,
    local_dense_max_dim=0,
    complementary_operator_families=None,
    bond=None,
    complementary_boundary_payloads=None,
    complementary_split_stats=None,
    complementary_family_environments=None,
    complementary_direct_family_environments=None,
):
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
    if weights is None: 
        weights = [1.0/nstates] * nstates
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

            if noise is not None and float(noise) > 0.0:
                AA = inject_noise_symmetric(AA, noise_val=float(noise), sym_mgr=sym_mgr)
        else:
            raise ValueError(f"Unexpected tensor rank {A.rank} in symmetric opt")
        H_op = HamiltonianMultiplyU1(
            E,
            [W1, W2],
            F,
            complementary_operator_families=complementary_operator_families,
            bond=bond,
            complementary_boundary_payloads=complementary_boundary_payloads,
            complementary_split_stats=complementary_split_stats,
            complementary_family_environments=complementary_family_environments,
            complementary_direct_family_environments=complementary_direct_family_environments,
        )
        norm = AA.norm()
        AA = AA * (1.0/norm)

        if nstates == 1:
            H_dense, dense_layout = H_op.dense_matrix(AA, max_dim=local_dense_max_dim)
            if H_dense is not None:
                evals, evecs = np.linalg.eigh(H_dense)
                energy = evals[0]
                AA_new = H_op._unflatten(evecs[:, 0], AA, dense_layout)
            else:
                energy, AA_new = solve_davidson(
                    H_op,
                    AA,
                    n_eig=1,
                    tol=float(davidson_tol),
                    max_iter=int(davidson_max_iter),
                )
            # AA_new is (L, R, Phys_L, Phys_R)
            # We need to return A(L, M, P_L) and B(M, R, P_R)
            U, V, S_dict, trunc, m_kept = svd_symmetric(AA_new, m_max=m)
            # U is (L, P_L, M).
            # V is (M, R, P_R).
            # We need to contract S_dict into either U or V depending on 'dir'. (left or right sweep)

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
        else: # state average dmrg
            guess_list = [AA]
            if init_vecs is not None:
                valid = [g for g in init_vecs if compatible_blocktensor_structure(g, AA)]
                if len(valid) > 0:
                    guess_list = valid[:nstates]

            # Add tiny random companions if we need more than one state
            # and no previous local state guesses are being passed in yet.
            rng = np.random.default_rng(1234)
            for _ in range(1, nstates):
                data = {}
                for k, blk in AA.data.items():
                    noise = rng.standard_normal(blk.shape)
                    if np.iscomplexobj(blk):
                        noise = noise + 1j * rng.standard_normal(blk.shape)
                    data[k] = noise.astype(blk.dtype, copy=False)
                guess = BlockTensor(data, AA.qns[:], AA.dirs[:])

                # Bias toward the current AA sector structure a bit
                guess = AA * 1e-3 + guess
                guess_list.append(guess)

            energies, AA_new_list = solve_davidson_block(
                H_op,
                guess_list,
                n_eig=nstates,
                tol=float(davidson_tol),
                max_iter=int(davidson_max_iter),
                max_subspace=max(8, 4 * nstates),
            )

            U, V, S_dict, trunc, m_kept = sa_svd_symmetric(AA_new_list, weights, dir=dir, m_max=m)
            if dir == 'right':
                A_new, B_new = U.transpose(0, 2, 1), multiply_S_V(S_dict, V)
            else:
                A_new, B_new = multiply_U_S(U, S_dict).transpose(0, 2, 1), V
            return energies, A_new, B_new, trunc, m_kept, AA_new_list
    else: # Dense branch ( MPS index standardized to Left, Phys, Right)
        W = coarse_grain_MPO(W1,W2)
        # Returns (Left, Phys_A, Phys_B, Right)
        AA = coarse_grain_MPS(A,B)
        # Optimize
        H = HamiltonianMultiply(E,W,F)
        nloc = AA.size
        if nstates >= nloc:
            use_dense_solver = True
        else:
            use_dense_solver = False
        try:
            if use_dense_solver:
                raise ValueError("dense fallback requested")
            E, V_flat = sparse.linalg.eigsh(
                H, nstates, v0=AA, which='SA', tol=1e-9, maxiter=5000
            )
        except (sparse.linalg.ArpackNoConvergence, ValueError):
            # Robust fallback for small local spaces when ARPACK stalls.
            if nloc > 4096:
                raise
            H_dense = np.zeros((nloc, nloc), dtype=np.result_type(AA.dtype, np.complex128))
            for col in range(nloc):
                e_col = np.zeros(nloc, dtype=AA.dtype)
                e_col[col] = 1.0
                H_dense[:, col] = H.matvec(e_col)
            H_dense = 0.5 * (H_dense + H_dense.T.conj())
            evals, evecs = np.linalg.eigh(H_dense)
            E = evals[:nstates]
            V_flat = evecs[:, :nstates]

        order = np.argsort(E)
        E = np.asarray(E)[order]
        V_flat = np.asarray(V_flat)
        if V_flat.ndim == 1:
            V_flat = V_flat[:, np.newaxis]
        V_flat = V_flat[:, order]

        # Fine Grain (SVD Split).  V_flat columns are
        # (Left * Phys_A * Phys_B * Right) two-site wavefunctions.
        AA_list = [
            V_flat[:, root].reshape(A.shape[0], A.shape[1], B.shape[1], B.shape[2])
            for root in range(nstates)
        ]
        if nstates == 1:
            A,S,B = fine_grain_MPS(AA_list[0], [A.shape[1], B.shape[1]])
            A,S,B,trunc,m = truncate_SVD(A,S,B,m)
        else:
            A,S,B,trunc,m = sa_svd_dense(AA_list, weights, dir, m_max=m)
        if (dir == 'right'):
            # B = S * B.  S is (m,), B is (m, d, R).
            # Contract S with B[0] (Left bond of B)
            B = np.tensordot(np.diag(S), B, axes=(1, 0))
        else:
            assert dir == 'left'
            # A = A * S.  A is (L, d, m), S is (m,)
            # Contract A[2] (Right bond) with S
            A = np.tensordot(A, np.diag(S), axes=(2, 0))
        if nstates == 1:
            return E[0], A, B, trunc, m
        return E, A, B, trunc, m, AA_list

def two_site_dmrg(
    mps,
    mpo,
    m,
    sweeps=50,
    conv=1e-6,
    U1=False,
    target_qn=None,
    not_conv_err=True,
    sym_mgr=None,
    nstates=1,
    weights=None,
    verbose=0,
    sweep_callback=None,
    davidson_tol=1e-5,
    davidson_max_iter=30,
    noise=1e-4,
    noise_decay=0.1,
    noise_cutoff=1e-9,
    local_dense_max_dim=0,
    complementary_operator_families=None,
    complementary_operator_mpos=None,
    complementary_operator_term_maps=None,
    complementary_operator_generator_entries=None,
    site_qn_maps=None,
):
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
    verbose = int(verbose)
    if weights is None:
        weights = [1.0/nstates] * nstates
    weights = np.array(weights)
    MPS = mps 
    MPO = mpo 
    
    E = construct_E(MPS, MPO, MPS)
    F = construct_F(MPS, MPO, MPS, target_qn=target_qn)
    F.pop()
    complementary_operator_mpos = {
        str(name): factors
        for name, factors in (complementary_operator_mpos or {}).items()
    }
    complementary_operator_term_maps = {
        str(name): dict(term_map)
        for name, term_map in (complementary_operator_term_maps or {}).items()
    }
    complementary_operator_generator_entries = {
        str(name): dict(entries)
        for name, entries in (complementary_operator_generator_entries or {}).items()
    }
    comp_family_E = {
        name: construct_E(MPS, factors, MPS)
        for name, factors in complementary_operator_mpos.items()
    }
    comp_family_F = {
        name: construct_F(MPS, factors, MPS, target_qn=target_qn)
        for name, factors in complementary_operator_mpos.items()
    }
    for stack in comp_family_F.values():
        stack.pop()

    # Skip dense expectation check for U1 to avoid crash
    Eold = 0.0
    converged = False
    gauge = None

    def _noise(half_sweep):
        nz = float(noise or 0.0)
        if nz <= 0.0:
            return 0.0
        nz *= float(noise_decay) ** int(half_sweep)
        return 0.0 if nz < float(noise_cutoff) else nz

    comp_stack, comp_payload_map = _make_complementary_boundary_stack(
        complementary_operator_families,
        len(MPS),
    )
    comp_split_stats = {
        "enabled": comp_stack is not None,
        "calls": 0,
        "modes": {},
        "bonds": {},
    }

    def _comp_payload(bond):
        if comp_stack is None:
            return None
        bond = int(bond)
        return {
            "stack": comp_stack,
            "left": comp_payload_map.get(("left", bond)),
            "right": comp_payload_map.get(("right", bond + 1)),
        }

    def _comp_split_snapshot():
        if comp_stack is None:
            return None
        return deepcopy(comp_split_stats)

    def _comp_family_env(bond):
        bond = int(bond)
        envs = {}
        for name, factors in complementary_operator_mpos.items():
            try:
                envs[name] = (
                    comp_family_E[name][-1],
                    [factors[bond], factors[bond + 1]],
                    comp_family_F[name][-1],
                )
            except (KeyError, IndexError):
                continue
        return envs or None

    def _one_site_operator_mpo(symbol):
        from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

        ops = spatial_local_ops()
        return np.asarray(ops[str(symbol)], dtype=complex).reshape(1, 1, 4, 4)

    generator_expansion_cache = {}
    generator_term_map_cache = None
    direct_family_sym_pattern_cache = {}
    direct_family_left_env_cache = {}
    direct_family_right_env_cache = {}
    direct_family_env_revision = [0]
    direct_family_pattern_terms_cache = None

    def _invalidate_direct_family_env_cache():
        direct_family_env_revision[0] += 1
        direct_family_left_env_cache.clear()
        direct_family_right_env_cache.clear()

    def _generator_expansion(p, q):
        key = ("E", int(p), int(q))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        from pyqed.qchem.dmrg.spatial_terms import spatial_jw_term_spec

        terms = []
        for create, destroy in (("cdu", "cu"), ("cdd", "cd")):
            symbol, dofs, factor = spatial_jw_term_spec(
                [create, destroy],
                [int(p), int(q)],
                1.0,
            )
            if symbol and abs(factor) > 1.0e-14:
                terms.append((symbol, tuple(dofs), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _two_generator_expansion(p, q, r, s):
        key = ("EE", int(p), int(q), int(r), int(s))
        cached = generator_expansion_cache.get(key)
        if cached is not None:
            return cached
        from pyqed.qchem.dmrg.spatial_terms import spatial_jw_term_spec

        terms = []
        for left_create, left_destroy in (("cdu", "cu"), ("cdd", "cd")):
            for right_create, right_destroy in (("cdu", "cu"), ("cdd", "cd")):
                symbol, dofs, factor = spatial_jw_term_spec(
                    [left_create, left_destroy, right_create, right_destroy],
                    [int(p), int(q), int(r), int(s)],
                    1.0,
                )
                if symbol and abs(factor) > 1.0e-14:
                    terms.append((symbol, tuple(dofs), complex(factor)))
        generator_expansion_cache[key] = tuple(terms)
        return generator_expansion_cache[key]

    def _iter_direct_family_terms():
        if complementary_operator_term_maps:
            for family_name, term_map in complementary_operator_term_maps.items():
                for (symbol, dofs), coeff in term_map.items():
                    yield str(family_name), str(symbol), tuple(dofs), complex(coeff)
        if complementary_operator_generator_entries:
            nonlocal generator_term_map_cache
            if generator_term_map_cache is None:
                from pyqed.qchem.dmrg.spatial_terms import accumulate_symbolic_term

                generator_term_map_cache = {"R": {}, "P": {}}
                for (p, q), coeff in complementary_operator_generator_entries.get("R", {}).items():
                    for symbol, dofs, factor in _generator_expansion(p, q):
                        accumulate_symbolic_term(
                            generator_term_map_cache["R"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
                for (p, q, r, s), coeff in complementary_operator_generator_entries.get("P", {}).items():
                    for symbol, dofs, factor in _two_generator_expansion(p, q, r, s):
                        accumulate_symbolic_term(
                            generator_term_map_cache["P"],
                            symbol,
                            dofs,
                            complex(coeff) * factor,
                        )
            for family_name, term_map in generator_term_map_cache.items():
                for (symbol, dofs), coeff in term_map.items():
                    yield str(family_name), str(symbol), tuple(dofs), complex(coeff)

    def _direct_family_pattern_terms():
        nonlocal direct_family_pattern_terms_cache
        if direct_family_pattern_terms_cache is not None:
            return direct_family_pattern_terms_cache
        L = len(MPS)
        grouped = {}
        for family_name, symbol, dofs, coeff in _iter_direct_family_terms():
            per_site = ["I"] * L
            for piece, site in zip(str(symbol).split(), dofs):
                per_site[int(site)] = str(piece)
            grouped.setdefault(str(family_name), []).append(
                (tuple(per_site), complex(coeff))
            )
        direct_family_pattern_terms_cache = {
            name: tuple(terms)
            for name, terms in grouped.items()
        }
        return direct_family_pattern_terms_cache

    def _direct_family_env(bond):
        if not complementary_operator_term_maps and not complementary_operator_generator_entries:
            return None
        if site_qn_maps is None:
            return None
        bond = int(bond)
        out = {}
        L = len(MPS)

        def _sym_mpo_for_pattern(pattern):
            cached = direct_family_sym_pattern_cache.get(pattern)
            if cached is not None:
                return cached
            dense_pattern_mpo = [
                _one_site_operator_mpo(piece).copy()
                for piece in pattern
            ]
            sym_pattern_mpo = dense_to_symmetric_mpo(
                dense_pattern_mpo,
                site_qn_maps,
            )
            direct_family_sym_pattern_cache[pattern] = sym_pattern_mpo
            return sym_pattern_mpo

        def _left_env(pattern, sym_pattern_mpo):
            key = (direct_family_env_revision[0], bond, tuple(pattern[:bond]))
            cached = direct_family_left_env_cache.get(key)
            if cached is not None:
                return cached
            if bond == 0:
                env = initial_E(sym_pattern_mpo[0])
            else:
                env = initial_E(sym_pattern_mpo[0])
                for site in range(bond):
                    env = contract_from_left(
                        sym_pattern_mpo[site],
                        MPS[site],
                        env,
                        MPS[site],
                    )
            direct_family_left_env_cache[key] = env
            return env

        def _right_env(pattern, sym_pattern_mpo):
            key = (direct_family_env_revision[0], bond, tuple(pattern[bond + 2:]))
            cached = direct_family_right_env_cache.get(key)
            if cached is not None:
                return cached
            if bond + 2 >= L:
                env = initial_F(
                    sym_pattern_mpo[bond + 1],
                    target_qn=target_qn if target_qn is not None else 0,
                )
            else:
                env = initial_F(
                    sym_pattern_mpo[-1],
                    target_qn=target_qn if target_qn is not None else 0,
                )
                for site in range(L - 1, bond + 1, -1):
                    env = contract_from_right(
                        sym_pattern_mpo[site],
                        MPS[site],
                        env,
                        MPS[site],
                )
            direct_family_right_env_cache[key] = env
            return env

        for family_name, terms in _direct_family_pattern_terms().items():
            entries = []
            for pattern, coeff in terms:
                try:
                    sym_term_mpo = _sym_mpo_for_pattern(pattern)
                    E_term = _left_env(pattern, sym_term_mpo)
                    F_term = _right_env(pattern, sym_term_mpo)
                except Exception:
                    entries = None
                    break
                entries.append((
                    E_term,
                    [sym_term_mpo[bond] * complex(coeff), sym_term_mpo[bond + 1]],
                    F_term,
                ))
            if entries:
                out[str(family_name)] = tuple(entries)
        return out or None

    if len(MPS) == 2:
        if nstates > 1:
            Energy, MPS[0], MPS[1], trunc, states, last_AA_list = optimize_two_sites(
                MPS[0], MPS[1], MPO[0], MPO[1], E[-1], F[-1], m, 'right',
                U1=U1, sym_mgr=sym_mgr, nstates=nstates, weights=weights,
                davidson_tol=davidson_tol,
                davidson_max_iter=davidson_max_iter,
                noise=_noise(0),
                local_dense_max_dim=local_dense_max_dim,
                complementary_operator_families=complementary_operator_families,
                bond=0,
                complementary_boundary_payloads=_comp_payload(0),
                complementary_split_stats=comp_split_stats,
                complementary_family_environments=_comp_family_env(0),
                complementary_direct_family_environments=_direct_family_env(0),
            )
            _invalidate_direct_family_env_cache()
            final_states = []
            for k in range(nstates):
                MPS_k = [B.copy() for B in MPS]
                if U1:
                    U, V, S_dict, _, _ = svd_symmetric(last_AA_list[k], m_max=None)
                    A_US = multiply_U_S(U, S_dict)
                    MPS_k[0] = A_US.transpose(0, 2, 1)
                    MPS_k[1] = V
                else:
                    A_root, S_root, B_root = fine_grain_MPS(
                        last_AA_list[k], [MPS[0].shape[1], MPS[1].shape[1]]
                    )
                    A_root, S_root, B_root, _, _ = truncate_SVD(
                        A_root, S_root, B_root, m
                    )
                    MPS_k[0] = np.tensordot(A_root, np.diag(S_root), axes=(2, 0))
                    MPS_k[1] = B_root
                final_states.append(MPS_k)
            return Energy, final_states, "Right", True
        else:
            Energy, MPS[0], MPS[1], trunc, states = optimize_two_sites(
                MPS[0], MPS[1], MPO[0], MPO[1], E[-1], F[-1], m, 'right',
                U1=U1, sym_mgr=sym_mgr,
                davidson_tol=davidson_tol,
                davidson_max_iter=davidson_max_iter,
                noise=_noise(0),
                local_dense_max_dim=local_dense_max_dim,
                complementary_operator_families=complementary_operator_families,
                bond=0,
                complementary_boundary_payloads=_comp_payload(0),
                complementary_split_stats=comp_split_stats,
                complementary_family_environments=_comp_family_env(0),
                complementary_direct_family_environments=_direct_family_env(0),
            )
            _invalidate_direct_family_env_cache()
        return Energy, MPS, "Right", True
    
    last_i = 0
    last_AA_list = None
    bond_guess_cache = {}
    def _notify_sweep(sweep_index, direction, energy, truncation, states_kept):
        if sweep_callback is None:
            return
        sweep_callback(
            sweep=sweep_index,
            direction=direction,
            energy=energy,
            truncation=truncation,
            states_kept=states_kept,
            mps=MPS,
            last_i=last_i,
            last_AA_list=last_AA_list,
            gauge="Left" if direction == "lr" else "Right",
            complementary_operator_stack=(
                None if comp_stack is None else comp_stack.stats
            ),
            complementary_split_stats=_comp_split_snapshot(),
        )

    for sweep in range(0, int(sweeps/2)):
        nz = _noise(sweep * 2)
        for i in range(0, len(MPS)-2):
            if nstates > 1:
                # init_vecs = bond_guess_cache.get(i, None)
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1, sym_mgr, nstates, weights,
                    init_vecs=last_AA_list, davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
                E_ground_state = Energy[0]
                bond_guess_cache[i] = last_AA_list
            else:
                Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1=U1, sym_mgr=sym_mgr,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
                E_ground_state = Energy
            logging.info("Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}".format(sweep*2,i,i+1, E_ground_state, states, trunc))

            E.append(contract_from_left(MPO[i], MPS[i], E[-1], MPS[i]))
            for name, factors in complementary_operator_mpos.items():
                comp_family_E[name].append(
                    contract_from_left(factors[i], MPS[i], comp_family_E[name][-1], MPS[i])
                )
            F.pop()
            for stack in comp_family_F.values():
                stack.pop()
            last_i = i

        if nstates > 1:
            if verbose >= 1:
                print(Energy)
            e_avg = np.sum(weights * Energy) 
        else:
            e_avg = Energy
        _notify_sweep(sweep * 2, "lr", Energy, trunc, states)
        
        if abs(e_avg - Eold) < conv:
            if verbose >= 1:
                print("DMRG Converged at sweep {}. \n average energy = {}".format(sweep, e_avg))
            converged = True
            gauge = "Left"
            break
        else:
            Eold = e_avg

        nz = _noise(sweep * 2 + 1)
        for i in range(len(MPS)-2, 0, -1):
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1=U1, sym_mgr=sym_mgr,
                    nstates=nstates, weights=weights, davidson_tol=davidson_tol,
                    davidson_max_iter=davidson_max_iter, noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
                E_ground_state = Energy[0]
            else:
                Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1=U1, sym_mgr=sym_mgr,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=nz, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
                
                E_ground_state = Energy
            logging.info("Sweep {} Sites {},{}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2+1, i, i+1, E_ground_state, states, trunc))
            F.append(contract_from_right(MPO[i+1], MPS[i+1], F[-1], MPS[i+1]))
            for name, factors in complementary_operator_mpos.items():
                comp_family_F[name].append(
                    contract_from_right(
                        factors[i + 1],
                        MPS[i + 1],
                        comp_family_F[name][-1],
                        MPS[i + 1],
                    )
                )
            E.pop()
            for stack in comp_family_E.values():
                stack.pop()
            last_i = i
            
        if nstates > 1:
            e_avg = np.sum(weights * Energy) 
        else:
            e_avg = Energy
        _notify_sweep(sweep * 2 + 1, "rl", Energy, trunc, states)
        if abs(e_avg - Eold) < conv:
            if verbose >= 1:
                print("DMRG Converged at sweep {}. \n average energy = {}".format(sweep, e_avg))
            converged = True
            gauge = "Right"
            break
        else:
            Eold = e_avg

    if not_conv_err == True:
        if converged == False:
            raise ValueError("DMRG did not converge within the given number of sweeps, if you wish to disable this error, set not_conv_err = False. or you should increase the number of sweeps.")
    else:
        if converged == False:
            if verbose >= 1:
                print("DMRG did not converge within {sweeps} sweeps, returning the last result.")
    if gauge == None:
        gauge = "Right"

    center_i = len(MPS) // 2 - 1
    
    if gauge == "Left" and last_i > center_i: 
        # Broke after Right Sweep. E and F are perfectly set up for a Left sweep start.
        for i in range(len(MPS)-2, center_i - 1, -1):
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1, sym_mgr, nstates, weights,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
            else:
                Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'left', U1, sym_mgr,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
            if i > center_i: # Don't shift environments on the final stop
                F.append(contract_from_right(MPO[i+1], MPS[i+1], F[-1], MPS[i+1]))
                for name, factors in complementary_operator_mpos.items():
                    comp_family_F[name].append(
                        contract_from_right(
                            factors[i + 1],
                            MPS[i + 1],
                            comp_family_F[name][-1],
                            MPS[i + 1],
                        )
                    )
                E.pop()
                for stack in comp_family_E.values():
                    stack.pop()
            last_i = i

    elif gauge == "Right" and last_i < center_i: 
        # Broke after Left Sweep. E and F are perfectly set up for a Right sweep start.
        for i in range(0, center_i + 1):
            if nstates > 1:
                Energy, MPS[i], MPS[i+1], trunc, states, last_AA_list = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1, sym_mgr, nstates, weights,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
            else:
                Energy, MPS[i], MPS[i+1], trunc, states = optimize_two_sites(
                    MPS[i], MPS[i+1], MPO[i], MPO[i+1], E[-1], F[-1], m, 'right', U1, sym_mgr,
                    davidson_tol=davidson_tol, davidson_max_iter=davidson_max_iter,
                    noise=0.0, local_dense_max_dim=local_dense_max_dim,
                    complementary_operator_families=complementary_operator_families, bond=i,
                    complementary_boundary_payloads=_comp_payload(i),
                    complementary_split_stats=comp_split_stats,
                    complementary_family_environments=_comp_family_env(i),
                    complementary_direct_family_environments=_direct_family_env(i))
                _invalidate_direct_family_env_cache()
            if i < center_i: # Don't shift environments on the final stop
                E.append(contract_from_left(MPO[i], MPS[i], E[-1], MPS[i]))
                for name, factors in complementary_operator_mpos.items():
                    comp_family_E[name].append(
                        contract_from_left(
                            factors[i],
                            MPS[i],
                            comp_family_E[name][-1],
                            MPS[i],
                        )
                    )
                F.pop()
                for stack in comp_family_F.values():
                    stack.pop()
            last_i = i
            
    if nstates == 1:
            return Energy, MPS, gauge, converged
    else:
        final_states = []
        for k in range(nstates):
            MPS_k = [B.copy() for B in MPS]
            # Unspool the exact roots found at the last bond
            if U1:
                U, V, S_dict, _, _ = svd_symmetric(last_AA_list[k], m_max=None)
                A_US = multiply_U_S(U, S_dict)
                MPS_k[last_i] = A_US.transpose(0, 2, 1)
                MPS_k[last_i+1] = V
            else:
                A_root, S_root, B_root = fine_grain_MPS(
                    last_AA_list[k], [MPS[last_i].shape[1], MPS[last_i+1].shape[1]]
                )
                A_root, S_root, B_root, _, _ = truncate_SVD(
                    A_root, S_root, B_root, m
                )
                MPS_k[last_i] = np.tensordot(A_root, np.diag(S_root), axes=(2, 0))
                MPS_k[last_i+1] = B_root
            final_states.append(MPS_k)
        return Energy, final_states, gauge, converged


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

# Helper for RDM calculation
def build_annihilation_mpo_symmetric(site_idx, L, sym_mgr, spin_sector):
    """    
    Constructs U(1) symmetric MPO for annihilation operator a_k.
    Handles both d=2 (spin-orbital) and d=4 (spatial-orbital) mappings.
    """
    if not SYMMETRY_AVAILABLE: raise ImportError("Symmetry required")
    
    vac_qn = sym_mgr.get_vac_qn()
    d_local = len(sym_mgr.phys_qns)
    
    # Identify standard QNs from the list
    if d_local == 2:
        q_emp, q_occ = sym_mgr.phys_qns[0], sym_mgr.phys_qns[1]
        q_particle = q_occ
    elif d_local == 4:
        q_emp, q_up, q_dn, q_docc = sym_mgr.phys_qns
        q_particle = q_up if spin_sector == 'up' else q_dn
    else:
        raise NotImplementedError(f"Unsupported d={d_local} in annihilation builder.")
        
    tensors = []
    
    for i in range(L):
        data = {}
        if d_local == 2:
            if i < site_idx:
                data[(vac_qn, vac_qn, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(vac_qn, vac_qn, q_occ, q_occ)] = np.array([[[[-1.0]]]])
            elif i == site_idx:
                is_up = (i % 2 == 0)
                if (spin_sector == 'up' and is_up) or (spin_sector == 'down' and not is_up):
                    data[(vac_qn, q_particle, q_emp, q_occ)] = np.array([[[[1.0]]]])
            else: 
                data[(q_particle, q_particle, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_occ, q_occ)] = np.array([[[[1.0]]]])
                
        elif d_local == 4:
            if i < site_idx:
                data[(vac_qn, vac_qn, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(vac_qn, vac_qn, q_up, q_up)] = np.array([[[[-1.0]]]])
                data[(vac_qn, vac_qn, q_dn, q_dn)] = np.array([[[[-1.0]]]])
                data[(vac_qn, vac_qn, q_docc, q_docc)] = np.array([[[[1.0]]]])
            elif i == site_idx:
                if spin_sector == 'up':
                    data[(vac_qn, q_particle, q_emp, q_up)] = np.array([[[[1.0]]]])
                    data[(vac_qn, q_particle, q_dn, q_docc)] = np.array([[[[1.0]]]])
                elif spin_sector == 'down':
                    data[(vac_qn, q_particle, q_emp, q_dn)] = np.array([[[[1.0]]]])
                    data[(vac_qn, q_particle, q_up, q_docc)] = np.array([[[[-1.0]]]])
            else:
                data[(q_particle, q_particle, q_emp, q_emp)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_up, q_up)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_dn, q_dn)] = np.array([[[[1.0]]]])
                data[(q_particle, q_particle, q_docc, q_docc)] = np.array([[[[1.0]]]])

        if not data:
             qL = vac_qn if i <= site_idx else q_particle
             qR = vac_qn if i < site_idx else q_particle
             data[(qL, qR, q_emp, q_occ if d_local==2 else q_up)] = np.zeros((1,1,1,1))

        used_L = sorted(list(set(k[0] for k in data)))
        used_R = sorted(list(set(k[1] for k in data)))
        used_Out = sorted(list(set(k[2] for k in data)))
        used_In = sorted(list(set(k[3] for k in data)))
        
        tensors.append(BlockTensor(data, [used_L, used_R, used_Out, used_In], [1, -1, 1, -1]))
        
    return tensors


def apply_mpo_symmetric(W_list, M_list):
    """
    Symmetric application |Psi'> = W |Psi>. 
    Robustly handles block fusion by pre-calculating dimensions.
    """
    import collections
    new_mps = []
    L = len(M_list)
    
    # [FIX] Dynamically determine the Vacuum QN from the first tensor's first bond
    # This handles both int (0) and QN objects (QN(0,0))
    first_key = next(iter(M_list[0].data.keys()))
    vac_qn = first_key[0] # Left bond QN
    
    # Initialize map with the correct QN object
    # Structure: { q_new: [ ((qw, qm), dim_prod), ... ] }
    last_right_basis_map = {vac_qn: [((vac_qn, vac_qn), 1)]} 
    
    # Helper to get dimensions of a leg
    def _get_qn_dims(bt, leg_idx):
        dims = {}
        for key, block in bt.data.items():
            q = key[leg_idx]; d = block.shape[leg_idx]
            dims[q] = d
        return dims

    for i in range(L):
        W = W_list[i]; M = M_list[i]
        
        # 1. Contract Phys Indices: W[In] with M[Phys]
        T = tensordot(W, M, axes=([3], [2]))
        
        # 2. Determine new Right Basis
        current_right_basis_map = collections.defaultdict(list)
        w_dims_r = _get_qn_dims(W, 1)
        m_dims_r = _get_qn_dims(M, 1)
        
        for key_T in T.data:
            qw_r, qm_r = key_T[1], key_T[4]
            q_r_new = qw_r + qm_r
            
            if qw_r in w_dims_r and qm_r in m_dims_r:
                d = w_dims_r[qw_r] * m_dims_r[qm_r]
                pair_info = ((qw_r, qm_r), d)
                if pair_info not in current_right_basis_map[q_r_new]:
                    current_right_basis_map[q_r_new].append(pair_info)
        
        for q in current_right_basis_map:
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

        for sector, comps in blocks_by_sector.items():
            q_l_new, q_r_new, q_p_out = sector
            
            row_info_list = last_right_basis_map.get(q_l_new, [])
            col_info_list = current_right_basis_map.get(q_r_new, [])
            
            if not row_info_list or not col_info_list: continue 
            
            r_dim = sum(d for _, d in row_info_list)
            c_dim = sum(d for _, d in col_info_list)
            
            row_offsets = {}; current_r = 0
            for pair, d in row_info_list:
                row_offsets[pair] = (current_r, d)
                current_r += d
            col_offsets = {}; current_c = 0
            for pair, d in col_info_list:
                col_offsets[pair] = (current_c, d)
                current_c += d

            # Boundary Condition: Force Dim=1 at last site
            if i == L-1:
                if c_dim > 1: c_dim = 1

            new_block = np.zeros((r_dim, c_dim, 1), dtype=complex) 
            
            for ((w_l, m_l), (w_r, m_r)), blk in comps.items():
                if (w_l, m_l) not in row_offsets or (w_r, m_r) not in col_offsets: continue
                r_start, r_len = row_offsets[(w_l, m_l)]
                c_start, c_len_full = col_offsets[(w_r, m_r)]
                
                blk_perm = blk.transpose(0, 3, 1, 4, 2)
                dim_p = blk.shape[2]
                if new_block.shape[2] != dim_p: 
                    new_block = np.zeros((r_dim, c_dim, dim_p), dtype=complex)
                
                to_fill = blk_perm.reshape(blk.shape[0]*blk.shape[3], blk.shape[1]*blk.shape[4], dim_p)
                actual_r = min(r_len, to_fill.shape[0])
                actual_c = min(c_len_full, c_dim - c_start)
                actual_c = min(actual_c, to_fill.shape[1])
                
                if actual_r > 0 and actual_c > 0:
                    new_block[r_start:r_start+actual_r, c_start:c_start+actual_c, :] = to_fill[:actual_r, :actual_c, :]

            if np.sum(np.abs(new_block)) > 1e-16:
                new_data[sector] = new_block

        qns_L = sorted(list(set(k[0] for k in new_data)))
        qns_R = sorted(list(set(k[1] for k in new_data)))
        qns_P = list(W.qns[2])
        
        new_mps.append(BlockTensor(new_data, [qns_L, qns_R, qns_P], [-1, 1, 1]))
        last_right_basis_map = current_right_basis_map

    return new_mps



if __name__ == '__main__':
    from pyqed.mps.dmrg import DMRG
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
