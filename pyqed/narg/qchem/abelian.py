#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:18:41 2025

@author: Bing Gu (gubing@westlake.edu.cn)

NARG for fermionic chain models (e.g. Hubbard model, quantum chemistry)
"""


from pyqed.qchem.jordan_wigner.spinful import annihilate, create
from pyqed import dag, tensor, transform, expect, hadamard, pauli
from pyqed.mps.fermion import SpinHalfFermionChain

from pyqed import SpinHalfFermionOperators, eigh, sort
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, create

from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, gmres
from scipy.sparse import kron, eye, csr_matrix, coo_matrix, issparse

from opt_einsum import contract

import numpy as np

try:
    from pyqed import TFIM, multispin
except ImportError:  # optional spin-model helpers are not needed by qchem NARG
    TFIM = None
    multispin = None
from pyqed.qchem import Molecule, build_atom_from_coords
from pyqed.phys import eigh
from pyqed.qchem.ci.fci import FCI
from pyqed.phys import obs, isdiag

import logging as log

import logging
from dataclasses import dataclass
from functools import lru_cache

from .active_space import CAS_OPTION_DEFAULTS, pop_active_space_options, prepare_active_space
from .rdm import make_rdm1_from_narg, make_rdm2_from_narg
from pyqed.narg.hamiltonian import normalize_orbital_blocks
from ..core import Block, NARGBase, Step

try:
    from numba import njit
except Exception:  # pragma: no cover - optional accelerator
    njit = None

try:
    from pyqed.narg.irrep_tensor import Irrep, IrrepSite, IrrepTensor, OpIrrep, ProductSymmetry, U1Symmetry
except Exception:  # pragma: no cover - optional bridge
    Irrep = IrrepSite = IrrepTensor = OpIrrep = ProductSymmetry = U1Symmetry = None

try:
    from . import abelian_cython as _abelian_cython
except Exception:  # pragma: no cover - optional accelerator
    _abelian_cython = None

try:
    from . import compiled as _compiled_qchem
except Exception:  # pragma: no cover - optional accelerator
    _compiled_qchem = None

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)



# rotate the basis to the new representation

def rotate(A, B, U):
    r"""
    rotate :math:`V = A \otimes B` to the adiabatic representation
    :math:`|\phi_{\alpha_l n_{l+1}} \rangle \otimes | n_{l+1} \rangle

    Parameters
    ----------
    A : TYPE
        operator in the first l sites
    B : TYPE
        operator in the l+1 site
    U : ndarray [primitive basis index, adiabatic state index, (l+1)th site state index]
        DESCRIPTION.

    Returns
    -------
    v : TYPE
        DESCRIPTION.

    """
    B_arr = np.asarray(B)
    if _compiled_qchem is not None and np.count_nonzero(B_arr) < B_arr.size:
        return _compiled_qchem.rotate(A, B_arr, U)

    n, D, d = U.shape

    assert(n == A.shape[0])
    assert(d == B_arr.shape[1])

    # Columns are ordered as (local state, block state), matching the final
    # reshape of the old mbna tensor contraction.
    Umat = U.transpose(0, 2, 1).reshape(n, d * D)
    rotated = np.asarray(Umat.conj().T @ np.asarray(A @ Umat))
    return (
        rotated.reshape(d, D, d, D)
        * B_arr[:, None, :, None]
    ).reshape(d * D, d * D)


def valid_qn_mask(qns):
    """Mask padded sentinel states introduced by ``pad_branch``."""
    return np.all(qn_array(qns) > -10**8, axis=1)


def array_result_dtype(*arrays):
    dtypes = []
    for array in arrays:
        if isinstance(array, type):
            dtype = np.dtype(array)
        else:
            dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.asarray(array).dtype
        dtypes.append(dtype)
    return np.result_type(*dtypes)


def operator_dtype(op):
    dtype = getattr(op, "dtype", None)
    return np.asarray(op).dtype if dtype is None else dtype


def dense_operator(op):
    return op.toarray() if issparse(op) else np.asarray(op)


class RotationPlan:
    """Cached symmetry/local-state slices for repeated Abelian rotations."""

    __slots__ = (
        "U",
        "output_qn",
        "n",
        "D",
        "d",
        "out_dim",
        "dtype",
        "sectors",
        "transitions",
        "local_transitions",
        "rotations",
    )

    def __init__(self, U, output_qn):
        self.U = U
        self.output_qn = qn_array(output_qn)
        self.n, self.D, self.d = U.shape
        self.out_dim = self.D * self.d
        if len(self.output_qn) != self.out_dim:
            raise ValueError(f"output_qn has length {len(self.output_qn)}, expected {self.out_dim}")
        self.dtype = np.result_type(U, complex)
        self.sectors = self._build_sectors()
        self.transitions = {}
        self.local_transitions = {}
        self.rotations = {}

    def _build_sectors(self):
        valid = valid_qn_mask(self.output_qn)
        sectors = {}
        for qn in sorted({qn_key(row) for row in self.output_qn[valid]}):
            flat = np.flatnonzero(valid & np.all(self.output_qn == np.asarray(qn), axis=1))
            local = []
            for local_id in range(self.d):
                idx = flat[flat // self.D == local_id]
                local.append((idx, idx % self.D))
            sectors[qn] = tuple(local)
        return sectors

    def sector_transitions(self, shift):
        shift = qn_key(shift)
        cached = self.transitions.get(shift)
        if cached is not None:
            return cached
        shift_arr = np.asarray(shift, dtype=int)
        pairs = []
        for ket_qn, ket_parts in self.sectors.items():
            bra_parts = self.sectors.get(qn_key(np.asarray(ket_qn, dtype=int) + shift_arr))
            if bra_parts is not None:
                pairs.append((bra_parts, ket_parts))
        self.transitions[shift] = tuple(pairs)
        return self.transitions[shift]

    def nonzero_local_transitions(self, B):
        B = np.asarray(B)
        key = (B.shape, B.dtype.str, B.tobytes())
        cached = self.local_transitions.get(key)
        if cached is not None:
            return cached
        rows, cols = np.nonzero(np.abs(B) > 0)
        cached = tuple((int(row), int(col), B[row, col]) for row, col in zip(rows, cols))
        self.local_transitions[key] = cached
        return cached

    def local_key(self, B):
        B = np.asarray(B)
        return (B.shape, B.dtype.str, B.tobytes())

    def rotate_cached(self, A, B, shift):
        key = (id(A), self.local_key(B), qn_key(shift))
        cached = self.rotations.get(key)
        if cached is not None and cached[0] is A:
            return cached[1]
        result = self.rotate(A, B, shift)
        self.rotations[key] = (A, result)
        return result

    def rotate(self, A, B, shift):
        if not self.sector_transitions(shift):
            return np.zeros((self.out_dim, self.out_dim), dtype=np.result_type(A, B, self.U, complex))
        out = np.zeros((self.out_dim, self.out_dim), dtype=np.result_type(A, B, self.U, complex))
        local_transitions = self.nonzero_local_transitions(B)
        if not local_transitions:
            return out
        for bra_parts, ket_parts in self.sector_transitions(shift):
            for bra_local, ket_local, coeff in local_transitions:
                bra, bra_states = bra_parts[bra_local]
                ket, ket_states = ket_parts[ket_local]
                if bra.size == 0 or ket.size == 0:
                    continue
                left = self.U[:, bra_states, bra_local]
                right = self.U[:, ket_states, ket_local]
                out[bra[:, None], ket[None, :]] += coeff * (left.conj().T @ (A @ right))
        return out

    def rotate_with_applied_cache(self, A, B, shift, applied):
        if not self.sector_transitions(shift):
            return np.zeros((self.out_dim, self.out_dim), dtype=np.result_type(A, B, self.U, complex))
        out = np.zeros((self.out_dim, self.out_dim), dtype=np.result_type(A, B, self.U, complex))
        local_transitions = self.nonzero_local_transitions(B)
        if not local_transitions:
            return out
        for bra_parts, ket_parts in self.sector_transitions(shift):
            for bra_local, ket_local, coeff in local_transitions:
                bra, bra_states = bra_parts[bra_local]
                ket, ket_states = ket_parts[ket_local]
                if bra.size == 0 or ket.size == 0:
                    continue
                left = self.U[:, bra_states, bra_local]
                key = (ket_local, id(ket_states))
                right = applied.get(key)
                if right is None:
                    right = A @ self.U[:, ket_states, ket_local]
                    applied[key] = right
                out[bra[:, None], ket[None, :]] += coeff * (left.conj().T @ right)
        return out


def blockwise_rotate(A, B, U, output_qn, shift, plan=None):
    """Rotate ``A tensor B`` while filling only symmetry-allowed blocks.

    ``output_qn`` labels the rotated basis ordered as ``local_state * D + state``.
    ``shift`` is the total (Ne, 2Sz) shift of ``A tensor B``.
    """
    if plan is not None:
        return plan.rotate_cached(A, B, shift)
    output_qn = qn_array(output_qn)
    shift = np.asarray(shift, dtype=int)
    n, D, d = U.shape
    out_dim = d * D
    if len(output_qn) != out_dim:
        raise ValueError(f"output_qn has length {len(output_qn)}, expected {out_dim}")
    if not has_qn_transition(output_qn, shift):
        return np.zeros((out_dim, out_dim), dtype=array_result_dtype(A, B, U, complex))

    B = np.asarray(B)
    out = np.zeros((out_dim, out_dim), dtype=array_result_dtype(A, B, U, complex))
    valid = valid_qn_mask(output_qn)
    sectors = {}
    for qn in sorted({qn_key(row) for row in output_qn[valid]}):
        sectors[qn] = np.flatnonzero(valid & np.all(output_qn == np.asarray(qn), axis=1))

    for ket_qn, ket_flat in sectors.items():
        bra_qn = qn_key(np.asarray(ket_qn) + shift)
        bra_flat = sectors.get(bra_qn)
        if bra_flat is None:
            continue

        for bra_local in range(d):
            bra = bra_flat[bra_flat // D == bra_local]
            if bra.size == 0:
                continue
            bra_states = bra % D
            left = U[:, bra_states, bra_local]

            for ket_local in np.flatnonzero(np.abs(B[bra_local]) > 0):
                ket = ket_flat[ket_flat // D == ket_local]
                if ket.size == 0:
                    continue
                ket_states = ket % D
                right = U[:, ket_states, ket_local]
                out[np.ix_(bra, ket)] += B[bra_local, ket_local] * (left.conj().T @ (A @ right))
    return out


def blockwise_rotate_sparse(A, B, U, output_qn, shift, plan=None, atol=0.0):
    """Sparse CSR variant of ``blockwise_rotate`` for projected operator tables."""
    output_qn = qn_array(output_qn)
    shift = np.asarray(shift, dtype=int)
    n, D, d = U.shape
    out_dim = d * D
    dtype = array_result_dtype(A, B, U, complex)
    if len(output_qn) != out_dim:
        raise ValueError(f"output_qn has length {len(output_qn)}, expected {out_dim}")
    if not has_qn_transition(output_qn, shift):
        return csr_matrix((out_dim, out_dim), dtype=dtype)

    B = np.asarray(B)
    rows = []
    cols = []
    data = []

    if plan is not None:
        transitions = plan.sector_transitions(shift)
        local_transitions = plan.nonzero_local_transitions(B)
        for bra_parts, ket_parts in transitions:
            for bra_local, ket_local, coeff in local_transitions:
                bra, bra_states = bra_parts[bra_local]
                ket, ket_states = ket_parts[ket_local]
                if bra.size == 0 or ket.size == 0:
                    continue
                left = U[:, bra_states, bra_local]
                right = U[:, ket_states, ket_local]
                block = coeff * (left.conj().T @ (A @ right))
                if atol:
                    brow, bcol = np.nonzero(np.abs(block) > atol)
                else:
                    brow, bcol = np.nonzero(block)
                if brow.size:
                    rows.append(bra[brow])
                    cols.append(ket[bcol])
                    data.append(block[brow, bcol])
    else:
        valid = valid_qn_mask(output_qn)
        sectors = {}
        for qn in sorted({qn_key(row) for row in output_qn[valid]}):
            sectors[qn] = np.flatnonzero(valid & np.all(output_qn == np.asarray(qn), axis=1))

        for ket_qn, ket_flat in sectors.items():
            bra_qn = qn_key(np.asarray(ket_qn) + shift)
            bra_flat = sectors.get(bra_qn)
            if bra_flat is None:
                continue
            for bra_local in range(d):
                bra = bra_flat[bra_flat // D == bra_local]
                if bra.size == 0:
                    continue
                bra_states = bra % D
                left = U[:, bra_states, bra_local]
                for ket_local in np.flatnonzero(np.abs(B[bra_local]) > 0):
                    ket = ket_flat[ket_flat // D == ket_local]
                    if ket.size == 0:
                        continue
                    ket_states = ket % D
                    right = U[:, ket_states, ket_local]
                    block = B[bra_local, ket_local] * (left.conj().T @ (A @ right))
                    if atol:
                        brow, bcol = np.nonzero(np.abs(block) > atol)
                    else:
                        brow, bcol = np.nonzero(block)
                    if brow.size:
                        rows.append(bra[brow])
                        cols.append(ket[bcol])
                        data.append(block[brow, bcol])

    if not data:
        return csr_matrix((out_dim, out_dim), dtype=dtype)
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(out_dim, out_dim),
        dtype=dtype,
    ).tocsr()


def rotate_symmetry(
    A, B, U, output_qn=None, shift=None, use_irrep_blocks=False, plan=None, sparse_output=False
):
    use_blocks = bool(use_irrep_blocks)
    if output_qn is not None and shift is not None and not use_blocks:
        B_arr = np.asarray(B)
        use_blocks = np.count_nonzero(B_arr) < B_arr.size
    if use_blocks and output_qn is not None and shift is not None:
        if sparse_output:
            return blockwise_rotate_sparse(A, B, U, output_qn, shift, plan=plan)
        return blockwise_rotate(A, B, U, output_qn, shift, plan=plan)
    return rotate(A, B, U)




# logger = logging.getLogger('foo')
# log.setLevel(log.INFO)
# logger.info(f'active')

#### fermion chain
ops = SpinHalfFermionOperators()
cd = ops['Cd']
cu = ops['Cu']
cdu = ops['Cdu']
cdd = ops['Cdd']
JW = ops['JW']
Ntot = ops['Ntot']
Nu = ops['Nu']
Nd = ops['Nd']
Sz = ops['Sz']
Sp = ops['Sp']
Sm = ops['Sm']
S2_LOCAL = Sz @ Sz + 0.5 * (Sp @ Sm + Sm @ Sp)

# eigenstates of c^\dagger_\uparrow + c_\uparrow
# m = Cdu + Cdu

LOCAL_QN = np.column_stack((
    np.asarray(np.diag(Ntot), dtype=int),
    np.asarray(np.diag(Nu - Nd), dtype=int),
))

OPERATOR_QN_SHIFT = {
    'Cu': np.array([-1, -1], dtype=int),
    'Cd': np.array([-1, 1], dtype=int),
    'Cdu': np.array([1, 1], dtype=int),
    'Cdd': np.array([1, -1], dtype=int),
    'JW': np.array([0, 0], dtype=int),
}


def qn_key(qn):
    return tuple(int(x) for x in np.asarray(qn, dtype=int).tolist())


def qn_array(qns):
    qns = np.asarray(qns, dtype=int)
    if qns.ndim == 1:
        qns = qns.reshape((-1, LOCAL_QN.shape[1]))
    return qns


def irrep_tensor_available():
    return all(x is not None for x in (Irrep, IrrepSite, IrrepTensor, OpIrrep, ProductSymmetry, U1Symmetry))


def irrep_site_from_qn(basis_qn):
    """Build an IrrepSite from existing Abelian (Ne, 2Sz) labels."""
    if not irrep_tensor_available():
        raise ImportError("irrep_tensor module is not available.")
    basis_qn = qn_array(basis_qn)
    symmetry = ProductSymmetry((U1Symmetry("Ne"), U1Symmetry("2Sz")), name="U1xU1")
    sector_indices = {}
    dims = {}
    for qn in sorted({qn_key(row) for row in basis_qn}):
        irrep = Irrep(qn)
        idx = np.flatnonzero(np.all(basis_qn == np.asarray(qn), axis=1))
        sector_indices[irrep] = idx
        dims[irrep] = len(idx)
    return IrrepSite(symmetry, dims), sector_indices


def op_charge_add(left, right):
    left = tuple(np.asarray(left, dtype=int).reshape(-1).tolist())
    right = tuple(np.asarray(right, dtype=int).reshape(-1).tolist())
    if len(left) != len(right):
        raise ValueError(f"operator charges have different ranks: {left} and {right}")
    return tuple(a + b for a, b in zip(left, right))


def labeled_irrep_tensor(matrix, bra_qn, ket_qn=None, op=(0, 0), *, atol=0.0):
    """Wrap a labeled dense/sparse matrix as an Abelian charge-block tensor."""
    if not irrep_tensor_available():
        raise ImportError("irrep_tensor module is not available.")
    matrix = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    bra_qn = qn_array(bra_qn)
    ket_qn = bra_qn if ket_qn is None else qn_array(ket_qn)
    if matrix.shape != (len(bra_qn), len(ket_qn)):
        raise ValueError(f"matrix shape {matrix.shape} does not match labels {(len(bra_qn), len(ket_qn))}")
    bra_site, bra_idx = irrep_site_from_qn(bra_qn)
    ket_site, ket_idx = irrep_site_from_qn(ket_qn)
    op = OpIrrep(tuple(np.asarray(op, dtype=int).reshape(-1).tolist()))
    blocks = {}
    for bra_irrep, rows in bra_idx.items():
        for ket_irrep, cols in ket_idx.items():
            if not bra_site.symmetry.allows(bra_irrep.charge, op.charge, ket_irrep.charge):
                continue
            block = matrix[np.ix_(rows, cols)]
            if np.any(np.abs(block) > atol):
                blocks[(bra_irrep, ket_irrep)] = block.copy()
    return IrrepTensor(bra_site, ket_site, op, blocks)


def labeled_dense(tensor, bra_qn, ket_qn=None):
    """Expand an IrrepTensor back to the original labeled basis order."""
    bra_qn = qn_array(bra_qn)
    ket_qn = bra_qn if ket_qn is None else qn_array(ket_qn)
    _, bra_idx = irrep_site_from_qn(bra_qn)
    _, ket_idx = irrep_site_from_qn(ket_qn)
    dense = np.zeros((len(bra_qn), len(ket_qn)), dtype=tensor.dtype)
    for bra_irrep, rows in bra_idx.items():
        for ket_irrep, cols in ket_idx.items():
            block = tensor.block(bra_irrep, ket_irrep)
            if block.size:
                dense[np.ix_(rows, cols)] = block
    return dense


def matmul_tensors(left, right, *, atol=1e-14):
    """Block-multiply Abelian IrrepTensors, adding their charge shifts."""
    if left.ket != right.bra:
        raise ValueError("inner IrrepSite mismatch")
    op = OpIrrep(op_charge_add(left.op.charge, right.op.charge))
    blocks = {}
    for (bra, mid), left_block in left.blocks.items():
        for (mid2, ket), right_block in right.blocks.items():
            if mid2 != mid:
                continue
            if not left.bra.symmetry.allows(bra.charge, op.charge, ket.charge):
                continue
            key = (bra, ket)
            block = left_block @ right_block
            blocks[key] = block if key not in blocks else blocks[key] + block
    blocks = {key: block for key, block in blocks.items() if np.any(np.abs(block) > atol)}
    return IrrepTensor(left.bra, right.ket, op, blocks)


def add_tensors(*tensors, atol=1e-14):
    """Add matching Abelian IrrepTensor blocks."""
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        raise ValueError("at least one tensor is required")
    first = tensors[0]
    blocks = {}
    for tensor in tensors:
        if tensor.bra != first.bra or tensor.ket != first.ket or tensor.op != first.op:
            raise ValueError("all tensors must share bra, ket, and operator charge")
        for key, block in tensor.blocks.items():
            blocks[key] = block.copy() if key not in blocks else blocks[key] + block
    blocks = {key: block for key, block in blocks.items() if np.any(np.abs(block) > atol)}
    return IrrepTensor(first.bra, first.ket, first.op, blocks)


def scale_tensor(tensor, factor, *, atol=1e-14):
    """Scale an IrrepTensor, dropping numerically empty blocks."""
    if abs(factor) <= atol:
        return IrrepTensor(tensor.bra, tensor.ket, tensor.op, {})
    blocks = {
        key: factor * block
        for key, block in tensor.blocks.items()
        if np.any(np.abs(factor * block) > atol)
    }
    return IrrepTensor(tensor.bra, tensor.ket, tensor.op, blocks)


def zero_tensor(site, op):
    return IrrepTensor(site, site, OpIrrep(tuple(np.asarray(op, dtype=int).reshape(-1).tolist())), {})


def sum_tensor_terms(site, op, terms, *, atol=1e-14):
    """Weighted sum of IrrepTensors with a known target operator charge."""
    out = zero_tensor(site, op)
    blocks = {}
    for coeff, tensor in terms:
        if abs(coeff) <= atol:
            continue
        if tensor.bra != site or tensor.ket != site:
            raise ValueError("tensor site mismatch")
        if tensor.op != out.op:
            raise ValueError(f"tensor charge {tensor.op.charge} does not match expected {out.op.charge}")
        for key, block in tensor.blocks.items():
            blocks[key] = coeff * block if key not in blocks else blocks[key] + coeff * block
    blocks = {key: block for key, block in blocks.items() if np.any(np.abs(block) > atol)}
    return IrrepTensor(site, site, out.op, blocks)


def branch_qn(block_qn, local_qn=LOCAL_QN):
    """Direct-product labels in local-major order, matching ``np.kron(B, A)``."""
    block_qn = qn_array(block_qn)
    local_qn = qn_array(local_qn)
    return (local_qn[:, None, :] + block_qn[None, :, :]).reshape((-1, block_qn.shape[1]))


def branch_transform_tensor(U, block_qn, output_qn):
    """IrrepTensor for the branch transform from source product space to kept states."""
    n, D, d = U.shape
    block_qn = qn_array(block_qn)
    if len(block_qn) != n:
        raise ValueError(f"block_qn has length {len(block_qn)}, expected {n}")
    source_qn = branch_qn(block_qn, LOCAL_QN[:d])
    output_qn = qn_array(output_qn)
    if len(output_qn) != D * d:
        raise ValueError(f"output_qn has length {len(output_qn)}, expected {D * d}")
    dense = np.zeros((n * d, D * d), dtype=np.result_type(U, complex))
    for local_id in range(d):
        rows = local_id * n + np.arange(n)
        cols = local_id * D + np.arange(D)
        dense[np.ix_(rows, cols)] = U[:, :, local_id]
    return labeled_irrep_tensor(dense, source_qn, output_qn, op=(0, 0)), source_qn


def product_irrep_tensor(A, B, block_qn, shift):
    """IrrepTensor for ``A tensor B`` in the branch source basis."""
    source_qn = branch_qn(block_qn, LOCAL_QN[:np.asarray(B).shape[0]])
    dense = np.kron(np.asarray(B), A.toarray() if issparse(A) else np.asarray(A))
    return labeled_irrep_tensor(dense, source_qn, source_qn, op=shift), source_qn


def local_operator_shift(names):
    shift = np.zeros(LOCAL_QN.shape[1], dtype=int)
    for name in names:
        if name != 'JW':
            shift += OPERATOR_QN_SHIFT[name]
    return tuple(int(x) for x in shift)


def product_irrep_tensor_from_block(block_tensor, B, block_qn, local_shift=None, *, atol=1e-14):
    """Build ``block_tensor tensor B`` directly in product charge sectors."""
    B = np.asarray(B)
    block_qn = qn_array(block_qn)
    local_qn = LOCAL_QN[:B.shape[0]]
    source_qn = branch_qn(block_qn, local_qn)
    source_site, source_idx = irrep_site_from_qn(source_qn)
    _, block_idx = irrep_site_from_qn(block_qn)
    local_shift = tuple(np.zeros(LOCAL_QN.shape[1], dtype=int)) if local_shift is None else local_shift
    op = OpIrrep(op_charge_add(block_tensor.op.charge, local_shift))
    source_dim = len(source_qn)
    block_dim = len(block_qn)
    sector_pos = {}
    for irrep, idx in source_idx.items():
        pos = np.full(source_dim, -1, dtype=int)
        pos[idx] = np.arange(len(idx))
        sector_pos[irrep] = pos

    local_rows, local_cols = np.nonzero(np.abs(B) > 0)
    blocks = {}
    for (bra_block_irrep, ket_block_irrep), block in block_tensor.blocks.items():
        bra_block_idx = block_idx[bra_block_irrep]
        ket_block_idx = block_idx[ket_block_irrep]
        if block.shape != (len(bra_block_idx), len(ket_block_idx)):
            raise ValueError("block tensor shape does not match block_qn labels")
        for bra_local, ket_local in zip(local_rows, local_cols):
            coeff = B[bra_local, ket_local]
            bra_charge = qn_key(np.asarray(bra_block_irrep.charge) + local_qn[bra_local])
            ket_charge = qn_key(np.asarray(ket_block_irrep.charge) + local_qn[ket_local])
            if not source_site.symmetry.allows(bra_charge, op.charge, ket_charge):
                continue
            bra_irrep = Irrep(bra_charge)
            ket_irrep = Irrep(ket_charge)
            if bra_irrep not in source_site.dims or ket_irrep not in source_site.dims:
                continue
            key = (bra_irrep, ket_irrep)
            out = blocks.get(key)
            if out is None:
                out = np.zeros(
                    (source_site.sector_dim(bra_irrep), source_site.sector_dim(ket_irrep)),
                    dtype=np.result_type(block, B, complex),
                )
                blocks[key] = out
            row_flat = bra_local * block_dim + bra_block_idx
            col_flat = ket_local * block_dim + ket_block_idx
            rows = sector_pos[bra_irrep][row_flat]
            cols = sector_pos[ket_irrep][col_flat]
            out[rows[:, None], cols[None, :]] += coeff * block

    blocks = {key: block for key, block in blocks.items() if np.any(np.abs(block) > atol)}
    return IrrepTensor(source_site, source_site, op, blocks), source_qn


def rotate_irrep_product(A, B, U, block_qn, output_qn, shift):
    """Rotate ``A tensor B`` with Abelian IrrepTensor blocks."""
    transform_tensor, source_qn = branch_transform_tensor(U, block_qn, output_qn)
    product_tensor, _ = product_irrep_tensor(A, B, block_qn, shift)
    rotated = matmul_tensors(
        matmul_tensors(transform_tensor.adjoint(), product_tensor),
        transform_tensor,
    )
    return rotated, source_qn


def rotate_irrep_tensor_product(block_tensor, B, U, block_qn, output_qn, local_shift=None):
    """Rotate ``block_tensor tensor B`` with Abelian charge-block tensors."""
    transform_tensor, source_qn = branch_transform_tensor(U, block_qn, output_qn)
    product_tensor, _ = product_irrep_tensor_from_block(block_tensor, B, block_qn, local_shift)
    rotated = matmul_tensors(
        matmul_tensors(transform_tensor.adjoint(), product_tensor),
        transform_tensor,
    )
    return rotated, source_qn


@dataclass
class AbelianBlock:
    """IrrepTensor-backed Abelian NARG block."""

    h: IrrepTensor
    qn: np.ndarray
    ops: dict
    residuals: dict | None = None
    spins: dict | None = None

    @property
    def site(self):
        return self.h.bra

    @property
    def dim(self):
        return self.site.dim

    def dense_h(self):
        return labeled_dense(self.h, self.qn)


def operator_pattern_shift(pattern):
    shift = np.zeros(LOCAL_QN.shape[1], dtype=int)
    for name in pattern:
        shift += OPERATOR_QN_SHIFT[name]
    return tuple(int(x) for x in shift)


def irrep_operator_table(table, qn, *, atol=1e-14):
    """Convert the dense/sparse Abelian operator table to IrrepTensor blocks."""
    tensors = {}
    for pattern, entries in table.items():
        shift = operator_pattern_shift(pattern)
        for indices, op in entries.items():
            tensors[(pattern, indices)] = labeled_irrep_tensor(op, qn, op=shift, atol=atol)
    return tensors


def dense_operator_table(irrep_table, qn):
    """Convert a flat IrrepTensor operator table back to the nested dense layout."""
    table = {}
    for (pattern, indices), tensor in irrep_table.items():
        table.setdefault(pattern, {})[indices] = labeled_dense(tensor, qn)
    return table


def dense_pair_sums(irrep_pair_sums, qn):
    return {key: labeled_dense(tensor, qn) for key, tensor in irrep_pair_sums.items()}


def dense_triple_residuals(irrep_residuals, qn):
    return {
        q: (labeled_dense(v1u, qn), labeled_dense(v1d, qn))
        for q, (v1u, v1d) in irrep_residuals.items()
    }


def extend_irrep_operator_table(table, patterns, U, block_qn, output_qn, new_site, required=None):
    """Project composite operators as Abelian IrrepTensors when adding one orbital."""
    old_site, _ = irrep_site_from_qn(block_qn)
    Iblock = IrrepTensor.identity(old_site)
    Isite = np.eye(4)
    local = {
        'Cu': cu,
        'Cd': cd,
        'Cdu': cdu,
        'Cdd': cdd,
    }
    new_table = {}
    for pattern in patterns:
        if required is None:
            index_iter = np.ndindex((new_site + 1,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            old_pattern = []
            old_indices = []
            local_op = Isite
            local_names = []

            for name, idx in zip(pattern, indices):
                if idx == new_site:
                    local_op = local_op @ local[name]
                    local_names.append(name)
                else:
                    old_pattern.append(name)
                    old_indices.append(idx)
                    local_op = local_op @ JW
                    local_names.append('JW')

            if old_pattern:
                block_op = table[(tuple(old_pattern), tuple(old_indices))]
            else:
                block_op = Iblock
            new_table[(pattern, indices)] = rotate_irrep_tensor_product(
                block_op,
                local_op,
                U,
                block_qn,
                output_qn,
                local_operator_shift(local_names),
            )[0]
    return new_table


def abelian_block_from_dense(h, qn, table=None, residuals=None, spins=None, *, atol=1e-14):
    """Build an IrrepTensor-backed block from the current dense Abelian state."""
    qn = qn_array(qn).copy()
    block = AbelianBlock(
        h=labeled_irrep_tensor(h, qn, op=(0, 0), atol=atol),
        qn=qn,
        ops={} if table is None else irrep_operator_table(table, qn, atol=atol),
        residuals=residuals,
        spins=spins,
    )
    return block


def scalar_irrep_tensor_from_labeled_matrix(H, basis_qn):
    """Wrap a scalar matrix over labeled basis states as an IrrepTensor."""
    site, sector_indices = irrep_site_from_qn(basis_qn)
    blocks = {}
    for irrep, idx in sector_indices.items():
        block = H[np.ix_(idx, idx)]
        block = block.toarray() if issparse(block) else np.asarray(block)
        blocks[(irrep, irrep)] = 0.5 * (block + block.T.conj())
    return IrrepTensor(site, site, OpIrrep((0, 0)), blocks), sector_indices


def irrep_scalar_diagonalize(H, basis_qn, nroots, allowed_qn=None):
    """Diagonalize scalar IrrepTensor blocks and return vectors in original order."""
    tensor, _ = scalar_irrep_tensor_from_labeled_matrix(H, basis_qn)
    E, X, qn = diagonalize_scalar_irrep_tensor(tensor, basis_qn, nroots, allowed_qn=allowed_qn)
    return E, X, qn, tensor


def diagonalize_scalar_irrep_tensor(tensor, basis_qn, nroots, allowed_qn=None, allow_empty=False):
    """Diagonalize an existing scalar Abelian IrrepTensor without rebuilding dense blocks."""
    basis_qn = qn_array(basis_qn)
    site, sector_indices = irrep_site_from_qn(basis_qn)
    scalar = tuple(0 for _ in basis_qn[0])
    if tuple(tensor.op.charge) != scalar:
        raise ValueError(f"expected scalar tensor charge {scalar}, got {tensor.op.charge}")
    if tensor.bra != site or tensor.ket != site:
        raise ValueError("tensor site does not match basis_qn labels")
    allowed = None if allowed_qn is None else {qn_key(q) for q in allowed_qn}
    roots = []

    for irrep in site.irreps:
        if allowed is not None and qn_key(irrep.charge) not in allowed:
            continue
        block = tensor.block(irrep, irrep)
        if block.size == 0 and site.sector_dim(irrep) == 0:
            continue
        if block.size == 0:
            block = np.zeros((site.sector_dim(irrep), site.sector_dim(irrep)), dtype=tensor.dtype)
        if block.shape[0] == 1:
            evals = np.array([block[0, 0]])
            evecs = np.ones((1, 1), dtype=np.result_type(block, complex))
        elif block.shape[0] <= max(64, int(nroots) + 2):
            block = 0.5 * (block + block.T.conj())
            evals, evecs = np.linalg.eigh(block)
        else:
            k = min(int(nroots), block.shape[0] - 2)
            evals, evecs = eigsh(block, k=k, which='SA')
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]
        idx = sector_indices[irrep]
        for col, energy in enumerate(evals):
            roots.append((float(np.real(energy)), irrep.charge, idx, evecs[:, col].copy()))

    if not roots:
        if allow_empty:
            return _empty_sector_result(np.empty((0, 0), dtype=tensor.dtype), basis_qn)
        raise ValueError(f"No states found for allowed charges {allowed_qn}.")

    roots.sort(key=lambda item: item[0])
    nselect = min(int(nroots), len(roots))
    energies = np.empty(nselect)
    vectors = np.zeros((len(basis_qn), nselect), dtype=np.result_type(tensor.dtype, complex))
    qn = np.empty((nselect, basis_qn.shape[1]), dtype=int)
    for col, (energy, charge, idx, vec) in enumerate(roots[:nselect]):
        energies[col] = energy
        qn[col] = charge
        vectors[idx, col] = vec
    return energies, vectors, qn


def _empty_sector_result(H, basis_qn):
    basis_qn = qn_array(basis_qn)
    dtype = np.result_type(getattr(H, "dtype", complex), complex)
    return (
        np.empty(0, dtype=float),
        np.zeros((len(basis_qn), 0), dtype=dtype),
        np.empty((0, basis_qn.shape[1]), dtype=int),
    )


def diagonalize_by_qn(H, basis_qn, nroots, allowed_qn=None, use_irrep_tensor=False, allow_empty=False):
    """Dispatch sector diagonalization through the old path or IrrepTensor."""
    if use_irrep_tensor == 'auto':
        use_irrep_tensor = irrep_tensor_available()
    if use_irrep_tensor:
        if not irrep_tensor_available():
            raise ImportError("use_irrep_tensor=True requires the irrep_tensor module.")
        try:
            E, X, qn, _ = irrep_scalar_diagonalize(H, basis_qn, nroots, allowed_qn=allowed_qn)
        except ValueError as exc:
            if allow_empty and "No states found" in str(exc):
                return _empty_sector_result(H, basis_qn)
            raise
        return E, X, qn
    return charge_diagonalize(H, basis_qn, nroots, allowed_qn=allowed_qn, allow_empty=allow_empty)


def pattern_qn_shift(pattern):
    shift = np.zeros(LOCAL_QN.shape[1], dtype=int)
    for name in pattern:
        shift += OPERATOR_QN_SHIFT[name]
    return tuple(int(x) for x in shift)


def has_qn_transition(basis_qn, shift):
    shift = np.asarray(shift, dtype=int)
    if not np.any(shift):
        return True
    qn = qn_array(basis_qn)
    sectors = {qn_key(row) for row in qn}
    for row in qn:
        if qn_key(row - shift) in sectors:
            return True
    return False


def primitive_charge_labels(nsites):
    """(Ne, 2Sz) labels for the direct-product local basis."""
    dims = (len(LOCAL_QN),) * nsites
    labels = np.zeros((len(LOCAL_QN) ** nsites, LOCAL_QN.shape[1]), dtype=int)
    for flat in range(labels.shape[0]):
        labels[flat] = np.sum(LOCAL_QN[list(np.unravel_index(flat, dims))], axis=0)
    return labels


def supersite_charge_labels(norbitals):
    """(Ne, 2Sz) labels for one composite site containing ``norbitals`` orbitals."""
    return primitive_charge_labels(norbitals)


def energy_groups(eps, *, tol=1e-10, max_size=2):
    """Energy-ordered orbital groups for explicit supersite growth.

    Nearly degenerate neighbors are grouped up to ``max_size`` orbitals, while
    nondegenerate band edges such as Hubbard ``k=0`` and ``k=pi`` remain
    one-orbital ``d=4`` sites.
    """
    eps = np.asarray(eps, dtype=float).reshape(-1)
    if eps.size == 0:
        return tuple()
    if max_size is not None and int(max_size) < 1:
        raise ValueError("max_size must be positive or None.")
    max_size = None if max_size is None else int(max_size)
    order = np.argsort(eps, kind="stable")
    groups = []
    group = [int(order[0])]
    ref = float(eps[order[0]])
    for idx in order[1:]:
        idx = int(idx)
        full = max_size is not None and len(group) >= max_size
        close = abs(float(eps[idx]) - ref) <= float(tol)
        if close and not full:
            group.append(idx)
        else:
            groups.append(tuple(group))
            group = [idx]
            ref = float(eps[idx])
    groups.append(tuple(group))
    return tuple(groups)


def supersite_kernel(
    h1e,
    eri,
    groups,
    *,
    D=20,
    nstates=1,
    nelec=None,
    return_tensors=False,
    return_tensor_qns=False,
):
    """Projection-based NARG for composite spinful-orbital supersites.

    Each entry of ``groups`` is a tuple/list of spatial-orbital indices.  The
    Hamiltonian is reordered into this grouped order, and each group is added
    as one local site with dimension ``4 ** len(group)``.  A ``k,-k`` pair is
    therefore a literal ``d=16`` site rather than two sequential ``d=4`` sites.
    """
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    validate_eri_representation(eri, h1e.shape[0])
    groups = [tuple(int(i) for i in group) for group in groups]
    if not groups or any(len(group) < 1 for group in groups):
        raise ValueError("groups must contain at least one non-empty supersite.")
    order = tuple(i for group in groups for i in group)
    norb = h1e.shape[0]
    if sorted(order) != list(range(norb)):
        raise ValueError("groups must partition all orbital indices exactly once.")
    if nelec is None:
        nelec = getattr(mol, "nelec")
    nelec = np.asarray(nelec, dtype=int).reshape(-1)
    target_nelec = int(np.sum(nelec))
    target_sz2 = int(nelec[0] - nelec[1]) if nelec.size == 2 else int(getattr(mol, "spin", 0))
    target_qn = (target_nelec, target_sz2)
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")

    h = h1e[np.ix_(order, order)]
    v = eri_subspace(eri, order)
    prefix = 0
    basis = np.ones((1, 1), dtype=np.result_type(h, v, complex))
    basis_qn = np.zeros((1, LOCAL_QN.shape[1]), dtype=int)
    tensors = []
    tensor_qns = []
    energies = None

    for group in groups:
        local_norb = len(group)
        local_dim = len(LOCAL_QN) ** local_norb
        next_prefix = prefix + local_norb
        prefix_eri = eri_to_dense(eri_subspace(v, range(next_prefix)))
        model = SpinHalfFermionChain(h[:next_prefix, :next_prefix], prefix_eri)
        full_h = model.jordan_wigner()
        projector = kron(csr_matrix(basis), eye(local_dim, format="csr", dtype=basis.dtype), format="csr")
        projected = projector.conj().T @ (full_h @ projector)
        projected = projected.toarray() if issparse(projected) else np.asarray(projected)
        projected = 0.5 * (projected + projected.conj().T)

        local_qn = supersite_charge_labels(local_norb)
        projected_qn = (basis_qn[:, None, :] + local_qn[None, :, :]).reshape((-1, LOCAL_QN.shape[1]))
        keep = min(D, projected.shape[0])
        energies, coeff, next_qn = diagonalize_by_qn(
            projected,
            projected_qn,
            keep,
            allowed_qn=feasible_qns(target_qn, next_prefix, norb),
            allow_empty=True,
        )
        if coeff.shape[1] == 0:
            raise ValueError(f"No feasible retained states after supersite ending at orbital {next_prefix - 1}.")
        primitive_basis = projector @ coeff
        basis = primitive_basis.toarray() if issparse(primitive_basis) else np.asarray(primitive_basis)
        basis_qn = next_qn
        if return_tensors:
            tensor = coeff.reshape((-1, local_dim, coeff.shape[1])).transpose(0, 2, 1).copy()
            tensors.append(tensor)
        if return_tensor_qns:
            tensor_qns.append(
                {
                    "row_qn": projected_qn.copy(),
                    "right_qn_by_next": next_qn.copy(),
                    "local_qn": local_qn.copy(),
                    "local_dim": local_dim,
                    "orbitals": tuple(order[prefix:next_prefix]),
                    "growth_sites": local_norb,
                }
            )
        prefix = next_prefix

    if energies is None:
        raise ValueError("at least one supersite is required.")
    e = energies[:int(nstates)]
    x = basis[:, :int(nstates)]
    results = [e, x]
    if return_tensors:
        results.append(tensors)
    if return_tensor_qns:
        results.append({"factors": tensor_qns})
    return tuple(results) if len(results) > 2 else (e, x)


def reduced_supersite_kernel(
    h1e,
    eri,
    groups,
    *,
    D=20,
    nstates=1,
    nelec=None,
    eri_cutoff=0.0,
    fast=False,
    use_numba_terms='auto',
    use_irrep_tensor=False,
    use_irrep_blocks=False,
    sparse_operator_table='auto',
    use_sparse_operator_projection='auto',
    return_tensors=False,
    return_tensor_qns=False,
):
    """Reduced recursive NARG for one- and two-orbital supersite schedules.

    Unlike :func:`supersite_kernel`, this path never rebuilds and sandwiches the
    full primitive prefix Hamiltonian after the initial local block.  It carries
    the retained Hamiltonian, Abelian charge labels, reduced operator table, and
    future triple residuals through each cluster append.
    """
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    validate_eri_representation(eri, h1e.shape[0])
    groups = [tuple(int(i) for i in group) for group in groups]
    if not groups or any(len(group) < 1 for group in groups):
        raise ValueError("groups must contain at least one non-empty supersite.")
    unsupported = [group for group in groups if len(group) > 2]
    if unsupported:
        raise NotImplementedError(
            "Reduced clustered Abelian NARG currently supports one- and two-orbital "
            "supersites; larger clusters need a reduced multi-site environment merge."
        )
    order = tuple(i for group in groups for i in group)
    norb = h1e.shape[0]
    if sorted(order) != list(range(norb)):
        raise ValueError("groups must partition all orbital indices exactly once.")
    if nelec is None:
        nelec = getattr(mol, "nelec")
    nelec = np.asarray(nelec, dtype=int).reshape(-1)
    target_nelec = int(np.sum(nelec))
    target_sz2 = int(nelec[0] - nelec[1]) if nelec.size == 2 else int(getattr(mol, "spin", 0))
    target_qn = (target_nelec, target_sz2)
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    if fast:
        if use_numba_terms == 'auto':
            use_numba_terms = True
        if use_irrep_blocks is False:
            use_irrep_blocks = True
    if sparse_operator_table == 'auto':
        sparse_operator_table = True
    else:
        sparse_operator_table = bool(sparse_operator_table)
    if use_sparse_operator_projection == 'auto':
        use_sparse_operator_projection = bool(fast)
    else:
        use_sparse_operator_projection = bool(use_sparse_operator_projection)

    h = h1e[np.ix_(order, order)]
    v = eri_subspace(eri, order)
    pair_terms, triple_terms = precompute_integral_terms(v, eri_cutoff, use_numba=use_numba_terms)

    first = groups[0]
    prefix = len(first)
    initial_eri = eri_to_dense(eri_subspace(v, range(prefix)))
    model = SpinHalfFermionChain(h[:prefix, :prefix], initial_eri, nelec=nelec)
    model.jordan_wigner(forward=False)
    primitive_qn = primitive_charge_labels(prefix)
    nroots = min(D, model.H.shape[0])
    energy, basis, block_qn = diagonalize_by_qn(
        model.H,
        primitive_qn,
        nroots,
        allowed_qn=feasible_qns(target_qn, prefix, norb),
        use_irrep_tensor=use_irrep_tensor,
        allow_empty=True,
    )
    if basis.shape[1] == 0:
        raise ValueError(f"No feasible retained states for initial clustered block {first}.")
    Hblock = np.diag(energy).astype(np.result_type(model.H, basis, complex))

    required_entries = (
        required_operator_entries(pair_terms, triple_terms, norb, prefix)
        if sparse_operator_table
        else None
    )
    primitive_table = make_operator_table(
        {
            'Cdu': model.Cdu,
            'Cdd': model.Cdd,
            'Cu': model.Cu,
            'Cd': model.Cd,
        },
        OPERATOR_PATTERNS,
        prefix,
        required=required_entries,
    )
    table = _rotate_operator_table_to_basis(
        primitive_table,
        basis,
        sparse_output=use_sparse_operator_projection,
    )
    residuals = build_triple_residuals_from_table(table, prefix, range(prefix, norb), triple_terms)

    collect_tensors = return_tensors or return_tensor_qns
    tensors = []
    tensor_qns = []
    initial_local_qn = supersite_charge_labels(prefix)
    if collect_tensors:
        tensors.append(basis.reshape((initial_local_qn.shape[0], basis.shape[1], 1)).copy())
    if return_tensor_qns:
        tensor_qns.append(
            {
                "row_qn": initial_local_qn.copy(),
                "right_qn_by_next": block_qn[:, None, :].copy(),
                "local_qn": initial_local_qn.copy(),
                "local_dim": initial_local_qn.shape[0],
                "orbitals": tuple(first),
                "growth_sites": prefix,
                "path": "reduced",
            }
        )

    for group in groups[1:]:
        local_norb = len(group)
        if local_norb == 1:
            Hblock, tensor, block_qn, branch_qn, local_qn, branch_energy = _append_one_site_reduced(
                Hblock,
                block_qn,
                table,
                residuals,
                prefix,
                D,
                h,
                v,
                pair_terms,
                target_qn,
                norb,
                use_irrep_tensor=use_irrep_tensor,
                use_irrep_blocks=use_irrep_blocks,
            )
            table_tensor = tensor
            plan = RotationPlan(table_tensor, block_qn)
            new_residuals = extend_triple_residuals(
                residuals,
                table,
                table_tensor,
                prefix,
                norb,
                triple_terms,
                block_qn,
                use_irrep_blocks,
                plan,
            )
            required_entries = (
                required_operator_entries(pair_terms, triple_terms, norb, prefix + 1)
                if sparse_operator_table
                else None
            )
            table = extend_operator_table(
                table,
                OPERATOR_PATTERNS,
                table_tensor,
                prefix,
                block_qn,
                use_irrep_blocks,
                plan,
                required=required_entries,
                sparse_output=use_sparse_operator_projection,
            )
            residuals = new_residuals
        else:
            (
                Hblock,
                tensor,
                table_tensor,
                block_qn,
                branch_qn,
                local_qn,
                branch_energy,
            ) = _append_pair_supersite_reduced(
                Hblock,
                block_qn,
                table,
                residuals,
                prefix,
                D,
                h,
                v,
                target_qn,
                norb,
                use_irrep_tensor=use_irrep_tensor,
            )
            plan = RotationPlan(table_tensor, block_qn)
            new_residuals = extend_triple_residuals_two_site(
                residuals,
                table,
                table_tensor,
                prefix,
                norb,
                triple_terms,
                block_qn,
                use_irrep_blocks,
                plan,
            )
            required_entries = (
                required_operator_entries(pair_terms, triple_terms, norb, prefix + 2)
                if sparse_operator_table
                else None
            )
            table = extend_pair_table(
                table,
                OPERATOR_PATTERNS,
                table_tensor,
                prefix,
                block_qn,
                use_irrep_blocks,
                plan,
                required=required_entries,
                sparse_output=use_sparse_operator_projection,
            )
            residuals = new_residuals

        prefix += local_norb
        if collect_tensors:
            tensors.append(tensor.copy())
        if return_tensor_qns:
            tensor_qns.append(
                {
                    "right_qn_by_next": branch_qn.copy(),
                    "local_qn": local_qn.copy(),
                    "local_dim": int(local_qn.shape[0]),
                    "orbitals": tuple(group),
                    "growth_sites": local_norb,
                    "branch_energy": branch_energy.copy(),
                    "path": "reduced",
                }
            )

    E, X, final_qn = charge_diagonalize(Hblock, block_qn, nstates, allowed_qn={target_qn})
    E = E[:nstates]
    X = X[:, :nstates]
    final_qn = final_qn[:nstates]
    results = [E, X]
    if return_tensors or return_tensor_qns:
        if not tensors:
            raise ValueError("internal error: clustered NARG did not build any tensors.")
        terminal_local_shape = tuple(int(dim) for dim in tensors[-1].shape[2:])
        terminal_local_dim = int(np.prod(terminal_local_shape))
        final_bond_dim = int(tensors[-1].shape[1])
        if X.shape[0] != terminal_local_dim * final_bond_dim:
            raise ValueError("final eigenvector shape is incompatible with clustered NARG tensors.")
    if return_tensors:
        coeff = np.asarray(X).reshape(terminal_local_dim, final_bond_dim, X.shape[1])
        results.append(tensors + [coeff])
    if return_tensor_qns:
        terminal_total_qn = qn_array(block_qn).reshape(terminal_local_dim, final_bond_dim, -1)
        results.append(
            {
                "factors": tensor_qns,
                "terminal_total_qn_by_site": terminal_total_qn.copy(),
                "local_qn": np.zeros((terminal_local_dim, LOCAL_QN.shape[1]), dtype=int)
                if terminal_local_dim == 1
                else qn_array(tensor_qns[-1]["local_qn"]).copy(),
                "local_shape": terminal_local_shape,
                "target_qn": np.asarray(target_qn, dtype=int),
                "final_qn": np.asarray(final_qn, dtype=int),
                "path": "reduced",
            }
        )
    return tuple(results) if len(results) > 2 else (E, X)


@dataclass
class TreeNRGNode:
    orbitals: tuple
    basis: np.ndarray
    qn: np.ndarray
    energy: np.ndarray
    children: tuple | None = None


@dataclass
class HierBlock:
    orbitals: tuple
    h: np.ndarray
    qn: np.ndarray
    ops: dict
    parity: np.ndarray
    energy: np.ndarray
    basis: np.ndarray | None = None
    children: tuple | None = None


def _restricted_integrals(h1e, eri, orbitals):
    orbitals = tuple(int(i) for i in orbitals)
    h = h1e[np.ix_(orbitals, orbitals)]
    v = eri_to_dense(eri_subspace(eri, orbitals))
    return h, v


def _check_hier_exact_size(orbitals, max_exact_orbitals):
    norb = len(tuple(orbitals))
    if max_exact_orbitals is not None and norb > int(max_exact_orbitals):
        raise ValueError(
            f"Hierarchical exact merge for {norb} orbitals would require primitive "
            f"dimension {4 ** norb}; increase max_exact_orbitals or use an "
            "MPO/environment hierarchical merge."
        )


def _local_tree_nrg_node(h1e, eri, orbitals, keep, target_qn, total_sites, max_exact_orbitals):
    _check_hier_exact_size(orbitals, max_exact_orbitals)
    h, v = _restricted_integrals(h1e, eri, orbitals)
    model = SpinHalfFermionChain(h, v)
    H = model.jordan_wigner()
    qn = primitive_charge_labels(len(orbitals))
    nroots = min(int(keep), H.shape[0])
    energy, coeff, kept_qn = diagonalize_by_qn(
        H,
        qn,
        nroots,
        allowed_qn=feasible_qns(target_qn, len(orbitals), total_sites),
        allow_empty=True,
    )
    if coeff.shape[1] == 0:
        raise ValueError(f"No feasible states for hierarchical block {orbitals}.")
    return TreeNRGNode(tuple(orbitals), coeff, kept_qn, energy)


def _merge_tree_nrg_nodes(left, right, h1e, eri, keep, target_qn, total_sites, max_exact_orbitals):
    orbitals = tuple(left.orbitals + right.orbitals)
    _check_hier_exact_size(orbitals, max_exact_orbitals)
    h, v = _restricted_integrals(h1e, eri, orbitals)
    model = SpinHalfFermionChain(h, v)
    H = model.jordan_wigner()
    projector = kron(
        csr_matrix(left.basis),
        csr_matrix(right.basis),
        format="csr",
    )
    projected = projector.conj().T @ (H @ projector)
    projected = projected.toarray() if issparse(projected) else np.asarray(projected)
    projected = 0.5 * (projected + projected.T.conj())
    projected_qn = (
        qn_array(left.qn)[:, None, :] + qn_array(right.qn)[None, :, :]
    ).reshape((-1, LOCAL_QN.shape[1]))
    nroots = min(int(keep), projected.shape[0])
    energy, coeff, kept_qn = diagonalize_by_qn(
        projected,
        projected_qn,
        nroots,
        allowed_qn=feasible_qns(target_qn, len(orbitals), total_sites),
        allow_empty=True,
    )
    if coeff.shape[1] == 0:
        raise ValueError(
            f"No feasible states after merging hierarchical blocks {left.orbitals} "
            f"and {right.orbitals}."
        )
    basis = projector @ coeff
    basis = basis.toarray() if issparse(basis) else np.asarray(basis)
    return TreeNRGNode(orbitals, basis, kept_qn, energy, children=(left, right))


def tree_nrg_kernel(
    h1e,
    eri,
    *,
    D=64,
    leaf_size=2,
    nstates=1,
    nelec=None,
    order=None,
    max_exact_orbitals=8,
    return_tree=False,
):
    """Balanced tree-NRG projection by recursively merging orbital blocks.

    This is not NARG: each node keeps the lowest charge-feasible
    states of an isolated orbital block, then parent nodes diagonalize the exact
    Hamiltonian of the union projected into the product of child bases.
    ``max_exact_orbitals`` caps the primitive Hamiltonian built at any node; set
    it to ``None`` only for deliberately small experiments.
    """
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    norb = h1e.shape[0]
    validate_eri_representation(eri, norb)
    if order is None:
        order = tuple(range(norb))
    else:
        order = tuple(int(i) for i in order)
    if sorted(order) != list(range(norb)):
        raise ValueError("order must be a permutation of all orbital indices.")
    if nelec is None:
        nelec = getattr(mol, "nelec")
    nelec = np.asarray(nelec, dtype=int).reshape(-1)
    target_nelec = int(np.sum(nelec))
    target_sz2 = int(nelec[0] - nelec[1]) if nelec.size == 2 else int(getattr(mol, "spin", 0))
    target_qn = (target_nelec, target_sz2)
    D = int(D)
    leaf_size = int(leaf_size)
    if D < 1:
        raise ValueError("D must be positive.")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive.")

    nodes = [
        _local_tree_nrg_node(
            h1e,
            eri,
            order[i:i + leaf_size],
            D,
            target_qn,
            norb,
            max_exact_orbitals,
        )
        for i in range(0, norb, leaf_size)
    ]
    levels = [nodes]
    while len(nodes) > 1:
        next_nodes = []
        idx = 0
        while idx < len(nodes):
            if idx + 1 == len(nodes):
                next_nodes.append(nodes[idx])
                idx += 1
            else:
                next_nodes.append(
                    _merge_tree_nrg_nodes(
                        nodes[idx],
                        nodes[idx + 1],
                        h1e,
                        eri,
                        D,
                        target_qn,
                        norb,
                        max_exact_orbitals,
                    )
                )
                idx += 2
        nodes = next_nodes
        levels.append(nodes)

    root = nodes[0]
    allowed_final = {target_qn}
    final_keep = []
    for idx, qn in enumerate(qn_array(root.qn)):
        if qn_key(qn) in allowed_final:
            final_keep.append(idx)
        if len(final_keep) >= int(nstates):
            break
    if len(final_keep) < int(nstates):
        raise ValueError(f"Hierarchical NARG found only {len(final_keep)} target-sector roots.")
    final_keep = np.asarray(final_keep, dtype=int)
    result = [root.energy[final_keep], root.basis[:, final_keep]]
    if return_tree:
        result.append({"root": root, "levels": levels, "order": order, "target_qn": target_qn, "method": "tree_nrg"})
    return tuple(result)


def _project_dense_operator(op, basis):
    op = dense_operator(op)
    return basis.conj().T @ (op @ basis)


def _block_parity_from_qn(qn):
    qn = qn_array(qn)
    return np.diag(1 - 2 * (qn[:, 0] % 2)).astype(complex)


def _local_hier_block(h1e, eri, orbitals, keep, target_qn, total_sites, max_state_orbitals):
    orbitals = tuple(int(i) for i in orbitals)
    h, v = _restricted_integrals(h1e, eri, orbitals)
    model = SpinHalfFermionChain(h, v)
    H = model.jordan_wigner()
    qn = primitive_charge_labels(len(orbitals))
    energy, coeff, kept_qn = diagonalize_by_qn(
        H,
        qn,
        min(int(keep), H.shape[0]),
        allowed_qn=feasible_qns(target_qn, len(orbitals), total_sites),
        allow_empty=True,
    )
    if coeff.shape[1] == 0:
        raise ValueError(f"No feasible states for hierarchical block {orbitals}.")

    source_ops = {
        "Cdu": model.Cdu,
        "Cdd": model.Cdd,
        "Cu": model.Cu,
        "Cd": model.Cd,
    }
    ops = {name: {} for name in source_ops}
    for pos, orb in enumerate(orbitals):
        for name, values in source_ops.items():
            ops[name][orb] = _project_dense_operator(values[pos], coeff)

    basis = coeff if max_state_orbitals is None or len(orbitals) <= int(max_state_orbitals) else None
    return HierBlock(
        orbitals=orbitals,
        h=np.diag(energy).astype(np.result_type(coeff, complex)),
        qn=kept_qn,
        ops=ops,
        parity=_block_parity_from_qn(kept_qn),
        energy=energy,
        basis=basis,
    )


def _hier_product_ops(left, right):
    left_orbs = set(left.orbitals)
    right_orbs = set(right.orbitals)
    eye_left = np.eye(left.h.shape[0], dtype=complex)
    eye_right = np.eye(right.h.shape[0], dtype=complex)
    ops = {name: {} for name in ("Cdu", "Cdd", "Cu", "Cd")}
    for name in ops:
        for orb in left.orbitals:
            ops[name][orb] = np.kron(left.ops[name][orb], eye_right)
        for orb in right.orbitals:
            ops[name][orb] = np.kron(left.parity, right.ops[name][orb])
    return ops, left_orbs, right_orbs


def _same_child(indices, left_orbs, right_orbs):
    in_left = [idx in left_orbs for idx in indices]
    return all(in_left) or not any(in_left)


def _merge_hier_blocks(left, right, h1e, eri, keep, target_qn, total_sites, max_state_orbitals):
    orbitals = tuple(left.orbitals + right.orbitals)
    left_dim = left.h.shape[0]
    right_dim = right.h.shape[0]
    eye_left = np.eye(left_dim, dtype=complex)
    eye_right = np.eye(right_dim, dtype=complex)
    H = np.kron(left.h, eye_right) + np.kron(eye_left, right.h)

    ops, left_orbs, right_orbs = _hier_product_ops(left, right)
    for p in orbitals:
        for q in orbitals:
            if _same_child((p, q), left_orbs, right_orbs):
                continue
            coeff = h1e[p, q]
            if coeff == 0:
                continue
            H += coeff * (ops["Cdu"][p] @ ops["Cu"][q] + ops["Cdd"][p] @ ops["Cd"][q])

    for p in orbitals:
        for q in orbitals:
            for r in orbitals:
                for s in orbitals:
                    if _same_child((p, q, r, s), left_orbs, right_orbs):
                        continue
                    coeff = 0.5 * eri_value(eri, p, q, r, s)
                    if coeff == 0:
                        continue
                    H += coeff * (
                        ops["Cdu"][p] @ ops["Cdu"][r] @ ops["Cu"][s] @ ops["Cu"][q]
                        + ops["Cdu"][p] @ ops["Cdd"][r] @ ops["Cd"][s] @ ops["Cu"][q]
                        + ops["Cdd"][p] @ ops["Cdu"][r] @ ops["Cu"][s] @ ops["Cd"][q]
                        + ops["Cdd"][p] @ ops["Cdd"][r] @ ops["Cd"][s] @ ops["Cd"][q]
                    )

    H = 0.5 * (H + H.T.conj())
    product_qn = (
        qn_array(left.qn)[:, None, :] + qn_array(right.qn)[None, :, :]
    ).reshape((-1, LOCAL_QN.shape[1]))
    energy, coeff, kept_qn = diagonalize_by_qn(
        H,
        product_qn,
        min(int(keep), H.shape[0]),
        allowed_qn=feasible_qns(target_qn, len(orbitals), total_sites),
        allow_empty=True,
    )
    if coeff.shape[1] == 0:
        raise ValueError(
            f"No feasible states after fusing hierarchical NARG blocks {left.orbitals} "
            f"and {right.orbitals}."
        )

    merged_ops = {name: {} for name in ops}
    for name, entries in ops.items():
        for orb, op in entries.items():
            merged_ops[name][orb] = coeff.conj().T @ (op @ coeff)

    basis = None
    if left.basis is not None and right.basis is not None:
        if max_state_orbitals is None or len(orbitals) <= int(max_state_orbitals):
            basis = np.kron(left.basis, right.basis) @ coeff

    return HierBlock(
        orbitals=orbitals,
        h=np.diag(energy).astype(np.result_type(coeff, complex)),
        qn=kept_qn,
        ops=merged_ops,
        parity=_block_parity_from_qn(kept_qn),
        energy=energy,
        basis=basis,
        children=(left, right),
    )


def hierarchical_kernel(
    h1e,
    eri,
    *,
    D=64,
    leaf_size=2,
    nstates=1,
    nelec=None,
    order=None,
    max_state_orbitals=8,
    return_tree=False,
):
    """Balanced hierarchical NARG using compressed block/operator fusion."""
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    norb = h1e.shape[0]
    validate_eri_representation(eri, norb)
    if order is None:
        order = tuple(range(norb))
    else:
        order = tuple(int(i) for i in order)
    if sorted(order) != list(range(norb)):
        raise ValueError("order must be a permutation of all orbital indices.")
    if nelec is None:
        nelec = getattr(mol, "nelec")
    nelec = np.asarray(nelec, dtype=int).reshape(-1)
    target_nelec = int(np.sum(nelec))
    target_sz2 = int(nelec[0] - nelec[1]) if nelec.size == 2 else int(getattr(mol, "spin", 0))
    target_qn = (target_nelec, target_sz2)
    D = int(D)
    leaf_size = int(leaf_size)
    if D < 1:
        raise ValueError("D must be positive.")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive.")

    nodes = [
        _local_hier_block(h1e, eri, order[i:i + leaf_size], D, target_qn, norb, max_state_orbitals)
        for i in range(0, norb, leaf_size)
    ]
    levels = [nodes]
    while len(nodes) > 1:
        next_nodes = []
        idx = 0
        while idx < len(nodes):
            if idx + 1 == len(nodes):
                next_nodes.append(nodes[idx])
                idx += 1
            else:
                next_nodes.append(
                    _merge_hier_blocks(
                        nodes[idx],
                        nodes[idx + 1],
                        h1e,
                        eri,
                        D,
                        target_qn,
                        norb,
                        max_state_orbitals,
                    )
                )
                idx += 2
        nodes = next_nodes
        levels.append(nodes)

    root = nodes[0]
    final_keep = []
    for idx, qn in enumerate(qn_array(root.qn)):
        if qn_key(qn) == target_qn:
            final_keep.append(idx)
        if len(final_keep) >= int(nstates):
            break
    if len(final_keep) < int(nstates):
        raise ValueError(f"Hierarchical NARG found only {len(final_keep)} target-sector roots.")
    final_keep = np.asarray(final_keep, dtype=int)
    x = root.basis[:, final_keep] if root.basis is not None else np.eye(root.h.shape[0])[:, final_keep]
    result = [root.energy[final_keep], x]
    if return_tree:
        result.append(
            {
                "root": root,
                "levels": levels,
                "order": order,
                "target_qn": target_qn,
                "method": "hierarchical_narg",
                "basis_is_primitive": root.basis is not None,
            }
        )
    return tuple(result)


def possible_qns(nsites):
    """All possible (Ne, 2Sz) labels for n spinful spatial orbitals."""
    return {
        (nup + ndn, nup - ndn)
        for nup in range(nsites + 1)
        for ndn in range(nsites + 1)
    }


def feasible_qns(target_qn, nsites, total_sites):
    """Partial-block (Ne, 2Sz) sectors that can still reach target_qn."""
    target_qn = qn_key(target_qn)
    remaining = total_sites - nsites
    possible_here = possible_qns(nsites)
    possible_rest = possible_qns(remaining)
    return {
        qn
        for qn in possible_here
        if (target_qn[0] - qn[0], target_qn[1] - qn[1]) in possible_rest
    }


def feasible_branch_qns(target_qn, block_nsites, total_sites, local_qn):
    """Block sectors allowed before adding a local state with local_qn."""
    target_qn = qn_key(target_qn)
    local_qn = qn_key(local_qn)
    remaining = total_sites - block_nsites - 1
    possible_block = possible_qns(block_nsites)
    possible_rest = possible_qns(remaining)
    return {
        qn
        for qn in possible_block
        if (
            target_qn[0] - local_qn[0] - qn[0],
            target_qn[1] - local_qn[1] - qn[1],
        )
        in possible_rest
    }


def feasible_multi_branch_qns(target_qn, block_nsites, total_sites, local_qn, nlocal_sites):
    """Block sectors allowed before adding a multi-orbital local branch."""
    target_qn = qn_key(target_qn)
    local_qn = qn_key(local_qn)
    remaining = total_sites - block_nsites - int(nlocal_sites)
    possible_block = possible_qns(block_nsites)
    possible_rest = possible_qns(remaining)
    return {
        qn
        for qn in possible_block
        if (
            target_qn[0] - local_qn[0] - qn[0],
            target_qn[1] - local_qn[1] - qn[1],
        )
        in possible_rest
    }


def charge_diagonalize(H, basis_qn, nroots, allowed_qn=None, allow_empty=False):
    """Diagonalize H by Abelian (Ne, 2Sz) IrrepSite sectors."""
    basis_qn = qn_array(basis_qn)
    allowed = None if allowed_qn is None else {qn_key(q) for q in allowed_qn}
    dim = len(basis_qn)
    site, sector_indices = irrep_site_from_qn(basis_qn)
    roots = []

    for irrep in site.irreps:
        q = qn_key(irrep.charge)
        if allowed is not None and q not in allowed:
            continue
        idx = sector_indices[irrep]
        if idx.size == 0:
            continue
        block = H[np.ix_(idx, idx)]
        if issparse(block):
            block = block.tocsr()
        else:
            block = np.asarray(block)
            block = 0.5 * (block + block.T.conj())

        if idx.size == 1:
            evals = np.array([block[0, 0] if not issparse(block) else block[0, 0]])
            evecs = np.ones((1, 1), dtype=np.result_type(block, complex))
        elif idx.size <= max(64, int(nroots) + 2):
            dense = block.toarray() if issparse(block) else block
            dense = 0.5 * (dense + dense.T.conj())
            evals, evecs = np.linalg.eigh(dense)
        else:
            k = min(int(nroots), idx.size - 2)
            evals, evecs = eigsh(block, k=k, which='SA')
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]

        for local_col, energy in enumerate(evals):
            roots.append((float(np.real(energy)), q, idx, evecs[:, local_col].copy()))

    if not roots and allow_empty:
        return _empty_sector_result(H, basis_qn)
    if not roots:
        raise ValueError(f"No states found for allowed charges {allowed_qn}.")

    roots.sort(key=lambda item: item[0])
    nselect = min(int(nroots), len(roots))
    energies = np.empty(nselect)
    vectors = np.zeros((dim, nselect), dtype=np.result_type(H, complex))
    qn = np.empty((nselect, basis_qn.shape[1]), dtype=int)
    for col, (energy, q, idx, vec) in enumerate(roots[:nselect]):
        energies[col] = energy
        qn[col] = q
        vectors[idx, col] = vec
    return energies, vectors, qn


def _submatrix(op, idx):
    return op[np.ix_(idx, idx)]


def _add_scaled_submatrix(block, op, idx, scale):
    if scale == 0:
        return block
    sub = _submatrix(op, idx)
    if issparse(block):
        return block + scale * (sub.tocsr() if issparse(sub) else csr_matrix(sub))
    sub = sub.toarray() if issparse(sub) else np.asarray(sub)
    return block + scale * sub


def _diagonalize_sector_blocks(blocks, basis_qn, nroots, *, dtype, allow_empty=False):
    """Collect lowest roots from preassembled Abelian scalar blocks."""
    basis_qn = qn_array(basis_qn)
    roots = []
    for q, idx, block in blocks:
        if idx.size == 0:
            continue
        if issparse(block):
            block = block.tocsr()
            block = 0.5 * (block + block.getH())
        else:
            block = np.asarray(block)
            block = 0.5 * (block + block.T.conj())

        if idx.size == 1:
            evals = np.array([block[0, 0] if not issparse(block) else block[0, 0]])
            evecs = np.ones((1, 1), dtype=np.result_type(block, complex))
        elif idx.size <= max(64, int(nroots) + 2):
            dense = block.toarray() if issparse(block) else block
            dense = 0.5 * (dense + dense.T.conj())
            evals, evecs = np.linalg.eigh(dense)
        else:
            k = min(int(nroots), idx.size - 2)
            evals, evecs = eigsh(block, k=k, which='SA')
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]

        for local_col, energy in enumerate(evals):
            roots.append((float(np.real(energy)), q, idx, evecs[:, local_col].copy()))

    if not roots and allow_empty:
        return _empty_sector_result(np.empty((0, 0), dtype=dtype), basis_qn)
    if not roots:
        raise ValueError("No states found for the requested Abelian block sectors.")

    roots.sort(key=lambda item: item[0])
    nselect = min(int(nroots), len(roots))
    energies = np.empty(nselect)
    vectors = np.zeros((len(basis_qn), nselect), dtype=np.result_type(dtype, complex))
    qn = np.empty((nselect, basis_qn.shape[1]), dtype=int)
    for col, (energy, q, idx, vec) in enumerate(roots[:nselect]):
        energies[col] = energy
        qn[col] = q
        vectors[idx, col] = vec
    return energies, vectors, qn


def branch_diagonalize_block_sparse(
    H0, pair_sums, nu, nd, basis_qn, nroots, allowed_qn=None, allow_empty=False
):
    """Diagonalize a branch Hamiltonian by assembling only requested charge blocks."""
    basis_qn = qn_array(basis_qn)
    allowed = None if allowed_qn is None else {qn_key(q) for q in allowed_qn}
    site, sector_indices = irrep_site_from_qn(basis_qn)
    blocks = []
    for irrep in site.irreps:
        q = qn_key(irrep.charge)
        if allowed is not None and q not in allowed:
            continue
        idx = sector_indices[irrep]
        block = _submatrix(H0, idx)
        if issparse(block):
            block = block.tocsr()
        else:
            block = np.asarray(block)
        block = _add_scaled_submatrix(block, pair_sums['density'], idx, nu + nd)
        block = _add_scaled_submatrix(block, pair_sums['exchange_u'], idx, -nu)
        block = _add_scaled_submatrix(block, pair_sums['exchange_d'], idx, -nd)
        blocks.append((q, idx, block))
    return _diagonalize_sector_blocks(
        blocks,
        basis_qn,
        nroots,
        dtype=np.result_type(operator_dtype(H0), *[operator_dtype(op) for op in pair_sums.values()], complex),
        allow_empty=allow_empty,
    )


def pad_branch(energies, vectors, qn, nroots):
    """Pad a symmetry sector branch so all local branches have equal width."""
    nroots = int(nroots)
    nstate = len(energies)
    if nstate >= nroots:
        return energies[:nroots], vectors[:, :nroots], qn[:nroots]

    padded_energies = np.full(nroots, 1e12, dtype=np.asarray(energies).dtype)
    padded_vectors = np.zeros((vectors.shape[0], nroots), dtype=vectors.dtype)
    padded_qn = np.full((nroots, qn.shape[1]), -10**9, dtype=qn.dtype)
    padded_energies[:nstate] = energies
    padded_vectors[:, :nstate] = vectors
    padded_qn[:nstate] = qn
    return padded_energies, padded_vectors, padded_qn


SINGLE_PATTERNS = (
    ('Cu',),
    ('Cd',),
    ('Cdu',),
    ('Cdd',),
)

PAIR_PATTERNS = tuple((left[0], right[0]) for left in SINGLE_PATTERNS for right in SINGLE_PATTERNS)

OPERATOR_PATTERNS = SINGLE_PATTERNS + PAIR_PATTERNS


def is_pair_factor_eri(eri):
    return np.ndim(eri) == 3


def validate_eri_representation(eri, norb):
    shape = np.shape(eri)
    dense_shape = (norb, norb, norb, norb)
    if shape == dense_shape:
        return
    if len(shape) == 3 and shape[1:] == (norb, norb):
        return
    raise ValueError(
        f"eri must have shape {dense_shape} or (naux, {norb}, {norb}); got {shape}."
    )


def eri_value(eri, p, q, r, s):
    if is_pair_factor_eri(eri):
        factors = np.asarray(eri)
        return np.dot(factors[:, p, q], factors[:, r, s])
    return eri[p, q, r, s]


def eri_subspace(eri, orbitals):
    orbitals = np.asarray(tuple(int(i) for i in orbitals), dtype=int)
    if is_pair_factor_eri(eri):
        factors = np.asarray(eri)
        return factors[:, orbitals][:, :, orbitals]
    return np.asarray(eri)[np.ix_(orbitals, orbitals, orbitals, orbitals)]


def eri_to_dense(eri):
    if is_pair_factor_eri(eri):
        factors = np.asarray(eri)
        return np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    return np.asarray(eri)



def required_operator_entries(pair_terms, triple_terms, total_sites, nsites):
    """Operator-table entries needed for all future pair/triple residual builds."""
    required = {pattern: set() for pattern in OPERATOR_PATTERNS}
    for name in ('Cu', 'Cd', 'Cdu', 'Cdd'):
        required[(name,)].update((i,) for i in range(nsites))

    for q in range(nsites, total_sites):
        for i, j, _coeff in pair_terms[q]['density']:
            if i < nsites and j < nsites:
                required[('Cdu', 'Cu')].add((i, j))
                required[('Cdd', 'Cd')].add((i, j))
        for i, j, _coeff in pair_terms[q]['exchange']:
            if i < nsites and j < nsites:
                required[('Cdu', 'Cu')].add((i, j))
                required[('Cdd', 'Cd')].add((i, j))
                required[('Cdd', 'Cu')].add((i, j))
        for i, j, _coeff in pair_terms[q]['v2b']:
            if i < nsites and j < nsites:
                required[('Cd', 'Cu')].add((i, j))
                required[('Cu', 'Cd')].add((i, j))
        for i, j, k, _coeff in triple_terms[q]:
            old = [idx for idx in (k, j, i) if idx < nsites]
            names_u = [name for name, idx in zip(('Cdu', 'Cdu', 'Cu'), (k, j, i)) if idx < nsites]
            names_ud = [name for name, idx in zip(('Cdu', 'Cdd', 'Cd'), (k, j, i)) if idx < nsites]
            names_d = [name for name, idx in zip(('Cdd', 'Cdu', 'Cu'), (k, j, i)) if idx < nsites]
            names_dd = [name for name, idx in zip(('Cdd', 'Cdd', 'Cd'), (k, j, i)) if idx < nsites]
            for names in (names_u, names_ud, names_d, names_dd):
                if names and tuple(names) in required:
                    required[tuple(names)].add(tuple(old))
    return required


def add_initial_spin_entries(required, nsites):
    """Ensure the sparse initial table can build block spin operators."""
    if required is None:
        return None
    for i in range(nsites):
        required[('Cdu', 'Cu')].add((i, i))
        required[('Cdd', 'Cd')].add((i, i))
        required[('Cdu', 'Cd')].add((i, i))
        required[('Cdd', 'Cu')].add((i, i))
    return required


def make_operator_table(single_ops, patterns, nsites, required=None):
    """Composite fermion operators in the current block basis."""
    table = {}
    for pattern in patterns:
        entries = {}
        if required is None:
            index_iter = np.ndindex((nsites,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            op = None
            for name, idx in zip(pattern, indices):
                factor = single_ops[name][idx]
                op = factor if op is None else op @ factor
            entries[indices] = op
        table[pattern] = entries
    return table


def is_zero_operator(op):
    if issparse(op):
        return op.nnz == 0
    return not np.any(op)


def zero_like_operator(reference):
    """Complex zero operator with the same shape/storage family as reference."""
    shape = reference.shape
    if issparse(reference):
        return csr_matrix(shape, dtype=complex)
    return np.zeros(shape, dtype=complex)


def keep_integral(value, cutoff):
    return abs(value) > cutoff


def maybe_print(verbose, *args):
    if verbose:
        print(*args)


@lru_cache(maxsize=None)
def local_product(names):
    local = {
        'Cu': cu,
        'Cd': cd,
        'Cdu': cdu,
        'Cdd': cdd,
        'JW': JW,
    }
    op = np.eye(4)
    for name in names:
        op = op @ local[name]
    return op


@lru_cache(maxsize=None)
def local_pair_product(names, roles):
    """Local operator product for two appended spatial orbitals.

    ``roles`` uses ``-1`` for an operator on the old block, ``0`` for the
    first appended orbital, and ``1`` for the second appended orbital.  Old
    block fermion operators carry parity strings through both appended
    orbitals, matching two consecutive Jordan-Wigner extensions.
    """
    local = {
        'Cu': cu,
        'Cd': cd,
        'Cdu': cdu,
        'Cdd': cdd,
        'JW': JW,
    }
    identity = np.eye(4)
    parity_pair = np.kron(JW, JW)
    op = np.eye(16)
    for name, role in zip(names, roles):
        if role == -1:
            factor = parity_pair
        elif role == 0:
            factor = np.kron(local[name], JW)
        elif role == 1:
            factor = np.kron(identity, local[name])
        else:
            raise ValueError(f"unknown two-site local role {role!r}")
        op = op @ factor
    return op


def pair_charge_labels():
    """(Ne, 2Sz) labels for a two-spatial-orbital local branch."""
    return (LOCAL_QN[:, None, :] + LOCAL_QN[None, :, :]).reshape((-1, LOCAL_QN.shape[1]))


def pair_spin_operators():
    """Spin operators for two appended spatial orbitals in local-pair order."""
    identity = np.eye(4)
    sz_pair = np.kron(Sz, identity) + np.kron(identity, Sz)
    sp_pair = np.kron(Sp, identity) + np.kron(identity, Sp)
    sm_pair = np.kron(Sm, identity) + np.kron(identity, Sm)
    s2_pair = sz_pair @ sz_pair + 0.5 * (sp_pair @ sm_pair + sm_pair @ sp_pair)
    return {
        's2': s2_pair,
        'sz': sz_pair,
        'sp': sp_pair,
        'sm': sm_pair,
    }


def _old_block_operator(table, names, indices, old_dim):
    names = tuple(names)
    indices = tuple(indices)
    if not names:
        return np.eye(old_dim, dtype=complex)
    if len(names) <= 2 and names in table and indices in table[names]:
        return dense_operator(table[names][indices])
    if len(names) > 2:
        raise NotImplementedError("three-old-operator pieces are handled by triple residuals.")
    op = np.eye(old_dim, dtype=complex)
    for name, idx in zip(names, indices):
        op = op @ dense_operator(table[(name,)][(idx,)])
    return op


def _split_two_site_pattern(pattern, indices, first_site):
    old_names = []
    old_indices = []
    local_names = []
    local_roles = []
    second_site = first_site + 1
    for name, idx in zip(pattern, indices):
        idx = int(idx)
        if idx < first_site:
            old_names.append(name)
            old_indices.append(idx)
            local_names.append(name)
            local_roles.append(-1)
        elif idx == first_site:
            local_names.append(name)
            local_roles.append(0)
        elif idx == second_site:
            local_names.append(name)
            local_roles.append(1)
        else:
            raise ValueError("two-site operator index lies outside the active prefix")
    return tuple(old_names), tuple(old_indices), tuple(local_names), tuple(local_roles)


def _two_site_operator_factor(table, pattern, indices, first_site, old_dim):
    old_names, old_indices, local_names, local_roles = _split_two_site_pattern(pattern, indices, first_site)
    block_op = _old_block_operator(table, old_names, old_indices, old_dim)
    local_op = local_pair_product(local_names, local_roles)
    return block_op, local_op


def add_local_kron_blocks(target4, block_op, local_op, coeff=1.0, *, check_zero=True):
    """Add ``coeff * kron(block_op, local_op)`` to a 4D Kronecker view."""
    if abs(coeff) <= 0:
        return
    local = np.asarray(local_op)
    if issparse(block_op):
        if block_op.nnz == 0:
            return
        if (
            _compiled_qchem is not None
            and target4.dtype == np.dtype(np.complex128)
            and _compiled_qchem.add_sparse_local_kron_blocks(target4, block_op, local, coeff)
        ):
            return
        block_coo = block_op.tocoo()
        rows, cols = np.nonzero(local)
        values = coeff * local[rows, cols]
        for row, col, value in zip(rows, cols, values):
            target4[block_coo.row, row, block_coo.col, col] += value * block_coo.data
        return
    block = dense_operator(block_op)
    if check_zero and not np.any(block):
        return
    if (
        _compiled_qchem is not None
        and target4.dtype == np.dtype(np.complex128)
        and _compiled_qchem.add_local_kron_blocks(target4, block, local, coeff)
    ):
        return
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return
    values = coeff * local[rows, cols]
    target4[:, rows, :, cols] += block[:, None, :] * values[None, :, None]


def add_local_kron_blocks_hc(target4, block_op, local_op, coeff=1.0):
    """Add ``X + X^dagger`` for ``X = coeff * kron(block_op, local_op)``."""
    if abs(coeff) <= 0:
        return
    local = np.asarray(local_op)
    if issparse(block_op):
        if block_op.nnz == 0:
            return
        if (
            _compiled_qchem is not None
            and target4.dtype == np.dtype(np.complex128)
            and _compiled_qchem.add_sparse_local_kron_blocks_hc(target4, block_op, local, coeff)
        ):
            return
        block_coo = block_op.tocoo()
        rows, cols = np.nonzero(local)
        values = coeff * local[rows, cols]
        for row, col, value in zip(rows, cols, values):
            target4[block_coo.row, row, block_coo.col, col] += value * block_coo.data
            target4[block_coo.col, col, block_coo.row, row] += np.conjugate(value * block_coo.data)
        return
    block = dense_operator(block_op)
    if not np.any(block):
        return
    if (
        _compiled_qchem is not None
        and target4.dtype == np.dtype(np.complex128)
        and _compiled_qchem.add_local_kron_blocks_hc(target4, block, local, coeff)
    ):
        return
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return
    block_h = block.T.conj()
    values = coeff * local[rows, cols]
    target4[:, rows, :, cols] += block[:, None, :] * values[None, :, None]
    target4[:, cols, :, rows] += block_h[:, None, :] * np.conjugate(values)[None, :, None]


def add_local_kron_blocks_lloo(target4, block_op, local_op, coeff=1.0, *, check_zero=True):
    """Add ``coeff * kron(block_op, local_op)`` to ``(local, local, old, old)`` blocks."""
    if abs(coeff) <= 0:
        return
    local = np.asarray(local_op)
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return
    if issparse(block_op):
        if block_op.nnz == 0:
            return
        block_coo = block_op.tocoo()
        values = coeff * local[rows, cols]
        for row, col, value in zip(rows, cols, values):
            target4[row, col, block_coo.row, block_coo.col] += value * block_coo.data
        return
    block = dense_operator(block_op)
    if check_zero and not np.any(block):
        return
    values = coeff * local[rows, cols]
    for row, col, value in zip(rows, cols, values):
        target4[row, col] += value * block


def add_local_kron_blocks_hc_lloo(target4, block_op, local_op, coeff=1.0):
    """Add ``X + X^dagger`` to ``(local, local, old, old)`` blocks."""
    if abs(coeff) <= 0:
        return
    local = np.asarray(local_op)
    rows, cols = np.nonzero(local)
    if rows.size == 0:
        return
    if issparse(block_op):
        if block_op.nnz == 0:
            return
        block_coo = block_op.tocoo()
        values = coeff * local[rows, cols]
        for row, col, value in zip(rows, cols, values):
            target4[row, col, block_coo.row, block_coo.col] += value * block_coo.data
            target4[col, row, block_coo.col, block_coo.row] += np.conjugate(value * block_coo.data)
        return
    block = dense_operator(block_op)
    if not np.any(block):
        return
    block_h = block.T.conj()
    values = coeff * local[rows, cols]
    for row, col, value in zip(rows, cols, values):
        target4[row, col] += value * block
        target4[col, row] += np.conjugate(value) * block_h


def hermitize_pair_blocks(Hpair4):
    """Hermitize a pair Hamiltonian in local-block form in place."""
    local_dim = Hpair4.shape[1]
    for bra_local in range(local_dim):
        block = Hpair4[:, bra_local, :, bra_local]
        block[:] = 0.5 * (block + block.T.conj())
        for ket_local in range(bra_local + 1, local_dim):
            upper = 0.5 * (
                Hpair4[:, bra_local, :, ket_local]
                + Hpair4[:, ket_local, :, bra_local].T.conj()
            )
            Hpair4[:, bra_local, :, ket_local] = upper
            Hpair4[:, ket_local, :, bra_local] = upper.T.conj()
    return Hpair4


def hermitize_pair_blocks_lloo(Hpair4):
    """Hermitize ``(local, local, old, old)`` pair-Hamiltonian blocks in place."""
    local_dim = Hpair4.shape[0]
    for bra_local in range(local_dim):
        block = Hpair4[bra_local, bra_local]
        block[:] = 0.5 * (block + block.T.conj())
        for ket_local in range(bra_local + 1, local_dim):
            upper = 0.5 * (
                Hpair4[bra_local, ket_local]
                + Hpair4[ket_local, bra_local].T.conj()
            )
            Hpair4[bra_local, ket_local] = upper
            Hpair4[ket_local, bra_local] = upper.T.conj()
    return Hpair4


def extend_pair_table(
    table,
    patterns,
    U,
    first_site,
    output_qn=None,
    use_irrep_blocks=False,
    plan=None,
    required=None,
    sparse_output=False,
):
    """Project composite operators when adding a two-orbital supersite."""
    old_dim = U.shape[0]
    out_dim = U.shape[1] * U.shape[2]
    new_nsites = int(first_site) + 2
    new_table = {}
    if output_qn is not None and plan is None:
        plan = RotationPlan(U, output_qn)
    local_cache = {}
    block_cache = {}
    applied_caches = {}

    def factor_cached(pattern, indices):
        old_names, old_indices, local_names, local_roles = _split_two_site_pattern(pattern, indices, first_site)
        block_key = (old_names, old_indices)
        block_op = block_cache.get(block_key)
        if block_op is None:
            block_op = _old_block_operator(table, old_names, old_indices, old_dim)
            block_cache[block_key] = block_op
        local_key = (local_names, local_roles)
        local_op = local_cache.get(local_key)
        if local_op is None:
            local_op = local_pair_product(local_names, local_roles)
            local_cache[local_key] = local_op
        return block_op, local_op

    def rotate_entry(block_op, local_op, shift):
        if plan is not None and output_qn is not None and shift is not None:
            cache_key = id(block_op)
            cached = applied_caches.get(cache_key)
            if cached is None or cached[0] is not block_op:
                cached = (block_op, {})
                applied_caches[cache_key] = cached
            rotated = plan.rotate_with_applied_cache(block_op, local_op, shift, cached[1])
            return csr_matrix(rotated) if sparse_output else rotated
        return rotate_symmetry(
            block_op,
            local_op,
            U,
            output_qn,
            shift,
            use_irrep_blocks,
            plan,
            sparse_output=sparse_output,
        )

    for pattern in patterns:
        entries = {}
        shift = pattern_qn_shift(pattern)
        allowed_by_qn = output_qn is None or has_qn_transition(output_qn, shift)
        if required is None:
            index_iter = np.ndindex((new_nsites,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            if not allowed_by_qn:
                shape = (out_dim, out_dim)
                entries[indices] = csr_matrix(shape, dtype=complex) if sparse_output else np.zeros(shape, dtype=complex)
                continue
            block_op, local_op = factor_cached(pattern, indices)
            if is_zero_operator(block_op) or is_zero_operator(local_op):
                shape = (out_dim, out_dim)
                dtype = np.result_type(block_op, local_op, U, complex)
                entries[indices] = csr_matrix(shape, dtype=dtype) if sparse_output else np.zeros(shape, dtype=dtype)
            else:
                entries[indices] = rotate_entry(block_op, local_op, shift)
        new_table[pattern] = entries
    return new_table


def project_two_site_operator(block_op, local_op, projector, *, sparse_output=False):
    """Project ``block_op tensor local_op`` with a general two-site projector.

    ``projector`` is ordered as ``projector[old_state, pair_local_state,
    new_state]``.  Unlike ``rotate_symmetry`` this does not require the output
    basis to factor as ``kept_state x pair_local_state``; rolling two-site NARG
    uses it to keep only ``D`` states per lookahead branch.
    """
    local_op = np.asarray(local_op)
    block = dense_operator(block_op)
    old_dim, local_dim, out_dim = projector.shape
    if block.shape != (old_dim, old_dim):
        raise ValueError(f"block_op has shape {block.shape}, expected {(old_dim, old_dim)}.")
    if local_op.shape != (local_dim, local_dim):
        raise ValueError(f"local_op has shape {local_op.shape}, expected {(local_dim, local_dim)}.")
    dtype = np.result_type(block, local_op, projector, complex)
    out = np.zeros((out_dim, out_dim), dtype=dtype)
    rows, cols = np.nonzero(local_op)
    for row, col in zip(rows, cols):
        coeff = local_op[row, col]
        if coeff == 0:
            continue
        left = projector[:, row, :]
        right = projector[:, col, :]
        out += coeff * (left.conj().T @ (block @ right))
    return csr_matrix(out) if sparse_output else out


def extend_operator_table_projector(
    table,
    patterns,
    projector,
    new_site,
    output_qn=None,
    required=None,
    sparse_output=False,
    consume=False,
):
    """Project one-site composite operators through a general isometry."""
    old_dim, local_dim, out_dim = projector.shape
    if local_dim != len(LOCAL_QN):
        raise ValueError(
            f"one-site projector local dimension must be {len(LOCAL_QN)}."
        )
    identity_block = eye(old_dim)
    identity_site = np.eye(local_dim)
    local = {"Cu": cu, "Cd": cd, "Cdu": cdu, "Cdd": cdd}
    new_table = {}

    source_uses = {}
    if consume:
        for pattern in patterns:
            index_iter = (
                np.ndindex((int(new_site) + 1,) * len(pattern))
                if required is None
                else sorted(required.get(pattern, ()))
            )
            for indices in index_iter:
                old_pattern = tuple(
                    name
                    for name, index in zip(pattern, indices)
                    if int(index) != int(new_site)
                )
                old_indices = tuple(
                    int(index) for index in indices if int(index) != int(new_site)
                )
                if old_pattern:
                    key = (old_pattern, old_indices)
                    source_uses[key] = source_uses.get(key, 0) + 1
        for pattern, entries in table.items():
            for indices in tuple(entries):
                if (pattern, indices) not in source_uses:
                    entries.pop(indices)

    def release_source(pattern, indices):
        if not consume or not pattern:
            return
        key = (pattern, indices)
        source_uses[key] -= 1
        if source_uses[key] == 0:
            table[pattern].pop(indices, None)

    for pattern in patterns:
        entries = {}
        shift = pattern_qn_shift(pattern)
        allowed = output_qn is None or has_qn_transition(output_qn, shift)
        if required is None:
            index_iter = np.ndindex((int(new_site) + 1,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            shape = (out_dim, out_dim)
            dtype = np.result_type(projector, complex)
            if not allowed:
                entries[indices] = (
                    csr_matrix(shape, dtype=dtype)
                    if sparse_output
                    else np.zeros(shape, dtype=dtype)
                )
                continue

            old_pattern = []
            old_indices = []
            local_op = identity_site
            for name, index in zip(pattern, indices):
                if int(index) == int(new_site):
                    local_op = local_op @ local[name]
                else:
                    old_pattern.append(name)
                    old_indices.append(int(index))
                    local_op = local_op @ JW
            block_op = (
                table[tuple(old_pattern)][tuple(old_indices)]
                if old_pattern
                else identity_block
            )
            if is_zero_operator(block_op) or is_zero_operator(local_op):
                dtype = np.result_type(block_op, local_op, projector, complex)
                entries[indices] = (
                    csr_matrix(shape, dtype=dtype)
                    if sparse_output
                    else np.zeros(shape, dtype=dtype)
                )
            else:
                entries[indices] = project_two_site_operator(
                    block_op,
                    local_op,
                    projector,
                    sparse_output=sparse_output,
                )
            release_source(tuple(old_pattern), tuple(old_indices))
        new_table[pattern] = entries
    if consume:
        table.clear()
    return new_table


def extend_triple_residuals_projector(
    residuals,
    table,
    projector,
    new_site,
    total_sites,
    triple_terms,
):
    """Project future weighted triple residuals through a general isometry."""
    old_dim = projector.shape[0]
    identity_block = eye(old_dim)
    identity_site = np.eye(len(LOCAL_QN))
    local = {"Cu": cu, "Cd": cd, "Cdu": cdu, "Cdd": cdd}
    projected = {}

    def project_term(pattern, indices):
        old_pattern = []
        old_indices = []
        local_op = identity_site
        for name, index in zip(pattern, indices):
            if int(index) == int(new_site):
                local_op = local_op @ local[name]
            else:
                old_pattern.append(name)
                old_indices.append(int(index))
                local_op = local_op @ JW
        block_op = (
            table[tuple(old_pattern)][tuple(old_indices)]
            if old_pattern
            else identity_block
        )
        return project_two_site_operator(block_op, local_op, projector)

    for future in range(int(new_site) + 1, int(total_sites)):
        old_u, old_d = residuals[future]
        value_u = project_two_site_operator(old_u, JW, projector)
        value_d = project_two_site_operator(old_d, JW, projector)
        for i, j, k, coefficient in triple_terms[future]:
            if i > new_site or j > new_site or k > new_site:
                continue
            if i != new_site and j != new_site and k != new_site:
                continue
            indices = (k, j, i)
            value_u += coefficient * project_term(
                ("Cdu", "Cdu", "Cu"), indices
            )
            value_u += coefficient * project_term(
                ("Cdu", "Cdd", "Cd"), indices
            )
            value_d += coefficient * project_term(
                ("Cdd", "Cdu", "Cu"), indices
            )
            value_d += coefficient * project_term(
                ("Cdd", "Cdd", "Cd"), indices
            )
        projected[future] = (value_u, value_d)
    return projected


def extend_triple_residuals_two_site_projector(
    residuals,
    table,
    projector,
    first_site,
    total_sites,
    triple_terms,
):
    """Project future triple residuals through a general two-site isometry."""
    first_site = int(first_site)
    second_site = first_site + 1
    old_dim = projector.shape[0]
    pair_parity = local_pair_product(("JW",), (-1,))
    projected = {}
    for future in range(second_site + 1, int(total_sites)):
        old_u, old_d = residuals[future]
        value_u = project_two_site_operator(old_u, pair_parity, projector)
        value_d = project_two_site_operator(old_d, pair_parity, projector)
        for i, j, k, coefficient in triple_terms[future]:
            if i > second_site or j > second_site or k > second_site:
                continue
            if i < first_site and j < first_site and k < first_site:
                continue
            indices = (k, j, i)
            for pattern, target in (
                (("Cdu", "Cdu", "Cu"), "u"),
                (("Cdu", "Cdd", "Cd"), "u"),
                (("Cdd", "Cdu", "Cu"), "d"),
                (("Cdd", "Cdd", "Cd"), "d"),
            ):
                block_op, local_op = _two_site_operator_factor(
                    table, pattern, indices, first_site, old_dim
                )
                term = coefficient * project_two_site_operator(
                    block_op, local_op, projector
                )
                if target == "u":
                    value_u += term
                else:
                    value_d += term
        projected[future] = (value_u, value_d)
    return projected


def extend_operator_table_two_site_projector(
    table,
    patterns,
    projector,
    first_site,
    output_qn=None,
    required=None,
    sparse_output=False,
):
    """Project composite operators with a general old x two-site projector."""
    old_dim, _local_dim, out_dim = projector.shape
    new_nsites = int(first_site) + 2
    new_table = {}

    for pattern in patterns:
        entries = {}
        shift = pattern_qn_shift(pattern)
        allowed_by_qn = output_qn is None or has_qn_transition(output_qn, shift)
        if required is None:
            index_iter = np.ndindex((new_nsites,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            dtype = np.result_type(projector, complex)
            if not allowed_by_qn:
                shape = (out_dim, out_dim)
                entries[indices] = csr_matrix(shape, dtype=dtype) if sparse_output else np.zeros(shape, dtype=dtype)
                continue
            block_op, local_op = _two_site_operator_factor(table, pattern, indices, first_site, old_dim)
            if is_zero_operator(block_op) or is_zero_operator(local_op):
                shape = (out_dim, out_dim)
                dtype = np.result_type(block_op, local_op, projector, complex)
                entries[indices] = csr_matrix(shape, dtype=dtype) if sparse_output else np.zeros(shape, dtype=dtype)
            else:
                entries[indices] = project_two_site_operator(
                    block_op,
                    local_op,
                    projector,
                    sparse_output=sparse_output,
                )
        new_table[pattern] = entries
    return new_table


def build_triple_residuals_from_table(table, nsites, future_sites, triple_terms):
    """Rebuild future-site triple residuals from the projected single operators."""
    reference = next(iter(table[('Cu',)].values()))
    zero = lambda: zero_like_operator(reference)
    residuals = {}
    for q in future_sites:
        v1u = zero()
        v1d = zero()
        for i, j, k, coeff in triple_terms[q]:
            if i >= nsites or j >= nsites or k >= nsites:
                continue
            cdu_k = table[('Cdu',)][(k,)]
            cdd_k = table[('Cdd',)][(k,)]
            cdu_j = table[('Cdu',)][(j,)]
            cdd_j = table[('Cdd',)][(j,)]
            cu_i = table[('Cu',)][(i,)]
            cd_i = table[('Cd',)][(i,)]
            scalar = cdu_j @ cu_i + cdd_j @ cd_i
            v1u += coeff * (cdu_k @ scalar)
            v1d += coeff * (cdd_k @ scalar)
        residuals[q] = (v1u, v1d)
    return residuals


def extend_spin_operators_two_site(spin_ops, U):
    """Project total spin operators through a two-site NARG tensor."""
    old_dim = spin_ops['sz'].shape[0]
    iblock = eye(old_dim)
    ipair = np.eye(U.shape[2])
    pair_ops = pair_spin_operators()

    sz_total = rotate(spin_ops['sz'], ipair, U) + rotate(iblock, pair_ops['sz'], U)
    sp_total = rotate(spin_ops['sp'], ipair, U) + rotate(iblock, pair_ops['sp'], U)
    sm_total = rotate(spin_ops['sm'], ipair, U) + rotate(iblock, pair_ops['sm'], U)
    s2_total = (
        rotate(spin_ops['s2'], ipair, U)
        + rotate(iblock, pair_ops['s2'], U)
        + 2.0 * rotate(spin_ops['sz'], pair_ops['sz'], U)
        + rotate(spin_ops['sp'], pair_ops['sm'], U)
        + rotate(spin_ops['sm'], pair_ops['sp'], U)
    )
    return {
        's2': 0.5 * (s2_total + s2_total.T.conj()),
        'sz': sz_total,
        'sp': sp_total,
        'sm': sm_total,
    }


if njit is not None:
    @njit(cache=False)
    def _collect_integral_terms_numba(eri, cutoff):
        L = eri.shape[0]
        ndensity = 0
        nexchange = 0
        nv2b = 0
        ntriple = 0

        for q in range(L):
            for i in range(q):
                for j in range(q):
                    if abs(eri[i, j, q, q]) > cutoff:
                        ndensity += 1
                    if abs(eri[i, q, q, j]) > cutoff:
                        nexchange += 1
                    if abs(eri[q, i, q, j]) > cutoff:
                        nv2b += 1
                    for k in range(q):
                        if abs(eri[k, q, j, i]) > cutoff:
                            ntriple += 1

        density_idx = np.empty((ndensity, 3), dtype=np.int64)
        exchange_idx = np.empty((nexchange, 3), dtype=np.int64)
        v2b_idx = np.empty((nv2b, 3), dtype=np.int64)
        triple_idx = np.empty((ntriple, 4), dtype=np.int64)
        density_val = np.empty(ndensity, dtype=np.complex128)
        exchange_val = np.empty(nexchange, dtype=np.complex128)
        v2b_val = np.empty(nv2b, dtype=np.complex128)
        triple_val = np.empty(ntriple, dtype=np.complex128)

        idensity = 0
        iexchange = 0
        iv2b = 0
        itriple = 0
        for q in range(L):
            for i in range(q):
                for j in range(q):
                    coeff = eri[i, j, q, q]
                    if abs(coeff) > cutoff:
                        density_idx[idensity, 0] = q
                        density_idx[idensity, 1] = i
                        density_idx[idensity, 2] = j
                        density_val[idensity] = coeff
                        idensity += 1

                    coeff = eri[i, q, q, j]
                    if abs(coeff) > cutoff:
                        exchange_idx[iexchange, 0] = q
                        exchange_idx[iexchange, 1] = i
                        exchange_idx[iexchange, 2] = j
                        exchange_val[iexchange] = coeff
                        iexchange += 1

                    coeff = eri[q, i, q, j]
                    if abs(coeff) > cutoff:
                        v2b_idx[iv2b, 0] = q
                        v2b_idx[iv2b, 1] = i
                        v2b_idx[iv2b, 2] = j
                        v2b_val[iv2b] = 0.5 * coeff
                        iv2b += 1

                    for k in range(q):
                        coeff = eri[k, q, j, i]
                        if abs(coeff) > cutoff:
                            triple_idx[itriple, 0] = q
                            triple_idx[itriple, 1] = i
                            triple_idx[itriple, 2] = j
                            triple_idx[itriple, 3] = k
                            triple_val[itriple] = coeff
                            itriple += 1

        return (
            density_idx, density_val,
            exchange_idx, exchange_val,
            v2b_idx, v2b_val,
            triple_idx, triple_val,
        )
else:
    _collect_integral_terms_numba = None


def _empty_integral_terms(L):
    pair_terms = {}
    triple_terms = {}
    for q in range(L):
        pair_terms[q] = {
            'density': [],
            'exchange': [],
            'v2b': [],
        }
        triple_terms[q] = []
    return pair_terms, triple_terms


def _precompute_integral_terms_python(eri, cutoff=0.0):
    L = eri.shape[1] if is_pair_factor_eri(eri) else eri.shape[0]
    pair_terms, triple_terms = _empty_integral_terms(L)
    for q in range(L):
        for i in range(q):
            for j in range(q):
                coeff = eri_value(eri, i, j, q, q)
                if keep_integral(coeff, cutoff):
                    pair_terms[q]['density'].append((i, j, coeff))

                coeff = eri_value(eri, i, q, q, j)
                if keep_integral(coeff, cutoff):
                    pair_terms[q]['exchange'].append((i, j, coeff))

                coeff = eri_value(eri, q, i, q, j)
                if keep_integral(coeff, cutoff):
                    pair_terms[q]['v2b'].append((i, j, 0.5 * coeff))

                for k in range(q):
                    coeff = eri_value(eri, k, q, j, i)
                    if keep_integral(coeff, cutoff):
                        triple_terms[q].append((i, j, k, coeff))
    return pair_terms, triple_terms


def _precompute_integral_terms_factors(pair_factors, cutoff=0.0):
    pair_factors = np.asarray(pair_factors)
    L = pair_factors.shape[1]
    pair_terms, triple_terms = _empty_integral_terms(L)
    cutoff = float(cutoff)
    for q in range(L):
        old = slice(0, q)
        diagonal = pair_factors[:, q, q]
        density = np.einsum(
            "Pij,P->ij",
            pair_factors[:, old, old],
            diagonal,
            optimize=True,
        )
        exchange = np.einsum(
            "Pi,Pj->ij",
            pair_factors[:, old, q],
            pair_factors[:, q, old],
            optimize=True,
        )
        pair = np.einsum(
            "Pi,Pj->ij",
            pair_factors[:, q, old],
            pair_factors[:, q, old],
            optimize=True,
        )
        triples = np.einsum(
            "Pk,Pji->ijk",
            pair_factors[:, old, q],
            pair_factors[:, old, old],
            optimize=True,
        )

        for i, j in np.argwhere(np.abs(density) > cutoff):
            pair_terms[q]["density"].append((int(i), int(j), density[i, j]))
        for i, j in np.argwhere(np.abs(exchange) > cutoff):
            pair_terms[q]["exchange"].append((int(i), int(j), exchange[i, j]))
        for i, j in np.argwhere(np.abs(pair) > cutoff):
            pair_terms[q]["v2b"].append((int(i), int(j), 0.5 * pair[i, j]))
        for i, j, k in np.argwhere(np.abs(triples) > cutoff):
            triple_terms[q].append((int(i), int(j), int(k), triples[i, j, k]))
    return pair_terms, triple_terms


def _integral_arrays_to_terms(arrays, L):
    density_idx, density_val, exchange_idx, exchange_val, v2b_idx, v2b_val, triple_idx, triple_val = arrays
    pair_terms, triple_terms = _empty_integral_terms(L)

    for (q, i, j), coeff in zip(density_idx, density_val):
        pair_terms[int(q)]['density'].append((int(i), int(j), coeff))
    for (q, i, j), coeff in zip(exchange_idx, exchange_val):
        pair_terms[int(q)]['exchange'].append((int(i), int(j), coeff))
    for (q, i, j), coeff in zip(v2b_idx, v2b_val):
        pair_terms[int(q)]['v2b'].append((int(i), int(j), coeff))
    for (q, i, j, k), coeff in zip(triple_idx, triple_val):
        triple_terms[int(q)].append((int(i), int(j), int(k), coeff))

    return pair_terms, triple_terms


def _precompute_integral_terms_cython(eri, cutoff=0.0):
    if getattr(_abelian_cython, "precompute_integral_terms", None) is not None:
        return _abelian_cython.precompute_integral_terms(eri, float(cutoff))
    arrays = _abelian_cython.collect_integral_terms(eri, float(cutoff))
    return _integral_arrays_to_terms(arrays, eri.shape[0])


def _precompute_integral_terms_numba(eri, cutoff=0.0):
    arrays = _collect_integral_terms_numba(np.asarray(eri), float(cutoff))
    return _integral_arrays_to_terms(arrays, eri.shape[0])


def precompute_integral_terms(eri, cutoff=0.0, use_numba='auto'):
    if is_pair_factor_eri(eri):
        return _precompute_integral_terms_factors(eri, cutoff)
    if use_numba == 'auto':
        use_numba = eri.shape[0] >= 12
    if (
        use_numba
        and _abelian_cython is not None
        and getattr(_abelian_cython, "CYTHON_AVAILABLE", False)
    ):
        try:
            return _precompute_integral_terms_cython(eri, cutoff)
        except Exception:
            pass
    if use_numba and _collect_integral_terms_numba is not None:
        try:
            return _precompute_integral_terms_numba(eri, cutoff)
        except Exception:
            pass
    return _precompute_integral_terms_python(eri, cutoff)


def build_pair_sums(table, pair_terms, q):
    dim = next(iter(table[('Cu',)].values())).shape[0]
    zero = csr_matrix((dim, dim), dtype=complex)
    density = zero.copy()
    exchange_u = zero.copy()
    exchange_d = zero.copy()
    v2a = zero.copy()
    v2b = zero.copy()

    for i, j, coeff in pair_terms[q]['density']:
        density += coeff * (
            table[('Cdu', 'Cu')][(i, j)] + table[('Cdd', 'Cd')][(i, j)]
        )

    for i, j, coeff in pair_terms[q]['exchange']:
        exchange_u += coeff * table[('Cdu', 'Cu')][(i, j)]
        exchange_d += coeff * table[('Cdd', 'Cd')][(i, j)]
        v2a -= coeff * table[('Cdd', 'Cu')][(i, j)]

    for i, j, coeff in pair_terms[q]['v2b']:
        v2b += coeff * (
            table[('Cd', 'Cu')][(i, j)] - table[('Cu', 'Cd')][(i, j)]
        )

    return {
        'density': density,
        'exchange_u': exchange_u,
        'exchange_d': exchange_d,
        'v2a': v2a,
        'v2b': v2b,
    }


def build_pair_sums_irrep(table, pair_terms, q):
    """Build pair-sum operators from an IrrepTensor operator table."""
    site = next(iter(table.values())).bra
    scalar = (0, 0)
    v2a_charge = operator_pattern_shift(('Cdd', 'Cu'))
    v2b_charge = operator_pattern_shift(('Cd', 'Cu'))

    density_terms = []
    exchange_u_terms = []
    exchange_d_terms = []
    v2a_terms = []
    v2b_terms = []

    for i, j, coeff in pair_terms[q]['density']:
        density_terms.append((coeff, table[(('Cdu', 'Cu'), (i, j))]))
        density_terms.append((coeff, table[(('Cdd', 'Cd'), (i, j))]))

    for i, j, coeff in pair_terms[q]['exchange']:
        exchange_u_terms.append((coeff, table[(('Cdu', 'Cu'), (i, j))]))
        exchange_d_terms.append((coeff, table[(('Cdd', 'Cd'), (i, j))]))
        v2a_terms.append((-coeff, table[(('Cdd', 'Cu'), (i, j))]))

    for i, j, coeff in pair_terms[q]['v2b']:
        v2b_terms.append((coeff, table[(('Cd', 'Cu'), (i, j))]))
        v2b_terms.append((-coeff, table[(('Cu', 'Cd'), (i, j))]))

    return {
        'density': sum_tensor_terms(site, scalar, density_terms),
        'exchange_u': sum_tensor_terms(site, scalar, exchange_u_terms),
        'exchange_d': sum_tensor_terms(site, scalar, exchange_d_terms),
        'v2a': sum_tensor_terms(site, v2a_charge, v2a_terms),
        'v2b': sum_tensor_terms(site, v2b_charge, v2b_terms),
    }


def branch_hamiltonian(H0, pair_sums, nu, nd):
    return (
        H0.copy()
        + (nu + nd) * pair_sums['density']
        - nu * pair_sums['exchange_u']
        - nd * pair_sums['exchange_d']
    )


def branch_hamiltonian_irrep(H0, pair_sums, nu, nd):
    """Branch Hamiltonian from scalar IrrepTensor pieces."""
    return add_tensors(
        H0,
        scale_tensor(pair_sums['density'], nu + nd),
        scale_tensor(pair_sums['exchange_u'], -nu),
        scale_tensor(pair_sums['exchange_d'], -nd),
    )


def single_site_hamiltonian_from_integrals(h1e, eri, site_id):
    site_id = int(site_id)
    return (
        h1e[site_id, site_id] * (cdu @ cu + cdd @ cd)
        + eri_value(eri, site_id, site_id, site_id, site_id) * Nu @ Nd
    )


def _rotate_operator_table_to_basis(table, basis, *, sparse_output=False):
    """Rotate primitive block operator table into a retained block basis."""
    basis = np.asarray(basis)
    basis_h = basis.conj().T
    rotated = {}
    for pattern, entries in table.items():
        new_entries = {}
        for indices, op in entries.items():
            projected = basis_h @ (dense_operator(op) @ basis)
            new_entries[indices] = csr_matrix(projected) if sparse_output else projected
        rotated[pattern] = new_entries
    return rotated


def _compressed_superblock_one_site_lloo(
    H0,
    input_qn,
    table,
    residuals,
    site_id,
    h1e,
    eri,
    pair_terms,
):
    """Exact retained-block-plus-orbital Hamiltonian in local-major blocks."""
    site_id = int(site_id)
    old_dim = H0.shape[0]
    local_dim = len(LOCAL_QN)
    identity_block = np.eye(old_dim)
    identity_site = np.eye(local_dim)
    h_site = single_site_hamiltonian_from_integrals(h1e, eri, site_id)
    pair_sums = build_pair_sums(table, pair_terms, site_id)
    dtype = np.result_type(H0, h1e, getattr(eri, "dtype", eri), complex)
    h_full = np.kron(dense_operator(H0), identity_site).astype(dtype, copy=False)
    h_full += np.kron(identity_block, h_site)
    h_full += np.kron(dense_operator(pair_sums["density"]), Ntot)
    h_full -= np.kron(dense_operator(pair_sums["exchange_u"]), Nu)
    h_full -= np.kron(dense_operator(pair_sums["exchange_d"]), Nd)

    op_lists = operator_lists(table)
    v1u = zero_like_operator(H0)
    v1d = zero_like_operator(H0)
    for index in range(site_id):
        v1u += h1e[index, site_id] * dense_operator(op_lists["Cdu"][index])
        v1d += h1e[index, site_id] * dense_operator(op_lists["Cdd"][index])
    if site_id in residuals:
        v1u += residuals[site_id][0]
        v1d += residuals[site_id][1]

    def add_hermitian(block_op, local_op):
        nonlocal h_full
        term = np.kron(dense_operator(block_op), np.asarray(local_op))
        h_full += term + term.T.conj()

    add_hermitian(v1u, JW @ cu)
    add_hermitian(v1d, JW @ cd)
    add_hermitian(pair_sums["v2a"], cdu @ cd)
    add_hermitian(pair_sums["v2b"], cdu @ cdd)

    v3u = zero_like_operator(H0)
    v3d = zero_like_operator(H0)
    for index in range(site_id):
        v3u += eri_value(eri, index, site_id, site_id, site_id) * dense_operator(
            op_lists["Cdu"][index]
        )
        v3d += eri_value(eri, index, site_id, site_id, site_id) * dense_operator(
            op_lists["Cdd"][index]
        )
    add_hermitian(v3u, JW @ Nd @ cu)
    add_hermitian(v3d, JW @ Nu @ cd)

    h_full = 0.5 * (h_full + h_full.T.conj())
    h_lloo = h_full.reshape(
        old_dim, local_dim, old_dim, local_dim
    ).transpose(1, 3, 0, 2)
    primitive_qn = (
        qn_array(input_qn)[:, None, :] + LOCAL_QN[None, :, :]
    ).reshape((-1, LOCAL_QN.shape[1]))
    return h_lloo, primitive_qn


class OneSiteHamiltonianAction:
    """Matrix-free sum of old-block/local-site Kronecker terms."""

    def __init__(self, old_dim, local_dim, dtype=complex):
        self.old_dim = int(old_dim)
        self.local_dim = int(local_dim)
        self.dtype = np.dtype(np.result_type(dtype, complex))
        self.terms = []

    def add(self, block_op, local_op, coefficient=1.0, *, hermitian=False):
        if abs(coefficient) == 0 or is_zero_operator(block_op) or is_zero_operator(local_op):
            return
        self.terms.append(
            (block_op, np.asarray(local_op), coefficient, bool(hermitian))
        )

    def diagonal_block(self, local_state):
        local_state = int(local_state)
        block = np.zeros((self.old_dim, self.old_dim), dtype=self.dtype)
        for block_op, local_op, coefficient, hermitian in self.terms:
            local_value = local_op[local_state, local_state]
            if local_value == 0:
                continue
            dense = dense_operator(block_op)
            value = coefficient * local_value
            block += value * dense
            if hermitian:
                block += np.conjugate(value) * dense.T.conj()
        return 0.5 * (block + block.T.conj())

    def diagonal(self):
        """Return the product-basis diagonal without materializing the action."""
        diagonal = np.zeros(
            (self.old_dim, self.local_dim),
            dtype=self.dtype,
        )
        for block_op, local_op, coefficient, hermitian in self.terms:
            block_diagonal = np.asarray(block_op.diagonal()).reshape(-1)
            local_diagonal = np.diag(local_op)
            value = coefficient * block_diagonal[:, None] * local_diagonal[None, :]
            diagonal += value
            if hermitian:
                diagonal += value.conj()
        return diagonal.reshape(-1)

    def matmat(self, vectors):
        vectors = np.asarray(vectors)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        states = vectors.reshape(self.old_dim, self.local_dim, -1)
        result = np.zeros_like(states, dtype=np.result_type(states, self.dtype))
        for block_op, local_op, coefficient, hermitian in self.terms:
            rows, cols = np.nonzero(local_op)
            for row, col in zip(rows, cols):
                value = coefficient * local_op[row, col]
                result[:, row, :] += value * (block_op @ states[:, col, :])
                if hermitian:
                    result[:, col, :] += np.conjugate(value) * (
                        block_op.T.conj() @ states[:, row, :]
                    )
        flattened = result.reshape(self.old_dim * self.local_dim, -1)
        return flattened[:, 0] if was_vector else flattened


def _compressed_superblock_one_site_action(
    H0,
    input_qn,
    table,
    residuals,
    site_id,
    h1e,
    eri,
    pair_terms,
):
    """Matrix-free retained-block-plus-orbital Hamiltonian."""
    site_id = int(site_id)
    old_dim = H0.shape[0]
    local_dim = len(LOCAL_QN)
    identity_block = np.eye(old_dim)
    identity_site = np.eye(local_dim)
    action = OneSiteHamiltonianAction(
        old_dim,
        local_dim,
        dtype=np.result_type(H0, h1e, getattr(eri, "dtype", eri), complex),
    )
    action.add(H0, identity_site)
    action.add(identity_block, single_site_hamiltonian_from_integrals(h1e, eri, site_id))
    pair_sums = build_pair_sums(table, pair_terms, site_id)
    action.add(pair_sums["density"], Ntot)
    action.add(pair_sums["exchange_u"], Nu, -1.0)
    action.add(pair_sums["exchange_d"], Nd, -1.0)

    op_lists = operator_lists(table)
    v1u = zero_like_operator(H0)
    v1d = zero_like_operator(H0)
    for index in range(site_id):
        v1u += h1e[index, site_id] * dense_operator(op_lists["Cdu"][index])
        v1d += h1e[index, site_id] * dense_operator(op_lists["Cdd"][index])
    if site_id in residuals:
        v1u += residuals[site_id][0]
        v1d += residuals[site_id][1]
    action.add(v1u, JW @ cu, hermitian=True)
    action.add(v1d, JW @ cd, hermitian=True)
    action.add(pair_sums["v2a"], cdu @ cd, hermitian=True)
    action.add(pair_sums["v2b"], cdu @ cdd, hermitian=True)

    v3u = zero_like_operator(H0)
    v3d = zero_like_operator(H0)
    for index in range(site_id):
        v3u += eri_value(eri, index, site_id, site_id, site_id) * dense_operator(
            op_lists["Cdu"][index]
        )
        v3d += eri_value(eri, index, site_id, site_id, site_id) * dense_operator(
            op_lists["Cdd"][index]
        )
    action.add(v3u, JW @ Nd @ cu, hermitian=True)
    action.add(v3d, JW @ Nu @ cd, hermitian=True)
    primitive_qn = (
        qn_array(input_qn)[:, None, :] + LOCAL_QN[None, :, :]
    ).reshape((-1, LOCAL_QN.shape[1]))
    return action, primitive_qn


def detached_frame_transition_projector(
    h_lloo,
    input_qn,
    *,
    frame_dim,
    bond_dim,
    allowed_frame_qn=None,
    allowed_output_qn=None,
    use_irrep_tensor=False,
    adapt_tol=None,
    max_frame_rank=None,
    expand_dim=1,
):
    r"""Construct a nested one-site isometry from detached conditional frames.

    The ordinary conditional branches are first combined into a common frame
    ``W`` without changing their joint span.  The detached candidate space is

    .. math::

        \mathcal V_{\rm det} = \operatorname{span}(W) \otimes \mathbb C^d.

    It contains the matched ordinary NARG space

    .. math::

        \mathcal V_{\rm std}
        = \operatorname{span}\{W_{s,a} \otimes |s\rangle\}.

    The matched branch columns are retained as anchors; additional columns are
    selected from low-energy detached eigenvectors.  Thus the retained space
    remains a variational enlargement of the ordinary conditional space.
    """
    matrix_free = isinstance(h_lloo, OneSiteHamiltonianAction)
    input_qn = qn_array(input_qn)
    if matrix_free:
        local_dim = local_dim_ket = h_lloo.local_dim
        old_dim = old_dim_ket = h_lloo.old_dim
        h_dtype = h_lloo.dtype
    else:
        h_lloo = np.asarray(h_lloo)
        local_dim, local_dim_ket, old_dim, old_dim_ket = h_lloo.shape
        h_dtype = h_lloo.dtype
        if local_dim != local_dim_ket or old_dim != old_dim_ket:
            raise ValueError(
                "h_lloo must have shape (local_dim, local_dim, old_dim, old_dim)."
            )
    if local_dim != len(LOCAL_QN) or len(input_qn) != old_dim:
        raise ValueError("detached frames are incompatible with the local or block labels.")
    frame_dim = int(frame_dim)
    bond_dim = int(bond_dim)
    if frame_dim < 1 or bond_dim < 1:
        raise ValueError("frame_dim and bond_dim must be positive.")
    if adapt_tol is not None and float(adapt_tol) < 0.0:
        raise ValueError("adapt_tol must be non-negative when provided.")
    expand_dim = int(expand_dim)
    if expand_dim < 1:
        raise ValueError("expand_dim must be positive.")
    max_frame_rank = (
        old_dim if max_frame_rank is None else min(int(max_frame_rank), old_dim)
    )
    if max_frame_rank < 1:
        raise ValueError("max_frame_rank must be positive.")

    allowed_frames = (
        None
        if allowed_frame_qn is None
        else {qn_key(qn) for qn in allowed_frame_qn}
    )
    valid = valid_qn_mask(input_qn)
    sectors = {
        charge: np.flatnonzero(valid & np.all(input_qn == np.asarray(charge), axis=1))
        for charge in sorted({qn_key(row) for row in input_qn[valid]})
        if allowed_frames is None or charge in allowed_frames
    }
    frames = []
    frame_labels = []
    branch_ranks = []
    output_charges = (
        None
        if allowed_output_qn is None
        else {qn_key(qn) for qn in allowed_output_qn}
    )

    for local_state in range(local_dim):
        branch_allowed = allowed_frames
        if output_charges is not None:
            local_charge = LOCAL_QN[local_state]
            branch_allowed = {
                qn_key(np.asarray(charge) - local_charge)
                for charge in output_charges
            }
            if allowed_frames is not None:
                branch_allowed &= allowed_frames
        branch_h = (
            h_lloo.diagonal_block(local_state)
            if matrix_free
            else h_lloo[local_state, local_state]
        )
        branch_h = 0.5 * (branch_h + branch_h.T.conj())
        candidates = []
        for charge, rows in sectors.items():
            if branch_allowed is not None and charge not in branch_allowed:
                continue
            h_sector = branch_h[np.ix_(rows, rows)]
            energies, vectors = np.linalg.eigh(h_sector)
            candidates.extend(
                (float(np.real(energy)), charge, rows, vectors[:, column])
                for column, energy in enumerate(energies)
            )
        candidates.sort(key=lambda item: item[0])
        selected = candidates[:frame_dim]
        branch = np.zeros(
            (old_dim, len(selected)),
            dtype=np.result_type(h_dtype, complex),
        )
        labels = np.empty((len(selected), input_qn.shape[1]), dtype=int)
        for column, (_energy, charge, rows, vector) in enumerate(selected):
            branch[rows, column] = vector
            labels[column] = charge
        frames.append(branch)
        frame_labels.append(labels)
        branch_ranks.append(branch.shape[1])

    joint_columns = {}
    for branch, labels in zip(frames, frame_labels):
        for column, charge in enumerate(labels):
            joint_columns.setdefault(qn_key(charge), []).append(branch[:, column])
    common_frames = []
    common_labels = []
    for charge in sorted(joint_columns):
        matrix = np.column_stack(joint_columns[charge])
        rows = sectors[charge]
        left, singular_values, _right = np.linalg.svd(
            matrix[rows],
            full_matrices=False,
        )
        if singular_values.size == 0:
            continue
        threshold = (
            np.finfo(singular_values.dtype).eps
            * max(matrix.shape)
            * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        if rank:
            sector_frame = np.zeros(
                (old_dim, rank),
                dtype=np.result_type(matrix, complex),
            )
            sector_frame[rows] = left[:, :rank]
            common_frames.append(sector_frame)
            common_labels.extend([charge] * rank)
    if not common_frames:
        raise ValueError("no feasible detached-frame states were found.")
    frame_basis = np.column_stack(common_frames)
    frame_qn = np.asarray(common_labels, dtype=int)
    frame_rank = frame_basis.shape[1]
    h_full = None
    if not matrix_free:
        h_full = np.ascontiguousarray(
            h_lloo.transpose(2, 0, 3, 1)
        ).reshape(old_dim * local_dim, old_dim * local_dim)

    def apply_full(vectors):
        if matrix_free:
            return h_lloo.matmat(vectors)
        return h_full @ np.asarray(vectors)

    def diagonalize_projected_basis(basis, basis_qn, nroots):
        """Lowest charge-resolved Ritz pairs without a full projected matrix."""
        basis_qn = qn_array(basis_qn)
        allowed = (
            None
            if allowed_output_qn is None
            else {qn_key(charge) for charge in allowed_output_qn}
        )
        roots = []
        for charge in sorted({qn_key(row) for row in basis_qn}):
            if allowed is not None and charge not in allowed:
                continue
            columns = np.flatnonzero(
                np.all(basis_qn == np.asarray(charge), axis=1)
            )
            sector_basis = basis[:, columns]
            keep = min(int(nroots), columns.size)

            def sector_action(vector):
                return sector_basis.conj().T @ apply_full(sector_basis @ vector)

            if columns.size <= 64 or keep >= columns.size - 1:
                block = sector_basis.conj().T @ apply_full(sector_basis)
                block = 0.5 * (block + block.T.conj())
                values, vectors = np.linalg.eigh(block)
            else:
                operator = LinearOperator(
                    (columns.size, columns.size),
                    matvec=sector_action,
                    matmat=sector_action,
                    dtype=np.result_type(h_dtype, basis, complex),
                )
                try:
                    values, vectors = eigsh(
                        operator,
                        k=keep,
                        which="SA",
                        tol=1.0e-12,
                        maxiter=max(1000, 20 * columns.size),
                    )
                except ArpackNoConvergence:
                    block = sector_basis.conj().T @ apply_full(sector_basis)
                    block = 0.5 * (block + block.T.conj())
                    values, vectors = np.linalg.eigh(block)
                order = np.argsort(values)
                values = values[order]
                vectors = vectors[:, order]
            roots.extend(
                (float(np.real(value)), charge, columns, vectors[:, column])
                for column, value in enumerate(values[:keep])
            )
        roots.sort(key=lambda item: item[0])
        selected = roots[: min(int(nroots), len(roots))]
        energies = np.asarray([item[0] for item in selected], dtype=float)
        vectors = np.zeros(
            (basis.shape[1], len(selected)),
            dtype=np.result_type(h_dtype, basis, complex),
        )
        labels = np.empty((len(selected), basis_qn.shape[1]), dtype=int)
        for column, (_energy, charge, rows, vector) in enumerate(selected):
            vectors[rows, column] = vector
            labels[column] = charge
        return energies, vectors, labels

    anchor_rank = sum(branch_ranks)
    anchor = np.zeros(
        (old_dim, local_dim, anchor_rank),
        dtype=np.result_type(h_dtype, complex),
    )
    anchor_qn = np.empty((anchor_rank, input_qn.shape[1]), dtype=int)
    offset = 0
    for local_state, (branch, labels) in enumerate(zip(frames, frame_labels)):
        width = branch.shape[1]
        columns = slice(offset, offset + width)
        anchor[:, local_state, columns] = branch
        anchor_qn[columns] = labels + LOCAL_QN[local_state]
        offset += width
    anchor_matrix = anchor.reshape(old_dim * local_dim, anchor_rank)
    h_anchor = anchor_matrix.conj().T @ apply_full(anchor_matrix)
    h_anchor = 0.5 * (h_anchor + h_anchor.T.conj())
    anchor_lowest_energy = float(np.linalg.eigvalsh(h_anchor)[0])

    def solve(current_basis, current_qn):
        current_rank = current_basis.shape[1]
        base = np.zeros(
            (old_dim, local_dim, local_dim * current_rank),
            dtype=np.result_type(current_basis, complex),
        )
        detached_qn = np.empty(
            (local_dim * current_rank, input_qn.shape[1]),
            dtype=int,
        )
        for local_state in range(local_dim):
            columns = slice(
                local_state * current_rank,
                (local_state + 1) * current_rank,
            )
            base[:, local_state, columns] = current_basis
            detached_qn[columns] = current_qn + LOCAL_QN[local_state]
        base_matrix = base.reshape(
            old_dim * local_dim, local_dim * current_rank
        )
        anchor_coefficients = base_matrix.conj().T @ anchor_matrix
        anchor_error = float(
            np.linalg.norm(
                anchor_matrix - base_matrix @ anchor_coefficients,
                ord="fro",
            )
        )
        if anchor_error > 1.0e-8:
            raise RuntimeError("detached frame does not contain the standard NARG anchor.")

        search_roots = min(
            base_matrix.shape[1],
            bond_dim + anchor_rank,
        )
        _trial_energies, trial_vectors, trial_qn = diagonalize_projected_basis(
            base_matrix,
            detached_qn,
            search_roots,
        )
        anchor_by_charge = {}
        for column, charge in enumerate(anchor_qn):
            anchor_by_charge.setdefault(qn_key(charge), []).append(
                anchor_coefficients[:, column]
            )
        retained_sectors = []
        retained_labels = []
        for charge in sorted(anchor_by_charge):
            matrix = np.column_stack(anchor_by_charge[charge])
            rows = np.flatnonzero(
                np.all(
                    detached_qn == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            left, singular_values, _right = np.linalg.svd(
                matrix[rows],
                full_matrices=False,
            )
            threshold = (
                np.finfo(singular_values.dtype).eps
                * max(matrix.shape)
                * singular_values[0]
            )
            rank = int(np.count_nonzero(singular_values > threshold))
            sector_basis = np.zeros(
                (matrix.shape[0], rank),
                dtype=np.result_type(matrix, complex),
            )
            sector_basis[rows] = left[:, :rank]
            retained_sectors.append(sector_basis)
            retained_labels.extend([charge] * rank)
        retained_basis = np.column_stack(retained_sectors)
        retained_labels = list(retained_labels)
        target_rank = min(bond_dim, base_matrix.shape[1])
        for column, charge in enumerate(trial_qn):
            if retained_basis.shape[1] >= target_rank:
                break
            vector = trial_vectors[:, column].copy()
            for _ in range(2):
                vector -= retained_basis @ (retained_basis.conj().T @ vector)
            norm = np.linalg.norm(vector)
            if norm <= 1.0e-12:
                continue
            retained_basis = np.column_stack(
                (retained_basis, vector / norm)
            )
            retained_labels.append(qn_key(charge))
        retained_labels = np.asarray(retained_labels, dtype=int)
        retained_label_corrections = 0
        detached_sectors = {
            charge: np.flatnonzero(
                np.all(
                    detached_qn == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            for charge in sorted({qn_key(row) for row in detached_qn})
        }
        for column in range(retained_basis.shape[1]):
            weights = {
                charge: float(
                    np.linalg.norm(retained_basis[rows, column]) ** 2
                )
                for charge, rows in detached_sectors.items()
            }
            charge, weight = max(weights.items(), key=lambda item: item[1])
            leakage = max(0.0, sum(weights.values()) - weight)
            if leakage > 1.0e-8:
                raise RuntimeError(
                    "detached retained basis mixes Abelian sectors: "
                    f"column={column}, leakage={leakage:.3e}."
                )
            retained_label_corrections += int(
                qn_key(retained_labels[column]) != charge
            )
            retained_labels[column] = charge
        for charge in sorted({qn_key(row) for row in retained_labels}):
            columns = np.flatnonzero(
                np.all(
                    retained_labels == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            rows = np.flatnonzero(
                np.all(
                    detached_qn == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            sector = retained_basis[np.ix_(rows, columns)]
            overlap = sector.conj().T @ sector
            overlap = 0.5 * (overlap + overlap.T.conj())
            values, vectors = np.linalg.eigh(overlap)
            if values.size and values[0] <= 1.0e-12:
                raise RuntimeError(
                    "detached retained basis lost rank during sector cleanup."
                )
            inverse_sqrt = (
                vectors * (1.0 / np.sqrt(values))[None, :]
            ) @ vectors.T.conj()
            retained_basis[:, columns] = 0.0
            retained_basis[np.ix_(rows, columns)] = sector @ inverse_sqrt
        retained_physical = base_matrix @ retained_basis
        energies, rotation, output_qn = diagonalize_projected_basis(
            retained_physical,
            retained_labels,
            retained_basis.shape[1],
        )
        if rotation.shape[1] == 0:
            raise ValueError(
                "no feasible output states remain in the detached-frame space."
            )
        projector = (retained_physical @ rotation).reshape(
            old_dim, local_dim, rotation.shape[1]
        )
        primitive_qn = (
            input_qn[:, None, :] + LOCAL_QN[None, :, :]
        ).reshape(old_dim * local_dim, -1)
        flat_projector = projector.reshape(old_dim * local_dim, -1)
        inferred_qn = np.empty_like(output_qn)
        label_corrections = retained_label_corrections
        maximum_sector_leakage = 0.0
        primitive_sectors = {
            charge: np.flatnonzero(
                np.all(
                    primitive_qn == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            for charge in sorted({qn_key(row) for row in primitive_qn})
        }
        for column in range(flat_projector.shape[1]):
            weights = {
                charge: float(
                    np.linalg.norm(flat_projector[rows, column]) ** 2
                )
                for charge, rows in primitive_sectors.items()
            }
            charge, weight = max(weights.items(), key=lambda item: item[1])
            leakage = max(0.0, sum(weights.values()) - weight)
            maximum_sector_leakage = max(maximum_sector_leakage, leakage)
            if leakage > 1.0e-8:
                raise RuntimeError(
                    "detached projector mixes Abelian output sectors: "
                    f"column={column}, leakage={leakage:.3e}."
                )
            inferred_qn[column] = charge
            label_corrections += int(
                qn_key(output_qn[column]) != charge
            )
        output_qn = inferred_qn
        for charge in sorted({qn_key(row) for row in output_qn}):
            columns = np.flatnonzero(
                np.all(
                    output_qn == np.asarray(charge, dtype=int),
                    axis=1,
                )
            )
            rows = primitive_sectors[charge]
            sector = flat_projector[np.ix_(rows, columns)]
            overlap = 0.5 * (
                sector.conj().T @ sector
                + (sector.conj().T @ sector).T.conj()
            )
            values, vectors = np.linalg.eigh(overlap)
            if values.size and values[0] <= 1.0e-12:
                raise RuntimeError(
                    "detached projector lost rank during sector cleanup."
                )
            inverse_sqrt = (
                vectors * (1.0 / np.sqrt(values))[None, :]
            ) @ vectors.T.conj()
            flat_projector[:, columns] = 0.0
            flat_projector[np.ix_(rows, columns)] = sector @ inverse_sqrt
        projector = flat_projector.reshape(old_dim, local_dim, -1)
        h_new = flat_projector.conj().T @ apply_full(flat_projector)
        h_new = 0.5 * (h_new + h_new.T.conj())
        retained_matrix = projector.reshape(old_dim * local_dim, -1)
        retained_anchor_error = float(
            np.linalg.norm(
                anchor_matrix
                - retained_matrix @ (retained_matrix.conj().T @ anchor_matrix),
                ord="fro",
            )
        )
        if retained_anchor_error > 1.0e-8:
            raise RuntimeError("retained detached space lost the standard NARG anchor.")
        return (
            h_new,
            projector,
            output_qn,
            energies,
            anchor_error,
            retained_anchor_error,
            label_corrections,
            maximum_sector_leakage,
        )

    def residual_expansion(current_basis, projector, output_qn):
        nstate = projector.shape[2]
        action = apply_full(projector.reshape(old_dim * local_dim, nstate)).reshape(
            old_dim, local_dim, nstate
        )
        grouped = {}
        residual_sq = 0.0
        for local_state in range(local_dim):
            residual = action[:, local_state, :]
            residual -= current_basis @ (current_basis.conj().T @ residual)
            residual_sq += float(np.vdot(residual, residual).real)
            for state in range(nstate):
                charge = qn_key(output_qn[state] - LOCAL_QN[local_state])
                rows = sectors.get(charge)
                if rows is None:
                    continue
                vector = residual[rows, state]
                if np.linalg.norm(vector) > 1.0e-14:
                    grouped.setdefault(charge, []).append(vector)

        candidates = []
        for charge, vectors in grouped.items():
            matrix = np.column_stack(vectors)
            left, singular_values, _right = np.linalg.svd(
                matrix, full_matrices=False
            )
            rows = sectors[charge]
            for column, singular_value in enumerate(singular_values):
                if singular_value <= 1.0e-14:
                    continue
                vector = np.zeros(old_dim, dtype=np.result_type(matrix, complex))
                vector[rows] = left[:, column]
                candidates.append((float(singular_value), charge, vector))
        candidates.sort(key=lambda item: item[0], reverse=True)
        residual_norm = np.sqrt(residual_sq / max(nstate, 1))
        return residual_norm, candidates

    initial_frame_rank = frame_rank
    residual_history = []
    adapt_iterations = 0
    while True:
        (
            h_new,
            projector,
            output_qn,
            energies,
            anchor_error,
            retained_anchor_error,
            label_corrections,
            maximum_sector_leakage,
        ) = solve(frame_basis, frame_qn)
        residual_norm, candidates = residual_expansion(
            frame_basis, projector, output_qn
        )
        residual_history.append(float(residual_norm))
        if (
            adapt_tol is None
            or residual_norm <= float(adapt_tol)
            or frame_basis.shape[1] >= max_frame_rank
            or not candidates
        ):
            break

        additions = []
        addition_qn = []
        room = max_frame_rank - frame_basis.shape[1]
        for _singular_value, charge, vector in candidates:
            vector = vector - frame_basis @ (frame_basis.conj().T @ vector)
            if additions:
                added = np.column_stack(additions)
                vector = vector - added @ (added.conj().T @ vector)
            norm = np.linalg.norm(vector)
            if norm <= 1.0e-12:
                continue
            additions.append(vector / norm)
            addition_qn.append(charge)
            if len(additions) >= min(expand_dim, room):
                break
        if not additions:
            break
        frame_basis = np.column_stack((frame_basis, *additions))
        frame_qn = np.vstack(
            (frame_qn, np.asarray(addition_qn, dtype=int))
        )
        adapt_iterations += 1

    frame_rank = frame_basis.shape[1]
    overlap = frame_basis.conj().T @ frame_basis
    orthogonality_error = float(
        np.linalg.norm(overlap - np.eye(frame_rank), ord=np.inf)
    )
    if orthogonality_error > 1.0e-8:
        raise RuntimeError("detached conditional frames are not orthonormal.")
    diagnostics = {
        "frame_dim": frame_dim,
        "branch_ranks": tuple(branch_ranks),
        "anchor_rank": anchor_rank,
        "anchor_inclusion_error": anchor_error,
        "retained_anchor_error": retained_anchor_error,
        "sector_label_corrections": label_corrections,
        "maximum_sector_leakage": maximum_sector_leakage,
        "anchor_lowest_energy": anchor_lowest_energy,
        "frame_rank": frame_rank,
        "initial_frame_rank": initial_frame_rank,
        "adapted_rank": frame_rank - initial_frame_rank,
        "adapt_iterations": adapt_iterations,
        "detached_dim": local_dim * frame_rank,
        "retained_dim": projector.shape[2],
        "orthogonality_error": orthogonality_error,
        "frame_residual_norm": residual_history[-1],
        "frame_residual_history": tuple(residual_history),
        "lowest_energy": float(energies[0]),
        "detached_improvement": anchor_lowest_energy - float(energies[0]),
    }
    return h_new, projector, output_qn, diagnostics


def _compressed_superblock_two_site_lloo(H0, input_qn, table, residuals, first_site, h1e, eri):
    """Exact compressed Hamiltonian as ``(local, local, old, old)`` blocks."""
    first_site = int(first_site)
    old_dim = H0.shape[0]
    d = len(LOCAL_QN)
    local_dim = d * d
    dtype = np.result_type(H0, h1e, getattr(eri, "dtype", eri), complex)
    Hpair4 = np.zeros((local_dim, local_dim, old_dim, old_dim), dtype=dtype)
    H0_dense = dense_operator(H0).astype(dtype, copy=False)
    for local_id in range(local_dim):
        Hpair4[local_id, local_id] += H0_dense
    second_site = first_site + 1
    block_cache = {}
    block_zero_cache = {}
    local_cache = {}
    term_coeffs = {}

    def factor_cached(old_names, old_indices, local_names, local_roles):
        block_key = (old_names, old_indices)
        block_op = block_cache.get(block_key)
        if block_op is None:
            block_op = _old_block_operator(table, old_names, old_indices, old_dim)
            block_cache[block_key] = block_op
        local_key = (local_names, local_roles)
        local_op = local_cache.get(local_key)
        if local_op is None:
            local_op = local_pair_product(local_names, local_roles)
            local_cache[local_key] = local_op
        return block_op, local_op

    def accumulate_term(coeff, pattern, indices):
        if abs(coeff) <= 0:
            return
        if all(int(idx) < first_site for idx in indices):
            return
        old_names, old_indices, local_names, local_roles = _split_two_site_pattern(pattern, indices, first_site)
        if len(old_names) > 2:
            return
        key = (old_names, old_indices, local_names, local_roles)
        term_coeffs[key] = term_coeffs.get(key, 0.0) + coeff

    def block_is_zero(block_op):
        key = id(block_op)
        cached = block_zero_cache.get(key)
        if cached is None:
            cached = is_zero_operator(block_op)
            block_zero_cache[key] = cached
        return cached

    for a in range(second_site + 1):
        for b in range(second_site + 1):
            coeff = h1e[a, b]
            accumulate_term(coeff, ('Cdu', 'Cu'), (a, b))
            accumulate_term(coeff, ('Cdd', 'Cd'), (a, b))

    for p_idx in range(second_site + 1):
        for q_idx in range(second_site + 1):
            for r_idx in range(second_site + 1):
                for s_idx in range(second_site + 1):
                    coeff = 0.5 * eri_value(eri, p_idx, q_idx, r_idx, s_idx)
                    accumulate_term(coeff, ('Cdu', 'Cdu', 'Cu', 'Cu'), (p_idx, r_idx, s_idx, q_idx))
                    accumulate_term(coeff, ('Cdu', 'Cdd', 'Cd', 'Cu'), (p_idx, r_idx, s_idx, q_idx))
                    accumulate_term(coeff, ('Cdd', 'Cdu', 'Cu', 'Cd'), (p_idx, r_idx, s_idx, q_idx))
                    accumulate_term(coeff, ('Cdd', 'Cdd', 'Cd', 'Cd'), (p_idx, r_idx, s_idx, q_idx))

    for (old_names, old_indices, local_names, local_roles), coeff in term_coeffs.items():
        if abs(coeff) <= 0:
            continue
        block_op, local_op = factor_cached(old_names, old_indices, local_names, local_roles)
        if block_is_zero(block_op):
            continue
        add_local_kron_blocks_lloo(Hpair4, block_op, local_op, coeff, check_zero=False)

    for site_id, role in ((first_site, 0), (second_site, 1)):
        if site_id in residuals:
            v1u, v1d = residuals[site_id]
            local_u = local_pair_product(('JW', 'JW', 'JW', 'Cu'), (-1, -1, -1, role))
            local_d = local_pair_product(('JW', 'JW', 'JW', 'Cd'), (-1, -1, -1, role))
            add_local_kron_blocks_hc_lloo(Hpair4, v1u, local_u)
            add_local_kron_blocks_hc_lloo(Hpair4, v1d, local_d)

    hermitize_pair_blocks_lloo(Hpair4)
    local_qn = pair_charge_labels()
    output_qn = (qn_array(input_qn)[:, None, :] + local_qn[None, :, :]).reshape((-1, LOCAL_QN.shape[1]))
    return Hpair4, output_qn, local_qn


def _compressed_superblock_two_site(H0, input_qn, table, residuals, first_site, h1e, eri):
    """Exact compressed Hamiltonian for a retained block plus two new orbitals."""
    Hpair4_lloo, output_qn, local_qn = _compressed_superblock_two_site_lloo(
        H0, input_qn, table, residuals, first_site, h1e, eri
    )
    local_dim, _, old_dim, _ = Hpair4_lloo.shape
    Hpair = np.ascontiguousarray(Hpair4_lloo.transpose(2, 0, 3, 1)).reshape(
        old_dim * local_dim,
        old_dim * local_dim,
    )
    return Hpair, output_qn, local_qn


def project_pair_hamiltonian(Hpair4, branch_tensor):
    """Project a pair superblock Hamiltonian without forming the zero-padded basis."""
    old_dim, local_dim, old_dim_ket, local_dim_ket = Hpair4.shape
    if old_dim != old_dim_ket or local_dim != local_dim_ket:
        raise ValueError("Hpair4 must have shape (old_dim, local_dim, old_dim, local_dim).")
    if branch_tensor.shape[0] != old_dim or branch_tensor.shape[2] != local_dim:
        raise ValueError("branch_tensor is incompatible with Hpair4.")
    keep = branch_tensor.shape[1]
    dtype = np.result_type(Hpair4, branch_tensor, complex)
    Hnew = np.zeros((local_dim * keep, local_dim * keep), dtype=dtype)
    for bra_local in range(local_dim):
        left = branch_tensor[:, :, bra_local]
        row = slice(bra_local * keep, (bra_local + 1) * keep)
        for ket_local in range(bra_local, local_dim):
            block = Hpair4[:, bra_local, :, ket_local]
            if not np.any(block):
                continue
            right = branch_tensor[:, :, ket_local]
            col = slice(ket_local * keep, (ket_local + 1) * keep)
            projected = left.conj().T @ (block @ right)
            Hnew[row, col] = projected
            if ket_local != bra_local:
                Hnew[col, row] = projected.T.conj()
    return Hnew


def project_pair_hamiltonian_lloo(Hpair4, branch_tensor):
    """Project local-major pair blocks without forming a zero-padded basis."""
    local_dim, local_dim_ket, old_dim, old_dim_ket = Hpair4.shape
    if old_dim != old_dim_ket or local_dim != local_dim_ket:
        raise ValueError("Hpair4 must have shape (local_dim, local_dim, old_dim, old_dim).")
    if branch_tensor.shape[0] != old_dim or branch_tensor.shape[2] != local_dim:
        raise ValueError("branch_tensor is incompatible with Hpair4.")
    keep = branch_tensor.shape[1]
    dtype = np.result_type(Hpair4, branch_tensor, complex)
    Hnew = np.zeros((local_dim * keep, local_dim * keep), dtype=dtype)
    for bra_local in range(local_dim):
        left = branch_tensor[:, :, bra_local]
        row = slice(bra_local * keep, (bra_local + 1) * keep)
        for ket_local in range(bra_local, local_dim):
            block = Hpair4[bra_local, ket_local]
            if not np.any(block):
                continue
            right = branch_tensor[:, :, ket_local]
            col = slice(ket_local * keep, (ket_local + 1) * keep)
            projected = left.conj().T @ (block @ right)
            Hnew[row, col] = projected
            if ket_local != bra_local:
                Hnew[col, row] = projected.T.conj()
    return Hnew


def conditional_cc_transition_projector(
    Hpair4,
    primitive_qn,
    projector,
    output_qn,
    *,
    level_shift=0.0,
    response_tol=1.0e-10,
    max_responses=None,
):
    r"""Dress a retained projector with symmetry-preserving responses.

    For every retained Abelian sector, this constructs first-order discarded
    responses

    .. math::

        (H_{QQ} - E_i + \Delta)t_i = -H_{QP}c_i,

    and Ritz-compresses ``span(P, Q T)`` back to the original number of
    retained states.  The resulting isometry is general in the local branch
    labels, so it carries the cross-branch amplitudes ``T^{st}`` missing from a
    branch-local conditional rotation. Both dense local-major Hamiltonians and
    matrix-free one-site actions are supported.
    """
    primitive_qn = qn_array(primitive_qn)
    output_qn = qn_array(output_qn)
    projector = np.asarray(projector)
    old_dim, local_dim, out_dim = projector.shape
    full_dim = old_dim * local_dim
    matrix_free = isinstance(Hpair4, OneSiteHamiltonianAction)
    if matrix_free:
        if Hpair4.old_dim != old_dim or Hpair4.local_dim != local_dim:
            raise ValueError("Hamiltonian action is incompatible with the projector dimensions.")
        h_dtype = Hpair4.dtype

        def apply_full(vectors):
            return Hpair4.matmat(vectors)
    else:
        Hpair4 = np.asarray(Hpair4)
        if Hpair4.shape != (local_dim, local_dim, old_dim, old_dim):
            raise ValueError(
                "Hpair4 must have shape (local_dim, local_dim, old_dim, old_dim)."
            )
        h_dtype = Hpair4.dtype
        h_full = np.ascontiguousarray(Hpair4.transpose(2, 0, 3, 1)).reshape(
            full_dim, full_dim
        )
        h_full = 0.5 * (h_full + h_full.T.conj())

        def apply_full(vectors):
            return h_full @ np.asarray(vectors)
    if len(primitive_qn) != full_dim or len(output_qn) != out_dim:
        raise ValueError("quantum-number labels are inconsistent with projector dimensions.")
    if float(response_tol) < 0.0:
        raise ValueError("response_tol must be non-negative.")
    if max_responses is not None and int(max_responses) < 1:
        raise ValueError("max_responses must be positive when provided.")

    plain = projector.reshape(full_dim, out_dim)
    dressed = np.array(
        plain,
        dtype=np.result_type(plain, h_dtype, complex),
        copy=True,
    )
    valid = valid_qn_mask(output_qn)
    diagnostics = {
        "discarded_residual_norm": 0.0,
        "response_rank": 0,
        "transition_mixing": 0.0,
        "sector_energy_gain": 0.0,
        "iterative_solves": 0,
        "iterative_iterations": 0,
        "iterative_fallbacks": 0,
        "maximum_response_residual": 0.0,
    }
    mixing_weight = 0.0
    mixing_states = 0
    full_diagonal = (
        Hpair4.diagonal()
        if matrix_free
        else np.real(np.diag(h_full))
    )

    for charge in sorted({qn_key(row) for row in output_qn[valid]}):
        charge_array = np.asarray(charge, dtype=int)
        pcols = np.flatnonzero(valid & np.all(output_qn == charge_array, axis=1))
        rows = np.flatnonzero(np.all(primitive_qn == charge_array, axis=1))
        if pcols.size == 0 or rows.size <= pcols.size:
            continue

        p_sector = plain[np.ix_(rows, pcols)]
        overlap = p_sector.conj().T @ p_sector
        if not np.allclose(overlap, np.eye(pcols.size), atol=1.0e-9):
            sector_error = float(
                np.linalg.norm(overlap - np.eye(pcols.size), ord=np.inf)
            )
            leakage_rows = np.flatnonzero(
                ~np.all(primitive_qn == charge_array, axis=1)
            )
            leakage = float(
                np.linalg.norm(plain[np.ix_(leakage_rows, pcols)], ord="fro")
            )
            actual_sectors = []
            for actual_charge in sorted(
                {qn_key(row) for row in primitive_qn}
            ):
                actual_rows = np.flatnonzero(
                    np.all(
                        primitive_qn == np.asarray(actual_charge, dtype=int),
                        axis=1,
                    )
                )
                weight = float(
                    np.linalg.norm(
                        plain[np.ix_(actual_rows, pcols)],
                        ord="fro",
                    )
                    ** 2
                )
                if weight > 1.0e-12:
                    actual_sectors.append((actual_charge, weight))
            raise ValueError(
                "retained rolling projector is not sector-orthonormal: "
                f"charge={charge}, error={sector_error:.3e}, "
                f"cross-sector leakage={leakage:.3e}, "
                f"actual sector weights={actual_sectors}."
            )
        def sector_apply(vectors):
            vectors = np.asarray(vectors)
            was_vector = vectors.ndim == 1
            if was_vector:
                vectors = vectors[:, None]
            embedded = np.zeros(
                (full_dim, vectors.shape[1]),
                dtype=np.result_type(h_dtype, projector, vectors, complex),
            )
            embedded[rows] = vectors
            result = apply_full(embedded)[rows]
            return result[:, 0] if was_vector else result

        hp = sector_apply(p_sector)
        h_pp = p_sector.conj().T @ hp
        h_pp = 0.5 * (h_pp + h_pp.T.conj())
        p_energy, p_coeff = np.linalg.eigh(h_pp)

        qdim = rows.size - pcols.size
        if qdim == 0:
            continue
        response_vectors = np.zeros(
            (rows.size, pcols.size),
            dtype=np.result_type(h_dtype, projector, complex),
        )
        sector_residual_sq = 0.0
        use_iterative = qdim > 64
        q_sector = None
        h_qq = None
        h_qp = None
        if not use_iterative:
            complete, _singular, _adjoint = np.linalg.svd(
                p_sector, full_matrices=True
            )
            q_sector = complete[:, pcols.size :]
            h_qp = q_sector.conj().T @ hp
            h_qq = q_sector.conj().T @ sector_apply(q_sector)
            h_qq = 0.5 * (h_qq + h_qq.T.conj())

        def q_project(vectors):
            return vectors - p_sector @ (p_sector.conj().T @ vectors)

        for state in range(pcols.size):
            hp_state = hp @ p_coeff[:, state]
            residual = q_project(hp_state)
            residual_norm = float(np.linalg.norm(residual))
            sector_residual_sq += residual_norm**2
            if residual_norm <= float(response_tol):
                continue
            energy = float(p_energy[state])
            if not use_iterative:
                rhs = h_qp @ p_coeff[:, state]
                shifted = h_qq + (
                    float(level_shift) - energy
                ) * np.eye(h_qq.shape[0])
                amplitudes = np.linalg.lstsq(shifted, -rhs, rcond=1.0e-12)[0]
                response_vectors[:, state] = q_sector @ amplitudes
                equation_residual = shifted @ amplitudes + rhs
                relative_residual = float(
                    np.linalg.norm(equation_residual) / max(residual_norm, 1.0e-30)
                )
            else:
                shift = float(level_shift) - energy

                def response_action(vector):
                    qvector = q_project(vector)
                    return q_project(sector_apply(qvector) + shift * qvector) + (
                        p_sector @ (p_sector.conj().T @ vector)
                    )

                diagonal = np.asarray(full_diagonal)[rows] + shift
                scale = max(1.0, float(np.max(np.abs(diagonal))))
                safe = np.where(
                    np.abs(diagonal) > 1.0e-10 * scale,
                    diagonal,
                    np.where(np.real(diagonal) < 0.0, -1.0, 1.0) * 1.0e-10 * scale,
                )

                def precondition(vector):
                    return q_project(vector / safe) + p_sector @ (
                        p_sector.conj().T @ vector
                    )

                operator = LinearOperator(
                    (rows.size, rows.size),
                    matvec=response_action,
                    dtype=np.result_type(h_dtype, projector, complex),
                )
                preconditioner = LinearOperator(
                    (rows.size, rows.size),
                    matvec=precondition,
                    dtype=operator.dtype,
                )
                iterations = [0]

                def count_iteration(_residual):
                    iterations[0] += 1

                amplitude, info = gmres(
                    operator,
                    -residual,
                    M=preconditioner,
                    rtol=max(float(response_tol), 1.0e-12),
                    atol=0.0,
                    restart=min(80, rows.size),
                    maxiter=max(20, 4 * rows.size),
                    callback=count_iteration,
                    callback_type="pr_norm",
                )
                amplitude = q_project(amplitude)
                equation_residual = response_action(amplitude) + residual
                relative_residual = float(
                    np.linalg.norm(q_project(equation_residual))
                    / max(residual_norm, 1.0e-30)
                )
                diagnostics["iterative_solves"] += 1
                diagnostics["iterative_iterations"] += iterations[0]
                if info != 0 or relative_residual > max(1.0e-8, 10.0 * float(response_tol)):
                    diagnostics["iterative_fallbacks"] += 1
                    sector_basis = np.zeros(
                        (full_dim, rows.size),
                        dtype=np.result_type(h_dtype, projector, complex),
                    )
                    sector_basis[rows, np.arange(rows.size)] = 1.0
                    h_sector = apply_full(sector_basis)[rows]
                    h_sector = 0.5 * (h_sector + h_sector.T.conj())
                    complete, _singular, _adjoint = np.linalg.svd(
                        p_sector, full_matrices=True
                    )
                    q_ref = complete[:, pcols.size :]
                    h_qp_ref = q_ref.conj().T @ (h_sector @ p_sector)
                    h_qq_ref = q_ref.conj().T @ (h_sector @ q_ref)
                    rhs_ref = h_qp_ref @ p_coeff[:, state]
                    shifted_ref = h_qq_ref + shift * np.eye(q_ref.shape[1])
                    amplitudes = np.linalg.lstsq(
                        shifted_ref, -rhs_ref, rcond=1.0e-12
                    )[0]
                    amplitude = q_ref @ amplitudes
                    relative_residual = float(
                        np.linalg.norm(shifted_ref @ amplitudes + rhs_ref)
                        / max(residual_norm, 1.0e-30)
                    )
                response_vectors[:, state] = amplitude
            diagnostics["maximum_response_residual"] = max(
                diagnostics["maximum_response_residual"],
                relative_residual,
            )
        diagnostics["discarded_residual_norm"] = float(
            np.hypot(diagnostics["discarded_residual_norm"], np.sqrt(sector_residual_sq))
        )

        left, singular, _right = np.linalg.svd(response_vectors, full_matrices=False)
        if singular.size == 0 or singular[0] <= float(response_tol):
            continue
        rank = int(np.count_nonzero(singular > float(response_tol) * singular[0]))
        if max_responses is not None:
            rank = min(rank, int(max_responses))
        if rank == 0:
            continue
        response_basis = left[:, :rank]
        trial = np.column_stack((p_sector, response_basis))
        h_trial = trial.conj().T @ sector_apply(trial)
        h_trial = 0.5 * (h_trial + h_trial.T.conj())
        trial_energy, trial_coeff = np.linalg.eigh(h_trial)
        selected = trial_coeff[:, : pcols.size]
        dressed[np.ix_(rows, pcols)] = trial @ selected

        diagnostics["response_rank"] += rank
        diagnostics["sector_energy_gain"] += float(
            np.sum(p_energy) - np.sum(trial_energy[: pcols.size])
        )
        mixing_weight += float(np.sum(np.abs(selected[pcols.size :, :]) ** 2))
        mixing_states += pcols.size

    diagnostics["transition_mixing"] = (
        mixing_weight / mixing_states if mixing_states else 0.0
    )
    dressed_projector = dressed.reshape(old_dim, local_dim, out_dim)
    h_dressed = dressed.conj().T @ apply_full(dressed)
    h_dressed = 0.5 * (h_dressed + h_dressed.T.conj())
    return h_dressed, dressed_projector, diagnostics


def _append_pair_supersite_reduced(
    H0,
    input_qn,
    table,
    residuals,
    first_site,
    keep,
    h1e,
    eri,
    target_qn,
    total_sites,
    *,
    use_irrep_tensor=False,
):
    """Append a two-orbital supersite using only retained-space operators."""
    old_dim = H0.shape[0]
    keep = min(int(keep), old_dim)
    d = len(LOCAL_QN)
    Hpair4, _primitive_pair_qn, local_qn = _compressed_superblock_two_site_lloo(
        H0, input_qn, table, residuals, first_site, h1e, eri
    )
    local_dim = local_qn.shape[0]
    branch_tensor = np.zeros((old_dim, keep, local_dim), dtype=np.result_type(Hpair4, complex))
    branch_qn = np.empty((local_dim, keep, LOCAL_QN.shape[1]), dtype=int)
    branch_energy = np.empty((local_dim, keep), dtype=float)

    for local_id, qn_local in enumerate(local_qn):
        branch_h = Hpair4[local_id, local_id]
        allowed_qn = feasible_multi_branch_qns(target_qn, first_site, total_sites, qn_local, 2)
        E, Ubranch, qn_branch = diagonalize_by_qn(
            branch_h,
            input_qn,
            keep,
            allowed_qn=allowed_qn,
            use_irrep_tensor=use_irrep_tensor,
            allow_empty=True,
        )
        E, Ubranch, qn_branch = pad_branch(E, Ubranch, qn_branch, keep)
        branch_energy[local_id] = E
        branch_tensor[:, :, local_id] = Ubranch
        branch_qn[local_id] = qn_branch

    Hnew = project_pair_hamiltonian_lloo(Hpair4, branch_tensor)
    Hnew = 0.5 * (Hnew + Hnew.T.conj())
    output_qn = (branch_qn + local_qn[:, None, :]).reshape((-1, LOCAL_QN.shape[1]))
    tensor = branch_tensor.reshape((old_dim, keep, d, d)).copy()
    return Hnew, tensor, branch_tensor, output_qn, branch_qn, local_qn, branch_energy


def _append_one_site_reduced(
    H0,
    input_qn,
    table,
    residuals,
    site_id,
    keep,
    h1e,
    eri,
    pair_terms,
    target_qn,
    total_sites,
    *,
    use_irrep_tensor=False,
    use_irrep_blocks=False,
):
    """Append one spatial orbital using retained-space operators."""
    site_id = int(site_id)
    keep = min(int(keep), H0.shape[0])
    d = len(LOCAL_QN)
    pair_sums = build_pair_sums(table, pair_terms, site_id)
    branch_allowed = lambda local_state: feasible_branch_qns(
        target_qn, site_id, total_sites, LOCAL_QN[local_state]
    )
    branch_inputs = (
        (0, 0, H0),
        (1, 0, branch_hamiltonian(H0, pair_sums, 1, 0)),
        (0, 1, branch_hamiltonian(H0, pair_sums, 0, 1)),
        (1, 1, branch_hamiltonian(H0, pair_sums, 1, 1)),
    )
    energies = []
    branches = []
    branch_qns = []
    for local_state, (_nu, _nd, branch_h) in enumerate(branch_inputs):
        E, Ubranch, qn_branch = diagonalize_by_qn(
            branch_h,
            input_qn,
            keep,
            allowed_qn=branch_allowed(local_state),
            use_irrep_tensor=use_irrep_tensor,
            allow_empty=True,
        )
        E, Ubranch, qn_branch = pad_branch(E, Ubranch, qn_branch, keep)
        energies.append(E)
        branches.append(Ubranch)
        branch_qns.append(qn_branch)

    branch_energy = np.zeros((d, keep), dtype=float)
    branch_tensor = np.zeros((H0.shape[0], keep, d), dtype=np.result_type(*branches, complex))
    h = single_site_hamiltonian_from_integrals(h1e, eri, site_id)
    for local_state in range(d):
        branch_energy[local_state] = energies[local_state] + h[local_state, local_state]
        branch_tensor[:, :, local_state] = branches[local_state]
    output_qn = np.concatenate(
        tuple(branch_qns[local_state] + LOCAL_QN[local_state] for local_state in range(d))
    )
    plan = RotationPlan(branch_tensor, output_qn)
    scalar_shift = (0, 0)
    Hnew = np.diag(branch_energy.reshape((keep * d))).astype(np.result_type(branch_tensor, complex))

    op_lists = operator_lists(table)
    Cdu = op_lists['Cdu']
    Cdd = op_lists['Cdd']
    v1u = zero_like_operator(H0)
    v1d = zero_like_operator(H0)
    for i in range(site_id):
        v1u += h1e[i, site_id] * dense_operator(Cdu[i])
        v1d += h1e[i, site_id] * dense_operator(Cdd[i])
    if site_id in residuals:
        v1u += residuals[site_id][0]
        v1d += residuals[site_id][1]

    V1 = (
        rotate_symmetry(v1u, JW @ cu, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
        + rotate_symmetry(v1d, JW @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
    )
    Hnew += V1 + dag(V1)

    V2a = rotate_symmetry(pair_sums['v2a'], cdu @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
    V2b = rotate_symmetry(pair_sums['v2b'], cdu @ cdd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
    Hnew += V2a + dag(V2a) + V2b + dag(V2b)

    v3u = zero_like_operator(H0)
    v3d = zero_like_operator(H0)
    for i in range(site_id):
        v3u += eri_value(eri, i, site_id, site_id, site_id) * dense_operator(Cdu[i])
        v3d += eri_value(eri, i, site_id, site_id, site_id) * dense_operator(Cdd[i])
    V3 = (
        rotate_symmetry(v3u, JW @ Nd @ cu, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
        + rotate_symmetry(v3d, JW @ Nu @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
    )
    Hnew += V3 + dag(V3)
    Hnew = 0.5 * (Hnew + Hnew.T.conj())
    return Hnew, branch_tensor, output_qn, np.stack(branch_qns, axis=0).copy(), LOCAL_QN.copy(), branch_energy


def initial_spin_operators(op_table):
    nsites = len(op_table[('Cu',)])
    old_dim = next(iter(op_table[('Cu',)].values())).shape[0]
    zero = csr_matrix((old_dim, old_dim), dtype=complex)
    sz_block = zero.copy()
    sp_block = zero.copy()
    sm_block = zero.copy()

    for i in range(nsites):
        sz_block += 0.5 * (
            op_table[('Cdu', 'Cu')][(i, i)] - op_table[('Cdd', 'Cd')][(i, i)]
        )
        sp_block += op_table[('Cdu', 'Cd')][(i, i)]
        sm_block += op_table[('Cdd', 'Cu')][(i, i)]

    s2_block = sz_block @ sz_block + 0.5 * (sp_block @ sm_block + sm_block @ sp_block)
    return {
        's2': s2_block,
        'sz': sz_block,
        'sp': sp_block,
        'sm': sm_block,
    }


def extend_spin_operators(spin_ops, U):
    old_dim = spin_ops['sz'].shape[0]
    iblock = eye(old_dim)
    isite = np.eye(4)

    sz_total = rotate(spin_ops['sz'], isite, U) + rotate(iblock, Sz, U)
    sp_total = rotate(spin_ops['sp'], isite, U) + rotate(iblock, Sp, U)
    sm_total = rotate(spin_ops['sm'], isite, U) + rotate(iblock, Sm, U)
    s2_total = (
        rotate(spin_ops['s2'], isite, U)
        + rotate(iblock, S2_LOCAL, U)
        + 2.0 * rotate(spin_ops['sz'], Sz, U)
        + rotate(spin_ops['sp'], Sm, U)
        + rotate(spin_ops['sm'], Sp, U)
    )
    return {
        's2': 0.5 * (s2_total + s2_total.T.conj()),
        'sz': sz_total,
        'sp': sp_total,
        'sm': sm_total,
    }


def final_spin_square_operator(spin_ops, U):
    return extend_spin_operators(spin_ops, U)['s2']


def spin_expectations(S2, X):
    s2 = np.real(np.einsum('ia,ij,ja->a', X.conj(), S2, X, optimize=True))
    s2 = np.maximum(s2, 0.0)
    spin = 0.5 * (np.sqrt(1.0 + 4.0 * s2) - 1.0)
    return s2, spin


def filter_spin_roots(E, X, final_qn, s2, spin, target_spin, spin_tol, nstates):
    target_s2 = float(target_spin) * (float(target_spin) + 1.0)
    keep = np.flatnonzero(np.abs(s2 - target_s2) <= spin_tol)
    keep = keep[:nstates]
    return E[keep], X[:, keep], final_qn[keep], s2[keep], spin[keep]


def build_initial_triple_residuals(single_ops, nsites, future_sites, triple_terms):
    dim = single_ops['Cu'][0].shape[0]
    residuals = {}
    for q in future_sites:
        v1u = csr_matrix((dim, dim), dtype=complex)
        v1d = csr_matrix((dim, dim), dtype=complex)
        for i, j, k, coeff in triple_terms[q]:
            if i >= nsites or j >= nsites or k >= nsites:
                continue
            v1u += coeff * (
                single_ops['Cdu'][k] @ single_ops['Cdu'][j] @ single_ops['Cu'][i]
                + single_ops['Cdu'][k] @ single_ops['Cdd'][j] @ single_ops['Cd'][i]
            )
            v1d += coeff * (
                single_ops['Cdd'][k] @ single_ops['Cdu'][j] @ single_ops['Cu'][i]
                + single_ops['Cdd'][k] @ single_ops['Cdd'][j] @ single_ops['Cd'][i]
            )
        residuals[q] = (v1u, v1d)
    return residuals


def build_initial_triple_residuals_irrep(table, nsites, future_sites, triple_terms):
    """Initial future-site triple residuals from an IrrepTensor operator table."""
    site = next(iter(table.values())).bra
    shift_u = pattern_qn_shift(('Cdu', 'Cdu', 'Cu'))
    shift_d = pattern_qn_shift(('Cdd', 'Cdu', 'Cu'))
    residuals = {}
    for q in future_sites:
        terms_u = []
        terms_d = []
        for i, j, k, coeff in triple_terms[q]:
            if i >= nsites or j >= nsites or k >= nsites:
                continue
            scalar_u = table[(('Cdu', 'Cu'), (j, i))]
            scalar_d = table[(('Cdd', 'Cd'), (j, i))]
            terms_u.append((coeff, matmul_tensors(table[(('Cdu',), (k,))], scalar_u)))
            terms_u.append((coeff, matmul_tensors(table[(('Cdu',), (k,))], scalar_d)))
            terms_d.append((coeff, matmul_tensors(table[(('Cdd',), (k,))], scalar_u)))
            terms_d.append((coeff, matmul_tensors(table[(('Cdd',), (k,))], scalar_d)))
        residuals[q] = (
            sum_tensor_terms(site, shift_u, terms_u),
            sum_tensor_terms(site, shift_d, terms_d),
        )
    return residuals


def project_weighted_new_site_terms(
    table, U, new_site, terms, output_qn=None, shift=None, use_irrep_blocks=False, plan=None
):
    old_dim = next(iter(table[('Cu',)].values())).shape[0]
    Iblock = eye(old_dim)
    grouped = {}
    out_dim = 4 * U.shape[1]

    if output_qn is not None and shift is not None and not has_qn_transition(output_qn, shift):
        return np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))

    for coeff, pattern, indices in terms:
        old_pattern = []
        old_indices = []
        local_key = []
        has_new_site = False

        for name, idx in zip(pattern, indices):
            if idx == new_site:
                has_new_site = True
                local_key.append(name)
            else:
                old_pattern.append(name)
                old_indices.append(idx)
                local_key.append('JW')

        if not has_new_site:
            continue

        if old_pattern:
            block_op = table[tuple(old_pattern)][tuple(old_indices)]
        else:
            block_op = Iblock

        key = tuple(local_key)
        grouped[key] = coeff * block_op if key not in grouped else grouped[key] + coeff * block_op

    projected = np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
    for local_key, block_op in grouped.items():
        if is_zero_operator(block_op):
            continue
        projected += rotate_symmetry(
            block_op, local_product(local_key), U, output_qn, shift, use_irrep_blocks, plan
        )
    return projected


def project_weighted_new_site_terms_irrep(table, U, block_qn, new_site, terms, output_qn, shift):
    """Project weighted new-site residual terms as Abelian IrrepTensors."""
    old_site, _ = irrep_site_from_qn(block_qn)
    output_site, _ = irrep_site_from_qn(output_qn)
    Iblock = IrrepTensor.identity(old_site)
    grouped = {}

    for coeff, pattern, indices in terms:
        old_pattern = []
        old_indices = []
        local_key = []
        has_new_site = False

        for name, idx in zip(pattern, indices):
            if idx == new_site:
                has_new_site = True
                local_key.append(name)
            else:
                old_pattern.append(name)
                old_indices.append(idx)
                local_key.append('JW')

        if not has_new_site:
            continue

        block_op = table[(tuple(old_pattern), tuple(old_indices))] if old_pattern else Iblock
        grouped.setdefault(tuple(local_key), []).append((coeff, block_op))

    projected = []
    for local_key, terms_for_local in grouped.items():
        first = terms_for_local[0][1]
        block_op = sum_tensor_terms(old_site, first.op.charge, terms_for_local)
        if not block_op.blocks:
            continue
        projected.append(
            rotate_irrep_tensor_product(
                block_op,
                local_product(local_key),
                U,
                block_qn,
                output_qn,
                local_operator_shift(local_key),
            )[0]
        )
    if not projected:
        return zero_tensor(output_site, shift)
    return add_tensors(*projected)


def extend_triple_residuals(
    residuals, table, U, new_site, total_sites, triple_terms, output_qn=None, use_irrep_blocks=False, plan=None
):
    new_residuals = {}
    out_dim = 4 * U.shape[1]
    shift_u = pattern_qn_shift(('Cdu', 'Cdu', 'Cu'))
    shift_d = pattern_qn_shift(('Cdd', 'Cdu', 'Cu'))
    allow_u = output_qn is None or has_qn_transition(output_qn, shift_u)
    allow_d = output_qn is None or has_qn_transition(output_qn, shift_d)
    for q in range(new_site + 1, total_sites):
        old_v1u, old_v1d = residuals[q]
        v1u = (
            rotate_symmetry(old_v1u, JW, U, output_qn, shift_u, use_irrep_blocks, plan)
            if allow_u else np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
        )
        v1d = (
            rotate_symmetry(old_v1d, JW, U, output_qn, shift_d, use_irrep_blocks, plan)
            if allow_d else np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
        )

        terms_u = []
        terms_d = []
        for i, j, k, coeff in triple_terms[q]:
            if i > new_site or j > new_site or k > new_site:
                continue
            if i != new_site and j != new_site and k != new_site:
                continue
            indices = (k, j, i)
            terms_u.append((coeff, ('Cdu', 'Cdu', 'Cu'), indices))
            terms_u.append((coeff, ('Cdu', 'Cdd', 'Cd'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdu', 'Cu'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdd', 'Cd'), indices))

        v1u += project_weighted_new_site_terms(
            table, U, new_site, terms_u, output_qn, shift_u, use_irrep_blocks, plan
        )
        v1d += project_weighted_new_site_terms(
            table, U, new_site, terms_d, output_qn, shift_d, use_irrep_blocks, plan
        )
        new_residuals[q] = (v1u, v1d)
    return new_residuals


def extend_triple_residuals_two_site(
    residuals,
    table,
    U,
    first_site,
    total_sites,
    triple_terms,
    output_qn=None,
    use_irrep_blocks=False,
    plan=None,
):
    """Update future-site triple residuals after a two-orbital reduced append."""
    first_site = int(first_site)
    second_site = first_site + 1
    out_dim = U.shape[1] * U.shape[2]
    old_dim = U.shape[0]
    shift_u = pattern_qn_shift(('Cdu', 'Cdu', 'Cu'))
    shift_d = pattern_qn_shift(('Cdd', 'Cdu', 'Cu'))
    allow_u = output_qn is None or has_qn_transition(output_qn, shift_u)
    allow_d = output_qn is None or has_qn_transition(output_qn, shift_d)
    pair_parity = local_pair_product(('JW',), (-1,))
    new_residuals = {}

    def projected_terms(terms, shift):
        out = np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
        for coeff, pattern, indices in terms:
            block_op, local_op = _two_site_operator_factor(table, pattern, indices, first_site, old_dim)
            out += coeff * rotate_symmetry(
                block_op,
                local_op,
                U,
                output_qn,
                shift,
                use_irrep_blocks,
                plan,
            )
        return out

    for q in range(second_site + 1, total_sites):
        old_v1u, old_v1d = residuals[q]
        v1u = (
            rotate_symmetry(old_v1u, pair_parity, U, output_qn, shift_u, use_irrep_blocks, plan)
            if allow_u else np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
        )
        v1d = (
            rotate_symmetry(old_v1d, pair_parity, U, output_qn, shift_d, use_irrep_blocks, plan)
            if allow_d else np.zeros((out_dim, out_dim), dtype=np.result_type(U, complex))
        )

        terms_u = []
        terms_d = []
        for i, j, k, coeff in triple_terms[q]:
            if i > second_site or j > second_site or k > second_site:
                continue
            if i < first_site and j < first_site and k < first_site:
                continue
            indices = (k, j, i)
            terms_u.append((coeff, ('Cdu', 'Cdu', 'Cu'), indices))
            terms_u.append((coeff, ('Cdu', 'Cdd', 'Cd'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdu', 'Cu'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdd', 'Cd'), indices))

        if allow_u:
            v1u += projected_terms(terms_u, shift_u)
        if allow_d:
            v1d += projected_terms(terms_d, shift_d)
        new_residuals[q] = (v1u, v1d)
    return new_residuals


def extend_triple_residuals_irrep(
    residuals, table, U, block_qn, new_site, total_sites, triple_terms, output_qn
):
    """Update future-site triple residuals using IrrepTensor operators."""
    output_site, _ = irrep_site_from_qn(output_qn)
    new_residuals = {}
    shift_u = pattern_qn_shift(('Cdu', 'Cdu', 'Cu'))
    shift_d = pattern_qn_shift(('Cdd', 'Cdu', 'Cu'))
    for q in range(new_site + 1, total_sites):
        old_v1u, old_v1d = residuals[q]
        v1u = rotate_irrep_tensor_product(old_v1u, JW, U, block_qn, output_qn, (0, 0))[0]
        v1d = rotate_irrep_tensor_product(old_v1d, JW, U, block_qn, output_qn, (0, 0))[0]

        terms_u = []
        terms_d = []
        for i, j, k, coeff in triple_terms[q]:
            if i > new_site or j > new_site or k > new_site:
                continue
            if i != new_site and j != new_site and k != new_site:
                continue
            indices = (k, j, i)
            terms_u.append((coeff, ('Cdu', 'Cdu', 'Cu'), indices))
            terms_u.append((coeff, ('Cdu', 'Cdd', 'Cd'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdu', 'Cu'), indices))
            terms_d.append((coeff, ('Cdd', 'Cdd', 'Cd'), indices))

        add_u = project_weighted_new_site_terms_irrep(
            table, U, block_qn, new_site, terms_u, output_qn, shift_u
        )
        add_d = project_weighted_new_site_terms_irrep(
            table, U, block_qn, new_site, terms_d, output_qn, shift_d
        )
        new_residuals[q] = (
            add_tensors(v1u, add_u) if (v1u.blocks or add_u.blocks) else zero_tensor(output_site, shift_u),
            add_tensors(v1d, add_d) if (v1d.blocks or add_d.blocks) else zero_tensor(output_site, shift_d),
        )
    return new_residuals


def operator_lists(table):
    return {
        name: [table[(name,)][(idx,)] for idx in range(len(table[(name,)]))]
        for name in ('Cu', 'Cd', 'Cdu', 'Cdd')
    }


def extend_operator_table(
    table,
    patterns,
    U,
    new_site,
    output_qn=None,
    use_irrep_blocks=False,
    plan=None,
    required=None,
    sparse_output=False,
    consume=False,
):
    """Project full composite operators when adding one spatial orbital."""
    old_dim = next(iter(table[('Cu',)].values())).shape[0]
    Iblock = eye(old_dim)
    Isite = np.eye(4)
    local = {
        'Cu': cu,
        'Cd': cd,
        'Cdu': cdu,
        'Cdd': cdd,
    }
    new_table = {}

    source_uses = {}
    if consume:
        for pattern in patterns:
            index_iter = (
                np.ndindex((new_site + 1,) * len(pattern))
                if required is None
                else sorted(required.get(pattern, ()))
            )
            for indices in index_iter:
                old_pattern = tuple(
                    name for name, idx in zip(pattern, indices) if idx != new_site
                )
                old_indices = tuple(idx for idx in indices if idx != new_site)
                if old_pattern:
                    key = (old_pattern, old_indices)
                    source_uses[key] = source_uses.get(key, 0) + 1
        for pattern, entries in table.items():
            for indices in tuple(entries):
                if (pattern, indices) not in source_uses:
                    entries.pop(indices)

    def release_source(pattern, indices):
        if not consume or not pattern:
            return
        key = (pattern, indices)
        source_uses[key] -= 1
        if source_uses[key] == 0:
            table[pattern].pop(indices, None)

    for pattern in patterns:
        entries = {}
        shift = pattern_qn_shift(pattern)
        allowed_by_qn = output_qn is None or has_qn_transition(output_qn, shift)
        if required is None:
            index_iter = np.ndindex((new_site + 1,) * len(pattern))
        else:
            index_iter = sorted(required.get(pattern, ()))
        for indices in index_iter:
            old_pattern = []
            old_indices = []
            local_op = Isite

            for name, idx in zip(pattern, indices):
                if idx == new_site:
                    local_op = local_op @ local[name]
                else:
                    old_pattern.append(name)
                    old_indices.append(idx)
                    local_op = local_op @ JW

            if old_pattern:
                block_op = table[tuple(old_pattern)][tuple(old_indices)]
            else:
                block_op = Iblock

            if (not allowed_by_qn) or is_zero_operator(block_op) or is_zero_operator(local_op):
                shape = (4 * U.shape[1], 4 * U.shape[1])
                if sparse_output:
                    entries[indices] = csr_matrix(shape, dtype=np.result_type(U, local_op))
                else:
                    entries[indices] = np.zeros(shape, dtype=np.result_type(U, local_op))
            else:
                entries[indices] = rotate_symmetry(
                    block_op, local_op, U, output_qn, shift, use_irrep_blocks, plan, sparse_output=sparse_output
                )
            release_source(tuple(old_pattern), tuple(old_indices))
        new_table[pattern] = entries

    if consume:
        table.clear()
    return new_table


def atomic_chain(natom, z, element='H', basis='631g', spin=0):

    # ds = np.linspace(-4, 4, natom)

    elements = [element, ] * natom

    R = np.zeros((natom, 3))
    R[:, 2] = z

    atom = build_atom_from_coords(elements, R)

    mol = Molecule(
        atom = atom,
        basis = basis,
        unit = 'b',
        spin = spin,
        )

    return mol




def kernel(
    h1e,
    eri,
    D=20,
    n0=4,
    nstates=1,
    verbose=False,
    growth_sites=1,
    two_site_intermediate_dim=None,
    two_site_max_dim=None,
    two_site_energy_tol=None,
    two_site_gap_factor=1.0,
    eri_cutoff=0.0,
    fast=False,
    use_numba_terms='auto',
    use_irrep_tensor=False,
    use_irrep_blocks=False,
    use_irrep_operator_table=False,
    sparse_operator_table='auto',
    use_block_sparse_hamiltonian='auto',
    use_sparse_operator_projection='auto',
    two_site_mode='supersite',
    dressing=None,
    chi=None,
    frame_adapt_tol=None,
    frame_max_dim=None,
    frame_expand_dim=1,
    cc_level_shift=0.0,
    cc_response_tol=1.0e-10,
    cc_max_responses=None,
    target_spin=None,
    spin_tol=1e-3,
    spin_search_factor=4,
    return_spin=False,
    return_tensors=False,
    return_tensor_qns=False,
):
    
    # C = mf.mo_coeff
    
    # h1e = mol.hcore
    
    # h1e = dag(C) @ h1e @ C
    
    # eri = mol.eri
    L = h1e.shape[-1]
    validate_eri_representation(eri, L)
    if fast:
        if use_numba_terms == 'auto':
            use_numba_terms = True
        if use_irrep_blocks is False:
            use_irrep_blocks = True
    if isinstance(growth_sites, str):
        if growth_sites != "auto":
            raise ValueError("growth_sites must be 1, 2, 3, 4, or 'auto'.")
    else:
        growth_sites = int(growth_sites)
        if growth_sites not in {1, 2, 3, 4}:
            raise ValueError("growth_sites must be 1, 2, 3, 4, or 'auto'.")
    if two_site_intermediate_dim is not None and int(two_site_intermediate_dim) < 1:
        raise ValueError("two_site_intermediate_dim must be positive when provided.")
    if two_site_max_dim is not None and int(two_site_max_dim) < 1:
        raise ValueError("two_site_max_dim must be positive when provided.")
    if two_site_energy_tol is not None and float(two_site_energy_tol) < 0.0:
        raise ValueError("two_site_energy_tol must be non-negative when provided.")
    two_site_gap_factor = float(two_site_gap_factor)
    if two_site_gap_factor < 0.0:
        raise ValueError("two_site_gap_factor must be non-negative.")
    two_site_mode = str(two_site_mode).lower().replace("-", "_")
    if two_site_mode in {"rebranch", "rebranched", "pair", "pair_branch"}:
        two_site_mode = "supersite"
    elif two_site_mode in {"two_site", "true_two_site", "rolling_two_site"}:
        two_site_mode = "rolling"
    if two_site_mode not in {"sequential", "supersite", "rolling"}:
        raise ValueError("two_site_mode must be 'sequential', 'supersite', or 'rolling'.")
    dressing_key = (
        "none" if dressing is None else str(dressing).lower().replace("-", "_")
    )
    if dressing_key in {"none", "off", "false"}:
        dressing_key = "none"
    elif dressing_key in {
        "cc",
        "conditional_cc",
        "conditional_cc_transition",
        "transition_cc",
    }:
        dressing_key = "conditional_cc"
    elif dressing_key in {
        "detached",
        "detached_frame",
        "detached_frames",
    }:
        dressing_key = "detached_frames"
    elif dressing_key in {
        "detached+cc",
        "detached_cc",
        "detached_frames_cc",
        "cc_detached",
    }:
        dressing_key = "detached_cc"
    else:
        raise ValueError(
            "dressing must be None, 'conditional_cc', 'detached_frames', or 'detached_cc'."
        )
    if (
        dressing_key == "conditional_cc"
        and growth_sites in {2, "auto"}
        and two_site_mode == "supersite"
    ):
        raise ValueError(
            "conditional_cc with paired supersite growth is not implemented; "
            "use growth_sites=1 or two_site_mode='rolling'."
        )
    if dressing_key in {"detached_frames", "detached_cc"}:
        if growth_sites != 1:
            raise ValueError("detached dressing currently requires growth_sites=1.")
        frame_space = len(LOCAL_QN) * int(D)
        detached_space = len(LOCAL_QN) * frame_space
        chi = 2 * frame_space if chi is None else int(chi)
        if not frame_space < chi <= detached_space:
            raise ValueError(
                f"detached dressing requires {frame_space} < chi <= {detached_space} "
                f"for D={int(D)}."
            )
        if frame_adapt_tol is not None and float(frame_adapt_tol) < 0.0:
            raise ValueError("frame_adapt_tol must be non-negative when provided.")
        if frame_max_dim is not None:
            frame_max_dim = int(frame_max_dim)
            if not frame_space <= frame_max_dim <= chi:
                raise ValueError(
                    f"frame_max_dim must satisfy {frame_space} <= "
                    f"frame_max_dim <= chi={chi}."
                )
        frame_expand_dim = int(frame_expand_dim)
        if frame_expand_dim < 1:
            raise ValueError("frame_expand_dim must be positive.")
    elif chi is not None:
        raise ValueError("chi is only used with detached dressing.")
    elif (
        frame_adapt_tol is not None
        or frame_max_dim is not None
        or int(frame_expand_dim) != 1
    ):
        raise ValueError("frame adaptation options require detached dressing.")
    cc_response_tol = float(cc_response_tol)
    if cc_response_tol < 0.0:
        raise ValueError("cc_response_tol must be non-negative.")
    if cc_max_responses is not None and int(cc_max_responses) < 1:
        raise ValueError("cc_max_responses must be positive when provided.")
    orbital_energy = np.real(np.diag(h1e))
    adjacent_gaps = np.abs(np.diff(orbital_energy))
    if two_site_energy_tol is None:
        two_site_energy_tol = 0.0 if adjacent_gaps.size == 0 else two_site_gap_factor * float(np.median(adjacent_gaps))
    else:
        two_site_energy_tol = float(two_site_energy_tol)
    nelec = np.asarray(mol.nelec, dtype=int).reshape(-1)
    target_nelec = int(np.sum(nelec))
    target_sz2 = int(nelec[0] - nelec[1]) if nelec.size == 2 else int(getattr(mol, "spin", 0))
    target_qn = (target_nelec, target_sz2)
    allowed = lambda nsites: feasible_qns(target_qn, nsites, L)
    branch_allowed = lambda block_nsites, local_state: feasible_branch_qns(
        target_qn, block_nsites, L, LOCAL_QN[local_state]
    )
    need_spin = target_spin is not None or return_spin or verbose
    if dressing_key != "none" and need_spin:
        raise NotImplementedError(
            "general-projector dressing does not yet project spin observables."
        )
    if dressing_key != "none" and use_irrep_operator_table:
        raise NotImplementedError(
            "general-projector dressing requires the dense/sparse Abelian operator table."
        )
    if two_site_mode == "rolling" and need_spin:
        raise NotImplementedError("rolling two-site qchem NARG does not yet support spin observables.")
    if two_site_mode == "rolling" and return_tensors:
        raise NotImplementedError(
            "rolling two-site qchem NARG does not yet return reconstructable tensors."
        )
    if sparse_operator_table == 'auto':
        sparse_operator_table = True
    else:
        sparse_operator_table = bool(sparse_operator_table)
    if use_block_sparse_hamiltonian == 'auto':
        use_block_sparse_hamiltonian = False
    else:
        use_block_sparse_hamiltonian = bool(use_block_sparse_hamiltonian)
    if use_sparse_operator_projection == 'auto':
        use_sparse_operator_projection = bool(fast)
    else:
        use_sparse_operator_projection = bool(use_sparse_operator_projection)
    
    # # transform to MOs
    # eri = contract('ijkl, ip, jq, kr, ls -> pqrs', eri, C.conj(), C, C.conj(), C)
    v = eri
    pair_terms, triple_terms = precompute_integral_terms(eri, eri_cutoff, use_numba=use_numba_terms)
    
    # initiate the block with l0 Spin-Orbitals
    nstart = n0
    initial_eri = eri_to_dense(eri_subspace(v, range(nstart)))
    model = SpinHalfFermionChain(
        h1e[:nstart, :nstart],
        initial_eri,
        nelec=mol.nelec,
    )
    # model.fix_nelec(s=2)
    
    model.jordan_wigner(forward=False)
    
    # D = 160 # retained adiabatic eigenstates
    block_qn = primitive_charge_labels(nstart)
    H0 = model.H
    d = len(LOCAL_QN)
    scalar_shift = (0, 0)

    def single_site_hamiltonian(site_id):
        return single_site_hamiltonian_from_integrals(h1e, eri, site_id)

    required_entries = (
        required_operator_entries(pair_terms, triple_terms, L, nstart)
        if sparse_operator_table
        else None
    )
    if need_spin:
        required_entries = add_initial_spin_entries(required_entries, nstart)
    initial_operators = {
        "Cdu": model.Cdu,
        "Cdd": model.Cdd,
        "Cu": model.Cu,
        "Cd": model.Cd,
    }
    op_table = make_operator_table(
        initial_operators,
        OPERATOR_PATTERNS,
        nstart,
        required=required_entries,
    )
    irrep_op_table = (
        irrep_operator_table(op_table, block_qn)
        if use_irrep_operator_table
        else None
    )
    op_table_qn = block_qn
    triple_residuals = build_initial_triple_residuals(
        initial_operators,
        nstart,
        range(nstart, L),
        triple_terms,
    )
    irrep_triple_residuals = (
        build_initial_triple_residuals_irrep(
            irrep_op_table,
            nstart,
            range(nstart, L),
            triple_terms,
        )
        if use_irrep_operator_table
        else None
    )
    spin_ops = initial_spin_operators(op_table) if need_spin else None
    narg_tensors = []
    tensor_qns = []

    def extend_current_block(
        table, irrep_table, table_qn, residuals, irrep_residuals, spins, basis_tensor, site_id, output_qn
    ):
        plan = RotationPlan(basis_tensor, output_qn) if use_irrep_blocks else None
        required_entries = (
            required_operator_entries(pair_terms, triple_terms, L, site_id + 1)
            if sparse_operator_table
            else None
        )
        residuals = extend_triple_residuals(
            residuals, table, basis_tensor, site_id, L, triple_terms, output_qn, use_irrep_blocks, plan
        )
        if use_irrep_operator_table:
            irrep_residuals = extend_triple_residuals_irrep(
                irrep_residuals, irrep_table, basis_tensor, table_qn, site_id, L, triple_terms, output_qn
            )
        if need_spin:
            spins = extend_spin_operators(spins, basis_tensor)
        if use_irrep_operator_table:
            irrep_table = extend_irrep_operator_table(
                irrep_table,
                OPERATOR_PATTERNS,
                basis_tensor,
                table_qn,
                output_qn,
                site_id,
                required=required_entries,
            )
            table_qn = output_qn
        table = extend_operator_table(
            table,
            OPERATOR_PATTERNS,
            basis_tensor,
            site_id,
            output_qn,
            use_irrep_blocks,
            plan,
            required=required_entries,
            sparse_output=use_sparse_operator_projection,
            consume=True,
        )
        return table, irrep_table, table_qn, residuals, irrep_residuals, spins

    def compressed_superblock_two_site(H0, input_qn, table, residuals, first_site):
        return _compressed_superblock_two_site_lloo(H0, input_qn, table, residuals, first_site, h1e, eri)

    def append_supersite(H0, input_qn, table, residuals, first_site, keep):
        old_dim = H0.shape[0]
        keep = min(int(keep), old_dim)
        Hpair4, _primitive_pair_qn, local_qn = compressed_superblock_two_site(H0, input_qn, table, residuals, first_site)
        local_dim = local_qn.shape[0]
        branch_tensor = np.zeros((old_dim, keep, local_dim), dtype=np.result_type(Hpair4, complex))
        branch_qn = np.empty((local_dim, keep, LOCAL_QN.shape[1]), dtype=int)
        branch_energy = np.empty((local_dim, keep), dtype=float)

        for local_id, qn_local in enumerate(local_qn):
            branch_h = Hpair4[local_id, local_id]
            allowed_qn = feasible_multi_branch_qns(target_qn, first_site, L, qn_local, 2)
            E, Ubranch, qn_branch = diagonalize_by_qn(
                branch_h,
                input_qn,
                keep,
                allowed_qn=allowed_qn,
                use_irrep_tensor=use_irrep_tensor,
                allow_empty=True,
            )
            E, Ubranch, qn_branch = pad_branch(E, Ubranch, qn_branch, keep)
            branch_energy[local_id] = E
            branch_tensor[:, :, local_id] = Ubranch
            branch_qn[local_id] = qn_branch

        Hnew = project_pair_hamiltonian_lloo(Hpair4, branch_tensor)
        Hnew = 0.5 * (Hnew + Hnew.T.conj())
        output_qn = (branch_qn + local_qn[:, None, :]).reshape((-1, LOCAL_QN.shape[1]))
        tensor = branch_tensor.reshape((old_dim, keep, d, d)).copy()
        return Hnew, tensor, branch_tensor, output_qn, branch_qn, local_qn, branch_energy

    def append_rolling_two_sites(H0, input_qn, table, residuals, first_site, keep):
        old_dim = H0.shape[0]
        keep = min(int(keep), old_dim)
        second_site = int(first_site) + 1
        Hpair4, primitive_pair_qn, local_qn = compressed_superblock_two_site(
            H0, input_qn, table, residuals, first_site
        )
        local_dim = local_qn.shape[0]
        branch_tensor = np.zeros((old_dim, keep, local_dim), dtype=np.result_type(Hpair4, complex))
        branch_qn = np.empty((local_dim, keep, LOCAL_QN.shape[1]), dtype=int)
        branch_energy = np.empty((local_dim, keep), dtype=float)
        rows0 = np.arange(old_dim)

        for local_id, qn_local in enumerate(local_qn):
            branch_h = Hpair4[local_id, local_id]
            allowed_qn = feasible_multi_branch_qns(target_qn, first_site, L, qn_local, 2)
            E, Ubranch, qn_branch = diagonalize_by_qn(
                branch_h,
                input_qn,
                keep,
                allowed_qn=allowed_qn,
                use_irrep_tensor=use_irrep_tensor,
                allow_empty=True,
            )
            E, Ubranch, qn_branch = pad_branch(E, Ubranch, qn_branch, keep)
            branch_energy[local_id] = E
            branch_tensor[:, :, local_id] = Ubranch
            branch_qn[local_id] = qn_branch

        supersite_basis = np.zeros(
            (old_dim * local_dim, local_dim * keep),
            dtype=np.result_type(branch_tensor, complex),
        )
        for local_id in range(local_dim):
            supersite_basis[
                rows0 * local_dim + local_id,
                local_id * keep : (local_id + 1) * keep,
            ] = branch_tensor[:, :, local_id]
        Hmid = project_pair_hamiltonian_lloo(Hpair4, branch_tensor)
        Hmid = 0.5 * (Hmid + Hmid.T.conj())
        mid_qn = (branch_qn + local_qn[:, None, :]).reshape((-1, LOCAL_QN.shape[1]))

        rolling_basis = np.zeros((local_dim * keep, d * keep), dtype=np.result_type(Hmid, complex))
        rolling_qn = np.empty((d, keep, LOCAL_QN.shape[1]), dtype=int)
        rolling_energy = np.empty((d, keep), dtype=float)
        rolling_tensor = np.zeros((old_dim, keep, d, d), dtype=np.result_type(Hmid, branch_tensor, complex))

        for second_state in range(d):
            cols = []
            for first_state in range(d):
                local_id = first_state * d + second_state
                cols.extend((local_id * keep + np.arange(keep)).tolist())
            cols = np.asarray(cols, dtype=int)
            branch_h = Hmid[np.ix_(cols, cols)]
            E, Ubranch, qn_branch = diagonalize_by_qn(
                branch_h,
                mid_qn[cols],
                keep,
                allowed_qn=allowed(second_site + 1),
                use_irrep_tensor=use_irrep_tensor,
                allow_empty=True,
            )
            E, Ubranch, qn_branch = pad_branch(E, Ubranch, qn_branch, keep)
            rolling_energy[second_state] = E
            rolling_qn[second_state] = qn_branch
            rolling_basis[cols, second_state * keep : (second_state + 1) * keep] = Ubranch

            for first_state in range(d):
                local_id = first_state * d + second_state
                source = slice(first_state * keep, (first_state + 1) * keep)
                rolling_tensor[:, :, first_state, second_state] = (
                    branch_tensor[:, :, local_id] @ Ubranch[source, :]
                )

        projector = supersite_basis @ rolling_basis
        Hnew = rolling_basis.conj().T @ (Hmid @ rolling_basis)
        Hnew = 0.5 * (Hnew + Hnew.T.conj())
        output_qn = rolling_qn.reshape((-1, LOCAL_QN.shape[1]))
        projector = projector.reshape((old_dim, local_dim, d * keep))
        cc_diagnostics = None
        if dressing_key == "conditional_cc":
            Hnew, projector, cc_diagnostics = conditional_cc_transition_projector(
                Hpair4,
                primitive_pair_qn,
                projector,
                output_qn,
                level_shift=cc_level_shift,
                response_tol=cc_response_tol,
                max_responses=cc_max_responses,
            )
        return (
            Hnew,
            rolling_tensor,
            projector,
            output_qn,
            rolling_qn,
            local_qn,
            branch_energy,
            rolling_energy,
            cc_diagnostics,
        )

    def append_one_site(H0, H0_irrep, input_qn, table, irrep_table, residuals, irrep_residuals, site_id, keep):
        keep = min(int(keep), H0.shape[0])
        if dressing_key in {"detached_frames", "detached_cc"}:
            h_grown, primitive_qn = _compressed_superblock_one_site_action(
                H0,
                input_qn,
                table,
                residuals,
                site_id,
                h1e,
                eri,
                pair_terms,
            )
            output_allowed = allowed(site_id + 1)
            frame_allowed = {
                qn_key(np.asarray(output_charge) - local_charge)
                for output_charge in output_allowed
                for local_charge in LOCAL_QN
            }
            Hnew, projector, output_qn, detached_diagnostics = (
                detached_frame_transition_projector(
                    h_grown,
                    input_qn,
                    frame_dim=keep,
                    bond_dim=chi,
                    allowed_frame_qn=frame_allowed,
                    allowed_output_qn=output_allowed,
                    use_irrep_tensor=use_irrep_tensor,
                    adapt_tol=frame_adapt_tol,
                    max_frame_rank=frame_max_dim,
                    expand_dim=frame_expand_dim,
                )
            )
            diagnostics = detached_diagnostics
            if dressing_key == "detached_cc":
                Hnew, projector, cc_diagnostics = conditional_cc_transition_projector(
                    h_grown,
                    primitive_qn,
                    projector,
                    output_qn,
                    level_shift=cc_level_shift,
                    response_tol=cc_response_tol,
                    max_responses=cc_max_responses,
                )
                diagnostics = {
                    "detached_diagnostics": detached_diagnostics,
                    "cc_diagnostics": cc_diagnostics,
                }
            return (
                Hnew,
                None,
                projector.transpose(0, 2, 1).copy(),
                output_qn,
                None,
                projector,
                diagnostics,
            )
        direct_irrep_hamiltonian = False
        if use_irrep_operator_table:
            pair_sums_irrep = build_pair_sums_irrep(irrep_table, pair_terms, site_id)
        else:
            pair_sums_irrep = None
        pair_sums = build_pair_sums(table, pair_terms, site_id)

        if use_block_sparse_hamiltonian and not use_irrep_tensor:
            E0, U0, qn0 = branch_diagonalize_block_sparse(
                H0, pair_sums, 0, 0, input_qn, keep, allowed_qn=branch_allowed(site_id, 0), allow_empty=True
            )
            E1, U1, qn1 = branch_diagonalize_block_sparse(
                H0, pair_sums, 1, 0, input_qn, keep, allowed_qn=branch_allowed(site_id, 1), allow_empty=True
            )
            E2, U2, qn2 = branch_diagonalize_block_sparse(
                H0, pair_sums, 0, 1, input_qn, keep, allowed_qn=branch_allowed(site_id, 2), allow_empty=True
            )
            E3, U3, qn3 = branch_diagonalize_block_sparse(
                H0, pair_sums, 1, 1, input_qn, keep, allowed_qn=branch_allowed(site_id, 3), allow_empty=True
            )
        else:
            E0, U0, qn0 = diagonalize_by_qn(
                H0, input_qn, keep, allowed_qn=branch_allowed(site_id, 0),
                use_irrep_tensor=use_irrep_tensor, allow_empty=True
            )
            H1 = branch_hamiltonian(H0, pair_sums, 1, 0)
            E1, U1, qn1 = diagonalize_by_qn(
                H1, input_qn, keep, allowed_qn=branch_allowed(site_id, 1),
                use_irrep_tensor=use_irrep_tensor, allow_empty=True
            )
            H2 = branch_hamiltonian(H0, pair_sums, 0, 1)
            E2, U2, qn2 = diagonalize_by_qn(
                H2, input_qn, keep, allowed_qn=branch_allowed(site_id, 2),
                use_irrep_tensor=use_irrep_tensor, allow_empty=True
            )
            H3 = branch_hamiltonian(H0, pair_sums, 1, 1)
            E3, U3, qn3 = diagonalize_by_qn(
                H3, input_qn, keep, allowed_qn=branch_allowed(site_id, 3),
                use_irrep_tensor=use_irrep_tensor, allow_empty=True
            )

        E0, U0, qn0 = pad_branch(E0, U0, qn0, keep)
        E1, U1, qn1 = pad_branch(E1, U1, qn1, keep)
        E2, U2, qn2 = pad_branch(E2, U2, qn2, keep)
        E3, U3, qn3 = pad_branch(E3, U3, qn3, keep)

        E = np.zeros((d, keep))
        branch_tensor = np.zeros((H0.shape[0], keep, d), dtype=np.result_type(U0, U1, U2, U3))
        h = single_site_hamiltonian(site_id)
        log.info('site %s H = %s', site_id, np.diag(h))

        E[0, :] = E0 + h[0, 0]
        E[1, :] = E1 + h[1, 1]
        E[2, :] = E2 + h[2, 2]
        E[3, :] = E3 + h[3, 3]
        branch_tensor[:, :, 0] = U0
        branch_tensor[:, :, 1] = U1
        branch_tensor[:, :, 2] = U2
        branch_tensor[:, :, 3] = U3
        output_qn = np.concatenate(
            (qn0 + LOCAL_QN[0], qn1 + LOCAL_QN[1], qn2 + LOCAL_QN[2], qn3 + LOCAL_QN[3])
        )
        plan = RotationPlan(branch_tensor, output_qn) if use_irrep_blocks else None

        Hnew_diag = np.diag(E.reshape((keep * d))).astype(np.result_type(branch_tensor, complex))
        if use_irrep_operator_table and direct_irrep_hamiltonian:
            site = H0_irrep.bra
            hnew_diag_irrep = labeled_irrep_tensor(Hnew_diag, output_qn, op=(0, 0))
            v1u_terms = [(h1e[i, site_id], irrep_table[(('Cdu',), (i,))]) for i in range(site_id)]
            v1d_terms = [(h1e[i, site_id], irrep_table[(('Cdd',), (i,))]) for i in range(site_id)]
            v1u_terms.append((1.0, irrep_residuals[site_id][0]))
            v1d_terms.append((1.0, irrep_residuals[site_id][1]))
            v1u = sum_tensor_terms(site, OPERATOR_QN_SHIFT['Cdu'], v1u_terms)
            v1d = sum_tensor_terms(site, OPERATOR_QN_SHIFT['Cdd'], v1d_terms)

            V1u = rotate_irrep_tensor_product(
                v1u, JW @ cu, branch_tensor, input_qn, output_qn, local_operator_shift(('JW', 'Cu'))
            )[0]
            V1d = rotate_irrep_tensor_product(
                v1d, JW @ cd, branch_tensor, input_qn, output_qn, local_operator_shift(('JW', 'Cd'))
            )[0]
            V2a = rotate_irrep_tensor_product(
                pair_sums_irrep['v2a'], cdu @ cd, branch_tensor, input_qn, output_qn, local_operator_shift(('Cdu', 'Cd'))
            )[0]
            V2b = rotate_irrep_tensor_product(
                pair_sums_irrep['v2b'], cdu @ cdd, branch_tensor, input_qn, output_qn, local_operator_shift(('Cdu', 'Cdd'))
            )[0]
            v3u = sum_tensor_terms(
                site,
                OPERATOR_QN_SHIFT['Cdu'],
                [(eri_value(eri, i, site_id, site_id, site_id), irrep_table[(('Cdu',), (i,))]) for i in range(site_id)],
            )
            v3d = sum_tensor_terms(
                site,
                OPERATOR_QN_SHIFT['Cdd'],
                [(eri_value(eri, i, site_id, site_id, site_id), irrep_table[(('Cdd',), (i,))]) for i in range(site_id)],
            )
            V3u = rotate_irrep_tensor_product(
                v3u, JW @ Nd @ cu, branch_tensor, input_qn, output_qn, OPERATOR_QN_SHIFT['Cu']
            )[0]
            V3d = rotate_irrep_tensor_product(
                v3d, JW @ Nu @ cd, branch_tensor, input_qn, output_qn, OPERATOR_QN_SHIFT['Cd']
            )[0]
            Hnew_irrep = add_tensors(
                hnew_diag_irrep,
                V1u, V1u.adjoint(),
                V1d, V1d.adjoint(),
                V2a, V2a.adjoint(),
                V2b, V2b.adjoint(),
                V3u, V3u.adjoint(),
                V3d, V3d.adjoint(),
            )
            Hnew = labeled_dense(Hnew_irrep, output_qn)
        else:
            Hnew = Hnew_diag

            op_lists = operator_lists(table)
            Cdu = op_lists['Cdu']
            Cdd = op_lists['Cdd']
            Cu = op_lists['Cu']
            Cd = op_lists['Cd']

            v1u = zero_like_operator(H0)
            v1d = zero_like_operator(H0)
            for i in range(site_id):
                v1u += h1e[i, site_id] * dense_operator(Cdu[i])
                v1d += h1e[i, site_id] * dense_operator(Cdd[i])
            v1u += residuals[site_id][0]
            v1d += residuals[site_id][1]

            V1 = (
                rotate_symmetry(v1u, JW @ cu, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
                + rotate_symmetry(v1d, JW @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
            )
            Hnew += V1 + dag(V1)

            V2a = rotate_symmetry(pair_sums['v2a'], cdu @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
            V2b = rotate_symmetry(pair_sums['v2b'], cdu @ cdd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
            Hnew += V2a + dag(V2a) + V2b + dag(V2b)

            v3u = zero_like_operator(H0)
            v3d = zero_like_operator(H0)
            for i in range(site_id):
                v3u += eri_value(eri, i, site_id, site_id, site_id) * dense_operator(Cdu[i])
                v3d += eri_value(eri, i, site_id, site_id, site_id) * dense_operator(Cdd[i])
            V3 = (
                rotate_symmetry(v3u, JW @ Nd @ cu, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
                + rotate_symmetry(v3d, JW @ Nu @ cd, branch_tensor, output_qn, scalar_shift, use_irrep_blocks, plan)
            )
            Hnew += V3 + dag(V3)
            Hnew_irrep = labeled_irrep_tensor(Hnew, output_qn, op=(0, 0)) if use_irrep_operator_table else None
        branch_qn = np.stack((qn0, qn1, qn2, qn3), axis=0).copy()
        projector = None
        cc_diagnostics = None
        if dressing_key == "conditional_cc":
            h_grown, primitive_qn = _compressed_superblock_one_site_lloo(
                H0,
                input_qn,
                table,
                residuals,
                site_id,
                h1e,
                eri,
                pair_terms,
            )
            projector = np.zeros(
                (H0.shape[0], d, d * keep),
                dtype=np.result_type(branch_tensor, complex),
            )
            for local_state in range(d):
                columns = slice(local_state * keep, (local_state + 1) * keep)
                projector[:, local_state, columns] = branch_tensor[:, :, local_state]
            Hnew, projector, cc_diagnostics = conditional_cc_transition_projector(
                h_grown,
                primitive_qn,
                projector,
                output_qn,
                level_shift=cc_level_shift,
                response_tol=cc_response_tol,
                max_responses=cc_max_responses,
            )
            Hnew_irrep = None
            branch_tensor = projector.transpose(0, 2, 1).copy()
        return (
            Hnew,
            Hnew_irrep,
            branch_tensor,
            output_qn,
            branch_qn,
            projector,
            cc_diagnostics,
        )

    class AbelianGrowth(NARGBase):
        def __init__(self, table, irrep_table, table_qn, h_irrep, residuals, irrep_residuals, spins, table_nsites):
            super().__init__(
                D=D,
                growth_sites=growth_sites,
                two_site_dim=two_site_intermediate_dim,
                two_site_max_dim=two_site_max_dim,
                site_dim=d,
                two_site_mode=two_site_mode,
            )
            self.table = table
            self.irrep_table = irrep_table
            self.table_qn = table_qn
            self.h_irrep = h_irrep
            self.residuals = residuals
            self.irrep_residuals = irrep_residuals
            self.spins = spins
            self.table_nsites = int(table_nsites)

        def before_site(self, block, site):
            if self.table_nsites >= site.idx:
                return block
            if self.table_nsites != site.idx - 1:
                raise RuntimeError(
                    f"operator table contains {self.table_nsites} sites, cannot prepare site {site.idx}."
                )
            (
                self.table,
                self.irrep_table,
                self.table_qn,
                self.residuals,
                self.irrep_residuals,
                self.spins,
            ) = extend_current_block(
                self.table,
                self.irrep_table,
                self.table_qn,
                self.residuals,
                self.irrep_residuals,
                self.spins,
                block.tensor,
                site.idx - 1,
                block.qn,
            )
            self.table_nsites = site.idx
            return block

        def choose_growth_sites(self, block, site, remaining_sites):
            if self.growth_sites != "auto" or remaining_sites < 2:
                return super().choose_growth_sites(block, site, remaining_sites)
            gap = abs(orbital_energy[site.idx + 1] - orbital_energy[site.idx])
            if gap > two_site_energy_tol:
                return 1
            if self.two_site_max_dim is not None:
                intermediate = self.full_dim(block, site) if self.two_site_dim is None else self.two_site_dim
                if intermediate > self.two_site_max_dim:
                    return 1
            return 2

        def grow_one(self, block, site, keep):
            maybe_print(verbose, '\n--- adding the {}th orbital ---'.format(site.idx + 1))
            maybe_print(verbose, 'p = ', site.idx)
            (
                h_new,
                h_new_irrep,
                tensor,
                qn,
                branch_qn,
                projector,
                cc_diagnostics,
            ) = append_one_site(
                block.h.copy(),
                self.h_irrep,
                block.qn,
                self.table,
                self.irrep_table,
                self.residuals,
                self.irrep_residuals,
                site.idx,
                keep,
            )
            self.h_irrep = h_new_irrep
            if projector is not None:
                next_residuals = extend_triple_residuals_projector(
                    self.residuals,
                    self.table,
                    projector,
                    site.idx,
                    L,
                    triple_terms,
                )
                required_entries = (
                    required_operator_entries(
                        pair_terms, triple_terms, L, site.idx + 1
                    )
                    if sparse_operator_table
                    else None
                )
                self.table = extend_operator_table_projector(
                    self.table,
                    OPERATOR_PATTERNS,
                    projector,
                    site.idx,
                    qn,
                    required=required_entries,
                    sparse_output=use_sparse_operator_projection,
                    consume=True,
                )
                self.table_qn = qn
                self.irrep_table = None
                self.residuals = next_residuals
                self.irrep_residuals = None
                self.table_nsites = site.idx + 1
            meta = {}
            if cc_diagnostics is not None:
                meta["dressing"] = dressing_key
                if dressing_key == "detached_cc":
                    meta["detached_diagnostics"] = dict(
                        cc_diagnostics["detached_diagnostics"]
                    )
                    meta["cc_diagnostics"] = dict(
                        cc_diagnostics["cc_diagnostics"]
                    )
                else:
                    diagnostics_key = (
                        "cc_diagnostics"
                        if dressing_key == "conditional_cc"
                        else "detached_diagnostics"
                    )
                    meta[diagnostics_key] = dict(cc_diagnostics)
                meta["general_projector"] = True
            return Step(
                site=site,
                block=Block(h=h_new, qn=qn, tensor=tensor),
                tensor=None if tensor is None else tensor.copy(),
                qn=branch_qn,
                meta=meta,
            )

        def grow_two(self, block, first, second, keep):
            if use_irrep_operator_table:
                raise NotImplementedError("qchem two-site growth does not yet support use_irrep_operator_table.")
            maybe_print(verbose, '\n--- adding orbitals {} and {} together ---'.format(first.idx + 1, second.idx + 1))
            if self.two_site_mode == "rolling":
                (
                    h_new,
                    tensor,
                    projector,
                    qn,
                    branch_qn,
                    local_qn,
                    branch_energy,
                    rolling_energy,
                    cc_diagnostics,
                ) = append_rolling_two_sites(
                    block.h.copy(),
                    block.qn,
                    self.table,
                    self.residuals,
                    first.idx,
                    keep,
                )
                plan = None
            else:
                h_new, tensor, tensor3, qn, branch_qn, local_qn, branch_energy = append_supersite(
                    block.h.copy(),
                    block.qn,
                    self.table,
                    self.residuals,
                    first.idx,
                    keep,
                )
                projector = None
                rolling_energy = None
                cc_diagnostics = None
                plan = RotationPlan(tensor3, qn)
            required_entries = (
                required_operator_entries(pair_terms, triple_terms, L, second.idx + 1)
                if sparse_operator_table
                else None
            )
            if need_spin:
                required_entries = add_initial_spin_entries(required_entries, second.idx + 1)
            if self.two_site_mode == "rolling":
                next_residuals = extend_triple_residuals_two_site_projector(
                    self.residuals,
                    self.table,
                    projector,
                    first.idx,
                    L,
                    triple_terms,
                )
                self.table = extend_operator_table_two_site_projector(
                    self.table,
                    OPERATOR_PATTERNS,
                    projector,
                    first.idx,
                    qn,
                    required=required_entries,
                    sparse_output=use_sparse_operator_projection,
                )
            else:
                self.table = extend_pair_table(
                    self.table,
                    OPERATOR_PATTERNS,
                    tensor3,
                    first.idx,
                    qn,
                    use_irrep_blocks,
                    plan,
                    required=required_entries,
                    sparse_output=use_sparse_operator_projection,
                )
            self.table_qn = qn
            self.h_irrep = None
            if self.two_site_mode == "rolling":
                self.residuals = next_residuals
            else:
                self.residuals = build_triple_residuals_from_table(
                    self.table,
                    second.idx + 1,
                    range(second.idx + 1, L),
                    triple_terms,
                )
            self.irrep_residuals = None
            if need_spin:
                if self.two_site_mode == "rolling":
                    raise NotImplementedError("rolling two-site qchem NARG does not yet support spin observables.")
                self.spins = extend_spin_operators_two_site(self.spins, tensor3)
            self.table_nsites = second.idx + 1
            meta = {
                "local_qn": local_qn.copy(),
                "local_dim": d if self.two_site_mode == "rolling" else d * d,
                "block_dim": h_new.shape[0],
                "two_site_mode": self.two_site_mode,
                "branch_energy": branch_energy.copy(),
            }
            if rolling_energy is not None:
                meta["rolling_energy"] = rolling_energy.copy()
                meta["temporary_local_dim"] = d * d
            if cc_diagnostics is not None:
                meta["dressing"] = dressing_key
                meta["cc_diagnostics"] = dict(cc_diagnostics)
            return Step(
                site=first,
                block=Block(h=h_new, qn=qn, tensor=tensor),
                tensor=tensor,
                qn=branch_qn,
                meta=meta,
            )

    Htot_irrep = labeled_irrep_tensor(H0, block_qn, op=(0, 0)) if use_irrep_operator_table else None
    growth = AbelianGrowth(
        op_table, irrep_op_table, op_table_qn, Htot_irrep, triple_residuals, irrep_triple_residuals, spin_ops, nstart
    )
    block = Block(h=H0, qn=block_qn, tensor=None)
    for first_site, last_site in ((nstart, nstart), (nstart + 1, L - 1)):
        for step in growth.grow_range(block, first_site, last_site):
            block = step.block
            narg_tensors.append(step.tensor)
            tensor_qns.append(step.meta)

    Htot = block.h
    htot_qn = block.qn
    U = block.tensor
    op_table = growth.table
    triple_residuals = growth.residuals
    irrep_triple_residuals = growth.irrep_residuals
    spin_ops = growth.spins
    table_nsites = growth.table_nsites
    
    ###############################
    
    
    ### Final diagonalization
    
    
    # nroots = 20
    final_nstates = nstates
    if target_spin is not None:
        final_nstates = max(nstates, int(np.ceil(nstates * spin_search_factor)))
    if use_irrep_tensor == 'auto':
        final_irrep_diagonalization = irrep_tensor_available()
    else:
        final_irrep_diagonalization = bool(use_irrep_tensor)

    if final_irrep_diagonalization:
        if not irrep_tensor_available():
            raise ImportError("use_irrep_tensor=True requires the irrep_tensor module.")
        E, X, final_qn, _final_irrep_tensor = irrep_scalar_diagonalize(
            Htot, htot_qn, final_nstates, allowed_qn={target_qn}
        )
        maybe_print(verbose, 'Final diagonalization: IrrepTensor U1xU1 blocks')
    else:
        E, X, final_qn = charge_diagonalize(Htot, htot_qn, final_nstates, allowed_qn={target_qn})

    spin_info = None
    if need_spin:
        if table_nsites >= L:
            S2 = spin_ops['s2']
        else:
            S2 = final_spin_square_operator(spin_ops, U)
        s2, spin = spin_expectations(S2, X)
        if target_spin is not None:
            E, X, final_qn, s2, spin = filter_spin_roots(
                E, X, final_qn, s2, spin, target_spin, spin_tol, nstates
            )
        else:
            E = E[:nstates]
            X = X[:, :nstates]
            final_qn = final_qn[:nstates]
            s2 = s2[:nstates]
            spin = spin[:nstates]
        spin_info = {'s2': s2, 'spin': spin}
    else:
        E = E[:nstates]
        X = X[:, :nstates]
        final_qn = final_qn[:nstates]
    
    maybe_print(verbose, 'NARG quantum numbers [Ne, 2Sz] = ', final_qn)
    maybe_print(verbose, 'NARG energy = ', E + mol.energy_nuc())
    if spin_info is not None:
        maybe_print(verbose, 'NARG <S^2> = ', spin_info['s2'])
        maybe_print(verbose, 'NARG spin S = ', spin_info['spin'])

    result = [E + mol.energy_nuc(), X]
    if return_spin:
        result.append(spin_info)
    if return_tensors or return_tensor_qns:
        if tensor_qns[-1].get("general_projector", False):
            terminal_local_shape = (1,)
            terminal_local_dim = 1
            final_bond_dim = len(htot_qn)
        else:
            final_bond_dim = narg_tensors[-1].shape[1]
            final_local_shape = tuple(int(dim) for dim in narg_tensors[-1].shape[2:])
            final_local_dim = int(np.prod(final_local_shape))
            final_x_rows = np.asarray(X).shape[0]
            if final_x_rows == final_local_dim * final_bond_dim:
                terminal_local_shape = final_local_shape
                terminal_local_dim = final_local_dim
            elif final_local_shape and final_x_rows == final_local_shape[-1] * final_bond_dim:
                terminal_local_shape = (final_local_shape[-1],)
                terminal_local_dim = final_local_shape[-1]
            else:
                raise ValueError(
                    "final eigenvector shape is incompatible with the returned NARG tensor dimensions."
                )
    if return_tensors:
        coeff = np.asarray(X).reshape(terminal_local_dim, final_bond_dim, X.shape[1])
        result.append(narg_tensors + [coeff])
    if return_tensor_qns:
        terminal_total_qn = qn_array(htot_qn).reshape(terminal_local_dim, final_bond_dim, -1)
        if terminal_local_shape == (d,):
            terminal_local_qn = LOCAL_QN.copy()
        elif all(dim == d for dim in terminal_local_shape):
            terminal_local_qn = primitive_charge_labels(len(terminal_local_shape))
        else:
            terminal_local_qn = np.empty((terminal_local_dim, LOCAL_QN.shape[1]), dtype=int)
        result.append(
            {
                "factors": tensor_qns,
                "terminal_total_qn_by_site": terminal_total_qn.copy(),
                "local_qn": terminal_local_qn,
                "local_shape": terminal_local_shape,
                "target_qn": np.asarray(target_qn, dtype=int),
                "final_qn": np.asarray(final_qn, dtype=int),
            }
        )
    return tuple(result)


def _normalize_narg_orbital_blocks(orbital_blocks, nsites, active_space=None):
    """Normalize orbital blocks into active-space coordinates."""
    if orbital_blocks is None:
        return None
    nsites = int(nsites)
    try:
        return normalize_orbital_blocks(orbital_blocks, norb=nsites)
    except ValueError as direct_error:
        if active_space is None:
            raise direct_error
        ncore = int(active_space.ncore)
        shifted = tuple(tuple(int(i) - ncore for i in block) for block in orbital_blocks)
        try:
            return normalize_orbital_blocks(shifted, norb=nsites)
        except ValueError:
            raise direct_error from None


class NARG:
    """Small object API for the Abelian quantum-chemistry NARG driver.

    Examples
    --------
    >>> e_tot, x = NARG(mf, D=120, nstates=20).run()
    >>> e_tot, x, spin = NARG(mf).run(D=120, return_spin=True)
    """

    DEFAULT_OPTIONS = {
        "D": 20,
        "n0": 4,
        "nstates": 1,
        "verbose": False,
        "growth_sites": 1,
        "two_site_intermediate_dim": None,
        "two_site_max_dim": None,
        "two_site_energy_tol": None,
        "two_site_gap_factor": 1.0,
        "eri_cutoff": 0.0,
        "fast": False,
        "use_numba_terms": "auto",
        "use_irrep_tensor": False,
        "use_irrep_blocks": False,
        "use_irrep_operator_table": False,
        "sparse_operator_table": "auto",
        "use_block_sparse_hamiltonian": False,
        "use_sparse_operator_projection": "auto",
        "two_site_mode": "supersite",
        "dressing": None,
        "chi": None,
        "frame_adapt_tol": None,
        "frame_max_dim": None,
        "frame_expand_dim": 1,
        "cc_level_shift": 0.0,
        "cc_response_tol": 1.0e-10,
        "cc_max_responses": None,
        "target_spin": None,
        "spin_tol": 1e-3,
        "spin_search_factor": 4,
        "return_spin": False,
        "store_tensors": "auto",
        "orbital_blocks": None,
        **CAS_OPTION_DEFAULTS,
    }

    def __init__(self, mf, *, mol=None, h1e=None, eri=None, **options):
        self.mf = mf
        self.mol = mol if mol is not None else getattr(mf, "mol", None)
        self.h1e = h1e
        self.eri = eri
        self.options = dict(self.DEFAULT_OPTIONS)
        self.options.update(options)
        self.e_tot = None
        self.vectors = None
        self.spin_info = None
        self.tensors = None
        self.tensor_qns = None
        self.local_dims = None
        self.orbital_blocks = None
        self.dressing_history = []
        self.detached_history = []
        self.site = "spatial"
        self.n0 = None
        self.active_space = None
        self.ncas = None
        self.nelecas = None
        self.ncore = None
        self.mo_core = None
        self.mo_cas = None
        self.e_core = None
        self.result = None

    def integrals(self):
        """Return MO one- and two-electron integrals for the wrapped mean field."""
        opts = dict(self.options)
        cas_options = pop_active_space_options(opts)
        h1e, eri, _, _ = prepare_active_space(
            self.mf,
            self.mol,
            h1e=self.h1e,
            eri=self.eri,
            prefer_factors=True,
            **cas_options,
        )
        return h1e, eri

    def _set_active_space(self, active_space):
        self.active_space = active_space
        if active_space is None:
            self.ncas = self.nelecas = self.ncore = None
            self.mo_core = self.mo_cas = None
            self.e_core = None
            return
        self.ncas = active_space.ncas
        self.nelecas = active_space.nelecas
        self.ncore = active_space.ncore
        self.mo_core = active_space.mo_core
        self.mo_cas = active_space.mo_cas
        self.e_core = active_space.energy_core

    def run(self, **options):
        """Run NARG and return the same tuple as ``kernel``."""
        opts = dict(self.options)
        opts.update(options)
        cas_options = pop_active_space_options(opts)
        h1e = opts.pop("h1e", None)
        eri = opts.pop("eri", None)

        global mol
        active_mol = opts.pop("mol", None)
        if active_mol is not None:
            self.mol = active_mol
        if self.mol is None:
            self.mol = getattr(self.mf, "mol", None)
        if self.mol is None:
            raise ValueError("NARG needs a Molecule; pass NARG(mf, mol=mol) or run(..., mol=mol).")

        h1e, eri, prepared_mol, active_space = prepare_active_space(
            self.mf,
            self.mol,
            h1e=h1e,
            eri=eri,
            prefer_factors=True,
            **cas_options,
        )
        self.h1e = h1e
        self.eri = eri
        self.mol = prepared_mol
        self._set_active_space(active_space)
        mol = self.mol
        nsites = int(np.asarray(h1e).shape[-1])
        orbital_blocks = _normalize_narg_orbital_blocks(
            opts.pop("orbital_blocks", self.DEFAULT_OPTIONS["orbital_blocks"]),
            nsites,
            active_space=active_space,
        )
        if int(opts.get("n0", self.DEFAULT_OPTIONS["n0"])) >= nsites:
            if nsites < 2:
                raise ValueError("QChem NARG needs at least two spatial orbitals.")
            opts["n0"] = nsites - 1

        return_spin = bool(opts.get("return_spin", False))
        dressing_requested = opts.get("dressing", None) not in {
            None,
            False,
            "none",
            "off",
            "false",
        }
        store_tensors_option = opts.pop(
            "store_tensors", self.DEFAULT_OPTIONS["store_tensors"]
        )
        if isinstance(store_tensors_option, str):
            if store_tensors_option != "auto":
                raise ValueError("store_tensors must be True, False, or 'auto'.")
            mode = str(opts.get("two_site_mode", "supersite")).lower().replace("-", "_")
            store_tensors = (
                mode not in {
                    "rolling",
                    "two_site",
                    "true_two_site",
                    "rolling_two_site",
                }
            )
        else:
            store_tensors = bool(store_tensors_option)
        if orbital_blocks is not None:
            if return_spin or opts.get("target_spin", None) is not None:
                raise NotImplementedError(
                    "Clustered Abelian NARG via orbital_blocks does not yet carry spin observables; "
                    "run without return_spin/target_spin or use unclustered growth."
                )
            supersite_result = reduced_supersite_kernel(
                h1e,
                eri,
                orbital_blocks,
                D=opts.get("D", self.DEFAULT_OPTIONS["D"]),
                nstates=opts.get("nstates", self.DEFAULT_OPTIONS["nstates"]),
                nelec=getattr(self.mol, "nelec", None),
                eri_cutoff=opts.get("eri_cutoff", self.DEFAULT_OPTIONS["eri_cutoff"]),
                fast=opts.get("fast", self.DEFAULT_OPTIONS["fast"]),
                use_numba_terms=opts.get("use_numba_terms", self.DEFAULT_OPTIONS["use_numba_terms"]),
                use_irrep_tensor=opts.get("use_irrep_tensor", self.DEFAULT_OPTIONS["use_irrep_tensor"]),
                use_irrep_blocks=opts.get("use_irrep_blocks", self.DEFAULT_OPTIONS["use_irrep_blocks"]),
                sparse_operator_table=opts.get("sparse_operator_table", self.DEFAULT_OPTIONS["sparse_operator_table"]),
                use_sparse_operator_projection=opts.get(
                    "use_sparse_operator_projection",
                    self.DEFAULT_OPTIONS["use_sparse_operator_projection"],
                ),
                return_tensors=store_tensors,
                return_tensor_qns=store_tensors,
            )
            enuc = self.mol.energy_nuc() if self.mol is not None else 0.0
            if store_tensors:
                e_elec, self.vectors, self.tensors, self.tensor_qns = supersite_result
            else:
                e_elec, self.vectors = supersite_result
                self.tensors = None
                self.tensor_qns = None
            self.e_tot = np.asarray(e_elec) + enuc
            self.spin_info = None
            self.result = (self.e_tot, self.vectors)
            self.n0 = None
            self.local_dims = tuple(4 ** len(block) for block in orbital_blocks)
            self.orbital_blocks = orbital_blocks
            self.dressing_history = []
            self.detached_history = []
            return self.result

        return_metadata = store_tensors or dressing_requested
        opts["return_tensors"] = store_tensors
        opts["return_tensor_qns"] = return_metadata

        kernel_result = kernel(h1e, eri, **opts)
        if return_spin:
            if store_tensors:
                self.e_tot, self.vectors, self.spin_info, self.tensors, self.tensor_qns = kernel_result
            else:
                self.e_tot, self.vectors, self.spin_info = kernel_result
                self.tensor_qns = None
            self.result = (self.e_tot, self.vectors, self.spin_info)
        else:
            if store_tensors:
                self.e_tot, self.vectors, self.tensors, self.tensor_qns = kernel_result
            elif return_metadata:
                self.e_tot, self.vectors, self.tensor_qns = kernel_result
                self.tensors = None
            else:
                self.e_tot, self.vectors = kernel_result
                self.tensor_qns = None
                self.tensors = None
            self.spin_info = None
            self.result = (self.e_tot, self.vectors)
        self.n0 = int(opts.get("n0", self.DEFAULT_OPTIONS["n0"]))
        self.local_dims = (4,) * int(np.asarray(h1e).shape[-1])
        self.orbital_blocks = None
        factors = [] if self.tensor_qns is None else self.tensor_qns.get("factors", [])
        self.dressing_history = [
            dict(factor["cc_diagnostics"])
            for factor in factors
            if "cc_diagnostics" in factor
        ]
        self.detached_history = [
            dict(factor["detached_diagnostics"])
            for factor in factors
            if "detached_diagnostics" in factor
        ]
        return self.result

    def make_rdm1(
        self,
        state_id=0,
        spatial=True,
        with_core=False,
        with_vir=False,
        representation="mo",
        repr=None,
    ):
        """Return the spin-traced 1-RDM for the selected NARG root."""
        return make_rdm1_from_narg(
            self,
            state_id=state_id,
            with_core=with_core,
            with_vir=with_vir,
            representation=representation,
            repr=repr,
        )

    def make_rdm2(self, state_id=0, spatial=True, with_core=False, with_vir=False):
        """Return the spin-traced 2-RDM for the selected NARG root."""
        return make_rdm2_from_narg(
            self,
            state_id=state_id,
            with_core=with_core,
            with_vir=with_vir,
        )

    def make_rdm12(self, state_id=0, spatial=True, with_core=False):
        return (
            self.make_rdm1(state_id, spatial=spatial, with_core=with_core),
            self.make_rdm2(state_id, spatial=spatial, with_core=with_core),
        )
