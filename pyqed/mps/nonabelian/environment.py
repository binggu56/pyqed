#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense environment and effective-H helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional runtime acceleration
    sp = None

from .contraction import normalize_site_tensor_layout
from .mpo import MPO, IrreducibleMPO, RankCoupledMPO
from .solver import (
    LocalOperator,
    ReducedStateVector,
    pack_two_site_state,
    unpack_two_site_state,
)
from .tensor import NonabelianTensor


def _cached_einsum_path(signature, *shapes):
    operands = [np.zeros(shape, dtype=float) for shape in shapes]
    path, _ = np.einsum_path(signature, *operands, optimize="greedy")
    return path


_LEFT_BLOCK_PATH = _cached_einsum_path(
    "xij,iap,xypq,jbq->yab",
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2),
)
_RIGHT_BLOCK_PATH = _cached_einsum_path(
    "iap,xypq,yab,jbq->xij",
    (2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)
_TWO_SITE_KERNEL_PATH = _cached_einsum_path(
    "xal,xypo,yzqr,zbc->apqblorc",
    (2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2),
)
_IDENTITY_TWO_SITE_KERNEL_PATH = _cached_einsum_path(
    "al,po,qr,bc->apqblorc",
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
)
_FACTORIZED_PACKED_LOCAL_DIM = 2048
_PACKED_DENSE_LOCAL_DIM = 512
_PACKED_CSR_LOCAL_DIM = 128
_FACTORIZED_TWO_SITE_BATCH_PATH = _cached_einsum_path(
    "talwpi,lijr,twbjqr->apbq",
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
)
_DIAG_BLOCK_PATH = _cached_einsum_path(
    "al,abp,bcq,cr->lprq",
    (2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2),
)


def _tensor_dense_layout(tensor, axis_overrides=None):
    if not isinstance(tensor, NonabelianTensor):
        raise TypeError("_tensor_dense_layout expects a NonabelianTensor.")
    axis_overrides = axis_overrides or {}

    sector_dims = [dict() for _ in range(tensor.rank)]
    for key, block in tensor.data.items():
        for axis, sector in enumerate(key):
            dim = int(block.shape[axis])
            known = sector_dims[axis].get(sector)
            if known is None:
                sector_dims[axis][sector] = dim
            elif known != dim:
                raise ValueError(
                    f"Inconsistent dense dimension for sector {sector!r} on axis {axis}: "
                    f"{known} vs {dim}."
                )

    sector_slices = []
    dense_shape = []
    for axis, leg_sectors in enumerate(tensor.qns):
        axis_slices = {}
        offset = 0
        override = axis_overrides.get(axis)
        if override is not None:
            ordered_sectors = tuple(override["sectors"])
            dims_for_axis = dict(override["dims"])
        else:
            ordered_sectors = tuple(leg_sectors)
            dims_for_axis = sector_dims[axis]
        for sector in ordered_sectors:
            if sector in axis_slices:
                continue
            dim = dims_for_axis.get(sector)
            if dim is None:
                if tensor.rank == 3 and axis == 1 and hasattr(sector, "dim"):
                    # Physical legs must keep their full local Hilbert-space
                    # sectors even when the current MPS has zero amplitude in
                    # one sector; otherwise dense MPO -> sparse MPO conversion
                    # silently drops operator channels.
                    dim = int(sector.dim)
                else:
                    dim = sum(1 for item in leg_sectors if item == sector)
            if dim is None:
                continue
            axis_slices[sector] = slice(offset, offset + dim)
            offset += dim
        if offset == 0:
            raise ValueError(
                f"Cannot build dense layout for axis {axis}: no block dimensions were found."
            )
        sector_slices.append(axis_slices)
        dense_shape.append(offset)

    block_slices = {}
    for key in tensor.data:
        block_slices[key] = tuple(
            sector_slices[axis][sector] for axis, sector in enumerate(key)
        )

    return {
        "sector_dims": sector_dims,
        "sector_slices": sector_slices,
        "block_slices": block_slices,
        "shape": tuple(dense_shape),
    }


def _site_physical_override(site_layout, site):
    return {
        "sectors": tuple(site.qns[1]),
        "dims": dict(site_layout["sector_dims"][1]),
    }


def _site_physical_dims(site_layout):
    return dict(site_layout["sector_dims"][1])


def _mpo_left_dim(mpo_core):
    if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
        return mpo_core.left_dim
    return int(np.asarray(mpo_core).shape[0])


def _mpo_right_dim(mpo_core):
    if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
        return mpo_core.right_dim
    return int(np.asarray(mpo_core).shape[1])


def _mpo_dtype(mpo_core):
    if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
        return mpo_core.dtype
    return np.asarray(mpo_core).dtype


def _is_identity_mpo_core(mpo_core, *, tol=1e-12):
    if not isinstance(mpo_core, MPO):
        return False
    if mpo_core.left_dim != 1 or mpo_core.right_dim != 1:
        return False
    for q_out in mpo_core.phys_out_sectors:
        for q_in in mpo_core.phys_in_sectors:
            block = mpo_core.block(q_out, q_in)
            if q_out == q_in:
                if block is None:
                    return False
                eye = np.eye(mpo_core.phys_out_leg.dim(q_out), dtype=np.asarray(block).dtype)
                if not np.allclose(np.asarray(block)[0, 0], eye, atol=tol, rtol=tol):
                    return False
            elif block is not None and np.linalg.norm(np.asarray(block).reshape(-1)) > tol:
                return False
    return True


def _normalize_dense_mpo_core(mpo_core, *, phys_out_slices, phys_in_slices=None):
    if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
        return mpo_core.as_dense(phys_out_slices, phys_in_slices)
    return np.asarray(mpo_core)


def _normalize_block_sparse_mpo_core(
    mpo_core,
    *,
    phys_out_slices=None,
    phys_in_slices=None,
    phys_out_dims=None,
    phys_in_dims=None,
):
    if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
        return mpo_core
    return MPO.from_dense(
        mpo_core,
        phys_out_slices=phys_out_slices,
        phys_in_slices=phys_in_slices,
        phys_out_dims=phys_out_dims,
        phys_in_dims=phys_in_dims,
    )


def _normalize_dense_mpo_factors(mpo_factors, *, site_layouts):
    return [
        _normalize_dense_mpo_core(
            mpo_core,
            phys_out_slices=site_layout["sector_slices"][1],
        )
        for mpo_core, site_layout in zip(mpo_factors, site_layouts)
    ]


def _normalize_block_sparse_mpo_factors(mpo_factors, *, site_layouts):
    return [
        _normalize_block_sparse_mpo_core(
            mpo_core,
            phys_out_slices=site_layout["sector_slices"][1],
            phys_in_slices=site_layout["sector_slices"][1],
        )
        for mpo_core, site_layout in zip(mpo_factors, site_layouts)
    ]


def _is_rank_coupled_chain(mpo_factors):
    return bool(mpo_factors) and all(isinstance(core, RankCoupledMPO) for core in mpo_factors)




def _tensor_to_dense(tensor, dense_layout=None):
    if dense_layout is None:
        dense_layout = _tensor_dense_layout(tensor)
    dtype = np.result_type(*(np.asarray(block).dtype for block in tensor.data.values()))
    dense = np.zeros(dense_layout["shape"], dtype=dtype)
    for key, block in tensor.data.items():
        slices = dense_layout["block_slices"].get(key)
        if slices is None:
            slices = tuple(
                dense_layout["sector_slices"][axis][sector]
                for axis, sector in enumerate(key)
            )
        dense[slices] = np.asarray(block)
    return dense


def _packed_vector_to_dense(vector, *, pack_layout, dense_layout):
    vector = np.asarray(vector)
    dense = np.zeros(dense_layout["shape"], dtype=vector.dtype)
    for entry in pack_layout:
        dense[dense_layout["block_slices"][entry.key]] = vector[
            entry.offset:entry.offset + entry.size
        ].reshape(entry.shape)
    return dense


def _dense_to_packed_vector(dense, *, pack_layout, dense_layout):
    dense = np.asarray(dense)
    dtype = dense.dtype
    packed = np.zeros(sum(entry.size for entry in pack_layout), dtype=dtype)
    for entry in pack_layout:
        packed[entry.offset:entry.offset + entry.size] = dense[
            dense_layout["block_slices"][entry.key]
        ].reshape(entry.size)
    return packed


def _site_to_dense(site, dense_layout=None):
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise ValueError("_site_to_dense expects a rank-3 NonabelianTensor.")
    return _tensor_to_dense(site, dense_layout=dense_layout)


def _initial_left_env(W):
    E = np.zeros((_mpo_left_dim(W), 1, 1), dtype=_mpo_dtype(W))
    E[0, 0, 0] = 1.0
    return E


def _initial_right_env(W):
    F = np.zeros((_mpo_right_dim(W), 1, 1), dtype=_mpo_dtype(W))
    F[0, 0, 0] = 1.0
    return F


def _contract_from_left_dense(W, A, E, B):
    # A, B: (L, P, R)
    T1 = np.tensordot(E, A.conj(), axes=(1, 0))
    T2 = np.tensordot(T1, W, axes=([0, 2], [0, 2]))
    E_new = np.tensordot(T2, B, axes=([0, 3], [0, 1]))
    return E_new.transpose(1, 0, 2)


def _contract_from_right_dense(W, A, F, B):
    # A, B: (L, P, R)
    T1 = np.tensordot(F, A.conj(), axes=(1, 2))
    T2 = np.tensordot(T1, W, axes=([0, 3], [1, 2]))
    F_new = np.tensordot(T2, B, axes=([0, 3], [2, 1]))
    return F_new.transpose(1, 0, 2)


def _apply_two_site_dense(E, W1, W2, F, theta):
    """
    Apply the effective two-site Hamiltonian to a dense two-site tensor.

    Parameters
    ----------
    E, F
        Dense left/right environments with shape ``(w, bra, ket)``.
    W1, W2
        Dense MPO tensors with shape ``(wL, wR, pOut, pIn)``.
    theta
        Dense two-site tensor with layout ``(left, phys1, phys2, right)``.
    """
    T1 = np.tensordot(E, theta, axes=(2, 0))
    # T1: (wL, braL, p1in, right, p2in)
    T2 = np.tensordot(T1, W1, axes=([0, 2], [0, 3]))
    # T2: (braL, p2in, right, w1R, p1out)
    T3 = np.tensordot(T2, W2, axes=([1, 3], [3, 0]))
    # T3: (braL, right, p1out, w2R, p2out)
    T4 = np.tensordot(T3, F, axes=([1, 3], [2, 0]))
    # T4: (braL, p1out, p2out, braR)
    return T4


def _factorize_two_site_dense_term(E, W1, W2, F):
    """
    Precontract one two-site effective-operator term into left/right factors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(LHeff, RHeff)`` with shapes
        ``(bra_left, ket_left, w_mid, p1_out, p1_in)`` and
        ``(w_mid, bra_right, ket_right, p2_out, p2_in)``.
    """
    left = np.tensordot(np.asarray(E), np.asarray(W1), axes=([0], [0]))
    right = np.tensordot(np.asarray(W2), np.asarray(F), axes=([1], [0]))
    return left, np.transpose(right, (0, 3, 4, 1, 2))


def _factorize_left_two_site_dense_term(E, W1):
    return np.tensordot(np.asarray(E), np.asarray(W1), axes=([0], [0]))


def _factorize_right_two_site_dense_term(W2, F):
    right = np.tensordot(np.asarray(W2), np.asarray(F), axes=([1], [0]))
    return np.transpose(right, (0, 3, 4, 1, 2))


def _apply_two_site_dense_factorized(LHeff, RHeff, theta):
    """
    Apply a precontracted factorized two-site effective operator term.
    """
    tmp = np.tensordot(np.asarray(LHeff), np.asarray(theta), axes=([1, 4], [0, 1]))
    out = np.tensordot(
        tmp,
        np.asarray(RHeff),
        axes=([1, 3, 4], [0, 4, 2]),
    )
    return np.transpose(out, (0, 1, 3, 2))


def _two_site_diagonal_block(E, W1, W2, F):
    """
    Return the diagonal action of the effective two-site operator block.

    Parameters
    ----------
    E, F
        Environment blocks with shape ``(w, bra, ket)``.
    W1, W2
        MPO blocks with shape ``(wL, wR, pOut, pIn)``.

    Returns
    -------
    np.ndarray
        Block-shaped diagonal with layout ``(left, phys1, phys2, right)``.
    """
    E_diag = np.diagonal(np.asarray(E), axis1=1, axis2=2)
    W1_diag = np.diagonal(np.asarray(W1), axis1=2, axis2=3)
    W2_diag = np.diagonal(np.asarray(W2), axis1=2, axis2=3)
    F_diag = np.diagonal(np.asarray(F), axis1=1, axis2=2)
    tmp = np.tensordot(E_diag, W1_diag, axes=([0], [0]))
    # tmp: (left, w1R, p1)
    tmp = np.tensordot(tmp, W2_diag, axes=([1], [0]))
    # tmp: (left, p1, w2R, p2)
    tmp = np.tensordot(tmp, F_diag, axes=([2], [0]))
    # tmp: (left, p1, p2, right)
    return tmp


def _key_slices(dense_layout, key):
    return tuple(
        dense_layout["sector_slices"][axis][sector]
        for axis, sector in enumerate(key)
    )


def _shape_from_slices(slices):
    return tuple(s.stop - s.start for s in slices)


def _apply_two_site_block_sparse(E, W1, W2, F, theta, dense_layout):
    if not isinstance(theta, NonabelianTensor) or theta.rank != 4:
        raise ValueError("_apply_two_site_block_sparse expects a rank-4 NonabelianTensor.")

    out_data = {}
    dtype = np.result_type(
        E.dtype,
        W1.dtype,
        W2.dtype,
        F.dtype,
        *(np.asarray(block).dtype for block in theta.data.values()),
    )

    out_specs = {}
    for out_key in theta.data:
        out_slices = _key_slices(dense_layout, out_key)
        out_specs[out_key] = out_slices
        out_data[out_key] = np.zeros(_shape_from_slices(out_slices), dtype=dtype)

    for in_key, in_block in theta.data.items():
        in_slices = _key_slices(dense_layout, in_key)
        l_in, p1_in, p2_in, r_in = in_slices
        block_in = np.asarray(in_block)
        for out_key, out_slices in out_specs.items():
            l_out, p1_out, p2_out, r_out = out_slices
            E_slice = np.asarray(E[:, l_out, l_in])
            W1_slice = np.asarray(W1[:, :, p1_out, p1_in])
            W2_slice = np.asarray(W2[:, :, p2_out, p2_in])
            F_slice = np.asarray(F[:, r_out, r_in])
            tmp = _apply_two_site_dense(E_slice, W1_slice, W2_slice, F_slice, block_in)
            out_data[out_key] += tmp

    return NonabelianTensor(
        out_data,
        [leg[:] for leg in theta.qns],
        theta.dirs[:],
        fusion_legs=theta.fusion_legs[:],
        metadata=theta.metadata.copy(),
    )


def _build_packed_local_actions(E, W1, W2, F, two_site_template, layout, dense_layout):
    dim = sum(entry.size for entry in layout)

    def dense_apply_from_vector(vector):
        dense_in = _packed_vector_to_dense(
            vector,
            pack_layout=layout,
            dense_layout=dense_layout,
        )
        dense_out = _apply_two_site_dense(E, W1, W2, F, dense_in)
        return _dense_to_packed_vector(
            dense_out,
            pack_layout=layout,
            dense_layout=dense_layout,
        )

    diag = np.zeros(dim, dtype=float)
    for entry in layout:
        l_out, p1_out, p2_out, r_out = _key_slices(dense_layout, entry.key)
        diag_block = _two_site_diagonal_block(
            np.asarray(E[:, l_out, l_out]),
            np.asarray(W1[:, :, p1_out, p1_out]),
            np.asarray(W2[:, :, p2_out, p2_out]),
            np.asarray(F[:, r_out, r_out]),
        )
        diag[entry.offset:entry.offset + entry.size] = np.real(diag_block).reshape(entry.size)
    return dense_apply_from_vector, diag


def _build_tensor_local_actions(E, W1, W2, F, two_site_template, layout, dense_layout):
    dim = sum(entry.size for entry in layout)

    def tensor_apply(two_site):
        return _apply_two_site_block_sparse(E, W1, W2, F, two_site, dense_layout)

    diag = np.zeros(dim, dtype=float)
    for entry in layout:
        l_out, p1_out, p2_out, r_out = _key_slices(dense_layout, entry.key)
        diag_block = _two_site_diagonal_block(
            np.asarray(E[:, l_out, l_out]),
            np.asarray(W1[:, :, p1_out, p1_out]),
            np.asarray(W2[:, :, p2_out, p2_out]),
            np.asarray(F[:, r_out, r_out]),
        )
        diag[entry.offset:entry.offset + entry.size] = np.real(diag_block).reshape(entry.size)
    return tensor_apply, diag


def _initial_left_env_blocks(site_layout, W):
    env = {}
    for sector, dim in site_layout["sector_dims"][0].items():
        block = np.zeros((_mpo_left_dim(W), dim, dim), dtype=_mpo_dtype(W))
        block[0] = np.eye(dim, dtype=_mpo_dtype(W))
        env[(sector, sector)] = block
    return env


def _initial_right_env_blocks(site_layout, W):
    env = {}
    for sector, dim in site_layout["sector_dims"][2].items():
        block = np.zeros((_mpo_right_dim(W), dim, dim), dtype=_mpo_dtype(W))
        block[0] = np.eye(dim, dtype=_mpo_dtype(W))
        env[(sector, sector)] = block
    return env


def _initial_left_env_blocks_rank_coupled(site_layout, W):
    env = {}
    for sector, dim in site_layout["sector_dims"][0].items():
        entries = []
        for idx, irrep in enumerate(W.left_channel_irreps):
            block = np.zeros((irrep.dim, dim, dim), dtype=_mpo_dtype(W))
            if idx == 0:
                block[0] = np.eye(dim, dtype=_mpo_dtype(W))
            entries.append(block)
        env[(sector, sector)] = tuple(entries)
    return env


def _initial_right_env_blocks_rank_coupled(site_layout, W):
    env = {}
    for sector, dim in site_layout["sector_dims"][2].items():
        entries = []
        for idx, irrep in enumerate(W.right_channel_irreps):
            block = np.zeros((irrep.dim, dim, dim), dtype=_mpo_dtype(W))
            if idx == 0:
                block[0] = np.eye(dim, dtype=_mpo_dtype(W))
            entries.append(block)
        env[(sector, sector)] = tuple(entries)
    return env


def _copy_rank_coupled_env_map(env_map):
    return {
        key: tuple(np.array(block, copy=True) for block in blocks)
        for key, blocks in env_map.items()
    }


def _flatten_rank_coupled_env_map(env_map):
    flat = {}
    for key, blocks in env_map.items():
        flat[key] = np.concatenate([np.asarray(block) for block in blocks], axis=0)
    return flat


def _environment_map_expectation(env_map, *, rank_coupled):
    value = 0.0 + 0.0j
    if rank_coupled:
        for blocks in env_map.values():
            for block in blocks:
                value += np.trace(np.asarray(block).sum(axis=0))
    else:
        for block in env_map.values():
            value += np.trace(np.asarray(block)[0])
    return value


def _contract_from_left_blocks(W, A, E_map, B, phys_slices):
    out = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        for (q_lk, q_pk, q_rk), B_block in B.data.items():
            E_block = E_map.get((q_lb, q_lk))
            if E_block is None:
                continue
            if isinstance(W, (MPO, IrreducibleMPO, RankCoupledMPO)):
                W_slice = W.block(q_pb, q_pk)
            else:
                p_slice_b = phys_slices[q_pb]
                p_slice_k = phys_slices[q_pk]
                W_slice = np.asarray(W[:, :, p_slice_b, p_slice_k])
            if W_slice is None:
                continue
            contrib = np.einsum(
                "xij,ipr,xypq,jqs->yrs",
                np.asarray(E_block),
                np.asarray(A_block).conj(),
                np.asarray(W_slice),
                np.asarray(B_block),
                optimize=True,
            )
            key = (q_rb, q_rk)
            if key in out:
                out[key] = out[key] + contrib
            else:
                out[key] = contrib
    return out


def _contract_from_left_blocks_rank_coupled(W, A, E_map, B):
    out = {}
    mpo_dtype = _mpo_dtype(W)
    a_blocks_by_left = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        arr = np.asarray(A_block)
        a_blocks_by_left.setdefault(q_lb, []).append(
            (q_rb, q_pb, arr.conj(), int(arr.shape[2]), arr.dtype)
        )
    b_blocks_by_left = {}
    for (q_lk, q_pk, q_rk), B_block in B.data.items():
        arr = np.asarray(B_block)
        b_blocks_by_left.setdefault(q_lk, []).append(
            (q_rk, q_pk, arr, int(arr.shape[2]), arr.dtype)
        )
    reduced_cache = {}
    for (q_lb, q_lk), E_blocks in E_map.items():
        a_entries = a_blocks_by_left.get(q_lb)
        b_entries = b_blocks_by_left.get(q_lk)
        if not a_entries or not b_entries:
            continue
        e_arrays = tuple(np.asarray(block) for block in E_blocks)
        e_dtypes = tuple(block.dtype for block in e_arrays)
        for q_rb, q_pb, A_conj, bra_dim, a_dtype in a_entries:
            for q_rk, q_pk, B_arr, ket_dim, b_dtype in b_entries:
                reduced_key = (q_pb, q_pk)
                reduced = reduced_cache.get(reduced_key)
                if reduced is None:
                    reduced = W.reduced_block(q_pb, q_pk)
                    reduced_cache[reduced_key] = reduced
                if not reduced:
                    continue
                key = (q_rb, q_rk)
                target = out.get(key)
                if target is None:
                    dtype = np.result_type(mpo_dtype, a_dtype, b_dtype, *e_dtypes)
                    target = [
                        np.zeros((irrep.dim, bra_dim, ket_dim), dtype=dtype)
                        for irrep in W.right_channel_irreps
                    ]
                    out[key] = target
                for (left_idx, right_idx), w_block in reduced.items():
                    target[right_idx] += np.einsum(
                        "xij,ipr,xypq,jqs->yrs",
                        e_arrays[left_idx],
                        A_conj,
                        np.asarray(w_block),
                        B_arr,
                        optimize=_LEFT_BLOCK_PATH,
                    )
    return {key: tuple(blocks) for key, blocks in out.items()}


def _contract_from_right_blocks(W, A, F_map, B, phys_slices):
    out = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        for (q_lk, q_pk, q_rk), B_block in B.data.items():
            F_block = F_map.get((q_rb, q_rk))
            if F_block is None:
                continue
            if isinstance(W, (MPO, IrreducibleMPO, RankCoupledMPO)):
                W_slice = W.block(q_pb, q_pk)
            else:
                p_slice_b = phys_slices[q_pb]
                p_slice_k = phys_slices[q_pk]
                W_slice = np.asarray(W[:, :, p_slice_b, p_slice_k])
            if W_slice is None:
                continue
            contrib = np.einsum(
                "ipr,xypq,yrs,jqs->xij",
                np.asarray(A_block).conj(),
                np.asarray(W_slice),
                np.asarray(F_block),
                np.asarray(B_block),
                optimize=True,
            )
            key = (q_lb, q_lk)
            if key in out:
                out[key] = out[key] + contrib
            else:
                out[key] = contrib
    return out


def _contract_from_right_blocks_rank_coupled(W, A, F_map, B):
    out = {}
    mpo_dtype = _mpo_dtype(W)
    a_blocks_by_right = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        arr = np.asarray(A_block)
        a_blocks_by_right.setdefault(q_rb, []).append(
            (q_lb, q_pb, arr.conj(), int(arr.shape[0]), arr.dtype)
        )
    b_blocks_by_right = {}
    for (q_lk, q_pk, q_rk), B_block in B.data.items():
        arr = np.asarray(B_block)
        b_blocks_by_right.setdefault(q_rk, []).append(
            (q_lk, q_pk, arr, int(arr.shape[0]), arr.dtype)
        )
    reduced_cache = {}
    for (q_rb, q_rk), F_blocks in F_map.items():
        a_entries = a_blocks_by_right.get(q_rb)
        b_entries = b_blocks_by_right.get(q_rk)
        if not a_entries or not b_entries:
            continue
        f_arrays = tuple(np.asarray(block) for block in F_blocks)
        f_dtypes = tuple(block.dtype for block in f_arrays)
        for q_lb, q_pb, A_conj, bra_dim, a_dtype in a_entries:
            for q_lk, q_pk, B_arr, ket_dim, b_dtype in b_entries:
                reduced_key = (q_pb, q_pk)
                reduced = reduced_cache.get(reduced_key)
                if reduced is None:
                    reduced = W.reduced_block(q_pb, q_pk)
                    reduced_cache[reduced_key] = reduced
                if not reduced:
                    continue
                key = (q_lb, q_lk)
                target = out.get(key)
                if target is None:
                    dtype = np.result_type(mpo_dtype, a_dtype, b_dtype, *f_dtypes)
                    target = [
                        np.zeros((irrep.dim, bra_dim, ket_dim), dtype=dtype)
                        for irrep in W.left_channel_irreps
                    ]
                    out[key] = target
                for (left_idx, right_idx), w_block in reduced.items():
                    target[left_idx] += np.einsum(
                        "ipr,xypq,yrs,jqs->xij",
                        A_conj,
                        np.asarray(w_block),
                        f_arrays[right_idx],
                        B_arr,
                        optimize=_RIGHT_BLOCK_PATH,
                    )
    return {key: tuple(blocks) for key, blocks in out.items()}


def _precompute_two_site_block_env_transitions(
    E_map,
    W1,
    W2,
    F_map,
    layout,
    phys1_slices,
    phys2_slices,
):
    transitions = {}
    kernel_cache = {}
    out_entries = tuple((entry.key, entry.shape) for entry in layout)
    out_index = {entry.key: idx for idx, entry in enumerate(layout)}
    left_blocks_by_ket = {}
    right_blocks_by_ket = {}
    for (q_out, q_in), block in E_map.items():
        left_blocks_by_ket.setdefault(q_in, []).append((q_out, block))
    for (q_out, q_in), block in F_map.items():
        right_blocks_by_ket.setdefault(q_in, []).append((q_out, block))

    w1_blocks_by_in = _group_mpo_blocks_by_input(W1, phys1_slices)
    w2_blocks_by_in = _group_mpo_blocks_by_input(W2, phys2_slices)

    for in_entry in layout:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_transitions = []
        for q_lb, E_block in left_blocks_by_ket.get(q_lk, ()):
            for q_p1b, W1_slice in w1_blocks_by_in.get(q_p1k, ()):
                for q_rb, F_block in right_blocks_by_ket.get(q_rk, ()):
                    for q_p2b, W2_slice in w2_blocks_by_in.get(q_p2k, ()):
                        out_key = (q_lb, q_p1b, q_p2b, q_rb)
                        out_idx = out_index.get(out_key)
                        if out_idx is None:
                            continue
                        out_entry = layout[out_idx]
                        kernel_key = (
                            id(E_block),
                            id(W1_slice),
                            id(W2_slice),
                            id(F_block),
                            tuple(int(x) for x in in_entry.shape),
                            tuple(int(x) for x in out_entry.shape),
                        )
                        kernel = kernel_cache.get(kernel_key)
                        if kernel is None:
                            kernel = np.einsum(
                                "xal,xypo,yzqr,zbc->apqblorc",
                                np.asarray(E_block),
                                np.asarray(W1_slice),
                                np.asarray(W2_slice),
                                np.asarray(F_block),
                                optimize=_TWO_SITE_KERNEL_PATH,
                            ).reshape(
                                int(np.prod(out_entry.shape, dtype=int)),
                                int(np.prod(in_entry.shape, dtype=int)),
                            )
                            kernel_cache[kernel_key] = kernel
                        in_transitions.append((out_idx, kernel))
        transitions[in_entry.key] = tuple(in_transitions)
    return out_entries, transitions


def _group_mpo_blocks_by_input(W, phys_slices):
    grouped = {}
    if isinstance(W, (MPO, IrreducibleMPO, RankCoupledMPO)):
        phys_in_sectors = getattr(W, "phys_in_sectors", None)
        phys_out_sectors = getattr(W, "phys_out_sectors", None)
        if phys_in_sectors is None:
            phys_in_sectors = W.phys_in_leg.sectors
        if phys_out_sectors is None:
            phys_out_sectors = W.phys_out_leg.sectors
        for q_in in phys_in_sectors:
            entries = []
            for q_out in phys_out_sectors:
                block = W.block(q_out, q_in)
                if block is not None:
                    entries.append((q_out, np.asarray(block)))
            grouped[q_in] = tuple(entries)
        return grouped
    for q_in, p_in in phys_slices.items():
        entries = []
        for q_out, p_out in phys_slices.items():
            block = np.asarray(W[:, :, p_out, p_in])
            if np.any(block != 0):
                entries.append((q_out, block))
        grouped[q_in] = tuple(entries)
    return grouped


def _precompute_two_site_block_env_factorized_terms(
    E_map,
    W1,
    W2,
    F_map,
    layout,
    phys1_slices,
    phys2_slices,
):
    out_entries = tuple((entry.key, entry.shape) for entry in layout)
    out_index = {entry.key: idx for idx, entry in enumerate(layout)}
    left_blocks_by_ket = {}
    right_blocks_by_ket = {}
    left_factor_cache = {}
    right_factor_cache = {}
    terms = {}
    for (q_out, q_in), block in E_map.items():
        left_blocks_by_ket.setdefault(q_in, []).append((q_out, np.asarray(block)))
    for (q_out, q_in), block in F_map.items():
        right_blocks_by_ket.setdefault(q_in, []).append((q_out, np.asarray(block)))

    w1_blocks_by_in = _group_mpo_blocks_by_input(W1, phys1_slices)
    w2_blocks_by_in = _group_mpo_blocks_by_input(W2, phys2_slices)

    for in_entry in layout:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_terms = []
        for q_lb, E_block in left_blocks_by_ket.get(q_lk, ()):
            for q_p1b, W1_slice in w1_blocks_by_in.get(q_p1k, ()):
                left_key = (id(E_block), id(W1_slice))
                left_factor = left_factor_cache.get(left_key)
                if left_factor is None:
                    left_factor = np.asarray(_factorize_left_two_site_dense_term(E_block, W1_slice))
                    left_factor_cache[left_key] = left_factor
                for q_rb, F_block in right_blocks_by_ket.get(q_rk, ()):
                    for q_p2b, W2_slice in w2_blocks_by_in.get(q_p2k, ()):
                        out_key = (q_lb, q_p1b, q_p2b, q_rb)
                        out_idx = out_index.get(out_key)
                        if out_idx is None:
                            continue
                        right_key = (id(W2_slice), id(F_block))
                        right_factor = right_factor_cache.get(right_key)
                        if right_factor is None:
                            right_factor = np.asarray(
                                _factorize_right_two_site_dense_term(W2_slice, F_block)
                            )
                            right_factor_cache[right_key] = right_factor
                        in_terms.append((out_idx, left_factor, right_factor))
        terms[in_entry.key] = tuple(in_terms)
    return out_entries, terms


def _apply_two_site_block_env(transitions, theta, out_entries, *, base_dtype):
    if not isinstance(theta, NonabelianTensor) or theta.rank != 4:
        raise ValueError("_apply_two_site_block_env expects a rank-4 NonabelianTensor.")

    dtype = np.result_type(
        base_dtype,
        *(np.asarray(block).dtype for block in theta.data.values()),
    )
    out_blocks = [
        np.zeros(shape, dtype=dtype)
        for _key, shape in out_entries
    ]

    for in_key, in_block in theta.data.items():
        block_in = np.asarray(in_block)
        vec_in = block_in.reshape(-1)
        for out_idx, kernel in transitions.get(in_key, ()):
            _out_key, out_shape = out_entries[out_idx]
            contrib = (np.asarray(kernel) @ vec_in).reshape(out_shape)
            out_blocks[out_idx] += contrib

    out_data = {
        key: block
        for (key, _shape), block in zip(out_entries, out_blocks)
    }

    return NonabelianTensor(
        out_data,
        [leg[:] for leg in theta.qns],
        theta.dirs[:],
        fusion_legs=theta.fusion_legs[:],
        metadata=theta.metadata.copy(),
    )


def _apply_two_site_block_env_reduced(transitions, state, out_entries, *, base_dtype):
    if not isinstance(state, ReducedStateVector):
        raise TypeError("_apply_two_site_block_env_reduced expects a ReducedStateVector.")

    dtype = np.result_type(
        base_dtype,
        *(np.asarray(block).dtype for block in state.blocks.values()),
    )
    out_blocks = {
        key: np.zeros(shape, dtype=dtype)
        for key, shape in out_entries
    }

    for in_key, in_block in state.blocks.items():
        block_in = np.asarray(in_block)
        vec_in = block_in.reshape(-1)
        for out_idx, kernel in transitions.get(in_key, ()):
            out_key, _out_shape = out_entries[out_idx]
            contrib = (np.asarray(kernel) @ vec_in).reshape(_out_shape)
            out_blocks[out_key] += contrib

    return ReducedStateVector(layout=state.layout, blocks=out_blocks)


def _apply_two_site_block_env_reduced_compiled(compiled_transitions, state, out_entries, *, base_dtype):
    if not isinstance(state, ReducedStateVector):
        raise TypeError("_apply_two_site_block_env_reduced_compiled expects a ReducedStateVector.")

    dtype = np.result_type(
        base_dtype,
        *(np.asarray(block).dtype for block in state.blocks.values()),
    )
    out_blocks = [None] * len(out_entries)
    items = compiled_transitions["items"]

    for idx, entry in enumerate(state.layout.entries):
        in_block = state.blocks.get(entry.key)
        if in_block is None:
            continue
        compiled = items[idx]
        if compiled is None:
            continue
        _in_offset, in_size, kernel, segments = compiled
        vec_in = np.asarray(in_block).reshape(in_size)
        contrib = kernel @ vec_in
        cursor = 0
        for offset, size in segments:
            remaining = size
            local_offset = int(offset)
            while remaining > 0:
                out_idx = compiled_transitions["offset_to_out_idx"][local_offset]
                out_key, out_shape = out_entries[out_idx]
                out_entry_size = int(np.prod(out_shape, dtype=int))
                piece = contrib[cursor:cursor + out_entry_size].reshape(out_shape)
                existing = out_blocks[out_idx]
                out_blocks[out_idx] = piece if existing is None else existing + piece
                cursor += out_entry_size
                local_offset += out_entry_size
                remaining -= out_entry_size

    blocks = {}
    for (key, _shape), block in zip(out_entries, out_blocks):
        if block is None:
            continue
        if np.linalg.norm(np.asarray(block).reshape(-1)) > 1.0e-15:
            blocks[key] = block
    return ReducedStateVector(layout=state.layout, blocks=blocks)


def _apply_two_site_block_env_packed(transitions, vector, layout, *, base_dtype):
    vec = np.asarray(vector)
    dtype = np.result_type(base_dtype, vec.dtype)
    out = np.zeros(sum(entry.size for entry in layout), dtype=dtype)

    for in_entry in layout:
        vec_in = vec[in_entry.offset:in_entry.offset + in_entry.size]
        if vec_in.size == 0:
            continue
        for out_idx, kernel in transitions.get(in_entry.key, ()):
            out_entry = layout[out_idx]
            out[out_entry.offset:out_entry.offset + out_entry.size] += np.asarray(kernel) @ vec_in
    return out


def _compile_packed_transitions(transitions, layout):
    compiled_items = []
    total_dim = sum(entry.size for entry in layout)
    offset_to_out_idx = {}
    diagonal_blocks = []
    for out_idx, entry in enumerate(layout):
        offset_to_out_idx[int(entry.offset)] = out_idx
    for in_idx, in_entry in enumerate(layout):
        terms = transitions.get(in_entry.key, ())
        if not terms:
            compiled_items.append(None)
            diagonal_blocks.append(None)
            continue
        grouped = {}
        for out_idx, kernel in terms:
            arr = np.asarray(kernel)
            if out_idx in grouped:
                grouped[out_idx] = grouped[out_idx] + arr
            else:
                grouped[out_idx] = arr
        ordered = sorted(grouped.items(), key=lambda item: layout[item[0]].offset)
        if not ordered:
            compiled_items.append(None)
            diagonal_blocks.append(None)
            continue
        self_kernel = grouped.get(in_idx)
        if self_kernel is not None:
            diagonal_blocks.append(np.ascontiguousarray(self_kernel))
        else:
            diagonal_blocks.append(None)
        kernels = []
        segments = []
        current_offset = None
        current_size = 0
        for out_idx, kernel in ordered:
            out_entry = layout[out_idx]
            kernels.append(np.asarray(kernel))
            if current_offset is None:
                current_offset = int(out_entry.offset)
                current_size = int(out_entry.size)
            elif current_offset + current_size == int(out_entry.offset):
                current_size += int(out_entry.size)
            else:
                segments.append((current_offset, current_size))
                current_offset = int(out_entry.offset)
                current_size = int(out_entry.size)
        if current_offset is not None:
            segments.append((current_offset, current_size))
        kernel_matrix = np.ascontiguousarray(np.vstack(kernels))
        compiled_items.append(
            (
                int(in_entry.offset),
                int(in_entry.size),
                kernel_matrix,
                tuple(segments),
            )
        )
    return {
        "items": tuple(compiled_items),
        "total_dim": int(total_dim),
        "offset_to_out_idx": offset_to_out_idx,
        "diagonal_blocks": tuple(diagonal_blocks),
    }


def _materialize_packed_matrix_from_compiled(compiled_transitions, *, dtype=None):
    total_dim = int(compiled_transitions["total_dim"])
    present = [
        item[2].dtype
        for item in compiled_transitions["items"]
        if item is not None
    ]
    if dtype is None:
        dtype = np.result_type(*(present or [float]))
    matrix = np.zeros((total_dim, total_dim), dtype=dtype)
    for item in compiled_transitions["items"]:
        if item is None:
            continue
        in_offset, in_size, kernel, segments = item
        cursor = 0
        for offset, size in segments:
            matrix[offset:offset + size, in_offset:in_offset + in_size] = kernel[
                cursor:cursor + size
            ]
            cursor += size
    return matrix


def _materialize_packed_csr_from_compiled(compiled_transitions, *, dtype=None):
    if sp is None:
        return None
    total_dim = int(compiled_transitions["total_dim"])
    data = []
    rows = []
    cols = []
    present = []
    for item in compiled_transitions["items"]:
        if item is not None:
            present.append(item[2].dtype)
    if dtype is None:
        dtype = np.result_type(*(present or [float]))

    for item in compiled_transitions["items"]:
        if item is None:
            continue
        in_offset, in_size, kernel, segments = item
        cursor = 0
        for offset, size in segments:
            block = np.asarray(kernel[cursor:cursor + size, :], dtype=dtype)
            nz_rows, nz_cols = np.nonzero(np.abs(block) > 0.0)
            if nz_rows.size:
                rows.extend((nz_rows + offset).tolist())
                cols.extend((nz_cols + in_offset).tolist())
                data.extend(block[nz_rows, nz_cols].tolist())
            cursor += size
    return sp.csr_matrix((data, (rows, cols)), shape=(total_dim, total_dim), dtype=dtype)


def _apply_two_site_block_env_packed_compiled(compiled_transitions, vector, *, base_dtype):
    vec = np.asarray(vector)
    dtype = np.result_type(base_dtype, vec.dtype)
    out = np.zeros(int(compiled_transitions["total_dim"]), dtype=dtype)

    for compiled in compiled_transitions["items"]:
        if compiled is None:
            continue
        in_offset, in_size, kernel, segments = compiled
        vec_in = vec[in_offset:in_offset + in_size]
        if vec_in.size == 0:
            continue
        contrib = kernel @ vec_in
        if len(segments) == 1:
            offset, size = segments[0]
            out[offset:offset + size] += contrib
            continue
        cursor = 0
        for offset, size in segments:
            out[offset:offset + size] += contrib[cursor:cursor + size]
            cursor += size
    return out


def _apply_two_site_block_env_packed_factorized(factorized_terms, vector, layout, *, base_dtype):
    vec = np.asarray(vector)
    dtype = np.result_type(base_dtype, vec.dtype)
    out = np.zeros(sum(entry.size for entry in layout), dtype=dtype)

    for in_entry in layout:
        terms = factorized_terms.get(in_entry.key)
        if not terms:
            continue
        vec_in = vec[in_entry.offset:in_entry.offset + in_entry.size]
        if vec_in.size == 0:
            continue
        block_in = vec_in.reshape(in_entry.shape)
        for offset, size, _out_shape, left_stack, right_stack in terms:
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                np.asarray(left_stack),
                block_in,
                np.asarray(right_stack),
                optimize=_FACTORIZED_TWO_SITE_BATCH_PATH,
            )
            out[offset:offset + size] += np.asarray(contrib).reshape(size)
    return out


def _compile_factorized_terms(factorized_terms, layout):
    compiled = {}
    for in_entry in layout:
        terms = factorized_terms.get(in_entry.key, ())
        if not terms:
            continue
        grouped = {}
        for out_idx, left_factor, right_factor in terms:
            bucket = grouped.setdefault(out_idx, {"left": [], "right": []})
            bucket["left"].append(np.asarray(left_factor))
            bucket["right"].append(np.asarray(right_factor))
        compiled_terms = []
        for out_idx in sorted(grouped, key=lambda idx: layout[idx].offset):
            out_entry = layout[out_idx]
            bucket = grouped[out_idx]
            compiled_terms.append(
                (
                    out_entry.offset,
                    out_entry.size,
                    out_entry.shape,
                    np.stack(bucket["left"], axis=0),
                    np.stack(bucket["right"], axis=0),
                )
            )
        compiled[in_entry.key] = tuple(compiled_terms)
    return compiled


def _transitions_are_identity_operator(layout, transitions, *, tol=1e-12):
    for entry in layout:
        terms = transitions.get(entry.key, ())
        diag_found = False
        for out_idx, kernel in terms:
            out_entry = layout[out_idx]
            arr = np.asarray(kernel)
            if out_entry.key == entry.key and out_entry.shape == entry.shape:
                eye = np.eye(entry.size, dtype=arr.dtype)
                if np.allclose(arr, eye, atol=tol, rtol=tol):
                    if diag_found:
                        return False
                    diag_found = True
                elif np.linalg.norm(arr) > tol:
                    return False
            elif np.linalg.norm(arr) > tol:
                return False
        if not diag_found:
            return False
    return True


def _build_identity_mpo_local_actions(E_map, F_map, layout, *, base_dtype):
    def _env_to_matrix(block):
        arr = np.asarray(block, dtype=base_dtype)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        raise ValueError(
            "Identity-MPO local actions expect rank-2 environment blocks or "
            f"rank-3 blocks with leading dimension 1, got {arr.shape!r}."
        )

    transitions = {}
    out_entries = tuple((entry.key, entry.shape) for entry in layout)
    out_index = {entry.key: idx for idx, entry in enumerate(layout)}
    kernel_cache = {}

    for in_entry in layout:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_transitions = []
        eye_p1 = np.eye(int(in_entry.shape[1]), dtype=base_dtype)
        eye_p2 = np.eye(int(in_entry.shape[2]), dtype=base_dtype)
        for q_lb, q_lk_again in E_map:
            if q_lk_again != q_lk:
                continue
            E_block = _env_to_matrix(E_map[(q_lb, q_lk)])
            for q_rb, q_rk_again in F_map:
                if q_rk_again != q_rk:
                    continue
                out_key = (q_lb, q_p1k, q_p2k, q_rb)
                out_idx = out_index.get(out_key)
                if out_idx is None:
                    continue
                F_block = _env_to_matrix(F_map[(q_rb, q_rk)])
                out_entry = layout[out_idx]
                kernel_key = (
                    id(E_block),
                    id(F_block),
                    tuple(int(dim) for dim in in_entry.shape),
                    tuple(int(dim) for dim in out_entry.shape),
                )
                kernel = kernel_cache.get(kernel_key)
                if kernel is None:
                    kernel = np.einsum(
                        "al,po,qr,bc->apqblorc",
                        E_block,
                        eye_p1,
                        eye_p2,
                        F_block,
                        optimize=_IDENTITY_TWO_SITE_KERNEL_PATH,
                    ).reshape(
                        int(np.prod(out_entry.shape, dtype=int)),
                        int(np.prod(in_entry.shape, dtype=int)),
                    )
                    kernel_cache[kernel_key] = kernel
                in_transitions.append((out_idx, kernel))
        transitions[in_entry.key] = tuple(in_transitions)
    compiled_transitions = _compile_packed_transitions(transitions, layout)

    def tensor_apply(two_site):
        return _apply_two_site_block_env(
            transitions,
            two_site,
            out_entries,
            base_dtype=base_dtype,
        )

    def reduced_apply(state):
        return _apply_two_site_block_env_reduced(
            transitions,
            state,
            out_entries,
            base_dtype=base_dtype,
        )

    if compiled_transitions["total_dim"] <= _PACKED_DENSE_LOCAL_DIM:
        packed_matrix = _materialize_packed_matrix_from_compiled(
            compiled_transitions,
            dtype=base_dtype,
        )

        def packed_apply(vector):
            return packed_matrix @ np.asarray(vector)

        packed_apply.backend = "compiled-dense"
        packed_apply.matrix = packed_matrix
        packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]
    elif sp is not None and compiled_transitions["total_dim"] >= _PACKED_CSR_LOCAL_DIM:
        packed_csr = _materialize_packed_csr_from_compiled(
            compiled_transitions,
            dtype=base_dtype,
        )

        def packed_apply(vector):
            return np.asarray(packed_csr @ np.asarray(vector))

        packed_apply.backend = "compiled-csr"
        packed_apply.matrix = packed_csr
        packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]
    else:
        def packed_apply(vector):
            return _apply_two_site_block_env_packed_compiled(
                compiled_transitions,
                vector,
                base_dtype=base_dtype,
            )

        packed_apply.backend = "compiled"
        packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]

    diag = np.zeros(sum(entry.size for entry in layout), dtype=float)
    for entry in layout:
        q_l, _q_p1, _q_p2, q_r = entry.key
        E_block = E_map.get((q_l, q_l))
        F_block = F_map.get((q_r, q_r))
        if E_block is None or F_block is None:
            continue
        diag_left = np.real(np.diag(_env_to_matrix(E_block)))
        diag_right = np.real(np.diag(_env_to_matrix(F_block)))
        diag_block = np.einsum(
            "l,p,q,r->lpqr",
            diag_left,
            np.ones(int(entry.shape[1]), dtype=float),
            np.ones(int(entry.shape[2]), dtype=float),
            diag_right,
            optimize=True,
        )
        diag[entry.offset:entry.offset + entry.size] = diag_block.reshape(entry.size)

    identity_like = _transitions_are_identity_operator(layout, transitions)
    return tensor_apply, reduced_apply, packed_apply, diag, identity_like


def _build_block_sparse_local_actions(
    E_map,
    W1,
    W2,
    F_map,
    two_site_template,
    layout,
    phys1_slices,
    phys2_slices,
):
    dim = sum(entry.size for entry in layout)
    out_dtype = np.result_type(
        _mpo_dtype(W1),
        _mpo_dtype(W2),
        *(np.asarray(block).dtype for block in two_site_template.data.values()),
    )
    if _is_identity_mpo_core(W1) and _is_identity_mpo_core(W2):
        return _build_identity_mpo_local_actions(
            E_map,
            F_map,
            layout,
            base_dtype=out_dtype,
        )
    if dim > _FACTORIZED_PACKED_LOCAL_DIM:
        transition_cache = {}

        def _lazy_transitions():
            cached = transition_cache.get("value")
            if cached is None:
                cached = _precompute_two_site_block_env_transitions(
                    E_map,
                    W1,
                    W2,
                    F_map,
                    layout,
                    phys1_slices,
                    phys2_slices,
                )
                transition_cache["value"] = cached
            return cached

        def tensor_apply(two_site):
            out_entries, transitions = _lazy_transitions()
            return _apply_two_site_block_env(
                transitions,
                two_site,
                out_entries,
                base_dtype=out_dtype,
            )

        def reduced_apply(state):
            out_entries, transitions = _lazy_transitions()
            return _apply_two_site_block_env_reduced(
                transitions,
                state,
                out_entries,
                base_dtype=out_dtype,
            )

        _packed_out_entries, factorized_terms = _precompute_two_site_block_env_factorized_terms(
            E_map,
            W1,
            W2,
            F_map,
            layout,
            phys1_slices,
            phys2_slices,
        )
        compiled_factorized_terms = _compile_factorized_terms(factorized_terms, layout)

        def packed_apply(vector):
            return _apply_two_site_block_env_packed_factorized(
                compiled_factorized_terms,
                vector,
                layout,
                base_dtype=out_dtype,
            )

        packed_apply.backend = "factorized-batched"
        packed_apply.out_entries = _packed_out_entries
        identity_like = False
    else:
        out_entries, transitions = _precompute_two_site_block_env_transitions(
            E_map,
            W1,
            W2,
            F_map,
            layout,
            phys1_slices,
            phys2_slices,
        )

        def tensor_apply(two_site):
            return _apply_two_site_block_env(
                transitions,
                two_site,
                out_entries,
                base_dtype=out_dtype,
            )

        def reduced_apply(state):
            return _apply_two_site_block_env_reduced(
                transitions,
                state,
                out_entries,
                base_dtype=out_dtype,
            )

        compiled_transitions = _compile_packed_transitions(transitions, layout)
        if compiled_transitions["total_dim"] <= _PACKED_DENSE_LOCAL_DIM:
            packed_matrix = _materialize_packed_matrix_from_compiled(
                compiled_transitions,
                dtype=out_dtype,
            )

            def packed_apply(vector):
                return packed_matrix @ np.asarray(vector)

            packed_apply.backend = "compiled-dense"
            packed_apply.matrix = packed_matrix
            packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]
        elif sp is not None and compiled_transitions["total_dim"] >= _PACKED_CSR_LOCAL_DIM:
            packed_csr = _materialize_packed_csr_from_compiled(
                compiled_transitions,
                dtype=out_dtype,
            )

            def packed_apply(vector):
                return np.asarray(packed_csr @ np.asarray(vector))

            packed_apply.backend = "compiled-csr"
            packed_apply.matrix = packed_csr
            packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]
        else:
            def packed_apply(vector):
                return _apply_two_site_block_env_packed_compiled(
                    compiled_transitions,
                    vector,
                    base_dtype=out_dtype,
                )

            packed_apply.backend = "compiled"
            packed_apply.block_matrices = compiled_transitions["diagonal_blocks"]
        identity_like = _transitions_are_identity_operator(layout, transitions)

    diag = np.zeros(dim, dtype=float)
    for entry in layout:
        q_l, q_p1, q_p2, q_r = entry.key
        E_block = E_map.get((q_l, q_l))
        F_block = F_map.get((q_r, q_r))
        if E_block is None or F_block is None:
            continue
        if isinstance(W1, (MPO, IrreducibleMPO, RankCoupledMPO)):
            W1_slice = W1.block(q_p1, q_p1)
        else:
            p1 = phys1_slices.get(q_p1)
            W1_slice = None if p1 is None else np.asarray(W1[:, :, p1, p1])
        if isinstance(W2, (MPO, IrreducibleMPO, RankCoupledMPO)):
            W2_slice = W2.block(q_p2, q_p2)
        else:
            p2 = phys2_slices.get(q_p2)
            W2_slice = None if p2 is None else np.asarray(W2[:, :, p2, p2])
        if W1_slice is None or W2_slice is None:
            continue
        diag_block = _two_site_diagonal_block(E_block, W1_slice, W2_slice, F_block)
        diag[entry.offset:entry.offset + entry.size] = np.real(diag_block).reshape(entry.size)
    return tensor_apply, reduced_apply, packed_apply, diag, identity_like


@dataclass
class DenseEnvironmentChain:
    """
    Dense left/right environments for a fixed chain of site tensors and MPO cores.
    """

    sites: list
    mpo_factors: list
    site_layouts: list
    left_envs: list
    right_envs: list

    @classmethod
    def build(cls, sites, mpo_factors):
        if len(sites) != len(mpo_factors):
            raise ValueError("DenseEnvironmentChain requires one MPO core per site tensor.")
        if len(sites) < 2:
            raise ValueError("DenseEnvironmentChain requires at least two sites.")
        sites = [normalize_site_tensor_layout(site) for site in sites]

        site_layouts = [_tensor_dense_layout(site) for site in sites]
        dense_mpo_factors = _normalize_dense_mpo_factors(
            mpo_factors,
            site_layouts=site_layouts,
        )
        dense_sites = [
            _site_to_dense(
                site,
                dense_layout=_tensor_dense_layout(
                    site,
                    axis_overrides={1: _site_physical_override(layout, site)},
                ),
            )
            for site, layout in zip(sites, site_layouts)
        ]
        left_envs = [_initial_left_env(dense_mpo_factors[0])]
        for i in range(len(sites) - 1):
            left_envs.append(
                _contract_from_left_dense(
                    dense_mpo_factors[i],
                    dense_sites[i],
                    left_envs[-1],
                    dense_sites[i],
                )
            )

        right_envs = [_initial_right_env(dense_mpo_factors[-1])]
        for i in range(len(sites) - 1, 0, -1):
            right_envs.append(
                _contract_from_right_dense(
                    dense_mpo_factors[i],
                    dense_sites[i],
                    right_envs[-1],
                    dense_sites[i],
                )
            )
        right_envs = list(reversed(right_envs))
        return cls(list(sites), list(dense_mpo_factors), site_layouts, left_envs, right_envs)

    def _bond_operator_from_envs(self, bond, two_site_template, *, left_env, right_env):
        if bond < 0 or bond >= len(self.sites) - 1:
            raise IndexError(f"Bond {bond} out of range for chain length {len(self.sites)}.")
        E = np.asarray(left_env)
        F = np.asarray(right_env)
        W1 = np.asarray(self.mpo_factors[bond])
        W2 = np.asarray(self.mpo_factors[bond + 1])
        _, layout = pack_two_site_state(two_site_template)
        dense_layout = _tensor_dense_layout(
            two_site_template,
            axis_overrides={
                1: _site_physical_override(self.site_layouts[bond], self.sites[bond]),
                2: _site_physical_override(self.site_layouts[bond + 1], self.sites[bond + 1]),
            },
        )
        tensor_matvec, diag = _build_tensor_local_actions(
            E, W1, W2, F, two_site_template, layout, dense_layout
        )
        return LocalOperator(
            tensor_matvec=tensor_matvec,
            diag=diag,
            name=f"bond-{bond}-effective-H",
        )

    def bond_operator(self, bond, two_site_template):
        return self._bond_operator_from_envs(
            bond,
            two_site_template,
            left_env=self.left_envs[bond],
            right_env=self.right_envs[bond + 1],
        )

    def start_sweep(self, direction="lr"):
        return DenseEnvironmentSweep.from_chain(self, direction=direction)


@dataclass
class DenseEnvironmentSweep:
    """
    Incremental dense environment state for one sweep direction.
    """

    chain: DenseEnvironmentChain
    direction: str
    current_env: np.ndarray

    @classmethod
    def from_chain(cls, chain, direction="lr"):
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError(f"Unknown sweep direction {direction!r}.")
        if direction == "lr":
            current_env = chain.left_envs[0]
        else:
            current_env = chain.right_envs[-1]
        return cls(chain=chain, direction=direction, current_env=np.asarray(current_env))

    def bond_operator(self, bond, two_site_template):
        if self.direction == "lr":
            return self.chain._bond_operator_from_envs(
                bond,
                two_site_template,
                left_env=self.current_env,
                right_env=self.chain.right_envs[bond + 1],
            )
        return self.chain._bond_operator_from_envs(
            bond,
            two_site_template,
            left_env=self.chain.left_envs[bond],
            right_env=self.current_env,
        )

    def advance_after_update(self, bond, left_site, right_site):
        if self.direction == "lr":
            dense_left = _site_to_dense(
                left_site,
                dense_layout=_tensor_dense_layout(
                    left_site,
                    axis_overrides={
                        1: _site_physical_override(
                            self.chain.site_layouts[bond],
                            self.chain.sites[bond],
                        )
                    },
                ),
            )
            self.current_env = _contract_from_left_dense(
                np.asarray(self.chain.mpo_factors[bond]),
                dense_left,
                self.current_env,
                dense_left,
            )
        else:
            dense_right = _site_to_dense(
                right_site,
                dense_layout=_tensor_dense_layout(
                    right_site,
                    axis_overrides={
                        1: _site_physical_override(
                            self.chain.site_layouts[bond + 1],
                            self.chain.sites[bond + 1],
                        )
                    },
                ),
            )
            self.current_env = _contract_from_right_dense(
                np.asarray(self.chain.mpo_factors[bond + 1]),
                dense_right,
                self.current_env,
                dense_right,
            )
        return self

    def clone(self):
        return DenseEnvironmentSweep(
            chain=self.chain,
            direction=self.direction,
            current_env=np.array(self.current_env, copy=True),
        )


def build_dense_bond_operator(sites, mpo_factors, bond, two_site_template):
    """
    Convenience helper: build a dense effective two-site local operator for one bond.
    """
    env_chain = DenseEnvironmentChain.build(sites, mpo_factors)
    E = env_chain.left_envs[bond]
    F = env_chain.right_envs[bond + 1]
    W1 = np.asarray(env_chain.mpo_factors[bond])
    W2 = np.asarray(env_chain.mpo_factors[bond + 1])
    _, layout = pack_two_site_state(two_site_template)
    dense_layout = _tensor_dense_layout(
        two_site_template,
        axis_overrides={
            1: _site_physical_override(env_chain.site_layouts[bond], env_chain.sites[bond]),
            2: _site_physical_override(env_chain.site_layouts[bond + 1], env_chain.sites[bond + 1]),
        },
    )
    dense_matvec, diag = _build_packed_local_actions(
        E, W1, W2, F, two_site_template, layout, dense_layout
    )
    return LocalOperator(
        matvec=dense_matvec,
        diag=diag,
        name=f"bond-{bond}-dense-effective-H",
    )


@dataclass
class BlockSparseEnvironmentChain:
    """
    Block-sparse left/right environments keyed by bond-sector pairs.
    """

    sites: list
    mpo_factors: list
    site_layouts: list
    left_envs: list
    right_envs: list
    rank_coupled: bool = False

    @classmethod
    def build(cls, sites, mpo_factors):
        if len(sites) != len(mpo_factors):
            raise ValueError("BlockSparseEnvironmentChain requires one MPO core per site tensor.")
        if len(sites) < 2:
            raise ValueError("BlockSparseEnvironmentChain requires at least two sites.")
        sites = [normalize_site_tensor_layout(site) for site in sites]

        site_layouts = [_tensor_dense_layout(site) for site in sites]
        phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
        sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
            mpo_factors,
            site_layouts=site_layouts,
        )

        rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
        if rank_coupled:
            left_envs = [_initial_left_env_blocks_rank_coupled(site_layouts[0], sparse_mpo_factors[0])]
            for i in range(len(sites) - 1):
                left_envs.append(
                    _contract_from_left_blocks_rank_coupled(
                        sparse_mpo_factors[i],
                        sites[i],
                        left_envs[-1],
                        sites[i],
                    )
                )

            right_envs = [_initial_right_env_blocks_rank_coupled(site_layouts[-1], sparse_mpo_factors[-1])]
            for i in range(len(sites) - 1, 0, -1):
                right_envs.append(
                    _contract_from_right_blocks_rank_coupled(
                        sparse_mpo_factors[i],
                        sites[i],
                        right_envs[-1],
                        sites[i],
                    )
                )
        else:
            left_envs = [_initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0])]
            for i in range(len(sites) - 1):
                left_envs.append(
                    _contract_from_left_blocks(
                        sparse_mpo_factors[i],
                        sites[i],
                        left_envs[-1],
                        sites[i],
                        phys_slice_maps[i],
                    )
                )

            right_envs = [_initial_right_env_blocks(site_layouts[-1], sparse_mpo_factors[-1])]
            for i in range(len(sites) - 1, 0, -1):
                right_envs.append(
                    _contract_from_right_blocks(
                        sparse_mpo_factors[i],
                        sites[i],
                        right_envs[-1],
                        sites[i],
                        phys_slice_maps[i],
                    )
                )
        right_envs = list(reversed(right_envs))
        return cls(list(sites), list(sparse_mpo_factors), site_layouts, left_envs, right_envs, rank_coupled)

    def _bond_operator_from_envs(self, bond, two_site_template, *, left_env, right_env):
        if bond < 0 or bond >= len(self.sites) - 1:
            raise IndexError(f"Bond {bond} out of range for chain length {len(self.sites)}.")
        W1 = self.mpo_factors[bond]
        W2 = self.mpo_factors[bond + 1]
        _, layout = pack_two_site_state(two_site_template)
        phys1_slices = self.site_layouts[bond]["sector_slices"][1]
        phys2_slices = self.site_layouts[bond + 1]["sector_slices"][1]
        if self.rank_coupled:
            left_env = _flatten_rank_coupled_env_map(left_env)
            right_env = _flatten_rank_coupled_env_map(right_env)
        tensor_matvec, reduced_matvec, packed_matvec, diag, identity_like = _build_block_sparse_local_actions(
            left_env,
            W1,
            W2,
            right_env,
            two_site_template,
            layout,
            phys1_slices,
            phys2_slices,
        )
        return LocalOperator(
            tensor_matvec=tensor_matvec,
            aux_reduced_matvec=reduced_matvec,
            aux_packed_matvec=packed_matvec,
            packed_block_matrices=getattr(packed_matvec, "block_matrices", None),
            diag=diag,
            name=f"bond-{bond}-block-sparse-effective-H",
            identity_like=identity_like,
        )

    def bond_operator(self, bond, two_site_template):
        return self._bond_operator_from_envs(
            bond,
            two_site_template,
            left_env=self.left_envs[bond],
            right_env=self.right_envs[bond + 1],
        )

    def start_sweep(self, direction="lr"):
        return BlockSparseEnvironmentSweep.from_chain(self, direction=direction)


@dataclass
class BlockSparseEnvironmentSweep:
    """
    Incremental block-sparse environment state for one sweep direction.
    """

    chain: BlockSparseEnvironmentChain
    direction: str
    current_env: dict

    @classmethod
    def from_chain(cls, chain, direction="lr"):
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError(f"Unknown sweep direction {direction!r}.")
        if direction == "lr":
            current_env = (
                _copy_rank_coupled_env_map(chain.left_envs[0])
                if chain.rank_coupled
                else {key: np.array(block, copy=True) for key, block in chain.left_envs[0].items()}
            )
        else:
            current_env = (
                _copy_rank_coupled_env_map(chain.right_envs[-1])
                if chain.rank_coupled
                else {key: np.array(block, copy=True) for key, block in chain.right_envs[-1].items()}
            )
        return cls(chain=chain, direction=direction, current_env=current_env)

    def bond_operator(self, bond, two_site_template):
        if self.direction == "lr":
            return self.chain._bond_operator_from_envs(
                bond,
                two_site_template,
                left_env=self.current_env,
                right_env=self.chain.right_envs[bond + 1],
            )
        return self.chain._bond_operator_from_envs(
            bond,
            two_site_template,
            left_env=self.chain.left_envs[bond],
            right_env=self.current_env,
        )

    def advance_after_update(self, bond, left_site, right_site):
        if self.direction == "lr":
            self.current_env = (
                _contract_from_left_blocks_rank_coupled(
                    self.chain.mpo_factors[bond],
                    left_site,
                    self.current_env,
                    left_site,
                )
                if self.chain.rank_coupled
                else _contract_from_left_blocks(
                    self.chain.mpo_factors[bond],
                    left_site,
                    self.current_env,
                    left_site,
                    self.chain.site_layouts[bond]["sector_slices"][2],
                )
            )
        else:
            self.current_env = (
                _contract_from_right_blocks_rank_coupled(
                    self.chain.mpo_factors[bond + 1],
                    right_site,
                    self.current_env,
                    right_site,
                )
                if self.chain.rank_coupled
                else _contract_from_right_blocks(
                    self.chain.mpo_factors[bond + 1],
                    right_site,
                    self.current_env,
                    right_site,
                    self.chain.site_layouts[bond + 1]["sector_slices"][2],
                )
            )
        return self

    def final_expectation(self, sites):
        if len(sites) != len(self.chain.sites):
            raise ValueError("final_expectation expects one site tensor per chain site.")
        if self.direction == "lr":
            site_index = len(sites) - 1
            site = sites[site_index]
            if self.chain.rank_coupled:
                env = _contract_from_left_blocks_rank_coupled(
                    self.chain.mpo_factors[site_index],
                    site,
                    self.current_env,
                    site,
                )
            else:
                env = _contract_from_left_blocks(
                    self.chain.mpo_factors[site_index],
                    site,
                    self.current_env,
                    site,
                    self.chain.site_layouts[site_index]["sector_slices"][2],
                )
        else:
            site_index = 0
            site = sites[site_index]
            if self.chain.rank_coupled:
                env = _contract_from_right_blocks_rank_coupled(
                    self.chain.mpo_factors[site_index],
                    site,
                    self.current_env,
                    site,
                )
            else:
                env = _contract_from_right_blocks(
                    self.chain.mpo_factors[site_index],
                    site,
                    self.current_env,
                    site,
                    self.chain.site_layouts[site_index]["sector_slices"][2],
                )
        return _environment_map_expectation(env, rank_coupled=self.chain.rank_coupled)

    def clone(self):
        return BlockSparseEnvironmentSweep(
            chain=self.chain,
            direction=self.direction,
            current_env=(
                _copy_rank_coupled_env_map(self.current_env)
                if self.chain.rank_coupled
                else {key: np.array(block, copy=True) for key, block in self.current_env.items()}
            ),
        )


def build_block_sparse_bond_operator(sites, mpo_factors, bond, two_site_template):
    """
    Convenience helper: build a block-structured effective two-site local operator for one bond.
    """
    env_chain = BlockSparseEnvironmentChain.build(sites, mpo_factors)
    return env_chain.bond_operator(bond, two_site_template)


def contract_chain_expectation(sites, mpo_factors):
    """
    Contract ``<sites|MPO|sites>`` for one non-Abelian MPS chain.

    Parameters
    ----------
    sites
        Sequence of rank-3 non-Abelian MPS site tensors.
    mpo_factors
        Sequence of MPO cores (dense or block-sparse ``MPO`` objects), one per
        site.

    Returns
    -------
    complex
        Scalar expectation value of the MPO on the provided MPS.
    """
    if len(sites) != len(mpo_factors):
        raise ValueError("contract_chain_expectation requires one MPO core per site tensor.")
    if not sites:
        raise ValueError("contract_chain_expectation requires at least one site tensor.")

    site_layouts = [_tensor_dense_layout(site) for site in sites]
    phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
    sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
        mpo_factors,
        site_layouts=site_layouts,
    )

    rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
    if rank_coupled:
        env = _initial_left_env_blocks_rank_coupled(site_layouts[0], sparse_mpo_factors[0])
        for idx in range(len(sites)):
            env = _contract_from_left_blocks_rank_coupled(
                sparse_mpo_factors[idx],
                sites[idx],
                env,
                sites[idx],
            )
    else:
        env = _initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0])
        for idx in range(len(sites)):
            env = _contract_from_left_blocks(
                sparse_mpo_factors[idx],
                sites[idx],
                env,
                sites[idx],
                phys_slice_maps[idx],
            )

    value = 0.0 + 0.0j
    return _environment_map_expectation(env, rank_coupled=rank_coupled)
