#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense environment and effective-H helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace

import numpy as np

from .contraction import normalize_site_tensor_layout
from .basis import TwoSiteBasis
from .local_operator import (
    apply_compiled_transition_reduced,
    apply_compiled_packed_transitions,
    apply_factorized_packed_terms,
    apply_packed_transitions,
    apply_transition_reduced,
    apply_transition_tensor,
    build_identity_mpo_local_actions,
    compile_factorized_terms,
    compile_packed_transitions,
    diagonal_from_factorized_terms,
    identity_env_to_matrix,
    identity_mpo_transitions,
    materialize_packed_csr,
    materialize_packed_matrix,
    transitions_are_identity_operator,
)
from .mpo import MPO, IrreducibleMPO, RankCoupledMPO
from .solver import (
    LocalOperator,
    ReducedStateVector,
    build_orthonormalized_local_problem,
    pack_two_site_state,
    two_site_state_basis,
)
from .renormalized import RenormalizedBlockStack, RenormalizedOperatorStack
from .tensor import NonabelianTensor


def _basis_cache_signature(basis):
    """
    Return a hashable signature for a local two-site basis.

    :param basis: Explicit local basis with packed entries.
    :returns: Tuple describing sector keys and block shapes.
    """

    return tuple((entry.key, entry.shape, entry.size) for entry in basis)


def _array_digest(array):
    if isinstance(array, dict):
        digest = hashlib.blake2b(digest_size=16)
        for key, value in sorted(array.items(), key=lambda item: repr(item[0])):
            digest.update(repr(key).encode("utf8"))
            digest.update(_array_digest(value).encode("utf8"))
        return digest.hexdigest()
    if isinstance(array, (tuple, list)):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(type(array).__name__).encode("utf8"))
        digest.update(np.asarray([len(array)], dtype=np.int64).tobytes())
        for item in array:
            digest.update(_array_digest(item).encode("utf8"))
        return digest.hexdigest()
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("utf8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _environment_cache_signature(env):
    """
    Return a content signature for a block environment.

    :param env: Left or right block environment.
    :returns: Hashable signature over rank-coupled flag, keys, shapes, and
        exact block bytes.
    """

    items = []
    for key, block in sorted(env.items(), key=lambda item: repr(item[0])):
        if isinstance(block, dict):
            subitems = tuple(
                (subkey, _array_digest(subblock))
                for subkey, subblock in sorted(block.items(), key=lambda item: repr(item[0]))
            )
            items.append((key, subitems))
        else:
            items.append((key, _array_digest(block)))
    return (bool(getattr(env, "rank_coupled", False)), tuple(items))


def _renormalized_entry_cache_signature(entry, fallback_env):
    """
    Return a stable cache signature for a renormalized boundary.

    :param entry: Optional persisted renormalized block entry.
    :param fallback_env: Environment block used when no entry signature exists.
    :returns: Hashable boundary signature.
    """

    if entry is not None and getattr(entry, "signature", None) is not None:
        return (
            "entry",
            str(getattr(entry, "namespace", "")),
            str(getattr(entry, "side", "")),
            int(getattr(entry, "bond", -1)),
            getattr(entry, "signature"),
        )
    return ("env", _environment_cache_signature(fallback_env))


def __getattr__(name):
    """
    Lazily expose moved environment-adjacent classes.

    :param name: Requested module attribute.
    :returns: The requested compatibility attribute.
    :raises AttributeError: If ``name`` is not a moved attribute.
    """

    if name == "EffectiveBlockOperator":
        from .effective import EffectiveBlockOperator

        return EffectiveBlockOperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cached_einsum_path(signature, *shapes):
    operands = [np.zeros(shape, dtype=float) for shape in shapes]
    path, _ = np.einsum_path(signature, *operands, optimize="greedy")
    return path


_TWO_SITE_KERNEL_PATH = _cached_einsum_path(
    "xal,xypo,yzqr,zbc->apqblorc",
    (2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2),
)
_FACTORIZED_PACKED_LOCAL_DIM = 2048
_DIAG_BLOCK_PATH = _cached_einsum_path(
    "al,abp,bcq,cr->lprq",
    (2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2),
)


def _contract_rank_coupled_left_step(E_block, A_conj, W_block, B_block):
    """
    Contract one left environment update term without dynamic path planning.

    :param E_block: Left environment component with indices ``xij``.
    :param A_conj: Conjugated bra MPS block with indices ``ipr``.
    :param W_block: Reduced MPO block with indices ``xypq``.
    :param B_block: Ket MPS block with indices ``jqs``.
    :returns: Contribution with indices ``yrs``.
    """

    tmp = np.einsum("xij,ipr->xjpr", E_block, A_conj, optimize=False)
    tmp = np.einsum("xjpr,jqs->xprqs", tmp, B_block, optimize=False)
    return np.einsum("xprqs,xypq->yrs", tmp, W_block, optimize=False)


def _contract_rank_coupled_right_step(A_conj, W_block, F_block, B_block):
    """
    Contract one right environment update term without dynamic path planning.

    :param A_conj: Conjugated bra MPS block with indices ``ipr``.
    :param W_block: Reduced MPO block with indices ``xypq``.
    :param F_block: Right environment component with indices ``yrs``.
    :param B_block: Ket MPS block with indices ``jqs``.
    :returns: Contribution with indices ``xij``.
    """

    tmp = np.einsum("yrs,jqs->yrjq", F_block, B_block, optimize=False)
    tmp = np.einsum("xypq,yrjq->xprj", W_block, tmp, optimize=False)
    return np.einsum("ipr,xprj->xij", A_conj, tmp, optimize=False)


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
    if isinstance(pack_layout, TwoSiteBasis):
        for entry, block in pack_layout.iter_packed_blocks(vector, drop_zeros=False):
            dense[dense_layout["block_slices"][entry.key]] = block
        return dense
    for entry in pack_layout:
        dense[dense_layout["block_slices"][entry.key]] = vector[
            entry.offset:entry.offset + entry.size
        ].reshape(entry.shape)
    return dense


def _dense_to_packed_vector(dense, *, pack_layout, dense_layout):
    dense = np.asarray(dense)
    dtype = dense.dtype
    if isinstance(pack_layout, TwoSiteBasis):
        blocks = {
            entry.key: dense[dense_layout["block_slices"][entry.key]]
            for entry in pack_layout
        }
        return pack_layout.blocks_to_packed(blocks, dtype=dtype)

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


def _build_packed_local_actions(E, W1, W2, F, two_site_template, basis, dense_layout):
    dim = basis.size

    def dense_apply_from_vector(vector):
        dense_in = _packed_vector_to_dense(
            vector,
            pack_layout=basis,
            dense_layout=dense_layout,
        )
        dense_out = _apply_two_site_dense(E, W1, W2, F, dense_in)
        return _dense_to_packed_vector(
            dense_out,
            pack_layout=basis,
            dense_layout=dense_layout,
        )

    diag = np.zeros(dim, dtype=float)
    for entry in basis:
        l_out, p1_out, p2_out, r_out = _key_slices(dense_layout, entry.key)
        diag_block = _two_site_diagonal_block(
            np.asarray(E[:, l_out, l_out]),
            np.asarray(W1[:, :, p1_out, p1_out]),
            np.asarray(W2[:, :, p2_out, p2_out]),
            np.asarray(F[:, r_out, r_out]),
        )
        diag[entry.offset:entry.offset + entry.size] = np.real(diag_block).reshape(entry.size)
    return dense_apply_from_vector, diag


def _build_tensor_local_actions(E, W1, W2, F, two_site_template, basis, dense_layout):
    dim = basis.size

    def tensor_apply(two_site):
        return _apply_two_site_block_sparse(E, W1, W2, F, two_site, dense_layout)

    diag = np.zeros(dim, dtype=float)
    for entry in basis:
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


def _nonzero_rank_coupled_blocks(blocks, *, tol=0.0):
    out = []
    for idx, block in enumerate(blocks):
        arr = np.asarray(block)
        if arr.size and np.any(np.abs(arr) > tol):
            out.append((idx, arr))
    return tuple(out)


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


@dataclass(frozen=True)
class _BlockEnvironment:
    """
    Mapping-like renormalized environment block.

    :param data: Sector-pair keyed environment blocks.
    :param rank_coupled: Whether ``data`` stores rank-coupled channel tuples.
    """

    data: object
    rank_coupled: bool = False

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        """Return an environment block by key."""

        return self.data.get(key, default)

    def items(self):
        """Return sector-pair keyed environment items."""

        return self.data.items()

    def keys(self):
        """Return environment sector-pair keys."""

        return self.data.keys()

    def values(self):
        """Return environment block values."""

        return self.data.values()

    def copy(self):
        """
        Return a deep array copy of this environment block.

        :returns: Environment block of the same side and rank-coupled type.
        """

        copied = (
            _copy_rank_coupled_env_map(self.data)
            if self.rank_coupled
            else {key: np.array(block, copy=True) for key, block in self.data.items()}
        )
        return type(self)(copied, rank_coupled=self.rank_coupled)

    def expectation(self):
        """
        Contract the scalar trace represented by this environment block.

        :returns: Complex scalar environment contraction.
        """

        return _environment_map_expectation(self.data, rank_coupled=self.rank_coupled)


@dataclass(frozen=True)
class LeftBlock(_BlockEnvironment):
    """
    Left renormalized block environment for a sweep boundary.

    This mirrors block-DMRG terminology: the object owns the left block basis
    sector-pair map and advances by absorbing one MPS/MPO site.
    """

    def advance(self, W, bra_site, ket_site, *, phys_slices=None):
        """
        Absorb one site into this left block.

        :param W: MPO core for the absorbed site.
        :param bra_site: Bra-side site tensor.
        :param ket_site: Ket-side site tensor.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Advanced ``LeftBlock``.
        """

        if self.rank_coupled:
            data = _contract_from_left_blocks_rank_coupled(W, bra_site, self, ket_site)
        else:
            if phys_slices is None:
                raise ValueError("LeftBlock.advance requires phys_slices for non-rank-coupled blocks.")
            data = _contract_from_left_blocks(W, bra_site, self, ket_site, phys_slices)
        return LeftBlock(data, rank_coupled=self.rank_coupled)


@dataclass(frozen=True)
class RightBlock(_BlockEnvironment):
    """
    Right renormalized block environment for a sweep boundary.

    The object owns the right block sector-pair map and advances by absorbing
    one MPS/MPO site from the right.
    """

    def advance(self, W, bra_site, ket_site, *, phys_slices=None):
        """
        Absorb one site into this right block.

        :param W: MPO core for the absorbed site.
        :param bra_site: Bra-side site tensor.
        :param ket_site: Ket-side site tensor.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Advanced ``RightBlock``.
        """

        if self.rank_coupled:
            data = _contract_from_right_blocks_rank_coupled(W, bra_site, self, ket_site)
        else:
            if phys_slices is None:
                raise ValueError("RightBlock.advance requires phys_slices for non-rank-coupled blocks.")
            data = _contract_from_right_blocks(W, bra_site, self, ket_site, phys_slices)
        return RightBlock(data, rank_coupled=self.rank_coupled)


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
            contrib = _contract_rank_coupled_left_step(
                np.asarray(E_block),
                np.asarray(A_block).conj(),
                np.asarray(W_slice),
                np.asarray(B_block),
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
        e_entries = _nonzero_rank_coupled_blocks(E_blocks)
        if not e_entries:
            continue
        e_arrays = tuple(block for _idx, block in e_entries)
        e_arrays_by_rank = {idx: block for idx, block in e_entries}
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
                    e_array = e_arrays_by_rank.get(left_idx)
                    if e_array is None:
                        continue
                    target[right_idx] += _contract_rank_coupled_left_step(
                        e_array,
                        A_conj,
                        np.asarray(w_block),
                        B_arr,
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
            contrib = _contract_rank_coupled_right_step(
                np.asarray(A_block).conj(),
                np.asarray(W_slice),
                np.asarray(F_block),
                np.asarray(B_block),
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
        f_entries = _nonzero_rank_coupled_blocks(F_blocks)
        if not f_entries:
            continue
        f_arrays = tuple(block for _idx, block in f_entries)
        f_arrays_by_rank = {idx: block for idx, block in f_entries}
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
                    f_array = f_arrays_by_rank.get(right_idx)
                    if f_array is None:
                        continue
                    target[left_idx] += _contract_rank_coupled_right_step(
                        A_conj,
                        np.asarray(w_block),
                        f_array,
                        B_arr,
                    )
    return {key: tuple(blocks) for key, blocks in out.items()}


def _precompute_two_site_block_env_transitions(
    E_map,
    W1,
    W2,
    F_map,
    basis,
    phys1_slices,
    phys2_slices,
    left_blocks_by_ket=None,
    right_blocks_by_ket=None,
):
    transitions = {}
    kernel_cache = {}
    out_entries = basis.out_entries
    out_index = basis.index_by_key()
    if left_blocks_by_ket is None:
        left_blocks_by_ket = {}
        for (q_out, q_in), block in E_map.items():
            left_blocks_by_ket.setdefault(q_in, []).append((q_out, block))
    if right_blocks_by_ket is None:
        right_blocks_by_ket = {}
        for (q_out, q_in), block in F_map.items():
            right_blocks_by_ket.setdefault(q_in, []).append((q_out, block))

    w1_blocks_by_in = _group_mpo_blocks_by_input(W1, phys1_slices)
    w2_blocks_by_in = _group_mpo_blocks_by_input(W2, phys2_slices)

    for in_entry in basis:
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
                        out_entry = basis[out_idx]
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


def _group_rank_coupled_reduced_blocks_by_input(W):
    grouped = {}
    for q_in in W.phys_in_leg.sectors:
        entries = []
        for q_out in W.phys_out_leg.sectors:
            reduced = W.reduced_block(q_out, q_in)
            if reduced:
                entries.append((q_out, reduced))
        grouped[q_in] = tuple(entries)
    return grouped


def _precompute_two_site_rank_coupled_factorized_terms(
    E_map,
    W1,
    W2,
    F_map,
    basis,
    left_blocks_by_ket=None,
    right_blocks_by_ket=None,
    left_factor_table=None,
    right_factor_table=None,
):
    out_entries = basis.out_entries
    out_index = basis.index_by_key()
    left_factor_cache = {}
    right_factor_cache = {}
    terms = {}
    if left_blocks_by_ket is None:
        left_blocks_by_ket = {}
        for (q_out, q_in), blocks in E_map.items():
            entries = _nonzero_rank_coupled_blocks(blocks)
            if not entries:
                continue
            left_blocks_by_ket.setdefault(q_in, []).append(
                (q_out, dict(entries))
            )
    if right_blocks_by_ket is None:
        right_blocks_by_ket = {}
        for (q_out, q_in), blocks in F_map.items():
            entries = _nonzero_rank_coupled_blocks(blocks)
            if not entries:
                continue
            right_blocks_by_ket.setdefault(q_in, []).append(
                (q_out, dict(entries))
            )

    w1_blocks_by_in = _group_rank_coupled_reduced_blocks_by_input(W1)
    w2_blocks_by_in = _group_rank_coupled_reduced_blocks_by_input(W2)

    for in_entry in basis:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_terms = []
        if left_factor_table is not None and right_factor_table is not None:
            left_entries = left_factor_table.get((q_lk, q_p1k), ())
            right_entries = right_factor_table.get((q_rk, q_p2k), ())
            for q_lb, q_p1b, middle_idx, left_factor in left_entries:
                for q_rb, q_p2b, middle_idx_2, right_factor in right_entries:
                    if middle_idx_2 != middle_idx:
                        continue
                    out_key = (q_lb, q_p1b, q_p2b, q_rb)
                    out_idx = out_index.get(out_key)
                    if out_idx is not None:
                        in_terms.append((out_idx, left_factor, right_factor))
        else:
            for q_lb, E_entries in left_blocks_by_ket.get(q_lk, ()):
                for q_p1b, W1_blocks in w1_blocks_by_in.get(q_p1k, ()):
                    for q_rb, F_entries in right_blocks_by_ket.get(q_rk, ()):
                        for q_p2b, W2_blocks in w2_blocks_by_in.get(q_p2k, ()):
                            out_key = (q_lb, q_p1b, q_p2b, q_rb)
                            out_idx = out_index.get(out_key)
                            if out_idx is None:
                                continue
                            for (left_idx, middle_idx), W1_block in W1_blocks.items():
                                E_block = E_entries.get(left_idx)
                                if E_block is None:
                                    continue
                                left_key = (id(E_block), id(W1_block))
                                left_factor = left_factor_cache.get(left_key)
                                if left_factor is None:
                                    left_factor = np.asarray(
                                        _factorize_left_two_site_dense_term(E_block, W1_block)
                                    )
                                    left_factor_cache[left_key] = left_factor
                                for (middle_idx_2, right_idx), W2_block in W2_blocks.items():
                                    if middle_idx_2 != middle_idx:
                                        continue
                                    F_block = F_entries.get(right_idx)
                                    if F_block is None:
                                        continue
                                    right_key = (id(W2_block), id(F_block))
                                    right_factor = right_factor_cache.get(right_key)
                                    if right_factor is None:
                                        right_factor = np.asarray(
                                            _factorize_right_two_site_dense_term(W2_block, F_block)
                                        )
                                        right_factor_cache[right_key] = right_factor
                                    in_terms.append((out_idx, left_factor, right_factor))
        terms[in_entry.key] = tuple(in_terms)
    return out_entries, terms


def _build_rank_coupled_left_factor_table(left_blocks_by_ket, W):
    from .renormalized import build_rank_coupled_left_factor_table

    return build_rank_coupled_left_factor_table(left_blocks_by_ket, W)


def _build_rank_coupled_right_factor_table(right_blocks_by_ket, W):
    from .renormalized import build_rank_coupled_right_factor_table

    return build_rank_coupled_right_factor_table(right_blocks_by_ket, W)


def _precompute_two_site_block_env_factorized_terms(
    E_map,
    W1,
    W2,
    F_map,
    basis,
    phys1_slices,
    phys2_slices,
    left_blocks_by_ket=None,
    right_blocks_by_ket=None,
    left_factor_table=None,
    right_factor_table=None,
):
    out_entries = basis.out_entries
    out_index = basis.index_by_key()
    left_factor_cache = {}
    right_factor_cache = {}
    terms = {}
    if left_blocks_by_ket is None:
        left_blocks_by_ket = {}
        for (q_out, q_in), block in E_map.items():
            left_blocks_by_ket.setdefault(q_in, []).append((q_out, np.asarray(block)))
    if right_blocks_by_ket is None:
        right_blocks_by_ket = {}
        for (q_out, q_in), block in F_map.items():
            right_blocks_by_ket.setdefault(q_in, []).append((q_out, np.asarray(block)))

    w1_blocks_by_in = _group_mpo_blocks_by_input(W1, phys1_slices)
    w2_blocks_by_in = _group_mpo_blocks_by_input(W2, phys2_slices)

    for in_entry in basis:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_terms = []
        if left_factor_table is not None and right_factor_table is not None:
            for q_lb, q_p1b, left_factor in left_factor_table.get((q_lk, q_p1k), ()):
                for q_rb, q_p2b, right_factor in right_factor_table.get((q_rk, q_p2k), ()):
                    out_key = (q_lb, q_p1b, q_p2b, q_rb)
                    out_idx = out_index.get(out_key)
                    if out_idx is not None:
                        in_terms.append((out_idx, left_factor, right_factor))
        else:
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


def _build_left_factor_table(left_blocks_by_ket, W, phys_slices):
    from .renormalized import build_left_factor_table

    return build_left_factor_table(left_blocks_by_ket, W, phys_slices)


def _build_right_factor_table(right_blocks_by_ket, W, phys_slices):
    from .renormalized import build_right_factor_table

    return build_right_factor_table(right_blocks_by_ket, W, phys_slices)


def _apply_two_site_block_env(transitions, theta, out_entries, *, base_dtype):
    return apply_transition_tensor(
        transitions,
        theta,
        out_entries,
        base_dtype=base_dtype,
    )


def _apply_two_site_block_env_reduced(transitions, state, out_entries, *, base_dtype):
    return apply_transition_reduced(
        transitions,
        state,
        out_entries,
        base_dtype=base_dtype,
    )


def _apply_two_site_block_env_reduced_compiled(compiled_transitions, state, *, base_dtype):
    """
    Apply compiled packed transitions to a reduced state vector.

    :param compiled_transitions: Basis-aware compiled transition kernels.
    :param state: Reduced local state in the same two-site basis.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Reduced state vector with blocks assembled from the compiled basis.
    """
    return apply_compiled_transition_reduced(
        compiled_transitions,
        state,
        base_dtype=base_dtype,
    )


def _apply_two_site_block_env_packed(transitions, vector, basis, *, base_dtype):
    return apply_packed_transitions(transitions, vector, basis, base_dtype=base_dtype)


def _compile_packed_transitions(transitions, basis):
    return compile_packed_transitions(transitions, basis)


def _materialize_packed_matrix_from_compiled(compiled_transitions, *, dtype=None):
    return materialize_packed_matrix(compiled_transitions, dtype=dtype)


def _materialize_packed_csr_from_compiled(compiled_transitions, *, dtype=None):
    return materialize_packed_csr(compiled_transitions, dtype=dtype)


def _apply_two_site_block_env_packed_compiled(compiled_transitions, vector, *, base_dtype):
    return apply_compiled_packed_transitions(
        compiled_transitions,
        vector,
        base_dtype=base_dtype,
    )


def _apply_two_site_block_env_packed_factorized(factorized_terms, vector, basis, *, base_dtype):
    return apply_factorized_packed_terms(
        factorized_terms,
        vector,
        basis,
        base_dtype=base_dtype,
    )


def _compile_factorized_terms(factorized_terms, basis):
    return compile_factorized_terms(factorized_terms, basis)


def _diagonal_from_factorized_terms(factorized_terms, basis, *, dtype=float):
    return diagonal_from_factorized_terms(factorized_terms, basis, dtype=dtype)


def _transitions_are_identity_operator(basis, transitions, *, tol=1e-12):
    return transitions_are_identity_operator(basis, transitions, tol=tol)


def _identity_env_to_matrix(block, *, dtype):
    return identity_env_to_matrix(block, dtype=dtype)


def _identity_mpo_transitions(E_map, F_map, basis, *, base_dtype):
    return identity_mpo_transitions(E_map, F_map, basis, base_dtype=base_dtype)


def _build_identity_mpo_local_actions(E_map, F_map, basis, *, base_dtype):
    return build_identity_mpo_local_actions(
        E_map,
        F_map,
        basis,
        base_dtype=base_dtype,
    )


def _group_boundary_blocks_by_ket(block_map, representation):
    grouped = {}
    if representation == "rank_coupled_by_ket":
        for (q_out, q_in), blocks in block_map.items():
            entries = _nonzero_rank_coupled_blocks(blocks)
            if entries:
                grouped.setdefault(q_in, []).append((q_out, dict(entries)))
    elif representation == "array_by_ket":
        for (q_out, q_in), block in block_map.items():
            grouped.setdefault(q_in, []).append((q_out, np.asarray(block)))
    else:
        for (q_out, q_in), block in block_map.items():
            grouped.setdefault(q_in, []).append((q_out, block))
    return {key: tuple(value) for key, value in grouped.items()}


def _renormalized_side_table_builders(*, rank_coupled):
    """
    Return builders for one-sided renormalized operator tables.

    :param rank_coupled: Whether rank-coupled side tables are needed.
    :returns: Mapping from representation name to grouping callable.
    """

    if rank_coupled:
        return {
            "rank_coupled_by_ket": lambda block: _group_boundary_blocks_by_ket(
                block,
                "rank_coupled_by_ket",
            )
        }
    return {
        "block_by_ket": lambda block: _group_boundary_blocks_by_ket(
            block,
            "block_by_ket",
        ),
        "array_by_ket": lambda block: _group_boundary_blocks_by_ket(
            block,
            "array_by_ket",
        ),
    }


def _effective_block_operator_from_parts(
    E_map,
    W1,
    W2,
    F_map,
    two_site_template,
    basis,
    *,
    phys1_slices=None,
    phys2_slices=None,
    rank_coupled=False,
    left_entry=None,
    right_entry=None,
    complementary_operator_families=None,
    name=None,
):
    """
    Assemble an explicit two-site effective block operator from block pieces.

    :param E_map: Left renormalized environment block mapping.
    :param W1: MPO core on the left active site.
    :param W2: MPO core on the right active site.
    :param F_map: Right renormalized environment block mapping.
    :param two_site_template: Rank-4 two-site tensor defining local sectors.
    :param basis: Explicit packed two-site basis.
    :param phys1_slices: Physical sector slices for the left active site.
    :param phys2_slices: Physical sector slices for the right active site.
    :param rank_coupled: Whether the operator uses rank-coupled reduced blocks.
    :param left_entry: Optional persisted left boundary-stack entry.
    :param right_entry: Optional persisted right boundary-stack entry.
    :param complementary_operator_families: Optional block2-style
        complementary families owned by the Hamiltonian stack.
    :param name: Optional operator name used by diagnostics.
    :returns: ``EffectiveBlockOperator`` owning the local block problem.
    """

    from .effective import EffectiveBlockOperator

    return EffectiveBlockOperator(
        left_block=E_map,
        mpo_left=W1,
        mpo_right=W2,
        right_block=F_map,
        two_site_template=two_site_template,
        basis=basis,
        phys1_slices=phys1_slices,
        phys2_slices=phys2_slices,
        rank_coupled=rank_coupled,
        left_entry=left_entry,
        right_entry=right_entry,
        complementary_operator_families=complementary_operator_families,
        name=name,
    )


def _local_actions_without_diag(
    E_map,
    W1,
    W2,
    F_map,
    two_site_template,
    basis,
    phys1_slices,
    phys2_slices,
    *,
    out_dtype,
):
    return _effective_block_operator_from_parts(
        E_map,
        W1,
        W2,
        F_map,
        two_site_template,
        basis,
        phys1_slices=phys1_slices,
        phys2_slices=phys2_slices,
    )._standard_local_actions(out_dtype=out_dtype)


def _build_block_sparse_local_actions(
    E_map,
    W1,
    W2,
    F_map,
    two_site_template,
    basis,
    phys1_slices,
    phys2_slices,
):
    return _effective_block_operator_from_parts(
        E_map,
        W1,
        W2,
        F_map,
        two_site_template,
        basis,
        phys1_slices=phys1_slices,
        phys2_slices=phys2_slices,
    ).local_actions()


def _build_rank_coupled_block_sparse_local_actions(
    E_map,
    W1,
    W2,
    F_map,
    two_site_template,
    basis,
):
    return _effective_block_operator_from_parts(
        E_map,
        W1,
        W2,
        F_map,
        two_site_template,
        basis,
        rank_coupled=True,
    ).local_actions()

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
        basis = two_site_state_basis(two_site_template, layout=layout)
        dense_layout = _tensor_dense_layout(
            two_site_template,
            axis_overrides={
                1: _site_physical_override(self.site_layouts[bond], self.sites[bond]),
                2: _site_physical_override(self.site_layouts[bond + 1], self.sites[bond + 1]),
            },
        )
        tensor_matvec, diag = _build_tensor_local_actions(
            E, W1, W2, F, two_site_template, basis, dense_layout
        )
        return LocalOperator(
            tensor_matvec=tensor_matvec,
            basis=basis,
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
    generation: int = 0

    @classmethod
    def from_chain(cls, chain, direction="lr"):
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError(f"Unknown sweep direction {direction!r}.")
        if direction == "lr":
            current_env = chain.left_envs[0]
        else:
            current_env = chain.right_envs[-1]
        return cls(chain=chain, direction=direction, current_env=np.asarray(current_env), generation=0)

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

    def orthonormal_bond_operator(
        self,
        bond,
        two_site_template,
        norm_sweep,
        *,
        tol=1.0e-12,
        max_dim=None,
        cache=None,
        require_block_sparse_table=False,
        profile=False,
    ):
        """
        Build a standard local problem in an orthonormal renormalized basis.

        :param bond: Active left-site index.
        :param two_site_template: Rank-4 two-site tensor defining local sectors.
        :param norm_sweep: Matching identity/norm environment sweep.
        :param tol: Metric eigenvalue cutoff.
        :param max_dim: Optional maximum parent packed dimension to transform.
        :param cache: Optional mutable cache for transformed local operator
            tables keyed by environment and basis signatures.
        :param require_block_sparse_table: If True, disallow dense/global
            transformed-operator fallback.
        :returns: ``OrthonormalizedLocalProblem`` or ``None`` when skipped by
            ``max_dim``.
        """

        operator = self.bond_operator(bond, two_site_template)
        norm_operator = norm_sweep.bond_operator(bond, two_site_template)
        _, layout = pack_two_site_state(two_site_template)
        basis = two_site_state_basis(two_site_template, layout=layout)
        if cache is not None:
            right_env = self.chain.right_envs[bond + 1] if self.direction == "lr" else self.current_env
            left_env = self.current_env if self.direction == "lr" else self.chain.left_envs[bond]
            norm_right_env = norm_sweep.chain.right_envs[bond + 1] if self.direction == "lr" else norm_sweep.current_env
            norm_left_env = norm_sweep.current_env if self.direction == "lr" else norm_sweep.chain.left_envs[bond]
            cache_key = (
                "dense",
                self.direction,
                int(bond),
                id(left_env),
                id(right_env),
                id(norm_left_env),
                id(norm_right_env),
                int(self.generation),
                int(norm_sweep.generation),
                _basis_cache_signature(basis),
                float(tol),
                None if max_dim is None else int(max_dim),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return replace(cached, cache_hit=True)
        problem = build_orthonormalized_local_problem(
            operator,
            norm_operator,
            two_site_template,
            basis,
            tol=tol,
            max_dim=max_dim,
            require_block_sparse_table=require_block_sparse_table,
            name=f"bond-{bond}-orthonormal-renormalized-H",
            source="dense_environment_sweep",
            cache_hit=False,
            moving_environment_cache=(
                None
                if getattr(self, "renormalized_blocks", None) is None
                else self.renormalized_blocks.moving_environment_cache
            ),
        )
        if cache is not None and problem is not None:
            cache[cache_key] = problem
        return problem

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
        self.generation += 1
        return self

    def clone(self):
        return DenseEnvironmentSweep(
            chain=self.chain,
            direction=self.direction,
            current_env=np.array(self.current_env, copy=True),
            generation=int(self.generation),
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
    basis = two_site_state_basis(two_site_template, layout=layout)
    dense_layout = _tensor_dense_layout(
        two_site_template,
        axis_overrides={
            1: _site_physical_override(env_chain.site_layouts[bond], env_chain.sites[bond]),
            2: _site_physical_override(env_chain.site_layouts[bond + 1], env_chain.sites[bond + 1]),
        },
    )
    dense_matvec, diag = _build_packed_local_actions(
        E, W1, W2, F, two_site_template, basis, dense_layout
    )
    return LocalOperator(
        matvec=dense_matvec,
        basis=basis,
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
    renormalized_blocks: RenormalizedBlockStack | None = None
    require_symbolic_payloads: bool = False

    @classmethod
    def build(
        cls,
        sites,
        mpo_factors,
        *,
        renormalized_blocks=None,
        require_symbolic_payloads=False,
        sweep_direction=None,
    ):
        """
        Build block-sparse renormalized environments for a chain.

        :param sites: MPS site tensors.
        :param mpo_factors: One MPO core per site.
        :param renormalized_blocks: Optional persistent boundary-stack owner.
        :param require_symbolic_payloads: If True, local operators must use
            symbolic boundary payloads.
        :param sweep_direction: Optional sweep direction.  When ``"lr"``, only
            the right boundary stack and initial left block are prebuilt; when
            ``"rl"``, only the left boundary stack and initial right block are
            prebuilt.  ``None`` preserves the full two-sided build.
        :returns: :class:`BlockSparseEnvironmentChain`.
        """

        if len(sites) != len(mpo_factors):
            raise ValueError("BlockSparseEnvironmentChain requires one MPO core per site tensor.")
        if len(sites) < 2:
            raise ValueError("BlockSparseEnvironmentChain requires at least two sites.")
        if sweep_direction is not None:
            sweep_direction = str(sweep_direction).lower()
            if sweep_direction not in {"lr", "rl"}:
                raise ValueError(f"Unknown sweep direction {sweep_direction!r}.")
        sites = [normalize_site_tensor_layout(site) for site in sites]

        site_layouts = [_tensor_dense_layout(site) for site in sites]
        phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
        sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
            mpo_factors,
            site_layouts=site_layouts,
        )

        rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
        nsites = len(sites)
        build_left = sweep_direction is None or sweep_direction == "rl"
        build_right = sweep_direction is None or sweep_direction == "lr"
        if rank_coupled:
            initial_left = LeftBlock(
                _initial_left_env_blocks_rank_coupled(site_layouts[0], sparse_mpo_factors[0]),
                rank_coupled=True,
            )
            if build_left:
                left_envs = [initial_left]
                for i in range(nsites - 1):
                    left_envs.append(
                        left_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                        )
                    )
            else:
                left_envs = [initial_left] + [None] * (nsites - 1)

            initial_right = RightBlock(
                _initial_right_env_blocks_rank_coupled(site_layouts[-1], sparse_mpo_factors[-1]),
                rank_coupled=True,
            )
            if build_right:
                right_envs = [initial_right]
                for i in range(nsites - 1, 0, -1):
                    right_envs.append(
                        right_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                        )
                    )
                right_envs = list(reversed(right_envs))
            else:
                right_envs = [None] * (nsites - 1) + [initial_right]
        else:
            initial_left = LeftBlock(
                _initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0]),
                rank_coupled=False,
            )
            if build_left:
                left_envs = [initial_left]
                for i in range(nsites - 1):
                    left_envs.append(
                        left_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                            phys_slices=phys_slice_maps[i],
                        )
                    )
            else:
                left_envs = [initial_left] + [None] * (nsites - 1)

            initial_right = RightBlock(
                _initial_right_env_blocks(site_layouts[-1], sparse_mpo_factors[-1]),
                rank_coupled=False,
            )
            if build_right:
                right_envs = [initial_right]
                for i in range(nsites - 1, 0, -1):
                    right_envs.append(
                        right_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                            phys_slices=phys_slice_maps[i],
                        )
                    )
                right_envs = list(reversed(right_envs))
            else:
                right_envs = [None] * (nsites - 1) + [initial_right]
        chain = cls(
            list(sites),
            list(sparse_mpo_factors),
            site_layouts,
            left_envs,
            right_envs,
            rank_coupled,
            renormalized_blocks,
            bool(require_symbolic_payloads),
        )
        chain.store_boundary_blocks()
        return chain

    def store_boundary_blocks(self):
        """
        Persist all prebuilt left/right boundary blocks into the stack.

        :returns: ``self`` for chaining.
        """

        if self.renormalized_blocks is None:
            return self
        side_table_builders = _renormalized_side_table_builders(
            rank_coupled=self.rank_coupled,
        )
        for bond, block in enumerate(self.left_envs):
            if block is None:
                continue
            entry = self.renormalized_blocks.initialize(
                "left",
                bond,
                block,
                side_table_builders=side_table_builders,
            )
            self._prepopulate_boundary_factor_tables(entry)
        for bond, block in enumerate(self.right_envs):
            if block is None:
                continue
            entry = self.renormalized_blocks.initialize(
                "right",
                bond,
                block,
                side_table_builders=side_table_builders,
            )
            self._prepopulate_boundary_factor_tables(entry)
        return self

    def _prepopulate_boundary_factor_tables(self, entry):
        """
        Persist adjacent one-site factor tables on a boundary entry.

        Factor tables depend on the boundary side and adjacent MPO core.  They
        are stored on the same boundary entry as the grouped side tables so the
        local effective-H builder can reuse them without rebuilding raw
        environment-map contractions.

        :param entry: Boundary-stack entry to populate.
        :returns: ``entry``.
        """

        t0 = time.perf_counter()
        try:
            return self._prepopulate_boundary_factor_tables_impl(entry)
        finally:
            if entry is not None:
                entry.advance_timing["factor_table_prepopulate"] = (
                    entry.advance_timing.get("factor_table_prepopulate", 0.0)
                    + (time.perf_counter() - t0)
                )

    def _prepopulate_boundary_factor_tables_impl(self, entry):
        """
        Implementation for :meth:`_prepopulate_boundary_factor_tables`.

        :param entry: Boundary-stack entry to populate.
        :returns: ``entry``.
        """

        if entry is None:
            return entry
        symbolic_table = getattr(entry, "symbolic_operator_table", None)
        if symbolic_table is None:
            return entry
        if entry.side == "left":
            if entry.bond >= len(self.mpo_factors) - 1:
                return entry
            factor_representation = (
                "rank_coupled_left_factor_by_ket"
                if self.rank_coupled
                else "left_factor_by_ket"
            )
        elif entry.side == "right":
            if entry.bond <= 0 or entry.bond >= len(self.mpo_factors):
                return entry
            factor_representation = (
                "rank_coupled_right_factor_by_ket"
                if self.rank_coupled
                else "right_factor_by_ket"
            )
        else:
            return entry

        W = self.mpo_factors[entry.bond]
        key = (
            "side_operator_table",
            factor_representation,
            entry.signature,
            id(W),
        )
        if key in entry.side_operator_tables:
            return entry
        base_representation = "rank_coupled_by_ket" if self.rank_coupled else "array_by_ket"
        base_key = ("side_operator_table", base_representation, entry.signature)
        base_record = entry.side_operator_tables.get(base_key)
        phys_slices = None if self.rank_coupled else self.site_layouts[entry.bond]["sector_slices"][1]
        grouped = symbolic_table.factor_boundary_blocks(
            factor_representation,
            W,
            phys_slices=phys_slices,
        )
        entry.put_side_operator_table(
            key,
            grouped,
            representation=factor_representation,
            source="symbolic_prepared",
            parent_table=base_record,
        )
        return entry

    def boundary_block(self, side, bond, default=None):
        """
        Return a persisted boundary block when available.

        :param side: Boundary side, either ``"left"`` or ``"right"``.
        :param bond: Boundary bond index.
        :param default: Fallback environment block.
        :returns: Persisted or fallback environment block.
        """

        if self.renormalized_blocks is None:
            return default
        return self.renormalized_blocks.block(side, bond, default)

    def boundary_entry(self, side, bond):
        """
        Return a persisted boundary stack entry when available.

        :param side: Boundary side, either ``"left"`` or ``"right"``.
        :param bond: Boundary bond index.
        :returns: Boundary entry or ``None``.
        """

        if self.renormalized_blocks is None:
            return None
        return self.renormalized_blocks.get(side, bond)

    def _bond_operator_from_envs(
        self,
        bond,
        two_site_template,
        *,
        left_env,
        right_env,
        left_entry=None,
        right_entry=None,
    ):
        return self.effective_block_operator(
            bond,
            two_site_template,
            left_env=left_env,
            right_env=right_env,
            left_entry=left_entry,
            right_entry=right_entry,
        ).to_local_operator()

    def effective_block_operator(
        self,
        bond,
        two_site_template,
        *,
        left_env=None,
        right_env=None,
        left_entry=None,
        right_entry=None,
    ):
        """
        Build an explicit two-site effective block operator for one bond.

        :param bond: Active left-site index.
        :param two_site_template: Rank-4 two-site tensor defining local sectors.
        :param left_env: Optional left block override.
        :param right_env: Optional right block override.
        :param left_entry: Optional persisted left boundary-stack entry.
        :param right_entry: Optional persisted right boundary-stack entry.
        :returns: ``EffectiveBlockOperator`` for the active bond.
        """

        if bond < 0 or bond >= len(self.sites) - 1:
            raise IndexError(f"Bond {bond} out of range for chain length {len(self.sites)}.")
        W1 = self.mpo_factors[bond]
        W2 = self.mpo_factors[bond + 1]
        if left_env is None:
            left_entry = self.boundary_entry("left", bond)
            left_env = left_entry.block if left_entry is not None else self.left_envs[bond]
        if right_env is None:
            right_entry = self.boundary_entry("right", bond + 1)
            right_env = (
                right_entry.block if right_entry is not None else self.right_envs[bond + 1]
            )
        if self.require_symbolic_payloads:
            self._require_symbolic_payload_entry(left_entry, "left", bond)
            self._require_symbolic_payload_entry(right_entry, "right", bond + 1)
        _, layout = pack_two_site_state(two_site_template)
        basis = two_site_state_basis(two_site_template, layout=layout)
        phys1_slices = self.site_layouts[bond]["sector_slices"][1]
        phys2_slices = self.site_layouts[bond + 1]["sector_slices"][1]
        operator = _effective_block_operator_from_parts(
            left_env,
            W1,
            W2,
            right_env,
            two_site_template,
            basis,
            phys1_slices=phys1_slices,
            phys2_slices=phys2_slices,
            rank_coupled=self.rank_coupled,
            left_entry=left_entry,
            right_entry=right_entry,
            complementary_operator_families=(
                None
                if self.renormalized_blocks is None
                else self.renormalized_blocks.complementary_operator_families
            ),
            name=f"bond-{bond}-block-sparse-effective-H",
        )
        if left_entry is not None or right_entry is not None:
            from .effective import RenormalizedLocalOperatorTableBuilder

            builder = RenormalizedLocalOperatorTableBuilder(operator)
            operator.prepare_local_side_operator_tables(
                builder.representation(),
                require_symbolic_payloads=self.require_symbolic_payloads,
            )
            builder.build()
        return operator

    def _require_symbolic_payload_entry(self, entry, side, bond):
        """
        Require a boundary entry with symbolic-owned numeric payloads.

        This is the block2-like mode guard: local effective operators should be
        assembled from persisted recursive operator stacks, not by falling back
        to raw environment maps.

        :param entry: Boundary stack entry.
        :param side: Boundary side used in diagnostics.
        :param bond: Boundary bond index.
        :returns: ``None``.
        """

        if entry is None:
            raise RuntimeError(
                f"Strict symbolic local build requires {side} boundary entry at bond {bond}."
            )
        table = getattr(entry, "symbolic_operator_table", None)
        if table is None:
            raise RuntimeError(
                f"Strict symbolic local build requires symbolic table for "
                f"{side} boundary {bond}."
            )
        if not table.stats.get("owns_numeric_payloads", False):
            raise RuntimeError(
                f"Strict symbolic local build requires numeric payload ownership for "
                f"{side} boundary {bond}."
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
    current_env: _BlockEnvironment
    generation: int = 0
    renormalized_blocks: RenormalizedBlockStack | None = None
    current_entry: object | None = None

    @classmethod
    def from_chain(cls, chain, direction="lr"):
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError(f"Unknown sweep direction {direction!r}.")
        if direction == "lr":
            current_env = chain.left_envs[0].copy()
        else:
            current_env = chain.right_envs[-1].copy()
        current_entry = None
        if chain.renormalized_blocks is not None:
            side_table_builders = _renormalized_side_table_builders(
                rank_coupled=chain.rank_coupled,
            )
            current_entry = chain.renormalized_blocks.initialize(
                "left" if direction == "lr" else "right",
                0 if direction == "lr" else len(chain.sites) - 1,
                current_env,
                side_table_builders=side_table_builders,
            )
        return cls(
            chain=chain,
            direction=direction,
            current_env=current_env,
            generation=0,
            renormalized_blocks=chain.renormalized_blocks,
            current_entry=current_entry,
        )

    def _boundary_block(self, side, bond, default):
        if self.renormalized_blocks is None:
            return default
        return self.renormalized_blocks.block(side, bond, default)

    def _boundary_entry(self, side, bond):
        if self.renormalized_blocks is None:
            return None
        return self.renormalized_blocks.get(side, bond)

    def _store_current_boundary(self, bond):
        if self.renormalized_blocks is None:
            return
        self.current_entry = self.renormalized_blocks.put(
            "left" if self.direction == "lr" else "right",
            int(bond),
            self.current_env,
        )

    def bond_operator(self, bond, two_site_template):
        if self.direction == "lr":
            right_entry = self._boundary_entry("right", bond + 1)
            return self.chain._bond_operator_from_envs(
                bond,
                two_site_template,
                left_env=self.current_env,
                right_env=self._boundary_block(
                    "right",
                    bond + 1,
                    self.chain.right_envs[bond + 1],
                ),
                left_entry=self.current_entry,
                right_entry=right_entry,
            )
        left_entry = self._boundary_entry("left", bond)
        return self.chain._bond_operator_from_envs(
            bond,
            two_site_template,
            left_env=self._boundary_block("left", bond, self.chain.left_envs[bond]),
            right_env=self.current_env,
            left_entry=left_entry,
            right_entry=self.current_entry,
        )

    def orthonormal_bond_operator(
        self,
        bond,
        two_site_template,
        norm_sweep,
        *,
        tol=1.0e-12,
        max_dim=None,
        cache=None,
        require_block_sparse_table=False,
        profile=False,
    ):
        """
        Build a standard local problem in an orthonormal reduced block basis.

        :param bond: Active left-site index.
        :param two_site_template: Rank-4 two-site tensor defining local sectors.
        :param norm_sweep: Matching identity/norm environment sweep.
        :param tol: Metric eigenvalue cutoff.
        :param max_dim: Optional maximum parent packed dimension to transform.
        :param cache: Optional mutable cache for transformed local operator
            tables keyed by environment and basis signatures.
        :param require_block_sparse_table: If True, disallow dense/global
            transformed-operator fallback.
        :param profile: If True, attach operator-factory timing diagnostics.
        :returns: ``OrthonormalizedLocalProblem`` or ``None`` when skipped by
            ``max_dim``.
        """

        timing = {
            "hamiltonian_operator": 0.0,
            "norm_operator": 0.0,
            "cache_lookup": 0.0,
            "build_orthonormalized_problem": 0.0,
        } if profile else None
        _, layout = pack_two_site_state(two_site_template)
        basis = two_site_state_basis(two_site_template, layout=layout)
        cache_key = None
        if cache is not None:
            t0 = time.perf_counter() if profile else None
            if self.direction == "lr":
                right_entry = self._boundary_entry("right", bond + 1)
                left_entry = self.current_entry
                norm_right_entry = norm_sweep._boundary_entry("right", bond + 1)
                norm_left_entry = norm_sweep.current_entry
                right_env = right_entry.block if right_entry is not None else self.chain.right_envs[bond + 1]
                left_env = self.current_env
                norm_right_env = (
                    norm_right_entry.block
                    if norm_right_entry is not None
                    else norm_sweep.chain.right_envs[bond + 1]
                )
                norm_left_env = norm_sweep.current_env
            else:
                right_entry = self.current_entry
                left_entry = self._boundary_entry("left", bond)
                norm_right_entry = norm_sweep.current_entry
                norm_left_entry = norm_sweep._boundary_entry("left", bond)
                right_env = self.current_env
                left_env = left_entry.block if left_entry is not None else self.chain.left_envs[bond]
                norm_right_env = norm_sweep.current_env
                norm_left_env = (
                    norm_left_entry.block
                    if norm_left_entry is not None
                    else norm_sweep.chain.left_envs[bond]
                )
            cache_key = (
                "renormalized_operator_stack",
                self.direction,
                int(bond),
                _renormalized_entry_cache_signature(left_entry, left_env),
                _renormalized_entry_cache_signature(right_entry, right_env),
                _renormalized_entry_cache_signature(norm_left_entry, norm_left_env),
                _renormalized_entry_cache_signature(norm_right_entry, norm_right_env),
                _basis_cache_signature(basis),
                float(tol),
                None if max_dim is None else int(max_dim),
            )
            cached = cache.get(cache_key)
            if profile:
                timing["cache_lookup"] += time.perf_counter() - t0
            if cached is not None:
                if profile:
                    metadata = dict(cached.metadata or {})
                    metadata["orthonormal_operator_factory_timing"] = {
                        key: float(value)
                        for key, value in timing.items()
                    }
                    cached = replace(cached, cache_hit=True, metadata=metadata)
                else:
                    cached = replace(cached, cache_hit=True)
                return cached
        t0 = time.perf_counter() if profile else None
        operator = self.bond_operator(bond, two_site_template)
        if profile:
            timing["hamiltonian_operator"] += time.perf_counter() - t0
            t0 = time.perf_counter()
        norm_operator = norm_sweep.bond_operator(bond, two_site_template)
        if profile:
            timing["norm_operator"] += time.perf_counter() - t0
        t0 = time.perf_counter() if profile else None
        problem = build_orthonormalized_local_problem(
            operator,
            norm_operator,
            two_site_template,
            basis,
            tol=tol,
            max_dim=max_dim,
            require_block_sparse_table=require_block_sparse_table,
            name=f"bond-{bond}-orthonormal-renormalized-H",
            source="block_sparse_environment_sweep",
            cache_hit=False,
            profile=profile,
            moving_environment_cache=(
                None
                if self.renormalized_blocks is None
                else self.renormalized_blocks.moving_environment_cache
            ),
        )
        if profile:
            timing["build_orthonormalized_problem"] += time.perf_counter() - t0
            if problem is not None:
                metadata = dict(problem.metadata or {})
                metadata["orthonormal_operator_factory_timing"] = {
                    key: float(value)
                    for key, value in timing.items()
                }
                problem = replace(problem, metadata=metadata)
        if cache is not None and problem is not None:
            if isinstance(cache, RenormalizedOperatorStack):
                cache.put(cache_key, problem)
            else:
                cache[cache_key] = problem
        return problem

    def advance_after_update(self, bond, left_site, right_site):
        if self.direction == "lr":
            phys_slices = (
                None
                if self.chain.rank_coupled
                else self.chain.site_layouts[bond]["sector_slices"][2]
            )
            if self.renormalized_blocks is not None and self.current_entry is not None:
                self.current_entry = self.renormalized_blocks.advance_left(
                    self.current_entry,
                    bond + 1,
                    self.chain.mpo_factors[bond],
                    left_site,
                    phys_slices=phys_slices,
                    side_table_builders=_renormalized_side_table_builders(
                        rank_coupled=self.chain.rank_coupled,
                    ),
                )
                self.chain._prepopulate_boundary_factor_tables(self.current_entry)
                self.current_env = self.current_entry.block
            else:
                self.current_env = self.current_env.advance(
                    self.chain.mpo_factors[bond],
                    left_site,
                    left_site,
                    phys_slices=phys_slices,
                )
                self._store_current_boundary(bond + 1)
        else:
            phys_slices = (
                None
                if self.chain.rank_coupled
                else self.chain.site_layouts[bond + 1]["sector_slices"][2]
            )
            if self.renormalized_blocks is not None and self.current_entry is not None:
                self.current_entry = self.renormalized_blocks.advance_right(
                    self.current_entry,
                    bond,
                    self.chain.mpo_factors[bond + 1],
                    right_site,
                    phys_slices=phys_slices,
                    side_table_builders=_renormalized_side_table_builders(
                        rank_coupled=self.chain.rank_coupled,
                    ),
                )
                self.chain._prepopulate_boundary_factor_tables(self.current_entry)
                self.current_env = self.current_entry.block
            else:
                self.current_env = self.current_env.advance(
                    self.chain.mpo_factors[bond + 1],
                    right_site,
                    right_site,
                    phys_slices=phys_slices,
                )
                self._store_current_boundary(bond)
        self.generation += 1
        return self

    def final_expectation(self, sites):
        if len(sites) != len(self.chain.sites):
            raise ValueError("final_expectation expects one site tensor per chain site.")
        if self.direction == "lr":
            site_index = len(sites) - 1
            site = sites[site_index]
            env = self.current_env.advance(
                self.chain.mpo_factors[site_index],
                site,
                site,
                phys_slices=(
                    None
                    if self.chain.rank_coupled
                    else self.chain.site_layouts[site_index]["sector_slices"][2]
                ),
            )
        else:
            site_index = 0
            site = sites[site_index]
            env = self.current_env.advance(
                self.chain.mpo_factors[site_index],
                site,
                site,
                phys_slices=(
                    None
                    if self.chain.rank_coupled
                    else self.chain.site_layouts[site_index]["sector_slices"][2]
                ),
            )
        return env.expectation()

    def clone(self):
        return BlockSparseEnvironmentSweep(
            chain=self.chain,
            direction=self.direction,
            current_env=self.current_env.copy(),
            generation=int(self.generation),
            renormalized_blocks=self.renormalized_blocks,
            current_entry=self.current_entry,
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
