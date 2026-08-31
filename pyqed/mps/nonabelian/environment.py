#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense environment and effective-H helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

import hashlib
import os
import ctypes
import sys
import time
import weakref
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .contraction import normalize_site_tensor_layout
from .coupling import clebsch_gordan, ordered_two_m_values
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
from .mpo import MPO, IrreducibleMPO, RankCoupledMPO, iter_virtual_routes
from .solver import (
    LocalOperator,
    ReducedStateVector,
    build_orthonormalized_local_problem,
    pack_two_site_state,
    two_site_state_basis,
)
from .renormalized import RenormalizedBlockStack, RenormalizedOperatorStack
from .tensor import NonabelianTensor
from pyqed.mps.su2 import SU2Irrep


@lru_cache(maxsize=1)
def _numeric_allocator_relief():
    """Return the platform allocator's free-page release hook, if available."""

    try:
        if sys.platform == "darwin":
            library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            hook = library.malloc_zone_pressure_relief
            hook.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            hook.restype = ctypes.c_size_t
            return lambda: hook(None, 0)
        if sys.platform.startswith("linux"):
            library = ctypes.CDLL(None)
            hook = library.malloc_trim
            hook.argtypes = [ctypes.c_size_t]
            hook.restype = ctypes.c_int
            return lambda: hook(0)
    except (AttributeError, OSError):
        pass
    return None


def _release_free_numeric_pages():
    """Return freed large numerical arenas to the OS when supported."""

    hook = _numeric_allocator_relief()
    if hook is not None:
        hook()


_USE_REAL_RANK_COUPLED_ACCUMULATE = True
_IDENTITY_MPO_CORE_CACHE = {}
_RANK_COUPLED_REAL_TERM_COALESCE = (
    os.environ.get("PYQED_SU2_RANK_COUPLED_COALESCE_TERMS", "0")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
_RANK_COUPLED_REAL_TERM_COALESCE_STATS = {
    "calls": 0,
    "input_terms": 0,
    "output_terms": 0,
    "merged_terms": 0,
}


def rank_coupled_real_term_coalesce_stats(*, reset=False):
    """Return packed rank-coupled term coalescing counters."""

    stats = {
        str(key): int(value)
        for key, value in _RANK_COUPLED_REAL_TERM_COALESCE_STATS.items()
    }
    if reset:
        for key in _RANK_COUPLED_REAL_TERM_COALESCE_STATS:
            _RANK_COUPLED_REAL_TERM_COALESCE_STATS[key] = 0
    return stats


@lru_cache(maxsize=1)
def _su2_kernel_module():
    """Return the optional compiled SU(2) helper module, if available."""

    try:
        from pyqed.mps.nonabelian import _su2_kernel as module
    except Exception:
        return None
    return module


def _real64_contiguous_or_none(array):
    """Return a real float64 buffer, or ``None`` for genuinely complex data."""

    arr = np.asarray(array)
    if arr.dtype.kind == "c":
        if arr.size:
            scale = max(1.0, float(np.max(np.abs(arr.real))))
            tolerance = 64.0 * np.finfo(np.float64).eps * scale
            if float(np.max(np.abs(arr.imag))) > tolerance:
                return None
        arr = arr.real
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _rank_coupled_real_term_arrays(reduced):
    """Pack reduced rank-coupled terms for the real Cython accumulate path."""

    if not _RANK_COUPLED_REAL_TERM_COALESCE:
        left_indices = []
        right_indices = []
        w_blocks = []
        for left_idx, right_idx, w_block in reduced:
            w_arr = _real64_contiguous_or_none(w_block)
            if w_arr is None:
                n_terms = len(left_indices) + 1
                _RANK_COUPLED_REAL_TERM_COALESCE_STATS["calls"] += 1
                _RANK_COUPLED_REAL_TERM_COALESCE_STATS["input_terms"] += int(n_terms)
                _RANK_COUPLED_REAL_TERM_COALESCE_STATS["output_terms"] += int(n_terms)
                return None
            left_indices.append(int(left_idx))
            right_indices.append(int(right_idx))
            w_blocks.append(w_arr)
        n_terms = len(w_blocks)
        _RANK_COUPLED_REAL_TERM_COALESCE_STATS["calls"] += 1
        _RANK_COUPLED_REAL_TERM_COALESCE_STATS["input_terms"] += int(n_terms)
        _RANK_COUPLED_REAL_TERM_COALESCE_STATS["output_terms"] += int(n_terms)
        return (
            np.asarray(left_indices, dtype=np.int64),
            np.asarray(right_indices, dtype=np.int64),
            tuple(w_blocks),
        )

    input_terms = 0
    grouped = {}
    order = []
    for left_idx, right_idx, w_block in reduced:
        input_terms += 1
        w_arr = _real64_contiguous_or_none(w_block)
        if w_arr is None:
            _RANK_COUPLED_REAL_TERM_COALESCE_STATS["calls"] += 1
            _RANK_COUPLED_REAL_TERM_COALESCE_STATS["input_terms"] += int(input_terms)
            _RANK_COUPLED_REAL_TERM_COALESCE_STATS["output_terms"] += int(input_terms)
            return None
        key = (
            int(left_idx),
            int(right_idx),
            tuple(int(dim) for dim in w_arr.shape),
        )
        current = grouped.get(key)
        if current is None:
            grouped[key] = w_arr
            order.append(key)
        else:
            grouped[key] = current + w_arr
    left_indices = []
    right_indices = []
    w_blocks = []
    for left_idx, right_idx, _shape in order:
        left_indices.append(int(left_idx))
        right_indices.append(int(right_idx))
        w_blocks.append(np.ascontiguousarray(grouped[(left_idx, right_idx, _shape)]))
    output_terms = len(w_blocks)
    _RANK_COUPLED_REAL_TERM_COALESCE_STATS["calls"] += 1
    _RANK_COUPLED_REAL_TERM_COALESCE_STATS["input_terms"] += int(input_terms)
    _RANK_COUPLED_REAL_TERM_COALESCE_STATS["output_terms"] += int(output_terms)
    _RANK_COUPLED_REAL_TERM_COALESCE_STATS["merged_terms"] += max(
        0,
        int(input_terms) - int(output_terms),
    )
    return (
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
        tuple(w_blocks),
    )


def _basis_cache_signature(basis):
    """
    Return a hashable signature for a local two-site basis.

    :param basis: Explicit local basis with packed entries.
    :returns: Tuple describing sector keys and block shapes.
    """

    return tuple((entry.key, entry.shape, entry.size) for entry in basis)


def _uses_channel_resolved_local_basis(two_site, W1, W2, *, rank_coupled):
    """Use intermediate-channel blocks only for genuinely fully reduced cores."""

    if (
        not rank_coupled
        or not bool(
            two_site.metadata.get(
                "contracted_channel_blocks_current",
                False,
            )
        )
    ):
        return False
    factors = (W1, W2)
    normal_complementary = tuple(
        factor
        for factor in factors
        if getattr(factor, "normal_complementary_plan", None) is not None
    )
    if normal_complementary:
        return all(
            bool(
                getattr(
                    factor,
                    "normal_complementary_fully_reduced",
                    False,
                )
            )
            for factor in normal_complementary
        )
    return all(
        bool(getattr(factor, "fully_reduced_identity", False))
        for factor in factors
    )


def _array_digest(array):
    if hasattr(array, "data") and isinstance(array.data, dict):
        array = array.data
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
_RANK_COUPLED_SMALL_CONTRACTION_WORK = int(
    os.environ.get("PYQED_SU2_RANK_COUPLED_ACCUMULATE_WORK", "1000000000")
)
_RANK_COUPLED_GREEDY_CONTRACTION_WORK = int(
    os.environ.get("PYQED_SU2_RANK_COUPLED_GREEDY_WORK", "8000000")
)


def _rank_coupled_left_work(E_block, A_conj, W_block, B_block):
    return (
        int(E_block.shape[0])
        * int(W_block.shape[1])
        * int(E_block.shape[1])
        * int(E_block.shape[2])
        * int(A_conj.shape[1])
        * int(W_block.shape[3])
        * int(A_conj.shape[2])
        * int(B_block.shape[2])
    )


def _rank_coupled_right_work(A_conj, W_block, F_block, B_block):
    return (
        int(W_block.shape[0])
        * int(F_block.shape[0])
        * int(A_conj.shape[0])
        * int(B_block.shape[0])
        * int(A_conj.shape[1])
        * int(W_block.shape[3])
        * int(A_conj.shape[2])
        * int(F_block.shape[2])
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

    if (
        E_block.shape[0] == 1
        and W_block.shape[0] == 1
        and W_block.shape[1] == 1
        and W_block.shape[2] == A_conj.shape[1]
        and W_block.shape[3] == B_block.shape[1]
    ):
        module = _su2_kernel_module()
        kernel = (
            None
            if module is None
            else getattr(module, "contract_rank_coupled_left_scalar_channel", None)
        )
        if kernel is not None:
            return kernel(E_block, A_conj, W_block, B_block)
    work = _rank_coupled_left_work(E_block, A_conj, W_block, B_block)
    if work <= _RANK_COUPLED_SMALL_CONTRACTION_WORK:
        module = _su2_kernel_module()
        kernel = (
            None
            if module is None
            else getattr(module, "contract_rank_coupled_left_general", None)
        )
        if kernel is not None:
            out = kernel(E_block, A_conj, W_block, B_block)
            if out is not None:
                return out
    optimize = "greedy" if work >= _RANK_COUPLED_GREEDY_CONTRACTION_WORK else False
    tmp = np.einsum("xij,ipr->xjpr", E_block, A_conj, optimize=optimize)
    tmp = np.einsum("xjpr,jqs->xprqs", tmp, B_block, optimize=optimize)
    return np.einsum("xprqs,xypq->yrs", tmp, W_block, optimize=optimize)


def _contract_rank_coupled_right_step(A_conj, W_block, F_block, B_block):
    """
    Contract one right environment update term without dynamic path planning.

    :param A_conj: Conjugated bra MPS block with indices ``ipr``.
    :param W_block: Reduced MPO block with indices ``xypq``.
    :param F_block: Right environment component with indices ``yrs``.
    :param B_block: Ket MPS block with indices ``jqs``.
    :returns: Contribution with indices ``xij``.
    """

    if (
        F_block.shape[0] == 1
        and W_block.shape[0] == 1
        and W_block.shape[1] == 1
        and W_block.shape[2] == A_conj.shape[1]
        and W_block.shape[3] == B_block.shape[1]
    ):
        module = _su2_kernel_module()
        kernel = (
            None
            if module is None
            else getattr(module, "contract_rank_coupled_right_scalar_channel", None)
        )
        if kernel is not None:
            return kernel(A_conj, W_block, F_block, B_block)
    work = _rank_coupled_right_work(A_conj, W_block, F_block, B_block)
    if work <= _RANK_COUPLED_SMALL_CONTRACTION_WORK:
        module = _su2_kernel_module()
        kernel = (
            None
            if module is None
            else getattr(module, "contract_rank_coupled_right_general", None)
        )
        if kernel is not None:
            out = kernel(A_conj, W_block, F_block, B_block)
            if out is not None:
                return out
    optimize = "greedy" if work >= _RANK_COUPLED_GREEDY_CONTRACTION_WORK else False
    tmp = np.einsum("yrs,jqs->yrjq", F_block, B_block, optimize=optimize)
    tmp = np.einsum("xypq,yrjq->xprj", W_block, tmp, optimize=optimize)
    return np.einsum("ipr,xprj->xij", A_conj, tmp, optimize=optimize)


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
    cache_key = (id(mpo_core), float(tol))
    cached = _IDENTITY_MPO_CORE_CACHE.get(cache_key)
    if cached is not None and cached[0]() is mpo_core:
        return bool(cached[1])
    if cached is not None:
        _IDENTITY_MPO_CORE_CACHE.pop(cache_key, None)
    if not isinstance(mpo_core, MPO):
        return False
    if mpo_core.left_dim != 1 or mpo_core.right_dim != 1:
        _IDENTITY_MPO_CORE_CACHE[cache_key] = (weakref.ref(mpo_core), False)
        return False
    for q_out in mpo_core.phys_out_sectors:
        for q_in in mpo_core.phys_in_sectors:
            block = mpo_core.block(q_out, q_in)
            if q_out == q_in:
                if block is None:
                    _IDENTITY_MPO_CORE_CACHE[cache_key] = (weakref.ref(mpo_core), False)
                    return False
                eye = np.eye(mpo_core.phys_out_leg.sector_dim(q_out), dtype=np.asarray(block).dtype)
                if not np.allclose(np.asarray(block)[0, 0], eye, atol=tol, rtol=tol):
                    _IDENTITY_MPO_CORE_CACHE[cache_key] = (weakref.ref(mpo_core), False)
                    return False
            elif block is not None and np.linalg.norm(np.asarray(block).reshape(-1)) > tol:
                _IDENTITY_MPO_CORE_CACHE[cache_key] = (weakref.ref(mpo_core), False)
                return False
    if len(_IDENTITY_MPO_CORE_CACHE) > 256:
        _IDENTITY_MPO_CORE_CACHE.clear()
    _IDENTITY_MPO_CORE_CACHE[cache_key] = (weakref.ref(mpo_core), True)
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
    initial_channel = (
        1
        if getattr(W, "normal_complementary_plan", None) is not None
        else 0
    )
    for sector, dim in site_layout["sector_dims"][0].items():
        block = np.zeros(
            (W.left_channel_irreps[initial_channel].dim, dim, dim),
            dtype=_mpo_dtype(W),
        )
        block[0] = np.eye(dim, dtype=_mpo_dtype(W))
        env[(sector, sector)] = RankCoupledChannelBlocks(
            {initial_channel: block},
            len(W.left_channel_irreps),
        )
    return env


def _initial_right_env_blocks_rank_coupled(site_layout, W):
    env = {}
    for sector, dim in site_layout["sector_dims"][2].items():
        block = np.zeros(
            (W.right_channel_irreps[0].dim, dim, dim),
            dtype=_mpo_dtype(W),
        )
        block[0] = np.eye(dim, dtype=_mpo_dtype(W))
        env[(sector, sector)] = RankCoupledChannelBlocks(
            {0: block},
            len(W.right_channel_irreps),
        )
    return env


class RankCoupledChannelBlocks:
    """Sparse channel-indexed reduced boundary blocks."""

    __slots__ = ("data", "n_channels")

    def __init__(self, data=None, n_channels=0):
        self.data = dict(data or {})
        self.n_channels = int(n_channels)

    def __len__(self):
        return self.n_channels

    def __contains__(self, channel):
        return int(channel) in self.data

    def __getitem__(self, channel):
        return self.data[int(channel)]

    def __setitem__(self, channel, block):
        self.data[int(channel)] = block

    def get(self, channel, default=None):
        return self.data.get(int(channel), default)

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def copy(self):
        return type(self)(
            {channel: np.array(block, copy=True) for channel, block in self.data.items()},
            self.n_channels,
        )


class _PackedRankCoupledEnvironmentMap:
    """Sector-pair mapping view that owns no arrays beyond a packed boundary."""

    def __init__(self, packed_table, n_channels=None):
        self.packed_table = packed_table
        sectors = tuple(packed_table.sector_codec.sectors)
        self._entries = {}
        max_channel = max(
            (int(channel) for channel in packed_table.channel_ids),
            default=-1,
        )
        self._n_channels = max(
            max_channel + 1,
            int(0 if n_channels is None else n_channels),
        )
        for row_idx, ket_id in enumerate(packed_table.ket_sector_ids):
            q_in = sectors[int(ket_id)]
            start = int(packed_table.entry_offsets[row_idx])
            stop = int(packed_table.entry_offsets[row_idx + 1])
            for entry_idx in range(start, stop):
                q_out = sectors[int(packed_table.out_sector_ids[entry_idx])]
                self._entries[(q_out, q_in)] = int(entry_idx)
        self._blocks = {}

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, key):
        return key in self._entries

    def __getitem__(self, key):
        cached = self._blocks.get(key)
        if cached is not None:
            return cached
        entry_idx = self._entries[key]
        table = self.packed_table
        start = int(table.channel_offsets[entry_idx])
        stop = int(table.channel_offsets[entry_idx + 1])
        cached = RankCoupledChannelBlocks(
            {
                int(table.channel_ids[channel_idx]):
                table.block_pool.array(channel_idx)
                for channel_idx in range(start, stop)
            },
            self._n_channels,
        )
        self._blocks[key] = cached
        return cached

    def get(self, key, default=None):
        return self[key] if key in self._entries else default

    def items(self):
        return ((key, self[key]) for key in self._entries)

    def keys(self):
        return self._entries.keys()

    def values(self):
        return (self[key] for key in self._entries)


def _rank_coupled_channel_items(blocks):
    if isinstance(blocks, RankCoupledChannelBlocks):
        return blocks.items()
    return enumerate(tuple(blocks or ()))


def _copy_rank_coupled_env_map(env_map):
    return {
        key: (
            blocks.copy()
            if isinstance(blocks, RankCoupledChannelBlocks)
            else tuple(np.array(block, copy=True) for block in blocks)
        )
        for key, blocks in env_map.items()
    }


def _nonzero_rank_coupled_blocks(blocks, *, tol=0.0):
    out = []
    for idx, block in _rank_coupled_channel_items(blocks):
        arr = np.asarray(block)
        if arr.size and np.any(np.abs(arr) > tol):
            out.append((idx, arr))
    return tuple(out)


def _flatten_rank_coupled_env_map(env_map):
    flat = {}
    for key, blocks in env_map.items():
        flat[key] = np.concatenate(
            [np.asarray(block) for _channel, block in _rank_coupled_channel_items(blocks)],
            axis=0,
        )
    return flat


def _sector_irrep(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None:
        return irrep
    if hasattr(sector, "labels") and "su2" in sector.labels:
        return sector.components[sector.labels.index("su2")]
    raise TypeError(f"Sector {sector!r} does not carry an SU(2) irrep.")


def _rank_coupled_degeneracy_only_physical(W):
    if not isinstance(W, RankCoupledMPO):
        return False
    return all(W.phys_out_leg.sector_dim(sector) == 1 for sector in W.phys_out_leg.sectors) and all(
        W.phys_in_leg.sector_dim(sector) == 1 for sector in W.phys_in_leg.sectors
    )


def _reduced_operator_rank(operator):
    base = getattr(operator, "base_operator", operator)
    rank = getattr(base, "rank_irrep", None)
    if not isinstance(rank, SU2Irrep):
        raise TypeError("Reduced tensor operator does not expose a rank_irrep.")
    return rank


def _rank_coupled_component(left_irrep, right_irrep, two_m_left, two_m_right):
    if left_irrep.two_j == 0 and right_irrep.two_j != 0:
        return int(two_m_right)
    if right_irrep.two_j == 0 and left_irrep.two_j != 0:
        return int(two_m_left)
    return int(two_m_right - two_m_left)


def _virtual_rank_coupling(left_irrep, op_rank, right_irrep, two_m_left, two_m_component, two_m_right):
    return clebsch_gordan(
        left_irrep,
        op_rank,
        right_irrep,
        int(two_m_left),
        int(two_m_component),
        int(two_m_right),
    )


def _virtual_dual_metric(left_irrep, op_rank, right_irrep):
    if right_irrep.two_j == 0 and left_irrep.two_j == op_rank.two_j and op_rank.two_j != 0:
        return -float(op_rank.dim)
    return 1.0


_FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY = "__fully_reduced_one_body_split__"


def _rank_coupled_core_has_family(W, family):
    family = str(family)
    for transition in getattr(W, "symbolic_transitions", ()) or ():
        if len(transition) < 4:
            continue
        label = transition[3]
        if (
            isinstance(label, tuple)
            and label
            and isinstance(label[-1], tuple)
            and family in label[-1]
        ):
            return True
    return False


@lru_cache(maxsize=None)
def _component_basis_matrix(q_out, q_in, rank_irrep, two_m_component):
    out_irrep = _sector_irrep(q_out)
    in_irrep = _sector_irrep(q_in)
    block = np.zeros((out_irrep.dim, in_irrep.dim), dtype=float)
    for row, two_m_out in enumerate(ordered_two_m_values(out_irrep)):
        for col, two_m_in in enumerate(ordered_two_m_values(in_irrep)):
            coeff = clebsch_gordan(
                in_irrep,
                rank_irrep,
                out_irrep,
                two_m_in,
                int(two_m_component),
                two_m_out,
            )
            if coeff:
                block[row, col] = coeff
    return block


@lru_cache(maxsize=None)
def _component_basis_norm(q_out, q_in, rank_irrep, two_m_component):
    """Return the reduced-component connection weight at an SU(2) cut."""

    block = _component_basis_matrix(
        q_out,
        q_in,
        rank_irrep,
        int(two_m_component),
    )
    return float(np.vdot(block, block).real)


@lru_cache(maxsize=None)
def _dual_component_basis_matrix(q_out, q_in, rank_irrep, two_m_component):
    phase = (-1.0) ** ((rank_irrep.two_j - int(two_m_component)) // 2)
    return phase * _component_basis_matrix(q_out, q_in, rank_irrep, -int(two_m_component))


@lru_cache(maxsize=None)
def _physical_component_matrix(q_out, q_in, rank_irrep, two_m_component, diagonal_scalar):
    out_irrep = _sector_irrep(q_out)
    in_irrep = _sector_irrep(q_in)
    if diagonal_scalar:
        if out_irrep != in_irrep:
            return np.zeros((out_irrep.dim, in_irrep.dim), dtype=float)
        return np.eye(out_irrep.dim, in_irrep.dim, dtype=float)
    out_charge = getattr(q_out, "charge", None)
    in_charge = getattr(q_in, "charge", None)
    scale = np.sqrt(float(rank_irrep.dim))
    if out_charge is not None and in_charge is not None and int(out_charge) > int(in_charge):
        return scale * _component_basis_matrix(q_in, q_out, rank_irrep, int(two_m_component)).T
    return scale * _component_basis_matrix(q_out, q_in, rank_irrep, int(two_m_component))


def _rank_coupled_reduced_terms_block(W, phys_out, phys_in):
    reduced = {}
    dtype = _mpo_dtype(W)
    for operator, i, j, row, col, component, coeff in W._reduced_actions():
        op_block = operator.component_block(component, phys_out, phys_in)
        if op_block is None:
            continue
        local = reduced.get((i, j))
        if local is None:
            local = np.zeros(
                (
                    W.left_channel_irreps[i].dim,
                    W.right_channel_irreps[j].dim,
                    W.phys_out_leg.sector_dim(phys_out),
                    W.phys_in_leg.sector_dim(phys_in),
                ),
                dtype=dtype,
            )
            reduced[(i, j)] = local
        local[row, col] += np.asarray(coeff, dtype=dtype) * np.asarray(
            op_block,
            dtype=dtype,
        )
    return reduced


@lru_cache(maxsize=None)
def _left_reduced_recoupling_coeff(
    q_lb,
    q_lk,
    q_pb,
    q_pk,
    q_rb,
    q_rk,
    left_irrep,
    right_irrep,
    op_rank,
    two_m_left,
    two_m_right,
    two_m_component,
    dual_right_basis=True,
):
    """
    Project one reduced left-environment update through hidden spin components.
    """
    left_basis = _component_basis_matrix(q_lb, q_lk, left_irrep, int(two_m_left))
    local_basis = _physical_component_matrix(
        q_pb,
        q_pk,
        op_rank,
        int(two_m_component),
        op_rank.two_j == 0,
    )
    right_basis = (
        _dual_component_basis_matrix(q_rb, q_rk, right_irrep, int(two_m_right))
        if dual_right_basis
        else _component_basis_matrix(q_rb, q_rk, right_irrep, int(two_m_right))
    )
    if not np.any(left_basis) or not np.any(local_basis) or not np.any(right_basis):
        return 0.0

    q_lb_irrep = _sector_irrep(q_lb)
    q_lk_irrep = _sector_irrep(q_lk)
    q_pb_irrep = _sector_irrep(q_pb)
    q_pk_irrep = _sector_irrep(q_pk)
    q_rb_irrep = _sector_irrep(q_rb)
    q_rk_irrep = _sector_irrep(q_rk)
    projected = np.zeros((q_rb_irrep.dim, q_rk_irrep.dim), dtype=float)

    for lb_row, two_m_lb in enumerate(ordered_two_m_values(q_lb_irrep)):
        for lk_col, two_m_lk in enumerate(ordered_two_m_values(q_lk_irrep)):
            env_coeff = left_basis[lb_row, lk_col]
            if env_coeff == 0:
                continue
            for pb_row, two_m_pb in enumerate(ordered_two_m_values(q_pb_irrep)):
                for pk_col, two_m_pk in enumerate(ordered_two_m_values(q_pk_irrep)):
                    op_coeff = local_basis[pb_row, pk_col]
                    if op_coeff == 0:
                        continue
                    for rb_row, two_m_rb in enumerate(ordered_two_m_values(q_rb_irrep)):
                        bra_cg = clebsch_gordan(
                            q_lb_irrep,
                            q_pb_irrep,
                            q_rb_irrep,
                            two_m_lb,
                            two_m_pb,
                            two_m_rb,
                        )
                        if bra_cg == 0:
                            continue
                        for rk_col, two_m_rk in enumerate(ordered_two_m_values(q_rk_irrep)):
                            ket_cg = clebsch_gordan(
                                q_lk_irrep,
                                q_pk_irrep,
                                q_rk_irrep,
                                two_m_lk,
                                two_m_pk,
                                two_m_rk,
                            )
                            if ket_cg:
                                projected[rb_row, rk_col] += (
                                    env_coeff * op_coeff * bra_cg * ket_cg
                                )

    norm = np.vdot(right_basis, right_basis)
    if abs(norm) <= 1.0e-14:
        return 0.0
    return np.vdot(right_basis, projected) / norm


@lru_cache(maxsize=None)
def _right_reduced_recoupling_coeff(
    q_lb,
    q_lk,
    q_pb,
    q_pk,
    q_rb,
    q_rk,
    left_irrep,
    right_irrep,
    op_rank,
    two_m_left,
    two_m_right,
    two_m_component,
    dual_right_basis=False,
):
    """Right-environment analogue of :func:`_left_reduced_recoupling_coeff`."""
    left_basis = _component_basis_matrix(q_lb, q_lk, left_irrep, int(two_m_left))
    local_basis = _physical_component_matrix(
        q_pb,
        q_pk,
        op_rank,
        int(two_m_component),
        op_rank.two_j == 0,
    )
    right_basis = (
        _dual_component_basis_matrix(
            q_rb,
            q_rk,
            right_irrep,
            int(two_m_right),
        )
        if dual_right_basis
        else _component_basis_matrix(
            q_rb,
            q_rk,
            right_irrep,
            int(two_m_right),
        )
    )
    if not np.any(left_basis) or not np.any(local_basis) or not np.any(right_basis):
        return 0.0

    q_lb_irrep = _sector_irrep(q_lb)
    q_lk_irrep = _sector_irrep(q_lk)
    q_pb_irrep = _sector_irrep(q_pb)
    q_pk_irrep = _sector_irrep(q_pk)
    q_rb_irrep = _sector_irrep(q_rb)
    q_rk_irrep = _sector_irrep(q_rk)
    projected = np.zeros((q_lb_irrep.dim, q_lk_irrep.dim), dtype=float)

    for rb_row, two_m_rb in enumerate(ordered_two_m_values(q_rb_irrep)):
        for rk_col, two_m_rk in enumerate(ordered_two_m_values(q_rk_irrep)):
            env_coeff = right_basis[rb_row, rk_col]
            if env_coeff == 0:
                continue
            for pb_row, two_m_pb in enumerate(ordered_two_m_values(q_pb_irrep)):
                for pk_col, two_m_pk in enumerate(ordered_two_m_values(q_pk_irrep)):
                    op_coeff = local_basis[pb_row, pk_col]
                    if op_coeff == 0:
                        continue
                    for lb_row, two_m_lb in enumerate(ordered_two_m_values(q_lb_irrep)):
                        bra_cg = clebsch_gordan(
                            q_lb_irrep,
                            q_pb_irrep,
                            q_rb_irrep,
                            two_m_lb,
                            two_m_pb,
                            two_m_rb,
                        )
                        if bra_cg == 0:
                            continue
                        for lk_col, two_m_lk in enumerate(ordered_two_m_values(q_lk_irrep)):
                            ket_cg = clebsch_gordan(
                                q_lk_irrep,
                                q_pk_irrep,
                                q_rk_irrep,
                                two_m_lk,
                                two_m_pk,
                                two_m_rk,
                            )
                            if ket_cg:
                                projected[lb_row, lk_col] += (
                                    env_coeff * op_coeff * bra_cg * ket_cg
                                )

    norm = np.vdot(left_basis, left_basis)
    if abs(norm) <= 1.0e-14:
        return 0.0
    return np.vdot(left_basis, projected) / norm


def _left_reduced_rank_coupled_block(W, q_lb, q_lk, q_pb, q_pk, q_rb, q_rk):
    cache_key = (
        "left",
        q_lb,
        q_lk,
        q_pb,
        q_pk,
        q_rb,
        q_rk,
        bool(getattr(W, "normal_complementary_right_dual", False)),
        id(getattr(W, "normal_complementary_plan", None)),
    )
    cache = W._environment_reduced_block_cache
    if cache_key in cache:
        return cache[cache_key]
    reduced = {}
    dense_block = W.dense_blocks.get((q_pb, q_pk))
    dtype = _mpo_dtype(W)
    relaxed_scalar_transfer = _rank_coupled_core_has_family(
        W,
        _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY,
    )
    if dense_block is not None:
        for left_idx, right_idx, local_payload in iter_virtual_routes(dense_block):
                left_irrep = W.left_channel_irreps[left_idx]
                right_irrep = W.right_channel_irreps[right_idx]
                local_scalar = np.asarray(local_payload[0, 0], dtype=dtype)
                block = np.zeros((left_irrep.dim, right_irrep.dim, 1, 1), dtype=dtype)
                for row, two_m_left in enumerate(ordered_two_m_values(left_irrep)):
                    for col, two_m_right in enumerate(ordered_two_m_values(right_irrep)):
                        component = _rank_coupled_component(
                            left_irrep,
                            right_irrep,
                            int(two_m_left),
                            int(two_m_right),
                        )
                        if relaxed_scalar_transfer:
                            virtual_cg = 1.0
                        else:
                            virtual_cg = _virtual_rank_coupling(
                                left_irrep,
                                SU2Irrep(0),
                                right_irrep,
                                int(two_m_left),
                                component,
                                int(two_m_right),
                            )
                            if virtual_cg == 0:
                                continue
                        coeff = (
                            _left_reduced_recoupling_coeff(
                                q_lb,
                                q_lk,
                                q_pb,
                                q_pk,
                                q_rb,
                                q_rk,
                                left_irrep,
                                right_irrep,
                                SU2Irrep(0),
                                int(two_m_left),
                                int(two_m_right),
                                0,
                                False,
                            )
                            if relaxed_scalar_transfer
                            else _left_reduced_recoupling_coeff(
                                q_lb,
                                q_lk,
                                q_pb,
                                q_pk,
                                q_rb,
                                q_rk,
                                left_irrep,
                                right_irrep,
                                SU2Irrep(0),
                                int(two_m_left),
                                int(two_m_right),
                                0,
                                bool(
                                    getattr(
                                        W,
                                        "normal_complementary_right_dual",
                                        False,
                                    )
                                ),
                            )
                        )
                        if coeff:
                            block[row, col, 0, 0] += local_scalar * virtual_cg * coeff
                if np.any(block):
                    reduced[(left_idx, right_idx)] = block

    if relaxed_scalar_transfer:
        for key, block in _rank_coupled_reduced_terms_block(W, q_pb, q_pk).items():
            reduced[key] = reduced.get(key, 0) + block
        cache[cache_key] = reduced
        return reduced

    for term in W.reduced_terms:
        op_rank = _reduced_operator_rank(term.reduced_operator)
        for left_idx, right_idx, visible_coeff in iter_virtual_routes(
            term.visible_virtual_block
        ):
                left_irrep = W.left_channel_irreps[left_idx]
                right_irrep = W.right_channel_irreps[right_idx]
                block = reduced.get(
                    (left_idx, right_idx),
                    np.zeros((left_irrep.dim, right_irrep.dim, 1, 1), dtype=dtype),
                )
                for row, two_m_left in enumerate(ordered_two_m_values(left_irrep)):
                    for col, two_m_right in enumerate(ordered_two_m_values(right_irrep)):
                        if getattr(term, "use_cg_coupling", False):
                            oriented_left = (
                                int(term.left_component_orientation)
                                * int(two_m_left)
                            )
                            oriented_right = (
                                int(term.right_component_orientation)
                                * int(two_m_right)
                            )
                            component = oriented_right - oriented_left
                            if op_rank.two_j == 0:
                                component = 0
                            if term.dual_right_coupling:
                                component = -(
                                    int(two_m_left)
                                    + int(two_m_right)
                                )
                                if (
                                    getattr(
                                        W,
                                        "normal_complementary_plan",
                                        None,
                                    )
                                    is not None
                                    and left_irrep.two_j == 2
                                    and right_irrep.two_j == 1
                                    and abs(
                                        int(W.left_channel_charges[left_idx])
                                    )
                                    == 2
                                ):
                                    component *= int(
                                        term.left_component_orientation
                                    )
                            if term.orient_virtual_coupling:
                                coupling_left = oriented_left
                                coupling_right = oriented_right
                                coupling_component = component
                            else:
                                coupling_left = int(two_m_left)
                                coupling_right = int(two_m_right)
                                coupling_component = (
                                    coupling_right - coupling_left
                                )
                        else:
                            component = _rank_coupled_component(
                                left_irrep,
                                right_irrep,
                                int(two_m_left),
                                int(two_m_right),
                            )
                            oriented_left = int(two_m_left)
                            oriented_right = int(two_m_right)
                            coupling_left = oriented_left
                            coupling_right = oriented_right
                            coupling_component = component
                        virtual_cg = _virtual_rank_coupling(
                            left_irrep,
                            op_rank,
                            right_irrep,
                            coupling_left,
                            coupling_component,
                            coupling_right,
                        )
                        if term.dual_right_coupling:
                            virtual_cg = (
                                (-1.0)
                                ** (
                                    (
                                        right_irrep.two_j
                                        + int(two_m_right)
                                    )
                                    // 2
                                )
                                * _virtual_rank_coupling(
                                    left_irrep,
                                    op_rank,
                                    right_irrep,
                                    int(two_m_left),
                                    -(
                                        int(two_m_left)
                                        + int(two_m_right)
                                    ),
                                    -int(two_m_right),
                                )
                            )
                        if term.phase_from_charged_scalar_source:
                            virtual_cg *= -int(two_m_right)
                        if term.phase_to_charged_pair_target:
                            virtual_cg *= -int(two_m_left)
                        if (
                            getattr(W, "normal_complementary_plan", None)
                            is not None
                            and op_rank.two_j == 0
                            and left_irrep.two_j == right_irrep.two_j
                            and left_irrep.two_j != 0
                            and getattr(
                                term.reduced_operator,
                                "operator_id",
                                -1,
                            )
                            in (0, 12, 18)
                        ):
                            virtual_cg = (
                                (-1.0)
                                ** (
                                    (
                                        left_irrep.two_j
                                        + int(two_m_left)
                                    )
                                    // 2
                                )
                                if int(two_m_right)
                                == -int(two_m_left)
                                else 0.0
                            )
                            if (
                                getattr(
                                    term.reduced_operator,
                                    "operator_id",
                                    -1,
                                )
                                == 0
                                and abs(
                                    int(W.left_channel_charges[left_idx])
                                )
                                != 2
                            ):
                                virtual_cg *= (
                                    term.left_component_orientation
                                )
                        if virtual_cg == 0:
                            continue
                        dual_factor = _virtual_dual_metric(
                            left_irrep,
                            op_rank,
                            right_irrep,
                        )
                        if (
                            getattr(W, "normal_complementary_plan", None)
                            is not None
                            and op_rank.two_j == 1
                            and right_irrep.two_j == 0
                            and left_irrep.two_j == 1
                        ):
                            dual_factor = -1.0
                        op_block = term.reduced_operator.component_block(component, q_pb, q_pk)
                        if op_block is None:
                            continue
                        recoupled = _left_reduced_recoupling_coeff(
                            q_lb,
                            q_lk,
                            q_pb,
                            q_pk,
                            q_rb,
                            q_rk,
                            left_irrep,
                            right_irrep,
                            op_rank,
                            int(two_m_left),
                            int(two_m_right),
                            component,
                        )
                        if recoupled:
                            block[row, col, 0, 0] += (
                                np.asarray(visible_coeff, dtype=dtype)
                                * np.asarray(virtual_cg, dtype=dtype)
                                * np.asarray(op_block[0, 0], dtype=dtype)
                                * np.asarray(dual_factor, dtype=dtype)
                                * recoupled
                            )
                if np.any(block):
                    reduced[(left_idx, right_idx)] = block
    cache[cache_key] = reduced
    return reduced


def _right_reduced_rank_coupled_block(W, q_lb, q_lk, q_pb, q_pk, q_rb, q_rk):
    cache_key = (
        "right",
        q_lb,
        q_lk,
        q_pb,
        q_pk,
        q_rb,
        q_rk,
        bool(getattr(W, "normal_complementary_right_dual", False)),
        id(getattr(W, "normal_complementary_plan", None)),
    )
    cache = W._environment_reduced_block_cache
    if cache_key in cache:
        return cache[cache_key]
    reduced = {}
    dense_block = W.dense_blocks.get((q_pb, q_pk))
    dtype = _mpo_dtype(W)
    relaxed_scalar_transfer = _rank_coupled_core_has_family(
        W,
        _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY,
    )
    if dense_block is not None:
        for left_idx, right_idx, local_payload in iter_virtual_routes(dense_block):
                left_irrep = W.left_channel_irreps[left_idx]
                right_irrep = W.right_channel_irreps[right_idx]
                local_scalar = np.asarray(local_payload[0, 0], dtype=dtype)
                block = np.zeros((left_irrep.dim, right_irrep.dim, 1, 1), dtype=dtype)
                for row, two_m_left in enumerate(ordered_two_m_values(left_irrep)):
                    for col, two_m_right in enumerate(ordered_two_m_values(right_irrep)):
                        component = _rank_coupled_component(
                            left_irrep,
                            right_irrep,
                            int(two_m_left),
                            int(two_m_right),
                        )
                        if relaxed_scalar_transfer:
                            virtual_cg = 1.0
                        else:
                            virtual_cg = _virtual_rank_coupling(
                                left_irrep,
                                SU2Irrep(0),
                                right_irrep,
                                int(two_m_left),
                                component,
                                int(two_m_right),
                            )
                            if virtual_cg == 0:
                                continue
                        coeff = (
                            _right_reduced_recoupling_coeff(
                                q_lb,
                                q_lk,
                                q_pb,
                                q_pk,
                                q_rb,
                                q_rk,
                                left_irrep,
                                right_irrep,
                                SU2Irrep(0),
                                int(two_m_left),
                                int(two_m_right),
                                0,
                                bool(
                                    getattr(
                                        W,
                                        "normal_complementary_right_dual",
                                        False,
                                    )
                                ),
                            )
                            if relaxed_scalar_transfer
                            else _right_reduced_recoupling_coeff(
                                q_lb,
                                q_lk,
                                q_pb,
                                q_pk,
                                q_rb,
                                q_rk,
                                left_irrep,
                                right_irrep,
                                SU2Irrep(0),
                                int(two_m_left),
                                int(two_m_right),
                                0,
                            )
                        )
                        if coeff:
                            block[row, col, 0, 0] += local_scalar * virtual_cg * coeff
                if np.any(block):
                    reduced[(left_idx, right_idx)] = block

    if relaxed_scalar_transfer:
        for key, block in _rank_coupled_reduced_terms_block(W, q_pb, q_pk).items():
            reduced[key] = reduced.get(key, 0) + block
        cache[cache_key] = reduced
        return reduced

    for term in W.reduced_terms:
        op_rank = _reduced_operator_rank(term.reduced_operator)
        for left_idx, right_idx, visible_coeff in iter_virtual_routes(
            term.visible_virtual_block
        ):
                left_irrep = W.left_channel_irreps[left_idx]
                right_irrep = W.right_channel_irreps[right_idx]
                block = reduced.get(
                    (left_idx, right_idx),
                    np.zeros((left_irrep.dim, right_irrep.dim, 1, 1), dtype=dtype),
                )
                for row, two_m_left in enumerate(ordered_two_m_values(left_irrep)):
                    for col, two_m_right in enumerate(ordered_two_m_values(right_irrep)):
                        if getattr(term, "use_cg_coupling", False):
                            oriented_left = (
                                int(term.left_component_orientation)
                                * int(two_m_left)
                            )
                            oriented_right = (
                                int(term.right_component_orientation)
                                * int(two_m_right)
                            )
                            component = oriented_right - oriented_left
                            if op_rank.two_j == 0:
                                component = 0
                            if term.dual_right_coupling:
                                component = -(
                                    int(two_m_left)
                                    + int(two_m_right)
                                )
                                if (
                                    getattr(
                                        W,
                                        "normal_complementary_plan",
                                        None,
                                    )
                                    is not None
                                    and left_irrep.two_j == 2
                                    and right_irrep.two_j == 1
                                    and abs(
                                        int(W.left_channel_charges[left_idx])
                                    )
                                    == 2
                                ):
                                    component *= int(
                                        term.left_component_orientation
                                    )
                            if term.orient_virtual_coupling:
                                coupling_left = oriented_left
                                coupling_right = oriented_right
                                coupling_component = component
                            else:
                                coupling_left = int(two_m_left)
                                coupling_right = int(two_m_right)
                                coupling_component = (
                                    coupling_right - coupling_left
                                )
                        else:
                            component = _rank_coupled_component(
                                left_irrep,
                                right_irrep,
                                int(two_m_left),
                                int(two_m_right),
                            )
                            oriented_left = int(two_m_left)
                            oriented_right = int(two_m_right)
                            coupling_left = oriented_left
                            coupling_right = oriented_right
                            coupling_component = component
                        virtual_cg = _virtual_rank_coupling(
                            left_irrep,
                            op_rank,
                            right_irrep,
                            coupling_left,
                            coupling_component,
                            coupling_right,
                        )
                        if term.dual_right_coupling:
                            virtual_cg = (
                                (-1.0)
                                ** (
                                    (
                                        right_irrep.two_j
                                        + int(two_m_right)
                                    )
                                    // 2
                                )
                                * _virtual_rank_coupling(
                                    left_irrep,
                                    op_rank,
                                    right_irrep,
                                    int(two_m_left),
                                    -(
                                        int(two_m_left)
                                        + int(two_m_right)
                                    ),
                                    -int(two_m_right),
                                )
                            )
                        if term.phase_from_charged_scalar_source:
                            virtual_cg *= -int(two_m_right)
                        if term.phase_to_charged_pair_target:
                            virtual_cg *= -int(two_m_left)
                        if (
                            getattr(W, "normal_complementary_plan", None)
                            is not None
                            and op_rank.two_j == 0
                            and left_irrep.two_j == right_irrep.two_j
                            and left_irrep.two_j != 0
                            and getattr(
                                term.reduced_operator,
                                "operator_id",
                                -1,
                            )
                            in (0, 12, 18)
                        ):
                            virtual_cg = (
                                (-1.0)
                                ** (
                                    (
                                        left_irrep.two_j
                                        + int(two_m_left)
                                    )
                                    // 2
                                )
                                if int(two_m_right)
                                == -int(two_m_left)
                                else 0.0
                            )
                            if (
                                getattr(
                                    term.reduced_operator,
                                    "operator_id",
                                    -1,
                                )
                                == 0
                                and abs(
                                    int(W.left_channel_charges[left_idx])
                                )
                                != 2
                            ):
                                virtual_cg *= (
                                    term.left_component_orientation
                                )
                        if virtual_cg == 0:
                            continue
                        dual_factor = _virtual_dual_metric(
                            left_irrep,
                            op_rank,
                            right_irrep,
                        )
                        if (
                            getattr(W, "normal_complementary_plan", None)
                            is not None
                            and op_rank.two_j == 1
                            and right_irrep.two_j == 0
                            and left_irrep.two_j == 1
                        ):
                            dual_factor = -1.0
                        op_block = term.reduced_operator.component_block(component, q_pb, q_pk)
                        if op_block is None:
                            continue
                        recoupled = _right_reduced_recoupling_coeff(
                            q_lb,
                            q_lk,
                            q_pb,
                            q_pk,
                            q_rb,
                            q_rk,
                            left_irrep,
                            right_irrep,
                            op_rank,
                            int(two_m_left),
                            int(two_m_right),
                            component,
                            bool(
                                getattr(
                                    W,
                                    "normal_complementary_right_dual",
                                    False,
                                )
                            ),
                        )
                        if recoupled:
                            block[row, col, 0, 0] += (
                                np.asarray(visible_coeff, dtype=dtype)
                                * np.asarray(virtual_cg, dtype=dtype)
                                * np.asarray(op_block[0, 0], dtype=dtype)
                                * np.asarray(dual_factor, dtype=dtype)
                                * recoupled
                            )
                if np.any(block):
                    reduced[(left_idx, right_idx)] = block
    cache[cache_key] = reduced
    return reduced


def _environment_map_expectation(env_map, *, rank_coupled):
    value = 0.0 + 0.0j
    if rank_coupled:
        for blocks in env_map.values():
            for _channel, block in _rank_coupled_channel_items(blocks):
                value += np.trace(np.asarray(block).sum(axis=0))
    else:
        for block in env_map.values():
            value += np.trace(np.asarray(block)[0])
    return value


def _rank_coupled_channel_expectation(env_map, channel):
    """Trace one scalar reduced channel from a completed NC boundary."""

    value = 0.0 + 0.0j
    channel = int(channel)
    for blocks in env_map.values():
        for current, block in _rank_coupled_channel_items(blocks):
            if int(current) == channel:
                value += np.trace(np.asarray(block).sum(axis=0))
    return value


def _rank_coupled_cut_expectation(left, right, channel_irreps):
    """Contract independently reduced SU(2) boundaries across one MPO cut."""

    channel_irreps = tuple(channel_irreps)
    value = 0.0 + 0.0j
    for (q_out, q_in), left_blocks in left.items():
        right_blocks = right.get((q_out, q_in))
        if right_blocks is None:
            continue
        for channel, left_block in _rank_coupled_channel_items(left_blocks):
            channel = int(channel)
            right_block = right_blocks.get(channel)
            if right_block is None:
                continue
            irrep = channel_irreps[channel]
            weights = np.asarray(
                [
                    _component_basis_norm(
                        q_out,
                        q_in,
                        irrep,
                        two_m,
                    )
                    for two_m in ordered_two_m_values(irrep)
                ],
                dtype=float,
            )
            left_array = np.asarray(left_block)
            right_array = np.asarray(right_block)
            if (
                left_array.shape != right_array.shape
                or left_array.ndim < 1
                or left_array.shape[0] != weights.size
            ):
                raise ValueError(
                    "Rank-coupled cut blocks do not match their channel irrep."
                )
            value += np.sum(
                weights.reshape((-1,) + (1,) * (left_array.ndim - 1))
                * left_array
                * right_array
            )
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
    packed_table: object | None = field(default=None, compare=False, repr=False)
    cpp_owned_boundary: bool = field(default=False, compare=False, repr=False)
    cpp_topology_revision: int = field(default=0, compare=False, repr=False)
    cpp_numeric_revision: int = field(default=0, compare=False, repr=False)

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

    def ensure_packed(self, *, side, bond):
        """Return the persistent packed reduced-sector boundary table."""

        if not self.rank_coupled:
            return None
        packed = self.packed_table
        if (
            packed is not None
            and getattr(packed, "side", None) == str(side)
            and int(getattr(packed, "bond", -1)) == int(bond)
        ):
            return packed
        from .su2_qchem_plan import pack_rank_coupled_boundary_table_from_block_map

        n_channels = max(
            (
                int(getattr(blocks, "n_channels", len(blocks)))
                for blocks in self.data.values()
            ),
            default=0,
        )
        packed = pack_rank_coupled_boundary_table_from_block_map(
            self,
            side=side,
            bond=bond,
            representation="rank_coupled_by_ket",
        )
        object.__setattr__(self, "packed_table", packed)
        if packed is not None:
            object.__setattr__(
                self,
                "data",
                _PackedRankCoupledEnvironmentMap(packed, n_channels=n_channels),
            )
        return packed

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

    def advance(
        self,
        W,
        bra_site,
        ket_site,
        *,
        phys_slices=None,
        moving_environment=None,
        parent_bond=None,
        child_bond=None,
        numeric_revision=None,
    ):
        """
        Absorb one site into this left block.

        :param W: MPO core for the absorbed site.
        :param bra_site: Bra-side site tensor.
        :param ket_site: Ket-side site tensor.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Advanced ``LeftBlock``.
        """

        if self.rank_coupled:
            cpp_result = None
            if moving_environment is not None:
                cpp_result = _contract_rank_coupled_boundary_cpp(
                    W,
                    bra_site,
                    self,
                    ket_site,
                    moving_environment=moving_environment,
                    side="left",
                    parent_bond=parent_bond,
                    child_bond=child_bond,
                    numeric_revision=numeric_revision,
                )
            if cpp_result is None:
                data = _contract_from_left_blocks_rank_coupled(
                    W,
                    bra_site,
                    self,
                    ket_site,
                )
                return LeftBlock(data, rank_coupled=True)
            data, packed, topology_revision, numeric_revision = cpp_result
            return LeftBlock(
                data,
                rank_coupled=True,
                packed_table=packed,
                cpp_owned_boundary=True,
                cpp_topology_revision=int(topology_revision),
                cpp_numeric_revision=int(numeric_revision),
            )
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

    def advance(
        self,
        W,
        bra_site,
        ket_site,
        *,
        phys_slices=None,
        moving_environment=None,
        parent_bond=None,
        child_bond=None,
        numeric_revision=None,
    ):
        """
        Absorb one site into this right block.

        :param W: MPO core for the absorbed site.
        :param bra_site: Bra-side site tensor.
        :param ket_site: Ket-side site tensor.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Advanced ``RightBlock``.
        """

        if self.rank_coupled:
            cpp_result = None
            if moving_environment is not None:
                cpp_result = _contract_rank_coupled_boundary_cpp(
                    W,
                    bra_site,
                    self,
                    ket_site,
                    moving_environment=moving_environment,
                    side="right",
                    parent_bond=parent_bond,
                    child_bond=child_bond,
                    numeric_revision=numeric_revision,
                )
            if cpp_result is None:
                data = _contract_from_right_blocks_rank_coupled(
                    W,
                    bra_site,
                    self,
                    ket_site,
                )
                return RightBlock(data, rank_coupled=True)
            data, packed, topology_revision, numeric_revision = cpp_result
            return RightBlock(
                data,
                rank_coupled=True,
                packed_table=packed,
                cpp_owned_boundary=True,
                cpp_topology_revision=int(topology_revision),
                cpp_numeric_revision=int(numeric_revision),
            )
        else:
            if phys_slices is None:
                raise ValueError("RightBlock.advance requires phys_slices for non-rank-coupled blocks.")
            data = _contract_from_right_blocks(W, bra_site, self, ket_site, phys_slices)
        return RightBlock(data, rank_coupled=self.rank_coupled)


def _contract_from_left_blocks(W, A, E_map, B, phys_slices):
    if _is_identity_mpo_core(W):
        return _contract_from_left_identity_blocks(A, E_map, B)
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


def _contract_from_left_identity_blocks(A, E_map, B):
    out = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        A_conj = np.asarray(A_block).conj()
        for (q_lk, q_pk, q_rk), B_block in B.data.items():
            if q_pb != q_pk:
                continue
            E_block = E_map.get((q_lb, q_lk))
            if E_block is None:
                continue
            E_arr = np.asarray(E_block)
            B_arr = np.asarray(B_block)
            if E_arr.shape[0] != 1:
                contrib = _contract_rank_coupled_left_step(
                    E_arr,
                    A_conj,
                    np.eye(A_conj.shape[1], dtype=np.result_type(E_arr, A_conj, B_arr))[
                        None,
                        None,
                    ],
                    B_arr,
                )
            else:
                block = np.einsum(
                    "ij,ipr,jps->rs",
                    E_arr[0],
                    A_conj,
                    B_arr,
                    optimize=False,
                )
                contrib = block.reshape(1, int(block.shape[0]), int(block.shape[1]))
            key = (q_rb, q_rk)
            if key in out:
                out[key] = out[key] + contrib
            else:
                out[key] = contrib
    return out


def _rank_coupled_site_entries_by_edge(tensor, edge):
    """
    Return site blocks grouped by left or right virtual sector.

    Nonabelian site tensors are replaced, not reshaped in place, during sweeps.
    Caching this grouping on the tensor avoids rebuilding the same small Python
    dictionaries when multiple rank-coupled environments absorb the same site.
    The entries keep live array references; conjugation is still done at use.
    """

    if edge not in ("left", "right"):
        raise ValueError(f"Unknown rank-coupled site grouping edge {edge!r}.")
    metadata = getattr(tensor, "metadata", None)
    cache_key = f"_rank_coupled_site_entries_by_{edge}"
    data_id_key = f"{cache_key}_data_id"
    if isinstance(metadata, dict) and metadata.get(data_id_key) == id(tensor.data):
        cached = metadata.get(cache_key)
        if cached is not None:
            return cached

    grouped = {}
    if edge == "left":
        for (q_l, q_p, q_r), block in tensor.data.items():
            arr = np.asarray(block)
            grouped.setdefault(q_l, []).append(
                (q_r, q_p, arr, int(arr.shape[2]), arr.dtype)
            )
    else:
        for (q_l, q_p, q_r), block in tensor.data.items():
            arr = np.asarray(block)
            grouped.setdefault(q_r, []).append(
                (q_l, q_p, arr, int(arr.shape[0]), arr.dtype)
            )
    grouped = {key: tuple(value) for key, value in grouped.items()}
    if isinstance(metadata, dict):
        metadata[cache_key] = grouped
        metadata[data_id_key] = id(tensor.data)
    return grouped


def _packed_boundary_labels(packed):
    """Return the integer topology payload installed in ``SU2MovingEnvironment``."""

    topology_arrays = (
        np.asarray(packed.ket_sector_ids, dtype=np.int64),
        np.asarray(packed.entry_offsets, dtype=np.int64),
        np.asarray(packed.out_sector_ids, dtype=np.int64),
        np.asarray(packed.channel_offsets, dtype=np.int64),
        np.asarray(packed.channel_ids, dtype=np.int64),
        np.asarray(packed.block_pool.shape_offsets, dtype=np.int64),
        np.asarray(packed.block_pool.shapes, dtype=np.int64),
    )
    header = np.asarray([array.size for array in topology_arrays], dtype=np.int64)
    sector_quantum_numbers = []
    for sector in packed.sector_codec.sectors:
        charge = getattr(sector, "charge", None)
        irrep = getattr(sector, "irrep", None)
        two_j = getattr(irrep, "two_j", None)
        if two_j is None:
            two_j = getattr(sector, "two_j", None)
        if charge is None or two_j is None:
            sector_quantum_numbers = []
            break
        sector_quantum_numbers.extend((int(charge), int(two_j)))
    labels = np.concatenate(
        (
            header,
            *topology_arrays,
            np.asarray(sector_quantum_numbers, dtype=np.int64),
        )
    )
    digest = hashlib.blake2b(labels.tobytes(), digest_size=8).digest()
    return labels, int.from_bytes(digest, "little") or 1


def _empty_packed_boundary_from_specs(output_specs, *, side, bond):
    """Build packed output topology without allocating its numerical arena."""

    from .su2_qchem_plan import (
        PackedArrayPool,
        PackedSU2BoundaryTable,
        SectorCodec,
        _sort_key,
    )

    grouped = {}
    for (q_out, q_in, channel), shape in output_specs.items():
        grouped.setdefault(q_in, {}).setdefault(q_out, {})[int(channel)] = tuple(
            int(dim) for dim in shape
        )
    sectors = [
        sector
        for q_in, out_map in grouped.items()
        for sector in (q_in, *out_map)
    ]
    codec = SectorCodec.from_iterable(sectors)
    ket_sector_ids = []
    entry_offsets = [0]
    out_sector_ids = []
    channel_offsets = [0]
    channel_ids = []
    ordered_keys = []
    offsets = [0]
    shape_offsets = [0]
    shapes = []
    for q_in, out_map in sorted(grouped.items(), key=lambda item: _sort_key(item[0])):
        ket_sector_ids.append(codec.id(q_in))
        for q_out, channel_map in sorted(
            out_map.items(),
            key=lambda item: _sort_key(item[0]),
        ):
            out_sector_ids.append(codec.id(q_out))
            for channel, shape in sorted(channel_map.items()):
                key = (q_out, q_in, int(channel))
                channel_ids.append(int(channel))
                ordered_keys.append(key)
                size = int(np.prod(shape, dtype=np.int64))
                offsets.append(offsets[-1] + size)
                shapes.extend(shape)
                shape_offsets.append(len(shapes))
            channel_offsets.append(len(channel_ids))
        entry_offsets.append(len(out_sector_ids))
    pool = PackedArrayPool(
        data=np.empty(0, dtype=np.float64),
        offsets=np.asarray(offsets, dtype=np.int64),
        shape_offsets=np.asarray(shape_offsets, dtype=np.int64),
        shapes=np.asarray(shapes, dtype=np.int64),
    )
    packed = PackedSU2BoundaryTable(
        side=str(side),
        bond=int(bond),
        representation="rank_coupled_by_ket",
        sector_codec=codec,
        ket_sector_ids=np.asarray(ket_sector_ids, dtype=np.int64),
        entry_offsets=np.asarray(entry_offsets, dtype=np.int64),
        out_sector_ids=np.asarray(out_sector_ids, dtype=np.int64),
        channel_offsets=np.asarray(channel_offsets, dtype=np.int64),
        channel_ids=np.asarray(channel_ids, dtype=np.int64),
        block_pool=pool,
    )
    return packed, {key: index for index, key in enumerate(ordered_keys)}


def _contract_rank_coupled_boundary_cpp(
    W,
    A,
    E_map,
    B,
    *,
    moving_environment,
    side,
    parent_bond,
    child_bond,
    numeric_revision,
):
    """Plan one reduced boundary update and execute all numerical routes in C++."""

    if (
        not isinstance(W, RankCoupledMPO)
        or parent_bond is None
        or child_bond is None
        or not hasattr(moving_environment, "advance_boundary")
    ):
        return None
    side = str(side).lower()
    edge = "left" if side == "left" else "right"
    packed_parent = E_map.ensure_packed(side=side, bond=int(parent_bond))
    if packed_parent is None:
        return None
    # LETTA commonly carries numerically real tensors in complex arrays.  Keep
    # those on the real boundary store installed by the local contextual
    # action; only select the complex route when the parent or site tensors
    # contain a material imaginary component.  A genuinely complex MPO block
    # is caught by ``register`` below and falls back to the Python contraction.
    complex_update = bool(
        _real64_contiguous_or_none(packed_parent.block_pool.data) is None
        or any(
            _real64_contiguous_or_none(block) is None
            for block in A.data.values()
        )
        or any(
            _real64_contiguous_or_none(block) is None
            for block in B.data.values()
        )
    )
    if complex_update and not hasattr(moving_environment, "advance_boundary_complex"):
        return None
    if (
        not complex_update
        and getattr(W, "normal_complementary_owner", None) is moving_environment
        and getattr(W, "normal_complementary_plan", None) is not None
    ):
        return _contract_normal_complementary_boundary_cpp(
            W,
            A,
            E_map,
            B,
            moving_environment=moving_environment,
            side=side,
            parent_bond=parent_bond,
            child_bond=child_bond,
            numeric_revision=numeric_revision,
        )

    sectors = tuple(packed_parent.sector_codec.sectors)
    parent_blocks = {}
    for row_index, ket_id in enumerate(packed_parent.ket_sector_ids):
        q_in = sectors[int(ket_id)]
        entry_start = int(packed_parent.entry_offsets[row_index])
        entry_stop = int(packed_parent.entry_offsets[row_index + 1])
        for entry_index in range(entry_start, entry_stop):
            q_out = sectors[int(packed_parent.out_sector_ids[entry_index])]
            channel_start = int(packed_parent.channel_offsets[entry_index])
            channel_stop = int(packed_parent.channel_offsets[entry_index + 1])
            parent_blocks[(q_out, q_in)] = {
                int(packed_parent.channel_ids[channel_index]): int(channel_index)
                for channel_index in range(channel_start, channel_stop)
            }

    a_entries_by_edge = _rank_coupled_site_entries_by_edge(A, edge)
    b_entries_by_edge = _rank_coupled_site_entries_by_edge(B, edge)
    bra_arrays = []
    ket_arrays = []
    mpo_arrays = []
    bra_keys = []
    ket_keys = []
    bra_indices = {}
    ket_indices = {}
    mpo_indices = {}
    routes = []
    output_specs = {}
    reduced_cache = {}
    reduced_physical = (
        bool(getattr(W, "normal_complementary_fully_reduced", False))
        or (not W.reduced_terms)
        or _rank_coupled_core_has_family(
            W,
            _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY,
        )
    )

    def register(array, arrays, index, *, block_key=None, keys=None):
        packed = (
            np.ascontiguousarray(array, dtype=np.complex128)
            if complex_update
            else _real64_contiguous_or_none(array)
        )
        if packed is None:
            return None
        key = id(packed)
        found = index.get(key)
        if found is None:
            found = len(arrays)
            index[key] = found
            arrays.append(packed)
            if keys is not None:
                keys.append(block_key)
        return int(found)

    for (q_boundary_bra, q_boundary_ket), channel_map in parent_blocks.items():
        a_entries = a_entries_by_edge.get(q_boundary_bra)
        b_entries = b_entries_by_edge.get(q_boundary_ket)
        if not a_entries or not b_entries:
            continue
        for q_next_bra, q_phys_bra, A_arr, bra_dim, _a_dtype in a_entries:
            bra_key = (
                (q_boundary_bra, q_phys_bra, q_next_bra)
                if edge == "left"
                else (q_next_bra, q_phys_bra, q_boundary_bra)
            )
            bra_index = register(
                A_arr,
                bra_arrays,
                bra_indices,
                block_key=bra_key,
                keys=bra_keys,
            )
            if bra_index is None:
                return None
            for q_next_ket, q_phys_ket, B_arr, ket_dim, _b_dtype in b_entries:
                ket_key = (
                    (q_boundary_ket, q_phys_ket, q_next_ket)
                    if edge == "left"
                    else (q_next_ket, q_phys_ket, q_boundary_ket)
                )
                ket_index = register(
                    B_arr,
                    ket_arrays,
                    ket_indices,
                    block_key=ket_key,
                    keys=ket_keys,
                )
                if ket_index is None:
                    return None
                reduced_key = (
                    (
                        q_boundary_bra,
                        q_boundary_ket,
                        q_phys_bra,
                        q_phys_ket,
                        q_next_bra,
                        q_next_ket,
                    )
                    if reduced_physical
                    else (q_phys_bra, q_phys_ket)
                )
                reduced = reduced_cache.get(reduced_key)
                if reduced is None:
                    if side == "left":
                        raw_reduced = (
                            _left_reduced_rank_coupled_block(
                                W,
                                q_boundary_bra,
                                q_boundary_ket,
                                q_phys_bra,
                                q_phys_ket,
                                q_next_bra,
                                q_next_ket,
                            )
                            if reduced_physical
                            else W.reduced_block(q_phys_bra, q_phys_ket)
                        )
                    else:
                        raw_reduced = (
                            _right_reduced_rank_coupled_block(
                                W,
                                q_next_bra,
                                q_next_ket,
                                q_phys_bra,
                                q_phys_ket,
                                q_boundary_bra,
                                q_boundary_ket,
                            )
                            if reduced_physical
                            else W.reduced_block(q_phys_bra, q_phys_ket)
                        )
                    reduced = tuple(
                        (int(left_channel), int(right_channel), np.asarray(block))
                        for (left_channel, right_channel), block in (
                            raw_reduced or {}
                        ).items()
                    )
                    reduced_cache[reduced_key] = reduced
                for left_channel, right_channel, W_arr in reduced:
                    parent_channel = (
                        int(left_channel)
                        if side == "left"
                        else int(right_channel)
                    )
                    parent_index = channel_map.get(parent_channel)
                    if parent_index is None:
                        continue
                    mpo_index = register(W_arr, mpo_arrays, mpo_indices)
                    if mpo_index is None:
                        return None
                    output_channel = (
                        int(right_channel)
                        if side == "left"
                        else int(left_channel)
                    )
                    output_key = (
                        q_next_bra,
                        q_next_ket,
                        output_channel,
                    )
                    output_shape = (
                        (
                            int(W_arr.shape[1]),
                            int(bra_dim),
                            int(ket_dim),
                        )
                        if side == "left"
                        else (
                            int(W_arr.shape[0]),
                            int(bra_dim),
                            int(ket_dim),
                        )
                    )
                    previous_shape = output_specs.setdefault(
                        output_key,
                        output_shape,
                    )
                    if previous_shape != output_shape:
                        raise ValueError(
                            "SU(2) boundary routes disagree on an output block shape."
                        )
                    routes.append(
                        (
                            int(parent_index),
                            int(bra_index),
                            int(ket_index),
                            int(mpo_index),
                            output_key,
                        )
                    )
    if not routes:
        return None

    output_table, output_indices = _empty_packed_boundary_from_specs(
        output_specs,
        side=side,
        bond=int(child_bond),
    )
    integer_routes = np.asarray(
        [
            (
                parent_index,
                bra_index,
                ket_index,
                mpo_index,
                output_indices[output_key],
            )
            for (
                parent_index,
                bra_index,
                ket_index,
                mpo_index,
                output_key,
            ) in routes
        ],
        dtype=np.int64,
    )
    from .su2_qchem_plan import PackedArrayPool

    bra_pool = PackedArrayPool.from_arrays(bra_arrays)
    ket_pool = PackedArrayPool.from_arrays(ket_arrays)
    mpo_pool = PackedArrayPool.from_arrays(mpo_arrays)
    labels, topology_revision = _packed_boundary_labels(output_table)
    numeric_revision = int(
        1 if numeric_revision is None else numeric_revision
    )
    metric_route_coefficients = None
    if (
        bool(getattr(W, "fully_reduced_identity", False))
        and side == "right"
    ):
        metric_route_coefficients = np.asarray(
            [
                (
                    float(_sector_irrep(bra_keys[int(route[1])][2]).dim)
                    * float(_sector_irrep(ket_keys[int(route[2])][2]).dim)
                    / (
                        float(_sector_irrep(bra_keys[int(route[1])][0]).dim)
                        * float(_sector_irrep(ket_keys[int(route[2])][0]).dim)
                    )
                )
                for route in integer_routes
            ],
            dtype=np.float64,
        )
    metric_route_digest = hashlib.blake2b(digest_size=8)
    metric_route_digest.update(integer_routes.view(np.uint8))
    metric_route_digest.update(
        np.asarray(
            (
                np.ones(len(integer_routes), dtype=np.float64)
                if metric_route_coefficients is None
                else metric_route_coefficients
            ),
            dtype=np.float64,
        ).view(np.uint8)
    )
    metric_route_digest.update(
        np.asarray(mpo_pool.offsets, dtype=np.int64).view(np.uint8)
    )
    metric_route_digest.update(
        np.asarray(mpo_pool.shapes, dtype=np.int64).view(np.uint8)
    )
    metric_route_topology_revision = (
        int.from_bytes(metric_route_digest.digest(), "little") or 1
    )
    bra_split_marker = (
        (getattr(A, "metadata", None) or {}).get("_cpp_split_site")
    )
    ket_split_marker = (
        (getattr(B, "metadata", None) or {}).get("_cpp_split_site")
    )
    metric_split_action = (
        not complex_update
        and bool(getattr(W, "fully_reduced_identity", False))
        and bra_split_marker is not None
        and bra_split_marker == ket_split_marker
        and len(bra_split_marker) == 4
        and callable(
            getattr(
                moving_environment,
                "advance_metric_boundary_from_split_site",
                None,
            )
        )
        and moving_environment.split_site_installed(
            int(bra_split_marker[1]),
            bra_split_marker,
        )
    )
    if metric_split_action:
        values, _same_topology = (
            moving_environment.advance_metric_boundary_from_split_site(
                side,
                int(parent_bond),
                int(child_bond),
                int(bra_split_marker[1]),
                integer_routes,
                tuple(bra_keys),
                tuple(ket_keys),
                bra_split_marker,
                mpo_pool.data,
                mpo_pool.offsets,
                mpo_pool.shape_offsets,
                mpo_pool.shapes,
                output_table.block_pool.offsets,
                output_table.block_pool.shape_offsets,
                output_table.block_pool.shapes,
                labels,
                int(topology_revision),
                int(numeric_revision),
                int(metric_route_topology_revision),
                route_coefficients=metric_route_coefficients,
            )
        )
    else:
        advance = (
            moving_environment.advance_boundary_complex
            if complex_update
            else moving_environment.advance_boundary
        )
        values, _same_topology = advance(
            side,
            int(parent_bond),
            int(child_bond),
            integer_routes,
            bra_pool.data,
            bra_pool.offsets,
            bra_pool.shape_offsets,
            bra_pool.shapes,
            ket_pool.data,
            ket_pool.offsets,
            ket_pool.shape_offsets,
            ket_pool.shapes,
            mpo_pool.data,
            mpo_pool.offsets,
            mpo_pool.shape_offsets,
            mpo_pool.shapes,
            output_table.block_pool.offsets,
            output_table.block_pool.shape_offsets,
            output_table.block_pool.shapes,
            labels,
            int(topology_revision),
            int(numeric_revision),
            metric_boundary=bool(
                getattr(W, "fully_reduced_identity", False)
            ),
            route_coefficients=metric_route_coefficients,
        )
    output_pool = replace(
        output_table.block_pool,
        data=np.ascontiguousarray(
            values,
            dtype=np.complex128 if complex_update else np.float64,
        ),
        _shape_cache=None,
        _array_cache=None,
    )
    output_table = replace(output_table, block_pool=output_pool)
    data = _PackedRankCoupledEnvironmentMap(
        output_table,
        n_channels=(
            len(W.right_channel_irreps)
            if side == "left"
            else len(W.left_channel_irreps)
        ),
    )
    return (
        data,
        output_table,
        int(topology_revision),
        int(numeric_revision),
    )


def _pack_normal_complementary_boundary_routes_cpp(
    W,
    A,
    B,
    packed_parent,
    *,
    side,
    child_bond,
):
    """Pack NC topology in compiled code, leaving only sector objects in Python."""

    module = _su2_kernel_module()
    kernel = (
        None
        if module is None
        else getattr(
            module,
            "pack_normal_complementary_boundary_routes",
            None,
        )
    )
    if kernel is None:
        return None
    plan = W.normal_complementary_plan
    source = np.asarray(plan["source"], dtype=np.int64)
    target = np.asarray(plan["target"], dtype=np.int64)
    operator_ids = np.asarray(plan["operator"], dtype=np.int64)
    parent_channels = source if side == "left" else target
    output_channels = target if side == "left" else source
    n_parent_channels = int(
        plan["left_channels"] if side == "left" else plan["right_channels"]
    )
    schedule_key = f"_transition_schedule_{side}"
    schedule = plan.get(schedule_key)
    if schedule is None:
        buckets = [[] for _ in range(n_parent_channels)]
        for transition, channel in enumerate(parent_channels):
            buckets[int(channel)].append(int(transition))
        transition_offsets = np.asarray(
            np.cumsum([0, *(len(bucket) for bucket in buckets)]),
            dtype=np.int64,
        )
        transition_ids = np.asarray(
            [transition for bucket in buckets for transition in bucket],
            dtype=np.int64,
        )
        schedule = (transition_offsets, transition_ids)
        plan[schedule_key] = schedule
    transition_offsets, transition_ids = schedule

    edge = "left" if side == "left" else "right"
    a_items = tuple(A.data.items())
    b_items = tuple(B.data.items())
    next_sectors = []
    for key, _block in (*a_items, *b_items):
        q_left, _q_phys, q_right = key
        next_sectors.append(q_right if edge == "left" else q_left)
    next_sectors = tuple(dict.fromkeys(next_sectors))
    next_index = {sector: idx for idx, sector in enumerate(next_sectors)}
    parent_index = packed_parent.sector_codec.index

    def site_entries(items):
        entries = []
        keys = []
        for key, block in items:
            q_left, q_phys, q_right = key
            array = np.asarray(block)
            if edge == "left":
                q_boundary = q_left
                q_next = q_right
                dimension = int(array.shape[2])
            else:
                q_boundary = q_right
                q_next = q_left
                dimension = int(array.shape[0])
            boundary_id = parent_index.get(q_boundary)
            if boundary_id is None:
                continue
            array_index = len(keys)
            keys.append(key)
            entries.append(
                (
                    int(boundary_id),
                    int(next_index[q_next]),
                    int(q_phys.charge),
                    int(q_phys.irrep.two_j),
                    int(q_boundary.irrep.two_j),
                    int(q_next.irrep.two_j),
                    dimension,
                    array_index,
                )
            )
        return (
            np.asarray(entries, dtype=np.int64).reshape((-1, 8)),
            tuple(keys),
        )

    bra_entries, bra_keys = site_entries(a_items)
    ket_entries, ket_keys = site_entries(b_items)
    primitive_nonzero = np.asarray(
        plan.get("_primitive_nonzero"),
        dtype=np.uint8,
    )
    if (
        primitive_nonzero.ndim != 3
        or primitive_nonzero.shape[:2] != (3, 3)
    ):
        return None
    integer_routes, output_specs = kernel(
        packed_parent.ket_sector_ids,
        packed_parent.entry_offsets,
        packed_parent.out_sector_ids,
        packed_parent.channel_offsets,
        packed_parent.channel_ids,
        transition_offsets,
        transition_ids,
        operator_ids,
        output_channels,
        primitive_nonzero,
        bra_entries,
        ket_entries,
        len(next_sectors),
        int(
            plan["right_channels"]
            if side == "left"
            else plan["left_channels"]
        ),
    )
    integer_routes = np.ascontiguousarray(integer_routes, dtype=np.int32)
    output_specs = np.asarray(output_specs, dtype=np.int64).reshape((-1, 6))
    if integer_routes.size == 0:
        return None
    output_quantum_numbers = np.asarray(
        plan[
            "right_channel_quantum_numbers"
            if side == "left"
            else "left_channel_quantum_numbers"
        ],
        dtype=np.int64,
    )
    specs = {}
    output_keys = [None] * int(output_specs.shape[0])
    for (
        next_bra,
        next_ket,
        output_channel,
        bra_dim,
        ket_dim,
        output_id,
    ) in output_specs:
        output_key = (
            next_sectors[int(next_bra)],
            next_sectors[int(next_ket)],
            int(output_channel),
        )
        output_keys[int(output_id)] = output_key
        specs[output_key] = (
            int(output_quantum_numbers[int(output_channel), 1]) + 1,
            int(bra_dim),
            int(ket_dim),
        )
    output_table, output_indices = _empty_packed_boundary_from_specs(
        specs,
        side=side,
        bond=int(child_bond),
    )
    output_remap = np.asarray(
        [output_indices[key] for key in output_keys],
        dtype=np.int64,
    )
    integer_routes[:, 6] = output_remap[integer_routes[:, 6]]
    route_topology_revision = int.from_bytes(
        hashlib.blake2b(
            integer_routes.view(np.uint8),
            digest_size=8,
        ).digest(),
        "little",
    ) or 1
    labels, topology_revision = _packed_boundary_labels(output_table)
    return {
        "bra_keys": bra_keys,
        "ket_keys": ket_keys,
        "routes": integer_routes,
        "output_table": output_table,
        "labels": labels,
        "topology_revision": int(topology_revision),
        "route_topology_revision": int(route_topology_revision),
    }


def _contract_normal_complementary_boundary_cpp(
    W,
    A,
    E_map,
    B,
    *,
    moving_environment,
    side,
    parent_bond,
    child_bond,
    numeric_revision,
):
    """Advance one NC boundary without constructing Python MPO component cores."""

    if (
        parent_bond is None
        or child_bond is None
        or not hasattr(
            moving_environment,
            "advance_normal_complementary_boundary",
        )
    ):
        return None
    side = str(side).lower()
    edge = "left" if side == "left" else "right"
    packed_parent = E_map.ensure_packed(side=side, bond=int(parent_bond))
    if (
        packed_parent is None
        or np.iscomplexobj(packed_parent.block_pool.data)
    ):
        return None
    parent_labels, parent_topology_revision = _packed_boundary_labels(
        packed_parent
    )
    parent_numeric_revision = int(
        getattr(E_map, "cpp_numeric_revision", 0) or 1
    )
    if not moving_environment.boundary_installed(
        side,
        int(parent_bond),
        int(parent_topology_revision),
        parent_numeric_revision,
    ):
        moving_environment.install_boundary(
            side,
            int(parent_bond),
            packed_parent.block_pool.data,
            packed_parent.block_pool.offsets,
            parent_labels,
            int(parent_topology_revision),
            parent_numeric_revision,
        )

    plan = W.normal_complementary_plan
    def tensor_topology_signature(tensor):
        metadata = getattr(tensor, "metadata", None)
        cache_key = "_normal_complementary_tensor_topology"
        if (
            isinstance(metadata, dict)
            and cache_key in metadata
        ):
            return metadata[cache_key]
        signature = tuple(
            (key, tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in tensor.data.items()
        )
        if isinstance(metadata, dict):
            metadata[cache_key] = signature
        return signature

    parent_sector_signature = tuple(
        (
            int(getattr(sector, "charge", 0)),
            int(getattr(getattr(sector, "irrep", None), "two_j", 0)),
            repr(getattr(sector, "point_group", None)),
        )
        for sector in packed_parent.sector_codec.sectors
    )

    route_cache = plan.setdefault("_boundary_action_cache", {})
    route_cache_key = (
        side,
        int(parent_bond),
        int(child_bond),
        int(parent_topology_revision),
        parent_sector_signature,
        tensor_topology_signature(A),
        tensor_topology_signature(B),
    )
    cached_route_plan = route_cache.get(route_cache_key)
    if (
        cached_route_plan is not None
        and cached_route_plan.get("routes") is None
    ):
        # A fully C++ half sweep may replace the action stored for this bond
        # after Python discarded its duplicate route array. Repack the compact
        # topology for this explicit reference/expectation contraction instead
        # of assuming that the last C++ action still has this cache revision.
        cached_route_plan = None
    if cached_route_plan is None:
        cached_route_plan = _pack_normal_complementary_boundary_routes_cpp(
            W,
            A,
            B,
            packed_parent,
            side=side,
            child_bond=child_bond,
        )
        if cached_route_plan is not None:
            for key in tuple(route_cache):
                if (
                    key[:3] == route_cache_key[:3]
                    and key != route_cache_key
                ):
                    route_cache.pop(key)
            route_cache[route_cache_key] = cached_route_plan
    if cached_route_plan is not None:
        bra_keys = cached_route_plan["bra_keys"]
        ket_keys = cached_route_plan["ket_keys"]
        integer_routes = cached_route_plan["routes"]
        output_table = cached_route_plan["output_table"]
        labels = cached_route_plan["labels"]
        topology_revision = cached_route_plan["topology_revision"]
        route_topology_revision = cached_route_plan[
            "route_topology_revision"
        ]
    else:
        source = np.asarray(plan["source"], dtype=np.int64)
        target = np.asarray(plan["target"], dtype=np.int64)
        operator_ids = np.asarray(plan["operator"], dtype=np.int64)
        primitive_nonzero = plan.get("_primitive_nonzero")
        parent_channels = source if side == "left" else target
        output_channels = target if side == "left" else source
        output_quantum_numbers = np.asarray(
            plan[
                "right_channel_quantum_numbers"
                if side == "left"
                else "left_channel_quantum_numbers"
            ],
            dtype=np.int64,
        )
        transitions_by_parent = {}
        for transition, channel in enumerate(parent_channels):
            transitions_by_parent.setdefault(int(channel), []).append(
                int(transition)
            )

        sectors = tuple(packed_parent.sector_codec.sectors)
        parent_blocks = {}
        for row_index, ket_id in enumerate(packed_parent.ket_sector_ids):
            q_in = sectors[int(ket_id)]
            entry_start = int(packed_parent.entry_offsets[row_index])
            entry_stop = int(packed_parent.entry_offsets[row_index + 1])
            for entry_index in range(entry_start, entry_stop):
                q_out = sectors[int(packed_parent.out_sector_ids[entry_index])]
                channel_start = int(packed_parent.channel_offsets[entry_index])
                channel_stop = int(packed_parent.channel_offsets[entry_index + 1])
                parent_blocks[(q_out, q_in)] = {
                    int(packed_parent.channel_ids[channel_index]): int(channel_index)
                    for channel_index in range(channel_start, channel_stop)
                }

        a_entries_by_edge = _rank_coupled_site_entries_by_edge(A, edge)
        b_entries_by_edge = _rank_coupled_site_entries_by_edge(B, edge)
        a_keys_by_array = {
            id(np.asarray(block)): key for key, block in A.data.items()
        }
        b_keys_by_array = {
            id(np.asarray(block)): key for key, block in B.data.items()
        }
        bra_arrays = []
        ket_arrays = []
        bra_keys = []
        ket_keys = []
        bra_indices = {}
        ket_indices = {}
        routes = []
        output_specs = {}
        cacheable = True

        def register(array, arrays, keys, index, source_keys):
            nonlocal cacheable
            source_array = np.asarray(array)
            real = _real64_contiguous_or_none(source_array)
            if real is None:
                return None
            array_id = id(source_array)
            found = index.get(array_id)
            if found is None:
                found = len(arrays)
                index[array_id] = found
                arrays.append(real)
                source_key = source_keys.get(array_id)
                if source_key is None:
                    cacheable = False
                keys.append(source_key)
            return int(found)

        for (q_boundary_bra, q_boundary_ket), channel_map in parent_blocks.items():
            a_entries = a_entries_by_edge.get(q_boundary_bra)
            b_entries = b_entries_by_edge.get(q_boundary_ket)
            if not a_entries or not b_entries:
                continue
            for q_next_bra, q_phys_bra, A_arr, bra_dim, _a_dtype in a_entries:
                bra_index = register(
                    A_arr,
                    bra_arrays,
                    bra_keys,
                    bra_indices,
                    a_keys_by_array,
                )
                if bra_index is None:
                    return None
                for q_next_ket, q_phys_ket, B_arr, ket_dim, _b_dtype in b_entries:
                    ket_index = register(
                        B_arr,
                        ket_arrays,
                        ket_keys,
                        ket_indices,
                        b_keys_by_array,
                    )
                    if ket_index is None:
                        return None
                    physical_out_charge = int(q_phys_bra.charge)
                    physical_in_charge = int(q_phys_ket.charge)
                    for parent_channel, parent_index in channel_map.items():
                        for transition in transitions_by_parent.get(
                            int(parent_channel),
                            (),
                        ):
                            operator_id = int(operator_ids[transition])
                            if (
                                primitive_nonzero is not None
                                and not primitive_nonzero[
                                    physical_out_charge,
                                    physical_in_charge,
                                    operator_id,
                                ]
                            ):
                                continue
                            output_channel = int(output_channels[transition])
                            output_key = (
                                q_next_bra,
                                q_next_ket,
                                output_channel,
                            )
                            channel_two_j = int(
                                output_quantum_numbers[output_channel, 1]
                            )
                            output_shape = (
                                channel_two_j + 1,
                                int(bra_dim),
                                int(ket_dim),
                            )
                            previous_shape = output_specs.setdefault(
                                output_key,
                                output_shape,
                            )
                            if previous_shape != output_shape:
                                raise ValueError(
                                    "SU(2) NC routes disagree on an output block shape."
                                )
                            routes.append(
                                (
                                    int(parent_index),
                                    int(bra_index),
                                    int(ket_index),
                                    int(transition),
                                    physical_out_charge,
                                    physical_in_charge,
                                    output_key,
                                    int(q_boundary_bra.irrep.two_j),
                                    int(q_boundary_ket.irrep.two_j),
                                    int(q_phys_bra.irrep.two_j),
                                    int(q_phys_ket.irrep.two_j),
                                    int(q_next_bra.irrep.two_j),
                                    int(q_next_ket.irrep.two_j),
                                )
                            )
        if not routes:
            return None

        output_table, output_indices = _empty_packed_boundary_from_specs(
            output_specs,
            side=side,
            bond=int(child_bond),
        )
        integer_routes = np.asarray(
            [
                (
                    parent_index,
                    bra_index,
                    ket_index,
                    transition,
                    physical_out_charge,
                    physical_in_charge,
                    output_indices[output_key],
                    q_boundary_bra,
                    q_boundary_ket,
                    q_phys_bra,
                    q_phys_ket,
                    q_next_bra,
                    q_next_ket,
                )
                for (
                    parent_index,
                    bra_index,
                    ket_index,
                    transition,
                    physical_out_charge,
                    physical_in_charge,
                    output_key,
                    q_boundary_bra,
                    q_boundary_ket,
                    q_phys_bra,
                    q_phys_ket,
                    q_next_bra,
                    q_next_ket,
                ) in routes
            ],
            dtype=np.int32,
        )
        route_topology_revision = int.from_bytes(
            hashlib.blake2b(
                integer_routes.view(np.uint8),
                digest_size=8,
            ).digest(),
            "little",
        ) or 1
        labels, topology_revision = _packed_boundary_labels(output_table)
        if cacheable:
            for key in tuple(route_cache):
                if (
                    key[:3] == route_cache_key[:3]
                    and key != route_cache_key
                ):
                    route_cache.pop(key)
            route_cache[route_cache_key] = {
                "bra_keys": tuple(bra_keys),
                "ket_keys": tuple(ket_keys),
                "routes": integer_routes,
                "output_table": output_table,
                "labels": labels,
                "topology_revision": int(topology_revision),
                "route_topology_revision": int(route_topology_revision),
            }
    numeric_revision = int(
        1 if numeric_revision is None else numeric_revision
    )
    bra_split_marker = (
        (getattr(A, "metadata", None) or {}).get("_cpp_split_site")
    )
    ket_split_marker = (
        (getattr(B, "metadata", None) or {}).get("_cpp_split_site")
    )
    split_site_action = (
        bra_split_marker is not None
        and bra_split_marker == ket_split_marker
        and len(bra_split_marker) == 4
        and int(bra_split_marker[1]) == int(W.normal_complementary_site)
        and all(key is not None for key in (*bra_keys, *ket_keys))
        and hasattr(
            moving_environment,
            "advance_normal_complementary_boundary_from_split_site",
        )
        and moving_environment.split_site_installed(
            int(W.normal_complementary_site),
            bra_split_marker,
        )
    )
    if split_site_action:
        values, _same_topology = (
            moving_environment.advance_normal_complementary_boundary_from_split_site(
                side,
                int(parent_bond),
                int(child_bond),
                int(W.normal_complementary_site),
                bool(getattr(W, "normal_complementary_fully_reduced", False)),
                integer_routes,
                bra_keys,
                ket_keys,
                bra_split_marker,
                output_table.block_pool.offsets,
                output_table.block_pool.shape_offsets,
                output_table.block_pool.shapes,
                labels,
                int(topology_revision),
                int(numeric_revision),
                bool(getattr(W, "normal_complementary_right_dual", False)),
                int(route_topology_revision),
            )
        )
    else:
        bra_arrays = [
            _real64_contiguous_or_none(A.data[key])
            for key in bra_keys
        ]
        ket_arrays = [
            _real64_contiguous_or_none(B.data[key])
            for key in ket_keys
        ]
        if any(array is None for array in (*bra_arrays, *ket_arrays)):
            return None
        from .su2_qchem_plan import PackedArrayPool

        bra_pool = PackedArrayPool.from_arrays(bra_arrays)
        ket_pool = PackedArrayPool.from_arrays(ket_arrays)
        values, _same_topology = (
            moving_environment.advance_normal_complementary_boundary(
                side,
                int(parent_bond),
                int(child_bond),
                int(W.normal_complementary_site),
                bool(getattr(W, "normal_complementary_fully_reduced", False)),
                integer_routes,
                bra_pool.data,
                bra_pool.offsets,
                bra_pool.shape_offsets,
                bra_pool.shapes,
                ket_pool.data,
                ket_pool.offsets,
                ket_pool.shape_offsets,
                ket_pool.shapes,
                output_table.block_pool.offsets,
                output_table.block_pool.shape_offsets,
                output_table.block_pool.shapes,
                labels,
                int(topology_revision),
                int(numeric_revision),
                bool(getattr(W, "normal_complementary_right_dual", False)),
                int(route_topology_revision),
            )
        )
    if cached_route_plan is not None:
        cached_route_plan["routes"] = None
    output_pool = replace(
        output_table.block_pool,
        data=np.ascontiguousarray(values, dtype=np.float64),
        _shape_cache=None,
        _array_cache=None,
    )
    output_table = replace(output_table, block_pool=output_pool)
    data = _PackedRankCoupledEnvironmentMap(
        output_table,
        n_channels=int(
            plan["right_channels"]
            if side == "left"
            else plan["left_channels"]
        ),
    )
    return (
        data,
        output_table,
        int(topology_revision),
        int(numeric_revision),
    )


def _contract_from_left_blocks_rank_coupled(W, A, E_map, B):
    out = {}
    mpo_dtype = _mpo_dtype(W)
    module = _su2_kernel_module()
    batch_kernel = (
        None
        if module is None
        else getattr(module, "accumulate_rank_coupled_left_terms", None)
    )
    real_batch_kernel = (
        None
        if module is None or not _USE_REAL_RANK_COUPLED_ACCUMULATE
        else getattr(module, "accumulate_rank_coupled_left_real_terms", None)
    )
    reduced_physical = (
        bool(getattr(W, "normal_complementary_fully_reduced", False))
        or (not W.reduced_terms)
        or _rank_coupled_core_has_family(
            W,
            _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY,
        )
    )
    a_blocks_by_left = _rank_coupled_site_entries_by_edge(A, "left")
    b_blocks_by_left = _rank_coupled_site_entries_by_edge(B, "left")
    reduced_cache = {}
    active_reduced_cache = {}
    for (q_lb, q_lk), E_blocks in E_map.items():
        a_entries = a_blocks_by_left.get(q_lb)
        b_entries = b_blocks_by_left.get(q_lk)
        if not a_entries or not b_entries:
            continue
        e_arrays = RankCoupledChannelBlocks(
            {
                int(channel): np.asarray(block)
                for channel, block in _rank_coupled_channel_items(E_blocks)
            },
            len(E_blocks),
        )
        if real_batch_kernel is not None:
            e_real_data = {
                channel: _real64_contiguous_or_none(block)
                for channel, block in e_arrays.items()
            }
            if any(block is None for block in e_real_data.values()):
                e_real_arrays = None
            else:
                e_real_arrays = RankCoupledChannelBlocks(
                    e_real_data,
                    len(e_arrays),
                )
        else:
            e_real_arrays = None
        e_dtypes = tuple(block.dtype for block in e_arrays.values())
        for q_rb, q_pb, A_arr, bra_dim, a_dtype in a_entries:
            A_conj = A_arr if not np.iscomplexobj(A_arr) else A_arr.conj()
            A_real = (
                None
                if real_batch_kernel is None
                else _real64_contiguous_or_none(A_conj)
            )
            for q_rk, q_pk, B_arr, ket_dim, b_dtype in b_entries:
                B_real = (
                    None
                    if real_batch_kernel is None
                    else _real64_contiguous_or_none(B_arr)
                )
                reduced_key = (
                    (
                        id(q_lb),
                        id(q_lk),
                        id(q_pb),
                        id(q_pk),
                        id(q_rb),
                        id(q_rk),
                    )
                    if reduced_physical
                    else (id(q_pb), id(q_pk))
                )
                cached = reduced_cache.get(reduced_key)
                if cached is None:
                    raw_reduced = (
                        _left_reduced_rank_coupled_block(
                            W,
                            q_lb,
                            q_lk,
                            q_pb,
                            q_pk,
                            q_rb,
                            q_rk,
                        )
                        if reduced_physical
                        else W.reduced_block(q_pb, q_pk)
                    )
                    reduced = tuple(
                        (int(left_idx), int(right_idx), np.asarray(block))
                        for (left_idx, right_idx), block in (raw_reduced or {}).items()
                    )
                    cached = (
                        reduced,
                        (
                            None
                            if real_batch_kernel is None
                            else _rank_coupled_real_term_arrays(reduced)
                        ),
                    )
                    reduced_cache[reduced_key] = cached
                reduced, _reduced_real = cached
                active_key = (reduced_key, tuple(e_arrays.data))
                active_cached = active_reduced_cache.get(active_key)
                if active_cached is None:
                    reduced = tuple(
                        term for term in reduced if int(term[0]) in e_arrays
                    )
                    active_cached = (
                        reduced,
                        (
                            None
                            if real_batch_kernel is None
                            else _rank_coupled_real_term_arrays(reduced)
                        ),
                    )
                    active_reduced_cache[active_key] = active_cached
                reduced, reduced_real = active_cached
                if not reduced:
                    continue
                key = (q_rb, q_rk)
                dtype = np.result_type(mpo_dtype, a_dtype, b_dtype, *e_dtypes)
                target = out.get(key)
                if target is None:
                    target = RankCoupledChannelBlocks(
                        {},
                        len(W.right_channel_irreps),
                    )
                    out[key] = target
                for _left_idx, right_idx, _w_block in reduced:
                    if int(right_idx) not in target:
                        irrep = W.right_channel_irreps[int(right_idx)]
                        target.data[int(right_idx)] = np.zeros(
                            (irrep.dim, bra_dim, ket_dim),
                            dtype=dtype,
                        )
                real_attempted = (
                    real_batch_kernel is not None
                    and reduced_real is not None
                    and e_real_arrays is not None
                    and A_real is not None
                    and B_real is not None
                )
                if real_attempted:
                    if real_batch_kernel(
                        target,
                        e_real_arrays,
                        A_real,
                        B_real,
                        reduced_real[0],
                        reduced_real[1],
                        reduced_real[2],
                        _RANK_COUPLED_SMALL_CONTRACTION_WORK,
                    ):
                        continue
                elif batch_kernel is not None and batch_kernel(
                    target,
                    e_arrays,
                    A_conj,
                    B_arr,
                    reduced,
                    _RANK_COUPLED_SMALL_CONTRACTION_WORK,
                ):
                    continue
                for left_idx, right_idx, w_block in reduced:
                    if left_idx >= len(e_arrays):
                        continue
                    target[right_idx] += _contract_rank_coupled_left_step(
                        e_arrays[left_idx],
                        A_conj,
                        w_block,
                        B_arr,
                    )
    return out


def _contract_from_right_blocks(W, A, F_map, B, phys_slices):
    if _is_identity_mpo_core(W):
        return _contract_from_right_identity_blocks(A, F_map, B)
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


def _contract_from_right_identity_blocks(A, F_map, B):
    out = {}
    for (q_lb, q_pb, q_rb), A_block in A.data.items():
        A_conj = np.asarray(A_block).conj()
        for (q_lk, q_pk, q_rk), B_block in B.data.items():
            if q_pb != q_pk:
                continue
            F_block = F_map.get((q_rb, q_rk))
            if F_block is None:
                continue
            F_arr = np.asarray(F_block)
            B_arr = np.asarray(B_block)
            if F_arr.shape[0] != 1:
                contrib = _contract_rank_coupled_right_step(
                    A_conj,
                    np.eye(A_conj.shape[1], dtype=np.result_type(A_conj, F_arr, B_arr))[
                        None,
                        None,
                    ],
                    F_arr,
                    B_arr,
                )
            else:
                block = np.einsum(
                    "ipr,rs,jps->ij",
                    A_conj,
                    F_arr[0],
                    B_arr,
                    optimize=False,
                )
                contrib = block.reshape(1, int(block.shape[0]), int(block.shape[1]))
            key = (q_lb, q_lk)
            if key in out:
                out[key] = out[key] + contrib
            else:
                out[key] = contrib
    return out


def _contract_from_right_blocks_rank_coupled(W, A, F_map, B):
    out = {}
    mpo_dtype = _mpo_dtype(W)
    module = _su2_kernel_module()
    batch_kernel = (
        None
        if module is None
        else getattr(module, "accumulate_rank_coupled_right_terms", None)
    )
    real_batch_kernel = (
        None
        if module is None or not _USE_REAL_RANK_COUPLED_ACCUMULATE
        else getattr(module, "accumulate_rank_coupled_right_real_terms", None)
    )
    reduced_physical = (
        bool(getattr(W, "normal_complementary_fully_reduced", False))
        or (not W.reduced_terms)
        or _rank_coupled_core_has_family(
            W,
            _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY,
        )
    )
    a_blocks_by_right = _rank_coupled_site_entries_by_edge(A, "right")
    b_blocks_by_right = _rank_coupled_site_entries_by_edge(B, "right")
    reduced_cache = {}
    active_reduced_cache = {}
    for (q_rb, q_rk), F_blocks in F_map.items():
        a_entries = a_blocks_by_right.get(q_rb)
        b_entries = b_blocks_by_right.get(q_rk)
        if not a_entries or not b_entries:
            continue
        f_arrays = RankCoupledChannelBlocks(
            {
                int(channel): np.asarray(block)
                for channel, block in _rank_coupled_channel_items(F_blocks)
            },
            len(F_blocks),
        )
        if real_batch_kernel is not None:
            f_real_data = {
                channel: _real64_contiguous_or_none(block)
                for channel, block in f_arrays.items()
            }
            if any(block is None for block in f_real_data.values()):
                f_real_arrays = None
            else:
                f_real_arrays = RankCoupledChannelBlocks(
                    f_real_data,
                    len(f_arrays),
                )
        else:
            f_real_arrays = None
        f_dtypes = tuple(block.dtype for block in f_arrays.values())
        for q_lb, q_pb, A_arr, bra_dim, a_dtype in a_entries:
            A_conj = A_arr if not np.iscomplexobj(A_arr) else A_arr.conj()
            A_real = (
                None
                if real_batch_kernel is None
                else _real64_contiguous_or_none(A_conj)
            )
            for q_lk, q_pk, B_arr, ket_dim, b_dtype in b_entries:
                B_real = (
                    None
                    if real_batch_kernel is None
                    else _real64_contiguous_or_none(B_arr)
                )
                reduced_key = (
                    (
                        id(q_lb),
                        id(q_lk),
                        id(q_pb),
                        id(q_pk),
                        id(q_rb),
                        id(q_rk),
                    )
                    if reduced_physical
                    else (id(q_pb), id(q_pk))
                )
                cached = reduced_cache.get(reduced_key)
                if cached is None:
                    raw_reduced = (
                        _right_reduced_rank_coupled_block(
                            W,
                            q_lb,
                            q_lk,
                            q_pb,
                            q_pk,
                            q_rb,
                            q_rk,
                        )
                        if reduced_physical
                        else W.reduced_block(q_pb, q_pk)
                    )
                    reduced = tuple(
                        (int(left_idx), int(right_idx), np.asarray(block))
                        for (left_idx, right_idx), block in (raw_reduced or {}).items()
                    )
                    cached = (
                        reduced,
                        (
                            None
                            if real_batch_kernel is None
                            else _rank_coupled_real_term_arrays(reduced)
                        ),
                    )
                    reduced_cache[reduced_key] = cached
                reduced, _reduced_real = cached
                active_key = (reduced_key, tuple(f_arrays.data))
                active_cached = active_reduced_cache.get(active_key)
                if active_cached is None:
                    reduced = tuple(
                        term for term in reduced if int(term[1]) in f_arrays
                    )
                    active_cached = (
                        reduced,
                        (
                            None
                            if real_batch_kernel is None
                            else _rank_coupled_real_term_arrays(reduced)
                        ),
                    )
                    active_reduced_cache[active_key] = active_cached
                reduced, reduced_real = active_cached
                if not reduced:
                    continue
                key = (q_lb, q_lk)
                dtype = np.result_type(mpo_dtype, a_dtype, b_dtype, *f_dtypes)
                target = out.get(key)
                if target is None:
                    target = RankCoupledChannelBlocks(
                        {},
                        len(W.left_channel_irreps),
                    )
                    out[key] = target
                for left_idx, _right_idx, _w_block in reduced:
                    if int(left_idx) not in target:
                        irrep = W.left_channel_irreps[int(left_idx)]
                        target.data[int(left_idx)] = np.zeros(
                            (irrep.dim, bra_dim, ket_dim),
                            dtype=dtype,
                        )
                real_attempted = (
                    real_batch_kernel is not None
                    and reduced_real is not None
                    and f_real_arrays is not None
                    and A_real is not None
                    and B_real is not None
                )
                if real_attempted:
                    if real_batch_kernel(
                        target,
                        A_real,
                        B_real,
                        f_real_arrays,
                        reduced_real[0],
                        reduced_real[1],
                        reduced_real[2],
                        _RANK_COUPLED_SMALL_CONTRACTION_WORK,
                    ):
                        continue
                elif batch_kernel is not None and batch_kernel(
                    target,
                    A_conj,
                    B_arr,
                    f_arrays,
                    reduced,
                    _RANK_COUPLED_SMALL_CONTRACTION_WORK,
                ):
                    continue
                for left_idx, right_idx, w_block in reduced:
                    if right_idx >= len(f_arrays):
                        continue
                    target[left_idx] += _contract_rank_coupled_right_step(
                        A_conj,
                        w_block,
                        f_arrays[right_idx],
                        B_arr,
                    )
    return out


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
    right_factor_middle_cache = {}

    for in_entry in basis:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_terms = []
        if left_factor_table is not None and right_factor_table is not None:
            left_entries = left_factor_table.get((q_lk, q_p1k), ())
            right_entries = right_factor_table.get((q_rk, q_p2k), ())
            right_by_middle = right_factor_middle_cache.get(id(right_entries))
            if right_by_middle is None:
                right_by_middle = {}
                for right_item in right_entries:
                    right_by_middle.setdefault(int(right_item[2]), []).append(right_item)
                right_factor_middle_cache[id(right_entries)] = right_by_middle
            for left_item in left_entries:
                q_lb, q_p1b, middle_idx, left_factor = left_item[:4]
                left_families = left_item[4] if len(left_item) > 4 else ()
                for right_item in right_by_middle.get(int(middle_idx), ()):
                    q_rb, q_p2b, middle_idx_2, right_factor = right_item[:4]
                    right_families = right_item[4] if len(right_item) > 4 else ()
                    out_key = (q_lb, q_p1b, q_p2b, q_rb)
                    out_idx = out_index.get(out_key)
                    if out_idx is not None:
                        families = tuple(
                            sorted(
                                {
                                    str(name)
                                    for name in tuple(left_families) + tuple(right_families)
                                    if name is not None
                                }
                            )
                        )
                        in_terms.append((out_idx, left_factor, right_factor, families))
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
    su2_operator_engine=None,
    su2_moving_environment=None,
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
    :param su2_operator_engine: Persistent packed SU(2) operator owner.
    :param su2_moving_environment: Persistent C++ sweep owner.
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
        su2_operator_engine=su2_operator_engine,
        su2_moving_environment=su2_moving_environment,
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
        basis = operator.basis
        if not basis.compatible_with_layout(norm_operator.basis):
            raise ValueError(
                "Hamiltonian and norm operators selected incompatible "
                "two-site bases."
            )
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
            if isinstance(cache, RenormalizedOperatorStack):
                cache.prepare_miss(cache_key)
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
        reuse_prebuilt_boundary_side=None,
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
        :param reuse_prebuilt_boundary_side: Optional side, ``"left"`` or
            ``"right"``, whose boundary entries are already valid in
            ``renormalized_blocks`` from the previous sweep.  That side is not
            rebuilt here, making the sweep use the stack as a moving
            environment.
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
        if reuse_prebuilt_boundary_side is not None:
            reuse_prebuilt_boundary_side = str(reuse_prebuilt_boundary_side).lower()
            if reuse_prebuilt_boundary_side not in {"left", "right"}:
                raise ValueError(
                    "reuse_prebuilt_boundary_side must be 'left', 'right', or None."
                )
            if renormalized_blocks is None:
                raise ValueError(
                    "Reusing a prebuilt boundary side requires renormalized_blocks."
                )
            if sweep_direction == "lr" and reuse_prebuilt_boundary_side != "right":
                raise ValueError("A left-to-right sweep can only reuse the right boundary side.")
            if sweep_direction == "rl" and reuse_prebuilt_boundary_side != "left":
                raise ValueError("A right-to-left sweep can only reuse the left boundary side.")
        sites = [normalize_site_tensor_layout(site) for site in sites]

        site_layouts = [_tensor_dense_layout(site) for site in sites]
        phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
        sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
            mpo_factors,
            site_layouts=site_layouts,
        )
        lightweight_owner = getattr(
            sparse_mpo_factors[0],
            "normal_complementary_owner",
            None,
        )
        if (
            renormalized_blocks is None
            and lightweight_owner is not None
            and all(
                getattr(factor, "normal_complementary_plan", None) is not None
                and not factor.reduced_terms
                for factor in sparse_mpo_factors
            )
        ):
            from pyqed.qchem.dmrg.backends.reduced import (
                build_su2_normal_complementary_mpo,
            )

            right_dual = bool(
                getattr(
                    sparse_mpo_factors[0],
                    "normal_complementary_right_dual",
                    False,
                )
            )
            sparse_mpo_factors = build_su2_normal_complementary_mpo(
                lightweight_owner,
                fully_reduced=bool(
                    getattr(
                        sparse_mpo_factors[0],
                        "normal_complementary_fully_reduced",
                        False,
                    )
                ),
                materialize_reduced_terms=True,
            )
            for factor in sparse_mpo_factors:
                object.__setattr__(
                    factor,
                    "normal_complementary_right_dual",
                    right_dual,
                )

        rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
        nsites = len(sites)
        build_left = sweep_direction is None or sweep_direction == "rl"
        build_right = sweep_direction is None or sweep_direction == "lr"
        if reuse_prebuilt_boundary_side == "left":
            build_left = False
        elif reuse_prebuilt_boundary_side == "right":
            build_right = False
        if rank_coupled:
            normal_complementary_owner = getattr(
                sparse_mpo_factors[0],
                "normal_complementary_owner",
                None,
            )
            nc_moving_environment = (
                (
                    getattr(
                        renormalized_blocks,
                        "su2_moving_environment",
                        None,
                    )
                    or getattr(
                        renormalized_blocks,
                        "su2_boundary_environment",
                        None,
                    )
                )
                if getattr(
                    sparse_mpo_factors[0],
                    "normal_complementary_plan",
                    None,
                )
                is not None
                else None
            ) or normal_complementary_owner
            initial_left = LeftBlock(
                _initial_left_env_blocks_rank_coupled(site_layouts[0], sparse_mpo_factors[0]),
                rank_coupled=True,
            )
            if build_left:
                if (
                    nc_moving_environment is not None
                    and renormalized_blocks is not None
                ):
                    renormalized_blocks.initialize(
                        "left",
                        0,
                        initial_left,
                        side_table_builders=_renormalized_side_table_builders(
                            rank_coupled=True,
                        ),
                    )
                left_envs = [initial_left]
                for i in range(nsites - 1):
                    left_envs.append(
                        left_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                            moving_environment=nc_moving_environment,
                            parent_bond=i,
                            child_bond=i + 1,
                            numeric_revision=i + 1,
                        )
                    )
            else:
                left_envs = [initial_left] + [None] * (nsites - 1)

            initial_right = RightBlock(
                _initial_right_env_blocks_rank_coupled(site_layouts[-1], sparse_mpo_factors[-1]),
                rank_coupled=True,
            )
            if build_right:
                if (
                    nc_moving_environment is not None
                    and renormalized_blocks is not None
                ):
                    renormalized_blocks.initialize(
                        "right",
                        nsites - 1,
                        initial_right,
                        side_table_builders=_renormalized_side_table_builders(
                            rank_coupled=True,
                        ),
                    )
                right_envs = [initial_right]
                for i in range(nsites - 1, 0, -1):
                    right_envs.append(
                        right_envs[-1].advance(
                            sparse_mpo_factors[i],
                            sites[i],
                            sites[i],
                            moving_environment=nc_moving_environment,
                            parent_bond=i,
                            child_bond=i - 1,
                            numeric_revision=nsites - i,
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
        for bond, block in enumerate(self.right_envs):
            if block is None:
                continue
            entry = self.renormalized_blocks.initialize(
                "right",
                bond,
                block,
                side_table_builders=side_table_builders,
            )
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
        grouped = None
        packed_table = None
        if self.rank_coupled and base_record is not None:
            try:
                from .su2_qchem_plan import pack_rank_coupled_factor_table_from_boundary

                packed_table = pack_rank_coupled_factor_table_from_boundary(
                    getattr(base_record, "packed_table", None),
                    W,
                    side=entry.side,
                    bond=entry.bond,
                    representation=factor_representation,
                )
            except Exception:
                packed_table = None
        if packed_table is None:
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
            packed_table=packed_table,
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
        channel_resolved = _uses_channel_resolved_local_basis(
            two_site_template,
            W1,
            W2,
            rank_coupled=self.rank_coupled,
        )
        basis = (
            TwoSiteBasis.from_channel_tensor(two_site_template)
            if channel_resolved
            else two_site_state_basis(two_site_template, layout=layout)
        )
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
            su2_operator_engine=(
                None
                if self.renormalized_blocks is None
                else self.renormalized_blocks.su2_operator_engine
            ),
            su2_moving_environment=(
                None
                if self.renormalized_blocks is None
                else (
                    self.renormalized_blocks.su2_moving_environment
                    or getattr(
                        self.renormalized_blocks,
                        "su2_boundary_environment",
                        None,
                    )
                    or getattr(W1, "normal_complementary_owner", None)
                )
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
        W1 = self.chain.mpo_factors[bond]
        W2 = self.chain.mpo_factors[bond + 1]
        basis = (
            TwoSiteBasis.from_channel_tensor(two_site_template)
            if _uses_channel_resolved_local_basis(
                two_site_template,
                W1,
                W2,
                rank_coupled=self.chain.rank_coupled,
            )
            else two_site_state_basis(two_site_template, layout=layout)
        )
        if max_dim is not None and int(basis.size) > int(max_dim):
            return None
        cpp_owner = (
            None
            if self.renormalized_blocks is None
            else self.renormalized_blocks.su2_moving_environment
        )
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
            if isinstance(cache, RenormalizedOperatorStack):
                cache.prepare_miss(cache_key)
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
            su2_moving_environment=cpp_owner,
            local_operator_key=(
                f"{self.direction}:{int(bond)}:orthonormal-factor-route"
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
            consumed_moving_entry = self.current_entry
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
                self.current_env = self.current_entry.block
            else:
                self.current_env = self.current_env.advance(
                    self.chain.mpo_factors[bond],
                    left_site,
                    left_site,
                    phys_slices=phys_slices,
                )
                self._store_current_boundary(bond + 1)
            if self.renormalized_blocks is not None:
                self.renormalized_blocks.release_consumed_numeric_tables(
                    "right",
                    bond + 1,
                )
                if self.renormalized_blocks.release_consumed_boundary(
                    "right",
                    bond + 1,
                ):
                    self.chain.right_envs[bond + 1] = None
                if consumed_moving_entry is not None:
                    self.renormalized_blocks.release_consumed_numeric_tables(
                        consumed_moving_entry.side,
                        consumed_moving_entry.bond,
                    )
        else:
            phys_slices = (
                None
                if self.chain.rank_coupled
                else self.chain.site_layouts[bond + 1]["sector_slices"][2]
            )
            consumed_moving_entry = self.current_entry
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
                self.current_env = self.current_entry.block
            else:
                self.current_env = self.current_env.advance(
                    self.chain.mpo_factors[bond + 1],
                    right_site,
                    right_site,
                    phys_slices=phys_slices,
                )
                self._store_current_boundary(bond)
            if self.renormalized_blocks is not None:
                self.renormalized_blocks.release_consumed_numeric_tables(
                    "left",
                    bond,
                )
                if self.renormalized_blocks.release_consumed_boundary(
                    "left",
                    bond,
                ):
                    self.chain.left_envs[bond] = None
                if consumed_moving_entry is not None:
                    self.renormalized_blocks.release_consumed_numeric_tables(
                        consumed_moving_entry.side,
                        consumed_moving_entry.bond,
                    )
        if self.renormalized_blocks is not None:
            operator_engine = self.renormalized_blocks.su2_operator_engine
            if operator_engine is not None:
                operator_engine.release_numeric()
                _release_free_numeric_pages()
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
        if getattr(
            self.chain.mpo_factors[site_index],
            "normal_complementary_plan",
            None,
        ) is not None:
            return _rank_coupled_channel_expectation(
                env.data,
                0 if self.direction == "lr" else 1,
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


def contract_chain_expectation(
    sites,
    mpo_factors,
    *,
    bra_sites=None,
    moving_environment=None,
    site_layouts=None,
):
    """
    Contract ``<bra|MPO|sites>`` for one non-Abelian MPS chain.

    Parameters
    ----------
    sites
        Ket sequence of rank-3 non-Abelian MPS site tensors.
    bra_sites
        Optional bra sequence. Omitting it computes an expectation value.
    mpo_factors
        Sequence of MPO cores (dense or block-sparse ``MPO`` objects), one per
        site.
    moving_environment
        Optional persistent C++ owner for direct normal/complementary
        boundary actions. Omitting it preserves the Python reference path.
    site_layouts
        Optional precomputed dense-layout metadata for ``sites``. Supplying it
        avoids rebuilding identical sector maps when many operators are
        contracted against the same MPS, as in reduced 1-/2-RDM evaluation.

    Returns
    -------
    complex
        Scalar expectation value of the MPO on the provided MPS.
    """
    if len(sites) != len(mpo_factors):
        raise ValueError("contract_chain_expectation requires one MPO core per site tensor.")
    if not sites:
        raise ValueError("contract_chain_expectation requires at least one site tensor.")
    bra_sites = sites if bra_sites is None else list(bra_sites)
    if len(bra_sites) != len(sites):
        raise ValueError("Bra and ket MPS lengths must match.")

    if site_layouts is None:
        site_layouts = [_tensor_dense_layout(site) for site in sites]
    else:
        site_layouts = list(site_layouts)
        if len(site_layouts) != len(sites):
            raise ValueError("site_layouts must contain one layout per MPS site.")
    phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
    sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
        mpo_factors,
        site_layouts=site_layouts,
    )

    rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
    if rank_coupled:
        direct_normal_complementary = bool(
            bra_sites is sites
            and moving_environment is not None
            and getattr(
                sparse_mpo_factors[0],
                "normal_complementary_owner",
                None,
            )
            is moving_environment
        )
        if direct_normal_complementary:
            moving_environment.clear_boundaries()
            env = LeftBlock(
                _initial_left_env_blocks_rank_coupled(
                    site_layouts[0],
                    sparse_mpo_factors[0],
                ),
                rank_coupled=True,
            )
            for idx in range(len(sites)):
                env = env.advance(
                    sparse_mpo_factors[idx],
                    bra_sites[idx],
                    sites[idx],
                    moving_environment=moving_environment,
                    parent_bond=idx,
                    child_bond=idx + 1,
                    numeric_revision=idx + 1,
                )
        else:
            env = _initial_left_env_blocks_rank_coupled(
                site_layouts[0],
                sparse_mpo_factors[0],
            )
            for idx in range(len(sites)):
                env = _contract_from_left_blocks_rank_coupled(
                    sparse_mpo_factors[idx],
                    bra_sites[idx],
                    env,
                    sites[idx],
                )
    else:
        env = _initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0])
        for idx in range(len(sites)):
            env = _contract_from_left_blocks(
                sparse_mpo_factors[idx],
                bra_sites[idx],
                env,
                sites[idx],
                phys_slice_maps[idx],
            )

    value = 0.0 + 0.0j
    if rank_coupled and getattr(
        sparse_mpo_factors[0],
        "normal_complementary_plan",
        None,
    ) is not None:
        return _rank_coupled_channel_expectation(env, 0)
    return _environment_map_expectation(env, rank_coupled=rank_coupled)

def contract_chain_transition(bra_sites, mpo_factors, ket_sites):
    """Contract ``<bra|MPO|ket>`` in the native reduced representation."""
    if len(bra_sites) != len(ket_sites) or len(ket_sites) != len(mpo_factors):
        raise ValueError(
            "contract_chain_transition requires equally sized bra, MPO, and ket chains."
        )
    if not ket_sites:
        raise ValueError("contract_chain_transition requires at least one site tensor.")

    bra_sites = [normalize_site_tensor_layout(site) for site in bra_sites]
    ket_sites = [normalize_site_tensor_layout(site) for site in ket_sites]
    site_layouts = [_tensor_dense_layout(site) for site in ket_sites]
    sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
        mpo_factors,
        site_layouts=site_layouts,
    )
    rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
    if rank_coupled:
        env = _initial_left_env_blocks_rank_coupled(
            site_layouts[0], sparse_mpo_factors[0]
        )
        for core, bra, ket in zip(sparse_mpo_factors, bra_sites, ket_sites):
            env = _contract_from_left_blocks_rank_coupled(core, bra, env, ket)
    else:
        phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
        env = _initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0])
        for core, bra, ket, phys_slices in zip(
            sparse_mpo_factors, bra_sites, ket_sites, phys_slice_maps
        ):
            env = _contract_from_left_blocks(
                core, bra, env, ket, phys_slices
            )
    if rank_coupled and getattr(
        sparse_mpo_factors[0], "normal_complementary_plan", None
    ) is not None:
        return _rank_coupled_channel_expectation(env, 0)
    return _environment_map_expectation(env, rank_coupled=rank_coupled)


@dataclass(frozen=True)
class LocalTransitionPlan:
    """Cached exact contraction plan for a single varying MPS site.

    The fixed half of the chain is contracted once.  Each subsequent
    ``contract`` call only absorbs the varying bra/ket tensor and the shorter
    unfixed half.  Rank-coupled MPO cores remain in their reduced
    Wigner--Eckart representation throughout.
    """

    site: int
    sites: tuple
    mpo_factors: tuple
    phys_slices: tuple
    rank_coupled: bool
    direction: str
    anchor: object

    @classmethod
    def build(cls, sites, mpo_factors, site, *, direction=None):
        if len(sites) != len(mpo_factors):
            raise ValueError(
                "LocalTransitionPlan requires one MPO core per site tensor."
            )
        if not sites:
            raise ValueError("LocalTransitionPlan requires a nonempty chain.")
        site = int(site)
        if site < 0 or site >= len(sites):
            raise IndexError(f"Site {site} is outside a chain of length {len(sites)}.")

        sites = tuple(normalize_site_tensor_layout(tensor) for tensor in sites)
        layouts = tuple(_tensor_dense_layout(tensor) for tensor in sites)
        factors = tuple(
            _normalize_block_sparse_mpo_factors(
                mpo_factors,
                site_layouts=layouts,
            )
        )
        rank_coupled = _is_rank_coupled_chain(factors)
        phys_slices = tuple(layout["sector_slices"][1] for layout in layouts)

        # Cache the longer side so a repeated local transition traverses only
        # the shorter side of the chain.
        if direction is None:
            direction = "lr" if site >= len(sites) - site - 1 else "rl"
        else:
            direction = str(direction).lower()
            if direction not in {"lr", "rl"}:
                raise ValueError("LocalTransitionPlan direction must be 'lr' or 'rl'.")
        if direction == "lr":
            if rank_coupled:
                anchor = LeftBlock(
                    _initial_left_env_blocks_rank_coupled(layouts[0], factors[0]),
                    rank_coupled=True,
                )
            else:
                anchor = LeftBlock(
                    _initial_left_env_blocks(layouts[0], factors[0]),
                    rank_coupled=False,
                )
            for index in range(site):
                anchor = anchor.advance(
                    factors[index],
                    sites[index],
                    sites[index],
                    phys_slices=None if rank_coupled else phys_slices[index],
                )
        else:
            direction = "rl"
            if rank_coupled:
                anchor = RightBlock(
                    _initial_right_env_blocks_rank_coupled(layouts[-1], factors[-1]),
                    rank_coupled=True,
                )
            else:
                anchor = RightBlock(
                    _initial_right_env_blocks(layouts[-1], factors[-1]),
                    rank_coupled=False,
                )
            for index in range(len(sites) - 1, site, -1):
                anchor = anchor.advance(
                    factors[index],
                    sites[index],
                    sites[index],
                    phys_slices=None if rank_coupled else phys_slices[index],
                )

        return cls(
            site=site,
            sites=sites,
            mpo_factors=factors,
            phys_slices=phys_slices,
            rank_coupled=rank_coupled,
            direction=direction,
            anchor=anchor,
        )

    @property
    def cached_sites(self):
        if self.direction == "lr":
            return int(self.site)
        return int(len(self.sites) - self.site - 1)

    @property
    def traversed_sites(self):
        return int(len(self.sites) - self.cached_sites)

    def contract(self, bra_site, ket_site):
        bra_site = normalize_site_tensor_layout(bra_site)
        ket_site = normalize_site_tensor_layout(ket_site)
        physical = None if self.rank_coupled else self.phys_slices[self.site]
        env = self.anchor.advance(
            self.mpo_factors[self.site],
            bra_site,
            ket_site,
            phys_slices=physical,
        )
        if self.direction == "lr":
            for index in range(self.site + 1, len(self.sites)):
                env = env.advance(
                    self.mpo_factors[index],
                    self.sites[index],
                    self.sites[index],
                    phys_slices=(
                        None if self.rank_coupled else self.phys_slices[index]
                    ),
                )
        else:
            for index in range(self.site - 1, -1, -1):
                env = env.advance(
                    self.mpo_factors[index],
                    self.sites[index],
                    self.sites[index],
                    phys_slices=(
                        None if self.rank_coupled else self.phys_slices[index]
                    ),
                )
        if self.rank_coupled and getattr(
            self.mpo_factors[0], "normal_complementary_plan", None
        ) is not None:
            return _rank_coupled_channel_expectation(
                env.data,
                0 if self.direction == "lr" else 1,
            )
        return env.expectation()


@dataclass(frozen=True)
class AdjacentPairTransitionPlan:
    """Cached exact contraction plan for two adjacent varying MPS sites.

    Unlike the generic merged-pair effective operator, this plan keeps the
    intermediate reduced bond explicit.  That distinction matters whenever
    several SU(2) fusion channels have the same four external sector labels.
    The longer fixed side of the chain is contracted once; each call absorbs
    the two varying tensors and only traverses the shorter fixed side.
    """

    bond: int
    sites: tuple
    mpo_factors: tuple
    phys_slices: tuple
    rank_coupled: bool
    direction: str
    anchor: object

    @classmethod
    def build(cls, sites, mpo_factors, bond, *, direction=None):
        if len(sites) != len(mpo_factors):
            raise ValueError(
                "AdjacentPairTransitionPlan requires one MPO core per site tensor."
            )
        if len(sites) < 2:
            raise ValueError("AdjacentPairTransitionPlan requires at least two sites.")
        bond = int(bond)
        if bond < 0 or bond >= len(sites) - 1:
            raise IndexError(
                f"Bond {bond} is outside a chain of length {len(sites)}."
            )

        sites = tuple(normalize_site_tensor_layout(tensor) for tensor in sites)
        layouts = tuple(_tensor_dense_layout(tensor) for tensor in sites)
        factors = tuple(
            _normalize_block_sparse_mpo_factors(
                mpo_factors,
                site_layouts=layouts,
            )
        )
        rank_coupled = _is_rank_coupled_chain(factors)
        phys_slices = tuple(layout["sector_slices"][1] for layout in layouts)

        left_fixed = bond
        right_fixed = len(sites) - bond - 2
        if direction is None:
            direction = "lr" if left_fixed >= right_fixed else "rl"
        else:
            direction = str(direction).lower()
            if direction not in {"lr", "rl"}:
                raise ValueError(
                    "AdjacentPairTransitionPlan direction must be 'lr' or 'rl'."
                )

        if direction == "lr":
            if rank_coupled:
                anchor = LeftBlock(
                    _initial_left_env_blocks_rank_coupled(layouts[0], factors[0]),
                    rank_coupled=True,
                )
            else:
                anchor = LeftBlock(
                    _initial_left_env_blocks(layouts[0], factors[0]),
                    rank_coupled=False,
                )
            for index in range(bond):
                anchor = anchor.advance(
                    factors[index],
                    sites[index],
                    sites[index],
                    phys_slices=None if rank_coupled else phys_slices[index],
                )
        else:
            if rank_coupled:
                anchor = RightBlock(
                    _initial_right_env_blocks_rank_coupled(layouts[-1], factors[-1]),
                    rank_coupled=True,
                )
            else:
                anchor = RightBlock(
                    _initial_right_env_blocks(layouts[-1], factors[-1]),
                    rank_coupled=False,
                )
            for index in range(len(sites) - 1, bond + 1, -1):
                anchor = anchor.advance(
                    factors[index],
                    sites[index],
                    sites[index],
                    phys_slices=None if rank_coupled else phys_slices[index],
                )

        return cls(
            bond=bond,
            sites=sites,
            mpo_factors=factors,
            phys_slices=phys_slices,
            rank_coupled=rank_coupled,
            direction=direction,
            anchor=anchor,
        )

    @property
    def cached_sites(self):
        if self.direction == "lr":
            return int(self.bond)
        return int(len(self.sites) - self.bond - 2)

    @property
    def traversed_sites(self):
        return int(len(self.sites) - self.cached_sites)

    def contract(self, bra_left, bra_right, ket_left, ket_right):
        bra_left = normalize_site_tensor_layout(bra_left)
        bra_right = normalize_site_tensor_layout(bra_right)
        ket_left = normalize_site_tensor_layout(ket_left)
        ket_right = normalize_site_tensor_layout(ket_right)
        physical = lambda index: (
            None if self.rank_coupled else self.phys_slices[index]
        )

        if self.direction == "lr":
            env = self.anchor.advance(
                self.mpo_factors[self.bond],
                bra_left,
                ket_left,
                phys_slices=physical(self.bond),
            )
            env = env.advance(
                self.mpo_factors[self.bond + 1],
                bra_right,
                ket_right,
                phys_slices=physical(self.bond + 1),
            )
            for index in range(self.bond + 2, len(self.sites)):
                env = env.advance(
                    self.mpo_factors[index],
                    self.sites[index],
                    self.sites[index],
                    phys_slices=physical(index),
                )
        else:
            env = self.anchor.advance(
                self.mpo_factors[self.bond + 1],
                bra_right,
                ket_right,
                phys_slices=physical(self.bond + 1),
            )
            env = env.advance(
                self.mpo_factors[self.bond],
                bra_left,
                ket_left,
                phys_slices=physical(self.bond),
            )
            for index in range(self.bond - 1, -1, -1):
                env = env.advance(
                    self.mpo_factors[index],
                    self.sites[index],
                    self.sites[index],
                    phys_slices=physical(index),
                )
        if self.rank_coupled and getattr(
            self.mpo_factors[0], "normal_complementary_plan", None
        ) is not None:
            return _rank_coupled_channel_expectation(
                env.data,
                0 if self.direction == "lr" else 1,
            )
        return env.expectation()
