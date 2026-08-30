#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal two-site update helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors

from .contraction import (
    merge_mps_sites,
    normalize_site_tensor_layout,
    split_mps_sites_from_packed,
)
from .decompose import state_averaged_svd_two_site, svd_two_site
from .linalg import sector_state_weight
from .renormalized import FactorizedRouteMetricBlock, KroneckerMetricBlock
from .solver import LocalOperator, TwoSiteEffectiveH, solve_local_two_site
from .tensor import FusionPipe, FusionPipeEntry, FusionLeg
from .tensor import NonabelianTensor


def _sector_dim_from_data(tensor, axis):
    dims = {}
    for key, block in tensor.data.items():
        dims.setdefault(key[axis], int(block.shape[axis]))
    for sector in tensor.qns[axis]:
        if sector in dims:
            continue
        if axis == 1 and (tensor.metadata or {}).get("physical_basis") == "fully_reduced_su2":
            dims[sector] = 1
            continue
        dim = getattr(sector, "dim", None)
        dims[sector] = int(dim) if dim is not None else 1
    return dims


def _sector_multiplicity(qns, sector):
    return sum(1 for item in qns if item == sector)


def _fuse_sectors(left, right):
    if hasattr(left, "fuse"):
        return tuple(left.fuse(right))
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return tuple(fuse_charge_spin_sectors(left, right))
    raise TypeError(
        f"Cannot fuse sectors of type {type(left).__name__} and {type(right).__name__}."
    )


def _normalize_contracted_channel_values(value):
    if isinstance(value, tuple):
        raw_values = value
    else:
        raw_values = (value,)
    normalized = []
    for item in raw_values:
        if isinstance(item, tuple) and len(item) == 1:
            normalized.append(item[0])
        else:
            normalized.append(item)
    return tuple(normalized)


def _channel_blocks_match_tensor_data(tensor, *, atol=1.0e-12, rtol=1.0e-10):
    """Check that cached intermediate-channel blocks still represent ``data``."""
    metadata = tensor.metadata or {}
    if not metadata.get("contracted_channel_blocks_current", False):
        return False
    aggregate = {}
    for key, block in metadata.get("contracted_channel_blocks", {}).items():
        outer_key = (key[0], key[1], key[3], key[4])
        value = np.asarray(block)
        aggregate[outer_key] = aggregate.get(outer_key, 0) + value
    if set(aggregate) != set(tensor.data):
        return False
    return all(
        np.asarray(aggregate[key]).shape == np.asarray(tensor.data[key]).shape
        and np.allclose(aggregate[key], tensor.data[key], atol=atol, rtol=rtol)
        for key in tensor.data
    )


def _expand_two_site_support(A, B, merged):
    """
    Add zero blocks for all symmetry-allowed two-site keys on the current bond.
    """
    A = normalize_site_tensor_layout(A)
    B = normalize_site_tensor_layout(B)
    left_dims = {sector: _sector_multiplicity(A.qns[0], sector) for sector in set(A.qns[0])}
    right_dims = {sector: _sector_multiplicity(B.qns[2], sector) for sector in set(B.qns[2])}
    phys1_dims = _sector_dim_from_data(A, 1)
    phys2_dims = _sector_dim_from_data(B, 1)

    data = {key: np.array(block, copy=True) for key, block in merged.data.items()}
    channel_blocks_current = bool(
        merged.metadata.get("contracted_channel_blocks_current", False)
    )
    channel_blocks = (
        {
            tuple(key): np.array(block, copy=True)
            for key, block in merged.metadata.get(
                "contracted_channel_blocks",
                {},
            ).items()
        }
        if channel_blocks_current
        else {}
    )
    seeded_channel_map = {}
    channel_map = defaultdict(set)
    for key, value in merged.metadata.get("contracted_channels", {}).items():
        values = _normalize_contracted_channel_values(value)
        seeded_channel_map[key] = values
        channel_map[key].update(values)

    def _allowed_mid_channels_for_key(key):
        q_left, q_phys1, q_phys2, q_right = key
        if (
            q_left not in left_dims
            or q_right not in right_dims
            or q_phys1 not in phys1_dims
            or q_phys2 not in phys2_dims
        ):
            return ()
        fused_left = _fuse_sectors(q_left, q_phys1)
        allowed = [
            q_mid
            for q_mid in fused_left
            if q_right in _fuse_sectors(q_mid, q_phys2)
        ]
        return tuple(sorted(set(allowed)))

    for q_left in set(A.qns[0]):
        for q_phys1 in set(A.qns[1]):
            mids = list(_fuse_sectors(q_left, q_phys1))
            if not mids:
                continue
            for q_mid in mids:
                for q_phys2 in set(B.qns[1]):
                    rights = [q_right for q_right in set(B.qns[2]) if q_right in _fuse_sectors(q_mid, q_phys2)]
                    for q_right in rights:
                        key = (q_left, q_phys1, q_phys2, q_right)
                        channel_map[key].add(q_mid)
                        if key not in data:
                            data[key] = np.zeros(
                                (
                                    left_dims[q_left],
                                    phys1_dims[q_phys1],
                                    phys2_dims[q_phys2],
                                    right_dims[q_right],
                                ),
                                dtype=np.result_type(
                                    *[block.dtype for block in merged.data.values()],
                                    float,
                                ),
                            )

    projected_weight = 0.0
    for key in list(data):
        if key in seeded_channel_map:
            # Preserve canonical merged keys/channels provided by merge metadata.
            continue
        allowed_channels = _allowed_mid_channels_for_key(key)
        if not allowed_channels:
            projected_weight += float(np.linalg.norm(data[key]) ** 2)
            data.pop(key, None)
            channel_map.pop(key, None)
            continue

        current = set(channel_map.get(key, ()))
        filtered = [q_mid for q_mid in allowed_channels if q_mid in current]
        if filtered:
            channel_map[key] = set(filtered)
        else:
            channel_map[key] = set(allowed_channels)

    metadata = merged.metadata.copy()
    contracted_channels = {
        key: tuple(sorted(channels))
        for key, channels in sorted(channel_map.items())
    }
    metadata["contracted_channels"] = contracted_channels
    if channel_blocks_current:
        for key, channels in contracted_channels.items():
            shape = tuple(int(dim) for dim in data[key].shape)
            for q_mid in channels:
                channel_key = (
                    key[0],
                    key[1],
                    q_mid,
                    key[2],
                    key[3],
                )
                channel_blocks.setdefault(
                    channel_key,
                    np.zeros(shape, dtype=np.asarray(data[key]).dtype),
                )
        metadata["contracted_channel_blocks"] = channel_blocks
        metadata["contracted_channel_blocks_current"] = True
    else:
        metadata["contracted_channel_blocks_current"] = False
    if projected_weight > 0.0:
        metadata["projected_invalid_weight"] = projected_weight

    fused_sectors = tuple(sorted({channel for channels in contracted_channels.values() for channel in channels}))
    slot_counts = {sector: 0 for sector in fused_sectors}
    offset_counts = {sector: 0 for sector in fused_sectors}
    pipe_entries = []
    for key in sorted(contracted_channels):
        local_dim = int(np.prod(data[key].shape, dtype=int))
        for fused_sector in contracted_channels[key]:
            pipe_entries.append(
                FusionPipeEntry(
                    child_sectors=tuple(key),
                    fused_sector=fused_sector,
                    slot=slot_counts[fused_sector],
                    offset=offset_counts[fused_sector],
                    local_dim=local_dim,
                    selected_shape=tuple(int(x) for x in data[key].shape),
                )
            )
            slot_counts[fused_sector] += 1
            offset_counts[fused_sector] += local_dim
    if pipe_entries:
        metadata["contracted_fusion_leg"] = FusionLeg(
            child_legs=tuple(range(merged.rank)),
            child_sector_lists=tuple(tuple(leg_qns) for leg_qns in merged.qns),
            child_dirs=tuple(merged.dirs),
            sectors=fused_sectors,
            orientation=1,
            coupling="contracted",
            pipe=FusionPipe.from_entries(
                child_legs=tuple(range(merged.rank)),
                child_sector_lists=tuple(tuple(leg_qns) for leg_qns in merged.qns),
                child_dirs=tuple(merged.dirs),
                fused_sectors=fused_sectors,
                entries=tuple(pipe_entries),
                orientation=1,
                coupling="contracted",
            ),
        )

    return NonabelianTensor(
        data,
        [leg[:] for leg in merged.qns],
        merged.dirs[:],
        fusion_legs=merged.fusion_legs[:],
        metadata=metadata,
    )


def _normalize_solver_output(result):
    """
    Normalize a solver callback result into ``(optimized_tensor, objective_dict)``.
    """
    if isinstance(result, NonabelianTensor):
        return result, {}

    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "Solver tuple results must be ``(optimized_tensor, objective)``."
            )
        optimized, objective = result
        if isinstance(objective, dict):
            return optimized, dict(objective)
        return optimized, {"value": objective}

    if isinstance(result, dict):
        optimized = (
            result.get("optimized")
            or result.get("optimized_two_site")
            or result.get("tensor")
            or result.get("two_site")
        )
        if optimized is None:
            raise ValueError(
                "Solver dict results must include one of: "
                "'optimized', 'optimized_two_site', 'tensor', or 'two_site'."
            )
        objective = {}
        if "objective" in result:
            obj = result["objective"]
            if isinstance(obj, dict):
                objective.update(obj)
            else:
                objective["value"] = obj
        if "local_objective" in result:
            obj = result["local_objective"]
            if isinstance(obj, dict):
                objective.update(obj)
            else:
                objective["value"] = obj
        for key in ("energy", "metric", "variance", "residual", "value"):
            if key in result:
                objective[key] = result[key]
        return optimized, objective

    raise TypeError(
        "Solver must return a NonabelianTensor, (tensor, objective), or a dict payload."
    )


def _normalize_local_operator_spec(spec):
    """
    Normalize a local-operator payload into
    ``(operator, norm_operator, canonical_norm)``.
    """
    if isinstance(spec, TwoSiteEffectiveH):
        return spec.operator, spec.norm_operator, bool(spec.canonical_norm)
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError("Local-operator tuple payloads must be (operator, norm_operator).")
        return spec[0], spec[1], False
    if isinstance(spec, dict) and "operator" in spec:
        return spec["operator"], spec.get("norm_operator"), bool(spec.get("canonical_norm", False))
    return spec, None, False


def _prefer_reduced_local_operator(operator, norm_operator=None):
    """
    Replace tensor operators by cached reduced operators when available.
    """
    if operator is not None and getattr(operator, "aux_packed_matvec", None) is not None:
        operator = LocalOperator(
            packed_matvec=operator.aux_packed_matvec,
            packed_block_matrices=getattr(operator, "packed_block_matrices", None),
            basis=getattr(operator, "basis", None),
            diag=operator.diag,
            name=operator.name,
            identity_like=getattr(operator, "identity_like", False),
        )
    elif operator is not None and getattr(operator, "aux_reduced_matvec", None) is not None:
        operator = LocalOperator(
            reduced_matvec=operator.aux_reduced_matvec,
            basis=getattr(operator, "basis", None),
            diag=operator.diag,
            name=operator.name,
            identity_like=getattr(operator, "identity_like", False),
        )
    if norm_operator is not None and getattr(norm_operator, "aux_packed_matvec", None) is not None:
        norm_operator = LocalOperator(
            packed_matvec=norm_operator.aux_packed_matvec,
            packed_block_matrices=getattr(norm_operator, "packed_block_matrices", None),
            basis=getattr(norm_operator, "basis", None),
            diag=norm_operator.diag,
            name=norm_operator.name,
            identity_like=getattr(norm_operator, "identity_like", False),
        )
    elif norm_operator is not None and getattr(norm_operator, "aux_reduced_matvec", None) is not None:
        norm_operator = LocalOperator(
            reduced_matvec=norm_operator.aux_reduced_matvec,
            basis=getattr(norm_operator, "basis", None),
            diag=norm_operator.diag,
            name=norm_operator.name,
            identity_like=getattr(norm_operator, "identity_like", False),
        )
    return operator, norm_operator


def _component_problem_metric_factors(problem):
    """Recover compact boundary factors retained by a component metric basis."""

    component_basis = getattr(problem, "component_basis", None)
    if component_basis is None:
        return None, None
    basis = component_basis.parent_basis
    entries = tuple(basis)

    def entry_at(index):
        index = int(index)
        for entry_idx, entry in enumerate(entries):
            if int(entry.offset) <= index < int(entry.offset) + int(entry.size):
                return entry_idx, entry
        return None, None

    factor_blocks = {}
    factor_routes = []
    dense_components = []
    for indices, metric_block in zip(
        component_basis.component_indices,
        component_basis.metric_blocks,
    ):
        indices = np.asarray(indices, dtype=int)
        if isinstance(metric_block, KroneckerMetricBlock):
            _entry_idx, entry = entry_at(indices[0])
            if entry is not None and int(indices.size) == int(entry.size):
                factor_blocks[entry.key] = (
                    metric_block.left,
                    metric_block.right,
                )
            continue
        if isinstance(metric_block, np.ndarray):
            _entry_idx, entry = entry_at(indices[0])
            if (
                entry is not None
                and int(indices.size) == int(entry.size)
                and np.array_equal(
                    indices,
                    np.arange(
                        int(entry.offset),
                        int(entry.offset) + int(entry.size),
                    ),
                )
            ):
                d_left, d_phys1, d_phys2, d_right = (
                    int(value) for value in entry.shape
                )
                dense = np.asarray(metric_block).reshape(
                    d_left,
                    d_phys1,
                    d_phys2,
                    d_right,
                    d_left,
                    d_phys1,
                    d_phys2,
                    d_right,
                )
                virtual = np.zeros(
                    (d_left, d_right, d_left, d_right),
                    dtype=dense.dtype,
                )
                for p1 in range(d_phys1):
                    for p2 in range(d_phys2):
                        virtual += dense[:, p1, p2, :, :, p1, p2, :]
                virtual /= float(d_phys1 * d_phys2)
                left = np.einsum("aqbq->ab", virtual, optimize=True)
                right = np.einsum("aqas->qs", virtual, optimize=True)
                trace = float(np.real(np.trace(left)))
                if trace > 1.0e-14:
                    scale = np.sqrt(trace)
                    left = left / scale
                    right = right / scale
                    reconstructed = np.kron(
                        np.kron(
                            np.kron(left, np.eye(d_phys1)),
                            np.eye(d_phys2),
                        ),
                        right,
                    )
                    if np.allclose(
                        reconstructed,
                        metric_block,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    ):
                        factor_blocks[entry.key] = (left, right)
            dense_components.append((indices, np.asarray(metric_block)))
            continue
        if not isinstance(metric_block, FactorizedRouteMetricBlock):
            continue
        for (
            in_slice,
            out_slice,
            _in_shape,
            _out_shape,
            left,
            right,
        ) in metric_block.routes:
            in_global = int(indices[int(in_slice.start)])
            out_global = int(indices[int(out_slice.start)])
            in_idx, in_entry = entry_at(in_global)
            out_idx, out_entry = entry_at(out_global)
            if in_entry is None or out_entry is None:
                continue
            factor_routes.append(
                (
                    int(in_idx),
                    int(out_idx),
                    in_entry,
                    out_entry,
                    left,
                    right,
                )
            )

    dense_edges = []
    for indices, metric_block in dense_components:
        local_position = {
            int(global_index): int(local_index)
            for local_index, global_index in enumerate(indices)
        }
        component_entries = []
        for entry_idx, entry in enumerate(entries):
            global_indices = tuple(
                range(
                    int(entry.offset),
                    int(entry.offset) + int(entry.size),
                )
            )
            if all(index in local_position for index in global_indices):
                component_entries.append(
                    (
                        int(entry_idx),
                        entry,
                        np.asarray(
                            [local_position[index] for index in global_indices],
                            dtype=int,
                        ),
                    )
                )
        for in_idx, in_entry, in_positions in component_entries:
            for out_idx, out_entry, out_positions in component_entries:
                in_physical = (
                    in_entry.key[1:4]
                    if len(in_entry.key) == 5
                    else in_entry.key[1:3]
                )
                out_physical = (
                    out_entry.key[1:4]
                    if len(out_entry.key) == 5
                    else out_entry.key[1:3]
                )
                if in_physical != out_physical:
                    continue
                block = metric_block[np.ix_(out_positions, in_positions)]
                if np.linalg.norm(block.reshape(-1)) <= 1.0e-14:
                    continue
                dl_out, dp1_out, dp2_out, dr_out = (
                    int(value) for value in out_entry.shape
                )
                dl_in, dp1_in, dp2_in, dr_in = (
                    int(value) for value in in_entry.shape
                )
                if dp1_out != dp1_in or dp2_out != dp2_in:
                    continue
                tensor = block.reshape(
                    dl_out,
                    dp1_out,
                    dp2_out,
                    dr_out,
                    dl_in,
                    dp1_in,
                    dp2_in,
                    dr_in,
                )
                virtual = np.zeros(
                    (dl_out, dr_out, dl_in, dr_in),
                    dtype=tensor.dtype,
                )
                for p1 in range(dp1_out):
                    for p2 in range(dp2_out):
                        virtual += tensor[:, p1, p2, :, :, p1, p2, :]
                virtual /= float(dp1_out * dp2_out)
                rearranged = virtual.transpose(0, 2, 1, 3).reshape(
                    dl_out * dl_in,
                    dr_out * dr_in,
                )
                if np.linalg.norm(rearranged) <= 1.0e-14:
                    continue
                left_key = (out_entry.key[0], in_entry.key[0])
                right_key = (out_entry.key[-1], in_entry.key[-1])
                dense_edges.append(
                    {
                        "left_key": left_key,
                        "right_key": right_key,
                        "matrix": rearranged,
                        "left_shape": (dl_out, dl_in),
                        "right_shape": (dr_out, dr_in),
                        "in_idx": in_idx,
                        "out_idx": out_idx,
                        "in_entry": in_entry,
                        "out_entry": out_entry,
                    }
                )

    if dense_edges:
        left_vectors = {}
        right_vectors = {}
        remaining = set(range(len(dense_edges)))
        while remaining:
            progressed = False
            for edge_idx in tuple(remaining):
                edge = dense_edges[edge_idx]
                matrix = edge["matrix"]
                left_key = edge["left_key"]
                right_key = edge["right_key"]
                left_vector = left_vectors.get(left_key)
                right_vector = right_vectors.get(right_key)
                if left_vector is not None:
                    denom = float(np.vdot(left_vector, left_vector).real)
                    if denom <= 1.0e-15:
                        continue
                    candidate = left_vector.conj() @ matrix / denom
                    if right_vector is None:
                        right_vectors[right_key] = candidate
                    remaining.remove(edge_idx)
                    progressed = True
                elif right_vector is not None:
                    denom = float(np.vdot(right_vector, right_vector).real)
                    if denom <= 1.0e-15:
                        continue
                    candidate = matrix @ right_vector.conj() / denom
                    left_vectors[left_key] = candidate
                    remaining.remove(edge_idx)
                    progressed = True
            if progressed:
                continue
            edge_idx = next(iter(remaining))
            edge = dense_edges[edge_idx]
            u, singular_values, vh = np.linalg.svd(
                edge["matrix"],
                full_matrices=False,
            )
            if singular_values.size == 0 or float(singular_values[0]) <= 1.0e-15:
                remaining.remove(edge_idx)
                continue
            scale = np.sqrt(float(singular_values[0]))
            left_vectors[edge["left_key"]] = u[:, 0] * scale
            right_vectors[edge["right_key"]] = vh[0, :] * scale

        for edge in dense_edges:
            left_vector = left_vectors.get(edge["left_key"])
            right_vector = right_vectors.get(edge["right_key"])
            if left_vector is None or right_vector is None:
                continue
            left_vector = np.real_if_close(left_vector, tol=1000)
            right_vector = np.real_if_close(right_vector, tol=1000)
            reconstructed = left_vector[:, None] * right_vector[None, :]
            if not np.allclose(
                reconstructed,
                edge["matrix"],
                rtol=1.0e-9,
                atol=1.0e-11,
            ):
                continue
            factor_routes.append(
                (
                    edge["in_idx"],
                    edge["out_idx"],
                    edge["in_entry"],
                    edge["out_entry"],
                    left_vector.reshape(edge["left_shape"]),
                    right_vector.reshape(edge["right_shape"]),
                )
            )
    return (
        (None if factor_routes else (factor_blocks or None)),
        tuple(factor_routes) if factor_routes else None,
    )


def _project_guess_to_merged_layout(guess, merged):
    """
    Project a cached rank-4 guess tensor onto the current merged layout.

    Bond dimensions can change between sweeps even when the sector keys stay
    the same. This helper keeps only blocks whose shapes still match the
    current merged tensor and zeros everything else, so the result is always
    packable against the current local layout.
    """
    if not isinstance(guess, NonabelianTensor) or guess.rank != 4:
        return guess

    data = {}
    has_overlap = False
    for key, block in merged.data.items():
        guess_block = guess.data.get(key)
        if guess_block is not None and np.asarray(guess_block).shape == np.asarray(block).shape:
            data[key] = np.array(guess_block, copy=True)
            has_overlap = True
        else:
            data[key] = np.zeros_like(block)

    if not has_overlap:
        return None
    projected = NonabelianTensor(
        data,
        [leg[:] for leg in merged.qns],
        merged.dirs[:],
        fusion_legs=merged.fusion_legs[:],
        metadata=merged.metadata.copy(),
    )
    if sum(float(np.linalg.norm(block.reshape(-1))) for block in projected.data.values()) <= 1e-15:
        return None
    return projected


def _seed_zero_block_guess(merged, *, scale, rng):
    """
    Seed tiny amplitudes into zero-valued local blocks of a merged two-site tensor.

    This is intended for local DMRG mixer behavior: it enriches only the
    *initial guess* on the active bond, without perturbing the surrounding MPS
    tensors or the canonical environments.
    """
    if scale <= 0.0:
        return merged.copy()

    data = {}
    seeded = False
    for key, block in merged.data.items():
        arr = np.array(block, copy=True)
        if np.linalg.norm(arr.reshape(-1)) < 1e-14:
            arr += rng.normal(scale=scale, size=arr.shape)
            seeded = True
        data[key] = arr

    if not seeded:
        return merged.copy()
    return NonabelianTensor(
        data,
        [leg[:] for leg in merged.qns],
        merged.dirs[:],
        fusion_legs=merged.fusion_legs[:],
        metadata=merged.metadata.copy(),
    )


def two_site_update(
    A,
    B,
    *,
    merged_two_site=None,
    optimized_two_site=None,
    solver=None,
    local_operator=None,
    local_solver=None,
    post_split=None,
    local_solver_kwargs=None,
    prefer_reduced_local_operator=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_rng=None,
    bond_coupling="left",
    max_bond=None,
    max_bond_mode="reduced",
    cutoff=1e-10,
    retain_sector_topology=False,
    absorb="right",
    profile=False,
    lifecycle_owner=None,
):
    """
    Perform one minimal non-Abelian two-site update step.

    Parameters
    ----------
    A, B
        Neighboring rank-3 site tensors with the standard ``(L, R, P)`` /
        ``(L, P, R)``-compatible non-Abelian layout already used by
        :func:`merge_mps_sites`.
    optimized_two_site
        Optional rank-4 merged two-site tensor to split back into updated site
        tensors. If omitted, the current merged tensor is reused.
    merged_two_site
        Optional precontracted current two-site tensor. The C++ sweep owner uses
        this to avoid repeating the active-bond merge in Python.
    solver
        Optional callable taking the current merged two-site tensor and
        returning either an optimized merged tensor, ``(tensor, objective)``,
        or a dict payload containing the optimized tensor plus simple local
        objective data such as ``energy`` or ``metric``. Mutually exclusive
        with ``optimized_two_site``.
    local_operator
        Optional effective local operator for Davidson-based optimization.
        Mutually exclusive with ``solver`` and ``optimized_two_site``.
    local_solver
        Optional replacement for the Davidson solve. It receives the merged
        tensor, resolved local operator, norm operator, and canonical-norm
        flag. This is used by reduced-space real-time propagators.
    post_split
        Optional callback applied to the split site pair while the same local
        effective operator is still alive. Projector-splitting TDVP uses it for
        the compensating one-site backward evolution.
    local_solver_kwargs
        Optional keyword arguments forwarded to :func:`solve_local_two_site`.
    bond_coupling, max_bond, max_bond_mode, cutoff, absorb
        Passed through to :func:`svd_two_site`.

    Returns
    -------
    dict
        Dictionary with keys ``merged``, ``optimized``, ``left``, ``right``,
        ``singular_values``, ``trunc_err``, ``kept``, and ``local_objective``.
    """
    if not isinstance(A, NonabelianTensor) or not isinstance(B, NonabelianTensor):
        raise TypeError("two_site_update expects NonabelianTensor site tensors.")
    if A.rank != 3 or B.rank != 3:
        raise ValueError("two_site_update expects rank-3 site tensors.")
    n_modes = sum(x is not None for x in (solver, optimized_two_site, local_operator))
    if n_modes > 1:
        raise ValueError("Specify only one of solver, optimized_two_site, or local_operator.")
    if local_solver is not None and local_operator is None:
        raise ValueError("local_solver requires local_operator.")
    if post_split is not None and local_operator is None:
        raise ValueError("post_split requires local_operator.")

    timing = {
        "merge_expand": 0.0,
        "operator_factory": 0.0,
        "local_solve": 0.0,
        "optimized_expand": 0.0,
        "svd": 0.0,
        "total": 0.0,
    } if profile else None
    total_t0 = time.perf_counter() if profile else None

    t0 = time.perf_counter() if profile else None
    if merged_two_site is not None:
        if (
            not isinstance(merged_two_site, NonabelianTensor)
            or merged_two_site.rank != 4
        ):
            raise ValueError(
                "merged_two_site must be a rank-4 NonabelianTensor."
            )
        current_merged = merged_two_site
    else:
        current_merged = merge_mps_sites(A, B)
    merged = _expand_two_site_support(A, B, current_merged)
    if profile:
        timing["merge_expand"] += time.perf_counter() - t0

    local_objective = {}
    local_guess_used = False
    if solver is not None:
        t0 = time.perf_counter() if profile else None
        optimized, local_objective = _normalize_solver_output(solver(merged))
        if profile:
            timing["local_solve"] += time.perf_counter() - t0
    elif local_operator is not None:
        solver_kwargs = dict(local_solver_kwargs or {})
        if isinstance(solver_kwargs.get("guess"), NonabelianTensor):
            projected_guess = _project_guess_to_merged_layout(solver_kwargs["guess"], merged)
            if projected_guess is None:
                solver_kwargs.pop("guess", None)
            else:
                solver_kwargs["guess"] = projected_guess
                local_guess_used = True
        elif mixer_zero_block_noise_scale > 0.0:
            if mixer_rng is None:
                mixer_rng = np.random.default_rng()
            solver_kwargs["guess"] = _seed_zero_block_guess(
                merged,
                scale=float(mixer_zero_block_noise_scale),
                rng=mixer_rng,
            )
            local_guess_used = True
        t0 = time.perf_counter() if profile else None
        operator_spec = (
            local_operator(merged)
            if getattr(local_operator, "_is_local_operator_factory", False)
            else local_operator
        )
        root_target_operator = solver_kwargs.get("root_target_operator")
        if getattr(root_target_operator, "_is_local_operator_factory", False):
            solver_kwargs["root_target_operator"] = root_target_operator(merged)
        if profile:
            timing["operator_factory"] += time.perf_counter() - t0
        operator_spec, norm_operator, canonical_norm = _normalize_local_operator_spec(operator_spec)
        if prefer_reduced_local_operator:
            operator_spec, norm_operator = _prefer_reduced_local_operator(
                operator_spec,
                norm_operator,
            )
        t0 = time.perf_counter() if profile else None
        if local_solver is None:
            optimized, local_objective = solve_local_two_site(
                merged,
                operator_spec,
                norm_operator=norm_operator,
                canonical_norm=canonical_norm,
                profile=profile,
                **solver_kwargs,
            )
        else:
            optimized, local_objective = local_solver(
                merged,
                operator_spec,
                norm_operator=norm_operator,
                canonical_norm=canonical_norm,
                profile=profile,
                **solver_kwargs,
            )
        if lifecycle_owner is not None:
            lifecycle_owner.release_operator_numeric()
            lifecycle_owner.clear_local_operator()
        if profile:
            timing["local_solve"] += time.perf_counter() - t0
    elif optimized_two_site is not None:
        optimized = optimized_two_site
    else:
        optimized = merged

    if not isinstance(optimized, NonabelianTensor) or optimized.rank != 4:
        raise ValueError("two_site_update requires a rank-4 NonabelianTensor as the optimized two-site tensor.")
    split_owner = (
        None
        if lifecycle_owner is None
        else getattr(lifecycle_owner, "su2_moving_environment", None)
    )
    cpp_bond_transaction = bool(
        split_owner is not None
        and hasattr(split_owner, "stage_bond_update")
    )
    if lifecycle_owner is not None and not cpp_bond_transaction:
        lifecycle_owner.mark_bond_solved()
    if (
        optimized is not merged
        and not bool(
            local_objective.get("cpp_active_solution_owned", False)
        )
    ):
        t0 = time.perf_counter() if profile else None
        optimized = _expand_two_site_support(A, B, optimized)
        if profile:
            timing["optimized_expand"] += time.perf_counter() - t0

    optimized_roots = None
    if isinstance(local_objective, dict):
        optimized_roots = local_objective.pop("optimized_roots", None)
    cpp_split_result = None
    cpp_split_staged = False
    direct_cpp_split = (
        optimized_roots is None
        and isinstance(local_objective, dict)
        and bool(
            local_objective.get("direct_complementary_action_executor", False)
        )
        and split_owner is not None
        and callable(
            getattr(split_owner, "active_bond_solution_ready", None)
        )
        and callable(
            getattr(split_owner, "split_active_bond_solution", None)
        )
        and split_owner.active_bond_solution_ready()
    )
    if direct_cpp_split:
        root_site_pairs = None
        t0 = time.perf_counter() if profile else None
        cpp_split_result = split_owner.split_active_bond_solution(
            cutoff=float(cutoff),
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            retain_sector_topology=bool(retain_sector_topology),
            absorb=absorb,
        )
        left, right, singular_values = split_mps_sites_from_packed(
            A,
            B,
            cpp_split_result,
        )
        active_bond = int(split_owner.active_bond)
        split_owner.record_cpp_split_site(
            active_bond,
            left,
            cpp_split_result["left_revision"],
        )
        split_owner.record_cpp_split_site(
            active_bond + 1,
            right,
            cpp_split_result["right_revision"],
        )
        trunc_err = float(cpp_split_result["truncation_error"])
        kept = int(np.sum(cpp_split_result["bond_dims"]))
        cpp_split_staged = True
        local_objective["cpp_active_bond_split"] = True
        if profile:
            timing["svd"] += time.perf_counter() - t0

    elif optimized_roots is not None:
        t0 = time.perf_counter() if profile else None
        optimized_roots = [
            _expand_two_site_support(A, B, root) if root is not merged else root
            for root in optimized_roots
        ]
        state_average_weights = local_objective.get("state_average_weights")
        if state_average_weights is not None:
            state_average_weights = np.asarray(state_average_weights, dtype=float).reshape(-1)
            if state_average_weights.size == len(optimized_roots):
                active = np.abs(state_average_weights) > 1.0e-14
                if np.any(active) and not np.all(active):
                    local_objective["optimized_root_candidates"] = int(len(optimized_roots))
                    optimized_roots = [
                        root for root, keep in zip(optimized_roots, active)
                        if bool(keep)
                    ]
                    state_average_weights = state_average_weights[active]
                    state_average_weights = state_average_weights / np.sum(state_average_weights)
                    local_objective["state_average_weights"] = [
                        float(weight) for weight in state_average_weights
                    ]
        left, right, singular_values, trunc_err, kept, root_site_pairs = state_averaged_svd_two_site(
            optimized_roots,
            local_objective.get("state_average_weights"),
            bond_coupling=bond_coupling,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            cutoff=cutoff,
            retain_sector_topology=retain_sector_topology,
            absorb=absorb,
        )
        if profile:
            timing["svd"] += time.perf_counter() - t0
        local_objective["state_averaged_svd"] = True
    else:
        root_site_pairs = None
        t0 = time.perf_counter() if profile else None
        # Component metrics belong to the transient Davidson coordinates.
        # Feeding them into only the bonds that use a component-local solver
        # creates a mixed persistent MPS gauge when a larger bond falls back
        # to the generalized solver.  Always split the returned parent tensor
        # in the chain's reduced canonical metric.
        left, right, singular_values, trunc_err, kept = svd_two_site(
            optimized,
            bond_coupling=bond_coupling,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            cutoff=cutoff,
            retain_sector_topology=retain_sector_topology,
            absorb=absorb,
            sweep_engine=(
                None
                if lifecycle_owner is None
                else getattr(lifecycle_owner, "su2_moving_environment", None)
            ),
        )
        if profile:
            timing["svd"] += time.perf_counter() - t0

    if post_split is not None:
        left, right, post_objective = post_split(
            left,
            right,
            operator_spec,
            norm_operator=norm_operator,
            canonical_norm=canonical_norm,
        )
        if post_objective:
            local_objective.update(post_objective)

    if local_operator is not None:
        del operator_spec, norm_operator
    kept_states = sum(
        sector_state_weight(q_mid) * int(block.shape[0])
        for q_mid, block in singular_values.items()
    )
    if lifecycle_owner is not None:
        if cpp_bond_transaction and not cpp_split_staged:
            try:
                split_owner.stage_bond_update(
                    left,
                    right,
                    kept_states=int(kept_states),
                    truncation_seconds=(
                        float(timing["svd"]) if profile else 0.0
                    ),
                )
            except ValueError as error:
                maximum_imaginary = max(
                    (
                        float(np.max(np.abs(np.asarray(block).imag)))
                        for tensor in (left, right)
                        for block in tensor.data.values()
                        if np.iscomplexobj(block) and np.asarray(block).size
                    ),
                    default=0.0,
                )
                raise ValueError(
                    "C++ split-site staging failed at bond "
                    f"{int(split_owner.active_bond)} "
                    f"(maximum imaginary value {maximum_imaginary:.3e})."
                ) from error
        elif not cpp_split_staged and (
            split_owner is not None
            and hasattr(split_owner, "install_split_site")
        ):
            active_bond = int(split_owner.active_bond)
            split_owner.install_split_site(active_bond, left)
            split_owner.install_split_site(active_bond + 1, right)
        if not cpp_bond_transaction:
            lifecycle_owner.mark_bond_split(
                int(kept_states),
                float(timing["svd"]) if profile else 0.0,
            )
    if isinstance(local_objective, dict):
        local_objective = dict(local_objective)
        if profile and optimized_roots is None:
            reconstructed = merge_mps_sites(left, right)
            optimized_blocks = (
                optimized.metadata.get("contracted_channel_blocks", {})
                if optimized.metadata.get(
                    "contracted_channel_blocks_current",
                    False,
                )
                else optimized.data
            )
            reconstructed_blocks = (
                reconstructed.metadata.get("contracted_channel_blocks", {})
                if reconstructed.metadata.get(
                    "contracted_channel_blocks_current",
                    False,
                )
                else reconstructed.data
            )
            difference_sq = 0.0
            reference_sq = 0.0
            for key in set(optimized_blocks) | set(reconstructed_blocks):
                optimized_block = np.asarray(optimized_blocks.get(key, 0.0))
                reconstructed_block = np.asarray(
                    reconstructed_blocks.get(key, 0.0)
                )
                difference_sq += float(
                    np.linalg.norm(
                        reconstructed_block - optimized_block
                    ) ** 2
                )
                reference_sq += float(np.linalg.norm(optimized_block) ** 2)
            local_objective["split_reconstruction_error"] = float(
                np.sqrt(difference_sq)
            )
            local_objective["split_relative_reconstruction_error"] = float(
                np.sqrt(difference_sq / max(reference_sq, 1.0e-300))
            )
        local_objective.setdefault(
            "metric_weighted_split",
            bool(
                left.metadata.get("canonical_metric") == "factorized_boundary"
                or right.metadata.get("canonical_metric")
                == "factorized_boundary"
            ),
        )
        local_objective.setdefault("trunc_err", float(trunc_err))
        if isinstance(kept, dict):
            objective_kept = {key: list(value) for key, value in kept.items()}
        else:
            objective_kept = int(kept)
        local_objective.setdefault("kept", objective_kept)
        local_objective.setdefault("kept_states", int(kept_states))
    if local_guess_used:
        local_objective = dict(local_objective)
        local_objective["local_guess_used"] = True
    if profile:
        timing["total"] = time.perf_counter() - total_t0
        local_objective = dict(local_objective)
        local_objective["update_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
    return {
        "merged": merged,
        "optimized": optimized,
        "optimized_roots": optimized_roots,
        "root_site_pairs": root_site_pairs,
        "left": left,
        "right": right,
        "singular_values": singular_values,
        "trunc_err": trunc_err,
        "kept": kept,
        "kept_states": kept_states,
        "local_objective": local_objective,
        "local_guess_used": local_guess_used,
    }
