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

from .contraction import merge_mps_sites, normalize_site_tensor_layout
from .decompose import state_averaged_svd_two_site, svd_two_site
from .linalg import sector_state_weight
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
    optimized_two_site=None,
    solver=None,
    local_operator=None,
    local_solver_kwargs=None,
    prefer_reduced_local_operator=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_rng=None,
    bond_coupling="left",
    max_bond=None,
    max_bond_mode="reduced",
    cutoff=1e-10,
    absorb="right",
    profile=False,
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
    solver
        Optional callable taking the current merged two-site tensor and
        returning either an optimized merged tensor, ``(tensor, objective)``,
        or a dict payload containing the optimized tensor plus simple local
        objective data such as ``energy`` or ``metric``. Mutually exclusive
        with ``optimized_two_site``.
    local_operator
        Optional effective local operator for Davidson-based optimization.
        Mutually exclusive with ``solver`` and ``optimized_two_site``.
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
    merged = _expand_two_site_support(A, B, merge_mps_sites(A, B))
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
        optimized, local_objective = solve_local_two_site(
            merged,
            operator_spec,
            norm_operator=norm_operator,
            canonical_norm=canonical_norm,
            profile=profile,
            **solver_kwargs,
        )
        if profile:
            timing["local_solve"] += time.perf_counter() - t0
    elif optimized_two_site is not None:
        optimized = optimized_two_site
    else:
        optimized = merged

    if not isinstance(optimized, NonabelianTensor) or optimized.rank != 4:
        raise ValueError("two_site_update requires a rank-4 NonabelianTensor as the optimized two-site tensor.")
    if optimized is not merged:
        t0 = time.perf_counter() if profile else None
        optimized = _expand_two_site_support(A, B, optimized)
        if profile:
            timing["optimized_expand"] += time.perf_counter() - t0

    optimized_roots = None
    if isinstance(local_objective, dict):
        optimized_roots = local_objective.pop("optimized_roots", None)
    if optimized_roots is not None:
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
            absorb=absorb,
        )
        if profile:
            timing["svd"] += time.perf_counter() - t0
        local_objective["state_averaged_svd"] = True
    else:
        root_site_pairs = None
        t0 = time.perf_counter() if profile else None
        left, right, singular_values, trunc_err, kept = svd_two_site(
            optimized,
            bond_coupling=bond_coupling,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            cutoff=cutoff,
            absorb=absorb,
        )
        if profile:
            timing["svd"] += time.perf_counter() - t0
    kept_states = sum(
        sector_state_weight(q_mid) * int(block.shape[0])
        for q_mid, block in singular_values.items()
    )
    if isinstance(local_objective, dict):
        local_objective = dict(local_objective)
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
