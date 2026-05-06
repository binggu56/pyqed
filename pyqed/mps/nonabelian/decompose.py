#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced decompositions for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

import numpy as np

from .basis import BondBasis
from .contraction import split_legs
from .coupling import normalize_coupling_scheme, reduced_bond_space
from .linalg import (
    ReducedProjectedSVD,
    normalize_max_bond_mode,
    project_reduced_sector,
    truncate_reduced_svds,
)
from .tensor import FusionLeg, FusionPipe, FusionPipeEntry, NonabelianTensor


def _bond_basis_from_singular_values(singular_values, *, direction, name=None):
    sectors = tuple(sorted(singular_values))
    dims = {}
    for sector in sectors:
        values = np.asarray(singular_values[sector])
        if values.ndim == 0:
            dim = 1
        else:
            dim = int(values.shape[0])
        if dim > 0:
            dims[sector] = dim
    sectors = tuple(sector for sector in sectors if sector in dims)
    if not sectors:
        raise ValueError("Cannot build a bond basis from an empty singular-value layout.")
    return BondBasis(sectors=sectors, dims=dims, direction=direction, name=name)


def _apply_singular_values_left(U_tensor, singular_values):
    new_data = {}
    for (qL, qP, qM), block in U_tensor.data.items():
        if qM in singular_values:
            new_block = np.tensordot(block, singular_values[qM], axes=([2], [0]))
            new_data[(qL, qP, qM)] = new_block
    return NonabelianTensor(
        new_data,
        [leg_qns[:] for leg_qns in U_tensor.qns],
        U_tensor.dirs[:],
        fusion_legs=U_tensor.fusion_legs[:],
        metadata=U_tensor.metadata.copy(),
    )


def _apply_singular_values_right(V_tensor, singular_values):
    new_data = {}
    for (qM, qP, qR), block in V_tensor.data.items():
        if qM in singular_values:
            new_block = np.tensordot(singular_values[qM], block, axes=([1], [0]))
            new_data[(qM, qP, qR)] = new_block
    return NonabelianTensor(
        new_data,
        [leg_qns[:] for leg_qns in V_tensor.qns],
        V_tensor.dirs[:],
        fusion_legs=V_tensor.fusion_legs[:],
        metadata=V_tensor.metadata.copy(),
    )


def _internal_bond_entries(two_site):
    contracted_leg = two_site.metadata.get("contracted_fusion_leg")
    if isinstance(contracted_leg, FusionLeg) and contracted_leg.pipe is not None:
        return tuple(
            (entry.child_sectors, entry.fused_sector)
            for entry in contracted_leg.pipe.entries
        )

    contracted_channels = two_site.metadata.get("contracted_channels")
    if contracted_channels:
        entries = []
        for key, q_mid_tuple in contracted_channels.items():
            if q_mid_tuple is None or len(q_mid_tuple) != 1:
                raise ValueError(
                    f"Expected a single contracted bond sector for key {key!r}, got {q_mid_tuple!r}."
                )
            entries.append((tuple(key), q_mid_tuple[0]))
        return tuple(entries)

    raise ValueError(
        "svd_two_site requires contracted bond metadata from merge_mps_sites/tensordot."
    )


def _get_site_bond_layout(two_site, *, side, axis):
    """
    Recover stored reduced bond-layout metadata for one site/bond axis.
    """
    meta = two_site.metadata or {}
    if "left_metadata" in meta or "right_metadata" in meta:
        site_meta = meta.get("left_metadata", {}) if side == "left" else meta.get("right_metadata", {})
    else:
        site_meta = meta
    layouts = site_meta.get("bond_layouts", {})
    return layouts.get(axis, {})


def _build_side_pipe(entries, q_mid, *, side, coupling="left", source_layouts=None):
    """
    Build a transient FusionPipe describing row/column packing for one bond sector.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    if side == "left":
        child_legs = (0, 1)
        child_sectors_fn = lambda key: (key[0], key[1])
        selected_shape_fn = lambda block: (block.shape[0], block.shape[1])
        child_dirs = (-1, 1)
    else:
        child_legs = (2, 3)
        child_sectors_fn = lambda key: (key[2], key[3])
        selected_shape_fn = lambda block: (block.shape[2], block.shape[3])
        child_dirs = (1, 1)

    child_sector_lists = tuple(
        tuple(sorted({child_sectors_fn(key)[i] for key, _ in entries}))
        for i in range(2)
    )

    if source_layouts is not None:
        source = source_layouts.get(q_mid)
        if source is not None:
            source_pipe = source["pipe"]
            if (
                source_pipe.child_legs == child_legs
                and source_pipe.child_sector_lists == child_sector_lists
                and source_pipe.child_dirs == child_dirs
            ):
                return source_pipe, source["basis_map"], source.get("channel_map", {})

    coupling = normalize_coupling_scheme(coupling, default="left")
    packed = {}
    for key, block in entries:
        combo = child_sectors_fn(key)
        selected_shape = selected_shape_fn(block)
        packed.setdefault((combo, selected_shape), selected_shape)

    offset = 0
    pipe_entries = []
    basis_map = {}
    channel_map = {}
    for child_sectors, selected_shape in sorted(packed):
        product_dim = int(np.prod(selected_shape, dtype=int))
        # The surrounding MPS/MPO legs are already reduced multiplet spaces, not
        # explicit uncoupled m-bases. Treat the side spaces of svd_two_site as a
        # pure product packing of the current block shapes; attempting another
        # CG reduction here can silently drop valid channels during the split.
        local_dim = product_dim
        pipe_entries.append(
            FusionPipeEntry(
                child_sectors=child_sectors,
                fused_sector=q_mid,
                slot=0,
                offset=offset,
                local_dim=local_dim,
                selected_shape=selected_shape,
            )
        )
        basis_map[(child_sectors, selected_shape, 0)] = np.eye(local_dim, dtype=float)
        channel_map[(child_sectors, selected_shape, 0)] = None
        offset += local_dim

    return (
        FusionPipe.from_entries(
            child_legs=child_legs,
            child_sector_lists=child_sector_lists,
            child_dirs=child_dirs,
            fused_sectors=(q_mid,),
            entries=tuple(pipe_entries),
            orientation=1,
            coupling=coupling,
        ),
        basis_map,
        channel_map,
    )


def _merge_side_pipes(
    layouts,
    *,
    sectors,
    side,
    child_legs=None,
    child_sector_lists=None,
    child_dirs=None,
):
    if side == "left":
        default_child_legs = (0, 1)
        default_child_sector_lists = (tuple(), tuple())
        default_child_dirs = (-1, 1)
    elif side == "right":
        default_child_legs = (2, 3)
        default_child_sector_lists = (tuple(), tuple())
        default_child_dirs = (1, 1)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    if child_legs is None:
        child_legs = default_child_legs
    if child_sector_lists is None:
        child_sector_lists = default_child_sector_lists
    if child_dirs is None:
        child_dirs = default_child_dirs

    entries = []
    coupling = None
    for sector in sectors:
        layout = layouts.get(sector)
        if layout is None:
            continue
        pipe = layout["pipe"]
        coupling = pipe.coupling
        entries.extend(pipe.entries_for_sector(sector))

    return FusionPipe.from_entries(
        child_legs=child_legs,
        child_sector_lists=child_sector_lists,
        child_dirs=child_dirs,
        fused_sectors=tuple(sectors),
        entries=tuple(entries),
        orientation=1,
        coupling=coupling or "left",
    )


def svd_two_site(
    two_site,
    max_bond=None,
    cutoff=1e-10,
    absorb="right",
    bond_coupling="left",
    max_bond_mode="reduced",
):
    """
    Reduced SVD/truncation helper for a merged two-site tensor.
    """
    if not isinstance(two_site, NonabelianTensor) or two_site.rank != 4:
        raise ValueError("svd_two_site expects a rank-4 NonabelianTensor.")
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")
    max_bond_mode = normalize_max_bond_mode(max_bond_mode, default="reduced")

    blocks_by_mid = {}
    bond_entries = dict(_internal_bond_entries(two_site))
    for key, block in two_site.data.items():
        q_mid = bond_entries.get(key)
        if q_mid is None:
            raise ValueError(f"Missing contracted bond sector for key {key!r}.")
        blocks_by_mid.setdefault(q_mid, []).append((key, block))

    sector_svds = {}
    left_source_layouts = _get_site_bond_layout(two_site, side="left", axis=1)
    right_source_layouts = _get_site_bond_layout(two_site, side="right", axis=0)
    left_output_layouts = {}
    right_output_layouts = {}

    bond_coupling = normalize_coupling_scheme(bond_coupling, default="left")

    for q_mid, entries in blocks_by_mid.items():
        left_pipe, left_basis_map, _left_channel_map = _build_side_pipe(
            entries,
            q_mid,
            side="left",
            coupling=bond_coupling,
            source_layouts=left_source_layouts,
        )
        right_pipe, right_basis_map, _right_channel_map = _build_side_pipe(
            entries,
            q_mid,
            side="right",
            coupling=bond_coupling,
            source_layouts=right_source_layouts,
        )
        left_output_layouts[q_mid] = {
            "pipe": left_pipe,
            "basis_map": left_basis_map,
            "channel_map": _left_channel_map,
        }
        right_output_layouts[q_mid] = {
            "pipe": right_pipe,
            "basis_map": right_basis_map,
            "channel_map": _right_channel_map,
        }
        reduced_sector = project_reduced_sector(
            entries,
            q_mid,
            left_pipe,
            right_pipe,
            left_basis_map,
            right_basis_map,
        )
        svd_result = reduced_sector.svd(full_matrices=False)
        sector_svds[q_mid] = svd_result

    truncation = truncate_reduced_svds(
        sector_svds,
        cutoff=cutoff,
        max_bond=max_bond,
        mode=max_bond_mode,
    )
    kept = {
        q_mid: list(idxs)
        for q_mid, idxs in truncation.kept_indices_by_sector.items()
    }

    singular_values = truncation.singular_values_by_sector()
    right_bond_basis = _bond_basis_from_singular_values(
        singular_values,
        direction=1,
        name="svd-right-bond",
    )
    left_bond_basis = _bond_basis_from_singular_values(
        singular_values,
        direction=-1,
        name="svd-left-bond",
    )
    bond_qns = truncation.bond_qns
    right_split_basis_map = {}

    kept_bond_sectors = tuple(sorted(singular_values))
    slot_counts = {sector: 0 for sector in kept_bond_sectors}
    pipe_entries = []
    for q_mid, idxs in sorted(kept.items()):
        for _idx in idxs:
            pipe_entries.append(
                FusionPipeEntry(
                    child_sectors=(q_mid,),
                    fused_sector=q_mid,
                    slot=slot_counts[q_mid],
                    offset=slot_counts[q_mid],
                    local_dim=1,
                    selected_shape=(1,),
                )
            )
            slot_counts[q_mid] += 1
    bond_pipe = FusionPipe.from_entries(
        child_legs=(0,),
        child_sector_lists=(tuple(kept_bond_sectors),),
        child_dirs=(1,),
        fused_sectors=kept_bond_sectors,
        entries=tuple(pipe_entries),
        orientation=1,
        coupling=bond_coupling,
    )
    bond_leg = FusionLeg(
        child_legs=(0,),
        child_sector_lists=(tuple(kept_bond_sectors),),
        child_dirs=(1,),
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=bond_coupling,
        pipe=bond_pipe,
    )

    left_child_sector_lists = (tuple(two_site.qns[0]), tuple(two_site.qns[1]))
    left_child_dirs = (two_site.dirs[0], two_site.dirs[1])
    right_child_sector_lists = (tuple(two_site.qns[2]), tuple(two_site.qns[3]))
    right_child_dirs = (two_site.dirs[2], two_site.dirs[3])
    left_fused_pipe = _merge_side_pipes(
        left_output_layouts,
        sectors=kept_bond_sectors,
        side="left",
        child_legs=(0, 1),
        child_sector_lists=left_child_sector_lists,
        child_dirs=left_child_dirs,
    )
    left_fused_leg = FusionLeg(
        child_legs=(0, 1),
        child_sector_lists=left_child_sector_lists,
        child_dirs=left_child_dirs,
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=left_fused_pipe.coupling,
        pipe=left_fused_pipe,
    )
    right_fused_pipe = _merge_side_pipes(
        right_output_layouts,
        sectors=kept_bond_sectors,
        side="right",
        child_legs=(2, 3),
        child_sector_lists=right_child_sector_lists,
        child_dirs=right_child_dirs,
    )
    right_fused_leg = FusionLeg(
        child_legs=(2, 3),
        child_sector_lists=right_child_sector_lists,
        child_dirs=right_child_dirs,
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=right_fused_pipe.coupling,
        pipe=right_fused_pipe,
    )

    left_reduced = {}
    left_split_basis_map = {}
    right_reduced = {}
    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
        svd_result = sector_svds[q_mid]
        left_reduced[(q_mid, q_mid)] = svd_result.left_matrix(idxs)
        right_reduced[(q_mid, q_mid)] = svd_result.right_matrix(idxs)
        for entry in svd_result.left_entries:
            left_split_basis_map[(entry.child_sectors, q_mid, entry.slot)] = svd_result.left_basis_map[
                (entry.child_sectors, entry.selected_shape, entry.slot)
            ]
        for entry in svd_result.right_entries:
            right_split_basis_map[(entry.child_sectors, q_mid, entry.slot)] = svd_result.right_basis_map[
                (entry.child_sectors, entry.selected_shape, entry.slot)
            ]

    left_fused_tensor = NonabelianTensor(
        left_reduced,
        [list(kept_bond_sectors), bond_qns],
        [1, 1],
        fusion_legs=[left_fused_leg, bond_leg],
        metadata={"split_basis_maps": {0: left_split_basis_map}},
    )
    U_split = split_legs(left_fused_tensor, 0)
    U_tensor = NonabelianTensor(
        U_split.data,
        U_split.qns,
        U_split.dirs,
        fusion_legs=[two_site.fusion_legs[0], two_site.fusion_legs[1], bond_leg],
        metadata={},
    )
    right_fused_tensor = NonabelianTensor(
        right_reduced,
        [bond_qns, list(kept_bond_sectors)],
        [-1, 1],
        fusion_legs=[bond_leg, right_fused_leg],
        metadata={"split_basis_maps": {1: right_split_basis_map}},
    )
    V_split = split_legs(right_fused_tensor, 1)
    V_tensor = NonabelianTensor(
        V_split.data,
        V_split.qns,
        V_split.dirs,
        fusion_legs=[bond_leg, two_site.fusion_legs[2], two_site.fusion_legs[3]],
        metadata={},
    )

    if absorb == "left":
        A_tensor = _apply_singular_values_left(U_tensor, singular_values)
        B_tensor = V_tensor
    else:
        A_tensor = U_tensor
        B_tensor = _apply_singular_values_right(V_tensor, singular_values)

    A_tensor = NonabelianTensor(
        A_tensor.data,
        A_tensor.qns,
        A_tensor.dirs,
        fusion_legs=A_tensor.fusion_legs,
        metadata={
            "svd_role": "left",
            "source": "svd_two_site",
            "bond_layouts": {2: left_output_layouts},
            "bond_bases": {2: right_bond_basis},
        },
    )
    B_tensor = NonabelianTensor(
        B_tensor.data,
        B_tensor.qns,
        B_tensor.dirs,
        fusion_legs=B_tensor.fusion_legs,
        metadata={
            "svd_role": "right",
            "source": "svd_two_site",
            "bond_layouts": {0: right_output_layouts},
            "bond_bases": {0: left_bond_basis},
        },
    )

    return A_tensor, B_tensor, singular_values, truncation.trunc_err, truncation.kept


def state_averaged_svd_two_site(
    two_site_roots,
    weights,
    *,
    max_bond=None,
    cutoff=1e-10,
    absorb="right",
    bond_coupling="left",
    max_bond_mode="reduced",
):
    """
    State-averaged reduced SVD/truncation for optimized two-site root tensors.

    The weighted reduced density matrix selects one shared bond basis. The first
    root is propagated through that shared basis, matching the existing Abelian
    state-average convention in this codebase.
    """
    roots = list(two_site_roots)
    if not roots:
        raise ValueError("state_averaged_svd_two_site requires at least one root tensor.")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != len(roots):
        raise ValueError("State-average weights must match the number of root tensors.")
    weight_sum = float(np.sum(weights))
    if abs(weight_sum) <= 1e-15:
        raise ValueError("State-average weights must not sum to zero.")
    weights = weights / weight_sum

    ref = roots[0]
    if not isinstance(ref, NonabelianTensor) or ref.rank != 4:
        raise ValueError("state_averaged_svd_two_site expects rank-4 NonabelianTensor roots.")
    for root in roots[1:]:
        if not isinstance(root, NonabelianTensor) or root.rank != 4:
            raise ValueError("state_averaged_svd_two_site expects rank-4 NonabelianTensor roots.")
        if root.qns != ref.qns or root.dirs != ref.dirs:
            raise ValueError("State-average roots must share the same reduced layout.")
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")
    max_bond_mode = normalize_max_bond_mode(max_bond_mode, default="reduced")

    blocks_by_mid = {}
    mid_by_key = {}
    for root in roots:
        bond_entries = dict(_internal_bond_entries(root))
        for key, block in root.data.items():
            q_mid = bond_entries.get(key)
            if q_mid is None:
                raise ValueError(f"Missing contracted bond sector for key {key!r}.")
            previous_mid = mid_by_key.setdefault(key, q_mid)
            if previous_mid != q_mid:
                raise ValueError(
                    f"State-average roots assign key {key!r} to inconsistent bond sectors."
                )
            entries = blocks_by_mid.setdefault(q_mid, {})
            if key in entries and entries[key].shape != block.shape:
                raise ValueError(
                    f"State-average roots use inconsistent block shapes for key {key!r}."
                )
            entries.setdefault(key, block)

    sector_svds = {}
    root_matrices_by_sector = {}
    left_source_layouts = _get_site_bond_layout(ref, side="left", axis=1)
    right_source_layouts = _get_site_bond_layout(ref, side="right", axis=0)
    left_output_layouts = {}
    right_output_layouts = {}
    bond_coupling = normalize_coupling_scheme(bond_coupling, default="left")

    for q_mid, block_map in blocks_by_mid.items():
        ref_entries = list(block_map.items())
        left_pipe, left_basis_map, left_channel_map = _build_side_pipe(
            ref_entries,
            q_mid,
            side="left",
            coupling=bond_coupling,
            source_layouts=left_source_layouts,
        )
        right_pipe, right_basis_map, right_channel_map = _build_side_pipe(
            ref_entries,
            q_mid,
            side="right",
            coupling=bond_coupling,
            source_layouts=right_source_layouts,
        )
        left_output_layouts[q_mid] = {
            "pipe": left_pipe,
            "basis_map": left_basis_map,
            "channel_map": left_channel_map,
        }
        right_output_layouts[q_mid] = {
            "pipe": right_pipe,
            "basis_map": right_basis_map,
            "channel_map": right_channel_map,
        }

        matrices = []
        projection0 = None
        for root_idx, root in enumerate(roots):
            entries = [(key, root.data.get(key, np.zeros_like(block))) for key, block in ref_entries]
            projection = project_reduced_sector(
                entries,
                q_mid,
                left_pipe,
                right_pipe,
                left_basis_map,
                right_basis_map,
            )
            if root_idx == 0:
                projection0 = projection
            matrices.append(projection.as_matrix())
        root_matrices_by_sector[q_mid] = matrices

        if absorb == "right":
            rho = np.zeros((projection0.left_dim, projection0.left_dim), dtype=np.result_type(*matrices))
            for weight, matrix in zip(weights, matrices):
                rho += weight * (matrix @ matrix.conj().T)
            eigvals, U = np.linalg.eigh(0.5 * (rho + rho.conj().T))
            idx = np.argsort(np.real(eigvals))[::-1]
            eigvals = np.maximum(np.real(eigvals[idx]), 0.0)
            U = U[:, idx]
            singular_values = np.sqrt(eigvals)
            inv_s = np.zeros_like(singular_values)
            mask = singular_values > 1e-12
            inv_s[mask] = 1.0 / singular_values[mask]
            Vh = np.diag(inv_s) @ U.conj().T @ matrices[0]
        else:
            rho = np.zeros((projection0.right_dim, projection0.right_dim), dtype=np.result_type(*matrices))
            for weight, matrix in zip(weights, matrices):
                rho += weight * (matrix.conj().T @ matrix)
            eigvals, V = np.linalg.eigh(0.5 * (rho + rho.conj().T))
            idx = np.argsort(np.real(eigvals))[::-1]
            eigvals = np.maximum(np.real(eigvals[idx]), 0.0)
            V = V[:, idx]
            singular_values = np.sqrt(eigvals)
            inv_s = np.zeros_like(singular_values)
            mask = singular_values > 1e-12
            inv_s[mask] = 1.0 / singular_values[mask]
            U = matrices[0] @ V @ np.diag(inv_s)
            Vh = V.conj().T

        sector_svds[q_mid] = ReducedProjectedSVD(
            projection=projection0,
            singular_values=singular_values,
            U=U,
            Vh=Vh,
        )

    truncation = truncate_reduced_svds(
        sector_svds,
        cutoff=cutoff,
        max_bond=max_bond,
        mode=max_bond_mode,
    )
    kept = {q_mid: list(idxs) for q_mid, idxs in truncation.kept_indices_by_sector.items()}

    singular_values = truncation.singular_values_by_sector()
    right_bond_basis = _bond_basis_from_singular_values(
        singular_values,
        direction=1,
        name="state-averaged-svd-right-bond",
    )
    left_bond_basis = _bond_basis_from_singular_values(
        singular_values,
        direction=-1,
        name="state-averaged-svd-left-bond",
    )
    bond_qns = truncation.bond_qns
    right_split_basis_map = {}

    kept_bond_sectors = tuple(sorted(singular_values))
    slot_counts = {sector: 0 for sector in kept_bond_sectors}
    pipe_entries = []
    for q_mid, idxs in sorted(kept.items()):
        for _idx in idxs:
            pipe_entries.append(
                FusionPipeEntry(
                    child_sectors=(q_mid,),
                    fused_sector=q_mid,
                    slot=slot_counts[q_mid],
                    offset=slot_counts[q_mid],
                    local_dim=1,
                    selected_shape=(1,),
                )
            )
            slot_counts[q_mid] += 1
    bond_pipe = FusionPipe.from_entries(
        child_legs=(0,),
        child_sector_lists=(tuple(kept_bond_sectors),),
        child_dirs=(1,),
        fused_sectors=kept_bond_sectors,
        entries=tuple(pipe_entries),
        orientation=1,
        coupling=bond_coupling,
    )
    bond_leg = FusionLeg(
        child_legs=(0,),
        child_sector_lists=(tuple(kept_bond_sectors),),
        child_dirs=(1,),
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=bond_coupling,
        pipe=bond_pipe,
    )

    left_child_sector_lists = (tuple(ref.qns[0]), tuple(ref.qns[1]))
    left_child_dirs = (ref.dirs[0], ref.dirs[1])
    right_child_sector_lists = (tuple(ref.qns[2]), tuple(ref.qns[3]))
    right_child_dirs = (ref.dirs[2], ref.dirs[3])
    left_fused_pipe = _merge_side_pipes(
        left_output_layouts,
        sectors=kept_bond_sectors,
        side="left",
        child_legs=(0, 1),
        child_sector_lists=left_child_sector_lists,
        child_dirs=left_child_dirs,
    )
    left_fused_leg = FusionLeg(
        child_legs=(0, 1),
        child_sector_lists=left_child_sector_lists,
        child_dirs=left_child_dirs,
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=left_fused_pipe.coupling,
        pipe=left_fused_pipe,
    )
    right_fused_pipe = _merge_side_pipes(
        right_output_layouts,
        sectors=kept_bond_sectors,
        side="right",
        child_legs=(2, 3),
        child_sector_lists=right_child_sector_lists,
        child_dirs=right_child_dirs,
    )
    right_fused_leg = FusionLeg(
        child_legs=(2, 3),
        child_sector_lists=right_child_sector_lists,
        child_dirs=right_child_dirs,
        sectors=kept_bond_sectors,
        orientation=1,
        coupling=right_fused_pipe.coupling,
        pipe=right_fused_pipe,
    )

    left_split_basis_map = {}
    right_split_basis_map = {}
    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
        svd_result = sector_svds[q_mid]
        for entry in svd_result.left_entries:
            left_split_basis_map[(entry.child_sectors, q_mid, entry.slot)] = svd_result.left_basis_map[
                (entry.child_sectors, entry.selected_shape, entry.slot)
            ]
        for entry in svd_result.right_entries:
            right_split_basis_map[(entry.child_sectors, q_mid, entry.slot)] = svd_result.right_basis_map[
                (entry.child_sectors, entry.selected_shape, entry.slot)
            ]

    def _split_from_reduced(left_reduced, right_reduced, *, source):
        left_fused_tensor = NonabelianTensor(
            left_reduced,
            [list(kept_bond_sectors), bond_qns],
            [1, 1],
            fusion_legs=[left_fused_leg, bond_leg],
            metadata={"split_basis_maps": {0: left_split_basis_map}},
        )
        U_split = split_legs(left_fused_tensor, 0)
        U_tensor = NonabelianTensor(
            U_split.data,
            U_split.qns,
            U_split.dirs,
            fusion_legs=[ref.fusion_legs[0], ref.fusion_legs[1], bond_leg],
            metadata={},
        )
        right_fused_tensor = NonabelianTensor(
            right_reduced,
            [bond_qns, list(kept_bond_sectors)],
            [-1, 1],
            fusion_legs=[bond_leg, right_fused_leg],
            metadata={"split_basis_maps": {1: right_split_basis_map}},
        )
        V_split = split_legs(right_fused_tensor, 1)
        V_tensor = NonabelianTensor(
            V_split.data,
            V_split.qns,
            V_split.dirs,
            fusion_legs=[bond_leg, ref.fusion_legs[2], ref.fusion_legs[3]],
            metadata={},
        )
        A_tensor = NonabelianTensor(
            U_tensor.data,
            U_tensor.qns,
            U_tensor.dirs,
            fusion_legs=U_tensor.fusion_legs,
            metadata={
                "svd_role": "left",
                "source": source,
                "bond_layouts": {2: left_output_layouts},
                "bond_bases": {2: right_bond_basis},
            },
        )
        B_tensor = NonabelianTensor(
            V_tensor.data,
            V_tensor.qns,
            V_tensor.dirs,
            fusion_legs=V_tensor.fusion_legs,
            metadata={
                "svd_role": "right",
                "source": source,
                "bond_layouts": {0: right_output_layouts},
                "bond_bases": {0: left_bond_basis},
            },
        )
        return A_tensor, B_tensor

    root_site_pairs = []
    for root_idx in range(len(roots)):
        left_reduced = {}
        right_reduced = {}
        for q_mid, idxs in kept.items():
            idxs = sorted(idxs)
            svd_result = sector_svds[q_mid]
            matrix = root_matrices_by_sector[q_mid][root_idx]
            if absorb == "right":
                left_kept = svd_result.left_matrix(idxs)
                left_reduced[(q_mid, q_mid)] = left_kept
                right_reduced[(q_mid, q_mid)] = left_kept.conj().T @ matrix
            else:
                right_kept = svd_result.right_matrix(idxs)
                left_reduced[(q_mid, q_mid)] = matrix @ right_kept.conj().T
                right_reduced[(q_mid, q_mid)] = right_kept
        root_site_pairs.append(
            _split_from_reduced(
                left_reduced,
                right_reduced,
                source="state_averaged_svd_two_site_root",
            )
        )

    A_tensor, B_tensor = root_site_pairs[0]

    return A_tensor, B_tensor, singular_values, truncation.trunc_err, truncation.kept, root_site_pairs
