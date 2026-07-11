#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced decompositions for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from .contraction import split_legs
from .coupling import (
    clebsch_gordan,
    normalize_coupling_scheme,
    ordered_two_m_values,
    reduced_bond_space,
)
from .linalg import (
    ReducedProjectedSVD,
    normalize_max_bond_mode,
    project_reduced_sector,
    select_kept_singular_values,
    sector_state_weight,
    truncate_reduced_svds,
)
from .tensor import FusionLeg, FusionPipe, FusionPipeEntry, NonabelianTensor
from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors


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


def _physical_basis_from_metadata(metadata):
    metadata = metadata or {}
    basis = metadata.get("physical_basis")
    if basis is not None:
        return basis
    left_basis = metadata.get("left_metadata", {}).get("physical_basis")
    right_basis = metadata.get("right_metadata", {}).get("physical_basis")
    if left_basis == right_basis:
        return left_basis
    return None


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


def _irrep_dim(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None and hasattr(irrep, "dim"):
        return int(irrep.dim)
    dim = getattr(sector, "dim", None)
    if dim is not None:
        return int(dim)
    return 1


def _sector_irrep(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None:
        return irrep
    labels = getattr(sector, "labels", ())
    components = getattr(sector, "components", ())
    if "su2" in labels:
        return components[labels.index("su2")]
    raise TypeError(f"Sector {sector!r} does not carry an SU(2) irrep.")


def _fuse_pair(left, right):
    if hasattr(left, "fuse"):
        return tuple(left.fuse(right))
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return tuple(fuse_charge_spin_sectors(left, right))
    raise TypeError(
        f"Cannot fuse sectors of type {type(left).__name__} and {type(right).__name__}."
    )


@lru_cache(maxsize=None)
def _pair_channel_basis(child_left, child_right, fused_sector, slot, coupling):
    space = reduced_bond_space((child_left, child_right), fused_sector, scheme=coupling)
    for channel, basis in zip(space.channels, space.basis_matrices):
        if int(channel.slot) == int(slot):
            return np.asarray(basis, dtype=float)
    raise ValueError(
        f"Missing physical pair channel {slot} for {(child_left, child_right)!r} -> {fused_sector!r}."
    )


@lru_cache(maxsize=None)
def _coupled_to_mps_tree_recoupling(q_left, q_p1, q_p2, q_pair, q_mid, q_right, pair_slot, pair_coupling):
    """
    Overlap between ``q_left x (q_p1 x q_p2)`` and ``(q_left x q_p1) x q_p2`` trees.
    """
    left_irrep = _sector_irrep(q_left)
    p1_irrep = _sector_irrep(q_p1)
    p2_irrep = _sector_irrep(q_p2)
    pair_irrep = _sector_irrep(q_pair)
    mid_irrep = _sector_irrep(q_mid)
    right_irrep = _sector_irrep(q_right)
    pair_basis = _pair_channel_basis(q_p1, q_p2, q_pair, int(pair_slot), pair_coupling)

    coeff = 0.0
    left_ms = ordered_two_m_values(left_irrep)
    p1_ms = ordered_two_m_values(p1_irrep)
    p2_ms = ordered_two_m_values(p2_irrep)
    pair_ms = ordered_two_m_values(pair_irrep)
    mid_ms = ordered_two_m_values(mid_irrep)
    right_ms = ordered_two_m_values(right_irrep)
    for i_left, two_m_left in enumerate(left_ms):
        for i_p1, two_m_p1 in enumerate(p1_ms):
            for i_p2, two_m_p2 in enumerate(p2_ms):
                pair_row = i_p1 * len(p2_ms) + i_p2
                for i_pair, two_m_pair in enumerate(pair_ms):
                    pair_coeff = pair_basis[pair_row, i_pair]
                    if pair_coeff == 0:
                        continue
                    for two_m_right in right_ms:
                        left_tree = clebsch_gordan(
                            left_irrep,
                            pair_irrep,
                            right_irrep,
                            two_m_left,
                            two_m_pair,
                            two_m_right,
                        )
                        if left_tree == 0:
                            continue
                        for two_m_mid in mid_ms:
                            first = clebsch_gordan(
                                left_irrep,
                                p1_irrep,
                                mid_irrep,
                                two_m_left,
                                two_m_p1,
                                two_m_mid,
                            )
                            if first == 0:
                                continue
                            second = clebsch_gordan(
                                mid_irrep,
                                p2_irrep,
                                right_irrep,
                                two_m_mid,
                                two_m_p2,
                                two_m_right,
                            )
                            if second:
                                coeff += pair_coeff * left_tree * first * second
    return float(coeff) / float(right_irrep.dim)


def _right_reduced_physical_metric(projection):
    """
    Column metric for degeneracy-only right side coordinates.

    For a right site block ``q_mid x q_phys -> q_right`` with physical irrep
    components implicit, summing over the hidden representation indices gives
    a scalar weight ``dim(q_right) / dim(q_mid)`` for each reduced coordinate.
    Legacy explicit-physical blocks keep the Euclidean metric.
    """
    weights = []
    q_mid_dim = max(_irrep_dim(projection.sector), 1)
    for entry in projection.right_entries:
        _q_phys, q_right = entry.child_sectors
        if len(entry.selected_shape) == 2 and int(entry.selected_shape[0]) == 1:
            weight = _irrep_dim(q_right) / q_mid_dim
        else:
            weight = 1.0
        weights.append(np.full(entry.local_dim, float(weight), dtype=float))
    if not weights:
        return np.ones(projection.right_dim, dtype=float)
    metric = np.concatenate(weights)
    if metric.size != projection.right_dim:
        raise ValueError(
            f"Right CG metric has size {metric.size}, expected {projection.right_dim}."
        )
    metric[np.abs(metric) <= 1.0e-15] = 1.0
    return metric


def _right_metric_weighted_projected_svd(projection, *, full_matrices=False):
    matrix = projection.as_matrix()
    right_metric = _right_reduced_physical_metric(projection)
    if np.allclose(right_metric, 1.0):
        return projection.svd(full_matrices=full_matrices)
    sqrt_metric = np.sqrt(right_metric)
    weighted = matrix * sqrt_metric[None, :]
    U, S, W_h = np.linalg.svd(weighted, full_matrices=full_matrices)
    Vh = W_h / sqrt_metric[None, :]
    return ReducedProjectedSVD(
        projection=projection,
        singular_values=S,
        U=U,
        Vh=Vh,
    )


def svd_coupled_two_site(
    coupled_two_site,
    two_site_template,
    max_bond=None,
    cutoff=1.0e-10,
    absorb="right",
    bond_coupling="left",
    max_bond_mode="reduced",
):
    """
    Split a two-site tensor kept in the coupled physical-pair basis.

    Degeneracy-only physical legs cannot safely be uncoupled by summing the
    physical-pair channels: different ``p1 x p2 -> p`` fusion paths can occupy
    the same reduced sector key.  This routine recouples those channels into
    the MPS internal tree ``(left x p1) x p2`` before doing the per-sector SVD.
    """
    if not isinstance(coupled_two_site, NonabelianTensor) or coupled_two_site.rank != 3:
        raise ValueError("svd_coupled_two_site expects a rank-3 coupled tensor.")
    if not isinstance(two_site_template, NonabelianTensor) or two_site_template.rank != 4:
        raise ValueError("svd_coupled_two_site expects a rank-4 two-site template.")
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")
    fused_leg = coupled_two_site.fusion_legs[1]
    if fused_leg is None or fused_leg.pipe is None:
        raise ValueError("Coupled two-site tensor is missing physical-pair fusion metadata.")
    pipe = fused_leg.pipe
    if len(pipe.child_sector_lists) != 2:
        raise ValueError("Coupled two-site tensor must fuse exactly two physical legs.")
    max_bond_mode = normalize_max_bond_mode(max_bond_mode, default="reduced")
    bond_coupling = normalize_coupling_scheme(bond_coupling, default="left")

    row_dims = {}
    col_dims = {}
    contributions = {}
    dtype = np.result_type(*(np.asarray(block).dtype for block in coupled_two_site.data.values()))
    for (q_left, q_pair, q_right), block in coupled_two_site.data.items():
        arr = np.asarray(block)
        for entry in pipe.entries_for_sector(q_pair):
            if int(entry.local_dim) != 1:
                raise NotImplementedError(
                    "svd_coupled_two_site currently supports degeneracy-only physical pair channels."
                )
            q_p1, q_p2 = entry.child_sectors
            local = arr[:, int(entry.offset), :]
            for q_mid in _fuse_pair(q_left, q_p1):
                if q_right not in _fuse_pair(q_mid, q_p2):
                    continue
                recoupling = _coupled_to_mps_tree_recoupling(
                    q_left,
                    q_p1,
                    q_p2,
                    q_pair,
                    q_mid,
                    q_right,
                    int(entry.slot),
                    pipe.coupling,
                )
                if abs(recoupling) <= 1.0e-14:
                    continue
                row_dims.setdefault(q_mid, {})[(q_left, q_p1)] = int(arr.shape[0])
                col_dims.setdefault(q_mid, {})[(q_p2, q_right)] = int(arr.shape[2])
                contributions.setdefault(q_mid, []).append(
                    (q_left, q_p1, q_p2, q_right, local, recoupling)
                )

    matrices = {}
    row_offsets = {}
    col_offsets = {}
    for q_mid, terms in contributions.items():
        rows = sorted(row_dims[q_mid])
        cols = sorted(col_dims[q_mid])
        row_cursor = 0
        row_offsets[q_mid] = {}
        for key in rows:
            dim = row_dims[q_mid][key]
            row_offsets[q_mid][key] = (row_cursor, row_cursor + dim)
            row_cursor += dim
        col_cursor = 0
        col_offsets[q_mid] = {}
        for key in cols:
            dim = col_dims[q_mid][key]
            col_offsets[q_mid][key] = (col_cursor, col_cursor + dim)
            col_cursor += dim
        matrix = np.zeros((row_cursor, col_cursor), dtype=dtype)
        for q_left, q_p1, q_p2, q_right, local, recoupling in terms:
            r0, r1 = row_offsets[q_mid][(q_left, q_p1)]
            c0, c1 = col_offsets[q_mid][(q_p2, q_right)]
            matrix[r0:r1, c0:c1] += np.asarray(recoupling, dtype=dtype) * local
        if np.any(matrix):
            matrices[q_mid] = matrix

    if not matrices:
        raise ValueError("Coupled two-site SVD found no recoupled MPS-bond sectors.")

    svds = {}
    singular_items = []
    for q_mid, matrix in matrices.items():
        metric_parts = []
        q_mid_dim = max(_irrep_dim(q_mid), 1)
        for q_p2, q_right in sorted(col_dims[q_mid]):
            dim = col_dims[q_mid][(q_p2, q_right)]
            weight = _irrep_dim(q_right) / q_mid_dim
            metric_parts.append(np.full(dim, float(weight), dtype=float))
        right_metric = np.concatenate(metric_parts) if metric_parts else np.ones(matrix.shape[1])
        right_metric[np.abs(right_metric) <= 1.0e-15] = 1.0
        sqrt_metric = np.sqrt(right_metric)
        weighted = matrix * sqrt_metric[None, :]
        U, S, W_h = np.linalg.svd(weighted, full_matrices=False)
        Vh = W_h / sqrt_metric[None, :]
        svds[q_mid] = (U, S, Vh)
        weight = sector_state_weight(q_mid)
        for idx, sval in enumerate(S):
            if float(sval) > float(cutoff):
                singular_items.append((float(sval), q_mid, idx, weight))
    if not singular_items:
        raise ValueError("All coupled two-site singular values were truncated.")
    singular_items.sort(reverse=True, key=lambda item: item[0])
    kept_items = select_kept_singular_values(
        singular_items,
        max_bond,
        mode=max_bond_mode,
    )
    kept_by_sector = {}
    for _sval, sector, idx, _weight in kept_items:
        kept_by_sector.setdefault(sector, []).append(idx)
    kept_by_sector = {
        sector: tuple(sorted(indices))
        for sector, indices in sorted(kept_by_sector.items())
    }
    full_sq_norm = sum(item[0] ** 2 for item in singular_items)
    kept_sq_norm = sum(item[0] ** 2 for item in kept_items)
    trunc_err = 0.0 if full_sq_norm <= 1.0e-15 else 1.0 - kept_sq_norm / full_sq_norm

    kept_sectors = tuple(sorted(kept_by_sector))
    bond_qns = []
    pipe_entries = []
    offset_counts = {sector: 0 for sector in kept_sectors}
    for q_mid in kept_sectors:
        for _idx in kept_by_sector[q_mid]:
            slot = offset_counts[q_mid]
            pipe_entries.append(
                FusionPipeEntry(
                    child_sectors=(q_mid,),
                    fused_sector=q_mid,
                    slot=slot,
                    offset=slot,
                    local_dim=1,
                    selected_shape=(1,),
                )
            )
            bond_qns.append(q_mid)
            offset_counts[q_mid] += 1
    bond_pipe = FusionPipe.from_entries(
        child_legs=(0,),
        child_sector_lists=(kept_sectors,),
        child_dirs=(1,),
        fused_sectors=kept_sectors,
        entries=tuple(pipe_entries),
        orientation=1,
        coupling=bond_coupling,
    )
    bond_leg = FusionLeg(
        child_legs=(0,),
        child_sector_lists=(kept_sectors,),
        child_dirs=(1,),
        sectors=kept_sectors,
        orientation=1,
        coupling=bond_coupling,
        pipe=bond_pipe,
    )

    left_data = {}
    right_data = {}
    singular_values = {}
    for q_mid, kept in kept_by_sector.items():
        U, S, Vh = svds[q_mid]
        kept = tuple(kept)
        if absorb == "left":
            left_matrix = U[:, kept] @ np.diag(S[list(kept)])
            right_matrix = Vh[list(kept), :]
        else:
            left_matrix = U[:, kept]
            right_matrix = np.diag(S[list(kept)]) @ Vh[list(kept), :]
        singular_values[q_mid] = np.diag(S[list(kept)])

        for (q_left, q_p1), (r0, r1) in row_offsets[q_mid].items():
            block = left_matrix[r0:r1, :].reshape(r1 - r0, 1, len(kept))
            if np.any(block):
                left_data[(q_left, q_p1, q_mid)] = block
        for (q_p2, q_right), (c0, c1) in col_offsets[q_mid].items():
            block = right_matrix[:, c0:c1].reshape(len(kept), 1, c1 - c0)
            if np.any(block):
                right_data[(q_mid, q_p2, q_right)] = block

    physical_basis = _physical_basis_from_metadata(two_site_template.metadata)
    left_tensor = NonabelianTensor(
        left_data,
        [list(two_site_template.qns[0]), list(two_site_template.qns[1]), bond_qns],
        [two_site_template.dirs[0], two_site_template.dirs[1], 1],
        fusion_legs=[two_site_template.fusion_legs[0], two_site_template.fusion_legs[1], bond_leg],
        metadata={
            "svd_role": "left",
            "source": "svd_coupled_two_site",
            **({"physical_basis": physical_basis} if physical_basis is not None else {}),
        },
    )
    right_tensor = NonabelianTensor(
        right_data,
        [bond_qns, list(two_site_template.qns[2]), list(two_site_template.qns[3])],
        [-1, two_site_template.dirs[2], two_site_template.dirs[3]],
        fusion_legs=[bond_leg, two_site_template.fusion_legs[2], two_site_template.fusion_legs[3]],
        metadata={
            "svd_role": "right",
            "source": "svd_coupled_two_site",
            **({"physical_basis": physical_basis} if physical_basis is not None else {}),
        },
    )
    kept = sum(len(indices) for indices in kept_by_sector.values())
    return left_tensor, right_tensor, singular_values, trunc_err, kept


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
    physical_basis = _physical_basis_from_metadata(two_site.metadata)
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
        svd_result = _right_metric_weighted_projected_svd(
            reduced_sector,
            full_matrices=False,
        )
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
            **({"physical_basis": physical_basis} if physical_basis is not None else {}),
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
            **({"physical_basis": physical_basis} if physical_basis is not None else {}),
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

    ref_blocks_by_mid = {}
    bond_entries = dict(_internal_bond_entries(ref))
    for key, block in ref.data.items():
        q_mid = bond_entries.get(key)
        if q_mid is None:
            raise ValueError(f"Missing contracted bond sector for key {key!r}.")
        ref_blocks_by_mid.setdefault(q_mid, []).append((key, block))

    sector_svds = {}
    left_source_layouts = _get_site_bond_layout(ref, side="left", axis=1)
    right_source_layouts = _get_site_bond_layout(ref, side="right", axis=0)
    left_output_layouts = {}
    right_output_layouts = {}
    bond_coupling = normalize_coupling_scheme(bond_coupling, default="left")

    for q_mid, ref_entries in ref_blocks_by_mid.items():
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
            "source": "state_averaged_svd_two_site",
            "bond_layouts": {2: left_output_layouts},
        },
    )
    B_tensor = NonabelianTensor(
        B_tensor.data,
        B_tensor.qns,
        B_tensor.dirs,
        fusion_legs=B_tensor.fusion_legs,
        metadata={
            "svd_role": "right",
            "source": "state_averaged_svd_two_site",
            "bond_layouts": {0: right_output_layouts},
        },
    )

    return A_tensor, B_tensor, singular_values, truncation.trunc_err, truncation.kept
