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
    ReducedTruncation,
    normalize_max_bond_mode,
    project_reduced_sector,
    sector_state_weight,
    truncate_reduced_svds,
)
from .tensor import (
    FusionLeg,
    FusionPipe,
    FusionPipeEntry,
    IdentityBasisTransform,
    NonabelianTensor,
)


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
            if q_mid_tuple is None or len(q_mid_tuple) == 0:
                raise ValueError(
                    f"Expected contracted bond sectors for key {key!r}, got {q_mid_tuple!r}."
                )
            entries.extend((tuple(key), q_mid) for q_mid in q_mid_tuple)
        return tuple(entries)

    raise ValueError(
        "svd_two_site requires contracted bond metadata from merge_mps_sites/tensordot."
    )


def _internal_bond_channel_map(two_site):
    channel_map = {}
    for child_sectors, fused_sector in _internal_bond_entries(two_site):
        channel_map.setdefault(tuple(child_sectors), set()).add(fused_sector)
    return {
        key: tuple(sorted(channels))
        for key, channels in channel_map.items()
    }


def _align_state_average_root_layouts(roots):
    """Pad reduced two-site roots into their common sector-block union."""

    reference = roots[0]
    if any(root.dirs != reference.dirs for root in roots[1:]):
        raise ValueError("State-average roots must use the same leg directions.")

    common_qns = []
    for axis in range(reference.rank):
        sector_order = []
        multiplicities = {}
        for root in roots:
            counts = {}
            for sector in root.qns[axis]:
                counts[sector] = counts.get(sector, 0) + 1
                if sector not in sector_order:
                    sector_order.append(sector)
            for sector, count in counts.items():
                multiplicities[sector] = max(
                    multiplicities.get(sector, 0),
                    count,
                )
        common_qns.append(
            [
                sector
                for sector in sector_order
                for _ in range(multiplicities[sector])
            ]
        )

    keys = sorted(
        set().union(*(root.data for root in roots)),
    )
    shapes = {}
    channels = {}
    for root in roots:
        root_channels = _internal_bond_channel_map(root)
        for key, root_channels_for_key in root_channels.items():
            channels.setdefault(key, set()).update(root_channels_for_key)
        for key, block in root.data.items():
            shape = tuple(int(value) for value in np.asarray(block).shape)
            previous = shapes.get(key)
            shapes[key] = shape if previous is None else tuple(
                max(left, right) for left, right in zip(previous, shape)
            )

    physical_basis = None
    for root in roots:
        metadata = root.metadata or {}
        if (
            metadata.get("physical_basis") == "fully_reduced_su2"
            or (metadata.get("left_metadata") or {}).get("physical_basis")
                == "fully_reduced_su2"
            or (metadata.get("right_metadata") or {}).get("physical_basis")
                == "fully_reduced_su2"
        ):
            physical_basis = "fully_reduced_su2"
            break
    common_metadata = {
        "contracted_channels": {
            key: tuple(sorted(values)) for key, values in channels.items()
        },
        "contracted_channel_blocks_current": False,
        **({"physical_basis": physical_basis} if physical_basis else {}),
    }

    aligned = []
    for root in roots:
        dtype = np.result_type(
            *[np.asarray(block).dtype for block in root.data.values()],
            float,
        )
        data = {}
        for key in keys:
            target_shape = shapes[key]
            target = np.zeros(target_shape, dtype=dtype)
            source = root.data.get(key)
            if source is not None:
                source = np.asarray(source)
                target[tuple(slice(0, size) for size in source.shape)] = source
            data[key] = target
        aligned.append(
            NonabelianTensor(
                data,
                [leg[:] for leg in common_qns],
                reference.dirs[:],
                fusion_legs=reference.fusion_legs[:],
                metadata=common_metadata.copy(),
            )
        )
    return aligned


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

    coupling = normalize_coupling_scheme(coupling, default="left")
    packed = {}
    for key, block in entries:
        combo = child_sectors_fn(key)
        selected_shape = selected_shape_fn(block)
        packed.setdefault((combo, selected_shape), selected_shape)

    child_sector_lists = tuple(
        tuple(sorted({combo[i] for combo, _shape in packed}))
        for i in range(2)
    )

    if source_layouts is not None:
        source = source_layouts.get(q_mid)
        if source is not None:
            source_pipe = source["pipe"]
            source_entries = source_pipe.entries_for_sector(q_mid)
            source_keys = {
                (entry.child_sectors, entry.selected_shape, int(entry.local_dim))
                for entry in source_entries
            }
            packed_keys = {
                (combo, shape, int(np.prod(shape, dtype=int)))
                for combo, shape in packed
            }
            if (
                source_pipe.child_legs == child_legs
                and source_pipe.child_sector_lists == child_sector_lists
                and source_pipe.child_dirs == child_dirs
                and source_keys == packed_keys
            ):
                return source_pipe, source["basis_map"], source.get("channel_map", {})

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
        basis_map[(child_sectors, selected_shape, 0)] = IdentityBasisTransform(local_dim)
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
    labels = getattr(sector, "labels", ())
    components = getattr(sector, "components", ())
    if "su2" in labels:
        irrep = components[labels.index("su2")]
        if hasattr(irrep, "dim"):
            return int(irrep.dim)
    dim = getattr(sector, "dim", None)
    if dim is not None:
        return int(dim)
    return 1


def _right_reduced_physical_metric(projection):
    """
    Diagonal metric for fully reduced right-side coordinates.

    For a reduced right site channel ``q_mid x q_phys -> q_right`` whose
    physical leg stores only the multiplet reduced coordinate, summing over
    hidden spin components contributes ``dim(q_right) / dim(q_mid)``.  Explicit
    physical-component blocks are already Euclidean and keep unit weight.
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
            f"Right reduced metric has size {metric.size}, expected {projection.right_dim}."
        )
    metric[np.abs(metric) <= 1.0e-15] = 1.0
    return metric


def _right_metric_weighted_projected_svd(
    projection,
    *,
    full_matrices=False,
    apply_reduced_metric=True,
):
    matrix = projection.as_matrix()
    if not apply_reduced_metric:
        return projection.svd(full_matrices=full_matrices)
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


def _boundary_metric_factor_maps(metric_factor_blocks, metric_factor_routes):
    """Recover left/right boundary metric blocks from compact local routes."""

    left = {}
    right = {}

    def put(mapping, key, value):
        value = np.asarray(value)
        previous = mapping.get(key)
        if previous is None:
            mapping[key] = value
            return True
        return bool(np.allclose(previous, value, rtol=1.0e-11, atol=1.0e-13))

    for key, factors in dict(metric_factor_blocks or {}).items():
        q_left, q_right = key[0], key[-1]
        if not put(left, (q_left, q_left), factors[0]):
            return None
        if not put(right, (q_right, q_right), factors[1]):
            return None

    for route in tuple(metric_factor_routes or ()):
        _in_idx, _out_idx, in_entry, out_entry, left_factor, right_factor = route
        if not put(
            left,
            (out_entry.key[0], in_entry.key[0]),
            left_factor,
        ):
            return None
        if not put(
            right,
            (out_entry.key[-1], in_entry.key[-1]),
            right_factor,
        ):
            return None
    if not left or not right:
        return None
    return left, right


def _projected_side_metric(entries, factors, *, side):
    """Assemble one exact boundary-factor metric in projected SVD coordinates."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    dim = max(
        (int(entry.offset) + int(entry.local_dim) for entry in entries),
        default=0,
    )
    metric = np.zeros(
        (dim, dim),
        dtype=np.result_type(*(np.asarray(value).dtype for value in factors.values())),
    )
    for out_entry in entries:
        for in_entry in entries:
            if side == "left":
                q_out, p_out = out_entry.child_sectors
                q_in, p_in = in_entry.child_sectors
                if p_out != p_in:
                    continue
                factor = factors.get((q_out, q_in))
                if factor is None:
                    continue
                p_dim_out = int(out_entry.selected_shape[1])
                p_dim_in = int(in_entry.selected_shape[1])
                if p_dim_out != p_dim_in:
                    continue
                block = np.kron(
                    np.asarray(factor),
                    np.eye(p_dim_out, dtype=metric.dtype),
                )
            else:
                p_out, q_out = out_entry.child_sectors
                p_in, q_in = in_entry.child_sectors
                if p_out != p_in:
                    continue
                factor = factors.get((q_out, q_in))
                if factor is None:
                    continue
                p_dim_out = int(out_entry.selected_shape[0])
                p_dim_in = int(in_entry.selected_shape[0])
                if p_dim_out != p_dim_in:
                    continue
                block = np.kron(
                    np.eye(p_dim_out, dtype=metric.dtype),
                    np.asarray(factor),
                )
            if block.shape != (
                int(out_entry.local_dim),
                int(in_entry.local_dim),
            ):
                return None
            out_slice = slice(
                int(out_entry.offset),
                int(out_entry.offset) + int(out_entry.local_dim),
            )
            in_slice = slice(
                int(in_entry.offset),
                int(in_entry.offset) + int(in_entry.local_dim),
            )
            metric[out_slice, in_slice] = block
    return 0.5 * (metric + metric.conj().T)


def _positive_metric_sqrt(metric, *, tol=1.0e-12):
    """Return a Hermitian square root and pseudoinverse square root."""

    metric = np.asarray(metric)
    if np.iscomplexobj(metric):
        scale = max(1.0, float(np.max(np.abs(metric.real), initial=0.0)))
        if float(np.max(np.abs(metric.imag), initial=0.0)) <= float(tol) * scale:
            metric = np.asarray(metric.real)
    values, vectors = np.linalg.eigh(metric)
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    keep = values > float(tol) * scale
    if not np.any(keep):
        return None
    positive_vectors = vectors[:, keep]
    positive_values = np.real(values[keep])
    sqrt_metric = (
        positive_vectors * np.sqrt(positive_values)[None, :]
    ) @ positive_vectors.conj().T
    inverse_sqrt = (
        positive_vectors * (1.0 / np.sqrt(positive_values))[None, :]
    ) @ positive_vectors.conj().T
    return sqrt_metric, inverse_sqrt


def _factor_metric_weighted_projected_svd(
    projection,
    *,
    left_factors,
    right_factors,
    full_matrices=False,
    tol=1.0e-12,
):
    """SVD a sector using the exact factorized left/right norm metric."""

    left_metric = _projected_side_metric(
        projection.left_entries,
        left_factors,
        side="left",
    )
    right_metric = _projected_side_metric(
        projection.right_entries,
        right_factors,
        side="right",
    )
    if left_metric is None or right_metric is None:
        return None
    left_pair = _positive_metric_sqrt(left_metric, tol=tol)
    right_pair = _positive_metric_sqrt(right_metric, tol=tol)
    if left_pair is None or right_pair is None:
        return None
    left_sqrt, left_inverse_sqrt = left_pair
    right_sqrt, right_inverse_sqrt = right_pair
    weighted = (
        left_sqrt
        @ projection.as_matrix()
        @ right_sqrt.T
    )
    if np.iscomplexobj(weighted):
        scale = max(1.0, float(np.max(np.abs(weighted.real), initial=0.0)))
        if float(np.max(np.abs(weighted.imag), initial=0.0)) <= float(tol) * scale:
            weighted = np.asarray(weighted.real)
    U_weighted, singular_values, Vh_weighted = np.linalg.svd(
        weighted,
        full_matrices=full_matrices,
    )
    return ReducedProjectedSVD(
        projection=projection,
        singular_values=singular_values,
        U=left_inverse_sqrt @ U_weighted,
        Vh=Vh_weighted @ right_inverse_sqrt.T,
        state_weight_override=1,
    )


def svd_two_site(
    two_site,
    max_bond=None,
    cutoff=1e-10,
    absorb="right",
    bond_coupling="left",
    max_bond_mode="reduced",
    metric_factor_blocks=None,
    metric_factor_routes=None,
    retain_sector_topology=False,
    sweep_engine=None,
):
    """
    Reduced SVD/truncation helper for a merged two-site tensor.
    """
    if not isinstance(two_site, NonabelianTensor) or two_site.rank != 4:
        raise ValueError("svd_two_site expects a rank-4 NonabelianTensor.")
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")
    max_bond_mode = normalize_max_bond_mode(max_bond_mode, default="reduced")
    left_meta = (two_site.metadata or {}).get("left_metadata", {})
    right_meta = (two_site.metadata or {}).get("right_metadata", {})
    use_reduced_right_metric = (
        left_meta.get("physical_basis") == "fully_reduced_su2"
        or right_meta.get("physical_basis") == "fully_reduced_su2"
        or (two_site.metadata or {}).get("physical_basis") == "fully_reduced_su2"
    )
    physical_basis = "fully_reduced_su2" if use_reduced_right_metric else None

    blocks_by_mid = {}
    if bool(
        two_site.metadata.get("contracted_channel_blocks_current", False)
    ):
        for channel_key, block in two_site.metadata.get(
            "contracted_channel_blocks",
            {},
        ).items():
            q_left, q_phys1, q_mid, q_phys2, q_right = channel_key
            key = (q_left, q_phys1, q_phys2, q_right)
            blocks_by_mid.setdefault(q_mid, []).append((key, block))
    else:
        bond_entries = _internal_bond_channel_map(two_site)
        for key, block in two_site.data.items():
            q_mids = bond_entries.get(key)
            if not q_mids:
                raise ValueError(
                    f"Missing contracted bond sector for key {key!r}."
                )
            if not use_reduced_right_metric and len(q_mids) > 1:
                q_mids = (q_mids[0],)
            for q_mid in q_mids:
                blocks_by_mid.setdefault(q_mid, []).append((key, block))

    sector_svds = {}
    sector_projections = {}
    left_source_layouts = _get_site_bond_layout(two_site, side="left", axis=1)
    right_source_layouts = _get_site_bond_layout(two_site, side="right", axis=0)
    left_output_layouts = {}
    right_output_layouts = {}

    bond_coupling = normalize_coupling_scheme(bond_coupling, default="left")
    metric_factor_maps = _boundary_metric_factor_maps(
        metric_factor_blocks,
        metric_factor_routes,
    )
    used_factor_metric = False

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
        sector_projections[q_mid] = reduced_sector

    cpp_blockwise_svd = (
        None
        if sweep_engine is None
        else getattr(sweep_engine, "blockwise_svd", None)
    )
    if cpp_blockwise_svd is not None and metric_factor_maps is None:
        ordered_sectors = list(sector_projections)
        matrices = []
        inverse_right_scales = []
        for q_mid in ordered_sectors:
            projection = sector_projections[q_mid]
            matrix = projection.as_matrix()
            inverse_scale = None
            if use_reduced_right_metric:
                right_metric = _right_reduced_physical_metric(projection)
                if not np.allclose(right_metric, 1.0):
                    sqrt_metric = np.sqrt(right_metric)
                    matrix = matrix * sqrt_metric[None, :]
                    inverse_scale = 1.0 / sqrt_metric
            matrices.append(matrix)
            inverse_right_scales.append(inverse_scale)
        cpp_result = cpp_blockwise_svd(
            matrices,
            [sector_state_weight(q_mid) for q_mid in ordered_sectors],
            cutoff=cutoff,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            retain_sector_topology=retain_sector_topology,
        )
        kept_indices_by_sector = {}
        for index, q_mid in enumerate(ordered_sectors):
            Vh = np.asarray(cpp_result["right"][index])
            inverse_scale = inverse_right_scales[index]
            if inverse_scale is not None:
                Vh = Vh * inverse_scale[None, :]
            sector_svds[q_mid] = ReducedProjectedSVD(
                projection=sector_projections[q_mid],
                singular_values=np.asarray(
                    cpp_result["singular_values"][index],
                    dtype=float,
                ),
                U=np.asarray(cpp_result["left"][index]),
                Vh=Vh,
            )
            indices = tuple(
                int(value) for value in cpp_result["kept_indices"][index]
            )
            if indices:
                kept_indices_by_sector[q_mid] = indices
        truncation = ReducedTruncation(
            sector_svds=sector_svds,
            kept_indices_by_sector=kept_indices_by_sector,
            trunc_err=float(cpp_result["truncation_error"]),
            full_sq_norm=float(cpp_result["full_squared_norm"]),
            kept_sq_norm=float(cpp_result["kept_squared_norm"]),
            mode=max_bond_mode,
        )
    else:
        for q_mid, reduced_sector in sector_projections.items():
            svd_result = None
            if metric_factor_maps is not None:
                svd_result = _factor_metric_weighted_projected_svd(
                    reduced_sector,
                    left_factors=metric_factor_maps[0],
                    right_factors=metric_factor_maps[1],
                    full_matrices=False,
                )
                used_factor_metric = bool(
                    svd_result is not None or used_factor_metric
                )
            if svd_result is None:
                svd_result = _right_metric_weighted_projected_svd(
                    reduced_sector,
                    full_matrices=False,
                    apply_reduced_metric=use_reduced_right_metric,
                )
            sector_svds[q_mid] = svd_result
        truncation = truncate_reduced_svds(
            sector_svds,
            cutoff=cutoff,
            max_bond=max_bond,
            mode=max_bond_mode,
            retain_sector_topology=retain_sector_topology,
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
            **(
                {"canonical_metric": "factorized_boundary"}
                if used_factor_metric
                else {}
            ),
            **({"physical_basis": physical_basis} if physical_basis else {}),
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
            **(
                {"canonical_metric": "factorized_boundary"}
                if used_factor_metric
                else {}
            ),
            **({"physical_basis": physical_basis} if physical_basis else {}),
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
    retain_sector_topology=False,
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
    if any(root.qns != ref.qns or root.dirs != ref.dirs for root in roots[1:]):
        roots = _align_state_average_root_layouts(roots)
        ref = roots[0]
    if absorb not in {"left", "right"}:
        raise ValueError("absorb must be 'left' or 'right'.")
    max_bond_mode = normalize_max_bond_mode(max_bond_mode, default="reduced")
    left_meta = (ref.metadata or {}).get("left_metadata", {})
    right_meta = (ref.metadata or {}).get("right_metadata", {})
    use_reduced_right_metric = (
        left_meta.get("physical_basis") == "fully_reduced_su2"
        or right_meta.get("physical_basis") == "fully_reduced_su2"
        or (ref.metadata or {}).get("physical_basis") == "fully_reduced_su2"
    )
    physical_basis = "fully_reduced_su2" if use_reduced_right_metric else None

    blocks_by_mid = {}
    mid_by_key = {}
    for root in roots:
        bond_entries = _internal_bond_channel_map(root)
        for key, block in root.data.items():
            q_mids = bond_entries.get(key)
            if not q_mids:
                raise ValueError(f"Missing contracted bond sector for key {key!r}.")
            if not use_reduced_right_metric and len(q_mids) > 1:
                q_mids = (q_mids[0],)
            previous_mids = mid_by_key.setdefault(key, q_mids)
            if previous_mids != q_mids:
                raise ValueError(
                    f"State-average roots assign key {key!r} to inconsistent bond sectors."
                )
            for q_mid in q_mids:
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
        right_metric = (
            _right_reduced_physical_metric(projection0)
            if use_reduced_right_metric
            else np.ones(projection0.right_dim, dtype=float)
        )
        sqrt_right_metric = np.sqrt(right_metric)

        if absorb == "right":
            rho = np.zeros((projection0.left_dim, projection0.left_dim), dtype=np.result_type(*matrices))
            for weight, matrix in zip(weights, matrices):
                rho += weight * ((matrix * right_metric[None, :]) @ matrix.conj().T)
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
                weighted_matrix = matrix * sqrt_right_metric[None, :]
                rho += weight * (weighted_matrix.conj().T @ weighted_matrix)
            eigvals, W = np.linalg.eigh(0.5 * (rho + rho.conj().T))
            idx = np.argsort(np.real(eigvals))[::-1]
            eigvals = np.maximum(np.real(eigvals[idx]), 0.0)
            W = W[:, idx]
            singular_values = np.sqrt(eigvals)
            inv_s = np.zeros_like(singular_values)
            mask = singular_values > 1e-12
            inv_s[mask] = 1.0 / singular_values[mask]
            Vh = W.conj().T / sqrt_right_metric[None, :]
            U = ((matrices[0] * right_metric[None, :]) @ Vh.conj().T) @ np.diag(inv_s)

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
        retain_sector_topology=retain_sector_topology,
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
                **({"physical_basis": physical_basis} if physical_basis else {}),
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
                **({"physical_basis": physical_basis} if physical_basis else {}),
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
                right_metric = (
                    _right_reduced_physical_metric(svd_result.projection)
                    if use_reduced_right_metric
                    else np.ones(svd_result.projection.right_dim, dtype=float)
                )
                left_reduced[(q_mid, q_mid)] = (matrix * right_metric[None, :]) @ right_kept.conj().T
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
