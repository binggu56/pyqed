#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-layout reduced contractions for :mod:`pyqed.mps.nonabelian`.
"""

from __future__ import annotations

import numpy as np

from .coupling import normalize_coupling_scheme, reduced_bond_space
from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors
from .tensor import FusionLeg, FusionPipe, FusionPipeEntry, NonabelianTensor


def _normalize_axes(axes):
    a_ax, b_ax = axes
    if isinstance(a_ax, int):
        a_ax = [a_ax]
    if isinstance(b_ax, int):
        b_ax = [b_ax]
    a_ax = list(a_ax)
    b_ax = list(b_ax)
    if len(a_ax) != len(b_ax):
        raise ValueError(
            f"Contraction axis mismatch: {len(a_ax)} axes on A but {len(b_ax)} axes on B."
        )
    return a_ax, b_ax


def _validate_contraction_metadata(A, B, a_ax, b_ax):
    for axis_a, axis_b in zip(a_ax, b_ax):
        if A.dirs[axis_a] != -B.dirs[axis_b]:
            raise ValueError(
                f"Nonabelian contraction requires opposite leg directions, got "
                f"A[{axis_a}]={A.dirs[axis_a]} and B[{axis_b}]={B.dirs[axis_b]}."
            )
        if A.fusion_legs[axis_a] != B.fusion_legs[axis_b]:
            raise ValueError(
                f"Nonabelian contraction requires matching fixed fusion-tree metadata on "
                f"contracted legs, got {A.fusion_legs[axis_a]!r} vs {B.fusion_legs[axis_b]!r}."
            )


def _metadata_equal(a, b):
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_metadata_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_metadata_equal(x, y) for x, y in zip(a, b))
    return a == b


def _fuse_site_sectors(left, phys):
    if hasattr(left, "fuse"):
        return tuple(left.fuse(phys))
    if isinstance(left, SpinChargeSector) and isinstance(phys, SpinChargeSector):
        return tuple(fuse_charge_spin_sectors(left, phys))
    return ()


def _site_is_left_phys_right(site):
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        return False
    score_new = 0
    score_old = 0
    for key in site.data:
        if key[2] in _fuse_site_sectors(key[0], key[1]):
            score_new += 1
        if key[1] in _fuse_site_sectors(key[0], key[2]):
            score_old += 1
    if score_new != score_old:
        return score_new > score_old
    bond_like_1 = site.fusion_legs[1] is not None
    bond_like_2 = site.fusion_legs[2] is not None
    if bond_like_1 != bond_like_2:
        return bond_like_2
    return True


def normalize_site_tensor_layout(site):
    """
    Return a rank-3 site tensor in canonical ``(left, phys, right)`` order.
    """
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        return site
    if _site_is_left_phys_right(site):
        return site
    return site.transpose(0, 2, 1)


def tensordot(A, B, axes):
    """
    Fixed-layout contraction for :class:`NonabelianTensor`.

    This helper intentionally implements only the simplest non-Abelian case:
    contracted legs must carry identical reduced sectors and identical
    fusion-edge metadata.  That is enough to support fixed fusion-tree layouts
    without yet introducing explicit recoupling coefficients.
    """
    if not isinstance(A, NonabelianTensor) or not isinstance(B, NonabelianTensor):
        raise TypeError("tensordot expects two NonabelianTensor objects.")

    a_ax, b_ax = _normalize_axes(axes)
    _validate_contraction_metadata(A, B, a_ax, b_ax)

    free_A = [i for i in range(A.rank) if i not in a_ax]
    free_B = [i for i in range(B.rank) if i not in b_ax]

    new_qns = [A.qns[i][:] for i in free_A] + [B.qns[i][:] for i in free_B]
    new_dirs = [A.dirs[i] for i in free_A] + [B.dirs[i] for i in free_B]
    new_fusion_legs = [A.fusion_legs[i] for i in free_A] + [B.fusion_legs[i] for i in free_B]

    b_map = {}
    for key_B, block_B in B.data.items():
        contracted = tuple(key_B[i] for i in b_ax)
        b_map.setdefault(contracted, []).append((key_B, block_B))

    new_data = {}
    contracted_channels = {}
    for key_A, block_A in A.data.items():
        contracted = tuple(key_A[i] for i in a_ax)
        if contracted not in b_map:
            continue

        for key_B, block_B in b_map[contracted]:
            block_C = np.tensordot(block_A, block_B, axes=(a_ax, b_ax))
            key_C = tuple(key_A[i] for i in free_A) + tuple(key_B[i] for i in free_B)
            contracted_channels[key_C] = contracted
            if key_C in new_data:
                new_data[key_C] = new_data[key_C] + block_C
            else:
                new_data[key_C] = block_C

    metadata = {}
    if _metadata_equal(A.metadata, B.metadata):
        metadata = A.metadata.copy()
    elif A.metadata or B.metadata:
        metadata = {
            "left_metadata": A.metadata.copy(),
            "right_metadata": B.metadata.copy(),
        }
    if contracted_channels:
        metadata["contracted_channels"] = contracted_channels
        if len(a_ax) == 1 and len(b_ax) == 1:
            fused_sectors = tuple(sorted({contracted[0] for contracted in contracted_channels.values()}))
            slot_counts = {sector: 0 for sector in fused_sectors}
            offset_counts = {sector: 0 for sector in fused_sectors}
            pipe_entries = []
            for key_C in sorted(contracted_channels):
                contracted = contracted_channels[key_C]
                if len(contracted) != 1:
                    continue
                fused_sector = contracted[0]
                local_dim = int(np.prod(new_data[key_C].shape, dtype=int))
                pipe_entries.append(
                    FusionPipeEntry(
                        child_sectors=tuple(key_C),
                        fused_sector=fused_sector,
                        slot=slot_counts[fused_sector],
                        offset=offset_counts[fused_sector],
                        local_dim=local_dim,
                        selected_shape=tuple(int(x) for x in new_data[key_C].shape),
                    )
                )
                slot_counts[fused_sector] += 1
                offset_counts[fused_sector] += local_dim
            contracted_pipe = FusionPipe.from_entries(
                child_legs=tuple(range(len(new_qns))),
                child_sector_lists=tuple(tuple(leg_qns) for leg_qns in new_qns),
                child_dirs=tuple(new_dirs),
                fused_sectors=fused_sectors,
                entries=tuple(pipe_entries),
                orientation=1,
                coupling="contracted",
            )
            metadata["contracted_fusion_leg"] = FusionLeg(
                child_legs=tuple(range(len(new_qns))),
                child_sector_lists=tuple(tuple(leg_qns) for leg_qns in new_qns),
                child_dirs=tuple(new_dirs),
                sectors=fused_sectors,
                orientation=1,
                coupling="contracted",
                selected_channel=None,
                pipe=contracted_pipe,
            )

    return NonabelianTensor(
        new_data,
        new_qns,
        new_dirs,
        fusion_legs=new_fusion_legs,
        metadata=metadata,
    )


def merge_mps_sites(A, B):
    """
    Merge neighboring non-Abelian MPS sites with standard ``(L, P, R)`` layout.

    The returned two-site tensor has layout ``(L, P_left, P_right, R)``.
    """
    if A.rank != 3 or B.rank != 3:
        raise ValueError("merge_mps_sites expects rank-3 site tensors.")
    A = normalize_site_tensor_layout(A)
    B = normalize_site_tensor_layout(B)
    return tensordot(A, B, axes=([2], [0]))


def combine_legs(tensor, legs, new_axis=None, fusion_leg=None, use_cg=False):
    """
    Combine multiple legs into one fused leg using a fixed fusion map.

    The fused axis is packed by concatenating the reshaped child-leg blocks in
    the order implied by ``fusion_leg.slot_for(...)``.

    If ``use_cg=True``, the child axes are transformed from the uncoupled product
    basis into the coupled reduced basis implied by ``fusion_leg``. For two
    legs this is the ordinary CG transform; for longer products it uses the
    explicit reduced bond spaces defined in :mod:`pyqed.mps.nonabelian.coupling`.
    """
    if not isinstance(tensor, NonabelianTensor):
        raise TypeError("combine_legs expects a NonabelianTensor.")

    legs = tuple(int(ax) for ax in legs)
    if len(legs) == 0:
        raise ValueError("combine_legs requires at least one axis.")
    if len(set(legs)) != len(legs):
        raise ValueError(f"combine_legs got duplicate axes: {legs!r}")
    if any(ax < 0 or ax >= tensor.rank for ax in legs):
        raise ValueError(f"combine_legs axes out of range for rank-{tensor.rank} tensor: {legs!r}")

    free_axes = [ax for ax in range(tensor.rank) if ax not in legs]
    new_rank = tensor.rank - len(legs) + 1
    if new_axis is None:
        new_axis = min(legs)
    if new_axis < 0:
        new_axis += new_rank
    if new_axis < 0 or new_axis >= new_rank:
        raise ValueError(f"Invalid new_axis {new_axis} for resulting rank {new_rank}.")

    if fusion_leg is None:
        child_sector_lists = [tensor.qns[ax] for ax in legs]
        child_dirs = [tensor.dirs[ax] for ax in legs]
        fusion_leg = FusionLeg.from_children(
            legs,
            child_sector_lists,
            child_dirs=child_dirs,
            orientation=child_dirs[0],
            coupling="fixed",
        )
    output_spec = []
    free_iter = iter(free_axes)
    for out_pos in range(new_rank):
        if out_pos == new_axis:
            output_spec.append(None)
        else:
            output_spec.append(next(free_iter))

    perm = []
    for spec in output_spec:
        if spec is None:
            perm.extend(legs)
        else:
            perm.append(spec)

    new_qns = []
    new_dirs = []
    new_fusion_legs = []
    for spec in output_spec:
        if spec is None:
            new_qns.append(list(fusion_leg.sectors))
            new_dirs.append(fusion_leg.orientation)
            new_fusion_legs.append(fusion_leg)
        else:
            new_qns.append(tensor.qns[spec][:])
            new_dirs.append(tensor.dirs[spec])
            new_fusion_legs.append(tensor.fusion_legs[spec])

    packed = {}
    pipe_entries = {}
    split_basis_maps = {}
    for key, block in tensor.data.items():
        child_combo = tuple(key[ax] for ax in legs)
        block_perm = np.transpose(block, perm)
        before = block_perm.shape[:new_axis]
        selected_shape = block_perm.shape[new_axis:new_axis + len(legs)]
        after = block_perm.shape[new_axis + len(legs):]
        if use_cg:
            fused_sectors = tuple(sorted(set(fusion_leg.candidate_sectors(child_combo))))
            if fusion_leg.selected_channel is not None:
                fused_sectors = tuple(
                    sector for sector in fused_sectors if sector == fusion_leg.selected_channel
                )
            product_dim = int(np.prod(selected_shape, dtype=int))
            block_flat = block_perm.reshape(
                (-1, product_dim, int(np.prod(after or (1,), dtype=int)))
            )
            for fused_sector in fused_sectors:
                bond_space = fusion_leg.bond_space(
                    child_combo,
                    fused_sector,
                    scheme=fusion_leg.coupling_scheme,
                )
                reduced_degeneracy_cg = (
                    product_dim == 1
                    and all(int(dim) == 1 for dim in selected_shape)
                    and bond_space.product_dim != 1
                )
                if product_dim != bond_space.product_dim and not reduced_degeneracy_cg:
                    raise ValueError(
                        "combine_legs(use_cg=True) requires child-leg block dimensions to match the "
                        "explicit uncoupled irrep basis. "
                        f"Got child shape {selected_shape!r} but reduced basis expects {bond_space.product_dim} states."
                    )
                for channel, basis in zip(bond_space.channels, bond_space.basis_matrices):
                    if reduced_degeneracy_cg:
                        local_dim = 1
                        block_pack = block_flat.reshape(before + (local_dim,) + after)
                        split_basis_maps[(child_combo, fused_sector, channel.slot)] = np.eye(
                            1,
                            dtype=float,
                        )
                    else:
                        local_dim = bond_space.fused_dim
                        block_coupled = np.einsum("bxa,xy->bya", block_flat, basis, optimize=True)
                        block_pack = block_coupled.reshape(before + (local_dim,) + after)
                    slot = channel.slot
                    key_out = []
                    for spec in output_spec:
                        if spec is None:
                            key_out.append(fused_sector)
                        else:
                            key_out.append(key[spec])
                    key_out = tuple(key_out)
                    packed.setdefault(key_out, []).append(
                        {
                            "slot": slot,
                            "child_combo": child_combo,
                            "fused_sector": fused_sector,
                            "selected_shape": tuple(int(x) for x in selected_shape),
                            "local_dim": local_dim,
                            "block": block_pack,
                        }
                    )
                    pipe_key = (child_combo, fused_sector, slot)
                    entry = pipe_entries.get(pipe_key)
                    if entry is None:
                        pipe_entries[pipe_key] = {
                            "slot": slot,
                            "offset": None,
                            "local_dim": local_dim,
                            "selected_shape": tuple(int(x) for x in selected_shape),
                        }
                    else:
                        if (
                            entry["slot"] != slot
                            or entry["local_dim"] != local_dim
                            or entry["selected_shape"] != tuple(int(x) for x in selected_shape)
                        ):
                            raise ValueError(
                                "combine_legs requires a consistent packed shape for each child-sector tuple/channel. "
                                f"Got incompatible entries for {pipe_key!r}."
                            )
        else:
            fused_sector = fusion_leg.resolve_sector(child_combo)
            slot = fusion_leg.slot_for(child_combo, fused_sector)

            key_out = []
            for spec in output_spec:
                if spec is None:
                    key_out.append(fused_sector)
                else:
                    key_out.append(key[spec])
            key_out = tuple(key_out)

            local_dim = int(np.prod(selected_shape, dtype=int))
            block_pack = block_perm.reshape(before + (local_dim,) + after)

            packed.setdefault(key_out, []).append(
                {
                    "slot": slot,
                    "child_combo": child_combo,
                    "fused_sector": fused_sector,
                    "selected_shape": tuple(int(x) for x in selected_shape),
                    "local_dim": local_dim,
                    "block": block_pack,
                }
            )
            pipe_key = (child_combo, fused_sector, slot)
            entry = pipe_entries.get(pipe_key)
            if entry is None:
                pipe_entries[pipe_key] = {
                    "slot": slot,
                    "offset": None,
                    "local_dim": local_dim,
                    "selected_shape": tuple(int(x) for x in selected_shape),
                }
            else:
                if entry["slot"] != slot or entry["local_dim"] != local_dim or entry["selected_shape"] != tuple(int(x) for x in selected_shape):
                    raise ValueError(
                        "combine_legs requires a consistent packed shape for each child-sector tuple. "
                        f"Got incompatible entries for {pipe_key!r}."
                    )

    channel_order = {}
    for child_combo, fused_sector, slot in pipe_entries:
        channel_order.setdefault(fused_sector, []).append((slot, child_combo))
    for fused_sector, order in channel_order.items():
        channel_order[fused_sector] = tuple(
            (child_combo, fused_sector, slot)
            for slot, child_combo in sorted(order, key=lambda item: (item[0], item[1]))
        )

    new_data = {}
    for key_out, entries in packed.items():
        fused_sector = key_out[new_axis]
        ordered_channels = channel_order[fused_sector]
        if not entries:
            continue

        entry_map = {
            (item["child_combo"], item["fused_sector"], item["slot"]): item
            for item in entries
        }
        sample_block = entries[0]["block"]
        before_shape = sample_block.shape[:new_axis]
        after_shape = sample_block.shape[new_axis + 1 :]
        block_dtype = np.result_type(*(item["block"].dtype for item in entries))

        blocks = []
        offset = 0
        for pipe_key in ordered_channels:
            item = entry_map.get(pipe_key)
            if item is None:
                local_dim = pipe_entries[pipe_key]["local_dim"]
                block_piece = np.zeros(
                    before_shape + (local_dim,) + after_shape,
                    dtype=block_dtype,
                )
            else:
                local_dim = item["local_dim"]
                block_piece = item["block"]
            blocks.append(block_piece)
            if pipe_entries[pipe_key]["offset"] is None:
                pipe_entries[pipe_key]["offset"] = offset
            elif pipe_entries[pipe_key]["offset"] != offset:
                raise ValueError(
                    "combine_legs found inconsistent packed offsets for the same fused-sector channel "
                    f"{pipe_key!r}."
                )
            offset += local_dim
        new_data[key_out] = np.concatenate(blocks, axis=new_axis)

    pipe = FusionPipe.from_entries(
        child_legs=fusion_leg.child_legs,
        child_sector_lists=fusion_leg.child_sector_lists,
        child_dirs=fusion_leg.child_dirs,
        fused_sectors=fusion_leg.sectors,
        entries=tuple(
            FusionPipeEntry(
                child_sectors=child_combo,
                fused_sector=fused_sector,
                slot=item["slot"],
                offset=item["offset"],
                local_dim=item["local_dim"],
                selected_shape=item["selected_shape"],
            )
            for (child_combo, fused_sector, _slot), item in sorted(
                pipe_entries.items(),
                key=lambda kv: (kv[0][1], kv[1]["slot"], kv[0][0]),
            )
        ),
        orientation=fusion_leg.orientation,
        coupling=fusion_leg.coupling_scheme if use_cg else fusion_leg.coupling,
        selected_channel=fusion_leg.selected_channel,
    )
    fused_leg = fusion_leg.with_pipe(pipe)
    new_fusion_legs[new_axis] = fused_leg

    metadata = tensor.metadata.copy()
    if split_basis_maps:
        axis_maps = dict(metadata.get("split_basis_maps") or {})
        axis_maps[new_axis] = {
            **dict(axis_maps.get(new_axis, {})),
            **split_basis_maps,
        }
        metadata["split_basis_maps"] = axis_maps

    return NonabelianTensor(
        new_data,
        new_qns,
        new_dirs,
        fusion_legs=new_fusion_legs,
        metadata=metadata,
    )


def split_legs(tensor, axis):
    """
    Split a previously fused leg created by :func:`combine_legs`.
    """
    if not isinstance(tensor, NonabelianTensor):
        raise TypeError("split_legs expects a NonabelianTensor.")

    axis = int(axis)
    if axis < 0:
        axis += tensor.rank
    if axis < 0 or axis >= tensor.rank:
        raise ValueError(f"split_legs axis {axis} out of range for rank-{tensor.rank} tensor.")

    fusion_leg = tensor.fusion_legs[axis]
    if fusion_leg is None:
        raise ValueError(f"Tensor leg {axis} has no FusionLeg metadata to split.")

    child_sector_lists = fusion_leg.child_sector_lists
    child_dirs = fusion_leg.child_dirs
    if not child_sector_lists or not child_dirs:
        raise ValueError("FusionLeg needs child_sector_lists and child_dirs to support split_legs.")
    if fusion_leg.pipe is None:
        raise ValueError("split_legs requires a FusionLeg with an attached FusionPipe.")

    new_qns = []
    new_dirs = []
    new_fusion_legs = []
    for i in range(axis):
        new_qns.append(tensor.qns[i][:])
        new_dirs.append(tensor.dirs[i])
        new_fusion_legs.append(tensor.fusion_legs[i])
    for sectors, direction in zip(child_sector_lists, child_dirs):
        new_qns.append(list(sectors))
        new_dirs.append(direction)
        new_fusion_legs.append(None)
    for i in range(axis + 1, tensor.rank):
        new_qns.append(tensor.qns[i][:])
        new_dirs.append(tensor.dirs[i])
        new_fusion_legs.append(tensor.fusion_legs[i])

    split_basis_maps = {}
    if tensor.metadata:
        split_basis_maps = (tensor.metadata.get("split_basis_maps") or {}).get(axis, {})

    new_data = {}
    for key, block in tensor.data.items():
        fused_sector = key[axis]
        layout_entries = fusion_leg.pipe.entries_for_sector(fused_sector)
        if not layout_entries:
            raise ValueError(f"Missing FusionPipe layout for fused sector {fused_sector!r}.")

        before = block.shape[:axis]
        after = block.shape[axis + 1:]
        for entry in layout_entries:
            slicer = [slice(None)] * block.ndim
            slicer[axis] = slice(entry.offset, entry.offset + entry.local_dim)
            sliced = block[tuple(slicer)]
            if fusion_leg.pipe.coupling in {"cg", "left", "right"}:
                transform = split_basis_maps.get(
                    (entry.child_sectors, entry.fused_sector, entry.slot)
                )
                if transform is None:
                    bond_space = fusion_leg.bond_space(
                        entry.child_sectors,
                        entry.fused_sector,
                        scheme=normalize_coupling_scheme(fusion_leg.pipe.coupling, default="left"),
                    )
                    basis_by_slot = {
                        channel.slot: basis
                        for channel, basis in zip(bond_space.channels, bond_space.basis_matrices)
                    }
                    transform = basis_by_slot.get(entry.slot)
                if transform is None:
                    raise ValueError(
                        f"Missing reduced basis transform for slot {entry.slot} and "
                        f"child sectors {entry.child_sectors!r}."
                    )
                flat = sliced.reshape((-1, entry.local_dim, int(np.prod(after or (1,), dtype=int))))
                expanded = np.einsum("bya,xy->bxa", flat, transform, optimize=True)
                piece = expanded.reshape(before + tuple(entry.selected_shape) + after)
            else:
                piece = sliced.reshape(before + tuple(entry.selected_shape) + after)

            key_out = tuple(key[:axis]) + tuple(entry.child_sectors) + tuple(key[axis + 1:])
            if key_out in new_data:
                new_data[key_out] = new_data[key_out] + piece
            else:
                new_data[key_out] = piece

    return NonabelianTensor(
        new_data,
        new_qns,
        new_dirs,
        fusion_legs=new_fusion_legs,
        metadata=tensor.metadata.copy(),
    )


def recouple_fused_leg(tensor, axis, target_scheme):
    """
    Recouple a fused reduced leg between equivalent parenthesization schemes.

    The physical fused sector is unchanged; only the multiplicity-channel basis
    is rotated using the explicit reduced bond-space recoupling matrix.
    """
    if not isinstance(tensor, NonabelianTensor):
        raise TypeError("recouple_fused_leg expects a NonabelianTensor.")

    axis = int(axis)
    if axis < 0:
        axis += tensor.rank
    if axis < 0 or axis >= tensor.rank:
        raise ValueError(f"recouple_fused_leg axis {axis} out of range for rank-{tensor.rank} tensor.")

    fusion_leg = tensor.fusion_legs[axis]
    if fusion_leg is None or fusion_leg.pipe is None:
        raise ValueError(f"Tensor leg {axis} has no fused-leg metadata to recouple.")

    source_scheme = fusion_leg.coupling_scheme
    target_scheme = normalize_coupling_scheme(target_scheme, default=source_scheme)
    if target_scheme == source_scheme:
        return tensor.copy()

    pipe = fusion_leg.pipe
    new_data = {}
    for key, block in tensor.data.items():
        fused_sector = key[axis]
        entries = sorted(pipe.entries_for_sector(fused_sector), key=lambda entry: entry.slot)
        if not entries:
            continue
        child_combo = entries[0].child_sectors
        if any(entry.child_sectors != child_combo for entry in entries):
            raise ValueError(
                "recouple_fused_leg currently requires a fixed child-sector tuple per fused-sector block."
            )
        bond_space_source = fusion_leg.bond_space(child_combo, fused_sector, scheme=source_scheme)
        bond_space_target = fusion_leg.bond_space(child_combo, fused_sector, scheme=target_scheme)
        recouple = bond_space_source.recouple_to(bond_space_target)
        if recouple.shape[0] != len(entries):
            raise ValueError(
                f"Recoupling multiplicity mismatch for fused sector {fused_sector!r}: "
                f"matrix has {recouple.shape[0]} rows but packed tensor has {len(entries)} channels."
            )
        fused_dim = bond_space_source.fused_dim
        before = block.shape[:axis]
        after = block.shape[axis + 1:]
        tail_dim = int(np.prod(after or (1,), dtype=int))
        stacked = []
        for entry in entries:
            slicer = [slice(None)] * block.ndim
            slicer[axis] = slice(entry.offset, entry.offset + entry.local_dim)
            sliced = block[tuple(slicer)]
            stacked.append(sliced.reshape(before + (fused_dim,) + after))
        stacked = np.stack(stacked, axis=axis)
        lead_dim = int(np.prod(before or (1,), dtype=int))
        stacked = stacked.reshape((lead_dim, len(entries), fused_dim, tail_dim))
        rotated = np.einsum("ji,lifb->ljfb", recouple, stacked, optimize=True)
        rotated = rotated.reshape(before + (len(entries) * fused_dim,) + after)
        new_data[key] = rotated

    new_pipe = FusionPipe.from_entries(
        child_legs=pipe.child_legs,
        child_sector_lists=pipe.child_sector_lists,
        child_dirs=pipe.child_dirs,
        fused_sectors=pipe.fused_sectors,
        entries=pipe.entries,
        orientation=pipe.orientation,
        coupling=target_scheme,
        selected_channel=pipe.selected_channel,
    )
    new_fusion_leg = FusionLeg(
        child_legs=fusion_leg.child_legs,
        child_sector_lists=fusion_leg.child_sector_lists,
        child_dirs=fusion_leg.child_dirs,
        sectors=fusion_leg.sectors,
        orientation=fusion_leg.orientation,
        coupling=target_scheme,
        coupling_channels=fusion_leg.coupling_channels,
        fusion_map=fusion_leg.fusion_map,
        selected_channel=fusion_leg.selected_channel,
        pipe=new_pipe,
    )
    new_fusion_legs = tensor.fusion_legs[:]
    new_fusion_legs[axis] = new_fusion_leg
    return NonabelianTensor(
        new_data,
        [leg_qns[:] for leg_qns in tensor.qns],
        tensor.dirs[:],
        fusion_legs=new_fusion_legs,
        metadata=tensor.metadata.copy(),
    )
