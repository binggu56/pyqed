#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-layout reduced contractions for :mod:`pyqed.mps.nonabelian`.
"""

from __future__ import annotations

import numpy as np

from .coupling import normalize_coupling_scheme, reduced_bond_space
from pyqed.mps.su2 import SU2Irrep, SpinChargeSector, fuse_charge_spin_sectors
from pyqed.symmetry import IrrepTensor, Leg
from .tensor import (
    FusionLeg,
    FusionPipe,
    FusionPipeEntry,
    IdentityBasisTransform,
)


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
    if not isinstance(site, IrrepTensor) or site.rank != 3:
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
    if not isinstance(site, IrrepTensor) or site.rank != 3:
        return site
    if _site_is_left_phys_right(site):
        return site
    return site.transpose(0, 2, 1)


def tensordot(A, B, axes):
    """
    Fixed-layout contraction for :class:`IrrepTensor`.

    This helper intentionally implements only the simplest non-Abelian case:
    contracted legs must carry identical reduced sectors and identical
    fusion-edge metadata.  That is enough to support fixed fusion-tree layouts
    without yet introducing explicit recoupling coefficients.
    """
    if not isinstance(A, IrrepTensor) or not isinstance(B, IrrepTensor):
        raise TypeError("tensordot expects two IrrepTensor objects.")

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
            contracted_channels.setdefault(key_C, set()).add(contracted)
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
        if len(a_ax) == 1 and len(b_ax) == 1:
            metadata["contracted_channels"] = {
                key: tuple(channel[0] for channel in sorted(channels))
                for key, channels in contracted_channels.items()
            }
        else:
            metadata["contracted_channels"] = {
                key: tuple(sorted(channels))
                for key, channels in contracted_channels.items()
            }
        if len(a_ax) == 1 and len(b_ax) == 1:
            fused_sectors = tuple(
                sorted(
                    {
                        contracted[0]
                        for channels in contracted_channels.values()
                        for contracted in channels
                    }
                )
            )
            slot_counts = {sector: 0 for sector in fused_sectors}
            offset_counts = {sector: 0 for sector in fused_sectors}
            pipe_entries = []
            for key_C in sorted(contracted_channels):
                local_dim = int(np.prod(new_data[key_C].shape, dtype=int))
                for contracted in sorted(contracted_channels[key_C]):
                    if len(contracted) != 1:
                        continue
                    fused_sector = contracted[0]
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

    return IrrepTensor(
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
    merged = tensordot(A, B, axes=([2], [0]))
    channel_blocks = {}
    for (q_left, q_phys1, q_mid), left_block in A.data.items():
        for (q_mid_right, q_phys2, q_right), right_block in B.data.items():
            if q_mid_right != q_mid:
                continue
            key = (q_left, q_phys1, q_mid, q_phys2, q_right)
            contribution = np.tensordot(
                np.asarray(left_block),
                np.asarray(right_block),
                axes=([2], [0]),
            )
            if key in channel_blocks:
                channel_blocks[key] = channel_blocks[key] + contribution
            else:
                channel_blocks[key] = contribution
    merged.metadata["contracted_channel_blocks"] = channel_blocks
    merged.metadata["contracted_channel_blocks_current"] = True
    # Two-site decomposition consumes contracted bond channels as one-element
    # axis tuples, while the public single-axis tensordot metadata exposes the
    # contracted sectors directly.
    channels = merged.metadata.get("contracted_channels")
    if channels:
        merged.metadata["contracted_channels"] = {
            key: tuple((sector,) for sector in sectors)
            for key, sectors in channels.items()
        }
    return merged


def merge_mps_sites_from_packed(A, B, packed):
    """
    Restore a two-site reduced tensor produced by the C++ sweep owner.

    The numerical contraction and channel aggregation have already happened
    in C++; this function only reconnects the packed sector labels to the
    Python tensor metadata still consumed by the local solver and SVD.
    """
    if A.rank != 3 or B.rank != 3:
        raise ValueError("merge_mps_sites_from_packed expects rank-3 site tensors.")
    A = normalize_site_tensor_layout(A)
    B = normalize_site_tensor_layout(B)
    _validate_contraction_metadata(A, B, [2], [0])

    def sector_label(sector):
        charge = getattr(sector, "charge", None)
        irrep = getattr(sector, "irrep", None)
        two_j = getattr(irrep, "two_j", None)
        if two_j is None:
            two_j = getattr(sector, "two_j", None)
        if charge is None or two_j is None:
            raise TypeError(
                "Packed C++ MPS merge requires charge/two_j sector labels."
            )
        return int(charge), int(two_j)

    sectors = {}
    for leg in A.qns + B.qns:
        for sector in leg:
            label = sector_label(sector)
            previous = sectors.setdefault(label, sector)
            if previous != sector:
                raise ValueError(
                    f"Ambiguous SU(2) sector label {label!r} in packed MPS merge."
                )

    def unpack(arena, label_rank, value_rank):
        values = np.asarray(arena["values"], dtype=np.float64).reshape(-1)
        offsets = np.asarray(arena["offsets"], dtype=np.int64).reshape(-1)
        labels = np.asarray(arena["labels"], dtype=np.int64).reshape(
            (-1, 2 * label_rank)
        )
        shape_offsets = np.asarray(
            arena["shape_offsets"],
            dtype=np.int64,
        ).reshape(-1)
        shapes = np.asarray(arena["shapes"], dtype=np.int64).reshape(-1)
        nblocks = max(0, offsets.size - 1)
        if labels.shape[0] != nblocks or shape_offsets.size != nblocks + 1:
            raise ValueError("Malformed packed C++ MPS merge topology.")
        data = {}
        for block in range(nblocks):
            shape = tuple(
                int(value)
                for value in shapes[
                    int(shape_offsets[block]):int(shape_offsets[block + 1])
                ]
            )
            if len(shape) != value_rank:
                raise ValueError(
                    f"Packed C++ MPS merge block has rank {len(shape)}, "
                    f"expected {value_rank}."
                )
            key = []
            for axis in range(label_rank):
                label = (
                    int(labels[block, 2 * axis]),
                    int(labels[block, 2 * axis + 1]),
                )
                sector = sectors.get(label)
                if sector is None:
                    sector = SpinChargeSector(label[0], SU2Irrep(label[1]))
                    sectors[label] = sector
                key.append(sector)
            key = tuple(key)
            start = int(offsets[block])
            stop = int(offsets[block + 1])
            if int(np.prod(shape, dtype=int)) != stop - start:
                raise ValueError("Packed C++ MPS merge shape disagrees with its offsets.")
            data[key] = np.array(values[start:stop], copy=True).reshape(shape)
        return data

    data = unpack(packed["merged"], 4, 4)
    channel_blocks = unpack(packed["channels"], 5, 4)
    new_qns = [A.qns[0][:], A.qns[1][:], B.qns[1][:], B.qns[2][:]]
    new_dirs = [A.dirs[0], A.dirs[1], B.dirs[1], B.dirs[2]]
    new_fusion_legs = [
        A.fusion_legs[0],
        A.fusion_legs[1],
        B.fusion_legs[1],
        B.fusion_legs[2],
    ]

    if _metadata_equal(A.metadata, B.metadata):
        metadata = A.metadata.copy()
    elif A.metadata or B.metadata:
        metadata = {
            "left_metadata": A.metadata.copy(),
            "right_metadata": B.metadata.copy(),
        }
    else:
        metadata = {}

    contracted_channels = {}
    for q_left, q_phys1, q_mid, q_phys2, q_right in channel_blocks:
        key = (q_left, q_phys1, q_phys2, q_right)
        contracted_channels.setdefault(key, set()).add(q_mid)
    contracted_channels = {
        key: tuple(sorted(channels))
        for key, channels in contracted_channels.items()
    }
    metadata["contracted_channels"] = contracted_channels
    metadata["contracted_channel_blocks"] = channel_blocks
    metadata["contracted_channel_blocks_current"] = True

    if contracted_channels:
        fused_sectors = tuple(
            sorted(
                {
                    sector
                    for channels in contracted_channels.values()
                    for sector in channels
                }
            )
        )
        slot_counts = {sector: 0 for sector in fused_sectors}
        offset_counts = {sector: 0 for sector in fused_sectors}
        pipe_entries = []
        for key in sorted(contracted_channels):
            local_dim = int(np.prod(data[key].shape, dtype=int))
            for sector in contracted_channels[key]:
                pipe_entries.append(
                    FusionPipeEntry(
                        child_sectors=tuple(key),
                        fused_sector=sector,
                        slot=slot_counts[sector],
                        offset=offset_counts[sector],
                        local_dim=local_dim,
                        selected_shape=tuple(int(x) for x in data[key].shape),
                    )
                )
                slot_counts[sector] += 1
                offset_counts[sector] += local_dim
        contracted_pipe = FusionPipe.from_entries(
            child_legs=tuple(range(4)),
            child_sector_lists=tuple(tuple(leg_qns) for leg_qns in new_qns),
            child_dirs=tuple(new_dirs),
            fused_sectors=fused_sectors,
            entries=tuple(pipe_entries),
            orientation=1,
            coupling="contracted",
        )
        metadata["contracted_fusion_leg"] = FusionLeg(
            child_legs=tuple(range(4)),
            child_sector_lists=tuple(tuple(leg_qns) for leg_qns in new_qns),
            child_dirs=tuple(new_dirs),
            sectors=fused_sectors,
            orientation=1,
            coupling="contracted",
            selected_channel=None,
            pipe=contracted_pipe,
        )

    return IrrepTensor(
        data,
        new_qns,
        new_dirs,
        fusion_legs=new_fusion_legs,
        metadata=metadata,
    )


def split_mps_sites_from_packed(A, B, packed):
    """Restore the two rank-3 site views produced by the C++ bond split."""

    if A.rank != 3 or B.rank != 3:
        raise ValueError("split_mps_sites_from_packed expects rank-3 sites.")
    A = normalize_site_tensor_layout(A)
    B = normalize_site_tensor_layout(B)

    def sector_label(sector):
        return int(sector.charge), int(sector.irrep.two_j)

    sectors = {}
    for leg in A.qns + B.qns:
        for sector in leg:
            sectors.setdefault(sector_label(sector), sector)
    bond_labels = np.asarray(packed["bond_labels"], dtype=np.int64).reshape(-1, 2)
    bond_dims = np.asarray(packed["bond_dims"], dtype=np.int64).reshape(-1)
    if bond_labels.shape[0] != bond_dims.size or np.any(bond_dims <= 0):
        raise ValueError("Malformed C++ split bond-sector topology.")
    bond_sectors = []
    for charge, two_j in bond_labels:
        label = (int(charge), int(two_j))
        bond_sectors.append(
            sectors.setdefault(
                label,
                SpinChargeSector(label[0], SU2Irrep(label[1])),
            )
        )
    bond_qns = [
        sector
        for sector, dim in zip(bond_sectors, bond_dims)
        for _ in range(int(dim))
    ]

    def unpack(arena):
        values = np.asarray(arena["values"], dtype=np.float64).reshape(-1)
        offsets = np.asarray(arena["offsets"], dtype=np.int64).reshape(-1)
        labels = np.asarray(arena["labels"], dtype=np.int64).reshape(-1, 6)
        shape_offsets = np.asarray(
            arena["shape_offsets"], dtype=np.int64
        ).reshape(-1)
        shapes = np.asarray(arena["shapes"], dtype=np.int64).reshape(-1)
        if (
            offsets.size != labels.shape[0] + 1
            or shape_offsets.size != offsets.size
        ):
            raise ValueError("Malformed C++ split-site arena.")
        data = {}
        for block in range(labels.shape[0]):
            shape = tuple(
                int(value)
                for value in shapes[
                    int(shape_offsets[block]):int(shape_offsets[block + 1])
                ]
            )
            if len(shape) != 3:
                raise ValueError("A C++ split-site block is not rank three.")
            key = tuple(
                sectors.setdefault(
                    (
                        int(labels[block, 2 * axis]),
                        int(labels[block, 2 * axis + 1]),
                    ),
                    SpinChargeSector(
                        int(labels[block, 2 * axis]),
                        SU2Irrep(int(labels[block, 2 * axis + 1])),
                    ),
                )
                for axis in range(3)
            )
            start = int(offsets[block])
            stop = int(offsets[block + 1])
            if int(np.prod(shape, dtype=int)) != stop - start:
                raise ValueError(
                    "A C++ split-site shape disagrees with its value range."
                )
            data[key] = np.array(values[start:stop], copy=True).reshape(shape)
        return data

    left_data = unpack(packed["left"])
    right_data = unpack(packed["right"])
    slot_counts = {sector: 0 for sector in bond_sectors}
    pipe_entries = []
    for sector, dim in zip(bond_sectors, bond_dims):
        for _ in range(int(dim)):
            slot = slot_counts[sector]
            pipe_entries.append(
                FusionPipeEntry(
                    child_sectors=(sector,),
                    fused_sector=sector,
                    slot=slot,
                    offset=slot,
                    local_dim=1,
                    selected_shape=(1,),
                )
            )
            slot_counts[sector] += 1
    bond_pipe = FusionPipe.from_entries(
        child_legs=(0,),
        child_sector_lists=(tuple(bond_sectors),),
        child_dirs=(1,),
        fused_sectors=tuple(bond_sectors),
        entries=tuple(pipe_entries),
        orientation=1,
        coupling="left",
    )
    bond_leg = FusionLeg(
        child_legs=(0,),
        child_sector_lists=(tuple(bond_sectors),),
        child_dirs=(1,),
        sectors=tuple(bond_sectors),
        orientation=1,
        coupling="left",
        pipe=bond_pipe,
    )
    right_bond_basis = Leg(
        tuple(bond_sectors),
        {
            sector: int(dim)
            for sector, dim in zip(bond_sectors, bond_dims)
        },
        direction=1,
        name="cpp-split-right-bond",
    )
    left_bond_basis = Leg(
        tuple(bond_sectors),
        right_bond_basis.dims,
        direction=-1,
        name="cpp-split-left-bond",
    )
    fully_reduced = (
        A.metadata.get("physical_basis") == "fully_reduced_su2"
        or B.metadata.get("physical_basis") == "fully_reduced_su2"
    )
    common_metadata = {
        "source": "cpp_active_bond_split",
        **({"physical_basis": "fully_reduced_su2"} if fully_reduced else {}),
    }
    left = IrrepTensor(
        left_data,
        [A.qns[0][:], A.qns[1][:], bond_qns],
        [A.dirs[0], A.dirs[1], A.dirs[2]],
        fusion_legs=[A.fusion_legs[0], A.fusion_legs[1], bond_leg],
        metadata={
            **common_metadata,
            "svd_role": "left",
            "bond_bases": {2: right_bond_basis},
        },
    )
    right = IrrepTensor(
        right_data,
        [bond_qns, B.qns[1][:], B.qns[2][:]],
        [B.dirs[0], B.dirs[1], B.dirs[2]],
        fusion_legs=[bond_leg, B.fusion_legs[1], B.fusion_legs[2]],
        metadata={
            **common_metadata,
            "svd_role": "right",
            "bond_bases": {0: left_bond_basis},
        },
    )
    singular_values = {}
    values = np.asarray(packed["singular_values"], dtype=float).reshape(-1)
    offsets = np.asarray(packed["singular_offsets"], dtype=np.int64).reshape(-1)
    if offsets.size != len(bond_sectors) + 1:
        raise ValueError("Malformed C++ split singular-value topology.")
    for index, sector in enumerate(bond_sectors):
        singular_values[sector] = np.diag(
            values[int(offsets[index]):int(offsets[index + 1])]
        )
    return left, right, singular_values


def mps_site_from_packed(
    template,
    arena,
    *,
    left_bond=None,
    right_bond=None,
    svd_role=None,
):
    """Restore one final MPS site exported by a C++-owned half sweep."""

    if template.rank != 3:
        raise ValueError("mps_site_from_packed expects a rank-3 site.")
    template = normalize_site_tensor_layout(template)

    def label(sector):
        return int(sector.charge), int(sector.irrep.two_j)

    sectors = {}
    for leg in template.qns:
        for sector in leg:
            sectors.setdefault(label(sector), sector)

    values = np.asarray(arena["values"], dtype=np.float64).reshape(-1)
    offsets = np.asarray(arena["offsets"], dtype=np.int64).reshape(-1)
    labels = np.asarray(arena["labels"], dtype=np.int64).reshape(-1, 6)
    shape_offsets = np.asarray(
        arena["shape_offsets"], dtype=np.int64
    ).reshape(-1)
    shapes = np.asarray(arena["shapes"], dtype=np.int64).reshape(-1)
    if offsets.size != labels.shape[0] + 1 or shape_offsets.size != offsets.size:
        raise ValueError("Malformed C++ final-site arena.")
    data = {}
    for block in range(labels.shape[0]):
        shape = tuple(
            int(value)
            for value in shapes[
                int(shape_offsets[block]):int(shape_offsets[block + 1])
            ]
        )
        if len(shape) != 3:
            raise ValueError("A C++ final-site block is not rank three.")
        key = tuple(
            sectors.setdefault(
                (
                    int(labels[block, 2 * axis]),
                    int(labels[block, 2 * axis + 1]),
                ),
                SpinChargeSector(
                    int(labels[block, 2 * axis]),
                    SU2Irrep(int(labels[block, 2 * axis + 1])),
                ),
            )
            for axis in range(3)
        )
        start = int(offsets[block])
        stop = int(offsets[block + 1])
        if int(np.prod(shape, dtype=int)) != stop - start:
            raise ValueError(
                "A C++ final-site shape disagrees with its value range."
            )
        data[key] = np.array(values[start:stop], copy=True).reshape(shape)

    qns = [leg[:] for leg in template.qns]
    fusion_legs = list(template.fusion_legs)
    bond_bases = {}

    def install_bond(axis, topology):
        bond_labels, bond_dims = topology
        bond_labels = np.asarray(bond_labels, dtype=np.int64).reshape(-1, 2)
        bond_dims = np.asarray(bond_dims, dtype=np.int64).reshape(-1)
        if bond_labels.shape[0] != bond_dims.size or np.any(bond_dims <= 0):
            raise ValueError("Malformed C++ final bond-sector topology.")
        bond_sectors = [
            sectors.setdefault(
                (int(charge), int(two_j)),
                SpinChargeSector(int(charge), SU2Irrep(int(two_j))),
            )
            for charge, two_j in bond_labels
        ]
        qns[axis] = [
            sector
            for sector, dim in zip(bond_sectors, bond_dims)
            for _ in range(int(dim))
        ]
        slot_counts = {sector: 0 for sector in bond_sectors}
        pipe_entries = []
        for sector, dim in zip(bond_sectors, bond_dims):
            for _ in range(int(dim)):
                slot = slot_counts[sector]
                pipe_entries.append(
                    FusionPipeEntry(
                        child_sectors=(sector,),
                        fused_sector=sector,
                        slot=slot,
                        offset=slot,
                        local_dim=1,
                        selected_shape=(1,),
                    )
                )
                slot_counts[sector] += 1
        bond_pipe = FusionPipe.from_entries(
            child_legs=(0,),
            child_sector_lists=(tuple(bond_sectors),),
            child_dirs=(1,),
            fused_sectors=tuple(bond_sectors),
            entries=tuple(pipe_entries),
            orientation=1,
            coupling="left",
        )
        fusion_legs[axis] = FusionLeg(
            child_legs=(0,),
            child_sector_lists=(tuple(bond_sectors),),
            child_dirs=(1,),
            sectors=tuple(bond_sectors),
            orientation=1,
            coupling="left",
            pipe=bond_pipe,
        )
        direction = -1 if axis == 0 else 1
        bond_bases[axis] = Leg(
            tuple(bond_sectors),
            {
                sector: int(dim)
                for sector, dim in zip(bond_sectors, bond_dims)
            },
            direction=direction,
            name=f"cpp-final-bond-{axis}",
        )

    if left_bond is not None:
        install_bond(0, left_bond)
    if right_bond is not None:
        install_bond(2, right_bond)
    fully_reduced = (
        template.metadata.get("physical_basis") == "fully_reduced_su2"
    )
    metadata = {
        "source": "cpp_owned_half_sweep",
        **({"physical_basis": "fully_reduced_su2"} if fully_reduced else {}),
        **({"svd_role": str(svd_role)} if svd_role is not None else {}),
        **({"bond_bases": bond_bases} if bond_bases else {}),
    }
    return IrrepTensor(
        data,
        qns,
        list(template.dirs),
        fusion_legs=fusion_legs,
        metadata=metadata,
    )


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
    if not isinstance(tensor, IrrepTensor):
        raise TypeError("combine_legs expects an IrrepTensor.")

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
                if product_dim != bond_space.product_dim:
                    raise ValueError(
                        "combine_legs(use_cg=True) requires child-leg block dimensions to match the "
                        "explicit uncoupled irrep basis. "
                        f"Got child shape {selected_shape!r} but reduced basis expects {bond_space.product_dim} states."
                    )
                for channel, basis in zip(bond_space.channels, bond_space.basis_matrices):
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

    return IrrepTensor(
        new_data,
        new_qns,
        new_dirs,
        fusion_legs=new_fusion_legs,
        metadata=tensor.metadata.copy(),
    )


def split_legs(tensor, axis):
    """
    Split a previously fused leg created by :func:`combine_legs`.
    """
    if not isinstance(tensor, IrrepTensor):
        raise TypeError("split_legs expects an IrrepTensor.")

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
                if isinstance(transform, IdentityBasisTransform):
                    piece = sliced.reshape(before + tuple(entry.selected_shape) + after)
                else:
                    flat = sliced.reshape(
                        (-1, entry.local_dim, int(np.prod(after or (1,), dtype=int)))
                    )
                    expanded = np.einsum("bya,xy->bxa", flat, transform, optimize=True)
                    piece = expanded.reshape(before + tuple(entry.selected_shape) + after)
            else:
                piece = sliced.reshape(before + tuple(entry.selected_shape) + after)

            key_out = tuple(key[:axis]) + tuple(entry.child_sectors) + tuple(key[axis + 1:])
            if key_out in new_data:
                new_data[key_out] = new_data[key_out] + piece
            else:
                new_data[key_out] = piece

    return IrrepTensor(
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
    if not isinstance(tensor, IrrepTensor):
        raise TypeError("recouple_fused_leg expects an IrrepTensor.")

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
        before = block.shape[:axis]
        after = block.shape[axis + 1:]
        lead_dim = int(np.prod(before or (1,), dtype=int))
        tail_dim = int(np.prod(after or (1,), dtype=int))
        rotated_block = np.zeros_like(block)

        entries_by_child_combo = {}
        for entry in entries:
            entries_by_child_combo.setdefault(entry.child_sectors, []).append(entry)

        for child_combo, child_entries in entries_by_child_combo.items():
            child_entries = sorted(child_entries, key=lambda entry: entry.slot)
            bond_space_source = fusion_leg.bond_space(child_combo, fused_sector, scheme=source_scheme)
            bond_space_target = fusion_leg.bond_space(child_combo, fused_sector, scheme=target_scheme)
            recouple = bond_space_source.recouple_to(bond_space_target)
            if recouple.shape != (len(child_entries), len(child_entries)):
                raise ValueError(
                    f"Recoupling multiplicity mismatch for child sectors {child_combo!r} "
                    f"and fused sector {fused_sector!r}: matrix has shape {recouple.shape} "
                    f"but packed tensor has {len(child_entries)} channels."
                )
            fused_dim = bond_space_source.fused_dim
            stacked = []
            for entry in child_entries:
                slicer = [slice(None)] * block.ndim
                slicer[axis] = slice(entry.offset, entry.offset + entry.local_dim)
                sliced = block[tuple(slicer)]
                stacked.append(sliced.reshape(before + (fused_dim,) + after))
            stacked = np.stack(stacked, axis=axis)
            stacked = stacked.reshape((lead_dim, len(child_entries), fused_dim, tail_dim))
            rotated = np.einsum("ji,lifb->ljfb", recouple, stacked, optimize=True)
            for target_index, entry in enumerate(child_entries):
                piece = rotated[:, target_index, :, :].reshape(
                    before + (entry.local_dim,) + after
                )
                slicer = [slice(None)] * block.ndim
                slicer[axis] = slice(entry.offset, entry.offset + entry.local_dim)
                rotated_block[tuple(slicer)] = piece
        new_data[key] = rotated_block

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
    return IrrepTensor(
        new_data,
        [leg_qns[:] for leg_qns in tensor.qns],
        tensor.dirs[:],
        fusion_legs=new_fusion_legs,
        metadata=tensor.metadata.copy(),
    )
