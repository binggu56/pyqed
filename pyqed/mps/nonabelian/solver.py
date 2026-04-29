#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local two-site operator helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.davidson import davidson
from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors

from .contraction import combine_legs, split_legs
from .tensor import NonabelianTensor

_BASIS_TRANSFORM_DENSE_MATVEC_SIZE = 0


@dataclass(frozen=True)
class PackedEntry:
    key: tuple
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass(frozen=True)
class ReducedStateLayout:
    entries: tuple[PackedEntry, ...]

    @property
    def size(self):
        return sum(entry.size for entry in self.entries)

    def basis_vector(self, index, *, dtype=complex):
        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError(f"Basis index {index} is out of range for size {self.size}.")
        for entry in self.entries:
            if entry.offset <= index < entry.offset + entry.size:
                flat = np.zeros(entry.size, dtype=dtype)
                flat[index - entry.offset] = 1.0
                return ReducedStateVector(
                    layout=self,
                    blocks={entry.key: flat.reshape(entry.shape)},
                )
        raise IndexError(f"Basis index {index} is out of range for size {self.size}.")

    def from_packed(self, vector):
        vector = np.asarray(vector)
        blocks = {}
        for entry in self.entries:
            piece = vector[entry.offset:entry.offset + entry.size]
            if np.linalg.norm(piece) > 0.0:
                blocks[entry.key] = piece.reshape(entry.shape)
        return ReducedStateVector(layout=self, blocks=blocks)

    def to_packed(self, state, *, dtype=None):
        if dtype is None:
            present = [np.asarray(block).dtype for block in state.blocks.values()]
            dtype = np.result_type(*(present or [float]))
        vec = np.zeros(self.size, dtype=dtype)
        for entry in self.entries:
            if entry.key not in state.blocks:
                continue
            block = np.asarray(state.blocks[entry.key])
            vec[entry.offset:entry.offset + entry.size] = block.reshape(entry.size)
        return vec


@dataclass(frozen=True)
class ReducedStateVector:
    layout: ReducedStateLayout
    blocks: dict

    def to_packed(self, *, dtype=None):
        return self.layout.to_packed(self, dtype=dtype)

    def with_blocks(self, blocks):
        return ReducedStateVector(layout=self.layout, blocks=blocks)


@dataclass(frozen=True)
class ReducedDiagonalPreconditioner:
    layout: ReducedStateLayout
    h_blocks: dict
    n_blocks: dict

    @classmethod
    def from_packed_diagonals(cls, layout, h_diag, *, n_diag=None):
        h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
        if h_diag.size != layout.size:
            raise ValueError("Hamiltonian diagonal must match the reduced-state layout size.")
        if n_diag is None:
            n_diag = np.ones(layout.size, dtype=float)
        else:
            n_diag = np.asarray(n_diag, dtype=float).reshape(-1)
            if n_diag.size != layout.size:
                raise ValueError("Norm diagonal must match the reduced-state layout size.")

        h_blocks = {}
        n_blocks = {}
        for entry in layout.entries:
            h_blocks[entry.key] = h_diag[entry.offset:entry.offset + entry.size].reshape(entry.shape)
            n_blocks[entry.key] = n_diag[entry.offset:entry.offset + entry.size].reshape(entry.shape)
        return cls(layout=layout, h_blocks=h_blocks, n_blocks=n_blocks)

    def apply(self, resid, theta):
        if resid.layout != self.layout:
            raise ValueError("Reduced preconditioner layout must match the residual layout.")
        blocks = {}
        for key, block in resid.blocks.items():
            denom = theta * np.asarray(self.n_blocks[key]) - np.asarray(self.h_blocks[key])
            safe = np.where(
                np.abs(denom) > 1e-12,
                denom,
                np.where(denom >= 0, 1e-12, -1e-12),
            )
            corrected = np.asarray(block) / safe
            if np.linalg.norm(corrected.reshape(-1)) > 1.0e-15:
                blocks[key] = corrected
        return ReducedStateVector(layout=self.layout, blocks=blocks)


@dataclass(frozen=True)
class PackedBlockPreconditioner:
    layout: tuple[PackedEntry, ...]
    h_blocks: tuple[np.ndarray | None, ...]
    n_blocks: tuple[np.ndarray | None, ...]

    @classmethod
    def from_layout_blocks(cls, layout, h_blocks, *, n_blocks=None):
        layout = tuple(layout)
        if len(h_blocks) != len(layout):
            raise ValueError("Hamiltonian block list must match the packed layout length.")
        if n_blocks is None:
            n_blocks = [None] * len(layout)
        elif len(n_blocks) != len(layout):
            raise ValueError("Norm block list must match the packed layout length.")

        normalized_h = []
        normalized_n = []
        for entry, h_block, n_block in zip(layout, h_blocks, n_blocks):
            if h_block is None:
                normalized_h.append(None)
            else:
                arr = np.asarray(h_block, dtype=complex)
                if arr.shape != (entry.size, entry.size):
                    raise ValueError(
                        f"Hamiltonian block for {entry.key!r} has shape {arr.shape!r}, "
                        f"expected {(entry.size, entry.size)!r}."
                    )
                normalized_h.append(arr)
            if n_block is None:
                normalized_n.append(None)
            else:
                arr = np.asarray(n_block, dtype=complex)
                if arr.shape != (entry.size, entry.size):
                    raise ValueError(
                        f"Norm block for {entry.key!r} has shape {arr.shape!r}, "
                        f"expected {(entry.size, entry.size)!r}."
                    )
                normalized_n.append(arr)
        return cls(layout=layout, h_blocks=tuple(normalized_h), n_blocks=tuple(normalized_n))

    def apply(self, resid, theta):
        resid = np.asarray(resid, dtype=complex).reshape(-1)
        size = sum(entry.size for entry in self.layout)
        if resid.size != size:
            raise ValueError("Packed residual size must match the preconditioner layout.")

        out = np.zeros_like(resid)
        for entry, h_block, n_block in zip(self.layout, self.h_blocks, self.n_blocks):
            piece = resid[entry.offset:entry.offset + entry.size]
            if np.linalg.norm(piece) <= 1e-15:
                continue
            if h_block is None:
                out[entry.offset:entry.offset + entry.size] = piece
                continue
            metric = np.eye(entry.size, dtype=complex) if n_block is None else n_block
            system = theta * metric - h_block
            # Small regularization keeps nearly singular local blocks usable.
            system = 0.5 * (system + system.conj().T)
            reg = 1e-10 * np.eye(entry.size, dtype=complex)
            try:
                corrected = np.linalg.solve(system + reg, piece)
            except np.linalg.LinAlgError:
                corrected, *_ = np.linalg.lstsq(system + reg, piece, rcond=None)
            out[entry.offset:entry.offset + entry.size] = corrected
        return out


def _pack_tensor_state(tensor, *, layout=None):
    if not isinstance(tensor, NonabelianTensor):
        raise ValueError("_pack_tensor_state expects a NonabelianTensor.")

    if layout is None:
        entries = []
        offset = 0
        for key in sorted(tensor.data):
            block = np.asarray(tensor.data[key])
            size = int(block.size)
            entries.append(PackedEntry(tuple(key), tuple(block.shape), offset, size))
            offset += size
        layout = tuple(entries)

    if layout:
        present_dtypes = [
            np.asarray(tensor.data[entry.key]).dtype
            for entry in layout
            if entry.key in tensor.data
        ]
        dtype = np.result_type(*(present_dtypes or [float]))
    else:
        dtype = np.result_type(*(np.asarray(tensor.data[entry.key]).dtype for entry in layout))
    vec = np.zeros(sum(entry.size for entry in layout), dtype=dtype)
    for entry in layout:
        if entry.key not in tensor.data:
            continue
        block = np.asarray(tensor.data[entry.key])
        vec[entry.offset:entry.offset + entry.size] = block.reshape(entry.size)
    return vec, tuple(layout)


def _unpack_tensor_state(vector, template, *, layout):
    if not isinstance(template, NonabelianTensor):
        raise ValueError("_unpack_tensor_state expects a NonabelianTensor template.")

    vector = np.asarray(vector)
    data = {}
    for entry in layout:
        piece = vector[entry.offset:entry.offset + entry.size]
        data[entry.key] = piece.reshape(entry.shape)
    return NonabelianTensor(
        data,
        [leg[:] for leg in template.qns],
        template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=template.metadata.copy(),
    )


def _reduced_state_layout(layout):
    return ReducedStateLayout(tuple(layout))


def _tensor_to_reduced_state(tensor, *, state_layout):
    blocks = {}
    for entry in state_layout.entries:
        if entry.key not in tensor.data:
            continue
        block = np.asarray(tensor.data[entry.key])
        if np.linalg.norm(block.reshape(-1)) > 0.0:
            blocks[entry.key] = np.array(block, copy=True)
    return ReducedStateVector(layout=state_layout, blocks=blocks)


def _reduced_state_to_tensor(state, template):
    data = {}
    for entry in state.layout.entries:
        if entry.key in state.blocks:
            data[entry.key] = np.array(state.blocks[entry.key], copy=True).reshape(entry.shape)
        elif entry.key in template.data:
            data[entry.key] = np.zeros(entry.shape, dtype=np.asarray(template.data[entry.key]).dtype)
        else:
            data[entry.key] = np.zeros(entry.shape, dtype=float)
    return NonabelianTensor(
        data,
        [leg[:] for leg in template.qns],
        template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=template.metadata.copy(),
    )


def _reduced_add(a, b, *, alpha=1.0, beta=1.0):
    if a.layout != b.layout:
        raise ValueError("Reduced state vectors must share the same layout.")
    blocks = {}
    keys = set(a.blocks) | set(b.blocks)
    for key in keys:
        aval = np.asarray(a.blocks[key]) if key in a.blocks else 0.0
        bval = np.asarray(b.blocks[key]) if key in b.blocks else 0.0
        out = alpha * aval + beta * bval
        if np.linalg.norm(np.asarray(out).reshape(-1)) > 1.0e-15:
            blocks[key] = out
    return ReducedStateVector(layout=a.layout, blocks=blocks)


def _reduced_scale(a, scalar):
    if abs(scalar) <= 0.0:
        return ReducedStateVector(layout=a.layout, blocks={})
    return ReducedStateVector(
        layout=a.layout,
        blocks={
            key: scalar * np.asarray(block)
            for key, block in a.blocks.items()
            if np.linalg.norm(np.asarray(block).reshape(-1)) > 1.0e-15
        },
    )


def _reduced_dot(a, b):
    if a.layout != b.layout:
        raise ValueError("Reduced state vectors must share the same layout.")
    total = 0.0 + 0.0j
    for key in set(a.blocks) & set(b.blocks):
        total += np.vdot(np.asarray(a.blocks[key]).reshape(-1), np.asarray(b.blocks[key]).reshape(-1))
    return total


def _reduced_norm(a):
    return float(np.sqrt(max(0.0, np.real(_reduced_dot(a, a)))))


def _reduced_linear_combination(vectors, coeffs):
    if len(vectors) != len(coeffs):
        raise ValueError("Vector and coefficient counts must match.")
    if not vectors:
        raise ValueError("Need at least one reduced state vector.")
    layout = vectors[0].layout
    blocks = {}
    for vec, coeff in zip(vectors, coeffs):
        if vec.layout != layout:
            raise ValueError("Reduced state vectors must share the same layout.")
        if abs(coeff) <= 0.0:
            continue
        for key, block in vec.blocks.items():
            if key not in blocks:
                blocks[key] = coeff * np.asarray(block)
            else:
                blocks[key] = blocks[key] + coeff * np.asarray(block)
    blocks = {
        key: value
        for key, value in blocks.items()
        if np.linalg.norm(np.asarray(value).reshape(-1)) > 1.0e-15
    }
    return ReducedStateVector(layout=layout, blocks=blocks)


def _orthonormalize_reduced_vectors(vectors, *, tol=1e-12):
    basis = []
    for vec in vectors:
        work = vec
        for prev in basis:
            work = _reduced_add(work, prev, alpha=1.0, beta=-_reduced_dot(prev, work))
        norm = _reduced_norm(work)
        if norm > tol:
            basis.append(_reduced_scale(work, 1.0 / norm))
    return basis


def _reduced_operator_to_matvec(op, template, state_layout):
    def matvec(state):
        if op.packed_matvec is not None:
            out = np.asarray(op.packed_matvec(state.to_packed(dtype=complex)))
            return state_layout.from_packed(out)
        if op.reduced_matvec is not None:
            out = op.reduced_matvec(state)
            if not isinstance(out, ReducedStateVector):
                raise TypeError("reduced_matvec must return a ReducedStateVector.")
            if out.layout != state_layout:
                raise ValueError("reduced_matvec must preserve the reduced-state layout.")
            return out
        tensor = _reduced_state_to_tensor(state, template)
        out = op.tensor_matvec(tensor)
        if not isinstance(out, NonabelianTensor):
            raise TypeError("tensor_matvec must return a NonabelianTensor.")
        return _tensor_to_reduced_state(out, state_layout=state_layout)

    return matvec


@dataclass(frozen=True)
class LocalOperator:
    """
    Small wrapper for an effective two-site operator.

    Exactly one of ``matrix``, ``matvec``, ``tensor_matvec``, or
    ``reduced_matvec`` should be
    supplied.
    """

    matrix: object | None = None
    matvec: object | None = None
    tensor_matvec: object | None = None
    reduced_matvec: object | None = None
    packed_matvec: object | None = None
    aux_reduced_matvec: object | None = None
    aux_packed_matvec: object | None = None
    packed_block_matrices: object | None = None
    diag: object | None = None
    name: str | None = None
    identity_like: bool = False

    def __post_init__(self):
        count = sum(
            value is not None
            for value in (
                self.matrix,
                self.matvec,
                self.tensor_matvec,
                self.reduced_matvec,
                self.packed_matvec,
            )
        )
        if count != 1:
            raise ValueError(
                "LocalOperator requires exactly one of matrix, matvec, tensor_matvec, reduced_matvec, or packed_matvec."
            )


@dataclass(frozen=True)
class TwoSiteEffectiveH:
    """
    Effective two-site local problem, similar in spirit to TenPy's ``TwoSiteH``.

    Parameters
    ----------
    operator
        Effective local Hamiltonian.
    norm_operator
        Optional local norm operator. When omitted, the local problem is a
        standard Hermitian eigenproblem.
    canonical_norm
        Hint that the supplied norm operator is identity in the current local
        packed basis, so the standard eigenproblem path can be used directly.
    """

    operator: object
    norm_operator: object | None = None
    canonical_norm: bool = False
    name: str | None = None


def pack_two_site_state(two_site, *, layout=None):
    """
    Pack a rank-4 non-Abelian two-site tensor into a dense vector.
    """
    if not isinstance(two_site, NonabelianTensor) or two_site.rank != 4:
        raise ValueError("pack_two_site_state expects a rank-4 NonabelianTensor.")

    return _pack_tensor_state(two_site, layout=layout)


def unpack_two_site_state(vector, template, *, layout):
    """
    Rebuild a rank-4 non-Abelian tensor from a packed vector and template.
    """
    if not isinstance(template, NonabelianTensor) or template.rank != 4:
        raise ValueError("unpack_two_site_state expects a rank-4 NonabelianTensor template.")

    return _unpack_tensor_state(vector, template, layout=layout)


def _coupled_two_site_template(two_site):
    """
    Build a rank-3 coupled-basis template by fusing the two physical legs.
    """
    if not isinstance(two_site, NonabelianTensor) or two_site.rank != 4:
        raise ValueError("_coupled_two_site_template expects a rank-4 NonabelianTensor.")
    return _filter_coupled_template_to_boundary_target(
        combine_legs(two_site, (1, 2), new_axis=1, use_cg=True)
    )


def _fuses_to_boundary_target(left, middle, right):
    if isinstance(left, SpinChargeSector) and isinstance(middle, SpinChargeSector):
        return right in fuse_charge_spin_sectors(left, middle)
    if hasattr(left, "fuse"):
        try:
            return right in left.fuse(middle)
        except Exception:
            return True
    return True


def _filter_coupled_template_to_boundary_target(coupled):
    """
    Keep only coupled two-site blocks whose left x physical sector can reach the
    right boundary sector.
    """
    if not isinstance(coupled, NonabelianTensor) or coupled.rank != 3:
        return coupled
    data = {
        key: block
        for key, block in coupled.data.items()
        if _fuses_to_boundary_target(key[0], key[1], key[2])
    }
    if not data:
        raise ValueError("Coupled two-site template has no blocks compatible with the boundary target sector.")
    return NonabelianTensor(
        data,
        [leg[:] for leg in coupled.qns],
        coupled.dirs[:],
        fusion_legs=coupled.fusion_legs[:],
        metadata=coupled.metadata.copy(),
    )


def _couple_two_site_tensor(two_site):
    """
    Couple the two physical legs of a rank-4 local tensor into a rank-3 tensor.
    """
    return _coupled_two_site_template(two_site)


def _uncouple_two_site_tensor(coupled_two_site):
    """
    Undo :func:`_coupled_two_site_template` and restore ``(L, P_left, P_right, R)`` order.
    """
    if (
        not isinstance(coupled_two_site, NonabelianTensor)
        or coupled_two_site.rank != 3
        or coupled_two_site.fusion_legs[1] is None
        or coupled_two_site.fusion_legs[1].pipe is None
        or len(coupled_two_site.fusion_legs[1].child_sector_lists) != 2
    ):
        return split_legs(coupled_two_site, 1)

    fused_leg = coupled_two_site.fusion_legs[1]
    pipe = fused_leg.pipe
    coupling = pipe.coupling
    transform_cache = coupled_two_site.metadata.setdefault("_uncouple_local_basis_cache", {})
    data = {}

    def _local_transform(entry):
        key = (entry.child_sectors, entry.fused_sector, entry.slot)
        transform = transform_cache.get(key)
        if transform is not None:
            return transform
        if coupling in {"cg", "left", "right"}:
            bond_space = fused_leg.bond_space(
                entry.child_sectors,
                entry.fused_sector,
                scheme=coupling,
            )
            basis_by_slot = {
                channel.slot: basis
                for channel, basis in zip(bond_space.channels, bond_space.basis_matrices)
            }
            transform = basis_by_slot.get(entry.slot)
            if transform is None:
                raise ValueError(
                    f"Missing reduced basis transform for slot {entry.slot} and child sectors {entry.child_sectors!r}."
                )
        else:
            local_dim = int(np.prod(entry.selected_shape, dtype=int))
            transform = np.eye(local_dim, dtype=float)
        transform_cache[key] = transform
        return transform

    for (q_left, q_fused, q_right), block in coupled_two_site.data.items():
        arr = np.asarray(block)
        for entry in sorted(pipe.entries_for_sector(q_fused), key=lambda item: item.slot):
            sl = slice(entry.offset, entry.offset + entry.local_dim)
            local_block = arr[:, sl, :]
            transform = _local_transform(entry)
            expanded = np.tensordot(local_block, transform, axes=(1, 1))
            piece = np.transpose(expanded, (0, 2, 1)).reshape(
                arr.shape[0],
                *entry.selected_shape,
                arr.shape[2],
            )
            key_out = (q_left, entry.child_sectors[0], entry.child_sectors[1], q_right)
            if key_out in data:
                data[key_out] = data[key_out] + piece
            else:
                data[key_out] = piece

    return NonabelianTensor(
        data,
        [
            list(coupled_two_site.qns[0]),
            list(fused_leg.child_sector_lists[0]),
            list(fused_leg.child_sector_lists[1]),
            list(coupled_two_site.qns[2]),
        ],
        [
            coupled_two_site.dirs[0],
            fused_leg.child_dirs[0],
            fused_leg.child_dirs[1],
            coupled_two_site.dirs[2],
        ],
        fusion_legs=[
            coupled_two_site.fusion_legs[0],
            None,
            None,
            coupled_two_site.fusion_legs[2],
        ],
        metadata=coupled_two_site.metadata.copy(),
    )


def _entry_uncouple_matrix(left_dim, right_dim, selected_shape, transform):
    local_dim = int(np.asarray(transform).shape[1])
    eye = np.eye(left_dim * local_dim * right_dim, dtype=np.asarray(transform).dtype).reshape(
        left_dim,
        local_dim,
        right_dim,
        left_dim * local_dim * right_dim,
    )
    expanded = np.tensordot(eye, np.asarray(transform), axes=(1, 1))
    piece = np.transpose(expanded, (0, 3, 1, 2)).reshape(
        left_dim,
        *selected_shape,
        right_dim,
        left_dim * local_dim * right_dim,
    )
    return piece.reshape(
        left_dim * int(np.prod(selected_shape, dtype=int)) * right_dim,
        left_dim * local_dim * right_dim,
    )


def _apply_basis_transform_blocks(vector, blocks, out_size, *, adjoint=False):
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    out = np.zeros(out_size, dtype=vector.dtype)
    for row_slice, in_indices, submat in blocks:
        if adjoint:
            out[in_indices] += np.asarray(submat).conj().T @ vector[row_slice]
        else:
            out[row_slice] += np.asarray(submat) @ vector[in_indices]
    return out


def _build_basis_transform_direct(two_site, coupled, coupled_layout, orig_layout):
    if (
        not isinstance(coupled, NonabelianTensor)
        or coupled.rank != 3
        or coupled.fusion_legs[1] is None
        or coupled.fusion_legs[1].pipe is None
        or len(coupled.fusion_legs[1].child_sector_lists) != 2
    ):
        return None

    fused_leg = coupled.fusion_legs[1]
    pipe = fused_leg.pipe
    coupling = pipe.coupling
    out_entry_map = {entry.key: entry for entry in orig_layout}
    dtype = complex
    transform = np.zeros(
        (sum(entry.size for entry in orig_layout), sum(entry.size for entry in coupled_layout)),
        dtype=dtype,
    )

    transform_cache = two_site.metadata.setdefault("_direct_basis_transform_cache", {})
    transform_blocks = []

    for c_entry in coupled_layout:
        q_left, q_fused, q_right = c_entry.key
        left_dim, fused_dim_total, right_dim = c_entry.shape
        entries = sorted(pipe.entries_for_sector(q_fused), key=lambda item: item.slot)
        for entry in entries:
            cache_key = (entry.child_sectors, entry.fused_sector, entry.slot)
            local_transform = transform_cache.get(cache_key)
            if local_transform is None:
                if coupling in {"cg", "left", "right"}:
                    bond_space = fused_leg.bond_space(
                        entry.child_sectors,
                        entry.fused_sector,
                        scheme=coupling,
                    )
                    basis_by_slot = {
                        channel.slot: basis
                        for channel, basis in zip(bond_space.channels, bond_space.basis_matrices)
                    }
                    local_transform = basis_by_slot.get(entry.slot)
                    if local_transform is None:
                        return None
                else:
                    local_transform = np.eye(int(np.prod(entry.selected_shape, dtype=int)), dtype=float)
                transform_cache[cache_key] = local_transform

            key_out = (q_left, entry.child_sectors[0], entry.child_sectors[1], q_right)
            out_entry = out_entry_map.get(key_out)
            if out_entry is None:
                continue

            submat = _entry_uncouple_matrix(
                left_dim,
                right_dim,
                entry.selected_shape,
                local_transform,
            )
            in_indices = []
            for l in range(left_dim):
                for f_local in range(entry.local_dim):
                    for r in range(right_dim):
                        idx = ((l * fused_dim_total) + (entry.offset + f_local)) * right_dim + r
                        in_indices.append(c_entry.offset + idx)
            row_slice = slice(out_entry.offset, out_entry.offset + out_entry.size)
            transform[row_slice, in_indices] += submat
            transform_blocks.append((row_slice, np.asarray(in_indices, dtype=int), submat))

    two_site.metadata["_basis_transform_struct_cache"] = {
        "coupled_layout": tuple(coupled_layout),
        "uncoupled_layout": tuple(orig_layout),
        "blocks": tuple(transform_blocks),
        "coupled_size": sum(entry.size for entry in coupled_layout),
        "uncoupled_size": sum(entry.size for entry in orig_layout),
    }

    return transform


def _build_basis_transform(two_site, *, coupled=None, coupled_layout=None, uncoupled_layout=None):
    """
    Return the dense basis transform from coupled rank-3 to uncoupled rank-4 packing.
    """
    cache = two_site.metadata.get("_basis_transform_cache")
    if cache is not None:
        cached_uncoupled_layout = cache.get("uncoupled_layout")
        cached_coupled_layout = cache.get("coupled_layout")
        if (
            cached_uncoupled_layout == tuple(uncoupled_layout) if uncoupled_layout is not None else True
        ) and (
            cached_coupled_layout == tuple(coupled_layout) if coupled_layout is not None else True
        ):
            return cache["coupled"], cache["coupled_layout"], cache["transform"]

    if coupled is None:
        coupled = _coupled_two_site_template(two_site)
    if uncoupled_layout is None:
        orig_vec, orig_layout = pack_two_site_state(two_site)
        _ = orig_vec
    else:
        orig_layout = tuple(uncoupled_layout)
    if coupled_layout is None:
        coupled_vec, coupled_layout = _pack_tensor_state(coupled)
    else:
        coupled_vec, coupled_layout = _pack_tensor_state(coupled, layout=coupled_layout)

    direct = _build_basis_transform_direct(two_site, coupled, coupled_layout, orig_layout)
    if direct is not None:
        two_site.metadata["_basis_transform_cache"] = {
            "coupled": coupled,
            "coupled_layout": tuple(coupled_layout),
            "uncoupled_layout": tuple(orig_layout),
            "transform": direct,
        }
        return coupled, coupled_layout, direct

    transform = np.zeros((sum(entry.size for entry in orig_layout), coupled_vec.size), dtype=complex)
    for col in range(coupled_vec.size):
        basis = np.zeros(coupled_vec.size, dtype=complex)
        basis[col] = 1.0
        coupled_tensor = _unpack_tensor_state(basis, coupled, layout=coupled_layout)
        uncoupled_tensor = _uncouple_two_site_tensor(coupled_tensor)
        packed, _ = pack_two_site_state(uncoupled_tensor, layout=orig_layout)
        transform[:, col] = packed
    two_site.metadata["_basis_transform_cache"] = {
        "coupled": coupled,
        "coupled_layout": tuple(coupled_layout),
        "uncoupled_layout": tuple(orig_layout),
        "transform": transform,
    }
    return coupled, coupled_layout, transform


def _transform_has_orthonormal_columns(transform, *, tol=1e-10):
    transform = np.asarray(transform)
    gram = transform.conj().T @ transform
    eye = np.eye(gram.shape[0], dtype=gram.dtype)
    return np.allclose(gram, eye, atol=tol, rtol=tol)


def _normalize_local_operator(local_operator):
    if isinstance(local_operator, TwoSiteEffectiveH):
        local_operator = local_operator.operator
    if isinstance(local_operator, LocalOperator):
        return local_operator
    if isinstance(local_operator, dict):
        return LocalOperator(
            matrix=local_operator.get("matrix"),
            matvec=local_operator.get("matvec"),
            tensor_matvec=local_operator.get("tensor_matvec"),
            reduced_matvec=local_operator.get("reduced_matvec"),
            packed_matvec=local_operator.get("packed_matvec"),
            diag=local_operator.get("diag"),
            name=local_operator.get("name"),
        )
    if callable(local_operator):
        return LocalOperator(matvec=local_operator)
    return LocalOperator(matrix=np.asarray(local_operator))


def _resolve_davidson_operator(local_operator, template, layout):
    op = _normalize_local_operator(local_operator)

    if op.matrix is not None:
        matrix = np.asarray(op.matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("LocalOperator matrix must be square.")
        diag = np.asarray(np.diag(matrix), dtype=float)
        return matrix, diag

    if op.tensor_matvec is not None:
        def matvec(vec):
            tensor = unpack_two_site_state(vec, template, layout=layout)
            out = op.tensor_matvec(tensor)
            if not isinstance(out, NonabelianTensor):
                raise TypeError("tensor_matvec must return a NonabelianTensor.")
            packed, _ = pack_two_site_state(out, layout=layout)
            return packed
        diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
        return matvec, diag

    if op.reduced_matvec is not None:
        state_layout = _reduced_state_layout(layout)

        def matvec(vec):
            state = state_layout.from_packed(vec)
            out = op.reduced_matvec(state)
            if not isinstance(out, ReducedStateVector):
                raise TypeError("reduced_matvec must return a ReducedStateVector.")
            if out.layout != state_layout:
                raise ValueError("reduced_matvec must preserve the reduced-state layout.")
            return out.to_packed(dtype=complex)

        diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
        return matvec, diag

    if op.packed_matvec is not None:
        diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
        return op.packed_matvec, diag

    diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
    return op.matvec, diag


def _materialize_local_matrix(operator, dim):
    """
    Build a dense matrix from a linear-operator callback for small local problems.
    """
    matrix = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        matrix[:, col] = np.asarray(operator(basis))
    return 0.5 * (matrix + matrix.conj().T)


def _orthonormalize_columns_dense(V, tol=1e-12):
    V = np.asarray(V)
    if V.size == 0:
        return np.zeros((V.shape[0], 0), dtype=V.dtype)
    cols = []
    for idx in range(V.shape[1]):
        vec = np.array(V[:, idx], dtype=V.dtype, copy=True)
        for prev in cols:
            vec -= prev * np.vdot(prev, vec)
        norm = np.linalg.norm(vec)
        if norm > tol:
            cols.append(vec / norm)
    if not cols:
        return np.zeros((V.shape[0], 0), dtype=V.dtype)
    return np.column_stack(cols)


def _build_iterative_guess(diag_h, neigen, *, guess=None, diag_n=None):
    diag_h = np.asarray(diag_h, dtype=float)
    n = diag_h.size
    if diag_n is None:
        scores = diag_h
    else:
        diag_n = np.asarray(diag_n, dtype=float)
        safe_n = np.where(np.abs(diag_n) > 1e-12, diag_n, 1.0)
        scores = diag_h / safe_n

    cols = []
    if guess is not None:
        guess_arr = np.asarray(guess)
        if guess_arr.ndim == 1:
            cols.append(guess_arr.reshape(n).astype(complex))
        else:
            cols.extend(guess_arr[:, i].reshape(n).astype(complex) for i in range(guess_arr.shape[1]))
    for idx in np.argsort(scores):
        e = np.zeros(n, dtype=complex)
        e[idx] = 1.0
        cols.append(e)
        if len(cols) >= max(2 * neigen, neigen + 1):
            break
    return _orthonormalize_columns_dense(np.column_stack(cols))


def _build_iterative_guess_reduced(diag_h, state_layout, *, neigen=1, guess=None, diag_n=None):
    diag_h = np.asarray(diag_h, dtype=float)
    n = diag_h.size
    if diag_n is None:
        scores = diag_h
    else:
        diag_n = np.asarray(diag_n, dtype=float)
        safe_n = np.where(np.abs(diag_n) > 1e-12, diag_n, 1.0)
        scores = diag_h / safe_n

    vectors = []
    if guess is not None:
        if isinstance(guess, ReducedStateVector):
            vectors.append(guess)
        else:
            guess_arr = np.asarray(guess)
            if guess_arr.ndim == 1:
                vectors.append(state_layout.from_packed(guess_arr.reshape(n).astype(complex)))
            else:
                for i in range(guess_arr.shape[1]):
                    vectors.append(state_layout.from_packed(guess_arr[:, i].reshape(n).astype(complex)))
    for idx in np.argsort(scores):
        vectors.append(state_layout.basis_vector(idx, dtype=complex))
        if len(vectors) >= max(2 * neigen, neigen + 1):
            break
    return _orthonormalize_reduced_vectors(vectors)


def _prepare_reduced_guess(template, state_layout, guess):
    guess_state = _tensor_to_reduced_state(template, state_layout=state_layout)
    if guess is not None:
        if isinstance(guess, NonabelianTensor):
            guess_state = _tensor_to_reduced_state(guess, state_layout=state_layout)
        elif isinstance(guess, ReducedStateVector):
            guess_state = guess
        else:
            guess_state = state_layout.from_packed(np.asarray(guess))
    if _reduced_norm(guess_state) < 1e-15:
        raise ValueError("Initial guess for tensor Davidson must have nonzero norm.")
    return guess_state


def _packed_matrix_from_reduced_vectors(vectors, state_layout):
    if not vectors:
        return np.zeros((state_layout.size, 0), dtype=complex)
    return np.column_stack([vec.to_packed(dtype=complex) for vec in vectors])


def _solve_reduced_generalized_davidson(
    guess_state,
    H,
    *,
    state_layout,
    h_diag,
    N=None,
    n_diag=None,
    tol=1e-8,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    precond=None,
):
    if not isinstance(guess_state, ReducedStateVector):
        raise TypeError("_solve_reduced_generalized_davidson expects a ReducedStateVector guess.")
    if _reduced_norm(guess_state) < 1e-15:
        raise ValueError("Initial guess for Davidson must have nonzero norm.")

    h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
    if h_diag.size != state_layout.size:
        raise ValueError("Hamiltonian diagonal guess must match the packed state dimension.")

    has_norm_operator = N is not None
    if N is None:
        N = lambda vec: vec
    if n_diag is None:
        n_diag = np.ones(state_layout.size, dtype=float)
    else:
        n_diag = np.asarray(n_diag, dtype=float).reshape(-1)
        if n_diag.size != state_layout.size:
            raise ValueError("Norm diagonal guess must match the packed state dimension.")

    if max_space is None:
        max_space = min(state_layout.size, 24)
    tol_res = np.sqrt(tol) if tol_residual is None else tol_residual

    V = _build_iterative_guess_reduced(h_diag, state_layout, neigen=1, guess=guess_state, diag_n=n_diag)
    AV = [H(vec) for vec in V]
    BV = [N(vec) for vec in V]
    reduced_preconditioner = None
    preconditioner_mode = None

    if precond is None:
        reduced_preconditioner = ReducedDiagonalPreconditioner.from_packed_diagonals(
            state_layout,
            h_diag,
            n_diag=n_diag,
        )
        precondition = lambda resid, theta, vec: reduced_preconditioner.apply(resid, theta)
        preconditioner_mode = "reduced_diagonal"
    elif callable(precond):
        def callback_precond(resid, theta, vec):
            out = precond(
                resid.to_packed(dtype=complex),
                theta,
                vec.to_packed(dtype=complex),
            )
            if isinstance(out, ReducedStateVector):
                return out
            return state_layout.from_packed(np.asarray(out))

        precondition = callback_precond
        preconditioner_mode = "callback"
    else:
        precond_arr = np.asarray(precond, dtype=float)
        reduced_preconditioner = ReducedDiagonalPreconditioner.from_packed_diagonals(
            state_layout,
            precond_arr,
            n_diag=np.ones(state_layout.size, dtype=float),
        )
        precondition = lambda resid, theta, vec: reduced_preconditioner.apply(resid, theta)
        preconditioner_mode = "reduced_diagonal"

    prev_theta = None
    converged = False
    residual_norm = None
    iterations = 0
    restarts = 0

    for iterations in range(1, itermax + 1):
        Vp = _packed_matrix_from_reduced_vectors(V, state_layout)
        AVp = _packed_matrix_from_reduced_vectors(AV, state_layout)
        BVp = _packed_matrix_from_reduced_vectors(BV, state_layout)
        Hs = Vp.conj().T @ AVp
        Ns = Vp.conj().T @ BVp
        theta, coeff, _ = _solve_generalized_dense(Hs, Ns, tol=max(tol, 1e-12))
        ritz_p = Vp @ coeff
        aritz_p = AVp @ coeff
        britz_p = BVp @ coeff
        resid_p = aritz_p - theta * britz_p
        residual_norm = float(np.linalg.norm(resid_p))
        de = np.inf if prev_theta is None else abs(theta - prev_theta)
        if residual_norm <= tol_res and (prev_theta is None or de <= tol):
            converged = True
            break

        ritz = state_layout.from_packed(ritz_p)
        resid = state_layout.from_packed(resid_p)
        corr = precondition(resid, theta, ritz)
        corr_p = corr.to_packed(dtype=complex)
        if Vp.shape[1]:
            corr_p = corr_p - Vp @ (Vp.conj().T @ corr_p)
        corr_norm = float(np.linalg.norm(corr_p))
        if corr_norm <= lindep:
            break
        corr = state_layout.from_packed(corr_p / corr_norm)

        if len(V) + 1 > max_space:
            V = _orthonormalize_reduced_vectors([ritz, corr], tol=lindep)
            restarts += 1
            AV = [H(vec) for vec in V]
            BV = [N(vec) for vec in V]
        else:
            V = V + [corr]
            AV = AV + [H(corr)]
            BV = BV + [N(corr)]
        prev_theta = theta

    Vp = _packed_matrix_from_reduced_vectors(V, state_layout)
    AVp = _packed_matrix_from_reduced_vectors(AV, state_layout)
    BVp = _packed_matrix_from_reduced_vectors(BV, state_layout)
    Hs = Vp.conj().T @ AVp
    Ns = Vp.conj().T @ BVp
    theta, coeff, _ = _solve_generalized_dense(Hs, Ns, tol=max(tol, 1e-12))
    vec_packed = _canonicalize_eigenvector(Vp @ coeff, reference=guess_state.to_packed(dtype=complex))
    vec = state_layout.from_packed(vec_packed)
    residual_norm = float(np.linalg.norm((AVp @ coeff) - theta * (BVp @ coeff)))
    return float(theta), vec, {
        "metric": residual_norm,
        "residual": residual_norm,
        "davidson_iterations": int(iterations),
        "davidson_converged": bool(converged),
        "subspace_dim": int(len(V)),
        "generalized_norm": has_norm_operator,
        "tensor_davidson": True,
        "reduced_krylov": True,
        "preconditioner_mode": preconditioner_mode,
        "reduced_preconditioner": reduced_preconditioner is not None,
        "restarts": int(restarts),
    }


def _solve_packed_generalized_davidson(
    guess_packed,
    H,
    *,
    h_diag,
    N=None,
    n_diag=None,
    tol=1e-8,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    precond=None,
    block_preconditioner=None,
):
    guess_packed = np.asarray(guess_packed, dtype=complex).reshape(-1)
    if np.linalg.norm(guess_packed) < 1e-15:
        raise ValueError("Initial guess for Davidson must have nonzero norm.")

    h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
    if h_diag.size != guess_packed.size:
        raise ValueError("Hamiltonian diagonal guess must match the packed state dimension.")

    has_norm_operator = N is not None
    if N is None:
        N = lambda vec: vec
    if n_diag is None:
        n_diag = np.ones_like(h_diag)
    else:
        n_diag = np.asarray(n_diag, dtype=float).reshape(-1)
        if n_diag.size != guess_packed.size:
            raise ValueError("Norm diagonal guess must match the packed state dimension.")

    if max_space is None:
        max_space = min(guess_packed.size, 48)
    tol_res = np.sqrt(tol) if tol_residual is None else tol_residual

    Vp = _build_iterative_guess(h_diag, 1, guess=guess_packed, diag_n=n_diag)
    AVp = np.column_stack([np.asarray(H(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
    BVp = np.column_stack([np.asarray(N(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
    preconditioner_mode = None

    if precond is None and block_preconditioner is not None:
        def precondition(resid, theta, vec):
            corrected = block_preconditioner.apply(resid, theta)
            if np.linalg.norm(corrected) <= 1e-15:
                denom = theta * n_diag - h_diag
                safe = np.where(
                    np.abs(denom) > 1e-12,
                    denom,
                    np.where(denom >= 0, 1e-12, -1e-12),
                )
                return resid / safe
            return corrected

        preconditioner_mode = "packed_block"
    elif precond is None:
        def precondition(resid, theta, vec):
            denom = theta * n_diag - h_diag
            safe = np.where(
                np.abs(denom) > 1e-12,
                denom,
                np.where(denom >= 0, 1e-12, -1e-12),
            )
            return resid / safe

        preconditioner_mode = "packed_diagonal"
    elif callable(precond):
        def precondition(resid, theta, vec):
            return np.asarray(precond(resid, theta, vec), dtype=complex).reshape(-1)

        preconditioner_mode = "callback"
    else:
        precond_arr = np.asarray(precond, dtype=float).reshape(-1)
        if precond_arr.size != guess_packed.size:
            raise ValueError("Packed preconditioner diagonal must match the packed state dimension.")

        def precondition(resid, theta, vec):
            denom = theta - precond_arr
            safe = np.where(
                np.abs(denom) > 1e-12,
                denom,
                np.where(denom >= 0, 1e-12, -1e-12),
            )
            return resid / safe

        preconditioner_mode = "packed_diagonal"

    def _expand_projected_matrix(mat, overlap_col, diag_val):
        if mat.size == 0:
            return np.asarray([[diag_val]], dtype=complex)
        overlap_col = np.asarray(overlap_col, dtype=complex).reshape(-1, 1)
        top = np.hstack([mat, overlap_col])
        bottom = np.hstack([overlap_col.conj().T, np.asarray([[diag_val]], dtype=complex)])
        return np.vstack([top, bottom])

    prev_theta = None
    converged = False
    residual_norm = None
    iterations = 0
    restarts = 0
    Hs = Vp.conj().T @ AVp
    Ns = Vp.conj().T @ BVp

    def _generalized_restart_vectors(Hs_local, Ns_local, keep, *, tol_local):
        Hs_local = 0.5 * (np.asarray(Hs_local) + np.asarray(Hs_local).conj().T)
        Ns_local = 0.5 * (np.asarray(Ns_local) + np.asarray(Ns_local).conj().T)
        s, U = np.linalg.eigh(Ns_local)
        mask = s > max(float(tol_local), 1e-12)
        if not np.any(mask):
            return None
        X = U[:, mask] @ np.diag(1.0 / np.sqrt(s[mask]))
        H_ortho = 0.5 * (X.conj().T @ Hs_local @ X + (X.conj().T @ Hs_local @ X).conj().T)
        evals, evecs = np.linalg.eigh(H_ortho)
        keep = min(int(keep), evecs.shape[1])
        coeffs = X @ evecs[:, :keep]
        for col in range(coeffs.shape[1]):
            norm = np.sqrt(np.real(np.vdot(coeffs[:, col], Ns_local @ coeffs[:, col])))
            if norm > 1e-15:
                coeffs[:, col] = coeffs[:, col] / norm
        return coeffs

    for iterations in range(1, itermax + 1):
        theta, coeff, _ = _solve_generalized_dense(Hs, Ns, tol=max(tol, 1e-12))
        ritz_p = Vp @ coeff
        aritz_p = AVp @ coeff
        britz_p = BVp @ coeff
        resid_p = aritz_p - theta * britz_p
        residual_norm = float(np.linalg.norm(resid_p))
        de = np.inf if prev_theta is None else abs(theta - prev_theta)
        if residual_norm <= tol_res and (prev_theta is None or de <= tol):
            converged = True
            break

        corr_p = precondition(resid_p, theta, ritz_p)
        if Vp.shape[1]:
            corr_p = corr_p - Vp @ (Vp.conj().T @ corr_p)
        corr_norm = float(np.linalg.norm(corr_p))
        if corr_norm <= lindep:
            break
        corr_p = corr_p / corr_norm

        if Vp.shape[1] + 1 > max_space:
            restart_keep = min(max(2, max_space // 32), 4)
            restart_coeffs = _generalized_restart_vectors(Hs, Ns, restart_keep, tol_local=max(tol, 1e-12))
            restart_vectors = []
            if restart_coeffs is not None:
                restart_vectors.extend(Vp @ restart_coeffs[:, i] for i in range(restart_coeffs.shape[1]))
            else:
                restart_vectors.append(ritz_p)
            restart_vectors.append(corr_p)
            Vp = _orthonormalize_columns_dense(np.column_stack(restart_vectors), tol=lindep)
            restarts += 1
            AVp = np.column_stack([np.asarray(H(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
            BVp = np.column_stack([np.asarray(N(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
            Hs = Vp.conj().T @ AVp
            Ns = Vp.conj().T @ BVp
        else:
            h_corr = np.asarray(H(corr_p), dtype=complex).reshape(-1)
            n_corr = np.asarray(N(corr_p), dtype=complex).reshape(-1)
            h_overlap = Vp.conj().T @ h_corr
            n_overlap = Vp.conj().T @ n_corr
            Vp = np.column_stack([Vp, corr_p])
            AVp = np.column_stack([AVp, h_corr])
            BVp = np.column_stack([BVp, n_corr])
            Hs = _expand_projected_matrix(Hs, h_overlap, np.vdot(corr_p, h_corr))
            Ns = _expand_projected_matrix(Ns, n_overlap, np.vdot(corr_p, n_corr))
        prev_theta = theta

    theta, coeff, _ = _solve_generalized_dense(Hs, Ns, tol=max(tol, 1e-12))
    vec_packed = _canonicalize_eigenvector(Vp @ coeff, reference=guess_packed)
    residual_norm = float(np.linalg.norm((AVp @ coeff) - theta * (BVp @ coeff)))
    return float(theta), vec_packed, {
        "metric": residual_norm,
        "residual": residual_norm,
        "davidson_iterations": int(iterations),
        "davidson_converged": bool(converged),
        "subspace_dim": int(Vp.shape[1]),
        "generalized_norm": has_norm_operator,
        "tensor_davidson": True,
        "reduced_krylov": False,
        "packed_krylov": True,
        "preconditioner_mode": preconditioner_mode,
        "reduced_preconditioner": False,
        "restarts": int(restarts),
    }


def _canonicalize_eigenvector(vec, *, reference=None, tol=1e-12):
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    if vec.size == 0:
        return vec

    phase_ref = None
    if reference is not None:
        ref = np.asarray(reference, dtype=complex).reshape(-1)
        if ref.shape == vec.shape:
            overlap = np.vdot(ref, vec)
            if abs(overlap) > tol:
                phase_ref = overlap / abs(overlap)

    if phase_ref is None:
        pivot = int(np.argmax(np.abs(vec)))
        if abs(vec[pivot]) > tol:
            phase_ref = vec[pivot] / abs(vec[pivot])

    if phase_ref is not None:
        vec = vec / phase_ref

    if np.max(np.abs(vec.imag)) <= tol * max(1.0, np.max(np.abs(vec.real))):
        vec = vec.real.astype(float)
    return vec


def _solve_tensor_davidson(
    template,
    layout,
    operator,
    *,
    norm_operator=None,
    guess=None,
    tol=1e-8,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    precond=None,
):
    op = _normalize_local_operator(operator)
    if (
        op.matrix is None
        and op.tensor_matvec is None
        and op.reduced_matvec is None
        and op.packed_matvec is None
    ):
        raise TypeError("_solve_tensor_davidson requires a matrix, tensor_matvec, reduced_matvec, or packed_matvec operator.")
    norm_op = None if norm_operator is None else _normalize_local_operator(norm_operator)
    if (
        norm_op is not None
        and norm_op.matrix is None
        and norm_op.tensor_matvec is None
        and norm_op.reduced_matvec is None
        and norm_op.packed_matvec is None
    ):
        raise TypeError("_solve_tensor_davidson requires matrix, tensor_matvec, reduced_matvec, or packed_matvec norm operators.")

    state_layout = _reduced_state_layout(layout)
    guess_state = _prepare_reduced_guess(template, state_layout, guess)
    guess_packed = guess_state.to_packed(dtype=complex)

    h_diag = (
        np.asarray(op.diag, dtype=float)
        if op.diag is not None
        else np.zeros(state_layout.size, dtype=float)
    )
    n_diag = (
        np.asarray(norm_op.diag, dtype=float)
        if norm_op is not None and norm_op.diag is not None
        else np.ones(state_layout.size, dtype=float)
    )

    if op.matrix is not None and (norm_op is None or norm_op.matrix is not None):
        matrix = np.asarray(op.matrix)
        if norm_op is not None:
            norm_matrix = np.asarray(norm_op.matrix)
            theta, vec_packed, residual = _solve_generalized_dense(matrix, norm_matrix, tol=tol)
            vec_packed = _canonicalize_eigenvector(vec_packed, reference=guess_packed)
            objective = {
                "energy": float(theta),
                "metric": residual,
                "residual": residual,
                "davidson_iterations": int(itermax),
                "davidson_converged": False,
                "subspace_dim": int(matrix.shape[0]),
                "generalized_norm": True,
                "tensor_davidson": True,
                "reduced_krylov": False,
                "packed_krylov": False,
                "preconditioner_mode": None,
                "reduced_preconditioner": False,
                "restarts": 0,
            }
        else:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
            vec_packed = _canonicalize_eigenvector(eigvecs[:, 0], reference=guess_packed)
            objective = {
                "energy": float(np.real(eigvals[0])),
                "metric": 0.0,
                "residual": 0.0,
                "davidson_iterations": int(itermax),
                "davidson_converged": False,
                "subspace_dim": int(matrix.shape[0]),
                "generalized_norm": False,
                "tensor_davidson": True,
                "reduced_krylov": False,
                "packed_krylov": False,
                "preconditioner_mode": None,
                "reduced_preconditioner": False,
                "restarts": 0,
            }
        optimized = _unpack_tensor_state(vec_packed, template, layout=layout)
        objective["operator_representation"] = "dense"
        if norm_op is not None:
            objective["norm_operator_representation"] = "dense"
        return optimized, objective

    if op.packed_matvec is not None and (norm_op is None or norm_op.packed_matvec is not None):
        H_packed = op.packed_matvec
        N_packed = norm_op.packed_matvec if norm_op is not None else None
        block_preconditioner = None
        if precond is None and op.packed_block_matrices is not None:
            block_preconditioner = PackedBlockPreconditioner.from_layout_blocks(
                layout,
                op.packed_block_matrices,
                n_blocks=(
                    norm_op.packed_block_matrices
                    if norm_op is not None and norm_op.packed_block_matrices is not None
                    else None
                ),
            )
        theta, vec_packed, objective = _solve_packed_generalized_davidson(
            guess_packed,
            H_packed,
            h_diag=h_diag,
            N=N_packed,
            n_diag=n_diag,
            tol=tol,
            itermax=itermax,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            precond=precond,
            block_preconditioner=block_preconditioner,
        )
        optimized = _unpack_tensor_state(vec_packed, template, layout=layout)
        objective["energy"] = float(theta)
        objective["operator_representation"] = "reduced"
        if norm_op is not None:
            objective["norm_operator_representation"] = "reduced"
        return optimized, objective

    H = _reduced_operator_to_matvec(op, template, state_layout)
    N = (
        _reduced_operator_to_matvec(norm_op, template, state_layout)
        if norm_op is not None
        else None
    )
    theta, vec, objective = _solve_reduced_generalized_davidson(
        guess_state,
        H,
        state_layout=state_layout,
        h_diag=h_diag,
        N=N,
        n_diag=n_diag,
        tol=tol,
        itermax=itermax,
        max_space=max_space,
        tol_residual=tol_residual,
        lindep=lindep,
        precond=precond,
    )
    optimized = _reduced_state_to_tensor(vec, template)
    objective["energy"] = float(theta)
    objective["operator_representation"] = (
        "reduced" if (op.reduced_matvec is not None or op.packed_matvec is not None) else "tensor"
    )
    if norm_op is not None:
        objective["norm_operator_representation"] = (
            "reduced" if (norm_op.reduced_matvec is not None or norm_op.packed_matvec is not None) else "tensor"
        )
    return optimized, objective


def _coupled_operator_tensor_matvec(op, two_site, uncoupled_layout):
    uncoupled_state_layout = _reduced_state_layout(uncoupled_layout)

    def apply(coupled_tensor):
        uncoupled_tensor = _uncouple_two_site_tensor(coupled_tensor)
        if op.tensor_matvec is not None:
            out_uncoupled = op.tensor_matvec(uncoupled_tensor)
            if not isinstance(out_uncoupled, NonabelianTensor):
                raise TypeError("tensor_matvec must return a NonabelianTensor.")
        elif op.reduced_matvec is not None:
            uncoupled_state = _tensor_to_reduced_state(uncoupled_tensor, state_layout=uncoupled_state_layout)
            out_state = op.reduced_matvec(uncoupled_state)
            if not isinstance(out_state, ReducedStateVector):
                raise TypeError("reduced_matvec must return a ReducedStateVector.")
            if out_state.layout != uncoupled_state_layout:
                raise ValueError("reduced_matvec must preserve the reduced-state layout.")
            out_uncoupled = _reduced_state_to_tensor(out_state, two_site)
        else:
            uncoupled_vec, _ = pack_two_site_state(uncoupled_tensor, layout=uncoupled_layout)
            if op.matrix is not None:
                out_vec = np.asarray(op.matrix) @ uncoupled_vec
            else:
                out_vec = np.asarray(op.matvec(uncoupled_vec))
            out_uncoupled = unpack_two_site_state(out_vec, two_site, layout=uncoupled_layout)
        return _couple_two_site_tensor(out_uncoupled)

    return apply


def _lift_operator_to_coupled(
    op,
    two_site,
    uncoupled_layout,
    coupled_template,
    coupled_layout,
    *,
    transform=None,
    transform_dag=None,
):
    coupled_state_layout = _reduced_state_layout(coupled_layout)
    uncoupled_state_layout = _reduced_state_layout(uncoupled_layout)
    if transform is None:
        _, _, transform = _build_basis_transform(
            two_site,
            coupled=coupled_template,
            coupled_layout=coupled_layout,
            uncoupled_layout=uncoupled_layout,
        )
        transform = np.asarray(transform)
    if transform_dag is None:
        transform_dag = np.asarray(transform).conj().T
    struct_cache = two_site.metadata.get("_basis_transform_struct_cache")
    use_struct = (
        struct_cache is not None
        and struct_cache.get("coupled_layout") == tuple(coupled_layout)
        and struct_cache.get("uncoupled_layout") == tuple(uncoupled_layout)
    )
    transform_blocks = struct_cache.get("blocks") if use_struct else None
    use_dense_transform = np.asarray(transform).size <= _BASIS_TRANSFORM_DENSE_MATVEC_SIZE
    if use_dense_transform:
        dense_transform = np.asarray(transform)
        dense_transform_dag = np.asarray(transform_dag)

        def _to_uncoupled_packed(coupled_packed):
            return dense_transform @ coupled_packed

        def _from_uncoupled_packed(uncoupled_packed):
            return dense_transform_dag @ uncoupled_packed

    elif transform_blocks is not None:
        def _to_uncoupled_packed(coupled_packed):
            return _apply_basis_transform_blocks(
                coupled_packed,
                transform_blocks,
                struct_cache["uncoupled_size"],
                adjoint=False,
            )

        def _from_uncoupled_packed(uncoupled_packed):
            return _apply_basis_transform_blocks(
                uncoupled_packed,
                transform_blocks,
                struct_cache["coupled_size"],
                adjoint=True,
            )
    else:
        def _to_uncoupled_packed(coupled_packed):
            return transform @ coupled_packed

        def _from_uncoupled_packed(uncoupled_packed):
            return transform_dag @ uncoupled_packed

    def _to_uncoupled_state(coupled_state):
        packed = coupled_state.to_packed(dtype=complex)
        uncoupled_packed = _to_uncoupled_packed(packed)
        return uncoupled_state_layout.from_packed(uncoupled_packed)

    def _from_uncoupled_state(uncoupled_state):
        packed = uncoupled_state.to_packed(dtype=complex)
        coupled_packed = _from_uncoupled_packed(packed)
        return coupled_state_layout.from_packed(coupled_packed)

    reduced_callback = op.reduced_matvec or op.aux_reduced_matvec
    packed_callback = op.packed_matvec or op.aux_packed_matvec
    if packed_callback is not None:
        def packed_apply(coupled_packed):
            coupled_packed = np.asarray(coupled_packed)
            uncoupled_packed = _to_uncoupled_packed(coupled_packed)
            out_uncoupled = np.asarray(packed_callback(uncoupled_packed))
            return _from_uncoupled_packed(out_uncoupled)

        diag = None
        if op.diag is not None:
            unc_diag = np.asarray(op.diag, dtype=float).reshape(-1)
            diag = np.real((np.abs(transform) ** 2).T @ unc_diag)
        return LocalOperator(
            packed_matvec=packed_apply,
            packed_block_matrices=None,
            diag=diag,
            name=op.name,
        )

    if reduced_callback is not None:
        def reduced_apply(coupled_state):
            uncoupled_state = _to_uncoupled_state(coupled_state)
            out_state = reduced_callback(uncoupled_state)
            if not isinstance(out_state, ReducedStateVector):
                raise TypeError("reduced_matvec must return a ReducedStateVector.")
            if out_state.layout != uncoupled_state_layout:
                raise ValueError("reduced_matvec must preserve the reduced-state layout.")
            return _from_uncoupled_state(out_state)

        diag = None
        if op.diag is not None:
            unc_diag = np.asarray(op.diag, dtype=float).reshape(-1)
            diag = np.real((np.abs(transform) ** 2).T @ unc_diag)
        return LocalOperator(
            reduced_matvec=reduced_apply,
            diag=diag,
            name=op.name,
        )

    if op.matrix is not None:
        matrix = np.asarray(op.matrix)
        def reduced_apply(coupled_state):
            uncoupled_state = _to_uncoupled_state(coupled_state)
            uncoupled_vec = uncoupled_state.to_packed(dtype=complex)
            out_vec = matrix @ uncoupled_vec
            out_state = uncoupled_state_layout.from_packed(out_vec)
            return _from_uncoupled_state(out_state)

        coupled_diag = np.real(np.diag(transform_dag @ matrix @ transform))
        return LocalOperator(
            reduced_matvec=reduced_apply,
            diag=coupled_diag,
            name=op.name,
        )

    if op.matvec is not None:
        def reduced_apply(coupled_state):
            uncoupled_state = _to_uncoupled_state(coupled_state)
            uncoupled_vec = uncoupled_state.to_packed(dtype=complex)
            out_vec = np.asarray(op.matvec(uncoupled_vec))
            out_state = uncoupled_state_layout.from_packed(out_vec)
            return _from_uncoupled_state(out_state)

        diag = None
        if op.diag is not None:
            unc_diag = np.asarray(op.diag, dtype=float).reshape(-1)
            diag = np.real((np.abs(transform) ** 2).T @ unc_diag)
        return LocalOperator(
            reduced_matvec=reduced_apply,
            diag=diag,
            name=op.name,
        )

    return LocalOperator(
        tensor_matvec=_coupled_operator_tensor_matvec(op, two_site, uncoupled_layout),
        name=op.name,
    )


def _solve_generalized_dense(H, N, *, tol):
    """
    Solve the Hermitian generalized eigenproblem ``H x = E N x``.
    """
    H = 0.5 * (np.asarray(H) + np.asarray(H).conj().T)
    N = 0.5 * (np.asarray(N) + np.asarray(N).conj().T)
    s, U = np.linalg.eigh(N)
    keep = s > max(float(tol), 1e-12)
    if not np.any(keep):
        raise ValueError("Generalized norm operator is numerically singular.")
    X = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
    H_ortho = X.conj().T @ H @ X
    evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
    coeff = X @ evecs[:, 0]
    norm = np.sqrt(np.real(np.vdot(coeff, N @ coeff)))
    if norm > 1e-15:
        coeff = coeff / norm
    resid = H @ coeff - evals[0] * (N @ coeff)
    return float(np.real(evals[0])), coeff, float(np.linalg.norm(resid))


def _solve_generalized_dense_roots(H, N=None, *, nroots=1, tol=1e-12):
    """Return the lowest roots of a dense standard or generalized problem."""
    H = 0.5 * (np.asarray(H, dtype=complex) + np.asarray(H, dtype=complex).conj().T)
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be positive.")
    if N is None:
        evals, evecs = np.linalg.eigh(H)
        idx = np.argsort(np.real(evals))[:nroots]
        roots = []
        residuals = []
        for i in idx:
            vec = evecs[:, i]
            roots.append(vec)
            residuals.append(float(np.linalg.norm(H @ vec - evals[i] * vec)))
        return np.real(evals[idx]).astype(float), roots, residuals

    N = 0.5 * (np.asarray(N, dtype=complex) + np.asarray(N, dtype=complex).conj().T)
    s, U = np.linalg.eigh(N)
    keep = s > max(float(tol), 1e-12)
    if not np.any(keep):
        raise ValueError("Generalized norm operator is numerically singular.")
    X = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
    H_ortho = X.conj().T @ H @ X
    evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
    idx = np.argsort(np.real(evals))[:nroots]
    roots = []
    residuals = []
    for i in idx:
        vec = X @ evecs[:, i]
        norm = np.sqrt(np.real(np.vdot(vec, N @ vec)))
        if norm > 1e-15:
            vec = vec / norm
        roots.append(vec)
        residuals.append(float(np.linalg.norm(H @ vec - evals[i] * (N @ vec))))
    return np.real(evals[idx]).astype(float), roots, residuals


def _solve_standard_davidson_roots(
    operator,
    template,
    layout,
    guess_vec,
    *,
    nroots=1,
    projector_basis=None,
    tol=1e-8,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    precond=None,
):
    """Return low roots of a standard local problem without materializing H."""
    op_resolved, diag = _resolve_davidson_operator(operator, template, layout)
    guess_arr = np.asarray(guess_vec, dtype=complex)
    dim = int(guess_arr.shape[0]) if guess_arr.ndim > 1 else int(guess_arr.reshape(-1).size)
    if diag is None:
        diag = np.zeros(dim, dtype=float)
        missing_diag = True
    else:
        diag = np.asarray(diag, dtype=float).reshape(dim)
        missing_diag = False
    guess_vec = guess_arr.reshape(dim, -1) if guess_arr.ndim > 1 else guess_arr.reshape(dim)
    if projector_basis is not None:
        projector_basis = np.asarray(projector_basis, dtype=complex)
        if projector_basis.ndim != 2 or projector_basis.shape[0] != dim:
            raise ValueError("projector_basis must have shape (full_dim, projected_dim).")
        proj_dim = int(projector_basis.shape[1])
        if int(nroots) > proj_dim:
            raise ValueError("Cannot request more Davidson roots than the projected dimension.")

        def full_matvec(vec):
            return np.asarray(op_resolved(vec) if callable(op_resolved) else op_resolved @ vec, dtype=complex).reshape(dim)

        def projected_matvec(coeff):
            full = projector_basis @ np.asarray(coeff, dtype=complex).reshape(proj_dim)
            return projector_basis.conj().T @ full_matvec(full)

        projected_diag = np.real((np.abs(projector_basis) ** 2).T @ diag)
        projected_guess = projector_basis.conj().T @ guess_vec
        if np.linalg.norm(projected_guess) <= 1e-15:
            projected_guess = None
        projected_dense = None
        try:
            energies, vecs, info = davidson(
                projected_matvec,
                int(nroots),
                tol=tol,
                itermax=itermax,
                diag=projected_diag,
                precond=precond,
                guess=projected_guess,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                return_info=True,
            )
        except (RuntimeError, ValueError, IndexError):
            projected_dense = _materialize_local_matrix(projected_matvec, proj_dim)
        if projected_dense is None and np.asarray(vecs).shape[1] < int(nroots):
            projected_dense = _materialize_local_matrix(projected_matvec, proj_dim)
        if projected_dense is not None:
            evals, evecs = np.linalg.eigh(0.5 * (projected_dense + projected_dense.conj().T))
            order = np.argsort(np.real(evals))[: int(nroots)]
            energies = np.real(evals[order]).astype(float)
            vecs = evecs[:, order]
            info = {
                "iterations": 0,
                "converged": True,
                "subspace_dim": int(proj_dim),
                "restarts": 0,
            }
        roots = [
            projector_basis @ np.asarray(vecs[:, i], dtype=complex).reshape(proj_dim)
            for i in range(int(nroots))
        ]
        residuals = [
            float(np.linalg.norm(full_matvec(root) - energies[i] * root))
            for i, root in enumerate(roots)
        ]
        return np.real(energies).astype(float), roots, residuals, {
            "davidson_iterations": int(info.get("iterations", 0)),
            "davidson_converged": bool(info.get("converged", False)),
            "subspace_dim": int(info.get("subspace_dim", 0)),
            "restarts": int(info.get("restarts", 0)),
            "missing_diagonal_preconditioner": bool(missing_diag),
            "projected_dim": int(proj_dim),
        }
    energies, vecs, info = davidson(
        op_resolved,
        int(nroots),
        tol=tol,
        itermax=itermax,
        diag=diag,
        precond=precond,
        guess=guess_vec,
        max_space=max_space,
        tol_residual=tol_residual,
        lindep=lindep,
        return_info=True,
    )
    roots = [np.asarray(vecs[:, i], dtype=complex).reshape(dim) for i in range(int(nroots))]
    residuals = info.get("residual_norms")
    if residuals is None:
        residuals = [
            float(np.linalg.norm(np.asarray(op_resolved(root), dtype=complex).reshape(dim) - energies[i] * root))
            if callable(op_resolved)
            else float(np.linalg.norm(np.asarray(op_resolved) @ root - energies[i] * root))
            for i, root in enumerate(roots)
        ]
    return np.real(energies).astype(float), roots, [float(x) for x in residuals], {
        "davidson_iterations": int(info.get("iterations", 0)),
        "davidson_converged": bool(info.get("converged", False)),
        "subspace_dim": int(info.get("subspace_dim", 0)),
        "restarts": int(info.get("restarts", 0)),
        "missing_diagonal_preconditioner": bool(missing_diag),
    }


def _target_projector_basis(
    target_operator,
    template,
    layout,
    *,
    target_value,
    target_tol,
    min_dim,
    max_dim=4096,
):
    op_resolved, _ = _resolve_davidson_operator(target_operator, template, layout)
    dim = sum(entry.size for entry in layout)
    if dim > int(max_dim):
        return None, None
    matrix = (
        np.asarray(op_resolved, dtype=complex)
        if isinstance(op_resolved, np.ndarray)
        else _materialize_local_matrix(op_resolved, dim)
    )
    matrix = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(matrix)
    distance = np.abs(np.real(evals) - float(target_value))
    keep = np.where(distance <= float(target_tol))[0]
    if keep.size < int(min_dim):
        keep = np.argsort(distance)[: int(min_dim)]
    if keep.size < int(min_dim):
        return None, None
    return evecs[:, keep], np.real(evals[keep]).astype(float)


def _operator_expectations(operator, template, layout, root_vecs, *, norm_operator=None):
    op_resolved, _ = _resolve_davidson_operator(operator, template, layout)
    norm_resolved = None
    if norm_operator is not None:
        norm_resolved, _ = _resolve_davidson_operator(norm_operator, template, layout)
    values = []
    for vec in root_vecs:
        vec = np.asarray(vec, dtype=complex).reshape(-1)
        out = np.asarray(op_resolved(vec) if callable(op_resolved) else op_resolved @ vec, dtype=complex).reshape(-1)
        if norm_resolved is None:
            denom = np.vdot(vec, vec)
        else:
            nvec = np.asarray(
                norm_resolved(vec) if callable(norm_resolved) else norm_resolved @ vec,
                dtype=complex,
            ).reshape(-1)
            denom = np.vdot(vec, nvec)
        if abs(denom) <= 1e-15:
            values.append(np.inf)
        else:
            values.append(float(np.real(np.vdot(vec, out) / denom)))
    return values


def _select_targeted_roots(
    energies,
    root_vecs,
    residuals,
    *,
    nstates,
    target_values=None,
    target_value=None,
    target_tol=1e-6,
):
    order = list(range(len(root_vecs)))
    if target_values is not None and target_value is not None:
        target_values = [float(x) for x in target_values]
        target_value = float(target_value)
        target_tol = float(target_tol)
        matching = [
            idx
            for idx in order
            if abs(target_values[idx] - target_value) <= target_tol
        ]
        if len(matching) >= int(nstates):
            order = sorted(matching, key=lambda idx: (float(energies[idx]), abs(target_values[idx] - target_value)))
        else:
            order = sorted(order, key=lambda idx: (abs(target_values[idx] - target_value), float(energies[idx])))
    else:
        order = sorted(order, key=lambda idx: float(energies[idx]))
    selected = order[: int(nstates)]
    return (
        np.asarray([energies[idx] for idx in selected], dtype=float),
        [root_vecs[idx] for idx in selected],
        [residuals[idx] for idx in selected],
        selected,
    )


def _solve_orthonormalized_dense(H, N, *, tol):
    """
    Solve a generalized local problem by orthonormalizing the metric first.

    This returns the same ground-state solution as the generalized solve, but
    makes the "canonical local basis" step explicit: first orthonormalize with
    respect to ``N``, then diagonalize the transformed Hermitian operator.
    """
    H = 0.5 * (np.asarray(H) + np.asarray(H).conj().T)
    N = 0.5 * (np.asarray(N) + np.asarray(N).conj().T)
    s, U = np.linalg.eigh(N)
    keep = s > max(float(tol), 1e-12)
    if not np.any(keep):
        raise ValueError("Canonical local metric is numerically singular.")
    X = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
    H_ortho = X.conj().T @ H @ X
    evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
    coeff = X @ evecs[:, 0]
    norm = np.sqrt(np.real(np.vdot(coeff, N @ coeff)))
    if norm > 1e-15:
        coeff = coeff / norm
    resid = H @ coeff - evals[0] * (N @ coeff)
    return float(np.real(evals[0])), coeff, float(np.linalg.norm(resid))


def solve_local_two_site(
    two_site,
    local_operator,
    *,
    norm_operator=None,
    canonical_norm=False,
    guess=None,
    tol=1e-8,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    precond=None,
    dense_fallback_dim=512,
    orthonormalized_dense_dim=96,
    couple_physical=False,
    nstates=1,
    weights=None,
    root_guesses=None,
    root_target_operator=None,
    root_target_value=None,
    root_target_tol=1e-6,
    root_selection_buffer=2,
):
    """
    Solve a local effective two-site problem with Davidson.

    Returns
    -------
    tuple
        ``(optimized_two_site_tensor, local_objective_dict)``.
    """
    if isinstance(local_operator, TwoSiteEffectiveH):
        if norm_operator is not None:
            raise ValueError(
                "Specify norm_operator either through TwoSiteEffectiveH or via the norm_operator argument, not both."
            )
        canonical_norm = bool(canonical_norm or local_operator.canonical_norm)
        norm_operator = local_operator.norm_operator
        local_operator = local_operator.operator

    vec0, layout = pack_two_site_state(two_site)
    if guess is None:
        guess_vec = vec0
    elif isinstance(guess, NonabelianTensor):
        guess_vec, _ = pack_two_site_state(guess, layout=layout)
    else:
        guess_vec = np.asarray(guess)

    if np.linalg.norm(guess_vec) < 1e-15:
        raise ValueError("Initial guess for solve_local_two_site must have nonzero norm.")
    root_guess_vecs = None
    if root_guesses is not None:
        root_guess_vecs = []
        for root_guess in root_guesses:
            try:
                if isinstance(root_guess, NonabelianTensor):
                    root_vec, _ = pack_two_site_state(root_guess, layout=layout)
                else:
                    root_vec = np.asarray(root_guess)
            except ValueError:
                continue
            root_vec = np.asarray(root_vec, dtype=complex).reshape(-1)
            if root_vec.shape == np.asarray(guess_vec).reshape(-1).shape and np.linalg.norm(root_vec) > 1e-15:
                root_guess_vecs.append(root_vec)
        if not root_guess_vecs:
            root_guess_vecs = None
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")
    if weights is None:
        weights = np.ones(nstates, dtype=float) / nstates
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != nstates:
            raise ValueError("weights must match nstates.")
        weight_sum = float(np.sum(weights))
        if abs(weight_sum) <= 1e-15:
            raise ValueError("weights must not sum to zero.")
        weights = weights / weight_sum

    coupled_mode = couple_physical
    if coupled_mode not in {False, True, "auto"}:
        raise ValueError("couple_physical must be one of False, True, or 'auto'.")

    transform_error = None
    if coupled_mode is not False:
        try:
            coupled_template = _coupled_two_site_template(two_site)
        except Exception as exc:
            if coupled_mode is True:
                raise
            transform_error = exc
            coupled_template = coupled_layout = None
        else:
            _, coupled_layout = _pack_tensor_state(coupled_template)
            transform_error = None
            _, _, transform = _build_basis_transform(
                two_site,
                coupled=coupled_template,
                coupled_layout=coupled_layout,
                uncoupled_layout=layout,
            )
            transform = np.asarray(transform)
            transform_dag = transform.conj().T
    else:
        coupled_template = coupled_layout = None
        transform = transform_dag = None

    op_preview = _normalize_local_operator(local_operator)
    norm_preview = None if norm_operator is None else _normalize_local_operator(norm_operator)
    canonical_norm_requested = bool(canonical_norm or (norm_preview is not None and norm_preview.identity_like))
    canonical_norm = bool(canonical_norm_requested)
    use_uncoupled_canonical_path = False
    use_uncoupled_orthonormalized_path = False
    if canonical_norm and coupled_template is None:
        norm_operator = None
        norm_preview = None
    use_aux_packed_operator = guess_vec.size > 64
    effective_op = op_preview
    if (
        use_aux_packed_operator
        and op_preview.packed_matvec is None
        and op_preview.aux_packed_matvec is not None
    ):
        effective_op = LocalOperator(
            packed_matvec=op_preview.aux_packed_matvec,
            packed_block_matrices=op_preview.packed_block_matrices,
            diag=op_preview.diag,
            name=op_preview.name,
            identity_like=op_preview.identity_like,
        )
    effective_norm_op = norm_preview
    if (
        use_aux_packed_operator
        and norm_preview is not None
        and norm_preview.packed_matvec is None
        and norm_preview.aux_packed_matvec is not None
    ):
        effective_norm_op = LocalOperator(
            packed_matvec=norm_preview.aux_packed_matvec,
            packed_block_matrices=norm_preview.packed_block_matrices,
            diag=norm_preview.diag,
            name=norm_preview.name,
            identity_like=norm_preview.identity_like,
        )
    if canonical_norm and effective_norm_op is not None:
        effective_norm_op = None
        norm_operator = None
        norm_preview = None

    root_target_op = None
    if root_target_operator is not None:
        root_target_op = _normalize_local_operator(root_target_operator)
        if (
            use_aux_packed_operator
            and root_target_op.packed_matvec is None
            and root_target_op.aux_packed_matvec is not None
        ):
            root_target_op = LocalOperator(
                packed_matvec=root_target_op.aux_packed_matvec,
                packed_block_matrices=root_target_op.packed_block_matrices,
                diag=root_target_op.diag,
                name=root_target_op.name,
                identity_like=root_target_op.identity_like,
            )

    if nstates > 1:
        if coupled_template is not None:
            if guess is None:
                guess_coupled = coupled_template
            elif isinstance(guess, NonabelianTensor):
                guess_coupled = _couple_two_site_tensor(guess)
            else:
                guess_uncoupled = unpack_two_site_state(np.asarray(guess_vec), two_site, layout=layout)
                guess_coupled = _couple_two_site_tensor(guess_uncoupled)

            coupled_operator = _lift_operator_to_coupled(
                effective_op,
                two_site,
                layout,
                coupled_template,
                coupled_layout,
                transform=transform,
                transform_dag=transform_dag,
            )
            coupled_norm_operator = (
                None
                if effective_norm_op is None
                else _lift_operator_to_coupled(
                    effective_norm_op,
                    two_site,
                    layout,
                    coupled_template,
                    coupled_layout,
                    transform=transform,
                    transform_dag=transform_dag,
                )
            )
            coupled_target_operator = (
                None
                if root_target_op is None
                else _lift_operator_to_coupled(
                    root_target_op,
                    two_site,
                    layout,
                    coupled_template,
                    coupled_layout,
                    transform=transform,
                    transform_dag=transform_dag,
                )
            )
            guess_coupled_vec, _ = _pack_tensor_state(guess_coupled, layout=coupled_layout)
            root_coupled_guess_matrix = None
            if root_guess_vecs:
                cols = []
                for root_vec in root_guess_vecs:
                    root_uncoupled = unpack_two_site_state(root_vec, two_site, layout=layout)
                    root_coupled = _couple_two_site_tensor(root_uncoupled)
                    root_coupled_vec, _ = _pack_tensor_state(root_coupled, layout=coupled_layout)
                    cols.append(np.asarray(root_coupled_vec, dtype=complex).reshape(-1))
                if cols:
                    root_coupled_guess_matrix = np.column_stack(cols)
            dim = guess_coupled_vec.size
            nsolve = int(nstates)
            projected_min_dim = min(
                dim,
                max(nsolve, int(nstates) + max(0, int(root_selection_buffer))),
            )
            projector_basis = None
            projector_target_values = None
            if coupled_target_operator is not None and root_target_value is not None:
                projector_basis, projector_target_values = _target_projector_basis(
                    coupled_target_operator,
                    coupled_template,
                    coupled_layout,
                    target_value=root_target_value,
                    target_tol=root_target_tol,
                    min_dim=projected_min_dim,
                )
                if projector_basis is not None:
                    nsolve = min(nsolve, int(projector_basis.shape[1]))
            if projector_basis is None and coupled_target_operator is not None and root_target_value is not None:
                nsolve = min(dim, max(nsolve, int(nstates) + max(0, int(root_selection_buffer))))
            solver_info = {}
            if coupled_norm_operator is None:
                energies, root_vecs, residuals, solver_info = _solve_standard_davidson_roots(
                    coupled_operator,
                    coupled_template,
                    coupled_layout,
                    root_coupled_guess_matrix if root_coupled_guess_matrix is not None else guess_coupled_vec,
                    nroots=nsolve,
                    projector_basis=projector_basis,
                    tol=tol,
                    itermax=itermax,
                    max_space=max_space,
                    tol_residual=tol_residual,
                    lindep=lindep,
                    precond=precond,
                )
            elif dense_fallback_dim is None or dim > int(dense_fallback_dim):
                raise NotImplementedError(
                    "State-averaged coupled non-Abelian local solves currently require "
                    "a standard/canonical local norm for block Davidson, or "
                    f"a dense generalized fallback with dim <= {dense_fallback_dim}; got dim={dim}."
                )
            else:
                operator_dense, _ = _resolve_davidson_operator(coupled_operator, coupled_template, coupled_layout)
                H_matrix = (
                    np.asarray(operator_dense)
                    if isinstance(operator_dense, np.ndarray)
                    else _materialize_local_matrix(operator_dense, dim)
                )
                norm_dense, _ = _resolve_davidson_operator(coupled_norm_operator, coupled_template, coupled_layout)
                N_matrix = (
                    np.asarray(norm_dense)
                    if isinstance(norm_dense, np.ndarray)
                    else _materialize_local_matrix(norm_dense, dim)
                )
                energies, root_vecs, residuals = _solve_generalized_dense_roots(
                    H_matrix,
                    N_matrix,
                    nroots=nsolve,
                    tol=max(tol, 1e-12),
                )
                solver_info = {
                    "davidson_iterations": 0,
                    "davidson_converged": True,
                    "subspace_dim": int(dim),
                    "restarts": 0,
                    "generalized_dense_fallback": True,
                }
            target_values = None
            if coupled_target_operator is not None and root_target_value is not None:
                target_values = _operator_expectations(
                    coupled_target_operator,
                    coupled_template,
                    coupled_layout,
                    root_vecs,
                    norm_operator=coupled_norm_operator,
                )
            energies, root_vecs, residuals, selected_roots = _select_targeted_roots(
                energies,
                root_vecs,
                residuals,
                nstates=nstates,
                target_values=target_values,
                target_value=root_target_value,
                target_tol=root_target_tol,
            )
            optimized_roots = []
            for root_idx, vec in enumerate(root_vecs):
                reference = guess_coupled_vec if root_idx == 0 else None
                vec = _canonicalize_eigenvector(vec, reference=reference)
                optimized_coupled = _unpack_tensor_state(vec, coupled_template, layout=coupled_layout)
                optimized_roots.append(_uncouple_two_site_tensor(optimized_coupled))
            local_weights = weights[: len(energies)]
            local_weights = local_weights / np.sum(local_weights)
            residual = max(residuals) if residuals else 0.0
            return optimized_roots[0], {
                "energy": float(energies[0]),
                "state_energies": [float(x) for x in energies],
                "state_average_energy": float(np.dot(local_weights, energies)),
                "state_average_weights": [float(x) for x in local_weights],
                "optimized_roots": optimized_roots,
                "metric": float(residual),
                "residual": float(residual),
                "davidson_iterations": int(solver_info.get("davidson_iterations", 0)),
                "davidson_converged": bool(solver_info.get("davidson_converged", True)),
                "subspace_dim": int(solver_info.get("subspace_dim", dim)),
                "dense_fallback": bool(solver_info.get("generalized_dense_fallback", False)),
                "block_davidson": not bool(solver_info.get("generalized_dense_fallback", False)),
                "restarts": int(solver_info.get("restarts", 0)),
                "coupled_physical_used": True,
                "canonical_norm": canonical_norm,
                "effective_local_problem": (
                    "state_averaged_coupled_dense"
                    if solver_info.get("generalized_dense_fallback", False)
                    else "state_averaged_coupled_davidson"
                ),
                "target_irrep_filtered": True,
                "root_target_values": (
                    [float(target_values[idx]) for idx in selected_roots]
                    if target_values is not None
                    else None
                ),
                "root_target_value": None if root_target_value is None else float(root_target_value),
                "root_selection_used": target_values is not None,
                "root_projector_dim": int(projector_basis.shape[1]) if projector_basis is not None else None,
                "root_projector_target_values": (
                    [float(x) for x in projector_target_values]
                    if projector_target_values is not None
                    else None
                ),
                "root_selection_candidates": int(nsolve),
                "nstates": int(nstates),
            }

        dim = guess_vec.size
        root_guess_matrix = None
        if root_guess_vecs:
            root_guess_matrix = np.column_stack(root_guess_vecs)
        nsolve = int(nstates)
        projected_min_dim = min(
            dim,
            max(nsolve, int(nstates) + max(0, int(root_selection_buffer))),
        )
        projector_basis = None
        projector_target_values = None
        if root_target_op is not None and root_target_value is not None:
            projector_basis, projector_target_values = _target_projector_basis(
                root_target_op,
                two_site,
                layout,
                target_value=root_target_value,
                target_tol=root_target_tol,
                min_dim=projected_min_dim,
            )
            if projector_basis is not None:
                nsolve = min(nsolve, int(projector_basis.shape[1]))
        if projector_basis is None and root_target_op is not None and root_target_value is not None:
            nsolve = min(dim, max(nsolve, int(nstates) + max(0, int(root_selection_buffer))))
        solver_info = {}
        if effective_norm_op is None:
            energies, root_vecs, residuals, solver_info = _solve_standard_davidson_roots(
                effective_op,
                two_site,
                layout,
                root_guess_matrix if root_guess_matrix is not None else guess_vec,
                nroots=nsolve,
                projector_basis=projector_basis,
                tol=tol,
                itermax=itermax,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                precond=precond,
            )
        elif dense_fallback_dim is None or dim > int(dense_fallback_dim):
            raise NotImplementedError(
                "State-averaged non-Abelian local solves currently require "
                "a standard/canonical local norm for block Davidson, or "
                f"a dense generalized fallback with dim <= {dense_fallback_dim}; got dim={dim}."
            )
        else:
            operator_dense, _ = _resolve_davidson_operator(effective_op, two_site, layout)
            H_matrix = (
                np.asarray(operator_dense)
                if isinstance(operator_dense, np.ndarray)
                else _materialize_local_matrix(operator_dense, dim)
            )
            norm_dense, _ = _resolve_davidson_operator(effective_norm_op, two_site, layout)
            N_matrix = (
                np.asarray(norm_dense)
                if isinstance(norm_dense, np.ndarray)
                else _materialize_local_matrix(norm_dense, dim)
            )
            energies, root_vecs, residuals = _solve_generalized_dense_roots(
                H_matrix,
                N_matrix,
                nroots=nsolve,
                tol=max(tol, 1e-12),
            )
            solver_info = {
                "davidson_iterations": 0,
                "davidson_converged": True,
                "subspace_dim": int(dim),
                "restarts": 0,
                "generalized_dense_fallback": True,
            }
        target_values = None
        if root_target_op is not None and root_target_value is not None:
            target_values = _operator_expectations(
                root_target_op,
                two_site,
                layout,
                root_vecs,
                norm_operator=effective_norm_op,
            )
        energies, root_vecs, residuals, selected_roots = _select_targeted_roots(
            energies,
            root_vecs,
            residuals,
            nstates=nstates,
            target_values=target_values,
            target_value=root_target_value,
            target_tol=root_target_tol,
        )
        optimized_roots = []
        for root_idx, vec in enumerate(root_vecs):
            reference = guess_vec if root_idx == 0 else None
            vec = _canonicalize_eigenvector(vec, reference=reference)
            optimized_roots.append(unpack_two_site_state(vec, two_site, layout=layout))
        local_weights = weights[: len(energies)]
        local_weights = local_weights / np.sum(local_weights)
        residual = max(residuals) if residuals else 0.0
        return optimized_roots[0], {
            "energy": float(energies[0]),
            "state_energies": [float(x) for x in energies],
            "state_average_energy": float(np.dot(local_weights, energies)),
            "state_average_weights": [float(x) for x in local_weights],
            "optimized_roots": optimized_roots,
            "metric": float(residual),
            "residual": float(residual),
            "davidson_iterations": int(solver_info.get("davidson_iterations", 0)),
            "davidson_converged": bool(solver_info.get("davidson_converged", True)),
            "subspace_dim": int(solver_info.get("subspace_dim", dim)),
            "dense_fallback": bool(solver_info.get("generalized_dense_fallback", False)),
            "block_davidson": not bool(solver_info.get("generalized_dense_fallback", False)),
            "restarts": int(solver_info.get("restarts", 0)),
            "coupled_physical_used": False,
            "canonical_norm": canonical_norm,
            "effective_local_problem": (
                "state_averaged_dense"
                if solver_info.get("generalized_dense_fallback", False)
                else "state_averaged_davidson"
            ),
            "root_target_values": (
                [float(target_values[idx]) for idx in selected_roots]
                if target_values is not None
                else None
            ),
            "root_target_value": None if root_target_value is None else float(root_target_value),
            "root_selection_used": target_values is not None,
            "root_projector_dim": int(projector_basis.shape[1]) if projector_basis is not None else None,
            "root_projector_target_values": (
                [float(x) for x in projector_target_values]
                if projector_target_values is not None
                else None
            ),
            "root_selection_candidates": int(nsolve),
            "nstates": int(nstates),
        }

    if coupled_template is not None:
        if guess is None:
            guess_coupled = coupled_template
        elif isinstance(guess, NonabelianTensor):
            guess_coupled = _couple_two_site_tensor(guess)
        else:
            guess_uncoupled = unpack_two_site_state(np.asarray(guess_vec), two_site, layout=layout)
            guess_coupled = _couple_two_site_tensor(guess_uncoupled)

        coupled_operator = _lift_operator_to_coupled(
            op_preview,
            two_site,
            layout,
            coupled_template,
            coupled_layout,
            transform=transform,
            transform_dag=transform_dag,
        )
        coupled_norm_operator = (
            None
            if norm_preview is None
            else _lift_operator_to_coupled(
                norm_preview,
                two_site,
                layout,
                coupled_template,
                coupled_layout,
                transform=transform,
                transform_dag=transform_dag,
            )
        )
        coupled_dim = sum(entry.size for entry in coupled_layout)
        ortho_limit = None
        if orthonormalized_dense_dim is not None:
            ortho_limit = int(orthonormalized_dense_dim)
        elif dense_fallback_dim is not None:
            ortho_limit = int(dense_fallback_dim)
        if coupled_norm_operator is not None and ortho_limit is not None and coupled_dim <= ortho_limit:
                if coupled_mode == "auto":
                    optimized_coupled, objective = _solve_tensor_davidson(
                        coupled_template,
                        coupled_layout,
                        coupled_operator,
                        norm_operator=coupled_norm_operator,
                        guess=guess_coupled,
                        tol=tol,
                        itermax=itermax,
                        max_space=max_space,
                        tol_residual=tol_residual,
                        lindep=lindep,
                        precond=precond,
                    )
                    optimized = _uncouple_two_site_tensor(optimized_coupled)
                    objective["coupled_physical"] = True
                    objective["coupled_physical_used"] = False
                    objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
                    objective["canonical_norm"] = canonical_norm_requested
                    objective["canonical_norm_used"] = canonical_norm_requested
                    objective["effective_local_problem"] = "orthonormalized_dense"
                    objective["dense_fallback"] = True
                    return optimized, objective
                H_coupled, _ = _resolve_davidson_operator(
                    coupled_operator,
                    coupled_template,
                    coupled_layout,
                )
                N_coupled, _ = _resolve_davidson_operator(
                    coupled_norm_operator,
                    coupled_template,
                    coupled_layout,
                )
                H_matrix = (
                    np.asarray(H_coupled)
                    if isinstance(H_coupled, np.ndarray)
                    else _materialize_local_matrix(H_coupled, coupled_dim)
                )
                N_matrix = (
                    np.asarray(N_coupled)
                    if isinstance(N_coupled, np.ndarray)
                    else _materialize_local_matrix(N_coupled, coupled_dim)
                )
                guess_coupled_vec, _ = _pack_tensor_state(guess_coupled, layout=coupled_layout)
                energy, vec, residual = _solve_orthonormalized_dense(H_matrix, N_matrix, tol=tol)
                vec = _canonicalize_eigenvector(vec, reference=guess_coupled_vec)
                optimized_coupled = _unpack_tensor_state(vec, coupled_template, layout=coupled_layout)
                optimized = _uncouple_two_site_tensor(optimized_coupled)
                objective = {
                    "energy": float(energy),
                    "metric": residual,
                    "residual": residual,
                    "davidson_iterations": 0,
                    "davidson_converged": False,
                    "subspace_dim": int(coupled_dim),
                    "coupled_physical": True,
                    "coupled_physical_used": bool(coupled_mode is True),
                    "canonical_norm": canonical_norm_requested,
                    "canonical_norm_used": canonical_norm_requested,
                    "effective_local_problem": (
                        "orthonormalized_standard" if canonical_norm_requested else "orthonormalized_dense"
                    ),
                    "dense_fallback": True,
                    "operator_representation": "dense",
                    "norm_operator_representation": "dense",
                }
                if coupled_mode == "auto":
                    objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
                return optimized, objective
        if canonical_norm_requested:
            use_uncoupled_canonical_path = True
            norm_operator = None
            norm_preview = None
        elif (
            coupled_mode == "auto"
            and coupled_norm_operator is not None
            and ortho_limit is not None
            and guess_vec.size <= ortho_limit
        ):
            use_uncoupled_orthonormalized_path = True
            norm_operator = effective_norm_op
            norm_preview = effective_norm_op
        else:
            optimized_coupled, objective = _solve_tensor_davidson(
                coupled_template,
                coupled_layout,
                coupled_operator,
                norm_operator=coupled_norm_operator,
                guess=guess_coupled,
                tol=tol,
                itermax=itermax,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                precond=precond,
            )
            optimized = _uncouple_two_site_tensor(optimized_coupled)
            objective["coupled_physical"] = True
            objective["coupled_physical_used"] = True
            objective["canonical_norm"] = canonical_norm
            objective["effective_local_problem"] = "standard" if canonical_norm else "generalized"
            objective["dense_fallback"] = False
            if canonical_norm_requested and coupled_template is not None:
                objective["canonical_norm_skipped"] = "coupled_physical_path"
            return optimized, objective

    use_reduced_operator_path = (
        effective_op.reduced_matvec is not None
        or effective_op.packed_matvec is not None
        or (effective_norm_op is not None and (
            effective_norm_op.reduced_matvec is not None
            or effective_norm_op.packed_matvec is not None
        ))
    )
    use_tensor_generalized_path = (
        effective_norm_op is not None
        and (
            effective_op.tensor_matvec is not None
            or effective_op.reduced_matvec is not None
            or effective_op.packed_matvec is not None
        )
        and (
            effective_norm_op.tensor_matvec is not None
            or effective_norm_op.reduced_matvec is not None
            or effective_norm_op.packed_matvec is not None
        )
    )
    if effective_norm_op is not None and (
        use_uncoupled_orthonormalized_path
        or (canonical_norm_requested and orthonormalized_dense_dim is not None)
    ):
        dim = guess_vec.size
        if dim <= int(orthonormalized_dense_dim):
            operator_dense, _ = _resolve_davidson_operator(effective_op, two_site, layout)
            norm_dense, _ = _resolve_davidson_operator(effective_norm_op, two_site, layout)
            H_matrix = (
                np.asarray(operator_dense)
                if isinstance(operator_dense, np.ndarray)
                else _materialize_local_matrix(operator_dense, dim)
            )
            N_matrix = (
                np.asarray(norm_dense)
                if isinstance(norm_dense, np.ndarray)
                else _materialize_local_matrix(norm_dense, dim)
            )
            energy, vec, residual = _solve_orthonormalized_dense(H_matrix, N_matrix, tol=tol)
            vec = _canonicalize_eigenvector(vec, reference=guess_vec)
            optimized = unpack_two_site_state(vec, two_site, layout=layout)
            objective = {
                "energy": float(energy),
                "metric": residual,
                "residual": residual,
                "davidson_iterations": 0,
                "davidson_converged": False,
                "subspace_dim": int(dim),
                "coupled_physical_used": False,
                "canonical_norm": canonical_norm,
                "dense_fallback": True,
                "operator_representation": "dense",
                "norm_operator_representation": "dense",
                "effective_local_problem": (
                    "orthonormalized_standard" if canonical_norm else "orthonormalized_dense"
                ),
            }
            if use_uncoupled_canonical_path:
                objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
            elif use_uncoupled_orthonormalized_path:
                objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
            elif canonical_norm_requested and coupled_template is not None:
                objective["canonical_norm_skipped"] = "coupled_physical_path"
            if transform_error is not None and coupled_mode == "auto":
                objective["coupled_physical_skipped"] = type(transform_error).__name__
            return optimized, objective
    if use_reduced_operator_path or use_tensor_generalized_path:
        optimized, objective = _solve_tensor_davidson(
            two_site,
            layout,
            effective_op,
            norm_operator=effective_norm_op,
            guess=guess,
            tol=tol,
            itermax=itermax,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            precond=precond,
        )
        objective["dense_fallback"] = False
        objective["coupled_physical_used"] = False
        objective["canonical_norm"] = canonical_norm
        objective["effective_local_problem"] = "standard" if canonical_norm else "generalized"
        if use_uncoupled_canonical_path:
            objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
        elif use_uncoupled_orthonormalized_path:
            objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
        elif canonical_norm_requested and coupled_template is not None:
            objective["canonical_norm_skipped"] = "coupled_physical_path"
        if transform_error is not None and coupled_mode == "auto":
            objective["coupled_physical_skipped"] = type(transform_error).__name__
        return optimized, objective

    operator, diag = _resolve_davidson_operator(local_operator, two_site, layout)
    norm_resolved = None
    if norm_operator is not None:
        norm_resolved, _ = _resolve_davidson_operator(norm_operator, two_site, layout)
    fallback_used = False
    try:
        if norm_resolved is not None:
            raise RuntimeError("Use dense generalized solve path when a norm operator is supplied.")
        theta, vecs, info = davidson(
            operator,
            neigen=1,
            tol=tol,
            itermax=itermax,
            diag=diag,
            precond=precond,
            guess=guess_vec.reshape(-1, 1),
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            return_info=True,
        )
        vec = _canonicalize_eigenvector(vecs[:, 0], reference=guess_vec)
        optimized = unpack_two_site_state(vec, two_site, layout=layout)
        residual = float(info["residual_norms"][0]) if info["residual_norms"] is not None else None
        objective = {
            "energy": float(theta[0]),
            "metric": residual,
            "residual": residual,
            "davidson_iterations": int(info["iterations"]),
            "davidson_converged": bool(info["converged"]),
            "subspace_dim": int(info["subspace_dim"]),
        }
    except Exception:
        dim = guess_vec.size
        if dense_fallback_dim is None or dim > int(dense_fallback_dim):
            if norm_resolved is not None:
                matrix = (
                    operator
                    if isinstance(operator, np.ndarray)
                    else _materialize_local_matrix(operator, dim)
                )
                norm_matrix = (
                    norm_resolved
                    if isinstance(norm_resolved, np.ndarray)
                    else _materialize_local_matrix(norm_resolved, dim)
                )
                energy, vec, residual = _solve_generalized_dense(matrix, norm_matrix, tol=tol)
                vec = _canonicalize_eigenvector(vec, reference=guess_vec)
                optimized = unpack_two_site_state(vec, two_site, layout=layout)
                objective = {
                    "energy": float(energy),
                    "metric": residual,
                    "residual": residual,
                    "davidson_iterations": int(itermax),
                    "davidson_converged": False,
                    "subspace_dim": int(dim),
                    "generalized_norm": True,
                    "dense_fallback": True,
                }
                objective["coupled_physical_used"] = False
                objective["canonical_norm"] = canonical_norm
                objective["effective_local_problem"] = "standard" if canonical_norm else "generalized"
                if use_uncoupled_canonical_path:
                    objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
                elif canonical_norm_requested and coupled_template is not None:
                    objective["canonical_norm_skipped"] = "coupled_physical_path"
                if transform_error is not None and coupled_mode == "auto":
                    objective["coupled_physical_skipped"] = type(transform_error).__name__
                return optimized, objective
            raise
        matrix = operator if isinstance(operator, np.ndarray) else _materialize_local_matrix(operator, dim)
        if norm_resolved is not None:
            norm_matrix = (
                norm_resolved if isinstance(norm_resolved, np.ndarray)
                else _materialize_local_matrix(norm_resolved, dim)
            )
            energy, vec, residual = _solve_generalized_dense(matrix, norm_matrix, tol=tol)
            vec = _canonicalize_eigenvector(vec, reference=guess_vec)
            optimized = unpack_two_site_state(vec, two_site, layout=layout)
            objective = {
                "energy": float(energy),
                "metric": residual,
                "residual": residual,
                "davidson_iterations": int(itermax),
                "davidson_converged": False,
                "subspace_dim": int(dim),
                "generalized_norm": True,
            }
        else:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            vec = _canonicalize_eigenvector(eigvecs[:, 0], reference=guess_vec)
            optimized = unpack_two_site_state(vec, two_site, layout=layout)
            objective = {
                "energy": float(np.real(eigvals[0])),
                "metric": 0.0,
                "residual": 0.0,
                "davidson_iterations": int(itermax),
                "davidson_converged": False,
                "subspace_dim": int(dim),
            }
        fallback_used = True
    objective["dense_fallback"] = fallback_used
    objective["coupled_physical_used"] = False
    objective["canonical_norm"] = canonical_norm
    objective["effective_local_problem"] = "standard" if canonical_norm else "generalized"
    if use_uncoupled_canonical_path:
        objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
    elif canonical_norm_requested and coupled_template is not None:
        objective["canonical_norm_skipped"] = "coupled_physical_path"
    if transform_error is not None and coupled_mode == "auto":
        objective["coupled_physical_skipped"] = type(transform_error).__name__
    return optimized, objective
