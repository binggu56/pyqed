#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local two-site operator helpers for fixed-layout non-Abelian tensors.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
import time

import numpy as np
try:
    from scipy import linalg as scipy_linalg
except Exception:  # pragma: no cover - SciPy is optional for lightweight installs.
    scipy_linalg = None

try:
    from scipy.sparse.linalg import LinearOperator, eigsh
except Exception:  # pragma: no cover - SciPy is optional for lightweight installs.
    LinearOperator = None
    eigsh = None

from pyqed.davidson import davidson
from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors

from .basis import TwoSiteBasis
from .contraction import combine_legs, split_legs
from .renormalized import (
    BlockSparseOrthonormalizedLocalProblem,
    ComponentOrthonormalizedLocalProblem,
    DiagonalMetricBlock,
    DiagonalMetricTransform,
    DirectOrthonormalFactorizedTable,
    FactorizedRouteMetricBlock,
    KroneckerMetricBlock,
    KroneckerMetricTransform,
    OrthonormalizedLocalProblem,
    RenormalizedComponentBasis,
    compile_orthonormal_block_table,
    get_direct_factorized_orthonormal_kernel_policy,
    get_su2_kernel_policy,
)
from pyqed.symmetry import IrrepTensor

_BASIS_TRANSFORM_DENSE_MATVEC_SIZE = 0
_SU2_CPP_DAVIDSON_BLOCK_SIZE = 1
_SU2_UNPRECONDITIONED_SPARSE_MIN_DIM = 256
_COMPONENT_BASIS_CACHE_MAX_SIZE = 128
_COMPONENT_BASIS_CACHE_MAX_NUMERIC_ELEMENTS = 1_000_000
_COMPONENT_BASIS_CACHE_TOTAL_NUMERIC_ELEMENTS = 2_000_000
_COMPONENT_BASIS_CACHE = {}
_METRIC_ENTRY_KERNEL_SIGNATURE_CACHE_MAX_SIZE = 256
_METRIC_ENTRY_KERNEL_SIGNATURE_CACHE = {}
_METRIC_BLOCK_TRANSFORM_CACHE_MAX_SIZE = 256
_METRIC_BLOCK_TRANSFORM_CACHE_MAX_ELEMENTS = 1_000_000
_METRIC_BLOCK_TRANSFORM_CACHE_TOTAL_ELEMENTS = 2_000_000
_METRIC_BLOCK_TRANSFORM_CACHE = {}
_METRIC_BLOCK_TRANSFORM_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "puts": 0,
    "real_fast": 0,
    "large_diagonal_fast": 0,
    "large_diagonal_sample_rejects": 0,
    "cholesky_fast": 0,
    "scipy_subset_eigh": 0,
}
_QCHEM_COMPONENT_METRIC_MAX_DENSE_PARENT_DIM = 2048
_PACKED_DAVIDSON_BASIS_MAX_BYTES = 32 * 1024 * 1024
_PACKED_DAVIDSON_OWNED_BASIS_ARRAYS = 3
_CPP_CANONICAL_METRIC_MAX_COMPONENT_ELEMENTS = 4 * 1024 * 1024
_CPP_CANONICAL_METRIC_MAX_TRANSFORM_ELEMENTS = 4 * 1024 * 1024


@dataclass(frozen=True)
class PackedEntry:
    key: tuple
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass(frozen=True)
class ReducedStateLayout:
    """
    Packed reduced-state layout for a local tensor problem.

    :param entries: Ordered block entries defining the packed vector slices.
    :param basis: Optional explicit two-site symmetry basis for ``entries``.
        The basis is metadata only for comparison and hashing, preserving the
        older tuple-layout cache behavior.
    """

    entries: tuple[PackedEntry, ...]
    basis: TwoSiteBasis | None = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        if self.basis is not None and not self.basis.compatible_with_layout(entries):
            raise ValueError("ReducedStateLayout basis is incompatible with its packed entries.")

    @property
    def size(self):
        if self.basis is not None:
            return self.basis.size
        return sum(entry.size for entry in self.entries)

    def basis_vector(self, index, *, dtype=complex):
        if self.basis is not None:
            key, block = self.basis.basis_block(index, dtype=dtype)
            return ReducedStateVector(layout=self, blocks={key: block})

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
        if self.basis is not None:
            return ReducedStateVector(
                layout=self,
                blocks=self.basis.blocks_from_packed(vector),
            )

        vector = np.asarray(vector)
        blocks = {}
        for entry in self.entries:
            piece = vector[entry.offset:entry.offset + entry.size]
            if np.linalg.norm(piece) > 0.0:
                blocks[entry.key] = piece.reshape(entry.shape)
        return ReducedStateVector(layout=self, blocks=blocks)

    def to_packed(self, state, *, dtype=None):
        if self.basis is not None:
            return self.basis.blocks_to_packed(state.blocks, dtype=dtype)

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
        """
        Build a diagonal reduced preconditioner from packed diagonals.

        :param layout: Reduced-state layout or compatible two-site basis.
        :param h_diag: Packed Hamiltonian diagonal.
        :param n_diag: Optional packed norm diagonal.
        :returns: ``ReducedDiagonalPreconditioner`` with sector-keyed blocks.
        """

        layout = _reduced_state_layout(layout)
        h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
        if h_diag.size != layout.size:
            raise ValueError("Hamiltonian diagonal must match the reduced-state layout size.")
        if n_diag is None:
            n_diag = np.ones(layout.size, dtype=float)
        else:
            n_diag = np.asarray(n_diag, dtype=float).reshape(-1)
            if n_diag.size != layout.size:
                raise ValueError("Norm diagonal must match the reduced-state layout size.")

        if layout.basis is not None:
            h_blocks = layout.basis.blocks_from_packed(h_diag, drop_zeros=False)
            n_blocks = layout.basis.blocks_from_packed(n_diag, drop_zeros=False)
        else:
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


def _packed_layout_object(layout):
    """
    Return the most explicit packed-layout object available.

    :param layout: ``TwoSiteBasis``, ``ReducedStateLayout``, or packed entries.
    :returns: ``TwoSiteBasis`` when present, otherwise packed entries.
    """

    if isinstance(layout, TwoSiteBasis):
        return layout
    if isinstance(layout, ReducedStateLayout) and layout.basis is not None:
        return layout.basis
    return _layout_entries(layout)


def _packed_layout_size(layout):
    """
    Return the packed dimension for a local layout object.

    :param layout: ``TwoSiteBasis`` or packed entries.
    :returns: Packed vector size.
    """

    if isinstance(layout, TwoSiteBasis):
        return layout.size
    return sum(entry.size for entry in layout)


class _PackedBlockMatrixSource:
    """
    Lazy block-matrix accessor aligned with packed local-layout entries.

    :param blocks: Sequence of blocks or provider exposing ``block_matrix_for``.
    :param entries: Packed local layout entries.
    :param label: Human-readable block role for validation errors.
    :param default: Matrix returned for every entry when no provider is given.
    """

    def __init__(self, blocks, entries, *, label, default=None):
        self.entries = tuple(entries)
        self.label = str(label)
        self.default = default
        self.provider = blocks if hasattr(blocks, "block_matrix_for") else None
        self.blocks = None if self.provider is not None else tuple(blocks)
        if self.blocks is not None and len(self.blocks) != len(self.entries):
            raise ValueError(f"{self.label} block list must match the packed layout length.")
        self._cache = {}

    def __len__(self):
        """Return the number of packed layout entries."""

        return len(self.entries)

    def __getitem__(self, index):
        """
        Return one validated block matrix.

        :param index: Packed entry index.
        :returns: Dense square block or ``None``.
        """

        index = int(index)
        if index in self._cache:
            return self._cache[index]
        entry = self.entries[index]
        if self.provider is None:
            block = self.default if self.blocks is None else self.blocks[index]
        else:
            block = self.provider.block_matrix_for(entry)
        if block is None:
            normalized = None
        else:
            normalized = np.asarray(block, dtype=complex)
            if normalized.shape != (entry.size, entry.size):
                raise ValueError(
                    f"{self.label} block for {entry.key!r} has shape {normalized.shape!r}, "
                    f"expected {(entry.size, entry.size)!r}."
                )
        self._cache[index] = normalized
        return normalized

    @property
    def materialized_count(self):
        """
        Return the number of provider blocks queried so far.

        :returns: Count of cached block lookups.
        """

        return len(self._cache)


@dataclass(frozen=True)
class PackedBlockPreconditioner:
    layout: object
    h_blocks: object
    n_blocks: object

    @classmethod
    def from_layout_blocks(cls, layout, h_blocks, *, n_blocks=None):
        """
        Build a packed block preconditioner from per-entry matrices.

        :param layout: ``TwoSiteBasis``, ``ReducedStateLayout``, or packed entries.
        :param h_blocks: Hamiltonian block matrices aligned with layout entries.
        :param n_blocks: Optional norm block matrices aligned with layout entries.
        :returns: ``PackedBlockPreconditioner``.
        """

        layout = _packed_layout_object(layout)
        entries = _layout_entries(layout)
        h_blocks = _PackedBlockMatrixSource(
            h_blocks,
            entries,
            label="Hamiltonian",
        )
        if n_blocks is None:
            n_blocks = _PackedBlockMatrixSource(
                [None] * len(entries),
                entries,
                label="Norm",
            )
        else:
            n_blocks = _PackedBlockMatrixSource(
                n_blocks,
                entries,
                label="Norm",
            )
        return cls(layout=layout, h_blocks=h_blocks, n_blocks=n_blocks)

    @property
    def materialized_block_count(self):
        """
        Return the number of Hamiltonian self-blocks materialized so far.

        :returns: Number of cached block matrices in the Hamiltonian source.
        """

        return getattr(self.h_blocks, "materialized_count", 0)

    def apply(self, resid, theta):
        resid = np.asarray(resid, dtype=complex).reshape(-1)
        size = _packed_layout_size(self.layout)
        if resid.size != size:
            raise ValueError("Packed residual size must match the preconditioner layout.")

        out = np.zeros_like(resid)
        if isinstance(self.layout, TwoSiteBasis):
            for idx, (entry, block) in enumerate(self.layout.iter_packed_blocks(resid, drop_zeros=False)):
                piece = np.asarray(block).reshape(entry.size)
                if np.linalg.norm(piece) <= 1e-15:
                    continue
                h_block = self.h_blocks[idx]
                n_block = self.n_blocks[idx]
                if h_block is None:
                    self.layout.write_packed_block(out, entry, piece)
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
                self.layout.write_packed_block(out, entry, corrected)
            return out

        for idx, entry in enumerate(_layout_entries(self.layout)):
            piece = resid[entry.offset:entry.offset + entry.size]
            if np.linalg.norm(piece) <= 1e-15:
                continue
            h_block = self.h_blocks[idx]
            n_block = self.n_blocks[idx]
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
    if not isinstance(tensor, IrrepTensor):
        raise ValueError("_pack_tensor_state expects an IrrepTensor.")

    if isinstance(layout, TwoSiteBasis):
        blocks = layout.blocks_from_two_site_tensor(
            tensor,
            drop_zeros=False,
            copy=False,
        )
        present_dtypes = [
            np.asarray(block).dtype
            for block in blocks.values()
        ]
        dtype = np.result_type(*(present_dtypes or [float]))
        return layout.blocks_to_packed(blocks, dtype=dtype), layout.entries

    if layout is None:
        entries = []
        offset = 0
        for key in sorted(tensor.data):
            block = np.asarray(tensor.data[key])
            size = int(block.size)
            entries.append(PackedEntry(tuple(key), tuple(block.shape), offset, size))
            offset += size
        layout = tuple(entries)
    else:
        layout = _layout_entries(layout)

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
    if not isinstance(template, IrrepTensor):
        raise ValueError("_unpack_tensor_state expects an IrrepTensor template.")

    if isinstance(layout, TwoSiteBasis):
        return layout.tensor_from_blocks(
            layout.blocks_from_packed(vector, drop_zeros=False),
            template,
        )

    layout = _layout_entries(layout)
    vector = np.asarray(vector)
    data = {}
    for entry in layout:
        piece = vector[entry.offset:entry.offset + entry.size]
        data[entry.key] = piece.reshape(entry.shape)
    return IrrepTensor(
        data,
        [leg[:] for leg in template.qns],
        template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=template.metadata.copy(),
    )


def _layout_entries(layout):
    """
    Return packed entries from any supported local-layout descriptor.

    :param layout: A ``ReducedStateLayout``, ``TwoSiteBasis``, or iterable of
        ``PackedEntry`` objects.
    :returns: Tuple of packed entries.
    """

    if isinstance(layout, ReducedStateLayout):
        return layout.entries
    if isinstance(layout, TwoSiteBasis):
        return layout.entries
    return tuple(layout)


def _two_site_basis_for_layout(template, layout):
    """
    Build or recover the explicit two-site basis for a local layout.

    :param template: Rank-4 two-site tensor defining the sector axes.
    :param layout: Packed local layout or an existing ``TwoSiteBasis``.
    :returns: ``TwoSiteBasis`` when the template is a two-site tensor,
        otherwise ``None``.
    """

    if isinstance(layout, TwoSiteBasis):
        return layout
    if not isinstance(template, IrrepTensor) or template.rank != 4:
        return None
    return TwoSiteBasis.from_tensor_and_layout(template, _layout_entries(layout))


def _reduced_state_layout(layout, *, basis=None):
    """
    Normalize a layout descriptor to ``ReducedStateLayout``.

    :param layout: Packed entries, ``ReducedStateLayout``, or ``TwoSiteBasis``.
    :param basis: Optional explicit basis attached when ``layout`` is raw
        packed entries.
    :returns: Reduced-state layout object.
    """

    if isinstance(layout, ReducedStateLayout):
        return layout
    if isinstance(layout, TwoSiteBasis):
        return ReducedStateLayout(layout.entries, basis=layout)
    return ReducedStateLayout(tuple(layout), basis=basis)


def _reduced_state_layout_for_tensor(template, layout):
    """
    Build a reduced-state layout carrying the tensor's two-site basis.

    :param template: Rank-4 local tensor.
    :param layout: Packed local layout.
    :returns: Reduced-state layout with compatible ``TwoSiteBasis`` metadata.
    """

    return _reduced_state_layout(
        layout,
        basis=_two_site_basis_for_layout(template, layout),
    )


def _reduced_state_layout_for_operator(op, template, layout):
    """
    Build a reduced-state layout, preferring operator-supplied basis metadata.

    :param op: Local operator that may carry a ``TwoSiteBasis``.
    :param template: Rank-4 local tensor.
    :param layout: Packed local layout.
    :returns: Reduced-state layout with explicit basis metadata.
    """

    basis = op.basis if isinstance(getattr(op, "basis", None), TwoSiteBasis) else None
    return _reduced_state_layout(
        layout,
        basis=basis or _two_site_basis_for_layout(template, layout),
    )


def _operator_basis_for_layout(op, layout):
    """
    Return an operator-supplied basis when it matches a packed layout.

    :param op: Local operator that may carry ``TwoSiteBasis`` metadata.
    :param layout: Packed local layout used to check compatibility.
    :returns: Compatible ``TwoSiteBasis`` or ``None``.
    :raises ValueError: If the operator basis is present but incompatible.
    """

    basis = getattr(op, "basis", None)
    if basis is None:
        return None
    if not isinstance(basis, TwoSiteBasis):
        raise TypeError("LocalOperator.basis must be a TwoSiteBasis when provided.")
    if not basis.compatible_with_layout(_layout_entries(layout)):
        raise ValueError("LocalOperator.basis is incompatible with the current two-site layout.")
    return basis


def _tensor_to_reduced_state(tensor, *, state_layout):
    if state_layout.basis is not None:
        return ReducedStateVector(
            layout=state_layout,
            blocks=state_layout.basis.blocks_from_two_site_tensor(tensor),
        )

    blocks = {}
    for entry in state_layout.entries:
        if entry.key not in tensor.data:
            continue
        block = np.asarray(tensor.data[entry.key])
        if np.linalg.norm(block.reshape(-1)) > 0.0:
            blocks[entry.key] = np.array(block, copy=True)
    return ReducedStateVector(layout=state_layout, blocks=blocks)


def _reduced_state_to_tensor(state, template):
    if state.layout.basis is not None:
        return state.layout.basis.tensor_from_blocks(state.blocks, template)
    else:
        data = {}
        for entry in state.layout.entries:
            if entry.key in state.blocks:
                data[entry.key] = np.array(state.blocks[entry.key], copy=True).reshape(entry.shape)
            elif entry.key in template.data:
                data[entry.key] = np.zeros(entry.shape, dtype=np.asarray(template.data[entry.key]).dtype)
            else:
                data[entry.key] = np.zeros(entry.shape, dtype=float)
    metadata = template.metadata.copy()
    metadata["contracted_channel_blocks_current"] = False
    return IrrepTensor(
        data,
        [leg[:] for leg in template.qns],
        template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=metadata,
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
        if not isinstance(out, IrrepTensor):
            raise TypeError("tensor_matvec must return an IrrepTensor.")
        return _tensor_to_reduced_state(out, state_layout=state_layout)

    return matvec


@dataclass(frozen=True)
class LocalOperator:
    """
    Small wrapper for an effective two-site operator.

    Exactly one of ``matrix``, ``matvec``, ``tensor_matvec``,
    ``reduced_matvec``, or ``packed_matvec`` should be supplied.

    :param matrix: Dense local operator matrix.
    :param matvec: Packed-vector matvec callback.
    :param tensor_matvec: Tensor-to-tensor matvec callback.
    :param reduced_matvec: ``ReducedStateVector`` matvec callback.
    :param packed_matvec: Packed-vector matvec callback with direct reduced
        layout support.
    :param aux_reduced_matvec: Optional reduced callback used when callers
        prefer reduced operators.
    :param aux_packed_matvec: Optional packed callback used when callers
        prefer packed operators.
    :param packed_block_matrices: Optional per-layout block matrices for block
        preconditioning.
    :param basis: Optional explicit ``TwoSiteBasis`` for the packed local
        problem.
    :param diag: Optional packed diagonal.
    :param name: Optional operator label.
    :param identity_like: Whether the operator acts as an identity metric in
        the current basis.
    :param metadata: Optional diagnostics describing the operator source.
    :param local_operator_table: Optional typed renormalized local table
        provider.
    """

    matrix: object | None = None
    matvec: object | None = None
    tensor_matvec: object | None = None
    reduced_matvec: object | None = None
    packed_matvec: object | None = None
    aux_reduced_matvec: object | None = None
    aux_packed_matvec: object | None = None
    packed_block_matrices: object | None = None
    basis: object | None = None
    diag: object | None = None
    name: str | None = None
    identity_like: bool = False
    metadata: dict | None = None
    local_operator_table: object | None = None

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
    if not isinstance(two_site, IrrepTensor) or two_site.rank != 4:
        raise ValueError("pack_two_site_state expects a rank-4 IrrepTensor.")

    return _pack_tensor_state(two_site, layout=layout)


def two_site_state_basis(two_site, *, layout=None):
    """
    Return the explicit basis for the packed rank-4 two-site local problem.

    :param two_site: Rank-4 non-Abelian two-site tensor.
    :param layout: Optional packed layout. When omitted, the tensor is packed
        to derive the canonical layout.
    :returns: Explicit ``TwoSiteBasis`` matching the packed layout.
    """
    if not isinstance(two_site, IrrepTensor) or two_site.rank != 4:
        raise ValueError("two_site_state_basis expects a rank-4 IrrepTensor.")
    if layout is None:
        _, layout = pack_two_site_state(two_site)
    return TwoSiteBasis.from_tensor_and_layout(two_site, _layout_entries(layout))


def unpack_two_site_state(vector, template, *, layout):
    """
    Rebuild a rank-4 non-Abelian tensor from a packed vector and template.
    """
    if not isinstance(template, IrrepTensor) or template.rank != 4:
        raise ValueError("unpack_two_site_state expects a rank-4 IrrepTensor template.")

    return _unpack_tensor_state(vector, template, layout=layout)


def _coupled_two_site_template(two_site, *, filter_boundary_target=False):
    """
    Build a rank-3 coupled-basis template by fusing the two physical legs.
    """
    if not isinstance(two_site, IrrepTensor) or two_site.rank != 4:
        raise ValueError("_coupled_two_site_template expects a rank-4 IrrepTensor.")
    coupled = combine_legs(two_site, (1, 2), new_axis=1, use_cg=True)
    if filter_boundary_target:
        return _filter_coupled_template_to_boundary_target(coupled)
    return coupled


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
    if not isinstance(coupled, IrrepTensor) or coupled.rank != 3:
        return coupled
    data = {
        key: block
        for key, block in coupled.data.items()
        if _fuses_to_boundary_target(key[0], key[1], key[2])
    }
    if not data:
        raise ValueError("Coupled two-site template has no blocks compatible with the boundary target sector.")
    return IrrepTensor(
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
        not isinstance(coupled_two_site, IrrepTensor)
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

    return IrrepTensor(
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


@dataclass(frozen=True)
class _KroneckerBasisTransformBlock:
    """A contiguous parent block carrying only its physical-leg transform."""

    row_slice: slice
    orthonormal_indices: np.ndarray
    left_dim: int
    selected_dim: int
    local_dim: int
    right_dim: int
    local_transform: np.ndarray

    def apply(self, vector):
        source = np.asarray(vector)[self.orthonormal_indices].reshape(
            self.left_dim,
            self.local_dim,
            self.right_dim,
        )
        return np.einsum(
            "sf,lfr->lsr",
            self.local_transform,
            source,
            optimize=True,
        ).reshape(-1)

    def adjoint_apply(self, vector):
        source = np.asarray(vector)[self.row_slice].reshape(
            self.left_dim,
            self.selected_dim,
            self.right_dim,
        )
        return np.einsum(
            "sf,lsr->lfr",
            self.local_transform.conj(),
            source,
            optimize=True,
        ).reshape(-1)

    def project_diagonal(self, diagonal):
        source = np.asarray(diagonal)[self.row_slice].reshape(
            self.left_dim,
            self.selected_dim,
            self.right_dim,
        )
        return np.einsum(
            "sf,lsr->lfr",
            np.abs(self.local_transform) ** 2,
            source,
            optimize=True,
        ).reshape(-1)

    def dense(self, dtype=None):
        dtype = self.local_transform.dtype if dtype is None else dtype
        result = np.zeros(
            (
                self.left_dim * self.selected_dim * self.right_dim,
                self.left_dim * self.local_dim * self.right_dim,
            ),
            dtype=dtype,
        )
        for left in range(self.left_dim):
            for right in range(self.right_dim):
                rows = (
                    (left * self.selected_dim + np.arange(self.selected_dim))
                    * self.right_dim
                    + right
                )
                cols = (
                    (left * self.local_dim + np.arange(self.local_dim))
                    * self.right_dim
                    + right
                )
                result[np.ix_(rows, cols)] = self.local_transform
        return result


def _apply_basis_transform_blocks(vector, blocks, out_size, *, adjoint=False):
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    out = np.zeros(out_size, dtype=vector.dtype)
    for block in blocks:
        if isinstance(block, _KroneckerBasisTransformBlock):
            if adjoint:
                out[block.orthonormal_indices] += block.adjoint_apply(vector)
            else:
                out[block.row_slice] += block.apply(vector)
            continue
        row_slice, in_indices, submat = block
        if adjoint:
            out[in_indices] += np.asarray(submat).conj().T @ vector[row_slice]
        else:
            out[row_slice] += np.asarray(submat) @ vector[in_indices]
    return out


@dataclass(frozen=True)
class _StructuredBasisTransform:
    """Block-sparse coupled-to-uncoupled basis transform."""

    blocks: tuple
    uncoupled_size: int
    coupled_size: int

    @property
    def shape(self):
        return int(self.uncoupled_size), int(self.coupled_size)

    @property
    def size(self):
        return int(self.uncoupled_size) * int(self.coupled_size)

    def __matmul__(self, vector):
        return _apply_basis_transform_blocks(
            vector,
            self.blocks,
            int(self.uncoupled_size),
            adjoint=False,
        )

    def adjoint_apply(self, vector):
        return _apply_basis_transform_blocks(
            vector,
            self.blocks,
            int(self.coupled_size),
            adjoint=True,
        )

    def project_diagonal(self, diagonal):
        diagonal = np.asarray(diagonal, dtype=float).reshape(
            int(self.uncoupled_size)
        )
        out = np.zeros(int(self.coupled_size), dtype=float)
        for block in self.blocks:
            if isinstance(block, _KroneckerBasisTransformBlock):
                out[block.orthonormal_indices] += block.project_diagonal(
                    diagonal
                )
                continue
            row_slice, in_indices, submat = block
            out[in_indices] += (
                np.abs(np.asarray(submat)) ** 2
            ).T @ diagonal[row_slice]
        return out

    def __array__(self, dtype=None, copy=None):
        dtype = complex if dtype is None else np.dtype(dtype)
        out = np.zeros(self.shape, dtype=dtype)
        for block in self.blocks:
            if isinstance(block, _KroneckerBasisTransformBlock):
                row_slice = block.row_slice
                in_indices = block.orthonormal_indices
                submat = block.dense(dtype=dtype)
            else:
                row_slice, in_indices, submat = block
            rows = np.arange(
                int(row_slice.start),
                int(row_slice.stop),
                dtype=int,
            )
            out[np.ix_(rows, np.asarray(in_indices, dtype=int))] += np.asarray(
                submat,
                dtype=dtype,
            )
        return np.array(out, copy=True) if copy is not False else out


def _build_basis_transform_direct(two_site, coupled, coupled_layout, orig_layout):
    if (
        not isinstance(coupled, IrrepTensor)
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

            local_transform = np.ascontiguousarray(local_transform)
            selected_dim = int(np.prod(entry.selected_shape, dtype=int))
            local_dim = int(local_transform.shape[1])
            in_indices = []
            for l in range(left_dim):
                for f_local in range(local_dim):
                    for r in range(right_dim):
                        idx = ((l * fused_dim_total) + (entry.offset + f_local)) * right_dim + r
                        in_indices.append(c_entry.offset + idx)
            row_slice = slice(out_entry.offset, out_entry.offset + out_entry.size)
            transform_blocks.append(
                _KroneckerBasisTransformBlock(
                    row_slice=row_slice,
                    orthonormal_indices=np.asarray(in_indices, dtype=int),
                    left_dim=int(left_dim),
                    selected_dim=selected_dim,
                    local_dim=local_dim,
                    right_dim=int(right_dim),
                    local_transform=local_transform,
                )
            )

    structured = _StructuredBasisTransform(
        blocks=tuple(transform_blocks),
        coupled_size=sum(entry.size for entry in coupled_layout),
        uncoupled_size=sum(entry.size for entry in orig_layout),
    )
    two_site.metadata["_basis_transform_struct_cache"] = {
        "coupled_layout": tuple(coupled_layout),
        "uncoupled_layout": tuple(orig_layout),
        "blocks": structured.blocks,
        "coupled_size": structured.coupled_size,
        "uncoupled_size": structured.uncoupled_size,
    }

    return structured


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
            aux_reduced_matvec=local_operator.get("aux_reduced_matvec"),
            aux_packed_matvec=local_operator.get("aux_packed_matvec"),
            packed_block_matrices=local_operator.get("packed_block_matrices"),
            basis=local_operator.get("basis"),
            diag=local_operator.get("diag"),
            name=local_operator.get("name"),
            identity_like=bool(local_operator.get("identity_like", False)),
            metadata=local_operator.get("metadata"),
            local_operator_table=local_operator.get("local_operator_table"),
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
            if not isinstance(out, IrrepTensor):
                raise TypeError("tensor_matvec must return an IrrepTensor.")
            packed, _ = pack_two_site_state(out, layout=layout)
            return packed
        diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
        return matvec, diag

    if op.reduced_matvec is not None:
        state_layout = _reduced_state_layout_for_operator(op, template, layout)

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


def _metric_orthogonalize_packed_vector(vector, metric_vector, basis, metric_basis, *, tol):
    """
    Orthogonalize one packed vector against an ``N``-orthonormal basis.

    :param vector: Candidate vector in the packed basis.
    :param metric_vector: ``N @ vector``.
    :param basis: Existing packed basis columns.
    :param metric_basis: Existing ``N @ basis`` columns.
    :param tol: Linear-dependence cutoff in the metric norm.
    :returns: ``(vector, metric_vector)`` normalized to unit metric norm, or
        ``(None, None)`` when the candidate is dependent.
    """

    vector = np.asarray(vector, dtype=complex).reshape(-1)
    metric_vector = np.asarray(metric_vector, dtype=complex).reshape(-1)
    if basis.size:
        for _ in range(2):
            overlap = basis.conj().T @ metric_vector
            vector = vector - basis @ overlap
            metric_vector = metric_vector - metric_basis @ overlap
    norm2 = float(np.real(np.vdot(vector, metric_vector)))
    if norm2 <= float(tol) ** 2:
        return None, None
    norm = np.sqrt(norm2)
    return vector / norm, metric_vector / norm


def _metric_orthonormalize_packed_columns(columns, metric_operator, *, tol=1e-12):
    """
    Build an ``N``-orthonormal packed basis from candidate columns.

    :param columns: Candidate packed basis vectors.
    :param metric_operator: Callable applying the positive local metric ``N``.
    :param tol: Linear-dependence cutoff in the metric norm.
    :returns: ``(V, N @ V)`` with ``V.conj().T @ N @ V = I``.
    """

    columns = np.asarray(columns, dtype=complex)
    if columns.ndim == 1:
        columns = columns.reshape(-1, 1)
    vectors = []
    metric_vectors = []
    size = columns.shape[0]
    for col in range(columns.shape[1]):
        vector = columns[:, col]
        metric_vector = np.asarray(metric_operator(vector), dtype=complex).reshape(-1)
        if metric_vector.size != size:
            raise ValueError("Metric operator output must match the packed vector size.")
        basis = (
            np.column_stack(vectors)
            if vectors
            else np.zeros((size, 0), dtype=complex)
        )
        metric_basis = (
            np.column_stack(metric_vectors)
            if metric_vectors
            else np.zeros((size, 0), dtype=complex)
        )
        vector, metric_vector = _metric_orthogonalize_packed_vector(
            vector,
            metric_vector,
            basis,
            metric_basis,
            tol=tol,
        )
        if vector is None:
            continue
        vectors.append(vector)
        metric_vectors.append(metric_vector)
    if not vectors:
        return (
            np.zeros((size, 0), dtype=complex),
            np.zeros((size, 0), dtype=complex),
        )
    return np.column_stack(vectors), np.column_stack(metric_vectors)


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
        if isinstance(guess, IrrepTensor):
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
    use_block_preconditioner=True,
    allow_unconverged=False,
    profile=False,
):
    if not isinstance(guess_state, ReducedStateVector):
        raise TypeError("_solve_reduced_generalized_davidson expects a ReducedStateVector guess.")
    if _reduced_norm(guess_state) < 1e-15:
        raise ValueError("Initial guess for Davidson must have nonzero norm.")

    h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
    if h_diag.size != state_layout.size:
        raise ValueError("Hamiltonian diagonal guess must match the packed state dimension.")

    timing = {
        "davidson": 0.0,
        "matvec": 0.0,
        "projected": 0.0,
        "precondition": 0.0,
    } if profile else None
    total_t0 = time.perf_counter() if profile else None
    h_matvec_count = 0
    n_matvec_count = 0
    H_raw = H
    N_raw = N
    has_norm_operator = N is not None

    def H_timed(vec):
        nonlocal h_matvec_count
        if not profile:
            return H_raw(vec)
        t0 = time.perf_counter()
        out = H_raw(vec)
        timing["matvec"] += time.perf_counter() - t0
        h_matvec_count += 1
        return out

    def N_timed(vec):
        nonlocal n_matvec_count
        if N_raw is None:
            return vec
        if not profile:
            return N_raw(vec)
        t0 = time.perf_counter()
        out = N_raw(vec)
        timing["matvec"] += time.perf_counter() - t0
        n_matvec_count += 1
        return out

    H = H_timed
    N = N_timed
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
        t0 = time.perf_counter() if profile else None
        Vp = _packed_matrix_from_reduced_vectors(V, state_layout)
        AVp = _packed_matrix_from_reduced_vectors(AV, state_layout)
        BVp = _packed_matrix_from_reduced_vectors(BV, state_layout)
        Hs = Vp.conj().T @ AVp
        Ns = Vp.conj().T @ BVp
        theta, coeff, _ = _solve_generalized_dense(Hs, Ns, tol=max(tol, 1e-12))
        if profile:
            timing["projected"] += time.perf_counter() - t0
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
        t0 = time.perf_counter() if profile else None
        corr = precondition(resid, theta, ritz)
        if profile:
            timing["precondition"] += time.perf_counter() - t0
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
    if profile:
        timing["davidson"] = time.perf_counter() - total_t0
    info = {
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
    if profile:
        info["solver_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
        info["matvec_count"] = int(h_matvec_count)
        info["norm_matvec_count"] = int(n_matvec_count)
    return float(theta), vec, info


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
    use_block_preconditioner=True,
    profile=False,
    initial_vectors=None,
    return_recycle_space=False,
):
    timing = {
        "davidson": 0.0,
        "matvec": 0.0,
        "projected": 0.0,
        "precondition": 0.0,
        "orthogonalize": 0.0,
        "restart": 0.0,
        "basis_update": 0.0,
        "final_reference": 0.0,
    } if profile else None
    h_matvec_count = 0
    n_matvec_count = 0
    total_t0 = time.perf_counter() if profile else None

    H_raw = H
    N_raw = N

    def H_timed(vec):
        nonlocal h_matvec_count
        if not profile:
            return H_raw(vec)
        t0 = time.perf_counter()
        out = H_raw(vec)
        timing["matvec"] += time.perf_counter() - t0
        h_matvec_count += 1
        return out

    def N_timed(vec):
        nonlocal n_matvec_count
        if N_raw is None:
            return vec
        if not profile:
            return N_raw(vec)
        t0 = time.perf_counter()
        out = N_raw(vec)
        timing["matvec"] += time.perf_counter() - t0
        n_matvec_count += 1
        return out

    guess_packed = np.asarray(guess_packed, dtype=complex).reshape(-1)
    if np.linalg.norm(guess_packed) < 1e-15:
        raise ValueError("Initial guess for Davidson must have nonzero norm.")

    h_diag = np.asarray(h_diag, dtype=float).reshape(-1)
    if h_diag.size != guess_packed.size:
        raise ValueError("Hamiltonian diagonal guess must match the packed state dimension.")

    has_norm_operator = N is not None
    H = H_timed
    N = N_timed
    if n_diag is None:
        n_diag = np.ones_like(h_diag)
    else:
        n_diag = np.asarray(n_diag, dtype=float).reshape(-1)
        if n_diag.size != guess_packed.size:
            raise ValueError("Norm diagonal guess must match the packed state dimension.")

    requested_max_space = (
        min(guess_packed.size, 48)
        if max_space is None
        else min(guess_packed.size, int(max_space))
    )
    basis_column_bytes = (
        int(guess_packed.size)
        * np.dtype(np.complex128).itemsize
        * int(_PACKED_DAVIDSON_OWNED_BASIS_ARRAYS)
    )
    budget_columns = (
        int(_PACKED_DAVIDSON_BASIS_MAX_BYTES) // max(1, basis_column_bytes)
    )
    minimum_columns = min(int(guess_packed.size), 2)
    max_space = max(
        minimum_columns,
        min(int(requested_max_space), max(1, int(budget_columns))),
    )
    tol_res = np.sqrt(tol) if tol_residual is None else tol_residual

    metric_orthonormal_krylov = bool(has_norm_operator)
    Vp = _build_iterative_guess(h_diag, 1, guess=guess_packed, diag_n=n_diag)
    if initial_vectors is not None:
        recycled = np.asarray(initial_vectors, dtype=complex)
        if recycled.ndim == 1:
            recycled = recycled[:, None]
        if recycled.ndim != 2 or recycled.shape[0] != guess_packed.size:
            raise ValueError(
                "Recycled Davidson vectors must match the packed dimension."
            )
        Vp = np.column_stack((Vp, recycled))
    if metric_orthonormal_krylov:
        t0 = time.perf_counter() if profile else None
        Vp, BVp = _metric_orthonormalize_packed_columns(Vp, N, tol=lindep)
        if profile:
            timing["orthogonalize"] += time.perf_counter() - t0
        if Vp.shape[1] == 0:
            raise ValueError("Initial generalized Davidson basis is singular in the local metric.")
    else:
        Vp = _orthonormalize_columns_dense(Vp, tol=lindep)
        BVp = np.column_stack([np.asarray(N(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
    AVp = np.column_stack([np.asarray(H(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
    preconditioner_mode = None
    seed_scores = h_diag / np.where(np.abs(n_diag) > 1.0e-12, n_diag, 1.0)
    seed_order = np.argsort(seed_scores)

    def _next_seed_vector():
        for idx in seed_order:
            seed = np.zeros_like(guess_packed, dtype=complex)
            seed[int(idx)] = 1.0
            if Vp.shape[1]:
                seed = seed - Vp @ (Vp.conj().T @ seed)
            norm = float(np.linalg.norm(seed))
            if norm > lindep:
                return seed / norm
        return None

    if precond is None and use_block_preconditioner and block_preconditioner is not None:
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
    Ns = None if metric_orthonormal_krylov else Vp.conj().T @ BVp

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

    def _lowest_projected_root_with_reference(Hs_local, Ns_local=None):
        Hs_local = 0.5 * (np.asarray(Hs_local) + np.asarray(Hs_local).conj().T)
        ref_proj = Vp.conj().T @ guess_packed
        if Ns_local is None:
            return _lowest_hermitian_projected_root(
                Hs_local,
                reference=ref_proj,
                tol=tol,
            )

        Ns_local = 0.5 * (np.asarray(Ns_local) + np.asarray(Ns_local).conj().T)
        if np.allclose(Ns_local, np.eye(Ns_local.shape[0], dtype=Ns_local.dtype), atol=1.0e-10, rtol=1.0e-10):
            return _lowest_projected_root_with_reference(Hs_local, None)
        theta_local, coeff_local, _ = _solve_generalized_dense(Hs_local, Ns_local, tol=max(tol, 1e-12))
        return theta_local, coeff_local

    for iterations in range(1, itermax + 1):
        t0 = time.perf_counter() if profile else None
        if metric_orthonormal_krylov:
            theta, coeff = _lowest_projected_root_with_reference(Hs, None)
        else:
            theta, coeff = _lowest_projected_root_with_reference(Hs, Ns)
        if profile:
            timing["projected"] += time.perf_counter() - t0
        ritz_p = Vp @ coeff
        aritz_p = AVp @ coeff
        britz_p = BVp @ coeff
        resid_p = aritz_p - theta * britz_p
        residual_norm = float(np.linalg.norm(resid_p))
        de = np.inf if prev_theta is None else abs(theta - prev_theta)
        min_explored_dim = min(guess_packed.size, int(max_space), 16)
        if (
            residual_norm <= tol_res
            and (prev_theta is None or de <= tol)
            and Vp.shape[1] >= min_explored_dim
        ):
            converged = True
            break

        t0 = time.perf_counter() if profile else None
        corr_p = precondition(resid_p, theta, ritz_p)
        if profile:
            timing["precondition"] += time.perf_counter() - t0
        if metric_orthonormal_krylov:
            t0 = time.perf_counter() if profile else None
            corr_n = np.asarray(N(corr_p), dtype=complex).reshape(-1)
            corr_p, corr_n = _metric_orthogonalize_packed_vector(
                corr_p,
                corr_n,
                Vp,
                BVp,
                tol=lindep,
            )
            if profile:
                timing["orthogonalize"] += time.perf_counter() - t0
            if corr_p is None:
                break
        else:
            t0 = time.perf_counter() if profile else None
            if Vp.shape[1]:
                corr_p = corr_p - Vp @ (Vp.conj().T @ corr_p)
            corr_norm = float(np.linalg.norm(corr_p))
            if corr_norm <= lindep:
                corr_p = _next_seed_vector()
                if corr_p is None:
                    break
            else:
                corr_p = corr_p / corr_norm
            if profile:
                timing["orthogonalize"] += time.perf_counter() - t0

        if Vp.shape[1] + 1 > max_space:
            if int(max_space) <= 1:
                break
            t0 = time.perf_counter() if profile else None
            restart_keep = max(
                1,
                min(int(max_space) - 1, max(2, int(max_space) // 32), 4),
            )
            if metric_orthonormal_krylov:
                Hs = 0.5 * (Hs + Hs.conj().T)
                _evals, restart_coeffs = np.linalg.eigh(Hs)
                restart_coeffs = restart_coeffs[:, :restart_keep]
            else:
                restart_coeffs = _generalized_restart_vectors(Hs, Ns, restart_keep, tol_local=max(tol, 1e-12))
            restart_vectors = []
            if restart_coeffs is not None:
                restart_vectors.extend(Vp @ restart_coeffs[:, i] for i in range(restart_coeffs.shape[1]))
            else:
                restart_vectors.append(ritz_p)
            restart_vectors.append(corr_p)
            if metric_orthonormal_krylov:
                Vp, BVp = _metric_orthonormalize_packed_columns(
                    np.column_stack(restart_vectors),
                    N,
                    tol=lindep,
                )
            else:
                Vp = _orthonormalize_columns_dense(np.column_stack(restart_vectors), tol=lindep)
                BVp = np.column_stack([np.asarray(N(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
            restarts += 1
            AVp = np.column_stack([np.asarray(H(Vp[:, i]), dtype=complex).reshape(-1) for i in range(Vp.shape[1])])
            Hs = Vp.conj().T @ AVp
            Ns = None if metric_orthonormal_krylov else Vp.conj().T @ BVp
            if profile:
                timing["restart"] += time.perf_counter() - t0
        else:
            t0 = time.perf_counter() if profile else None
            h_corr = np.asarray(H(corr_p), dtype=complex).reshape(-1)
            if not metric_orthonormal_krylov:
                corr_n = np.asarray(N(corr_p), dtype=complex).reshape(-1)
            h_overlap = Vp.conj().T @ h_corr
            Vp = np.column_stack([Vp, corr_p])
            AVp = np.column_stack([AVp, h_corr])
            BVp = np.column_stack([BVp, corr_n])
            Hs = _expand_projected_matrix(Hs, h_overlap, np.vdot(corr_p, h_corr))
            if not metric_orthonormal_krylov:
                n_overlap = Vp[:, :-1].conj().T @ corr_n
                Ns = _expand_projected_matrix(Ns, n_overlap, np.vdot(corr_p, corr_n))
            if profile:
                timing["basis_update"] += time.perf_counter() - t0
        prev_theta = theta

    t0 = time.perf_counter() if profile else None
    if metric_orthonormal_krylov:
        theta, coeff = _lowest_projected_root_with_reference(Hs, None)
    else:
        theta, coeff = _lowest_projected_root_with_reference(Hs, Ns)
    guess_h = np.asarray(H(guess_packed), dtype=complex).reshape(-1)
    guess_n = np.asarray(N(guess_packed), dtype=complex).reshape(-1)
    guess_denom = np.vdot(guess_packed, guess_n)
    use_reference_root = False
    if abs(guess_denom) > 1.0e-15:
        guess_theta = float(np.real(np.vdot(guess_packed, guess_h) / guess_denom))
        guess_residual = float(np.linalg.norm(guess_h - theta * guess_n))
        use_reference_root = (
            abs(guess_theta - theta) <= max(float(tol), 1.0e-12)
            and guess_residual <= max(float(tol_res), 1.0e-12)
        )
    if profile:
        timing["final_reference"] += time.perf_counter() - t0
    vec_packed = (
        np.array(guess_packed, copy=True)
        if use_reference_root
        else Vp @ coeff
    )
    vec_packed = _canonicalize_eigenvector(vec_packed, reference=guess_packed)
    residual_norm = float(np.linalg.norm((AVp @ coeff) - theta * (BVp @ coeff)))
    if profile:
        timing["davidson"] = time.perf_counter() - total_t0
    info = {
        "metric": residual_norm,
        "residual": residual_norm,
        "davidson_iterations": int(iterations),
        "davidson_converged": bool(converged),
        "subspace_dim": int(Vp.shape[1]),
        "generalized_norm": has_norm_operator,
        "tensor_davidson": True,
        "reduced_krylov": False,
        "packed_krylov": True,
        "metric_orthonormal_krylov": metric_orthonormal_krylov,
        "projected_problem": "standard" if metric_orthonormal_krylov else "generalized",
        "preconditioner_mode": preconditioner_mode,
        "reduced_preconditioner": False,
        "restarts": int(restarts),
        "packed_dimension": int(guess_packed.size),
        "requested_max_space": int(requested_max_space),
        "workspace_max_space": int(max_space),
        "workspace_budget_bytes": int(_PACKED_DAVIDSON_BASIS_MAX_BYTES),
        "estimated_basis_workspace_bytes": int(
            basis_column_bytes * int(max_space)
        ),
        "workspace_limited": bool(int(max_space) < int(requested_max_space)),
    }
    if profile:
        info["solver_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
        info["matvec_count"] = int(h_matvec_count)
        info["norm_matvec_count"] = int(n_matvec_count)
    if return_recycle_space:
        info["_recycle_space"] = np.array(
            Vp[:, : min(4, Vp.shape[1])], copy=True
        )
    return float(theta), vec_packed, info


def _solve_cpp_factor_route_davidson(
    guess_packed,
    H,
    *,
    h_diag,
    tol,
    itermax,
    max_space,
    profile,
):
    """Run an identity-metric raw-route Davidson solve entirely in C++."""

    projection_blocks = getattr(H, "coupled_transform_blocks", None)
    parent_H = getattr(H, "uncoupled_packed_matvec", H)
    compiled = getattr(parent_H, "compiled_factorized_terms", None)
    owner = getattr(compiled, "su2_moving_environment", None)
    factor_route_key = getattr(compiled, "_cpp_factor_route_key", None)
    dimension = int(np.asarray(guess_packed).size)
    parent_dimension = (
        int(getattr(H, "coupled_parent_dimension", -1))
        if projection_blocks is not None
        else dimension
    )
    if (
        compiled is None
        or owner is None
        or factor_route_key is None
        or not bool(getattr(compiled, "_cpp_factor_routes_installed", False))
        or parent_dimension < 1
        or not owner.factor_route_installed(
            factor_route_key,
            parent_dimension,
        )
    ):
        return None
    if (
        projection_blocks is not None
        and int(getattr(H, "coupled_dimension", -1)) != dimension
    ):
        return None

    projection_key = None
    if projection_blocks is not None:
        from . import _su2_kernel

        topology_values = [
            np.asarray(
                [parent_dimension, dimension, len(projection_blocks)],
                dtype=np.int64,
            )
        ]
        numeric_values = []
        for block in projection_blocks:
            if isinstance(block, _KroneckerBasisTransformBlock):
                row_slice = block.row_slice
                indices = block.orthonormal_indices
                transform = block.local_transform
                shape = np.asarray(
                    [
                        block.left_dim,
                        block.selected_dim,
                        block.local_dim,
                        block.right_dim,
                    ],
                    dtype=np.int64,
                )
            else:
                row_slice, indices, transform = block
                shape = np.asarray(np.asarray(transform).shape, dtype=np.int64)
            topology_values.extend(
                (
                    np.asarray(
                        [int(row_slice.start), int(row_slice.stop)],
                        dtype=np.int64,
                    ),
                    np.asarray(indices, dtype=np.int64),
                    shape,
                )
            )
            numeric_values.append(np.asarray(transform))
        topology_revision = _su2_kernel._cpp_array_revision(*topology_values)
        numeric_revision = _su2_kernel._cpp_array_revision(*numeric_values)
        projection_key = (
            f"standard-coupling:{factor_route_key}:"
            f"{int(topology_revision)}"
        )
        owner.install_indexed_factor_route_projection(
            projection_key,
            factor_route_key,
            tuple(projection_blocks),
            parent_dimension,
            dimension,
            int(topology_revision),
            int(numeric_revision),
        )

    requested_max_space = (
        min(dimension, 48)
        if max_space is None
        else min(dimension, int(max_space))
    )
    basis_column_bytes = (
        dimension
        * np.dtype(np.complex128).itemsize
        * int(_PACKED_DAVIDSON_OWNED_BASIS_ARRAYS)
    )
    budget_columns = (
        int(_PACKED_DAVIDSON_BASIS_MAX_BYTES) // max(1, basis_column_bytes)
    )
    workspace_max_space = max(
        min(dimension, 2),
        min(requested_max_space, max(1, int(budget_columns))),
    )
    started = time.perf_counter() if profile else None
    active_solve = getattr(
        owner,
        "active_bond_complementary_davidson",
        None,
    )
    active_key = (
        factor_route_key if projection_key is None else projection_key
    )
    active_ready = getattr(
        owner,
        "active_bond_complementary_action_ready",
        None,
    )
    direct_active_solve = bool(
        callable(active_solve)
        and callable(active_ready)
        and getattr(compiled, "cpp_owned_basis_topology", False)
        and active_ready(active_key, dimension)
    )
    if direct_active_solve:
        result = active_solve(
            active_key,
            np.asarray(guess_packed, dtype=complex),
            float(tol),
            int(itermax),
            int(workspace_max_space),
            True,
        )
    else:
        solve = (
            owner.factor_route_davidson
            if projection_key is None
            else owner.factor_route_projected_davidson
        )
        result = solve(
            factor_route_key if projection_key is None else projection_key,
            np.asarray(h_diag, dtype=complex),
            np.asarray(guess_packed, dtype=complex),
            float(tol),
            int(itermax),
            int(workspace_max_space),
            True,
        )
    elapsed = time.perf_counter() - started if profile else None
    if not bool(result.get("accepted", False)):
        raise RuntimeError("The C++ factor-route Davidson solve was not accepted.")
    info = {
        "metric": float(result["residual_norm"]),
        "residual": float(result["residual_norm"]),
        "davidson_iterations": int(result["iterations"]),
        "davidson_converged": bool(result["converged"]),
        "subspace_dim": int(result["basis_size"]),
        "generalized_norm": False,
        "tensor_davidson": True,
        "reduced_krylov": False,
        "packed_krylov": True,
        "metric_orthonormal_krylov": False,
        "projected_problem": "standard",
        "preconditioner_mode": "packed_diagonal",
        "reduced_preconditioner": False,
        "restarts": int(result["restarts"]),
        "packed_dimension": dimension,
        "requested_max_space": int(requested_max_space),
        "workspace_max_space": int(workspace_max_space),
        "workspace_budget_bytes": int(_PACKED_DAVIDSON_BASIS_MAX_BYTES),
        "estimated_basis_workspace_bytes": int(
            basis_column_bytes * workspace_max_space
        ),
        "workspace_limited": bool(workspace_max_space < requested_max_space),
        "cpp_davidson": True,
        "cpp_davidson_kind": str(result.get("kind")),
        "direct_complementary_action_executor": bool(direct_active_solve),
        "cpp_workspace_reused": bool(result.get("workspace_reused", False)),
        "matvec_count": int(result.get("matvec_calls", 0)),
    }
    if profile:
        info["solver_timing"] = {"davidson": float(elapsed)}
    vector = _canonicalize_eigenvector(
        np.asarray(result["vector"]),
        reference=np.asarray(guess_packed),
    )
    return float(result["energy"]), vector, info


def _solve_cpp_factor_route_generalized_davidson(
    guess_packed,
    H,
    N,
    *,
    h_diag,
    n_diag,
    tol,
    itermax,
    max_space,
    tol_residual,
    lindep,
    profile,
):
    """Run a raw-route Hamiltonian and compact metric entirely in C++."""

    projection_blocks = getattr(H, "coupled_transform_blocks", None)
    parent_H = getattr(H, "uncoupled_packed_matvec", H)
    parent_N = getattr(N, "uncoupled_packed_matvec", N)
    if projection_blocks is not None:
        if (
            getattr(N, "coupled_transform_blocks", None) is None
            or int(getattr(H, "coupled_dimension", -1))
            != int(np.asarray(guess_packed).size)
            or int(getattr(H, "coupled_parent_dimension", -1))
            != int(getattr(N, "coupled_parent_dimension", -2))
        ):
            return None
    compiled = getattr(parent_H, "compiled_factorized_terms", None)
    owner = getattr(compiled, "su2_moving_environment", None)
    factor_route_key = getattr(compiled, "_cpp_factor_route_key", None)
    if (
        compiled is None
        or owner is None
        or factor_route_key is None
        or not bool(getattr(compiled, "_cpp_factor_routes_installed", False))
        or parent_N is None
    ):
        return None
    metric_blocks = getattr(parent_N, "factorized_metric_blocks", None)
    metric_route_values = getattr(parent_N, "factorized_metric_routes", None)
    metric_compiled = getattr(parent_N, "compiled_factorized_terms", None)
    direct_metric_key = getattr(metric_compiled, "_cpp_metric_key", None)
    direct_metric = (
        direct_metric_key is not None
        and getattr(metric_compiled, "su2_moving_environment", None) is owner
        and bool(
            getattr(
                metric_compiled,
                "_cpp_metric_routes_installed",
                False,
            )
        )
    )
    if metric_route_values is None:
        route_builder = getattr(
            metric_compiled,
            "factorized_metric_routes",
            None,
        )
        if route_builder is not None:
            metric_route_values = route_builder()
    basis = getattr(parent_H, "basis", None)
    if basis is None:
        return None
    if direct_metric:
        metric_routes = None
    elif metric_blocks is not None:
        metric_routes = []
        for entry in basis:
            factors = metric_blocks.get(entry.key)
            if factors is None:
                return None
            metric_routes.append((entry, entry, factors[0], factors[1]))
    elif metric_route_values is not None:
        metric_routes = [
            (in_entry, out_entry, left, right)
            for (
                _in_idx,
                _out_idx,
                in_entry,
                out_entry,
                left,
                right,
            ) in metric_route_values
        ]
    else:
        return None
    if not direct_metric and not metric_routes:
        return None

    from . import _su2_kernel

    metric_setup_started = time.perf_counter() if profile else None
    metric_dimension = (
        int(getattr(H, "coupled_parent_dimension"))
        if projection_blocks is not None
        else int(np.asarray(guess_packed).size)
    )
    if direct_metric:
        metric_key = str(direct_metric_key)
    else:
        topology_values = [
            np.asarray(
                [
                    int(getattr(basis, "size", len(guess_packed))),
                    len(metric_routes),
                ],
                dtype=np.int64,
            )
        ]
        numeric_values = []
        for in_entry, out_entry, left, right in metric_routes:
            topology_values.extend(
                (
                    np.asarray(
                        [int(in_entry.offset), int(out_entry.offset)],
                        dtype=np.int64,
                    ),
                    np.asarray(in_entry.shape, dtype=np.int64),
                    np.asarray(out_entry.shape, dtype=np.int64),
                    np.asarray(np.asarray(left).shape, dtype=np.int64),
                    np.asarray(np.asarray(right).shape, dtype=np.int64),
                )
            )
            numeric_values.extend((np.asarray(left), np.asarray(right)))
        topology_revision = _su2_kernel._cpp_array_revision(
            *topology_values
        )
        numeric_revision = _su2_kernel._cpp_array_revision(*numeric_values)
        metric_key = (
            f"metric:{factor_route_key}:{int(topology_revision)}"
        )
        owner.install_factorized_metric(
            metric_key,
            tuple(metric_routes),
            metric_dimension,
            int(topology_revision),
            int(numeric_revision),
        )
    metric_setup_elapsed = (
        time.perf_counter() - metric_setup_started
        if profile
        else None
    )
    active_canonical_solve = getattr(
        owner,
        "solve_active_bond_canonical",
        None,
    )
    if (
        projection_blocks is None
        and direct_metric
        and getattr(compiled, "cpp_owned_basis_topology", False)
        and callable(active_canonical_solve)
        and owner.active_bond_complementary_action_ready(
            factor_route_key,
            metric_dimension,
        )
    ):
        started = time.perf_counter() if profile else None
        result = active_canonical_solve(
            metric_key,
            projection_tolerance=max(float(lindep), 1.0e-12),
            max_component_elements=(
                _CPP_CANONICAL_METRIC_MAX_COMPONENT_ELEMENTS
            ),
            max_transform_elements=(
                _CPP_CANONICAL_METRIC_MAX_TRANSFORM_ELEMENTS
            ),
            davidson_tolerance=float(tol),
            max_iterations=int(itermax),
            max_space=max_space,
            workspace_budget_bytes=int(
                _PACKED_DAVIDSON_BASIS_MAX_BYTES
            ),
            workspace_basis_arrays=int(
                _PACKED_DAVIDSON_OWNED_BASIS_ARRAYS
            ),
            accept_unconverged=True,
        )
        elapsed = (
            time.perf_counter() - started
            if profile
            else None
        )
        if bool(result.get("compatible", False)):
            if not bool(result.get("accepted", False)):
                raise RuntimeError(
                    "The C++ canonical active-bond solve was not accepted."
                )
            orthonormal_dimension = int(
                result["orthonormal_dimension"]
            )
            requested_max_space = int(result["requested_max_space"])
            workspace_max_space = int(result["workspace_max_space"])
            info = {
                "metric": float(result["residual_norm"]),
                "residual": float(result["residual_norm"]),
                "davidson_iterations": int(result["iterations"]),
                "davidson_converged": bool(result["converged"]),
                "subspace_dim": int(result["basis_size"]),
                "generalized_norm": False,
                "tensor_davidson": True,
                "reduced_krylov": False,
                "packed_krylov": True,
                "metric_orthonormal_krylov": True,
                "canonical_reduced_basis": True,
                "projected_problem": "canonical_reduced_standard",
                "preconditioner_mode": "projected_packed_diagonal",
                "reduced_preconditioner": False,
                "restarts": int(result["restarts"]),
                "packed_dimension": int(metric_dimension),
                "orthonormalized_dim": orthonormal_dimension,
                "requested_max_space": requested_max_space,
                "workspace_max_space": workspace_max_space,
                "workspace_budget_bytes": int(
                    _PACKED_DAVIDSON_BASIS_MAX_BYTES
                ),
                "estimated_basis_workspace_bytes": int(
                    result["estimated_basis_workspace_bytes"]
                ),
                "workspace_limited": bool(
                    workspace_max_space < requested_max_space
                ),
                "cpp_davidson": True,
                "cpp_davidson_kind": str(result.get("kind")),
                "cpp_workspace_reused": bool(
                    result.get("workspace_reused", False)
                ),
                "matvec_count": int(result.get("matvec_calls", 0)),
                "norm_matvec_count": 1,
                "no_python_bond_callbacks": True,
                "coupled_projection_in_cpp": True,
                "direct_cpp_metric": True,
                "direct_complementary_action_executor": True,
                "cpp_active_solution_owned": True,
                "cpp_owned_merged_guess": True,
                "canonical_projection_reused": bool(
                    result.get("projection_reused", False)
                ),
                "canonical_projection_components": int(
                    result.get("projection_components", 0)
                ),
                "canonical_projection_max_component_dimension": int(
                    result.get(
                        "projection_max_component_dimension",
                        0,
                    )
                ),
                "canonical_projection_transform_elements": int(
                    result.get("projection_transform_elements", 0)
                ),
                "canonical_projection_whitening_residual": float(
                    result.get(
                        "projection_whitening_residual",
                        0.0,
                    )
                ),
            }
            if profile:
                projection_seconds = float(
                    result.get("projection_build_seconds", 0.0)
                )
                solve_seconds = float(
                    result.get("solve_seconds", elapsed)
                )
                info["solver_timing"] = {
                    "davidson": max(
                        0.0,
                        solve_seconds - projection_seconds,
                    ),
                    "matvec": max(
                        0.0,
                        solve_seconds - projection_seconds,
                    ),
                    "metric_setup": float(metric_setup_elapsed),
                    "canonical_projection": projection_seconds,
                    "projected": 0.0,
                    "precondition": 0.0,
                    "orthogonalize": 0.0,
                    "restart": 0.0,
                    "basis_update": 0.0,
                    "final_reference": 0.0,
                }
            return float(result["energy"]), None, info

    canonical_projection = None
    if (
        projection_blocks is None
        and direct_metric
        and getattr(compiled, "cpp_owned_basis_topology", False)
        and callable(
            getattr(
                owner,
                "prepare_canonical_reduced_projection",
                None,
            )
        )
        and callable(
            getattr(
                owner,
                "canonical_reduced_projection_guess",
                None,
            )
        )
        and callable(
            getattr(
                owner,
                "lift_factor_route_projection_vector",
                None,
            )
        )
        and callable(
            getattr(
                owner,
                "active_bond_complementary_davidson",
                None,
            )
        )
        and owner.active_bond_complementary_action_ready(
            factor_route_key,
            metric_dimension,
        )
    ):
        canonical_projection = (
            owner.prepare_canonical_reduced_projection(
                metric_key,
                tolerance=max(float(lindep), 1.0e-12),
                max_component_elements=(
                    _CPP_CANONICAL_METRIC_MAX_COMPONENT_ELEMENTS
                ),
                max_transform_elements=(
                    _CPP_CANONICAL_METRIC_MAX_TRANSFORM_ELEMENTS
                ),
            )
        )
        if bool(canonical_projection.get("compatible", False)):
            projection_key = str(
                canonical_projection["projection_key"]
            )
            orthonormal_dimension = int(
                canonical_projection["orthonormal_dimension"]
            )
            canonical_guess = (
                owner.canonical_reduced_projection_guess(
                    projection_key,
                    metric_key,
                    np.asarray(guess_packed, dtype=complex),
                    orthonormal_dimension,
                )
            )
            requested_max_space = (
                min(orthonormal_dimension, 48)
                if max_space is None
                else min(orthonormal_dimension, int(max_space))
            )
            basis_column_bytes = (
                orthonormal_dimension
                * np.dtype(np.complex128).itemsize
                * int(_PACKED_DAVIDSON_OWNED_BASIS_ARRAYS)
            )
            budget_columns = (
                int(_PACKED_DAVIDSON_BASIS_MAX_BYTES)
                // max(1, basis_column_bytes)
            )
            workspace_max_space = max(
                min(orthonormal_dimension, 2),
                min(
                    requested_max_space,
                    max(1, int(budget_columns)),
                ),
            )
            started = time.perf_counter() if profile else None
            result = owner.active_bond_complementary_davidson(
                projection_key,
                np.asarray(canonical_guess, dtype=complex),
                float(tol),
                int(itermax),
                int(workspace_max_space),
                True,
            )
            elapsed = (
                time.perf_counter() - started
                if profile
                else None
            )
            if not bool(result.get("accepted", False)):
                raise RuntimeError(
                    "The C++ canonical reduced Davidson solve was not accepted."
                )
            parent_vector = owner.lift_factor_route_projection_vector(
                projection_key,
                np.asarray(result["vector"], dtype=complex),
                metric_dimension,
            )
            parent_vector = _canonicalize_eigenvector(
                np.asarray(parent_vector),
                reference=np.asarray(guess_packed),
            )
            info = {
                "metric": float(result["residual_norm"]),
                "residual": float(result["residual_norm"]),
                "davidson_iterations": int(result["iterations"]),
                "davidson_converged": bool(result["converged"]),
                "subspace_dim": int(result["basis_size"]),
                "generalized_norm": False,
                "tensor_davidson": True,
                "reduced_krylov": False,
                "packed_krylov": True,
                "metric_orthonormal_krylov": True,
                "canonical_reduced_basis": True,
                "projected_problem": "canonical_reduced_standard",
                "preconditioner_mode": "projected_packed_diagonal",
                "reduced_preconditioner": False,
                "restarts": int(result["restarts"]),
                "packed_dimension": int(metric_dimension),
                "orthonormalized_dim": int(orthonormal_dimension),
                "requested_max_space": int(requested_max_space),
                "workspace_max_space": int(workspace_max_space),
                "workspace_budget_bytes": int(
                    _PACKED_DAVIDSON_BASIS_MAX_BYTES
                ),
                "estimated_basis_workspace_bytes": int(
                    basis_column_bytes * int(workspace_max_space)
                ),
                "workspace_limited": bool(
                    int(workspace_max_space)
                    < int(requested_max_space)
                ),
                "cpp_davidson": True,
                "cpp_davidson_kind": str(result.get("kind")),
                "cpp_workspace_reused": bool(
                    result.get("workspace_reused", False)
                ),
                "matvec_count": int(result.get("matvec_calls", 0)),
                "norm_matvec_count": 1,
                "no_python_bond_callbacks": True,
                "coupled_projection_in_cpp": True,
                "direct_cpp_metric": True,
                "direct_complementary_action_executor": True,
                "canonical_projection_reused": bool(
                    canonical_projection.get("reused", False)
                ),
                "canonical_projection_components": int(
                    canonical_projection.get("components", 0)
                ),
                "canonical_projection_max_component_dimension": int(
                    canonical_projection.get(
                        "max_component_dimension",
                        0,
                    )
                ),
                "canonical_projection_transform_elements": int(
                    canonical_projection.get("transform_elements", 0)
                ),
                "canonical_projection_whitening_residual": float(
                    canonical_projection.get(
                        "whitening_residual",
                        0.0,
                    )
                ),
            }
            if profile:
                info["solver_timing"] = {
                    "davidson": float(elapsed),
                    "matvec": float(elapsed),
                    "metric_setup": float(metric_setup_elapsed),
                    "canonical_projection": float(
                        canonical_projection.get(
                            "build_seconds",
                            0.0,
                        )
                    ),
                    "projected": 0.0,
                    "precondition": 0.0,
                    "orthogonalize": 0.0,
                    "restart": 0.0,
                    "basis_update": 0.0,
                    "final_reference": 0.0,
                }
            return float(result["energy"]), parent_vector, info

    projection_key = None
    if projection_blocks is not None:
        projection_topology_values = [
            np.asarray(
                [
                    metric_dimension,
                    int(np.asarray(guess_packed).size),
                    len(projection_blocks),
                ],
                dtype=np.int64,
            )
        ]
        projection_numeric_values = []
        for block in projection_blocks:
            if isinstance(block, _KroneckerBasisTransformBlock):
                row_slice = block.row_slice
                orthonormal_indices = block.orthonormal_indices
                transform = block.local_transform
                shape_metadata = np.asarray(
                    [
                        block.left_dim,
                        block.selected_dim,
                        block.local_dim,
                        block.right_dim,
                    ],
                    dtype=np.int64,
                )
            else:
                row_slice, orthonormal_indices, transform = block
                shape_metadata = np.asarray(
                    np.asarray(transform).shape,
                    dtype=np.int64,
                )
            projection_topology_values.extend(
                (
                    np.asarray(
                        [int(row_slice.start), int(row_slice.stop)],
                        dtype=np.int64,
                    ),
                    np.asarray(orthonormal_indices, dtype=np.int64),
                    shape_metadata,
                )
            )
            projection_numeric_values.append(np.asarray(transform))
        projection_topology_revision = _su2_kernel._cpp_array_revision(
            *projection_topology_values
        )
        projection_numeric_revision = _su2_kernel._cpp_array_revision(
            *projection_numeric_values
        )
        projection_key = (
            f"coupling:{factor_route_key}:"
            f"{int(projection_topology_revision)}"
        )
        owner.install_indexed_factor_route_projection(
            projection_key,
            factor_route_key,
            tuple(projection_blocks),
            metric_dimension,
            int(np.asarray(guess_packed).size),
            int(projection_topology_revision),
            int(projection_numeric_revision),
        )

    requested_max_space = (
        min(np.asarray(guess_packed).size, 48)
        if max_space is None
        else min(np.asarray(guess_packed).size, int(max_space))
    )
    basis_column_bytes = (
        int(np.asarray(guess_packed).size)
        * np.dtype(np.complex128).itemsize
        * int(_PACKED_DAVIDSON_OWNED_BASIS_ARRAYS)
    )
    budget_columns = (
        int(_PACKED_DAVIDSON_BASIS_MAX_BYTES) // max(1, basis_column_bytes)
    )
    minimum_columns = min(int(np.asarray(guess_packed).size), 2)
    workspace_max_space = max(
        minimum_columns,
        min(int(requested_max_space), max(1, int(budget_columns))),
    )
    residual_tolerance = (
        np.sqrt(tol)
        if tol_residual is None
        else float(tol_residual)
    )
    started = time.perf_counter() if profile else None
    active_solve = getattr(
        owner,
        "active_bond_complementary_generalized_davidson",
        None,
    )
    active_key = (
        factor_route_key if projection_key is None else projection_key
    )
    active_ready = getattr(
        owner,
        "active_bond_complementary_action_ready",
        None,
    )
    direct_active_solve = bool(
        direct_metric
        and callable(active_solve)
        and callable(active_ready)
        and getattr(compiled, "cpp_owned_basis_topology", False)
        and active_ready(
            active_key,
            int(np.asarray(guess_packed).size),
        )
    )
    if direct_active_solve:
        result = active_solve(
            active_key,
            metric_key,
            np.asarray(guess_packed, dtype=complex),
            float(tol),
            float(residual_tolerance),
            float(lindep),
            int(itermax),
            int(workspace_max_space),
            True,
        )
    else:
        solve = (
            owner.factor_route_generalized_davidson
            if projection_key is None
            else owner.factor_route_projected_generalized_davidson
        )
        result = solve(
            factor_route_key if projection_key is None else projection_key,
            metric_key,
            np.asarray(h_diag, dtype=complex),
            np.asarray(n_diag, dtype=complex),
            np.asarray(guess_packed, dtype=complex),
            float(tol),
            float(residual_tolerance),
            float(lindep),
            int(itermax),
            int(workspace_max_space),
            True,
        )
    elapsed = time.perf_counter() - started if profile else None
    if not bool(result.get("accepted", False)):
        raise RuntimeError("The C++ generalized Davidson solve was not accepted.")
    info = {
        "metric": float(result["residual_norm"]),
        "residual": float(result["residual_norm"]),
        "davidson_iterations": int(result["iterations"]),
        "davidson_converged": bool(result["converged"]),
        "subspace_dim": int(result["basis_size"]),
        "generalized_norm": True,
        "tensor_davidson": True,
        "reduced_krylov": False,
        "packed_krylov": True,
        "metric_orthonormal_krylov": True,
        "projected_problem": "standard",
        "preconditioner_mode": "packed_diagonal",
        "reduced_preconditioner": False,
        "restarts": int(result["restarts"]),
        "packed_dimension": int(np.asarray(guess_packed).size),
        "requested_max_space": int(requested_max_space),
        "workspace_max_space": int(workspace_max_space),
        "workspace_budget_bytes": int(_PACKED_DAVIDSON_BASIS_MAX_BYTES),
        "estimated_basis_workspace_bytes": int(
            basis_column_bytes * int(workspace_max_space)
        ),
        "workspace_limited": bool(
            int(workspace_max_space) < int(requested_max_space)
        ),
        "cpp_davidson": True,
        "cpp_davidson_kind": str(result.get("kind")),
        "cpp_workspace_reused": bool(result.get("workspace_reused", False)),
        "matvec_count": int(result.get("matvec_calls", 0)),
        "norm_matvec_count": int(result.get("norm_matvec_calls", 0)),
        "no_python_bond_callbacks": True,
        "coupled_projection_in_cpp": bool(projection_key is not None),
        "direct_cpp_metric": bool(direct_metric),
        "direct_complementary_action_executor": bool(direct_active_solve),
    }
    if profile:
        info["solver_timing"] = {
            "davidson": float(elapsed),
            "matvec": float(elapsed),
            "metric_setup": float(metric_setup_elapsed),
            "projected": 0.0,
            "precondition": 0.0,
            "orthogonalize": 0.0,
            "restart": 0.0,
            "basis_update": 0.0,
            "final_reference": 0.0,
        }
    vector = _canonicalize_eigenvector(
        np.asarray(result["vector"]),
        reference=np.asarray(guess_packed),
    )
    return float(result["energy"]), vector, info


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
    use_block_preconditioner=True,
    profile=False,
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

    state_layout = _reduced_state_layout_for_operator(op, template, layout)
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

    H_direct_packed = op.packed_matvec or op.aux_packed_matvec
    N_direct_packed = (
        None
        if norm_op is None
        else norm_op.packed_matvec or norm_op.aux_packed_matvec
    )
    if H_direct_packed is not None and (norm_op is None or N_direct_packed is not None):
        H_packed = H_direct_packed
        N_packed = N_direct_packed
        block_preconditioner = None
        packed_block_matrices = (
            op.packed_block_matrices
            if op.packed_block_matrices is not None
            else getattr(H_packed, "block_matrices", None)
        )
        norm_packed_block_matrices = (
            None
            if norm_op is None
            else (
                norm_op.packed_block_matrices
                if norm_op.packed_block_matrices is not None
                else getattr(N_packed, "block_matrices", None)
            )
        )
        cpp_generalized = _solve_cpp_factor_route_generalized_davidson(
            guess_packed,
            H_packed,
            N_packed,
            h_diag=h_diag,
            n_diag=n_diag,
            tol=tol,
            itermax=itermax,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            profile=profile,
        )
        if cpp_generalized is not None:
            theta, vec_packed, objective = cpp_generalized
            if (
                vec_packed is None
                and objective.get("cpp_active_solution_owned", False)
            ):
                optimized = template
            else:
                optimized = _unpack_tensor_state(
                    vec_packed,
                    template,
                    layout=layout,
                )
            objective["energy"] = float(theta)
            objective["operator_representation"] = "reduced"
            objective["packed_matvec_backend"] = getattr(
                H_packed,
                "backend",
                None,
            )
            objective["packed_matvec_source"] = (
                "primary" if op.packed_matvec is H_packed else "auxiliary"
            )
            objective["block_preconditioner"] = False
            objective["block_preconditioner_blocks"] = 0
            objective["norm_operator_representation"] = "reduced"
            objective["norm_packed_matvec_backend"] = getattr(
                N_packed,
                "backend",
                None,
            )
            objective["norm_packed_matvec_source"] = (
                "primary"
                if norm_op.packed_matvec is N_packed
                else "auxiliary"
            )
            return optimized, objective
        if norm_op is None:
            cpp_standard = _solve_cpp_factor_route_davidson(
                guess_packed,
                H_packed,
                h_diag=h_diag,
                tol=tol,
                itermax=itermax,
                max_space=max_space,
                profile=profile,
            )
            if cpp_standard is not None:
                theta, vec_packed, objective = cpp_standard
                optimized = _unpack_tensor_state(
                    vec_packed,
                    template,
                    layout=layout,
                )
                objective["energy"] = float(theta)
                objective["operator_representation"] = "reduced"
                objective["packed_matvec_backend"] = getattr(
                    H_packed,
                    "backend",
                    None,
                )
                objective["packed_matvec_source"] = (
                    getattr(H_packed, "source", None)
                    or getattr(H_packed, "name", None)
                )
                return optimized, objective
        use_auxiliary_packed = op.packed_matvec is not H_packed
        if (
            precond is None
            and packed_block_matrices is not None
            and not use_auxiliary_packed
            and (norm_op is None or norm_packed_block_matrices is not None)
        ):
            block_preconditioner = PackedBlockPreconditioner.from_layout_blocks(
                layout,
                packed_block_matrices,
                n_blocks=norm_packed_block_matrices,
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
            use_block_preconditioner=use_block_preconditioner,
            profile=profile,
        )
        optimized = _unpack_tensor_state(vec_packed, template, layout=layout)
        objective["energy"] = float(theta)
        objective["operator_representation"] = "reduced"
        objective["packed_matvec_backend"] = getattr(H_packed, "backend", None)
        objective["packed_matvec_source"] = (
            "primary" if op.packed_matvec is H_packed else "auxiliary"
        )
        objective["block_preconditioner"] = block_preconditioner is not None
        objective["block_preconditioner_blocks"] = (
            int(block_preconditioner.materialized_block_count)
            if block_preconditioner is not None
            else 0
        )
        if "solver_timing" in objective:
            objective["solver_timing"] = objective["solver_timing"]
        if "matvec_count" in objective:
            objective["matvec_count"] = objective["matvec_count"]
        if norm_op is not None:
            objective["norm_operator_representation"] = "reduced"
            objective["norm_packed_matvec_backend"] = getattr(N_packed, "backend", None)
            objective["norm_packed_matvec_source"] = (
                "primary" if norm_op.packed_matvec is N_packed else "auxiliary"
            )
        return optimized, objective

    H = _reduced_operator_to_matvec(op, template, state_layout)
    N = (
        _reduced_operator_to_matvec(norm_op, template, state_layout)
        if norm_op is not None
        else None
    )
    def H_packed(vector):
        state = state_layout.from_packed(vector)
        return H(state).to_packed(dtype=complex)

    def N_packed(vector):
        state = state_layout.from_packed(vector)
        return N(state).to_packed(dtype=complex)

    theta, vec_packed, objective = _solve_packed_generalized_davidson(
        guess_state.to_packed(dtype=complex),
        H_packed,
        h_diag=h_diag,
        N=N_packed if N is not None else None,
        n_diag=n_diag,
        tol=tol,
        itermax=itermax,
        max_space=max_space,
        tol_residual=tol_residual,
        lindep=lindep,
        precond=precond,
        use_block_preconditioner=use_block_preconditioner,
        profile=profile,
    )
    optimized = _reduced_state_to_tensor(state_layout.from_packed(vec_packed), template)
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
    uncoupled_state_layout = _reduced_state_layout_for_tensor(two_site, uncoupled_layout)

    def apply(coupled_tensor):
        uncoupled_tensor = _uncouple_two_site_tensor(coupled_tensor)
        if op.tensor_matvec is not None:
            out_uncoupled = op.tensor_matvec(uncoupled_tensor)
            if not isinstance(out_uncoupled, IrrepTensor):
                raise TypeError("tensor_matvec must return an IrrepTensor.")
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
    uncoupled_state_layout = _reduced_state_layout_for_tensor(two_site, uncoupled_layout)
    if transform is None:
        _, _, transform = _build_basis_transform(
            two_site,
            coupled=coupled_template,
            coupled_layout=coupled_layout,
            uncoupled_layout=uncoupled_layout,
        )
    if transform_dag is None and not isinstance(
        transform,
        _StructuredBasisTransform,
    ):
        transform_dag = np.asarray(transform).conj().T
    struct_cache = two_site.metadata.get("_basis_transform_struct_cache")
    use_struct = (
        struct_cache is not None
        and struct_cache.get("coupled_layout") == tuple(coupled_layout)
        and struct_cache.get("uncoupled_layout") == tuple(uncoupled_layout)
    )
    transform_blocks = struct_cache.get("blocks") if use_struct else None
    use_dense_transform = (
        not isinstance(transform, _StructuredBasisTransform)
        and np.asarray(transform).size
        <= _BASIS_TRANSFORM_DENSE_MATVEC_SIZE
    )
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

    def _coupled_diagonal(uncoupled_diagonal):
        uncoupled_diagonal = np.asarray(
            uncoupled_diagonal,
            dtype=float,
        ).reshape(-1)
        if isinstance(transform, _StructuredBasisTransform):
            return np.real(transform.project_diagonal(uncoupled_diagonal))
        return np.real((np.abs(transform) ** 2).T @ uncoupled_diagonal)

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
            diag = _coupled_diagonal(op.diag)
        packed_apply.backend = f"coupled-{getattr(packed_callback, 'backend', 'packed')}"
        packed_apply.basis = coupled_layout
        packed_apply.uncoupled_packed_matvec = packed_callback
        packed_apply.uncoupled_basis = getattr(packed_callback, "basis", None)
        packed_apply.coupled_parent_dimension = int(
            uncoupled_state_layout.size
        )
        packed_apply.coupled_dimension = int(coupled_state_layout.size)
        if transform_blocks is not None:
            packed_apply.coupled_transform_blocks = tuple(transform_blocks)
        elif isinstance(transform, _StructuredBasisTransform):
            packed_apply.coupled_transform_blocks = tuple(transform.blocks)
        elif use_dense_transform:
            packed_apply.coupled_transform_blocks = (
                (
                    slice(0, int(uncoupled_state_layout.size)),
                    np.arange(
                        int(coupled_state_layout.size),
                        dtype=np.int64,
                    ),
                    np.asarray(dense_transform),
                ),
            )
        return LocalOperator(
            packed_matvec=packed_apply,
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
            diag = _coupled_diagonal(op.diag)
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

        if isinstance(transform, _StructuredBasisTransform):
            coupled_diag = np.empty(int(transform.coupled_size), dtype=float)
            for column in range(int(transform.coupled_size)):
                unit = np.zeros(int(transform.coupled_size), dtype=complex)
                unit[column] = 1.0
                uncoupled = transform @ unit
                coupled_diag[column] = float(
                    np.real(np.vdot(uncoupled, matrix @ uncoupled))
                )
        else:
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
            diag = _coupled_diagonal(op.diag)
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


def _generalized_eigh_all(H, N, *, tol=1e-12):
    """Return all roots of ``H x = e N x`` in the nonsingular metric subspace."""

    H = 0.5 * (np.asarray(H, dtype=complex) + np.asarray(H, dtype=complex).conj().T)
    N = 0.5 * (np.asarray(N, dtype=complex) + np.asarray(N, dtype=complex).conj().T)
    if scipy_linalg is not None:
        try:
            evals, evecs = scipy_linalg.eigh(H, N, check_finite=False)
        except Exception:
            evals = evecs = None
        else:
            return np.real(evals).astype(float), np.asarray(evecs, dtype=complex)
    s, U = np.linalg.eigh(N)
    keep = s > max(float(tol), 1.0e-12)
    if not np.any(keep):
        raise ValueError("Generalized norm operator is numerically singular.")
    X = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
    H_ortho = X.conj().T @ H @ X
    evals, evecs_ortho = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
    return np.real(evals).astype(float), X @ evecs_ortho


def _lowest_hermitian_projected_root(H, *, reference=None, tol=1e-12, subset=4):
    """
    Return the lowest eigenpair of a small Hermitian projected matrix.

    Only a few lowest roots are requested when SciPy is available.  The extra
    roots preserve the old degenerate-root reference projection behavior without
    paying for a full eigendecomposition on every Davidson iteration.
    """

    H = 0.5 * (np.asarray(H) + np.asarray(H).conj().T)
    dim = int(H.shape[0])
    if dim == 0:
        raise ValueError("Projected Hermitian matrix must be nonempty.")
    if scipy_linalg is not None and dim > 1:
        nsubset = min(dim, max(1, int(subset)))
        evals, evecs = scipy_linalg.eigh(
            H,
            subset_by_index=(0, nsubset - 1),
            check_finite=False,
        )
    else:
        evals, evecs = np.linalg.eigh(H)
        order = np.argsort(np.real(evals))
        evals = evals[order]
        evecs = evecs[:, order]

    lowest = float(np.real(evals[0]))
    degenerate = [
        idx
        for idx, value in enumerate(evals)
        if abs(float(np.real(value)) - lowest) <= max(float(tol), 1.0e-12)
    ]
    if len(degenerate) > 1 and reference is not None:
        ref = np.asarray(reference, dtype=complex).reshape(-1)
        subspace = evecs[:, degenerate]
        coeff = subspace @ (subspace.conj().T @ ref)
        norm = np.linalg.norm(coeff)
        if norm > 1.0e-15:
            return lowest, coeff / norm
    return lowest, evecs[:, 0]


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


def _solve_projected_dense_roots(H, N=None, *, nroots=1, projector_basis=None, tol=1e-12):
    """
    Solve a dense local root problem, optionally restricted to a projector basis.

    ``projector_basis`` is expressed in the parent packed basis.  This helper
    keeps the target-spin projected path honest for generalized local problems:
    the Hamiltonian and norm are both projected before diagonalization, then
    roots are lifted back to the parent basis.
    """

    if projector_basis is None:
        return _solve_generalized_dense_roots(H, N, nroots=nroots, tol=tol)
    P = _orthonormalize_columns_dense(np.asarray(projector_basis, dtype=complex))
    if P.shape[1] == 0:
        raise ValueError("Projected dense solve received an empty projector basis.")
    nroots = min(int(nroots), int(P.shape[1]))
    H = 0.5 * (np.asarray(H, dtype=complex) + np.asarray(H, dtype=complex).conj().T)
    H_projected = P.conj().T @ H @ P
    if N is None:
        N_projected = None
    else:
        N = 0.5 * (np.asarray(N, dtype=complex) + np.asarray(N, dtype=complex).conj().T)
        N_projected = P.conj().T @ N @ P
    energies, projected_roots, residuals = _solve_generalized_dense_roots(
        H_projected,
        N_projected,
        nroots=nroots,
        tol=tol,
    )
    parent_roots = [P @ np.asarray(root, dtype=complex).reshape(-1) for root in projected_roots]
    parent_residuals = []
    for energy, root in zip(energies, parent_roots):
        if N is None:
            parent_residuals.append(float(np.linalg.norm(H @ root - float(energy) * root)))
        else:
            parent_residuals.append(float(np.linalg.norm(H @ root - float(energy) * (N @ root))))
    return energies, parent_roots, parent_residuals


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
    use_block_preconditioner=True,
    allow_unconverged=False,
    profile=False,
):
    """Return low roots of a standard local problem without materializing H."""
    _ = profile
    op = _normalize_local_operator(operator)
    op_resolved, diag = _resolve_davidson_operator(op, template, layout)
    guess_arr = np.asarray(guess_vec, dtype=complex)
    dim = int(guess_arr.shape[0]) if guess_arr.ndim > 1 else int(guess_arr.reshape(-1).size)
    if diag is None:
        diag = np.zeros(dim, dtype=float)
        missing_diag = True
    else:
        diag = np.asarray(diag, dtype=float).reshape(dim)
        missing_diag = False
    guess_vec = guess_arr.reshape(dim, -1) if guess_arr.ndim > 1 else guess_arr.reshape(dim)
    block_preconditioner = None
    davidson_precond = precond
    if (
        davidson_precond is None
        and use_block_preconditioner
        and op.packed_block_matrices is not None
    ):
        block_preconditioner = PackedBlockPreconditioner.from_layout_blocks(
            layout,
            op.packed_block_matrices,
        )

        def davidson_precond(resid, theta, _vec):
            return block_preconditioner.apply(resid, theta)

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

        projected_precond = precond
        if projected_precond is None and block_preconditioner is not None:
            def projected_precond(resid, theta, _vec):
                full_resid = projector_basis @ np.asarray(resid, dtype=complex).reshape(proj_dim)
                full_corr = block_preconditioner.apply(full_resid, theta)
                return projector_basis.conj().T @ full_corr

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
                precond=projected_precond,
                guess=projected_guess,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                return_info=True,
                return_partial=allow_unconverged,
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
            "block_preconditioner": block_preconditioner is not None,
            "block_preconditioner_blocks": (
                int(block_preconditioner.materialized_block_count)
                if block_preconditioner is not None
                else 0
            ),
            "packed_matvec_backend": getattr(op.packed_matvec, "backend", None),
        }
    energies, vecs, info = davidson(
        op_resolved,
        int(nroots),
        tol=tol,
        itermax=itermax,
        diag=diag,
        precond=davidson_precond,
        guess=guess_vec,
        max_space=max_space,
        tol_residual=tol_residual,
        lindep=lindep,
        return_info=True,
        return_partial=allow_unconverged,
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
        "block_preconditioner": block_preconditioner is not None,
        "block_preconditioner_blocks": (
            int(block_preconditioner.materialized_block_count)
            if block_preconditioner is not None
            else 0
        ),
        "packed_matvec_backend": getattr(op.packed_matvec, "backend", None),
    }


def _target_projector_basis(
    target_operator,
    template,
    layout,
    *,
    norm_operator=None,
    target_value,
    target_tol,
    min_dim,
    target_dim=None,
    max_dim=65536,
    dense_dim=256,
    itermax=80,
    max_space=None,
):
    op_resolved, _ = _resolve_davidson_operator(target_operator, template, layout)
    norm_resolved = None
    if norm_operator is not None:
        norm_resolved, _ = _resolve_davidson_operator(norm_operator, template, layout)
    dim = sum(entry.size for entry in layout)
    if dim > int(max_dim):
        return None, None
    if dim > int(dense_dim):
        if norm_resolved is not None:
            return None, None
        if target_dim is None:
            return None, None
        target = float(target_value)

        def _apply_target(vec):
            out = (
                op_resolved(vec)
                if callable(op_resolved)
                else op_resolved @ np.asarray(vec, dtype=complex).reshape(dim)
            )
            return np.asarray(out, dtype=complex).reshape(dim) - target * np.asarray(vec, dtype=complex).reshape(dim)

        def _shifted_square(vec):
            return _apply_target(_apply_target(vec))

        diag = None
        if not callable(op_resolved):
            diag = np.real(np.diag(np.asarray(op_resolved, dtype=complex)))
        if diag is None:
            diag = np.ones(dim, dtype=float)
        else:
            diag = (np.asarray(diag, dtype=float).reshape(dim) - target) ** 2
            diag = np.where(diag > 1.0e-14, diag, 1.0e-14)
        requested_dim = int(min_dim) if target_dim is None else max(int(min_dim), int(target_dim))
        nroots = min(dim, max(requested_dim, 1))
        try:
            evals, vecs, info = davidson(
                _shifted_square,
                nroots,
                tol=min(1.0e-10, max(float(target_tol) ** 2, 1.0e-14)),
                itermax=int(itermax),
                diag=diag,
                max_space=max_space or min(dim, max(64, 16 * nroots)),
                tol_residual=max(float(target_tol) ** 2, 1.0e-12),
                return_info=True,
            )
        except RuntimeError:
            return None, None
        vectors = [
            np.asarray(vecs[:, i], dtype=complex).reshape(dim)
            for i in range(int(nroots))
        ]
        target_values = []
        for vec in vectors:
            out = (
                op_resolved(vec)
                if callable(op_resolved)
                else op_resolved @ vec
            )
            denom = np.vdot(vec, vec)
            if abs(denom) <= 1.0e-15:
                target_values.append(np.inf)
            else:
                target_values.append(float(np.real(np.vdot(vec, out) / denom)))
        distance = np.abs(np.asarray(target_values, dtype=float) - target)
        keep = np.where(distance <= float(target_tol))[0]
        if keep.size < int(min_dim):
            return None, None
        basis = _orthonormalize_columns_dense(
            np.column_stack([vectors[int(idx)] for idx in keep])
        )
        if basis.shape[1] < int(min_dim):
            return None, None
        return basis, np.asarray([target_values[int(idx)] for idx in keep], dtype=float)
    matrix = (
        np.asarray(op_resolved, dtype=complex)
        if isinstance(op_resolved, np.ndarray)
        else _materialize_local_matrix(op_resolved, dim)
    )
    matrix = 0.5 * (matrix + matrix.conj().T)
    if norm_resolved is None:
        evals, evecs = np.linalg.eigh(matrix)
    else:
        norm_matrix = (
            np.asarray(norm_resolved, dtype=complex)
            if isinstance(norm_resolved, np.ndarray)
            else _materialize_local_matrix(norm_resolved, dim)
        )
        norm_matrix = 0.5 * (norm_matrix + norm_matrix.conj().T)
        evals, evecs = _generalized_eigh_all(matrix, norm_matrix, tol=max(float(target_tol), 1.0e-12))
    distance = np.abs(np.real(evals) - float(target_value))
    keep = np.where(distance <= float(target_tol))[0]
    if keep.size < int(min_dim):
        return None, None
    return _orthonormalize_columns_dense(evecs[:, keep]), np.real(evals[keep]).astype(float)


def _target_projector_basis_by_blocks(
    target_operator,
    template,
    layout,
    *,
    norm_operator=None,
    target_value,
    target_tol,
    min_dim,
    max_block_size=512,
    max_columns=512,
    offdiag_tol=1.0e-9,
):
    """
    Build an exact target-sector basis from packed self-sector blocks.

    :param target_operator: Local target operator, normally local ``S^2``.
    :param template: Tensor template used by the packed operator.
    :param layout: Packed local layout.
    :param target_value: Desired target-operator eigenvalue.
    :param target_tol: Eigenvalue tolerance used to keep target states.
    :param min_dim: Minimum number of projected vectors required.
    :param max_block_size: Largest packed sector block to diagonalize.
    :param max_columns: Maximum total basis columns to probe.
    :param offdiag_tol: Maximum allowed leakage outside each packed sector.
    :returns: ``(basis, eigenvalues)`` or ``(None, None)``.

    The routine is intentionally conservative: if the target operator couples
    different packed sectors, it refuses to construct a block projector. This
    keeps the local root selection exact in the current reduced basis instead
    of reintroducing the old truncated global ``S^2`` approximation.
    """

    entries = _layout_entries(layout)
    dim = _packed_layout_size(layout)
    if dim <= 0 or max_block_size is None:
        return None, None
    max_block_size = int(max_block_size)
    max_columns = int(max_columns)
    if max_block_size <= 0 or max_columns <= 0:
        return None, None
    if sum(entry.size for entry in entries) > max_columns:
        return None, None

    op_resolved, _ = _resolve_davidson_operator(target_operator, template, layout)
    norm_resolved = None
    if norm_operator is not None:
        norm_resolved, _ = _resolve_davidson_operator(norm_operator, template, layout)
    target = float(target_value)
    vectors = []
    target_values = []
    max_leakage = 0.0

    for entry in entries:
        if entry.size > max_block_size:
            return None, None
        block = np.zeros((entry.size, entry.size), dtype=complex)
        norm_block = None if norm_resolved is None else np.zeros((entry.size, entry.size), dtype=complex)
        for col in range(entry.size):
            basis_vec = np.zeros(dim, dtype=complex)
            basis_vec[entry.offset + col] = 1.0
            out = (
                op_resolved(basis_vec)
                if callable(op_resolved)
                else op_resolved @ basis_vec
            )
            out = np.asarray(out, dtype=complex).reshape(dim)
            sl = slice(entry.offset, entry.offset + entry.size)
            block[:, col] = out[sl]
            outside_norm = np.linalg.norm(out) ** 2 - np.linalg.norm(out[sl]) ** 2
            if outside_norm > 0.0:
                max_leakage = max(max_leakage, float(np.sqrt(outside_norm)))
            if norm_resolved is not None:
                nout = (
                    norm_resolved(basis_vec)
                    if callable(norm_resolved)
                    else norm_resolved @ basis_vec
                )
                nout = np.asarray(nout, dtype=complex).reshape(dim)
                norm_block[:, col] = nout[sl]
                noutside_norm = np.linalg.norm(nout) ** 2 - np.linalg.norm(nout[sl]) ** 2
                if noutside_norm > 0.0:
                    max_leakage = max(max_leakage, float(np.sqrt(noutside_norm)))
        if max_leakage > float(offdiag_tol):
            return None, None
        block = 0.5 * (block + block.conj().T)
        if norm_block is None:
            evals, evecs = np.linalg.eigh(block)
        else:
            norm_block = 0.5 * (norm_block + norm_block.conj().T)
            evals, evecs = _generalized_eigh_all(
                block,
                norm_block,
                tol=max(float(target_tol), 1.0e-12),
            )
        distance = np.abs(np.real(evals) - target)
        keep = np.where(distance <= float(target_tol))[0]
        for idx in keep:
            vec = np.zeros(dim, dtype=complex)
            vec[entry.offset:entry.offset + entry.size] = evecs[:, int(idx)]
            vectors.append(vec)
            target_values.append(float(np.real(evals[int(idx)])))

    if len(vectors) < int(min_dim):
        return None, None
    basis = _orthonormalize_columns_dense(np.column_stack(vectors))
    if basis.shape[1] < int(min_dim):
        return None, None
    return basis, np.asarray(target_values, dtype=float)


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


def _operator_expectations_with_metric_matvec(operator, template, layout, root_vecs, metric_matvec):
    """
    Evaluate local operator expectations with an external metric action.

    This is used by pre-orthonormalized local problems, where the norm is
    already owned by the reduced problem rather than represented as a
    :class:`LocalOperator`.
    """

    op_resolved, _ = _resolve_davidson_operator(operator, template, layout)
    values = []
    for vec in root_vecs:
        vec = np.asarray(vec, dtype=complex).reshape(-1)
        out = np.asarray(
            op_resolved(vec) if callable(op_resolved) else op_resolved @ vec,
            dtype=complex,
        ).reshape(-1)
        nvec = np.asarray(metric_matvec(vec), dtype=complex).reshape(-1)
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


def _match_selected_roots_to_guesses(
    energies,
    root_vecs,
    residuals,
    selected_roots,
    guess_matrix,
):
    """Keep state-averaged root identities aligned with the previous MPS roots."""

    if guess_matrix is None or len(root_vecs) <= 1:
        return energies, root_vecs, residuals, selected_roots, None
    guesses = np.asarray(guess_matrix, dtype=complex)
    if guesses.ndim == 1:
        guesses = guesses.reshape(-1, 1)
    if guesses.ndim != 2 or guesses.shape[1] == 0:
        return energies, root_vecs, residuals, selected_roots, None
    roots = [np.asarray(vec, dtype=complex).reshape(-1) for vec in root_vecs]
    if guesses.shape[0] != roots[0].size:
        return energies, root_vecs, residuals, selected_roots, None

    nmatch = min(len(roots), guesses.shape[1])
    overlap = np.zeros((nmatch, len(roots)), dtype=float)
    for guess_idx in range(nmatch):
        guess = guesses[:, guess_idx].reshape(-1)
        guess_norm = np.linalg.norm(guess)
        if guess_norm <= 1.0e-15:
            continue
        guess = guess / guess_norm
        for root_idx, root in enumerate(roots):
            root_norm = np.linalg.norm(root)
            if root_norm <= 1.0e-15:
                continue
            overlap[guess_idx, root_idx] = abs(np.vdot(guess, root / root_norm))

    assigned = []
    used = set()
    for guess_idx in range(nmatch):
        order = np.argsort(overlap[guess_idx])[::-1]
        for root_idx in order:
            root_idx = int(root_idx)
            if root_idx not in used:
                assigned.append(root_idx)
                used.add(root_idx)
                break
    assigned.extend(idx for idx in range(len(roots)) if idx not in used)
    return (
        np.asarray([energies[idx] for idx in assigned], dtype=float),
        [root_vecs[idx] for idx in assigned],
        [residuals[idx] for idx in assigned],
        [selected_roots[idx] for idx in assigned],
        overlap,
    )


def _can_match_all_selected_roots(weights, nroots, *, tol=1.0e-15):
    """
    Return whether overlap matching may reorder all selected roots.

    Multiroot SA solves often request extra zero-weight candidates.  Those
    buffer roots must stay behind the weighted roots after target/energy
    selection, otherwise the density matrix is built from the wrong states.
    """

    if weights is None:
        return True
    local_weights = np.asarray(weights, dtype=float).reshape(-1)[: int(nroots)]
    if local_weights.size < int(nroots):
        return False
    return bool(np.all(np.abs(local_weights) > float(tol)))


def _solve_orthonormalized_dense(H, N, *, tol, basis=None):
    """
    Solve a generalized local problem by orthonormalizing the metric first.

    This returns the same ground-state solution as the generalized solve, but
    makes the "canonical local basis" step explicit: first orthonormalize with
    respect to ``N``, then diagonalize the transformed Hermitian operator.
    """
    H = 0.5 * (np.asarray(H) + np.asarray(H).conj().T)
    N = 0.5 * (np.asarray(N) + np.asarray(N).conj().T)
    if basis is not None:
        orthonormalization = basis.metric_orthonormalization(N, tol=tol)
        H_ortho = orthonormalization.operator_to_orthonormal(H)
    else:
        s, U = np.linalg.eigh(N)
        keep = s > max(float(tol), 1e-12)
        if not np.any(keep):
            raise ValueError("Canonical local metric is numerically singular.")
        X = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
        orthonormalization = None
        H_ortho = X.conj().T @ H @ X
    evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
    if orthonormalization is not None:
        coeff = orthonormalization.from_orthonormal_vector(evecs[:, 0])
    else:
        coeff = X @ evecs[:, 0]
    norm = np.sqrt(np.real(np.vdot(coeff, N @ coeff)))
    if norm > 1e-15:
        coeff = coeff / norm
    resid = H @ coeff - evals[0] * (N @ coeff)
    return float(np.real(evals[0])), coeff, float(np.linalg.norm(resid))


def _solve_orthonormalized_operator_davidson(
    operator,
    norm_operator,
    template,
    layout,
    guess_vec,
    *,
    tol,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    allow_unconverged=False,
    profile=False,
):
    """
    Solve ``H x = E N x`` as a standard Davidson problem in ``N``-orthonormal
    local coordinates.

    The local metric is materialized once to build ``x = X y`` with
    ``X.conj().T @ N @ X = I``.  The Hamiltonian remains matrix-free through
    ``y -> X.conj().T @ H @ X @ y``.  This mirrors the block-DMRG convention
    of solving the local problem in an orthonormal renormalized basis while
    retaining PyQED's packed local operator implementation.

    :param operator: Local Hamiltonian operator.
    :param norm_operator: Local metric operator.
    :param template: Two-site tensor template.
    :param layout: Packed two-site basis layout.
    :param guess_vec: Initial packed vector in the original local basis.
    :param tol: Davidson convergence tolerance and metric cutoff.
    :returns: ``(energy, packed_vector, residual, info)``.
    """

    guess_vec = np.asarray(guess_vec, dtype=complex).reshape(-1)
    dim = int(guess_vec.size)
    timing = {
        "resolve": 0.0,
        "metric": 0.0,
        "davidson": 0.0,
        "matvec": 0.0,
        "residual": 0.0,
    } if profile else None
    total_t0 = time.perf_counter() if profile else None
    t0 = time.perf_counter() if profile else None
    H_resolved, h_diag = _resolve_davidson_operator(operator, template, layout)
    N_resolved, _ = _resolve_davidson_operator(norm_operator, template, layout)
    if profile:
        timing["resolve"] += time.perf_counter() - t0
        t0 = time.perf_counter()
    N_matrix = (
        np.asarray(N_resolved, dtype=complex)
        if isinstance(N_resolved, np.ndarray)
        else _materialize_local_matrix(N_resolved, dim)
    )
    basis = layout if isinstance(layout, TwoSiteBasis) else _two_site_basis_for_layout(template, layout)
    if basis is not None:
        orthonormalization = basis.metric_orthonormalization(N_matrix, tol=tol)
        X = np.asarray(orthonormalization.transform, dtype=complex)

        def to_orthonormal(vector):
            return orthonormalization.to_orthonormal_vector(vector)

        def from_orthonormal(vector):
            return orthonormalization.from_orthonormal_vector(vector)
    else:
        N_matrix = 0.5 * (N_matrix + N_matrix.conj().T)
        eigvals, eigvecs = np.linalg.eigh(N_matrix)
        keep = eigvals > max(float(tol), 1.0e-12)
        if not np.any(keep):
            raise ValueError("Canonical local metric is numerically singular.")
        X = eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))

        def to_orthonormal(vector):
            return X.conj().T @ (N_matrix @ np.asarray(vector, dtype=complex).reshape(dim))

        def from_orthonormal(vector):
            return X @ np.asarray(vector, dtype=complex).reshape(-1)

    ortho_dim = int(X.shape[1])
    if profile:
        timing["metric"] += time.perf_counter() - t0

    matvec_count = 0
    def H_full(vector):
        nonlocal matvec_count
        matvec_count += 1
        t0 = time.perf_counter() if profile else None
        vector = np.asarray(vector, dtype=complex).reshape(dim)
        out = (
            np.asarray(H_resolved @ vector, dtype=complex).reshape(dim)
            if isinstance(H_resolved, np.ndarray)
            else np.asarray(H_resolved(vector), dtype=complex).reshape(dim)
        )
        if profile:
            timing["matvec"] += time.perf_counter() - t0
        return out

    def H_orthonormal(y):
        return X.conj().T @ H_full(from_orthonormal(y))

    if h_diag is None:
        ortho_diag = np.zeros(ortho_dim, dtype=float)
        missing_diag = True
    else:
        h_diag = np.asarray(h_diag, dtype=float).reshape(dim)
        ortho_diag = np.real((np.abs(X) ** 2).T @ h_diag)
        missing_diag = False

    guess_y = to_orthonormal(guess_vec)
    if np.linalg.norm(guess_y) <= 1.0e-15:
        guess_y = None

    dense_fallback = False
    try:
        t0 = time.perf_counter() if profile else None
        energies, vecs, info = davidson(
            H_orthonormal,
            1,
            tol=tol,
            itermax=itermax,
            diag=ortho_diag,
            guess=guess_y,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            return_info=True,
            return_partial=allow_unconverged,
        )
        if profile:
            timing["davidson"] += time.perf_counter() - t0
    except (RuntimeError, ValueError, IndexError):
        t0 = time.perf_counter() if profile else None
        H_ortho = _materialize_local_matrix(H_orthonormal, ortho_dim)
        evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
        if profile:
            timing["davidson"] += time.perf_counter() - t0
        energies = np.asarray([np.real(evals[0])], dtype=float)
        vecs = np.asarray(evecs[:, :1], dtype=complex)
        info = {
            "iterations": 0,
            "converged": True,
            "subspace_dim": int(ortho_dim),
            "restarts": 0,
        }
        dense_fallback = True

    y = np.asarray(vecs[:, 0], dtype=complex).reshape(ortho_dim)
    vec = _canonicalize_eigenvector(from_orthonormal(y), reference=guess_vec)
    nvec = N_matrix @ vec
    norm = np.sqrt(max(0.0, float(np.real(np.vdot(vec, nvec)))))
    if norm > 1.0e-15:
        vec = vec / norm
        nvec = nvec / norm
    energy = float(np.real(np.asarray(energies).reshape(-1)[0]))
    t0 = time.perf_counter() if profile else None
    residual = float(np.linalg.norm(H_full(vec) - energy * nvec))
    if profile:
        timing["residual"] += time.perf_counter() - t0
        timing["total"] = time.perf_counter() - total_t0
    info_out = {
        "davidson_iterations": int(info.get("iterations", 0)),
        "davidson_converged": bool(info.get("converged", False)),
        "subspace_dim": int(info.get("subspace_dim", ortho_dim)),
        "restarts": int(info.get("restarts", 0)),
        "orthonormalized_dim": int(ortho_dim),
        "missing_diagonal_preconditioner": bool(missing_diag),
        "dense_fallback": bool(dense_fallback),
    }
    if profile:
        info_out["solver_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
        info_out["matvec_count"] = int(matvec_count)
    return energy, vec, residual, info_out


def _solve_orthonormalized_operator_davidson_roots(
    operator,
    norm_operator,
    template,
    layout,
    guess_matrix,
    *,
    nroots,
    projector_basis=None,
    tol,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    allow_unconverged=False,
    profile=False,
):
    """Multi-root standard Davidson in metric-orthonormal local coordinates."""

    guesses = np.asarray(guess_matrix, dtype=complex)
    if guesses.ndim == 1:
        guesses = guesses.reshape(-1, 1)
    dim = int(guesses.shape[0])
    nroots = int(nroots)
    timing = {
        "resolve": 0.0,
        "metric": 0.0,
        "davidson": 0.0,
        "matvec": 0.0,
        "residual": 0.0,
    } if profile else None
    total_t0 = time.perf_counter() if profile else None
    t0 = time.perf_counter() if profile else None
    H_resolved, h_diag = _resolve_davidson_operator(operator, template, layout)
    N_resolved, _ = _resolve_davidson_operator(norm_operator, template, layout)
    if profile:
        timing["resolve"] += time.perf_counter() - t0
        t0 = time.perf_counter()
    N_matrix = (
        np.asarray(N_resolved, dtype=complex)
        if isinstance(N_resolved, np.ndarray)
        else _materialize_local_matrix(N_resolved, dim)
    )
    basis = layout if isinstance(layout, TwoSiteBasis) else _two_site_basis_for_layout(template, layout)
    if basis is not None:
        orthonormalization = basis.metric_orthonormalization(N_matrix, tol=tol)
        X = np.asarray(orthonormalization.transform, dtype=complex)

        def to_orthonormal(vector):
            return orthonormalization.to_orthonormal_vector(vector)

        def from_orthonormal(vector):
            return orthonormalization.from_orthonormal_vector(vector)
    else:
        N_matrix = 0.5 * (N_matrix + N_matrix.conj().T)
        eigvals, eigvecs = np.linalg.eigh(N_matrix)
        keep = eigvals > max(float(tol), 1.0e-12)
        if not np.any(keep):
            raise ValueError("Canonical local metric is numerically singular.")
        X = eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))

        def to_orthonormal(vector):
            return X.conj().T @ (N_matrix @ np.asarray(vector, dtype=complex).reshape(dim))

        def from_orthonormal(vector):
            return X @ np.asarray(vector, dtype=complex).reshape(-1)

    ortho_dim = int(X.shape[1])
    if profile:
        timing["metric"] += time.perf_counter() - t0

    matvec_count = 0

    def H_full(vector):
        nonlocal matvec_count
        matvec_count += 1
        t0 = time.perf_counter() if profile else None
        vector = np.asarray(vector, dtype=complex).reshape(dim)
        out = (
            np.asarray(H_resolved @ vector, dtype=complex).reshape(dim)
            if isinstance(H_resolved, np.ndarray)
            else np.asarray(H_resolved(vector), dtype=complex).reshape(dim)
        )
        if profile:
            timing["matvec"] += time.perf_counter() - t0
        return out

    def H_orthonormal(y):
        return X.conj().T @ H_full(from_orthonormal(y))

    if h_diag is None:
        ortho_diag = np.zeros(ortho_dim, dtype=float)
        missing_diag = True
    else:
        h_diag = np.asarray(h_diag, dtype=float).reshape(dim)
        ortho_diag = np.real((np.abs(X) ** 2).T @ h_diag)
        missing_diag = False

    guess_cols = []
    for idx in range(guesses.shape[1]):
        y = np.asarray(to_orthonormal(guesses[:, idx]), dtype=complex).reshape(-1)
        if np.linalg.norm(y) > 1.0e-15:
            guess_cols.append(y)
    guess_y = np.column_stack(guess_cols) if guess_cols else None

    projector_y = None
    if projector_basis is not None:
        columns = []
        for idx in range(int(projector_basis.shape[1])):
            y = np.asarray(to_orthonormal(projector_basis[:, idx]), dtype=complex).reshape(-1)
            if np.linalg.norm(y) > 1.0e-15:
                columns.append(y)
        if columns:
            projector_y = _orthonormalize_columns_dense(np.column_stack(columns))
            nroots = min(nroots, int(projector_y.shape[1]))

    dense_fallback = False
    try:
        t0 = time.perf_counter() if profile else None
        energies, vecs, info = davidson(
            H_orthonormal,
            nroots,
            tol=tol,
            itermax=itermax,
            diag=ortho_diag,
            guess=guess_y,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            return_info=True,
            return_partial=allow_unconverged,
        ) if projector_y is None else davidson(
            lambda coeff: projector_y.conj().T @ H_orthonormal(projector_y @ coeff),
            nroots,
            tol=tol,
            itermax=itermax,
            diag=np.real((np.abs(projector_y) ** 2).T @ ortho_diag),
            guess=(
                projector_y.conj().T @ guess_y
                if guess_y is not None and guess_y.shape[0] == projector_y.shape[0]
                else None
            ),
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            return_info=True,
            return_partial=allow_unconverged,
        )
        if projector_y is not None:
            vecs = projector_y @ np.asarray(vecs, dtype=complex)
        if profile:
            timing["davidson"] += time.perf_counter() - t0
    except (RuntimeError, ValueError, IndexError):
        t0 = time.perf_counter() if profile else None
        H_ortho = _materialize_local_matrix(
            H_orthonormal if projector_y is None else (
                lambda coeff: projector_y.conj().T @ H_orthonormal(projector_y @ coeff)
            ),
            ortho_dim if projector_y is None else int(projector_y.shape[1]),
        )
        evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
        if profile:
            timing["davidson"] += time.perf_counter() - t0
        energies = np.asarray(evals[:nroots], dtype=float)
        vecs = np.asarray(evecs[:, :nroots], dtype=complex)
        if projector_y is not None:
            vecs = projector_y @ vecs
        info = {
            "iterations": 0,
            "converged": True,
            "subspace_dim": int(H_ortho.shape[0]),
            "restarts": 0,
        }
        dense_fallback = True

    root_vecs = []
    residuals = []
    t0 = time.perf_counter() if profile else None
    for root_idx in range(int(np.asarray(vecs).shape[1])):
        y = np.asarray(vecs[:, root_idx], dtype=complex).reshape(ortho_dim)
        reference = guesses[:, root_idx] if root_idx < guesses.shape[1] else guesses[:, 0]
        vec = _canonicalize_eigenvector(from_orthonormal(y), reference=reference)
        nvec = N_matrix @ vec
        norm = np.sqrt(max(0.0, float(np.real(np.vdot(vec, nvec)))))
        if norm > 1.0e-15:
            vec = vec / norm
            nvec = nvec / norm
        energy = float(np.real(np.asarray(energies).reshape(-1)[root_idx]))
        residuals.append(float(np.linalg.norm(H_full(vec) - energy * nvec)))
        root_vecs.append(vec)
    if profile:
        timing["residual"] += time.perf_counter() - t0
        timing["total"] = time.perf_counter() - total_t0
    info_out = {
        "davidson_iterations": int(info.get("iterations", 0)),
        "davidson_converged": bool(info.get("converged", False)),
        "subspace_dim": int(info.get("subspace_dim", ortho_dim)),
        "restarts": int(info.get("restarts", 0)),
        "orthonormalized_dim": int(ortho_dim),
        "missing_diagonal_preconditioner": bool(missing_diag),
        "dense_fallback": bool(dense_fallback),
    }
    if profile:
        info_out["solver_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
        info_out["matvec_count"] = int(matvec_count)
    return (
        np.asarray(energies[: len(root_vecs)], dtype=float),
        root_vecs,
        residuals,
        info_out,
    )


def _prefer_component_sparse_direct_table(op):
    """
    Return whether this local operator asks for component-direct projection.

    The SU(2) qchem block2-like route can supply packed factor schedules whose
    direct parent-block builder bypasses the generic sector block table.  This
    predicate keeps that preference explicit and still lets unsupported
    layouts fall back through the regular block-sparse builder.
    """

    H_packed = getattr(op, "packed_matvec", None) or getattr(
        op,
        "aux_packed_matvec",
        None,
    )
    H_table = getattr(op, "local_operator_table", None)
    compiled_factorized_terms = (
        getattr(H_table, "compiled_factorized_terms", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_factorized_terms", None)
    return bool(
        compiled_factorized_terms is not None
        and (
            getattr(
                compiled_factorized_terms,
                "prefer_direct_orthonormal_projection",
                False,
            )
            or getattr(
                compiled_factorized_terms,
                "prefer_recursive_operator_matvec",
                False,
            )
        )
    )


def build_orthonormalized_local_problem(
    operator,
    norm_operator,
    template,
    layout,
    *,
    tol=1.0e-12,
    max_dim=None,
    block_sparse=True,
    require_block_sparse_table=False,
    max_block_kernel_elements=None,
    name=None,
    source="renormalized_environment",
    cache_hit=False,
    profile=False,
    moving_environment_cache=None,
    su2_moving_environment=None,
    local_operator_key=None,
):
    """
    Store a local effective operator in an orthonormal reduced basis.

    This is the environment-level analogue of block2's local renormalized
    operator storage: the local metric is materialized once, the parent packed
    basis is orthonormalized, and the returned object owns the transformed
    standard Hamiltonian action.  The Hamiltonian itself remains matrix-free in
    the parent basis and is only wrapped by ``X^H H X``.

    :param operator: Local Hamiltonian operator in the parent packed basis.
    :param norm_operator: Local identity/norm operator in the same basis.
    :param template: Rank-4 two-site tensor defining the parent basis.
    :param layout: Parent packed two-site layout.
    :param tol: Metric eigenvalue cutoff.
    :param max_dim: Optional maximum parent dimension to transform.
    :param block_sparse: If True, first try to build a per-sector transformed
        operator table without global dense transforms.
    :param require_block_sparse_table: If True, raise when the per-sector table
        cannot be built instead of falling back to a global transformed
        operator.
    :param max_block_kernel_elements: Optional cap for materializing one
        sector-to-sector Hamiltonian block while building the table.
    :param name: Optional diagnostic label.
    :param source: Description of the renormalized-operator owner.
    :param cache_hit: Whether this object was returned from an environment
        operator-table cache.
    :param profile: Attach local operator-table construction timing metadata.
    :param moving_environment_cache: Optional sweep-persistent cache for
        structural moving-environment contraction plans.
    :param su2_moving_environment: Persistent C++ owner for local numerical
        tables and workspaces.
    :param local_operator_key: Stable sweep/bond key for the active C++ table.
    :returns: :class:`OrthonormalizedLocalProblem`, or ``None`` if ``max_dim``
        excludes the current local dimension.
    """

    op = _normalize_local_operator(operator)
    norm_op = _normalize_local_operator(norm_operator)
    metadata = getattr(op, "metadata", None)
    basis = (
        _operator_basis_for_layout(op, layout)
        or _operator_basis_for_layout(norm_op, layout)
        or _two_site_basis_for_layout(template, layout)
    )
    if basis is None:
        _, raw_layout = pack_two_site_state(template, layout=layout)
        basis = two_site_state_basis(template, layout=raw_layout)
    dim = int(basis.size)
    build_timing = {} if profile else None

    if max_dim is not None and dim > int(max_dim):
        return None

    if block_sparse:
        if _prefer_component_sparse_direct_table(op):
            t0 = time.perf_counter() if profile else None
            sparse_timing = {} if profile else None
            component_problem = _build_component_sparse_orthonormalized_local_problem(
                op,
                norm_op,
                basis,
                tol=tol,
                max_block_kernel_elements=max_block_kernel_elements,
                name=name,
                source=f"{source}:component_sparse_operator_table",
                cache_hit=cache_hit,
                metadata=metadata,
                timing=sparse_timing,
                moving_environment_cache=moving_environment_cache,
                su2_moving_environment=su2_moving_environment,
                local_operator_key=local_operator_key,
            )
            if profile:
                build_timing["component_sparse_preferred_table"] = (
                    time.perf_counter() - t0
                )
                for key, value in (sparse_timing or {}).items():
                    build_timing[key] = float(value)
            if component_problem is not None:
                if profile:
                    metadata = dict(component_problem.metadata or {})
                    existing_timing = dict(
                        metadata.get("renormalized_operator_build_timing") or {}
                    )
                    existing_timing.update(
                        {key: float(value) for key, value in build_timing.items()}
                    )
                    metadata["renormalized_operator_build_timing"] = existing_timing
                    component_problem = replace(component_problem, metadata=metadata)
                return component_problem
        t0 = time.perf_counter() if profile else None
        sparse_timing = {} if profile else None
        block_sparse_problem = _build_block_sparse_orthonormalized_local_problem(
            op,
            norm_op,
            template,
            basis,
            tol=tol,
            max_block_kernel_elements=max_block_kernel_elements,
            name=name,
            source=f"{source}:block_sparse_operator_table",
            cache_hit=cache_hit,
            metadata=metadata,
            timing=sparse_timing,
            moving_environment_cache=moving_environment_cache,
        )
        if profile:
            build_timing["block_sparse_table"] = time.perf_counter() - t0
            for key, value in (sparse_timing or {}).items():
                build_timing[key] = float(value)
        if block_sparse_problem is not None:
            if profile:
                metadata = dict(block_sparse_problem.metadata or {})
                existing_timing = dict(
                    metadata.get("renormalized_operator_build_timing") or {}
                )
                existing_timing.update({
                    key: float(value)
                    for key, value in build_timing.items()
                })
                metadata["renormalized_operator_build_timing"] = existing_timing
                block_sparse_problem = replace(block_sparse_problem, metadata=metadata)
            return block_sparse_problem
        if require_block_sparse_table:
            raise NotImplementedError(
                "A block-sparse orthonormalized renormalized-operator table "
                "could not be built for this local effective problem."
            )

    t0 = time.perf_counter() if profile else None
    H_resolved, h_diag = _resolve_davidson_operator(op, template, basis)
    N_resolved, _ = _resolve_davidson_operator(norm_op, template, basis)
    N_matrix = (
        np.asarray(N_resolved, dtype=complex)
        if isinstance(N_resolved, np.ndarray)
        else _materialize_local_matrix(N_resolved, dim)
    )
    if profile:
        build_timing["dense_metric"] = time.perf_counter() - t0
        t0 = time.perf_counter()
    orthonormalization = basis.metric_orthonormalization(N_matrix, tol=tol)
    X = np.asarray(orthonormalization.transform, dtype=complex)
    ortho_dim = int(X.shape[1])
    if profile:
        build_timing["dense_orthonormalize"] = time.perf_counter() - t0

    def H_full(vector):
        vector = np.asarray(vector, dtype=complex).reshape(dim)
        return (
            np.asarray(H_resolved @ vector, dtype=complex).reshape(dim)
            if isinstance(H_resolved, np.ndarray)
            else np.asarray(H_resolved(vector), dtype=complex).reshape(dim)
        )

    def H_orthonormal(y):
        y = np.asarray(y, dtype=complex).reshape(ortho_dim)
        return X.conj().T @ H_full(X @ y)

    if h_diag is None:
        diag = np.zeros(ortho_dim, dtype=float)
    else:
        h_diag = np.asarray(h_diag, dtype=float).reshape(dim)
        diag = np.real((np.abs(X) ** 2).T @ h_diag)

    if profile:
        metadata = dict(metadata or {})
        existing_timing = dict(
            metadata.get("renormalized_operator_build_timing") or {}
        )
        existing_timing.update({
            key: float(value)
            for key, value in build_timing.items()
        })
        metadata["renormalized_operator_build_timing"] = existing_timing
    return OrthonormalizedLocalProblem(
        basis=basis,
        transform=X,
        metric=np.asarray(N_matrix, dtype=complex),
        matvec=H_orthonormal,
        full_matvec=H_full,
        diag=diag,
        name=name,
        source=source,
        cache_hit=cache_hit,
        metadata=metadata,
    )


def _entry_self_metric_block_from_matvec(norm_op, entry, total_size, *, tol):
    """
    Materialize one exact metric self-block and reject cross-entry coupling.

    The block-sparse orthonormalized local problem is only valid when the local
    metric is block diagonal in the packed basis entries.  Provider-side
    diagonal blocks can be incomplete for larger renormalized environments, so
    this helper probes the actual packed metric action.
    """

    metric_action = getattr(norm_op, "packed_matvec", None) or getattr(norm_op, "aux_packed_matvec", None)
    if metric_action is None:
        return None
    size = int(entry.size)
    total_size = int(total_size)
    block = np.zeros((size, size), dtype=complex)
    off_norm_sq = 0.0
    full_norm_sq = 0.0
    for col in range(size):
        basis_vec = np.zeros(total_size, dtype=complex)
        basis_vec[int(entry.offset) + col] = 1.0
        out = np.asarray(metric_action(basis_vec), dtype=complex).reshape(total_size)
        block[:, col] = out[entry.slice]
        off = np.array(out, copy=True)
        off[entry.slice] = 0.0
        off_norm_sq += float(np.linalg.norm(off) ** 2)
        full_norm_sq += float(np.linalg.norm(out) ** 2)
    off_norm = np.sqrt(off_norm_sq)
    full_norm = np.sqrt(full_norm_sq)
    if off_norm > max(float(tol), 1.0e-10) * max(1.0, full_norm):
        return False
    return 0.5 * (block + block.conj().T)


def _entry_self_metric_block(norm_op, entry):
    if norm_op is None:
        return np.eye(entry.size, dtype=complex)
    provider = getattr(norm_op, "packed_block_matrices", None)
    if provider is None and getattr(norm_op, "packed_matvec", None) is not None:
        provider = getattr(norm_op.packed_matvec, "block_matrices", None)
    if provider is None and getattr(norm_op, "aux_packed_matvec", None) is not None:
        provider = getattr(norm_op.aux_packed_matvec, "block_matrices", None)
    if provider is None:
        return None
    if hasattr(provider, "block_matrix_for"):
        block = provider.block_matrix_for(entry)
    else:
        try:
            block = provider[int(entry.offset)]
        except Exception:
            index = None
            basis = getattr(provider, "basis", None)
            if basis is not None and hasattr(basis, "entry_index"):
                index = basis.entry_index(entry.key)
            block = None if index is None else provider[index]
    if block is None:
        return np.eye(entry.size, dtype=complex)
    block = np.asarray(block, dtype=complex)
    if block.shape != (entry.size, entry.size):
        raise ValueError("Metric self-block shape does not match the local basis entry.")
    return 0.5 * (block + block.conj().T)


def _metric_block_transform_cache_key(metric_block, *, tol):
    cutoff = max(float(tol), 1.0e-14)
    raw = np.asarray(metric_block)
    if np.iscomplexobj(raw):
        complex_block = np.ascontiguousarray(raw, dtype=np.complex128)
        scale = max(
            1.0,
            float(np.max(np.abs(complex_block.real))) if complex_block.size else 0.0,
        )
        if (
            complex_block.size
            and float(np.max(np.abs(complex_block.imag))) <= cutoff * scale
        ):
            block = np.ascontiguousarray(complex_block.real, dtype=np.float64)
            _METRIC_BLOCK_TRANSFORM_CACHE_STATS["real_fast"] += 1
        else:
            block = complex_block
    else:
        block = np.ascontiguousarray(raw, dtype=np.float64)
    digest = hashlib.blake2b(block.view(np.uint8), digest_size=16).digest()
    return (
        tuple(int(dim) for dim in block.shape),
        block.dtype.str,
        cutoff,
        digest,
    ), block


def _put_metric_block_transform_cache(cache_key, transform):
    transform_elements = int(np.asarray(transform).size)
    if transform_elements > _METRIC_BLOCK_TRANSFORM_CACHE_MAX_ELEMENTS:
        return transform
    cached_elements = sum(
        int(np.asarray(value).size)
        for value in _METRIC_BLOCK_TRANSFORM_CACHE.values()
    )
    while _METRIC_BLOCK_TRANSFORM_CACHE and (
        len(_METRIC_BLOCK_TRANSFORM_CACHE) >= _METRIC_BLOCK_TRANSFORM_CACHE_MAX_SIZE
        or cached_elements + transform_elements
        > _METRIC_BLOCK_TRANSFORM_CACHE_TOTAL_ELEMENTS
    ):
        _old_key = next(iter(_METRIC_BLOCK_TRANSFORM_CACHE))
        old_value = _METRIC_BLOCK_TRANSFORM_CACHE.pop(_old_key)
        cached_elements -= int(np.asarray(old_value).size)
    _METRIC_BLOCK_TRANSFORM_CACHE[cache_key] = transform
    _METRIC_BLOCK_TRANSFORM_CACHE_STATS["puts"] += 1
    return transform


def _positive_metric_eigh(hermitian, *, cutoff):
    """
    Return positive metric eigenpairs, using subset extraction when available.
    """

    dim = int(hermitian.shape[0])
    if scipy_linalg is not None and dim >= 128:
        try:
            eigvals, eigvecs = scipy_linalg.eigh(
                hermitian,
                subset_by_value=(float(cutoff), np.inf),
                driver="evr",
                check_finite=False,
                overwrite_a=False,
            )
            _METRIC_BLOCK_TRANSFORM_CACHE_STATS["scipy_subset_eigh"] += 1
            return eigvals, eigvecs
        except Exception:
            pass
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    keep = eigvals > float(cutoff)
    return eigvals[keep], eigvecs[:, keep]


def _large_metric_has_sampled_offdiag(metric_block, *, cutoff):
    """
    Return True when sampled rows prove a large metric block is not diagonal.
    """

    dim = int(metric_block.shape[0])
    if dim <= 256:
        return False
    diag = np.diagonal(metric_block)
    diag_scale = max(
        1.0,
        float(np.max(np.abs(diag))) if diag.size else 0.0,
    )
    threshold = float(cutoff) * diag_scale
    rows = np.unique(np.linspace(0, dim - 1, min(dim, 16), dtype=int))
    for row in rows:
        row = int(row)
        max_off = 0.0
        if row > 0:
            max_off = max(max_off, float(np.max(np.abs(metric_block[row, :row]))))
        if row + 1 < dim:
            max_off = max(max_off, float(np.max(np.abs(metric_block[row, row + 1:]))))
        if max_off > threshold:
            _METRIC_BLOCK_TRANSFORM_CACHE_STATS["large_diagonal_sample_rejects"] += 1
            return True
    return False


def _metric_block_transform(metric_block, *, tol):
    cache_key, metric_block = _metric_block_transform_cache_key(metric_block, tol=tol)
    cached = _METRIC_BLOCK_TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        _METRIC_BLOCK_TRANSFORM_CACHE_STATS["hits"] += 1
        return cached
    _METRIC_BLOCK_TRANSFORM_CACHE_STATS["misses"] += 1
    dim = int(metric_block.shape[0])
    if dim == 0:
        return np.zeros((0, 0), dtype=complex)
    if metric_block.shape != (dim, dim):
        raise ValueError("Metric block must be square.")
    cutoff = max(float(tol), 1.0e-14)
    if dim <= 256:
        diag = np.diagonal(metric_block)
        off = metric_block.copy()
        off[np.diag_indices(dim)] = 0.0
        scale = max(1.0, float(np.linalg.norm(metric_block)))
        if np.linalg.norm(off) <= cutoff * scale:
            diag_real = np.real(diag)
            keep = diag_real > cutoff
            if not np.any(keep):
                return None
            if np.max(np.abs(np.imag(diag))) <= cutoff * scale:
                if np.all(keep):
                    if np.max(np.abs(diag_real - 1.0)) <= cutoff:
                        return _put_metric_block_transform_cache(
                            cache_key,
                            np.eye(dim, dtype=metric_block.dtype),
                        )
                    return _put_metric_block_transform_cache(
                        cache_key,
                        np.diag(1.0 / np.sqrt(diag_real)).astype(metric_block.dtype),
                    )
                transform = np.zeros(
                    (dim, int(np.count_nonzero(keep))),
                    dtype=metric_block.dtype,
                )
                rows = np.flatnonzero(keep)
                transform[rows, np.arange(rows.size)] = 1.0 / np.sqrt(diag_real[keep])
                return _put_metric_block_transform_cache(cache_key, transform)
    else:
        diag = np.diagonal(metric_block)
        if not _large_metric_has_sampled_offdiag(metric_block, cutoff=cutoff):
            off = metric_block.copy()
            off[np.diag_indices(dim)] = 0.0
            scale = max(1.0, float(np.linalg.norm(metric_block)))
            if np.linalg.norm(off) <= cutoff * scale:
                diag_real = np.real(diag)
                keep = diag_real > cutoff
                if not np.any(keep):
                    return None
                if np.max(np.abs(np.imag(diag))) <= cutoff * scale:
                    _METRIC_BLOCK_TRANSFORM_CACHE_STATS["large_diagonal_fast"] += 1
                    if np.all(keep):
                        if np.max(np.abs(diag_real - 1.0)) <= cutoff:
                            transform = np.eye(dim, dtype=metric_block.dtype)
                        else:
                            transform = np.diag(1.0 / np.sqrt(diag_real)).astype(
                                metric_block.dtype
                            )
                    else:
                        transform = np.zeros(
                            (dim, int(np.count_nonzero(keep))),
                            dtype=metric_block.dtype,
                        )
                        rows = np.flatnonzero(keep)
                        transform[rows, np.arange(rows.size)] = 1.0 / np.sqrt(
                            diag_real[keep]
                        )
                    return _put_metric_block_transform_cache(cache_key, transform)
    hermitian = 0.5 * (metric_block + metric_block.conj().T)
    if dim > 32:
        try:
            if scipy_linalg is not None:
                chol = scipy_linalg.cholesky(
                    hermitian,
                    lower=True,
                    check_finite=False,
                    overwrite_a=False,
                )
            else:
                chol = np.linalg.cholesky(hermitian)
        except np.linalg.LinAlgError:
            chol = None
        if chol is not None:
            pivots = np.abs(np.diagonal(chol))
            pivot_floor = math.sqrt(cutoff) * max(1.0, float(np.max(pivots)))
            if pivots.size and float(np.min(pivots)) > pivot_floor:
                eye = np.eye(dim, dtype=hermitian.dtype)
                if scipy_linalg is not None:
                    transform = scipy_linalg.solve_triangular(
                        chol.conj().T,
                        eye,
                        lower=False,
                        check_finite=False,
                    )
                else:
                    transform = np.linalg.solve(chol.conj().T, eye)
                _METRIC_BLOCK_TRANSFORM_CACHE_STATS["cholesky_fast"] += 1
                return _put_metric_block_transform_cache(cache_key, transform)
    eigvals, eigvecs = _positive_metric_eigh(hermitian, cutoff=cutoff)
    if eigvals.size == 0:
        return None
    transform = np.ascontiguousarray(eigvecs * (1.0 / np.sqrt(eigvals))[None, :])
    return _put_metric_block_transform_cache(cache_key, transform)


def _factorized_route_metric_transform(metric_block, *, tol):
    """
    Factor an owned route-metric buffer without retaining dense work copies.

    Full-rank local metrics dominate the center-bond peak when the generic
    path separately owns the Hermitian matrix, Cholesky factor, identity, and
    inverse.  LAPACK may overwrite this freshly materialized Fortran buffer
    first with its Cholesky factor and then with the triangular inverse.
    """

    cutoff = max(float(tol), 1.0e-14)
    if scipy_linalg is not None:
        dense = metric_block.to_dense(dtype=metric_block.dtype, order="F")
        try:
            factor = scipy_linalg.cholesky(
                dense,
                lower=False,
                check_finite=False,
                overwrite_a=True,
            )
        except (np.linalg.LinAlgError, ValueError):
            factor = None
        if factor is not None:
            pivots = np.abs(np.diagonal(factor))
            pivot_floor = math.sqrt(cutoff) * max(
                1.0,
                float(np.max(pivots, initial=0.0)),
            )
            if pivots.size and float(np.min(pivots)) > pivot_floor:
                trtri = scipy_linalg.lapack.get_lapack_funcs(
                    "trtri",
                    (factor,),
                )
                transform, info = trtri(
                    factor,
                    lower=0,
                    unitdiag=0,
                    overwrite_c=1,
                )
                if int(info) == 0:
                    _METRIC_BLOCK_TRANSFORM_CACHE_STATS["cholesky_fast"] += 1
                    return transform
        del dense
    return _metric_block_transform(metric_block.to_dense(), tol=tol)


def _metric_connected_components(norm_op, basis, *, tol):
    """
    Return connected components of the exact local metric in packed entries.

    :param norm_op: Local norm operator with a packed matvec.
    :param basis: Parent packed two-site basis.
    :param tol: Numerical threshold for detecting entry coupling.
    :returns: Tuple of entry-index tuples, or ``None`` when no exact metric
        action is available.
    """

    metric_action = getattr(norm_op, "packed_matvec", None) or getattr(norm_op, "aux_packed_matvec", None)
    if metric_action is None:
        return None
    threshold = max(float(tol), 1.0e-10)
    adjacency = [set() for _entry in basis]
    for in_idx, entry in enumerate(basis):
        adjacency[in_idx].add(in_idx)
        for col in range(int(entry.size)):
            vec = np.zeros(int(basis.size), dtype=complex)
            vec[int(entry.offset) + col] = 1.0
            out = np.asarray(metric_action(vec), dtype=complex).reshape(int(basis.size))
            out_norm = max(1.0, float(np.linalg.norm(out)))
            for out_idx, out_entry in enumerate(basis):
                if np.linalg.norm(out[out_entry.slice]) > threshold * out_norm:
                    adjacency[in_idx].add(out_idx)
                    adjacency[out_idx].add(in_idx)

    seen = set()
    components = []
    for idx in range(len(adjacency)):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component = []
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _metric_connected_components_from_factor_routes(basis, routes, *, tol):
    """Return entry components from sparse identity-metric route metadata."""

    adjacency = [set((idx,)) for idx, _entry in enumerate(basis)]
    for route in tuple(routes or ()):
        in_idx = int(route[0])
        out_idx = int(route[1])
        in_entry = route[2]
        route_norm = (
            float(np.linalg.norm(np.asarray(route[4]).reshape(-1)))
            * float(np.linalg.norm(np.asarray(route[5]).reshape(-1)))
            * math.sqrt(int(in_entry.shape[1]) * int(in_entry.shape[2]))
        )
        if route_norm <= max(float(tol), 1.0e-10):
            continue
        adjacency[in_idx].add(out_idx)
        adjacency[out_idx].add(in_idx)
    seen = set()
    components = []
    for idx in range(len(adjacency)):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component = []
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_factorized_route_metric_block(basis, component, routes):
    """Build a compact component-local view of global factor routes."""

    local_slices = {}
    cursor = 0
    for entry_idx in component:
        entry = basis.entries[int(entry_idx)]
        local_slices[int(entry_idx)] = slice(cursor, cursor + int(entry.size))
        cursor += int(entry.size)
    local_routes = []
    for in_idx, out_idx, in_entry, out_entry, left, right in tuple(routes or ()):
        in_idx = int(in_idx)
        out_idx = int(out_idx)
        if in_idx not in local_slices or out_idx not in local_slices:
            continue
        local_routes.append(
            (
                local_slices[in_idx],
                local_slices[out_idx],
                tuple(int(dim) for dim in in_entry.shape),
                tuple(int(dim) for dim in out_entry.shape),
                left,
                right,
            )
        )
    if not local_routes:
        return None
    return FactorizedRouteMetricBlock(
        dim=int(cursor),
        routes=tuple(local_routes),
    )


def _component_parent_indices(basis, component):
    pieces = [
        np.arange(
            int(basis.entries[idx].offset),
            int(basis.entries[idx].offset) + int(basis.entries[idx].size),
            dtype=int,
        )
        for idx in component
    ]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=int)


def _component_metric_block(norm_op, basis, parent_indices, *, tol):
    metric_action = getattr(norm_op, "packed_matvec", None) or getattr(norm_op, "aux_packed_matvec", None)
    if metric_action is None:
        return None
    parent_indices = np.asarray(parent_indices, dtype=int)
    block = np.zeros((parent_indices.size, parent_indices.size), dtype=complex)
    mask = np.ones(int(basis.size), dtype=bool)
    mask[parent_indices] = False
    off_norm_sq = 0.0
    full_norm_sq = 0.0
    for col, parent_col in enumerate(parent_indices):
        vec = np.zeros(int(basis.size), dtype=complex)
        vec[int(parent_col)] = 1.0
        out = np.asarray(metric_action(vec), dtype=complex).reshape(int(basis.size))
        block[:, col] = out[parent_indices]
        off_norm_sq += float(np.linalg.norm(out[mask]) ** 2)
        full_norm_sq += float(np.linalg.norm(out) ** 2)
    off_norm = np.sqrt(off_norm_sq)
    full_norm = np.sqrt(full_norm_sq)
    if off_norm > max(float(tol), 1.0e-10) * max(1.0, full_norm):
        return None
    return 0.5 * (block + block.conj().T)


def _component_operator_terms_from_matvec(
    H_packed,
    basis,
    component_indices,
    component_transforms,
    *,
    max_block_kernel_elements,
):
    terms_by_input = [[] for _component in component_indices]
    parent_dim = int(basis.size)
    for in_idx, in_parent in enumerate(component_indices):
        in_size = int(in_parent.size)
        outputs = [
            np.zeros((int(out_parent.size), in_size), dtype=complex)
            for out_parent in component_indices
        ]
        for col, parent_col in enumerate(in_parent):
            vec = np.zeros(parent_dim, dtype=complex)
            vec[int(parent_col)] = 1.0
            out = np.asarray(H_packed(vec), dtype=complex).reshape(parent_dim)
            for out_idx, out_parent in enumerate(component_indices):
                outputs[out_idx][:, col] = out[out_parent]
        X_in = component_transforms[in_idx]
        for out_idx, parent_kernel in enumerate(outputs):
            elements = int(parent_kernel.shape[0]) * int(parent_kernel.shape[1])
            if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                return None
            X_out = component_transforms[out_idx]
            transformed = X_out.conj().T @ parent_kernel @ X_in
            if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
                terms_by_input[in_idx].append((out_idx, transformed))
    return tuple(tuple(terms) for terms in terms_by_input)


def _entry_kernel_items_from_compiled_packed(
    packed_operator,
    basis,
    *,
    max_block_kernel_elements=None,
):
    compiled_transitions = getattr(packed_operator, "compiled_transitions", None)
    if compiled_transitions is not None:
        if getattr(compiled_transitions, "basis", None) is not basis and not basis.compatible_with_layout(
            getattr(compiled_transitions, "basis", basis).entries
        ):
            return None
        entry_kernel_items = []
        for in_idx, item in enumerate(getattr(compiled_transitions, "items", ())):
            if item is None:
                continue
            cursor = 0
            for segment in item.output_segments:
                pieces = _split_compiled_segment_kernel(item, segment, cursor, basis)
                if pieces is None:
                    return None
                for out_idx, kernel in pieces:
                    entry_kernel_items.append((in_idx, out_idx, np.asarray(kernel, dtype=complex)))
                cursor += int(segment.size)
        return tuple(entry_kernel_items)

    compiled_factorized_terms = getattr(packed_operator, "compiled_factorized_terms", None)
    if compiled_factorized_terms is not None:
        entry_kernel_provider = getattr(compiled_factorized_terms, "entry_kernel_items", None)
        if entry_kernel_provider is not None:
            return entry_kernel_provider(
                max_block_kernel_elements=max_block_kernel_elements,
            )
        entry_kernel_items = []
        for in_idx, terms in enumerate(getattr(compiled_factorized_terms, "items", ())):
            for term in terms:
                out_idx = basis.entry_index(term.output_entry.key)
                elements = int(term.output_entry.size) * int(term.input_entry.size)
                if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                    return None
                kernel = term.kernel_matrix(
                    term.input_entry.shape,
                    max_elements=max(elements, 1),
                )
                if kernel is None:
                    return None
                entry_kernel_items.append((in_idx, out_idx, np.asarray(kernel, dtype=complex)))
        return tuple(entry_kernel_items)

    return None


def _entry_kernel_items_from_local_operator_table(
    table,
    basis,
    *,
    max_block_kernel_elements=None,
):
    """
    Extract entry kernels from a typed renormalized local operator table.

    :param table: ``RenormalizedLocalOperatorTable`` provider.
    :param basis: Parent packed two-site basis.
    :param max_block_kernel_elements: Optional cap on materialized kernels.
    :returns: Entry-kernel tuples or ``None``.
    """

    if table is None:
        return None
    cache_key = _entry_kernel_items_cache_key(
        basis,
        max_block_kernel_elements=max_block_kernel_elements,
    )
    cached = table.get_entry_kernel_items(cache_key)
    if cached is not None:
        return cached
    entry_kernel_items = _entry_kernel_items_from_compiled_packed(
        table,
        basis,
        max_block_kernel_elements=max_block_kernel_elements,
    )
    if entry_kernel_items is None:
        return None
    return table.put_entry_kernel_items(cache_key, entry_kernel_items)


def _metric_connected_components_from_entry_kernels(basis, entry_kernel_items, *, tol):
    threshold = max(float(tol), 1.0e-10)
    adjacency = [set([idx]) for idx, _entry in enumerate(basis)]
    for in_idx, out_idx, kernel in entry_kernel_items:
        if np.linalg.norm(np.asarray(kernel).reshape(-1)) > threshold:
            adjacency[int(in_idx)].add(int(out_idx))
            adjacency[int(out_idx)].add(int(in_idx))

    seen = set()
    components = []
    for idx in range(len(adjacency)):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component = []
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_metric_block_from_entry_kernels(
    basis,
    component,
    entry_kernel_items,
    *,
    entry_slices=None,
    component_dim=None,
):
    if entry_slices is None or component_dim is None:
        entry_slices, component_dims = _component_entry_slices(basis, (component,))
        component_dim = int(component_dims[0])
    dim = int(component_dim)
    block = np.zeros((dim, dim), dtype=complex)
    component_set = set(int(idx) for idx in component)
    for in_idx, out_idx, kernel in entry_kernel_items:
        if int(in_idx) not in component_set and int(out_idx) not in component_set:
            continue
        if int(in_idx) not in component_set or int(out_idx) not in component_set:
            if np.linalg.norm(np.asarray(kernel).reshape(-1)) > 1.0e-10:
                return None
            continue
        _in_comp, in_slice = entry_slices[int(in_idx)]
        _out_comp, out_slice = entry_slices[int(out_idx)]
        block[out_slice, in_slice] += np.asarray(kernel, dtype=complex)
    return 0.5 * (block + block.conj().T)


def _component_metric_blocks_from_entry_kernels(
    basis,
    components,
    entry_kernel_items,
    *,
    entry_slices,
    component_dims,
    tol=1.0e-10,
):
    blocks = [
        np.zeros((int(dim), int(dim)), dtype=complex)
        for dim in component_dims
    ]
    threshold = max(float(tol), 1.0e-10)
    for in_idx, out_idx, kernel in entry_kernel_items:
        in_info = entry_slices.get(int(in_idx))
        out_info = entry_slices.get(int(out_idx))
        if in_info is None or out_info is None:
            if np.linalg.norm(np.asarray(kernel).reshape(-1)) > threshold:
                return None
            continue
        in_comp, in_slice = in_info
        out_comp, out_slice = out_info
        if int(in_comp) != int(out_comp):
            if np.linalg.norm(np.asarray(kernel).reshape(-1)) > threshold:
                return None
            continue
        blocks[int(in_comp)][out_slice, in_slice] += np.asarray(kernel, dtype=complex)
    return tuple(0.5 * (block + block.conj().T) for block in blocks)


def _component_entry_slices(basis, components):
    entry_slices = {}
    component_dims = []
    for comp_idx, component in enumerate(components):
        cursor = 0
        for entry_idx in component:
            entry = basis.entries[entry_idx]
            entry_slices[int(entry_idx)] = (int(comp_idx), slice(cursor, cursor + int(entry.size)))
            cursor += int(entry.size)
        component_dims.append(int(cursor))
    return entry_slices, tuple(component_dims)


def _component_parent_indices_from_slices(basis, components):
    indices = []
    for component in components:
        pieces = [
            np.arange(
                int(basis.entries[idx].offset),
                int(basis.entries[idx].offset) + int(basis.entries[idx].size),
                dtype=int,
            )
            for idx in component
        ]
        indices.append(np.concatenate(pieces) if pieces else np.zeros(0, dtype=int))
    return tuple(indices)


def _component_layout_cached(
    basis,
    components,
    *,
    moving_environment_cache=None,
    timing=None,
):
    cache_key = (
        "component_layout",
        _basis_structure_signature(basis),
        tuple(tuple(int(idx) for idx in component) for component in components),
    )
    if moving_environment_cache is not None:
        cached = moving_environment_cache.get(cache_key)
        if cached is not None:
            if timing is not None:
                timing["component_layout_cache_hit"] = timing.get(
                    "component_layout_cache_hit", 0.0
                ) + 1.0
            return cached
    if timing is not None:
        timing["component_layout_cache_miss"] = timing.get(
            "component_layout_cache_miss", 0.0
        ) + 1.0
    entry_slices, component_dims = _component_entry_slices(basis, components)
    parent_indices = _component_parent_indices_from_slices(basis, components)
    value = (entry_slices, component_dims, parent_indices)
    if moving_environment_cache is None:
        return value
    return moving_environment_cache.put(cache_key, value)


def _component_entry_slices_cached(
    basis,
    components,
    *,
    moving_environment_cache=None,
    timing=None,
):
    """
    Return component entry slices, optionally via the moving-environment cache.

    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param moving_environment_cache: Optional structural cache shared by the
        renormalized block stack.
    :param timing: Optional timing/counter dictionary.
    :returns: ``(entry_slices, component_dims)``.
    """

    if moving_environment_cache is None:
        return _component_entry_slices(basis, components)
    cache_key = (
        "component_entry_slices",
        _basis_structure_signature(basis),
        tuple(tuple(int(idx) for idx in component) for component in components),
    )
    cached = moving_environment_cache.get(cache_key)
    if cached is not None:
        if timing is not None:
            timing["component_entry_slices_cache_hit"] = timing.get(
                "component_entry_slices_cache_hit", 0.0
            ) + 1.0
        return cached
    if timing is not None:
        timing["component_entry_slices_cache_miss"] = timing.get(
            "component_entry_slices_cache_miss", 0.0
        ) + 1.0
    entry_slices, component_dims = _component_entry_slices(basis, components)
    return moving_environment_cache.put(cache_key, (entry_slices, component_dims))


def _basis_structure_signature(basis):
    """
    Return a hashable structural signature for a two-site basis.

    :param basis: :class:`TwoSiteBasis`-like object.
    :returns: Tuple of entry keys, shapes, offsets, and sizes.
    """

    return tuple(
        (entry.key, tuple(entry.shape), int(entry.offset), int(entry.size))
        for entry in basis
    )


def _entry_kernel_items_cache_key(basis, *, max_block_kernel_elements):
    """
    Return a cache key for dense entry-kernel extraction.

    :param basis: Parent packed two-site basis.
    :param max_block_kernel_elements: Optional dense-kernel cap.
    :returns: Hashable cache key.
    """

    return (
        "entry_kernel_items",
        _basis_structure_signature(basis),
        None if max_block_kernel_elements is None else int(max_block_kernel_elements),
    )


def _entry_kernel_content_signature(entry_kernel_items):
    """
    Return a stable content signature for compiled entry kernels.

    The component-basis cache must survive freshly constructed local operator
    wrappers.  Object ids of packed matvec callables are therefore too narrow;
    the metric kernels themselves define the norm source used for the
    orthonormal component basis.
    """

    if entry_kernel_items is None:
        return None
    fast_key = []
    for in_idx, out_idx, kernel in entry_kernel_items:
        arr = np.asarray(kernel)
        fast_key.append(
            (
                int(in_idx),
                int(out_idx),
                id(kernel),
                str(arr.dtype),
                tuple(int(dim) for dim in arr.shape),
                int(arr.size),
            )
        )
    fast_key = tuple(fast_key)
    cached = _METRIC_ENTRY_KERNEL_SIGNATURE_CACHE.get(fast_key)
    if cached is not None:
        return cached

    signature = []
    for in_idx, out_idx, kernel in entry_kernel_items:
        arr = np.ascontiguousarray(np.asarray(kernel))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(arr.dtype).encode("utf8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.view(np.uint8))
        signature.append(
            (
                int(in_idx),
                int(out_idx),
                str(arr.dtype),
                tuple(int(dim) for dim in arr.shape),
                digest.hexdigest(),
            )
        )
    signature = tuple(signature)
    if len(_METRIC_ENTRY_KERNEL_SIGNATURE_CACHE) >= _METRIC_ENTRY_KERNEL_SIGNATURE_CACHE_MAX_SIZE:
        _METRIC_ENTRY_KERNEL_SIGNATURE_CACHE.pop(
            next(iter(_METRIC_ENTRY_KERNEL_SIGNATURE_CACHE))
        )
    _METRIC_ENTRY_KERNEL_SIGNATURE_CACHE[fast_key] = signature
    return signature


def _component_basis_cache_key(norm_op, basis, metric_entry_kernels, *, tol):
    """
    Return a safe cache key for component orthonormal bases.

    :param norm_op: Local norm operator.
    :param basis: Parent two-site basis.
    :param metric_entry_kernels: Optional compiled metric entry kernels.
    :param tol: Metric cutoff.
    :returns: Hashable key or ``None`` when the norm source is not cacheable.
    """

    norm_table = getattr(norm_op, "local_operator_table", None)
    if metric_entry_kernels is not None:
        source_key = ("compiled", _entry_kernel_content_signature(metric_entry_kernels))
    elif norm_table is not None and getattr(norm_table, "key", None) is not None:
        source_key = ("table", norm_table.key)
    else:
        return None
    return (
        "component_basis",
        source_key,
        _basis_structure_signature(basis),
        float(tol),
    )


def _component_basis_cache_get(key):
    if key is None:
        return None
    return _COMPONENT_BASIS_CACHE.get(key)


def _component_basis_cache_put(key, value):
    if key is None:
        return value
    numeric_elements = sum(
        (
            int(item.stored_elements)
            if hasattr(item, "stored_elements")
            else int(np.asarray(item).size)
        )
        for field in ("component_transforms", "metric_blocks")
        for item in tuple(value.get(field, ()) or ())
    )
    if numeric_elements > _COMPONENT_BASIS_CACHE_MAX_NUMERIC_ELEMENTS:
        return value
    def cached_numeric_elements(cached):
        return sum(
            (
                int(item.stored_elements)
                if hasattr(item, "stored_elements")
                else int(np.asarray(item).size)
            )
            for field in ("component_transforms", "metric_blocks")
            for item in tuple(cached.get(field, ()) or ())
        )

    current_elements = sum(
        cached_numeric_elements(cached)
        for cached in _COMPONENT_BASIS_CACHE.values()
    )
    while _COMPONENT_BASIS_CACHE and (
        len(_COMPONENT_BASIS_CACHE) >= _COMPONENT_BASIS_CACHE_MAX_SIZE
        or current_elements + numeric_elements
        > _COMPONENT_BASIS_CACHE_TOTAL_NUMERIC_ELEMENTS
    ):
        _old_key = next(iter(_COMPONENT_BASIS_CACHE))
        old_value = _COMPONENT_BASIS_CACHE.pop(_old_key)
        current_elements -= cached_numeric_elements(old_value)
    _COMPONENT_BASIS_CACHE[key] = value
    return value


def _array_signature(array):
    """
    Return a compact content signature for a numeric array.

    :param array: Array-like object.
    :returns: Tuple containing dtype, shape, and a BLAKE2 digest.
    """

    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.dtype).encode("utf8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.view(np.uint8))
    return (str(arr.dtype), tuple(int(dim) for dim in arr.shape), digest.hexdigest())


def _component_transform_signature(component_transforms):
    """
    Return a hashable signature for component orthonormal transforms.

    :param component_transforms: Sequence of component transform matrices.
    :returns: Tuple of compact array signatures.
    """

    return tuple(_array_signature(transform) for transform in component_transforms)


def _component_transformed_table_cache_key(
    basis,
    components,
    component_transforms,
    *,
    max_block_kernel_elements,
    metric_basis_cache_key,
):
    """
    Return the cache key for a transformed component operator table.

    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param component_transforms: Component orthonormal transforms.
    :param max_block_kernel_elements: Optional dense-kernel cap.
    :param metric_basis_cache_key: Key identifying the local metric source.
    :returns: Hashable transformed-table cache key.
    """

    return (
        "component_transformed_operator_table",
        _basis_structure_signature(basis),
        tuple(tuple(int(idx) for idx in component) for component in components),
        _component_transform_signature(component_transforms),
        None if max_block_kernel_elements is None else int(max_block_kernel_elements),
        metric_basis_cache_key,
    )


def _component_terms_from_entry_kernels(
    basis,
    components,
    component_transforms,
    entry_kernel_items,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    plan = None
    n_items = len(entry_kernel_items)
    use_cache = moving_environment_cache is not None and 8 < n_items <= 2048
    if not use_cache:
        parent_blocks = _component_parent_blocks_from_entry_kernels(
            basis,
            components,
            entry_kernel_items,
            max_block_kernel_elements=max_block_kernel_elements,
            moving_environment_cache=moving_environment_cache,
            timing=timing,
        )
        if parent_blocks is None:
            return None
        return _component_terms_from_parent_blocks(
            parent_blocks,
            components,
            component_transforms,
        )
    if use_cache:
        cache_key = _component_kernel_plan_cache_key(
            basis,
            components,
            entry_kernel_items,
            max_block_kernel_elements=max_block_kernel_elements,
        )
        plan = moving_environment_cache.get(cache_key)
        if timing is not None:
            timing_key = (
                "component_kernel_plan_cache_hit"
                if plan is not None
                else "component_kernel_plan_cache_miss"
            )
            timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
    else:
        cache_key = None
    if plan is None:
        t0 = time.perf_counter() if timing is not None else None
        plan = _build_component_kernel_plan(
            basis,
            components,
            entry_kernel_items,
            max_block_kernel_elements=max_block_kernel_elements,
            moving_environment_cache=moving_environment_cache,
            timing=timing,
        )
        if timing is not None:
            timing["component_kernel_plan_build"] = timing.get(
                "component_kernel_plan_build", 0.0
            ) + (time.perf_counter() - t0)
        if plan is None:
            return None
        if use_cache:
            moving_environment_cache.put(cache_key, plan)

    parent_blocks = {
        key: np.zeros(shape, dtype=complex)
        for key, shape in plan["parent_shapes"]
    }
    for item_idx, key, in_slice, out_slice in plan["placements"]:
        parent_blocks[key][out_slice, in_slice] += np.asarray(
            entry_kernel_items[item_idx][2],
            dtype=complex,
        )
    return _component_terms_from_parent_blocks(
        parent_blocks,
        components,
        component_transforms,
    )


def _component_parent_blocks_from_entry_kernels(
    basis,
    components,
    entry_kernel_items,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    """
    Assemble component parent-basis kernels directly from entry kernels.

    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param entry_kernel_items: ``(input_entry, output_entry, kernel)`` tuples.
    :param max_block_kernel_elements: Optional dense parent-block cap.
    :returns: Mapping ``(input_component, output_component)`` to kernels, or
        ``None`` when a parent block exceeds the cap.
    """

    entry_slices, component_dims = _component_entry_slices_cached(
        basis,
        components,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )
    parent_blocks = {}
    for in_entry_idx, out_entry_idx, kernel in entry_kernel_items:
        in_comp, in_slice = entry_slices[int(in_entry_idx)]
        out_comp, out_slice = entry_slices[int(out_entry_idx)]
        key = (int(in_comp), int(out_comp))
        if key not in parent_blocks:
            elements = int(component_dims[out_comp]) * int(component_dims[in_comp])
            if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                return None
            parent_blocks[key] = np.zeros(
                (int(component_dims[out_comp]), int(component_dims[in_comp])),
                dtype=complex,
            )
        parent_blocks[key][out_slice, in_slice] += np.asarray(kernel, dtype=complex)
    return parent_blocks


def _component_terms_from_parent_blocks(
    parent_blocks,
    components,
    component_transforms,
):
    """
    Transform component parent blocks into orthonormal block terms.

    :param parent_blocks: Mapping ``(input_component, output_component)`` to
        parent-basis kernels.
    :param components: Metric-connected entry components.
    :param component_transforms: Parent-to-orthonormal transforms per component.
    :returns: Tuple of block terms grouped by input component.
    """

    terms_by_input = [[] for _component in components]
    for (in_comp, out_comp), parent_kernel in parent_blocks.items():
        X_in = component_transforms[in_comp]
        X_out = component_transforms[out_comp]
        transformed = X_out.conj().T @ parent_kernel @ X_in
        if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
            terms_by_input[in_comp].append((out_comp, transformed))
    return tuple(tuple(terms) for terms in terms_by_input)


def _component_kernel_plan_cache_key(
    basis,
    components,
    entry_kernel_items,
    *,
    max_block_kernel_elements,
):
    """
    Return a structural key for component parent-block assembly.

    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param entry_kernel_items: ``(input_entry, output_entry, kernel)`` tuples.
    :param max_block_kernel_elements: Optional dense parent-block cap.
    :returns: Hashable moving-environment cache key.
    """

    return (
        "component_kernel_plan",
        _basis_structure_signature(basis),
        tuple(tuple(int(idx) for idx in component) for component in components),
        tuple(
            (
                int(in_idx),
                int(out_idx),
                tuple(np.asarray(kernel).shape),
            )
            for in_idx, out_idx, kernel in entry_kernel_items
        ),
        None if max_block_kernel_elements is None else int(max_block_kernel_elements),
    )


def _build_component_kernel_plan(
    basis,
    components,
    entry_kernel_items,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    """
    Build a reusable parent-block assembly plan for component kernels.

    The plan contains only slices and block shapes.  Numeric kernels remain
    MPS/operator dependent and are applied after lookup.

    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param entry_kernel_items: ``(input_entry, output_entry, kernel)`` tuples.
    :param max_block_kernel_elements: Optional dense parent-block cap.
    :returns: Plan dictionary or ``None`` when a block exceeds the cap.
    """

    entry_slices, component_dims = _component_entry_slices_cached(
        basis,
        components,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )
    parent_shapes = {}
    placements = []
    for item_idx, (in_entry_idx, out_entry_idx, _kernel) in enumerate(entry_kernel_items):
        in_comp, in_slice = entry_slices[int(in_entry_idx)]
        out_comp, out_slice = entry_slices[int(out_entry_idx)]
        key = (int(in_comp), int(out_comp))
        if key not in parent_shapes:
            elements = int(component_dims[out_comp]) * int(component_dims[in_comp])
            if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                return None
            parent_shapes[key] = (
                int(component_dims[out_comp]),
                int(component_dims[in_comp]),
            )
        placements.append((int(item_idx), key, in_slice, out_slice))
    return {
        "parent_shapes": tuple(sorted(parent_shapes.items())),
        "placements": tuple(placements),
    }


def _component_terms_from_compiled_transitions(
    compiled_transitions,
    basis,
    components,
    component_transforms,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    entry_kernel_items = _entry_kernel_items_from_compiled_packed(
        compiled_transitions.packed_matvec(base_dtype=complex)
        if hasattr(compiled_transitions, "packed_matvec")
        else compiled_transitions,
        basis,
        max_block_kernel_elements=max_block_kernel_elements,
    )
    if entry_kernel_items is None:
        if getattr(compiled_transitions, "basis", None) is not basis and not basis.compatible_with_layout(
            getattr(compiled_transitions, "basis", basis).entries
        ):
            return None
        entry_kernel_items = []
        for in_idx, item in enumerate(getattr(compiled_transitions, "items", ())):
            if item is None:
                continue
            cursor = 0
            for segment in item.output_segments:
                pieces = _split_compiled_segment_kernel(item, segment, cursor, basis)
                if pieces is None:
                    return None
                for out_idx, kernel in pieces:
                    entry_kernel_items.append((in_idx, out_idx, kernel))
                cursor += int(segment.size)
    return _component_terms_from_entry_kernels(
        basis,
        components,
        component_transforms,
        entry_kernel_items,
        max_block_kernel_elements=max_block_kernel_elements,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )


def _component_terms_from_compiled_factorized(
    compiled_factorized_terms,
    basis,
    components,
    component_transforms,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    policy = get_direct_factorized_orthonormal_kernel_policy()
    if bool(policy.get("su2_qchem_direct_parent_blocks", False)):
        t0 = time.perf_counter() if timing is not None else None
        qchem_terms = _component_terms_from_qchem_parent_blocks(
            compiled_factorized_terms,
            basis,
            components,
            component_transforms,
            max_block_kernel_elements=max_block_kernel_elements,
            moving_environment_cache=moving_environment_cache,
            timing=timing,
        )
        if timing is not None:
            timing["component_qchem_parent_blocks"] = timing.get(
                "component_qchem_parent_blocks", 0.0
            ) + (time.perf_counter() - t0)
        if qchem_terms is not None:
            if timing is not None:
                timing["component_qchem_parent_block_kernel"] = timing.get(
                    "component_qchem_parent_block_kernel", 0.0
                ) + 1.0
            return qchem_terms

    owner_table = getattr(compiled_factorized_terms, "local_operator_table", None)
    if bool(getattr(compiled_factorized_terms, "prefer_direct_component_transform", False)):
        t0 = time.perf_counter() if timing is not None else None
        direct_terms = _component_terms_from_compiled_factorized_direct(
            compiled_factorized_terms,
            basis,
            components,
            component_transforms,
            moving_environment_cache=moving_environment_cache,
            timing=timing,
        )
        if direct_terms is not None:
            if timing is not None:
                timing["component_factorized_direct_transform"] = timing.get(
                    "component_factorized_direct_transform", 0.0
                ) + (time.perf_counter() - t0)
            return direct_terms
        if timing is not None:
            timing["component_factorized_direct_transform_fallback"] = timing.get(
                "component_factorized_direct_transform_fallback", 0.0
            ) + 1.0
    cache_key = (
        _entry_kernel_items_cache_key(
            basis,
            max_block_kernel_elements=max_block_kernel_elements,
        )
        if owner_table is not None
        else None
    )
    entry_kernel_items = (
        owner_table.get_entry_kernel_items(cache_key)
        if owner_table is not None
        else None
    )
    cache_hit = entry_kernel_items is not None
    t0 = time.perf_counter() if timing is not None else None
    if entry_kernel_items is None:
        entry_kernel_provider = getattr(
            compiled_factorized_terms,
            "entry_kernel_items",
            None,
        )
        if entry_kernel_provider is not None:
            entry_kernel_items = entry_kernel_provider(
                max_block_kernel_elements=max_block_kernel_elements,
            )
            if entry_kernel_items is None:
                return None
        else:
            entry_kernel_items = []
            for in_idx, terms in enumerate(getattr(compiled_factorized_terms, "items", ())):
                for term in terms:
                    out_idx = basis.entry_index(term.output_entry.key)
                    elements = int(term.output_entry.size) * int(term.input_entry.size)
                    if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                        return None
                    kernel = term.kernel_matrix(
                        term.input_entry.shape,
                        max_elements=max(elements, 1),
                    )
                    if kernel is None:
                        return None
                    entry_kernel_items.append((in_idx, out_idx, kernel))
            entry_kernel_items = tuple(entry_kernel_items)
        if owner_table is not None:
            owner_table.put_entry_kernel_items(cache_key, entry_kernel_items)
    if timing is not None:
        timing["component_factorized_kernel_materialize"] = timing.get(
            "component_factorized_kernel_materialize", 0.0
        ) + (time.perf_counter() - t0)
        timing_key = (
            "component_factorized_kernel_cache_hit"
            if cache_hit
            else "component_factorized_kernel_cache_miss"
        )
        timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
        t0 = time.perf_counter()
    out = _component_terms_from_entry_kernels(
        basis,
        components,
        component_transforms,
        entry_kernel_items,
        max_block_kernel_elements=max_block_kernel_elements,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )
    if timing is not None:
        timing["component_factorized_kernel_transform"] = timing.get(
            "component_factorized_kernel_transform", 0.0
        ) + (time.perf_counter() - t0)
    return out


def _component_terms_from_qchem_parent_blocks(
    compiled_factorized_terms,
    basis,
    components,
    component_transforms,
    *,
    max_block_kernel_elements,
    moving_environment_cache=None,
    timing=None,
):
    """
    Build component terms from a packed qchem parent-block builder.

    This is separate from ``DirectOrthonormalFactorizedTable`` because the
    packed qchem route can safely assemble parent component blocks without
    enabling the older component-direct tensor application plan.
    """

    builder = getattr(compiled_factorized_terms, "build_component_parent_blocks", None)
    if builder is None:
        return None
    compiled_basis = getattr(compiled_factorized_terms, "basis", None)
    if compiled_basis is not basis and not basis.compatible_with_layout(
        getattr(compiled_basis, "entries", basis.entries)
    ):
        return None
    _entry_slices, component_dims = _component_entry_slices_cached(
        basis,
        components,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )
    if max_block_kernel_elements is not None:
        for in_dim in component_dims:
            for out_dim in component_dims:
                if int(in_dim) * int(out_dim) > int(max_block_kernel_elements):
                    return None
    parent_blocks = builder(components, tuple(int(dim) for dim in component_dims))
    if parent_blocks is None:
        return None
    return _component_terms_from_parent_blocks(
        {
            (int(in_comp), int(out_comp)): np.asarray(block, dtype=complex)
            for in_comp, out_comp, block in parent_blocks
        },
        components,
        component_transforms,
    )


def _component_terms_from_compiled_factorized_direct(
    compiled_factorized_terms,
    basis,
    components,
    component_transforms,
    *,
    moving_environment_cache=None,
    timing=None,
):
    """
    Transform factorized local terms directly into component coordinates.

    This avoids assembling dense parent component kernels before applying the
    orthonormal transforms.  Each compiled factorized block contributes
    ``X_out^H L R X_in`` directly to the transformed component block.

    :param compiled_factorized_terms: Compiled factorized local terms.
    :param basis: Parent two-site basis.
    :param components: Metric-connected entry components.
    :param component_transforms: Component orthonormal transforms.
    :param moving_environment_cache: Optional structural cache for entry slices.
    :param timing: Optional timing dictionary.
    :returns: Component block terms, or ``None`` when metadata is incompatible.
    """

    if getattr(compiled_factorized_terms, "basis", None) is not basis and not basis.compatible_with_layout(
        getattr(compiled_factorized_terms, "basis", basis).entries
    ):
        return None
    entry_slices, _component_dims = _component_entry_slices_cached(
        basis,
        components,
        moving_environment_cache=moving_environment_cache,
        timing=timing,
    )
    terms_by_input = [[] for _component in components]
    accum = {}
    for in_idx, terms in enumerate(getattr(compiled_factorized_terms, "items", ())):
        in_info = entry_slices.get(int(in_idx))
        if in_info is None:
            return None
        in_comp, in_slice = in_info
        X_in = np.asarray(component_transforms[in_comp][in_slice, :], dtype=complex)
        X_in_tensor = X_in.reshape(tuple(basis.entries[int(in_idx)].shape) + (X_in.shape[1],))
        for term in terms:
            out_idx = basis.entry_index(term.output_entry.key)
            out_info = entry_slices.get(int(out_idx))
            if out_info is None:
                return None
            out_comp, out_slice = out_info
            X_out = np.asarray(component_transforms[out_comp][out_slice, :], dtype=complex)
            X_out_tensor = X_out.reshape(tuple(term.output_entry.shape) + (X_out.shape[1],))
            t0 = time.perf_counter() if timing is not None else None
            transformed = _transform_factorized_block_direct(
                term,
                X_out_tensor,
                X_in_tensor,
            )
            if timing is not None:
                timing["component_factorized_direct_block_contract"] = timing.get(
                    "component_factorized_direct_block_contract", 0.0
                ) + (time.perf_counter() - t0)
            if np.linalg.norm(transformed.reshape(-1)) <= 1.0e-15:
                continue
            key = (int(in_comp), int(out_comp))
            if key in accum:
                accum[key] += transformed
            else:
                accum[key] = np.ascontiguousarray(transformed)
    for (in_comp, out_comp), transformed in accum.items():
        terms_by_input[in_comp].append((out_comp, transformed))
    return tuple(tuple(terms) for terms in terms_by_input)


def _transform_factorized_block_direct(term, X_out_tensor, X_in_tensor):
    """
    Contract one factorized local block into orthonormal coordinates.

    :param term: ``CompiledFactorizedBlock``.
    :param X_out_tensor: Output-entry transform with shape
        ``term.output_entry.shape + (d_out,)``.
    :param X_in_tensor: Input-entry transform with shape
        ``term.input_entry.shape + (d_in,)``.
    :returns: Dense transformed block ``(d_out, d_in)``.
    """

    left = np.asarray(term.left_stack)
    right = np.asarray(term.right_stack)
    tmp = np.einsum(
        "ladqo,tlkwab->otkwbdq",
        np.asarray(X_out_tensor).conj(),
        left,
        optimize=True,
    )
    tmp = np.einsum(
        "otkwbdq,twqrdc->okbrc",
        tmp,
        right,
        optimize=True,
    )
    return np.einsum(
        "okbrc,kbcri->oi",
        tmp,
        np.asarray(X_in_tensor),
        optimize=True,
    )


def _split_compiled_segment_kernel(item, segment, cursor, basis):
    pieces = []
    local_offset = int(segment.offset)
    remaining = int(segment.size)
    row_cursor = int(cursor)
    offset_to_index = {int(entry.offset): idx for idx, entry in enumerate(basis)}
    while remaining > 0:
        out_idx = offset_to_index.get(local_offset)
        if out_idx is None:
            return None
        out_entry = basis[out_idx]
        if int(out_entry.offset) != local_offset or int(out_entry.size) > remaining:
            return None
        next_cursor = row_cursor + int(out_entry.size)
        pieces.append((out_idx, item.kernel[row_cursor:next_cursor, :]))
        row_cursor = next_cursor
        local_offset += int(out_entry.size)
        remaining -= int(out_entry.size)
    return pieces


def _compiled_transition_block_terms(compiled_transitions, basis, block_transforms):
    terms_by_input = [[] for _entry in basis]
    if getattr(compiled_transitions, "basis", None) is not basis and not basis.compatible_with_layout(
        getattr(compiled_transitions, "basis", basis).entries
    ):
        return None
    for in_idx, item in enumerate(getattr(compiled_transitions, "items", ())):
        if item is None:
            continue
        X_in = block_transforms[in_idx]
        cursor = 0
        for segment in item.output_segments:
            pieces = _split_compiled_segment_kernel(item, segment, cursor, basis)
            if pieces is None:
                return None
            for out_idx, kernel in pieces:
                X_out = block_transforms[out_idx]
                transformed = X_out.conj().T @ np.asarray(kernel, dtype=complex) @ X_in
                if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
                    terms_by_input[in_idx].append((out_idx, transformed))
            cursor += int(segment.size)
    return tuple(tuple(terms) for terms in terms_by_input)


def _compiled_factorized_block_terms(compiled_factorized_terms, basis, block_transforms, *, max_block_kernel_elements):
    terms_by_input = [[] for _entry in basis]
    entry_kernel_provider = getattr(compiled_factorized_terms, "entry_kernel_items", None)
    if entry_kernel_provider is not None:
        entry_kernel_items = entry_kernel_provider(
            max_block_kernel_elements=max_block_kernel_elements,
        )
        if entry_kernel_items is None:
            return None
        for in_idx, out_idx, kernel in entry_kernel_items:
            X_in = block_transforms[int(in_idx)]
            X_out = block_transforms[int(out_idx)]
            transformed = X_out.conj().T @ np.asarray(kernel, dtype=complex) @ X_in
            if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
                terms_by_input[int(in_idx)].append((int(out_idx), transformed))
        return tuple(tuple(terms) for terms in terms_by_input)
    for in_idx, terms in enumerate(getattr(compiled_factorized_terms, "items", ())):
        X_in = block_transforms[in_idx]
        for term in terms:
            out_idx = basis.entry_index(term.output_entry.key)
            X_out = block_transforms[out_idx]
            elements = int(term.output_entry.size) * int(term.input_entry.size)
            if max_block_kernel_elements is not None and elements > int(max_block_kernel_elements):
                return None
            kernel = term.kernel_matrix(
                term.input_entry.shape,
                max_elements=max(elements, 1),
            )
            if kernel is None:
                return None
            transformed = X_out.conj().T @ np.asarray(kernel, dtype=complex) @ X_in
            if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
                terms_by_input[in_idx].append((out_idx, transformed))
    return tuple(tuple(terms) for terms in terms_by_input)


def _build_component_sparse_orthonormalized_local_problem(
    op,
    norm_op,
    basis,
    *,
    tol,
    max_block_kernel_elements,
    name,
    source,
    cache_hit,
    metadata=None,
    timing=None,
    moving_environment_cache=None,
    su2_moving_environment=None,
    local_operator_key=None,
):
    H_packed = getattr(op, "packed_matvec", None) or getattr(op, "aux_packed_matvec", None)
    if H_packed is None or norm_op is None:
        return None
    norm_packed = getattr(norm_op, "packed_matvec", None) or getattr(norm_op, "aux_packed_matvec", None)
    H_table = getattr(op, "local_operator_table", None)
    norm_table = getattr(norm_op, "local_operator_table", None)
    h_compiled_factorized = (
        getattr(H_table, "compiled_factorized_terms", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_factorized_terms", None)
    compact_metric_diagonal = getattr(norm_packed, "diagonal_values", None)
    compact_metric_factors = getattr(
        norm_packed,
        "factorized_metric_blocks",
        None,
    )
    compact_metric_routes = getattr(
        norm_packed,
        "factorized_metric_routes",
        None,
    )
    use_qchem_native_metric = bool(
        h_compiled_factorized is not None
        and getattr(
            h_compiled_factorized,
            "qchem_packed_entry_kernel_provider",
            False,
        )
        and get_su2_kernel_policy().get("actual") == "cpp"
    )
    use_compact_factorized_metric = bool(
        compact_metric_factors is not None
        and use_qchem_native_metric
    )
    use_compact_route_metric = bool(
        compact_metric_routes is not None
        and use_qchem_native_metric
    )
    use_compact_diagonal_metric = bool(
        compact_metric_diagonal is not None
        and getattr(norm_packed, "diagonal_operator", False)
        and use_qchem_native_metric
    )
    if timing is not None and norm_packed is not None:
        timing["norm_cross_sector_blocks"] = timing.get(
            "norm_cross_sector_blocks",
            0.0,
        ) + float(
            getattr(
                norm_packed,
                "factorized_metric_cross_sector_blocks",
                0,
            )
        )
        timing["norm_cross_sector_elements"] = timing.get(
            "norm_cross_sector_elements",
            0.0,
        ) + float(
            getattr(
                norm_packed,
                "factorized_metric_cross_sector_elements",
                0,
            )
        )
        timing["norm_cross_sector_max_abs"] = max(
            float(timing.get("norm_cross_sector_max_abs", 0.0)),
            float(
                getattr(
                    norm_packed,
                    "factorized_metric_cross_sector_max_abs",
                    0.0,
                )
            ),
        )
    metric_entry_kernels = (
        None
        if use_compact_factorized_metric or use_compact_route_metric
        else (
            _entry_kernel_items_from_local_operator_table(norm_table, basis)
            or (
                None
                if norm_packed is None
                else _entry_kernel_items_from_compiled_packed(norm_packed, basis)
            )
        )
    )
    basis_cache_key = _component_basis_cache_key(
        norm_op,
        basis,
        metric_entry_kernels,
        tol=tol,
    )
    if basis_cache_key is not None and (
        use_compact_factorized_metric or use_compact_route_metric
    ):
        basis_cache_key = (
            (
                "compact_diagonal"
                if use_compact_diagonal_metric
                else (
                    "compact_factorized"
                    if use_compact_factorized_metric
                    else "compact_routes"
                )
            ),
            basis_cache_key,
        )
    cached_basis = _component_basis_cache_get(basis_cache_key)
    if cached_basis is not None:
        if timing is not None:
            timing["component_basis_cache_hit"] = timing.get("component_basis_cache_hit", 0.0) + 1.0
        components = cached_basis["components"]
        component_indices = cached_basis["component_indices"]
        component_transforms = cached_basis["component_transforms"]
        metric_blocks = cached_basis["metric_blocks"]
        orth_offsets = cached_basis["orth_offsets"]
        offset = int(cached_basis["orth_dim"])
    else:
        if timing is not None:
            timing["component_basis_cache_miss"] = timing.get("component_basis_cache_miss", 0.0) + 1.0
        t0 = time.perf_counter() if timing is not None else None
        components = (
            tuple((int(index),) for index, _entry in enumerate(basis))
            if use_compact_factorized_metric
            else (
                _metric_connected_components_from_factor_routes(
                    basis,
                    compact_metric_routes,
                    tol=tol,
                )
                if use_compact_route_metric
                else (
                    _metric_connected_components_from_entry_kernels(
                        basis,
                        metric_entry_kernels,
                        tol=tol,
                    )
                    if metric_entry_kernels is not None
                    else _metric_connected_components(norm_op, basis, tol=tol)
                )
            )
        )
        if timing is not None:
            timing["component_components"] = timing.get("component_components", 0.0) + (
                time.perf_counter() - t0
            )
        if not components:
            return None

        t0 = time.perf_counter() if timing is not None else None
        entry_slices, component_dims, parent_indices_by_component = (
            _component_layout_cached(
                basis,
                components,
                moving_environment_cache=moving_environment_cache,
                timing=timing,
            )
        )
        if timing is not None:
            timing["component_layout"] = timing.get("component_layout", 0.0) + (
                time.perf_counter() - t0
            )

        metric_blocks_from_kernels = None
        if (
            metric_entry_kernels is not None
            and not use_compact_factorized_metric
            and not use_compact_route_metric
        ):
            t0 = time.perf_counter() if timing is not None else None
            metric_blocks_from_kernels = _component_metric_blocks_from_entry_kernels(
                basis,
                components,
                metric_entry_kernels,
                entry_slices=entry_slices,
                component_dims=component_dims,
                tol=tol,
            )
            if timing is not None:
                timing["component_metric_blocks"] = timing.get(
                    "component_metric_blocks", 0.0
                ) + (time.perf_counter() - t0)
            if metric_blocks_from_kernels is None:
                return None

        component_indices = []
        component_transforms = []
        metric_blocks = []
        orth_offsets = []
        offset = 0
        metric_component_count = 0
        metric_parent_dim_sum = 0
        metric_parent_dim_max = 0
        metric_orth_dim_sum = 0
        metric_orth_dim_max = 0
        t0 = time.perf_counter() if timing is not None else None
        cache_stats0 = dict(_METRIC_BLOCK_TRANSFORM_CACHE_STATS)
        for comp_idx, component in enumerate(components):
            parent_indices = parent_indices_by_component[int(comp_idx)]
            if use_compact_diagonal_metric:
                metric_diagonal = np.asarray(
                    compact_metric_diagonal,
                    dtype=complex,
                ).reshape(-1)[parent_indices]
                metric_block = DiagonalMetricBlock(metric_diagonal)
                transform = DiagonalMetricTransform.from_metric_diagonal(
                    metric_diagonal,
                    tol=tol,
                )
                if transform is None:
                    return None
            elif use_compact_factorized_metric:
                entry = basis.entries[int(component[0])]
                factors = compact_metric_factors.get(entry.key)
                if factors is None:
                    return None
                left, right = factors
                left_transform = _metric_block_transform(left, tol=tol)
                right_transform = _metric_block_transform(right, tol=tol)
                if left_transform is None or right_transform is None:
                    return None
                phys_dims = (int(entry.shape[1]), int(entry.shape[2]))
                metric_block = KroneckerMetricBlock(
                    left=left,
                    right=right,
                    phys_dims=phys_dims,
                )
                transform = KroneckerMetricTransform(
                    left=left_transform,
                    right=right_transform,
                    phys_dims=phys_dims,
                )
            elif use_compact_route_metric:
                metric_block = _component_factorized_route_metric_block(
                    basis,
                    component,
                    compact_metric_routes,
                )
                if metric_block is None:
                    return None
                transform = _factorized_route_metric_transform(
                    metric_block,
                    tol=tol,
                )
                if transform is None:
                    return None
            else:
                metric_block = (
                    metric_blocks_from_kernels[int(comp_idx)]
                    if metric_entry_kernels is not None
                    else _component_metric_block(
                        norm_op,
                        basis,
                        parent_indices,
                        tol=tol,
                    )
                )
                if metric_block is None:
                    return None
                transform = _metric_block_transform(metric_block, tol=tol)
                if transform is None:
                    return None
            component_indices.append(np.asarray(parent_indices, dtype=int))
            if isinstance(
                transform,
                (DiagonalMetricTransform, KroneckerMetricTransform),
            ):
                stored_transform = transform
            else:
                stored_transform = np.asarray(transform)
                if np.iscomplexobj(stored_transform):
                    transform_scale = max(
                        1.0,
                        float(np.max(np.abs(stored_transform.real), initial=0.0)),
                    )
                    if (
                        float(
                            np.max(
                                np.abs(stored_transform.imag),
                                initial=0.0,
                            )
                        )
                        <= max(float(tol), 1.0e-14) * transform_scale
                    ):
                        stored_transform = np.ascontiguousarray(
                            stored_transform.real,
                            dtype=float,
                        )
                    else:
                        stored_transform = np.ascontiguousarray(
                            stored_transform,
                            dtype=complex,
                        )
            component_transforms.append(stored_transform)
            metric_blocks.append(
                metric_block
                if isinstance(
                    metric_block,
                    (
                        DiagonalMetricBlock,
                        FactorizedRouteMetricBlock,
                        KroneckerMetricBlock,
                    ),
                )
                else np.asarray(metric_block, dtype=complex)
            )
            orth_offsets.append(int(offset))
            parent_dim = int(metric_block.shape[0])
            orth_dim = int(transform.shape[1])
            metric_component_count += 1
            metric_parent_dim_sum += parent_dim
            metric_parent_dim_max = max(metric_parent_dim_max, parent_dim)
            metric_orth_dim_sum += orth_dim
            metric_orth_dim_max = max(metric_orth_dim_max, orth_dim)
            offset += int(transform.shape[1])
        if timing is not None:
            timing["component_metric_transforms"] = timing.get("component_metric_transforms", 0.0) + (
                time.perf_counter() - t0
            )
            timing["component_metric_transform_components"] = timing.get(
                "component_metric_transform_components",
                0.0,
            ) + float(metric_component_count)
            timing["component_metric_parent_dim_sum"] = timing.get(
                "component_metric_parent_dim_sum",
                0.0,
            ) + float(metric_parent_dim_sum)
            timing["component_metric_parent_dim_max"] = max(
                float(timing.get("component_metric_parent_dim_max", 0.0)),
                float(metric_parent_dim_max),
            )
            timing["component_metric_orth_dim_sum"] = timing.get(
                "component_metric_orth_dim_sum",
                0.0,
            ) + float(metric_orth_dim_sum)
            timing["component_metric_orth_dim_max"] = max(
                float(timing.get("component_metric_orth_dim_max", 0.0)),
                float(metric_orth_dim_max),
            )
            if use_compact_factorized_metric:
                timing["component_compact_factorized_metric"] = timing.get(
                    "component_compact_factorized_metric",
                    0.0,
                ) + 1.0
            if use_compact_route_metric:
                timing["component_compact_route_metric"] = timing.get(
                    "component_compact_route_metric",
                    0.0,
                ) + 1.0
            for key in (
                "hits",
                "misses",
                "puts",
                "real_fast",
                "large_diagonal_fast",
                "large_diagonal_sample_rejects",
                "cholesky_fast",
                "scipy_subset_eigh",
            ):
                timing[f"component_metric_transform_cache_{key}"] = timing.get(
                    f"component_metric_transform_cache_{key}",
                    0.0,
                ) + float(
                    int(_METRIC_BLOCK_TRANSFORM_CACHE_STATS.get(key, 0))
                    - int(cache_stats0.get(key, 0))
                )
        _component_basis_cache_put(
            basis_cache_key,
            {
                "components": tuple(components),
                "component_indices": tuple(component_indices),
                "component_transforms": tuple(component_transforms),
                "metric_blocks": tuple(metric_blocks),
                "orth_offsets": tuple(orth_offsets),
                "orth_dim": int(offset),
            },
        )

    compiled_transitions = (
        getattr(H_table, "compiled_transitions", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_transitions", None)
    compiled_factorized_terms = (
        getattr(H_table, "compiled_factorized_terms", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_factorized_terms", None)
    component_basis = RenormalizedComponentBasis(
        parent_basis=basis,
        component_indices=tuple(component_indices),
        component_transforms=tuple(component_transforms),
        metric_blocks=tuple(metric_blocks),
        orth_offsets=tuple(orth_offsets),
    )
    block_table = None
    block_terms = None
    prefer_direct_factorized = bool(
        compiled_factorized_terms is not None
        and (
            getattr(
                compiled_factorized_terms,
                "prefer_direct_orthonormal_projection",
                False,
            )
            or getattr(
                compiled_factorized_terms,
                "qchem_packed_entry_kernel_provider",
                False,
            )
        )
    )
    prefer_recursive_factorized = bool(
        compiled_factorized_terms is not None
        and getattr(compiled_factorized_terms, "prefer_recursive_operator_matvec", False)
    )
    cpp_numeric_owner = bool(
        get_su2_kernel_policy().get("actual") == "cpp"
        and su2_moving_environment is not None
        and (prefer_direct_factorized or prefer_recursive_factorized)
    )
    transformed_table_cache_key = None
    direct_factorized_table_cache_key = None
    if H_table is not None and not prefer_direct_factorized and not prefer_recursive_factorized:
        transformed_table_cache_key = _component_transformed_table_cache_key(
            basis,
            components,
            tuple(component_transforms),
            max_block_kernel_elements=max_block_kernel_elements,
            metric_basis_cache_key=basis_cache_key,
        )
        t0 = time.perf_counter() if timing is not None else None
        block_table = H_table.get_transformed_operator_table(transformed_table_cache_key)
        if timing is not None:
            timing_key = (
                "component_transformed_table_cache_hit"
                if block_table is not None
                else "component_transformed_table_cache_miss"
            )
            timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
            timing["component_transformed_table_cache_lookup"] = timing.get(
                "component_transformed_table_cache_lookup", 0.0
            ) + (time.perf_counter() - t0)
        if block_table is None and moving_environment_cache is not None:
            t0 = time.perf_counter() if timing is not None else None
            moving_cache_key = (
                "component_transformed_operator_table",
                getattr(H_table, "key", None),
                transformed_table_cache_key,
            )
            block_table = moving_environment_cache.get(moving_cache_key)
            if block_table is not None:
                H_table.put_transformed_operator_table(
                    transformed_table_cache_key,
                    block_table,
                )
            if timing is not None:
                timing_key = (
                    "component_transformed_table_moving_cache_hit"
                    if block_table is not None
                    else "component_transformed_table_moving_cache_miss"
                )
                timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
                timing["component_transformed_table_moving_cache_lookup"] = timing.get(
                    "component_transformed_table_moving_cache_lookup", 0.0
                ) + (time.perf_counter() - t0)
    elif (
        H_table is not None
        and (prefer_direct_factorized or prefer_recursive_factorized)
        and not cpp_numeric_owner
    ):
        direct_factorized_table_cache_key = (
            "component_direct_factorized_operator_table",
            (
                "recursive_complementary_operator_matvec"
                if prefer_recursive_factorized
                else getattr(
                    compiled_factorized_terms,
                    "direct_orthonormal_projection_source",
                    "component_sparse_direct_factorized",
                )
            ),
            _component_transformed_table_cache_key(
                basis,
                components,
                tuple(component_transforms),
                max_block_kernel_elements=None,
                metric_basis_cache_key=basis_cache_key,
            ),
            getattr(compiled_factorized_terms, "complementary_payload_signature", None),
        )
        t0 = time.perf_counter() if timing is not None else None
        block_table = H_table.get_transformed_operator_table(direct_factorized_table_cache_key)
        if timing is not None:
            timing_key = (
                "component_direct_factorized_table_cache_hit"
                if block_table is not None
                else "component_direct_factorized_table_cache_miss"
            )
            timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
            timing["component_direct_factorized_table_cache_lookup"] = timing.get(
                "component_direct_factorized_table_cache_lookup", 0.0
            ) + (time.perf_counter() - t0)
        if block_table is None and moving_environment_cache is not None:
            t0 = time.perf_counter() if timing is not None else None
            moving_cache_key = (
                "component_direct_factorized_operator_table",
                getattr(H_table, "key", None),
                direct_factorized_table_cache_key,
            )
            block_table = moving_environment_cache.get(moving_cache_key)
            if block_table is not None:
                H_table.put_transformed_operator_table(
                    direct_factorized_table_cache_key,
                    block_table,
                )
            if timing is not None:
                timing_key = (
                    "component_direct_factorized_table_moving_cache_hit"
                    if block_table is not None
                    else "component_direct_factorized_table_moving_cache_miss"
                )
                timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
                timing["component_direct_factorized_table_moving_cache_lookup"] = timing.get(
                    "component_direct_factorized_table_moving_cache_lookup", 0.0
                ) + (time.perf_counter() - t0)
    if block_table is not None:
        block_terms = None
    elif compiled_transitions is not None:
        t0 = time.perf_counter() if timing is not None else None
        block_terms = _component_terms_from_compiled_transitions(
            compiled_transitions,
            basis,
            components,
            tuple(component_transforms),
            max_block_kernel_elements=max_block_kernel_elements,
            moving_environment_cache=moving_environment_cache,
            timing=timing,
        )
        if timing is not None:
            timing["component_kernel_terms"] = timing.get("component_kernel_terms", 0.0) + (
                time.perf_counter() - t0
            )
    elif compiled_factorized_terms is not None:
        if prefer_direct_factorized or prefer_recursive_factorized:
            t0 = time.perf_counter() if timing is not None else None
            block_table = DirectOrthonormalFactorizedTable(
                component_basis=component_basis,
                packed_matvec=H_packed,
                source=str(
                    "recursive_complementary_operator_matvec"
                    if prefer_recursive_factorized
                    else getattr(
                            compiled_factorized_terms,
                            "direct_orthonormal_projection_source",
                            "component_sparse_direct_factorized",
                        )
                    )
                ,
                compiled_factorized_terms=compiled_factorized_terms,
                components=tuple(components),
                su2_moving_environment=su2_moving_environment,
                local_operator_key=local_operator_key,
            )
            if (
                not cpp_numeric_owner
                and H_table is not None
                and direct_factorized_table_cache_key is not None
            ):
                H_table.put_transformed_operator_table(
                    direct_factorized_table_cache_key,
                    block_table,
                )
                if moving_environment_cache is not None:
                    moving_environment_cache.put(
                        (
                            "component_direct_factorized_operator_table",
                            getattr(H_table, "key", None),
                            direct_factorized_table_cache_key,
                        ),
                        block_table,
                    )
            if timing is not None:
                timing["component_direct_factorized_table"] = timing.get(
                    "component_direct_factorized_table", 0.0
                ) + (
                    time.perf_counter() - t0
                )
                timing["component_direct_factorized_preferred"] = timing.get(
                    "component_direct_factorized_preferred", 0.0
                ) + 1.0
                if prefer_recursive_factorized:
                    timing["component_recursive_operator_matvec_preferred"] = timing.get(
                        "component_recursive_operator_matvec_preferred", 0.0
                    ) + 1.0
                timing_key = (
                    "component_complementary_family_table_kernel"
                    if getattr(block_table, "uses_complementary_family_table_kernel", False)
                    else (
                        "component_complementary_payload_tensor_kernel"
                        if getattr(block_table, "uses_complementary_payload_tensor_kernel", False)
                        and getattr(block_table, "uses_component_direct_kernel", False)
                        else (
                            "component_recursive_parent_block_kernel"
                            if getattr(block_table, "uses_component_parent_block_kernel", False)
                            else (
                                "component_direct_factorized_kernel"
                                if getattr(block_table, "uses_component_direct_kernel", False)
                                else "component_direct_factorized_full_matvec"
                            )
                        )
                    )
                )
                timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
        else:
            t0 = time.perf_counter() if timing is not None else None
            block_terms = _component_terms_from_compiled_factorized(
                compiled_factorized_terms,
                basis,
                components,
                tuple(component_transforms),
                max_block_kernel_elements=max_block_kernel_elements,
                moving_environment_cache=moving_environment_cache,
                timing=timing,
            )
            if timing is not None:
                timing["component_kernel_terms"] = timing.get("component_kernel_terms", 0.0) + (
                    time.perf_counter() - t0
                )
            if block_terms is None:
                t0 = time.perf_counter() if timing is not None else None
                block_table = DirectOrthonormalFactorizedTable(
                    component_basis=component_basis,
                    packed_matvec=H_packed,
                    source="component_sparse_direct_factorized_fallback",
                    compiled_factorized_terms=compiled_factorized_terms,
                    components=tuple(components),
                    su2_moving_environment=su2_moving_environment,
                    local_operator_key=local_operator_key,
                )
                if timing is not None:
                    timing["component_direct_factorized_table"] = timing.get(
                        "component_direct_factorized_table", 0.0
                    ) + (
                        time.perf_counter() - t0
                    )
                    timing_key = (
                        "component_complementary_family_table_kernel"
                        if getattr(block_table, "uses_complementary_family_table_kernel", False)
                        else (
                            "component_complementary_payload_tensor_kernel"
                            if getattr(block_table, "uses_complementary_payload_tensor_kernel", False)
                            and getattr(block_table, "uses_component_direct_kernel", False)
                            else (
                                "component_recursive_parent_block_kernel"
                                if getattr(block_table, "uses_component_parent_block_kernel", False)
                                else (
                                    "component_direct_factorized_kernel"
                                    if getattr(block_table, "uses_component_direct_kernel", False)
                                    else "component_direct_factorized_full_matvec"
                                )
                            )
                        )
                    )
                    timing[timing_key] = timing.get(timing_key, 0.0) + 1.0
    else:
        t0 = time.perf_counter() if timing is not None else None
        block_terms = _component_operator_terms_from_matvec(
            H_packed,
            basis,
            tuple(component_indices),
            tuple(component_transforms),
            max_block_kernel_elements=max_block_kernel_elements,
        )
        if timing is not None:
            timing["component_kernel_terms"] = timing.get("component_kernel_terms", 0.0) + (
                time.perf_counter() - t0
            )
    if block_terms is None and block_table is None:
        return None
    if block_table is None:
        t0 = time.perf_counter() if timing is not None else None
        block_table = compile_orthonormal_block_table(
            block_terms,
            tuple(component_transforms),
            tuple(orth_offsets),
        )
        if transformed_table_cache_key is not None and H_table is not None:
            H_table.put_transformed_operator_table(
                transformed_table_cache_key,
                block_table,
            )
            if moving_environment_cache is not None:
                moving_environment_cache.put(
                    (
                        "component_transformed_operator_table",
                        getattr(H_table, "key", None),
                        transformed_table_cache_key,
                    ),
                    block_table,
                )
        if timing is not None:
            timing["component_table_compile"] = timing.get("component_table_compile", 0.0) + (
                time.perf_counter() - t0
            )
            if transformed_table_cache_key is not None and H_table is not None:
                timing["component_transformed_table_cache_put"] = timing.get(
                    "component_transformed_table_cache_put", 0.0
                ) + 1.0
                if moving_environment_cache is not None:
                    timing["component_transformed_table_moving_cache_put"] = timing.get(
                        "component_transformed_table_moving_cache_put", 0.0
                    ) + 1.0

    t0 = time.perf_counter() if timing is not None else None
    orth_dim = int(offset)
    h_diag = None if op.diag is None else np.asarray(op.diag, dtype=float).reshape(basis.size)
    if h_diag is None:
        diag = np.zeros(orth_dim, dtype=float)
    else:
        diag = np.zeros(orth_dim, dtype=float)
        for idx, parent_indices in enumerate(component_indices):
            parent_diag = h_diag[parent_indices]
            transform = component_transforms[idx]
            local_diag = np.real(
                transform.project_diagonal(parent_diag)
                if hasattr(transform, "project_diagonal")
                else (np.abs(transform) ** 2).T @ parent_diag
            )
            start = int(orth_offsets[idx])
            diag[start:start + local_diag.size] = local_diag

    def H_full(vector):
        vector = np.asarray(vector, dtype=complex).reshape(basis.size)
        return np.asarray(H_packed(vector), dtype=complex).reshape(basis.size)

    if timing is not None:
        timing["component_finalize"] = timing.get("component_finalize", 0.0) + (
            time.perf_counter() - t0
        )

    return ComponentOrthonormalizedLocalProblem(
        component_basis=component_basis,
        block_table=block_table,
        full_matvec=H_full,
        diag=diag,
        name=name,
        source=source,
        cache_hit=cache_hit,
        metadata=metadata,
        metric_factor_blocks=compact_metric_factors,
        metric_factor_routes=(
            None
            if compact_metric_routes is None
            else tuple(compact_metric_routes)
        ),
    )


def _build_block_sparse_orthonormalized_local_problem(
    op,
    norm_op,
    template,
    basis,
    *,
    tol,
    max_block_kernel_elements,
    name,
    source,
    cache_hit,
    metadata=None,
    timing=None,
    moving_environment_cache=None,
):
    H_packed = getattr(op, "packed_matvec", None) or getattr(op, "aux_packed_matvec", None)
    if H_packed is None:
        return None
    H_table = getattr(op, "local_operator_table", None)
    h_diag = None if op.diag is None else np.asarray(op.diag, dtype=float)
    compiled_terms = getattr(H_packed, "compiled_factorized_terms", None)
    if (
        getattr(compiled_terms, "contextual_channel_resolved", False)
        and getattr(compiled_terms, "su2_moving_environment", None) is not None
    ):
        component_source = (
            source.rsplit(":block_sparse_operator_table", 1)[0]
            if str(source).endswith(":block_sparse_operator_table")
            else source
        )
        return _build_component_sparse_orthonormalized_local_problem(
            op,
            norm_op,
            basis,
            tol=tol,
            max_block_kernel_elements=max_block_kernel_elements,
            name=name,
            source=f"{component_source}:component_sparse_operator_table",
            cache_hit=cache_hit,
            metadata=metadata,
            timing=timing,
            moving_environment_cache=moving_environment_cache,
        )

    block_transforms = []
    metric_blocks = []
    orth_offsets = []
    offset = 0
    t0 = time.perf_counter() if timing is not None else None
    for entry in basis:
        metric_block = _entry_self_metric_block_from_matvec(
            norm_op,
            entry,
            basis.size,
            tol=tol,
        )
        if metric_block is False:
            component_source = (
                source.rsplit(":block_sparse_operator_table", 1)[0]
                if str(source).endswith(":block_sparse_operator_table")
                else source
            )
            return _build_component_sparse_orthonormalized_local_problem(
                op,
                norm_op,
                basis,
                tol=tol,
                max_block_kernel_elements=max_block_kernel_elements,
                name=name,
                source=f"{component_source}:component_sparse_operator_table",
                cache_hit=cache_hit,
                metadata=metadata,
                timing=timing,
                moving_environment_cache=moving_environment_cache,
            )
        if metric_block is None:
            metric_block = _entry_self_metric_block(norm_op, entry)
        if metric_block is None:
            return None
        transform = _metric_block_transform(metric_block, tol=tol)
        if transform is None:
            return None
        block_transforms.append(np.asarray(transform, dtype=complex))
        metric_blocks.append(np.asarray(metric_block, dtype=complex))
        orth_offsets.append(int(offset))
        offset += int(transform.shape[1])
    if timing is not None:
        timing["block_metric_transforms"] = timing.get("block_metric_transforms", 0.0) + (
            time.perf_counter() - t0
        )

    compiled_transitions = (
        getattr(H_table, "compiled_transitions", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_transitions", None)
    compiled_factorized_terms = (
        getattr(H_table, "compiled_factorized_terms", None)
        if H_table is not None
        else None
    ) or getattr(H_packed, "compiled_factorized_terms", None)
    if compiled_transitions is not None:
        t0 = time.perf_counter() if timing is not None else None
        block_terms = _compiled_transition_block_terms(
            compiled_transitions,
            basis,
            tuple(block_transforms),
        )
        if timing is not None:
            timing["block_kernel_terms"] = timing.get("block_kernel_terms", 0.0) + (
                time.perf_counter() - t0
            )
    elif compiled_factorized_terms is not None:
        t0 = time.perf_counter() if timing is not None else None
        block_terms = _compiled_factorized_block_terms(
            compiled_factorized_terms,
            basis,
            tuple(block_transforms),
            max_block_kernel_elements=max_block_kernel_elements,
        )
        if timing is not None:
            timing["block_kernel_terms"] = timing.get("block_kernel_terms", 0.0) + (
                time.perf_counter() - t0
            )
    else:
        return None
    if block_terms is None:
        return None
    t0 = time.perf_counter() if timing is not None else None
    block_table = compile_orthonormal_block_table(
        block_terms,
        tuple(block_transforms),
        tuple(orth_offsets),
    )
    if timing is not None:
        timing["block_table_compile"] = timing.get("block_table_compile", 0.0) + (
            time.perf_counter() - t0
        )

    t0 = time.perf_counter() if timing is not None else None
    orth_dim = int(offset)
    if h_diag is None:
        diag = np.zeros(orth_dim, dtype=float)
    else:
        h_diag = np.asarray(h_diag, dtype=float).reshape(basis.size)
        diag = np.zeros(orth_dim, dtype=float)
        for idx, entry in enumerate(basis):
            parent_diag = h_diag[entry.slice]
            local_diag = np.real((np.abs(block_transforms[idx]) ** 2).T @ parent_diag)
            start = orth_offsets[idx]
            diag[start:start + local_diag.size] = local_diag

    def H_full(vector):
        vector = np.asarray(vector, dtype=complex).reshape(basis.size)
        return np.asarray(H_packed(vector), dtype=complex).reshape(basis.size)
    if timing is not None:
        timing["block_finalize"] = timing.get("block_finalize", 0.0) + (
            time.perf_counter() - t0
        )

    return BlockSparseOrthonormalizedLocalProblem(
        basis=basis,
        block_transforms=tuple(block_transforms),
        metric_blocks=tuple(metric_blocks),
        block_table=block_table,
        orth_offsets=tuple(orth_offsets),
        full_matvec=H_full,
        diag=diag,
        name=name,
        source=source,
        cache_hit=cache_hit,
        metadata=metadata,
    )


def _solve_preorthonormalized_local_problem(
    problem,
    guess_vec,
    *,
    nroots=1,
    projector_basis=None,
    root_guess_vecs=None,
    tol,
    itermax=100,
    max_space=None,
    tol_residual=None,
    lindep=1.0e-12,
    dense_dim=None,
    allow_unconverged=False,
    profile=False,
):
    """
    Solve a prebuilt orthonormal reduced local problem.

    :param problem: :class:`OrthonormalizedLocalProblem` storing ``X^H H X``.
    :param guess_vec: Initial packed vector in the parent basis.
    :param tol: Davidson convergence tolerance.
    :param nroots: Number of lowest roots to solve.
    :param projector_basis: Optional projector in orthonormal coordinates.
    :param root_guess_vecs: Optional parent-basis guesses for additional roots.
    :returns: ``(energies, parent_vectors, residuals, info)``.
    """

    timing = {
        "guess_project": 0.0,
        "projector": 0.0,
        "davidson": 0.0,
        "matvec": 0.0,
        "residual": 0.0,
        "total": 0.0,
    } if profile else None
    matvec_count = 0
    total_t0 = time.perf_counter() if profile else None

    def timed_matvec(vec):
        nonlocal matvec_count
        if not profile:
            return problem.matvec(vec)
        t0 = time.perf_counter()
        out = problem.matvec(vec)
        timing["matvec"] += time.perf_counter() - t0
        matvec_count += 1
        return out

    def timed_matmat(vectors):
        nonlocal matvec_count
        vectors = np.asarray(vectors, dtype=complex)
        action = getattr(problem, "matmat", None)
        if not callable(action):
            return np.column_stack(
                [timed_matvec(vectors[:, idx]) for idx in range(vectors.shape[1])]
            )
        if not profile:
            return action(vectors)
        t0 = time.perf_counter()
        out = action(vectors)
        timing["matvec"] += time.perf_counter() - t0
        matvec_count += int(vectors.shape[1])
        return out

    timed_matvec.matmat = timed_matmat

    def _sparse_fallback():
        if eigsh is None or LinearOperator is None:
            return None
        dim = int(problem.orthonormal_dim)
        if dim <= 2 or int(nroots) >= dim:
            return None

        def matvec(vec):
            return np.asarray(timed_matvec(vec), dtype=complex).reshape(dim)

        linear_operator = LinearOperator(
            (dim, dim),
            matvec=matvec,
            dtype=np.dtype(complex),
        )
        v0 = None
        if guess_y is not None:
            first_guess = np.asarray(guess_y[:, 0] if guess_y.ndim == 2 else guess_y, dtype=complex).reshape(dim)
            if np.linalg.norm(first_guess) > 1.0e-15:
                v0 = first_guess / np.linalg.norm(first_guess)
        try:
            evals, evecs = eigsh(
                linear_operator,
                k=int(nroots),
                which="SA",
                v0=v0,
                tol=float(tol_residual if tol_residual is not None else max(tol, 1.0e-10)),
                maxiter=max(20, int(itermax) * max(2, int(nroots) + 1)),
            )
        except Exception:
            return None
        order = np.argsort(np.real(evals))[: int(nroots)]
        return (
            np.real(evals[order]).astype(float),
            np.asarray(evecs[:, order], dtype=complex),
            {
                "iterations": int(itermax),
                "converged": True,
                "subspace_dim": int(dim),
                "restarts": 0,
                "sparse_fallback": True,
            },
        )

    guess_vec = np.asarray(guess_vec, dtype=complex).reshape(problem.parent_dim)
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be positive.")
    t0 = time.perf_counter() if profile else None
    guess_cols = [problem.to_orthonormal(guess_vec)]
    for root_guess in root_guess_vecs or ():
        root_guess = np.asarray(root_guess, dtype=complex).reshape(problem.parent_dim)
        root_guess_y = problem.to_orthonormal(root_guess)
        if np.linalg.norm(root_guess_y) > 1.0e-15:
            guess_cols.append(root_guess_y)
    guess_y = np.column_stack(guess_cols)
    if np.linalg.norm(guess_y) <= 1.0e-15:
        guess_y = None
    guess_energy = None
    if profile and guess_y is not None:
        first_guess = np.asarray(guess_y[:, 0], dtype=complex).reshape(-1)
        denominator = np.vdot(first_guess, first_guess)
        if abs(denominator) > 1.0e-15:
            guess_energy = float(
                np.real(
                    np.vdot(first_guess, timed_matvec(first_guess))
                    / denominator
                )
            )
    if profile:
        timing["guess_project"] += time.perf_counter() - t0

    dense_fallback = False
    sparse_fallback = False
    dense_matrix_direct = False
    cpp_davidson_used = False
    projected_dim = None
    local_diag = (
        None
        if problem.diag is None
        else np.asarray(problem.diag, dtype=float).reshape(-1)
    )
    unpreconditioned_local_problem = bool(
        local_diag is None
        or local_diag.size == 0
        or not np.all(np.isfinite(local_diag))
        or float(np.ptp(local_diag))
        <= 1.0e-14 * max(1.0, float(np.max(np.abs(local_diag))))
    )
    def _dense_solve(matvec, dim):
        nonlocal dense_matrix_direct
        dense_getter = getattr(problem, "dense_operator_matrix", None)
        H_ortho = None
        if dense_getter is not None:
            candidate = dense_getter()
            if candidate is not None:
                candidate = np.asarray(candidate, dtype=complex)
                if candidate.shape == (int(dim), int(dim)):
                    H_ortho = candidate
                    dense_matrix_direct = True
        if H_ortho is None:
            H_ortho = _materialize_local_matrix(matvec, int(dim))
        evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
        order = np.argsort(np.real(evals))[:nroots]
        return (
            np.real(evals[order]).astype(float),
            np.asarray(evecs[:, order], dtype=complex),
            {
                "iterations": 0,
                "converged": True,
                "subspace_dim": int(dim),
                "restarts": 0,
            },
        )

    try:
        t0 = time.perf_counter() if profile else None
        cpp_davidson = getattr(
            getattr(problem, "block_table", None),
            "cpp_davidson",
            None,
        )
        cpp_davidson_table = getattr(
            getattr(problem, "block_table", None),
            "_cpp_davidson_table",
            None,
        )
        if (
            projector_basis is None
            and dense_dim is not None
            and int(problem.orthonormal_dim) <= int(dense_dim)
        ):
            energies, vecs, info = _dense_solve(
                timed_matvec,
                problem.orthonormal_dim,
            )
            dense_fallback = True
        elif (
            projector_basis is None
            and unpreconditioned_local_problem
            and int(problem.orthonormal_dim)
            >= int(_SU2_UNPRECONDITIONED_SPARSE_MIN_DIM)
        ):
            sparse_result = _sparse_fallback()
            if sparse_result is None:
                raise RuntimeError(
                    "Sparse eigensolver is unavailable for the large "
                    "unpreconditioned SU(2) local problem."
                )
            energies, vecs, info = sparse_result
            sparse_fallback = True
        elif (
            projector_basis is None
            and int(nroots) == 1
            and callable(cpp_davidson)
            and cpp_davidson_table is not None
        ):
            cpp_diag = (
                np.zeros(int(problem.orthonormal_dim), dtype=complex)
                if local_diag is None
                else np.asarray(local_diag, dtype=complex)
            )
            cpp_guess = np.asarray(guess_y[:, 0], dtype=complex)
            cpp_result = cpp_davidson(
                cpp_diag,
                cpp_guess,
                tol=float(tol_residual if tol_residual is not None else tol),
                max_iter=int(itermax),
                restart_dim=int(
                    max(
                        4,
                        min(
                            int(itermax),
                            int(max_space) if max_space is not None else 32,
                        ),
                    )
                ),
                accept_unconverged=True,
                block_size=_SU2_CPP_DAVIDSON_BLOCK_SIZE,
            )
            if cpp_result is None or not bool(cpp_result.get("accepted", False)):
                raise RuntimeError("C++ SU(2) block Davidson did not converge.")
            energies = np.asarray([cpp_result["energy"]], dtype=float)
            vecs = np.asarray(cpp_result["vector"], dtype=complex).reshape(-1, 1)
            info = {
                "iterations": int(cpp_result.get("iterations", 0)),
                "converged": bool(cpp_result.get("converged", False)),
                "subspace_dim": int(cpp_result.get("basis_size", 0)),
                "restarts": int(cpp_result.get("restarts", 0)),
                "cpp_davidson": True,
                "cpp_davidson_kind": cpp_result.get("kind"),
                "cpp_block_davidson": bool(
                    cpp_result.get("block_davidson", False)
                ),
                "cpp_block_size": int(
                    cpp_result.get("block_size", 1)
                ),
                "cpp_workspace_reused": bool(
                    cpp_result.get("workspace_reused", False)
                ),
            }
            if (
                not bool(cpp_result.get("converged", False))
                and not bool(allow_unconverged)
            ):
                raise RuntimeError(
                    "C++ SU(2) block Davidson reached its iteration limit."
                )
            cpp_davidson_used = True
        elif projector_basis is None:
            energies, vecs, info = davidson(
                timed_matvec,
                nroots,
                tol=tol,
                itermax=itermax,
                diag=problem.diag,
                guess=guess_y,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                return_info=True,
                return_partial=allow_unconverged,
            )
            if (
                not bool(info.get("converged", False))
                and int(problem.orthonormal_dim) > 256
            ):
                sparse_result = _sparse_fallback()
                if sparse_result is not None:
                    energies, vecs, info = sparse_result
                    sparse_fallback = True
        else:
            if profile:
                timing["projector"] += time.perf_counter() - t0
                t0 = time.perf_counter()
            projector_basis = np.asarray(projector_basis, dtype=complex)
            if (
                projector_basis.ndim != 2
                or projector_basis.shape[0] != int(problem.orthonormal_dim)
            ):
                raise ValueError(
                    "projector_basis must have shape "
                    "(problem.orthonormal_dim, projected_dim)."
                )
            projected_dim = int(projector_basis.shape[1])
            if int(nroots) > projected_dim:
                raise ValueError("Cannot request more roots than the projected dimension.")

            def projected_matvec(coeff):
                full = projector_basis @ np.asarray(coeff, dtype=complex).reshape(projected_dim)
                return projector_basis.conj().T @ timed_matvec(full)

            projected_diag = (
                None
                if problem.diag is None
                else np.real((np.abs(projector_basis) ** 2).T @ np.asarray(problem.diag, dtype=float))
            )
            projected_guess = None if guess_y is None else projector_basis.conj().T @ guess_y
            if projected_guess is not None and np.linalg.norm(projected_guess) <= 1.0e-15:
                projected_guess = None
            energies, vecs, info = davidson(
                projected_matvec,
                nroots,
                tol=tol,
                itermax=itermax,
                diag=projected_diag,
                guess=projected_guess,
                max_space=max_space,
                tol_residual=tol_residual,
                lindep=lindep,
                return_info=True,
                return_partial=allow_unconverged,
            )
            vecs = projector_basis @ np.asarray(vecs, dtype=complex)
        if profile:
            timing["davidson"] += time.perf_counter() - t0
    except (RuntimeError, ValueError, IndexError):
        sparse_result = None if projector_basis is not None else _sparse_fallback()
        if sparse_result is not None:
            energies, vecs, info = sparse_result
            sparse_fallback = True
        elif projector_basis is None:
            t0 = time.perf_counter() if profile else None
            H_ortho = _materialize_local_matrix(timed_matvec, problem.orthonormal_dim)
            if profile:
                timing["davidson"] += time.perf_counter() - t0
            evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
            order = np.argsort(np.real(evals))[:nroots]
            energies = np.real(evals[order]).astype(float)
            vecs = np.asarray(evecs[:, order], dtype=complex)
            info = {
                "iterations": 0,
                "converged": True,
                "subspace_dim": int(problem.orthonormal_dim),
                "restarts": 0,
            }
            dense_fallback = True
        else:
            projector_basis = np.asarray(projector_basis, dtype=complex)
            projected_dim = int(projector_basis.shape[1])

            def projected_matvec(coeff):
                full = projector_basis @ np.asarray(coeff, dtype=complex).reshape(projected_dim)
                return projector_basis.conj().T @ timed_matvec(full)

            t0 = time.perf_counter() if profile else None
            H_ortho = _materialize_local_matrix(projected_matvec, projected_dim)
            if profile:
                timing["davidson"] += time.perf_counter() - t0
            evals, evecs = np.linalg.eigh(0.5 * (H_ortho + H_ortho.conj().T))
            order = np.argsort(np.real(evals))[:nroots]
            energies = np.real(evals[order]).astype(float)
            vecs = np.asarray(evecs[:, order], dtype=complex)
            vecs = projector_basis @ vecs
            info = {
                "iterations": 0,
                "converged": True,
                "subspace_dim": int(projected_dim or problem.orthonormal_dim),
                "restarts": 0,
            }
            dense_fallback = True

    energies = np.real(np.asarray(energies).reshape(-1)[:nroots]).astype(float)
    root_vecs = []
    residuals = []
    t0 = time.perf_counter() if profile else None
    for root_idx in range(len(energies)):
        y = np.asarray(vecs[:, root_idx], dtype=complex).reshape(problem.orthonormal_dim)
        reference = guess_vec if root_idx == 0 else None
        vec = _canonicalize_eigenvector(problem.from_orthonormal(y), reference=reference)
        if hasattr(problem, "metric_matvec"):
            nvec = problem.metric_matvec(vec)
        else:
            nvec = problem.metric @ vec
        norm = np.sqrt(max(0.0, float(np.real(np.vdot(vec, nvec)))))
        if norm > 1.0e-15:
            vec = vec / norm
            nvec = nvec / norm
            y_residual = y / norm
        else:
            y_residual = y
        if isinstance(problem, ComponentOrthonormalizedLocalProblem):
            residuals.append(
                float(
                    np.linalg.norm(
                        problem.matvec(y_residual)
                        - float(energies[root_idx]) * y_residual
                    )
                )
            )
        else:
            residuals.append(float(np.linalg.norm(problem.full_matvec(vec) - float(energies[root_idx]) * nvec)))
        root_vecs.append(vec)
    if profile:
        timing["residual"] += time.perf_counter() - t0
        timing["total"] = time.perf_counter() - total_t0
    info_out = {
        "davidson_iterations": int(info.get("iterations", 0)),
        "davidson_converged": bool(info.get("converged", False)),
        "subspace_dim": int(info.get("subspace_dim", problem.orthonormal_dim)),
        "restarts": int(info.get("restarts", 0)),
        "orthonormalized_dim": int(problem.orthonormal_dim),
        "dense_fallback": bool(dense_fallback),
        "dense_matrix_direct": bool(dense_matrix_direct),
        "sparse_fallback": bool(sparse_fallback or info.get("sparse_fallback", False)),
        "cpp_davidson": bool(
            cpp_davidson_used or info.get("cpp_davidson", False)
        ),
        "cpp_davidson_kind": info.get("cpp_davidson_kind"),
        "cpp_block_davidson": bool(
            info.get("cpp_block_davidson", False)
        ),
        "cpp_block_size": int(info.get("cpp_block_size", 1)),
        "cpp_workspace_reused": bool(
            info.get("cpp_workspace_reused", False)
        ),
        "renormalized_operator_storage": problem.source,
        "renormalized_operator_cache_hit": bool(getattr(problem, "cache_hit", False)),
        "renormalized_operator_table_stats": getattr(problem, "table_stats", None),
        "renormalized_operator_metadata": getattr(problem, "metadata", None),
        "root_projector_dim": projected_dim,
        "guess_energy": guess_energy,
    }
    metadata = getattr(problem, "metadata", None) or {}
    if "orthonormal_operator_factory_timing" in metadata:
        info_out["operator_factory_timing"] = metadata["orthonormal_operator_factory_timing"]
    if "renormalized_operator_build_timing" in metadata:
        info_out["renormalized_operator_build_timing"] = metadata["renormalized_operator_build_timing"]
    if profile:
        info_out["solver_timing"] = {
            key: float(value)
            for key, value in timing.items()
        }
        info_out["matvec_count"] = int(matvec_count)
    return energies, root_vecs, residuals, info_out


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
    orthonormalize_generalized_dim=None,
    orthonormalize_generalized_operator=False,
    couple_physical=False,
    nstates=1,
    weights=None,
    root_guesses=None,
    root_target_operator=None,
    root_target_value=None,
    root_target_tol=1e-6,
    root_selection_buffer=2,
    root_projector_dim=None,
    root_projector_dense_dim=256,
    root_projector_block_dim=512,
    root_projector_block_max_columns=512,
    root_projector_block_offdiag_tol=1.0e-9,
    root_projector_max_dim=65536,
    filter_coupled_boundary=False,
    use_block_preconditioner=True,
    allow_unconverged_roots=False,
    profile=False,
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

    preorthonormalized_problem = None
    if isinstance(
        local_operator,
        (
            OrthonormalizedLocalProblem,
            BlockSparseOrthonormalizedLocalProblem,
            ComponentOrthonormalizedLocalProblem,
        ),
    ):
        if norm_operator is not None:
            raise ValueError(
                "OrthonormalizedLocalProblem already stores the local metric transform; "
                "do not pass a separate norm_operator."
            )
        preorthonormalized_problem = local_operator
        op_preview = None
        norm_preview = None
        canonical_norm = True
    else:
        op_preview = _normalize_local_operator(local_operator)
        norm_preview = None if norm_operator is None else _normalize_local_operator(norm_operator)

    supplied_basis = (
        getattr(preorthonormalized_problem, "basis", None)
        if preorthonormalized_problem is not None
        else getattr(op_preview, "basis", None)
    )
    if (
        isinstance(supplied_basis, TwoSiteBasis)
        and supplied_basis.channel_resolved
    ):
        vec0, _entries = _pack_tensor_state(
            two_site,
            layout=supplied_basis,
        )
        raw_layout = supplied_basis
    else:
        vec0, raw_layout = pack_two_site_state(two_site)
    if preorthonormalized_problem is not None:
        local_basis = preorthonormalized_problem.basis
        if not local_basis.compatible_with_layout(raw_layout):
            raise ValueError("Prebuilt orthonormalized local problem does not match the active two-site layout.")
    else:
        local_basis = (
            _operator_basis_for_layout(op_preview, raw_layout)
            or (
                _operator_basis_for_layout(norm_preview, raw_layout)
                if norm_preview is not None
                else None
            )
            or two_site_state_basis(two_site, layout=raw_layout)
        )
    layout = local_basis
    if guess is None:
        guess_vec = vec0
    elif isinstance(guess, IrrepTensor):
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
                if isinstance(root_guess, IrrepTensor):
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

    if preorthonormalized_problem is not None:
        preorth_root_target_op = (
            _normalize_local_operator(root_target_operator)
            if root_target_operator is not None
            else None
        )
        nsolve = int(nstates)
        projector_basis = None
        projector_target_values = None
        projector_method = None
        if preorth_root_target_op is not None and root_target_value is not None:
            projected_min_dim = min(
                int(preorthonormalized_problem.parent_dim),
                max(nsolve, int(nstates) + max(0, int(root_selection_buffer))),
            )
            parent_projector, projector_target_values = _target_projector_basis_by_blocks(
                preorth_root_target_op,
                two_site,
                layout,
                target_value=root_target_value,
                target_tol=root_target_tol,
                min_dim=projected_min_dim,
                max_block_size=root_projector_block_dim,
                max_columns=root_projector_block_max_columns,
                offdiag_tol=root_projector_block_offdiag_tol,
            )
            if parent_projector is not None:
                projector_method = "block"
            else:
                parent_projector, projector_target_values = _target_projector_basis(
                    preorth_root_target_op,
                    two_site,
                    layout,
                    target_value=root_target_value,
                    target_tol=root_target_tol,
                    min_dim=projected_min_dim,
                    target_dim=root_projector_dim,
                    dense_dim=root_projector_dense_dim,
                    max_dim=root_projector_max_dim,
                )
                if parent_projector is not None:
                    projector_method = (
                        "dense"
                        if int(preorthonormalized_problem.parent_dim) <= int(root_projector_dense_dim)
                        else "iterative"
                    )
            if parent_projector is not None:
                orth_columns = [
                    preorthonormalized_problem.to_orthonormal(parent_projector[:, idx])
                    for idx in range(int(parent_projector.shape[1]))
                ]
                projector_basis = _orthonormalize_columns_dense(np.column_stack(orth_columns))
                if projector_basis.shape[1] > 0:
                    nsolve = min(nsolve, int(projector_basis.shape[1]))
                else:
                    projector_basis = None
                    projector_method = None
            if projector_basis is None:
                projector_target_values = None
                projector_method = None
            nsolve = min(
                int(preorthonormalized_problem.orthonormal_dim),
                max(nsolve, int(nstates) + max(0, int(root_selection_buffer))),
            ) if projector_basis is None else nsolve
        energies, root_vecs, residuals, solver_info = _solve_preorthonormalized_local_problem(
            preorthonormalized_problem,
            guess_vec,
            nroots=nsolve,
            projector_basis=projector_basis,
            root_guess_vecs=root_guess_vecs,
            tol=tol,
            itermax=itermax,
            max_space=max_space,
            tol_residual=tol_residual,
            lindep=lindep,
            allow_unconverged=allow_unconverged_roots,
            dense_dim=orthonormalized_dense_dim,
            profile=profile,
        )
        target_values = None
        selected_roots = list(range(min(int(nstates), len(root_vecs))))
        if preorth_root_target_op is not None and root_target_value is not None:
            metric_matvec = (
                preorthonormalized_problem.metric_matvec
                if hasattr(preorthonormalized_problem, "metric_matvec")
                else lambda vector: preorthonormalized_problem.metric @ vector
            )
            target_values = _operator_expectations_with_metric_matvec(
                preorth_root_target_op,
                two_site,
                layout,
                root_vecs,
                metric_matvec,
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
        root_match_overlap = None
        if _can_match_all_selected_roots(weights, len(energies)):
            energies, root_vecs, residuals, selected_roots, root_match_overlap = (
                _match_selected_roots_to_guesses(
                    energies,
                    root_vecs,
                    residuals,
                    selected_roots,
                    np.column_stack(root_guess_vecs) if root_guess_vecs else None,
                )
            )
        optimized_roots = [
            unpack_two_site_state(vec, two_site, layout=layout)
            for vec in root_vecs
        ]
        optimized = optimized_roots[0]
        local_weights = weights[: len(energies)]
        local_weights = local_weights / np.sum(local_weights)
        residual = max(residuals) if residuals else 0.0
        return optimized, {
            "energy": float(energies[0]),
            "state_energies": [float(x) for x in energies],
            "state_average_energy": float(np.dot(local_weights, energies)),
            "state_average_weights": [float(x) for x in local_weights],
            "optimized_roots": optimized_roots if nstates > 1 else None,
            "metric": float(residual),
            "residual": float(residual),
            "davidson_iterations": int(solver_info.get("davidson_iterations", 0)),
            "davidson_converged": bool(solver_info.get("davidson_converged", False)),
            "subspace_dim": int(solver_info.get("subspace_dim", 0)),
            "restarts": int(solver_info.get("restarts", 0)),
            "coupled_physical_used": False,
            "canonical_norm": True,
            "canonical_norm_used": True,
            "dense_fallback": bool(solver_info.get("dense_fallback", False)),
            "dense_matrix_direct": bool(solver_info.get("dense_matrix_direct", False)),
            "cpp_davidson": bool(
                solver_info.get("cpp_davidson", False)
            ),
            "cpp_davidson_kind": solver_info.get(
                "cpp_davidson_kind"
            ),
            "cpp_block_davidson": bool(
                solver_info.get("cpp_block_davidson", False)
            ),
            "cpp_block_size": int(
                solver_info.get("cpp_block_size", 1)
            ),
            "cpp_workspace_reused": bool(
                solver_info.get("cpp_workspace_reused", False)
            ),
            "operator_representation": "orthonormalized_renormalized",
            "norm_operator_representation": "stored_metric_transform",
            "orthonormal_basis": "renormalized_environment",
            "effective_local_problem": "orthonormalized_operator_standard",
            "block_davidson": True,
            "orthonormalized_dim": solver_info.get("orthonormalized_dim"),
            "nstates": int(nstates),
            "renormalized_operator_storage": solver_info.get("renormalized_operator_storage"),
            "renormalized_operator_cache_hit": bool(
                solver_info.get("renormalized_operator_cache_hit", False)
            ),
            "renormalized_operator_table_stats": solver_info.get("renormalized_operator_table_stats"),
            "renormalized_operator_metadata": solver_info.get("renormalized_operator_metadata"),
            "target_irrep_filtered": preorth_root_target_op is not None,
            "root_target_values": (
                [float(target_values[idx]) for idx in selected_roots]
                if target_values is not None
                else None
            ),
            "root_target_value": None if root_target_value is None else float(root_target_value),
            "root_selection_used": target_values is not None,
            "root_selection_candidates": int(nsolve),
            "root_overlap_matching": root_match_overlap is not None,
            "root_overlap_matrix": (
                root_match_overlap.tolist() if root_match_overlap is not None else None
            ),
            "root_projector_dim": int(projector_basis.shape[1]) if projector_basis is not None else None,
            "root_projector_method": projector_method,
            "root_projector_target_values": (
                [float(x) for x in projector_target_values]
                if projector_target_values is not None
                else None
            ),
            "solver_timing": solver_info.get("solver_timing"),
            "matvec_count": solver_info.get("matvec_count"),
            "guess_energy": solver_info.get("guess_energy"),
            "operator_factory_timing": solver_info.get("operator_factory_timing"),
            "renormalized_operator_build_timing": solver_info.get(
                "renormalized_operator_build_timing"
            ),
        }

    transform_error = None
    if coupled_mode is not False:
        try:
            coupled_template = _coupled_two_site_template(
                two_site,
                filter_boundary_target=filter_coupled_boundary,
            )
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
            transform_dag = (
                None
                if isinstance(transform, _StructuredBasisTransform)
                else np.asarray(transform).conj().T
            )
    else:
        coupled_template = coupled_layout = None
        transform = transform_dag = None

    canonical_norm_requested = bool(canonical_norm or (norm_preview is not None and norm_preview.identity_like))
    canonical_norm = bool(canonical_norm_requested)
    use_uncoupled_canonical_path = False
    use_uncoupled_orthonormalized_path = False
    if canonical_norm and coupled_template is None:
        norm_operator = None
        norm_preview = None
    use_aux_packed_operator = (
        canonical_norm_requested
        and (
            op_preview.aux_packed_matvec is not None
            or (norm_preview is not None and norm_preview.aux_packed_matvec is not None)
        )
    )
    effective_op = op_preview
    if (
        use_aux_packed_operator
        and op_preview.packed_matvec is None
        and op_preview.aux_packed_matvec is not None
    ):
        effective_op = LocalOperator(
            packed_matvec=op_preview.aux_packed_matvec,
            packed_block_matrices=op_preview.packed_block_matrices,
            basis=op_preview.basis or local_basis,
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
            basis=norm_preview.basis or local_basis,
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
                basis=root_target_op.basis or local_basis,
                diag=root_target_op.diag,
                name=root_target_op.name,
                identity_like=root_target_op.identity_like,
            )

    if nstates > 1:
        if coupled_template is not None:
            if guess is None:
                guess_coupled = coupled_template
            elif isinstance(guess, IrrepTensor):
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
            projector_method = None
            if coupled_target_operator is not None and root_target_value is not None:
                projector_basis, projector_target_values = _target_projector_basis_by_blocks(
                    coupled_target_operator,
                    coupled_template,
                    coupled_layout,
                    norm_operator=coupled_norm_operator,
                    target_value=root_target_value,
                    target_tol=root_target_tol,
                    min_dim=projected_min_dim,
                    max_block_size=root_projector_block_dim,
                    max_columns=root_projector_block_max_columns,
                    offdiag_tol=root_projector_block_offdiag_tol,
                )
                if projector_basis is not None:
                    projector_method = "block"
                else:
                    projector_basis, projector_target_values = _target_projector_basis(
                        coupled_target_operator,
                        coupled_template,
                        coupled_layout,
                        norm_operator=coupled_norm_operator,
                        target_value=root_target_value,
                        target_tol=root_target_tol,
                        min_dim=projected_min_dim,
                        target_dim=root_projector_dim,
                        dense_dim=root_projector_dense_dim,
                        max_dim=root_projector_max_dim,
                    )
                    if projector_basis is not None:
                        projector_method = "dense" if dim <= int(root_projector_dense_dim) else "iterative"
                if projector_basis is not None:
                    nsolve = min(nsolve, int(projector_basis.shape[1]))
            if projector_basis is None and coupled_target_operator is not None and root_target_value is not None:
                nsolve = min(dim, max(nsolve, int(nstates) + max(0, int(root_selection_buffer))))
            nsolve = min(dim, nsolve)
            solver_info = {}
            if coupled_norm_operator is None:
                try:
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
                        use_block_preconditioner=use_block_preconditioner,
                        allow_unconverged=allow_unconverged_roots,
                    )
                except RuntimeError:
                    if dense_fallback_dim is None or dim > int(dense_fallback_dim):
                        raise
                    operator_dense, _ = _resolve_davidson_operator(coupled_operator, coupled_template, coupled_layout)
                    H_matrix = (
                        np.asarray(operator_dense)
                        if isinstance(operator_dense, np.ndarray)
                        else _materialize_local_matrix(operator_dense, dim)
                    )
                    energies, root_vecs, residuals = _solve_projected_dense_roots(
                        H_matrix,
                        None,
                        nroots=nsolve,
                        projector_basis=projector_basis,
                        tol=max(tol, 1e-12),
                    )
                    solver_info = {
                        "davidson_iterations": int(itermax),
                        "davidson_converged": False,
                        "subspace_dim": int(dim),
                        "restarts": 0,
                        "generalized_dense_fallback": True,
                        "standard_dense_fallback": True,
                    }
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
                energies, root_vecs, residuals = _solve_projected_dense_roots(
                    H_matrix,
                    N_matrix,
                    nroots=nsolve,
                    projector_basis=projector_basis,
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
            root_match_overlap = None
            if _can_match_all_selected_roots(weights, len(energies)):
                energies, root_vecs, residuals, selected_roots, root_match_overlap = (
                    _match_selected_roots_to_guesses(
                        energies,
                        root_vecs,
                        residuals,
                        selected_roots,
                        root_coupled_guess_matrix,
                    )
                )
            optimized_roots = []
            for root_idx, vec in enumerate(root_vecs):
                reference = None
                if root_coupled_guess_matrix is not None and root_idx < root_coupled_guess_matrix.shape[1]:
                    reference = root_coupled_guess_matrix[:, root_idx]
                elif root_idx == 0:
                    reference = guess_coupled_vec
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
                "block_preconditioner": bool(solver_info.get("block_preconditioner", False)),
                "block_preconditioner_blocks": int(solver_info.get("block_preconditioner_blocks", 0)),
                "packed_matvec_backend": solver_info.get("packed_matvec_backend"),
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
                "root_projector_method": projector_method,
                "root_projector_target_values": (
                    [float(x) for x in projector_target_values]
                    if projector_target_values is not None
                    else None
                ),
                "root_selection_candidates": int(nsolve),
                "root_overlap_matching": root_match_overlap is not None,
                "root_overlap_matrix": (
                    root_match_overlap.tolist() if root_match_overlap is not None else None
                ),
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
        projector_method = None
        if root_target_op is not None and root_target_value is not None:
            projector_basis, projector_target_values = _target_projector_basis_by_blocks(
                root_target_op,
                two_site,
                layout,
                norm_operator=effective_norm_op,
                target_value=root_target_value,
                target_tol=root_target_tol,
                min_dim=projected_min_dim,
                max_block_size=root_projector_block_dim,
                max_columns=root_projector_block_max_columns,
                offdiag_tol=root_projector_block_offdiag_tol,
            )
            if projector_basis is not None:
                projector_method = "block"
            else:
                projector_basis, projector_target_values = _target_projector_basis(
                    root_target_op,
                    two_site,
                    layout,
                    norm_operator=effective_norm_op,
                    target_value=root_target_value,
                    target_tol=root_target_tol,
                    min_dim=projected_min_dim,
                    target_dim=root_projector_dim,
                    dense_dim=root_projector_dense_dim,
                    max_dim=root_projector_max_dim,
                )
                if projector_basis is not None:
                    projector_method = "dense" if dim <= int(root_projector_dense_dim) else "iterative"
            if projector_basis is not None:
                nsolve = min(nsolve, int(projector_basis.shape[1]))
        if projector_basis is None and root_target_op is not None and root_target_value is not None:
            nsolve = min(dim, max(nsolve, int(nstates) + max(0, int(root_selection_buffer))))
        nsolve = min(dim, nsolve)
        solver_info = {}
        if effective_norm_op is None:
            try:
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
                    use_block_preconditioner=use_block_preconditioner,
                    allow_unconverged=allow_unconverged_roots,
                    profile=profile,
                )
            except RuntimeError:
                if dense_fallback_dim is None or dim > int(dense_fallback_dim):
                    raise
                operator_dense, _ = _resolve_davidson_operator(effective_op, two_site, layout)
                H_matrix = (
                    np.asarray(operator_dense)
                    if isinstance(operator_dense, np.ndarray)
                    else _materialize_local_matrix(operator_dense, dim)
                )
                energies, root_vecs, residuals = _solve_projected_dense_roots(
                    H_matrix,
                    None,
                    nroots=nsolve,
                    projector_basis=projector_basis,
                    tol=max(tol, 1e-12),
                )
                solver_info = {
                    "davidson_iterations": int(itermax),
                    "davidson_converged": False,
                    "subspace_dim": int(dim),
                    "restarts": 0,
                    "generalized_dense_fallback": True,
                    "standard_dense_fallback": True,
                }
        elif orthonormalize_generalized_operator and (
            orthonormalize_generalized_dim is None
            or dim <= int(orthonormalize_generalized_dim)
        ):
            energies, root_vecs, residuals, solver_info = (
                _solve_orthonormalized_operator_davidson_roots(
                    effective_op,
                    effective_norm_op,
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
                    allow_unconverged=allow_unconverged_roots,
                    profile=profile,
                )
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
            energies, root_vecs, residuals = _solve_projected_dense_roots(
                H_matrix,
                N_matrix,
                nroots=nsolve,
                projector_basis=projector_basis,
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
        root_match_overlap = None
        if _can_match_all_selected_roots(weights, len(energies)):
            energies, root_vecs, residuals, selected_roots, root_match_overlap = (
                _match_selected_roots_to_guesses(
                    energies,
                    root_vecs,
                    residuals,
                    selected_roots,
                    root_guess_matrix,
                )
            )
        optimized_roots = []
        for root_idx, vec in enumerate(root_vecs):
            reference = None
            if root_guess_matrix is not None and root_idx < root_guess_matrix.shape[1]:
                reference = root_guess_matrix[:, root_idx]
            elif root_idx == 0:
                reference = guess_vec
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
            "block_preconditioner": bool(solver_info.get("block_preconditioner", False)),
            "block_preconditioner_blocks": int(solver_info.get("block_preconditioner_blocks", 0)),
            "packed_matvec_backend": solver_info.get("packed_matvec_backend"),
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
            "root_projector_method": projector_method,
            "root_projector_target_values": (
                [float(x) for x in projector_target_values]
                if projector_target_values is not None
                else None
            ),
            "root_selection_candidates": int(nsolve),
            "root_overlap_matching": root_match_overlap is not None,
            "root_overlap_matrix": (
                root_match_overlap.tolist() if root_match_overlap is not None else None
            ),
            "nstates": int(nstates),
        }

    if coupled_template is not None:
        if guess is None:
            guess_coupled = coupled_template
        elif isinstance(guess, IrrepTensor):
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
                        use_block_preconditioner=use_block_preconditioner,
                        profile=profile,
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
                energy, vec, residual = _solve_orthonormalized_dense(
                    H_matrix,
                    N_matrix,
                    tol=tol,
                    basis=_two_site_basis_for_layout(coupled_template, coupled_layout),
                )
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
                    "orthonormal_basis": "TwoSiteBasis",
                }
                if coupled_mode == "auto":
                    objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
                return optimized, objective
        if canonical_norm_requested and coupled_norm_operator is not None:
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
                use_block_preconditioner=use_block_preconditioner,
                profile=profile,
            )
            optimized = _uncouple_two_site_tensor(optimized_coupled)
            objective["coupled_physical"] = True
            objective["coupled_physical_used"] = True
            objective["canonical_norm"] = canonical_norm
            objective["canonical_norm_used"] = bool(canonical_norm)
            objective["effective_local_problem"] = (
                "orthonormalized_standard" if canonical_norm else "generalized"
            )
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
        or orthonormalize_generalized_dim is not None
    ):
        dim = guess_vec.size
        ortho_dim_limit = orthonormalized_dense_dim
        if orthonormalize_generalized_dim is not None:
            ortho_dim_limit = max(
                int(orthonormalized_dense_dim or 0),
                int(orthonormalize_generalized_dim),
            )
        if dim <= int(ortho_dim_limit):
            solver_info = {}
            if orthonormalize_generalized_operator and int(nstates) > 1:
                guess_columns = (
                    np.column_stack(root_guess_vecs)
                    if root_guess_vecs
                    else np.asarray(guess_vec, dtype=complex).reshape(-1, 1)
                )
                energies, root_vecs, residuals, solver_info = (
                    _solve_orthonormalized_operator_davidson_roots(
                        effective_op,
                        effective_norm_op,
                        two_site,
                        layout,
                        guess_columns,
                        nroots=int(nstates),
                        tol=tol,
                        itermax=itermax,
                        max_space=max_space,
                        tol_residual=tol_residual,
                        lindep=lindep,
                        allow_unconverged=allow_unconverged_roots,
                        profile=profile,
                    )
                )
                optimized_roots = [
                    unpack_two_site_state(vec, two_site, layout=layout)
                    for vec in root_vecs
                ]
                local_weights = weights[: len(energies)]
                local_weights = local_weights / np.sum(local_weights)
                residual = max(residuals) if residuals else 0.0
                objective = {
                    "energy": float(energies[0]),
                    "state_energies": [float(x) for x in energies],
                    "state_average_energy": float(np.dot(local_weights, energies)),
                    "state_average_weights": [float(x) for x in local_weights],
                    "optimized_roots": optimized_roots,
                    "metric": float(residual),
                    "residual": float(residual),
                    "davidson_iterations": int(solver_info.get("davidson_iterations", 0)),
                    "davidson_converged": bool(solver_info.get("davidson_converged", False)),
                    "subspace_dim": int(solver_info.get("subspace_dim", dim)),
                    "coupled_physical_used": False,
                    "canonical_norm": canonical_norm,
                    "dense_fallback": bool(solver_info.get("dense_fallback", False)),
                    "dense_matrix_direct": bool(
                        solver_info.get("dense_matrix_direct", False)
                    ),
                    "operator_representation": "orthonormalized_operator",
                    "norm_operator_representation": "dense",
                    "orthonormal_basis": "TwoSiteBasis" if isinstance(layout, TwoSiteBasis) else "dense",
                    "effective_local_problem": "orthonormalized_operator_standard",
                    "block_davidson": True,
                    "orthonormalized_dim": solver_info.get("orthonormalized_dim"),
                    "missing_diagonal_preconditioner": bool(
                        solver_info.get("missing_diagonal_preconditioner", False)
                    ),
                    "solver_timing": solver_info.get("solver_timing"),
                    "matvec_count": solver_info.get("matvec_count"),
                    "target_irrep_filtered": False,
                }
                if use_uncoupled_canonical_path:
                    objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
                elif use_uncoupled_orthonormalized_path:
                    objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
                elif orthonormalize_generalized_dim is not None:
                    objective["coupled_physical_skipped"] = "metric_orthonormalized_generalized_path"
                elif canonical_norm_requested and coupled_template is not None:
                    objective["canonical_norm_skipped"] = "coupled_physical_path"
                if transform_error is not None and coupled_mode == "auto":
                    objective["coupled_physical_skipped"] = type(transform_error).__name__
                return optimized_roots[0], objective
            if orthonormalize_generalized_operator:
                energy, vec, residual, solver_info = _solve_orthonormalized_operator_davidson(
                    effective_op,
                    effective_norm_op,
                    two_site,
                    layout,
                    guess_vec,
                    tol=tol,
                    itermax=itermax,
                    max_space=max_space,
                    tol_residual=tol_residual,
                    lindep=lindep,
                    allow_unconverged=allow_unconverged_roots,
                )
            else:
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
                energy, vec, residual = _solve_orthonormalized_dense(
                    H_matrix,
                    N_matrix,
                    tol=tol,
                    basis=layout if isinstance(layout, TwoSiteBasis) else None,
                )
            vec = _canonicalize_eigenvector(vec, reference=guess_vec)
            optimized = unpack_two_site_state(vec, two_site, layout=layout)
            objective = {
                "energy": float(energy),
                "metric": residual,
                "residual": residual,
                "davidson_iterations": int(solver_info.get("davidson_iterations", 0)),
                "davidson_converged": bool(solver_info.get("davidson_converged", False)),
                "subspace_dim": int(solver_info.get("subspace_dim", dim)),
                "coupled_physical_used": False,
                "canonical_norm": canonical_norm,
                "dense_fallback": not bool(orthonormalize_generalized_operator) or bool(
                    solver_info.get("dense_fallback", False)
                ),
                "dense_matrix_direct": bool(
                    solver_info.get("dense_matrix_direct", False)
                ),
                "sparse_fallback": bool(
                    solver_info.get("sparse_fallback", False)
                ),
                "operator_representation": (
                    "orthonormalized_operator"
                    if orthonormalize_generalized_operator
                    else "dense"
                ),
                "norm_operator_representation": "dense",
                "orthonormal_basis": "TwoSiteBasis" if isinstance(layout, TwoSiteBasis) else "dense",
                "effective_local_problem": (
                    "orthonormalized_operator_standard"
                    if orthonormalize_generalized_operator
                    else "orthonormalized_standard"
                    if canonical_norm
                    else "orthonormalized_dense"
                ),
                "block_davidson": bool(orthonormalize_generalized_operator),
                "orthonormalized_dim": solver_info.get("orthonormalized_dim"),
                "missing_diagonal_preconditioner": bool(
                    solver_info.get("missing_diagonal_preconditioner", False)
                ),
                "solver_timing": solver_info.get("solver_timing"),
                "matvec_count": solver_info.get("matvec_count"),
            }
            if use_uncoupled_canonical_path:
                objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
            elif use_uncoupled_orthonormalized_path:
                objective["coupled_physical_skipped"] = "uncoupled_orthonormalized_path"
            elif orthonormalize_generalized_dim is not None:
                objective["coupled_physical_skipped"] = "metric_orthonormalized_generalized_path"
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
            use_block_preconditioner=use_block_preconditioner,
            profile=profile,
        )
        objective["dense_fallback"] = False
        objective["coupled_physical_used"] = False
        canonical_reduced_basis = bool(
            objective.get("canonical_reduced_basis", False)
        )
        objective["canonical_norm"] = bool(
            canonical_norm or canonical_reduced_basis
        )
        objective["canonical_norm_used"] = bool(
            canonical_norm or canonical_reduced_basis
        )
        objective["effective_local_problem"] = (
            "orthonormalized_standard"
            if canonical_norm and use_uncoupled_canonical_path
            else "orthonormalized_standard"
            if objective.get("metric_orthonormal_krylov", False)
            else "standard"
            if canonical_norm
            else "generalized"
        )
        if objective.get("metric_orthonormal_krylov", False):
            objective["orthonormal_basis"] = (
                "cpp_canonical_reduced"
                if canonical_reduced_basis
                else "metric_krylov"
            )
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
    objective["canonical_norm_used"] = bool(canonical_norm)
    objective["effective_local_problem"] = (
        "orthonormalized_standard"
        if canonical_norm and use_uncoupled_canonical_path
        else "standard"
        if canonical_norm
        else "generalized"
    )
    if use_uncoupled_canonical_path:
        objective["coupled_physical_skipped"] = "uncoupled_canonical_path"
    elif canonical_norm_requested and coupled_template is not None:
        objective["canonical_norm_skipped"] = "coupled_physical_path"
    if transform_error is not None and coupled_mode == "auto":
        objective["coupled_physical_skipped"] = type(transform_error).__name__
    return optimized, objective
