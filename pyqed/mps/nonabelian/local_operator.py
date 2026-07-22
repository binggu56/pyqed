#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compiled local effective-operator actions for non-Abelian DMRG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional runtime acceleration
    sp = None

from .solver import LocalOperator, ReducedStateVector, pack_two_site_state
from .tensor import NonabelianTensor


def _cached_einsum_path(signature, *shapes):
    operands = [np.zeros(shape, dtype=float) for shape in shapes]
    path, _ = np.einsum_path(signature, *operands, optimize="greedy")
    return path


_PACKED_DENSE_LOCAL_DIM = 0
_PACKED_CSR_LOCAL_DIM = 128
_FACTORIZED_DENSE_LOCAL_DIM = 0
_FACTORIZED_BLOCK_DENSE_KERNEL_MAX_ELEMENTS = 0
_FACTORIZED_TWO_SITE_BATCH_PATH = _cached_einsum_path(
    "talwpi,lijr,twbjqr->apbq",
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
)
_IDENTITY_TWO_SITE_KERNEL_PATH = _cached_einsum_path(
    "al,po,qr,bc->apqblorc",
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
)


def _factorized_block_two_step_matrices(left_stack, right_stack):
    """
    Return cached matrix views for the two-step factorized block contraction.
    """

    left_stack = np.asarray(left_stack)
    right_stack = np.asarray(right_stack)
    tdim, ldim, kdim, wdim, adim, bdim = left_stack.shape
    t_right, w_right, qdim, rdim, ddim, cdim = right_stack.shape
    if tdim != t_right or wdim != w_right:
        raise ValueError("Incompatible left/right factorized block stacks.")
    left_matrix = np.ascontiguousarray(
        np.transpose(left_stack, (0, 1, 3, 4, 2, 5)).reshape(
            tdim * ldim * wdim * adim,
            kdim * bdim,
        )
    )
    right_matrix = np.ascontiguousarray(
        np.transpose(right_stack, (0, 1, 5, 3, 4, 2)).reshape(
            tdim * wdim * cdim * rdim,
            ddim * qdim,
        )
    )
    shape_info = (tdim, ldim, kdim, wdim, adim, bdim, qdim, rdim, ddim, cdim)
    return left_matrix, right_matrix, shape_info


def _apply_factorized_block_two_step(left_stack, block_in, right_stack, *, matrices=None):
    """
    Apply one factorized block through two BLAS-backed contractions.
    """

    block_in = np.asarray(block_in)
    if matrices is None:
        left_matrix, right_matrix, shape_info = _factorized_block_two_step_matrices(
            left_stack,
            right_stack,
        )
    else:
        left_matrix, right_matrix, shape_info = matrices
    tdim, ldim, kdim, wdim, adim, bdim, qdim, rdim, ddim, cdim = shape_info
    k_in, b_in, c_in, r_in = block_in.shape
    if kdim != k_in or bdim != b_in or cdim != c_in or rdim != r_in:
        raise ValueError("Incompatible factorized block contraction shapes.")

    block_matrix = block_in.reshape(kdim * bdim, cdim * rdim)
    tmp = left_matrix @ block_matrix
    tmp = tmp.reshape(tdim, ldim, wdim, adim, cdim, rdim)
    tmp_matrix = np.transpose(tmp, (1, 3, 0, 2, 4, 5)).reshape(
        ldim * adim,
        tdim * wdim * cdim * rdim,
    )
    return (tmp_matrix @ right_matrix).reshape(ldim, adim, ddim, qdim)


@dataclass
class CompiledLocalActions:
    """
    Solver-facing compiled two-site effective-operator actions.

    The object is the local analogue of a block-DMRG effective Hamiltonian:
    it owns tensor, reduced, and packed matvec entry points plus the diagonal
    and identity metadata needed by the eigensolver/preconditioner layer.

    :param basis: Explicit local two-site basis.
    :param tensor_matvec: Callable acting on rank-4 ``NonabelianTensor`` input.
    :param reduced_matvec: Callable acting on ``ReducedStateVector`` input.
    :param packed_matvec: Callable acting on packed vector input.
    :param diag: Packed diagonal aligned with ``basis``.
    :param identity_like: Whether the compiled operator is exactly identity.
    :param name: Optional diagnostic operator name.
    :param metadata: Optional source metadata propagated to ``LocalOperator``.
    :param local_operator_table: Optional typed renormalized local table
        provider that owns these actions.
    """

    basis: object
    tensor_matvec: object
    reduced_matvec: object
    packed_matvec: object
    diag: object
    identity_like: bool = False
    name: str | None = None
    metadata: dict | None = None
    local_operator_table: object | None = None

    @property
    def packed_block_matrices(self):
        """
        Return the packed block-matrix provider used by preconditioners.

        :returns: Provider attached to the packed matvec, or ``None``.
        """

        return getattr(self.packed_matvec, "block_matrices", None)

    def as_tuple(self):
        """
        Return the legacy local-action tuple.

        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec, diag,
            identity_like)``.
        """

        return (
            self.tensor_matvec,
            self.reduced_matvec,
            self.packed_matvec,
            self.diag,
            self.identity_like,
        )

    def to_local_operator(self):
        """
        Convert to the solver-facing ``LocalOperator`` container.

        :returns: ``LocalOperator`` with all compiled actions and metadata.
        """

        return LocalOperator(
            tensor_matvec=self.tensor_matvec,
            aux_reduced_matvec=self.reduced_matvec,
            aux_packed_matvec=self.packed_matvec,
            packed_block_matrices=self.packed_block_matrices,
            basis=self.basis,
            diag=self.diag,
            name=self.name,
            identity_like=self.identity_like,
            metadata=self.metadata,
            local_operator_table=self.local_operator_table,
        )


def apply_transition_tensor(transitions, theta, out_entries, *, base_dtype):
    """
    Apply uncompiled transition kernels to a rank-4 two-site tensor.

    :param transitions: Mapping from input sector key to ``(out_idx, kernel)``
        transition tuples.
    :param theta: Rank-4 two-site input tensor.
    :param out_entries: Output ``(sector_key, block_shape)`` descriptors.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Output ``NonabelianTensor`` with the same external metadata.
    """

    if not isinstance(theta, NonabelianTensor) or theta.rank != 4:
        raise ValueError("apply_transition_tensor expects a rank-4 NonabelianTensor.")

    dtype = np.result_type(
        base_dtype,
        *(np.asarray(block).dtype for block in theta.data.values()),
    )
    out_blocks = [
        np.zeros(shape, dtype=dtype)
        for _key, shape in out_entries
    ]

    for in_key, in_block in theta.data.items():
        block_in = np.asarray(in_block)
        vec_in = block_in.reshape(-1)
        for out_idx, kernel in transitions.get(in_key, ()):
            _out_key, out_shape = out_entries[out_idx]
            contrib = (np.asarray(kernel) @ vec_in).reshape(out_shape)
            out_blocks[out_idx] += contrib

    out_data = {
        key: block
        for (key, _shape), block in zip(out_entries, out_blocks)
    }

    return NonabelianTensor(
        out_data,
        [leg[:] for leg in theta.qns],
        theta.dirs[:],
        fusion_legs=theta.fusion_legs[:],
        metadata=theta.metadata.copy(),
    )


def apply_transition_reduced(transitions, state, out_entries, *, base_dtype):
    """
    Apply uncompiled transition kernels to a reduced local state vector.

    :param transitions: Mapping from input sector key to ``(out_idx, kernel)``
        transition tuples.
    :param state: Reduced local input state.
    :param out_entries: Output ``(sector_key, block_shape)`` descriptors.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Reduced output state.
    """

    if not isinstance(state, ReducedStateVector):
        raise TypeError("apply_transition_reduced expects a ReducedStateVector.")

    dtype = np.result_type(
        base_dtype,
        *(np.asarray(block).dtype for block in state.blocks.values()),
    )
    out_blocks = {
        key: np.zeros(shape, dtype=dtype)
        for key, shape in out_entries
    }

    for in_key, in_block in state.blocks.items():
        block_in = np.asarray(in_block)
        vec_in = block_in.reshape(-1)
        for out_idx, kernel in transitions.get(in_key, ()):
            out_key, out_shape = out_entries[out_idx]
            contrib = (np.asarray(kernel) @ vec_in).reshape(out_shape)
            out_blocks[out_key] += contrib

    return ReducedStateVector(layout=state.layout, blocks=out_blocks)


def apply_compiled_transition_reduced(compiled_transitions, state, *, base_dtype):
    """
    Apply compiled packed transitions to a reduced state vector.

    :param compiled_transitions: Basis-aware compiled transition kernels.
    :param state: Reduced local state in the same two-site basis.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Reduced state vector with blocks assembled from the compiled basis.
    """

    return compiled_transitions.apply_reduced(state, base_dtype=base_dtype)


@dataclass
class PackedOutputSegment:
    """
    Contiguous output slice touched by one compiled packed block action.

    :param offset: Start offset in the packed two-site basis.
    :param size: Number of packed coefficients in the segment.
    """

    offset: int
    size: int

    def __post_init__(self):
        self.offset = int(self.offset)
        self.size = int(self.size)

    def add_flat(self, vector, values, *, cursor=0):
        """
        Add this segment's slice from a stacked contribution vector.

        :param vector: Packed output vector to update in-place.
        :param values: Stacked contribution vector.
        :param cursor: Start row in ``values`` for this segment.
        :returns: Cursor immediately after this segment.
        """

        next_cursor = int(cursor) + self.size
        vector[self.offset:self.offset + self.size] += values[int(cursor):next_cursor]
        return next_cursor


@dataclass
class CompiledPackedBlock:
    """
    Compiled action from one input sector block to one or more output segments.

    :param input_entry: Input ``LocalLayoutEntry`` in the local two-site basis.
    :param kernel: Dense stacked kernel for all output segments.
    :param output_segments: Contiguous output packed segments matching rows of
        ``kernel``.
    """

    input_entry: object
    kernel: np.ndarray
    output_segments: tuple[PackedOutputSegment, ...]

    @property
    def input_offset(self):
        return int(self.input_entry.offset)

    @property
    def input_size(self):
        return int(self.input_entry.size)

    @property
    def kernel_dtype(self):
        return self.kernel.dtype

    def apply_packed(self, vector, out):
        """
        Apply this block action to one packed input vector.

        :param vector: Packed input vector.
        :param out: Packed output vector updated in-place.
        :returns: The modified output vector.
        """

        vec_in = np.asarray(vector)[self.input_offset:self.input_offset + self.input_size]
        contrib = self.kernel @ vec_in
        cursor = 0
        for segment in self.output_segments:
            cursor = segment.add_flat(out, contrib, cursor=cursor)
        return out

    def add_to_dense(self, matrix):
        """
        Add this block action to a dense matrix representation.

        :param matrix: Dense packed operator matrix updated in-place.
        :returns: The modified matrix.
        """

        cursor = 0
        for segment in self.output_segments:
            matrix[
                segment.offset:segment.offset + segment.size,
                self.input_offset:self.input_offset + self.input_size,
            ] = self.kernel[cursor:cursor + segment.size]
            cursor += segment.size
        return matrix

    def add_to_csr_triplets(self, data, rows, cols, *, dtype):
        """
        Append sparse triplets for this block action.

        :param data: Sparse data list updated in-place.
        :param rows: Sparse row-index list updated in-place.
        :param cols: Sparse column-index list updated in-place.
        :param dtype: Output sparse dtype.
        """

        cursor = 0
        for segment in self.output_segments:
            block = np.asarray(self.kernel[cursor:cursor + segment.size, :], dtype=dtype)
            nz_rows, nz_cols = np.nonzero(np.abs(block) > 0.0)
            if nz_rows.size:
                rows.extend((nz_rows + segment.offset).tolist())
                cols.extend((nz_cols + self.input_offset).tolist())
                data.extend(block[nz_rows, nz_cols].tolist())
            cursor += segment.size

    def add_to_reduced_blocks(self, in_block, out_blocks, *, offset_to_out_idx, out_entries):
        """
        Add this block action to reduced output block storage.

        :param in_block: Input sector block.
        :param out_blocks: Mutable output block list aligned with ``out_entries``.
        :param offset_to_out_idx: Packed-offset to output-entry index mapping.
        :param out_entries: Output ``(key, shape)`` descriptors.
        """

        contrib = self.kernel @ np.asarray(in_block).reshape(self.input_size)
        cursor = 0
        for segment in self.output_segments:
            remaining = segment.size
            local_offset = int(segment.offset)
            while remaining > 0:
                out_idx = offset_to_out_idx[local_offset]
                _out_key, out_shape = out_entries[out_idx]
                out_entry_size = int(np.prod(out_shape, dtype=int))
                piece = contrib[cursor:cursor + out_entry_size].reshape(out_shape)
                existing = out_blocks[out_idx]
                out_blocks[out_idx] = piece if existing is None else existing + piece
                cursor += out_entry_size
                local_offset += out_entry_size
                remaining -= out_entry_size


@dataclass
class CompiledPackedTransitions:
    """
    Basis-aware compiled packed transition kernels.

    :param basis: Explicit local two-site basis.
    :param items: Per-input-entry packed kernels.
    :param offset_to_out_idx: Mapping from packed output offset to basis index.
    :param diagonal_blocks: Per-entry diagonal block matrices for preconditioning.
    """

    basis: object
    items: tuple
    offset_to_out_idx: dict
    diagonal_blocks: tuple

    @property
    def total_dim(self):
        return self.basis.size

    @property
    def out_entries(self):
        """Return ``(sector_key, block_shape)`` entries from the two-site basis."""

        return self.basis.out_entries

    @property
    def kernel_dtypes(self):
        """Return dtypes of all present compiled block kernels."""

        return tuple(item.kernel_dtype for item in self.items if item is not None)

    @property
    def block_matrices(self):
        """
        Return basis-aligned diagonal block matrices for preconditioning.

        :returns: Tuple aligned with ``basis.entries``. Entries are ``None``
            when the compiled local operator has no self-block for that sector.
        """

        return self.diagonal_blocks

    def block_matrix_for(self, entry_or_key):
        """
        Return the diagonal block matrix for a basis entry or sector key.

        :param entry_or_key: ``LocalLayoutEntry`` or sector key.
        :returns: Matching diagonal block matrix or ``None``.
        """

        entry = (
            entry_or_key
            if hasattr(entry_or_key, "key")
            else self.basis.entry_for_key(entry_or_key)
        )
        return self.block_matrices[self.basis.entry_index(entry.key)]

    def apply_packed(self, vector, *, base_dtype):
        """
        Apply the compiled transition operator to a packed vector.

        :param vector: Packed local input vector.
        :param base_dtype: Scalar dtype contribution from the surrounding operator.
        :returns: Packed local output vector.
        """

        vec = np.asarray(vector)
        out = np.zeros(int(self.total_dim), dtype=np.result_type(base_dtype, vec.dtype))
        for item in self.items:
            if item is not None:
                item.apply_packed(vec, out)
        return out

    def apply_tensor(self, theta, *, base_dtype):
        """
        Apply the compiled transition operator to a two-site tensor.

        :param theta: Rank-4 two-site tensor in a basis compatible with this
            compiled operator.
        :param base_dtype: Scalar dtype contribution from the surrounding
            effective operator.
        :returns: Output ``NonabelianTensor`` with the same external metadata.
        """

        if not isinstance(theta, NonabelianTensor) or theta.rank != 4:
            raise ValueError("CompiledPackedTransitions.apply_tensor expects a rank-4 NonabelianTensor.")
        if not self.basis.compatible_with_layout(pack_two_site_state(theta, layout=self.basis)[1]):
            raise ValueError("Two-site tensor layout is incompatible with the compiled two-site basis.")

        dtype = np.result_type(
            base_dtype,
            *(np.asarray(block).dtype for block in theta.data.values()),
        )
        out_blocks = [
            np.zeros(shape, dtype=dtype)
            for _key, shape in self.out_entries
        ]
        for item in self.items:
            if item is None:
                continue
            in_block = theta.data.get(item.input_entry.key)
            if in_block is None:
                continue
            item.add_to_reduced_blocks(
                in_block,
                out_blocks,
                offset_to_out_idx=self.offset_to_out_idx,
                out_entries=self.out_entries,
            )

        out_data = {
            key: block
            for (key, _shape), block in zip(self.out_entries, out_blocks)
        }
        return NonabelianTensor(
            out_data,
            [leg[:] for leg in theta.qns],
            theta.dirs[:],
            fusion_legs=theta.fusion_legs[:],
            metadata=theta.metadata.copy(),
        )

    def apply_reduced(self, state, *, base_dtype):
        """
        Apply the compiled transition operator to a reduced state vector.

        :param state: Reduced local input state.
        :param base_dtype: Scalar dtype contribution from the surrounding operator.
        :returns: Reduced local output state.
        """

        if not isinstance(state, ReducedStateVector):
            raise TypeError("CompiledPackedTransitions.apply_reduced expects a ReducedStateVector.")
        if not self.basis.compatible_with_layout(state.layout.entries):
            raise ValueError("Reduced state layout is incompatible with the compiled two-site basis.")

        out_entries = self.out_entries
        out_blocks = [None] * len(out_entries)
        for item in self.items:
            if item is None:
                continue
            in_block = state.blocks.get(item.input_entry.key)
            if in_block is None:
                continue
            item.add_to_reduced_blocks(
                in_block,
                out_blocks,
                offset_to_out_idx=self.offset_to_out_idx,
                out_entries=out_entries,
            )

        blocks = {}
        for (key, _shape), block in zip(out_entries, out_blocks):
            if block is not None and np.linalg.norm(np.asarray(block).reshape(-1)) > 1.0e-15:
                blocks[key] = block
        return ReducedStateVector(layout=state.layout, blocks=blocks)

    def materialize_dense(self, *, dtype=None):
        """
        Materialize the compiled transition operator as a dense matrix.

        :param dtype: Optional output dtype.
        :returns: Dense packed operator matrix.
        """

        if dtype is None:
            dtype = np.result_type(*(self.kernel_dtypes or (float,)))
        matrix = np.zeros((int(self.total_dim), int(self.total_dim)), dtype=dtype)
        for item in self.items:
            if item is not None:
                item.add_to_dense(matrix)
        return matrix

    def materialize_csr(self, *, dtype=None):
        """
        Materialize the compiled transition operator as a CSR sparse matrix.

        :param dtype: Optional output dtype.
        :returns: CSR matrix, or ``None`` when SciPy is unavailable.
        """

        if sp is None:
            return None
        if dtype is None:
            dtype = np.result_type(*(self.kernel_dtypes or (float,)))
        data = []
        rows = []
        cols = []
        for item in self.items:
            if item is not None:
                item.add_to_csr_triplets(data, rows, cols, dtype=dtype)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(int(self.total_dim), int(self.total_dim)),
            dtype=dtype,
        )

    def packed_matvec(
        self,
        *,
        base_dtype,
        dense_threshold=_PACKED_DENSE_LOCAL_DIM,
        csr_threshold=_PACKED_CSR_LOCAL_DIM,
    ):
        """
        Build the preferred packed-vector matvec for this compiled operator.

        Dense materialization is disabled by default for DMRG local solves.
        Large local operators use CSR when SciPy is available, and the
        remaining cases use the compiled block loop directly.

        :param base_dtype: Scalar dtype contribution from the surrounding
            effective operator.
        :param dense_threshold: Maximum packed dimension for dense
            materialization.  Defaults to zero so normal SU(2) DMRG uses
            matrix-free compiled actions.
        :param csr_threshold: Minimum packed dimension for CSR materialization.
        :returns: Callable packed-vector matvec carrying backend metadata.
        """

        if self.total_dim <= int(dense_threshold):
            packed_matrix = self.materialize_dense(dtype=base_dtype)

            def packed_apply(vector):
                return packed_matrix @ np.asarray(vector)

            packed_apply.backend = "compiled-dense"
            packed_apply.matrix = packed_matrix
        elif sp is not None and self.total_dim >= int(csr_threshold):
            packed_csr = self.materialize_csr(dtype=base_dtype)

            def packed_apply(vector):
                return np.asarray(packed_csr @ np.asarray(vector))

            packed_apply.backend = "compiled-csr"
            packed_apply.matrix = packed_csr
        else:
            def packed_apply(vector):
                return self.apply_packed(vector, base_dtype=base_dtype)

            packed_apply.backend = "compiled"

        packed_apply.basis = self.basis
        packed_apply.compiled_transitions = self
        packed_apply.block_matrices = self
        return packed_apply


def compile_packed_transitions(transitions, basis):
    """
    Compile sector-keyed transition kernels into a basis-aware block operator.

    :param transitions: Mapping from input sector key to ``(out_idx, kernel)``
        transition tuples.
    :param basis: Explicit local two-site basis.
    :returns: ``CompiledPackedTransitions`` for packed/tensor/reduced matvecs.
    """

    compiled_items = []
    offset_to_out_idx = {}
    diagonal_blocks = []
    for out_idx, entry in enumerate(basis):
        offset_to_out_idx[int(entry.offset)] = out_idx
    for in_idx, in_entry in enumerate(basis):
        terms = transitions.get(in_entry.key, ())
        if not terms:
            compiled_items.append(None)
            diagonal_blocks.append(None)
            continue
        grouped = {}
        for out_idx, kernel in terms:
            arr = np.asarray(kernel)
            if out_idx in grouped:
                grouped[out_idx] = grouped[out_idx] + arr
            else:
                grouped[out_idx] = arr
        ordered = sorted(grouped.items(), key=lambda item: basis[item[0]].offset)
        if not ordered:
            compiled_items.append(None)
            diagonal_blocks.append(None)
            continue
        self_kernel = grouped.get(in_idx)
        if self_kernel is not None:
            diagonal_blocks.append(np.ascontiguousarray(self_kernel))
        else:
            diagonal_blocks.append(None)
        kernels = []
        segments = []
        current_offset = None
        current_size = 0
        for out_idx, kernel in ordered:
            out_entry = basis[out_idx]
            kernels.append(np.asarray(kernel))
            if current_offset is None:
                current_offset = int(out_entry.offset)
                current_size = int(out_entry.size)
            elif current_offset + current_size == int(out_entry.offset):
                current_size += int(out_entry.size)
            else:
                segments.append(PackedOutputSegment(current_offset, current_size))
                current_offset = int(out_entry.offset)
                current_size = int(out_entry.size)
        if current_offset is not None:
            segments.append(PackedOutputSegment(current_offset, current_size))
        kernel_matrix = np.ascontiguousarray(np.vstack(kernels))
        compiled_items.append(
            CompiledPackedBlock(
                input_entry=in_entry,
                kernel=kernel_matrix,
                output_segments=tuple(segments),
            )
        )
    return CompiledPackedTransitions(
        basis=basis,
        items=tuple(compiled_items),
        offset_to_out_idx=offset_to_out_idx,
        diagonal_blocks=tuple(diagonal_blocks),
    )


def apply_packed_transitions(transitions, vector, basis, *, base_dtype):
    """
    Apply uncompiled transition kernels to a packed local vector.

    :param transitions: Mapping from input sector key to ``(out_idx, kernel)``
        transition tuples.
    :param vector: Packed input vector.
    :param basis: Explicit local two-site basis.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Packed output vector.
    """

    vec = np.asarray(vector)
    dtype = np.result_type(base_dtype, vec.dtype)
    out = np.zeros(basis.size, dtype=dtype)

    for in_entry, block_in in basis.iter_packed_blocks(vec, drop_zeros=False):
        vec_in = np.asarray(block_in).reshape(in_entry.size)
        for out_idx, kernel in transitions.get(in_entry.key, ()):
            out_entry = basis[out_idx]
            basis.add_packed_block(out, out_entry, np.asarray(kernel) @ vec_in)
    return out


def materialize_packed_matrix(compiled_transitions, *, dtype=None):
    """
    Materialize compiled packed transitions as a dense matrix.

    :param compiled_transitions: ``CompiledPackedTransitions`` provider.
    :param dtype: Optional output dtype.
    :returns: Dense packed matrix.
    """

    return compiled_transitions.materialize_dense(dtype=dtype)


def materialize_packed_csr(compiled_transitions, *, dtype=None):
    """
    Materialize compiled packed transitions as a CSR sparse matrix.

    :param compiled_transitions: ``CompiledPackedTransitions`` provider.
    :param dtype: Optional output dtype.
    :returns: CSR matrix, or ``None`` when SciPy is unavailable.
    """

    return compiled_transitions.materialize_csr(dtype=dtype)


def apply_compiled_packed_transitions(compiled_transitions, vector, *, base_dtype):
    """
    Apply compiled packed transitions to a packed local vector.

    :param compiled_transitions: ``CompiledPackedTransitions`` provider.
    :param vector: Packed input vector.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Packed output vector.
    """

    return compiled_transitions.apply_packed(vector, base_dtype=base_dtype)


@dataclass
class CompiledFactorizedBlock:
    """
    Batched factorized action into one output sector block.

    :param input_entry: Input ``LocalLayoutEntry`` in the local two-site basis.
    :param output_entry: Output ``LocalLayoutEntry`` in the local two-site basis.
    :param left_stack: Batched left effective factors.
    :param right_stack: Batched right effective factors.
    """

    input_entry: object
    output_entry: object
    left_stack: np.ndarray
    right_stack: np.ndarray
    family_names: tuple = ()

    def __post_init__(self):
        self._use_direct_contraction = self._estimate_direct_contraction()
        self._output_size = int(self.output_entry.size)
        self._output_slice = slice(
            int(self.output_entry.offset),
            int(self.output_entry.offset) + int(self.output_entry.size),
        )
        self.family_names = tuple(sorted({str(name) for name in self.family_names}))
        self._two_step_matrix_cache = None

    @property
    def input_shape(self):
        return self.input_entry.shape

    @property
    def input_size(self):
        return int(self.input_entry.size)

    @property
    def output_offset(self):
        return int(self.output_entry.offset)

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self.output_entry.shape

    def _two_step_matrices(self):
        cached = self._two_step_matrix_cache
        if cached is None:
            cached = _factorized_block_two_step_matrices(
                self.left_stack,
                self.right_stack,
            )
            self._two_step_matrix_cache = cached
        return cached

    def kernel_matrix(self, input_shape, *, max_elements=_FACTORIZED_BLOCK_DENSE_KERNEL_MAX_ELEMENTS):
        """
        Return a cached dense block kernel when the block is small enough.

        :param input_shape: Shape of the input sector block.
        :param max_elements: Maximum dense matrix elements to cache.
        :returns: Dense ``(output_size, input_size)`` kernel, or ``None``.
        """

        input_size = int(np.prod(tuple(int(dim) for dim in input_shape), dtype=int))
        output_size = int(self.output_size)
        if output_size * input_size > int(max_elements):
            return None
        cache_key = (tuple(int(dim) for dim in input_shape),)
        cached = getattr(self, "_kernel_matrix_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        kernel = np.einsum(
            "tlkwab,twqrdc->ladqkbcr",
            np.asarray(self.left_stack),
            np.asarray(self.right_stack),
            optimize=False,
        ).reshape(output_size, input_size)
        self._kernel_matrix_cache = (cache_key, np.ascontiguousarray(kernel))
        return self._kernel_matrix_cache[1]

    def _estimate_direct_contraction(self):
        """
        Decide whether a direct three-operand contraction is cheaper.

        :returns: ``True`` when the direct contraction has no higher scalar
            operation count than the two-step contraction.
        """

        t, ldim, kdim, wdim, adim, bdim = (
            int(dim) for dim in np.asarray(self.left_stack).shape
        )
        _t2, _w2, qdim, rdim_right, ddim, cdim_right = (
            int(dim) for dim in np.asarray(self.right_stack).shape
        )
        k_in, b_in, c_in, r_in = self.input_shape
        direct_cost = ldim * adim * ddim * qdim * t * wdim * k_in * b_in * c_in * r_in
        first_cost = t * ldim * wdim * adim * c_in * r_in * k_in * b_in
        second_cost = ldim * adim * ddim * qdim * t * wdim * c_in * r_in
        return bool(direct_cost <= first_cost + second_cost)

    def apply_packed(self, block_in, out, basis):
        """
        Apply this batched factorized block to one input block.

        :param block_in: Input sector block.
        :param out: Packed output vector updated in-place.
        :param basis: Local two-site basis owning the packed output.
        :returns: The modified packed output vector.
        """

        left_stack = np.asarray(self.left_stack)
        right_stack = np.asarray(self.right_stack)
        block_in = np.asarray(block_in)
        if _FACTORIZED_BLOCK_DENSE_KERNEL_MAX_ELEMENTS > 0:
            kernel = self.kernel_matrix(block_in.shape)
            if kernel is not None:
                contrib = kernel @ block_in.reshape(-1)
                out[self._output_slice] += np.asarray(contrib).reshape(self._output_size)
                return out
        if self._use_direct_contraction:
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                left_stack,
                block_in,
                right_stack,
                optimize=False,
            )
            out[self._output_slice] += np.asarray(contrib).reshape(self._output_size)
            return out
        contrib = _apply_factorized_block_two_step(
            left_stack,
            block_in,
            right_stack,
            matrices=self._two_step_matrices(),
        )
        out[self._output_slice] += np.asarray(contrib).reshape(self._output_size)
        return out

    def apply_block(self, block_in):
        """
        Apply this factorized block and return only the output block.

        This is the component-direct kernel used by orthonormalized local
        solvers.  It avoids scattering through the full packed parent vector
        when the caller already owns component-local input/output buffers.

        :param block_in: Input sector block.
        :returns: Flattened output-sector contribution.
        """

        left_stack = np.asarray(self.left_stack)
        right_stack = np.asarray(self.right_stack)
        block_in = np.asarray(block_in)
        if _FACTORIZED_BLOCK_DENSE_KERNEL_MAX_ELEMENTS > 0:
            kernel = self.kernel_matrix(block_in.shape)
            if kernel is not None:
                return np.asarray(kernel @ block_in.reshape(-1)).reshape(self._output_size)
        if self._use_direct_contraction:
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                left_stack,
                block_in,
                right_stack,
                optimize=False,
            )
            return np.asarray(contrib).reshape(self._output_size)
        contrib = _apply_factorized_block_two_step(
            left_stack,
            block_in,
            right_stack,
            matrices=self._two_step_matrices(),
        )
        return np.asarray(contrib).reshape(self._output_size)


@dataclass
class CompiledFactorizedTerms:
    """
    Basis-aware compiled factorized packed kernels.

    :param basis: Explicit local two-site basis.
    :param items: Per-input-entry factorized terms.
    """

    basis: object
    items: tuple

    @property
    def total_dim(self):
        return self.basis.size

    @property
    def family_names(self):
        """Return sorted complementary/qchem family labels carried by terms."""

        return tuple(
            sorted(
                {
                    str(name)
                    for terms in self.items
                    for term in terms
                    for name in getattr(term, "family_names", ())
                }
            )
        )

    @property
    def family_term_counts(self):
        """Return compiled block counts by family label."""

        counts = {}
        for terms in self.items:
            for term in terms:
                names = tuple(getattr(term, "family_names", ()) or ("unlabeled",))
                for name in names:
                    counts[str(name)] = counts.get(str(name), 0) + 1
        return dict(sorted(counts.items()))

    @property
    def block_matrices(self):
        """
        Return basis-aligned self-block matrices for preconditioning.

        :returns: Tuple aligned with ``basis.entries``. Entries are ``None``
            when no factorized term maps the sector block to itself.
        """

        cached = getattr(self, "_block_matrices_cache", None)
        if cached is not None:
            return cached
        blocks = tuple(self.block_matrix_for(entry) for entry in self.basis)
        self._block_matrices_cache = blocks
        return blocks

    def block_matrix_for(self, entry_or_key):
        """
        Return the self-block matrix for a basis entry or sector key.

        :param entry_or_key: ``LocalLayoutEntry`` or sector key.
        :returns: Dense self-block matrix, or ``None`` when absent.
        """

        entry = (
            entry_or_key
            if hasattr(entry_or_key, "key")
            else self.basis.entry_for_key(entry_or_key)
        )
        in_idx = self.basis.entry_index(entry.key)
        block = None
        for term in self.items[in_idx]:
            if term.output_entry.key != entry.key or term.output_entry.shape != entry.shape:
                continue
            kernel = np.einsum(
                "tlkwab,twqrdc->ladqkbcr",
                np.asarray(term.left_stack),
                np.asarray(term.right_stack),
                optimize=False,
            ).reshape(entry.size, entry.size)
            block = kernel if block is None else block + kernel
        return block

    def materialize_dense(self, *, dtype=None):
        """
        Materialize the factorized local operator as a dense packed matrix.

        :param dtype: Optional matrix dtype.
        :returns: Dense matrix aligned with the local two-site basis.
        """

        cached = getattr(self, "_dense_matrix_cache", None)
        if cached is not None and (dtype is None or cached.dtype == np.dtype(dtype)):
            return cached
        if dtype is None:
            dtype = np.result_type(
                *[
                    arr.dtype
                    for terms in self.items
                    for term in terms
                    for arr in (np.asarray(term.left_stack), np.asarray(term.right_stack))
                ],
                float,
            )
        matrix = np.zeros((self.total_dim, self.total_dim), dtype=dtype)
        for in_entry, terms in zip(self.basis, self.items):
            in_slice = slice(in_entry.offset, in_entry.offset + in_entry.size)
            for term in terms:
                out_entry = term.output_entry
                out_slice = slice(out_entry.offset, out_entry.offset + out_entry.size)
                kernel = np.einsum(
                    "tlkwab,twqrdc->ladqkbcr",
                    np.asarray(term.left_stack, dtype=dtype),
                    np.asarray(term.right_stack, dtype=dtype),
                    optimize=True,
                ).reshape(out_entry.size, in_entry.size)
                matrix[out_slice, in_slice] += kernel
        self._dense_matrix_cache = matrix
        return matrix

    def apply_packed(self, vector, *, base_dtype):
        """
        Apply the compiled factorized operator to a packed vector.

        :param vector: Packed local input vector.
        :param base_dtype: Scalar dtype contribution from the surrounding operator.
        :returns: Packed local output vector.
        """

        vec = np.asarray(vector)
        out = np.zeros(int(self.total_dim), dtype=np.result_type(base_dtype, vec.dtype))
        for in_entry, terms in zip(self.basis, self.items):
            if not terms:
                continue
            piece = vec[in_entry.slice]
            if not np.any(piece):
                continue
            block_in = piece.reshape(in_entry.shape)
            for term in terms:
                term.apply_packed(block_in, out, self.basis)
        return out

    def packed_matvec(self, *, base_dtype, backend, out_entries=None, block_matrices=None):
        """
        Build a packed-vector matvec for this compiled factorized operator.

        :param base_dtype: Scalar dtype contribution from the surrounding
            effective operator.
        :param backend: Backend label attached to the returned callable.
        :param out_entries: Optional basis output descriptors for diagnostics.
        :param block_matrices: Optional diagonal block-matrix provider.
        :returns: Callable packed-vector matvec carrying solver metadata.
        """

        dense_matrix = None
        if self.total_dim <= _FACTORIZED_DENSE_LOCAL_DIM:
            dense_matrix = self.materialize_dense(dtype=base_dtype)

        def packed_apply(vector):
            if dense_matrix is not None:
                return dense_matrix @ np.asarray(vector)
            return self.apply_packed(vector, base_dtype=base_dtype)

        packed_apply.backend = (
            f"{backend}-dense-matrix"
            if dense_matrix is not None
            else backend
        )
        packed_apply.basis = self.basis
        packed_apply.compiled_factorized_terms = self
        packed_apply.out_entries = out_entries
        packed_apply.block_matrices = block_matrices
        packed_apply.dense_matrix = dense_matrix
        packed_apply.family_names = self.family_names
        packed_apply.family_term_counts = self.family_term_counts
        return packed_apply


def compile_factorized_terms(factorized_terms, basis):
    """
    Compile factorized two-site terms into batched packed block kernels.

    :param factorized_terms: Mapping from input sector key to
        ``(out_idx, left_factor, right_factor)`` tuples.
    :param basis: Explicit local two-site basis.
    :returns: ``CompiledFactorizedTerms`` for packed factorized matvecs.
    """

    compiled_items = []
    for in_entry in basis:
        terms = factorized_terms.get(in_entry.key, ())
        if not terms:
            compiled_items.append(())
            continue
        grouped = {}
        for term in terms:
            out_idx, left_factor, right_factor = term[:3]
            family_names = term[3] if len(term) > 3 else ()
            left_arr = np.asarray(left_factor)
            right_arr = np.asarray(right_factor)
            shape_key = (
                out_idx,
                tuple(left_arr.shape),
                tuple(right_arr.shape),
            )
            bucket = grouped.setdefault(shape_key, {"left": [], "right": []})
            bucket["left"].append(left_arr)
            bucket["right"].append(right_arr)
            bucket.setdefault("families", []).append(family_names)
        compiled_terms = []
        for shape_key in sorted(grouped, key=lambda key: basis[key[0]].offset):
            out_idx = shape_key[0]
            out_entry = basis[out_idx]
            bucket = grouped[shape_key]
            family_names = tuple(
                sorted(
                    {
                        str(name)
                        for item in bucket.get("families", ())
                        for name in (
                            item if isinstance(item, (tuple, list, set)) else (item,)
                        )
                        if name is not None
                    }
                )
            )
            compiled_terms.append(
                CompiledFactorizedBlock(
                    input_entry=in_entry,
                    output_entry=out_entry,
                    left_stack=np.ascontiguousarray(np.stack(bucket["left"], axis=0)),
                    right_stack=np.ascontiguousarray(np.stack(bucket["right"], axis=0)),
                    family_names=family_names,
                )
            )
        compiled_items.append(tuple(compiled_terms))
    return CompiledFactorizedTerms(
        basis=basis,
        items=tuple(compiled_items),
    )


def diagonal_from_factorized_terms(factorized_terms, basis, *, dtype=float):
    """
    Build a packed diagonal from factorized local terms.

    :param factorized_terms: Mapping from input sector key to
        ``(out_idx, left_factor, right_factor)`` tuples.
    :param basis: Explicit local two-site basis.
    :param dtype: Working dtype for the block diagonal accumulation.
    :returns: Real packed diagonal aligned with ``basis``.
    """

    diag = np.zeros(basis.size, dtype=float)
    entry_index = basis.index_by_key()
    for in_entry in basis:
        terms = factorized_terms.get(in_entry.key, ())
        if not terms:
            continue
        diag_block = np.zeros(in_entry.shape, dtype=dtype)
        in_idx = entry_index[in_entry.key]
        for term in terms:
            out_idx, left_factor, right_factor = term[:3]
            if out_idx != in_idx:
                continue
            diag_block += np.einsum(
                "llwaa,wrrcc->lacr",
                np.asarray(left_factor),
                np.asarray(right_factor),
                optimize=True,
            )
        diag[in_entry.offset:in_entry.offset + in_entry.size] = np.real(
            diag_block
        ).reshape(in_entry.size)
    return diag


def transitions_are_identity_operator(basis, transitions, *, tol=1e-12):
    """
    Test whether transition kernels represent an exact identity operator.

    :param basis: Explicit local two-site basis.
    :param transitions: Mapping from input sector key to ``(out_idx, kernel)``
        transition tuples.
    :param tol: Absolute and relative tolerance for identity comparison.
    :returns: ``True`` if all transitions are identity self-blocks only.
    """

    for entry in basis:
        terms = transitions.get(entry.key, ())
        diag_found = False
        for out_idx, kernel in terms:
            out_entry = basis[out_idx]
            arr = np.asarray(kernel)
            if out_entry.key == entry.key and out_entry.shape == entry.shape:
                eye = np.eye(entry.size, dtype=arr.dtype)
                if np.allclose(arr, eye, atol=tol, rtol=tol):
                    if diag_found:
                        return False
                    diag_found = True
                elif np.linalg.norm(arr) > tol:
                    return False
            elif np.linalg.norm(arr) > tol:
                return False
        if not diag_found:
            return False
    return True


def identity_env_to_matrix(block, *, dtype):
    """
    Normalize an identity-MPO environment block to a rank-2 matrix.

    :param block: Environment block array.
    :param dtype: Matrix dtype.
    :returns: Rank-2 environment matrix.
    :raises ValueError: If the block shape is incompatible.
    """

    arr = np.asarray(block, dtype=dtype)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        "Identity-MPO local actions expect rank-2 environment blocks or "
        f"rank-3 blocks with leading dimension 1, got {arr.shape!r}."
    )


def identity_mpo_transitions(E_map, F_map, basis, *, base_dtype):
    """
    Build local transition kernels for an identity two-site MPO.

    :param E_map: Left environment block map.
    :param F_map: Right environment block map.
    :param basis: Explicit local two-site basis.
    :param base_dtype: Scalar dtype for generated kernels.
    :returns: ``(out_entries, transitions)`` for local action construction.
    """

    transitions = {}
    out_entries = basis.out_entries
    out_index = basis.index_by_key()
    kernel_cache = {}

    for in_entry in basis:
        q_lk, q_p1k, q_p2k, q_rk = in_entry.key
        in_transitions = []
        eye_p1 = np.eye(int(in_entry.shape[1]), dtype=base_dtype)
        eye_p2 = np.eye(int(in_entry.shape[2]), dtype=base_dtype)
        for q_lb, q_lk_again in E_map:
            if q_lk_again != q_lk:
                continue
            E_block = identity_env_to_matrix(E_map[(q_lb, q_lk)], dtype=base_dtype)
            for q_rb, q_rk_again in F_map:
                if q_rk_again != q_rk:
                    continue
                out_key = (q_lb, q_p1k, q_p2k, q_rb)
                out_idx = out_index.get(out_key)
                if out_idx is None:
                    continue
                F_block = identity_env_to_matrix(F_map[(q_rb, q_rk)], dtype=base_dtype)
                out_entry = basis[out_idx]
                kernel_key = (
                    id(E_block),
                    id(F_block),
                    tuple(int(dim) for dim in in_entry.shape),
                    tuple(int(dim) for dim in out_entry.shape),
                )
                kernel = kernel_cache.get(kernel_key)
                if kernel is None:
                    kernel = np.einsum(
                        "al,po,qr,bc->apqblorc",
                        E_block,
                        eye_p1,
                        eye_p2,
                        F_block,
                        optimize=_IDENTITY_TWO_SITE_KERNEL_PATH,
                    ).reshape(
                        int(np.prod(out_entry.shape, dtype=int)),
                        int(np.prod(in_entry.shape, dtype=int)),
                    )
                    kernel_cache[kernel_key] = kernel
                in_transitions.append((out_idx, kernel))
        transitions[in_entry.key] = tuple(in_transitions)
    return out_entries, transitions


def build_identity_mpo_local_actions(E_map, F_map, basis, *, base_dtype):
    """
    Build local tensor/reduced/packed actions for an identity two-site MPO.

    :param E_map: Left environment block map.
    :param F_map: Right environment block map.
    :param basis: Explicit local two-site basis.
    :param base_dtype: Scalar dtype for generated kernels.
    :returns: ``(tensor_matvec, reduced_matvec, packed_matvec, diag,
        identity_like)``.
    """

    out_entries, transitions = identity_mpo_transitions(
        E_map,
        F_map,
        basis,
        base_dtype=base_dtype,
    )
    compiled_transitions = compile_packed_transitions(transitions, basis)

    def tensor_apply(two_site):
        return compiled_transitions.apply_tensor(two_site, base_dtype=base_dtype)

    def reduced_apply(state):
        return compiled_transitions.apply_reduced(state, base_dtype=base_dtype)

    packed_apply = compiled_transitions.packed_matvec(base_dtype=base_dtype)

    diag = np.zeros(basis.size, dtype=float)
    for entry in basis:
        q_l, _q_p1, _q_p2, q_r = entry.key
        E_block = E_map.get((q_l, q_l))
        F_block = F_map.get((q_r, q_r))
        if E_block is None or F_block is None:
            continue
        diag_left = np.real(np.diag(identity_env_to_matrix(E_block, dtype=base_dtype)))
        diag_right = np.real(np.diag(identity_env_to_matrix(F_block, dtype=base_dtype)))
        diag_block = np.einsum(
            "l,p,q,r->lpqr",
            diag_left,
            np.ones(int(entry.shape[1]), dtype=float),
            np.ones(int(entry.shape[2]), dtype=float),
            diag_right,
            optimize=True,
        )
        diag[entry.offset:entry.offset + entry.size] = diag_block.reshape(entry.size)

    identity_like = transitions_are_identity_operator(basis, transitions)
    return tensor_apply, reduced_apply, packed_apply, diag, identity_like


def apply_factorized_packed_terms(factorized_terms, vector, basis, *, base_dtype):
    """
    Apply factorized two-site terms to a packed local vector.

    ``factorized_terms`` may be a ``CompiledFactorizedTerms`` object, the
    legacy packed dictionary format, or the uncompiled sector-keyed mapping.

    :param factorized_terms: Compiled, legacy, or uncompiled factorized terms.
    :param vector: Packed input vector.
    :param basis: Explicit local two-site basis.
    :param base_dtype: Scalar dtype contribution from the surrounding operator.
    :returns: Packed output vector.
    """

    vec = np.asarray(vector)
    dtype = np.result_type(base_dtype, vec.dtype)
    if isinstance(factorized_terms, CompiledFactorizedTerms):
        return factorized_terms.apply_packed(vec, base_dtype=base_dtype)
    if isinstance(factorized_terms, dict) and "items" in factorized_terms:
        items = factorized_terms["items"]
        total_dim = int(factorized_terms["total_dim"])
    else:
        return compile_factorized_terms(factorized_terms, basis).apply_packed(
            vec,
            base_dtype=base_dtype,
        )

    out = np.zeros(total_dim, dtype=dtype)
    for (in_entry, block_in), terms in zip(basis.iter_packed_blocks(vec, drop_zeros=False), items):
        if not terms:
            continue
        for offset, size, _out_shape, left_stack, right_stack in terms:
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                np.asarray(left_stack),
                block_in,
                np.asarray(right_stack),
                optimize=_FACTORIZED_TWO_SITE_BATCH_PATH,
            )
            basis.add_packed_block(out, basis.entry_for_index(offset), np.asarray(contrib).reshape(size))
    return out
