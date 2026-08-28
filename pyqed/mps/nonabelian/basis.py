#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First-class symmetry basis descriptors for non-Abelian MPS work.

These classes are intentionally lightweight.  They make the current implicit
``qns``/packed-layout conventions explicit without changing contraction
semantics yet, giving the DMRG code a stable target for the later
renormalized-basis refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.symmetry import Sector

from .mpo import Leg


def _ordered_unique(items):
    ordered = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class BasisBlock:
    """One contiguous sector block in a symmetry-adapted basis."""

    sector: Sector
    dim: int
    offset: int

    def __post_init__(self):
        if int(self.dim) <= 0:
            raise ValueError(f"BasisBlock dimension must be positive, got {self.dim}.")
        if int(self.offset) < 0:
            raise ValueError(f"BasisBlock offset must be non-negative, got {self.offset}.")
        object.__setattr__(self, "dim", int(self.dim))
        object.__setattr__(self, "offset", int(self.offset))

    @property
    def slice(self):
        return slice(self.offset, self.offset + self.dim)


@dataclass(frozen=True)
class SymmetryBasis:
    """
    Ordered orthonormal sector basis descriptor.

    ``dims`` stores multiplicity dimensions, not SU(2) magnetic degeneracy.
    """

    sectors: tuple[Sector, ...]
    dims: dict[Sector, int]
    direction: int = 1
    name: str | None = None

    def __post_init__(self):
        sectors = _ordered_unique(self.sectors)
        dims = {sector: int(dim) for sector, dim in dict(self.dims).items()}
        if self.direction not in (-1, 1):
            raise ValueError(f"SymmetryBasis direction must be +/-1, got {self.direction!r}.")
        if not sectors:
            raise ValueError("SymmetryBasis requires at least one sector.")
        for sector in sectors:
            if sector not in dims:
                raise ValueError(f"Missing dimension for basis sector {sector!r}.")
            if dims[sector] <= 0:
                raise ValueError(f"Basis dimension for sector {sector!r} must be positive.")
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "dims", dims)

    @property
    def total_dim(self):
        return sum(self.dims[sector] for sector in self.sectors)

    @property
    def blocks(self):
        offset = 0
        out = []
        for sector in self.sectors:
            dim = self.dims[sector]
            out.append(BasisBlock(sector, dim, offset))
            offset += dim
        return tuple(out)

    def dim(self, sector):
        return self.dims[sector]

    def slices(self):
        return {block.sector: block.slice for block in self.blocks}

    def compatible_with(self, other):
        return (
            isinstance(other, SymmetryBasis)
            and self.sectors == other.sectors
            and self.dims == other.dims
            and self.direction == other.direction
        )

    def same_blocks(self, other):
        return (
            isinstance(other, SymmetryBasis)
            and self.sectors == other.sectors
            and self.dims == other.dims
        )

    def dual_compatible_with(self, other):
        return self.same_blocks(other) and self.direction == -other.direction

    @classmethod
    def from_qns(cls, qns, *, dims=None, direction=1, name=None):
        sectors = _ordered_unique(qns)
        if dims is None:
            dims = {sector: sum(1 for item in qns if item == sector) for sector in sectors}
        return cls(sectors=sectors, dims=dims, direction=direction, name=name)

    @classmethod
    def from_tensor_axis(cls, tensor, axis, *, name=None):
        axis = int(axis)
        if axis < 0:
            axis += tensor.rank
        if axis < 0 or axis >= tensor.rank:
            raise ValueError(f"Tensor axis {axis} out of range for rank-{tensor.rank} tensor.")
        dims = {}
        for key, block in tensor.data.items():
            sector = key[axis]
            dim = int(np.asarray(block).shape[axis])
            known = dims.get(sector)
            if known is None:
                dims[sector] = dim
            elif known != dim:
                raise ValueError(
                    f"Inconsistent dimension for sector {sector!r} on axis {axis}: {known} vs {dim}."
                )
        for sector in _ordered_unique(tensor.qns[axis]):
            dims.setdefault(sector, sum(1 for item in tensor.qns[axis] if item == sector))
        return cls.from_qns(
            tensor.qns[axis],
            dims=dims,
            direction=tensor.dirs[axis],
            name=name,
        )


@dataclass(frozen=True)
class SiteBasis(SymmetryBasis):
    """Physical local basis descriptor."""

    def as_physical_leg(self):
        return Leg.from_dims(self.dims, sectors=self.sectors)


@dataclass(frozen=True)
class BondBasis(SymmetryBasis):
    """Virtual renormalized-basis descriptor."""


@dataclass(frozen=True)
class LocalLayoutEntry:
    """Packed local-vector entry tied to explicit sector bases."""

    key: tuple[Sector, ...]
    shape: tuple[int, ...]
    offset: int
    size: int

    def __post_init__(self):
        object.__setattr__(self, "key", tuple(self.key))
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "size", int(self.size))
        if self.offset < 0 or self.size <= 0:
            raise ValueError("LocalLayoutEntry requires non-negative offset and positive size.")
        expected = int(np.prod(self.shape, dtype=int))
        if expected != self.size:
            raise ValueError(f"LocalLayoutEntry shape {self.shape!r} has size {expected}, not {self.size}.")

    @property
    def slice(self):
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class TwoSiteBasis:
    """
    Explicit packed basis for a two-site local DMRG problem.

    This is the bridge from the current packed layout to a future native
    renormalized-basis solver.
    """

    left: BondBasis
    phys1: SiteBasis
    phys2: SiteBasis
    right: BondBasis
    entries: tuple[LocalLayoutEntry, ...]
    intermediate: BondBasis | None = None

    @property
    def channel_resolved(self):
        """Whether entries retain the intermediate SU(2) fusion sector."""

        return self.intermediate is not None

    def __len__(self):
        """
        Return the number of packed block entries.

        :returns: Number of local sector blocks in the two-site basis.
        """

        return len(self.entries)

    def __iter__(self):
        """
        Iterate over packed local basis entries.

        :returns: Iterator over ``LocalLayoutEntry`` objects.
        """

        return iter(self.entries)

    def __getitem__(self, item):
        """
        Return a packed entry by integer index or slice.

        :param item: Entry index or slice.
        :returns: One ``LocalLayoutEntry`` or a tuple of entries.
        """

        return self.entries[item]

    @property
    def size(self):
        return sum(entry.size for entry in self.entries)

    @property
    def bases(self):
        if self.intermediate is None:
            return self.left, self.phys1, self.phys2, self.right
        return (
            self.left,
            self.phys1,
            self.intermediate,
            self.phys2,
            self.right,
        )

    @property
    def out_entries(self):
        """
        Return compact ``(key, shape)`` descriptors for tensor outputs.

        :returns: Tuple of ``(sector_key, block_shape)`` pairs.
        """

        return tuple((entry.key, entry.shape) for entry in self.entries)

    def index_by_key(self):
        """
        Map each sector key to its packed-entry index.

        :returns: Dictionary mapping sector keys to integer entry positions.
        """

        return {entry.key: idx for idx, entry in enumerate(self.entries)}

    def slices(self):
        """
        Map each sector key to its packed vector slice.

        :returns: Dictionary mapping sector keys to packed-vector slices.
        """

        return {entry.key: entry.slice for entry in self.entries}

    def entry_for_index(self, index):
        """
        Return the packed entry containing a flat basis index.

        :param index: Flat packed-vector index.
        :returns: ``LocalLayoutEntry`` containing ``index``.
        :raises IndexError: If ``index`` is outside this basis.
        """

        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError(f"Basis index {index} is out of range for size {self.size}.")
        for entry in self.entries:
            if entry.offset <= index < entry.offset + entry.size:
                return entry
        raise IndexError(f"Basis index {index} is out of range for size {self.size}.")

    def basis_block(self, index, *, dtype=complex):
        """
        Return a one-hot block representation of a flat basis vector.

        :param index: Flat packed-vector index.
        :param dtype: Output block dtype.
        :returns: ``(sector_key, block)`` for the selected basis vector.
        """

        entry = self.entry_for_index(index)
        block = np.zeros(entry.size, dtype=dtype)
        block[index - entry.offset] = 1.0
        return entry.key, block.reshape(entry.shape)

    def iter_packed_blocks(self, vector, *, drop_zeros=False):
        """
        Iterate over block views of a packed vector in this basis.

        :param vector: Packed vector in this two-site basis.
        :param drop_zeros: When ``True``, skip numerically zero blocks.
        :returns: Iterator yielding ``(LocalLayoutEntry, block_view)`` pairs.
        """

        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError(f"Packed vector has size {vector.size}, expected {self.size}.")
        for entry in self.entries:
            piece = vector[entry.slice]
            if drop_zeros and np.linalg.norm(piece) <= 0.0:
                continue
            yield entry, piece.reshape(entry.shape)

    def write_packed_block(self, vector, entry_or_key, block):
        """
        Write one sector block into a packed vector.

        :param vector: Packed vector to modify in-place.
        :param entry_or_key: ``LocalLayoutEntry`` or sector key selecting the
            destination block.
        :param block: Block data compatible with the selected entry shape.
        :returns: The modified packed vector.
        """

        entry = (
            entry_or_key
            if isinstance(entry_or_key, LocalLayoutEntry)
            else self.entry_for_key(entry_or_key)
        )
        vector = np.asarray(vector)
        block = np.asarray(block)
        vector[entry.slice] = block.reshape(entry.size)
        return vector

    def add_packed_block(self, vector, entry_or_key, block):
        """
        Add one sector block into a packed vector.

        :param vector: Packed vector to modify in-place.
        :param entry_or_key: ``LocalLayoutEntry`` or sector key selecting the
            destination block.
        :param block: Block data compatible with the selected entry shape.
        :returns: The modified packed vector.
        """

        entry = (
            entry_or_key
            if isinstance(entry_or_key, LocalLayoutEntry)
            else self.entry_for_key(entry_or_key)
        )
        vector = np.asarray(vector)
        block = np.asarray(block)
        vector[entry.slice] += block.reshape(entry.size)
        return vector

    def blocks_from_packed(self, vector, *, drop_zeros=True):
        """
        Convert a packed vector into sector-keyed blocks.

        :param vector: Packed vector in this two-site basis.
        :param drop_zeros: When ``True``, omit numerically zero blocks.
        :returns: Dictionary mapping sector keys to reshaped blocks.
        """

        return {
            entry.key: block
            for entry, block in self.iter_packed_blocks(vector, drop_zeros=drop_zeros)
        }

    def blocks_to_packed(self, blocks, *, dtype=None):
        """
        Convert sector-keyed blocks into a packed vector.

        :param blocks: Mapping from sector keys to block arrays.
        :param dtype: Optional output dtype. When omitted, infer from blocks.
        :returns: Packed vector in this two-site basis.
        """

        if dtype is None:
            present = [np.asarray(block).dtype for block in blocks.values()]
            dtype = np.result_type(*(present or [float]))
        vec = np.zeros(self.size, dtype=dtype)
        for entry in self.entries:
            if entry.key not in blocks:
                continue
            self.write_packed_block(vec, entry, blocks[entry.key])
        return vec

    def blocks_from_tensor_data(self, data, *, drop_zeros=True, copy=True):
        """
        Extract basis-compatible blocks from tensor data.

        :param data: Mapping from sector keys to tensor blocks.
        :param drop_zeros: When ``True``, omit numerically zero blocks.
        :param copy: When ``True``, copy returned arrays.
        :returns: Dictionary mapping basis sector keys to block arrays.
        """

        blocks = {}
        for entry in self.entries:
            if entry.key not in data:
                continue
            block = np.asarray(data[entry.key])
            if block.shape != entry.shape:
                raise ValueError(
                    f"Tensor block {entry.key!r} has shape {block.shape!r}, "
                    f"expected {entry.shape!r}."
                )
            if drop_zeros and np.linalg.norm(block.reshape(-1)) <= 0.0:
                continue
            blocks[entry.key] = np.array(block, copy=copy)
        return blocks

    def blocks_from_two_site_tensor(self, tensor, *, drop_zeros=True, copy=True):
        """Extract ordinary or intermediate-channel blocks from a tensor."""

        if not self.channel_resolved:
            return self.blocks_from_tensor_data(
                tensor.data,
                drop_zeros=drop_zeros,
                copy=copy,
            )
        if not bool(
            tensor.metadata.get("contracted_channel_blocks_current", False)
        ):
            raise ValueError(
                "Channel-resolved basis requires current intermediate-channel "
                "blocks on the two-site tensor."
            )
        channel_blocks = tensor.metadata.get("contracted_channel_blocks", {})
        blocks = {}
        for entry in self.entries:
            block = channel_blocks.get(entry.key)
            if block is None:
                continue
            block = np.asarray(block)
            if block.shape != entry.shape:
                raise ValueError(
                    f"Channel block {entry.key!r} has shape {block.shape!r}, "
                    f"expected {entry.shape!r}."
                )
            if drop_zeros and np.linalg.norm(block.reshape(-1)) <= 0.0:
                continue
            blocks[entry.key] = np.array(block, copy=copy)
        return blocks

    def tensor_data_from_blocks(self, blocks, *, template_data=None, default_dtype=float):
        """
        Expand state blocks into full tensor data for this basis.

        :param blocks: Mapping from sector keys to state blocks.
        :param template_data: Optional tensor data used to choose zero-block
            dtypes when a state block is absent.
        :param default_dtype: Dtype for absent blocks not present in
            ``template_data``.
        :returns: Dictionary mapping every basis key to tensor block arrays.
        """

        template_data = {} if template_data is None else template_data
        data = {}
        for entry in self.entries:
            if entry.key in blocks:
                block = np.asarray(blocks[entry.key])
                if block.shape != entry.shape:
                    block = block.reshape(entry.shape)
                data[entry.key] = np.array(block, copy=True)
            elif entry.key in template_data:
                data[entry.key] = np.zeros(entry.shape, dtype=np.asarray(template_data[entry.key]).dtype)
            else:
                data[entry.key] = np.zeros(entry.shape, dtype=default_dtype)
        return data

    def tensor_from_blocks(self, blocks, template):
        """Rebuild a two-site tensor while preserving fusion-path amplitudes."""

        from .tensor import NonabelianTensor

        if not self.channel_resolved:
            data = self.tensor_data_from_blocks(
                blocks,
                template_data=template.data,
            )
            metadata = template.metadata.copy()
            metadata["contracted_channel_blocks_current"] = False
            return NonabelianTensor(
                data,
                [leg[:] for leg in template.qns],
                template.dirs[:],
                fusion_legs=template.fusion_legs[:],
                metadata=metadata,
            )

        channel_blocks = {}
        data = {}
        for entry in self.entries:
            block = np.asarray(
                blocks.get(
                    entry.key,
                    np.zeros(entry.shape, dtype=float),
                )
            ).reshape(entry.shape)
            channel_blocks[entry.key] = np.array(block, copy=True)
            outer_key = (
                entry.key[0],
                entry.key[1],
                entry.key[3],
                entry.key[4],
            )
            if outer_key in data:
                data[outer_key] = data[outer_key] + block
            else:
                data[outer_key] = np.array(block, copy=True)
        metadata = template.metadata.copy()
        metadata["contracted_channel_blocks"] = channel_blocks
        metadata["contracted_channel_blocks_current"] = True
        return NonabelianTensor(
            data,
            [leg[:] for leg in template.qns],
            template.dirs[:],
            fusion_legs=template.fusion_legs[:],
            metadata=metadata,
        )

    def entry_index(self, key):
        """
        Return the packed-entry index for a sector key.

        :param key: Four-sector two-site block key.
        :returns: Integer packed-entry index.
        :raises KeyError: If ``key`` is absent from the basis.
        """

        key = tuple(key)
        for idx, entry in enumerate(self.entries):
            if entry.key == key:
                return idx
        raise KeyError(f"No two-site layout entry for key {key!r}.")

    def entry_for_key(self, key):
        """
        Return the packed entry for a sector key.

        :param key: Four-sector two-site block key.
        :returns: Matching ``LocalLayoutEntry``.
        :raises KeyError: If ``key`` is absent from the basis.
        """

        key = tuple(key)
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(f"No two-site layout entry for key {key!r}.")

    def compatible_with_layout(self, layout):
        layout = tuple(layout)
        return tuple((entry.key, entry.shape, entry.offset, entry.size) for entry in self.entries) == tuple(
            (tuple(entry.key), tuple(entry.shape), int(entry.offset), int(entry.size))
            for entry in layout
        )

    @classmethod
    def from_tensor_and_layout(cls, two_site, layout):
        if two_site.rank != 4:
            raise ValueError("TwoSiteBasis expects a rank-4 two-site tensor.")
        entries = tuple(
            LocalLayoutEntry(entry.key, entry.shape, entry.offset, entry.size)
            for entry in layout
        )
        return cls(
            left=BondBasis.from_tensor_axis(two_site, 0, name="left"),
            phys1=SiteBasis.from_tensor_axis(two_site, 1, name="phys1"),
            phys2=SiteBasis.from_tensor_axis(two_site, 2, name="phys2"),
            right=BondBasis.from_tensor_axis(two_site, 3, name="right"),
            entries=entries,
        )

    @classmethod
    def from_channel_tensor(cls, two_site):
        """Build a packed basis that keeps each intermediate fusion path."""

        if two_site.rank != 4:
            raise ValueError(
                "Channel-resolved TwoSiteBasis expects a rank-4 tensor."
            )
        if not bool(
            two_site.metadata.get("contracted_channel_blocks_current", False)
        ):
            raise ValueError(
                "Two-site tensor has no current intermediate-channel blocks."
            )
        channel_blocks = two_site.metadata.get("contracted_channel_blocks", {})
        if not channel_blocks:
            raise ValueError("Two-site tensor has no intermediate-channel blocks.")
        entries = []
        offset = 0
        for key in sorted(channel_blocks):
            block = np.asarray(channel_blocks[key])
            size = int(block.size)
            entries.append(
                LocalLayoutEntry(
                    tuple(key),
                    tuple(int(dim) for dim in block.shape),
                    offset,
                    size,
                )
            )
            offset += size
        middle_sectors = _ordered_unique(entry.key[2] for entry in entries)
        return cls(
            left=BondBasis.from_tensor_axis(two_site, 0, name="left"),
            phys1=SiteBasis.from_tensor_axis(two_site, 1, name="phys1"),
            phys2=SiteBasis.from_tensor_axis(two_site, 2, name="phys2"),
            right=BondBasis.from_tensor_axis(two_site, 3, name="right"),
            entries=tuple(entries),
            intermediate=BondBasis.from_qns(
                middle_sectors,
                dims={sector: 1 for sector in middle_sectors},
                direction=1,
                name="intermediate",
            ),
        )

    def metric_is_identity(self, metric, *, tol=1.0e-10):
        metric = np.asarray(metric)
        if metric.shape != (self.size, self.size):
            return False
        eye = np.eye(self.size, dtype=metric.dtype)
        return bool(np.allclose(metric, eye, atol=tol, rtol=tol))

    def metric_orthonormalization(self, metric, *, tol=1.0e-12):
        """
        Build a metric-orthonormal packed local basis.

        The returned transform ``X`` maps orthonormal local coordinates ``y``
        back to the current packed basis, ``v = X y``, and satisfies
        ``X.conj().T @ metric @ X = I`` up to ``tol``.  This is the explicit
        local-basis step needed before treating a generalized effective problem
        as a standard eigenproblem.

        :param metric: Dense packed local norm matrix in this basis.
        :param tol: Eigenvalue cutoff for the metric null space.
        :returns: ``MetricOrthonormalization`` helper for vector and operator
            transforms.
        :raises ValueError: If ``metric`` has the wrong shape or no positive
            metric directions remain.
        """

        metric = np.asarray(metric)
        if metric.shape != (self.size, self.size):
            raise ValueError(
                f"Metric shape {metric.shape!r} does not match TwoSiteBasis size {self.size}."
            )
        metric = 0.5 * (metric + metric.conj().T)
        eigvals, eigvecs = np.linalg.eigh(metric)
        keep = eigvals > max(float(tol), 1.0e-14)
        if not np.any(keep):
            raise ValueError("Two-site metric has no positive orthonormal directions.")
        transform = eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))
        return MetricOrthonormalization(
            parent=self,
            metric=metric,
            transform=transform,
            eigvals=eigvals[keep],
        )


@dataclass(frozen=True)
class MetricOrthonormalization:
    """
    Dense metric-orthonormalized view of a ``TwoSiteBasis``.

    ``transform`` stores original packed-basis coefficients for each
    orthonormal local basis vector.  It therefore maps orthonormal coordinates
    back to the original packed vector space.

    :param parent: Original packed ``TwoSiteBasis``.
    :param metric: Hermitian packed local norm matrix.
    :param transform: Dense transform ``X`` with ``X^H metric X = I``.
    :param eigvals: Kept positive metric eigenvalues.
    """

    parent: TwoSiteBasis
    metric: np.ndarray
    transform: np.ndarray
    eigvals: np.ndarray

    @property
    def size(self):
        """
        Return the orthonormalized local dimension.

        :returns: Number of kept metric directions.
        """

        return int(self.transform.shape[1])

    def from_orthonormal_vector(self, vector):
        """
        Map orthonormal coordinates to the original packed basis.

        :param vector: Coordinates in the orthonormal metric basis.
        :returns: Packed vector in the parent ``TwoSiteBasis``.
        """

        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError(f"Vector has size {vector.size}, expected {self.size}.")
        return self.transform @ vector.reshape(-1)

    def to_orthonormal_vector(self, vector):
        """
        Project an original packed vector into orthonormal coordinates.

        :param vector: Packed vector in the parent basis.
        :returns: Coordinates in the orthonormal metric basis.
        """

        vector = np.asarray(vector)
        if vector.size != self.parent.size:
            raise ValueError(f"Vector has size {vector.size}, expected {self.parent.size}.")
        return self.transform.conj().T @ (self.metric @ vector.reshape(-1))

    def operator_to_orthonormal(self, operator):
        """
        Transform a packed operator to the orthonormal metric basis.

        :param operator: Dense packed operator in the parent basis.
        :returns: Dense operator matrix in orthonormal coordinates.
        """

        operator = np.asarray(operator)
        if operator.shape != (self.parent.size, self.parent.size):
            raise ValueError(
                f"Operator shape {operator.shape!r} does not match parent basis size {self.parent.size}."
            )
        transformed = self.transform.conj().T @ operator @ self.transform
        return 0.5 * (transformed + transformed.conj().T)
