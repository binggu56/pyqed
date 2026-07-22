"""Toy first-quantized NARG helpers for two particles on a DVR grid.

The routines here keep only the physical ordered wedge ``|i j>``, ``i < j``.
Exchange symmetry is folded into the Hamiltonian through the wedge kinetic
operator, so no artificial product-grid region is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass
class TwoElectronFirstQuantizedNARGResult:
    """Result of the two-electron ordered-wedge NARG toy solver."""

    energies: np.ndarray
    vectors: np.ndarray
    projected_hamiltonian: np.ndarray
    branch_basis: np.ndarray
    branch_energies: list[np.ndarray]
    branch_rows: list[np.ndarray]
    exact_energies: np.ndarray | None
    hamiltonian: np.ndarray
    pairs: np.ndarray
    D: int
    exchange: str


@dataclass
class ManyElectronFirstQuantizedNARGResult:
    """Result of the ordered-coordinate many-electron NARG toy solver."""

    energies: np.ndarray
    vectors: np.ndarray
    projected_hamiltonian: np.ndarray
    branch_basis: np.ndarray
    exact_energies: np.ndarray | None
    hamiltonian: object
    configs: object
    D: int
    nelec: int


@dataclass
class ParticleGrowthState:
    """Final states in the retained particle-growth layer basis."""

    layer: "ParticleGrowthLayer"
    coeff: np.ndarray

    def __post_init__(self):
        self.coeff = np.asarray(self.coeff)
        if self.coeff.ndim == 1:
            self.coeff = self.coeff[:, None]
        if self.coeff.shape[0] != self.layer.shape[1]:
            raise ValueError("coeff leading dimension must match the layer basis.")

    @property
    def shape(self):
        return self.layer.shape[0], self.coeff.shape[1]

    def to_tree_basis(self):
        return self.layer.to_tree().truncate(self.coeff)

    def to_dense(self):
        return self.to_tree_basis().to_dense()


@dataclass
class ParticleGrowthBlock:
    """One retained block in a shared particle-growth layer."""

    value: int
    coeff: np.ndarray
    energies: np.ndarray
    previous_layer: "ParticleGrowthLayer | None" = None
    child_values: tuple[int, ...] = ()

    def __post_init__(self):
        self.value = int(self.value)
        self.child_values = tuple(int(value) for value in self.child_values)
        self.coeff = np.asarray(self.coeff)
        if self.coeff.ndim == 1:
            self.coeff = self.coeff[:, None]
        self.energies = np.asarray(self.energies)
        if self.previous_layer is None:
            if self.coeff.shape[0] != 1:
                raise ValueError("one-particle growth blocks must have one candidate state.")
        elif self.coeff.shape[0] != self.candidate_size:
            raise ValueError("coeff leading dimension must match child candidate size.")
        if self.energies.shape[0] != self.coeff.shape[1]:
            raise ValueError("energies must match retained block dimension.")

    @property
    def candidate_size(self):
        if self.previous_layer is None:
            return 1
        return sum(self.previous_layer.blocks[value].shape[1] for value in self.child_values)

    @property
    def shape(self):
        return self.coeff.shape[1], self.coeff.shape[1]

    @property
    def dtype(self):
        return np.result_type(self.coeff, self.energies, float)

    def child_slices(self):
        if self.previous_layer is None:
            return {}
        slices = {}
        offset = 0
        for value in self.child_values:
            width = self.previous_layer.blocks[value].shape[1]
            slices[value] = slice(offset, offset + width)
            offset += width
        return slices


@dataclass
class ParticleGrowthLayer:
    """Retained nearest-neighbor particle-growth layer."""

    operator: object
    blocks: dict[int, ParticleGrowthBlock]
    D: int

    def __post_init__(self):
        self.blocks = {int(key): value for key, value in self.blocks.items()}
        self.D = int(self.D)
        if self.D < 1:
            raise ValueError("D must be positive.")

    @property
    def branches(self):
        return self.blocks

    @property
    def values(self):
        return tuple(sorted(self.blocks))

    @property
    def shape(self):
        return self.operator.shape[0], sum(self.blocks[value].shape[1] for value in self.values)

    def iter_branches(self):
        yield from self.to_tree().iter_branches()

    def to_tree(self):
        cache = {}
        children = [
            _particle_growth_block_to_branch_with_operator(self.blocks[value], self.operator, cache)
            for value in self.values
        ]
        child_bases = [child.basis for child in children]
        rows = self.operator.prefix_rows(())
        basis = (
            RecursiveCoordinateBasis.hstack(child_bases, rows=rows)
            if child_bases
            else RecursiveCoordinateBasis.leaf(
                self.operator.shape[0],
                rows,
                np.zeros((rows.size, 0), dtype=self.operator.dtype),
            )
        )
        root = CoordinateBranch(
            prefix=(),
            depth=0,
            rows=rows,
            basis=basis,
            energies=np.empty(0),
            children=children,
        )
        return CoordinateTreeBasis(root)

    def project(self, operator=None):
        operator = self.operator if operator is None else operator
        if operator.shape[0] != self.operator.shape[0]:
            raise ValueError("operator is incompatible with this particle-growth layer.")
        return self.project_values(self.values, operator=operator)

    def project_values(self, values, operator=None):
        operator = self.operator if operator is None else operator
        values = tuple(int(value) for value in values)
        if not _is_nearest_neighbor_kinetic(operator):
            return self._project_values_via_tree(values, operator)

        widths = [self.blocks[value].shape[1] for value in values]
        offsets = np.concatenate(([0], np.cumsum(widths)))
        projected = np.zeros((offsets[-1], offsets[-1]), dtype=np.result_type(operator.dtype, complex))
        for pos, value in enumerate(values):
            block = self.blocks[value]
            block_slice = slice(offsets[pos], offsets[pos + 1])
            projected[block_slice, block_slice] = np.diag(block.energies)

        for left_pos in range(len(values) - 1):
            left_value = values[left_pos]
            right_value = values[left_pos + 1]
            if right_value != left_value + 1:
                continue
            left = self.blocks[left_value]
            right = self.blocks[right_value]
            left_slice = slice(offsets[left_pos], offsets[left_pos + 1])
            right_slice = slice(offsets[left_pos + 1], offsets[left_pos + 2])
            projected[left_slice, right_slice] = _project_particle_growth_blocks(operator, left, right)
            projected[right_slice, left_slice] = _project_particle_growth_blocks(operator, right, left)
        return 0.5 * (projected + projected.T.conj())

    def _project_values_via_tree(self, values, operator):
        cache = {}
        branches = [
            _particle_growth_block_to_branch_with_operator(self.blocks[int(value)], self.operator, cache)
            for value in values
        ]
        if not branches:
            return np.zeros((0, 0), dtype=np.result_type(operator.dtype, complex))
        rows = np.unique(np.concatenate([_row_array(branch.rows) for branch in branches]))
        basis = RecursiveCoordinateBasis.hstack([branch.basis for branch in branches], rows=rows)
        return operator.project(basis)

    def truncate(self, coeff):
        return ParticleGrowthState(self, coeff)

    def to_sparse(self):
        return self.to_tree().to_sparse()


@dataclass
class ParticleGrowthNARGResult:
    """Result of the layer-only first-quantized particle-growth NARG solver."""

    energies: np.ndarray
    vectors: ParticleGrowthState
    projected_hamiltonian: np.ndarray
    branch_basis: ParticleGrowthLayer
    exact_energies: np.ndarray | None
    hamiltonian: object
    configs: object
    D: int
    nelec: int


@dataclass
class BranchNode:
    """Bookkeeping node for the recursive ordered-coordinate branch tree."""

    depth: int
    prefix: tuple[int, ...]
    rows: np.ndarray
    energies: np.ndarray
    children: list["BranchNode"]


def _rank_combination(config, npoints: int, nelec: int):
    """Lexicographic rank matching ``itertools.combinations(range(n), k)``."""

    config = tuple(int(value) for value in config)
    if len(config) != int(nelec):
        raise ValueError("configuration has the wrong electron count.")
    if any(config[idx] >= config[idx + 1] for idx in range(len(config) - 1)):
        raise ValueError("configuration must be strictly ordered.")
    if config and (config[0] < 0 or config[-1] >= int(npoints)):
        raise ValueError("configuration is outside the grid.")

    rank = 0
    start = 0
    for pos, value in enumerate(config):
        for trial in range(start, value):
            rank += comb(int(npoints) - trial - 1, int(nelec) - pos - 1)
        start = value + 1
    return int(rank)


def _unrank_combination(rank: int, npoints: int, nelec: int):
    """Inverse of ``_rank_combination`` in lexicographic combination order."""

    rank = int(rank)
    npoints = int(npoints)
    nelec = int(nelec)
    size = comb(npoints, nelec)
    if rank < 0 or rank >= size:
        raise IndexError("configuration rank is out of range.")

    out = []
    start = 0
    for pos in range(nelec):
        for value in range(start, npoints):
            count = comb(npoints - value - 1, nelec - pos - 1)
            if rank < count:
                out.append(value)
                start = value + 1
                break
            rank -= count
    return np.asarray(out, dtype=int)


@dataclass
class OrderedConfigurationSpace:
    """Lazy ordered configurations ``i1 < i2 < ... < iN``."""

    npoints: int
    nelec: int

    def __post_init__(self):
        self.npoints = int(self.npoints)
        self.nelec = int(self.nelec)
        if self.nelec < 1:
            raise ValueError("nelec must be positive.")
        if self.npoints < self.nelec:
            raise ValueError("npoints must be at least nelec.")
        self.size = comb(self.npoints, self.nelec)

    @property
    def shape(self):
        return self.size, self.nelec

    def __len__(self):
        return self.size

    def rank(self, config):
        return _rank_combination(config, self.npoints, self.nelec)

    def unrank(self, rank):
        return _unrank_combination(rank, self.npoints, self.nelec)

    def _validate_prefix(self, prefix):
        prefix = tuple(int(value) for value in prefix)
        if len(prefix) > self.nelec:
            raise ValueError("prefix is longer than nelec.")
        if any(prefix[idx] >= prefix[idx + 1] for idx in range(len(prefix) - 1)):
            raise ValueError("prefix must be strictly ordered.")
        if prefix and (prefix[0] < 0 or prefix[-1] >= self.npoints):
            raise ValueError("prefix is outside the grid.")
        return prefix

    def prefix_size(self, prefix):
        prefix = self._validate_prefix(prefix)
        remaining = self.nelec - len(prefix)
        if remaining == 0:
            return 1
        start = prefix[-1] + 1 if prefix else 0
        available = self.npoints - start
        return 0 if available < remaining else comb(available, remaining)

    def first_rank_with_prefix(self, prefix):
        prefix = self._validate_prefix(prefix)
        remaining = self.nelec - len(prefix)
        if remaining == 0:
            return self.rank(prefix)
        start = prefix[-1] + 1 if prefix else 0
        if self.npoints - start < remaining:
            raise ValueError("prefix leaves no valid suffix.")
        suffix = tuple(range(start, start + remaining))
        return self.rank((*prefix, *suffix))

    def prefix_rows(self, prefix):
        count = self.prefix_size(prefix)
        if count == 0:
            return PrefixCoordinateSpace(self, prefix, 0, 0)
        first = self.first_rank_with_prefix(prefix)
        return PrefixCoordinateSpace(self, prefix, first, count)

    def child_values(self, prefix):
        prefix = self._validate_prefix(prefix)
        depth = len(prefix)
        if depth >= self.nelec:
            return np.empty(0, dtype=int)
        lower = prefix[-1] + 1 if prefix else 0
        upper = self.npoints - (self.nelec - depth)
        if lower > upper:
            return np.empty(0, dtype=int)
        return np.arange(lower, upper + 1, dtype=int)

    def to_array(self):
        return np.asarray([self.unrank(row) for row in range(self.size)], dtype=int)

    def __array__(self, dtype=None):
        array = self.to_array()
        return array.astype(dtype, copy=False) if dtype is not None else array

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, col = key
            configs = self[rows]
            return configs[..., int(col)]
        if isinstance(key, (int, np.integer)):
            row = int(key)
            if row < 0:
                row += self.size
            return self.unrank(row)
        if isinstance(key, slice):
            rows = range(*key.indices(self.size))
            return np.asarray([self.unrank(row) for row in rows], dtype=int)
        rows = np.asarray(key)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        rows = rows.astype(int, copy=False)
        return np.asarray([self.unrank(row) for row in rows.ravel()], dtype=int).reshape(*rows.shape, self.nelec)


@dataclass(frozen=True)
class PrefixCoordinateSpace:
    """Implicit global row interval for configurations sharing a fixed prefix."""

    configurations: OrderedConfigurationSpace
    prefix: tuple[int, ...]
    start: int
    size: int

    def __post_init__(self):
        prefix = self.configurations._validate_prefix(self.prefix)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "size", int(self.size))
        if self.start < 0 or self.size < 0:
            raise ValueError("prefix row interval must be non-negative.")
        if self.start + self.size > self.configurations.size:
            raise ValueError("prefix row interval exceeds the configuration space.")

    @property
    def shape(self):
        return (self.size,)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(range(self.start, self.start + self.size))

    def __array__(self, dtype=None, copy=None):
        array = np.arange(self.start, self.start + self.size, dtype=int)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            local = int(key)
            if local < 0:
                local += self.size
            if local < 0 or local >= self.size:
                raise IndexError("prefix row index is out of range.")
            return self.start + local
        if isinstance(key, slice):
            local = np.arange(*key.indices(self.size), dtype=int)
            return self.start + local
        return np.asarray(self)[key]

    def to_array(self):
        return np.asarray(self)

    def global_row(self, local_index):
        return self[int(local_index)]

    def local_index(self, row):
        row = int(row)
        local = row - self.start
        if local < 0 or local >= self.size:
            raise ValueError("row is outside this prefix space.")
        return local

    def config(self, local_index):
        return self.configurations.unrank(self.global_row(local_index))

    def suffix(self, local_index):
        return self.config(local_index)[len(self.prefix):]


def _is_prefix_coordinate_space(rows):
    return isinstance(rows, PrefixCoordinateSpace)


def _row_size(rows):
    return int(rows.size if hasattr(rows, "size") else len(rows))


def _row_array(rows):
    return np.asarray(rows, dtype=int)


def _row_positions(parent_rows, child_rows):
    if _is_prefix_coordinate_space(parent_rows) and _is_prefix_coordinate_space(child_rows):
        offset = child_rows.start - parent_rows.start
        if offset < 0 or offset + child_rows.size > parent_rows.size:
            raise ValueError("child rows are not contained in the parent rows.")
        return offset + np.arange(child_rows.size, dtype=int)

    parent_array = _row_array(parent_rows)
    child_array = _row_array(child_rows)
    loc = np.searchsorted(parent_array, child_array)
    if loc.size and (loc[-1] >= parent_array.size or not np.array_equal(parent_array[loc], child_array)):
        raise ValueError("child rows are not contained in the parent rows.")
    return loc


def _row_localizer(rows):
    if _is_prefix_coordinate_space(rows):
        start = rows.start
        stop = rows.start + rows.size

        def local_index(row):
            row = int(row)
            return row - start if start <= row < stop else None

        return local_index

    row_index = {int(row): loc for loc, row in enumerate(rows)}
    return lambda row: row_index.get(int(row))


@dataclass
class SparseBasis:
    """Sparse column basis stored as ordered row/value supports."""

    nrows: int
    columns: list[tuple[np.ndarray, np.ndarray]]

    @property
    def shape(self):
        return int(self.nrows), len(self.columns)

    @classmethod
    def from_local(cls, nrows: int, rows, local_vectors):
        rows = np.asarray(rows, dtype=int)
        local_vectors = np.asarray(local_vectors)
        if local_vectors.ndim == 1:
            local_vectors = local_vectors[:, None]
        if local_vectors.shape[0] != rows.size:
            raise ValueError("local_vectors has incompatible leading dimension.")
        columns = []
        order = np.argsort(rows)
        sorted_rows = rows[order]
        for col in range(local_vectors.shape[1]):
            values = np.asarray(local_vectors[:, col])[order]
            columns.append((sorted_rows.copy(), values.copy()))
        return cls(int(nrows), columns)

    @classmethod
    def hstack(cls, bases):
        bases = list(bases)
        if not bases:
            raise ValueError("at least one basis is required.")
        nrows = bases[0].nrows
        columns = []
        for basis in bases:
            if basis.nrows != nrows:
                raise ValueError("basis dimensions do not match.")
            columns.extend((rows.copy(), values.copy()) for rows, values in basis.columns)
        return cls(nrows, columns)

    def column(self, index):
        rows, values = self.columns[int(index)]
        return rows, values

    def combine(self, coeff):
        coeff = np.asarray(coeff)
        if coeff.ndim == 1:
            coeff = coeff[:, None]
        if coeff.shape[0] != len(self.columns):
            raise ValueError("coeff leading dimension must match the number of columns.")
        if not self.columns:
            return SparseBasis(self.nrows, [])

        union = np.unique(np.concatenate([rows for rows, _values in self.columns]))
        local = np.zeros((union.size, coeff.shape[1]), dtype=np.result_type(coeff, *[v for _r, v in self.columns]))
        for source, (rows, values) in enumerate(self.columns):
            loc = np.searchsorted(union, rows)
            local[loc, :] += values[:, None] * coeff[source, :]
        return SparseBasis.from_local(self.nrows, union, local)

    def dot_sparse(self, index, rows, values):
        col_rows, col_values = self.column(index)
        rows = np.asarray(rows, dtype=int)
        values = np.asarray(values)
        common, left, right = np.intersect1d(col_rows, rows, assume_unique=True, return_indices=True)
        if common.size == 0:
            return 0.0
        return np.vdot(col_values[left], values[right])

    def to_dense(self):
        dense = np.zeros((self.nrows, len(self.columns)), dtype=np.result_type(*[v for _r, v in self.columns], float))
        for col, (rows, values) in enumerate(self.columns):
            dense[rows, col] = values
        return dense


@dataclass
class LocalCoordinateBasis:
    """Branch-local basis with rows stored only inside one coordinate prefix."""

    nrows: int
    rows: object
    vectors: np.ndarray

    def __post_init__(self):
        self.nrows = int(self.nrows)
        if not _is_prefix_coordinate_space(self.rows):
            self.rows = np.asarray(self.rows, dtype=int)
        self.vectors = np.asarray(self.vectors)
        if not _is_prefix_coordinate_space(self.rows) and self.rows.ndim != 1:
            raise ValueError("rows must be one-dimensional.")
        if self.vectors.ndim == 1:
            self.vectors = self.vectors[:, None]
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be one- or two-dimensional.")
        if self.vectors.shape[0] != _row_size(self.rows):
            raise ValueError("vectors has incompatible local dimension.")
        if _row_size(self.rows) and (self.rows[0] < 0 or self.rows[-1] >= self.nrows):
            raise ValueError("rows are outside the global dimension.")
        if not _is_prefix_coordinate_space(self.rows) and np.any(self.rows[:-1] >= self.rows[1:]):
            raise ValueError("rows must be strictly increasing.")

    @property
    def shape(self):
        return self.nrows, self.vectors.shape[1]

    @property
    def local_shape(self):
        return _row_size(self.rows), self.vectors.shape[1]

    @classmethod
    def from_local(cls, nrows: int, rows, local_vectors):
        local_vectors = np.asarray(local_vectors)
        if local_vectors.ndim == 1:
            local_vectors = local_vectors[:, None]
        if local_vectors.shape[0] != _row_size(rows):
            raise ValueError("local_vectors has incompatible leading dimension.")
        if _is_prefix_coordinate_space(rows):
            return cls(int(nrows), rows, local_vectors.copy())
        rows = np.asarray(rows, dtype=int)
        order = np.argsort(rows)
        return cls(int(nrows), rows[order].copy(), local_vectors[order, :].copy())

    @classmethod
    def hstack(cls, bases, rows=None):
        bases = list(bases)
        if not bases:
            raise ValueError("at least one basis is required.")
        nrows = bases[0].nrows
        for basis in bases:
            if basis.nrows != nrows:
                raise ValueError("basis dimensions do not match.")
        if rows is None:
            rows = np.unique(np.concatenate([_row_array(basis.rows) for basis in bases]))
        if not _is_prefix_coordinate_space(rows):
            rows = np.asarray(rows, dtype=int)
        ncols = sum(basis.shape[1] for basis in bases)
        dtype = np.result_type(*[basis.vectors for basis in bases], float)
        local = np.zeros((rows.size, ncols), dtype=dtype)
        offset = 0
        for basis in bases:
            loc = _row_positions(rows, basis.rows)
            width = basis.shape[1]
            local[loc, offset:offset + width] = basis.vectors
            offset += width
        return cls(nrows, rows, local)

    def column(self, index):
        values = self.vectors[:, int(index)]
        mask = values != 0
        return self.rows[mask], values[mask]

    def combine(self, coeff):
        coeff = np.asarray(coeff)
        if coeff.ndim == 1:
            coeff = coeff[:, None]
        if coeff.shape[0] != self.shape[1]:
            raise ValueError("coeff leading dimension must match the number of columns.")
        rows = self.rows if _is_prefix_coordinate_space(self.rows) else self.rows.copy()
        return LocalCoordinateBasis(self.nrows, rows, self.vectors @ coeff)

    def dot_sparse(self, index, rows, values):
        col_rows, col_values = self.column(index)
        return _sparse_inner(col_rows, col_values, rows, values)

    def to_sparse(self):
        return SparseBasis.from_local(self.nrows, self.rows, self.vectors)

    def to_dense(self):
        dense = np.zeros((self.nrows, self.shape[1]), dtype=np.result_type(self.vectors, float))
        dense[_row_array(self.rows), :] = self.vectors
        return dense


@dataclass
class RecursiveCoordinateBasis:
    """Recursive suffix-factor basis for one ordered-coordinate branch."""

    nrows: int
    rows: object
    children: list["RecursiveCoordinateBasis"]
    coeff: np.ndarray | None = None
    leaf_vectors: np.ndarray | None = None
    leaf_key: object | None = None

    def __post_init__(self):
        self.nrows = int(self.nrows)
        if not _is_prefix_coordinate_space(self.rows):
            self.rows = np.asarray(self.rows, dtype=int)
        self.children = list(self.children)
        if self.leaf_vectors is not None:
            self.leaf_vectors = np.asarray(self.leaf_vectors)
            if self.leaf_vectors.ndim == 1:
                self.leaf_vectors = self.leaf_vectors[:, None]
            if self.leaf_vectors.shape[0] != _row_size(self.rows):
                raise ValueError("leaf_vectors has incompatible local dimension.")
            if self.children:
                raise ValueError("leaf basis cannot also have children.")
        if self.coeff is not None:
            self.coeff = np.asarray(self.coeff)
            if self.coeff.ndim == 1:
                self.coeff = self.coeff[:, None]
            if self.coeff.shape[0] != self._candidate_ncols:
                raise ValueError("coeff leading dimension must match child candidate columns.")

    @property
    def _candidate_ncols(self):
        if self.leaf_vectors is not None:
            return self.leaf_vectors.shape[1]
        return sum(child.shape[1] for child in self.children)

    @property
    def shape(self):
        if self.coeff is not None:
            return self.nrows, self.coeff.shape[1]
        return self.nrows, self._candidate_ncols

    @property
    def local_shape(self):
        return _row_size(self.rows), self.shape[1]

    @property
    def dtype(self):
        pieces = [float]
        if self.leaf_vectors is not None:
            pieces.append(self.leaf_vectors)
        for child in self.children:
            pieces.append(child.dtype)
        if self.coeff is not None:
            pieces.append(self.coeff)
        return np.result_type(*pieces)

    @classmethod
    def leaf(cls, nrows: int, rows, local_vectors, coeff=None, leaf_key=None):
        return cls(
            nrows=int(nrows),
            rows=rows,
            children=[],
            coeff=coeff,
            leaf_vectors=local_vectors,
            leaf_key=leaf_key,
        )

    @classmethod
    def hstack(cls, bases, rows):
        bases = list(bases)
        if not bases:
            raise ValueError("at least one basis is required.")
        nrows = bases[0].nrows
        for basis in bases:
            if basis.nrows != nrows:
                raise ValueError("basis dimensions do not match.")
        return cls(nrows=nrows, rows=rows, children=bases)

    def to_local(self):
        if self.leaf_vectors is not None:
            vectors = self._leaf_effective_vectors()
        else:
            ncols = self._candidate_ncols
            vectors = np.zeros((_row_size(self.rows), ncols), dtype=self.dtype)
            offset = 0
            for child in self.children:
                child_basis = child.to_local()
                loc = _row_positions(self.rows, child_basis.rows)
                width = child_basis.shape[1]
                vectors[loc, offset:offset + width] = child_basis.vectors
                offset += width
        if self.coeff is not None and self.leaf_vectors is None:
            vectors = vectors @ self.coeff
        return LocalCoordinateBasis(self.nrows, self.rows, vectors)

    def _leaf_effective_vectors(self):
        if self.leaf_vectors is None:
            raise ValueError("basis is not a leaf.")
        if self.coeff is None:
            return self.leaf_vectors
        return self.leaf_vectors @ self.coeff

    def column(self, index):
        return self.to_local().column(index)

    def combine(self, coeff):
        coeff = np.asarray(coeff)
        if coeff.ndim == 1:
            coeff = coeff[:, None]
        if coeff.shape[0] != self.shape[1]:
            raise ValueError("coeff leading dimension must match the number of columns.")
        if self.leaf_vectors is not None:
            if self.coeff is None:
                return RecursiveCoordinateBasis.leaf(
                    self.nrows,
                    self.rows,
                    self.leaf_vectors,
                    coeff=coeff,
                    leaf_key=self.leaf_key,
                )
            return RecursiveCoordinateBasis.leaf(
                self.nrows,
                self.rows,
                self.leaf_vectors,
                coeff=self.coeff @ coeff,
                leaf_key=self.leaf_key,
            )
        if self.coeff is None:
            return RecursiveCoordinateBasis(self.nrows, self.rows, self.children, coeff=coeff)
        return RecursiveCoordinateBasis(self.nrows, self.rows, self.children, coeff=self.coeff @ coeff)

    def to_sparse(self):
        return self.to_local().to_sparse()

    def to_dense(self):
        return self.to_local().to_dense()


def _sparse_inner(left_rows, left_values, right_rows, right_values):
    left_rows = np.asarray(left_rows, dtype=int)
    right_rows = np.asarray(right_rows, dtype=int)
    left_values = np.asarray(left_values)
    right_values = np.asarray(right_values)
    common, left, right = np.intersect1d(
        left_rows,
        right_rows,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        return 0.0
    return np.vdot(left_values[left], right_values[right])


def _project_sparse_columns(operator, nrows, columns):
    columns = [
        (np.asarray(rows, dtype=int), np.asarray(values))
        for rows, values in columns
    ]
    if int(nrows) != operator.shape[0]:
        raise ValueError("basis has incompatible leading dimension.")
    ncols = len(columns)
    projected = np.zeros((ncols, ncols), dtype=np.result_type(operator.dtype, complex))
    for col, (rows, values) in enumerate(columns):
        out_rows, out_values = operator.apply_sparse(rows, values)
        for bra, (bra_rows, bra_values) in enumerate(columns):
            projected[bra, col] = _sparse_inner(bra_rows, bra_values, out_rows, out_values)
    return 0.5 * (projected + projected.T.conj())


def _apply_sparse_columns(operator, columns):
    applied = [
        operator.apply_sparse(rows, values)
        for rows, values in columns
    ]
    return SparseBasis(operator.shape[0], applied)


@dataclass
class CoordinateBranch:
    """Recursive branch for a fixed ordered-coordinate prefix."""

    prefix: tuple[int, ...]
    depth: int
    rows: object
    basis: RecursiveCoordinateBasis
    energies: np.ndarray
    children: list["CoordinateBranch"]

    @property
    def shape(self):
        return self.basis.shape

    def to_sparse(self):
        return self.basis.to_sparse()

    def iter_columns(self):
        for col in range(self.basis.shape[1]):
            yield self.basis.column(col)

    def project(self, operator):
        return operator.project(self.basis)

    def apply(self, operator):
        return _apply_sparse_columns(operator, self.iter_columns())

    def truncate(self, coeff):
        return self.basis.combine(coeff)

    def iter_branches(self):
        yield self
        for child in self.children:
            yield from child.iter_branches()


@dataclass
class CoordinateTreeBasis:
    """Recursive coordinate-branch basis for ordered first-quantized NARG."""

    root: CoordinateBranch

    @property
    def shape(self):
        return self.root.shape

    def to_sparse(self):
        return self.root.to_sparse()

    def project(self, operator):
        return self.root.project(operator)

    def apply(self, operator):
        return self.root.apply(operator)

    def truncate(self, coeff):
        return self.root.truncate(coeff)

    def iter_branches(self):
        return self.root.iter_branches()


@dataclass
class ManyElectronOrderedOperator:
    """Matrix-free same-spin ordered-coordinate Hamiltonian."""

    kinetic: object
    grid: np.ndarray
    nelec: int
    one_body: object
    two_body: object
    configs: object

    def __post_init__(self):
        self.grid = np.asarray(self.grid, dtype=float)
        self.nelec = int(self.nelec)
        self.kinetic, self._kinetic_row_hops, self._kinetic_col_hops, self._kinetic_dtype = (
            _as_kinetic_operator(self.kinetic, self.grid.size)
        )
        self._one_body_array, self._one_body_fn = _as_one_body_potential(self.one_body, self.grid)
        self._two_body_array, self._two_body_fn = _as_two_body_potential(self.two_body, self.grid)
        if isinstance(self.configs, OrderedConfigurationSpace):
            if self.configs.npoints != self.grid.size or self.configs.nelec != self.nelec:
                raise ValueError("configuration space is incompatible with grid/nelec.")
            self._dense_configs = None
        else:
            configs = np.asarray(self.configs, dtype=int)
            if configs.ndim != 2 or configs.shape[1] != self.nelec:
                raise ValueError("configs must have shape (nconfigs, nelec).")
            self.configs = configs
            self._dense_configs = configs
            self._dense_index = {tuple(config): row for row, config in enumerate(configs)}

    @property
    def shape(self):
        dim = self.configs.shape[0]
        return dim, dim

    @property
    def dtype(self):
        arrays = [self._kinetic_dtype, float]
        if self._one_body_array is not None:
            arrays.append(self._one_body_array)
        if self._two_body_array is not None:
            arrays.append(self._two_body_array)
        return np.result_type(*arrays)

    def config(self, row):
        return self.configs[int(row)]

    def row_index(self, config):
        ordered = tuple(int(value) for value in config)
        if isinstance(self.configs, OrderedConfigurationSpace):
            return self.configs.rank(ordered)
        return int(self._dense_index[ordered])

    def prefix_rows(self, prefix):
        prefix = tuple(int(value) for value in prefix)
        if isinstance(self.configs, OrderedConfigurationSpace):
            return self.configs.prefix_rows(prefix)
        if not prefix:
            return np.arange(self.configs.shape[0], dtype=int)
        mask = np.ones(self.configs.shape[0], dtype=bool)
        for depth, value in enumerate(prefix):
            mask &= self.configs[:, depth] == value
        return np.flatnonzero(mask)

    def child_values(self, prefix):
        prefix = tuple(int(value) for value in prefix)
        if isinstance(self.configs, OrderedConfigurationSpace):
            return self.configs.child_values(prefix)
        rows = self.prefix_rows(prefix)
        depth = len(prefix)
        if depth >= self.nelec or rows.size == 0:
            return np.empty(0, dtype=int)
        return np.unique(self.configs[rows, depth])

    def kinetic_terms(self, site, *, transpose=False):
        """Return nonzero one-particle kinetic hops from or into ``site``."""

        hops = self._kinetic_col_hops if transpose else self._kinetic_row_hops
        return hops[int(site)]

    def one_body_value(self, site):
        site = int(site)
        if self._one_body_array is not None:
            return self._one_body_array[site]
        return self._one_body_fn(float(self.grid[site]))

    def two_body_value(self, left, right):
        left = int(left)
        right = int(right)
        if self._two_body_array is not None:
            return self._two_body_array[left, right]
        return self._two_body_fn(float(self.grid[left]), float(self.grid[right]))

    def potential_value(self, config):
        config = np.asarray(config, dtype=int)
        value = sum(self.one_body_value(site) for site in config)
        value += sum(
            self.two_body_value(config[left], config[right])
            for left in range(self.nelec)
            for right in range(left + 1, self.nelec)
        )
        return value

    def diagonal_value(self, row, config=None):
        config = self.config(row) if config is None else np.asarray(config, dtype=int)
        return self.potential_value(config)

    def row_terms(self, row, config=None):
        """Yield ``(col, H[row, col])`` using analytic diagonal and sparse K hops."""

        row = int(row)
        config = self.config(row) if config is None else np.asarray(config, dtype=int)
        yield row, self.diagonal_value(row, config)
        for particle, site in enumerate(config):
            replacements, amplitudes = self.kinetic_terms(site)
            for replacement, amplitude in zip(replacements, amplitudes):
                if amplitude == 0:
                    continue
                trial = list(config)
                trial[particle] = int(replacement)
                if len(set(trial)) != self.nelec:
                    continue
                col = self.row_index(sorted(trial))
                yield col, amplitude * _permutation_sign(trial)

    def column_terms(self, col, config=None):
        """Yield ``(row, H[row, col])`` for sparse-vector Hamiltonian actions."""

        col = int(col)
        config = self.config(col) if config is None else np.asarray(config, dtype=int)
        yield col, self.diagonal_value(col, config)
        for particle, site in enumerate(config):
            replacements, amplitudes = self.kinetic_terms(site, transpose=True)
            for replacement, amplitude in zip(replacements, amplitudes):
                if amplitude == 0:
                    continue
                trial = list(config)
                trial[particle] = int(replacement)
                if len(set(trial)) != self.nelec:
                    continue
                row = self.row_index(sorted(trial))
                yield row, amplitude * _permutation_sign(trial)

    def matvec(self, vector):
        """Apply the ordered-sector Hamiltonian without forming its matrix."""

        vector = np.asarray(vector)
        if vector.shape != (self.shape[1],):
            raise ValueError("vector has incompatible shape.")
        out = np.zeros_like(vector, dtype=np.result_type(vector, self.dtype))
        for row in range(self.shape[0]):
            for col, amplitude in self.row_terms(row):
                out[row] += amplitude * vector[col]
        return out

    def matmat(self, matrix):
        """Apply the ordered-sector Hamiltonian to one or more vectors."""

        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            return self.matvec(matrix)
        if matrix.shape[0] != self.shape[1]:
            raise ValueError("matrix has incompatible leading dimension.")
        out = np.zeros_like(matrix, dtype=np.result_type(matrix, self.dtype))
        for row in range(self.shape[0]):
            for col, amplitude in self.row_terms(row):
                out[row, :] += amplitude * matrix[col, :]
        return out

    def apply_sparse(self, rows, values):
        """Apply the Hamiltonian to one sparse vector."""

        rows = np.asarray(rows, dtype=int)
        values = np.asarray(values)
        if rows.shape != values.shape:
            raise ValueError("rows and values must have the same shape.")
        dtype = np.result_type(values, self.dtype)
        out = {}
        for row, value in zip(rows, values):
            row = int(row)
            if value == 0:
                continue
            for dest, amplitude in self.column_terms(row):
                out[dest] = out.get(dest, dtype.type(0)) + amplitude * value
        if not out:
            return np.empty(0, dtype=int), np.empty(0, dtype=dtype)
        out_rows = np.fromiter(out.keys(), dtype=int, count=len(out))
        out_values = np.fromiter(out.values(), dtype=dtype, count=len(out))
        order = np.argsort(out_rows)
        return out_rows[order], out_values[order]

    def local_matvec(self, rows, vector):
        """Apply the Hamiltonian inside one local row space."""

        vector = np.asarray(vector)
        if vector.shape != (_row_size(rows),):
            raise ValueError("vector has incompatible local shape.")
        local_index = _row_localizer(rows)
        out = np.zeros_like(vector, dtype=np.result_type(vector, self.dtype))
        for row, value in zip(rows, vector):
            if value == 0:
                continue
            for dest, amplitude in self.column_terms(row):
                dest_loc = local_index(dest)
                if dest_loc is not None:
                    out[dest_loc] += amplitude * value
        return out

    def local_matmat(self, rows, matrix):
        """Apply the Hamiltonian to local vectors in one row space."""

        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            return self.local_matvec(rows, matrix)
        if matrix.shape[0] != _row_size(rows):
            raise ValueError("matrix has incompatible local leading dimension.")
        out = np.zeros_like(matrix, dtype=np.result_type(matrix, self.dtype))
        for col in range(matrix.shape[1]):
            out[:, col] = self.local_matvec(rows, matrix[:, col])
        return out

    def local_dense(self, rows):
        """Materialize a local Hamiltonian block through local matvecs."""

        eye = np.eye(_row_size(rows), dtype=self.dtype)
        dense = self.local_matmat(rows, eye)
        return 0.5 * (dense + dense.T.conj())

    def local_lowest_eigenpairs(self, rows, nstates):
        """Return the lowest local eigenpairs using local Hamiltonian actions."""

        nlocal = _row_size(rows)
        keep = min(int(nstates), nlocal)
        if keep < 1:
            return np.empty(0, dtype=self.dtype), np.zeros((nlocal, 0), dtype=self.dtype)

        if keep < nlocal - 1:
            try:
                from scipy.sparse.linalg import LinearOperator, eigsh

                dtype = np.dtype(self.dtype)

                def matvec(vector):
                    return self.local_matvec(rows, vector)

                linop = LinearOperator((nlocal, nlocal), matvec=matvec, dtype=dtype)
                evals, evecs = eigsh(linop, k=keep, which="SA", tol=1e-12)
                order = np.argsort(evals)
                return evals[order], evecs[:, order]
            except Exception:
                pass

        dense = self.local_dense(rows)
        evals, evecs = np.linalg.eigh(dense)
        return evals[:keep], evecs[:, :keep]

    def submatrix(self, rows):
        """Build a small dense restriction ``H[rows, rows]`` for local branches."""

        local_index = _row_localizer(rows)
        block = np.zeros((_row_size(rows), _row_size(rows)), dtype=self.dtype)
        for local_row, row in enumerate(rows):
            for col, amplitude in self.row_terms(row):
                local_col = local_index(col)
                if local_col is not None:
                    block[local_row, local_col] += amplitude
        return 0.5 * (block + block.T.conj())

    def project(self, basis):
        """Return ``basis.T @ H @ basis`` using matrix-free applications."""

        if isinstance(basis, RecursiveCoordinateBasis):
            return self.project_recursive(basis)
        if isinstance(basis, LocalCoordinateBasis):
            return self.project_local(basis)
        if isinstance(basis, SparseBasis):
            return self.project_sparse(basis)
        basis = np.asarray(basis)
        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        projected = basis.T.conj() @ self.matmat(basis)
        return 0.5 * (projected + projected.T.conj())

    def project_recursive(self, basis: RecursiveCoordinateBasis):
        """Return ``basis.T @ H @ basis`` without expanding recursive branches."""

        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        projected = self._project_recursive_between(basis, basis)
        return 0.5 * (projected + projected.T.conj())

    def _project_recursive_between(self, bra: RecursiveCoordinateBasis, ket: RecursiveCoordinateBasis):
        if bra.shape[0] != self.shape[0] or ket.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")

        if bra.leaf_vectors is None:
            blocks = [self._project_recursive_between(child, ket) for child in bra.children]
            dtype = np.result_type(self.dtype, bra.dtype, ket.dtype, complex)
            projected = (
                np.vstack(blocks)
                if blocks
                else np.zeros((0, ket.shape[1]), dtype=dtype)
            )
            if bra.coeff is not None:
                projected = bra.coeff.T.conj() @ projected
            return projected

        if ket.leaf_vectors is None:
            blocks = [self._project_recursive_between(bra, child) for child in ket.children]
            dtype = np.result_type(self.dtype, bra.dtype, ket.dtype, complex)
            projected = (
                np.hstack(blocks)
                if blocks
                else np.zeros((bra.shape[1], 0), dtype=dtype)
            )
            if ket.coeff is not None:
                projected = projected @ ket.coeff
            return projected

        return self._project_leaf_between(bra, ket)

    def _project_leaf_between(self, bra: RecursiveCoordinateBasis, ket: RecursiveCoordinateBasis):
        if bra.leaf_vectors is None or ket.leaf_vectors is None:
            raise ValueError("leaf projection requires leaf bases.")
        local_index = _row_localizer(bra.rows)
        bra_vectors = bra._leaf_effective_vectors()
        ket_vectors = ket._leaf_effective_vectors()
        nbra = bra.shape[1]
        nket = ket.shape[1]
        dtype = np.result_type(self.dtype, bra_vectors, ket_vectors, complex)
        projected = np.zeros((nbra, nket), dtype=dtype)
        for col in range(nket):
            local_out = np.zeros(_row_size(bra.rows), dtype=dtype)
            for row, value in zip(ket.rows, ket_vectors[:, col]):
                if value == 0:
                    continue
                for dest, amplitude in self.column_terms(row):
                    dest_loc = local_index(dest)
                    if dest_loc is not None:
                        local_out[dest_loc] += amplitude * value
            projected[:, col] = bra_vectors.T.conj() @ local_out
        return projected

    def project_local(self, basis: LocalCoordinateBasis):
        """Return ``basis.T @ H @ basis`` in a branch-local coordinate frame."""

        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        local_index = _row_localizer(basis.rows)
        ncols = basis.shape[1]
        projected = np.zeros((ncols, ncols), dtype=np.result_type(self.dtype, basis.vectors, complex))
        for col in range(ncols):
            local_out = np.zeros(basis.local_shape[0], dtype=np.result_type(self.dtype, basis.vectors))
            for loc, (row, value) in enumerate(zip(basis.rows, basis.vectors[:, col])):
                if value == 0:
                    continue
                for dest, amplitude in self.column_terms(row):
                    dest_loc = local_index(dest)
                    if dest_loc is not None:
                        local_out[dest_loc] += amplitude * value
            projected[:, col] = basis.vectors.T.conj() @ local_out
        return 0.5 * (projected + projected.T.conj())

    def project_sparse(self, basis):
        """Return ``basis.T @ H @ basis`` for sparse columns."""

        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        ncols = basis.shape[1]
        projected = np.zeros((ncols, ncols), dtype=np.result_type(self.dtype, complex))
        for col in range(ncols):
            rows, values = basis.column(col)
            out_rows, out_values = self.apply_sparse(rows, values)
            for bra in range(ncols):
                projected[bra, col] = basis.dot_sparse(bra, out_rows, out_values)
        return 0.5 * (projected + projected.T.conj())

    def to_dense(self):
        """Materialize the dense Hamiltonian, mainly for tests and diagnostics."""

        eye = np.eye(self.shape[0], dtype=self.dtype)
        dense = self.matmat(eye)
        return 0.5 * (dense + dense.T.conj())

    def exact_energies(self, nstates=None):
        """Return dense exact eigenvalues for small diagnostic calculations."""

        evals = np.linalg.eigvalsh(self.to_dense())
        if nstates is None:
            return evals
        return evals[: int(nstates)]


def sine_box_dvr(npoints: int, xmin: float = 0.0, xmax: float = 1.0, mass: float = 1.0):
    """Return grid points and the sine-DVR kinetic matrix for a box."""

    npoints = int(npoints)
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    length = float(xmax) - float(xmin)
    if length <= 0.0:
        raise ValueError("xmax must be larger than xmin.")
    mass = float(mass)
    if mass <= 0.0:
        raise ValueError("mass must be positive.")

    point_ids = np.arange(1, npoints + 1, dtype=float)
    mode_ids = np.arange(1, npoints + 1, dtype=float)
    grid = float(xmin) + length * point_ids / (npoints + 1)
    transform = np.sqrt(2.0 / (npoints + 1)) * np.sin(
        np.pi * np.outer(point_ids, mode_ids) / (npoints + 1)
    )
    kinetic_eigs = (np.pi * mode_ids / length) ** 2 / (2.0 * mass)
    kinetic = (transform * kinetic_eigs) @ transform.T
    return grid, 0.5 * (kinetic + kinetic.T)


def ordered_configurations(npoints: int, nelec: int):
    """Return same-spin ordered configurations ``i1 < i2 < ... < iN``."""

    npoints = int(npoints)
    nelec = int(nelec)
    if nelec < 1:
        raise ValueError("nelec must be positive.")
    if npoints < nelec:
        raise ValueError("npoints must be at least nelec.")
    return np.asarray(list(combinations(range(npoints), nelec)), dtype=int)


def _permutation_sign(values):
    inversions = 0
    values = tuple(int(value) for value in values)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            inversions += values[left] > values[right]
    return -1 if inversions % 2 else 1


def _coalesce_hops(cols, values):
    cols = np.asarray(cols, dtype=int)
    values = np.asarray(values)
    if cols.shape != values.shape:
        raise ValueError("kinetic hop columns and values must have the same shape.")
    if cols.size == 0:
        return cols, values
    order = np.argsort(cols)
    cols = cols[order]
    values = values[order]
    unique, starts = np.unique(cols, return_index=True)
    if unique.size == cols.size:
        return cols, values
    summed = np.add.reduceat(values, starts)
    keep = summed != 0
    return unique[keep], summed[keep]


def _normalise_hops(hops, npoints):
    try:
        cols, values = hops
    except (TypeError, ValueError) as exc:
        raise ValueError("kinetic hop callback must return (columns, values).") from exc
    cols, values = _coalesce_hops(cols, values)
    if np.any((cols < 0) | (cols >= int(npoints))):
        raise ValueError("kinetic hop column is outside the grid.")
    return cols.astype(int, copy=False), values


def _matrix_hops(matrix):
    matrix = np.asarray(matrix)
    hops = []
    for row in range(matrix.shape[0]):
        cols = np.flatnonzero(matrix[row] != 0)
        hops.append((cols.astype(int), matrix[row, cols].copy()))
    return tuple(hops)


def _transpose_hops(row_hops, npoints):
    cols_by_row = [[] for _ in range(int(npoints))]
    values_by_row = [[] for _ in range(int(npoints))]
    for row, (cols, values) in enumerate(row_hops):
        for col, value in zip(cols, values):
            cols_by_row[int(col)].append(row)
            values_by_row[int(col)].append(value)
    return tuple(
        _coalesce_hops(cols_by_row[row], values_by_row[row])
        for row in range(int(npoints))
    )


def _hops_dtype(row_hops):
    values = [values for _cols, values in row_hops if len(values)]
    if not values:
        return np.dtype(float)
    return np.result_type(*values)


def _as_kinetic_operator(kinetic, npoints):
    npoints = int(npoints)
    if callable(kinetic):
        row_hops = tuple(_normalise_hops(kinetic(site), npoints) for site in range(npoints))
        return None, row_hops, _transpose_hops(row_hops, npoints), _hops_dtype(row_hops)

    matrix = np.asarray(kinetic)
    if matrix.shape != (npoints, npoints):
        raise ValueError("kinetic must have shape (npoints, npoints) or be a hop callback.")
    row_hops = _matrix_hops(matrix)
    return matrix, row_hops, _matrix_hops(matrix.T), matrix.dtype


def _as_one_body_potential(external, grid):
    if external is None:
        return None, lambda _x: 0.0
    if callable(external):
        return None, external
    values = np.asarray(external)
    if values.shape != grid.shape:
        raise ValueError("one_body must be callable or an array with shape (npoints,).")
    return values, None


def _as_two_body_potential(interaction, grid):
    npoints = grid.size
    if interaction is None:
        return None, lambda _x, _y: 0.0
    if callable(interaction):
        return None, interaction
    values = np.asarray(interaction)
    if values.shape != (npoints, npoints):
        raise ValueError("two_body must be callable or an array with shape (npoints, npoints).")
    return values, None


def _one_body_values(external, grid):
    if external is None:
        return np.zeros_like(grid, dtype=float)
    if callable(external):
        return np.asarray([external(float(x)) for x in grid], dtype=float)
    values = np.asarray(external, dtype=float)
    if values.shape != grid.shape:
        raise ValueError("external must be callable or an array with shape (npoints,).")
    return values


def _two_body_values(interaction, grid):
    npoints = grid.size
    if interaction is None:
        return np.zeros((npoints, npoints), dtype=float)
    if callable(interaction):
        values = np.empty((npoints, npoints), dtype=float)
        for i, xi in enumerate(grid):
            for j, xj in enumerate(grid):
                values[i, j] = interaction(float(xi), float(xj))
        return values
    values = np.asarray(interaction, dtype=float)
    if values.shape != (npoints, npoints):
        raise ValueError("interaction must be callable or an array with shape (npoints, npoints).")
    return values


def ordered_operator(
    kinetic,
    grid: ArrayLike,
    *,
    nelec: int,
    external: Callable[[float], float] | ArrayLike | None = None,
    interaction: Callable[[float, float], float] | np.ndarray | None = None,
):
    """Return a matrix-free ordered-coordinate Hamiltonian.

    ``kinetic`` may be a dense one-particle matrix or a callback
    ``kinetic(site) -> (columns, values)`` that returns nonzero hops.
    """

    grid = np.asarray(grid, dtype=float)
    nelec = int(nelec)
    if not callable(kinetic):
        kinetic = np.asarray(kinetic, dtype=float)
        if kinetic.shape != (grid.size, grid.size):
            raise ValueError("kinetic must have shape (npoints, npoints) or be a hop callback.")
    configs = OrderedConfigurationSpace(grid.size, nelec)
    return ManyElectronOrderedOperator(
        kinetic=kinetic,
        grid=grid,
        nelec=nelec,
        one_body=external,
        two_body=interaction,
        configs=configs,
    )


def ordered_hamiltonian(
    kinetic,
    grid: ArrayLike,
    *,
    nelec: int,
    external: Callable[[float], float] | ArrayLike | None = None,
    interaction: Callable[[float, float], float] | np.ndarray | None = None,
):
    """Return the dense ordered-coordinate Hamiltonian and configurations."""

    operator = ordered_operator(
        kinetic,
        grid,
        nelec=nelec,
        external=external,
        interaction=interaction,
    )
    configs = operator.configs.to_array() if isinstance(operator.configs, OrderedConfigurationSpace) else operator.configs
    return operator.to_dense(), configs


def two_electron_wedge_hamiltonian(
    kinetic: np.ndarray,
    grid: ArrayLike,
    *,
    external: Callable[[float], float] | ArrayLike | None = None,
    interaction: Callable[[float, float], float] | np.ndarray | None = None,
    exchange: str = "antisymmetric",
):
    """Build the ordered-sector two-electron Hamiltonian on ``|i j>``, ``i < j``."""

    from pyqed.qchem.gdvr import two_electron_wedge_kinetic

    kinetic = np.asarray(kinetic, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if kinetic.shape != (grid.size, grid.size):
        raise ValueError("kinetic must have shape (npoints, npoints).")

    wedge_kinetic, pairs = two_electron_wedge_kinetic(kinetic, exchange=exchange)
    one_body = _one_body_values(external, grid)
    two_body = _two_body_values(interaction, grid)
    diagonal = one_body[pairs[:, 0]] + one_body[pairs[:, 1]] + two_body[pairs[:, 0], pairs[:, 1]]
    hamiltonian = wedge_kinetic + np.diag(diagonal)
    return 0.5 * (hamiltonian + hamiltonian.T), pairs


def conditional_branch_basis(hamiltonian: np.ndarray, pairs: np.ndarray, D: int):
    """Return the NARG branch basis grouped by the first ordered coordinate."""

    hamiltonian = np.asarray(hamiltonian)
    pairs = np.asarray(pairs, dtype=int)
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    if hamiltonian.shape != (pairs.shape[0], pairs.shape[0]):
        raise ValueError("hamiltonian shape is inconsistent with pairs.")

    columns = []
    branch_energies = []
    branch_rows = []
    for first in np.unique(pairs[:, 0]):
        rows = np.flatnonzero(pairs[:, 0] == first)
        block = hamiltonian[np.ix_(rows, rows)]
        evals, evecs = np.linalg.eigh(block)
        keep = min(D, rows.size)
        branch_energies.append(evals[:keep])
        branch_rows.append(rows)
        for alpha in range(keep):
            column = np.zeros(pairs.shape[0], dtype=evecs.dtype)
            column[rows] = evecs[:, alpha]
            columns.append(column)

    basis = np.column_stack(columns) if columns else np.zeros((pairs.shape[0], 0))
    return basis, branch_energies, branch_rows


def _recursive_prefix_basis(hamiltonian, configs, rows, D, depth, prefix, *, truncate_here):
    if rows.size == 0:
        basis = np.zeros((hamiltonian.shape[0], 0), dtype=hamiltonian.dtype)
        node = BranchNode(depth=depth, prefix=prefix, rows=rows, energies=np.empty(0), children=[])
        return basis, node

    nelec = configs.shape[1]
    if depth >= nelec - 1:
        block = hamiltonian[np.ix_(rows, rows)]
        evals, evecs = np.linalg.eigh(block)
        keep = min(int(D), rows.size)
        basis = np.zeros((hamiltonian.shape[0], keep), dtype=evecs.dtype)
        basis[rows, :] = evecs[:, :keep]
        node = BranchNode(depth=depth, prefix=prefix, rows=rows, energies=evals[:keep], children=[])
        return basis, node

    child_bases = []
    children = []
    for value in np.unique(configs[rows, depth]):
        child_rows = rows[configs[rows, depth] == value]
        child_basis, child_node = _recursive_prefix_basis(
            hamiltonian,
            configs,
            child_rows,
            D,
            depth + 1,
            (*prefix, int(value)),
            truncate_here=True,
        )
        child_bases.append(child_basis)
        children.append(child_node)

    candidate = np.column_stack(child_bases) if child_bases else np.zeros((hamiltonian.shape[0], 0))
    if not truncate_here:
        node = BranchNode(depth=depth, prefix=prefix, rows=rows, energies=np.empty(0), children=children)
        return candidate, node

    projected = candidate.T.conj() @ hamiltonian @ candidate
    evals, evecs = np.linalg.eigh(projected)
    keep = min(int(D), evals.size)
    basis = candidate @ evecs[:, :keep]
    node = BranchNode(depth=depth, prefix=prefix, rows=rows, energies=evals[:keep], children=children)
    return basis, node


def recursive_branch_basis(hamiltonian: np.ndarray, configs: np.ndarray, D: int):
    """Return a recursive ordered-coordinate NARG basis for ``configs``."""

    hamiltonian = np.asarray(hamiltonian)
    configs = np.asarray(configs, dtype=int)
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    if configs.ndim != 2:
        raise ValueError("configs must have shape (nconfigs, nelec).")
    if hamiltonian.shape != (configs.shape[0], configs.shape[0]):
        raise ValueError("hamiltonian shape is inconsistent with configs.")
    rows = np.arange(configs.shape[0])
    return _recursive_prefix_basis(hamiltonian, configs, rows, D, 0, (), truncate_here=False)


def _shared_suffix_key(operator, prefix):
    if not prefix:
        raise ValueError("a shared suffix key requires a nonempty prefix.")
    return "last-prefix", len(prefix), int(prefix[-1])


def _shared_suffix_training_prefixes(operator, prefix):
    if not isinstance(operator.configs, OrderedConfigurationSpace):
        return [prefix]
    prefix = tuple(int(value) for value in prefix)
    if len(prefix) != operator.nelec - 1 or operator.nelec < 2:
        return [prefix]

    last_fixed = int(prefix[-1])
    earlier_count = operator.nelec - 2
    return [
        (*earlier, last_fixed)
        for earlier in combinations(range(last_fixed), earlier_count)
    ]


def _shared_suffix_leaf_basis(operator, prefix, D, shared_leaf_cache):
    prefix = tuple(int(value) for value in prefix)
    rows = operator.prefix_rows(prefix)
    key = _shared_suffix_key(operator, prefix)
    D = int(D)

    if key not in shared_leaf_cache:
        samples = []
        for training_prefix in _shared_suffix_training_prefixes(operator, prefix):
            training_rows = operator.prefix_rows(training_prefix)
            if training_rows.size != rows.size:
                continue
            _evals, evecs = operator.local_lowest_eigenpairs(training_rows, D)
            if evecs.size:
                samples.append(evecs)

        if samples:
            stacked = np.column_stack(samples)
            uvecs, _singular_values, _vh = np.linalg.svd(stacked, full_matrices=False)
            keep = min(D, uvecs.shape[1])
            shared_leaf_cache[key] = uvecs[:, :keep]
        else:
            shared_leaf_cache[key] = np.zeros((rows.size, 0), dtype=operator.dtype)

    shared_vectors = shared_leaf_cache[key]
    shared_basis = LocalCoordinateBasis.from_local(operator.shape[0], rows, shared_vectors)
    projected = operator.project_local(shared_basis)
    evals, coeff = np.linalg.eigh(projected)
    keep = min(D, evals.size)
    basis = RecursiveCoordinateBasis.leaf(
        operator.shape[0],
        rows,
        shared_vectors,
        coeff=coeff[:, :keep],
        leaf_key=key,
    )
    return evals[:keep], basis




def _recursive_coordinate_branch(
    operator,
    prefix,
    D,
    *,
    truncate_here,
    share_suffix=False,
    shared_leaf_cache=None,
):
    prefix = tuple(int(value) for value in prefix)
    rows = operator.prefix_rows(prefix)
    depth = len(prefix)
    shared_leaf_cache = {} if shared_leaf_cache is None else shared_leaf_cache
    if rows.size == 0:
        basis = RecursiveCoordinateBasis.leaf(operator.shape[0], rows, np.zeros((0, 0), dtype=operator.dtype))
        return CoordinateBranch(prefix, depth, rows, basis, np.empty(0), [])

    if depth >= operator.nelec - 1:
        if share_suffix:
            evals, basis = _shared_suffix_leaf_basis(operator, prefix, D, shared_leaf_cache)
        else:
            evals, evecs = operator.local_lowest_eigenpairs(rows, D)
            basis = RecursiveCoordinateBasis.leaf(operator.shape[0], rows, evecs)
        return CoordinateBranch(prefix, depth, rows, basis, evals, [])

    child_bases = []
    children = []
    for value in operator.child_values(prefix):
        child = _recursive_coordinate_branch(
            operator,
            (*prefix, int(value)),
            D,
            truncate_here=True,
            share_suffix=share_suffix,
            shared_leaf_cache=shared_leaf_cache,
        )
        child_bases.append(child.basis)
        children.append(child)

    candidate = (
        RecursiveCoordinateBasis.hstack(child_bases, rows=rows)
        if child_bases
        else RecursiveCoordinateBasis.leaf(operator.shape[0], rows, np.zeros((rows.size, 0), dtype=operator.dtype))
    )
    if not truncate_here:
        return CoordinateBranch(prefix, depth, rows, candidate, np.empty(0), children)

    projected = operator.project(candidate)
    evals, evecs = np.linalg.eigh(projected)
    keep = min(int(D), evals.size)
    basis = candidate.combine(evecs[:, :keep])
    return CoordinateBranch(prefix, depth, rows, basis, evals[:keep], children)


def coordinate_tree_basis(
    operator: ManyElectronOrderedOperator,
    D: int,
    *,
    share_suffix: bool = False,
):
    """Return a recursive coordinate tree basis with prefix-level truncation."""

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    root = _recursive_coordinate_branch(
        operator,
        (),
        D,
        truncate_here=False,
        share_suffix=bool(share_suffix),
        shared_leaf_cache={},
    )
    return CoordinateTreeBasis(root)


def _validate_coordinate_suffix(operator, suffix):
    suffix = tuple(int(value) for value in suffix)
    if len(suffix) > operator.nelec:
        raise ValueError("coordinate suffix is longer than nelec.")
    if any(suffix[idx] >= suffix[idx + 1] for idx in range(len(suffix) - 1)):
        raise ValueError("coordinate suffix must be strictly ordered.")
    if suffix and (suffix[0] < 0 or suffix[-1] >= operator.grid.size):
        raise ValueError("coordinate suffix is outside the grid.")
    return suffix


def _rows_with_coordinate_suffix(operator, suffix):
    """Return ordered-sector rows whose rightmost coordinates equal ``suffix``."""

    suffix = _validate_coordinate_suffix(operator, suffix)
    if not suffix:
        return operator.prefix_rows(())

    free_count = operator.nelec - len(suffix)
    if free_count < 0 or suffix[0] < free_count:
        return np.empty(0, dtype=int)

    if isinstance(operator.configs, OrderedConfigurationSpace):
        rows = [
            operator.row_index((*prefix, *suffix))
            for prefix in combinations(range(suffix[0]), free_count)
        ]
        return np.asarray(rows, dtype=int)

    mask = np.ones(operator.configs.shape[0], dtype=bool)
    for offset, value in enumerate(suffix, start=operator.nelec - len(suffix)):
        mask &= operator.configs[:, offset] == value
    return np.flatnonzero(mask).astype(int, copy=False)


def _coordinate_suffix_has_rows(operator, suffix):
    """Cheap non-emptiness test for a fixed right-coordinate suffix."""

    suffix = _validate_coordinate_suffix(operator, suffix)
    if not suffix:
        return operator.shape[0] > 0

    free_count = operator.nelec - len(suffix)
    if free_count < 0 or suffix[0] < free_count:
        return False
    if isinstance(operator.configs, OrderedConfigurationSpace):
        return True
    return _rows_with_coordinate_suffix(operator, suffix).size > 0


def _coordinate_suffix_child_values(operator, suffix):
    suffix = _validate_coordinate_suffix(operator, suffix)
    free_count = operator.nelec - len(suffix)
    if free_count <= 0:
        return np.empty(0, dtype=int)

    if not isinstance(operator.configs, OrderedConfigurationSpace):
        rows = _rows_with_coordinate_suffix(operator, suffix)
        if rows.size == 0:
            return np.empty(0, dtype=int)
        return np.unique(operator.configs[rows, free_count - 1])

    upper = suffix[0] if suffix else operator.grid.size
    lower = free_count - 1
    if lower >= upper:
        return np.empty(0, dtype=int)
    return np.arange(lower, upper, dtype=int)


def _stage_ordered_operator(operator, nelec):
    nelec = int(nelec)
    if nelec == operator.nelec:
        return operator
    kinetic = operator.kinetic
    if kinetic is None:
        kinetic = lambda site: operator.kinetic_terms(site)
    return ordered_operator(
        kinetic,
        operator.grid,
        nelec=nelec,
        external=operator.one_body,
        interaction=operator.two_body,
    )


def _append_coordinate_rows(previous_operator, current_operator, rows, value):
    value = int(value)
    mapped = [
        current_operator.row_index((*previous_operator.config(row), value))
        for row in _row_array(rows)
    ]
    return np.asarray(mapped, dtype=int)


def _append_coordinate_basis(previous_basis, previous_operator, current_operator, value):
    rows = _append_coordinate_rows(previous_operator, current_operator, previous_basis.rows, value)
    children = [
        _append_coordinate_basis(child, previous_operator, current_operator, value)
        for child in previous_basis.children
    ]
    return RecursiveCoordinateBasis(
        nrows=current_operator.shape[0],
        rows=rows,
        children=children,
        coeff=previous_basis.coeff,
        leaf_vectors=previous_basis.leaf_vectors,
        leaf_key=previous_basis.leaf_key,
    )


def _append_coordinate_branch(previous_branch, previous_operator, current_operator, value):
    basis = _append_coordinate_basis(previous_branch.basis, previous_operator, current_operator, value)
    children = [
        _append_coordinate_branch(child, previous_operator, current_operator, value)
        for child in previous_branch.children
    ]
    return CoordinateBranch(
        prefix=(*previous_branch.prefix, int(value)),
        depth=previous_branch.depth + 1,
        rows=basis.rows,
        basis=basis,
        energies=previous_branch.energies,
        children=children,
    )


def _kinetic_diagonal(operator, site):
    sites, amplitudes = operator.kinetic_terms(site)
    matches = np.flatnonzero(sites == int(site))
    if matches.size == 0:
        return 0.0
    return amplitudes[int(matches[0])]


def _kinetic_amplitude(operator, row_site, col_site):
    sites, amplitudes = operator.kinetic_terms(col_site, transpose=True)
    matches = np.flatnonzero(sites == int(row_site))
    if matches.size == 0:
        return 0.0
    return amplitudes[int(matches[0])]


def _is_nearest_neighbor_kinetic(operator, *, atol=0.0):
    """Return True when kinetic hops cannot cross an occupied coordinate."""

    if atol == 0.0:
        cached = getattr(operator, "_narg_nearest_neighbor_kinetic", None)
        if cached is not None:
            return bool(cached)

    is_nearest = True
    for site in range(operator.grid.size):
        for transpose in (False, True):
            sites, amplitudes = operator.kinetic_terms(site, transpose=transpose)
            for target, amplitude in zip(sites, amplitudes):
                if np.abs(amplitude) <= atol:
                    continue
                if abs(int(target) - int(site)) > 1:
                    is_nearest = False
                    break
            if not is_nearest:
                break
        if not is_nearest:
            break

    if atol == 0.0:
        try:
            operator._narg_nearest_neighbor_kinetic = is_nearest
        except Exception:
            pass
    return is_nearest


def _block_candidate_offsets(block: ParticleGrowthBlock):
    offsets = {}
    offset = 0
    if block.previous_layer is None:
        return offsets
    for value in block.child_values:
        width = block.previous_layer.blocks[value].shape[1]
        offsets[value] = slice(offset, offset + width)
        offset += width
    return offsets


def _project_block_candidate_matrix(block: ParticleGrowthBlock, candidate_matrix):
    candidate_matrix = np.asarray(candidate_matrix)
    return block.coeff.T.conj() @ candidate_matrix @ block.coeff


def _candidate_overlap_between_blocks(bra: ParticleGrowthBlock, ket: ParticleGrowthBlock):
    if bra.previous_layer is None and ket.previous_layer is None:
        return bra.coeff.T.conj() @ ket.coeff
    if bra.previous_layer is not ket.previous_layer:
        raise ValueError("particle-growth block overlap requires a shared previous layer.")

    bra_offsets = _block_candidate_offsets(bra)
    ket_offsets = _block_candidate_offsets(ket)
    dtype = np.result_type(bra.dtype, ket.dtype, complex)
    overlap = np.zeros((bra.shape[1], ket.shape[1]), dtype=dtype)
    for value in sorted(set(bra.child_values) & set(ket.child_values)):
        bra_slice = bra_offsets[value]
        ket_slice = ket_offsets[value]
        overlap += bra.coeff[bra_slice, :].T.conj() @ ket.coeff[ket_slice, :]
    return overlap


def _project_interaction_with_site(block: ParticleGrowthBlock, operator, site, cache=None):
    cache = {} if cache is None else cache
    site = int(site)
    key = (id(block), site)
    if key in cache:
        return cache[key]

    if block.previous_layer is None:
        value = operator.two_body_value(block.value, site)
        projected = np.asarray([[value]], dtype=np.result_type(operator.dtype, block.dtype))
        cache[key] = projected
        return projected

    offsets = _block_candidate_offsets(block)
    candidate = np.zeros((block.candidate_size, block.candidate_size), dtype=np.result_type(operator.dtype, block.dtype, complex))
    site_value = operator.two_body_value(block.value, site)
    for child_value in block.child_values:
        child = block.previous_layer.blocks[child_value]
        child_slice = offsets[child_value]
        child_width = child.shape[1]
        child_projected = _project_interaction_with_site(child, operator, site, cache)
        child_projected = child_projected + site_value * np.eye(child_width)
        candidate[child_slice, child_slice] = child_projected
    projected = _project_block_candidate_matrix(block, candidate)
    cache[key] = projected
    return projected


def _project_particle_growth_blocks(operator, bra: ParticleGrowthBlock, ket: ParticleGrowthBlock):
    if bra.value == ket.value:
        return np.diag(bra.energies)

    amplitude = _kinetic_amplitude(operator, bra.value, ket.value)
    if amplitude == 0:
        return np.zeros((bra.shape[1], ket.shape[1]), dtype=np.result_type(operator.dtype, bra.dtype, ket.dtype, complex))
    return amplitude * _candidate_overlap_between_blocks(bra, ket)


def _particle_growth_block_to_branch(block: ParticleGrowthBlock, cache=None):
    cache = {} if cache is None else cache
    key = id(block)
    if key in cache:
        return cache[key]

    operator = block.previous_layer.operator if block.previous_layer is not None else None
    if block.previous_layer is None:
        # The one-particle layer stores one grid point per block.
        # The current operator is recovered from the cache owner in callers by
        # using the block's rows after appending; for a bare conversion we build
        # it from the value-only one-particle coordinate space.
        raise ValueError("one-particle block conversion needs the current operator.")

    previous_operator = block.previous_layer.operator
    current_operator = _stage_ordered_operator(previous_operator, previous_operator.nelec + 1)
    children = [
        _append_coordinate_branch(
            _particle_growth_block_to_branch_with_operator(block.previous_layer.blocks[value], previous_operator, cache),
            previous_operator,
            current_operator,
            block.value,
        )
        for value in block.child_values
    ]
    child_bases = [child.basis for child in children]
    rows = _rows_with_coordinate_suffix(current_operator, (block.value,))
    candidate = RecursiveCoordinateBasis.hstack(child_bases, rows=rows)
    basis = candidate.combine(block.coeff)
    branch = CoordinateBranch(
        prefix=(block.value,),
        depth=current_operator.nelec,
        rows=rows,
        basis=basis,
        energies=block.energies,
        children=children,
    )
    cache[key] = branch
    return branch


def _particle_growth_block_to_branch_with_operator(block: ParticleGrowthBlock, current_operator, cache=None):
    cache = {} if cache is None else cache
    key = (id(block), current_operator.nelec)
    if key in cache:
        return cache[key]

    if block.previous_layer is None:
        rows = _rows_with_coordinate_suffix(current_operator, (block.value,))
        basis = RecursiveCoordinateBasis.leaf(
            current_operator.shape[0],
            rows,
            np.ones((rows.size, 1), dtype=current_operator.dtype),
            coeff=block.coeff,
            leaf_key=("particle-site", block.value),
        )
        branch = CoordinateBranch(
            prefix=(block.value,),
            depth=1,
            rows=rows,
            basis=basis,
            energies=block.energies,
            children=[],
        )
        cache[key] = branch
        return branch

    previous_operator = block.previous_layer.operator
    children = [
        _append_coordinate_branch(
            _particle_growth_block_to_branch_with_operator(block.previous_layer.blocks[value], previous_operator, cache),
            previous_operator,
            current_operator,
            block.value,
        )
        for value in block.child_values
    ]
    child_bases = [child.basis for child in children]
    rows = _rows_with_coordinate_suffix(current_operator, (block.value,))
    candidate = RecursiveCoordinateBasis.hstack(child_bases, rows=rows)
    basis = candidate.combine(block.coeff)
    branch = CoordinateBranch(
        prefix=(block.value,),
        depth=current_operator.nelec,
        rows=rows,
        basis=basis,
        energies=block.energies,
        children=children,
    )
    cache[key] = branch
    return branch


def _project_diagonal_between(bra: RecursiveCoordinateBasis, ket: RecursiveCoordinateBasis, value_by_row):
    if bra.leaf_vectors is None:
        blocks = [_project_diagonal_between(child, ket, value_by_row) for child in bra.children]
        dtype = np.result_type(bra.dtype, ket.dtype, complex)
        projected = (
            np.vstack(blocks)
            if blocks
            else np.zeros((0, ket.shape[1]), dtype=dtype)
        )
        if bra.coeff is not None:
            projected = bra.coeff.T.conj() @ projected
        return projected

    if ket.leaf_vectors is None:
        blocks = [_project_diagonal_between(bra, child, value_by_row) for child in ket.children]
        dtype = np.result_type(bra.dtype, ket.dtype, complex)
        projected = (
            np.hstack(blocks)
            if blocks
            else np.zeros((bra.shape[1], 0), dtype=dtype)
        )
        if ket.coeff is not None:
            projected = projected @ ket.coeff
        return projected

    bra_rows = _row_array(bra.rows)
    ket_rows = _row_array(ket.rows)
    common, bra_idx, ket_idx = np.intersect1d(
        bra_rows,
        ket_rows,
        assume_unique=True,
        return_indices=True,
    )
    dtype = np.result_type(bra.dtype, ket.dtype, complex)
    if common.size == 0:
        return np.zeros((bra.shape[1], ket.shape[1]), dtype=dtype)

    bra_vectors = bra._leaf_effective_vectors()[bra_idx, :]
    ket_vectors = ket._leaf_effective_vectors()[ket_idx, :]
    values = np.asarray([value_by_row(int(row)) for row in common], dtype=dtype)
    return bra_vectors.T.conj() @ (values[:, None] * ket_vectors)


def _project_diagonal(basis: RecursiveCoordinateBasis, value_by_row):
    projected = _project_diagonal_between(basis, basis, value_by_row)
    return 0.5 * (projected + projected.T.conj())


def _adjacent_growth_projected_hamiltonian(previous_layer: ParticleGrowthLayer, current_operator, child_values, value):
    child_values = tuple(int(child_value) for child_value in child_values)
    projected = previous_layer.project_values(child_values)
    value = int(value)

    offsets = {}
    offset = 0
    for child_value in child_values:
        width = previous_layer.blocks[child_value].shape[1]
        offsets[child_value] = slice(offset, offset + width)
        offset += width

    scalar = current_operator.one_body_value(value) + _kinetic_diagonal(current_operator, value)
    projected = projected + scalar * np.eye(offset, dtype=np.result_type(projected, current_operator.dtype))

    interaction_cache = {}
    for child_value in child_values:
        child = previous_layer.blocks[child_value]
        child_slice = offsets[child_value]
        projected[child_slice, child_slice] += _project_interaction_with_site(
            child,
            current_operator,
            value,
            cache=interaction_cache,
        )
    return 0.5 * (projected + projected.T.conj())


def _one_particle_growth_layer(operator, D):
    blocks = {}
    for site in range(operator.grid.size):
        if not _coordinate_suffix_has_rows(operator, (site,)):
            continue
        energy = operator.one_body_value(site) + _kinetic_diagonal(operator, site)
        blocks[int(site)] = ParticleGrowthBlock(
            value=int(site),
            coeff=np.ones((1, 1), dtype=operator.dtype),
            energies=np.asarray([energy], dtype=operator.dtype),
        )
    return ParticleGrowthLayer(operator=operator, blocks=blocks, D=D)


def _adjacent_particle_growth_layer(previous_layer: ParticleGrowthLayer, current_operator, D):
    blocks = {}
    for value in range(current_operator.nelec - 1, current_operator.grid.size):
        if not _coordinate_suffix_has_rows(current_operator, (value,)):
            continue
        child_values = tuple(previous_value for previous_value in previous_layer.values if previous_value < value)
        if not child_values:
            continue

        projected = _adjacent_growth_projected_hamiltonian(
            previous_layer,
            current_operator,
            child_values,
            value,
        )
        evals, evecs = np.linalg.eigh(projected)
        keep = min(int(D), evals.size)
        blocks[int(value)] = ParticleGrowthBlock(
            value=int(value),
            coeff=evecs[:, :keep],
            energies=evals[:keep],
            previous_layer=previous_layer,
            child_values=child_values,
        )
    return ParticleGrowthLayer(operator=current_operator, blocks=blocks, D=D)


def particle_growth_layer(operator: ManyElectronOrderedOperator, D: int):
    """Return a retained nearest-neighbor particle-growth layer.

    This is a nearest-neighbor particle grow in first quantization.  When
    electron ``k`` is added, the previous ``k - 1`` electron layer is embedded
    under each allowed coordinate of electron ``k`` and only the adjacent
    coordinate layer is reoptimized/truncated to ``D``.  Earlier layers are
    reused as renormalized blocks rather than solved again for every new
    electron coordinate.
    """

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    if operator.nelec < 2:
        raise ValueError("particle_growth_basis requires at least two electrons.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")

    previous_operator = _stage_ordered_operator(operator, 1)
    layer = _one_particle_growth_layer(previous_operator, D)
    for nelec in range(2, operator.nelec + 1):
        current_operator = _stage_ordered_operator(operator, nelec)
        layer = _adjacent_particle_growth_layer(layer, current_operator, D)
        previous_operator = current_operator

    if layer.operator is not operator:
        layer = ParticleGrowthLayer(operator=operator, blocks=layer.blocks, D=D)
    return layer


def particle_growth_basis(operator: ManyElectronOrderedOperator, D: int):
    """Return a compatibility coordinate-tree view of the particle-growth layer."""

    return particle_growth_layer(operator, D).to_tree()


def two_electron_first_quantized_narg(
    hamiltonian: np.ndarray,
    pairs: np.ndarray,
    *,
    D: int,
    nstates: int = 4,
    exchange: str = "antisymmetric",
    exact: bool = True,
):
    """Diagonalize a branch-compressed ordered-wedge Hamiltonian."""

    hamiltonian = np.asarray(hamiltonian)
    pairs = np.asarray(pairs, dtype=int)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square.")
    if hamiltonian.shape[0] != pairs.shape[0]:
        raise ValueError("hamiltonian shape is inconsistent with pairs.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    basis, branch_energies, branch_rows = conditional_branch_basis(hamiltonian, pairs, D)
    projected = basis.T.conj() @ hamiltonian @ basis
    evals, coeff = np.linalg.eigh(projected)
    keep_states = min(nstates, evals.size)
    vectors = basis @ coeff[:, :keep_states]
    exact_energies = None
    if exact:
        exact_energies = np.linalg.eigvalsh(hamiltonian)[:keep_states]

    return TwoElectronFirstQuantizedNARGResult(
        energies=evals[:keep_states],
        vectors=vectors,
        projected_hamiltonian=projected,
        branch_basis=basis,
        branch_energies=branch_energies,
        branch_rows=branch_rows,
        exact_energies=exact_energies,
        hamiltonian=hamiltonian,
        pairs=pairs,
        D=int(D),
        exchange=str(exchange),
    )


def narg(
    hamiltonian,
    configs=None,
    *,
    D: int,
    nstates: int = 4,
    exact: bool | None = None,
    share_suffix: bool = False,
):
    """Run ordered-coordinate NARG for either dense or matrix-free input."""

    if isinstance(hamiltonian, ManyElectronOrderedOperator):
        if configs is not None:
            raise ValueError("configs is inferred from the matrix-free operator.")
        return narg_matrix_free(
            hamiltonian,
            D=D,
            nstates=nstates,
            exact=False if exact is None else exact,
            share_suffix=share_suffix,
        )
    if share_suffix:
        raise ValueError("share_suffix is only available for matrix-free ordered operators.")
    if configs is None:
        raise ValueError("configs is required when hamiltonian is a dense array.")
    exact = True if exact is None else exact

    hamiltonian = np.asarray(hamiltonian)
    configs = np.asarray(configs, dtype=int)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square.")
    if configs.ndim != 2:
        raise ValueError("configs must have shape (nconfigs, nelec).")
    if hamiltonian.shape[0] != configs.shape[0]:
        raise ValueError("hamiltonian shape is inconsistent with configs.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    basis, _tree = recursive_branch_basis(hamiltonian, configs, D)
    projected = basis.T.conj() @ hamiltonian @ basis
    evals, coeff = np.linalg.eigh(projected)
    keep_states = min(nstates, evals.size)
    vectors = basis @ coeff[:, :keep_states]
    exact_energies = None
    if exact:
        exact_energies = np.linalg.eigvalsh(hamiltonian)[:keep_states]

    return ManyElectronFirstQuantizedNARGResult(
        energies=evals[:keep_states],
        vectors=vectors,
        projected_hamiltonian=projected,
        branch_basis=basis,
        exact_energies=exact_energies,
        hamiltonian=hamiltonian,
        configs=configs,
        D=int(D),
        nelec=int(configs.shape[1]),
    )


def narg_matrix_free(
    operator: ManyElectronOrderedOperator,
    *,
    D: int,
    nstates: int = 4,
    exact: bool = False,
    share_suffix: bool = False,
):
    """Run ordered-coordinate NARG without materializing the dense Hamiltonian."""

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    basis = coordinate_tree_basis(operator, D, share_suffix=share_suffix)
    projected = basis.project(operator)
    evals, coeff = np.linalg.eigh(projected)
    keep_states = min(nstates, evals.size)
    vectors = basis.truncate(coeff[:, :keep_states])
    exact_energies = None
    if exact:
        exact_energies = operator.exact_energies(keep_states)

    return ManyElectronFirstQuantizedNARGResult(
        energies=evals[:keep_states],
        vectors=vectors,
        projected_hamiltonian=projected,
        branch_basis=basis,
        exact_energies=exact_energies,
        hamiltonian=operator,
        configs=operator.configs,
        D=int(D),
        nelec=int(operator.configs.shape[1]),
    )


def particle_growth_narg(
    operator: ManyElectronOrderedOperator,
    *,
    D: int,
    nstates: int = 4,
    exact: bool = False,
):
    """Run the particle-growth ordered-coordinate NARG prototype."""

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    basis = particle_growth_layer(operator, D)
    projected = basis.project(operator)
    evals, coeff = np.linalg.eigh(projected)
    keep_states = min(nstates, evals.size)
    vectors = basis.truncate(coeff[:, :keep_states])
    exact_energies = None
    if exact:
        exact_energies = operator.exact_energies(keep_states)

    return ParticleGrowthNARGResult(
        energies=evals[:keep_states],
        vectors=vectors,
        projected_hamiltonian=projected,
        branch_basis=basis,
        exact_energies=exact_energies,
        hamiltonian=operator,
        configs=operator.configs,
        D=int(D),
        nelec=int(operator.configs.shape[1]),
    )


__all__ = [
    "BranchNode",
    "CoordinateBranch",
    "CoordinateTreeBasis",
    "LocalCoordinateBasis",
    "ManyElectronFirstQuantizedNARGResult",
    "ManyElectronOrderedOperator",
    "OrderedConfigurationSpace",
    "ParticleGrowthLayer",
    "ParticleGrowthNARGResult",
    "ParticleGrowthState",
    "PrefixCoordinateSpace",
    "RecursiveCoordinateBasis",
    "SparseBasis",
    "TwoElectronFirstQuantizedNARGResult",
    "conditional_branch_basis",
    "coordinate_tree_basis",
    "narg",
    "narg_matrix_free",
    "ordered_hamiltonian",
    "ordered_operator",
    "ordered_configurations",
    "particle_growth_basis",
    "particle_growth_layer",
    "particle_growth_narg",
    "recursive_branch_basis",
    "sine_box_dvr",
    "two_electron_first_quantized_narg",
    "two_electron_wedge_hamiltonian",
]
