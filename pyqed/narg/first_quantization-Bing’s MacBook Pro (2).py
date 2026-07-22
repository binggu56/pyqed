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
            return np.empty(0, dtype=int)
        first = self.first_rank_with_prefix(prefix)
        return np.arange(first, first + count, dtype=int)

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
    rows: np.ndarray
    vectors: np.ndarray

    def __post_init__(self):
        self.nrows = int(self.nrows)
        self.rows = np.asarray(self.rows, dtype=int)
        self.vectors = np.asarray(self.vectors)
        if self.rows.ndim != 1:
            raise ValueError("rows must be one-dimensional.")
        if self.vectors.ndim == 1:
            self.vectors = self.vectors[:, None]
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be one- or two-dimensional.")
        if self.vectors.shape[0] != self.rows.size:
            raise ValueError("vectors has incompatible local dimension.")
        if self.rows.size and (self.rows[0] < 0 or self.rows[-1] >= self.nrows):
            raise ValueError("rows are outside the global dimension.")
        if np.any(self.rows[:-1] >= self.rows[1:]):
            raise ValueError("rows must be strictly increasing.")

    @property
    def shape(self):
        return self.nrows, self.vectors.shape[1]

    @property
    def local_shape(self):
        return self.rows.size, self.vectors.shape[1]

    @classmethod
    def from_local(cls, nrows: int, rows, local_vectors):
        rows = np.asarray(rows, dtype=int)
        local_vectors = np.asarray(local_vectors)
        if local_vectors.ndim == 1:
            local_vectors = local_vectors[:, None]
        if local_vectors.shape[0] != rows.size:
            raise ValueError("local_vectors has incompatible leading dimension.")
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
            rows = np.unique(np.concatenate([basis.rows for basis in bases]))
        rows = np.asarray(rows, dtype=int)
        ncols = sum(basis.shape[1] for basis in bases)
        dtype = np.result_type(*[basis.vectors for basis in bases], float)
        local = np.zeros((rows.size, ncols), dtype=dtype)
        offset = 0
        for basis in bases:
            loc = np.searchsorted(rows, basis.rows)
            if loc.size and (loc[-1] >= rows.size or not np.array_equal(rows[loc], basis.rows)):
                raise ValueError("basis rows are not contained in the requested local rows.")
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
        return LocalCoordinateBasis(self.nrows, self.rows.copy(), self.vectors @ coeff)

    def dot_sparse(self, index, rows, values):
        col_rows, col_values = self.column(index)
        return _sparse_inner(col_rows, col_values, rows, values)

    def to_sparse(self):
        return SparseBasis.from_local(self.nrows, self.rows, self.vectors)

    def to_dense(self):
        dense = np.zeros((self.nrows, self.shape[1]), dtype=np.result_type(self.vectors, float))
        dense[self.rows, :] = self.vectors
        return dense


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
    rows: np.ndarray
    basis: LocalCoordinateBasis
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

    def submatrix(self, rows):
        """Build a small dense restriction ``H[rows, rows]`` for local branches."""

        rows = np.asarray(rows, dtype=int)
        local_index = {int(row): pos for pos, row in enumerate(rows)}
        block = np.zeros((rows.size, rows.size), dtype=self.dtype)
        for local_row, row in enumerate(rows):
            for col, amplitude in self.row_terms(row):
                local_col = local_index.get(col)
                if local_col is not None:
                    block[local_row, local_col] += amplitude
        return 0.5 * (block + block.T.conj())

    def project(self, basis):
        """Return ``basis.T @ H @ basis`` using matrix-free applications."""

        if isinstance(basis, LocalCoordinateBasis):
            return self.project_local(basis)
        if isinstance(basis, SparseBasis):
            return self.project_sparse(basis)
        basis = np.asarray(basis)
        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        projected = basis.T.conj() @ self.matmat(basis)
        return 0.5 * (projected + projected.T.conj())

    def project_local(self, basis: LocalCoordinateBasis):
        """Return ``basis.T @ H @ basis`` in a branch-local coordinate frame."""

        if basis.shape[0] != self.shape[0]:
            raise ValueError("basis has incompatible leading dimension.")
        row_index = {int(row): loc for loc, row in enumerate(basis.rows)}
        ncols = basis.shape[1]
        projected = np.zeros((ncols, ncols), dtype=np.result_type(self.dtype, basis.vectors, complex))
        for col in range(ncols):
            local_out = np.zeros(basis.local_shape[0], dtype=np.result_type(self.dtype, basis.vectors))
            for loc, (row, value) in enumerate(zip(basis.rows, basis.vectors[:, col])):
                if value == 0:
                    continue
                for dest, amplitude in self.column_terms(row):
                    dest_loc = row_index.get(int(dest))
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




def _recursive_coordinate_branch(operator, prefix, D, *, truncate_here):
    prefix = tuple(int(value) for value in prefix)
    rows = operator.prefix_rows(prefix)
    depth = len(prefix)
    if rows.size == 0:
        basis = LocalCoordinateBasis(operator.shape[0], rows, np.zeros((0, 0), dtype=operator.dtype))
        return CoordinateBranch(prefix, depth, rows, basis, np.empty(0), [])

    if depth >= operator.nelec - 1:
        block = operator.submatrix(rows)
        evals, evecs = np.linalg.eigh(block)
        keep = min(int(D), rows.size)
        basis = LocalCoordinateBasis.from_local(operator.shape[0], rows, evecs[:, :keep])
        return CoordinateBranch(prefix, depth, rows, basis, evals[:keep], [])

    child_bases = []
    children = []
    for value in operator.child_values(prefix):
        child = _recursive_coordinate_branch(
            operator,
            (*prefix, int(value)),
            D,
            truncate_here=True,
        )
        child_bases.append(child.basis)
        children.append(child)

    candidate = (
        LocalCoordinateBasis.hstack(child_bases, rows=rows)
        if child_bases
        else LocalCoordinateBasis(operator.shape[0], rows, np.zeros((rows.size, 0), dtype=operator.dtype))
    )
    if not truncate_here:
        return CoordinateBranch(prefix, depth, rows, candidate, np.empty(0), children)

    projected = operator.project(candidate)
    evals, evecs = np.linalg.eigh(projected)
    keep = min(int(D), evals.size)
    basis = candidate.combine(evecs[:, :keep])
    return CoordinateBranch(prefix, depth, rows, basis, evals[:keep], children)


def coordinate_tree_basis(operator: ManyElectronOrderedOperator, D: int):
    """Return a recursive coordinate tree basis with prefix-level truncation."""

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    root = _recursive_coordinate_branch(operator, (), D, truncate_here=False)
    return CoordinateTreeBasis(root)


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


def narg(hamiltonian, configs=None, *, D: int, nstates: int = 4, exact: bool | None = None):
    """Run ordered-coordinate NARG for either dense or matrix-free input."""

    if isinstance(hamiltonian, ManyElectronOrderedOperator):
        if configs is not None:
            raise ValueError("configs is inferred from the matrix-free operator.")
        return narg_matrix_free(
            hamiltonian,
            D=D,
            nstates=nstates,
            exact=False if exact is None else exact,
        )
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
):
    """Run ordered-coordinate NARG without materializing the dense Hamiltonian."""

    if not isinstance(operator, ManyElectronOrderedOperator):
        raise TypeError("operator must be a ManyElectronOrderedOperator.")
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")

    basis = coordinate_tree_basis(operator, D)
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


__all__ = [
    "BranchNode",
    "CoordinateBranch",
    "CoordinateTreeBasis",
    "LocalCoordinateBasis",
    "ManyElectronFirstQuantizedNARGResult",
    "ManyElectronOrderedOperator",
    "OrderedConfigurationSpace",
    "SparseBasis",
    "TwoElectronFirstQuantizedNARGResult",
    "conditional_branch_basis",
    "coordinate_tree_basis",
    "narg",
    "narg_matrix_free",
    "ordered_hamiltonian",
    "ordered_operator",
    "ordered_configurations",
    "recursive_branch_basis",
    "sine_box_dvr",
    "two_electron_first_quantized_narg",
    "two_electron_wedge_hamiltonian",
]
