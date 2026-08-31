"""Finite rectangular projected-entangled-pair states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from pyqed.lattice import Site


AXES = ("left", "right", "up", "down", "physical")
_VIRTUAL_AXES = AXES[:4]


def _charge_tuple(charge):
    if isinstance(charge, tuple):
        return tuple(int(value) for value in charge)
    if isinstance(charge, (list, np.ndarray)):
        return tuple(int(value) for value in charge)
    return (int(charge),)


@dataclass(frozen=True)
class AbelianPEPSTensor:
    """Block-sparse rank-five PEPS tensor with Abelian leg charges.

    Blocks are keyed by ``(q_left, q_right, q_up, q_down, q_physical)``.
    ``sectors`` and ``sector_dims`` define the dense ordering on every leg.
    """

    blocks: dict
    sectors: tuple
    sector_dims: tuple
    directions: tuple = (-1, 1, -1, 1, 1)
    total_charge: tuple | None = None

    def __post_init__(self):
        if len(self.sectors) != 5 or len(self.sector_dims) != 5:
            raise ValueError("an Abelian PEPS tensor must describe five legs.")
        if len(self.directions) != 5 or any(d not in {-1, 1} for d in self.directions):
            raise ValueError("directions must contain five +1/-1 entries.")
        sectors = tuple(tuple(_charge_tuple(q) for q in leg) for leg in self.sectors)
        dims = tuple(
            {_charge_tuple(q): int(size) for q, size in dict(leg).items()}
            for leg in self.sector_dims
        )
        blocks = {}
        for raw_key, raw_block in dict(self.blocks).items():
            key = tuple(_charge_tuple(q) for q in raw_key)
            if len(key) != 5:
                raise ValueError("Abelian PEPS block keys must have length five.")
            expected = tuple(dims[axis][key[axis]] for axis in range(5))
            block = np.asarray(raw_block)
            if block.shape != expected:
                raise ValueError(
                    f"block {key!r} has shape {block.shape}; expected {expected}."
                )
            blocks[key] = np.array(block, copy=True)
        total = None if self.total_charge is None else _charge_tuple(self.total_charge)
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "sector_dims", dims)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "total_charge", total)
        self.check_flux()

    @property
    def shape(self):
        return tuple(sum(leg.values()) for leg in self.sector_dims)

    @property
    def ndim(self):
        return 5

    def copy(self):
        return type(self)(
            {key: block.copy() for key, block in self.blocks.items()},
            self.sectors,
            self.sector_dims,
            self.directions,
            self.total_charge,
        )

    def check_flux(self):
        """Validate charge conservation when ``total_charge`` is specified."""
        if self.total_charge is None:
            return True
        for key in self.blocks:
            rank = len(key[0])
            flux = tuple(
                sum(self.directions[axis] * key[axis][component] for axis in range(5))
                for component in range(rank)
            )
            if flux != self.total_charge:
                raise ValueError(
                    f"block {key!r} has flux {flux}, expected {self.total_charge}."
                )
        return True

    def to_dense(self):
        result = np.zeros(
            self.shape,
            dtype=np.result_type(*[block.dtype for block in self.blocks.values()])
            if self.blocks
            else float,
        )
        slices = []
        for leg_sectors, leg_dims in zip(self.sectors, self.sector_dims):
            offset = 0
            mapping = {}
            for charge in leg_sectors:
                size = leg_dims[charge]
                mapping[charge] = slice(offset, offset + size)
                offset += size
            slices.append(mapping)
        for key, block in self.blocks.items():
            result[tuple(slices[axis][key[axis]] for axis in range(5))] = block
        return result

    @classmethod
    def from_dense(
        cls,
        tensor,
        leg_charges,
        *,
        directions=(-1, 1, -1, 1, 1),
        total_charge=None,
        atol=0.0,
    ):
        """Split a dense tensor into charge blocks, optionally filtering flux."""
        tensor = np.asarray(tensor)
        if tensor.ndim != 5:
            raise ValueError("a PEPS tensor must have rank five.")
        if len(leg_charges) != 5:
            raise ValueError("leg_charges must contain one sequence per tensor leg.")
        grouped = []
        for axis, charges in enumerate(leg_charges):
            charges = tuple(_charge_tuple(q) for q in charges)
            if len(charges) != tensor.shape[axis]:
                raise ValueError(f"leg {axis} charge count does not match its dimension.")
            sectors = tuple(dict.fromkeys(charges))
            indices = {q: np.flatnonzero(np.array([item == q for item in charges])) for q in sectors}
            grouped.append((sectors, indices))
        target = None if total_charge is None else _charge_tuple(total_charge)
        blocks = {}
        for key in product(*[entry[0] for entry in grouped]):
            if target is not None:
                flux = tuple(
                    sum(directions[axis] * key[axis][component] for axis in range(5))
                    for component in range(len(key[0]))
                )
                if flux != target:
                    continue
            indices = [grouped[axis][1][key[axis]] for axis in range(5)]
            block = tensor[np.ix_(*indices)]
            if np.linalg.norm(block) > atol:
                blocks[key] = block
        sectors = tuple(entry[0] for entry in grouped)
        dims = tuple(
            {q: len(entry[1][q]) for q in entry[0]}
            for entry in grouped
        )
        return cls(blocks, sectors, dims, tuple(directions), target)


class PEPS:
    """Open-boundary rectangular PEPS with ``(l,r,u,d,p)`` tensors."""

    def __init__(self, tensors, *, sites=None, bond_singular_values=None):
        rows = tuple(tuple(row) for row in tensors)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("PEPS tensors must form a non-empty rectangle.")
        self.nrows = len(rows)
        self.ncols = len(rows[0])
        self.shape = (self.nrows, self.ncols)
        self.tensors = [[tensor for tensor in row] for row in rows]
        dims = []
        for row in rows:
            for tensor in row:
                shape = tensor.shape
                if len(shape) != 5 or any(int(size) < 1 for size in shape):
                    raise ValueError("every PEPS tensor must be a positive rank-five tensor.")
                dims.append(int(shape[4]))
        if sites is None:
            self.sites = tuple(Site.generic(dim) for dim in dims)
        else:
            self.sites = tuple(sites)
            if len(self.sites) != self.nrows * self.ncols:
                raise ValueError("sites must contain one entry per PEPS tensor.")
            if any(site.dim != dim for site, dim in zip(self.sites, dims)):
                raise ValueError("site dimensions must match PEPS physical dimensions.")
        self._check_bonds()
        self.bond_singular_values = {}
        supplied = {} if bond_singular_values is None else dict(bond_singular_values)
        for first, second, dimension in self.bonds():
            key = self.bond_key(first, second)
            values = np.asarray(supplied.get(key, np.ones(dimension)), dtype=float)
            if values.shape != (dimension,):
                raise ValueError(f"bond {key!r} requires {dimension} singular values.")
            self.bond_singular_values[key] = values.copy()

    def _check_bonds(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                shape = self.tensors[row][col].shape
                if col == 0 and shape[0] != 1 or col == self.ncols - 1 and shape[1] != 1:
                    raise ValueError("open PEPS left/right boundary bonds must have dimension one.")
                if row == 0 and shape[2] != 1 or row == self.nrows - 1 and shape[3] != 1:
                    raise ValueError("open PEPS up/down boundary bonds must have dimension one.")
                if col + 1 < self.ncols and shape[1] != self.tensors[row][col + 1].shape[0]:
                    raise ValueError(f"horizontal bond mismatch at {(row, col)}.")
                if row + 1 < self.nrows and shape[3] != self.tensors[row + 1][col].shape[2]:
                    raise ValueError(f"vertical bond mismatch at {(row, col)}.")

    def copy(self):
        return PEPS(
            [[self.dense_tensor((r, c)).copy() for c in range(self.ncols)] for r in range(self.nrows)],
            sites=self.sites,
            bond_singular_values=self.bond_singular_values,
        )

    def site(self, coordinate):
        row, col = self._coordinate(coordinate)
        return self.sites[row * self.ncols + col]

    def _coordinate(self, coordinate):
        row, col = map(int, coordinate)
        if not 0 <= row < self.nrows or not 0 <= col < self.ncols:
            raise IndexError(f"PEPS coordinate {(row, col)} is outside {self.shape}.")
        return row, col

    def dense_tensor(self, coordinate):
        row, col = self._coordinate(coordinate)
        tensor = self.tensors[row][col]
        return tensor.to_dense() if isinstance(tensor, AbelianPEPSTensor) else np.asarray(tensor)

    def network_tensor(self, coordinate):
        """Return a dense tensor with each left/up bond weight absorbed once.

        PEPS tensors are stored in the usual gamma/lambda simple-update form.
        Absorbing weights on the left and up legs gives an unambiguous dense
        network representation without counting any internal bond twice.
        """
        row, col = self._coordinate(coordinate)
        tensor = self.dense_tensor((row, col))
        if col:
            values = self.bond_singular_values[self.bond_key((row, col - 1), (row, col))]
            tensor = tensor * values.reshape(-1, 1, 1, 1, 1)
        if row:
            values = self.bond_singular_values[self.bond_key((row - 1, col), (row, col))]
            tensor = tensor * values.reshape(1, 1, -1, 1, 1)
        return tensor

    @staticmethod
    def bond_key(first, second):
        first, second = tuple(first), tuple(second)
        return tuple(sorted((first, second)))

    def bonds(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                if col + 1 < self.ncols:
                    yield (row, col), (row, col + 1), int(self.tensors[row][col].shape[1])
                if row + 1 < self.nrows:
                    yield (row, col), (row + 1, col), int(self.tensors[row][col].shape[3])

    @classmethod
    def product_state(cls, local_vectors, shape, *, sites=None, dtype=None):
        nrows, ncols = map(int, shape)
        vectors = tuple(np.asarray(vector, dtype=dtype) for vector in local_vectors)
        if len(vectors) != nrows * ncols or any(vector.ndim != 1 for vector in vectors):
            raise ValueError("local_vectors must contain one rank-one vector per lattice site.")
        tensors = []
        offset = 0
        for _row in range(nrows):
            row = []
            for _col in range(ncols):
                row.append(vectors[offset].reshape(1, 1, 1, 1, -1).copy())
                offset += 1
            tensors.append(row)
        return cls(tensors, sites=sites)

    def norm_squared(self, *, method="boundary", max_bond=None):
        from .contraction import overlap

        return overlap(self, self, method=method, max_bond=max_bond)

    def local_expectation(self, coordinate, operator, *, method="boundary", max_bond=None):
        from .contraction import local_expectation

        return local_expectation(self, coordinate, operator, method=method, max_bond=max_bond)


class AbelianPEPS(PEPS):
    """PEPS whose site tensors are explicitly stored in Abelian blocks."""

    def __init__(self, tensors, *, sites=None, bond_singular_values=None):
        if any(
            not isinstance(tensor, AbelianPEPSTensor)
            for row in tensors
            for tensor in row
        ):
            raise TypeError("AbelianPEPS requires AbelianPEPSTensor site tensors.")
        super().__init__(tensors, sites=sites, bond_singular_values=bond_singular_values)

    def to_dense(self):
        return PEPS(
            [[tensor.to_dense() for tensor in row] for row in self.tensors],
            sites=self.sites,
            bond_singular_values=self.bond_singular_values,
        )

    def copy(self):
        return type(self)(
            [[tensor.copy() for tensor in row] for row in self.tensors],
            sites=self.sites,
            bond_singular_values=self.bond_singular_values,
        )
