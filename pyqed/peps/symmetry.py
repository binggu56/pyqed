"""U(1)-blocked finite PEPS tensors and contractions."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from itertools import product
from numbers import Integral

import numpy as np
from opt_einsum import contract_expression, get_symbol

from pyqed.tn import Hamiltonian

from .contraction import _double_layer_kernel, exact_contract_layers, shared_executor
from .state import PEPS, _operator_product_factors, _rectangular_grid, _site_grid


def _charges_from_site(site):
    if site.charges is None:
        raise ValueError("U(1) PEPS requires charges on every physical Site.")
    charges = []
    for charge in site.charges:
        if len(charge) != 1:
            raise NotImplementedError("U1PEPS currently supports one charge component.")
        charges.append(int(charge[0]))
    return tuple(charges)


def _charge_groups(charges):
    groups = OrderedDict()
    for index, charge in enumerate(charges):
        groups.setdefault(int(charge), []).append(index)
    return OrderedDict((charge, tuple(indices)) for charge, indices in groups.items())


class U1PEPSTensor:
    r"""Rank-five U(1)-blocked PEPS tensor.

    Axes use ``(physical, up, right, down, left)`` order and default directions
    ``(+,+,+,-,-)``. A stored block obeys

    .. math::

        q_p + q_u + q_r - q_d - q_l = Q_A.
    """

    def __init__(
        self,
        blocks,
        axis_charges,
        *,
        target_charge=0,
        dirs=(1, 1, 1, -1, -1),
        copy=True,
    ):
        axis_charges = tuple(tuple(int(q) for q in axis) for axis in axis_charges)
        if len(axis_charges) != 5 or any(not axis for axis in axis_charges):
            raise ValueError("axis_charges must contain five nonempty charge tables.")
        dirs = tuple(int(value) for value in dirs)
        if len(dirs) != 5 or any(value not in {-1, 1} for value in dirs):
            raise ValueError("dirs must contain five entries equal to +1 or -1.")
        self.axis_charges = axis_charges
        self.qns = tuple(tuple(dict.fromkeys(axis)) for axis in axis_charges)
        self.groups = tuple(_charge_groups(axis) for axis in axis_charges)
        self.dirs = dirs
        self.target_charge = int(target_charge)
        self.data = OrderedDict()
        for raw_key, raw_block in dict(blocks).items():
            key = tuple(int(charge) for charge in raw_key)
            if len(key) != 5:
                raise ValueError("every U(1) PEPS block key must have rank five.")
            if any(charge not in self.groups[axis] for axis, charge in enumerate(key)):
                raise ValueError(f"block key {key} contains an unknown charge sector.")
            flux = sum(direction * charge for direction, charge in zip(self.dirs, key))
            if flux != self.target_charge:
                raise ValueError(
                    f"block {key} has flux {flux}, expected {self.target_charge}."
                )
            expected = tuple(len(self.groups[axis][charge]) for axis, charge in enumerate(key))
            block = np.asarray(raw_block)
            if block.shape != expected:
                raise ValueError(f"block {key} must have shape {expected}.")
            if np.any(~np.isfinite(block)):
                raise ValueError(f"block {key} contains nonfinite values.")
            self.data[key] = np.array(block, copy=bool(copy))

    @property
    def shape(self):
        return tuple(len(charges) for charges in self.axis_charges)

    @property
    def ndim(self):
        return 5

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def block_size(self):
        return sum(block.size for block in self.data.values())

    @property
    def storage_fraction(self):
        return self.block_size / self.size

    def copy(self):
        return type(self)(
            self.data,
            self.axis_charges,
            target_charge=self.target_charge,
            dirs=self.dirs,
            copy=True,
        )

    def scaled(self, scalar):
        return type(self)(
            {key: scalar * block for key, block in self.data.items()},
            self.axis_charges,
            target_charge=self.target_charge,
            dirs=self.dirs,
        )

    def to_dense(self):
        dtype = np.result_type(*[block.dtype for block in self.data.values()] or [float])
        dense = np.zeros(self.shape, dtype=dtype)
        for key, block in self.data.items():
            indices = tuple(self.groups[axis][charge] for axis, charge in enumerate(key))
            dense[np.ix_(*indices)] = block
        return dense

    @classmethod
    def from_dense(
        cls,
        tensor,
        axis_charges,
        *,
        target_charge=0,
        dirs=(1, 1, 1, -1, -1),
        atol=1.0e-13,
        project=False,
    ):
        tensor = np.asarray(tensor)
        axis_charges = tuple(tuple(int(q) for q in axis) for axis in axis_charges)
        expected = tuple(len(axis) for axis in axis_charges)
        if tensor.shape != expected:
            raise ValueError(f"dense tensor must have shape {expected}.")
        groups = tuple(_charge_groups(axis) for axis in axis_charges)
        blocks = OrderedDict()
        forbidden2 = 0.0
        for key in product(*[tuple(group) for group in groups]):
            indices = tuple(groups[axis][charge] for axis, charge in enumerate(key))
            block = tensor[np.ix_(*indices)]
            flux = sum(direction * charge for direction, charge in zip(dirs, key))
            if flux == int(target_charge):
                if np.linalg.norm(block) > float(atol):
                    blocks[key] = block
            else:
                forbidden2 += float(np.vdot(block.reshape(-1), block.reshape(-1)).real)
        forbidden_norm = float(np.sqrt(forbidden2))
        if forbidden_norm > float(atol) and not project:
            raise ValueError(
                "dense PEPS tensor contains symmetry-forbidden entries; "
                f"forbidden norm={forbidden_norm:.3e}."
            )
        result = cls(
            blocks,
            axis_charges,
            target_charge=target_charge,
            dirs=dirs,
            copy=True,
        )
        result.projection_error = forbidden_norm
        return result

    @classmethod
    def random(
        cls,
        axis_charges,
        *,
        target_charge=0,
        dirs=(1, 1, 1, -1, -1),
        rng=None,
        complex=False,
    ):
        rng = np.random.default_rng() if rng is None else rng
        groups = tuple(_charge_groups(axis) for axis in axis_charges)
        blocks = {}
        for key in product(*[tuple(group) for group in groups]):
            flux = sum(direction * charge for direction, charge in zip(dirs, key))
            if flux != int(target_charge):
                continue
            shape = tuple(len(groups[axis][charge]) for axis, charge in enumerate(key))
            block = rng.normal(size=shape)
            if complex:
                block = block + 1j * rng.normal(size=shape)
            blocks[key] = block / np.sqrt(max(np.prod(shape), 1))
        if not blocks:
            raise ValueError("the requested charges admit no nonzero PEPS blocks.")
        return cls(
            blocks,
            axis_charges,
            target_charge=target_charge,
            dirs=dirs,
        )


def _u1_double_layer(bra, ket, operator=None, *, atol=1.0e-14):
    if bra.shape[0] != ket.shape[0]:
        raise ValueError("bra and ket physical dimensions must match.")
    if operator is None:
        operator = np.eye(bra.shape[0], dtype=np.result_type(float))
    operator = np.asarray(operator)
    if operator.shape != (bra.shape[0], ket.shape[0]):
        raise ValueError("local operator has an incompatible dimension.")
    blocks = OrderedDict()
    for bra_key, bra_block in bra.data.items():
        bra_indices = bra.groups[0][bra_key[0]]
        for ket_key, ket_block in ket.data.items():
            ket_indices = ket.groups[0][ket_key[0]]
            operator_block = operator[np.ix_(bra_indices, ket_indices)]
            if np.linalg.norm(operator_block) <= float(atol):
                continue
            value = _double_layer_kernel(bra_block, ket_block, operator_block)
            key = tuple((bra_key[axis], ket_key[axis]) for axis in range(1, 5))
            old = blocks.get(key)
            blocks[key] = value if old is None else old + value
    return blocks


def _contract_u1_block_grid(block_grid, *, max_configurations=100000):
    block_grid = _rectangular_grid(block_grid, name="block layers")
    nrows, ncols = len(block_grid), len(block_grid[0])
    selected = [[None for _ in range(ncols)] for _ in range(nrows)]
    downward = {}
    rightward = {}
    value = 0.0j
    configurations = 0
    zero = (0, 0)

    def visit(position):
        nonlocal value, configurations
        if position == nrows * ncols:
            configurations += 1
            if configurations > int(max_configurations):
                raise RuntimeError(
                    "U(1) PEPS contraction exceeded max_configurations; "
                    "increase the guard or use a narrower sector layout."
                )
            value += exact_contract_layers(selected)
            return
        row, col = divmod(position, ncols)
        expected_up = zero if row == 0 else downward[(row - 1, col)]
        expected_left = zero if col == 0 else rightward[(row, col - 1)]
        for key, block in block_grid[row][col].items():
            if key[0] != expected_up or key[3] != expected_left:
                continue
            if row == nrows - 1 and key[2] != zero:
                continue
            if col == ncols - 1 and key[1] != zero:
                continue
            selected[row][col] = block
            downward[(row, col)] = key[2]
            rightward[(row, col)] = key[1]
            visit(position + 1)

    visit(0)
    return value, configurations


@lru_cache(maxsize=256)
def _block_row_expression(top_shape, block_shapes):
    ncols = len(block_shapes)
    up = tuple(range(ncols))
    down = tuple(range(ncols, 2 * ncols))
    horizontal = tuple(range(2 * ncols, 3 * ncols + 1))
    inputs = ["".join(get_symbol(label) for label in up)]
    for col in range(ncols):
        inputs.append(
            "".join(
                get_symbol(label)
                for label in (
                    up[col],
                    horizontal[col + 1],
                    down[col],
                    horizontal[col],
                )
            )
        )
    output = "".join(get_symbol(label) for label in down)
    return contract_expression(
        ",".join(inputs) + "->" + output,
        top_shape,
        *block_shapes,
        optimize="greedy",
    )


def _contract_u1_block_row(top, blocks):
    expression = _block_row_expression(
        tuple(top.shape),
        tuple(tuple(block.shape) for block in blocks),
    )
    return np.asarray(expression(top, *blocks))


def _prune_u1_frontier(frontier, *, max_frontiers, rtol, atol):
    if not frontier:
        return frontier, 0.0
    norms = {key: float(np.linalg.norm(value)) for key, value in frontier.items()}
    total2 = sum(value * value for value in norms.values())
    threshold = max(float(atol), float(rtol) * np.sqrt(total2))
    keep = [key for key, value in norms.items() if value > threshold]
    if not keep:
        keep = [max(norms, key=norms.get)]
    keep.sort(key=lambda key: norms[key], reverse=True)
    if max_frontiers is not None:
        keep = keep[: int(max_frontiers)]
    selected = set(keep)
    discarded = sum(
        value * value for key, value in norms.items() if key not in selected
    )
    return {key: frontier[key] for key in keep}, discarded


def _contract_u1_block_frontier(
    block_grid,
    *,
    max_frontiers=None,
    rtol=0.0,
    atol=0.0,
    frontier_guard=100000,
):
    """Contract charge blocks by merging equal vertical frontier sectors."""

    if max_frontiers is not None:
        if (
            isinstance(max_frontiers, bool)
            or not isinstance(max_frontiers, Integral)
            or int(max_frontiers) < 1
        ):
            raise ValueError("max_frontiers must be a positive integer or None.")
        max_frontiers = int(max_frontiers)
    if (
        isinstance(frontier_guard, bool)
        or not isinstance(frontier_guard, Integral)
        or int(frontier_guard) < 1
    ):
        raise ValueError("frontier_guard must be a positive integer.")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    block_grid = _rectangular_grid(block_grid, name="block layers")
    nrows, ncols = len(block_grid), len(block_grid[0])
    zero = (0, 0)
    frontier = {(zero,) * ncols: np.ones((1,) * ncols)}
    frontier_counts = [1]
    transitions = 0
    discarded_weight = 0.0

    indexed = []
    for row in range(nrows):
        indexed_row = []
        for col in range(ncols):
            choices = {}
            for key, block in block_grid[row][col].items():
                choices.setdefault((key[0], key[3]), []).append((key, block))
            indexed_row.append(choices)
        indexed.append(indexed_row)

    for row in range(nrows):
        following = {}
        for up_sectors, top in frontier.items():
            selected_keys = [None] * ncols
            selected_blocks = [None] * ncols

            def visit(col, left_sector):
                nonlocal transitions
                if col == ncols:
                    if left_sector != zero:
                        return
                    down_sectors = tuple(key[2] for key in selected_keys)
                    if row == nrows - 1 and any(
                        sector != zero for sector in down_sectors
                    ):
                        return
                    value = _contract_u1_block_row(top, selected_blocks)
                    old = following.get(down_sectors)
                    following[down_sectors] = value if old is None else old + value
                    transitions += 1
                    return
                for key, block in indexed[row][col].get(
                    (up_sectors[col], left_sector),
                    (),
                ):
                    if col == ncols - 1 and key[1] != zero:
                        continue
                    selected_keys[col] = key
                    selected_blocks[col] = block
                    visit(col + 1, key[1])

            visit(0, zero)
        if len(following) > int(frontier_guard):
            raise RuntimeError(
                "U(1) block frontier exceeded frontier_guard; set max_frontiers "
                "or increase the guard."
            )
        frontier, discarded = _prune_u1_frontier(
            following,
            max_frontiers=max_frontiers,
            rtol=rtol,
            atol=atol,
        )
        discarded_weight += discarded
        frontier_counts.append(len(frontier))

    final = frontier.get((zero,) * ncols)
    value = 0.0j if final is None else np.asarray(final).reshape(()).item()
    info = {
        "method": "u1-block-frontier",
        "frontier_counts": tuple(frontier_counts),
        "max_active_frontiers": max(frontier_counts),
        "transitions": transitions,
        "max_frontiers": max_frontiers,
        "discarded_weight": float(discarded_weight),
        "exact": max_frontiers is None and float(rtol) == 0.0 and float(atol) == 0.0,
    }
    return value, info


def _rotate_u1_block_grid(block_grid):
    nrows = len(block_grid)
    ncols = len(block_grid[0])
    rotated = [[None for _ in range(nrows)] for _ in range(ncols)]
    for row in range(nrows):
        for col in range(ncols):
            blocks = OrderedDict()
            for key, block in block_grid[row][col].items():
                blocks[(key[3], key[0], key[1], key[2])] = block.transpose(3, 0, 1, 2)
            rotated[col][nrows - 1 - row] = blocks
    return tuple(tuple(row) for row in rotated)


def _contract_u1_block_ctmrg(block_grid, **kwargs):
    directions = {"top": block_grid}
    directions["left"] = _rotate_u1_block_grid(directions["top"])
    directions["bottom"] = _rotate_u1_block_grid(directions["left"])
    directions["right"] = _rotate_u1_block_grid(directions["bottom"])
    values = {}
    infos = {}
    for name, grid in directions.items():
        values[name], infos[name] = _contract_u1_block_frontier(grid, **kwargs)
    value = sum(values.values()) / len(values)
    scale = max(1.0, abs(value))
    return value, {
        "method": "u1-block-ctmrg",
        "directional_values": values,
        "directional_spread": max(abs(item - value) for item in values.values()) / scale,
        "directions": infos,
        "discarded_weight": sum(item["discarded_weight"] for item in infos.values()),
        "exact": all(item["exact"] for item in infos.values()),
    }


class U1PEPS:
    """Finite PEPS stored entirely in U(1)-conserving tensor blocks."""

    def __init__(self, tensors, *, sites):
        tensor_grid = _rectangular_grid(tensors, name="tensors")
        site_grid = _site_grid(sites, (len(tensor_grid), len(tensor_grid[0])))
        if any(not isinstance(tensor, U1PEPSTensor) for row in tensor_grid for tensor in row):
            raise TypeError("U1PEPS tensors must be U1PEPSTensor objects.")
        self.nrows = len(tensor_grid)
        self.ncols = len(tensor_grid[0])
        self.shape = (self.nrows, self.ncols)
        self.nsites = self.nrows * self.ncols
        self.tensors = [[tensor.copy() for tensor in row] for row in tensor_grid]
        self.site_grid = site_grid
        self.sites = tuple(site for row in site_grid for site in row)
        self.dims = tuple(site.dim for site in self.sites)
        self.symmetry = "U1"
        self._validate()
        self.target_charge = sum(
            tensor.target_charge for row in self.tensors for tensor in row
        )
        self._layer_cache = {
            (row, col): {}
            for row in range(self.nrows)
            for col in range(self.ncols)
        }
        self._cache_hits = 0
        self._cache_misses = 0
        self._version = 0

    def _validate(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                tensor = self.tensors[row][col]
                site = self.site_grid[row][col]
                if tensor.axis_charges[0] != _charges_from_site(site):
                    raise ValueError(f"physical charges mismatch at {(row, col)}.")
                if row == 0 and tensor.axis_charges[1] != (0,):
                    raise ValueError("top U(1) PEPS boundaries must carry charge zero.")
                if col == self.ncols - 1 and tensor.axis_charges[2] != (0,):
                    raise ValueError("right U(1) PEPS boundaries must carry charge zero.")
                if row == self.nrows - 1 and tensor.axis_charges[3] != (0,):
                    raise ValueError("bottom U(1) PEPS boundaries must carry charge zero.")
                if col == 0 and tensor.axis_charges[4] != (0,):
                    raise ValueError("left U(1) PEPS boundaries must carry charge zero.")
                if col + 1 < self.ncols:
                    following = self.tensors[row][col + 1]
                    if tensor.axis_charges[2] != following.axis_charges[4]:
                        raise ValueError(f"horizontal charge mismatch after {(row, col)}.")
                if row + 1 < self.nrows:
                    following = self.tensors[row + 1][col]
                    if tensor.axis_charges[3] != following.axis_charges[1]:
                        raise ValueError(f"vertical charge mismatch below {(row, col)}.")

    @staticmethod
    def _edge_key(first, second):
        first, second = tuple(first), tuple(second)
        return (first, second) if first < second else (second, first)

    @classmethod
    def from_dense(
        cls,
        state,
        *,
        bond_charges=None,
        target_charges=None,
        project=False,
        atol=1.0e-13,
    ):
        if not isinstance(state, PEPS):
            raise TypeError("state must be a dense PEPS.")
        bond_charges = {
            cls._edge_key(*edge): tuple(int(q) for q in charges)
            for edge, charges in dict(bond_charges or {}).items()
        }
        if target_charges is None:
            target_charges = [0] * state.nsites
        target_charges = tuple(int(charge) for charge in target_charges)
        if len(target_charges) != state.nsites:
            raise ValueError("target_charges must contain one value per site.")

        tensors = []
        for row in range(state.nrows):
            tensor_row = []
            for col in range(state.ncols):
                coordinate = (row, col)
                dense = state.tensors[row][col]

                def charges_for(neighbor, axis):
                    if neighbor is None:
                        return (0,)
                    key = cls._edge_key(coordinate, neighbor)
                    charges = bond_charges.get(key)
                    if charges is None:
                        dimension = dense.shape[axis]
                        if dimension != 1:
                            raise ValueError(f"missing bond charges for edge {key}.")
                        return (0,)
                    if len(charges) != dense.shape[axis]:
                        raise ValueError(f"bond charges on edge {key} have the wrong length.")
                    return charges

                axis_charges = (
                    _charges_from_site(state.site_grid[row][col]),
                    charges_for((row - 1, col) if row else None, 1),
                    charges_for((row, col + 1) if col + 1 < state.ncols else None, 2),
                    charges_for((row + 1, col) if row + 1 < state.nrows else None, 3),
                    charges_for((row, col - 1) if col else None, 4),
                )
                tensor_row.append(
                    U1PEPSTensor.from_dense(
                        dense,
                        axis_charges,
                        target_charge=target_charges[state.site_index(coordinate)],
                        project=project,
                        atol=atol,
                    )
                )
            tensors.append(tensor_row)
        return cls(tensors, sites=state.sites)

    @classmethod
    def product_state(cls, sites, states, *, shape):
        dense = PEPS.product_state(sites, states, shape=shape)
        flat_states = tuple(state for row in _rectangular_grid(states, name="states") for state in row) if (
            len(tuple(states)) == shape[0] and all(hasattr(row, "__iter__") for row in states)
        ) else tuple(states)
        targets = []
        for site, state in zip(dense.sites, flat_states):
            if not isinstance(state, Integral):
                raise TypeError("U1PEPS.product_state currently requires basis indices.")
            targets.append(_charges_from_site(site)[int(state)])
        return cls.from_dense(dense, target_charges=targets)

    @classmethod
    def random(
        cls,
        sites,
        *,
        shape,
        bond_charges,
        target_charges=None,
        seed=None,
        complex=False,
    ):
        site_grid = _site_grid(sites, shape)
        nrows, ncols = shape
        bond_charges = {
            cls._edge_key(*edge): tuple(int(q) for q in charges)
            for edge, charges in dict(bond_charges).items()
        }
        if target_charges is None:
            target_charges = [0] * (nrows * ncols)
        rng = np.random.default_rng(seed)
        tensors = []
        for row in range(nrows):
            tensor_row = []
            for col in range(ncols):
                coordinate = (row, col)

                def edge(neighbor):
                    return (0,) if neighbor is None else bond_charges[cls._edge_key(coordinate, neighbor)]

                axis_charges = (
                    _charges_from_site(site_grid[row][col]),
                    edge((row - 1, col) if row else None),
                    edge((row, col + 1) if col + 1 < ncols else None),
                    edge((row + 1, col) if row + 1 < nrows else None),
                    edge((row, col - 1) if col else None),
                )
                tensor_row.append(
                    U1PEPSTensor.random(
                        axis_charges,
                        target_charge=target_charges[row * ncols + col],
                        rng=rng,
                        complex=complex,
                    )
                )
            tensors.append(tensor_row)
        state = cls(tensors, sites=sites)
        state.normalize()
        return state

    def coordinate(self, site):
        site = int(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site is out of range.")
        return divmod(site, self.ncols)

    def site_index(self, coordinate):
        row, col = (int(value) for value in coordinate)
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("coordinate is outside the U(1) PEPS grid.")
        return row * self.ncols + col

    @property
    def block_count(self):
        return sum(len(tensor.data) for row in self.tensors for tensor in row)

    @staticmethod
    def _operator_signature(operator):
        if operator is None:
            return None
        value = np.ascontiguousarray(operator)
        return value.dtype.str, value.shape, value.tobytes()

    def invalidate_cache(self, coordinates=None):
        if coordinates is None:
            coordinates = tuple(self._layer_cache)
        elif isinstance(coordinates, tuple) and len(coordinates) == 2 and all(
            isinstance(value, Integral) for value in coordinates
        ):
            coordinates = (coordinates,)
        for coordinate in coordinates:
            row, col = (int(value) for value in coordinate)
            self._layer_cache[(row, col)].clear()
        self._version += 1
        return self

    def _touch(self, *coordinates):
        return self.invalidate_cache(coordinates or None)

    def _cached_block_layer(self, other, row, col, operator=None):
        bra = self.tensors[row][col]
        ket = other.tensors[row][col]
        key = (id(bra), id(ket), self._operator_signature(operator))
        cache = self._layer_cache[(row, col)]
        layer = cache.get(key)
        if layer is not None:
            self._cache_hits += 1
            return layer
        self._cache_misses += 1
        layer = _u1_double_layer(bra, ket, operator)
        if len(cache) >= 16:
            cache.clear()
        cache[key] = layer
        return layer

    @property
    def bond_dims(self):
        return {
            "horizontal": tuple(
                self.tensors[row][col].shape[2]
                for row in range(self.nrows)
                for col in range(self.ncols - 1)
            ),
            "vertical": tuple(
                self.tensors[row][col].shape[3]
                for row in range(self.nrows - 1)
                for col in range(self.ncols)
            ),
        }

    @property
    def storage_fraction(self):
        stored = sum(tensor.block_size for row in self.tensors for tensor in row)
        dense = sum(tensor.size for row in self.tensors for tensor in row)
        return stored / dense

    def to_dense_peps(self):
        return PEPS(
            [[tensor.to_dense() for tensor in row] for row in self.tensors],
            sites=self.site_grid,
        )

    def to_dense(self):
        return self.to_dense_peps().to_dense()

    def _block_layers(self, other, operators=None):
        operators = {} if operators is None else operators
        return tuple(
            tuple(
                self._cached_block_layer(
                    other,
                    row,
                    col,
                    operators.get(self.site_index((row, col))),
                )
                for col in range(self.ncols)
            )
            for row in range(self.nrows)
        )

    def overlap(
        self,
        other=None,
        *,
        method="frontier",
        max_frontiers=None,
        rtol=0.0,
        atol=0.0,
        frontier_guard=100000,
        return_info=False,
        max_configurations=100000,
    ):
        other = self if other is None else other
        if (
            not isinstance(other, U1PEPS)
            or other.shape != self.shape
            or other.dims != self.dims
        ):
            raise ValueError("U(1) PEPS shapes must match.")
        method_key = str(method).lower().replace("_", "-")
        layers = self._block_layers(other)
        if method_key in {"frontier", "boundary", "block-frontier"}:
            value, info = _contract_u1_block_frontier(
                layers,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
                frontier_guard=frontier_guard,
            )
        elif method_key in {"ctmrg", "ctm", "corner"}:
            value, info = _contract_u1_block_ctmrg(
                layers,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
                frontier_guard=frontier_guard,
            )
        elif method_key in {"enumerate", "reference", "dfs"}:
            value, configurations = _contract_u1_block_grid(
                layers,
                max_configurations=max_configurations,
            )
            info = {
                "method": "u1-block-enumerate",
                "configurations": configurations,
                "exact": True,
            }
        else:
            raise ValueError("method must be 'frontier', 'ctmrg', or 'enumerate'.")
        info = dict(info)
        info["block_count"] = self.block_count
        return (value, info) if return_info else value

    def norm_squared(self, **kwargs):
        return self.overlap(self, **kwargs)

    def normalize(self):
        norm2 = np.real_if_close(self.norm_squared())
        if abs(np.imag(norm2)) > 1.0e-11 * max(1.0, abs(norm2)):
            raise FloatingPointError("U(1) PEPS norm is significantly complex.")
        norm2 = float(np.real(norm2))
        if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a zero U(1) PEPS.")
        first = self.tensors[0][0]
        self.tensors[0][0] = U1PEPSTensor(
            {key: block / np.sqrt(norm2) for key, block in first.data.items()},
            first.axis_charges,
            target_charge=first.target_charge,
            dirs=first.dirs,
        )
        self._touch((0, 0))
        return self

    def local_expectation(
        self,
        operators,
        *,
        normalize=True,
        method="frontier",
        max_frontiers=None,
        rtol=0.0,
        atol=0.0,
        return_info=False,
    ):
        normalized = {}
        for key, operator in dict(operators).items():
            site = self.site_index(key) if isinstance(key, tuple) else int(key)
            if site < 0 or site >= self.nsites:
                raise IndexError("operator site is out of range.")
            operator = np.asarray(operator)
            expected = (self.dims[site], self.dims[site])
            if operator.shape != expected:
                raise ValueError(f"operator on site {site} must have shape {expected}.")
            normalized[site] = operator
        layers = self._block_layers(self, normalized)
        method_key = str(method).lower().replace("_", "-")
        if method_key in {"frontier", "boundary", "block-frontier"}:
            numerator, info = _contract_u1_block_frontier(
                layers,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
            )
        elif method_key in {"ctmrg", "ctm", "corner"}:
            numerator, info = _contract_u1_block_ctmrg(
                layers,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
            )
        elif method_key in {"enumerate", "reference", "dfs"}:
            numerator, configurations = _contract_u1_block_grid(layers)
            info = {
                "method": "u1-block-enumerate",
                "configurations": configurations,
                "exact": True,
            }
        else:
            raise ValueError("method must be 'frontier', 'ctmrg', or 'enumerate'.")
        if normalize:
            numerator = numerator / self.norm_squared(
                method=method,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
            )
        value = np.real_if_close(numerator)
        return (value, info) if return_info else value

    def expectation(
        self,
        hamiltonian,
        *,
        method="frontier",
        max_frontiers=None,
        rtol=0.0,
        atol=0.0,
        workers=1,
        return_info=False,
    ):
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
        if hamiltonian.dims != self.dims:
            raise ValueError("Hamiltonian physical dimensions do not match the PEPS.")
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        workers = int(workers)
        cache_hits_before = self._cache_hits
        cache_misses_before = self._cache_misses
        norm = self.norm_squared(
            method=method,
            max_frontiers=max_frontiers,
            rtol=rtol,
            atol=atol,
        )
        numerator = hamiltonian.constant * norm
        factor_jobs = []
        for term in hamiltonian.terms:
            factors_list = _operator_product_factors(
                term.operator,
                tuple(self.dims[site] for site in term.sites),
            )
            for factors in factors_list:
                factor_jobs.append(dict(zip(term.sites, factors)))

        def contract_factor(operators):
            return self.local_expectation(
                operators,
                normalize=False,
                method=method,
                max_frontiers=max_frontiers,
                rtol=rtol,
                atol=atol,
            )

        if workers == 1 or len(factor_jobs) < 2:
            values = tuple(map(contract_factor, factor_jobs))
        else:
            values = tuple(
                shared_executor(workers).map(contract_factor, factor_jobs)
            )
        for value in values:
            numerator += value
        energy = float(np.real(np.real_if_close(numerator / norm)))
        info = {
            "method": f"u1-block-{str(method).lower().replace('_', '-')}",
            "contractions": len(values) + 1,
            "workers": workers,
            "storage_fraction": self.storage_fraction,
            "layer_cache_hits": self._cache_hits - cache_hits_before,
            "layer_cache_misses": self._cache_misses - cache_misses_before,
        }
        return (energy, info) if return_info else energy

    def evolve(self, hamiltonian, target, **kwargs):
        """Evolve this U(1)-blocked PEPS with sector-resolved gate splits."""

        from .evolution import PEPSEvolution

        step = kwargs.pop("step", 0.05)
        verbose = kwargs.pop("verbose", False)
        return PEPSEvolution(self, hamiltonian, **kwargs).run(
            target,
            step=step,
            verbose=verbose,
        )


def _flat_sector_values(axis_charges, formula):
    shape = tuple(len(axis) for axis in axis_charges)
    values = np.empty(int(np.prod(shape)), dtype=int)
    for flat, indices in enumerate(np.ndindex(shape)):
        values[flat] = int(formula(*(axis_charges[axis][index] for axis, index in enumerate(indices))))
    return values


def _sector_svd(matrix, row_sectors, col_sectors, *, max_D, cutoff):
    decompositions = {}
    forbidden2 = 0.0
    total2 = float(np.vdot(matrix.reshape(-1), matrix.reshape(-1)).real)
    for row_charge in set(row_sectors):
        rows = np.flatnonzero(row_sectors == row_charge)
        for col_charge in set(col_sectors):
            cols = np.flatnonzero(col_sectors == col_charge)
            block = matrix[np.ix_(rows, cols)]
            if row_charge != col_charge:
                forbidden2 += float(np.vdot(block.reshape(-1), block.reshape(-1)).real)
                continue
            if block.size == 0:
                continue
            decompositions[row_charge] = (rows, cols, *np.linalg.svd(block, full_matrices=False))
    scale = max(np.sqrt(total2), 1.0)
    if np.sqrt(forbidden2) > 1.0e-11 * scale:
        raise ValueError("two-site gate violates the U(1) charge selection rule.")
    candidates = []
    largest = 0.0
    for charge, (_rows, _cols, _u, singular_values, _vh) in decompositions.items():
        if singular_values.size:
            largest = max(largest, float(singular_values[0]))
        candidates.extend(
            (float(value), charge, index)
            for index, value in enumerate(singular_values)
        )
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = [item for item in candidates if item[0] > float(cutoff) * largest][
        : int(max_D)
    ]
    if not selected:
        raise ValueError("U(1) gate truncation removed every bond sector.")
    left = np.zeros((matrix.shape[0], len(selected)), dtype=matrix.dtype)
    right = np.zeros((len(selected), matrix.shape[1]), dtype=matrix.dtype)
    bond_charges = []
    kept2 = 0.0
    for bond, (value, charge, index) in enumerate(selected):
        rows, cols, u, singular_values, vh = decompositions[charge]
        root = np.sqrt(singular_values[index])
        left[rows, bond] = u[:, index] * root
        right[bond, cols] = root * vh[index]
        bond_charges.append(int(charge))
        kept2 += value * value
    discarded = max(total2 - kept2, 0.0)
    return left, right, tuple(bond_charges), {
        "kept_rank": len(selected),
        "sector_ranks": {
            charge: sum(bond_charge == charge for bond_charge in bond_charges)
            for charge in sorted(set(bond_charges))
        },
        "discarded_weight": float(discarded),
        "relative_error": float(np.sqrt(discarded / total2)) if total2 > 0.0 else 0.0,
        "forbidden_weight": float(forbidden2),
    }


def apply_u1_peps_local_gate(state, coordinate, gate):
    """Apply a charge-conserving local gate to one U(1)-blocked PEPS tensor."""

    row, col = coordinate
    tensor = state.tensors[row][col]
    dense = np.tensordot(np.asarray(gate), tensor.to_dense(), axes=(1, 0))
    state.tensors[row][col] = U1PEPSTensor.from_dense(
        dense,
        tensor.axis_charges,
        target_charge=tensor.target_charge,
        dirs=tensor.dirs,
        project=False,
    )
    state._touch((row, col))


def apply_u1_peps_pair_gate(state, first, second, gate, *, max_D, cutoff=1.0e-12):
    """Apply a two-site gate and perform a sector-resolved PEPS SVD split."""

    first = tuple(first)
    second = tuple(second)
    a = state.tensors[first[0]][first[1]]
    b = state.tensors[second[0]][second[1]]
    if first[0] == second[0] and second[1] == first[1] + 1:
        orientation = "horizontal"
        theta = np.tensordot(a.to_dense(), b.to_dense(), axes=(2, 4))
        left_dims = (a.shape[0], a.shape[1], a.shape[3], a.shape[4])
        right_dims = (b.shape[0], b.shape[1], b.shape[2], b.shape[3])
        left_charges = (
            a.axis_charges[0],
            a.axis_charges[1],
            a.axis_charges[3],
            a.axis_charges[4],
        )
        right_charges = (
            b.axis_charges[0],
            b.axis_charges[1],
            b.axis_charges[2],
            b.axis_charges[3],
        )
        row_sectors = _flat_sector_values(
            left_charges,
            lambda qp, qu, qd, ql: a.target_charge - qp - qu + qd + ql,
        )
        col_sectors = _flat_sector_values(
            right_charges,
            lambda qp, qu, qr, qd: qp + qu + qr - qd - b.target_charge,
        )
    elif first[1] == second[1] and second[0] == first[0] + 1:
        orientation = "vertical"
        theta = np.tensordot(a.to_dense(), b.to_dense(), axes=(3, 1))
        left_dims = (a.shape[0], a.shape[1], a.shape[2], a.shape[4])
        right_dims = (b.shape[0], b.shape[2], b.shape[3], b.shape[4])
        left_charges = (
            a.axis_charges[0],
            a.axis_charges[1],
            a.axis_charges[2],
            a.axis_charges[4],
        )
        right_charges = (
            b.axis_charges[0],
            b.axis_charges[2],
            b.axis_charges[3],
            b.axis_charges[4],
        )
        row_sectors = _flat_sector_values(
            left_charges,
            lambda qp, qu, qr, ql: qp + qu + qr - ql - a.target_charge,
        )
        col_sectors = _flat_sector_values(
            right_charges,
            lambda qp, qr, qd, ql: b.target_charge - qp - qr + qd + ql,
        )
    else:
        raise ValueError("U(1) PEPS gates require rightward or downward neighbors.")

    first_dim, second_dim = left_dims[0], right_dims[0]
    gate = np.asarray(gate).reshape(first_dim, second_dim, first_dim, second_dim)
    theta = np.tensordot(gate, theta, axes=((2, 3), (0, 4))).transpose(
        0, 2, 3, 4, 1, 5, 6, 7
    )
    matrix = theta.reshape(int(np.prod(left_dims)), int(np.prod(right_dims)))
    left, right, bond_charges, info = _sector_svd(
        matrix,
        row_sectors,
        col_sectors,
        max_D=max_D,
        cutoff=cutoff,
    )
    rank = len(bond_charges)
    left = left.reshape(left_dims + (rank,))
    right = right.reshape((rank,) + right_dims)
    if orientation == "horizontal":
        dense_a = left.transpose(0, 1, 4, 2, 3)
        dense_b = right.transpose(1, 2, 3, 4, 0)
        charges_a = (
            a.axis_charges[0],
            a.axis_charges[1],
            bond_charges,
            a.axis_charges[3],
            a.axis_charges[4],
        )
        charges_b = (
            b.axis_charges[0],
            b.axis_charges[1],
            b.axis_charges[2],
            b.axis_charges[3],
            bond_charges,
        )
    else:
        dense_a = left.transpose(0, 1, 2, 4, 3)
        dense_b = right.transpose(1, 0, 2, 3, 4)
        charges_a = (
            a.axis_charges[0],
            a.axis_charges[1],
            a.axis_charges[2],
            bond_charges,
            a.axis_charges[4],
        )
        charges_b = (
            b.axis_charges[0],
            bond_charges,
            b.axis_charges[2],
            b.axis_charges[3],
            b.axis_charges[4],
        )
    state.tensors[first[0]][first[1]] = U1PEPSTensor.from_dense(
        dense_a,
        charges_a,
        target_charge=a.target_charge,
    )
    state.tensors[second[0]][second[1]] = U1PEPSTensor.from_dense(
        dense_b,
        charges_b,
        target_charge=b.target_charge,
    )
    state._touch(first, second)
    info.update(
        {
            "coordinates": (first, second),
            "orientation": orientation,
            "backend": "u1-block-svd",
        }
    )
    return info


__all__ = [
    "U1PEPS",
    "U1PEPSTensor",
    "apply_u1_peps_local_gate",
    "apply_u1_peps_pair_gate",
]
