"""Contraction backends for finite open-boundary PEPS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from threading import Lock

import numpy as np
from opt_einsum import contract, contract_expression, get_symbol


_EXECUTORS = {}
_EXECUTOR_LOCK = Lock()


def shared_executor(workers):
    """Return a process-wide persistent contraction thread pool."""

    workers = int(workers)
    if workers < 2:
        return None
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"peps-{workers}",
            )
            _EXECUTORS[workers] = executor
    return executor


def _double_layer_kernel(bra, ket, operator):
    transformed = np.tensordot(operator, ket, axes=(1, 0))
    value = np.tensordot(bra.conj(), transformed, axes=(0, 0))
    return value.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        bra.shape[1] * ket.shape[1],
        bra.shape[2] * ket.shape[2],
        bra.shape[3] * ket.shape[3],
        bra.shape[4] * ket.shape[4],
    )


def double_layer_tensor(bra, ket, operator=None):
    r"""Return one PEPS double tensor in ``(up, right, down, left)`` order.

    ``bra`` and ``ket`` use ``(physical, up, right, down, left)`` ordering.
    When supplied, ``operator`` is inserted as
    :math:`\langle\mathrm{bra}|O|\mathrm{ket}\rangle`.
    """

    bra = np.asarray(bra)
    ket = np.asarray(ket)
    if bra.ndim != 5 or ket.ndim != 5:
        raise ValueError("PEPS tensors must have rank five.")
    if bra.shape[0] != ket.shape[0]:
        raise ValueError("bra and ket physical dimensions must match.")
    if operator is None:
        operator = np.eye(
            bra.shape[0],
            dtype=np.result_type(bra.dtype, ket.dtype),
        )
    operator = np.asarray(operator)
    expected = (bra.shape[0], ket.shape[0])
    if operator.shape != expected:
        raise ValueError(f"local operator must have shape {expected}.")
    return _double_layer_kernel(bra, ket, operator)


@lru_cache(maxsize=64)
def _exact_expression(shape_grid):
    nrows = len(shape_grid)
    ncols = len(shape_grid[0])
    next_label = 0
    horizontal = {}
    vertical = {}
    inputs = []
    shapes = []
    for row in range(nrows):
        for col in range(ncols):
            if row == 0:
                up = next_label
                next_label += 1
            else:
                up = vertical[(row - 1, col)]
            if col == 0:
                left = next_label
                next_label += 1
            else:
                left = horizontal[(row, col - 1)]
            right = next_label
            down = next_label + 1
            next_label += 2
            horizontal[(row, col)] = right
            vertical[(row, col)] = down
            inputs.append("".join(get_symbol(label) for label in (up, right, down, left)))
            shapes.append(shape_grid[row][col])
    return contract_expression(
        ",".join(inputs) + "->",
        *shapes,
        optimize="auto-hq",
    )


def exact_contract_layers(layers, *, optimize="auto-hq"):
    """Exactly contract a rectangular grid of rank-four layer tensors."""

    layers = tuple(tuple(np.asarray(tensor) for tensor in row) for row in layers)
    if not layers or not layers[0]:
        raise ValueError("layers must form a nonempty rectangular grid.")
    ncols = len(layers[0])
    if any(len(row) != ncols for row in layers):
        raise ValueError("layers must form a rectangular grid.")
    if any(tensor.ndim != 4 for row in layers for tensor in row):
        raise ValueError("every layer tensor must have rank four.")

    if optimize == "auto-hq":
        shape_grid = tuple(tuple(tensor.shape for tensor in row) for row in layers)
        expression = _exact_expression(shape_grid)
        return expression(*(tensor for row in layers for tensor in row))

    next_label = 0
    horizontal = {}
    vertical = {}
    operands = []
    nrows = len(layers)
    for row in range(nrows):
        for col in range(ncols):
            tensor = layers[row][col]
            if row == 0:
                up = next_label
                next_label += 1
            else:
                up = vertical[(row - 1, col)]
            if col == 0:
                left = next_label
                next_label += 1
            else:
                left = horizontal[(row, col - 1)]
            right = next_label
            next_label += 1
            down = next_label
            next_label += 1
            horizontal[(row, col)] = right
            vertical[(row, col)] = down
            operands.extend((tensor, [up, right, down, left]))
    operands.append([])
    return contract(*operands, optimize=optimize)


def exact_contract_layers_with_hole(layers, coordinate, *, optimize="auto-hq"):
    """Contract a layer network while leaving one tensor's legs open."""

    layers = tuple(
        tuple(None if tensor is None else np.asarray(tensor) for tensor in row)
        for row in layers
    )
    if not layers or not layers[0]:
        raise ValueError("layers must form a nonempty rectangular grid.")
    nrows = len(layers)
    ncols = len(layers[0])
    if any(len(row) != ncols for row in layers):
        raise ValueError("layers must form a rectangular grid.")
    hole_row, hole_col = (int(value) for value in coordinate)
    if not (0 <= hole_row < nrows and 0 <= hole_col < ncols):
        raise IndexError("hole coordinate is outside the layer grid.")
    if layers[hole_row][hole_col] is not None:
        raise ValueError("the selected hole tensor must be None.")

    next_label = 0
    horizontal = {}
    vertical = {}
    labels = {}
    operands = []
    for row in range(nrows):
        for col in range(ncols):
            up = vertical[(row - 1, col)] if row else next_label
            if row == 0:
                next_label += 1
            left = horizontal[(row, col - 1)] if col else next_label
            if col == 0:
                next_label += 1
            right = next_label
            down = next_label + 1
            next_label += 2
            horizontal[(row, col)] = right
            vertical[(row, col)] = down
            tensor_labels = [up, right, down, left]
            labels[(row, col)] = tensor_labels
            tensor = layers[row][col]
            if tensor is not None:
                if tensor.ndim != 4:
                    raise ValueError("every non-hole layer tensor must have rank four.")
                operands.extend((tensor, tensor_labels))

    output_labels = labels[(hole_row, hole_col)]
    hole_dims = []
    neighbors = (
        (hole_row - 1, hole_col, 2),
        (hole_row, hole_col + 1, 3),
        (hole_row + 1, hole_col, 0),
        (hole_row, hole_col - 1, 1),
    )
    for neighbor_row, neighbor_col, neighbor_axis in neighbors:
        if 0 <= neighbor_row < nrows and 0 <= neighbor_col < ncols:
            neighbor = layers[neighbor_row][neighbor_col]
            if neighbor is None:
                raise ValueError("only one hole is supported.")
            hole_dims.append(neighbor.shape[neighbor_axis])
        else:
            hole_dims.append(1)
    present_labels = {
        label
        for operand in operands[1::2]
        for label in operand
    }
    for label, dim in zip(output_labels, hole_dims):
        if label not in present_labels:
            operands.extend((np.ones(dim), [label]))
    operands.append(output_labels)
    return np.asarray(contract(*operands, optimize=optimize))


def _truncation_rank(singular_values, max_bond, rtol, atol):
    singular_values = np.asarray(singular_values)
    if singular_values.size == 0:
        return 1
    limit = singular_values.size if max_bond is None else min(
        singular_values.size,
        max_bond,
    )
    tolerance = max(float(atol), float(rtol) * float(np.linalg.norm(singular_values)))
    rank = singular_values.size
    if tolerance > 0.0:
        discarded2 = np.cumsum(np.abs(singular_values[::-1]) ** 2)[::-1]
        for candidate in range(1, singular_values.size + 1):
            tail = 0.0 if candidate == singular_values.size else discarded2[candidate]
            if np.sqrt(tail) <= tolerance:
                rank = candidate
                break
    return max(1, min(rank, limit))


def compress_boundary_mps(tensors, *, max_bond=None, rtol=0.0, atol=0.0):
    """Left-canonically compress boundary-MPS tensors by SVD."""

    tensors = [np.asarray(tensor).copy() for tensor in tensors]
    discarded2 = 0.0
    total2 = 0.0
    ranks = []
    spectra = []
    for site in range(len(tensors) - 1):
        left, physical, right = tensors[site].shape
        matrix = tensors[site].reshape(left * physical, right)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = _truncation_rank(singular_values, max_bond, rtol, atol)
        discarded2 += float(np.sum(np.abs(singular_values[rank:]) ** 2))
        total2 += float(np.sum(np.abs(singular_values) ** 2))
        tensors[site] = u[:, :rank].reshape(left, physical, rank)
        transfer = singular_values[:rank, None] * vh[:rank]
        following = tensors[site + 1]
        tensors[site + 1] = (transfer @ following.reshape(following.shape[0], -1)).reshape(
            transfer.shape[0],
            following.shape[1],
            following.shape[2],
        )
        ranks.append(rank)
        spectra.append(np.array(singular_values[:rank], copy=True))
    relative_error = np.sqrt(discarded2 / total2) if total2 > 0.0 else 0.0
    return tensors, {
        "bond_dims": tuple(ranks),
        "discarded_weight": discarded2,
        "relative_error": float(relative_error),
        "schmidt_values": tuple(spectra),
    }


def compress_boundary_mps_batch(
    batch,
    *,
    max_bond=None,
    rtol=0.0,
    atol=0.0,
):
    """Compress equally long boundary MPSs with shape-grouped batched SVDs."""

    states = [[np.asarray(tensor).copy() for tensor in tensors] for tensors in batch]
    if not states:
        return [], []
    nsites = len(states[0])
    if any(len(tensors) != nsites for tensors in states):
        raise ValueError("all batched boundary MPSs must have equal lengths.")
    diagnostics = [
        {"discarded2": 0.0, "total2": 0.0, "ranks": [], "spectra": []}
        for _ in states
    ]
    for site in range(nsites - 1):
        groups = defaultdict(list)
        for job, tensors in enumerate(states):
            groups[(tensors[site].shape, tensors[site + 1].shape)].append(job)
        for jobs in groups.values():
            matrices = np.stack(
                [
                    states[job][site].reshape(
                        states[job][site].shape[0] * states[job][site].shape[1],
                        states[job][site].shape[2],
                    )
                    for job in jobs
                ]
            )
            u, singular_values, vh = np.linalg.svd(matrices, full_matrices=False)
            ranks = [
                _truncation_rank(values, max_bond, rtol, atol)
                for values in singular_values
            ]
            rank_groups = defaultdict(list)
            for local, rank in enumerate(ranks):
                rank_groups[rank].append(local)
            for rank, locals_ in rank_groups.items():
                selected_u = u[locals_, :, :rank]
                selected_s = singular_values[locals_, :rank]
                selected_vh = vh[locals_, :rank, :]
                following = np.stack(
                    [states[jobs[local]][site + 1] for local in locals_]
                )
                transfer = selected_s[:, :, None] * selected_vh
                updated = transfer @ following.reshape(
                    following.shape[0],
                    following.shape[1],
                    -1,
                )
                for offset, local in enumerate(locals_):
                    job = jobs[local]
                    left, physical, _ = states[job][site].shape
                    states[job][site] = selected_u[offset].reshape(
                        left,
                        physical,
                        rank,
                    )
                    states[job][site + 1] = updated[offset].reshape(
                        rank,
                        following.shape[2],
                        following.shape[3],
                    )
            for local, job in enumerate(jobs):
                rank = ranks[local]
                values = singular_values[local]
                diagnostics[job]["discarded2"] += float(
                    np.sum(np.abs(values[rank:]) ** 2)
                )
                diagnostics[job]["total2"] += float(np.sum(np.abs(values) ** 2))
                diagnostics[job]["ranks"].append(rank)
                diagnostics[job]["spectra"].append(np.array(values[:rank], copy=True))

    infos = []
    for item in diagnostics:
        relative_error = (
            np.sqrt(item["discarded2"] / item["total2"])
            if item["total2"] > 0.0
            else 0.0
        )
        infos.append(
            {
                "bond_dims": tuple(item["ranks"]),
                "discarded_weight": item["discarded2"],
                "relative_error": float(relative_error),
                "schmidt_values": tuple(item["spectra"]),
                "batched": True,
            }
        )
    return states, infos


def _validated_layers(layers):
    layers = tuple(tuple(np.asarray(tensor) for tensor in row) for row in layers)
    if not layers or not layers[0]:
        raise ValueError("layers must form a nonempty rectangular grid.")
    ncols = len(layers[0])
    if any(len(row) != ncols for row in layers):
        raise ValueError("layers must form a rectangular grid.")
    if any(tensor.ndim != 4 for row in layers for tensor in row):
        raise ValueError("every layer tensor must have rank four.")
    return layers


def _rotate_layers_clockwise(layers):
    nrows = len(layers)
    ncols = len(layers[0])
    rotated = [[None for _ in range(nrows)] for _ in range(ncols)]
    for row in range(nrows):
        for col in range(ncols):
            rotated[col][nrows - 1 - row] = np.asarray(layers[row][col]).transpose(
                3, 0, 1, 2
            )
    return tuple(tuple(row) for row in rotated)


def _top_boundary(layers):
    return [
        np.ones((1, layers[0][col].shape[0], 1), dtype=layers[0][col].dtype)
        for col in range(len(layers[0]))
    ]


def _bottom_boundary(layers):
    return [
        np.ones((1, layers[-1][col].shape[2], 1), dtype=layers[-1][col].dtype)
        for col in range(len(layers[0]))
    ]


def _absorb_top_row(boundary, layer_row, *, max_bond, rtol, atol):
    updated = []
    for col, (boundary_tensor, layer) in enumerate(zip(boundary, layer_row)):
        if boundary_tensor.shape[1] != layer.shape[0]:
            raise ValueError(f"boundary dimension mismatch at column {col}.")
        value = np.tensordot(boundary_tensor, layer, axes=(1, 0)).transpose(
            0, 4, 3, 1, 2
        )
        updated.append(
            value.reshape(
                boundary_tensor.shape[0] * layer.shape[3],
                layer.shape[2],
                boundary_tensor.shape[2] * layer.shape[1],
            )
        )
    return compress_boundary_mps(
        updated,
        max_bond=max_bond,
        rtol=rtol,
        atol=atol,
    )


def _absorb_bottom_row(boundary, layer_row, *, max_bond, rtol, atol):
    updated = []
    for col, (boundary_tensor, layer) in enumerate(zip(boundary, layer_row)):
        if boundary_tensor.shape[1] != layer.shape[2]:
            raise ValueError(f"boundary dimension mismatch at column {col}.")
        value = np.tensordot(boundary_tensor, layer, axes=(1, 2)).transpose(
            0, 4, 2, 1, 3
        )
        updated.append(
            value.reshape(
                boundary_tensor.shape[0] * layer.shape[3],
                layer.shape[0],
                boundary_tensor.shape[2] * layer.shape[1],
            )
        )
    return compress_boundary_mps(
        updated,
        max_bond=max_bond,
        rtol=rtol,
        atol=atol,
    )


def _absorb_top_rows_batch(boundaries, layer_rows, *, max_bond, rtol, atol):
    updated_batch = []
    for boundary, layer_row in zip(boundaries, layer_rows):
        updated = []
        for col, (boundary_tensor, layer) in enumerate(zip(boundary, layer_row)):
            if boundary_tensor.shape[1] != layer.shape[0]:
                raise ValueError(f"boundary dimension mismatch at column {col}.")
            value = np.tensordot(boundary_tensor, layer, axes=(1, 0)).transpose(
                0, 4, 3, 1, 2
            )
            updated.append(
                value.reshape(
                    boundary_tensor.shape[0] * layer.shape[3],
                    layer.shape[2],
                    boundary_tensor.shape[2] * layer.shape[1],
                )
            )
        updated_batch.append(updated)
    return compress_boundary_mps_batch(
        updated_batch,
        max_bond=max_bond,
        rtol=rtol,
        atol=atol,
    )


def _contract_boundaries(top, bottom):
    if len(top) != len(bottom):
        raise ValueError("top and bottom boundaries must have equal lengths.")
    dtype = np.result_type(
        *[tensor.dtype for tensor in top],
        *[tensor.dtype for tensor in bottom],
    )
    value = np.ones((1, 1), dtype=dtype)
    for col, (top_tensor, bottom_tensor) in enumerate(zip(top, bottom)):
        if top_tensor.shape[1] != bottom_tensor.shape[1]:
            raise ValueError(f"boundary physical dimensions differ at column {col}.")
        transfer = np.tensordot(top_tensor, bottom_tensor, axes=(1, 1)).transpose(
            0, 2, 1, 3
        ).reshape(
            top_tensor.shape[0] * bottom_tensor.shape[0],
            top_tensor.shape[2] * bottom_tensor.shape[2],
        )
        if value.shape[1] != transfer.shape[0]:
            raise ValueError(f"boundary bond dimensions differ at column {col}.")
        value = value @ transfer
    if value.size != 1:
        raise ValueError("open PEPS boundary did not contract to a scalar.")
    return value.reshape(()).item()


class BoundaryMPSEnvironment:
    """Reusable two-sided boundary environment for local observables.

    The identity double layer is absorbed once from both open boundaries.
    Observable contractions then rebuild only the rows intersecting the local
    operator support and join them to the cached prefix and suffix boundaries.
    """

    def __init__(self, contractor, layers):
        self.contractor = contractor
        self.original_layers = _validated_layers(layers)
        self.original_shape = (len(self.original_layers), len(self.original_layers[0]))
        direction = contractor.direction
        if direction == "auto":
            direction = (
                "rows"
                if self.original_shape[1] <= self.original_shape[0]
                else "columns"
            )
        self.direction = direction
        self.layers = (
            self.original_layers
            if direction == "rows"
            else _rotate_layers_clockwise(self.original_layers)
        )
        self.nrows = len(self.layers)
        self.ncols = len(self.layers[0])

        top = _top_boundary(self.layers)
        self._prefixes = [top]
        self._prefix_infos = []
        for layer_row in self.layers:
            top, info = _absorb_top_row(
                top,
                layer_row,
                max_bond=contractor.max_bond,
                rtol=contractor.rtol,
                atol=contractor.atol,
            )
            self._prefixes.append(top)
            self._prefix_infos.append(info)

        self.value = _contract_boundaries(
            self._prefixes[self.nrows],
            _bottom_boundary(self.layers),
        )
        self._suffixes = None
        self._suffix_infos = None
        self.info = self._base_info()

    def _build_suffixes(self):
        if self._suffixes is not None:
            return
        bottom = _bottom_boundary(self.layers)
        self._suffixes = [None] * (self.nrows + 1)
        self._suffixes[self.nrows] = bottom
        self._suffix_infos = [None] * self.nrows
        for row in range(self.nrows - 1, -1, -1):
            bottom, info = _absorb_bottom_row(
                bottom,
                self.layers[row],
                max_bond=self.contractor.max_bond,
                rtol=self.contractor.rtol,
                atol=self.contractor.atol,
            )
            self._suffixes[row] = bottom
            self._suffix_infos[row] = info

    def _base_info(self):
        infos = self._prefix_infos
        return {
            "method": "boundary",
            "direction": self.direction,
            "max_bond": self.contractor.max_bond,
            "rtol": self.contractor.rtol,
            "atol": self.contractor.atol,
            "row_bond_dims": tuple(item["bond_dims"] for item in infos),
            "max_relative_error": max(
                (item["relative_error"] for item in infos),
                default=0.0,
            ),
            "discarded_weight": sum(item["discarded_weight"] for item in infos),
            "row_compressions": tuple(infos),
            "environment_reused": False,
            "environment_builds": 1,
        }

    def _oriented_replacements(self, replacements):
        replacements = dict(replacements)
        oriented = {}
        original_rows, original_cols = self.original_shape
        for coordinate, tensor in replacements.items():
            row, col = (int(value) for value in coordinate)
            if not (0 <= row < original_rows and 0 <= col < original_cols):
                raise IndexError("replacement coordinate is outside the layer grid.")
            tensor = np.asarray(tensor)
            if tensor.shape != self.original_layers[row][col].shape:
                raise ValueError("replacement layer has an incompatible shape.")
            if self.direction == "rows":
                oriented[(row, col)] = tensor
            else:
                oriented[(col, original_rows - 1 - row)] = tensor.transpose(3, 0, 1, 2)
        return oriented

    def contract_replacements(
        self,
        replacements,
        *,
        return_info=False,
        two_sided=False,
    ):
        """Contract local replacement layers using cached outer boundaries."""

        replacements = self._oriented_replacements(replacements)
        if not replacements:
            info = dict(self.info)
            info["environment_reused"] = True
            info["rows_absorbed"] = 0
            return (self.value, info) if return_info else self.value

        first_row = min(row for row, _ in replacements)
        last_row = max(row for row, _ in replacements)
        final_row = last_row if two_sided else self.nrows - 1
        if two_sided:
            self._build_suffixes()
        boundary = self._prefixes[first_row]
        local_infos = []
        for row in range(first_row, final_row + 1):
            layer_row = list(self.layers[row])
            for col in range(self.ncols):
                replacement = replacements.get((row, col))
                if replacement is not None:
                    layer_row[col] = replacement
            boundary, info = _absorb_top_row(
                boundary,
                layer_row,
                max_bond=self.contractor.max_bond,
                rtol=self.contractor.rtol,
                atol=self.contractor.atol,
            )
            local_infos.append(info)
        bottom = (
            self._suffixes[last_row + 1]
            if two_sided
            else _bottom_boundary(self.layers)
        )
        value = _contract_boundaries(boundary, bottom)

        used_infos = self._prefix_infos[:first_row] + local_infos
        if two_sided:
            used_infos += self._suffix_infos[last_row + 1 :]
        info = {
            "method": "boundary",
            "direction": self.direction,
            "max_bond": self.contractor.max_bond,
            "rtol": self.contractor.rtol,
            "atol": self.contractor.atol,
            "max_relative_error": max(
                (item["relative_error"] for item in used_infos),
                default=0.0,
            ),
            "discarded_weight": sum(
                item["discarded_weight"] for item in used_infos
            ),
            "environment_reused": True,
            "environment_builds": 0,
            "rows_absorbed": final_row - first_row + 1,
            "cached_rows": (
                first_row + self.nrows - last_row - 1
                if two_sided
                else first_row
            ),
            "two_sided": bool(two_sided),
        }
        return (value, info) if return_info else value

    def contract_many(
        self,
        replacement_maps,
        *,
        return_info=False,
        two_sided=False,
        workers=1,
    ):
        """Contract many observable channels with grouped batched frontiers."""

        oriented_maps = [self._oriented_replacements(item) for item in replacement_maps]
        results = [None] * len(oriented_maps)
        grouped = defaultdict(list)
        for index, replacements in enumerate(oriented_maps):
            if not replacements:
                info = dict(self.info)
                info.update(
                    {
                        "environment_reused": True,
                        "rows_absorbed": 0,
                        "batched_frontier": True,
                        "batch_size": 1,
                    }
                )
                results[index] = (self.value, info)
                continue
            first = min(row for row, _ in replacements)
            last = max(row for row, _ in replacements)
            grouped[(first, last if two_sided else self.nrows - 1)].append(
                (index, replacements, last)
            )
        if two_sided and grouped:
            self._build_suffixes()

        def contract_group(group):
            (first, final), jobs = group
            boundaries = [self._prefixes[first] for _ in jobs]
            local_infos = [[] for _ in jobs]
            for row in range(first, final + 1):
                layer_rows = []
                for _index, replacements, _last in jobs:
                    layer_row = list(self.layers[row])
                    for col in range(self.ncols):
                        replacement = replacements.get((row, col))
                        if replacement is not None:
                            layer_row[col] = replacement
                    layer_rows.append(layer_row)
                boundaries, infos = _absorb_top_rows_batch(
                    boundaries,
                    layer_rows,
                    max_bond=self.contractor.max_bond,
                    rtol=self.contractor.rtol,
                    atol=self.contractor.atol,
                )
                for job, info in enumerate(infos):
                    local_infos[job].append(info)

            group_results = []
            for job, ((index, _replacements, last), boundary) in enumerate(
                zip(jobs, boundaries)
            ):
                bottom = (
                    self._suffixes[last + 1]
                    if two_sided
                    else _bottom_boundary(self.layers)
                )
                value = _contract_boundaries(boundary, bottom)
                used_infos = self._prefix_infos[:first] + local_infos[job]
                if two_sided:
                    used_infos += self._suffix_infos[last + 1 :]
                info = {
                    "method": "boundary",
                    "direction": self.direction,
                    "max_bond": self.contractor.max_bond,
                    "rtol": self.contractor.rtol,
                    "atol": self.contractor.atol,
                    "max_relative_error": max(
                        (item["relative_error"] for item in used_infos),
                        default=0.0,
                    ),
                    "discarded_weight": sum(
                        item["discarded_weight"] for item in used_infos
                    ),
                    "environment_reused": True,
                    "environment_builds": 0,
                    "rows_absorbed": final - first + 1,
                    "cached_rows": (
                        first + self.nrows - last - 1 if two_sided else first
                    ),
                    "two_sided": bool(two_sided),
                    "batched_frontier": True,
                    "batch_size": len(jobs),
                }
                group_results.append((index, value, info))
            return group_results

        groups = list(grouped.items())
        if workers > 1 and len(groups) > 1:
            completed = shared_executor(workers).map(contract_group, groups)
        else:
            completed = map(contract_group, groups)
        for group_results in completed:
            for index, value, info in group_results:
                results[index] = (value, info)
        values = tuple(item[0] for item in results)
        infos = tuple(item[1] for item in results)
        return (values, infos) if return_info else values


@dataclass
class BoundaryMPSContractor:
    """Row-by-row finite-PEPS contractor with boundary-MPS compression."""

    max_bond: int | None = 64
    rtol: float = 1.0e-10
    atol: float = 0.0
    direction: str = "auto"

    def __post_init__(self):
        if self.max_bond is not None:
            if (
                isinstance(self.max_bond, bool)
                or not isinstance(self.max_bond, Integral)
                or int(self.max_bond) < 1
            ):
                raise ValueError("max_bond must be a positive integer or None.")
            self.max_bond = int(self.max_bond)
        self.rtol = float(self.rtol)
        self.atol = float(self.atol)
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative.")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        self.direction = str(self.direction).lower().replace("_", "-")
        if self.direction not in {"auto", "rows", "columns"}:
            raise ValueError("direction must be 'auto', 'rows', or 'columns'.")

    def contract(self, layers, *, return_info=False):
        """Contract a rectangular double-layer network from top to bottom."""
        layers = _validated_layers(layers)
        original_shape = (len(layers), len(layers[0]))
        direction = self.direction
        if direction == "auto":
            direction = "rows" if original_shape[1] <= original_shape[0] else "columns"
        if direction == "columns":
            layers = _rotate_layers_clockwise(layers)

        boundary = _top_boundary(layers)
        row_infos = []
        for layer_row in layers:
            boundary, compression = _absorb_top_row(
                boundary,
                layer_row,
                max_bond=self.max_bond,
                rtol=self.rtol,
                atol=self.atol,
            )
            row_infos.append(compression)
        scalar = _contract_boundaries(boundary, _bottom_boundary(layers))
        info = {
            "method": "boundary",
            "direction": direction,
            "max_bond": self.max_bond,
            "rtol": self.rtol,
            "atol": self.atol,
            "row_bond_dims": tuple(item["bond_dims"] for item in row_infos),
            "max_relative_error": max(
                (item["relative_error"] for item in row_infos),
                default=0.0,
            ),
            "discarded_weight": sum(
                item["discarded_weight"] for item in row_infos
            ),
            "row_compressions": tuple(row_infos),
        }
        return (scalar, info) if return_info else scalar

    def build_environment(self, layers):
        """Build reusable prefix/suffix boundaries for local observables."""

        return BoundaryMPSEnvironment(self, layers)


__all__ = [
    "BoundaryMPSContractor",
    "BoundaryMPSEnvironment",
    "compress_boundary_mps",
    "compress_boundary_mps_batch",
    "double_layer_tensor",
    "exact_contract_layers",
    "exact_contract_layers_with_hole",
    "shared_executor",
]
