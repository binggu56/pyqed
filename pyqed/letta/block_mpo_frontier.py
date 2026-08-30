"""Identity-aware block frontiers for graph-tied tensors and an MPO.

This is an exact prototype of :mod:`pyqed.letta.mpo_frontier` in which the
MPO bond is represented by separate blocks.  Whenever the operator suffix
from one MPO channel is diagonal on a tied physical variable, the
corresponding bra/ket frontier pair is stored as one shared index.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib

import numpy as np
from opt_einsum import contract_expression
from scipy.sparse import coo_matrix

from pyqed.tn.effective_operator import _array_module

from .copy_einsum import (
    contract_class_einsum as _contract_class_einsum,
    contract_copy_einsum as _contract_copy_einsum,
    native_available as _copy_einsum_native_available,
)


_DEFAULT_HOLE_TRANSITION_BATCH_SIZE = 2
_COPY_HOLE_AUTO_MAX_OPERATIONS = 100_000_000
_COPY_HOLE_AUTO_MAX_WORKSPACE_BYTES = 128 * 1024**2
_COPY_ADVANCE_AUTO_MAX_OPERATIONS = 100_000_000
_COPY_ADVANCE_AUTO_MAX_WORKSPACE_BYTES = 128 * 1024**2


def _ordered_bounded_map(function, values, *, max_workers, executor=None):
    """Yield ordered results with at most ``max_workers`` futures in flight."""
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    values = iter(values)
    if max_workers == 1:
        for value in values:
            yield function(value)
        return
    owned_executor = executor is None
    if owned_executor:
        executor = ThreadPoolExecutor(max_workers=max_workers)
    pending = deque()
    try:
        for _ in range(max_workers):
            value = next(values, None)
            if value is None:
                break
            pending.append(executor.submit(function, value))
        while pending:
            yield pending.popleft().result()
            value = next(values, None)
            if value is not None:
                pending.append(executor.submit(function, value))
    finally:
        if owned_executor:
            executor.shutdown()


def _exact_local_tt(tensor):
    """Factor ``(left,right,physical...)`` into an untruncated local TT."""
    tensor = np.asarray(tensor)
    order = (0, *range(2, tensor.ndim), 1)
    dimensions = tuple(tensor.shape[axis] for axis in order)
    work = tensor.transpose(order)
    cores = []
    left_rank = 1
    for dimension in dimensions[:-1]:
        matrix = work.reshape(left_rank * dimension, -1)
        q_factor, work = np.linalg.qr(matrix, mode="reduced")
        left_rank = q_factor.shape[1]
        cores.append(q_factor.reshape(-1, dimension, left_rank))
    cores.append(work.reshape(left_rank, dimensions[-1], 1))
    return tuple(cores)


def _validated_local_tt_options(max_rank, rtol, atol):
    if max_rank is not None:
        max_rank = int(max_rank)
        if max_rank < 1:
            raise ValueError("local_rank must be positive or None.")
    rtol = float(rtol)
    atol = float(atol)
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("local_rtol must be finite and nonnegative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("local_atol must be finite and nonnegative.")
    return max_rank, rtol, atol


def _charge_sum(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _configuration_charges(axis_qns):
    rank = len(axis_qns[0][0])
    result = (tuple(0 for _ in range(rank)),)
    for labels in axis_qns:
        result = tuple(
            _charge_sum(prefix, charge)
            for prefix in result
            for charge in labels
        )
    return result


def _retained_singular_indices(singular_values, *, max_rank, tolerance):
    indexed = [
        (float(value), block, index)
        for block, values in enumerate(singular_values)
        for index, value in enumerate(values)
    ]
    indexed.sort(reverse=True)
    total_squared = sum(value * value for value, _block, _index in indexed)
    discarded_squared = total_squared
    keep = 0
    while keep < len(indexed) and np.sqrt(max(0.0, discarded_squared)) > tolerance:
        discarded_squared -= indexed[keep][0] ** 2
        keep += 1
    if indexed:
        keep = max(1, keep)
    if max_rank is not None:
        keep = min(keep, max_rank)
    selected = [set() for _values in singular_values]
    for _value, block, index in indexed[:keep]:
        selected[block].add(index)
    discarded_squared = sum(
        float(value) ** 2
        for block, values in enumerate(singular_values)
        for index, value in enumerate(values)
        if index not in selected[block]
    )
    return tuple(selected), float(max(0.0, discarded_squared))


def _truncated_local_tt(
    tensor,
    *,
    max_rank=None,
    rtol=0.0,
    atol=0.0,
    axis_qns=None,
):
    """TT-SVD a local tensor, optionally in exact cumulative-charge blocks."""
    max_rank, rtol, atol = _validated_local_tt_options(max_rank, rtol, atol)
    tensor = np.asarray(tensor)
    order = (0, *range(2, tensor.ndim), 1)
    dimensions = tuple(tensor.shape[axis] for axis in order)
    work = tensor.transpose(order)
    if axis_qns is not None:
        axis_qns = tuple(
            tuple(tuple(int(value) for value in charge) for charge in labels)
            for labels in axis_qns
        )
        if len(axis_qns) != len(dimensions) or any(
            len(labels) != dimension
            for labels, dimension in zip(axis_qns, dimensions)
        ):
            raise ValueError("axis_qns are inconsistent with the local tensor.")
        charge_rank = len(axis_qns[0][0])
        if any(
            len(charge) != charge_rank
            for labels in axis_qns
            for charge in labels
        ):
            raise ValueError("all local TT charges must have the same rank.")
        left_qns = (tuple(0 for _ in range(charge_rank)),)
    else:
        left_qns = None

    cores = []
    ranks = []
    discarded_squared = []
    left_rank = 1
    for axis, dimension in enumerate(dimensions[:-1]):
        remaining_dimensions = dimensions[axis + 1 :]
        matrix = work.reshape(left_rank * dimension, -1)
        matrix_norm = float(np.linalg.norm(matrix))
        tolerance = max(atol, rtol * matrix_norm)
        if axis_qns is None:
            left, singular, right = np.linalg.svd(matrix, full_matrices=False)
            selected, discarded = _retained_singular_indices(
                (singular,),
                max_rank=max_rank,
                tolerance=tolerance,
            )
            indices = tuple(sorted(selected[0]))
            left = left[:, indices]
            work = singular[list(indices), None] * right[list(indices)]
            next_qns = None
        else:
            row_qns = tuple(
                _charge_sum(left_charge, physical_charge)
                for left_charge in left_qns
                for physical_charge in axis_qns[axis]
            )
            column_qns = _configuration_charges(axis_qns[axis + 1 :])
            sectors = tuple(sorted(set(row_qns)))
            decompositions = []
            for charge in sectors:
                rows = np.asarray(
                    [index for index, value in enumerate(row_qns) if value == charge],
                    dtype=int,
                )
                opposite = tuple(-value for value in charge)
                columns = np.asarray(
                    [
                        index
                        for index, value in enumerate(column_qns)
                        if value == opposite
                    ],
                    dtype=int,
                )
                if not rows.size or not columns.size:
                    continue
                block = matrix[np.ix_(rows, columns)]
                left, singular, right = np.linalg.svd(block, full_matrices=False)
                decompositions.append((charge, rows, columns, left, singular, right))
            selected, discarded = _retained_singular_indices(
                tuple(values[4] for values in decompositions),
                max_rank=max_rank,
                tolerance=tolerance,
            )
            next_rank = sum(len(indices) for indices in selected)
            left_factor = np.zeros(
                (matrix.shape[0], next_rank),
                dtype=np.result_type(matrix.dtype, np.float64),
            )
            following = np.zeros(
                (next_rank, matrix.shape[1]),
                dtype=np.result_type(matrix.dtype, np.float64),
            )
            next_qns = []
            offset = 0
            for decomposition, indices in zip(decompositions, selected):
                charge, rows, columns, left, singular, right = decomposition
                indices = tuple(sorted(indices))
                count = len(indices)
                if not count:
                    continue
                left_factor[np.ix_(rows, range(offset, offset + count))] = left[
                    :, indices
                ]
                following[np.ix_(range(offset, offset + count), columns)] = (
                    singular[list(indices), None] * right[list(indices)]
                )
                next_qns.extend((charge,) * count)
                offset += count
            left = left_factor
            work = following
            next_qns = tuple(next_qns)
        left_rank = left.shape[1]
        cores.append(left.reshape(-1, dimension, left_rank))
        work = work.reshape((left_rank, *remaining_dimensions))
        ranks.append(left_rank)
        discarded_squared.append(discarded)
        left_qns = next_qns
    cores.append(work.reshape(left_rank, dimensions[-1], 1))
    tensor_norm = float(np.linalg.norm(tensor))
    discarded_norm = float(np.sqrt(sum(discarded_squared)))
    diagnostics = {
        "ranks": tuple(ranks),
        "discarded_norm": discarded_norm,
        "relative_discarded_norm": (
            discarded_norm / tensor_norm if tensor_norm else 0.0
        ),
        "charge_resolved": axis_qns is not None,
    }
    return tuple(cores), diagnostics


@dataclass(frozen=True)
class BlockFrontierMessage:
    """Channel-blocked numerical message at one site cut."""

    cut: int
    blocks: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _FactorizedMPOProduct:
    """Sparse fused-channel representation of an exact MPO product."""

    left_tensors: tuple[np.ndarray, ...]
    right_tensors: tuple[np.ndarray, ...]
    bond_dims: tuple[int, ...]
    operators: tuple[dict[tuple[int, int], np.ndarray], ...]
    dtype: np.dtype
    stored_elements: int
    dense_elements: int


class BlockMPOFrontier:
    r"""Cache identity-aware frontier messages for a graph-tied state.

    For MPO channel ``a`` at cut ``c``, let ``Q[c, a]`` contain every future
    physical site on which at least one completed MPO suffix path can be
    off-diagonal.  A frontier site in ``Q[c, a]`` carries separate bra and ket
    values; every other site carries one value shared by bra and ket.  The
    latter reduction is exact because the corresponding suffix operator is
    diagonal on that site.

    The implementation deliberately consumes the uncompressed MPO tensors.
    A generic MPO compression can mix diagonal and off-diagonal automaton
    channels and therefore hide the block structure used here.
    """

    def __init__(
        self,
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
        *,
        optimize="greedy",
        local_qns=None,
        bond_qns=None,
        local_backend="dense",
        local_rank=None,
        local_rtol=0.0,
        local_atol=0.0,
    ):
        factorized_product = (
            mpo_tensors if isinstance(mpo_tensors, _FactorizedMPOProduct) else None
        )
        self.dims = tuple(int(dim) for dim in dims)
        self.physical_groups = tuple(
            tuple(int(site) for site in sites) for sites in physical_sites
        )
        self.tensor_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in tensor_shapes
        )
        self.factorized_mpo = factorized_product is not None
        if self.factorized_mpo:
            self.mpo_tensors = None
            self.left_mpo_tensors = factorized_product.left_tensors
            self.right_mpo_tensors = factorized_product.right_tensors
            self._transition_operators = factorized_product.operators
            self.stored_mpo_elements = factorized_product.stored_elements
            self.dense_mpo_elements = factorized_product.dense_elements
            mpo_nsites = len(factorized_product.operators)
        else:
            self.mpo_tensors = tuple(np.asarray(tensor) for tensor in mpo_tensors)
            self.left_mpo_tensors = None
            self.right_mpo_tensors = None
            self._transition_operators = None
            self.stored_mpo_elements = int(
                sum(tensor.size for tensor in self.mpo_tensors)
            )
            self.dense_mpo_elements = self.stored_mpo_elements
            mpo_nsites = len(self.mpo_tensors)
        self.optimize = optimize
        self.local_backend = str(local_backend).lower().replace("-", "_")
        if self.local_backend in {"tt", "sequential"}:
            self.local_backend = "tensor_train"
        if self.local_backend not in {"dense", "tensor_train"}:
            raise ValueError("local_backend must be 'dense' or 'tensor_train'.")
        self.local_rank, self.local_rtol, self.local_atol = (
            _validated_local_tt_options(local_rank, local_rtol, local_atol)
        )
        if self.local_backend == "dense" and (
            self.local_rank is not None or self.local_rtol or self.local_atol
        ):
            raise ValueError(
                "local_rank/local_rtol/local_atol require "
                "local_backend='tensor_train'."
            )
        self.local_qns = local_qns
        self.bond_qns = bond_qns
        self.charge_resolved = local_qns is not None or bond_qns is not None
        if self.charge_resolved:
            if local_qns is None or bond_qns is None:
                raise ValueError("local_qns and bond_qns must be supplied together.")
            self.local_qns = tuple(
                tuple(tuple(int(x) for x in charge) for charge in site)
                for site in local_qns
            )
            self.bond_qns = tuple(
                tuple(tuple(int(x) for x in charge) for charge in bond)
                for bond in bond_qns
            )
        self.nsites = len(self.dims)
        if not self.dims or any(dim < 1 for dim in self.dims):
            raise ValueError("dims must contain positive local dimensions.")
        if not (
            len(self.physical_groups)
            == len(self.tensor_shapes)
            == mpo_nsites
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")

        for site, (sites, shape) in enumerate(
            zip(self.physical_groups, self.tensor_shapes)
        ):
            if not sites or sites[0] != site:
                raise ValueError("physical_sites must begin with the tensor site.")
            if len(set(sites)) != len(sites) or any(
                physical_site < site or physical_site >= self.nsites
                for physical_site in sites
            ):
                raise ValueError("physical_sites must contain unique forward sites.")
            expected_physical = tuple(self.dims[index] for index in sites)
            if len(shape) != 2 + len(sites) or shape[2:] != expected_physical:
                raise ValueError(
                    f"tensor shape {site} is inconsistent with physical_sites."
                )
        self.virtual_bonds = (self.tensor_shapes[0][0],) + tuple(
            shape[1] for shape in self.tensor_shapes
        )
        if self.charge_resolved:
            if len(self.local_qns) != self.nsites:
                raise ValueError("local_qns must contain one entry per site.")
            if len(self.bond_qns) != self.nsites + 1:
                raise ValueError("bond_qns must contain one entry per virtual cut.")
            if any(
                len(labels) != dimension
                for labels, dimension in zip(self.local_qns, self.dims)
            ):
                raise ValueError("local_qns dimensions are inconsistent with dims.")
            if any(
                len(labels) != dimension
                for labels, dimension in zip(self.bond_qns, self.virtual_bonds)
            ):
                raise ValueError(
                    "bond_qns dimensions are inconsistent with virtual bonds."
                )
        if self.factorized_mpo:
            self.mpo_bonds = factorized_product.bond_dims
        else:
            self.mpo_bonds = (self.mpo_tensors[0].shape[0],) + tuple(
                tensor.shape[1] for tensor in self.mpo_tensors
            )
        for site, shape in enumerate(self.tensor_shapes):
            if shape[0] != self.virtual_bonds[site]:
                raise ValueError(f"virtual bond mismatch before site {site}.")
            if not self.factorized_mpo:
                mpo = self.mpo_tensors[site]
                if mpo.shape != (
                    self.mpo_bonds[site],
                    self.mpo_bonds[site + 1],
                    self.dims[site],
                    self.dims[site],
                ):
                    raise ValueError(f"MPO tensor {site} has an invalid shape.")
        if self.virtual_bonds[0] != 1 or self.virtual_bonds[-1] != 1:
            raise ValueError("frontier virtual boundary dimensions must be one.")
        if self.mpo_bonds[0] != 1 or self.mpo_bonds[-1] != 1:
            raise ValueError("frontier MPO boundary dimensions must be one.")

        first_reference = list(range(self.nsites))
        for left_site, group in enumerate(self.physical_groups):
            for physical_site in group[1:]:
                first_reference[physical_site] = min(
                    first_reference[physical_site], left_site
                )
        self.frontier_sites = tuple(
            tuple(
                physical_site
                for physical_site in range(cut, self.nsites)
                if first_reference[physical_site] < cut
            )
            for cut in range(self.nsites + 1)
        )

        start = 0
        self.bra_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.ket_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.bra_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.ket_physical = tuple(range(start, start + self.nsites))
        self._next_auxiliary_label = start + self.nsites

        if self.factorized_mpo:
            self._transitions = tuple(
                tuple(sorted(operators)) for operators in self._transition_operators
            )
        else:
            self._transitions = tuple(
                tuple(
                    (left, right)
                    for left in range(tensor.shape[0])
                    for right in range(tensor.shape[1])
                    if np.any(tensor[left, right] != 0)
                )
                for tensor in self.mpo_tensors
        )
        self._prefix_reachable = self._build_prefix_reachable()
        self._suffix_masks = self._build_suffix_masks()
        self._paired_sites_cache = {}
        self._block_shape_cache = {}
        self._message_elements_cache = {}
        self._active_channels = tuple(
            tuple(
                channel
                for channel, (from_left, to_right) in enumerate(
                    zip(self._prefix_reachable[cut], self._suffix_reachable[cut])
                )
                if from_left and to_right
            )
            for cut in range(self.nsites + 1)
        )
        self._active_channel_sets = tuple(map(frozenset, self._active_channels))
        self._bond_pairs = self._build_bond_pairs()
        self._left_transition_groups = tuple(
            self._group_transitions(site, "left") for site in range(self.nsites)
        )
        self._right_transition_groups = tuple(
            self._group_transitions(site, "right") for site in range(self.nsites)
        )
        self._hole_transition_groups = tuple(
            self._group_transitions(site, "hole") for site in range(self.nsites)
        )
        self._operator_batches = {}
        self._block_paths = {}
        self._copies = {dim: self._copy_tensor(dim) for dim in set(self.dims)}
        self._identities = {dim: np.eye(dim) for dim in set(self.dims)}
        self._pair_selectors = {}
        self._expressions = {}
        self._copy_hole_plans = {}
        self._copy_advance_plans = {}
        self._local_tt_labels = {}
        self._local_diagnostics = {
            "factorizations": 0,
            "peak_rank": 0,
            "max_relative_discarded_norm": 0.0,
            "last": None,
        }
        self.dtype = (
            factorized_product.dtype
            if self.factorized_mpo
            else np.dtype(
                np.result_type(*[tensor.dtype for tensor in self.mpo_tensors])
            )
        )
        self._compute_dtype_requested = not (
            compute_dtype is None
            or str(compute_dtype).lower() in {"same", "native"}
        )
        if not self._compute_dtype_requested:
            compute_dtype = self.dtype
        compute_dtype = np.dtype(compute_dtype)
        if self.dtype.kind == "c" and compute_dtype.kind != "c":
            compute_dtype = np.dtype(
                np.complex64 if compute_dtype.itemsize <= 4 else np.complex128
            )
        if compute_dtype.kind not in "fc":
            raise TypeError("compute_dtype must be a real or complex floating dtype.")
        self.compute_dtype = compute_dtype
        if route_memory is None:
            route_memory = 0
        self.route_memory = float(route_memory)
        if not np.isfinite(self.route_memory) or self.route_memory < 0.0:
            raise ValueError("route_memory must be finite and nonnegative.")
        self._packed_route_limit_bytes = int(self.route_memory * 1024**2)
        self.action_memory = float(action_memory)
        if not np.isfinite(self.action_memory) or self.action_memory < 0.0:
            raise ValueError("action_memory must be finite and nonnegative.")
        self._packed_dense_action_limit_bytes = int(self.action_memory * 1024**2)
        self._packed_route_cache_bytes = 0
        self._packed_route_lru = OrderedDict()
        self._packed_route_cache_hits = 0
        self._packed_route_cache_misses = 0
        self._packed_route_cache_evictions = 0
        self._charge_sum_support_cache = {}
        self._physical_support_cache = {}
        self._storage_coordinate_cache = {}
        self._storage_inverse_cache = {}
        self._bond_pair_array_cache = {}
        self._tensor_support_cache = {}
        self._tensor_support_array_cache = {}
        self._packed_advance_plan_cache = OrderedDict()
        self._packed_hole_plan_cache = OrderedDict()
        self.last_packed_advance_peak_bytes = 0
        self.last_packed_hole_peak_bytes = 0
        self._compiled_route_builders_cache = None
        self.packed_route_backend = "python"

    @staticmethod
    def _validated_mpo_factor(dims, tensors, name):
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        if len(tensors) != len(dims):
            raise ValueError(f"{name} must contain one tensor per site.")
        previous = 1
        bond_dims = [previous]
        for site, (dim, tensor) in enumerate(zip(dims, tensors)):
            if tensor.ndim != 4:
                raise ValueError(f"{name} tensor {site} must have four axes.")
            if tensor.shape[0] != previous:
                raise ValueError(f"{name} bond mismatch before site {site}.")
            if tensor.shape[2:] != (dim, dim):
                raise ValueError(f"{name} tensor {site} has wrong physical shape.")
            previous = int(tensor.shape[1])
            bond_dims.append(previous)
        if previous != 1:
            raise ValueError(f"the final {name} bond dimension must be one.")
        return tensors, tuple(bond_dims)

    @classmethod
    def from_product(
        cls,
        dims,
        physical_sites,
        tensor_shapes,
        left_mpo_tensors,
        right_mpo_tensors,
        **options,
    ):
        r"""Build an exact frontier for an MPO product without fusing dense cores.

        If the factor channels at a cut are ``a`` and ``c``, their fused
        channel is ``a * C + c``.  Thus a pair of transitions
        ``a -> b`` and ``c -> d`` becomes ``(a,c) -> (b,d)`` and stores only
        the nonzero local operator

        .. math:: W^{(L)}_{ab} W^{(R)}_{cd}.

        This is the same channel flattening and physical-index contraction as
        :meth:`LocalMPO.compose`, but zero fused transitions are never
        allocated.
        """
        dims = tuple(int(dim) for dim in dims)
        if not dims or any(dim < 1 for dim in dims):
            raise ValueError("dims must contain positive local dimensions.")
        left_tensors, left_bonds = cls._validated_mpo_factor(
            dims, left_mpo_tensors, "left_mpo_tensors"
        )
        right_tensors, right_bonds = cls._validated_mpo_factor(
            dims, right_mpo_tensors, "right_mpo_tensors"
        )
        bond_dims = tuple(
            left * right for left, right in zip(left_bonds, right_bonds)
        )
        operators = []
        stored_elements = 0
        dense_elements = 0
        for site, (dim, left_tensor, right_tensor) in enumerate(
            zip(dims, left_tensors, right_tensors)
        ):
            dense_elements += (
                bond_dims[site] * bond_dims[site + 1] * dim * dim
            )
            left_transitions = tuple(
                (left, following, left_tensor[left, following])
                for left in range(left_tensor.shape[0])
                for following in range(left_tensor.shape[1])
                if np.any(left_tensor[left, following] != 0)
            )
            right_transitions = tuple(
                (right, following, right_tensor[right, following])
                for right in range(right_tensor.shape[0])
                for following in range(right_tensor.shape[1])
                if np.any(right_tensor[right, following] != 0)
            )
            site_operators = {}
            right_left_bond = right_tensor.shape[0]
            right_right_bond = right_tensor.shape[1]
            for left, next_left, left_operator in left_transitions:
                for right, next_right, right_operator in right_transitions:
                    operator = left_operator @ right_operator
                    if not np.any(operator != 0):
                        continue
                    transition = (
                        left * right_left_bond + right,
                        next_left * right_right_bond + next_right,
                    )
                    site_operators[transition] = operator
                    stored_elements += operator.size
            operators.append(site_operators)
        product = _FactorizedMPOProduct(
            left_tensors=left_tensors,
            right_tensors=right_tensors,
            bond_dims=bond_dims,
            operators=tuple(operators),
            dtype=np.dtype(
                np.result_type(
                    *(tensor.dtype for tensor in left_tensors),
                    *(tensor.dtype for tensor in right_tensors),
                )
            ),
            stored_elements=int(stored_elements),
            dense_elements=int(dense_elements),
        )
        return cls(
            dims,
            physical_sites,
            tensor_shapes,
            product,
            **options,
        )

    @property
    def contraction_is_exact(self):
        return (
            self.local_backend == "dense"
            or (
                self.local_rank is None
                and self.local_rtol == 0.0
                and self.local_atol == 0.0
            )
        )

    @property
    def diagnostics(self):
        result = dict(self._local_diagnostics)
        result.update(
            {
                "factorized_mpo": self.factorized_mpo,
                "stored_mpo_elements": self.stored_mpo_elements,
                "dense_mpo_elements": self.dense_mpo_elements,
            }
        )
        return result

    def _operator(self, site, left, right):
        """Return one nonzero local MPO transition operator."""
        site = int(site)
        transition = (int(left), int(right))
        if self.factorized_mpo:
            return self._transition_operators[site][transition]
        return self.mpo_tensors[site][transition]

    def _local_axis_qns(self, site):
        if not self.charge_resolved:
            return None
        rank = len(self.local_qns[site][0])
        zero = tuple(0 for _ in range(rank))
        return (
            self.bond_qns[site],
            self.local_qns[site],
            *(
                tuple(zero for _ in range(self.dims[parent]))
                for parent in self.physical_sites[site][1:]
            ),
            tuple(
                tuple(-value for value in charge)
                for charge in self.bond_qns[site + 1]
            ),
        )

    def _factor_local_tensor(self, site, tensor):
        if self.contraction_is_exact:
            cores = _exact_local_tt(tensor)
            diagnostics = {
                "ranks": tuple(core.shape[2] for core in cores[:-1]),
                "discarded_norm": 0.0,
                "relative_discarded_norm": 0.0,
                "charge_resolved": self.charge_resolved,
            }
        else:
            cores, diagnostics = _truncated_local_tt(
                tensor,
                max_rank=self.local_rank,
                rtol=self.local_rtol,
                atol=self.local_atol,
                axis_qns=self._local_axis_qns(int(site)),
            )
        self._local_diagnostics["factorizations"] += 1
        self._local_diagnostics["peak_rank"] = max(
            self._local_diagnostics["peak_rank"],
            *diagnostics["ranks"],
        )
        self._local_diagnostics["max_relative_discarded_norm"] = max(
            self._local_diagnostics["max_relative_discarded_norm"],
            diagnostics["relative_discarded_norm"],
        )
        self._local_diagnostics["last"] = diagnostics
        return cores

    @staticmethod
    def _copy_tensor(dim):
        value = np.zeros((dim, dim, dim))
        diagonal = np.arange(dim)
        value[diagonal, diagonal, diagonal] = 1.0
        return value

    @staticmethod
    def _has_offdiagonal(matrix):
        matrix = np.asarray(matrix)
        rows, columns = np.nonzero(matrix)
        return bool(np.any(rows != columns))

    def _build_prefix_reachable(self):
        reachable = [np.zeros(bond, dtype=bool) for bond in self.mpo_bonds]
        reachable[0][0] = True
        for site in range(self.nsites):
            for left, right in self._transitions[site]:
                if reachable[site][left]:
                    reachable[site + 1][right] = True
        return tuple(reachable)

    @staticmethod
    def _charge_difference(left, right):
        return tuple(a - b for a, b in zip(left, right))

    def _build_bond_pairs(self):
        if not self.charge_resolved:
            return None
        transfers = [[set() for _ in range(bond)] for bond in self.mpo_bonds]
        zero = tuple(0 for _ in self.local_qns[0][0])
        transfers[0][0].add(zero)
        for site in range(self.nsites):
            for left, right in self._transitions[site]:
                local = {
                    self._charge_difference(
                        self.local_qns[site][bra], self.local_qns[site][ket]
                    )
                    for bra, ket in zip(
                        *np.nonzero(self._operator(site, left, right))
                    )
                }
                for prefix in transfers[site][left]:
                    for delta in local:
                        transfers[site + 1][right].add(
                            tuple(a + b for a, b in zip(prefix, delta))
                        )
        result = []
        for cut, labels in enumerate(self.bond_qns):
            cut_pairs = []
            for channel in range(self.mpo_bonds[cut]):
                allowed = transfers[cut][channel]
                cut_pairs.append(
                    tuple(
                        (bra, ket)
                        for bra, q_bra in enumerate(labels)
                        for ket, q_ket in enumerate(labels)
                        if self._charge_difference(q_bra, q_ket) in allowed
                    )
                )
            result.append(tuple(cut_pairs))
        return tuple(result)

    def _pair_selector(self, cut, channel):
        key = (int(cut), int(channel))
        value = self._pair_selectors.get(key)
        if value is None:
            pairs = self._bond_pairs[key[0]][key[1]]
            dimension = self.virtual_bonds[key[0]]
            value = np.zeros((len(pairs), dimension, dimension))
            for pair, (bra, ket) in enumerate(pairs):
                value[pair, bra, ket] = 1.0
            self._pair_selectors[key] = value
        return value

    def _build_suffix_masks(self):
        can_finish = [None] * (self.nsites + 1)
        masks = [None] * (self.nsites + 1)
        can_finish[-1] = np.ones(self.mpo_bonds[-1], dtype=bool)
        masks[-1] = (0,) * self.mpo_bonds[-1]
        for site in range(self.nsites - 1, -1, -1):
            reachable = np.zeros(self.mpo_bonds[site], dtype=bool)
            site_masks = [0] * self.mpo_bonds[site]
            for left, right in self._transitions[site]:
                if not can_finish[site + 1][right]:
                    continue
                reachable[left] = True
                site_masks[left] |= masks[site + 1][right]
                if self._has_offdiagonal(self._operator(site, left, right)):
                    site_masks[left] |= 1 << site
            can_finish[site] = reachable
            masks[site] = tuple(site_masks)
        self._suffix_reachable = tuple(can_finish)
        return tuple(masks)

    @property
    def plan_count(self):
        return len(self._expressions)

    def clear_contraction_plans(self):
        """Release value-independent contraction workspaces and selectors."""

        self._operator_batches.clear()
        self._block_paths.clear()
        self._pair_selectors.clear()
        self._expressions.clear()
        self._packed_advance_plan_cache.clear()
        self._packed_hole_plan_cache.clear()
        self._packed_route_lru.clear()
        self._packed_route_cache_bytes = 0

    @staticmethod
    def _packed_plan_bytes(plan):
        return int(sum(np.asarray(value).nbytes for value in plan))

    def _cached_packed_plan(self, kind, key):
        cache = (
            self._packed_advance_plan_cache
            if kind == "advance"
            else self._packed_hole_plan_cache
        )
        plan = cache.get(key)
        if plan is None:
            self._packed_route_cache_misses += 1
            return None
        self._packed_route_cache_hits += 1
        self._packed_route_lru.move_to_end((kind, key))
        cache.move_to_end(key)
        return plan

    def _cache_packed_plan(self, kind, key, plan):
        size = self._packed_plan_bytes(plan)
        if size > self._packed_route_limit_bytes:
            return plan
        cache = (
            self._packed_advance_plan_cache
            if kind == "advance"
            else self._packed_hole_plan_cache
        )
        old = cache.pop(key, None)
        if old is not None:
            old_size = self._packed_route_lru.pop((kind, key))
            self._packed_route_cache_bytes -= old_size
        while (
            self._packed_route_lru
            and self._packed_route_cache_bytes + size
            > self._packed_route_limit_bytes
        ):
            (old_kind, old_key), old_size = self._packed_route_lru.popitem(
                last=False
            )
            old_cache = (
                self._packed_advance_plan_cache
                if old_kind == "advance"
                else self._packed_hole_plan_cache
            )
            old_cache.pop(old_key, None)
            self._packed_route_cache_bytes -= old_size
            self._packed_route_cache_evictions += 1
        cache[key] = plan
        self._packed_route_lru[(kind, key)] = size
        self._packed_route_cache_bytes += size
        return plan

    @property
    def packed_route_cache_stats(self):
        return {
            "bytes": self._packed_route_cache_bytes,
            "limit_bytes": self._packed_route_limit_bytes,
            "plans": len(self._packed_route_lru),
            "hits": self._packed_route_cache_hits,
            "misses": self._packed_route_cache_misses,
            "evictions": self._packed_route_cache_evictions,
        }

    def paired_sites(self, cut, channel):
        cut = int(cut)
        channel = int(channel)
        key = (cut, channel)
        cached = self._paired_sites_cache.get(key)
        if cached is not None:
            return cached
        mask = self._suffix_masks[cut][channel]
        cached = tuple(
            site for site in self.frontier_sites[cut] if mask & (1 << site)
        )
        self._paired_sites_cache[key] = cached
        return cached

    def _group_transitions(self, site, mode):
        """Group MPO transitions with one common contraction topology."""
        groups = {}
        for left, right in self._transitions[int(site)]:
            if (
                left not in self._active_channel_sets[int(site)]
                or right not in self._active_channel_sets[int(site) + 1]
            ):
                continue
            if mode == "left":
                key = (
                    right,
                    self.paired_sites(site, left),
                    self._bond_pairs[site][left]
                    if self.charge_resolved else None,
                )
            elif mode == "right":
                key = (
                    left,
                    self.paired_sites(site + 1, right),
                    self._bond_pairs[site + 1][right]
                    if self.charge_resolved else None,
                )
            elif mode == "hole":
                key = (
                    self.paired_sites(site, left),
                    self.paired_sites(site + 1, right),
                    self._bond_pairs[site][left]
                    if self.charge_resolved else None,
                    self._bond_pairs[site + 1][right]
                    if self.charge_resolved else None,
                )
            else:
                raise ValueError("transition group mode is invalid.")
            groups.setdefault(key, []).append((left, right))
        return tuple(tuple(transitions) for transitions in groups.values())

    def _operators_for_transitions(self, site, transitions):
        if len(transitions) == 1:
            left, right = transitions[0]
            return self._operator(site, left, right)
        key = (int(site), transitions)
        cached = self._operator_batches.get(key)
        if cached is None:
            cached = np.stack(
                [
                    self._operator(site, left, right)
                    for left, right in transitions
                ]
            )
            self._operator_batches[key] = cached
        return cached

    def block_shape(self, cut, channel):
        cut = int(cut)
        channel = int(channel)
        key = (cut, channel)
        cached = self._block_shape_cache.get(key)
        if cached is not None:
            return cached
        paired = set(self.paired_sites(cut, channel))
        shape = (
            [len(self._bond_pairs[cut][channel])]
            if self.charge_resolved
            else [self.virtual_bonds[cut], self.virtual_bonds[cut]]
        )
        for site in self.frontier_sites[cut]:
            shape.append(self.dims[site])
            if site in paired:
                shape.append(self.dims[site])
        cached = tuple(shape)
        self._block_shape_cache[key] = cached
        return cached

    @staticmethod
    def _add_charge(left, right):
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _subtract_charge(left, right):
        return tuple(a - b for a, b in zip(left, right))

    def _charge_sum_support(self, sites):
        sites = tuple(int(site) for site in sites)
        cached = self._charge_sum_support_cache.get(sites)
        if cached is not None:
            return cached
        rank = len(self.local_qns[0][0])
        support = {tuple(0 for _ in range(rank))}
        for site in sites:
            support = {
                self._add_charge(prefix, charge)
                for prefix in support
                for charge in self.local_qns[site]
            }
        result = frozenset(support)
        self._charge_sum_support_cache[sites] = result
        return result

    def _physical_support_indices(self, cut, channel):
        if not self.charge_resolved:
            return None
        key = (int(cut), int(channel))
        if key in self._physical_support_cache:
            return self._physical_support_cache[key]
        cut, channel = key
        shape = self.block_shape(cut, channel)
        size = int(np.prod(shape, dtype=np.int64))
        frontier = self.frontier_sites[cut]
        free_sites = tuple(site for site in range(cut, self.nsites) if site not in frontier)
        free_support = self._charge_sum_support(free_sites)
        target = self.bond_qns[-1][0]
        paired = set(self.paired_sites(cut, channel))
        valid = np.zeros(size, dtype=bool)
        pairs = self._bond_pairs[cut][channel]
        for flat, configuration in enumerate(np.ndindex(*shape)):
            bra_bond, ket_bond = pairs[configuration[0]]
            bra_charge = self.bond_qns[cut][bra_bond]
            ket_charge = self.bond_qns[cut][ket_bond]
            offset = 1
            for site in frontier:
                bra_value = configuration[offset]
                offset += 1
                if site in paired:
                    ket_value = configuration[offset]
                    offset += 1
                else:
                    ket_value = bra_value
                bra_charge = self._add_charge(
                    bra_charge,
                    self.local_qns[site][bra_value],
                )
                ket_charge = self._add_charge(
                    ket_charge,
                    self.local_qns[site][ket_value],
                )
            valid[flat] = (
                self._subtract_charge(target, bra_charge) in free_support
                and self._subtract_charge(target, ket_charge) in free_support
            )
        indices = np.asarray(np.flatnonzero(valid), dtype=np.int32)
        result = None if indices.size == size else indices
        self._physical_support_cache[key] = result
        return result

    def storage_shape(self, cut, channel):
        support = self._physical_support_indices(cut, channel)
        return self.block_shape(cut, channel) if support is None else (int(support.size),)

    def _store_block(self, cut, channel, block):
        block = np.asarray(block)
        logical_shape = self.block_shape(cut, channel)
        if block.shape != logical_shape:
            block = block.reshape(logical_shape)
        support = self._physical_support_indices(cut, channel)
        return block if support is None else np.asarray(block).reshape(-1)[support].copy()

    def _logical_block(self, cut, channel, block):
        block = np.asarray(block)
        logical_shape = self.block_shape(cut, channel)
        support = self._physical_support_indices(cut, channel)
        if support is None:
            return block.reshape(logical_shape)
        result = np.zeros(logical_shape, dtype=block.dtype)
        result.reshape(-1)[support] = block.reshape(-1)
        return result

    def _message_block(self, message, channel):
        return self._logical_block(message.cut, int(channel), message.blocks[int(channel)])

    def _stack_message_blocks(self, message, channels):
        return np.stack([self._message_block(message, channel) for channel in channels])

    def _stored_message(self, cut, blocks):
        result = list(blocks)
        for channel in self._active_channels[int(cut)]:
            result[channel] = self._store_block(cut, channel, result[channel])
        return BlockFrontierMessage(int(cut), tuple(result))

    def _storage_coordinates(self, cut, channel):
        """Logical coordinates represented by one packed message block."""

        key = (int(cut), int(channel))
        cached = self._storage_coordinate_cache.get(key)
        if cached is not None:
            return cached
        shape = self.block_shape(*key)
        support = self._physical_support_indices(*key)
        flat = (
            np.arange(int(np.prod(shape, dtype=np.int64)), dtype=np.int32)
            if support is None
            else np.asarray(support, dtype=np.int32)
        )
        coordinates = np.empty((flat.size, len(shape)), dtype=np.int32)
        if flat.size == 0:
            self._storage_coordinate_cache[key] = coordinates
            return coordinates
        stride = int(np.prod(shape, dtype=np.int64))
        for axis, dimension in enumerate(shape):
            stride //= int(dimension)
            coordinates[:, axis] = (flat // stride) % int(dimension)
        self._storage_coordinate_cache[key] = coordinates
        return coordinates

    def _storage_inverse(self, cut, channel):
        """Map a logical flat index to its packed storage position."""

        key = (int(cut), int(channel))
        cached = self._storage_inverse_cache.get(key)
        if cached is not None:
            return cached
        shape = self.block_shape(*key)
        size = int(np.prod(shape, dtype=np.int64))
        support = self._physical_support_indices(*key)
        if support is None:
            inverse = np.arange(size, dtype=np.int32)
        else:
            inverse = np.full(size, -1, dtype=np.int32)
            inverse[np.asarray(support, dtype=np.int32)] = np.arange(
                len(support), dtype=np.int32
            )
        self._storage_inverse_cache[key] = inverse
        return inverse

    def _expanded_storage_coordinates(self, cut, channel):
        """Decode packed coordinates into bonds and bra/ket frontier values."""

        coordinates = self._storage_coordinates(cut, channel)
        key = (int(cut), int(channel))
        pairs = self._bond_pair_array_cache.get(key)
        if pairs is None:
            pairs = np.asarray(self._bond_pairs[key[0]][key[1]], dtype=np.int32)
            pairs = pairs.reshape(-1, 2)
            self._bond_pair_array_cache[key] = pairs
        pair_values = pairs[coordinates[:, 0]]
        bra = {}
        ket = {}
        paired = set(self.paired_sites(cut, channel))
        offset = 1
        for physical_site in self.frontier_sites[int(cut)]:
            bra[physical_site] = coordinates[:, offset]
            offset += 1
            if physical_site in paired:
                ket[physical_site] = coordinates[:, offset]
                offset += 1
            else:
                ket[physical_site] = bra[physical_site]
        return pair_values[:, 0], pair_values[:, 1], bra, ket

    def _tensor_support(self, site):
        """Return every structurally allowed U(1) entry of one graph tensor."""

        site = int(site)
        cached = self._tensor_support_cache.get(site)
        if cached is not None:
            return cached
        shape = self.tensor_shapes[site]
        physical_group = self.physical_groups[site]
        entries = []
        spectator_shape = tuple(self.dims[index] for index in physical_group[1:])
        for left, q_left in enumerate(self.bond_qns[site]):
            for owned, q_owned in enumerate(self.local_qns[site]):
                needed = self._add_charge(q_left, q_owned)
                for right, q_right in enumerate(self.bond_qns[site + 1]):
                    if q_right != needed:
                        continue
                    for spectators in np.ndindex(*spectator_shape):
                        physical = (owned, *spectators)
                        coordinate = (left, right, *physical)
                        entries.append(
                            (
                                int(np.ravel_multi_index(coordinate, shape)),
                                left,
                                right,
                                physical,
                            )
                        )
        result = tuple(entries)
        self._tensor_support_cache[site] = result
        return result

    def _compiled_route_builders(self):
        cached = self._compiled_route_builders_cache
        if cached is not None:
            return cached
        try:
            cpp = importlib.import_module("pyqed.mps.cpp_davidson")
            cached = (
                getattr(cpp, "build_packed_advance_routes", None),
                getattr(cpp, "build_packed_hole_routes", None),
            )
        except Exception:
            cached = (None, None)
        self._compiled_route_builders_cache = cached
        return cached

    @staticmethod
    def _physical_coordinate_matrix(values, sites, size):
        sites = tuple(sites)
        if not sites:
            return np.empty((int(size), 0), dtype=np.int32)
        return np.ascontiguousarray(
            np.stack([values[site] for site in sites], axis=1),
            dtype=np.int32,
        )

    def _tensor_support_arrays(self, site):
        site = int(site)
        cached = self._tensor_support_array_cache.get(site)
        if cached is not None:
            return cached
        entries = self._tensor_support(site)
        cached = (
            np.asarray([entry[0] for entry in entries], dtype=np.int32),
            np.asarray([entry[1] for entry in entries], dtype=np.int32),
            np.asarray([entry[2] for entry in entries], dtype=np.int32),
            np.asarray([entry[3] for entry in entries], dtype=np.int32),
        )
        self._tensor_support_array_cache[site] = cached
        return cached

    def _packed_advance_plan(
        self,
        direction,
        site,
        left_channel,
        right_channel,
        *,
        source_values=None,
        tensor_values=None,
        target_size=None,
        real_output=False,
    ):
        """Compile exact sparse routes between two packed U(1) messages."""

        key = (str(direction), int(site), int(left_channel), int(right_channel))
        contract = source_values is not None or tensor_values is not None
        if contract and (source_values is None or tensor_values is None):
            raise TypeError("source_values and tensor_values must be supplied together.")
        if not contract:
            cached = self._cached_packed_plan("advance", key)
            if cached is not None:
                return cached
        direction, site, left_channel, right_channel = key
        if direction == "left":
            source_cut, source_channel = site, left_channel
            target_cut, target_channel = site + 1, right_channel
            source_bond_axis = 0
            target_bond_axis = 1
        elif direction == "right":
            source_cut, source_channel = site + 1, right_channel
            target_cut, target_channel = site, left_channel
            source_bond_axis = 1
            target_bond_axis = 0
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        source_bra_bond, source_ket_bond, source_bra, source_ket = (
            self._expanded_storage_coordinates(source_cut, source_channel)
        )
        target_pairs = {
            pair: index
            for index, pair in enumerate(self._bond_pairs[target_cut][target_channel])
        }
        target_inverse = self._storage_inverse(target_cut, target_channel)
        target_shape = self.block_shape(target_cut, target_channel)
        target_paired = set(self.paired_sites(target_cut, target_channel))
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        source_frontier = set(self.frontier_sites[source_cut])
        tensor_entries = self._tensor_support(site)

        entry_groups = {}
        overlap = tuple(index for index in local_sites if index in source_frontier)
        advance_builder, _hole_builder = self._compiled_route_builders()
        if advance_builder is not None:
            source_sites = self.frontier_sites[source_cut]
            target_sites = self.frontier_sites[target_cut]
            tensor_flat, tensor_left, tensor_right, tensor_physical = (
                self._tensor_support_arrays(site)
            )
            pair_lookup = np.full(
                (
                    self.virtual_bonds[target_cut],
                    self.virtual_bonds[target_cut],
                ),
                -1,
                dtype=np.int32,
            )
            for pair_index, (bra_bond, ket_bond) in enumerate(
                self._bond_pairs[target_cut][target_channel]
            ):
                pair_lookup[bra_bond, ket_bond] = pair_index
            compiled = advance_builder(
                direction == "left",
                np.asarray(source_bra_bond, dtype=np.int32),
                np.asarray(source_ket_bond, dtype=np.int32),
                np.asarray(source_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    source_bra, source_sites, len(source_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    source_ket, source_sites, len(source_bra_bond)
                ),
                tensor_flat,
                tensor_left,
                tensor_right,
                np.asarray(local_sites, dtype=np.int64),
                np.ascontiguousarray(tensor_physical, dtype=np.int32),
                np.asarray(overlap, dtype=np.int64),
                np.asarray(target_sites, dtype=np.int64),
                np.asarray(
                    [site in target_paired for site in target_sites],
                    dtype=np.uint8,
                ),
                np.asarray(self.dims, dtype=np.int64),
                pair_lookup,
                np.asarray(target_shape, dtype=np.int64),
                np.asarray(target_inverse, dtype=np.int32),
                np.asarray(
                    self.mpo_tensors[site][left_channel, right_channel],
                    dtype=np.complex128,
                ),
                source_values,
                tensor_values,
                0 if target_size is None else int(target_size),
                bool(real_output),
            )
            if contract:
                self.packed_route_backend = "cpp-fused"
                return np.asarray(compiled)
            result = tuple(
                np.asarray(value, dtype=np.int32)
                for value in compiled[:4]
            ) + (
                np.asarray(
                    np.asarray(compiled[4]).real
                    if self.dtype.kind != "c"
                    else compiled[4],
                    dtype=self.dtype,
                ),
            )
            self.packed_route_backend = "cpp"
            return self._cache_packed_plan("advance", key, result)
        for flat, left_bond, right_bond, physical in tensor_entries:
            bond = left_bond if source_bond_axis == 0 else right_bond
            values = tuple(physical[local_position[index]] for index in overlap)
            entry_groups.setdefault((bond, values), []).append(
                (flat, left_bond, right_bond, physical)
            )

        source_indices = []
        target_indices = []
        bra_indices = []
        ket_indices = []
        operator_values = []
        operator = self.mpo_tensors[site][left_channel, right_channel]
        for source_index in range(len(source_bra_bond)):
            bra_key = (
                int(source_bra_bond[source_index]),
                tuple(int(source_bra[index][source_index]) for index in overlap),
            )
            ket_key = (
                int(source_ket_bond[source_index]),
                tuple(int(source_ket[index][source_index]) for index in overlap),
            )
            for bra_entry in entry_groups.get(bra_key, ()):
                bra_flat, bra_left, bra_right, bra_physical = bra_entry
                for ket_entry in entry_groups.get(ket_key, ()):
                    ket_flat, ket_left, ket_right, ket_physical = ket_entry
                    coefficient = operator[
                        bra_physical[0], ket_physical[0]
                    ]
                    if coefficient == 0:
                        continue
                    target_pair = (
                        (bra_left, ket_left)
                        if target_bond_axis == 0
                        else (bra_right, ket_right)
                    )
                    pair_index = target_pairs.get(target_pair)
                    if pair_index is None:
                        continue
                    target_coordinate = [pair_index]
                    compatible = True
                    for physical_site in self.frontier_sites[target_cut]:
                        if physical_site in local_position:
                            bra_value = bra_physical[local_position[physical_site]]
                            ket_value = ket_physical[local_position[physical_site]]
                        else:
                            bra_value = int(source_bra[physical_site][source_index])
                            ket_value = int(source_ket[physical_site][source_index])
                        target_coordinate.append(bra_value)
                        if physical_site in target_paired:
                            target_coordinate.append(ket_value)
                        elif bra_value != ket_value:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    logical_target = int(
                        np.ravel_multi_index(tuple(target_coordinate), target_shape)
                    )
                    packed_target = int(target_inverse[logical_target])
                    if packed_target < 0:
                        continue
                    source_indices.append(source_index)
                    target_indices.append(packed_target)
                    bra_indices.append(bra_flat)
                    ket_indices.append(ket_flat)
                    operator_values.append(coefficient)

        result = tuple(
            np.asarray(values, dtype=dtype)
            for values, dtype in (
                (source_indices, np.int32),
                (target_indices, np.int32),
                (bra_indices, np.int32),
                (ket_indices, np.int32),
                (operator_values, self.dtype),
            )
        )
        return self._cache_packed_plan("advance", key, result)

    def _packed_advance(self, direction, message, tensors, site):
        """Advance a charge-resolved message without logical dense tensors."""

        site = int(site)
        source_cut = site if direction == "left" else site + 1
        target_cut = site + 1 if direction == "left" else site
        message = self._validated_message(message, source_cut)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        result = [np.zeros(0, dtype=dtype) for _ in range(self.mpo_bonds[target_cut])]
        for channel in self._active_channels[target_cut]:
            result[channel] = np.zeros(self.storage_shape(target_cut, channel), dtype=dtype)
        groups = (
            self._left_transition_groups[site]
            if direction == "left"
            else self._right_transition_groups[site]
        )
        tensor_flat = tensor.reshape(-1)
        fused_advance = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[0] is not None
        )
        self.last_packed_advance_peak_bytes = 0
        for transitions in groups:
            for left_channel, right_channel in transitions:
                source_channel = left_channel if direction == "left" else right_channel
                target_channel = right_channel if direction == "left" else left_channel
                if fused_advance:
                    contribution = self._packed_advance_plan(
                        direction,
                        site,
                        left_channel,
                        right_channel,
                        source_values=np.asarray(
                            message.blocks[source_channel]
                        ).reshape(-1),
                        tensor_values=tensor_flat,
                        target_size=result[target_channel].size,
                        real_output=np.dtype(dtype).kind != "c",
                    )
                    result[target_channel].reshape(-1)[:] += contribution
                    continue
                (
                    source_index,
                    target_index,
                    bra_index,
                    ket_index,
                    coefficient,
                ) = self._packed_advance_plan(
                    direction, site, left_channel, right_channel
                )
                self.last_packed_advance_peak_bytes = max(
                    self.last_packed_advance_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                source_index,
                                target_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if source_index.size == 0:
                    continue
                values = (
                    np.asarray(message.blocks[source_channel]).reshape(-1)[source_index]
                    * np.conj(tensor_flat[bra_index])
                    * tensor_flat[ket_index]
                    * coefficient
                )
                np.add.at(result[target_channel].reshape(-1), target_index, values)
        return BlockFrontierMessage(target_cut, tuple(result))

    def _packed_hole_plan(
        self,
        site,
        left_channel,
        right_channel,
        *,
        left_values=None,
        right_values=None,
        real_output=False,
        dense_output=None,
    ):
        """Compile sparse routes for one charge-resolved local operator channel."""

        key = (int(site), int(left_channel), int(right_channel))
        bind = left_values is not None or right_values is not None
        if bind and (left_values is None or right_values is None):
            raise TypeError("left_values and right_values must be supplied together.")
        if not bind:
            cached = self._cached_packed_plan("hole", key)
            if cached is not None:
                return cached
        site, left_channel, right_channel = key
        left_bra_bond, left_ket_bond, left_bra, left_ket = (
            self._expanded_storage_coordinates(site, left_channel)
        )
        right_bra_bond, right_ket_bond, right_bra, right_ket = (
            self._expanded_storage_coordinates(site + 1, right_channel)
        )
        left_frontier = set(self.frontier_sites[site])
        right_frontier = set(self.frontier_sites[site + 1])
        common = tuple(sorted(left_frontier & right_frontier))
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        constrained = tuple(
            physical_site
            for physical_site in local_sites
            if physical_site in left_frontier or physical_site in right_frontier
        )
        _advance_builder, hole_builder = self._compiled_route_builders()
        if hole_builder is not None:
            left_sites = self.frontier_sites[site]
            right_sites = self.frontier_sites[site + 1]
            tensor_flat, tensor_left, tensor_right, tensor_physical = (
                self._tensor_support_arrays(site)
            )
            compiled = hole_builder(
                np.asarray(left_bra_bond, dtype=np.int32),
                np.asarray(left_ket_bond, dtype=np.int32),
                np.asarray(left_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    left_bra, left_sites, len(left_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    left_ket, left_sites, len(left_bra_bond)
                ),
                np.asarray(right_bra_bond, dtype=np.int32),
                np.asarray(right_ket_bond, dtype=np.int32),
                np.asarray(right_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    right_bra, right_sites, len(right_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    right_ket, right_sites, len(right_bra_bond)
                ),
                tensor_flat,
                tensor_left,
                tensor_right,
                np.asarray(local_sites, dtype=np.int64),
                np.ascontiguousarray(tensor_physical, dtype=np.int32),
                np.asarray(common, dtype=np.int64),
                np.asarray(constrained, dtype=np.int64),
                np.asarray(self.dims, dtype=np.int64),
                int(self.virtual_bonds[site + 1]),
                np.asarray(
                    self.mpo_tensors[site][left_channel, right_channel],
                    dtype=np.complex128,
                ),
                left_values,
                right_values,
                bool(real_output),
                dense_output,
            )
            if bind:
                self.packed_route_backend = "cpp-fused"
                if dense_output is not None:
                    return np.asarray(compiled)
                return tuple(np.asarray(value) for value in compiled)
            result = tuple(
                np.asarray(value, dtype=np.int32)
                for value in compiled[:4]
            ) + (
                np.asarray(
                    np.asarray(compiled[4]).real
                    if self.dtype.kind != "c"
                    else compiled[4],
                    dtype=self.dtype,
                ),
            )
            self.packed_route_backend = "cpp"
            return self._cache_packed_plan("hole", key, result)
        right_groups = {}
        for right_index in range(len(right_bra_bond)):
            common_key = tuple(
                (int(right_bra[index][right_index]), int(right_ket[index][right_index]))
                for index in common
            )
            right_groups.setdefault(common_key, []).append(right_index)
        tensor_groups = {}
        for flat, left_bond, right_bond, physical in self._tensor_support(site):
            values = tuple(physical[local_position[index]] for index in constrained)
            tensor_groups.setdefault((left_bond, right_bond, values), []).append(
                (flat, physical)
            )

        left_indices = []
        right_indices = []
        bra_indices = []
        ket_indices = []
        operator_values = []
        operator = self.mpo_tensors[site][left_channel, right_channel]
        for left_index in range(len(left_bra_bond)):
            common_key = tuple(
                (int(left_bra[index][left_index]), int(left_ket[index][left_index]))
                for index in common
            )
            for right_index in right_groups.get(common_key, ()):
                bra_values = []
                ket_values = []
                compatible = True
                for physical_site in constrained:
                    bra_value = ket_value = None
                    if physical_site in left_frontier:
                        bra_value = int(left_bra[physical_site][left_index])
                        ket_value = int(left_ket[physical_site][left_index])
                    if physical_site in right_frontier:
                        right_bra_value = int(right_bra[physical_site][right_index])
                        right_ket_value = int(right_ket[physical_site][right_index])
                        if bra_value is not None and (
                            bra_value != right_bra_value or ket_value != right_ket_value
                        ):
                            compatible = False
                            break
                        bra_value = right_bra_value
                        ket_value = right_ket_value
                    bra_values.append(bra_value)
                    ket_values.append(ket_value)
                if not compatible:
                    continue
                bra_key = (
                    int(left_bra_bond[left_index]),
                    int(right_bra_bond[right_index]),
                    tuple(bra_values),
                )
                ket_key = (
                    int(left_ket_bond[left_index]),
                    int(right_ket_bond[right_index]),
                    tuple(ket_values),
                )
                for bra_flat, bra_physical in tensor_groups.get(bra_key, ()):
                    for ket_flat, ket_physical in tensor_groups.get(ket_key, ()):
                        coefficient = operator[bra_physical[0], ket_physical[0]]
                        if coefficient == 0:
                            continue
                        left_indices.append(left_index)
                        right_indices.append(right_index)
                        bra_indices.append(bra_flat)
                        ket_indices.append(ket_flat)
                        operator_values.append(coefficient)

        result = tuple(
            np.asarray(values, dtype=dtype)
            for values, dtype in (
                (left_indices, np.int32),
                (right_indices, np.int32),
                (bra_indices, np.int32),
                (ket_indices, np.int32),
                (operator_values, self.dtype),
            )
        )
        return self._cache_packed_plan("hole", key, result)

    def _packed_hole_action(self, site, left, right, vector):
        """Apply a one-site hole directly from packed U(1) messages."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(-1)
        size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
        if vector.size != size:
            raise ValueError(f"vector must contain {size} entries.")
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(size, dtype=dtype)
        self.last_packed_hole_peak_bytes = 0
        for transitions in self._hole_transition_groups[site]:
            for left_channel, right_channel in transitions:
                left_index, right_index, bra_index, ket_index, coefficient = (
                    self._packed_hole_plan(site, left_channel, right_channel)
                )
                self.last_packed_hole_peak_bytes = max(
                    self.last_packed_hole_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                left_index,
                                right_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if left_index.size == 0:
                    continue
                values = (
                    np.asarray(left.blocks[left_channel]).reshape(-1)[left_index]
                    * np.asarray(right.blocks[right_channel]).reshape(-1)[right_index]
                    * coefficient
                    * vector[ket_index]
                )
                np.add.at(result, bra_index, values)
        return result

    def _packed_fixed_hole_routes(self, site, left, right, transitions=None):
        """Bind packed messages and return sparse ``(bra, ket, weight)`` routes."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        transition_groups = (
            self._hole_transition_groups[site]
            if transitions is None
            else (tuple(transitions),)
        )
        routes = []
        fused_hole = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[1] is not None
        )
        self.last_packed_hole_peak_bytes = 0
        for group in transition_groups:
            for left_channel, right_channel in group:
                if fused_hole:
                    left_values = np.asarray(
                        left.blocks[left_channel]
                    ).reshape(-1)
                    right_values = np.asarray(
                        right.blocks[right_channel]
                    ).reshape(-1)
                    bra_index, ket_index, weight = self._packed_hole_plan(
                        site,
                        left_channel,
                        right_channel,
                        left_values=left_values,
                        right_values=right_values,
                        real_output=np.result_type(
                            self.dtype,
                            left_values.dtype,
                            right_values.dtype,
                        ).kind
                        != "c",
                    )
                    if bra_index.size:
                        routes.append((bra_index, ket_index, weight))
                    continue
                left_index, right_index, bra_index, ket_index, coefficient = (
                    self._packed_hole_plan(site, left_channel, right_channel)
                )
                self.last_packed_hole_peak_bytes = max(
                    self.last_packed_hole_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                left_index,
                                right_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if left_index.size == 0:
                    continue
                weight = (
                    np.asarray(left.blocks[left_channel]).reshape(-1)[left_index]
                    * np.asarray(right.blocks[right_channel]).reshape(-1)[right_index]
                    * coefficient
                )
                routes.append((bra_index, ket_index, weight))
        return tuple(routes)

    def _prepare_packed_hole_action(self, site, left, right):
        """Bind a sector-native U(1) hole without dense frontier workspaces."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        shape = self.tensor_shapes[site]
        size = int(np.prod(shape, dtype=np.int64))
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[block.dtype for block in left.blocks if block is not None],
                *[block.dtype for block in right.blocks if block is not None],
            )
        )
        build_dtype = np.dtype(
            np.complex128 if native_dtype.kind == "c" else np.float64
        )
        dense_bytes = size * size * build_dtype.itemsize
        fused_dense = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[1] is not None
            and dense_bytes <= self._packed_dense_action_limit_bytes
        )
        if fused_dense:
            matrix = np.zeros((size, size), dtype=build_dtype)
            for group in self._hole_transition_groups[site]:
                for left_channel, right_channel in group:
                    self._packed_hole_plan(
                        site,
                        left_channel,
                        right_channel,
                        left_values=np.asarray(
                            left.blocks[left_channel]
                        ).reshape(-1),
                        right_values=np.asarray(
                            right.blocks[right_channel]
                        ).reshape(-1),
                        real_output=build_dtype.kind != "c",
                        dense_output=matrix,
                    )
            dtype = self.compute_dtype
            if native_dtype.kind == "c" and dtype.kind != "c":
                dtype = np.dtype(
                    np.complex64 if dtype.itemsize <= 4 else np.complex128
                )
            matrix = np.asarray(matrix, dtype=dtype)

            def dense_action(vector):
                vector = np.asarray(vector)
                if vector.size != size:
                    raise ValueError(f"vector must contain {size} entries.")
                return matrix @ vector.reshape(-1)

            def dense_actions(vectors):
                vectors = np.asarray(vectors)
                if vectors.ndim != 2 or vectors.shape[1] != size:
                    raise ValueError(f"vectors must have shape (batch, {size}).")
                return vectors @ matrix.T

            dense_action.many = dense_actions
            dense_action.backend = "packed-u1-dense-fused"
            dense_action.packed_route_count = 0
            dense_action.packed_route_bytes = int(matrix.nbytes)
            dense_action.dense_action_bytes = int(matrix.nbytes)
            if self._compute_dtype_requested and dtype != native_dtype:
                dense_action.verify = lambda vector: self._packed_hole_action(
                    site, left, right, vector
                )
            return dense_action

        routes = self._packed_fixed_hole_routes(site, left, right)
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
        )
        dtype = self.compute_dtype
        if native_dtype.kind == "c" and dtype.kind != "c":
            dtype = np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)
        bound = tuple(
            (bra, ket, np.asarray(weight, dtype=dtype))
            for bra, ket, weight in routes
        )
        route_count = int(sum(len(route[0]) for route in bound))
        if route_count:
            bra = np.concatenate([route[0] for route in bound])
            ket = np.concatenate([route[1] for route in bound])
            weight = np.concatenate([route[2] for route in bound])
            sparse_action = coo_matrix(
                (weight, (bra, ket)),
                shape=(size, size),
                dtype=dtype,
            ).tocsr()
            sparse_action.sum_duplicates()
            sparse_action.sort_indices()
        else:
            sparse_action = coo_matrix((size, size), dtype=dtype).tocsr()
        sparse_bytes = int(
            sparse_action.data.nbytes
            + sparse_action.indices.nbytes
            + sparse_action.indptr.nbytes
        )

        def action_dtype_for(value):
            value_dtype = np.asarray(value).dtype
            if value_dtype.kind == "c" and dtype.kind != "c":
                return np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)
            return dtype

        def action(vector):
            vector = np.asarray(vector)
            if vector.size != size:
                raise ValueError(f"vector must contain {size} entries.")
            action_dtype = action_dtype_for(vector)
            flat = np.asarray(vector, dtype=action_dtype).reshape(-1)
            return np.asarray(sparse_action @ flat).reshape(-1)

        def actions(vectors):
            vectors = np.asarray(vectors)
            if vectors.ndim != 2 or vectors.shape[1] != size:
                raise ValueError(f"vectors must have shape (batch, {size}).")
            action_dtype = action_dtype_for(vectors)
            flat = np.asarray(vectors, dtype=action_dtype)
            return np.asarray(sparse_action @ flat.T).T

        action.many = actions
        action.backend = "packed-u1-csr"
        action.packed_route_count = route_count
        action.packed_route_bytes = sparse_bytes
        if self._compute_dtype_requested and dtype != native_dtype:
            action.verify = lambda vector: self._packed_hole_action(
                site, left, right, vector
            )
        return action

    def message_elements(self, cut):
        cut = int(cut)
        cached = self._message_elements_cache.get(cut)
        if cached is not None:
            return cached
        cached = sum(
            int(np.prod(self.block_shape(cut, channel)))
            for channel in self._active_channels[cut]
        )
        self._message_elements_cache[cut] = cached
        return cached

    def logical_message_elements(self, cut):
        return sum(
            int(np.prod(self.block_shape(cut, channel)))
            for channel in self._active_channels[int(cut)]
        )

    @property
    def physical_packing_ratio(self):
        logical = sum(self.logical_message_elements(cut) for cut in range(self.nsites + 1))
        return self.total_message_elements / max(logical, 1)

    def dense_message_elements(self, cut):
        cut = int(cut)
        return int(
            self.virtual_bonds[cut] ** 2
            * self.mpo_bonds[cut]
            * np.prod(
                [self.dims[site] ** 2 for site in self.frontier_sites[cut]],
                dtype=np.int64,
            )
        )

    @property
    def peak_message_elements(self):
        return max(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def dense_peak_message_elements(self):
        return max(self.dense_message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def total_message_elements(self):
        return sum(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def dense_total_message_elements(self):
        return sum(self.dense_message_elements(cut) for cut in range(self.nsites + 1))

    def _new_label(self):
        label = self._next_auxiliary_label
        self._next_auxiliary_label += 1
        return label

    def _block_labels_and_copies(self, cut, channel):
        copies = []
        if self.charge_resolved:
            pair_label = self._new_label()
            labels = [pair_label]
            copies.append(
                (
                    self._pair_selector(cut, channel),
                    (pair_label, self.bra_bonds[cut], self.ket_bonds[cut]),
                )
            )
        else:
            labels = [self.bra_bonds[cut], self.ket_bonds[cut]]
        paired = set(self.paired_sites(cut, channel))
        for site in self.frontier_sites[cut]:
            if site in paired:
                labels.extend((self.bra_physical[site], self.ket_physical[site]))
            else:
                shared = self._new_label()
                labels.append(shared)
                copies.append(
                    (
                        self._copies[self.dims[site]],
                        (
                            shared,
                            self.bra_physical[site],
                            self.ket_physical[site],
                        ),
                    )
                )
        return tuple(labels), tuple(copies)

    def _tensor_labels(self, site, *, bra):
        bonds = self.bra_bonds if bra else self.ket_bonds
        physical = self.bra_physical if bra else self.ket_physical
        return (
            bonds[site],
            bonds[site + 1],
            *(physical[index] for index in self.physical_groups[site]),
        )

    def _local_tt_core_shapes(self, site):
        shape = self.tensor_shapes[int(site)]
        dimensions = (shape[0], *shape[2:], shape[1])
        result = []
        left_rank = 1
        remaining = int(np.prod(dimensions, dtype=np.int64))
        for dimension in dimensions[:-1]:
            remaining //= dimension
            rank = min(left_rank * dimension, remaining)
            result.append((left_rank, dimension, rank))
            left_rank = rank
        result.append((left_rank, dimensions[-1], 1))
        return tuple(result)

    def _local_tt_core_labels(self, site, *, bra):
        key = (int(site), bool(bra))
        cached = self._local_tt_labels.get(key)
        if cached is not None:
            return cached
        tensor_labels = self._tensor_labels(site, bra=bra)
        axis_labels = (tensor_labels[0], *tensor_labels[2:], tensor_labels[1])
        bonds = tuple(self._new_label() for _ in range(len(axis_labels) + 1))
        cached = tuple(
            (bonds[axis], label, bonds[axis + 1])
            for axis, label in enumerate(axis_labels)
        )
        self._local_tt_labels[key] = cached
        return cached

    @staticmethod
    def _compile_copy_class_plan(operands, output_labels, copies):
        """Compile raw einsum labels and copy tensors into equality classes."""
        operands = tuple(
            (
                tuple(int(dimension) for dimension in shape),
                tuple(int(label) for label in labels),
            )
            for shape, labels in operands
        )
        output_labels = tuple(int(label) for label in output_labels)
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("raw copy-contraction output labels must be unique.")
        dimensions = {}
        ordered_labels = []

        def register(label, dimension):
            label = int(label)
            dimension = int(dimension)
            if label not in dimensions:
                dimensions[label] = dimension
                ordered_labels.append(label)
            elif dimensions[label] != dimension:
                raise ValueError(
                    "copy-contraction labels have inconsistent dimensions."
                )

        for shape, labels in operands:
            if len(shape) != len(labels):
                raise ValueError(
                    "copy-contraction operand shapes and labels must match."
                )
            for dimension, label in zip(shape, labels):
                register(label, dimension)

        parents = {}

        def find(label):
            parents.setdefault(label, label)
            while parents[label] != label:
                parents[label] = parents[parents[label]]
                label = parents[label]
            return label

        def union(left, right):
            left = find(left)
            right = find(right)
            if left != right:
                parents[right] = left

        for value, labels in copies:
            value = np.asarray(value)
            dimension = int(value.shape[0])
            if value.shape != (dimension, dimension, dimension):
                raise ValueError("copy factors must be cubic three-way tensors.")
            labels = tuple(int(label) for label in labels)
            if len(labels) != 3:
                raise ValueError("copy factors must carry three labels.")
            for label in labels:
                register(label, dimension)
                find(label)
            union(labels[0], labels[1])
            union(labels[0], labels[2])
        for labels in (
            *(labels for _shape, labels in operands),
            output_labels,
        ):
            for label in labels:
                if label not in dimensions:
                    raise ValueError(
                        "every output label needs an operand or copy dimension."
                    )
                find(label)

        root_to_class = {}
        class_dimensions = []
        for label in ordered_labels:
            root = find(label)
            if root not in root_to_class:
                root_to_class[root] = len(class_dimensions)
                class_dimensions.append(dimensions[label])
            elif class_dimensions[root_to_class[root]] != dimensions[label]:
                raise ValueError(
                    "copy-connected labels have inconsistent dimensions."
                )

        def classes(labels):
            return np.asarray(
                [root_to_class[find(label)] for label in labels],
                dtype=np.intp,
            )

        operand_classes = tuple(classes(labels) for _shape, labels in operands)
        output_classes = classes(output_labels)
        class_dimensions = np.asarray(class_dimensions, dtype=np.intp)
        operation_count = int(np.prod(class_dimensions, dtype=object))
        output_class_set = set(int(value) for value in output_classes)
        contracted_count = int(
            np.prod(
                tuple(
                    dimension
                    for cls, dimension in enumerate(class_dimensions)
                    if cls not in output_class_set
                ),
                dtype=object,
            )
        )
        return {
            "operand_classes": operand_classes,
            "output_classes": output_classes,
            "class_dimensions": class_dimensions,
            "operation_count": operation_count,
            "contracted_count": contracted_count,
            "workspace_bytes": (
                len(operands)
                * np.dtype(np.intp).itemsize
                * contracted_count
            ),
        }

    def _advance_copy_plan(
        self,
        direction,
        site,
        left_channel,
        right_channel,
        *,
        batch_size,
        identity=False,
    ):
        """Plan a dense native frontier-message topology group."""
        if (
            self.charge_resolved
            or not _copy_einsum_native_available()
            or (not identity and self.local_backend != "dense")
        ):
            return None
        site = int(site)
        left_channel = int(left_channel)
        right_channel = int(right_channel)
        batch_size = int(batch_size)
        identity = bool(identity)
        key = (
            "identity" if identity else str(direction),
            site,
            left_channel,
            right_channel,
            batch_size,
        )
        cached = self._copy_advance_plans.get(key)
        if cached is not None:
            return cached
        if direction == "left":
            if identity:
                raise ValueError("identity advancement is right-directed.")
            source_cut, source_channel = site, left_channel
            target_cut, target_channel = site + 1, right_channel
        elif direction == "right":
            source_cut, source_channel = site + 1, right_channel
            target_cut, target_channel = site, left_channel
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        source_labels, source_copies = self._block_labels_and_copies(
            source_cut,
            source_channel,
        )
        target_labels, target_copies = self._block_labels_and_copies(
            target_cut,
            target_channel,
        )
        if identity:
            aliases = {
                self.bra_bonds[source_cut]: self.bra_bonds[target_cut],
                self.ket_bonds[source_cut]: self.ket_bonds[target_cut],
            }
            source_labels = tuple(
                aliases.get(label, label) for label in source_labels
            )
        source_shape = self.block_shape(source_cut, source_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (
            self.bra_physical[site],
            self.ket_physical[site],
        )
        if batch_size > 1:
            batch_label = self._new_label()
            source_shape = (batch_size, *source_shape)
            source_labels = (batch_label, *source_labels)
            operator_shape = (batch_size, *operator_shape)
            operator_labels = (batch_label, *operator_labels)
        operands = [(source_shape, source_labels)]
        if not identity:
            operands.extend(
                (
                    (
                        self.tensor_shapes[site],
                        self._tensor_labels(site, bra=True),
                    ),
                    (
                        self.tensor_shapes[site],
                        self._tensor_labels(site, bra=False),
                    ),
                )
            )
        operands.append((operator_shape, operator_labels))
        copies = source_copies + target_copies
        cached = self._compile_copy_class_plan(
            operands,
            target_labels,
            copies,
        )
        cached["copy_count"] = len(copies)
        self._copy_advance_plans[key] = cached
        return cached

    @staticmethod
    def _evaluate_advance_copy(plan, operands):
        return _contract_class_einsum(
            operands,
            plan["operand_classes"],
            plan["output_classes"],
            plan["class_dimensions"],
        )

    def _native_advance_enabled(self, copy_backend, dtype, *, identity=False):
        copy_backend = str(copy_backend).lower().replace("-", "_")
        if copy_backend not in {"auto", "native", "python"}:
            raise ValueError(
                "copy_backend must be 'auto', 'native', or 'python'."
            )
        if copy_backend == "python":
            return copy_backend, False
        if not _copy_einsum_native_available():
            if copy_backend == "native":
                raise RuntimeError(
                    "copy_backend='native' requires the optional copy-einsum "
                    "extension."
                )
            return copy_backend, False
        supported = not self.charge_resolved and (
            identity or self.local_backend == "dense"
        )
        if not supported:
            if copy_backend == "native":
                raise ValueError(
                    "copy_backend='native' requires dense, explicit virtual "
                    "bra/ket frontier blocks."
                )
            return copy_backend, False
        native_dtype = np.dtype(dtype) in {
            np.dtype(np.float64),
            np.dtype(np.complex128),
        }
        if not native_dtype:
            if copy_backend == "native":
                raise TypeError(
                    "copy_backend='native' supports only float64 and "
                    "complex128."
                )
            return copy_backend, False
        return copy_backend, True

    @staticmethod
    def _auto_advance_copy_plan(copy_backend, plan):
        if (
            copy_backend == "auto"
            and plan is not None
            and (
                plan["copy_count"] == 0
                or plan["operation_count"]
                > _COPY_ADVANCE_AUTO_MAX_OPERATIONS
                or plan["workspace_bytes"]
                > _COPY_ADVANCE_AUTO_MAX_WORKSPACE_BYTES
            )
        ):
            return None
        return plan

    def _advance_expression(
        self,
        direction,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
        core_shapes=None,
    ):
        batch_size = int(batch_size)
        site = int(site)
        if direction == "left":
            source_cut, source_channel = site, int(left_channel)
            target_cut, target_channel = site + 1, int(right_channel)
        elif direction == "right":
            source_cut, source_channel = site + 1, int(right_channel)
            target_cut, target_channel = site, int(left_channel)
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        source_labels, source_copies = self._block_labels_and_copies(
            source_cut, source_channel
        )
        target_labels, target_copies = self._block_labels_and_copies(
            target_cut, target_channel
        )
        copies = source_copies + target_copies
        source_shape = self.block_shape(source_cut, source_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (self.bra_physical[site], self.ket_physical[site])
        if batch_size > 1:
            batch_label = self._new_label()
            source_shape = (batch_size,) + source_shape
            source_labels = (batch_label,) + source_labels
            operator_shape = (batch_size,) + operator_shape
            operator_labels = (batch_label,) + operator_labels
        key = (
            "advance",
            direction,
            site,
            tuple(self.paired_sites(source_cut, source_channel)),
            tuple(self.paired_sites(target_cut, target_channel)),
            tuple(source_shape),
            tuple(self.block_shape(target_cut, target_channel)),
            batch_size,
            tuple(core_shapes) if core_shapes is not None else None,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached, tuple(value for value, _labels in copies)
        arguments = [
            source_shape,
            source_labels,
        ]
        if self.local_backend == "dense":
            arguments.extend(
                (
                    self.tensor_shapes[site],
                    self._tensor_labels(site, bra=True),
                    self.tensor_shapes[site],
                    self._tensor_labels(site, bra=False),
                )
            )
        else:
            if core_shapes is None:
                core_shapes = self._local_tt_core_shapes(site)
            for shape, labels in zip(
                core_shapes,
                self._local_tt_core_labels(site, bra=True),
            ):
                arguments.extend((shape, labels))
            for shape, labels in zip(
                core_shapes,
                self._local_tt_core_labels(site, bra=False),
            ):
                arguments.extend((shape, labels))
        arguments.extend((operator_shape, operator_labels))
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            target_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression, tuple(value for value, _labels in copies)

    def _advance_right_identity_expression(
        self,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
    ):
        """Contract a square virtual identity without materializing it."""
        site = int(site)
        batch_size = int(batch_size)
        source_cut = site + 1
        source_channel = int(right_channel)
        target_cut = site
        target_channel = int(left_channel)
        source_labels, source_copies = self._block_labels_and_copies(
            source_cut,
            source_channel,
        )
        target_labels, target_copies = self._block_labels_and_copies(
            target_cut,
            target_channel,
        )
        aliases = {
            self.bra_bonds[source_cut]: self.bra_bonds[target_cut],
            self.ket_bonds[source_cut]: self.ket_bonds[target_cut],
        }
        source_labels = tuple(aliases.get(label, label) for label in source_labels)
        copies = source_copies + target_copies
        source_shape = self.block_shape(source_cut, source_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (
            self.bra_physical[site],
            self.ket_physical[site],
        )
        if batch_size > 1:
            batch_label = self._new_label()
            source_shape = (batch_size,) + source_shape
            source_labels = (batch_label,) + source_labels
            operator_shape = (batch_size,) + operator_shape
            operator_labels = (batch_label,) + operator_labels
        key = (
            "advance_right_identity",
            site,
            tuple(self.paired_sites(source_cut, source_channel)),
            tuple(self.paired_sites(target_cut, target_channel)),
            tuple(source_shape),
            tuple(self.block_shape(target_cut, target_channel)),
            batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached, tuple(value for value, _labels in copies)
        arguments = [
            source_shape,
            source_labels,
            operator_shape,
            operator_labels,
        ]
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            target_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression, tuple(value for value, _labels in copies)

    def _validated_message(self, message, cut):
        if not isinstance(message, BlockFrontierMessage):
            raise TypeError("message must be a BlockFrontierMessage.")
        if message.cut != cut:
            raise ValueError(f"message belongs to cut {message.cut}, not cut {cut}.")
        if len(message.blocks) != self.mpo_bonds[cut]:
            raise ValueError("message has the wrong number of MPO-channel blocks.")
        for channel, block in enumerate(message.blocks):
            if channel not in self._active_channel_sets[cut]:
                if np.size(block) != 0:
                    raise ValueError(
                        f"inactive message block {channel} must be empty."
                    )
                continue
            if np.shape(block) != self.block_shape(cut, channel):
                raise ValueError(f"message block {channel} has the wrong shape.")
        return message

    def expand_virtual_pairs(self, message):
        """Expand a charge-compressed message to explicit bra/ket bonds."""
        message = self._validated_message(message, message.cut)
        if not self.charge_resolved:
            return message
        cut = message.cut
        dimension = self.virtual_bonds[cut]
        blocks = []
        for channel, block in enumerate(message.blocks):
            if channel not in self._active_channel_sets[cut]:
                blocks.append(np.zeros(0, dtype=np.asarray(block).dtype))
                continue
            dense = np.zeros(
                (dimension, dimension, *block.shape[1:]),
                dtype=block.dtype,
            )
            for packed, (bra, ket) in enumerate(self._bond_pairs[cut][channel]):
                dense[bra, ket] = block[packed]
            blocks.append(dense)
        return BlockFrontierMessage(cut, tuple(blocks))

    def left_boundary(self):
        blocks = [np.zeros(0, dtype=self.dtype) for _ in range(self.mpo_bonds[0])]
        blocks[0] = np.ones(self.block_shape(0, 0), dtype=self.dtype)
        return BlockFrontierMessage(
            0,
            tuple(blocks),
        )

    def right_boundary(self):
        blocks = [np.zeros(0, dtype=self.dtype) for _ in range(self.mpo_bonds[-1])]
        blocks[0] = np.ones(self.block_shape(self.nsites, 0), dtype=self.dtype)
        return BlockFrontierMessage(
            self.nsites,
            tuple(blocks),
        )

    def _empty_message(self, cut, dtype):
        result = [np.zeros(0, dtype=dtype) for _ in range(self.mpo_bonds[cut])]
        for channel in self._active_channels[cut]:
            result[channel] = np.zeros(self.block_shape(cut, channel), dtype=dtype)
        return result

    def advance_left(
        self,
        message,
        tensors,
        site,
        *,
        max_workers=1,
        executor=None,
        copy_backend="auto",
    ):
        site = int(site)
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        message = self._validated_message(message, site)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        copy_backend, use_native_copies = self._native_advance_enabled(
            copy_backend,
            dtype,
        )
        result = self._empty_message(site + 1, dtype)
        local_cores = (
            self._factor_local_tensor(site, tensor)
            if self.local_backend == "tensor_train"
            else None
        )
        local_factors = (
            (tensor.conj(), tensor)
            if local_cores is None
            else tuple(core.conj() for core in local_cores) + local_cores
        )
        prepared = []
        for transitions in self._left_transition_groups[site]:
            left, right = transitions[0]
            copy_plan = (
                self._advance_copy_plan(
                    "left",
                    site,
                    left,
                    right,
                    batch_size=len(transitions),
                )
                if use_native_copies
                else None
            )
            copy_plan = self._auto_advance_copy_plan(
                copy_backend,
                copy_plan,
            )
            if copy_plan is None:
                expression, copy_factors = self._advance_expression(
                    "left",
                    site,
                    left,
                    right,
                    batch_size=len(transitions),
                    core_shapes=(
                        tuple(core.shape for core in local_cores)
                        if local_cores is not None
                        else None
                    ),
                )
            else:
                expression, copy_factors = None, ()
            prepared.append(
                (transitions, copy_plan, expression, copy_factors)
            )

        def evaluate(arguments):
            transitions, copy_plan, expression, copy_factors = arguments
            left, right = transitions[0]
            source = message.blocks[left]
            operator = self._operators_for_transitions(site, transitions)
            if len(transitions) > 1:
                source = np.stack(
                    [message.blocks[channel] for channel, _ in transitions]
                )
            if copy_plan is not None:
                contribution = self._evaluate_advance_copy(
                    copy_plan,
                    (source, *local_factors, operator),
                )
            else:
                contribution = expression(
                    source,
                    *local_factors,
                    operator,
                    *copy_factors,
                )
            return right, contribution

        for right, contribution in _ordered_bounded_map(
            evaluate,
            prepared,
            max_workers=max_workers,
            executor=executor,
        ):
            result[right] += contribution
        return self._stored_message(site + 1, result)

    def advance_right(
        self,
        message,
        tensors,
        site,
        *,
        max_workers=1,
        executor=None,
        copy_backend="auto",
    ):
        site = int(site)
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        message = self._validated_message(message, site + 1)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        copy_backend, use_native_copies = self._native_advance_enabled(
            copy_backend,
            dtype,
        )
        result = self._empty_message(site, dtype)
        local_cores = (
            self._factor_local_tensor(site, tensor)
            if self.local_backend == "tensor_train"
            else None
        )
        local_factors = (
            (tensor.conj(), tensor)
            if local_cores is None
            else tuple(core.conj() for core in local_cores) + local_cores
        )
        prepared = []
        for transitions in self._right_transition_groups[site]:
            left, right = transitions[0]
            copy_plan = (
                self._advance_copy_plan(
                    "right",
                    site,
                    left,
                    right,
                    batch_size=len(transitions),
                )
                if use_native_copies
                else None
            )
            copy_plan = self._auto_advance_copy_plan(
                copy_backend,
                copy_plan,
            )
            if copy_plan is None:
                expression, copy_factors = self._advance_expression(
                    "right",
                    site,
                    left,
                    right,
                    batch_size=len(transitions),
                    core_shapes=(
                        tuple(core.shape for core in local_cores)
                        if local_cores is not None
                        else None
                    ),
                )
            else:
                expression, copy_factors = None, ()
            prepared.append(
                (transitions, copy_plan, expression, copy_factors)
            )

        def evaluate(arguments):
            transitions, copy_plan, expression, copy_factors = arguments
            left, right = transitions[0]
            source = message.blocks[right]
            operator = self._operators_for_transitions(site, transitions)
            if len(transitions) > 1:
                source = np.stack(
                    [message.blocks[channel] for _, channel in transitions]
                )
            if copy_plan is not None:
                contribution = self._evaluate_advance_copy(
                    copy_plan,
                    (source, *local_factors, operator),
                )
            else:
                contribution = expression(
                    source,
                    *local_factors,
                    operator,
                    *copy_factors,
                )
            return left, contribution

        for left, contribution in _ordered_bounded_map(
            evaluate,
            prepared,
            max_workers=max_workers,
            executor=executor,
        ):
            result[left] += contribution
        return self._stored_message(site, result)

    def advance_right_identity(
        self,
        message,
        site,
        *,
        max_workers=1,
        executor=None,
        copy_backend="auto",
    ):
        """Advance through ``delta(left, right)`` times physical all-ones."""
        site = int(site)
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if self.charge_resolved:
            raise ValueError(
                "identity advance requires explicit virtual bra/ket blocks."
            )
        tensor_shape = self.tensor_shapes[site]
        if tensor_shape[0] != tensor_shape[1]:
            raise ValueError("identity advance requires equal virtual dimensions.")
        message = self._validated_message(message, site + 1)
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        copy_backend, use_native_copies = self._native_advance_enabled(
            copy_backend,
            dtype,
            identity=True,
        )
        result = self._empty_message(site, dtype)
        prepared = []
        for transitions in self._right_transition_groups[site]:
            left, right = transitions[0]
            copy_plan = (
                self._advance_copy_plan(
                    "right",
                    site,
                    left,
                    right,
                    batch_size=len(transitions),
                    identity=True,
                )
                if use_native_copies
                else None
            )
            copy_plan = self._auto_advance_copy_plan(
                copy_backend,
                copy_plan,
            )
            if copy_plan is None:
                expression, copy_factors = (
                    self._advance_right_identity_expression(
                        site,
                        left,
                        right,
                        batch_size=len(transitions),
                    )
                )
            else:
                expression, copy_factors = None, ()
            prepared.append(
                (transitions, copy_plan, expression, copy_factors)
            )

        def evaluate(arguments):
            transitions, copy_plan, expression, copy_factors = arguments
            left, right = transitions[0]
            source = message.blocks[right]
            operator = self._operators_for_transitions(site, transitions)
            if len(transitions) > 1:
                source = np.stack(
                    [message.blocks[channel] for _, channel in transitions]
                )
            if copy_plan is not None:
                contribution = self._evaluate_advance_copy(
                    copy_plan,
                    (source, operator),
                )
            else:
                contribution = expression(
                    source,
                    operator,
                    *copy_factors,
                )
            return left, contribution

        for left, contribution in _ordered_bounded_map(
            evaluate,
            prepared,
            max_workers=max_workers,
            executor=executor,
        ):
            result[left] += contribution
        return BlockFrontierMessage(site, tuple(result))

    def build_left(
        self,
        tensors,
        *,
        max_workers=1,
        executor=None,
        copy_backend="auto",
    ):
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        if max_workers == 1:
            for site in range(self.nsites):
                messages[site + 1] = self.advance_left(
                    messages[site],
                    tensors,
                    site,
                    copy_backend=copy_backend,
                )
            return messages
        if executor is not None:
            for site in range(self.nsites):
                messages[site + 1] = self.advance_left(
                    messages[site],
                    tensors,
                    site,
                    max_workers=max_workers,
                    executor=executor,
                    copy_backend=copy_backend,
                )
            return messages
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for site in range(self.nsites):
                messages[site + 1] = self.advance_left(
                    messages[site],
                    tensors,
                    site,
                    max_workers=max_workers,
                    executor=executor,
                    copy_backend=copy_backend,
                )
        return messages

    def build_right(
        self,
        tensors,
        *,
        max_workers=1,
        executor=None,
        copy_backend="auto",
    ):
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        if max_workers == 1:
            for site in range(self.nsites - 1, -1, -1):
                messages[site] = self.advance_right(
                    messages[site + 1],
                    tensors,
                    site,
                    copy_backend=copy_backend,
                )
            return messages
        if executor is not None:
            for site in range(self.nsites - 1, -1, -1):
                messages[site] = self.advance_right(
                    messages[site + 1],
                    tensors,
                    site,
                    max_workers=max_workers,
                    executor=executor,
                    copy_backend=copy_backend,
                )
            return messages
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for site in range(self.nsites - 1, -1, -1):
                messages[site] = self.advance_right(
                    messages[site + 1],
                    tensors,
                    site,
                    max_workers=max_workers,
                    executor=executor,
                    copy_backend=copy_backend,
                )
        return messages

    def scalar(self, tensors):
        message = self.left_boundary()
        for site in range(self.nsites):
            message = self.advance_left(message, tensors, site)
        return self.boundary_scalar(message, self.nsites)

    def boundary_scalar(self, message, cut):
        """Extract a scalar from a completed left or right block message."""
        cut = int(cut)
        if cut not in {0, self.nsites}:
            raise ValueError("a scalar can only be extracted at a boundary cut.")
        message = self._validated_message(message, cut)
        value = message.blocks[0]
        return np.asarray(value).reshape(()).item()

    def dense_message(self, message):
        """Expand an identity/charge-block message to dense MPO-frontier form."""
        message = self._validated_message(message, message.cut)
        cut = message.cut
        shape = [
            self.virtual_bonds[cut],
            self.virtual_bonds[cut],
            self.mpo_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            shape.extend((self.dims[site], self.dims[site]))
        dense = np.zeros(shape, dtype=self.dtype)
        for channel in self._active_channels[cut]:
            block = self._message_block(message, channel)
            paired = set(self.paired_sites(cut, channel))
            for compressed_index in np.ndindex(*block.shape):
                offset = 0
                if self.charge_resolved:
                    bra, ket = self._bond_pairs[cut][channel][compressed_index[0]]
                    offset = 1
                else:
                    bra, ket = compressed_index[:2]
                    offset = 2
                physical = []
                for site in self.frontier_sites[cut]:
                    if site in paired:
                        physical.extend(compressed_index[offset : offset + 2])
                        offset += 2
                    else:
                        value = compressed_index[offset]
                        physical.extend((value, value))
                        offset += 1
                dense[(bra, ket, channel, *physical)] = block[compressed_index]
        return dense

    def uncharge_message(self, message):
        """Expand only the virtual charge-pair selector of a block message."""
        if not self.charge_resolved:
            return self._validated_message(message, message.cut)
        message = self._validated_message(message, message.cut)
        cut = message.cut
        result = [
            np.zeros(0, dtype=self.dtype)
            for _ in range(self.mpo_bonds[cut])
        ]
        for channel in self._active_channels[cut]:
            source = self._message_block(message, channel)
            shape = (
                self.virtual_bonds[cut],
                self.virtual_bonds[cut],
                *source.shape[1:],
            )
            target = np.zeros(shape, dtype=source.dtype)
            for pair, (bra, ket) in enumerate(
                self._bond_pairs[cut][channel]
            ):
                target[(bra, ket)] = source[pair]
            result[channel] = target
        return BlockFrontierMessage(cut, tuple(result))

    def _hole_expression(
        self,
        mode,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
        request_batch_size=1,
    ):
        batch_size = int(batch_size)
        request_batch_size = int(request_batch_size)
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive.")
        if mode not in {
            "matrix",
            "action",
            "physical_block",
            "physical_blocks",
        }:
            raise ValueError(
                "hole mode must be 'matrix', 'action', 'physical_block', "
                "or 'physical_blocks'."
            )
        site = int(site)
        left_labels, left_copies = self._block_labels_and_copies(
            site, int(left_channel)
        )
        right_labels, right_copies = self._block_labels_and_copies(
            site + 1, int(right_channel)
        )
        copies = left_copies + right_copies
        bra_labels = self._tensor_labels(site, bra=True)
        ket_labels = self._tensor_labels(site, bra=False)
        left_shape = self.block_shape(site, left_channel)
        right_shape = self.block_shape(site + 1, right_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (self.bra_physical[site], self.ket_physical[site])
        if batch_size > 1:
            batch_label = self._new_label()
            left_shape = (batch_size,) + left_shape
            right_shape = (batch_size,) + right_shape
            left_labels = (batch_label,) + left_labels
            right_labels = (batch_label,) + right_labels
            operator_shape = (batch_size,) + operator_shape
            operator_labels = (batch_label,) + operator_labels
        key = (
            "hole",
            mode,
            site,
            tuple(self.paired_sites(site, left_channel)),
            tuple(self.paired_sites(site + 1, right_channel)),
            tuple(left_shape),
            tuple(right_shape),
            batch_size,
            request_batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached, tuple(value for value, _labels in copies)
        arguments = [
            left_shape,
            left_labels,
            right_shape,
            right_labels,
            operator_shape,
            operator_labels,
        ]
        if mode == "action":
            arguments.extend((self.tensor_shapes[site], ket_labels))
            output_labels = bra_labels
        elif mode in {"physical_block", "physical_blocks"}:
            request_label = self._new_label() if mode == "physical_blocks" else None
            for physical_site in self.physical_sites[site]:
                selector_shape = (self.dims[physical_site],)
                bra_selector_labels = (self.bra_physical[physical_site],)
                ket_selector_labels = (self.ket_physical[physical_site],)
                if request_label is not None:
                    selector_shape = (request_batch_size, *selector_shape)
                    bra_selector_labels = (
                        request_label,
                        *bra_selector_labels,
                    )
                    ket_selector_labels = (
                        request_label,
                        *ket_selector_labels,
                    )
                arguments.extend(
                    (
                        selector_shape,
                        bra_selector_labels,
                        selector_shape,
                        ket_selector_labels,
                    )
                )
            output_labels = (
                self.bra_bonds[site],
                self.bra_bonds[site + 1],
                self.ket_bonds[site],
                self.ket_bonds[site + 1],
            )
            if request_label is not None:
                output_labels = (request_label, *output_labels)
        else:
            output_labels = bra_labels + ket_labels
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression, tuple(value for value, _labels in copies)

    def _hole_matrix_gemm_plan(
        self,
        site,
        left_channel,
        right_channel,
        *,
        batch_size,
    ):
        """Plan a no-copy hole contraction as batched matrix products."""
        site = int(site)
        left_channel = int(left_channel)
        right_channel = int(right_channel)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if (
            set(self.paired_sites(site, left_channel))
            != set(self.frontier_sites[site])
            or set(self.paired_sites(site + 1, right_channel))
            != set(self.frontier_sites[site + 1])
        ):
            return None

        left_labels, left_copies = self._block_labels_and_copies(
            site, left_channel
        )
        right_labels, right_copies = self._block_labels_and_copies(
            site + 1, right_channel
        )
        if left_copies or right_copies:
            return None
        output_labels = (
            *self._tensor_labels(site, bra=True),
            *self._tensor_labels(site, bra=False),
        )
        local_labels = (
            self.bra_physical[site],
            self.ket_physical[site],
        )
        left_set = set(left_labels)
        right_set = set(right_labels)
        output_set = set(output_labels)
        left_owns_local = set(local_labels).issubset(left_set)
        right_owns_local = set(local_labels).issubset(right_set)
        if (
            left_owns_local == right_owns_local
            or not set(local_labels).issubset(output_set)
        ):
            return None

        shared = tuple(
            label
            for label in output_labels
            if label in left_set and label in right_set
        )
        left_output = tuple(
            label
            for label in output_labels
            if label in left_set and label not in right_set
        )
        right_output = tuple(
            label
            for label in output_labels
            if label in right_set and label not in left_set
        )
        contracted = tuple(
            label
            for label in left_labels
            if label in right_set and label not in output_set
        )
        if (
            set(left_labels)
            != set(shared) | set(left_output) | set(contracted)
            or set(right_labels)
            != set(shared) | set(right_output) | set(contracted)
            or output_set
            != set(shared) | set(left_output) | set(right_output)
        ):
            return None

        left_shape = self.block_shape(site, left_channel)
        right_shape = self.block_shape(site + 1, right_channel)
        left_dims = dict(zip(left_labels, left_shape))
        right_dims = dict(zip(right_labels, right_shape))
        if any(
            left_dims[label] != right_dims[label]
            for label in shared + contracted
        ):
            return None
        dimensions = {**left_dims, **right_dims}
        if any(dimensions[label] != self.dims[site] for label in local_labels):
            return None

        batch_label = -1
        left_full_labels = (batch_label, *left_labels)
        right_full_labels = (batch_label, *right_labels)
        operator_labels = (batch_label, *local_labels)
        owner_labels = (
            left_full_labels if left_owns_local else right_full_labels
        )
        operator_axes = tuple(
            sorted(
                range(len(operator_labels)),
                key=lambda axis: owner_labels.index(operator_labels[axis]),
            )
        )
        operator_shape = (batch_size, self.dims[site], self.dims[site])
        broadcast_shape = [1] * len(owner_labels)
        for axis in operator_axes:
            label = operator_labels[axis]
            broadcast_shape[owner_labels.index(label)] = operator_shape[axis]

        shared_dims = tuple(dimensions[label] for label in shared)
        left_output_dims = tuple(dimensions[label] for label in left_output)
        right_output_dims = tuple(dimensions[label] for label in right_output)
        contracted_dims = tuple(dimensions[label] for label in contracted)
        matrix_labels = shared + left_output + right_output
        if set(matrix_labels) != output_set:
            return None
        output_axes = tuple(matrix_labels.index(label) for label in output_labels)
        return {
            "batch_size": batch_size,
            "owner": "left" if left_owns_local else "right",
            "operator_axes": operator_axes,
            "operator_broadcast_shape": tuple(broadcast_shape),
            "left_axes": tuple(
                left_full_labels.index(label)
                for label in (*shared, *left_output, batch_label, *contracted)
            ),
            "right_axes": tuple(
                right_full_labels.index(label)
                for label in (*shared, *right_output, batch_label, *contracted)
            ),
            "shared_dims": shared_dims,
            "left_output_dims": left_output_dims,
            "right_output_dims": right_output_dims,
            "left_size": int(np.prod(left_output_dims, dtype=np.int64)),
            "right_size": int(np.prod(right_output_dims, dtype=np.int64)),
            "contracted_size": int(
                batch_size * np.prod(contracted_dims, dtype=np.int64)
            ),
            "output_axes": output_axes,
        }

    def _hole_matrix_copy_plan(
        self,
        site,
        left_channel,
        right_channel,
        *,
        batch_size,
    ):
        """Plan a native hole contraction with three-way copy constraints."""
        if self.charge_resolved or not _copy_einsum_native_available():
            return None
        site = int(site)
        left_channel = int(left_channel)
        right_channel = int(right_channel)
        batch_size = int(batch_size)
        key = (site, left_channel, right_channel, batch_size)
        cached = self._copy_hole_plans.get(key)
        if cached is not None:
            return cached

        left_labels, left_copies = self._block_labels_and_copies(
            site, left_channel
        )
        right_labels, right_copies = self._block_labels_and_copies(
            site + 1, right_channel
        )
        copies = left_copies + right_copies
        if not copies:
            return None
        operator_labels = (
            self.bra_physical[site],
            self.ket_physical[site],
        )
        left_shape = self.block_shape(site, left_channel)
        right_shape = self.block_shape(site + 1, right_channel)
        operator_shape = (self.dims[site], self.dims[site])
        if batch_size > 1:
            batch_label = self._new_label()
            left_labels = (batch_label, *left_labels)
            right_labels = (batch_label, *right_labels)
            operator_labels = (batch_label, *operator_labels)
            left_shape = (batch_size, *left_shape)
            right_shape = (batch_size, *right_shape)
            operator_shape = (batch_size, *operator_shape)
        output_labels = (
            *self._tensor_labels(site, bra=True),
            *self._tensor_labels(site, bra=False),
        )
        dimensions = {}
        for shape, labels in (
            (left_shape, left_labels),
            (right_shape, right_labels),
            (operator_shape, operator_labels),
        ):
            for dimension, label in zip(shape, labels):
                previous = dimensions.setdefault(label, int(dimension))
                if previous != int(dimension):
                    raise ValueError(
                        "copy-aware hole labels have inconsistent dimensions."
                    )
        copy_labels = []
        copy_dimensions = []
        parents = {}

        def find(label):
            parents.setdefault(label, label)
            while parents[label] != label:
                parents[label] = parents[parents[label]]
                label = parents[label]
            return label

        def union(left, right):
            left = find(left)
            right = find(right)
            if left != right:
                parents[right] = left

        for value, labels in copies:
            dimension = int(value.shape[0])
            if value.shape != (dimension, dimension, dimension):
                raise ValueError("copy factors must be cubic three-way tensors.")
            labels = tuple(int(label) for label in labels)
            copy_labels.append(labels)
            copy_dimensions.append(dimension)
            for label in labels:
                previous = dimensions.setdefault(label, dimension)
                if previous != dimension:
                    raise ValueError(
                        "copy-aware hole labels have inconsistent dimensions."
                    )
            union(labels[0], labels[1])
            union(labels[0], labels[2])
        for labels in (left_labels, right_labels, operator_labels, output_labels):
            for label in labels:
                find(label)
        operation_dimensions = {}
        for label, dimension in dimensions.items():
            root = find(label)
            previous = operation_dimensions.setdefault(root, dimension)
            if previous != dimension:
                raise ValueError(
                    "copy-connected hole labels have inconsistent dimensions."
                )
        operation_count = int(
            np.prod(tuple(operation_dimensions.values()), dtype=object)
        )
        output_roots = {find(label) for label in output_labels}
        contracted_count = int(
            np.prod(
                tuple(
                    dimension
                    for root, dimension in operation_dimensions.items()
                    if root not in output_roots
                ),
                dtype=object,
            )
        )
        cached = {
            "left_labels": np.asarray(left_labels, dtype=np.intp),
            "right_labels": np.asarray(right_labels, dtype=np.intp),
            "operator_labels": np.asarray(operator_labels, dtype=np.intp),
            "output_labels": np.asarray(output_labels, dtype=np.intp),
            "copy_labels": np.asarray(copy_labels, dtype=np.intp).reshape(-1, 3),
            "copy_dimensions": np.asarray(copy_dimensions, dtype=np.intp),
            "operation_count": operation_count,
            "contracted_count": contracted_count,
            "workspace_bytes": 3 * np.dtype(np.intp).itemsize * contracted_count,
        }
        self._copy_hole_plans[key] = cached
        return cached

    @staticmethod
    def _evaluate_hole_matrix_copy(plan, left_blocks, right_blocks, operators):
        """Execute one native equality-constrained topology group."""
        return _contract_copy_einsum(
            left_blocks,
            right_blocks,
            operators,
            plan["left_labels"],
            plan["right_labels"],
            plan["operator_labels"],
            plan["output_labels"],
            plan["copy_labels"],
            plan["copy_dimensions"],
        )

    @staticmethod
    def _evaluate_hole_matrix_gemm(
        plan,
        left_blocks,
        right_blocks,
        operators,
    ):
        """Execute a plan returned by :meth:`_hole_matrix_gemm_plan`."""
        left_blocks = np.asarray(left_blocks)
        right_blocks = np.asarray(right_blocks)
        operators = np.asarray(operators)
        if plan["batch_size"] == 1:
            left_blocks = left_blocks[None, ...]
            right_blocks = right_blocks[None, ...]
            operators = operators[None, ...]
        operator = operators.transpose(plan["operator_axes"]).reshape(
            plan["operator_broadcast_shape"]
        )
        if plan["owner"] == "left":
            left_blocks = left_blocks * operator
        else:
            right_blocks = right_blocks * operator

        shared_dims = plan["shared_dims"]
        contracted_size = plan["contracted_size"]
        left_matrix = left_blocks.transpose(plan["left_axes"]).reshape(
            *shared_dims,
            plan["left_size"],
            contracted_size,
        )
        right_matrix = right_blocks.transpose(plan["right_axes"]).reshape(
            *shared_dims,
            plan["right_size"],
            contracted_size,
        )
        result = left_matrix @ right_matrix.swapaxes(-2, -1)
        result = result.reshape(
            *shared_dims,
            *plan["left_output_dims"],
            *plan["right_output_dims"],
        )
        if plan["output_axes"] != tuple(range(result.ndim)):
            result = result.transpose(plan["output_axes"])
        return result

    def hole_matrix(
        self,
        site,
        left,
        right,
        *,
        max_workers=1,
        parallel_min_size=512,
        copy_backend="auto",
    ):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            routes = self._packed_fixed_hole_routes(site, left, right)
            size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
            dtype = np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
            result = np.zeros((size, size), dtype=dtype)
            for bra, ket, weight in routes:
                np.add.at(result, (bra, ket), weight)
            return result
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        max_workers = int(max_workers)
        parallel_min_size = int(parallel_min_size)
        copy_backend = str(copy_backend).lower().replace("-", "_")
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if parallel_min_size < 1:
            raise ValueError("parallel_min_size must be positive.")
        if copy_backend not in {"auto", "native", "python"}:
            raise ValueError(
                "copy_backend must be 'auto', 'native', or 'python'."
            )
        if copy_backend == "native" and not _copy_einsum_native_available():
            raise RuntimeError(
                "copy_backend='native' requires the optional copy-einsum "
                "extension."
            )
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.dtype(
            np.result_type(
                self.dtype,
                np.float64,
                *[block.dtype for block in left.blocks if block is not None],
                *[block.dtype for block in right.blocks if block is not None],
            )
        )
        native_dtype = dtype in {
            np.dtype(np.float64),
            np.dtype(np.complex128),
        }
        if copy_backend == "native" and not native_dtype:
            raise TypeError(
                "copy_backend='native' supports only float64 and complex128."
            )
        use_native_copies = (
            copy_backend != "python"
            and _copy_einsum_native_available()
            and native_dtype
        )
        result = np.zeros((size, size), dtype=dtype)
        prepared = []
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            gemm_plan = self._hole_matrix_gemm_plan(
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            copy_plan = (
                self._hole_matrix_copy_plan(
                    site,
                    left_channel,
                    right_channel,
                    batch_size=batch_size,
                )
                if gemm_plan is None and use_native_copies
                else None
            )
            if (
                copy_backend == "auto"
                and copy_plan is not None
                and (
                    copy_plan["operation_count"]
                    > _COPY_HOLE_AUTO_MAX_OPERATIONS
                    or copy_plan["workspace_bytes"]
                    > _COPY_HOLE_AUTO_MAX_WORKSPACE_BYTES
                )
            ):
                copy_plan = None
            if gemm_plan is None and copy_plan is None:
                expression, copy_factors = self._hole_expression(
                    "matrix",
                    site,
                    left_channel,
                    right_channel,
                    batch_size=batch_size,
                )
            else:
                expression, copy_factors = None, ()
            prepared.append(
                (
                    transitions,
                    gemm_plan,
                    copy_plan,
                    expression,
                    copy_factors,
                )
            )

        def evaluate(arguments):
            (
                transitions,
                gemm_plan,
                copy_plan,
                expression,
                copy_factors,
            ) = arguments
            left_channel, right_channel = transitions[0]
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            operators = self._operators_for_transitions(site, transitions)
            if len(transitions) > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            if gemm_plan is not None:
                value = self._evaluate_hole_matrix_gemm(
                    gemm_plan,
                    left_blocks,
                    right_blocks,
                    operators,
                )
            elif copy_plan is not None:
                value = self._evaluate_hole_matrix_copy(
                    copy_plan,
                    left_blocks,
                    right_blocks,
                    operators,
                )
            else:
                value = expression(
                    left_blocks,
                    right_blocks,
                    operators,
                    *copy_factors,
                )
            return np.asarray(value).reshape(size, size)

        workers = (
            min(max_workers, len(prepared))
            if size >= parallel_min_size
            else 1
        )
        if workers <= 1:
            for arguments in prepared:
                result += evaluate(arguments)
            return result

        # Independent topology groups can contract concurrently. Futures are
        # nevertheless reduced in their original order, retaining the serial
        # floating-point summation and bounding live contributions by workers.
        for contribution in _ordered_bounded_map(
            evaluate,
            prepared,
            max_workers=workers,
        ):
            result += contribution
        return result

    def _validated_physical_configuration(self, site, configuration, *, name):
        configuration = tuple(int(value) for value in configuration)
        physical_sites = self.physical_groups[int(site)]
        if len(configuration) != len(physical_sites):
            raise ValueError(
                f"{name} must contain {len(physical_sites)} physical values."
            )
        if any(
            value < 0 or value >= self.dims[physical_site]
            for value, physical_site in zip(configuration, physical_sites)
        ):
            raise ValueError(f"{name} contains an out-of-range physical value.")
        return configuration

    def hole_block(self, site, left, right, bra_configuration, ket_configuration):
        """Return one fixed-physical virtual block without a dense hole matrix."""
        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        bra_configuration = self._validated_physical_configuration(
            site, bra_configuration, name="bra_configuration"
        )
        ket_configuration = self._validated_physical_configuration(
            site, ket_configuration, name="ket_configuration"
        )
        if self.charge_resolved and self.device == "cpu":
            routes = self._packed_fixed_hole_routes(site, left, right)
            virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
            physical_shape = self.tensor_shapes[site][2:]
            physical_size = int(np.prod(physical_shape, dtype=np.int64))
            bra_physical = int(np.ravel_multi_index(bra_configuration, physical_shape))
            ket_physical = int(np.ravel_multi_index(ket_configuration, physical_shape))
            dtype = np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
            result = np.zeros((virtual_size, virtual_size), dtype=dtype)
            for bra, ket, weight in routes:
                selected = (
                    (bra % physical_size == bra_physical)
                    & (ket % physical_size == ket_physical)
                )
                if np.any(selected):
                    np.add.at(
                        result,
                        (
                            bra[selected] // physical_size,
                            ket[selected] // physical_size,
                        ),
                        weight[selected],
                    )
            return result
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((virtual_size, virtual_size), dtype=dtype)
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_groups[site], bra_configuration, ket_configuration
        ):
            identity = self._identities[self.dims[physical_site]]
            selectors.extend((identity[bra_value], identity[ket_value]))
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "physical_block",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                *selectors,
                *copy_factors,
            )
            result += np.asarray(value).reshape(virtual_size, virtual_size)
        return result

    def hole_blocks(
        self,
        site,
        left,
        right,
        configuration_pairs,
        *,
        request_batch_size=64,
        transition_batch_size=_DEFAULT_HOLE_TRANSITION_BATCH_SIZE,
    ):
        """Return fixed-physical blocks with batched selector contractions."""
        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        request_batch_size = int(request_batch_size)
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive.")
        transition_batch_size = int(transition_batch_size)
        if transition_batch_size < 1:
            raise ValueError("transition_batch_size must be positive.")
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        prepared = []
        for row, column, bra_configuration, ket_configuration in configuration_pairs:
            bra_configuration = self._validated_physical_configuration(
                site, bra_configuration, name="bra_configuration"
            )
            ket_configuration = self._validated_physical_configuration(
                site, ket_configuration, name="ket_configuration"
            )
            prepared.append(
                (
                    int(row),
                    int(column),
                    bra_configuration,
                    ket_configuration,
                )
            )
        result = {
            (row, column): np.zeros((virtual_size, virtual_size), dtype=dtype)
            for row, column, _bra, _ket in prepared
        }
        if not prepared:
            return result
        selector_batches = []
        for axis, physical_site in enumerate(self.physical_sites[site]):
            identity = self._identities[self.dims[physical_site]]
            selector_batches.extend(
                (
                    identity[
                        np.asarray(
                            [bra[axis] for _row, _column, bra, _ket in prepared],
                            dtype=np.intp,
                        )
                    ],
                    identity[
                        np.asarray(
                            [ket[axis] for _row, _column, _bra, ket in prepared],
                            dtype=np.intp,
                        )
                    ],
                )
            )
        values = np.zeros(
            (len(prepared), virtual_size, virtual_size),
            dtype=dtype,
        )
        for transition_group in self._hole_transition_groups[site]:
            # Large transition batches can force opt_einsum into a very costly
            # multi-index c_einsum. Small exact partial sums retain BLAS-friendly
            # paths and substantially reduce the peak intermediate.
            for transition_start in range(
                0, len(transition_group), transition_batch_size
            ):
                transitions = transition_group[
                    transition_start : transition_start + transition_batch_size
                ]
                left_channel, right_channel = transitions[0]
                batch_size = len(transitions)
                left_blocks = left.blocks[left_channel]
                right_blocks = right.blocks[right_channel]
                operators = self._operators_for_transitions(site, transitions)
                if batch_size > 1:
                    left_blocks = np.stack(
                        [left.blocks[channel] for channel, _ in transitions]
                    )
                    right_blocks = np.stack(
                        [right.blocks[channel] for _, channel in transitions]
                    )
                for start in range(0, len(prepared), request_batch_size):
                    stop = min(start + request_batch_size, len(prepared))
                    chunk_size = stop - start
                    expression, copy_factors = self._hole_expression(
                        "physical_blocks",
                        site,
                        left_channel,
                        right_channel,
                        batch_size=batch_size,
                        request_batch_size=chunk_size,
                    )
                    value = expression(
                        left_blocks,
                        right_blocks,
                        operators,
                        *(
                            selectors[start:stop]
                            for selectors in selector_batches
                        ),
                        *copy_factors,
                    )
                    values[start:stop] += np.asarray(value).reshape(
                        chunk_size,
                        virtual_size,
                        virtual_size,
                    )
        for index, (row, column, _bra, _ket) in enumerate(prepared):
            result[(row, column)] += values[index]
        return result

    def hole_action(self, site, left, right, vector):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._packed_hole_action(site, left, right, vector)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(size, dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *copy_factors,
            )
            result += np.asarray(value).reshape(-1)
        return result

    def prepare_hole_action(self, site, left, right):
        """Bind fixed messages and grouped transition operands for a local solve."""

        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._prepare_packed_hole_action(site, left, right)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        routes = []
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            xp = self._xp
            route_dtype = self.compute_dtype
            if route_dtype.kind != "c" and any(
                np.asarray(value).dtype.kind == "c"
                for value in (left_blocks, right_blocks, operators)
            ):
                route_dtype = np.dtype(
                    np.complex64
                    if route_dtype.itemsize <= 4
                    else np.complex128
                )
            routes.append(
                (
                    expression,
                    left_channel,
                    right_channel,
                    batch_size,
                    xp.asarray(left_blocks, dtype=route_dtype),
                    xp.asarray(right_blocks, dtype=route_dtype),
                    xp.asarray(operators, dtype=route_dtype),
                    tuple(
                        xp.asarray(factor, dtype=route_dtype)
                        for factor in copy_factors
                    ),
                )
            )

        shape = self.tensor_shapes[site]
        size = int(np.prod(shape))
        dtype = np.dtype(
            np.result_type(
                self.compute_dtype,
                *[
                    value.dtype
                    for route in routes
                    for value in route[4:7]
                ],
            )
        )
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[block.dtype for block in left.blocks if block is not None],
                *[block.dtype for block in right.blocks if block is not None],
            )
        )

        def action_dtype_for(value):
            value_dtype = np.asarray(value).dtype
            if value_dtype.kind == "c" and dtype.kind != "c":
                return np.dtype(
                    np.complex64 if dtype.itemsize <= 4 else np.complex128
                )
            return dtype

        def action(vector):
            vector = np.asarray(vector)
            action_dtype = action_dtype_for(vector)
            xp = self._xp
            result = xp.zeros(size, dtype=action_dtype)
            tensor = xp.asarray(vector.reshape(shape), dtype=action_dtype)
            for (
                expression,
                _left_channel,
                _right_channel,
                _batch_size,
                left_blocks,
                right_blocks,
                operators,
                copies,
            ) in routes:
                operands = (
                    left_blocks,
                    right_blocks,
                    operators,
                    tensor,
                    *copies,
                )
                value = (
                    expression(*operands)
                    if self.device == "cpu"
                    else expression(*operands, backend="cupy")
                )
                result += value.reshape(-1)
            return xp.asnumpy(result) if self.device == "cuda" else np.asarray(result)

        def actions(vectors):
            vectors = np.asarray(vectors)
            if vectors.ndim != 2 or vectors.shape[1] != size:
                raise ValueError(f"vectors must have shape (batch, {size}).")
            vector_batch = vectors.shape[0]
            xp = self._xp
            action_dtype = action_dtype_for(vectors)
            result = xp.zeros(
                (vector_batch, size),
                dtype=action_dtype,
            )
            tensors = xp.asarray(
                vectors.reshape(vector_batch, *shape), dtype=action_dtype
            )
            for (
                _expression,
                left_channel,
                right_channel,
                transition_batch,
                left_blocks,
                right_blocks,
                operators,
                copies,
            ) in routes:
                expression, _copy_factors = self._hole_expression(
                    "actions",
                    site,
                    left_channel,
                    right_channel,
                    batch_size=transition_batch,
                    vector_batch_size=vector_batch,
                )
                operands = (
                    left_blocks,
                    right_blocks,
                    operators,
                    tensors,
                    *copies,
                )
                value = (
                    expression(*operands)
                    if self.device == "cpu"
                    else expression(*operands, backend="cupy")
                )
                result += value.reshape(vector_batch, size)
            return xp.asnumpy(result) if self.device == "cuda" else np.asarray(result)

        action.many = actions
        if self._compute_dtype_requested and dtype != native_dtype:
            action.verify = lambda vector: self.hole_action(site, left, right, vector)
        return action

    def hole_action_components(self, site, left, right, vector):
        """Yield exact sparse-MPO transition-group contributions.

        Keeping these contributions separate exposes the Hamiltonian channel
        range needed by AMEn-style subspace enrichment without assembling a
        dense effective Hamiltonian or retaining every contribution at once.
        Their sum is exactly :meth:`hole_action`.
        """
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            vector = np.asarray(vector).reshape(-1)
            size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
            if vector.size != size:
                raise ValueError(f"vector must contain {size} entries.")
            for transitions in self._hole_transition_groups[site]:
                routes = self._packed_fixed_hole_routes(
                    site, left, right, transitions
                )
                dtype = np.result_type(
                    self.dtype,
                    vector.dtype,
                    *[route[2].dtype for route in routes],
                )
                result = np.zeros(size, dtype=dtype)
                for bra, ket, weight in routes:
                    np.add.at(result, bra, weight * vector[ket])
                yield result
            return
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *copy_factors,
            )
            yield np.asarray(value).reshape(-1)

    def hole_action_component_count(self, site):
        return len(self._hole_transition_groups[int(site)])

    def _enrichment_expression(self, direction, site, left_channel, right_channel):
        """Plan one single-layer ``L W A`` or ``A W R`` contribution."""
        key = (
            "enrichment",
            str(direction),
            int(site),
            int(left_channel),
            int(right_channel),
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        if direction == "left":
            cut, channel = site, int(left_channel)
            row_labels = (
                self.bra_bonds[site],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site + 1],
            ]
        elif direction == "right":
            cut, channel = site + 1, int(right_channel)
            row_labels = (
                self.bra_bonds[site + 1],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site],
            ]
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        source_labels, source_copies = self._block_labels_and_copies(cut, channel)
        local = set(self.physical_groups[site])
        paired = set(self.paired_sites(cut, channel))
        for physical_site in self.frontier_sites[cut]:
            if physical_site in local:
                continue
            column_labels.append(self.bra_physical[physical_site])
            if physical_site in paired:
                column_labels.append(self.ket_physical[physical_site])

        arguments = [
            self.block_shape(cut, channel),
            source_labels,
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            (self.dims[site], self.dims[site]),
            (self.bra_physical[site], self.ket_physical[site]),
        ]
        identities = []
        for physical_site in self.physical_groups[site][1:]:
            identity = self._identities[self.dims[physical_site]]
            identities.append(identity)
            arguments.extend(
                (
                    identity.shape,
                    (
                        self.bra_physical[physical_site],
                        self.ket_physical[physical_site],
                    ),
                )
            )
        for value, labels in source_copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            row_labels + tuple(column_labels),
            optimize=self.optimize,
        )
        cached = (
            expression,
            tuple(identities),
            tuple(value for value, _labels in source_copies),
            int(
                self.virtual_bonds[site if direction == "left" else site + 1]
                * np.prod(
                    [self.dims[index] for index in self.physical_groups[site]],
                    dtype=np.int64,
                )
            ),
        )
        self._expressions[key] = cached
        return cached

    def _packed_enrichment_component(
        self,
        direction,
        site,
        message,
        vector,
        left_channel,
        right_channel,
    ):
        """Build one open AMEn component directly from packed U(1) storage."""

        site = int(site)
        if direction == "left":
            cut, channel, source_bond_axis = site, int(left_channel), 0
            opposite_bond_axis = 2
        elif direction == "right":
            cut, channel, source_bond_axis = site + 1, int(right_channel), 1
            opposite_bond_axis = 1
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        message = self._validated_message(message, cut)
        source_bra_bond, source_ket_bond, source_bra, source_ket = (
            self._expanded_storage_coordinates(cut, channel)
        )
        source = np.asarray(message.blocks[channel]).reshape(-1)
        tensor = np.asarray(vector).reshape(-1)
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        source_frontier = set(self.frontier_sites[cut])
        overlap = tuple(index for index in local_sites if index in source_frontier)
        outside = tuple(index for index in self.frontier_sites[cut] if index not in local_position)
        paired = set(self.paired_sites(cut, channel))

        tensor_groups = {}
        for flat, left_bond, right_bond, physical in self._tensor_support(site):
            bond = left_bond if source_bond_axis == 0 else right_bond
            values = tuple(physical[local_position[index]] for index in overlap)
            tensor_groups.setdefault((bond, values), []).append(
                (flat, left_bond, right_bond, physical)
            )

        row_shape = (
            self.virtual_bonds[cut],
            *(self.dims[index] for index in local_sites),
        )
        column_shape = [
            self.virtual_bonds[site + 1]
            if direction == "left"
            else self.virtual_bonds[site]
        ]
        for physical_site in outside:
            column_shape.append(self.dims[physical_site])
            if physical_site in paired:
                column_shape.append(self.dims[physical_site])
        row_size = int(np.prod(row_shape, dtype=np.int64))
        column_size = int(np.prod(column_shape, dtype=np.int64))
        operator = self.mpo_tensors[site][left_channel, right_channel]
        dtype = np.result_type(self.dtype, source.dtype, tensor.dtype)
        result = np.zeros((row_size, column_size), dtype=dtype)
        for source_index in range(len(source)):
            ket_key = (
                int(source_ket_bond[source_index]),
                tuple(int(source_ket[index][source_index]) for index in overlap),
            )
            column_tail = []
            for physical_site in outside:
                column_tail.append(int(source_bra[physical_site][source_index]))
                if physical_site in paired:
                    column_tail.append(int(source_ket[physical_site][source_index]))
            for ket_flat, ket_left, ket_right, ket_physical in tensor_groups.get(
                ket_key, ()
            ):
                bra_physical = list(ket_physical)
                compatible = True
                for physical_site in local_sites[1:]:
                    if physical_site in source_frontier and (
                        int(source_bra[physical_site][source_index])
                        != bra_physical[local_position[physical_site]]
                    ):
                        compatible = False
                        break
                if not compatible:
                    continue
                owned_bra_values = (
                    (int(source_bra[site][source_index]),)
                    if site in source_frontier
                    else range(self.dims[site])
                )
                for owned_bra in owned_bra_values:
                    coefficient = operator[owned_bra, ket_physical[0]]
                    if coefficient == 0:
                        continue
                    bra_physical[0] = owned_bra
                    row = int(
                        np.ravel_multi_index(
                            (
                                int(source_bra_bond[source_index]),
                                *bra_physical,
                            ),
                            row_shape,
                        )
                    )
                    opposite_bond = (
                        ket_right if opposite_bond_axis == 2 else ket_left
                    )
                    column = int(
                        np.ravel_multi_index(
                            (opposite_bond, *column_tail),
                            tuple(column_shape),
                        )
                    )
                    result[row, column] += (
                        source[source_index] * tensor[ket_flat] * coefficient
                    )
        return result

    def left_enrichment_components(self, site, left, vector):
        """Yield open ``L W A`` matrices for right-going AMEn enrichment."""
        site = int(site)
        left = self._validated_message(left, site)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for left_channel, right_channel in self._transitions[site]:
            if (
                left_channel not in self._active_channel_sets[site]
                or right_channel not in self._active_channel_sets[site + 1]
            ):
                continue
            if self.charge_resolved and self.device == "cpu":
                yield self._packed_enrichment_component(
                    "left",
                    site,
                    left,
                    vector,
                    left_channel,
                    right_channel,
                )
                continue
            expression, identities, copies, row_size = self._enrichment_expression(
                "left", site, left_channel, right_channel
            )
            value = expression(
                self._message_block(left, left_channel),
                vector,
                self.mpo_tensors[site][left_channel, right_channel],
                *identities,
                *copies,
            )
            yield np.asarray(value).reshape(row_size, -1)

    def right_enrichment_components(self, site, right, vector):
        """Yield open ``A W R`` matrices for left-going AMEn enrichment."""
        site = int(site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for left_channel, right_channel in self._transitions[site]:
            if (
                left_channel not in self._active_channel_sets[site]
                or right_channel not in self._active_channel_sets[site + 1]
            ):
                continue
            if self.charge_resolved and self.device == "cpu":
                yield self._packed_enrichment_component(
                    "right",
                    site,
                    right,
                    vector,
                    left_channel,
                    right_channel,
                )
                continue
            expression, identities, copies, row_size = self._enrichment_expression(
                "right", site, left_channel, right_channel
            )
            value = expression(
                self._message_block(right, right_channel),
                vector,
                self.mpo_tensors[site][left_channel, right_channel],
                *identities,
                *copies,
            )
            yield np.asarray(value).reshape(row_size, -1)

    def enrichment_component_count(self, site):
        site = int(site)
        return sum(
            left in self._active_channel_sets[site]
            and right in self._active_channel_sets[site + 1]
            for left, right in self._transitions[site]
        )

    def hole_actions(self, site, left, right, vectors):
        """Apply one block-frontier hole to a batch of local vectors."""
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._prepare_packed_hole_action(site, left, right).many(vectors)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        size = int(np.prod(self.tensor_shapes[site]))
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != size:
            raise ValueError(f"vectors must have shape (batch, {size}).")
        vector_batch_size = vectors.shape[0]
        reshaped = vectors.reshape(vector_batch_size, *self.tensor_shapes[site])
        dtype = np.result_type(
            self.dtype,
            vectors.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((vector_batch_size, size), dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "actions",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
                vector_batch_size=vector_batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                reshaped,
                *copy_factors,
            )
            result += np.asarray(value).reshape(vector_batch_size, size)
        return result

    def _block_transition_paths(self, start, stop):
        """Enumerate sparse MPO-channel paths through a short site block."""
        start = int(start)
        stop = int(stop)
        key = (start, stop)
        cached = self._block_paths.get(key)
        if cached is not None:
            return cached
        if start < 0 or stop > self.nsites or start >= stop:
            raise ValueError("block bounds are invalid.")
        outgoing = []
        for site in range(start, stop):
            by_left = {}
            for left, right in self._transitions[site]:
                if (
                    left in self._active_channel_sets[site]
                    and right in self._active_channel_sets[site + 1]
                ):
                    by_left.setdefault(left, []).append(right)
            outgoing.append(by_left)

        paths = []

        def extend(site, channels):
            if site == stop:
                paths.append(tuple(channels))
                return
            for right in outgoing[site - start].get(channels[-1], ()):
                extend(site + 1, (*channels, right))

        for channel in self._active_channels[start]:
            extend(start, (channel,))
        cached = tuple(paths)
        self._block_paths[key] = cached
        return cached

    def _block_hole_action_expression(
        self,
        start,
        stop,
        union_sites,
        merged_shape,
        left_channel,
        right_channel,
    ):
        key = (
            "block_hole_action",
            int(start),
            int(stop),
            tuple(union_sites),
            tuple(merged_shape),
            int(left_channel),
            int(right_channel),
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        start = int(start)
        stop = int(stop)
        union_sites = tuple(int(site) for site in union_sites)
        merged_shape = tuple(int(dim) for dim in merged_shape)
        left_labels, left_copies = self._block_labels_and_copies(
            start,
            int(left_channel),
        )
        right_labels, right_copies = self._block_labels_and_copies(
            stop,
            int(right_channel),
        )
        copies = left_copies + right_copies
        ket_labels = (
            self.ket_bonds[start],
            self.ket_bonds[stop],
            *(self.ket_physical[site] for site in union_sites),
        )
        bra_labels = (
            self.bra_bonds[start],
            self.bra_bonds[stop],
            *(self.bra_physical[site] for site in union_sites),
        )
        arguments = [
            self.block_shape(start, left_channel),
            left_labels,
            self.block_shape(stop, right_channel),
            right_labels,
        ]
        for site in range(start, stop):
            arguments.extend(
                (
                    (self.dims[site], self.dims[site]),
                    (
                        self.bra_physical[site],
                        self.ket_physical[site],
                    ),
                )
            )
        arguments.extend((merged_shape, ket_labels))
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            bra_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(value for value, _labels in copies))
        self._expressions[key] = cached
        return cached

    def block_hole_action(
        self,
        start,
        stop,
        left,
        right,
        union_sites,
        merged_shape,
        vector,
    ):
        """Apply a short merged-block Hamiltonian termwise.

        Unlike an artificial supersite construction, this preserves the
        original frontier at every physical site and never forms the enlarged
        intermediate message immediately to the right of the merged tensor.
        """
        start = int(start)
        stop = int(stop)
        left = self._validated_message(left, start)
        right = self._validated_message(right, stop)
        merged_shape = tuple(int(dim) for dim in merged_shape)
        vector = np.asarray(vector).reshape(merged_shape)
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(merged_shape, dtype=dtype)
        for path in self._block_transition_paths(start, stop):
            left_channel = path[0]
            right_channel = path[-1]
            left_block = self._message_block(left, left_channel)
            right_block = self._message_block(right, right_channel)
            if left_block is None or right_block is None:
                continue
            expression, copies = self._block_hole_action_expression(
                start,
                stop,
                union_sites,
                merged_shape,
                left_channel,
                right_channel,
            )
            operators = tuple(
                self.mpo_tensors[site][path[site - start], path[site - start + 1]]
                for site in range(start, stop)
            )
            result += expression(
                left_block,
                right_block,
                *operators,
                vector,
                *copies,
            )
        return result.reshape(-1)


__all__ = ["BlockFrontierMessage", "BlockMPOFrontier"]
