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

import numpy as np
from opt_einsum import contract_expression

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
        self.physical_sites = tuple(
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
            len(self.physical_sites)
            == len(self.tensor_shapes)
            == mpo_nsites
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")

        for site, (sites, shape) in enumerate(
            zip(self.physical_sites, self.tensor_shapes)
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
        for left_site, sites in enumerate(self.physical_sites):
            for physical_site in sites[1:]:
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
            *(physical[index] for index in self.physical_sites[site]),
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
        return BlockFrontierMessage(site + 1, tuple(result))

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
        return BlockFrontierMessage(site, tuple(result))

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
                right_blocks = np.stack(
                    [right.blocks[channel] for _, channel in transitions]
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
        physical_sites = self.physical_sites[int(site)]
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
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((virtual_size, virtual_size), dtype=dtype)
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_sites[site], bra_configuration, ket_configuration
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
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *copy_factors,
            )
            result += np.asarray(value).reshape(-1)
        return result


__all__ = ["BlockFrontierMessage", "BlockMPOFrontier"]
