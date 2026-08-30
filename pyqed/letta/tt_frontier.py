"""Tensor-train compression for graph-tied LETTA frontier messages.

The exact :class:`~pyqed.letta.mpo_frontier.MPOFrontier` stores every cut
message as one dense array.  This module provides the same scalar-contraction
interface while storing those arrays as labelled tensor trains (TTs).  The
default ``structured`` absorption path never materializes a dense frontier:
it converts only the site-local double-layer transfer factor to an exact TT,
multiplies the two labelled TTs, eliminates completed variables, and rounds
the result.

The labels are part of the representation.  They make changes in the active
frontier unambiguous and fix the variable order deterministically to the
order used by :class:`~pyqed.letta.mpo_frontier.MPOFrontier`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import sqrt
from typing import Hashable, Iterable

import numpy as np
from opt_einsum import contract
from pyqed.tn import MPO
from pyqed.tn.hamiltonian import _operator_string_mpo

from .block_mpo_frontier import BlockFrontierMessage, BlockMPOFrontier
from .mpo_frontier import MPOFrontier


def _validated_max_rank(max_rank):
    if max_rank is None:
        return None
    max_rank = int(max_rank)
    if max_rank < 1:
        raise ValueError("max_rank must be positive or None.")
    return max_rank


def _validated_tolerances(rtol, atol):
    rtol = float(rtol)
    atol = float(atol)
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and nonnegative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative.")
    return rtol, atol


def _truncation_rank(singular_values, max_rank, local_tolerance_sq):
    """Choose the smallest admissible rank satisfying a local error budget."""

    singular_values = np.asarray(singular_values)
    available = int(singular_values.size)
    if available == 0:
        return 1
    rank = available
    if local_tolerance_sq > 0.0:
        tail_sq = np.cumsum(np.abs(singular_values[::-1]) ** 2)[::-1]
        for candidate in range(1, available + 1):
            discarded = 0.0 if candidate == available else float(tail_sq[candidate])
            if discarded <= local_tolerance_sq:
                rank = candidate
                break
    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(1, rank)


@dataclass(frozen=True)
class TTRoundDiagnostics:
    """Diagnostics for one dense TT-SVD or TT rounding operation."""

    algorithm: str
    original_ranks: tuple[int, ...]
    ranks: tuple[int, ...]
    norm: float
    discarded_weight: float
    relative_discarded_weight: float
    max_rank: int | None
    rtol: float
    atol: float
    densified_input: bool


@dataclass(frozen=True)
class TTAdvanceDiagnostics:
    """Diagnostics for one site transfer between adjacent frontiers."""

    direction: str
    site: int
    source_cut: int
    target_cut: int
    absorption: str
    source_ranks: tuple[int, ...]
    product_ranks: tuple[int, ...]
    target_ranks: tuple[int, ...]
    source_storage_elements: int
    product_storage_elements: int
    target_storage_elements: int
    dense_source_elements: int
    dense_target_elements: int
    local_factor_elements: int
    local_factor_storage_elements: int
    local_factor_ranks: tuple[int, ...]
    local_factor_discarded_weight: float
    message_discarded_weight: float
    discarded_weight: float
    relative_discarded_weight: float
    used_dense_frontier: bool


@dataclass(frozen=True)
class TTContractionDiagnostics:
    """Aggregate diagnostics for the most recent message construction."""

    advances: tuple[TTAdvanceDiagnostics, ...]
    total_discarded_weight: float
    max_relative_discarded_weight: float
    peak_message_storage_elements: int
    peak_dense_message_elements: int
    peak_product_storage_elements: int
    peak_local_factor_elements: int
    dense_frontier_absorptions: int


@dataclass(frozen=True)
class TTHoleDiagnostics:
    """Diagnostics for one matrix-free local hole action."""

    site: int
    method: str
    left_ranks: tuple[int, ...]
    right_ranks: tuple[int, ...]
    environment_product_ranks: tuple[int, ...]
    full_product_ranks: tuple[int, ...]
    output_ranks: tuple[int, ...]
    environment_product_storage_elements: int
    full_product_storage_elements: int
    local_factor_elements: int
    dense_left_elements: int
    dense_right_elements: int
    used_dense_frontier: bool


class TTFrontier:
    """A labelled tensor train representing one frontier array.

    Each core has shape ``(r_left, dimension, r_right)``.  ``labels`` are
    unique hashable objects and their order is the physical TT axis order.
    Public transformations are non-mutating.
    """

    def __init__(self, cores, labels, *, last_round=None):
        self.cores = tuple(np.asarray(core) for core in cores)
        self.labels = tuple(labels)
        if not self.cores:
            raise ValueError("a TT frontier must contain at least one core.")
        if len(self.cores) != len(self.labels):
            raise ValueError("cores and labels must have the same length.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("TT labels must be unique.")
        previous_rank = 1
        for axis, core in enumerate(self.cores):
            if core.ndim != 3:
                raise ValueError(f"TT core {axis} must be three-dimensional.")
            if core.shape[0] != previous_rank:
                raise ValueError(f"TT rank mismatch before core {axis}.")
            if core.shape[1] < 1 or core.shape[2] < 1:
                raise ValueError("TT dimensions and ranks must be positive.")
            previous_rank = core.shape[2]
        if self.cores[0].shape[0] != 1 or self.cores[-1].shape[2] != 1:
            raise ValueError("TT boundary ranks must be one.")
        self.last_round = last_round

    @classmethod
    def from_dense(
        cls,
        array,
        labels=None,
        *,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
    ):
        """Construct a TT by deterministic left-to-right TT-SVD.

        The global tolerance is ``max(atol, rtol * ||array||_F)`` and is split
        equally in squared norm over the TT bonds.  ``max_rank`` is enforced
        independently and may therefore produce an error above that target.
        """

        array = np.asarray(array)
        if array.ndim < 1:
            raise ValueError("a TT frontier array must have at least one axis.")
        if any(dim < 1 for dim in array.shape):
            raise ValueError("TT dimensions must be positive.")
        if labels is None:
            labels = tuple(range(array.ndim))
        else:
            labels = tuple(labels)
        if len(labels) != array.ndim:
            raise ValueError("labels must contain one entry per array axis.")
        if len(set(labels)) != len(labels):
            raise ValueError("TT labels must be unique.")
        max_rank = _validated_max_rank(max_rank)
        rtol, atol = _validated_tolerances(rtol, atol)

        norm = float(np.linalg.norm(array.reshape(-1)))
        tolerance = max(atol, rtol * norm)
        nbonds = max(1, array.ndim - 1)
        local_tolerance_sq = tolerance**2 / nbonds
        work = array
        left_rank = 1
        cores = []
        discarded_weight = 0.0
        original_ranks = []
        for axis, dimension in enumerate(array.shape[:-1]):
            matrix = work.reshape(left_rank * dimension, -1)
            left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
            original_ranks.append(int(singular_values.size))
            rank = _truncation_rank(singular_values, max_rank, local_tolerance_sq)
            discarded_weight += float(np.sum(np.abs(singular_values[rank:]) ** 2))
            cores.append(left[:, :rank].reshape(left_rank, dimension, rank))
            work = singular_values[:rank, None] * right[:rank, :]
            left_rank = rank
        cores.append(work.reshape(left_rank, array.shape[-1], 1))
        ranks = tuple(core.shape[2] for core in cores[:-1])
        norm_sq = norm**2
        diagnostics = TTRoundDiagnostics(
            algorithm="dense_tt_svd",
            original_ranks=tuple(original_ranks),
            ranks=ranks,
            norm=norm,
            discarded_weight=discarded_weight,
            relative_discarded_weight=(
                discarded_weight / norm_sq if norm_sq > 0.0 else 0.0
            ),
            max_rank=max_rank,
            rtol=rtol,
            atol=atol,
            densified_input=True,
        )
        return cls(cores, labels, last_round=diagnostics)

    @classmethod
    def ones(cls, dims, labels=None, *, dtype=float):
        """Return a rank-one TT filled with ones."""

        dims = tuple(int(dim) for dim in dims)
        if not dims or any(dim < 1 for dim in dims):
            raise ValueError("dims must contain positive dimensions.")
        if labels is None:
            labels = tuple(range(len(dims)))
        labels = tuple(labels)
        if len(labels) != len(dims):
            raise ValueError("labels must contain one entry per dimension.")
        return cls(
            [np.ones((1, dim, 1), dtype=dtype) for dim in dims],
            labels,
        )

    @property
    def shape(self):
        return tuple(int(core.shape[1]) for core in self.cores)

    @property
    def dims(self):
        return self.shape

    @property
    def ranks(self):
        return tuple(int(core.shape[2]) for core in self.cores[:-1])

    @property
    def storage_elements(self):
        return int(sum(core.size for core in self.cores))

    @property
    def dense_elements(self):
        return int(np.prod(self.shape, dtype=int))

    @property
    def dtype(self):
        return np.result_type(*(core.dtype for core in self.cores))

    def copy(self):
        return type(self)(
            [core.copy() for core in self.cores],
            self.labels,
            last_round=self.last_round,
        )

    def conjugate_relabel(self, label_map, *, labels=None):
        """Conjugate a TT and simultaneously relabel its physical axes.

        ``label_map`` is applied simultaneously, so it may safely exchange two
        labels.  If ``labels`` is supplied, the result is then put in that
        order by exact TT swaps.  Neither operation materializes the dense
        tensor represented by the train.
        """

        label_map = dict(label_map)
        mapped_labels = tuple(label_map.get(label, label) for label in self.labels)
        if len(set(mapped_labels)) != len(mapped_labels):
            raise ValueError("conjugate relabelling must preserve unique TT labels.")
        result = type(self)(
            [core.conj() for core in self.cores],
            mapped_labels,
            last_round=self.last_round,
        )
        if labels is not None:
            result = result.transpose_labels(labels)
        return result

    def to_dense(self):
        """Materialize the represented frontier array."""

        value = self.cores[0][0]
        for core in self.cores[1:]:
            value = np.tensordot(value, core, axes=(-1, 0))
        return np.asarray(value[..., 0])

    def frobenius_norm(self):
        """Evaluate the Frobenius norm without materializing the TT."""

        gram = np.ones((1, 1), dtype=np.result_type(self.dtype, complex))
        for core in self.cores:
            gram = np.einsum("ab,aic,bid->cd", gram, core.conj(), core, optimize=True)
        norm_sq = float(max(0.0, np.real(np.asarray(gram).reshape(()).item())))
        return sqrt(norm_sq)

    def round(self, *, max_rank=None, rtol=0.0, atol=0.0):
        """Round this TT without constructing its dense frontier array."""

        max_rank = _validated_max_rank(max_rank)
        rtol, atol = _validated_tolerances(rtol, atol)
        cores = [core.copy() for core in self.cores]
        original_ranks = self.ranks
        if len(cores) == 1:
            norm = float(np.linalg.norm(cores[0]))
            diagnostics = TTRoundDiagnostics(
                algorithm="tt_round",
                original_ranks=original_ranks,
                ranks=(),
                norm=norm,
                discarded_weight=0.0,
                relative_discarded_weight=0.0,
                max_rank=max_rank,
                rtol=rtol,
                atol=atol,
                densified_input=False,
            )
            return type(self)(cores, self.labels, last_round=diagnostics)

        # Left orthogonalization makes each subsequent right-to-left SVD's
        # discarded singular values a Frobenius-norm error contribution.
        for axis in range(len(cores) - 1):
            left_rank, dimension, right_rank = cores[axis].shape
            matrix = cores[axis].reshape(left_rank * dimension, right_rank)
            q_factor, r_factor = np.linalg.qr(matrix, mode="reduced")
            rank = q_factor.shape[1]
            cores[axis] = q_factor.reshape(left_rank, dimension, rank)
            cores[axis + 1] = np.tensordot(r_factor, cores[axis + 1], axes=(1, 0))

        norm = float(np.linalg.norm(cores[-1]))
        tolerance = max(atol, rtol * norm)
        local_tolerance_sq = tolerance**2 / max(1, len(cores) - 1)
        discarded_weight = 0.0
        for axis in range(len(cores) - 1, 0, -1):
            left_rank, dimension, right_rank = cores[axis].shape
            matrix = cores[axis].reshape(left_rank, dimension * right_rank)
            left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
            rank = _truncation_rank(singular_values, max_rank, local_tolerance_sq)
            discarded_weight += float(np.sum(np.abs(singular_values[rank:]) ** 2))
            cores[axis] = right[:rank].reshape(rank, dimension, right_rank)
            transfer = left[:, :rank] * singular_values[:rank]
            cores[axis - 1] = np.tensordot(cores[axis - 1], transfer, axes=(2, 0))

        ranks = tuple(int(core.shape[2]) for core in cores[:-1])
        norm_sq = norm**2
        diagnostics = TTRoundDiagnostics(
            algorithm="tt_round",
            original_ranks=original_ranks,
            ranks=ranks,
            norm=norm,
            discarded_weight=discarded_weight,
            relative_discarded_weight=(
                discarded_weight / norm_sq if norm_sq > 0.0 else 0.0
            ),
            max_rank=max_rank,
            rtol=rtol,
            atol=atol,
            densified_input=False,
        )
        return type(self)(cores, self.labels, last_round=diagnostics)

    def _swap_adjacent(self, axis):
        axis = int(axis)
        if axis < 0 or axis + 1 >= len(self.cores):
            raise IndexError("adjacent TT swap axis is out of range.")
        cores = [core.copy() for core in self.cores]
        first, second = cores[axis], cores[axis + 1]
        left_rank, first_dim, middle_rank = first.shape
        if second.shape[0] != middle_rank:
            raise ValueError("invalid TT bond during adjacent swap.")
        second_dim, right_rank = second.shape[1:]
        pair = np.einsum("aib,bjc->aijc", first, second, optimize=True)
        matrix = pair.transpose(0, 2, 1, 3).reshape(
            left_rank * second_dim, first_dim * right_rank
        )
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        rank = singular_values.size
        cores[axis] = left.reshape(left_rank, second_dim, rank)
        cores[axis + 1] = (singular_values[:, None] * right).reshape(
            rank, first_dim, right_rank
        )
        labels = list(self.labels)
        labels[axis], labels[axis + 1] = labels[axis + 1], labels[axis]
        return type(self)(cores, labels)

    def transpose_labels(self, labels):
        """Return an exact TT permutation in the requested label order."""

        labels = tuple(labels)
        if len(labels) != len(self.labels) or set(labels) != set(self.labels):
            raise ValueError("the requested labels must be a permutation.")
        result = self.copy()
        current = list(result.labels)
        for target_axis, label in enumerate(labels):
            source_axis = current.index(label)
            while source_axis > target_axis:
                result = result._swap_adjacent(source_axis - 1)
                current[source_axis - 1], current[source_axis] = (
                    current[source_axis],
                    current[source_axis - 1],
                )
                source_axis -= 1
        return result

    def _insert_ones_axis(self, axis, label, dimension):
        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("inserted TT dimensions must be positive.")
        if label in self.labels:
            raise ValueError("cannot insert a duplicate TT label.")
        axis = int(axis)
        if axis < 0 or axis > len(self.cores):
            raise IndexError("TT insertion axis is out of range.")
        bond_rank = (
            self.cores[axis].shape[0]
            if axis < len(self.cores)
            else self.cores[-1].shape[2]
        )
        identity = np.eye(bond_rank, dtype=self.dtype)
        core = np.repeat(identity[:, None, :], dimension, axis=1)
        cores = list(self.cores)
        labels = list(self.labels)
        cores.insert(axis, core)
        labels.insert(axis, label)
        return type(self)(cores, labels)

    def embed(self, labels, dims):
        """Embed this TT in a larger labelled space using constant-one axes."""

        labels = tuple(labels)
        dims = tuple(int(dim) for dim in dims)
        if len(labels) != len(dims) or len(set(labels)) != len(labels):
            raise ValueError("embedding labels and dims must be unique and aligned.")
        if not set(self.labels).issubset(labels):
            raise ValueError("embedding labels must include all current labels.")
        dimension_map = dict(zip(labels, dims))
        for label, dimension in zip(self.labels, self.shape):
            if dimension_map[label] != dimension:
                raise ValueError(f"dimension mismatch for TT label {label!r}.")

        existing_order = tuple(label for label in labels if label in self.labels)
        result = self.transpose_labels(existing_order)
        for axis, (label, dimension) in enumerate(zip(labels, dims)):
            if label not in result.labels:
                result = result._insert_ones_axis(axis, label, dimension)
        return result

    def hadamard(self, other):
        """Return the exact elementwise product of two aligned TTs."""

        if not isinstance(other, TTFrontier):
            return NotImplemented
        if self.labels != other.labels or self.shape != other.shape:
            raise ValueError("Hadamard TT operands must have aligned labels and dims.")
        cores = []
        for left, right in zip(self.cores, other.cores):
            product = np.einsum("aib,cid->acibd", left, right, optimize=True)
            cores.append(
                product.reshape(
                    left.shape[0] * right.shape[0],
                    left.shape[1],
                    left.shape[2] * right.shape[2],
                )
            )
        return type(self)(cores, self.labels)

    def add(self, other):
        """Return the exact direct-sum TT for two aligned frontiers."""
        if not isinstance(other, TTFrontier):
            return NotImplemented
        if self.labels != other.labels or self.shape != other.shape:
            raise ValueError("TT sums require aligned labels and dimensions.")
        if len(self.cores) == 1:
            return type(self)([self.cores[0] + other.cores[0]], self.labels)
        dtype = np.result_type(self.dtype, other.dtype)
        cores = [
            np.concatenate(
                (
                    np.asarray(self.cores[0], dtype=dtype),
                    np.asarray(other.cores[0], dtype=dtype),
                ),
                axis=2,
            )
        ]
        for left, right in zip(self.cores[1:-1], other.cores[1:-1]):
            core = np.zeros(
                (
                    left.shape[0] + right.shape[0],
                    left.shape[1],
                    left.shape[2] + right.shape[2],
                ),
                dtype=dtype,
            )
            core[: left.shape[0], :, : left.shape[2]] = left
            core[left.shape[0] :, :, left.shape[2] :] = right
            cores.append(core)
        cores.append(
            np.concatenate(
                (
                    np.asarray(self.cores[-1], dtype=dtype),
                    np.asarray(other.cores[-1], dtype=dtype),
                ),
                axis=0,
            )
        )
        return type(self)(cores, self.labels)

    def sum_over(self, labels: Iterable[Hashable]):
        """Exactly sum the requested variables out of this TT."""

        result = self.copy()
        for label in tuple(labels):
            if label not in result.labels:
                raise ValueError(f"cannot eliminate absent TT label {label!r}.")
            if len(result.cores) == 1:
                raise ValueError("cannot eliminate the final TT axis.")
            axis = result.labels.index(label)
            cores = [core.copy() for core in result.cores]
            current = np.sum(cores[axis], axis=1)
            if axis + 1 < len(cores):
                cores[axis + 1] = np.tensordot(current, cores[axis + 1], axes=(1, 0))
            else:
                cores[axis - 1] = np.tensordot(cores[axis - 1], current, axes=(2, 0))
            del cores[axis]
            kept_labels = list(result.labels)
            del kept_labels[axis]
            result = type(self)(cores, kept_labels)
        return result


class TTMPOFrontier:
    """Approximate MPO frontier contractor using labelled tensor trains.

    The constructor is compatible with :class:`MPOFrontier`, with additional
    TT rounding controls.  ``absorption='structured'`` is the scalable path;
    ``absorption='dense'`` is an explicit validation fallback that first
    materializes the source and target frontier arrays.

    ``max_rank``, ``rtol``, and ``atol`` control boundary-message rounding.
    Site-local transfer factors are represented exactly by default.  Setting
    ``transfer_max_rank`` and/or the transfer tolerances also compresses those
    factors, substantially lowering transient TT-product ranks at the price of
    a separately diagnosed local approximation.

    Matrix-free local hole actions use the same labelled-factor machinery.
    The returned local vector is dense, but neither frontier message is
    materialized.  ``allow_dense=True`` selects an explicit reference path.
    """

    def __init__(
        self,
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
        *,
        paired_sites=None,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
        transfer_max_rank=None,
        transfer_rtol=0.0,
        transfer_atol=0.0,
        absorption="structured",
        optimize="greedy",
    ):
        self.max_rank = _validated_max_rank(max_rank)
        self.rtol, self.atol = _validated_tolerances(rtol, atol)
        self.transfer_max_rank = _validated_max_rank(transfer_max_rank)
        self.transfer_rtol, self.transfer_atol = _validated_tolerances(
            transfer_rtol, transfer_atol
        )
        self.absorption = str(absorption).lower().replace("-", "_")
        if self.absorption not in {"structured", "dense"}:
            raise ValueError("absorption must be 'structured' or 'dense'.")
        self.optimize = optimize
        self._metadata = MPOFrontier(
            dims,
            physical_sites,
            tensor_shapes,
            mpo_tensors,
            paired_sites=paired_sites,
            optimize=optimize,
        )
        self.dims = self._metadata.dims
        self.physical_groups = self._metadata.physical_groups
        self.tensor_shapes = self._metadata.tensor_shapes
        self.mpo_tensors = self._metadata.mpo_tensors
        self.paired_sites = self._metadata.paired_sites
        self.nsites = self._metadata.nsites
        self.frontier_sites = self._metadata.frontier_sites
        self.virtual_bonds = self._metadata.virtual_bonds
        self.mpo_bonds = self._metadata.mpo_bonds
        self._advance_diagnostics: list[TTAdvanceDiagnostics] = []
        self.last_hole_diagnostics: TTHoleDiagnostics | None = None

    @property
    def plan_count(self):
        return self._metadata.plan_count

    def message_shape(self, cut):
        return self._metadata.message_shape(cut)

    def dense_message_elements(self, cut):
        """Number of elements needed by the corresponding dense message."""

        return self._metadata.message_elements(cut)

    @property
    def dense_peak_message_elements(self):
        return self._metadata.peak_message_elements

    @property
    def dense_total_message_elements(self):
        return self._metadata.total_message_elements

    def message_labels(self, cut):
        """Return the deterministic TT variable order at ``cut``."""

        return self._metadata._message_labels(int(cut))

    def reset_diagnostics(self):
        self._advance_diagnostics.clear()

    @property
    def diagnostics(self):
        advances = tuple(self._advance_diagnostics)
        if not advances:
            return TTContractionDiagnostics((), 0.0, 0.0, 0, 0, 0, 0, 0)
        return TTContractionDiagnostics(
            advances=advances,
            total_discarded_weight=float(
                sum(item.discarded_weight for item in advances)
            ),
            max_relative_discarded_weight=float(
                max(item.relative_discarded_weight for item in advances)
            ),
            peak_message_storage_elements=max(
                max(item.source_storage_elements, item.target_storage_elements)
                for item in advances
            ),
            peak_dense_message_elements=max(
                max(item.dense_source_elements, item.dense_target_elements)
                for item in advances
            ),
            peak_product_storage_elements=max(
                item.product_storage_elements for item in advances
            ),
            peak_local_factor_elements=max(
                item.local_factor_elements for item in advances
            ),
            dense_frontier_absorptions=sum(
                int(item.used_dense_frontier) for item in advances
            ),
        )

    def left_boundary(self):
        return TTFrontier.ones(
            self.message_shape(0),
            self.message_labels(0),
            dtype=np.result_type(*(tensor.dtype for tensor in self.mpo_tensors)),
        )

    def right_boundary(self):
        return TTFrontier.ones(
            self.message_shape(self.nsites),
            self.message_labels(self.nsites),
            dtype=np.result_type(*(tensor.dtype for tensor in self.mpo_tensors)),
        )

    @staticmethod
    def _label_dimensions(labels_and_shapes):
        dimensions = {}
        for labels, shape in labels_and_shapes:
            if len(labels) != len(shape):
                raise ValueError("factor labels and shape are inconsistent.")
            for label, dimension in zip(labels, shape):
                dimension = int(dimension)
                previous = dimensions.setdefault(label, dimension)
                if previous != dimension:
                    raise ValueError(f"inconsistent dimension for label {label!r}.")
        return dimensions

    def _local_factor(self, site, union_labels, tensors):
        bra_labels = self._metadata._tensor_labels(site, bra=True)
        ket_labels = self._metadata._tensor_labels(site, bra=False)
        mpo_labels = (
            self._metadata.operator_bonds[site],
            self._metadata.operator_bonds[site + 1],
            self._metadata.bra_physical[site],
            self._metadata._ket_label(site),
        )
        local_set = set(bra_labels) | set(ket_labels) | set(mpo_labels)
        local_labels = tuple(label for label in union_labels if label in local_set)
        tensor = tensors[site]
        value = contract(
            tensor.conj(),
            bra_labels,
            tensor,
            ket_labels,
            self.mpo_tensors[site],
            mpo_labels,
            local_labels,
            optimize=self.optimize,
        )
        value = np.asarray(value)
        return (
            TTFrontier.from_dense(
                value,
                local_labels,
                max_rank=self.transfer_max_rank,
                rtol=self.transfer_rtol,
                atol=self.transfer_atol,
            ),
            local_labels,
            int(value.size),
        )

    def _structured_advance(self, message, tensors, site, direction):
        if not isinstance(message, TTFrontier):
            raise TypeError("structured absorption requires a TTFrontier message.")
        source_cut, target_cut = (
            (site, site + 1) if direction == "left" else (site + 1, site)
        )
        source_labels = self.message_labels(source_cut)
        target_labels = self.message_labels(target_cut)
        if message.labels != source_labels or message.shape != self.message_shape(
            source_cut
        ):
            raise ValueError(
                "message labels or dimensions do not match the source cut."
            )

        bra_labels = self._metadata._tensor_labels(site, bra=True)
        ket_labels = self._metadata._tensor_labels(site, bra=False)
        mpo_labels = (
            self._metadata.operator_bonds[site],
            self._metadata.operator_bonds[site + 1],
            self._metadata.bra_physical[site],
            self._metadata._ket_label(site),
        )
        local_labels_in_order = tuple(
            dict.fromkeys((*bra_labels, *ket_labels, *mpo_labels))
        )
        source_set, target_set = set(source_labels), set(target_labels)
        local_only = tuple(
            label
            for label in local_labels_in_order
            if label not in source_set and label not in target_set
        )
        source_only = tuple(label for label in source_labels if label not in target_set)
        preferred_union = tuple(
            dict.fromkeys((*source_only, *local_only, *target_labels))
        )
        if (
            tuple(label for label in preferred_union if label in source_set)
            == source_labels
        ):
            union_labels = preferred_union
        else:
            # The reverse direction can interleave dying and retained physical
            # variables.  Preserve the source order and exactly transpose the
            # target TT after elimination.
            target_only = tuple(
                label for label in target_labels if label not in source_set
            )
            union_labels = tuple(
                dict.fromkeys((*source_labels, *local_only, *target_only))
            )

        dimension_map = self._label_dimensions(
            (
                (source_labels, self.message_shape(source_cut)),
                (target_labels, self.message_shape(target_cut)),
                (bra_labels, self.tensor_shapes[site]),
                (ket_labels, self.tensor_shapes[site]),
                (mpo_labels, self.mpo_tensors[site].shape),
            )
        )
        union_dims = tuple(dimension_map[label] for label in union_labels)
        local_factor, _local_labels, local_factor_elements = self._local_factor(
            site, union_labels, tensors
        )
        local_round_diagnostics = local_factor.last_round
        local_factor_storage = local_factor.storage_elements
        local_factor_ranks = local_factor.ranks
        message_factor = message.embed(union_labels, union_dims)
        local_factor = local_factor.embed(union_labels, union_dims)
        product = message_factor.hadamard(local_factor)
        product_ranks = product.ranks
        product_storage = product.storage_elements
        eliminated = tuple(label for label in union_labels if label not in target_set)
        target = product.sum_over(eliminated)
        if target.labels != target_labels:
            target = target.transpose_labels(target_labels)
        target = target.round(
            max_rank=self.max_rank,
            rtol=self.rtol,
            atol=self.atol,
        )
        round_diagnostics = target.last_round
        advance = TTAdvanceDiagnostics(
            direction=direction,
            site=site,
            source_cut=source_cut,
            target_cut=target_cut,
            absorption="structured",
            source_ranks=message.ranks,
            product_ranks=product_ranks,
            target_ranks=target.ranks,
            source_storage_elements=message.storage_elements,
            product_storage_elements=product_storage,
            target_storage_elements=target.storage_elements,
            dense_source_elements=message.dense_elements,
            dense_target_elements=int(np.prod(target.shape, dtype=int)),
            local_factor_elements=local_factor_elements,
            local_factor_storage_elements=local_factor_storage,
            local_factor_ranks=local_factor_ranks,
            local_factor_discarded_weight=local_round_diagnostics.discarded_weight,
            message_discarded_weight=round_diagnostics.discarded_weight,
            discarded_weight=(
                local_round_diagnostics.discarded_weight
                + round_diagnostics.discarded_weight
            ),
            relative_discarded_weight=max(
                local_round_diagnostics.relative_discarded_weight,
                round_diagnostics.relative_discarded_weight,
            ),
            used_dense_frontier=False,
        )
        self._advance_diagnostics.append(advance)
        return target

    def _dense_advance(self, message, tensors, site, direction):
        if not isinstance(message, TTFrontier):
            raise TypeError("dense absorption requires a TTFrontier message.")
        source_cut, target_cut = (
            (site, site + 1) if direction == "left" else (site + 1, site)
        )
        dense_source = message.to_dense()
        if direction == "left":
            dense_target = self._metadata.advance_left(dense_source, tensors, site)
        else:
            dense_target = self._metadata.advance_right(dense_source, tensors, site)
        target = TTFrontier.from_dense(
            dense_target,
            self.message_labels(target_cut),
            max_rank=self.max_rank,
            rtol=self.rtol,
            atol=self.atol,
        )
        round_diagnostics = target.last_round
        self._advance_diagnostics.append(
            TTAdvanceDiagnostics(
                direction=direction,
                site=site,
                source_cut=source_cut,
                target_cut=target_cut,
                absorption="dense",
                source_ranks=message.ranks,
                product_ranks=(),
                target_ranks=target.ranks,
                source_storage_elements=message.storage_elements,
                product_storage_elements=int(np.size(dense_target)),
                target_storage_elements=target.storage_elements,
                dense_source_elements=int(np.size(dense_source)),
                dense_target_elements=int(np.size(dense_target)),
                local_factor_elements=0,
                local_factor_storage_elements=0,
                local_factor_ranks=(),
                local_factor_discarded_weight=0.0,
                message_discarded_weight=round_diagnostics.discarded_weight,
                discarded_weight=round_diagnostics.discarded_weight,
                relative_discarded_weight=round_diagnostics.relative_discarded_weight,
                used_dense_frontier=True,
            )
        )
        return target

    def _advance(self, message, tensors, site, direction):
        site = int(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site is out of range.")
        if self.absorption == "dense":
            return self._dense_advance(message, tensors, site, direction)
        return self._structured_advance(message, tensors, site, direction)

    def advance_left(self, message, tensors, site):
        return self._advance(message, tensors, site, "left")

    def advance_right(self, message, tensors, site):
        return self._advance(message, tensors, site, "right")

    def build_left(self, tensors, *, reset_diagnostics=True):
        if reset_diagnostics:
            self.reset_diagnostics()
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        for site in range(self.nsites):
            messages[site + 1] = self.advance_left(messages[site], tensors, site)
        return messages

    def build_right(self, tensors, *, reset_diagnostics=True):
        if reset_diagnostics:
            self.reset_diagnostics()
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        for site in range(self.nsites - 1, -1, -1):
            messages[site] = self.advance_right(messages[site + 1], tensors, site)
        return messages

    def scalar(self, tensors):
        self.reset_diagnostics()
        value = self.left_boundary()
        for site in range(self.nsites):
            value = self.advance_left(value, tensors, site)
        value = value.to_dense()
        return np.asarray(value).reshape(()).item()

    @staticmethod
    def _factor_union(*factors):
        return tuple(
            dict.fromkeys(label for factor in factors for label in factor.labels)
        )

    def _aligned_hadamard(self, left, right, dimension_map):
        union_labels = self._factor_union(left, right)
        union_dims = tuple(dimension_map[label] for label in union_labels)
        return left.embed(union_labels, union_dims).hadamard(
            right.embed(union_labels, union_dims)
        )

    def _hole_local_factor(
        self,
        site,
        vector,
        external_labels,
        *,
        mpo_tensor=None,
    ):
        bra_labels = (
            self._metadata.bra_bonds[site],
            self._metadata.bra_bonds[site + 1],
            *(
                self._metadata.bra_physical[index]
                for index in self.physical_groups[site]
            ),
        )
        ket_labels = (
            self._metadata.ket_bonds[site],
            self._metadata.ket_bonds[site + 1],
            *(
                (
                    self._metadata.ket_physical[index]
                    if index in self.paired_sites
                    else self._metadata.local_ket_physical[index]
                )
                for index in self.physical_groups[site]
            ),
        )
        own_ket_label = (
            self._metadata.ket_physical[site]
            if site in self.paired_sites
            else self._metadata.local_ket_physical[site]
        )
        mpo_labels = (
            self._metadata.operator_bonds[site],
            self._metadata.operator_bonds[site + 1],
            self._metadata.bra_physical[site],
            own_ket_label,
        )
        identity_factors = []
        for physical_site in self.physical_groups[site][1:]:
            if physical_site not in self.paired_sites:
                identity_factors.append(
                    (
                        np.eye(self.dims[physical_site]),
                        (
                            self._metadata.bra_physical[physical_site],
                            self._metadata.local_ket_physical[physical_site],
                        ),
                    )
                )

        local_input_labels = tuple(
            dict.fromkeys(
                (
                    *mpo_labels,
                    *(
                        label
                        for _identity, labels in identity_factors
                        for label in labels
                    ),
                    *ket_labels,
                )
            )
        )
        external_set = set(external_labels) | set(bra_labels)
        local_labels = tuple(
            label for label in local_input_labels if label in external_set
        )
        if mpo_tensor is None:
            mpo_tensor = self.mpo_tensors[site]
        mpo_tensor = np.asarray(mpo_tensor)
        if mpo_tensor.shape != self.mpo_tensors[site].shape:
            raise ValueError("the local MPO tensor has an invalid shape.")
        arguments = [mpo_tensor, mpo_labels]
        for identity, labels in identity_factors:
            arguments.extend((identity, labels))
        arguments.extend(
            (
                np.asarray(vector).reshape(self.tensor_shapes[site]),
                ket_labels,
                local_labels,
            )
        )
        value = np.asarray(contract(*arguments, optimize=self.optimize))
        return (
            TTFrontier.from_dense(value, local_labels),
            bra_labels,
            int(value.size),
        )

    def _structured_hole_action(
        self,
        site,
        left,
        right,
        vector,
        *,
        mpo_tensor=None,
        method="structured",
    ):
        expected_left = (self.message_labels(site), self.message_shape(site))
        expected_right = (
            self.message_labels(site + 1),
            self.message_shape(site + 1),
        )
        if (left.labels, left.shape) != expected_left:
            raise ValueError("left TT message does not match the local cut.")
        if (right.labels, right.shape) != expected_right:
            raise ValueError("right TT message does not match the local cut.")

        local, output_labels, local_factor_elements = self._hole_local_factor(
            site,
            vector,
            (*left.labels, *right.labels),
            mpo_tensor=mpo_tensor,
        )
        dimension_map = self._label_dimensions(
            (
                (left.labels, left.shape),
                (right.labels, right.shape),
                (local.labels, local.shape),
                (output_labels, self.tensor_shapes[site]),
            )
        )

        environment = self._aligned_hadamard(left, right, dimension_map)
        environment_ranks = environment.ranks
        environment_storage = environment.storage_elements
        needed = set(local.labels) | set(output_labels)
        environment = environment.sum_over(
            label for label in environment.labels if label not in needed
        )
        product = self._aligned_hadamard(environment, local, dimension_map)
        product_ranks = product.ranks
        product_storage = product.storage_elements
        output_set = set(output_labels)
        output = product.sum_over(
            label for label in product.labels if label not in output_set
        )
        if output.labels != output_labels:
            output = output.transpose_labels(output_labels)
        self.last_hole_diagnostics = TTHoleDiagnostics(
            site=site,
            method=method,
            left_ranks=left.ranks,
            right_ranks=right.ranks,
            environment_product_ranks=environment_ranks,
            full_product_ranks=product_ranks,
            output_ranks=output.ranks,
            environment_product_storage_elements=environment_storage,
            full_product_storage_elements=product_storage,
            local_factor_elements=local_factor_elements,
            dense_left_elements=left.dense_elements,
            dense_right_elements=right.dense_elements,
            used_dense_frontier=False,
        )
        return output.to_dense().reshape(-1)

    def adjoint_message(self, message, cut):
        """Conjugate a message and exchange its bra and ket variables.

        Virtual bra/ket bonds and every explicitly paired physical frontier
        variable are exchanged.  MPO bonds and unpaired physical variables
        retain their labels.  The returned TT uses the canonical label order
        for ``cut`` and no dense frontier is formed.
        """

        cut = int(cut)
        if cut < 0 or cut > self.nsites:
            raise IndexError("cut is out of range.")
        if not isinstance(message, TTFrontier):
            raise TypeError("adjoint messages require a TTFrontier.")
        expected = (self.message_labels(cut), self.message_shape(cut))
        if (message.labels, message.shape) != expected:
            raise ValueError("TT message does not match the requested cut.")

        label_map = {}
        for bra_label, ket_label in zip(
            self._metadata.bra_bonds,
            self._metadata.ket_bonds,
        ):
            label_map[bra_label] = ket_label
            label_map[ket_label] = bra_label
        for physical_site in self.paired_sites:
            bra_label = self._metadata.bra_physical[physical_site]
            ket_label = self._metadata.ket_physical[physical_site]
            label_map[bra_label] = ket_label
            label_map[ket_label] = bra_label
        return message.conjugate_relabel(label_map, labels=expected[0])

    def hole_action(self, site, left, right, vector, *, allow_dense=False):
        """Apply a local hole operator to a dense local vector.

        The default contracts compressed left/right messages as labelled TT
        factors and materializes only the final local vector.  The opt-in
        ``allow_dense`` path is useful for small-system validation.
        """

        site = int(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site is out of range.")
        if not isinstance(left, TTFrontier) or not isinstance(right, TTFrontier):
            raise TypeError("hole actions require TTFrontier messages.")
        if not allow_dense:
            return self._structured_hole_action(site, left, right, vector)
        value = self._metadata.hole_action(
            site, left.to_dense(), right.to_dense(), vector
        )
        self.last_hole_diagnostics = TTHoleDiagnostics(
            site=site,
            method="dense",
            left_ranks=left.ranks,
            right_ranks=right.ranks,
            environment_product_ranks=(),
            full_product_ranks=(),
            output_ranks=(),
            environment_product_storage_elements=0,
            full_product_storage_elements=int(np.size(value)),
            local_factor_elements=0,
            dense_left_elements=left.dense_elements,
            dense_right_elements=right.dense_elements,
            used_dense_frontier=True,
        )
        return value

    def hole_adjoint_action(self, site, left, right, vector, *, allow_dense=False):
        """Apply the exact numerical adjoint of :meth:`hole_action`.

        The structured path remains correct when independently rounded TT
        messages make the forward local operator non-Hermitian.  It
        conjugates and bra/ket-relabels both messages and uses the local MPO
        tensor with its physical indices transposed and conjugated.  Only the
        final local output vector is materialized.

        ``allow_dense=True`` is an explicit small-system reference path.
        """

        site = int(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site is out of range.")
        if not isinstance(left, TTFrontier) or not isinstance(right, TTFrontier):
            raise TypeError("hole adjoint actions require TTFrontier messages.")
        if not allow_dense:
            adjoint_left = self.adjoint_message(left, site)
            adjoint_right = self.adjoint_message(right, site + 1)
            adjoint_mpo = self.mpo_tensors[site].conj().swapaxes(-2, -1)
            return self._structured_hole_action(
                site,
                adjoint_left,
                adjoint_right,
                vector,
                mpo_tensor=adjoint_mpo,
                method="structured_adjoint",
            )

        matrix = self._metadata.hole_matrix(
            site,
            left.to_dense(),
            right.to_dense(),
        )
        value = matrix.T.conj() @ np.asarray(vector).reshape(-1)
        self.last_hole_diagnostics = TTHoleDiagnostics(
            site=site,
            method="dense_adjoint",
            left_ranks=left.ranks,
            right_ranks=right.ranks,
            environment_product_ranks=(),
            full_product_ranks=(),
            output_ranks=(),
            environment_product_storage_elements=0,
            full_product_storage_elements=int(np.size(value)),
            local_factor_elements=0,
            dense_left_elements=left.dense_elements,
            dense_right_elements=right.dense_elements,
            used_dense_frontier=True,
        )
        return np.asarray(value).reshape(-1)


def _product_mpo(dims, local_operators):
    """Build a bond-one MPO from a map of site-local operators."""

    dtype = np.result_type(
        *[np.asarray(operator).dtype for operator in local_operators.values()],
        float,
    )
    tensors = []
    for site, dim in enumerate(dims):
        operator = local_operators.get(site)
        if operator is None:
            operator = np.eye(dim, dtype=dtype)
        tensors.append(np.asarray(operator, dtype=dtype)[None, None])
    return MPO(tensors)


def _validated_local_qns(dims, local_qns):
    if local_qns is None:
        return None
    local_qns = tuple(
        tuple(tuple(int(value) for value in charge) for charge in site)
        for site in local_qns
    )
    if len(local_qns) != len(dims):
        raise ValueError("local_qns must contain one entry per physical site.")
    if any(len(site) != dim for site, dim in zip(local_qns, dims)):
        raise ValueError("local_qns dimensions do not match the local dimensions.")
    ranks = {len(charge) for site in local_qns for charge in site}
    if len(ranks) != 1:
        raise ValueError("all local_qns charges must have the same rank.")
    return local_qns


def _charge_difference(left, right):
    return tuple(int(a) - int(b) for a, b in zip(left, right))


def _charge_resolved_two_site_schmidt(matrix, left_qns, right_qns):
    """Split a two-site operator without mixing Abelian transfer sectors."""
    left_dim = len(left_qns)
    right_dim = len(right_qns)
    row_groups = {}
    column_groups = {}
    for bra in range(left_dim):
        for ket in range(left_dim):
            transfer = _charge_difference(left_qns[bra], left_qns[ket])
            row_groups.setdefault(transfer, []).append(bra * left_dim + ket)
    for bra in range(right_dim):
        for ket in range(right_dim):
            transfer = _charge_difference(right_qns[bra], right_qns[ket])
            column_groups.setdefault(transfer, []).append(bra * right_dim + ket)

    scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
    threshold = np.finfo(float).eps * max(matrix.shape) * scale
    components = []
    for left_transfer in sorted(row_groups):
        rows = np.asarray(row_groups[left_transfer], dtype=np.intp)
        for right_transfer in sorted(column_groups):
            columns = np.asarray(column_groups[right_transfer], dtype=np.intp)
            block = matrix[np.ix_(rows, columns)]
            if not np.any(np.abs(block) > threshold):
                continue
            left, singular_values, right = np.linalg.svd(
                block,
                full_matrices=False,
            )
            rank = int(np.count_nonzero(singular_values > threshold))
            for component in range(rank):
                left_vector = np.zeros(matrix.shape[0], dtype=left.dtype)
                right_vector = np.zeros(matrix.shape[1], dtype=right.dtype)
                left_vector[rows] = left[:, component] * singular_values[component]
                right_vector[columns] = right[component]
                components.append(
                    (
                        left_vector.reshape(left_dim, left_dim),
                        right_vector.reshape(right_dim, right_dim),
                        left_transfer,
                        right_transfer,
                    )
                )
    return tuple(components)


def _term_product_mpos(dims, term, *, local_qns=None):
    """Expand one finite-support term into bond-one product MPOs."""

    sites = tuple(term.sites)
    if len(sites) == 1:
        site = sites[0]
        dim = dims[site]
        return (_product_mpo(dims, {site: term.operator.reshape(dim, dim)}),)
    if len(sites) == 2:
        left_site, right_site = sites
        left_dim, right_dim = dims[left_site], dims[right_site]
        matrix = (
            np.asarray(term.operator)
            .reshape(left_dim, right_dim, left_dim, right_dim)
            .transpose(0, 2, 1, 3)
            .reshape(left_dim * left_dim, right_dim * right_dim)
        )
        if local_qns is not None:
            components = _charge_resolved_two_site_schmidt(
                matrix,
                local_qns[left_site],
                local_qns[right_site],
            )
            return tuple(
                _product_mpo(
                    dims,
                    {
                        left_site: left_operator,
                        right_site: right_operator,
                    },
                )
                for left_operator, right_operator, _left_transfer, _right_transfer
                in components
            )
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if singular_values.size:
            threshold = (
                np.finfo(singular_values.dtype).eps
                * max(matrix.shape)
                * singular_values[0]
            )
            rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        else:  # pragma: no cover - positive local dimensions prevent this
            rank = 1
        return tuple(
            _product_mpo(
                dims,
                {
                    left_site: (
                        left[:, component] * singular_values[component]
                    ).reshape(left_dim, left_dim),
                    right_site: right[component].reshape(right_dim, right_dim),
                },
            )
            for component in range(rank)
        )

    # General exact fallback.  Local Hamiltonian supports are normally one or
    # two sites; expanding matrix units keeps larger supports correct without
    # introducing an unprotected multi-channel MPO bond.
    support_dims = tuple(dims[site] for site in sites)
    operator = np.asarray(term.operator).reshape(support_dims + support_dims)
    components = []
    threshold = np.finfo(float).eps * max(1.0, float(np.max(np.abs(operator))))
    for index in np.ndindex(operator.shape):
        coefficient = operator[index]
        if abs(coefficient) <= threshold:
            continue
        bra = index[: len(sites)]
        ket = index[len(sites) :]
        local = {}
        for position, site in enumerate(sites):
            matrix = np.zeros((dims[site], dims[site]), dtype=operator.dtype)
            matrix[bra[position], ket[position]] = (
                coefficient if position == 0 else 1.0
            )
            local[site] = matrix
        components.append(_product_mpo(dims, local))
    return tuple(components)


class TermwiseTTMPOFrontier(TTMPOFrontier):
    """Channel-protected TT contraction of a sum of local terms.

    A generic Hamiltonian MPO carries a finite-state channel index.  Applying
    an ordinary TT rank cap across that index can discard the accumulated
    Hamiltonian channel and spuriously drive the numerator to zero.  This
    contractor expands each local term into bond-one operator-Schmidt product
    MPOs.  Every component has an independent TT frontier, so boundary
    rounding cannot mix or remove Hamiltonian automaton sectors.  When
    ``local_qns`` are supplied, the operator-Schmidt split is performed
    independently in each Abelian charge-transfer block.  Local transfers
    then remain exact and only the propagated boundary messages may be
    rounded.
    """

    def __init__(
        self,
        hamiltonian,
        physical_sites,
        tensor_shapes,
        *,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
        transfer_max_rank=None,
        transfer_rtol=0.0,
        transfer_atol=0.0,
        absorption="structured",
        optimize="greedy",
        local_qns=None,
    ):
        self.hamiltonian = hamiltonian
        self.dims = tuple(hamiltonian.dims)
        self.physical_groups = tuple(tuple(sites) for sites in physical_sites)
        self.tensor_shapes = tuple(tuple(shape) for shape in tensor_shapes)
        self.nsites = len(self.dims)
        self.absorption = str(absorption).lower().replace("-", "_")
        self.max_rank = _validated_max_rank(max_rank)
        self.transfer_max_rank = _validated_max_rank(transfer_max_rank)
        self.rtol, self.atol = _validated_tolerances(rtol, atol)
        self.transfer_rtol, self.transfer_atol = _validated_tolerances(
            transfer_rtol, transfer_atol
        )
        self.optimize = optimize
        self.local_qns = _validated_local_qns(self.dims, local_qns)
        if self.local_qns is not None and (
            self.transfer_max_rank is not None
            or self.transfer_rtol != 0.0
            or self.transfer_atol != 0.0
        ):
            raise ValueError(
                "charge-resolved termwise TT requires exact local transfers; "
                "truncate only boundary messages with max_rank/rtol/atol."
            )
        self._groups = []
        self._group_supports = []
        self._engines = []
        if hamiltonian.constant != 0.0:
            constant_mpo = _product_mpo(
                self.dims,
                {0: hamiltonian.constant * np.eye(self.dims[0])},
            )
            self._engines.append(
                TTMPOFrontier(
                    self.dims,
                    self.physical_groups,
                    self.tensor_shapes,
                    constant_mpo.tensors,
                    paired_sites=(),
                    max_rank=self.max_rank,
                    rtol=self.rtol,
                    atol=self.atol,
                    transfer_max_rank=self.transfer_max_rank,
                    transfer_rtol=self.transfer_rtol,
                    transfer_atol=self.transfer_atol,
                    absorption=self.absorption,
                    optimize=self.optimize,
                )
            )
            self._groups.append(((0,), True))
            self._group_supports.append((0,))
        for term in hamiltonian.local_terms:
            group = []
            for mpo in _term_product_mpos(
                self.dims,
                term,
                local_qns=self.local_qns,
            ):
                engine = TTMPOFrontier(
                    self.dims,
                    self.physical_groups,
                    self.tensor_shapes,
                    mpo.tensors,
                    paired_sites=term.sites,
                    max_rank=self.max_rank,
                    rtol=self.rtol,
                    atol=self.atol,
                    transfer_max_rank=self.transfer_max_rank,
                    transfer_rtol=self.transfer_rtol,
                    transfer_atol=self.transfer_atol,
                    absorption=self.absorption,
                    optimize=self.optimize,
                )
                group.append(len(self._engines))
                self._engines.append(engine)
            self._groups.append((tuple(group), np.allclose(
                term.operator, term.operator.T.conj()
            )))
            self._group_supports.append(tuple(term.sites))
        for product in hamiltonian.products:
            if product.coefficient == 0:
                continue
            mpo = _operator_string_product_mpo(self.dims, product)
            self._engines.append(
                TTMPOFrontier(
                    self.dims,
                    self.physical_groups,
                    self.tensor_shapes,
                    mpo.tensors,
                    paired_sites=_offdiagonal_product_sites(product),
                    max_rank=self.max_rank,
                    rtol=self.rtol,
                    atol=self.atol,
                    transfer_max_rank=self.transfer_max_rank,
                    transfer_rtol=self.transfer_rtol,
                    transfer_atol=self.transfer_atol,
                    absorption=self.absorption,
                    optimize=self.optimize,
                )
            )
            self._groups.append(((len(self._engines) - 1,), False))
            self._group_supports.append(tuple(product.sites))
        if not self._engines:
            raise ValueError("a termwise TT Hamiltonian requires at least one term.")
        self.frontier_sites = self._engines[0].frontier_sites
        self.virtual_bonds = self._engines[0].virtual_bonds
        self.mpo_bonds = tuple(1 for _ in range(self.nsites + 1))
        self.last_hole_diagnostics = None

    @property
    def plan_count(self):
        return sum(engine.plan_count for engine in self._engines)

    def dense_message_elements(self, cut):
        return sum(engine.dense_message_elements(cut) for engine in self._engines)

    @property
    def dense_peak_message_elements(self):
        return max(
            self.dense_message_elements(cut) for cut in range(self.nsites + 1)
        )

    @property
    def dense_total_message_elements(self):
        return sum(
            self.dense_message_elements(cut) for cut in range(self.nsites + 1)
        )

    @property
    def diagnostics(self):
        items = [engine.diagnostics for engine in self._engines]
        advances = tuple(
            advance for item in items for advance in item.advances
        )
        if not advances:
            return TTContractionDiagnostics((), 0.0, 0.0, 0, 0, 0, 0, 0)
        return TTContractionDiagnostics(
            advances=advances,
            total_discarded_weight=float(
                sum(item.total_discarded_weight for item in items)
            ),
            max_relative_discarded_weight=float(
                max(item.max_relative_discarded_weight for item in items)
            ),
            peak_message_storage_elements=max(
                sum(
                    engine.diagnostics.peak_message_storage_elements
                    for engine in self._engines
                ),
                1,
            ),
            peak_dense_message_elements=self.dense_peak_message_elements,
            peak_product_storage_elements=max(
                item.peak_product_storage_elements for item in items
            ),
            peak_local_factor_elements=max(
                item.peak_local_factor_elements for item in items
            ),
            dense_frontier_absorptions=sum(
                item.dense_frontier_absorptions for item in items
            ),
        )

    def left_boundary(self):
        return tuple(engine.left_boundary() for engine in self._engines)

    def right_boundary(self):
        return tuple(engine.right_boundary() for engine in self._engines)

    def advance_left(self, message, tensors, site):
        if len(message) != len(self._engines):
            raise ValueError("termwise TT message has the wrong component count.")
        return tuple(
            engine.advance_left(part, tensors, site)
            for engine, part in zip(self._engines, message)
        )

    def advance_right(self, message, tensors, site):
        if len(message) != len(self._engines):
            raise ValueError("termwise TT message has the wrong component count.")
        return tuple(
            engine.advance_right(part, tensors, site)
            for engine, part in zip(self._engines, message)
        )

    def build_left(self, tensors, *, reset_diagnostics=True):
        if reset_diagnostics:
            for engine in self._engines:
                engine.reset_diagnostics()
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        for site in range(self.nsites):
            messages[site + 1] = self.advance_left(messages[site], tensors, site)
        return messages

    def build_right(self, tensors, *, reset_diagnostics=True):
        if reset_diagnostics:
            for engine in self._engines:
                engine.reset_diagnostics()
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        for site in range(self.nsites - 1, -1, -1):
            messages[site] = self.advance_right(messages[site + 1], tensors, site)
        return messages

    def scalar(self, tensors):
        if self.channel_grouping == "component":
            return self._shared_component_scalar(tensors)
        values = [engine.scalar(tensors) for engine in self._engines]
        return self._grouped_scalar(values)

    def _shared_component_scalar(self, tensors):
        """Reuse identity prefixes/suffixes within each protected term."""
        for engine in self._engines:
            engine.reset_diagnostics()
        total = 0.0j
        for (indices, hermitian), sites in zip(
            self._groups, self._group_supports
        ):
            if len(indices) == 1:
                value = self._engines[indices[0]].scalar(tensors)
                total += np.real(value) if hermitian else value
                continue
            engines = tuple(self._engines[index] for index in indices)
            start = min(sites)
            stop = max(sites) + 1
            prefix = engines[0].left_boundary()
            for site in range(start):
                prefix = engines[0].advance_left(prefix, tensors, site)
            branches = []
            for engine in engines:
                message = prefix
                for site in range(start, stop):
                    message = engine.advance_left(message, tensors, site)
                branches.append(message)
            message = branches[0]
            for branch in branches[1:]:
                message = message.add(branch)
            message = message.round(
                max_rank=self.max_rank,
                rtol=self.rtol,
                atol=self.atol,
            )
            for site in range(stop, self.nsites):
                message = engines[0].advance_left(message, tensors, site)
            value = message.to_dense().reshape(()).item()
            total += np.real(value) if hermitian else value
        return total

    def _grouped_scalar(self, values):
        total = 0.0j
        for indices, hermitian in self._groups:
            value = sum(values[index] for index in indices)
            total += np.real(value) if hermitian else value
        return total

    def boundary_scalar(self, message, cut):
        cut = int(cut)
        if cut not in {0, self.nsites}:
            raise ValueError("a scalar can only be extracted at a boundary cut.")
        if len(message) != len(self._engines):
            raise ValueError("termwise TT message has the wrong component count.")
        values = [part.to_dense().reshape(()).item() for part in message]
        return self._grouped_scalar(values)

    def hole_action(self, site, left, right, vector, *, allow_dense=False):
        if len(left) != len(self._engines) or len(right) != len(self._engines):
            raise ValueError("termwise TT environment has the wrong component count.")
        return sum(
            engine.hole_action(
                site, left_part, right_part, vector, allow_dense=allow_dense
            )
            for engine, left_part, right_part in zip(
                self._engines, left, right
            )
        )

    def hole_adjoint_action(
        self, site, left, right, vector, *, allow_dense=False
    ):
        if len(left) != len(self._engines) or len(right) != len(self._engines):
            raise ValueError("termwise TT environment has the wrong component count.")
        return sum(
            engine.hole_adjoint_action(
                site, left_part, right_part, vector, allow_dense=allow_dense
            )
            for engine, left_part, right_part in zip(
                self._engines, left, right
            )
        )


class TermwiseBlockMPOFrontier:
    """Exact identity-block frontiers over bounded Hamiltonian chunks.

    Product strings are compiled into small shared-prefix MPO automata. Scalar
    contractions splice spatially grouped chunks between precomputed identity
    prefixes and suffixes, so only their coherence-safe active intervals are
    traversed. Independent chunks may run on a bounded thread pool.
    ``chunk_size=1`` is the original strictly termwise partition. Directional
    environments retain one block message per chunk; checkpoint/recompute
    therefore remains important during sweeps.
    """

    def __init__(
        self,
        hamiltonian,
        physical_sites,
        tensor_shapes,
        *,
        optimize="greedy",
        local_qns=None,
        bond_qns=None,
        chunk_size=8,
        chunk_memory=64,
        chunk_span=None,
        workers=1,
        compute_dtype=None,
        device="cpu",
    ):
        self.hamiltonian = hamiltonian
        self.dims = tuple(hamiltonian.dims)
        self.physical_groups = tuple(tuple(sites) for sites in physical_sites)
        self.tensor_shapes = tuple(tuple(shape) for shape in tensor_shapes)
        self.nsites = len(self.dims)
        self.optimize = optimize
        self.compute_dtype = compute_dtype
        self.device = str(device)
        self.local_qns = local_qns
        self.bond_qns = bond_qns
        if isinstance(chunk_size, (bool, np.bool_)):
            raise TypeError("chunk_size must be a positive integer.")
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_memory is None:
            self.chunk_memory = None
        else:
            self.chunk_memory = float(chunk_memory)
            if not np.isfinite(self.chunk_memory) or self.chunk_memory <= 0.0:
                raise ValueError("chunk_memory must be positive and finite or None.")
        self._chunk_limit_elements = (
            None
            if self.chunk_memory is None
            else max(1, int(self.chunk_memory * 2**20 // 16))
        )
        if chunk_span is None:
            self.chunk_span = None
        else:
            if isinstance(chunk_span, (bool, np.bool_)):
                raise TypeError("chunk_span must be a positive integer or None.")
            self.chunk_span = int(chunk_span)
            if self.chunk_span < 1:
                raise ValueError("chunk_span must be a positive integer or None.")
        if isinstance(workers, (bool, np.bool_)):
            raise TypeError("workers must be a positive integer.")
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be a positive integer.")
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="letta-frontier",
            )
            if self.workers > 1
            else None
        )
        self._engines = []
        self.chunk_sizes = []
        self.chunk_intervals = []
        first_reference = list(range(self.nsites))
        for owner, group in enumerate(self.physical_groups):
            for physical_site in group[1:]:
                first_reference[physical_site] = min(
                    first_reference[physical_site], owner
                )
        self._first_reference = tuple(first_reference)

        def make_engine(mpo):
            return BlockMPOFrontier(
                self.dims,
                self.physical_groups,
                self.tensor_shapes,
                mpo.tensors,
                optimize=self.optimize,
                local_qns=self.local_qns,
                bond_qns=self.bond_qns,
                compute_dtype=self.compute_dtype,
                device=self.device,
            )

        @dataclass(frozen=True)
        class ProductComponent:
            sites: tuple[int, ...]
            operators: tuple[np.ndarray, ...]
            coefficient: complex

        def mpo_product(mpo):
            sites = []
            operators = []
            coefficient = 1.0
            for site, tensor in enumerate(mpo.tensors):
                matrix = np.asarray(tensor)[0, 0]
                scalar = np.trace(matrix) / matrix.shape[0]
                residual = matrix - scalar * np.eye(
                    matrix.shape[0], dtype=matrix.dtype
                )
                if not np.any(residual):
                    coefficient *= scalar
                else:
                    sites.append(site)
                    operators.append(matrix)
            if not sites:
                sites = [0]
                operators = [np.eye(self.dims[0], dtype=self.hamiltonian.dtype)]
            return ProductComponent(
                tuple(sites),
                tuple(operators),
                np.asarray(coefficient).item(),
            )

        def component_interval(component):
            return int(component.sites[0]), int(component.sites[-1]) + 1

        def chunk_mpo(entries):
            return _operator_string_mpo(
                self.hamiltonian.sites,
                tuple(entries),
                self.hamiltonian.dtype,
            )

        def append_bounded_chunk(entries):
            engine = make_engine(chunk_mpo(entries))
            start = min(component.sites[0] for component in entries)
            stop = max(component.sites[-1] for component in entries) + 1
            while start:
                paired = engine.paired_sites(start, 0)
                if not paired:
                    break
                safe_start = min(
                    start,
                    *(self._first_reference[site] for site in paired),
                )
                if safe_start == start:
                    break
                start = safe_start
            if (
                self._chunk_limit_elements is not None
                and engine.peak_message_elements > self._chunk_limit_elements
                and len(entries) > 1
            ):
                middle = len(entries) // 2
                del engine
                append_bounded_chunk(entries[:middle])
                append_bounded_chunk(entries[middle:])
                return
            self._engines.append(engine)
            self.chunk_sizes.append(len(entries))
            self.chunk_intervals.append((int(start), int(stop)))

        components = []
        if hamiltonian.constant != 0.0:
            constant_mpo = _product_mpo(
                self.dims,
                {0: hamiltonian.constant * np.eye(self.dims[0])},
            )
            components.append(mpo_product(constant_mpo))
        for term in hamiltonian.local_terms:
            components.extend(
                mpo_product(mpo)
                for mpo in _term_product_mpos(
                    self.dims, term, local_qns=self.local_qns
                )
            )
        for product in hamiltonian.products:
            if product.coefficient != 0:
                components.append(product)
        if not components:
            raise ValueError("a termwise Hamiltonian requires at least one term.")

        components.sort(
            key=lambda component: (
                component.sites[0],
                component.sites[-1],
                len(component.sites),
            )
        )
        pending = []
        pending_start = pending_stop = None
        for component in components:
            start, stop = component_interval(component)
            combined_start = start if pending_start is None else min(pending_start, start)
            combined_stop = stop if pending_stop is None else max(pending_stop, stop)
            exceeds_span = (
                self.chunk_span is not None
                and pending
                and combined_stop - combined_start > self.chunk_span
            )
            if pending and (len(pending) >= self.chunk_size or exceeds_span):
                append_bounded_chunk(pending)
                pending = []
                pending_start = pending_stop = None
                combined_start, combined_stop = start, stop
            pending.append(component)
            pending_start, pending_stop = combined_start, combined_stop
        if pending:
            append_bounded_chunk(pending)
        self.chunk_sizes = tuple(self.chunk_sizes)
        self.chunk_intervals = tuple(self.chunk_intervals)

        reference = self._engines[0]
        self.frontier_sites = reference.frontier_sites
        self.virtual_bonds = reference.virtual_bonds
        self.mpo_bonds = tuple(1 for _ in range(self.nsites + 1))
        self.dtype = np.dtype(
            np.result_type(*(engine.dtype for engine in self._engines))
        )
        identity_mpo = _product_mpo(self.dims, {})
        self._identity_engine = make_engine(identity_mpo)

    def _map_chunks(self, function):
        if self._executor is None or len(self._engines) == 1:
            return [function(index, engine) for index, engine in enumerate(self._engines)]
        return list(
            self._executor.map(
                lambda item: function(*item),
                enumerate(self._engines),
            )
        )

    def close(self):
        """Release the persistent bounded worker pool."""

        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    def _identity_seed_block(self, engine, cut, block):
        identity = self._identity_engine
        source_shape = identity.block_shape(cut, 0)
        target_shape = engine.block_shape(cut, 0)
        if np.shape(block) != identity.storage_shape(cut, 0):
            raise ValueError("identity prefix has an invalid frontier shape.")
        if engine.charge_resolved:
            if engine._bond_pairs[cut][0] != identity._bond_pairs[cut][0]:
                raise ValueError("identity prefix has incompatible virtual-charge pairs.")
            source_bra_bond, source_ket_bond, source_bra, source_ket = (
                identity._expanded_storage_coordinates(cut, 0)
            )
            target_pairs = {
                pair: index
                for index, pair in enumerate(engine._bond_pairs[cut][0])
            }
            target_paired = set(engine.paired_sites(cut, 0))
            target_inverse = engine._storage_inverse(cut, 0)
            promoted = np.zeros(
                engine.storage_shape(cut, 0),
                dtype=np.asarray(block).dtype,
            )
            for source_index in range(len(source_bra_bond)):
                pair = (
                    int(source_bra_bond[source_index]),
                    int(source_ket_bond[source_index]),
                )
                coordinate = [target_pairs[pair]]
                for physical_site in engine.frontier_sites[cut]:
                    bra_value = int(source_bra[physical_site][source_index])
                    ket_value = int(source_ket[physical_site][source_index])
                    coordinate.append(bra_value)
                    if physical_site in target_paired:
                        coordinate.append(ket_value)
                    elif bra_value != ket_value:
                        raise ValueError(
                            "identity prefix is not diagonal on a shared frontier leg."
                        )
                logical = int(np.ravel_multi_index(tuple(coordinate), target_shape))
                packed = int(target_inverse[logical])
                if packed >= 0:
                    promoted.reshape(-1)[packed] += np.asarray(block).reshape(-1)[
                        source_index
                    ]
            return promoted
        block = identity._logical_block(cut, 0, block)
        leading = 2
        paired = set(engine.paired_sites(cut, 0))
        if not paired:
            if source_shape != target_shape:
                raise ValueError(
                    "identity prefix is incompatible with the chunk frontier."
                )
            return engine._store_block(cut, 0, block)

        source_labels = list(range(len(source_shape)))
        output_labels = source_labels[:leading]
        arguments = [block, tuple(source_labels)]
        next_label = len(source_labels)
        for offset, site in enumerate(identity.frontier_sites[cut]):
            shared = source_labels[leading + offset]
            if site not in paired:
                output_labels.append(shared)
                continue
            bra, ket = next_label, next_label + 1
            next_label += 2
            arguments.extend(
                (
                    identity._copies[self.dims[site]],
                    (shared, bra, ket),
                )
            )
            output_labels.extend((bra, ket))
        promoted = contract(*arguments, tuple(output_labels), optimize=self.optimize)
        if promoted.shape != target_shape:
            raise ValueError("promoted identity prefix has an invalid frontier shape.")
        return engine._store_block(cut, 0, promoted)

    def _seeded_message(self, engine, cut, block):
        blocks = [np.zeros(0, dtype=engine.dtype) for _ in range(engine.mpo_bonds[cut])]
        blocks[0] = self._identity_seed_block(engine, cut, block)
        return BlockFrontierMessage(int(cut), tuple(blocks))

    def _window_scalar(self, index, engine, tensors, identity_left, identity_right):
        start, stop = self.chunk_intervals[index]
        message = self._seeded_message(
            engine,
            start,
            identity_left[start].blocks[0],
        )
        for site in range(start, stop):
            message = engine.advance_left(message, tensors, site)
        suffix = identity_right[stop].blocks[0]
        value = message.blocks[0]
        if value.shape != suffix.shape:
            raise ValueError(
                "identity suffix is incompatible with the completed chunk frontier."
            )
        return np.sum(value * suffix)

    def _identity_window_boundaries(self, tensors):
        """Build only identity messages referenced by active chunk windows."""

        left_cuts = {start for start, _stop in self.chunk_intervals}
        right_cuts = {stop for _start, stop in self.chunk_intervals}
        left = {}
        message = self._identity_engine.left_boundary()
        if 0 in left_cuts:
            left[0] = message
        for site in range(self.nsites):
            message = self._identity_engine.advance_left(message, tensors, site)
            if site + 1 in left_cuts:
                left[site + 1] = message

        right = {}
        message = self._identity_engine.right_boundary()
        if self.nsites in right_cuts:
            right[self.nsites] = message
        for site in range(self.nsites - 1, -1, -1):
            message = self._identity_engine.advance_right(message, tensors, site)
            if site in right_cuts:
                right[site] = message
        return left, right

    @property
    def plan_count(self):
        return sum(engine.plan_count for engine in self._engines)

    @property
    def nchunks(self):
        return len(self._engines)

    def message_elements(self, cut):
        return sum(engine.message_elements(cut) for engine in self._engines)

    def dense_message_elements(self, cut):
        return sum(engine.dense_message_elements(cut) for engine in self._engines)

    @property
    def peak_message_elements(self):
        return max(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def stream_peak_message_elements(self):
        """Largest stored chunk message during a streamed scalar contraction."""

        chunk_peak = max(
            max(
                engine.message_elements(cut)
                for cut in range(start, stop + 1)
            )
            for engine, (start, stop) in zip(
                self._engines, self.chunk_intervals
            )
        )
        identity_peak = max(
            self._identity_engine.message_elements(cut)
            for cut in range(self.nsites + 1)
        )
        return max(chunk_peak, identity_peak)

    @property
    def dense_peak_message_elements(self):
        return max(
            self.dense_message_elements(cut) for cut in range(self.nsites + 1)
        )

    @property
    def total_message_elements(self):
        return sum(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def dense_total_message_elements(self):
        return sum(
            self.dense_message_elements(cut) for cut in range(self.nsites + 1)
        )

    def _validated_message(self, message, cut):
        if not isinstance(message, tuple) or len(message) != len(self._engines):
            raise ValueError("termwise block message has the wrong component count.")
        for engine, part in zip(self._engines, message):
            engine._validated_message(part, int(cut))
        return message

    def left_boundary(self):
        return tuple(engine.left_boundary() for engine in self._engines)

    def right_boundary(self):
        return tuple(engine.right_boundary() for engine in self._engines)

    def advance_left(self, message, tensors, site):
        self._validated_message(message, int(site))

        def advance(index, engine):
            return engine.advance_left(message[index], tensors, site)

        return tuple(self._map_chunks(advance))

    def advance_right(self, message, tensors, site):
        self._validated_message(message, int(site) + 1)

        def advance(index, engine):
            return engine.advance_right(message[index], tensors, site)

        return tuple(self._map_chunks(advance))

    def build_left(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        for site in range(self.nsites):
            messages[site + 1] = self.advance_left(messages[site], tensors, site)
        return messages

    def build_right(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        for site in range(self.nsites - 1, -1, -1):
            messages[site] = self.advance_right(messages[site + 1], tensors, site)
        return messages

    def scalar(self, tensors):
        identity_left, identity_right = self._identity_window_boundaries(tensors)

        def contract_window(index, engine):
            return self._window_scalar(
                index,
                engine,
                tensors,
                identity_left,
                identity_right,
            )

        return sum(self._map_chunks(contract_window), 0.0j)

    def boundary_scalar(self, message, cut):
        message = self._validated_message(message, int(cut))
        return sum(
            engine.boundary_scalar(part, cut)
            for engine, part in zip(self._engines, message)
        )

    def hole_matrix(self, site, left, right):
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        def action(index, engine):
            return engine.hole_matrix(
                site,
                left[index],
                right[index],
            )

        return sum(self._map_chunks(action), 0)

    def hole_block(self, site, left, right, bra_configuration, ket_configuration):
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        def action(index, engine):
            return engine.hole_block(
                site,
                left[index],
                right[index],
                bra_configuration,
                ket_configuration,
            )


        return sum(self._map_chunks(action), 0)

    def hole_blocks(
        self,
        site,
        left,
        right,
        bra_configurations,
        ket_configurations,
    ):
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        def action(index, engine):
            return engine.hole_blocks(
                site,
                left[index],
                right[index],
                bra_configurations,
                ket_configurations,
            )


        return sum(self._map_chunks(action), 0)

    def hole_action(self, site, left, right, vector):
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        def action(index, engine):
            return engine.hole_action(
                site,
                left[index],
                right[index],
                vector,
            )


        return sum(self._map_chunks(action), 0)

    def prepare_hole_action(self, site, left, right):
        """Bind all chunk messages and retain their compiled grouped plans."""

        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        actions = tuple(
            engine.prepare_hole_action(site, left_part, right_part)
            for engine, left_part, right_part in zip(self._engines, left, right)
        )

        def action(vector):
            def apply(index, _engine):
                return actions[index](vector)

            return sum(self._map_chunks(apply), 0)

        def actions_many(vectors):
            def apply(index, _engine):
                return actions[index].many(vectors)

            return sum(self._map_chunks(apply), 0)

        action.many = actions_many
        if any(hasattr(item, "verify") for item in actions):
            def verify(vector):
                def apply(index, _engine):
                    verifier = getattr(actions[index], "verify", actions[index])
                    return verifier(vector)

                return sum(self._map_chunks(apply), 0)

            action.verify = verify
        return action

    def hole_action_components(self, site, left, right, vector):
        """Yield one exact local-action contribution per Hamiltonian chunk."""
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        for engine, left_part, right_part in zip(self._engines, left, right):
            yield engine.hole_action(
                site,
                left_part,
                right_part,
                vector,
            )

    def hole_action_component_count(self, site):
        return len(self._engines)

    def left_enrichment_components(self, site, left, vector):
        """Stream exact open ``L W A`` ranges one Hamiltonian chunk at a time."""
        self._validated_message(left, int(site))
        for engine, left_part in zip(self._engines, left):
            yield from engine.left_enrichment_components(
                site,
                left_part,
                vector,
            )

    def right_enrichment_components(self, site, right, vector):
        """Stream exact open ``A W R`` ranges one Hamiltonian chunk at a time."""
        self._validated_message(right, int(site) + 1)
        for engine, right_part in zip(self._engines, right):
            yield from engine.right_enrichment_components(
                site,
                right_part,
                vector,
            )

    def enrichment_component_count(self, site):
        return sum(engine.enrichment_component_count(site) for engine in self._engines)

    def hole_actions(self, site, left, right, vectors):
        self._validated_message(left, int(site))
        self._validated_message(right, int(site) + 1)
        def action(index, engine):
            return engine.hole_actions(
                site,
                left[index],
                right[index],
                vectors,
            )


        return sum(self._map_chunks(action), 0)


__all__ = [
    "TTAdvanceDiagnostics",
    "TTContractionDiagnostics",
    "TTFrontier",
    "TTHoleDiagnostics",
    "TTMPOFrontier",
    "TermwiseBlockMPOFrontier",
    "TermwiseTTMPOFrontier",
    "TTRoundDiagnostics",
]
