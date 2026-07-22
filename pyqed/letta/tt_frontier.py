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

from dataclasses import dataclass
from math import sqrt
from typing import Hashable, Iterable

import numpy as np
from opt_einsum import contract

from .local_terms import LocalMPO
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
        self.physical_sites = self._metadata.physical_sites
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
        value = self.build_left(tensors)[-1].to_dense()
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
                for index in self.physical_sites[site]
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
                for index in self.physical_sites[site]
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
        for physical_site in self.physical_sites[site][1:]:
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
    return LocalMPO(dims, tensors)


def _term_product_mpos(dims, term):
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
    rounding cannot mix or remove Hamiltonian automaton sectors.
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
    ):
        self.hamiltonian = hamiltonian
        self.dims = tuple(hamiltonian.dims)
        self.physical_sites = tuple(tuple(sites) for sites in physical_sites)
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
        self._groups = []
        self._engines = []
        if hamiltonian.constant != 0.0:
            constant_mpo = _product_mpo(
                self.dims,
                {0: hamiltonian.constant * np.eye(self.dims[0])},
            )
            self._engines.append(
                TTMPOFrontier(
                    self.dims,
                    self.physical_sites,
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
        for term in hamiltonian.terms:
            group = []
            for mpo in _term_product_mpos(self.dims, term):
                engine = TTMPOFrontier(
                    self.dims,
                    self.physical_sites,
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
        values = [engine.scalar(tensors) for engine in self._engines]
        return self._grouped_scalar(values)

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


__all__ = [
    "TTAdvanceDiagnostics",
    "TTContractionDiagnostics",
    "TTFrontier",
    "TTHoleDiagnostics",
    "TTMPOFrontier",
    "TermwiseTTMPOFrontier",
    "TTRoundDiagnostics",
]
