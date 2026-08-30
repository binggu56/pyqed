"""Physical-slice block operators for one-site graph-LETTA updates.

For a local tensor with layout ``(D_left, D_right, *physical_dims)``, fixing
all local physical indices leaves a vector of ``D_left * D_right`` virtual
entries.  The exact norm operator is diagonal between these physical slices,
while a local Hamiltonian connects only the slices allowed by its finite-
support terms.  This module represents that structure without changing the
native C-order flattening used by :class:`FrontierTiedLETTA`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from pyqed.tn import Hamiltonian
from pyqed.tn.effective_operator import PackedBlockEffectiveOperator
from .matrix_free import DavidsonDiagnostics, lowest_generalized_davidson

try:  # Optional native kernel; the Python path below is the reference path.
    from . import _physical_blocks_cpp
except Exception:  # pragma: no cover - depends on optional build artifacts.
    _physical_blocks_cpp = None


@dataclass(frozen=True)
class PhysicalBlockSolveDiagnostics:
    """Convergence and sparsity information for a componentwise solve."""

    converged: bool
    message: str
    iterations: int
    hamiltonian_matvecs: int
    metric_matvecs: int
    restarts: int
    residual_norm: float
    metric_norm: float
    projected_rank: int
    subspace_dimension: int
    energy_history: tuple[float, ...]
    residual_history: tuple[float, ...]
    component_sizes: tuple[int, ...]
    positive_metric_components: int
    selected_component: int
    metric_blocks: int
    hamiltonian_blocks: int
    stored_elements: int
    component_dimensions: tuple[int, ...] = ()
    dense_components: tuple[bool, ...] = ()


@dataclass(frozen=True)
class PhysicalBlockLayout:
    """Map a flattened LETTA tensor to fixed-physical-index blocks."""

    tensor_shape: tuple[int, ...]
    virtual_shape: tuple[int, int]
    physical_shape: tuple[int, ...]
    block_indices: tuple[np.ndarray, ...]
    configurations: tuple[tuple[int, ...], ...]

    def __init__(self, tensor_shape):
        tensor_shape = tuple(int(dim) for dim in tensor_shape)
        if len(tensor_shape) < 3:
            raise ValueError(
                "a graph-LETTA tensor needs two virtual axes and at least "
                "one physical axis."
            )
        if any(dim < 1 for dim in tensor_shape):
            raise ValueError("tensor_shape must contain only positive dimensions.")
        virtual_shape = tensor_shape[:2]
        physical_shape = tensor_shape[2:]
        virtual_size = int(np.prod(virtual_shape))
        nblocks = int(np.prod(physical_shape))
        native_indices = np.arange(virtual_size * nblocks).reshape(
            virtual_size, nblocks
        )
        block_indices = tuple(
            np.array(native_indices[:, block], copy=True) for block in range(nblocks)
        )
        for indices in block_indices:
            indices.setflags(write=False)
        configurations = tuple(
            tuple(int(value) for value in np.unravel_index(block, physical_shape))
            for block in range(nblocks)
        )
        object.__setattr__(self, "tensor_shape", tensor_shape)
        object.__setattr__(self, "virtual_shape", virtual_shape)
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "block_indices", block_indices)
        object.__setattr__(self, "configurations", configurations)

    @property
    def size(self) -> int:
        return int(np.prod(self.tensor_shape))

    @property
    def virtual_size(self) -> int:
        return int(np.prod(self.virtual_shape))

    @property
    def nblocks(self) -> int:
        return int(np.prod(self.physical_shape))

    def as_blocks(self, vector) -> np.ndarray:
        """Return ``vector`` as ``(physical_block, virtual_entry)``."""
        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError(f"vector must contain {self.size} entries.")
        flat = vector.reshape(-1)
        return np.stack([flat[indices] for indices in self.block_indices])

    def from_blocks(self, blocks) -> np.ndarray:
        """Restore native C-order flattening from physical-block order."""
        blocks = np.asarray(blocks)
        expected = (self.nblocks, self.virtual_size)
        if blocks.shape != expected:
            raise ValueError(f"blocks must have shape {expected}.")
        vector = np.empty(self.size, dtype=blocks.dtype)
        for block, indices in enumerate(self.block_indices):
            vector[indices] = blocks[block]
        return vector


def hamiltonian_physical_connectivity(
    hamiltonian: Hamiltonian,
    physical_sites,
    *,
    operator_atol: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    r"""Return the structurally allowed local blocks ``(p_bra, p_ket)``.

    A term can change only physical variables in its support.  Variables tied
    into the optimized tensor but absent from that support must therefore have
    equal bra and ket values.  Variables in the support but outside the local
    tensor are left free when testing whether an operator entry exists.  The
    result is an exact structural over-approximation: environment cancellations
    may make an allowed block zero, but a forbidden block cannot contribute.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian.")
    physical_sites = tuple(int(site) for site in physical_sites)
    if not physical_sites:
        raise ValueError("physical_sites must contain the optimized site.")
    if len(physical_sites) != len(set(physical_sites)):
        raise ValueError("physical_sites must be unique.")
    if any(site < 0 or site >= len(hamiltonian.dims) for site in physical_sites):
        raise ValueError("physical_sites contains an invalid site.")
    operator_atol = float(operator_atol)
    if not np.isfinite(operator_atol) or operator_atol < 0.0:
        raise ValueError("operator_atol must be finite and nonnegative.")

    physical_shape = tuple(hamiltonian.dims[site] for site in physical_sites)
    nblocks = int(np.prod(physical_shape, dtype=np.int64))
    connected: set[tuple[int, int]] = set()
    if hamiltonian.constant != 0.0:
        connected.update((block, block) for block in range(nblocks))

    def add_connections(changed_axes, projected_transitions):
        changed_axes = tuple(changed_axes)
        changed_axes = tuple(
            int(axis) for axis in changed_axes
        )
        preserved_axes = tuple(
            axis for axis in range(len(physical_sites)) if axis not in changed_axes
        )
        preserved_shape = tuple(physical_shape[axis] for axis in preserved_axes)
        for preserved_values in np.ndindex(preserved_shape):
            for bra_changed, ket_changed in projected_transitions:
                bra = [0] * len(physical_sites)
                ket = [0] * len(physical_sites)
                for axis, value in zip(preserved_axes, preserved_values):
                    bra[axis] = ket[axis] = value
                for position, axis in enumerate(changed_axes):
                    bra[axis] = bra_changed[position]
                    ket[axis] = ket_changed[position]
                connected.add(
                    (
                        int(np.ravel_multi_index(tuple(bra), physical_shape)),
                        int(np.ravel_multi_index(tuple(ket), physical_shape)),
                    )
                )

    for term in hamiltonian.local_terms:
        term_axis = {site: axis for axis, site in enumerate(term.sites)}
        changed_axes = tuple(
            axis for axis, site in enumerate(physical_sites) if site in term_axis
        )
        term_dims = tuple(hamiltonian.dims[site] for site in term.sites)
        rows, columns = np.nonzero(np.abs(term.operator) > operator_atol)
        if rows.size == 0:
            continue
        bra_support = np.stack(np.unravel_index(rows, term_dims), axis=1)
        ket_support = np.stack(np.unravel_index(columns, term_dims), axis=1)
        projected_transitions = {
            (
                tuple(
                    int(bra_values[term_axis[physical_sites[axis]]])
                    for axis in changed_axes
                ),
                tuple(
                    int(ket_values[term_axis[physical_sites[axis]]])
                    for axis in changed_axes
                ),
            )
            for bra_values, ket_values in zip(bra_support, ket_support)
        }
        add_connections(changed_axes, projected_transitions)

    for product in hamiltonian.products:
        if abs(product.coefficient) <= operator_atol or any(
            not np.any(np.abs(operator) > operator_atol)
            for operator in product.operators
        ):
            continue
        operator_by_site = dict(zip(product.sites, product.operators))
        changed_axes = tuple(
            axis
            for axis, site in enumerate(physical_sites)
            if site in operator_by_site
        )
        local_transitions = []
        for axis in changed_axes:
            operator = operator_by_site[physical_sites[axis]]
            rows, columns = np.nonzero(np.abs(operator) > operator_atol)
            local_transitions.append(tuple(zip(rows.tolist(), columns.tolist())))
        projected_transitions = {((), ())}
        for transitions in local_transitions:
            projected_transitions = {
                (bra + (int(row),), ket + (int(column),))
                for bra, ket in projected_transitions
                for row, column in transitions
            }
        add_connections(changed_axes, projected_transitions)
    return tuple(sorted(connected))


class PhysicalBlockLinearOperator:
    """A square operator stored as selected physical-to-physical blocks."""

    def __init__(
        self,
        layout: PhysicalBlockLayout,
        blocks: Mapping[tuple[int, int], np.ndarray],
        *,
        dtype=None,
    ):
        if not isinstance(layout, PhysicalBlockLayout):
            raise TypeError("layout must be a PhysicalBlockLayout.")
        copied = {}
        inferred_dtypes = []
        expected_shape = (layout.virtual_size, layout.virtual_size)
        for pair, block in blocks.items():
            if len(pair) != 2:
                raise ValueError("each block key must be a (row, column) pair.")
            row, column = (int(pair[0]), int(pair[1]))
            if not (0 <= row < layout.nblocks and 0 <= column < layout.nblocks):
                raise ValueError("physical block index is out of range.")
            value = np.asarray(block)
            if value.shape != expected_shape:
                raise ValueError(f"operator blocks must have shape {expected_shape}.")
            if np.any(~np.isfinite(value)):
                raise ValueError("operator blocks must contain only finite values.")
            copied[(row, column)] = np.array(value, copy=True)
            inferred_dtypes.append(value.dtype)
        if dtype is None:
            dtype = np.result_type(*inferred_dtypes, np.float64)
        self.layout = layout
        self.dtype = np.dtype(dtype)
        self.blocks = {
            pair: np.asarray(block, dtype=self.dtype) for pair, block in copied.items()
        }
        self._native_rows = np.asarray(
            [pair[0] for pair in self.blocks],
            dtype=np.intp,
        )
        self._native_columns = np.asarray(
            [pair[1] for pair in self.blocks],
            dtype=np.intp,
        )
        self._native_blocks = (
            np.ascontiguousarray(np.stack(tuple(self.blocks.values())))
            if self.blocks
            else np.empty(
                (0, layout.virtual_size, layout.virtual_size),
                dtype=self.dtype,
            )
        )
        self.shape = (layout.size, layout.size)
        self._packed = PackedBlockEffectiveOperator(
            layout.block_indices,
            self.blocks,
            dtype=self.dtype,
        )

    @property
    def _use_native_matvec(self) -> bool:
        return bool(
            _physical_blocks_cpp is not None
            and self.layout.virtual_size <= 64
            and self._native_blocks.dtype
            in {np.dtype(np.float64), np.dtype(np.complex128)}
        )

    @property
    def connected_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self.blocks))

    @property
    def stored_elements(self) -> int:
        return sum(block.size for block in self.blocks.values())

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def backend(self) -> str:
        """Name of the grouped contraction backend used by this operator."""

        return self._packed.backend

    def matvec(self, vector) -> np.ndarray:
        inputs = self.layout.as_blocks(vector)
        if self._use_native_matvec:
            return self.layout.from_blocks(
                _physical_blocks_cpp.block_matvec(
                    self._native_blocks,
                    self._native_rows,
                    self._native_columns,
                    np.ascontiguousarray(inputs),
                    False,
                )
            )
        dtype = np.result_type(inputs.dtype, self.dtype)
        outputs = np.zeros(
            (self.layout.nblocks, self.layout.virtual_size), dtype=dtype
        )
        for (row, column), block in self.blocks.items():
            outputs[row] += block @ inputs[column]
        return self.layout.from_blocks(outputs)

    def rmatvec(self, vector) -> np.ndarray:
        inputs = self.layout.as_blocks(vector)
        if self._use_native_matvec:
            return self.layout.from_blocks(
                _physical_blocks_cpp.block_matvec(
                    self._native_blocks,
                    self._native_rows,
                    self._native_columns,
                    np.ascontiguousarray(inputs),
                    True,
                )
            )
        dtype = np.result_type(inputs.dtype, self.dtype)
        outputs = np.zeros(
            (self.layout.nblocks, self.layout.virtual_size), dtype=dtype
        )
        for (row, column), block in self.blocks.items():
            outputs[column] += block.T.conj() @ inputs[row]
        return self.layout.from_blocks(outputs)

    def __matmul__(self, vector):
        return self.matvec(vector)

    def aslinearoperator(self):
        """Return a SciPy ``LinearOperator`` sharing these block actions."""
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=self.dtype,
        )

    def to_dense(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=self.dtype)
        for (row, column), block in self.blocks.items():
            result[
                np.ix_(
                    self.layout.block_indices[row],
                    self.layout.block_indices[column],
                )
            ] = block
        return result

    @classmethod
    def from_dense(
        cls,
        matrix,
        layout: PhysicalBlockLayout,
        *,
        connected_pairs: Iterable[tuple[int, int]] | None = None,
        zero_atol: float = 0.0,
        omitted_atol: float | None = None,
    ):
        """Extract selected blocks from a dense reference operator.

        If ``connected_pairs`` is omitted, blocks whose maximum absolute entry
        exceeds ``zero_atol`` are retained.  If it is supplied and
        ``omitted_atol`` is not ``None``, all structurally forbidden blocks are
        checked against that floating-point tolerance.
        """
        matrix = np.asarray(matrix)
        if matrix.shape != (layout.size, layout.size):
            raise ValueError(f"matrix must have shape {(layout.size, layout.size)}.")
        zero_atol = float(zero_atol)
        if not np.isfinite(zero_atol) or zero_atol < 0.0:
            raise ValueError("zero_atol must be finite and nonnegative.")
        requested = None
        if connected_pairs is not None:
            requested = {(int(row), int(column)) for row, column in connected_pairs}
        blocks = {}
        largest_omitted = 0.0
        for row in range(layout.nblocks):
            for column in range(layout.nblocks):
                block = matrix[
                    np.ix_(layout.block_indices[row], layout.block_indices[column])
                ]
                include = (
                    (row, column) in requested
                    if requested is not None
                    else bool(np.any(np.abs(block) > zero_atol))
                )
                if include:
                    blocks[(row, column)] = block
                elif block.size:
                    largest_omitted = max(
                        largest_omitted, float(np.max(np.abs(block)))
                    )
        if omitted_atol is not None:
            omitted_atol = float(omitted_atol)
            if not np.isfinite(omitted_atol) or omitted_atol < 0.0:
                raise ValueError("omitted_atol must be finite and nonnegative.")
            if largest_omitted > omitted_atol:
                raise ValueError(
                    "a structurally omitted block is numerically nonzero: "
                    f"maximum magnitude {largest_omitted:.3e} exceeds "
                    f"omitted_atol={omitted_atol:.3e}."
                )
        return cls(layout, blocks, dtype=matrix.dtype)

    @classmethod
    def from_block_factory(
        cls,
        layout: PhysicalBlockLayout,
        connected_pairs,
        factory: Callable[[int, int], np.ndarray],
        *,
        dtype=None,
    ):
        """Build only requested blocks using ``factory(row, column)``."""
        blocks = {
            (int(row), int(column)): factory(int(row), int(column))
            for row, column in connected_pairs
        }
        return cls(layout, blocks, dtype=dtype)


class MatrixFreePhysicalBlockOperator:
    """A physical-block operator evaluated by a full vector action.

    ``connected_pairs`` retains the exact structural physical-block graph used
    to split symmetry components.  Numerical block matrices are never formed;
    ``matvec`` delegates directly to the supplied frontier action.
    """

    def __init__(
        self,
        layout: PhysicalBlockLayout,
        connected_pairs,
        action: Callable[[np.ndarray], np.ndarray],
        *,
        raction: Callable[[np.ndarray], np.ndarray] | None = None,
        dtype=None,
    ):
        if not isinstance(layout, PhysicalBlockLayout):
            raise TypeError("layout must be a PhysicalBlockLayout.")
        if not callable(action):
            raise TypeError("action must be callable.")
        pairs = {(int(row), int(column)) for row, column in connected_pairs}
        if any(
            row < 0
            or row >= layout.nblocks
            or column < 0
            or column >= layout.nblocks
            for row, column in pairs
        ):
            raise ValueError("connected_pairs contains an invalid block index.")
        pairs |= {(column, row) for row, column in pairs}
        self.layout = layout
        self._connected_pairs = tuple(sorted(pairs))
        self._action = action
        self._actions = getattr(action, "many", None)
        self._verification_action = getattr(action, "verify", None)
        self._raction = action if raction is None else raction
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        self.shape = (layout.size, layout.size)

    @property
    def connected_pairs(self) -> tuple[tuple[int, int], ...]:
        return self._connected_pairs

    @property
    def stored_elements(self) -> int:
        return int(getattr(self, "_stored_elements", 0))

    @property
    def block_count(self) -> int:
        return len(self._connected_pairs)

    def _validated_action(self, vector, action, *, name):
        vector = np.asarray(vector)
        if vector.size != self.layout.size:
            raise ValueError(f"vector must contain {self.layout.size} entries.")
        result = np.asarray(action(vector.reshape(-1))).reshape(-1)
        if result.size != self.layout.size:
            raise ValueError(
                f"{name} must return {self.layout.size} entries."
            )
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{name} returned non-finite entries.")
        return result

    def matvec(self, vector) -> np.ndarray:
        return self._validated_action(vector, self._action, name="action")

    def matvecs(self, vectors) -> np.ndarray:
        """Apply a native batched action when the frontier provides one."""

        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.layout.size:
            raise ValueError(
                f"vectors must have shape (batch, {self.layout.size})."
            )
        if self._actions is None:
            return np.stack([self.matvec(vector) for vector in vectors])
        result = np.asarray(self._actions(vectors))
        if result.shape != vectors.shape:
            raise ValueError(
                f"batched action must return shape {vectors.shape}."
            )
        if np.any(~np.isfinite(result)):
            raise ValueError("batched action returned non-finite entries.")
        return result

    def rmatvec(self, vector) -> np.ndarray:
        return self._validated_action(vector, self._raction, name="raction")

    @property
    def has_verification_action(self) -> bool:
        return self._verification_action is not None

    def verification_matvec(self, vector) -> np.ndarray:
        action = self._action if self._verification_action is None else self._verification_action
        return self._validated_action(vector, action, name="verification_action")

    def __matmul__(self, vector):
        return self.matvec(vector)

    def aslinearoperator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=self.dtype,
        )

    def to_dense(self) -> np.ndarray:
        """Materialize by actions; intended only for small validation/fallback."""
        result = np.empty(self.shape, dtype=self.dtype)
        basis = np.zeros(self.layout.size, dtype=self.dtype)
        for column in range(self.layout.size):
            basis[column] = 1
            result[:, column] = self.matvec(basis)
            basis[column] = 0
        return result


@dataclass(frozen=True)
class PhysicalBlockGeneralizedProblem:
    """Block-sparse representation of ``H x = E N x`` for one LETTA site."""

    layout: PhysicalBlockLayout
    metric: PhysicalBlockLinearOperator
    hamiltonian: PhysicalBlockLinearOperator | MatrixFreePhysicalBlockOperator
    _metric_eigensystems_cache: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def metric_eigensystems(self):
        """Return and retain the Hermitian eigensystem of every metric block."""
        cached = self._metric_eigensystems_cache
        if cached is None:
            cached = tuple(
                linalg.eigh(
                    self.metric.blocks[(block, block)],
                    check_finite=False,
                )
                for block in range(self.layout.nblocks)
            )
            object.__setattr__(self, "_metric_eigensystems_cache", cached)
        return cached

    @classmethod
    def from_block_factories(
        cls,
        tensor_shape,
        hamiltonian_pairs,
        metric_factory: Callable[[int, int], np.ndarray],
        hamiltonian_factory: Callable[[int, int], np.ndarray],
        *,
        dtype=None,
    ):
        r"""Construct the pencil directly from fixed-physical contractions.

        Only diagonal metric blocks and structurally allowed Hamiltonian
        blocks are requested.  Opposite Hamiltonian blocks are contracted and
        averaged as

        ``(H[p,q] + H[q,p]^dagger) / 2``,

        reproducing the Hermitization used by the dense local solver without
        ever constructing the full local matrix.
        """
        layout = PhysicalBlockLayout(tensor_shape)
        pairs = {(int(row), int(column)) for row, column in hamiltonian_pairs}
        if any(
            row < 0
            or row >= layout.nblocks
            or column < 0
            or column >= layout.nblocks
            for row, column in pairs
        ):
            raise ValueError("hamiltonian_pairs contains an invalid block index.")
        pairs |= {(column, row) for row, column in pairs}

        metric_blocks = {}
        for block in range(layout.nblocks):
            value = np.asarray(metric_factory(block, block))
            metric_blocks[(block, block)] = 0.5 * (value + value.T.conj())

        hamiltonian_blocks = {}
        for row, column in sorted(pairs):
            if row > column:
                continue
            forward = np.asarray(hamiltonian_factory(row, column))
            if row == column:
                value = 0.5 * (forward + forward.T.conj())
                hamiltonian_blocks[(row, row)] = value
                continue
            reverse = np.asarray(hamiltonian_factory(column, row))
            value = 0.5 * (forward + reverse.T.conj())
            hamiltonian_blocks[(row, column)] = value
            hamiltonian_blocks[(column, row)] = value.T.conj()

        return cls(
            layout,
            PhysicalBlockLinearOperator(layout, metric_blocks, dtype=dtype),
            PhysicalBlockLinearOperator(layout, hamiltonian_blocks, dtype=dtype),
        )

    @classmethod
    def from_batched_block_factories(
        cls,
        tensor_shape,
        hamiltonian_pairs,
        metric_factory,
        hamiltonian_factory,
        *,
        dtype=None,
        batch_size=256,
    ):
        """Construct a block pencil with vectorized fixed-physical factories."""
        layout = PhysicalBlockLayout(tensor_shape)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        pairs = {(int(row), int(column)) for row, column in hamiltonian_pairs}
        if any(
            row < 0
            or row >= layout.nblocks
            or column < 0
            or column >= layout.nblocks
            for row, column in pairs
        ):
            raise ValueError("hamiltonian_pairs contains an invalid block index.")
        pairs |= {(column, row) for row, column in pairs}

        def evaluate(factory, requested):
            requested = tuple(requested)
            values = {}
            for start in range(0, len(requested), batch_size):
                chunk = requested[start : start + batch_size]
                blocks = np.asarray(
                    factory(
                        tuple(row for row, _column in chunk),
                        tuple(column for _row, column in chunk),
                    )
                )
                expected = (
                    len(chunk),
                    layout.virtual_size,
                    layout.virtual_size,
                )
                if blocks.shape != expected:
                    raise ValueError(
                        f"batched block factory must return shape {expected}."
                    )
                values.update(zip(chunk, blocks))
            return values

        diagonal = tuple((block, block) for block in range(layout.nblocks))
        raw_metric = evaluate(metric_factory, diagonal)
        metric_blocks = {
            pair: 0.5 * (block + block.T.conj())
            for pair, block in raw_metric.items()
        }
        raw_hamiltonian = evaluate(hamiltonian_factory, sorted(pairs))
        hamiltonian_blocks = {}
        for row, column in sorted(pairs):
            if row > column:
                continue
            forward = raw_hamiltonian[(row, column)]
            if row == column:
                hamiltonian_blocks[(row, row)] = 0.5 * (
                    forward + forward.T.conj()
                )
            else:
                reverse = raw_hamiltonian[(column, row)]
                value = 0.5 * (forward + reverse.T.conj())
                hamiltonian_blocks[(row, column)] = value
                hamiltonian_blocks[(column, row)] = value.T.conj()

        return cls(
            layout,
            PhysicalBlockLinearOperator(layout, metric_blocks, dtype=dtype),
            PhysicalBlockLinearOperator(layout, hamiltonian_blocks, dtype=dtype),
        )

    @classmethod
    def from_batched_metric_factory_and_hamiltonian_action(
        cls,
        tensor_shape,
        hamiltonian_pairs,
        metric_factory,
        hamiltonian_action,
        *,
        dtype=None,
        batch_size=256,
    ):
        """Build the metric blocks while retaining a lazy Hamiltonian action."""
        layout = PhysicalBlockLayout(tensor_shape)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        pairs = {(int(row), int(column)) for row, column in hamiltonian_pairs}
        if any(
            row < 0
            or row >= layout.nblocks
            or column < 0
            or column >= layout.nblocks
            for row, column in pairs
        ):
            raise ValueError("hamiltonian_pairs contains an invalid block index.")
        pairs |= {(column, row) for row, column in pairs}

        metric_blocks = {}
        diagonal = tuple((block, block) for block in range(layout.nblocks))
        for start in range(0, len(diagonal), batch_size):
            chunk = diagonal[start : start + batch_size]
            values = np.asarray(
                metric_factory(
                    tuple(row for row, _column in chunk),
                    tuple(column for _row, column in chunk),
                )
            )
            expected = (
                len(chunk),
                layout.virtual_size,
                layout.virtual_size,
            )
            if values.shape != expected:
                raise ValueError(
                    f"batched metric factory must return shape {expected}."
                )
            for pair, block in zip(chunk, values):
                metric_blocks[pair] = 0.5 * (block + block.T.conj())

        return cls(
            layout,
            PhysicalBlockLinearOperator(layout, metric_blocks, dtype=dtype),
            MatrixFreePhysicalBlockOperator(
                layout,
                pairs,
                hamiltonian_action,
                dtype=dtype,
            ),
        )

    @classmethod
    def from_dense(
        cls,
        metric,
        hamiltonian,
        tensor_shape,
        *,
        hamiltonian_pairs=None,
        omitted_atol: float | None = 1.0e-12,
    ):
        layout = PhysicalBlockLayout(tensor_shape)
        diagonal_pairs = tuple((block, block) for block in range(layout.nblocks))
        metric_blocks = PhysicalBlockLinearOperator.from_dense(
            metric,
            layout,
            connected_pairs=diagonal_pairs,
            omitted_atol=omitted_atol,
        )
        hamiltonian_blocks = PhysicalBlockLinearOperator.from_dense(
            hamiltonian,
            layout,
            connected_pairs=hamiltonian_pairs,
            omitted_atol=omitted_atol if hamiltonian_pairs is not None else None,
        )
        return cls(layout, metric_blocks, hamiltonian_blocks)

    @classmethod
    def from_frontier_state(
        cls,
        state,
        site: int,
        *,
        environment=None,
        omitted_atol: float | None = 1.0e-12,
    ):
        """Reference adapter using a state's current dense local operators.

        This helper is intended for validation.  Production frontier updates
        use fixed-physical block factories and do not form dense local arrays.
        """
        site = int(site)
        metric, hamiltonian = state.local_operators(site, environment=environment)
        pairs = hamiltonian_physical_connectivity(
            state.hamiltonian, state.physical_groups[site]
        )
        return cls.from_dense(
            metric,
            hamiltonian,
            state.tensors[site].shape,
            hamiltonian_pairs=pairs,
            omitted_atol=omitted_atol,
        )

    @property
    def hamiltonian_components(self) -> tuple[tuple[int, ...], ...]:
        """Connected components of the undirected physical-block graph."""
        neighbors = [set() for _ in range(self.layout.nblocks)]
        for row, column in self.hamiltonian.connected_pairs:
            neighbors[row].add(column)
            neighbors[column].add(row)
        components = []
        unseen = set(range(self.layout.nblocks))
        while unseen:
            root = min(unseen)
            stack = [root]
            unseen.remove(root)
            component = []
            while stack:
                block = stack.pop()
                component.append(block)
                new = neighbors[block] & unseen
                unseen.difference_update(new)
                stack.extend(sorted(new, reverse=True))
            components.append(tuple(sorted(component)))
        return tuple(components)

    @property
    def stored_elements(self) -> int:
        return self.metric.stored_elements + self.hamiltonian.stored_elements

    @property
    def dense_elements(self) -> int:
        return 2 * self.layout.size**2

    @property
    def storage_fraction(self) -> float:
        return self.stored_elements / self.dense_elements

    def metric_rank(self, metric_tol: float = 1.0e-12) -> int:
        """Return the positive rank of the block-diagonal metric."""
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        scale = max(
            (
                float(np.linalg.norm(block, ord=np.inf))
                for block in self.metric.blocks.values()
            ),
            default=0.0,
        )
        scale = max(scale, np.finfo(float).tiny)
        return int(sum(
            np.count_nonzero(values > metric_tol * scale)
            for values, _vectors in self.metric_eigensystems()
        ))

    def solve(self, initial_vector, **davidson_options):
        """Solve every positive-metric component and select its lowest root.

        Solving the connected components independently is required for an
        exact global result: the current tensor can have zero overlap with the
        component containing the lowest eigenvalue.
        """
        initial_vector = np.asarray(initial_vector)
        if initial_vector.size != self.layout.size:
            raise ValueError(
                f"initial_vector must contain {self.layout.size} entries."
            )
        initial_vector = initial_vector.reshape(-1)
        dense_component_max_size = davidson_options.pop(
            "dense_component_max_size",
            0,
        )
        parallel_components = bool(davidson_options.pop("parallel_components", False))
        max_component_workers = davidson_options.pop("max_component_workers", None)
        if max_component_workers is not None:
            max_component_workers = int(max_component_workers)
            if max_component_workers < 1:
                raise ValueError("max_component_workers must be positive or None.")
        if dense_component_max_size is None:
            dense_component_max_size = 0
        dense_component_max_size = int(dense_component_max_size)
        if dense_component_max_size < 0:
            raise ValueError("dense_component_max_size must be nonnegative.")
        tolerance = float(davidson_options.get("tol", 1.0e-10))
        absolute_tolerance = float(davidson_options.get("atol", 0.0))
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        metric_tol = float(davidson_options.get("metric_tol", 1.0e-12))
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        metric_scale = max(
            (
                float(np.linalg.norm(block, ord=np.inf))
                for block in self.metric.blocks.values()
            ),
            default=0.0,
        )
        metric_scale = max(metric_scale, np.finfo(float).tiny)
        metric_threshold = metric_tol * metric_scale
        metric_bases = {}
        for block, (eigenvalues, eigenvectors) in enumerate(
            self.metric_eigensystems()
        ):
            keep = eigenvalues > metric_threshold
            if np.any(keep):
                metric_bases[block] = (
                    eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
                )
        active_neighbors = {block: set() for block in metric_bases}
        for row, column in self.hamiltonian.connected_pairs:
            if row in active_neighbors and column in active_neighbors:
                active_neighbors[row].add(column)
                active_neighbors[column].add(row)
        components = []
        unseen = set(metric_bases)
        while unseen:
            root = min(unseen)
            stack = [root]
            unseen.remove(root)
            component = []
            while stack:
                block = stack.pop()
                component.append(block)
                new = active_neighbors[block] & unseen
                unseen.difference_update(new)
                stack.extend(sorted(new, reverse=True))
            components.append(tuple(sorted(component)))
        components = tuple(components)
        candidates = []
        total_iterations = 0
        total_hamiltonian_matvecs = 0
        total_metric_matvecs = 0
        total_restarts = 0
        total_subspace = 0
        component_dimensions = []
        dense_components = []

        def solve_component(component_item):
            component_index, component = component_item
            active_blocks = component
            block_ranges = {}
            active_layout = []
            component_size = 0
            for block in active_blocks:
                basis = metric_bases[block]
                start = component_size
                component_size += basis.shape[1]
                block_ranges[block] = slice(start, component_size)
                active_layout.append((block, start, component_size, basis))
            dense_component = component_size <= dense_component_max_size

            stored_hamiltonian_blocks = getattr(
                self.hamiltonian,
                "blocks",
                None,
            )
            if stored_hamiltonian_blocks is not None:
                reduced_hamiltonian_blocks = tuple(
                    (
                        block_ranges[row],
                        block_ranges[column],
                        metric_bases[row].T.conj()
                        @ block
                        @ metric_bases[column],
                    )
                    for (row, column), block in stored_hamiltonian_blocks.items()
                    if row in block_ranges and column in block_ranges
                )

                def hamiltonian_action(
                    vector,
                    *,
                    entries=reduced_hamiltonian_blocks,
                    size=component_size,
                ):
                    vector = np.asarray(vector)
                    output = np.zeros(
                        size,
                        dtype=np.result_type(
                            vector.dtype,
                            self.hamiltonian.dtype,
                        ),
                    )
                    for row_slice, column_slice, block in entries:
                        output[row_slice] += block @ vector[column_slice]
                    return output

                def hamiltonian_actions(vectors):
                    vectors = np.asarray(vectors)
                    return np.stack(
                        [hamiltonian_action(vector) for vector in vectors]
                    )

                hamiltonian_action.many = hamiltonian_actions
                verification_hamiltonian_action = None

                reduced_diagonal = np.zeros(
                    component_size,
                    dtype=np.result_type(self.hamiltonian.dtype, np.float64),
                )
                for block in active_blocks:
                    diagonal_block = stored_hamiltonian_blocks.get((block, block))
                    if diagonal_block is None:
                        continue
                    basis = metric_bases[block]
                    reduced_diagonal[block_ranges[block]] = np.diag(
                        basis.T.conj() @ diagonal_block @ basis
                    )

            else:
                reduced_hamiltonian_blocks = None
                matrix_free_layout = tuple(active_layout)
                diagonal_blocks = getattr(
                    self.hamiltonian,
                    "diagonal_blocks",
                    None,
                )
                reduced_diagonal = None
                if diagonal_blocks is not None:
                    reduced_diagonal = np.zeros(
                        component_size,
                        dtype=np.result_type(
                            self.hamiltonian.dtype,
                            np.float64,
                        ),
                    )
                    for block in active_blocks:
                        diagonal_block = diagonal_blocks.get(block)
                        if diagonal_block is None:
                            continue
                        basis = metric_bases[block]
                        reduced_diagonal[block_ranges[block]] = np.diag(
                            basis.T.conj() @ diagonal_block @ basis
                        )

                def hamiltonian_action(
                    vector,
                    *,
                    entries=matrix_free_layout,
                    size=component_size,
                ):
                    vector = np.asarray(vector).reshape(-1)
                    full = np.zeros(
                        self.layout.size,
                        dtype=np.result_type(
                            vector.dtype,
                            self.hamiltonian.dtype,
                        ),
                    )
                    for block, start, stop, basis in entries:
                        full[self.layout.block_indices[block]] = (
                            basis @ vector[start:stop]
                        )
                    applied = self.hamiltonian.matvec(full)
                    output = np.empty(
                        size,
                        dtype=np.result_type(
                            applied.dtype,
                            self.hamiltonian.dtype,
                        ),
                    )
                    for block, start, stop, basis in entries:
                        output[start:stop] = (
                            basis.T.conj()
                            @ applied[self.layout.block_indices[block]]
                        )
                    return output

                def hamiltonian_actions(
                    vectors,
                    *,
                    entries=matrix_free_layout,
                    size=component_size,
                ):
                    vectors = np.asarray(vectors)
                    full = np.zeros(
                        (vectors.shape[0], self.layout.size),
                        dtype=np.result_type(
                            vectors.dtype,
                            self.hamiltonian.dtype,
                        ),
                    )
                    for block, start, stop, basis in entries:
                        full[:, self.layout.block_indices[block]] = (
                            vectors[:, start:stop] @ basis.T
                        )
                    applied = self.hamiltonian.matvecs(full)
                    output = np.empty(
                        (vectors.shape[0], size),
                        dtype=np.result_type(
                            applied.dtype,
                            self.hamiltonian.dtype,
                        ),
                    )
                    for block, start, stop, basis in entries:
                        output[:, start:stop] = (
                            applied[:, self.layout.block_indices[block]]
                            @ basis.conj()
                        )
                    return output

                hamiltonian_action.many = hamiltonian_actions
                if self.hamiltonian.has_verification_action:
                    def verification_hamiltonian_action(
                        vector,
                        *,
                        entries=matrix_free_layout,
                        size=component_size,
                    ):
                        vector = np.asarray(vector).reshape(-1)
                        full = np.zeros(
                            self.layout.size,
                            dtype=np.result_type(
                                vector.dtype,
                                self.hamiltonian.dtype,
                            ),
                        )
                        for block, start, stop, basis in entries:
                            full[self.layout.block_indices[block]] = (
                                basis @ vector[start:stop]
                            )
                        applied = self.hamiltonian.verification_matvec(full)
                        output = np.empty(
                            size,
                            dtype=np.result_type(
                                applied.dtype,
                                self.hamiltonian.dtype,
                            ),
                        )
                        for block, start, stop, basis in entries:
                            output[start:stop] = (
                                basis.T.conj()
                                @ applied[self.layout.block_indices[block]]
                            )
                        return output
                else:
                    verification_hamiltonian_action = None

            def metric_action(vector):
                return np.array(vector, copy=True)

            metric_action.many = lambda vectors: np.array(vectors, copy=True)

            component_initial = np.concatenate(
                [
                    basis.T.conj()
                    @ (
                        self.metric.blocks[(block, block)]
                        @ initial_vector[self.layout.block_indices[block]]
                    )
                    for block, _start, _stop, basis in active_layout
                ]
            )
            if float(np.linalg.norm(component_initial)) <= (
                128.0 * np.finfo(float).eps * np.sqrt(component_size)
            ):
                component_initial = np.ones(
                    component_size,
                    dtype=np.result_type(initial_vector.dtype, self.metric.dtype),
                )

            if component_size <= dense_component_max_size:
                dense_hamiltonian = np.zeros(
                    (component_size, component_size),
                    dtype=np.result_type(
                        initial_vector.dtype,
                        self.hamiltonian.dtype,
                    ),
                )
                for row_slice, column_slice, block in reduced_hamiltonian_blocks:
                    dense_hamiltonian[row_slice, column_slice] += block
                dense_hamiltonian = 0.5 * (
                    dense_hamiltonian + dense_hamiltonian.T.conj()
                )
                eigenvalues, eigenvectors = linalg.eigh(
                    dense_hamiltonian,
                    subset_by_index=[0, 0],
                    driver="evx",
                    check_finite=False,
                )
                energy = float(np.real(eigenvalues[0]))
                vector = eigenvectors[:, 0]
                h_vector = hamiltonian_action(vector)
                residual = h_vector - energy * vector
                residual_norm = float(np.linalg.norm(residual))
                diagnostics = DavidsonDiagnostics(
                    converged=True,
                    message="dense conditional component diagonalization",
                    iterations=1,
                    hamiltonian_matvecs=1,
                    metric_matvecs=1,
                    restarts=0,
                    residual_norm=residual_norm,
                    metric_norm=float(np.vdot(vector, vector).real),
                    projected_rank=component_size,
                    subspace_dimension=component_size,
                    energy_history=(energy,),
                    residual_history=(residual_norm,),
                )
            elif component_size == 1:
                unit = np.ones(1, dtype=component_initial.dtype)
                metric_value = float(np.real(metric_action(unit)[0]))
                vector = unit / np.sqrt(metric_value)
                h_vector = hamiltonian_action(vector)
                n_vector = metric_action(vector)
                energy = float(np.real(np.vdot(vector, h_vector)))
                residual = h_vector - energy * n_vector
                diagnostics = DavidsonDiagnostics(
                    converged=True,
                    message="converged",
                    iterations=1,
                    hamiltonian_matvecs=1,
                    metric_matvecs=2,
                    restarts=0,
                    residual_norm=float(np.linalg.norm(residual)),
                    metric_norm=float(np.real(np.vdot(vector, n_vector))),
                    projected_rank=1,
                    subspace_dimension=1,
                    energy_history=(energy,),
                    residual_history=(float(np.linalg.norm(residual)),),
                )
            else:
                options = dict(davidson_options)
                if options.get("max_subspace") is not None:
                    options["max_subspace"] = min(
                        component_size, int(options["max_subspace"])
                    )
                recycle_key = (
                    *recycle_prefix,
                    component_index,
                    component_size,
                )
                recycle_out = []
                use_recycle = (
                    recycle_spaces is not None
                    and component_size >= recycle_min_size
                )
                if use_recycle:
                    options["initial_subspace"] = recycle_spaces.get(
                        recycle_key
                    )
                    options["recycle_out"] = recycle_out
                if callable(preconditioner_option):
                    options["preconditioner"] = preconditioner_option
                elif (
                    preconditioner_option in {"auto", "jacobi"}
                    and reduced_diagonal is not None
                ):
                    diagonal_scale = max(
                        float(np.max(np.abs(reduced_diagonal))),
                        np.finfo(float).tiny,
                    )
                    denominator_floor = (
                        np.sqrt(np.finfo(float).eps) * diagonal_scale
                    )

                    def jacobi_preconditioner(
                        residual,
                        eigenvalue,
                        *,
                        diagonal=reduced_diagonal,
                        floor=denominator_floor,
                    ):
                        denominator = diagonal - eigenvalue
                        phase = np.ones_like(denominator)
                        nonzero = np.abs(denominator) > 0.0
                        phase[nonzero] = denominator[nonzero] / np.abs(
                            denominator[nonzero]
                        )
                        safe = np.where(
                            np.abs(denominator) >= floor,
                            denominator,
                            floor * phase,
                        )
                        return np.asarray(residual) / safe

                    options["preconditioner"] = jacobi_preconditioner
                if verification_hamiltonian_action is not None:
                    options["verification_hamiltonian_action"] = (
                        verification_hamiltonian_action
                    )
                energy, vector, diagnostics = lowest_generalized_davidson(
                    hamiltonian_action,
                    metric_action,
                    component_initial,
                    hamiltonian_actions=hamiltonian_action.many,
                    metric_actions=metric_action.many,
                    **options,
                )
                if use_recycle:
                    recycle_spaces[recycle_key] = tuple(recycle_out)
            if not diagnostics.converged:
                raise ValueError(
                    f"physical-block component {component_index} failed: "
                    f"{diagnostics.message}"
                )
            return (
                float(energy),
                component_index,
                tuple(active_layout),
                vector,
                diagnostics,
                component_size,
                component_size <= dense_component_max_size,
            )

        component_items = tuple(enumerate(components))
        if parallel_components and len(component_items) > 1:
            with ThreadPoolExecutor(max_workers=max_component_workers) as executor:
                component_results = tuple(executor.map(solve_component, component_items))
        else:
            component_results = tuple(solve_component(item) for item in component_items)

        for result in component_results:
            (
                energy,
                component_index,
                active_layout,
                vector,
                diagnostics,
                component_size,
                dense_component,
            ) = result
            total_iterations += diagnostics.iterations
            total_hamiltonian_matvecs += diagnostics.hamiltonian_matvecs
            total_metric_matvecs += diagnostics.metric_matvecs
            total_restarts += diagnostics.restarts
            total_subspace += diagnostics.subspace_dimension
            component_dimensions.append(component_size)
            dense_components.append(dense_component)
            candidates.append((energy, component_index, active_layout, vector, diagnostics))

        indexed_components = tuple(enumerate(components))
        if executor is not None and len(indexed_components) > 1:
            solved = list(executor.map(solve_component, indexed_components))
        else:
            solved = [solve_component(item) for item in indexed_components]
        candidates = [item[:5] for item in solved]
        total_iterations = sum(item[4].iterations for item in solved)
        total_hamiltonian_matvecs = sum(
            item[4].hamiltonian_matvecs for item in solved
        )
        total_metric_matvecs = sum(item[4].metric_matvecs for item in solved)
        total_restarts = sum(item[4].restarts for item in solved)
        total_subspace = sum(item[4].subspace_dimension for item in solved)
        component_dimensions = [item[5] for item in solved]
        dense_components = [item[6] for item in solved]

        if not candidates:
            raise ValueError("local overlap metric is numerically singular.")
        energy, selected_index, selected_layout, selected_vector, selected = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        vector = np.zeros(
            self.layout.size,
            dtype=np.result_type(selected_vector.dtype, initial_vector.dtype),
        )
        for block, start, stop, basis in selected_layout:
            vector[self.layout.block_indices[block]] = (
                basis @ selected_vector[start:stop]
            )
        n_vector = self.metric.matvec(vector)
        metric_norm = float(np.real(np.vdot(vector, n_vector)))
        # Each disconnected positive-metric component was solved and checked
        # independently.  Tiny action leakage into structurally disconnected
        # blocks is contraction roundoff, not a missing variational direction.
        converged = bool(selected.converged)
        residual_norm = float(selected.residual_norm)
        diagnostics = PhysicalBlockSolveDiagnostics(
            converged=bool(converged),
            message=(
                (
                    f"converged in {len(candidates)} positive-metric physical-block "
                    f"component(s); selected component {selected_index}"
                )
                if converged
                else selected.message
            ),
            iterations=total_iterations,
            hamiltonian_matvecs=total_hamiltonian_matvecs,
            metric_matvecs=total_metric_matvecs + 1,
            restarts=total_restarts,
            residual_norm=residual_norm,
            metric_norm=metric_norm,
            projected_rank=self.metric_rank(metric_tol),
            subspace_dimension=total_subspace,
            energy_history=selected.energy_history,
            residual_history=selected.residual_history,
            component_sizes=tuple(len(component) for component in components),
            positive_metric_components=len(candidates),
            selected_component=selected_index,
            metric_blocks=len(self.metric.blocks),
            hamiltonian_blocks=self.hamiltonian.block_count,
            stored_elements=self.stored_elements,
            component_dimensions=tuple(component_dimensions),
            dense_components=tuple(dense_components),
        )
        return energy, vector, diagnostics


__all__ = [
    "PhysicalBlockGeneralizedProblem",
    "PhysicalBlockLayout",
    "PhysicalBlockLinearOperator",
    "MatrixFreePhysicalBlockOperator",
    "PhysicalBlockSolveDiagnostics",
    "hamiltonian_physical_connectivity",
]
