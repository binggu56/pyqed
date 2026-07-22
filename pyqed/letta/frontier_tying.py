"""Exact and tensor-train graph-tied LETTA frontier contraction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .block_mpo_frontier import BlockMPOFrontier
from .core import (
    _lowest_generalized_eigenpair,
    _lowest_hermitian_eigenpair,
    _metric_basis,
)
from .cp_tying import (
    _normalized_fidelity,
    _validated_dims,
    _validated_parent_sets,
)
from .local_terms import LocalHamiltonian, LocalMPO
from .matrix_free import lowest_generalized_davidson
from .mpo_frontier import MPOFrontier
from .physical_blocks import (
    PhysicalBlockGeneralizedProblem,
    PhysicalBlockLayout,
    hamiltonian_physical_connectivity,
)
from .tt_frontier import TermwiseTTMPOFrontier, TTFrontier, TTMPOFrontier


@dataclass(frozen=True)
class FrontierSiteUpdate:
    """Diagnostic record for one frontier local-tensor update."""

    site: int
    raw_dim: int
    metric_rank: int
    metric_rank_is_projected: bool
    solver: str
    solver_converged: bool
    message: str
    hamiltonian_matvecs: int
    metric_matvecs: int
    iterations: int
    residual_norm: float
    energy_before: float
    energy: float
    accepted: bool
    physical_blocks: int = 0
    hamiltonian_blocks: int = 0
    block_component_sizes: tuple[int, ...] = ()
    stored_operator_elements: int = 0
    solver_metric_is_identity: bool = False
    solver_metric_identity_error: float = float("nan")
    solver_coordinate_residual_norm: float = float("nan")


@dataclass(frozen=True)
class FrontierSiteEnvironment:
    """Numerical left/right messages surrounding one local tensor."""

    site: int
    norm_left: object
    norm_right: object
    hamiltonian_left: object
    hamiltonian_right: object


@dataclass(frozen=True)
class FrontierNaturalGradientUpdate:
    """Diagnostic record for one simultaneous metric-preconditioned step."""

    energy_before: float
    energy: float
    accepted: bool
    message: str
    step_size: float
    backtracks: int
    gradient_norm: float
    preconditioned_norm: float
    metric_direction_norm: float
    directional_derivative: float
    max_relative_direction: float
    metric_ranks: tuple[int, ...]


@dataclass(frozen=True)
class FrontierGaugeUpdate:
    """Diagnostic record for one virtual-bond frontier gauge."""

    cut: int
    frontier_sites: tuple[int, ...]
    applied: bool
    message: str
    left_rank: int
    right_rank: int
    left_condition: float
    right_condition: float
    balanced_condition: float
    gauge_condition: float
    imbalance_before: float
    imbalance_after: float


@dataclass(frozen=True)
class FrontierBondExpansion:
    """Diagnostic record for one ansatz-preserving virtual-bond expansion."""

    cut: int
    old_dimension: int
    new_dimension: int
    seeded_directions: int
    direction: str
    strategy: str
    source_norm: float
    norm_error: float
    energy_before: float
    energy: float


@dataclass(frozen=True)
class FrontierTwoSiteUpdate:
    """Diagnostic record for one merge--optimize--split pair update."""

    sites: tuple[int, int]
    overlap_sites: tuple[int, ...]
    raw_merged_dim: int
    old_bond_dimension: int
    temporary_bond_dimension: int
    conditional_ranks: tuple[int, ...]
    relative_truncation_error: float
    energy_before: float
    merged_energy: float
    attempted_energy: float
    energy: float
    accepted: bool
    local_update: FrontierSiteUpdate
    split_strategy: str = "svd"
    selected_start: str = "svd"
    metric_projection_error: float = float("nan")
    factor_sweeps: int = 0
    factor_accepted_updates: int = 0
    factor_random_starts: int = 0


class FrontierTiedLETTA:
    r"""Unrestricted graph-tied LETTA contracted by frontier messages.

    This class represents the same local tensors as :class:`DenseTiedLETTA`,
    but it accepts a :class:`LocalHamiltonian` and never constructs the
    many-body configuration table during initialization or optimization.
    The local-term sum is converted to an exact finite-state MPO.  Numerical
    left/right double-layer messages are cached across each directional sweep
    and reused by every local matrix or Davidson action.  The ``compressed``
    and ``identity_block`` backends are exact.  The ``tensor_train`` backend
    keeps the cheaper norm frontier exact by default and stores the Hamiltonian
    frontier as a boundary MPS/TT.  It is fully exact only when its ranks and
    tolerances are unrestricted.  The dense-frontier cost is governed
    by the weighted frontier induced by the chosen site ordering and MPO bond;
    it is exponential in that width, which can still grow with system size for
    dense or poorly ordered graphs.
    """

    def __init__(
        self,
        hamiltonian: LocalHamiltonian,
        dims,
        parent_sets,
        *,
        bond_dim: int = 1,
        bond_dims=None,
        tensors=None,
        seed: int | None = None,
        frontier_backend="compressed",
        path_optimizer="greedy",
        tt_max_rank: int | None = None,
        tt_rtol: float = 0.0,
        tt_atol: float = 0.0,
        tt_transfer_max_rank: int | None = None,
        tt_transfer_rtol: float = 0.0,
        tt_transfer_atol: float = 0.0,
        tt_absorption="structured",
        tt_norm_backend="exact",
        tt_hermitize: bool = True,
    ):
        if not isinstance(hamiltonian, LocalHamiltonian):
            raise TypeError("hamiltonian must be a LocalHamiltonian.")
        self.dims = _validated_dims(dims)
        if hamiltonian.dims != self.dims:
            raise ValueError("hamiltonian dims are inconsistent with dims.")
        self.hamiltonian = hamiltonian
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.physical_sites = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        self._physical_block_connectivity_cache = {}
        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        if bond_dims is None:
            self._virtual_bond_dims = (
                (1,) + (bond_dim,) * max(0, len(self.dims) - 1) + (1,)
            )
        else:
            dimensions = tuple(int(dimension) for dimension in bond_dims)
            if len(dimensions) == max(0, len(self.dims) - 1):
                dimensions = (1,) + dimensions + (1,)
            elif len(dimensions) != len(self.dims) + 1:
                raise ValueError(
                    "bond_dims must contain the internal dimensions or all "
                    "dimensions including the two boundaries."
                )
            if dimensions[0] != 1 or dimensions[-1] != 1:
                raise ValueError("the boundary bond dimensions must be one.")
            if any(dimension < 1 for dimension in dimensions):
                raise ValueError("bond_dims must contain positive dimensions.")
            self._virtual_bond_dims = dimensions
        # Retain the historical scalar attribute as the maximum internal
        # dimension.  New code should use ``bond_dims`` or ``_bond_dims()``.
        self.bond_dim = max(self._virtual_bond_dims)
        self.path_optimizer = path_optimizer
        self.frontier_backend = str(frontier_backend).lower().replace("-", "_")
        if self.frontier_backend in {"tt", "boundary_mps"}:
            self.frontier_backend = "tensor_train"
        if self.frontier_backend not in {
            "compressed",
            "identity_block",
            "tensor_train",
        }:
            raise ValueError(
                "frontier_backend must be 'compressed', 'identity_block', "
                "or 'tensor_train'."
            )
        self.tt_options = {
            "max_rank": tt_max_rank,
            "rtol": tt_rtol,
            "atol": tt_atol,
            "transfer_max_rank": tt_transfer_max_rank,
            "transfer_rtol": tt_transfer_rtol,
            "transfer_atol": tt_transfer_atol,
            "absorption": tt_absorption,
        }
        self.tt_norm_backend = str(tt_norm_backend).lower().replace("-", "_")
        if self.tt_norm_backend in {"dense", "mpo"}:
            self.tt_norm_backend = "exact"
        if self.tt_norm_backend in {"tt", "boundary_mps"}:
            self.tt_norm_backend = "tensor_train"
        if self.tt_norm_backend not in {"exact", "tensor_train"}:
            raise ValueError("tt_norm_backend must be 'exact' or 'tensor_train'.")
        if not isinstance(tt_hermitize, (bool, np.bool_)):
            raise TypeError("tt_hermitize must be boolean.")
        self.tt_hermitize = bool(tt_hermitize)
        self.rng = np.random.default_rng(seed)

        bonds = self._bond_dims()
        shapes = tuple(
            (bonds[site], bonds[site + 1])
            + tuple(self.dims[index] for index in physical_sites)
            for site, physical_sites in enumerate(self.physical_sites)
        )
        parameter_dtype = np.result_type(self.hamiltonian.dtype, np.float64)
        if tensors is None:
            self.tensors = []
            for shape in shapes:
                tensor = self.rng.normal(size=shape) / np.sqrt(np.prod(shape))
                self.tensors.append(tensor.astype(parameter_dtype))
        else:
            if len(tensors) != len(self.dims):
                raise ValueError("tensors must contain one entry per site.")
            self.tensors = [
                np.asarray(
                    tensor,
                    dtype=np.result_type(np.asarray(tensor).dtype, parameter_dtype),
                ).copy()
                for tensor in tensors
            ]
            for site, (tensor, shape) in enumerate(zip(self.tensors, shapes)):
                if tensor.shape != shape:
                    raise ValueError(f"tensor {site} shape must be {shape}.")

        identity_mpo = LocalMPO(
            self.dims,
            [np.eye(dim, dtype=parameter_dtype)[None, None, :, :] for dim in self.dims],
        )
        uncompressed_hamiltonian_mpo = self.hamiltonian.to_mpo()
        self.uncompressed_hamiltonian_mpo_bond_dim = max(
            uncompressed_hamiltonian_mpo.bond_dims
        )
        compressed_hamiltonian_mpo = uncompressed_hamiltonian_mpo.compress()
        self.compressed_hamiltonian_mpo_bond_dim = max(
            compressed_hamiltonian_mpo.bond_dims
        )
        frontier_arguments = (self.dims, self.physical_sites, shapes)
        if (
            self.frontier_backend == "tensor_train"
            and self.tt_norm_backend == "tensor_train"
        ):
            self._norm_frontier = TTMPOFrontier(
                *frontier_arguments,
                identity_mpo.tensors,
                paired_sites=(),
                optimize=path_optimizer,
                **self.tt_options,
            )
        else:
            self._norm_frontier = MPOFrontier(
                *frontier_arguments,
                identity_mpo.tensors,
                paired_sites=(),
                optimize=path_optimizer,
            )
        if self.frontier_backend == "compressed":
            self.hamiltonian_mpo = compressed_hamiltonian_mpo
            self._hamiltonian_frontier = MPOFrontier(
                self.dims,
                self.physical_sites,
                shapes,
                self.hamiltonian_mpo.tensors,
                optimize=path_optimizer,
            )
        elif self.frontier_backend == "identity_block":
            self.hamiltonian_mpo = uncompressed_hamiltonian_mpo
            self._hamiltonian_frontier = BlockMPOFrontier(
                *frontier_arguments,
                self.hamiltonian_mpo.tensors,
                optimize=path_optimizer,
            )
        else:
            self.hamiltonian_mpo = compressed_hamiltonian_mpo
            self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                self.hamiltonian,
                self.physical_sites,
                shapes,
                optimize=path_optimizer,
                **self.tt_options,
            )
        self.history: list[dict] = []
        self.energy: float | None = None
        self.converged = False
        if self.norm_contraction_is_exact:
            self.balance_gauges()
        else:
            # A strongly truncated double layer need not remain positive, so
            # it is unsafe to use its approximate norm for initialization.
            # This balances tensor magnitudes with a net unit rescaling.
            self.balance_gauges(state_norm=1.0)
        self.energy = self.expectation()

    def _bond_dims(self):
        return self._virtual_bond_dims

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Virtual dimensions at every cut, including unit boundaries."""
        return self._bond_dims()

    @classmethod
    def from_dense(cls, state, hamiltonian: LocalHamiltonian, **kwargs):
        """Copy tensors from a dense-projector reference state."""
        result = cls(
            hamiltonian,
            state.dims,
            state.parent_sets,
            bond_dim=state.bond_dim,
            bond_dims=getattr(state, "bond_dims", None),
            tensors=[tensor.copy() for tensor in state.tensors],
            **kwargs,
        )
        result.rng.bit_generator.state = deepcopy(state.rng.bit_generator.state)
        return result

    def copy(self):
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            bond_dim=self.bond_dim,
            bond_dims=self.bond_dims,
            tensors=[tensor.copy() for tensor in self.tensors],
            frontier_backend=self.frontier_backend,
            path_optimizer=self.path_optimizer,
            tt_max_rank=self.tt_options["max_rank"],
            tt_rtol=self.tt_options["rtol"],
            tt_atol=self.tt_options["atol"],
            tt_transfer_max_rank=self.tt_options["transfer_max_rank"],
            tt_transfer_rtol=self.tt_options["transfer_rtol"],
            tt_transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
        )
        # Construction balances tensor magnitudes.  Restore the represented
        # state exactly, then recompute the approximate energy consistently.
        result.tensors = [tensor.copy() for tensor in self.tensors]
        result.history = list(self.history)
        result.energy = result.expectation()
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    def _rebuild_frontier_engines(self):
        """Replan contractions after a virtual-bond shape change."""
        shapes = tuple(tuple(tensor.shape) for tensor in self.tensors)
        bonds = self._bond_dims()
        for site, shape in enumerate(shapes):
            if shape[:2] != (bonds[site], bonds[site + 1]):
                raise ValueError(
                    f"tensor {site} virtual shape is inconsistent with bond_dims."
                )
        parameter_dtype = np.result_type(
            self.hamiltonian.dtype,
            *[tensor.dtype for tensor in self.tensors],
        )
        identity_mpo = LocalMPO(
            self.dims,
            [np.eye(dim, dtype=parameter_dtype)[None, None, :, :] for dim in self.dims],
        )
        frontier_arguments = (self.dims, self.physical_sites, shapes)
        if (
            self.frontier_backend == "tensor_train"
            and self.tt_norm_backend == "tensor_train"
        ):
            self._norm_frontier = TTMPOFrontier(
                *frontier_arguments,
                identity_mpo.tensors,
                paired_sites=(),
                optimize=self.path_optimizer,
                **self.tt_options,
            )
        else:
            self._norm_frontier = MPOFrontier(
                *frontier_arguments,
                identity_mpo.tensors,
                paired_sites=(),
                optimize=self.path_optimizer,
            )
        if self.frontier_backend == "compressed":
            self._hamiltonian_frontier = MPOFrontier(
                *frontier_arguments,
                self.hamiltonian_mpo.tensors,
                optimize=self.path_optimizer,
            )
        elif self.frontier_backend == "identity_block":
            self._hamiltonian_frontier = BlockMPOFrontier(
                *frontier_arguments,
                self.hamiltonian_mpo.tensors,
                optimize=self.path_optimizer,
            )
        else:
            self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                self.hamiltonian,
                self.physical_sites,
                shapes,
                optimize=self.path_optimizer,
                **self.tt_options,
            )

    @staticmethod
    def _random_matrix(shape, dtype, rng):
        matrix = rng.normal(size=shape)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            matrix = matrix + 1j * rng.normal(size=shape)
        return np.asarray(matrix, dtype=dtype)

    @classmethod
    def _orthogonal_enrichment(cls, old, source, count, *, rng):
        """Return source-led columns orthogonal to the existing column space."""
        old = np.asarray(old)
        source = np.asarray(source, dtype=np.result_type(old.dtype, source.dtype))
        count = int(count)
        if count < 1:
            return np.zeros((old.shape[0], 0), dtype=source.dtype)
        left, singular_values, _right = np.linalg.svd(old, full_matrices=False)
        scale = max(float(np.max(singular_values, initial=0.0)), 1.0)
        rank = int(
            np.count_nonzero(singular_values > 256.0 * np.finfo(float).eps * scale)
        )
        occupied = left[:, :rank]

        def projected(matrix, basis):
            if basis.shape[1]:
                matrix = matrix - basis @ (basis.conj().T @ matrix)
            return matrix

        source = projected(source, occupied)
        candidates, candidate_values, _right = np.linalg.svd(
            source,
            full_matrices=False,
        )
        candidate_scale = max(
            float(np.max(candidate_values, initial=0.0)),
            1.0,
        )
        candidate_rank = int(
            np.count_nonzero(
                candidate_values > 256.0 * np.finfo(float).eps * candidate_scale
            )
        )
        directions = candidates[:, : min(count, candidate_rank)]
        missing = min(count, old.shape[0] - rank) - directions.shape[1]
        while missing > 0:
            basis = np.concatenate((occupied, directions), axis=1)
            trial = cls._random_matrix(
                (old.shape[0], max(missing, 2)),
                source.dtype,
                rng,
            )
            trial = projected(trial, basis)
            trial, values, _right = np.linalg.svd(trial, full_matrices=False)
            threshold = (
                256.0
                * np.finfo(float).eps
                * max(float(np.max(values, initial=0.0)), 1.0)
            )
            usable = min(missing, int(np.count_nonzero(values > threshold)))
            if usable == 0:
                break
            directions = np.concatenate((directions, trial[:, :usable]), axis=1)
            missing -= usable
        return directions

    def expand_bond(
        self,
        cut: int,
        new_dimension: int,
        *,
        direction="right",
        strategy="residual",
        scale: float = 1.0e-3,
        seed=None,
    ) -> FrontierBondExpansion:
        r"""Open ansatz-preserving variational directions at one chain cut.

        An unrestricted two-site tensor over the union of tied physical legs
        need not split back at the old middle rank.  This method implements a
        safe one-site subspace-expansion alternative: it adds orthogonal
        residual-led channels to one tensor and initializes the matching
        channels of its neighbor to exactly zero.  The represented state is
        therefore unchanged before subsequent one-site relaxation.
        """
        cut = int(cut)
        if cut <= 0 or cut >= len(self.dims):
            raise ValueError("cut must be an internal virtual bond.")
        new_dimension = int(new_dimension)
        old_dimension = self._bond_dims()[cut]
        if new_dimension < old_dimension:
            raise ValueError("expand_bond only supports increasing dimensions.")
        direction = str(direction).lower().replace("_", "-")
        if direction in {"lr", "left-to-right", "forward"}:
            direction = "right"
        elif direction in {"rl", "right-to-left", "backward"}:
            direction = "left"
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'.")
        strategy = str(strategy).lower().replace("-", "_")
        if strategy not in {"residual", "random", "zero"}:
            raise ValueError("strategy must be 'residual', 'random', or 'zero'.")
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("scale must be finite and nonnegative.")

        energy_before = self.expectation()
        norm_before = float(np.real(self._norm_frontier.scalar(self.tensors)))
        if new_dimension == old_dimension:
            return FrontierBondExpansion(
                cut=cut,
                old_dimension=old_dimension,
                new_dimension=new_dimension,
                seeded_directions=0,
                direction=direction,
                strategy=strategy,
                source_norm=0.0,
                norm_error=0.0,
                energy_before=energy_before,
                energy=energy_before,
            )

        rng = self.rng if seed is None else np.random.default_rng(seed)
        left_site = cut - 1
        right_site = cut
        left_tensor = self.tensors[left_site]
        right_tensor = self.tensors[right_site]
        added = new_dimension - old_dimension
        source_norm = 0.0
        if direction == "right":
            axes = (0, *range(2, left_tensor.ndim), 1)
            inverse_axes = np.argsort(axes)
            old_ordered = left_tensor.transpose(axes)
            old_matrix = old_ordered.reshape(-1, old_dimension)
            if strategy == "residual":
                environment = self.site_environment(left_site)
                source_tensor = (
                    self.hamiltonian_action(
                        left_site,
                        left_tensor.reshape(-1),
                        environment=environment,
                    )
                    - energy_before
                    * self.metric_action(
                        left_site,
                        left_tensor.reshape(-1),
                        environment=environment,
                    )
                ).reshape(left_tensor.shape)
                source_matrix = source_tensor.transpose(axes).reshape(old_matrix.shape)
            elif strategy == "random":
                source_matrix = self._random_matrix(
                    (old_matrix.shape[0], max(old_dimension, added)),
                    left_tensor.dtype,
                    rng,
                )
            else:
                source_matrix = np.zeros_like(old_matrix)
            source_norm = float(np.linalg.norm(source_matrix))
            directions = (
                np.zeros((old_matrix.shape[0], 0), dtype=left_tensor.dtype)
                if strategy == "zero" or scale == 0.0
                else self._orthogonal_enrichment(
                    old_matrix,
                    source_matrix,
                    added,
                    rng=rng,
                )
            )
            amplitude = scale * max(
                float(np.linalg.norm(old_matrix)) / np.sqrt(old_dimension),
                np.finfo(float).tiny,
            )
            ordered_shape = old_ordered.shape[:-1] + (new_dimension,)
            expanded_ordered = np.zeros(
                ordered_shape,
                dtype=np.result_type(left_tensor.dtype, directions.dtype),
            )
            expanded_matrix = expanded_ordered.reshape(-1, new_dimension)
            expanded_matrix[:, :old_dimension] = old_matrix
            expanded_matrix[:, old_dimension : old_dimension + directions.shape[1]] = (
                amplitude * directions
            )
            expanded_left = expanded_ordered.transpose(inverse_axes)
            expanded_right = np.zeros(
                (new_dimension,) + right_tensor.shape[1:],
                dtype=right_tensor.dtype,
            )
            expanded_right[:old_dimension] = right_tensor
        else:
            old_matrix = right_tensor.reshape(old_dimension, -1)
            if strategy == "residual":
                environment = self.site_environment(right_site)
                source_matrix = (
                    self.hamiltonian_action(
                        right_site,
                        right_tensor.reshape(-1),
                        environment=environment,
                    )
                    - energy_before
                    * self.metric_action(
                        right_site,
                        right_tensor.reshape(-1),
                        environment=environment,
                    )
                ).reshape(old_matrix.shape)
            elif strategy == "random":
                source_matrix = self._random_matrix(
                    (max(old_dimension, added), old_matrix.shape[1]),
                    right_tensor.dtype,
                    rng,
                )
            else:
                source_matrix = np.zeros_like(old_matrix)
            source_norm = float(np.linalg.norm(source_matrix))
            directions = (
                np.zeros((0, old_matrix.shape[1]), dtype=right_tensor.dtype)
                if strategy == "zero" or scale == 0.0
                else self._orthogonal_enrichment(
                    old_matrix.T,
                    source_matrix.T,
                    added,
                    rng=rng,
                ).T
            )
            amplitude = scale * max(
                float(np.linalg.norm(old_matrix)) / np.sqrt(old_dimension),
                np.finfo(float).tiny,
            )
            expanded_right = np.zeros(
                (new_dimension,) + right_tensor.shape[1:],
                dtype=np.result_type(right_tensor.dtype, directions.dtype),
            )
            expanded_right[:old_dimension] = right_tensor
            expanded_right.reshape(new_dimension, -1)[
                old_dimension : old_dimension + directions.shape[0]
            ] = amplitude * directions
            expanded_left = np.zeros(
                (left_tensor.shape[0], new_dimension) + left_tensor.shape[2:],
                dtype=left_tensor.dtype,
            )
            expanded_left[:, :old_dimension] = left_tensor

        self.tensors[left_site] = expanded_left
        self.tensors[right_site] = expanded_right
        dimensions = list(self._bond_dims())
        dimensions[cut] = new_dimension
        self._virtual_bond_dims = tuple(dimensions)
        self.bond_dim = max(self._virtual_bond_dims)
        self._rebuild_frontier_engines()
        norm_after = float(np.real(self._norm_frontier.scalar(self.tensors)))
        energy_after = self.expectation()
        norm_error = abs(norm_after - norm_before) / max(abs(norm_before), 1.0)
        self.energy = energy_after
        self.history = []
        self.converged = False
        return FrontierBondExpansion(
            cut=cut,
            old_dimension=old_dimension,
            new_dimension=new_dimension,
            seeded_directions=int(directions.shape[1 if direction == "right" else 0]),
            direction=direction,
            strategy=strategy,
            source_norm=source_norm,
            norm_error=float(norm_error),
            energy_before=energy_before,
            energy=energy_after,
        )

    def expand_bond_dims(
        self,
        bond_dims,
        *,
        direction="right",
        strategy="residual",
        scale: float = 1.0e-3,
        seed=None,
    ) -> tuple[FrontierBondExpansion, ...]:
        """Increase selected per-cut dimensions with safe subspace seeds."""
        target = tuple(int(dimension) for dimension in bond_dims)
        if len(target) == max(0, len(self.dims) - 1):
            target = (1,) + target + (1,)
        if len(target) != len(self.dims) + 1:
            raise ValueError(
                "bond_dims must contain the internal dimensions or all "
                "dimensions including the two boundaries."
            )
        if target[0] != 1 or target[-1] != 1:
            raise ValueError("the boundary bond dimensions must be one.")
        current = self._bond_dims()
        if any(new < old for old, new in zip(current, target)):
            raise ValueError("expand_bond_dims only supports increasing dimensions.")
        normalized_direction = str(direction).lower().replace("_", "-")
        reverse = normalized_direction in {"left", "rl", "right-to-left", "backward"}
        cuts = range(len(self.dims) - 1, 0, -1) if reverse else range(1, len(self.dims))
        rng = self.rng if seed is None else np.random.default_rng(seed)
        records = []
        for cut in cuts:
            if target[cut] > self._bond_dims()[cut]:
                records.append(
                    self.expand_bond(
                        cut,
                        target[cut],
                        direction="left" if reverse else "right",
                        strategy=strategy,
                        scale=scale,
                        seed=int(rng.integers(np.iinfo(np.int64).max)),
                    )
                )
        return tuple(records)

    def _merged_pair_tensor(self, site):
        """Contract an adjacent pair while retaining each physical label once."""
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        following = site + 1
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
        union_sites = (site,) + tuple(
            sorted((set(left_sites) | set(right_sites)) - {site})
        )
        left_tensor = self.tensors[site]
        right_tensor = self.tensors[following]
        shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        merged = np.empty(
            shape,
            dtype=np.result_type(left_tensor.dtype, right_tensor.dtype),
        )
        physical_shape = tuple(self.dims[index] for index in union_sites)
        for configuration in np.ndindex(*physical_shape):
            values = dict(zip(union_sites, configuration))
            left = left_tensor[
                (slice(None), slice(None), *(values[index] for index in left_sites))
            ]
            right = right_tensor[
                (slice(None), slice(None), *(values[index] for index in right_sites))
            ]
            merged[(slice(None), slice(None), *configuration)] = left @ right
        return merged, union_sites

    def _split_merged_pair_tensor(self, site, merged, union_sites):
        r"""Conditionally SVD a merged pair back into its graph-leg pattern.

        If the two local tensors share tied physical labels, the split is one
        independent matrix factorization for every shared-label assignment.
        The usual MPS SVD is recovered when the physical-label sets are
        disjoint.
        """
        site = int(site)
        following = site + 1
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
        union_sites = tuple(int(index) for index in union_sites)
        expected_union = (site,) + tuple(
            sorted((set(left_sites) | set(right_sites)) - {site})
        )
        if union_sites != expected_union:
            raise ValueError("union_sites are inconsistent with the adjacent pair.")
        merged = np.asarray(merged)
        expected_shape = (
            self.tensors[site].shape[0],
            self.tensors[following].shape[1],
            *(self.dims[index] for index in union_sites),
        )
        if merged.shape != expected_shape:
            raise ValueError(f"merged tensor shape must be {expected_shape}.")

        overlap = tuple(sorted(set(left_sites) & set(right_sites)))
        left_only = tuple(index for index in left_sites if index not in overlap)
        right_only = tuple(index for index in right_sites if index not in overlap)
        overlap_shape = tuple(self.dims[index] for index in overlap)
        left_shape = tuple(self.dims[index] for index in left_only)
        right_shape = tuple(self.dims[index] for index in right_only)
        left_dimension = merged.shape[0]
        right_dimension = merged.shape[1]
        middle_dimension = self._bond_dims()[following]
        left_result = np.zeros_like(self.tensors[site], dtype=merged.dtype)
        right_result = np.zeros_like(self.tensors[following], dtype=merged.dtype)
        discarded_weight = 0.0
        total_weight = 0.0
        conditional_ranks = []
        overlap_configurations = np.ndindex(*overlap_shape) if overlap_shape else [()]
        for overlap_configuration in overlap_configurations:
            overlap_values = dict(zip(overlap, overlap_configuration))
            block = np.empty(
                (left_dimension, *left_shape, right_dimension, *right_shape),
                dtype=merged.dtype,
            )
            left_configurations = (
                tuple(np.ndindex(*left_shape)) if left_shape else ((),)
            )
            right_configurations = (
                tuple(np.ndindex(*right_shape)) if right_shape else ((),)
            )
            for left_configuration in left_configurations:
                for right_configuration in right_configurations:
                    values = dict(overlap_values)
                    values.update(zip(left_only, left_configuration))
                    values.update(zip(right_only, right_configuration))
                    union_configuration = tuple(values[index] for index in union_sites)
                    block[
                        (
                            slice(None),
                            *left_configuration,
                            slice(None),
                            *right_configuration,
                        )
                    ] = merged[(slice(None), slice(None), *union_configuration)]
            matrix = block.reshape(
                left_dimension * int(np.prod(left_shape, dtype=int)),
                right_dimension * int(np.prod(right_shape, dtype=int)),
            )
            left_vectors, singular_values, right_vectors = np.linalg.svd(
                matrix,
                full_matrices=False,
            )
            retained = min(middle_dimension, len(singular_values))
            scale = max(float(singular_values[0]), np.finfo(float).tiny)
            numerical_rank = int(
                np.count_nonzero(singular_values > 256.0 * np.finfo(float).eps * scale)
            )
            conditional_ranks.append(numerical_rank)
            total_weight += float(np.sum(singular_values**2))
            discarded_weight += float(np.sum(singular_values[retained:] ** 2))
            square_root = np.sqrt(singular_values[:retained])
            left_factor = (left_vectors[:, :retained] * square_root).reshape(
                left_dimension,
                *left_shape,
                retained,
            )
            right_factor = (square_root[:, None] * right_vectors[:retained]).reshape(
                retained,
                right_dimension,
                *right_shape,
            )
            for left_configuration in left_configurations:
                values = dict(overlap_values)
                values.update(zip(left_only, left_configuration))
                physical = tuple(values[index] for index in left_sites)
                left_result[(slice(None), slice(0, retained), *physical)] = left_factor[
                    (slice(None), *left_configuration, slice(None))
                ]
            for right_configuration in right_configurations:
                values = dict(overlap_values)
                values.update(zip(right_only, right_configuration))
                physical = tuple(values[index] for index in right_sites)
                right_result[(slice(0, retained), slice(None), *physical)] = (
                    right_factor[(slice(None), slice(None), *right_configuration)]
                )
        relative_error = np.sqrt(
            discarded_weight / max(total_weight, np.finfo(float).tiny)
        )
        return (
            left_result,
            right_result,
            overlap,
            tuple(conditional_ranks),
            float(relative_error),
        )

    def _pair_factor_design(
        self,
        site,
        union_sites,
        left_tensor,
        right_tensor,
        *,
        variable,
    ):
        r"""Return the exact linear map from one pair factor to its merge.

        The tied physical labels shared by the adjacent tensors are retained
        only once in the merged tensor.  Consequently this is not, in
        general, the ordinary Kronecker-product MPS design matrix.
        """
        site = int(site)
        following = site + 1
        union_sites = tuple(int(index) for index in union_sites)
        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        variable = str(variable).lower()
        if variable not in {"left", "right"}:
            raise ValueError("variable must be 'left' or 'right'.")
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
        expected_union = (site,) + tuple(
            sorted((set(left_sites) | set(right_sites)) - {site})
        )
        if union_sites != expected_union:
            raise ValueError("union_sites are inconsistent with the adjacent pair.")
        if left_tensor.shape[1] != right_tensor.shape[0]:
            raise ValueError("the pair factors have inconsistent middle dimensions.")
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        parameter_shape = (
            left_tensor.shape if variable == "left" else right_tensor.shape
        )
        design = np.zeros(
            (int(np.prod(merged_shape)), int(np.prod(parameter_shape))),
            dtype=np.result_type(left_tensor.dtype, right_tensor.dtype),
        )
        physical_shape = tuple(self.dims[index] for index in union_sites)
        for configuration in np.ndindex(*physical_shape):
            values = dict(zip(union_sites, configuration))
            left_configuration = tuple(values[index] for index in left_sites)
            right_configuration = tuple(values[index] for index in right_sites)
            for left_virtual in range(left_tensor.shape[0]):
                for right_virtual in range(right_tensor.shape[1]):
                    output = np.ravel_multi_index(
                        (left_virtual, right_virtual, *configuration),
                        merged_shape,
                    )
                    for middle in range(left_tensor.shape[1]):
                        if variable == "left":
                            parameter = np.ravel_multi_index(
                                (
                                    left_virtual,
                                    middle,
                                    *left_configuration,
                                ),
                                left_tensor.shape,
                            )
                            design[output, parameter] += right_tensor[
                                (middle, right_virtual, *right_configuration)
                            ]
                        else:
                            parameter = np.ravel_multi_index(
                                (
                                    middle,
                                    right_virtual,
                                    *right_configuration,
                                ),
                                right_tensor.shape,
                            )
                            design[output, parameter] += left_tensor[
                                (left_virtual, middle, *left_configuration)
                            ]
        return design

    def _merge_pair_factors(self, site, union_sites, left_tensor, right_tensor):
        """Merge explicitly supplied adjacent factors in the graph layout."""
        design = self._pair_factor_design(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variable="left",
        )
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        return (design @ np.asarray(left_tensor).reshape(-1)).reshape(merged_shape)

    @staticmethod
    def _pair_rayleigh(vector, metric, effective):
        vector = np.asarray(vector).reshape(-1)
        metric_vector = metric @ vector
        norm = np.vdot(vector, metric_vector)
        scale = max(
            float(np.linalg.norm(metric, ord=np.inf))
            * float(np.vdot(vector, vector).real),
            np.finfo(float).tiny,
        )
        if (
            not np.isfinite(norm)
            or float(np.real(norm)) <= 256.0 * np.finfo(float).eps * scale
        ):
            return float("inf")
        numerator = np.vdot(vector, effective @ vector)
        energy = float(np.real(numerator / norm))
        return energy if np.isfinite(energy) else float("inf")

    @staticmethod
    def _balance_pair_factors(left_tensor, right_tensor):
        left_norm = float(np.linalg.norm(left_tensor))
        right_norm = float(np.linalg.norm(right_tensor))
        if left_norm <= np.finfo(float).tiny or right_norm <= np.finfo(float).tiny:
            return left_tensor, right_tensor
        scale = np.sqrt(right_norm / left_norm)
        return left_tensor * scale, right_tensor / scale

    def _metric_project_pair_factors(
        self,
        site,
        target,
        union_sites,
        metric,
        left_tensor,
        right_tensor,
        *,
        metric_tol,
        max_sweeps,
    ):
        r"""Project a merged tensor with its physical environment metric.

        This minimizes the represented-state error

        .. math::

            \|A B-M_\star\|_{N_{\rm eff}}^2

        by alternating weighted least-squares solves.  It replaces the
        Euclidean Frobenius norm used by a conditional SVD.
        """
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).T.conj())
        values, vectors = np.linalg.eigh(metric)
        threshold = float(metric_tol) * max(
            float(np.linalg.norm(metric, ord=np.inf)),
            np.finfo(float).tiny,
        )
        keep = values > threshold
        if not np.any(keep):
            raise ValueError("the merged-pair metric is numerically singular.")
        root = np.sqrt(values[keep])[:, None] * vectors[:, keep].T.conj()
        target_vector = np.asarray(target).reshape(-1)
        weighted_target = root @ target_vector
        target_norm = max(float(np.linalg.norm(weighted_target)), np.finfo(float).tiny)
        left_tensor = np.asarray(left_tensor).copy()
        right_tensor = np.asarray(right_tensor).copy()
        approximation = self._merge_pair_factors(
            site,
            union_sites,
            left_tensor,
            right_tensor,
        ).reshape(-1)
        relative_error = float(
            np.linalg.norm(root @ (approximation - target_vector)) / target_norm
        )
        best_error = relative_error
        best_left = left_tensor.copy()
        best_right = right_tensor.copy()
        for _sweep in range(int(max_sweeps)):
            left_design = self._pair_factor_design(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                variable="left",
            )
            left_vector, *_unused = np.linalg.lstsq(
                root @ left_design,
                weighted_target,
                rcond=metric_tol,
            )
            left_tensor = left_vector.reshape(left_tensor.shape)
            right_design = self._pair_factor_design(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                variable="right",
            )
            right_vector, *_unused = np.linalg.lstsq(
                root @ right_design,
                weighted_target,
                rcond=metric_tol,
            )
            right_tensor = right_vector.reshape(right_tensor.shape)
            left_tensor, right_tensor = self._balance_pair_factors(
                left_tensor,
                right_tensor,
            )
            approximation = self._merge_pair_factors(
                site,
                union_sites,
                left_tensor,
                right_tensor,
            ).reshape(-1)
            next_error = float(
                np.linalg.norm(root @ (approximation - target_vector)) / target_norm
            )
            if next_error < best_error:
                best_error = next_error
                best_left = left_tensor.copy()
                best_right = right_tensor.copy()
            if relative_error - next_error <= 32.0 * np.finfo(float).eps:
                relative_error = next_error
                break
            relative_error = next_error
        return best_left, best_right, float(best_error)

    def _variational_pair_factors(
        self,
        site,
        union_sites,
        metric,
        effective,
        left_tensor,
        right_tensor,
        *,
        metric_tol,
        max_sweeps,
        energy_tol,
    ):
        """Alternately minimize the exact pair Rayleigh quotient."""
        left_tensor = np.asarray(left_tensor).copy()
        right_tensor = np.asarray(right_tensor).copy()

        def pair_energy(left, right):
            merged = self._merge_pair_factors(
                site,
                union_sites,
                left,
                right,
            )
            return self._pair_rayleigh(merged.reshape(-1), metric, effective)

        energy = pair_energy(left_tensor, right_tensor)
        best_energy = energy
        best_left = left_tensor.copy()
        best_right = right_tensor.copy()
        accepted_updates = 0
        completed_sweeps = 0
        tolerance = float(energy_tol) * max(1.0, abs(energy))
        for sweep in range(int(max_sweeps)):
            sweep_before = energy
            for variable in ("left", "right"):
                design = self._pair_factor_design(
                    site,
                    union_sites,
                    left_tensor,
                    right_tensor,
                    variable=variable,
                )
                local_metric = design.T.conj() @ metric @ design
                local_effective = design.T.conj() @ effective @ design
                try:
                    _local_energy, vector = _lowest_generalized_eigenpair(
                        local_effective,
                        local_metric,
                        metric_tol=metric_tol,
                    )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    continue
                if variable == "left":
                    proposed_left = vector.reshape(left_tensor.shape)
                    proposed_right = right_tensor
                else:
                    proposed_left = left_tensor
                    proposed_right = vector.reshape(right_tensor.shape)
                proposed_energy = pair_energy(proposed_left, proposed_right)
                # Near-null metric directions can make a nominal generalized
                # eigenvector unreliable.  Never commit such a half-step
                # without an independent Rayleigh check.
                if proposed_energy <= energy + tolerance:
                    left_tensor = proposed_left
                    right_tensor = proposed_right
                    energy = proposed_energy
                    accepted_updates += 1
                    if energy < best_energy:
                        best_energy = energy
                        best_left = left_tensor.copy()
                        best_right = right_tensor.copy()
            left_tensor, right_tensor = self._balance_pair_factors(
                left_tensor,
                right_tensor,
            )
            energy = pair_energy(left_tensor, right_tensor)
            completed_sweeps = sweep + 1
            if sweep_before - energy <= tolerance:
                break
        return (
            best_left,
            best_right,
            float(best_energy),
            completed_sweeps,
            accepted_updates,
        )

    def _pair_tangent_start(
        self,
        site,
        target,
        union_sites,
        metric,
        effective,
        left_tensor,
        right_tensor,
        *,
        metric_tol,
        backtracks=8,
    ):
        r"""Build a simultaneous rank-preserving direction toward ``target``.

        The linearized pair map contains both variations,

        .. math::

            \delta\Theta=\delta A\,B+A\,\delta B.

        Solving its environment-weighted least-squares problem avoids a
        coordinate saddle where neither factor can improve on its own.  The
        trial curve remains inside the original LETTA manifold and includes
        the coordinated quadratic term ``t**2 * delta_left * delta_right``.
        """
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).T.conj())
        values, vectors = np.linalg.eigh(metric)
        threshold = float(metric_tol) * max(
            float(np.linalg.norm(metric, ord=np.inf)),
            np.finfo(float).tiny,
        )
        keep = values > threshold
        if not np.any(keep):
            raise ValueError("the merged-pair metric is numerically singular.")
        root = np.sqrt(values[keep])[:, None] * vectors[:, keep].T.conj()
        left_design = self._pair_factor_design(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variable="left",
        )
        right_design = self._pair_factor_design(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variable="right",
        )
        tangent_design = np.concatenate((left_design, right_design), axis=1)
        merged = self._merge_pair_factors(
            site,
            union_sites,
            left_tensor,
            right_tensor,
        ).reshape(-1)
        residual = np.asarray(target).reshape(-1) - merged
        tangent, *_unused = np.linalg.lstsq(
            root @ tangent_design,
            root @ residual,
            rcond=metric_tol,
        )
        split = left_tensor.size
        delta_left = tangent[:split].reshape(left_tensor.shape)
        delta_right = tangent[split:].reshape(right_tensor.shape)

        best_left = np.asarray(left_tensor).copy()
        best_right = np.asarray(right_tensor).copy()
        best_energy = self._pair_rayleigh(merged, metric, effective)
        best_step = 0.0
        for backtrack in range(int(backtracks)):
            magnitude = 2.0 ** (-backtrack)
            for step in (magnitude, -magnitude):
                proposed_left = left_tensor + step * delta_left
                proposed_right = right_tensor + step * delta_right
                proposed = self._merge_pair_factors(
                    site,
                    union_sites,
                    proposed_left,
                    proposed_right,
                )
                proposed_energy = self._pair_rayleigh(
                    proposed.reshape(-1),
                    metric,
                    effective,
                )
                if proposed_energy < best_energy:
                    best_left = proposed_left
                    best_right = proposed_right
                    best_energy = proposed_energy
                    best_step = step
        best_left, best_right = self._balance_pair_factors(
            best_left,
            best_right,
        )
        return best_left, best_right, float(best_energy), float(best_step)

    def _variational_split_merged_pair(
        self,
        site,
        target,
        union_sites,
        metric,
        effective,
        euclidean_left,
        euclidean_right,
        *,
        metric_tol,
        metric_sweeps,
        variational_sweeps,
        random_starts,
        random_seed,
        energy_tol,
    ):
        """Environment-project and variationally retract a merged pair."""
        old_left = self.tensors[site]
        old_right = self.tensors[site + 1]
        try:
            metric_left, metric_right, projection_error = (
                self._metric_project_pair_factors(
                    site,
                    target,
                    union_sites,
                    metric,
                    euclidean_left,
                    euclidean_right,
                    metric_tol=metric_tol,
                    max_sweeps=metric_sweeps,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            metric_left = euclidean_left.copy()
            metric_right = euclidean_right.copy()
            projection_error = float("inf")

        starts = [
            ("old", old_left.copy(), old_right.copy()),
            ("svd", euclidean_left.copy(), euclidean_right.copy()),
            ("metric", metric_left, metric_right),
        ]

        def pair_energy(left, right):
            candidate = self._merge_pair_factors(
                site,
                union_sites,
                left,
                right,
            )
            return self._pair_rayleigh(
                candidate.reshape(-1),
                metric,
                effective,
            )

        # Relax the best structured proposal.  Running every expensive
        # projected pencil from every seed adds little once the metric fit is
        # already below the incumbent, while the incumbent is always retained
        # as a safe variational fallback.
        finite_starts = [
            (pair_energy(left, right), name, left, right)
            for name, left, right in starts
        ]
        finite_starts = [entry for entry in finite_starts if np.isfinite(entry[0])]
        if not finite_starts:
            finite_starts = [
                (
                    float("inf"),
                    "old",
                    old_left.copy(),
                    old_right.copy(),
                )
            ]
        _seed_energy, seed_name, start_left, start_right = min(
            finite_starts,
            key=lambda entry: entry[0],
        )

        left_scale = max(float(np.linalg.norm(old_left)), 1.0) / np.sqrt(old_left.size)
        right_scale = max(float(np.linalg.norm(old_right)), 1.0) / np.sqrt(
            old_right.size
        )
        best_left = old_left.copy()
        best_right = old_right.copy()
        best_merged = self._merge_pair_factors(
            site,
            union_sites,
            best_left,
            best_right,
        )
        best_energy = self._pair_rayleigh(best_merged.reshape(-1), metric, effective)
        selected_start = "old"
        total_sweeps = 0
        total_accepted = 0
        (
            candidate_left,
            candidate_right,
            candidate_energy,
            sweeps,
            accepted,
        ) = self._variational_pair_factors(
            site,
            union_sites,
            metric,
            effective,
            start_left,
            start_right,
            metric_tol=metric_tol,
            max_sweeps=variational_sweeps,
            energy_tol=energy_tol,
        )
        total_sweeps += sweeps
        total_accepted += accepted
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_left = candidate_left
            best_right = candidate_right
            selected_start = seed_name

        old_energy = pair_energy(old_left, old_right)
        improvement_tolerance = float(energy_tol) * max(1.0, abs(old_energy))
        if best_energy >= old_energy - improvement_tolerance:
            # A simultaneous tangent move is deterministic and can escape a
            # true one-site coordinate saddle.  It is only constructed when
            # the cheaper structured retraction failed to improve.
            try:
                (
                    tangent_left,
                    tangent_right,
                    tangent_energy,
                    tangent_step,
                ) = self._pair_tangent_start(
                    site,
                    target,
                    union_sites,
                    metric,
                    effective,
                    old_left,
                    old_right,
                    metric_tol=metric_tol,
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                tangent_energy = float("inf")
                tangent_step = 0.0
            if tangent_energy < best_energy:
                (
                    candidate_left,
                    candidate_right,
                    candidate_energy,
                    sweeps,
                    accepted,
                ) = self._variational_pair_factors(
                    site,
                    union_sites,
                    metric,
                    effective,
                    tangent_left,
                    tangent_right,
                    metric_tol=metric_tol,
                    max_sweeps=variational_sweeps,
                    energy_tol=energy_tol,
                )
                total_sweeps += sweeps
                total_accepted += accepted
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_left = candidate_left
                    best_right = candidate_right
                    selected_start = f"tangent({tangent_step:g})"

        attempted_random_starts = 0
        if best_energy >= old_energy - improvement_tolerance and int(random_starts) > 0:
            rng = np.random.default_rng(random_seed)
            for trial in range(int(random_starts)):
                attempted_random_starts += 1
                random_left = left_scale * self._random_matrix(
                    old_left.shape,
                    old_left.dtype,
                    rng,
                )
                random_right = right_scale * self._random_matrix(
                    old_right.shape,
                    old_right.dtype,
                    rng,
                )
                (
                    candidate_left,
                    candidate_right,
                    candidate_energy,
                    sweeps,
                    accepted,
                ) = self._variational_pair_factors(
                    site,
                    union_sites,
                    metric,
                    effective,
                    random_left,
                    random_right,
                    metric_tol=metric_tol,
                    max_sweeps=variational_sweeps,
                    energy_tol=energy_tol,
                )
                total_sweeps += sweeps
                total_accepted += accepted
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_left = candidate_left
                    best_right = candidate_right
                    selected_start = f"random[{trial}]"
        return (
            best_left,
            best_right,
            float(projection_error),
            total_sweeps,
            total_accepted,
            selected_start,
            attempted_random_starts,
        )

    def optimize_two_sites(
        self,
        site,
        *,
        solver="whitened",
        split_strategy="variational",
        split_metric_tol=1.0e-12,
        split_metric_sweeps=6,
        split_variational_sweeps=8,
        split_random_starts=0,
        split_random_seed=0,
        split_energy_tol=1.0e-12,
        **solver_options,
    ):
        r"""Optimize an adjacent pair and truncate it back to the current rank.

        The pair is first represented by one tensor over the union of its tied
        physical labels.  That tensor receives the configured local solve
        (exact for the compressed and identity-block frontier backends).
        By default the conditional SVD is only an initialization.  The merged
        tensor is projected with its exact environment metric, after which the
        two graph factors are optimized against the exact pair Rayleigh
        quotient.  Every factor half-step and the final global state are
        guarded variationally.  ``split_strategy="svd"`` retains the diagnostic
        Euclidean truncation path.
        """
        if type(self) is not FrontierTiedLETTA:
            raise TypeError(
                "two-site merge/split currently supports exact unrestricted "
                "FrontierTiedLETTA states; symmetry sectors need a "
                "charge-resolved split."
            )
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        if "environment" in solver_options or "energy_before" in solver_options:
            raise TypeError(
                "environment and energy_before are built for the merged pair."
            )
        split_strategy = str(split_strategy).lower().replace("-", "_")
        if split_strategy in {"environment", "metric", "als"}:
            split_strategy = "variational"
        if split_strategy not in {"variational", "svd"}:
            raise ValueError("split_strategy must be 'variational' or 'svd'.")
        split_metric_tol = float(split_metric_tol)
        split_energy_tol = float(split_energy_tol)
        split_metric_sweeps = int(split_metric_sweeps)
        split_variational_sweeps = int(split_variational_sweeps)
        split_random_starts = int(split_random_starts)
        if split_random_seed is not None:
            split_random_seed = int(split_random_seed)
        if not np.isfinite(split_metric_tol) or split_metric_tol <= 0.0:
            raise ValueError("split_metric_tol must be finite and positive.")
        if not np.isfinite(split_energy_tol) or split_energy_tol < 0.0:
            raise ValueError("split_energy_tol must be finite and nonnegative.")
        if split_metric_sweeps < 0 or split_variational_sweeps < 0:
            raise ValueError("split sweep counts must be nonnegative.")
        if split_random_starts < 0:
            raise ValueError("split_random_starts must be nonnegative.")
        following = site + 1
        energy_before = float(self.expectation())
        merged, union_sites = self._merged_pair_tensor(site)
        right_dimension = self._bond_dims()[following + 1]
        identity = np.eye(right_dimension, dtype=merged.dtype)
        right_sites = self.physical_sites[following]
        identity_tensor = np.broadcast_to(
            identity.reshape(
                right_dimension,
                right_dimension,
                *((1,) * len(right_sites)),
            ),
            (
                right_dimension,
                right_dimension,
                *(self.dims[index] for index in right_sites),
            ),
        ).copy()
        temporary_tensors = [tensor.copy() for tensor in self.tensors]
        temporary_tensors[site] = merged
        temporary_tensors[following] = identity_tensor
        temporary_parents = list(self.parent_sets)
        temporary_parents[site] = tuple(index for index in union_sites if index != site)
        temporary_bonds = list(self._bond_dims())
        temporary_bonds[following] = right_dimension
        temporary = FrontierTiedLETTA(
            self.hamiltonian,
            self.dims,
            tuple(temporary_parents),
            bond_dims=tuple(temporary_bonds),
            tensors=temporary_tensors,
            frontier_backend=self.frontier_backend,
            path_optimizer=self.path_optimizer,
            tt_max_rank=self.tt_options["max_rank"],
            tt_rtol=self.tt_options["rtol"],
            tt_atol=self.tt_options["atol"],
            tt_transfer_max_rank=self.tt_options["transfer_max_rank"],
            tt_transfer_rtol=self.tt_options["transfer_rtol"],
            tt_transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
        )
        temporary.tensors = [tensor.copy() for tensor in temporary_tensors]
        temporary.energy = temporary.expectation()
        pair_metric = None
        pair_effective = None
        if split_strategy == "variational":
            pair_metric, pair_effective = temporary.local_operators(site)
        local_update = temporary.optimize_site(
            site,
            solver=solver,
            energy_before=temporary.energy,
            **solver_options,
        )
        merged_energy = float(temporary.expectation())
        split_target = temporary.tensors[site]
        if split_strategy == "variational":
            split_target = split_target.copy()
            old_vector = merged.reshape(-1)
            target_vector = split_target.reshape(-1)
            old_norm = float(np.real(np.vdot(old_vector, pair_metric @ old_vector)))
            target_norm = float(
                np.real(np.vdot(target_vector, pair_metric @ target_vector))
            )
            if old_norm > np.finfo(float).tiny and target_norm > np.finfo(float).tiny:
                split_target *= np.sqrt(old_norm / target_norm)
                target_vector = split_target.reshape(-1)
            overlap_value = np.vdot(target_vector, pair_metric @ old_vector)
            if abs(overlap_value) > 256.0 * np.finfo(float).eps:
                split_target = split_target * (overlap_value / abs(overlap_value))
        (
            left_tensor,
            right_tensor,
            overlap,
            conditional_ranks,
            truncation_error,
        ) = self._split_merged_pair_tensor(
            site,
            split_target,
            union_sites,
        )
        projection_error = float("nan")
        factor_sweeps = 0
        factor_accepted_updates = 0
        if split_strategy == "variational":
            (
                left_tensor,
                right_tensor,
                projection_error,
                factor_sweeps,
                factor_accepted_updates,
                selected_start,
                attempted_random_starts,
            ) = self._variational_split_merged_pair(
                site,
                split_target,
                union_sites,
                pair_metric,
                pair_effective,
                left_tensor,
                right_tensor,
                metric_tol=split_metric_tol,
                metric_sweeps=split_metric_sweeps,
                variational_sweeps=split_variational_sweeps,
                random_starts=split_random_starts,
                random_seed=(
                    None
                    if split_random_seed is None
                    else split_random_seed + 104729 * site
                ),
                energy_tol=split_energy_tol,
            )
        else:
            selected_start = "svd"
            attempted_random_starts = 0
        old_left = self.tensors[site]
        old_right = self.tensors[following]
        self.tensors[site] = left_tensor
        self.tensors[following] = right_tensor
        try:
            energy_after = float(self.expectation())
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            energy_after = float("inf")
        tolerance = 512.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
        attempted_energy = float(energy_after)
        # The fresh contraction of the actual factorized state is the final
        # authority.  A merged solve can be rejected on its larger manifold
        # while the guarded rank-preserving factor relaxation still finds a
        # valid lower-energy update.
        accepted = bool(
            np.isfinite(energy_after) and energy_after <= energy_before + tolerance
        )
        if not accepted:
            self.tensors[site] = old_left
            self.tensors[following] = old_right
            energy_after = energy_before
        else:
            self.history = []
            self.converged = False
        self.energy = float(energy_after)
        return FrontierTwoSiteUpdate(
            sites=(site, following),
            overlap_sites=overlap,
            raw_merged_dim=int(temporary.tensors[site].size),
            old_bond_dimension=self._bond_dims()[following],
            temporary_bond_dimension=right_dimension,
            conditional_ranks=conditional_ranks,
            relative_truncation_error=truncation_error,
            energy_before=energy_before,
            merged_energy=merged_energy,
            attempted_energy=attempted_energy,
            energy=float(energy_after),
            accepted=accepted,
            local_update=local_update,
            split_strategy=split_strategy,
            selected_start=selected_start,
            metric_projection_error=projection_error,
            factor_sweeps=factor_sweeps,
            factor_accepted_updates=factor_accepted_updates,
            factor_random_starts=attempted_random_starts,
        )

    @property
    def nparameters(self) -> int:
        return int(sum(tensor.size for tensor in self.tensors))

    @property
    def contraction_plans(self) -> int:
        return self._norm_frontier.plan_count + self._hamiltonian_frontier.plan_count

    @property
    def peak_frontier_elements(self) -> int:
        """Largest dense-equivalent cut message in either network."""
        return max(
            self._dense_peak_elements(self._norm_frontier),
            self._dense_peak_elements(self._hamiltonian_frontier),
        )

    @property
    def norm_peak_frontier_elements(self) -> int:
        return self._dense_peak_elements(self._norm_frontier)

    @property
    def hamiltonian_peak_frontier_elements(self) -> int:
        return self._dense_peak_elements(self._hamiltonian_frontier)

    @staticmethod
    def _dense_peak_elements(engine) -> int:
        if isinstance(engine, TTMPOFrontier):
            return int(engine.dense_peak_message_elements)
        return int(engine.peak_message_elements)

    @staticmethod
    def _dense_message_elements(engine, cut) -> int:
        if isinstance(engine, TTMPOFrontier):
            return int(engine.dense_message_elements(cut))
        return int(engine.message_elements(cut))

    @property
    def cached_environment_elements(self) -> int:
        """Dense-equivalent elements in one fixed-side environment cache."""
        return sum(
            self._dense_message_elements(engine, cut)
            for engine in (self._norm_frontier, self._hamiltonian_frontier)
            for cut in range(len(self.dims) + 1)
        )

    @property
    def contraction_is_exact(self) -> bool:
        """Whether the configured contractor performs no TT truncation."""

        if self.frontier_backend != "tensor_train":
            return True
        return (
            self.tt_options["max_rank"] is None
            and self.tt_options["transfer_max_rank"] is None
            and float(self.tt_options["rtol"]) == 0.0
            and float(self.tt_options["atol"]) == 0.0
            and float(self.tt_options["transfer_rtol"]) == 0.0
            and float(self.tt_options["transfer_atol"]) == 0.0
        )

    @property
    def norm_contraction_is_exact(self) -> bool:
        """Whether the norm contractor performs no TT truncation."""

        if not isinstance(self._norm_frontier, TTMPOFrontier):
            return True
        return self.contraction_is_exact

    @property
    def hamiltonian_contraction_is_exact(self) -> bool:
        """Whether the Hamiltonian contractor performs no TT truncation."""

        return self.contraction_is_exact

    @property
    def hamiltonian_action_is_hermitian(self) -> bool:
        """Whether local Hamiltonian actions are explicitly Hermitian."""

        return self.hamiltonian_contraction_is_exact or self.tt_hermitize

    @property
    def tt_diagnostics(self):
        """Latest norm and Hamiltonian TT diagnostics, or ``None``."""

        if self.frontier_backend != "tensor_train":
            return None
        return {
            "norm": (
                self._norm_frontier.diagnostics
                if isinstance(self._norm_frontier, TTMPOFrontier)
                else None
            ),
            "hamiltonian": self._hamiltonian_frontier.diagnostics,
        }

    @property
    def peak_compressed_frontier_elements(self) -> int:
        """Peak TT message storage observed in the latest contractions."""

        if self.frontier_backend != "tensor_train":
            return self.peak_frontier_elements
        norm_storage = (
            self._norm_frontier.diagnostics.peak_message_storage_elements
            if isinstance(self._norm_frontier, TTMPOFrontier)
            else self._norm_frontier.peak_message_elements
        )
        return max(
            norm_storage,
            self._hamiltonian_frontier.diagnostics.peak_message_storage_elements,
        )

    def norm(self) -> float:
        value = self._norm_frontier.scalar(self.tensors)
        return float(np.real(value))

    @staticmethod
    def _completed_frontier_scalar(frontier, message, cut):
        """Validate and extract an endpoint directional-sweep message."""
        cut = int(cut)
        if cut not in {0, frontier.nsites}:
            raise ValueError("a scalar can only be extracted at a boundary cut.")
        if isinstance(frontier, BlockMPOFrontier):
            return frontier.boundary_scalar(message, cut)
        if isinstance(frontier, TermwiseTTMPOFrontier):
            return frontier.boundary_scalar(message, cut)
        if isinstance(frontier, TTMPOFrontier):
            if not isinstance(message, TTFrontier):
                raise TypeError("tensor-train frontier requires a TT message.")
            if message.labels != frontier.message_labels(
                cut
            ) or message.shape != frontier.message_shape(cut):
                raise ValueError(f"message at cut {cut} has the wrong shape.")
            return message.to_dense().reshape(()).item()
        if not isinstance(frontier, MPOFrontier):
            raise TypeError("frontier has an unsupported message representation.")
        value = np.asarray(message)
        if value.shape != frontier.message_shape(cut):
            raise ValueError(f"message at cut {cut} has the wrong shape.")
        return value.reshape(()).item()

    def normalize(self):
        norm_squared = self.norm()
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ValueError("frontier-tied LETTA state cannot be normalized.")
        norm = np.sqrt(norm_squared)
        self.tensors[0] /= norm
        return self

    def balance_gauges(self, *, state_norm=None):
        tensor_norms = np.asarray(
            [float(np.linalg.norm(tensor)) for tensor in self.tensors]
        )
        if np.any(~np.isfinite(tensor_norms)) or np.any(tensor_norms <= 0.0):
            bad_site = int(
                np.flatnonzero((~np.isfinite(tensor_norms)) | (tensor_norms <= 0.0))[0]
            )
            raise ValueError(f"tensor {bad_site} is zero or nonfinite.")
        if state_norm is None:
            norm = self.norm()
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("frontier-tied LETTA state cannot be normalized.")
            state_norm = np.sqrt(norm)
        state_norm = float(state_norm)
        if not np.isfinite(state_norm) or state_norm <= 0.0:
            raise ValueError("frontier-tied LETTA state cannot be normalized.")
        common_scale = float(
            np.exp(
                (np.sum(np.log(tensor_norms)) - np.log(state_norm)) / len(self.tensors)
            )
        )
        for site, (tensor, tensor_norm) in enumerate(zip(self.tensors, tensor_norms)):
            self.tensors[site] = tensor * (common_scale / tensor_norm)
        return self

    @staticmethod
    def _hermitian_part(matrix):
        matrix = np.asarray(matrix)
        return 0.5 * (matrix + matrix.conj().T)

    def frontier_bond_grams(
        self,
        cut,
        *,
        left_messages=None,
        right_messages=None,
        weighting="uniform",
    ):
        r"""Return graph-aware left/right virtual Grams at one chain cut.

        The exact norm-frontier message at ``cut`` retains the virtual bra and
        ket indices together with every tied physical variable crossing the
        cut.  Summing the physical frontier gives the virtual marginal of the
        corresponding boundary Gram.  The right message is transposed so a
        virtual gauge ``G`` acts as

        .. math::

            L \mapsto G^\dagger L G,
            \qquad
            R \mapsto G^{-1} R G^{-\dagger}.
        """
        if isinstance(self._norm_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "frontier-bond Grams currently require dense exact messages; "
                "use virtual canonicalization with the tensor-train backend."
            )
        cut = int(cut)
        if cut <= 0 or cut >= len(self.dims):
            raise ValueError("cut must be an internal virtual bond.")
        if left_messages is None:
            left_messages = self._norm_frontier.build_left(self.tensors)
        if right_messages is None:
            right_messages = self._norm_frontier.build_right(self.tensors)
        if len(left_messages) != len(self.dims) + 1:
            raise ValueError("left_messages has the wrong length.")
        if len(right_messages) != len(self.dims) + 1:
            raise ValueError("right_messages has the wrong length.")
        left_message = np.asarray(left_messages[cut])
        right_message = np.asarray(right_messages[cut])
        expected = self._norm_frontier.message_shape(cut)
        if left_message.shape != expected or right_message.shape != expected:
            raise ValueError("frontier message has the wrong shape.")
        weighting = str(weighting).lower().replace("-", "_")
        if weighting not in {"probability", "uniform"}:
            raise ValueError("weighting must be 'probability' or 'uniform'.")
        dimension = left_message.shape[0]
        conditional_left = left_message.reshape(dimension, dimension, -1)
        conditional_right = right_message.reshape(dimension, dimension, -1)
        if weighting == "probability":
            weights = np.einsum(
                "abf,abf->f",
                conditional_left,
                conditional_right,
                optimize=True,
            )
            scale = max(float(np.max(np.abs(weights.real), initial=0.0)), 1.0)
            if float(np.max(np.abs(weights.imag), initial=0.0)) > 1.0e-10 * scale:
                raise ValueError("frontier probabilities are not numerically real.")
            weights = weights.real
            if (
                float(np.min(weights, initial=0.0))
                < -1.0e3 * np.finfo(float).eps * scale
            ):
                raise ValueError("frontier probabilities are numerically negative.")
            weights = np.maximum(weights, 0.0)
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= np.finfo(float).tiny:
                raise ValueError("frontier probability is numerically zero.")
            weights = weights / total
        else:
            weights = np.ones(conditional_left.shape[-1])
        left = np.einsum(
            "f,abf->ab",
            weights,
            conditional_left,
            optimize=True,
        )
        right = np.einsum(
            "f,abf->ab",
            weights,
            conditional_right,
            optimize=True,
        ).T
        return self._hermitian_part(left), self._hermitian_part(right)

    def canonicalize_frontier_gauge(
        self,
        *,
        metric_tol: float = 1.0e-12,
        max_condition: float = 1.0e8,
        weighting="uniform",
    ):
        r"""Balance every full-rank virtual bond using exact frontier Grams.

        For virtual marginals ``L`` and ``R``, this constructs ``G`` such that

        .. math::

            G^\dagger L G
            = G^{-1} R G^{-\dagger}
            = \Lambda.

        The physical frontier is reduced only after its exact norm message has
        been built, so the gauge is graph aware without materializing the
        many-body state.  Rank-deficient bonds are conservatively left
        unchanged because a support-restricted construction is nonunique and
        can remove or badly scale variational bond directions.
        """
        if isinstance(self._norm_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "frontier canonicalization currently requires dense exact "
                "messages; use virtual canonicalization instead."
            )
        metric_tol = float(metric_tol)
        max_condition = float(max_condition)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(max_condition) or max_condition < 1.0:
            raise ValueError("max_condition must be finite and at least one.")

        left_messages = self._norm_frontier.build_left(self.tensors)
        right_messages = self._norm_frontier.build_right(self.tensors)
        updates = []
        eps = np.finfo(float).eps
        tiny = np.finfo(float).tiny
        for cut in range(1, len(self.dims)):
            dimension = self._bond_dims()[cut]
            left, right = self.frontier_bond_grams(
                cut,
                left_messages=left_messages,
                right_messages=right_messages,
                weighting=weighting,
            )
            left_values, left_vectors = np.linalg.eigh(left)
            right_values = np.linalg.eigvalsh(right)
            left_scale = max(float(np.max(left_values, initial=0.0)), tiny)
            right_scale = max(float(np.max(right_values, initial=0.0)), tiny)
            relative_floor = max(metric_tol, 128.0 * eps)
            left_threshold = relative_floor * left_scale
            right_threshold = relative_floor * right_scale
            left_rank = int(np.count_nonzero(left_values > left_threshold))
            right_rank = int(np.count_nonzero(right_values > right_threshold))
            left_condition = (
                float(left_values[-1] / left_values[0])
                if left_rank == dimension
                else float("inf")
            )
            right_condition = (
                float(right_values[-1] / right_values[0])
                if right_rank == dimension
                else float("inf")
            )
            denominator = max(
                float(np.linalg.norm(left) + np.linalg.norm(right)),
                tiny,
            )
            imbalance_before = float(2.0 * np.linalg.norm(left - right) / denominator)
            common = {
                "cut": cut,
                "frontier_sites": tuple(self._norm_frontier.frontier_sites[cut]),
                "left_rank": left_rank,
                "right_rank": right_rank,
                "left_condition": left_condition,
                "right_condition": right_condition,
                "imbalance_before": imbalance_before,
            }
            if left_rank < dimension or right_rank < dimension:
                updates.append(
                    FrontierGaugeUpdate(
                        applied=False,
                        message="rank-deficient frontier marginal",
                        balanced_condition=float("inf"),
                        gauge_condition=1.0,
                        imbalance_after=imbalance_before,
                        **common,
                    )
                )
                continue

            left_trace_scale = float(np.trace(left).real / dimension)
            right_trace_scale = float(np.trace(right).real / dimension)
            if left_trace_scale <= tiny or right_trace_scale <= tiny:
                updates.append(
                    FrontierGaugeUpdate(
                        applied=False,
                        message="frontier marginal has zero trace",
                        balanced_condition=float("inf"),
                        gauge_condition=1.0,
                        imbalance_after=imbalance_before,
                        **common,
                    )
                )
                continue

            normalized_left_values = left_values / left_trace_scale
            left_half = (
                left_vectors * np.sqrt(normalized_left_values)
            ) @ left_vectors.conj().T
            left_inverse_half = (
                left_vectors * (1.0 / np.sqrt(normalized_left_values))
            ) @ left_vectors.conj().T
            normalized_right = right / right_trace_scale
            center = self._hermitian_part(left_half @ normalized_right @ left_half)
            center_values, center_vectors = np.linalg.eigh(center)
            center_scale = max(float(np.max(center_values, initial=0.0)), tiny)
            center_rank = int(
                np.count_nonzero(center_values > relative_floor * center_scale)
            )
            if center_rank < dimension:
                updates.append(
                    FrontierGaugeUpdate(
                        applied=False,
                        message="balanced frontier product is rank deficient",
                        balanced_condition=float("inf"),
                        gauge_condition=1.0,
                        imbalance_after=imbalance_before,
                        **common,
                    )
                )
                continue
            center_fourth_root = (
                center_vectors * center_values**0.25
            ) @ center_vectors.conj().T
            gauge = (
                (right_trace_scale / left_trace_scale) ** 0.25
                * left_inverse_half
                @ center_fourth_root
            )
            gauge = np.real_if_close(gauge)
            gauge_condition = float(np.linalg.cond(gauge))
            if not np.isfinite(gauge_condition) or gauge_condition > max_condition:
                updates.append(
                    FrontierGaugeUpdate(
                        applied=False,
                        message="frontier gauge is too ill conditioned",
                        balanced_condition=float("inf"),
                        gauge_condition=gauge_condition,
                        imbalance_after=imbalance_before,
                        **common,
                    )
                )
                continue

            left_tensor = self.tensors[cut - 1]
            transformed_left = np.tensordot(
                left_tensor,
                gauge,
                axes=(1, 0),
            )
            self.tensors[cut - 1] = np.moveaxis(transformed_left, -1, 1)
            right_tensor = self.tensors[cut]
            transformed_right = np.linalg.solve(
                gauge,
                right_tensor.reshape(dimension, -1),
            )
            self.tensors[cut] = transformed_right.reshape(right_tensor.shape)

            inverse = np.linalg.solve(gauge, np.eye(dimension, dtype=gauge.dtype))
            balanced_left = self._hermitian_part(gauge.conj().T @ left @ gauge)
            balanced_right = self._hermitian_part(inverse @ right @ inverse.conj().T)
            balanced_values = np.linalg.eigvalsh(balanced_left)
            balanced_condition = float(balanced_values[-1] / balanced_values[0])
            balanced_denominator = max(
                float(np.linalg.norm(balanced_left) + np.linalg.norm(balanced_right)),
                tiny,
            )
            imbalance_after = float(
                2.0
                * np.linalg.norm(balanced_left - balanced_right)
                / balanced_denominator
            )
            updates.append(
                FrontierGaugeUpdate(
                    applied=True,
                    message="balanced",
                    balanced_condition=balanced_condition,
                    gauge_condition=gauge_condition,
                    imbalance_after=imbalance_after,
                    **common,
                )
            )
        return tuple(updates)

    def canonicalize_virtual(self, direction):
        """Move the virtual-chain gauge exactly while preserving tied legs."""
        direction = str(direction).lower().replace("_", "-")
        if direction in {"left", "left-canonical", "lr"}:
            for site in range(len(self.tensors) - 1):
                tensor = self.tensors[site]
                physical_axes = tuple(range(2, tensor.ndim))
                ordered = tensor.transpose((0, *physical_axes, 1))
                matrix = ordered.reshape(-1, tensor.shape[1])
                left, transfer = np.linalg.qr(matrix, mode="reduced")
                rank = left.shape[1]
                padded_left = np.zeros(
                    (matrix.shape[0], tensor.shape[1]),
                    dtype=np.result_type(left.dtype, tensor.dtype),
                )
                padded_transfer = np.zeros(
                    (tensor.shape[1], tensor.shape[1]),
                    dtype=np.result_type(transfer.dtype, tensor.dtype),
                )
                padded_left[:, :rank] = left
                padded_transfer[:rank] = transfer
                ordered = padded_left.reshape(ordered.shape)
                inverse = np.argsort((0, *physical_axes, 1))
                self.tensors[site] = ordered.transpose(inverse)
                self.tensors[site + 1] = np.tensordot(
                    padded_transfer,
                    self.tensors[site + 1],
                    axes=(1, 0),
                )
        elif direction in {"right", "right-canonical", "rl"}:
            for site in range(len(self.tensors) - 1, 0, -1):
                tensor = self.tensors[site]
                physical_axes = tuple(range(2, tensor.ndim))
                ordered = tensor.transpose((0, *physical_axes, 1))
                matrix = ordered.reshape(tensor.shape[0], -1)
                right, transfer = np.linalg.qr(matrix.T, mode="reduced")
                rank = right.shape[1]
                padded_right = np.zeros(
                    matrix.shape,
                    dtype=np.result_type(right.dtype, tensor.dtype),
                )
                padded_transfer = np.zeros(
                    (tensor.shape[0], tensor.shape[0]),
                    dtype=np.result_type(transfer.dtype, tensor.dtype),
                )
                padded_right[:rank] = right.T
                padded_transfer[:, :rank] = transfer.T
                ordered = padded_right.reshape(ordered.shape)
                inverse = np.argsort((0, *physical_axes, 1))
                self.tensors[site] = ordered.transpose(inverse)
                previous = np.tensordot(
                    self.tensors[site - 1],
                    padded_transfer,
                    axes=(1, 0),
                )
                self.tensors[site - 1] = np.moveaxis(previous, -1, 1)
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        return self

    def expectation(self) -> float:
        norm = self._norm_frontier.scalar(self.tensors)
        numerator = self._hamiltonian_frontier.scalar(self.tensors)
        if abs(norm) <= np.finfo(float).tiny:
            raise ValueError("frontier-tied LETTA state is numerically zero.")
        return float(np.real(numerator / norm))

    def site_environment(self, site: int) -> FrontierSiteEnvironment:
        """Build the four numerical cut messages surrounding ``site``.

        The returned environment is valid until a tensor strictly to the left
        or right of the hole is changed.  Directional sweeps update only the
        moving-side message and reuse the fixed-side messages.  This one-off
        helper builds complete left and right lists for both networks; use
        :meth:`run` for the lower-memory directional cache workflow.
        """
        site = self._validated_site(site)
        norm_left = self._norm_frontier.build_left(self.tensors)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_left = self._hamiltonian_frontier.build_left(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
        return FrontierSiteEnvironment(
            site=site,
            norm_left=norm_left[site],
            norm_right=norm_right[site + 1],
            hamiltonian_left=hamiltonian_left[site],
            hamiltonian_right=hamiltonian_right[site + 1],
        )

    def _resolved_environment(self, site, environment):
        if environment is None:
            return self.site_environment(site)
        if not isinstance(environment, FrontierSiteEnvironment):
            raise TypeError("environment must be a FrontierSiteEnvironment.")
        if environment.site != site:
            raise ValueError("environment belongs to a different site.")
        return environment

    def local_metric(self, site: int, *, environment=None) -> np.ndarray:
        if isinstance(self._norm_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "the tensor-train backend exposes matrix-free local actions, "
                "not a dense local metric."
            )
        site = self._validated_site(site)
        environment = self._resolved_environment(site, environment)
        metric = self._norm_frontier.hole_matrix(
            site,
            environment.norm_left,
            environment.norm_right,
        )
        return 0.5 * (metric + metric.T.conj())

    def local_operators(
        self,
        site: int,
        *,
        environment=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``(N_eff, H_eff)`` without a global projector."""
        if isinstance(self._hamiltonian_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "the tensor-train backend exposes matrix-free local actions, "
                "not dense local operators."
            )
        site = self._validated_site(site)
        environment = self._resolved_environment(site, environment)
        metric = self._norm_frontier.hole_matrix(
            site,
            environment.norm_left,
            environment.norm_right,
        )
        effective = self._hamiltonian_frontier.hole_matrix(
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
        )
        metric = 0.5 * (metric + metric.T.conj())
        effective = 0.5 * (effective + effective.T.conj())
        return metric, effective

    @staticmethod
    def _whiten_local_operators(metric, effective, *, metric_tol):
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).T.conj())
        effective = 0.5 * (np.asarray(effective) + np.asarray(effective).T.conj())
        basis = _metric_basis(metric, metric_tol=metric_tol)
        identity_metric = basis.T.conj() @ metric @ basis
        for _ in range(2):
            identity_metric = 0.5 * (identity_metric + identity_metric.T.conj())
            values, vectors = np.linalg.eigh(identity_metric)
            if values.size == 0 or float(values[0]) <= np.finfo(float).tiny:
                break
            correction = (vectors / np.sqrt(values)[None, :]) @ vectors.T.conj()
            basis = basis @ correction
            identity_metric = basis.T.conj() @ metric @ basis
        identity = np.eye(basis.shape[1], dtype=identity_metric.dtype)
        identity_error = float(np.linalg.norm(identity_metric - identity, ord=np.inf))
        hamiltonian = basis.T.conj() @ effective @ basis
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T.conj())
        return (
            basis,
            hamiltonian,
            {
                "raw_dim": int(metric.shape[0]),
                "metric_rank": int(basis.shape[1]),
                "identity_metric_error": identity_error,
            },
        )

    def local_whitened_problem(
        self,
        site: int,
        *,
        environment=None,
        metric_tol: float = 1.0e-12,
    ):
        r"""Return an exact moving local frame with identity overlap metric.

        The columns of ``basis`` span the nonsingular support of the native
        local metric ``S`` and obey

        .. math::

            B^\dagger S B = I.

        ``hamiltonian`` is ``B^\dagger H B``.  This is a local coordinate
        frame, not a persistent tensor gauge: for a general tied graph, ``B``
        can couple the left and right virtual indices and cannot be absorbed
        into neighboring tensors without changing the ansatz.
        """
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        metric, effective = self.local_operators(site, environment=environment)
        return self._whiten_local_operators(
            metric,
            effective,
            metric_tol=metric_tol,
        )

    def local_block_problem(
        self,
        site: int,
        *,
        environment=None,
    ) -> PhysicalBlockGeneralizedProblem:
        r"""Build the exact physical-slice block-sparse local pencil.

        For the tensor at site ``k``, let

        ``p = (s_k, s_{P_k})`` and ``mu = (a_{k-1}, a_k)``.

        The norm obeys ``N[(mu,p),(nu,q)] = delta[p,q] N_p[mu,nu]``.
        Hamiltonian blocks are contracted only for pairs ``(p,q)`` allowed by
        the supports and nonzero entries of :class:`LocalTerm` objects.  The
        fixed-physical contractions return only virtual matrices, so neither
        full local operator is materialized.
        """
        if self.frontier_backend == "tensor_train":
            raise NotImplementedError(
                "physical-slice block construction currently requires an exact "
                "dense or identity-block frontier; use solver='matrix_free' "
                "for tensor-train frontiers."
            )
        site = self._validated_site(site)
        environment = self._resolved_environment(site, environment)
        layout = PhysicalBlockLayout(self.tensors[site].shape)
        pairs = self._hamiltonian_physical_blocks(site)

        def metric_factory(row, column):
            return self._norm_frontier.hole_block(
                site,
                environment.norm_left,
                environment.norm_right,
                layout.configurations[row],
                layout.configurations[column],
            )

        def hamiltonian_factory(row, column):
            return self._hamiltonian_frontier.hole_block(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                layout.configurations[row],
                layout.configurations[column],
            )

        return PhysicalBlockGeneralizedProblem.from_block_factories(
            self.tensors[site].shape,
            pairs,
            metric_factory,
            hamiltonian_factory,
            dtype=np.result_type(
                self.tensors[site].dtype,
                self.hamiltonian.dtype,
            ),
        )

    def metric_action(self, site: int, vector, *, environment=None) -> np.ndarray:
        """Apply ``N_eff`` directly, without forming ``N_eff`` or ``P_k``."""
        site = self._validated_site(site)
        vector = self._validated_local_vector(site, vector)
        environment = self._resolved_environment(site, environment)
        return self._norm_frontier.hole_action(
            site,
            environment.norm_left,
            environment.norm_right,
            vector,
        )

    def hamiltonian_action(
        self,
        site: int,
        vector,
        *,
        environment=None,
    ) -> np.ndarray:
        """Apply ``H_eff`` from cached MPO frontier messages."""
        site = self._validated_site(site)
        vector = self._validated_local_vector(site, vector)
        environment = self._resolved_environment(site, environment)
        forward = self._hamiltonian_frontier.hole_action(
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
            vector,
        )
        if (
            isinstance(self._hamiltonian_frontier, TTMPOFrontier)
            and self.tt_hermitize
            and not self.hamiltonian_contraction_is_exact
        ):
            adjoint = self._hamiltonian_frontier.hole_adjoint_action(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                vector,
            )
            return 0.5 * (forward + adjoint)
        return forward

    def _validated_site(self, site):
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        return site

    def _validated_local_vector(self, site, vector):
        vector = np.asarray(vector)
        size = self.tensors[site].size
        if vector.size != size:
            raise ValueError(f"local vector must contain {size} entries.")
        return vector.reshape(-1)

    def _hamiltonian_physical_blocks(self, site):
        site = self._validated_site(site)
        cached = self._physical_block_connectivity_cache.get(site)
        if cached is None:
            cached = hamiltonian_physical_connectivity(
                self.hamiltonian,
                self.physical_sites[site],
            )
            self._physical_block_connectivity_cache[site] = cached
        return cached

    def optimize_site(
        self,
        site: int,
        *,
        metric_tol: float = 1.0e-12,
        solver="auto",
        matrix_free_threshold: int = 256,
        block_sparse_max_elements: int | None = 4_000_000,
        eig_tol: float = 1.0e-10,
        maxiter: int | None = None,
        max_subspace: int = 32,
        energy_before: float | None = None,
        environment=None,
    ) -> FrontierSiteUpdate:
        """Minimize one tensor with a dense, whitened, action-only, or block solver.

        ``auto`` retains the dense solver below ``matrix_free_threshold``.
        Above it, a structurally sparse physical-block pencil is used only
        when its estimated raw-plus-whitened block work arrays fit within
        ``block_sparse_max_elements``; otherwise the action-only solver is
        selected.  An explicit ``solver='block_sparse'`` is not capped.
        """
        site = self._validated_site(site)
        solver = str(solver).lower().replace("-", "_")
        if solver in {"block", "physical_block", "physical_blocks"}:
            solver = "block_sparse"
        if solver in {
            "canonical",
            "identity_metric",
            "local_canonical",
            "s_identity",
        }:
            solver = "whitened"
        if solver not in {
            "auto",
            "direct",
            "whitened",
            "matrix_free",
            "block_sparse",
        }:
            raise ValueError(
                "solver must be 'auto', 'direct', 'whitened', "
                "'matrix_free', or 'block_sparse'."
            )
        if not self.norm_contraction_is_exact:
            raise ValueError(
                "variational optimization requires an exact norm contraction; "
                "all-TT truncated norms are available only for scalar diagnostics."
            )
        if not self.hamiltonian_action_is_hermitian:
            raise ValueError(
                "variational optimization requires an exact or explicitly "
                "Hermitized Hamiltonian action; set tt_hermitize=True."
            )
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if block_sparse_max_elements is not None:
            block_sparse_max_elements = int(block_sparse_max_elements)
            if block_sparse_max_elements < 1:
                raise ValueError("block_sparse_max_elements must be positive or None.")
        old_tensor = self.tensors[site].copy()
        if energy_before is None:
            energy_before = self.expectation()
        energy_before = float(energy_before)
        environment = self._resolved_environment(site, environment)

        selected_solver = solver
        if selected_solver == "auto":
            if self.frontier_backend == "tensor_train":
                selected_solver = "matrix_free"
            else:
                if old_tensor.size < int(matrix_free_threshold):
                    selected_solver = "direct"
                else:
                    physical_blocks = int(np.prod(old_tensor.shape[2:]))
                    allowed_blocks = len(self._hamiltonian_physical_blocks(site))
                    block_storage = physical_blocks + allowed_blocks
                    dense_storage = 2 * physical_blocks**2
                    virtual_size = int(np.prod(old_tensor.shape[:2]))
                    estimated_block_work_elements = 2 * block_storage * virtual_size**2
                    selected_solver = (
                        "block_sparse"
                        if (
                            allowed_blocks < physical_blocks**2
                            and block_storage < dense_storage
                            and (
                                block_sparse_max_elements is None
                                or estimated_block_work_elements
                                <= block_sparse_max_elements
                            )
                        )
                        else "matrix_free"
                    )
        if (
            selected_solver in {"direct", "whitened", "block_sparse"}
            and self.frontier_backend == "tensor_train"
        ):
            raise ValueError(
                f"solver='{selected_solver}' is unavailable for tensor-train "
                "frontiers; "
                "use solver='matrix_free'."
            )
        accepted = False
        energy_after = energy_before
        metric_rank = 0
        hamiltonian_matvecs = 0
        metric_matvecs = 0
        iterations = 0
        residual_norm = float("inf")
        solver_record = selected_solver
        solver_converged = False
        message = "local solve not attempted"
        physical_blocks = 0
        hamiltonian_blocks = 0
        block_component_sizes = ()
        stored_operator_elements = 0
        solver_metric_is_identity = False
        solver_metric_identity_error = float("nan")
        solver_coordinate_residual_norm = float("nan")
        try:
            if selected_solver == "direct":
                metric, effective = self.local_operators(
                    site,
                    environment=environment,
                )
                eigenvalues = np.linalg.eigvalsh(metric)
                scale = max(
                    float(np.linalg.norm(metric, ord=np.inf)),
                    np.finfo(float).tiny,
                )
                metric_rank = int(np.count_nonzero(eigenvalues > metric_tol * scale))
                _energy, vector = _lowest_generalized_eigenpair(
                    effective,
                    metric,
                    metric_tol=metric_tol,
                )
                metric_vector = metric @ vector
                hamiltonian_vector = effective @ vector
                denominator = np.vdot(vector, metric_vector)
                energy_after = float(
                    np.real(np.vdot(vector, hamiltonian_vector) / denominator)
                )
                residual_norm = float(
                    np.linalg.norm(hamiltonian_vector - energy_after * metric_vector)
                )
                solver_converged = True
                message = "converged"
            elif selected_solver == "whitened":
                metric, effective = self.local_operators(
                    site,
                    environment=environment,
                )
                basis, whitened_hamiltonian, frame = self._whiten_local_operators(
                    metric,
                    effective,
                    metric_tol=metric_tol,
                )
                energy, reduced_vector = _lowest_hermitian_eigenpair(
                    whitened_hamiltonian
                )
                solver_coordinate_residual_norm = float(
                    np.linalg.norm(
                        whitened_hamiltonian @ reduced_vector - energy * reduced_vector
                    )
                )
                vector = basis @ reduced_vector
                metric_vector = metric @ vector
                hamiltonian_vector = effective @ vector
                denominator = np.vdot(vector, metric_vector)
                energy_after = float(
                    np.real(np.vdot(vector, hamiltonian_vector) / denominator)
                )
                residual_norm = float(
                    np.linalg.norm(hamiltonian_vector - energy_after * metric_vector)
                )
                metric_rank = int(frame["metric_rank"])
                solver_metric_is_identity = True
                solver_metric_identity_error = float(frame["identity_metric_error"])
                solver_converged = True
                message = "converged in the exact local S=I frame"
            elif selected_solver == "block_sparse":
                problem = self.local_block_problem(
                    site,
                    environment=environment,
                )
                energy_after, vector, diagnostics = problem.solve(
                    old_tensor.reshape(-1),
                    tol=eig_tol,
                    metric_tol=metric_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    random_seed=site,
                )
                metric_rank = diagnostics.projected_rank
                hamiltonian_matvecs = diagnostics.hamiltonian_matvecs
                metric_matvecs = diagnostics.metric_matvecs
                iterations = diagnostics.iterations
                residual_norm = diagnostics.residual_norm
                solver_converged = diagnostics.converged
                message = diagnostics.message
                physical_blocks = diagnostics.metric_blocks
                hamiltonian_blocks = diagnostics.hamiltonian_blocks
                block_component_sizes = diagnostics.component_sizes
                stored_operator_elements = diagnostics.stored_elements
                if not diagnostics.converged:
                    raise ValueError(diagnostics.message)
            else:
                energy_after, vector, diagnostics = lowest_generalized_davidson(
                    lambda trial: self.hamiltonian_action(
                        site,
                        trial,
                        environment=environment,
                    ),
                    lambda trial: self.metric_action(
                        site,
                        trial,
                        environment=environment,
                    ),
                    old_tensor.reshape(-1),
                    tol=eig_tol,
                    metric_tol=metric_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    random_seed=site,
                )
                metric_rank = diagnostics.projected_rank
                hamiltonian_matvecs = diagnostics.hamiltonian_matvecs
                metric_matvecs = diagnostics.metric_matvecs
                iterations = diagnostics.iterations
                residual_norm = diagnostics.residual_norm
                solver_converged = diagnostics.converged
                message = diagnostics.message
                if not diagnostics.converged:
                    raise ValueError(diagnostics.message)
            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            if (
                isinstance(self._hamiltonian_frontier, TTMPOFrontier)
                and not self.hamiltonian_contraction_is_exact
            ):
                # TT rounding makes the contracted scalar nonlinear in a local
                # tensor.  The Davidson solution is therefore only a proposal;
                # accept it against a fresh contraction of the actual objective.
                self.tensors[site][...] = vector.reshape(old_tensor.shape)
                checked_energy = self.expectation()
                accepted = (
                    np.isfinite(checked_energy)
                    and checked_energy <= energy_before + tolerance
                )
                if accepted:
                    energy_after = float(checked_energy)
                    message = f"{message}; accepted by global TT energy check"
                else:
                    energy_after = energy_before
                    message = f"{message}; rejected by global TT energy check"
            else:
                accepted = (
                    np.isfinite(energy_after)
                    and energy_after <= energy_before + tolerance
                )
                if accepted:
                    self.tensors[site][...] = vector.reshape(old_tensor.shape)
        except (ValueError, np.linalg.LinAlgError) as error:
            accepted = False
            solver_converged = False
            if message == "local solve not attempted":
                message = str(error)
            else:
                message = f"{message}; {error}"
        if not accepted:
            self.tensors[site][...] = old_tensor
            energy_after = energy_before
        self.energy = float(energy_after)
        return FrontierSiteUpdate(
            site=site,
            raw_dim=old_tensor.size,
            metric_rank=metric_rank,
            metric_rank_is_projected=(selected_solver == "matrix_free"),
            solver=solver_record,
            solver_converged=solver_converged,
            message=message,
            hamiltonian_matvecs=hamiltonian_matvecs,
            metric_matvecs=metric_matvecs,
            iterations=iterations,
            residual_norm=residual_norm,
            energy_before=energy_before,
            energy=float(energy_after),
            accepted=bool(accepted),
            physical_blocks=physical_blocks,
            hamiltonian_blocks=hamiltonian_blocks,
            block_component_sizes=block_component_sizes,
            stored_operator_elements=stored_operator_elements,
            solver_metric_is_identity=solver_metric_is_identity,
            solver_metric_identity_error=solver_metric_identity_error,
            solver_coordinate_residual_norm=solver_coordinate_residual_norm,
        )

    def natural_gradient_step(
        self,
        *,
        metric_tol: float = 1.0e-12,
        damping: float = 1.0e-6,
        trust_radius: float = 0.25,
        max_backtracks: int = 12,
        armijo: float = 1.0e-4,
    ) -> FrontierNaturalGradientUpdate:
        r"""Move all tensors along a block-metric natural-gradient direction.

        The exact local residual for tensor ``k`` is

        .. math::

            g_k = (H_k - E N_k)t_k.

        This uses the block-diagonal collection of local metrics, not the full
        cross-site metric.  Each direction is projected orthogonal to the
        radial state direction.  A shared metric trust radius and exact Armijo
        line search handle nonlinear cross terms when all tensors move.
        """
        if isinstance(self._hamiltonian_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "natural_gradient_step currently requires dense local operators; "
                "use matrix-free sweeps or LETTAVMC stochastic reconfiguration."
            )
        metric_tol = float(metric_tol)
        damping = float(damping)
        trust_radius = float(trust_radius)
        max_backtracks = int(max_backtracks)
        armijo = float(armijo)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be finite and nonnegative.")
        if not np.isfinite(trust_radius) or trust_radius <= 0.0:
            raise ValueError("trust_radius must be finite and positive.")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative.")
        if not np.isfinite(armijo) or not 0.0 < armijo < 1.0:
            raise ValueError("armijo must lie strictly between zero and one.")

        energy_before = float(self.expectation())
        state_norm = float(self.norm())
        if not np.isfinite(state_norm) or state_norm <= 0.0:
            raise ValueError("frontier-tied LETTA state is numerically zero.")
        norm_left = self._norm_frontier.build_left(self.tensors)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_left = self._hamiltonian_frontier.build_left(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)

        directions = []
        metric_ranks = []
        gradient_norm_squared = 0.0
        direction_norm_squared = 0.0
        metric_direction_norm_squared = 0.0
        directional_derivative = 0.0
        relative_directions = []
        for site, tensor in enumerate(self.tensors):
            environment = FrontierSiteEnvironment(
                site=site,
                norm_left=norm_left[site],
                norm_right=norm_right[site + 1],
                hamiltonian_left=hamiltonian_left[site],
                hamiltonian_right=hamiltonian_right[site + 1],
            )
            metric, effective = self.local_operators(
                site,
                environment=environment,
            )
            vector = tensor.reshape(-1)
            residual = effective @ vector - energy_before * (metric @ vector)
            eigenvalues, eigenvectors = np.linalg.eigh(metric)
            scale = max(
                float(np.max(np.abs(eigenvalues), initial=0.0)),
                np.finfo(float).tiny,
            )
            active = (
                eigenvalues
                > max(
                    metric_tol,
                    128.0 * np.finfo(float).eps,
                )
                * scale
            )
            rank = int(np.count_nonzero(active))
            metric_ranks.append(rank)
            if rank:
                basis = eigenvectors[:, active]
                values = eigenvalues[active]
                coefficients = basis.conj().T @ residual
                direction = -basis @ (coefficients / (values + damping * scale))
            else:
                direction = np.zeros_like(vector)
            metric_vector = metric @ vector
            metric_direction = metric @ direction
            radial_denominator = np.vdot(vector, metric_vector)
            if abs(radial_denominator) > np.finfo(float).tiny:
                direction = direction - vector * (
                    np.vdot(vector, metric_direction) / radial_denominator
                )
                metric_direction = metric @ direction
            direction = np.real_if_close(direction).astype(
                np.result_type(tensor.dtype, direction.dtype),
                copy=False,
            )
            metric_direction = metric @ direction
            directions.append(direction.reshape(tensor.shape))
            gradient_norm_squared += float(np.vdot(residual, residual).real)
            direction_norm_squared += float(np.vdot(direction, direction).real)
            metric_direction_norm_squared += max(
                float(np.vdot(direction, metric_direction).real / state_norm),
                0.0,
            )
            directional_derivative += float(
                2.0 * np.real(np.vdot(residual, direction)) / state_norm
            )
            relative_directions.append(
                float(np.linalg.norm(direction))
                / max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
            )

        maximum_relative = max(relative_directions, default=0.0)
        if not np.isfinite(maximum_relative) or maximum_relative == 0.0:
            self.energy = energy_before
            return FrontierNaturalGradientUpdate(
                energy_before=energy_before,
                energy=energy_before,
                accepted=False,
                message="natural-gradient direction is zero or nonfinite",
                step_size=0.0,
                backtracks=0,
                gradient_norm=float(np.sqrt(gradient_norm_squared)),
                preconditioned_norm=float(np.sqrt(direction_norm_squared)),
                metric_direction_norm=float(np.sqrt(metric_direction_norm_squared)),
                directional_derivative=float(directional_derivative),
                max_relative_direction=maximum_relative,
                metric_ranks=tuple(metric_ranks),
            )

        metric_direction_norm = float(np.sqrt(metric_direction_norm_squared))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            self.energy = energy_before
            return FrontierNaturalGradientUpdate(
                energy_before=energy_before,
                energy=energy_before,
                accepted=False,
                message="block-metric direction is not a descent direction",
                step_size=0.0,
                backtracks=0,
                gradient_norm=float(np.sqrt(gradient_norm_squared)),
                preconditioned_norm=float(np.sqrt(direction_norm_squared)),
                metric_direction_norm=metric_direction_norm,
                directional_derivative=float(directional_derivative),
                max_relative_direction=maximum_relative,
                metric_ranks=tuple(metric_ranks),
            )
        initial_step = min(
            1.0,
            trust_radius / max(metric_direction_norm, np.finfo(float).tiny),
        )
        old_tensors = [tensor.copy() for tensor in self.tensors]
        accepted = False
        accepted_step = 0.0
        energy_after = energy_before
        backtracks = 0
        for backtracks in range(max_backtracks + 1):
            step = initial_step * 0.5**backtracks
            candidate_tensors = [
                old + step * direction
                for old, direction in zip(old_tensors, directions)
            ]
            try:
                candidate_norm = float(
                    np.real(self._norm_frontier.scalar(candidate_tensors))
                )
                candidate_numerator = self._hamiltonian_frontier.scalar(
                    candidate_tensors
                )
                candidate_energy = float(np.real(candidate_numerator / candidate_norm))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if (
                np.isfinite(candidate_norm)
                and candidate_norm > 0.0
                and np.isfinite(candidate_energy)
                and candidate_energy
                <= energy_before + armijo * step * directional_derivative
            ):
                accepted = True
                accepted_step = step
                energy_after = float(candidate_energy)
                self.tensors = candidate_tensors
                self.balance_gauges(state_norm=np.sqrt(candidate_norm))
                break

        if not accepted:
            self.tensors = old_tensors
            energy_after = energy_before
        self.energy = float(energy_after)
        return FrontierNaturalGradientUpdate(
            energy_before=energy_before,
            energy=float(energy_after),
            accepted=accepted,
            message="accepted" if accepted else "backtracking found no energy decrease",
            step_size=float(accepted_step),
            backtracks=int(backtracks),
            gradient_norm=float(np.sqrt(gradient_norm_squared)),
            preconditioned_norm=float(np.sqrt(direction_norm_squared)),
            metric_direction_norm=metric_direction_norm,
            directional_derivative=float(directional_derivative),
            max_relative_direction=float(accepted_step * maximum_relative),
            metric_ranks=tuple(metric_ranks),
        )

    def run(
        self,
        *,
        nsweeps: int = 4,
        sweep_offset: int = 0,
        tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        solver="auto",
        matrix_free_threshold: int = 256,
        block_sparse_max_elements: int | None = 4_000_000,
        eig_tol: float = 1.0e-10,
        maxiter: int | None = None,
        max_subspace: int = 32,
        natural_gradient_every: int = 0,
        natural_gradient_damping: float = 1.0e-6,
        natural_gradient_trust_radius: float = 0.25,
        natural_gradient_max_backtracks: int = 12,
        virtual_canonicalization: bool = False,
        frontier_canonicalization: bool = False,
        frontier_gauge_max_condition: float = 1.0e8,
        frontier_gauge_weighting="uniform",
        verbose: bool = False,
    ):
        """Optimize all tensors without constructing the full Hilbert basis."""
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        sweep_offset = int(sweep_offset)
        if sweep_offset < 0:
            raise ValueError("sweep_offset must be nonnegative.")
        natural_gradient_every = int(natural_gradient_every)
        if natural_gradient_every < 0:
            raise ValueError("natural_gradient_every must be nonnegative.")
        if nsweeps and not self.norm_contraction_is_exact:
            raise ValueError(
                "variational sweeps require an exact norm contraction; all-TT "
                "truncated norms are available only for scalar diagnostics."
            )
        if nsweeps and not self.hamiltonian_action_is_hermitian:
            raise ValueError(
                "variational sweeps require an exact or explicitly Hermitized "
                "Hamiltonian action; set tt_hermitize=True."
            )
        if (
            isinstance(self._hamiltonian_frontier, TTMPOFrontier)
            and natural_gradient_every
        ):
            raise ValueError(
                "natural-gradient sweeps are unavailable for tensor-train "
                "frontiers; use LETTAVMC stochastic reconfiguration."
            )
        if isinstance(self._norm_frontier, TTMPOFrontier) and frontier_canonicalization:
            raise ValueError(
                "frontier canonicalization currently requires dense exact "
                "messages; use virtual_canonicalization instead."
            )
        if virtual_canonicalization and frontier_canonicalization:
            raise ValueError(
                "virtual and frontier canonicalization cannot both be enabled."
            )
        previous = self.expectation()
        self.energy = previous
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            directional_sweep = sweep_offset + sweep
            updates = []
            frontier_gauge = None
            if frontier_canonicalization:
                frontier_gauge = self.canonicalize_frontier_gauge(
                    metric_tol=metric_tol,
                    max_condition=frontier_gauge_max_condition,
                    weighting=frontier_gauge_weighting,
                )
            if directional_sweep % 2 == 0:
                if virtual_canonicalization:
                    self.canonicalize_virtual("right")
                    if not self.hamiltonian_contraction_is_exact:
                        previous = self.expectation()
                        self.energy = previous
                norm_right = self._norm_frontier.build_right(self.tensors)
                hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                for site in range(len(self.dims)):
                    environment = FrontierSiteEnvironment(
                        site=site,
                        norm_left=moving_norm,
                        norm_right=norm_right[site + 1],
                        hamiltonian_left=moving_hamiltonian,
                        hamiltonian_right=hamiltonian_right[site + 1],
                    )
                    updates.append(
                        self.optimize_site(
                            site,
                            metric_tol=metric_tol,
                            solver=solver,
                            matrix_free_threshold=matrix_free_threshold,
                            block_sparse_max_elements=block_sparse_max_elements,
                            eig_tol=eig_tol,
                            maxiter=maxiter,
                            max_subspace=max_subspace,
                            energy_before=self.energy,
                            environment=environment,
                        )
                    )
                    moving_norm = self._norm_frontier.advance_left(
                        moving_norm,
                        self.tensors,
                        site,
                    )
                    moving_hamiltonian = self._hamiltonian_frontier.advance_left(
                        moving_hamiltonian,
                        self.tensors,
                        site,
                    )
            else:
                if virtual_canonicalization:
                    self.canonicalize_virtual("left")
                    if not self.hamiltonian_contraction_is_exact:
                        previous = self.expectation()
                        self.energy = previous
                norm_left = self._norm_frontier.build_left(self.tensors)
                hamiltonian_left = self._hamiltonian_frontier.build_left(self.tensors)
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                for site in range(len(self.dims) - 1, -1, -1):
                    environment = FrontierSiteEnvironment(
                        site=site,
                        norm_left=norm_left[site],
                        norm_right=moving_norm,
                        hamiltonian_left=hamiltonian_left[site],
                        hamiltonian_right=moving_hamiltonian,
                    )
                    updates.append(
                        self.optimize_site(
                            site,
                            metric_tol=metric_tol,
                            solver=solver,
                            matrix_free_threshold=matrix_free_threshold,
                            block_sparse_max_elements=block_sparse_max_elements,
                            eig_tol=eig_tol,
                            maxiter=maxiter,
                            max_subspace=max_subspace,
                            energy_before=self.energy,
                            environment=environment,
                        )
                    )
                    moving_norm = self._norm_frontier.advance_right(
                        moving_norm,
                        self.tensors,
                        site,
                    )
                    moving_hamiltonian = self._hamiltonian_frontier.advance_right(
                        moving_hamiltonian,
                        self.tensors,
                        site,
                    )
            boundary_cut = len(self.dims) if directional_sweep % 2 == 0 else 0
            norm = float(
                np.real(
                    self._completed_frontier_scalar(
                        self._norm_frontier,
                        moving_norm,
                        boundary_cut,
                    )
                )
            )
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("frontier-tied LETTA state is numerically zero.")
            numerator = self._completed_frontier_scalar(
                self._hamiltonian_frontier,
                moving_hamiltonian,
                boundary_cut,
            )
            directional_endpoint_energy = float(np.real(numerator / norm))
            if self.hamiltonian_contraction_is_exact:
                self.balance_gauges(state_norm=np.sqrt(norm))
                energy = directional_endpoint_energy
            else:
                # Per-tensor gauge rescalings preserve the exact state but can
                # change a rank-truncated TT contraction through finite-rank
                # rounding.  Keep the gauge in which every proposal was checked.
                energy = self.expectation()
            self.energy = energy
            natural_gradient = None
            if (
                natural_gradient_every
                and (directional_sweep + 1) % natural_gradient_every == 0
            ):
                natural_gradient = self.natural_gradient_step(
                    metric_tol=metric_tol,
                    damping=natural_gradient_damping,
                    trust_radius=natural_gradient_trust_radius,
                    max_backtracks=natural_gradient_max_backtracks,
                )
                energy = float(self.energy)
            delta = abs(energy - previous)
            solver_failures = sum(not update.solver_converged for update in updates)
            self.energy = energy
            self.history.append(
                {
                    "sweep": directional_sweep,
                    "energy": energy,
                    "delta_energy": delta,
                    "accepted_sites": sum(update.accepted for update in updates),
                    "solver_failures": solver_failures,
                    "hamiltonian_matvecs": sum(
                        update.hamiltonian_matvecs for update in updates
                    ),
                    "metric_matvecs": sum(update.metric_matvecs for update in updates),
                    "natural_gradient": natural_gradient,
                    "frontier_gauge": frontier_gauge,
                    "contraction_is_exact": self.contraction_is_exact,
                    "norm_contraction_is_exact": self.norm_contraction_is_exact,
                    "hamiltonian_contraction_is_exact": (
                        self.hamiltonian_contraction_is_exact
                    ),
                    "hamiltonian_action_is_hermitian": (
                        self.hamiltonian_action_is_hermitian
                    ),
                    "directional_endpoint_energy": directional_endpoint_energy,
                    "tt_diagnostics": self.tt_diagnostics,
                    "updates": updates,
                }
            )
            if verbose:
                print(
                    f"frontier-LETTA sweep={directional_sweep:2d} "
                    f"E={energy: .12f} "
                    f"dE={delta:.3e} "
                    f"accepted={sum(update.accepted for update in updates)} "
                    f"failed={solver_failures} "
                    f"natural={None if natural_gradient is None else natural_gradient.accepted}",
                    flush=True,
                )
            natural_gradient_stationary = (
                natural_gradient is None
                or natural_gradient.accepted
                or abs(natural_gradient.directional_derivative) <= tol
            )
            if delta <= tol and solver_failures == 0 and natural_gradient_stationary:
                self.converged = True
                break
            previous = energy
        return self

    def state_vector(self, *, normalize=False):
        """Build the explicit state vector for small-system validation only."""
        configs = np.asarray(list(np.ndindex(*self.dims)), dtype=np.intp)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        environment = np.ones((len(configs), 1), dtype=dtype)
        for site, physical_sites in enumerate(self.physical_sites):
            tensor = self.tensors[site]
            left_dim, right_dim = tensor.shape[:2]
            columns = np.ravel_multi_index(
                tuple(configs[:, index] for index in physical_sites),
                tuple(self.dims[index] for index in physical_sites),
            )
            transfer = tensor.reshape(left_dim, right_dim, -1)[:, :, columns].transpose(
                2, 0, 1
            )
            environment = np.einsum(
                "ca,cab->cb",
                environment,
                transfer,
                optimize=True,
            )
        vector = environment[:, 0]
        if normalize:
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("frontier-tied LETTA state is zero or nonfinite.")
            vector = vector / norm
        return vector

    def fidelity(self, state) -> float:
        return _normalized_fidelity(self.state_vector(), state)


__all__ = [
    "FrontierBondExpansion",
    "FrontierGaugeUpdate",
    "FrontierNaturalGradientUpdate",
    "FrontierSiteEnvironment",
    "FrontierSiteUpdate",
    "FrontierTiedLETTA",
    "FrontierTwoSiteUpdate",
]
