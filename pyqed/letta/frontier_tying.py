"""Exact and tensor-train graph-tied LETTA frontier contraction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from pyqed.tn import MPO

from .block_mpo_frontier import BlockFrontierMessage, BlockMPOFrontier
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
from pyqed.tn import Hamiltonian
from pyqed.tn.effective_operator import resolve_workers
from .matrix_free import lowest_generalized_davidson
from .mpo_frontier import MPOFrontier
from .physical_blocks import (
    PhysicalBlockGeneralizedProblem,
    PhysicalBlockLayout,
    hamiltonian_physical_connectivity,
)
from .tt_frontier import (
    TermwiseBlockMPOFrontier,
    TermwiseTTMPOFrontier,
    TTFrontier,
    TTMPOFrontier,
)


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
class FrontierMergedSolveDiagnostics:
    """Verification record for one merged adjacent-pair eigenproblem."""

    method: str
    attempts: tuple[str, ...]
    verified: bool
    lowest_root_certified: bool
    fallback_reason: str
    dense_fallback: bool
    metric_requested_rank: int
    metric_numerical_rank: int
    metric_min_positive: float
    metric_condition: float
    backward_residual: float
    metric_dual_residual: float
    metric_dual_relative_residual: float
    null_residual: float
    warm_energy: float
    upper_bound_gap: float
    metric_support: str = "regularized"
    discarded_support_residual: float = 0.0


@dataclass(frozen=True)
class FrontierSiteEnvironment:
    """Numerical left/right messages surrounding one local tensor."""

    site: int
    norm_left: object
    norm_right: object
    hamiltonian_left: object
    hamiltonian_right: object


@dataclass(frozen=True)
class FrontierPairEnvironment:
    """Cached exact messages surrounding one adjacent merged pair."""

    sites: tuple[int, int]
    union_sites: tuple[int, ...]
    norm_left: object
    norm_right: object
    hamiltonian_left: object
    hamiltonian_right: object


@dataclass(frozen=True)
class FrontierBlockEnvironment:
    """Cached exact messages surrounding one contiguous merged block."""

    sites: tuple[int, ...]
    union_sites: tuple[int, ...]
    norm_left: object
    norm_right: object
    hamiltonian_left: object
    hamiltonian_right: object
    hamiltonian_outer_left: object | None = None
    hamiltonian_outer_right: object | None = None


@dataclass(frozen=True)
class _FrontierPairPlan:
    """Value-independent contraction topology for one adjacent pair."""

    site: int
    union_sites: tuple[int, ...]
    merged_shape: tuple[int, ...]
    identity_tensor: np.ndarray
    norm_engine: object
    hamiltonian_engine: object
    fingerprint: tuple


@dataclass(frozen=True)
class _FrontierBlockPlan:
    """Value-independent contraction topology for one contiguous block."""

    sites: tuple[int, ...]
    union_sites: tuple[int, ...]
    merged_shape: tuple[int, ...]
    identity_tensors: tuple[np.ndarray, ...]
    norm_engine: object
    hamiltonian_engine: object
    fingerprint: tuple


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
    residual_components: int = 0
    relative_discarded_weight: float = 0.0


@dataclass(frozen=True)
class FrontierBondReduction:
    """Diagnostic record for one null-space virtual-bond reduction."""

    cut: int
    old_dimension: int
    new_dimension: int
    support_source: str
    sector_dimensions: tuple[tuple[tuple[int, ...], int, int], ...]
    relative_discarded_weight: float
    norm_error: float
    energy_before: float
    energy: float


@dataclass(frozen=True)
class FrontierTieReduction:
    """Diagnostic record for removing one redundant future-physical tie."""

    edge: tuple[int, int]
    relative_discarded_weight: float
    norm_error: float
    energy_before: float
    energy: float
    exact: bool


@dataclass(frozen=True)
class FrontierBondRefresh:
    """Diagnostic record for one saturated AMEn expand--optimize--retract."""

    cut: int
    temporary_dimension: int
    target_dimension: int
    overlap_sites: tuple[int, ...]
    conditional_ranks: tuple[int, ...]
    relative_truncation_error: float
    sector_dimensions: tuple[tuple[tuple[int, ...], int], ...] = ()
    subspace_change: float = 0.0
    accepted: bool = True


@dataclass(frozen=True)
class _PendingAMEnRetraction:
    """Temporary saturated bond retained through its neighboring solve."""

    cut: int
    target_dimension: int
    expansion_direction: str
    source_site: int
    mixing_scale: float
    energy_before: float
    left_tensor: np.ndarray
    right_tensor: np.ndarray
    anchor_norm: object
    anchor_hamiltonian: object
    occupied_basis: tuple[np.ndarray, ...]


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
    merged_solve: FrontierMergedSolveDiagnostics | None = None
    outer_cycles: int = 1
    merged_energy_history: tuple[float, ...] = ()
    factor_energy_history: tuple[float, ...] = ()
    pair_operator_backend: str = "dense"
    pair_operator_stored_elements: int = 0


@dataclass(frozen=True)
class FrontierBlockUpdate:
    """Diagnostic record for one direct contiguous-block update."""

    sites: tuple[int, ...]
    raw_merged_dim: int
    energy_before: float
    merged_energy: float
    attempted_energy: float
    energy: float
    accepted: bool
    relative_truncation_error: float
    conditional_ranks: tuple[int, ...]
    local_update: FrontierSiteUpdate
    merged_solve: FrontierMergedSolveDiagnostics
    operator_backend: str
    operator_stored_elements: int


class FrontierTiedLETTA:
    r"""Unrestricted graph-tied LETTA contracted by frontier messages.

    This class represents the same local tensors as :class:`DenseTiedLETTA`,
    but it accepts a :class:`Hamiltonian` and never constructs the
    many-body configuration table during initialization or optimization.
    The local-term sum is converted to an exact finite-state MPO.  Numerical
    left/right double-layer messages are cached across each directional sweep
    and reused by every local matrix or Davidson action.  The ``compressed``
    and ``identity_block`` backends are exact. The exact ``termwise`` backend
    splits the Hamiltonian into bond-one strings and streams them for scalar
    contractions. The ``tensor_train`` backend keeps the cheaper norm frontier
    exact by default and stores the Hamiltonian frontier as a boundary MPS/TT.
    It is fully exact only when its ranks and tolerances are unrestricted. The
    dense-frontier cost is governed
    by the weighted frontier induced by the chosen site ordering and MPO bond;
    it is exponential in that width, which can still grow with system size for
    dense or poorly ordered graphs.
    """

    _has_charge_resolved_two_site_split = False
    _has_charge_resolved_block_split = False

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        parent_sets,
        *legacy_parent_sets,
        bond_dim: int = 1,
        bond_dims=None,
        tensors=None,
        seed: int | None = None,
        frontier_backend="compressed",
        chunk_size=8,
        chunk_memory=64,
        chunk_span=None,
        workers=1,
        path_optimizer="greedy",
        max_rank: int | None = None,
        rtol: float = 0.0,
        atol: float = 0.0,
        transfer_max_rank: int | None = None,
        transfer_rtol: float = 0.0,
        transfer_atol: float = 0.0,
        tt_absorption="structured",
        tt_norm_backend="exact",
        tt_hermitize: bool = True,
        tt_channels="component",
        tt_gauge: bool = False,
        compute_dtype=None,
        device="cpu",
        route_memory=32,
        action_memory=32,
    ):
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a Hamiltonian.")
        if len(legacy_parent_sets) > 1:
            raise TypeError(
                "FrontierTiedLETTA accepts (hamiltonian, parent_sets); the "
                "temporary legacy form is (hamiltonian, dims, parent_sets)."
            )
        if legacy_parent_sets:
            legacy_dims = _validated_dims(parent_sets)
            parent_sets = legacy_parent_sets[0]
            if hamiltonian.dims != legacy_dims:
                raise ValueError("hamiltonian dims are inconsistent with legacy dims.")
        self.hamiltonian = hamiltonian
        self.sites = hamiltonian.sites
        self.dims = hamiltonian.dims
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.physical_groups = tuple(
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
        if self.frontier_backend in {
            "protected",
            "protected_tt",
            "hamiltonian_tt",
        }:
            self.frontier_backend = "tensor_train"
        if self.frontier_backend in {"termwise_block", "termwise_exact", "streamed"}:
            self.frontier_backend = "termwise"
        if self.frontier_backend not in {
            "compressed",
            "identity_block",
            "termwise",
            "tensor_train",
        }:
            raise ValueError(
                "frontier_backend must be 'compressed', 'identity_block', "
                "'termwise', or 'tensor_train'."
            )
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
        if chunk_span is None:
            self.chunk_span = None
        else:
            if isinstance(chunk_span, (bool, np.bool_)):
                raise TypeError("chunk_span must be a positive integer or None.")
            self.chunk_span = int(chunk_span)
            if self.chunk_span < 1:
                raise ValueError("chunk_span must be a positive integer or None.")
        self.workers = resolve_workers(workers)
        self.route_memory = float(route_memory)
        if not np.isfinite(self.route_memory) or self.route_memory < 0.0:
            raise ValueError("route_memory must be finite and nonnegative.")
        self.action_memory = float(action_memory)
        if not np.isfinite(self.action_memory) or self.action_memory < 0.0:
            raise ValueError("action_memory must be finite and nonnegative.")
        self._solver_executor = (
            ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="letta-local",
            )
            if self.workers > 1
            else None
        )
        self.device = str(device).strip().lower().replace("_", "-")
        if self.device == "numpy":
            self.device = "cpu"
        elif self.device in {"gpu", "cupy"}:
            self.device = "cuda"
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")
        self.compute_dtype = (
            None
            if compute_dtype is None
            or str(compute_dtype).lower() in {"same", "native"}
            else np.dtype(compute_dtype)
        )
        if self.compute_dtype is not None and self.compute_dtype.kind not in "fc":
            raise TypeError("compute_dtype must be a real or complex floating dtype.")
        if (
            (self.compute_dtype is not None or self.device != "cpu")
            and self.frontier_backend not in {"identity_block", "termwise"}
        ):
            raise ValueError(
                "compute_dtype/device acceleration currently requires the "
                "identity_block or termwise frontier backend."
            )
        self.tt_options = {
            "max_rank": max_rank,
            "rtol": rtol,
            "atol": atol,
            "transfer_max_rank": transfer_max_rank,
            "transfer_rtol": transfer_rtol,
            "transfer_atol": transfer_atol,
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
        self.tt_channels = str(tt_channels).lower().replace("-", "_")
        if self.tt_channels not in {"component", "term"}:
            raise ValueError("tt_channels must be 'component' or 'term'.")
        if not isinstance(tt_gauge, (bool, np.bool_)):
            raise TypeError("tt_gauge must be boolean.")
        self.tt_gauge = bool(tt_gauge)
        self.rng = np.random.default_rng(seed)

        bonds = self._bond_dims()
        shapes = tuple(
            (bonds[site], bonds[site + 1])
            + tuple(self.dims[index] for index in physical_sites)
            for site, physical_sites in enumerate(self.physical_groups)
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
        self._project_initial_tensors()

        identity_mpo = MPO(
            [np.eye(dim, dtype=parameter_dtype)[None, None, :, :] for dim in self.dims],
            sites=self.sites,
        )
        uncompressed_hamiltonian_mpo = self.hamiltonian.to_mpo()
        self.uncompressed_hamiltonian_mpo_bond_dim = max(
            uncompressed_hamiltonian_mpo.bond_dims
        )
        compressed_hamiltonian_mpo = uncompressed_hamiltonian_mpo.compress()
        self.compressed_hamiltonian_mpo_bond_dim = max(
            compressed_hamiltonian_mpo.bond_dims
        )
        frontier_arguments = (self.dims, self.physical_groups, shapes)
        if self.tt_norm_backend == "tensor_train":
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
                self.physical_groups,
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
                local_qns=getattr(
                    getattr(self, "abelian_layout", None), "local_qns", None
                ),
                bond_qns=getattr(
                    getattr(self, "abelian_layout", None), "bond_qns", None
                ),
                compute_dtype=self.compute_dtype,
                device=self.device,
                route_memory=self.route_memory,
                action_memory=self.action_memory,
            )
        elif self.frontier_backend == "termwise":
            self.hamiltonian_mpo = uncompressed_hamiltonian_mpo
            self._hamiltonian_frontier = TermwiseBlockMPOFrontier(
                self.hamiltonian,
                self.physical_groups,
                shapes,
                optimize=path_optimizer,
                local_qns=getattr(
                    getattr(self, "abelian_layout", None), "local_qns", None
                ),
                bond_qns=getattr(
                    getattr(self, "abelian_layout", None), "bond_qns", None
                ),
                chunk_size=self.chunk_size,
                chunk_memory=self.chunk_memory,
                chunk_span=self.chunk_span,
                workers=self.workers,
                compute_dtype=self.compute_dtype,
                device=self.device,
            )
        else:
            self.hamiltonian_mpo = compressed_hamiltonian_mpo
            self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                self.hamiltonian,
                self.physical_groups,
                shapes,
                optimize=path_optimizer,
                local_qns=getattr(
                    getattr(self, "abelian_layout", None),
                    "local_qns",
                    None,
                ),
                channel_grouping=self.tt_channels,
                **self.tt_options,
            )
        self._pair_plan_cache: dict[int, _FrontierPairPlan] = {}
        self._block_plan_cache: dict[tuple[int, int], _FrontierBlockPlan] = {}
        self._davidson_recycle: dict[tuple, tuple[np.ndarray, ...]] = {}
        self._pair_last_visited: dict[int, int] = {}
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
        self.tt_gauge_updates = ()
        if (
            self.tt_gauge
            and self.frontier_backend == "tensor_train"
            and self.norm_contraction_is_exact
        ):
            try:
                self.tt_gauge_updates = tuple(
                    self.canonicalize_frontier_gauge(weighting="probability")
                )
            except NotImplementedError:
                self.tt_gauge_updates = ()
        self.energy = self.expectation()

    def _bond_dims(self):
        return self._virtual_bond_dims

    def _project_initial_tensors(self):
        """Subclass hook for structural support imposed before contraction."""

    def _prepare_amen_directions(self, cut, direction, directions):
        """Subclass hook for symmetry-resolved enrichment directions."""
        return np.asarray(directions)

    def _amen_compression_labels(self, cut, target_dimension):
        """Subclass hook returning labels retained by a temporary retraction."""
        return None

    def _set_amen_compressed_bond_layout(self, cut, labels):
        """Subclass hook updating structural metadata after AMEn retraction."""
        if labels is not None:
            raise TypeError("dense frontier bonds do not carry charge labels.")

    def _prepare_saturated_amen_directions(self, cut, direction, directions):
        """Prepare fixed-cap residual directions and optional temporary labels."""
        return np.asarray(directions), None

    def _bond_gauge_blocks(self, cut):
        """Return virtual-index blocks that a state-preserving gauge may mix."""
        dimension = self._bond_dims()[int(cut)]
        return (np.arange(dimension, dtype=np.intp),)

    def _apply_bond_gauge_constraints(self):
        """Subclass hook restoring exact structural zeros after a bond gauge."""

    def _local_action_mask(self, site):
        """Subclass hook restricting matrix-free conditional coordinates."""
        return None

    @property
    def uses_tensor_train_frontier(self) -> bool:
        return isinstance(self._norm_frontier, TTMPOFrontier) or isinstance(
            self._hamiltonian_frontier,
            TTMPOFrontier,
        )

    @property
    def requires_matrix_free_solver(self) -> bool:
        return self.uses_tensor_train_frontier or isinstance(
            self._hamiltonian_frontier,
            TermwiseBlockMPOFrontier,
        )

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Virtual dimensions at every cut, including unit boundaries."""
        return self._bond_dims()

    def draw(self, path=None, **kwargs):
        """Draw the backbone, physical legs, and tied physical indices."""
        from .drawing import draw_frontier_letta

        return draw_frontier_letta(self, path=path, **kwargs)

    def _copy_public_settings_to(self, result):
        for name in (
            "graph",
            "ordering",
            "inverse_ordering",
            "original_hamiltonian",
            "original_graph",
            "target_charge",
            "D",
            "adaptive_bond",
            "tie_backbone",
            "_maximum_bond_dims",
            "_null_reduced_cuts",
        ):
            if hasattr(self, name):
                value = getattr(self, name)
                setattr(
                    result,
                    name,
                    value if name == "original_hamiltonian" else deepcopy(value),
                )
        return result

    def _condition_sweep_bond(
        self,
        cut,
        message,
        *,
        direction,
        metric_tol,
        max_condition,
    ):
        r"""Whiten the already-contracted side of one active sweep bond.

        AMEn appends a residual direction using a Euclidean QR, but the next
        LETTA local problem is measured by the graph-dependent frontier norm.
        This applies a state-preserving, one-sided frontier gauge immediately
        after enrichment.  It uses only the moving norm message, so it does
        not rebuild or retain another set of environments.
        """
        cut = int(cut)
        direction = str(direction).lower()
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'.")
        dimension = self._bond_dims()[cut]
        conditional = np.asarray(message).reshape(dimension, dimension, -1)
        gram = np.sum(conditional, axis=-1)
        if direction == "left":
            gram = gram.T
        gram = self._hermitian_part(gram)
        metric_tol = max(float(metric_tol), 128.0 * np.finfo(float).eps)
        max_condition = float(max_condition)
        gauge = np.eye(dimension, dtype=np.result_type(gram, self.tensors[cut - 1]))
        applied = False
        tiny = np.finfo(float).tiny
        for indices in self._bond_gauge_blocks(cut):
            indices = np.asarray(indices, dtype=np.intp)
            if not indices.size:
                continue
            block = self._hermitian_part(gram[np.ix_(indices, indices)])
            values, vectors = np.linalg.eigh(block)
            scale = max(float(np.max(values, initial=0.0)), tiny)
            if np.any(values <= metric_tol * scale):
                # A truly null sector has no stable inverse.  Null-bond
                # reduction handles it after both sides have been varied.
                continue
            floor = scale / max_condition**2
            regularized = np.maximum(values, floor)
            mean = max(float(np.mean(regularized)), tiny)
            normalized = regularized / mean
            powers = (
                1.0 / np.sqrt(normalized)
                if direction == "right"
                else np.sqrt(normalized)
            )
            gauge[np.ix_(indices, indices)] = (
                vectors * powers
            ) @ vectors.conj().T
            applied = True
        if not applied:
            return False

        left_tensor = self.tensors[cut - 1]
        transformed_left = np.tensordot(left_tensor, gauge, axes=(1, 0))
        self.tensors[cut - 1] = np.real_if_close(
            np.moveaxis(transformed_left, -1, 1)
        )
        right_tensor = self.tensors[cut]
        transformed_right = np.linalg.solve(
            gauge,
            right_tensor.reshape(dimension, -1),
        )
        self.tensors[cut] = np.real_if_close(
            transformed_right.reshape(right_tensor.shape)
        )
        self._apply_bond_gauge_constraints()
        return True

    @staticmethod
    def _gram_support(matrix, *, rtol, atol):
        matrix = FrontierTiedLETTA._hermitian_part(matrix)
        values, vectors = np.linalg.eigh(matrix)
        scale = max(
            float(np.max(np.abs(values), initial=0.0)),
            np.finfo(float).tiny,
        )
        threshold = max(
            float(atol),
            float(rtol) * scale,
            256.0 * np.finfo(float).eps * scale,
        )
        active = values > threshold
        positive = np.maximum(values, 0.0)
        total = float(np.sum(positive))
        discarded = (
            float(np.sum(positive[~active]) / total)
            if total > np.finfo(float).tiny
            else 0.0
        )
        return vectors[:, active], int(np.count_nonzero(active)), discarded

    def _null_bond_basis(self, cut, left, right, *, rtol, atol):
        left_basis, left_rank, left_discarded = self._gram_support(
            left,
            rtol=rtol,
            atol=atol,
        )
        right_basis, right_rank, right_discarded = self._gram_support(
            right,
            rtol=rtol,
            atol=atol,
        )
        if left_rank <= right_rank:
            return left_basis, "left", (), left_discarded, None
        return right_basis, "right", (), right_discarded, None

    def _set_reduced_bond_layouts(self, labels_by_cut):
        if labels_by_cut:
            raise TypeError("dense frontier bonds do not carry charge labels.")

    def reduce_null_bonds(
        self,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> tuple[FrontierBondReduction, ...]:
        r"""Remove numerically null virtual directions without changing the state.

        With the default zero user tolerances, only directions below a
        machine-precision floor are removed.  Positive tolerances opt into
        approximate compression.  Exact frontier norm messages are required.
        """
        rtol = float(rtol)
        atol = float(atol)
        if not np.isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative.")
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        if isinstance(self._norm_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "null-space bond reduction requires exact dense frontier Grams."
            )

        old_dimensions = self._bond_dims()
        left_messages = self._norm_frontier.build_left(self.tensors)
        right_messages = self._norm_frontier.build_right(self.tensors)
        bases = {}
        diagnostics = {}
        labels_by_cut = {}
        for cut in range(1, len(self.dims)):
            left, right = self.frontier_bond_grams(
                cut,
                left_messages=left_messages,
                right_messages=right_messages,
            )
            (
                basis,
                source,
                sector_dimensions,
                discarded,
                labels,
            ) = self._null_bond_basis(
                cut,
                left,
                right,
                rtol=rtol,
                atol=atol,
            )
            new_dimension = int(basis.shape[1])
            if new_dimension < 1:
                raise ValueError(
                    f"bond {cut} has no occupied support; the state norm is zero."
                )
            if new_dimension >= old_dimensions[cut]:
                continue
            bases[cut] = np.asarray(basis)
            diagnostics[cut] = (
                source,
                tuple(sector_dimensions),
                float(discarded),
            )
            if labels is not None:
                labels_by_cut[cut] = tuple(labels)

        if not bases:
            return ()

        energy_before = self.expectation()
        norm_before = float(np.real(self._norm_frontier.scalar(self.tensors)))
        reduced_tensors = []
        for site, tensor in enumerate(self.tensors):
            left_basis = bases.get(site)
            if left_basis is not None:
                tensor = np.tensordot(
                    left_basis.conj().T,
                    tensor,
                    axes=(1, 0),
                )
            right_basis = bases.get(site + 1)
            if right_basis is not None:
                tensor = np.tensordot(
                    tensor,
                    right_basis,
                    axes=(1, 0),
                )
                tensor = np.moveaxis(tensor, -1, 1)
            reduced_tensors.append(np.asarray(tensor))
        self.tensors = reduced_tensors

        dimensions = list(old_dimensions)
        for cut, basis in bases.items():
            dimensions[cut] = int(basis.shape[1])
        self._virtual_bond_dims = tuple(dimensions)
        self.bond_dim = max(self._virtual_bond_dims)
        self._set_reduced_bond_layouts(labels_by_cut)
        self._rebuild_frontier_engines()

        norm_after = float(np.real(self._norm_frontier.scalar(self.tensors)))
        energy_after = self.expectation()
        norm_error = abs(norm_after - norm_before) / max(abs(norm_before), 1.0)
        self.energy = energy_after
        self.history = []
        self.converged = False
        suppressed = set(getattr(self, "_null_reduced_cuts", ()))
        suppressed.update(bases)
        self._null_reduced_cuts = suppressed
        return tuple(
            FrontierBondReduction(
                cut=cut,
                old_dimension=old_dimensions[cut],
                new_dimension=int(bases[cut].shape[1]),
                support_source=diagnostics[cut][0],
                sector_dimensions=diagnostics[cut][1],
                relative_discarded_weight=diagnostics[cut][2],
                norm_error=float(norm_error),
                energy_before=energy_before,
                energy=energy_after,
            )
            for cut in sorted(bases)
        )

    def _replace_parent_sets(self, parent_sets):
        """Install a new tie graph after tensors have been reshaped."""

        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.physical_groups = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        self._physical_block_connectivity_cache = {}
        if hasattr(self, "abelian_layout"):
            self.local_masks = self.abelian_layout.local_masks(
                self.physical_groups
            )
            self._apply_local_masks()
        self._rebuild_frontier_engines()

    def prune_ties(
        self,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
        energy_tol: float | None = None,
    ) -> tuple[FrontierTieReduction, ...]:
        r"""Remove future-physical legs whose local dependence has rank one.

        For a tie ``(i, j)``, the corresponding mode of tensor ``i`` is
        factorized as ``u(rest) v(s_j)``.  The factor ``v`` is absorbed into
        the owned physical leg of tensor ``j`` before the leg is removed, so
        an exactly rank-one mode preserves the represented state. Positive
        ``rtol``/``atol`` values opt into a rank-one approximation; every
        accepted proposal is additionally checked by exact norm and energy
        contractions. ``energy_tol`` is a relative allowed energy change and
        defaults to the same tolerance as the local discarded weight.
        """

        rtol = float(rtol)
        atol = float(atol)
        if not np.isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative.")
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        if energy_tol is None:
            energy_tol = max(rtol, 0.0)
        energy_tol = float(energy_tol)
        if not np.isfinite(energy_tol) or energy_tol < 0.0:
            raise ValueError("energy_tol must be finite and nonnegative.")
        if not self.contraction_is_exact:
            raise ValueError("tie pruning requires exact norm and energy contraction.")

        records = []
        machine_floor = 512.0 * np.finfo(float).eps
        for owner in range(len(self.dims) - 1):
            for parent in tuple(self.parent_sets[owner]):
                group = self.physical_groups[owner]
                axis = 2 + group.index(parent)
                tensor = self.tensors[owner]
                moved = np.moveaxis(tensor, axis, -1)
                matrix = moved.reshape(-1, self.dims[parent])
                left, singular_values, right = np.linalg.svd(
                    matrix,
                    full_matrices=False,
                )
                total = float(np.linalg.norm(singular_values))
                discarded = float(np.linalg.norm(singular_values[1:]))
                relative = discarded / max(total, np.finfo(float).tiny)
                threshold = max(
                    atol / max(total, np.finfo(float).tiny),
                    rtol,
                    machine_floor,
                )
                if relative > threshold:
                    continue

                energy_before = float(self.expectation())
                norm_before = float(
                    np.real(self._norm_frontier.scalar(self.tensors))
                )
                tensors_before = [value.copy() for value in self.tensors]
                parents_before = self.parent_sets

                reduced = (left[:, 0] * singular_values[0]).reshape(
                    moved.shape[:-1]
                )
                target = self.tensors[parent]
                factor_shape = [1] * target.ndim
                factor_shape[2] = self.dims[parent]
                self.tensors[owner] = reduced
                self.tensors[parent] = target * right[0].reshape(factor_shape)
                new_parents = list(self.parent_sets)
                new_parents[owner] = tuple(
                    candidate
                    for candidate in new_parents[owner]
                    if candidate != parent
                )
                try:
                    self._replace_parent_sets(tuple(new_parents))
                    norm_after = float(
                        np.real(self._norm_frontier.scalar(self.tensors))
                    )
                    energy_after = float(self.expectation())
                    norm_error = abs(norm_after - norm_before) / max(
                        abs(norm_before), 1.0
                    )
                    allowed_energy = max(
                        machine_floor,
                        energy_tol,
                    ) * max(1.0, abs(energy_before))
                    accepted = bool(
                        np.isfinite(energy_after)
                        and abs(energy_after - energy_before) <= allowed_energy
                    )
                except Exception:
                    accepted = False
                    norm_error = float("inf")
                    energy_after = energy_before
                if not accepted:
                    self.tensors = tensors_before
                    self._replace_parent_sets(parents_before)
                    continue

                records.append(
                    FrontierTieReduction(
                        edge=(int(owner), int(parent)),
                        relative_discarded_weight=relative,
                        norm_error=float(norm_error),
                        energy_before=energy_before,
                        energy=energy_after,
                        exact=bool(relative <= machine_floor),
                    )
                )

        if records:
            self.energy = float(self.expectation())
            self.history = []
            self.converged = False
        return tuple(records)

    def adapt_bonds(
        self,
        *,
        rtol: float = 1.0e-8,
        growth: int = 2,
        direction="right",
        strategy="residual",
        scale: float = 1.0e-3,
    ) -> tuple[FrontierBondExpansion, ...]:
        """Grow saturated virtual cuts up to the configured public ``D`` cap."""
        if not getattr(self, "adaptive_bond", False):
            return ()
        maximum = tuple(
            getattr(self, "_maximum_bond_dims", self._bond_dims())
        )
        if len(maximum) != len(self.dims) + 1:
            raise ValueError("the adaptive bond cap has the wrong number of cuts.")
        rtol = float(rtol)
        growth = int(growth)
        if not np.isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative.")
        if growth < 2:
            raise ValueError("growth must be at least two.")
        if isinstance(self._norm_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "adaptive bonds require exact dense frontier Grams."
            )

        suppressed = set(getattr(self, "_null_reduced_cuts", ()))
        candidates = tuple(
            cut
            for cut in range(1, len(self.dims))
            if self._bond_dims()[cut] < maximum[cut]
            and cut not in suppressed
        )
        if not candidates:
            return ()
        left_messages = self._norm_frontier.build_left(self.tensors)
        right_messages = self._norm_frontier.build_right(self.tensors)
        grow = []
        for cut in candidates:
            current = self._bond_dims()[cut]
            left, right = self.frontier_bond_grams(
                cut,
                left_messages=left_messages,
                right_messages=right_messages,
            )

            def numerical_rank(matrix):
                values = np.linalg.eigvalsh(self._hermitian_part(matrix))
                scale_value = max(float(np.max(values, initial=0.0)), 1.0)
                threshold = max(
                    rtol * scale_value,
                    256.0 * np.finfo(float).eps * scale_value,
                )
                return int(np.count_nonzero(values > threshold))

            if min(numerical_rank(left), numerical_rank(right)) == current:
                grow.append((cut, min(maximum[cut], growth * current)))

        records = []
        for cut, dimension in grow:
            records.append(
                self.expand_bond(
                    cut,
                    dimension,
                    direction=direction,
                    strategy=strategy,
                    scale=scale,
                )
            )
        return tuple(records)

    @classmethod
    def from_dense(cls, state, hamiltonian: Hamiltonian, **kwargs):
        """Copy tensors from a dense-projector reference state."""
        result = cls(
            hamiltonian,
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
            self.parent_sets,
            bond_dim=self.bond_dim,
            bond_dims=self.bond_dims,
            tensors=[tensor.copy() for tensor in self.tensors],
            frontier_backend=self.frontier_backend,
            chunk_size=self.chunk_size,
            chunk_memory=self.chunk_memory,
            chunk_span=self.chunk_span,
            workers=self.workers,
            path_optimizer=self.path_optimizer,
            max_rank=self.tt_options["max_rank"],
            rtol=self.tt_options["rtol"],
            atol=self.tt_options["atol"],
            transfer_max_rank=self.tt_options["transfer_max_rank"],
            transfer_rtol=self.tt_options["transfer_rtol"],
            transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
            tt_channels=self.tt_channels,
            tt_gauge=self.tt_gauge,
            compute_dtype=self.compute_dtype,
            device=self.device,
            route_memory=self.route_memory,
            action_memory=self.action_memory,
        )
        # Construction balances tensor magnitudes.  Restore the represented
        # state exactly, then recompute the approximate energy consistently.
        result.tensors = [tensor.copy() for tensor in self.tensors]
        result.history = list(self.history)
        result.energy = result.expectation()
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return self._copy_public_settings_to(result)

    def close(self):
        """Release bounded worker pools owned by this state."""

        frontier = getattr(self, "_hamiltonian_frontier", None)
        close_frontier = getattr(frontier, "close", None)
        if close_frontier is not None:
            close_frontier()
        executor = getattr(self, "_solver_executor", None)
        self._solver_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __del__(self):
        executor = getattr(self, "_solver_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    def _rebuild_frontier_engines(self):
        """Replan contractions after a virtual-bond shape change."""
        shapes = tuple(tuple(tensor.shape) for tensor in self.tensors)
        bonds = self._bond_dims()
        for site, shape in enumerate(shapes):
            if shape[:2] != (bonds[site], bonds[site + 1]):
                raise ValueError(
                    f"tensor {site} virtual shape is inconsistent with bond_dims."
                )
        old_hamiltonian_frontier = getattr(self, "_hamiltonian_frontier", None)
        close_frontier = getattr(old_hamiltonian_frontier, "close", None)
        if close_frontier is not None:
            close_frontier()
        parameter_dtype = np.result_type(
            self.hamiltonian.dtype,
            *[tensor.dtype for tensor in self.tensors],
        )
        identity_mpo = MPO(
            [np.eye(dim, dtype=parameter_dtype)[None, None, :, :] for dim in self.dims],
            sites=self.sites,
        )
        frontier_arguments = (self.dims, self.physical_groups, shapes)
        if self.tt_norm_backend == "tensor_train":
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
                local_qns=getattr(
                    getattr(self, "abelian_layout", None), "local_qns", None
                ),
                bond_qns=getattr(
                    getattr(self, "abelian_layout", None), "bond_qns", None
                ),
                compute_dtype=self.compute_dtype,
                device=self.device,
                route_memory=self.route_memory,
                action_memory=self.action_memory,
            )
        elif self.frontier_backend == "termwise":
            self._hamiltonian_frontier = TermwiseBlockMPOFrontier(
                self.hamiltonian,
                self.physical_groups,
                shapes,
                optimize=self.path_optimizer,
                local_qns=getattr(
                    getattr(self, "abelian_layout", None), "local_qns", None
                ),
                bond_qns=getattr(
                    getattr(self, "abelian_layout", None), "bond_qns", None
                ),
                chunk_size=self.chunk_size,
                chunk_memory=self.chunk_memory,
                chunk_span=self.chunk_span,
                workers=self.workers,
                compute_dtype=self.compute_dtype,
                device=self.device,
            )
        else:
            self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                self.hamiltonian,
                self.physical_groups,
                shapes,
                optimize=self.path_optimizer,
                local_qns=getattr(
                    getattr(self, "abelian_layout", None),
                    "local_qns",
                    None,
                ),
                channel_grouping=self.tt_channels,
                **self.tt_options,
            )
        self._pair_plan_cache = {}
        self._block_plan_cache = {}
        self._davidson_recycle = {}

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
        _directions=None,
        _source_norm=None,
        _evaluate=True,
        _reset_history=True,
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
        if strategy not in {"residual", "random", "zero", "amen", "amen_raw"}:
            raise ValueError(
                "strategy must be 'residual', 'random', 'zero', 'amen', "
                "or 'amen_raw'."
            )
        if strategy in {"amen", "amen_raw"} and _directions is None:
            raise ValueError(
                f"strategy='{strategy}' requires streamed residual directions."
            )
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("scale must be finite and nonnegative.")

        _evaluate = bool(_evaluate)
        _reset_history = bool(_reset_history)
        energy_before = (
            self.expectation()
            if _evaluate or self.energy is None
            else float(self.energy)
        )
        norm_before = (
            float(np.real(self._norm_frontier.scalar(self.tensors)))
            if _evaluate
            else None
        )
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
            if _directions is not None:
                directions = np.asarray(
                    _directions,
                    dtype=np.result_type(left_tensor.dtype, _directions),
                )
                if (
                    directions.ndim != 2
                    or directions.shape[0] != old_matrix.shape[0]
                    or directions.shape[1] > added
                ):
                    raise ValueError(
                        "right-going enrichment directions have an invalid shape."
                    )
                directions = self._prepare_amen_directions(
                    cut,
                    direction,
                    directions,
                )
                source_matrix = directions
            elif strategy == "residual":
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
            source_norm = (
                float(_source_norm)
                if _source_norm is not None
                else float(np.linalg.norm(source_matrix))
            )
            if _directions is None:
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
            if strategy == "amen":
                augmented = np.concatenate(
                    (old_matrix, amplitude * directions),
                    axis=1,
                )
                expanded_matrix, center = np.linalg.qr(augmented, mode="reduced")
                expanded_ordered = expanded_matrix.reshape(ordered_shape)
                expanded_left = expanded_ordered.transpose(inverse_axes)
                right_matrix = right_tensor.reshape(old_dimension, -1)
                expanded_right = (
                    center[:, :old_dimension] @ right_matrix
                ).reshape((new_dimension,) + right_tensor.shape[1:])
            else:
                expanded_ordered = np.zeros(
                    ordered_shape,
                    dtype=np.result_type(left_tensor.dtype, directions.dtype),
                )
                expanded_matrix = expanded_ordered.reshape(-1, new_dimension)
                expanded_matrix[:, :old_dimension] = old_matrix
                expanded_matrix[
                    :, old_dimension : old_dimension + directions.shape[1]
                ] = amplitude * directions
                expanded_left = expanded_ordered.transpose(inverse_axes)
                expanded_right = np.zeros(
                    (new_dimension,) + right_tensor.shape[1:],
                    dtype=right_tensor.dtype,
                )
                expanded_right[:old_dimension] = right_tensor
        else:
            old_matrix = right_tensor.reshape(old_dimension, -1)
            if _directions is not None:
                directions = np.asarray(
                    _directions,
                    dtype=np.result_type(right_tensor.dtype, _directions),
                )
                if (
                    directions.ndim != 2
                    or directions.shape[1] != old_matrix.shape[1]
                    or directions.shape[0] > added
                ):
                    raise ValueError(
                        "left-going enrichment directions have an invalid shape."
                    )
                directions = self._prepare_amen_directions(
                    cut,
                    direction,
                    directions,
                )
                source_matrix = directions
            elif strategy == "residual":
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
            source_norm = (
                float(_source_norm)
                if _source_norm is not None
                else float(np.linalg.norm(source_matrix))
            )
            if _directions is None:
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
            if strategy == "amen":
                augmented = np.concatenate(
                    (old_matrix, amplitude * directions),
                    axis=0,
                )
                right_basis, center = np.linalg.qr(augmented.T, mode="reduced")
                expanded_right = right_basis.T.reshape(
                    (new_dimension,) + right_tensor.shape[1:]
                )
                axes = (0, *range(2, left_tensor.ndim), 1)
                inverse_axes = np.argsort(axes)
                left_ordered = left_tensor.transpose(axes)
                left_matrix = left_ordered.reshape(-1, old_dimension)
                expanded_left_ordered = (
                    left_matrix @ center.T[:old_dimension]
                ).reshape(left_ordered.shape[:-1] + (new_dimension,))
                expanded_left = expanded_left_ordered.transpose(inverse_axes)
            else:
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
        if _evaluate:
            norm_after = float(np.real(self._norm_frontier.scalar(self.tensors)))
            energy_after = self.expectation()
            norm_error = abs(norm_after - norm_before) / max(abs(norm_before), 1.0)
        else:
            energy_after = energy_before
            norm_error = 0.0
        self.energy = energy_after
        if _reset_history:
            self.history = []
        self.converged = False
        suppressed = set(getattr(self, "_null_reduced_cuts", ()))
        suppressed.discard(cut)
        self._null_reduced_cuts = suppressed
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

    def _pair_plan_fingerprint(self):
        return (
            self.physical_groups,
            tuple(tuple(tensor.shape) for tensor in self.tensors),
            self.frontier_backend,
            tuple(
                tuple(tensor.shape)
                for tensor in getattr(self._norm_frontier, "mpo_tensors", ())
            ),
            tuple(
                tuple(tensor.shape)
                for tensor in getattr(
                    self._hamiltonian_frontier,
                    "mpo_tensors",
                    (),
                )
            ),
        )

    def _pair_plan(self, site) -> _FrontierPairPlan:
        """Return a cached value-independent topology for an adjacent pair."""
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        if not isinstance(self._norm_frontier, MPOFrontier) or not isinstance(
            self._hamiltonian_frontier,
            (MPOFrontier, BlockMPOFrontier),
        ):
            raise NotImplementedError(
                "cached two-site environments currently require exact "
                "compressed or identity-block frontier engines."
            )
        fingerprint = self._pair_plan_fingerprint()
        cached = self._pair_plan_cache.get(site)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached

        following = site + 1
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
        union_sites = (site,) + tuple(
            sorted((set(left_sites) | set(right_sites)) - {site})
        )
        merged_shape = (
            self.tensors[site].shape[0],
            self.tensors[following].shape[1],
            *(self.dims[index] for index in union_sites),
        )
        right_dimension = self._bond_dims()[following + 1]
        dtype = np.result_type(
            self.tensors[site].dtype,
            self.tensors[following].dtype,
        )
        identity = np.eye(right_dimension, dtype=dtype)
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
        pair_physical_sites = list(self.physical_groups)
        pair_physical_sites[site] = union_sites
        pair_shapes = [tuple(tensor.shape) for tensor in self.tensors]
        pair_shapes[site] = merged_shape
        pair_shapes[following] = tuple(identity_tensor.shape)
        arguments = (
            self.dims,
            tuple(pair_physical_sites),
            tuple(pair_shapes),
        )
        norm_engine = MPOFrontier(
            *arguments,
            self._norm_frontier.mpo_tensors,
            paired_sites=(),
            optimize=self.path_optimizer,
        )
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and getattr(self.hamiltonian, "block_physical_dims", None) is None
        ):
            hamiltonian_engine = BlockMPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
                compute_dtype=self.compute_dtype,
                device=self.device,
                route_memory=0,
                action_memory=self.action_memory,
            )
        elif isinstance(self._hamiltonian_frontier, BlockMPOFrontier):
            hamiltonian_engine = MPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
                physical_factor_dims=self.hamiltonian.block_physical_dims,
            )
        else:
            hamiltonian_engine = MPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
            )
        plan = _FrontierPairPlan(
            site=site,
            union_sites=union_sites,
            merged_shape=merged_shape,
            identity_tensor=identity_tensor,
            norm_engine=norm_engine,
            hamiltonian_engine=hamiltonian_engine,
            fingerprint=fingerprint,
        )
        self._pair_plan_cache[site] = plan
        return plan

    def _block_plan(self, start, stop) -> _FrontierBlockPlan:
        """Return a cached topology for one contiguous merged block."""
        start = int(start)
        stop = int(stop)
        if start < 0 or stop > len(self.dims) or stop - start < 2:
            raise ValueError("block must contain at least two valid consecutive sites.")
        if not isinstance(self._norm_frontier, MPOFrontier) or not isinstance(
            self._hamiltonian_frontier,
            (MPOFrontier, BlockMPOFrontier),
        ):
            raise NotImplementedError(
                "cached block environments currently require exact compressed "
                "or identity-block frontier engines."
            )
        key = (start, stop)
        fingerprint = self._pair_plan_fingerprint()
        cached = self._block_plan_cache.get(key)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached

        sites = tuple(range(start, stop))
        union_sites = (start,) + tuple(
            sorted(
                {
                    physical_site
                    for site in sites
                    for physical_site in self.physical_groups[site]
                }
                - {start}
            )
        )
        right_dimension = self._bond_dims()[stop]
        merged_shape = (
            self.tensors[start].shape[0],
            right_dimension,
            *(self.dims[index] for index in union_sites),
        )
        dtype = np.result_type(*(self.tensors[site].dtype for site in sites))
        identity_tensors = []
        for site in sites[1:]:
            physical_sites = self.physical_groups[site]
            identity = np.eye(right_dimension, dtype=dtype)
            identity_tensors.append(
                np.broadcast_to(
                    identity.reshape(
                        right_dimension,
                        right_dimension,
                        *((1,) * len(physical_sites)),
                    ),
                    (
                        right_dimension,
                        right_dimension,
                        *(self.dims[index] for index in physical_sites),
                    ),
                ).copy()
            )

        block_physical_sites = list(self.physical_groups)
        block_physical_sites[start] = union_sites
        block_shapes = [tuple(tensor.shape) for tensor in self.tensors]
        block_shapes[start] = merged_shape
        for site, identity_tensor in zip(sites[1:], identity_tensors):
            block_shapes[site] = tuple(identity_tensor.shape)
        arguments = (
            self.dims,
            tuple(block_physical_sites),
            tuple(block_shapes),
        )
        norm_engine = MPOFrontier(
            *arguments,
            self._norm_frontier.mpo_tensors,
            paired_sites=(),
            optimize=self.path_optimizer,
        )
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and getattr(self.hamiltonian, "block_physical_dims", None) is None
        ):
            hamiltonian_engine = BlockMPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
                compute_dtype=self.compute_dtype,
                device=self.device,
            )
        elif isinstance(self._hamiltonian_frontier, BlockMPOFrontier):
            hamiltonian_engine = MPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
                physical_factor_dims=self.hamiltonian.block_physical_dims,
            )
        else:
            hamiltonian_engine = MPOFrontier(
                *arguments,
                self._hamiltonian_frontier.mpo_tensors,
                optimize=self.path_optimizer,
            )
        plan = _FrontierBlockPlan(
            sites=sites,
            union_sites=union_sites,
            merged_shape=merged_shape,
            identity_tensors=tuple(identity_tensors),
            norm_engine=norm_engine,
            hamiltonian_engine=hamiltonian_engine,
            fingerprint=fingerprint,
        )
        self._block_plan_cache[key] = plan
        return plan

    def _block_environment_from_outer_messages(
        self,
        start,
        stop,
        norm_left,
        norm_outer_right,
        hamiltonian_left,
        hamiltonian_outer_right,
        *,
        action_only=False,
    ) -> FrontierBlockEnvironment:
        """Absorb a block's fixed identity carriers into outer messages."""
        plan = self._block_plan(start, stop)
        raw_hamiltonian_left = hamiltonian_left
        raw_hamiltonian_right = hamiltonian_outer_right
        block_tensors = list(self.tensors)
        for site, identity_tensor in zip(
            plan.sites[1:],
            plan.identity_tensors,
        ):
            block_tensors[site] = identity_tensor
        block_engine = isinstance(plan.hamiltonian_engine, BlockMPOFrontier)
        termwise_block_action = bool(
            action_only
            and isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
        )
        if (
            not termwise_block_action
            and block_engine
            and self._hamiltonian_frontier.charge_resolved
        ):
            if isinstance(hamiltonian_outer_right, BlockFrontierMessage):
                hamiltonian_outer_right = (
                    self._hamiltonian_frontier.uncharge_message(
                        hamiltonian_outer_right
                    )
                )
            if isinstance(hamiltonian_left, BlockFrontierMessage):
                hamiltonian_left = self._hamiltonian_frontier.uncharge_message(
                    hamiltonian_left
                )
        if (
            not termwise_block_action
            and isinstance(hamiltonian_outer_right, BlockFrontierMessage)
            and not block_engine
        ):
            hamiltonian_outer_right = (
                self._hamiltonian_frontier.dense_message(
                    hamiltonian_outer_right
                )
            )
        if (
            not termwise_block_action
            and isinstance(hamiltonian_left, BlockFrontierMessage)
            and not block_engine
        ):
            hamiltonian_left = self._hamiltonian_frontier.dense_message(
                hamiltonian_left
            )

        norm_right = norm_outer_right
        hamiltonian_right = hamiltonian_outer_right
        for site in reversed(plan.sites[1:]):
            norm_right = plan.norm_engine.advance_right(
                norm_right,
                block_tensors,
                site,
            )
            if not termwise_block_action:
                hamiltonian_right = plan.hamiltonian_engine.advance_right(
                    hamiltonian_right,
                    block_tensors,
                    site,
                )
        if termwise_block_action:
            hamiltonian_left = None
            hamiltonian_right = None
        return FrontierBlockEnvironment(
            sites=plan.sites,
            union_sites=plan.union_sites,
            norm_left=norm_left,
            norm_right=norm_right,
            hamiltonian_left=hamiltonian_left,
            hamiltonian_right=hamiltonian_right,
            hamiltonian_outer_left=raw_hamiltonian_left,
            hamiltonian_outer_right=raw_hamiltonian_right,
        )

    def block_environment(self, start, stop) -> FrontierBlockEnvironment:
        """Build exact outer messages for one contiguous merged block."""
        plan = self._block_plan(start, stop)
        norm_left = self._norm_frontier.build_left(self.tensors)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_left = self._hamiltonian_frontier.build_left(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
        return self._block_environment_from_outer_messages(
            start,
            stop,
            norm_left[start],
            norm_right[stop],
            hamiltonian_left[start],
            hamiltonian_right[stop],
        )

    def _resolved_block_environment(self, start, stop, environment):
        if environment is None:
            return self.block_environment(start, stop)
        if not isinstance(environment, FrontierBlockEnvironment):
            raise TypeError("environment must be a FrontierBlockEnvironment.")
        if environment.sites != tuple(range(int(start), int(stop))):
            raise ValueError("environment belongs to a different block.")
        return environment

    def block_local_operators(self, start, stop, *, environment=None):
        """Return exact merged-block ``(N_eff, H_eff)`` matrices."""
        plan = self._block_plan(start, stop)
        environment = self._resolved_block_environment(start, stop, environment)
        metric = plan.norm_engine.hole_matrix(
            int(start),
            environment.norm_left,
            environment.norm_right,
        )
        effective = plan.hamiltonian_engine.hole_matrix(
            int(start),
            environment.hamiltonian_left,
            environment.hamiltonian_right,
        )
        return self._hermitian_part(metric), self._hermitian_part(effective)

    def block_metric_action(self, start, stop, vector, *, environment=None):
        """Apply a merged-block norm operator without materializing it."""
        plan = self._block_plan(start, stop)
        environment = self._resolved_block_environment(start, stop, environment)
        vector = np.asarray(vector)
        if vector.size != int(np.prod(plan.merged_shape)):
            raise ValueError("merged-block vector has the wrong size.")
        return plan.norm_engine.hole_action(
            int(start),
            environment.norm_left,
            environment.norm_right,
            vector,
        )

    def block_hamiltonian_action(
        self,
        start,
        stop,
        vector,
        *,
        environment=None,
    ):
        """Apply a merged-block Hamiltonian from cached outer messages."""
        plan = self._block_plan(start, stop)
        environment = self._resolved_block_environment(start, stop, environment)
        vector = np.asarray(vector)
        if vector.size != int(np.prod(plan.merged_shape)):
            raise ValueError("merged-block vector has the wrong size.")
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and environment.hamiltonian_outer_left is not None
            and environment.hamiltonian_outer_right is not None
        ):
            return self._hamiltonian_frontier.block_hole_action(
                start,
                stop,
                environment.hamiltonian_outer_left,
                environment.hamiltonian_outer_right,
                plan.union_sites,
                plan.merged_shape,
                vector,
            )
        return plan.hamiltonian_engine.hole_action(
            int(start),
            environment.hamiltonian_left,
            environment.hamiltonian_right,
            vector,
        )

    def block_local_action_problem(
        self,
        start,
        stop,
        *,
        environment=None,
    ) -> PhysicalBlockGeneralizedProblem:
        """Build metric blocks and a lazy exact merged-block Hamiltonian."""
        plan = self._block_plan(start, stop)
        environment = self._resolved_block_environment(start, stop, environment)
        layout = PhysicalBlockLayout(plan.merged_shape)
        pairs = hamiltonian_physical_connectivity(
            self.hamiltonian,
            plan.union_sites,
        )

        def metric_factory(rows, columns):
            return plan.norm_engine.hole_blocks(
                int(start),
                environment.norm_left,
                environment.norm_right,
                tuple(layout.configurations[row] for row in rows),
                tuple(layout.configurations[column] for column in columns),
            )

        def hamiltonian_action(vector):
            return self.block_hamiltonian_action(
                start,
                stop,
                vector,
                environment=environment,
            )

        return (
            PhysicalBlockGeneralizedProblem
            .from_batched_metric_factory_and_hamiltonian_action(
                plan.merged_shape,
                pairs,
                metric_factory,
                hamiltonian_action,
                dtype=np.result_type(
                    *(self.tensors[site].dtype for site in plan.sites),
                    self.hamiltonian.dtype,
                ),
            )
        )

    def _pair_environment_from_outer_messages(
        self,
        site,
        norm_left,
        norm_outer_right,
        hamiltonian_left,
        hamiltonian_outer_right,
    ) -> FrontierPairEnvironment:
        """Absorb the fixed right identity into cached outer messages."""
        plan = self._pair_plan(site)
        following = int(site) + 1
        pair_tensors = list(self.tensors)
        pair_tensors[following] = plan.identity_tensor
        block_pair_engine = isinstance(
            plan.hamiltonian_engine,
            BlockMPOFrontier,
        )
        if (
            block_pair_engine
            and not plan.hamiltonian_engine.charge_resolved
            and self._hamiltonian_frontier.charge_resolved
        ):
            if isinstance(hamiltonian_outer_right, BlockFrontierMessage):
                hamiltonian_outer_right = (
                    self._hamiltonian_frontier.uncharge_message(
                        hamiltonian_outer_right
                    )
                )
            if isinstance(hamiltonian_left, BlockFrontierMessage):
                hamiltonian_left = (
                    self._hamiltonian_frontier.uncharge_message(
                        hamiltonian_left
                    )
                )
        if (
            isinstance(hamiltonian_outer_right, BlockFrontierMessage)
            and not block_pair_engine
        ):
            hamiltonian_outer_right = (
                self._hamiltonian_frontier.dense_message(
                    hamiltonian_outer_right
                )
            )
        if (
            isinstance(hamiltonian_left, BlockFrontierMessage)
            and not block_pair_engine
        ):
            hamiltonian_left = self._hamiltonian_frontier.dense_message(
                hamiltonian_left
            )
        norm_right = plan.norm_engine.advance_right(
            norm_outer_right,
            pair_tensors,
            following,
        )
        hamiltonian_right = plan.hamiltonian_engine.advance_right(
            hamiltonian_outer_right,
            pair_tensors,
            following,
        )
        return FrontierPairEnvironment(
            sites=(int(site), following),
            union_sites=plan.union_sites,
            norm_left=norm_left,
            norm_right=norm_right,
            hamiltonian_left=hamiltonian_left,
            hamiltonian_right=hamiltonian_right,
        )

    def pair_environment(self, site: int) -> FrontierPairEnvironment:
        """Build exact outer messages for one adjacent merged pair."""
        site = int(site)
        self._pair_plan(site)
        following = site + 1
        norm_left = self._norm_frontier.build_left(self.tensors)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_left = self._hamiltonian_frontier.build_left(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
        return self._pair_environment_from_outer_messages(
            site,
            norm_left[site],
            norm_right[following + 1],
            hamiltonian_left[site],
            hamiltonian_right[following + 1],
        )

    def _resolved_pair_environment(self, site, environment):
        if environment is None:
            return self.pair_environment(site)
        if not isinstance(environment, FrontierPairEnvironment):
            raise TypeError("environment must be a FrontierPairEnvironment.")
        if environment.sites != (int(site), int(site) + 1):
            raise ValueError("environment belongs to a different adjacent pair.")
        return environment

    def pair_local_operators(
        self,
        site: int,
        *,
        environment=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact merged-pair ``(N_eff, H_eff)`` from cached plans."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        metric = plan.norm_engine.hole_matrix(
            site,
            environment.norm_left,
            environment.norm_right,
        )
        effective = plan.hamiltonian_engine.hole_matrix(
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
        )
        return self._hermitian_part(metric), self._hermitian_part(effective)

    def pair_metric_action(self, site: int, vector, *, environment=None):
        """Apply the merged-pair norm operator without materializing it."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        vector = np.asarray(vector)
        if vector.size != int(np.prod(plan.merged_shape)):
            raise ValueError("merged-pair vector has the wrong size.")
        return plan.norm_engine.hole_action(
            site,
            environment.norm_left,
            environment.norm_right,
            vector,
        )

    def pair_hamiltonian_action(self, site: int, vector, *, environment=None):
        """Apply the merged-pair Hamiltonian from cached outer messages."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        vector = np.asarray(vector)
        if vector.size != int(np.prod(plan.merged_shape)):
            raise ValueError("merged-pair vector has the wrong size.")
        return plan.hamiltonian_engine.hole_action(
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
            vector,
        )

    def pair_local_block_problem(
        self,
        site: int,
        *,
        environment=None,
    ) -> PhysicalBlockGeneralizedProblem:
        """Build the exact conditional physical-block merged-pair pencil."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        layout = PhysicalBlockLayout(plan.merged_shape)
        pairs = hamiltonian_physical_connectivity(
            self.hamiltonian,
            plan.union_sites,
        )
        parameter_mask = self._pair_action_mask(site, plan)
        block_masks = (
            None
            if parameter_mask is None
            else layout.as_blocks(
                np.asarray(parameter_mask, dtype=bool).reshape(-1)
            )
        )

        def masked(blocks, rows, columns):
            if block_masks is None:
                return blocks
            blocks = np.asarray(blocks).copy()
            for offset, (row, column) in enumerate(zip(rows, columns)):
                blocks[offset] *= (
                    block_masks[row][:, None]
                    * block_masks[column][None, :]
                )
            return blocks

        def metric_factory(rows, columns):
            return masked(
                plan.norm_engine.hole_blocks(
                    site,
                    environment.norm_left,
                    environment.norm_right,
                    tuple(layout.configurations[row] for row in rows),
                    tuple(layout.configurations[column] for column in columns),
                ),
                rows,
                columns,
            )

        def hamiltonian_factory(rows, columns):
            return masked(
                plan.hamiltonian_engine.hole_blocks(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    tuple(layout.configurations[row] for row in rows),
                    tuple(layout.configurations[column] for column in columns),
                ),
                rows,
                columns,
            )

        return PhysicalBlockGeneralizedProblem.from_batched_block_factories(
            plan.merged_shape,
            pairs,
            metric_factory,
            hamiltonian_factory,
            dtype=np.result_type(
                self.tensors[site].dtype,
                self.tensors[site + 1].dtype,
                self.hamiltonian.dtype,
            ),
        )

    def pair_local_action_block_problem(
        self,
        site: int,
        *,
        environment=None,
    ) -> PhysicalBlockGeneralizedProblem:
        """Build metric blocks with a lazy exact merged-pair Hamiltonian."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        layout = PhysicalBlockLayout(plan.merged_shape)
        pairs = hamiltonian_physical_connectivity(
            self.hamiltonian,
            plan.union_sites,
        )
        parameter_mask = self._pair_action_mask(site, plan)
        if parameter_mask is None:
            flat_mask = None
            block_masks = None
        else:
            flat_mask = np.asarray(parameter_mask, dtype=bool).reshape(-1)
            if flat_mask.size != layout.size:
                raise ValueError("the merged-pair action mask has the wrong size.")
            block_masks = layout.as_blocks(flat_mask)

        prepare_action = getattr(
            plan.hamiltonian_engine,
            "prepare_hole_action",
            None,
        )
        prepared_action = (
            prepare_action(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
            )
            if prepare_action is not None
            else None
        )

        def metric_factory(rows, columns):
            blocks = plan.norm_engine.hole_blocks(
                site,
                environment.norm_left,
                environment.norm_right,
                tuple(layout.configurations[row] for row in rows),
                tuple(layout.configurations[column] for column in columns),
            )
            if block_masks is not None:
                blocks = np.asarray(blocks).copy()
                for offset, (row, column) in enumerate(zip(rows, columns)):
                    blocks[offset] *= (
                        block_masks[row][:, None]
                        * block_masks[column][None, :]
                    )
            return blocks

        def hamiltonian_action(vector):
            trial = np.asarray(vector)
            if flat_mask is not None:
                trial = np.where(flat_mask, trial, 0)
            result = (
                prepared_action(trial)
                if prepared_action is not None
                else plan.hamiltonian_engine.hole_action(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    trial,
                )
            )
            return result if flat_mask is None else np.where(flat_mask, result, 0)

        def hamiltonian_actions(vectors):
            trials = np.asarray(vectors)
            if flat_mask is not None:
                trials = np.where(flat_mask[None, :], trials, 0)
            prepared_many = getattr(prepared_action, "many", None)
            if prepared_many is not None:
                result = prepared_many(trials)
            else:
                result = np.stack(
                    [hamiltonian_action(trial) for trial in trials]
                )
            return (
                result
                if flat_mask is None
                else np.where(flat_mask[None, :], result, 0)
            )

        hamiltonian_action.many = hamiltonian_actions
        prepared_verify = getattr(prepared_action, "verify", None)
        if prepared_verify is not None:
            def hamiltonian_verify(vector):
                trial = np.asarray(vector)
                if flat_mask is not None:
                    trial = np.where(flat_mask, trial, 0)
                result = prepared_verify(trial)
                return (
                    result
                    if flat_mask is None
                    else np.where(flat_mask, result, 0)
                )

            hamiltonian_action.verify = hamiltonian_verify

        return (
            PhysicalBlockGeneralizedProblem
            .from_batched_metric_factory_and_hamiltonian_action(
                plan.merged_shape,
                pairs,
                metric_factory,
                hamiltonian_action,
                dtype=np.result_type(
                    self.tensors[site].dtype,
                    self.tensors[site + 1].dtype,
                    self.hamiltonian.dtype,
                ),
            )
        )

    def _pair_action_mask(self, site, plan):
        """Optional structural support for a merged adjacent-pair solve."""
        return None

    def pair_local_packed_action_block_problem(self, site, *, environment=None):
        """Fallback packed-pair entry point for unrestricted states."""
        return self.pair_local_action_block_problem(
            site,
            environment=environment,
        )

    def _merged_pair_tensor(self, site):
        """Contract an adjacent pair while retaining each physical label once."""
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        following = site + 1
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
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

    def _merged_block_tensor(self, start, stop):
        """Contract a contiguous block while retaining each physical label once."""
        plan = self._block_plan(start, stop)
        merged = np.empty(
            plan.merged_shape,
            dtype=np.result_type(
                *(self.tensors[site].dtype for site in plan.sites)
            ),
        )
        physical_shape = tuple(
            self.dims[index] for index in plan.union_sites
        )
        for configuration in np.ndindex(*physical_shape):
            values = dict(zip(plan.union_sites, configuration))
            product = None
            for site in plan.sites:
                tensor = self.tensors[site][
                    (
                        slice(None),
                        slice(None),
                        *(values[index] for index in self.physical_groups[site]),
                    )
                ]
                product = tensor if product is None else product @ tensor
            merged[(slice(None), slice(None), *configuration)] = product
        return merged, plan.union_sites

    def _split_merged_pair_tensor(
        self,
        site,
        merged,
        union_sites,
        *,
        middle_dimension=None,
        middle_labels=None,
    ):
        r"""Conditionally SVD a merged pair back into its graph-leg pattern.

        If the two local tensors share tied physical labels, the split is one
        independent matrix factorization for every shared-label assignment.
        The usual MPS SVD is recovered when the physical-label sets are
        disjoint.
        """
        site = int(site)
        following = site + 1
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
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
        if middle_labels is not None:
            raise TypeError("dense frontier bonds do not carry charge labels.")
        middle_dimension = (
            self._bond_dims()[following]
            if middle_dimension is None
            else int(middle_dimension)
        )
        if middle_dimension < 1:
            raise ValueError("middle_dimension must be positive.")
        left_result = np.zeros(
            (left_dimension, middle_dimension, *self.tensors[site].shape[2:]),
            dtype=merged.dtype,
        )
        right_result = np.zeros(
            (middle_dimension, right_dimension, *self.tensors[following].shape[2:]),
            dtype=merged.dtype,
        )
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
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
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
        site = int(site)
        following = site + 1
        union_sites = tuple(int(index) for index in union_sites)
        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        if left_tensor.shape[1] != right_tensor.shape[0]:
            raise ValueError("the pair factors have inconsistent middle dimensions.")
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
        expected_union = (site,) + tuple(
            sorted((set(left_sites) | set(right_sites)) - {site})
        )
        if union_sites != expected_union:
            raise ValueError("union_sites are inconsistent with the adjacent pair.")
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        merged = np.empty(
            merged_shape,
            dtype=np.result_type(left_tensor.dtype, right_tensor.dtype),
        )
        physical_shape = tuple(self.dims[index] for index in union_sites)
        for configuration in np.ndindex(*physical_shape):
            values = dict(zip(union_sites, configuration))
            left_configuration = tuple(values[index] for index in left_sites)
            right_configuration = tuple(values[index] for index in right_sites)
            left = left_tensor[
                (slice(None), slice(None), *left_configuration)
            ]
            right = right_tensor[
                (slice(None), slice(None), *right_configuration)
            ]
            merged[(slice(None), slice(None), *configuration)] = left @ right
        return merged

    def _pair_factor_action(
        self,
        site,
        union_sites,
        left_tensor,
        right_tensor,
        variation,
        *,
        variable,
    ):
        """Apply ``J_A`` or ``J_B`` without constructing a design matrix."""
        variable = str(variable).lower()
        if variable == "left":
            varied_left = np.asarray(variation).reshape(left_tensor.shape)
            varied_right = right_tensor
        elif variable == "right":
            varied_left = left_tensor
            varied_right = np.asarray(variation).reshape(right_tensor.shape)
        else:
            raise ValueError("variable must be 'left' or 'right'.")
        return self._merge_pair_factors(
            site,
            union_sites,
            varied_left,
            varied_right,
        ).reshape(-1)

    def _pair_factor_adjoint(
        self,
        site,
        union_sites,
        left_tensor,
        right_tensor,
        cotangent,
        *,
        variable,
    ):
        """Apply the exact Euclidean adjoint of ``J_A`` or ``J_B``."""
        site = int(site)
        following = site + 1
        union_sites = tuple(int(index) for index in union_sites)
        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        cotangent = np.asarray(cotangent)
        if cotangent.size != int(np.prod(merged_shape)):
            raise ValueError("cotangent has the wrong merged-pair size.")
        cotangent = cotangent.reshape(merged_shape)
        left_sites = self.physical_groups[site]
        right_sites = self.physical_groups[following]
        physical_shape = tuple(self.dims[index] for index in union_sites)
        variable = str(variable).lower()
        if variable == "left":
            result = np.zeros_like(
                left_tensor,
                dtype=np.result_type(
                    left_tensor.dtype,
                    right_tensor.dtype,
                    cotangent.dtype,
                ),
            )
        elif variable == "right":
            result = np.zeros_like(
                right_tensor,
                dtype=np.result_type(
                    left_tensor.dtype,
                    right_tensor.dtype,
                    cotangent.dtype,
                ),
            )
        else:
            raise ValueError("variable must be 'left' or 'right'.")
        for configuration in np.ndindex(*physical_shape):
            values = dict(zip(union_sites, configuration))
            left_configuration = tuple(values[index] for index in left_sites)
            right_configuration = tuple(values[index] for index in right_sites)
            left = left_tensor[
                (slice(None), slice(None), *left_configuration)
            ]
            right = right_tensor[
                (slice(None), slice(None), *right_configuration)
            ]
            block = cotangent[(slice(None), slice(None), *configuration)]
            if variable == "left":
                result[
                    (slice(None), slice(None), *left_configuration)
                ] += block @ right.T.conj()
            else:
                result[
                    (slice(None), slice(None), *right_configuration)
                ] += left.T.conj() @ block
        return result.reshape(-1)

    @staticmethod
    def _pair_residual_verification(
        metric,
        effective,
        energy,
        vector,
        metric_values,
        metric_vectors,
        active,
        numerical_active=None,
    ):
        vector = np.asarray(vector).reshape(-1)
        metric_vector = metric @ vector
        hamiltonian_vector = effective @ vector
        residual = hamiltonian_vector - float(energy) * metric_vector
        vector_norm = max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
        backward_scale = max(
            (
                float(np.linalg.norm(effective, ord=np.inf))
                + abs(float(energy)) * float(np.linalg.norm(metric, ord=np.inf))
            )
            * vector_norm,
            np.finfo(float).tiny,
        )
        backward = float(np.linalg.norm(residual) / backward_scale)
        active_vectors = metric_vectors[:, active]
        active_values = metric_values[active]
        projected_residual = active_vectors.T.conj() @ residual
        projected_hamiltonian = active_vectors.T.conj() @ hamiltonian_vector
        dual = float(
            np.sqrt(
                np.sum(np.abs(projected_residual) ** 2 / active_values)
            )
        )
        metric_norm = max(
            float(np.real(np.vdot(vector, metric_vector))),
            np.finfo(float).tiny,
        )
        dual_scale = max(
            float(
                np.sqrt(
                    np.sum(
                        np.abs(projected_hamiltonian) ** 2 / active_values
                    )
                )
            ),
            abs(float(energy)) * np.sqrt(metric_norm),
            (
                float(np.linalg.norm(effective, ord=np.inf))
                * vector_norm
                / np.sqrt(float(np.max(active_values)))
            ),
            np.finfo(float).tiny,
        )
        dual_relative = dual / dual_scale
        if numerical_active is None:
            numerical_active = active
        numerical_active = np.asarray(numerical_active, dtype=bool)
        null_scale = max(
            float(np.linalg.norm(hamiltonian_vector)),
            abs(float(energy)) * float(np.linalg.norm(metric_vector)),
            np.finfo(float).tiny,
        )
        discarded = numerical_active & ~active
        discarded_residual = (
            float(
                np.linalg.norm(
                    metric_vectors[:, discarded].T.conj() @ residual
                )
                / null_scale
            )
            if np.any(discarded)
            else 0.0
        )
        true_null = ~numerical_active
        null_residual = (
            float(
                np.linalg.norm(
                    metric_vectors[:, true_null].T.conj() @ residual
                )
                / null_scale
            )
            if np.any(true_null)
            else 0.0
        )
        return {
            "raw": float(np.linalg.norm(residual)),
            "backward": backward,
            "dual": dual,
            "dual_relative": float(dual_relative),
            "null": null_residual,
            "discarded": discarded_residual,
        }

    def _solve_verified_pair_pencil(
        self,
        site,
        metric,
        effective,
        warm,
        *,
        metric_tol,
        eig_tol,
        maxiter,
        max_subspace,
        dense_fallback_dim,
        metric_support="regularized",
    ):
        """Warm-start, verify, and if needed certify a merged pair root."""
        metric_support = str(metric_support).lower().replace("-", "_")
        if metric_support not in {"regularized", "numerical"}:
            raise ValueError(
                "metric_support must be 'regularized' or 'numerical'."
            )
        metric = self._hermitian_part(metric)
        effective = self._hermitian_part(effective)
        warm = np.asarray(warm).reshape(-1)
        size = warm.size
        if metric.shape != (size, size) or effective.shape != (size, size):
            raise ValueError("merged pair operators have incompatible shapes.")
        metric_values, metric_vectors = linalg.eigh(
            metric,
            check_finite=False,
        )
        metric_scale = max(
            float(np.linalg.norm(metric, ord=np.inf)),
            float(np.max(np.abs(metric_values), initial=0.0)),
            np.finfo(float).tiny,
        )
        numerical_floor = 64.0 * np.finfo(float).eps * metric_scale
        numerical_active = metric_values > numerical_floor
        numerical_rank = int(np.count_nonzero(numerical_active))
        if numerical_rank == 0:
            raise ValueError("merged-pair overlap metric is numerically singular.")
        requested_floor = max(
            float(metric_tol) * metric_scale,
            numerical_floor,
        )
        requested_active = metric_values > requested_floor
        requested_rank = int(np.count_nonzero(requested_active))
        active = (
            numerical_active
            if metric_support == "numerical"
            else requested_active
        )
        support_rank = int(np.count_nonzero(active))
        if support_rank == 0:
            raise ValueError("requested merged-pair metric support is empty.")
        positive_values = metric_values[active]
        numerical_values = metric_values[numerical_active]
        minimum_positive = float(np.min(numerical_values))
        condition = float(np.max(numerical_values) / minimum_positive)

        warm_norm = np.vdot(warm, metric @ warm)
        if float(np.real(warm_norm)) <= np.finfo(float).tiny:
            raise ValueError("warm merged-pair state has zero metric norm.")
        warm = warm / np.sqrt(float(np.real(warm_norm)))
        warm_energy = self._pair_rayleigh(warm, metric, effective)
        verification_tolerance = max(
            10.0 * float(eig_tol),
            8.0 * np.sqrt(np.finfo(float).eps),
        )

        def residual_is_verified(check):
            return bool(
                check["dual_relative"] <= verification_tolerance
                and check["null"] <= verification_tolerance
            )

        attempts = []
        fallback_reason = ""
        dense_fallback = False
        dense_root_certified = False
        selected_energy = warm_energy
        selected_vector = warm.copy()
        selected_check = self._pair_residual_verification(
            metric,
            effective,
            warm_energy,
            warm,
            metric_values,
            metric_vectors,
            active,
            numerical_active,
        )
        selected_method = "warm"
        davidson_diagnostics = None
        energy_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(warm_energy),
        )

        davidson_maxiter = min(
            128,
            int(maxiter) if maxiter is not None else 128,
        )
        davidson_max_subspace = min(size, int(max_subspace), 64)
        try:
            attempts.append("warm_davidson")
            energy, vector, davidson_diagnostics = lowest_generalized_davidson(
                lambda trial: effective @ trial,
                lambda trial: metric @ trial,
                warm,
                tol=float(eig_tol),
                metric_tol=(
                    64.0 * np.finfo(float).eps
                    if metric_support == "numerical"
                    else max(float(metric_tol), 64.0 * np.finfo(float).eps)
                ),
                maxiter=davidson_maxiter,
                max_subspace=davidson_max_subspace,
                random_seed=int(site),
            )
            check = self._pair_residual_verification(
                metric,
                effective,
                energy,
                vector,
                metric_values,
                metric_vectors,
                active,
                numerical_active,
            )
            if (
                davidson_diagnostics.converged
                and residual_is_verified(check)
                and energy <= warm_energy
            ):
                selected_energy = float(energy)
                selected_vector = np.asarray(vector).reshape(-1)
                selected_check = check
                selected_method = "warm_davidson"
            else:
                fallback_reason = "warm Davidson failed full metric-dual verification"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            fallback_reason = f"warm Davidson failed: {error}"

        # A small residual certifies an eigenpair, not that it is the lowest
        # eigenpair: a one-vector Krylov space can remain inside an invariant
        # excited sector.  At manageable dimensions the dense solve therefore
        # certifies every result.  Above this threshold ``verified`` means
        # residual-verified only, while ``lowest_root_certified`` remains false.
        davidson_stagnated = bool(
            selected_method == "warm_davidson"
            and selected_energy >= warm_energy - energy_tolerance
        )
        if size <= int(dense_fallback_dim):
            attempts.append("dense_certification")
            dense_fallback = True
            if not fallback_reason:
                if (
                    metric_support == "numerical"
                    and requested_rank < numerical_rank
                ):
                    fallback_reason = (
                        "numerical-support mode retained directions below the "
                        "requested metric cutoff"
                    )
                elif davidson_stagnated:
                    fallback_reason = (
                        "verified Davidson root did not improve the warm upper "
                        "bound"
                    )
                else:
                    fallback_reason = (
                        "dense certification of the globally lowest root at "
                        "manageable dimension"
                    )
            try:
                if support_rank == size:
                    eigenvalues, eigenvectors = linalg.eigh(
                        effective,
                        metric,
                        subset_by_index=[0, 0],
                        driver="gvx",
                        check_finite=False,
                    )
                    energy = float(np.real(eigenvalues[0]))
                    vector = eigenvectors[:, 0]
                else:
                    basis = metric_vectors[:, active] / np.sqrt(
                        positive_values
                    )[None, :]
                    reduced = self._hermitian_part(
                        basis.T.conj() @ effective @ basis
                    )
                    eigenvalues, eigenvectors = linalg.eigh(
                        reduced,
                        subset_by_index=[0, 0],
                        driver="evx",
                        check_finite=False,
                    )
                    energy = float(np.real(eigenvalues[0]))
                    vector = basis @ eigenvectors[:, 0]
                vector = vector / np.sqrt(
                    float(np.real(np.vdot(vector, metric @ vector)))
                )
                check = self._pair_residual_verification(
                    metric,
                    effective,
                    energy,
                    vector,
                    metric_values,
                    metric_vectors,
                    active,
                    numerical_active,
                )
                if residual_is_verified(check):
                    dense_root_certified = True
                    if energy <= selected_energy:
                        selected_energy = energy
                        selected_vector = vector
                        selected_check = check
                        selected_method = "dense_certified"
                    else:
                        reason = (
                            "certified dense lowest root in the requested "
                            "metric support was above the retained "
                            "variational state"
                        )
                        fallback_reason = (
                            f"{fallback_reason}; {reason}"
                            if fallback_reason
                            else reason
                        )
                else:
                    reason = "dense root failed full residual verification"
                    fallback_reason = (
                        f"{fallback_reason}; {reason}"
                        if fallback_reason
                        else reason
                    )
            except (ValueError, linalg.LinAlgError, FloatingPointError) as error:
                if fallback_reason:
                    fallback_reason = f"{fallback_reason}; dense solve failed: {error}"
                else:
                    fallback_reason = f"dense solve failed: {error}"

        accepted = bool(selected_energy <= warm_energy + energy_tolerance)
        if not accepted:
            selected_energy = warm_energy
            selected_vector = warm
            selected_method = "warm"
            selected_check = self._pair_residual_verification(
                metric,
                effective,
                warm_energy,
                warm,
                metric_values,
                metric_vectors,
                active,
                numerical_active,
            )
        verified = residual_is_verified(selected_check)

        if selected_method == "warm":
            if verified and dense_root_certified:
                solver_message = (
                    "retained residual-verified warm pair below the certified "
                    "requested-support root"
                )
            elif verified:
                solver_message = "retained residual-verified warm pair"
            else:
                solver_message = (
                    "retained variational warm pair; its full residual is "
                    "unverified"
                )
        else:
            solver_message = (
                "verified merged-pair root"
                if verified
                else "merged-pair root failed full residual verification"
            )

        diagnostics = FrontierMergedSolveDiagnostics(
            method=selected_method,
            attempts=tuple(attempts),
            verified=verified,
            lowest_root_certified=dense_root_certified,
            fallback_reason=fallback_reason,
            dense_fallback=dense_fallback,
            metric_requested_rank=requested_rank,
            metric_numerical_rank=numerical_rank,
            metric_min_positive=minimum_positive,
            metric_condition=condition,
            backward_residual=selected_check["backward"],
            metric_dual_residual=selected_check["dual"],
            metric_dual_relative_residual=selected_check["dual_relative"],
            null_residual=selected_check["null"],
            warm_energy=warm_energy,
            upper_bound_gap=float(selected_energy - warm_energy),
            metric_support=metric_support,
            discarded_support_residual=selected_check["discarded"],
        )
        local_update = FrontierSiteUpdate(
            site=int(site),
            raw_dim=size,
            metric_rank=support_rank,
            metric_rank_is_projected=support_rank < numerical_rank,
            solver=selected_method,
            solver_converged=verified,
            message=solver_message,
            hamiltonian_matvecs=(
                0
                if davidson_diagnostics is None
                else davidson_diagnostics.hamiltonian_matvecs
            ),
            metric_matvecs=(
                0
                if davidson_diagnostics is None
                else davidson_diagnostics.metric_matvecs
            ),
            iterations=(
                0
                if davidson_diagnostics is None
                else davidson_diagnostics.iterations
            ),
            residual_norm=selected_check["raw"],
            energy_before=warm_energy,
            energy=float(selected_energy),
            accepted=accepted,
            solver_coordinate_residual_norm=selected_check["dual_relative"],
        )
        return float(selected_energy), selected_vector, local_update, diagnostics

    @staticmethod
    def _block_operator_inf_norm(operator):
        if not hasattr(operator, "blocks"):
            return float("nan")
        layout = operator.layout
        row_sums = np.zeros((layout.nblocks, layout.virtual_size))
        for (row, _column), block in operator.blocks.items():
            row_sums[row] += np.sum(np.abs(block), axis=1)
        return float(np.max(row_sums, initial=0.0))

    def _pair_block_residual_verification(
        self,
        problem,
        energy,
        vector,
        *,
        numerical_floor,
        solve_floor=None,
        metric_eigensystems=None,
    ):
        vector = np.asarray(vector).reshape(-1)
        metric_vector = problem.metric.matvec(vector)
        hamiltonian_vector = (
            problem.hamiltonian.verification_matvec(vector)
            if getattr(
                problem.hamiltonian,
                "has_verification_action",
                False,
            )
            else problem.hamiltonian.matvec(vector)
        )
        residual = hamiltonian_vector - float(energy) * metric_vector
        vector_norm = max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
        metric_inf = self._block_operator_inf_norm(problem.metric)
        hamiltonian_inf = self._block_operator_inf_norm(problem.hamiltonian)
        if not np.isfinite(hamiltonian_inf):
            hamiltonian_inf = float(
                np.linalg.norm(hamiltonian_vector, ord=np.inf)
                / max(
                    np.linalg.norm(vector, ord=np.inf),
                    np.finfo(float).tiny,
                )
            )
        backward_scale = max(
            (hamiltonian_inf + abs(float(energy)) * metric_inf) * vector_norm,
            np.finfo(float).tiny,
        )
        dual_squared = 0.0
        projected_hamiltonian_squared = 0.0
        null_squared = 0.0
        maximum_metric_value = 0.0
        if solve_floor is None:
            solve_floor = numerical_floor
        discarded_squared = 0.0
        if metric_eigensystems is None:
            metric_eigensystems = problem.metric_eigensystems()
        for block, (values, vectors) in enumerate(metric_eigensystems):
            indices = problem.layout.block_indices[block]
            numerical_active = values > float(numerical_floor)
            active = values > float(solve_floor)
            maximum_metric_value = max(
                maximum_metric_value,
                float(np.max(values, initial=0.0)),
            )
            residual_coefficients = vectors.T.conj() @ residual[indices]
            hamiltonian_coefficients = (
                vectors.T.conj() @ hamiltonian_vector[indices]
            )
            if np.any(active):
                dual_squared += float(
                    np.sum(
                        np.abs(residual_coefficients[active]) ** 2
                        / values[active]
                    )
                )
                projected_hamiltonian_squared += float(
                    np.sum(
                        np.abs(hamiltonian_coefficients[active]) ** 2
                        / values[active]
                    )
                )
            discarded = numerical_active & ~active
            if np.any(discarded):
                discarded_squared += float(
                    np.sum(np.abs(residual_coefficients[discarded]) ** 2)
                )
            true_null = ~numerical_active
            if np.any(true_null):
                null_squared += float(
                    np.sum(np.abs(residual_coefficients[true_null]) ** 2)
                )
        dual = float(np.sqrt(max(dual_squared, 0.0)))
        metric_norm = max(
            float(np.real(np.vdot(vector, metric_vector))),
            np.finfo(float).tiny,
        )
        dual_scale = max(
            float(np.sqrt(max(projected_hamiltonian_squared, 0.0))),
            abs(float(energy)) * np.sqrt(metric_norm),
            (
                hamiltonian_inf
                * vector_norm
                / np.sqrt(
                    max(maximum_metric_value, np.finfo(float).tiny)
                )
            ),
            np.finfo(float).tiny,
        )
        null_scale = max(
            float(np.linalg.norm(hamiltonian_vector)),
            abs(float(energy)) * float(np.linalg.norm(metric_vector)),
            np.finfo(float).tiny,
        )
        return {
            "raw": float(np.linalg.norm(residual)),
            "backward": float(np.linalg.norm(residual) / backward_scale),
            "dual": dual,
            "dual_relative": float(dual / dual_scale),
            "null": float(np.sqrt(max(null_squared, 0.0)) / null_scale),
            "discarded": float(
                np.sqrt(max(discarded_squared, 0.0)) / null_scale
            ),
        }

    def _solve_verified_pair_block_pencil(
        self,
        site,
        problem,
        warm,
        *,
        metric_tol,
        eig_tol,
        maxiter,
        max_subspace,
        dense_fallback_dim,
        metric_support="regularized",
        preconditioner="auto",
        block_size=1,
        recycle=True,
        recycle_min_size=64,
    ):
        """Solve and verify a conditional physical-block pair pencil."""
        metric_support = str(metric_support).lower().replace("-", "_")
        if metric_support not in {"regularized", "numerical"}:
            raise ValueError(
                "metric_support must be 'regularized' or 'numerical'."
            )
        warm = np.asarray(warm).reshape(-1)
        size = warm.size
        metric_scale = max(
            (
                float(np.linalg.norm(block, ord=np.inf))
                for block in problem.metric.blocks.values()
            ),
            default=0.0,
        )
        metric_scale = max(metric_scale, np.finfo(float).tiny)
        numerical_floor = 64.0 * np.finfo(float).eps * metric_scale
        requested_floor = max(
            float(metric_tol) * metric_scale,
            numerical_floor,
        )
        solve_floor = (
            numerical_floor
            if metric_support == "numerical"
            else requested_floor
        )
        metric_eigensystems = problem.metric_eigensystems()
        metric_values = tuple(values for values, _vectors in metric_eigensystems)
        numerical_rank = int(
            sum(np.count_nonzero(values > numerical_floor) for values in metric_values)
        )
        requested_rank = int(
            sum(np.count_nonzero(values > requested_floor) for values in metric_values)
        )
        support_rank = int(
            sum(np.count_nonzero(values > solve_floor) for values in metric_values)
        )
        if numerical_rank == 0:
            raise ValueError("merged-pair overlap metric is numerically singular.")
        if support_rank == 0:
            raise ValueError("requested merged-pair metric support is empty.")
        positive_values = np.concatenate(
            [values[values > numerical_floor] for values in metric_values]
        )
        minimum_positive = float(np.min(positive_values))
        condition = float(np.max(positive_values) / minimum_positive)
        metric_warm = problem.metric.matvec(warm)
        warm_norm = float(np.real(np.vdot(warm, metric_warm)))
        if warm_norm <= np.finfo(float).tiny:
            raise ValueError("warm merged-pair state has zero metric norm.")
        warm = warm / np.sqrt(warm_norm)
        warm_energy = self._pair_rayleigh_actions(
            warm,
            problem.metric.matvec,
            problem.hamiltonian.matvec,
        )
        verification_tolerance = max(
            10.0 * float(eig_tol),
            8.0 * np.sqrt(np.finfo(float).eps),
        )
        materialized_hamiltonian = hasattr(problem.hamiltonian, "blocks")
        block_failure_reason = ""
        try:
            energy, vector, block_diagnostics = problem.solve(
                warm,
                tol=float(eig_tol),
                metric_tol=float(solve_floor / metric_scale),
                maxiter=min(128, int(maxiter) if maxiter is not None else 128),
                max_subspace=min(size, int(max_subspace), 64),
                random_seed=int(site),
                dense_component_max_size=(
                    int(dense_fallback_dim)
                    if materialized_hamiltonian
                    else 0
                ),
                recycle_spaces=(self._davidson_recycle if recycle else None),
                recycle_prefix=("pair", int(site)),
                recycle_min_size=int(recycle_min_size),
                preconditioner=preconditioner,
                block_size=int(block_size),
                executor=self._solver_executor,
            )
            check = self._pair_block_residual_verification(
                problem,
                energy,
                vector,
                numerical_floor=numerical_floor,
                solve_floor=solve_floor,
                metric_eigensystems=metric_eigensystems,
            )
            verified = bool(
                check["dual_relative"] <= verification_tolerance
                and check["null"] <= verification_tolerance
                and energy <= warm_energy
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            block_failure_reason = f"conditional block solve failed: {error}"
            if materialized_hamiltonian and size <= int(dense_fallback_dim):
                pair_metric = problem.metric.to_dense()
                pair_effective = problem.hamiltonian.to_dense()
                result = self._solve_verified_pair_pencil(
                    site,
                    pair_metric,
                    pair_effective,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=dense_fallback_dim,
                    metric_support=metric_support,
                )
                energy, vector, local_update, diagnostics = result
                diagnostics = replace(
                    diagnostics,
                    attempts=("conditional_blocks", *diagnostics.attempts),
                    fallback_reason=(
                        block_failure_reason
                        if not diagnostics.fallback_reason
                        else (
                            f"{block_failure_reason}; "
                            f"{diagnostics.fallback_reason}"
                        )
                    ),
                    dense_fallback=True,
                )
                local_update = replace(
                    local_update,
                    solver=diagnostics.method,
                    message="verified dense fallback after conditional blocks",
                )
                return energy, vector, local_update, diagnostics
            energy = warm_energy
            vector = warm
            check = self._pair_block_residual_verification(
                problem,
                energy,
                vector,
                numerical_floor=numerical_floor,
                solve_floor=solve_floor,
                metric_eigensystems=metric_eigensystems,
            )
            verified = False
            block_diagnostics = None

        if (
            not verified
            and materialized_hamiltonian
            and size <= int(dense_fallback_dim)
        ):
            pair_metric = problem.metric.to_dense()
            pair_effective = problem.hamiltonian.to_dense()
            energy, vector, local_update, diagnostics = (
                self._solve_verified_pair_pencil(
                    site,
                    pair_metric,
                    pair_effective,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=dense_fallback_dim,
                    metric_support=metric_support,
                )
            )
            diagnostics = replace(
                diagnostics,
                attempts=("conditional_blocks", *diagnostics.attempts),
                fallback_reason=(
                    "conditional block root failed full residual verification; "
                    f"{diagnostics.fallback_reason}"
                ),
                dense_fallback=True,
            )
            return energy, vector, local_update, diagnostics

        retained_warm = not verified
        if retained_warm:
            energy = warm_energy
            vector = warm
            check = self._pair_block_residual_verification(
                problem,
                energy,
                vector,
                numerical_floor=numerical_floor,
                solve_floor=solve_floor,
                metric_eigensystems=metric_eigensystems,
            )
            if not block_failure_reason:
                block_failure_reason = (
                    "conditional block root failed full residual verification"
                )
        lowest_root_certified = bool(
            not retained_warm
            and block_diagnostics is not None
            and block_diagnostics.dense_components
            and all(block_diagnostics.dense_components)
        )
        method = (
            "warm"
            if retained_warm
            else (
                "conditional_blocks_certified"
                if lowest_root_certified
                else "conditional_blocks_davidson"
            )
        )
        energy_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(warm_energy),
        )
        accepted = bool(energy <= warm_energy + energy_tolerance)
        diagnostics = FrontierMergedSolveDiagnostics(
            method=method,
            attempts=("conditional_blocks",),
            verified=verified,
            lowest_root_certified=lowest_root_certified,
            fallback_reason=block_failure_reason,
            dense_fallback=False,
            metric_requested_rank=requested_rank,
            metric_numerical_rank=numerical_rank,
            metric_min_positive=minimum_positive,
            metric_condition=condition,
            backward_residual=check["backward"],
            metric_dual_residual=check["dual"],
            metric_dual_relative_residual=check["dual_relative"],
            null_residual=check["null"],
            warm_energy=warm_energy,
            upper_bound_gap=float(energy - warm_energy),
            metric_support=metric_support,
            discarded_support_residual=check["discarded"],
        )
        local_update = FrontierSiteUpdate(
            site=int(site),
            raw_dim=size,
            metric_rank=support_rank,
            metric_rank_is_projected=support_rank < numerical_rank,
            solver=method,
            solver_converged=verified,
            message=(
                block_failure_reason
                if not verified
                else
                (
                    "converged in requested metric support"
                    if verified and not block_diagnostics.converged
                    else block_diagnostics.message
                )
                if block_diagnostics is not None
                else block_failure_reason
            ),
            hamiltonian_matvecs=(
                0
                if block_diagnostics is None
                else block_diagnostics.hamiltonian_matvecs
            ),
            metric_matvecs=(
                0
                if block_diagnostics is None
                else block_diagnostics.metric_matvecs
            ),
            iterations=(
                0 if block_diagnostics is None else block_diagnostics.iterations
            ),
            residual_norm=check["raw"],
            energy_before=warm_energy,
            energy=float(energy),
            accepted=accepted,
            physical_blocks=(
                len(problem.metric.blocks)
                if block_diagnostics is None
                else block_diagnostics.metric_blocks
            ),
            hamiltonian_blocks=(
                problem.hamiltonian.block_count
                if block_diagnostics is None
                else block_diagnostics.hamiltonian_blocks
            ),
            block_component_sizes=(
                tuple(len(component) for component in problem.hamiltonian_components)
                if block_diagnostics is None
                else block_diagnostics.component_sizes
            ),
            stored_operator_elements=(
                problem.stored_elements
                if block_diagnostics is None
                else block_diagnostics.stored_elements
            ),
            solver_coordinate_residual_norm=check["dual_relative"],
        )
        return float(energy), np.asarray(vector).reshape(-1), local_update, diagnostics

    @staticmethod
    def _pair_rayleigh(vector, metric, effective):
        return FrontierTiedLETTA._pair_rayleigh_actions(
            vector,
            lambda trial: metric @ trial,
            lambda trial: effective @ trial,
        )

    @staticmethod
    def _pair_rayleigh_actions(vector, metric_action, effective_action):
        """Evaluate a Rayleigh quotient from operator actions."""
        vector = np.asarray(vector).reshape(-1)
        metric_vector = np.asarray(metric_action(vector)).reshape(-1)
        norm = np.vdot(vector, metric_vector)
        euclidean_norm = max(
            float(np.vdot(vector, vector).real),
            np.finfo(float).tiny,
        )
        if not np.isfinite(norm) or float(np.real(norm)) <= (
            np.finfo(float).tiny * euclidean_norm
        ):
            return float("inf")
        numerator = np.vdot(
            vector,
            np.asarray(effective_action(vector)).reshape(-1),
        )
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

    def _pair_factor_support_indices(self, site, variable):
        """Optional exact parameter support for a pair-factor update."""
        return None

    def _project_pair_factor_support(self, site, left_tensor, right_tensor):
        """Project pair factors onto any symmetry-restricted coordinates."""
        result = []
        for variable, tensor in (("left", left_tensor), ("right", right_tensor)):
            tensor = np.asarray(tensor).copy()
            support = self._pair_factor_support_indices(site, variable)
            if support is not None:
                support = np.asarray(support, dtype=int)
                flat = np.zeros(tensor.size, dtype=tensor.dtype)
                flat[support] = tensor.reshape(-1)[support]
                tensor = flat.reshape(tensor.shape)
            result.append(tensor)
        return tuple(result)

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
        metric_action=None,
        metric_blocks=None,
    ):
        r"""Project a merged tensor with its physical environment metric.

        This minimizes the represented-state error

        .. math::

            \|A B-M_\star\|_{N_{\rm eff}}^2

        by alternating weighted least-squares solves.  It replaces the
        Euclidean Frobenius norm used by a conditional SVD.
        """
        if metric_action is None:
            if metric is None:
                raise ValueError("metric or metric_action must be supplied.")
            metric = self._hermitian_part(metric)

            def metric_action(vector):
                return metric @ vector
        target_vector = np.asarray(target).reshape(-1)
        target_norm_squared = max(
            float(
                np.real(
                    np.vdot(
                        target_vector,
                        np.asarray(metric_action(target_vector)).reshape(-1),
                    )
                )
            ),
            np.finfo(float).tiny,
        )
        target_norm = np.sqrt(target_norm_squared)
        left_tensor, right_tensor = self._project_pair_factor_support(
            site,
            left_tensor,
            right_tensor,
        )
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        layout = PhysicalBlockLayout(merged_shape)
        if metric_blocks is None:
            if metric is None:
                raise ValueError(
                    "conditional metric blocks are required for matrix-free "
                    "retraction."
                )
            metric_blocks = tuple(
                metric[np.ix_(indices, indices)]
                for indices in layout.block_indices
            )
        else:
            metric_blocks = tuple(
                self._hermitian_part(block) for block in metric_blocks
            )
            expected_block_shape = (
                layout.virtual_size,
                layout.virtual_size,
            )
            if len(metric_blocks) != layout.nblocks or any(
                block.shape != expected_block_shape for block in metric_blocks
            ):
                raise ValueError("metric_blocks are incompatible with the pair.")
        metric_dtype = np.result_type(
            target_vector.dtype,
            *(block.dtype for block in metric_blocks),
        )
        left_sites = self.physical_groups[int(site)]
        right_sites = self.physical_groups[int(site) + 1]
        union_shape = tuple(self.dims[index] for index in union_sites)
        entries = []
        for block, configuration in enumerate(np.ndindex(*union_shape)):
            values = dict(zip(union_sites, configuration))
            entries.append(
                (
                    block,
                    tuple(values[index] for index in left_sites),
                    tuple(values[index] for index in right_sites),
                )
            )

        def solve_psd(normal, rhs, reference):
            normal = self._hermitian_part(normal)
            eigenvalues, eigenvectors = np.linalg.eigh(normal)
            scale = max(
                float(np.linalg.norm(normal, ord=np.inf)),
                np.finfo(float).tiny,
            )
            keep = eigenvalues > max(
                float(metric_tol),
                64.0 * np.finfo(float).eps,
            ) * scale
            if not np.any(keep):
                return np.asarray(reference).reshape(-1).copy()
            active = eigenvectors[:, keep]
            solution = active @ (
                (active.T.conj() @ rhs) / eigenvalues[keep]
            )
            reference = np.asarray(reference).reshape(-1)
            solution += reference - active @ (active.T.conj() @ reference)
            return solution

        def relative_error(left, right):
            approximation = self._merge_pair_factors(
                site,
                union_sites,
                left,
                right,
            ).reshape(-1)
            residual = approximation - target_vector
            metric_residual = np.asarray(metric_action(residual)).reshape(-1)
            error_squared = float(np.real(np.vdot(residual, metric_residual)))
            return np.sqrt(max(error_squared, 0.0)) / target_norm

        current_error = relative_error(left_tensor, right_tensor)
        best_error = current_error
        best_left = left_tensor.copy()
        best_right = right_tensor.copy()
        left_shape = tuple(self.dims[index] for index in left_sites)
        right_shape = tuple(self.dims[index] for index in right_sites)
        for _sweep in range(int(max_sweeps)):
            updated_left = left_tensor.copy()
            for left_configuration in np.ndindex(*left_shape):
                local_size = left_tensor.shape[0] * left_tensor.shape[1]
                normal = np.zeros((local_size, local_size), dtype=metric_dtype)
                rhs = np.zeros(local_size, dtype=metric_dtype)
                for block, candidate_left, right_configuration in entries:
                    if candidate_left != left_configuration:
                        continue
                    right = right_tensor[
                        (slice(None), slice(None), *right_configuration)
                    ]
                    design = np.kron(
                        np.eye(left_tensor.shape[0], dtype=right.dtype),
                        right.T,
                    )
                    metric_block = metric_blocks[block]
                    target_block = target_vector[layout.block_indices[block]]
                    normal += design.T.conj() @ metric_block @ design
                    rhs += design.T.conj() @ metric_block @ target_block
                reference = left_tensor[
                    (slice(None), slice(None), *left_configuration)
                ]
                updated_left[
                    (slice(None), slice(None), *left_configuration)
                ] = solve_psd(normal, rhs, reference).reshape(reference.shape)
            left_tensor = updated_left

            updated_right = right_tensor.copy()
            for right_configuration in np.ndindex(*right_shape):
                local_size = right_tensor.shape[0] * right_tensor.shape[1]
                normal = np.zeros((local_size, local_size), dtype=metric_dtype)
                rhs = np.zeros(local_size, dtype=metric_dtype)
                for block, left_configuration, candidate_right in entries:
                    if candidate_right != right_configuration:
                        continue
                    left = left_tensor[
                        (slice(None), slice(None), *left_configuration)
                    ]
                    design = np.kron(
                        left,
                        np.eye(right_tensor.shape[1], dtype=left.dtype),
                    )
                    metric_block = metric_blocks[block]
                    target_block = target_vector[layout.block_indices[block]]
                    normal += design.T.conj() @ metric_block @ design
                    rhs += design.T.conj() @ metric_block @ target_block
                reference = right_tensor[
                    (slice(None), slice(None), *right_configuration)
                ]
                updated_right[
                    (slice(None), slice(None), *right_configuration)
                ] = solve_psd(normal, rhs, reference).reshape(reference.shape)
            right_tensor = updated_right
            left_tensor, right_tensor = self._balance_pair_factors(
                left_tensor,
                right_tensor,
            )
            next_error = relative_error(left_tensor, right_tensor)
            if next_error < best_error:
                best_error = next_error
                best_left = left_tensor.copy()
                best_right = right_tensor.copy()
            if current_error - next_error <= 32.0 * np.finfo(float).eps:
                break
            current_error = next_error
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
        factor_solver="matrix_free",
        eig_tol=1.0e-10,
        maxiter=256,
        max_subspace=32,
        metric_action=None,
        effective_action=None,
    ):
        """Alternately minimize the exact pair Rayleigh quotient."""
        left_tensor, right_tensor = self._project_pair_factor_support(
            site,
            left_tensor,
            right_tensor,
        )
        factor_solver = str(factor_solver).lower().replace("-", "_")
        if factor_solver not in {"matrix_free", "dense"}:
            raise ValueError("factor_solver must be 'matrix_free' or 'dense'.")
        if metric_action is None:
            if metric is None:
                raise ValueError("metric or metric_action must be supplied.")

            def metric_action(vector):
                return metric @ vector

        if effective_action is None:
            if effective is None:
                raise ValueError("effective or effective_action must be supplied.")

            def effective_action(vector):
                return effective @ vector

        def pair_energy(left, right):
            merged = self._merge_pair_factors(
                site,
                union_sites,
                left,
                right,
            )
            return self._pair_rayleigh_actions(
                merged.reshape(-1),
                metric_action,
                effective_action,
            )

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
                old_factor = (
                    left_tensor if variable == "left" else right_tensor
                )
                support = self._pair_factor_support_indices(site, variable)
                support = (
                    np.arange(old_factor.size, dtype=int)
                    if support is None
                    else np.asarray(support, dtype=int)
                )

                def embed(vector):
                    full = np.zeros(
                        old_factor.size,
                        dtype=np.result_type(vector, old_factor.dtype),
                    )
                    full[support] = np.asarray(vector).reshape(-1)
                    return full

                def forward(vector):
                    return self._pair_factor_action(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        embed(vector),
                        variable=variable,
                    )

                def adjoint(vector):
                    full = self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable=variable,
                    )
                    return np.asarray(full).reshape(-1)[support]

                def local_metric_action(vector):
                    return adjoint(metric_action(forward(vector)))

                def local_hamiltonian_action(vector):
                    return adjoint(effective_action(forward(vector)))

                def projected_local_matrices():
                    identity = np.eye(support.size, dtype=old_factor.dtype)
                    local_metric = np.column_stack(
                        [
                            local_metric_action(identity[:, column])
                            for column in range(support.size)
                        ]
                    )
                    local_effective = np.column_stack(
                        [
                            local_hamiltonian_action(identity[:, column])
                            for column in range(support.size)
                        ]
                    )
                    return (
                        self._hermitian_part(local_metric),
                        self._hermitian_part(local_effective),
                    )

                try:
                    if factor_solver == "matrix_free":
                        (
                            _local_energy,
                            vector,
                            diagnostics,
                        ) = lowest_generalized_davidson(
                            local_hamiltonian_action,
                            local_metric_action,
                            old_factor.reshape(-1)[support],
                            tol=eig_tol,
                            metric_tol=metric_tol,
                            maxiter=maxiter,
                            max_subspace=min(
                                int(max_subspace),
                                support.size,
                            ),
                            random_seed=int(site) + (variable == "right"),
                        )
                        if not diagnostics.converged:
                            raise ValueError(diagnostics.message)
                    else:
                        if metric is not None and effective is not None:
                            design = self._pair_factor_design(
                                site,
                                union_sites,
                                left_tensor,
                                right_tensor,
                                variable=variable,
                            )
                            design = design[:, support]
                            local_metric = design.T.conj() @ metric @ design
                            local_effective = (
                                design.T.conj() @ effective @ design
                            )
                        else:
                            local_metric, local_effective = (
                                projected_local_matrices()
                            )
                        _local_energy, vector = _lowest_generalized_eigenpair(
                            local_effective,
                            local_metric,
                            metric_tol=metric_tol,
                        )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    # The dense projected fallback preserves robustness while
                    # still avoiding a persistent dense-J path when Davidson
                    # converges normally.
                    local_metric, local_effective = projected_local_matrices()
                    try:
                        _local_energy, vector = _lowest_generalized_eigenpair(
                            local_effective,
                            local_metric,
                            metric_tol=metric_tol,
                        )
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                        continue
                if variable == "left":
                    proposed_left = embed(vector).reshape(left_tensor.shape)
                    proposed_right = right_tensor
                else:
                    proposed_left = left_tensor
                    proposed_right = embed(vector).reshape(right_tensor.shape)
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
        metric_action=None,
        effective_action=None,
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
        if metric_action is None:
            if metric is None:
                raise ValueError("metric or metric_action must be supplied.")

            def metric_action(vector):
                return metric @ vector

        if effective_action is None:
            if effective is None:
                raise ValueError("effective or effective_action must be supplied.")

            def effective_action(vector):
                return effective @ vector
        left_tensor, right_tensor = self._project_pair_factor_support(
            site,
            left_tensor,
            right_tensor,
        )
        left_support = self._pair_factor_support_indices(site, "left")
        right_support = self._pair_factor_support_indices(site, "right")
        left_support = (
            np.arange(left_tensor.size, dtype=int)
            if left_support is None
            else np.asarray(left_support, dtype=int)
        )
        right_support = (
            np.arange(right_tensor.size, dtype=int)
            if right_support is None
            else np.asarray(right_support, dtype=int)
        )

        def embed(vector, support, size, dtype):
            full = np.zeros(size, dtype=np.result_type(vector, dtype))
            full[support] = np.asarray(vector).reshape(-1)
            return full

        merged = self._merge_pair_factors(
            site,
            union_sites,
            left_tensor,
            right_tensor,
        ).reshape(-1)
        residual = np.asarray(target).reshape(-1) - merged
        split = left_support.size
        tangent_size = split + right_support.size

        def tangent_forward(vector):
            vector = np.asarray(vector).reshape(-1)
            return self._pair_factor_action(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                embed(
                    vector[:split],
                    left_support,
                    left_tensor.size,
                    left_tensor.dtype,
                ),
                variable="left",
            ) + self._pair_factor_action(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                embed(
                    vector[split:],
                    right_support,
                    right_tensor.size,
                    right_tensor.dtype,
                ),
                variable="right",
            )

        def tangent_adjoint(vector):
            return np.concatenate(
                (
                    np.asarray(self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable="left",
                    )).reshape(-1)[left_support],
                    np.asarray(self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable="right",
                    )).reshape(-1)[right_support],
                )
            )

        def normal_action(vector):
            return tangent_adjoint(metric_action(tangent_forward(vector)))

        normal = LinearOperator(
            (tangent_size, tangent_size),
            matvec=normal_action,
            rmatvec=normal_action,
            dtype=np.result_type(
                left_tensor.dtype,
                right_tensor.dtype,
                residual.dtype,
            ),
        )
        tangent_rhs = tangent_adjoint(metric_action(residual))
        iterative_tolerance = max(float(metric_tol), np.finfo(float).eps)
        tangent, cg_info = cg(
            normal,
            tangent_rhs,
            x0=np.zeros_like(tangent_rhs),
            rtol=iterative_tolerance,
            atol=0.0,
            maxiter=max(32, 4 * tangent_size),
        )
        if cg_info != 0:
            lsmr_result = lsmr(
                normal,
                tangent_rhs,
                atol=iterative_tolerance,
                btol=iterative_tolerance,
                maxiter=max(32, 4 * tangent_size),
            )
            tangent = lsmr_result[0]
            if lsmr_result[1] not in {1, 2, 4, 5}:
                raise ValueError("matrix-free tangent solve did not converge.")
        if not np.all(np.isfinite(tangent)):
            raise ValueError("matrix-free tangent solve produced nonfinite values.")
        delta_left = embed(
            tangent[:split],
            left_support,
            left_tensor.size,
            left_tensor.dtype,
        ).reshape(left_tensor.shape)
        delta_right = embed(
            tangent[split:],
            right_support,
            right_tensor.size,
            right_tensor.dtype,
        ).reshape(right_tensor.shape)

        best_left = np.asarray(left_tensor).copy()
        best_right = np.asarray(right_tensor).copy()
        best_energy = self._pair_rayleigh_actions(
            merged,
            metric_action,
            effective_action,
        )
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
                proposed_energy = self._pair_rayleigh_actions(
                    proposed.reshape(-1),
                    metric_action,
                    effective_action,
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
        incumbent_left=None,
        incumbent_right=None,
        factor_solver="matrix_free",
        factor_eig_tol=1.0e-10,
        factor_maxiter=256,
        factor_max_subspace=32,
        metric_action=None,
        effective_action=None,
        metric_blocks=None,
    ):
        """Environment-project and variationally retract a merged pair."""
        if metric_action is None:
            if metric is None:
                raise ValueError("metric or metric_action must be supplied.")

            def metric_action(vector):
                return metric @ vector

        if effective_action is None:
            if effective is None:
                raise ValueError("effective or effective_action must be supplied.")

            def effective_action(vector):
                return effective @ vector
        old_left = (
            self.tensors[site]
            if incumbent_left is None
            else np.asarray(incumbent_left)
        )
        old_right = (
            self.tensors[site + 1]
            if incumbent_right is None
            else np.asarray(incumbent_right)
        )
        old_left, old_right = self._project_pair_factor_support(
            site,
            old_left,
            old_right,
        )
        euclidean_left, euclidean_right = self._project_pair_factor_support(
            site,
            euclidean_left,
            euclidean_right,
        )
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
                    metric_action=metric_action,
                    metric_blocks=metric_blocks,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            metric_left = euclidean_left.copy()
            metric_right = euclidean_right.copy()
            projection_error = float("inf")
        metric_left, metric_right = self._project_pair_factor_support(
            site,
            metric_left,
            metric_right,
        )

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
            return self._pair_rayleigh_actions(
                candidate.reshape(-1),
                metric_action,
                effective_action,
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
        best_energy = self._pair_rayleigh_actions(
            best_merged.reshape(-1),
            metric_action,
            effective_action,
        )
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
            factor_solver=factor_solver,
            eig_tol=factor_eig_tol,
            maxiter=factor_maxiter,
            max_subspace=factor_max_subspace,
            metric_action=metric_action,
            effective_action=effective_action,
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
                    metric_action=metric_action,
                    effective_action=effective_action,
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
                    factor_solver=factor_solver,
                    eig_tol=factor_eig_tol,
                    maxiter=factor_maxiter,
                    max_subspace=factor_max_subspace,
                    metric_action=metric_action,
                    effective_action=effective_action,
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
                    factor_solver=factor_solver,
                    eig_tol=factor_eig_tol,
                    maxiter=factor_maxiter,
                    max_subspace=factor_max_subspace,
                    metric_action=metric_action,
                    effective_action=effective_action,
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
            float(best_energy),
        )

    def _optimize_two_sites_legacy(
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
        if (
            type(self) is not FrontierTiedLETTA
            and not self._has_charge_resolved_two_site_split
        ):
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
        if split_strategy in {"auto", "adaptive"}:
            split_strategy = "hybrid"
        if split_strategy not in {"variational", "svd", "hybrid"}:
            raise ValueError(
                "split_strategy must be 'svd', 'variational', or 'hybrid'."
            )
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
        right_sites = self.physical_groups[following]
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
            tuple(temporary_parents),
            bond_dims=tuple(temporary_bonds),
            tensors=temporary_tensors,
            frontier_backend=self.frontier_backend,
            chunk_size=self.chunk_size,
            chunk_memory=self.chunk_memory,
            chunk_span=self.chunk_span,
            workers=self.workers,
            path_optimizer=self.path_optimizer,
            max_rank=self.tt_options["max_rank"],
            rtol=self.tt_options["rtol"],
            atol=self.tt_options["atol"],
            transfer_max_rank=self.tt_options["transfer_max_rank"],
            transfer_rtol=self.tt_options["transfer_rtol"],
            transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
            tt_channels=self.tt_channels,
            tt_gauge=self.tt_gauge,
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
                _factor_energy,
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

    def optimize_two_sites(
        self,
        site,
        *,
        solver="verified",
        environment=None,
        verify_global=True,
        pair_operator_backend="auto",
        pair_dense_max_elements=4_000_000,
        packed_min_size=4096,
        merged_dense_fallback_dim=2048,
        metric_support="regularized",
        outer_cycles=8,
        factor_solver="matrix_free",
        split_strategy="variational",
        split_metric_tol=1.0e-12,
        split_metric_sweeps=6,
        split_variational_sweeps=8,
        split_random_starts=0,
        split_random_seed=0,
        split_energy_tol=1.0e-12,
        **solver_options,
    ):
        r"""Run a verified, cached, fixed-rank adjacent-pair update.

        The merged pencil is warm-started, checked with a metric-dual
        residual, and densely certified when its selected metric support is
        manageable.  ``metric_support="regularized"`` respects ``metric_tol``;
        ``"numerical"`` is an explicit full-positive-support diagnostic.
        Retraction uses conditional norm blocks, and the factor equations use
        direct ``J``/``J^dagger`` actions by default.  Up to ``outer_cycles``
        merge--retract--relax corrections reuse the same exact pair
        environment and operators.
        """
        if (
            type(self) is not FrontierTiedLETTA
            and not self._has_charge_resolved_two_site_split
        ):
            raise TypeError(
                "two-site merge/split currently supports exact unrestricted "
                "FrontierTiedLETTA states; symmetry sectors need a "
                "charge-resolved split."
            )
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        solver = str(solver).lower().replace("-", "_")
        if solver in {"metric_orthonormal", "orthonormal_metric"}:
            solver = "whitened"
        if solver not in {
            "verified",
            "whitened",
            "direct",
            "matrix_free",
            "block_sparse",
        }:
            raise ValueError(
                "solver must be 'verified', 'metric_orthonormal' "
                "(alias 'whitened'), 'direct', "
                "'matrix_free', or 'block_sparse'."
            )
        split_strategy = str(split_strategy).lower().replace("-", "_")
        if split_strategy in {"environment", "metric", "als"}:
            split_strategy = "variational"
        if split_strategy in {"auto", "adaptive"}:
            split_strategy = "hybrid"
        if split_strategy not in {"variational", "svd", "hybrid"}:
            raise ValueError(
                "split_strategy must be 'svd', 'variational', or 'hybrid'."
            )
        pair_operator_backend = str(pair_operator_backend).lower().replace(
            "-", "_"
        )
        if pair_operator_backend in {
            "action_block",
            "lazy",
            "lazy_block",
            "matrix_free",
            "matrix_free_block",
        }:
            pair_operator_backend = "action"
        if pair_operator_backend not in {
            "auto", "dense", "block", "action", "packed"
        }:
            raise ValueError(
                "pair_operator_backend must be 'auto', 'dense', 'block', "
                "'action', or 'packed'."
            )
        factor_solver = str(factor_solver).lower().replace("-", "_")
        if factor_solver not in {"matrix_free", "dense"}:
            raise ValueError("factor_solver must be 'matrix_free' or 'dense'.")
        metric_support = str(metric_support).lower().replace("-", "_")
        if metric_support not in {"regularized", "numerical"}:
            raise ValueError(
                "metric_support must be 'regularized' or 'numerical'."
            )
        verify_global = bool(verify_global)
        outer_cycles = int(outer_cycles)
        if outer_cycles < 1:
            raise ValueError("outer_cycles must be positive.")
        merged_dense_fallback_dim = int(merged_dense_fallback_dim)
        if merged_dense_fallback_dim < 1:
            raise ValueError("merged_dense_fallback_dim must be positive.")
        pair_dense_max_elements = int(pair_dense_max_elements)
        if pair_dense_max_elements < 1:
            raise ValueError("pair_dense_max_elements must be positive.")
        packed_min_size = int(packed_min_size)
        if packed_min_size < 1:
            raise ValueError("packed_min_size must be positive.")
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

        options = dict(solver_options)
        metric_tol = float(options.pop("metric_tol", 1.0e-12))
        eig_tol = float(options.pop("eig_tol", 1.0e-10))
        maxiter = options.pop("maxiter", 1600)
        max_subspace = int(options.pop("max_subspace", 96))
        preconditioner = options.pop("preconditioner", "auto")
        block_size_option = options.pop("block_size", 1)
        automatic_block_size = bool(
            isinstance(block_size_option, str)
            and block_size_option.lower().replace("-", "_") == "auto"
        )
        block_size = 1 if automatic_block_size else int(block_size_option)
        recycle = bool(options.pop("recycle", True))
        recycle_min_size = int(options.pop("recycle_min_size", 64))
        # These one-site dispatch controls have no role once a pair backend is
        # selected explicitly, but accepting them keeps the public solver
        # options consistent with one-site calls.
        options.pop("matrix_free_threshold", None)
        options.pop("block_sparse_max_elements", None)
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"unexpected two-site solver option(s): {names}")
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(eig_tol) or eig_tol < 0.0:
            raise ValueError("eig_tol must be finite and nonnegative.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if recycle_min_size < 1:
            raise ValueError("recycle_min_size must be positive.")
        if maxiter is not None:
            maxiter = int(maxiter)
            if maxiter < 1:
                raise ValueError("maxiter must be positive or None.")
        if max_subspace < 2:
            raise ValueError("max_subspace must be at least two.")

        following = site + 1
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        global_energy_before = (
            float(self.expectation()) if verify_global else None
        )
        old_left = self.tensors[site].copy()
        old_right = self.tensors[following].copy()
        incumbent_left = old_left.copy()
        incumbent_right = old_right.copy()
        warm = self._merge_pair_factors(
            site,
            plan.union_sites,
            incumbent_left,
            incumbent_right,
        )
        merged_size = warm.size
        if automatic_block_size:
            block_size = 2 if merged_size >= 16384 else 1

        selected_backend = pair_operator_backend
        if selected_backend == "auto":
            if (
                solver in {"block_sparse", "matrix_free"}
                or 2 * merged_size**2 > pair_dense_max_elements
            ):
                selected_backend = (
                    "packed"
                    if (
                        merged_size >= packed_min_size
                        and self._pair_action_mask(site, plan) is not None
                    )
                    else "action"
                )
            else:
                selected_backend = "dense"
        block_problem = None
        pair_metric = None
        pair_effective = None
        metric_blocks = None
        if selected_backend == "dense":
            pair_metric, pair_effective = self.pair_local_operators(
                site,
                environment=environment,
            )
            stored_elements = int(pair_metric.size + pair_effective.size)

            def metric_action(vector):
                return pair_metric @ vector

            def effective_action(vector):
                return pair_effective @ vector

        elif selected_backend in {"block", "action", "packed"}:
            if selected_backend == "action":
                block_problem = self.pair_local_action_block_problem(
                    site,
                    environment=environment,
                )
            elif selected_backend == "packed":
                block_problem = self.pair_local_packed_action_block_problem(
                    site,
                    environment=environment,
                )
            else:
                block_problem = self.pair_local_block_problem(
                    site,
                    environment=environment,
                )
            metric_action = block_problem.metric.matvec
            effective_action = block_problem.hamiltonian.matvec
            metric_blocks = tuple(
                block_problem.metric.blocks[(block, block)]
                for block in range(block_problem.layout.nblocks)
            )
            stored_elements = int(block_problem.stored_elements)
        incumbent_energy = self._pair_rayleigh_actions(
            warm,
            metric_action,
            effective_action,
        )
        pair_energy_before = float(incumbent_energy)
        energy_before = (
            pair_energy_before
            if global_energy_before is None
            else global_energy_before
        )
        merged_energy_history = []
        factor_energy_history = []
        local_update = None
        merged_diagnostics = None
        overlap = tuple(
            sorted(
                set(self.physical_groups[site])
                & set(self.physical_groups[following])
            )
        )
        conditional_ranks = ()
        truncation_error = float("nan")
        projection_error = float("nan")
        factor_sweeps = 0
        factor_accepted_updates = 0
        selected_start = "old"
        attempted_random_starts = 0
        completed_cycles = 0
        maximum_cycles = 1 if split_strategy == "svd" else outer_cycles
        certified_root = None
        for cycle in range(maximum_cycles):
            completed_cycles = cycle + 1
            reused_certified_root = certified_root is not None
            if reused_certified_root:
                (
                    merged_energy,
                    merged_vector,
                    cycle_local_update,
                    cycle_diagnostics,
                ) = certified_root
                warm_energy = self._pair_rayleigh_actions(
                    warm,
                    metric_action,
                    effective_action,
                )
                cycle_diagnostics = replace(
                    cycle_diagnostics,
                    method=f"cached_{cycle_diagnostics.method}",
                    attempts=("cached_certified_root",),
                    warm_energy=warm_energy,
                    upper_bound_gap=float(merged_energy - warm_energy),
                )
                cycle_local_update = replace(
                    cycle_local_update,
                    solver=cycle_diagnostics.method,
                    energy_before=warm_energy,
                    accepted=bool(
                        merged_energy
                        <= warm_energy
                        + 512.0
                        * np.finfo(float).eps
                        * max(1.0, abs(warm_energy))
                    ),
                    message="reused exact certified merged-pair root",
                )
            elif block_problem is None:
                (
                    merged_energy,
                    merged_vector,
                    cycle_local_update,
                    cycle_diagnostics,
                ) = self._solve_verified_pair_pencil(
                    site,
                    pair_metric,
                    pair_effective,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=merged_dense_fallback_dim,
                    metric_support=metric_support,
                )
            else:
                (
                    merged_energy,
                    merged_vector,
                    cycle_local_update,
                    cycle_diagnostics,
                ) = self._solve_verified_pair_block_pencil(
                    site,
                    block_problem,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=merged_dense_fallback_dim,
                    metric_support=metric_support,
                    preconditioner=preconditioner,
                    block_size=block_size,
                    recycle=recycle,
                    recycle_min_size=recycle_min_size,
                )
            if (
                not reused_certified_root
                and cycle_diagnostics.lowest_root_certified
                and cycle_diagnostics.method
                in {"dense_certified", "conditional_blocks_certified"}
            ):
                certified_root = (
                    float(merged_energy),
                    np.asarray(merged_vector).copy(),
                    cycle_local_update,
                    cycle_diagnostics,
                )
            local_update = cycle_local_update
            merged_diagnostics = cycle_diagnostics
            merged_energy_history.append(float(merged_energy))
            split_target = merged_vector.reshape(plan.merged_shape).copy()

            warm_vector = warm.reshape(-1)
            target_vector = split_target.reshape(-1)
            warm_norm = float(
                np.real(np.vdot(warm_vector, metric_action(warm_vector)))
            )
            target_norm = float(
                np.real(np.vdot(target_vector, metric_action(target_vector)))
            )
            if warm_norm > np.finfo(float).tiny and target_norm > np.finfo(float).tiny:
                split_target *= np.sqrt(warm_norm / target_norm)
                target_vector = split_target.reshape(-1)
            overlap_value = np.vdot(target_vector, metric_action(warm_vector))
            if abs(overlap_value) > 256.0 * np.finfo(float).eps:
                split_target = split_target * (
                    overlap_value / abs(overlap_value)
                )

            (
                euclidean_left,
                euclidean_right,
                overlap,
                conditional_ranks,
                truncation_error,
            ) = self._split_merged_pair_tensor(
                site,
                split_target,
                plan.union_sites,
            )
            use_svd_candidate = split_strategy in {"svd", "hybrid"}
            if use_svd_candidate:
                candidate_left = euclidean_left
                candidate_right = euclidean_right
                candidate_energy = self._pair_rayleigh_actions(
                    self._merge_pair_factors(
                        site,
                        plan.union_sites,
                        candidate_left,
                        candidate_right,
                    ).reshape(-1),
                    metric_action,
                    effective_action,
                )
                projection_error = float("nan")
                selected_start = "svd"
            tolerance = split_energy_tol * max(1.0, abs(incumbent_energy))
            if (
                split_strategy == "hybrid"
                and (
                    not np.isfinite(candidate_energy)
                    or candidate_energy > incumbent_energy + tolerance
                )
            ):
                use_svd_candidate = False
            if not use_svd_candidate:
                (
                    candidate_left,
                    candidate_right,
                    cycle_projection_error,
                    cycle_factor_sweeps,
                    cycle_factor_updates,
                    cycle_start,
                    cycle_random_starts,
                    candidate_energy,
                ) = self._variational_split_merged_pair(
                    site,
                    split_target,
                    plan.union_sites,
                    pair_metric,
                    pair_effective,
                    euclidean_left,
                    euclidean_right,
                    metric_tol=split_metric_tol,
                    metric_sweeps=split_metric_sweeps,
                    variational_sweeps=split_variational_sweeps,
                    random_starts=split_random_starts,
                    random_seed=(
                        None
                        if split_random_seed is None
                        else split_random_seed + 104729 * site + cycle
                    ),
                    energy_tol=split_energy_tol,
                    incumbent_left=incumbent_left,
                    incumbent_right=incumbent_right,
                    factor_solver=factor_solver,
                    factor_eig_tol=eig_tol,
                    factor_maxiter=min(maxiter or 256, 256),
                    factor_max_subspace=min(max_subspace, 32),
                    metric_action=metric_action,
                    effective_action=effective_action,
                    metric_blocks=metric_blocks,
                )
                projection_error = cycle_projection_error
                factor_sweeps += cycle_factor_sweeps
                factor_accepted_updates += cycle_factor_updates
                attempted_random_starts += cycle_random_starts
                if selected_start == "old" and cycle_start != "old":
                    # Preserve the seed that first escaped the incumbent.
                    # Later correction cycles commonly converge from their
                    # updated incumbent and would otherwise erase that useful
                    # diagnostic with the uninformative label ``old``.
                    selected_start = cycle_start
            factor_energy_history.append(float(candidate_energy))
            improvement = incumbent_energy - candidate_energy
            if split_strategy == "svd" and np.isfinite(candidate_energy):
                # Keep the Euclidean-SVD mode as an explicit diagnostic path:
                # install its truncation candidate and let the fresh global
                # contraction below accept or roll it back.  The variational
                # path remains monotone inside every outer correction cycle.
                incumbent_left = np.asarray(candidate_left).copy()
                incumbent_right = np.asarray(candidate_right).copy()
                incumbent_energy = float(candidate_energy)
                warm = self._merge_pair_factors(
                    site,
                    plan.union_sites,
                    incumbent_left,
                    incumbent_right,
                )
            elif (
                np.isfinite(candidate_energy)
                and candidate_energy <= incumbent_energy + tolerance
            ):
                incumbent_left = np.asarray(candidate_left).copy()
                incumbent_right = np.asarray(candidate_right).copy()
                incumbent_energy = float(candidate_energy)
                warm = self._merge_pair_factors(
                    site,
                    plan.union_sites,
                    incumbent_left,
                    incumbent_right,
                )
            if split_strategy == "svd" or improvement <= tolerance:
                break

        self.tensors[site] = incumbent_left
        self.tensors[following] = incumbent_right
        if verify_global:
            try:
                energy_after = float(self.expectation())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                energy_after = float("inf")
        else:
            energy_after = float(incumbent_energy)
        global_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(energy_before),
        )
        attempted_energy = float(energy_after)
        accepted = bool(
            np.isfinite(energy_after)
            and energy_after <= energy_before + global_tolerance
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
            raw_merged_dim=merged_size,
            old_bond_dimension=self._bond_dims()[following],
            temporary_bond_dimension=plan.identity_tensor.shape[0],
            conditional_ranks=conditional_ranks,
            relative_truncation_error=float(truncation_error),
            energy_before=energy_before,
            merged_energy=min(merged_energy_history),
            attempted_energy=attempted_energy,
            energy=float(energy_after),
            accepted=accepted,
            local_update=local_update,
            split_strategy=split_strategy,
            selected_start=selected_start,
            metric_projection_error=float(projection_error),
            factor_sweeps=factor_sweeps,
            factor_accepted_updates=factor_accepted_updates,
            factor_random_starts=attempted_random_starts,
            merged_solve=merged_diagnostics,
            outer_cycles=completed_cycles,
            merged_energy_history=tuple(merged_energy_history),
            factor_energy_history=tuple(factor_energy_history),
            pair_operator_backend=selected_backend,
            pair_operator_stored_elements=stored_elements,
        )

    def optimize_block(
        self,
        start,
        stop,
        *,
        environment=None,
        operator_backend="action",
        verify_global=True,
        metric_tol=1.0e-12,
        eig_tol=1.0e-10,
        maxiter=1600,
        max_subspace=96,
        merged_dense_fallback_dim=2048,
        metric_support="regularized",
    ) -> FrontierBlockUpdate:
        r"""Optimize one contiguous block as a single matrix-free tensor.

        The block is merged exactly, optimized in its conditional physical
        blocks, and retracted through the subclass's symmetry-preserving block
        factorization.  ``operator_backend="action"`` stores metric blocks but
        applies the Hamiltonian lazily through the frontier.
        """
        if not self._has_charge_resolved_block_split:
            raise TypeError(
                "direct block optimization currently requires a "
                "charge-resolved block split."
            )
        start = int(start)
        stop = int(stop)
        plan = self._block_plan(start, stop)
        environment = self._resolved_block_environment(
            start,
            stop,
            environment,
        )
        operator_backend = str(operator_backend).lower().replace("-", "_")
        if operator_backend in {"matrix_free", "lazy"}:
            operator_backend = "action"
        if operator_backend not in {"action", "dense"}:
            raise ValueError("operator_backend must be 'action' or 'dense'.")
        metric_tol = float(metric_tol)
        eig_tol = float(eig_tol)
        max_subspace = int(max_subspace)
        merged_dense_fallback_dim = int(merged_dense_fallback_dim)
        metric_support = str(metric_support).lower().replace("-", "_")
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(eig_tol) or eig_tol < 0.0:
            raise ValueError("eig_tol must be finite and nonnegative.")
        if maxiter is not None:
            maxiter = int(maxiter)
            if maxiter < 1:
                raise ValueError("maxiter must be positive or None.")
        if max_subspace < 2:
            raise ValueError("max_subspace must be at least two.")
        if merged_dense_fallback_dim < 1:
            raise ValueError("merged_dense_fallback_dim must be positive.")
        if metric_support not in {"regularized", "numerical"}:
            raise ValueError(
                "metric_support must be 'regularized' or 'numerical'."
            )

        global_energy_before = (
            float(self.expectation()) if verify_global else None
        )
        old_tensors = [
            self.tensors[site].copy() for site in plan.sites
        ]
        warm, union_sites = self._merged_block_tensor(start, stop)
        if operator_backend == "dense":
            metric, effective = self.block_local_operators(
                start,
                stop,
                environment=environment,
            )
            stored_elements = int(metric.size + effective.size)

            def metric_action(vector):
                return metric @ vector

            def effective_action(vector):
                return effective @ vector

            merged_energy, merged_vector, local_update, diagnostics = (
                self._solve_verified_pair_pencil(
                    start,
                    metric,
                    effective,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=merged_dense_fallback_dim,
                    metric_support=metric_support,
                )
            )
        else:
            problem = self.block_local_action_problem(
                start,
                stop,
                environment=environment,
            )
            stored_elements = int(problem.stored_elements)
            metric_action = problem.metric.matvec
            effective_action = problem.hamiltonian.matvec
            merged_energy, merged_vector, local_update, diagnostics = (
                self._solve_verified_pair_block_pencil(
                    start,
                    problem,
                    warm,
                    metric_tol=metric_tol,
                    eig_tol=eig_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    dense_fallback_dim=merged_dense_fallback_dim,
                    metric_support=metric_support,
                )
            )
        local_energy_before = self._pair_rayleigh_actions(
            warm.reshape(-1),
            metric_action,
            effective_action,
        )

        split_target = np.asarray(merged_vector).reshape(plan.merged_shape)
        factors, conditional_ranks, truncation_error = (
            self._split_merged_block_tensor(
                start,
                stop,
                split_target,
                union_sites,
            )
        )
        for site, factor in zip(plan.sites, factors):
            self.tensors[site] = factor

        if verify_global:
            try:
                energy_after = float(self.expectation())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                energy_after = float("inf")
            energy_before = float(global_energy_before)
        else:
            candidate, _union_sites = self._merged_block_tensor(start, stop)
            energy_before = float(local_energy_before)
            energy_after = self._pair_rayleigh_actions(
                candidate.reshape(-1),
                metric_action,
                effective_action,
            )
        tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(energy_before),
        )
        attempted_energy = float(energy_after)
        accepted = bool(
            np.isfinite(energy_after)
            and energy_after <= energy_before + tolerance
        )
        if not accepted:
            for site, tensor in zip(plan.sites, old_tensors):
                self.tensors[site] = tensor
            energy_after = energy_before
        else:
            self.converged = False
        self.energy = float(energy_after)
        return FrontierBlockUpdate(
            sites=plan.sites,
            raw_merged_dim=int(warm.size),
            energy_before=float(energy_before),
            merged_energy=float(merged_energy),
            attempted_energy=attempted_energy,
            energy=float(energy_after),
            accepted=accepted,
            relative_truncation_error=float(truncation_error),
            conditional_ranks=conditional_ranks,
            local_update=local_update,
            merged_solve=diagnostics,
            operator_backend=operator_backend,
            operator_stored_elements=stored_elements,
        )

    @staticmethod
    def _block_environment_checkpoint_cuts(blocks, interval):
        """Return checkpoint cuts aligned with logical block boundaries."""
        interval = int(interval)
        if interval < 1:
            raise ValueError("block checkpoint interval must be positive.")
        boundaries = (blocks[0][0],) + tuple(
            block[-1] + 1 for block in blocks
        )
        indices = list(range(0, len(blocks), interval))
        if indices[-1] != len(blocks):
            indices.append(len(blocks))
        return tuple(boundaries[index] for index in indices)

    def fixed_block_environment_cache_elements(
        self,
        blocks,
        *,
        mode="checkpointed",
        interval=None,
    ):
        """Estimate fixed-side storage for a logical-block sweep."""
        blocks = tuple(tuple(int(site) for site in block) for block in blocks)
        mode = str(mode).lower().replace("-", "_")
        if mode in {"checkpoint", "recompute"}:
            mode = "checkpointed"
        if mode == "full":
            return self.cached_environment_elements
        if mode != "checkpointed":
            raise ValueError("mode must be 'checkpointed' or 'full'.")
        if interval is None:
            interval = max(1, int(np.ceil(np.sqrt(len(blocks)))))
        cuts = self._block_environment_checkpoint_cuts(blocks, interval)
        engines = (self._norm_frontier, self._hamiltonian_frontier)
        boundaries = tuple(block[-1] + 1 for block in blocks[:-1])
        checkpoints = sum(
            self._dense_message_elements(engine, cut)
            for engine in engines
            for cut in cuts
        )
        interiors = max(
            (
                sum(
                    self._dense_message_elements(engine, cut)
                    for engine in engines
                    for cut in boundaries
                    if start < cut < stop
                )
                for start, stop in zip(cuts[:-1], cuts[1:])
            ),
            default=0,
        )
        return int(checkpoints + interiors)

    def _recompute_block_environment_segment(
        self,
        frontier,
        *,
        direction,
        blocks,
        checkpoint,
    ):
        """Recompute only logical-boundary messages inside one segment."""
        blocks = tuple(blocks)
        if direction == "left":
            message = checkpoint
            messages = {blocks[0][0]: message}
            for block in blocks:
                for site in block:
                    message = frontier.advance_left(
                        message,
                        self.tensors,
                        site,
                    )
                messages[block[-1] + 1] = message
            return messages
        if direction == "right":
            message = checkpoint
            messages = {blocks[-1][-1] + 1: message}
            for block in reversed(blocks):
                for site in reversed(block):
                    message = frontier.advance_right(
                        message,
                        self.tensors,
                        site,
                    )
                messages[block[0]] = message
            return messages
        raise ValueError("direction must be 'left' or 'right'.")

    def run_blocks(
        self,
        blocks,
        *,
        nsweeps=2,
        sweep_offset=0,
        tol=1.0e-10,
        environment_cache="checkpointed",
        environment_checkpoint_interval=None,
        verbose=False,
        **block_options,
    ):
        """Sweep over direct blocks with checkpointed fixed environments."""
        blocks = tuple(tuple(int(site) for site in block) for block in blocks)
        if not blocks or any(len(block) < 2 for block in blocks):
            raise ValueError("blocks must contain groups of at least two sites.")
        if tuple(site for block in blocks for site in block) != tuple(
            range(len(self.dims))
        ):
            raise ValueError(
                "blocks must partition the chain into consecutive ordered groups."
            )
        if any(
            block != tuple(range(block[0], block[-1] + 1))
            for block in blocks
        ):
            raise ValueError("every block must be consecutive.")
        nsweeps = int(nsweeps)
        sweep_offset = int(sweep_offset)
        tol = float(tol)
        if nsweeps < 0 or sweep_offset < 0:
            raise ValueError("sweep counts must be nonnegative.")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        if "environment" in block_options:
            raise TypeError("run_blocks constructs block environments.")
        verify_global = bool(block_options.pop("verify_global", False))
        selected_operator_backend = str(
            block_options.get("operator_backend", "action")
        ).lower().replace("-", "_")
        direct_action_only = selected_operator_backend in {
            "action",
            "matrix_free",
            "lazy",
        }
        environment_cache = str(environment_cache).lower().replace("-", "_")
        if environment_cache in {"checkpoint", "recompute"}:
            environment_cache = "checkpointed"
        if environment_cache not in {"checkpointed", "full"}:
            raise ValueError(
                "environment_cache must be 'checkpointed' or 'full'."
            )
        if environment_checkpoint_interval is None:
            environment_checkpoint_interval = min(
                range(1, len(blocks) + 1),
                key=lambda interval: self.fixed_block_environment_cache_elements(
                    blocks,
                    interval=interval,
                ),
            )
        else:
            environment_checkpoint_interval = int(
                environment_checkpoint_interval
            )
            if environment_checkpoint_interval < 1:
                raise ValueError(
                    "environment_checkpoint_interval must be positive."
                )
        checkpoint_cuts = self._block_environment_checkpoint_cuts(
            blocks,
            environment_checkpoint_interval,
        )
        block_by_start = {block[0]: block for block in blocks}
        block_by_stop = {block[-1] + 1: block for block in blocks}

        previous = float(self.expectation())
        self.energy = previous
        history = []
        self.history = history
        self.converged = False
        for sweep in range(nsweeps):
            direction = sweep_offset + sweep
            updates = []
            if direction % 2 == 0:
                if environment_cache == "full":
                    norm_right = self._norm_frontier.build_right(self.tensors)
                    hamiltonian_right = (
                        self._hamiltonian_frontier.build_right(self.tensors)
                    )
                    right_checkpoints = None
                elif environment_cache == "checkpointed":
                    norm_right = hamiltonian_right = None
                    right_checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="right",
                            cuts=checkpoint_cuts,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="right",
                            cuts=checkpoint_cuts,
                        ),
                    )
                else:
                    norm_right = hamiltonian_right = right_checkpoints = None
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                for segment_start, segment_stop in zip(
                    checkpoint_cuts[:-1],
                    checkpoint_cuts[1:],
                ):
                    segment_blocks = tuple(
                        block
                        for block in blocks
                        if segment_start <= block[0] < segment_stop
                    )
                    if right_checkpoints is not None:
                        norm_right = self._recompute_block_environment_segment(
                            self._norm_frontier,
                            direction="right",
                            blocks=segment_blocks,
                            checkpoint=right_checkpoints[0][segment_stop],
                        )
                        hamiltonian_right = (
                            self._recompute_block_environment_segment(
                                self._hamiltonian_frontier,
                                direction="right",
                                blocks=segment_blocks,
                                checkpoint=right_checkpoints[1][segment_stop],
                            )
                        )
                    start = segment_start
                    while start < segment_stop:
                        block = block_by_start[start]
                        stop = block[-1] + 1
                        environment = (
                            self._block_environment_from_outer_messages(
                                start,
                                stop,
                                moving_norm,
                                norm_right[stop],
                                moving_hamiltonian,
                                hamiltonian_right[stop],
                                action_only=direct_action_only,
                            )
                        )
                        updates.append(
                            self.optimize_block(
                                start,
                                stop,
                                environment=environment,
                                verify_global=verify_global,
                                **block_options,
                            )
                        )
                        for site in block:
                            moving_norm = self._norm_frontier.advance_left(
                                moving_norm,
                                self.tensors,
                                site,
                            )
                            moving_hamiltonian = (
                                self._hamiltonian_frontier.advance_left(
                                    moving_hamiltonian,
                                    self.tensors,
                                    site,
                                )
                            )
                        start = stop
                    environment = None
                    if right_checkpoints is not None:
                        norm_right = hamiltonian_right = None
            else:
                if environment_cache == "full":
                    norm_left = self._norm_frontier.build_left(self.tensors)
                    hamiltonian_left = (
                        self._hamiltonian_frontier.build_left(self.tensors)
                    )
                    left_checkpoints = None
                else:
                    norm_left = hamiltonian_left = None
                    left_checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="left",
                            cuts=checkpoint_cuts,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="left",
                            cuts=checkpoint_cuts,
                        ),
                    )
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                for segment_start, segment_stop in reversed(
                    tuple(zip(checkpoint_cuts[:-1], checkpoint_cuts[1:]))
                ):
                    segment_blocks = tuple(
                        block
                        for block in blocks
                        if segment_start <= block[0] < segment_stop
                    )
                    if left_checkpoints is not None:
                        norm_left = self._recompute_block_environment_segment(
                            self._norm_frontier,
                            direction="left",
                            blocks=segment_blocks,
                            checkpoint=left_checkpoints[0][segment_start],
                        )
                        hamiltonian_left = (
                            self._recompute_block_environment_segment(
                                self._hamiltonian_frontier,
                                direction="left",
                                blocks=segment_blocks,
                                checkpoint=left_checkpoints[1][segment_start],
                            )
                        )
                    stop = segment_stop
                    while stop > segment_start:
                        block = block_by_stop[stop]
                        start = block[0]
                        environment = (
                            self._block_environment_from_outer_messages(
                                start,
                                stop,
                                norm_left[start],
                                moving_norm,
                                hamiltonian_left[start],
                                moving_hamiltonian,
                                action_only=direct_action_only,
                            )
                        )
                        updates.append(
                            self.optimize_block(
                                start,
                                stop,
                                environment=environment,
                                verify_global=verify_global,
                                **block_options,
                            )
                        )
                        for site in reversed(block):
                            moving_norm = self._norm_frontier.advance_right(
                                moving_norm,
                                self.tensors,
                                site,
                            )
                            moving_hamiltonian = (
                                self._hamiltonian_frontier.advance_right(
                                    moving_hamiltonian,
                                    self.tensors,
                                    site,
                                )
                            )
                        stop = start
                    environment = None
                    if left_checkpoints is not None:
                        norm_left = hamiltonian_left = None

            energy = float(self.expectation())
            record = {
                "sweep": direction,
                "direction": "left-to-right" if direction % 2 == 0 else "right-to-left",
                "energy": energy,
                "updates": tuple(updates),
                "environment_cache": environment_cache,
                "environment_checkpoint_interval": (
                    environment_checkpoint_interval
                    if environment_cache == "checkpointed"
                    else None
                ),
            }
            history.append(record)
            if verbose:
                print(
                    f"block sweep {direction + 1}: energy={energy:.12f}, "
                    f"accepted={sum(update.accepted for update in updates)}/"
                    f"{len(updates)}"
                )
            improvement = previous - energy
            previous = energy
            self.energy = energy
            if abs(improvement) < tol:
                self.converged = True
                break
        return self

    def pair_residual_scores(self, *, energy=None):
        """Return normalized residual indicators for every adjacent pair.

        The site residuals use exact left/right frontier messages and the
        matrix-free local actions.  A pair receives the Euclidean combination
        of its two site indicators, which is cheap enough to refresh before a
        selective two-site pass without constructing any pair Hamiltonian.
        """
        nsites = len(self.dims)
        if nsites < 2:
            return ()
        if energy is None:
            energy = self.expectation()
        energy = float(energy)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
        moving_norm = self._norm_frontier.left_boundary()
        moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
        tiny = np.finfo(float).tiny
        site_scores = []
        for site, tensor in enumerate(self.tensors):
            vector = np.asarray(tensor).reshape(-1)
            prepare_action = getattr(
                self._hamiltonian_frontier,
                "prepare_hole_action",
                None,
            )
            prepared = (
                prepare_action(
                    site,
                    moving_hamiltonian,
                    hamiltonian_right[site + 1],
                )
                if prepare_action is not None
                else None
            )
            hamiltonian_vector = (
                prepared(vector)
                if prepared is not None
                else self._hamiltonian_frontier.hole_action(
                    site,
                    moving_hamiltonian,
                    hamiltonian_right[site + 1],
                    vector,
                )
            )
            metric_vector = self._norm_frontier.hole_action(
                site,
                moving_norm,
                norm_right[site + 1],
                vector,
            )
            residual = hamiltonian_vector - energy * metric_vector
            mask = self._local_action_mask(site)
            if mask is not None:
                residual = np.where(
                    np.asarray(mask, dtype=bool).reshape(-1),
                    residual,
                    0,
                )
            scale = max(
                float(np.linalg.norm(hamiltonian_vector)),
                abs(energy) * float(np.linalg.norm(metric_vector)),
                tiny,
            )
            site_scores.append(float(np.linalg.norm(residual) / scale))
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
        return tuple(
            float(np.hypot(site_scores[site], site_scores[site + 1]))
            for site in range(nsites - 1)
        )

    def pair_residual_certificates(self, *, energy=None):
        r"""Return exact normalized residuals of every adjacent-pair problem.

        Unlike :meth:`pair_residual_scores`, which combines inexpensive
        one-site indicators for scheduling, this method applies each merged
        pair Hamiltonian and metric once.  It therefore certifies stationarity
        of the complete two-site tangent space without running a Davidson
        solve at every pair.
        """

        nsites = len(self.dims)
        if nsites < 2:
            return ()
        if not self.norm_contraction_is_exact:
            raise ValueError("pair residual certification requires an exact norm.")
        if not self.hamiltonian_action_is_hermitian:
            raise ValueError(
                "pair residual certification requires a Hermitian Hamiltonian action."
            )
        if energy is None:
            energy = self.expectation()
        energy = float(energy)
        norm_right = self._norm_frontier.build_right(self.tensors)
        hamiltonian_right = self._hamiltonian_frontier.build_right(self.tensors)
        moving_norm = self._norm_frontier.left_boundary()
        moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
        tiny = np.finfo(float).tiny
        scores = []
        for site in range(nsites - 1):
            environment = self._pair_environment_from_outer_messages(
                site,
                moving_norm,
                norm_right[site + 2],
                moving_hamiltonian,
                hamiltonian_right[site + 2],
            )
            plan = self._pair_plan(site)
            vector = self._merge_pair_factors(
                site,
                plan.union_sites,
                self.tensors[site],
                self.tensors[site + 1],
            ).reshape(-1)
            mask = self._pair_action_mask(site, plan)
            flat_mask = (
                None
                if mask is None
                else np.asarray(mask, dtype=bool).reshape(-1)
            )
            trial = vector if flat_mask is None else np.where(flat_mask, vector, 0)
            prepare_hamiltonian = getattr(
                plan.hamiltonian_engine,
                "prepare_hole_action",
                None,
            )
            prepared_hamiltonian = (
                prepare_hamiltonian(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                )
                if prepare_hamiltonian is not None
                else None
            )
            verification_action = getattr(
                prepared_hamiltonian,
                "verify",
                None,
            )
            hamiltonian_vector = (
                verification_action(trial)
                if verification_action is not None
                else prepared_hamiltonian(trial)
                if prepared_hamiltonian is not None
                else plan.hamiltonian_engine.hole_action(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    trial,
                )
            )
            metric_vector = plan.norm_engine.hole_action(
                site,
                environment.norm_left,
                environment.norm_right,
                trial,
            )
            if flat_mask is not None:
                hamiltonian_vector = np.where(
                    flat_mask,
                    hamiltonian_vector,
                    0,
                )
                metric_vector = np.where(flat_mask, metric_vector, 0)
            residual = hamiltonian_vector - energy * metric_vector
            scale = max(
                float(np.linalg.norm(hamiltonian_vector)),
                abs(energy) * float(np.linalg.norm(metric_vector)),
                tiny,
            )
            scores.append(float(np.linalg.norm(residual) / scale))
            # A certificate touches every adjacent-pair topology once.  Do
            # not retain one contraction workspace per cut: the plans stay
            # cached, but their potentially large temporary expressions are
            # rebuilt only when that pair is selected for optimization.
            for engine in (plan.norm_engine, plan.hamiltonian_engine):
                clear = getattr(engine, "clear_contraction_plans", None)
                if clear is not None:
                    clear()
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
        return tuple(scores)

    @staticmethod
    def _select_residual_pairs(
        scores,
        *,
        fraction,
        minimum,
        maximum,
        mandatory=(),
    ):
        """Select the smallest high-residual set plus mandatory stale pairs."""
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or np.any(~np.isfinite(scores)):
            raise ValueError("pair residual scores must be a finite vector.")
        weights = np.maximum(scores, 0.0) ** 2
        order = sorted(range(scores.size), key=lambda pair: (-scores[pair], pair))
        selected = set(int(pair) for pair in mandatory)
        total = float(np.sum(weights))
        captured = float(np.sum(weights[list(selected)])) if selected else 0.0
        for pair in order:
            if pair in selected:
                continue
            if len(selected) >= minimum and (
                total <= np.finfo(float).tiny
                or captured >= fraction * total
            ):
                break
            if maximum is not None and len(selected) >= maximum:
                break
            selected.add(pair)
            captured += float(weights[pair])
        if len(selected) < minimum:
            for pair in order:
                selected.add(pair)
                if len(selected) >= minimum:
                    break
        captured_fraction = 1.0 if total <= np.finfo(float).tiny else captured / total
        return selected, float(min(captured_fraction, 1.0))

    def run_two_site(
        self,
        *,
        nsweeps: int = 2,
        sweep_offset: int = 0,
        tol: float = 1.0e-10,
        solver="verified",
        adaptive_solver: bool = False,
        eig_tol_initial: float = 1.0e-5,
        pair_selection="all",
        max_pairs: int | None = None,
        residual_fraction: float = 0.9,
        min_pairs: int = 1,
        coverage_every: int = 4,
        reuse_residual_scores: bool = True,
        certify_residual: bool = False,
        residual_tol: float | None = None,
        certify_every: int = 1,
        verify_pair_energies: bool = False,
        verbose: bool = False,
        **pair_options,
    ):
        r"""Optimize overlapping adjacent pairs with directional caches.

        Each directional sweep builds the fixed-side norm and Hamiltonian
        frontier messages once.  The moving-side messages are advanced after
        every accepted or rejected pair update, while cached pair contraction
        plans are reused across sweeps.  With ``verify_pair_energies=False``
        the exact pair Rayleigh quotient supplies the per-pair variational
        gate; the completed endpoint messages independently verify the energy
        once per directional sweep. Convergence is assessed only after a
        complete pair of directional sweeps, so an LR/RL cycle is never
        mistaken for convergence from one favorable endpoint.

        ``adaptive_solver=True`` tightens the pair Davidson tolerance from
        ``eig_tol_initial`` according to the preceding complete-cycle energy
        gain. ``pair_selection='residual'`` updates the smallest set capturing
        ``residual_fraction`` of squared residual weight, bounded by
        ``min_pairs`` and ``max_pairs``. Scores are shared by the two
        directions of a cycle, and ``coverage_every`` forces stale pairs back
        into the schedule. ``certify_residual=True`` replaces an expensive
        final optimize-every-pair pass by one exact merged-pair residual
        application per bond. Selective cycles may then converge without
        visiting every pair when the full two-site tangent residual is below
        ``residual_tol``.
        """
        nsweeps = int(nsweeps)
        sweep_offset = int(sweep_offset)
        tol = float(tol)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        if sweep_offset < 0:
            raise ValueError("sweep_offset must be nonnegative.")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        adaptive_solver = bool(adaptive_solver)
        target_eig_tol = float(pair_options.get("eig_tol", 1.0e-10))
        eig_tol_initial = float(eig_tol_initial)
        if not np.isfinite(target_eig_tol) or target_eig_tol < 0.0:
            raise ValueError("eig_tol must be finite and nonnegative.")
        if (
            not np.isfinite(eig_tol_initial)
            or eig_tol_initial < target_eig_tol
        ):
            raise ValueError(
                "eig_tol_initial must be finite and at least eig_tol."
            )
        pair_selection = str(pair_selection).lower().replace("-", "_")
        if pair_selection not in {"all", "residual"}:
            raise ValueError("pair_selection must be 'all' or 'residual'.")
        if max_pairs is not None:
            max_pairs = int(max_pairs)
            if max_pairs < 1:
                raise ValueError("max_pairs must be positive.")
        residual_fraction = float(residual_fraction)
        min_pairs = int(min_pairs)
        coverage_every = int(coverage_every)
        reuse_residual_scores = bool(reuse_residual_scores)
        certify_residual = bool(certify_residual)
        residual_tol = (
            max(8.0 * np.sqrt(np.finfo(float).eps), 10.0 * target_eig_tol)
            if residual_tol is None
            else float(residual_tol)
        )
        certify_every = int(certify_every)
        if not np.isfinite(residual_fraction) or not 0.0 < residual_fraction <= 1.0:
            raise ValueError("residual_fraction must lie in (0, 1].")
        if min_pairs < 1:
            raise ValueError("min_pairs must be positive.")
        if coverage_every < 0:
            raise ValueError("coverage_every must be nonnegative.")
        if not np.isfinite(residual_tol) or residual_tol < 0.0:
            raise ValueError("residual_tol must be finite and nonnegative.")
        if certify_every < 1:
            raise ValueError("certify_every must be positive.")
        if "environment" in pair_options or "verify_global" in pair_options:
            raise TypeError(
                "run_two_site constructs pair environments and controls "
                "endpoint verification."
            )
        if nsweeps and len(self.dims) < 2:
            raise ValueError("two-site sweeps require at least two sites.")
        if nsweeps:
            self._pair_plan(0)
        if nsweeps and not self.norm_contraction_is_exact:
            raise ValueError("two-site sweeps require an exact norm contraction.")
        if nsweeps and not self.hamiltonian_action_is_hermitian:
            raise ValueError(
                "two-site sweeps require an exact or explicitly Hermitian "
                "Hamiltonian action."
            )

        previous = float(self.expectation())
        self.energy = previous
        sweep_history = []
        self.history = sweep_history
        self.converged = False
        nsites = len(self.dims)
        cycle_start_energy = previous
        cycle_endpoints_accepted = True
        cycle_endpoints_stationary = True
        cycle_started = False
        cycle_visited_pairs = set()
        last_relative_cycle_gain = float("inf")
        cached_residual_scores = None
        for pair in range(max(0, nsites - 1)):
            self._pair_last_visited.setdefault(pair, sweep_offset)
        for sweep in range(nsweeps):
            directional_sweep = sweep_offset + sweep
            if directional_sweep % 2 == 0:
                cycle_start_energy = previous
                cycle_endpoints_accepted = True
                cycle_endpoints_stationary = True
                cycle_started = True
                cycle_visited_pairs = set()
            active_pair_options = dict(pair_options)
            active_eig_tol = target_eig_tol
            if adaptive_solver:
                gain_tolerance = (
                    eig_tol_initial
                    if not np.isfinite(last_relative_cycle_gain)
                    else 0.1 * last_relative_cycle_gain
                )
                active_eig_tol = max(
                    target_eig_tol,
                    min(eig_tol_initial, gain_tolerance),
                )
            active_pair_options["eig_tol"] = active_eig_tol
            residual_scores = None
            residual_scores_reused = False
            residual_captured_fraction = 1.0
            selected_pairs = set(range(nsites - 1))
            if pair_selection == "residual" and nsites > 1:
                residual_scores_reused = bool(
                    reuse_residual_scores
                    and directional_sweep % 2 == 1
                    and cached_residual_scores is not None
                )
                residual_scores = (
                    cached_residual_scores
                    if residual_scores_reused
                    else self.pair_residual_scores(energy=previous)
                )
                if not residual_scores_reused:
                    cached_residual_scores = residual_scores
                mandatory = (
                    tuple(
                        pair
                        for pair in range(nsites - 1)
                        if directional_sweep - self._pair_last_visited[pair]
                        >= coverage_every
                    )
                    if coverage_every
                    else ()
                )
                selected_pairs, residual_captured_fraction = (
                    self._select_residual_pairs(
                        residual_scores,
                        fraction=residual_fraction,
                        minimum=min(min_pairs, nsites - 1),
                        maximum=(
                            None
                            if max_pairs is None
                            else min(max_pairs, nsites - 1)
                        ),
                        mandatory=mandatory,
                    )
                )
            cycle_visited_pairs.update(selected_pairs)
            sweep_tensors = [tensor.copy() for tensor in self.tensors]
            updates = []
            if directional_sweep % 2 == 0:
                norm_right = self._norm_frontier.build_right(self.tensors)
                hamiltonian_right = self._hamiltonian_frontier.build_right(
                    self.tensors
                )
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                for site in range(nsites - 1):
                    if site in selected_pairs:
                        environment = self._pair_environment_from_outer_messages(
                            site,
                            moving_norm,
                            norm_right[site + 2],
                            moving_hamiltonian,
                            hamiltonian_right[site + 2],
                        )
                        updates.append(
                            self.optimize_two_sites(
                                site,
                                solver=solver,
                                environment=environment,
                                verify_global=verify_pair_energies,
                                **active_pair_options,
                            )
                        )
                        self._pair_last_visited[site] = directional_sweep
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
                moving_norm = self._norm_frontier.advance_left(
                    moving_norm,
                    self.tensors,
                    nsites - 1,
                )
                moving_hamiltonian = self._hamiltonian_frontier.advance_left(
                    moving_hamiltonian,
                    self.tensors,
                    nsites - 1,
                )
                boundary_cut = nsites
            else:
                norm_left = self._norm_frontier.build_left(self.tensors)
                hamiltonian_left = self._hamiltonian_frontier.build_left(
                    self.tensors
                )
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                for site in range(nsites - 2, -1, -1):
                    if site in selected_pairs:
                        environment = self._pair_environment_from_outer_messages(
                            site,
                            norm_left[site],
                            moving_norm,
                            hamiltonian_left[site],
                            moving_hamiltonian,
                        )
                        updates.append(
                            self.optimize_two_sites(
                                site,
                                solver=solver,
                                environment=environment,
                                verify_global=verify_pair_energies,
                                **active_pair_options,
                            )
                        )
                        self._pair_last_visited[site] = directional_sweep
                    moving_norm = self._norm_frontier.advance_right(
                        moving_norm,
                        self.tensors,
                        site + 1,
                    )
                    moving_hamiltonian = self._hamiltonian_frontier.advance_right(
                        moving_hamiltonian,
                        self.tensors,
                        site + 1,
                    )
                moving_norm = self._norm_frontier.advance_right(
                    moving_norm,
                    self.tensors,
                    0,
                )
                moving_hamiltonian = self._hamiltonian_frontier.advance_right(
                    moving_hamiltonian,
                    self.tensors,
                    0,
                )
                boundary_cut = 0

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
            attempted_energy = float(np.real(numerator / norm))
            endpoint_tolerance = 512.0 * np.finfo(float).eps * max(
                1.0,
                abs(previous),
            )
            endpoint_accepted = bool(
                np.isfinite(attempted_energy)
                and attempted_energy <= previous + endpoint_tolerance
            )
            if endpoint_accepted:
                energy = attempted_energy
                self.balance_gauges(state_norm=np.sqrt(norm))
            else:
                self.tensors = sweep_tensors
                energy = previous
            self.energy = energy
            delta = abs(energy - previous)
            cycle_endpoints_accepted = bool(
                cycle_endpoints_accepted and endpoint_accepted
            )
            endpoint_stationary = bool(
                endpoint_accepted
                or (
                    np.isfinite(attempted_energy)
                    and attempted_energy
                    <= previous + max(endpoint_tolerance, tol)
                )
            )
            cycle_endpoints_stationary = bool(
                cycle_endpoints_stationary and endpoint_stationary
            )
            cycle_complete = bool(
                cycle_started and directional_sweep % 2 == 1
            )
            cycle_delta = (
                abs(energy - cycle_start_energy) if cycle_complete else None
            )
            cycle_pair_coverage_complete = bool(
                cycle_complete
                and len(cycle_visited_pairs) == max(0, nsites - 1)
            )
            if cycle_complete:
                last_relative_cycle_gain = cycle_delta / max(
                    1.0,
                    abs(cycle_start_energy),
                )
            residual_certificates = None
            residual_certified = False
            certification_due = bool(
                certify_residual
                and cycle_complete
                and (directional_sweep // 2 + 1) % certify_every == 0
            )
            if certification_due:
                residual_certificates = self.pair_residual_certificates(
                    energy=energy
                )
                residual_certified = bool(
                    not residual_certificates
                    or max(residual_certificates) <= residual_tol
                )
            sweep_history.append(
                {
                    "sweep": directional_sweep,
                    "energy": energy,
                    "attempted_energy": attempted_energy,
                    "delta": delta,
                    "cycle": directional_sweep // 2,
                    "cycle_complete": cycle_complete,
                    "cycle_start_energy": float(cycle_start_energy),
                    "cycle_delta": cycle_delta,
                    "cycle_endpoints_accepted": (
                        cycle_endpoints_accepted if cycle_complete else None
                    ),
                    "cycle_endpoints_stationary": (
                        cycle_endpoints_stationary if cycle_complete else None
                    ),
                    "accepted": endpoint_accepted,
                    "direction": (
                        "left_to_right"
                        if directional_sweep % 2 == 0
                        else "right_to_left"
                    ),
                    "updates": tuple(updates),
                    "accepted_updates": sum(update.accepted for update in updates),
                    "adaptive_solver": adaptive_solver,
                    "eig_tol": active_eig_tol,
                    "pair_selection": pair_selection,
                    "selected_pairs": tuple(sorted(selected_pairs)),
                    "residual_scores": residual_scores,
                    "residual_scores_reused": residual_scores_reused,
                    "residual_fraction": residual_fraction,
                    "residual_captured_fraction": residual_captured_fraction,
                    "coverage_every": coverage_every,
                    "cycle_pair_coverage_complete": (
                        cycle_pair_coverage_complete
                        if cycle_complete
                        else None
                    ),
                    "residual_certification_due": certification_due,
                    "pair_residual_certificates": residual_certificates,
                    "maximum_pair_residual": (
                        None
                        if residual_certificates is None
                        else max(residual_certificates, default=0.0)
                    ),
                    "residual_tol": residual_tol,
                    "residual_certified": residual_certified,
                    "dense_fallbacks": sum(
                        bool(
                            update.merged_solve is not None
                            and update.merged_solve.dense_fallback
                        )
                        for update in updates
                    ),
                }
            )
            if verbose:
                print(
                    f"two-site sweep {directional_sweep:3d}  "
                    f"energy={energy:.14f}  delta={delta:.3e}  "
                    f"accepted={endpoint_accepted}"
                )
            if (
                cycle_complete
                and cycle_endpoints_stationary
                and (
                    residual_certified
                    if certify_residual
                    else (
                        pair_selection == "all"
                        or cycle_pair_coverage_complete
                    )
                )
                and cycle_delta < tol
            ):
                self.converged = True
                break
            if cycle_complete:
                cycle_started = False
            previous = energy
        # Accepted pair updates invalidate standalone optimization history.
        # Keep the directional records in a local accumulator so those resets
        # cannot discard rows from earlier sweeps.
        self.history = sweep_history
        return self

    def run_optimized(
        self,
        *,
        warmup_sweeps: int = 2,
        two_site_cycles: int = 4,
        polish_sweeps: int = 12,
        tol: float = 1.0e-8,
        residual_tol: float | None = None,
        rank_rtol: float = 0.0,
        rank_atol: float = 0.0,
        tie_rtol: float = 0.0,
        tie_atol: float = 0.0,
        adaptive_bond: bool = True,
        growth_rtol: float = 1.0e-8,
        bond_growth: int = 2,
        one_site_options=None,
        two_site_options=None,
    ):
        r"""Run the adaptive production schedule for an exact frontier.

        The schedule uses inexpensive one-site warmup, residual-selected
        two-site sector discovery and a fixed-structure one-site polish. A
        complete pair-residual certificate is evaluated only after polishing.
        Exact null bonds and rank-one ties are removed before a final polish.
        Detailed phase histories are retained
        in ``optimization_history`` while ``history`` remains the most recent
        low-level sweep history. The production pair solve regularizes overlap
        directions below ``1e-8`` by default; pass ``metric_tol`` through
        ``two_site_options`` to tighten or disable that approximation.
        """

        warmup_sweeps = int(warmup_sweeps)
        two_site_cycles = int(two_site_cycles)
        polish_sweeps = int(polish_sweeps)
        tol = float(tol)
        if min(warmup_sweeps, two_site_cycles, polish_sweeps) < 0:
            raise ValueError("optimized sweep counts must be nonnegative.")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        if not self.contraction_is_exact:
            raise ValueError("run_optimized requires exact frontier contractions.")
        local_eig_tol = max(1.0e-9, min(1.0e-6, tol * 0.1))

        one_options = {
            "solver": "matrix_free",
            "adaptive_solver": True,
            "eig_tol": local_eig_tol,
            "eig_tol_initial": 1.0e-4,
            "preconditioner": "auto",
            "block_size": 2,
            "gauge": "frontier",
            "gauge_weight": "probability",
            "environment_cache": "checkpointed",
        }
        one_options.update(dict(one_site_options or {}))
        pair_options = {
            "solver": "matrix_free",
            "adaptive_solver": True,
            "eig_tol": local_eig_tol,
            "eig_tol_initial": 1.0e-4,
            "pair_selection": "residual",
            "residual_fraction": 0.9,
            "min_pairs": min(2, max(1, len(self.dims) - 1)),
            "max_pairs": (
                None
                if len(self.dims) < 2
                else min(len(self.dims) - 1, 4)
            ),
            "coverage_every": 0,
            "reuse_residual_scores": False,
            "certify_residual": False,
            "residual_tol": residual_tol,
            "pair_operator_backend": "auto",
            "factor_solver": "matrix_free",
            "split_strategy": "hybrid",
            "split_metric_sweeps": 1,
            "split_variational_sweeps": 1,
            "outer_cycles": 1,
            "metric_tol": 1.0e-8,
            "preconditioner": "auto",
            "block_size": 2,
            "recycle": True,
            "recycle_min_size": 1,
            "maxiter": 60,
            "max_subspace": 32,
        }
        pair_options.update(dict(two_site_options or {}))
        phases = []
        bond_expansions = []

        if warmup_sweeps:
            self.run(nsweeps=warmup_sweeps, tol=tol, **one_options)
            phases.append(("warmup", tuple(self.history)))
        if adaptive_bond:
            expansions = self.adapt_bonds(
                rtol=growth_rtol,
                growth=bond_growth,
                strategy="residual",
            )
            bond_expansions.extend(expansions)
            if expansions and warmup_sweeps:
                self.run(nsweeps=warmup_sweeps, tol=tol, **one_options)
                phases.append(("expanded_warmup", tuple(self.history)))
        if two_site_cycles:
            self.run_two_site(
                nsweeps=2 * two_site_cycles,
                tol=tol,
                **pair_options,
            )
            phases.append(("two_site", tuple(self.history)))
        if polish_sweeps:
            self.run(nsweeps=polish_sweeps, tol=tol, **one_options)
            phases.append(("polish", tuple(self.history)))
        polish_converged = bool(self.converged) if polish_sweeps else True

        tie_reductions = self.prune_ties(
            rtol=tie_rtol,
            atol=tie_atol,
        )
        bond_reductions = self.reduce_null_bonds(
            rtol=rank_rtol,
            atol=rank_atol,
        )
        if (tie_reductions or bond_reductions) and polish_sweeps:
            self.run(nsweeps=polish_sweeps, tol=tol, **one_options)
            phases.append(("reduced_polish", tuple(self.history)))
            polish_converged = bool(self.converged)

        final_energy = float(self.expectation())
        certificates = self.pair_residual_certificates(energy=final_energy)
        effective_residual_tol = pair_options.get("residual_tol")
        if effective_residual_tol is None:
            effective_residual_tol = max(
                8.0 * np.sqrt(np.finfo(float).eps),
                10.0 * float(pair_options["eig_tol"]),
            )
        residual_converged = bool(
            not certificates
            or max(certificates) <= float(effective_residual_tol)
        )
        self.energy = final_energy
        self.converged = bool(
            polish_converged
            and residual_converged
        )
        self.optimization_history = tuple(phases)
        self.optimization_summary = {
            "energy": final_energy,
            "converged": self.converged,
            "maximum_pair_residual": max(certificates, default=0.0),
            "residual_tol": float(effective_residual_tol),
            "tie_reductions": tie_reductions,
            "bond_reductions": bond_reductions,
            "bond_expansions": tuple(bond_expansions),
            "phase_passes": {
                name: len(history) for name, history in phases
            },
        }
        return self

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

    @property
    def hamiltonian_chunks(self) -> tuple[int, ...]:
        """Numbers of Hamiltonian components in the active exact chunks."""

        return tuple(getattr(self._hamiltonian_frontier, "chunk_sizes", ()))

    @property
    def hamiltonian_windows(self) -> tuple[tuple[int, int], ...]:
        """Half-open active site intervals of exact Hamiltonian chunks."""

        return tuple(getattr(self._hamiltonian_frontier, "chunk_intervals", ()))

    @property
    def stream_peak_frontier_elements(self) -> int:
        """Largest stored message during a scalar expectation contraction."""

        if not isinstance(self._hamiltonian_frontier, TermwiseBlockMPOFrontier):
            return self.peak_frontier_elements
        return max(
            self._dense_peak_elements(self._norm_frontier),
            int(self._hamiltonian_frontier.stream_peak_message_elements),
        )

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

    def fixed_environment_cache_elements(
        self,
        *,
        mode="checkpointed",
        interval: int | None = None,
    ) -> int:
        """Estimated peak stored elements for the fixed side of a sweep."""
        mode = str(mode).lower().replace("-", "_")
        if mode == "checkpoint":
            mode = "checkpointed"
        if mode == "recompute":
            cut_elements = [
                sum(
                    self._dense_message_elements(engine, cut)
                    for engine in (self._norm_frontier, self._hamiltonian_frontier)
                )
                for cut in range(len(self.dims) + 1)
            ]
            adjacent = max(
                left + right
                for left, right in zip(cut_elements[:-1], cut_elements[1:])
            )
            return int(adjacent)
        if mode == "full":
            return self.cached_environment_elements
        if mode != "checkpointed":
            raise ValueError("mode must be 'checkpointed' or 'full'.")
        if interval is None:
            interval = max(1, int(np.ceil(np.sqrt(len(self.dims)))))
        interval = int(interval)
        if interval < 1:
            raise ValueError("interval must be positive.")
        cuts = self._environment_checkpoint_cuts(interval)
        engines = (self._norm_frontier, self._hamiltonian_frontier)
        checkpoint_elements = sum(
            self._dense_message_elements(engine, cut)
            for engine in engines
            for cut in cuts
        )
        block_elements = max(
            (
                sum(
                    self._dense_message_elements(engine, cut)
                    for engine in engines
                    for cut in range(start + 1, end)
                )
                for start, end in zip(cuts[:-1], cuts[1:])
            ),
            default=0,
        )
        return int(checkpoint_elements + block_elements)

    @property
    def contraction_is_exact(self) -> bool:
        """Whether the configured contractor performs no TT truncation."""
        return self.norm_contraction_is_exact and self.hamiltonian_contraction_is_exact

    @property
    def _tt_contraction_is_exact(self) -> bool:
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
        return self._tt_contraction_is_exact

    @property
    def hamiltonian_contraction_is_exact(self) -> bool:
        """Whether the Hamiltonian contractor performs no TT truncation."""

        if not isinstance(self._hamiltonian_frontier, TTMPOFrontier):
            return True
        return self._tt_contraction_is_exact

    @property
    def hamiltonian_action_is_hermitian(self) -> bool:
        """Whether local Hamiltonian actions are explicitly Hermitian."""

        return self.hamiltonian_contraction_is_exact or self.tt_hermitize

    @property
    def tt_diagnostics(self):
        """Latest norm and Hamiltonian TT diagnostics, or ``None``."""

        norm_is_tt = isinstance(self._norm_frontier, TTMPOFrontier)
        hamiltonian_is_tt = isinstance(
            self._hamiltonian_frontier,
            TTMPOFrontier,
        )
        if not norm_is_tt and not hamiltonian_is_tt:
            return None
        return {
            "norm": (
                self._norm_frontier.diagnostics
                if norm_is_tt
                else None
            ),
            "hamiltonian": (
                self._hamiltonian_frontier.diagnostics
                if hamiltonian_is_tt
                else None
            ),
        }

    @property
    def peak_compressed_frontier_elements(self) -> int:
        """Peak TT message storage observed in the latest contractions."""

        if not isinstance(self._norm_frontier, TTMPOFrontier) and not isinstance(
            self._hamiltonian_frontier,
            TTMPOFrontier,
        ):
            return self.peak_frontier_elements
        norm_storage = (
            self._norm_frontier.diagnostics.peak_message_storage_elements
            if isinstance(self._norm_frontier, TTMPOFrontier)
            else self._norm_frontier.peak_message_elements
        )
        hamiltonian_storage = (
            self._hamiltonian_frontier.diagnostics.peak_message_storage_elements
            if isinstance(self._hamiltonian_frontier, TTMPOFrontier)
            else self._hamiltonian_frontier.peak_message_elements
        )
        return max(norm_storage, hamiltonian_storage)

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
        if isinstance(
            frontier,
            (TermwiseBlockMPOFrontier, TermwiseTTMPOFrontier),
        ):
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
        helper streams directly to the two surrounding cuts and retains no
        unrelated messages.
        """
        site = self._validated_site(site)
        norm_left = self._message_at_cut(self._norm_frontier, site, direction="left")
        norm_right = self._message_at_cut(
            self._norm_frontier, site + 1, direction="right"
        )
        hamiltonian_left = self._message_at_cut(
            self._hamiltonian_frontier, site, direction="left"
        )
        hamiltonian_right = self._message_at_cut(
            self._hamiltonian_frontier, site + 1, direction="right"
        )
        return FrontierSiteEnvironment(
            site=site,
            norm_left=norm_left,
            norm_right=norm_right,
            hamiltonian_left=hamiltonian_left,
            hamiltonian_right=hamiltonian_right,
        )

    def _message_at_cut(self, frontier, cut, *, direction):
        cut = int(cut)
        if cut < 0 or cut > len(self.dims):
            raise IndexError("cut is out of range.")
        if direction == "left":
            message = frontier.left_boundary()
            for site in range(cut):
                message = frontier.advance_left(message, self.tensors, site)
            return message
        if direction == "right":
            message = frontier.right_boundary()
            for site in range(len(self.dims) - 1, cut - 1, -1):
                message = frontier.advance_right(message, self.tensors, site)
            return message
        raise ValueError("direction must be 'left' or 'right'.")

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
        if self.requires_matrix_free_solver:
            raise NotImplementedError(
                "physical-slice block construction currently requires an exact "
                "dense or identity-block frontier; use solver='matrix_free' "
                "for this frontier."
            )
        site = self._validated_site(site)
        environment = self._resolved_environment(site, environment)
        layout = PhysicalBlockLayout(self.tensors[site].shape)
        pairs = self._hamiltonian_physical_blocks(site)
        parameter_mask = self._local_action_mask(site)
        block_masks = (
            None
            if parameter_mask is None
            else layout.as_blocks(np.asarray(parameter_mask, dtype=bool).reshape(-1))
        )

        def masked(block, row, column):
            if block_masks is None:
                return block
            return np.asarray(block) * (
                block_masks[row][:, None] * block_masks[column][None, :]
            )

        def metric_factory(row, column):
            return masked(
                self._norm_frontier.hole_block(
                    site,
                    environment.norm_left,
                    environment.norm_right,
                    layout.configurations[row],
                    layout.configurations[column],
                ),
                row,
                column,
            )

        def hamiltonian_factory(row, column):
            return masked(
                self._hamiltonian_frontier.hole_block(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    layout.configurations[row],
                    layout.configurations[column],
                ),
                row,
                column,
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

    def local_action_block_problem(
        self,
        site: int,
        *,
        environment=None,
    ) -> PhysicalBlockGeneralizedProblem:
        r"""Build conditional norm blocks with a matrix-free Hamiltonian.

        The norm is exactly block diagonal in the tied physical
        configuration.  Whitening those small virtual blocks supplies the
        conditional-canonical coordinates needed by Davidson, while the
        substantially larger local Hamiltonian is retained only as an action.
        """
        site = self._validated_site(site)
        environment = self._resolved_environment(site, environment)
        layout = PhysicalBlockLayout(self.tensors[site].shape)
        pairs = self._hamiltonian_physical_blocks(site)
        parameter_mask = self._local_action_mask(site)
        if parameter_mask is None:
            flat_mask = None
            block_masks = None
        else:
            flat_mask = np.asarray(parameter_mask, dtype=bool).reshape(-1)
            if flat_mask.size != layout.size:
                raise ValueError("the local action mask has the wrong size.")
            block_masks = layout.as_blocks(flat_mask)

        prepare_action = getattr(
            self._hamiltonian_frontier,
            "prepare_hole_action",
            None,
        )
        prepared_action = (
            prepare_action(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
            )
            if prepare_action is not None
            else None
        )

        def metric_factory(rows, columns):
            blocks = self._norm_frontier.hole_blocks(
                site,
                environment.norm_left,
                environment.norm_right,
                tuple(layout.configurations[row] for row in rows),
                tuple(layout.configurations[column] for column in columns),
            )
            if block_masks is not None:
                blocks = np.asarray(blocks).copy()
                for offset, (row, column) in enumerate(zip(rows, columns)):
                    blocks[offset] *= (
                        block_masks[row][:, None]
                        * block_masks[column][None, :]
                    )
            return blocks

        def hamiltonian_action(vector):
            trial = np.asarray(vector)
            if flat_mask is not None:
                trial = np.where(flat_mask, trial, 0)
            result = (
                prepared_action(trial)
                if prepared_action is not None
                else self.hamiltonian_action(
                    site,
                    trial,
                    environment=environment,
                )
            )
            return (
                result
                if flat_mask is None
                else np.where(flat_mask, result, 0)
            )

        def hamiltonian_actions(vectors):
            trials = np.asarray(vectors)
            if flat_mask is not None:
                trials = np.where(flat_mask[None, :], trials, 0)
            prepared_many = getattr(prepared_action, "many", None)
            if prepared_many is not None:
                result = prepared_many(trials)
            else:
                result = np.stack(
                    [hamiltonian_action(trial) for trial in trials]
                )
            return (
                result
                if flat_mask is None
                else np.where(flat_mask[None, :], result, 0)
            )

        hamiltonian_action.many = hamiltonian_actions
        prepared_verify = getattr(prepared_action, "verify", None)
        if prepared_verify is not None:
            def hamiltonian_verify(vector):
                trial = np.asarray(vector)
                if flat_mask is not None:
                    trial = np.where(flat_mask, trial, 0)
                result = prepared_verify(trial)
                return (
                    result
                    if flat_mask is None
                    else np.where(flat_mask, result, 0)
                )

            hamiltonian_action.verify = hamiltonian_verify

        return (
            PhysicalBlockGeneralizedProblem
            .from_batched_metric_factory_and_hamiltonian_action(
                self.tensors[site].shape,
                pairs,
                metric_factory,
                hamiltonian_action,
                dtype=np.result_type(
                    self.tensors[site].dtype,
                    self.hamiltonian.dtype,
                ),
            )
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

    def _amen_enrichment_directions(
        self,
        site,
        *,
        environment,
        direction,
        rank,
        rtol,
    ):
        r"""Compress the conditional Hamiltonian-component ranges at a cut.

        Each exact MPO transition group or termwise Hamiltonian chunk is
        applied separately.  For every assignment of the physical labels
        shared by the tensors across the cut, its component orthogonal to that
        assignment's occupied virtual range updates an independent small
        running SVD.  The :math:`k`-th direction from every assignment is then
        packed into one virtual channel.  This matches the conditional SVD
        used when the temporary bond is retracted: one nominal bond channel
        may represent a different vector in every tied-label block.

        With no shared physical labels there is one block and this reduces to
        ordinary MPS AMEn.  The metric component :math:`-E N A` is included as
        the identity channel of the residual.
        """
        site = self._validated_site(site)
        direction = str(direction).lower()
        rank = int(rank)
        rtol = float(rtol)
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'.")
        if rank < 1:
            raise ValueError("enrichment rank must be positive.")
        if not np.isfinite(rtol) or rtol < 0.0:
            raise ValueError("enrichment tolerance must be finite and nonnegative.")

        tensor = self.tensors[site]
        if direction == "right":
            axes = (0, *range(2, tensor.ndim), 1)
            occupied_matrix = tensor.transpose(axes).reshape(-1, tensor.shape[1])

            def ordered(value):
                return np.asarray(value).reshape(tensor.shape).transpose(axes).reshape(
                    occupied_matrix.shape
                )

            transpose_result = False
        else:
            occupied_matrix = tensor.reshape(tensor.shape[0], -1).T

            def ordered(value):
                return np.asarray(value).reshape(tensor.shape).reshape(
                    tensor.shape[0], -1
                ).T

            transpose_result = True

        condition_rows = self._amen_condition_rows(site, direction)
        occupied = []
        conditional_available = []
        for rows in condition_rows:
            block = occupied_matrix[rows]
            left, singular_values, _right = np.linalg.svd(
                block,
                full_matrices=False,
            )
            occupied_scale = max(
                float(np.max(singular_values, initial=0.0)),
                1.0,
            )
            occupied_rank = int(
                np.count_nonzero(
                    singular_values
                    > 256.0 * np.finfo(float).eps * occupied_scale
                )
            )
            occupied.append(left[:, :occupied_rank])
            conditional_available.append(max(len(rows) - occupied_rank, 0))
        # The augmented QR must fit ``current bond + new directions`` in the
        # local row space.  Rank deficiency of the occupied columns does not
        # create additional tensor rows; those null channels are handled by
        # bond reduction or by the capped retraction.
        available = occupied_matrix.shape[0] - occupied_matrix.shape[1]
        requested = min(
            rank,
            max(available, 0),
            max(conditional_available, default=0),
        )
        if requested == 0:
            empty = np.zeros(
                (0, occupied_matrix.shape[0])
                if transpose_result
                else (occupied_matrix.shape[0], 0),
                dtype=tensor.dtype,
            )
            return empty, 0.0, 0, 0.0

        vector = tensor.reshape(-1)
        frontier = self._hamiltonian_frontier
        open_action = getattr(
            frontier,
            (
                "left_enrichment_components"
                if direction == "right"
                else "right_enrichment_components"
            ),
            None,
        )
        if open_action is not None:
            components = open_action(
                site,
                (
                    environment.hamiltonian_left
                    if direction == "right"
                    else environment.hamiltonian_right
                ),
                vector,
            )
            open_components = True
        else:
            component_action = getattr(frontier, "hole_action_components", None)
            if component_action is None:
                components = (
                    self.hamiltonian_action(site, vector, environment=environment),
                )
            else:
                components = component_action(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    vector,
                )
            open_components = False

        if open_components:
            count_action = getattr(frontier, "enrichment_component_count", None)
            expected_components = (
                int(count_action(site)) if count_action is not None else None
            )
            metric_share = None
        else:
            count_action = getattr(frontier, "hole_action_component_count", None)
            expected_components = (
                int(count_action(site)) if count_action is not None else 1
            )
            metric_share = (
                float(self.energy)
                / expected_components
                * self.metric_action(site, vector, environment=environment)
            )
        bases = [
            np.zeros((len(rows), 0), dtype=tensor.dtype)
            for rows in condition_rows
        ]
        values = [np.zeros(0, dtype=float) for _rows in condition_rows]
        workspace_ranks = [
            min(block_available, requested + 8)
            for block_available in conditional_available
        ]
        total_weight = 0.0
        component_count = 0
        if expected_components is not None and expected_components < 1:
            raise ValueError("Hamiltonian action has no enrichment components.")

        def consume(component):
            nonlocal total_weight, component_count
            residual = (
                np.asarray(component)
                if open_components
                else ordered(np.asarray(component) - metric_share)
            )
            if residual.ndim != 2 or residual.shape[0] != occupied_matrix.shape[0]:
                raise ValueError("open enrichment component has an invalid shape.")
            for block_index, rows in enumerate(condition_rows):
                block = residual[rows]
                occupied_block = occupied[block_index]
                if occupied_block.shape[1]:
                    block = block - occupied_block @ (
                        occupied_block.conj().T @ block
                    )
                weight = float(np.linalg.norm(block) ** 2)
                total_weight += weight
                if weight <= np.finfo(float).tiny:
                    continue
                weighted_basis = (
                    bases[block_index] * values[block_index][None, :]
                )
                work = np.concatenate((weighted_basis, block), axis=1)
                next_basis, next_values, _right = np.linalg.svd(
                    work,
                    full_matrices=False,
                )
                retained = min(
                    workspace_ranks[block_index],
                    next_values.size,
                )
                bases[block_index] = next_basis[:, :retained]
                values[block_index] = next_values[:retained]
            component_count += 1

        for component in components:
            consume(component)
        if (
            expected_components is not None
            and component_count != expected_components
        ):
            raise ValueError(
                "Hamiltonian enrichment component count changed during streaming."
            )

        leading_value = max(
            (float(block_values[0]) for block_values in values if block_values.size),
            default=0.0,
        )
        numerical_threshold = (
            256.0
            * np.finfo(float).eps
            * max(leading_value, 1.0)
        )
        selected_per_block = []
        for block_values in values:
            if not block_values.size:
                selected_per_block.append(0)
                continue
            threshold = max(
                rtol * float(block_values[0]),
                numerical_threshold,
            )
            selected_per_block.append(
                min(requested, int(np.count_nonzero(block_values > threshold)))
            )
        selected = max(selected_per_block, default=0)
        packed = np.zeros(
            (occupied_matrix.shape[0], selected),
            dtype=np.result_type(
                tensor.dtype,
                *(basis.dtype for basis in bases),
            ),
        )
        retained_weight = 0.0
        for rows, basis, block_values, block_selected in zip(
            condition_rows,
            bases,
            values,
            selected_per_block,
        ):
            if block_selected:
                packed[np.ix_(rows, np.arange(block_selected))] = (
                    basis[:, :block_selected]
                    * block_values[None, :block_selected]
                )
                retained_weight += float(
                    np.sum(block_values[:block_selected] ** 2)
                )
        if selected:
            norms = np.linalg.norm(packed, axis=0)
            usable = norms > numerical_threshold
            packed = packed[:, usable]
            packed /= np.linalg.norm(packed, axis=0)[None, :]
        discarded = (
            max(0.0, min(1.0, 1.0 - retained_weight / total_weight))
            if total_weight > np.finfo(float).tiny
            else 0.0
        )
        directions = packed.T if transpose_result else packed
        return (
            directions,
            float(np.sqrt(total_weight)),
            component_count,
            discarded,
        )

    def _amen_condition_rows(self, site, direction):
        """Return ordered-matrix rows for each shared-tie assignment."""
        site = self._validated_site(site)
        direction = str(direction).lower()
        if direction == "right":
            following = site + 1
            if following >= len(self.dims):
                tensor = self.tensors[site]
                size = tensor.shape[0] * int(np.prod(tensor.shape[2:], dtype=int))
                return (np.arange(size, dtype=np.intp),)
            tensor = self.tensors[site]
            physical_sites = self.physical_groups[site]
            overlap = tuple(
                sorted(
                    set(physical_sites)
                    & set(self.physical_groups[following])
                )
            )
            row_shape = (tensor.shape[0],) + tuple(
                self.dims[index] for index in physical_sites
            )
        elif direction == "left":
            preceding = site - 1
            if preceding < 0:
                tensor = self.tensors[site]
                size = tensor.shape[1] * int(np.prod(tensor.shape[2:], dtype=int))
                return (np.arange(size, dtype=np.intp),)
            tensor = self.tensors[site]
            physical_sites = self.physical_groups[site]
            overlap = tuple(
                sorted(
                    set(physical_sites)
                    & set(self.physical_groups[preceding])
                )
            )
            row_shape = (tensor.shape[1],) + tuple(
                self.dims[index] for index in physical_sites
            )
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        rows = np.arange(np.prod(row_shape, dtype=int), dtype=np.intp).reshape(
            row_shape
        )
        if not overlap:
            return (rows.reshape(-1),)
        result = []
        overlap_shape = tuple(self.dims[index] for index in overlap)
        for configuration in np.ndindex(*overlap_shape):
            selection = [slice(None)] * len(row_shape)
            for physical_site, value in zip(overlap, configuration):
                selection[1 + physical_sites.index(physical_site)] = value
            result.append(rows[tuple(selection)].reshape(-1))
        return tuple(result)

    def _amen_occupied_basis(self, cut, direction):
        """Return occupied bases conditional on shared tied configurations."""
        cut = int(cut)
        direction = str(direction).lower()
        left_site = cut - 1
        right_site = cut
        overlap = tuple(
            sorted(
                set(self.physical_groups[left_site])
                & set(self.physical_groups[right_site])
            )
        )
        overlap_shape = tuple(self.dims[index] for index in overlap)
        configurations = (
            np.ndindex(*overlap_shape) if overlap_shape else ((),)
        )
        if direction == "right":
            tensor = self.tensors[left_site]
            physical_sites = self.physical_groups[left_site]
            exclusive = tuple(
                index for index in physical_sites if index not in overlap
            )
            exclusive_shape = tuple(self.dims[index] for index in exclusive)

            def matrix_at(overlap_configuration):
                values = dict(zip(overlap, overlap_configuration))
                block = np.empty(
                    (tensor.shape[0], *exclusive_shape, tensor.shape[1]),
                    dtype=tensor.dtype,
                )
                exclusive_configurations = (
                    np.ndindex(*exclusive_shape) if exclusive_shape else ((),)
                )
                for configuration in exclusive_configurations:
                    values.update(zip(exclusive, configuration))
                    physical = tuple(values[index] for index in physical_sites)
                    block[(slice(None), *configuration, slice(None))] = tensor[
                        (slice(None), slice(None), *physical)
                    ]
                return block.reshape(-1, tensor.shape[1])
        elif direction == "left":
            tensor = self.tensors[right_site]
            physical_sites = self.physical_groups[right_site]
            exclusive = tuple(
                index for index in physical_sites if index not in overlap
            )
            exclusive_shape = tuple(self.dims[index] for index in exclusive)

            def matrix_at(overlap_configuration):
                values = dict(zip(overlap, overlap_configuration))
                block = np.empty(
                    (tensor.shape[0], tensor.shape[1], *exclusive_shape),
                    dtype=tensor.dtype,
                )
                exclusive_configurations = (
                    np.ndindex(*exclusive_shape) if exclusive_shape else ((),)
                )
                for configuration in exclusive_configurations:
                    values.update(zip(exclusive, configuration))
                    physical = tuple(values[index] for index in physical_sites)
                    block[(slice(None), slice(None), *configuration)] = tensor[
                        (slice(None), slice(None), *physical)
                    ]
                return block.reshape(tensor.shape[0], -1).conj().T
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        result = []
        for configuration in configurations:
            matrix = matrix_at(configuration)
            basis, values, _right = np.linalg.svd(matrix, full_matrices=False)
            scale = max(float(np.max(values, initial=0.0)), 1.0)
            rank = int(
                np.count_nonzero(
                    values > 256.0 * np.finfo(float).eps * scale
                )
            )
            result.append(basis[:, :rank])
        return tuple(result)

    def _amen_subspace_change(self, before, cut, direction):
        """Frobenius distance between pre-expansion and retained projectors."""
        before = tuple(np.asarray(basis) for basis in before)
        after = self._amen_occupied_basis(cut, direction)
        if len(before) != len(after):
            raise ValueError("AMEn conditional subspace counts are incompatible.")
        distance_squared = 0.0
        for old_basis, new_basis in zip(before, after):
            if old_basis.shape[0] != new_basis.shape[0]:
                raise ValueError(
                    "AMEn occupied subspaces have incompatible supports."
                )
            overlap_weight = float(
                np.linalg.norm(old_basis.conj().T @ new_basis) ** 2
            )
            distance_squared += max(
                0.0,
                old_basis.shape[1]
                + new_basis.shape[1]
                - 2.0 * overlap_weight,
            )
        return float(np.sqrt(distance_squared))

    def _amen_expand_after_site(
        self,
        site,
        *,
        environment,
        direction,
        rank,
        rtol,
        scale,
        refresh_saturated=True,
        metric_tol=1.0e-12,
        max_condition=1.0e8,
        energy_before=None,
    ):
        """Expand the outgoing sweep bond from a streamed residual range."""
        site = self._validated_site(site)
        cut = site + 1 if direction == "right" else site
        if cut <= 0 or cut >= len(self.dims):
            return None, None
        maximum = tuple(getattr(self, "_maximum_bond_dims", self._bond_dims()))
        current = self._bond_dims()[cut]
        if current > maximum[cut]:
            return None, None
        if cut in set(getattr(self, "_null_reduced_cuts", ())):
            return None, None
        saturated = current == maximum[cut]
        if saturated and not bool(refresh_saturated):
            return None, None
        growth = (
            maximum[cut] - current
            if not saturated
            else min(int(rank), maximum[cut])
        )
        if growth < 1:
            return None, None
        directions, source_norm, components, discarded = (
            self._amen_enrichment_directions(
                site,
                environment=environment,
                direction=direction,
                rank=min(int(rank), growth),
                rtol=rtol,
            )
        )
        count = directions.shape[1] if direction == "right" else directions.shape[0]
        if count == 0:
            return None, None
        if saturated:
            occupied_basis = self._amen_occupied_basis(cut, direction)
            left_tensor = self.tensors[cut - 1].copy()
            right_tensor = self.tensors[cut].copy()
            if energy_before is None:
                energy_before = self.expectation()
            directions, temporary_labels = (
                self._prepare_saturated_amen_directions(
                    cut,
                    direction,
                    directions,
                )
            )
            record = self._expand_saturated_amen_bond(
                cut,
                direction=direction,
                directions=directions,
                scale=scale,
                source_norm=source_norm,
                temporary_labels=temporary_labels,
            )
            pending = _PendingAMEnRetraction(
                cut=cut,
                target_dimension=current,
                expansion_direction=direction,
                source_site=site,
                mixing_scale=float(scale),
                energy_before=float(energy_before),
                left_tensor=left_tensor,
                right_tensor=right_tensor,
                anchor_norm=(
                    environment.norm_left
                    if direction == "right"
                    else environment.norm_right
                ),
                anchor_hamiltonian=(
                    environment.hamiltonian_left
                    if direction == "right"
                    else environment.hamiltonian_right
                ),
                occupied_basis=occupied_basis,
            )
        else:
            record = self.expand_bond(
                cut,
                current + count,
                direction=direction,
                strategy="amen",
                scale=scale,
                _directions=directions,
                _source_norm=source_norm,
                _evaluate=False,
                _reset_history=False,
            )
            pending = None
        if not saturated:
            self._condition_after_site(
                site,
                environment=environment,
                direction=direction,
                metric_tol=metric_tol,
                max_condition=max_condition,
            )
        return (
            replace(
                record,
                residual_components=int(components),
                relative_discarded_weight=float(discarded),
            ),
            pending,
        )

    def _expand_saturated_amen_bond(
        self,
        cut,
        *,
        direction,
        directions,
        scale,
        source_norm,
        temporary_labels=None,
    ):
        """Open a temporary dense bond retained through the neighboring solve."""
        if temporary_labels is not None:
            raise TypeError("dense frontier bonds do not carry charge labels.")
        count = (
            directions.shape[1]
            if direction == "right"
            else directions.shape[0]
        )
        return self.expand_bond(
            cut,
            self._bond_dims()[int(cut)] + count,
            direction=direction,
            strategy="amen",
            scale=scale,
            _directions=directions,
            _source_norm=source_norm,
            _evaluate=False,
            _reset_history=False,
        )

    def _retract_amen_bond(self, cut, target_dimension, *, direction):
        """Conditionally split an optimized pair back to its configured cap."""
        cut = int(cut)
        target_dimension = int(target_dimension)
        temporary_dimension = self._bond_dims()[cut]
        if cut <= 0 or cut >= len(self.dims):
            raise ValueError("cut must be an internal virtual bond.")
        if target_dimension < 1 or target_dimension > temporary_dimension:
            raise ValueError("invalid AMEn target bond dimension.")
        if target_dimension == temporary_dimension:
            return None
        direction = str(direction).lower()
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'.")

        left_site = cut - 1
        labels = self._amen_compression_labels(cut, target_dimension)
        merged, union_sites = self._merged_pair_tensor(left_site)
        (
            new_left,
            new_right,
            overlap,
            conditional_ranks,
            truncation_error,
        ) = self._split_merged_pair_tensor(
            left_site,
            merged,
            union_sites,
            middle_dimension=target_dimension,
            middle_labels=labels,
        )
        self.tensors[left_site] = np.asarray(new_left)
        self.tensors[cut] = np.asarray(new_right)
        dimensions = list(self._bond_dims())
        dimensions[cut] = target_dimension
        self._virtual_bond_dims = tuple(dimensions)
        self.bond_dim = max(self._virtual_bond_dims)
        self._set_amen_compressed_bond_layout(cut, labels)
        self._rebuild_frontier_engines()

        sector_dimensions = ()
        if labels is not None:
            sector_dimensions = tuple(
                (charge, labels.count(charge))
                for charge in dict.fromkeys(labels)
            )
        return FrontierBondRefresh(
            cut=cut,
            temporary_dimension=temporary_dimension,
            target_dimension=target_dimension,
            overlap_sites=overlap,
            conditional_ranks=tuple(conditional_ranks),
            relative_truncation_error=float(truncation_error),
            sector_dimensions=sector_dimensions,
        )

    def _condition_after_site(
        self,
        site,
        *,
        environment,
        direction,
        metric_tol,
        max_condition,
    ):
        """Put the outgoing bond into the moving frontier-metric gauge."""
        cut = site + 1 if direction == "right" else site
        if cut <= 0 or cut >= len(self.dims):
            return False
        if direction == "right":
            metric_message = self._norm_frontier.advance_left(
                environment.norm_left,
                self.tensors,
                site,
            )
        else:
            metric_message = self._norm_frontier.advance_right(
                environment.norm_right,
                self.tensors,
                site,
            )
        return self._condition_sweep_bond(
            cut,
            metric_message,
            direction=direction,
            metric_tol=metric_tol,
            max_condition=max_condition,
        )

    def _finish_amen_retraction(self, pending, environment, site):
        """Retract after the neighboring solve and locally guard the result."""
        if not isinstance(pending, _PendingAMEnRetraction):
            raise TypeError("pending must be an AMEn retraction record.")
        site = self._validated_site(site)
        if pending.expansion_direction == "right":
            if site != pending.source_site + 1:
                raise ValueError("right-going AMEn must retract at the next site.")
            self.tensors[pending.cut][pending.target_dimension :] *= (
                pending.mixing_scale
            )
            refresh = self._retract_amen_bond(
                pending.cut,
                pending.target_dimension,
                direction="left",
            )
        elif pending.expansion_direction == "left":
            if site != pending.source_site - 1:
                raise ValueError("left-going AMEn must retract at the previous site.")
            self.tensors[pending.cut - 1][
                :, pending.target_dimension :
            ] *= pending.mixing_scale
            refresh = self._retract_amen_bond(
                pending.cut,
                pending.target_dimension,
                direction="right",
            )
        else:
            raise ValueError("AMEn expansion direction is invalid.")
        environment = self._repair_amen_environment(
            pending,
            environment,
        )
        subspace_change = self._amen_subspace_change(
            pending.occupied_basis,
            pending.cut,
            pending.expansion_direction,
        )
        candidate_energy = self._amen_environment_energy(site, environment)
        tolerance = (
            512.0
            * np.finfo(float).eps
            * max(1.0, abs(pending.energy_before))
        )
        energy_accepted = bool(
            np.isfinite(candidate_energy)
            and candidate_energy <= pending.energy_before + tolerance
        )
        effective = bool(
            subspace_change > 1024.0 * np.finfo(float).eps
        )
        retry = not energy_accepted
        if retry:
            self.tensors[pending.cut - 1] = pending.left_tensor.copy()
            self.tensors[pending.cut] = pending.right_tensor.copy()
            self._apply_bond_gauge_constraints()
            self._rebuild_frontier_engines()
            environment = self._repair_amen_environment(
                pending,
                environment,
            )
        return (
            replace(
                refresh,
                subspace_change=subspace_change,
                accepted=energy_accepted and effective,
            ),
            environment,
            retry,
        )

    def _repair_amen_environment(self, pending, environment):
        """Recompute the fixed moving side changed by a two-tensor retraction."""
        if pending.expansion_direction == "right":
            norm_left = self._norm_frontier.advance_left(
                pending.anchor_norm,
                self.tensors,
                pending.source_site,
            )
            hamiltonian_left = self._hamiltonian_frontier.advance_left(
                pending.anchor_hamiltonian,
                self.tensors,
                pending.source_site,
            )
            return replace(
                environment,
                norm_left=norm_left,
                hamiltonian_left=hamiltonian_left,
            )
        norm_right = self._norm_frontier.advance_right(
            pending.anchor_norm,
            self.tensors,
            pending.source_site,
        )
        hamiltonian_right = self._hamiltonian_frontier.advance_right(
            pending.anchor_hamiltonian,
            self.tensors,
            pending.source_site,
        )
        return replace(
            environment,
            norm_right=norm_right,
            hamiltonian_right=hamiltonian_right,
        )

    def _amen_environment_energy(self, site, environment):
        """Evaluate the exact Rayleigh quotient in one repaired local frame."""
        vector = self.tensors[int(site)].reshape(-1)
        metric_vector = self.metric_action(
            site,
            vector,
            environment=environment,
        )
        norm = float(np.real(np.vdot(vector, metric_vector)))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            return float("inf")
        hamiltonian_vector = self.hamiltonian_action(
            site,
            vector,
            environment=environment,
        )
        return float(np.real(np.vdot(vector, hamiltonian_vector)) / norm)

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
                self.physical_groups[site],
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
        preconditioner="auto",
        block_size: int = 1,
        energy_before: float | None = None,
        environment=None,
    ) -> FrontierSiteUpdate:
        """Minimize one tensor with a dense, metric-orthonormal, action, or block solver.

        ``auto`` retains the dense solver below ``matrix_free_threshold``.
        Above it, a structurally sparse physical-block pencil is used only
        when its estimated raw-plus-whitened block work arrays fit within
        ``block_sparse_max_elements``; otherwise the action-only solver is
        selected.  An explicit ``solver='block_sparse'`` is not capped.
        """
        site = self._validated_site(site)
        solver = str(solver).lower().replace("-", "_")
        solver_record = solver
        if solver in {"block", "physical_block", "physical_blocks"}:
            solver = "block_sparse"
        if solver in {
            "canonical",
            "identity_metric",
            "local_canonical",
            "metric_orthonormal",
            "orthonormal_metric",
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
                "solver must be 'auto', 'direct', 'metric_orthonormal' "
                "(alias 'whitened'), "
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
            if self.requires_matrix_free_solver:
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
            and self.requires_matrix_free_solver
        ):
            raise ValueError(
                f"solver='{selected_solver}' is unavailable for this "
                "matrix-free frontier; "
                "use solver='matrix_free'."
            )
        accepted = False
        energy_after = energy_before
        metric_rank = 0
        hamiltonian_matvecs = 0
        metric_matvecs = 0
        iterations = 0
        residual_norm = float("inf")
        solver_record = (
            "metric_orthonormal"
            if solver_record in {"metric_orthonormal", "orthonormal_metric"}
            else selected_solver
        )
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
                    recycle_spaces=self._davidson_recycle,
                    recycle_prefix=("block", site),
                    executor=self._solver_executor,
                    preconditioner=preconditioner,
                    block_size=block_size,
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
                layout = PhysicalBlockLayout(old_tensor.shape)
                conditional_metric_elements = (
                    layout.nblocks * layout.virtual_size**2
                )
                use_conditional_metric = (
                    block_sparse_max_elements is None
                    or conditional_metric_elements <= block_sparse_max_elements
                )
                if use_conditional_metric:
                    problem = self.local_action_block_problem(
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
                        dense_component_max_size=0,
                        recycle_spaces=self._davidson_recycle,
                        recycle_prefix=("action", site),
                        executor=self._solver_executor,
                        preconditioner=preconditioner,
                        block_size=block_size,
                    )
                else:
                    recycle_key = ("global", site, old_tensor.size)
                    recycle_out = []
                    use_recycle = old_tensor.size >= 256
                    energy_after, vector, diagnostics = (
                        lowest_generalized_davidson(
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
                            initial_subspace=(
                                self._davidson_recycle.get(recycle_key)
                                if use_recycle
                                else None
                            ),
                            recycle_out=(recycle_out if use_recycle else None),
                            preconditioner=(
                                preconditioner if callable(preconditioner) else None
                            ),
                            block_size=block_size,
                        )
                    )
                    if use_recycle:
                        self._davidson_recycle[recycle_key] = tuple(recycle_out)
                metric_rank = diagnostics.projected_rank
                hamiltonian_matvecs = diagnostics.hamiltonian_matvecs
                metric_matvecs = diagnostics.metric_matvecs
                iterations = diagnostics.iterations
                residual_norm = diagnostics.residual_norm
                solver_converged = diagnostics.converged
                message = diagnostics.message
                if use_conditional_metric:
                    physical_blocks = diagnostics.metric_blocks
                    hamiltonian_blocks = diagnostics.hamiltonian_blocks
                    block_component_sizes = diagnostics.component_sizes
                    stored_operator_elements = diagnostics.stored_elements
                    solver_metric_is_identity = True
                else:
                    message = (
                        f"{message}; conditional metric exceeds the storage cap"
                    )
                if not diagnostics.converged:
                    raise ValueError(diagnostics.message)
            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            candidate = np.real_if_close(vector.reshape(old_tensor.shape))
            if (
                isinstance(self._hamiltonian_frontier, TTMPOFrontier)
                and not self.hamiltonian_contraction_is_exact
            ):
                # TT rounding makes the contracted scalar nonlinear in a local
                # tensor.  The Davidson solution is therefore only a proposal;
                # accept it against a fresh contraction of the actual objective.
                self.tensors[site] = np.array(candidate, copy=True)
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
                    self.tensors[site] = np.array(candidate, copy=True)
        except (ValueError, np.linalg.LinAlgError) as error:
            accepted = False
            solver_converged = False
            if message == "local solve not attempted":
                message = str(error)
            else:
                message = f"{message}; {error}"
        if not accepted:
            self.tensors[site] = old_tensor
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

    def _natural_gradient_support_indices(self, site):
        """Return an optional reduced parameter support for global relaxation."""
        return None

    def _natural_gradient_local_data(self, site, environment, energy):
        """Return the local metric, residual, and active parameter support."""
        metric, effective = self.local_operators(
            site,
            environment=environment,
        )
        vector = self.tensors[site].reshape(-1)
        residual = effective @ vector - float(energy) * (metric @ vector)
        support = self._natural_gradient_support_indices(site)
        if support is None:
            return metric, vector, residual, None
        support = np.asarray(support, dtype=np.intp)
        if support.ndim != 1:
            raise ValueError(
                "natural-gradient parameter support must be one-dimensional."
            )
        if support.size == 0:
            raise ValueError(f"tensor {site} has empty natural-gradient support.")
        return (
            metric[np.ix_(support, support)],
            vector[support],
            residual[support],
            support,
        )

    def natural_gradient_step(
        self,
        *,
        metric_tol: float = 1.0e-12,
        damping: float = 1.0e-6,
        trust_radius: float = 0.25,
        max_backtracks: int = 12,
        armijo: float = 1.0e-4,
        energy_before: float | None = None,
        state_norm: float | None = None,
    ) -> FrontierNaturalGradientUpdate:
        r"""Move all tensors along a block-metric natural-gradient direction.

        The exact local residual for tensor ``k`` is

        .. math::

            g_k = (H_k - E N_k)t_k.

        This uses the block-diagonal collection of local metrics, not the full
        cross-site metric.  Subclasses may restrict every local solve to an
        exact structural parameter support.  Each direction is projected
        orthogonal to the radial state direction.  A shared metric trust radius
        and exact Armijo line search handle nonlinear cross terms when all
        tensors move.
        """
        if isinstance(self._hamiltonian_frontier, TTMPOFrontier):
            raise NotImplementedError(
                "natural_gradient_step currently requires dense local operators; "
                "use matrix-free sweeps or VMC stochastic reconfiguration."
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

        if energy_before is None:
            energy_before = float(self.expectation())
        else:
            energy_before = float(energy_before)
            if not np.isfinite(energy_before):
                raise ValueError("energy_before must be finite.")
        if state_norm is None:
            state_norm = float(self.norm())
        else:
            state_norm = float(state_norm)
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
            (
                local_metric,
                local_vector,
                residual,
                support,
            ) = self._natural_gradient_local_data(
                site,
                environment,
                energy_before,
            )
            vector = tensor.reshape(-1)
            eigenvalues, eigenvectors = np.linalg.eigh(local_metric)
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
                local_direction = -basis @ (
                    coefficients / (values + damping * scale)
                )
            else:
                local_direction = np.zeros_like(local_vector)
            local_metric_vector = local_metric @ local_vector
            local_metric_direction = local_metric @ local_direction
            radial_denominator = np.vdot(local_vector, local_metric_vector)
            if abs(radial_denominator) > np.finfo(float).tiny:
                local_direction = local_direction - local_vector * (
                    np.vdot(local_vector, local_metric_direction)
                    / radial_denominator
                )
                local_metric_direction = local_metric @ local_direction
            if support is None:
                direction = local_direction
            else:
                direction = np.zeros(
                    vector.shape,
                    dtype=np.result_type(vector.dtype, local_direction.dtype),
                )
                direction[support] = local_direction
            direction = np.real_if_close(direction).astype(
                np.result_type(tensor.dtype, direction.dtype),
                copy=False,
            )
            directions.append(direction.reshape(tensor.shape))
            gradient_norm_squared += float(np.vdot(residual, residual).real)
            direction_norm_squared += float(np.vdot(direction, direction).real)
            metric_direction_norm_squared += max(
                float(
                    np.vdot(local_direction, local_metric_direction).real
                    / state_norm
                ),
                0.0,
            )
            directional_derivative += float(
                2.0
                * np.real(np.vdot(residual, local_direction))
                / state_norm
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

    def _environment_checkpoint_cuts(self, interval):
        interval = int(interval)
        nsites = len(self.dims)
        nominal = list(range(0, nsites, interval))
        if not nominal or nominal[-1] != nsites:
            nominal.append(nsites)
        if len(nominal) <= 2 or interval <= 1:
            return tuple(nominal)

        radius = max(1, interval // 2)
        cuts = [0]
        for center in nominal[1:-1]:
            lower = max(cuts[-1] + 1, center - radius)
            upper = min(nsites - 1, center + radius)
            cut = min(
                range(lower, upper + 1),
                key=lambda candidate: (
                    sum(
                        self._dense_message_elements(engine, candidate)
                        for engine in (
                            self._norm_frontier,
                            self._hamiltonian_frontier,
                        )
                    ),
                    abs(candidate - center),
                ),
            )
            cuts.append(cut)
        cuts.append(nsites)

        def storage(candidate_cuts):
            engines = (self._norm_frontier, self._hamiltonian_frontier)
            checkpoints = sum(
                self._dense_message_elements(engine, cut)
                for engine in engines
                for cut in candidate_cuts
            )
            interior = max(
                (
                    sum(
                        self._dense_message_elements(engine, cut)
                        for engine in engines
                        for cut in range(start + 1, end)
                    )
                    for start, end in zip(candidate_cuts[:-1], candidate_cuts[1:])
                ),
                default=0,
            )
            return checkpoints + interior

        return tuple(cuts if storage(cuts) < storage(nominal) else nominal)

    def _build_environment_checkpoints(
        self,
        frontier,
        *,
        direction,
        interval=None,
        cuts=None,
    ):
        """Contract one side once while retaining only selected cut messages."""
        if cuts is None:
            if interval is None:
                raise TypeError("interval or cuts must be supplied.")
            cuts = self._environment_checkpoint_cuts(interval)
        cuts = tuple(int(cut) for cut in cuts)
        if (
            not cuts
            or cuts[0] != 0
            or cuts[-1] != len(self.dims)
            or any(left >= right for left, right in zip(cuts, cuts[1:]))
        ):
            raise ValueError(
                "checkpoint cuts must increase from zero to the terminal cut."
            )
        cuts = frozenset(cuts)
        checkpoints = {}
        if direction == "left":
            message = frontier.left_boundary()
            checkpoints[0] = message
            for site in range(len(self.dims)):
                message = frontier.advance_left(message, self.tensors, site)
                cut = site + 1
                if cut in cuts:
                    checkpoints[cut] = message
        elif direction == "right":
            message = frontier.right_boundary()
            checkpoints[len(self.dims)] = message
            for site in range(len(self.dims) - 1, -1, -1):
                message = frontier.advance_right(message, self.tensors, site)
                if site in cuts:
                    checkpoints[site] = message
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        return checkpoints

    def _recompute_environment_block(
        self,
        frontier,
        *,
        direction,
        start,
        end,
        checkpoint,
    ):
        """Materialize only the fixed-side messages needed by one block."""
        if direction == "left":
            messages = {start: checkpoint}
            message = checkpoint
            for site in range(start, end - 1):
                message = frontier.advance_left(message, self.tensors, site)
                messages[site + 1] = message
            return messages
        if direction == "right":
            messages = {end: checkpoint}
            message = checkpoint
            for site in range(end - 1, start, -1):
                message = frontier.advance_right(message, self.tensors, site)
                messages[site] = message
            return messages
        raise ValueError("direction must be 'left' or 'right'.")

    def run(
        self,
        *,
        nsweeps: int = 4,
        sweep_offset: int = 0,
        tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        solver=None,
        matrix_free_threshold: int = 256,
        block_sparse_max_elements: int | None = 4_000_000,
        eig_tol: float = 1.0e-10,
        adaptive_solver: bool = False,
        eig_tol_initial: float = 1.0e-5,
        maxiter: int | None = None,
        max_subspace: int = 32,
        preconditioner="auto",
        block_size: int = 1,
        enrich=None,
        enrich_rank: int = 8,
        enrich_tol: float = 1.0e-7,
        enrich_scale: float = 1.0e-3,
        enrich_every: int = 8,
        enrich_trigger: float | None = 1.0e-4,
        natural_gradient_every: int = 0,
        natural_gradient_damping: float = 1.0e-6,
        natural_gradient_trust_radius: float = 0.25,
        natural_gradient_max_backtracks: int = 12,
        natural_gradient_adaptive: bool = True,
        natural_gradient_max_interval: int | None = None,
        natural_gradient_min_relative_gain: float = 1.0e-8,
        gauge="auto",
        gauge_max_condition: float = 1.0e8,
        gauge_weight="uniform",
        environment_cache="auto",
        environment_memory=64,
        environment_checkpoint_interval: int | None = None,
        verbose: bool = False,
    ):
        r"""Optimize all tensors without constructing the full Hilbert basis.

        By default, exact dense-frontier sweeps use the
        ``metric_orthonormal`` local solver and a uniformly conditioned
        frontier gauge. Tensor-train frontiers select the matrix-free solver
        and leave gauge transformations disabled. ``gauge`` may be ``"auto"``,
        ``"frontier"``, ``"virtual"``, or ``None``.

        ``environment_cache="checkpointed"`` stores fixed-side messages only
        at block boundaries and recomputes at most one block at a time.  With
        the default interval near :math:`\sqrt{N}`, the number of simultaneously
        retained messages falls from :math:`O(N)` to :math:`O(\sqrt{N})` while
        preserving the exact directional-sweep environments.  ``"full"``
        retains the historical all-cut cache. ``"recompute"`` retains no
        fixed-side cache and reconstructs it from the boundary at every local
        update; this minimizes memory at the cost of quadratic sweep work.
        The default ``"auto"`` retains the full packed environment when it
        fits within ``environment_memory`` MiB and otherwise checkpoints it.

        ``enrich="amen"`` replaces the pre-sweep rank-saturation heuristic by
        in-sweep residual enrichment.  Exact Hamiltonian transition groups or
        termwise chunks update a streamed residual covariance; at most
        ``enrich_rank`` significant directions are appended to the outgoing
        bond before the neighboring one-site solve.  The opposite tensor is
        initialized so enrichment itself leaves the represented state and
        energy unchanged while the bond grows. At its configured cap, the
        temporary :math:`D+r` bond is retained through that neighboring solve
        and only then truncated back to :math:`D`. Saturated refreshes are
        considered every ``enrich_every`` sweeps and applied only after the
        preceding relative sweep gain falls below ``enrich_trigger``. Pass
        ``enrich_trigger=None`` to use the unconditional periodic schedule.
        This mode requires ``adaptive_bond=True``.

        ``adaptive_solver=True`` starts local eigensolves at
        ``eig_tol_initial`` and tightens them from the observed sweep energy
        gain down to ``eig_tol``. ``block_size>1`` enables block Davidson on
        matrix-free components, while ``preconditioner="auto"`` uses exact
        block-Jacobi diagonals whenever stored local blocks are available.

        When natural-gradient relaxation is enabled, the adaptive controller
        treats ``natural_gradient_every`` as its initial and minimum interval.
        Low-gain or rejected global steps back off the interval and trust
        radius; accurately predicted steps restore the initial interval and
        enlarge the trust radius.  Set ``natural_gradient_adaptive=False`` for
        the historical fixed-frequency schedule.
        """
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        adaptive_solver = bool(adaptive_solver)
        eig_tol = float(eig_tol)
        eig_tol_initial = float(eig_tol_initial)
        if not np.isfinite(eig_tol) or eig_tol < 0.0:
            raise ValueError("eig_tol must be finite and nonnegative.")
        if not np.isfinite(eig_tol_initial) or eig_tol_initial < eig_tol:
            raise ValueError("eig_tol_initial must be finite and at least eig_tol.")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if solver is None:
            solver = (
                "matrix_free"
                if self.requires_matrix_free_solver
                else "metric_orthonormal"
            )
        if enrich is None:
            enrich = "none"
        else:
            enrich = str(enrich).lower().replace("-", "_")
        if enrich in {"off", "false"}:
            enrich = "none"
        if enrich in {"3s", "dmrg3s", "subspace_expansion"}:
            enrich = "amen"
        if enrich not in {"none", "amen"}:
            raise ValueError("enrich must be 'amen' or None.")
        enrich_rank = int(enrich_rank)
        enrich_tol = float(enrich_tol)
        enrich_scale = float(enrich_scale)
        enrich_every = int(enrich_every)
        if enrich_trigger is not None:
            enrich_trigger = float(enrich_trigger)
        if enrich_rank < 1:
            raise ValueError("enrich_rank must be positive.")
        if not np.isfinite(enrich_tol) or enrich_tol < 0.0:
            raise ValueError("enrich_tol must be finite and nonnegative.")
        if not np.isfinite(enrich_scale) or enrich_scale < 0.0:
            raise ValueError("enrich_scale must be finite and nonnegative.")
        if enrich_every < 1:
            raise ValueError("enrich_every must be positive.")
        if (
            enrich_trigger is not None
            and (
                not np.isfinite(enrich_trigger)
                or enrich_trigger < 0.0
            )
        ):
            raise ValueError(
                "enrich_trigger must be finite and nonnegative or None."
            )
        if enrich == "amen":
            if not getattr(self, "adaptive_bond", False):
                raise ValueError(
                    "enrich='amen' requires adaptive_bond=True so D defines "
                    "the expansion cap."
                )
            if self.uses_tensor_train_frontier:
                raise ValueError(
                    "enrich='amen' currently requires an exact dense, "
                    "identity-block, or termwise frontier."
                )
            if not self.contraction_is_exact:
                raise ValueError("enrich='amen' requires exact contraction.")
        if gauge is None:
            gauge = "none"
        else:
            gauge = str(gauge).lower().replace("-", "_")
        if gauge == "auto":
            gauge = (
                "none"
                if self.uses_tensor_train_frontier
                else "frontier"
            )
        if gauge not in {"none", "frontier", "virtual"}:
            raise ValueError(
                "gauge must be 'auto', 'frontier', 'virtual', or None."
            )
        gauge_weight = str(gauge_weight).lower().replace("-", "_")
        if gauge_weight not in {"uniform", "probability"}:
            raise ValueError("gauge_weight must be 'uniform' or 'probability'.")
        sweep_offset = int(sweep_offset)
        if sweep_offset < 0:
            raise ValueError("sweep_offset must be nonnegative.")
        requested_environment_cache = str(environment_cache).lower().replace(
            "-", "_"
        )
        environment_cache = requested_environment_cache
        if environment_cache == "checkpoint":
            environment_cache = "checkpointed"
        if environment_cache not in {"auto", "checkpointed", "full", "recompute"}:
            raise ValueError(
                "environment_cache must be 'auto', 'checkpointed', "
                "'recompute', or 'full'."
            )
        environment_memory = float(environment_memory)
        if not np.isfinite(environment_memory) or environment_memory < 0.0:
            raise ValueError("environment_memory must be finite and nonnegative.")
        full_environment_bytes = sum(
            self._dense_message_elements(engine, cut)
            * np.dtype(getattr(engine, "dtype", np.complex128)).itemsize
            for engine in (self._norm_frontier, self._hamiltonian_frontier)
            for cut in range(len(self.dims) + 1)
        )
        if environment_cache == "auto":
            packed_u1_environment = (
                isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
                and self._hamiltonian_frontier.charge_resolved
            )
            environment_cache = (
                "full"
                if packed_u1_environment
                and full_environment_bytes <= environment_memory * 1024**2
                else "checkpointed"
            )
        nsites = len(self.dims)
        if environment_checkpoint_interval is None:
            environment_checkpoint_interval = max(1, int(np.ceil(np.sqrt(nsites))))
        else:
            environment_checkpoint_interval = int(environment_checkpoint_interval)
            if environment_checkpoint_interval < 1:
                raise ValueError(
                    "environment_checkpoint_interval must be positive."
                )
        natural_gradient_every = int(natural_gradient_every)
        if natural_gradient_every < 0:
            raise ValueError("natural_gradient_every must be nonnegative.")
        natural_gradient_adaptive = bool(natural_gradient_adaptive)
        natural_gradient_trust_radius = float(natural_gradient_trust_radius)
        if (
            not np.isfinite(natural_gradient_trust_radius)
            or natural_gradient_trust_radius <= 0.0
        ):
            raise ValueError(
                "natural_gradient_trust_radius must be finite and positive."
            )
        natural_gradient_min_relative_gain = float(
            natural_gradient_min_relative_gain
        )
        if (
            not np.isfinite(natural_gradient_min_relative_gain)
            or natural_gradient_min_relative_gain < 0.0
        ):
            raise ValueError(
                "natural_gradient_min_relative_gain must be finite and "
                "nonnegative."
            )
        if natural_gradient_max_interval is None:
            natural_gradient_max_interval = (
                4 * natural_gradient_every if natural_gradient_every else 0
            )
        else:
            natural_gradient_max_interval = int(natural_gradient_max_interval)
            if natural_gradient_max_interval < natural_gradient_every:
                raise ValueError(
                    "natural_gradient_max_interval must be at least "
                    "natural_gradient_every."
                )
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
                "frontiers; use VMC stochastic reconfiguration."
            )
        if isinstance(self._norm_frontier, TTMPOFrontier) and gauge == "frontier":
            raise ValueError(
                "frontier canonicalization currently requires dense exact "
                "messages; use gauge='virtual' or gauge=None."
            )
        retained_run_history = tuple(self.history)
        previous = self.expectation()
        self.energy = previous
        self.history = []
        self.converged = False
        cycle_start_energy = previous
        cycle_stationary = True
        cycle_started = False
        natural_gradient_interval = natural_gradient_every
        natural_gradient_next_sweep = (
            (
                sweep_offset // natural_gradient_every + 1
            )
            * natural_gradient_every
            if natural_gradient_every
            else None
        )
        current_natural_gradient_trust_radius = natural_gradient_trust_radius
        minimum_natural_gradient_trust_radius = (
            natural_gradient_trust_radius / 16.0
        )
        maximum_natural_gradient_trust_radius = (
            4.0 * natural_gradient_trust_radius
        )
        natural_gradient_attempts = 0
        last_natural_gradient_was_useful = False
        if retained_run_history:
            last_record = retained_run_history[-1]
            last_relative_sweep_gain = float(
                last_record.get(
                    "relative_sweep_gain",
                    abs(float(last_record.get("delta_energy", np.inf)))
                    / max(1.0, abs(float(last_record.get("energy", previous)))),
                )
            )
        else:
            last_relative_sweep_gain = float("inf")
        for sweep in range(nsweeps):
            directional_sweep = sweep_offset + sweep
            if directional_sweep % 2 == 0:
                cycle_start_energy = previous
                cycle_stationary = True
                cycle_started = True
            active_eig_tol = eig_tol
            if adaptive_solver:
                gain_tolerance = (
                    eig_tol_initial
                    if not np.isfinite(last_relative_sweep_gain)
                    else 0.1 * last_relative_sweep_gain
                )
                active_eig_tol = max(
                    eig_tol,
                    min(eig_tol_initial, gain_tolerance),
                )
            amen_refresh_scheduled = bool(
                enrich == "amen"
                and directional_sweep % enrich_every == 0
            )
            amen_refresh_due = bool(
                amen_refresh_scheduled
                and (
                    enrich_trigger is None
                    or last_relative_sweep_gain <= enrich_trigger
                )
            )
            updates = []
            gauge_update = None
            bond_refreshes = []
            amen_refresh_accepted = True
            amen_sweep_snapshot = None
            retained_history = self.history
            bond_reductions = (
                self.reduce_null_bonds()
                if (
                    getattr(self, "adaptive_bond", False)
                    # An AMEn channel opened in a left-to-right pass has only
                    # been varied from its right tensor.  Keep it through the
                    # reverse pass before deciding that it is exactly null.
                    and not (enrich == "amen" and directional_sweep % 2 == 1)
                )
                else ()
            )
            self.history = retained_history
            bond_expansions = list(
                self.adapt_bonds(
                    direction="left" if directional_sweep % 2 == 0 else "right"
                )
            )
            self.history = retained_history
            if bond_expansions:
                previous = self.expectation()
                self.energy = previous
            if gauge == "frontier":
                gauge_update = self.canonicalize_frontier_gauge(
                    metric_tol=metric_tol,
                    max_condition=gauge_max_condition,
                    weighting=gauge_weight,
                )
            if enrich == "amen":
                amen_sweep_snapshot = {
                    "tensors": [tensor.copy() for tensor in self.tensors],
                    "bond_dims": self._bond_dims(),
                    "bond_dim": self.bond_dim,
                    "energy": float(self.expectation()),
                    "layout": deepcopy(getattr(self, "abelian_layout", None)),
                    "null_reduced_cuts": set(
                        getattr(self, "_null_reduced_cuts", ())
                    ),
                }
            pending_amen = None
            if directional_sweep % 2 == 0:
                if gauge == "virtual":
                    self.canonicalize_virtual("right")
                    if not self.hamiltonian_contraction_is_exact:
                        previous = self.expectation()
                        self.energy = previous
                if environment_cache == "full":
                    norm_right = self._norm_frontier.build_right(self.tensors)
                    hamiltonian_right = self._hamiltonian_frontier.build_right(
                        self.tensors
                    )
                    right_checkpoints = None
                elif environment_cache == "checkpointed":
                    norm_right = hamiltonian_right = None
                    right_checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="right",
                            interval=environment_checkpoint_interval,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="right",
                            interval=environment_checkpoint_interval,
                        ),
                    )
                else:
                    norm_right = hamiltonian_right = right_checkpoints = None
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                checkpoint_cuts = (
                    (0, nsites)
                    if environment_cache == "recompute"
                    else self._environment_checkpoint_cuts(
                        environment_checkpoint_interval
                    )
                )
                for block_start, block_end in zip(
                    checkpoint_cuts[:-1], checkpoint_cuts[1:]
                ):
                    if right_checkpoints is not None:
                        norm_right = self._recompute_environment_block(
                            self._norm_frontier,
                            direction="right",
                            start=block_start,
                            end=block_end,
                            checkpoint=right_checkpoints[0][block_end],
                        )
                        hamiltonian_right = self._recompute_environment_block(
                            self._hamiltonian_frontier,
                            direction="right",
                            start=block_start,
                            end=block_end,
                            checkpoint=right_checkpoints[1][block_end],
                        )
                    for site in range(block_start, block_end):
                        if environment_cache == "recompute":
                            site_norm_right = self._message_at_cut(
                                self._norm_frontier,
                                site + 1,
                                direction="right",
                            )
                            site_hamiltonian_right = self._message_at_cut(
                                self._hamiltonian_frontier,
                                site + 1,
                                direction="right",
                            )
                        else:
                            site_norm_right = norm_right[site + 1]
                            site_hamiltonian_right = hamiltonian_right[site + 1]
                        environment = FrontierSiteEnvironment(
                            site=site,
                            norm_left=moving_norm,
                            norm_right=site_norm_right,
                            hamiltonian_left=moving_hamiltonian,
                            hamiltonian_right=site_hamiltonian_right,
                        )
                        updates.append(
                            self.optimize_site(
                                site,
                                metric_tol=metric_tol,
                                solver=solver,
                                matrix_free_threshold=matrix_free_threshold,
                                block_sparse_max_elements=block_sparse_max_elements,
                                eig_tol=active_eig_tol,
                                maxiter=maxiter,
                                max_subspace=max_subspace,
                                preconditioner=preconditioner,
                                block_size=block_size,
                                energy_before=self.energy,
                                environment=environment,
                            )
                        )
                        if enrich == "amen":
                            if pending_amen is not None:
                                pending = pending_amen
                                refresh, environment, retry = (
                                    self._finish_amen_retraction(
                                        pending,
                                        environment,
                                        site,
                                    )
                                )
                                bond_refreshes.append(refresh)
                                pending_amen = None
                                if retry:
                                    updates[-1] = self.optimize_site(
                                        site,
                                        metric_tol=metric_tol,
                                        solver=solver,
                                        matrix_free_threshold=(
                                            matrix_free_threshold
                                        ),
                                        block_sparse_max_elements=(
                                            block_sparse_max_elements
                                        ),
                                        eig_tol=active_eig_tol,
                                        maxiter=maxiter,
                                        max_subspace=max_subspace,
                                        preconditioner=preconditioner,
                                        block_size=block_size,
                                        energy_before=pending.energy_before,
                                        environment=environment,
                                    )
                            expansion, pending_amen = self._amen_expand_after_site(
                                site,
                                environment=environment,
                                direction="right",
                                rank=enrich_rank,
                                rtol=enrich_tol,
                                scale=enrich_scale,
                                refresh_saturated=amen_refresh_due,
                                metric_tol=metric_tol,
                                max_condition=gauge_max_condition,
                                energy_before=updates[-1].energy,
                            )
                            if expansion is not None:
                                bond_expansions.append(expansion)
                        moving_norm = self._norm_frontier.advance_left(
                            environment.norm_left,
                            self.tensors,
                            site,
                        )
                        moving_hamiltonian = self._hamiltonian_frontier.advance_left(
                            environment.hamiltonian_left,
                            self.tensors,
                            site,
                        )
            else:
                if gauge == "virtual":
                    self.canonicalize_virtual("left")
                    if not self.hamiltonian_contraction_is_exact:
                        previous = self.expectation()
                        self.energy = previous
                if environment_cache == "full":
                    norm_left = self._norm_frontier.build_left(self.tensors)
                    hamiltonian_left = self._hamiltonian_frontier.build_left(
                        self.tensors
                    )
                    left_checkpoints = None
                elif environment_cache == "checkpointed":
                    norm_left = hamiltonian_left = None
                    left_checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="left",
                            interval=environment_checkpoint_interval,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="left",
                            interval=environment_checkpoint_interval,
                        ),
                    )
                else:
                    norm_left = hamiltonian_left = left_checkpoints = None
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                checkpoint_cuts = (
                    (0, nsites)
                    if environment_cache == "recompute"
                    else self._environment_checkpoint_cuts(
                        environment_checkpoint_interval
                    )
                )
                for block_start, block_end in reversed(
                    tuple(zip(checkpoint_cuts[:-1], checkpoint_cuts[1:]))
                ):
                    if left_checkpoints is not None:
                        norm_left = self._recompute_environment_block(
                            self._norm_frontier,
                            direction="left",
                            start=block_start,
                            end=block_end,
                            checkpoint=left_checkpoints[0][block_start],
                        )
                        hamiltonian_left = self._recompute_environment_block(
                            self._hamiltonian_frontier,
                            direction="left",
                            start=block_start,
                            end=block_end,
                            checkpoint=left_checkpoints[1][block_start],
                        )
                    for site in range(block_end - 1, block_start - 1, -1):
                        if environment_cache == "recompute":
                            site_norm_left = self._message_at_cut(
                                self._norm_frontier,
                                site,
                                direction="left",
                            )
                            site_hamiltonian_left = self._message_at_cut(
                                self._hamiltonian_frontier,
                                site,
                                direction="left",
                            )
                        else:
                            site_norm_left = norm_left[site]
                            site_hamiltonian_left = hamiltonian_left[site]
                        environment = FrontierSiteEnvironment(
                            site=site,
                            norm_left=site_norm_left,
                            norm_right=moving_norm,
                            hamiltonian_left=site_hamiltonian_left,
                            hamiltonian_right=moving_hamiltonian,
                        )
                        updates.append(
                            self.optimize_site(
                                site,
                                metric_tol=metric_tol,
                                solver=solver,
                                matrix_free_threshold=matrix_free_threshold,
                                block_sparse_max_elements=block_sparse_max_elements,
                                eig_tol=active_eig_tol,
                                maxiter=maxiter,
                                max_subspace=max_subspace,
                                preconditioner=preconditioner,
                                block_size=block_size,
                                energy_before=self.energy,
                                environment=environment,
                            )
                        )
                        if enrich == "amen":
                            if pending_amen is not None:
                                pending = pending_amen
                                refresh, environment, retry = (
                                    self._finish_amen_retraction(
                                        pending,
                                        environment,
                                        site,
                                    )
                                )
                                bond_refreshes.append(refresh)
                                pending_amen = None
                                if retry:
                                    updates[-1] = self.optimize_site(
                                        site,
                                        metric_tol=metric_tol,
                                        solver=solver,
                                        matrix_free_threshold=(
                                            matrix_free_threshold
                                        ),
                                        block_sparse_max_elements=(
                                            block_sparse_max_elements
                                        ),
                                        eig_tol=active_eig_tol,
                                        maxiter=maxiter,
                                        max_subspace=max_subspace,
                                        preconditioner=preconditioner,
                                        block_size=block_size,
                                        energy_before=pending.energy_before,
                                        environment=environment,
                                    )
                            expansion, pending_amen = self._amen_expand_after_site(
                                site,
                                environment=environment,
                                direction="left",
                                rank=enrich_rank,
                                rtol=enrich_tol,
                                scale=enrich_scale,
                                refresh_saturated=amen_refresh_due,
                                metric_tol=metric_tol,
                                max_condition=gauge_max_condition,
                                energy_before=updates[-1].energy,
                            )
                            if expansion is not None:
                                bond_expansions.append(expansion)
                        moving_norm = self._norm_frontier.advance_right(
                            environment.norm_right,
                            self.tensors,
                            site,
                        )
                        moving_hamiltonian = self._hamiltonian_frontier.advance_right(
                            environment.hamiltonian_right,
                            self.tensors,
                            site,
                        )
            if pending_amen is not None:
                raise RuntimeError("AMEn temporary bond reached a sweep boundary.")
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
                if bond_refreshes:
                    candidate_energy = directional_endpoint_energy
                    start_energy = amen_sweep_snapshot["energy"]
                    acceptance_tolerance = (
                        512.0
                        * np.finfo(float).eps
                        * max(1.0, abs(start_energy))
                    )
                    amen_refresh_accepted = bool(
                        np.isfinite(candidate_energy)
                        and candidate_energy <= start_energy + acceptance_tolerance
                    )
                    if amen_refresh_accepted:
                        energy = candidate_energy
                    else:
                        self.tensors = [
                            tensor.copy()
                            for tensor in amen_sweep_snapshot["tensors"]
                        ]
                        self._virtual_bond_dims = tuple(
                            amen_sweep_snapshot["bond_dims"]
                        )
                        self.bond_dim = int(amen_sweep_snapshot["bond_dim"])
                        if amen_sweep_snapshot["layout"] is not None:
                            self.abelian_layout = amen_sweep_snapshot["layout"]
                            self.local_masks = self.abelian_layout.local_masks(
                                self.physical_groups
                            )
                            self._apply_bond_gauge_constraints()
                        self._null_reduced_cuts = set(
                            amen_sweep_snapshot["null_reduced_cuts"]
                        )
                        self._rebuild_frontier_engines()
                        norm = float(
                            np.real(self._norm_frontier.scalar(self.tensors))
                        )
                        energy = start_energy
                        bond_refreshes = tuple(
                            replace(record, accepted=False)
                            for record in bond_refreshes
                        )
                else:
                    energy = directional_endpoint_energy
                self.balance_gauges(state_norm=np.sqrt(norm))
            else:
                # Per-tensor gauge rescalings preserve the exact state but can
                # change a rank-truncated TT contraction through finite-rank
                # rounding.  Keep the gauge in which every proposal was checked.
                energy = self.expectation()
            self.energy = energy
            sweep_delta = abs(energy - previous)
            natural_gradient = None
            natural_gradient_quality_ratio = float("nan")
            natural_gradient_relative_gain = 0.0
            completed_sweeps = directional_sweep + 1
            if natural_gradient_adaptive:
                stagnation_trigger = bool(
                    sweep_delta <= tol
                    and (
                        natural_gradient_attempts == 0
                        or last_natural_gradient_was_useful
                    )
                )
                natural_gradient_due = bool(
                    natural_gradient_every
                    and (
                        completed_sweeps >= natural_gradient_next_sweep
                        or stagnation_trigger
                    )
                )
            else:
                natural_gradient_due = bool(
                    natural_gradient_every
                    and completed_sweeps % natural_gradient_every == 0
                )
            used_natural_gradient_trust_radius = (
                current_natural_gradient_trust_radius
            )
            if natural_gradient_due:
                natural_gradient_attempts += 1
                natural_gradient = self.natural_gradient_step(
                    metric_tol=metric_tol,
                    damping=natural_gradient_damping,
                    trust_radius=used_natural_gradient_trust_radius,
                    max_backtracks=natural_gradient_max_backtracks,
                    energy_before=energy,
                    state_norm=1.0,
                )
                energy = float(self.energy)
                actual_gain = max(
                    float(natural_gradient.energy_before - natural_gradient.energy),
                    0.0,
                )
                natural_gradient_relative_gain = actual_gain / max(
                    1.0,
                    abs(float(natural_gradient.energy_before)),
                )
                last_natural_gradient_was_useful = bool(
                    natural_gradient.accepted
                    and natural_gradient_relative_gain
                    >= natural_gradient_min_relative_gain
                )
                predicted_gain = (
                    -float(natural_gradient.step_size)
                    * float(natural_gradient.directional_derivative)
                )
                if predicted_gain > np.finfo(float).tiny:
                    natural_gradient_quality_ratio = actual_gain / predicted_gain
                if natural_gradient_adaptive:
                    low_utility = (
                        not natural_gradient.accepted
                        or natural_gradient_relative_gain
                        < natural_gradient_min_relative_gain
                    )
                    poor_model = (
                        natural_gradient.backtracks > 1
                        or (
                            np.isfinite(natural_gradient_quality_ratio)
                            and natural_gradient_quality_ratio < 0.25
                        )
                    )
                    good_model = bool(
                        natural_gradient.accepted
                        and natural_gradient.backtracks == 0
                        and np.isfinite(natural_gradient_quality_ratio)
                        and natural_gradient_quality_ratio > 0.75
                    )
                    if low_utility:
                        natural_gradient_interval = min(
                            natural_gradient_max_interval,
                            max(
                                natural_gradient_interval + 1,
                                2 * natural_gradient_interval,
                            ),
                        )
                        current_natural_gradient_trust_radius = max(
                            minimum_natural_gradient_trust_radius,
                            0.5 * current_natural_gradient_trust_radius,
                        )
                    elif poor_model:
                        natural_gradient_interval = min(
                            natural_gradient_max_interval,
                            natural_gradient_interval + 1,
                        )
                        current_natural_gradient_trust_radius = max(
                            minimum_natural_gradient_trust_radius,
                            0.5 * current_natural_gradient_trust_radius,
                        )
                    elif good_model:
                        natural_gradient_interval = max(
                            natural_gradient_every,
                            (natural_gradient_interval + 1) // 2,
                        )
                        current_natural_gradient_trust_radius = min(
                            maximum_natural_gradient_trust_radius,
                            1.25 * current_natural_gradient_trust_radius,
                        )
                    natural_gradient_next_sweep = (
                        completed_sweeps + natural_gradient_interval
                    )
                else:
                    natural_gradient_next_sweep = (
                        completed_sweeps + natural_gradient_every
                    )
            delta = abs(energy - previous)
            relative_sweep_gain = delta / max(1.0, abs(previous))
            solver_failures = sum(not update.solver_converged for update in updates)
            bond_regrowth_cooldown = tuple(
                sorted(set(getattr(self, "_null_reduced_cuts", ())))
            )
            maximum_bonds = tuple(
                getattr(self, "_maximum_bond_dims", self._bond_dims())
            )
            permanent_bond_expansions = tuple(
                expansion
                for expansion in bond_expansions
                if expansion.new_dimension <= maximum_bonds[expansion.cut]
            )
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
                    "natural_gradient_adaptive": natural_gradient_adaptive,
                    "natural_gradient_interval": natural_gradient_interval,
                    "natural_gradient_next_sweep": natural_gradient_next_sweep,
                    "natural_gradient_trust_radius": (
                        used_natural_gradient_trust_radius
                    ),
                    "natural_gradient_next_trust_radius": (
                        current_natural_gradient_trust_radius
                    ),
                    "natural_gradient_quality_ratio": (
                        natural_gradient_quality_ratio
                    ),
                    "natural_gradient_relative_gain": (
                        natural_gradient_relative_gain
                    ),
                    "gauge": gauge,
                    "gauge_weight": gauge_weight,
                    "gauge_update": gauge_update,
                    "bond_reductions": bond_reductions,
                    "bond_expansions": tuple(bond_expansions),
                    "permanent_bond_expansions": permanent_bond_expansions,
                    "bond_refreshes": bond_refreshes,
                    "amen_refresh_accepted": amen_refresh_accepted,
                    "bond_regrowth_cooldown": bond_regrowth_cooldown,
                    "enrich": None if enrich == "none" else enrich,
                    "enrich_rank": enrich_rank,
                    "enrich_tol": enrich_tol,
                    "enrich_scale": enrich_scale,
                    "enrich_every": enrich_every,
                    "enrich_trigger": enrich_trigger,
                    "amen_refresh_scheduled": amen_refresh_scheduled,
                    "amen_refresh_due": amen_refresh_due,
                    "relative_sweep_gain": relative_sweep_gain,
                    "cycle": directional_sweep // 2,
                    "cycle_complete": bool(
                        cycle_started and directional_sweep % 2 == 1
                    ),
                    "cycle_start_energy": float(cycle_start_energy),
                    "cycle_delta": (
                        abs(energy - cycle_start_energy)
                        if cycle_started and directional_sweep % 2 == 1
                        else None
                    ),
                    "adaptive_solver": adaptive_solver,
                    "eig_tol": active_eig_tol,
                    "block_size": block_size,
                    "preconditioner": (
                        "callable" if callable(preconditioner) else preconditioner
                    ),
                    "bond_dims": self.bond_dims,
                    "contraction_is_exact": self.contraction_is_exact,
                    "norm_contraction_is_exact": self.norm_contraction_is_exact,
                    "hamiltonian_contraction_is_exact": (
                        self.hamiltonian_contraction_is_exact
                    ),
                    "hamiltonian_action_is_hermitian": (
                        self.hamiltonian_action_is_hermitian
                    ),
                    "directional_endpoint_energy": directional_endpoint_energy,
                    "environment_cache": environment_cache,
                    "environment_cache_requested": requested_environment_cache,
                    "environment_memory": environment_memory,
                    "full_environment_bytes": full_environment_bytes,
                    "environment_checkpoint_interval": (
                        environment_checkpoint_interval
                    ),
                    "fixed_environment_cache_elements": (
                        self.fixed_environment_cache_elements(
                            mode=environment_cache,
                            interval=environment_checkpoint_interval,
                        )
                    ),
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
            # Reduction suppresses only same-pass regrowth.  The opposite
            # sweep has a changed residual and may legitimately reopen a cut.
            self._null_reduced_cuts = set()
            last_relative_sweep_gain = relative_sweep_gain
            natural_gradient_stationary = (
                natural_gradient is None
                or natural_gradient.accepted
                or abs(natural_gradient.directional_derivative) <= tol
            )
            directional_stationary = bool(
                delta <= tol
                and solver_failures == 0
                and natural_gradient_stationary
                and not bond_reductions
                and not permanent_bond_expansions
                and amen_refresh_accepted
                and (enrich != "amen" or amen_refresh_due)
            )
            cycle_stationary = bool(cycle_stationary and directional_stationary)
            cycle_complete = bool(
                cycle_started and directional_sweep % 2 == 1
            )
            cycle_delta = (
                abs(energy - cycle_start_energy) if cycle_complete else None
            )
            self.history[-1]["cycle_stationary"] = (
                cycle_stationary if cycle_complete else None
            )
            if cycle_complete and cycle_stationary and cycle_delta <= tol:
                self.converged = True
                break
            if cycle_complete:
                cycle_started = False
            previous = energy
        return self

    def state_vector(self, *, normalize=False):
        """Build the explicit state vector for small-system validation only."""
        configs = np.asarray(list(np.ndindex(*self.dims)), dtype=np.intp)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        environment = np.ones((len(configs), 1), dtype=dtype)
        for site, physical_sites in enumerate(self.physical_groups):
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
    "FrontierBondRefresh",
    "FrontierTieReduction",
    "FrontierGaugeUpdate",
    "FrontierNaturalGradientUpdate",
    "FrontierSiteEnvironment",
    "FrontierSiteUpdate",
    "FrontierTiedLETTA",
    "FrontierTwoSiteUpdate",
]
