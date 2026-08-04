"""Exact and tensor-train graph-tied LETTA frontier contraction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg, lsmr

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
from .local_terms import LocalHamiltonian, LocalMPO, LocalMPOProduct
from .matrix_free import (
    lowest_generalized_davidson,
    lowest_recycled_block_davidson,
)
from .mpo_frontier import MPOFrontier
from .physical_blocks import (
    PhysicalBlockGeneralizedProblem,
    PhysicalBlockLayout,
    PhysicalBlockLinearOperator,
    hamiltonian_physical_connectivity,
)
from .renormalized_frontier import TermRenormalizedFrontier
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
    hamiltonian_action_calls: int = 0
    hamiltonian_batch_calls: int = 0
    recycled_vectors: int = 0
    preconditioner_blocks: int = 0


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
    action_relative_residual: float = float("nan")
    verification_kind: str = "metric_dual"
    metric_rank_complete: bool = True
    hamiltonian_action_calls: int = 0
    hamiltonian_vector_products: int = 0
    hamiltonian_batch_calls: int = 0
    recycled_vectors: int = 0
    preconditioner_blocks: int = 0


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
class _FrontierPairRightEnvironment:
    """Fixed outer-right messages bound to one merged-pair topology."""

    sites: tuple[int, int]
    union_sites: tuple[int, ...]
    norm_right: object
    hamiltonian_right: object


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
    merged_solve: FrontierMergedSolveDiagnostics | None = None
    outer_cycles: int = 1
    merged_energy_history: tuple[float, ...] = ()
    factor_energy_history: tuple[float, ...] = ()
    pair_operator_backend: str = "dense"
    pair_operator_stored_elements: int = 0
    pair_operator_peak_elements: int = 0
    pair_operator_stored_bytes: int = 0
    pair_operator_peak_bytes: int = 0
    wall_time_seconds: float = 0.0
    operator_assembly_seconds: float = 0.0
    merged_solve_seconds: float = 0.0
    split_seconds: float = 0.0
    pair_operator_requested_backend: str = "auto"
    pair_operator_selection_reason: str = ""
    dense_estimated_peak_bytes: int = 0
    matrix_free_estimated_peak_bytes: int = 0
    pair_action_backend: str = ""
    factor_solver: str = "auto"
    pair_operator_workers: int = 1


class FrontierTiedLETTA:
    r"""Unrestricted graph-tied LETTA contracted by frontier messages.

    This class represents the same local tensors as :class:`DenseTiedLETTA`,
    but it accepts a :class:`LocalHamiltonian` and never constructs the
    many-body configuration table during initialization or optimization.
    The local-term sum is converted to an exact finite-state operator
    network.  Numerical left/right double-layer messages are cached across
    each directional sweep and reused by every local matrix or Davidson
    action.  The ``compressed``, untruncated ``identity_block``, and
    complementary-operator ``renormalized`` backends are exact.  The latter
    compiles one- and two-site terms directly without first constructing the
    generic Hamiltonian MPO.  The ``tensor_train`` backend
    keeps the cheaper norm frontier exact by default and stores the Hamiltonian
    frontier as a boundary MPS/TT.  It is fully exact only when its ranks and
    tolerances are unrestricted.  For the identity-block backend,
    ``local_backend='tensor_train'`` can similarly truncate each absorbed
    graph tensor with ``local_rank``, ``local_rtol``, and ``local_atol``;
    candidate updates are still accepted against an exact block contraction.
    The dense-frontier cost is governed
    by the weighted frontier induced by the chosen site ordering and MPO bond;
    it is exponential in that width, which can still grow with system size for
    dense or poorly ordered graphs.
    """

    _preserve_pair_metric_null_components = False

    def __init__(
        self,
        hamiltonian: LocalHamiltonian,
        dims=None,
        parent_sets=None,
        *,
        bond_dim: int = 1,
        bond_dims=None,
        tensors=None,
        seed: int | None = None,
        frontier_backend="compressed",
        path_optimizer="greedy",
        local_backend="dense",
        local_rank: int | None = None,
        local_rtol: float = 0.0,
        local_atol: float = 0.0,
        max_rank: int | None = None,
        rtol: float = 0.0,
        atol: float = 0.0,
        transfer_max_rank: int | None = None,
        transfer_rtol: float = 0.0,
        transfer_atol: float = 0.0,
        tt_absorption="structured",
        tt_norm_backend="exact",
        tt_hermitize: bool = True,
        _norm_mpo: LocalMPO | None = None,
        _objective_mpo: LocalMPO | LocalMPOProduct | None = None,
        _objective_is_hermitian: bool | None = None,
        _balance_initial_gauges: bool = True,
    ):
        if not isinstance(hamiltonian, LocalHamiltonian):
            raise TypeError("hamiltonian must be a LocalHamiltonian.")
        if parent_sets is None:
            raise TypeError("parent_sets is required.")
        if dims is None:
            dims = hamiltonian.dims
        self.dims = _validated_dims(dims)
        self.symmetry = None
        if hamiltonian.dims != self.dims:
            raise ValueError("hamiltonian dims are inconsistent with dims.")
        self.hamiltonian = hamiltonian
        if (_norm_mpo is None) != (_objective_mpo is None):
            raise TypeError("_norm_mpo and _objective_mpo must be supplied together.")
        if _norm_mpo is not None and not isinstance(_norm_mpo, LocalMPO):
            raise TypeError("_norm_mpo must be a LocalMPO.")
        if _objective_mpo is not None and not isinstance(
            _objective_mpo,
            (LocalMPO, LocalMPOProduct),
        ):
            raise TypeError(
                "_objective_mpo must be a LocalMPO or LocalMPOProduct."
            )
        for name, mpo in (
            ("_norm_mpo", _norm_mpo),
            ("_objective_mpo", _objective_mpo),
        ):
            if mpo is not None and mpo.dims != self.dims:
                raise ValueError(f"{name} dimensions are inconsistent with dims.")
        self._uses_custom_operator_mpos = _norm_mpo is not None
        if _objective_is_hermitian is None:
            _objective_is_hermitian = not self._uses_custom_operator_mpos
        if not isinstance(_objective_is_hermitian, (bool, np.bool_)):
            raise TypeError("_objective_is_hermitian must be boolean or None.")
        self._objective_is_hermitian = bool(_objective_is_hermitian)
        if hamiltonian.sites is not None:
            self.sites = tuple(hamiltonian.sites)
            self.physical_legs = (
                hamiltonian.physical_legs
                if hasattr(hamiltonian, "physical_legs")
                else tuple(
                    site.physical_leg if hasattr(site, "physical_leg") else None
                    for site in self.sites
                )
            )
        else:
            self.sites = None
            self.physical_legs = None
        self.norm_mpo = _norm_mpo
        self.objective_mpo = _objective_mpo
        if not isinstance(_balance_initial_gauges, (bool, np.bool_)):
            raise TypeError("_balance_initial_gauges must be boolean.")
        self._balance_initial_gauges = bool(_balance_initial_gauges)
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.physical_sites = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        self._physical_block_connectivity_cache = {}
        self._pair_support_cache = {}
        self._pair_block_mask_cache = {}
        self._pair_davidson_start_cache = {}
        self._pair_matrix_free_recycle_cache = {}
        self._pair_backend_profile_cache = {}
        self._pair_block_request_cache = {}
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
        self.local_backend = str(local_backend).lower().replace("-", "_")
        if self.local_backend in {"tt", "sequential"}:
            self.local_backend = "tensor_train"
        if self.local_backend not in {"dense", "tensor_train"}:
            raise ValueError("local_backend must be 'dense' or 'tensor_train'.")
        self.local_options = {
            "rank": None if local_rank is None else int(local_rank),
            "rtol": float(local_rtol),
            "atol": float(local_atol),
        }
        self.frontier_backend = str(frontier_backend).lower().replace("-", "_")
        if self.frontier_backend in {"tt", "boundary_mps"}:
            self.frontier_backend = "tensor_train"
        if self.frontier_backend in {
            "narg",
            "term_recursive",
            "term_renormalized",
        }:
            self.frontier_backend = "renormalized"
        if self.frontier_backend not in {
            "compressed",
            "identity_block",
            "renormalized",
            "tensor_train",
        }:
            raise ValueError(
                "frontier_backend must be 'compressed', 'identity_block', "
                "'renormalized', or 'tensor_train'."
            )
        if self._uses_custom_operator_mpos and self.frontier_backend == "renormalized":
            raise NotImplementedError(
                "custom norm/objective MPOs currently require the compressed, "
                "identity-block, or tensor-train frontier backend."
            )
        if (
            self.local_options["rank"] is not None
            and self.local_options["rank"] < 1
        ):
            raise ValueError("local_rank must be positive or None.")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (
                self.local_options["rtol"],
                self.local_options["atol"],
            )
        ):
            raise ValueError("local_rtol and local_atol must be finite and nonnegative.")
        if (
            self.local_options["rank"] is not None
            or self.local_options["rtol"]
            or self.local_options["atol"]
        ) and not (
            self.frontier_backend in {"identity_block", "renormalized"}
            and self.local_backend == "tensor_train"
        ):
            raise ValueError(
                "local_rank/local_rtol/local_atol require an identity-aware "
                "frontier with local_backend='tensor_train'."
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
        if self.norm_mpo is None:
            self.norm_mpo = LocalMPO(
                self.dims,
                [
                    np.eye(dim, dtype=parameter_dtype)[None, None, :, :]
                    for dim in self.dims
                ],
            )
        frontier_arguments = (self.dims, self.physical_sites, shapes)
        uncompressed_hamiltonian_mpo = None
        compressed_hamiltonian_mpo = None
        if self.frontier_backend != "renormalized":
            uncompressed_hamiltonian_mpo = (
                self.objective_mpo
                if self.objective_mpo is not None
                else self.hamiltonian.to_mpo()
            )
            self.objective_mpo = uncompressed_hamiltonian_mpo
            self.uncompressed_hamiltonian_mpo_bond_dim = max(
                uncompressed_hamiltonian_mpo.bond_dims
            )
            self.hamiltonian_mpo_compression_performed = not (
                self._uses_custom_operator_mpos
                and self.frontier_backend == "identity_block"
            )
            if self.hamiltonian_mpo_compression_performed:
                materialized = (
                    uncompressed_hamiltonian_mpo.materialize()
                    if isinstance(
                        uncompressed_hamiltonian_mpo,
                        LocalMPOProduct,
                    )
                    else uncompressed_hamiltonian_mpo
                )
                compressed_hamiltonian_mpo = materialized.compress()
            else:
                compressed_hamiltonian_mpo = uncompressed_hamiltonian_mpo
            self.compressed_hamiltonian_mpo_bond_dim = max(
                compressed_hamiltonian_mpo.bond_dims
            )
        if self.tt_norm_backend == "tensor_train":
            self._norm_frontier = TTMPOFrontier(
                *frontier_arguments,
                self.norm_mpo.tensors,
                paired_sites=(),
                optimize=path_optimizer,
                **self.tt_options,
            )
        else:
            self._norm_frontier = MPOFrontier(
                *frontier_arguments,
                self.norm_mpo.tensors,
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
            self._hamiltonian_frontier = self._new_identity_block_frontier(
                frontier_arguments,
                self.hamiltonian_mpo,
            )
        elif self.frontier_backend == "renormalized":
            self.hamiltonian_mpo = None
            self._hamiltonian_frontier = TermRenormalizedFrontier(
                self.hamiltonian,
                self.physical_sites,
                shapes,
                optimize=path_optimizer,
                local_backend=self.local_backend,
                local_rank=self.local_options["rank"],
                local_rtol=self.local_options["rtol"],
                local_atol=self.local_options["atol"],
            )
            renormalized_bond_dim = max(
                self._hamiltonian_frontier.mpo_bonds
            )
            self.uncompressed_hamiltonian_mpo_bond_dim = (
                renormalized_bond_dim
            )
            self.compressed_hamiltonian_mpo_bond_dim = (
                renormalized_bond_dim
            )
        else:
            self.hamiltonian_mpo = compressed_hamiltonian_mpo
            if self._uses_custom_operator_mpos:
                self._hamiltonian_frontier = TTMPOFrontier(
                    *frontier_arguments,
                    self.hamiltonian_mpo.tensors,
                    optimize=path_optimizer,
                    **self.tt_options,
                )
            else:
                self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                    self.hamiltonian,
                    self.physical_sites,
                    shapes,
                    optimize=path_optimizer,
                    **self.tt_options,
                )
        self._exact_hamiltonian_frontier = None
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and not self._hamiltonian_frontier.contraction_is_exact
        ):
            self._exact_hamiltonian_frontier = self._new_identity_block_frontier(
                frontier_arguments,
                self.hamiltonian_mpo,
                exact=True,
            )
        self._pair_plan_cache: dict[int, _FrontierPairPlan] = {}
        self.history: list[dict] = []
        self.energy: float | None = None
        self.converged = False
        if self._balance_initial_gauges:
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

    def _new_identity_block_frontier(
        self,
        frontier_arguments,
        mpo,
        *,
        exact=False,
    ):
        options = {
            "optimize": self.path_optimizer,
            "local_backend": "dense" if exact else self.local_backend,
            "local_rank": None if exact else self.local_options["rank"],
            "local_rtol": 0.0 if exact else self.local_options["rtol"],
            "local_atol": 0.0 if exact else self.local_options["atol"],
        }
        if isinstance(mpo, LocalMPOProduct):
            return BlockMPOFrontier.from_product(
                *frontier_arguments,
                mpo.left.tensors,
                mpo.right.tensors,
                **options,
            )
        return BlockMPOFrontier(
            *frontier_arguments,
            mpo.tensors,
            **options,
        )

    @property
    def uses_tensor_train_frontier(self) -> bool:
        return isinstance(self._norm_frontier, TTMPOFrontier) or isinstance(
            self._hamiltonian_frontier,
            TTMPOFrontier,
        )

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
            local_backend=self.local_backend,
            local_rank=self.local_options["rank"],
            local_rtol=self.local_options["rtol"],
            local_atol=self.local_options["atol"],
            max_rank=self.tt_options["max_rank"],
            rtol=self.tt_options["rtol"],
            atol=self.tt_options["atol"],
            transfer_max_rank=self.tt_options["transfer_max_rank"],
            transfer_rtol=self.tt_options["transfer_rtol"],
            transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
            _norm_mpo=self.norm_mpo if self._uses_custom_operator_mpos else None,
            _objective_mpo=(
                self.objective_mpo if self._uses_custom_operator_mpos else None
            ),
            _objective_is_hermitian=self._objective_is_hermitian,
            _balance_initial_gauges=False,
        )
        # Construction was asked to preserve the supplied representation.
        result.history = list(self.history)
        result.converged = self.converged
        result.symmetry = deepcopy(self.symmetry)
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
        frontier_arguments = (self.dims, self.physical_sites, shapes)
        if self.tt_norm_backend == "tensor_train":
            self._norm_frontier = TTMPOFrontier(
                *frontier_arguments,
                self.norm_mpo.tensors,
                paired_sites=(),
                optimize=self.path_optimizer,
                **self.tt_options,
            )
        else:
            self._norm_frontier = MPOFrontier(
                *frontier_arguments,
                self.norm_mpo.tensors,
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
            self._hamiltonian_frontier = self._new_identity_block_frontier(
                frontier_arguments,
                self.hamiltonian_mpo,
            )
        elif self.frontier_backend == "renormalized":
            self._hamiltonian_frontier = TermRenormalizedFrontier(
                self.hamiltonian,
                self.physical_sites,
                shapes,
                optimize=self.path_optimizer,
                local_backend=self.local_backend,
                local_rank=self.local_options["rank"],
                local_rtol=self.local_options["rtol"],
                local_atol=self.local_options["atol"],
            )
        else:
            if self._uses_custom_operator_mpos:
                self._hamiltonian_frontier = TTMPOFrontier(
                    *frontier_arguments,
                    self.hamiltonian_mpo.tensors,
                    optimize=self.path_optimizer,
                    **self.tt_options,
                )
            else:
                self._hamiltonian_frontier = TermwiseTTMPOFrontier(
                    self.hamiltonian,
                    self.physical_sites,
                    shapes,
                    optimize=self.path_optimizer,
                    **self.tt_options,
                )
        self._exact_hamiltonian_frontier = None
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and not self._hamiltonian_frontier.contraction_is_exact
        ):
            self._exact_hamiltonian_frontier = self._new_identity_block_frontier(
                frontier_arguments,
                self.hamiltonian_mpo,
                exact=True,
            )
        self._pair_plan_cache = {}
        self._pair_support_cache = {}
        self._pair_block_mask_cache = {}
        self._pair_davidson_start_cache = {}
        self._pair_matrix_free_recycle_cache = {}
        self._pair_backend_profile_cache = {}
        self._pair_block_request_cache = {}

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

    def _pair_plan_fingerprint(self):
        hamiltonian_mpo_shapes = (
            (
                "factorized",
                tuple(
                    tuple(tensor.shape)
                    for tensor in self._hamiltonian_frontier.left_mpo_tensors
                ),
                tuple(
                    tuple(tensor.shape)
                    for tensor in self._hamiltonian_frontier.right_mpo_tensors
                ),
            )
            if (
                isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
                and self._hamiltonian_frontier.factorized_mpo
            )
            else tuple(
                tuple(tensor.shape)
                for tensor in (
                    getattr(
                        self._hamiltonian_frontier,
                        "mpo_tensors",
                        (),
                    )
                    or ()
                )
            )
        )
        return (
            self.physical_sites,
            tuple(tuple(tensor.shape) for tensor in self.tensors),
            self.frontier_backend,
            tuple(
                tuple(tensor.shape)
                for tensor in getattr(self._norm_frontier, "mpo_tensors", ())
            ),
            hamiltonian_mpo_shapes,
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
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
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
        pair_physical_sites = list(self.physical_sites)
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
        if isinstance(self._hamiltonian_frontier, TermRenormalizedFrontier):
            hamiltonian_engine = self._hamiltonian_frontier.bind_pair(
                site,
                union_sites,
                merged_shape,
            )
        elif isinstance(self._hamiltonian_frontier, BlockMPOFrontier):
            # The temporary tensor at ``site`` owns both pair physical legs,
            # while the MPO cut after it has advanced through only one site.
            # A one-site virtual charge map is therefore not valid on this
            # auxiliary network.
            hamiltonian_engine = self._new_identity_block_frontier(
                arguments,
                self.hamiltonian_mpo,
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

    def _bind_pair_right_environment(
        self,
        site,
        norm_outer_right,
        hamiltonian_outer_right,
        *,
        hamiltonian_workers=1,
        hamiltonian_executor=None,
    ) -> _FrontierPairRightEnvironment:
        """Bind the fixed outer-right side to one merged-pair topology."""
        plan = self._pair_plan(site)
        following = int(site) + 1
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and self._hamiltonian_frontier.charge_resolved
            and not getattr(plan.hamiltonian_engine, "charge_resolved", False)
        ):
            hamiltonian_outer_right = (
                self._hamiltonian_frontier.expand_virtual_pairs(
                    hamiltonian_outer_right
                )
            )
        pair_tensors = list(self.tensors)
        pair_tensors[following] = plan.identity_tensor
        norm_right = plan.norm_engine.advance_right(
            norm_outer_right,
            pair_tensors,
            following,
        )
        if getattr(plan.hamiltonian_engine, "uses_outer_messages", False):
            hamiltonian_right = hamiltonian_outer_right
        elif isinstance(plan.hamiltonian_engine, BlockMPOFrontier):
            hamiltonian_right = (
                plan.hamiltonian_engine.advance_right_identity(
                    hamiltonian_outer_right,
                    following,
                    max_workers=hamiltonian_workers,
                    executor=hamiltonian_executor,
                )
            )
        else:
            hamiltonian_right = plan.hamiltonian_engine.advance_right(
                hamiltonian_outer_right,
                pair_tensors,
                following,
            )
        return _FrontierPairRightEnvironment(
            sites=(int(site), following),
            union_sites=plan.union_sites,
            norm_right=norm_right,
            hamiltonian_right=hamiltonian_right,
        )

    def _pair_environment_from_bound_right(
        self,
        site,
        norm_left,
        hamiltonian_left,
        bound_right,
    ) -> FrontierPairEnvironment:
        """Combine a moving left side with a bound fixed right side."""
        site = int(site)
        plan = self._pair_plan(site)
        following = site + 1
        if not isinstance(bound_right, _FrontierPairRightEnvironment):
            raise TypeError(
                "bound_right must be a merged-pair right environment."
            )
        if (
            bound_right.sites != (site, following)
            or bound_right.union_sites != plan.union_sites
        ):
            raise ValueError("bound right environment belongs to another pair.")
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and self._hamiltonian_frontier.charge_resolved
            and not getattr(plan.hamiltonian_engine, "charge_resolved", False)
        ):
            hamiltonian_left = self._hamiltonian_frontier.expand_virtual_pairs(
                hamiltonian_left
            )
        return FrontierPairEnvironment(
            sites=(site, following),
            union_sites=plan.union_sites,
            norm_left=norm_left,
            norm_right=bound_right.norm_right,
            hamiltonian_left=hamiltonian_left,
            hamiltonian_right=bound_right.hamiltonian_right,
        )

    def _pair_environment_from_outer_messages(
        self,
        site,
        norm_left,
        norm_outer_right,
        hamiltonian_left,
        hamiltonian_outer_right,
        *,
        hamiltonian_workers=1,
        hamiltonian_executor=None,
    ) -> FrontierPairEnvironment:
        """Bind cached outer messages to an adjacent merged-pair plan."""
        bound_right = self._bind_pair_right_environment(
            site,
            norm_outer_right,
            hamiltonian_outer_right,
            hamiltonian_workers=hamiltonian_workers,
            hamiltonian_executor=hamiltonian_executor,
        )
        return self._pair_environment_from_bound_right(
            site,
            norm_left,
            hamiltonian_left,
            bound_right,
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
        hamiltonian_workers=1,
        copy_backend="auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact merged-pair ``(N_eff, H_eff)`` from cached plans."""
        site = int(site)
        hamiltonian_workers = int(hamiltonian_workers)
        if hamiltonian_workers < 1:
            raise ValueError("hamiltonian_workers must be positive.")
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        metric = plan.norm_engine.hole_matrix(
            site,
            environment.norm_left,
            environment.norm_right,
        )
        hamiltonian_arguments = (
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
        )
        if isinstance(plan.hamiltonian_engine, BlockMPOFrontier):
            effective = plan.hamiltonian_engine.hole_matrix(
                *hamiltonian_arguments,
                max_workers=hamiltonian_workers,
                copy_backend=copy_backend,
            )
        else:
            effective = plan.hamiltonian_engine.hole_matrix(
                *hamiltonian_arguments
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

    def pair_hamiltonian_actions(self, site: int, vectors, *, environment=None):
        """Apply the merged-pair Hamiltonian to a column batch."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        vectors = np.asarray(vectors)
        merged_size = int(np.prod(plan.merged_shape))
        if vectors.ndim != 2 or vectors.shape[0] != merged_size:
            raise ValueError(
                "merged-pair vectors must have shape (merged_size, nvec)."
            )
        batch_action = getattr(
            plan.hamiltonian_engine,
            "hole_action_batch",
            None,
        )
        if batch_action is not None:
            values = batch_action(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                vectors,
            )
            values = np.asarray(values)
            if values.shape != vectors.shape:
                raise ValueError(
                    "batched pair Hamiltonian action returned an invalid shape."
                )
            return values
        return np.column_stack(
            [
                plan.hamiltonian_engine.hole_action(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    vectors[:, column],
                )
                for column in range(vectors.shape[1])
            ]
        )

    def pair_hamiltonian_support_actions(
        self,
        site: int,
        support,
        vectors,
        *,
        environment=None,
    ):
        """Apply the pair Hamiltonian to packed support-vector columns."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        merged_size = int(np.prod(plan.merged_shape))
        support = np.asarray(support, dtype=np.intp).reshape(-1)
        if (
            np.any(support < 0)
            or np.any(support >= merged_size)
            or np.unique(support).size != support.size
        ):
            raise ValueError("support must contain unique merged-pair indices.")
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != support.size:
            raise ValueError(
                "packed pair vectors must have shape (support_size, nvec)."
            )
        support_action = getattr(
            plan.hamiltonian_engine,
            "hole_action_support_batch",
            None,
        )
        if support_action is not None:
            values = support_action(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                support,
                vectors,
            )
            values = np.asarray(values)
            if values.shape != vectors.shape:
                raise ValueError(
                    "packed pair Hamiltonian action returned an invalid shape."
                )
            return values
        lifted = np.zeros(
            (merged_size, vectors.shape[1]),
            dtype=np.result_type(self.hamiltonian.dtype, vectors),
        )
        lifted[support] = vectors
        return self.pair_hamiltonian_actions(
            site,
            lifted,
            environment=environment,
        )[support]

    def pair_hamiltonian_support_operator(
        self,
        site: int,
        support,
        *,
        environment=None,
        action_backend="auto",
        expected_action_calls=0,
        prepared_min_action_calls=7,
        action_batch_size=2,
    ):
        """Prepare an environment-bound action on packed pair support."""
        site = int(site)
        plan = self._pair_plan(site)
        environment = self._resolved_pair_environment(site, environment)
        merged_size = int(np.prod(plan.merged_shape))
        support = np.asarray(support, dtype=np.intp).reshape(-1)
        if (
            np.any(support < 0)
            or np.any(support >= merged_size)
            or np.unique(support).size != support.size
        ):
            raise ValueError("support must contain unique merged-pair indices.")
        prepare = getattr(
            plan.hamiltonian_engine,
            "prepare_hole_action_support",
            None,
        )
        fused = getattr(
            plan.hamiltonian_engine,
            "hole_action_support_fused_batch",
            None,
        )
        action_backend = str(action_backend).lower().replace("-", "_")
        if action_backend not in {"auto", "full", "fused", "prepared"}:
            raise ValueError(
                "action_backend must be 'auto', 'full', 'fused', or "
                "'prepared'."
            )
        expected_action_calls = int(expected_action_calls)
        prepared_min_action_calls = int(prepared_min_action_calls)
        action_batch_size = int(action_batch_size)
        if expected_action_calls < 0:
            raise ValueError("expected_action_calls must be nonnegative.")
        if prepared_min_action_calls < 1:
            raise ValueError(
                "prepared_min_action_calls must be positive."
            )
        if action_batch_size < 1:
            raise ValueError("action_batch_size must be positive.")
        selected = action_backend
        scaled_prepared_min_calls = prepared_min_action_calls
        if selected == "auto":
            virtual_size = (
                plan.merged_shape[0] * plan.merged_shape[1]
            )
            scaled_prepared_min_calls = (
                prepared_min_action_calls
                * max(1, (virtual_size + 15) // 16)
            )
            if (
                prepare is not None
                and expected_action_calls >= scaled_prepared_min_calls
            ):
                selected = "prepared"
            else:
                selected = "full"
        if selected == "prepared" and prepare is not None:
            action = prepare(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                support,
            )
            action.backend = "prepared_support_csr"
            action.selection_threshold = scaled_prepared_min_calls
            return action
        if selected == "fused" and fused is not None:

            def action(vectors):
                return fused(
                    site,
                    environment.hamiltonian_left,
                    environment.hamiltonian_right,
                    support,
                    vectors,
                )

            action.backend = "fused_support"
            estimator = getattr(
                plan.hamiltonian_engine,
                "fused_support_action_workspace_elements",
                None,
            )
            workspace = (
                {}
                if estimator is None
                else estimator(support, action_batch_size)
            )
            action.stored_elements = int(
                workspace.get("cached_selector_elements", 0)
            )
            action.peak_elements = int(
                workspace.get("upper_bound_elements", 0)
            )
            action.workspace_diagnostics = workspace
        else:

            def action(vectors):
                vectors = np.asarray(vectors)
                if vectors.ndim != 2 or vectors.shape[0] != support.size:
                    raise ValueError(
                        "packed pair vectors must have shape "
                        "(support_size, nvec)."
                    )
                lifted = np.zeros(
                    (merged_size, vectors.shape[1]),
                    dtype=np.result_type(
                        self.hamiltonian.dtype,
                        vectors,
                    ),
                )
                lifted[support] = vectors
                return self.pair_hamiltonian_actions(
                    site,
                    lifted,
                    environment=environment,
                )[support]

            action.backend = "full_scatter"
            action.stored_elements = 0
            action.peak_elements = int(
                2 * merged_size * action_batch_size
            )
            action.workspace_diagnostics = {}
        if not hasattr(action, "stored_elements"):
            action.stored_elements = 0
        if not hasattr(action, "peak_elements"):
            action.peak_elements = 0
        action.assembly_seconds = 0.0
        action.connected_blocks = 0
        action.selection_threshold = scaled_prepared_min_calls
        action.dtype = np.dtype(
            np.result_type(self.hamiltonian.dtype, *self.tensors)
        )
        return action

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
        request_key = (
            site,
            plan.union_sites,
            plan.merged_shape,
            self._objective_is_hermitian,
        )
        cached_requests = self._pair_block_request_cache.get(request_key)
        if cached_requests is None:
            pairs = hamiltonian_physical_connectivity(
                self.hamiltonian,
                plan.union_sites,
            )
            pairs = {(int(row), int(column)) for row, column in pairs}
            pairs |= {(column, row) for row, column in pairs}
            if any(
                row < 0
                or row >= layout.nblocks
                or column < 0
                or column >= layout.nblocks
                for row, column in pairs
            ):
                raise ValueError("hamiltonian_pairs contains an invalid block index.")
            metric_requests = tuple(
                (
                    block,
                    block,
                    layout.configurations[block],
                    layout.configurations[block],
                )
                for block in range(layout.nblocks)
            )
            hamiltonian_requests = tuple(
                (
                    row,
                    column,
                    layout.configurations[row],
                    layout.configurations[column],
                )
                for row, column in sorted(pairs)
                # Standard Hamiltonians and the charge-conserving H P_Q
                # objective are Hermitian. Contract one orientation in that
                # established case; arbitrary internal custom MPO objectives
                # retain both orientations for explicit hermitization.
                if not self._objective_is_hermitian or row <= column
            )
            cached_requests = (
                metric_requests,
                hamiltonian_requests,
                tuple(sorted(pairs)),
            )
            self._pair_block_request_cache[request_key] = cached_requests
        metric_requests, hamiltonian_requests, hamiltonian_pairs = (
            cached_requests
        )
        metric_raw = plan.norm_engine.hole_blocks(
            site,
            environment.norm_left,
            environment.norm_right,
            metric_requests,
        )
        hamiltonian_raw = plan.hamiltonian_engine.hole_blocks(
            site,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
            hamiltonian_requests,
        )
        metric_blocks = {
            (block, block): 0.5 * (value + value.T.conj())
            for (block, _column), value in metric_raw.items()
        }
        hamiltonian_blocks = {}
        if self._objective_is_hermitian:
            for row, column, _bra, _ket in hamiltonian_requests:
                value = np.asarray(hamiltonian_raw[(row, column)])
                if row == column:
                    hamiltonian_blocks[(row, row)] = 0.5 * (
                        value + value.T.conj()
                    )
                    continue
                hamiltonian_blocks[(row, column)] = value
                hamiltonian_blocks[(column, row)] = value.T.conj()
        else:
            for row, column in hamiltonian_pairs:
                if row > column:
                    continue
                forward = np.asarray(hamiltonian_raw[(row, column)])
                if row == column:
                    value = 0.5 * (forward + forward.T.conj())
                else:
                    reverse = np.asarray(hamiltonian_raw[(column, row)])
                    value = 0.5 * (forward + reverse.T.conj())
                hamiltonian_blocks[(row, column)] = value
                hamiltonian_blocks[(column, row)] = value.T.conj()
        dtype = np.result_type(
            self.tensors[site].dtype,
            self.tensors[site + 1].dtype,
            self.hamiltonian.dtype,
        )
        return PhysicalBlockGeneralizedProblem(
            layout,
            PhysicalBlockLinearOperator(layout, metric_blocks, dtype=dtype),
            PhysicalBlockLinearOperator(layout, hamiltonian_blocks, dtype=dtype),
        )

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
        site = int(site)
        following = site + 1
        union_sites = tuple(int(index) for index in union_sites)
        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        if left_tensor.shape[1] != right_tensor.shape[0]:
            raise ValueError("the pair factors have inconsistent middle dimensions.")
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
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

    def _pair_factor_metric_operator(
        self,
        site,
        union_sites,
        left_tensor,
        right_tensor,
        metric_blocks,
        *,
        variable,
    ):
        """Build factor-slice metrics from pair physical blocks."""
        if metric_blocks is None:
            return None
        site = int(site)
        union_sites = tuple(int(index) for index in union_sites)
        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        variable = str(variable).lower()
        if variable not in {"left", "right"}:
            raise ValueError("variable must be 'left' or 'right'.")
        merged_shape = (
            left_tensor.shape[0],
            right_tensor.shape[1],
            *(self.dims[index] for index in union_sites),
        )
        merged_layout = PhysicalBlockLayout(merged_shape)
        metric_blocks = tuple(self._hermitian_part(block) for block in metric_blocks)
        expected_pair_shape = (
            merged_layout.virtual_size,
            merged_layout.virtual_size,
        )
        if len(metric_blocks) != merged_layout.nblocks or any(
            block.shape != expected_pair_shape for block in metric_blocks
        ):
            raise ValueError("metric_blocks are incompatible with the pair.")
        factor = left_tensor if variable == "left" else right_tensor
        factor_layout = PhysicalBlockLayout(factor.shape)
        dtype = np.result_type(factor, *(block.dtype for block in metric_blocks))
        normals = [
            np.zeros(
                (factor_layout.virtual_size, factor_layout.virtual_size),
                dtype=dtype,
            )
            for _ in range(factor_layout.nblocks)
        ]
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[site + 1]
        union_shape = tuple(self.dims[index] for index in union_sites)
        for block, configuration in enumerate(np.ndindex(*union_shape)):
            values = dict(zip(union_sites, configuration))
            left_configuration = tuple(values[index] for index in left_sites)
            right_configuration = tuple(values[index] for index in right_sites)
            if variable == "left":
                factor_block = int(
                    np.ravel_multi_index(
                        left_configuration,
                        factor_layout.physical_shape,
                    )
                )
                right = right_tensor[
                    (slice(None), slice(None), *right_configuration)
                ]
                design = np.kron(
                    np.eye(left_tensor.shape[0], dtype=right.dtype),
                    right.T,
                )
            else:
                factor_block = int(
                    np.ravel_multi_index(
                        right_configuration,
                        factor_layout.physical_shape,
                    )
                )
                left = left_tensor[
                    (slice(None), slice(None), *left_configuration)
                ]
                design = np.kron(
                    left,
                    np.eye(right_tensor.shape[1], dtype=left.dtype),
                )
            normals[factor_block] += (
                design.T.conj() @ metric_blocks[block] @ design
            )
        return PhysicalBlockLinearOperator(
            factor_layout,
            {
                (block, block): self._hermitian_part(normal)
                for block, normal in enumerate(normals)
            },
            dtype=dtype,
        )

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
        left_sites = self.physical_sites[site]
        right_sites = self.physical_sites[following]
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

    def _complete_pair_metric_solution(
        self,
        old_vector,
        vector,
        *,
        metric=None,
        metric_operator=None,
        metric_action=None,
        metric_eigensystem=None,
    ):
        """Subclass hook for projector-null merged or factor coordinates."""
        del (
            old_vector,
            metric,
            metric_operator,
            metric_action,
            metric_eigensystem,
        )
        return None

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
        skip_redundant_full_rank_davidson=False,
        redundant_full_rank_davidson_min_dimension=0,
        metric_support="regularized",
        metric_eigensystem=None,
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
        if metric_eigensystem is None:
            metric_values, metric_vectors = (
                self._dense_pair_metric_eigensystem(metric)
            )
        else:
            metric_values, metric_vectors = metric_eigensystem
            metric_values = np.asarray(metric_values)
            metric_vectors = np.asarray(metric_vectors)
            if metric_values.shape != (size,) or metric_vectors.shape != (
                size,
                size,
            ):
                raise ValueError(
                    "merged-pair metric eigensystem has incompatible shapes."
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
        skip_davidson = bool(
            skip_redundant_full_rank_davidson
            and support_rank == size
            and size <= int(dense_fallback_dim)
            and size >= int(redundant_full_rank_davidson_min_dimension)
        )
        if not skip_davidson:
            try:
                attempts.append("warm_davidson")
                energy, vector, davidson_diagnostics = (
                    lowest_generalized_davidson(
                        lambda trial: effective @ trial,
                        lambda trial: metric @ trial,
                        warm,
                        tol=float(eig_tol),
                        metric_tol=(
                            64.0 * np.finfo(float).eps
                            if metric_support == "numerical"
                            else max(
                                float(metric_tol),
                                64.0 * np.finfo(float).eps,
                            )
                        ),
                        maxiter=davidson_maxiter,
                        max_subspace=davidson_max_subspace,
                        random_seed=int(site),
                    )
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
                    fallback_reason = (
                        "warm Davidson failed full metric-dual verification"
                    )
            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ) as error:
                fallback_reason = f"warm Davidson failed: {error}"
        else:
            attempts.append("warm_davidson_skipped_full_rank")
            fallback_reason = (
                "warm Davidson is redundant before full-rank dense "
                "lowest-root certification"
            )

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
            metric_rank_complete=support_rank == numerical_rank,
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
                else getattr(davidson_diagnostics, "metric_matvecs", 0)
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

    def _solve_verified_pair_actions(
        self,
        site,
        metric,
        hamiltonian_action,
        warm,
        *,
        hamiltonian_action_batch=None,
        initial_subspace=None,
        jacobi_blocks=(),
        block_size=4,
        recycle_dimension=6,
        recycle_output=None,
        metric_tol,
        eig_tol,
        maxiter,
        max_subspace,
        metric_support="regularized",
    ):
        """Solve a metric-whitened pair pencil from Hamiltonian actions."""
        metric_support = str(metric_support).lower().replace("-", "_")
        if metric_support not in {"regularized", "numerical"}:
            raise ValueError(
                "metric_support must be 'regularized' or 'numerical'."
            )
        metric = self._hermitian_part(metric)
        warm = np.asarray(warm).reshape(-1)
        size = warm.size
        if metric.shape != (size, size):
            raise ValueError("merged pair metric has an incompatible shape.")
        metric_scale = max(
            float(np.linalg.norm(metric, ord=np.inf)),
            np.finfo(float).tiny,
        )
        prepared_jacobi = []
        occupied = np.zeros(size, dtype=bool)
        for indices, hamiltonian_block in jacobi_blocks:
            indices = np.asarray(indices, dtype=np.intp).reshape(-1)
            if (
                indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= size)
                or np.unique(indices).size != indices.size
                or np.any(occupied[indices])
            ):
                raise ValueError(
                    "Jacobi blocks must contain disjoint valid pair indices."
                )
            hamiltonian_block = self._hermitian_part(
                np.asarray(hamiltonian_block)
            )
            if hamiltonian_block.shape != (indices.size, indices.size):
                raise ValueError(
                    "a Jacobi block has an incompatible shape."
                )
            occupied[indices] = True
            prepared_jacobi.append((indices, hamiltonian_block))
        structured_metric = bool(prepared_jacobi and np.all(occupied))
        if structured_metric:
            group_ids = np.empty(size, dtype=np.intp)
            for group, (indices, _hamiltonian) in enumerate(
                prepared_jacobi
            ):
                group_ids[indices] = group
            off_block = group_ids[:, None] != group_ids[None, :]
            structured_metric = bool(
                np.linalg.norm(metric[off_block])
                <= 1.0e-11 * metric_scale
            )
        metric_groups = []
        if structured_metric:
            metric_values_parts = []
            metric_vectors = np.zeros(
                (size, size),
                dtype=np.result_type(metric, np.float64),
            )
            offset = 0
            for indices, _hamiltonian in prepared_jacobi:
                values, vectors = linalg.eigh(
                    metric[np.ix_(indices, indices)],
                    check_finite=False,
                )
                columns = np.arange(
                    offset,
                    offset + indices.size,
                    dtype=np.intp,
                )
                metric_values_parts.append(values)
                metric_vectors[np.ix_(indices, columns)] = vectors
                metric_groups.append((indices, columns))
                offset += indices.size
            metric_values = np.concatenate(metric_values_parts)
        else:
            metric_values, metric_vectors = linalg.eigh(
                metric,
                check_finite=False,
            )
            metric_groups = [
                (
                    np.arange(size, dtype=np.intp),
                    np.arange(size, dtype=np.intp),
                )
            ]
        metric_scale = max(
            metric_scale,
            float(np.max(np.abs(metric_values), initial=0.0)),
        )
        numerical_floor = 64.0 * np.finfo(float).eps * metric_scale
        numerical_active = metric_values > numerical_floor
        numerical_rank = int(np.count_nonzero(numerical_active))
        if numerical_rank == 0:
            raise ValueError("merged-pair overlap metric is numerically singular.")
        requested_floor = max(float(metric_tol) * metric_scale, numerical_floor)
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
        basis = metric_vectors[:, active] / np.sqrt(positive_values)[None, :]
        identity_error = float(
            np.linalg.norm(
                basis.T.conj() @ metric @ basis - np.eye(support_rank),
                ord=np.inf,
            )
        )
        jacobi_coordinate_blocks = []
        if structured_metric:
            full_to_active = np.full(size, -1, dtype=np.intp)
            full_to_active[np.flatnonzero(active)] = np.arange(
                support_rank,
                dtype=np.intp,
            )
            for (
                (indices, columns),
                (_same_indices, hamiltonian_block),
            ) in zip(metric_groups, prepared_jacobi):
                coordinates = full_to_active[columns]
                coordinates = coordinates[coordinates >= 0]
                if coordinates.size == 0:
                    continue
                local_basis = basis[np.ix_(indices, coordinates)]
                transformed = self._hermitian_part(
                    local_basis.T.conj()
                    @ hamiltonian_block
                    @ local_basis
                )
                jacobi_coordinate_blocks.append(
                    (coordinates, transformed)
                )

        action_count = 0
        action_batches = 0

        def apply_h(vector):
            nonlocal action_count, action_batches
            action_count += 1
            action_batches += 1
            value = np.asarray(hamiltonian_action(vector)).reshape(-1)
            if value.shape != (size,) or np.any(~np.isfinite(value)):
                raise ValueError(
                    "hamiltonian_action returned an invalid pair vector."
                )
            return value

        def apply_h_batch(vectors):
            nonlocal action_count, action_batches
            vectors = np.asarray(vectors)
            if vectors.ndim != 2 or vectors.shape[0] != size:
                raise ValueError(
                    "Hamiltonian batch input has an invalid pair shape."
                )
            if hamiltonian_action_batch is None:
                return np.column_stack(
                    [apply_h(vectors[:, column]) for column in range(vectors.shape[1])]
                )
            action_count += vectors.shape[1]
            action_batches += 1
            values = np.asarray(hamiltonian_action_batch(vectors))
            if values.shape != vectors.shape or np.any(~np.isfinite(values)):
                raise ValueError(
                    "hamiltonian_action_batch returned an invalid pair batch."
                )
            return values

        def action_check(energy, vector, hamiltonian_vector=None):
            vector = np.asarray(vector).reshape(-1)
            metric_vector = metric @ vector
            if hamiltonian_vector is None:
                hamiltonian_vector = apply_h(vector)
            else:
                hamiltonian_vector = np.asarray(
                    hamiltonian_vector
                ).reshape(-1)
            residual = hamiltonian_vector - float(energy) * metric_vector
            residual_norm = float(np.linalg.norm(residual))
            scale = max(
                float(np.linalg.norm(hamiltonian_vector)),
                abs(float(energy)) * float(np.linalg.norm(metric_vector)),
                np.finfo(float).tiny,
            )
            numerical_null = ~numerical_active
            null_residual = (
                float(
                    np.linalg.norm(
                        metric_vectors[:, numerical_null].T.conj() @ residual
                    )
                    / scale
                )
                if np.any(numerical_null)
                else 0.0
            )
            discarded = numerical_active & ~active
            discarded_residual = (
                float(
                    np.linalg.norm(
                        metric_vectors[:, discarded].T.conj() @ residual
                    )
                    / scale
                )
                if np.any(discarded)
                else 0.0
            )
            return {
                "raw": residual_norm,
                "relative": residual_norm / scale,
                "null": null_residual,
                "discarded": discarded_residual,
            }

        warm_norm = float(np.real(np.vdot(warm, metric @ warm)))
        if warm_norm <= np.finfo(float).tiny:
            raise ValueError("warm merged-pair state has zero metric norm.")
        warm = warm / np.sqrt(warm_norm)
        warm_hamiltonian = apply_h(warm)
        warm_energy = float(
            np.real(np.vdot(warm, warm_hamiltonian))
            / np.real(np.vdot(warm, metric @ warm))
        )
        warm_check = action_check(
            warm_energy,
            warm,
            hamiltonian_vector=warm_hamiltonian,
        )
        verification_tolerance = max(
            10.0 * float(eig_tol),
            8.0 * np.sqrt(np.finfo(float).eps),
        )
        energy_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(warm_energy),
        )

        warm_coordinates = (
            np.sqrt(positive_values)
            * (metric_vectors[:, active].T.conj() @ warm)
        )
        coordinate_norm = float(np.linalg.norm(warm_coordinates))
        if coordinate_norm <= np.finfo(float).tiny:
            raise ValueError("warm pair has no weight in the requested metric support.")
        warm_coordinates = warm_coordinates / coordinate_norm

        def whitened_hamiltonian(coordinates):
            return basis.T.conj() @ apply_h(basis @ coordinates)

        def whitened_hamiltonian_batch(coordinates):
            return basis.T.conj() @ apply_h_batch(basis @ coordinates)

        coordinate_starts = [warm_coordinates]
        if initial_subspace is not None:
            starts = np.asarray(initial_subspace)
            if starts.ndim == 1:
                starts = starts[:, None]
            if starts.ndim != 2 or starts.shape[0] != size:
                raise ValueError(
                    "initial_subspace has an incompatible pair dimension."
                )
            for column in range(starts.shape[1]):
                coordinates = (
                    np.sqrt(positive_values)
                    * (
                        metric_vectors[:, active].T.conj()
                        @ starts[:, column]
                    )
                )
                if (
                    np.all(np.isfinite(coordinates))
                    and np.linalg.norm(coordinates)
                    > 128.0 * np.finfo(float).eps
                ):
                    coordinate_starts.append(coordinates)
        coordinate_starts = np.column_stack(coordinate_starts)

        davidson_diagnostics = None
        recycled_coordinates = None
        fallback_reason = ""
        use_block_solver = bool(
            hamiltonian_action_batch is not None
            or initial_subspace is not None
            or jacobi_coordinate_blocks
        )
        solver_attempt = (
            "recycled_block_action_davidson"
            if use_block_solver
            else "whitened_action_davidson"
        )
        try:
            if use_block_solver:
                (
                    energy,
                    coordinates,
                    recycled_coordinates,
                    davidson_diagnostics,
                ) = lowest_recycled_block_davidson(
                    whitened_hamiltonian,
                    coordinate_starts,
                    hamiltonian_batch_action=(
                        whitened_hamiltonian_batch
                        if hamiltonian_action_batch is not None
                        else None
                    ),
                    preconditioner_blocks=jacobi_coordinate_blocks,
                    block_size=min(int(block_size), support_rank),
                    recycle_dimension=min(
                        max(1, int(recycle_dimension)),
                        support_rank,
                    ),
                    tol=float(eig_tol),
                    maxiter=min(
                        128,
                        int(maxiter) if maxiter is not None else 128,
                    ),
                    max_subspace=min(
                        support_rank,
                        int(max_subspace),
                        64,
                    ),
                    random_seed=int(site),
                )
            else:
                energy, coordinates, davidson_diagnostics = (
                    lowest_generalized_davidson(
                        whitened_hamiltonian,
                        lambda vector: np.asarray(vector),
                        warm_coordinates,
                        tol=float(eig_tol),
                        metric_tol=64.0 * np.finfo(float).eps,
                        maxiter=min(
                            128,
                            int(maxiter) if maxiter is not None else 128,
                        ),
                        max_subspace=min(
                            support_rank,
                            int(max_subspace),
                            64,
                        ),
                        random_seed=int(site),
                    )
                )
                recycled_coordinates = coordinates[:, None]
            vector = basis @ coordinates
            vector = vector / np.sqrt(
                float(np.real(np.vdot(vector, metric @ vector)))
            )
            check = action_check(energy, vector)
            candidate_verified = bool(
                davidson_diagnostics.converged
                and check["relative"] <= verification_tolerance
                and check["null"] <= verification_tolerance
                and energy <= warm_energy + energy_tolerance
            )
            if not candidate_verified:
                fallback_reason = (
                    f"{solver_attempt} failed fresh action-residual verification"
                )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            energy = warm_energy
            vector = warm
            check = warm_check
            candidate_verified = False
            fallback_reason = f"{solver_attempt} failed: {error}"

        if recycle_output is not None:
            recycle_output.clear()
            if candidate_verified and recycled_coordinates is not None:
                recycle_output["vectors"] = np.asarray(
                    basis @ recycled_coordinates
                )
            else:
                recycle_output["vectors"] = warm[:, None].copy()

        if candidate_verified:
            selected_energy = float(energy)
            selected_vector = np.asarray(vector).reshape(-1)
            selected_check = check
            selected_method = solver_attempt
        else:
            selected_energy = warm_energy
            selected_vector = warm
            selected_check = warm_check
            selected_method = "warm"
        verified = bool(
            selected_check["relative"] <= verification_tolerance
            and selected_check["null"] <= verification_tolerance
        )
        accepted = bool(selected_energy <= warm_energy + energy_tolerance)
        diagnostics = FrontierMergedSolveDiagnostics(
            method=selected_method,
            attempts=(solver_attempt,),
            verified=verified,
            lowest_root_certified=False,
            fallback_reason=fallback_reason,
            dense_fallback=False,
            metric_requested_rank=requested_rank,
            metric_numerical_rank=numerical_rank,
            metric_min_positive=minimum_positive,
            metric_condition=condition,
            backward_residual=selected_check["relative"],
            metric_dual_residual=float("nan"),
            metric_dual_relative_residual=float("nan"),
            null_residual=selected_check["null"],
            warm_energy=warm_energy,
            upper_bound_gap=float(selected_energy - warm_energy),
            metric_support=metric_support,
            discarded_support_residual=selected_check["discarded"],
            action_relative_residual=selected_check["relative"],
            verification_kind="action_residual",
            metric_rank_complete=support_rank == numerical_rank,
            hamiltonian_action_calls=action_batches,
            hamiltonian_vector_products=action_count,
            hamiltonian_batch_calls=(
                0
                if davidson_diagnostics is None
                else getattr(
                    davidson_diagnostics,
                    "batch_action_calls",
                    0,
                )
            ),
            recycled_vectors=(
                0
                if recycled_coordinates is None
                else recycled_coordinates.shape[1]
            ),
            preconditioner_blocks=len(jacobi_coordinate_blocks),
        )
        local_update = FrontierSiteUpdate(
            site=int(site),
            raw_dim=size,
            metric_rank=support_rank,
            metric_rank_is_projected=support_rank < numerical_rank,
            solver=selected_method,
            solver_converged=verified,
            message=(
                "verified metric-whitened Hamiltonian-action root"
                if selected_method != "warm" and verified
                else (
                    "retained residual-verified warm pair"
                    if verified
                    else "matrix-free pair root failed action verification"
                )
            ),
            hamiltonian_matvecs=action_count,
            metric_matvecs=(
                0
                if davidson_diagnostics is None
                else getattr(davidson_diagnostics, "metric_matvecs", 0)
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
            solver_metric_is_identity=True,
            solver_metric_identity_error=identity_error,
            solver_coordinate_residual_norm=selected_check["relative"],
            hamiltonian_action_calls=action_batches,
            hamiltonian_batch_calls=(
                0
                if davidson_diagnostics is None
                else getattr(
                    davidson_diagnostics,
                    "batch_action_calls",
                    0,
                )
            ),
            recycled_vectors=(
                0
                if recycled_coordinates is None
                else recycled_coordinates.shape[1]
            ),
            preconditioner_blocks=len(jacobi_coordinate_blocks),
        )
        return float(selected_energy), selected_vector, local_update, diagnostics

    @staticmethod
    def _block_operator_inf_norm(operator):
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
    ):
        vector = np.asarray(vector).reshape(-1)
        metric_vector = problem.metric.matvec(vector)
        hamiltonian_vector = problem.hamiltonian.matvec(vector)
        residual = hamiltonian_vector - float(energy) * metric_vector
        vector_norm = max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
        metric_inf = self._block_operator_inf_norm(problem.metric)
        hamiltonian_inf = self._block_operator_inf_norm(problem.hamiltonian)
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
        for block in range(problem.layout.nblocks):
            indices = problem.layout.block_indices[block]
            metric_block = problem.metric.blocks[(block, block)]
            values, vectors = linalg.eigh(metric_block, check_finite=False)
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
        block_dense_component_max_size=0,
        davidson_initial=None,
        verify_root=True,
        parallel_components=False,
        max_component_workers=None,
        metric_support="regularized",
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
        metric_values = tuple(
            linalg.eigvalsh(block, check_finite=False)
            for block in problem.metric.blocks.values()
        )
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
        solve_initial = warm
        if davidson_initial is not None:
            candidate_initial = np.asarray(davidson_initial).reshape(-1)
            if (
                candidate_initial.size == size
                and np.all(np.isfinite(candidate_initial))
            ):
                candidate_metric = problem.metric.matvec(candidate_initial)
                candidate_norm = float(
                    np.real(np.vdot(candidate_initial, candidate_metric))
                )
                if candidate_norm > np.finfo(float).tiny:
                    solve_initial = candidate_initial / np.sqrt(candidate_norm)
        verification_tolerance = max(
            10.0 * float(eig_tol),
            8.0 * np.sqrt(np.finfo(float).eps),
        )
        energy_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(warm_energy),
        )
        block_failure_reason = ""
        try:
            energy, vector, block_diagnostics = problem.solve(
                solve_initial,
                tol=float(eig_tol),
                metric_tol=float(solve_floor / metric_scale),
                maxiter=min(128, int(maxiter) if maxiter is not None else 128),
                max_subspace=min(size, int(max_subspace), 64),
                random_seed=int(site),
                dense_component_max_size=int(block_dense_component_max_size),
                parallel_components=bool(parallel_components),
                max_component_workers=max_component_workers,
            )
            if verify_root:
                check = self._pair_block_residual_verification(
                    problem,
                    energy,
                    vector,
                    numerical_floor=numerical_floor,
                    solve_floor=solve_floor,
                )
                verified = bool(
                    check["dual_relative"] <= verification_tolerance
                    and check["null"] <= verification_tolerance
                    and energy <= warm_energy + energy_tolerance
                )
            else:
                check = {
                    "raw": block_diagnostics.residual_norm,
                    "backward": float("nan"),
                    "dual": float("nan"),
                    "dual_relative": block_diagnostics.residual_norm,
                    "null": 0.0,
                    "discarded": 0.0,
                }
                verified = bool(
                    block_diagnostics.converged
                    and energy <= warm_energy + energy_tolerance
                )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            block_failure_reason = f"conditional block solve failed: {error}"
            if size <= int(dense_fallback_dim):
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
            )
            verified = False
            block_diagnostics = None

        if not verified and size <= int(dense_fallback_dim):
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
            metric_rank_complete=support_rank == numerical_rank,
        )
        local_update = FrontierSiteUpdate(
            site=int(site),
            raw_dim=size,
            metric_rank=support_rank,
            metric_rank_is_projected=support_rank < numerical_rank,
            solver=method,
            solver_converged=verified,
            message=(
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
                len(problem.hamiltonian.blocks)
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
        left_tensor = np.asarray(left_tensor).copy()
        right_tensor = np.asarray(right_tensor).copy()
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
        left_sites = self.physical_sites[int(site)]
        right_sites = self.physical_sites[int(site) + 1]
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
        metric_blocks=None,
    ):
        """Alternately minimize the exact pair Rayleigh quotient."""
        left_tensor = np.asarray(left_tensor).copy()
        right_tensor = np.asarray(right_tensor).copy()
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
                local_metric = None

                def forward(vector):
                    return self._pair_factor_action(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable=variable,
                    )

                def adjoint(vector):
                    return self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable=variable,
                    )

                def local_metric_action(vector):
                    return adjoint(metric_action(forward(vector)))

                def local_hamiltonian_action(vector):
                    return adjoint(effective_action(forward(vector)))

                local_metric_operator = (
                    self._pair_factor_metric_operator(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        metric_blocks,
                        variable=variable,
                    )
                    if self._preserve_pair_metric_null_components
                    else None
                )

                def projected_local_matrices():
                    identity = np.eye(old_factor.size, dtype=old_factor.dtype)
                    local_metric = np.column_stack(
                        [
                            local_metric_action(identity[:, column])
                            for column in range(old_factor.size)
                        ]
                    )
                    local_effective = np.column_stack(
                        [
                            local_hamiltonian_action(identity[:, column])
                            for column in range(old_factor.size)
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
                            old_factor.reshape(-1),
                            tol=eig_tol,
                            metric_tol=metric_tol,
                            maxiter=maxiter,
                            max_subspace=min(
                                int(max_subspace),
                                old_factor.size,
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
                completed = self._complete_pair_metric_solution(
                    old_factor.reshape(-1),
                    vector,
                    metric=local_metric,
                    metric_operator=local_metric_operator,
                    metric_action=local_metric_action,
                )
                if completed is not None:
                    vector = completed
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
        merged = self._merge_pair_factors(
            site,
            union_sites,
            left_tensor,
            right_tensor,
        ).reshape(-1)
        residual = np.asarray(target).reshape(-1) - merged
        split = left_tensor.size
        tangent_size = split + right_tensor.size

        def tangent_forward(vector):
            vector = np.asarray(vector).reshape(-1)
            return self._pair_factor_action(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                vector[:split],
                variable="left",
            ) + self._pair_factor_action(
                site,
                union_sites,
                left_tensor,
                right_tensor,
                vector[split:],
                variable="right",
            )

        def tangent_adjoint(vector):
            return np.concatenate(
                (
                    self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable="left",
                    ),
                    self._pair_factor_adjoint(
                        site,
                        union_sites,
                        left_tensor,
                        right_tensor,
                        vector,
                        variable="right",
                    ),
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
        delta_left = tangent[:split].reshape(left_tensor.shape)
        delta_right = tangent[split:].reshape(right_tensor.shape)

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
            metric_blocks=metric_blocks,
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
                    metric_blocks=metric_blocks,
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
                    metric_blocks=metric_blocks,
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
        if type(self) is not FrontierTiedLETTA or self._uses_custom_operator_mpos:
            raise TypeError(
                "two-site merge/split currently supports exact unrestricted "
                "FrontierTiedLETTA states with the standard objective."
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
            local_backend=self.local_backend,
            local_rank=self.local_options["rank"],
            local_rtol=self.local_options["rtol"],
            local_atol=self.local_options["atol"],
            max_rank=self.tt_options["max_rank"],
            rtol=self.tt_options["rtol"],
            atol=self.tt_options["atol"],
            transfer_max_rank=self.tt_options["transfer_max_rank"],
            transfer_rtol=self.tt_options["transfer_rtol"],
            transfer_atol=self.tt_options["transfer_atol"],
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
        pair_dense_max_elements=2_000_000,
        pair_operator_workers=1,
        merged_dense_fallback_dim=2048,
        metric_support="regularized",
        outer_cycles=8,
        factor_solver="auto",
        split_strategy="variational",
        split_metric_tol=1.0e-12,
        split_metric_sweeps=6,
        split_variational_sweeps=8,
        split_random_starts=0,
        split_random_seed=0,
        split_energy_tol=1.0e-12,
        temporary_bond_dimension=None,
        temporary_bond_strategy="zero",
        temporary_bond_scale=0.0,
        **solver_options,
    ):
        r"""Run a verified, cached adjacent-pair update.

        The merged pencil is warm-started, checked with a metric-dual
        residual, and densely certified when its selected metric support is
        manageable.  ``metric_support="regularized"`` respects ``metric_tol``;
        ``"numerical"`` is an explicit full-positive-support diagnostic.
        Retraction uses conditional norm blocks.  ``factor_solver="auto"``
        reuses a resident dense local pencil when one was selected, otherwise
        it applies the factor equations through direct ``J``/``J^dagger``
        actions.  Up to ``outer_cycles`` merge--retract--relax corrections
        reuse the same exact pair environment and operators.
        ``temporary_bond_dimension`` can grow the middle cut before splitting;
        ``"square"``/``"dmrg"`` uses the old dimension squared.
        """
        site = int(site)
        if site < 0 or site + 1 >= len(self.dims):
            raise ValueError("site must be the left member of an adjacent pair.")
        solver = str(solver).lower().replace("-", "_")
        if solver not in {
            "verified",
            "whitened",
            "direct",
            "matrix_free",
            "block_sparse",
        }:
            raise ValueError(
                "solver must be 'verified', 'whitened', 'direct', "
                "'matrix_free', or 'block_sparse'."
            )
        split_strategy = str(split_strategy).lower().replace("-", "_")
        if split_strategy in {"environment", "metric", "als"}:
            split_strategy = "variational"
        if split_strategy not in {"variational", "svd"}:
            raise ValueError("split_strategy must be 'variational' or 'svd'.")
        pair_operator_backend = str(pair_operator_backend).lower().replace(
            "-", "_"
        )
        if pair_operator_backend not in {"auto", "dense", "block"}:
            raise ValueError(
                "pair_operator_backend must be 'auto', 'dense', or 'block'."
            )
        factor_solver = str(factor_solver).lower().replace("-", "_")
        if factor_solver not in {"auto", "matrix_free", "dense"}:
            raise ValueError(
                "factor_solver must be 'auto', 'matrix_free', or 'dense'."
            )
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
        pair_operator_workers = int(pair_operator_workers)
        if pair_operator_workers < 1:
            raise ValueError("pair_operator_workers must be positive.")
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
        temporary_bond_strategy = str(temporary_bond_strategy).lower().replace(
            "-", "_"
        )
        if temporary_bond_strategy not in {"zero", "residual", "random"}:
            raise ValueError(
                "temporary_bond_strategy must be 'zero', 'residual', or 'random'."
            )
        temporary_bond_scale = float(temporary_bond_scale)
        if not np.isfinite(temporary_bond_scale) or temporary_bond_scale < 0.0:
            raise ValueError(
                "temporary_bond_scale must be finite and nonnegative."
            )

        options = dict(solver_options)
        block_dense_component_max_size = int(
            options.pop("block_dense_component_max_size", 64)
        )
        if block_dense_component_max_size < 0:
            raise ValueError("block_dense_component_max_size must be nonnegative.")
        metric_tol = float(options.pop("metric_tol", 1.0e-12))
        eig_tol = float(options.pop("eig_tol", 1.0e-10))
        maxiter = options.pop("maxiter", 1600)
        max_subspace = int(options.pop("max_subspace", 96))
        skip_redundant_full_rank_davidson = bool(
            options.pop(
                "skip_redundant_full_rank_davidson",
                False,
            )
        )
        redundant_full_rank_davidson_min_dimension = int(
            options.pop(
                "redundant_full_rank_davidson_min_dimension",
                0,
            )
        )
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
        if maxiter is not None:
            maxiter = int(maxiter)
            if maxiter < 1:
                raise ValueError("maxiter must be positive or None.")
        if max_subspace < 2:
            raise ValueError("max_subspace must be at least two.")
        if redundant_full_rank_davidson_min_dimension < 0:
            raise ValueError(
                "redundant_full_rank_davidson_min_dimension must be "
                "nonnegative."
            )

        following = site + 1
        old_middle_dimension = self._bond_dims()[following]
        if temporary_bond_dimension is None:
            temporary_middle_dimension = old_middle_dimension
        elif isinstance(temporary_bond_dimension, str):
            target = temporary_bond_dimension.lower().replace("-", "_")
            target = target.replace(" ", "")
            if target in {"fixed", "current", "none", "old"}:
                temporary_middle_dimension = old_middle_dimension
            elif target in {"square", "squared", "d2", "d^2", "dmrg"}:
                temporary_middle_dimension = old_middle_dimension * old_middle_dimension
            else:
                try:
                    temporary_middle_dimension = int(target)
                except ValueError as exc:
                    raise ValueError(
                        "temporary_bond_dimension must be None, an integer, "
                        "'fixed', or 'square'/'dmrg'."
                    ) from exc
        else:
            temporary_middle_dimension = int(temporary_bond_dimension)
        if temporary_middle_dimension < old_middle_dimension:
            raise ValueError(
                "temporary_bond_dimension cannot shrink the current bond."
            )
        if temporary_middle_dimension > old_middle_dimension:
            self.expand_bond(
                following,
                temporary_middle_dimension,
                direction="right",
                strategy=temporary_bond_strategy,
                scale=temporary_bond_scale,
            )
            environment = None
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

        requested_backend = pair_operator_backend
        selected_backend = requested_backend
        dense_pair_elements = 2 * merged_size**2
        selection_reason = "explicit request"
        if selected_backend == "auto":
            if solver in {"block_sparse", "matrix_free"}:
                selected_backend = "block"
                selection_reason = f"solver={solver} requests operator actions"
            elif dense_pair_elements > pair_dense_max_elements:
                selected_backend = "block"
                selection_reason = (
                    f"{dense_pair_elements} dense elements exceed the "
                    f"{pair_dense_max_elements} element budget"
                )
            else:
                selected_backend = "dense"
                selection_reason = (
                    f"{dense_pair_elements} dense elements fit within the "
                    f"{pair_dense_max_elements} element budget"
                )
        selected_factor_solver = factor_solver
        if selected_factor_solver == "auto":
            selected_factor_solver = (
                "dense" if selected_backend == "dense" else "matrix_free"
            )
        selected_dense_fallback_dim = merged_dense_fallback_dim
        if requested_backend == "auto" and selected_backend == "block":
            # Automatic memory selection must remain authoritative: a failed
            # block solve cannot silently rematerialize the dense pair pencil.
            selected_dense_fallback_dim = 1
        operator_assembly_start = perf_counter()
        block_problem = None
        pair_metric = None
        pair_effective = None
        metric_blocks = None
        if selected_backend == "dense":
            pair_metric, pair_effective = self.pair_local_operators(
                site,
                environment=environment,
                hamiltonian_workers=pair_operator_workers,
            )
            pair_metric_eigensystem = (
                self._dense_pair_metric_eigensystem(pair_metric)
            )
            pair_layout = PhysicalBlockLayout(plan.merged_shape)
            metric_blocks = tuple(
                pair_metric[np.ix_(indices, indices)]
                for indices in pair_layout.block_indices
            )
            stored_elements = int(pair_metric.size + pair_effective.size)

            def metric_action(vector):
                return pair_metric @ vector

            def effective_action(vector):
                return pair_effective @ vector

        elif selected_backend == "block":
            pair_metric_eigensystem = None
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
        operator_assembly_seconds = float(
            perf_counter() - operator_assembly_start
        )

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
                set(self.physical_sites[site])
                & set(self.physical_sites[following])
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
        merged_solve_seconds = 0.0
        split_seconds = 0.0
        for cycle in range(maximum_cycles):
            completed_cycles = cycle + 1
            reused_certified_root = certified_root is not None
            merged_solve_start = perf_counter()
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
                    dense_fallback_dim=selected_dense_fallback_dim,
                    skip_redundant_full_rank_davidson=(
                        skip_redundant_full_rank_davidson
                    ),
                    redundant_full_rank_davidson_min_dimension=(
                        redundant_full_rank_davidson_min_dimension
                    ),
                    metric_support=metric_support,
                    metric_eigensystem=pair_metric_eigensystem,
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
                    dense_fallback_dim=selected_dense_fallback_dim,
                    block_dense_component_max_size=block_dense_component_max_size,
                    metric_support=metric_support,
                )
            merged_solve_seconds += perf_counter() - merged_solve_start
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
            split_start = perf_counter()
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
            completed_split_target = self._complete_pair_metric_solution(
                warm_vector,
                split_target.reshape(-1),
                metric=pair_metric,
                metric_operator=(
                    None if block_problem is None else block_problem.metric
                ),
                metric_action=metric_action,
                metric_eigensystem=pair_metric_eigensystem,
            )
            if completed_split_target is not None:
                split_target = completed_split_target.reshape(plan.merged_shape)

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
            if split_strategy == "svd":
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
            else:
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
                    factor_solver=selected_factor_solver,
                    factor_eig_tol=eig_tol,
                    factor_maxiter=min(maxiter or 256, 256),
                    factor_max_subspace=min(max_subspace, 32),
                    metric_action=metric_action,
                    effective_action=effective_action,
                    metric_blocks=metric_blocks,
                )
                candidate_merged = self._merge_pair_factors(
                    site,
                    plan.union_sites,
                    candidate_left,
                    candidate_right,
                ).reshape(-1)
                completed_candidate = self._complete_pair_metric_solution(
                    warm.reshape(-1),
                    candidate_merged,
                    metric=pair_metric,
                    metric_operator=(
                        None if block_problem is None else block_problem.metric
                    ),
                    metric_action=metric_action,
                    metric_eigensystem=pair_metric_eigensystem,
                )
                if completed_candidate is not None:
                    (
                        restored_left,
                        restored_right,
                        _restored_overlap,
                        _restored_ranks,
                        restored_error,
                    ) = self._split_merged_pair_tensor(
                        site,
                        completed_candidate.reshape(plan.merged_shape),
                        plan.union_sites,
                    )
                    representability_tolerance = max(
                        2048.0 * np.finfo(float).eps,
                        split_metric_tol,
                    )
                    if restored_error <= representability_tolerance:
                        restored_energy = self._pair_rayleigh_actions(
                            self._merge_pair_factors(
                                site,
                                plan.union_sites,
                                restored_left,
                                restored_right,
                            ).reshape(-1),
                            metric_action,
                            effective_action,
                        )
                        if (
                            np.isfinite(restored_energy)
                            and restored_energy
                            <= candidate_energy
                            + split_energy_tol
                            * max(1.0, abs(candidate_energy))
                        ):
                            candidate_left = restored_left
                            candidate_right = restored_right
                            candidate_energy = float(restored_energy)
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
            tolerance = split_energy_tol * max(1.0, abs(incumbent_energy))
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
            split_seconds += perf_counter() - split_start
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
            old_bond_dimension=old_middle_dimension,
            temporary_bond_dimension=self._bond_dims()[following],
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
            pair_operator_stored_bytes=int(
                stored_elements
                * np.dtype(
                    np.result_type(
                        self.hamiltonian.dtype,
                        self.tensors[site].dtype,
                        self.tensors[following].dtype,
                    )
                ).itemsize
            ),
            operator_assembly_seconds=operator_assembly_seconds,
            merged_solve_seconds=float(merged_solve_seconds),
            split_seconds=float(split_seconds),
            pair_operator_requested_backend=requested_backend,
            pair_operator_selection_reason=selection_reason,
            dense_estimated_peak_bytes=int(
                dense_pair_elements
                * np.dtype(
                    np.result_type(
                        self.hamiltonian.dtype,
                        self.tensors[site].dtype,
                        self.tensors[following].dtype,
                    )
                ).itemsize
            ),
            factor_solver=selected_factor_solver,
            pair_operator_workers=pair_operator_workers,
        )

    def run_two_site(
        self,
        *,
        nsweeps: int = 2,
        sweep_offset: int = 0,
        tol: float = 1.0e-10,
        solver="verified",
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
        once per directional sweep.
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
        if "environment" in pair_options or "verify_global" in pair_options:
            raise TypeError(
                "run_two_site constructs pair environments and controls "
                "endpoint verification."
            )
        if (
            "temporary_bond_dim" in pair_options
            and "temporary_bond_dimension" not in pair_options
        ):
            pair_options = dict(pair_options)
            pair_options["temporary_bond_dimension"] = pair_options.pop(
                "temporary_bond_dim"
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
        temporary_bond_dimension = pair_options.get(
            "temporary_bond_dimension",
            pair_options.get("temporary_bond_dim", None),
        )
        if isinstance(temporary_bond_dimension, str):
            fixed_bond_sweep = temporary_bond_dimension.lower().replace(
                "-", "_"
            ) in {"fixed", "current", "none", "old"}
        else:
            fixed_bond_sweep = temporary_bond_dimension is None

        previous = float(self.expectation())
        self.energy = previous
        sweep_history = []
        self.history = sweep_history
        self.converged = False
        nsites = len(self.dims)
        for sweep in range(nsweeps):
            directional_sweep = sweep_offset + sweep
            sweep_tensors = [tensor.copy() for tensor in self.tensors]
            updates = []
            if not fixed_bond_sweep:
                sweep_state = self.copy()
                sites = (
                    range(nsites - 1)
                    if directional_sweep % 2 == 0
                    else range(nsites - 2, -1, -1)
                )
                for site in sites:
                    pair_start = perf_counter()
                    update = self.optimize_two_sites(
                        site,
                        solver=solver,
                        environment=None,
                        verify_global=verify_pair_energies,
                        **pair_options,
                    )
                    updates.append(
                        replace(
                            update,
                            wall_time_seconds=float(perf_counter() - pair_start),
                        )
                    )
                attempted_energy = float(self.expectation())
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
                    self.balance_gauges()
                else:
                    self.tensors = [tensor.copy() for tensor in sweep_state.tensors]
                    self._virtual_bond_dims = tuple(sweep_state.bond_dims)
                    self.bond_dim = sweep_state.bond_dim
                    self._rebuild_frontier_engines()
                    energy = previous
                self.energy = energy
                delta = abs(energy - previous)
                sweep_history.append(
                    {
                        "sweep": directional_sweep,
                        "energy": energy,
                        "attempted_energy": attempted_energy,
                        "delta": delta,
                        "accepted": endpoint_accepted,
                        "direction": (
                            "left_to_right"
                            if directional_sweep % 2 == 0
                            else "right_to_left"
                        ),
                        "updates": tuple(updates),
                        "accepted_updates": sum(update.accepted for update in updates),
                        "dense_fallbacks": sum(
                            bool(
                                update.merged_solve is not None
                                and update.merged_solve.dense_fallback
                            )
                            for update in updates
                        ),
                        "pair_wall_time_seconds": float(
                            sum(update.wall_time_seconds for update in updates)
                        ),
                        "slowest_pair_wall_time_seconds": float(
                            max(
                                (update.wall_time_seconds for update in updates),
                                default=0.0,
                            )
                        ),
                        "operator_assembly_seconds": float(
                            sum(update.operator_assembly_seconds for update in updates)
                        ),
                        "merged_solve_seconds": float(
                            sum(update.merged_solve_seconds for update in updates)
                        ),
                        "split_seconds": float(
                            sum(update.split_seconds for update in updates)
                        ),
                    }
                )
                if verbose:
                    print(
                        f"two-site sweep {directional_sweep:3d}  "
                        f"energy={energy:.14f}  delta={delta:.3e}  "
                        f"accepted={endpoint_accepted}"
                    )
                if endpoint_accepted and delta < tol:
                    self.converged = True
                    break
                previous = energy
                continue
            if directional_sweep % 2 == 0:
                norm_right = self._norm_frontier.build_right(self.tensors)
                hamiltonian_right = self._hamiltonian_frontier.build_right(
                    self.tensors
                )
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                for site in range(nsites - 1):
                    environment = self._pair_environment_from_outer_messages(
                        site,
                        moving_norm,
                        norm_right[site + 2],
                        moving_hamiltonian,
                        hamiltonian_right[site + 2],
                    )
                    pair_start = perf_counter()
                    update = self.optimize_two_sites(
                        site,
                        solver=solver,
                        environment=environment,
                        verify_global=verify_pair_energies,
                        **pair_options,
                        )
                    updates.append(
                        replace(
                            update,
                            wall_time_seconds=float(perf_counter() - pair_start),
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
                    environment = self._pair_environment_from_outer_messages(
                        site,
                        norm_left[site],
                        moving_norm,
                        hamiltonian_left[site],
                        moving_hamiltonian,
                    )
                    pair_start = perf_counter()
                    update = self.optimize_two_sites(
                        site,
                        solver=solver,
                        environment=environment,
                        verify_global=verify_pair_energies,
                        **pair_options,
                        )
                    updates.append(
                        replace(
                            update,
                            wall_time_seconds=float(perf_counter() - pair_start),
                        )
                    )
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
            sweep_history.append(
                {
                    "sweep": directional_sweep,
                    "energy": energy,
                    "attempted_energy": attempted_energy,
                    "delta": delta,
                    "accepted": endpoint_accepted,
                    "direction": (
                        "left_to_right"
                        if directional_sweep % 2 == 0
                        else "right_to_left"
                    ),
                    "updates": tuple(updates),
                    "accepted_updates": sum(update.accepted for update in updates),
                    "dense_fallbacks": sum(
                        bool(
                            update.merged_solve is not None
                            and update.merged_solve.dense_fallback
                        )
                        for update in updates
                    ),
                    "pair_wall_time_seconds": float(
                        sum(update.wall_time_seconds for update in updates)
                    ),
                    "slowest_pair_wall_time_seconds": float(
                        max(
                            (update.wall_time_seconds for update in updates),
                            default=0.0,
                        )
                    ),
                    "operator_assembly_seconds": float(
                        sum(update.operator_assembly_seconds for update in updates)
                    ),
                    "merged_solve_seconds": float(
                        sum(update.merged_solve_seconds for update in updates)
                    ),
                    "split_seconds": float(
                        sum(update.split_seconds for update in updates)
                    ),
                }
            )
            if verbose:
                print(
                    f"two-site sweep {directional_sweep:3d}  "
                    f"energy={energy:.14f}  delta={delta:.3e}  "
                    f"accepted={endpoint_accepted}"
                )
            if endpoint_accepted and delta < tol:
                self.converged = True
                break
            previous = energy
        # Accepted pair updates invalidate standalone optimization history.
        # Keep the directional records in a local accumulator so those resets
        # cannot discard rows from earlier sweeps.
        self.history = sweep_history
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
        if mode in {"checkpoint", "recompute"}:
            mode = "checkpointed"
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

        if isinstance(self._hamiltonian_frontier, BlockMPOFrontier):
            return self._hamiltonian_frontier.contraction_is_exact
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
        local_is_tt = (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and self._hamiltonian_frontier.local_backend == "tensor_train"
        )
        if not norm_is_tt and not hamiltonian_is_tt and not local_is_tt:
            return None
        return {
            "norm": (
                self._norm_frontier.diagnostics
                if norm_is_tt
                else None
            ),
            "hamiltonian": (
                self._hamiltonian_frontier.diagnostics
                if hamiltonian_is_tt or local_is_tt
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

    def _dense_pair_metric_eigensystem(self, metric):
        """Diagonalize one dense merged-pair metric for all update stages."""
        return linalg.eigh(
            self._hermitian_part(metric),
            check_finite=False,
        )

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
        hamiltonian_frontier = (
            self._exact_hamiltonian_frontier
            if self._exact_hamiltonian_frontier is not None
            else self._hamiltonian_frontier
        )
        numerator = hamiltonian_frontier.scalar(self.tensors)
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
        if self.uses_tensor_train_frontier:
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
        if (
            isinstance(self._hamiltonian_frontier, BlockMPOFrontier)
            and not self.hamiltonian_contraction_is_exact
            and self.tt_hermitize
        ):
            effective = self._hamiltonian_frontier.hole_matrix(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
            )
            effective = 0.5 * (effective + effective.T.conj())
            return effective @ vector
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

    def _complete_local_solution(
        self,
        site,
        old_vector,
        vector,
        *,
        metric,
        metric_tol,
        environment,
    ):
        """Subclass hook for locally invisible parameter directions."""
        del site, old_vector, metric, metric_tol, environment
        return np.asarray(vector).reshape(-1)

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
            if self.uses_tensor_train_frontier:
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
            and self.uses_tensor_train_frontier
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
        local_metric_matrix = None
        local_effective_matrix = None
        try:
            if selected_solver == "direct":
                metric, effective = self.local_operators(
                    site,
                    environment=environment,
                )
                local_metric_matrix = metric
                local_effective_matrix = effective
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
                local_metric_matrix = metric
                local_effective_matrix = effective
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
            vector = self._complete_local_solution(
                site,
                old_tensor.reshape(-1),
                vector,
                metric=local_metric_matrix,
                metric_tol=metric_tol,
                environment=environment,
            )
            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            if not self.hamiltonian_contraction_is_exact:
                # TT rounding makes the contracted scalar nonlinear in a local
                # tensor.  The Davidson solution is therefore only a proposal,
                # which must be checked against a fresh exact contraction.
                self.tensors[site][...] = vector.reshape(old_tensor.shape)
                checked_energy = self.expectation()
                accepted = (
                    np.isfinite(checked_energy)
                    and checked_energy <= energy_before + tolerance
                )
                if accepted:
                    energy_after = float(checked_energy)
                    message = f"{message}; accepted by exact global energy check"
                else:
                    energy_after = energy_before
                    message = f"{message}; rejected by exact global energy check"
            elif self._uses_custom_operator_mpos:
                # Exact projected objectives can have metric-null tensor
                # coordinates.  Completing the local solution retains their
                # incumbent component.  Verify that completed vector with the
                # cached exact local environment rather than recontracting the
                # full network after every site.
                if local_metric_matrix is None:
                    metric_vector = self.metric_action(
                        site,
                        vector,
                        environment=environment,
                    )
                else:
                    metric_vector = local_metric_matrix @ vector
                if local_effective_matrix is None:
                    hamiltonian_vector = self.hamiltonian_action(
                        site,
                        vector,
                        environment=environment,
                    )
                else:
                    hamiltonian_vector = local_effective_matrix @ vector
                denominator = np.vdot(vector, metric_vector)
                if abs(denominator) <= np.finfo(float).tiny:
                    raise ValueError(
                        "completed local solution has zero projected norm."
                    )
                checked_energy = float(
                    np.real(np.vdot(vector, hamiltonian_vector) / denominator)
                )
                accepted = (
                    np.isfinite(checked_energy)
                    and checked_energy <= energy_before + tolerance
                )
                if accepted:
                    self.tensors[site][...] = vector.reshape(old_tensor.shape)
                    energy_after = checked_energy
                    message = f"{message}; accepted by exact local energy check"
                else:
                    energy_after = energy_before
                    message = f"{message}; rejected by exact local energy check"
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

    def _build_environment_checkpoints(self, frontier, *, direction, interval):
        """Contract one side once while retaining only selected cut messages."""
        cuts = frozenset(self._environment_checkpoint_cuts(interval))
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
        environment_cache="checkpointed",
        environment_checkpoint_interval: int | None = None,
        verbose: bool = False,
    ):
        r"""Optimize all tensors without constructing the full Hilbert basis.

        ``environment_cache="checkpointed"`` stores fixed-side messages only
        at block boundaries and recomputes at most one block at a time.  With
        the default interval near :math:`\sqrt{N}`, the number of simultaneously
        retained messages falls from :math:`O(N)` to :math:`O(\sqrt{N})` while
        preserving the exact directional-sweep environments.  ``"full"``
        retains the historical all-cut cache.
        """
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        sweep_offset = int(sweep_offset)
        if sweep_offset < 0:
            raise ValueError("sweep_offset must be nonnegative.")
        environment_cache = str(environment_cache).lower().replace("-", "_")
        if environment_cache in {"checkpoint", "recompute"}:
            environment_cache = "checkpointed"
        if environment_cache not in {"checkpointed", "full"}:
            raise ValueError(
                "environment_cache must be 'checkpointed' or 'full'."
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
                if environment_cache == "full":
                    norm_right = self._norm_frontier.build_right(self.tensors)
                    hamiltonian_right = self._hamiltonian_frontier.build_right(
                        self.tensors
                    )
                    right_checkpoints = None
                else:
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
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                checkpoint_cuts = self._environment_checkpoint_cuts(
                    environment_checkpoint_interval
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
                if environment_cache == "full":
                    norm_left = self._norm_frontier.build_left(self.tensors)
                    hamiltonian_left = self._hamiltonian_frontier.build_left(
                        self.tensors
                    )
                    left_checkpoints = None
                else:
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
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                checkpoint_cuts = self._environment_checkpoint_cuts(
                    environment_checkpoint_interval
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
                    "environment_cache": environment_cache,
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


# Backward-compatible but shorter public aliases for the frontier graph ansatz.
GraphLETTA = FrontierTiedLETTA


__all__ = [
    "FrontierBondExpansion",
    "FrontierGaugeUpdate",
    "FrontierNaturalGradientUpdate",
    "FrontierSiteEnvironment",
    "FrontierSiteUpdate",
    "FrontierTiedLETTA",
    "GraphLETTA",
    "FrontierTwoSiteUpdate",
]
