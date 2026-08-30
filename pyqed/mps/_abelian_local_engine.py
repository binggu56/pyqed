"""Abelian block-sparse local Hamiltonian actions."""

from ._mps_common import *
from ._mps_state import (
    DenseLocalProblem,
    coarse_grain_MPO,
    coarse_grain_MPS,
    contract_from_left,
    contract_from_right,
)


class HamiltonianMultiplyU1:
    """
    Symmetric version of HamiltonianMultiply using BlockTensor.
    """
    _GLOBAL_BOUNDARY_EINSUM_EXPR_CACHE = {}
    _GLOBAL_BOUNDARY_BATCHED_EINSUM_EXPR_CACHE = {}
    _GLOBAL_BOUNDARY_DIRECT_EINSUM_EXPR_CACHE = {}
    _GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE = {}
    _GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE = {}
    _GLOBAL_FLAT_COMPLEMENTARY_ACTION_PATTERN_CACHE = OrderedDict()

    def __init__(
        self,
        E,
        W,
        F,
        complementary_operator_families=None,
        bond=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
        matvec_options=None,
    ):
        self.E = E
        self.W = W
        self.F = F
        self.complementary_operator_families = complementary_operator_families
        self.bond = None if bond is None else int(bond)
        self.complementary_boundary_payloads = complementary_boundary_payloads or {}
        self.complementary_split_stats = complementary_split_stats
        self.complementary_family_environments = complementary_family_environments or {}
        self.complementary_direct_family_environments = (
            complementary_direct_family_environments or {}
        )
        self._matvec_options_ref = matvec_options
        self._complementary_operator_families_options_ref = (
            complementary_operator_families
        )

        def _option(name, default):
            if matvec_options is not None:
                if isinstance(matvec_options, dict) and name in matvec_options:
                    return matvec_options[name]
                if not isinstance(matvec_options, dict) and hasattr(matvec_options, name):
                    return getattr(matvec_options, name)
            return getattr(complementary_operator_families, name, default)

        self._prefer_boundary_factorized = bool(
            _option(
                "prefer_complementary_payload_tensor_matvec",
                False,
            )
        )
        self._debug_boundary_channel_matrices = bool(
            _option(
                "debug_boundary_channel_matrices",
                False,
            )
        )
        self._debug_complementary_split_metadata = bool(
            _option(
                "debug_complementary_split_metadata",
                False,
            )
        )
        self._debug_complementary_action_check = bool(
            _option(
                "debug_complementary_action_check",
                False,
            )
        )
        self._debug_complementary_action_check_tol = float(
            _option(
                "debug_complementary_action_check_tol",
                1.0e-10,
            )
        )
        self._debug_complementary_action_check_limit = int(
            _option(
                "debug_complementary_action_check_limit",
                32,
            )
        )
        self._prefer_precontracted_family_environment = bool(
            _option(
                "prefer_precontracted_family_environment",
                False,
            )
        )
        self._direct_operator_batch_min_entries = int(
            _option(
                "direct_operator_batch_min_entries",
                2,
            )
        )
        if self._direct_operator_batch_min_entries < 2:
            self._direct_operator_batch_min_entries = 2
        self._direct_operator_selector_enabled = bool(
            _option("direct_operator_selector_enabled", True)
        )
        self._direct_operator_selector_min_entries = int(
            _option(
                "direct_operator_selector_min_entries",
                96,
            )
        )
        if self._direct_operator_selector_min_entries < 1:
            self._direct_operator_selector_min_entries = 1
        self._direct_operator_selector_edge_max_entries = int(
            getattr(
                complementary_operator_families,
                "direct_operator_selector_edge_max_entries",
                40,
            )
        )
        if self._direct_operator_selector_edge_max_entries < 0:
            self._direct_operator_selector_edge_max_entries = 0
        self._generic_chain_selector_enabled = bool(
            _option(
                "generic_chain_selector_enabled",
                False,
            )
        )
        self._matrix_chain_selector_enabled = bool(
            _option(
                "matrix_chain_selector_enabled",
                False,
            )
        )
        self._compact_matrix_chain_selector_enabled = bool(
            _option(
                "compact_matrix_chain_selector_enabled",
                False,
            )
        )
        self._native_compact_matrix_chain_selector_enabled = bool(
            _option(
                "native_compact_matrix_chain_selector_enabled",
                False,
            )
        )
        self._batched_compact_matrix_chain_selector_enabled = bool(
            _option(
                "batched_compact_matrix_chain_selector_enabled",
                False,
            )
        )
        self._matrix_chain_force = bool(
            _option(
                "matrix_chain_force",
                False,
            )
        )
        self._compact_matrix_chain_force = bool(
            _option(
                "compact_matrix_chain_force",
                False,
            )
        )
        self._native_compact_matrix_chain_force = bool(
            _option(
                "native_compact_matrix_chain_force",
                False,
            )
        )
        self._batched_compact_matrix_chain_force = bool(
            _option(
                "batched_compact_matrix_chain_force",
                False,
            )
        )
        self._matrix_chain_speedup_threshold = float(
            _option(
                "matrix_chain_speedup_threshold",
                0.8,
            )
        )
        self._compact_matrix_chain_speedup_threshold = float(
            _option(
                "compact_matrix_chain_speedup_threshold",
                0.8,
            )
        )
        self._native_compact_matrix_chain_speedup_threshold = float(
            _option(
                "native_compact_matrix_chain_speedup_threshold",
                0.8,
            )
        )
        self._batched_compact_matrix_chain_speedup_threshold = float(
            _option(
                "batched_compact_matrix_chain_speedup_threshold",
                0.8,
            )
        )
        self._batched_compact_matrix_chain_max_batch_entries = int(
            _option(
                "batched_compact_matrix_chain_max_batch_entries",
                64,
            )
        )
        if self._batched_compact_matrix_chain_max_batch_entries < 1:
            self._batched_compact_matrix_chain_max_batch_entries = 1
        self._batched_compact_matrix_chain_compiled_kernel = bool(
            _option(
                "batched_compact_matrix_chain_compiled_kernel",
                False,
            )
        )
        self._batched_compact_matrix_chain_cython_kernel = bool(
            _option(
                "batched_compact_matrix_chain_cython_kernel",
                False,
            )
        )
        self._batched_compact_matrix_chain_compiled_parallel_kernel = bool(
            _option(
                "batched_compact_matrix_chain_compiled_parallel_kernel",
                False,
            )
        )
        self._batched_compact_matrix_chain_compiled_parallel_min_work = int(
            _option(
                "batched_compact_matrix_chain_compiled_parallel_min_work",
                0,
            )
        )
        if self._batched_compact_matrix_chain_compiled_parallel_min_work < 0:
            self._batched_compact_matrix_chain_compiled_parallel_min_work = 0
        self._matvec_selector_per_layout = bool(
            _option(
                "matvec_selector_per_layout",
                False,
            )
        )
        self._batched_action_selector_enabled = bool(
            _option(
                "batched_action_selector_enabled",
                False,
            )
        )
        self._packed_local_davidson = bool(
            _option(
                "packed_local_davidson",
                False,
            )
        )
        self._packed_local_davidson_min_dim = int(
            _option(
                "packed_local_davidson_min_dim",
                0,
            )
        )
        if self._packed_local_davidson_min_dim < 0:
            self._packed_local_davidson_min_dim = 0
        self._packed_local_davidson_max_dim = int(
            _option(
                "packed_local_davidson_max_dim",
                0,
            )
        )
        if self._packed_local_davidson_max_dim < 0:
            self._packed_local_davidson_max_dim = 0
        self._packed_local_davidson_max_iter = int(
            _option(
                "packed_local_davidson_max_iter",
                0,
            )
        )
        if self._packed_local_davidson_max_iter < 0:
            self._packed_local_davidson_max_iter = 0
        self._packed_local_davidson_restart_dim = int(
            _option(
                "packed_local_davidson_restart_dim",
                0,
            )
        )
        if self._packed_local_davidson_restart_dim < 0:
            self._packed_local_davidson_restart_dim = 0
        self._packed_local_block_preconditioner = bool(
            _option(
                "packed_local_block_preconditioner",
                False,
            )
        )
        self._moving_environment_cpp_davidson = bool(
            _option(
                "moving_environment_cpp_davidson",
                False,
            )
        )
        self._moving_environment_cpp_debug_compare = bool(
            _option(
                "moving_environment_cpp_debug_compare",
                False,
            )
        )
        self._moving_environment_cpp_debug_return_cpp = bool(
            _option(
                "moving_environment_cpp_debug_return_cpp",
                False,
            )
        )
        if self._moving_environment_cpp_debug_compare:
            self._moving_environment_cpp_davidson = True
        self._packed_local_allow_layout_expansion = bool(
            _option(
                "packed_local_allow_layout_expansion",
                True,
            )
        )
        self._packed_local_safe_layout_expansion = bool(
            _option(
                "packed_local_safe_layout_expansion",
                True,
            )
        )
        self._packed_local_preflight_safe_closure = bool(
            _option(
                "packed_local_preflight_safe_closure",
                True,
            )
        )
        self._packed_local_use_safe_closure = bool(
            _option(
                "packed_local_use_safe_closure",
                True,
            )
        )
        self._packed_local_project_current_support = bool(
            _option(
                "packed_local_project_current_support",
                False,
            )
        )
        self._packed_local_project_current_support_truncate = bool(
            _option(
                "packed_local_project_current_support_truncate",
                True,
            )
        )
        self._packed_local_accept_projected_unconverged = bool(
            _option(
                "packed_local_accept_projected_unconverged",
                True,
            )
        )
        self._packed_local_accept_unconverged = bool(
            _option(
                "packed_local_accept_unconverged",
                False,
            )
        )
        self._packed_local_fallback_warm_start = bool(
            _option(
                "packed_local_fallback_warm_start",
                True,
            )
        )
        self._packed_local_fallback_warm_start_max_dim = int(
            _option(
                "packed_local_fallback_warm_start_max_dim",
                4096,
            )
        )
        if self._packed_local_fallback_warm_start_max_dim < 0:
            self._packed_local_fallback_warm_start_max_dim = 0
        self._packed_local_flat_matvec = bool(
            _option(
                "packed_local_flat_matvec",
                True,
            )
        )
        self._packed_local_flat_projected_matvec = bool(
            _option(
                "packed_local_flat_projected_matvec",
                True,
            )
        )
        self._packed_local_flat_preconditioner = bool(
            _option(
                "packed_local_flat_preconditioner",
                True,
            )
        )
        self._packed_local_family_flat_matvec = bool(
            _option(
                "packed_local_family_flat_matvec",
                False,
            )
        )
        self._packed_local_family_flat_matvec_max_dim = int(
            _option(
                "packed_local_family_flat_matvec_max_dim",
                512,
            )
        )
        if self._packed_local_family_flat_matvec_max_dim < 0:
            self._packed_local_family_flat_matvec_max_dim = 0
        self._shared_flat_complementary_action_table_cache = _option(
            "packed_local_family_flat_action_cache",
            None,
        )
        self._shared_flat_complementary_action_table_cache_max_entries = int(
            _option(
                "packed_local_family_flat_action_cache_max_entries",
                256,
            )
        )
        if self._shared_flat_complementary_action_table_cache_max_entries < 0:
            self._shared_flat_complementary_action_table_cache_max_entries = 0
        self._shared_direct_operator_block_index_cache = _option(
            "direct_operator_block_index_cache",
            None,
        )
        self._shared_direct_operator_block_index_cache_max_entries = int(
            _option(
                "direct_operator_block_index_cache_max_entries",
                8192,
            )
        )
        if self._shared_direct_operator_block_index_cache_max_entries < 0:
            self._shared_direct_operator_block_index_cache_max_entries = 0
        self._packed_local_family_flat_matvec_build_after_calls = int(
            _option(
                "packed_local_family_flat_matvec_build_after_calls",
                0,
            )
        )
        if self._packed_local_family_flat_matvec_build_after_calls < 0:
            self._packed_local_family_flat_matvec_build_after_calls = 0
        self._packed_local_family_flat_sparse_entry_emitter = bool(
            _option(
                "packed_local_family_flat_sparse_entry_emitter",
                False,
            )
        )
        self._packed_local_family_flat_direct_csr_build_max_dim = int(
            _option(
                "packed_local_family_flat_direct_csr_build_max_dim",
                4096,
            )
        )
        if self._packed_local_family_flat_direct_csr_build_max_dim < 0:
            self._packed_local_family_flat_direct_csr_build_max_dim = 0
        self._packed_local_family_flat_direct_csr_extract_backend = str(
            _option(
                "packed_local_family_flat_direct_csr_extract_backend",
                "cython",
            )
        ).strip().lower()
        self._packed_local_family_flat_group_identity_csr = bool(
            _option(
                "packed_local_family_flat_group_identity_csr",
                False,
            )
        )
        self._packed_local_family_flat_group_local_generator_csr = bool(
            _option(
                "packed_local_family_flat_group_local_generator_csr",
                False,
            )
        )
        self._packed_local_family_flat_defer_identity_scale = bool(
            _option(
                "packed_local_family_flat_defer_identity_scale",
                False,
            )
        )
        self._packed_local_family_flat_collect_entry_groups = bool(
            _option(
                "packed_local_family_flat_collect_entry_groups",
                False,
            )
        )
        self._packed_local_family_flat_direct_matvec = bool(
            _option(
                "packed_local_family_flat_direct_matvec",
                False,
            )
        )
        self._packed_local_family_flat_direct_matvec_backend = str(
            _option(
                "packed_local_family_flat_direct_matvec_backend",
                "compiled",
            )
        ).strip().lower()
        if self._packed_local_family_flat_direct_matvec_backend in {
            "block2",
            "block2_like",
            "block2-like",
            "block2_table",
            "renormalized",
            "renormalized_operator_table",
            "renormalized-operator-table",
        }:
            self._packed_local_family_flat_direct_matvec_backend = (
                "renormalized_table"
            )
        self._packed_local_family_flat_direct_matvec_min_dim = int(
            _option(
                "packed_local_family_flat_direct_matvec_min_dim",
                0,
            )
        )
        if self._packed_local_family_flat_direct_matvec_min_dim < 0:
            self._packed_local_family_flat_direct_matvec_min_dim = 0
        self._renormalized_operator_table_dense_block_max_elements = int(
            _option(
                "renormalized_operator_table_dense_block_max_elements",
                20_000_000,
            )
        )
        if self._renormalized_operator_table_dense_block_max_elements < 0:
            self._renormalized_operator_table_dense_block_max_elements = 0
        self._renormalized_operator_table_sparse_density_threshold = float(
            _option(
                "renormalized_operator_table_sparse_density_threshold",
                0.0,
            )
        )
        if self._renormalized_operator_table_sparse_density_threshold < 0.0:
            self._renormalized_operator_table_sparse_density_threshold = 0.0
        self._shared_flat_complementary_action_pattern_cache = _option(
            "packed_local_family_flat_action_pattern_cache",
            None,
        )
        self._shared_flat_complementary_action_pattern_cache_max_entries = int(
            _option(
                "packed_local_family_flat_action_pattern_cache_max_entries",
                0,
            )
        )
        if self._shared_flat_complementary_action_pattern_cache_max_entries < 0:
            self._shared_flat_complementary_action_pattern_cache_max_entries = 0
        self._packed_local_family_flat_action_pattern_cache_max_lookup_dim = int(
            _option(
                "packed_local_family_flat_action_pattern_cache_max_lookup_dim",
                4096,
            )
        )
        if self._packed_local_family_flat_action_pattern_cache_max_lookup_dim < 0:
            self._packed_local_family_flat_action_pattern_cache_max_lookup_dim = 0
        self._packed_local_cython_arena = bool(
            _option(
                "packed_local_cython_arena",
                False,
            )
        )
        self._packed_local_projected_accept_min_retained_norm = float(
            _option(
                "packed_local_projected_accept_min_retained_norm",
                0.0,
            )
        )
        if self._packed_local_projected_accept_min_retained_norm < 0.0:
            self._packed_local_projected_accept_min_retained_norm = 0.0
        self._packed_local_projected_accept_max_residual = float(
            _option(
                "packed_local_projected_accept_max_residual",
                0.0,
            )
        )
        if self._packed_local_projected_accept_max_residual < 0.0:
            self._packed_local_projected_accept_max_residual = 0.0
        self._packed_local_return_current_on_rejected_projected = bool(
            _option(
                "packed_local_return_current_on_rejected_projected",
                False,
            )
        )
        self._packed_local_disable_generic_fallback = bool(
            _option(
                "packed_local_disable_generic_fallback",
                False,
            )
        )
        self._packed_local_large_safe_max_dim = int(
            _option(
                "packed_local_large_safe_max_dim",
                0,
            )
        )
        if self._packed_local_large_safe_max_dim < 0:
            self._packed_local_large_safe_max_dim = 0
        self._packed_local_large_safe_restart_dim = int(
            _option(
                "packed_local_large_safe_restart_dim",
                0,
            )
        )
        if self._packed_local_large_safe_restart_dim < 0:
            self._packed_local_large_safe_restart_dim = 0
        self._packed_local_large_safe_require_flat = bool(
            _option(
                "packed_local_large_safe_require_flat",
                True,
            )
        )
        self._packed_local_block_preconditioner_max_block_dim = int(
            _option(
                "packed_local_block_preconditioner_max_block_dim",
                16,
            )
        )
        if self._packed_local_block_preconditioner_max_block_dim < 1:
            self._packed_local_block_preconditioner_max_block_dim = 1
        self._packed_local_block_preconditioner_max_total_dim = int(
            _option(
                "packed_local_block_preconditioner_max_total_dim",
                128,
            )
        )
        if self._packed_local_block_preconditioner_max_total_dim < 0:
            self._packed_local_block_preconditioner_max_total_dim = 0
        self.dtype = np.float64
        self._dense_cache = {}
        self._action_cache = {}
        self._compiled_action_cap = 4
        self._boundary_factorized_cache = {}
        self._combined_direct_family_plan_cache = {}
        self._direct_operator_batched_plan_cache = {}
        self._direct_operator_matrix_kernel_cache = {}
        self._direct_operator_block_index_cache = (
            self._shared_direct_operator_block_index_cache
            if self._shared_direct_operator_block_index_cache is not None
            else {}
        )
        self._direct_operator_left_kernel_cache = {}
        self._direct_operator_right_kernel_cache = {}
        self._generic_chain_plan_cache = {}
        self._generic_matrix_chain_plan_cache = {}
        self._compact_matrix_chain_plan_cache = {}
        self._boundary_factorized_action_cap = 8192
        self._w12_cache = None
        self._local_complementary_cache = {}
        self._local_complementary_channel_cache = {}
        self._boundary_table_cache = {}
        self._flat_complementary_action_table_cache = {}
        self._flat_complementary_family_matvec_counts = {}
        self._flat_named_family_kernel_cache = {}
        self._flat_direct_family_kernel_cache = {}
        self._flat_generator_family_kernel_cache = {}
        self._flat_renormalized_operator_table_cache = {}
        self._flat_direct_renormalized_operator_table_cache = {}
        self._fused_family_environment_cache = None
        self._boundary_table_max_dim = int(
            getattr(
                complementary_operator_families,
                "boundary_table_max_dim",
                32,
            )
        )
        self._boundary_einsum_path_cache = {}
        self._boundary_einsum_expr_cache = self._GLOBAL_BOUNDARY_EINSUM_EXPR_CACHE
        self._boundary_batched_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_BATCHED_EINSUM_EXPR_CACHE
        )
        self._boundary_direct_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_DIRECT_EINSUM_EXPR_CACHE
        )
        self._diagonal_cache = {}
        self._flat_diagonal_cache = {}
        self.profile_stats = {
            "bond": self.bond,
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "paths": {},
            "plan_builds": {},
            "preconditioner": {},
        }
        self._matvec_action_choice = None
        self._matvec_action_choice_by_layout = {}
        self.last_packed_davidson_candidate = None
        self.last_packed_davidson_candidate_flat = None
        self.last_packed_davidson_candidate_layout = None
        self.last_packed_davidson_candidate_energy = None
        self.last_packed_davidson_candidate_residual = None
        self.last_packed_davidson_solution_flat = None
        self.last_packed_davidson_solution_layout = None
        self.last_packed_davidson_solution_energy = None
        self.last_packed_davidson_solution_residual = None
        self.last_packed_davidson_solution_converged = None

    def reset_local_problem(
        self,
        E,
        W,
        F,
        *,
        complementary_operator_families=None,
        bond=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
        matvec_options=None,
    ):
        """Retarget an existing local operator without reparsing stable options."""
        families = (
            self._complementary_operator_families_options_ref
            if complementary_operator_families is None
            else complementary_operator_families
        )
        options = self._matvec_options_ref if matvec_options is None else matvec_options
        if (
            families is not self._complementary_operator_families_options_ref
            or options is not self._matvec_options_ref
        ):
            return False

        self.E = E
        self.W = W
        self.F = F
        self.complementary_operator_families = families
        self.bond = None if bond is None else int(bond)
        self.complementary_boundary_payloads = complementary_boundary_payloads or {}
        self.complementary_split_stats = complementary_split_stats
        self.complementary_family_environments = complementary_family_environments or {}
        self.complementary_direct_family_environments = (
            complementary_direct_family_environments or {}
        )

        self.dtype = np.float64
        self._dense_cache = {}
        self._action_cache = {}
        self._compiled_action_cap = 4
        self._boundary_factorized_cache = {}
        self._combined_direct_family_plan_cache = {}
        self._direct_operator_batched_plan_cache = {}
        self._direct_operator_matrix_kernel_cache = {}
        self._direct_operator_block_index_cache = (
            self._shared_direct_operator_block_index_cache
            if self._shared_direct_operator_block_index_cache is not None
            else {}
        )
        self._direct_operator_left_kernel_cache = {}
        self._direct_operator_right_kernel_cache = {}
        self._generic_chain_plan_cache = {}
        self._generic_matrix_chain_plan_cache = {}
        self._compact_matrix_chain_plan_cache = {}
        self._boundary_factorized_action_cap = 8192
        self._w12_cache = None
        self._local_complementary_cache = {}
        self._local_complementary_channel_cache = {}
        self._boundary_table_cache = {}
        self._flat_complementary_action_table_cache = {}
        self._flat_complementary_family_matvec_counts = {}
        self._flat_named_family_kernel_cache = {}
        self._flat_direct_family_kernel_cache = {}
        self._flat_generator_family_kernel_cache = {}
        self._flat_renormalized_operator_table_cache = {}
        self._flat_direct_renormalized_operator_table_cache = {}
        self._fused_family_environment_cache = None
        self._boundary_table_max_dim = int(
            getattr(
                families,
                "boundary_table_max_dim",
                32,
            )
        )
        self._boundary_einsum_path_cache = {}
        self._boundary_einsum_expr_cache = self._GLOBAL_BOUNDARY_EINSUM_EXPR_CACHE
        self._boundary_batched_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_BATCHED_EINSUM_EXPR_CACHE
        )
        self._boundary_direct_einsum_expr_cache = (
            self._GLOBAL_BOUNDARY_DIRECT_EINSUM_EXPR_CACHE
        )
        self._diagonal_cache = {}
        self._flat_diagonal_cache = {}
        self.profile_stats = {
            "bond": self.bond,
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "paths": {},
            "plan_builds": {},
            "preconditioner": {},
        }
        self._matvec_action_choice = None
        self._matvec_action_choice_by_layout = {}
        self.last_packed_davidson_candidate = None
        self.last_packed_davidson_candidate_flat = None
        self.last_packed_davidson_candidate_layout = None
        self.last_packed_davidson_candidate_energy = None
        self.last_packed_davidson_candidate_residual = None
        self.last_packed_davidson_solution_flat = None
        self.last_packed_davidson_solution_layout = None
        self.last_packed_davidson_solution_energy = None
        self.last_packed_davidson_solution_residual = None
        self.last_packed_davidson_solution_converged = None
        return True

    def _record_plan_profile(self, name, elapsed, **metadata):
        plans = self.profile_stats.setdefault("plan_builds", {})
        entry = plans.setdefault(
            str(name),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)
        entry["last"] = {str(key): value for key, value in metadata.items()}

    def _record_matvec_profile(self, path, elapsed):
        self.profile_stats["matvec_calls"] = int(self.profile_stats.get("matvec_calls", 0)) + 1
        self.profile_stats["matvec_seconds"] = float(self.profile_stats.get("matvec_seconds", 0.0)) + float(elapsed)
        paths = self.profile_stats.setdefault("paths", {})
        entry = paths.setdefault(
            str(path),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)

    def profile_summary(self):
        paths = self.profile_stats.get("paths", {})
        dominant = None
        if paths:
            dominant = max(paths.items(), key=lambda item: float(item[1].get("seconds", 0.0)))[0]
        return {
            "bond": self.bond,
            "matvec_calls": int(self.profile_stats.get("matvec_calls", 0)),
            "matvec_seconds": float(self.profile_stats.get("matvec_seconds", 0.0)),
            "dominant_path": dominant,
            "paths": {
                str(name): {
                    "calls": int(stats.get("calls", 0)),
                    "seconds": float(stats.get("seconds", 0.0)),
                    "last_seconds": float(stats.get("last_seconds", 0.0)),
                }
                for name, stats in paths.items()
            },
            "plan_builds": self.profile_stats.get("plan_builds", {}),
            "local_solver": self.profile_stats.get("local_solver", {}),
            "packed_local_davidson": self.profile_stats.get("packed_local_davidson", {}),
            "preconditioner": self.profile_stats.get("preconditioner", {}),
            "action_selector": self.profile_stats.get("action_selector", {}),
            "direct_operator": self.profile_stats.get("direct_operator", {}),
            "generic_chain": self.profile_stats.get("generic_chain", {}),
            "generic_matrix_chain": self.profile_stats.get("generic_matrix_chain", {}),
            "compact_matrix_chain": self.profile_stats.get("compact_matrix_chain", {}),
            "native_compact_matrix_chain": self.profile_stats.get("native_compact_matrix_chain", {}),
            "batched_compact_matrix_chain": self.profile_stats.get("batched_compact_matrix_chain", {}),
            "packed_flat_batched_compact_matrix_chain": self.profile_stats.get(
                "packed_flat_batched_compact_matrix_chain",
                {},
            ),
            "packed_flat_complementary_family_action": self.profile_stats.get(
                "packed_flat_complementary_family_action",
                {},
            ),
            "packed_flat_preconditioner": self.profile_stats.get(
                "packed_flat_preconditioner",
                {},
            ),
        }

    @staticmethod
    def _component_action_token(E, W, F):
        return (
            HamiltonianMultiplyU1._tensor_token(E),
            HamiltonianMultiplyU1._tensor_token(W[0]),
            HamiltonianMultiplyU1._tensor_token(W[1]),
            HamiltonianMultiplyU1._tensor_token(F),
        )

    @staticmethod
    def _component_action_identity_token(E, W, F):
        return (id(E), id(W[0]), id(W[1]), id(F))

    def _cached_block_index(self, tensor, axes):
        axes_key = axes if isinstance(axes, tuple) else tuple(int(axis) for axis in axes)
        key = (id(tensor), axes_key)
        if self._shared_direct_operator_block_index_cache is None:
            cached = self._direct_operator_block_index_cache.get(key)
            if cached is not None:
                return cached
            indexed = self._block_index(tensor, axes_key)
            self._direct_operator_block_index_cache[key] = indexed
            return indexed
        cached = self._direct_operator_block_index_cache.get(key)
        if cached is not None:
            if (
                isinstance(cached, tuple)
                and len(cached) == 2
                and cached[0] is tensor
            ):
                if hasattr(self._direct_operator_block_index_cache, "move_to_end"):
                    self._direct_operator_block_index_cache.move_to_end(key)
                return cached[1]
            if not isinstance(cached, tuple):
                if hasattr(self._direct_operator_block_index_cache, "move_to_end"):
                    self._direct_operator_block_index_cache.move_to_end(key)
                return cached
            self._direct_operator_block_index_cache.pop(key, None)
        indexed = self._block_index(tensor, axes_key)
        self._direct_operator_block_index_cache[key] = (tensor, indexed)
        cache = self._direct_operator_block_index_cache
        cap = int(self._shared_direct_operator_block_index_cache_max_entries)
        while cap > 0 and len(cache) > cap and hasattr(cache, "popitem"):
            try:
                cache.popitem(last=False)
            except TypeError:
                first_key = next(iter(cache))
                del cache[first_key]
        return indexed

    def _build_direct_operator_plan(self, A, E, W, F, label, *, build_expr=True):
        layout = self._layout(A)
        component_token = (
            self._component_action_token(E, W, F)
            if build_expr
            else self._component_action_identity_token(E, W, F)
        )
        cache_key = (
            component_token,
            layout,
            "direct_operator",
            str(label),
            bool(build_expr),
        )
        if cache_key in self._boundary_factorized_cache:
            return self._boundary_factorized_cache[cache_key]

        e_by_ket_l = self._cached_block_index(E, (2,))
        w1_by_left_in = self._cached_block_index(W[0], (0, 3))
        w2_by_left_in = self._cached_block_index(W[1], (0, 3))
        f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))

        left_operator_cache = {}
        right_operator_cache = {}
        entries = []
        out_shapes = {}
        channels = set()
        dtype_args = []
        left_eq = "aij,abux->bijux"
        right_eq = "bcvy,clk->bvylk"
        action_eq = "bijux,jkxy,bvylk->iluv"
        build_start = time.perf_counter()

        def _left_operator(e_key, e_blk, w1_key, w1_blk):
            key = (id(e_blk), id(w1_blk))
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            cached = left_operator_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(left_eq, e_blk, w1_blk, optimize=False)
            )
            left_operator_cache[key] = op
            self._direct_operator_left_kernel_cache[key] = op
            return op

        def _right_operator(w2_key, w2_blk, f_key, f_blk):
            key = (id(w2_blk), id(f_blk))
            cached = self._direct_operator_right_kernel_cache.get(key)
            if cached is not None:
                return cached
            cached = right_operator_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(right_eq, w2_blk, f_blk, optimize=False)
            )
            right_operator_cache[key] = op
            self._direct_operator_right_kernel_cache[key] = op
            return op

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._boundary_factorized_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._boundary_factorized_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._boundary_factorized_cache[cache_key] = None
                        return None
                    channel = w1_key[1]
                    left_op = _left_operator(e_key, e_blk, w1_key, w1_blk)
                    for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._boundary_factorized_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            right_op = _right_operator(w2_key, w2_blk, f_key, f_blk)
                            if left_op.shape[0] != right_op.shape[0]:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                left_op.shape[1],
                                right_op.shape[3],
                                left_op.shape[3],
                                right_op.shape[1],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            expr = None
                            if build_expr:
                                expr_key = (left_op.shape, a_blk.shape, right_op.shape)
                                expr = self._boundary_direct_einsum_expr_cache.get(expr_key)
                                if expr is None:
                                    try:
                                        import opt_einsum as oe

                                        expr = oe.contract_expression(
                                            action_eq,
                                            left_op.shape,
                                            a_blk.shape,
                                            right_op.shape,
                                            optimize="greedy",
                                        )
                                    except Exception:
                                        try:
                                            expr = np.einsum_path(
                                                action_eq,
                                                left_op,
                                                a_blk,
                                                right_op,
                                                optimize="greedy",
                                            )[0]
                                        except ValueError:
                                            self._boundary_factorized_cache[cache_key] = None
                                            return None
                                    self._boundary_direct_einsum_expr_cache[expr_key] = expr
                            dtype_args.extend((left_op.dtype, right_op.dtype))
                            channels.add(channel)
                            entries.append((channel, a_key, out_key, left_op, right_op, expr))
                            if len(entries) > int(self._boundary_factorized_action_cap):
                                self._boundary_factorized_cache[cache_key] = None
                                return None

        if not entries:
            plan = ((), (), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._boundary_factorized_cache[cache_key] = plan
            self._record_plan_profile("direct_operator", time.perf_counter() - build_start, entries=0)
            return plan
        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            tuple(entries),
            out_shapes,
            out_qns,
            A.dirs[:],
            dtype,
        )
        self._boundary_factorized_cache[cache_key] = plan
        self._record_plan_profile(
            "direct_operator",
            time.perf_counter() - build_start,
            entries=int(len(entries)),
            output_blocks=int(len(out_shapes)),
            middle_channels=int(len(channels)),
        )
        return plan

    def _build_boundary_direct_operator_plan(self, A):
        return self._build_direct_operator_plan(
            A,
            self.E,
            self.W,
            self.F,
            "boundary_direct_operator",
        )

    def _direct_operator_batched_plan(self, A, entries):
        cache_key = (id(entries), self._layout(A), "direct_operator_batches")
        cached = self._direct_operator_batched_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        batch_sources = {}
        for index, (channel, a_key, out_key, left_op, right_op, expr) in enumerate(entries):
            a_blk = A.data.get(a_key)
            if a_blk is None:
                continue
            batch_key = (
                out_key,
                left_op.shape,
                a_blk.shape,
                right_op.shape,
            )
            batch_sources.setdefault(batch_key, []).append(
                (index, channel, a_key, left_op, right_op, expr)
            )

        batched_groups = []
        batched_indices = set()
        batch_eq = "zbijux,zjkxy,zbvylk->iluv"
        for batch_key, members in batch_sources.items():
            if len(members) < int(self._direct_operator_batch_min_entries):
                continue
            out_key, left_shape, a_shape, right_shape = batch_key
            expr_key = (len(members), left_shape, a_shape, right_shape, "direct_batch")
            expr = self._boundary_batched_einsum_expr_cache.get(expr_key)
            if expr is None:
                try:
                    import opt_einsum as oe

                    expr = oe.contract_expression(
                        batch_eq,
                        (len(members),) + tuple(left_shape),
                        (len(members),) + tuple(a_shape),
                        (len(members),) + tuple(right_shape),
                        optimize="greedy",
                    )
                except Exception:
                    expr = None
                self._boundary_batched_einsum_expr_cache[expr_key] = expr
            if expr is None:
                continue
            try:
                left_stack = np.stack([entry[3] for entry in members], axis=0)
                right_stack = np.stack([entry[4] for entry in members], axis=0)
            except ValueError:
                continue
            indices = tuple(int(entry[0]) for entry in members)
            batched_indices.update(indices)
            batched_groups.append(
                (
                    out_key,
                    tuple(entry[2] for entry in members),
                    left_stack,
                    right_stack,
                    expr,
                    indices,
                )
            )

        scalar_entries = tuple(
            entry
            for index, entry in enumerate(entries)
            if index not in batched_indices
        )
        plan = (scalar_entries, tuple(batched_groups))
        self._direct_operator_batched_plan_cache[cache_key] = plan
        return plan

    def _direct_operator_matrix_kernel(self, left_op, a_shape, right_op):
        if left_op.ndim != 5 or right_op.ndim != 5 or len(a_shape) != 4:
            return None
        nb, ni, nj, nu, nx = left_op.shape
        nb_r, nv, ny, nl, nk = right_op.shape
        if nb != nb_r:
            return None
        if tuple(a_shape) != (nj, nk, nx, ny):
            return None
        cache_key = (id(left_op), tuple(a_shape), id(right_op), "matrix_kernel")
        cached = self._direct_operator_matrix_kernel_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            left_stack = np.ascontiguousarray(
                left_op.transpose(0, 1, 3, 2, 4).reshape(nb, ni * nu, nj * nx)
            )
            right_stack = np.ascontiguousarray(
                right_op.transpose(0, 4, 2, 3, 1).reshape(nb, nk * ny, nl * nv)
            )
        except ValueError:
            self._direct_operator_matrix_kernel_cache[cache_key] = None
            return None
        kernel = (left_stack, right_stack, (ni, nl, nu, nv), (nj, nx, nk, ny))
        self._direct_operator_matrix_kernel_cache[cache_key] = kernel
        return kernel

    def _use_cpp_raw_payload_builder(self):
        moving_environment = getattr(self, "_moving_environment", None)
        if moving_environment is None:
            return False
        backend = getattr(moving_environment, "compiled_backend", None)
        if backend is None:
            return False
        checker = getattr(backend, "use_cpp_raw_payload_builder", None)
        return bool(checker is not None and checker())

    def _new_cpp_raw_route_plan(self):
        moving_environment = getattr(self, "_moving_environment", None)
        if moving_environment is None:
            return None
        backend = getattr(moving_environment, "compiled_backend", None)
        checker = getattr(backend, "use_cpp_raw_route_plan", None)
        if checker is None or not checker():
            return None
        route_cls = getattr(_cpp_davidson, "RawRoutePlan", None)
        if route_cls is None:
            return None
        try:
            return route_cls()
        except Exception:
            return None

    def _new_direct_csr_collected(self, *, build_groups=True):
        raw_builder = None
        if not bool(build_groups) and self._use_cpp_raw_payload_builder():
            builder_cls = getattr(_cpp_davidson, "RawPayloadBuilder", None)
            if builder_cls is not None:
                try:
                    raw_builder = builder_cls()
                except Exception:
                    raw_builder = None
        collected = {
            "left": [],
            "right": [],
            "dims": [],
            "in_starts": [],
            "out_starts": [],
            "scales": [],
            "family_names": [],
            "entry_count": 0,
        }
        if raw_builder is not None:
            collected["raw_builder"] = raw_builder
        return collected

    @staticmethod
    def _direct_csr_kernel_count(collected):
        if collected is None:
            return 0
        if "entry_count" in collected:
            return int(collected.get("entry_count", 0))
        builder = collected.get("raw_builder")
        if builder is not None:
            try:
                return int(builder.size())
            except Exception:
                return 0
        return int(len(collected.get("left", ())))

    def _append_direct_csr_kernel(
        self,
        out,
        left_stack,
        right_stack,
        dims,
        in_start,
        out_start,
        scale=1.0 + 0.0j,
    ):
        dims = tuple(int(value) for value in dims)
        builder = out.get("raw_builder")
        if builder is not None:
            builder.add(
                left_stack,
                right_stack,
                np.asarray(dims, dtype=np.int64),
                int(in_start),
                int(out_start),
                complex(scale),
            )
        else:
            out["left"].append(left_stack)
            out["right"].append(right_stack)
            out["dims"].append(dims)
            out["in_starts"].append(int(in_start))
            out["out_starts"].append(int(out_start))
            out.setdefault("scales", []).append(complex(scale))
        out["entry_count"] = int(out.get("entry_count", 0)) + 1

    @staticmethod
    def _finalize_direct_csr_collected(collected):
        builder = collected.get("raw_builder")
        if builder is not None:
            collected["matvec_groups"] = None
            return collected
        collected["dims_array"] = np.asarray(collected["dims"], dtype=np.int64)
        collected["in_starts_array"] = np.asarray(
            collected["in_starts"],
            dtype=np.int64,
        )
        collected["out_starts_array"] = np.asarray(
            collected["out_starts"],
            dtype=np.int64,
        )
        if len(collected.get("scales", ())) != len(collected["left"]):
            collected["scales"] = [1.0 + 0.0j] * len(collected["left"])
        scales_array = np.asarray(
            collected.get("scales", ()),
            dtype=np.complex128,
        )
        collected["scales_array"] = (
            scales_array if np.any(scales_array != (1.0 + 0.0j)) else None
        )
        return collected

    @staticmethod
    def _materialize_raw_builder_collected(collected):
        builder = collected.get("raw_builder")
        if builder is None:
            return collected
        materialized = dict(collected)
        materialized["left"] = list(builder.left_entries())
        materialized["right"] = list(builder.right_entries())
        dims_array = np.asarray(builder.dims_array(), dtype=np.int64)
        in_starts_array = np.asarray(builder.in_starts_array(), dtype=np.int64)
        out_starts_array = np.asarray(builder.out_starts_array(), dtype=np.int64)
        materialized["dims_array"] = dims_array
        materialized["in_starts_array"] = in_starts_array
        materialized["out_starts_array"] = out_starts_array
        materialized["dims"] = [tuple(int(v) for v in row) for row in dims_array]
        materialized["in_starts"] = [int(v) for v in in_starts_array]
        materialized["out_starts"] = [int(v) for v in out_starts_array]
        scales = builder.scales_array()
        if scales is None:
            materialized["scales"] = [1.0 + 0.0j] * int(builder.size())
            materialized["scales_array"] = None
        else:
            scales_array = np.asarray(scales, dtype=np.complex128)
            materialized["scales"] = list(scales_array)
            materialized["scales_array"] = scales_array
        materialized["entry_count"] = int(builder.size())
        return materialized

    def _collect_single_direct_operator_csr_kernels_fast(
        self,
        A,
        E,
        W,
        F,
        layout,
        out,
        *,
        route_plan=None,
        route_family_name=None,
    ):
        if E is None or F is None or W is None or len(W) != 2:
            return False
        layout = tuple(layout)
        layout_shapes = {key: tuple(shape) for key, shape in layout}
        offsets, _dim = self._layout_offsets(layout)
        a_entries = []
        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4 or len(a_key) != 4:
                return False
            if a_key not in layout_shapes:
                continue
            left_qn, right_qn, p1_in, p2_in = a_key
            a_entries.append((a_key, tuple(a_blk.shape), left_qn, right_qn, p1_in, p2_in))
        if not a_entries:
            return True

        e_by_ket_l = self._cached_block_index(E, (2,))
        w1_by_left_in = self._cached_block_index(W[0], (0, 3))
        w2_by_left_in = self._cached_block_index(W[1], (0, 3))
        f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))
        left_eq = "aij,abux->biujx"
        right_eq = "bcvy,clk->bkylv"
        emitted = 0
        moving_environment = getattr(self, "_moving_environment", None)
        use_cpp_stack_kernels = False
        if moving_environment is not None:
            use_cpp_stack_kernels = bool(
                MovingEnvironment._option_value(
                    getattr(moving_environment, "matvec_options", None),
                    "moving_environment_cpp_raw_payload_stack_kernels",
                    True,
                )
            ) and bool(
                getattr(
                    getattr(moving_environment, "compiled_backend", None),
                    "use_cpp_raw_grouped_renormalized_table",
                    lambda: False,
                )()
            )
        cpp_left_stack = (
            None
            if _cpp_davidson is None or not use_cpp_stack_kernels
            else getattr(_cpp_davidson, "direct_left_stack", None)
        )
        cpp_right_stack = (
            None
            if _cpp_davidson is None or not use_cpp_stack_kernels
            else getattr(_cpp_davidson, "direct_right_stack", None)
        )

        def _left_stack(e_blk, w1_blk):
            key = (id(e_blk), id(w1_blk), "csr_left_stack")
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            try:
                if cpp_left_stack is not None:
                    stack = np.asarray(
                        cpp_left_stack(e_blk, w1_blk),
                        dtype=np.complex128,
                    )
                else:
                    stack5 = np.einsum(left_eq, e_blk, w1_blk, optimize=False)
                    nb, ni, nu, nj, nx = stack5.shape
                    stack = np.ascontiguousarray(
                        stack5.reshape(nb, ni * nu, nj * nx)
                    )
            except ValueError:
                self._direct_operator_left_kernel_cache[key] = None
                return None
            self._direct_operator_left_kernel_cache[key] = stack
            return stack

        def _right_stack(w2_blk, f_blk):
            key = (id(w2_blk), id(f_blk), "csr_right_stack")
            cached = self._direct_operator_right_kernel_cache.get(key)
            if cached is not None:
                return cached
            try:
                if cpp_right_stack is not None:
                    stack = np.asarray(
                        cpp_right_stack(w2_blk, f_blk),
                        dtype=np.complex128,
                    )
                else:
                    stack5 = np.einsum(right_eq, w2_blk, f_blk, optimize=False)
                    nb, nk, ny, nl, nv = stack5.shape
                    stack = np.ascontiguousarray(
                        stack5.reshape(nb, nk * ny, nl * nv)
                    )
            except ValueError:
                self._direct_operator_right_kernel_cache[key] = None
                return None
            self._direct_operator_right_kernel_cache[key] = stack
            return stack

        for a_key, a_shape, left_qn, right_qn, p1_in, p2_in in a_entries:
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    return False
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        return False
                    left_stack = _left_stack(e_blk, w1_blk)
                    if left_stack is None or left_stack.dtype != np.complex128:
                        return False
                    nb, left_rows, left_cols = left_stack.shape
                    channel = w1_key[1]
                    for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                        if w2_blk.ndim != 4:
                            return False
                        for f_key, f_blk in f_by_mpo_ket_r.get(
                            (w2_key[1], right_qn),
                            (),
                        ):
                            if f_blk.ndim != 3:
                                return False
                            right_stack = _right_stack(w2_blk, f_blk)
                            if (
                                right_stack is None
                                or right_stack.dtype != np.complex128
                                or right_stack.shape[0] != nb
                            ):
                                return False
                            out_key = (
                                e_key[1],
                                f_key[1],
                                w1_key[2],
                                w2_key[2],
                            )
                            out_shape = layout_shapes.get(out_key)
                            if out_shape is None:
                                continue
                            ni = int(e_blk.shape[1])
                            nj = int(e_blk.shape[2])
                            nu = int(w1_blk.shape[2])
                            nx = int(w1_blk.shape[3])
                            nv = int(w2_blk.shape[2])
                            ny = int(w2_blk.shape[3])
                            nl = int(f_blk.shape[1])
                            nk = int(f_blk.shape[2])
                            expected_in_shape = (nj, nk, nx, ny)
                            expected_out_shape = (ni, nl, nu, nv)
                            if (
                                tuple(a_shape) != expected_in_shape
                                or tuple(out_shape) != expected_out_shape
                                or left_rows != ni * nu
                                or left_cols != nj * nx
                                or right_stack.shape[1] != nk * ny
                                or right_stack.shape[2] != nl * nv
                            ):
                                return False
                            in_start, in_size = offsets[a_key]
                            out_start, out_size = offsets[out_key]
                            if (
                                int(in_size) != int(np.prod(expected_in_shape, dtype=int))
                                or int(out_size) != int(np.prod(expected_out_shape, dtype=int))
                            ):
                                return False
                            self._append_direct_csr_kernel(
                                out,
                                left_stack,
                                right_stack,
                                (ni, nl, nu, nv, nj, nx, nk, ny),
                                in_start,
                                out_start,
                            )
                            if route_plan is not None:
                                try:
                                    route_dims = np.asarray(
                                        (ni, nl, nu, nv, nj, nx, nk, ny),
                                        dtype=np.int64,
                                    )
                                    add_with_layout = getattr(
                                        route_plan,
                                        "add_with_layout",
                                        None,
                                    )
                                    if add_with_layout is None:
                                        route_plan.add(
                                            route_family_name,
                                            e_key,
                                            w1_key,
                                            w2_key,
                                            f_key,
                                            route_dims,
                                            int(in_start),
                                            int(out_start),
                                            1.0 + 0.0j,
                                        )
                                    else:
                                        add_with_layout(
                                            route_family_name,
                                            e_key,
                                            w1_key,
                                            w2_key,
                                            f_key,
                                            route_dims,
                                            int(in_start),
                                            int(out_start),
                                            1.0 + 0.0j,
                                            a_key,
                                            out_key,
                                        )
                                except Exception:
                                    route_plan = None
                            emitted += 1
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["single_direct_csr_fast_collector_calls"] = int(
            stats.get("single_direct_csr_fast_collector_calls", 0)
        ) + 1
        stats["single_direct_csr_fast_collector_entries"] = int(
            stats.get("single_direct_csr_fast_collector_entries", 0)
        ) + int(emitted)
        return True

    def _scaled_direct_left_stack(self, left_stack, coeff, source):
        key = (
            id(left_stack),
            complex(coeff),
            str(source),
            "scaled_direct_left_stack",
        )
        cached = self._direct_operator_left_kernel_cache.get(key)
        if cached is not None:
            return cached
        stack = np.ascontiguousarray(
            np.asarray(left_stack, dtype=np.complex128) * complex(coeff),
            dtype=np.complex128,
        )
        self._direct_operator_left_kernel_cache[key] = stack
        return stack

    @staticmethod
    def _direct_operator_matrix_contribution(kernel, a_blk):
        left_stack, right_stack, out_shape, a_matrix_shape = kernel
        nj, nx, nk, ny = a_matrix_shape
        a_mat = np.ascontiguousarray(
            a_blk.transpose(0, 2, 1, 3).reshape(nj * nx, nk * ny)
        )
        tmp = np.matmul(left_stack, a_mat)
        mat = np.matmul(tmp, right_stack).sum(axis=0)
        ni, nl, nu, nv = out_shape
        return mat.reshape(ni, nu, nl, nv).transpose(0, 2, 1, 3)

    def _record_direct_operator_profile(
        self,
        *,
        total_entries,
        batched_entries,
        batched_groups,
        matrix_entries,
        einsum_entries,
        elapsed,
    ):
        stats = self.profile_stats.setdefault(
            "direct_operator",
            {
                "calls": 0,
                "seconds": 0.0,
                "entries": 0,
                "batched_entries": 0,
                "batched_groups": 0,
                "matrix_entries": 0,
                "einsum_entries": 0,
            },
        )
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["entries"] = int(stats.get("entries", 0)) + int(total_entries)
        stats["batched_entries"] = int(stats.get("batched_entries", 0)) + int(batched_entries)
        stats["batched_groups"] = int(stats.get("batched_groups", 0)) + int(batched_groups)
        stats["matrix_entries"] = int(stats.get("matrix_entries", 0)) + int(matrix_entries)
        stats["einsum_entries"] = int(stats.get("einsum_entries", 0)) + int(einsum_entries)
        stats["last"] = {
            "entries": int(total_entries),
            "batched_entries": int(batched_entries),
            "batched_groups": int(batched_groups),
            "matrix_entries": int(matrix_entries),
            "einsum_entries": int(einsum_entries),
            "seconds": float(elapsed),
        }

    def _apply_direct_operator_plan(self, A, plan):
        if plan is None:
            return None
        apply_start = time.perf_counter()
        channels, entries, out_shapes, out_qns, out_dirs, dtype = plan
        action_eq = "bijux,jkxy,bvylk->iluv"
        total_data = {
            key: np.zeros(shape, dtype=dtype)
            for key, shape in out_shapes.items()
        }
        scalar_entries, batched_groups = self._direct_operator_batched_plan(A, entries)
        self._record_direct_operator_batch_stats(entries, batched_groups)
        batched_entries = int(sum(len(group[1]) for group in batched_groups))
        matrix_entries = 0
        einsum_entries = 0
        for out_key, a_keys, left_stack, right_stack, expr, _indices in batched_groups:
            a_stack = np.stack([A.data[a_key] for a_key in a_keys], axis=0)
            total_data[out_key] += expr(left_stack, a_stack, right_stack)
        for _channel, a_key, out_key, left_op, right_op, expr in scalar_entries:
            a_blk = A.data[a_key]
            kernel = self._direct_operator_matrix_kernel(left_op, a_blk.shape, right_op)
            if kernel is not None:
                contribution = self._direct_operator_matrix_contribution(kernel, a_blk)
                matrix_entries += 1
            elif callable(expr):
                contribution = expr(left_op, A.data[a_key], right_op)
                einsum_entries += 1
            else:
                contribution = np.einsum(
                    action_eq,
                    left_op,
                    A.data[a_key],
                    right_op,
                    optimize=expr,
                )
                einsum_entries += 1
            total_data[out_key] += contribution
        self._record_direct_operator_profile(
            total_entries=len(entries),
            batched_entries=batched_entries,
            batched_groups=len(batched_groups),
            matrix_entries=matrix_entries,
            einsum_entries=einsum_entries,
            elapsed=time.perf_counter() - apply_start,
        )
        return (
            self._tensor_from_block_data_like(A, total_data, out_qns, out_dirs),
            channels,
            entries,
        )

    def _accumulate_direct_operator_plan(self, A, plan, total_data):
        if plan is None:
            return None
        apply_start = time.perf_counter()
        channels, entries, out_shapes, out_qns, out_dirs, dtype = plan
        action_eq = "bijux,jkxy,bvylk->iluv"
        for key, shape in out_shapes.items():
            if key not in total_data:
                total_data[key] = np.zeros(shape, dtype=dtype)
            elif total_data[key].shape != tuple(shape):
                return None
        scalar_entries, batched_groups = self._direct_operator_batched_plan(A, entries)
        self._record_direct_operator_batch_stats(entries, batched_groups)
        batched_entries = int(sum(len(group[1]) for group in batched_groups))
        matrix_entries = 0
        einsum_entries = 0
        for out_key, a_keys, left_stack, right_stack, expr, _indices in batched_groups:
            a_stack = np.stack([A.data[a_key] for a_key in a_keys], axis=0)
            contribution = expr(left_stack, a_stack, right_stack)
            if out_key in total_data and total_data[out_key].shape != contribution.shape:
                return None
            total_data[out_key] += contribution
        for _channel, a_key, out_key, left_op, right_op, expr in scalar_entries:
            a_blk = A.data[a_key]
            kernel = self._direct_operator_matrix_kernel(left_op, a_blk.shape, right_op)
            if kernel is not None:
                contribution = self._direct_operator_matrix_contribution(kernel, a_blk)
                matrix_entries += 1
            elif callable(expr):
                contribution = expr(left_op, A.data[a_key], right_op)
                einsum_entries += 1
            else:
                contribution = np.einsum(
                    action_eq,
                    left_op,
                    A.data[a_key],
                    right_op,
                    optimize=expr,
                )
                einsum_entries += 1
            if out_key in total_data and total_data[out_key].shape != contribution.shape:
                return None
            total_data[out_key] += contribution
        self._record_direct_operator_profile(
            total_entries=len(entries),
            batched_entries=batched_entries,
            batched_groups=len(batched_groups),
            matrix_entries=matrix_entries,
            einsum_entries=einsum_entries,
            elapsed=time.perf_counter() - apply_start,
        )
        return channels, entries, out_qns, out_dirs

    def _build_combined_direct_family_plan_fast(self, A, name, component_entries):
        layout = self._layout(A)
        cache_key = (
            layout,
            str(name),
            id(component_entries),
            int(len(component_entries)),
            "combined_direct_family_plan_fast",
        )
        if cache_key in self._combined_direct_family_plan_cache:
            return self._combined_direct_family_plan_cache[cache_key]
        component_entries, packed_count, packed_unique, packed_cancelled = (
            _coalesced_packed_identity_local_entries(component_entries)
        )
        if any(
            isinstance(
                entry,
                (AbelianPackedIdentityLocalEntry, AbelianPackedLocalGeneratorEntry),
            )
            for entry in component_entries
        ):
            self._combined_direct_family_plan_cache[cache_key] = None
            return None
        a_entries = []
        dtype_args = []
        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4 or len(a_key) != 4:
                self._combined_direct_family_plan_cache[cache_key] = None
                return None
            left_qn, right_qn, p1_in, p2_in = a_key
            a_entries.append((a_key, a_blk, left_qn, right_qn, p1_in, p2_in))
            dtype_args.append(a_blk.dtype)

        channels = set()
        entries = []
        out_shapes = {}
        left_eq = "aij,abux->bijux"
        right_eq = "bcvy,clk->bvylk"
        build_start = time.perf_counter()
        emitted = 0
        def _left_operator(e_blk, w1_blk):
            key = (id(e_blk), id(w1_blk))
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(left_eq, e_blk, w1_blk, optimize=False)
            )
            self._direct_operator_left_kernel_cache[key] = op
            return op

        def _right_operator(w2_blk, f_blk):
            key = (id(w2_blk), id(f_blk))
            cached = self._direct_operator_right_kernel_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(right_eq, w2_blk, f_blk, optimize=False)
            )
            self._direct_operator_right_kernel_cache[key] = op
            return op

        for component in component_entries:
            E, W, F = component
            e_by_ket_l = self._cached_block_index(E, (2,))
            w1_by_left_in = self._cached_block_index(W[0], (0, 3))
            w2_by_left_in = self._cached_block_index(W[1], (0, 3))
            f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))
            for a_key, _a_blk, left_qn, right_qn, p1_in, p2_in in a_entries:
                for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                    if e_blk.ndim != 3:
                        self._combined_direct_family_plan_cache[cache_key] = None
                        return None
                    for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                        if w1_blk.ndim != 4:
                            self._combined_direct_family_plan_cache[cache_key] = None
                            return None
                        channel = w1_key[1]
                        left_op = _left_operator(e_blk, w1_blk)
                        for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                            if w2_blk.ndim != 4:
                                self._combined_direct_family_plan_cache[cache_key] = None
                                return None
                            for f_key, f_blk in f_by_mpo_ket_r.get(
                                (w2_key[1], right_qn),
                                (),
                            ):
                                if f_blk.ndim != 3:
                                    self._combined_direct_family_plan_cache[cache_key] = None
                                    return None
                                right_op = _right_operator(w2_blk, f_blk)
                                if left_op.shape[0] != right_op.shape[0]:
                                    self._combined_direct_family_plan_cache[cache_key] = None
                                    return None
                                out_key = (
                                    e_key[1],
                                    f_key[1],
                                    w1_key[2],
                                    w2_key[2],
                                )
                                out_shape = (
                                    left_op.shape[1],
                                    right_op.shape[3],
                                    left_op.shape[3],
                                    right_op.shape[1],
                                )
                                old_shape = out_shapes.get(out_key)
                                if old_shape is not None and old_shape != out_shape:
                                    self._combined_direct_family_plan_cache[cache_key] = None
                                    return None
                                out_shapes[out_key] = out_shape
                                dtype_args.extend((left_op.dtype, right_op.dtype))
                                channels.add(channel)
                                entries.append((
                                    channel,
                                    a_key,
                                    out_key,
                                    left_op,
                                    right_op,
                                    None,
                                ))

        dtype = np.result_type(*(dtype_args or [complex]))
        out_qns = self._qns_from_layout(
            tuple((key, out_shapes[key]) for key in sorted(out_shapes))
        )
        plan = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            tuple(entries),
            out_shapes,
            out_qns,
            A.dirs[:],
            dtype,
        )
        self._combined_direct_family_plan_cache[cache_key] = plan
        self._record_plan_profile(
            "combined_direct_family_operator_fast",
            time.perf_counter() - build_start,
            components=int(len(component_entries)),
            entries=int(len(entries)),
            output_blocks=int(len(out_shapes)),
            middle_channels=int(len(channels)),
        )
        return plan

    def _collect_combined_direct_family_csr_kernels_fast(
        self,
        A,
        name,
        component_entries,
        layout,
        out,
        *,
        route_plan=None,
        route_family_name=None,
        route_group_index=-1,
    ):
        layout = tuple(layout)
        layout_shapes = {key: tuple(shape) for key, shape in layout}
        offsets, _dim = self._layout_offsets(layout)
        route_direct_refs = bool(
            route_plan is not None
            and getattr(route_plan, "add_packed_identity_ref", None) is not None
        )
        if route_direct_refs:
            component_entries = tuple(component_entries or ())
            packed_count = 0
            packed_unique = 0
            packed_cancelled = 0
            packed_local_count = 0
            packed_local_unique = 0
            packed_local_cancelled = 0
        else:
            component_entries, packed_count, packed_unique, packed_cancelled = (
                _coalesced_packed_identity_local_entries(component_entries)
            )
            (
                component_entries,
                packed_local_count,
                packed_local_unique,
                packed_local_cancelled,
            ) = _coalesced_packed_local_generator_entries(component_entries)
        a_entries = []
        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4 or len(a_key) != 4:
                return False
            if a_key not in layout_shapes:
                continue
            left_qn, right_qn, p1_in, p2_in = a_key
            a_entries.append((a_key, a_blk.shape, left_qn, right_qn, p1_in, p2_in))
        if not a_entries:
            return True

        left_eq = "aij,abux->bijux"
        right_eq = "bcvy,clk->bvylk"
        build_start = time.perf_counter()
        emitted = 0
        packed_identity_accum = OrderedDict()
        packed_identity_raw_kernels = 0
        packed_local_generator_accum = OrderedDict()
        packed_local_generator_raw_kernels = 0

        def _left_operator(e_blk, w1_blk):
            key = (id(e_blk), id(w1_blk))
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(left_eq, e_blk, w1_blk, optimize=False)
            )
            self._direct_operator_left_kernel_cache[key] = op
            return op

        def _right_operator(w2_blk, f_blk):
            key = (id(w2_blk), id(f_blk))
            cached = self._direct_operator_right_kernel_cache.get(key)
            if cached is not None:
                return cached
            op = np.ascontiguousarray(
                np.einsum(right_eq, w2_blk, f_blk, optimize=False)
            )
            self._direct_operator_right_kernel_cache[key] = op
            return op

        def _identity_left_stack(e_blk, nx):
            key = (id(e_blk), int(nx), "identity_left_stack")
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            arr = np.asarray(e_blk, dtype=np.complex128)
            if arr.ndim != 3:
                return None
            nb, ni, nj = arr.shape
            nx = int(nx)
            if nx < 1:
                return None
            if nx == 1:
                stack = np.ascontiguousarray(arr.reshape(nb, ni, nj))
            else:
                eye = np.eye(nx, dtype=np.complex128)
                stack = np.empty((nb, ni * nx, nj * nx), dtype=np.complex128)
                for b in range(nb):
                    stack[b] = np.kron(arr[b], eye)
            self._direct_operator_left_kernel_cache[key] = stack
            return stack

        def _identity_right_stack(f_blk, ny):
            key = (id(f_blk), int(ny), "identity_right_stack")
            cached = self._direct_operator_right_kernel_cache.get(key)
            if cached is not None:
                return cached
            arr = np.asarray(f_blk, dtype=np.complex128)
            if arr.ndim != 3:
                return None
            nb, nl, nk = arr.shape
            ny = int(ny)
            if ny < 1:
                return None
            if ny == 1:
                stack = np.ascontiguousarray(arr.transpose(0, 2, 1))
            else:
                eye = np.eye(ny, dtype=np.complex128)
                stack = np.empty((nb, nk * ny, nl * ny), dtype=np.complex128)
                for b in range(nb):
                    stack[b] = np.kron(arr[b].T, eye)
            self._direct_operator_right_kernel_cache[key] = stack
            return stack

        def _identity_scaled_left_stack(left_base, coeff):
            key = (
                id(left_base),
                complex(coeff),
                "identity_scaled_left_stack",
            )
            cached = self._direct_operator_left_kernel_cache.get(key)
            if cached is not None:
                return cached
            stack = np.ascontiguousarray(
                left_base * complex(coeff),
                dtype=np.complex128,
            )
            self._direct_operator_left_kernel_cache[key] = stack
            return stack

        def _collect_packed_identity(entry, component_index):
            nonlocal emitted, packed_identity_raw_kernels
            E = getattr(entry, "E", None)
            F = getattr(entry, "F", None)
            coeff = complex(getattr(entry, "coeff", 1.0))
            source = str(getattr(entry, "source", ""))
            if E is None or F is None:
                return False
            e_by_ket_l = self._cached_block_index(E, (2,))
            f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))
            local_emitted = 0

            def _accumulate(left_base, right_stack, dims, in_start, out_start):
                nonlocal emitted, local_emitted, packed_identity_raw_kernels
                if (
                    not self._packed_local_family_flat_group_identity_csr
                    or out.get("raw_builder") is not None
                ):
                    if (
                        self._packed_local_family_flat_defer_identity_scale
                        or out.get("raw_builder") is not None
                    ):
                        left_stack = left_base
                        scale = coeff
                    else:
                        left_stack = _identity_scaled_left_stack(left_base, coeff)
                        scale = 1.0 + 0.0j
                    self._append_direct_csr_kernel(
                        out,
                        left_stack,
                        right_stack,
                        dims,
                        in_start,
                        out_start,
                        scale,
                    )
                    emitted += 1
                    local_emitted += 1
                    packed_identity_raw_kernels += 1
                    return
                if "same_side_right" in source:
                    key = (
                        "right",
                        id(left_base),
                        tuple(int(v) for v in dims),
                        int(in_start),
                        int(out_start),
                        tuple(int(v) for v in right_stack.shape),
                    )
                    rec = packed_identity_accum.get(key)
                    scaled = np.ascontiguousarray(
                        right_stack * coeff,
                        dtype=np.complex128,
                    )
                    if rec is None:
                        packed_identity_accum[key] = [
                            left_base,
                            scaled,
                            tuple(int(v) for v in dims),
                            int(in_start),
                            int(out_start),
                        ]
                    else:
                        rec[1] += scaled
                else:
                    key = (
                        "left",
                        id(right_stack),
                        tuple(int(v) for v in dims),
                        int(in_start),
                        int(out_start),
                        tuple(int(v) for v in left_base.shape),
                    )
                    rec = packed_identity_accum.get(key)
                    scaled = np.ascontiguousarray(
                        left_base * coeff,
                        dtype=np.complex128,
                    )
                    if rec is None:
                        packed_identity_accum[key] = [
                            scaled,
                            right_stack,
                            tuple(int(v) for v in dims),
                            int(in_start),
                            int(out_start),
                        ]
                    else:
                        rec[0] += scaled
                local_emitted += 1
                packed_identity_raw_kernels += 1

            for a_key, a_shape, left_qn, right_qn, p1_in, p2_in in a_entries:
                if len(a_shape) != 4:
                    return False
                nj_a, nk_a, nx, ny = (int(v) for v in a_shape)
                for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                    e_arr = np.asarray(e_blk)
                    if e_arr.ndim != 3:
                        return False
                    nb, ni, nj = (int(v) for v in e_arr.shape)
                    if int(nj) != int(nj_a):
                        return False
                    channel = e_key[0]
                    for f_key, f_blk in f_by_mpo_ket_r.get((channel, right_qn), ()):
                        f_arr = np.asarray(f_blk)
                        if f_arr.ndim != 3:
                            return False
                        nb_r, nl, nk = (int(v) for v in f_arr.shape)
                        if int(nb) != int(nb_r) or int(nk) != int(nk_a):
                            return False
                        out_key = (e_key[1], f_key[1], p1_in, p2_in)
                        out_shape = layout_shapes.get(out_key)
                        if out_shape is None:
                            continue
                        out_shape = tuple(int(v) for v in out_shape)
                        if out_shape != (int(ni), int(nl), int(nx), int(ny)):
                            return False
                        left_base = _identity_left_stack(e_blk, nx)
                        right_stack = _identity_right_stack(f_blk, ny)
                        if left_base is None or right_stack is None:
                            return False
                        in_start, in_size = offsets[a_key]
                        out_start, out_size = offsets[out_key]
                        if (
                            int(nj) * int(nx) * int(nk) * int(ny) != int(in_size)
                            or int(ni) * int(nl) * int(nx) * int(ny) != int(out_size)
                        ):
                            return False
                        dims_tuple = (
                            int(ni),
                            int(nl),
                            int(nx),
                            int(ny),
                            int(nj),
                            int(nx),
                            int(nk),
                            int(ny),
                        )
                        _accumulate(
                            left_base,
                            right_stack,
                            dims_tuple,
                            in_start,
                            out_start,
                        )
                        if (
                            route_plan is not None
                            and (
                                not self._packed_local_family_flat_group_identity_csr
                                or out.get("raw_builder") is not None
                            )
                        ):
                            try:
                                add_ref = getattr(
                                    route_plan,
                                    "add_packed_identity_ref_with_layout",
                                    None,
                                )
                                add_ref_with_layout = add_ref is not None
                                if add_ref is None:
                                    add_ref = getattr(
                                        route_plan,
                                        "add_packed_identity_ref",
                                        None,
                                    )
                                if add_ref is not None:
                                    args = (
                                        route_family_name,
                                        int(route_group_index),
                                        int(component_index),
                                        e_key,
                                        f_key,
                                        np.asarray(dims_tuple, dtype=np.int64),
                                        int(in_start),
                                        int(out_start),
                                        coeff,
                                    )
                                    if add_ref_with_layout:
                                        add_ref(*args, a_key, out_key)
                                    else:
                                        add_ref(*args)
                                else:
                                    route_plan.add_packed_identity(
                                        route_family_name,
                                        entry,
                                        e_key,
                                        f_key,
                                        np.asarray(dims_tuple, dtype=np.int64),
                                        int(in_start),
                                        int(out_start),
                                        coeff,
                                    )
                            except Exception:
                                pass
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["packed_identity_local_entries"] = int(
                stats.get("packed_identity_local_entries", 0)
            ) + 1
            stats["packed_identity_local_kernels"] = int(
                stats.get("packed_identity_local_kernels", 0)
            ) + int(local_emitted)
            return True

        def _collect_packed_local_generator(entry, component_index):
            nonlocal emitted, packed_local_generator_raw_kernels
            E = getattr(entry, "E", None)
            W_left = getattr(entry, "W_left", None)
            W_right = getattr(entry, "W_right", None)
            F = getattr(entry, "F", None)
            coeff = complex(getattr(entry, "coeff", 1.0))
            source = str(getattr(entry, "source", "packed_local_generator"))
            if E is None or W_left is None or W_right is None or F is None:
                return False
            e_by_ket_l = self._cached_block_index(E, (2,))
            w1_by_left_in = self._cached_block_index(W_left, (0, 3))
            w2_by_left_in = self._cached_block_index(W_right, (0, 3))
            f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))
            local_emitted = 0

            def _accumulate_generator(left_stack, right_stack, dims, in_start, out_start):
                nonlocal emitted, local_emitted, packed_local_generator_raw_kernels
                dims = tuple(int(v) for v in dims)
                packed_local_generator_raw_kernels += 1
                local_emitted += 1
                if (
                    not self._packed_local_family_flat_group_local_generator_csr
                    or out.get("raw_builder") is not None
                ):
                    if out.get("raw_builder") is not None:
                        self._append_direct_csr_kernel(
                            out,
                            left_stack,
                            right_stack,
                            dims,
                            in_start,
                            out_start,
                            coeff,
                        )
                    else:
                        self._append_direct_csr_kernel(
                            out,
                            self._scaled_direct_left_stack(
                                left_stack,
                                coeff,
                                source,
                            ),
                            right_stack,
                            dims,
                            in_start,
                            out_start,
                        )
                    emitted += 1
                    return
                if "right_boundary" in source or "same_side_right" in source:
                    key = (
                        "right",
                        id(left_stack),
                        dims,
                        int(in_start),
                        int(out_start),
                        tuple(int(v) for v in np.asarray(right_stack).shape),
                    )
                    scaled = np.ascontiguousarray(
                        np.asarray(right_stack, dtype=np.complex128) * coeff,
                        dtype=np.complex128,
                    )
                    rec = packed_local_generator_accum.get(key)
                    if rec is None:
                        packed_local_generator_accum[key] = [
                            left_stack,
                            scaled,
                            dims,
                            int(in_start),
                            int(out_start),
                        ]
                    else:
                        rec[1] += scaled
                else:
                    key = (
                        "left",
                        id(right_stack),
                        dims,
                        int(in_start),
                        int(out_start),
                        tuple(int(v) for v in np.asarray(left_stack).shape),
                    )
                    scaled = np.ascontiguousarray(
                        np.asarray(left_stack, dtype=np.complex128) * coeff,
                        dtype=np.complex128,
                    )
                    rec = packed_local_generator_accum.get(key)
                    if rec is None:
                        packed_local_generator_accum[key] = [
                            scaled,
                            right_stack,
                            dims,
                            int(in_start),
                            int(out_start),
                        ]
                    else:
                        rec[0] += scaled

            for a_key, a_shape, left_qn, right_qn, p1_in, p2_in in a_entries:
                for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                    if e_blk.ndim != 3:
                        return False
                    for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                        if w1_blk.ndim != 4:
                            return False
                        channel = w1_key[1]
                        left_op = _left_operator(e_blk, w1_blk)
                        for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                            if w2_blk.ndim != 4:
                                return False
                            for f_key, f_blk in f_by_mpo_ket_r.get(
                                (w2_key[1], right_qn),
                                (),
                            ):
                                if f_blk.ndim != 3:
                                    return False
                                right_op = _right_operator(w2_blk, f_blk)
                                if left_op.shape[0] != right_op.shape[0]:
                                    return False
                                out_key = (
                                    e_key[1],
                                    f_key[1],
                                    w1_key[2],
                                    w2_key[2],
                                )
                                out_shape = layout_shapes.get(out_key)
                                if out_shape is None:
                                    continue
                                kernel = self._direct_operator_matrix_kernel(
                                    left_op,
                                    a_shape,
                                    right_op,
                                )
                                if kernel is None:
                                    return False
                                left_stack, right_stack, out_shape_kernel, a_shape_kernel = kernel
                                in_start, in_size = offsets[a_key]
                                out_start, out_size = offsets[out_key]
                                if (
                                    left_stack.dtype != np.complex128
                                    or right_stack.dtype != np.complex128
                                    or tuple(out_shape_kernel) != tuple(out_shape)
                                    or int(np.prod(a_shape_kernel, dtype=int)) != int(in_size)
                                    or int(np.prod(out_shape_kernel, dtype=int)) != int(out_size)
                                ):
                                    return False
                                dims_tuple = (
                                    out_shape_kernel[0],
                                    out_shape_kernel[1],
                                    out_shape_kernel[2],
                                    out_shape_kernel[3],
                                    a_shape_kernel[0],
                                    a_shape_kernel[1],
                                    a_shape_kernel[2],
                                    a_shape_kernel[3],
                                )
                                _accumulate_generator(
                                    left_stack,
                                    right_stack,
                                    dims_tuple,
                                    in_start,
                                    out_start,
                                )
                                if (
                                    route_plan is not None
                                    and (
                                        not self._packed_local_family_flat_group_local_generator_csr
                                        or out.get("raw_builder") is not None
                                    )
                                ):
                                    try:
                                        add_ref = getattr(
                                            route_plan,
                                            "add_packed_local_generator_ref_with_layout",
                                            None,
                                        )
                                        add_ref_with_layout = add_ref is not None
                                        if add_ref is None:
                                            add_ref = getattr(
                                                route_plan,
                                                "add_packed_local_generator_ref",
                                                None,
                                            )
                                        if add_ref is not None:
                                            args = (
                                                route_family_name,
                                                int(route_group_index),
                                                int(component_index),
                                                e_key,
                                                w1_key,
                                                w2_key,
                                                f_key,
                                                np.asarray(dims_tuple, dtype=np.int64),
                                                int(in_start),
                                                int(out_start),
                                                coeff,
                                            )
                                            if add_ref_with_layout:
                                                add_ref(*args, a_key, out_key)
                                            else:
                                                add_ref(*args)
                                        else:
                                            route_plan.add_packed_local_generator(
                                                route_family_name,
                                                entry,
                                                e_key,
                                                w1_key,
                                                w2_key,
                                                f_key,
                                                np.asarray(dims_tuple, dtype=np.int64),
                                                int(in_start),
                                                int(out_start),
                                                coeff,
                                            )
                                    except Exception:
                                        pass
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["packed_local_generator_entries"] = int(
                stats.get("packed_local_generator_entries", 0)
            ) + 1
            stats["packed_local_generator_kernels"] = int(
                stats.get("packed_local_generator_kernels", 0)
            ) + int(local_emitted)
            return True

        if packed_count:
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["packed_identity_local_coalesced_entries"] = int(
                stats.get("packed_identity_local_coalesced_entries", 0)
            ) + int(packed_count)
            stats["packed_identity_local_coalesced_unique"] = int(
                stats.get("packed_identity_local_coalesced_unique", 0)
            ) + int(packed_unique)
            stats["packed_identity_local_coalesced_cancelled"] = int(
                stats.get("packed_identity_local_coalesced_cancelled", 0)
            ) + int(packed_cancelled)
        if packed_local_count:
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["packed_local_generator_coalesced_entries"] = int(
                stats.get("packed_local_generator_coalesced_entries", 0)
            ) + int(packed_local_count)
            stats["packed_local_generator_coalesced_unique"] = int(
                stats.get("packed_local_generator_coalesced_unique", 0)
            ) + int(packed_local_unique)
            stats["packed_local_generator_coalesced_cancelled"] = int(
                stats.get("packed_local_generator_coalesced_cancelled", 0)
            ) + int(packed_local_cancelled)

        for component_index, component in enumerate(component_entries):
            if isinstance(component, AbelianPackedIdentityLocalEntry):
                if not _collect_packed_identity(component, component_index):
                    return False
                continue
            if isinstance(component, AbelianPackedLocalGeneratorEntry):
                if not _collect_packed_local_generator(component, component_index):
                    return False
                continue
            E, W, F = component
            e_by_ket_l = self._cached_block_index(E, (2,))
            w1_by_left_in = self._cached_block_index(W[0], (0, 3))
            w2_by_left_in = self._cached_block_index(W[1], (0, 3))
            f_by_mpo_ket_r = self._cached_block_index(F, (0, 2))
            for a_key, a_shape, left_qn, right_qn, p1_in, p2_in in a_entries:
                for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                    if e_blk.ndim != 3:
                        return False
                    for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                        if w1_blk.ndim != 4:
                            return False
                        channel = w1_key[1]
                        left_op = _left_operator(e_blk, w1_blk)
                        for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                            if w2_blk.ndim != 4:
                                return False
                            for f_key, f_blk in f_by_mpo_ket_r.get(
                                (w2_key[1], right_qn),
                                (),
                            ):
                                if f_blk.ndim != 3:
                                    return False
                                right_op = _right_operator(w2_blk, f_blk)
                                if left_op.shape[0] != right_op.shape[0]:
                                    return False
                                out_key = (
                                    e_key[1],
                                    f_key[1],
                                    w1_key[2],
                                    w2_key[2],
                                )
                                out_shape = layout_shapes.get(out_key)
                                if out_shape is None:
                                    continue
                                kernel = self._direct_operator_matrix_kernel(
                                    left_op,
                                    a_shape,
                                    right_op,
                                )
                                if kernel is None:
                                    return False
                                left_stack, right_stack, out_shape_kernel, a_shape_kernel = kernel
                                in_start, in_size = offsets[a_key]
                                out_start, out_size = offsets[out_key]
                                if (
                                    left_stack.dtype != np.complex128
                                    or right_stack.dtype != np.complex128
                                    or tuple(out_shape_kernel) != tuple(out_shape)
                                    or int(np.prod(a_shape_kernel, dtype=int)) != int(in_size)
                                    or int(np.prod(out_shape_kernel, dtype=int)) != int(out_size)
                                ):
                                    return False
                                self._append_direct_csr_kernel(
                                    out,
                                    left_stack,
                                    right_stack,
                                    (
                                        out_shape_kernel[0],
                                        out_shape_kernel[1],
                                        out_shape_kernel[2],
                                        out_shape_kernel[3],
                                        a_shape_kernel[0],
                                        a_shape_kernel[1],
                                        a_shape_kernel[2],
                                        a_shape_kernel[3],
                                    ),
                                    in_start,
                                    out_start,
                                )
                                if route_plan is not None:
                                    try:
                                        dims_array = np.asarray(
                                            (
                                                out_shape_kernel[0],
                                                out_shape_kernel[1],
                                                out_shape_kernel[2],
                                                out_shape_kernel[3],
                                                a_shape_kernel[0],
                                                a_shape_kernel[1],
                                                a_shape_kernel[2],
                                                a_shape_kernel[3],
                                            ),
                                            dtype=np.int64,
                                        )
                                        add_ref = getattr(
                                            route_plan,
                                            "add_direct_component_ref_with_layout",
                                            None,
                                        )
                                        add_ref_with_layout = add_ref is not None
                                        if add_ref is None:
                                            add_ref = getattr(
                                                route_plan,
                                                "add_direct_component_ref",
                                                None,
                                            )
                                        if add_ref is not None:
                                            args = (
                                                route_family_name,
                                                int(route_group_index),
                                                int(component_index),
                                                e_key,
                                                w1_key,
                                                w2_key,
                                                f_key,
                                                dims_array,
                                                int(in_start),
                                                int(out_start),
                                                1.0 + 0.0j,
                                            )
                                            if add_ref_with_layout:
                                                add_ref(*args, a_key, out_key)
                                            else:
                                                add_ref(*args)
                                        else:
                                            route_plan.add_direct_component(
                                                route_family_name,
                                                component,
                                                e_key,
                                                w1_key,
                                                w2_key,
                                                f_key,
                                                dims_array,
                                                int(in_start),
                                                int(out_start),
                                                1.0 + 0.0j,
                                            )
                                    except Exception:
                                        route_plan = None
                                emitted += 1
        packed_identity_grouped_kernels = 0
        for (
            left_stack,
            right_stack,
            dims,
            in_start,
            out_start,
        ) in packed_identity_accum.values():
            self._append_direct_csr_kernel(
                out,
                left_stack,
                right_stack,
                dims,
                in_start,
                out_start,
            )
            emitted += 1
            packed_identity_grouped_kernels += 1
        packed_local_generator_grouped_kernels = 0
        for (
            left_stack,
            right_stack,
            dims,
            in_start,
            out_start,
        ) in packed_local_generator_accum.values():
            self._append_direct_csr_kernel(
                out,
                left_stack,
                right_stack,
                dims,
                in_start,
                out_start,
            )
            emitted += 1
            packed_local_generator_grouped_kernels += 1
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        if packed_identity_raw_kernels:
            effective_identity_grouped = (
                packed_identity_grouped_kernels
                if (
                    self._packed_local_family_flat_group_identity_csr
                    and out.get("raw_builder") is None
                )
                else packed_identity_raw_kernels
            )
            stats["packed_identity_local_raw_kernels"] = int(
                stats.get("packed_identity_local_raw_kernels", 0)
            ) + int(packed_identity_raw_kernels)
            stats["packed_identity_local_grouped_kernels"] = int(
                stats.get("packed_identity_local_grouped_kernels", 0)
            ) + int(effective_identity_grouped)
            stats["packed_identity_local_merged_kernels"] = int(
                stats.get("packed_identity_local_merged_kernels", 0)
            ) + int(packed_identity_raw_kernels - effective_identity_grouped)
        if packed_local_generator_raw_kernels:
            effective_local_generator_grouped = (
                packed_local_generator_grouped_kernels
                if (
                    self._packed_local_family_flat_group_local_generator_csr
                    and out.get("raw_builder") is None
                )
                else packed_local_generator_raw_kernels
            )
            stats["packed_local_generator_raw_kernels"] = int(
                stats.get("packed_local_generator_raw_kernels", 0)
            ) + int(packed_local_generator_raw_kernels)
            stats["packed_local_generator_grouped_kernels"] = int(
                stats.get("packed_local_generator_grouped_kernels", 0)
            ) + int(effective_local_generator_grouped)
            stats["packed_local_generator_merged_kernels"] = int(
                stats.get("packed_local_generator_merged_kernels", 0)
            ) + int(
                packed_local_generator_raw_kernels
                - effective_local_generator_grouped
            )
        stats["direct_csr_fast_collector_calls"] = int(
            stats.get("direct_csr_fast_collector_calls", 0)
        ) + 1
        stats["direct_csr_fast_collector_entries"] = int(
            stats.get("direct_csr_fast_collector_entries", 0)
        ) + int(emitted)
        stats["direct_csr_fast_collector_components"] = int(
            stats.get("direct_csr_fast_collector_components", 0)
        ) + int(len(component_entries))
        stats["direct_csr_fast_collector_packed_components"] = int(
            stats.get("direct_csr_fast_collector_packed_components", 0)
        ) + int(packed_unique + packed_local_unique)
        stats["direct_csr_fast_collector_a_entries"] = int(
            stats.get("direct_csr_fast_collector_a_entries", 0)
        ) + int(len(a_entries))
        elapsed = float(time.perf_counter() - build_start)
        stats["direct_csr_fast_collector_seconds"] = float(
            stats.get("direct_csr_fast_collector_seconds", 0.0)
        ) + elapsed
        stats["direct_csr_fast_collector_last_family"] = str(name)
        family_key = str(name).split(":", 1)[0]
        stats[f"direct_csr_fast_collector_seconds_{family_key}"] = float(
            stats.get(f"direct_csr_fast_collector_seconds_{family_key}", 0.0)
        ) + elapsed
        stats[f"direct_csr_fast_collector_entries_{family_key}"] = int(
            stats.get(f"direct_csr_fast_collector_entries_{family_key}", 0)
        ) + int(emitted)
        return True

    def _build_combined_direct_family_plan(
        self,
        A,
        name,
        component_entries,
        *,
        build_expr=True,
    ):
        if not build_expr:
            return self._build_combined_direct_family_plan_fast(
                A,
                name,
                component_entries,
            )
        layout = self._layout(A)
        cache_key = (
            layout,
            str(name),
            int(len(component_entries)),
            bool(build_expr),
            "combined_direct_family_plan",
        )
        cached = self._combined_direct_family_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        channels = set()
        entries = []
        out_shapes = {}
        out_qns = None
        out_dirs = None
        dtype_args = []
        for index, component in enumerate(component_entries):
            if isinstance(component, AbelianPackedLocalGeneratorEntry):
                E = component.E
                W = [
                    scale_abelian_boundary_tensor(
                        component.W_left,
                        component.coeff,
                        source="combined_direct_family_plan_scale",
                    ),
                    component.W_right,
                ]
                F = component.F
            else:
                E, W, F = component
            plan = self._build_direct_operator_plan(
                A,
                E,
                W,
                F,
                ("direct_symbolic_family", str(name), index),
                build_expr=build_expr,
            )
            if plan is None:
                self._combined_direct_family_plan_cache[cache_key] = None
                return None
            plan_channels, plan_entries, plan_shapes, plan_qns, plan_dirs, plan_dtype = plan
            channels.update(plan_channels)
            entries.extend(plan_entries)
            dtype_args.append(plan_dtype)
            if out_qns is None:
                out_qns = plan_qns
                out_dirs = plan_dirs
            for key, shape in plan_shapes.items():
                old_shape = out_shapes.get(key)
                if old_shape is not None and old_shape != shape:
                    self._combined_direct_family_plan_cache[cache_key] = None
                    return None
                out_shapes[key] = shape

        dtype = np.result_type(*(dtype_args or [complex]))
        combined = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            tuple(entries),
            out_shapes,
            out_qns if out_qns is not None else A.qns[:],
            out_dirs if out_dirs is not None else A.dirs[:],
            dtype,
        )
        self._combined_direct_family_plan_cache[cache_key] = combined
        return combined

    def _matvec_boundary_direct_operator(self, A, *, local_channels=None):
        plan = self._build_boundary_direct_operator_plan(A)
        applied = self._apply_direct_operator_plan(A, plan)
        if applied is None:
            return None
        total, channels, entries = applied
        local_channels = local_channels or {}
        for tensor in local_channels.values():
            total = total - tensor
        return {
            "total": total,
            "channels": {},
            "stats": {
                "kind": "abelian_complementary_boundary_direct_operator_action",
                "source": "precontracted_left_right_middle_channel_operators",
                "bond": self.bond,
                "n_entries": int(len(entries)),
                "n_mpo_middle_channels": int(len(channels)),
                "n_channels": int(len(channels) + len(local_channels)),
                "channel_names": tuple(
                    f"mpo_middle:{repr(channel)}" for channel in channels
                )
                + tuple(f"subtract_local:{name}" for name in local_channels),
                "channels_materialized": False,
            },
        }

    def _build_family_action_table_from_direct_family_environments(self, proto):
        if not self.complementary_direct_family_environments:
            return None
        cap = int(self._boundary_table_max_dim)
        layout = self._closed_layout(proto, cap)
        if layout is None:
            return None
        dim = self._size(layout)
        if dim <= 0 or dim > cap:
            return None
        H = np.zeros((dim, dim), dtype=complex)
        local_layout = AbelianLocalVectorLayout.from_layout(layout, proto=proto)
        qns = [list(axis_qns) for axis_qns in local_layout.qns]
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        for col in range(dim):
            basis = self._tensor_from_block_data_like(
                proto,
                local_layout.basis_data(col, dtype=dtype),
                qns,
                list(local_layout.dirs),
            )
            direct = self._matvec_direct_symbolic_family_channels(basis)
            if direct is None:
                return None
            family_channels, _family_stats = direct
            total = None
            for tensor in family_channels.values():
                total = tensor if total is None else total + tensor
            local = self._matvec_local_complementary(basis)
            if total is None or local is None:
                return None
            H[:, col] = self._flatten(total - local, layout)
        return AbelianComplementaryBoundaryActionTable(
            H,
            layout,
            qns,
            list(local_layout.dirs),
            bond=self.bond,
            source="direct_family_term_environments_minus_local_RP",
            boundary_family_tables=self._boundary_family_tables(),
        )

    def _append_local_complementary_matrix_coo(
        self,
        proto,
        layout,
        rows,
        cols,
        values,
        *,
        scale=-1.0,
        tol=1.0e-14,
    ):
        mat = self.local_complementary_matrix(proto)
        if mat is None:
            return False
        if len(getattr(proto, "qns", ())) < 4:
            return False
        p1_qns = tuple(proto.qns[2])
        p2_qns = tuple(proto.qns[3])
        pair_index = {
            (p1, p2): idx
            for idx, (p1, p2) in enumerate((q1, q2) for q1 in p1_qns for q2 in p2_qns)
        }
        n_pair = int(len(pair_index))
        mat = np.asarray(mat, dtype=np.complex128)
        if mat.shape != (n_pair, n_pair):
            return False
        offsets, _dim = self._layout_offsets(layout)
        layout_shapes = {key: tuple(shape) for key, shape in tuple(layout)}
        local_rows = []
        local_cols = []
        local_values = []
        for in_key, in_shape in tuple(layout):
            if len(in_key) < 4:
                return False
            ql, qr, p1_in, p2_in = in_key[:4]
            in_pair = pair_index.get((p1_in, p2_in))
            if in_pair is None:
                return False
            in_start, in_size = offsets[in_key]
            in_shape = tuple(in_shape)
            for (p1_out, p2_out), out_pair in pair_index.items():
                coeff = mat[out_pair, in_pair]
                if abs(coeff) <= float(tol):
                    continue
                out_key = (ql, qr, p1_out, p2_out)
                out_shape = layout_shapes.get(out_key)
                if out_shape is None:
                    return False
                out_start, out_size = offsets[out_key]
                if tuple(out_shape) != in_shape or int(out_size) != int(in_size):
                    return False
                local_rows.append(
                    np.arange(
                        int(out_start),
                        int(out_start) + int(out_size),
                        dtype=np.int64,
                    )
                )
                local_cols.append(
                    np.arange(
                        int(in_start),
                        int(in_start) + int(in_size),
                        dtype=np.int64,
                    )
                )
                local_values.append(
                    np.full(
                        int(in_size),
                        complex(scale) * complex(coeff),
                        dtype=np.complex128,
                    )
                )
        if local_values:
            rows.append(np.concatenate(local_rows))
            cols.append(np.concatenate(local_cols))
            values.append(np.concatenate(local_values))
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["direct_family_local_rp_subtractions"] = int(
            stats.get("direct_family_local_rp_subtractions", 0)
        ) + 1
        stats["direct_family_local_rp_nnz"] = int(
            stats.get("direct_family_local_rp_nnz", 0)
        ) + int(sum(chunk.size for chunk in local_values))
        return True

    @staticmethod
    def _project_flat_to_layout(tensor, layout):
        return abelian_project_tensor_to_layout(
            tensor,
            layout,
            extra_policy="zero",
            extra_zero_tol=1.0e-14,
        )

    def _direct_operator_entry_contribution(self, left_op, in_shape, right_op, local_col, dtype):
        block = np.zeros(tuple(in_shape), dtype=dtype)
        block.reshape(-1)[int(local_col)] = 1.0
        kernel = self._direct_operator_matrix_kernel(left_op, tuple(in_shape), right_op)
        if kernel is not None:
            return self._direct_operator_matrix_contribution(kernel, block)
        return np.einsum(
            "bijux,jkxy,bvylk->iluv",
            left_op,
            block,
            right_op,
            optimize=True,
        )

    def _accumulate_direct_operator_plan_coo(self, plan, layout, rows, cols, values):
        if plan is None:
            return False
        _channels, entries, out_shapes, _out_qns, _out_dirs, dtype = plan
        layout = tuple(layout)
        layout_shapes = {key: tuple(shape) for key, shape in layout}
        offsets, _dim = self._layout_offsets(layout)
        batch_left = []
        batch_right = []
        batch_dims = []
        batch_in_starts = []
        batch_out_starts = []
        batch_raw_capacity = 0

        def _flush_compiled_entry_batch():
            nonlocal batch_left, batch_right, batch_dims
            nonlocal batch_in_starts, batch_out_starts, batch_raw_capacity
            if not batch_left:
                return
            entry_rows, entry_cols, entry_values = _packed_cython.direct_operator_entries_coo(
                batch_left,
                batch_right,
                np.asarray(batch_dims, dtype=np.int64),
                np.asarray(batch_in_starts, dtype=np.int64),
                np.asarray(batch_out_starts, dtype=np.int64),
                1.0e-14,
            )
            rows.append(entry_rows)
            cols.append(entry_cols)
            values.append(entry_values)
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["compiled_entry_coo_batches"] = int(
                stats.get("compiled_entry_coo_batches", 0)
            ) + 1
            stats["compiled_entry_coo_calls"] = int(
                stats.get("compiled_entry_coo_calls", 0)
            ) + int(len(batch_left))
            stats["compiled_entry_coo_raw_nnz"] = int(
                stats.get("compiled_entry_coo_raw_nnz", 0)
            ) + int(entry_values.size)
            stats["compiled_entry_coo_raw_capacity"] = int(
                stats.get("compiled_entry_coo_raw_capacity", 0)
            ) + int(batch_raw_capacity)
            batch_left = []
            batch_right = []
            batch_dims = []
            batch_in_starts = []
            batch_out_starts = []
            batch_raw_capacity = 0

        for _channel, in_key, out_key, left_op, right_op, _expr in entries:
            in_shape = layout_shapes.get(in_key)
            out_shape = layout_shapes.get(out_key)
            if in_shape is None or out_shape is None:
                continue
            expected = tuple(out_shapes.get(out_key, out_shape))
            if tuple(out_shape) != expected:
                return False
            in_start, in_size = offsets[in_key]
            out_start, _out_size = offsets[out_key]
            kernel = self._direct_operator_matrix_kernel(left_op, in_shape, right_op)
            if (
                kernel is not None
                and _packed_cython is not None
                and getattr(_packed_cython, "CYTHON_AVAILABLE", False)
                and getattr(_packed_cython, "direct_operator_entry_coo", None) is not None
            ):
                left_stack, right_stack, out_shape_kernel, a_shape_kernel = kernel
                if (
                    left_stack.dtype == np.complex128
                    and right_stack.dtype == np.complex128
                    and tuple(out_shape_kernel) == tuple(out_shape)
                    and int(np.prod(a_shape_kernel, dtype=int)) == int(in_size)
                    and int(np.prod(out_shape_kernel, dtype=int)) == int(_out_size)
                ):
                    dims_tuple = (
                            out_shape_kernel[0],
                            out_shape_kernel[1],
                            out_shape_kernel[2],
                            out_shape_kernel[3],
                            a_shape_kernel[0],
                            a_shape_kernel[1],
                            a_shape_kernel[2],
                            a_shape_kernel[3],
                    )
                    if (
                        not self._packed_local_family_flat_sparse_entry_emitter
                        and getattr(_packed_cython, "direct_operator_entries_coo", None)
                        is not None
                    ):
                        batch_left.append(left_stack)
                        batch_right.append(right_stack)
                        batch_dims.append(dims_tuple)
                        batch_in_starts.append(int(in_start))
                        batch_out_starts.append(int(out_start))
                        batch_raw_capacity += int(in_size) * int(_out_size)
                        continue
                    dims = np.asarray(dims_tuple, dtype=np.int64)
                    sparse_entry = None
                    sparse_emitter = getattr(
                        _packed_cython,
                        "direct_operator_entry_sparse_product_coo",
                        None,
                    )
                    if (
                        self._packed_local_family_flat_sparse_entry_emitter
                        and sparse_emitter is not None
                    ):
                        sparse_entry = sparse_emitter(
                            left_stack,
                            right_stack,
                            dims,
                            int(in_start),
                            int(out_start),
                            1.0e-14,
                            int(in_size) * int(_out_size),
                        )
                    if sparse_entry is not None:
                        entry_rows, entry_cols, entry_values = sparse_entry
                        stats = self.profile_stats.setdefault(
                            "packed_flat_complementary_family_action",
                            {},
                        )
                        stats["compiled_sparse_entry_coo_calls"] = int(
                            stats.get("compiled_sparse_entry_coo_calls", 0)
                        ) + 1
                        stats["compiled_sparse_entry_coo_raw_nnz"] = int(
                            stats.get("compiled_sparse_entry_coo_raw_nnz", 0)
                        ) + int(entry_values.size)
                    else:
                        entry_rows, entry_cols, entry_values = (
                            _packed_cython.direct_operator_entry_coo(
                                left_stack,
                                right_stack,
                                dims,
                                int(in_start),
                                int(out_start),
                                1.0e-14,
                            )
                        )
                        stats = self.profile_stats.setdefault(
                            "packed_flat_complementary_family_action",
                            {},
                        )
                        stats["compiled_entry_coo_calls"] = int(
                            stats.get("compiled_entry_coo_calls", 0)
                        ) + 1
                        stats["compiled_entry_coo_raw_nnz"] = int(
                            stats.get("compiled_entry_coo_raw_nnz", 0)
                        ) + int(entry_values.size)
                    rows.append(entry_rows)
                    cols.append(entry_cols)
                    values.append(entry_values)
                    continue
            _flush_compiled_entry_batch()
            for local_col in range(int(in_size)):
                contribution = self._direct_operator_entry_contribution(
                    left_op,
                    in_shape,
                    right_op,
                    local_col,
                    dtype,
                )
                if tuple(contribution.shape) != tuple(out_shape):
                    return False
                flat = np.asarray(contribution).reshape(-1)
                nz = np.nonzero(np.abs(flat) > 1.0e-14)[0]
                rows.extend(int(out_start + row) for row in nz)
                cols.extend([int(in_start + local_col)] * int(len(nz)))
                values.extend(complex(flat[row]) for row in nz)
        _flush_compiled_entry_batch()
        return True

    def _collect_direct_operator_plan_csr_kernels(self, plan, layout, out):
        if plan is None:
            return False
        _channels, entries, out_shapes, _out_qns, _out_dirs, _dtype = plan
        layout = tuple(layout)
        layout_shapes = {key: tuple(shape) for key, shape in layout}
        offsets, _dim = self._layout_offsets(layout)
        for _channel, in_key, out_key, left_op, right_op, _expr in entries:
            in_shape = layout_shapes.get(in_key)
            out_shape = layout_shapes.get(out_key)
            if in_shape is None or out_shape is None:
                continue
            expected = tuple(out_shapes.get(out_key, out_shape))
            if tuple(out_shape) != expected:
                return False
            in_start, in_size = offsets[in_key]
            out_start, out_size = offsets[out_key]
            kernel = self._direct_operator_matrix_kernel(left_op, in_shape, right_op)
            if kernel is None:
                return False
            left_stack, right_stack, out_shape_kernel, a_shape_kernel = kernel
            if (
                left_stack.dtype != np.complex128
                or right_stack.dtype != np.complex128
                or tuple(out_shape_kernel) != tuple(out_shape)
                or int(np.prod(a_shape_kernel, dtype=int)) != int(in_size)
                or int(np.prod(out_shape_kernel, dtype=int)) != int(out_size)
            ):
                return False
            self._append_direct_csr_kernel(
                out,
                left_stack,
                right_stack,
                (
                    out_shape_kernel[0],
                    out_shape_kernel[1],
                    out_shape_kernel[2],
                    out_shape_kernel[3],
                    a_shape_kernel[0],
                    a_shape_kernel[1],
                    a_shape_kernel[2],
                    a_shape_kernel[3],
                ),
                in_start,
                out_start,
            )
        return True

    @staticmethod
    def _direct_operator_csr_pattern_key(layout, collected, dim, bond):
        return (
            "direct_operator_entries_csr_pattern",
            None if bond is None else int(bond),
            int(dim),
            tuple(layout),
            tuple(str(name) for name in collected.get("family_names", ())),
            tuple(tuple(int(v) for v in dims) for dims in collected["dims"]),
            tuple(int(v) for v in collected["in_starts"]),
            tuple(int(v) for v in collected["out_starts"]),
        )

    def _get_direct_operator_csr_pattern(self, key):
        cache = self._shared_flat_complementary_action_pattern_cache
        if cache is None or key is None:
            return None
        try:
            pattern = cache.get(key)
        except AttributeError:
            return None
        if pattern is not None and hasattr(cache, "move_to_end"):
            cache.move_to_end(key)
        return pattern

    def _put_direct_operator_csr_pattern(self, key, pattern):
        cache = self._shared_flat_complementary_action_pattern_cache
        if cache is None or key is None or pattern is None:
            return
        try:
            cache[key] = pattern
            if hasattr(cache, "move_to_end"):
                cache.move_to_end(key)
            cap = int(self._shared_flat_complementary_action_pattern_cache_max_entries)
            while cap > 0 and len(cache) > cap:
                if hasattr(cache, "popitem"):
                    try:
                        cache.popitem(last=False)
                    except TypeError:
                        first_key = next(iter(cache))
                        del cache[first_key]
                else:
                    break
        except Exception:
            return

    def _direct_operator_csr_lookup(self, indptr, indices, dim):
        if (
            _packed_cython is None
            or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            or getattr(_packed_cython, "csr_dense_lookup", None) is None
        ):
            return None
        max_dim = int(
            self._packed_local_family_flat_action_pattern_cache_max_lookup_dim
        )
        if max_dim <= 0 or int(dim) > max_dim:
            return None
        return _packed_cython.csr_dense_lookup(indptr, indices, int(dim))

    @staticmethod
    def _qns_from_block_data(data, rank):
        qns = []
        for axis in range(int(rank)):
            values = {key[axis] for key in data}
            qns.append(sorted(values, key=repr))
        return qns

    @staticmethod
    def _family_aux_label(family_index, axis_name, value):
        return ("__pyqed_family__", int(family_index), str(axis_name), repr(value))

    def _fused_named_family_environment(self):
        cached = self._fused_family_environment_cache
        if cached is not None:
            return cached
        if not self.complementary_family_environments:
            self._fused_family_environment_cache = None
            return None
        e_data = {}
        w1_data = {}
        w2_data = {}
        f_data = {}
        e_dirs = w1_dirs = w2_dirs = f_dirs = None
        n_families = 0
        for family_index, (name, env) in enumerate(
            sorted(self.complementary_family_environments.items(), key=lambda item: str(item[0]))
        ):
            try:
                E, W, F = env
            except Exception:
                continue
            if E is None or F is None or W is None or len(W) != 2:
                continue
            if e_dirs is None:
                e_dirs = E.dirs[:]
                w1_dirs = W[0].dirs[:]
                w2_dirs = W[1].dirs[:]
                f_dirs = F.dirs[:]
            elif (
                list(E.dirs) != list(e_dirs)
                or list(W[0].dirs) != list(w1_dirs)
                or list(W[1].dirs) != list(w2_dirs)
                or list(F.dirs) != list(f_dirs)
            ):
                self._fused_family_environment_cache = None
                return None
            for key, block in E.data.items():
                if len(key) != 3:
                    self._fused_family_environment_cache = None
                    return None
                new_key = (
                    self._family_aux_label(family_index, "L", key[0]),
                    key[1],
                    key[2],
                )
                e_data[new_key] = np.ascontiguousarray(block)
            for key, block in W[0].data.items():
                if len(key) != 4:
                    self._fused_family_environment_cache = None
                    return None
                new_key = (
                    self._family_aux_label(family_index, "L", key[0]),
                    self._family_aux_label(family_index, "M", key[1]),
                    key[2],
                    key[3],
                )
                w1_data[new_key] = np.ascontiguousarray(block)
            for key, block in W[1].data.items():
                if len(key) != 4:
                    self._fused_family_environment_cache = None
                    return None
                new_key = (
                    self._family_aux_label(family_index, "M", key[0]),
                    self._family_aux_label(family_index, "R", key[1]),
                    key[2],
                    key[3],
                )
                w2_data[new_key] = np.ascontiguousarray(block)
            for key, block in F.data.items():
                if len(key) != 3:
                    self._fused_family_environment_cache = None
                    return None
                new_key = (
                    self._family_aux_label(family_index, "R", key[0]),
                    key[1],
                    key[2],
                )
                f_data[new_key] = np.ascontiguousarray(block)
            n_families += 1
        if n_families <= 0 or not e_data or not w1_data or not w2_data or not f_data:
            self._fused_family_environment_cache = None
            return None
        fused = (
            AbelianEnvironmentTensorData(
                e_data,
                self._qns_from_block_data(e_data, 3),
                e_dirs,
            ),
            [
                AbelianSiteTensorData(
                    w1_data,
                    self._qns_from_block_data(w1_data, 4),
                    w1_dirs,
                    copy=False,
                ),
                AbelianSiteTensorData(
                    w2_data,
                    self._qns_from_block_data(w2_data, 4),
                    w2_dirs,
                    copy=False,
                ),
            ],
            AbelianEnvironmentTensorData(
                f_data,
                self._qns_from_block_data(f_data, 3),
                f_dirs,
            ),
            int(n_families),
        )
        self._fused_family_environment_cache = fused
        return fused

    def _flat_named_family_csr_kernels(self, proto, layout, *, build_groups=True):
        if not self.complementary_family_environments:
            return None
        layout = tuple(layout)
        cache_key = (
            "flat_named_family_csr_kernels",
            bool(build_groups),
            bool(not build_groups and self._use_cpp_raw_payload_builder()),
            layout,
            tuple(proto.dirs),
        )
        if cache_key in self._flat_named_family_kernel_cache:
            return self._flat_named_family_kernel_cache[cache_key]
        moving_environment = getattr(self, "_moving_environment", None)

        def _record_phase(name, elapsed):
            if moving_environment is None:
                return
            stats = moving_environment.moving_profile_stats
            key = str(name)
            stats[key] = float(stats.get(key, 0.0)) + float(elapsed)

        collected = self._new_direct_csr_collected(build_groups=build_groups)
        raw_route_plan = None
        raw_route_plan_complete = False
        if not bool(build_groups):
            raw_route_plan = self._new_cpp_raw_route_plan()
            raw_route_plan_complete = raw_route_plan is not None
        probe = self._zero_proto_from_layout(
            proto,
            layout,
            self._block_data_dtype(proto, complex),
        )
        named_loop_start = time.perf_counter()
        for name, env in self.complementary_family_environments.items():
            try:
                E, W, F = env
            except Exception:
                continue
            if E is None or F is None or W is None or len(W) != 2:
                continue
            before = self._direct_csr_kernel_count(collected)
            fast_start = time.perf_counter()
            fast_ok = self._collect_single_direct_operator_csr_kernels_fast(
                probe,
                E,
                W,
                F,
                layout,
                collected,
                route_plan=raw_route_plan if raw_route_plan_complete else None,
                route_family_name=name,
            )
            _record_phase(
                "renormalized_operator_payload_collect_named_fast_seconds",
                time.perf_counter() - fast_start,
            )
            if fast_ok:
                if self._direct_csr_kernel_count(collected) > before:
                    collected["family_names"].append(str(name))
                continue
            raw_route_plan_complete = False
            self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )["single_direct_csr_fast_collector_fallbacks"] = int(
                self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                ).get("single_direct_csr_fast_collector_fallbacks", 0)
            ) + 1
            plan_start = time.perf_counter()
            plan = self._build_direct_operator_plan(
                probe,
                E,
                W,
                F,
                ("flat_named_family", str(name)),
                build_expr=False,
            )
            _record_phase(
                "renormalized_operator_payload_collect_named_plan_seconds",
                time.perf_counter() - plan_start,
            )
            kernel_start = time.perf_counter()
            if not self._collect_direct_operator_plan_csr_kernels(
                plan,
                layout,
                collected,
            ):
                self._flat_named_family_kernel_cache[cache_key] = None
                return None
            _record_phase(
                "renormalized_operator_payload_collect_named_kernel_seconds",
                time.perf_counter() - kernel_start,
            )
            if self._direct_csr_kernel_count(collected) > before:
                collected["family_names"].append(str(name))
        _record_phase(
            "renormalized_operator_payload_collect_named_loop_seconds",
            time.perf_counter() - named_loop_start,
        )
        if self._direct_csr_kernel_count(collected) <= 0:
            self._flat_named_family_kernel_cache[cache_key] = None
            return None
        collected = self._finalize_direct_csr_collected(collected)
        if raw_route_plan_complete and raw_route_plan is not None:
            try:
                route_size = int(raw_route_plan.size())
            except Exception:
                route_size = -1
            if route_size == self._direct_csr_kernel_count(collected):
                collected["raw_route_plan"] = raw_route_plan
        if (
            build_groups
            and
            self._packed_local_family_flat_direct_matvec
            and self._packed_local_family_flat_direct_matvec_backend
            in {"grouped_blas", "grouped_compiled", "renormalized_table"}
        ):
            named_group_start = time.perf_counter()
            grouped = OrderedDict()
            scales = collected.get("scales", ())
            for left_stack, right_stack, dims, in_start, out_start, scale in zip(
                collected["left"],
                collected["right"],
                collected["dims"],
                collected["in_starts"],
                collected["out_starts"],
                scales,
            ):
                key = (
                    tuple(int(v) for v in dims),
                    int(in_start),
                    int(out_start),
                )
                entry = grouped.setdefault(key, {"left": [], "right": [], "scales": []})
                entry["left"].append(left_stack)
                entry["right"].append(right_stack)
                entry["scales"].append(complex(scale))
            matvec_groups = []
            for (dims, in_start, out_start), entry in grouped.items():
                try:
                    left_group = np.ascontiguousarray(
                        np.concatenate(entry["left"], axis=0),
                        dtype=np.complex128,
                    )
                    right_group = np.ascontiguousarray(
                        np.concatenate(entry["right"], axis=0),
                        dtype=np.complex128,
                    )
                    if all(
                        complex(scale) == (1.0 + 0.0j)
                        for scale in entry["scales"]
                    ):
                        scale_group = None
                    else:
                        scale_group = np.ascontiguousarray(
                            np.concatenate(
                                [
                                    np.full(
                                        np.asarray(left_stack).shape[0],
                                        scale,
                                        dtype=np.complex128,
                                    )
                                    for left_stack, scale in zip(
                                        entry["left"],
                                        entry["scales"],
                                    )
                                ],
                                axis=0,
                            ),
                            dtype=np.complex128,
                        )
                except ValueError:
                    matvec_groups = None
                    break
                matvec_groups.append(
                    {
                        "dims": dims,
                        "in_start": int(in_start),
                        "out_start": int(out_start),
                        "left": left_group,
                        "right": right_group,
                        "scales": scale_group,
                        "entries": int(len(entry["left"])),
                        "channels": int(left_group.shape[0]),
                    }
                )
            collected["matvec_groups"] = matvec_groups
            if matvec_groups is not None:
                collected["group_left"] = [group["left"] for group in matvec_groups]
                collected["group_right"] = [group["right"] for group in matvec_groups]
                collected["group_dims_array"] = np.asarray(
                    [group["dims"] for group in matvec_groups],
                    dtype=np.int64,
                )
                collected["group_in_starts_array"] = np.asarray(
                    [group["in_start"] for group in matvec_groups],
                    dtype=np.int64,
                )
                collected["group_out_starts_array"] = np.asarray(
                    [group["out_start"] for group in matvec_groups],
                    dtype=np.int64,
                )
                group_scales = [group["scales"] for group in matvec_groups]
                collected["group_scales"] = (
                    group_scales
                    if any(scale is not None for scale in group_scales)
                    else None
                )
            _record_phase(
                "renormalized_operator_payload_collect_named_group_seconds",
                time.perf_counter() - named_group_start,
            )
        else:
            collected["matvec_groups"] = None
        self._flat_named_family_kernel_cache[cache_key] = collected
        return collected

    def _flat_direct_family_csr_kernels(self, proto, layout, *, build_groups=True):
        if not self.complementary_direct_family_environments:
            return None
        layout = tuple(layout)
        cache_key = (
            "flat_direct_family_csr_kernels",
            bool(build_groups),
            bool(not build_groups and self._use_cpp_raw_payload_builder()),
            layout,
            tuple(proto.dirs),
        )
        if cache_key in self._flat_direct_family_kernel_cache:
            return self._flat_direct_family_kernel_cache[cache_key]
        collected = self._new_direct_csr_collected(build_groups=build_groups)
        raw_route_plan = None
        raw_route_plan_complete = False
        if not bool(build_groups):
            raw_route_plan = self._new_cpp_raw_route_plan()
            raw_route_plan_complete = raw_route_plan is not None
        probe = self._zero_proto_from_layout(
            proto,
            layout,
            self._block_data_dtype(proto, complex),
        )

        def _collect_scratch(group_name, group_entries, group_index=-1):
            nonlocal raw_route_plan_complete
            scratch = self._new_direct_csr_collected(build_groups=build_groups)
            ok = self._collect_combined_direct_family_csr_kernels_fast(
                probe,
                group_name,
                group_entries,
                layout,
                scratch,
                route_plan=raw_route_plan if raw_route_plan_complete else None,
                route_family_name=str(name),
                route_group_index=int(group_index),
            )
            if not ok:
                raw_route_plan_complete = False
                scratch = self._new_direct_csr_collected(build_groups=build_groups)
                plan = self._build_combined_direct_family_plan(
                    probe,
                    group_name,
                    group_entries,
                    build_expr=False,
                )
                ok = self._collect_direct_operator_plan_csr_kernels(
                    plan,
                    layout,
                    scratch,
                )
            return scratch if ok else None

        def _append_scratch(scratch):
            if not scratch or self._direct_csr_kernel_count(scratch) <= 0:
                return
            target_builder = collected.get("raw_builder")
            scratch_builder = scratch.get("raw_builder")
            if target_builder is not None and scratch_builder is not None:
                target_builder.extend(scratch_builder)
                collected["entry_count"] = int(
                    collected.get("entry_count", 0)
                ) + int(scratch_builder.size())
                return
            if target_builder is None and scratch_builder is not None:
                scratch = self._materialize_raw_builder_collected(scratch)
            for key in (
                "left",
                "right",
                "dims",
                "in_starts",
                "out_starts",
                "scales",
            ):
                collected[key].extend(scratch.get(key, ()))
            collected["entry_count"] = int(collected.get("entry_count", 0)) + int(
                scratch.get("entry_count", len(scratch.get("left", ())))
            )

        for name, entries in self.complementary_direct_family_environments.items():
            family_route_count = -1
            if raw_route_plan is not None:
                try:
                    family_route_count = int(raw_route_plan.size())
                except Exception:
                    family_route_count = -1
            before = self._direct_csr_kernel_count(collected)
            entry_groups = tuple(getattr(entries, "entry_groups", ()) or ())
            try:
                if len(entry_groups) >= int(len(entries)):
                    entry_groups = ()
            except TypeError:
                entry_groups = ()
            if entry_groups and not self._packed_local_family_flat_collect_entry_groups:
                scratch = _collect_scratch(str(name), entries, -1)
                if scratch is not None:
                    _append_scratch(scratch)
                    if (
                        raw_route_plan_complete
                        and raw_route_plan is not None
                        and family_route_count >= 0
                    ):
                        try:
                            if int(raw_route_plan.size()) - family_route_count != self._direct_csr_kernel_count(scratch):
                                raw_route_plan_complete = False
                        except Exception:
                            raw_route_plan_complete = False
                    stats = self.profile_stats.setdefault(
                        "packed_flat_complementary_family_action",
                        {},
                    )
                    stats["direct_csr_whole_family_collector_hits"] = int(
                        stats.get("direct_csr_whole_family_collector_hits", 0)
                    ) + 1
                    if self._direct_csr_kernel_count(collected) > before:
                        collected["family_names"].append(str(name))
                    continue
                stats = self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                )
                stats["direct_csr_whole_family_collector_fallbacks"] = int(
                    stats.get("direct_csr_whole_family_collector_fallbacks", 0)
                ) + 1
                raw_route_plan_complete = False
            group_keys = tuple(getattr(entries, "group_keys", ()) or ())
            if entry_groups:
                for group_index, group_entries in enumerate(entry_groups):
                    group_name = (
                        f"{name}:group:{group_index}:"
                        f"{repr(group_keys[group_index]) if group_index < len(group_keys) else '?'}"
                    )
                    group_route_count = -1
                    if raw_route_plan is not None:
                        try:
                            group_route_count = int(raw_route_plan.size())
                        except Exception:
                            group_route_count = -1
                    scratch = _collect_scratch(group_name, group_entries, group_index)
                    if scratch is None:
                        self._flat_direct_family_kernel_cache[cache_key] = None
                        return None
                    _append_scratch(scratch)
                    if (
                        raw_route_plan_complete
                        and raw_route_plan is not None
                        and group_route_count >= 0
                    ):
                        try:
                            if int(raw_route_plan.size()) - group_route_count != self._direct_csr_kernel_count(scratch):
                                raw_route_plan_complete = False
                        except Exception:
                            raw_route_plan_complete = False
            else:
                group_route_count = -1
                if raw_route_plan is not None:
                    try:
                        group_route_count = int(raw_route_plan.size())
                    except Exception:
                        group_route_count = -1
                scratch = _collect_scratch(name, entries, -1)
                if scratch is None:
                    self._flat_direct_family_kernel_cache[cache_key] = None
                    return None
                _append_scratch(scratch)
                if (
                    raw_route_plan_complete
                    and raw_route_plan is not None
                    and group_route_count >= 0
                ):
                    try:
                        if int(raw_route_plan.size()) - group_route_count != self._direct_csr_kernel_count(scratch):
                            raw_route_plan_complete = False
                    except Exception:
                        raw_route_plan_complete = False
            if self._direct_csr_kernel_count(collected) > before:
                collected["family_names"].append(str(name))
        if self._direct_csr_kernel_count(collected) <= 0:
            self._flat_direct_family_kernel_cache[cache_key] = None
            return None
        collected = self._finalize_direct_csr_collected(collected)
        if raw_route_plan_complete and raw_route_plan is not None:
            try:
                route_size = int(raw_route_plan.size())
            except Exception:
                route_size = -1
            if route_size == self._direct_csr_kernel_count(collected):
                collected["raw_route_plan"] = raw_route_plan
        matvec_groups = (
            self._group_direct_operator_collected(collected)
            if build_groups
            else None
        )
        collected["matvec_groups"] = matvec_groups
        if matvec_groups is not None:
            collected["group_left"] = [group["left"] for group in matvec_groups]
            collected["group_right"] = [group["right"] for group in matvec_groups]
            group_scales = [group["scales"] for group in matvec_groups]
            collected["group_scales"] = (
                group_scales
                if any(scale is not None for scale in group_scales)
                else None
            )
            collected["group_dims_array"] = np.asarray(
                [group["dims"] for group in matvec_groups],
                dtype=np.int64,
            )
            collected["group_in_starts_array"] = np.asarray(
                [group["in_start"] for group in matvec_groups],
                dtype=np.int64,
            )
            collected["group_out_starts_array"] = np.asarray(
                [group["out_start"] for group in matvec_groups],
                dtype=np.int64,
            )
        self._flat_direct_family_kernel_cache[cache_key] = collected
        return collected

    @staticmethod
    def _group_direct_operator_collected(collected):
        grouped = OrderedDict()
        scales = collected.get("scales")
        if scales is None or len(scales) != len(collected["left"]):
            scales = [1.0 + 0.0j] * len(collected["left"])
        for left_stack, right_stack, dims, in_start, out_start, scale in zip(
            collected["left"],
            collected["right"],
            collected["dims"],
            collected["in_starts"],
            collected["out_starts"],
            scales,
        ):
            key = (
                tuple(int(v) for v in dims),
                int(in_start),
                int(out_start),
            )
            entry = grouped.setdefault(key, {"left": [], "right": [], "scales": []})
            entry["left"].append(left_stack)
            entry["right"].append(right_stack)
            entry["scales"].append(complex(scale))
        matvec_groups = []
        for (dims, in_start, out_start), entry in grouped.items():
            try:
                left_group = np.ascontiguousarray(
                    np.concatenate(entry["left"], axis=0),
                    dtype=np.complex128,
                )
                right_group = np.ascontiguousarray(
                    np.concatenate(entry["right"], axis=0),
                    dtype=np.complex128,
                )
                if all(
                    complex(scale) == (1.0 + 0.0j)
                    for scale in entry["scales"]
                ):
                    scale_group = None
                else:
                    scale_group = np.ascontiguousarray(
                        np.concatenate(
                            [
                                np.full(
                                    np.asarray(left_stack).shape[0],
                                    scale,
                                    dtype=np.complex128,
                                )
                                for left_stack, scale in zip(
                                    entry["left"],
                                    entry["scales"],
                                )
                            ],
                            axis=0,
                        ),
                        dtype=np.complex128,
                    )
            except ValueError:
                return None
            matvec_groups.append(
                {
                    "dims": dims,
                    "in_start": int(in_start),
                    "out_start": int(out_start),
                    "left": left_group,
                    "right": right_group,
                    "scales": scale_group,
                    "entries": int(len(entry["left"])),
                    "channels": int(left_group.shape[0]),
                }
            )
        return matvec_groups

    def _flat_generator_family_csr_kernels(self, proto, layout, *, build_groups=True):
        if not self.complementary_direct_family_environments:
            return None
        layout = tuple(layout)
        cache_key = (
            "flat_generator_family_csr_kernels",
            bool(build_groups),
            bool(not build_groups and self._use_cpp_raw_payload_builder()),
            layout,
            tuple(proto.dirs),
        )
        if cache_key in self._flat_generator_family_kernel_cache:
            return self._flat_generator_family_kernel_cache[cache_key]
        moving_environment = getattr(self, "_moving_environment", None)

        def _record_phase(name, elapsed):
            if moving_environment is None:
                return
            stats = moving_environment.moving_profile_stats
            key = str(name)
            stats[key] = float(stats.get(key, 0.0)) + float(elapsed)

        direct_start = time.perf_counter()
        direct = self._flat_direct_family_csr_kernels(
            proto,
            layout,
            build_groups=False,
        )
        _record_phase(
            "renormalized_operator_payload_collect_direct_seconds",
            time.perf_counter() - direct_start,
        )
        if direct is None:
            self._flat_generator_family_kernel_cache[cache_key] = None
            return None
        if not self.complementary_family_environments:
            self._flat_generator_family_kernel_cache[cache_key] = direct
            return direct
        collected = self._new_direct_csr_collected(build_groups=build_groups)
        raw_route_plan = None
        raw_route_plan_complete = False
        if not bool(build_groups):
            raw_route_plan = self._new_cpp_raw_route_plan()
            raw_route_plan_complete = raw_route_plan is not None

        def _merge_collected(source):
            nonlocal raw_route_plan_complete
            if source is None:
                return
            if raw_route_plan_complete and raw_route_plan is not None:
                source_route_plan = source.get("raw_route_plan")
                if source_route_plan is None:
                    raw_route_plan_complete = False
                else:
                    try:
                        raw_route_plan.extend(source_route_plan)
                    except Exception:
                        raw_route_plan_complete = False
            target_builder = collected.get("raw_builder")
            source_builder = source.get("raw_builder")
            if target_builder is not None and source_builder is not None:
                target_builder.extend(source_builder)
                collected["entry_count"] = int(
                    collected.get("entry_count", 0)
                ) + int(source_builder.size())
            else:
                if source_builder is not None:
                    source = self._materialize_raw_builder_collected(source)
                for key in (
                    "left",
                    "right",
                    "dims",
                    "in_starts",
                    "out_starts",
                    "scales",
                ):
                    collected[key].extend(source.get(key, ()))
                collected["entry_count"] = int(
                    collected.get("entry_count", 0)
                ) + int(source.get("entry_count", len(source.get("left", ()))))
            collected["family_names"].extend(source.get("family_names", ()))

        _merge_collected(direct)
        if self.complementary_family_environments:
            named_start = time.perf_counter()
            named = self._flat_named_family_csr_kernels(
                proto,
                layout,
                build_groups=False,
            )
            _record_phase(
                "renormalized_operator_payload_collect_named_seconds",
                time.perf_counter() - named_start,
            )
            if named is not None:
                _merge_collected(named)
        if self._direct_csr_kernel_count(collected) <= 0:
            self._flat_generator_family_kernel_cache[cache_key] = None
            return None
        collected = self._finalize_direct_csr_collected(collected)
        if raw_route_plan_complete and raw_route_plan is not None:
            try:
                route_size = int(raw_route_plan.size())
            except Exception:
                route_size = -1
            if route_size == self._direct_csr_kernel_count(collected):
                collected["raw_route_plan"] = raw_route_plan
        if build_groups:
            group_start = time.perf_counter()
            groups = self._group_direct_operator_collected(collected)
            _record_phase(
                "renormalized_operator_payload_collect_group_seconds",
                time.perf_counter() - group_start,
            )
            collected["matvec_groups"] = groups
            if groups is not None:
                collected["group_left"] = [group["left"] for group in groups]
                collected["group_right"] = [group["right"] for group in groups]
                group_scales = [group["scales"] for group in groups]
                collected["group_scales"] = (
                    group_scales
                    if any(scale is not None for scale in group_scales)
                    else None
                )
                collected["group_dims_array"] = np.asarray(
                    [group["dims"] for group in groups],
                    dtype=np.int64,
                )
                collected["group_in_starts_array"] = np.asarray(
                    [group["in_start"] for group in groups],
                    dtype=np.int64,
                )
                collected["group_out_starts_array"] = np.asarray(
                    [group["out_start"] for group in groups],
                    dtype=np.int64,
                )
        else:
            collected["matvec_groups"] = None
        self._flat_generator_family_kernel_cache[cache_key] = collected
        return collected

    def _direct_family_plan_action_table_direct_csr(
        self,
        proto,
        layout,
        *,
        subtract_local=True,
    ):
        if (
            _packed_cython is None
            or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            or getattr(_packed_cython, "direct_operator_entries_csr", None) is None
            or self._packed_local_family_flat_sparse_entry_emitter
        ):
            return None
        dim = int(self._size(layout))
        cap = int(self._packed_local_family_flat_direct_csr_build_max_dim)
        if dim <= 0 or cap <= 0 or dim > cap:
            return None
        collected = self._flat_direct_family_csr_kernels(proto, layout)
        if collected is None:
            return None
        csr_builder = _packed_cython.direct_operator_entries_csr
        if (
            self._packed_local_family_flat_direct_csr_extract_backend
            in {"numpy", "np", "vectorized"}
            and getattr(_packed_cython, "direct_operator_entries_csr_np_extract", None)
            is not None
        ):
            csr_builder = _packed_cython.direct_operator_entries_csr_np_extract
        indptr, indices, csr_values, raw_nnz = csr_builder(
            collected["left"],
            collected["right"],
            collected["dims_array"],
            collected["in_starts_array"],
            collected["out_starts_array"],
            int(dim),
            1.0e-14,
            collected.get("scales_array"),
        )
        rows = [
            np.repeat(np.arange(int(dim), dtype=np.int64), np.diff(indptr)),
        ]
        cols = [np.asarray(indices, dtype=np.int64)]
        values = [np.asarray(csr_values, dtype=np.complex128)]
        if subtract_local:
            if not self._append_local_complementary_matrix_coo(
                proto,
                layout,
                rows,
                cols,
                values,
                scale=-1.0,
            ):
                return None
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["direct_family_compiled_csr_builds"] = int(
            stats.get("direct_family_compiled_csr_builds", 0)
        ) + 1
        stats["direct_family_compiled_csr_extract_backend"] = str(
            self._packed_local_family_flat_direct_csr_extract_backend
        )
        stats["direct_family_compiled_csr_entries"] = int(
            stats.get("direct_family_compiled_csr_entries", 0)
        ) + int(self._direct_csr_kernel_count(collected))
        stats["direct_family_compiled_csr_raw_nnz"] = int(
            stats.get("direct_family_compiled_csr_raw_nnz", 0)
        ) + int(raw_nnz)
        return AbelianSparseComplementaryBoundaryActionTable(
            rows,
            cols,
            values,
            dim,
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source=(
                "sparse_flat_direct_family_plan_entries_direct_csr_minus_local_RP"
                if subtract_local
                else "sparse_flat_direct_family_plan_entries_direct_csr"
            ),
            boundary_family_tables=self._boundary_family_tables(),
        )

    def _direct_family_plan_action_table(self, proto, layout, *, subtract_local=True):
        if not self.complementary_direct_family_environments:
            return None
        direct_csr = self._direct_family_plan_action_table_direct_csr(
            proto,
            layout,
            subtract_local=subtract_local,
        )
        if direct_csr is not None:
            return direct_csr
        rows = []
        cols = []
        values = []
        probe = self._zero_proto_from_layout(
            proto,
            layout,
            self._block_data_dtype(proto, complex),
        )
        for name, entries in self.complementary_direct_family_environments.items():
            entry_groups = tuple(getattr(entries, "entry_groups", ()) or ())
            try:
                if len(entry_groups) >= int(len(entries)):
                    entry_groups = ()
            except TypeError:
                entry_groups = ()
            group_keys = tuple(getattr(entries, "group_keys", ()) or ())
            groups = entry_groups or (entries,)
            for group_index, group_entries in enumerate(groups):
                group_name = str(name)
                if entry_groups:
                    group_name = (
                        f"{name}:group:{group_index}:"
                        f"{repr(group_keys[group_index]) if group_index < len(group_keys) else '?'}"
                    )
                plan = self._build_combined_direct_family_plan(
                    probe,
                    group_name,
                    group_entries,
                    build_expr=False,
                )
                if not self._accumulate_direct_operator_plan_coo(
                    plan,
                    layout,
                    rows,
                    cols,
                    values,
                ):
                    return None
        if not rows and not values:
            return None
        if subtract_local:
            if not self._append_local_complementary_matrix_coo(
                probe,
                layout,
                rows,
                cols,
                values,
                scale=-1.0,
            ):
                return None
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["direct_family_coo_builds"] = int(
            stats.get("direct_family_coo_builds", 0)
        ) + 1
        return AbelianSparseComplementaryBoundaryActionTable(
            rows,
            cols,
            values,
            self._size(layout),
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source=(
                "sparse_flat_direct_family_plan_entries_minus_local_RP"
                if subtract_local
                else "sparse_flat_direct_family_plan_entries"
            ),
            boundary_family_tables=self._boundary_family_tables(),
        )

    def _flat_generator_family_plan_action_table(self, proto, layout):
        if not self.complementary_direct_family_environments:
            return None
        rows = []
        cols = []
        values = []
        direct = self._direct_family_plan_action_table(
            proto,
            layout,
            subtract_local=False,
        )
        if direct is None:
            return None
        rows.append(direct.rows)
        cols.append(direct.cols)
        values.append(direct.values)
        named = None
        if self.complementary_family_environments:
            named = self._flat_named_family_plan_action_table(proto, layout)
            if named is None:
                return None
            rows.append(named.rows)
            cols.append(named.cols)
            values.append(named.values)
        table = AbelianSparseComplementaryBoundaryActionTable(
            rows,
            cols,
            values,
            self._size(layout),
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source=(
                "sparse_flat_generator_direct_and_named_family_plan_entries"
                if named is not None
                else "sparse_flat_generator_direct_family_plan_entries"
            ),
            boundary_family_tables=self._boundary_family_tables(),
        )
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["generator_family_plan_table_builds"] = int(
            stats.get("generator_family_plan_table_builds", 0)
        ) + 1
        stats["generator_family_plan_table_direct_nnz"] = int(
            getattr(direct, "nnz", 0)
        )
        stats["generator_family_plan_table_named_nnz"] = int(
            0 if named is None else getattr(named, "nnz", 0)
        )
        stats["generator_family_plan_table_nnz"] = int(table.nnz)
        return table

    def _build_direct_family_action_table_from_plans(self, proto):
        if not self.complementary_direct_family_environments:
            return None
        cap = int(self._boundary_table_max_dim)
        layout = self._closed_layout(proto, cap)
        if layout is None:
            return None
        dim = self._size(layout)
        if dim <= 0 or dim > cap:
            return None
        probe = self._zero_proto_from_layout(
            proto,
            layout,
            self._block_data_dtype(proto, complex),
        )
        return self._direct_family_plan_action_table(probe, layout)

    def _flat_named_family_renormalized_operator_table(self, proto, layout):
        moving_environment = getattr(self, "_moving_environment", None)
        if moving_environment is not None:
            return moving_environment.renormalized_operator_table(self, proto, layout)

        layout = tuple(layout)
        cache_key = (
            "flat_named_family_renormalized_operator_table",
            layout,
            tuple(proto.dirs),
        )
        cached = self._flat_renormalized_operator_table_cache.get(cache_key)
        if cached is not None:
            return cached
        collected = self._flat_named_family_csr_kernels(proto, layout)
        if collected is None:
            self._flat_renormalized_operator_table_cache[cache_key] = None
            return None
        table = AbelianRenormalizedOperatorActionTable(
            collected,
            self._size(layout),
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source="flat_named_family_renormalized_operator_table",
            boundary_family_tables=self._boundary_family_tables(),
            max_dense_block_elements=(
                self._renormalized_operator_table_dense_block_max_elements
            ),
            sparse_density_threshold=(
                self._renormalized_operator_table_sparse_density_threshold
            ),
        )
        self._flat_renormalized_operator_table_cache[cache_key] = table
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["renormalized_operator_table_builds"] = int(
            stats.get("renormalized_operator_table_builds", 0)
        ) + 1
        stats["renormalized_operator_table_entries"] = int(
            stats.get("renormalized_operator_table_entries", 0)
        ) + int(table.n_entries)
        stats["renormalized_operator_table_groups"] = int(
            stats.get("renormalized_operator_table_groups", 0)
        ) + int(table.n_groups)
        stats["renormalized_operator_table_group_channels"] = int(
            stats.get("renormalized_operator_table_group_channels", 0)
        ) + int(table.n_group_channels)
        stats["renormalized_operator_table_block_matrices"] = int(
            stats.get("renormalized_operator_table_block_matrices", 0)
        ) + int(table.n_block_matrices)
        stats["renormalized_operator_table_block_matrix_elements"] = int(
            stats.get("renormalized_operator_table_block_matrix_elements", 0)
        ) + int(table.block_matrix_elements)
        stats["renormalized_operator_table_block_sparse_nnz"] = int(
            stats.get("renormalized_operator_table_block_sparse_nnz", 0)
        ) + int(table.block_sparse_nnz)
        stats["renormalized_operator_table_last_storage"] = str(table.storage)
        return table

    def _flat_direct_family_renormalized_operator_table(self, proto, layout):
        if not self.complementary_direct_family_environments:
            return None
        layout = tuple(layout)
        cache_key = (
            "flat_direct_family_renormalized_operator_table",
            layout,
            tuple(proto.dirs),
        )
        cached = self._flat_direct_renormalized_operator_table_cache.get(cache_key)
        if cached is not None:
            return cached
        collected = self._flat_generator_family_csr_kernels(proto, layout)
        if collected is None:
            self._flat_direct_renormalized_operator_table_cache[cache_key] = None
            return None
        table = AbelianRenormalizedOperatorActionTable(
            collected,
            self._size(layout),
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source="flat_direct_generator_family_renormalized_operator_table",
            boundary_family_tables=self._boundary_family_tables(),
            max_dense_block_elements=(
                self._renormalized_operator_table_dense_block_max_elements
            ),
            sparse_density_threshold=(
                self._renormalized_operator_table_sparse_density_threshold
            ),
        )
        self._flat_direct_renormalized_operator_table_cache[cache_key] = table
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["direct_renormalized_operator_table_builds"] = int(
            stats.get("direct_renormalized_operator_table_builds", 0)
        ) + 1
        stats["direct_renormalized_operator_table_entries"] = int(
            stats.get("direct_renormalized_operator_table_entries", 0)
        ) + int(table.n_entries)
        stats["direct_renormalized_operator_table_groups"] = int(
            stats.get("direct_renormalized_operator_table_groups", 0)
        ) + int(table.n_groups)
        stats["direct_renormalized_operator_table_group_channels"] = int(
            stats.get("direct_renormalized_operator_table_group_channels", 0)
        ) + int(table.n_group_channels)
        stats["direct_renormalized_operator_table_block_matrices"] = int(
            stats.get("direct_renormalized_operator_table_block_matrices", 0)
        ) + int(table.n_block_matrices)
        stats["direct_renormalized_operator_table_block_matrix_elements"] = int(
            stats.get("direct_renormalized_operator_table_block_matrix_elements", 0)
        ) + int(table.block_matrix_elements)
        stats["direct_renormalized_operator_table_block_sparse_nnz"] = int(
            stats.get("direct_renormalized_operator_table_block_sparse_nnz", 0)
        ) + int(table.block_sparse_nnz)
        stats["direct_renormalized_operator_table_last_storage"] = str(table.storage)
        return table

    def _flat_named_family_direct_matvec(self, vec, proto, layout):
        dim = int(self._size(layout))
        if dim <= 0:
            return None
        min_dim = int(self._packed_local_family_flat_direct_matvec_min_dim)
        if min_dim > 0 and dim < min_dim:
            return None
        vector = np.asarray(vec, dtype=np.complex128).reshape(dim)
        backend = self._packed_local_family_flat_direct_matvec_backend
        if backend == "fused_compact_chain":
            fused = self._fused_named_family_environment()
            if fused is None:
                return None
            E, W, F, n_families = fused
            old_E, old_W, old_F = self.E, self.W, self.F
            old_flat_matvec = self._packed_local_flat_matvec
            try:
                self.E, self.W, self.F = E, W, F
                self._packed_local_flat_matvec = True
                probe = self._zero_proto_from_layout(
                    proto,
                    layout,
                    self._block_data_dtype(proto, complex),
                )
                if self._build_compact_matrix_chain_plan(
                    probe,
                    target_layout=layout,
                ) is None:
                    stats = self.profile_stats.setdefault(
                        "packed_flat_complementary_family_action",
                        {},
                    )
                    stats["fused_compact_chain_rejected"] = int(
                        stats.get("fused_compact_chain_rejected", 0)
                    ) + 1
                    stats["fused_compact_chain_rejected_reason"] = "plan_unavailable"
                    return None
                out = self._flat_batched_compact_matrix_chain(
                    vector,
                    proto,
                    layout,
                    project_output=True,
                )
                if out is None:
                    stats = self.profile_stats.setdefault(
                        "packed_flat_complementary_family_action",
                        {},
                    )
                    stats["fused_compact_chain_rejected"] = int(
                        stats.get("fused_compact_chain_rejected", 0)
                    ) + 1
                    stats["fused_compact_chain_rejected_reason"] = "apply_unavailable"
                    return None
            finally:
                self.E, self.W, self.F = old_E, old_W, old_F
                self._packed_local_flat_matvec = old_flat_matvec
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["compiled_direct_matvec_calls"] = int(
                stats.get("compiled_direct_matvec_calls", 0)
            ) + 1
            stats["compiled_direct_matvec_backend"] = str(backend)
            stats["compiled_direct_matvec_family_chains"] = int(
                stats.get("compiled_direct_matvec_family_chains", 0)
            ) + int(n_families)
            stats["fused_compact_chain_calls"] = int(
                stats.get("fused_compact_chain_calls", 0)
            ) + 1
            return np.asarray(out, dtype=np.complex128).reshape(dim)
        if backend == "compact_chain":
            total = np.zeros(dim, dtype=np.complex128)
            n_families = 0
            old_E, old_W, old_F = self.E, self.W, self.F
            old_flat_matvec = self._packed_local_flat_matvec
            try:
                self._packed_local_flat_matvec = True
                for _name, env in self.complementary_family_environments.items():
                    try:
                        E, W, F = env
                    except Exception:
                        continue
                    if E is None or F is None or W is None or len(W) != 2:
                        continue
                    self.E, self.W, self.F = E, W, F
                    probe = self._zero_proto_from_layout(
                        proto,
                        layout,
                        self._block_data_dtype(proto, complex),
                    )
                    if self._build_compact_matrix_chain_plan(
                        probe,
                        target_layout=layout,
                    ) is None:
                        stats = self.profile_stats.setdefault(
                            "packed_flat_complementary_family_action",
                            {},
                        )
                        stats["compact_chain_rejected"] = int(
                            stats.get("compact_chain_rejected", 0)
                        ) + 1
                        stats["compact_chain_rejected_reason"] = "plan_unavailable"
                        stats["compact_chain_rejected_family"] = str(_name)
                        return None
                    out_part = self._flat_batched_compact_matrix_chain(
                        vector,
                        proto,
                        layout,
                        project_output=True,
                    )
                    if out_part is None:
                        stats = self.profile_stats.setdefault(
                            "packed_flat_complementary_family_action",
                            {},
                        )
                        stats["compact_chain_rejected"] = int(
                            stats.get("compact_chain_rejected", 0)
                        ) + 1
                        stats["compact_chain_rejected_reason"] = "apply_unavailable"
                        stats["compact_chain_rejected_family"] = str(_name)
                        return None
                    total += np.asarray(out_part, dtype=np.complex128).reshape(dim)
                    n_families += 1
            finally:
                self.E, self.W, self.F = old_E, old_W, old_F
                self._packed_local_flat_matvec = old_flat_matvec
            if n_families <= 0:
                return None
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["compiled_direct_matvec_calls"] = int(
                stats.get("compiled_direct_matvec_calls", 0)
            ) + 1
            stats["compiled_direct_matvec_backend"] = str(backend)
            stats["compiled_direct_matvec_family_chains"] = int(
                stats.get("compiled_direct_matvec_family_chains", 0)
            ) + int(n_families)
            return total
        if backend == "chain":
            tensor = self._unflatten(
                vector,
                proto,
                layout,
                drop_zero_blocks=True,
                zero_tol=0.0,
            )
            total = None
            n_families = 0
            for _name, env in self.complementary_family_environments.items():
                try:
                    E, W, F = env
                except Exception:
                    continue
                if E is None or F is None or W is None or len(W) != 2:
                    continue
                image = self._matvec_family_components_chain(E, W, F, tensor)
                if image is None:
                    return None
                total = image if total is None else total + image
                n_families += 1
            if total is None:
                return None
            out = self._project_flat_to_layout(total, layout)
            if out is None:
                return None
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["compiled_direct_matvec_calls"] = int(
                stats.get("compiled_direct_matvec_calls", 0)
            ) + 1
            stats["compiled_direct_matvec_backend"] = str(backend)
            stats["compiled_direct_matvec_family_chains"] = int(
                stats.get("compiled_direct_matvec_family_chains", 0)
            ) + int(n_families)
            return np.asarray(out, dtype=np.complex128).reshape(dim)
        if backend == "renormalized_table":
            table = self._flat_named_family_renormalized_operator_table(proto, layout)
            if table is None:
                return None
            moving_environment = getattr(self, "_moving_environment", None)
            if moving_environment is None:
                out = table.matvec(vector)
            else:
                out = moving_environment.compiled_backend.apply_renormalized_operator_table(
                    table,
                    vector,
                )
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["compiled_direct_matvec_calls"] = int(
                stats.get("compiled_direct_matvec_calls", 0)
            ) + 1
            stats["compiled_direct_matvec_backend"] = str(backend)
            stats["renormalized_operator_table_calls"] = int(
                stats.get("renormalized_operator_table_calls", 0)
            ) + 1
            stats["compiled_direct_matvec_entries"] = int(
                stats.get("compiled_direct_matvec_entries", 0)
            ) + int(table.n_entries)
            stats["compiled_direct_matvec_groups"] = int(
                stats.get("compiled_direct_matvec_groups", 0)
            ) + int(table.n_groups)
            stats["compiled_direct_matvec_group_channels"] = int(
                stats.get("compiled_direct_matvec_group_channels", 0)
            ) + int(table.n_group_channels)
            stats["renormalized_operator_table_storage"] = str(table.storage)
            stats["renormalized_operator_table_block_matrices_last"] = int(
                table.n_block_matrices
            )
            stats["renormalized_operator_table_block_matrix_elements_last"] = int(
                table.block_matrix_elements
            )
            stats["renormalized_operator_table_block_sparse_nnz_last"] = int(
                table.block_sparse_nnz
            )
            return np.asarray(out, dtype=np.complex128).reshape(dim)
        collected = self._flat_named_family_csr_kernels(proto, layout)
        if collected is None:
            return None
        if backend == "grouped_compiled":
            groups = collected.get("matvec_groups")
            if (
                groups is None
                or _packed_cython is None
                or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
                or getattr(_packed_cython, "direct_operator_groups_matvec", None) is None
            ):
                return None
            out = _packed_cython.direct_operator_groups_matvec(
                collected["group_left"],
                collected["group_right"],
                collected["group_dims_array"],
                collected["group_in_starts_array"],
                collected["group_out_starts_array"],
                vector,
                dim,
                collected.get("group_scales"),
            )
        elif backend in {"blas", "grouped_blas"}:
            out = np.zeros(dim, dtype=np.complex128)
            if backend == "grouped_blas":
                groups = collected.get("matvec_groups")
                if groups is None:
                    return None
                iterator = (
                    (
                        group["left"],
                        group["right"],
                        group["dims"],
                        group["in_start"],
                        group["out_start"],
                        group.get("scales"),
                    )
                    for group in groups
                )
            else:
                iterator = zip(
                    collected["left"],
                    collected["right"],
                    collected["dims"],
                    collected["in_starts"],
                    collected["out_starts"],
                    collected.get(
                        "scales",
                        (1.0 + 0.0j for _ in collected["left"]),
                    ),
                )
            for left_stack, right_stack, dims, in_start, out_start, scales in iterator:
                ni, nl, nu, nv, nj, nx, nk, ny = (int(v) for v in dims)
                in_size = nj * nx * nk * ny
                out_size = ni * nl * nu * nv
                block = vector[int(in_start) : int(in_start) + in_size]
                if not np.any(block):
                    continue
                a_mat = np.ascontiguousarray(
                    block.reshape(nj, nk, nx, ny)
                    .transpose(0, 2, 1, 3)
                    .reshape(nj * nx, nk * ny)
                )
                tmp = np.matmul(left_stack, a_mat)
                mat_stack = np.matmul(tmp, right_stack)
                if scales is None:
                    mat = mat_stack.sum(axis=0)
                else:
                    mat = (
                        mat_stack
                        * np.asarray(scales, dtype=np.complex128).reshape(-1, 1, 1)
                    ).sum(axis=0)
                out_block = (
                    mat.reshape(ni, nu, nl, nv)
                    .transpose(0, 2, 1, 3)
                    .reshape(out_size)
                )
                out[int(out_start) : int(out_start) + out_size] += out_block
        else:
            if (
                _packed_cython is None
                or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
                or getattr(_packed_cython, "direct_operator_entries_matvec", None) is None
            ):
                return None
            out = _packed_cython.direct_operator_entries_matvec(
                collected["left"],
                collected["right"],
                collected["dims_array"],
                collected["in_starts_array"],
                collected["out_starts_array"],
                vector,
                dim,
                collected.get("scales_array"),
            )
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["compiled_direct_matvec_calls"] = int(
            stats.get("compiled_direct_matvec_calls", 0)
        ) + 1
        stats["compiled_direct_matvec_backend"] = str(backend)
        stats["compiled_direct_matvec_entries"] = int(
            stats.get("compiled_direct_matvec_entries", 0)
        ) + int(self._direct_csr_kernel_count(collected))
        if backend in {"grouped_blas", "grouped_compiled"}:
            groups = collected.get("matvec_groups") or ()
            stats["compiled_direct_matvec_groups"] = int(
                stats.get("compiled_direct_matvec_groups", 0)
            ) + int(len(groups))
            stats["compiled_direct_matvec_group_channels"] = int(
                stats.get("compiled_direct_matvec_group_channels", 0)
            ) + int(sum(int(group.get("channels", 0)) for group in groups))
        return np.asarray(out, dtype=np.complex128).reshape(dim)

    def _flat_direct_family_direct_matvec(self, vec, proto, layout):
        dim = int(self._size(layout))
        if dim <= 0 or not self.complementary_direct_family_environments:
            return None
        min_dim = int(self._packed_local_family_flat_direct_matvec_min_dim)
        if min_dim > 0 and dim < min_dim:
            return None
        collected = self._flat_direct_family_csr_kernels(proto, layout)
        if collected is None:
            return None
        vector = np.asarray(vec, dtype=np.complex128).reshape(dim)
        backend = self._packed_local_family_flat_direct_matvec_backend
        if backend in {
            "compiled",
            "entry_compiled",
            "entries",
            "renormalized_table",
            "renormalized",
            "block2_table",
            "block2_like",
        }:
            backend = "renormalized_direct_table"
        groups = collected.get("matvec_groups")
        if backend == "renormalized_direct_table":
            table = self._flat_direct_family_renormalized_operator_table(proto, layout)
            if table is None:
                return None
            moving_environment = getattr(self, "_moving_environment", None)
            if moving_environment is None:
                out = table.matvec(vector)
            else:
                out = moving_environment.compiled_backend.apply_renormalized_operator_table(
                    table,
                    vector,
                )
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["direct_renormalized_operator_table_calls"] = int(
                stats.get("direct_renormalized_operator_table_calls", 0)
            ) + 1
            stats["direct_renormalized_operator_table_last_storage"] = str(
                table.storage
            )
        elif backend == "grouped_compiled":
            if (
                groups is None
                or _packed_cython is None
                or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
                or getattr(_packed_cython, "direct_operator_groups_matvec", None)
                is None
            ):
                return None
            out = _packed_cython.direct_operator_groups_matvec(
                collected["group_left"],
                collected["group_right"],
                collected["group_dims_array"],
                collected["group_in_starts_array"],
                collected["group_out_starts_array"],
                vector,
                dim,
                collected.get("group_scales"),
            )
        elif backend in {"grouped_blas", "blas"}:
            out = np.zeros(dim, dtype=np.complex128)
            if backend == "grouped_blas":
                if groups is None:
                    return None
                iterator = (
                    (
                        group["left"],
                        group["right"],
                        group["dims"],
                        group["in_start"],
                        group["out_start"],
                        group.get("scales"),
                    )
                    for group in groups
                )
            else:
                iterator = zip(
                    collected["left"],
                    collected["right"],
                    collected["dims"],
                    collected["in_starts"],
                    collected["out_starts"],
                    collected.get(
                        "scales",
                        (1.0 + 0.0j for _ in collected["left"]),
                    ),
                )
            for left_stack, right_stack, dims, in_start, out_start, scales in iterator:
                ni, nl, nu, nv, nj, nx, nk, ny = (int(v) for v in dims)
                in_size = nj * nx * nk * ny
                out_size = ni * nl * nu * nv
                block = vector[int(in_start) : int(in_start) + in_size]
                if not np.any(block):
                    continue
                a_mat = np.ascontiguousarray(
                    block.reshape(nj, nk, nx, ny)
                    .transpose(0, 2, 1, 3)
                    .reshape(nj * nx, nk * ny)
                )
                tmp = np.matmul(left_stack, a_mat)
                mat_stack = np.matmul(tmp, right_stack)
                if scales is None:
                    mat = mat_stack.sum(axis=0)
                else:
                    mat = (
                        mat_stack
                        * np.asarray(scales, dtype=np.complex128).reshape(-1, 1, 1)
                    ).sum(axis=0)
                out_block = (
                    mat.reshape(ni, nu, nl, nv)
                    .transpose(0, 2, 1, 3)
                    .reshape(out_size)
                )
                out[int(out_start) : int(out_start) + out_size] += out_block
        else:
            if (
                _packed_cython is None
                or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
                or getattr(_packed_cython, "direct_operator_entries_matvec", None)
                is None
            ):
                return None
            out = _packed_cython.direct_operator_entries_matvec(
                collected["left"],
                collected["right"],
                collected["dims_array"],
                collected["in_starts_array"],
                collected["out_starts_array"],
                vector,
                dim,
                collected.get("scales_array"),
            )
        if (
            backend != "renormalized_direct_table"
            and self.complementary_family_environments
        ):
            named = self._flat_named_family_direct_matvec(vec, proto, layout)
            if named is None:
                return None
            out = np.asarray(out, dtype=np.complex128).reshape(dim) + np.asarray(
                named,
                dtype=np.complex128,
            ).reshape(dim)
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["compiled_generator_direct_matvec_calls"] = int(
            stats.get("compiled_generator_direct_matvec_calls", 0)
        ) + 1
        stats["compiled_generator_direct_matvec_entries"] = int(
            stats.get("compiled_generator_direct_matvec_entries", 0)
        ) + int(self._direct_csr_kernel_count(collected))
        stats["compiled_generator_direct_matvec_backend"] = "compiled"
        stats["compiled_generator_direct_matvec_kernel"] = str(backend)
        if groups is not None:
            stats["compiled_generator_direct_matvec_groups"] = int(
                stats.get("compiled_generator_direct_matvec_groups", 0)
            ) + int(len(groups))
            stats["compiled_generator_direct_matvec_group_channels"] = int(
                stats.get("compiled_generator_direct_matvec_group_channels", 0)
            ) + int(sum(int(group.get("channels", 0)) for group in groups))
        return np.asarray(out, dtype=np.complex128).reshape(dim)

    def _flat_named_family_plan_action_table_direct_csr(self, proto, layout):
        if (
            _packed_cython is None
            or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            or getattr(_packed_cython, "direct_operator_entries_csr", None) is None
            or self._packed_local_family_flat_sparse_entry_emitter
        ):
            return None
        dim = int(self._size(layout))
        cap = int(self._packed_local_family_flat_direct_csr_build_max_dim)
        if dim <= 0 or cap <= 0 or dim > cap:
            return None
        collected = self._flat_named_family_csr_kernels(proto, layout)
        if collected is None:
            return None
        dims_array = collected["dims_array"]
        in_starts = collected["in_starts_array"]
        out_starts = collected["out_starts_array"]
        use_pattern_cache = (
            self._shared_flat_complementary_action_pattern_cache is not None
            and int(self._shared_flat_complementary_action_pattern_cache_max_entries) > 0
        )
        pattern_key = None
        pattern = None
        if use_pattern_cache:
            pattern_key = self._direct_operator_csr_pattern_key(
                layout,
                collected,
                dim,
                self.bond,
            )
            pattern = self._get_direct_operator_csr_pattern(pattern_key)
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        if (
            pattern is not None
            and pattern.get("lookup") is not None
            and getattr(_packed_cython, "direct_operator_entries_csr_refill", None)
            is not None
        ):
            try:
                csr_values, raw_nnz, missing_nnz = (
                    _packed_cython.direct_operator_entries_csr_refill(
                        collected["left"],
                        collected["right"],
                        dims_array,
                        in_starts,
                        out_starts,
                        pattern["lookup"],
                        int(dim),
                        int(pattern["indices"].size),
                        1.0e-14,
                        collected.get("scales_array"),
                    )
                )
            except Exception:
                csr_values = None
                raw_nnz = 0
                missing_nnz = 1
            if csr_values is not None and int(missing_nnz) == 0:
                stats["compiled_direct_csr_pattern_hits"] = int(
                    stats.get("compiled_direct_csr_pattern_hits", 0)
                ) + 1
                stats["compiled_direct_csr_pattern_refill_raw_nnz"] = int(
                    stats.get("compiled_direct_csr_pattern_refill_raw_nnz", 0)
                ) + int(raw_nnz)
                return AbelianSparseComplementaryBoundaryActionTable.from_csr(
                    pattern["indptr"],
                    pattern["indices"],
                    csr_values,
                    dim,
                    layout,
                    self._qns_from_layout_with_proto(layout, proto),
                    proto.dirs[:],
                    raw_nnz=int(raw_nnz),
                    bond=self.bond,
                    source="sparse_flat_named_family_plan_entries_direct_csr_pattern",
                    boundary_family_tables=self._boundary_family_tables(),
                )
            stats["compiled_direct_csr_pattern_misses"] = int(
                stats.get("compiled_direct_csr_pattern_misses", 0)
            ) + 1
            stats["compiled_direct_csr_pattern_missing_nnz"] = int(
                stats.get("compiled_direct_csr_pattern_missing_nnz", 0)
            ) + int(missing_nnz)
        csr_builder = _packed_cython.direct_operator_entries_csr
        if (
            self._packed_local_family_flat_direct_csr_extract_backend
            in {"numpy", "np", "vectorized"}
            and getattr(_packed_cython, "direct_operator_entries_csr_np_extract", None)
            is not None
        ):
            csr_builder = _packed_cython.direct_operator_entries_csr_np_extract
        indptr, indices, csr_values, raw_nnz = csr_builder(
            collected["left"],
            collected["right"],
            dims_array,
            in_starts,
            out_starts,
            int(dim),
            1.0e-14,
            collected.get("scales_array"),
        )
        stats["compiled_direct_csr_builds"] = int(
            stats.get("compiled_direct_csr_builds", 0)
        ) + 1
        stats["compiled_direct_csr_extract_backend"] = str(
            self._packed_local_family_flat_direct_csr_extract_backend
        )
        stats["compiled_direct_csr_entries"] = int(
            stats.get("compiled_direct_csr_entries", 0)
        ) + int(self._direct_csr_kernel_count(collected))
        stats["compiled_direct_csr_raw_nnz"] = int(
            stats.get("compiled_direct_csr_raw_nnz", 0)
        ) + int(raw_nnz)
        if use_pattern_cache:
            promoted = False
            cached_indptr = None if pattern is None else pattern.get("indptr")
            cached_indices = None if pattern is None else pattern.get("indices")
            if (
                cached_indptr is not None
                and cached_indices is not None
                and np.array_equal(cached_indptr, indptr)
                and np.array_equal(cached_indices, indices)
            ):
                lookup = self._direct_operator_csr_lookup(indptr, indices, dim)
                if lookup is not None:
                    self._put_direct_operator_csr_pattern(
                        pattern_key,
                        {
                            "indptr": np.ascontiguousarray(indptr, dtype=np.int64),
                            "indices": np.ascontiguousarray(indices, dtype=np.int64),
                            "lookup": lookup,
                        },
                    )
                    promoted = True
                    stats["compiled_direct_csr_pattern_promotions"] = int(
                        stats.get("compiled_direct_csr_pattern_promotions", 0)
                    ) + 1
            if not promoted:
                self._put_direct_operator_csr_pattern(
                    pattern_key,
                    {
                        "indptr": np.ascontiguousarray(indptr, dtype=np.int64),
                        "indices": np.ascontiguousarray(indices, dtype=np.int64),
                        "lookup": None,
                    },
                )
            stats["compiled_direct_csr_pattern_stores"] = int(
                stats.get("compiled_direct_csr_pattern_stores", 0)
            ) + 1
        return AbelianSparseComplementaryBoundaryActionTable.from_csr(
            indptr,
            indices,
            csr_values,
            dim,
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            raw_nnz=int(raw_nnz),
            bond=self.bond,
            source="sparse_flat_named_family_plan_entries_direct_csr",
            boundary_family_tables=self._boundary_family_tables(),
        )

    def _flat_named_family_plan_action_table(self, proto, layout):
        if not self.complementary_family_environments:
            return None
        direct_csr = self._flat_named_family_plan_action_table_direct_csr(
            proto,
            layout,
        )
        if direct_csr is not None:
            return direct_csr
        rows = []
        cols = []
        values = []
        probe = self._zero_proto_from_layout(
            proto,
            layout,
            self._block_data_dtype(proto, complex),
        )
        for name, env in self.complementary_family_environments.items():
            try:
                E, W, F = env
            except Exception:
                continue
            if E is None or F is None or W is None or len(W) != 2:
                continue
            plan = self._build_direct_operator_plan(
                probe,
                E,
                W,
                F,
                ("flat_named_family", str(name)),
                build_expr=False,
            )
            if not self._accumulate_direct_operator_plan_coo(
                plan,
                layout,
                rows,
                cols,
                values,
            ):
                return None
        if not rows and not values:
            return None
        return AbelianSparseComplementaryBoundaryActionTable(
            rows,
            cols,
            values,
            self._size(layout),
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source="sparse_flat_named_family_plan_entries",
            boundary_family_tables=self._boundary_family_tables(),
        )

    def _flat_named_family_shared_cache_key(self, proto, layout):
        if not self.complementary_family_environments:
            return None
        tokens = []
        for name, env in sorted(
            self.complementary_family_environments.items(),
            key=lambda item: str(item[0]),
        ):
            try:
                E, W, F = env
            except Exception:
                return None
            if E is None or F is None or W is None or len(W) != 2:
                return None
            tokens.append((str(name), self._component_action_token(E, W, F)))
        if not tokens:
            return None
        return (
            "flat_named_family_plan_action_table",
            tuple(tokens),
            tuple(layout),
            tuple(proto.dirs),
        )

    def _get_shared_flat_complementary_action_table(self, key):
        cache = self._shared_flat_complementary_action_table_cache
        if cache is None or key is None:
            return None
        try:
            table = cache.get(key)
        except AttributeError:
            return None
        if table is not None and hasattr(cache, "move_to_end"):
            cache.move_to_end(key)
        return table

    def _put_shared_flat_complementary_action_table(self, key, table):
        cache = self._shared_flat_complementary_action_table_cache
        if cache is None or key is None or table is None:
            return
        try:
            cache[key] = table
            if hasattr(cache, "move_to_end"):
                cache.move_to_end(key)
            cap = int(self._shared_flat_complementary_action_table_cache_max_entries)
            while cap > 0 and len(cache) > cap:
                if hasattr(cache, "popitem"):
                    try:
                        cache.popitem(last=False)
                    except TypeError:
                        first_key = next(iter(cache))
                        del cache[first_key]
                else:
                    break
        except Exception:
            return

    def _flat_complementary_action_table(self, proto, layout):
        if not self._packed_local_family_flat_matvec:
            return None
        if self.complementary_operator_families is None:
            return None
        layout = tuple(layout)
        dim = int(self._size(layout))
        cap = int(self._packed_local_family_flat_matvec_max_dim)
        if dim <= 0 or cap <= 0 or dim > cap:
            return None
        key = ("flat_complementary_action", int(self.bond or 0), layout)
        cached = self._flat_complementary_action_table_cache.get(key)
        if cached is not None:
            return cached

        shared_key = (
            None
            if self.complementary_direct_family_environments
            else self._flat_named_family_shared_cache_key(proto, layout)
        )
        shared = self._get_shared_flat_complementary_action_table(shared_key)
        if shared is not None:
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["shared_cache_hits"] = int(stats.get("shared_cache_hits", 0)) + 1
            stats["last_dimension"] = int(shared.dim)
            stats["last_nnz"] = int(shared.nnz)
            stats["last_raw_nnz"] = int(getattr(shared, "raw_nnz", shared.nnz))
            stats["last_storage"] = str(getattr(shared, "storage", "sparse_coo"))
            stats["last_basis_source"] = "direct_operator_plan_entries"
            stats["last_cache"] = "shared_hit"
            self._flat_complementary_action_table_cache[key] = shared
            return shared

        build_start = time.perf_counter()
        table = (
            self._flat_generator_family_plan_action_table(proto, layout)
            if self.complementary_direct_family_environments
            else self._flat_named_family_plan_action_table(proto, layout)
        )
        if table is not None:
            stats = self.profile_stats.setdefault(
                "packed_flat_complementary_family_action",
                {},
            )
            stats["builds"] = int(stats.get("builds", 0)) + 1
            stats["last_build_seconds"] = float(time.perf_counter() - build_start)
            stats["build_seconds"] = float(stats.get("build_seconds", 0.0)) + float(
                stats["last_build_seconds"]
            )
            stats["last_dimension"] = int(table.dim)
            stats["last_nnz"] = int(table.nnz)
            stats["last_raw_nnz"] = int(getattr(table, "raw_nnz", table.nnz))
            stats["last_storage"] = str(getattr(table, "storage", "sparse_coo"))
            stats["last_basis_source"] = "direct_operator_plan_entries"
            stats["shared_cache_misses"] = int(stats.get("shared_cache_misses", 0)) + (
                1 if shared_key is not None else 0
            )
            stats["last_cache"] = "built"
            self._flat_complementary_action_table_cache[key] = table
            self._put_shared_flat_complementary_action_table(shared_key, table)
            return table

        rows = []
        cols = []
        values = []
        source = "sparse_flat_full_complementary_family_action"
        basis_source = "full_matvec"
        for col in range(dim):
            basis_vec = np.zeros(dim, dtype=complex)
            basis_vec[col] = 1.0
            basis = self._unflatten(
                basis_vec,
                proto,
                layout,
                drop_zero_blocks=True,
                zero_tol=0.0,
            )
            image = None
            if self.complementary_family_environments:
                named = self._matvec_named_family_channels(basis)
                if named is not None:
                    named_channels, _named_stats = named
                    total = None
                    for tensor in named_channels.values():
                        total = tensor if total is None else total + tensor
                    if total is not None:
                        image = total
                        source = "sparse_flat_named_family_environment"
                        basis_source = "named_family_plans"
            if image is None:
                image = self.matvec(basis)
            flat = self._project_flat_to_layout(image, layout)
            if flat is None:
                stats = self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                )
                stats["rejected_reason"] = "layout_not_closed"
                stats["rejected_dimension"] = int(dim)
                return None
            nz = np.nonzero(np.abs(flat) > 1.0e-14)[0]
            rows.extend(int(row) for row in nz)
            cols.extend([int(col)] * int(len(nz)))
            values.extend(complex(flat[row]) for row in nz)

        table = AbelianSparseComplementaryBoundaryActionTable(
            rows,
            cols,
            values,
            dim,
            layout,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=self.bond,
            source=source,
            boundary_family_tables=self._boundary_family_tables(),
        )
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["builds"] = int(stats.get("builds", 0)) + 1
        stats["last_build_seconds"] = float(time.perf_counter() - build_start)
        stats["build_seconds"] = float(stats.get("build_seconds", 0.0)) + float(
            stats["last_build_seconds"]
        )
        stats["last_dimension"] = int(dim)
        stats["last_nnz"] = int(table.nnz)
        stats["last_raw_nnz"] = int(getattr(table, "raw_nnz", table.nnz))
        stats["last_storage"] = str(getattr(table, "storage", "sparse_coo"))
        stats["last_basis_source"] = str(basis_source)
        self._flat_complementary_action_table_cache[key] = table
        return table

    def _direct_complementary_family_flat_matvec(self, vec, proto, layout):
        if self.complementary_direct_family_environments:
            return None
        tensor = self._unflatten(
            np.asarray(vec, dtype=np.complex128).reshape(int(self._size(layout))),
            proto,
            layout,
            drop_zero_blocks=True,
            zero_tol=0.0,
        )
        named = self._matvec_named_family_channels(tensor)
        if named is None:
            return None
        channels, _stats = named
        total = None
        for channel_tensor in channels.values():
            total = channel_tensor if total is None else total + channel_tensor
        if total is None:
            return None
        return self._project_flat_to_layout(total, layout)

    def _flat_complementary_family_matvec(self, vec, proto, layout):
        start = time.perf_counter()
        layout = tuple(layout)
        lazy_key = ("flat_complementary_family_matvec", layout)
        call_index = int(
            self._flat_complementary_family_matvec_counts.get(lazy_key, 0)
        )
        build_after = int(self._packed_local_family_flat_matvec_build_after_calls)
        if build_after > 0 and call_index < build_after:
            out = self._direct_complementary_family_flat_matvec(vec, proto, layout)
            if out is not None:
                self._flat_complementary_family_matvec_counts[lazy_key] = call_index + 1
                elapsed = float(time.perf_counter() - start)
                stats = self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                )
                stats["calls"] = int(stats.get("calls", 0)) + 1
                stats["seconds"] = float(stats.get("seconds", 0.0)) + elapsed
                stats["last_seconds"] = elapsed
                stats["lazy_direct_calls"] = int(stats.get("lazy_direct_calls", 0)) + 1
                stats["last_cache"] = "lazy_direct"
                stats["last"] = {
                    "dimension": int(self._size(layout)),
                    "nnz": None,
                    "raw_nnz": None,
                    "source": "direct_named_family_before_flat_table",
                    "storage": "direct",
                    "cache": "lazy_direct",
                    "bond": None if self.bond is None else int(self.bond),
                }
                return out
        if (
            self._packed_local_family_flat_direct_matvec
            and not self.complementary_direct_family_environments
        ):
            out = self._flat_named_family_direct_matvec(vec, proto, layout)
            if out is not None:
                self._flat_complementary_family_matvec_counts[lazy_key] = call_index + 1
                elapsed = float(time.perf_counter() - start)
                stats = self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                )
                stats["calls"] = int(stats.get("calls", 0)) + 1
                stats["seconds"] = float(stats.get("seconds", 0.0)) + elapsed
                stats["last_seconds"] = elapsed
                stats["last_cache"] = "direct_compiled"
                stats["last"] = {
                    "dimension": int(self._size(layout)),
                    "nnz": None,
                    "raw_nnz": None,
                    "source": "direct_compiled_named_family_matvec",
                    "storage": "direct_compiled",
                    "cache": "direct_compiled",
                    "bond": None if self.bond is None else int(self.bond),
                }
                return out
        if (
            self._packed_local_family_flat_direct_matvec
            and self.complementary_direct_family_environments
        ):
            out = self._flat_direct_family_direct_matvec(vec, proto, layout)
            if out is not None:
                self._flat_complementary_family_matvec_counts[lazy_key] = call_index + 1
                elapsed = float(time.perf_counter() - start)
                stats = self.profile_stats.setdefault(
                    "packed_flat_complementary_family_action",
                    {},
                )
                stats["calls"] = int(stats.get("calls", 0)) + 1
                stats["seconds"] = float(stats.get("seconds", 0.0)) + elapsed
                stats["last_seconds"] = elapsed
                stats["last_cache"] = "direct_generator_compiled"
                stats["last"] = {
                    "dimension": int(self._size(layout)),
                    "nnz": None,
                    "raw_nnz": None,
                    "source": "direct_compiled_generator_family_matvec",
                    "storage": "direct_compiled",
                    "cache": "direct_compiled",
                    "bond": None if self.bond is None else int(self.bond),
                }
                return out
        table = self._flat_complementary_action_table(proto, layout)
        if table is None:
            return None
        self._flat_complementary_family_matvec_counts[lazy_key] = call_index + 1
        out = table.matvec(vec)
        stats = self.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        elapsed = float(time.perf_counter() - start)
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + elapsed
        stats["last_seconds"] = elapsed
        stats["last"] = {
            "dimension": int(table.dim),
            "nnz": int(table.nnz),
            "raw_nnz": int(getattr(table, "raw_nnz", table.nnz)),
            "source": str(table.source),
            "storage": str(getattr(table, "storage", "sparse_coo")),
            "cache": stats.get("last_cache"),
            "bond": None if self.bond is None else int(self.bond),
        }
        return out

    def _record_complementary_split(
        self,
        mode,
        *,
        local=None,
        local_channels=None,
        boundary=None,
        boundary_table=None,
        boundary_operator=None,
    ):
        stats = self.complementary_split_stats
        if stats is None:
            return
        mode = str(mode)
        stats["calls"] = int(stats.get("calls", 0)) + 1
        modes = stats.setdefault("modes", {})
        modes[mode] = int(modes.get(mode, 0)) + 1
        table_stats = None if boundary_table is None else boundary_table.stats
        entry = {
            "bond": self.bond,
            "mode": mode,
            "local_norm": None if local is None else float(local.norm()),
            "local_channels": {
                str(name): float(value.norm())
                for name, value in (local_channels or {}).items()
            },
            "boundary_norm": None if boundary is None else float(boundary.norm()),
            "boundary_table": table_stats,
            "boundary_operator": boundary_operator,
            "metadata": (
                self._compact_complementary_split_metadata()
                if self._debug_complementary_split_metadata
                else None
            ),
        }
        stats["last"] = entry
        if self.bond is not None:
            bonds = stats.setdefault("bonds", {})
            bond_key = int(self.bond)
            bond_stats = bonds.setdefault(bond_key, {"calls": 0, "modes": {}})
            bond_stats["calls"] = int(bond_stats.get("calls", 0)) + 1
            bond_modes = bond_stats.setdefault("modes", {})
            bond_modes[mode] = int(bond_modes.get(mode, 0)) + 1
            bond_stats["last"] = entry

    def _audit_complementary_action(self, A, candidate, mode):
        if not self._debug_complementary_action_check:
            return None
        stats = self.complementary_split_stats
        if stats is not None:
            audits = stats.setdefault(
                "action_audits",
                {
                    "enabled": True,
                    "calls": 0,
                    "checked": 0,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                    "tol": float(self._debug_complementary_action_check_tol),
                    "failures": 0,
                },
            )
            if int(audits.get("checked", 0)) >= self._debug_complementary_action_check_limit:
                audits["calls"] = int(audits.get("calls", 0)) + 1
                return None
        else:
            audits = None

        exact = self._matvec_generic(A)
        diff = candidate - exact
        abs_diff = float(diff.norm())
        ref_norm = max(float(exact.norm()), 1.0e-30)
        rel_diff = abs_diff / ref_norm
        record = {
            "bond": self.bond,
            "mode": str(mode),
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "tol": float(self._debug_complementary_action_check_tol),
        }
        if audits is not None:
            audits["calls"] = int(audits.get("calls", 0)) + 1
            audits["checked"] = int(audits.get("checked", 0)) + 1
            audits["max_abs_diff"] = max(float(audits.get("max_abs_diff", 0.0)), abs_diff)
            audits["max_rel_diff"] = max(float(audits.get("max_rel_diff", 0.0)), rel_diff)
            audits["last"] = record
            if abs_diff > self._debug_complementary_action_check_tol:
                audits["failures"] = int(audits.get("failures", 0)) + 1
        if abs_diff > self._debug_complementary_action_check_tol:
            raise AssertionError(
                "Complementary local action mismatch on bond {} in mode {}: "
                "abs_diff={:.6e}, rel_diff={:.6e}, tol={:.6e}".format(
                    self.bond,
                    mode,
                    abs_diff,
                    rel_diff,
                    self._debug_complementary_action_check_tol,
                )
            )
        return record

    @staticmethod
    def _layout(A):
        return abelian_local_layout_from_tensor(A)

    @staticmethod
    def _qns_from_layout(layout):
        return abelian_qns_from_layout(layout)

    @staticmethod
    def _qns_from_layout_with_proto(layout, proto):
        return abelian_qns_from_layout(layout, proto)

    @staticmethod
    def _size(layout):
        return abelian_local_layout_size(layout)

    @staticmethod
    def _sector_signature(key, dirs):
        return abelian_sector_signature(key, dirs)

    @staticmethod
    def _two_site_mps_flow_valid(key):
        return abelian_two_site_mps_flow_valid(key)

    @staticmethod
    def _dense_policy(max_dim):
        if isinstance(max_dim, str):
            policy = max_dim.strip().lower()
            if policy == "auto":
                return True, 256, 32
            if policy in {"off", "none", "false", "0"}:
                return False, 0, 0
            try:
                max_dim = int(policy)
            except ValueError as exc:
                raise ValueError("local_dense_max_dim must be an integer, 0, or 'auto'.") from exc
        max_dim = int(max_dim)
        return False, max_dim, 0

    @staticmethod
    def _block_data_dtype(*objects):
        return abelian_block_data_dtype(*objects)

    def _local_action_dtype(self, *objects):
        return self._block_data_dtype(
            self.E,
            self.W,
            self.F,
            self.complementary_family_environments,
            self.complementary_direct_family_environments,
            *objects,
        )

    @staticmethod
    def _flatten(A, layout):
        return abelian_flatten_to_layout(A, layout)

    @staticmethod
    def _tensor_from_block_data_like(proto, data, qns, dirs):
        if isinstance(proto, AbelianSiteTensorData):
            return AbelianSiteTensorData(data, qns, dirs)
        return BlockTensor(data, qns, dirs)

    @staticmethod
    def _unflatten(vec, proto, layout, *, drop_zero_blocks=False, zero_tol=0.0):
        data, qns, dirs = abelian_unflatten_data_from_layout(
            vec,
            layout,
            proto=proto,
            drop_zero_blocks=drop_zero_blocks,
            zero_tol=zero_tol,
        )
        return HamiltonianMultiplyU1._tensor_from_block_data_like(
            proto,
            data,
            qns,
            dirs,
        )

    @staticmethod
    def _zero_proto_from_layout(proto, layout, dtype):
        data, qns, dirs = abelian_zero_data_from_layout(
            layout,
            proto=proto,
            dtype=dtype,
        )
        return HamiltonianMultiplyU1._tensor_from_block_data_like(
            proto,
            data,
            qns,
            dirs,
        )

    @staticmethod
    def _layout_offsets(layout):
        return abelian_layout_offsets(layout)

    def _flat_jacobi_diagonal(self, proto, layout):
        if not self._packed_local_flat_preconditioner:
            return None
        layout = tuple(layout)
        moving_environment = getattr(self, "_moving_environment", None)
        if moving_environment is not None:
            diagonal = moving_environment.flat_jacobi_diagonal(self, proto, layout)
            if diagonal is not None:
                return diagonal
        cache_key = (self._action_token(), layout, "flat_jacobi_diagonal")
        if cache_key in self._flat_diagonal_cache:
            return self._flat_diagonal_cache[cache_key]
        build_start = time.perf_counter()
        flat_result = abelian_flat_qchem_jacobi_diagonal(
            layout,
            self.E,
            self.W,
            self.F,
        )
        if flat_result.flat is not None:
            flat = np.asarray(flat_result.flat, dtype=np.complex128).reshape(
                self._size(layout)
            )
            self._flat_diagonal_cache[cache_key] = flat
            self._record_plan_profile(
                "flat_jacobi_diagonal",
                time.perf_counter() - build_start,
                candidate_entries=int(flat_result.candidate_entries),
                diagonal_contributions=int(flat_result.contributions),
                diagonal_blocks=int(flat_result.diagonal_blocks),
                backend="block_data",
            )
            self.profile_stats["preconditioner"] = {
                "kind": "flat_jacobi_diagonal",
                "available": True,
                "backend": "block_data",
                "diagonal_blocks": int(flat_result.diagonal_blocks),
                "diagonal_contributions": int(flat_result.contributions),
            }
            return flat

        fallback_stats = self.profile_stats.setdefault("packed_flat_preconditioner", {})
        fallback_stats["blocktensor_diagonal_fallbacks"] = int(
            fallback_stats.get("blocktensor_diagonal_fallbacks", 0)
        ) + 1
        fallback_stats["last_block_data_rejected_reason"] = (
            flat_result.rejected_reason
        )

        dtype = self._local_action_dtype(proto)
        proto_full = self._zero_proto_from_layout(proto, layout, dtype)
        diagonal = self.diagonal(proto_full)
        if diagonal is None:
            self._flat_diagonal_cache[cache_key] = None
            return None
        flat = self._flatten(diagonal, layout)
        self._flat_diagonal_cache[cache_key] = flat
        return flat

    @staticmethod
    def _remap_flat(vec, old_layout, new_layout):
        return abelian_remap_flat_layout(vec, old_layout, new_layout)

    @staticmethod
    def _axis_sector_dims(tensor, axis):
        return abelian_axis_sector_dims(tensor, axis)

    @classmethod
    def _merge_axis_sector_dims(cls, dims, tensor, axis):
        for qn, dim in abelian_axis_sector_dims(tensor, axis).items():
            dims[qn] = max(int(dims.get(qn, 0)), int(dim))

    def _safe_two_site_layout_map(self, proto):
        return abelian_safe_two_site_layout_map(proto, self.W)

    @staticmethod
    def _layout_from_map(layout_map):
        return abelian_layout_from_map(layout_map)

    @staticmethod
    def _merge_layout_tensor(
        layout_map,
        tensor,
        *,
        dirs=None,
        allowed_signatures=None,
        allowed_layout_map=None,
        require_two_site_mps_flow=False,
    ):
        return abelian_merge_layout_tensor(
            layout_map,
            tensor,
            dirs=dirs,
            allowed_signatures=allowed_signatures,
            allowed_layout_map=allowed_layout_map,
            require_two_site_mps_flow=require_two_site_mps_flow,
        )

    def solve_packed_davidson(
        self,
        v0,
        *,
        tol=1.0e-5,
        max_iter=30,
        preconditioner=None,
        current=None,
        return_flat=False,
        initial_flat=None,
        initial_layout=None,
        initial_is_current=False,
        return_update=False,
        update_direction="right",
        update_m_max=None,
    ):
        """Single-root Davidson with flat NumPy vectors and dynamic Abelian blocks."""
        packed_solve_start = time.perf_counter()
        self.last_packed_davidson_candidate = None
        self.last_packed_davidson_candidate_flat = None
        self.last_packed_davidson_candidate_layout = None
        self.last_packed_davidson_candidate_energy = None
        self.last_packed_davidson_candidate_residual = None
        self.last_packed_davidson_solution_flat = None
        self.last_packed_davidson_solution_layout = None
        self.last_packed_davidson_solution_energy = None
        self.last_packed_davidson_solution_residual = None
        self.last_packed_davidson_solution_converged = None
        self.last_packed_davidson_solution_update = None
        current_tensor = v0 if current is None else current
        layout_map = {key: tuple(shape) for key, shape in self._layout(v0)}
        current_layout = self._layout_from_map(layout_map)
        current_layout_dimension = int(self._size(current_layout))
        allowed_layout_map = None
        safe_layout_dimension = None
        safe_layout_blocks = None
        project_current_support = False
        large_safe_layout = False
        projected_from_safe_layout_too_large = False
        projected_truncated_current_support = False
        projected_retained_norm = None
        projected_retained_blocks = None
        projected_original_blocks = int(len(layout_map))
        active_max_dim = int(self._packed_local_davidson_max_dim)
        active_restart_dim = int(self._packed_local_davidson_restart_dim)

        if self._packed_local_safe_layout_expansion:
            allowed_layout_map = self._safe_two_site_layout_map(v0)
            if allowed_layout_map is not None:
                for key, shape in layout_map.items():
                    if allowed_layout_map.get(key) != tuple(shape):
                        self.profile_stats["packed_local_davidson"] = {
                            "iterations": 0,
                            "dimension": int(self._size(self._layout_from_map(layout_map))),
                            "basis_size": 0,
                            "layout_blocks": int(len(layout_map)),
                            "layout_expansions": 0,
                            "converged": False,
                            "rejected_reason": "initial_layout_not_safe",
                            "current_layout_dimension": int(current_layout_dimension),
                        }
                        return None
                safe_layout = self._layout_from_map(allowed_layout_map)
                safe_layout_dimension = int(self._size(safe_layout))
                safe_layout_blocks = int(len(safe_layout))
                large_safe_max_dim = int(self._packed_local_large_safe_max_dim)
                can_use_large_safe_layout = (
                    large_safe_max_dim > 0
                    and safe_layout_dimension <= large_safe_max_dim
                    and bool(self._packed_local_use_safe_closure)
                    and bool(self._packed_local_flat_matvec)
                    and not bool(self._packed_local_project_current_support)
                )
                if (
                    self._packed_local_preflight_safe_closure
                    and active_max_dim > 0
                    and safe_layout_dimension > int(active_max_dim)
                    and can_use_large_safe_layout
                ):
                    active_max_dim = int(large_safe_max_dim)
                    large_safe_layout = True
                    large_restart = int(self._packed_local_large_safe_restart_dim)
                    if large_restart > 0 and (
                        active_restart_dim <= 0 or active_restart_dim > large_restart
                    ):
                        active_restart_dim = int(large_restart)
                elif (
                    self._packed_local_preflight_safe_closure
                    and active_max_dim > 0
                    and safe_layout_dimension > int(active_max_dim)
                ):
                    if (
                        self._packed_local_project_current_support
                    ):
                        if current_layout_dimension > int(active_max_dim):
                            if self._packed_local_project_current_support_truncate:
                                truncation = abelian_truncate_layout_map_by_norm(
                                    layout_map,
                                    v0,
                                    active_max_dim,
                                    current_dim=current_layout_dimension,
                                )
                            else:
                                truncation = None
                            if truncation is not None and truncation.layout_map is not None:
                                layout_map = truncation.layout_map
                                projected_truncated_current_support = bool(
                                    truncation.truncated
                                )
                                projected_retained_blocks = truncation.retained_blocks
                                projected_retained_norm = truncation.retained_norm
                                project_current_support = True
                                projected_from_safe_layout_too_large = True
                            else:
                                project_current_support = False
                        else:
                            project_current_support = True
                            projected_from_safe_layout_too_large = True
                    else:
                        self.profile_stats["packed_local_davidson"] = {
                            "iterations": 0,
                            "dimension": int(current_layout_dimension),
                            "basis_size": 0,
                            "layout_blocks": int(len(layout_map)),
                            "layout_expansions": 0,
                            "converged": False,
                            "rejected_reason": "safe_layout_too_large",
                            "rejected_dimension": int(safe_layout_dimension),
                            "rejected_layout_blocks": int(safe_layout_blocks),
                            "safe_layout_dimension": int(safe_layout_dimension),
                            "safe_layout_blocks": int(safe_layout_blocks),
                            "current_layout_dimension": int(current_layout_dimension),
                            "large_safe_layout": False,
                            "large_safe_max_dim": int(large_safe_max_dim),
                        }
                        return None
                    if not project_current_support:
                        self.profile_stats["packed_local_davidson"] = {
                            "iterations": 0,
                            "dimension": int(current_layout_dimension),
                            "basis_size": 0,
                            "layout_blocks": int(len(layout_map)),
                            "layout_expansions": 0,
                            "converged": False,
                            "rejected_reason": "safe_layout_too_large",
                            "rejected_dimension": int(safe_layout_dimension),
                            "rejected_layout_blocks": int(safe_layout_blocks),
                            "safe_layout_dimension": int(safe_layout_dimension),
                            "safe_layout_blocks": int(safe_layout_blocks),
                            "current_layout_dimension": int(current_layout_dimension),
                            "large_safe_layout": False,
                            "large_safe_max_dim": int(large_safe_max_dim),
                            "projected_truncated_current_support": bool(
                                projected_truncated_current_support
                            ),
                        }
                        return None
                if self._packed_local_use_safe_closure and not project_current_support:
                    layout_map = dict(allowed_layout_map)
        layout = self._layout_from_map(layout_map)
        dim = self._size(layout)
        if dim <= 0 or dim < int(self._packed_local_davidson_min_dim):
            return None
        if (
            active_max_dim > 0
            and dim > int(active_max_dim)
        ):
            self.profile_stats["packed_local_davidson"] = {
                "iterations": 0,
                "dimension": int(dim),
                "basis_size": 0,
                "layout_blocks": int(len(layout)),
                "layout_expansions": 0,
                "converged": False,
                "rejected_reason": "layout_too_large",
                "safe_layout_blocks": (
                    safe_layout_blocks
                    if safe_layout_blocks is not None
                    else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                ),
                "safe_layout_dimension": safe_layout_dimension,
                "current_layout_dimension": int(current_layout_dimension),
                "large_safe_layout": bool(large_safe_layout),
                "large_safe_max_dim": int(self._packed_local_large_safe_max_dim),
                "projected_current_support": bool(project_current_support),
                "projected_truncated_current_support": bool(
                    projected_truncated_current_support
                ),
            }
            return None

        V = []
        HV = []
        expansions = 0
        rejected_reason = None
        rejected_dimension = None
        rejected_layout_blocks = None
        moving_environment = getattr(self, "_moving_environment", None)
        moving_compiled_family_matvec = None
        moving_compact_plan_matvec = None

        def _record_moving_local_phase(name, elapsed):
            if moving_environment is None:
                return
            stats = moving_environment.moving_profile_stats
            key = str(name)
            stats[key] = float(stats.get(key, 0.0)) + float(elapsed)

        _record_moving_local_phase(
            "packed_local_setup_seconds",
            time.perf_counter() - packed_solve_start,
        )
        allow_layout_expansion = bool(self._packed_local_allow_layout_expansion)
        if project_current_support:
            allow_layout_expansion = False
        if self._packed_local_safe_layout_expansion and allowed_layout_map is None:
            allow_layout_expansion = False
        projected_pack_calls = 0
        projected_discarded_blocks = 0
        projected_discarded_norm_sq = 0.0

        def _apply_layout(new_layout):
            nonlocal layout, V, HV, expansions, moving_compiled_family_matvec
            nonlocal moving_compact_plan_matvec
            if new_layout == layout:
                return
            old_layout = layout
            layout = new_layout
            V = [self._remap_flat(vec, old_layout, layout) for vec in V]
            HV = [self._remap_flat(vec, old_layout, layout) for vec in HV]
            expansions += 1
            moving_compiled_family_matvec = None
            moving_compact_plan_matvec = None

        def _ensure_tensor(tensor):
            nonlocal rejected_reason, rejected_dimension, rejected_layout_blocks
            merged, changed = self._merge_layout_tensor(
                layout_map,
                tensor,
                dirs=v0.dirs,
                allowed_signatures=None,
                allowed_layout_map=allowed_layout_map,
                require_two_site_mps_flow=True,
            )
            if merged is None:
                rejected_reason = "invalid_layout"
                return False
            if changed:
                new_dim = self._size(merged)
                if (
                    active_max_dim > 0
                    and new_dim > int(active_max_dim)
                ):
                    rejected_reason = "layout_too_large"
                    rejected_dimension = int(new_dim)
                    rejected_layout_blocks = int(len(merged))
                    return False
                if not allow_layout_expansion:
                    rejected_reason = "layout_expanded"
                    return False
                _apply_layout(merged)
            return True

        def _pack(tensor):
            if project_current_support:
                flat = _project_tensor_flat(tensor)
                if flat is None:
                    self.profile_stats["packed_local_davidson"] = {
                        "iterations": int(len(HV)),
                        "dimension": int(self._size(layout)),
                        "basis_size": int(len(V)),
                        "layout_blocks": int(len(layout)),
                        "layout_expansions": int(expansions),
                        "converged": False,
                        "rejected_reason": "projected_pack_failed",
                        "safe_layout_blocks": (
                            safe_layout_blocks
                            if safe_layout_blocks is not None
                            else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                        ),
                        "safe_layout_dimension": safe_layout_dimension,
                        "current_layout_dimension": int(current_layout_dimension),
                        "projected_current_support": True,
                        "projected_truncated_current_support": bool(
                            projected_truncated_current_support
                        ),
                    }
                    return None
                return flat
            if not _ensure_tensor(tensor):
                self.profile_stats["packed_local_davidson"] = {
                    "iterations": int(len(HV)),
                    "dimension": int(self._size(layout)),
                    "basis_size": int(len(V)),
                    "layout_blocks": int(len(layout)),
                    "layout_expansions": int(expansions),
                    "converged": False,
                    "rejected_reason": rejected_reason or "pack_failed",
                    "rejected_dimension": rejected_dimension,
                    "rejected_layout_blocks": rejected_layout_blocks,
                    "safe_layout_blocks": (
                        safe_layout_blocks
                        if safe_layout_blocks is not None
                        else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                    ),
                    "safe_layout_dimension": safe_layout_dimension,
                    "current_layout_dimension": int(current_layout_dimension),
                    "projected_current_support": bool(project_current_support),
                    "projected_truncated_current_support": bool(
                        projected_truncated_current_support
                    ),
                }
                return None
            return self._flatten(tensor, layout)

        def _project_tensor_flat(tensor):
            nonlocal projected_pack_calls
            nonlocal projected_discarded_blocks
            nonlocal projected_discarded_norm_sq
            projected_pack_calls += 1
            dtype = self._block_data_dtype(tensor, v0)
            result = abelian_project_tensor_to_layout_with_stats(
                tensor,
                layout,
                proto=v0,
                dtype=dtype,
            )
            projected_discarded_blocks += int(result.discarded_blocks)
            projected_discarded_norm_sq += float(result.discarded_norm_sq)
            return result.flat

        def _project_flat(tensor, fixed_layout):
            dtype = self._block_data_dtype(tensor, v0)
            return abelian_project_tensor_to_layout(
                tensor,
                fixed_layout,
                proto=v0,
                dtype=dtype,
                extra_policy="ignore",
            )

        def _unpack(vec):
            return self._unflatten(vec, v0, layout)

        def _unpack_sparse(vec, fixed_layout):
            return self._unflatten(
                vec,
                v0,
                fixed_layout,
                drop_zero_blocks=True,
                zero_tol=0.0,
            )

        def _record_flat_solution(vec, energy, *, residual_norm=None, converged=None):
            normalized = abelian_normalize_flat_vector(vec, min_norm=1.0e-12)
            if not normalized.accepted:
                return None
            self.last_packed_davidson_solution_flat = np.asarray(
                normalized.vector,
                dtype=np.asarray(vec).dtype,
            ).copy()
            self.last_packed_davidson_solution_layout = tuple(layout)
            self.last_packed_davidson_solution_energy = energy
            self.last_packed_davidson_solution_residual = (
                None if residual_norm is None else float(residual_norm)
            )
            self.last_packed_davidson_solution_converged = (
                None if converged is None else bool(converged)
            )
            stats = self.profile_stats.setdefault("packed_local_davidson", {})
            stats["flat_solution_available"] = True
            stats["flat_solution_norm_before_normalize"] = float(normalized.norm)
            stats["flat_solution_layout_blocks"] = int(len(layout))
            return self.last_packed_davidson_solution_flat

        def _matvec_packed(vec):
            nonlocal projected_pack_calls
            nonlocal projected_discarded_blocks
            nonlocal rejected_reason
            nonlocal moving_compiled_family_matvec
            nonlocal moving_compact_plan_matvec
            flat_full_layout = (
                allowed_layout_map is not None
                and layout == self._layout_from_map(allowed_layout_map)
            )
            flat_projected_layout = (
                project_current_support
                and bool(self._packed_local_flat_projected_matvec)
            )
            if (
                self._packed_local_family_flat_matvec
                and self.complementary_operator_families is not None
                and not flat_projected_layout
            ):
                flat = None
                if moving_environment is not None:
                    if (
                        moving_compiled_family_matvec is None
                        or moving_compiled_family_matvec.layout != tuple(layout)
                    ):
                        moving_compiled_family_matvec = (
                            moving_environment.compiled_flat_matvec(
                                self,
                                v0,
                                layout,
                            )
                        )
                    if moving_compiled_family_matvec is not None:
                        flat = moving_compiled_family_matvec.matvec(vec)
                if flat is None:
                    flat = self._flat_complementary_family_matvec(vec, v0, layout)
                if flat is not None:
                    return flat
            if self._packed_local_flat_matvec and (flat_full_layout or flat_projected_layout):
                if (
                    moving_environment is not None
                    and flat_full_layout
                    and not flat_projected_layout
                    and bool(
                        moving_environment._option_value(
                            moving_environment.matvec_options,
                            "moving_environment_cpp_compact_plan_matvec",
                            False,
                        )
                    )
                ):
                    if (
                        moving_compact_plan_matvec is None
                        or moving_compact_plan_matvec.layout != tuple(layout)
                    ):
                        moving_compact_plan_matvec = (
                            moving_environment.compact_renormalized_table(
                                self,
                                v0,
                                layout,
                            )
                        )
                    if (
                        moving_compact_plan_matvec is not None
                        and moving_environment._validate_compact_plan_operator(
                            moving_compact_plan_matvec,
                            self,
                            v0,
                            layout,
                            vec,
                        )
                    ):
                        start = time.perf_counter()
                        flat = moving_compact_plan_matvec.matvec(vec)
                        elapsed = float(time.perf_counter() - start)
                        moving_stats = moving_environment.moving_profile_stats
                        moving_stats["compact_plan_matvec_calls"] = int(
                            moving_stats.get("compact_plan_matvec_calls", 0)
                        ) + 1
                        moving_stats["compact_plan_matvec_seconds"] = float(
                            moving_stats.get("compact_plan_matvec_seconds", 0.0)
                        ) + elapsed
                        moving_stats["compact_plan_matvec_last_seconds"] = elapsed
                        return flat
                flat = self._flat_batched_compact_matrix_chain(
                    vec,
                    v0,
                    layout,
                    project_output=bool(flat_projected_layout and not flat_full_layout),
                )
                if flat is not None:
                    if flat_projected_layout and not flat_full_layout:
                        projected_pack_calls += 1
                        flat_stats = self.profile_stats.get(
                            "packed_flat_batched_compact_matrix_chain",
                            {},
                        )
                        projected_discarded_blocks += int(
                            flat_stats.get("last", {}).get("projected_output_blocks", 0)
                        )
                    return flat
                if large_safe_layout and self._packed_local_large_safe_require_flat:
                    rejected_reason = "flat_compact_matvec_unavailable"
                    return None
            hv_tensor = self.matvec(_unpack(vec))
            return _pack(hv_tensor)

        initial_flat_present = initial_flat is not None
        initial_is_current = bool(initial_is_current)
        initial_flat_used = False
        initial_flat_nocopy_used = False
        initial_flat_rejected_reason = ""

        def _annotate_initial_flat_stats(stats):
            if not initial_flat_present:
                return
            prefix = "initial_current_flat" if initial_is_current else "initial_flat_guess"
            stats[f"{prefix}_present"] = True
            stats[f"{prefix}_used"] = bool(initial_flat_used)
            stats[f"{prefix}_nocopy_used"] = bool(initial_flat_nocopy_used)
            if initial_flat_rejected_reason:
                stats[f"{prefix}_rejected_reason"] = (
                    initial_flat_rejected_reason
                )

        pack_start = time.perf_counter()
        v = None
        if initial_flat_present:
            if initial_layout is None:
                initial_flat_rejected_reason = "missing_layout"
            else:
                try:
                    flat_layout = tuple(
                        (tuple(key), tuple(int(dim) for dim in shape))
                        for key, shape in tuple(initial_layout or ())
                    )
                    flat_vec = np.asarray(initial_flat)
                    if flat_layout != tuple(layout):
                        try:
                            flat_vec = abelian_remap_flat_layout(
                                flat_vec,
                                flat_layout,
                                layout,
                            )
                            initial_flat_rejected_reason = ""
                        except Exception as exc:
                            initial_flat_rejected_reason = (
                                f"layout_mismatch:{repr(exc)}"
                            )
                    if initial_flat_rejected_reason:
                        pass
                    elif int(flat_vec.size) != int(dim):
                        initial_flat_rejected_reason = "dimension_mismatch"
                    else:
                        reshaped = flat_vec.reshape(int(dim))
                        target_dtype = np.result_type(flat_vec.dtype, np.complex128)
                        can_reuse_current_flat = (
                            initial_is_current
                            and flat_layout == tuple(layout)
                            and np.dtype(reshaped.dtype) == np.dtype(target_dtype)
                            and bool(reshaped.flags.c_contiguous)
                        )
                        if can_reuse_current_flat:
                            v = reshaped
                            initial_flat_nocopy_used = True
                        else:
                            v = np.asarray(
                                reshaped,
                                dtype=target_dtype,
                            ).copy()
                        initial_flat_used = True
                except Exception as exc:
                    initial_flat_rejected_reason = repr(exc)
        if v is None:
            v = _pack(v0)
        _record_moving_local_phase(
            "packed_local_initial_pack_seconds",
            time.perf_counter() - pack_start,
        )
        if moving_environment is not None and initial_flat_present:
            moving_stats = moving_environment.moving_profile_stats
            prefix = "initial_current_flat" if initial_is_current else "initial_flat_guess"
            moving_stats[f"{prefix}_attempts"] = int(
                moving_stats.get(f"{prefix}_attempts", 0)
            ) + 1
            if initial_flat_used:
                moving_stats[f"{prefix}_used"] = int(
                    moving_stats.get(f"{prefix}_used", 0)
                ) + 1
                if initial_flat_nocopy_used:
                    moving_stats[f"{prefix}_nocopy_used"] = int(
                        moving_stats.get(f"{prefix}_nocopy_used", 0)
                    ) + 1
            else:
                moving_stats[f"{prefix}_rejected"] = int(
                    moving_stats.get(f"{prefix}_rejected", 0)
                ) + 1
                if initial_flat_rejected_reason:
                    moving_stats[f"{prefix}_last_rejected_reason"] = (
                        initial_flat_rejected_reason
                    )
        if v is None:
            return None
        norm = np.linalg.norm(v)
        if norm < 1.0e-12:
            rng = np.random.default_rng(1234)
            v = rng.standard_normal(v.shape).astype(v.dtype, copy=False)
            norm = np.linalg.norm(v)
        if norm < 1.0e-12:
            return None
        cpp_debug_result = None
        if (
            moving_environment is not None
            and not project_current_support
            and (
                (
                    self._packed_local_family_flat_matvec
                    and self.complementary_operator_families is not None
                )
                or bool(
                    moving_environment._option_value(
                        moving_environment.matvec_options,
                        "moving_environment_cpp_compact_plan",
                        False,
                    )
                )
                or bool(
                    moving_environment._option_value(
                        moving_environment.matvec_options,
                        "moving_environment_compact_block_table",
                        False,
                    )
                )
            )
        ):
            cpp_restart_dim = int(active_restart_dim)
            cpp_accept_unconverged = bool(
                moving_environment._option_value(
                    moving_environment.matvec_options,
                    "moving_environment_cpp_accept_unconverged",
                    False,
                )
            )
            cpp_total_start = time.perf_counter()
            cpp_update = None
            cpp_update_result = None
            if return_update:
                cpp_update_result = moving_environment.solve_cpp_davidson_update(
                    self,
                    v0,
                    layout,
                    v,
                    tol=float(tol),
                    max_iter=int(max_iter),
                    restart_dim=cpp_restart_dim,
                    accept_unconverged=cpp_accept_unconverged,
                    direction=update_direction,
                    m_max=update_m_max,
                )
            if cpp_update_result is not None:
                cpp_result, cpp_update = cpp_update_result
            else:
                cpp_result = moving_environment.solve_cpp_davidson(
                    self,
                    v0,
                    layout,
                    v,
                    tol=float(tol),
                    max_iter=int(max_iter),
                    restart_dim=cpp_restart_dim,
                    accept_unconverged=cpp_accept_unconverged,
                )
            _record_moving_local_phase(
                "cpp_davidson_total_seconds",
                time.perf_counter() - cpp_total_start,
            )
            if cpp_result is not None and bool(cpp_result.get("accepted", False)):
                best_vec_cpp = np.asarray(
                    cpp_result.get("vector"),
                    dtype=np.complex128,
                ).reshape(int(self._size(layout)))
                if best_vec_cpp.size:
                    phase_idx = int(np.argmax(np.abs(best_vec_cpp)))
                    phase_ref = complex(best_vec_cpp[phase_idx])
                    if abs(phase_ref) > 1.0e-14:
                        best_vec_cpp = best_vec_cpp * (
                            np.conj(phase_ref) / abs(phase_ref)
                        )
                best_w_cpp = complex(cpp_result.get("energy"))
                best_resid_cpp = float(cpp_result.get("residual_norm", math.inf))
                cpp_solution_valid = True
                if bool(
                    moving_environment._option_value(
                        moving_environment.matvec_options,
                        "moving_environment_cpp_validate_solution",
                        True,
                    )
                ):
                    ref_hv = None
                    table_source = str(cpp_result.get("table_source", ""))
                    if table_source in {
                        "compact_renormalized_table",
                        "compact_plan",
                        "compact_block_table",
                    }:
                        ref_hv = self._flat_batched_compact_matrix_chain(
                            best_vec_cpp,
                            v0,
                            layout,
                        )
                    if ref_hv is not None:
                        ref_resid = float(
                            np.linalg.norm(ref_hv - best_w_cpp * best_vec_cpp)
                        )
                        ref_energy = complex(np.vdot(best_vec_cpp, ref_hv))
                        norm_cpp = float(np.linalg.norm(best_vec_cpp))
                        if norm_cpp > 1.0e-14:
                            ref_energy /= norm_cpp * norm_cpp
                        factor = float(
                            moving_environment._option_value(
                                moving_environment.matvec_options,
                                "moving_environment_cpp_solution_residual_tol_factor",
                                25.0,
                            )
                        )
                        abs_tol = float(
                            moving_environment._option_value(
                                moving_environment.matvec_options,
                                "moving_environment_cpp_solution_residual_abs_tol",
                                1.0e-9,
                            )
                        )
                        limit = max(abs_tol, factor * float(tol))
                        moving_stats = moving_environment.moving_profile_stats
                        moving_stats["cpp_solution_validation_calls"] = int(
                            moving_stats.get("cpp_solution_validation_calls", 0)
                        ) + 1
                        moving_stats["cpp_solution_validation_last_residual"] = ref_resid
                        moving_stats["cpp_solution_validation_last_energy"] = ref_energy
                        moving_stats["cpp_solution_validation_last_limit"] = limit
                        if not np.isfinite(ref_resid) or ref_resid > limit:
                            cpp_solution_valid = False
                            moving_stats["cpp_solution_validation_failures"] = int(
                                moving_stats.get(
                                    "cpp_solution_validation_failures",
                                    0,
                                )
                            ) + 1
                if self._moving_environment_cpp_debug_compare:
                    cpp_debug_result = {
                        "energy": best_w_cpp,
                        "vector": best_vec_cpp,
                        "residual_norm": best_resid_cpp,
                        "iterations": int(cpp_result.get("iterations", 0)),
                        "basis_size": int(cpp_result.get("basis_size", 0)),
                        "restarts": int(cpp_result.get("restarts", 0)),
                        "converged": bool(cpp_result.get("converged", False)),
                    }
                elif cpp_solution_valid:
                    self.profile_stats["packed_local_davidson"] = {
                        "iterations": int(cpp_result.get("iterations", 0)),
                        "dimension": int(best_vec_cpp.size),
                        "basis_size": int(cpp_result.get("basis_size", 0)),
                        "layout_blocks": int(len(layout)),
                        "layout_expansions": int(expansions),
                        "safe_layout_blocks": (
                            safe_layout_blocks
                            if safe_layout_blocks is not None
                            else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                        ),
                        "safe_layout_dimension": safe_layout_dimension,
                        "restarts": int(cpp_result.get("restarts", 0)),
                        "residual_norm": float(best_resid_cpp),
                        "converged": bool(cpp_result.get("converged", False)),
                        "current_layout_dimension": int(current_layout_dimension),
                        "active_max_dim": int(active_max_dim),
                        "active_restart_dim": int(active_restart_dim),
                        "large_safe_layout": bool(large_safe_layout),
                        "large_safe_max_dim": int(self._packed_local_large_safe_max_dim),
                        "projected_current_support": False,
                        "cpp_davidson": True,
                        "cpp_davidson_table_source": str(
                            cpp_result.get("table_source", "")
                        ),
                    }
                    _annotate_initial_flat_stats(
                        self.profile_stats["packed_local_davidson"]
                    )
                    unpack_start = time.perf_counter()
                    flat_cpp = _record_flat_solution(
                        best_vec_cpp,
                        best_w_cpp,
                        residual_norm=best_resid_cpp,
                        converged=bool(cpp_result.get("converged", False)),
                    )
                    if return_flat and flat_cpp is not None:
                        _record_moving_local_phase(
                            "packed_local_final_unpack_seconds",
                            time.perf_counter() - unpack_start,
                        )
                        if return_update and cpp_update is not None:
                            self.last_packed_davidson_solution_update = cpp_update
                            return best_w_cpp, cpp_update
                        return best_w_cpp, flat_cpp
                    best_tensor_cpp = _unpack(
                        best_vec_cpp if flat_cpp is None else flat_cpp
                    )
                    _record_moving_local_phase(
                        "packed_local_final_unpack_seconds",
                        time.perf_counter() - unpack_start,
                    )
                    return best_w_cpp, best_tensor_cpp
        V.append(v / norm)

        T = np.zeros((0, 0), dtype=V[0].dtype)
        best_w = None
        best_vec = V[0]
        best_resid = math.inf
        converged = False
        restarts = 0
        block_preconditioner_cache = {}

        def _block_preconditioner_blocks(current_layout):
            cache_key = current_layout
            cached = block_preconditioner_cache.get(cache_key)
            if cached is not None:
                return cached
            build_start = time.perf_counter()
            max_block_dim = int(self._packed_local_block_preconditioner_max_block_dim)
            max_total_dim = int(self._packed_local_block_preconditioner_max_total_dim)

            def _preconditioner_matvec_flat(basis, basis_layout):
                flat = None
                if self._packed_local_flat_matvec:
                    flat = self._flat_batched_compact_matrix_chain(
                        basis,
                        v0,
                        basis_layout,
                        project_output=bool(project_current_support),
                    )
                if flat is not None:
                    return flat
                image = self.matvec(_unpack_sparse(basis, basis_layout))
                return _project_flat(image, basis_layout)

            result = abelian_build_block_preconditioner_blocks(
                current_layout,
                _preconditioner_matvec_flat,
                max_block_dim=max_block_dim,
                max_total_dim=max_total_dim,
                dtype=complex,
            )
            blocks = result.blocks
            stats = self.profile_stats.setdefault("packed_block_preconditioner", {})
            stats["builds"] = int(stats.get("builds", 0)) + 1
            stats["last_blocks"] = int(len(blocks))
            stats["last_dimension"] = int(result.used_dim)
            stats["last_attempted_blocks"] = int(result.attempted_blocks)
            stats["last_failed_blocks"] = int(result.failed_blocks)
            stats["last_skipped_blocks"] = int(result.skipped_blocks)
            stats["last_columns"] = int(result.columns)
            stats["last_seconds"] = float(time.perf_counter() - build_start)
            stats["seconds"] = float(stats.get("seconds", 0.0)) + float(stats["last_seconds"])
            block_preconditioner_cache[cache_key] = blocks
            return blocks

        def _apply_block_preconditioner(resid, theta, base):
            if not self._packed_local_block_preconditioner:
                return base
            blocks = _block_preconditioner_blocks(layout)
            if not blocks:
                return base
            return abelian_apply_block_preconditioner(resid, theta, base, blocks)

        def _apply_flat_jacobi_preconditioner(resid, theta):
            if not self._packed_local_flat_preconditioner:
                return None
            diagonal = self._flat_jacobi_diagonal(v0, layout)
            if diagonal is None or int(diagonal.size) != int(resid.size):
                return None
            stats = self.profile_stats.setdefault(
                "packed_flat_preconditioner",
                {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
            )
            start = time.perf_counter()
            out = abelian_apply_jacobi_preconditioner(
                resid,
                theta,
                diagonal,
                floor=1.0e-8,
            )
            if out is None:
                return None
            elapsed = time.perf_counter() - start
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
            stats["last_seconds"] = float(elapsed)
            stats["dimension"] = int(resid.size)
            stats["layout_blocks"] = int(len(layout))
            stats["available"] = True
            return out

        for it in range(int(max_iter)):
            hv = _matvec_packed(V[-1])
            if hv is None:
                self.profile_stats["packed_local_davidson"] = {
                    "iterations": int(len(HV)),
                    "dimension": int(self._size(layout)),
                    "basis_size": int(len(V)),
                    "layout_blocks": int(len(layout)),
                    "layout_expansions": int(expansions),
                    "converged": False,
                    "rejected_reason": rejected_reason or "matvec_failed",
                    "safe_layout_blocks": (
                        safe_layout_blocks
                        if safe_layout_blocks is not None
                        else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                    ),
                    "safe_layout_dimension": safe_layout_dimension,
                    "current_layout_dimension": int(current_layout_dimension),
                    "large_safe_layout": bool(large_safe_layout),
                    "large_safe_max_dim": int(self._packed_local_large_safe_max_dim),
                    "projected_current_support": bool(project_current_support),
                    "projected_truncated_current_support": bool(
                        projected_truncated_current_support
                    ),
                }
                return None
            HV.append(hv)

            T = abelian_extend_projected_hamiltonian(T, V, hv)
            ritz = abelian_lowest_ritz_state(T, V, HV)
            if ritz is None:
                break
            resid = ritz.residual
            resid_norm = float(ritz.residual_norm)
            best_w = ritz.energy
            best_vec = ritz.vector
            best_resid = resid_norm
            if resid_norm < float(tol):
                converged = True
                break

            if (
                active_restart_dim > 0
                and len(V) >= int(active_restart_dim)
            ):
                restarted = abelian_restart_basis_from_vector(
                    ritz.vector,
                    min_norm=1.0e-12,
                )
                if restarted is None:
                    break
                V = [restarted]
                HV = []
                T = np.zeros((0, 0), dtype=V[0].dtype)
                restarts += 1
                continue

            q = None
            q = _apply_flat_jacobi_preconditioner(resid, ritz.energy)
            if preconditioner is not None:
                if q is None:
                    q_tensor = preconditioner(_unpack(resid), ritz.energy)
                    if q_tensor is not None:
                        q = _pack(q_tensor)
                        if q is None:
                            self.profile_stats["packed_local_davidson"] = {
                                "iterations": int(len(HV)),
                                "dimension": int(self._size(layout)),
                                "basis_size": int(len(V)),
                                "layout_blocks": int(len(layout)),
                                "layout_expansions": int(expansions),
                                "converged": False,
                                "rejected_reason": rejected_reason or "preconditioner_pack_failed",
                                "safe_layout_blocks": (
                                    safe_layout_blocks
                                    if safe_layout_blocks is not None
                                    else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
                                ),
                                "safe_layout_dimension": safe_layout_dimension,
                                "current_layout_dimension": int(current_layout_dimension),
                                "large_safe_layout": bool(large_safe_layout),
                                "large_safe_max_dim": int(self._packed_local_large_safe_max_dim),
                                "projected_current_support": bool(project_current_support),
                            }
                            return None
            if q is None:
                q = resid * -10.0
            q = _apply_block_preconditioner(resid, ritz.energy, q)

            q, qn = abelian_orthogonalize_candidate(q, V, passes=2, min_norm=1.0e-9)
            if q is None:
                break
            V.append(q)

        if best_w is None:
            return None
        self.profile_stats["packed_local_davidson"] = {
            "iterations": int(len(HV)),
            "dimension": int(best_vec.size),
            "basis_size": int(len(V)),
            "layout_blocks": int(len(layout)),
            "layout_expansions": int(expansions),
            "safe_layout_blocks": (
                safe_layout_blocks
                if safe_layout_blocks is not None
                else (None if allowed_layout_map is None else int(len(allowed_layout_map)))
            ),
            "safe_layout_dimension": safe_layout_dimension,
            "restarts": int(restarts),
            "residual_norm": float(best_resid),
            "converged": bool(converged),
            "current_layout_dimension": int(current_layout_dimension),
            "active_max_dim": int(active_max_dim),
            "active_restart_dim": int(active_restart_dim),
            "large_safe_layout": bool(large_safe_layout),
            "large_safe_max_dim": int(self._packed_local_large_safe_max_dim),
            "projected_current_support": bool(project_current_support),
            "projected_from_safe_layout_too_large": bool(
                projected_from_safe_layout_too_large
            ),
            "projected_truncated_current_support": bool(
                projected_truncated_current_support
            ),
        }
        _annotate_initial_flat_stats(self.profile_stats["packed_local_davidson"])
        if project_current_support:
            self.profile_stats["packed_local_davidson"]["projected_pack_calls"] = int(
                projected_pack_calls
            )
            self.profile_stats["packed_local_davidson"]["projected_discarded_blocks"] = int(
                projected_discarded_blocks
            )
            self.profile_stats["packed_local_davidson"]["projected_discarded_norm"] = float(
                math.sqrt(max(projected_discarded_norm_sq, 0.0))
            )
            self.profile_stats["packed_local_davidson"]["projected_original_blocks"] = int(
                projected_original_blocks
            )
            if projected_retained_blocks is not None:
                self.profile_stats["packed_local_davidson"]["projected_retained_blocks"] = int(
                    projected_retained_blocks
                )
            if projected_retained_norm is not None:
                self.profile_stats["packed_local_davidson"]["projected_retained_norm"] = float(
                    projected_retained_norm
                )
        if self._packed_local_block_preconditioner:
            self.profile_stats["packed_local_davidson"]["block_preconditioner"] = dict(
                self.profile_stats.get("packed_block_preconditioner", {})
            )
        if cpp_debug_result is not None:
            cpp_vec = np.asarray(cpp_debug_result["vector"], dtype=np.complex128)
            py_vec = np.asarray(best_vec, dtype=np.complex128)
            overlap = complex(np.vdot(cpp_vec, py_vec))
            if abs(overlap) > 1.0e-14:
                cpp_vec = cpp_vec * (overlap / abs(overlap))
            energy_delta = complex(cpp_debug_result["energy"] - best_w)
            self.profile_stats["packed_local_davidson"].update(
                {
                    "cpp_debug_compare": True,
                    "cpp_debug_energy": complex(cpp_debug_result["energy"]),
                    "cpp_debug_energy_delta_real": float(np.real(energy_delta)),
                    "cpp_debug_energy_abs_delta": float(abs(energy_delta)),
                    "cpp_debug_residual_norm": float(
                        cpp_debug_result["residual_norm"]
                    ),
                    "cpp_debug_iterations": int(cpp_debug_result["iterations"]),
                    "cpp_debug_basis_size": int(cpp_debug_result["basis_size"]),
                    "cpp_debug_restarts": int(cpp_debug_result["restarts"]),
                    "cpp_debug_converged": bool(cpp_debug_result["converged"]),
                    "cpp_debug_vector_delta": float(np.linalg.norm(cpp_vec - py_vec)),
                    "cpp_debug_overlap_abs": float(abs(overlap)),
                }
            )
            if self._moving_environment_cpp_debug_return_cpp:
                self.profile_stats["packed_local_davidson"][
                    "cpp_debug_returned_cpp"
                ] = True
                flat_cpp = _record_flat_solution(
                    cpp_debug_result["vector"],
                    cpp_debug_result["energy"],
                    residual_norm=cpp_debug_result["residual_norm"],
                    converged=bool(cpp_debug_result["converged"]),
                )
                if return_flat and flat_cpp is not None:
                    return cpp_debug_result["energy"], flat_cpp
                return cpp_debug_result["energy"], _unpack(
                    cpp_debug_result["vector"] if flat_cpp is None else flat_cpp
                )
        projected_rejection_reasons = []
        if project_current_support:
            max_projected_resid = float(self._packed_local_projected_accept_max_residual)
            if max_projected_resid > 0.0 and float(best_resid) > max_projected_resid:
                projected_rejection_reasons.append("projected_residual_too_large")
            min_retained = float(self._packed_local_projected_accept_min_retained_norm)
            if (
                min_retained > 0.0
                and projected_truncated_current_support
                and projected_retained_norm is not None
                and float(projected_retained_norm) < min_retained
            ):
                projected_rejection_reasons.append("projected_retained_norm_too_small")
        if projected_rejection_reasons:
            self.profile_stats["packed_local_davidson"][
                "projected_accept_rejected_reasons"
            ] = list(projected_rejection_reasons)
            if self._packed_local_return_current_on_rejected_projected:
                current_norm = current_tensor.norm()
                if current_norm > 1.0e-12:
                    rayleigh_start = time.perf_counter()
                    current_unit = current_tensor * (1.0 / current_norm)
                    current_image = self.matvec(current_unit)
                    current_energy = current_unit.dot(current_image)
                    self.profile_stats["packed_local_davidson"][
                        "returned_current_state"
                    ] = True
                    self.profile_stats["packed_local_davidson"][
                        "returned_current_seconds"
                    ] = float(time.perf_counter() - rayleigh_start)
                    self.profile_stats["packed_local_davidson"][
                        "returned_current_energy"
                    ] = float(np.real(np.asarray(current_energy).reshape(-1)[0]))
                    self.profile_stats["packed_local_davidson"].pop(
                        "rejected_reason",
                        None,
                    )
                    return current_energy, current_unit
            self.profile_stats["packed_local_davidson"][
                "rejected_reason"
            ] = "projected_accept_rejected"
            return None
        if not converged:
            self.profile_stats["packed_local_davidson"]["rejected_reason"] = "not_converged"
            warm_cap = int(self._packed_local_fallback_warm_start_max_dim)
            if (
                self._packed_local_fallback_warm_start
                and best_vec is not None
                and np.isfinite(best_resid)
                and (warm_cap <= 0 or int(best_vec.size) <= warm_cap)
            ):
                normalized = abelian_normalize_flat_vector(best_vec, min_norm=1.0e-12)
                if normalized.accepted:
                    self.last_packed_davidson_candidate_flat = np.asarray(
                        normalized.vector,
                        dtype=np.asarray(best_vec).dtype,
                    ).copy()
                    self.last_packed_davidson_candidate_layout = tuple(layout)
                    self.last_packed_davidson_candidate = _unpack(
                        self.last_packed_davidson_candidate_flat
                    )
                    self.last_packed_davidson_candidate_energy = best_w
                    self.last_packed_davidson_candidate_residual = float(best_resid)
                    self.profile_stats["packed_local_davidson"]["warm_start_candidate"] = {
                        "available": True,
                        "dimension": int(best_vec.size),
                        "energy": float(np.real(best_w)),
                        "residual_norm": float(best_resid),
                        "flat_norm_before_normalize": float(normalized.norm),
                    }
            accept_unconverged = bool(self._packed_local_accept_unconverged) or (
                project_current_support
                and bool(self._packed_local_accept_projected_unconverged)
            )
            if accept_unconverged:
                self.profile_stats["packed_local_davidson"].pop("rejected_reason", None)
                self.profile_stats["packed_local_davidson"]["accepted_unconverged"] = True
                self.profile_stats["packed_local_davidson"]["accepted_reason"] = (
                    "projected_current_support"
                    if project_current_support
                    else "packed_local_davidson"
                )
                flat = _record_flat_solution(
                    best_vec,
                    best_w,
                    residual_norm=best_resid,
                    converged=False,
                )
                if return_flat and flat is not None:
                    return best_w, flat
                return best_w, _unpack(best_vec if flat is None else flat)
            return None
        flat = _record_flat_solution(
            best_vec,
            best_w,
            residual_norm=best_resid,
            converged=True,
        )
        if return_flat and flat is not None:
            return best_w, flat
        return best_w, _unpack(best_vec if flat is None else flat)

    def _closed_layout(self, proto, max_dim, *, frozen_axes=()):
        _auto, cap, _min_dim = self._dense_policy(max_dim)
        max_dim = cap
        layout_map = {key: shape for key, shape in self._layout(proto)}
        frozen_axes = tuple(int(axis) for axis in frozen_axes)
        frozen_sectors = {
            axis: {key[axis] for key in layout_map}
            for axis in frozen_axes
        }
        dirs = proto.dirs[:]
        allowed_signatures = {
            self._sector_signature(key, dirs)
            for key in layout_map
        }
        for _ in range(4):
            layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
            if self._size(layout) > int(max_dim):
                return None
            data = {}
            rng = np.random.default_rng(123)
            for key, shape in layout:
                data[key] = rng.standard_normal(shape).astype(complex)
            qns = self._qns_from_layout_with_proto(layout, proto)
            trial = self._tensor_from_block_data_like(proto, data, qns, dirs)
            y = self._matvec_generic(trial)
            changed = False
            for key, blk in y.data.items():
                if any(key[axis] not in frozen_sectors[axis] for axis in frozen_axes):
                    return None
                if self._sector_signature(key, dirs) not in allowed_signatures:
                    return None
                if key not in layout_map:
                    layout_map[key] = blk.shape
                    changed = True
                elif layout_map[key] != blk.shape:
                    return None
            if not changed:
                return tuple((key, layout_map[key]) for key in sorted(layout_map))
        return None

    def _layout_is_closed(self, proto, layout):
        layout_shapes = {key: shape for key, shape in layout}
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        rng = np.random.default_rng(123)
        data = {
            key: rng.standard_normal(shape).astype(dtype, copy=False)
            for key, shape in layout
        }
        trial = self._tensor_from_block_data_like(
            proto,
            data,
            self._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
        )
        image = self._matvec_generic(trial)
        allowed_signatures = {
            self._sector_signature(key, proto.dirs)
            for key, _shape in layout
        }
        for key, block in image.data.items():
            if key not in layout_shapes:
                return False
            if tuple(block.shape) != tuple(layout_shapes[key]):
                return False
            if self._sector_signature(key, proto.dirs) not in allowed_signatures:
                return False
        return True

    def dense_matrix(self, proto, max_dim=256, *, allow_layout_expansion=False, frozen_axes=()):
        auto, cap, min_dim = self._dense_policy(max_dim)
        if cap <= 0:
            return None, None
        max_dim = cap
        if allow_layout_expansion:
            layout = self._closed_layout(proto, max_dim, frozen_axes=frozen_axes)
        else:
            layout = self._layout(proto)
            if self._size(layout) > int(max_dim) or not self._layout_is_closed(proto, layout):
                layout = None
        if layout is None:
            return None, None
        dim = self._size(layout)
        if dim <= 0 or dim > int(max_dim):
            return None, None
        if auto and dim < int(min_dim):
            return None, None
        cache_key = layout
        cached = self._dense_cache.get(cache_key)
        if cached is not None:
            return cached, layout

        H = np.zeros((dim, dim), dtype=complex)
        local_layout = AbelianLocalVectorLayout.from_layout(layout, proto=proto)
        qns = [list(axis_qns) for axis_qns in local_layout.qns]
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        for col in range(dim):
            basis = self._tensor_from_block_data_like(
                proto,
                local_layout.basis_data(col, dtype=dtype),
                qns,
                list(local_layout.dirs),
            )
            H[:, col] = self._flatten(self.matvec(basis), layout)
        H = 0.5 * (H + H.conj().T)
        self._dense_cache[cache_key] = H
        return H, layout

    @staticmethod
    def _tensor_token(T):
        return (id(T.data), len(T.data))

    def _action_token(self):
        return (
            self._tensor_token(self.E),
            self._tensor_token(self.W[0]),
            self._tensor_token(self.W[1]),
            self._tensor_token(self.F),
        )

    @staticmethod
    def _block_index(T, axes):
        axes = tuple(axes)
        idx = {}
        if len(axes) == 1:
            ax0 = int(axes[0])
            for key, blk in T.data.items():
                bucket = (key[ax0],)
                entries = idx.get(bucket)
                if entries is None:
                    idx[bucket] = [(key, blk)]
                else:
                    entries.append((key, blk))
            return idx
        if len(axes) == 2:
            ax0 = int(axes[0])
            ax1 = int(axes[1])
            for key, blk in T.data.items():
                bucket = (key[ax0], key[ax1])
                entries = idx.get(bucket)
                if entries is None:
                    idx[bucket] = [(key, blk)]
                else:
                    entries.append((key, blk))
            return idx
        for key, blk in T.data.items():
            bucket = tuple(key[int(ax)] for ax in axes)
            entries = idx.get(bucket)
            if entries is None:
                idx[bucket] = [(key, blk)]
            else:
                entries.append((key, blk))
        return idx

    def _build_action_plan(self, A):
        layout = self._layout(A)
        cache_key = (self._action_token(), layout)
        if cache_key in self._action_cache:
            return self._action_cache[cache_key]

        e_by_ket_l = self._block_index(self.E, (2,))
        w1_by_left_in = self._block_index(self.W[0], (0, 3))
        w2_by_left_in = self._block_index(self.W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(self.F, (0, 2))

        entries = []
        out_shapes = {}
        dtype_args = []
        eq = "aij,jkxy,abux,bcvy,clk->iluv"

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._action_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._action_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._action_cache[cache_key] = None
                        return None
                    for w2_key, w2_blk in w2_by_left_in.get((w1_key[1], p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._action_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._action_cache[cache_key] = None
                                return None
                            try:
                                path = np.einsum_path(
                                    eq, e_blk, a_blk, w1_blk, w2_blk, f_blk,
                                    optimize="greedy",
                                )[0]
                            except ValueError:
                                self._action_cache[cache_key] = None
                                return None
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                e_blk.shape[1],
                                f_blk.shape[1],
                                w1_blk.shape[2],
                                w2_blk.shape[2],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._action_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            dtype_args.extend((e_blk.dtype, w1_blk.dtype, w2_blk.dtype, f_blk.dtype))
                            entries.append((a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, path))
                            if len(entries) > self._compiled_action_cap:
                                self._action_cache[cache_key] = None
                                return None

        if not entries:
            plan = ((), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._action_cache[cache_key] = plan
            return plan

        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (entries, out_shapes, out_qns, A.dirs[:], dtype)
        self._action_cache[cache_key] = plan
        return plan

    def _matvec_compiled(self, A):
        plan = self._build_action_plan(A)
        if plan is None:
            return None
        entries, out_shapes, out_qns, out_dirs, dtype = plan
        out = {key: np.zeros(shape, dtype=dtype) for key, shape in out_shapes.items()}
        eq = "aij,jkxy,abux,bcvy,clk->iluv"
        for a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, path in entries:
            out[out_key] += np.einsum(
                eq, e_blk, A.data[a_key], w1_blk, w2_blk, f_blk,
                optimize=path,
            )
        return self._tensor_from_block_data_like(A, out, out_qns, out_dirs)

    def _matvec_generic(self, A):
        # A is BlockTensor with indices (Left, Right, Phys_L, Phys_R)
        # E: (MPO_L, MPS_L, MPS_L')
        # W: (MPO_L, MPO_R, Phys_Out, Phys_In)
        # F: (MPO_R, MPS_R, MPS_R')

        if isinstance(A, AbelianSiteTensorData):
            R = abelian_tensor_data_tensordot(self.E, A, ([2], [0]))
            T2 = abelian_tensor_data_tensordot(R, self.W[0], ([0, 3], [0, 3]))
            T3 = abelian_tensor_data_tensordot(T2, self.W[1], ([3, 2], [0, 3]))
            T4 = abelian_tensor_data_tensordot(T3, self.F, ([3, 1], [0, 2]))
            return abelian_transpose_tensor_data(
                T4,
                (0, 3, 1, 2),
                carrier=AbelianSiteTensorData,
            )

        # 1. Contract E with A
        # E indices: (a, i, j) -> (MPO, Bra, Ket)
        # A indices: (j, k, s1, s2) -> (Left, Right, PhysL, PhysR)
        # Contract E[Ket] with A[Left] -> E[2] with A[0]
        # Result R: (a, i, k, s1, s2)
        R = tensordot(self.E, A, axes=([2], [0]))

        # 2. Contract R with W1 (Left Site)
        # W1: (a, b, s1', s1) -> (Left, Right, Out, In)
        # R: (a, i, k, s1, s2)
        # Contract R[MPO_L]=R[0] with W1[Left]=W1[0]
        # Contract R[Phys1]=R[3] with W1[In]=W1[3]
        T2 = tensordot(R, self.W[0], axes=([0, 3], [0, 3]))
        # T2: (i, k, s2, b, s1') -> (Bra_L, Right, PhysR, MPO_R, PhysL_Out)

        # 3. Contract T2 with W2 (Right Site)
        # W2: (b, c, s2', s2) -> (Left, Right, Out, In)
        # T2: (i, k, s2, b, s1')
        # Contract T2[MPO_R]=T2[3] with W2[Left]=W2[0]
        # Contract T2[PhysR]=T2[2] with W2[In]=W2[3]
        T3 = tensordot(T2, self.W[1], axes=([3, 2], [0, 3]))
        # T3: (i, k, s1', c, s2') -> (Bra_L, Right, PhysL_Out, MPO_R, PhysR_Out)

        # 4. Contract T3 with F
        # F: (c, k, l) -> (MPO_R, Bra_R, Ket_R)
        # contract T3[Right]=T3[1] (which corresponds to A's Right/Ket)
        # with F[Ket]=F[2].
        # And T3[MPO_R]=T3[3] with F[MPO]=F[0].
        T4 = tensordot(T3, self.F, axes=([3, 1], [0, 2]))
        # Result indices: (i, s1', s2', l) -> (Bra_L, PhysL_Out, PhysR_Out, Bra_R)

        # 5. Transpose to match A structure (Left, Right, PhysL, PhysR)
        # Current: (Bra_L, PhysL, PhysR, Bra_R) -> (0, 1, 2, 3)
        # Target: (Bra_L, Bra_R, PhysL, PhysR) -> (0, 3, 1, 2)
        A_new = T4.transpose(0, 3, 1, 2)

        return A_new

    @staticmethod
    def _matrix_chain_record_shape(shapes, key, shape):
        shape = tuple(int(dim) for dim in shape)
        old_shape = shapes.get(key)
        if old_shape is not None and old_shape != shape:
            return False
        shapes[key] = shape
        return True

    def _build_generic_matrix_chain_plan(self, A):
        layout = self._layout(A)
        cache_key = (self._action_token(), layout, "generic_matrix_chain")
        cached = self._generic_matrix_chain_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        build_start = time.perf_counter()
        e_shapes = {key: block.shape for key, block in self.E.data.items()}
        a_shapes = {key: block.shape for key, block in A.data.items()}
        w1_shapes = {key: block.shape for key, block in self.W[0].data.items()}
        w2_shapes = {key: block.shape for key, block in self.W[1].data.items()}
        f_shapes = {key: block.shape for key, block in self.F.data.items()}

        a_by_left = defaultdict(list)
        for key in a_shapes:
            a_by_left[key[0]].append(key)
        w1_by_left_in = defaultdict(list)
        for key in w1_shapes:
            w1_by_left_in[(key[0], key[3])].append(key)
        w2_by_left_in = defaultdict(list)
        for key in w2_shapes:
            w2_by_left_in[(key[0], key[3])].append(key)
        f_by_mpo_ket = defaultdict(list)
        for key in f_shapes:
            f_by_mpo_ket[(key[0], key[2])].append(key)

        r_entries = []
        r_shapes = {}
        for e_key in sorted(e_shapes):
            e_shape = tuple(e_shapes[e_key])
            if len(e_shape) != 3:
                self._generic_matrix_chain_plan_cache[cache_key] = None
                return None
            for a_key in sorted(a_by_left.get(e_key[2], ())):
                a_shape = tuple(a_shapes[a_key])
                if len(a_shape) != 4 or e_shape[2] != a_shape[0]:
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                out_key = (e_key[0], e_key[1], a_key[1], a_key[2], a_key[3])
                out_shape = (
                    e_shape[0],
                    e_shape[1],
                    a_shape[1],
                    a_shape[2],
                    a_shape[3],
                )
                if not self._matrix_chain_record_shape(r_shapes, out_key, out_shape):
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                r_entries.append((e_key, a_key, out_key))

        t2_entries = []
        t2_shapes = {}
        for r_key in sorted(r_shapes):
            r_shape = tuple(r_shapes[r_key])
            for w_key in sorted(w1_by_left_in.get((r_key[0], r_key[3]), ())):
                w_shape = tuple(w1_shapes[w_key])
                if len(w_shape) != 4 or r_shape[0] != w_shape[0] or r_shape[3] != w_shape[3]:
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                out_key = (r_key[1], r_key[2], r_key[4], w_key[1], w_key[2])
                out_shape = (
                    r_shape[1],
                    r_shape[2],
                    r_shape[4],
                    w_shape[1],
                    w_shape[2],
                )
                if not self._matrix_chain_record_shape(t2_shapes, out_key, out_shape):
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                t2_entries.append((r_key, w_key, out_key))

        t3_entries = []
        t3_shapes = {}
        for t2_key in sorted(t2_shapes):
            t2_shape = tuple(t2_shapes[t2_key])
            for w_key in sorted(w2_by_left_in.get((t2_key[3], t2_key[2]), ())):
                w_shape = tuple(w2_shapes[w_key])
                if len(w_shape) != 4 or t2_shape[3] != w_shape[0] or t2_shape[2] != w_shape[3]:
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                out_key = (t2_key[0], t2_key[1], t2_key[4], w_key[1], w_key[2])
                out_shape = (
                    t2_shape[0],
                    t2_shape[1],
                    t2_shape[4],
                    w_shape[1],
                    w_shape[2],
                )
                if not self._matrix_chain_record_shape(t3_shapes, out_key, out_shape):
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                t3_entries.append((t2_key, w_key, out_key))

        out_entries = []
        out_shapes = {}
        for t3_key in sorted(t3_shapes):
            t3_shape = tuple(t3_shapes[t3_key])
            for f_key in sorted(f_by_mpo_ket.get((t3_key[3], t3_key[1]), ())):
                f_shape = tuple(f_shapes[f_key])
                if len(f_shape) != 3 or t3_shape[3] != f_shape[0] or t3_shape[1] != f_shape[2]:
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                out_key = (t3_key[0], f_key[1], t3_key[2], t3_key[4])
                out_shape = (
                    t3_shape[0],
                    f_shape[1],
                    t3_shape[2],
                    t3_shape[4],
                )
                if not self._matrix_chain_record_shape(out_shapes, out_key, out_shape):
                    self._generic_matrix_chain_plan_cache[cache_key] = None
                    return None
                out_entries.append((t3_key, f_key, out_key))

        dtype_args = (
            [block.dtype for block in self.E.data.values()]
            + [block.dtype for block in A.data.values()]
            + [block.dtype for block in self.W[0].data.values()]
            + [block.dtype for block in self.W[1].data.values()]
            + [block.dtype for block in self.F.data.values()]
        )
        dtype = np.result_type(*(dtype_args or [complex]))
        out_layout = tuple((key, out_shapes[key]) for key in sorted(out_shapes))
        out_qns = self._qns_from_layout_with_proto(out_layout, A)
        plan = {
            "r_entries": tuple(r_entries),
            "r_shapes": r_shapes,
            "t2_entries": tuple(t2_entries),
            "t2_shapes": t2_shapes,
            "t3_entries": tuple(t3_entries),
            "t3_shapes": t3_shapes,
            "out_entries": tuple(out_entries),
            "out_shapes": out_shapes,
            "out_qns": out_qns if out_qns else A.qns[:],
            "out_dirs": A.dirs[:],
            "dtype": dtype,
        }
        self._generic_matrix_chain_plan_cache[cache_key] = plan
        self._record_plan_profile(
            "generic_matrix_chain",
            time.perf_counter() - build_start,
            r_entries=int(len(r_entries)),
            t2_entries=int(len(t2_entries)),
            t3_entries=int(len(t3_entries)),
            out_entries=int(len(out_entries)),
            output_blocks=int(len(out_shapes)),
        )
        return plan

    def _filter_matrix_chain_plan_to_layout(self, base, target_layout, fallback_qns):
        """Trim a generic matrix-chain plan to outputs present in target_layout."""
        target_keys = {key for key, _shape in target_layout}
        kept_out_entries = tuple(
            entry for entry in base["out_entries"] if entry[2] in target_keys
        )
        kept_out_keys = {entry[2] for entry in kept_out_entries}
        needed_t3 = {entry[0] for entry in kept_out_entries}

        kept_t3_entries = tuple(
            entry for entry in base["t3_entries"] if entry[2] in needed_t3
        )
        kept_t3_keys = {entry[2] for entry in kept_t3_entries}
        needed_t2 = {entry[0] for entry in kept_t3_entries}

        kept_t2_entries = tuple(
            entry for entry in base["t2_entries"] if entry[2] in needed_t2
        )
        kept_t2_keys = {entry[2] for entry in kept_t2_entries}
        needed_r = {entry[0] for entry in kept_t2_entries}

        kept_r_entries = tuple(
            entry for entry in base["r_entries"] if entry[2] in needed_r
        )
        kept_r_keys = {entry[2] for entry in kept_r_entries}

        out_shapes = {
            key: base["out_shapes"][key]
            for key in sorted(kept_out_keys)
        }
        out_qns = self._qns_from_layout(
            tuple((key, out_shapes[key]) for key in sorted(out_shapes))
        )
        return {
            "r_entries": kept_r_entries,
            "r_shapes": {
                key: base["r_shapes"][key]
                for key in sorted(kept_r_keys)
            },
            "t2_entries": kept_t2_entries,
            "t2_shapes": {
                key: base["t2_shapes"][key]
                for key in sorted(kept_t2_keys)
            },
            "t3_entries": kept_t3_entries,
            "t3_shapes": {
                key: base["t3_shapes"][key]
                for key in sorted(kept_t3_keys)
            },
            "out_entries": kept_out_entries,
            "out_shapes": out_shapes,
            "out_qns": out_qns if out_qns else fallback_qns,
            "out_dirs": base["out_dirs"],
            "dtype": base["dtype"],
            "_projection_stats": {
                "full_r_entries": int(len(base["r_entries"])),
                "full_t2_entries": int(len(base["t2_entries"])),
                "full_t3_entries": int(len(base["t3_entries"])),
                "full_out_entries": int(len(base["out_entries"])),
                "full_output_blocks": int(len(base["out_shapes"])),
                "kept_r_entries": int(len(kept_r_entries)),
                "kept_t2_entries": int(len(kept_t2_entries)),
                "kept_t3_entries": int(len(kept_t3_entries)),
                "kept_out_entries": int(len(kept_out_entries)),
                "kept_output_blocks": int(len(out_shapes)),
            },
        }

    @staticmethod
    def _matrix_chain_e_a(e_blk, a_blk):
        na, ni, nj = e_blk.shape
        nj_a, nk, nx, ny = a_blk.shape
        if nj != nj_a:
            raise ValueError("matrix-chain E-A shape mismatch")
        left = np.reshape(e_blk, (na * ni, nj))
        right = np.reshape(a_blk, (nj, nk * nx * ny))
        return (left @ right).reshape(na, ni, nk, nx, ny)

    @staticmethod
    def _matrix_chain_r_w1(r_blk, w_blk):
        na, ni, nk, nx, ny = r_blk.shape
        na_w, nb, nu, nx_w = w_blk.shape
        if na != na_w or nx != nx_w:
            raise ValueError("matrix-chain R-W1 shape mismatch")
        left = np.ascontiguousarray(r_blk.transpose(1, 2, 4, 0, 3)).reshape(
            ni * nk * ny,
            na * nx,
        )
        right = np.ascontiguousarray(w_blk.transpose(0, 3, 1, 2)).reshape(
            na * nx,
            nb * nu,
        )
        return (left @ right).reshape(ni, nk, ny, nb, nu)

    @staticmethod
    def _matrix_chain_t2_w2(t2_blk, w_blk):
        ni, nk, ny, nb, nu = t2_blk.shape
        nb_w, nc, nv, ny_w = w_blk.shape
        if nb != nb_w or ny != ny_w:
            raise ValueError("matrix-chain T2-W2 shape mismatch")
        left = np.ascontiguousarray(t2_blk.transpose(0, 1, 4, 3, 2)).reshape(
            ni * nk * nu,
            nb * ny,
        )
        right = np.ascontiguousarray(w_blk.transpose(0, 3, 1, 2)).reshape(
            nb * ny,
            nc * nv,
        )
        return (left @ right).reshape(ni, nk, nu, nc, nv)

    @staticmethod
    def _matrix_chain_t3_f(t3_blk, f_blk):
        ni, nk, nu, nc, nv = t3_blk.shape
        nc_f, nl, nk_f = f_blk.shape
        if nc != nc_f or nk != nk_f:
            raise ValueError("matrix-chain T3-F shape mismatch")
        left = np.ascontiguousarray(t3_blk.transpose(0, 2, 4, 3, 1)).reshape(
            ni * nu * nv,
            nc * nk,
        )
        right = np.ascontiguousarray(f_blk.transpose(0, 2, 1)).reshape(
            nc * nk,
            nl,
        )
        return (left @ right).reshape(ni, nu, nv, nl).transpose(0, 3, 1, 2)

    @staticmethod
    def _matrix_chain_zero_data(shapes, dtype):
        return {
            key: np.zeros(shape, dtype=dtype)
            for key, shape in shapes.items()
        }

    @staticmethod
    def _compact_plan_input_work(plan):
        groups = plan.get("a_groups", {})
        shapes = groups.get("shapes", ())
        blocks = groups.get("blocks", ())
        work = 0
        for shape, group_blocks in zip(shapes, blocks):
            work += int(np.prod(shape, dtype=int)) * int(len(group_blocks))
        return int(work)

    def _use_parallel_compact_kernel(self, plan):
        if not self._batched_compact_matrix_chain_compiled_parallel_kernel:
            return False, self._compact_plan_input_work(plan)
        work = self._compact_plan_input_work(plan)
        min_work = int(self._batched_compact_matrix_chain_compiled_parallel_min_work)
        if min_work > 0 and work < min_work:
            return False, work
        available = (
            _numba_batched_matrix_chain_e_a_accum_parallel is not None
            and _numba_batched_matrix_chain_r_w_accum_parallel is not None
            and _numba_batched_matrix_chain_t2_w_accum_parallel is not None
            and _numba_batched_matrix_chain_t3_f_accum_parallel is not None
            and all(spec is not None for spec in plan["batched_r_specs"])
            and all(spec is not None for spec in plan["batched_t2_specs"])
            and all(spec is not None for spec in plan["batched_t3_specs"])
            and all(spec is not None for spec in plan["batched_out_specs"])
        )
        return bool(available), work

    def _matvec_generic_matrix_chain(self, A):
        plan = self._build_generic_matrix_chain_plan(A)
        if plan is None:
            return None
        start = time.perf_counter()
        dtype = plan["dtype"]
        r_data = self._matrix_chain_zero_data(plan["r_shapes"], dtype)
        for e_key, a_key, out_key in plan["r_entries"]:
            r_data[out_key] += self._matrix_chain_e_a(self.E.data[e_key], A.data[a_key])

        t2_data = self._matrix_chain_zero_data(plan["t2_shapes"], dtype)
        for r_key, w_key, out_key in plan["t2_entries"]:
            t2_data[out_key] += self._matrix_chain_r_w1(r_data[r_key], self.W[0].data[w_key])

        t3_data = self._matrix_chain_zero_data(plan["t3_shapes"], dtype)
        for t2_key, w_key, out_key in plan["t3_entries"]:
            t3_data[out_key] += self._matrix_chain_t2_w2(t2_data[t2_key], self.W[1].data[w_key])

        out_data = self._matrix_chain_zero_data(plan["out_shapes"], dtype)
        for t3_key, f_key, out_key in plan["out_entries"]:
            out_data[out_key] += self._matrix_chain_t3_f(t3_data[t3_key], self.F.data[f_key])

        stats = self.profile_stats.setdefault(
            "generic_matrix_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["plan_cache_entries"] = int(len(self._generic_matrix_chain_plan_cache))
        stats["last"] = {
            "r_entries": int(len(plan["r_entries"])),
            "t2_entries": int(len(plan["t2_entries"])),
            "t3_entries": int(len(plan["t3_entries"])),
            "out_entries": int(len(plan["out_entries"])),
            "output_blocks": int(len(plan["out_shapes"])),
        }
        return self._tensor_from_block_data_like(
            A,
            out_data,
            plan["out_qns"],
            plan["out_dirs"],
        )

    @staticmethod
    def _compact_entry_table(entries, *maps):
        return tuple(
            tuple(int(maps[index][key]) for index, key in enumerate(entry))
            for entry in entries
        )

    @staticmethod
    def _compact_shape_groups(shapes):
        group_by_shape = {}
        group_shapes = []
        group_blocks = []
        block_group = []
        block_pos = []
        for block_index, shape in enumerate(shapes):
            shape = tuple(int(dim) for dim in shape)
            group_index = group_by_shape.get(shape)
            if group_index is None:
                group_index = len(group_shapes)
                group_by_shape[shape] = group_index
                group_shapes.append(shape)
                group_blocks.append([])
            block_group.append(group_index)
            block_pos.append(len(group_blocks[group_index]))
            group_blocks[group_index].append(int(block_index))
        return {
            "shapes": tuple(group_shapes),
            "blocks": tuple(tuple(blocks) for blocks in group_blocks),
            "block_group": np.asarray(block_group, dtype=np.int64),
            "block_pos": np.asarray(block_pos, dtype=np.int64),
        }

    @staticmethod
    def _batched_compact_entry_groups(entries, key_func):
        grouped = defaultdict(list)
        for entry in entries:
            entry = tuple(int(value) for value in entry)
            grouped[key_func(entry)].append(entry)
        return tuple(
            np.asarray(grouped[key], dtype=np.int64)
            for key in sorted(grouped)
        )

    @staticmethod
    def _batched_compact_static_stacks(entry_groups, blocks, column):
        stacks = []
        for entries in entry_groups:
            if len(entries) == 0:
                stacks.append(None)
            else:
                stacks.append(
                    np.stack([blocks[int(index)] for index in entries[:, int(column)]], axis=0)
                )
        return tuple(stacks)

    @staticmethod
    def _batched_compact_position_specs(entry_groups, input_groups, input_column, output_groups, output_column):
        specs = []
        input_block_group = input_groups["block_group"]
        input_block_pos = input_groups["block_pos"]
        output_block_group = output_groups["block_group"]
        output_block_pos = output_groups["block_pos"]
        for entries in entry_groups:
            if len(entries) == 0:
                specs.append(None)
                continue
            input_indices = entries[:, int(input_column)]
            output_indices = entries[:, int(output_column)]
            input_group_ids = input_block_group[input_indices]
            output_group_ids = output_block_group[output_indices]
            input_group = int(input_group_ids[0])
            output_group = int(output_group_ids[0])
            if np.any(input_group_ids != input_group) or np.any(output_group_ids != output_group):
                specs.append(None)
                continue
            specs.append(
                {
                    "input_group": input_group,
                    "input_pos": np.ascontiguousarray(input_block_pos[input_indices], dtype=np.int64),
                    "output_group": output_group,
                    "output_pos": np.ascontiguousarray(output_block_pos[output_indices], dtype=np.int64),
                }
            )
        return tuple(specs)

    @staticmethod
    def _cython_compact_payload(plan):
        scratch = plan.setdefault("_scratch", {})
        payload = scratch.get("cython_compact_payload")
        if payload is not None:
            return payload

        def _complex_stacks(name):
            return tuple(
                None if stack is None else np.ascontiguousarray(stack, dtype=np.complex128)
                for stack in plan[name]
            )

        def _specs(name, input_groups, output_groups):
            out = []
            for spec in plan[name]:
                if spec is None:
                    return None
                input_group = int(spec["input_group"])
                output_group = int(spec["output_group"])
                out.append(
                    (
                        input_group,
                        np.ascontiguousarray(spec["input_pos"], dtype=np.int64),
                        output_group,
                        np.ascontiguousarray(spec["output_pos"], dtype=np.int64),
                        np.ascontiguousarray(
                            tuple(int(v) for v in input_groups["shapes"][input_group])
                            + tuple(int(v) for v in output_groups["shapes"][output_group]),
                            dtype=np.int64,
                        ),
                    )
                )
            return tuple(out)

        payload = {
            "r_e": _complex_stacks("batched_r_e_stacks"),
            "t2_w": _complex_stacks("batched_t2_w_stacks"),
            "t3_w": _complex_stacks("batched_t3_w_stacks"),
            "out_f": _complex_stacks("batched_out_f_stacks"),
            "r_specs": _specs("batched_r_specs", plan["a_groups"], plan["r_groups"]),
            "t2_specs": _specs("batched_t2_specs", plan["r_groups"], plan["t2_groups"]),
            "t3_specs": _specs("batched_t3_specs", plan["t2_groups"], plan["t3_groups"]),
            "out_specs": _specs("batched_out_specs", plan["t3_groups"], plan["out_groups"]),
        }
        if any(payload[key] is None for key in ("r_specs", "t2_specs", "t3_specs", "out_specs")):
            return None
        scratch["cython_compact_payload"] = payload
        return payload

    @staticmethod
    def _compact_stage_specs_payload(plan):
        scratch = plan.setdefault("_scratch", {})
        payload = scratch.get("compact_stage_specs_payload")
        if payload is not None:
            return payload

        def _specs(name, input_groups, output_groups):
            out = []
            for spec in plan[name]:
                if spec is None:
                    return None
                input_group = int(spec["input_group"])
                output_group = int(spec["output_group"])
                out.append(
                    (
                        input_group,
                        np.ascontiguousarray(spec["input_pos"], dtype=np.int64),
                        output_group,
                        np.ascontiguousarray(spec["output_pos"], dtype=np.int64),
                        np.ascontiguousarray(
                            tuple(int(v) for v in input_groups["shapes"][input_group])
                            + tuple(int(v) for v in output_groups["shapes"][output_group]),
                            dtype=np.int64,
                        ),
                    )
                )
            return tuple(out)

        payload = {
            "r_specs": _specs("batched_r_specs", plan["a_groups"], plan["r_groups"]),
            "t2_specs": _specs("batched_t2_specs", plan["r_groups"], plan["t2_groups"]),
            "t3_specs": _specs("batched_t3_specs", plan["t2_groups"], plan["t3_groups"]),
            "out_specs": _specs("batched_out_specs", plan["t3_groups"], plan["out_groups"]),
        }
        if any(payload[key] is None for key in ("r_specs", "t2_specs", "t3_specs", "out_specs")):
            return None
        scratch["compact_stage_specs_payload"] = payload
        return payload

    @staticmethod
    def _cython_arena_info(plan):
        scratch = plan.setdefault("_scratch", {})
        info = scratch.get("cython_arena_info")
        if info is not None:
            return info

        def _group_info(groups):
            offsets = []
            sizes = []
            total = 0
            for shape, blocks in zip(groups["shapes"], groups["blocks"]):
                size = int(np.prod(shape, dtype=int))
                offsets.append(total)
                sizes.append(size)
                total += size * int(len(blocks))
            return (
                np.ascontiguousarray(offsets, dtype=np.int64),
                np.ascontiguousarray(sizes, dtype=np.int64),
                int(total),
            )

        info = {
            "a": _group_info(plan["a_groups"]),
            "r": _group_info(plan["r_groups"]),
            "t2": _group_info(plan["t2_groups"]),
            "t3": _group_info(plan["t3_groups"]),
            "out": _group_info(plan["out_groups"]),
        }
        scratch["cython_arena_info"] = info
        return info

    @staticmethod
    def _cython_arena_buffers(plan, dtype):
        scratch = plan.setdefault("_scratch", {})
        info = HamiltonianMultiplyU1._cython_arena_info(plan)
        buffers = scratch.get("cython_arena_buffers")
        expected = {
            name: int(values[2])
            for name, values in info.items()
        }
        if (
            buffers is None
            or any(
                buffers[name].dtype != np.dtype(dtype)
                or int(buffers[name].size) != int(size)
                for name, size in expected.items()
            )
        ):
            buffers = {
                name: np.zeros(size, dtype=dtype)
                for name, size in expected.items()
            }
            scratch["cython_arena_buffers"] = buffers
        else:
            for buf in buffers.values():
                buf.fill(0)
        return info, buffers

    @staticmethod
    def _batched_compact_r_left_stacks(stacks):
        out = []
        for stack in stacks:
            if stack is None:
                out.append(None)
                continue
            batch, na, ni, nj = stack.shape
            out.append(stack.reshape(batch, na * ni, nj))
        return tuple(out)

    @staticmethod
    def _batched_compact_w_right_stacks(stacks):
        out = []
        for stack in stacks:
            if stack is None:
                out.append(None)
                continue
            batch, na, nb, nu, nx = stack.shape
            out.append(
                np.ascontiguousarray(stack.transpose(0, 1, 4, 2, 3)).reshape(
                    batch,
                    na * nx,
                    nb * nu,
                )
            )
        return tuple(out)

    @staticmethod
    def _batched_compact_f_right_stacks(stacks):
        out = []
        for stack in stacks:
            if stack is None:
                out.append(None)
                continue
            batch, nc, nl, nk = stack.shape
            out.append(
                np.ascontiguousarray(stack.transpose(0, 1, 3, 2)).reshape(
                    batch,
                    nc * nk,
                    nl,
                )
            )
        return tuple(out)

    def _build_compact_matrix_chain_plan(self, A, *, target_layout=None):
        layout = self._layout(A)
        target_layout_key = None if target_layout is None else tuple(target_layout)
        cache_key = (
            self._action_token(),
            layout,
            "compact_matrix_chain",
            target_layout_key,
        )
        cached = self._compact_matrix_chain_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        build_start = time.perf_counter()
        base = self._build_generic_matrix_chain_plan(A)
        if base is None:
            self._compact_matrix_chain_plan_cache[cache_key] = None
            return None
        if target_layout_key is not None:
            base = self._filter_matrix_chain_plan_to_layout(
                base,
                target_layout_key,
                A.qns[:],
            )

        e_keys = tuple(sorted(self.E.data))
        a_keys = tuple(key for key, _shape in layout)
        w1_keys = tuple(sorted(self.W[0].data))
        w2_keys = tuple(sorted(self.W[1].data))
        f_keys = tuple(sorted(self.F.data))
        r_keys = tuple(sorted(base["r_shapes"]))
        t2_keys = tuple(sorted(base["t2_shapes"]))
        t3_keys = tuple(sorted(base["t3_shapes"]))
        out_keys = tuple(sorted(base["out_shapes"]))

        e_index = {key: i for i, key in enumerate(e_keys)}
        a_index = {key: i for i, key in enumerate(a_keys)}
        w1_index = {key: i for i, key in enumerate(w1_keys)}
        w2_index = {key: i for i, key in enumerate(w2_keys)}
        f_index = {key: i for i, key in enumerate(f_keys)}
        r_index = {key: i for i, key in enumerate(r_keys)}
        t2_index = {key: i for i, key in enumerate(t2_keys)}
        t3_index = {key: i for i, key in enumerate(t3_keys)}
        out_index = {key: i for i, key in enumerate(out_keys)}

        plan = {
            "e_blocks": tuple(np.ascontiguousarray(self.E.data[key]) for key in e_keys),
            "a_keys": a_keys,
            "w1_blocks": tuple(np.ascontiguousarray(self.W[0].data[key]) for key in w1_keys),
            "w2_blocks": tuple(np.ascontiguousarray(self.W[1].data[key]) for key in w2_keys),
            "f_blocks": tuple(np.ascontiguousarray(self.F.data[key]) for key in f_keys),
            "r_entries": self._compact_entry_table(
                base["r_entries"],
                e_index,
                a_index,
                r_index,
            ),
            "t2_entries": self._compact_entry_table(
                base["t2_entries"],
                r_index,
                w1_index,
                t2_index,
            ),
            "t3_entries": self._compact_entry_table(
                base["t3_entries"],
                t2_index,
                w2_index,
                t3_index,
            ),
            "out_entries": self._compact_entry_table(
                base["out_entries"],
                t3_index,
                f_index,
                out_index,
            ),
            "r_shapes": tuple(tuple(base["r_shapes"][key]) for key in r_keys),
            "t2_shapes": tuple(tuple(base["t2_shapes"][key]) for key in t2_keys),
            "t3_shapes": tuple(tuple(base["t3_shapes"][key]) for key in t3_keys),
            "out_keys": out_keys,
            "out_shapes": tuple(tuple(base["out_shapes"][key]) for key in out_keys),
            "out_qns": base["out_qns"],
            "out_dirs": base["out_dirs"],
            "dtype": base["dtype"],
            "projected_plan": target_layout_key is not None,
            "projection_stats": base.get("_projection_stats", {}),
            "_scratch": {},
        }
        e_shapes = tuple(tuple(block.shape) for block in plan["e_blocks"])
        a_shapes = tuple(tuple(shape) for _key, shape in layout)
        w1_shapes = tuple(tuple(block.shape) for block in plan["w1_blocks"])
        w2_shapes = tuple(tuple(block.shape) for block in plan["w2_blocks"])
        f_shapes = tuple(tuple(block.shape) for block in plan["f_blocks"])
        a_groups = self._compact_shape_groups(a_shapes)
        r_groups = self._compact_shape_groups(plan["r_shapes"])
        t2_groups = self._compact_shape_groups(plan["t2_shapes"])
        t3_groups = self._compact_shape_groups(plan["t3_shapes"])
        out_groups = self._compact_shape_groups(plan["out_shapes"])
        plan.update(
            {
                "a_groups": a_groups,
                "r_groups": r_groups,
                "t2_groups": t2_groups,
                "t3_groups": t3_groups,
                "out_groups": out_groups,
                "batched_r_entries": self._batched_compact_entry_groups(
                    plan["r_entries"],
                    lambda entry: (
                        e_shapes[entry[0]],
                        a_shapes[entry[1]],
                        plan["r_shapes"][entry[2]],
                    ),
                ),
                "batched_t2_entries": self._batched_compact_entry_groups(
                    plan["t2_entries"],
                    lambda entry: (
                        plan["r_shapes"][entry[0]],
                        w1_shapes[entry[1]],
                        plan["t2_shapes"][entry[2]],
                    ),
                ),
                "batched_t3_entries": self._batched_compact_entry_groups(
                    plan["t3_entries"],
                    lambda entry: (
                        plan["t2_shapes"][entry[0]],
                        w2_shapes[entry[1]],
                        plan["t3_shapes"][entry[2]],
                    ),
                ),
                "batched_out_entries": self._batched_compact_entry_groups(
                    plan["out_entries"],
                    lambda entry: (
                        plan["t3_shapes"][entry[0]],
                        f_shapes[entry[1]],
                        plan["out_shapes"][entry[2]],
                    ),
                ),
            }
        )
        plan.update(
            {
                "batched_r_e_stacks": self._batched_compact_static_stacks(
                    plan["batched_r_entries"],
                    plan["e_blocks"],
                    0,
                ),
                "batched_t2_w_stacks": self._batched_compact_static_stacks(
                    plan["batched_t2_entries"],
                    plan["w1_blocks"],
                    1,
                ),
                "batched_t3_w_stacks": self._batched_compact_static_stacks(
                    plan["batched_t3_entries"],
                    plan["w2_blocks"],
                    1,
                ),
                "batched_out_f_stacks": self._batched_compact_static_stacks(
                    plan["batched_out_entries"],
                    plan["f_blocks"],
                    1,
                ),
            }
        )
        plan.update(
            {
                "batched_r_e_left_stacks": self._batched_compact_r_left_stacks(
                    plan["batched_r_e_stacks"],
                ),
                "batched_t2_w_right_stacks": self._batched_compact_w_right_stacks(
                    plan["batched_t2_w_stacks"],
                ),
                "batched_t3_w_right_stacks": self._batched_compact_w_right_stacks(
                    plan["batched_t3_w_stacks"],
                ),
                "batched_out_f_right_stacks": self._batched_compact_f_right_stacks(
                    plan["batched_out_f_stacks"],
                ),
            }
        )
        plan.update(
            {
                "batched_r_specs": self._batched_compact_position_specs(
                    plan["batched_r_entries"],
                    a_groups,
                    1,
                    r_groups,
                    2,
                ),
                "batched_t2_specs": self._batched_compact_position_specs(
                    plan["batched_t2_entries"],
                    r_groups,
                    0,
                    t2_groups,
                    2,
                ),
                "batched_t3_specs": self._batched_compact_position_specs(
                    plan["batched_t3_entries"],
                    t2_groups,
                    0,
                    t3_groups,
                    2,
                ),
                "batched_out_specs": self._batched_compact_position_specs(
                    plan["batched_out_entries"],
                    t3_groups,
                    0,
                    out_groups,
                    2,
                ),
            }
        )
        self._compact_matrix_chain_plan_cache[cache_key] = plan
        self._record_plan_profile(
            "compact_matrix_chain",
            time.perf_counter() - build_start,
            r_entries=int(len(plan["r_entries"])),
            t2_entries=int(len(plan["t2_entries"])),
            t3_entries=int(len(plan["t3_entries"])),
            out_entries=int(len(plan["out_entries"])),
            output_blocks=int(len(plan["out_keys"])),
            projected_plan=bool(plan["projected_plan"]),
            **plan["projection_stats"],
        )
        return plan

    @staticmethod
    def _compact_buffers(plan, name, shapes, dtype):
        scratch = plan.setdefault("_scratch", {})
        buffers = scratch.get(name)
        if (
            buffers is None
            or len(buffers) != len(shapes)
            or any(buf.dtype != np.dtype(dtype) for buf in buffers)
        ):
            buffers = [np.zeros(shape, dtype=dtype) for shape in shapes]
            scratch[name] = buffers
        else:
            for buf in buffers:
                buf.fill(0)
        return buffers

    @staticmethod
    def _compact_group_buffers(plan, name, groups, dtype, *, zero=True):
        scratch = plan.setdefault("_scratch", {})
        shapes = tuple(groups["shapes"])
        blocks = tuple(groups["blocks"])
        buffers = scratch.get(name)
        expected = tuple((len(blocks[i]),) + tuple(shape) for i, shape in enumerate(shapes))
        if buffers is None or len(buffers) != len(expected) or any(
            tuple(buf.shape) != shape or buf.dtype != np.dtype(dtype)
            for buf, shape in zip(buffers, expected)
        ):
            buffers = [np.zeros(shape, dtype=dtype) for shape in expected]
            scratch[name] = buffers
        elif zero:
            for buf in buffers:
                buf.fill(0)
        return buffers

    @staticmethod
    def _pack_compact_group_blocks(buffers, groups, keys, data):
        block_group = groups["block_group"]
        block_pos = groups["block_pos"]
        for block_index, key in enumerate(keys):
            try:
                block = data[key]
            except KeyError:
                return False
            buffers[int(block_group[block_index])][int(block_pos[block_index])][...] = block
        return True

    @staticmethod
    def _take_compact_group_blocks(buffers, groups, block_indices):
        block_indices = np.asarray(block_indices, dtype=np.int64)
        if block_indices.size == 0:
            return None
        group_ids = groups["block_group"][block_indices]
        group_id = int(group_ids[0])
        if np.any(group_ids != group_id):
            return np.stack(
                [
                    buffers[int(groups["block_group"][idx])][int(groups["block_pos"][idx])]
                    for idx in block_indices
                ],
                axis=0,
            )
        return buffers[group_id][groups["block_pos"][block_indices]]

    @staticmethod
    def _scatter_compact_group_blocks(buffers, groups, block_indices, values):
        block_indices = np.asarray(block_indices, dtype=np.int64)
        if block_indices.size == 0:
            return
        group_ids = groups["block_group"][block_indices]
        group_id = int(group_ids[0])
        positions = groups["block_pos"][block_indices]
        if np.any(group_ids != group_id):
            for idx, value in zip(block_indices, values):
                buffers[int(groups["block_group"][idx])][int(groups["block_pos"][idx])] += value
            return
        np.add.at(buffers[group_id], positions, values)

    def _matvec_compact_matrix_chain(self, A):
        plan = self._build_compact_matrix_chain_plan(A)
        if plan is None:
            return None
        start = time.perf_counter()
        dtype = plan["dtype"]
        try:
            a_blocks = tuple(np.asarray(A.data[key]) for key in plan["a_keys"])
        except KeyError:
            return None
        r_data = self._compact_buffers(plan, "r", plan["r_shapes"], dtype)
        t2_data = self._compact_buffers(plan, "t2", plan["t2_shapes"], dtype)
        t3_data = self._compact_buffers(plan, "t3", plan["t3_shapes"], dtype)
        out_data = [np.zeros(shape, dtype=dtype) for shape in plan["out_shapes"]]

        e_blocks = plan["e_blocks"]
        w1_blocks = plan["w1_blocks"]
        w2_blocks = plan["w2_blocks"]
        f_blocks = plan["f_blocks"]
        for e_i, a_i, r_i in plan["r_entries"]:
            r_data[r_i] += self._matrix_chain_e_a(e_blocks[e_i], a_blocks[a_i])
        for r_i, w_i, t2_i in plan["t2_entries"]:
            t2_data[t2_i] += self._matrix_chain_r_w1(r_data[r_i], w1_blocks[w_i])
        for t2_i, w_i, t3_i in plan["t3_entries"]:
            t3_data[t3_i] += self._matrix_chain_t2_w2(t2_data[t2_i], w2_blocks[w_i])
        for t3_i, f_i, out_i in plan["out_entries"]:
            out_data[out_i] += self._matrix_chain_t3_f(t3_data[t3_i], f_blocks[f_i])

        stats = self.profile_stats.setdefault(
            "compact_matrix_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["plan_cache_entries"] = int(len(self._compact_matrix_chain_plan_cache))
        stats["last"] = {
            "r_entries": int(len(plan["r_entries"])),
            "t2_entries": int(len(plan["t2_entries"])),
            "t3_entries": int(len(plan["t3_entries"])),
            "out_entries": int(len(plan["out_entries"])),
            "output_blocks": int(len(plan["out_keys"])),
        }
        return self._tensor_from_block_data_like(
            A,
            {key: out_data[i] for i, key in enumerate(plan["out_keys"])},
            plan["out_qns"],
            plan["out_dirs"],
        )

    def _matvec_batched_compact_matrix_chain(self, A):
        plan = self._build_compact_matrix_chain_plan(A)
        if plan is None:
            return None
        start = time.perf_counter()
        dtype = plan["dtype"]
        max_entries = int(self._batched_compact_matrix_chain_max_batch_entries)

        a_groups = plan["a_groups"]
        r_groups = plan["r_groups"]
        t2_groups = plan["t2_groups"]
        t3_groups = plan["t3_groups"]
        out_groups = plan["out_groups"]
        a_data = self._compact_group_buffers(plan, "batched_a", a_groups, dtype, zero=False)
        if not self._pack_compact_group_blocks(a_data, a_groups, plan["a_keys"], A.data):
            return None
        r_data = self._compact_group_buffers(plan, "batched_r", r_groups, dtype)
        t2_data = self._compact_group_buffers(plan, "batched_t2", t2_groups, dtype)
        t3_data = self._compact_group_buffers(plan, "batched_t3", t3_groups, dtype)
        out_data = self._compact_group_buffers(plan, "batched_out", out_groups, dtype)

        parallel_compiled_kernel, parallel_work = self._use_parallel_compact_kernel(plan)
        specs_available = (
            all(spec is not None for spec in plan["batched_r_specs"])
            and all(spec is not None for spec in plan["batched_t2_specs"])
            and all(spec is not None for spec in plan["batched_t3_specs"])
            and all(spec is not None for spec in plan["batched_out_specs"])
        )
        numba_compiled_kernel = bool(
            self._batched_compact_matrix_chain_compiled_kernel
            and not parallel_compiled_kernel
            and _numba_batched_matrix_chain_e_a_accum is not None
            and _numba_batched_matrix_chain_r_w_accum is not None
            and _numba_batched_matrix_chain_t2_w_accum is not None
            and _numba_batched_matrix_chain_t3_f_accum is not None
            and specs_available
        )
        cython_compiled_kernel = bool(
            self._batched_compact_matrix_chain_compiled_kernel
            and not parallel_compiled_kernel
            and (
                self._batched_compact_matrix_chain_cython_kernel
                or not numba_compiled_kernel
            )
            and _packed_cython is not None
            and getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            and getattr(_packed_cython, "run_batched_matrix_chain", None) is not None
            and np.dtype(dtype) == np.dtype(np.complex128)
            and specs_available
        )
        cython_payload = None
        if cython_compiled_kernel:
            cython_payload = self._cython_compact_payload(plan)
            if cython_payload is None:
                cython_compiled_kernel = False
        compiled_kernel = bool(numba_compiled_kernel and not cython_compiled_kernel)
        try:
            if parallel_compiled_kernel:
                for group_index, spec in enumerate(plan["batched_r_specs"]):
                    _numba_batched_matrix_chain_e_a_accum_parallel(
                        plan["batched_r_e_stacks"][group_index],
                        a_data[int(spec["input_group"])],
                        spec["input_pos"],
                        r_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t2_specs"]):
                    _numba_batched_matrix_chain_r_w_accum_parallel(
                        r_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t2_w_stacks"][group_index],
                        t2_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t3_specs"]):
                    _numba_batched_matrix_chain_t2_w_accum_parallel(
                        t2_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t3_w_stacks"][group_index],
                        t3_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_out_specs"]):
                    _numba_batched_matrix_chain_t3_f_accum_parallel(
                        t3_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_out_f_stacks"][group_index],
                        out_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )

            if cython_compiled_kernel:
                _packed_cython.run_batched_matrix_chain(
                    cython_payload["r_e"],
                    cython_payload["t2_w"],
                    cython_payload["t3_w"],
                    cython_payload["out_f"],
                    cython_payload["r_specs"],
                    cython_payload["t2_specs"],
                    cython_payload["t3_specs"],
                    cython_payload["out_specs"],
                    a_data,
                    r_data,
                    t2_data,
                    t3_data,
                    out_data,
                )

            if compiled_kernel:
                for group_index, spec in enumerate(plan["batched_r_specs"]):
                    _numba_batched_matrix_chain_e_a_accum(
                        plan["batched_r_e_stacks"][group_index],
                        a_data[int(spec["input_group"])],
                        spec["input_pos"],
                        r_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t2_specs"]):
                    _numba_batched_matrix_chain_r_w_accum(
                        r_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t2_w_stacks"][group_index],
                        t2_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t3_specs"]):
                    _numba_batched_matrix_chain_t2_w_accum(
                        t2_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t3_w_stacks"][group_index],
                        t3_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_out_specs"]):
                    _numba_batched_matrix_chain_t3_f_accum(
                        t3_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_out_f_stacks"][group_index],
                        out_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )

            if not compiled_kernel and not cython_compiled_kernel and not parallel_compiled_kernel:
                for group_index, group in enumerate(plan["batched_r_entries"]):
                    static_e_stack = plan["batched_r_e_stacks"][group_index]
                    static_e_left = plan["batched_r_e_left_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        e_stack = static_e_stack[start_index:start_index + len(entries)]
                        e_left = static_e_left[start_index:start_index + len(entries)]
                        a_stack = self._take_compact_group_blocks(a_data, a_groups, entries[:, 1])
                        batch, na, ni, nj = e_stack.shape
                        _batch_a, _nj_a, nk, nx, ny = a_stack.shape
                        right = a_stack.reshape(batch, nj, nk * nx * ny)
                        values = np.matmul(e_left, right).reshape(batch, na, ni, nk, nx, ny)
                        self._scatter_compact_group_blocks(r_data, r_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_t2_entries"]):
                    static_w_stack = plan["batched_t2_w_stacks"][group_index]
                    static_w_right = plan["batched_t2_w_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        r_stack = self._take_compact_group_blocks(r_data, r_groups, entries[:, 0])
                        w_stack = static_w_stack[start_index:start_index + len(entries)]
                        w_right = static_w_right[start_index:start_index + len(entries)]
                        batch, na, ni, nk, nx, ny = r_stack.shape
                        _batch_w, _na_w, nb, nu, _nx_w = w_stack.shape
                        left = np.ascontiguousarray(
                            r_stack.transpose(0, 2, 3, 5, 1, 4)
                        ).reshape(batch, ni * nk * ny, na * nx)
                        values = np.matmul(left, w_right).reshape(batch, ni, nk, ny, nb, nu)
                        self._scatter_compact_group_blocks(t2_data, t2_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_t3_entries"]):
                    static_w_stack = plan["batched_t3_w_stacks"][group_index]
                    static_w_right = plan["batched_t3_w_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        t2_stack = self._take_compact_group_blocks(t2_data, t2_groups, entries[:, 0])
                        w_stack = static_w_stack[start_index:start_index + len(entries)]
                        w_right = static_w_right[start_index:start_index + len(entries)]
                        batch, ni, nk, ny, nb, nu = t2_stack.shape
                        _batch_w, _nb_w, nc, nv, _ny_w = w_stack.shape
                        left = np.ascontiguousarray(
                            t2_stack.transpose(0, 1, 2, 5, 4, 3)
                        ).reshape(batch, ni * nk * nu, nb * ny)
                        values = np.matmul(left, w_right).reshape(batch, ni, nk, nu, nc, nv)
                        self._scatter_compact_group_blocks(t3_data, t3_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_out_entries"]):
                    static_f_stack = plan["batched_out_f_stacks"][group_index]
                    static_f_right = plan["batched_out_f_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        t3_stack = self._take_compact_group_blocks(t3_data, t3_groups, entries[:, 0])
                        f_stack = static_f_stack[start_index:start_index + len(entries)]
                        f_right = static_f_right[start_index:start_index + len(entries)]
                        batch, ni, nk, nu, nc, nv = t3_stack.shape
                        _batch_f, _nc_f, nl, _nk_f = f_stack.shape
                        left = np.ascontiguousarray(
                            t3_stack.transpose(0, 1, 3, 5, 4, 2)
                        ).reshape(batch, ni * nu * nv, nc * nk)
                        values = np.matmul(left, f_right).reshape(batch, ni, nu, nv, nl)
                        values = values.transpose(0, 1, 4, 2, 3)
                        self._scatter_compact_group_blocks(out_data, out_groups, entries[:, 2], values)
        except MemoryError:
            return None

        stats = self.profile_stats.setdefault(
            "batched_compact_matrix_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["plan_cache_entries"] = int(len(self._compact_matrix_chain_plan_cache))
        stats["last"] = {
            "r_entry_groups": int(len(plan["batched_r_entries"])),
            "t2_entry_groups": int(len(plan["batched_t2_entries"])),
            "t3_entry_groups": int(len(plan["batched_t3_entries"])),
            "out_entry_groups": int(len(plan["batched_out_entries"])),
            "r_entries": int(len(plan["r_entries"])),
            "t2_entries": int(len(plan["t2_entries"])),
            "t3_entries": int(len(plan["t3_entries"])),
            "out_entries": int(len(plan["out_entries"])),
            "output_blocks": int(len(plan["out_keys"])),
            "max_batch_entries": int(max_entries),
            "compiled_kernel": bool(compiled_kernel or parallel_compiled_kernel),
            "compiled_kernel_mode": (
                "parallel"
                if parallel_compiled_kernel
                else ("cython" if cython_compiled_kernel else ("serial" if compiled_kernel else "blas"))
            ),
            "parallel_work": int(parallel_work),
            "parallel_min_work": int(
                self._batched_compact_matrix_chain_compiled_parallel_min_work
            ),
            "numba_threads": (
                int(_numba_get_num_threads())
                if parallel_compiled_kernel and _numba_get_num_threads is not None
                else None
            ),
        }
        out_block_group = out_groups["block_group"]
        out_block_pos = out_groups["block_pos"]
        return self._tensor_from_block_data_like(
            A,
            {
                key: out_data[int(out_block_group[i])][int(out_block_pos[i])].copy()
                for i, key in enumerate(plan["out_keys"])
            },
            plan["out_qns"],
            plan["out_dirs"],
        )

    def _flat_batched_compact_matrix_chain(self, vec, proto, layout, *, project_output=False):
        """Apply the compact matrix-chain action directly to a packed vector."""
        if not self._packed_local_flat_matvec:
            return None
        layout = tuple(layout)
        offsets, total_dim = self._layout_offsets(layout)
        vec = np.asarray(vec)
        if int(vec.size) != int(total_dim):
            return None
        dtype = np.result_type(vec.dtype, self._local_action_dtype(proto))
        proto_full = self._zero_proto_from_layout(proto, layout, dtype)
        plan = self._build_compact_matrix_chain_plan(
            proto_full,
            target_layout=layout if project_output else None,
        )
        if plan is None:
            return None
        if tuple(plan["a_keys"]) != tuple(key for key, _shape in layout):
            return None
        layout_shapes = {key: tuple(shape) for key, shape in layout}
        start = time.perf_counter()
        max_entries = int(self._batched_compact_matrix_chain_max_batch_entries)

        a_groups = plan["a_groups"]
        r_groups = plan["r_groups"]
        t2_groups = plan["t2_groups"]
        t3_groups = plan["t3_groups"]
        out_groups = plan["out_groups"]
        parallel_compiled_kernel, parallel_work = self._use_parallel_compact_kernel(plan)
        cython_arena_kernel = bool(
            self._packed_local_cython_arena
            and
            self._batched_compact_matrix_chain_compiled_kernel
            and not parallel_compiled_kernel
            and _packed_cython is not None
            and getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            and getattr(_packed_cython, "run_batched_matrix_chain_arenas", None) is not None
            and np.dtype(dtype) == np.dtype(np.complex128)
            and all(spec is not None for spec in plan["batched_r_specs"])
            and all(spec is not None for spec in plan["batched_t2_specs"])
            and all(spec is not None for spec in plan["batched_t3_specs"])
            and all(spec is not None for spec in plan["batched_out_specs"])
        )
        if cython_arena_kernel:
            cython_payload = self._cython_compact_payload(plan)
            if cython_payload is not None:
                info, buffers = self._cython_arena_buffers(plan, np.complex128)
                a_offsets, a_sizes, _a_total = info["a"]
                r_offsets, r_sizes, _r_total = info["r"]
                t2_offsets, t2_sizes, _t2_total = info["t2"]
                t3_offsets, t3_sizes, _t3_total = info["t3"]
                out_offsets, out_sizes, _out_total = info["out"]
                a_arena = buffers["a"]
                for block_index, key in enumerate(plan["a_keys"]):
                    pos, n = offsets[key]
                    group = int(a_groups["block_group"][block_index])
                    block_pos = int(a_groups["block_pos"][block_index])
                    base = int(a_offsets[group]) + int(block_pos) * int(a_sizes[group])
                    a_arena[base:base + n] = vec[pos:pos + n]
                try:
                    _packed_cython.run_batched_matrix_chain_arenas(
                        cython_payload["r_e"],
                        cython_payload["t2_w"],
                        cython_payload["t3_w"],
                        cython_payload["out_f"],
                        cython_payload["r_specs"],
                        cython_payload["t2_specs"],
                        cython_payload["t3_specs"],
                        cython_payload["out_specs"],
                        buffers["a"],
                        a_offsets,
                        a_sizes,
                        buffers["r"],
                        r_offsets,
                        r_sizes,
                        buffers["t2"],
                        t2_offsets,
                        t2_sizes,
                        buffers["t3"],
                        t3_offsets,
                        t3_sizes,
                        buffers["out"],
                        out_offsets,
                        out_sizes,
                    )
                except MemoryError:
                    return None

                out = np.zeros(total_dim, dtype=dtype)
                out_block_group = out_groups["block_group"]
                out_block_pos = out_groups["block_pos"]
                projected_output_blocks = 0
                projected_output_dim = 0
                for out_index, key in enumerate(plan["out_keys"]):
                    offset = offsets.get(key)
                    if offset is None:
                        if not project_output:
                            return None
                        projected_output_blocks += 1
                        projected_output_dim += int(np.prod(plan["out_shapes"][out_index], dtype=int))
                        continue
                    pos, n = offset
                    if tuple(plan["out_shapes"][out_index]) != layout_shapes[key]:
                        return None
                    group = int(out_block_group[out_index])
                    block_pos = int(out_block_pos[out_index])
                    base = int(out_offsets[group]) + int(block_pos) * int(out_sizes[group])
                    out[pos:pos + n] = buffers["out"][base:base + n]

                stats = self.profile_stats.setdefault(
                    "packed_flat_batched_compact_matrix_chain",
                    {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
                )
                elapsed = time.perf_counter() - start
                stats["calls"] = int(stats.get("calls", 0)) + 1
                stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
                stats["last_seconds"] = float(elapsed)
                stats["plan_cache_entries"] = int(len(self._compact_matrix_chain_plan_cache))
                stats["last"] = {
                    "dimension": int(total_dim),
                    "r_entries": int(len(plan["r_entries"])),
                    "t2_entries": int(len(plan["t2_entries"])),
                    "t3_entries": int(len(plan["t3_entries"])),
                    "out_entries": int(len(plan["out_entries"])),
                    "output_blocks": int(len(plan["out_keys"])),
                    "max_batch_entries": int(max_entries),
                    "compiled_kernel": True,
                    "compiled_kernel_mode": "cython",
                    "compiled_kernel_backend": "arena",
                    "parallel_work": int(parallel_work),
                    "parallel_min_work": int(
                        self._batched_compact_matrix_chain_compiled_parallel_min_work
                    ),
                    "project_output": bool(project_output),
                    "projected_plan": bool(plan.get("projected_plan", False)),
                    "projected_output_blocks": int(projected_output_blocks),
                    "projected_output_dim": int(projected_output_dim),
                    **plan.get("projection_stats", {}),
                    "numba_threads": None,
                }
                return out
        a_data = self._compact_group_buffers(plan, "packed_flat_a", a_groups, dtype, zero=False)
        for block_index, key in enumerate(plan["a_keys"]):
            pos, n = offsets[key]
            group = int(a_groups["block_group"][block_index])
            block_pos = int(a_groups["block_pos"][block_index])
            a_data[group][block_pos][...] = vec[pos:pos + n].reshape(layout_shapes[key])

        r_data = self._compact_group_buffers(plan, "packed_flat_r", r_groups, dtype)
        t2_data = self._compact_group_buffers(plan, "packed_flat_t2", t2_groups, dtype)
        t3_data = self._compact_group_buffers(plan, "packed_flat_t3", t3_groups, dtype)
        out_data = self._compact_group_buffers(plan, "packed_flat_out", out_groups, dtype)

        specs_available = (
            all(spec is not None for spec in plan["batched_r_specs"])
            and all(spec is not None for spec in plan["batched_t2_specs"])
            and all(spec is not None for spec in plan["batched_t3_specs"])
            and all(spec is not None for spec in plan["batched_out_specs"])
        )
        numba_compiled_kernel = bool(
            self._batched_compact_matrix_chain_compiled_kernel
            and not parallel_compiled_kernel
            and _numba_batched_matrix_chain_e_a_accum is not None
            and _numba_batched_matrix_chain_r_w_accum is not None
            and _numba_batched_matrix_chain_t2_w_accum is not None
            and _numba_batched_matrix_chain_t3_f_accum is not None
            and specs_available
        )
        cython_compiled_kernel = bool(
            self._batched_compact_matrix_chain_compiled_kernel
            and not parallel_compiled_kernel
            and (
                self._batched_compact_matrix_chain_cython_kernel
                or not numba_compiled_kernel
            )
            and _packed_cython is not None
            and getattr(_packed_cython, "CYTHON_AVAILABLE", False)
            and getattr(_packed_cython, "run_batched_matrix_chain", None) is not None
            and np.dtype(dtype) == np.dtype(np.complex128)
            and specs_available
        )
        cython_payload = None
        if cython_compiled_kernel:
            cython_payload = self._cython_compact_payload(plan)
            if cython_payload is None:
                cython_compiled_kernel = False
        compiled_kernel = bool(numba_compiled_kernel and not cython_compiled_kernel)
        try:
            if parallel_compiled_kernel:
                for group_index, spec in enumerate(plan["batched_r_specs"]):
                    _numba_batched_matrix_chain_e_a_accum_parallel(
                        plan["batched_r_e_stacks"][group_index],
                        a_data[int(spec["input_group"])],
                        spec["input_pos"],
                        r_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t2_specs"]):
                    _numba_batched_matrix_chain_r_w_accum_parallel(
                        r_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t2_w_stacks"][group_index],
                        t2_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t3_specs"]):
                    _numba_batched_matrix_chain_t2_w_accum_parallel(
                        t2_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t3_w_stacks"][group_index],
                        t3_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_out_specs"]):
                    _numba_batched_matrix_chain_t3_f_accum_parallel(
                        t3_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_out_f_stacks"][group_index],
                        out_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )

            if cython_compiled_kernel:
                _packed_cython.run_batched_matrix_chain(
                    cython_payload["r_e"],
                    cython_payload["t2_w"],
                    cython_payload["t3_w"],
                    cython_payload["out_f"],
                    cython_payload["r_specs"],
                    cython_payload["t2_specs"],
                    cython_payload["t3_specs"],
                    cython_payload["out_specs"],
                    a_data,
                    r_data,
                    t2_data,
                    t3_data,
                    out_data,
                )

            if compiled_kernel:
                for group_index, spec in enumerate(plan["batched_r_specs"]):
                    _numba_batched_matrix_chain_e_a_accum(
                        plan["batched_r_e_stacks"][group_index],
                        a_data[int(spec["input_group"])],
                        spec["input_pos"],
                        r_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t2_specs"]):
                    _numba_batched_matrix_chain_r_w_accum(
                        r_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t2_w_stacks"][group_index],
                        t2_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_t3_specs"]):
                    _numba_batched_matrix_chain_t2_w_accum(
                        t2_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_t3_w_stacks"][group_index],
                        t3_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )
                for group_index, spec in enumerate(plan["batched_out_specs"]):
                    _numba_batched_matrix_chain_t3_f_accum(
                        t3_data[int(spec["input_group"])],
                        spec["input_pos"],
                        plan["batched_out_f_stacks"][group_index],
                        out_data[int(spec["output_group"])],
                        spec["output_pos"],
                    )

            if not compiled_kernel and not cython_compiled_kernel and not parallel_compiled_kernel:
                for group_index, group in enumerate(plan["batched_r_entries"]):
                    static_e_stack = plan["batched_r_e_stacks"][group_index]
                    static_e_left = plan["batched_r_e_left_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        e_stack = static_e_stack[start_index:start_index + len(entries)]
                        e_left = static_e_left[start_index:start_index + len(entries)]
                        a_stack = self._take_compact_group_blocks(a_data, a_groups, entries[:, 1])
                        batch, na, ni, nj = e_stack.shape
                        _batch_a, _nj_a, nk, nx, ny = a_stack.shape
                        right = a_stack.reshape(batch, nj, nk * nx * ny)
                        values = np.matmul(e_left, right).reshape(batch, na, ni, nk, nx, ny)
                        self._scatter_compact_group_blocks(r_data, r_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_t2_entries"]):
                    static_w_stack = plan["batched_t2_w_stacks"][group_index]
                    static_w_right = plan["batched_t2_w_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        r_stack = self._take_compact_group_blocks(r_data, r_groups, entries[:, 0])
                        w_stack = static_w_stack[start_index:start_index + len(entries)]
                        w_right = static_w_right[start_index:start_index + len(entries)]
                        batch, na, ni, nk, nx, ny = r_stack.shape
                        _batch_w, _na_w, nb, nu, _nx_w = w_stack.shape
                        left = np.ascontiguousarray(
                            r_stack.transpose(0, 2, 3, 5, 1, 4)
                        ).reshape(batch, ni * nk * ny, na * nx)
                        values = np.matmul(left, w_right).reshape(batch, ni, nk, ny, nb, nu)
                        self._scatter_compact_group_blocks(t2_data, t2_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_t3_entries"]):
                    static_w_stack = plan["batched_t3_w_stacks"][group_index]
                    static_w_right = plan["batched_t3_w_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        t2_stack = self._take_compact_group_blocks(t2_data, t2_groups, entries[:, 0])
                        w_stack = static_w_stack[start_index:start_index + len(entries)]
                        w_right = static_w_right[start_index:start_index + len(entries)]
                        batch, ni, nk, ny, nb, nu = t2_stack.shape
                        _batch_w, _nb_w, nc, nv, _ny_w = w_stack.shape
                        left = np.ascontiguousarray(
                            t2_stack.transpose(0, 1, 2, 5, 4, 3)
                        ).reshape(batch, ni * nk * nu, nb * ny)
                        values = np.matmul(left, w_right).reshape(batch, ni, nk, nu, nc, nv)
                        self._scatter_compact_group_blocks(t3_data, t3_groups, entries[:, 2], values)

                for group_index, group in enumerate(plan["batched_out_entries"]):
                    static_f_stack = plan["batched_out_f_stacks"][group_index]
                    static_f_right = plan["batched_out_f_right_stacks"][group_index]
                    for start_index in range(0, int(len(group)), max_entries):
                        entries = group[start_index:start_index + max_entries]
                        t3_stack = self._take_compact_group_blocks(t3_data, t3_groups, entries[:, 0])
                        f_stack = static_f_stack[start_index:start_index + len(entries)]
                        f_right = static_f_right[start_index:start_index + len(entries)]
                        batch, ni, nk, nu, nc, nv = t3_stack.shape
                        _batch_f, _nc_f, nl, _nk_f = f_stack.shape
                        left = np.ascontiguousarray(
                            t3_stack.transpose(0, 1, 3, 5, 4, 2)
                        ).reshape(batch, ni * nu * nv, nc * nk)
                        values = np.matmul(left, f_right).reshape(batch, ni, nu, nv, nl)
                        values = values.transpose(0, 1, 4, 2, 3)
                        self._scatter_compact_group_blocks(out_data, out_groups, entries[:, 2], values)
        except MemoryError:
            return None

        out = np.zeros(total_dim, dtype=dtype)
        out_block_group = out_groups["block_group"]
        out_block_pos = out_groups["block_pos"]
        projected_output_blocks = 0
        projected_output_dim = 0
        for out_index, key in enumerate(plan["out_keys"]):
            offset = offsets.get(key)
            if offset is None:
                if not project_output:
                    return None
                projected_output_blocks += 1
                projected_output_dim += int(np.prod(plan["out_shapes"][out_index], dtype=int))
                continue
            pos, n = offset
            if tuple(plan["out_shapes"][out_index]) != layout_shapes[key]:
                return None
            block = out_data[int(out_block_group[out_index])][int(out_block_pos[out_index])]
            out[pos:pos + n] = block.reshape(-1)

        stats = self.profile_stats.setdefault(
            "packed_flat_batched_compact_matrix_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["plan_cache_entries"] = int(len(self._compact_matrix_chain_plan_cache))
        stats["last"] = {
            "dimension": int(total_dim),
            "r_entries": int(len(plan["r_entries"])),
            "t2_entries": int(len(plan["t2_entries"])),
            "t3_entries": int(len(plan["t3_entries"])),
            "out_entries": int(len(plan["out_entries"])),
            "output_blocks": int(len(plan["out_keys"])),
            "max_batch_entries": int(max_entries),
            "compiled_kernel": bool(compiled_kernel or parallel_compiled_kernel),
            "compiled_kernel_mode": (
                "parallel"
                if parallel_compiled_kernel
                else ("cython" if cython_compiled_kernel else ("serial" if compiled_kernel else "blas"))
            ),
            "parallel_work": int(parallel_work),
            "parallel_min_work": int(
                self._batched_compact_matrix_chain_compiled_parallel_min_work
            ),
            "project_output": bool(project_output),
            "projected_plan": bool(plan.get("projected_plan", False)),
            "projected_output_blocks": int(projected_output_blocks),
            "projected_output_dim": int(projected_output_dim),
            **plan.get("projection_stats", {}),
            "numba_threads": (
                int(_numba_get_num_threads())
                if parallel_compiled_kernel and _numba_get_num_threads is not None
                else None
            ),
        }
        return out

    @staticmethod
    def _native_compact_matrix_chain_available():
        return (
            _numba_matrix_chain_e_a_accum is not None
            and _numba_matrix_chain_r_w1_accum is not None
            and _numba_matrix_chain_t2_w2_accum is not None
            and _numba_matrix_chain_t3_f_accum is not None
        )

    def _matvec_native_compact_matrix_chain(self, A):
        if not self._native_compact_matrix_chain_available():
            return None
        plan = self._build_compact_matrix_chain_plan(A)
        if plan is None:
            return None
        start = time.perf_counter()
        dtype = plan["dtype"]
        try:
            a_blocks = tuple(np.ascontiguousarray(A.data[key]) for key in plan["a_keys"])
        except KeyError:
            return None
        r_data = self._compact_buffers(plan, "native_r", plan["r_shapes"], dtype)
        t2_data = self._compact_buffers(plan, "native_t2", plan["t2_shapes"], dtype)
        t3_data = self._compact_buffers(plan, "native_t3", plan["t3_shapes"], dtype)
        out_data = [np.zeros(shape, dtype=dtype) for shape in plan["out_shapes"]]

        e_blocks = plan["e_blocks"]
        w1_blocks = plan["w1_blocks"]
        w2_blocks = plan["w2_blocks"]
        f_blocks = plan["f_blocks"]
        for e_i, a_i, r_i in plan["r_entries"]:
            _numba_matrix_chain_e_a_accum(e_blocks[e_i], a_blocks[a_i], r_data[r_i])
        for r_i, w_i, t2_i in plan["t2_entries"]:
            _numba_matrix_chain_r_w1_accum(r_data[r_i], w1_blocks[w_i], t2_data[t2_i])
        for t2_i, w_i, t3_i in plan["t3_entries"]:
            _numba_matrix_chain_t2_w2_accum(t2_data[t2_i], w2_blocks[w_i], t3_data[t3_i])
        for t3_i, f_i, out_i in plan["out_entries"]:
            _numba_matrix_chain_t3_f_accum(t3_data[t3_i], f_blocks[f_i], out_data[out_i])

        stats = self.profile_stats.setdefault(
            "native_compact_matrix_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["available"] = True
        stats["plan_cache_entries"] = int(len(self._compact_matrix_chain_plan_cache))
        stats["last"] = {
            "r_entries": int(len(plan["r_entries"])),
            "t2_entries": int(len(plan["t2_entries"])),
            "t3_entries": int(len(plan["t3_entries"])),
            "out_entries": int(len(plan["out_entries"])),
            "output_blocks": int(len(plan["out_keys"])),
        }
        return self._tensor_from_block_data_like(
            A,
            {key: out_data[i] for i, key in enumerate(plan["out_keys"])},
            plan["out_qns"],
            plan["out_dirs"],
        )

    def _tensordot_cached_plan(self, A, B, axes, label):
        a_ax, b_ax = axes
        if isinstance(a_ax, int):
            a_ax = (a_ax,)
        else:
            a_ax = tuple(a_ax)
        if isinstance(b_ax, int):
            b_ax = (b_ax,)
        else:
            b_ax = tuple(b_ax)
        cache_key = (
            str(label),
            self._layout(A),
            self._layout(B),
            tuple(a_ax),
            tuple(b_ax),
            tuple(A.dirs),
            tuple(B.dirs),
        )
        cached = self._generic_chain_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        free_A = tuple(i for i in range(A.rank) if i not in a_ax)
        free_B = tuple(i for i in range(B.rank) if i not in b_ax)
        new_dirs = [A.dirs[i] for i in free_A] + [B.dirs[i] for i in free_B]
        new_qns = [A.qns[i] for i in free_A] + [B.qns[i] for i in free_B]

        b_by_contract = defaultdict(list)
        for qn_B, block_B in B.data.items():
            b_by_contract[tuple(qn_B[i] for i in b_ax)].append((qn_B, block_B.shape))

        entries = []
        out_shapes = {}
        for qn_A, block_A in A.data.items():
            key_contract = tuple(qn_A[i] for i in a_ax)
            for qn_B, _shape_B in b_by_contract.get(key_contract, ()):
                qn_C = tuple(qn_A[i] for i in free_A) + tuple(qn_B[i] for i in free_B)
                out_shape = (
                    tuple(block_A.shape[i] for i in free_A)
                    + tuple(B.data[qn_B].shape[i] for i in free_B)
                )
                old_shape = out_shapes.get(qn_C)
                if old_shape is not None and old_shape != out_shape:
                    self._generic_chain_plan_cache[cache_key] = None
                    return None
                out_shapes[qn_C] = out_shape
                entries.append((qn_A, qn_B, qn_C))

        dtype = np.result_type(
            *[block.dtype for block in A.data.values()],
            *[block.dtype for block in B.data.values()],
            complex,
        )
        plan = (tuple(entries), out_shapes, new_qns, new_dirs, (tuple(a_ax), tuple(b_ax)), dtype)
        self._generic_chain_plan_cache[cache_key] = plan
        return plan

    def _tensordot_cached_apply(self, A, B, axes, label):
        plan = self._tensordot_cached_plan(A, B, axes, label)
        if plan is None:
            return None
        entries, out_shapes, new_qns, new_dirs, cached_axes, dtype = plan
        out = {key: np.zeros(shape, dtype=dtype) for key, shape in out_shapes.items()}
        for qn_A, qn_B, qn_C in entries:
            contribution = np.tensordot(A.data[qn_A], B.data[qn_B], axes=cached_axes)
            if qn_C in out:
                out[qn_C] += contribution
            else:
                out[qn_C] = contribution
        proto = None
        if isinstance(A, AbelianSiteTensorData):
            proto = A
        elif isinstance(B, AbelianSiteTensorData):
            proto = B
        if proto is not None:
            return self._tensor_from_block_data_like(proto, out, new_qns, new_dirs)
        return BlockTensor(out, new_qns, new_dirs)

    def _matvec_generic_cached_chain(self, A):
        start = time.perf_counter()
        R = self._tensordot_cached_apply(self.E, A, ([2], [0]), "chain_E_A")
        if R is None:
            return None
        T2 = self._tensordot_cached_apply(R, self.W[0], ([0, 3], [0, 3]), "chain_R_W1")
        if T2 is None:
            return None
        T3 = self._tensordot_cached_apply(T2, self.W[1], ([3, 2], [0, 3]), "chain_T2_W2")
        if T3 is None:
            return None
        T4 = self._tensordot_cached_apply(T3, self.F, ([3, 1], [0, 2]), "chain_T3_F")
        if T4 is None:
            return None
        out = T4.transpose(0, 3, 1, 2)
        stats = self.profile_stats.setdefault(
            "generic_chain",
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        elapsed = time.perf_counter() - start
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["seconds"] = float(stats.get("seconds", 0.0)) + float(elapsed)
        stats["last_seconds"] = float(elapsed)
        stats["plan_cache_entries"] = int(len(self._generic_chain_plan_cache))
        return out

    @staticmethod
    def _matvec_generic_components(E, W, F, A):
        R = tensordot(E, A, axes=([2], [0]))
        T2 = tensordot(R, W[0], axes=([0, 3], [0, 3]))
        T3 = tensordot(T2, W[1], axes=([3, 2], [0, 3]))
        T4 = tensordot(T3, F, axes=([3, 1], [0, 2]))
        return T4.transpose(0, 3, 1, 2)

    def _matvec_family_components_chain(self, E, W, F, A):
        e_by_ket_l = self._block_index(E, (2,))
        w1_by_left_in = self._block_index(W[0], (0, 3))
        w2_by_left_in = self._block_index(W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(F, (0, 2))
        dtype = self._block_data_dtype(E, W, F, A, complex)

        r_data = {}
        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                return None
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    return None
                key = (e_key[0], e_key[1], right_qn, p1_in, p2_in)
                contrib = np.einsum(
                    "aij,jkxy->aikxy",
                    e_blk,
                    a_blk,
                    optimize="greedy",
                )
                r_data[key] = contrib if key not in r_data else r_data[key] + contrib

        t2_data = {}
        for r_key, r_blk in r_data.items():
            mpo_left, left_out, right_qn, p1_in, p2_in = r_key
            for w1_key, w1_blk in w1_by_left_in.get((mpo_left, p1_in), ()):
                if w1_blk.ndim != 4:
                    return None
                key = (w1_key[1], left_out, right_qn, p2_in, w1_key[2])
                contrib = np.einsum(
                    "aikxy,abux->ikybu",
                    r_blk,
                    w1_blk,
                    optimize="greedy",
                )
                t2_data[key] = contrib if key not in t2_data else t2_data[key] + contrib

        t3_data = {}
        for t2_key, t2_blk in t2_data.items():
            channel, left_out, right_qn, p2_in, p1_out = t2_key
            for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                if w2_blk.ndim != 4:
                    return None
                key = (w2_key[1], left_out, right_qn, p1_out, w2_key[2])
                contrib = np.einsum(
                    "ikybu,bcvy->ikucv",
                    t2_blk,
                    w2_blk,
                    optimize="greedy",
                )
                t3_data[key] = contrib if key not in t3_data else t3_data[key] + contrib

        out_data = {}
        for t3_key, t3_blk in t3_data.items():
            mpo_right, left_out, right_qn, p1_out, p2_out = t3_key
            for f_key, f_blk in f_by_mpo_ket_r.get((mpo_right, right_qn), ()):
                if f_blk.ndim != 3:
                    return None
                key = (left_out, f_key[1], p1_out, p2_out)
                contrib = np.einsum(
                    "ikucv,clk->iluv",
                    t3_blk,
                    f_blk,
                    optimize="greedy",
                )
                out_data[key] = (
                    contrib if key not in out_data else out_data[key] + contrib
                )
        if not out_data:
            return None
        layout = tuple((key, block.shape) for key, block in out_data.items())
        qns = self._qns_from_layout_with_proto(layout, A)
        return self._tensor_from_block_data_like(A, out_data, qns, A.dirs[:])

    @staticmethod
    def _family_base_name(name):
        return str(name).split(":", 1)[0]

    def _record_family_environment_timing(self, name, phase, elapsed):
        stats = self.complementary_split_stats
        if stats is None:
            return
        family_stats = stats.setdefault("family_environment_timings", {})
        phases = family_stats.setdefault(str(name), {})
        entry = phases.setdefault(
            str(phase),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)
        if self.bond is not None:
            entry["last_bond"] = int(self.bond)

    def _record_direct_operator_batch_stats(self, entries, batched_groups):
        stats = self.complementary_split_stats
        if stats is None:
            return
        total_entries = int(len(entries))
        batched_entries = int(sum(len(group[1]) for group in batched_groups))
        scalar_entries = int(total_entries - batched_entries)
        batch_stats = stats.setdefault(
            "direct_operator_batching",
            {
                "calls": 0,
                "entries": 0,
                "batched_entries": 0,
                "scalar_entries": 0,
                "batched_groups": 0,
                "max_group_size": 0,
            },
        )
        batch_stats["calls"] = int(batch_stats.get("calls", 0)) + 1
        batch_stats["entries"] = int(batch_stats.get("entries", 0)) + total_entries
        batch_stats["batched_entries"] = (
            int(batch_stats.get("batched_entries", 0)) + batched_entries
        )
        batch_stats["scalar_entries"] = (
            int(batch_stats.get("scalar_entries", 0)) + scalar_entries
        )
        batch_stats["batched_groups"] = (
            int(batch_stats.get("batched_groups", 0)) + int(len(batched_groups))
        )
        max_group = max((len(group[1]) for group in batched_groups), default=0)
        batch_stats["max_group_size"] = max(
            int(batch_stats.get("max_group_size", 0)),
            int(max_group),
        )
        batch_stats["last"] = {
            "entries": total_entries,
            "batched_entries": batched_entries,
            "scalar_entries": scalar_entries,
            "batched_groups": int(len(batched_groups)),
            "max_group_size": int(max_group),
            "min_entries": int(self._direct_operator_batch_min_entries),
            "bond": self.bond,
        }

    def _matvec_named_family_channels(self, A):
        if not self.complementary_family_environments:
            return None
        channels = {}
        stats = {}
        for name, env in self.complementary_family_environments.items():
            t0 = time.perf_counter()
            try:
                E, W, F = env
            except Exception:
                continue
            if E is None or F is None or W is None or len(W) != 2:
                continue
            if self._prefer_precontracted_family_environment:
                plan = self._build_direct_operator_plan(
                    A,
                    E,
                    W,
                    F,
                    ("named_family", str(name)),
                )
                applied = self._apply_direct_operator_plan(A, plan)
            else:
                applied = None
            if applied is None:
                tensor = self._matvec_generic_components(E, W, F, A)
                n_entries = None
                n_channels = None
                source = "generic_tensordot_family_environment"
            else:
                tensor, family_channels, entries = applied
                n_entries = int(len(entries))
                n_channels = int(len(family_channels))
                source = "precontracted_family_environment"
            if tensor is not None:
                raw_name = str(name)
                base_name = self._family_base_name(raw_name)
                channels[base_name] = (
                    tensor
                    if base_name not in channels
                    else channels[base_name] + tensor
                )
                raw_stats = {
                    "source": source,
                    "n_entries": n_entries,
                    "n_mpo_middle_channels": n_channels,
                    "seconds": float(time.perf_counter() - t0),
                }
                self._record_family_environment_timing(
                    raw_name,
                    "apply",
                    raw_stats["seconds"],
                )
                base_stats = stats.setdefault(
                    base_name,
                    {
                        "source": "family_mpo_environments",
                        "split_family_names": (),
                        "n_entries": 0,
                        "n_mpo_middle_channels": 0,
                        "seconds": 0.0,
                        "raw_family_channel_stats": {},
                    },
                )
                base_stats["split_family_names"] = (
                    tuple(base_stats["split_family_names"]) + (raw_name,)
                )
                if n_entries is None:
                    base_stats["n_entries"] = None
                elif base_stats["n_entries"] is not None:
                    base_stats["n_entries"] = int(base_stats["n_entries"]) + int(
                        n_entries
                    )
                if n_channels is None:
                    base_stats["n_mpo_middle_channels"] = None
                elif base_stats["n_mpo_middle_channels"] is not None:
                    base_stats["n_mpo_middle_channels"] = int(
                        base_stats["n_mpo_middle_channels"]
                    ) + int(n_channels)
                base_stats["seconds"] = float(base_stats["seconds"]) + raw_stats[
                    "seconds"
                ]
                base_stats["raw_family_channel_stats"][raw_name] = raw_stats
        if not channels:
            return None
        return channels, stats

    def _matvec_direct_symbolic_family_channels(self, A):
        if not self.complementary_direct_family_environments:
            return None
        channels = {}
        stats = {}
        for name, entries in self.complementary_direct_family_environments.items():
            t0 = time.perf_counter()
            total = None
            total_data = {}
            total_qns = None
            total_dirs = None
            n_entries = 0
            n_precontracted = 0
            n_generic = 0
            n_middle_channels = 0
            entry_groups = tuple(getattr(entries, "entry_groups", ()) or ())
            if len(entry_groups) >= int(len(entries)):
                entry_groups = ()
            group_keys = tuple(getattr(entries, "group_keys", ()) or ())
            if self._prefer_precontracted_family_environment:
                if entry_groups:
                    trial_data = {}
                    trial_qns = None
                    trial_dirs = None
                    trial_middle_channels = 0
                    grouped_ok = True
                    for group_index, group_entries in enumerate(entry_groups):
                        group_name = (
                            f"{name}:group:{group_index}:"
                            f"{repr(group_keys[group_index]) if group_index < len(group_keys) else '?'}"
                        )
                        combined_plan = self._build_combined_direct_family_plan(
                            A,
                            group_name,
                            group_entries,
                        )
                        plan_data = {}
                        applied = self._accumulate_direct_operator_plan(
                            A,
                            combined_plan,
                            plan_data,
                        )
                        if applied is None:
                            grouped_ok = False
                            break
                        middle_channels, plan_entries, out_qns, out_dirs = applied
                        for key, block in plan_data.items():
                            if key in trial_data and trial_data[key].shape != block.shape:
                                grouped_ok = False
                                break
                            trial_data[key] = (
                                trial_data[key] + block
                                if key in trial_data
                                else block.copy()
                            )
                        if not grouped_ok:
                            break
                        if trial_qns is None:
                            trial_qns = out_qns
                            trial_dirs = out_dirs
                        trial_middle_channels += int(len(middle_channels))
                    if grouped_ok and trial_data:
                        total_data = trial_data
                        total_qns = trial_qns
                        total_dirs = trial_dirs
                        n_entries = int(len(entries))
                        n_precontracted = int(len(entries))
                        n_middle_channels = int(trial_middle_channels)
                else:
                    combined_plan = self._build_combined_direct_family_plan(
                        A,
                        name,
                        entries,
                    )
                    plan_data = {}
                    applied = self._accumulate_direct_operator_plan(
                        A,
                        combined_plan,
                        plan_data,
                    )
                    if applied is not None:
                        middle_channels, plan_entries, total_qns, total_dirs = applied
                        total_data = plan_data
                        n_entries = int(len(entries))
                        n_precontracted = int(len(entries))
                        n_middle_channels = int(len(middle_channels))
            if not n_precontracted:
                for component in entries:
                    if isinstance(component, AbelianPackedLocalGeneratorEntry):
                        E = component.E
                        W = [
                            scale_abelian_boundary_tensor(
                                component.W_left,
                                component.coeff,
                                source="direct_symbolic_family_scale",
                            ),
                            component.W_right,
                        ]
                        F = component.F
                    else:
                        E, W, F = component
                    if self._prefer_precontracted_family_environment:
                        plan = self._build_direct_operator_plan(
                            A,
                            E,
                            W,
                            F,
                            ("direct_symbolic_family", str(name), n_entries),
                        )
                        plan_data = {}
                        applied = self._accumulate_direct_operator_plan(
                            A,
                            plan,
                            plan_data,
                        )
                    else:
                        applied = None
                    if applied is None:
                        tensor = self._matvec_generic_components(E, W, F, A)
                        if total_data:
                            for key, block in tensor.data.items():
                                if key in total_data:
                                    total_data[key] = total_data[key] + block
                                else:
                                    total_data[key] = block.copy()
                            if total_qns is None:
                                total_qns = tensor.qns
                                total_dirs = tensor.dirs
                            tensor = None
                        else:
                            total = tensor if total is None else total + tensor
                        n_generic += 1
                    else:
                        middle_channels, plan_entries, out_qns, out_dirs = applied
                        compatible = True
                        for key, block in plan_data.items():
                            if key in total_data and total_data[key].shape != block.shape:
                                compatible = False
                                break
                        if compatible:
                            for key, block in plan_data.items():
                                if key in total_data:
                                    total_data[key] = total_data[key] + block
                                else:
                                    total_data[key] = block.copy()
                        else:
                            applied = None
                        if applied is None:
                            tensor = self._matvec_generic_components(E, W, F, A)
                            for key, block in tensor.data.items():
                                if key in total_data:
                                    total_data[key] = total_data[key] + block
                                else:
                                    total_data[key] = block.copy()
                            if total_qns is None:
                                total_qns = tensor.qns
                                total_dirs = tensor.dirs
                            n_generic += 1
                            n_entries += 1
                            continue
                        if total_qns is None:
                            total_qns = out_qns
                            total_dirs = out_dirs
                        n_precontracted += 1
                        n_middle_channels += int(len(middle_channels))
                    n_entries += 1
            if total_data:
                tensor = self._tensor_from_block_data_like(
                    A,
                    total_data,
                    total_qns,
                    total_dirs,
                )
                total = tensor if total is None else total + tensor
            if total is not None:
                key = str(name)
                channels[key] = total
                source = (
                    "precontracted_direct_symbolic_term_environments"
                    if n_precontracted and not n_generic
                    else "mixed_direct_symbolic_term_environments"
                    if n_precontracted
                    else "direct_symbolic_term_environments"
                )
                elapsed = float(time.perf_counter() - t0)
                stats[key] = {
                    "source": source,
                    "n_entries": int(n_entries),
                    "n_precontracted_entries": int(n_precontracted),
                    "n_generic_entries": int(n_generic),
                    "n_mpo_middle_channels": int(n_middle_channels),
                    "seconds": elapsed,
                    "n_entry_groups": int(len(entry_groups)),
                    "entry_group_sizes": tuple(
                        int(len(group)) for group in entry_groups
                    ),
                }
                self._record_family_environment_timing(
                    key,
                    "direct_apply",
                    elapsed,
                )
        if not channels:
            return None
        return channels, stats

    def _matvec_middle_mpo_channel(self, A, channel):
        w1_data = {
            key: value
            for key, value in self.W[0].data.items()
            if key[1] == channel
        }
        w2_data = {
            key: value
            for key, value in self.W[1].data.items()
            if key[0] == channel
        }
        if not w1_data or not w2_data:
            return None
        W1 = self._tensor_from_block_data_like(
            A,
            w1_data,
            self.W[0].qns[:],
            self.W[0].dirs[:],
        )
        W2 = self._tensor_from_block_data_like(
            A,
            w2_data,
            self.W[1].qns[:],
            self.W[1].dirs[:],
        )
        original = self.W
        try:
            self.W = [W1, W2]
            return self._matvec_generic(A)
        finally:
            self.W = original

    def _middle_mpo_channels(self):
        channels = {
            key[1]
            for key in self.W[0].data
        }.intersection({
            key[0]
            for key in self.W[1].data
        })
        return tuple(sorted(channels, key=lambda item: repr(item)))

    def _build_boundary_factorized_plan(self, A):
        layout = self._layout(A)
        cache_key = (self._action_token(), layout, "boundary_factorized")
        if cache_key in self._boundary_factorized_cache:
            return self._boundary_factorized_cache[cache_key]
        build_start = time.perf_counter()

        e_by_ket_l = self._block_index(self.E, (2,))
        w1_by_left_in = self._block_index(self.W[0], (0, 3))
        w2_by_left_in = self._block_index(self.W[1], (0, 3))
        f_by_mpo_ket_r = self._block_index(self.F, (0, 2))

        entries = []
        out_shapes = {}
        dtype_args = []
        channels = set()
        eq = "aij,jkxy,abux,bcvy,clk->iluv"

        for a_key, a_blk in A.data.items():
            if a_blk.ndim != 4:
                self._boundary_factorized_cache[cache_key] = None
                return None
            dtype_args.append(a_blk.dtype)
            left_qn, right_qn, p1_in, p2_in = a_key
            for e_key, e_blk in e_by_ket_l.get((left_qn,), ()):
                if e_blk.ndim != 3:
                    self._boundary_factorized_cache[cache_key] = None
                    return None
                for w1_key, w1_blk in w1_by_left_in.get((e_key[0], p1_in), ()):
                    if w1_blk.ndim != 4:
                        self._boundary_factorized_cache[cache_key] = None
                        return None
                    channel = w1_key[1]
                    for w2_key, w2_blk in w2_by_left_in.get((channel, p2_in), ()):
                        if w2_blk.ndim != 4:
                            self._boundary_factorized_cache[cache_key] = None
                            return None
                        for f_key, f_blk in f_by_mpo_ket_r.get((w2_key[1], right_qn), ()):
                            if f_blk.ndim != 3:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            expr_key = (
                                e_blk.shape,
                                a_blk.shape,
                                w1_blk.shape,
                                w2_blk.shape,
                                f_blk.shape,
                            )
                            expr = self._boundary_einsum_expr_cache.get(expr_key)
                            if expr is None:
                                try:
                                    import opt_einsum as oe

                                    expr = oe.contract_expression(
                                        eq,
                                        e_blk.shape,
                                        a_blk.shape,
                                        w1_blk.shape,
                                        w2_blk.shape,
                                        f_blk.shape,
                                        optimize="greedy",
                                    )
                                except Exception:
                                    try:
                                        expr = np.einsum_path(
                                            eq,
                                            e_blk,
                                            a_blk,
                                            w1_blk,
                                            w2_blk,
                                            f_blk,
                                            optimize="greedy",
                                        )[0]
                                    except ValueError:
                                        self._boundary_factorized_cache[cache_key] = None
                                        return None
                                self._boundary_einsum_expr_cache[expr_key] = expr
                            out_key = (e_key[1], f_key[1], w1_key[2], w2_key[2])
                            out_shape = (
                                e_blk.shape[1],
                                f_blk.shape[1],
                                w1_blk.shape[2],
                                w2_blk.shape[2],
                            )
                            old_shape = out_shapes.get(out_key)
                            if old_shape is not None and old_shape != out_shape:
                                self._boundary_factorized_cache[cache_key] = None
                                return None
                            out_shapes[out_key] = out_shape
                            dtype_args.extend((e_blk.dtype, w1_blk.dtype, w2_blk.dtype, f_blk.dtype))
                            channels.add(channel)
                            entries.append((channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, expr))
                            if len(entries) > int(self._boundary_factorized_action_cap):
                                self._boundary_factorized_cache[cache_key] = None
                                return None

        if not entries:
            plan = ((), (), (), {}, A.qns[:], A.dirs[:], np.result_type(*dtype_args, complex))
            self._boundary_factorized_cache[cache_key] = plan
            self._record_plan_profile("batched_action", time.perf_counter() - build_start, entries=0)
            return plan

        batch_sources = {}
        for channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, _expr in entries:
            batch_key = (
                out_key,
                e_blk.shape,
                A.data[a_key].shape,
                w1_blk.shape,
                w2_blk.shape,
                f_blk.shape,
            )
            batch_sources.setdefault(batch_key, []).append(
                (channel, a_key, e_blk, w1_blk, w2_blk, f_blk)
            )

        batched_groups = []
        batch_eq = "zaij,zjkxy,zabux,zbcvy,zclk->iluv"
        for batch_key, members in batch_sources.items():
            out_key, e_shape, a_shape, w1_shape, w2_shape, f_shape = batch_key
            if len(members) < 2:
                continue
            expr_key = (len(members), e_shape, a_shape, w1_shape, w2_shape, f_shape)
            expr = self._boundary_batched_einsum_expr_cache.get(expr_key)
            if expr is None:
                try:
                    import opt_einsum as oe

                    expr = oe.contract_expression(
                        batch_eq,
                        (len(members),) + tuple(e_shape),
                        (len(members),) + tuple(a_shape),
                        (len(members),) + tuple(w1_shape),
                        (len(members),) + tuple(w2_shape),
                        (len(members),) + tuple(f_shape),
                        optimize="greedy",
                    )
                except Exception:
                    expr = None
                self._boundary_batched_einsum_expr_cache[expr_key] = expr
            if expr is None:
                continue
            try:
                e_stack = np.stack([entry[2] for entry in members], axis=0)
                w1_stack = np.stack([entry[3] for entry in members], axis=0)
                w2_stack = np.stack([entry[4] for entry in members], axis=0)
                f_stack = np.stack([entry[5] for entry in members], axis=0)
            except ValueError:
                continue
            batched_groups.append(
                (
                    out_key,
                    tuple(entry[0] for entry in members),
                    tuple(entry[1] for entry in members),
                    e_stack,
                    w1_stack,
                    w2_stack,
                    f_stack,
                    expr,
                )
            )

        batched_entry_ids = {
            (out_key, a_key, channel)
            for out_key, channels, a_keys, *_rest in batched_groups
            for channel, a_key in zip(channels, a_keys)
        }
        scalar_entries = tuple(
            entry
            for entry in entries
            if (entry[2], entry[1], entry[0]) not in batched_entry_ids
        )

        out_qns = self._qns_from_layout(tuple((k, out_shapes[k]) for k in sorted(out_shapes)))
        dtype = np.result_type(*dtype_args, complex)
        plan = (
            tuple(sorted(channels, key=lambda item: repr(item))),
            scalar_entries,
            tuple(batched_groups),
            out_shapes,
            out_qns,
            A.dirs[:],
            dtype,
        )
        self._boundary_factorized_cache[cache_key] = plan
        self._record_plan_profile(
            "batched_action",
            time.perf_counter() - build_start,
            entries=int(len(entries)),
            scalar_entries=int(len(scalar_entries)),
            batched_groups=int(len(batched_groups)),
            batched_entries=int(sum(len(group[2]) for group in batched_groups)),
            output_blocks=int(len(out_shapes)),
        )
        return plan

    def _matvec_boundary_factorized(
        self,
        A,
        *,
        local_channels=None,
        collect_channels=False,
    ):
        plan = self._build_boundary_factorized_plan(A)
        if plan is None:
            return None
        channels, entries, batched_groups, out_shapes, out_qns, out_dirs, dtype = plan
        eq = "aij,jkxy,abux,bcvy,clk->iluv"
        total_data = {
            key: np.zeros(shape, dtype=dtype)
            for key, shape in out_shapes.items()
        }
        channel_data = None
        if collect_channels:
            channel_data = {
                channel: {
                    key: np.zeros(shape, dtype=dtype)
                    for key, shape in out_shapes.items()
                }
                for channel in channels
            }
        if not collect_channels:
            for out_key, _channels, a_keys, e_stack, w1_stack, w2_stack, f_stack, expr in batched_groups:
                a_stack = np.stack([A.data[a_key] for a_key in a_keys], axis=0)
                total_data[out_key] += expr(
                    e_stack,
                    a_stack,
                    w1_stack,
                    w2_stack,
                    f_stack,
                )
        for channel, a_key, out_key, e_blk, w1_blk, w2_blk, f_blk, expr in entries:
            if callable(expr):
                contribution = expr(e_blk, A.data[a_key], w1_blk, w2_blk, f_blk)
            else:
                contribution = np.einsum(
                    eq,
                    e_blk,
                    A.data[a_key],
                    w1_blk,
                    w2_blk,
                    f_blk,
                    optimize=expr,
                )
            total_data[out_key] += contribution
            if channel_data is not None:
                channel_data[channel][out_key] += contribution

        total = self._tensor_from_block_data_like(A, total_data, out_qns, out_dirs)
        if not total.data:
            return None

        channel_tensors = {}
        if channel_data is not None:
            channel_tensors = {
                f"mpo_middle:{repr(channel)}": self._tensor_from_block_data_like(
                    A,
                    data,
                    out_qns,
                    out_dirs,
                )
                for channel, data in channel_data.items()
            }

        local_channels = local_channels or {}
        for name, tensor in local_channels.items():
            if collect_channels:
                channel_tensors[f"subtract_local:{name}"] = tensor * -1.0
            total = total - tensor

        return {
            "total": total,
            "channels": channel_tensors,
            "stats": {
                "kind": "abelian_complementary_boundary_factorized_action",
                "source": "middle_mpo_channels_minus_local_RP",
                "bond": self.bond,
                "n_entries": int(
                    len(entries)
                    + sum(len(group[2]) for group in batched_groups)
                ),
                "n_batched_groups": int(len(batched_groups)),
                "n_scalar_entries": int(len(entries)),
                "n_mpo_middle_channels": int(len(channels)),
                "n_channels": int(len(channels) + len(local_channels)),
                "channel_names": tuple(
                    f"mpo_middle:{repr(channel)}" for channel in channels
                )
                + tuple(f"subtract_local:{name}" for name in local_channels),
                "channels_materialized": bool(collect_channels),
            },
        }

    def _two_site_mpo(self):
        if self._w12_cache is not None:
            return self._w12_cache
        W12 = tensordot(self.W[0], self.W[1], axes=([1], [0]))
        W12 = W12.transpose(0, 3, 1, 4, 2, 5)
        self._w12_cache = W12
        return W12

    def _matvec_fused_mpo(self, A):
        W12 = self._two_site_mpo()
        R = tensordot(self.E, A, axes=([2], [0]))
        T = tensordot(R, W12, axes=([0, 3, 4], [0, 4, 5]))
        T = tensordot(T, self.F, axes=([2, 1], [0, 2]))
        return T.transpose(0, 3, 1, 2)

    def _matvec_batched_action(self, A):
        action = self._matvec_boundary_factorized(A, local_channels=None)
        if action is None:
            return None
        return action["total"]

    def _matvec_direct_operator_action(self, A, *, plan=None):
        if plan is None:
            plan = self._build_boundary_direct_operator_plan(A)
        applied = self._apply_direct_operator_plan(A, plan)
        if applied is None:
            return None
        total, _channels, _entries = applied
        return total

    @staticmethod
    def _relative_blocktensor_error(reference, candidate):
        ref_keys = set(reference.data)
        if ref_keys != set(candidate.data):
            return math.inf
        num = 0.0
        den = 0.0
        for key, ref_block in reference.data.items():
            cand_block = candidate.data[key]
            if cand_block.shape != ref_block.shape:
                return math.inf
            diff = cand_block - ref_block
            num += float(np.vdot(diff, diff).real)
            den += float(np.vdot(ref_block, ref_block).real)
        if not np.isfinite(num) or not np.isfinite(den):
            return math.inf
        return math.sqrt(max(num, 0.0)) / max(1.0, math.sqrt(max(den, 0.0)))

    def _matvec_auto_block_action(self, A):
        layout = self._layout(A)
        choice_key = layout if self._matvec_selector_per_layout else ("global",)
        layout_choice = self._matvec_action_choice_by_layout.get(choice_key)

        def _set_layout_choice(choice):
            self._matvec_action_choice_by_layout[choice_key] = str(choice)
            self._matvec_action_choice = str(choice)

        if layout_choice == "batched_compact_matrix_chain":
            result = self._matvec_batched_compact_matrix_chain(A)
            if result is not None:
                return result, "batched_compact_matrix_chain"
            _set_layout_choice("generic")
        if layout_choice == "native_compact_matrix_chain":
            result = self._matvec_native_compact_matrix_chain(A)
            if result is not None:
                return result, "native_compact_matrix_chain"
            _set_layout_choice("generic")
        if layout_choice == "compact_matrix_chain":
            result = self._matvec_compact_matrix_chain(A)
            if result is not None:
                return result, "compact_matrix_chain"
            _set_layout_choice("generic")
        if layout_choice == "generic_matrix_chain":
            result = self._matvec_generic_matrix_chain(A)
            if result is not None:
                return result, "generic_matrix_chain"
            _set_layout_choice("generic")
        if layout_choice == "generic_chain":
            result = self._matvec_generic_cached_chain(A)
            if result is not None:
                return result, "generic_chain"
            _set_layout_choice("generic")
        if layout_choice == "direct_operator":
            result = self._matvec_direct_operator_action(A)
            if result is not None:
                return result, "direct_operator"
            _set_layout_choice("generic")
        if layout_choice == "batched_action":
            result = self._matvec_batched_action(A)
            if result is not None:
                return result, "batched_action"
            _set_layout_choice("generic")
        if layout_choice == "generic":
            return self._matvec_generic(A), "generic"

        if self._matrix_chain_force:
            result = self._matvec_generic_matrix_chain(A)
            if result is not None:
                _set_layout_choice("generic_matrix_chain")
                self.profile_stats["action_selector"] = {
                    "choice": "generic_matrix_chain",
                    "matrix_chain_forced": True,
                    "matrix_chain_available": True,
                    "matrix_chain_selector_enabled": bool(self._matrix_chain_selector_enabled),
                    "selector_per_layout": bool(self._matvec_selector_per_layout),
                    "layout_choices": int(len(self._matvec_action_choice_by_layout)),
                }
                return result, "generic_matrix_chain"

        if self._compact_matrix_chain_force:
            result = self._matvec_compact_matrix_chain(A)
            if result is not None:
                _set_layout_choice("compact_matrix_chain")
                self.profile_stats["action_selector"] = {
                    "choice": "compact_matrix_chain",
                    "compact_matrix_chain_forced": True,
                    "compact_matrix_chain_available": True,
                    "compact_matrix_chain_selector_enabled": bool(
                        self._compact_matrix_chain_selector_enabled
                    ),
                    "selector_per_layout": bool(self._matvec_selector_per_layout),
                    "layout_choices": int(len(self._matvec_action_choice_by_layout)),
                }
                return result, "compact_matrix_chain"

        if self._batched_compact_matrix_chain_force:
            result = self._matvec_batched_compact_matrix_chain(A)
            if result is not None:
                _set_layout_choice("batched_compact_matrix_chain")
                self.profile_stats["action_selector"] = {
                    "choice": "batched_compact_matrix_chain",
                    "batched_compact_matrix_chain_forced": True,
                    "batched_compact_matrix_chain_available": True,
                    "batched_compact_matrix_chain_selector_enabled": bool(
                        self._batched_compact_matrix_chain_selector_enabled
                    ),
                    "selector_per_layout": bool(self._matvec_selector_per_layout),
                    "layout_choices": int(len(self._matvec_action_choice_by_layout)),
                }
                return result, "batched_compact_matrix_chain"

        if self._native_compact_matrix_chain_force:
            result = self._matvec_native_compact_matrix_chain(A)
            if result is not None:
                _set_layout_choice("native_compact_matrix_chain")
                self.profile_stats["action_selector"] = {
                    "choice": "native_compact_matrix_chain",
                    "native_compact_matrix_chain_forced": True,
                    "native_compact_matrix_chain_available": True,
                    "native_compact_matrix_chain_selector_enabled": bool(
                        self._native_compact_matrix_chain_selector_enabled
                    ),
                    "selector_per_layout": bool(self._matvec_selector_per_layout),
                    "layout_choices": int(len(self._matvec_action_choice_by_layout)),
                }
                return result, "native_compact_matrix_chain"

        generic_start = time.perf_counter()
        generic = self._matvec_generic(A)
        generic_seconds = time.perf_counter() - generic_start

        batched_compact_matrix_chain = None
        batched_compact_matrix_chain_seconds = math.inf
        if self._batched_compact_matrix_chain_selector_enabled:
            batched_compact_start = time.perf_counter()
            try:
                batched_compact_matrix_chain = self._matvec_batched_compact_matrix_chain(A)
            finally:
                batched_compact_matrix_chain_seconds = time.perf_counter() - batched_compact_start

        native_compact_matrix_chain = None
        native_compact_matrix_chain_seconds = math.inf
        if self._native_compact_matrix_chain_selector_enabled:
            native_compact_start = time.perf_counter()
            try:
                native_compact_matrix_chain = self._matvec_native_compact_matrix_chain(A)
            finally:
                native_compact_matrix_chain_seconds = time.perf_counter() - native_compact_start

        compact_matrix_chain = None
        compact_matrix_chain_seconds = math.inf
        if self._compact_matrix_chain_selector_enabled:
            compact_start = time.perf_counter()
            try:
                compact_matrix_chain = self._matvec_compact_matrix_chain(A)
            finally:
                compact_matrix_chain_seconds = time.perf_counter() - compact_start

        matrix_chain = None
        matrix_chain_seconds = math.inf
        if self._matrix_chain_selector_enabled:
            matrix_chain_start = time.perf_counter()
            try:
                matrix_chain = self._matvec_generic_matrix_chain(A)
            finally:
                matrix_chain_seconds = time.perf_counter() - matrix_chain_start

        chain = None
        chain_seconds = math.inf
        if self._generic_chain_selector_enabled:
            chain_start = time.perf_counter()
            try:
                chain = self._matvec_generic_cached_chain(A)
            finally:
                chain_seconds = time.perf_counter() - chain_start

        direct = None
        direct_seconds = math.inf
        direct_plan_build_seconds = math.inf
        direct_entries = 0
        direct_skipped_reason = None
        if self._direct_operator_selector_enabled:
            direct_build_start = time.perf_counter()
            direct_plan = self._build_boundary_direct_operator_plan(A)
            direct_plan_build_seconds = time.perf_counter() - direct_build_start
            if direct_plan is not None:
                direct_entries = int(len(direct_plan[1]))
                try_direct = (
                    direct_entries <= int(self._direct_operator_selector_edge_max_entries)
                    or direct_entries >= int(self._direct_operator_selector_min_entries)
                )
                if try_direct:
                    direct_start = time.perf_counter()
                    direct = self._matvec_direct_operator_action(A, plan=direct_plan)
                    direct_seconds = time.perf_counter() - direct_start
                else:
                    direct_skipped_reason = "entry_count_in_generic_preferred_band"
        else:
            direct_skipped_reason = "selector_disabled"

        batched = None
        batched_seconds = math.inf
        if self._batched_action_selector_enabled:
            batched_start = time.perf_counter()
            try:
                batched = self._matvec_batched_action(A)
            finally:
                batched_seconds = time.perf_counter() - batched_start

        rel_error = math.inf
        if batched is not None:
            rel_error = self._relative_blocktensor_error(generic, batched)
        direct_rel_error = math.inf
        if direct is not None:
            direct_rel_error = self._relative_blocktensor_error(generic, direct)
        chain_rel_error = math.inf
        if chain is not None:
            chain_rel_error = self._relative_blocktensor_error(generic, chain)
        matrix_chain_rel_error = math.inf
        if matrix_chain is not None:
            matrix_chain_rel_error = self._relative_blocktensor_error(generic, matrix_chain)
        compact_matrix_chain_rel_error = math.inf
        if compact_matrix_chain is not None:
            compact_matrix_chain_rel_error = self._relative_blocktensor_error(
                generic,
                compact_matrix_chain,
            )
        batched_compact_matrix_chain_rel_error = math.inf
        if batched_compact_matrix_chain is not None:
            batched_compact_matrix_chain_rel_error = self._relative_blocktensor_error(
                generic,
                batched_compact_matrix_chain,
            )
        native_compact_matrix_chain_rel_error = math.inf
        if native_compact_matrix_chain is not None:
            native_compact_matrix_chain_rel_error = self._relative_blocktensor_error(
                generic,
                native_compact_matrix_chain,
            )

        chain_threshold = 0.5
        batched_compact_matrix_chain_threshold = float(
            self._batched_compact_matrix_chain_speedup_threshold
        )
        native_compact_matrix_chain_threshold = float(
            self._native_compact_matrix_chain_speedup_threshold
        )
        compact_matrix_chain_threshold = float(self._compact_matrix_chain_speedup_threshold)
        matrix_chain_threshold = float(self._matrix_chain_speedup_threshold)
        direct_threshold = 0.5
        batched_threshold = 0.5
        if (
            batched_compact_matrix_chain is not None
            and batched_compact_matrix_chain_rel_error <= 1.0e-9
            and batched_compact_matrix_chain_seconds
            < batched_compact_matrix_chain_threshold * generic_seconds
        ):
            _set_layout_choice("batched_compact_matrix_chain")
            choice = "batched_compact_matrix_chain"
            result = batched_compact_matrix_chain
        elif (
            native_compact_matrix_chain is not None
            and native_compact_matrix_chain_rel_error <= 1.0e-9
            and native_compact_matrix_chain_seconds < native_compact_matrix_chain_threshold * generic_seconds
        ):
            _set_layout_choice("native_compact_matrix_chain")
            choice = "native_compact_matrix_chain"
            result = native_compact_matrix_chain
        elif (
            compact_matrix_chain is not None
            and compact_matrix_chain_rel_error <= 1.0e-9
            and compact_matrix_chain_seconds < compact_matrix_chain_threshold * generic_seconds
        ):
            _set_layout_choice("compact_matrix_chain")
            choice = "compact_matrix_chain"
            result = compact_matrix_chain
        elif (
            matrix_chain is not None
            and matrix_chain_rel_error <= 1.0e-9
            and matrix_chain_seconds < matrix_chain_threshold * generic_seconds
        ):
            _set_layout_choice("generic_matrix_chain")
            choice = "generic_matrix_chain"
            result = matrix_chain
        elif (
            chain is not None
            and chain_rel_error <= 1.0e-9
            and chain_seconds < chain_threshold * generic_seconds
        ):
            _set_layout_choice("generic_chain")
            choice = "generic_chain"
            result = chain
        elif (
            direct is not None
            and direct_rel_error <= 1.0e-9
            and direct_seconds < direct_threshold * generic_seconds
        ):
            _set_layout_choice("direct_operator")
            choice = "direct_operator"
            result = direct
        elif (
            batched is not None
            and rel_error <= 1.0e-9
            and batched_seconds < batched_threshold * generic_seconds
        ):
            _set_layout_choice("batched_action")
            choice = "batched_action"
            result = batched
        else:
            _set_layout_choice("generic")
            choice = "generic"
            result = generic

        self.profile_stats["action_selector"] = {
            "choice": choice,
            "generic_seconds": float(generic_seconds),
            "batched_compact_matrix_chain_seconds": (
                None
                if not np.isfinite(batched_compact_matrix_chain_seconds)
                else float(batched_compact_matrix_chain_seconds)
            ),
            "batched_compact_matrix_chain_relative_error": (
                None
                if not np.isfinite(batched_compact_matrix_chain_rel_error)
                else float(batched_compact_matrix_chain_rel_error)
            ),
            "batched_compact_matrix_chain_available": batched_compact_matrix_chain is not None,
            "batched_compact_matrix_chain_selector_enabled": bool(
                self._batched_compact_matrix_chain_selector_enabled
            ),
            "batched_compact_matrix_chain_forced": bool(
                self._batched_compact_matrix_chain_force
            ),
            "native_compact_matrix_chain_seconds": (
                None
                if not np.isfinite(native_compact_matrix_chain_seconds)
                else float(native_compact_matrix_chain_seconds)
            ),
            "native_compact_matrix_chain_relative_error": (
                None
                if not np.isfinite(native_compact_matrix_chain_rel_error)
                else float(native_compact_matrix_chain_rel_error)
            ),
            "native_compact_matrix_chain_available": native_compact_matrix_chain is not None,
            "native_compact_matrix_chain_selector_enabled": bool(
                self._native_compact_matrix_chain_selector_enabled
            ),
            "native_compact_matrix_chain_forced": bool(
                self._native_compact_matrix_chain_force
            ),
            "compact_matrix_chain_seconds": (
                None
                if not np.isfinite(compact_matrix_chain_seconds)
                else float(compact_matrix_chain_seconds)
            ),
            "compact_matrix_chain_relative_error": (
                None
                if not np.isfinite(compact_matrix_chain_rel_error)
                else float(compact_matrix_chain_rel_error)
            ),
            "compact_matrix_chain_available": compact_matrix_chain is not None,
            "compact_matrix_chain_selector_enabled": bool(
                self._compact_matrix_chain_selector_enabled
            ),
            "compact_matrix_chain_forced": bool(self._compact_matrix_chain_force),
            "matrix_chain_seconds": (
                None
                if not np.isfinite(matrix_chain_seconds)
                else float(matrix_chain_seconds)
            ),
            "matrix_chain_relative_error": (
                None
                if not np.isfinite(matrix_chain_rel_error)
                else float(matrix_chain_rel_error)
            ),
            "matrix_chain_available": matrix_chain is not None,
            "matrix_chain_selector_enabled": bool(self._matrix_chain_selector_enabled),
            "matrix_chain_forced": bool(self._matrix_chain_force),
            "chain_seconds": None if not np.isfinite(chain_seconds) else float(chain_seconds),
            "chain_relative_error": None if not np.isfinite(chain_rel_error) else float(chain_rel_error),
            "chain_available": chain is not None,
            "chain_selector_enabled": bool(self._generic_chain_selector_enabled),
            "direct_seconds": None if not np.isfinite(direct_seconds) else float(direct_seconds),
            "direct_plan_build_seconds": (
                None
                if not np.isfinite(direct_plan_build_seconds)
                else float(direct_plan_build_seconds)
            ),
            "direct_relative_error": None if not np.isfinite(direct_rel_error) else float(direct_rel_error),
            "direct_available": direct is not None,
            "direct_selector_enabled": bool(self._direct_operator_selector_enabled),
            "direct_entries": int(direct_entries),
            "direct_skipped_reason": direct_skipped_reason,
            "direct_edge_max_entries": int(self._direct_operator_selector_edge_max_entries),
            "batched_seconds": None if not np.isfinite(batched_seconds) else float(batched_seconds),
            "relative_error": None if not np.isfinite(rel_error) else float(rel_error),
            "batched_available": batched is not None,
            "batched_compact_matrix_chain_speedup_threshold": batched_compact_matrix_chain_threshold,
            "native_compact_matrix_chain_speedup_threshold": native_compact_matrix_chain_threshold,
            "compact_matrix_chain_speedup_threshold": compact_matrix_chain_threshold,
            "matrix_chain_speedup_threshold": matrix_chain_threshold,
            "chain_speedup_threshold": chain_threshold,
            "direct_speedup_threshold": direct_threshold,
            "batched_speedup_threshold": batched_threshold,
            "batched_selector_enabled": bool(self._batched_action_selector_enabled),
            "selector_per_layout": bool(self._matvec_selector_per_layout),
            "layout_choices": int(len(self._matvec_action_choice_by_layout)),
        }
        return result, choice

    def diagonal(self, proto):
        cache_key = (self._action_token(), self._layout(proto), "jacobi_diagonal")
        cached = self._diagonal_cache.get(cache_key)
        if cached is not None:
            return cached
        build_start = time.perf_counter()
        diagonal_result = abelian_flat_qchem_jacobi_diagonal(
            self._layout(proto),
            self.E,
            self.W,
            self.F,
        )

        self._record_plan_profile(
            "jacobi_diagonal",
            time.perf_counter() - build_start,
            candidate_entries=int(diagonal_result.candidate_entries),
            diagonal_contributions=int(diagonal_result.contributions),
            diagonal_blocks=int(diagonal_result.diagonal_blocks),
            backend="block_data",
            rejected_reason=diagonal_result.rejected_reason,
        )

        if diagonal_result.flat is None or diagonal_result.block_data is None:
            self._diagonal_cache[cache_key] = None
            return None
        diagonal = self._tensor_from_block_data_like(
            proto,
            diagonal_result.block_data,
            proto.qns[:],
            proto.dirs[:],
        )
        self._diagonal_cache[cache_key] = diagonal
        self.profile_stats["preconditioner"] = {
            "kind": "jacobi_diagonal",
            "available": True,
            "backend": "block_data",
            "diagonal_blocks": int(diagonal_result.diagonal_blocks),
            "diagonal_contributions": int(diagonal_result.contributions),
        }
        return diagonal

    def jacobi_preconditioner(self, proto, *, floor=1.0e-8):
        diagonal = self.diagonal(proto)
        if diagonal is None:
            self.profile_stats["preconditioner"] = {
                "kind": "jacobi_diagonal",
                "available": False,
            }
            return None

        def apply(residual, theta):
            theta = complex(theta)
            out = {}
            for key, block in residual.data.items():
                diag = diagonal.data.get(key)
                if diag is None or diag.shape != block.shape:
                    out[key] = block.copy()
                    continue
                denom = theta - diag
                finite = np.isfinite(np.real(denom)) & np.isfinite(np.imag(denom))
                small = np.abs(denom) < float(floor)
                replace = (~finite) | small
                if np.any(replace):
                    sign = np.where(np.real(denom) >= 0.0, 1.0, -1.0)
                    denom = np.where(replace, sign * float(floor), denom)
                out[key] = block / denom
            return self._tensor_from_block_data_like(
                residual,
                out,
                residual.qns[:],
                residual.dirs[:],
            )

        return apply

    @staticmethod
    def _sector_components(sector):
        labels = tuple(getattr(sector, "labels", ()))
        comps = tuple(getattr(sector, "components", ()))
        return dict(zip(labels, comps))

    @classmethod
    def _spatial_state_indices_from_qns(cls, qns):
        found = {}
        for idx, qn in enumerate(qns):
            comps = cls._sector_components(qn)
            if "charge" not in comps or "sz" not in comps:
                return None
            key = (int(comps["charge"]), int(comps["sz"]))
            if key in found:
                return None
            found[key] = int(idx)
        required = ((0, 0), (1, 1), (1, -1), (2, 0))
        if any(key not in found for key in required):
            return None
        return tuple(found[key] for key in required)

    @staticmethod
    def _axis_dims(A, axis):
        dims = {}
        for key, block in A.data.items():
            qn = key[axis]
            dim = int(block.shape[axis])
            old = dims.get(qn)
            if old is not None and old != dim:
                return None
            dims[qn] = dim
        return dims

    def local_complementary_matrix(self, A=None):
        if self.complementary_operator_families is None or self.bond is None:
            return None
        cache_key = (id(self.complementary_operator_families), int(self.bond))
        mat = self._local_complementary_cache.get(cache_key)
        if mat is None:
            mat = self._GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE.get(cache_key)
        if mat is None:
            try:
                from pyqed.qchem.dmrg.spatial_terms import (
                    spatial_complementary_local_matrix,
                )
            except Exception:
                return None
            mat = spatial_complementary_local_matrix(
                self.complementary_operator_families,
                int(self.bond),
            )
            self._local_complementary_cache[cache_key] = mat
            self._GLOBAL_LOCAL_COMPLEMENTARY_MATRIX_CACHE[cache_key] = mat
        if A is None:
            return mat
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        order = [4 * i + j for i in left_order for j in right_order]
        return mat[np.ix_(order, order)]

    def local_complementary_channel_matrices(self, A=None):
        if self.complementary_operator_families is None or self.bond is None:
            return None
        cache_key = (id(self.complementary_operator_families), int(self.bond))
        mats = self._local_complementary_channel_cache.get(cache_key)
        if mats is None:
            mats = self._GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE.get(cache_key)
        if mats is None:
            try:
                from pyqed.qchem.dmrg.spatial_terms import (
                    spatial_complementary_local_matrices,
                )
            except Exception:
                return None
            mats = spatial_complementary_local_matrices(
                self.complementary_operator_families,
                int(self.bond),
            )
            self._local_complementary_channel_cache[cache_key] = mats
            self._GLOBAL_LOCAL_COMPLEMENTARY_CHANNEL_MATRIX_CACHE[cache_key] = mats
        if A is None:
            return mats
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        order = [4 * i + j for i in left_order for j in right_order]
        return {
            name: np.asarray(mat, dtype=complex)[np.ix_(order, order)]
            for name, mat in mats.items()
        }

    @property
    def complementary_local_metadata(self):
        mat = self.local_complementary_matrix()
        if mat is None:
            return None
        return {
            "enabled": True,
            "bond": self.bond,
            "matrix_shape": tuple(int(x) for x in mat.shape),
            "nnz": int(np.count_nonzero(np.abs(mat) > 1.0e-14)),
            "norm": float(np.linalg.norm(mat)),
            "channels": {
                str(name): {
                    "matrix_shape": tuple(int(x) for x in np.asarray(channel_mat).shape),
                    "nnz": int(np.count_nonzero(np.abs(channel_mat) > 1.0e-14)),
                    "norm": float(np.linalg.norm(channel_mat)),
                }
                for name, channel_mat in (
                    self.local_complementary_channel_matrices() or {}
                ).items()
            },
        }

    def _apply_local_complementary_matrix(self, A, mat, tol=1.0e-14):
        if mat is None:
            return None
        left_order = self._spatial_state_indices_from_qns(A.qns[2])
        right_order = self._spatial_state_indices_from_qns(A.qns[3])
        if left_order is None or right_order is None:
            return None
        p1_dims = self._axis_dims(A, 2)
        p2_dims = self._axis_dims(A, 3)
        if p1_dims is None or p2_dims is None:
            return None
        p1_qns = tuple(A.qns[2])
        p2_qns = tuple(A.qns[3])
        pair_qns = tuple((q1, q2) for q1 in p1_qns for q2 in p2_qns)
        n_pair = len(pair_qns)
        mat = np.asarray(mat, dtype=complex).reshape(n_pair, n_pair)
        out = {}
        for in_key, block in A.data.items():
            ql, qr, p1_in, p2_in = in_key
            try:
                in_col = pair_qns.index((p1_in, p2_in))
            except ValueError:
                continue
            for out_row, (p1_out, p2_out) in enumerate(pair_qns):
                coeff = mat[out_row, in_col]
                if abs(coeff) <= tol:
                    continue
                p1_dim = int(p1_dims.get(p1_out, block.shape[2]))
                p2_dim = int(p2_dims.get(p2_out, block.shape[3]))
                if p1_dim != block.shape[2]:
                    continue
                if p2_dim != block.shape[3]:
                    continue
                out_key = (ql, qr, p1_out, p2_out)
                contrib = coeff * block
                if out_key in out:
                    out[out_key] += contrib
                else:
                    out[out_key] = contrib.copy()
        return self._tensor_from_block_data_like(A, out, A.qns[:], A.dirs[:])

    def _matvec_local_complementary_channels(self, A, tol=1.0e-14):
        mats = self.local_complementary_channel_matrices(A)
        if mats is None:
            return None
        out = {}
        for name, mat in mats.items():
            applied = self._apply_local_complementary_matrix(A, mat, tol=tol)
            if applied is not None and applied.norm() > tol:
                out[str(name)] = applied
        return out

    def _matvec_local_complementary(self, A, tol=1.0e-14):
        return self._apply_local_complementary_matrix(
            A,
            self.local_complementary_matrix(A),
            tol=tol,
        )

    def split_local_action(self, A):
        """
        Return the exact complementary split for this local problem.

        When complementary payload matvecs are enabled, the boundary term is
        tried through the renormalized family table/direct family environments
        before falling back to factorized middle-channel plans.  Otherwise,
        small local spaces may use a dense boundary table before falling back
        to the exact residual ``full - local``.
        """
        if self._prefer_boundary_factorized:
            local_channels = self._matvec_local_complementary_channels(A)
            if local_channels is None:
                local = None
            else:
                local = None
                for value in local_channels.values():
                    local = value if local is None else local + value
            boundary_family_table = (
                None
                if self.complementary_direct_family_environments
                and self.complementary_family_environments
                else self._matvec_boundary_family_operator_table(A)
            )
            if boundary_family_table is not None:
                boundary = boundary_family_table["total"]
                mode = (
                    "local_RP_plus_boundary_family_operator_table"
                    if local is not None
                    else "boundary_family_operator_table_no_local_complementary"
                )
                total = boundary if local is None else boundary + local
                self._audit_complementary_action(A, total, mode)
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_table=boundary_family_table.get("table"),
                    boundary_operator=boundary_family_table["stats"],
                )
                return {
                    "total": total,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_channels": boundary_family_table["channels"],
                    "boundary_operator": boundary_family_table["stats"],
                    "boundary_table": boundary_family_table.get("table"),
                    "mode": mode,
                }
            direct_symbolic = self._matvec_direct_symbolic_family_channels(A)
            if direct_symbolic is not None:
                direct_channels, direct_stats = direct_symbolic
                family_sources = {"direct": tuple(direct_channels)}
                if self.complementary_family_environments:
                    named_family = self._matvec_named_family_channels(A)
                    if named_family is not None:
                        named_family_channels, named_family_stats = named_family
                        family_sources["named"] = tuple(named_family_channels)
                        for name, tensor in named_family_channels.items():
                            if name in direct_channels:
                                direct_channels[name] = direct_channels[name] + tensor
                            else:
                                direct_channels[name] = tensor
                            if name in direct_stats:
                                old_stats = dict(direct_stats[name])
                                direct_stats[name] = {
                                    "source": "hybrid_direct_and_named_family_environment",
                                    "direct_family_stats": old_stats,
                                    "named_family_stats": named_family_stats.get(name),
                                }
                            else:
                                direct_stats[name] = named_family_stats.get(name, {})
                total = None
                boundary_channels = {}
                for name, tensor in direct_channels.items():
                    total = tensor if total is None else total + tensor
                    boundary = tensor
                    if local_channels is not None and name in local_channels:
                        boundary = boundary - local_channels[name]
                    boundary_channels[f"{name}:boundary"] = boundary
                if total is not None:
                    direct_sources = tuple(
                        str(item.get("source", ""))
                        for item in direct_stats.values()
                    )
                    precontracted = bool(direct_sources) and all(
                        source == "precontracted_direct_symbolic_term_environments"
                        for source in direct_sources
                    )
                    if precontracted:
                        mode = (
                            "direct_RP_precontracted_family_environment"
                            if local is not None
                            else "direct_precontracted_family_environment"
                        )
                    else:
                        mode = (
                            "direct_RP_symbolic_environment"
                            if local is not None
                            else "direct_symbolic_environment"
                        )
                    if "named" in family_sources:
                        mode = (
                            "hybrid_direct_named_RP_family_environment"
                            if local is not None
                            else "hybrid_direct_named_family_environment"
                        )
                    self._audit_complementary_action(A, total, mode)
                    stats = {
                        "kind": "abelian_direct_symbolic_family_environment_action",
                        "source": (
                            "hybrid_direct_and_named_family_environments"
                            if "named" in family_sources
                            else
                            "precontracted_direct_family_environments"
                            if precontracted
                            else "direct_symbolic_term_environments"
                        ),
                        "bond": self.bond,
                        "family_sources": family_sources,
                        "family_names": tuple(direct_channels),
                        "n_channels": int(
                            len(direct_channels)
                            + len(local_channels or {})
                        ),
                        "channel_names": tuple(
                            f"{name}:full" for name in direct_channels
                        )
                        + tuple(
                            f"{name}:local" for name in (local_channels or {})
                        ),
                        "channels_materialized": True,
                        "family_channel_stats": direct_stats,
                    }
                    self._record_complementary_split(
                        mode,
                        local=local,
                        local_channels=local_channels,
                        boundary=None if local is None else total - local,
                        boundary_operator=stats,
                    )
                    return {
                        "total": total,
                        "local": local,
                        "local_channels": local_channels,
                        "boundary": None if local is None else total - local,
                        "boundary_channels": boundary_channels,
                        "boundary_operator": stats,
                        "family_channels": direct_channels,
                        "mode": mode,
                    }
            named_family = self._matvec_named_family_channels(A)
            if named_family is not None:
                named_family_channels, named_family_stats = named_family
                total = None
                boundary_channels = {}
                for name, tensor in named_family_channels.items():
                    total = tensor if total is None else total + tensor
                    boundary = tensor
                    if local_channels is not None and name in local_channels:
                        boundary = boundary - local_channels[name]
                    boundary_channels[f"{name}:boundary"] = boundary
                if total is not None:
                    mode = (
                        "named_RP_family_environment"
                        if local is not None
                        else "named_family_environment"
                    )
                    self._audit_complementary_action(A, total, mode)
                    stats = {
                        "kind": "abelian_named_family_environment_action",
                        "source": "family_mpo_environments",
                        "bond": self.bond,
                        "family_names": tuple(named_family_channels),
                        "n_channels": int(
                            len(named_family_channels)
                            + len(local_channels or {})
                        ),
                        "channel_names": tuple(
                            f"{name}:full" for name in named_family_channels
                        )
                        + tuple(
                            f"{name}:local" for name in (local_channels or {})
                        ),
                        "channels_materialized": True,
                        "family_channel_stats": named_family_stats,
                    }
                    self._record_complementary_split(
                        mode,
                        local=local,
                        local_channels=local_channels,
                        boundary=None if local is None else total - local,
                        boundary_operator=stats,
                    )
                    return {
                        "total": total,
                        "local": local,
                        "local_channels": local_channels,
                        "boundary": None if local is None else total - local,
                        "boundary_channels": boundary_channels,
                        "boundary_operator": stats,
                        "family_channels": named_family_channels,
                        "mode": mode,
                    }
        else:
            local_channels = None
            local = self._matvec_local_complementary(A)
        if local is None:
            local = None
        if self._prefer_boundary_factorized:
            boundary_direct = self._matvec_boundary_direct_operator(
                A,
                local_channels=local_channels,
            )
            if boundary_direct is not None:
                boundary = boundary_direct["total"]
                mode = (
                    "local_RP_plus_boundary_direct_operator_table"
                    if local is not None
                    else "boundary_direct_operator_no_local_complementary"
                )
                total = boundary if local is None else boundary + local
                self._audit_complementary_action(A, total, mode)
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_operator=boundary_direct["stats"],
                )
                return {
                    "total": total,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_channels": boundary_direct["channels"],
                    "boundary_operator": boundary_direct["stats"],
                    "mode": mode,
                }
            boundary_factorized = self._matvec_boundary_factorized(
                A,
                local_channels=local_channels,
            )
            if boundary_factorized is not None:
                boundary = boundary_factorized["total"]
                mode = (
                    "local_RP_plus_boundary_factorized_operator_table"
                    if local is not None
                    else "boundary_factorized_no_local_complementary"
                )
                total = boundary if local is None else boundary + local
                self._audit_complementary_action(A, total, mode)
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_operator=boundary_factorized["stats"],
                )
                return {
                    "total": total,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_channels": boundary_factorized["channels"],
                    "boundary_operator": boundary_factorized["stats"],
                    "mode": mode,
                }
        if local is not None:
            boundary_table = self._boundary_table(A)
            if boundary_table is not None:
                boundary = self._tensor_from_action_table_data_like(
                    A,
                    boundary_table,
                    boundary_table.apply_data(getattr(A, "data", {}) or {}),
                )
                mode = "local_RP_plus_boundary_operator_table"
                self._record_complementary_split(
                    mode,
                    local=local,
                    local_channels=local_channels,
                    boundary=boundary,
                    boundary_table=boundary_table,
                )
                return {
                    "total": boundary + local,
                    "local": local,
                    "local_channels": local_channels,
                    "boundary": boundary,
                    "boundary_table": boundary_table,
                    "mode": mode,
                }

        full = self._matvec_generic(A)
        if local is None:
            mode = "full_mpo_no_local_complementary"
            self._record_complementary_split(mode, local=None, boundary=full)
            return {
                "total": full,
                "local": None,
                "local_channels": None,
                "boundary": full,
                "mode": mode,
            }
        boundary = full - local
        mode = "local_RP_plus_boundary_residual"
        self._record_complementary_split(
            mode,
            local=local,
            local_channels=local_channels,
            boundary=boundary,
        )
        return {
            "total": boundary + local,
            "local": local,
            "local_channels": local_channels,
            "boundary": boundary,
            "mode": mode,
        }

    def _boundary_table(self, proto):
        cap = int(self._boundary_table_max_dim)
        layout = self._closed_layout(proto, cap)
        if layout is None:
            return None
        dim = self._size(layout)
        if dim <= 0 or dim > cap:
            return None
        cache_key = (int(self.bond or 0), layout)
        cached = self._boundary_table_cache.get(cache_key)
        if cached is not None:
            return cached

        H = np.zeros((dim, dim), dtype=complex)
        channel_matrices = {}
        if self._debug_boundary_channel_matrices:
            middle_channels = self._middle_mpo_channels()
            channel_matrices = {
                f"mpo_middle:{repr(channel)}": np.zeros((dim, dim), dtype=complex)
                for channel in middle_channels
            }
            local_channel_names = tuple(
                (self.local_complementary_channel_matrices(proto) or {}).keys()
            )
            for name in local_channel_names:
                channel_matrices[f"subtract_local:{name}"] = np.zeros((dim, dim), dtype=complex)
        else:
            middle_channels = ()
        local_layout = AbelianLocalVectorLayout.from_layout(layout, proto=proto)
        qns = [list(axis_qns) for axis_qns in local_layout.qns]
        dtype = np.result_type(*[blk.dtype for blk in proto.data.values()], complex)
        for col in range(dim):
            basis = self._tensor_from_block_data_like(
                proto,
                local_layout.basis_data(col, dtype=dtype),
                qns,
                list(local_layout.dirs),
            )
            full = self._matvec_generic(basis)
            local = self._matvec_local_complementary(basis)
            if local is None:
                return None
            H[:, col] = self._flatten(full - local, layout)
            if self._debug_boundary_channel_matrices:
                for channel in middle_channels:
                    contribution = self._matvec_middle_mpo_channel(basis, channel)
                    if contribution is not None:
                        channel_matrices[f"mpo_middle:{repr(channel)}"][:, col] = (
                            self._flatten(contribution, layout)
                        )
                local_channels = self._matvec_local_complementary_channels(basis) or {}
                for name, contribution in local_channels.items():
                    channel_matrices[f"subtract_local:{name}"][:, col] = -self._flatten(
                        contribution,
                        layout,
                    )
        if channel_matrices:
            channel_sum = sum(channel_matrices.values(), np.zeros_like(H))
            unresolved = H - channel_sum
            if np.linalg.norm(unresolved) > 1.0e-10:
                channel_matrices["unresolved_residual"] = unresolved
        table = AbelianComplementaryBoundaryActionTable(
            H,
            layout,
            qns,
            list(local_layout.dirs),
            bond=self.bond,
            source="exact_full_mpo_minus_local_RP",
            boundary_family_tables=self._boundary_family_tables(),
            channel_matrices=channel_matrices,
        )
        self._boundary_table_cache[cache_key] = table
        return table

    def _matvec_boundary_table(self, A):
        table = self._boundary_table(A)
        if table is None:
            return None
        return self._tensor_from_action_table_data_like(
            A,
            table,
            table.apply_data(getattr(A, "data", {}) or {}),
        )

    def _tensor_from_action_table_data_like(self, proto, table, data):
        layout = getattr(table, "vector_layout", None)
        if layout is None:
            tensor = table.unflatten(table.flatten_data(data))
            return self._tensor_from_block_data_like(
                proto,
                tensor.data,
                tensor.qns,
                tensor.dirs,
            )
        return self._tensor_from_block_data_like(
            proto,
            data,
            [list(q) for q in layout.qns],
            list(layout.dirs),
        )

    def _boundary_family_tables(self):
        payloads = self.complementary_boundary_payloads or {}
        tables = []
        for side in ("left", "right"):
            entry = payloads.get(side)
            table = None if entry is None else entry.family_operator_table
            if table is not None:
                tables.append(table)
        return tuple(tables)

    def _boundary_family_action_key(self, A):
        layout = self._layout(A)
        return (
            "abelian_boundary_action",
            int(self.bond or 0),
            layout,
        ), layout

    def _stored_boundary_family_action_table(self, key):
        if key is None:
            return None
        for table in self._boundary_family_tables():
            getter = getattr(table, "get_numeric_action_table", None)
            if getter is None:
                continue
            action_table = getter(key)
            if action_table is not None:
                return action_table
        return None

    def _store_boundary_family_action_table(self, key, action_table):
        if key is None or action_table is None:
            return action_table
        stored = False
        for table in self._boundary_family_tables():
            putter = getattr(table, "put_numeric_action_table", None)
            if putter is None:
                continue
            putter(key, action_table)
            stored = True
        return action_table if stored else None

    def _boundary_family_action_table(self, A):
        tables = self._boundary_family_tables()
        if not tables:
            return None
        key, _layout = self._boundary_family_action_key(A)
        cached = self._stored_boundary_family_action_table(key)
        if cached is not None:
            return cached
        built = self._build_direct_family_action_table_from_plans(A)
        if built is None:
            built = self._build_family_action_table_from_direct_family_environments(A)
        if built is None:
            return None
        source = str(built.source)
        if not source.startswith("renormalized_family_operator_tables:"):
            built.source = "renormalized_family_operator_tables:" + source
        stored = self._store_boundary_family_action_table(key, built)
        return built if stored is not None else None

    def _matvec_boundary_family_operator_table(self, A):
        table = self._boundary_family_action_table(A)
        if table is None:
            return None
        boundary = self._tensor_from_action_table_data_like(
            A,
            table,
            table.apply_data(getattr(A, "data", {}) or {}),
        )
        channel_tensors = {}
        if getattr(self, "_debug_boundary_channel_matrices", False):
            channel_data = (
                table.apply_channels_data(getattr(A, "data", {}) or {})
                if getattr(table, "apply_channels_data", None) is not None
                else {}
            )
            channel_tensors = {
                name: self._tensor_from_action_table_data_like(A, table, data)
                for name, data in channel_data.items()
            }
        return {
            "total": boundary,
            "channels": channel_tensors,
            "table": table,
            "stats": {
                "kind": "abelian_complementary_boundary_family_operator_table_action",
                "source": str(table.source),
                "bond": self.bond,
                "n_family_operator_tables": int(len(self._boundary_family_tables())),
                "active_family_names": tuple(table.stats.get("active_family_names", ())),
                "n_channels": int(len(channel_tensors)),
                "channel_names": tuple(channel_tensors),
                "channels_materialized": bool(channel_tensors),
                "table": table.stats,
            },
        }

    def _compact_complementary_split_metadata(self):
        def _table_summary(entry):
            table = None if entry is None else entry.family_operator_table
            if table is None:
                return None
            return {
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            }

        def _payload_summary(entry):
            if entry is None:
                return None
            return {
                "side": str(entry.side),
                "bond": int(entry.bond),
                "numeric_payload_terms": int(
                    sum(payload.n_terms for payload in entry.family_payloads.values())
                ),
                "numeric_payload_cross_terms": int(
                    sum(payload.cross_terms for payload in entry.family_payloads.values())
                ),
            }

        payloads = self.complementary_boundary_payloads or {}
        left = payloads.get("left")
        right = payloads.get("right")
        local = self.complementary_local_metadata
        local_summary = None if local is None else {
            "bond": local.get("bond"),
            "matrix_shape": local.get("matrix_shape"),
            "nnz": local.get("nnz"),
            "norm": local.get("norm"),
            "channels": local.get("channels", {}),
        }
        return {
            "enabled": self.complementary_operator_families is not None,
            "bond": self.bond,
            "local": local_summary,
            "boundary_table_cache_entries": int(len(self._boundary_table_cache)),
            "left_family_operator_table": _table_summary(left),
            "right_family_operator_table": _table_summary(right),
            "left_payload": _payload_summary(left),
            "right_payload": _payload_summary(right),
        }

    @property
    def complementary_split_metadata(self):
        payloads = self.complementary_boundary_payloads or {}
        left = payloads.get("left")
        right = payloads.get("right")
        return {
            "enabled": self.complementary_operator_families is not None,
            "bond": self.bond,
            "local": self.complementary_local_metadata,
            "boundary_source": "cached_boundary_table_or_full_mpo_residual",
            "boundary_table_max_dim": int(self._boundary_table_max_dim),
            "boundary_table_cache_entries": int(len(self._boundary_table_cache)),
            "boundary_action_tables": tuple(
                table.stats for table in self._boundary_table_cache.values()
            ),
            "left_family_operator_table": (
                None
                if left is None or left.family_operator_table is None
                else left.family_operator_table.stats
            ),
            "right_family_operator_table": (
                None
                if right is None or right.family_operator_table is None
                else right.family_operator_table.stats
            ),
            "left_payload": None if left is None else left.stats,
            "right_payload": None if right is None else right.stats,
        }

    def matvec(self, A):
        start = time.perf_counter()
        path = "generic"
        try:
            if self.complementary_operator_families is not None:
                split = self.split_local_action(A)
                path = "complementary:" + str(split.get("mode", "unknown"))
                return split["total"]
            A_new = self._matvec_compiled(A)
            if A_new is not None:
                path = "compiled_small"
                return A_new
            A_new, path = self._matvec_auto_block_action(A)
            return A_new
        finally:
            self._record_matvec_profile(path, time.perf_counter() - start)
