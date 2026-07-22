#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:15:58 2026

@author: Shuoyi Hu, Sha Mo, Bing Gu
"""

from pyqed.mps import MPS, MPO, fDMRG_1site_GS_OBC, two_site_dmrg, dense_to_symmetric,\
    expect_mps
from pyqed.mps.mps import (
    _abelian_data_factor_list,
    initial_E,
    contract_from_left,
    svd_symmetric,
    multiply_U_S,
    multiply_S_V,
)
from pyqed.mps.abelian_direct import (
    AbelianSiteTensorData,
    abelian_right_canonicalize_site_tensors,
)
from pyqed.mps.abelian_storage import (
    SymmetryManager,
    abelian_environment_scalar,
    is_legacy_abelian_tensor,
    legacy_tensordot,
    make_identity_mpo_site_from_mps_site,
)
import numpy as np
import pickle
from pathlib import Path


def _contract_mps_mpo(factors, mpo):
    if len(factors) != len(mpo):
        raise ValueError("MPS and MPO lengths must match.")
    env = initial_E(mpo[0])
    for site_mpo, site in zip(mpo, factors):
        env = contract_from_left(site_mpo, site, env, site)
    return abelian_environment_scalar(env)


def _mps_norm(factors):
    identity_mpo = [make_identity_mpo_site_from_mps_site(site) for site in factors]
    norm = _contract_mps_mpo(factors, identity_mpo)
    norm = np.real_if_close(norm)
    return float(np.real(norm))


def _normalize_mps_state(state):
    """Normalize an MPS in-place, including Abelian block-data carriers."""
    norm = _mps_norm(state.factors)
    if norm <= 1.0e-14:
        raise ValueError("cannot normalize a near-zero DMRG state.")
    scale = 1.0 / np.sqrt(norm)
    state.factors[0] = state.factors[0] * scale
    state.Bs = state.data = state.factors
    return state


def _normalize_factors_in_place(factors):
    norm = _mps_norm(factors)
    if norm <= 1.0e-14:
        raise ValueError("cannot normalize a near-zero DMRG state.")
    factors[0] = factors[0] * (1.0 / np.sqrt(norm))
    return factors


def _right_canonicalize_symmetric_factors(factors, max_bond_dim=None):
    """Put an Abelian MPS in right-canonical form for a left-to-right sweep."""
    if factors and isinstance(factors[0], AbelianSiteTensorData):
        out = abelian_right_canonicalize_site_tensors(
            factors,
            max_bond_dim=max_bond_dim,
        )
        return _normalize_factors_in_place(out)

    if not (factors and is_legacy_abelian_tensor(factors[0])):
        return factors
    out = [factor.copy() if hasattr(factor, "copy") else factor for factor in factors]
    for site in range(len(out) - 1, 0, -1):
        AA = legacy_tensordot(
            out[site - 1],
            out[site],
            axes=([1], [0]),
        ).transpose(0, 2, 1, 3)
        U, V, S_dict, _trunc, _m_kept = svd_symmetric(AA, m_max=max_bond_dim)
        out[site - 1] = multiply_U_S(U, S_dict).transpose(0, 2, 1)
        out[site] = V
    return _normalize_factors_in_place(out)


def _normalized_mps_mpo_expectation(factors, mpo):
    norm = _mps_norm(factors)
    if norm <= 1.0e-14:
        raise ValueError("cannot evaluate the energy of a near-zero DMRG state.")
    energy = _contract_mps_mpo(factors, mpo) / norm
    energy = np.real_if_close(energy)
    return float(np.real(energy))


def dmrg_matvec_options(policy="auto"):
    """Return canonical Abelian two-site matvec/local-solver options.

    The low-level Abelian DMRG implementation exposes many experimental knobs.
    Public callers should normally choose one policy here and pass targeted
    overrides only when benchmarking.
    """

    policy = str(policy or "auto").strip().lower().replace("_", "-")
    aliases = {
        "default": "auto",
        "safe": "auto",
        "block2": "packed-cpp-fast",
        "block2-like": "packed-cpp-fast",
        "block2-style": "packed-cpp-fast",
        "packed-block2-style": "packed-cpp-fast",
        "cpp": "packed-cpp-fast",
        "c++": "packed-cpp-fast",
        "block2-cpp": "packed-cpp-fast",
        "projector-fast": "packed-projector-fast",
    }
    policy = aliases.get(policy, policy)
    if policy == "auto":
        return dmrg_matvec_options("packed-cpp-fast")
    if policy in {"legacy-auto", "selector-auto"}:
        return {
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_speedup_threshold": 0.95,
        }
    if policy == "fast":
        return {
            "native_site_storage": True,
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
        }
    if policy == "compiled-fast":
        return {
            "native_site_storage": True,
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "batched_compact_matrix_chain_compiled_kernel": True,
        }
    if policy == "packed-fast":
        return {
            "native_site_storage": True,
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 32768,
            "packed_local_davidson_max_iter": 24,
            "packed_local_project_current_support": True,
            "packed_local_project_current_support_truncate": True,
            "packed_local_accept_projected_unconverged": True,
            "packed_local_accept_unconverged": True,
        }
    if policy == "packed-projector-fast":
        opts = dmrg_matvec_options("packed-fast")
        opts.update(
            {
                "packed_local_davidson_max_dim": 262144,
                "packed_local_fallback_warm_start_max_dim": 262144,
                "batched_compact_matrix_chain_compiled_parallel_kernel": True,
                "batched_compact_matrix_chain_compiled_parallel_min_work": 22000,
                "packed_local_flat_matvec": True,
                "packed_local_flat_projected_matvec": True,
                "packed_local_flat_preconditioner": True,
                "packed_local_projected_accept_min_retained_norm": 0.999,
                "packed_local_projected_accept_max_residual": 1.0e-5,
                "packed_local_return_current_on_rejected_projected": True,
            }
        )
        return opts
    if policy == "packed-compiled-fast":
        opts = dmrg_matvec_options("packed-fast")
        opts.update(
            {
                "packed_local_davidson_max_dim": 1048576,
                "packed_local_large_safe_max_dim": 4194304,
                "packed_local_large_safe_restart_dim": 10,
                "packed_local_large_safe_require_flat": True,
                "packed_local_fallback_warm_start_max_dim": 4194304,
                "packed_local_davidson_max_iter": 48,
                "packed_local_davidson_restart_dim": 64,
                "packed_local_project_current_support": False,
                "packed_local_project_current_support_truncate": False,
                "packed_local_accept_projected_unconverged": False,
                "packed_local_projected_accept_min_retained_norm": 0.0,
                "packed_local_projected_accept_max_residual": 0.0,
                "packed_local_return_current_on_rejected_projected": False,
                "packed_local_disable_generic_fallback": True,
                "batched_compact_matrix_chain_compiled_kernel": True,
                "batched_compact_matrix_chain_compiled_parallel_kernel": True,
                "batched_compact_matrix_chain_compiled_parallel_min_work": 22000,
                "packed_local_flat_matvec": True,
                "packed_local_flat_projected_matvec": False,
                "packed_local_flat_preconditioner": True,
                "moving_environment": True,
                "moving_environment_flat_preconditioner": True,
                "packed_local_family_flat_direct_matvec": True,
                "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
                "generator_table_packed_route_table": "auto",
                "generator_table_precompute_contextual_boundaries": False,
                "generator_table_exact_component_compression_fast_max_group_size": 1,
            }
        )
        return opts
    if policy == "packed-cpp-fast":
        opts = dmrg_matvec_options("packed-compiled-fast")
        opts.update(
            {
                "moving_environment_cpp_davidson": True,
                "moving_environment_cpp_accept_unconverged": False,
                "moving_environment_cpp_validate_solution": True,
                "moving_environment_cpp_solution_residual_tol_factor": 25.0,
                "moving_environment_cpp_solution_residual_abs_tol": 1.0e-9,
                "moving_environment_cpp_validate_matvec": False,
                "moving_environment_cpp_validate_matvec_random_vectors": 0,
                "moving_environment_cpp_compact_plan": True,
                "moving_environment_cpp_compact_plan_matvec": True,
                "moving_environment_cpp_compact_plan_bond_slots": True,
                "moving_environment_cpp_state_owner": True,
                "moving_environment_operatorless_local_problem": True,
                "moving_environment_cpp_site_split_owner": True,
                "moving_environment_cpp_sweep_cursor": True,
                "moving_environment_cpp_owner_half_sweep_typed_records": True,
                "moving_environment_cpp_owner_half_sweep_step_records": True,
                "moving_environment_compact_block_table": True,
                "moving_environment_compact_block_table_max_dim": 4096,
                "moving_environment_cpp_grouped_renormalized_table": True,
                "moving_environment_cpp_grouped_factorized_table": False,
                "moving_environment_cpp_grouped_raw_table": True,
                "moving_environment_cpp_raw_payload_builder": True,
                "moving_environment_cpp_raw_payload_stack_kernels": True,
                "moving_environment_cpp_named_raw_payload_builder": True,
                "moving_environment_cpp_named_raw_payload_plan": True,
                "moving_environment_cpp_raw_route_plan": True,
                "moving_environment_cpp_raw_route_plan_rebind_layout": True,
                "generator_table_packed_route_table": "auto",
                "generator_table_packed_boundary_tensors": True,
                "generator_table_allow_unpacked_boundary_tensor_fallback": False,
                "generator_table_allow_legacy_blocktensor_boundary_tables": False,
                "generator_table_allow_reference_validation_fallback": False,
                "generator_table_precompute_contextual_boundaries": True,
                "generator_table_precompute_contextual_boundaries_min_records": 0,
                "generator_table_planned_contextual_without_precompute": True,
                "generator_table_planned_contextual_without_precompute_table_lookup": True,
                "generator_table_packed_direct_family_entries": True,
                "generator_table_packed_direct_family_entries_reason": (
                    "enabled for exact planned packed contextual route"
                ),
                "generator_table_allow_planned_packed_contextual_entries": True,
                "generator_table_allow_table_backed_planned_contextual_entries": (
                    "auto"
                ),
                "generator_table_prebuild_same_side_native_p": True,
                "generator_table_incremental_same_side_pair_prebuild": True,
                "generator_table_native_boundary_p_policy": "auto",
                "generator_table_native_boundary_p_auto_max_terms": 1024,
                "generator_table_use_disjoint_same_side_native_p": False,
                "generator_table_use_true_packed_identity_entries": False,
                "generator_table_planned_native_p_identity_entries": True,
                "moving_environment_cpp_grouped_bond_slots": True,
                "moving_environment_cpp_environment_update": True,
                "packed_local_family_flat_group_identity_csr": True,
                "packed_local_family_flat_group_local_generator_csr": True,
            }
        )
        return opts
    if policy == "packed-block-fast":
        return {
            "native_site_storage": True,
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 8192,
            "packed_local_davidson_max_iter": 80,
            "packed_local_davidson_restart_dim": 32,
            "packed_local_project_current_support": True,
            "packed_local_project_current_support_truncate": True,
            "packed_local_accept_projected_unconverged": True,
            "packed_local_accept_unconverged": True,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 16,
            "packed_local_block_preconditioner_max_total_dim": 128,
        }
    if policy == "packed-block-full":
        return {
            "native_site_storage": True,
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 20000,
            "packed_local_davidson_max_iter": 80,
            "packed_local_davidson_restart_dim": 48,
            "packed_local_project_current_support": True,
            "packed_local_project_current_support_truncate": True,
            "packed_local_accept_projected_unconverged": True,
            "packed_local_accept_unconverged": True,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 64,
            "packed_local_block_preconditioner_max_total_dim": 512,
        }
    if policy == "generic":
        return {
            "direct_operator_selector_enabled": False,
            "generic_chain_selector_enabled": False,
            "matrix_chain_selector_enabled": False,
            "compact_matrix_chain_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": False,
            "native_compact_matrix_chain_selector_enabled": False,
            "batched_action_selector_enabled": False,
        }
    if policy == "matrix-chain":
        return {
            "matrix_chain_selector_enabled": True,
            "matrix_chain_speedup_threshold": 0.95,
        }
    if policy == "compact-chain":
        return {
            "compact_matrix_chain_selector_enabled": True,
            "compact_matrix_chain_speedup_threshold": 0.95,
        }
    if policy == "batched-compact-chain":
        return {
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_speedup_threshold": 0.95,
        }
    if policy == "native-compact-chain":
        return {
            "native_compact_matrix_chain_selector_enabled": True,
            "native_compact_matrix_chain_speedup_threshold": 0.95,
        }
    if policy == "force-matrix-chain":
        return {
            "direct_operator_selector_enabled": False,
            "matrix_chain_selector_enabled": True,
            "matrix_chain_force": True,
        }
    if policy == "force-compact-chain":
        return {
            "direct_operator_selector_enabled": False,
            "compact_matrix_chain_selector_enabled": True,
            "compact_matrix_chain_force": True,
        }
    if policy == "force-batched-compact-chain":
        return {
            "direct_operator_selector_enabled": False,
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
        }
    if policy == "force-native-compact-chain":
        return {
            "direct_operator_selector_enabled": False,
            "native_compact_matrix_chain_selector_enabled": True,
            "native_compact_matrix_chain_force": True,
        }
    if policy == "probe-all":
        return {
            "generic_chain_selector_enabled": True,
            "matrix_chain_selector_enabled": True,
            "compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_selector_enabled": True,
            "native_compact_matrix_chain_selector_enabled": True,
            "matrix_chain_speedup_threshold": 0.95,
            "compact_matrix_chain_speedup_threshold": 0.95,
            "batched_compact_matrix_chain_speedup_threshold": 0.95,
            "native_compact_matrix_chain_speedup_threshold": 0.95,
            "batched_action_selector_enabled": True,
        }
    raise ValueError(f"Unknown DMRG performance policy {policy!r}.")


def resolve_abelian_matvec_options(performance="auto", overrides=None):
    """Resolve public DMRG performance policy plus optional raw overrides."""

    options = dmrg_matvec_options(performance)
    if overrides:
        options.update(dict(overrides))
    if (
        bool(options.get("moving_environment_cpp_davidson", False))
        or bool(options.get("moving_environment_cpp_matvec", False))
    ):
        options.setdefault("native_site_storage", True)
        options.setdefault("moving_environment_cpp_state_owner", True)
        options.setdefault("moving_environment_cpp_solve_site_update_owner", True)
    return options


class DMRG:
    """
    ground state finite DMRG in MPO/MPS framework
    """
    def __init__(self, H, D, init_guess=None, nsweeps=50, opt='2site',\
                symmetry=False, charge=None, spin = None,\
                target_qn = None, sym_mgr = None, not_conv_err=True,
                nstates=1, weights=None, verbose=0, sweep_callback=None,
                sweep_tol=1e-6, davidson_tol=1e-5, davidson_max_iter=30,
                noise=1e-4, noise_decay=0.1, noise_cutoff=1e-9,
                local_dense_max_dim=0, complementary_operator_families=None,
                complementary_operator_mpos=None,
                complementary_operator_term_maps=None,
                complementary_operator_generator_entries=None,
                site_qn_maps=None, checkpoint_path=None, resume_from=None,
                checkpoint_interval=1, recenter_final=True,
                final_expectation=True, performance="auto",
                abelian_matvec_options=None):
        """
        Parameters
        ----------
        H : MPO
            MPO of the Hamiltonian.
        D : int
            maximum bond dimension.
        nsweeps : int
            Number of sweeps to perform.
        nstates : int
            Number of states for State-Averaged DMRG.
        weights : list
            Weights for state averaging.
        """

        self.H = H
        self.L = len(self.H)
        self.D = D
        self.nsweeps = nsweeps
        self.sweep_tol = float(sweep_tol)
        self.davidson_tol = float(davidson_tol)
        self.davidson_max_iter = int(davidson_max_iter)
        self.noise = 0.0 if noise is None else float(noise)
        self.noise_decay = float(noise_decay)
        self.noise_cutoff = float(noise_cutoff)
        # Optional Abelian dense local solve cap; 0 keeps the Davidson path.
        self.local_dense_max_dim = local_dense_max_dim
        self.complementary_operator_families = complementary_operator_families
        self.complementary_operator_mpos = complementary_operator_mpos
        self.complementary_operator_term_maps = complementary_operator_term_maps
        self.complementary_operator_generator_entries = complementary_operator_generator_entries
        self.site_qn_maps = site_qn_maps
        self.checkpoint_path = checkpoint_path
        self.resume_from = resume_from
        self.checkpoint_interval = max(1, int(checkpoint_interval or 1))
        self.recenter_final = bool(recenter_final)
        self.final_expectation = bool(final_expectation)
        self.performance = str(performance or "auto")
        self.abelian_matvec_options = resolve_abelian_matvec_options(
            self.performance,
            abelian_matvec_options,
        )
        self.opt = opt

        self.init_guess = init_guess
        self.e_tot = None
        self.U1 = self.symmetry = symmetry
        

        self.nstates = nstates
        self.weights = weights if weights is not None else [1.0/nstates]*nstates

        # Symmetry Logic
        if target_qn is not None and (sym_mgr is None):
            raise ValueError("Symmetry manager must be provided when target quantum number is specified.")
        elif target_qn is None and sym_mgr is not None:
            raise ValueError("Target quantum number must be specified when sym_mgr is given.")
        elif (charge is not None) and (spin is not None):
            sym_mgr = SymmetryManager(['charge', 'sz'])
            target_qn = sym_mgr.get_target_qn(charge, 2*spin)
        elif (charge is not None) and (spin is None):
            sym_mgr = SymmetryManager(['charge'])
            target_qn = sym_mgr.get_target_qn(charge)
        elif (charge is None) and (spin is not None):
            sym_mgr = SymmetryManager(['sz'])
            target_qn = sym_mgr.get_target_qn(2*spin)
            
        self.charge = charge
        self.target_qn = target_qn 
        self.sym_mgr = sym_mgr

        self.ground_state = None # Holds Root 0
        self.states = None       # Holds list of all Roots
        self.not_conv_err = not_conv_err
        self.converged = False
        self.verbose = int(verbose)
        self.sweep_callback = sweep_callback
        self.sweep_history = []
        self.complementary_operator_stack_stats = None
        self.complementary_split_stats = None
        self.environment_profile = None
        self.resume_payload = None

    @staticmethod
    def _copy_factors(factors):
        out = []
        for tensor in factors:
            if hasattr(tensor, "copy"):
                out.append(tensor.copy())
            else:
                out.append(np.asarray(tensor).copy())
        return out

    @staticmethod
    def load_checkpoint(path):
        with Path(path).expanduser().open("rb") as handle:
            return pickle.load(handle)

    def _write_checkpoint(self, *, factors, row=None, final=False, energy=None, gauge=None):
        if self.checkpoint_path is None:
            return None
        path = Path(self.checkpoint_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {} if row is None else dict(row)
        payload = {
            "version": 1,
            "final": bool(final),
            "mps": self._copy_factors(factors),
            "sweep_history": list(self.sweep_history),
            "completed_sweeps": int(row.get("sweep", -1)) + 1 if row else len(self.sweep_history),
            "last_sweep": row,
            "energy": None if energy is None else float(np.real(np.asarray(energy).reshape(-1)[0])),
            "gauge": gauge,
            "params": {
                "length": int(self.L),
                "bond_dim": int(self.D),
                "nsweeps": int(self.nsweeps),
                "symmetry": bool(self.symmetry),
                "target_qn": self.target_qn,
                "opt": self.opt,
                "performance": self.performance,
                "native_site_storage": bool(
                    self.abelian_matvec_options.get("native_site_storage", False)
                ),
            },
        }
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return path

    def run(self):

        resume_history = []
        resume_sweep_offset = 0
        if self.resume_from is not None:
            self.resume_payload = self.load_checkpoint(self.resume_from)
            if "mps" not in self.resume_payload:
                raise ValueError(f"DMRG checkpoint {self.resume_from!r} does not contain an MPS.")
            self.init_guess = self._copy_factors(self.resume_payload["mps"])
            resume_history = list(self.resume_payload.get("sweep_history", []))
            resume_sweep_offset = int(self.resume_payload.get("completed_sweeps", len(resume_history)))

        if self.init_guess is None:
            raise ValueError('Please provide an initial guess.')

        # Standardize MPS to ['lv', 'p', 'rv']
        # but currently we are not using the initial guess as MPS objects a lot, but i do think that is the better option. so need to fix initial guess in dmrg.py. remve this TODO when fixed.
        if isinstance(self.init_guess, MPS):
            if self.symmetry and hasattr(self.init_guess.factors[0], 'qns'):
                # U(1) branch uses (L, R, P) ordering before optional data conversion.
                mps_list = self.init_guess.to_order(['lv', 'rv', 'p']).factors
            else:
                mps_list = self.init_guess.to_order(['lv', 'p', 'rv']).factors
        else:
            # If it's a raw list, we assume it respects the convention. TODO: maybe add auto check and warning and raise error.
            mps_list = self.init_guess

        mpo_list = self.H.factors if isinstance(self.H, MPO) else self.H
        use_native_site_storage = bool(
            self.abelian_matvec_options.get("native_site_storage", False)
        )

        if self.symmetry and isinstance(mps_list[0], AbelianSiteTensorData):
            mps_list = _right_canonicalize_symmetric_factors(mps_list, max_bond_dim=self.D)
        elif self.symmetry and not is_legacy_abelian_tensor(mps_list[0]):
            phys_qns = None
            if self.site_qn_maps is not None:
                first_map = self.site_qn_maps[0]
                phys_qns = [first_map[i] for i in sorted(first_map)]
            mps_list = dense_to_symmetric(
                mps_list,
                phys_qns=phys_qns,
                native_site_storage=use_native_site_storage,
            )
            mps_list = _right_canonicalize_symmetric_factors(mps_list, max_bond_dim=self.D)
        elif self.symmetry and is_legacy_abelian_tensor(mps_list[0]):
            mps_list = _right_canonicalize_symmetric_factors(mps_list, max_bond_dim=self.D)
        elif not (self.symmetry and hasattr(mps_list[0], "qns")):
            # The two-site sweep assumes environments are built from a
            # canonical state.  Put dense initial guesses in right-canonical
            # form so the first left-to-right local problem has an identity
            # norm on the right block, matching the non-Abelian sweep contract.
            mps_list = MPS(mps_list, labels=["lv", "p", "rv"]).right_canonicalize().factors

        if (
            self.symmetry
            and use_native_site_storage
        ):
            mps_list = _abelian_data_factor_list(
                mps_list,
                native_site_storage=True,
            )
            mpo_list = _abelian_data_factor_list(
                mpo_list,
                native_site_storage=True,
            )

        if self.opt == '1site':

            fDMRG_1site_GS_OBC(mpo_list, self.D, self.nsweeps)

        elif self.opt == '2site':
            self.sweep_history = resume_history

            def cb(**info):
                row = dict(info)
                if "sweep" in row:
                    row["sweep"] = int(row["sweep"]) + resume_sweep_offset
                for key in ("energy", "truncation"):
                    val = row.get(key)
                    try:
                        row[key] = float(np.real(np.asarray(val).reshape(-1)[0]))
                    except Exception:
                        pass
                # Keep history metadata light; checkpoints carry tensor data.
                row.pop("mps", None)
                row.pop("last_AA_list", None)
                self.sweep_history.append(row)
                if (
                    self.checkpoint_path is not None
                    and (int(row.get("sweep", 0)) + 1) % self.checkpoint_interval == 0
                ):
                    self._write_checkpoint(
                        factors=info["mps"],
                        row=row,
                        final=False,
                        energy=row.get("energy"),
                        gauge=row.get("gauge"),
                    )
                if self.sweep_callback is not None:
                    callback_info = dict(info)
                    callback_info["sweep"] = row.get("sweep", callback_info.get("sweep"))
                    self.sweep_callback(**callback_info)

            res = two_site_dmrg(
                mps_list, mpo_list, self.D, self.nsweeps, 
                U1=self.U1, target_qn=self.target_qn, 
                not_conv_err=self.not_conv_err, sym_mgr=self.sym_mgr,
                nstates=self.nstates, weights=self.weights,
                verbose=self.verbose,
                conv=self.sweep_tol,
                sweep_callback=cb,
                davidson_tol=self.davidson_tol,
                davidson_max_iter=self.davidson_max_iter,
                noise=self.noise,
                noise_decay=self.noise_decay,
                noise_cutoff=self.noise_cutoff,
                local_dense_max_dim=self.local_dense_max_dim,
                complementary_operator_families=self.complementary_operator_families,
                complementary_operator_mpos=self.complementary_operator_mpos,
                complementary_operator_term_maps=self.complementary_operator_term_maps,
                complementary_operator_generator_entries=self.complementary_operator_generator_entries,
                site_qn_maps=self.site_qn_maps,
                recenter_final=self.recenter_final,
                abelian_matvec_options=self.abelian_matvec_options,
            )
            e_elec, mps_out, self.gauge, self.converged = res
            if not self.sweep_history:
                try:
                    diagnostic_energy = float(np.real(np.asarray(e_elec).reshape(-1)[0]))
                except Exception:
                    diagnostic_energy = e_elec
                self.sweep_history.append(
                    {
                        "sweep": 0,
                        "direction": "local",
                        "energy": diagnostic_energy,
                        "truncation": None,
                        "states_kept": None,
                        "gauge": self.gauge,
                    }
                )
            if self.sweep_history:
                self.complementary_operator_stack_stats = self.sweep_history[-1].get(
                    "complementary_operator_stack"
                )
                self.complementary_split_stats = self.sweep_history[-1].get(
                    "complementary_split_stats"
                )
                self.environment_profile = self.sweep_history[-1].get(
                    "environment_profile"
                )

            shift = getattr(self.H, 'constant', 0.0)
            
            labels = ['lv', 'rv', 'p'] if self.U1 else ['lv', 'p', 'rv']
            center = (len(self.H) - 1) if self.gauge.lower() == "left" else 0

            if self.nstates == 1:
                self.ground_state = MPS(mps_out, labels=labels, center=center)
                self.states = [self.ground_state]
            else:
                self.states = [MPS(s, labels=labels, center=center) for s in mps_out]
                self.ground_state = self.states[0]

            for s in self.states:
                if self.gauge.lower() == "left": 
                    s.left_canonicalize()
                else: 
                    s.right_canonicalize()
                _normalize_mps_state(s)

            if self.final_expectation:
                state_energies = [
                    _normalized_mps_mpo_expectation(s.factors, mpo_list) + shift
                    for s in self.states
                ]
            else:
                if not self.sweep_history:
                    raise ValueError("Cannot skip final expectation without a sweep energy.")
                local_energy = self.sweep_history[-1].get("energy")
                if local_energy is None:
                    raise ValueError("Cannot skip final expectation without a local sweep energy.")
                if self.nstates == 1:
                    state_energies = [float(np.real(local_energy)) + shift]
                else:
                    state_energies = [
                        float(np.real(np.asarray(energy).reshape(-1)[0])) + shift
                        for energy in np.asarray(local_energy).reshape(-1)
                    ]
            if self.nstates == 1:
                self.e_tot = state_energies[0]
            else:
                self.e_tot = state_energies

            self._write_checkpoint(
                factors=self.ground_state.factors,
                row=self.sweep_history[-1] if self.sweep_history else None,
                final=True,
                energy=self.e_tot if self.nstates == 1 else self.e_tot[0],
                gauge=self.gauge,
            )

        return self
    def expect(self, e_ops):
        """
        Compute expectation value of ground states

        Parameters
        ----------
        e_ops : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """

        psi = self.ground_state

        return [expect_mps(psi, e_op) for e_op in e_ops]

    def make_rdm1(self):
        """
        Calculate the global 1-site reduced density matrix of the optimized ground state.
        
        Wrapper for `MPS.make_rdm1`. Computes the matrix $\\gamma_{ij} = \\langle 0 | c_i^\\dagger c_j | 0 \\rangle$.

        Parameters
        ----------
        idx : optional
            Placeholder parameter to maintain API compatibility. Currently ignored as 
            the function computes the full `(L, L)` global matrix. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L)` representing the global 1-RDM.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_rdm1(sym_mgr=self.sym_mgr)

    def make_local_site_rdm(self, idx=None):
        """
        Calculate the local reduced density matrices for individual, isolated sites.
        
        Wrapper for `MPS._calc_local_site_rdms`. Traces out the rest of the chain 
        to isolate the internal $d \\times d$ quantum state of specific sites.

        Parameters
        ----------
        idx : int or list of int, optional
            The specific site index (or indices) to evaluate. If None, evaluates 
            the local density matrices for all sites in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices to their corresponding 
            $d \\times d$ local density matrices (as numpy arrays).
        """
        return self.ground_state._calc_local_site_rdms(idx=idx)

    def make_rdm2(self, idx_pairs=None):
        """
        Calculate the full global 2-site reduced density matrix of the ground state.
        
        Wrapper for `MPS.make_rdm2`. Computes the complete $\\mathcal{O}(L^4)$ tensor 
        $\\Gamma_{pqrs} = \\langle c_p^\\dagger c_r^\\dagger c_s c_q \\rangle$.

        Parameters
        ----------
        idx_pairs : optional
            Placeholder parameter to maintain API compatibility. Currently ignored as 
            the function computes the full `(L, L, L, L)` global tensor. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L, L, L)`.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_rdm2(sym_mgr=self.sym_mgr)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the diagonal blocks of the 2-site reduced density matrix.
        
        Wrapper for `MPS.make_diagonal_rdm2`. Extracts the two-site quantum state $\\rho_{ij}$ needed to compute density-density correlations like $\\langle n_i n_j \\rangle$ without evaluating the full $\\mathcal{O}(L^4)$ tensor.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the 2-site RDM for. 
            If None, computes RDMs for all possible unique pairs. By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding 
            dense reduced density matrix numpy array.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_diagonal_rdm2(idx_pairs=idx_pairs)


if __name__ == '__main__':

    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dmrg = DMRG(H, D=20, nsweeps=8)
    dmrg.init_guess = neel
    dmrg.run()
    
