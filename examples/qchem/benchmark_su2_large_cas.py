#!/usr/bin/env python3
"""Large-CAS architectural benchmark for PyQED SU(2) DMRG versus block2."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.hf import RHF


PRESETS = {
    name: {
        "atom": "; ".join(f"H 0 0 {1.6 * i}" for i in range(ncas)),
        "ncas": ncas,
        "nelecas": ncas,
        "bond_dim": bond_dim,
    }
    for name, ncas, bond_dim in (
        ("h8", 8, 32),
        ("h10", 10, 64),
        ("h12", 12, 128),
        ("h14", 14, 128),
        ("h16", 16, 128),
    )
}


def _rss_bytes():
    """Return process peak RSS in bytes on Linux and macOS."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_rss_bytes():
    """Return current RSS, falling back to the high-water mark."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return _rss_bytes()


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _sum_nested_timings(history, key):
    totals = {}
    for sweep in history:
        for objective in sweep.get("bond_objectives") or ():
            for name, value in (objective.get(key) or {}).items():
                totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def _compact_operator_stats(stats):
    output_keys = {
        "kind": "kind",
        "source": "source",
        "orthonormal_dim": "orthonormal_dim",
        "n_blocks": "n_blocks",
        "stored_kernel_elements": "stored_kernel_elements",
        "component_parent_block_kernel": "component_parent_block_kernel",
        "component_orthonormal_block_kernel": "component_orthonormal_block_kernel",
        "component_orthonormal_dense_kernel": "component_orthonormal_dense_kernel",
        "cpp_block_table": "cpp_block_table",
        "cpp_factor_route_projection": "cpp_factor_route_projection",
        "cpp_block_table_stats": "cpp_block_table_stats",
        "su2_kernel_backend_actual": "su2_kernel_backend_actual",
        "oversized_parent_block_fallback": "oversized_parent_block_fallback",
    }
    return {
        output_key: stats[input_key]
        for input_key, output_key in output_keys.items()
        if input_key in stats
    }


def _compact_bond_memory(objective):
    stats = objective.get("renormalized_operator_table_stats") or {}
    cpp_stats = stats.get("cpp_block_table_stats") or {}
    plan = stats.get("su2_qchem_sweep_plan") or {}
    left_pool = ((plan.get("left_factor_table") or {}).get("factor_pool") or {})
    right_pool = ((plan.get("right_factor_table") or {}).get("factor_pool") or {})
    memory = objective.get("memory_profile") or {}
    return {
        "bond": int(objective.get("bond", -1)),
        **{str(key): int(value) for key, value in memory.items()},
        "orthonormal_dim": int(stats.get("orthonormal_dim", 0)),
        "parent_dim": int(stats.get("parent_dim", 0)),
        "max_component_parent_dim": int(
            stats.get("max_component_parent_dim", 0)
        ),
        "metric_storage_elements": int(
            stats.get("metric_storage_elements", 0)
        ),
        "transform_storage_elements": int(
            stats.get("transform_storage_elements", 0)
        ),
        "parent_block_elements": int(stats.get("component_parent_block_elements", 0)),
        "orthonormal_block_elements": int(
            stats.get("component_orthonormal_block_elements", 0)
        ),
        "orthonormal_dense_elements": int(
            stats.get("component_orthonormal_dense_elements", 0)
        ),
        "oversized_parent_block_fallback": bool(
            stats.get("oversized_parent_block_fallback", False)
        ),
        "packed_dimension": int(objective.get("packed_dimension", 0)),
        "davidson_iterations": int(
            objective.get("davidson_iterations", 0)
        ),
        "davidson_converged": bool(
            objective.get("davidson_converged", False)
        ),
        "davidson_restarts": int(objective.get("restarts", 0)),
        "davidson_matvecs": int(objective.get("matvec_count", 0)),
        "davidson_residual": float(objective.get("residual", 0.0)),
        "requested_max_space": int(objective.get("requested_max_space", 0)),
        "workspace_max_space": int(objective.get("workspace_max_space", 0)),
        "estimated_basis_workspace_bytes": int(
            objective.get("estimated_basis_workspace_bytes", 0)
        ),
        "workspace_limited": bool(objective.get("workspace_limited", False)),
        "canonical_reduced_basis": bool(
            objective.get("canonical_reduced_basis", False)
        ),
        "direct_complementary_action_executor": bool(
            objective.get("direct_complementary_action_executor", False)
        ),
        "cpp_bond_split": bool(
            objective.get("cpp_active_bond_split", False)
        ),
        "canonical_projection_components": int(
            objective.get("canonical_projection_components", 0)
        ),
        "canonical_projection_max_component_dimension": int(
            objective.get(
                "canonical_projection_max_component_dimension",
                0,
            )
        ),
        "canonical_projection_transform_elements": int(
            objective.get("canonical_projection_transform_elements", 0)
        ),
        "cpp_factor_route_calls": int(
            stats.get("su2_qchem_cpp_factor_route_calls", 0)
        ),
        "cpp_factor_diagonal_calls": int(
            stats.get("su2_qchem_cpp_factor_diagonal_calls", 0)
        ),
        "raw_factor_routes": int(cpp_stats.get("factor_route_count", 0)),
        "raw_route_groups": int(cpp_stats.get("raw_route_group_count", 0)),
        "fused_raw_route_groups": int(
            cpp_stats.get("fused_raw_route_group_count", 0)
        ),
        "fused_raw_routes": int(cpp_stats.get("fused_raw_route_count", 0)),
        "dense_pair_kernels": int(cpp_stats.get("dense_pair_kernel_count", 0)),
        "dense_pair_routes": int(cpp_stats.get("dense_pair_route_count", 0)),
        "raw_execution_groups": int(
            cpp_stats.get("raw_execution_group_count", 0)
        ),
        "raw_execution_actions": int(
            cpp_stats.get("raw_execution_action_count", 0)
        ),
        "raw_factor_gemm_calls": int(cpp_stats.get("raw_factor_gemm_calls", 0)),
        "raw_factor_cache_hits": int(
            cpp_stats.get("raw_factor_cache_hits", 0)
        ),
        "raw_factor_cache_misses": int(
            cpp_stats.get("raw_factor_cache_misses", 0)
        ),
        "raw_factor_build_seconds": float(
            cpp_stats.get("raw_factor_build_seconds", 0.0)
        ),
        "factor_route_matvec_calls": int(
            cpp_stats.get("factor_route_matvec_calls", 0)
        ),
        "factor_route_projected_matvec_calls": int(
            cpp_stats.get("factor_route_projected_matvec_calls", 0)
        ),
        "solver_timing": {
            str(key): float(value)
            for key, value in (objective.get("solver_timing") or {}).items()
        },
        "left_factor_elements": int(left_pool.get("stored_elements", 0)),
        "right_factor_elements": int(right_pool.get("stored_elements", 0)),
    }


def _fresh_worker(connection, function, args, kwargs):
    try:
        connection.send(("ok", function(*args, **kwargs)))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _run_fresh(function, *args, **kwargs):
    """Run one solver in a fresh fork so peak RSS is not cumulative."""

    if "fork" not in mp.get_all_start_methods():
        return function(*args, **kwargs)
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_fresh_worker,
        args=(child, function, args, kwargs),
    )
    process.start()
    child.close()
    status, payload = parent.recv()
    process.join()
    if status != "ok":
        raise RuntimeError(payload)
    if process.exitcode != 0:
        raise RuntimeError(f"benchmark worker exited with status {process.exitcode}")
    return payload


def build_cpp_reference(case, basis):
    """Build one shared mean-field reference with compiled C++ AO integrals."""

    started = time.perf_counter()
    molecule = Molecule(atom=case["atom"], unit="bohr", basis=basis)
    molecule.build(
        driver="builtin",
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    build_info = molecule._builtin_build_info
    if (
        build_info.get("eri_backend") != "cpp"
        or not str(build_info.get("dense_builder", "")).startswith("cpp-")
    ):
        raise RuntimeError("Large-CAS SU(2) benchmarks require _integrals_cpp.")
    mean_field = RHF(molecule).run()
    return mean_field, time.perf_counter() - started, dict(build_info)


def build_active_tensors(mean_field, case):
    """Build active tensors once and return the PyQED owner used to obtain them."""

    dmrg = DMRG(
        mean_field,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=1,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
        integral_backend="cholesky",
    )
    dmrg.build()
    h1e = np.asarray(dmrg.h1e)
    g2e = np.asarray(dmrg.h2e)
    return {
        "ncas": int(dmrg.ncas),
        "n_elec": int(dmrg.nelecas),
        "spin": int(dmrg.spin),
        "ecore": float(dmrg.e_core),
        "h1e": np.ascontiguousarray(h1e[0]),
        "g2e": np.ascontiguousarray(g2e[0, 0]),
        "orb_sym": [0] * int(dmrg.ncas),
    }


def run_pyqed(
    mean_field,
    case,
    *,
    bond_dim,
    nsweeps,
    max_bond_mode="reduced",
    conv_tol=-1.0,
    require_convergence=False,
    initial_bond_multiplicity=None,
    davidson_tol=1.0e-3,
    nstates=1,
):
    """Run one fresh PyQED solver while reusing the shared C++ integrals."""

    if initial_bond_multiplicity is None:
        initial_bond_multiplicity = max(
            2,
            int(bond_dim) // (2 * int(case["ncas"])),
        )
    initial_bond_multiplicity = int(initial_bond_multiplicity)
    rss_before = _rss_bytes()
    total_started = time.perf_counter()
    dmrg = DMRG(
        mean_field,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=bond_dim,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
        integral_backend="cholesky",
    )
    build_started = time.perf_counter()
    dmrg.build()
    system_seconds = time.perf_counter() - build_started
    peak_rss_after_system = _rss_bytes()
    current_rss_after_system = _current_rss_bytes()
    run_started = time.perf_counter()
    dmrg.run(
        # The public count is complete LR+RL sweeps; each engine adapter
        # translates it to two directional passes.
        nsweeps=int(nsweeps),
        nstates=int(nstates),
        weights=(
            None
            if int(nstates) == 1
            else np.ones(int(nstates), dtype=float) / int(nstates)
        ),
        conv_tol=float(conv_tol),
        require_convergence=bool(require_convergence),
        local_basis_policy="block2_like",
        su2_kernel_backend="cpp",
        # The persistent C++ metric-Krylov route avoids rebuilding dense
        # orthonormal local matrices at every moving bond.
        orthonormalized_operator_dim=0,
        # block2's SU(2) bond dimension is one global reduced-state budget,
        # not a separate allowance for every symmetry sector.
        max_bond_mode=str(max_bond_mode),
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
        bond_multiplicity=initial_bond_multiplicity,
        davidson_max_iter=100,
        davidson_tol=float(davidson_tol),
        verify_returned_mps_energy=True,
        profile=True,
    )
    run_seconds = time.perf_counter() - run_started
    history = list(dmrg.dmrg.history or ())
    sweep_seconds = sum(
        float((entry.get("timing") or {}).get("total", 0.0))
        for entry in history
    )
    moving = (
        (history[-1].get("moving_environment_stats") or {})
        if history
        else {}
    )
    sweep_owner = moving.get("su2_moving_environment") or {}
    stack_stats = (
        (history[-1].get("renormalized_block_stack_stats") or {})
        if history
        else {}
    )
    norm_stack_stats = (
        (history[-1].get("norm_renormalized_block_stack_stats") or {})
        if history
        else {}
    )
    owned_half_sweeps = int(
        sweep_owner.get("owned_half_sweep_calls", 0)
    )
    callback_half_sweeps = int(
        sweep_owner.get("half_sweep_executor_calls", 0)
    )
    returned_energy = float(
        history[-1].get(
            "returned_mps_energy",
            _scalar(dmrg.e_tot),
        )
    )
    includes_core_energy = bool(
        (dmrg._active_integral_build_info or {}).get(
            "includes_core_energy",
            False,
        )
    )
    returned_total_energy = (
        returned_energy
        if includes_core_energy
        else returned_energy + float(dmrg.e_core)
    )
    returned_active_energy = (
        returned_energy - float(dmrg.e_core)
        if includes_core_energy
        else returned_energy
    )
    objectives = [
        objective
        for sweep in history
        for objective in (sweep.get("bond_objectives") or ())
    ]
    bond_updates = int(
        sum(len(entry.get("updates") or ()) for entry in history)
    )
    ground_state = dmrg.dmrg.ground_state
    canonical_errors = (
        ground_state.canonical_errors()
        if ground_state is not None and ground_state.center is not None
        else (None, None)
    )
    sweep_stage_timing = {}
    for entry in history:
        for name, value in (entry.get("timing") or {}).items():
            sweep_stage_timing[name] = (
                sweep_stage_timing.get(name, 0.0) + float(value)
            )
    native_delta_fields = (
        "contextual_route_plan_builds",
        "complementary_execution_graph_builds",
        "contextual_route_match_seconds",
        "contextual_route_activation_seconds",
        "reduced_contextual_numeric_refresh_seconds",
        "reduced_contextual_execution_refresh_seconds",
        "matvec_seconds",
        "matvec_calls",
        "davidson_iterations",
        "raw_execution_pack_seconds",
        "boundary_update_seconds",
        "truncation_seconds",
    )
    half_sweep_native_deltas = []
    previous_native_stats = {name: 0.0 for name in native_delta_fields}
    for entry in history:
        native_stats = (
            (entry.get("moving_environment_stats") or {}).get(
                "su2_moving_environment"
            )
            or {}
        )
        deltas = {}
        for name in native_delta_fields:
            current = float(native_stats.get(name, 0.0))
            deltas[name] = current - previous_native_stats[name]
            previous_native_stats[name] = current
        half_sweep_native_deltas.append(deltas)
    return {
        "backend": "pyqed-cpp-su2",
        "energy": _scalar(dmrg.e_tot),
        "state_energies": [
            float(value) for value in np.asarray(dmrg.e_tot).reshape(-1)
        ],
        "returned_mps_active_energy": returned_active_energy,
        "returned_mps_total_energy": returned_total_energy,
        "reported_expectation_error": abs(
            _scalar(dmrg.e_tot)
            - returned_total_energy
        ),
        "canonical_center": (
            None if ground_state is None else ground_state.center
        ),
        "left_canonical_error": canonical_errors[0],
        "right_canonical_error": canonical_errors[1],
        "system_seconds": float(system_seconds),
        "state_initialization_seconds": float(
            history[-1].get("state_initialization_seconds", 0.0)
        ),
        "run_seconds": float(run_seconds),
        "sweep_seconds": float(sweep_seconds),
        "half_sweep_seconds": [
            float((entry.get("timing") or {}).get("total", 0.0))
            for entry in history
        ],
        "half_sweep_native_deltas": half_sweep_native_deltas,
        "total_seconds": float(time.perf_counter() - total_started),
        "bond_updates": bond_updates,
        "complete_sweeps": int(nsweeps),
        "half_sweeps": 2 * int(nsweeps),
        "converged": bool(dmrg.dmrg.converged),
        "completed_sweeps": int(dmrg.dmrg.ncompleted),
        "completed_half_sweeps": int(dmrg.dmrg.ncompleted_half_sweeps),
        "max_bond_mode": str(max_bond_mode),
        "initial_bond_multiplicity": initial_bond_multiplicity,
        "davidson_tol": float(davidson_tol),
        "bond_updates_per_second": (
            float(bond_updates / sweep_seconds) if sweep_seconds > 0.0 else 0.0
        ),
        "peak_rss_bytes": int(_rss_bytes()),
        "current_rss_bytes": int(_current_rss_bytes()),
        "peak_rss_after_system_bytes": int(peak_rss_after_system),
        "current_rss_after_system_bytes": int(current_rss_after_system),
        "peak_rss_delta_bytes": max(0, int(_rss_bytes() - rss_before)),
        "cpp_solver_memory_bytes": int(sweep_owner.get("memory_bytes", 0)),
        "moving_environment": sweep_owner,
        "renormalized_block_stack_stats": stack_stats,
        "norm_renormalized_block_stack_stats": norm_stack_stats,
        "execution": {
            "cpp_half_sweep": (
                owned_half_sweeps == 2 * int(nsweeps)
            ),
            "cpp_half_sweep_boundary": (
                owned_half_sweeps + callback_half_sweeps
                == 2 * int(nsweeps)
            ),
            "python_owned_bond_updates": int(
                sweep_owner.get(
                    "half_sweep_python_bond_callbacks",
                    bond_updates,
                )
            ),
            "cpp_site_merges": int(
                sweep_owner.get("site_merge_calls", 0)
            ),
            "python_owned_site_merges": max(
                0,
                int(bond_updates)
                - int(sweep_owner.get("site_merge_calls", 0)),
            ),
            "cpp_site_merge_seconds": float(
                sweep_owner.get("site_merge_seconds", 0.0)
            ),
            "cpp_complementary_action_prepares": int(
                sweep_owner.get(
                    "active_bond_complementary_prepares",
                    0,
                )
            ),
            "cpp_complementary_action_fallbacks": int(
                sweep_owner.get(
                    "active_bond_complementary_fallbacks",
                    0,
                )
            ),
            "cpp_complementary_davidson_calls": int(
                sweep_owner.get(
                    "active_bond_complementary_davidson_calls",
                    0,
                )
            ),
            "cpp_complementary_generalized_davidson_calls": int(
                sweep_owner.get(
                    "active_bond_complementary_generalized_davidson_calls",
                    0,
                )
            ),
            "cpp_active_bond_metric_prepares": int(
                sweep_owner.get("active_bond_metric_prepares", 0)
            ),
            "cpp_canonical_projection_davidson_calls": int(
                sweep_owner.get(
                    "canonical_projection_davidson_calls",
                    0,
                )
            ),
            "cpp_canonical_projection_builds": int(
                sweep_owner.get("canonical_projection_builds", 0)
            ),
            "cpp_canonical_projection_reuses": int(
                sweep_owner.get("canonical_projection_reuses", 0)
            ),
            "cpp_canonical_projection_cache_entries": int(
                sweep_owner.get(
                    "canonical_projection_cache_entries",
                    0,
                )
            ),
            "cpp_canonical_projection_cache_transform_elements": int(
                sweep_owner.get(
                    "canonical_projection_cache_transform_elements",
                    0,
                )
            ),
            "cpp_canonical_projection_cache_evictions": int(
                sweep_owner.get(
                    "canonical_projection_cache_evictions",
                    0,
                )
            ),
            "cpp_bond_splits": int(
                sweep_owner.get("active_bond_cpp_splits", 0)
            ),
            "python_owned_bond_splits": max(
                0,
                int(bond_updates)
                - int(sweep_owner.get("active_bond_cpp_splits", 0)),
            ),
            "python_environment_advances": int(
                moving.get("hamiltonian_boundary_advances", 0)
            ),
            "cpp_cached_environment_replays": int(
                sweep_owner.get("cached_boundary_replays", 0)
            ),
            "cpp_complementary_execution_slab_bytes": int(
                sweep_owner.get("complementary_execution_slab_bytes", 0)
            ),
            "cpp_complementary_execution_slab_capacity_bytes": int(
                sweep_owner.get(
                    "complementary_execution_slab_capacity_bytes",
                    0,
                )
            ),
            "cpp_complementary_execution_slab_budget_bytes": int(
                sweep_owner.get(
                    "complementary_execution_slab_budget_bytes",
                    0,
                )
            ),
            "cpp_peak_complementary_execution_slab_required_bytes": int(
                sweep_owner.get(
                    "peak_complementary_execution_slab_required_bytes",
                    0,
                )
            ),
            "cpp_complementary_execution_slab_full_prepares": int(
                sweep_owner.get(
                    "complementary_execution_slab_full_prepares",
                    0,
                )
            ),
            "cpp_complementary_execution_slab_partial_prepares": int(
                sweep_owner.get(
                    "complementary_execution_slab_partial_prepares",
                    0,
                )
            ),
            "cpp_complementary_execution_slab_matvec_repacks": int(
                sweep_owner.get(
                    "complementary_execution_slab_matvec_repacks",
                    0,
                )
            ),
            "cpp_direct_complementary_action_calls": int(
                sweep_owner.get(
                    "direct_complementary_action_calls",
                    0,
                )
            ),
            "cpp_direct_complementary_actions": int(
                sweep_owner.get("direct_complementary_actions", 0)
            ),
            "cpp_direct_source_factor_loads": int(
                sweep_owner.get("direct_source_factor_loads", 0)
            ),
            "dense_projected_growth": False,
        },
        "solver_timing": _sum_nested_timings(history, "solver_timing"),
        "sweep_stage_timing": sweep_stage_timing,
        "operator_build_timing": _sum_nested_timings(
            history,
            "renormalized_operator_build_timing",
        ),
        "last_operator_table_stats": (
            {}
            if not objectives
            else _compact_operator_stats(
                objectives[-1].get("renormalized_operator_table_stats") or {}
            )
        ),
        "bond_memory_trace": [
            _compact_bond_memory(objective) for objective in objectives
        ],
        "cpp_davidson_bonds": int(
            sum(
                bool(objective.get("cpp_davidson"))
                or int(
                    (
                        (objective.get("renormalized_operator_table_stats") or {})
                        .get("cpp_block_table_stats")
                        or {}
                    ).get("davidson_calls", 0)
                )
                > 0
                for objective in objectives
            )
        ),
        "cpp_boundary_sync_failures": int(
            (
                (history[-1].get("renormalized_block_stack_stats") or {})
                if history
                else {}
            ).get("cpp_boundary_sync_failures", 0)
        ),
    }


def run_block2(
    active,
    *,
    bond_dim,
    nsweeps,
    seed,
    davidson_tol=1.0e-3,
    nstates=1,
):
    """Run one fresh block2 solver on the same active tensors."""

    half_sweeps = 2 * int(nsweeps)
    rss_before = _rss_bytes()
    total_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyqed-block2-large-cas-") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.bw.b.Random.rand_seed(int(seed))
        setup_started = time.perf_counter()
        driver.initialize_system(
            n_sites=active["ncas"],
            n_elec=active["n_elec"],
            spin=active["spin"],
            orb_sym=active["orb_sym"],
        )
        mpo = driver.get_qc_mpo(
            active["h1e"],
            active["g2e"],
            ecore=active["ecore"],
            iprint=0,
        )
        system_seconds = time.perf_counter() - setup_started
        state_started = time.perf_counter()
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=bond_dim,
            nroots=int(nstates),
        )
        state_initialization_seconds = time.perf_counter() - state_started
        sweep_started = time.perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=half_sweeps,
            bond_dims=[bond_dim] * half_sweeps,
            noises=[0.0] * half_sweeps,
            # block2 specifies the squared residual, while PyQED specifies
            # the residual norm itself.
            thrds=[float(davidson_tol) ** 2] * half_sweeps,
            tol=1.0e-9,
            iprint=0,
            dav_max_iter=100,
            dav_def_max_size=50,
        )
        sweep_seconds = time.perf_counter() - sweep_started
        expectation_started = time.perf_counter()
        if int(nstates) == 1:
            exported_energies = [
                float(
                    driver.expectation(ket, mpo, ket, iprint=0)
                    / driver.expectation(
                        ket,
                        driver.get_identity_mpo(),
                        ket,
                        iprint=0,
                    )
                )
            ]
        else:
            roots = [
                driver.split_mps(ket, root, f"ROOT{root}")
                for root in range(int(nstates))
            ]
            identity = driver.get_identity_mpo()
            exported_energies = [
                float(
                    driver.expectation(root, mpo, root, iprint=0)
                    / driver.expectation(root, identity, root, iprint=0)
                )
                for root in roots
            ]
        expectation_seconds = time.perf_counter() - expectation_started
        exported_energy = exported_energies[0]
        solver_energies = [
            float(value) for value in np.asarray(energy).reshape(-1)
        ]
    return {
        "backend": "block2-su2",
        # block2's DMRG return is the optimized two-site energy before the
        # final SVD truncation.  Compare solvers using the expectation of the
        # actual exported MPS, which is also what PyQED reports.
        "energy": exported_energy,
        "state_energies": exported_energies,
        "solver_energy": solver_energies[0],
        "solver_state_energies": solver_energies,
        "returned_mps_total_energy": exported_energy,
        "reported_expectation_error": abs(
            solver_energies[0] - exported_energy
        ),
        "expectation_seconds": float(expectation_seconds),
        "system_seconds": float(system_seconds),
        "state_initialization_seconds": float(state_initialization_seconds),
        "run_seconds": float(sweep_seconds),
        "sweep_seconds": float(sweep_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
        "bond_updates": int(max(0, active["ncas"] - 1) * half_sweeps),
        "bond_updates_per_second": float(
            max(0, active["ncas"] - 1) * half_sweeps
            / max(sweep_seconds, 1.0e-15)
        ),
        "davidson_tol": float(davidson_tol),
        "peak_rss_bytes": int(_rss_bytes()),
        "peak_rss_delta_bytes": max(0, int(_rss_bytes() - rss_before)),
    }


def _median(rows, key):
    return float(statistics.median(float(row[key]) for row in rows))


def _scaling_exponent(results, key):
    if len(results) < 2:
        return None
    sizes = np.log(np.asarray([row["ncas"] for row in results], dtype=float))
    values = np.log(
        np.maximum(
            np.asarray([row["median"][key] for row in results], dtype=float),
            1.0e-15,
        )
    )
    return float(np.polyfit(sizes, values, 1)[0])


def run_case(
    name,
    *,
    basis,
    bond_dim,
    nsweeps,
    repeats,
    seed,
    max_bond_mode="reduced",
    energy_validation_sweeps=0,
    nstates=1,
):
    """Run one CAS size and summarize fresh-solver medians."""

    case = PRESETS[name]
    mean_field, integral_rhf_seconds, integral_info = build_cpp_reference(case, basis)
    active = build_active_tensors(mean_field, case)

    # Warm extension imports and one-time C++ route setup outside the sample.
    _ = _run_fresh(
        run_pyqed,
        mean_field,
        case,
        bond_dim=min(int(bond_dim), 16),
        nsweeps=1,
        max_bond_mode=max_bond_mode,
        nstates=nstates,
    )
    pyqed_rows = [
        _run_fresh(
            run_pyqed,
            mean_field,
            case,
            bond_dim=bond_dim,
            nsweeps=nsweeps,
            max_bond_mode=max_bond_mode,
            nstates=nstates,
        )
        for _ in range(repeats)
    ]
    block2_rows = [
        _run_fresh(
            run_block2,
            active,
            bond_dim=bond_dim,
            nsweeps=nsweeps,
            seed=seed + repeat,
            nstates=nstates,
        )
        for repeat in range(repeats)
    ]
    py_sweep = _median(pyqed_rows, "sweep_seconds")
    b2_sweep = _median(block2_rows, "sweep_seconds")
    py_system = _median(pyqed_rows, "system_seconds")
    b2_system = _median(block2_rows, "system_seconds")
    py_rss = int(statistics.median(row["peak_rss_bytes"] for row in pyqed_rows))
    b2_rss = int(statistics.median(row["peak_rss_bytes"] for row in block2_rows))
    py_solver_memory = int(
        statistics.median(
            row["cpp_solver_memory_bytes"] for row in pyqed_rows
        )
    )
    energy_error = abs(
        _median(pyqed_rows, "energy") - _median(block2_rows, "energy")
    )
    energy_validation = None
    if int(energy_validation_sweeps) > 0:
        validation_sweeps = max(
            int(nsweeps),
            int(energy_validation_sweeps),
        )
        pyqed_validation = _run_fresh(
            run_pyqed,
            mean_field,
            case,
            bond_dim=bond_dim,
            nsweeps=validation_sweeps,
            max_bond_mode=max_bond_mode,
            conv_tol=-1.0,
            require_convergence=False,
            # A compact symmetry-complete seed is more reliable for
            # convergence than the wider timing seed; keep this run outside
            # the performance samples.
            initial_bond_multiplicity=2,
            davidson_tol=1.0e-8,
            nstates=nstates,
        )
        block2_validation = _run_fresh(
            run_block2,
            active,
            bond_dim=bond_dim,
            nsweeps=validation_sweeps,
            seed=seed,
            davidson_tol=1.0e-8,
            nstates=nstates,
        )
        energy_validation = {
            "max_sweeps": validation_sweeps,
            "initial_bond_multiplicity": 2,
            "davidson_tol": 1.0e-8,
            "energy_tolerance": 1.0e-6,
            "pyqed_energy": float(pyqed_validation["energy"]),
            "block2_energy": float(block2_validation["energy"]),
            "energy_error": abs(
                float(pyqed_validation["energy"])
                - float(block2_validation["energy"])
            ),
            "pyqed_reported_expectation_error": float(
                pyqed_validation["reported_expectation_error"]
            ),
            "block2_reported_expectation_error": float(
                block2_validation["reported_expectation_error"]
            ),
            "pyqed_sweep_seconds": float(
                pyqed_validation["sweep_seconds"]
            ),
            "block2_sweep_seconds": float(
                block2_validation["sweep_seconds"]
            ),
            "forced_sweeps": True,
            "pyqed_converged": bool(pyqed_validation["converged"]),
            "pyqed_completed_half_sweeps": int(
                pyqed_validation["completed_half_sweeps"]
            ),
        }
        energy_validation["passed"] = bool(
            energy_validation["energy_error"]
            <= energy_validation["energy_tolerance"]
            and energy_validation["pyqed_reported_expectation_error"]
            <= 1.0e-10
        )
    return {
        "system": name,
        "ncas": int(case["ncas"]),
        "nelecas": int(case["nelecas"]),
        "bond_dim": int(bond_dim),
        "nstates": int(nstates),
        "max_bond_mode": str(max_bond_mode),
        "initial_bond_multiplicity": int(
            pyqed_rows[0]["initial_bond_multiplicity"]
        ),
        "nsweeps": int(nsweeps),
        "repeats": int(repeats),
        "energy_validation": energy_validation,
        "integral_backend": "cpp",
        "integral_builder": integral_info.get("dense_builder"),
        "integral_rhf_seconds": float(integral_rhf_seconds),
        "pyqed": pyqed_rows,
        "block2": block2_rows,
        "median": {
            "pyqed_system_seconds": py_system,
            "block2_system_seconds": b2_system,
            "system_ratio": py_system / max(b2_system, 1.0e-15),
            "pyqed_state_initialization_seconds": _median(
                pyqed_rows,
                "state_initialization_seconds",
            ),
            "block2_state_initialization_seconds": _median(
                block2_rows,
                "state_initialization_seconds",
            ),
            "pyqed_sweep_seconds": py_sweep,
            "block2_sweep_seconds": b2_sweep,
            "sweep_ratio": py_sweep / max(b2_sweep, 1.0e-15),
            "fixed_sweep_energy_error": float(energy_error),
            "pyqed_peak_rss_bytes": py_rss,
            "pyqed_cpp_solver_memory_bytes": py_solver_memory,
            "block2_peak_rss_bytes": b2_rss,
            "memory_ratio": py_rss / max(b2_rss, 1),
            "pyqed_bond_updates_per_second": _median(
                pyqed_rows,
                "bond_updates_per_second",
            ),
            "block2_bond_updates_per_second": _median(
                block2_rows,
                "bond_updates_per_second",
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", choices=sorted(PRESETS))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--D", type=int)
    parser.add_argument(
        "--max-bond-mode",
        choices=("reduced", "states", "per_sector"),
        default="reduced",
    )
    parser.add_argument("--nsweeps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--energy-validation-sweeps",
        type=int,
        default=0,
        help=(
            "run one separate energy comparison with this many sweeps; "
            "it is excluded from performance medians"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce-first-gate", action="store_true")
    args = parser.parse_args()

    systems = args.system or ["h8", "h10", "h12"]
    results = [
        run_case(
            name,
            basis=args.basis,
            bond_dim=(
                PRESETS[name]["bond_dim"]
                if args.D is None
                else int(args.D)
            ),
            nsweeps=int(args.nsweeps),
            repeats=int(args.repeats),
            seed=int(args.seed),
            max_bond_mode=str(args.max_bond_mode),
            nstates=int(args.nstates),
            energy_validation_sweeps=max(
                int(args.energy_validation_sweeps),
                28 if args.enforce_first_gate else 0,
            ),
        )
        for name in systems
    ]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            median = result["median"]
            print(
                f"{result['system'].upper()} CAS({result['nelecas']},{result['ncas']}) "
                f"D={result['bond_dim']} roots={result['nstates']}: "
                f"sweep={median['pyqed_sweep_seconds']:.3f}s/"
                f"{median['block2_sweep_seconds']:.3f}s "
                f"ratio={median['sweep_ratio']:.2f}x "
                f"build={median['system_ratio']:.2f}x "
                f"memory={median['memory_ratio']:.2f}x "
                "timing-state-dE="
                f"{median['fixed_sweep_energy_error']:.3e}"
            )
            if result["energy_validation"] is not None:
                print(
                    "  tight exported-MPS energy: "
                    f"PyQED={result['energy_validation']['pyqed_energy']:.12f} "
                    f"block2={result['energy_validation']['block2_energy']:.12f} "
                    f"dE={result['energy_validation']['energy_error']:.3e} "
                    f"(max {result['energy_validation']['max_sweeps']} sweeps)"
                )
    if args.enforce_first_gate:
        # This is the fixed-four-sweep performance gate. Energy agreement is
        # validated in a separate longer run; a truncated two-site block2
        # sweep can still change its exported-MPS expectation here.
        failures = []
        for result in results:
            if result["ncas"] < 10:
                continue
            if result["median"]["sweep_ratio"] > 5.0:
                failures.append(
                    f"{result['system']} sweep ratio "
                    f"{result['median']['sweep_ratio']:.2f}x > 5x"
                )
            if result["median"]["memory_ratio"] > 2.0:
                failures.append(
                    f"{result['system']} memory ratio "
                    f"{result['median']['memory_ratio']:.2f}x > 2x"
                )
            if any(
                row["cpp_boundary_sync_failures"] != 0
                for row in result["pyqed"]
            ):
                failures.append(f"{result['system']} C++ boundary sync failed")
            if any(
                row["reported_expectation_error"] > 1.0e-10
                for row in result["pyqed"]
            ):
                failures.append(
                    f"{result['system']} reported energy disagrees with MPS"
                )
            validation = result["energy_validation"]
            if (
                validation is None
                or not validation["passed"]
            ):
                failures.append(
                    f"{result['system']} longer-run exported-MPS energy "
                    "does not agree with block2 within 1e-6"
                )
            if any(
                not row["execution"]["cpp_half_sweep"]
                or row["execution"]["python_owned_bond_updates"] != 0
                or row["execution"]["python_environment_advances"] != 0
                or row["execution"]["dense_projected_growth"]
                for row in result["pyqed"]
            ):
                failures.append(
                    f"{result['system']} did not execute a fully C++ half sweep"
                )
        py_time_exponent = _scaling_exponent(results, "pyqed_sweep_seconds")
        b2_time_exponent = _scaling_exponent(results, "block2_sweep_seconds")
        py_memory_exponent = _scaling_exponent(results, "pyqed_peak_rss_bytes")
        b2_memory_exponent = _scaling_exponent(results, "block2_peak_rss_bytes")
        if (
            py_time_exponent is not None
            and py_time_exponent > b2_time_exponent
        ):
            failures.append(
                "PyQED runtime scaling exponent "
                f"{py_time_exponent:.2f} exceeds block2 {b2_time_exponent:.2f}"
            )
        if (
            py_memory_exponent is not None
            and py_memory_exponent > b2_memory_exponent
        ):
            failures.append(
                "PyQED memory scaling exponent "
                f"{py_memory_exponent:.2f} exceeds block2 {b2_memory_exponent:.2f}"
            )
        if failures:
            raise SystemExit("large-CAS gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
