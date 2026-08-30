#!/usr/bin/env python3
"""Ablate the five fixed-graph upgrades of the LETTA two-site update.

The benchmark deliberately separates accuracy, execution time, and structural
storage.  Every variational row starts from the same saved bond-expansion
checkpoint.  Pair indices are zero based; the default representative pairs
are ``(14, 15)`` and ``(20, 21)``, while the more expensive full update grid is
run only for ``(14, 15)`` by default.

This script does not set BLAS thread variables.  For stable timings invoke it
with the repository's usual single-thread numerical environment, for example

``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ...``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy import linalg

from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.continue_frontier_letta_bond_expansion_6x6 import (
    DEFAULT_SOURCE_RESULT,
    DEFAULT_SOURCE_SNAPSHOT,
    _load_state,
)
from pyqed.letta import FrontierTiedLETTA
from pyqed.letta.core import _lowest_hermitian_eigenpair
import pyqed.letta.frontier_tying as frontier_tying_module


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_two_site_five_stage_6x6.json"
DEFAULT_PAIR_SITES = (14, 20)
DEFAULT_FULL_PAIR_SITES = (14,)
DEFAULT_OUTER_CYCLES = (1, 3)
DEFAULT_CONVERGENCE_CYCLES = 8
DEFAULT_OPERATOR_BACKENDS = ("dense", "block")
DEFAULT_FACTOR_SOLVERS = ("matrix_free", "dense")
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_ints(value):
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_strings(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _timing(samples):
    values = [float(value) for value in samples]
    return {
        "samples_seconds": values,
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "repeats": len(values),
    }


def _measure(function, *, warmups, repeats):
    for _ in range(int(warmups)):
        function()
    samples = []
    result = None
    for _ in range(int(repeats)):
        start = perf_counter()
        result = function()
        samples.append(perf_counter() - start)
    return result, _timing(samples)


def _maximum_absolute_difference(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0))


def _pair_metadata(state, site):
    plan = state._pair_plan(site)
    following = int(site) + 1
    overlap = tuple(
        sorted(
            set(state.sites[site])
            & set(state.sites[following])
        )
    )
    return {
        "sites": [int(site), following],
        "spanned_cut": following,
        "union_sites": list(plan.union_sites),
        "overlap_sites": list(overlap),
        "merged_shape": list(plan.merged_shape),
        "merged_dimension": int(np.prod(plan.merged_shape)),
        "left_factor_shape": list(state.tensors[site].shape),
        "right_factor_shape": list(state.tensors[following].shape),
        "bond_dimension": int(state.bond_dims[following]),
    }


def _shared_pair_environments(state, pair_sites):
    """Build all-cut messages once and extract fixed-source pair holes."""
    for site in pair_sites:
        state._pair_plan(site)
    norm_left = state._norm_frontier.build_left(state.tensors)
    norm_right = state._norm_frontier.build_right(state.tensors)
    hamiltonian_left = state._hamiltonian_frontier.build_left(state.tensors)
    hamiltonian_right = state._hamiltonian_frontier.build_right(state.tensors)
    return {
        int(site): state._pair_environment_from_outer_messages(
            site,
            norm_left[site],
            norm_right[site + 2],
            hamiltonian_left[site],
            hamiltonian_right[site + 2],
        )
        for site in pair_sites
    }


def _metric_support(metric, metric_tol):
    values, vectors = linalg.eigh(metric, check_finite=False)
    scale = max(
        float(np.linalg.norm(metric, ord=np.inf)),
        float(np.max(np.abs(values), initial=0.0)),
        np.finfo(float).tiny,
    )
    numerical_floor = 64.0 * np.finfo(float).eps * scale
    requested_floor = max(float(metric_tol) * scale, numerical_floor)
    numerical_active = values > numerical_floor
    requested_active = values > requested_floor
    return values, vectors, requested_active, numerical_active, {
        "requested_rank": int(np.count_nonzero(requested_active)),
        "numerical_rank": int(np.count_nonzero(numerical_active)),
        "metric_scale": scale,
        "minimum_numerically_positive": float(
            np.min(values[numerical_active])
        ),
        "numerical_condition": float(
            np.max(values[numerical_active]) / np.min(values[numerical_active])
        ),
    }


def _cutoff_whitened_pair_solve(state, metric, effective, metric_tol, eig_tol):
    basis, whitened, frame = state._whiten_local_operators(
        metric,
        effective,
        metric_tol=metric_tol,
    )
    coordinate_energy, reduced = _lowest_hermitian_eigenpair(
        whitened,
        tol=eig_tol,
    )
    vector = basis @ reduced
    vector /= np.sqrt(float(np.real(np.vdot(vector, metric @ vector))))
    energy = state._pair_rayleigh(vector, metric, effective)
    coordinate_residual = float(
        np.linalg.norm(whitened @ reduced - coordinate_energy * reduced)
    )
    return energy, vector, frame, coordinate_residual


def _merged_solver_ablation(
    state,
    site,
    environment,
    *,
    metric_tol,
    eig_tol,
    maxiter,
    max_subspace,
    dense_fallback_dim,
    warmups,
    repeats,
):
    metric, effective = state.pair_local_operators(site, environment=environment)
    plan = state._pair_plan(site)
    warm = state._merge_pair_factors(
        site,
        plan.union_sites,
        state.tensors[site],
        state.tensors[site + 1],
    ).reshape(-1)
    warm_energy = state._pair_rayleigh(warm, metric, effective)
    values, vectors, requested_active, numerical_active, support = (
        _metric_support(metric, metric_tol)
    )

    cutoff_result, cutoff_timing = _measure(
        lambda: _cutoff_whitened_pair_solve(
            state,
            metric,
            effective,
            metric_tol,
            eig_tol,
        ),
        warmups=warmups,
        repeats=repeats,
    )
    cutoff_energy, cutoff_vector, frame, coordinate_residual = cutoff_result
    cutoff_check = state._pair_residual_verification(
        metric,
        effective,
        cutoff_energy,
        cutoff_vector,
        values,
        vectors,
        requested_active,
        numerical_active,
    )

    verified = {}
    for metric_support in ("regularized", "numerical"):
        verified_result, verified_timing = _measure(
            lambda support=metric_support: state._solve_verified_pair_pencil(
                site,
                metric,
                effective,
                warm,
                metric_tol=metric_tol,
                eig_tol=eig_tol,
                maxiter=maxiter,
                max_subspace=max_subspace,
                dense_fallback_dim=dense_fallback_dim,
                metric_support=support,
            ),
            warmups=warmups,
            repeats=repeats,
        )
        verified_energy, _vector, local_update, diagnostics = verified_result
        verified[metric_support] = {
            "energy": float(verified_energy),
            "energy_gain": float(warm_energy - verified_energy),
            "diagnostics": diagnostics,
            "local_update": local_update,
            "timing": verified_timing,
        }

    dense_result, dense_timing = _measure(
        lambda: linalg.eigh(
            effective,
            metric,
            subset_by_index=[0, 0],
            driver="gvx",
            check_finite=False,
        ),
        warmups=warmups,
        repeats=repeats,
    )
    dense_values, dense_vectors = dense_result
    dense_vector = dense_vectors[:, 0]
    dense_energy = state._pair_rayleigh(dense_vector, metric, effective)
    dense_check = state._pair_residual_verification(
        metric,
        effective,
        dense_energy,
        dense_vector,
        values,
        vectors,
        numerical_active,
        numerical_active,
    )
    return {
        "operator_build_excluded_from_solver_timings": True,
        "warm_energy": float(warm_energy),
        "metric_support": support,
        "cutoff_whitened": {
            "energy": float(cutoff_energy),
            "energy_gain": float(warm_energy - cutoff_energy),
            "retained_rank": int(frame["metric_rank"]),
            "identity_metric_error": float(frame["identity_metric_error"]),
            "coordinate_residual_norm": coordinate_residual,
            "full_residual": cutoff_check,
            "lowest_root_certified": False,
            "timing": cutoff_timing,
        },
        "verified": verified,
        "dense_full_numerical_support_kernel": {
            "coordinate_eigenvalue": float(np.real(dense_values[0])),
            "rayleigh_energy": float(dense_energy),
            "full_residual": dense_check,
            "timing": dense_timing,
            "scope": "lowest generalized eigensolver kernel only",
        },
    }


@contextmanager
def _solver_instrumentation():
    """Record factor Davidson and projected-dense solves without changing them."""
    original_davidson = frontier_tying_module.lowest_generalized_davidson
    original_dense = frontier_tying_module._lowest_generalized_eigenpair
    original_design = FrontierTiedLETTA._pair_factor_design
    records = {"davidson": [], "projected_dense": [], "design": []}

    def davidson(*args, **kwargs):
        size = int(np.asarray(args[2]).size)
        try:
            result = original_davidson(*args, **kwargs)
        except Exception as error:
            records["davidson"].append(
                {"size": size, "returned": False, "error": str(error)}
            )
            raise
        diagnostics = result[2]
        records["davidson"].append(
            {
                "size": size,
                "returned": True,
                "converged": bool(diagnostics.converged),
                "iterations": int(diagnostics.iterations),
                "hamiltonian_matvecs": int(diagnostics.hamiltonian_matvecs),
                "metric_matvecs": int(diagnostics.metric_matvecs),
                "residual_norm": float(diagnostics.residual_norm),
            }
        )
        return result

    def projected_dense(hamiltonian, metric, *args, **kwargs):
        size = int(np.asarray(metric).shape[0])
        try:
            result = original_dense(hamiltonian, metric, *args, **kwargs)
        except Exception as error:
            records["projected_dense"].append(
                {"size": size, "returned": False, "error": str(error)}
            )
            raise
        records["projected_dense"].append({"size": size, "returned": True})
        return result

    def factor_design(state, *args, **kwargs):
        design = original_design(state, *args, **kwargs)
        records["design"].append(
            {
                "shape": list(design.shape),
                "elements": int(design.size),
            }
        )
        return design

    frontier_tying_module.lowest_generalized_davidson = davidson
    frontier_tying_module._lowest_generalized_eigenpair = projected_dense
    FrontierTiedLETTA._pair_factor_design = factor_design
    try:
        yield records
    finally:
        frontier_tying_module.lowest_generalized_davidson = original_davidson
        frontier_tying_module._lowest_generalized_eigenpair = original_dense
        FrontierTiedLETTA._pair_factor_design = original_design


def _update_record(update, source_energy, fresh_energy, instrumentation):
    merged_size = int(update.raw_merged_dim)
    factor_davidson = [
        row for row in instrumentation["davidson"] if int(row["size"]) < merged_size
    ]
    factor_dense = [
        row
        for row in instrumentation["projected_dense"]
        if int(row["size"]) < merged_size
    ]
    dense_fallbacks = sum(
        1
        for row in factor_davidson
        if not row.get("returned", False) or not row.get("converged", False)
    )
    designs = instrumentation["design"]
    return {
        "sites": list(update.sites),
        "accepted": bool(update.accepted),
        "source_energy": float(source_energy),
        "energy": float(update.energy),
        "fresh_energy": float(fresh_energy),
        "fresh_energy_error": float(fresh_energy - update.energy),
        "energy_gain": float(source_energy - fresh_energy),
        "merged_energy": float(update.merged_energy),
        "merged_energy_history": list(update.merged_energy_history),
        "factor_energy_history": list(update.factor_energy_history),
        "requested_outer_cycles": None,
        "completed_outer_cycles": int(update.outer_cycles),
        "factor_sweeps": int(update.factor_sweeps),
        "factor_accepted_half_updates": int(update.factor_accepted_updates),
        "selected_retraction_start": update.selected_start,
        "metric_projection_error": float(update.metric_projection_error),
        "relative_euclidean_truncation_error": float(
            update.relative_truncation_error
        ),
        "pair_operator_backend": update.pair_operator_backend,
        "pair_operator_stored_elements": int(update.pair_operator_stored_elements),
        "merged_solve": update.merged_solve,
        "certified_merged_root_reused": bool(
            update.merged_solve is not None
            and str(update.merged_solve.method).startswith("cached_")
        ),
        "factor_solver_instrumentation": {
            "davidson_calls": factor_davidson,
            "projected_dense_calls": factor_dense,
            "observed_dense_fallbacks": int(dense_fallbacks),
            "design_calls": designs,
            "full_design_matrices_constructed": bool(designs),
            "total_design_elements": int(
                sum(row["elements"] for row in designs)
            ),
            "largest_design_elements": int(
                max((row["elements"] for row in designs), default=0)
            ),
            "note": (
                "Matrix-free factor solves and their projected-dense fallback "
                "do not construct J. The explicit dense baseline may construct "
                "J when dense pair operators are already resident."
            ),
        },
    }


def _full_update_variant(
    source_result,
    source_state,
    site,
    environment,
    *,
    backend,
    factor_solver,
    outer_cycles,
    metric_tol,
    eig_tol,
    maxiter,
    max_subspace,
    dense_fallback_dim,
    metric_support,
    split_metric_sweeps,
    split_variational_sweeps,
    repeats,
):
    rows = []
    plan_samples = []
    update_samples = []
    total_samples = []
    for _ in range(int(repeats)):
        candidate, _payload = _load_state(source_result, source_state)
        source_energy = float(candidate.expectation())
        total_start = perf_counter()
        plan_start = perf_counter()
        candidate._pair_plan(site)
        plan_samples.append(perf_counter() - plan_start)
        update_start = perf_counter()
        with _solver_instrumentation() as instrumentation:
            update = candidate.optimize_two_sites(
                site,
                solver="verified",
                environment=environment,
                verify_global=True,
                pair_operator_backend=backend,
                merged_dense_fallback_dim=dense_fallback_dim,
                metric_support=metric_support,
                outer_cycles=outer_cycles,
                factor_solver=factor_solver,
                split_metric_sweeps=split_metric_sweeps,
                split_variational_sweeps=split_variational_sweeps,
                split_random_starts=0,
                split_random_seed=0,
                metric_tol=metric_tol,
                eig_tol=eig_tol,
                maxiter=maxiter,
                max_subspace=max_subspace,
            )
        update_samples.append(perf_counter() - update_start)
        fresh_energy = float(candidate.expectation())
        total_samples.append(perf_counter() - total_start)
        row = _update_record(update, source_energy, fresh_energy, instrumentation)
        row["requested_outer_cycles"] = int(outer_cycles)
        row["requested_operator_backend"] = str(backend)
        row["requested_factor_solver"] = str(factor_solver)
        rows.append(row)
    energies = np.asarray([row["fresh_energy"] for row in rows])
    representative = rows[-1]
    representative["repeat_energy_spread"] = float(
        np.max(energies) - np.min(energies)
    )
    representative["timing"] = {
        "pair_plan_build_or_lookup": _timing(plan_samples),
        "update_with_prebuilt_environment": _timing(update_samples),
        "plan_plus_update_and_fresh_check": _timing(total_samples),
        "fixed_source_pair_environment_reused": True,
        "state_load_and_construction_excluded": True,
    }
    return representative


def _build_temporary_pair_reference(state, site):
    """Reproduce the old rebuild-and-replan merged-pair representation."""
    merged, union_sites = state._merged_pair_tensor(site)
    following = int(site) + 1
    right_dimension = state._bond_dims()[following + 1]
    right_sites = state.sites[following]
    identity = np.eye(right_dimension, dtype=merged.dtype)
    identity_tensor = np.broadcast_to(
        identity.reshape(
            right_dimension,
            right_dimension,
            *((1,) * len(right_sites)),
        ),
        (
            right_dimension,
            right_dimension,
            *(state.dims[index] for index in right_sites),
        ),
    ).copy()
    tensors = [tensor.copy() for tensor in state.tensors]
    tensors[site] = merged
    tensors[following] = identity_tensor
    parents = list(state.parent_sets)
    parents[site] = tuple(index for index in union_sites if index != site)
    bonds = list(state._bond_dims())
    bonds[following] = right_dimension
    temporary = FrontierTiedLETTA(
        state.hamiltonian,
        state.dims,
        tuple(parents),
        bond_dims=tuple(bonds),
        tensors=tensors,
        frontier_backend=state.frontier_backend,
        path_optimizer=state.path_optimizer,
    )
    temporary.tensors = tensors
    return temporary


def _environment_cache_benchmark(state, pair_sites, *, repeats):
    per_pair = {}
    for site in pair_sites:
        # The cached path is meant to represent repeated sweeps.  Compile its
        # value-independent pair plan and contraction expressions once outside
        # the timed samples; the temporary reference is rebuilt every sample.
        state._pair_plan(site)
        prewarm_environment = state.pair_environment(site)
        state.pair_local_operators(site, environment=prewarm_environment)
        rebuild_rows = []
        cached_rows = []
        reference_operators = None
        cached_operators = None
        for _ in range(int(repeats)):
            total_start = perf_counter()
            setup_start = perf_counter()
            temporary = _build_temporary_pair_reference(state, site)
            setup_seconds = perf_counter() - setup_start
            environment_start = perf_counter()
            temporary_environment = temporary.site_environment(site)
            temporary_environment_seconds = perf_counter() - environment_start
            operator_start = perf_counter()
            reference_operators = temporary.local_operators(
                site,
                environment=temporary_environment,
            )
            operator_seconds = perf_counter() - operator_start
            rebuild_rows.append(
                {
                    "temporary_state_construction_seconds": setup_seconds,
                    "environment_seconds": temporary_environment_seconds,
                    "operator_seconds": operator_seconds,
                    "total_seconds": perf_counter() - total_start,
                }
            )

            plan_start = perf_counter()
            state._pair_plan(site)
            plan_seconds = perf_counter() - plan_start
            total_start = perf_counter()
            environment_start = perf_counter()
            environment = state.pair_environment(site)
            cached_environment_seconds = perf_counter() - environment_start
            operator_start = perf_counter()
            cached_operators = state.pair_local_operators(
                site,
                environment=environment,
            )
            cached_operator_seconds = perf_counter() - operator_start
            cached_rows.append(
                {
                    "cached_plan_lookup_seconds": plan_seconds,
                    "environment_seconds": cached_environment_seconds,
                    "operator_seconds": cached_operator_seconds,
                    "total_seconds": perf_counter() - total_start,
                }
            )
        metric_error = _maximum_absolute_difference(
            reference_operators[0], cached_operators[0]
        )
        effective_error = _maximum_absolute_difference(
            reference_operators[1], cached_operators[1]
        )
        per_pair[str(site)] = {
            "sites": [int(site), int(site) + 1],
            "cached_path_prewarmed_once": True,
            "temporary_path_rebuilt_each_sample": True,
            "temporary_rebuild_samples": rebuild_rows,
            "cached_pair_samples": cached_rows,
            "temporary_rebuild_total": _timing(
                [row["total_seconds"] for row in rebuild_rows]
            ),
            "cached_pair_total": _timing(
                [row["total_seconds"] for row in cached_rows]
            ),
            "operator_only_rebuild": _timing(
                [row["operator_seconds"] for row in rebuild_rows]
            ),
            "operator_only_cached": _timing(
                [row["operator_seconds"] for row in cached_rows]
            ),
            "maximum_metric_difference": metric_error,
            "maximum_hamiltonian_difference": effective_error,
            "cached_over_rebuild_total_ratio": float(
                np.median([row["total_seconds"] for row in cached_rows])
                / np.median([row["total_seconds"] for row in rebuild_rows])
            ),
            "cached_over_rebuild_operator_only_ratio": float(
                np.median([row["operator_seconds"] for row in cached_rows])
                / np.median([row["operator_seconds"] for row in rebuild_rows])
            ),
        }

    shared_samples = []
    shared_operators = None
    for _ in range(int(repeats)):
        total_start = perf_counter()
        message_start = perf_counter()
        norm_left = state._norm_frontier.build_left(state.tensors)
        norm_right = state._norm_frontier.build_right(state.tensors)
        hamiltonian_left = state._hamiltonian_frontier.build_left(state.tensors)
        hamiltonian_right = state._hamiltonian_frontier.build_right(state.tensors)
        message_seconds = perf_counter() - message_start
        extraction_start = perf_counter()
        environments = {
            int(site): state._pair_environment_from_outer_messages(
                site,
                norm_left[site],
                norm_right[site + 2],
                hamiltonian_left[site],
                hamiltonian_right[site + 2],
            )
            for site in pair_sites
        }
        extraction_seconds = perf_counter() - extraction_start
        operator_start = perf_counter()
        shared_operators = {
            int(site): state.pair_local_operators(
                site,
                environment=environments[int(site)],
            )
            for site in pair_sites
        }
        operator_seconds = perf_counter() - operator_start
        shared_samples.append(
            {
                "all_cut_message_seconds": message_seconds,
                "pair_environment_extraction_seconds": extraction_seconds,
                "operator_seconds": operator_seconds,
                "total_seconds": perf_counter() - total_start,
            }
        )
    shared_errors = {}
    for site in pair_sites:
        # The last cached one-pair operators and the shared-message operators
        # are evaluated on the same unmodified source state.
        environment = state.pair_environment(site)
        reference = state.pair_local_operators(site, environment=environment)
        shared_errors[str(site)] = {
            "maximum_metric_difference": _maximum_absolute_difference(
                reference[0], shared_operators[int(site)][0]
            ),
            "maximum_hamiltonian_difference": _maximum_absolute_difference(
                reference[1], shared_operators[int(site)][1]
            ),
        }
    return {
        "scope": (
            "two fixed-source pair environments and pencils; this is not an "
            "end-to-end sweep timing"
        ),
        "per_pair_rebuild_vs_cached": per_pair,
        "shared_all_cut_messages": {
            "samples": shared_samples,
            "total": _timing([row["total_seconds"] for row in shared_samples]),
            "equivalence": shared_errors,
        },
    }


def _block_microbenchmark(
    state,
    site,
    environment,
    *,
    warmups,
    repeats,
):
    dense, dense_timing = _measure(
        lambda: state.pair_local_operators(site, environment=environment),
        warmups=warmups,
        repeats=repeats,
    )
    problem, block_timing = _measure(
        lambda: state.pair_local_block_problem(site, environment=environment),
        warmups=warmups,
        repeats=repeats,
    )
    densify_start = perf_counter()
    block_metric = problem.metric.to_dense()
    block_effective = problem.hamiltonian.to_dense()
    densify_seconds = perf_counter() - densify_start
    overlap = tuple(
        sorted(
            set(state.sites[site])
            & set(state.sites[site + 1])
        )
    )
    overlap_axes = tuple(environment.union_sites.index(label) for label in overlap)
    changes_overlap = any(
        any(
            problem.layout.configurations[row][axis]
            != problem.layout.configurations[column][axis]
            for axis in overlap_axes
        )
        for row, column in problem.hamiltonian.connected_pairs
    )
    dense_elements = int(dense[0].size + dense[1].size)
    return {
        "sites": [int(site), int(site) + 1],
        "dense": {
            "stored_elements": dense_elements,
            "build_timing": dense_timing,
        },
        "conditional_blocks": {
            "stored_elements": int(problem.stored_elements),
            "dense_equivalent_elements": int(problem.dense_elements),
            "storage_fraction": float(problem.storage_fraction),
            "metric_blocks": len(problem.metric.blocks),
            "hamiltonian_blocks": len(problem.hamiltonian.blocks),
            "hamiltonian_component_sizes": [
                len(component) for component in problem.hamiltonian_components
            ],
            "build_timing": block_timing,
            "block_over_dense_build_time_ratio": float(
                block_timing["median_seconds"] / dense_timing["median_seconds"]
            ),
            "validation_only_densify_seconds": float(densify_seconds),
            "maximum_metric_difference": _maximum_absolute_difference(
                dense[0], block_metric
            ),
            "maximum_hamiltonian_difference": _maximum_absolute_difference(
                dense[1], block_effective
            ),
            "overlap_sectors_independent": not changes_overlap,
            "hamiltonian_changes_shared_label": bool(changes_overlap),
        },
        "interpretation": (
            "The block update uses block actions directly.  Densification above "
            "is performed only to validate equivalence in this benchmark; any "
            "dense fallback in a full update is reported by merged_solve."
        ),
    }


def run_benchmark(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_state=DEFAULT_SOURCE_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    pair_sites=DEFAULT_PAIR_SITES,
    full_pair_sites=DEFAULT_FULL_PAIR_SITES,
    outer_cycles=DEFAULT_OUTER_CYCLES,
    operator_backends=DEFAULT_OPERATOR_BACKENDS,
    factor_solvers=DEFAULT_FACTOR_SOLVERS,
    convergence_cycles=DEFAULT_CONVERGENCE_CYCLES,
    repeats=1,
    warmups=0,
    metric_tol=1.0e-12,
    eig_tol=1.0e-10,
    maxiter=1600,
    max_subspace=96,
    dense_fallback_dim=2048,
    metric_support="regularized",
    split_metric_sweeps=6,
    split_variational_sweeps=8,
):
    total_start = perf_counter()
    pair_sites = tuple(dict.fromkeys(int(site) for site in pair_sites))
    full_pair_sites = tuple(dict.fromkeys(int(site) for site in full_pair_sites))
    outer_cycles = tuple(dict.fromkeys(int(value) for value in outer_cycles))
    operator_backends = tuple(dict.fromkeys(str(value) for value in operator_backends))
    factor_solvers = tuple(dict.fromkeys(str(value) for value in factor_solvers))
    convergence_cycles = int(convergence_cycles)
    repeats = int(repeats)
    warmups = int(warmups)
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative.")
    if not pair_sites or any(site < 0 or site + 1 >= 36 for site in pair_sites):
        raise ValueError("pair_sites must contain valid left sites.")
    if any(site not in pair_sites for site in full_pair_sites):
        raise ValueError("full_pair_sites must be a subset of pair_sites.")
    if any(value < 1 for value in outer_cycles):
        raise ValueError("outer_cycles values must be positive.")
    if any(value not in {"dense", "block"} for value in operator_backends):
        raise ValueError("operator_backends must contain only dense or block.")
    if any(value not in {"matrix_free", "dense"} for value in factor_solvers):
        raise ValueError("factor_solvers must contain only matrix_free or dense.")
    if convergence_cycles < 1:
        raise ValueError("convergence_cycles must be positive.")
    metric_support = str(metric_support).lower().replace("-", "_")
    if metric_support not in {"regularized", "numerical"}:
        raise ValueError("metric_support must be regularized or numerical.")

    source, source_payload = _load_state(source_result, source_state)
    source_energy = float(source.expectation())
    environments = _shared_pair_environments(source, pair_sites)
    pair_metadata = {
        str(site): _pair_metadata(source, site) for site in pair_sites
    }

    merged_solver = {}
    for site in pair_sites:
        print(f"merged-solver ablation pair {(site, site + 1)}", flush=True)
        merged_solver[str(site)] = _merged_solver_ablation(
            source,
            site,
            environments[site],
            metric_tol=metric_tol,
            eig_tol=eig_tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
            dense_fallback_dim=dense_fallback_dim,
            warmups=warmups,
            repeats=repeats,
        )

    full_updates = {}
    for site in full_pair_sites:
        rows = {}
        for backend in operator_backends:
            for factor_solver in factor_solvers:
                for cycles in outer_cycles:
                    name = f"{backend}__{factor_solver}__outer_{cycles}"
                    print(f"full update pair {(site, site + 1)}: {name}", flush=True)
                    rows[name] = _full_update_variant(
                        source_result,
                        source_state,
                        site,
                        environments[site],
                        backend=backend,
                        factor_solver=factor_solver,
                        outer_cycles=cycles,
                        metric_tol=metric_tol,
                        eig_tol=eig_tol,
                        maxiter=maxiter,
                        max_subspace=max_subspace,
                        dense_fallback_dim=dense_fallback_dim,
                        metric_support=metric_support,
                        split_metric_sweeps=split_metric_sweeps,
                        split_variational_sweeps=split_variational_sweeps,
                        repeats=repeats,
                    )
        if "dense" in operator_backends and "dense" in factor_solvers:
            name = f"dense__dense__outer_{convergence_cycles}_until_stalled"
            print(
                f"full update pair {(site, site + 1)}: {name}",
                flush=True,
            )
            rows[name] = _full_update_variant(
                source_result,
                source_state,
                site,
                environments[site],
                backend="dense",
                factor_solver="dense",
                outer_cycles=convergence_cycles,
                metric_tol=metric_tol,
                eig_tol=eig_tol,
                maxiter=maxiter,
                max_subspace=max_subspace,
                dense_fallback_dim=dense_fallback_dim,
                metric_support=metric_support,
                split_metric_sweeps=split_metric_sweeps,
                split_variational_sweeps=split_variational_sweeps,
                repeats=repeats,
            )
            rows[name]["stopping_probe"] = True
        for backend in operator_backends:
            for factor_solver in factor_solvers:
                baseline_name = f"{backend}__{factor_solver}__outer_1"
                if baseline_name not in rows:
                    continue
                baseline_energy = float(rows[baseline_name]["fresh_energy"])
                for cycles in outer_cycles:
                    name = f"{backend}__{factor_solver}__outer_{cycles}"
                    rows[name]["incremental_gain_vs_outer_1"] = float(
                        baseline_energy - float(rows[name]["fresh_energy"])
                    )
        convergence_name = (
            f"dense__dense__outer_{convergence_cycles}_until_stalled"
        )
        dense_baseline = "dense__dense__outer_1"
        if convergence_name in rows and dense_baseline in rows:
            rows[convergence_name]["incremental_gain_vs_outer_1"] = float(
                float(rows[dense_baseline]["fresh_energy"])
                - float(rows[convergence_name]["fresh_energy"])
            )
        full_updates[str(site)] = rows

    print("environment-cache microbenchmark", flush=True)
    cache_benchmark = _environment_cache_benchmark(
        source,
        pair_sites,
        repeats=repeats,
    )
    print("conditional-block microbenchmark", flush=True)
    block_benchmark = {
        str(site): _block_microbenchmark(
            source,
            site,
            environments[site],
            warmups=warmups,
            repeats=repeats,
        )
        for site in pair_sites
    }

    frontier_path = Path(frontier_tying_module.__file__).resolve()
    payload = {
        "status": "complete",
        "protocol": {
            "benchmark": "five fixed-graph LETTA two-site upgrades",
            "site_indexing": "zero based",
            "representative_left_sites": list(pair_sites),
            "full_update_left_sites": list(full_pair_sites),
            "outer_cycle_variants": list(outer_cycles),
            "outer_convergence_probe": convergence_cycles,
            "operator_backends": list(operator_backends),
            "factor_solvers": list(factor_solvers),
            "timing_warmups": warmups,
            "timing_repeats": repeats,
            "metric_tol": float(metric_tol),
            "metric_support": metric_support,
            "eig_tol": float(eig_tol),
            "maxiter": int(maxiter),
            "max_subspace": int(max_subspace),
            "merged_dense_fallback_dim": int(dense_fallback_dim),
            "split_metric_sweeps": int(split_metric_sweeps),
            "split_variational_sweeps": int(split_variational_sweeps),
            "random_starts": 0,
            "symmetry": "none",
            "timing_policy": (
                "state construction is excluded from solver/update timing; raw "
                "samples are retained and ratios must be formed from matching rows"
            ),
        },
        "software_and_host": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_VARIABLES
            },
        },
        "source": {
            "result": str(Path(source_result).resolve()),
            "snapshot": str(Path(source_state).resolve()),
            "result_sha256": _file_sha256(source_result),
            "snapshot_sha256": _file_sha256(source_state),
            "frontier_tying_source": str(frontier_path),
            "frontier_tying_sha256": _file_sha256(frontier_path),
            "model": source_payload["model"],
            "energy": source_energy,
            "parameters": int(source.nparameters),
            "bond_dims": list(source.bond_dims),
        },
        "pair_metadata": pair_metadata,
        "merged_solver_ablation": merged_solver,
        "full_update_grid": full_updates,
        "environment_cache_microbenchmark": cache_benchmark,
        "conditional_block_microbenchmark": block_benchmark,
        "interpretation_guardrails": [
            "A residual-verifed eigenpair is not automatically the lowest root; use lowest_root_certified.",
            "Environment-cache timings cover the stated pair preparation scope, not a full sweep.",
            "Matrix-free factor rows may use projected-dense fallback; instrumentation reports it.",
            "The matrix-free factor path avoids J; the explicit dense baseline may construct J and reports it.",
            "Conditional-block storage is structural; check dense_fallback before claiming end-to-end memory avoidance.",
            "Historical timings from other scripts or runs are not used in speed ratios.",
        ],
        "timing_seconds": {"total": float(perf_counter() - total_start)},
    }
    payload = _jsonable(payload)
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pair-sites",
        type=_parse_ints,
        default=DEFAULT_PAIR_SITES,
        help="comma-separated representative left sites (default: 14,20)",
    )
    parser.add_argument(
        "--full-pair-sites",
        type=_parse_ints,
        default=DEFAULT_FULL_PAIR_SITES,
        help="subset receiving the full update grid (default: 14)",
    )
    parser.add_argument(
        "--outer-cycles",
        type=_parse_ints,
        default=DEFAULT_OUTER_CYCLES,
        help="comma-separated outer-cycle counts (default: 1,3)",
    )
    parser.add_argument(
        "--operator-backends",
        type=_parse_strings,
        default=DEFAULT_OPERATOR_BACKENDS,
        help="comma-separated dense/block backends",
    )
    parser.add_argument(
        "--factor-solvers",
        type=_parse_strings,
        default=DEFAULT_FACTOR_SOLVERS,
        help="comma-separated matrix_free/dense factor solvers",
    )
    parser.add_argument(
        "--convergence-cycles",
        type=int,
        default=DEFAULT_CONVERGENCE_CYCLES,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--max-subspace", type=int, default=96)
    parser.add_argument("--dense-fallback-dim", type=int, default=2048)
    parser.add_argument(
        "--metric-support",
        choices=("regularized", "numerical"),
        default="regularized",
    )
    parser.add_argument("--split-metric-sweeps", type=int, default=6)
    parser.add_argument("--split-variational-sweeps", type=int, default=8)
    args = parser.parse_args()
    payload = run_benchmark(
        source_result=args.source_result,
        source_state=args.source_state,
        output=args.output,
        pair_sites=args.pair_sites,
        full_pair_sites=args.full_pair_sites,
        outer_cycles=args.outer_cycles,
        operator_backends=args.operator_backends,
        factor_solvers=args.factor_solvers,
        convergence_cycles=args.convergence_cycles,
        repeats=args.repeats,
        warmups=args.warmups,
        metric_tol=args.metric_tol,
        eig_tol=args.eig_tol,
        maxiter=args.maxiter,
        max_subspace=args.max_subspace,
        dense_fallback_dim=args.dense_fallback_dim,
        metric_support=args.metric_support,
        split_metric_sweeps=args.split_metric_sweeps,
        split_variational_sweeps=args.split_variational_sweeps,
    )
    summary = {
        "output": str(Path(args.output).resolve()),
        "source_energy": payload["source"]["energy"],
        "full_update_rows": {
            site: {
                name: {
                    "energy": row["fresh_energy"],
                    "gain": row["energy_gain"],
                    "seconds": row["timing"][
                        "update_with_prebuilt_environment"
                    ]["median_seconds"],
                }
                for name, row in rows.items()
            }
            for site, rows in payload["full_update_grid"].items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
