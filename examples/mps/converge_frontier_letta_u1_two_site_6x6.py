#!/usr/bin/env python3
"""Converge 6x6 U(1) graph-LETTA with block-sector two-site sweeps."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.benchmark_frontier_letta_u1_4x4 import (
    _neel_cores,
    _u1_mps_run,
)
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    SymmetryLayout,
    abelian_frontier_tied_letta_from_mps,
)
from pyqed.letta.frontier_abelian import _normalized_charge_assignment
from pyqed.mps import MPO


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_u1_two_site_6x6.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_u1_two_site_6x6.npz"
CHECKPOINT_ENERGY_TOL = 5.0e-8
TIE_GRAPHS = ("all-j1", "nonvirtual-j1")
FRONTIER_BACKENDS = ("identity-block", "renormalized")


def _charge_record(charges):
    return [[list(charge) for charge in labels] for labels in charges]


def _layout_record(layout):
    return {
        "local_qns": _charge_record(layout.local_qns),
        "bond_qns": _charge_record(layout.bond_qns),
        "target": list(layout.target),
    }


def _layout_from_record(record):
    return SymmetryLayout(
        local_qns=tuple(
            tuple(tuple(int(value) for value in charge) for charge in labels)
            for labels in record["local_qns"]
        ),
        bond_qns=tuple(
            tuple(tuple(int(value) for value in charge) for charge in labels)
            for labels in record["bond_qns"]
        ),
        target=tuple(int(value) for value in record["target"]),
    )


def _protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _state_protocol(protocol):
    keys = (
        "model",
        "nrows",
        "ncols",
        "j1",
        "j2",
        "target",
        "tie_graph_mode",
        "bond_dim",
    )
    record = {key: protocol.get(key) for key in keys}
    record["parameter_target"] = protocol.get("parameter_target", "fixed_layout")
    record["charge_assignment"] = protocol.get("charge_assignment", "occurrence")
    if record["parameter_target"] != "fixed_layout":
        record["parameter_maximum_bond_dim"] = protocol.get(
            "parameter_maximum_bond_dim"
        )
        record["parameter_expansion_scale"] = protocol.get(
            "parameter_expansion_scale"
        )
    return record


def _parameter_diagnostics(state):
    active = int(state.nparameters)
    dense = int(state.dense_nparameters)
    return {
        "parameters": active,
        "dense_equivalent_parameters": dense,
        "symmetry_parameter_coverage": float(active / dense) if dense else 1.0,
        "symmetry_removed_parameters": int(dense - active),
    }


def _saturated_bond_dims(nsites, maximum):
    """Open-boundary ranks capped by the Hilbert space on either side."""
    return tuple(
        min(int(maximum), 2 ** min(cut, int(nsites) - cut))
        for cut in range(int(nsites) + 1)
    )


def _expansion_record_payload(record):
    if is_dataclass(record):
        return asdict(record)
    return dict(record)


def _expansion_summary(records, selected_cuts):
    rows = [_expansion_record_payload(record) for record in records]
    return {
        "updates": rows,
        "selected_cuts": [int(cut) for cut in selected_cuts],
        "maximum_relative_norm_error": max(
            (float(row["norm_error"]) for row in rows),
            default=0.0,
        ),
        "maximum_absolute_energy_change": max(
            (
                abs(float(row["energy"] - row["energy_before"]))
                for row in rows
            ),
            default=0.0,
        ),
        "seeded_directions": int(
            sum(int(row.get("seeded_directions", 0)) for row in rows)
        ),
    }


def _expand_to_parameter_target(state, target, *, maximum, scale, seed):
    """Greedily add U1 sector channels until active parameters best match target."""
    target = int(target)
    maximum = int(maximum)
    if target <= int(state.nparameters):
        return (), ()
    if maximum < max(state.bond_dims):
        raise ValueError("parameter maximum bond dimension is below current bonds.")
    caps = _saturated_bond_dims(len(state.dims), maximum)
    records = []
    selected_cuts = []
    rng = np.random.default_rng(seed)
    while True:
        current = int(state.nparameters)
        current_error = abs(target - current)
        candidates = []
        for cut in range(1, len(state.dims)):
            if state.bond_dims[cut] >= caps[cut]:
                continue
            labels = state._automatic_expansion_labels(cut, 1)
            layout = state.abelian_layout.with_expanded_bond(
                cut,
                labels,
                charge_increment_qns=getattr(state, "site_increment_qns", None),
            )
            count = int(
                sum(
                    np.count_nonzero(mask)
                    for mask in layout.local_masks(
                        state.physical_sites,
                        charge_rules=getattr(state, "charge_rules", None),
                    )
                )
            )
            error = abs(target - count)
            center_distance = abs(cut - len(state.dims) / 2)
            candidates.append((error, center_distance, cut))
        if not candidates:
            break
        error, _center_distance, cut = min(candidates)
        if error >= current_error:
            break
        records.append(
            state.expand_bond(
                cut,
                state.bond_dims[cut] + 1,
                direction="right",
                strategy="residual",
                scale=scale,
                seed=int(rng.integers(np.iinfo(np.int64).max)),
            )
        )
        selected_cuts.append(cut)
    return tuple(records), tuple(selected_cuts)


def _temporary_bond_dimension(value, bond_dim):
    if value is None:
        return None, "fixed"
    if isinstance(value, str):
        normalized = value.lower().replace("-", "_").replace(" ", "")
        if normalized in {"fixed", "current", "none", "old"}:
            return None, "fixed"
        if normalized in {"square", "squared", "d2", "d^2", "dmrg"}:
            return int(bond_dim) * int(bond_dim), "square"
        try:
            resolved = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "two_site_temporary_bond_dim must be 'fixed', 'square', or an integer."
            ) from exc
        return resolved, str(resolved)
    resolved = int(value)
    return resolved, str(resolved)


def _selected_tie_edges(nearest, tie_graph):
    tie_graph = str(tie_graph).lower().replace("_", "-")
    if tie_graph == "all-j1":
        return tuple(nearest)
    if tie_graph == "nonvirtual-j1":
        return tuple(
            (left, right)
            for left, right in nearest
            if abs(int(right) - int(left)) > 1
        )
    raise ValueError(f"tie_graph must be one of {TIE_GRAPHS}.")


def _model(nrows, ncols, j2, tie_graph="all-j1"):
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, float(j2)) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(int(nrows) * int(ncols), weighted)
    tie_edges = _selected_tie_edges(nearest, tie_graph)
    return (
        hamiltonian,
        parent_sets_from_edges(len(hamiltonian.dims), tie_edges),
        nearest,
        tie_edges,
    )


def _save_snapshot(path, state, *, cycle, low_gain_streak, protocol_fingerprint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    arrays = {
        f"tensor_{site:03d}": np.asarray(tensor)
        for site, tensor in enumerate(state.tensors)
    }
    arrays.update(
        {
            f"pair_recycle_{site:03d}": np.asarray(vectors)
            for site, vectors in getattr(
                state,
                "_pair_matrix_free_recycle_cache",
                {},
            ).items()
        }
    )
    profiles = getattr(state, "_pair_backend_profile_cache", {})
    profile_sites = np.asarray(sorted(profiles), dtype=np.int64)
    arrays["pair_profile_sites"] = profile_sites
    for name in (
        "hamiltonian_vector_products",
        "hamiltonian_action_calls",
        "hamiltonian_batch_calls",
    ):
        arrays[f"pair_profile_{name}"] = np.asarray(
            [int(profiles[site][name]) for site in profile_sites],
            dtype=np.int64,
        )
    np.savez_compressed(
        temporary,
        recorded_energy=np.asarray(float(state.expectation())),
        completed_cycles=np.asarray(int(cycle), dtype=np.int64),
        low_gain_streak=np.asarray(int(low_gain_streak), dtype=np.int64),
        protocol_fingerprint=np.asarray(str(protocol_fingerprint)),
        **arrays,
    )
    temporary.replace(path)


def _state_from_snapshot(
    path,
    *,
    model,
    layout,
    frontier_backend="identity_block",
    charge_assignment="physical",
):
    hamiltonian, parents, _nearest, _tie_edges = _model(
        model["nrows"],
        model["ncols"],
        model["j2"],
        model.get("tie_graph_mode", "all-j1"),
    )
    with np.load(path, allow_pickle=False) as archive:
        tensors = [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(len(hamiltonian.dims))
        ]
        recycle = {
            int(name.removeprefix("pair_recycle_")): np.array(
                archive[name],
                copy=True,
            )
            for name in archive.files
            if name.startswith("pair_recycle_")
        }
        profile_sites = (
            np.asarray(archive["pair_profile_sites"], dtype=np.int64)
            if "pair_profile_sites" in archive.files
            else np.empty(0, dtype=np.int64)
        )
        profiles = {
            int(site): {
                name: int(archive[f"pair_profile_{name}"][offset])
                for name in (
                    "hamiltonian_vector_products",
                    "hamiltonian_action_calls",
                    "hamiltonian_batch_calls",
                )
            }
            for offset, site in enumerate(profile_sites)
        }
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        charge_assignment=charge_assignment,
        tensors=tensors,
        frontier_backend=frontier_backend,
    )
    state.tensors = [tensor.copy() for tensor in tensors]
    state._pair_matrix_free_recycle_cache = recycle
    state._pair_backend_profile_cache = profiles
    state.energy = state.expectation()
    return state


def _snapshot_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            "energy": float(archive["recorded_energy"]),
            "completed_cycles": int(archive["completed_cycles"]),
            "low_gain_streak": int(archive["low_gain_streak"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
        }


def _pair_diagnostics(updates):
    merged = [update.merged_solve for update in updates]
    backend_counts = {
        backend: int(
            sum(update.pair_operator_backend == backend for update in updates)
        )
        for backend in sorted(
            {update.pair_operator_backend for update in updates}
        )
    }
    action_backend_counts = {
        backend: int(
            sum(update.pair_action_backend == backend for update in updates)
        )
        for backend in sorted(
            {
                update.pair_action_backend
                for update in updates
                if update.pair_action_backend
            }
        )
    }
    return {
        "pair_updates": len(updates),
        "accepted_pair_updates": int(sum(update.accepted for update in updates)),
        "verified_merged_roots": int(
            sum(bool(item is not None and item.verified) for item in merged)
        ),
        "lowest_root_certified": int(
            sum(bool(item is not None and item.lowest_root_certified) for item in merged)
        ),
        "dense_fallbacks": int(
            sum(bool(item is not None and item.dense_fallback) for item in merged)
        ),
        "dense_certifications": int(
            sum(
                bool(
                    item is not None
                    and item.dense_fallback
                    and item.lowest_root_certified
                )
                for item in merged
            )
        ),
        "maximum_metric_dual_relative_residual": float(
            max(
                (
                    item.metric_dual_relative_residual
                    for item in merged
                    if item is not None
                    and np.isfinite(item.metric_dual_relative_residual)
                ),
                default=0.0,
            )
        ),
        "maximum_action_relative_residual": float(
            max(
                (
                    item.action_relative_residual
                    for item in merged
                    if item is not None
                    and np.isfinite(item.action_relative_residual)
                ),
                default=0.0,
            )
        ),
        "verification_kinds": sorted(
            {
                item.verification_kind
                for item in merged
                if item is not None
            }
        ),
        "hamiltonian_action_calls": int(
            sum(
                item.hamiltonian_action_calls
                for item in merged
                if item is not None
            )
        ),
        "hamiltonian_vector_products": int(
            sum(
                item.hamiltonian_vector_products
                for item in merged
                if item is not None
            )
        ),
        "hamiltonian_batch_calls": int(
            sum(
                item.hamiltonian_batch_calls
                for item in merged
                if item is not None
            )
        ),
        "maximum_recycled_vectors": int(
            max(
                (
                    item.recycled_vectors
                    for item in merged
                    if item is not None
                ),
                default=0,
            )
        ),
        "preconditioner_blocks": int(
            sum(
                item.preconditioner_blocks
                for item in merged
                if item is not None
            )
        ),
        "maximum_pair_operator_stored_elements": int(
            max((update.pair_operator_stored_elements for update in updates), default=0)
        ),
        "maximum_operator_peak_elements": int(
            max((update.pair_operator_peak_elements for update in updates), default=0)
        ),
        "maximum_operator_stored_bytes": int(
            max((update.pair_operator_stored_bytes for update in updates), default=0)
        ),
        "maximum_operator_peak_bytes": int(
            max((update.pair_operator_peak_bytes for update in updates), default=0)
        ),
        "maximum_dense_equivalent_elements": int(
            max((2 * update.raw_merged_dim**2 for update in updates), default=0)
        ),
        "backends": sorted({update.pair_operator_backend for update in updates}),
        "backend_counts": backend_counts,
        "action_backend_counts": action_backend_counts,
        "selection_reasons": sorted(
            {update.pair_operator_selection_reason for update in updates}
        ),
        "maximum_dense_estimated_peak_bytes": int(
            max((update.dense_estimated_peak_bytes for update in updates), default=0)
        ),
        "maximum_matrix_free_estimated_peak_bytes": int(
            max(
                (
                    update.matrix_free_estimated_peak_bytes
                    for update in updates
                ),
                default=0,
            )
        ),
        "pair_wall_time_seconds": float(
            sum(update.wall_time_seconds for update in updates)
        ),
        "operator_assembly_seconds": float(
            sum(update.operator_assembly_seconds for update in updates)
        ),
        "merged_solve_seconds": float(
            sum(update.merged_solve_seconds for update in updates)
        ),
        "split_seconds": float(sum(update.split_seconds for update in updates)),
    }


def _certify_pair_stationarity(
    state,
    *,
    sweep_offset,
    stopping_gain_per_site,
    pair_dense_max_bytes,
    dense_estimated_peak_bytes,
    backend="auto",
    eig_tol=1.0e-10,
    maxiter=600,
    max_subspace=48,
):
    """Run one exact LR/RL lowest-root certification cycle."""
    requested_backend = str(backend).lower().replace("-", "_")
    if requested_backend not in {"auto", "dense", "block"}:
        raise ValueError(
            "final certification backend must be 'auto', 'dense', or 'block'."
        )
    dense_fits = int(dense_estimated_peak_bytes) <= int(
        pair_dense_max_bytes
    )
    if requested_backend == "auto":
        selected_backend = "dense" if dense_fits else "block"
    elif requested_backend == "dense" and not dense_fits:
        selected_backend = "block"
    else:
        selected_backend = requested_backend
    raw_dimension = max(
        (
            int(np.prod(state._pair_plan(site).merged_shape))
            for site in range(len(state.dims) - 1)
        ),
        default=1,
    )
    itemsize = np.dtype(
        np.result_type(*state.tensors, state.hamiltonian.dtype)
    ).itemsize
    component_memory_limit = max(
        1,
        int(np.sqrt(max(1, pair_dense_max_bytes) / (2 * itemsize))),
    )
    block_component_limit = min(raw_dimension, component_memory_limit)
    energy_before = float(state.expectation())
    started = perf_counter()
    state.run_two_site(
        nsweeps=2,
        sweep_offset=int(sweep_offset),
        tol=0.0,
        solver="verified",
        pair_operator_backend=selected_backend,
        pair_dense_max_bytes=int(pair_dense_max_bytes),
        outer_cycles=1,
        metric_support="numerical",
        eig_tol=float(eig_tol),
        maxiter=int(maxiter),
        max_subspace=int(max_subspace),
        merged_dense_fallback_dim=(
            raw_dimension if selected_backend == "dense" else 0
        ),
        block_dense_component_max_size=block_component_limit,
        verify_pair_roots=True,
        verify_pair_energies=False,
        verbose=True,
    )
    seconds = float(perf_counter() - started)
    rows = list(state.history)
    updates = tuple(update for row in rows for update in row["updates"])
    diagnostics = _pair_diagnostics(updates)
    energy = float(state.expectation())
    gain = float(energy_before - energy)
    gain_tolerance = 1024.0 * np.finfo(float).eps * max(
        1.0,
        abs(energy_before),
    )
    endpoints_accepted = bool(
        len(rows) == 2 and all(row["accepted"] for row in rows)
    )
    roots_certified = bool(
        updates
        and all(
            update.merged_solve is not None
            and update.merged_solve.verified
            and update.merged_solve.lowest_root_certified
            and update.merged_solve.metric_rank_complete
            for update in updates
        )
    )
    low_gain = bool(
        gain >= -gain_tolerance
        and max(gain, 0.0) / len(state.dims)
        < float(stopping_gain_per_site)
    )
    passed = bool(endpoints_accepted and roots_certified and low_gain)
    if not endpoints_accepted:
        failure_reason = "certification_endpoint_rejected"
    elif not roots_certified:
        failure_reason = "uncertified_memory_limit"
    elif not low_gain:
        failure_reason = "certification_lowered_energy"
    else:
        failure_reason = ""
    return {
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "dense_fit": bool(dense_fits),
        "attempted": True,
        "passed": passed,
        "failure_reason": failure_reason,
        "energy_before": energy_before,
        "energy": energy,
        "energy_gain": gain,
        "energy_gain_per_site": gain / len(state.dims),
        "seconds": seconds,
        "endpoints_accepted": endpoints_accepted,
        "roots_certified": diagnostics["lowest_root_certified"],
        "roots_total": diagnostics["pair_updates"],
        "full_metric_rank_roots": int(
            sum(
                update.merged_solve is not None
                and update.merged_solve.metric_rank_complete
                for update in updates
            )
        ),
        "diagnostics": diagnostics,
    }


def _initial_state(
    *,
    nrows,
    ncols,
    j2,
    tie_graph,
    bond_dim,
    mps_sweeps,
    tolerance,
    tie_noise,
    seed,
    frontier_backend,
    charge_assignment,
):
    hamiltonian, parents, _nearest, _tie_edges = _model(
        nrows, ncols, j2, tie_graph
    )
    dense_mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    product = _neel_cores(len(hamiltonian.dims))
    _solver, dense_state, layout, mps_record = _u1_mps_run(
        dense_mpo,
        product,
        bond_dim=bond_dim,
        sweeps=mps_sweeps,
        tolerance=tolerance,
    )
    charge_assignment = _normalized_charge_assignment(charge_assignment)
    project_incompatible = charge_assignment != "physical"
    if project_incompatible:
        physical_sites = tuple(
            (site,) + parents for site, parents in enumerate(parents)
        )
        layout = SymmetryLayout.from_physical_charges(
            layout.local_qns,
            physical_sites,
            target=layout.target,
            bond_dims=layout.bond_dims,
            charge_assignment=charge_assignment,
        )
    state = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        dense_state.factors,
        abelian_layout=layout,
        tie_noise=tie_noise,
        charge_assignment=charge_assignment,
        project_incompatible=project_incompatible,
        seed=seed,
        frontier_backend=frontier_backend,
    )
    return state, layout, mps_record


def converge(
    *,
    nrows=6,
    ncols=6,
    j2=0.5,
    tie_graph="all-j1",
    bond_dim=4,
    mps_sweeps=12,
    mps_tolerance=1.0e-10,
    pre_one_site_sweeps=2,
    maximum_cycles=8,
    stopping_gain_per_site=1.0e-7,
    required_consecutive_cycles=2,
    tie_noise=1.0e-3,
    seed=7,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    resume=True,
    pair_operator_backend="auto",
    pair_dense_max_bytes=64 * 1024**2,
    merged_dense_fallback_dim=512,
    block_dense_component_max_size=64,
    parallel_block_components=False,
    max_component_workers=None,
    verify_pair_roots=True,
    frontier_backend="identity_block",
    matrix_free_batch_size=2,
    matrix_free_recycle_vectors=6,
    matrix_free_max_action_vectors=256,
    matrix_free_eig_tol=1.0e-9,
    matrix_free_preconditioner="adaptive",
    matrix_free_action_backend="auto",
    matrix_free_prepared_min_action_calls=7,
    final_certification=True,
    final_certification_backend="auto",
    two_site_temporary_bond_dim="square",
    charge_assignment="physical",
    parameter_target=None,
    parameter_maximum_bond_dim=8,
    parameter_expansion_scale=1.0e-3,
):
    output = Path(output)
    snapshot = Path(snapshot)
    tie_graph = str(tie_graph).lower().replace("_", "-")
    selected_tie_description = {
        "all-j1": "all nearest-neighbor J1 bonds",
        "nonvirtual-j1": "J1 bonds not adjacent on the MPS backbone",
    }.get(tie_graph)
    if selected_tie_description is None:
        raise ValueError(f"tie_graph must be one of {TIE_GRAPHS}.")
    frontier_backend = str(frontier_backend).lower().replace("-", "_")
    if frontier_backend in {"narg", "term_recursive", "term_renormalized"}:
        frontier_backend = "renormalized"
    if frontier_backend not in {"identity_block", "renormalized"}:
        raise ValueError(
            "frontier_backend must be 'identity_block' or 'renormalized'."
        )
    charge_assignment = _normalized_charge_assignment(charge_assignment)
    pair_operator_backend = str(pair_operator_backend).lower().replace("-", "_")
    if pair_operator_backend not in {
        "auto",
        "dense",
        "block",
        "matrix_free",
    }:
        raise ValueError(
            "pair_operator_backend must be 'auto', 'dense', 'block', "
            "or 'matrix_free'."
        )
    final_certification_backend = str(
        final_certification_backend
    ).lower().replace("-", "_")
    if final_certification_backend not in {"auto", "dense", "block"}:
        raise ValueError(
            "final_certification_backend must be 'auto', 'dense', or 'block'."
        )
    matrix_free_preconditioner = str(
        matrix_free_preconditioner
    ).lower().replace("-", "_")
    if matrix_free_preconditioner not in {
        "adaptive",
        "none",
        "charge_block_jacobi",
    }:
        raise ValueError(
            "matrix_free_preconditioner must be 'adaptive', 'none', or "
            "'charge_block_jacobi'."
        )
    matrix_free_action_backend = str(
        matrix_free_action_backend
    ).lower().replace("-", "_")
    if matrix_free_action_backend not in {
        "auto",
        "full",
        "fused",
        "prepared",
    }:
        raise ValueError(
            "matrix_free_action_backend must be 'auto', 'full', 'fused', "
            "or 'prepared'."
        )
    matrix_free_prepared_min_action_calls = int(
        matrix_free_prepared_min_action_calls
    )
    if matrix_free_prepared_min_action_calls < 1:
        raise ValueError(
            "matrix_free_prepared_min_action_calls must be positive."
        )
    pair_dense_max_bytes = int(pair_dense_max_bytes)
    if pair_dense_max_bytes < 1:
        raise ValueError("pair_dense_max_bytes must be positive.")
    merged_dense_fallback_dim = int(merged_dense_fallback_dim)
    if merged_dense_fallback_dim < 0:
        raise ValueError("merged_dense_fallback_dim must be nonnegative.")
    (
        resolved_temporary_bond_dim,
        temporary_bond_dim_mode,
    ) = _temporary_bond_dimension(two_site_temporary_bond_dim, bond_dim)
    if (
        resolved_temporary_bond_dim is not None
        and resolved_temporary_bond_dim < int(bond_dim)
    ):
        raise ValueError("two_site_temporary_bond_dim cannot be smaller than bond_dim.")
    parameter_target = None if parameter_target is None else int(parameter_target)
    parameter_maximum_bond_dim = int(parameter_maximum_bond_dim)
    parameter_expansion_scale = float(parameter_expansion_scale)
    if parameter_target is not None and parameter_target < 1:
        raise ValueError("parameter_target must be positive when provided.")
    if parameter_maximum_bond_dim < int(bond_dim):
        raise ValueError("parameter_maximum_bond_dim cannot be smaller than bond_dim.")
    if (
        not np.isfinite(parameter_expansion_scale)
        or parameter_expansion_scale < 0.0
    ):
        raise ValueError("parameter_expansion_scale must be finite and nonnegative.")
    protocol = {
        "model": "open 6x6 J1-J2 Heisenberg",
        "nrows": int(nrows),
        "ncols": int(ncols),
        "j1": 1.0,
        "j2": float(j2),
        "target": "U1 half filling / fixed Sz=0",
        "charge_assignment": charge_assignment,
        "tie_graph": selected_tie_description,
        "tie_graph_mode": tie_graph,
        "bond_dim": int(bond_dim),
        "warm_start": "U1 two-site DMRG",
        "mps_sweeps": int(mps_sweeps),
        "mps_tolerance": float(mps_tolerance),
        "pre_one_site_sweeps": int(pre_one_site_sweeps),
        "pre_one_site_solver": "block_sparse",
        "pre_one_site_frontier_gauge": "sector_probability",
        "letta_optimizer": (
            "two-site U1 sector, recycled batched Hamiltonian actions"
            if pair_operator_backend == "matrix_free"
            else (
                "two-site U1 sector, adaptive dense/matrix-free/block"
                if pair_operator_backend == "auto"
                else "two-site U1 sector, dense-first"
            )
        ),
        "frontier_backend": frontier_backend,
        "pair_hamiltonian_binding": (
            "direct_two_step"
            if frontier_backend == "renormalized"
            else "temporary_identity_absorption"
        ),
        "pair_operator_backend": pair_operator_backend,
        "backend_policy_version": "dense_support_fused_csr_v3",
        "pair_dense_max_bytes": pair_dense_max_bytes,
        "merged_dense_fallback_dim": merged_dense_fallback_dim,
        "matrix_free_batch_size": int(matrix_free_batch_size),
        "matrix_free_recycle_vectors": int(
            matrix_free_recycle_vectors
        ),
        "matrix_free_max_action_vectors": int(
            matrix_free_max_action_vectors
        ),
        "matrix_free_eig_tol": float(matrix_free_eig_tol),
        "matrix_free_preconditioner": matrix_free_preconditioner,
        "matrix_free_action_backend": matrix_free_action_backend,
        "matrix_free_prepared_min_action_calls": (
            matrix_free_prepared_min_action_calls
        ),
        "final_certification": (
            "required_after_matrix_free"
            if final_certification
            else "disabled"
        ),
        "final_certification_backend": final_certification_backend,
        "certification_metric_support": "numerical",
        "split_strategy": "sector_variational",
        "two_site_temporary_bond_dim": (
            "fixed"
            if resolved_temporary_bond_dim is None
            else int(resolved_temporary_bond_dim)
        ),
        "two_site_temporary_bond_dim_mode": temporary_bond_dim_mode,
        "parameter_target": (
            "fixed_layout" if parameter_target is None else int(parameter_target)
        ),
        "parameter_maximum_bond_dim": int(parameter_maximum_bond_dim),
        "parameter_expansion_scale": float(parameter_expansion_scale),
        "outer_cycles": 1,
        "eig_tol": 1.0e-10,
        "maxiter": 600,
        "max_subspace": 48,
        "block_dense_component_max_size": int(block_dense_component_max_size),
        "parallel_block_components": bool(parallel_block_components),
        "max_component_workers": max_component_workers,
        "verify_pair_roots": bool(verify_pair_roots),
        "verify_pair_energies": False,
        "stopping_gain_per_site": float(stopping_gain_per_site),
        "required_consecutive_cycles": int(required_consecutive_cycles),
        "tie_noise": float(tie_noise),
        "seed": int(seed),
    }
    fingerprint = _protocol_fingerprint(protocol)

    if resume and output.is_file() and snapshot.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _snapshot_metadata(snapshot)
        if payload["protocol_fingerprint"] != metadata["protocol_fingerprint"]:
            raise RuntimeError("snapshot and JSON protocol fingerprints disagree")
        cycles = list(payload["cycles"])
        if metadata["completed_cycles"] != len(cycles):
            raise RuntimeError(
                "snapshot cycle count is inconsistent with JSON metadata"
            )
        certifications = list(payload.get("certifications", []))
        previous_protocol = payload.get("protocol", {})
        execution_migrated = payload["protocol_fingerprint"] != fingerprint
        if execution_migrated and _state_protocol(
            previous_protocol
        ) != _state_protocol(protocol):
            raise RuntimeError(
                "checkpoint state definition does not match requested settings"
            )
        layout = _layout_from_record(payload["layout"])
        state = _state_from_snapshot(
            snapshot,
            model=payload["model"],
            layout=layout,
            frontier_backend=frontier_backend,
            charge_assignment=charge_assignment,
        )
        measured = float(state.expectation())
        if abs(measured - metadata["energy"]) > CHECKPOINT_ENERGY_TOL:
            raise RuntimeError("snapshot energy is inconsistent with JSON metadata")
        low_gain_streak = (
            0 if execution_migrated else int(metadata["low_gain_streak"])
        )
        mps_record = payload["warm_start"]["mps"]
        elapsed_before = float(payload["timing_seconds"]["total"])
        if execution_migrated:
            history = list(payload.get("execution_history", []))
            history.append(
                {
                    "protocol": previous_protocol,
                    "protocol_fingerprint": payload["protocol_fingerprint"],
                    "cycles_completed": len(cycles),
                    "low_gain_streak": int(metadata["low_gain_streak"]),
                }
            )
            payload["execution_history"] = history
            payload["protocol"] = protocol
            payload["protocol_fingerprint"] = fingerprint
            payload["status"] = "running"
            _write_json(output, payload)
            _save_snapshot(
                snapshot,
                state,
                cycle=len(cycles),
                low_gain_streak=low_gain_streak,
                protocol_fingerprint=fingerprint,
            )
        elif payload.get("result", {}).get("converged", False):
            return payload
    else:
        setup_start = perf_counter()
        state, layout, mps_record = _initial_state(
            nrows=nrows,
            ncols=ncols,
            j2=j2,
            tie_graph=tie_graph,
            bond_dim=bond_dim,
            mps_sweeps=mps_sweeps,
            tolerance=mps_tolerance,
            tie_noise=tie_noise,
            seed=seed,
            frontier_backend=frontier_backend,
        )
        if parameter_target is None:
            expansion_records = ()
            expansion_cuts = ()
        else:
            expansion_records, expansion_cuts = _expand_to_parameter_target(
                state,
                parameter_target,
                maximum=parameter_maximum_bond_dim,
                scale=parameter_expansion_scale,
                seed=seed + 1_000_003,
            )
            layout = state.abelian_layout
        setup_seconds = perf_counter() - setup_start
        pre_start = perf_counter()
        pre_initial_energy = float(state.expectation())
        state.run(
            nsweeps=int(pre_one_site_sweeps),
            tol=0.0,
            solver="block_sparse",
            frontier_canonicalization=True,
            frontier_gauge_weighting="probability",
            eig_tol=1.0e-10,
            maxiter=600,
            max_subspace=48,
            verbose=True,
        )
        pre_seconds = perf_counter() - pre_start
        pre_history = list(state.history)
        cycles = []
        certifications = []
        low_gain_streak = 0
        elapsed_before = setup_seconds + pre_seconds
        nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
        tie_edges = _selected_tie_edges(nearest, tie_graph)
        payload = {
            "status": "running",
            "model": {
                "nrows": int(nrows),
                "ncols": int(ncols),
                "nsites": int(nrows) * int(ncols),
                "j1": 1.0,
                "j2": float(j2),
                "boundary": "open",
                "tie_graph": selected_tie_description,
                "tie_graph_mode": tie_graph,
                "tie_edges": len(tie_edges),
                "removed_virtual_neighbor_ties": len(nearest) - len(tie_edges),
                "j2_diagonal_edges": len(diagonals),
            },
            "protocol": protocol,
            "protocol_fingerprint": fingerprint,
            "layout": _layout_record(layout),
            "parameter_expansion": {
                "enabled": parameter_target is not None,
                "target": None if parameter_target is None else int(parameter_target),
                "maximum_bond_dim": int(parameter_maximum_bond_dim),
                **_expansion_summary(expansion_records, expansion_cuts),
            },
            "warm_start": {
                "mps": mps_record,
                "letta_lift_energy": pre_initial_energy,
                "letta_lift_energy_per_site": pre_initial_energy / len(state.dims),
            },
            "pre_one_site": {
                "sweeps": int(pre_one_site_sweeps),
                "initial_energy": pre_initial_energy,
                "energy": float(state.expectation()),
                "energy_per_site": float(state.expectation() / len(state.dims)),
                "energy_gain": float(pre_initial_energy - state.expectation()),
                "energy_gain_per_site": float(
                    (pre_initial_energy - state.expectation()) / len(state.dims)
                ),
                "seconds": float(pre_seconds),
                "directional_pass_energies": [
                    float(row["energy"]) for row in pre_history
                ],
                "accepted_updates": int(
                    sum(row.get("accepted_sites", 0) for row in pre_history)
                ),
                "solver_failures": int(
                    sum(row.get("solver_failures", 0) for row in pre_history)
                ),
            },
            "source": {
                **_parameter_diagnostics(state),
                "bond_dims": list(state.bond_dims),
                "peak_frontier_elements": int(state.peak_frontier_elements),
            },
            "cycles": cycles,
            "certifications": certifications,
            "result": {},
            "timing_seconds": {
                "setup": setup_seconds,
                "pre_one_site": pre_seconds,
                "total": elapsed_before,
            },
        }
        _write_json(output, payload)
        _save_snapshot(
            snapshot,
            state,
            cycle=0,
            low_gain_streak=low_gain_streak,
            protocol_fingerprint=fingerprint,
        )

    run_start = perf_counter()
    certification_attempted_this_run = False
    stop_reason = "maximum cycles reached"
    for cycle_index in range(len(cycles), int(maximum_cycles)):
        energy_before = float(state.expectation())
        cycle_start = perf_counter()
        rows = []
        for direction in range(2):
            state.run_two_site(
                nsweeps=1,
                sweep_offset=2 * cycle_index + direction,
                tol=0.0,
                solver="verified",
                pair_operator_backend=pair_operator_backend,
                pair_dense_max_bytes=pair_dense_max_bytes,
                outer_cycles=1,
                eig_tol=1.0e-10,
                maxiter=600,
                max_subspace=48,
                matrix_free_batch_size=int(matrix_free_batch_size),
                matrix_free_recycle_vectors=int(
                    matrix_free_recycle_vectors
                ),
                matrix_free_max_action_vectors=int(
                    matrix_free_max_action_vectors
                ),
                matrix_free_eig_tol=float(matrix_free_eig_tol),
                matrix_free_preconditioner=matrix_free_preconditioner,
                matrix_free_action_backend=matrix_free_action_backend,
                matrix_free_prepared_min_action_calls=(
                    matrix_free_prepared_min_action_calls
                ),
                block_dense_component_max_size=int(block_dense_component_max_size),
                parallel_block_components=bool(parallel_block_components),
                max_component_workers=max_component_workers,
                verify_pair_roots=bool(verify_pair_roots),
                merged_dense_fallback_dim=merged_dense_fallback_dim,
                temporary_bond_dimension=resolved_temporary_bond_dim,
                verify_pair_energies=False,
                verbose=True,
            )
            if len(state.history) != 1:
                raise RuntimeError("directional pass did not produce one history row")
            rows.append(state.history[0])
            direction_energy = float(state.expectation())
            direction_snapshot = snapshot.with_name(
                f"{snapshot.stem}_cycle{cycle_index + 1}_"
                f"{'lr' if direction == 0 else 'rl'}{snapshot.suffix}"
            )
            _save_snapshot(
                direction_snapshot,
                state,
                cycle=len(cycles),
                low_gain_streak=low_gain_streak,
                protocol_fingerprint=fingerprint,
            )
            direction_diagnostics = _pair_diagnostics(rows[-1]["updates"])
            direction_payload = dict(payload)
            direction_payload["status"] = "running"
            direction_payload["directional_checkpoint"] = {
                "cycle": cycle_index + 1,
                "direction": rows[-1]["direction"],
                "energy": direction_energy,
                "energy_per_site": direction_energy / len(state.dims),
                "snapshot": str(direction_snapshot),
                "pair_wall_time_seconds": rows[-1].get(
                    "pair_wall_time_seconds",
                    float("nan"),
                ),
                "slowest_pair_wall_time_seconds": rows[-1].get(
                    "slowest_pair_wall_time_seconds",
                    float("nan"),
                ),
                "operator_assembly_seconds": rows[-1].get(
                    "operator_assembly_seconds",
                    float("nan"),
                ),
                "merged_solve_seconds": rows[-1].get(
                    "merged_solve_seconds",
                    float("nan"),
                ),
                "split_seconds": rows[-1].get("split_seconds", float("nan")),
                "backends": direction_diagnostics["backends"],
                "backend_counts": direction_diagnostics["backend_counts"],
                "maximum_operator_peak_bytes": direction_diagnostics[
                    "maximum_operator_peak_bytes"
                ],
                "maximum_operator_stored_bytes": direction_diagnostics[
                    "maximum_operator_stored_bytes"
                ],
            }
            _write_json(
                output.with_name(
                    f"{output.stem}_cycle{cycle_index + 1}_"
                    f"{'lr' if direction == 0 else 'rl'}{output.suffix}"
                ),
                direction_payload,
            )
        cycle_seconds = perf_counter() - cycle_start
        energy = float(state.expectation())
        gain = float(energy_before - energy)
        updates = tuple(update for row in rows for update in row["updates"])
        diagnostics = _pair_diagnostics(updates)
        allowed_backends = (
            {"block_sector"}
            if pair_operator_backend == "block"
            else (
                {"dense_sector"}
                if pair_operator_backend == "dense"
                else (
                {"matrix_free_sector"}
                    if pair_operator_backend == "matrix_free"
                    else {
                        "dense_sector",
                        "matrix_free_sector",
                        "block_sector",
                    }
                )
            )
        )
        backends_valid = bool(
            diagnostics["backends"]
            and set(diagnostics["backends"]) <= allowed_backends
        )
        update_roots_valid = all(
            update.merged_solve is not None
            and update.merged_solve.verified
            and (
                update.pair_operator_backend != "dense_sector"
                or update.merged_solve.lowest_root_certified
            )
            and (
                pair_operator_backend != "block"
                or not update.merged_solve.dense_fallback
            )
            for update in updates
        )
        verified = bool(
            all(row["accepted"] for row in rows)
            and diagnostics["verified_merged_roots"] == diagnostics["pair_updates"]
            and update_roots_valid
            and backends_valid
        )
        gain_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0, abs(energy_before)
        )
        below_threshold = bool(
            verified
            and gain >= -gain_tolerance
            and max(gain, 0.0) / len(state.dims)
            < float(stopping_gain_per_site)
        )
        low_gain_streak = low_gain_streak + 1 if below_threshold else 0
        cycles.append(
            {
                "cycle": cycle_index + 1,
                "energy_before": energy_before,
                "energy": energy,
                "energy_per_site": energy / len(state.dims),
                "energy_gain": gain,
                "energy_gain_per_site": gain / len(state.dims),
                "seconds": float(cycle_seconds),
                "directional_sweeps": [int(row["sweep"]) for row in rows],
                "endpoint_energies": [float(row["energy"]) for row in rows],
                "endpoints_accepted": bool(all(row["accepted"] for row in rows)),
                "cycle_verified": verified,
                "low_gain_streak": int(low_gain_streak),
                "diagnostics": diagnostics,
            }
        )
        payload["cycles"] = cycles
        provisional_converged = bool(
            low_gain_streak >= int(required_consecutive_cycles)
        )
        matrix_free_used = any(
            "matrix_free_sector"
            in cycle["diagnostics"].get("backends", ())
            for cycle in cycles
        )
        certification_required = bool(
            final_certification and matrix_free_used
        )
        certification = None
        strict_converged = provisional_converged
        if provisional_converged and certification_required:
            certification_attempted_this_run = True
            certification = _certify_pair_stationarity(
                state,
                sweep_offset=2 * (cycle_index + 1),
                stopping_gain_per_site=stopping_gain_per_site,
                pair_dense_max_bytes=pair_dense_max_bytes,
                dense_estimated_peak_bytes=diagnostics[
                    "maximum_dense_estimated_peak_bytes"
                ],
                backend=final_certification_backend,
            )
            certifications.append(certification)
            payload["certifications"] = certifications
            strict_converged = bool(certification["passed"])
            energy = float(certification["energy"])
            if not strict_converged:
                low_gain_streak = 0
                cycles[-1]["low_gain_streak"] = 0
                cycles[-1]["certification_failure_reason"] = certification[
                    "failure_reason"
                ]
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - run_start
        )
        payload["result"] = {
            "converged": strict_converged,
            "stop_reason": stop_reason,
            "cycles_completed": len(cycles),
            "energy": energy,
            "energy_per_site": energy / len(state.dims),
            "energy_gain_from_warm_start": float(
                payload["warm_start"].get(
                    "letta_lift_energy",
                    payload["warm_start"].get("letta_initial_energy", energy),
                )
                - energy
            ),
            "energy_gain_from_two_site_start": float(
                payload.get("pre_one_site", {}).get(
                    "energy",
                    payload["warm_start"].get("letta_initial_energy", energy),
                )
                - energy
            ),
            "energy_vs_mps_warm_start": float(energy - mps_record["energy"]),
            **_parameter_diagnostics(state),
            "bond_dims": list(state.bond_dims),
            "snapshot": str(snapshot),
            "optimization_residual_verified": bool(verified),
            "optimization_provisionally_converged": provisional_converged,
            "certification_required": certification_required,
            "certification_attempted": certification is not None,
            "certification_passed": bool(
                certification is not None and certification["passed"]
            ),
            "certification_failure_reason": (
                ""
                if certification is None
                else certification["failure_reason"]
            ),
        }
        if payload["result"]["converged"]:
            stop_reason = "converged"
            payload["result"]["stop_reason"] = stop_reason
        _write_json(output, payload)
        _save_snapshot(
            snapshot,
            state,
            cycle=len(cycles),
            low_gain_streak=low_gain_streak,
            protocol_fingerprint=fingerprint,
        )
        print(
            f"cycle {cycle_index + 1:3d}: E={energy:.12f} "
            f"E/N={energy / len(state.dims):.12f} "
            f"gain/site={gain / len(state.dims):.3e} "
            f"verified={verified} low_streak={low_gain_streak}",
            flush=True,
        )
        if payload["result"]["converged"]:
            break
    else:
        payload["result"]["stop_reason"] = stop_reason
        payload["status"] = "complete"
        _write_json(output, payload)

    capped_matrix_free = bool(
        cycles
        and any(
            "matrix_free_sector"
            in cycle["diagnostics"].get("backends", ())
            for cycle in cycles
        )
        and len(cycles) >= int(maximum_cycles)
    )
    needs_capped_certificate = bool(
        final_certification
        and capped_matrix_free
        and not payload["result"].get("certification_passed", False)
        and not certification_attempted_this_run
    )
    if needs_capped_certificate:
        certification_attempted_this_run = True
        certification = _certify_pair_stationarity(
            state,
            sweep_offset=2 * len(cycles),
            stopping_gain_per_site=stopping_gain_per_site,
            pair_dense_max_bytes=pair_dense_max_bytes,
            dense_estimated_peak_bytes=cycles[-1]["diagnostics"][
                "maximum_dense_estimated_peak_bytes"
            ],
            backend=final_certification_backend,
        )
        certifications.append(certification)
        payload["certifications"] = certifications
        energy = float(certification["energy"])
        payload["timing_seconds"]["total"] = float(
            payload["timing_seconds"]["total"] + certification["seconds"]
        )
        payload["result"].update(
            {
                "converged": bool(certification["passed"]),
                "stop_reason": (
                    "converged"
                    if certification["passed"]
                    else certification["failure_reason"]
                ),
                "energy": energy,
                "energy_per_site": energy / len(state.dims),
                "energy_gain_from_warm_start": float(
                    payload["warm_start"].get(
                        "letta_lift_energy",
                        payload["warm_start"].get(
                            "letta_initial_energy",
                            energy,
                        ),
                    )
                    - energy
                ),
                "energy_gain_from_two_site_start": float(
                    payload.get("pre_one_site", {}).get(
                        "energy",
                        payload["warm_start"].get(
                            "letta_initial_energy",
                            energy,
                        ),
                    )
                    - energy
                ),
                "energy_vs_mps_warm_start": float(
                    energy - mps_record["energy"]
                ),
                "certification_required": True,
                "certification_attempted": True,
                "certification_passed": bool(certification["passed"]),
                "certification_failure_reason": certification[
                    "failure_reason"
                ],
            }
        )
        if certification["passed"]:
            low_gain_streak = int(required_consecutive_cycles)
        _save_snapshot(
            snapshot,
            state,
            cycle=len(cycles),
            low_gain_streak=low_gain_streak,
            protocol_fingerprint=fingerprint,
        )

    payload["status"] = (
        "complete"
        if payload["result"].get("converged")
        else "capped"
        if len(cycles) >= int(maximum_cycles)
        else "running"
    )
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--tie-graph", choices=TIE_GRAPHS, default="all-j1")
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument(
        "--two-site-temporary-bond-dim",
        default="square",
        help="Temporary middle-bond cap for U1 two-site splits: 'square', 'fixed', or an integer.",
    )
    parser.add_argument(
        "--charge-assignment",
        choices=(
            "physical",
            "copy-neutral",
            "balanced",
            "occurrence",
            "first-occurrence",
            "owner",
        ),
        default="physical",
        help="How LETTA physical occurrences contribute to U1 charge flow.",
    )
    parser.add_argument(
        "--parameter-target",
        type=int,
        default=None,
        help="Expand U1 sector channels before optimization until active parameters best match this target.",
    )
    parser.add_argument(
        "--parameter-maximum-bond-dim",
        type=int,
        default=8,
        help="Maximum virtual bond dimension allowed during parameter-target expansion.",
    )
    parser.add_argument(
        "--parameter-expansion-scale",
        type=float,
        default=1.0e-3,
        help="Residual seed scale for newly added U1 sector channels.",
    )
    parser.add_argument("--mps-sweeps", type=int, default=12)
    parser.add_argument("--mps-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--pre-one-site-sweeps", type=int, default=2)
    parser.add_argument("--maximum-cycles", type=int, default=8)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--required-consecutive-cycles", type=int, default=2)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--frontier-backend",
        choices=FRONTIER_BACKENDS,
        default="identity-block",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--pair-operator-backend",
        choices=("auto", "dense", "block", "matrix-free"),
        default="auto",
    )
    parser.add_argument("--pair-dense-max-mib", type=float, default=64.0)
    parser.add_argument("--merged-dense-fallback-dim", type=int, default=512)
    parser.add_argument("--block-dense-component-max-size", type=int, default=64)
    parser.add_argument("--matrix-free-batch-size", type=int, default=2)
    parser.add_argument("--matrix-free-recycle-vectors", type=int, default=6)
    parser.add_argument(
        "--matrix-free-max-action-vectors",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--matrix-free-eig-tol",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument(
        "--matrix-free-preconditioner",
        choices=("adaptive", "charge-block-jacobi", "none"),
        default="adaptive",
    )
    parser.add_argument(
        "--matrix-free-action-backend",
        choices=("auto", "full", "fused", "prepared"),
        default="auto",
    )
    parser.add_argument(
        "--matrix-free-prepared-min-action-calls",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--final-certification-backend",
        choices=("auto", "dense", "block"),
        default="auto",
    )
    parser.add_argument("--no-final-certification", action="store_true")
    parser.add_argument("--parallel-block-components", action="store_true")
    parser.add_argument("--max-component-workers", type=int, default=None)
    parser.add_argument(
        "--skip-pair-root-verification",
        action="store_true",
        help="Speed mode: rely on component solver and endpoint checks instead of per-pair metric-dual certification.",
    )
    args = parser.parse_args()
    result = converge(
        j2=args.j2,
        tie_graph=args.tie_graph,
        bond_dim=args.bond_dim,
        two_site_temporary_bond_dim=args.two_site_temporary_bond_dim,
        charge_assignment=args.charge_assignment,
        parameter_target=args.parameter_target,
        parameter_maximum_bond_dim=args.parameter_maximum_bond_dim,
        parameter_expansion_scale=args.parameter_expansion_scale,
        mps_sweeps=args.mps_sweeps,
        mps_tolerance=args.mps_tolerance,
        pre_one_site_sweeps=args.pre_one_site_sweeps,
        maximum_cycles=args.maximum_cycles,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_consecutive_cycles=args.required_consecutive_cycles,
        tie_noise=args.tie_noise,
        seed=args.seed,
        output=args.output,
        snapshot=args.snapshot,
        resume=not args.no_resume,
        pair_operator_backend=args.pair_operator_backend,
        pair_dense_max_bytes=int(args.pair_dense_max_mib * 1024**2),
        merged_dense_fallback_dim=args.merged_dense_fallback_dim,
        block_dense_component_max_size=args.block_dense_component_max_size,
        parallel_block_components=args.parallel_block_components,
        max_component_workers=args.max_component_workers,
        verify_pair_roots=not args.skip_pair_root_verification,
        frontier_backend=args.frontier_backend,
        matrix_free_batch_size=args.matrix_free_batch_size,
        matrix_free_recycle_vectors=args.matrix_free_recycle_vectors,
        matrix_free_max_action_vectors=args.matrix_free_max_action_vectors,
        matrix_free_eig_tol=args.matrix_free_eig_tol,
        matrix_free_preconditioner=args.matrix_free_preconditioner,
        matrix_free_action_backend=args.matrix_free_action_backend,
        matrix_free_prepared_min_action_calls=(
            args.matrix_free_prepared_min_action_calls
        ),
        final_certification=not args.no_final_certification,
        final_certification_backend=args.final_certification_backend,
    )
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
