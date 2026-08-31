#!/usr/bin/env python3
"""Initialize or continue exact projected-U(1) LETTA on the 6x6 J1-J2 model.

The optimized state is ``P_{2Sz=0}|Psi(A)>``.  The projector is fixed, while
every coordinate of every unrestricted LETTA tensor remains variational.
The default performs no sweep because even one directional projected sweep is
substantially more expensive than constructing and measuring the state.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.converge_frontier_letta_bond_expansion_6x6 import (
    _state_from_snapshot,
)
from pyqed.letta import SectorProjectedLETTA


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = (
    RESULTS
    / "frontier_letta_two_site_numerical_full_ng40_pair1_ng40_pair_converged_6x6.json"
)
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_SOURCE_RESULT.with_suffix(".npz")
DEFAULT_MPS_REFERENCE = RESULTS / "frontier_letta_block_sparse_6x6_mps_references.json"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_sector_projected_u1_6x6_j2_0p5.json"
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT.with_suffix(".npz")
EXPECTED_PARAMETERS = 4008
NSITES = 36
CHECKPOINT_PAIR_ENERGY_TOL = 5.0e-10
RECONSTRUCTION_ENERGY_TOL_PER_SITE = 1.0e-7


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_snapshot_atomic(
    path,
    state,
    *,
    tie_edges,
    energy,
    initial_projected_energy,
    completed_passes,
    protocol_fingerprint,
    checkpoint_id,
    raw_source_energy,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        tie_edges=np.asarray(tie_edges, dtype=np.int64),
        recorded_energy=np.asarray(float(energy)),
        initial_projected_energy=np.asarray(float(initial_projected_energy)),
        completed_passes=np.asarray(int(completed_passes), dtype=np.int64),
        protocol_fingerprint=np.asarray(str(protocol_fingerprint)),
        checkpoint_id=np.asarray(str(checkpoint_id)),
        target_two_sz=np.asarray(0, dtype=np.int64),
        local_two_sz=np.asarray(((1, -1),) * NSITES, dtype=np.int64),
        raw_source_energy=np.asarray(float(raw_source_energy)),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    temporary.replace(path)


def _fingerprint(protocol):
    stable = dict(protocol)
    # Increasing the cap is the intended safe-resume workflow.
    stable.pop("maximum_directional_passes", None)
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _checkpoint_id(protocol_fingerprint, completed_passes, energy):
    encoded = (
        f"{protocol_fingerprint}:{int(completed_passes)}:{float(energy).hex()}"
    ).encode()
    return sha256(encoded).hexdigest()


def _read_snapshot_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        tensor_keys = tuple(key for key in archive.files if key.startswith("tensor_"))
        required = {
            "virtual_bond_dims",
            "tie_edges",
            "recorded_energy",
            "initial_projected_energy",
            "completed_passes",
            "protocol_fingerprint",
            "checkpoint_id",
            "target_two_sz",
            "local_two_sz",
            "raw_source_energy",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"resume snapshot is missing metadata: {', '.join(missing)}"
            )
        return {
            "tensor_count": len(tensor_keys),
            "bond_dims": tuple(int(value) for value in archive["virtual_bond_dims"]),
            "tie_edges": np.array(archive["tie_edges"], copy=True),
            "recorded_energy": float(archive["recorded_energy"]),
            "initial_projected_energy": float(archive["initial_projected_energy"]),
            "completed_passes": int(archive["completed_passes"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
            "checkpoint_id": str(archive["checkpoint_id"]),
            "target_two_sz": int(archive["target_two_sz"]),
            "local_two_sz": np.array(archive["local_two_sz"], copy=True),
            "raw_source_energy": float(archive["raw_source_energy"]),
        }


def _source_tie_edges(path):
    with np.load(path, allow_pickle=False) as archive:
        return np.array(archive["tie_edges"], dtype=np.int64, copy=True)


def _source_projection_metadata(path):
    required = {"recorded_energy", "target_two_sz", "local_two_sz"}
    projection_markers = {"target_two_sz", "local_two_sz"}
    with np.load(path, allow_pickle=False) as archive:
        present_markers = projection_markers.intersection(archive.files)
        if not present_markers:
            return None
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                "projected source checkpoint is missing metadata: " + ", ".join(missing)
            )
        local_two_sz = np.array(archive["local_two_sz"], copy=True)
        if int(archive["target_two_sz"]) != 0:
            raise RuntimeError("projected source checkpoint has the wrong target.")
        if local_two_sz.shape != (NSITES, 2) or not np.array_equal(
            local_two_sz,
            np.asarray(((1, -1),) * NSITES),
        ):
            raise RuntimeError(
                "projected source checkpoint has different local charges."
            )
        return {"recorded_energy": float(archive["recorded_energy"])}


def _mps_d32_energy(path, source):
    path = Path(path)
    if path.is_file():
        reference = json.loads(path.read_text(encoding="utf-8"))
        row = reference.get("results", {}).get("mps_d32")
        if row is not None and "energy" in row:
            return float(row["energy"])
    value = source.get("result", {}).get("dense_mps_d32_energy")
    return None if value is None else float(value)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _pass_record(row, *, pass_index, seconds, energy_before, energy):
    updates = row["updates"]
    failures = [update for update in updates if not update.solver_converged]
    identity_errors = [
        update.solver_metric_identity_error
        for update in updates
        if update.solver_metric_is_identity
    ]
    coordinate_residuals = [
        update.solver_coordinate_residual_norm
        for update in updates
        if np.isfinite(update.solver_coordinate_residual_norm)
    ]
    return {
        "pass": int(pass_index + 1),
        "absolute_sweep": int(row["sweep"]),
        "direction": (
            "left_to_right" if int(row["sweep"]) % 2 == 0 else "right_to_left"
        ),
        "energy": float(energy),
        "energy_per_site": float(energy / NSITES),
        "energy_gain": float(energy_before - energy),
        "energy_gain_per_site": float((energy_before - energy) / NSITES),
        "seconds": float(seconds),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "site_updates": len(updates),
        "solver_failures": len(failures),
        "failure_sites": [int(update.site) for update in failures],
        "maximum_residual_norm": _finite_or_none(
            max((update.residual_norm for update in updates), default=0.0)
        ),
        "maximum_metric_rank": int(
            max((update.metric_rank for update in updates), default=0)
        ),
        "identity_metric_sites": int(
            sum(update.solver_metric_is_identity for update in updates)
        ),
        "maximum_identity_metric_error": _finite_or_none(
            max(identity_errors, default=0.0)
        ),
        "maximum_solver_coordinate_residual_norm": _finite_or_none(
            max(coordinate_residuals, default=0.0)
        ),
        "hamiltonian_matvecs": int(
            sum(update.hamiltonian_matvecs for update in updates)
        ),
        "metric_matvecs": int(sum(update.metric_matvecs for update in updates)),
        "environment_cache": str(row["environment_cache"]),
        "environment_checkpoint_interval": int(row["environment_checkpoint_interval"]),
        "fixed_environment_cache_elements": int(
            row["fixed_environment_cache_elements"]
        ),
    }


def _result_record(
    state,
    *,
    energy,
    initial_projected_energy,
    raw_source_energy,
    mps_d32_energy,
    passes,
    snapshot,
    checkpoint_id,
    source_is_projected,
    gain_tolerance,
    maximum_passes,
    sweep_offset,
):
    energy = float(energy)
    mps_gap = None if mps_d32_energy is None else float(energy - mps_d32_energy)
    last = passes[-1] if passes else None
    last_cycle = passes[-2:]
    cycle_gain = (
        None
        if len(last_cycle) < 2
        else max(abs(float(row["energy_gain"])) for row in last_cycle)
    )
    cycle_gain_per_site = (
        None if cycle_gain is None else float(cycle_gain / NSITES)
    )
    converged = bool(
        len(last_cycle) == 2
        and all(int(row["solver_failures"]) == 0 for row in last_cycle)
        and cycle_gain_per_site < float(gain_tolerance)
    )
    if converged:
        stop_reason = (
            "both directional one-site gains per site in the last cycle are "
            f"below {float(gain_tolerance):.6e}"
        )
    elif len(passes) >= int(maximum_passes):
        stop_reason = "maximum directional one-site pass cap reached"
    else:
        stop_reason = "more directional one-site passes requested"
    return {
        "converged": converged,
        "stop_reason": stop_reason,
        "directional_passes_completed": len(passes),
        "next_directional_sweep": int(sweep_offset + len(passes)),
        "energy": energy,
        "energy_per_site": energy / NSITES,
        "last_directional_gain": (
            None if last is None else float(last["energy_gain"])
        ),
        "last_directional_gain_per_site": (
            None if last is None else float(last["energy_gain"]) / NSITES
        ),
        "last_cycle_maximum_gain": cycle_gain,
        "last_cycle_maximum_gain_per_site": cycle_gain_per_site,
        "projection_energy_change_from_raw_source": (
            None if source_is_projected else energy - float(raw_source_energy)
        ),
        "energy_gain_from_source": float(initial_projected_energy - energy),
        "optimization_energy_gain": float(initial_projected_energy - energy),
        "optimization_energy_gain_per_site": float(
            (initial_projected_energy - energy) / NSITES
        ),
        "dense_mps_d32_energy": mps_d32_energy,
        "energy_above_dense_mps_d32": mps_gap,
        "energy_above_dense_mps_d32_per_site": (
            None if mps_gap is None else mps_gap / NSITES
        ),
        "parameters": int(state.nparameters),
        "dense_parameters": int(state.dense_nparameters),
        "serialized_tensors": len(state.tensors),
        "bond_dims": list(state.bond_dims),
        "checkpoint_id": checkpoint_id,
        "snapshot": str(Path(snapshot).resolve()),
    }


def _checkpoint(
    *,
    output,
    snapshot,
    payload,
    state,
    energy,
    tie_edges,
    initial_projected_energy,
    raw_source_energy,
):
    energy = float(energy)
    completed_passes = len(payload["directional_passes"])
    checkpoint_id = _checkpoint_id(
        payload["protocol_fingerprint"],
        completed_passes,
        energy,
    )
    payload["result"] = _result_record(
        state,
        energy=energy,
        initial_projected_energy=initial_projected_energy,
        raw_source_energy=raw_source_energy,
        mps_d32_energy=payload["reference"]["mps_d32_energy"],
        passes=payload["directional_passes"],
        snapshot=snapshot,
        checkpoint_id=checkpoint_id,
        source_is_projected=(payload["protocol"].get("source_state") == "projected"),
        gain_tolerance=payload["protocol"]["gain_tolerance"],
        maximum_passes=payload["protocol"]["maximum_directional_passes"],
        sweep_offset=payload["protocol"]["sweep_offset"],
    )
    if payload["result"]["converged"]:
        payload["status"] = "converged"
    elif completed_passes >= int(
        payload["protocol"]["maximum_directional_passes"]
    ):
        payload["status"] = "maximum_passes"
    else:
        payload["status"] = "running"
    _save_snapshot_atomic(
        snapshot,
        state,
        tie_edges=tie_edges,
        energy=energy,
        initial_projected_energy=initial_projected_energy,
        completed_passes=completed_passes,
        protocol_fingerprint=payload["protocol_fingerprint"],
        checkpoint_id=checkpoint_id,
        raw_source_energy=raw_source_energy,
    )
    _write_json_atomic(output, payload)


def _validate_resume(
    payload,
    metadata,
    *,
    protocol_fingerprint,
    maximum_passes,
):
    completed = len(payload.get("directional_passes", ()))
    expected_checkpoint_id = payload.get("result", {}).get("checkpoint_id")
    checks = (
        (payload.get("protocol_fingerprint"), protocol_fingerprint, "JSON protocol"),
        (
            metadata["protocol_fingerprint"],
            protocol_fingerprint,
            "snapshot protocol",
        ),
        (metadata["checkpoint_id"], expected_checkpoint_id, "checkpoint ID"),
        (metadata["completed_passes"], completed, "completed pass count"),
        (metadata["tensor_count"], NSITES, "serialized tensor count"),
        (metadata["target_two_sz"], 0, "target 2Sz"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RuntimeError(
                f"unsafe resume: {label} mismatch ({actual!r} != {expected!r})."
            )
    if int(maximum_passes) < completed:
        raise ValueError(
            f"maximum_passes={maximum_passes} is below the {completed} "
            "already completed passes."
        )
    if abs(metadata["recorded_energy"] - float(payload["result"]["energy"])) > float(
        payload["protocol"]["checkpoint_pair_energy_tolerance"]
    ):
        raise RuntimeError("unsafe resume: JSON and snapshot energies differ.")
    if metadata["local_two_sz"].shape != (NSITES, 2) or not np.array_equal(
        metadata["local_two_sz"],
        np.asarray(((1, -1),) * NSITES),
    ):
        raise RuntimeError("unsafe resume: local projected charges differ.")


def converge(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    mps_reference=DEFAULT_MPS_REFERENCE,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    maximum_passes=0,
    sweep_offset=0,
    gain_tolerance=1.0e-6,
    solver="whitened",
    metric_tol=1.0e-12,
    eig_tol=1.0e-10,
    maxiter=1600,
    max_subspace=96,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    mps_reference = Path(mps_reference)
    output = Path(output)
    snapshot = Path(snapshot)
    maximum_passes = int(maximum_passes)
    sweep_offset = int(sweep_offset)
    gain_tolerance = float(gain_tolerance)
    if maximum_passes < 0:
        raise ValueError("maximum_passes must be nonnegative.")
    if sweep_offset < 0:
        raise ValueError("sweep_offset must be nonnegative.")
    if not np.isfinite(gain_tolerance) or gain_tolerance < 0.0:
        raise ValueError("gain_tolerance must be finite and nonnegative.")
    if str(solver) not in {"direct", "whitened"}:
        raise ValueError("solver must be 'direct' or 'whitened'.")
    protected = {source_result.resolve(), source_snapshot.resolve()}
    if output.resolve() in protected or snapshot.resolve() in protected:
        raise ValueError("output checkpoints must not overwrite source artifacts.")

    output_exists = output.is_file()
    snapshot_exists = snapshot.is_file()
    if resume and output_exists != snapshot_exists:
        raise RuntimeError(
            "unsafe resume: JSON and NPZ checkpoints must either both exist "
            "or both be absent."
        )
    resumed = bool(resume and output_exists and snapshot_exists)

    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        model = payload["model"]
        protocol = dict(payload["protocol"])
        requested = {
            "sweep_offset": sweep_offset,
            "solver": str(solver),
            "metric_tol": float(metric_tol),
            "eig_tol": float(eig_tol),
            "maxiter": int(maxiter),
            "max_subspace": int(max_subspace),
            "gain_tolerance": gain_tolerance,
        }
        for key, value in requested.items():
            if protocol.get(key) != value:
                raise RuntimeError(
                    f"unsafe resume: requested {key}={value!r} differs from "
                    f"checkpoint value {protocol.get(key)!r}."
                )
        protocol["maximum_directional_passes"] = maximum_passes
        source = None
        source_projection = None
        source_is_projected = protocol.get("source_state") == "projected"
        source_recorded_energy = float(payload["source"]["energy_recorded"])
    else:
        source = json.loads(source_result.read_text(encoding="utf-8"))
        model = source["model"]
        source_projection = _source_projection_metadata(source_snapshot)
        source_is_projected = source_projection is not None
        source_recorded_energy = float(source["result"]["energy"])
        protocol = {
            "ansatz": "P_Q |Psi(A)> with every unrestricted A coordinate retained",
            "symmetry": "exact variation-after-projection U(1)",
            "source_state": "projected" if source_is_projected else "unrestricted",
            "target_two_sz": 0,
            "local_two_sz": [1, -1],
            "local_tensor_masks": False,
            "expected_parameters": EXPECTED_PARAMETERS,
            "source_result": str(source_result.resolve()),
            "source_snapshot": str(source_snapshot.resolve()),
            "frontier_backend": "identity_block",
            "objective_mpo": "sparse factorized H times P_Q",
            "materialize_objective_mpo": False,
            "balance_initial_gauges": False,
            "optimization": "directional one-site variational sweeps",
            "sweep_offset": sweep_offset,
            "solver": str(solver),
            "metric_tol": float(metric_tol),
            "eig_tol": float(eig_tol),
            "maxiter": int(maxiter),
            "max_subspace": int(max_subspace),
            "environment_cache": "checkpointed",
            "frontier_canonicalization": False,
            "maximum_directional_passes": maximum_passes,
            "gain_tolerance": gain_tolerance,
            "gain_tolerance_units": "energy_per_site",
            "convergence_rule": (
                "both directional one-site gains per site in one complete "
                "alternating cycle are below gain_tolerance, with no solver failures"
            ),
            "checkpoint_pair_energy_tolerance": CHECKPOINT_PAIR_ENERGY_TOL,
            "reconstruction_energy_tolerance_per_site": (
                RECONSTRUCTION_ENERGY_TOL_PER_SITE
            ),
            "safe_resume": True,
        }
    if (
        int(model["nrows"]) != 6
        or int(model["ncols"]) != 6
        or int(model["nsites"]) != NSITES
    ):
        raise ValueError("the source checkpoint must be the 6x6 model.")
    protocol_fingerprint = _fingerprint(protocol)
    invocation_start = perf_counter()

    if resumed:
        metadata = _read_snapshot_metadata(snapshot)
        _validate_resume(
            payload,
            metadata,
            protocol_fingerprint=protocol_fingerprint,
            maximum_passes=maximum_passes,
        )
        completed = len(payload["directional_passes"])
        if payload.get("result", {}).get("converged") or completed == maximum_passes:
            print(
                f"checkpoint already has {completed} projected pass(es): "
                f"E={payload['result']['energy']:.12f}",
                flush=True,
            )
            return payload
        restore_start = perf_counter()
        raw_state = _state_from_snapshot(model, snapshot)
        raw_restore_seconds = perf_counter() - restore_start
        tie_edges = metadata["tie_edges"]
        initial_projected_energy = metadata["initial_projected_energy"]
        raw_source_energy = metadata["raw_source_energy"]
        cumulative_before = float(payload["timing_seconds"]["total"])
    else:
        restore_start = perf_counter()
        raw_state = _state_from_snapshot(model, source_snapshot)
        raw_restore_seconds = perf_counter() - restore_start
        tie_edges = _source_tie_edges(source_snapshot)
        if source_is_projected:
            raw_source_energy = source_recorded_energy
        else:
            raw_source_energy = float(raw_state.energy)
            if (
                abs(raw_source_energy - source_recorded_energy)
                > NSITES * protocol["reconstruction_energy_tolerance_per_site"]
            ):
                raise RuntimeError(
                    "source JSON and unrestricted snapshot energies are inconsistent."
                )
        initial_projected_energy = None
        cumulative_before = 0.0

    projection_start = perf_counter()
    local_charges = tuple(((1,), (-1,)) for _ in range(NSITES))
    state = SectorProjectedLETTA.from_unrestricted(
        raw_state,
        local_charges=local_charges,
        target=(0,),
        frontier_backend="identity_block",
        _balance_initial_gauges=False,
    )
    projection_seconds = perf_counter() - projection_start
    if state.nparameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAMETERS} unrestricted tensor parameters, "
            f"got {state.nparameters}."
        )
    if hasattr(state, "local_masks"):
        raise RuntimeError("projected LETTA unexpectedly contains local masks.")
    if any(active != dense for active, dense in state.local_support_sizes()):
        raise RuntimeError("not every unrestricted tensor coordinate is active.")
    measured_energy = float(state.energy)
    if initial_projected_energy is None:
        initial_projected_energy = measured_energy
        if source_is_projected and (
            abs(measured_energy - source_recorded_energy)
            > NSITES * protocol["reconstruction_energy_tolerance_per_site"]
            or abs(source_projection["recorded_energy"] - source_recorded_energy)
            > protocol["checkpoint_pair_energy_tolerance"]
        ):
            raise RuntimeError(
                "source JSON, projected snapshot, and reconstructed projected "
                "energy are inconsistent: "
                f"recorded={source_recorded_energy:.16g}, "
                f"reconstructed={measured_energy:.16g}, "
                f"difference={measured_energy - source_recorded_energy:.6e}."
            )
    elif (
        abs(measured_energy - metadata["recorded_energy"])
        > NSITES * protocol["reconstruction_energy_tolerance_per_site"]
    ):
        raise RuntimeError(
            "resume snapshot does not reproduce its projected energy: "
            f"recorded={metadata['recorded_energy']:.16g}, "
            f"reconstructed={measured_energy:.16g}, "
            f"difference={measured_energy - metadata['recorded_energy']:.6e}."
        )

    if not resumed:
        setup_seconds = perf_counter() - invocation_start
        mps_energy = _mps_d32_energy(mps_reference, source)
        source_payload = {
            "result": str(source_result.resolve()),
            "snapshot": str(source_snapshot.resolve()),
            "input_state": ("projected" if source_is_projected else "unrestricted"),
            "energy_recorded": source_recorded_energy,
            "energy_reconstructed": (
                measured_energy if source_is_projected else raw_source_energy
            ),
            "energy_per_site": (
                measured_energy if source_is_projected else raw_source_energy
            )
            / NSITES,
            "parameters": int(raw_state.nparameters),
            "bond_dims": list(raw_state.bond_dims),
            "directional_passes_completed": int(
                source["result"].get("directional_passes_completed", 0)
            ),
        }
        payload = {
            "status": "maximum_passes" if maximum_passes == 0 else "initialized",
            "model": model,
            "protocol": protocol,
            "protocol_fingerprint": protocol_fingerprint,
            "source": source_payload,
            "projection": {
                "definition": "P_{2Sz=0} |Psi(A)>",
                "fixed_projector": True,
                "all_tensor_coordinates_variational": True,
                "local_tensor_masks": False,
                "target_two_sz": 0,
                "local_two_sz": [1, -1],
                "projector_bond_dims": list(state.projection.mpo_bond_dims),
                "maximum_projector_bond": int(state.projection.max_mpo_bond),
                "hamiltonian_mpo_representation": (
                    state.projected_hamiltonian_mpo_diagnostics["representation"]
                ),
                "maximum_hamiltonian_mpo_bond": int(
                    state.projected_hamiltonian_mpo_diagnostics["max_bond_dim"]
                ),
                "source_reconstruction_energy_difference": float(
                    measured_energy - source_recorded_energy
                ),
                "source_reconstruction_energy_difference_per_site": float(
                    (measured_energy - source_recorded_energy) / NSITES
                ),
                "parameters": int(state.nparameters),
                "dense_parameters": int(state.dense_nparameters),
                "serialized_tensors": len(state.tensors),
            },
            "contraction": {
                "frontier_backend": state.frontier_backend,
                "exact": bool(state.contraction_is_exact),
                "factorized_objective_mpo": bool(
                    state._hamiltonian_frontier.factorized_mpo
                ),
                "stored_objective_mpo_elements": int(
                    state._hamiltonian_frontier.stored_mpo_elements
                ),
                "materialized_objective_mpo_elements": int(
                    state._hamiltonian_frontier.dense_mpo_elements
                ),
                "peak_frontier_elements": int(state.peak_frontier_elements),
                "norm_peak_frontier_elements": int(state.norm_peak_frontier_elements),
                "hamiltonian_peak_frontier_elements": int(
                    state.hamiltonian_peak_frontier_elements
                ),
                "norm_contraction_plans": int(state._norm_frontier.plan_count),
                "hamiltonian_contraction_plans": int(
                    state._hamiltonian_frontier.plan_count
                ),
                "full_environment_cache_elements": int(
                    state.cached_environment_elements
                ),
                "checkpointed_environment_cache_elements": int(
                    state.fixed_environment_cache_elements(mode="checkpointed")
                ),
            },
            "reference": {
                "mps_reference": str(mps_reference.resolve()),
                "mps_d32_energy": mps_energy,
            },
            "directional_passes": [],
            "result": {},
            "timing_seconds": {
                "source_restore": float(raw_restore_seconds),
                "projector_and_frontier_construction": float(projection_seconds),
                "initial_setup": float(setup_seconds),
                "directional_passes": [],
                "optimization_total": 0.0,
                "total": float(setup_seconds),
            },
        }
        _checkpoint(
            output=output,
            snapshot=snapshot,
            payload=payload,
            state=state,
            energy=measured_energy,
            tie_edges=tie_edges,
            initial_projected_energy=initial_projected_energy,
            raw_source_energy=raw_source_energy,
        )
        if maximum_passes == 0:
            print(
                f"projected E={payload['result']['energy']:.12f}, "
                f"passes=0, parameters={state.nparameters}, "
                f"P bond={state.projection.max_mpo_bond}",
                flush=True,
            )
            return payload
    else:
        payload["protocol"]["maximum_directional_passes"] = maximum_passes
        payload["timing_seconds"]["last_resume_source_restore"] = float(
            raw_restore_seconds
        )
        payload["timing_seconds"]["last_resume_projector_construction"] = float(
            projection_seconds
        )

    passes = payload["directional_passes"]
    for pass_index in range(len(passes), maximum_passes):
        energy_before = float(state.energy)
        pass_start = perf_counter()
        state.run(
            nsweeps=1,
            sweep_offset=sweep_offset + pass_index,
            tol=0.0,
            solver=str(solver),
            metric_tol=float(metric_tol),
            eig_tol=float(eig_tol),
            maxiter=int(maxiter),
            max_subspace=int(max_subspace),
            frontier_canonicalization=False,
            environment_cache="checkpointed",
            verbose=True,
        )
        pass_seconds = perf_counter() - pass_start
        energy = float(state.energy)
        passes.append(
            _pass_record(
                state.history[0],
                pass_index=pass_index,
                seconds=pass_seconds,
                energy_before=energy_before,
                energy=energy,
            )
        )
        payload["timing_seconds"]["directional_passes"] = [
            float(row["seconds"]) for row in passes
        ]
        payload["timing_seconds"]["optimization_total"] = float(
            sum(row["seconds"] for row in passes)
        )
        payload["timing_seconds"]["total"] = float(
            cumulative_before + perf_counter() - invocation_start
        )
        _checkpoint(
            output=output,
            snapshot=snapshot,
            payload=payload,
            state=state,
            energy=energy,
            tie_edges=tie_edges,
            initial_projected_energy=initial_projected_energy,
            raw_source_energy=raw_source_energy,
        )
        if payload["result"]["converged"]:
            break

    print(
        f"projected E={payload['result']['energy']:.12f}, "
        f"passes={len(passes)}, parameters={state.nparameters}, "
        f"P bond={state.projection.max_mpo_bond}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--mps-reference", type=Path, default=DEFAULT_MPS_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--maximum-passes", type=int, default=0)
    parser.add_argument("--sweep-offset", type=int, default=0)
    parser.add_argument(
        "--gain-tolerance",
        type=float,
        default=1.0e-6,
        help="maximum energy gain per site in each direction of the last cycle",
    )
    parser.add_argument(
        "--solver",
        choices=("direct", "whitened"),
        default="whitened",
    )
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--max-subspace", type=int, default=96)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    converge(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        mps_reference=args.mps_reference,
        output=args.output,
        snapshot=args.snapshot,
        maximum_passes=args.maximum_passes,
        sweep_offset=args.sweep_offset,
        gain_tolerance=args.gain_tolerance,
        solver=args.solver,
        metric_tol=args.metric_tol,
        eig_tol=args.eig_tol,
        maxiter=args.maxiter,
        max_subspace=args.max_subspace,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
