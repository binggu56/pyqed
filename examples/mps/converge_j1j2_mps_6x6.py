#!/usr/bin/env python3
"""Outer-converged dense-MPS reference for the open 6x6 J1-J2 model.

The low-level DMRG convergence flag compares final local eigenvalues from
different end bonds before the two-site SVD truncation.  This driver disables
that stopping test, measures normalized post-truncation expectations after
both directions, and retains the lowest-energy state seen in each cycle.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import square_j1_j2_bonds
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.mps import DMRG, MPS, MPO
from pyqed.mps.dmrg import _normalized_mps_mpo_expectation


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "j1j2_mps_outer_converged_6x6.json"
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT.with_suffix(".npz")
CHECKPOINT_ENERGY_TOL = 5.0e-10


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _validated_bond_dims(values) -> tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if not dims or any(value < 1 for value in dims):
        raise ValueError("bond_dims must contain positive integers.")
    if any(right <= left for left, right in zip(dims, dims[1:])):
        raise ValueError("bond_dims must be strictly increasing.")
    return dims


def _random_mps(nsites: int, bond_dim: int, seed: int) -> MPS:
    ranks = tuple(
        min(int(bond_dim), 2 ** min(cut, int(nsites) - cut))
        for cut in range(int(nsites) + 1)
    )
    rng = np.random.default_rng(seed)
    tensors = [
        rng.normal(size=(ranks[site], 2, ranks[site + 1]))
        / np.sqrt(2 * ranks[site] * ranks[site + 1])
        for site in range(int(nsites))
    ]
    return MPS(tensors, labels=["lv", "p", "rv"]).right_canonicalize()


def _model(nrows: int, ncols: int, j2: float):
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, float(j2)) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nrows * ncols, weighted)
    mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    return mpo, nearest, diagonals


def _energy(state: MPS, mpo: MPO) -> float:
    ordered = state.to_order(["lv", "p", "rv"])
    return float(_normalized_mps_mpo_expectation(ordered.factors, mpo.factors))


def _copy_dense_state(factors) -> MPS:
    return MPS(
        [np.asarray(tensor).copy() for tensor in factors],
        labels=["lv", "p", "rv"],
    )


def _state_parameter_counts(state: MPS) -> dict:
    ordered = state.to_order(["lv", "p", "rv"])
    tensors = [np.asarray(tensor) for tensor in ordered.factors]
    bond_dims = [int(tensors[0].shape[0])]
    bond_dims.extend(int(tensor.shape[2]) for tensor in tensors)
    return {
        "stored_parameters": int(sum(tensor.size for tensor in tensors)),
        "actual_bond_dims": bond_dims,
        "maximum_actual_bond_dim": max(bond_dims),
    }


def _profile_solver_records(value, path=()):
    if isinstance(value, dict):
        if "residual_norm" in value and (
            "converged" in value
            or path[-1:] in {("local_solver",), ("packed_local_davidson",)}
        ):
            yield value
        for key, child in value.items():
            yield from _profile_solver_records(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _profile_solver_records(child, path)


def _cycle_diagnostics(rows) -> dict:
    truncations = []
    residuals = []
    failed_updates = 0
    profiled_updates = 0
    update_count = 0
    for row in rows:
        for update in row.get("updates", ()):
            update_count += 1
            truncation = update.get("truncation")
            if truncation is not None and np.isfinite(truncation):
                truncations.append(float(truncation))
            records = list(
                _profile_solver_records(update.get("matvec_profile") or {})
            )
            if records:
                profiled_updates += 1
                if any(record.get("converged") is False for record in records):
                    failed_updates += 1
                residuals.extend(
                    float(record["residual_norm"])
                    for record in records
                    if np.isfinite(record.get("residual_norm", np.nan))
                )
    return {
        "local_updates": int(update_count),
        "profiled_local_updates": int(profiled_updates),
        "reported_solver_failures": int(failed_updates),
        "maximum_reported_local_residual": max(residuals, default=None),
        "maximum_truncation_proxy": max(truncations, default=None),
        "truncation_definition": "sum of discarded singular values",
    }


def _below_gain_tolerance(changes, diagnostics, gain_tolerance: float) -> bool:
    """Test convergence of the retained monotonic variational state."""
    return bool(
        diagnostics["reported_solver_failures"] == 0
        and changes["maximum_accepted_directional_gain_per_site"]
        < float(gain_tolerance)
    )


def _monotonic_direction_record(
    energy_before: float,
    candidates,
    *,
    nsites: int,
) -> dict:
    """Score LR/RL endpoints while retaining only variational improvements."""

    best = float(energy_before)
    numerical_slack = 512.0 * np.finfo(float).eps * max(1.0, abs(best))
    rows = []
    for direction, energy in candidates:
        energy = float(energy)
        reference_energy = best
        signed_change = energy - reference_energy
        absolute_change = abs(signed_change)
        raw_gain = -signed_change
        accepted = bool(raw_gain > numerical_slack)
        accepted_gain = raw_gain if accepted else 0.0
        if accepted:
            best = energy
        rows.append(
            {
                "direction": str(direction),
                "candidate_energy": energy,
                "reference_energy": reference_energy,
                "signed_change_from_running_best": signed_change,
                "absolute_change_from_running_best": absolute_change,
                "absolute_change_from_running_best_per_site": (
                    absolute_change / int(nsites)
                ),
                "raw_gain_from_running_best": raw_gain,
                "accepted": accepted,
                "accepted_gain": accepted_gain,
                "accepted_gain_per_site": accepted_gain / int(nsites),
                "running_best_energy": best,
                "energy_increase_rejected": bool(
                    signed_change > numerical_slack
                ),
            }
        )
    accepted_gains = [row["accepted_gain"] for row in rows]
    absolute_changes = [
        row["absolute_change_from_running_best"] for row in rows
    ]
    return {
        "directional_candidates": rows,
        "directional_energies": [
            row["candidate_energy"] for row in rows
        ],
        "directional_energy_gains": accepted_gains,
        "maximum_accepted_directional_gain": max(accepted_gains, default=0.0),
        "maximum_accepted_directional_gain_per_site": (
            max(accepted_gains, default=0.0) / int(nsites)
        ),
        "maximum_absolute_directional_change": max(
            absolute_changes, default=0.0
        ),
        "maximum_absolute_directional_change_per_site": (
            max(absolute_changes, default=0.0) / int(nsites)
        ),
        "best_energy": best,
        "rejected_energy_increases": int(
            sum(row["energy_increase_rejected"] for row in rows)
        ),
    }


def _stage_record(
    bond_dim: int,
    energy: float,
    *,
    initialization: str,
    pass_limit: int,
) -> dict:
    return {
        "bond_dim": int(bond_dim),
        "optimizer": "dense two-site DMRG with monotonic outer convergence",
        "symmetry": "none",
        "initialization": str(initialization),
        "status": "running",
        "converged": False,
        "initial_energy": float(energy),
        "energy": float(energy),
        "energy_per_site": None,
        "directional_pass_limit": int(pass_limit),
        "directional_passes_completed": 0,
        "low_gain_streak": 0,
        "cycles": [],
        "optimization_seconds": 0.0,
    }


def _stage_summary(stage: dict, nsites: int) -> dict:
    last = stage["cycles"][-1] if stage["cycles"] else {}
    return {
        "optimizer": stage["optimizer"],
        "symmetry": stage["symmetry"],
        "bond_dim": int(stage["bond_dim"]),
        "initialization": stage["initialization"],
        "initial_energy": float(stage["initial_energy"]),
        "energy": float(stage["energy"]),
        "energy_per_site": float(stage["energy"]) / int(nsites),
        "optimization_seconds": float(stage["optimization_seconds"]),
        "directional_pass_limit": int(stage["directional_pass_limit"]),
        "directional_passes_completed": int(
            stage["directional_passes_completed"]
        ),
        "converged": bool(stage["converged"]),
        "final_delta_energy": last.get("maximum_accepted_directional_gain"),
        "final_delta_energy_per_site": last.get(
            "maximum_accepted_directional_gain_per_site"
        ),
        "final_maximum_absolute_directional_change": last.get(
            "maximum_absolute_directional_change"
        ),
        "final_maximum_absolute_directional_change_per_site": last.get(
            "maximum_absolute_directional_change_per_site"
        ),
        "directional_pass_energies": [
            energy
            for cycle in stage["cycles"]
            for energy in cycle["directional_energies"]
        ],
        **stage.get("state", {}),
    }


def _refresh_result(
    payload: dict,
    *,
    stage_index: int,
    snapshot: Path,
) -> None:
    stages = payload["stages"]
    stage = stages[int(stage_index)]
    nsites = int(payload["model"]["nsites"])
    last = stage["cycles"][-1] if stage["cycles"] else {}
    final_requested = stage_index == len(payload["protocol"]["bond_dims"]) - 1
    converged = bool(stage["converged"] and final_requested)
    if converged:
        stop_reason = "all requested fixed-D stages converged"
    elif stage["status"] == "maximum_passes":
        stop_reason = (
            f"MPS D={stage['bond_dim']} reached its directional-pass limit"
        )
    else:
        stop_reason = f"optimizing MPS D={stage['bond_dim']}"
    payload["status"] = (
        "complete"
        if converged
        else "maximum_passes"
        if stage["status"] == "maximum_passes"
        else "running"
    )
    payload["results"] = {
        f"mps_d{item['bond_dim']}": _stage_summary(item, nsites)
        for item in stages
    }
    directional_gains = last.get("directional_energy_gains", ())
    payload["result"] = {
        "converged": converged,
        "stop_reason": stop_reason,
        "bond_dim": int(stage["bond_dim"]),
        "energy": float(stage["energy"]),
        "energy_per_site": float(stage["energy"]) / nsites,
        # tools/letta_hpc.py uses this as the current-stage resume cap.
        "directional_passes_completed": int(
            stage["directional_passes_completed"]
        ),
        "total_directional_passes_completed": int(
            sum(item["directional_passes_completed"] for item in stages)
        ),
        "next_directional_sweep": int(stage["directional_passes_completed"]),
        "last_directional_gain": (
            directional_gains[-1] if directional_gains else None
        ),
        "last_directional_gain_per_site": (
            directional_gains[-1] / nsites if directional_gains else None
        ),
        "last_cycle_maximum_gain": last.get(
            "maximum_accepted_directional_gain"
        ),
        "last_cycle_maximum_gain_per_site": last.get(
            "maximum_accepted_directional_gain_per_site"
        ),
        "last_cycle_maximum_absolute_change": last.get(
            "maximum_absolute_directional_change"
        ),
        "last_cycle_maximum_absolute_change_per_site": last.get(
            "maximum_absolute_directional_change_per_site"
        ),
        "low_gain_streak": int(stage["low_gain_streak"]),
        "requested_bond_dims": list(payload["protocol"]["bond_dims"]),
        "completed_bond_dims": [
            int(item["bond_dim"]) for item in stages if item["converged"]
        ],
        "snapshot": str(snapshot),
    }


def _checkpoint_arrays(
    state: MPS,
    *,
    checkpoint_id: str,
    energy: float,
    protocol_fingerprint: str,
    stage_index: int,
    low_gain_streak: int,
    directional_passes_completed: int,
) -> dict[str, np.ndarray]:
    dense = state.to_order(["lv", "p", "rv"])
    arrays = {
        f"tensor_{site:03d}": np.asarray(tensor)
        for site, tensor in enumerate(dense.factors)
    }
    arrays.update(
        {
            "checkpoint_id": np.asarray(str(checkpoint_id)),
            "recorded_energy": np.asarray(float(energy)),
            "protocol_fingerprint": np.asarray(str(protocol_fingerprint)),
            "stage_index": np.asarray(int(stage_index), dtype=np.int64),
            "low_gain_streak": np.asarray(
                int(low_gain_streak), dtype=np.int64
            ),
            "directional_passes_completed": np.asarray(
                int(directional_passes_completed), dtype=np.int64
            ),
            "tensor_count": np.asarray(len(dense.factors), dtype=np.int64),
        }
    )
    return arrays


def _write_checkpoint_pair(
    output: Path,
    snapshot: Path,
    payload: dict,
    state: MPS,
    *,
    stage_index: int,
    energy: float,
) -> None:
    checkpoint_id = uuid.uuid4().hex
    stage = payload["stages"][int(stage_index)]
    _refresh_result(payload, stage_index=stage_index, snapshot=snapshot)
    payload["result"]["checkpoint_id"] = checkpoint_id
    payload["result"]["checkpoint_energy_tolerance"] = CHECKPOINT_ENERGY_TOL
    arrays = _checkpoint_arrays(
        state,
        checkpoint_id=checkpoint_id,
        energy=energy,
        protocol_fingerprint=payload["protocol_fingerprint"],
        stage_index=stage_index,
        low_gain_streak=stage["low_gain_streak"],
        directional_passes_completed=stage["directional_passes_completed"],
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_name(snapshot.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(snapshot)
    _atomic_json(output, payload)


def _load_checkpoint(
    snapshot: Path,
    *,
    payload: dict,
    mpo: MPO,
) -> tuple[MPS, int]:
    result = payload.get("result", {})
    checkpoint_id = str(result.get("checkpoint_id") or "")
    if not checkpoint_id:
        raise RuntimeError("result JSON has no checkpoint_id.")
    with np.load(snapshot, allow_pickle=False) as archive:
        snapshot_id = str(np.asarray(archive["checkpoint_id"]).item())
        energy = float(np.asarray(archive["recorded_energy"]).item())
        fingerprint = str(np.asarray(archive["protocol_fingerprint"]).item())
        stage_index = int(np.asarray(archive["stage_index"]).item())
        tensor_count = int(np.asarray(archive["tensor_count"]).item())
        stored_streak = int(np.asarray(archive["low_gain_streak"]).item())
        stored_passes = int(
            np.asarray(archive["directional_passes_completed"]).item()
        )
        tensors = [
            np.asarray(archive[f"tensor_{site:03d}"]).copy()
            for site in range(tensor_count)
        ]
    if snapshot_id != checkpoint_id:
        raise RuntimeError("result JSON and state snapshot checkpoint IDs disagree.")
    if fingerprint != payload["protocol_fingerprint"]:
        raise RuntimeError("state snapshot protocol fingerprint is incompatible.")
    if abs(energy - float(result["energy"])) > CHECKPOINT_ENERGY_TOL:
        raise RuntimeError("result JSON and state snapshot energies disagree.")
    if not 0 <= stage_index < len(payload["stages"]):
        raise RuntimeError("state snapshot stage index is invalid.")
    stage = payload["stages"][stage_index]
    if (
        stored_streak != int(stage["low_gain_streak"])
        or stored_passes != int(stage["directional_passes_completed"])
    ):
        raise RuntimeError("result JSON and state snapshot progress disagree.")
    state = MPS(tensors, labels=["lv", "p", "rv"])
    measured = _energy(state, mpo)
    if abs(measured - energy) > 5.0e-9:
        raise RuntimeError(
            "restored MPS energy is inconsistent with checkpoint metadata."
        )
    return state, stage_index


def _run_full_cycle(
    state: MPS,
    mpo: MPO,
    *,
    bond_dim: int,
    performance: str,
    davidson_tol: float,
    davidson_max_iter: int,
) -> tuple[MPS, list[dict], list[tuple[str, float]], float]:
    """Run LR+RL and return the lowest fresh-expectation endpoint."""

    candidates: list[tuple[str, float, MPS]] = []

    def capture(**info):
        direction = str(info.get("direction", ""))
        if direction not in {"lr", "rl"}:
            return
        candidate = _copy_dense_state(info["mps"])
        candidates.append((direction, _energy(candidate, mpo), candidate))

    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=state,
        nsweeps=2,
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=0,
        sweep_tol=0.0,
        davidson_tol=float(davidson_tol),
        davidson_max_iter=int(davidson_max_iter),
        noise=0.0,
        recenter_final=False,
        final_expectation=True,
        performance=performance,
        sweep_callback=capture,
    ).run()
    rows = [
        row
        for row in solver.sweep_history
        if row.get("direction") in {"lr", "rl"}
    ]
    if len(rows) != 2 or len(candidates) != 2:
        raise RuntimeError(
            "one outer MPS cycle must produce exactly one LR and one RL endpoint."
        )
    initial_energy = _energy(state, mpo)
    final_state = solver.ground_state.copy()
    final_energy = float(solver.e_tot)
    choices = [("start", initial_energy, state), *candidates]
    if abs(final_energy - candidates[-1][1]) > 5.0e-10:
        choices.append(("final", final_energy, final_state))
    _label, best_energy, best_state = min(choices, key=lambda item: item[1])
    directional = [(direction, energy) for direction, energy, _state in candidates]
    return best_state.copy(), rows, directional, float(best_energy)


def converge(
    *,
    nrows: int = 6,
    ncols: int = 6,
    j2: float = 0.5,
    bond_dims=(4, 8, 16, 32),
    maximum_directional_passes: int = 200,
    gain_tolerance: float = 1.0e-7,
    required_consecutive_cycles: int = 2,
    davidson_tol: float = 1.0e-10,
    davidson_max_iter: int = 100,
    performance: str = "auto",
    seed: int = 7,
    output: Path = DEFAULT_OUTPUT,
    snapshot: Path = DEFAULT_SNAPSHOT,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    nrows = int(nrows)
    ncols = int(ncols)
    nsites = nrows * ncols
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive.")
    if not np.isfinite(j2) or float(j2) < 0.0:
        raise ValueError("j2 must be finite and nonnegative.")
    bond_dims = _validated_bond_dims(bond_dims)
    maximum_directional_passes = int(maximum_directional_passes)
    if maximum_directional_passes < 2 or maximum_directional_passes % 2:
        raise ValueError(
            "maximum_directional_passes must be a positive even integer."
        )
    gain_tolerance = float(gain_tolerance)
    if not np.isfinite(gain_tolerance) or gain_tolerance < 0.0:
        raise ValueError("gain_tolerance must be finite and nonnegative.")
    required_consecutive_cycles = int(required_consecutive_cycles)
    if required_consecutive_cycles < 1:
        raise ValueError("required_consecutive_cycles must be positive.")
    output = Path(output)
    snapshot = Path(snapshot)

    mpo, nearest, diagonals = _model(nrows, ncols, j2)
    protocol = {
        "model": "open spin-1/2 J1-J2 Heisenberg",
        "nrows": nrows,
        "ncols": ncols,
        "j1": 1.0,
        "j2": float(j2),
        "site_order": "row-wise snake",
        "symmetry": "none",
        "bond_dims": list(bond_dims),
        "optimizer": "two-site DMRG",
        "convergence_scope": "fresh normalized post-truncation expectations",
        "convergence_cycle": "left-to-right plus right-to-left",
        "convergence_statistic": (
            "maximum accepted variational energy gain per site over both "
            "directions"
        ),
        "monotonic_best_state": True,
        "gain_tolerance": gain_tolerance,
        "gain_tolerance_units": "energy_per_site",
        "required_consecutive_cycles": required_consecutive_cycles,
        "davidson_tol": float(davidson_tol),
        "davidson_max_iter": int(davidson_max_iter),
        "performance": str(performance),
        "initialization": "reproducible random MPS",
        "seed": int(seed),
        "noise": 0.0,
        "recenter_final": False,
    }
    fingerprint = _fingerprint(protocol)

    if resume and (output.exists() or snapshot.exists()):
        if not (output.is_file() and snapshot.is_file()):
            raise RuntimeError(
                "safe resume requires both result.json and state.npz."
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("checkpoint protocol does not match this run.")
        state, stage_index = _load_checkpoint(
            snapshot,
            payload=payload,
            mpo=mpo,
        )
        if payload.get("result", {}).get("converged", False):
            return payload
        elapsed_before = float(payload.get("timing_seconds", {}).get("total", 0.0))
    else:
        state = _random_mps(nsites, bond_dims[0], seed)
        energy = _energy(state, mpo)
        stage_index = 0
        stage = _stage_record(
            bond_dims[0],
            energy,
            initialization=f"random seed {int(seed)}",
            pass_limit=maximum_directional_passes,
        )
        stage["energy_per_site"] = energy / nsites
        stage["state"] = _state_parameter_counts(state)
        payload = {
            "schema_version": 1,
            "status": "running",
            "model": {
                "nrows": nrows,
                "ncols": ncols,
                "nsites": nsites,
                "j1": 1.0,
                "j2": float(j2),
                "boundary": "open",
                "site_order": "row-wise snake",
                "j1_nearest_edges": len(nearest),
                "j2_diagonal_edges": len(diagonals),
            },
            "protocol": protocol,
            "protocol_fingerprint": fingerprint,
            "initial_state": {
                "kind": "reproducible random MPS",
                "seed": int(seed),
                "energy": energy,
                "energy_per_site": energy / nsites,
            },
            "stages": [stage],
            "results": {},
            "result": {},
            "timing_seconds": {"total": 0.0},
        }
        elapsed_before = 0.0
        _write_checkpoint_pair(
            output,
            snapshot,
            payload,
            state,
            stage_index=stage_index,
            energy=energy,
        )

    run_started = perf_counter()
    while True:
        stage = payload["stages"][stage_index]
        stage["directional_pass_limit"] = maximum_directional_passes
        if stage["converged"]:
            if stage_index == len(bond_dims) - 1:
                break
            stage_index += 1
            energy = _energy(state, mpo)
            if stage_index == len(payload["stages"]):
                payload["stages"].append(
                    _stage_record(
                        bond_dims[stage_index],
                        energy,
                        initialization=(
                            f"converged mps_d{bond_dims[stage_index - 1]}"
                        ),
                        pass_limit=maximum_directional_passes,
                    )
                )
                payload["stages"][-1]["energy_per_site"] = energy / nsites
                payload["stages"][-1]["state"] = _state_parameter_counts(state)
            _write_checkpoint_pair(
                output,
                snapshot,
                payload,
                state,
                stage_index=stage_index,
                energy=energy,
            )
            continue

        completed = int(stage["directional_passes_completed"])
        if completed >= maximum_directional_passes:
            stage["status"] = "maximum_passes"
            break
        stage["status"] = "running"
        energy_before = _energy(state, mpo)
        cycle_started = perf_counter()
        candidate_state, rows, directional, candidate_energy = _run_full_cycle(
            state,
            mpo,
            bond_dim=stage["bond_dim"],
            performance=performance,
            davidson_tol=davidson_tol,
            davidson_max_iter=davidson_max_iter,
        )
        seconds = perf_counter() - cycle_started
        changes = _monotonic_direction_record(
            energy_before,
            directional,
            nsites=nsites,
        )
        numerical_slack = 512.0 * np.finfo(float).eps * max(
            1.0, abs(energy_before)
        )
        if candidate_energy < energy_before - numerical_slack:
            state = candidate_state
            energy = float(candidate_energy)
        else:
            energy = float(energy_before)
        if abs(energy - changes["best_energy"]) > 5.0e-9:
            raise RuntimeError(
                "monotonic endpoint selection and convergence record disagree."
            )
        diagnostics = _cycle_diagnostics(rows)
        below = _below_gain_tolerance(changes, diagnostics, gain_tolerance)
        streak = int(stage["low_gain_streak"]) + 1 if below else 0
        record = {
            "cycle": len(stage["cycles"]) + 1,
            "directional_passes_before": completed,
            "directional_passes_completed": completed + 2,
            "energy_before": float(energy_before),
            "energy": float(energy),
            "energy_per_site": float(energy) / nsites,
            **changes,
            "below_tolerance": below,
            "low_gain_streak": streak,
            "seconds": float(seconds),
            "diagnostics": diagnostics,
        }
        stage["cycles"].append(record)
        stage["directional_passes_completed"] = completed + 2
        stage["low_gain_streak"] = streak
        stage["energy"] = float(energy)
        stage["energy_per_site"] = float(energy) / nsites
        stage["optimization_seconds"] = float(stage["optimization_seconds"]) + seconds
        stage["state"] = _state_parameter_counts(state)
        if streak >= required_consecutive_cycles:
            stage["converged"] = True
            stage["status"] = "complete"
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - run_started
        )
        _write_checkpoint_pair(
            output,
            snapshot,
            payload,
            state,
            stage_index=stage_index,
            energy=energy,
        )
        if verbose:
            print(
                f"J2={float(j2):.2f} D={stage['bond_dim']:d} "
                f"cycle={record['cycle']:d} E={energy:.12f} "
                "max gain/N="
                f"{changes['maximum_accepted_directional_gain_per_site']:.3e} "
                f"streak={streak}/{required_consecutive_cycles}",
                flush=True,
            )

    payload["timing_seconds"]["total"] = float(
        elapsed_before + perf_counter() - run_started
    )
    energy = _energy(state, mpo)
    _write_checkpoint_pair(
        output,
        snapshot,
        payload,
        state,
        stage_index=stage_index,
        energy=energy,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrows", type=int, default=6)
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
    )
    parser.add_argument(
        "--maximum-directional-passes",
        type=int,
        default=int(os.environ.get("LETTA_MAXIMUM_DIRECTIONAL_PASSES", "200")),
        help="Per-bond-dimension cap; must be even.",
    )
    parser.add_argument(
        "--gain-tolerance",
        type=float,
        default=float(os.environ.get("LETTA_GAIN_TOLERANCE", "1e-7")),
        help="Maximum accepted variational energy gain per site.",
    )
    parser.add_argument("--required-consecutive-cycles", type=int, default=2)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-10)
    parser.add_argument("--davidson-max-iter", type=int, default=100)
    parser.add_argument("--performance", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    payload = converge(
        nrows=args.nrows,
        ncols=args.ncols,
        j2=args.j2,
        bond_dims=args.bond_dims,
        maximum_directional_passes=args.maximum_directional_passes,
        gain_tolerance=args.gain_tolerance,
        required_consecutive_cycles=args.required_consecutive_cycles,
        davidson_tol=args.davidson_tol,
        davidson_max_iter=args.davidson_max_iter,
        performance=args.performance,
        seed=args.seed,
        output=args.output,
        snapshot=args.snapshot,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
    print(json.dumps(payload["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
