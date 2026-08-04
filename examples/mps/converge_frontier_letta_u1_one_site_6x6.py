#!/usr/bin/env python3
"""Converge 6x6 U(1) graph-LETTA with one-site sector sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.converge_frontier_letta_u1_two_site_6x6 import (
    CHECKPOINT_ENERGY_TOL,
    DEFAULT_SNAPSHOT,
    FRONTIER_BACKENDS,
    RESULTS,
    TIE_GRAPHS,
    _expand_to_parameter_target,
    _expansion_summary,
    _initial_state,
    _layout_from_record,
    _layout_record,
    _model,
    _normalized_charge_assignment,
    _parameter_diagnostics,
    _protocol_fingerprint,
    _save_snapshot,
    _snapshot_metadata,
    _state_from_snapshot,
    _state_protocol,
)
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json


DEFAULT_OUTPUT = RESULTS / "frontier_letta_u1_one_site_6x6.json"
DEFAULT_ONE_SITE_SNAPSHOT = DEFAULT_SNAPSHOT.with_name(
    "frontier_letta_u1_one_site_6x6.npz"
)


def _site_diagnostics(row):
    updates = tuple(row.get("updates", ()))
    solvers = sorted({update.solver for update in updates})
    return {
        "site_updates": len(updates),
        "accepted_site_updates": int(sum(update.accepted for update in updates)),
        "solver_failures": int(sum(not update.solver_converged for update in updates)),
        "solvers": solvers,
        "hamiltonian_matvecs": int(row.get("hamiltonian_matvecs", 0)),
        "metric_matvecs": int(row.get("metric_matvecs", 0)),
        "maximum_metric_rank": int(
            max((update.metric_rank for update in updates), default=0)
        ),
        "maximum_raw_dim": int(max((update.raw_dim for update in updates), default=0)),
        "maximum_residual_norm": float(
            max(
                (
                    update.residual_norm
                    for update in updates
                    if np.isfinite(update.residual_norm)
                ),
                default=0.0,
            )
        ),
        "maximum_coordinate_residual_norm": float(
            max(
                (
                    update.solver_coordinate_residual_norm
                    for update in updates
                    if np.isfinite(update.solver_coordinate_residual_norm)
                ),
                default=0.0,
            )
        ),
        "maximum_metric_identity_error": float(
            max(
                (
                    update.solver_metric_identity_error
                    for update in updates
                    if np.isfinite(update.solver_metric_identity_error)
                ),
                default=0.0,
            )
        ),
        "physical_blocks": int(sum(update.physical_blocks for update in updates)),
        "hamiltonian_blocks": int(
            sum(update.hamiltonian_blocks for update in updates)
        ),
        "stored_operator_elements": int(
            sum(update.stored_operator_elements for update in updates)
        ),
        "environment_cache": row.get("environment_cache", ""),
        "environment_checkpoint_interval": int(
            row.get("environment_checkpoint_interval", 0)
        ),
        "norm_contraction_is_exact": bool(
            row.get("norm_contraction_is_exact", False)
        ),
        "hamiltonian_action_is_hermitian": bool(
            row.get("hamiltonian_action_is_hermitian", False)
        ),
    }


def _pass_record(index, energy_before, row, seconds):
    energy = float(row["energy"])
    gain = float(energy_before - energy)
    return {
        "pass": int(index),
        "sweep": int(row["sweep"]),
        "direction": "left_to_right" if int(row["sweep"]) % 2 == 0 else "right_to_left",
        "energy_before": float(energy_before),
        "energy": energy,
        "energy_per_site": energy / 36,
        "energy_gain": gain,
        "energy_gain_per_site": gain / 36,
        "delta_energy": float(row.get("delta_energy", abs(gain))),
        "seconds": float(seconds),
        "diagnostics": _site_diagnostics(row),
    }


def converge(
    *,
    nrows=6,
    ncols=6,
    j2=0.5,
    tie_graph="all-j1",
    bond_dim=4,
    mps_sweeps=12,
    mps_tolerance=1.0e-10,
    maximum_passes=160,
    stopping_gain_per_site=1.0e-7,
    required_consecutive_passes=2,
    tie_noise=1.0e-1,
    seed=23,
    charge_assignment="physical",
    parameter_target=None,
    parameter_maximum_bond_dim=8,
    parameter_expansion_scale=1.0e-3,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_ONE_SITE_SNAPSHOT,
    resume=True,
    frontier_backend="identity_block",
):
    output = Path(output)
    snapshot = Path(snapshot)
    tie_graph = str(tie_graph).lower().replace("_", "-")
    if tie_graph not in TIE_GRAPHS:
        raise ValueError(f"tie_graph must be one of {TIE_GRAPHS}.")
    frontier_backend = str(frontier_backend).lower().replace("-", "_")
    if frontier_backend not in {"identity_block", "renormalized"}:
        raise ValueError(
            "frontier_backend must be 'identity_block' or 'renormalized'."
        )
    maximum_passes = int(maximum_passes)
    required_consecutive_passes = int(required_consecutive_passes)
    if maximum_passes < 0:
        raise ValueError("maximum_passes must be nonnegative.")
    if required_consecutive_passes < 1:
        raise ValueError("required_consecutive_passes must be positive.")
    stopping_gain_per_site = float(stopping_gain_per_site)
    if not np.isfinite(stopping_gain_per_site) or stopping_gain_per_site < 0.0:
        raise ValueError("stopping_gain_per_site must be finite and nonnegative.")
    charge_assignment = _normalized_charge_assignment(charge_assignment)
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
        "tie_graph_mode": tie_graph,
        "bond_dim": int(bond_dim),
        "warm_start": "U1 two-site DMRG",
        "mps_sweeps": int(mps_sweeps),
        "mps_tolerance": float(mps_tolerance),
        "letta_optimizer": "one-site U1 sector block-sparse sweeps",
        "frontier_backend": frontier_backend,
        "one_site_solver": "block_sparse",
        "one_site_frontier_gauge": "sector_probability",
        "one_site_metric_tol": 1.0e-12,
        "one_site_eig_tol": 1.0e-10,
        "maxiter": 600,
        "max_subspace": 48,
        "environment_cache": "checkpointed",
        "stopping_gain_per_site": stopping_gain_per_site,
        "required_consecutive_passes": required_consecutive_passes,
        "parameter_target": (
            "fixed_layout" if parameter_target is None else int(parameter_target)
        ),
        "parameter_maximum_bond_dim": int(parameter_maximum_bond_dim),
        "parameter_expansion_scale": float(parameter_expansion_scale),
        "tie_noise": float(tie_noise),
        "seed": int(seed),
    }
    fingerprint = _protocol_fingerprint(protocol)

    if resume and output.is_file() and snapshot.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _snapshot_metadata(snapshot)
        if payload["protocol_fingerprint"] != metadata["protocol_fingerprint"]:
            raise RuntimeError("snapshot and JSON protocol fingerprints disagree")
        previous_protocol = payload.get("protocol", {})
        if _state_protocol(previous_protocol) != _state_protocol(protocol):
            raise RuntimeError(
                "checkpoint state definition does not match requested settings"
            )
        if payload["protocol_fingerprint"] != fingerprint:
            raise RuntimeError("checkpoint protocol does not match requested run")
        if metadata["completed_cycles"] != len(payload.get("passes", [])):
            raise RuntimeError("snapshot pass count is inconsistent with JSON")
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
        passes = list(payload.get("passes", []))
        low_gain_streak = int(metadata["low_gain_streak"])
        elapsed_before = float(payload.get("timing_seconds", {}).get("total", 0.0))
        if payload.get("result", {}).get("converged", False):
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
            charge_assignment=charge_assignment,
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
        setup_seconds = float(perf_counter() - setup_start)
        energy = float(state.expectation())
        passes = []
        low_gain_streak = 0
        elapsed_before = setup_seconds
        hamiltonian, _parents, nearest, _tie_edges = _model(
            nrows, ncols, j2, tie_graph
        )
        _ = hamiltonian
        payload = {
            "status": "running",
            "model": {
                "nrows": int(nrows),
                "ncols": int(ncols),
                "nsites": int(nrows) * int(ncols),
                "j1": 1.0,
                "j2": float(j2),
                "boundary": "open",
                "tie_graph_mode": tie_graph,
                "j1_edges": len(nearest),
                "j2_diagonal_edges": 2 * (int(nrows) - 1) * (int(ncols) - 1),
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
                "letta_lift_energy": energy,
                "letta_lift_energy_per_site": energy / len(state.dims),
            },
            "source": {
                **_parameter_diagnostics(state),
                "bond_dims": list(state.bond_dims),
                "peak_frontier_elements": int(state.peak_frontier_elements),
            },
            "passes": passes,
            "result": {},
            "timing_seconds": {"setup": setup_seconds, "total": elapsed_before},
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
    stop_reason = "maximum passes reached"
    for pass_index in range(len(passes), maximum_passes):
        energy_before = float(state.expectation())
        started = perf_counter()
        state.run(
            nsweeps=1,
            sweep_offset=pass_index,
            tol=0.0,
            solver="block_sparse",
            frontier_canonicalization=True,
            frontier_gauge_weighting="probability",
            eig_tol=1.0e-10,
            maxiter=600,
            max_subspace=48,
            verbose=True,
        )
        if len(state.history) != 1:
            raise RuntimeError("one-site pass did not produce one history row")
        record = _pass_record(
            pass_index + 1,
            energy_before,
            state.history[0],
            perf_counter() - started,
        )
        gain_tolerance = 512.0 * np.finfo(float).eps * max(
            1.0,
            abs(energy_before),
        )
        below_threshold = bool(
            record["diagnostics"]["solver_failures"] == 0
            and record["energy_gain"] >= -gain_tolerance
            and max(record["energy_gain"], 0.0) / len(state.dims)
            < stopping_gain_per_site
        )
        low_gain_streak = low_gain_streak + 1 if below_threshold else 0
        record["low_gain_streak"] = int(low_gain_streak)
        passes.append(record)
        energy = float(record["energy"])
        payload["passes"] = passes
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - run_start
        )
        payload["result"] = {
            "converged": bool(low_gain_streak >= required_consecutive_passes),
            "stop_reason": stop_reason,
            "passes_completed": len(passes),
            "energy": energy,
            "energy_per_site": energy / len(state.dims),
            "energy_gain_from_warm_start": float(
                payload["warm_start"]["letta_lift_energy"] - energy
            ),
            "energy_vs_mps_warm_start": float(
                energy - payload["warm_start"]["mps"]["energy"]
            ),
            **_parameter_diagnostics(state),
            "bond_dims": list(state.bond_dims),
            "snapshot": str(snapshot),
            "optimization_residual_verified": bool(
                record["diagnostics"]["solver_failures"] == 0
            ),
        }
        _write_json(output, payload)
        _save_snapshot(
            snapshot,
            state,
            cycle=len(passes),
            low_gain_streak=low_gain_streak,
            protocol_fingerprint=fingerprint,
        )
        if payload["result"]["converged"]:
            stop_reason = "one-site small gain threshold reached"
            payload["result"]["stop_reason"] = stop_reason
            break

    payload["status"] = (
        "complete"
        if payload.get("result", {}).get("converged")
        else "capped"
        if len(passes) >= maximum_passes
        else "running"
    )
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--tie-graph", choices=TIE_GRAPHS, default="all-j1")
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=12)
    parser.add_argument("--mps-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--maximum-passes", type=int, default=160)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--tie-noise", type=float, default=1.0e-1)
    parser.add_argument("--seed", type=int, default=23)
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
    parser.add_argument(
        "--frontier-backend",
        choices=FRONTIER_BACKENDS,
        default="identity-block",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_ONE_SITE_SNAPSHOT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = converge(
        j2=args.j2,
        tie_graph=args.tie_graph,
        bond_dim=args.bond_dim,
        mps_sweeps=args.mps_sweeps,
        mps_tolerance=args.mps_tolerance,
        maximum_passes=args.maximum_passes,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_consecutive_passes=args.required_consecutive_passes,
        tie_noise=args.tie_noise,
        seed=args.seed,
        charge_assignment=args.charge_assignment,
        parameter_target=args.parameter_target,
        parameter_maximum_bond_dim=args.parameter_maximum_bond_dim,
        parameter_expansion_scale=args.parameter_expansion_scale,
        frontier_backend=args.frontier_backend,
        output=args.output,
        snapshot=args.snapshot,
        resume=not args.no_resume,
    )
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
