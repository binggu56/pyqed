#!/usr/bin/env python3
"""Checkpointed convergence of the integrated 6x6 graph-LETTA state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = RESULTS / "frontier_letta_variable_bond_adaptive_tie_6x6.json"
DEFAULT_SOURCE_STATE = RESULTS / "frontier_letta_variable_bond_adaptive_tie_6x6.npz"
DEFAULT_CONTROL = (
    RESULTS / "frontier_letta_bond_expansion_6x6_continued_30passes.json"
)
DEFAULT_OUTPUT = (
    RESULTS
    / "frontier_letta_variable_bond_adaptive_tie_6x6_parity_corrected.json"
)
DEFAULT_OUTPUT_STATE = (
    RESULTS
    / "frontier_letta_variable_bond_adaptive_tie_6x6_parity_corrected.npz"
)


def _hamiltonian_and_graph(model, tied_edge):
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    return (
        heisenberg_local_hamiltonian(36, weighted),
        parent_sets_from_edges(36, nearest + (tuple(tied_edge),)),
    )


def _load_state(result_path, state_path, *, resumed=False):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    model = payload["model"]
    tied_edge = (
        payload["protocol"]["selected_tie"]
        if not resumed
        else payload["protocol"]["tied_edge"]
    )
    hamiltonian, parent_sets = _hamiltonian_and_graph(model, tied_edge)
    with np.load(state_path, allow_pickle=False) as archive:
        bond_dims = tuple(int(value) for value in archive["virtual_bond_dims"])
        tensors = [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(36)
        ]
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dims=bond_dims,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    recorded = (
        float(payload["relaxed"]["energy"])
        if not resumed
        else float(payload["result"]["energy"])
    )
    measured = float(state.expectation())
    if abs(measured - recorded) > 2.0e-8:
        raise RuntimeError(
            "checkpoint energy does not match its JSON record: "
            f"{measured:.16g} versus {recorded:.16g}."
        )
    return state, payload


def _save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    temporary.replace(path)


def _pass_record(index, previous_energy, row):
    energy = float(row["energy"])
    absolute_sweep = int(row["sweep"])
    updates = row["updates"]
    failures = [update for update in updates if not update.solver_converged]
    identity_errors = [
        update.solver_metric_identity_error
        for update in updates
        if update.solver_metric_is_identity
    ]
    return {
        "pass": int(index),
        "absolute_sweep": absolute_sweep,
        "direction": (
            "left_to_right" if absolute_sweep % 2 == 0 else "right_to_left"
        ),
        "energy": energy,
        "energy_per_site": energy / 36,
        "energy_gain": float(previous_energy - energy),
        "energy_gain_per_site": float((previous_energy - energy) / 36),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "site_updates": len(updates),
        "solver_failures": len(failures),
        "failure_sites": [int(update.site) for update in failures],
        "identity_metric_sites": int(
            sum(update.solver_metric_is_identity for update in updates)
        ),
        "maximum_identity_metric_error": float(max(identity_errors, default=0.0)),
    }


def _control_comparison(control_path, directional_passes, cycles, final_energy):
    control = json.loads(Path(control_path).read_text(encoding="utf-8"))
    control_additional_passes = int(control["result"]["passes_completed"])
    control_source_passes = int(
        control.get("source", {}).get("source_directional_passes", 0)
    )
    control_rows = list(control.get("directional_passes", ()))
    control_points = [
        {
            "total_passes": control_source_passes,
            "energy": float(control["source"]["energy"]),
            "kind": "control source",
        }
    ]
    control_points.extend(
        {
            "total_passes": control_source_passes + int(row["pass"]),
            "energy": float(row["energy"]),
            "kind": "control continuation",
        }
        for row in control_rows
    )
    integrated_prior_passes = 1
    integrated_total_passes = integrated_prior_passes + len(directional_passes)
    matched = next(
        (
            point
            for point in control_points
            if point["total_passes"] == integrated_total_passes
        ),
        None,
    )
    nearest = min(
        control_points,
        key=lambda point: abs(point["total_passes"] - integrated_total_passes),
    )
    milestone_total_passes = control_source_passes
    milestone_integrated_pass = milestone_total_passes - integrated_prior_passes
    milestone_row = next(
        (
            row
            for row in directional_passes
            if int(row["pass"]) == milestone_integrated_pass
        ),
        None,
    )
    return {
        "control_result": str(Path(control_path).resolve()),
        "control_parameters": int(control["result"]["parameters"]),
        "control_source_passes_from_common_variable_bond_state": (
            control_source_passes
        ),
        "control_additional_passes": control_additional_passes,
        "control_final_total_passes": (
            control_source_passes + control_additional_passes
        ),
        "control_final_energy": float(control["result"]["energy"]),
        "integrated_source_prior_passes_from_common_state": (
            integrated_prior_passes
        ),
        "matched_total_post_expansion_pass_count": (
            None
            if matched is None
            else {
                "total_passes": integrated_total_passes,
                "integrated_additional_passes": len(directional_passes),
                "integrated_energy": float(final_energy),
                "control_energy": float(matched["energy"]),
                "control_point": matched["kind"],
                "integrated_below_control": float(
                    matched["energy"] - final_energy
                ),
                "integrated_below_control_per_site": float(
                    (matched["energy"] - final_energy) / 36
                ),
            }
        ),
        "common_control_source_milestone": (
            None
            if milestone_row is None
            else {
                "total_passes": milestone_total_passes,
                "integrated_additional_passes": milestone_integrated_pass,
                "integrated_energy": float(milestone_row["energy"]),
                "control_energy": float(control["source"]["energy"]),
                "integrated_below_control": float(
                    control["source"]["energy"] - milestone_row["energy"]
                ),
                "integrated_below_control_per_site": float(
                    (
                        control["source"]["energy"]
                        - milestone_row["energy"]
                    )
                    / 36
                ),
            }
        ),
        "nearest_available_control_to_final": {
            "integrated_total_passes": integrated_total_passes,
            "integrated_additional_passes": len(directional_passes),
            "integrated_cycles": len(cycles),
            "control_total_passes": int(nearest["total_passes"]),
            "pass_count_difference": abs(
                integrated_total_passes - nearest["total_passes"]
            ),
            "integrated_energy": float(final_energy),
            "control_energy": float(nearest["energy"]),
            "integrated_below_control": float(nearest["energy"] - final_energy),
            "integrated_below_control_per_site": float(
                (nearest["energy"] - final_energy) / 36
            ),
        },
    }


def continue_run(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_state=DEFAULT_SOURCE_STATE,
    control_result=DEFAULT_CONTROL,
    output=DEFAULT_OUTPUT,
    output_state=DEFAULT_OUTPUT_STATE,
    maximum_passes=20,
    stopping_cycle_gain_per_site=1.0e-7,
    resume=True,
):
    output = Path(output)
    output_state = Path(output_state)
    maximum_passes = int(maximum_passes)
    if maximum_passes < 2:
        raise ValueError("maximum_passes must be at least two.")
    if maximum_passes % 2:
        raise ValueError("maximum_passes must be even to checkpoint full cycles.")
    stopping_cycle_gain_per_site = float(stopping_cycle_gain_per_site)
    if not np.isfinite(stopping_cycle_gain_per_site) or stopping_cycle_gain_per_site < 0:
        raise ValueError("stopping_cycle_gain_per_site must be finite and nonnegative.")

    can_resume = bool(resume and output.is_file() and output_state.is_file())
    if can_resume:
        state, payload = _load_state(output, output_state, resumed=True)
        if bool(payload.get("result", {}).get("converged", False)):
            return payload
        source_payload = json.loads(Path(source_result).read_text(encoding="utf-8"))
        initial_energy = float(payload["source"]["energy"])
        passes = list(payload.get("directional_passes", ()))
        cycles = list(payload.get("cycles", ()))
        prior_seconds = float(payload.get("timing_seconds", {}).get("total", 0.0))
    else:
        state, source_payload = _load_state(source_result, source_state)
        initial_energy = float(state.expectation())
        passes = []
        cycles = []
        prior_seconds = 0.0

    tied_edge = tuple(source_payload["protocol"]["selected_tie"])
    run_start = perf_counter()
    stop_reason = "maximum directional passes reached"
    converged = False
    while len(passes) < maximum_passes:
        cycle_index = len(cycles) + 1
        energy_before_cycle = float(state.expectation())
        cycle_start = perf_counter()
        sweep_offset = 1 + len(passes)
        state.run(
            nsweeps=2,
            sweep_offset=sweep_offset,
            tol=0.0,
            solver="whitened",
            metric_tol=1.0e-12,
            eig_tol=1.0e-10,
            maxiter=1600,
            max_subspace=96,
            gauge="frontier",
            gauge_weight="uniform",
            verbose=True,
        )
        previous_energy = energy_before_cycle
        cycle_records = []
        for row in state.history:
            pass_index = len(passes) + 1
            record = _pass_record(
                pass_index,
                previous_energy,
                row,
            )
            passes.append(record)
            cycle_records.append(record)
            previous_energy = float(record["energy"])
        fresh_energy = float(state.expectation())
        cycle_gain = float(energy_before_cycle - fresh_energy)
        cycle_failures = int(sum(row["solver_failures"] for row in cycle_records))
        identity_error = float(
            max(
                (row["maximum_identity_metric_error"] for row in cycle_records),
                default=0.0,
            )
        )
        criterion_passed = bool(
            cycle_gain / 36 < stopping_cycle_gain_per_site and cycle_failures == 0
        )
        cycles.append(
            {
                "cycle": cycle_index,
                "passes": [int(row["pass"]) for row in cycle_records],
                "energy_before": energy_before_cycle,
                "energy": fresh_energy,
                "energy_per_site": fresh_energy / 36,
                "energy_gain": cycle_gain,
                "energy_gain_per_site": cycle_gain / 36,
                "solver_failures": cycle_failures,
                "maximum_identity_metric_error": identity_error,
                "criterion_passed": criterion_passed,
                "seconds": float(perf_counter() - cycle_start),
            }
        )
        converged = criterion_passed
        if converged:
            stop_reason = (
                "full-cycle gain/site below "
                f"{stopping_cycle_gain_per_site:.3e} with no solver failures"
            )

        elapsed = prior_seconds + perf_counter() - run_start
        all_failures = int(sum(row["solver_failures"] for row in passes))
        payload = {
            "status": "converged" if converged else "running",
            "model": source_payload["model"],
            "protocol": {
                "source_result": str(Path(source_result).resolve()),
                "source_state": str(Path(source_state).resolve()),
                "tied_edge": list(tied_edge),
                "bond_dims": list(state.bond_dims),
                "solver": "whitened exact local S=I frame",
                "gauge": "frontier",
                "maximum_additional_directional_passes": maximum_passes,
                "checkpoint_interval": "one full left-right cycle (two passes)",
                "source_absolute_directional_sweeps": 1,
                "resume_sweep_offset_rule": "1 + completed additional passes",
                "first_continuation_direction": "right_to_left",
                "stopping_cycle_gain_per_site": stopping_cycle_gain_per_site,
                "resume_supported": True,
            },
            "source": {
                "energy": initial_energy,
                "energy_per_site": initial_energy / 36,
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
                "prior_passes_from_common_variable_bond_state": 1,
            },
            "directional_passes": passes,
            "cycles": cycles,
            "result": {
                "converged": converged,
                "stop_reason": stop_reason,
                "passes_completed": len(passes),
                "cycles_completed": len(cycles),
                "energy": fresh_energy,
                "energy_per_site": fresh_energy / 36,
                "continuation_energy_gain": initial_energy - fresh_energy,
                "continuation_energy_gain_per_site": (
                    initial_energy - fresh_energy
                )
                / 36,
                "last_cycle_gain_per_site": cycle_gain / 36,
                "solver_failures": all_failures,
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
                "snapshot": str(output_state.resolve()),
            },
            "control_comparison": _control_comparison(
                control_result, passes, cycles, fresh_energy
            ),
            "timing_seconds": {
                "cycles": [float(row["seconds"]) for row in cycles],
                "total": float(elapsed),
            },
        }
        _save_state(output_state, state)
        _write_json(output, payload)
        print(
            f"cycle={cycle_index:2d} E={fresh_energy:.12f} "
            f"gain/site={cycle_gain/36:.3e} failures={cycle_failures}",
            flush=True,
        )
        if converged:
            break

    if not converged:
        stop_reason = "maximum directional passes reached"
        payload["status"] = "pass_cap_reached"
        payload["result"]["stop_reason"] = stop_reason
        payload["control_comparison"] = _control_comparison(
            control_result,
            passes,
            cycles,
            float(payload["result"]["energy"]),
        )
        payload["timing_seconds"]["total"] = float(
            prior_seconds + perf_counter() - run_start
        )
        _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE_STATE)
    parser.add_argument("--control-result", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-state", type=Path, default=DEFAULT_OUTPUT_STATE)
    parser.add_argument("--maximum-passes", type=int, default=20)
    parser.add_argument("--stopping-cycle-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="restart from the integrated source instead of resuming output",
    )
    args = parser.parse_args()
    result = continue_run(
        source_result=args.source_result,
        source_state=args.source_state,
        control_result=args.control_result,
        output=args.output,
        output_state=args.output_state,
        maximum_passes=args.maximum_passes,
        stopping_cycle_gain_per_site=args.stopping_cycle_gain_per_site,
        resume=not args.fresh,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
