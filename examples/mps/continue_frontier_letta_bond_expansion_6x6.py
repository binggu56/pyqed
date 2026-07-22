#!/usr/bin/env python3
"""Continue the selected variable-bond 6x6 LETTA snapshot."""

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
DEFAULT_SOURCE_RESULT = RESULTS / "frontier_letta_bond_expansion_6x6.json"
DEFAULT_SOURCE_SNAPSHOT = RESULTS / "frontier_letta_bond_expansion_6x6.npz"
DEFAULT_MPS_REFERENCE = RESULTS / "frontier_letta_block_sparse_6x6_mps_references.json"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_bond_expansion_6x6_continued.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_bond_expansion_6x6_continued.npz"


def _load_state(result_path, snapshot_path):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    model = payload["model"]
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(36, weighted_bonds)
    with np.load(snapshot_path, allow_pickle=False) as archive:
        bond_dims = tuple(int(value) for value in archive["virtual_bond_dims"])
        tensors = [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(36)
        ]
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(36, nearest),
        bond_dims=bond_dims,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    recorded = float(payload["relaxed"]["energy"])
    measured = float(state.expectation())
    if abs(measured - recorded) > 2.0e-8:
        raise RuntimeError(
            "expanded snapshot energy does not match its result record: "
            f"{measured:.16g} versus {recorded:.16g}."
        )
    return state, payload


def _save_snapshot(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )


def continue_run(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    mps_reference=DEFAULT_MPS_REFERENCE,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    maximum_passes=10,
    stopping_gain_per_site=1.0e-7,
    required_consecutive=2,
):
    state, source = _load_state(source_result, source_snapshot)
    reference = json.loads(Path(mps_reference).read_text(encoding="utf-8"))
    mps_d32_energy = float(reference["results"]["mps_d32"]["energy"])
    initial_energy = float(state.expectation())
    baseline_energy = float(source["baseline"]["energy"])
    source_passes = int(
        source.get("protocol", {}).get(
            "relaxation_passes",
            len(source.get("relaxed", {}).get("directional_pass_energies", ())),
        )
    )
    passes = []
    pair_timings = []
    low_gain_streak = 0
    stop_reason = "maximum directional passes reached"
    total_start = perf_counter()

    # Keep the absolute sweep parity explicit across every checkpointed call.
    for pair in range((int(maximum_passes) + 1) // 2):
        remaining = int(maximum_passes) - len(passes)
        pair_passes = min(2, remaining)
        sweep_offset = source_passes + len(passes)
        energy_before_pair = float(state.expectation())
        pair_start = perf_counter()
        state.run(
            nsweeps=pair_passes,
            sweep_offset=sweep_offset,
            tol=0.0,
            solver="whitened",
            metric_tol=1.0e-12,
            eig_tol=1.0e-10,
            maxiter=1600,
            max_subspace=96,
            frontier_canonicalization=True,
            frontier_gauge_weighting="uniform",
            verbose=True,
        )
        pair_timings.append(float(perf_counter() - pair_start))
        previous_energy = energy_before_pair
        for row in state.history:
            pass_index = len(passes) + 1
            absolute_sweep = int(row["sweep"])
            energy = float(row["energy"])
            gain = float(previous_energy - energy)
            gain_per_site = gain / 36
            updates = row["updates"]
            failures = [update for update in updates if not update.solver_converged]
            passes.append(
                {
                    "pass": pass_index,
                    "absolute_sweep": absolute_sweep,
                    "direction": (
                        "left_to_right"
                        if absolute_sweep % 2 == 0
                        else "right_to_left"
                    ),
                    "energy": energy,
                    "energy_per_site": energy / 36,
                    "energy_gain": gain,
                    "energy_gain_per_site": gain_per_site,
                    "accepted_updates": int(sum(update.accepted for update in updates)),
                    "site_updates": len(updates),
                    "solver_failures": len(failures),
                    "failure_sites": [int(update.site) for update in failures],
                    "identity_metric_sites": int(
                        sum(update.solver_metric_is_identity for update in updates)
                    ),
                    "maximum_identity_metric_error": float(
                        max(
                            (
                                update.solver_metric_identity_error
                                for update in updates
                                if update.solver_metric_is_identity
                            ),
                            default=0.0,
                        )
                    ),
                }
            )
            low_gain_streak = (
                low_gain_streak + 1
                if gain_per_site < float(stopping_gain_per_site)
                and len(failures) == 0
                else 0
            )
            previous_energy = energy
        _save_snapshot(snapshot, state)
        if low_gain_streak >= int(required_consecutive):
            stop_reason = (
                f"{required_consecutive} consecutive directional gains/site "
                f"below {stopping_gain_per_site:.3e}"
            )
            break
        if passes and passes[-1]["energy_gain_per_site"] <= 1.0e-13:
            stop_reason = "no meaningful variational progress"
            break

    final_energy = float(state.expectation())
    all_failures = int(sum(record["solver_failures"] for record in passes))
    payload = {
        "status": "complete",
        "model": source["model"],
        "protocol": {
            "source_result": str(Path(source_result).resolve()),
            "source_snapshot": str(Path(source_snapshot).resolve()),
            "symmetry": "none",
            "solver": "whitened exact local S=I frame",
            "frontier_canonicalization": True,
            "frontier_gauge_weighting": "uniform",
            "maximum_directional_passes": int(maximum_passes),
            "source_absolute_directional_sweeps": source_passes,
            "resume_sweep_offset_rule": "source passes + completed passes",
            "stopping_gain_per_site": float(stopping_gain_per_site),
            "required_consecutive_low_gain_passes": int(required_consecutive),
        },
        "source": {
            "energy": initial_energy,
            "energy_per_site": initial_energy / 36,
            "baseline_d4_energy": baseline_energy,
            "parameters": int(state.nparameters),
            "bond_dims": list(state.bond_dims),
        },
        "directional_passes": passes,
        "result": {
            "stop_reason": stop_reason,
            "passes_completed": len(passes),
            "energy": final_energy,
            "energy_per_site": final_energy / 36,
            "continuation_energy_gain": float(initial_energy - final_energy),
            "continuation_energy_gain_per_site": float(
                (initial_energy - final_energy) / 36
            ),
            "fresh_energy_gain_from_d4_baseline": float(
                baseline_energy - final_energy
            ),
            "fresh_energy_gain_per_site_from_d4_baseline": float(
                (baseline_energy - final_energy) / 36
            ),
            "dense_mps_d32_energy": mps_d32_energy,
            "energy_above_dense_mps_d32": float(final_energy - mps_d32_energy),
            "energy_above_dense_mps_d32_per_site": float(
                (final_energy - mps_d32_energy) / 36
            ),
            "solver_failures": all_failures,
            "parameters": int(state.nparameters),
            "bond_dims": list(state.bond_dims),
            "snapshot": str(Path(snapshot).resolve()),
        },
        "timing_seconds": {
            "pairs": pair_timings,
            "total": float(perf_counter() - total_start),
        },
    }
    _save_snapshot(snapshot, state)
    _write_json(output, payload)
    print(
        f"continued E={final_energy:.12f}, "
        f"gain={initial_energy-final_energy:.6e}, "
        f"passes={len(passes)}, failures={all_failures}",
        flush=True,
    )
    print(
        f"gap to dense MPS D32/site="
        f"{(final_energy-mps_d32_energy)/36:.6e}; {stop_reason}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument(
        "--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT
    )
    parser.add_argument("--mps-reference", type=Path, default=DEFAULT_MPS_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--maximum-passes", type=int, default=10)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--required-consecutive", type=int, default=2)
    args = parser.parse_args()
    continue_run(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        mps_reference=args.mps_reference,
        output=args.output,
        snapshot=args.snapshot,
        maximum_passes=args.maximum_passes,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_consecutive=args.required_consecutive,
    )


if __name__ == "__main__":
    main()
