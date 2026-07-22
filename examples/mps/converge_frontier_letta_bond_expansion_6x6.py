#!/usr/bin/env python3
"""Safely resume and converge the variable-bond 6x6 LETTA checkpoint."""

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
DEFAULT_SOURCE_RESULT = RESULTS / "frontier_letta_bond_expansion_6x6_continued.json"
DEFAULT_SOURCE_SNAPSHOT = RESULTS / "frontier_letta_bond_expansion_6x6_continued.npz"
DEFAULT_MPS_REFERENCE = RESULTS / "frontier_letta_block_sparse_6x6_mps_references.json"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_bond_expansion_6x6_continued_30passes.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_bond_expansion_6x6_continued_30passes.npz"


def _state_from_snapshot(model, snapshot_path):
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
    return FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(36, nearest),
        bond_dims=bond_dims,
        tensors=tensors,
        frontier_backend="identity_block",
    )


def _save_snapshot(path, state):
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


def _result_record(
    state,
    *,
    initial_energy,
    baseline_energy,
    mps_d32_energy,
    passes,
    stop_reason,
    snapshot,
    stopping_gain_per_site,
):
    energy = float(state.expectation())
    return {
        "converged": bool(
            passes
            and passes[-1]["energy_gain_per_site"] < stopping_gain_per_site
        ),
        "stop_reason": stop_reason,
        "passes_completed": len(passes),
        "energy": energy,
        "energy_per_site": energy / 36,
        "continuation_energy_gain": float(initial_energy - energy),
        "continuation_energy_gain_per_site": float((initial_energy - energy) / 36),
        "fresh_energy_gain_from_d4_baseline": float(baseline_energy - energy),
        "fresh_energy_gain_per_site_from_d4_baseline": float(
            (baseline_energy - energy) / 36
        ),
        "dense_mps_d32_energy": mps_d32_energy,
        "energy_above_dense_mps_d32": float(energy - mps_d32_energy),
        "energy_above_dense_mps_d32_per_site": float(
            (energy - mps_d32_energy) / 36
        ),
        "solver_failures": int(
            sum(record["solver_failures"] for record in passes)
        ),
        "parameters": int(state.nparameters),
        "bond_dims": list(state.bond_dims),
        "snapshot": str(Path(snapshot).resolve()),
    }


def converge(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    mps_reference=DEFAULT_MPS_REFERENCE,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    maximum_additional_passes=20,
    stopping_gain_per_site=1.0e-7,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    output = Path(output)
    snapshot = Path(snapshot)
    source = json.loads(source_result.read_text(encoding="utf-8"))
    model = source["model"]
    source_energy = float(source["result"]["energy"])
    baseline_energy = float(source["source"]["baseline_d4_energy"])
    source_passes = int(source["result"]["passes_completed"])
    reference = json.loads(Path(mps_reference).read_text(encoding="utf-8"))
    mps_d32_energy = float(reference["results"]["mps_d32"]["energy"])

    resumed = bool(resume and output.is_file() and snapshot.is_file())
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        state = _state_from_snapshot(model, snapshot)
        expected = float(payload["result"]["energy"])
        passes = list(payload["directional_passes"])
        elapsed_before = float(payload["timing_seconds"]["total"])
        initial_energy = float(payload["source"]["energy"])
        if abs(state.expectation() - expected) > 2.0e-8:
            raise RuntimeError("resume snapshot and JSON energies are inconsistent.")
        if payload["result"]["converged"] or len(passes) >= int(
            maximum_additional_passes
        ):
            return payload
    else:
        state = _state_from_snapshot(model, source_snapshot)
        measured = float(state.expectation())
        if abs(measured - source_energy) > 2.0e-8:
            raise RuntimeError("source snapshot and JSON energies are inconsistent.")
        passes = []
        elapsed_before = 0.0
        initial_energy = measured
        payload = {
            "status": "running",
            "model": model,
            "protocol": {
                "source_result": str(source_result.resolve()),
                "source_snapshot": str(source_snapshot.resolve()),
                "symmetry": "none",
                "solver": "whitened exact local S=I frame",
                "frontier_canonicalization": True,
                "frontier_gauge_weighting": "uniform",
                "maximum_additional_directional_passes": int(
                    maximum_additional_passes
                ),
                "stopping_rule": "first directional energy gain/site < threshold",
                "stopping_gain_per_site": float(stopping_gain_per_site),
                "safe_resume": True,
            },
            "source": {
                "energy": initial_energy,
                "energy_per_site": initial_energy / 36,
                "baseline_d4_energy": baseline_energy,
                "source_directional_passes": source_passes,
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
            },
            "directional_passes": passes,
            "result": {},
            "timing_seconds": {"total": 0.0},
        }

    run_start = perf_counter()
    stop_reason = "maximum additional directional passes reached"
    for pass_index in range(len(passes), int(maximum_additional_passes)):
        energy_before = float(state.expectation())
        pass_start = perf_counter()
        # sweep_offset retains L->R/R->L parity even though every pass is
        # checkpointed through a separate run call.
        state.run(
            nsweeps=1,
            sweep_offset=source_passes + pass_index,
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
        pass_seconds = perf_counter() - pass_start
        row = state.history[0]
        energy = float(state.expectation())
        gain = float(energy_before - energy)
        updates = row["updates"]
        failures = [update for update in updates if not update.solver_converged]
        direction = "left_to_right" if row["sweep"] % 2 == 0 else "right_to_left"
        passes.append(
            {
                "pass": pass_index + 1,
                "absolute_sweep": int(row["sweep"]),
                "direction": direction,
                "energy": energy,
                "energy_per_site": energy / 36,
                "energy_gain": gain,
                "energy_gain_per_site": gain / 36,
                "seconds": float(pass_seconds),
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
        if gain / 36 < float(stopping_gain_per_site) and not failures:
            stop_reason = (
                f"directional energy gain/site {gain/36:.6e} is below "
                f"{stopping_gain_per_site:.6e}"
            )

        payload["directional_passes"] = passes
        payload["result"] = _result_record(
            state,
            initial_energy=initial_energy,
            baseline_energy=baseline_energy,
            mps_d32_energy=mps_d32_energy,
            passes=passes,
            stop_reason=stop_reason,
            snapshot=snapshot,
            stopping_gain_per_site=float(stopping_gain_per_site),
        )
        payload["status"] = (
            "complete"
            if payload["result"]["converged"]
            or len(passes) >= int(maximum_additional_passes)
            else "running"
        )
        payload["timing_seconds"] = {
            "total": float(elapsed_before + perf_counter() - run_start)
        }
        _save_snapshot(snapshot, state)
        _write_json(output, payload)
        if payload["result"]["converged"]:
            break

    final = payload["result"]
    print(
        f"final E={final['energy']:.12f}, passes={final['passes_completed']}, "
        f"failures={final['solver_failures']}, {final['stop_reason']}",
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
    parser.add_argument("--maximum-additional-passes", type=int, default=20)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    converge(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        mps_reference=args.mps_reference,
        output=args.output,
        snapshot=args.snapshot,
        maximum_additional_passes=args.maximum_additional_passes,
        stopping_gain_per_site=args.stopping_gain_per_site,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
