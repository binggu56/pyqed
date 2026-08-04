#!/usr/bin/env python3
"""Benchmark packed pair-action backends on the converged 6x6 U(1) state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.converge_frontier_letta_u1_two_site_6x6 import (
    _layout_from_record,
    _pair_diagnostics,
    _state_from_snapshot,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE = (
    RESULTS
    / "frontier_letta_u1_j2_0p5_allj1_D4_matrix_free_v3_adaptive_6x6.json"
)
DEFAULT_SNAPSHOT = DEFAULT_SOURCE.with_suffix(".npz")
DEFAULT_OUTPUT = RESULTS / "frontier_letta_u1_support_actions_6x6.json"
ACTION_BACKENDS = ("full", "fused", "prepared", "auto")


def benchmark(
    *,
    source=DEFAULT_SOURCE,
    snapshot=DEFAULT_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    action_backends=ACTION_BACKENDS,
    prepared_min_action_calls=7,
    nsweeps=1,
    merged_dense_fallback_dim=0,
):
    source = Path(source)
    snapshot = Path(snapshot)
    nsweeps = int(nsweeps)
    if nsweeps < 1:
        raise ValueError("nsweeps must be positive.")
    merged_dense_fallback_dim = int(merged_dense_fallback_dim)
    if merged_dense_fallback_dim < 0:
        raise ValueError("merged_dense_fallback_dim must be nonnegative.")
    payload = json.loads(source.read_text(encoding="utf-8"))
    layout = _layout_from_record(payload["layout"])
    records = []
    for action_backend in action_backends:
        state = _state_from_snapshot(
            snapshot,
            model=payload["protocol"],
            layout=layout,
            frontier_backend="renormalized",
        )
        energy_before = float(state.expectation())
        start = perf_counter()
        state.run_two_site(
            nsweeps=nsweeps,
            sweep_offset=0,
            tol=0.0,
            solver="verified",
            pair_operator_backend="matrix_free",
            matrix_free_action_backend=action_backend,
            matrix_free_prepared_min_action_calls=prepared_min_action_calls,
            pair_dense_max_bytes=64 * 1024**2,
            merged_dense_fallback_dim=merged_dense_fallback_dim,
            outer_cycles=1,
            eig_tol=1.0e-10,
            matrix_free_eig_tol=1.0e-9,
            maxiter=600,
            max_subspace=48,
            matrix_free_batch_size=2,
            matrix_free_recycle_vectors=6,
            matrix_free_preconditioner="adaptive",
            verify_pair_roots=True,
            verify_pair_energies=False,
        )
        seconds = float(perf_counter() - start)
        updates = tuple(
            update
            for row in state.history
            for update in row["updates"]
        )
        diagnostics = _pair_diagnostics(updates)
        energy = float(state.expectation())
        records.append(
            {
                "requested_action_backend": action_backend,
                "selected_action_backends": diagnostics[
                    "action_backend_counts"
                ],
                "seconds": seconds,
                "operator_assembly_seconds": diagnostics[
                    "operator_assembly_seconds"
                ],
                "merged_solve_seconds": diagnostics[
                    "merged_solve_seconds"
                ],
                "split_seconds": diagnostics["split_seconds"],
                "hamiltonian_action_calls": diagnostics[
                    "hamiltonian_action_calls"
                ],
                "hamiltonian_vector_products": diagnostics[
                    "hamiltonian_vector_products"
                ],
                "hamiltonian_batch_calls": diagnostics[
                    "hamiltonian_batch_calls"
                ],
                "maximum_operator_peak_bytes": diagnostics[
                    "maximum_operator_peak_bytes"
                ],
                "maximum_operator_stored_bytes": diagnostics[
                    "maximum_operator_stored_bytes"
                ],
                "energy_before": energy_before,
                "energy": energy,
                "energy_gain": float(energy_before - energy),
                "verified_roots": diagnostics["verified_merged_roots"],
                "certified_roots": diagnostics["lowest_root_certified"],
                "dense_fallbacks": diagnostics["dense_fallbacks"],
                "pair_updates": diagnostics["pair_updates"],
            }
        )
        print(
            f"{action_backend:>8s}: {seconds:.3f} s, "
            f"E={energy:.12f}, actions="
            f"{diagnostics['hamiltonian_action_calls']}",
            flush=True,
        )
    full_seconds = next(
        (
            record["seconds"]
            for record in records
            if record["requested_action_backend"] == "full"
        ),
        records[0]["seconds"],
    )
    for record in records:
        record["speedup_vs_full"] = float(full_seconds / record["seconds"])
    result = {
        "model": payload["protocol"]["model"],
        "j2_over_j1": payload["protocol"]["j2"],
        "bond_dim": payload["protocol"]["bond_dim"],
        "source": str(source),
        "snapshot": str(snapshot),
        "direction": (
            "left-to-right"
            if nsweeps == 1
            else f"{nsweeps} alternating directional sweeps"
        ),
        "directional_sweeps": nsweeps,
        "matrix_free_batch_size": 2,
        "prepared_min_action_calls": int(prepared_min_action_calls),
        "merged_dense_fallback_dim": merged_dense_fallback_dim,
        "records": records,
    }
    _write_json(Path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--action-backends",
        nargs="+",
        choices=ACTION_BACKENDS,
        default=ACTION_BACKENDS,
    )
    parser.add_argument(
        "--prepared-min-action-calls",
        type=int,
        default=7,
    )
    parser.add_argument("--nsweeps", type=int, default=1)
    parser.add_argument(
        "--merged-dense-fallback-dim",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    result = benchmark(
        source=args.source,
        snapshot=args.snapshot,
        output=args.output,
        action_backends=tuple(args.action_backends),
        prepared_min_action_calls=args.prepared_min_action_calls,
        nsweeps=args.nsweeps,
        merged_dense_fallback_dim=args.merged_dense_fallback_dim,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
