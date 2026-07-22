#!/usr/bin/env python3
"""Replay and extend the representative graph-LETTA state at J2/J1 = 0.5.

The original scan files retain convergence histories but not variational
tensors.  This script deterministically replays only the dependencies of the
reported D=4 LETTA branch, verifies the replayed energy against the archived
scan, then continues that state in chunks while overwriting a latest-state
tensor snapshot after each chunk.  It avoids rerunning unrelated bond
dimensions and reference calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from examples.mps import scan_frontier_letta_vs_mps_j2_4x4 as scan4
from examples.mps import scan_frontier_letta_vs_mps_j2_8x4 as scan8
from pyqed.mps import MPO


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE_4X4 = RESULTS / "frontier_letta_vs_mps_j2_scan_4x4.json"
SOURCE_8X4 = RESULTS / "frontier_letta_vs_mps_j2_scan_8x4_best.json"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _portable_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(HERE.parents[1]))
    except ValueError:
        return str(path)


def _save_tensors(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"tensor_{site:03d}": tensor for site, tensor in enumerate(state.tensors)},
    )


def _source_row(source, *, seed):
    return next(
        row
        for row in source["records"]
        if np.isclose(float(row["j2_ratio"]), 0.5)
        and int(row["seed"]) == int(seed)
        and row["method"] == "letta_d4"
    )


def _trace_rows(history, *, first_sweep, phase):
    return [
        {
            "sweep": int(first_sweep + offset),
            "phase": str(phase),
            "energy": float(row["energy"]),
            "delta_energy": float(row["delta_energy"]),
            "accepted_sites": int(row["accepted_sites"]),
            "solver_failures": int(row["solver_failures"]),
        }
        for offset, row in enumerate(history, start=1)
    ]


def _continuation_payload(
    *,
    geometry,
    seed,
    source_path,
    state_output,
    source_row,
    replay_records,
    trace,
    state,
    target_tolerance,
    requested_additional_passes,
    chunk_passes,
    gauge_weighting,
    continuation_seconds,
    final_diagnostics,
):
    baseline_passes = len(replay_records[-1]["directional_pass_energies"])
    completed = len(trace) - baseline_passes
    return {
        "model": {
            "geometry": geometry,
            "j1": 1.0,
            "j2_ratio": 0.5,
            "boundary": "open",
            "tie_graph": "all-j1-nearest-neighbor-bonds",
        },
        "settings": {
            "seed": int(seed),
            "bond_dim": 4,
            "source_scan": _portable_path(source_path),
            "target_tolerance": float(target_tolerance),
            "requested_additional_passes": int(requested_additional_passes),
            "chunk_passes": int(chunk_passes),
            "frontier_backend": "compressed",
            "solver": "direct",
            "frontier_canonicalization": True,
            "frontier_gauge_weighting": str(gauge_weighting),
            "state_snapshot": _portable_path(state_output),
            "snapshot_policy": "overwrite latest tensor state after each chunk",
            "snapshot_is_resumable_by_this_script": False,
        },
        "source_record": {
            "energy": float(source_row["energy"]),
            "passes": int(source_row["directional_passes_completed"]),
            "final_delta_energy": float(source_row["final_delta_energy"]),
        },
        "replay": {
            "records": replay_records,
            "target_energy": float(replay_records[-1]["energy"]),
            "absolute_difference_from_source": float(
                abs(replay_records[-1]["energy"] - source_row["energy"])
            ),
        },
        "continuation": {
            "baseline_passes": int(baseline_passes),
            "additional_passes_completed": int(completed),
            "total_passes": int(len(trace)),
            "converged": bool(state.converged),
            "final_energy": float(state.energy),
            "energy_lowering_from_source": float(source_row["energy"] - state.energy),
            "final_delta_energy": float(trace[-1]["delta_energy"]),
            "seconds": float(continuation_seconds),
            "trace": trace,
        },
        "final_diagnostics": final_diagnostics,
    }


def _continue_in_chunks(
    state,
    *,
    geometry,
    seed,
    source_path,
    source_row,
    replay_records,
    baseline_trace,
    output,
    state_output,
    additional_passes,
    chunk_passes,
    target_tolerance,
    gauge_weighting,
    diagnostics,
):
    trace = list(baseline_trace)
    baseline_passes = len(trace)
    continuation_seconds = 0.0
    completed = 0
    while completed < additional_passes and not state.converged:
        count = min(chunk_passes, additional_passes - completed)
        start = perf_counter()
        state.run(
            nsweeps=count,
            sweep_offset=baseline_passes + completed,
            tol=target_tolerance,
            solver="direct",
            frontier_canonicalization=True,
            frontier_gauge_weighting=gauge_weighting,
        )
        elapsed = perf_counter() - start
        continuation_seconds += elapsed
        rows = _trace_rows(
            state.history,
            first_sweep=baseline_passes + completed,
            phase="continuation",
        )
        trace.extend(rows)
        completed += len(rows)
        final_diagnostics = diagnostics(state)
        payload = _continuation_payload(
            geometry=geometry,
            seed=seed,
            source_path=source_path,
            state_output=state_output,
            source_row=source_row,
            replay_records=replay_records,
            trace=trace,
            state=state,
            target_tolerance=target_tolerance,
            requested_additional_passes=additional_passes,
            chunk_passes=chunk_passes,
            gauge_weighting=gauge_weighting,
            continuation_seconds=continuation_seconds,
            final_diagnostics=final_diagnostics,
        )
        _save_tensors(state_output, state)
        _write_json(output, payload)
        print(
            f"{geometry} seed={seed} total_passes={len(trace)} "
            f"E={state.energy:.12f} dE={trace[-1]['delta_energy']:.3e} "
            f"chunk={elapsed:.2f}s converged={state.converged}",
            flush=True,
        )
        if not rows:
            break
    return payload


def continue_4x4(
    *, seed, additional_passes, chunk_passes, target_tolerance, output, state_output
):
    source = _load_json(SOURCE_4X4)
    archived = _source_row(source, seed=seed)
    ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    nrows = ncols = 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    previous_mps = None
    previous_letta = None
    previous_exact = None
    replay_records = []
    target_sparse = target_exact_state = target_exact = None
    for ratio_index, ratio in enumerate(ratios):
        weighted = tuple((i, j, 1.0) for i, j in nearest)
        weighted += tuple((i, j, ratio) for i, j in diagonals)
        hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
        sparse = sparse_heisenberg_hamiltonian(nsites, weighted)
        exact_state, exact = scan4._exact_reference(sparse, previous_exact)
        previous_exact = exact_state
        mps, mps_record = scan4._optimize_mps(
            hamiltonian,
            sparse,
            exact_state,
            exact["energy"],
            bond_dim=4,
            seed=scan4._child_seed(seed, 101, 4),
            pass_limit=80,
            tolerance=1.0e-10,
            initial_state=previous_mps,
        )
        previous_mps = mps.copy()
        state, record = scan4._optimize_letta(
            hamiltonian,
            sparse,
            exact_state,
            exact["energy"],
            parent_sets,
            mps,
            previous_letta,
            bond_dim=4,
            tie_seed=scan4._child_seed(seed, 202, 4, ratio_index),
            tie_noise=1.0e-3,
            pass_limit=40,
            tolerance=1.0e-10,
            frontier_gauge=True,
            frontier_gauge_weighting="probability",
            warm_mps_seconds=mps_record["optimization_seconds"],
        )
        previous_letta = state
        replay_records.append({"j2_ratio": ratio, **record})
        print(
            f"4x4 replay ratio={ratio:.1f} seed={seed} "
            f"E={record['energy']:.12f} dE={record['final_delta_energy']:.3e}",
            flush=True,
        )
        if np.isclose(ratio, 0.5):
            target_sparse = sparse
            target_exact_state = exact_state
            target_exact = exact

    difference = abs(state.energy - archived["energy"])
    if difference > 1.0e-8:
        raise RuntimeError(
            f"4x4 replay differs from archived state by {difference:.3e}; "
            "refusing to label it as a continuation."
        )
    baseline_trace = [
        {
            "sweep": index,
            "phase": "replayed_baseline",
            "energy": float(energy),
            "delta_energy": float(
                abs(energy - record["initial_energy"])
                if index == 1
                else abs(energy - record["directional_pass_energies"][index - 2])
            ),
            "accepted_sites": None,
            "solver_failures": None,
        }
        for index, energy in enumerate(record["directional_pass_energies"], start=1)
    ]

    def diagnostics(current):
        return scan4._diagnostics(
            current.state_vector(normalize=True),
            target_sparse,
            target_exact_state,
            target_exact["energy"],
            nsites,
        )

    return _continue_in_chunks(
        state,
        geometry="4x4",
        seed=seed,
        source_path=SOURCE_4X4,
        source_row=archived,
        replay_records=replay_records,
        baseline_trace=baseline_trace,
        output=output,
        state_output=state_output,
        additional_passes=additional_passes,
        chunk_passes=chunk_passes,
        target_tolerance=target_tolerance,
        gauge_weighting="probability",
        diagnostics=diagnostics,
    )


def continue_8x4(
    *, seed, additional_passes, chunk_passes, target_tolerance, output, state_output
):
    source = _load_json(SOURCE_8X4)
    archived = _source_row(source, seed=seed)
    ratios = (0.0, 0.5)
    nrows, ncols = 8, 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    previous_mps = {2: None, 4: None}
    previous_letta = None
    replay_records = []
    target_reference = None
    for ratio_index, ratio in enumerate(ratios):
        weighted = tuple((i, j, 1.0) for i, j in nearest)
        weighted += tuple((i, j, ratio) for i, j in diagonals)
        hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
        local_mpo = hamiltonian.to_mpo().compress()
        mpo = MPO(list(local_mpo.tensors))
        current_mps = {}
        mps_records = {}
        for bond_dim in (2, 4):
            candidates = {}
            if previous_mps[bond_dim] is not None:
                candidates["previous_ratio"] = previous_mps[bond_dim]
            if current_mps:
                lower_dim = max(current_mps)
                candidates[f"current_mps_d{lower_dim}"] = current_mps[lower_dim]
            if candidates:
                initial, initialization, _ = scan8._lower_energy_mps_candidate(
                    mpo, candidates
                )
            else:
                initial = None
                initialization = "random"
            mps, mps_record = scan8._optimize_mps(
                mpo,
                nsites=nsites,
                bond_dim=bond_dim,
                seed=scan4._child_seed(seed, 101, bond_dim),
                pass_limit=80,
                tolerance=1.0e-9,
                initial_state=initial,
                initialization=initialization,
            )
            previous_mps[bond_dim] = mps.copy()
            current_mps[bond_dim] = mps
            mps_records[bond_dim] = mps_record
        state, record = scan8._optimize_letta(
            hamiltonian,
            parent_sets,
            current_mps[4],
            previous_letta,
            bond_dim=4,
            tie_seed=scan4._child_seed(seed, 202, 4, ratio_index),
            tie_noise=1.0e-3,
            pass_limit=80,
            tolerance=1.0e-9,
            frontier_gauge=True,
            frontier_gauge_weighting="uniform",
            warm_mps_seconds=mps_records[4]["optimization_seconds"],
        )
        previous_letta = state
        replay_records.append({"j2_ratio": ratio, **record})
        print(
            f"8x4 replay ratio={ratio:.1f} seed={seed} "
            f"E={record['energy']:.12f} dE={record['final_delta_energy']:.3e}",
            flush=True,
        )
        if np.isclose(ratio, 0.5):
            target_reference = min(
                float(row["energy"])
                for row in source["reference_runs"]
                if np.isclose(float(row["j2_ratio"]), 0.5)
                and int(row["bond_dim"]) == 32
            )

    difference = abs(state.energy - archived["energy"])
    # The 32-site DMRG warm start can vary at the 1e-7 total-energy level
    # across otherwise identical threaded linear-algebra runs.  This is far
    # below the final per-pass change (~1e-4) and does not identify a distinct
    # variational branch.
    if difference > 1.0e-6:
        raise RuntimeError(
            f"8x4 replay differs from archived ascending state by {difference:.3e}; "
            "refusing to label it as a continuation."
        )
    baseline_trace = [
        {
            "sweep": index,
            "phase": "replayed_baseline",
            "energy": float(energy),
            "delta_energy": float(
                abs(energy - record["initial_energy"])
                if index == 1
                else abs(energy - record["directional_pass_energies"][index - 2])
            ),
            "accepted_sites": None,
            "solver_failures": None,
        }
        for index, energy in enumerate(record["directional_pass_energies"], start=1)
    ]

    def diagnostics(current):
        energy = float(current.energy)
        return {
            "energy": energy,
            "energy_per_site": energy / nsites,
            "mps_d32_reference_energy": float(target_reference),
            "energy_above_reference_per_site": (energy - target_reference) / nsites,
            "frontier_peak_elements": int(current.peak_frontier_elements),
        }

    return _continue_in_chunks(
        state,
        geometry="8x4",
        seed=seed,
        source_path=SOURCE_8X4,
        source_row=archived,
        replay_records=replay_records,
        baseline_trace=baseline_trace,
        output=output,
        state_output=state_output,
        additional_passes=additional_passes,
        chunk_passes=chunk_passes,
        target_tolerance=target_tolerance,
        gauge_weighting="uniform",
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("4x4", "8x4"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--additional-passes", type=int, default=320)
    parser.add_argument("--chunk-passes", type=int, default=40)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-output", type=Path)
    args = parser.parse_args()
    if args.additional_passes < 1:
        parser.error("--additional-passes must be positive.")
    if args.chunk_passes < 1 or args.chunk_passes % 2:
        parser.error("--chunk-passes must be a positive even integer.")

    stem = f"frontier_letta_j2_half_convergence_{args.geometry}_seed{args.seed}"
    output = args.output or RESULTS / f"{stem}.json"
    state_output = args.state_output or RESULTS / f"{stem}.npz"
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 1.0e-10 if args.geometry == "4x4" else 1.0e-9
    runner = continue_4x4 if args.geometry == "4x4" else continue_8x4
    runner(
        seed=args.seed,
        additional_passes=args.additional_passes,
        chunk_passes=args.chunk_passes,
        target_tolerance=float(tolerance),
        output=output,
        state_output=state_output,
    )
    print(f"wrote {output}", flush=True)
    print(f"wrote {state_output}", flush=True)


if __name__ == "__main__":
    main()
