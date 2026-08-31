#!/usr/bin/env python3
"""Residual-led variable-bond continuation of the converged 6x6 LETTA state.

Each candidate starts from the same saved D=4 state.  A selected virtual bond
is enlarged without changing the represented many-body state, and one exact
local S=I update of the newly opened partner tensor measures the useful energy
gain.  The exact ground state is never constructed or used for selection.
"""

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
from examples.mps.continue_frontier_letta_block_sparse_6x6 import (
    DEFAULT_RESULT as DEFAULT_BASELINE_RESULT,
    DEFAULT_SNAPSHOT as DEFAULT_BASELINE_SNAPSHOT,
    _load_tensors,
    _write_json,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_bond_expansion_6x6.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_bond_expansion_6x6.npz"
# Row boundaries plus two central within-row cuts in the snake ordering.
DEFAULT_CANDIDATE_CUTS = (6, 12, 15, 18, 21, 24, 30)


def _save_tensors(path, state):
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


def _state_norm(state):
    return float(np.real(state._norm_frontier.scalar(state.tensors)))


def _build_state(baseline_result, baseline_snapshot):
    payload = json.loads(Path(baseline_result).read_text(encoding="utf-8"))
    model = payload["model"]
    if (
        int(model["nrows"]) != 6
        or int(model["ncols"]) != 6
        or int(model["nsites"]) != 36
    ):
        raise ValueError("the input checkpoint must be the 6x6 LETTA benchmark.")
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(36, weighted_bonds)
    tensors = _load_tensors(baseline_snapshot, 36)
    bond_dims = (1,) + tuple(tensor.shape[1] for tensor in tensors)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(36, nearest),
        bond_dims=bond_dims,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    recorded = float(payload["results"]["letta_d4"]["fresh_energy"])
    measured = float(state.expectation())
    if abs(measured - recorded) > 2.0e-8:
        raise RuntimeError(
            "baseline snapshot energy does not match its JSON record: "
            f"{measured:.16g} versus {recorded:.16g}."
        )
    return state, payload


def _expansion_invariants(before, after, cut, old_dimension, direction):
    left = int(cut) - 1
    right = int(cut)
    slice_errors = [
        np.max(np.abs(after[left][:, :old_dimension] - before[left])),
        np.max(np.abs(after[right][:old_dimension] - before[right])),
    ]
    zero_partner = (
        after[right][old_dimension:]
        if direction == "right"
        else after[left][:, old_dimension:]
    )
    return {
        "preserved_slice_max_abs_error": float(max(slice_errors)),
        "zero_partner_norm": float(np.linalg.norm(zero_partner)),
        "algebraically_state_preserving": bool(
            max(slice_errors) == 0.0 and np.linalg.norm(zero_partner) == 0.0
        ),
    }


def _candidate_record(
    baseline,
    *,
    cut,
    direction,
    new_dimension,
    scale,
    solver,
    seed,
):
    start = perf_counter()
    state = baseline.copy()
    before = [tensor.copy() for tensor in state.tensors]
    energy_before = float(state.expectation())
    norm_before = _state_norm(state)
    parameters_before = state.nparameters
    expansion = state.expand_bond(
        cut,
        new_dimension,
        direction=direction,
        strategy="residual",
        scale=scale,
        seed=seed,
    )
    energy_expanded = float(state.expectation())
    norm_expanded = _state_norm(state)
    invariants = _expansion_invariants(
        before,
        state.tensors,
        cut,
        expansion.old_dimension,
        direction,
    )
    relaxed_site = int(cut) if direction == "right" else int(cut) - 1
    update = state.optimize_site(
        relaxed_site,
        solver=solver,
        metric_tol=1.0e-12,
        eig_tol=1.0e-10,
        maxiter=1600,
        max_subspace=96,
    )
    energy_after = float(state.expectation())
    added_parameters = int(state.nparameters - parameters_before)
    gain = float(energy_before - energy_after)
    return {
        "cut": int(cut),
        "direction": direction,
        "relaxed_site": relaxed_site,
        "old_dimension": int(expansion.old_dimension),
        "new_dimension": int(expansion.new_dimension),
        "seeded_directions": int(expansion.seeded_directions),
        "residual_source_norm": float(expansion.source_norm),
        "added_parameters": added_parameters,
        "energy_before": energy_before,
        "energy_immediately_after_expansion": energy_expanded,
        "pre_relaxation_energy_error": float(energy_expanded - energy_before),
        "pre_relaxation_norm_error": float(norm_expanded - norm_before),
        **invariants,
        "local_solver": update.solver,
        "local_solver_converged": bool(update.solver_converged),
        "local_update_accepted": bool(update.accepted),
        "local_solver_message": update.message,
        "local_metric_rank": int(update.metric_rank),
        "local_raw_dimension": int(update.raw_dim),
        "local_residual_norm": float(update.residual_norm),
        "local_metric_is_identity": bool(update.solver_metric_is_identity),
        "local_metric_identity_error": float(
            update.solver_metric_identity_error
        ),
        "energy_after_candidate_relaxation": energy_after,
        "candidate_energy_gain": gain,
        "gain_per_added_parameter": (
            gain / added_parameters if added_parameters else 0.0
        ),
        "seconds": float(perf_counter() - start),
    }


def _select_candidates(records, count, minimum_separation):
    # The better direction at each cut is the cut's residual-led score.
    best_by_cut = {}
    for record in records:
        cut = record["cut"]
        previous = best_by_cut.get(cut)
        key = (record["candidate_energy_gain"], -record["added_parameters"])
        if previous is None or key > (
            previous["candidate_energy_gain"],
            -previous["added_parameters"],
        ):
            best_by_cut[cut] = record
    ranked = sorted(
        best_by_cut.values(),
        key=lambda row: (-row["candidate_energy_gain"], row["added_parameters"]),
    )
    selected = []
    for record in ranked:
        if all(
            abs(record["cut"] - chosen["cut"]) >= minimum_separation
            for chosen in selected
        ):
            selected.append(record)
        if len(selected) == count:
            break
    return ranked, selected


def run_benchmark(
    *,
    baseline_result=DEFAULT_BASELINE_RESULT,
    baseline_snapshot=DEFAULT_BASELINE_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    candidate_cuts=DEFAULT_CANDIDATE_CUTS,
    new_dimension=6,
    selected_bonds=2,
    minimum_cut_separation=2,
    relaxation_passes=2,
    solver="whitened",
    expansion_scale=1.0e-3,
    seed=23,
):
    total_start = perf_counter()
    baseline, source_payload = _build_state(baseline_result, baseline_snapshot)
    baseline_energy = float(baseline.expectation())
    baseline_norm = _state_norm(baseline)
    baseline_parameters = baseline.nparameters
    candidate_cuts = tuple(dict.fromkeys(int(cut) for cut in candidate_cuts))
    if any(cut <= 0 or cut >= len(baseline.dims) for cut in candidate_cuts):
        raise ValueError("candidate cuts must be internal chain cuts.")
    if new_dimension <= max(baseline.bond_dims):
        raise ValueError("new_dimension must exceed the baseline bond dimension.")

    print(
        f"baseline E={baseline_energy:.12f}, parameters={baseline_parameters}",
        flush=True,
    )
    candidate_start = perf_counter()
    records = []
    for index, cut in enumerate(candidate_cuts):
        for direction_index, direction in enumerate(("right", "left")):
            record = _candidate_record(
                baseline,
                cut=cut,
                direction=direction,
                new_dimension=new_dimension,
                scale=expansion_scale,
                solver=solver,
                seed=int(seed + 2 * index + direction_index),
            )
            records.append(record)
            print(
                f"cut={cut:2d} {direction:5s} "
                f"gain={record['candidate_energy_gain']:.6e} "
                f"source={record['residual_source_norm']:.3e} "
                f"failure={not record['local_solver_converged']}",
                flush=True,
            )
    candidate_seconds = perf_counter() - candidate_start
    ranked, selected = _select_candidates(
        records,
        int(selected_bonds),
        int(minimum_cut_separation),
    )
    if len(selected) != int(selected_bonds):
        raise RuntimeError("not enough separated candidate cuts were available.")

    state = baseline.copy()
    expansion_records = []
    for selection_rank, selected_record in enumerate(selected, start=1):
        cut = int(selected_record["cut"])
        direction = selected_record["direction"]
        before = [tensor.copy() for tensor in state.tensors]
        norm_before = _state_norm(state)
        energy_before = float(state.expectation())
        expansion = state.expand_bond(
            cut,
            int(new_dimension),
            direction=direction,
            strategy="residual",
            scale=float(expansion_scale),
            seed=int(seed + 1000 + selection_rank),
        )
        invariants = _expansion_invariants(
            before,
            state.tensors,
            cut,
            expansion.old_dimension,
            direction,
        )
        expansion_records.append(
            {
                "selection_rank": selection_rank,
                "cut": cut,
                "direction": direction,
                "old_dimension": int(expansion.old_dimension),
                "new_dimension": int(expansion.new_dimension),
                "seeded_directions": int(expansion.seeded_directions),
                "residual_source_norm": float(expansion.source_norm),
                "energy_error": float(state.expectation() - energy_before),
                "norm_error": float(_state_norm(state) - norm_before),
                **invariants,
            }
        )

    pre_relaxation_energy = float(state.expectation())
    pre_relaxation_norm = _state_norm(state)
    relaxation_start = perf_counter()
    state.run(
        nsweeps=int(relaxation_passes),
        tol=0.0,
        solver=solver,
        metric_tol=1.0e-12,
        eig_tol=1.0e-10,
        maxiter=1600,
        max_subspace=96,
        gauge="frontier",
        gauge_weight="uniform",
        verbose=True,
    )
    relaxation_seconds = perf_counter() - relaxation_start
    final_energy = float(state.expectation())
    updates = [update for row in state.history for update in row["updates"]]
    failures = [update for update in updates if not update.solver_converged]

    payload = {
        "status": "complete",
        "model": source_payload["model"],
        "protocol": {
            "selection": (
                "current-state residual expansion followed by one variational "
                "partner-site relaxation; no exact-state oracle"
            ),
            "baseline_result": str(Path(baseline_result).resolve()),
            "baseline_snapshot": str(Path(baseline_snapshot).resolve()),
            "candidate_cuts": list(candidate_cuts),
            "candidate_directions": ["right", "left"],
            "old_bond_dimension": int(max(baseline.bond_dims)),
            "new_bond_dimension": int(new_dimension),
            "selected_bonds": int(selected_bonds),
            "minimum_cut_separation": int(minimum_cut_separation),
            "expansion_strategy": "residual",
            "expansion_scale": float(expansion_scale),
            "candidate_solver": solver,
            "relaxation_solver": solver,
            "relaxation_passes": int(relaxation_passes),
            "gauge": "frontier",
            "gauge_weight": "uniform",
            "seed": int(seed),
            "exact_state_used_for_selection": False,
        },
        "baseline": {
            "bond_dims": list(baseline.bond_dims),
            "parameters": int(baseline_parameters),
            "norm": baseline_norm,
            "energy": baseline_energy,
            "energy_per_site": baseline_energy / 36,
        },
        "candidate_ranking": [
            {**record, "rank": rank}
            for rank, record in enumerate(ranked, start=1)
        ],
        "all_directional_candidates": records,
        "selected_expansions": expansion_records,
        "expanded_pre_relaxation": {
            "bond_dims": list(state.bond_dims),
            "parameters": int(state.nparameters),
            "added_parameters": int(state.nparameters - baseline_parameters),
            "norm": pre_relaxation_norm,
            "norm_error": float(pre_relaxation_norm - baseline_norm),
            "energy": pre_relaxation_energy,
            "energy_error": float(pre_relaxation_energy - baseline_energy),
            "all_expansions_algebraically_state_preserving": all(
                record["algebraically_state_preserving"]
                for record in expansion_records
            ),
        },
        "relaxed": {
            "bond_dims": list(state.bond_dims),
            "parameters": int(state.nparameters),
            "energy": final_energy,
            "energy_per_site": final_energy / 36,
            "fresh_energy_gain": float(baseline_energy - final_energy),
            "fresh_energy_gain_per_site": float(
                (baseline_energy - final_energy) / 36
            ),
            "directional_pass_energies": [
                float(row["energy"]) for row in state.history
            ],
            "accepted_updates": int(sum(update.accepted for update in updates)),
            "site_updates": len(updates),
            "solver_failures": len(failures),
            "failure_sites": [int(update.site) for update in failures],
            "failure_messages": sorted({update.message for update in failures}),
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
            "snapshot": str(Path(snapshot).resolve()),
        },
        "timing_seconds": {
            "candidate_ranking": float(candidate_seconds),
            "final_relaxation": float(relaxation_seconds),
            "total": float(perf_counter() - total_start),
        },
    }
    _save_tensors(snapshot, state)
    _write_json(output, payload)
    print(
        f"selected cuts={[record['cut'] for record in selected]} "
        f"bond_dims={state.bond_dims}",
        flush=True,
    )
    print(
        f"final E={final_energy:.12f}, gain={baseline_energy-final_energy:.6e}, "
        f"failures={len(failures)}, time={payload['timing_seconds']['total']:.2f}s",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument(
        "--baseline-snapshot", type=Path, default=DEFAULT_BASELINE_SNAPSHOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--candidate-cuts", type=int, nargs="+", default=DEFAULT_CANDIDATE_CUTS
    )
    parser.add_argument("--new-dimension", type=int, default=6)
    parser.add_argument("--selected-bonds", type=int, default=2)
    parser.add_argument("--minimum-cut-separation", type=int, default=2)
    parser.add_argument("--relaxation-passes", type=int, default=2)
    parser.add_argument(
        "--solver", choices=("whitened", "block_sparse"), default="whitened"
    )
    parser.add_argument("--expansion-scale", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    run_benchmark(
        baseline_result=args.baseline_result,
        baseline_snapshot=args.baseline_snapshot,
        output=args.output,
        snapshot=args.snapshot,
        candidate_cuts=args.candidate_cuts,
        new_dimension=args.new_dimension,
        selected_bonds=args.selected_bonds,
        minimum_cut_separation=args.minimum_cut_separation,
        relaxation_passes=args.relaxation_passes,
        solver=args.solver,
        expansion_scale=args.expansion_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
