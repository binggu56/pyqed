#!/usr/bin/env python3
"""Combine variable virtual bonds and an adaptive J2 tie on the 6x6 state."""

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
from examples.mps.adaptive_graph_frontier_letta_6x6 import _cost_record
from examples.mps.benchmark_frontier_letta_bond_expansion_6x6 import (
    DEFAULT_OUTPUT as DEFAULT_EXPANSION_RESULT,
    DEFAULT_SNAPSHOT as DEFAULT_EXPANSION_STATE,
    _save_tensors,
)
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.evaluate_adaptive_graph_shortlist_6x6 import (
    DEFAULT_OUTPUT as DEFAULT_SHORTLIST,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import (
    FrontierTiedLETTA,
    TieSignal,
    TieSignalBatch,
    rank_tie_graph_proposals,
    state_with_tie_graph_proposal,
    tie_edges,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_variable_bond_adaptive_tie_6x6.json"
DEFAULT_OUTPUT_STATE = RESULTS / "frontier_letta_variable_bond_adaptive_tie_6x6.npz"


def _load_expanded_state(result_path, state_path):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    with np.load(state_path, allow_pickle=False) as archive:
        tensors = [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(36)
        ]
        bond_dims = tuple(int(value) for value in archive["virtual_bond_dims"])
    if list(bond_dims) != payload["relaxed"]["bond_dims"]:
        raise RuntimeError("variable-bond snapshot dimensions disagree with JSON.")
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    j2 = float(payload["model"]["j2"])
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, j2) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(36, weighted)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(36, nearest),
        bond_dims=bond_dims,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    measured = float(state.expectation())
    recorded = float(payload["relaxed"]["energy"])
    if abs(measured - recorded) > 2.0e-8:
        raise RuntimeError("variable-bond checkpoint energy disagrees with JSON.")
    return state, payload


def run_integrated_check(
    *,
    expansion_result=DEFAULT_EXPANSION_RESULT,
    expansion_state=DEFAULT_EXPANSION_STATE,
    shortlist_result=DEFAULT_SHORTLIST,
    output=DEFAULT_OUTPUT,
    output_state=DEFAULT_OUTPUT_STATE,
):
    total_start = perf_counter()
    state, expansion_payload = _load_expanded_state(
        expansion_result, expansion_state
    )
    shortlist = json.loads(Path(shortlist_result).read_text(encoding="utf-8"))
    edge = tuple(int(site) for site in shortlist["selection"]["selected_edge"])
    if edge != (27, 33):
        raise ValueError("the expected adaptive shortlist edge is (27, 33).")
    selected_row = next(
        row
        for row in shortlist["selection"]["candidates"]
        if tuple(row["edge"]) == edge
    )
    signal = TieSignal(
        edge=edge,
        connected_correlation=float(selected_row["connected_correlation"]),
        residual_coupling=float(selected_row["residual_coupling"]),
        score=float(selected_row["sample_signal"]),
    )
    signal_batch = TieSignalBatch(
        signals=(signal,),
        nsamples=int(shortlist["source"]["sample_filter"]["nsamples"]),
        mean_local_energy=0.0j,
        local_energy_variance=0.0,
        acceptance_rate=float(
            shortlist["source"]["sample_filter"]["acceptance_rate"]
        ),
    )
    proposal = rank_tie_graph_proposals(
        state,
        signal_batch,
        operations=("add",),
        cost_weight=0.2,
        max_frontier_width=7,
    )[0]

    expanded_energy = float(state.expectation())
    expanded_norm = float(np.real(state._norm_frontier.scalar(state.tensors)))
    expanded_parameters = int(state.nparameters)
    expanded_bond_dims = tuple(state.bond_dims)
    control = state.copy()
    control_start = perf_counter()
    control.run(
        nsweeps=1,
        tol=0.0,
        solver="whitened",
        metric_tol=1.0e-12,
        eig_tol=1.0e-10,
        maxiter=1600,
        max_subspace=96,
        frontier_canonicalization=True,
        frontier_gauge_weighting="uniform",
        verbose=False,
    )
    control_seconds = perf_counter() - control_start
    control_energy = float(control.expectation())
    migration_start = perf_counter()
    candidate = state_with_tie_graph_proposal(state, proposal)
    migrated_energy = float(candidate.expectation())
    migrated_norm = float(
        np.real(candidate._norm_frontier.scalar(candidate.tensors))
    )
    migration_seconds = perf_counter() - migration_start
    if candidate.bond_dims != expanded_bond_dims:
        raise RuntimeError("adaptive tie migration changed variable bond dimensions.")
    if abs(migrated_energy - expanded_energy) > 2.0e-8:
        raise RuntimeError("adaptive tie insertion did not preserve the state energy.")
    if abs(migrated_norm - expanded_norm) > 2.0e-8:
        raise RuntimeError("adaptive tie insertion did not preserve the state norm.")

    relaxation_start = perf_counter()
    candidate.run(
        nsweeps=1,
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
    relaxation_seconds = perf_counter() - relaxation_start
    fresh_energy = float(candidate.expectation())
    fresh_gain = float(expanded_energy - fresh_energy)
    tolerance = 512.0 * np.finfo(float).eps * max(1.0, abs(expanded_energy))
    fresh_check_passed = bool(fresh_energy <= expanded_energy + tolerance)
    if not fresh_check_passed:
        raise RuntimeError("integrated relaxation failed the fresh energy check.")
    updates = [
        update for sweep in candidate.history for update in sweep["updates"]
    ]
    failures = [update for update in updates if not update.solver_converged]

    output = Path(output)
    output_state = Path(output_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    _save_tensors(output_state, candidate)
    payload = {
        "status": "complete",
        "model": expansion_payload["model"],
        "protocol": {
            "variable_bond_source": str(Path(expansion_state).resolve()),
            "adaptive_graph_source": str(Path(shortlist_result).resolve()),
            "selected_tie": list(edge),
            "relaxation_passes": 1,
            "solver": "whitened",
            "frontier_canonicalization": "uniform",
        },
        "expanded_state": {
            "bond_dims": list(expanded_bond_dims),
            "expanded_cuts": [
                cut for cut, dimension in enumerate(expanded_bond_dims) if dimension == 6
            ],
            "parameters": expanded_parameters,
            "ties": len(tie_edges(state.parent_sets)),
            "norm": expanded_norm,
            "energy": expanded_energy,
            "energy_per_site": expanded_energy / 36,
        },
        "tie_migration": {
            "edge": list(edge),
            "parameters": int(candidate.nparameters),
            "parameter_increase": int(candidate.nparameters - expanded_parameters),
            "ties": len(tie_edges(candidate.parent_sets)),
            "bond_dims_preserved": candidate.bond_dims == expanded_bond_dims,
            "energy": migrated_energy,
            "energy_error": migrated_energy - expanded_energy,
            "norm": migrated_norm,
            "norm_error": migrated_norm - expanded_norm,
            "state_preserved": bool(
                abs(migrated_energy - expanded_energy) <= 2.0e-8
                and abs(migrated_norm - expanded_norm) <= 2.0e-8
            ),
            "frontier_cost_before": _cost_record(proposal.cost_before),
            "frontier_cost_after": _cost_record(proposal.cost_after),
            "delta_log_frontier_cost": float(proposal.delta_log_cost),
        },
        "no_new_tie_control": {
            "passes": 1,
            "energy": control_energy,
            "energy_per_site": control_energy / 36,
            "fresh_energy_gain_from_expanded_state": (
                expanded_energy - control_energy
            ),
            "solver_failures": int(
                sum(
                    not update.solver_converged
                    for sweep in control.history
                    for update in sweep["updates"]
                )
            ),
        },
        "relaxed": {
            "bond_dims": list(candidate.bond_dims),
            "parameters": int(candidate.nparameters),
            "energy": fresh_energy,
            "energy_per_site": fresh_energy / 36,
            "fresh_energy_gain_from_expanded_state": fresh_gain,
            "fresh_energy_gain_per_site": fresh_gain / 36,
            "energy_below_no_new_tie_control": control_energy - fresh_energy,
            "energy_below_no_new_tie_control_per_site": (
                control_energy - fresh_energy
            )
            / 36,
            "fresh_energy_check_passed": fresh_check_passed,
            "directional_pass_energies": [
                float(sweep["energy"]) for sweep in candidate.history
            ],
            "accepted_updates": int(sum(update.accepted for update in updates)),
            "site_updates": len(updates),
            "solver_failures": len(failures),
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
            "snapshot": str(output_state.resolve()),
        },
        "timing_seconds": {
            "no_new_tie_control": float(control_seconds),
            "migration": float(migration_seconds),
            "relaxation": float(relaxation_seconds),
            "total": float(perf_counter() - total_start),
        },
    }
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expansion-result", type=Path, default=DEFAULT_EXPANSION_RESULT
    )
    parser.add_argument(
        "--expansion-state", type=Path, default=DEFAULT_EXPANSION_STATE
    )
    parser.add_argument("--shortlist-result", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-state", type=Path, default=DEFAULT_OUTPUT_STATE)
    args = parser.parse_args()
    result = run_integrated_check(
        expansion_result=args.expansion_result,
        expansion_state=args.expansion_state,
        shortlist_result=args.shortlist_result,
        output=args.output,
        output_state=args.output_state,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
