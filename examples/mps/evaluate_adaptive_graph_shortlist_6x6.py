#!/usr/bin/env python3
"""Variationally select among the confirmed top-five 6x6 tie proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.adaptive_graph_frontier_letta_6x6 import _cost_record
from examples.mps.continue_frontier_letta_block_sparse_6x6 import (
    DEFAULT_RESULT,
    DEFAULT_SNAPSHOT,
    _load_tensors,
    _save_tensors,
    _write_json,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import (
    FrontierTiedLETTA,
    TieSignal,
    TieSignalBatch,
    evaluate_tie_graph_proposal,
    rank_tie_graph_proposals,
    tie_edges,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_CONFIRMATION = RESULTS / "frontier_letta_adaptive_j2_tie_6x6_confirm.json"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_adaptive_j2_shortlist_6x6.json"
DEFAULT_OUTPUT_STATE = RESULTS / "frontier_letta_adaptive_j2_shortlist_6x6.npz"


def evaluate_shortlist(
    *,
    baseline_result=DEFAULT_RESULT,
    baseline_state=DEFAULT_SNAPSHOT,
    confirmation=DEFAULT_CONFIRMATION,
    output=DEFAULT_OUTPUT,
    output_state=DEFAULT_OUTPUT_STATE,
    shortlist=5,
):
    start = perf_counter()
    baseline_payload = json.loads(Path(baseline_result).read_text(encoding="utf-8"))
    confirmation_payload = json.loads(
        Path(confirmation).read_text(encoding="utf-8")
    )
    shortlist = int(shortlist)
    rows = confirmation_payload["ranking"]["top_signals"][:shortlist]
    if len(rows) != shortlist:
        raise ValueError("confirmation JSON does not contain the requested shortlist.")

    nearest, diagonals = square_j1_j2_bonds(6, 6)
    j2 = float(baseline_payload["model"]["j2"])
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, j2) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(36, weighted)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(36, nearest),
        bond_dim=int(baseline_payload["settings"]["bond_dim"]),
        tensors=_load_tensors(baseline_state, 36),
        frontier_backend="identity_block",
    )
    baseline_energy = float(state.expectation())
    recorded = float(baseline_payload["results"]["letta_d4"]["fresh_energy"])
    if abs(baseline_energy - recorded) > 2.0e-8:
        raise RuntimeError("baseline state does not match its recorded energy.")

    signals = tuple(
        TieSignal(
            edge=tuple(int(site) for site in row["edge"]),
            connected_correlation=float(row["connected_correlation"]),
            residual_coupling=float(row["residual_coupling"]),
            score=float(row["signal"]),
        )
        for row in rows
    )
    sampling = confirmation_payload["sampling"]
    mean_energy = sampling["mean_local_energy"]
    batch = TieSignalBatch(
        signals=signals,
        nsamples=int(sampling["nsamples"]),
        mean_local_energy=complex(mean_energy["real"], mean_energy["imag"]),
        local_energy_variance=float(sampling["local_energy_variance"]),
        acceptance_rate=float(sampling["acceptance_rate"]),
    )
    proposals = rank_tie_graph_proposals(
        state,
        batch,
        operations=("add",),
        cost_weight=float(confirmation_payload["ranking"]["cost_weight"]),
        max_frontier_width=7,
    )
    evaluations = []
    evaluation_seconds = []
    for index, proposal in enumerate(proposals):
        proposal_start = perf_counter()
        evaluation = evaluate_tie_graph_proposal(
            state,
            proposal,
            relaxation_sweeps=1,
            minimum_energy_gain=0.0,
            energy_cost_weight=0.0,
            run_options={
                "solver": "whitened",
                "tol": 0.0,
                "metric_tol": 1.0e-12,
                "gauge": "frontier",
                "gauge_weight": "uniform",
                "verbose": True,
            },
        )
        evaluations.append(evaluation)
        evaluation_seconds.append(perf_counter() - proposal_start)
        print(
            f"candidate={index + 1}/{len(proposals)} edge={proposal.edge} "
            f"gain={evaluation.energy_gain:.12e}",
            flush=True,
        )
    accepted = [item for item in evaluations if item.accepted]
    if not accepted:
        raise RuntimeError("none of the confirmed proposals lowered the energy.")
    selected = max(accepted, key=lambda item: item.energy_gain)
    final_state = selected.candidate_state
    final_energy = float(final_state.expectation())
    if abs(final_energy - selected.energy_after) > 2.0e-8:
        raise RuntimeError("selected candidate failed its final energy check.")

    output = Path(output)
    output_state = Path(output_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    _save_tensors(output_state, final_state.tensors)
    result = {
        "model": {
            "nrows": 6,
            "ncols": 6,
            "nsites": 36,
            "j1": 1.0,
            "j2": j2,
            "boundary": "open",
        },
        "source": {
            "baseline_result": str(baseline_result),
            "baseline_state": str(baseline_state),
            "confirmation": str(confirmation),
            "sample_filter": {
                "chains": int(sampling["chains"]),
                "nsamples": int(sampling["nsamples"]),
                "sector": sampling["sector"],
                "acceptance_rate": float(sampling["acceptance_rate"]),
            },
        },
        "baseline": {
            "energy": baseline_energy,
            "energy_per_site": baseline_energy / 36,
            "parameters": int(state.nparameters),
            "ties": len(tie_edges(state.parent_sets)),
        },
        "selection": {
            "method": "sample-filtered top five, selected by fresh one-pass energy gain",
            "shortlist_size": len(evaluations),
            "candidates": [
                {
                    "sample_rank": index + 1,
                    "edge": list(evaluation.proposal.edge),
                    "connected_correlation": float(
                        evaluation.proposal.signal.connected_correlation
                    ),
                    "residual_coupling": float(
                        evaluation.proposal.signal.residual_coupling
                    ),
                    "sample_signal": float(evaluation.proposal.signal.score),
                    "proxy_utility": float(evaluation.proposal.proxy_utility),
                    "delta_log_cost": float(evaluation.proposal.delta_log_cost),
                    "cost_before": _cost_record(evaluation.proposal.cost_before),
                    "cost_after": _cost_record(evaluation.proposal.cost_after),
                    "fresh_energy": float(evaluation.energy_after),
                    "energy_gain": float(evaluation.energy_gain),
                    "energy_gain_per_site": float(evaluation.energy_gain / 36),
                    "fresh_energy_check_passed": bool(
                        evaluation.fresh_energy_check_passed
                    ),
                    "accepted": bool(evaluation.accepted),
                    "seconds": float(seconds),
                }
                for index, (evaluation, seconds) in enumerate(
                    zip(evaluations, evaluation_seconds)
                )
            ],
            "selected_edge": list(selected.proposal.edge),
        },
        "final": {
            "energy": final_energy,
            "energy_per_site": final_energy / 36,
            "energy_gain": baseline_energy - final_energy,
            "energy_gain_per_site": (baseline_energy - final_energy) / 36,
            "parameters": int(final_state.nparameters),
            "parameter_increase": int(final_state.nparameters - state.nparameters),
            "ties": len(tie_edges(final_state.parent_sets)),
            "state_file": str(output_state),
        },
        "total_seconds": float(perf_counter() - start),
    }
    _write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--baseline-state", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-state", type=Path, default=DEFAULT_OUTPUT_STATE)
    parser.add_argument("--shortlist", type=int, default=5)
    args = parser.parse_args()
    result = evaluate_shortlist(
        baseline_result=args.baseline_result,
        baseline_state=args.baseline_state,
        confirmation=args.confirmation,
        output=args.output,
        output_state=args.output_state,
        shortlist=args.shortlist,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
