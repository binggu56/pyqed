#!/usr/bin/env python3
"""Add one cost-aware J2 tie to the converged 6x6 graph-LETTA state."""

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
    LETTAVMC,
    adaptive_tie_graph_step,
    graph_signals_from_samples,
    tie_edges,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_adaptive_j2_tie_6x6.json"
DEFAULT_OUTPUT_STATE = RESULTS / "frontier_letta_adaptive_j2_tie_6x6.npz"


def _cost_record(cost):
    return {
        "peak_width": int(cost.peak_width),
        "peak_physical_states": int(cost.peak_physical_states),
        "peak_norm_elements": int(cost.peak_norm_elements),
        "total_norm_elements": int(cost.total_norm_elements),
        "log_objective": float(cost.log_objective),
    }


def run_adaptive_diagonal(
    *,
    baseline_result=DEFAULT_RESULT,
    baseline_state=DEFAULT_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    output_state=DEFAULT_OUTPUT_STATE,
    nsamples=256,
    burn_in=100,
    sweeps_between=2,
    chains=1,
    sampler_proposal="mixed",
    exchange_probability=0.75,
    cost_weight=0.2,
    seed=29,
):
    baseline_result = Path(baseline_result)
    baseline_state = Path(baseline_state)
    payload = json.loads(baseline_result.read_text(encoding="utf-8"))
    model = payload["model"]
    settings = payload["settings"]
    if (int(model["nrows"]), int(model["ncols"])) != (6, 6):
        raise ValueError("the baseline result must be the 6x6 calculation.")
    nsites = 36
    nearest, diagonals = square_j1_j2_bonds(6, 6)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)

    setup_start = perf_counter()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(nsites, nearest),
        bond_dim=int(settings["bond_dim"]),
        tensors=_load_tensors(baseline_state, nsites),
        frontier_backend="identity_block",
    )
    baseline_energy = float(state.expectation())
    recorded_energy = float(payload["results"]["letta_d4"]["fresh_energy"])
    if abs(baseline_energy - recorded_energy) > 2.0e-8:
        raise RuntimeError("baseline tensor checkpoint does not match its JSON energy.")
    setup_seconds = perf_counter() - setup_start

    chains = int(chains)
    nsamples = int(nsamples)
    if chains < 1 or nsamples < chains or nsamples % chains:
        raise ValueError("nsamples must be a positive multiple of chains.")
    sample_start = perf_counter()
    configurations = []
    local_energies = []
    attempts = 0
    accepted = 0
    for chain in range(chains):
        initial_configuration = np.repeat((0, 1), nsites // 2)
        chain_rng = np.random.default_rng(int(seed) + 10_000 + chain)
        chain_rng.shuffle(initial_configuration)
        vmc = LETTAVMC(
            state,
            state.hamiltonian,
            seed=int(seed) + chain,
            initial_configuration=initial_configuration,
            proposal=str(sampler_proposal),
            exchange_probability=float(exchange_probability),
        )
        samples = vmc.sample(
            nsamples // chains,
            burn_in=int(burn_in),
            sweeps_between=int(sweeps_between),
        )
        configurations.append(samples.configurations)
        local_energies.append(samples.local_energies)
        attempts += samples.diagnostics.attempts
        accepted += samples.diagnostics.accepted
    signal_batch = graph_signals_from_samples(
        np.concatenate(configurations, axis=0),
        np.concatenate(local_energies, axis=0),
        state.dims,
        diagonals,
        correlation_weight=1.0,
        residual_weight=1.0,
        acceptance_rate=float(accepted / attempts) if attempts else 0.0,
    )
    sample_seconds = perf_counter() - sample_start

    adaptive_start = perf_counter()
    result = adaptive_tie_graph_step(
        state,
        candidate_edges=diagonals,
        signal_batch=signal_batch,
        operations=("add",),
        shortlist=1,
        cost_weight=float(cost_weight),
        energy_cost_weight=0.0,
        max_frontier_width=7,
        relaxation_sweeps=1,
        minimum_energy_gain=0.0,
        run_options={
            "solver": "whitened",
            "tol": 0.0,
            "metric_tol": 1.0e-12,
            "frontier_canonicalization": True,
            "frontier_gauge_weighting": "uniform",
            "verbose": True,
        },
    )
    adaptive_seconds = perf_counter() - adaptive_start
    if not result.evaluations:
        raise RuntimeError("no J2 diagonal passed the frontier-width constraint.")
    evaluation = result.evaluations[0]
    proposal = evaluation.proposal
    candidate = evaluation.candidate_state
    final_state = result.state
    fresh_final_energy = float(final_state.expectation())
    if result.selected is not None and abs(
        fresh_final_energy - evaluation.energy_after
    ) > 2.0e-8:
        raise RuntimeError("selected state failed the final energy consistency check.")

    output = Path(output)
    output_state = Path(output_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    _save_tensors(output_state, final_state.tensors)
    updates = [
        update
        for sweep in candidate.history
        for update in sweep.get("updates", ())
    ]
    record = {
        "model": {
            "nrows": 6,
            "ncols": 6,
            "nsites": nsites,
            "j1": 1.0,
            "j2": float(model["j2"]),
            "boundary": model.get("boundary", "open"),
        },
        "baseline": {
            "result": str(baseline_result),
            "state": str(baseline_state),
            "energy": baseline_energy,
            "energy_per_site": baseline_energy / nsites,
            "parameters": int(state.nparameters),
            "ties": len(tie_edges(state.parent_sets)),
        },
        "sampling": {
            "seed": int(seed),
            "chains": chains,
            "chain_seeds": [int(seed) + chain for chain in range(chains)],
            "nsamples": int(result.signal_batch.nsamples),
            "burn_in": int(burn_in),
            "sweeps_between": int(sweeps_between),
            "proposal": str(sampler_proposal),
            "exchange_probability": float(exchange_probability),
            "sector": "half filling (total Sz = 0)",
            "acceptance_rate": float(result.signal_batch.acceptance_rate),
            "mean_local_energy": {
                "real": float(result.signal_batch.mean_local_energy.real),
                "imag": float(result.signal_batch.mean_local_energy.imag),
            },
            "local_energy_variance": float(
                result.signal_batch.local_energy_variance
            ),
        },
        "ranking": {
            "candidate_family": "missing J2 diagonal ties",
            "candidate_count": len(diagonals),
            "cost_weight": float(cost_weight),
            "correlation_weight": 1.0,
            "residual_weight": 1.0,
            "top_signals": [
                {
                    "edge": list(item.edge),
                    "connected_correlation": float(
                        item.signal.connected_correlation
                    ),
                    "residual_coupling": float(item.signal.residual_coupling),
                    "signal": float(item.signal.score),
                    "delta_log_cost": float(item.delta_log_cost),
                    "proxy_utility": float(item.proxy_utility),
                    "peak_width_after": int(item.cost_after.peak_width),
                }
                for item in result.proposals[:10]
            ],
        },
        "selected_proposal": {
            "edge": list(proposal.edge),
            "operation": proposal.operation,
            "connected_correlation": float(
                proposal.signal.connected_correlation
            ),
            "residual_coupling": float(proposal.signal.residual_coupling),
            "signal": float(proposal.signal.score),
            "proxy_utility": float(proposal.proxy_utility),
            "cost_before": _cost_record(proposal.cost_before),
            "cost_after": _cost_record(proposal.cost_after),
            "delta_log_cost": float(proposal.delta_log_cost),
        },
        "relaxation": {
            "directional_passes": 1,
            "solver": "whitened",
            "frontier_canonicalization": "uniform",
            "migrated_energy": float(evaluation.migrated_energy),
            "fresh_energy": float(evaluation.energy_after),
            "fresh_energy_per_site": float(evaluation.energy_after / nsites),
            "energy_gain": float(evaluation.energy_gain),
            "energy_gain_per_site": float(evaluation.energy_gain / nsites),
            "fresh_energy_check_passed": bool(
                evaluation.fresh_energy_check_passed
            ),
            "accepted": bool(evaluation.accepted),
            "selected": result.selected is not None,
            "directional_energies": list(evaluation.relaxation_energies),
            "accepted_sites": sum(update.accepted for update in updates),
            "solver_failures": sum(
                not update.solver_converged for update in updates
            ),
        },
        "final": {
            "energy": fresh_final_energy,
            "energy_per_site": fresh_final_energy / nsites,
            "parameters": int(final_state.nparameters),
            "parameter_increase": int(final_state.nparameters - state.nparameters),
            "ties": len(tie_edges(final_state.parent_sets)),
            "state_file": str(output_state),
        },
        "timing_seconds": {
            "setup": float(setup_seconds),
            "sampling": float(sample_seconds),
            "ranking_and_relaxation": float(adaptive_seconds),
            "total": float(setup_seconds + sample_seconds + adaptive_seconds),
        },
    }
    _write_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--baseline-state", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-state", type=Path, default=DEFAULT_OUTPUT_STATE)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--sweeps-between", type=int, default=2)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--sampler-proposal",
        choices=("single_site", "exchange", "mixed"),
        default="mixed",
    )
    parser.add_argument("--exchange-probability", type=float, default=0.75)
    parser.add_argument("--cost-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    result = run_adaptive_diagonal(
        baseline_result=args.baseline_result,
        baseline_state=args.baseline_state,
        output=args.output,
        output_state=args.output_state,
        nsamples=args.samples,
        burn_in=args.burn_in,
        sweeps_between=args.sweeps_between,
        chains=args.chains,
        sampler_proposal=args.sampler_proposal,
        exchange_probability=args.exchange_probability,
        cost_weight=args.cost_weight,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
