#!/usr/bin/env python3
"""Compare dense, factorized, and adaptively selected ties on 6x6 J1-J2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.benchmark_frontier_letta_u1_4x4 import (
    _neel_cores,
    _u1_mps_run,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import (
    abelian_frontier_tied_letta_from_mps,
    adapt_tie_graph,
    conditional_frontier_letta_from_mps,
    tie_edges,
    tie_frontier_cost,
)


def _cost_record(state):
    cost = tie_frontier_cost(
        state.dims,
        state.parent_sets,
        bond_dims=state.bond_dims,
    )
    return {
        "peak_width": cost.peak_width,
        "peak_physical_states": cost.peak_physical_states,
        "peak_norm_elements": cost.peak_norm_elements,
        "total_norm_elements": cost.total_norm_elements,
    }


def benchmark(
    *,
    D=4,
    mps_sweeps=12,
    adaptive_steps=4,
    adaptive_polish_sweeps=1,
    samples=128,
    factor_sweeps=1,
    paired_sweeps=1,
    dense_sweeps=1,
    seed=7,
):
    nrows = ncols = 6
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, 0.5) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)

    start = perf_counter()
    _solver, mps, layout, mps_record = _u1_mps_run(
        hamiltonian.to_mpo().compress(),
        _neel_cores(nsites),
        bond_dim=D,
        sweeps=mps_sweeps,
        tolerance=1.0e-7,
    )
    mps_seconds = perf_counter() - start
    initial_energy = float(mps_record["energy"])
    common = {
        "abelian_layout": layout,
        "frontier_backend": "identity_block",
        "route_memory": 32,
        "action_memory": 32,
        "seed": seed,
    }

    empty = ((),) * nsites
    adaptive = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        empty,
        mps.factors,
        **common,
    )
    start = perf_counter()
    adaptive_run = adapt_tie_graph(
        adaptive,
        nsteps=adaptive_steps,
        candidate_edges=nearest,
        operations=("add",),
        shortlist=1,
        cost_weight=0.03,
        max_frontier_width=2,
        relaxation_sweeps=1,
        nsamples=samples,
        burn_in=40,
        sweeps_between=1,
        seed=seed,
        run_options={
            "solver": "matrix_free",
            "eig_tol": 1.0e-5,
            "maxiter": 40,
            "gauge": None,
            "environment_cache": "checkpointed",
            "environment_memory": 32,
        },
    )
    adaptive_seconds = perf_counter() - start
    adaptive = adaptive_run.state
    adaptive_selected_energy = float(adaptive.expectation())
    start = perf_counter()
    adaptive.run(
        nsweeps=adaptive_polish_sweeps,
        tol=0.0,
        solver="matrix_free",
        eig_tol=1.0e-5,
        maxiter=40,
        gauge=None,
        environment_cache="checkpointed",
        environment_memory=32,
    )
    adaptive_polish_seconds = perf_counter() - start

    all_nearest = parent_sets_from_edges(nsites, nearest)
    factorized = conditional_frontier_letta_from_mps(
        hamiltonian,
        all_nearest,
        mps.factors,
        **common,
    )
    factor_initial = float(factorized.expectation())
    start = perf_counter()
    factorized.run(
        nsweeps=factor_sweeps,
        tol=0.0,
        metric_tol=1.0e-10,
        environment_cache="checkpointed",
    )
    factor_seconds = perf_counter() - start

    paired = conditional_frontier_letta_from_mps(
        hamiltonian,
        all_nearest,
        mps.factors,
        parent_group_size=2,
        **common,
    )
    paired_initial = float(paired.expectation())
    start = perf_counter()
    paired.run(
        nsweeps=paired_sweeps,
        tol=0.0,
        metric_tol=1.0e-10,
        environment_cache="checkpointed",
    )
    paired_seconds = perf_counter() - start

    dense = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        all_nearest,
        mps.factors,
        **common,
    )
    dense_initial = float(dense.expectation())
    start = perf_counter()
    dense.run(
        nsweeps=dense_sweeps,
        tol=0.0,
        solver="matrix_free",
        eig_tol=1.0e-5,
        maxiter=40,
        gauge=None,
        environment_cache="checkpointed",
        environment_memory=32,
    )
    dense_seconds = perf_counter() - start

    selected = []
    for step in adaptive_run.steps:
        if step.selected is None:
            break
        selected.append(
            {
                "edge": list(step.selected.proposal.edge),
                "signal": step.selected.proposal.signal.score,
                "energy": step.selected.energy_after,
                "energy_gain": step.selected.energy_gain,
            }
        )
    return {
        "model": "6x6 J1-J2 Heisenberg, J2/J1=0.5, U(1)",
        "D": D,
        "mps": {
            "energy": initial_energy,
            "seconds": mps_seconds,
            "sweeps": mps_sweeps,
        },
        "adaptive_ties": {
            "selection_energy": adaptive_selected_energy,
            "energy": float(adaptive.expectation()),
            "energy_gain": initial_energy - float(adaptive.expectation()),
            "selection_seconds": adaptive_seconds,
            "polish_seconds": adaptive_polish_seconds,
            "polish_sweeps": adaptive_polish_sweeps,
            "ties": [list(edge) for edge in tie_edges(adaptive.parent_sets)],
            "selected": selected,
            "cost": _cost_record(adaptive),
        },
        "factorized_all_nearest": {
            "initial_energy": factor_initial,
            "energy": float(factorized.expectation()),
            "energy_gain": initial_energy - float(factorized.expectation()),
            "seconds": factor_seconds,
            "sweeps": factor_sweeps,
            "stored_parameters": factorized.nparameters,
            "unfactorized_parameters": factorized.unfactorized_nparameters,
            "storage_ratio": factorized.compression_ratio,
            "cost": _cost_record(factorized),
        },
        "pair_grouped_all_nearest": {
            "initial_energy": paired_initial,
            "energy": float(paired.expectation()),
            "energy_gain": initial_energy - float(paired.expectation()),
            "seconds": paired_seconds,
            "sweeps": paired_sweeps,
            "stored_parameters": paired.nparameters,
            "unfactorized_parameters": paired.unfactorized_nparameters,
            "storage_ratio": paired.compression_ratio,
            "cost": _cost_record(paired),
        },
        "dense_all_nearest": {
            "initial_energy": dense_initial,
            "energy": float(dense.expectation()),
            "energy_gain": initial_energy - float(dense.expectation()),
            "seconds": dense_seconds,
            "sweeps": dense_sweeps,
            "stored_parameters": int(
                sum(mask.sum() for mask in dense.local_masks)
            ),
            "cost": _cost_record(dense),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=12)
    parser.add_argument("--adaptive-steps", type=int, default=4)
    parser.add_argument("--adaptive-polish-sweeps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--factor-sweeps", type=int, default=1)
    parser.add_argument("--paired-sweeps", type=int, default=1)
    parser.add_argument("--dense-sweeps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/factorized_adaptive_ties_6x6.json"),
    )
    args = parser.parse_args()
    result = benchmark(
        D=args.D,
        mps_sweeps=args.mps_sweeps,
        adaptive_steps=args.adaptive_steps,
        adaptive_polish_sweeps=args.adaptive_polish_sweeps,
        samples=args.samples,
        factor_sweeps=args.factor_sweeps,
        paired_sweeps=args.paired_sweeps,
        dense_sweeps=args.dense_sweeps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
