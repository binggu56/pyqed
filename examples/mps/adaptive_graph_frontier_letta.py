"""Oracle-free adaptive graph LETTA on a small frustrated ladder.

The initial graph contains only horizontal ties.  Current-state samples rank
the missing vertical and diagonal ties by connected correlation plus a local-
energy residual, with an explicit exact-frontier cost penalty.  Each shortlisted
tie is embedded without changing the state, relaxed briefly, and accepted only
after a fresh exact energy contraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA, adaptive_tie_graph_step, tie_edges


def run_adaptive_ladder(
    *,
    ncols=3,
    j2=0.5,
    bond_dim=2,
    initial_sweeps=2,
    samples=512,
    seed=7,
):
    nrows = 2
    nsites = nrows * int(ncols)
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    horizontal = tuple(
        edge for edge in nearest if edge[0] // ncols == edge[1] // ncols
    )
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, float(j2)) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets_from_edges(nsites, horizontal),
        bond_dim=bond_dim,
        seed=seed,
        frontier_backend="identity_block",
    )
    state.run(nsweeps=initial_sweeps, solver="whitened")
    energy_before = state.expectation()

    result = adaptive_tie_graph_step(
        state,
        candidate_edges=nearest + diagonals,
        operations=("add",),
        shortlist=2,
        cost_weight=0.2,
        max_frontier_width=ncols,
        relaxation_sweeps=1,
        run_options={"solver": "whitened", "tol": 0.0},
        nsamples=samples,
        burn_in=100,
        sweeps_between=2,
        seed=seed + 1,
        sampler_proposal="mixed",
    )
    selected = result.selected
    return {
        "model": {
            "shape": [nrows, int(ncols)],
            "j1": 1.0,
            "j2": float(j2),
            "bond_dim": int(bond_dim),
        },
        "energy_before": float(energy_before),
        "energy_after": float(result.state.expectation()),
        "initial_ties": [list(edge) for edge in horizontal],
        "final_ties": [list(edge) for edge in tie_edges(result.state.parent_sets)],
        "sampling": {
            "nsamples": result.signal_batch.nsamples,
            "acceptance_rate": result.signal_batch.acceptance_rate,
            "local_energy_variance": result.signal_batch.local_energy_variance,
        },
        "ranked_proposals": [
            {
                "edge": list(proposal.edge),
                "correlation": proposal.signal.connected_correlation,
                "residual": proposal.signal.residual_coupling,
                "signal": proposal.signal.score,
                "delta_log_frontier_cost": proposal.delta_log_cost,
                "proxy_utility": proposal.proxy_utility,
                "peak_width_after": proposal.cost_after.peak_width,
            }
            for proposal in result.proposals
        ],
        "evaluations": [
            {
                "edge": list(evaluation.proposal.edge),
                "energy_after": evaluation.energy_after,
                "energy_gain": evaluation.energy_gain,
                "fresh_energy_check_passed": (
                    evaluation.fresh_energy_check_passed
                ),
                "accepted": evaluation.accepted,
            }
            for evaluation in result.evaluations
        ],
        "selected_edge": None if selected is None else list(selected.proposal.edge),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--initial-sweeps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_adaptive_ladder(
        ncols=args.ncols,
        j2=args.j2,
        bond_dim=args.bond_dim,
        initial_sweeps=args.initial_sweeps,
        samples=args.samples,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
