#!/usr/bin/env python3
"""Rank-convergence test for boundary-TT frontier messages on 6x6 J1-J2."""

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
    AbelianFrontierTiedLETTA,
    abelian_frontier_tied_letta_from_mps,
)


def _timed(callable_):
    start = perf_counter()
    value = callable_()
    return value, perf_counter() - start


def benchmark(*, D=4, mps_sweeps=12, preparation_sweeps=4, ranks=(2, 4, 8, 16)):
    nrows = ncols = 6
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, 0.5) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
    parents = parent_sets_from_edges(nsites, nearest)

    (_solver, mps, layout, mps_record), mps_seconds = _timed(
        lambda: _u1_mps_run(
            hamiltonian.to_mpo().compress(),
            _neel_cores(nsites),
            bond_dim=D,
            sweeps=mps_sweeps,
            tolerance=1.0e-7,
        )
    )
    exact = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        mps.factors,
        abelian_layout=layout,
        frontier_backend="identity_block",
        route_memory=32,
        action_memory=32,
    )
    _, preparation_seconds = _timed(
        lambda: exact.run(
            nsweeps=preparation_sweeps,
            tol=0.0,
            solver="matrix_free",
            eig_tol=1.0e-5,
            maxiter=40,
            gauge=None,
            environment_cache="checkpointed",
            environment_memory=32,
        )
    )
    gauge_updates, gauge_seconds = _timed(
        lambda: exact.canonicalize_frontier_gauge(weighting="probability")
    )
    reference_energy, reference_seconds = _timed(exact.expectation)

    rows = []
    for rank in ranks:
        print(f"boundary TT rank {rank}", flush=True)
        def construct():
            return AbelianFrontierTiedLETTA(
                hamiltonian,
                parents,
                abelian_layout=layout,
                bond_dims=layout.bond_dims,
                tensors=[tensor.copy() for tensor in exact.tensors],
                frontier_backend="tensor_train",
                max_rank=rank,
                rtol=0.0,
                atol=0.0,
                transfer_max_rank=None,
                transfer_rtol=0.0,
                transfer_atol=0.0,
                tt_norm_backend="exact",
                tt_hermitize=True,
                tt_channels="component",
                tt_gauge=True,
            )

        state, setup_seconds = _timed(construct)
        energy = float(state.energy)
        diagnostics = state.tt_diagnostics["hamiltonian"]
        rows.append(
            {
                "rank": rank,
                "energy": float(energy),
                "absolute_energy_error": float(abs(energy - reference_energy)),
                "error_per_site": float(abs(energy - reference_energy) / nsites),
                "setup_and_first_expectation_seconds": setup_seconds,
                "peak_tt_message_elements": int(
                    diagnostics.peak_message_storage_elements
                ),
                "peak_dense_equivalent_elements": int(
                    diagnostics.peak_dense_message_elements
                ),
                "dense_to_tt_message_ratio": float(
                    diagnostics.peak_dense_message_elements
                    / diagnostics.peak_message_storage_elements
                ),
                "maximum_relative_discarded_weight": float(
                    diagnostics.max_relative_discarded_weight
                ),
                "norm_exact": bool(state.norm_contraction_is_exact),
                "hamiltonian_exact": bool(
                    state.hamiltonian_contraction_is_exact
                ),
                "channel_grouping": state.tt_channels,
                "channel_blocks": len(state._hamiltonian_frontier._engines),
                "gauge_cuts_applied": sum(
                    update.applied for update in state.tt_gauge_updates
                ),
            }
        )
        print(
            f"  E={energy:.12f} error={abs(energy-reference_energy):.3e} "
            f"stored={diagnostics.peak_message_storage_elements}",
            flush=True,
        )
        state.close()

    return {
        "model": "6x6 J1-J2 Heisenberg, J2/J1=0.5, U(1)",
        "D": D,
        "mps": {
            "energy": float(mps_record["energy"]),
            "sweeps": mps_sweeps,
            "seconds": mps_seconds,
        },
        "prepared_exact_state": {
            "energy": float(reference_energy),
            "sweeps": preparation_sweeps,
            "preparation_seconds": preparation_seconds,
            "gauge_seconds": gauge_seconds,
            "gauge_cuts_applied": sum(update.applied for update in gauge_updates),
            "gauge_cuts_total": len(gauge_updates),
            "expectation_seconds": reference_seconds,
            "peak_frontier_elements": exact.peak_frontier_elements,
        },
        "boundary_tt": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=12)
    parser.add_argument("--preparation-sweeps", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/boundary_tt_frontier_6x6.json"),
    )
    args = parser.parse_args()
    result = benchmark(
        D=args.D,
        mps_sweeps=args.mps_sweeps,
        preparation_sweeps=args.preparation_sweeps,
        ranks=tuple(args.ranks),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
