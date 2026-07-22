#!/usr/bin/env python3
"""Two-site DMRG benchmark for an open 2D spin-1/2 Heisenberg model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.mps import DMRG, MPS, MPO


def lattice_order(rows: int, cols: int, ordering: str) -> tuple[int, ...]:
    """Return physical row-major site labels in the requested MPS order."""

    if ordering == "row-major":
        return tuple(range(rows * cols))
    if ordering == "row-snake":
        return tuple(
            row * cols + col
            for row in range(rows)
            for col in (
                range(cols) if row % 2 == 0 else reversed(range(cols))
            )
        )
    if ordering == "column-snake":
        return tuple(
            row * cols + col
            for col in range(cols)
            for row in (
                range(rows) if col % 2 == 0 else reversed(range(rows))
            )
        )
    raise ValueError(f"unknown ordering {ordering!r}")


def ordered_nearest_neighbors(
    rows: int, cols: int, ordering: str
) -> tuple[tuple[int, int], ...]:
    """Return square-lattice bonds in MPS-chain indices."""

    order = lattice_order(rows, cols, ordering)
    physical_to_chain = {physical: chain for chain, physical in enumerate(order)}
    bonds = []
    for row in range(rows):
        for col in range(cols):
            physical = row * cols + col
            if col + 1 < cols:
                neighbor = physical + 1
                bonds.append(
                    tuple(sorted((physical_to_chain[physical], physical_to_chain[neighbor])))
                )
            if row + 1 < rows:
                neighbor = physical + cols
                bonds.append(
                    tuple(sorted((physical_to_chain[physical], physical_to_chain[neighbor])))
                )
    return tuple(sorted(bonds))


def random_mps(nsites: int, bond_dim: int, seed: int) -> MPS:
    """Construct a reproducible random open-boundary MPS."""

    ranks = tuple(
        min(bond_dim, 2 ** min(cut, nsites - cut))
        for cut in range(nsites + 1)
    )
    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(size=(ranks[site], 2, ranks[site + 1]))
        / np.sqrt(2 * ranks[site] * ranks[site + 1])
        for site in range(nsites)
    ]
    return MPS(factors, labels=["lv", "p", "rv"]).right_canonicalize()


def run_benchmark(
    *,
    rows: int,
    cols: int,
    ordering: str,
    bond_dims: tuple[int, ...],
    passes: int,
    tolerance: float,
    seed: int,
) -> dict:
    """Run warm-started two-site DMRG over an increasing bond-dimension list."""

    nsites = rows * cols
    bonds = ordered_nearest_neighbors(rows, cols, ordering)
    hamiltonian = heisenberg_local_hamiltonian(
        nsites, tuple((left, right, 1.0) for left, right in bonds)
    )
    mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    state = None
    records = []
    for bond_dim in bond_dims:
        if state is None:
            state = random_mps(nsites, bond_dim, seed)
        start = perf_counter()
        solver = DMRG(
            mpo,
            D=bond_dim,
            init_guess=state,
            nsweeps=passes,
            opt="2site",
            symmetry=False,
            not_conv_err=False,
            verbose=0,
            sweep_tol=tolerance,
            davidson_tol=min(tolerance, 1.0e-11),
            davidson_max_iter=120,
            noise=0.0,
            recenter_final=False,
            performance="auto",
        ).run()
        elapsed = perf_counter() - start
        history = [
            row
            for row in solver.sweep_history
            if row.get("direction") in {"lr", "rl"}
        ]
        delta = (
            abs(float(history[-1]["energy"]) - float(history[-2]["energy"]))
            if len(history) >= 2
            else None
        )
        stored_parameters = int(
            sum(np.asarray(factor).size for factor in solver.ground_state.factors)
        )
        record = {
            "bond_dim": bond_dim,
            "energy": float(solver.e_tot),
            "energy_per_site": float(solver.e_tot) / nsites,
            "final_delta_energy": delta,
            "directional_passes": len(history),
            "stored_parameters": stored_parameters,
            "optimization_seconds": elapsed,
            "converged": bool(solver.converged),
        }
        records.append(record)
        print(
            f"D={bond_dim:4d} E={record['energy']:.12f} "
            f"delta={delta if delta is not None else float('nan'):.3e} "
            f"parameters={stored_parameters} time={elapsed:.3f}s",
            flush=True,
        )
        state = solver.ground_state.copy()
    return {
        "model": "open spin-1/2 nearest-neighbor Heisenberg",
        "rows": rows,
        "cols": cols,
        "nsites": nsites,
        "ordering": ordering,
        "site_order": list(lattice_order(rows, cols, ordering)),
        "nearest_neighbor_bonds": len(bonds),
        "passes": passes,
        "tolerance": tolerance,
        "seed": seed,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument(
        "--ordering",
        choices=("row-major", "row-snake", "column-snake"),
        default="column-snake",
    )
    parser.add_argument("--bond-dims", type=int, nargs="+", default=(16, 32, 64, 128))
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        rows=args.rows,
        cols=args.cols,
        ordering=args.ordering,
        bond_dims=tuple(args.bond_dims),
        passes=args.passes,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
