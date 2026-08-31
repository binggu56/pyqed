#!/usr/bin/env python3
"""Raster-ordered frontier LETTA for the open 3x3x3 Ising model.

The Hamiltonian convention is

    H = -J sum_<ij> Z_i Z_j - g sum_i X_i,

with raster index ``i = x + nx * (y + ny * z)``.  The x-direction bonds are
adjacent along the virtual backbone.  By default only the nonconsecutive y/z
bonds are physical ties, avoiding redundant x ties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed.lattice import SpinHalfSite
from pyqed.letta import FrontierLETTA
from pyqed.tn import Hamiltonian


SHAPE = (3, 3, 3)


def raster_index(x, y, z, shape=SHAPE):
    """Return the x-fastest raster index for ``(x, y, z)``."""

    nx, ny, nz = (int(length) for length in shape)
    x, y, z = int(x), int(y), int(z)
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
        raise ValueError("coordinate lies outside the lattice")
    return x + nx * (y + ny * z)


def cubic_open_bonds(shape=SHAPE):
    """Return labeled positive-direction bonds of an open cubic lattice."""

    nx, ny, nz = (int(length) for length in shape)
    bonds = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                site = raster_index(x, y, z, shape)
                if x + 1 < nx:
                    bonds.append((site, raster_index(x + 1, y, z, shape), "x"))
                if y + 1 < ny:
                    bonds.append((site, raster_index(x, y + 1, z, shape), "y"))
                if z + 1 < nz:
                    bonds.append((site, raster_index(x, y, z + 1, shape), "z"))
    return tuple(bonds)


def ising_hamiltonian(*, coupling=1.0, field=1.5, shape=SHAPE):
    """Build ``-J ZZ - g X`` using the canonical site/Hamiltonian API."""

    nsites = int(np.prod(shape))
    sites = tuple(SpinHalfSite() for _ in range(nsites))
    hamiltonian = Hamiltonian(sites)
    for left, right, _axis in cubic_open_bonds(shape):
        hamiltonian.add_product(-float(coupling), (left, "Z"), (right, "Z"))
    for site in range(nsites):
        hamiltonian.add_product(-float(field), (site, "X"))
    return hamiltonian


def raster_tie_graph(*, tie_x=False, shape=SHAPE):
    """Tie transverse bonds; optionally include redundant raster-adjacent x bonds."""

    return tuple(
        (left, right)
        for left, right, axis in cubic_open_bonds(shape)
        if tie_x or axis != "x"
    )


def history_rows(state):
    """Return a JSON-safe directional-sweep summary."""

    rows = []
    for index, row in enumerate(state.history, start=1):
        rows.append(
            {
                "pass": index,
                "direction": row.get("direction"),
                "energy": float(row["energy"]),
                "delta_energy": float(row["delta_energy"]),
                "solver_failures": int(row["solver_failures"]),
            }
        )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", dest="coupling", type=float, default=1.0)
    parser.add_argument("--g", dest="field", type=float, default=1.5)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--warm-start-d1",
        action="store_true",
        help="converge D=1, embed it exactly at the requested D, then continue",
    )
    parser.add_argument(
        "--d1-sweeps",
        type=int,
        default=30,
        help="maximum D=1 sweeps used by --warm-start-d1",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tie-x", action="store_true")
    parser.add_argument("--fixed-bond", action="store_true")
    parser.add_argument(
        "--solver",
        choices=("matrix_free", "metric_orthonormal", "direct"),
        default="matrix_free",
    )
    parser.add_argument(
        "--frontier-backend",
        choices=("identity_block", "compressed"),
        default="identity_block",
    )
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--draw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.coupling <= 0.0 or args.field < 0.0:
        parser.error("J must be positive and g must be nonnegative")
    if args.D < 1 or args.sweeps < 1 or args.d1_sweeps < 1:
        parser.error("D, sweeps, and d1-sweeps must be positive")
    if args.warm_start_d1 and args.D < 2:
        parser.error("--warm-start-d1 requires D >= 2")
    if args.tol < 0.0:
        parser.error("tolerance must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    bonds = cubic_open_bonds()
    ties = raster_tie_graph(tie_x=args.tie_x)
    hamiltonian = ising_hamiltonian(
        coupling=args.coupling,
        field=args.field,
    )

    setup_start = perf_counter()
    initial_D = 1 if args.warm_start_d1 else args.D
    state = FrontierLETTA(
        hamiltonian,
        graph=ties,
        D=initial_D,
        adaptive_bond=False if args.warm_start_d1 else not args.fixed_bond,
        seed=args.seed,
        frontier_backend=args.frontier_backend,
        path_optimizer="greedy",
    )
    setup_seconds = perf_counter() - setup_start

    print(
        "3x3x3 transverse-field Ising frontier LETTA\n"
        f"H=-{args.coupling:g} sum(ZZ)-{args.field:g} sum(X), open boundary\n"
        "ordering: raster i=x+3y+9z (x fastest)\n"
        f"sites=27 Hamiltonian_bonds={len(bonds)} ties={len(ties)} "
        f"D_max={args.D} adaptive_bond={not args.fixed_bond} "
        f"warm_start_D1={args.warm_start_d1}\n"
        f"backend={args.frontier_backend} solver={args.solver}"
    )
    print(f"initial energy = {state.energy:.12f}")

    d1_stage = None
    if args.warm_start_d1:
        d1_start = perf_counter()
        state.run(
            nsweeps=args.d1_sweeps,
            tol=args.tol,
            solver=args.solver,
            gauge="auto",
            gauge_weight="probability",
            environment_cache="checkpointed",
            verbose=args.verbose,
        )
        d1_seconds = perf_counter() - d1_start
        d1_energy = float(state.energy)
        d1_history = history_rows(state)
        print(
            f"converged D=1 energy = {d1_energy:.12f} "
            f"({len(d1_history)} sweeps, {d1_seconds:.3f} s)"
        )

        target_bonds = (1,) + (args.D,) * (len(state.dims) - 1) + (1,)
        expansion_start = perf_counter()
        expansions = state.expand_bond_dims(
            target_bonds,
            direction="right",
            strategy="residual",
            scale=1.0e-3,
            seed=args.seed + 1,
        )
        expansion_seconds = perf_counter() - expansion_start
        expansion_error = abs(float(state.energy) - d1_energy)
        state.D = args.D
        state.adaptive_bond = not args.fixed_bond
        state._maximum_bond_dims = target_bonds
        print(
            f"expanded D=1 -> D={args.D}: cuts={len(expansions)} "
            f"|delta E|={expansion_error:.3e} ({expansion_seconds:.3f} s)"
        )
        d1_stage = {
            "energy": d1_energy,
            "sweeps": len(d1_history),
            "seconds": d1_seconds,
            "history": d1_history,
            "expanded_cuts": len(expansions),
            "expansion_seconds": expansion_seconds,
            "expansion_energy_error": expansion_error,
        }

    sweep_start = perf_counter()
    state.run(
        nsweeps=args.sweeps,
        tol=args.tol,
        solver=args.solver,
        gauge="auto",
        gauge_weight="probability",
        environment_cache="checkpointed",
        verbose=args.verbose,
    )
    sweep_seconds = perf_counter() - sweep_start

    print(f"energy = {state.energy:.12f}")
    print(f"energy per site = {state.energy / 27:.12f}")
    print(f"bond dimensions = {state.bond_dims}")
    print(f"peak frontier elements = {state.peak_frontier_elements}")
    print(f"setup seconds = {setup_seconds:.3f}")
    print(f"sweep seconds = {sweep_seconds:.3f}")

    if args.draw is not None:
        args.draw.parent.mkdir(parents=True, exist_ok=True)
        state.draw(args.draw)
        print(f"saved {args.draw}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "transverse_field_ising",
            "shape": list(SHAPE),
            "boundary": "open",
            "ordering": "raster_x_fastest_i=x+3y+9z",
            "J": args.coupling,
            "g": args.field,
            "D_max": args.D,
            "adaptive_bond": not args.fixed_bond,
            "warm_start_D1": args.warm_start_d1,
            "D1_stage": d1_stage,
            "hamiltonian_bonds": len(bonds),
            "tie_edges": len(ties),
            "tie_x": args.tie_x,
            "frontier_backend": args.frontier_backend,
            "solver": args.solver,
            "energy": float(state.energy),
            "energy_per_site": float(state.energy / 27),
            "bond_dims": list(state.bond_dims),
            "parameters": int(state.nparameters),
            "peak_frontier_elements": int(state.peak_frontier_elements),
            "setup_seconds": setup_seconds,
            "sweep_seconds": sweep_seconds,
            "history": history_rows(state),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
