#!/usr/bin/env python3
"""Benchmark PEPS frontier batching, caches, CTMRG warm starts, and U(1)."""

from __future__ import annotations

import argparse
from time import perf_counter

from pyqed.lattice import SpinHalfSite
from pyqed.peps import PEPS, U1PEPS
from pyqed.tn import Hamiltonian


def heisenberg(shape):
    rows, cols = shape
    sites = tuple(SpinHalfSite() for _ in range(rows * cols))
    hamiltonian = Hamiltonian(sites)
    for row in range(rows):
        for col in range(cols):
            first = row * cols + col
            neighbors = []
            if col + 1 < cols:
                neighbors.append(first + 1)
            if row + 1 < rows:
                neighbors.append(first + cols)
            for second in neighbors:
                for operator in ("X", "Y", "Z"):
                    hamiltonian.add_product(
                        0.25,
                        (first, operator),
                        (second, operator),
                    )
    return sites, hamiltonian


def timed(function):
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def u1_state(shape, seed):
    rows, cols = shape
    sites = tuple(SpinHalfSite() for _ in range(rows * cols))
    charges = {}
    targets = []
    for row in range(rows):
        for col in range(cols):
            degree = (
                (row > 0)
                + (row + 1 < rows)
                + (col > 0)
                + (col + 1 < cols)
            )
            targets.append(1 if degree % 2 == 0 else 0)
            if col + 1 < cols:
                charges[((row, col), (row, col + 1))] = (-1, 1)
            if row + 1 < rows:
                charges[((row, col), (row + 1, col))] = (-1, 1)
    return U1PEPS.random(
        sites,
        shape=shape,
        bond_charges=charges,
        target_charges=targets,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sites, hamiltonian = heisenberg((args.size, args.size))
    state = PEPS.random(
        sites,
        shape=(args.size, args.size),
        D=args.D,
        seed=args.seed,
        normalize=False,
    )
    first, first_time = timed(
        lambda: state.expectation(
            hamiltonian,
            max_bond=args.chi,
            workers=1,
            return_info=True,
        )
    )
    cached, cached_time = timed(
        lambda: state.expectation(
            hamiltonian,
            max_bond=args.chi,
            workers=args.workers,
            return_info=True,
        )
    )
    cold, cold_time = timed(lambda: state.ctmrg(chi=args.chi))
    warm, warm_time = timed(lambda: state.ctmrg(chi=args.chi))

    u1 = u1_state((3, 4), args.seed)
    u1_frontier, frontier_time = timed(
        lambda: u1.norm_squared(return_info=True)
    )
    u1_reference, reference_time = timed(
        lambda: u1.norm_squared(method="enumerate", return_info=True)
    )

    print(
        f"dense {args.size}x{args.size} D={args.D} chi={args.chi}: "
        f"first={first_time:.4f}s cached+{args.workers}w={cached_time:.4f}s "
        f"speedup={first_time / cached_time:.2f}x dE={abs(first[0] - cached[0]):.3e}"
    )
    print(
        f"  channels={cached[1]['frontier_channels']} "
        f"cache_hits={cached[1]['layer_cache_hits']} "
        f"cache_misses={cached[1]['layer_cache_misses']}"
    )
    print(
        f"CTMRG: cold={cold_time:.4f}s/{len(cold.history)} iterations "
        f"warm={warm_time:.4f}s/{len(warm.history)} iterations "
        f"speedup={cold_time / warm_time:.2f}x"
    )
    print(
        f"U1 3x4: frontier={frontier_time:.4f}s "
        f"enumerate={reference_time:.4f}s "
        f"speedup={reference_time / frontier_time:.2f}x "
        f"error={abs(u1_frontier[0] - u1_reference[0]):.3e}"
    )
    print(
        f"  active_frontiers={u1_frontier[1]['max_active_frontiers']} "
        f"full_configurations={u1_reference[1]['configurations']}"
    )


if __name__ == "__main__":
    main()
