#!/usr/bin/env python3
"""Optimize a small open-boundary square-lattice Heisenberg PEPS."""

from __future__ import annotations

import argparse

from pyqed.lattice import SpinHalfSite
from pyqed.peps import PEPS
from pyqed.tn import Hamiltonian


def heisenberg(rows, cols, *, coupling=1.0):
    sites = tuple(SpinHalfSite() for _ in range(rows * cols))
    hamiltonian = Hamiltonian(sites)
    for row in range(rows):
        for col in range(cols):
            site = row * cols + col
            neighbors = []
            if col + 1 < cols:
                neighbors.append(site + 1)
            if row + 1 < rows:
                neighbors.append(site + cols)
            for neighbor in neighbors:
                for operator in ("X", "Y", "Z"):
                    hamiltonian.add_product(
                        0.25 * coupling,
                        (site, operator),
                        (neighbor, operator),
                    )
    return sites, hamiltonian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.D, args.sweeps, args.workers) < 1:
        parser.error("rows, cols, D, sweeps, and workers must be positive")

    sites, hamiltonian = heisenberg(args.rows, args.cols)
    state = PEPS.random(
        sites,
        shape=(args.rows, args.cols),
        D=args.D,
        seed=args.seed,
        complex=True,
        contraction="exact",
    )
    initial = state.expectation(hamiltonian, method="exact")
    optimizer = state.optimize(
        hamiltonian,
        sweeps=args.sweeps,
        verbose=True,
    )
    boundary, info = state.expectation(
        hamiltonian,
        method="boundary",
        max_bond=64,
        rtol=1.0e-12,
        workers=args.workers,
        return_info=True,
    )
    ctmrg, ctm_info = state.expectation(
        hamiltonian,
        method="ctmrg",
        max_bond=64,
        rtol=1.0e-12,
        workers=args.workers,
        return_info=True,
    )

    print(f"initial energy = {initial:.12f}")
    print(f"optimized exact energy = {optimizer.energy:.12f}")
    print(f"boundary-MPS energy = {boundary:.12f}")
    print(f"CTMRG energy = {ctmrg:.12f}")
    print(f"PEPS bonds = {state.bond_dims}")
    print(f"boundary diagnostic = {info['max_relative_error']:.3e}")
    print(f"CTMRG directional spread = {ctm_info['norm']['directional_spread']:.3e}")


if __name__ == "__main__":
    main()
