#!/usr/bin/env python3
"""Finite DMRG for an open 3D transverse-field Ising lattice."""

from __future__ import annotations

import argparse
import time

import numpy as np

from pyqed.mps import DMRG, MPO, MPS


def snake_order(lx: int, ly: int, lz: int) -> list[tuple[int, int, int]]:
    """Return a continuous layer-by-layer snake through a rectangular box."""
    layer = [
        (x, y)
        for y in range(ly)
        for x in (range(lx) if y % 2 == 0 else range(lx - 1, -1, -1))
    ]
    return [
        (x, y, z)
        for z in range(lz)
        for x, y in (layer if z % 2 == 0 else reversed(layer))
    ]


def cubic_bonds(lx: int, ly: int, lz: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Nearest-neighbor bonds of an open rectangular cubic lattice."""
    bonds = []
    for z in range(lz):
        for y in range(ly):
            for x in range(lx):
                site = (x, y, z)
                if x + 1 < lx:
                    bonds.append((site, (x + 1, y, z)))
                if y + 1 < ly:
                    bonds.append((site, (x, y + 1, z)))
                if z + 1 < lz:
                    bonds.append((site, (x, y, z + 1)))
    return bonds


def ising_mpo(
    lx: int,
    ly: int,
    lz: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
) -> tuple[MPO, list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Build an exact compact MPO for ``-J sum ZZ - h sum X``."""
    order = snake_order(lx, ly, lz)
    position = {site: i for i, site in enumerate(order)}
    bonds = [
        tuple(sorted((position[a], position[b])))
        for a, b in cubic_bonds(lx, ly, lz)
    ]
    nsites = len(order)
    identity = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    start, final = ("start",), ("final",)

    def states_at_cut(cut: int):
        if cut < 0:
            return [start]
        if cut >= nsites - 1:
            return [final]
        crossing = [("bond", i, j) for i, j in bonds if i <= cut < j]
        return [start, final, *crossing]

    factors = []
    for site in range(nsites):
        left_states = states_at_cut(site - 1)
        right_states = states_at_cut(site)
        left = {state: i for i, state in enumerate(left_states)}
        right = {state: i for i, state in enumerate(right_states)}
        tensor = np.zeros((len(left), len(right), 2, 2))

        if start in left and start in right:
            tensor[left[start], right[start]] = identity
        if final in left and final in right:
            tensor[left[final], right[final]] = identity
        if start in left and final in right:
            tensor[left[start], right[final]] = -field * sigma_x

        for i, j in bonds:
            channel = ("bond", i, j)
            if site == i:
                tensor[left[start], right[channel]] = -coupling * sigma_z
            elif i < site < j:
                tensor[left[channel], right[channel]] = identity
            elif site == j:
                tensor[left[channel], right[final]] = sigma_z
        factors.append(tensor)

    return MPO(factors), order, bonds


def random_mps(nsites: int, bond_dim: int, seed: int) -> MPS:
    """Small random MPS with feasible open-boundary bond dimensions."""
    rng = np.random.default_rng(seed)
    bonds = [
        min(bond_dim, 2 ** min(cut + 1, nsites - cut - 1))
        for cut in range(nsites - 1)
    ]
    dimensions = [1, *bonds, 1]
    factors = [
        rng.normal(size=(dimensions[i], 2, dimensions[i + 1]))
        for i in range(nsites)
    ]
    return MPS(factors)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=3)
    parser.add_argument("--lz", type=int, default=3)
    parser.add_argument("-J", "--coupling", type=float, default=1.0)
    parser.add_argument("-g", "--field", type=float, default=1.0)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--initial-bond", type=int, default=4)
    parser.add_argument("--nsweeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    return parser.parse_args()


def main():
    args = parse_args()
    mpo, order, bonds = ising_mpo(
        args.lx,
        args.ly,
        args.lz,
        coupling=args.coupling,
        field=args.field,
    )
    initial = random_mps(len(order), args.initial_bond, args.seed)
    start = time.perf_counter()
    solver = DMRG(
        mpo,
        D=args.max_bond,
        init_guess=initial,
        nsweeps=args.nsweeps,
        sweep_tol=args.sweep_tol,
        davidson_tol=1.0e-8,
        davidson_max_iter=60,
        noise=1.0e-5,
        noise_decay=0.1,
        not_conv_err=False,
        verbose=0,
        performance="auto",
    ).run()
    elapsed = time.perf_counter() - start

    print(
        f"3D TFIM DMRG: {args.lx}x{args.ly}x{args.lz}, N={len(order)}, "
        f"bonds={len(bonds)}, J={args.coupling:g}, h={args.field:g}"
    )
    print(
        f"ordering=snake, MPO max bond={max(mpo.bond_orders())}, "
        f"MPS max bond={args.max_bond}, sweeps={args.nsweeps}, seed={args.seed}"
    )
    previous_energy = None
    for index, row in enumerate(solver.sweep_history, start=1):
        energy = float(row["energy"])
        energy_change = np.nan if previous_energy is None else energy - previous_energy
        print(
            f"step {index:02d} {row['direction']}: "
            f"E={energy:.12f}, dE={energy_change:.3e}"
        )
        previous_energy = energy
    print(f"ground-state energy = {float(solver.e_tot):.12f}")
    print(f"energy per site = {float(solver.e_tot) / len(order):.12f}")
    print(f"converged={solver.converged}, elapsed={elapsed:.3f} s")


if __name__ == "__main__":
    main()
