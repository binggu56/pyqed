#!/usr/bin/env python3
"""
Non-Abelian DMRG calculation for a finite 2D spinful Hubbard lattice.

The 2D square lattice is mapped to a one-dimensional MPS order, then encoded as
a long-range AutoMPO with Jordan-Wigner strings.  This is meant as a compact
comparison point for ``hubbard_2d_narg.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hubbard_2d_ed import square_lattice_bonds
from hubbard_2d_narg import lattice_site_order
from pyqed.mps.nonabelian import (
    AutoMPO,
    SweepDriver,
    build_random_spatial_mps,
    reduced_spatial_fermion_annihilation,
    spatial_double_occupancy,
    spatial_number,
)


def build_2d_hubbard_mpo(
    sites,
    lx: int,
    ly: int,
    *,
    hopping_t: float,
    onsite_u: float,
    chemical_potential: float = 0.0,
    ordering: str = "row-major",
    periodic_x: bool = False,
    periodic_y: bool = False,
):
    """Build a 2D Hubbard AutoMPO on the supplied MPS site order."""
    order = lattice_site_order(lx, ly, ordering=ordering)
    lattice_to_orbital = {site: orbital for orbital, site in enumerate(order)}

    autompo = AutoMPO.from_sites(sites)
    number = spatial_number(dtype=float)
    doublon = spatial_double_occupancy(dtype=float)
    reduced_annihilation = reduced_spatial_fermion_annihilation(dtype=float)
    reduced_creation = reduced_annihilation.adjoint()

    for site in range(len(sites)):
        if chemical_potential:
            autompo.add_onsite(site, number, coeff=-float(chemical_potential))
        if onsite_u:
            autompo.add_onsite(site, doublon, coeff=float(onsite_u))

    for i, j in square_lattice_bonds(lx, ly, periodic_x=periodic_x, periodic_y=periodic_y):
        p = lattice_to_orbital[i]
        q = lattice_to_orbital[j]
        left, right = sorted((p, q))
        autompo.add_fermionic_reduced_bilinear(
            left,
            reduced_creation,
            right,
            reduced_annihilation,
            coeff=-float(hopping_t),
        )
        autompo.add_fermionic_reduced_bilinear(
            left,
            reduced_annihilation,
            right,
            reduced_creation,
            coeff=+float(hopping_t),
        )

    return autompo.build()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4, help="Number of sites along x.")
    parser.add_argument("--ly", type=int, default=4, help="Number of sites along y.")
    parser.add_argument("-t", "--hopping", type=float, default=1.0, help="Nearest-neighbor hopping t.")
    parser.add_argument("-U", "--onsite-u", type=float, default=4.0, help="On-site repulsion U.")
    parser.add_argument("--mu", type=float, default=0.0, help="Chemical potential.")
    parser.add_argument("--periodic-x", action="store_true", help="Use periodic boundary conditions in x.")
    parser.add_argument("--periodic-y", action="store_true", help="Use periodic boundary conditions in y.")
    parser.add_argument("--ordering", choices=("row-major", "snake"), default="snake")
    parser.add_argument("--max-bond", type=int, default=16, help="Maximum DMRG bond multiplicity.")
    parser.add_argument("--bond-multiplicity", type=int, default=2, help="Initial random MPS bond multiplicity.")
    parser.add_argument("--nsweeps", type=int, default=2, help="Number of two-site DMRG sweeps.")
    parser.add_argument("--cutoff", type=float, default=1.0e-10, help="SVD truncation cutoff.")
    parser.add_argument("--seed", type=int, default=7, help="Random MPS seed.")
    parser.add_argument("--profile", action="store_true", help="Collect timing profile data.")
    return parser.parse_args()


def main():
    args = parse_args()
    nsites = int(args.lx) * int(args.ly)
    start = time.perf_counter()
    sites = build_random_spatial_mps(
        nsites,
        seed=int(args.seed),
        bond_multiplicity=int(args.bond_multiplicity),
    )
    mpo = build_2d_hubbard_mpo(
        sites,
        args.lx,
        args.ly,
        hopping_t=args.hopping,
        onsite_u=args.onsite_u,
        chemical_potential=args.mu,
        ordering=args.ordering,
        periodic_x=args.periodic_x,
        periodic_y=args.periodic_y,
    )
    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=int(args.nsweeps),
        mpo_factors=mpo,
        max_bond=int(args.max_bond),
        cutoff=float(args.cutoff),
        profile=bool(args.profile),
    )
    driver.run()
    elapsed = time.perf_counter() - start

    print(
        f"2D Hubbard DMRG {args.lx}x{args.ly}, "
        f"N={nsites}, t={args.hopping:g}, U={args.onsite_u:g}, "
        f"mu={args.mu:g}, ordering={args.ordering}"
    )
    print(
        f"max_bond={args.max_bond}, nsweeps={args.nsweeps}, "
        f"bond_multiplicity={args.bond_multiplicity}, seed={args.seed}"
    )
    for idx, entry in enumerate(driver.history):
        energy = entry.get("energy")
        direction = entry.get("direction")
        if energy is not None:
            print(f"sweep {idx:02d} {direction}: E = {float(np.real(energy)):.12f}")
    print(f"DMRG best E = {float(np.real(driver.last_energy)):.12f}")
    print(f"converged={driver.converged}, completed={driver.ncompleted}, time_s={elapsed:.3f}")


if __name__ == "__main__":
    main()
