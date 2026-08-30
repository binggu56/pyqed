#!/usr/bin/env python3
"""U(1)-blocked purified-MPS thermodynamics of an open XXZ chain."""

from __future__ import annotations

import argparse

from pyqed.lattice import SpinHalfSite
from pyqed.mps import PurifiedMPS
from pyqed.tn import Hamiltonian


def xxz_chain(nsites, *, Jxy=1.0, Jz=1.0, field=0.0):
    sites = tuple(SpinHalfSite() for _ in range(nsites))
    hamiltonian = Hamiltonian(sites)
    for site in range(nsites - 1):
        hamiltonian.add_product(
            0.5 * float(Jxy),
            (site, "Sp"),
            (site + 1, "Sm"),
            add_hc=True,
        )
        hamiltonian.add_product(
            float(Jz),
            (site, "Sz"),
            (site + 1, "Sz"),
        )
    for site in range(nsites):
        hamiltonian.add_product(-float(field), (site, "Sz"))
    return hamiltonian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--Jxy", type=float, default=1.0)
    parser.add_argument("--Jz", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.sites < 1 or args.D < 1:
        parser.error("sites and D must be positive")
    if args.beta < 0.0 or args.step <= 0.0:
        parser.error("beta must be nonnegative and step must be positive")

    thermal = PurifiedMPS(
        xxz_chain(
            args.sites,
            Jxy=args.Jxy,
            Jz=args.Jz,
            field=args.field,
        ),
        D=args.D,
        symmetry="U1",
    ).run(
        args.beta,
        step=args.step,
        verbose=not args.quiet,
    )

    print(f"beta = {thermal.beta:.12g}")
    print(f"energy = {thermal.energy:.12f}")
    print(f"energy/site = {thermal.energy / args.sites:.12f}")
    print(f"log(Z) = {thermal.log_partition_function:.12f}")
    print(f"free energy = {thermal.free_energy:.12f}")
    print(f"bond dimensions = {thermal.bond_dims}")
    print(f"backend = {thermal.history[-1]['backend'] if thermal.history else 'block-sparse'}")


if __name__ == "__main__":
    main()
