#!/usr/bin/env python3
"""Purified-MPS thermodynamics of an open transverse-field Ising chain."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.lattice import SpinHalfSite
from pyqed.mps import PurifiedMPS
from pyqed.tn import Hamiltonian


def ising_chain(nsites, *, coupling=1.0, field=1.0):
    sites = tuple(SpinHalfSite() for _ in range(nsites))
    hamiltonian = Hamiltonian(sites)
    for site in range(nsites - 1):
        hamiltonian.add_product(
            -float(coupling),
            (site, "Z"),
            (site + 1, "Z"),
        )
    for site in range(nsites):
        hamiltonian.add_product(-float(field), (site, "X"))
    return hamiltonian


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=1.0e-12)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.sites < 1 or args.D < 1:
        parser.error("sites and D must be positive")
    if args.beta < 0.0 or args.step <= 0.0:
        parser.error("beta must be nonnegative and step must be positive")
    return args


def main():
    args = parse_args()
    hamiltonian = ising_chain(
        args.sites,
        coupling=args.J,
        field=args.g,
    )
    thermal = PurifiedMPS(
        hamiltonian,
        D=args.D,
        cutoff=args.cutoff,
    ).run(
        args.beta,
        step=args.step,
        verbose=not args.quiet,
    )

    print(f"beta = {thermal.beta:.12g}")
    print(f"temperature = {np.inf if thermal.beta == 0 else 1 / thermal.beta:.12g}")
    print(f"energy = {thermal.energy:.12f}")
    print(f"energy/site = {thermal.energy / args.sites:.12f}")
    print(f"log(Z) = {thermal.log_partition_function:.12f}")
    print(f"free energy = {thermal.free_energy:.12f}")
    print(f"bond dimensions = {thermal.bond_dims}")


if __name__ == "__main__":
    main()
