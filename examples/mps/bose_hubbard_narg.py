#!/usr/bin/env python3
"""Run open-chain Bose-Hubbard NARG and optional fixed-number ED."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import BoseHubbardNARG, bose_hubbard_observables, exact_bose_hubbard


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=6, help="Number of lattice sites.")
    parser.add_argument("-N", "--nbosons", type=int, default=None, help="Total boson number.")
    parser.add_argument("--nmax", type=int, default=None, help="Maximum bosons per site.")
    parser.add_argument("-t", "--hopping", type=float, default=1.0, help="Nearest-neighbor hopping.")
    parser.add_argument("-U", "--onsite-u", type=float, default=1.0, help="On-site interaction U.")
    parser.add_argument("--sweep-u", type=float, nargs="+", default=None, help="Sweep U values.")
    parser.add_argument("--mu", type=float, default=0.0, help="Chemical potential.")
    parser.add_argument("-D", "--bond-dim", type=int, default=32, help="Retained NARG states.")
    parser.add_argument("--nroots", type=int, default=1, help="Number of lowest roots.")
    parser.add_argument("--no-ed", action="store_true", help="Skip exact diagonalization.")
    return parser.parse_args()


def main():
    args = parse_args()
    nbosons = args.nsites if args.nbosons is None else int(args.nbosons)
    nmax = nbosons if args.nmax is None else int(args.nmax)

    if args.sweep_u:
        print(
            f"# Bose-Hubbard open-chain sweep: L={args.nsites}, N={nbosons}, "
            f"nmax={nmax}, D={args.bond_dim}, t={args.hopping:g}"
        )
        if args.no_ed:
            print("U/t      E_NARG          f0_NARG  var_NARG edge_NARG")
        else:
            print(
                "U/t      E_ED            E_NARG          err       "
                "f0_ED    f0_NARG  var_ED   var_NARG gap_ED"
            )
        for onsite_u in args.sweep_u:
            result = BoseHubbardNARG(
                nsites=args.nsites,
                nbosons=nbosons,
                t=args.hopping,
                U=onsite_u,
                mu=args.mu,
                nmax=nmax,
                D=args.bond_dim,
            ).run(nroots=1)
            narg_obs = result.observables[0]
            if args.no_ed:
                print(
                    f"{onsite_u:4.1f} {result.energies[0]:16.9f} "
                    f"{narg_obs.condensate_fraction:8.5f} "
                    f"{narg_obs.average_number_variance:8.5f} "
                    f"{narg_obs.edge_correlation:9.5f}"
                )
                continue
            ed, ed_vectors, basis = exact_bose_hubbard(
                args.nsites,
                nbosons,
                t=args.hopping,
                U=onsite_u,
                mu=args.mu,
                nmax=nmax,
                nroots=1,
            )
            ed_minus, _, _ = exact_bose_hubbard(
                args.nsites,
                nbosons - 1,
                t=args.hopping,
                U=onsite_u,
                mu=args.mu,
                nmax=nmax,
                nroots=1,
            )
            ed_plus, _, _ = exact_bose_hubbard(
                args.nsites,
                nbosons + 1,
                t=args.hopping,
                U=onsite_u,
                mu=args.mu,
                nmax=nmax,
                nroots=1,
            )
            ed_obs = bose_hubbard_observables(ed_vectors[:, 0], basis)
            gap = ed_plus[0] + ed_minus[0] - 2.0 * ed[0]
            print(
                f"{onsite_u:4.1f} {ed[0]:16.9f} {result.energies[0]:16.9f} "
                f"{result.energies[0] - ed[0]:9.2e} "
                f"{ed_obs.condensate_fraction:8.5f} "
                f"{narg_obs.condensate_fraction:8.5f} "
                f"{ed_obs.average_number_variance:8.5f} "
                f"{narg_obs.average_number_variance:8.5f} {gap:8.5f}"
            )
        return

    result = BoseHubbardNARG(
        nsites=args.nsites,
        nbosons=nbosons,
        t=args.hopping,
        U=args.onsite_u,
        mu=args.mu,
        nmax=nmax,
        D=args.bond_dim,
    ).run(nroots=args.nroots)

    print(
        f"Bose-Hubbard open chain: L={args.nsites}, N={nbosons}, "
        f"nmax={nmax}, t={args.hopping:g}, U={args.onsite_u:g}, mu={args.mu:g}"
    )
    print(f"NARG D={args.bond_dim}, nroots={args.nroots}")
    for root, energy in enumerate(result.energies):
        print(f"NARG root {root}: E = {energy:.12f}")
        obs = result.observables[root]
        print(
            f"  f0={obs.condensate_fraction:.6f} "
            f"var(n)={obs.average_number_variance:.6f} "
            f"<b1^dag bL>={obs.edge_correlation:.6f}"
        )
    print("growth:")
    for step in result.steps:
        print(
            f"  site={step.site:2d} product_dim={step.product_dim:5d} "
            f"kept={step.kept:4d} lowest={step.lowest_energy:.12f}"
        )

    if not args.no_ed:
        ed, ed_vectors, basis = exact_bose_hubbard(
            args.nsites,
            nbosons,
            t=args.hopping,
            U=args.onsite_u,
            mu=args.mu,
            nmax=nmax,
            nroots=args.nroots,
        )
        print(f"ED Hilbert dimension={len(basis)}")
        for root, energy in enumerate(ed):
            delta = (
                result.energies[root] - energy
                if root < len(result.energies)
                else np.nan
            )
            print(f"ED root {root}:   E = {energy:.12f}   NARG-ED = {delta:.3e}")
            obs = bose_hubbard_observables(ed_vectors[:, root], basis)
            print(
                f"  f0={obs.condensate_fraction:.6f} "
                f"var(n)={obs.average_number_variance:.6f} "
                f"<b1^dag bL>={obs.edge_correlation:.6f}"
            )


if __name__ == "__main__":
    main()
