#!/usr/bin/env python3
"""Half-filled Holstein NARG with exact conditional electronic states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import SpinfulHolsteinAdiabaticElectronicNARG


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=4)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("-g", "--coupling", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=0.0)
    parser.add_argument("--ngrid", type=int, default=9)
    parser.add_argument(
        "--xmax",
        "--qmax",
        dest="xmax",
        type=float,
        default=6.0,
        help="Dimensionless sine-DVR box boundary qmax; grid spans [-qmax, qmax].",
    )
    parser.add_argument(
        "--phonon-basis",
        choices=("sine-dvr", "finite-difference"),
        default="sine-dvr",
        help="Coordinate basis for active phonon modes.",
    )
    parser.add_argument(
        "--backend",
        choices=("sequential", "joint"),
        default="sequential",
        help="Add modes one at a time or build the joint active-mode grid.",
    )
    parser.add_argument(
        "-D",
        "--bond-dim",
        type=int,
        default=64,
        help="Sequential NARG block dimension after each mode addition.",
    )
    parser.add_argument(
        "--initial-electronic-states",
        type=int,
        default=None,
        help="Initial low electronic states retained before adding modes.",
    )
    parser.add_argument(
        "--active-modes",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based local phonon modes to include. Defaults to all modes.",
    )
    parser.add_argument(
        "--mode-transform",
        choices=("local", "density-response"),
        default="local",
        help="Use local phonon modes or density-response ordered collective modes.",
    )
    parser.add_argument(
        "--ncollective",
        type=int,
        default=None,
        help="Number of collective density-response modes to retain.",
    )
    parser.add_argument(
        "--nlow-electronic",
        type=int,
        default=None,
        help="Number of low electronic eigenstates used to build response modes.",
    )
    parser.add_argument(
        "--uncentered-density-response",
        action="store_true",
        help="Keep density identity components when building response modes.",
    )
    parser.add_argument(
        "--nstates-per-point",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Conditional electronic states retained per active-mode grid point.",
    )
    parser.add_argument("--nroots", type=int, default=1)
    parser.add_argument("--skip-exact", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode_transform != "local" and args.active_modes is not None:
        raise ValueError("--active-modes is only valid with --mode-transform local.")

    mode_report = None
    active_modes = None if args.active_modes is None else tuple(args.active_modes)
    mode_transform = None
    mode_strengths = None
    if args.mode_transform == "density-response":
        seed_model = SpinfulHolsteinAdiabaticElectronicNARG(
            nsites=args.nsites,
            t=args.hopping,
            omega=args.omega,
            g=args.coupling,
            hubbard_u=args.hubbard_u,
            ngrid=args.ngrid,
            xmax=args.xmax,
            phonon_basis=args.phonon_basis,
        )
        mode_report = seed_model.density_response_mode_transform(
            nlow=args.nlow_electronic,
            center=not args.uncentered_density_response,
        )
        ncollective = args.ncollective
        if ncollective is None:
            ncollective = args.nsites
        if ncollective < 1 or ncollective > args.nsites:
            raise ValueError("--ncollective must be between 1 and nsites.")
        mode_transform = mode_report.transform[:ncollective]
        mode_strengths = mode_report.strengths[:ncollective]

    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=args.nsites,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        hubbard_u=args.hubbard_u,
        ngrid=args.ngrid,
        xmax=args.xmax,
        phonon_basis=args.phonon_basis,
        active_modes=active_modes,
        mode_transform=mode_transform,
        mode_strengths=mode_strengths,
    )

    mode_label = (
        f"active_modes={model._active_modes_tuple()}"
        if mode_transform is None
        else f"collective_modes={model._active_modes_tuple()}"
    )
    print(
        f"Adiabatic electronic Holstein NARG: L={args.nsites}, "
        f"target={model.target}, {mode_label}, "
        f"basis={model._phonon_basis_name()}, ngrid={args.ngrid}, "
        f"q_range=[{-args.xmax:g},{args.xmax:g}], electronic_dim={model.electronic_dim}"
    )
    if mode_report is not None:
        print(
            "density-response strengths: "
            + " ".join(f"{value:.6g}" for value in mode_report.strengths)
        )
        for index, (strength, row) in enumerate(zip(mode_strengths, mode_transform)):
            weights = " ".join(f"{value:+.3f}" for value in row)
            print(f"mode {index:2d}: strength={strength:.6g} weights=[{weights}]")

    exact = None
    if args.backend == "joint" and not args.skip_exact:
        start = perf_counter()
        exact, _ = model.exact(nroots=args.nroots)
        print(f"coordinate-grid exact E0 = {exact[0]:.12f}  sec={perf_counter() - start:.3f}")

    if args.backend == "sequential":
        print("nstates     energy          seconds   final_dim   step_dims")
        for nstates in args.nstates_per_point:
            start = perf_counter()
            result = model.run_sequential(
                nstates_per_point=int(nstates),
                bond_dim=args.bond_dim,
                initial_electronic_states=args.initial_electronic_states,
                nroots=args.nroots,
            )
            step_dims = ",".join(
                f"{step.grid_dim}x{step.conditional_dim}->{step.kept}"
                for step in result.steps
            )
            print(
                f"{int(nstates):7d} {result.energies[0]: .12f} "
                f"{perf_counter() - start:9.3f}   {result.block_hamiltonian.shape[0]:9d}   "
                f"{step_dims}"
            )
    else:
        print("nstates     energy          error       seconds   basis_dim")
        for nstates in args.nstates_per_point:
            start = perf_counter()
            result = model.run(nstates_per_point=int(nstates), nroots=args.nroots)
            error = result.energies[0] - exact[0] if exact is not None else float("nan")
            print(
                f"{int(nstates):7d} {result.energies[0]: .12f} {error: .3e} "
                f"{perf_counter() - start:9.3f}   {result.hamiltonian.shape[0]}"
            )


if __name__ == "__main__":
    main()
