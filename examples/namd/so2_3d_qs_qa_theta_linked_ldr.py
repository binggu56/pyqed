#!/usr/bin/env python3
"""Full SO2 linked-LDR dynamics in matched (q_s, q_a, theta) coordinates."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs

from h3plus_3d_sine_legendre_linked_ldr import (
    electronic_populations,
    load_cached_scan,
    phase_projected_packet,
    working_directory,
)


SQRT2 = np.sqrt(2.0)
REFERENCE_BOND = 2.70
MATCHED_STEP = 0.12 / SQRT2
MATCHED_HALF_DOMAIN = 5.0 * MATCHED_STEP


def so2_body_frame(r=2.70, theta=np.deg2rad(119.5)):
    return [
        ["O", (float(r), 0.0, 0.0)],
        ["S", (0.0, 0.0, 0.0)],
        ["O", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def coordinate_moments(solver, psilist):
    means = np.zeros((len(psilist), solver.ndim))
    variances = np.zeros_like(means)
    for sample, psi in enumerate(psilist):
        probability = np.sum(np.abs(psi) ** 2, axis=-1)
        for axis, grid in enumerate(solver.x):
            shape = [1] * solver.ndim
            shape[axis] = len(grid)
            coordinate = np.asarray(grid).reshape(shape)
            mean = np.sum(probability * coordinate)
            means[sample, axis] = mean
            variances[sample, axis] = np.sum(
                probability * (coordinate - mean) ** 2
            )
    return means, variances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qs", type=int, default=9)
    parser.add_argument("--n-qa", type=int, default=9)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument(
        "--qs-min",
        type=float,
        default=SQRT2 * REFERENCE_BOND - MATCHED_HALF_DOMAIN,
    )
    parser.add_argument(
        "--qs-max",
        type=float,
        default=SQRT2 * REFERENCE_BOND + MATCHED_HALF_DOMAIN,
    )
    parser.add_argument(
        "--qa-min",
        type=float,
        default=-MATCHED_HALF_DOMAIN,
    )
    parser.add_argument(
        "--qa-max",
        type=float,
        default=MATCHED_HALF_DOMAIN,
    )
    parser.add_argument("--theta-min-deg", type=float, default=90.0)
    parser.add_argument("--theta-max-deg", type=float, default=150.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--center-r", type=float, default=REFERENCE_BOND)
    parser.add_argument("--center-theta-deg", type=float, default=119.5)
    parser.add_argument("--sigma-stretch", type=float, default=0.16)
    parser.add_argument("--sigma-theta-deg", type=float, default=7.5)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--electronic-method",
        choices=("am1/meci", "uam1/meci", "casci", "rohf/casci"),
        default="casci",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nelecas", type=int, default=4)
    parser.add_argument("--scf-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--reuse-scan", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name(
            "so2_3d_qs_qa_theta_linked_ldr"
        ),
    )
    args = parser.parse_args()

    theta_min = np.deg2rad(args.theta_min_deg)
    theta_max = np.deg2rad(args.theta_max_deg)
    center = np.array(
        [
            SQRT2 * args.center_r,
            0.0,
            np.deg2rad(args.center_theta_deg),
        ]
    )
    widths = np.array(
        [
            args.sigma_stretch,
            args.sigma_stretch,
            np.deg2rad(args.sigma_theta_deg),
        ]
    )
    solver = Triatom(
        so2_body_frame(args.center_r, center[2]),
        nstates=args.nstates,
        basis=args.basis,
        charge=0,
        spin=0,
        unit="bohr",
        coordinates="qs-qa-theta",
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.set_dvr(
        domains=[
            [args.qs_min, args.qs_max],
            [args.qa_min, args.qa_max],
            [theta_min, theta_max],
        ],
        npts=[args.n_qs, args.n_qa, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with working_directory(args.outdir):
        reused = args.reuse_scan and load_cached_scan(solver, args.outdir)
        if not reused:
            solver.scan_pes(
                electronic_method=args.electronic_method,
                basis=args.basis,
                nstates=args.nstates,
                ncas=args.ncas,
                nelecas=args.nelecas,
                overlap_method="link-only",
                n_workers=args.n_workers,
                worker_threads=1,
                scf_tol=args.scf_tol,
                max_cycle=args.max_cycle,
                damping=args.damping,
            )
    np.savez(
        args.outdir / "so2_3d_qs_qa_theta_grid.npz",
        qs=solver.x[0],
        qa=solver.x[1],
        theta=solver.x[2],
    )
    if args.scan_only:
        print(f"[scan] completed in {time.perf_counter() - start:.2f} s")
        return

    psi0, reference_index = phase_projected_packet(
        solver,
        state=args.initial_state,
        center=center,
        sigma_r=args.sigma_stretch,
        sigma_theta=widths[2],
        mode="adiabatic",
    )
    result = solver.run(
        psi0,
        dt=args.dt_fs / au2fs,
        nt=args.nt,
        nout=args.nout,
        kinetic_propagator="expm_multiply",
        kinetic_action="matrix-free",
    )
    times_fs = np.asarray(result["times"]) * au2fs
    populations = electronic_populations(result["psilist"])
    means, variances = coordinate_moments(solver, result["psilist"])
    norms = np.asarray([solver.norm(psi) for psi in result["psilist"]])
    output = args.outdir / "so2_3d_qs_qa_theta_linked_ldr_observables.npz"
    np.savez(
        output,
        times_fs=times_fs,
        psi_t=np.asarray(result["psilist"]),
        qs=solver.x[0],
        qa=solver.x[1],
        theta=solver.x[2],
        theta_deg=np.rad2deg(solver.x[2]),
        electronic_populations=populations,
        coordinate_expectations=means,
        coordinate_variances=variances,
        coordinate_names=np.asarray(solver.coordinate_labels),
        norms=norms,
        reference_index=np.asarray(reference_index),
    )
    print(f"[initial] center={center}, widths={widths}")
    print(f"[initial] reference_index={reference_index}, norm={norms[0]:.12f}")
    print(f"[final] populations={populations[-1]}")
    print(f"[final] coordinates={means[-1]}")
    print(f"[data] {output}")
    print(f"[timing] {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
