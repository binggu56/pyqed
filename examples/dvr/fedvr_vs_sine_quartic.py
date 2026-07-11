#!/usr/bin/env python3
"""Compare FE-DVR and sine DVR on a 1D quartic oscillator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR, SineDVR


def potential(x, omega=1.0, anharm=0.1):
    return 0.5 * omega**2 * x**2 + anharm * x**4


def solve_sine(xmin, xmax, npts, nstates, omega, anharm):
    dvr = SineDVR(xmin, xmax, npts, mass=1.0)
    t0 = time.perf_counter()
    H = dvr.t() + np.diag(potential(dvr.x, omega=omega, anharm=anharm))
    evals = scipy.linalg.eigh(H, eigvals_only=True, subset_by_index=[0, nstates - 1])
    elapsed = time.perf_counter() - t0
    return dvr.npts, H.size, evals, elapsed


def solve_fedvr(xmin, xmax, n_elements, n_lobatto, nstates, omega, anharm):
    dvr = FEDVR(xmin, xmax, n_elements=n_elements, n_lobatto=n_lobatto, mass=1.0)
    t0 = time.perf_counter()
    H = dvr.kinetic_sparse() + sp.diags(potential(dvr.x, omega=omega, anharm=anharm))
    evals = sla.eigsh(H, k=nstates, which="SA", return_eigenvectors=False)
    evals = np.sort(evals)
    elapsed = time.perf_counter() - t0
    return dvr.npts, H.nnz, evals, elapsed


def print_row(label, npts, nnz, evals, ref, elapsed):
    errors = np.abs(evals - ref)
    print(
        f"{label:14s} {npts:5d} {nnz:9d} {elapsed:9.4f} "
        f"{np.max(errors):12.3e} "
        + " ".join(f"{e:13.8f}" for e in evals)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmin", type=float, default=-8.0)
    parser.add_argument("--xmax", type=float, default=8.0)
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--anharm", type=float, default=0.1)
    parser.add_argument("--ref-sine-npts", type=int, default=320)
    args = parser.parse_args()

    _, _, ref, ref_time = solve_sine(
        args.xmin,
        args.xmax,
        args.ref_sine_npts,
        args.nstates,
        args.omega,
        args.anharm,
    )

    print(
        f"Quartic oscillator: V(x)=0.5*{args.omega:g}^2*x^2 + "
        f"{args.anharm:g}*x^4 on [{args.xmin:g}, {args.xmax:g}]"
    )
    print(f"Reference: sine DVR npts={args.ref_sine_npts}, time={ref_time:.4f} s")
    print("reference energies:", " ".join(f"{e:.10f}" for e in ref))
    print()
    print(
        f"{'method':14s} {'npts':>5s} {'nnz':>9s} {'time/s':>9s} "
        f"{'max|err|':>12s} "
        + " ".join(f"E{i:<12d}" for i in range(args.nstates))
    )

    for npts in (40, 80, 120, 180):
        n, nnz, evals, elapsed = solve_sine(
            args.xmin,
            args.xmax,
            npts,
            args.nstates,
            args.omega,
            args.anharm,
        )
        print_row(f"sine-{npts}", n, nnz, evals, ref, elapsed)

    for n_elements, n_lobatto in ((8, 5), (10, 6), (12, 6), (16, 6), (20, 6)):
        n, nnz, evals, elapsed = solve_fedvr(
            args.xmin,
            args.xmax,
            n_elements,
            n_lobatto,
            args.nstates,
            args.omega,
            args.anharm,
        )
        print_row(f"fedvr-{n_elements}x{n_lobatto}", n, nnz, evals, ref, elapsed)


if __name__ == "__main__":
    main()
