#!/usr/bin/env python3
"""Scan the LiF potential-energy curve with state-averaged native pyqed CASSCF."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import CASSCF, Molecule


def run_point(mf, r_bohr: float, ncas: int, nelecas: int, nstates: int):
    sink = io.StringIO()
    with redirect_stdout(sink):
        mc = CASSCF(
            mf,
            ncas=ncas,
            nelecas=nelecas,
            max_cycle=30,
            conv_tol=1.0e-6,
            conv_tol_grad_relaxed=2.0e-3,
            optimizer="AH",
            ah_hessian="analytic",
            ci_method="direct_ci",
        )
        mc.state_average(np.ones(nstates) / nstates)
        mc.run(nstates=nstates)

    e_states = np.asarray(mc.e_tot, dtype=float)
    return {
        "r_bohr": float(r_bohr),
        "ehf_h": float(mf.e_tot),
        "state0_h": float(e_states[0]),
        "state1_h": float(e_states[1]),
        "e_avg_h": float(np.mean(e_states)),
        "converged": bool(mc.converged),
        "n_macro": int(len(mc.history)),
    }


def build_scanner(first_r: float, basis: str):
    mol0 = Molecule(atom=f"Li 0 0 0; F 0 0 {first_r}", unit="bohr", basis=basis)
    mol0.build()
    mf0 = mol0.RHF()
    mf0.max_cycle = 200
    with redirect_stdout(io.StringIO()):
        mf0.run(init_guess="hcore", max_cycle=200, tol=1.0e-9)
    return mf0.as_scanner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nelecas", type=int, default=4)
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument(
        "--distances",
        type=float,
        nargs="*",
        default=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Li-F distances in bohr.",
    )
    parser.add_argument(
        "--output",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44.csv",
        help="CSV output path relative to the repo root.",
    )
    args = parser.parse_args()

    rows = []
    scanner = build_scanner(float(args.distances[0]), args.basis)

    for i, r_bohr in enumerate(args.distances):
        try:
            if i == 0:
                mf = scanner.mf
            else:
                mol = Molecule(atom=f"Li 0 0 0; F 0 0 {r_bohr}", unit="bohr", basis=args.basis)
                mol.build()
                with redirect_stdout(io.StringIO()):
                    scanner(mol)
                mf = scanner.mf

            row = run_point(mf, r_bohr, args.ncas, args.nelecas, args.nstates)
            rows.append(row)
            print(
                "R = {r_bohr:4.1f} bohr  "
                "E0 = {state0_h: .12f}  "
                "E1 = {state1_h: .12f}  "
                "Eavg = {e_avg_h: .12f}  "
                "macro = {n_macro:2d}".format(**row)
            )
        except (RuntimeError, SystemExit) as exc:
            print(f"R = {r_bohr:4.1f} bohr  FAILED  {exc}")
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "r_bohr",
                "ehf_h",
                "state0_h",
                "state1_h",
                "e_avg_h",
                "converged",
                "n_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} points to {output}")


if __name__ == "__main__":
    main()
