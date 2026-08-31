#!/usr/bin/env python3
"""Small H2O state-averaged CASSCF cut.

This is a quick stability check before wiring SA-CASSCF into the rovibronic LDR
grid scanner.  It scans the H-O-H bend with fixed O-H lengths and plots the
state-averaged CASSCF roots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.units import au2ev
from pyqed.qchem import CASSCF, Molecule

HARTREE_TO_EV = au2ev


def h2o_atom(r, theta_deg):
    theta = np.deg2rad(theta_deg)
    return [
        ["H", (r, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (r * np.cos(theta), r * np.sin(theta), 0.0)],
    ]


def run_point(args, theta_deg):
    mol = Molecule(
        atom=h2o_atom(args.r_oh, theta_deg),
        basis=args.basis,
        charge=0,
        spin=0,
        unit="bohr",
    )
    mol.build(eri=args.eri)
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    weights = np.ones(args.nstates) / args.nstates
    mc = (
        CASSCF(
            mf,
            ncas=args.ncas,
            nelecas=args.nelecas,
            max_cycle=args.max_cycle,
            verbose=args.verbose,
            optimizer=args.optimizer,
            conv_tol=args.conv_tol,
            conv_tol_grad=args.conv_tol_grad,
        )
        .state_average(weights)
        .run(nstates=args.nstates)
    )
    return np.asarray(mc.e_tot, dtype=float), bool(mc.converged), len(mc.history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--eri", default="dense")
    p.add_argument("--ncas", type=int, default=4)
    p.add_argument("--nelecas", type=int, default=4)
    p.add_argument("--nstates", type=int, default=3)
    p.add_argument("--r-oh", type=float, default=1.80965)
    p.add_argument("--theta-min", type=float, default=80.0)
    p.add_argument("--theta-max", type=float, default=130.0)
    p.add_argument("--n-theta", type=int, default=7)
    p.add_argument("--max-cycle", type=int, default=20)
    p.add_argument("--optimizer", default="AH")
    p.add_argument("--conv-tol", type=float, default=1.0e-8)
    p.add_argument("--conv-tol-grad", type=float, default=1.0e-6)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_sa_casscf_cut"))
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    energies = np.zeros((args.n_theta, args.nstates))
    converged = np.zeros(args.n_theta, dtype=bool)
    cycles = np.zeros(args.n_theta, dtype=int)

    for i, angle in enumerate(theta):
        energies[i], converged[i], cycles[i] = run_point(args, float(angle))
        print(
            f"theta={angle:7.2f} deg  converged={converged[i]}  "
            f"cycles={cycles[i]:2d}  E={energies[i]}"
        )

    rel_ev = (energies - energies[:, [0]]) * HARTREE_TO_EV
    csv = args.outdir / "h2o_sa_casscf_bend_cut.csv"
    header = "theta_deg," + ",".join(f"E_S{s}_hartree" for s in range(args.nstates))
    header += ",converged,cycles"
    np.savetxt(
        csv,
        np.column_stack([theta, energies, converged.astype(int), cycles]),
        delimiter=",",
        header=header,
        comments="",
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    for state in range(args.nstates):
        ax.plot(theta, rel_ev[:, state], marker="o", label=f"S{state}")
    ax.set_xlabel("H-O-H angle / deg")
    ax.set_ylabel("energy relative to S0 / eV")
    ax.set_title(f"H2O SA{args.nstates}-CASSCF({args.nelecas}e,{args.ncas}o)")
    ax.legend()
    png = args.outdir / "h2o_sa_casscf_bend_cut.png"
    fig.savefig(png, dpi=220)
    plt.close(fig)

    print(f"all_converged={bool(np.all(converged))}")
    print(f"csv={csv}")
    print(f"png={png}")


if __name__ == "__main__":
    main()
