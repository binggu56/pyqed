#!/usr/bin/env python3
"""Overlay native pyqed and PySCF LiF SA-CASSCF curves on one plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, mcscf, scf


def load_pyqed_scan(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No scan data found in {path}")

    data = {
        "r_bohr": np.array([float(row["r_bohr"]) for row in rows]),
        "state0_h": np.array([float(row["state0_h"]) for row in rows]),
        "state1_h": np.array([float(row["state1_h"]) for row in rows]),
        "e_avg_h": np.array([float(row["e_avg_h"]) for row in rows]),
    }
    return data


def run_pyscf_scan(distances, basis: str, ncas: int, nelecas: int, nstates: int):
    rows = []
    for r_bohr in distances:
        mol = gto.M(
            atom=f"Li 0 0 0; F 0 0 {float(r_bohr)}",
            basis=basis,
            unit="Bohr",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1.0e-10
        mf.kernel()

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.conv_tol = 1.0e-7
        mc.max_cycle_macro = 40
        mc.max_stepsize = 0.02
        mc.fcisolver.nroots = nstates
        mc = mc.state_average_([1.0 / nstates] * nstates)
        mc.kernel()

        e_states = np.array(mc.e_states, dtype=float)
        rows.append(
            {
                "r_bohr": float(r_bohr),
                "state0_h": float(e_states[0]),
                "state1_h": float(e_states[1]),
                "e_avg_h": float(np.mean(e_states)),
            }
        )
    return rows


def save_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["r_bohr", "state0_h", "state1_h", "e_avg_h"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44.csv",
        help="CSV file produced by lif_casscf_scan.py",
    )
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nelecas", type=int, default=4)
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument(
        "--pyscf-output",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44_pyscf.csv",
        help="Output CSV path for the PySCF reference data",
    )
    parser.add_argument(
        "--output",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44_vs_pyscf.png",
        help="Output image path",
    )
    args = parser.parse_args()

    pyqed = load_pyqed_scan(Path(args.input))
    pyscf_rows = run_pyscf_scan(
        pyqed["r_bohr"],
        basis=args.basis,
        ncas=args.ncas,
        nelecas=args.nelecas,
        nstates=args.nstates,
    )
    save_rows(Path(args.pyscf_output), pyscf_rows)

    pyscf_data = {
        "r_bohr": np.array([row["r_bohr"] for row in pyscf_rows], dtype=float),
        "state0_h": np.array([row["state0_h"] for row in pyscf_rows], dtype=float),
        "state1_h": np.array([row["state1_h"] for row in pyscf_rows], dtype=float),
        "e_avg_h": np.array([row["e_avg_h"] for row in pyscf_rows], dtype=float),
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)

    colors = {
        "state0": "#1f77b4",
        "state1": "#d62728",
        "avg": "#2ca02c",
    }

    ax.plot(
        pyqed["r_bohr"], pyqed["state0_h"], marker="o", linewidth=2.0,
        color=colors["state0"], label="pyqed State 0",
    )
    ax.plot(
        pyqed["r_bohr"], pyqed["state1_h"], marker="s", linewidth=2.0,
        color=colors["state1"], label="pyqed State 1",
    )
    ax.plot(
        pyqed["r_bohr"], pyqed["e_avg_h"], marker="^", linewidth=2.0, linestyle="--",
        color=colors["avg"], label="pyqed State average",
    )

    ax.plot(
        pyscf_data["r_bohr"], pyscf_data["state0_h"], marker="x", linewidth=1.5, linestyle=":",
        color=colors["state0"], label="PySCF State 0",
    )
    ax.plot(
        pyscf_data["r_bohr"], pyscf_data["state1_h"], marker="x", linewidth=1.5, linestyle=":",
        color=colors["state1"], label="PySCF State 1",
    )
    ax.plot(
        pyscf_data["r_bohr"], pyscf_data["e_avg_h"], marker="x", linewidth=1.5, linestyle="-.",
        color=colors["avg"], label="PySCF State average",
    )

    ax.set_xlabel("Li-F distance (bohr)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("LiF CASSCF PEC Comparison (6-31g, SA-2, CAS(4e,4o))")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote PySCF reference to {args.pyscf_output}")
    print(f"Wrote comparison plot to {output}")


if __name__ == "__main__":
    main()
