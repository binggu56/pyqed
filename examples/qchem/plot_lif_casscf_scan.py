#!/usr/bin/env python3
"""Plot the LiF state-averaged native pyqed CASSCF scan from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scan(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No scan data found in {path}")

    numeric_keys = ["r_bohr", "ehf_h", "state0_h", "state1_h", "e_avg_h", "n_macro"]
    data = {key: np.array([float(row[key]) for row in rows]) for key in numeric_keys}
    data["converged"] = np.array([row["converged"].strip().lower() == "true" for row in rows])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44.csv",
        help="CSV file produced by lif_casscf_scan.py",
    )
    parser.add_argument(
        "--output",
        default="examples/qchem/lif_casscf_scan_6-31g_sa2_cas44.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data = load_scan(Path(args.input))

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
    ax.plot(data["r_bohr"], data["state0_h"], marker="o", linewidth=2.0, label="State 0")
    ax.plot(data["r_bohr"], data["state1_h"], marker="s", linewidth=2.0, label="State 1")
    ax.plot(
        data["r_bohr"],
        data["e_avg_h"],
        marker="^",
        linewidth=2.0,
        linestyle="--",
        label="State average",
    )

    ax.set_xlabel("Li-F distance (bohr)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("LiF CASSCF PEC (6-31g, SA-2, CAS(4e,4o))")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
