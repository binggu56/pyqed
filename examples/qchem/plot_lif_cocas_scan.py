#!/usr/bin/env python3
"""Plot the LiF state-averaged COCASCI scan from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scan(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="examples/qchem/lif_cocas_scan_6-31g_sa2.csv",
        help="CSV file produced by lif_cocas_scan.py",
    )
    parser.add_argument(
        "--output",
        default="examples/qchem/lif_cocas_scan_6-31g_sa2.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data = load_scan(Path(args.input))

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
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
    ax.set_title("LiF COCASCI PEC (6-31g, SA-2, CAS(2e,2o))")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
