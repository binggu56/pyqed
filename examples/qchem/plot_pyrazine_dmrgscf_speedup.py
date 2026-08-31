#!/usr/bin/env python3
"""Plot matched pyrazine DMRG-SCF timing summaries before and after optimization."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(path.read_text()) for path in (args.before, args.after)]
    labels = ["before", "optimized"]
    wall = np.array([row["wall_time_seconds"] for row in rows])
    rdm = np.array([row["dmrgscf_timing"]["rdm_seconds"] for row in rows])
    orbital = np.array(
        [row["dmrgscf_timing"]["orbital_opt_seconds"] for row in rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, wall, color=["0.55", "C0"])
    axes[0].set(ylabel="Wall time (s)", title=f"Total: {wall[0] / wall[1]:.2f}x faster")
    axes[0].grid(axis="y", alpha=0.3)

    x = np.arange(2)
    width = 0.36
    axes[1].bar(x - width / 2, rdm, width, label="RDM")
    axes[1].bar(x + width / 2, orbital, width, label="orbital optimizer")
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Measured component time (s)",
        title="DMRG-SCF contraction costs",
    )
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Pyrazine CAS(10,10), D=16, two sweeps, one macro cycle")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
