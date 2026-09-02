#!/usr/bin/env python3
"""Compare matched H3+ and D3+ MACE/TNLDR dynamics runs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrogen", type=Path, required=True)
    parser.add_argument("--deuterium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        r"H$_3^+$": np.load(args.hydrogen),
        r"D$_3^+$": np.load(args.deuterium),
    }
    figure, panels = plt.subplots(
        1, 2, figsize=(7.2, 2.8), constrained_layout=True
    )
    colors = {r"H$_3^+$": "tab:blue", r"D$_3^+$": "tab:orange"}
    for label, run in runs.items():
        panels[0].plot(
            run["time_fs"], run["populations"][:, 0],
            color=colors[label], label=label,
        )
        panels[1].plot(
            run["snapshot_times_fs"], run["snapshot_edge_probability"],
            "o-", color=colors[label], label=label,
        )
    panels[0].set(
        xlabel="time / fs", ylabel=r"$S_1$ population",
        title="Isotope effect on transfer", ylim=(-0.02, 0.62),
    )
    panels[1].axhline(0.05, color="0.35", linestyle="--", label="5% warning")
    panels[1].set(
        xlabel="time / fs", ylabel="outer-layer probability",
        title="Boundary diagnostic", ylim=(-0.01, 0.28),
    )
    for panel in panels:
        panel.grid(alpha=0.2)
        panel.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=320)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
