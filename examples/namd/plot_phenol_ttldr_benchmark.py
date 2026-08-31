#!/usr/bin/env python3
"""Plot before/after timings from two phenol component-TDVP benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def timings(summary):
    return {
        int(item["workers"]): float(item["seconds_per_step"])
        for item in summary["timings"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    native = json.loads(args.native.read_text())
    old = timings(baseline)
    new = timings(native)
    workers = sorted(set(old).intersection(new))
    if not workers:
        raise ValueError("the benchmark summaries have no common worker counts")

    figure, panels = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    positions = list(range(len(workers)))
    width = 0.36
    panels[0].bar(
        [position - width / 2 for position in positions],
        [old[worker] for worker in workers],
        width=width,
        label="NumPy sum",
        color="#999999",
    )
    panels[0].bar(
        [position + width / 2 for position in positions],
        [new[worker] for worker in workers],
        width=width,
        label="native compact sum",
        color="#0072B2",
    )
    panels[0].set(
        xticks=positions,
        xticklabels=workers,
        xlabel="workers",
        ylabel="seconds per TDVP2 step",
        title="Phenol rank-40 propagation",
    )
    panels[0].legend(frameon=False)

    speedups = [old[worker] / new[worker] for worker in workers]
    panels[1].bar(positions, speedups, color="#009E73", width=0.62)
    panels[1].axhline(1.0, color="black", linewidth=0.8)
    panels[1].set(
        xticks=positions,
        xticklabels=workers,
        xlabel="workers",
        ylabel="speedup over NumPy sum",
        title="Compiled-kernel gain",
    )
    for position, speedup in zip(positions, speedups):
        panels[1].text(position, speedup + 0.04, f"{speedup:.2f}×", ha="center")
    for label, panel in zip("ab", panels):
        panel.text(
            0.02,
            0.97,
            label,
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
        panel.grid(axis="y", alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
