#!/usr/bin/env python3
"""Plot packed-wave versus compiled-growth-graph SU(2)-NARG results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    packed = load(args.packed)
    graph = load(args.graph)
    labels = ["Packed wave", "Compiled graph"]
    times = [packed["chain"][0]["seconds"], graph["chain"][0]["seconds"]]
    memory = [packed["peak_rss_mib"], graph["peak_rss_mib"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    colors = ["#8f9aa3", "#2673a8"]
    for axis, values, ylabel, title, unit in (
        (axes[0], times, "Time (s)", "Warmed serial runtime", "s"),
        (axes[1], memory, "Peak RSS (MiB)", "Process peak memory", "MiB"),
    ):
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set(ylabel=ylabel, title=title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f} {unit}" if unit == "s" else f"{value:.0f} {unit}",
                ha="center",
                va="bottom",
            )
    fig.suptitle("SU(2)-NARG reduced-growth execution")
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
