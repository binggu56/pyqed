#!/usr/bin/env python3
"""Compare direct-frame and spectral latent-H fits for SO2 links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def held(metrics, key):
    return np.asarray([metrics[key][str(axis)]["held_out"] for axis in range(3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank8", type=Path)
    parser.add_argument("rank10", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank8 = json.loads(args.rank8.read_text())
    rank10 = json.loads(args.rank10.read_text())
    direct = held(rank8, "direct_y_link_relative_error")
    labels = (r"$r_1$", r"$r_2$", r"$\theta$")
    x = np.arange(3)
    width = 0.24
    colors = ("#777777", "#0072B2", "#D55E00")

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    full = (direct, held(rank8, "link_relative_error"), held(rank10, "link_relative_error"))
    magnitude = (
        direct,
        held(rank8, "link_magnitude_relative_error"),
        held(rank10, "link_magnitude_relative_error"),
    )
    for axis, values, ylabel in zip(
        axes,
        (full, magnitude),
        ("Held-out complex-link error", "Held-out magnitude error"),
    ):
        for offset, value, color, label in zip(
            (-width, 0.0, width), values, colors, ("Direct $Y$", "Latent $H$, rank 8", "Latent $H$, rank 10")
        ):
            axis.bar(x + offset, 100.0 * value, width=width, color=color, label=label)
        axis.set(xticks=x, xticklabels=labels, ylabel=ylabel + " (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
    axes[0].text(0.02, 0.98, "a", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(0.02, 0.98, "b", transform=axes[1].transAxes, va="top", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.08, 1.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
