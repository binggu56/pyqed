#!/usr/bin/env python3
"""Plot the SO2 MACE data-density and link-representation comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILES = {
    "3x3 random\nendpoint": Path("/private/tmp/so2_mace_fit_atomistic.json"),
    "5x5 invariant block\nendpoint": Path(
        "/private/tmp/so2_mace_fit_5x5x5_invariant_block.json"
    ),
    "5x5 invariant block\ndirectional": Path(
        "/private/tmp/so2_mace_fit_5x5x5_directional.json"
    ),
}
OUTPUT = Path("/private/tmp/so2_mace_fitting_comparison.png")


def main():
    values = {label: json.loads(path.read_text()) for label, path in FILES.items()}
    labels = list(values)
    energy = [values[label]["energy_relative_error"]["held_out"] for label in labels]
    links = np.asarray(
        [
            [values[label]["link_relative_error"][str(axis)]["held_out"] for axis in range(3)]
            for label in labels
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, energy, color="#0072B2")
    axes[0].set_xticks(x, labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Held-out energy relative error")
    width = 0.23
    colors = ("#0072B2", "#D55E00", "#009E73")
    for axis, color in enumerate(colors):
        axes[1].bar(x + (axis - 1) * width, links[:, axis], width, color=color, label=f"axis {axis}")
    axes[1].set_xticks(x, labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("Held-out link relative error")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    for label, axis in zip("ab", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(OUTPUT, dpi=300)
    print(OUTPUT)


if __name__ == "__main__":
    main()
