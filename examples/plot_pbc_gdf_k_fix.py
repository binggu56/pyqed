#!/usr/bin/env python3
"""Plot the periodic GDF metric and J/K errors before and after a fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2mev

HARTREE_TO_MEV = au2mev


def _study(path):
    payload = json.loads(Path(path).read_text())
    studies = payload.get("studies", [])
    if len(studies) != 1:
        raise ValueError(f"{path} must contain exactly one validation study.")
    return studies[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_k_fix.pdf"),
    )
    args = parser.parse_args()

    before = _study(args.before)
    after = _study(args.after)
    labels = ("Before", "Corrected")
    colors = ("#4C78A8", "#E45756")
    metric_error = np.asarray(
        [
            before["max_pair_metric_relative_error"],
            after["max_pair_metric_relative_error"],
        ]
    )
    jk_error = HARTREE_TO_MEV * np.asarray(
        [
            [before["max_abs_J_error_Ha"], before["max_abs_K_error_Ha"]],
            [after["max_abs_J_error_Ha"], after["max_abs_K_error_Ha"]],
        ]
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.86, wspace=0.36)

    ax = axes[0]
    x = np.arange(2)
    ax.bar(x, metric_error, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Relative pair-metric error")
    ax.set_xticks(x, labels)
    ax.set_ylim(1.0e-9, 1.0e-4)
    ax.set_title("AO-pair Coulomb metric")
    ax.grid(axis="y", which="major", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(2)
    width = 0.34
    ax.bar(
        x - width / 2,
        jk_error[:, 0],
        width,
        color=colors[0],
        edgecolor="black",
        linewidth=0.6,
        label=r"$J$",
    )
    ax.bar(
        x + width / 2,
        jk_error[:, 1],
        width,
        color=colors[1],
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
        label=r"$K$",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute error (meV)")
    ax.set_xticks(x, labels)
    ax.set_ylim(1.0e-4, 1.0e1)
    ax.set_title(r"Diamond $\Gamma$-point J/K")
    ax.grid(axis="y", which="major", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.14,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
