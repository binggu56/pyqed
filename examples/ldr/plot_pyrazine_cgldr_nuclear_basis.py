#!/usr/bin/env python3
"""Compare four-mode pyrazine CGLDR nuclear-basis calculations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


COORDINATE_LABELS = (
    r"$\langle Q_{t1}\rangle$",
    r"$\langle Q_{t2}\rangle$",
    r"$\langle Q_{t3}\rangle$",
    r"$\langle Q_c\rangle$",
)


def load_results(filename):
    with np.load(filename) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def plot_basis_comparison(coarse, large, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
    })
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    styles = (
        (coarse, r"$3^4$ (81)", "#666666", "--"),
        (large, r"$5\times5\times15^2$ (5625)", "#0072B2", "-"),
    )

    for results, label, color, linestyle in styles:
        axes[0, 0].plot(
            results["times_fs"],
            results["cg_populations"][:, 0],
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0, 0].set_ylabel(r"$P(S_1)$")
    axes[0, 0].set_ylim(-0.02, 1.02)

    coordinate_axes = (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0])
    for coordinate, (axis, ylabel) in enumerate(
        zip(coordinate_axes, COORDINATE_LABELS)
    ):
        for results, _label, color, linestyle in styles:
            axis.plot(
                results["times_fs"],
                results["cg_coordinate_means"][:, coordinate],
                color=color,
                linestyle=linestyle,
            )
        axis.axhline(0.0, color="#aaaaaa", linewidth=0.6)
        axis.set_ylabel(ylabel)

    norm_axis = axes[2, 1]
    for results, _label, color, linestyle in styles:
        error = np.maximum(np.abs(results["cg_norms"] - 1.0), 1.0e-16)
        norm_axis.semilogy(
            results["times_fs"],
            error,
            color=color,
            linestyle=linestyle,
        )
    norm_axis.set_ylabel(r"$|\langle\Psi|\Psi\rangle-1|$")
    norm_axis.set_ylim(1.0e-16, 1.0e-9)

    for panel, (letter, axis) in enumerate(zip("abcdef", axes.flat)):
        if panel >= 4:
            axis.set_xlabel("Time (fs)")
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.text(
            0.02,
            0.95,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(pdf)
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("large", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    png, pdf = plot_basis_comparison(
        load_results(args.coarse),
        load_results(args.large),
        args.output,
    )
    print("figures:", png, pdf)


if __name__ == "__main__":
    main()
