#!/usr/bin/env python3
"""Plot the scalable SO2 q_a polar-metric correction benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt


SOURCES = {
    "Full LDR": Path(
        "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
        "full_ldr_dynamics.npz"
    ),
    "3A": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_inner_three_anchor_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
    r"3A + fitted $P$": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_three_anchor_fitted_p_harmonic_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
}

STYLES = {
    "Full LDR": dict(color="black", linestyle="-", linewidth=1.75),
    "3A": dict(color="#0072B2", linestyle="--", linewidth=1.55),
    r"3A + fitted $P$": dict(
        color="#D55E00", linestyle=":", linewidth=1.65
    ),
}


def load(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def format_axis(axis):
    axis.format(
        xlim=(0.0, 20.0),
        xticks=np.arange(0.0, 20.1, 5.0),
        tickdir="in",
        tickminor=True,
        grid=False,
    )
    axis.grid(axis="y", color="0.88", linewidth=0.45, alpha=0.75)


def save(figure, output_dir, stem):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas6e6o_631gstar_p_dressing_plots"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {label: load(path) for label, path in SOURCES.items()}
    times = results["Full LDR"]["times_fs"]
    for values in results.values():
        np.testing.assert_allclose(values["times_fs"], times)

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.5,
            "savefig.transparent": False,
        }
    )

    figure, axes = uplt.subplots(
        array=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 3, 3, 4, 4],
            [2, 2, 3, 3, 4, 4],
        ],
        width=7.2,
        height=4.8,
        share=False,
        wspace=4.0,
        hspace=2.2,
    )
    panels = (
        ("reference_populations", 1, lambda value: value),
        ("means", 0, lambda value: value),
        ("means", 1, np.rad2deg),
        ("means", 2, lambda value: value),
    )
    titles = (
        r"$\mathbf{(a)}$  Population dynamics",
        r"$\mathbf{(b)}$  Symmetric stretch",
        r"$\mathbf{(c)}$  Bend angle",
        r"$\mathbf{(d)}$  Antisymmetric stretch",
    )
    ylabels = (
        r"$P_1$",
        r"$\langle q_s\rangle$ (bohr)",
        r"$\langle\theta\rangle$ (deg)",
        r"$\langle q_a\rangle$ (bohr)",
    )
    ylimits = (
        (-0.01, 0.47),
        (3.815, 3.882),
        (119.40, 119.57),
        (-0.0012, 0.0002),
    )
    for axis, (field, index, transform), title, ylabel, ylim in zip(
        axes,
        panels,
        titles,
        ylabels,
        ylimits,
    ):
        for label, values in results.items():
            axis.plot(
                times,
                transform(values[field][:, index]),
                label=label,
                **STYLES[label],
            )
        format_axis(axis)
        axis.format(
            title=title,
            titleloc="left",
            ylabel=ylabel,
            ylim=ylim,
        )
    axes[0].tick_params(labelbottom=False)
    axes[1].format(xlabel="Time (fs)")
    axes[2].format(xlabel="Time (fs)")
    axes[3].format(xlabel="Time (fs)")
    axes[3].axhline(0.0, color="0.55", linewidth=0.65, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="top",
        ncols=3,
        frame=False,
        order="F",
    )
    save(figure, args.output_dir, "so2_p_dressing_dynamics")


if __name__ == "__main__":
    main()
