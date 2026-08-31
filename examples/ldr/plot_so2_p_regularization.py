#!/usr/bin/env python3
"""Plot diagnostics for regularizing the SO2 three-anchor P fit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import ultraplot as uplt


FLOOR = Path(
    "/private/tmp/so2_cas6e6o_631gstar_2p1s_three_anchor_"
    "fitted_p_floor_repro_9x9x9_20fs/so2_cgldr_dynamics_dense.npz"
)
HARMONIC = Path(
    "/private/tmp/so2_cas6e6o_631gstar_2p1s_three_anchor_"
    "fitted_p_harmonic_9x9x9_20fs/so2_cgldr_dynamics_dense.npz"
)


def load(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def minimum_link_singular_value(data):
    singular_values = np.linalg.svd(data["qa_p_nearest_links"], compute_uv=False)
    return np.min(singular_values, axis=(-2, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas6e6o_631gstar_p_dressing_plots"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    floor = load(FLOOR)
    harmonic = load(HARMONIC)
    np.testing.assert_allclose(floor["qs"], harmonic["qs"])
    np.testing.assert_allclose(floor["theta"], harmonic["theta"])

    qs = floor["qs"]
    theta = np.rad2deg(floor["theta"])
    reliability = np.min(floor["qa_p_reliability_ratios"], axis=0)
    valid = reliability >= 1.0e-6
    link_floor = minimum_link_singular_value(floor)
    link_harmonic = minimum_link_singular_value(harmonic)

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.transparent": False,
        }
    )
    figure, axes = uplt.subplots(
        ncols=3,
        width=7.3,
        height=2.65,
        share=False,
        wspace=2.5,
    )
    axes[0].pcolormesh(
        qs,
        theta,
        valid.T.astype(float),
        shading="auto",
        cmap=ListedColormap(("#B2182B", "#F2F2F2")),
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].legend(
        handles=(
            Patch(facecolor="#F2F2F2", edgecolor="0.4", label="Reliable"),
            Patch(facecolor="#B2182B", edgecolor="none", label="Continued"),
        ),
        loc="ll",
        ncols=1,
        frame=False,
        fontsize=7.5,
    )

    link_mesh = None
    for axis, values in zip(
        axes[1:],
        (link_floor, link_harmonic),
    ):
        link_mesh = axis.pcolormesh(
            qs,
            theta,
            values.T,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=0.87,
        )
    Figure.colorbar(
        figure,
        mpl.cm.ScalarMappable(norm=Normalize(0.0, 0.87), cmap="viridis"),
        ax=[axes[1], axes[2]],
        fraction=0.035,
        pad=0.03,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        label=r"$\sigma_{\min}(P_{k,k+1})$",
    )

    titles = (
        r"$\mathbf{(a)}$  Anchor reliability",
        r"$\mathbf{(b)}$  Direct $\log P$",
        r"$\mathbf{(c)}$  Regularized $\log P$",
    )
    for axis, title in zip(axes, titles):
        axis.plot(
            np.sqrt(2.0) * 2.70,
            119.5,
            marker="x",
            markersize=5.0,
            markeredgewidth=1.0,
            color="black",
            zorder=3,
        )
        axis.format(
            title=title,
            titleloc="left",
            xlabel=r"$q_s$ (bohr)",
            tickdir="in",
            grid=False,
        )
    axes[0].format(ylabel=r"$\theta$ (deg)")
    for axis in axes[1:]:
        axis.format(ylabel="")

    png = args.output_dir / "so2_p_metric_regularization.png"
    pdf = args.output_dir / "so2_p_metric_regularization.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
