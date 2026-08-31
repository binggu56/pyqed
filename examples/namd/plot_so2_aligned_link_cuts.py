#!/usr/bin/env python3
"""Plot complete central cuts of the raw Procrustes-aligned SO2 links."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def cell_edges(points):
    points = np.asarray(points, dtype=float)
    edges = np.empty(len(points) + 1)
    edges[1:-1] = 0.5 * (points[:-1] + points[1:])
    edges[0] = points[0] - 0.5 * (points[1] - points[0])
    edges[-1] = points[-1] + 0.5 * (points[-1] - points[-2])
    return edges


def central_cut(values, varied_axis, center):
    selection = [min(index, size - 1) for index, size in zip(center, values.shape[:3])]
    selection[varied_axis] = slice(None)
    return values[tuple(selection)].reshape(values.shape[varied_axis], -1).T


def plot(grids_path, gauge_path, output):
    with np.load(grids_path, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    with np.load(gauge_path, allow_pickle=False) as archive:
        links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
        center = tuple(map(int, archive["center"]))
        boundaries = tuple(
            int(archive[name])
            for name in ("low_boundary_theta_index", "high_boundary_theta_index")
            if name in archive
        )

    maximum_imaginary = max(float(np.max(np.abs(values.imag))) for values in links)
    if maximum_imaginary > 1.0e-12:
        raise ValueError(
            "aligned links are complex; this real-channel plot would omit information"
        )

    cuts = {}
    coordinates = {}
    for direction, values in enumerate(links):
        for varied_axis, grid in enumerate(grids):
            coordinates[direction, varied_axis] = (
                0.5 * (grid[:-1] + grid[1:])
                if direction == varied_axis
                else grid
            )
            cuts[direction, varied_axis] = central_cut(
                values.real, varied_axis, center
            )

    labels = (r"$q_s$ (bohr)", r"$\theta$ (degree)", r"$q_a$ (bohr)")
    directions = (r"$\bar L_{q_s}$", r"$\bar L_\theta$", r"$\bar L_{q_a}$")
    channels = [rf"${left + 1}{right + 1}$" for left in range(4) for right in range(4)]
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(8.6, 6.5),
        sharey=True,
        constrained_layout=True,
    )

    for direction in range(3):
        limit = max(
            float(np.max(np.abs(cuts[direction, varied_axis])))
            for varied_axis in range(3)
        )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for varied_axis in range(3):
            axis = axes[direction, varied_axis]
            x = coordinates[direction, varied_axis]
            if varied_axis == 1:
                x = np.rad2deg(x)
            image = axis.pcolormesh(
                cell_edges(x),
                np.arange(17) - 0.5,
                cuts[direction, varied_axis],
                cmap="RdBu_r",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            axis.invert_yaxis()
            axis.set_xlabel(labels[varied_axis])
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(direction="out")
            if direction == 0:
                axis.set_title(rf"Cut along {labels[varied_axis].split(' (')[0]}", fontsize=9)
            if varied_axis == 1:
                for boundary in boundaries:
                    location = np.rad2deg(
                        0.5 * (grids[1][boundary] + grids[1][boundary + 1])
                    )
                    axis.axvline(location, color="k", lw=0.8, ls="--")
        axes[direction, 0].text(
            0.98,
            0.04,
            directions[direction],
            transform=axes[direction, 0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        colorbar = figure.colorbar(image, ax=axes[direction], pad=0.015, shrink=0.92)
        colorbar.set_label(r"$\mathrm{Re}\,\bar L_{\mu,ij}$", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    axes[0, 0].set_yticks(np.arange(16), channels, fontsize=6.5)
    for direction in (1, 2):
        axes[direction, 0].tick_params(labelleft=False)
    figure.supylabel("Matrix element", x=0.005, fontsize=9)
    for label, axis in zip("abcdefghi", axes.ravel()):
        axis.text(
            -0.01,
            1.02,
            label,
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontweight="bold",
            color="black",
            clip_on=False,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        type=Path,
        default=Path("/private/tmp/so2_9x9x9_grids.npz"),
    )
    parser.add_argument(
        "--gauge",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas4state_three_patch_9x9x9/procrustes_gauge.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas4state_three_patch_9x9x9/"
            "so2_bar_link_raw_cuts.png"
        ),
    )
    args = parser.parse_args()
    plot(args.grids, args.gauge, args.output)
    print(f"figure: {args.output}")
    print(f"vector: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
