#!/usr/bin/env python3
"""Plot raw Procrustes-aligned SO2 CASCI nearest-neighbor links."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.so2_mace_ttldr import aligned_fields


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


def _edge_axis(grid, axis):
    values = 0.5 * (grid[:-1] + grid[1:])
    return np.rad2deg(values) if axis == 2 else values


def _endpoint_labels(grid, axis):
    values = np.rad2deg(grid) if axis == 2 else grid
    precision = 1 if axis == 2 else 2
    return [
        "$" + f"{left:.{precision}f}" + "$\n"
        + r"$\downarrow$" + "\n$" + f"{right:.{precision}f}" + "$"
        for left, right in zip(values[:-1], values[1:])
    ]


def _samples(values, axis):
    moved = np.moveaxis(values, axis, 0)
    return moved.reshape(moved.shape[0], -1, *values.shape[-2:])


def _center_slice(values, axis):
    selection = [slice(None) if coordinate == axis else size // 2
                 for coordinate, size in enumerate(values.shape[:-2])]
    return values[tuple(selection)]


def plot(dataset, output):
    grids, energy, links, _geometry = aligned_fields(dataset)
    labels = (r"$r_1$", r"$r_2$", r"$\theta$")
    units = ("bohr", "bohr", "degree")
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(9.4, 7.4),
        constrained_layout=True,
    )

    diagonal = ((0, 0), (1, 1), (2, 2))
    off_diagonal = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
    for axis, values in enumerate(links):
        x = _edge_axis(grids[axis], axis)
        endpoint_labels = _endpoint_labels(grids[axis], axis)
        cloud = _samples(values, axis)
        center = _center_slice(values, axis)

        for color, marker, (left, right) in zip(COLORS, MARKERS, diagonal):
            axes[axis, 0].scatter(
                np.repeat(x, cloud.shape[1]),
                cloud[:, :, left, right].real.ravel(),
                s=7,
                color=color,
                alpha=0.14,
                edgecolors="none",
            )
            axes[axis, 0].plot(
                x,
                center[:, left, right].real,
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.25,
                label=rf"$\bar L_{{{left}{right}}}$",
            )

        for color, marker, (left, right) in zip(COLORS, MARKERS, off_diagonal):
            axes[axis, 1].scatter(
                np.repeat(x, cloud.shape[1]),
                cloud[:, :, left, right].real.ravel(),
                s=7,
                color=color,
                alpha=0.14,
                edgecolors="none",
            )
            axes[axis, 1].plot(
                x,
                center[:, left, right].real,
                color=color,
                marker=marker,
                ms=3.0,
                lw=1.15,
                label=rf"$\bar L_{{{left}{right}}}$",
            )

        singular = np.linalg.svd(cloud, compute_uv=False)
        center_singular = np.linalg.svd(center, compute_uv=False)
        for state, (color, marker) in enumerate(
            zip(COLORS[: center.shape[-1]], MARKERS[: center.shape[-1]])
        ):
            axes[axis, 2].scatter(
                np.repeat(x, singular.shape[1]),
                singular[:, :, state].ravel(),
                s=7,
                color=color,
                alpha=0.14,
                edgecolors="none",
            )
            axes[axis, 2].plot(
                x,
                center_singular[:, state],
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.25,
                label=rf"$\sigma_{state + 1}$",
            )

        for column in range(3):
            axes[axis, column].set(
                xlabel=rf"{labels[axis]} link endpoints ({units[axis]})",
                ylabel=(
                    rf"$\bar L_{{{labels[axis][1:-1]}}}$ diagonal"
                    if column == 0
                    else rf"$\bar L_{{{labels[axis][1:-1]}}}$ off-diagonal"
                    if column == 1
                    else "Singular value"
                ),
            )
            axes[axis, column].set_xticks(x, endpoint_labels)
            axes[axis, column].spines[["top", "right"]].set_visible(False)
            axes[axis, column].tick_params(direction="out", axis="x", labelsize=7.5)

    axes[0, 0].set_title("Diagonal elements")
    axes[0, 1].set_title("Signed off-diagonal elements")
    axes[0, 2].set_title("Link singular values")
    axes[0, 0].legend(ncols=3, frameon=False, fontsize=7.5, loc="lower left")
    axes[0, 1].legend(ncols=2, frameon=False, fontsize=7.0, loc="best")
    axes[0, 2].legend(ncols=3, frameon=False, fontsize=7.5, loc="lower left")
    for label, axis_handle in zip("abcdefghi", axes.ravel()):
        axis_handle.text(
            0.02,
            0.98,
            label,
            transform=axis_handle.transAxes,
            va="top",
            fontweight="bold",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    np.savez(
        output.with_suffix(".npz"),
        r1=grids[0],
        r2=grids[1],
        theta=grids[2],
        aligned_energy=energy,
        aligned_link_r1=links[0],
        aligned_link_r2=links[1],
        aligned_link_theta=links[2],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_casci_aligned_links.png"),
    )
    args = parser.parse_args()
    plot(args.dataset, args.output)
    print(f"figure: {args.output}")
    print(f"data: {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
