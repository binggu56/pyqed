#!/usr/bin/env python3
"""Plot central coordinate cuts of the raw Procrustes-aligned SO2 energy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


def central_cut(values, axis, center):
    selection = list(center)
    selection[axis] = slice(None)
    return values[tuple(selection)]


def plot(grids_path, gauge_path, output):
    with np.load(grids_path, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    with np.load(gauge_path, allow_pickle=False) as archive:
        energy = np.asarray(archive["aligned_local_hamiltonian"])
        center = tuple(map(int, archive["center"]))
        if "patch_boundary_theta_index" in archive:
            patch_boundaries = (int(archive["patch_boundary_theta_index"]),)
        else:
            patch_boundaries = tuple(
                int(archive[name])
                for name in (
                    "low_boundary_theta_index",
                    "high_boundary_theta_index",
                )
                if name in archive
            )

    labels = (r"$q_s$ (bohr)", r"$\theta$ (degree)", r"$q_a$ (bohr)")
    nstates = energy.shape[-1]
    diagonal = tuple((state, state) for state in range(nstates))
    off_diagonal = tuple(
        (left, right)
        for left in range(nstates)
        for right in range(left + 1, nstates)
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(8.2, 4.8),
        sharex="col",
        constrained_layout=True,
    )

    for axis, grid in enumerate(grids):
        x = np.rad2deg(grid) if axis == 1 else grid
        cut = central_cut(energy, axis, center).real
        for color, marker, (left, right) in zip(COLORS, MARKERS, diagonal):
            axes[0, axis].plot(
                x,
                cut[:, left, right],
                color=color,
                marker=marker,
                ms=4.0,
                lw=1.35,
                label=rf"$\bar E_{{{left + 1}{right + 1}}}$",
            )
        for color, marker, (left, right) in zip(COLORS, MARKERS, off_diagonal):
            axes[1, axis].plot(
                x,
                cut[:, left, right],
                color=color,
                marker=marker,
                ms=4.0,
                lw=1.35,
                label=rf"$\bar E_{{{left + 1}{right + 1}}}$",
            )

        axes[1, axis].axhline(0.0, color="0.55", lw=0.7, zorder=0)
        axes[1, axis].set_xlabel(labels[axis])
        for row in range(2):
            axes[row, axis].grid(axis="y", color="0.9", linewidth=0.6)
            axes[row, axis].spines[["top", "right"]].set_visible(False)
            axes[row, axis].tick_params(direction="out")

    for patch_boundary in patch_boundaries:
        boundary = 0.5 * (
            grids[1][patch_boundary] + grids[1][patch_boundary + 1]
        )
        boundary = np.rad2deg(boundary)
        for row in range(2):
            axes[row, 1].axvline(
                boundary,
                color="0.35",
                lw=0.9,
                ls="--",
            )

    axes[0, 0].set_ylabel(r"Diagonal $\bar E_{ii}$ ($E_h$)")
    axes[1, 0].set_ylabel(r"Off-diagonal $\bar E_{ij}$ ($E_h$)")
    axes[0, 0].legend(frameon=False, ncols=2, fontsize=7.5, loc="lower center")
    axes[1, 0].legend(frameon=False, ncols=3, fontsize=7.2, loc="upper center")
    for label, axis_handle in zip("abcdef", axes.ravel()):
        axis_handle.text(
            0.01,
            1.04,
            label,
            transform=axis_handle.transAxes,
            va="top",
            fontweight="bold",
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
            "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
            "procrustes_gauge.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_bar_energy_raw_cuts.png"),
    )
    args = parser.parse_args()
    plot(args.grids, args.gauge, args.output)
    print(f"figure: {args.output}")
    print(f"vector: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
