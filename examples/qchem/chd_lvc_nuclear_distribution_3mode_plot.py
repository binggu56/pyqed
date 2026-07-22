"""Plot time-resolved nuclear distributions from the CHD three-mode LVC model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("chd_c2_lvc_coupled_wavepacket_3mode.npz")
OUTPUT_SNAPSHOTS = Path("chd_c2_lvc_nuclear_distribution_3mode_snapshots")
OUTPUT_MARGINALS = Path("chd_c2_lvc_nuclear_distribution_3mode_marginals")


def total_density(pair_density):
    return pair_density.sum(axis=1)


def save_snapshots(grid, times, density_58, density_826):
    selected = np.arange(6)
    rows = [density_58[selected], density_826[selected]]
    row_labels = [(r"$q_5$", r"$q_8$"), (r"$q_8$", r"$q_{26}$")]
    vmax = max(float(row.max()) for row in rows)
    extent = [grid[0], grid[-1], grid[0], grid[-1]]
    fig, axes = plt.subplots(2, selected.size, figsize=(10.2, 3.75), sharex=True, sharey=True)
    for row_index, (row, labels) in enumerate(zip(rows, row_labels)):
        for column_index, snapshot_index in enumerate(selected):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                row[column_index].T, origin="lower", extent=extent, cmap="magma",
                vmin=0.0, vmax=vmax, interpolation="bilinear", aspect="equal",
            )
            if row_index == 0:
                axis.set_title(rf"{times[snapshot_index]:.0f} fs", pad=3)
            if column_index == 0:
                axis.set_ylabel(labels[1])
            if row_index == 1:
                axis.set_xlabel(labels[0])
            axis.set_xlim(-3.0, 3.0)
            axis.set_ylim(-3.0, 3.0)
            axis.tick_params(labelsize=7, length=2)
    axes[0, 0].text(-0.45, 1.05, "a", transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight="bold")
    axes[1, 0].text(-0.45, 1.05, "b", transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.018)
    colorbar.set_label("Marginal probability density")
    fig.suptitle("CHD three-mode LVC nuclear wavepacket", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.86, wspace=0.08, hspace=0.12)
    fig.savefig(OUTPUT_SNAPSHOTS.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_SNAPSHOTS.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_marginals(grid, times, marginal_density):
    total = marginal_density.sum(axis=1)
    mode_ids = [5, 8, 26]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True, sharey=True)
    for index, (axis, mode_id) in enumerate(zip(axes, mode_ids)):
        image = axis.pcolormesh(times, grid, total[:, index].T, shading="auto", cmap="magma")
        axis.set_ylabel(rf"$q_{{{mode_id}}}$")
        axis.set_ylim(-3.0, 3.0)
        axis.text(-0.10, 1.02, chr(ord("a") + index), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
        colorbar = fig.colorbar(image, ax=axis, pad=0.015, fraction=0.025)
        colorbar.set_label("Probability density")
    axes[-1].set_xlabel("Time delay (fs)")
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.08, top=0.98, hspace=0.14)
    fig.savefig(OUTPUT_MARGINALS.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_MARGINALS.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    with np.load(DATA) as data:
        grid = data["dimensionless_grid"]
        times = data["times_fs"]
        snapshot_times = data["snapshot_times_fs"]
        density_58 = total_density(data["pair_density_modes_5_8"])
        density_826 = total_density(data["pair_density_modes_8_26"])
        marginal_density = data["marginal_density_1d"]
    save_snapshots(grid, snapshot_times, density_58, density_826)
    save_marginals(grid, times, marginal_density)


if __name__ == "__main__":
    main()
