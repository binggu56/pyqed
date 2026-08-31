#!/usr/bin/env python3
"""Plot fixed and adaptive 49-geometry pyrazine SG-LDR benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(directory):
    directory = Path(directory)
    with (directory / "summary.json").open() as stream:
        summary = json.load(stream)
    data = np.load(directory / "pyrazine_casci_abinitio_ttldr.npz")
    return summary, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed",
        type=Path,
        default=Path("/private/tmp/pyrazine_casci_tensor_y_n49_d4_10fs"),
    )
    parser.add_argument(
        "--adaptive",
        type=Path,
        default=Path(
            "/private/tmp/pyrazine_casci_adaptive_weighted_y_n49_d4_10fs"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyrazine_adaptive_sampling_comparison"),
    )
    args = parser.parse_args()

    fixed_summary, fixed = load(args.fixed)
    adaptive_summary, adaptive = load(args.adaptive)
    history = adaptive_summary["abinitio_fit"]["history"]
    initial_count = adaptive_summary["abinitio_fit"]["initial_geometries"]
    all_points = np.asarray(adaptive_summary["abinitio_fit"]["points"])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), constrained_layout=True)
    grid = np.asarray(list(np.ndindex(tuple(adaptive_summary["grid"]))))
    axes[0].scatter(grid[:, 0], grid[:, 1], s=9, color="0.86", zorder=0)
    initial = all_points[:initial_count]
    axes[0].scatter(
        initial[:, 0], initial[:, 1], s=25, marker="s", color="0.15",
        label="Initial (25)", zorder=2,
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    for color, record in zip(colors, history[:-1]):
        selected = np.asarray(record["acquisition"]["selected"])
        axes[0].scatter(
            selected[:, 0], selected[:, 1], s=31, color=color,
            edgecolor="white", linewidth=0.45,
            label=f"Batch {record['round'] + 1}", zorder=3,
        )
    axes[0].set(
        xlabel=r"$Q_0$ index", ylabel=r"$Q_1$ index",
        xlim=(-0.6, 10.6), ylim=(-0.6, 10.6), aspect="equal",
    )
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    times = adaptive["times_fs"]
    direct = adaptive["exact_path_populations"]
    fixed_pop = fixed["fitted_dense_populations"]
    adaptive_pop = adaptive["fitted_dense_populations"]
    state_colors = ("#0072B2", "#D55E00", "#009E73")
    for state, color in zip((1, 2, 3), state_colors):
        axes[1].plot(times, direct[:, state], color=color, lw=1.6)
        axes[1].plot(times, fixed_pop[:, state], color=color, ls=":", lw=1.5)
        axes[1].plot(times, adaptive_pop[:, state], color=color, ls="--", lw=1.5)
    for state, color in zip((1, 2, 3), state_colors):
        axes[1].plot([], [], color=color, label=f"S{state}")
    axes[1].plot([], [], color="0.2", label="Direct LDR")
    axes[1].plot([], [], color="0.2", ls=":", label="Fixed 49")
    axes[1].plot([], [], color="0.2", ls="--", label="Adaptive 49")
    axes[1].set(
        xlabel="Time / fs", ylabel="Adiabatic population", ylim=(-0.03, 1.03)
    )
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="center right")

    fixed_error = np.max(np.abs(fixed_pop - fixed["exact_path_populations"]), axis=1)
    adaptive_error = np.max(np.abs(adaptive_pop - direct), axis=1)
    axes[2].plot(times, fixed_error, color="#D55E00", ls=":", label="Fixed 49")
    axes[2].plot(
        times, adaptive_error, color="#0072B2", ls="--", label="Adaptive 49"
    )
    axes[2].set(
        xlabel="Time / fs", ylabel="Maximum population error"
    )
    axes[2].set_ylim(bottom=0.0)
    axes[2].legend(frameon=False, fontsize=8)

    for label, axis in zip("abc", axes):
        axis.text(
            0.02, 0.97, f"({label})", transform=axis.transAxes,
            va="top", fontweight="bold",
        )
        axis.grid(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=350)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
