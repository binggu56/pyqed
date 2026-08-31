#!/usr/bin/env python3
"""Plot timing and numerical checks for periodic GDF workflow JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    path = Path(path)
    data = json.loads(path.read_text())
    npz_path = Path(data.get("npz", path.with_suffix(".npz")))
    return data, np.load(npz_path)


def _max_delta(left, right):
    deltas = []
    for key in left.files:
        if key not in right.files or not np.issubdtype(left[key].dtype, np.number):
            continue
        deltas.append(float(np.max(np.abs(left[key] - right[key]))))
    return max(deltas, default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--intermediate", type=Path)
    parser.add_argument("--preoptimized", type=Path)
    parser.add_argument("--baseline-label", default="One pair")
    parser.add_argument("--intermediate-label", default="Pair batch")
    parser.add_argument("--preoptimized-label", default="Multi-q + metric")
    parser.add_argument("--optimized-label", default="Pair-FT reuse")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_batch_benchmark.pdf"),
    )
    args = parser.parse_args()

    baseline, baseline_npz = _load(args.baseline)
    optimized, optimized_npz = _load(args.optimized)
    datasets = [baseline]
    labels = [args.baseline_label]
    if args.intermediate is not None:
        intermediate, _intermediate_npz = _load(args.intermediate)
        datasets.append(intermediate)
        labels.append(args.intermediate_label)
    if args.preoptimized is not None:
        preoptimized, _preoptimized_npz = _load(args.preoptimized)
        datasets.append(preoptimized)
        labels.append(args.preoptimized_label)
    datasets.append(optimized)
    labels.append(args.optimized_label)
    phases = [
        ("GDF build", "gdf_prebuild_seconds"),
        ("KRHF", "krhf_seconds_after_prebuild"),
        ("GW", "gw_seconds"),
    ]
    phase_times = np.asarray(
        [
            [dataset["timings"][key] for _label, key in phases]
            for dataset in datasets
        ],
        dtype=float,
    )
    energy_delta = abs(
        optimized["krhf_energy_Ha_per_cell"]
        - baseline["krhf_energy_Ha_per_cell"]
    )
    qp_delta = float(
        np.max(
            np.abs(
                np.asarray(optimized["qp_energy_Ha"], dtype=float)
                - np.asarray(baseline["qp_energy_Ha"], dtype=float)
            )
        )
    )
    npz_delta = _max_delta(baseline_npz, optimized_npz)

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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.84, wspace=0.34)

    x = np.arange(len(phases))
    width = 0.78 / len(datasets)
    dataset_colors = ["#7A8793", "#E69F00", "#4C78A8", "#009E73"]
    if len(datasets) == 2:
        dataset_colors = [dataset_colors[0], dataset_colors[-1]]
    elif len(datasets) == 3:
        dataset_colors = [dataset_colors[0], dataset_colors[1], dataset_colors[-1]]
    ax = axes[0]
    offsets = (np.arange(len(datasets)) - 0.5 * (len(datasets) - 1)) * width
    for index, (label, color, offset) in enumerate(
        zip(labels, dataset_colors, offsets)
    ):
        ax.bar(
            x + offset,
            phase_times[index],
            width,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [label for label, _key in phases])
    ax.set_ylabel("Wall time (s)")
    ax.set_title("LiH $2\\times2\\times2$ workflow")
    ax.grid(axis="y", color="0.9", linewidth=0.7, which="both")
    ax.legend(frameon=False)
    speedup = phase_times[0, 0] / phase_times[-1, 0]
    ax.text(
        offsets[-1],
        phase_times[-1, 0] * 1.20,
        f"{speedup:.2f}$\\times$",
        ha="center",
        va="bottom",
        color="#006B4F",
        fontweight="bold",
    )

    ax = axes[1]
    totals = np.sum(phase_times, axis=1)
    bottoms = np.zeros(len(datasets))
    colors = ["#4C78A8", "#F2CF5B", "#E45756"]
    for index, ((label, _key), color) in enumerate(zip(phases, colors)):
        values = phase_times[:, index]
        ax.bar(
            np.arange(len(datasets)),
            values,
            bottom=bottoms,
            color=color,
            label=label,
        )
        bottoms += values
    tick_labels = [label.replace(" ", "\n", 1) for label in labels]
    ax.set_xticks(np.arange(len(datasets)), tick_labels)
    ax.set_ylabel("GDF + KRHF + GW wall time (s)")
    max_energy_delta = max(energy_delta, qp_delta)
    ax.set_title(
        f"End-to-end cost\nmax energy delta = {max_energy_delta:.1e} Ha"
    )
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    for index, total in enumerate(totals):
        ax.text(index, total + 0.02 * totals.max(), f"{total:.1f}", ha="center")
    ax.set_ylim(0.0, 1.18 * totals.max())

    for label, ax in zip(("a", "b"), axes):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"Max NPZ array delta: {npz_delta:.6e}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
