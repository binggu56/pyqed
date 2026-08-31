#!/usr/bin/env python3
"""Plot periodic GDF/KRHF/GW validation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2mev

HARTREE_TO_MEV = au2mev


def _row(path):
    payload = json.loads(Path(path).read_text())
    studies = payload.get("studies", [])
    if len(studies) != 1:
        raise ValueError(f"{path} must contain exactly one validation study")
    return studies[0]


def _errors(row):
    values = [
        abs(float(row["max_abs_J_error_Ha"])) * HARTREE_TO_MEV,
        abs(float(row["max_abs_K_error_Ha"])) * HARTREE_TO_MEV,
        abs(float(row["native_krhf"]["energy_error_vs_pyscf_gdf_Ha"]))
        * HARTREE_TO_MEV,
        abs(float(row["gw"]["max_abs_qp_error_Ha"])) * HARTREE_TO_MEV,
    ]
    return np.maximum(values, np.finfo(float).tiny)


def _timings(row):
    return np.asarray(
        [
            float(row["pyscf_gdf_build_seconds"]),
            float(row["native_gdf_seconds"]),
            float(row["pyscf_gdf_krhf_seconds"]),
            float(row["native_krhf"]["seconds"]),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--labels", help="comma-separated run labels")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_validation_comparison.pdf"),
    )
    args = parser.parse_args()

    rows = [_row(path) for path in args.inputs]
    labels = (
        [item.strip() for item in args.labels.split(",")]
        if args.labels
        else [path.stem for path in args.inputs]
    )
    if len(labels) != len(rows):
        parser.error("--labels must provide one label per input")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    colors = ["#4C78A8", "#E45756", "#59A14F", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.24, top=0.8, wspace=0.34)

    error_labels = ["$J$", "$K$", "KRHF $E$", "GW QP"]
    x = np.arange(len(error_labels), dtype=float)
    width = 0.72 / len(rows)
    for index, (row, label) in enumerate(zip(rows, labels)):
        offset = (index - 0.5 * (len(rows) - 1)) * width
        axes[0].bar(
            x + offset,
            _errors(row),
            width=width,
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Absolute discrepancy (meV)")
    axes[0].set_xticks(x, error_labels)
    axes[0].grid(axis="y", which="major", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(2, len(rows)),
    )
    axes[0].text(-0.16, 1.12, "a", transform=axes[0].transAxes, fontweight="bold")

    timing_labels = ["PySCF\nGDF", "PyQED\nGDF", "PySCF\nGDF+KRHF", "PyQED\nGDF+KRHF"]
    timing_x = np.arange(len(timing_labels), dtype=float)
    timing_max = 0.0
    for index, row in enumerate(rows):
        offset = (index - 0.5 * (len(rows) - 1)) * width
        values = _timings(row)
        bars = axes[1].bar(
            timing_x + offset,
            values,
            width=width,
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].bar_label(
            bars,
            fmt="%.1f",
            label_type="center",
            fontsize=6.5,
            color="white",
        )
        timing_max = max(timing_max, float(np.max(values)))
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_xticks(timing_x, timing_labels)
    axes[1].set_ylim(0.0, 1.18 * timing_max)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].text(-0.16, 1.12, "b", transform=axes[1].transAxes, fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
