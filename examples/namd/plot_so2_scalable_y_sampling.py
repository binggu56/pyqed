#!/usr/bin/env python3
"""Compare full-grid SO2 endpoint errors for scalable sampling designs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = (
    Path("/private/tmp/so2_casci_sobol_probes_y.json"),
    Path("/private/tmp/so2_casci_sobol_probes_y_25.json"),
    Path("/private/tmp/so2_casci_anisotropic_y.json"),
    Path("/private/tmp/so2_casci_anisotropic_y_13x5.json"),
)


def load_results(paths):
    labels = ("Sobol+probes 17", "Sobol+probes 25", "anisotropic 9x5", "anisotropic 13x5")
    rows = []
    for label, path in zip(labels, paths):
        values = json.loads(Path(path).read_text())
        rows.append({"label": label, **values})
    return rows


def plot(rows, output):
    colors = ("#56B4E9", "#0072B2", "#E69F00", "#D55E00")
    markers = ("o", "o", "s", "s")
    samples = np.asarray([row["total_geometries"] for row in rows])
    energy = np.asarray([row["energy_mae_ev"] for row in rows])
    link = np.asarray([row["link_magnitude_axis_rms"] for row in rows])
    training = np.asarray([row["training_link_rms"] for row in rows])

    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), constrained_layout=True)
    for row, count, value, color, marker in zip(rows, samples, energy, colors, markers):
        axes[0].scatter(count, value, color=color, marker=marker, s=56, label=row["label"])
    axes[0].set(xlabel="CASCI geometries", ylabel="Energy MAE (eV)", yscale="log")
    axes[0].legend(frameon=False, fontsize=7, handletextpad=0.4)

    for row, count, value, color, marker in zip(rows, samples, link[:, 2], colors, markers):
        axes[1].scatter(count, value, color=color, marker=marker, s=56)
    axes[1].set(
        xlabel="CASCI geometries", ylabel=r"Bend-link $|L_{ij}|$ RMS", yscale="log"
    )

    for row, x, y, color, marker in zip(rows, training, link[:, 2], colors, markers):
        axes[2].scatter(x, y, color=color, marker=marker, s=56)
        axes[2].annotate(
            str(row["total_geometries"]), (x, y), xytext=(4, 3),
            textcoords="offset points", fontsize=7,
        )
    limits = [min(training.min(), link[:, 2].min()) * 0.7, max(training.max(), link[:, 2].max()) * 1.4]
    axes[2].plot(limits, limits, color="0.5", ls="--", lw=1, label="train = full grid")
    axes[2].set(
        xlabel="Training graph link RMS", ylabel=r"Full-grid bend-link RMS",
        xscale="log", yscale="log", xlim=limits, ylim=limits,
    )
    axes[2].legend(frameon=False, fontsize=7)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18, which="both")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/so2_scalable_y_sampling_comparison.png"),
    )
    args = parser.parse_args()
    if len(args.results) != 4:
        raise ValueError("provide four result JSON files in the documented order")
    rows = load_results(args.results)
    plot(rows, args.output)
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
