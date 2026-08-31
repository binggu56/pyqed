#!/usr/bin/env python3
"""Plot held-out diagnostics for fitted-Y pyrazine sampling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in args.summaries]
    records.sort(key=lambda record: record["sampled_geometries"])
    samples = np.asarray([record["sampled_geometries"] for record in records])
    fractions = 100.0 * np.asarray([record["geometry_fraction"] for record in records])
    held_links = np.asarray(
        [record["rms_held_out_relative_link_error"] for record in records],
        dtype=float,
    )
    hamiltonian = np.asarray(
        [record["fitted_vs_exact_path_hamiltonian_error"] for record in records]
    )
    potential = np.asarray(
        [record["rms_potential_block_error_Eh"] for record in records]
    )
    layouts = [record.get("sample_layout", "maximin") for record in records]
    colors = {
        "maximin": "#D1495B",
        "tensor": "#126782",
        "crosshatch": "#6A994E",
    }

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    values = (held_links, hamiltonian, potential)
    labels = (
        "Held-out link RMS error",
        r"$\|H_Y-H_{\rm path}\|/\|H_{\rm path}\|$",
        r"Potential RMS error / $E_{\rm h}$",
    )
    for label, axis, value in zip("abc", axes, values):
        for layout in dict.fromkeys(layouts):
            selected = np.asarray([item == layout for item in layouts])
            axis.scatter(
                samples[selected], value[selected], s=38,
                color=colors.get(layout, "#555555"), label=layout,
            )
        axis.set_yscale("log")
        axis.margins(y=0.25)
        for count, fraction, error, layout in zip(samples, fractions, value, layouts):
            axis.annotate(
                f"{layout}, {fraction:.0f}%",
                (count, error),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set(xlabel="Sampled geometries", ylabel=labels[ord(label) - ord("a")])
        axis.text(
            0.03, 0.96, f"({label})", transform=axis.transAxes,
            va="top", fontweight="bold",
        )
        axis.grid(False)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
