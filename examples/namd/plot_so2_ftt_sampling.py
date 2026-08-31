#!/usr/bin/env python3
"""Plot SO2 direct-link FTT convergence with electronic-structure samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in args.summaries]
    records.sort(key=lambda record: record["fit"]["sampled_geometries"])
    samples = np.asarray([record["fit"]["sampled_geometries"] for record in records])
    total = int(np.prod(records[0]["grid"]))
    energy = np.asarray([record["fit"]["energy_relative_error"] for record in records])
    links = np.asarray(
        [
            [record["fit"]["link_relative_error"][label] for record in records]
            for label in ("r1", "r2", "theta")
        ]
    )
    hamiltonian = np.asarray([record["hamiltonian_relative_error"] for record in records])
    population = np.asarray(
        [record["maximum_ttldr_population_error_vs_exact"] for record in records]
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    colors = ("#555555", "#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "^", "D")
    axes[0].plot(samples, 100.0 * energy, marker=markers[0], color=colors[0], label=r"$\bar E$")
    for values, color, marker, label in zip(
        links,
        colors[1:],
        markers[1:],
        (r"$\bar L_{r_1}$", r"$\bar L_{r_2}$", r"$\bar L_\theta$"),
    ):
        axes[0].plot(samples, 100.0 * values, marker=marker, color=color, label=label)
    axes[1].plot(samples, 100.0 * hamiltonian, "o-", color="#0072B2", label=r"$H_{\rm LDR}$")
    axes[1].plot(samples, 100.0 * population, "s-", color="#D55E00", label="Population")

    for label, axis in zip("ab", axes):
        axis.set_yscale("log")
        axis.set_xlabel(f"Sampled geometries (of {total})")
        axis.set_xticks(samples, [*(str(value) for value in samples[:-1]), f"{samples[-1]}\nfull cross"])
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.08, 1.02, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Relative field error (%)")
    axes[1].set_ylabel("End-to-end error (%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
