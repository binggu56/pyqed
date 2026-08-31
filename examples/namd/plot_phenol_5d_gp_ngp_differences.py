#!/usr/bin/env python3
"""Plot coordinate-resolved differences from the phenol 5D GP/NGP driver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "dataset/phenol_5d_production/dynamics/gp_ngp_pilot_phase_only"
)
COLORS = {"gp": "#0072B2", "ngp": "#D55E00"}


def plot(directory):
    directory = Path(directory)
    data = np.load(directory / "phenol_5d_gp_ngp.npz")
    summary = json.loads((directory / "summary.json").read_text())
    axes = [data[f"axis_{axis}"] for axis in range(5)]
    gp = [data[f"final_marginal_{axis}_gp"] for axis in range(5)]
    ngp = [data[f"final_marginal_{axis}_ngp"] for axis in range(5)]
    initial = [data[f"initial_marginal_{axis}"] for axis in range(5)]
    variation = 0.5 * np.asarray(
        [np.sum(np.abs(left - right)) for left, right in zip(gp, ngp)]
    )
    fidelity = float(summary["dynamics"]["final_gp_ngp_fidelity"])
    time_fs = float(summary["dynamics"]["time_fs"])

    figure, panels = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    panels[0, 0].plot(axes[1], gp[1], color=COLORS["gp"], label="GP")
    panels[0, 0].plot(axes[1], ngp[1], color=COLORS["ngp"], label="NGP")
    panels[0, 0].plot(axes[1], initial[1], "--", color="0.35", label="initial")
    panels[0, 1].axhline(0.0, color="0.65", linewidth=0.8)
    panels[0, 1].plot(axes[1], gp[1] - ngp[1], color="#009E73")
    labels = [r"$R_{OH}$", r"$\phi$", r"$\theta$", r"$q_{16}$", r"$q_8$"]
    panels[1, 0].bar(np.arange(5), 100.0 * variation, color="#6A51A3")
    panels[1, 1].axhline(0.0, color="0.65", linewidth=0.8)
    panels[1, 1].plot(axes[0], gp[0] - ngp[0], color="#CC79A7")

    panels[0, 0].set(
        xlabel=r"$\phi$ (rad)", ylabel="probability",
        title="Torsional marginal",
    )
    panels[0, 1].set(
        xlabel=r"$\phi$ (rad)", ylabel=r"$P_{\rm GP}-P_{\rm NGP}$",
        title="GP-induced torsional redistribution",
    )
    panels[1, 0].set(
        xticks=np.arange(5), xticklabels=labels,
        ylabel="total-variation distance (%)",
        title="Coordinate-resolved GP difference",
    )
    panels[1, 1].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
        ylabel=r"$P_{\rm GP}-P_{\rm NGP}$",
        title="Radial redistribution",
    )
    for label, panel in zip("abcd", panels.flat):
        panel.grid(alpha=0.18)
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
    panels[0, 0].legend(frameon=False)
    figure.suptitle(
        rf"Phenol 5D phase-only GP/NGP pilot at {time_fs:g} fs: "
        rf"$|\langle\Psi_{{GP}}|\Psi_{{NGP}}\rangle|^2={fidelity:.4f}$"
    )
    png = directory / "phenol_5d_gp_ngp_coordinate_differences.png"
    pdf = directory / "phenol_5d_gp_ngp_coordinate_differences.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    print(
        json.dumps(
            {
                "fidelity": fidelity,
                "total_variation": dict(zip(("R_OH", "phi", "theta", "q16", "q8"), variation)),
                "png": str(png),
                "pdf": str(pdf),
            },
            indent=2,
            default=float,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_INPUT)
    plot(parser.parse_args().directory)


if __name__ == "__main__":
    main()
