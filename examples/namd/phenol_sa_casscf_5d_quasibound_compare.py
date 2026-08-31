#!/usr/bin/env python3
"""Compare the original and exact-point-corrected phenol 5D quasibound states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("dataset/phenol_5d_production/states")
DEFAULT_ORIGINAL = ROOT / "s1_origin_5d_quasibound_localwell_h3"
DEFAULT_CORRECTED = ROOT / "s1_origin_5d_quasibound_localwell_h3_corrected"
WAVENUMBER_PER_HARTREE = 219474.6313632


def _load(directory: Path):
    data = np.load(directory / "phenol_sa_casscf_5d_s1_quasibound.npz")
    summary = json.loads((directory / "summary.json").read_text())
    axes = [np.asarray(data[f"axis_{site}"]) for site in range(5)]
    marginals = [np.asarray(data[f"marginal_{site}"]) for site in range(5)]
    return axes, marginals, summary


def _energy_residual(summary):
    history = summary["imaginary_time_history"]
    time = np.asarray([row["imaginary_time_au"] for row in history])
    energy = np.asarray([row["energy_hartree"] for row in history])
    residual = np.maximum(np.abs(energy - energy[-1]) * WAVENUMBER_PER_HARTREE, 1.0e-8)
    return time, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--corrected", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    old_axes, old_marginals, old_summary = _load(args.original)
    axes, marginals, summary = _load(args.corrected)
    for old_axis, axis in zip(old_axes, axes):
        if not np.allclose(old_axis, axis):
            raise ValueError("the two states must use the same coordinate grids")

    figure, panels = plt.subplots(2, 3, figsize=(13.0, 7.3), constrained_layout=True)
    plot_axes = (axes[0], axes[1], np.rad2deg(axes[2]), axes[3], axes[4])
    labels = (
        r"$R_{OH}$ (angstrom)",
        r"$q_1=\phi$ (rad)",
        r"$q_2=\theta$ (deg)",
        r"$q_3=Q_{16a}$",
        r"$q_4=Q_{8a}$",
    )
    titles = (
        "O–H stretch",
        "Torsion",
        "O–H bend",
        "Ring deformation",
        "Ring deformation",
    )
    for panel, axis, old, new, label, title in zip(
        panels.flat[:5], plot_axes, old_marginals, marginals, labels, titles
    ):
        panel.plot(axis, old, "--", color="0.45", linewidth=1.8, label="Original fit")
        panel.plot(axis, new, "o-", color="C0", markersize=3.0, label="Exact-point refit")
        panel.fill_between(axis, 0.0, new, color="C0", alpha=0.10)
        panel.set(xlabel=label, ylabel="probability", title=title)
    panels[0, 0].legend(frameon=False)

    old_time, old_residual = _energy_residual(old_summary)
    time, residual = _energy_residual(summary)
    panels[1, 2].semilogy(old_time, old_residual, "--", color="0.45", label="Original fit")
    panels[1, 2].semilogy(time, residual, "-", color="C0", label="Exact-point refit")
    panels[1, 2].set(
        xlabel=r"imaginary time $\tau$ (a.u.)",
        ylabel=r"$|E-E_f|$ (cm$^{-1}$)",
        title="Eigenstate convergence",
    )
    panels[1, 2].legend(frameon=False)

    figure.suptitle(
        "Phenol 5D $S_1$ local-well state: removal of the fitted bimodality",
        fontsize=14,
    )
    output = args.output or args.corrected / "phenol_sa_casscf_5d_quasibound_comparison"
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
