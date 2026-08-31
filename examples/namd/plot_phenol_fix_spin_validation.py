#!/usr/bin/env python3
"""Plot fixed-root ``fix_spin`` validation for the phenol SA-CASSCF scan."""

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

from pyqed.units import au2mev

HARTREE_TO_MEV = au2mev


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-anchor", type=Path, required=True)
    parser.add_argument("--reference-anchor", type=Path, required=True)
    parser.add_argument("--fixed-stretched", type=Path, required=True)
    parser.add_argument("--reference-stretched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path without suffix")
    args = parser.parse_args()

    fixed = [load(args.fixed_anchor), load(args.fixed_stretched)]
    reference = [load(args.reference_anchor), load(args.reference_stretched)]
    labels = [r"$R_{\rm eq}=0.970$ Å", r"$R_{\rm OH}=1.55$ Å"]
    colors = ["#0072B2", "#D55E00"]
    markers = ["o", "s"]
    errors = np.asarray(
        [(record["energies"] - ref["energies"]) * HARTREE_TO_MEV
         for record, ref in zip(fixed, reference)]
    )
    spins = np.abs(np.asarray([record["spins"] for record in fixed]))
    requested = np.asarray([int(record["ci_requested_nstates"]) for record in fixed])
    solved = np.asarray([int(record["ci_solved_nstates"]) for record in fixed])
    walls = np.asarray([float(record["wall_seconds"]) for record in fixed])

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, panels = plt.subplots(1, 3, figsize=(11.5, 3.75), constrained_layout=True)
    roots = np.arange(1, errors.shape[1] + 1)

    for index, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        panels[0].plot(
            roots,
            errors[index],
            marker=marker,
            color=color,
            lw=1.35,
            ms=4.5,
            label=label,
        )
        panels[1].semilogy(
            roots,
            np.maximum(spins[index], 1.0e-14),
            marker=marker,
            color=color,
            lw=1.35,
            ms=4.5,
            label=label,
        )

    panels[0].axhline(0.0, color="0.5", lw=0.9)
    panels[0].axhline(1.0, color="0.65", ls=":", lw=0.9)
    panels[0].axhline(-1.0, color="0.65", ls=":", lw=0.9)
    panels[0].set(
        xlabel="state root",
        ylabel=r"$E_{\rm PyQED}-E_{\rm PySCF}$ (meV)",
        title="a  Singlet-energy invariance",
        xticks=roots,
    )
    panels[0].legend(frameon=False, loc="best")

    panels[1].axhline(1.0e-6, color="0.45", ls=":", lw=1.0, label=r"$10^{-6}$")
    panels[1].set_ylim(1.0e-12, 3.0e-6)
    panels[1].set(
        xlabel="state root",
        ylabel=r"$|\langle \hat S^2\rangle|$",
        title="b  Spin purity",
        xticks=roots,
    )

    x = np.arange(len(labels))
    width = 0.34
    panels[2].bar(
        x - width / 2,
        requested,
        width,
        color="#56B4E9",
        edgecolor="0.25",
        linewidth=0.6,
        label="requested",
    )
    panels[2].bar(
        x + width / 2,
        solved,
        width,
        color="#009E73",
        edgecolor="0.25",
        linewidth=0.6,
        label="solved",
    )
    panels[2].set_ylim(0, 7.4)
    panels[2].set(
        ylabel="number of CI roots",
        title="c  Fixed Davidson window",
        xticks=x,
        xticklabels=[r"$R_{\rm eq}$", "1.55 Å"],
    )
    panels[2].legend(frameon=False, loc="upper left", ncol=2)
    for index, wall in enumerate(walls):
        panels[2].text(
            index,
            0.38,
            f"{wall:.2f} s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.25",
        )

    for panel in panels:
        panel.grid(axis="y", color="0.90", lw=0.65)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    data = args.output.with_suffix(".json")
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "method": "SA(6)-CASSCF(10e,10o)/6-31+G* with fix_spin(ss=0, shift=1.0)",
        "distances_angstrom": [0.96994, 1.55],
        "energy_errors_mev": errors.tolist(),
        "max_abs_energy_error_mev": np.max(np.abs(errors), axis=1).tolist(),
        "spin_square": spins.tolist(),
        "max_abs_spin_square": np.max(spins, axis=1).tolist(),
        "requested_roots": requested.tolist(),
        "solved_roots": solved.tolist(),
        "wall_seconds": walls.tolist(),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
    }
    data.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
