#!/usr/bin/env python3
"""Plot phenol SA-CASSCF convergence before and after plateau recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pyqed.units import au2mev

HARTREE_TO_MEV = au2mev


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def convergence_trace(record):
    history = np.asarray(record["macro_history"], dtype=float)
    macro = np.arange(1, len(history) + 1)
    residual = np.maximum(
        np.abs(history[:, 1] - history[-1, 1]) * HARTREE_TO_MEV,
        1.0e-7,
    )
    return macro, residual, history[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-110", type=Path, required=True)
    parser.add_argument("--new-155", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path without suffix")
    args = parser.parse_args()

    distances = (1.10, 1.55)
    new_paths = (args.new_110, args.new_155)
    old = [
        load(args.old_root / "pyqed" / "increasing" / f"r{distance:.5f}.npz")
        for distance in distances
    ]
    new = [load(path) for path in new_paths]
    colors = {"old": "#6C6C6C", "new": "#0072B2"}

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, panels = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)

    for column, (distance, old_record, new_record) in enumerate(zip(distances, old, new)):
        for label, record, linestyle, marker in (
            ("Before", old_record, "--", None),
            ("After", new_record, "-", "o"),
        ):
            macro, residual, gradient = convergence_trace(record)
            color = colors["old" if label == "Before" else "new"]
            panels[0, column].semilogy(
                macro,
                residual,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(macro) // 10),
                ms=3.2,
                lw=1.35,
                color=color,
            )
            panels[1, column].semilogy(
                macro,
                np.maximum(np.abs(gradient), 1.0e-12),
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(macro) // 10),
                ms=3.2,
                lw=1.35,
                color=color,
            )

        old_n = len(old_record["macro_history"])
        new_n = len(new_record["macro_history"])
        panels[0, column].set_title(
            rf"{'ab'[column]}  $R_{{\rm OH}}={distance:.2f}$ Å: "
            f"{old_n} → {new_n} macros"
        )
        panels[0, column].set_xlabel("macroiteration")
        panels[0, column].set_ylabel(
            r"$|E_{\rm SA}-E_{\rm final}|$ (meV)" if column == 0 else ""
        )
        panels[1, column].axhline(1.0e-5, color="#D55E00", ls=":", lw=1.1)
        panels[1, column].set_xlabel("macroiteration")
        panels[1, column].set_ylabel(
            r"orbital-gradient norm" if column == 0 else ""
        )
        panels[1, column].set_title(
            f"{'cd'[column]}  wall time: "
            f"{float(old_record['wall_seconds']):.0f} → "
            f"{float(new_record['wall_seconds']):.0f} s"
        )

    for panel in panels.flat:
        panel.grid(axis="y", color="0.90", lw=0.65)

    figure.legend(
        handles=[
            Line2D([], [], color=colors["old"], ls="--", lw=1.4, label="before"),
            Line2D([], [], color=colors["new"], marker="o", ms=3.5, lw=1.4, label="after"),
            Line2D([], [], color="#D55E00", ls=":", lw=1.1, label=r"$10^{-5}$ threshold"),
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    data = args.output.with_suffix(".json")
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "method": "SA(6)-CASSCF(10e,10o)/6-31+G* with fix_spin(ss=0, shift=1.0)",
        "distances_angstrom": list(distances),
        "macroiterations_before": [len(record["macro_history"]) for record in old],
        "macroiterations_after": [len(record["macro_history"]) for record in new],
        "wall_seconds_before": [float(record["wall_seconds"]) for record in old],
        "wall_seconds_after": [float(record["wall_seconds"]) for record in new],
        "zero_step_recoveries": [int(record["zero_step_recoveries"]) for record in new],
        "external_restarts": [int(record["external_restarts"]) for record in new],
        "final_gradient_norm": [float(record["orbital_gradient"]) for record in new],
        "figure_png": str(png),
        "figure_pdf": str(pdf),
    }
    data.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
