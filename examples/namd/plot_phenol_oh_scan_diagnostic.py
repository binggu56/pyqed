#!/usr/bin/env python3
"""Plot the phenol O--H SA-CASSCF backend and optimizer diagnostics."""

from __future__ import annotations
from pyqed.units import au2ev

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

HARTREE_TO_EV = au2ev
HARTREE_TO_MEV = 1000.0 * HARTREE_TO_EV
ANCHOR_DISTANCE = 0.96994


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def record_path(root, backend, distance):
    if np.isclose(distance, ANCHOR_DISTANCE, atol=1.0e-8):
        return root / backend / "anchor.npz"
    direction = "increasing" if distance > ANCHOR_DISTANCE else "decreasing"
    return root / backend / direction / f"r{distance:.5f}.npz"


def macroiteration_count(record):
    cycles = np.asarray(record["macro_history"])[:, 0]
    return int(1 + np.count_nonzero(cycles[1:] != cycles[:-1])) if cycles.size else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("reference_seeded", type=Path)
    parser.add_argument("output", type=Path, help="Output path without a suffix")
    args = parser.parse_args()

    summary = json.loads((args.scan / "summary.json").read_text())
    distances = np.asarray(summary["distances_angstrom"], dtype=float)
    pyscf = [load_record(record_path(args.scan, "pyscf", r)) for r in distances]
    sequential = [load_record(record_path(args.scan, "pyqed", r)) for r in distances]
    seeded = load_record(args.reference_seeded)
    seeded_distance = float(seeded["distance"])
    seeded_index = int(np.argmin(np.abs(distances - seeded_distance)))
    if not np.isclose(distances[seeded_index], seeded_distance, atol=1.0e-8):
        raise ValueError("reference-seeded geometry is absent from the scan")

    matched = list(sequential)
    matched[seeded_index] = seeded
    pyscf_energies = np.asarray([record["energies"] for record in pyscf])
    sequential_energies = np.asarray([record["energies"] for record in sequential])
    matched_energies = np.asarray([record["energies"] for record in matched])
    reference_energy = float(pyscf_energies[np.argmin(np.abs(distances - ANCHOR_DISTANCE)), 0])
    pyscf_relative = (pyscf_energies - reference_energy) * HARTREE_TO_EV
    matched_relative = (matched_energies - reference_energy) * HARTREE_TO_EV
    sequential_relative = (sequential_energies - reference_energy) * HARTREE_TO_EV

    matched_error = np.abs(matched_energies - pyscf_energies) * HARTREE_TO_MEV
    sequential_error = np.abs(sequential_energies - pyscf_energies) * HARTREE_TO_MEV
    matched_max = np.max(matched_error, axis=1)
    sequential_max = np.max(sequential_error, axis=1)
    pyqed_wall = np.asarray([float(record["wall_seconds"]) for record in sequential])
    pyscf_wall = np.asarray([float(record["wall_seconds"]) for record in pyscf])
    wall_ratio = pyqed_wall / pyscf_wall
    pyqed_cycles = np.asarray([macroiteration_count(record) for record in sequential])
    pyscf_cycles = np.asarray([macroiteration_count(record) for record in pyscf])

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, pyscf_energies.shape[1]))
    figure, panels = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.35),
        gridspec_kw={"width_ratios": (1.72, 1.0, 1.0)},
        constrained_layout=True,
    )

    for state, color in enumerate(colors):
        panels[0].plot(
            distances,
            pyscf_relative[:, state],
            "-o",
            color=color,
            lw=1.35,
            ms=3.8,
        )
        panels[0].plot(
            distances,
            matched_relative[:, state],
            linestyle="none",
            marker="s",
            ms=4.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.05,
        )
        panels[0].plot(
            distances[seeded_index],
            sequential_relative[seeded_index, state],
            linestyle="none",
            marker="x",
            color="#D55E00",
            ms=5.0,
            mew=1.15,
        )
        panels[0].annotate(
            f"S{state}",
            (distances[-1], pyscf_relative[-1, state]),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=8.2,
            va="center",
        )
    panels[0].set_xlim(distances[0] - 0.04, distances[-1] + 0.16)
    panels[0].set_ylim(-0.4, 8.75)
    panels[0].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel=r"$E_i(R)-E_{S_0}(R_{\rm eq})$ (eV)",
        title="a  Spin-pure SA(6)-CASSCF surfaces",
    )
    panels[0].legend(
        handles=[
            Line2D([], [], color="0.25", marker="o", label="PySCF"),
            Line2D(
                [], [], color="0.25", linestyle="none", marker="s",
                markerfacecolor="white", label="PyQED matched",
            ),
            Line2D(
                [], [], color="#D55E00", linestyle="none", marker="x",
                label="alternate basin",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
    )

    panels[1].semilogy(
        distances,
        sequential_max,
        "--^",
        color="#D55E00",
        lw=1.1,
        ms=4.2,
        label="sequential branch",
    )
    panels[1].semilogy(
        distances,
        matched_max,
        "-o",
        color="#0072B2",
        lw=1.35,
        ms=4.2,
        label="matched branch",
    )
    panels[1].axhline(1.0, color="0.45", ls=":", lw=1.0, label="1 meV")
    panels[1].set_ylim(0.02, max(500.0, 1.35 * float(np.max(sequential_max))))
    panels[1].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="maximum state error (meV)",
        title="b  Backend agreement",
    )
    panels[1].legend(loc="upper left")

    panels[2].plot(
        distances,
        pyscf_cycles,
        "-o",
        color="0.35",
        lw=1.2,
        ms=4.0,
        label="PySCF",
    )
    panels[2].plot(
        distances,
        pyqed_cycles,
        "-o",
        color="#6A3D9A",
        lw=1.35,
        ms=4.2,
        label="PyQED",
    )
    panels[2].set_ylim(0, 1.17 * max(float(np.max(pyqed_cycles)), float(np.max(pyscf_cycles))))
    panels[2].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="macroiterations including restarts",
        title="c  Orbital-optimizer effort",
    )
    panels[2].legend(loc="upper right")
    for index in np.argsort(pyqed_cycles)[-2:]:
        panels[2].annotate(
            str(pyqed_cycles[index]),
            (distances[index], pyqed_cycles[index]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
        )

    for panel in panels:
        panel.grid(axis="y", color="0.90", lw=0.65)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    data_path = args.output.with_suffix(".json")
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    output = {
        "method": summary["method"],
        "distances_angstrom": distances.tolist(),
        "matched_max_state_error_mev": matched_max.tolist(),
        "sequential_max_state_error_mev": sequential_max.tolist(),
        "matched_global_max_state_error_mev": float(np.max(matched_max)),
        "alternate_basin_distance_angstrom": seeded_distance,
        "alternate_basin_max_state_error_mev": float(sequential_max[seeded_index]),
        "reference_seeded_max_state_error_mev": float(matched_max[seeded_index]),
        "sequential_pyqed_wall_seconds": pyqed_wall.tolist(),
        "pyscf_wall_seconds": pyscf_wall.tolist(),
        "sequential_wall_time_ratio": wall_ratio.tolist(),
        "sequential_pyqed_macroiterations": pyqed_cycles.tolist(),
        "pyscf_macroiterations": pyscf_cycles.tolist(),
        "reference_seeded_wall_seconds": float(seeded["wall_seconds"]),
        "reference_seeded_spins": np.asarray(seeded["spins"]).tolist(),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
    }
    data_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
