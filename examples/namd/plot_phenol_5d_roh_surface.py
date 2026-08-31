#!/usr/bin/env python3
"""Plot the fitted phenol 5D electronic surface along the planar O--H cut."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ml import (
    CorrectedMatrixField,
    MACE,
    RadialMatrixCorrection,
    ReflectionScalarMLP,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = 27.211386245988
ROOT = Path("dataset/phenol_5d_production")
DEFAULT_DATA = ROOT / "inputs/periodic_torsion_inward/phenol_sa6_5d_p_gauge_inward.npz"
DEFAULT_MACE = ROOT / "model/mace_y/phenol_sa6_5d_mace_y.pt"
DEFAULT_MACE_SUMMARY = ROOT / "model/mace_y/summary.json"
DEFAULT_RADIAL_CORRECTION = ROOT / "model/radial_correction/phenol_sa6_5d_radial_delta.npz"
DEFAULT_SCALAR = ROOT / "fits/scalar_parent_periodic_h3_bimodality_corrected/phenol_sa6_5d_scalar_parent.npz"
DEFAULT_QUASIBOUND = ROOT / "states/s1_origin_5d_quasibound_localwell_h3_corrected/summary.json"
DEFAULT_OUTPUT = ROOT / "figures/roh_surface"
COLORS = ("#0072B2", "#D55E00", "#009E73")


def _errors(predicted, exact):
    difference = np.asarray(predicted) - np.asarray(exact)
    return {
        "rms_mev": float(np.sqrt(np.mean(difference**2)) * 1000.0),
        "maximum_mev": float(np.max(np.abs(difference)) * 1000.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mace", type=Path, default=DEFAULT_MACE)
    parser.add_argument("--mace-summary", type=Path, default=DEFAULT_MACE_SUMMARY)
    parser.add_argument("--radial-correction", type=Path, default=DEFAULT_RADIAL_CORRECTION)
    parser.add_argument("--scalar-parent", type=Path, default=DEFAULT_SCALAR)
    parser.add_argument("--quasibound-summary", type=Path, default=DEFAULT_QUASIBOUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=451)
    parser.add_argument("--cap-start", type=float, default=2.45)
    args = parser.parse_args()

    chart = PhenolReactiveChart()
    equilibrium = np.asarray(chart.equilibrium, dtype=float)
    radius = np.linspace(0.75, 3.0, args.points)
    coordinates = np.repeat(equilibrium[None, :], len(radius), axis=0)
    coordinates[:, 0] = radius

    mace = MACE.load(args.mace, chart.geometry, device="cpu", distill=False)
    field = CorrectedMatrixField(
        mace.neural_energy, RadialMatrixCorrection.load(args.radial_correction)
    )
    energy_shift = float(
        json.loads(args.mace_summary.read_text())["model"]["energy_shift_hartree"]
    )
    fitted_hamiltonian = (
        field.predict(coordinates) + energy_shift * np.eye(3)[None, :, :]
    )
    fitted_energies = np.linalg.eigvalsh(fitted_hamiltonian)
    scalar = ReflectionScalarMLP.load(args.scalar_parent)
    fitted_parent = scalar.predict(coordinates)

    with np.load(args.data) as archive:
        data_coordinates = np.asarray(archive["coordinates"], dtype=float)
        data_hamiltonian = np.asarray(archive["p_hamiltonian"], dtype=complex)
        holdout = np.asarray(archive["energy_holdout"], dtype=bool)
    planar = np.max(np.abs(data_coordinates[:, 1:] - equilibrium[1:]), axis=1) < 1.0e-8
    order = np.argsort(data_coordinates[planar, 0])
    exact_radius = data_coordinates[planar, 0][order]
    exact_hamiltonian = data_hamiltonian[planar][order]
    exact_energies = np.linalg.eigvalsh(exact_hamiltonian)
    exact_parent = exact_hamiltonian[:, 1, 1].real
    exact_holdout = holdout[planar][order]
    exact_coordinates = np.repeat(equilibrium[None, :], len(exact_radius), axis=0)
    exact_coordinates[:, 0] = exact_radius
    predicted_exact = np.linalg.eigvalsh(
        field.predict(exact_coordinates) + energy_shift * np.eye(3)[None, :, :]
    )
    predicted_exact_parent = scalar.predict(exact_coordinates)

    equilibrium_index = int(np.argmin(np.abs(exact_radius - equilibrium[0])))
    ground_reference = float(exact_energies[equilibrium_index, 0])
    parent_reference = float(fitted_parent[np.argmin(np.abs(radius - equilibrium[0]))])
    adiabatic_ev = (fitted_energies - ground_reference) * HARTREE_TO_EV
    exact_adiabatic_ev = (exact_energies - ground_reference) * HARTREE_TO_EV
    fitted_gaps_mev = np.diff(fitted_energies, axis=1) * HARTREE_TO_EV * 1000.0
    exact_gaps_mev = np.diff(exact_energies, axis=1) * HARTREE_TO_EV * 1000.0
    parent_ev = (fitted_parent - parent_reference) * HARTREE_TO_EV
    e1_ev = (fitted_energies[:, 1] - parent_reference) * HARTREE_TO_EV
    exact_parent_ev = (exact_parent - parent_reference) * HARTREE_TO_EV
    exact_e1_ev = (exact_energies[:, 1] - parent_reference) * HARTREE_TO_EV

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, panels = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    for state, color in enumerate(COLORS):
        panels[0].plot(radius, adiabatic_ev[:, state], color=color, label=rf"fitted $E_{state}$")
        panels[0].scatter(
            exact_radius,
            exact_adiabatic_ev[:, state],
            s=22,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
    panels[0].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
        ylabel=r"energy relative to $E_0(R_{\rm FC})$ (eV)",
        title="Adiabatic surfaces",
    )
    panels[0].legend(frameon=False, ncol=1, fontsize=8)

    gap_labels = (r"$E_1-E_0$", r"$E_2-E_1$")
    for gap, (label, color) in enumerate(zip(gap_labels, COLORS[1:])):
        panels[1].semilogy(radius, np.maximum(fitted_gaps_mev[:, gap], 1.0e-3), color=color, label=label)
        panels[1].scatter(
            exact_radius,
            exact_gaps_mev[:, gap],
            s=22,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
    panels[1].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
        ylabel="adiabatic gap (meV)",
        title="Electronic gaps",
    )
    panels[1].legend(frameon=False)

    panels[2].plot(radius, e1_ev, color=COLORS[0], label=r"adiabatic $E_1$")
    panels[2].plot(radius, parent_ev, color=COLORS[1], label=r"corrected $H_{11}^{P}$ parent")
    panels[2].scatter(
        exact_radius,
        exact_e1_ev,
        s=22,
        facecolors="white",
        edgecolors=COLORS[0],
        linewidths=1.0,
        zorder=3,
    )
    panels[2].scatter(
        exact_radius,
        exact_parent_ev,
        marker="s",
        s=20,
        facecolors="white",
        edgecolors=COLORS[1],
        linewidths=1.0,
        zorder=3,
    )
    if args.quasibound_summary.is_file():
        quasibound = json.loads(args.quasibound_summary.read_text())
        quasibound_ev = (
            float(quasibound["energy_hartree"]) - parent_reference
        ) * HARTREE_TO_EV
        panels[2].axhline(
            quasibound_ev,
            color=COLORS[2],
            linestyle=":",
            label="5D local-well state",
        )
    panels[2].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
        ylabel=r"energy relative to $H_{11}^{P}(R_{\rm FC})$ (eV)",
        title=r"Adiabatic $E_1$ versus fitted parent",
    )
    panels[2].legend(frameon=False, fontsize=8)

    for label, panel in zip("abc", panels):
        panel.axvline(equilibrium[0], color="0.35", linestyle="--", linewidth=1.0)
        panel.axvspan(args.cap_start, radius[-1], color="0.6", alpha=0.10)
        panel.grid(alpha=0.16, linewidth=0.6)
        panel.text(
            0.02,
            0.96,
            label,
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
    panels[0].text(
        equilibrium[0] + 0.025,
        panels[0].get_ylim()[0] + 0.05 * np.ptp(panels[0].get_ylim()),
        "FC",
        color="0.3",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / "phenol_5d_fitted_roh_surface"
    figure.savefig(base.with_suffix(".png"), dpi=350)
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)
    np.savez_compressed(
        base.with_suffix(".npz"),
        radius=radius,
        coordinates=coordinates,
        fitted_energies=fitted_energies,
        fitted_parent=fitted_parent,
        exact_radius=exact_radius,
        exact_energies=exact_energies,
        exact_parent=exact_parent,
        exact_holdout=exact_holdout,
    )
    summary = {
        "cut": {
            "R_OH_angstrom": [float(radius[0]), float(radius[-1])],
            "fixed_coordinates": equilibrium[1:].tolist(),
            "exact_points": int(len(exact_radius)),
        },
        "mace_planar_backbone_spectral_error": _errors(
            (predicted_exact - exact_energies) * HARTREE_TO_EV,
            np.zeros_like(exact_energies),
        ),
        "scalar_parent_planar_backbone_error": _errors(
            (predicted_exact_parent - exact_parent) * HARTREE_TO_EV,
            np.zeros_like(exact_parent),
        ),
        "outputs": {
            "png": str(base.with_suffix(".png")),
            "pdf": str(base.with_suffix(".pdf")),
            "data": str(base.with_suffix(".npz")),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
