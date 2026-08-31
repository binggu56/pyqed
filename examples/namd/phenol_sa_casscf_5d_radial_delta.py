#!/usr/bin/env python3
"""Fit a radial matrix residual that extends the phenol 5D MACE-Y field."""

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

from pyqed.units import au2ev
from pyqed.ml import CorrectedMatrixField, MACE, RadialMatrixCorrection
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
COLORS = ("#0072B2", "#D55E00", "#009E73")


def metrics(predicted, target, mask):
    error = np.abs(
        np.linalg.eigvalsh(predicted[mask]) - np.linalg.eigvalsh(target[mask])
    ) * HARTREE_TO_EV
    return {
        "count": int(np.count_nonzero(mask)),
        "rms_ev": float(np.sqrt(np.mean(error**2))),
        "maximum_ev": float(np.max(error)),
        "mae_ev": float(np.mean(error)),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    chart = PhenolReactiveChart(modes=data["modes"])
    fit = MACE.load(args.checkpoint, chart.geometry, device="cpu", distill=False)
    coordinates = data["coordinates"]
    shift = float(np.min(np.linalg.eigvalsh(data["p_hamiltonian"][0])))
    target = data["p_hamiltonian"] - shift * np.eye(3)
    baseline = fit.neural_energy.predict(coordinates)
    correction = RadialMatrixCorrection.fit(
        coordinates,
        target,
        baseline,
        coordinate=0,
        representation=data["reflection"],
    )
    corrected = CorrectedMatrixField(fit.neural_energy, correction)
    predicted = corrected.predict(coordinates)
    features = fit.neural_feature.predict(coordinates)
    predicted_links = (
        features[data["pairs"][:, 0]].conj().swapaxes(-1, -2)
        @ features[data["pairs"][:, 1]]
    )
    link_error = np.linalg.norm(
        predicted_links - data["p_links"], axis=(1, 2)
    ) / np.linalg.norm(data["p_links"], axis=(1, 2))
    inward = data["source_is_inward_backbone"]
    new_links = (
        inward[data["pairs"][:, 0]] | inward[data["pairs"][:, 1]]
    )
    validation = {
        "original_points": metrics(predicted, target, ~inward),
        "inward_points": metrics(predicted, target, inward),
        "all_points": metrics(predicted, target, np.ones(len(coordinates), dtype=bool)),
        "original_link_relative_rms": float(np.sqrt(np.mean(link_error[~new_links] ** 2))),
        "original_link_relative_maximum": float(np.max(link_error[~new_links])),
        "inward_link_relative_rms": float(np.sqrt(np.mean(link_error[new_links] ** 2))),
        "inward_link_relative_maximum": float(np.max(link_error[new_links])),
    }
    reflection = np.asarray(data["reflection"])
    probes = np.column_stack(
        (
            np.linspace(correction.knots[0], correction.knots[-1], 251),
            np.zeros((251, 4)),
        )
    )
    correction_values = correction.predict(probes)
    reflection_defect = float(
        np.max(
            np.linalg.norm(
                correction_values
                - reflection.conj().T @ correction_values @ reflection,
                axis=(1, 2),
            )
        )
    )
    hermiticity_defect = float(
        np.max(
            np.linalg.norm(
                correction_values - correction_values.conj().swapaxes(-1, -2),
                axis=(1, 2),
            )
        )
    )
    gates = {
        "inward_spectral_rms_below_0p005_ev": validation["inward_points"]["rms_ev"] < 0.005,
        "inward_spectral_max_below_0p01_ev": validation["inward_points"]["maximum_ev"] < 0.01,
        "all_spectral_rms_below_0p05_ev": validation["all_points"]["rms_ev"] < 0.05,
        "inward_link_relative_max_below_0p01": validation["inward_link_relative_maximum"] < 0.01,
        "exact_hermiticity": hermiticity_defect < 1.0e-12,
        "exact_reflection_covariance": reflection_defect < 1.0e-12,
    }
    correction_path = correction.save(args.output / "phenol_sa6_5d_radial_delta.npz")

    radial_order = np.argsort(coordinates[:, 0])
    base_error = np.abs(
        np.linalg.eigvalsh(baseline) - np.linalg.eigvalsh(target)
    ) * HARTREE_TO_EV
    corrected_error = np.abs(
        np.linalg.eigvalsh(predicted) - np.linalg.eigvalsh(target)
    ) * HARTREE_TO_EV
    figure, panels = plt.subplots(2, 2, figsize=(9.7, 7.0), constrained_layout=True)
    planar = np.all(
        np.isclose(
            coordinates[:, 1:],
            (
                0.0,
                min(
                    np.unique(coordinates[:, 2]),
                    key=lambda value: abs(value - np.deg2rad(108.8)),
                ),
                0.0,
                0.0,
            ),
            atol=1.0e-8,
        ),
        axis=1,
    )
    planar_order = np.argsort(coordinates[planar, 0])
    radial = coordinates[planar, 0][planar_order]
    reference_energy = np.linalg.eigvalsh(target[planar][planar_order])
    base_energy = np.linalg.eigvalsh(baseline[planar][planar_order])
    corrected_energy = np.linalg.eigvalsh(predicted[planar][planar_order])
    zero = float(np.min(reference_energy))
    for state, color in enumerate(COLORS):
        panels[0, 0].plot(
            radial, (reference_energy[:, state] - zero) * HARTREE_TO_EV,
            color=color, lw=1.5,
        )
        panels[0, 0].plot(
            radial, (base_energy[:, state] - zero) * HARTREE_TO_EV,
            color=color, ls=":", lw=1.0,
        )
        panels[0, 0].plot(
            radial, (corrected_energy[:, state] - zero) * HARTREE_TO_EV,
            color=color, ls="--", lw=1.0,
        )
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="relative energy (eV)",
        title="Planar backbone: solid reference, dotted MACE, dashed corrected",
    )
    panels[0, 1].semilogy(
        coordinates[radial_order, 0],
        np.max(base_error, axis=1)[radial_order], ".", color="0.55", label="MACE",
    )
    panels[0, 1].semilogy(
        coordinates[radial_order, 0],
        np.maximum(np.max(corrected_error, axis=1)[radial_order], 1.0e-8),
        ".", color=COLORS[0], label=r"MACE+$\Delta_R$",
    )
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="maximum spectral error (eV)",
        title=f"All {len(coordinates)} electronic points",
    )
    panels[0, 1].legend(frameon=False)
    panels[1, 0].plot(
        correction.knots,
        np.linalg.norm(correction.values, axis=(1, 2)) * HARTREE_TO_EV,
        "o-", color=COLORS[1],
    )
    panels[1, 0].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel=r"$\|\Delta_R\|_F$ (eV)",
        title="Learned radial matrix baseline",
    )
    panels[1, 1].plot(
        np.arange(np.count_nonzero(new_links)), link_error[new_links],
        "o-", color=COLORS[2],
    )
    panels[1, 1].set(
        xlabel="inward radial edge", ylabel="relative link error",
        title="Unchanged MACE-Y endpoint frames",
    )
    for panel in panels.flat:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=0.16)
    figure_path = args.output / "phenol_sa6_5d_radial_delta.png"
    figure.savefig(figure_path, dpi=300)
    figure.savefig(args.output / "phenol_sa6_5d_radial_delta.pdf")
    plt.close(figure)

    summary = {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "model": "unchanged 5D MACE-Y plus reflection-symmetric radial matrix delta",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "correction": str(correction_path),
        "figure": str(figure_path),
        "knots_angstrom": correction.knots.tolist(),
        "validation": validation,
        "maximum_hermiticity_defect": hermiticity_defect,
        "maximum_reflection_covariance_defect": reflection_defect,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("radial delta qualification failed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_5d_inward_20260822/"
            "phenol_sa6_5d_p_gauge_inward.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_5d_mace_y_production_final_20260822/"
            "phenol_sa6_5d_mace_y.pt"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_5d_radial_delta_20260822"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
