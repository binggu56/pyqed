#!/usr/bin/env python3
"""Audit MACE and FTT errors on the region visited by an H3+/D3+ packet."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyqed.ml import MACE
from pyqed.units import au2ev

from h3plus_fci_mace_dynamics import mace_geometry


def weighted_rms(values, weights):
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * np.asarray(values) ** 2) / np.sum(weights)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dynamics = np.load(args.dynamics)
    fields = np.load(args.fields)
    summary = json.loads(args.summary.read_text())
    qualification = json.loads(args.qualification.read_text())
    axes = tuple(np.asarray(axis) for axis in dynamics["axes"])
    density = np.mean(dynamics["snapshot_densities"], axis=0)
    density /= max(float(np.max(density)), np.finfo(float).tiny)
    interpolator = RegularGridInterpolator(
        axes, density, bounds_error=False, fill_value=0.0
    )

    coordinates = fields["validation_coordinates"]
    support = np.clip(interpolator(coordinates), 0.0, None)
    selected = support >= 1.0e-3
    if np.count_nonzero(selected) < 8:
        selected = support >= np.quantile(support, 0.90)
    weights = np.maximum(support[selected], 1.0e-12)

    fit = MACE.load(args.checkpoint, mace_geometry, distill=False)
    reference_h = fields["validation_hamiltonians"]
    predicted_h = fit.neural_energy.predict(coordinates)
    eigen_error = (
        np.linalg.eigvalsh(predicted_h) - np.linalg.eigvalsh(reference_h)
    ) * au2ev * 1.0e3
    selected_error = eigen_error[selected]
    selected_state_error = np.linalg.norm(selected_error, axis=1)

    pairs = fields["validation_pairs"]
    features = fit.neural_feature.predict(coordinates)
    predicted_links = (
        features[pairs[:, 0]].conj().swapaxes(-1, -2)
        @ features[pairs[:, 1]]
    )
    reference_links = fields["validation_links"]
    link_error = np.linalg.norm(
        predicted_links - reference_links, axis=(-2, -1)
    ) / np.maximum(
        np.linalg.norm(reference_links, axis=(-2, -1)), np.finfo(float).tiny
    )
    pair_support = np.minimum(support[pairs[:, 0]], support[pairs[:, 1]])
    selected_links = pair_support >= 1.0e-3

    fit.grids = axes
    fit.shape = tuple(len(axis) for axis in axes)
    ranks = summary["ranks"]
    fit.distill_y(
        rank=int(ranks["distill"]),
        degree=min(12, len(axes[0]) - 1),
        method="grid",
        validation_points=256,
        seed=31,
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_coordinates = np.stack(mesh, axis=-1).reshape(-1, 3)
    neural_h = fit.neural_energy.predict(grid_coordinates)
    ftt_h = fit.energy.predict(grid_coordinates)
    ftt_error = (
        np.linalg.eigvalsh(ftt_h) - np.linalg.eigvalsh(neural_h)
    ) * au2ev * 1.0e3
    grid_weights = density.ravel()
    occupied = grid_weights >= 1.0e-4

    metrics = {
        "independent_fci_points": int(len(coordinates)),
        "visited_support_fci_points": int(np.count_nonzero(selected)),
        "support_threshold_relative_density": 1.0e-3,
        "visited_support_weighted_rms_eigenvalue_error_mev": weighted_rms(
            selected_state_error, weights
        ),
        "visited_support_maximum_absolute_eigenvalue_error_mev": float(
            np.max(np.abs(selected_error))
        ),
        "visited_support_link_count": int(np.count_nonzero(selected_links)),
        "visited_support_maximum_relative_link_error": (
            float(np.max(link_error[selected_links]))
            if np.any(selected_links) else None
        ),
        "visited_support_relative_link_error": (
            float(np.linalg.norm(
                predicted_links[selected_links] - reference_links[selected_links]
            ) / np.linalg.norm(reference_links[selected_links]))
            if np.any(selected_links) else None
        ),
        "density_weighted_ftt_rms_eigenvalue_error_mev": weighted_rms(
            np.linalg.norm(ftt_error, axis=1), grid_weights
        ),
        "occupied_grid_ftt_maximum_absolute_eigenvalue_error_mev": float(
            np.max(np.abs(ftt_error[occupied]))
        ),
        "global_model_accepted_for_production": bool(
            qualification["accepted_for_production"]
        ),
        "global_validation": qualification["validation"],
        "maximum_symmetry_covariance_error_hartree": qualification[
            "maximum_symmetry_covariance_error_hartree"
        ],
        "interpretation": (
            "dynamic-support audit of independent cached FCI validation points; "
            "no new electronic-structure calculations"
        ),
    }
    report = args.output_dir / "dynamic_support_audit.json"
    report.write_text(json.dumps(metrics, indent=2) + "\n")

    figure, panels = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    panels[0].scatter(
        np.maximum(support, 1.0e-8), np.max(np.abs(eigen_error), axis=1),
        s=14, alpha=0.65,
    )
    panels[0].axvline(1.0e-3, color="0.3", linestyle="--")
    panels[0].set_xscale("log")
    panels[0].set(
        xlabel="time-averaged relative density", ylabel="maximum error / meV",
        title="MACE vs independent FCI",
    )
    panels[1].scatter(
        np.maximum(grid_weights, 1.0e-8), np.max(np.abs(ftt_error), axis=1),
        s=5, alpha=0.35, color="tab:orange",
    )
    panels[1].set_xscale("log")
    panels[1].set(
        xlabel="time-averaged relative density", ylabel="maximum error / meV",
        title="FTT vs neural MACE",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    output = args.output_dir / "dynamic_support_audit.png"
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
