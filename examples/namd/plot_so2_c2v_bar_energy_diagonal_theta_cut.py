#!/usr/bin/env python3
"""Plot the diagonal P-gauge SO2 Hamiltonian along a symmetric bend cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.namd.generate_so2_casci_singlets import (
    geometry,
    so2_point_group_representations,
)
from pyqed.units import au2ev
from pyqed.ml import MACE


HARTREE_TO_EV = au2ev
IRREP_BY_CHARACTERS = {
    (1, 1, 1, 1): "A1",
    (1, 1, -1, -1): "A2",
    (1, -1, 1, -1): "B1",
    (1, -1, -1, 1): "B2",
}


def point_group_characters(theta, radius, options, cache):
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            if np.allclose(archive["theta"], theta):
                return np.asarray(archive["characters"], dtype=int)
    characters = []
    for index, angle in enumerate(theta, start=1):
        _names, representations, _raw, _diagnostics = (
            so2_point_group_representations(radius, angle, options)
        )
        characters.append(np.where(
            np.real(np.diagonal(representations, axis1=1, axis2=2)) >= 0.0,
            1,
            -1,
        ))
        print(f"[C2v] {index}/{len(theta)}", flush=True)
    characters = np.asarray(characters, dtype=int)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, theta=theta, characters=characters)
    return characters


def reorder_by_irrep(energies, characters, target_characters):
    reordered = np.full_like(energies, np.nan, dtype=float)
    present = np.zeros((len(energies), len(target_characters)), dtype=bool)
    for point, point_characters in enumerate(characters):
        signatures = point_characters.T
        for state, target in enumerate(target_characters):
            matches = np.flatnonzero(np.all(signatures == target, axis=1))
            if len(matches) == 1:
                reordered[point, state] = energies[point, matches[0]]
                present[point, state] = True
    return reordered, present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/private/tmp/"
            "so2_casci_c2v_anisotropic_rank45_direct_y_5000_theta80_160.pt"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/private/tmp/so2_casci_c2v_theta_cut_80_160_17.npz"),
    )
    parser.add_argument(
        "--character-cache",
        type=Path,
        default=Path(
            "/private/tmp/so2_casci_c2v_theta_cut_80_160_characters.npz"
        ),
    )
    parser.add_argument("--r", type=float, default=2.8)
    parser.add_argument("--points", type=int, default=321)
    parser.add_argument("--training-step-deg", type=float, default=10.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=32)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_c2v_bar_energy_diagonal_theta_cut.png"),
    )
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as archive:
        theta_reference = np.asarray(archive["theta"])
        reference_energies = np.asarray(archive["energies"])[0, 0]
    options = SimpleNamespace(**vars(args))
    characters = point_group_characters(
        theta_reference, args.r, options, args.character_cache
    )

    fit = MACE.load(
        args.checkpoint, lambda coordinate: geometry(*coordinate), distill=False
    )
    symmetry = fit.coordinate_exchange_
    c2 = np.real(np.diag(symmetry["electronic_representation"]))
    sigma = np.real(np.diag(
        symmetry["fixed_electronic_representations"][0]
    ))
    target_characters = np.column_stack((
        np.ones(args.nstates, dtype=int),
        np.where(c2 >= 0.0, 1, -1),
        np.where(sigma >= 0.0, 1, -1),
        np.where(c2 * sigma >= 0.0, 1, -1),
    ))
    irrep_labels = [
        IRREP_BY_CHARACTERS.get(tuple(value), str(tuple(value)))
        for value in target_characters
    ]
    reference_bar, present = reorder_by_irrep(
        reference_energies, characters, target_characters
    )

    theta_dense = np.linspace(theta_reference[0], theta_reference[-1], args.points)
    dense_coordinates = np.column_stack((
        np.full(args.points, args.r),
        np.full(args.points, args.r),
        theta_dense,
    ))
    reference_coordinates = np.column_stack((
        np.full(len(theta_reference), args.r),
        np.full(len(theta_reference), args.r),
        theta_reference,
    ))
    predicted_dense_matrix = fit.neural_energy.predict(dense_coordinates)
    predicted_reference_matrix = fit.neural_energy.predict(reference_coordinates)
    predicted_dense = np.real(np.diagonal(
        predicted_dense_matrix, axis1=1, axis2=2
    ))
    predicted_reference = np.real(np.diagonal(
        predicted_reference_matrix, axis1=1, axis2=2
    ))
    off_diagonal_max = float(np.max(np.abs(
        predicted_dense_matrix
        - np.asarray([np.diag(np.diag(value)) for value in predicted_dense_matrix])
    )))

    center = int(np.argmin(np.abs(theta_reference - np.deg2rad(120.0))))
    reference_zero = reference_bar[center, 0]
    predicted_zero = predicted_reference[center, 0]
    reference_relative = (reference_bar - reference_zero) * HARTREE_TO_EV
    predicted_dense_relative = (
        predicted_dense - predicted_zero
    ) * HARTREE_TO_EV
    predicted_reference_relative = (
        predicted_reference - predicted_zero
    ) * HARTREE_TO_EV
    errors_mev = 1.0e3 * (
        predicted_reference_relative - reference_relative
    )

    theta_reference_deg = np.rad2deg(theta_reference)
    theta_dense_deg = np.rad2deg(theta_dense)
    scaled = (theta_reference_deg - theta_reference_deg[0]) / args.training_step_deg
    training = np.isclose(scaled, np.rint(scaled), atol=1.0e-8)
    holdout = ~training

    def masked_rms(mask):
        return [
            float(np.sqrt(np.nanmean(errors_mev[mask, state] ** 2)))
            for state in range(args.nstates)
        ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: args.nstates]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.2)}, constrained_layout=True,
    )
    for state, (color, irrep) in enumerate(zip(colors, irrep_labels)):
        axes[0].plot(
            theta_dense_deg, predicted_dense_relative[:, state], color=color,
            lw=2.0, label=rf"$\bar E_{{{state + 1}{state + 1}}}$ ({irrep})",
        )
        available_training = training & present[:, state]
        available_holdout = holdout & present[:, state]
        axes[0].scatter(
            theta_reference_deg[available_training],
            reference_relative[available_training, state],
            s=30, facecolor="white", edgecolor=color, linewidth=1.5, zorder=3,
        )
        axes[0].scatter(
            theta_reference_deg[available_holdout],
            reference_relative[available_holdout, state],
            s=34, marker="D", facecolor="white", edgecolor=color,
            linewidth=1.5, zorder=3,
        )
        axes[1].plot(
            theta_reference_deg[present[:, state]],
            errors_mev[present[:, state], state],
            color=color, marker="o", ms=3.8, lw=1.4,
        )
    axes[0].set_ylabel(r"$\bar E_{ii}(\theta)-\bar E_{11}(120^\circ)$ (eV)")
    state_handles = [
        Line2D([], [], color=color, lw=2,
               label=rf"$\bar E_{{{state + 1}{state + 1}}}$ ({irrep})")
        for state, (color, irrep) in enumerate(zip(colors, irrep_labels))
    ]
    sample_handles = [
        Line2D([], [], marker="o", markerfacecolor="white", color="0.35",
               linestyle="none", label="CASCI training"),
        Line2D([], [], marker="D", markerfacecolor="white", color="0.35",
               linestyle="none", label="CASCI interleaved validation"),
    ]
    axes[0].legend(handles=state_handles + sample_handles, frameon=False, ncol=2)
    axes[0].set_title(
        rf"SO$_2$ P-gauge diagonal, $r_1=r_2={args.r:.2f}$ bohr"
    )
    axes[1].axhline(0.0, color="0.7", lw=0.8)
    axes[1].set(xlabel=r"$\theta$ (degree)", ylabel="MACE - CASCI (meV)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    result = {
        "irrep_labels": irrep_labels,
        "target_characters": target_characters.tolist(),
        "characters_by_angle": characters.tolist(),
        "irrep_present": present.tolist(),
        "theta_reference_deg": theta_reference_deg.tolist(),
        "training_mask": training.tolist(),
        "theta_dense_deg": theta_dense_deg.tolist(),
        "reference_relative_ev": reference_relative.tolist(),
        "fit_reference_relative_ev": predicted_reference_relative.tolist(),
        "fit_dense_relative_ev": predicted_dense_relative.tolist(),
        "error_mev": errors_mev.tolist(),
        "training_rms_error_mev": masked_rms(training),
        "holdout_rms_error_mev": masked_rms(holdout),
        "maximum_off_diagonal_hartree": off_diagonal_max,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "irrep_labels": irrep_labels,
        "irrep_counts": np.sum(present, axis=0).tolist(),
        "training_rms_error_mev": result["training_rms_error_mev"],
        "holdout_rms_error_mev": result["holdout_rms_error_mev"],
        "maximum_off_diagonal_hartree": off_diagonal_max,
        "figure": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
