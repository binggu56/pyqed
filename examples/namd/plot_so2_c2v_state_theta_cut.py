#!/usr/bin/env python3
"""Plot all adiabatic SO2 states along a symmetric bend cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.namd.generate_so2_casci_singlets import geometry
from pyqed.units import au2ev
from pyqed.ml import MACE


HARTREE_TO_EV = au2ev


def reference_cut(path, radius):
    with np.load(path, allow_pickle=False) as archive:
        if {"r1", "r2", "theta"}.issubset(archive.files):
            r1 = np.asarray(archive["r1"])
            r2 = np.asarray(archive["r2"])
            i = int(np.argmin(np.abs(r1 - radius)))
            j = int(np.argmin(np.abs(r2 - radius)))
            return (
                float(r1[i]),
                np.asarray(archive["theta"]),
                np.asarray(archive["energies"])[i, j],
            )
        coordinates = np.asarray(archive["coordinates"])
        symmetric = np.isclose(coordinates[:, 0], coordinates[:, 1])
        radii = coordinates[symmetric, 0]
        selected_radius = float(radii[np.argmin(np.abs(radii - radius))])
        selected = symmetric & np.isclose(coordinates[:, 0], selected_radius)
        order = np.argsort(coordinates[selected, 2])
        return (
            selected_radius,
            coordinates[selected, 2][order],
            np.asarray(archive["energies"])[selected][order],
        )


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
    parser.add_argument("--r", type=float, default=2.8)
    parser.add_argument("--points", type=int, default=321)
    parser.add_argument("--training-step-deg", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_c2v_all_states_theta_cut_80_160.png"),
    )
    args = parser.parse_args()

    radius, theta_reference, reference = reference_cut(args.reference, args.r)
    theta_dense = np.linspace(theta_reference[0], theta_reference[-1], args.points)
    coordinates = np.column_stack((
        np.full(args.points, radius),
        np.full(args.points, radius),
        theta_dense,
    ))
    reference_coordinates = np.column_stack((
        np.full(len(theta_reference), radius),
        np.full(len(theta_reference), radius),
        theta_reference,
    ))
    fit = MACE.load(
        args.checkpoint, lambda coordinate: geometry(*coordinate), distill=False
    )
    dense = np.linalg.eigvalsh(fit.neural_energy.predict(coordinates)).real
    fitted_reference = np.linalg.eigvalsh(
        fit.neural_energy.predict(reference_coordinates)
    ).real
    center = int(np.argmin(np.abs(theta_reference - np.deg2rad(120.0))))
    reference_relative = (reference - reference[center, 0]) * HARTREE_TO_EV
    fitted_relative = (dense - fitted_reference[center, 0]) * HARTREE_TO_EV
    fitted_reference_relative = (
        fitted_reference - fitted_reference[center, 0]
    ) * HARTREE_TO_EV
    errors_mev = 1.0e3 * (fitted_reference_relative - reference_relative)

    theta_reference_deg = np.rad2deg(theta_reference)
    theta_dense_deg = np.rad2deg(theta_dense)
    scaled = (theta_reference_deg - theta_reference_deg[0]) / args.training_step_deg
    training = np.isclose(scaled, np.rint(scaled), atol=1.0e-8)
    holdout = ~training
    rms_training = np.sqrt(np.mean(errors_mev[training] ** 2, axis=0))
    rms_holdout = np.sqrt(np.mean(errors_mev[holdout] ** 2, axis=0))
    max_holdout = np.max(np.abs(errors_mev[holdout]), axis=0)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: reference.shape[1]]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.2)}, constrained_layout=True,
    )
    for state, color in enumerate(colors):
        axes[0].plot(
            theta_dense_deg, fitted_relative[:, state], color=color, lw=2.0,
            label=rf"MACE $E_{state + 1}$",
        )
        axes[0].scatter(
            theta_reference_deg[training], reference_relative[training, state],
            s=30, facecolor="white", edgecolor=color, linewidth=1.5, zorder=3,
        )
        axes[0].scatter(
            theta_reference_deg[holdout], reference_relative[holdout, state],
            s=34, marker="D", facecolor="white", edgecolor=color,
            linewidth=1.5, zorder=3,
        )
        axes[1].plot(
            theta_reference_deg, errors_mev[:, state], color=color,
            marker="o", ms=3.8, lw=1.4, label=rf"$E_{state + 1}$",
        )
    axes[0].set_ylabel(r"$E_i(\theta)-E_1(120^\circ)$ (eV)")
    state_handles = [
        Line2D([], [], color=color, lw=2, label=rf"$E_{state + 1}$")
        for state, color in enumerate(colors)
    ]
    sample_handles = [
        Line2D([], [], marker="o", markerfacecolor="white", color="0.35",
               linestyle="none", label="CASCI training"),
        Line2D([], [], marker="D", markerfacecolor="white", color="0.35",
               linestyle="none", label="CASCI interleaved validation"),
    ]
    axes[0].legend(handles=state_handles + sample_handles, frameon=False, ncol=2)
    axes[0].set_title(
        rf"SO$_2$ adiabatic states, $r_1=r_2={radius:.2f}$ bohr"
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
        "r_bohr": radius,
        "theta_reference_deg": theta_reference_deg.tolist(),
        "training_mask": training.tolist(),
        "theta_dense_deg": theta_dense_deg.tolist(),
        "reference_relative_ev": reference_relative.tolist(),
        "fit_reference_relative_ev": fitted_reference_relative.tolist(),
        "fit_dense_relative_ev": fitted_relative.tolist(),
        "error_mev": errors_mev.tolist(),
        "training_rms_error_mev": rms_training.tolist(),
        "holdout_rms_error_mev": rms_holdout.tolist(),
        "holdout_max_abs_error_mev": max_holdout.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "training_rms_error_mev": result["training_rms_error_mev"],
        "holdout_rms_error_mev": result["holdout_rms_error_mev"],
        "holdout_max_abs_error_mev": result["holdout_max_abs_error_mev"],
        "figure": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
