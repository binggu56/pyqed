#!/usr/bin/env python3
"""Plot the gauge-invariant mean SO2 energy along a symmetric bend cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.generate_so2_casci_singlets import geometry
from pyqed.units import au2ev
from pyqed.ml import MACE


HARTREE_TO_EV = au2ev


def mean_energy(values):
    """Return Tr(H) / number of states for matrix or eigenvalue data."""

    values = np.asarray(values)
    if values.ndim >= 2 and values.shape[-2] == values.shape[-1]:
        return np.trace(values, axis1=-2, axis2=-1).real / values.shape[-1]
    return np.mean(values.real, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/private/tmp/so2_casci_c2v_anisotropic_rank45_direct_y_5000.pt"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/private/tmp/so2_casci_singlet_5x5x5.npz"),
    )
    parser.add_argument("--r", type=float, default=2.8)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_c2v_mean_energy_theta_cut.png"),
    )
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as archive:
        if {"r1", "r2", "theta"}.issubset(archive.files):
            r1 = np.asarray(archive["r1"])
            r2 = np.asarray(archive["r2"])
            i = int(np.argmin(np.abs(r1 - args.r)))
            j = int(np.argmin(np.abs(r2 - args.r)))
            if not np.isclose(r1[i], r2[j]):
                raise ValueError("the requested cut is not represented on r1 = r2")
            radius = float(r1[i])
            theta_reference = np.asarray(archive["theta"])
            cut_energies = np.asarray(archive["energies"])[i, j]
        else:
            coordinates = np.asarray(archive["coordinates"])
            symmetric = np.isclose(coordinates[:, 0], coordinates[:, 1])
            radii = coordinates[symmetric, 0]
            radius = float(radii[np.argmin(np.abs(radii - args.r))])
            selected = symmetric & np.isclose(coordinates[:, 0], radius)
            order = np.argsort(coordinates[selected, 2])
            theta_reference = coordinates[selected, 2][order]
            cut_energies = np.asarray(archive["energies"])[selected][order]
    theta_dense = np.linspace(theta_reference[0], theta_reference[-1], args.points)
    coordinates = np.column_stack((
        np.full(args.points, radius),
        np.full(args.points, radius),
        theta_dense,
    ))

    fit = MACE.load(
        args.checkpoint, lambda coordinate: geometry(*coordinate), distill=False
    )
    fitted = mean_energy(fit.neural_energy.predict(coordinates))
    reference = mean_energy(cut_energies)
    center = len(theta_reference) // 2
    center_coordinate = np.asarray([[radius, radius, theta_reference[center]]])
    fitted_center = mean_energy(fit.neural_energy.predict(center_coordinate))[0]
    fitted_relative = (fitted - fitted_center) * HARTREE_TO_EV
    reference_relative = (reference - reference[center]) * HARTREE_TO_EV

    reference_coordinates = np.column_stack((
        np.full(len(theta_reference), radius),
        np.full(len(theta_reference), radius),
        theta_reference,
    ))
    fitted_at_reference = mean_energy(
        fit.neural_energy.predict(reference_coordinates)
    )
    fitted_at_reference = (fitted_at_reference - fitted_center) * HARTREE_TO_EV
    errors_mev = 1.0e3 * (fitted_at_reference - reference_relative)

    theta_dense_deg = np.rad2deg(theta_dense)
    theta_reference_deg = np.rad2deg(theta_reference)
    figure, axes = plt.subplots(
        2, 1, figsize=(6.2, 5.0), sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)}, constrained_layout=True,
    )
    axes[0].plot(theta_dense_deg, fitted_relative, lw=2.0, label="C2v MACE")
    axes[0].scatter(
        theta_reference_deg, reference_relative, s=34, zorder=3,
        facecolor="white", edgecolor="black", label="CASCI",
    )
    axes[0].axhline(0.0, color="0.75", lw=0.8)
    axes[0].set_ylabel(r"$\bar E(\theta)-\bar E(120^\circ)$ (eV)")
    axes[0].legend(frameon=False)
    axes[0].set_title(
        rf"SO$_2$ symmetric bend cut, $r_1=r_2={radius:.2f}$ bohr"
    )
    axes[1].plot(theta_reference_deg, errors_mev, marker="o", lw=1.4)
    axes[1].axhline(0.0, color="0.75", lw=0.8)
    axes[1].set(xlabel=r"$\theta$ (degree)", ylabel="error (meV)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    result = {
        "r_bohr": radius,
        "theta_center_deg": float(theta_reference_deg[center]),
        "reference_mean_energy_center_hartree": float(reference[center]),
        "theta_reference_deg": theta_reference_deg.tolist(),
        "reference_relative_ev": reference_relative.tolist(),
        "fit_at_reference_relative_ev": fitted_at_reference.tolist(),
        "error_mev": errors_mev.tolist(),
        "theta_dense_deg": theta_dense_deg.tolist(),
        "fit_dense_relative_ev": fitted_relative.tolist(),
        "rms_error_mev": float(np.sqrt(np.mean(errors_mev**2))),
        "max_abs_error_mev": float(np.max(np.abs(errors_mev))),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "r_bohr": result["r_bohr"],
        "reference_mean_energy_center_hartree": result[
            "reference_mean_energy_center_hartree"
        ],
        "rms_error_mev": result["rms_error_mev"],
        "max_abs_error_mev": result["max_abs_error_mev"],
        "figure": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
