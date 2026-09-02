#!/usr/bin/env python3
"""Plot cached ab-initio and MACE cuts of the gauged H3+ Hamiltonian."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pyqed.ml import MACE
from pyqed.units import au2ev


default_output = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_output
        / "h3plus_fci_physical_fitted_pes_cuts_diagnostic_rejected_"
        "curvilinear_expanded_abinitio.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_output
        / "physical_s3_quotient_mace_y_curvilinear_expanded_abinitio.pt",
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path("/private/tmp/h3plus_fci_augccpvdz_physical")
        / "h3plus_fci_initial_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output / "h3plus_gauged_hamiltonian_cuts",
    )
    return parser.parse_args()


def geometry_map(equilibrium, distortion_limit=0.65):
    root3 = np.sqrt(3.0)
    triangle = np.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )

    def geometry(q):
        qs, qx, qy = np.asarray(q, dtype=float)
        radius = np.sqrt(qx**2 + qy**2 + 1.0e-16)
        amplitude = distortion_limit * np.tanh(radius / distortion_limit)
        traceless = np.asarray(((qx, qy), (qy, -qx)))
        strain = (
            np.cosh(amplitude / equilibrium) * np.eye(2)
            + np.sinh(amplitude / equilibrium) / radius * traceless
        )
        transform = np.exp(qs / equilibrium) * strain
        xyz = triangle.copy()
        xyz[:, :2] = equilibrium * triangle[:, :2] @ transform
        return xyz

    return geometry


def components(hamiltonians):
    """Return coefficients of H = h0 I + hx sigma_x + hy sigma_y + hz sigma_z."""
    hamiltonians = np.asarray(hamiltonians)
    return {
        "h0": 0.5 * np.real(hamiltonians[:, 0, 0] + hamiltonians[:, 1, 1]),
        "hx": 0.5 * np.real(hamiltonians[:, 0, 1] + hamiltonians[:, 1, 0]),
        "hy": 0.5 * np.imag(hamiltonians[:, 1, 0] - hamiltonians[:, 0, 1]),
        "hz": 0.5 * np.real(hamiltonians[:, 0, 0] - hamiltonians[:, 1, 1]),
    }


def main():
    args = parse_args()
    preparation = json.loads(args.preparation.read_text())
    geometry = geometry_map(float(preparation["equilibrium_bond_bohr"]))
    model = MACE.load(args.checkpoint, geometry, distill=False)
    cached = np.load(args.data)

    angle15 = np.deg2rad(15.0)
    mixed = np.asarray((1.0, 0.8 * np.cos(angle15), 0.8 * np.sin(angle15)))
    mixed /= np.linalg.norm(mixed)
    cuts = (
        ("breathing", r"breathing: $Q_s$", np.asarray((1.0, 0.0, 0.0))),
        ("branching", r"branching: $Q_x$", np.asarray((0.0, 1.0, 0.0))),
        ("mixed", r"mixed: $Q_s+Q_{15^\circ}$", mixed),
    )
    rows = (
        ("h0", r"$h_0(q)-h_0(0)$ (eV)", "#0072B2"),
        ("hz", r"$h_z(q)$ (eV)", "#D55E00"),
        ("hx", r"$h_x(q)$ (eV)", "#009E73"),
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(9.0, 7.3),
        sharey="row",
        constrained_layout=True,
    )
    output = {}
    errors = {}
    max_imaginary = 0.0

    for column, (key, title, direction) in enumerate(cuts):
        raw_axis = cached[f"{key}_raw_coordinate"]
        raw_hamiltonian = cached[f"{key}_raw_fci_hamiltonian"]
        dense_axis = cached[f"{key}_dense_coordinate"]
        raw_coordinates = raw_axis[:, None] * direction[None, :]
        dense_coordinates = dense_axis[:, None] * direction[None, :]
        raw_fit = model.neural_energy.predict(raw_coordinates)
        dense_fit = model.neural_energy.predict(dense_coordinates)
        raw_parts = components(raw_hamiltonian)
        fit_parts = components(raw_fit)
        dense_parts = components(dense_fit)
        max_imaginary = max(
            max_imaginary,
            float(np.max(np.abs(raw_parts["hy"]))),
            float(np.max(np.abs(fit_parts["hy"]))),
        )
        errors[key] = {}
        output[f"{key}_raw_coordinate"] = raw_axis
        output[f"{key}_dense_coordinate"] = dense_axis

        for row, (name, ylabel, color) in enumerate(rows):
            panel = axes[row, column]
            raw_value = raw_parts[name].copy()
            fit_value = dense_parts[name].copy()
            raw_fit_value = fit_parts[name].copy()
            if name == "h0":
                raw_origin = np.interp(0.0, raw_axis, raw_value)
                fit_origin = np.interp(0.0, dense_axis, fit_value)
                raw_value -= raw_origin
                raw_fit_value -= fit_origin
                fit_value -= fit_origin
            raw_value *= au2ev
            raw_fit_value *= au2ev
            fit_value *= au2ev
            errors[key][name] = float(
                np.max(np.abs(raw_fit_value - raw_value)) * 1000.0
            )
            panel.plot(dense_axis, fit_value, color=color, lw=1.8)
            panel.scatter(
                raw_axis,
                raw_value,
                s=20,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            panel.axhline(0.0, color="0.82", lw=0.7, zorder=0)
            panel.spines[["top", "right"]].set_visible(False)
            if row == 0:
                panel.set_title(title)
            if column == 0:
                panel.set_ylabel(ylabel)
            if row == 2:
                panel.set_xlabel(r"cut coordinate $q$ (bohr)")
            output[f"{key}_{name}_raw_ev"] = raw_value
            output[f"{key}_{name}_fit_at_raw_ev"] = raw_fit_value
            output[f"{key}_{name}_fit_dense_ev"] = fit_value

    figure.legend(
        handles=(
            Line2D([], [], color="#4D4D4D", lw=1.8, label="MACE fit"),
            Line2D(
                [], [], marker="o", ls="none", ms=4.5, markerfacecolor="white",
                markeredgecolor="black", label="ab initio"
            ),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(r"Procrustes-gauged $\bar H=h_0I+h_x\sigma_x+h_z\sigma_z$")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    np.savez_compressed(args.output.with_suffix(".npz"), **output)

    print(f"maximum |hy| = {max_imaginary * au2ev * 1000:.3e} meV")
    for key, values in errors.items():
        summary = ", ".join(f"{name} {value:.2f}" for name, value in values.items())
        print(f"{key}: maximum component errors [meV]: {summary}")
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".npz"))


if __name__ == "__main__":
    main()
