#!/usr/bin/env python3
"""Plot two-coordinate cuts through the fitted phenol P-gauge Hamiltonian."""

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
from matplotlib.colors import TwoSlopeNorm

from pyqed.ml import CorrectedMatrixField, MACE, RadialMatrixCorrection
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.units import au2ev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALCULATION_ROOT = (
    PROJECT_ROOT / "dataset/phenol_sa6_casscf_production/dynamics/5d_50fs"
)
DEFAULT_DATA = (
    CALCULATION_ROOT
    / "inputs/p_gauge_5d_inward/phenol_sa6_5d_p_gauge_inward.npz"
)
DEFAULT_CHECKPOINT = (
    CALCULATION_ROOT / "model/final/phenol_sa6_5d_mace_y.pt"
)
DEFAULT_CORRECTION = (
    CALCULATION_ROOT
    / "model/radial_correction/phenol_sa6_5d_radial_delta.npz"
)
DEFAULT_OUTPUT = CALCULATION_ROOT / "diabatic_surfaces"


CUTS = (
    (1, r"torsion $\phi$ (rad)", lambda x: x, lambda x: x),
    (2, r"bend $\theta$ (deg)", np.rad2deg, np.deg2rad),
    (3, r"$Q_{16a}$ ($\mathrm{\AA}\sqrt{\mathrm{amu}}$)", lambda x: x, lambda x: x),
    (4, r"$Q_{8a}$ ($\mathrm{\AA}\sqrt{\mathrm{amu}}$)", lambda x: x, lambda x: x),
)


def _predict_batched(field, coordinates, batch_size):
    return np.concatenate(
        [
            field.predict(coordinates[start : start + batch_size])
            for start in range(0, len(coordinates), batch_size)
        ],
        axis=0,
    )


def _supported_points(coordinates, equilibrium, axis):
    other = [index for index in range(1, 5) if index != axis]
    mask = np.all(
        np.isclose(coordinates[:, other], equilibrium[other], atol=1.0e-7), axis=1
    )
    return coordinates[mask][:, (0, axis)]


def _plot_grid(
    surfaces,
    radial,
    transverse,
    samples,
    equilibrium,
    *,
    labels,
    cmap,
    norm,
    colorbar_label,
    output,
):
    figure, panels = plt.subplots(
        4, 3, figsize=(11.0, 11.2), sharex=True, constrained_layout=True
    )
    mappable = None
    for row, (axis, ylabel, display, _) in enumerate(CUTS):
        y = display(transverse[row])
        sample_y = display(samples[row][:, 1])
        for column in range(3):
            panel = panels[row, column]
            mappable = panel.pcolormesh(
                radial,
                y,
                surfaces[row, :, :, column],
                shading="auto",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            panel.scatter(
                samples[row][:, 0],
                sample_y,
                s=8,
                facecolors="none",
                edgecolors="white",
                linewidths=0.38,
                alpha=0.8,
            )
            panel.plot(
                equilibrium[0],
                display(equilibrium[axis]),
                marker="+",
                ms=7,
                mew=1.0,
                color="black",
            )
            if row == 0:
                panel.set_title(labels[column])
            if column == 0:
                panel.set_ylabel(ylabel)
            if row == 3:
                panel.set_xlabel(r"$R_{\mathrm{OH}}$ ($\mathrm{\AA}$)")
            panel.tick_params(direction="in", top=True, right=True)
    colorbar = figure.colorbar(mappable, ax=panels, shrink=0.87, pad=0.02)
    colorbar.set_label(colorbar_label)
    figure.savefig(output.with_suffix(".png"), dpi=260)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    chart = PhenolReactiveChart(modes=data["modes"])
    model = MACE.load(args.checkpoint, chart.geometry, device="cpu", distill=False)
    correction = RadialMatrixCorrection.load(args.radial_correction)
    field = CorrectedMatrixField(model.neural_energy, correction)

    coordinates = data["coordinates"]
    equilibrium = chart.equilibrium
    radial = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), args.radial_points)
    all_matrices = []
    transverse = []
    samples = []
    for axis, _, _, _ in CUTS:
        values = np.linspace(
            coordinates[:, axis].min(), coordinates[:, axis].max(), args.transverse_points
        )
        r_mesh, q_mesh = np.meshgrid(radial, values)
        grid = np.tile(equilibrium, (r_mesh.size, 1))
        grid[:, 0] = r_mesh.ravel()
        grid[:, axis] = q_mesh.ravel()
        matrices = _predict_batched(field, grid, args.batch_size)
        matrices = matrices.reshape(args.transverse_points, args.radial_points, 3, 3)
        all_matrices.append(matrices)
        transverse.append(values)
        samples.append(_supported_points(coordinates, equilibrium, axis))
    matrices = np.asarray(all_matrices)

    equilibrium_matrix = field.predict(equilibrium[None, :])[0]
    energy_zero = float(np.linalg.eigvalsh(equilibrium_matrix).min())
    diagonal = (
        np.real(np.diagonal(matrices, axis1=-2, axis2=-1)) - energy_zero
    ) * au2ev
    couplings = np.stack(
        (matrices[..., 0, 1], matrices[..., 0, 2], matrices[..., 1, 2]), axis=-1
    )
    coupling_real = np.real(couplings) * au2ev * 1000.0
    maximum_imaginary_mev = float(np.max(np.abs(np.imag(couplings))) * au2ev * 1000.0)

    diagonal_min = float(np.min(diagonal))
    diagonal_max = float(np.max(diagonal))
    coupling_limit = float(np.max(np.abs(coupling_real)))
    if coupling_limit == 0.0:
        coupling_limit = 1.0

    _plot_grid(
        diagonal,
        radial,
        transverse,
        samples,
        equilibrium,
        labels=(r"$\bar H_{11}$", r"$\bar H_{22}$", r"$\bar H_{33}$"),
        cmap="viridis",
        norm=matplotlib.colors.Normalize(diagonal_min, diagonal_max),
        colorbar_label="diabatic potential relative to the equilibrium minimum (eV)",
        output=args.output / "phenol_5d_p_gauge_diabatic_potentials",
    )
    _plot_grid(
        coupling_real,
        radial,
        transverse,
        samples,
        equilibrium,
        labels=(r"$\mathrm{Re}\,\bar H_{12}$", r"$\mathrm{Re}\,\bar H_{13}$", r"$\mathrm{Re}\,\bar H_{23}$"),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-coupling_limit, vcenter=0.0, vmax=coupling_limit),
        colorbar_label="signed diabatic coupling (meV)",
        output=args.output / "phenol_5d_p_gauge_diabatic_couplings",
    )

    np.savez_compressed(
        args.output / "phenol_5d_p_gauge_diabatic_surfaces.npz",
        radial_angstrom=radial,
        transverse_0=transverse[0],
        transverse_1=transverse[1],
        transverse_2=transverse[2],
        transverse_3=transverse[3],
        diagonal_ev=diagonal,
        coupling_real_mev=coupling_real,
        equilibrium=equilibrium,
        equilibrium_energy_zero_hartree=np.asarray(energy_zero),
    )
    summary = {
        "representation": "three-state P-gauge quasi-diabatic Hamiltonian",
        "radial_range_angstrom": [float(radial[0]), float(radial[-1])],
        "radial_points": int(args.radial_points),
        "transverse_points_per_cut": int(args.transverse_points),
        "diabatic_potential_range_ev": [diagonal_min, diagonal_max],
        "signed_coupling_range_mev": [
            float(np.min(coupling_real)),
            float(np.max(coupling_real)),
        ],
        "maximum_imaginary_coupling_mev": maximum_imaginary_mev,
        "sample_counts_on_cuts": [int(len(points)) for points in samples],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--radial-correction", type=Path, default=DEFAULT_CORRECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radial-points", type=int, default=161)
    parser.add_argument("--transverse-points", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
