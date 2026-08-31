#!/usr/bin/env python3
"""Fit standard AbInitioFit artifacts to the completed phenol 3D P-gauge data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.ldr import AbInitioFit
from pyqed.mps.functional import FunctionalTT


HARTREE_TO_EV = au2ev


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _coordinates(grids):
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def _fit_field(grids, values, *, rank, hermitian, seed):
    degrees = tuple(len(grid) - 1 for grid in grids)
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    model = FunctionalTT(
        degrees=degrees,
        rank=int(rank),
        bounds=bounds,
        hermitian=bool(hermitian),
        normalization="frobenius",
        regularization=1.0e-14,
        sweeps=50,
        rtol=1.0e-13,
        random_state=int(seed),
    ).fit_grid(grids, values)
    predicted = np.asarray(model.predict(_coordinates(grids))).reshape(values.shape)
    residual = predicted - values
    return model, residual


def _dense_surfaces(output, energy, grids, shift):
    radial = np.linspace(grids[0][0], grids[0][-1], 151)
    torsion = np.linspace(grids[1][0], grids[1][-1], 121)
    bend = np.linspace(grids[2][0], grids[2][-1], 101)
    rr, pp = np.meshgrid(radial, torsion, indexing="ij")
    planar = np.column_stack(
        (rr.reshape(-1), pp.reshape(-1), np.full(rr.size, grids[2][1]))
    )
    planar_energy = np.linalg.eigvalsh(energy.predict(planar)).reshape(
        len(radial), len(torsion), 3
    )
    rr_bend, tt = np.meshgrid(radial, bend, indexing="ij")
    untwisted = np.column_stack(
        (rr_bend.reshape(-1), np.zeros(rr_bend.size), tt.reshape(-1))
    )
    bend_energy = np.linalg.eigvalsh(energy.predict(untwisted)).reshape(
        len(radial), len(bend), 3
    )
    reference = float(np.min(planar_energy[0, len(torsion) // 2]))
    figure, panels = plt.subplots(2, 3, figsize=(11.7, 6.5), constrained_layout=True)
    levels = 30
    for state in range(3):
        values = (planar_energy[:, :, state] - reference) * HARTREE_TO_EV
        image = panels[0, state].contourf(
            radial, torsion, values.T, levels=levels, cmap="viridis"
        )
        panels[0, state].set(
            xlabel=r"$R_{OH}$ ($\AA$)",
            ylabel=r"$\phi$ (rad)",
            title=rf"P{state}: $\theta=108.8^\circ$",
        )
        figure.colorbar(image, ax=panels[0, state], label="energy (eV)")
        values = (bend_energy[:, :, state] - reference) * HARTREE_TO_EV
        image = panels[1, state].contourf(
            radial,
            np.rad2deg(bend),
            values.T,
            levels=levels,
            cmap="viridis",
        )
        panels[1, state].set(
            xlabel=r"$R_{OH}$ ($\AA$)",
            ylabel=r"$\theta$ (degree)",
            title=rf"P{state}: $\phi=0$",
        )
        figure.colorbar(image, ax=panels[1, state], label="energy (eV)")
    figure.suptitle("Phenol fitted three-state P-gauge surfaces")
    png = output / "phenol_3d_p_gauge_fitted_surfaces.png"
    pdf = output / "phenol_3d_p_gauge_fitted_surfaces.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _fit_diagnostics(output, energy_residual, link_residuals, link_values):
    energy_error = np.linalg.norm(energy_residual, axis=(-2, -1)) * HARTREE_TO_EV
    figure, panels = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    panels[0].semilogy(np.sort(energy_error.reshape(-1)), "o", ms=3.0)
    panels[0].set(
        xlabel="sorted grid point",
        ylabel=r"$\|\Delta H_P\|_F$ (eV)",
        title="Energy-field reconstruction",
    )
    for axis, (residual, values) in enumerate(zip(link_residuals, link_values)):
        error = np.linalg.norm(residual, axis=(-2, -1)).reshape(-1)
        panels[1].semilogy(
            np.sort(error), "o", ms=2.8, label=(r"$R_{OH}$", r"$\phi$", r"$\theta$")[axis]
        )
        relative = error / np.maximum(
            np.linalg.norm(values, axis=(-2, -1)).reshape(-1),
            np.finfo(float).tiny,
        )
        panels[2].semilogy(
            np.sort(relative), "o", ms=2.8, label=(r"$R_{OH}$", r"$\phi$", r"$\theta$")[axis]
        )
    panels[1].set(
        xlabel="sorted edge",
        ylabel=r"$\|\Delta S^P\|_F$",
        title="Link-field reconstruction",
    )
    panels[2].set(
        xlabel="sorted edge",
        ylabel="relative link error",
        title="Relative link reconstruction",
    )
    panels[1].legend(fontsize=7.5)
    panels[2].legend(fontsize=7.5)
    for panel in panels:
        panel.grid(alpha=0.2)
    png = output / "phenol_3d_p_gauge_fit_diagnostics.png"
    pdf = output / "phenol_3d_p_gauge_fit_diagnostics.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_3d_p_gauge_20260820/"
            "phenol_sa6_3d_p_gauge_data.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_fit_20260820"),
    )
    parser.add_argument("--rank", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with np.load(args.data, allow_pickle=False) as archive:
        grids = tuple(
            np.asarray(archive[name]) for name in ("r_oh", "phi", "theta")
        )
        points = tuple(map(tuple, archive["points"]))
        pairs = tuple((tuple(left), tuple(right)) for left, right in archive["pairs"])
        p_hamiltonian = np.asarray(archive["p_hamiltonian"])
        p_links = np.asarray(archive["p_links"])
    shape = tuple(len(grid) for grid in grids)
    if len(points) != int(np.prod(shape)):
        raise ValueError("P-gauge fitting requires the completed product grid")
    energy_values = np.empty((*shape, 3, 3), dtype=complex)
    for point, value in zip(points, p_hamiltonian):
        energy_values[point] = value
    energy_shift = float(np.min(np.linalg.eigvalsh(energy_values[0, 2, 1])))
    shifted = energy_values - energy_shift * np.eye(3)
    energy, energy_residual = _fit_field(
        grids, shifted, rank=args.rank, hermitian=True, seed=17
    )

    pair_values = dict(zip(pairs, p_links))
    link_models = []
    link_values = []
    link_residuals = []
    for axis in range(3):
        axis_grids = list(grids)
        axis_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
        axis_grids = tuple(axis_grids)
        axis_shape = tuple(len(grid) for grid in axis_grids)
        values = np.empty((*axis_shape, 3, 3), dtype=complex)
        for left in np.ndindex(axis_shape):
            right = list(left)
            right[axis] += 1
            values[left] = pair_values[(left, tuple(right))]
        model, residual = _fit_field(
            axis_grids,
            values,
            rank=args.rank,
            hermitian=False,
            seed=31 + axis,
        )
        link_models.append(model)
        link_values.append(values)
        link_residuals.append(residual)

    energy_error = np.linalg.norm(energy_residual, axis=(-2, -1)) * HARTREE_TO_EV
    link_error = [
        np.linalg.norm(value, axis=(-2, -1))
        for value in link_residuals
    ]
    rng = np.random.default_rng(73)
    coordinates = np.column_stack(
        [rng.uniform(grid[0], grid[-1], 4096) for grid in grids]
    )
    dense_hamiltonian = np.asarray(energy.predict(coordinates))
    mirror = np.array(coordinates, copy=True)
    mirror[:, 1] *= -1.0
    mirror_hamiltonian = np.asarray(energy.predict(mirror))
    reflection_error = float(
        np.max(
            np.abs(
                np.linalg.eigvalsh(dense_hamiltonian)
                - np.linalg.eigvalsh(mirror_hamiltonian)
            )
        )
        * HARTREE_TO_EV
    )
    hermiticity = float(
        np.max(
            np.linalg.norm(
                dense_hamiltonian
                - dense_hamiltonian.conj().swapaxes(-1, -2),
                axis=(-2, -1),
            )
        )
    )
    interpolated_link_singular_ranges = []
    for model in link_models:
        link_coordinates = np.column_stack(
            [rng.uniform(lower, upper, 4096) for lower, upper in model.bounds]
        )
        singular = np.linalg.svd(model.predict(link_coordinates), compute_uv=False)
        interpolated_link_singular_ranges.append(
            [float(np.min(singular)), float(np.max(singular))]
        )
    info = {
        "backend": "completed-grid-graph-p-gauge-functional-tt",
        "grid": shape,
        "rank": args.rank,
        "energy_ranks": energy.ranks_,
        "link_ranks": [model.ranks_ for model in link_models],
        "maximum_energy_grid_error_ev": float(np.max(energy_error)),
        "rms_energy_grid_error_ev": float(np.sqrt(np.mean(energy_error**2))),
        "maximum_link_grid_error": [float(np.max(value)) for value in link_error],
        "rms_link_grid_error": [
            float(np.sqrt(np.mean(value**2))) for value in link_error
        ],
        "dense_energy_hermiticity_defect": hermiticity,
        "dense_reflection_spectral_error_ev": reflection_error,
        "interpolated_link_singular_ranges": interpolated_link_singular_ranges,
        "validation_scope": (
            "exact completed-grid reconstruction plus dense structural checks; "
            "no off-grid ab-initio holdout"
        ),
        "source": str(args.data),
    }
    fit = AbInitioFit(grids, 3, anchor=(0, 2, 1), energy_shift=energy_shift)
    fit.energy = energy
    fit.links = tuple(link_models)
    fit.info = info
    fit.config = {
        "representation": "graph-p-gauge-links",
        "rank": args.rank,
        "degrees": [len(grid) - 1 for grid in grids],
        "completed_grid": True,
    }
    fit.seconds = time.perf_counter() - started
    fit.success = True
    fit.message = "fitted"
    fit.frames.points.update(np.ndindex(shape))
    fields = args.output / "fields"
    fit.save(
        fields,
        labels=("R_OH", "phi_CCOH", "theta_COH"),
        metadata={
            "system": "phenol",
            "electronic_space": "equilibrium-selected three-state P gauge",
            "source_data": str(args.data),
        },
    )
    surface_png, surface_pdf = _dense_surfaces(
        args.output, energy, grids, energy_shift
    )
    diagnostic_png, diagnostic_pdf = _fit_diagnostics(
        args.output, energy_residual, link_residuals, link_values
    )
    summary = {
        "passed": bool(
            info["maximum_energy_grid_error_ev"] <= 1.0e-8
            and max(info["maximum_link_grid_error"]) <= 1.0e-10
            and hermiticity <= 1.0e-12
            and reflection_error <= 1.0e-4
            and min(value[0] for value in interpolated_link_singular_ranges) >= 0.5
            and max(value[1] for value in interpolated_link_singular_ranges) <= 1.01
        ),
        "energy_shift_hartree": energy_shift,
        "fit": info,
        "fields": str(fields),
        "figures": {
            "surfaces": str(surface_png),
            "surfaces_pdf": str(surface_pdf),
            "diagnostics": str(diagnostic_png),
            "diagnostics_pdf": str(diagnostic_pdf),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
