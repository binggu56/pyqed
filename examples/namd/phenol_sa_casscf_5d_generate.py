#!/usr/bin/env python3
"""Generate a restartable space-filling phenol 5D SA-CASSCF database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pyqed")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    phenol_sa6_protocol,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.units import au2ev


PARITIES = np.asarray((1.0, -1.0, 1.0, -1.0, 1.0))
R_GRID = np.asarray((0.95, 1.15, 1.30, 1.55, 1.85, 2.10, 2.40, 2.70, 3.00))
PHI_GRID = np.asarray((-0.40, -0.20, 0.0, 0.20, 0.40))
THETA_GRID = np.deg2rad(np.asarray((104.0, 108.8, 114.0)))
Q16A_GRID = np.asarray((-0.50, -0.25, 0.0, 0.25, 0.50))
Q8A_GRID = np.asarray((-0.20, -0.10, 0.0, 0.10, 0.20))
GRIDS = (R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID)
SCALES = np.asarray((0.25, 0.20, np.deg2rad(5.0), 0.25, 0.10))
ANCHOR = (0, 2, 1, 2, 2)


class PhenolGeometry:
    """Pickle-safe 5D geometry callback with exact stored seed geometries."""

    def __init__(self, modes, exact_geometries=None):
        self.modes = np.asarray(modes, dtype=float)
        self.exact_geometries = {
            tuple(map(float, coordinate)): np.asarray(geometry, dtype=float)
            for coordinate, geometry in (exact_geometries or {}).items()
        }

    def __call__(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        exact = self.exact_geometries.get(tuple(map(float, coordinate)))
        if exact is not None:
            return np.array(exact, copy=True)
        return PhenolReactiveChart(modes=self.modes).geometry(coordinate)


def recover_chart(database, run_id):
    """Recover the production modes and exact geometry map from a saved run."""

    run = database.run(run_id)
    exact = {}
    representatives = {}
    for item in run["records"]:
        sample = item["sample"]
        coordinate = tuple(map(float, sample["coordinates"]))
        exact[coordinate] = np.asarray(sample["geometry"], dtype=float)
        representative_coordinate = tuple(
            map(float, sample.get("representative_coordinates", coordinate))
        )
        representative_geometry = np.asarray(
            sample.get("representative_geometry", sample["geometry"]), dtype=float
        )
        exact[representative_coordinate] = representative_geometry
        representatives[representative_coordinate] = representative_geometry
    if not representatives:
        raise RuntimeError(f"electronic run {run_id!r} contains no geometries")

    coordinates = np.asarray(list(representatives), dtype=float)
    geometries = np.asarray(list(representatives.values()), dtype=float)
    base = np.asarray(
        [
            PhenolReactiveChart().geometry((*coordinate[:3], 0.0, 0.0))
            for coordinate in coordinates
        ]
    )
    amplitudes = coordinates[:, 3:5]
    modes = np.linalg.lstsq(
        amplitudes,
        (geometries - base).reshape(len(coordinates), -1),
        rcond=None,
    )[0].reshape(2, *geometries.shape[1:])
    chart = PhenolReactiveChart(modes=modes)
    reconstructed = base + np.einsum(
        "nk,kia->nia", amplitudes, chart.modes, optimize=True
    )
    maximum_defect = float(np.max(np.abs(reconstructed - geometries)))
    if maximum_defect > 1.0e-12:
        raise RuntimeError(
            f"recovered normal modes miss stored geometries by {maximum_defect:.3e} A"
        )
    metadata = run["metadata"]
    frequencies = np.asarray(metadata["mode_frequencies_cm1"], dtype=float)
    labels = tuple(map(str, metadata["mode_labels"]))
    return chart, frequencies, labels, exact, {
        "source_run_id": str(run_id),
        "stored_geometries": len(exact),
        "canonical_geometries": len(representatives),
        "maximum_reconstruction_defect_angstrom": maximum_defect,
    }


def _canonical(index):
    index = tuple(map(int, index))
    coordinates = np.asarray([grid[value] for grid, value in zip(GRIDS, index)])
    for axis in (1, 3):
        if abs(coordinates[axis]) > 1.0e-12:
            if coordinates[axis] < 0.0:
                reflected = list(index)
                reflected[1] = len(PHI_GRID) - 1 - reflected[1]
                reflected[3] = len(Q16A_GRID) - 1 - reflected[3]
                return tuple(reflected)
            break
    return index


def design_points(count, seed):
    """Chemical coordinate crosses followed by a scrambled Sobol design."""

    count = int(count)
    if count < len(R_GRID) * 11:
        raise ValueError(f"at least {len(R_GRID) * 11} representatives are required")
    points = []
    for radial in range(len(R_GRID)):
        for torsion in (2, 3, 4):
            points.append((radial, torsion, 1, 2, 2))
        for bend in (0, 1, 2):
            points.append((radial, 2, bend, 2, 2))
        for q16a in (3, 4):
            points.append((radial, 2, 1, q16a, 2))
        for q8a in (0, 1, 3, 4):
            points.append((radial, 2, 1, 2, q8a))
    points = list(dict.fromkeys(_canonical(point) for point in points))
    engine = qmc.Sobol(5, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(max(4 * count, 2))))
    for unit in engine.random_base2(power):
        index = tuple(
            min(int(value * size), size - 1)
            for value, size in zip(unit, map(len, GRIDS))
        )
        index = _canonical(index)
        if index not in points:
            points.append(index)
        if len(points) >= count:
            break
    if ANCHOR in points:
        points.remove(ANCHOR)
    return tuple((ANCHOR, *points))[:count]


def chart_reflection_defect(chart, indices):
    reflection = np.diag((1.0, 1.0, -1.0))
    defect = 0.0
    for index in indices:
        coordinate = np.asarray([grid[value] for grid, value in zip(GRIDS, index)])
        defect = max(
            defect,
            float(
                np.max(
                    np.abs(
                        chart.geometry(coordinate) @ reflection.T
                        - chart.geometry(coordinate * PARITIES)
                    )
                )
            ),
        )
    return defect


def _plot(output, coordinates, records):
    energies = np.asarray([record["energies"] for record in records])
    gradients = np.asarray(
        [float(record.get("orbital_gradient", np.nan)) for record in records]
    )
    macroiterations = np.asarray(
        [len(np.asarray(record.get("macro_history", ()))) for record in records]
    )
    reference = float(np.nanmin(energies[:, 0]))

    figure, panels = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
    scatter = panels[0, 0].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=np.rad2deg(coordinates[:, 2]),
        s=13,
        cmap="viridis",
        alpha=0.72,
    )
    figure.colorbar(scatter, ax=panels[0, 0], label=r"$\theta$ (deg)")
    panels[0, 0].set(
        xlabel=r"$R_{\mathrm{OH}}$ ($\mathrm{\AA}$)",
        ylabel=r"$\phi$ (rad)",
        title="Radial, torsional, and bend coverage",
    )

    scatter = panels[0, 1].scatter(
        coordinates[:, 3],
        coordinates[:, 4],
        c=coordinates[:, 0],
        s=13,
        cmap="plasma",
        alpha=0.72,
    )
    figure.colorbar(scatter, ax=panels[0, 1], label=r"$R_{\mathrm{OH}}$ ($\mathrm{\AA}$)")
    panels[0, 1].set(
        xlabel=r"$Q_{16a}$ ($\mathrm{\AA}\sqrt{\mathrm{amu}}$)",
        ylabel=r"$Q_{8a}$ ($\mathrm{\AA}\sqrt{\mathrm{amu}}$)",
        title="Normal-coordinate coverage",
    )

    for root in range(energies.shape[1]):
        panels[1, 0].scatter(
            coordinates[:, 0],
            (energies[:, root] - reference) * au2ev,
            s=7,
            alpha=0.35,
            label=f"root {root}",
        )
    panels[1, 0].set(
        xlabel=r"$R_{\mathrm{OH}}$ ($\mathrm{\AA}$)",
        ylabel="energy relative to the global minimum (eV)",
        title="All six state-averaged roots",
    )
    panels[1, 0].legend(frameon=False, ncol=2, fontsize=7)

    finite = np.isfinite(gradients) & (gradients > 0.0)
    panels[1, 1].scatter(
        macroiterations[finite], gradients[finite], s=12, alpha=0.6, color="#6A3D9A"
    )
    panels[1, 1].axhline(1.0e-5, color="#D55E00", ls="--", lw=1.0)
    panels[1, 1].set_yscale("log")
    panels[1, 1].set(
        xlabel="CASSCF macroiterations",
        ylabel="final orbital gradient",
        title="Orbital-relaxation convergence",
    )
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.97, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(alpha=0.16)
    png = output / "phenol_sa6_5d_generation_diagnostics.png"
    pdf = output / "phenol_sa6_5d_generation_diagnostics.pdf"
    figure.savefig(png, dpi=260)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    database = ElectronicDatabase(args.database)
    chart, frequencies, labels, exact_geometries, chart_recovery = recover_chart(
        database, args.source_run_id
    )
    protocol = phenol_sa6_protocol()
    canonical = design_points(args.samples, args.seed)
    reflection_defect = chart_reflection_defect(chart, canonical)
    symmetry = PhenolReflectionSymmetry(
        torsion_axis=1,
        odd_axes=(1, 3),
        tolerance=max(1.0e-10, 1.05 * reflection_defect),
    )
    provider = PhenolSACASSCFProvider(
        database, protocol, coordinate_scale=SCALES, verbose=int(not args.quiet)
    )
    run_id = f"phenol-sa6-5d-spacefill-v1-s{args.seed}-n{args.samples}"

    def progress(index, stats):
        coordinate = tuple(GRIDS[axis][value] for axis, value in enumerate(index))
        print(
            f"[electronic] built {stats['built']}/{len(canonical)} canonical: "
            f"R={coordinate[0]:.2f}, phi={coordinate[1]:+.2f}, "
            f"theta={np.rad2deg(coordinate[2]):.1f}, "
            f"Q16a={coordinate[3]:+.2f}, Q8a={coordinate[4]:+.2f}",
            flush=True,
        )

    try:
        with AbInitioFit(
            GRIDS,
            6,
            electronic=provider,
            geometry=PhenolGeometry(chart.modes, exact_geometries),
            symmetry=symmetry,
            database=database,
            protocol=protocol,
            run_id=run_id,
            run_metadata={
                "purpose": "space-filling five-dimensional electronic database",
                "coordinates": chart.names,
                "mode_labels": labels,
                "mode_frequencies_cm1": frequencies,
                "chart_recovery": chart_recovery,
                "canonical_samples": int(args.samples),
                "workers": int(args.workers),
                "sampling": "deterministic chemical crosses plus scrambled Sobol",
            },
            frame=lambda record: record,
            energies=lambda record: record["energies"],
            anchor=ANCHOR,
            workers=args.workers,
            progress=progress,
            energy_shift=None,
        ) as fit:
            points = fit.expand_points(canonical)
            records = fit.frames.get_many(points)
            coordinates = np.asarray([fit.coordinates(point) for point in points])
            record_ids = [fit.frames.record_id(point) for point in points]
    finally:
        provider.close()
        database.close()

    converged = np.asarray(
        [bool(record.get("orbital_relaxed", record.get("converged", False))) for record in records]
    )
    gradients = np.asarray(
        [float(record.get("orbital_gradient", np.nan)) for record in records]
    )
    png, pdf = _plot(args.output, coordinates, records)
    np.savez_compressed(
        args.output / "phenol_sa6_5d_generation_index.npz",
        coordinates=coordinates,
        record_ids=np.asarray(record_ids),
        orbital_relaxed=converged,
        orbital_gradients=gradients,
    )
    summary = {
        "method": "SA(6)-CASSCF(10e,10o)/6-31+G*",
        "run_id": run_id,
        "canonical_samples": len(canonical),
        "effective_samples": len(points),
        "workers": int(args.workers),
        "all_orbitals_relaxed": bool(np.all(converged)),
        "failed_or_unrelaxed": int(np.count_nonzero(~converged)),
        "maximum_orbital_gradient": float(np.nanmax(gradients)),
        "seconds": float(time.perf_counter() - started),
        "database": str(args.database.resolve()),
        "figure": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not np.all(converged):
        raise RuntimeError(f"{np.count_nonzero(~converged)} geometries did not relax")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run-id", default="phenol-sa6-5d-pilot-v1-s61-n128")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
