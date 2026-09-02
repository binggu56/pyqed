#!/usr/bin/env python3
"""Two-dimensional ab initio ethylene CI -> MACE -> FTT -> TNLDR benchmark.

The calculation is an adaptation, not an exact reproduction, of the ethylene
MRCI conical-intersection example in ``Machine Learning Seams of Conical
Intersection: A Characteristic Polynomial Approach``, J. Phys. Chem. Lett.
2023, https://doi.org/10.1021/acs.jpclett.3c01649.  The published
twisted--pyramidalized MECI geometry is the source template; an additional
pyramidalization recenters the chart at the restricted SA-CASSCF/6-31G*
crossing.  The finite angular deformations are not optimized branching vectors.
Production data use SA(2)-CASSCF(2,2), omitting MRCI dynamic correlation.

The persistent electronic database defaults to OneDrive ``data/pyqed`` and is
explicitly rejected if it resolves inside the PyQED repository.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import qmc

from pyqed.dvr import DVR, ExponentialDVR, SineDVR
from pyqed.ldr import (
    AbInitioFit,
    Coord,
    ETHYLENE_CI_BOUNDS,
    ETHYLENE_SPECIES,
    EthyleneCIElectronicDriver,
    default_ethylene_database_path,
    ethylene_ci_geometry,
    keo,
)
from pyqed.ml import MACE
from pyqed.namd import TNLDR
from pyqed.units import au2ev


REPOSITORY = Path(__file__).resolve().parents[2]


def external_database_path(path):
    """Resolve ``path`` and reject database storage inside this repository."""

    resolved = Path(path).expanduser().resolve()
    repository = REPOSITORY.resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError(
            "the ethylene ab initio database must be outside the PyQED repository"
        )
    return resolved


def graph_pairs(
    coordinates, neighbors=3, bounds=ETHYLENE_CI_BOUNDS, periodic_axes=(0,)
):
    """Return a connected local graph for electronic link targets."""

    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        raise ValueError("at least two coordinates are required")
    bounds = np.asarray(bounds, dtype=float)
    scale = np.maximum(np.ptp(bounds, axis=1), 1.0e-12)
    delta = np.abs(coordinates[:, None, :] - coordinates[None, :, :])
    for axis in periodic_axes:
        delta[..., axis] = np.minimum(
            delta[..., axis], scale[axis] - delta[..., axis]
        )
    distances = np.linalg.norm(delta / scale, axis=-1)
    tree = minimum_spanning_tree(distances).tocoo()
    pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    np.fill_diagonal(distances, np.inf)
    count = min(int(neighbors), len(coordinates) - 1)
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], count - 1)[:count]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    return np.asarray(sorted(pairs), dtype=int)


def sobol_coordinates(count, seed, bounds=ETHYLENE_CI_BOUNDS, include_origin=False):
    """Return deterministic scattered chart coordinates."""

    count = int(count)
    minimum = 2 if not include_origin else 3
    if count < minimum:
        raise ValueError(f"count must be at least {minimum}")
    requested = count - int(include_origin)
    power = int(np.ceil(np.log2(max(2, requested))))
    unit = qmc.Sobol(2, scramble=True, seed=int(seed)).random_base2(power)
    box = np.asarray(bounds, dtype=float)
    coordinates = qmc.scale(unit[:requested], box[:, 0], box[:, 1])
    if include_origin:
        coordinates = np.vstack((np.zeros(2), coordinates))
    return coordinates


def model_errors(energy_model, feature_model, fields):
    predicted_h = np.asarray(energy_model.predict(fields["coordinates"]))
    exact_h = np.asarray(fields["hamiltonians"])
    pairs = np.asarray(fields["pairs"], dtype=int)
    feature = np.asarray(feature_model.predict(fields["coordinates"]))
    predicted_links = (
        feature[pairs[:, 0]].conj().swapaxes(-1, -2)
        @ feature[pairs[:, 1]]
    )
    exact_links = np.asarray(fields["links"])
    return {
        "maximum_hamiltonian_error_hartree": float(
            np.max(np.linalg.norm(predicted_h - exact_h, axis=(-2, -1)))
        ),
        "rms_hamiltonian_error_hartree": float(
            np.sqrt(np.mean(np.abs(predicted_h - exact_h) ** 2))
        ),
        "relative_link_error": float(
            np.linalg.norm(predicted_links - exact_links)
            / max(np.linalg.norm(exact_links), np.finfo(float).tiny)
        ),
    }


def plot_database_coverage(output, fields):
    coordinates = np.asarray(fields["coordinates"])
    energies = np.linalg.eigvalsh(np.asarray(fields["hamiltonians"]))
    reference = float(np.min(energies))
    values = (
        (energies[:, 0] - reference) * au2ev,
        (energies[:, 1] - reference) * au2ev,
        (energies[:, 1] - energies[:, 0]) * au2ev,
    )
    titles = (r"Lower root", r"Upper root", r"$S_1-S_0$ gap")
    figure, panels = plt.subplots(
        1, 3, figsize=(10.8, 3.4), sharex=True, sharey=True, constrained_layout=True
    )
    for panel, value, title in zip(panels, values, titles):
        artist = panel.scatter(
            np.rad2deg(coordinates[:, 0]),
            np.rad2deg(coordinates[:, 1]),
            c=value,
            s=28,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.25,
        )
        figure.colorbar(artist, ax=panel, label="Energy (eV)")
        panel.set(
            xlabel=r"Torsion displacement (degree)",
            title=title,
        )
    panels[0].set_ylabel(r"Pyramid displacement (degree)")
    path = output / "ethylene_ci_2d_database_coverage"
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def plot_direct_periodic(output, axes, energies):
    """Plot the direct periodic-grid surfaces and their seam closure."""

    torsion, pyramid = (np.rad2deg(axis) for axis in axes)
    energies = np.asarray(energies)
    reference = float(np.min(energies))
    figure, panels = plt.subplots(
        1, 3, figsize=(10.5, 3.2), constrained_layout=True
    )
    fields = (
        (energies[..., 0] - reference) * au2ev,
        (energies[..., 1] - reference) * au2ev,
        (energies[..., 1] - energies[..., 0]) * au2ev,
    )
    titles = (r"Lower state", r"Upper state", r"$S_1-S_0$ gap")
    for number, (panel, field, title) in enumerate(zip(panels, fields, titles)):
        levels = (
            np.linspace(0.0, min(2.0, float(np.max(field))), 29)
            if number == 2
            else 28
        )
        artist = panel.contourf(
            torsion, pyramid, field.T, levels=levels, cmap="viridis",
            extend="max" if number == 2 else "neither",
        )
        figure.colorbar(artist, ax=panel, label="Energy (eV)")
        panel.set(
            xlabel="Periodic torsion (degree)",
            ylabel="Pyramidalization (degree)",
            title=title,
        )
    path = output / "ethylene_ci_2d_direct_periodic"
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def plot_benchmark(output, axes, exact, point_error, link_midpoints, link_error):
    torsion, pyramid = (np.rad2deg(axis) for axis in axes)
    reference = float(np.min(exact))
    gap = (exact[..., 1] - exact[..., 0]) * au2ev
    levels = 28
    figure, panels = plt.subplots(1, 4, figsize=(12.0, 2.9), constrained_layout=True)
    lower = panels[0].contourf(
        torsion, pyramid, ((exact[..., 0] - reference) * au2ev).T,
        levels=levels, cmap="viridis",
    )
    figure.colorbar(lower, ax=panels[0], label="Energy (eV)")
    gap_artist = panels[1].contourf(
        torsion, pyramid, gap.T, levels=levels, cmap="magma",
    )
    figure.colorbar(gap_artist, ax=panels[1], label="Gap (eV)")
    error_artist = panels[2].contourf(
        torsion, pyramid, (point_error * 1000.0 * au2ev).T,
        levels=levels, cmap="cividis",
    )
    figure.colorbar(error_artist, ax=panels[2], label="Max error (meV)")
    link_artist = panels[3].scatter(
        np.rad2deg(link_midpoints[:, 0]),
        np.rad2deg(link_midpoints[:, 1]),
        c=link_error,
        cmap="plasma",
        s=22,
    )
    figure.colorbar(
        link_artist, ax=panels[3], label=r"$\|\sigma(L_\mathrm{fit})-\sigma(L_\mathrm{AI})\|_2$"
    )
    titles = (
        r"(a) Direct lower surface",
        r"(b) Direct adiabatic gap",
        r"(c) FTT--MACE PES error",
        r"(d) Gauge-invariant link-spectrum error",
    )
    for panel, title in zip(panels, titles):
        panel.set(
            xlabel=r"Torsion displacement (degree)",
            ylabel=r"Pyramidalization displacement (degree)",
            title=title,
        )
    path = output / "ethylene_ci_2d_tnldr_benchmark"
    figure.savefig(path.with_suffix(".png"), dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=default_ethylene_database_path()
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_ethylene_database_path().parent / "runs" / "standard",
    )
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument(
        "--electronic-method", choices=("sa-casscf", "casci"), default="sa-casscf"
    )
    parser.add_argument("--training", type=int, default=160)
    parser.add_argument("--validation", type=int, default=40)
    parser.add_argument("--torsion-grid", type=int, default=13)
    parser.add_argument("--pyramid-grid", type=int, default=37)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--tt-rank", type=int, default=24)
    parser.add_argument("--tt-degree", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="populate/validate the external electronic database and stop before MACE",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="train and distill the electronic fields without a direct DVR grid",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="build the periodic direct ab initio LDR grid without fitting MACE",
    )
    parser.add_argument("--quiet-electronic", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    database = external_database_path(args.database)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    if args.torsion_grid < 5 or args.pyramid_grid < 5:
        raise ValueError("both grid sizes must be at least 5")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    axes = (
        ExponentialDVR(
            npts=args.torsion_grid,
            L=2.0 * np.pi,
            x0=np.pi / args.torsion_grid,
        ),
        SineDVR(*ETHYLENE_CI_BOUNDS[1], args.pyramid_grid),
    )
    grid = DVR.from_axes(axes, names=("torsion", "pyramidalization"))
    sampling_bounds = ETHYLENE_CI_BOUNDS
    coord = Coord(
        to_cartesian=ethylene_ci_geometry,
        bounds=ETHYLENE_CI_BOUNDS,
        periodic_axes=(0,),
    )
    driver = EthyleneCIElectronicDriver(
        basis=args.basis,
        method=args.electronic_method,
        nroots=2,
        verbose=0 if args.quiet_electronic else 1,
    )
    training = sobol_coordinates(
        args.training, args.seed, bounds=sampling_bounds, include_origin=True
    )
    validation = sobol_coordinates(
        args.validation, args.seed + 1, bounds=sampling_bounds
    )
    training_pairs = graph_pairs(training)
    validation_pairs = graph_pairs(validation)
    started = perf_counter()

    with AbInitioFit(
        driver,
        coord=coord,
        states=(0, 1),
        nroots=2,
        fit_options={"degrees": (args.tt_degree, args.tt_degree), "rank": args.tt_rank},
        database=database,
        protocol=driver.protocol,
        workers=args.workers,
        progress=not args.quiet_electronic,
        energy_shift=None,
    ) as sampler:
        training_fields = sampler.continuous_fields(training, training_pairs)
        validation_fields = sampler.continuous_fields(validation, validation_pairs)
        preparation_seconds = perf_counter() - started
        coverage_path = plot_database_coverage(
            output,
            {
                "coordinates": np.vstack((training, validation)),
                "hamiltonians": np.vstack(
                    (training_fields["hamiltonians"], validation_fields["hamiltonians"])
                ),
            },
        )
        base_summary = {
            "fidelity": driver.protocol,
            "database": str(database),
            "database_outside_repository": True,
            "training_geometries": len(training),
            "validation_geometries": len(validation),
            "preparation_seconds": preparation_seconds,
            "coverage_figure": str(coverage_path.with_suffix(".png")),
            "database_stats": sampler.database.stats,
        }
        if args.prepare_only:
            summary_path = output / "summary.json"
            summary_path.write_text(json.dumps(base_summary, indent=2) + "\n")
            print(json.dumps(base_summary, indent=2), flush=True)
            print(coverage_path.with_suffix(".png"), flush=True)
            return base_summary

        if args.direct_only:
            nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=driver.mol)
            started = perf_counter()
            direct = sampler.direct_product(
                grid,
                keo=nuclear_keo,
                workers=args.workers,
                progress=not args.quiet_electronic,
            )
            direct_seconds = perf_counter() - started
            energies = np.sort(np.asarray(direct.energies), axis=-1)
            seam_links = np.asarray(
                [
                    link
                    for (axis, left), link in direct.links.items()
                    if axis == 0 and left[0] == grid.shape[0] - 1
                ]
            )
            seam_singular_values = np.linalg.svd(seam_links, compute_uv=False)
            figure_path = plot_direct_periodic(output, grid.x, energies)
            archive_path = output / "ethylene_ci_2d_direct_periodic.npz"
            np.savez_compressed(
                archive_path,
                torsion=np.asarray(grid.x[0]),
                pyramidalization=np.asarray(grid.x[1]),
                energies=np.asarray(direct.energies),
                seam_links=seam_links,
            )
            summary = {
                **base_summary,
                "grid": list(grid.shape),
                "boundary_conditions": [
                    "periodic-fourier",
                    "nonperiodic-sine-dvr",
                ],
                "direct_build_seconds": direct_seconds,
                "periodic_seam_links": len(seam_links),
                "minimum_seam_link_singular_value": float(
                    np.min(seam_singular_values)
                ),
                "maximum_seam_link_singular_value": float(
                    np.max(seam_singular_values)
                ),
                "figure": str(figure_path.with_suffix(".png")),
                "archive": str(archive_path),
                "database_stats": sampler.database.stats,
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2), flush=True)
            print(figure_path.with_suffix(".png"), flush=True)
            return summary

        started = perf_counter()
        fit = MACE(
            grid.x,
            ETHYLENE_SPECIES,
            ethylene_ci_geometry,
            2,
            chart_features=True,
            chart_bounds=ETHYLENE_CI_BOUNDS,
            periodic_axes=(0,),
            geometry_units="bohr",
            channels=10,
            max_ell=2,
            interactions=2,
            correlation=2,
            radial_basis=8,
            radial_mlp=(48, 48),
            cutoff=4.5,
        ).fit_y(
            (training, training_fields["hamiltonians"]),
            training,
            training_pairs,
            training_fields["links"],
            feature_rank=8,
            feature_objective="links-only",
            ambient_representation="full",
            energy_representation="direct",
            hidden=(64, 64),
            epochs=args.epochs,
            learning_rate=1.5e-3,
            weight_decay=1.0e-8,
            energy_weight=80.0,
            seed=args.seed,
            distill=False,
        )
        neural_validation = model_errors(
            fit.neural_energy, fit.neural_feature, validation_fields
        )
        fit.distill_y(
            rank=args.tt_rank,
            degree=args.tt_degree,
            bases=("fourier", "chebyshev"),
            method="grid",
            seed=args.seed,
        )
        distilled_validation = model_errors(fit.energy, fit.feature, validation_fields)
        fit.save(output / "ethylene_ci_2d_mace_ftt.pt")
        fit_seconds = perf_counter() - started

        if args.fit_only:
            summary = {
                **base_summary,
                "grid": list(grid.shape),
                "mace_epochs": args.epochs,
                "neural_validation": neural_validation,
                "distilled_validation": distilled_validation,
                "fit_seconds": fit_seconds,
                "checkpoint": str(output / "ethylene_ci_2d_mace_ftt.pt"),
                "database_stats": sampler.database.stats,
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2), flush=True)
            return summary

        nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=driver.mol)
        started = perf_counter()
        direct = sampler.direct_product(
            grid,
            keo=nuclear_keo,
            workers=args.workers,
            progress=not args.quiet_electronic,
        )
        direct_seconds = perf_counter() - started
        tnldr = TNLDR(
            fit,
            grid=grid,
            coord=coord,
            keo=nuclear_keo,
            overlap_rank=24,
            operator_rank=96,
        ).build()

        coordinates = np.stack(
            np.meshgrid(*grid.x, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        exact = np.sort(np.asarray(direct.energies), axis=-1)
        fitted = np.linalg.eigvalsh(np.asarray(fit.energy.predict(coordinates))).reshape(
            *grid.shape, 2
        )
        point_error = np.max(np.abs(fitted - exact), axis=-1)
        feature = np.asarray(fit.feature.predict(coordinates))
        point_ids = {point: number for number, point in enumerate(np.ndindex(grid.shape))}
        link_midpoints = []
        link_errors = []
        for (axis, left), exact_link in sorted(direct.links.items()):
            right = list(left)
            right[axis] = (right[axis] + 1) % grid.shape[axis]
            right = tuple(right)
            predicted_link = (
                feature[point_ids[left]].conj().T @ feature[point_ids[right]]
            )
            left_coordinate = np.asarray([grid.x[i][left[i]] for i in range(2)])
            right_coordinate = np.asarray([grid.x[i][right[i]] for i in range(2)])
            midpoint = 0.5 * (left_coordinate + right_coordinate)
            if axis == 0 and left[axis] == grid.shape[axis] - 1:
                midpoint[axis] = (
                    left_coordinate[axis]
                    + 0.5 * (2.0 * np.pi + right_coordinate[axis] - left_coordinate[axis])
                )
                midpoint[axis] = (midpoint[axis] + np.pi) % (2.0 * np.pi) - np.pi
            link_midpoints.append(midpoint)
            predicted_spectrum = np.linalg.svd(predicted_link, compute_uv=False)
            exact_spectrum = np.linalg.svd(exact_link, compute_uv=False)
            link_errors.append(np.linalg.norm(predicted_spectrum - exact_spectrum))
        link_midpoints = np.asarray(link_midpoints)
        link_errors = np.asarray(link_errors)
        figure_path = plot_benchmark(
            output,
            grid.x,
            exact,
            point_error,
            link_midpoints,
            link_errors,
        )
        summary = {
            **base_summary,
            "grid": list(grid.shape),
            "mace_epochs": args.epochs,
            "neural_validation": neural_validation,
            "distilled_validation": distilled_validation,
            "maximum_direct_grid_pes_error_hartree": float(np.max(point_error)),
            "rms_direct_grid_pes_error_hartree": float(
                np.sqrt(np.mean((fitted - exact) ** 2))
            ),
            "maximum_direct_grid_link_spectrum_error": float(np.max(link_errors)),
            "rms_direct_grid_link_spectrum_error": float(
                np.sqrt(np.mean(link_errors**2))
            ),
            "fit_seconds": fit_seconds,
            "direct_build_seconds": direct_seconds,
            "tnldr_operator_bond_order": int(
                max(tnldr.hamiltonian.bond_orders(), default=1)
            ),
            "benchmark_figure": str(figure_path.with_suffix(".png")),
            "database_stats": sampler.database.stats,
        }
        np.savez_compressed(
            output / "ethylene_ci_2d_tnldr_benchmark.npz",
            axes=np.asarray(grid.x),
            exact_energies=exact,
            fitted_energies=fitted,
            point_errors=point_error,
            link_midpoints=link_midpoints,
            link_errors=link_errors,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        print(figure_path.with_suffix(".png"), flush=True)
        return summary


if __name__ == "__main__":
    main()
