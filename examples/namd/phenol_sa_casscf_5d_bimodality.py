#!/usr/bin/env python3
"""Validate the fitted phenol torsion--Q16a double well with SA(6)-CASSCF."""

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
from scipy.spatial.distance import cdist

from examples.namd.phenol_sa_casscf_5d_pilot import (
    PARITIES,
    PhenolGeometry,
    diagnostic_overlap_many,
    diagnostic_records,
)
from examples.namd.phenol_sa_casscf_paths import DEFAULT_PHENOL_SA6_DATABASE
from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    phenol_sa6_protocol,
)
from pyqed.ldr.overlap import procrustes, track_states_graph
from pyqed.ml.corrections import ReflectionScalarMLP
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.units import au2ev


HARTREE_TO_EV = au2ev
R_VALUE = 1.0
THETA_VALUE = np.deg2rad(108.8)
Q8A_VALUE = 0.0
GRIDS = (
    np.asarray((0.95, R_VALUE, 1.05)),
    np.asarray((-0.60, -0.40, -0.20, 0.0, 0.20, 0.40, 0.60)),
    np.deg2rad(np.asarray((104.0, 108.8, 114.0))),
    np.asarray((-0.70, -0.55, -0.30, 0.0, 0.30, 0.55, 0.70)),
    np.asarray((-0.10, Q8A_VALUE, 0.10)),
)
SCALES = np.asarray((0.25, 0.20, np.deg2rad(5.0), 0.25, 0.10))
CANONICAL_COORDINATES = (
    (R_VALUE, 0.0, THETA_VALUE, 0.0, Q8A_VALUE),
    (R_VALUE, 0.20, THETA_VALUE, -0.30, Q8A_VALUE),
    (R_VALUE, 0.40, THETA_VALUE, -0.55, Q8A_VALUE),
    (R_VALUE, 0.60, THETA_VALUE, -0.70, Q8A_VALUE),
    (R_VALUE, 0.40, THETA_VALUE, 0.0, Q8A_VALUE),
    (R_VALUE, 0.0, THETA_VALUE, 0.55, Q8A_VALUE),
    (R_VALUE, 0.40, THETA_VALUE, 0.55, Q8A_VALUE),
)
LABELS = (
    "center",
    "valley 0.2",
    "valley 0.4",
    "valley 0.6",
    "torsion only",
    "Q16a only",
    "same sign",
)


def _grid_index(coordinate):
    index = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(GRIDS, coordinate)
    )
    rebuilt = np.asarray([grid[item] for grid, item in zip(GRIDS, index)])
    if not np.allclose(rebuilt, coordinate, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"validation coordinate {coordinate!r} is absent from the grid")
    return index


def validation_design():
    return tuple(_grid_index(coordinate) for coordinate in CANONICAL_COORDINATES)


def _reflection_defect(chart, coordinates=CANONICAL_COORDINATES):
    reflection = np.diag((1.0, 1.0, -1.0))
    defect = 0.0
    for coordinate in coordinates:
        coordinate = np.asarray(coordinate)
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


def _scaled_distance(left, right, scales):
    difference = np.asarray(left)[:, None, :] - np.asarray(right)[None, :, :]
    difference[..., 1] = (difference[..., 1] + np.pi) % (2.0 * np.pi) - np.pi
    return np.linalg.norm(difference / np.asarray(scales), axis=-1)


def _fixed_gauge_extension(points, pairs, links, fixed, weights, sweeps=200):
    """Synchronize new gauges while leaving the production gauges unchanged."""

    point_ids = {point: index for index, point in enumerate(points)}
    rotations = procrustes(np.asarray(links))[0]
    weights = np.asarray(weights, dtype=float)
    gauges = np.tile(np.eye(links.shape[-1], dtype=complex), (len(points), 1, 1))
    fixed_ids = set()
    for point, gauge in fixed.items():
        gauges[point_ids[point]] = gauge
        fixed_ids.add(point_ids[point])
    adjacency = [[] for _ in points]
    for edge, (left, right) in enumerate(pairs):
        left_id, right_id = point_ids[left], point_ids[right]
        adjacency[left_id].append((right_id, edge, False))
        adjacency[right_id].append((left_id, edge, True))
    for _ in range(int(sweeps)):
        maximum_change = 0.0
        for point in range(len(points)):
            if point in fixed_ids:
                continue
            mean = np.zeros_like(gauges[point])
            for neighbor, edge, reverse in adjacency[point]:
                rotation = rotations[edge].conj().T if reverse else rotations[edge]
                mean += weights[edge] * rotation @ gauges[neighbor]
            updated = procrustes(mean)[0]
            maximum_change = max(maximum_change, np.linalg.norm(updated - gauges[point]))
            gauges[point] = updated
        if maximum_change < 1.0e-12:
            break
    return gauges


def _database_records(database, ids, coordinates, chart, symmetry):
    entries = {entry["id"]: entry for entry in database.entries()}
    records = []
    for record_id, coordinate in zip(ids, coordinates):
        entry = entries[str(record_id)]
        record = database.get(entry["specification"])
        image = symmetry.resolve(coordinate)
        record = symmetry.transform_record(
            record,
            image,
            representative_geometry=record["geometry"],
            requested_geometry=chart.geometry(coordinate),
            protocol=entry["specification"]["protocol"],
        )
        records.append(record)
    return records


def _graph_edges(base_coordinates, new_coordinates, base_neighbors=4, new_neighbors=3):
    base_distance = _scaled_distance(new_coordinates, base_coordinates, SCALES)
    edges = set()
    selected_base = set()
    for new_id, row in enumerate(base_distance):
        for base_id in np.argsort(row)[: int(base_neighbors)]:
            base_id = int(base_id)
            selected_base.add(base_id)
            edges.add(((0, base_id), (1, new_id)))
    new_distance = _scaled_distance(new_coordinates, new_coordinates, SCALES)
    np.fill_diagonal(new_distance, np.inf)
    for left, row in enumerate(new_distance):
        for right in np.argsort(row)[: int(new_neighbors)]:
            pair = tuple(sorted(((1, int(left)), (1, int(right)))))
            edges.add(pair)
    return tuple(sorted(selected_base)), tuple(sorted(edges))


def _plot(output, coordinates, canonical_ids, exact, predicted, uncertainty, model):
    phi = np.linspace(-0.72, 0.72, 181)
    q16a = np.linspace(-0.82, 0.82, 181)
    pp, qq = np.meshgrid(phi, q16a, indexing="ij")
    query = np.column_stack(
        (
            np.full(pp.size, R_VALUE),
            pp.ravel(),
            np.full(pp.size, THETA_VALUE),
            qq.ravel(),
            np.full(pp.size, Q8A_VALUE),
        )
    )
    surface = model.predict(query).reshape(pp.shape)
    reference = min(float(np.min(surface)), float(np.min(exact)))
    surface = (surface - reference) * HARTREE_TO_EV
    exact_ev = (exact - reference) * HARTREE_TO_EV
    predicted_ev = (predicted - reference) * HARTREE_TO_EV

    figure, panels = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    contour = panels[0].contourf(phi, q16a, surface.T, levels=24, cmap="viridis")
    panels[0].scatter(
        coordinates[:, 1], coordinates[:, 3], c=exact_ev, cmap="viridis",
        edgecolors="white", linewidths=0.8, s=58,
    )
    panels[0].set(
        xlabel=r"$\phi$ (rad)", ylabel=r"$Q_{16a}$",
        title=r"MLP surface + exact $H_{11}^{P}$ points",
    )
    figure.colorbar(contour, ax=panels[0], label="energy above minimum (eV)")

    x = np.arange(len(canonical_ids))
    width = 0.38
    panels[1].bar(x - width / 2, exact_ev[canonical_ids], width, label="SA(6) + overlaps")
    panels[1].bar(x + width / 2, predicted_ev[canonical_ids], width, label="scalar fit")
    panels[1].set_xticks(x, LABELS, rotation=38, ha="right")
    panels[1].set(ylabel="energy above minimum (eV)", title="Seven canonical checks")
    panels[1].legend(frameon=False)

    error = (predicted - exact) * HARTREE_TO_EV
    panels[2].axhline(0.0, color="0.35", linewidth=1.0)
    panels[2].errorbar(
        x, error[canonical_ids], yerr=uncertainty[canonical_ids], fmt="o",
        capsize=3, label=r"fit error $\pm$ gauge-path spread",
    )
    panels[2].set_xticks(x, LABELS, rotation=38, ha="right")
    panels[2].set(ylabel="fit $-$ ab initio (eV)", title="Interpolation and transport check")
    panels[2].legend(frameon=False)

    png = output / "phenol_sa6_5d_bimodality_validation.png"
    pdf = output / "phenol_sa6_5d_bimodality_validation.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_PHENOL_SA6_DATABASE)
    parser.add_argument(
        "--base-data", type=Path,
        default=Path("dataset/phenol_5d_production/inputs/periodic_torsion/phenol_sa6_5d_p_gauge.npz"),
    )
    parser.add_argument(
        "--scalar-potential", type=Path,
        default=Path("dataset/phenol_5d_production/fits/scalar_parent_periodic_h3/phenol_sa6_5d_scalar_parent.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("dataset/phenol_5d_production/validation/bimodality"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--diagnostic-workers", type=int, default=6)
    parser.add_argument("--overlap-workers", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with np.load(args.base_data, allow_pickle=False) as saved:
        base = {name: np.asarray(saved[name]) for name in saved.files}
    chart = PhenolReactiveChart(modes=base["modes"])
    reflection_defect = max(
        _reflection_defect(chart),
        _reflection_defect(chart, base["coordinates"]),
    )
    symmetry = PhenolReflectionSymmetry(
        torsion_axis=1,
        odd_axes=(1, 3),
        tolerance=max(1.0e-10, 1.05 * reflection_defect),
    )
    protocol = phenol_sa6_protocol()
    model = ReflectionScalarMLP.load(args.scalar_potential)
    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database, protocol, coordinate_scale=SCALES, verbose=0 if args.quiet else 1
    )
    overlap = PhenolCASSCFOverlap()
    canonical = validation_design()
    run_id = "phenol-sa6-5d-bimodality-validation-v1-n7"

    def progress(_index, stats):
        print(f"[bimodality SA(6)] {stats['built']}/{len(canonical)} calculated", flush=True)

    try:
        with AbInitioFit(
            GRIDS,
            6,
            electronic=provider,
            geometry=PhenolGeometry(chart.modes),
            symmetry=symmetry,
            database=database,
            protocol=protocol,
            run_id=run_id,
            run_metadata={
                "purpose": "direct validation of the fitted torsion-Q16a double well",
                "canonical_coordinates": CANONICAL_COORDINATES,
                "maximum_chart_reflection_defect_angstrom": reflection_defect,
                "workers": int(args.workers),
            },
            frame=lambda record: record,
            energies=lambda record: record["energies"],
            overlap=overlap,
            overlap_protocol=overlap.protocol,
            anchor=canonical[0],
            workers=args.workers,
            progress=progress,
            energy_shift=None,
        ) as fit:
            points = fit.expand_points(canonical)
            production = fit.frames.get_many(points)
            fit_stats = dict(fit.frames.stats)
            diagnostic, diagnostic_ids, diagnostic_stats = diagnostic_records(
                fit,
                points,
                canonical,
                production,
                database,
                protocol,
                symmetry,
                nroots=10,
                workers=args.diagnostic_workers,
                run_id=run_id,
            )
            coordinates = np.asarray([fit.coordinates(point) for point in points])

            selected_base, graph_pairs = _graph_edges(base["coordinates"], coordinates)
            base_keys = tuple((0, index) for index in selected_base)
            new_keys = tuple((1, index) for index in range(len(points)))
            graph_points = (*base_keys, *new_keys)
            coordinate_map = {
                **{(0, index): base["coordinates"][index] for index in selected_base},
                **{(1, index): coordinates[index] for index in range(len(points))},
            }
            base_records = _database_records(
                database,
                base["diagnostic_record_ids"][list(selected_base)],
                base["coordinates"][list(selected_base)],
                chart,
                symmetry,
            )
            record_map = {
                **dict(zip(base_keys, base_records)),
                **dict(zip(new_keys, diagnostic)),
            }
            id_map = {
                **dict(zip(base_keys, base["diagnostic_record_ids"][list(selected_base)])),
                **dict(zip(new_keys, diagnostic_ids)),
            }
            raw, overlap_stats = diagnostic_overlap_many(
                database,
                graph_points,
                graph_pairs,
                [record_map[point] for point in graph_points],
                [id_map[point] for point in graph_points],
                symmetry,
                overlap,
                workers=args.overlap_workers,
                coordinate_of=coordinate_map.__getitem__,
            )
            fixed_roots = {
                (0, index): base["root_indices"][index] for index in selected_base
            }
            anchor = base_keys[0]
            roots, _ = track_states_graph(
                graph_points,
                graph_pairs,
                raw,
                anchor=anchor,
                states=fixed_roots[anchor],
                fixed=fixed_roots,
            )
            graph_ids = {point: index for index, point in enumerate(graph_points)}
            point_ids = {point: index for index, point in enumerate(points)}
            for point in points:
                representative = fit.frames.representative(point)
                roots[graph_ids[(1, point_ids[point])]] = roots[
                    graph_ids[(1, point_ids[representative])]
                ]
            selected = np.asarray(
                [
                    block[np.ix_(roots[graph_ids[left]], roots[graph_ids[right]])]
                    for (left, right), block in zip(graph_pairs, raw)
                ]
            )
            singular = np.linalg.svd(selected, compute_uv=False)
            fixed_gauges = {
                (0, index): base["gauges"][index] for index in selected_base
            }
            gauges = _fixed_gauge_extension(
                graph_points,
                graph_pairs,
                selected,
                fixed_gauges,
                np.maximum(singular[:, -1], 1.0e-4),
            )
            reflection = base["reflection"]
            for point in points:
                representative = fit.frames.representative(point)
                if point != representative:
                    gauges[graph_ids[(1, point_ids[point])]] = (
                        gauges[graph_ids[(1, point_ids[representative])]] @ reflection
                    )

            new_roots = np.asarray([roots[graph_ids[key]] for key in new_keys])
            new_gauges = np.asarray([gauges[graph_ids[key]] for key in new_keys])
            diagnostic_energy = np.asarray([record["energies"] for record in diagnostic])
            p_hamiltonian = np.asarray(
                [
                    gauge.conj().T @ np.diag(energy[root]) @ gauge
                    for gauge, energy, root in zip(new_gauges, diagnostic_energy, new_roots)
                ]
            )
            exact = p_hamiltonian[:, 1, 1].real
            predicted = np.asarray(model.predict(coordinates), dtype=float)

            path_values = [[] for _ in points]
            for edge, (left, right) in enumerate(graph_pairs):
                if left[0] == right[0]:
                    continue
                base_key, new_key = (left, right) if left[0] == 0 else (right, left)
                block = selected[edge] if left[0] == 0 else selected[edge].conj().T
                rotation = procrustes(fixed_gauges[base_key].conj().T @ block)[0]
                gauge = rotation.conj().T
                new_id = new_key[1]
                energy = diagnostic_energy[new_id][new_roots[new_id]]
                value = (gauge.conj().T @ np.diag(energy) @ gauge)[1, 1].real
                path_values[new_id].append(value)
            uncertainty = np.asarray(
                [
                    0.5 * (max(values) - min(values)) * HARTREE_TO_EV
                    if values else np.nan
                    for values in path_values
                ]
            )
            canonical_ids = np.asarray([point_ids[point] for point in canonical])
            center = canonical_ids[0]
            valley_candidates = canonical_ids[1:4]
            valley = valley_candidates[int(np.argmin(exact[valley_candidates]))]
            same_sign = canonical_ids[6]
            error_ev = (predicted - exact) * HARTREE_TO_EV
            center_barrier = (exact[center] - exact[valley]) * HARTREE_TO_EV
            same_sign_penalty = (exact[same_sign] - exact[valley]) * HARTREE_TO_EV
            summary = {
                "run_id": run_id,
                "canonical_geometries": len(canonical),
                "effective_geometries": len(points),
                "new_sa6_records_this_invocation": int(fit_stats["built"]),
                "reused_sa6_records_this_invocation": int(fit_stats["database_hits"]),
                "diagnostic_records": diagnostic_stats,
                "overlaps": overlap_stats,
                "minimum_selected_overlap_singular_value": float(np.min(singular[:, -1])),
                "fit_rms_ev": float(np.sqrt(np.mean(error_ev**2))),
                "fit_maximum_ev": float(np.max(np.abs(error_ev))),
                "maximum_gauge_path_spread_ev": float(np.nanmax(uncertainty)),
                "center_minus_correlated_valley_ev": float(center_barrier),
                "correlated_valley_label": LABELS[
                    int(np.flatnonzero(canonical_ids == valley)[0])
                ],
                "fit_center_minus_correlated_valley_ev": float(
                    (predicted[center] - np.min(predicted[valley_candidates]))
                    * HARTREE_TO_EV
                ),
                "same_sign_minus_correlated_valley_ev": float(same_sign_penalty),
                "double_well_confirmed": bool(center_barrier > 0.02 and same_sign_penalty > 0.02),
                "points": [
                    {
                        "label": LABELS[number],
                        "coordinates": list(map(float, CANONICAL_COORDINATES[number])),
                        "p_h11_hartree": float(exact[index]),
                        "fit_h11_hartree": float(predicted[index]),
                        "fit_error_ev": float(error_ev[index]),
                        "gauge_path_spread_ev": float(uncertainty[index]),
                        "tracked_roots": list(map(int, new_roots[index])),
                    }
                    for number, index in enumerate(canonical_ids)
                ],
                "all_records_orbitally_relaxed": all(
                    bool(record["orbital_relaxed"]) for record in production
                ),
                "all_diagnostic_roots_singlets": bool(
                    max(np.max(np.abs(record["spins"])) for record in diagnostic) < 1.0e-5
                ),
                "maximum_sa6_diagnostic_agreement_hartree": float(
                    max(float(record["sa_energy_agreement"]) for record in diagnostic)
                ),
                "diagnostic_first6_reproduce_sa6": bool(
                    max(float(record["sa_energy_agreement"]) for record in diagnostic) <= 1.0e-6
                ),
                "wall_seconds": float(time.perf_counter() - started),
            }
            data_path = args.output / "phenol_sa6_5d_bimodality_validation.npz"
            np.savez_compressed(
                data_path,
                coordinates=coordinates,
                canonical_indices=canonical_ids,
                sa_energies=np.asarray([record["energies"] for record in production]),
                diagnostic_energies=diagnostic_energy,
                root_indices=new_roots,
                gauges=new_gauges,
                p_hamiltonian=p_hamiltonian,
                fit_h11=predicted,
                fit_error_ev=error_ev,
                gauge_path_spread_ev=uncertainty,
                record_ids=np.asarray([fit.frames.record_id(point) for point in points]),
                diagnostic_record_ids=diagnostic_ids,
            )
            png, pdf = _plot(
                args.output, coordinates, canonical_ids, exact, predicted, uncertainty, model
            )
            summary.update(
                {
                    "data": str(data_path.resolve()),
                    "figure_png": str(png.resolve()),
                    "figure_pdf": str(pdf.resolve()),
                }
            )
            summary_path = args.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        provider.close()
        database.close()


if __name__ == "__main__":
    main()
