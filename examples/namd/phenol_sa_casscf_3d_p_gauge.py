#!/usr/bin/env python3
"""Track three phenol states, build a graph P gauge, and refine the 3D cross."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phenol_sa_casscf_3d_data import (
    HARTREE_TO_EV,
    PHI_GRID,
    R_GRID,
    THETA_GRID,
    canonical_cross_points,
    phenol_geometry,
)
from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    phenol_sa6_protocol,
)
from pyqed.ldr.overlap import (
    procrustes,
    synchronize_link_gauge,
    track_states_graph,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def nearest_pairs(points):
    points = set(map(tuple, points))
    pairs = []
    shape = (len(R_GRID), len(PHI_GRID), len(THETA_GRID))
    for left in sorted(points):
        for axis, size in enumerate(shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            if right in points:
                pairs.append((left, right))
    return tuple(pairs)


def _traceless(matrix):
    matrix = np.asarray(matrix)
    return matrix - np.trace(matrix) * np.eye(matrix.shape[-1]) / matrix.shape[-1]


def graph_analysis(points, pairs, energies, overlaps):
    points = tuple(map(tuple, points))
    pairs = tuple((tuple(left), tuple(right)) for left, right in pairs)
    roots, selected = track_states_graph(
        points,
        pairs,
        overlaps,
        anchor=(0, 2, 1),
        states=(0, 1, 2),
    )
    singular = np.linalg.svd(selected, compute_uv=False)
    weights = np.maximum(singular[:, -1], 1.0e-4)
    gauges, p_links = synchronize_link_gauge(
        points,
        pairs,
        selected,
        anchor=(0, 2, 1),
        weights=weights,
    )
    rotations = procrustes(p_links)[0]
    rotation_defect = np.linalg.norm(
        rotations - np.eye(3), axis=(-2, -1)
    )
    p_hamiltonian = np.asarray(
        [
            gauge.conj().T @ np.diag(energies[point][root]) @ gauge
            for point, root, gauge in zip(points, roots, gauges)
        ]
    )
    return {
        "points": points,
        "pairs": pairs,
        "roots": roots,
        "selected_links": selected,
        "selected_singular": singular,
        "gauges": gauges,
        "p_links": p_links,
        "p_hamiltonian": p_hamiltonian,
        "rotation_defect": rotation_defect,
    }


def interaction_acquisition(seed, count):
    points = seed["points"]
    hamiltonian = dict(zip(points, seed["p_hamiltonian"]))
    candidates = []
    for radial in range(len(R_GRID)):
        center = hamiltonian[(radial, 2, 1)]
        for torsion in (3, 4):
            torsional = _traceless(hamiltonian[(radial, torsion, 1)] - center)
            for bend in (0, 2):
                angular = _traceless(hamiltonian[(radial, 2, bend)] - center)
                risk = float(np.linalg.norm(torsional) * np.linalg.norm(angular))
                candidates.append(((radial, torsion, bend), risk))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return tuple(candidates[: min(int(count), len(candidates))])


def interaction_diagnostics(analysis, selected):
    hamiltonian = dict(zip(analysis["points"], analysis["p_hamiltonian"]))
    values = []
    for point, risk in selected:
        radial, torsion, bend = point
        additive = (
            hamiltonian[(radial, torsion, 1)]
            + hamiltonian[(radial, 2, bend)]
            - hamiltonian[(radial, 2, 1)]
        )
        residual = hamiltonian[point] - additive
        matrix_error = float(np.linalg.norm(_traceless(residual)) * HARTREE_TO_EV)
        spectral_error = float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(hamiltonian[point])
                    - np.linalg.eigvalsh(additive)
                )
            )
            * HARTREE_TO_EV
        )
        values.append(
            {
                "point": point,
                "risk": risk,
                "matrix_error_ev": matrix_error,
                "spectral_error_ev": spectral_error,
            }
        )
    return values


def _plot_surfaces(output, analysis, interaction):
    point_ids = {point: index for index, point in enumerate(analysis["points"])}
    roots = analysis["roots"]
    hamiltonian = analysis["p_hamiltonian"]
    reference = float(np.min(np.linalg.eigvalsh(hamiltonian[point_ids[(0, 2, 1)]])))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 3))
    figure, panels = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for torsion, marker in zip((2, 3, 4), ("o", "s", "^")):
        for state, color in enumerate(colors):
            values = []
            for radial in range(len(R_GRID)):
                index = point_ids[(radial, torsion, 1)]
                values.append(np.linalg.eigvalsh(hamiltonian[index])[state])
            panels[0].plot(
                R_GRID,
                (np.asarray(values) - reference) * HARTREE_TO_EV,
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.0,
                label=f"P{state}" if torsion == 2 else None,
            )
    panels[0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="tracked energy (eV)",
        title=r"$\theta=108.8^\circ$",
    )
    panels[0].legend(fontsize=7.5)

    for bend, marker in zip((0, 1, 2), ("v", "o", "^")):
        for state, color in enumerate(colors):
            values = []
            for radial in range(len(R_GRID)):
                index = point_ids[(radial, 2, bend)]
                values.append(np.linalg.eigvalsh(hamiltonian[index])[state])
            panels[1].plot(
                R_GRID,
                (np.asarray(values) - reference) * HARTREE_TO_EV,
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.0,
            )
    panels[1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="tracked energy (eV)",
        title=r"$\phi=0$",
    )

    labels = [
        f"{R_GRID[item['point'][0]]:.2f}, {PHI_GRID[item['point'][1]]:.1f}, "
        f"{np.rad2deg(THETA_GRID[item['point'][2]]):.0f}\N{DEGREE SIGN}"
        for item in interaction
    ]
    positions = np.arange(len(interaction))
    panels[2].bar(
        positions - 0.18,
        [item["matrix_error_ev"] for item in interaction],
        width=0.36,
        label=r"$\|\Delta H_P^0\|_F$",
    )
    panels[2].bar(
        positions + 0.18,
        [item["spectral_error_ev"] for item in interaction],
        width=0.36,
        label="maximum energy error",
    )
    panels[2].set_xticks(positions, labels, rotation=55, ha="right", fontsize=6.5)
    panels[2].set(
        ylabel="nonadditive interaction (eV)",
        title="Adaptive validation points",
    )
    panels[2].legend(fontsize=6.5)
    for panel in panels:
        panel.grid(alpha=0.2)
    figure.suptitle("Phenol three-state graph-tracked P gauge")
    png = output / "phenol_sa6_3d_p_gauge_surfaces.png"
    pdf = output / "phenol_sa6_3d_p_gauge_surfaces.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _plot_graph_diagnostics(output, analysis):
    pairs = analysis["pairs"]
    axes = np.asarray(
        [np.flatnonzero(np.asarray(right) - np.asarray(left))[0] for left, right in pairs]
    )
    colors = ("#3366a8", "#c15c2d", "#2f8b62")
    labels = (r"$R_{OH}$", r"$\phi$", r"$\theta$")
    figure, panels = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
    for axis in range(3):
        mask = axes == axis
        panels[0].semilogy(
            np.flatnonzero(mask),
            analysis["selected_singular"][mask, -1],
            "o",
            ms=3.5,
            color=colors[axis],
            label=labels[axis],
        )
        panels[1].plot(
            np.flatnonzero(mask),
            analysis["rotation_defect"][mask],
            "o",
            ms=3.5,
            color=colors[axis],
            label=labels[axis],
        )
    panels[0].axhline(0.5, color="0.35", ls=":", lw=1.0)
    panels[0].set(
        xlabel="sampled-graph edge",
        ylabel=r"minimum $\sigma(S_{ij}^{(3)})$",
        title="Tracked subspace continuity",
    )
    panels[1].set(
        xlabel="sampled-graph edge",
        ylabel=r"$\|U_{ij}^{P}-I\|_F$",
        title="Globally synchronized gauge",
    )
    panels[0].legend(fontsize=7.5)
    panels[1].legend(fontsize=7.5)

    point_ids = {point: index for index, point in enumerate(analysis["points"])}
    for channel, marker in enumerate(("o", "s", "^")):
        for torsion in (2, 3, 4):
            values = [
                analysis["roots"][point_ids[(radial, torsion, 1)], channel]
                for radial in range(len(R_GRID))
            ]
            panels[2].plot(
                R_GRID,
                values,
                marker=marker,
                ms=3.0,
                lw=0.9,
                alpha=0.75,
                label=f"channel {channel}" if torsion == 2 else None,
            )
    panels[2].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="adiabatic root index",
        title="Discrete root tracking",
        yticks=range(6),
    )
    panels[2].legend(fontsize=7.0)
    for panel in panels:
        panel.grid(alpha=0.2)
    png = output / "phenol_sa6_3d_p_gauge_diagnostics.png"
    pdf = output / "phenol_sa6_3d_p_gauge_diagnostics.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_3d_cross_20260820/"
            "phenol_sa6_3d_cross_data.npz"
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_production_qualification_20260820/"
            "electronic.sqlite"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_p_gauge_20260820"),
    )
    parser.add_argument("--refine", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--interaction-tolerance-ev", type=float, default=0.02)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.seed, allow_pickle=False) as archive:
        seed_points = tuple(map(tuple, archive["points"]))
        seed_pairs = tuple(
            (tuple(left), tuple(right)) for left, right in archive["pairs"]
        )
        seed_energies = np.asarray(archive["energies"])
        seed_overlaps = np.asarray(archive["overlaps"])
    seed = graph_analysis(seed_points, seed_pairs, seed_energies, seed_overlaps)
    selected = interaction_acquisition(seed, args.refine)
    canonical = tuple(
        dict.fromkeys((*canonical_cross_points(), *(point for point, _risk in selected)))
    )

    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database, protocol, verbose=0 if args.quiet else 1
    )

    def progress(index, stats):
        print(
            f"[P-gauge] built R={R_GRID[index[0]]:.2f} A, "
            f"phi={PHI_GRID[index[1]]:+.2f}, "
            f"theta={np.rad2deg(THETA_GRID[index[2]]):.1f} deg "
            f"({stats['built']} new)",
            flush=True,
        )

    with AbInitioFit(
        (R_GRID, PHI_GRID, THETA_GRID),
        6,
        electronic=provider,
        geometry=phenol_geometry,
        symmetry=PhenolReflectionSymmetry(torsion_axis=1),
        database=database,
        protocol=protocol,
        run_id="phenol-sa6-3d-p-gauge-refine-v1",
        run_metadata={
            "purpose": "graph-tracked three-state P gauge and interaction refinement",
            "workers": args.workers,
            "refinement_geometries": len(selected),
        },
        frame=lambda record: record,
        energies=lambda record: record["energies"],
        overlap=overlap,
        overlap_protocol=overlap.protocol,
        anchor=(0, 2, 1),
        workers=args.workers,
        progress=progress,
        energy_shift=None,
    ) as fit:
        points = fit.expand_points(canonical)
        fit.frames.get_many(points)
        records = {point: fit.frames.get(point) for point in points}
        energies = np.full((*fit.shape, 6), np.nan)
        record_ids = np.full(fit.shape, "", dtype="U64")
        for point, record in records.items():
            energies[point] = record["energies"]
            record_ids[point] = fit.frames.record_id(point)
        pairs = nearest_pairs(points)
        raw_overlaps = fit.oracle.raw_overlap_many(pairs)
        analysis = graph_analysis(points, pairs, energies, raw_overlaps)
        interaction = interaction_diagnostics(analysis, selected)

        surface_png, surface_pdf = _plot_surfaces(args.output, analysis, interaction)
        diagnostic_png, diagnostic_pdf = _plot_graph_diagnostics(args.output, analysis)
        data_path = args.output / "phenol_sa6_3d_p_gauge_data.npz"
        np.savez_compressed(
            data_path,
            r_oh=R_GRID,
            phi=PHI_GRID,
            theta=THETA_GRID,
            points=np.asarray(analysis["points"]),
            pairs=np.asarray(analysis["pairs"]),
            energies=energies,
            record_ids=record_ids,
            raw_overlaps=raw_overlaps,
            root_indices=analysis["roots"],
            selected_overlaps=analysis["selected_links"],
            selected_overlap_singular_values=analysis["selected_singular"],
            gauges=analysis["gauges"],
            p_hamiltonian=analysis["p_hamiltonian"],
            p_links=analysis["p_links"],
            p_link_rotation_defect=analysis["rotation_defect"],
            refinement_points=np.asarray([point for point, _risk in selected]),
            refinement_risk=np.asarray([risk for _point, risk in selected]),
        )
        maximum_interaction = max(
            (item["matrix_error_ev"] for item in interaction), default=0.0
        )
        full_grid = len(points) == int(np.prod(fit.shape))
        gauge_unitarity = float(
            np.max(
                np.linalg.norm(
                    analysis["gauges"].conj().swapaxes(-1, -2)
                    @ analysis["gauges"]
                    - np.eye(3),
                    axis=(-2, -1),
                )
            )
        )
        gates = {
            "all_records_orbitally_relaxed": all(
                bool(record["orbital_relaxed"]) for record in records.values()
            ),
            "all_records_singlets": bool(
                max(np.max(np.abs(record["spins"])) for record in records.values())
                <= 1.0e-5
            ),
            "tracked_subspace_continuous": bool(
                np.min(analysis["selected_singular"][:, -1]) >= 0.5
            ),
            "gauge_unitary": gauge_unitarity <= 1.0e-10,
            "no_active_claims": database.stats["claims"] == 0,
        }
        summary = {
            "passed": all(gates.values()),
            "gates": gates,
            "workers": args.workers,
            "geometries": len(points),
            "canonical_geometries": len(canonical),
            "graph_edges": len(pairs),
            "new_records": fit.frames.stats["built"],
            "reused_records": fit.frames.stats["database_hits"],
            "minimum_selected_overlap_singular_value": float(
                np.min(analysis["selected_singular"][:, -1])
            ),
            "median_selected_overlap_singular_value": float(
                np.median(analysis["selected_singular"][:, -1])
            ),
            "maximum_p_link_rotation_defect": float(
                np.max(analysis["rotation_defect"])
            ),
            "rms_p_link_rotation_defect": float(
                np.sqrt(np.mean(analysis["rotation_defect"] ** 2))
            ),
            "maximum_gauge_unitarity_defect": gauge_unitarity,
            "interaction_tolerance_ev": args.interaction_tolerance_ev,
            "maximum_nonadditive_interaction_ev": maximum_interaction,
            "cross_additivity_rejected": maximum_interaction > args.interaction_tolerance_ev,
            "full_discrete_grid": full_grid,
            "needs_more_refinement": (
                not full_grid
                and maximum_interaction > args.interaction_tolerance_ev
            ),
            "refinement": interaction,
            "database": str(args.database),
            "database_stats": database.stats,
            "frame_stats": fit.frames.stats,
            "data": str(data_path),
            "figures": {
                "surfaces": str(surface_png),
                "surfaces_pdf": str(surface_pdf),
                "diagnostics": str(diagnostic_png),
                "diagnostics_pdf": str(diagnostic_pdf),
            },
        }
        database.update_run(
            fit.run_id, "sampled" if summary["passed"] else "failed"
        )
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
        print(json.dumps(_jsonable(summary), indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
