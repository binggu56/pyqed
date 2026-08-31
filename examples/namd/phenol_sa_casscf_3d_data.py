#!/usr/bin/env python3
"""Build a reusable three-dimensional phenol SA-CASSCF cross dataset."""

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

from examples.namd.phenol_sa_casscf_paths import DEFAULT_PHENOL_SA6_DATABASE

from pyqed.units import au2ev
from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    phenol_sa6_protocol,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
R_GRID = np.asarray((0.95, 1.15, 1.30, 1.55, 1.85))
PHI_GRID = np.asarray((-0.40, -0.20, 0.0, 0.20, 0.40))
THETA_GRID = np.deg2rad(np.asarray((104.0, 108.8, 114.0)))


def phenol_geometry(coordinates):
    chart = PhenolReactiveChart()
    coordinate = np.array(chart.equilibrium, copy=True)
    coordinate[:3] = coordinates
    return chart.geometry(coordinate)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_cross_points():
    points = []
    for radial in range(len(R_GRID)):
        for torsion in range(len(PHI_GRID) // 2, len(PHI_GRID)):
            points.append((radial, torsion, 1))
        for bend in range(len(THETA_GRID)):
            points.append((radial, 2, bend))
    return tuple(dict.fromkeys(points))


def cross_pairs():
    pairs = []
    for radial in range(len(R_GRID)):
        for torsion in range(len(PHI_GRID) - 1):
            pairs.append(
                ((radial, torsion, 1), (radial, torsion + 1, 1))
            )
    for radial in range(len(R_GRID) - 1):
        for torsion in range(len(PHI_GRID)):
            pairs.append(
                ((radial, torsion, 1), (radial + 1, torsion, 1))
            )
    for radial in range(len(R_GRID)):
        for bend in range(len(THETA_GRID) - 1):
            pairs.append(((radial, 2, bend), (radial, 2, bend + 1)))
    for radial in range(len(R_GRID) - 1):
        for bend in range(len(THETA_GRID)):
            pairs.append(((radial, 2, bend), (radial + 1, 2, bend)))
    return tuple(dict.fromkeys(pairs))


def _macroiterations(record):
    return int(len(np.asarray(record.get("macro_history", ()))))


def _plot_energies(output, records):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    anchor = float(records[(0, 2, 1)]["energies"][0])
    figure, panels = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    for torsion in (2, 3, 4):
        energies = np.asarray(
            [records[(radial, torsion, 1)]["energies"] for radial in range(5)]
        )
        for state, color in enumerate(colors):
            panels[0].plot(
                R_GRID,
                (energies[:, state] - anchor) * HARTREE_TO_EV,
                marker={2: "o", 3: "s", 4: "^"}[torsion],
                color=color,
                lw=1.0,
                ms=3.5,
                alpha=0.85,
                label=(
                    f"S{state}" if torsion == 2 else None
                ),
            )
    panels[0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"$E_i-E_0(0.95\,\AA,0,108.8^\circ)$ (eV)",
        title=r"Equilibrium bend: $\phi=0,0.2,0.4$ rad",
    )
    panels[0].legend(ncol=2, fontsize=7.5)

    for bend in range(3):
        energies = np.asarray(
            [records[(radial, 2, bend)]["energies"] for radial in range(5)]
        )
        for state, color in enumerate(colors):
            panels[1].plot(
                R_GRID,
                (energies[:, state] - anchor) * HARTREE_TO_EV,
                marker=("v", "o", "^")[bend],
                color=color,
                lw=1.0,
                ms=3.5,
                alpha=0.85,
            )
    panels[1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"relative energy (eV)",
        title=r"Planar torsion: $\theta=104,108.8,114^\circ$",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    figure.suptitle("Phenol SA(6)-CASSCF three-dimensional cross data")
    png = output / "phenol_sa6_3d_cross_energies.png"
    pdf = output / "phenol_sa6_3d_cross_energies.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _plot_diagnostics(output, points, records, pairs, singular_values):
    macro = np.asarray([_macroiterations(records[point]) for point in points])
    wall = np.asarray([float(records[point]["wall_seconds"]) for point in points])
    edge_minimum = singular_values[:, -1]
    figure, panels = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    panels[0].hist(macro, bins=np.arange(macro.min(), macro.max() + 2) - 0.5, color="#2673b8")
    panels[0].set(xlabel="macroiterations", ylabel="records", title="Orbital convergence")
    panels[1].hist(wall, bins=min(12, max(4, len(wall) // 3)), color="#27845d")
    panels[1].set(xlabel="wall time (s)", ylabel="records", title="Calculation cost")
    panels[2].semilogy(
        np.arange(len(pairs)), edge_minimum, "o", ms=3.2, color="#d06434"
    )
    panels[2].axhline(0.9, color="0.35", ls=":", lw=1.0)
    panels[2].set(
        xlabel="cross-graph edge",
        ylabel=r"minimum $\sigma(S_{ij}^{(6)})$",
        title="Exact signed overlap blocks",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    png = output / "phenol_sa6_3d_cross_diagnostics.png"
    pdf = output / "phenol_sa6_3d_cross_diagnostics.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf, macro, wall, edge_minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_PHENOL_SA6_DATABASE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_cross_20260820"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database, protocol, verbose=0 if args.quiet else 1
    )
    run_id = "phenol-sa6-3d-cross-v1"

    def progress(index, stats):
        print(
            f"[3D] built R={R_GRID[index[0]]:.2f} A, "
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
        run_id=run_id,
        run_metadata={
            "purpose": "three-dimensional cross seed",
            "coordinates": ["R_OH_angstrom", "phi_CCOH_radian", "theta_COH_radian"],
            "workers": args.workers,
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
        canonical = canonical_cross_points()
        points = fit.expand_points(canonical)
        fit.frames.get_many(points)
        records = {point: fit.frames.get(point) for point in points}
        pairs = cross_pairs()
        raw_overlaps = fit.oracle.raw_overlap_many(pairs)
        singular_values = np.linalg.svd(raw_overlaps, compute_uv=False)

        mask = np.zeros(fit.shape, dtype=bool)
        energies = np.full((*fit.shape, 6), np.nan)
        record_ids = np.full(fit.shape, "", dtype="U64")
        for point, record in records.items():
            mask[point] = True
            energies[point] = record["energies"]
            record_ids[point] = fit.frames.record_id(point)

        energy_png, energy_pdf = _plot_energies(args.output, records)
        diagnostic_png, diagnostic_pdf, macro, wall, edge_minimum = _plot_diagnostics(
            args.output, points, records, pairs, singular_values
        )
        data_path = args.output / "phenol_sa6_3d_cross_data.npz"
        np.savez_compressed(
            data_path,
            r_oh=R_GRID,
            phi=PHI_GRID,
            theta=THETA_GRID,
            sampled_mask=mask,
            energies=energies,
            record_ids=record_ids,
            points=np.asarray(points),
            pairs=np.asarray(pairs),
            overlaps=raw_overlaps,
            overlap_singular_values=singular_values,
            macroiterations=macro,
            wall_seconds=wall,
        )
        gates = {
            "all_records_orbitally_relaxed": all(
                bool(record["orbital_relaxed"]) for record in records.values()
            ),
            "all_records_singlets": bool(
                max(
                    np.max(np.abs(record["spins"]))
                    for record in records.values()
                )
                <= 1.0e-5
            ),
            "all_overlap_blocks_finite": bool(np.all(np.isfinite(raw_overlaps))),
            "no_active_claims": database.stats["claims"] == 0,
        }
        passed = all(gates.values())
        database.update_run(run_id, "sampled" if passed else "failed")
        summary = {
            "passed": passed,
            "gates": gates,
            "workers": args.workers,
            "canonical_geometries": len(canonical),
            "effective_geometries": len(points),
            "cross_edges": len(pairs),
            "new_records": fit.frames.stats["built"],
            "reused_records": fit.frames.stats["database_hits"],
            "median_macroiterations": float(np.median(macro)),
            "maximum_macroiterations": int(np.max(macro)),
            "median_wall_seconds": float(np.median(wall)),
            "minimum_full6_overlap_singular_value": float(np.min(edge_minimum)),
            "database": str(args.database),
            "database_stats": database.stats,
            "frame_stats": fit.frames.stats,
            "data": str(data_path),
            "figures": {
                "energies": str(energy_png),
                "energies_pdf": str(energy_pdf),
                "diagnostics": str(diagnostic_png),
                "diagnostics_pdf": str(diagnostic_pdf),
            },
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
        print(json.dumps(_jsonable(summary), indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
