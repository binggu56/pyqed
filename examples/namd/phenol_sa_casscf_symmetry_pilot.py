#!/usr/bin/env python3
"""Qualify DB-backed, reflection-reduced phenol SA(6)-CASSCF sampling."""

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
from pyqed.ldr.database import canonical_json
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
R_GRID = np.asarray((0.95, 1.25, 1.55))
PHI_GRID = np.asarray((-0.40, -0.20, 0.0, 0.20, 0.40))
CANONICAL_POINTS = (
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 4),
    (2, 4),
)
VALIDATION_POINTS = ((1, 1), (2, 0))


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


def _geometry_function():
    chart = PhenolReactiveChart()

    def geometry(coordinates):
        coordinate = np.array(chart.equilibrium, copy=True)
        coordinate[:2] = coordinates
        return chart.geometry(coordinate)

    return geometry


def _neighbor_pairs():
    pairs = []
    for i in range(len(R_GRID)):
        for j in range(len(PHI_GRID) - 1):
            pairs.append(((i, j), (i, j + 1)))
    for i in range(len(R_GRID) - 1):
        for j in range(len(PHI_GRID)):
            pairs.append(((i, j), (i + 1, j)))
    return tuple(pairs)


def _macroiterations(record):
    return int(len(np.asarray(record.get("macro_history", ()))))


def _planar_continuity(database, protocol, overlap):
    entries = []
    protocol_json = canonical_json(protocol)
    for entry in database.entries():
        specification = entry["specification"]
        if canonical_json(specification.get("protocol")) != protocol_json:
            continue
        geometry = np.asarray(specification["geometry"], dtype=float)
        oh = geometry[7] - geometry[6]
        radius = float(np.linalg.norm(oh))
        torsion = float(np.arctan2(oh[2], oh[1]))
        if abs(torsion) <= 1.0e-10 and 0.90 - 1.0e-10 <= radius <= 1.55 + 1.0e-10:
            entries.append((radius, entry))
    entries.sort(key=lambda item: item[0])
    blocks = []
    for (left_radius, left), (right_radius, right) in zip(entries[:-1], entries[1:]):
        block = database.get_overlap(left["id"], right["id"], overlap.protocol)
        if block is None:
            left_record = database.get(left["specification"])
            right_record = database.get(right["specification"])
            block = overlap(left_record, right_record)
            database.put_overlap(
                left["id"],
                right["id"],
                overlap.protocol,
                block,
                metadata={"diagnostic": "planar local continuity"},
            )
        blocks.append(block)
    return (
        np.asarray([item[0] for item in entries]),
        np.asarray([np.linalg.svd(block, compute_uv=False) for block in blocks]),
    )


def _plot_pes(output, canonical):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    anchor = float(canonical[(1, 2)]["energies"][0])
    figure, panels = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for panel, j in zip(panels, (2, 3, 4)):
        energies = np.asarray([canonical[(i, j)]["energies"] for i in range(3)])
        relative = (energies - anchor) * HARTREE_TO_EV
        for state, color in enumerate(colors):
            panel.plot(
                R_GRID,
                relative[:, state],
                "o-",
                color=color,
                lw=1.2,
                ms=4.0,
                label=f"S{state}",
            )
        panel.set_title(rf"$\phi_{{CCOH}}={PHI_GRID[j]:.1f}$ rad")
        panel.set_xlabel(r"$R_{OH}$ ($\AA$)")
        panel.grid(alpha=0.2)
    panels[0].set_ylabel(r"$E_i-E_{S_0}(1.25\,\AA,0)$ (eV)")
    panels[-1].legend(ncol=2, fontsize=7.5)
    figure.suptitle("Phenol SA(6)-CASSCF(10,10)/6-31+G* canonical pilot")
    png = output / "phenol_sa6_symmetry_pilot_pes.png"
    pdf = output / "phenol_sa6_symmetry_pilot_pes.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _plot_convergence(output, canonical):
    macro = np.asarray(
        [[_macroiterations(canonical[(i, j)]) for j in (2, 3, 4)] for i in range(3)]
    )
    wall = np.asarray(
        [[float(canonical[(i, j)]["wall_seconds"]) for j in (2, 3, 4)] for i in range(3)]
    )
    gradient = np.asarray(
        [[float(canonical[(i, j)]["orbital_gradient"]) for j in (2, 3, 4)] for i in range(3)]
    )
    figure, panels = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)
    values = (macro, wall, np.log10(np.maximum(gradient, 1.0e-16)))
    titles = ("macroiterations", "wall time (s)", r"$\log_{10}|g_{orb}|$")
    formats = ("d", ".0f", ".1f")
    for panel, array, title, fmt in zip(panels, values, titles, formats):
        image = panel.imshow(array, origin="lower", aspect="auto", cmap="magma")
        for i in range(3):
            for j in range(3):
                panel.text(j, i, format(array[i, j], fmt), ha="center", va="center", color="white")
        panel.set(
            xticks=range(3),
            xticklabels=("0.0", "0.2", "0.4"),
            yticks=range(3),
            yticklabels=[f"{value:.2f}" for value in R_GRID],
            xlabel=r"$\phi_{CCOH}$ (rad)",
            title=title,
        )
        figure.colorbar(image, ax=panel, shrink=0.82)
    panels[0].set_ylabel(r"$R_{OH}$ ($\AA$)")
    png = output / "phenol_sa6_symmetry_pilot_convergence.png"
    pdf = output / "phenol_sa6_symmetry_pilot_convergence.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf, macro, wall, gradient


def _plot_validation(output, validations):
    labels = [
        rf"{item['coordinates'][0]:.2f} $\AA$, {item['coordinates'][1]:.1f} rad"
        for item in validations
    ]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    figure, panels = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    x = np.arange(len(validations))
    width = 0.12
    for state, color in enumerate(colors):
        errors = [item["energy_error_mev"][state] for item in validations]
        panels[0].bar(x + (state - 2.5) * width, errors, width, color=color, label=f"S{state}")
    panels[0].axhline(0.0, color="0.25", lw=0.8)
    panels[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="explicit minus reflected energy (meV)",
        title="Independent negative-$\\phi$ calculations",
    )
    panels[0].legend(ncol=2, fontsize=7.5)
    singular = np.asarray([item["overlap_singular_values"] for item in validations])
    for state, color in enumerate(colors):
        panels[1].plot(
            x, singular[:, state], "o", color=color, label=rf"$\sigma_{state}$"
        )
    panels[1].axhline(0.98, color="0.4", ls=":", lw=1.0)
    panels[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="six-state overlap singular value",
        ylim=(min(0.95, float(np.min(singular)) - 0.005), 1.005),
        title="Reflected vs explicit state subspaces",
    )
    png = output / "phenol_sa6_symmetry_pilot_mirror_validation.png"
    pdf = output / "phenol_sa6_symmetry_pilot_mirror_validation.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _plot_planar_continuity(output, radii, singular_values):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, singular_values.shape[1]))
    midpoint = 0.5 * (radii[:-1] + radii[1:])
    figure, panel = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for state, color in enumerate(colors):
        panel.semilogy(
            midpoint,
            singular_values[:, state],
            "o-",
            color=color,
            lw=1.1,
            ms=3.8,
            label=rf"$\sigma_{state}$",
        )
    panel.axhline(0.90, color="0.35", ls=":", lw=1.0, label="fit-ready gate")
    panel.set(
        xlabel=r"edge midpoint $R_{OH}$ ($\AA$)",
        ylabel="six-state overlap singular value",
        ylim=(1.0e-7, 1.1),
        title=r"Local continuity of the planar SA(6) state subspace",
    )
    panel.grid(alpha=0.2, which="both")
    panel.legend(ncol=2, fontsize=7.5)
    png = output / "phenol_sa6_planar_overlap_continuity.png"
    pdf = output / "phenol_sa6_planar_overlap_continuity.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


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
        default=Path("/private/tmp/phenol_sa6_symmetry_pilot_20260820"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_function()
    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    run_id = "phenol-sa6-symmetry-pilot-v1"

    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database, protocol, verbose=0 if args.quiet else 1
    )

    def progress(index, stats):
        coordinates = (R_GRID[index[0]], PHI_GRID[index[1]])
        print(
            f"[pilot] built R={coordinates[0]:.2f} A phi={coordinates[1]:+.2f} "
            f"({stats['built']} new)",
            flush=True,
        )

    with AbInitioFit(
        (R_GRID, PHI_GRID),
        6,
        electronic=provider,
        geometry=geometry,
        symmetry=PhenolReflectionSymmetry(),
        database=database,
        protocol=protocol,
        run_id=run_id,
        run_metadata={
            "purpose": "production symmetry pilot",
            "canonical_points": [list(point) for point in CANONICAL_POINTS],
            "explicit_negative_validations": [list(point) for point in VALIDATION_POINTS],
        },
        frame=lambda record: record,
        energies=lambda record: record["energies"],
        overlap=overlap,
        overlap_protocol=overlap.protocol,
        anchor=(1, 2),
        workers=1,
        progress=progress,
        energy_shift=None,
    ) as fit:
        fit.frames.get_many(CANONICAL_POINTS)
        all_points = tuple(
            (i, j) for i in range(len(R_GRID)) for j in range(len(PHI_GRID))
        )
        fit.frames.get_many(all_points)
        canonical = {
            (i, j): fit.frames.get((i, j))
            for i in range(3)
            for j in (2, 3, 4)
        }
        all_records = {point: fit.frames.get(point) for point in all_points}
        pairs = _neighbor_pairs()
        aligned_overlaps = fit.oracle.overlap_many(pairs)
        edge_singular = np.asarray(
            [np.linalg.svd(block, compute_uv=False) for block in aligned_overlaps]
        )
        planar_radii, planar_singular = _planar_continuity(
            database, protocol, overlap
        )

        validation_path = args.output / "validation" / "electronic.sqlite"
        validation_database = ElectronicDatabase(validation_path)
        validation_run = "explicit-negative-phi-v1"
        validation_database.start_run(
            validation_run,
            metadata={"protocol": protocol, "points": [list(point) for point in VALIDATION_POINTS]},
            status="calculating",
        )
        validations = []
        for point in VALIDATION_POINTS:
            coordinates = np.asarray((R_GRID[point[0]], PHI_GRID[point[1]]))
            sample = {
                "index": point,
                "coordinates": coordinates,
                "geometry": geometry(coordinates),
            }
            specification = {"geometry": sample["geometry"], "protocol": protocol}
            explicit = validation_database.get(specification)
            source = "database"
            if explicit is None:
                explicit = provider.calculate(
                    sample,
                    initial_record=all_records[point],
                    initial_record_id=fit.frames.record_id(point),
                )
                record_id, _inserted = validation_database.put(
                    specification,
                    explicit,
                    metadata={**sample, "purpose": "explicit symmetry validation"},
                )
                source = "built"
            else:
                record_id = validation_database.identifier(specification)
            validation_database.note_run_record(
                validation_run, record_id, sample, source
            )
            virtual = all_records[point]
            forward = overlap(virtual, explicit)
            backward = overlap(explicit, virtual)
            energy_error = (
                np.asarray(explicit["energies"]) - np.asarray(virtual["energies"])
            )
            validations.append(
                {
                    "index": point,
                    "coordinates": coordinates,
                    "source": source,
                    "record_id": record_id,
                    "energy_error_hartree": energy_error,
                    "energy_error_mev": energy_error * HARTREE_TO_EV * 1000.0,
                    "overlap_real": forward.real,
                    "overlap_imag": forward.imag,
                    "overlap_singular_values": np.linalg.svd(
                        forward, compute_uv=False
                    ),
                    "overlap_reciprocity_error": float(
                        np.linalg.norm(forward - backward.T.conj())
                    ),
                    "macroiterations": _macroiterations(explicit),
                    "wall_seconds": float(explicit["wall_seconds"]),
                    "orbital_gradient": float(explicit["orbital_gradient"]),
                    "spins": np.asarray(explicit["spins"]),
                    "converged": bool(explicit["orbital_relaxed"]),
                }
            )
            print(
                f"[validation] R={coordinates[0]:.2f} A phi={coordinates[1]:+.2f} "
                f"max |dE|={np.max(np.abs(energy_error))*HARTREE_TO_EV*1000:.4f} meV",
                flush=True,
            )

        pilot_records = [
            record for record in canonical.values() if "initial_record_id" in record
        ]
        all_canonical = list(canonical.values())
        pilot_macro = np.asarray([_macroiterations(record) for record in pilot_records])
        all_spin = np.concatenate([np.asarray(record["spins"]) for record in all_canonical])
        gates = {
            "all_canonical_orbitals_relaxed": all(
                bool(record["orbital_relaxed"]) for record in all_canonical
            ),
            "all_canonical_singlets": bool(np.max(np.abs(all_spin)) <= 1.0e-5),
            "new_pilot_median_macroiterations_le_25": bool(
                pilot_macro.size and np.median(pilot_macro) <= 25
            ),
            "new_pilot_max_macroiterations_le_50": bool(
                pilot_macro.size and np.max(pilot_macro) <= 50
            ),
            "mirror_energy_error_le_1e-5_hartree": bool(
                max(
                    np.max(np.abs(item["energy_error_hartree"]))
                    for item in validations
                )
                <= 1.0e-5
            ),
            "mirror_subspace_min_singular_value_ge_0_98": bool(
                min(
                    np.min(item["overlap_singular_values"])
                    for item in validations
                )
                >= 0.98
            ),
            "signed_overlap_reciprocity_le_1e-8": bool(
                max(item["overlap_reciprocity_error"] for item in validations)
                <= 1.0e-8
            ),
            "all_validation_orbitals_relaxed": all(
                item["converged"] for item in validations
            ),
            "all_validation_singlets": bool(
                max(np.max(np.abs(item["spins"])) for item in validations)
                <= 1.0e-5
            ),
            "overlaps_persisted": bool(database.stats["overlaps"] > 0),
            "planar_local_subspace_min_singular_value_ge_0_90": bool(
                np.min(planar_singular) >= 0.90
            ),
        }
        passed = all(gates.values())
        database.update_run(run_id, "qualified" if passed else "pilot-failed")
        validation_database.update_run(
            validation_run, "qualified" if passed else "pilot-failed"
        )

        pes_png, pes_pdf = _plot_pes(args.output, canonical)
        convergence_png, convergence_pdf, macro, wall, gradient = _plot_convergence(
            args.output, canonical
        )
        validation_png, validation_pdf = _plot_validation(args.output, validations)
        continuity_png, continuity_pdf = _plot_planar_continuity(
            args.output, planar_radii, planar_singular
        )
        energies = np.asarray(
            [
                [canonical[(i, j)]["energies"] for j in (2, 3, 4)]
                for i in range(3)
            ]
        )
        np.savez_compressed(
            args.output / "phenol_sa6_symmetry_pilot_data.npz",
            r_oh=R_GRID,
            phi_canonical=PHI_GRID[(2, 3, 4),],
            energies=energies,
            macroiterations=macro,
            wall_seconds=wall,
            orbital_gradient=gradient,
            edge_pairs=np.asarray(pairs),
            edge_overlap_singular_values=edge_singular,
            validation_energy_error_hartree=np.asarray(
                [item["energy_error_hartree"] for item in validations]
            ),
            validation_overlap_singular_values=np.asarray(
                [item["overlap_singular_values"] for item in validations]
            ),
            planar_radii=planar_radii,
            planar_overlap_singular_values=planar_singular,
        )
        summary = {
            "passed": passed,
            "gates": gates,
            "protocol": protocol,
            "main_database": str(args.database),
            "validation_database": str(validation_path),
            "main_database_stats": database.stats,
            "frame_stats": fit.frames.stats,
            "oracle_stats": fit.oracle.stats,
            "canonical_requested": len(CANONICAL_POINTS),
            "effective_geometries": len(all_points),
            "new_pilot_records": len(pilot_records),
            "new_pilot_macroiterations": pilot_macro,
            "new_pilot_median_macroiterations": (
                None if not pilot_macro.size else float(np.median(pilot_macro))
            ),
            "new_pilot_max_macroiterations": (
                None if not pilot_macro.size else int(np.max(pilot_macro))
            ),
            "edge_minimum_overlap_singular_value": float(np.min(edge_singular)),
            "planar_minimum_overlap_singular_value": float(
                np.min(planar_singular)
            ),
            "planar_discontinuous_edges": [
                {
                    "left_radius": float(planar_radii[index]),
                    "right_radius": float(planar_radii[index + 1]),
                    "minimum_singular_value": float(planar_singular[index, -1]),
                }
                for index in np.flatnonzero(planar_singular[:, -1] < 0.90)
            ],
            "validations": validations,
            "figures": {
                "pes": str(pes_png),
                "pes_pdf": str(pes_pdf),
                "convergence": str(convergence_png),
                "convergence_pdf": str(convergence_pdf),
                "mirror_validation": str(validation_png),
                "mirror_validation_pdf": str(validation_pdf),
                "planar_continuity": str(continuity_png),
                "planar_continuity_pdf": str(continuity_pdf),
            },
        }
        (args.output / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2) + "\n"
        )
        validation_database.close()
        print(json.dumps(_jsonable(summary), indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
