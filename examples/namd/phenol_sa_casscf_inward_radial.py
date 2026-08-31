#!/usr/bin/env python3
"""Extend the database-backed phenol SA(6)-CASSCF backbone inward."""

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
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolSACASSCFProvider,
    phenol_sa6_protocol,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
TARGET_RADII = (0.85, 0.80, 0.75)


def geometry(radius):
    chart = PhenolReactiveChart()
    coordinate = np.array(chart.equilibrium, copy=True)
    coordinate[0] = float(radius)
    return chart.geometry(coordinate)


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


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_PHENOL_SA6_DATABASE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_inward_radial_20260821"),
    )
    parser.add_argument(
        "--radii", type=float, nargs="+", default=TARGET_RADII
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _plot(output, radii, records, singular):
    colors = plt.cm.viridis(np.linspace(0.06, 0.90, 6))
    reference = float(records[-1]["energies"][0])
    figure, panels = plt.subplots(1, 2, figsize=(8.3, 3.5), constrained_layout=True)
    for state, color in enumerate(colors):
        energy = np.asarray([record["energies"][state] for record in records])
        panels[0].plot(
            radii,
            (energy - reference) * HARTREE_TO_EV,
            "o-",
            color=color,
            ms=3.8,
            lw=1.2,
            label=f"S{state}",
        )
    panels[0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="relative energy (eV)",
        title="Inward SA(6)-CASSCF backbone",
    )
    panels[0].legend(frameon=False, ncol=2, fontsize=7.5)
    midpoint = 0.5 * (radii[:-1] + radii[1:])
    for state, color in enumerate(colors):
        panels[1].plot(
            midpoint,
            singular[:, state],
            "o-",
            color=color,
            ms=3.8,
            lw=1.2,
            label=rf"$\sigma_{state}$",
        )
    panels[1].axhline(0.90, color="#555555", ls=":", lw=1.0)
    panels[1].set(
        xlabel=r"edge midpoint $R_{OH}$ ($\AA$)",
        ylabel="overlap singular value",
        ylim=(0.0, 1.03),
        title="Six-root subspace continuity",
    )
    for label, panel in zip("ab", panels, strict=True):
        panel.text(
            -0.14,
            1.03,
            label,
            transform=panel.transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
        )
        panel.grid(axis="y", alpha=0.2)
    png = output / "phenol_sa6_inward_radial.png"
    pdf = output / "phenol_sa6_inward_radial.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main():
    args = _arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = sorted(set(map(float, args.radii)), reverse=True)
    protocol = phenol_sa6_protocol()
    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database, protocol, verbose=0 if args.quiet else 1
    )
    overlap = PhenolCASSCFOverlap()
    run_id = "phenol-sa6-inward-radial-v1"
    database.start_run(
        run_id,
        metadata={
            "purpose": "extend the physical initial-state radial box inward",
            "target_radii_angstrom": targets,
            "transport": "sequential from the nearest larger-radius record",
        },
        status="calculating",
    )

    first_sample = {
        "coordinates": np.asarray((targets[0], 0.0, PhenolReactiveChart().equilibrium[2])),
        "geometry": geometry(targets[0]),
    }
    nearest = provider.nearest(first_sample)
    if nearest is None:
        raise RuntimeError("no qualified SA-CASSCF record can seed the inward chain")
    previous_id, previous, _coordinates, _distance = nearest
    built = reused = 0
    chain = []
    try:
        for radius in targets:
            sample = {
                "coordinates": np.asarray(
                    (radius, 0.0, PhenolReactiveChart().equilibrium[2])
                ),
                "geometry": geometry(radius),
            }
            specification = {"geometry": sample["geometry"], "protocol": protocol}
            record = database.get(specification)
            source = "database"
            if record is None:
                claim = database.claim(specification, run_id)
                if claim == "complete":
                    record = database.get(specification)
                elif claim != "acquired":
                    raise RuntimeError(f"the R={radius:.3f} A record is already claimed")
                else:
                    try:
                        record = provider.calculate(
                            sample,
                            initial_record=previous,
                            initial_record_id=previous_id,
                        )
                        record_id, _ = database.put(
                            specification,
                            record,
                            metadata={
                                "distance_angstrom": radius,
                                "qualified": True,
                                "purpose": "inward physical initial-state extension",
                            },
                        )
                    except Exception:
                        database.release_claim(specification, run_id)
                        raise
                    source = "built"
                    built += 1
            if source == "database":
                record_id = database.identifier(specification)
                reused += 1
            if not bool(record.get("orbital_relaxed", record.get("converged", False))):
                raise RuntimeError(f"R={radius:.3f} A is not orbitally relaxed")
            if np.max(np.abs(record["spins"])) > 1.0e-5:
                raise RuntimeError(f"R={radius:.3f} A contains spin-contaminated roots")
            database.note_run_record(run_id, record_id, sample, source)
            previous_id, previous = record_id, record
            chain.append((radius, record_id, record, source))
            print(
                f"[inward] R={radius:.3f} A {source}, "
                f"macro={len(np.asarray(record.get('macro_history', ())))}, "
                f"wall={float(record.get('wall_seconds', np.nan)):.1f} s",
                flush=True,
            )
        database.update_run(run_id, "sampled")
    except Exception:
        database.update_run(run_id, "failed")
        raise

    outer_radius = 0.90
    outer_specification = {"geometry": geometry(outer_radius), "protocol": protocol}
    outer = database.get(outer_specification)
    outer_id = database.identifier(outer_specification)
    ordered = list(reversed(chain)) + [(outer_radius, outer_id, outer, "database")]
    radii = np.asarray([item[0] for item in ordered])
    records = [item[2] for item in ordered]
    blocks = np.asarray(
        [overlap(left, right) for left, right in zip(records[:-1], records[1:])]
    )
    singular = np.linalg.svd(blocks, compute_uv=False)
    png, pdf = _plot(args.output, radii, records, singular)
    artifact = args.output / "phenol_sa6_inward_radial.npz"
    np.savez_compressed(
        artifact,
        radii=radii,
        record_ids=np.asarray([item[1] for item in ordered]),
        energies=np.asarray([record["energies"] for record in records]),
        overlaps=blocks,
        overlap_singular_values=singular,
    )
    summary = {
        "passed": bool(
            np.min(singular) >= 0.90
            and all(
                bool(record.get("orbital_relaxed", record.get("converged", False)))
                for record in records
            )
        ),
        "radii_angstrom": radii,
        "built_records": built,
        "reused_records": reused,
        "minimum_overlap_singular_value": float(np.min(singular)),
        "artifact": artifact,
        "figure": png,
        "figure_pdf": pdf,
    }
    (args.output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n"
    )
    database.close()
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
