#!/usr/bin/env python3
"""Augment saved SO2 v5 records with fixed-sector v6 CASCI/SOMF data.

The converged v5 SA-CASSCF orbitals are reused exactly.  Only AO integrals,
larger-root CASCI diagonalizations, the SOMF contraction, and electronic
overlaps are rebuilt; neither SCF iteration nor orbital optimization is run.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
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

from examples.namd.generate_so2_cas88_somf import (
    TARGET_PLANE_PARITIES,
    augment_from_saved_orbitals,
    default_output,
    plot,
    protocol,
    specification,
)
from pyqed.ldr import ElectronicDatabase
from pyqed.ldr.so2 import full_spin_overlap, sparse_overlap_graph
from pyqed.units import au2ev


SOURCE_SCHEMA = "pyqed-so2-cas88-somf-v5"
OVERLAP_PROTOCOL = {
    "method": "raw CASCI wavefunction overlap",
    "state_window": "v6 fixed molecular-plane sectors",
    "unitarized": False,
}


def _settings(source_protocol, arguments):
    active = source_protocol["active_space"]
    orbitals = source_protocol["orbitals"]
    return SimpleNamespace(
        basis=source_protocol["basis"],
        ncas=int(active["orbitals"]),
        nelecas=int(active["electrons"]),
        singlet_roots=len(TARGET_PLANE_PARITIES),
        triplet_roots=len(TARGET_PLANE_PARITIES),
        singlet_candidates=int(arguments.singlet_candidates),
        triplet_candidates=int(arguments.triplet_candidates),
        orbital_backend=orbitals.get("optimizer", "pyscf"),
        symmetry_adapted=bool(orbitals.get("point_group_constrained", True)),
        spin_root_cushion=int(arguments.spin_root_cushion),
        spin_tol=float(arguments.spin_tol),
        verbose=int(arguments.verbose),
        reuse_saved_orbitals=True,
    )


def _source_entries(database, limit=None, source_summary=None):
    entries = [
        item
        for item in database.entries()
        if item["specification"].get("protocol", {}).get("schema") == SOURCE_SCHEMA
    ]
    entries.sort(
        key=lambda item: (
            item["metadata"].get("name") != "center",
            item["created_at"],
        )
    )
    if source_summary is not None:
        summary = json.loads(Path(source_summary).read_text())
        by_id = {item["id"]: item for item in entries}
        selected = []
        for name, point in summary["points"].items():
            key = specification(point["coordinate"], summary["protocol"])
            identifier = database.identifier(key)
            if identifier not in by_id:
                raise KeyError(
                    f"source summary record {name!r} ({identifier}) is absent"
                )
            item = copy.deepcopy(by_id[identifier])
            item["metadata"]["name"] = name
            selected.append(item)
        entries = selected
    if limit is not None:
        entries = entries[: int(limit)]
    return entries


def _worker(source_record, settings):
    return augment_from_saved_orbitals(source_record, settings)


def _record_name(entry, index, used):
    base = str(entry["metadata"].get("name", f"record-{index:03d}"))
    name = base
    suffix = 1
    while name in used:
        name = f"{base}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _overlap_diagnostics(database, records, identifiers):
    if len(records) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)
    coordinates = np.asarray([record["coordinate"] for record in records])
    bounds = (2.55, 3.05, 0.25, np.deg2rad(100.0), np.deg2rad(140.0))
    pairs, _lengths = sparse_overlap_graph(coordinates, bounds, neighbors=3)
    minima = []
    for left, right in pairs:
        overlap = database.get_overlap(
            identifiers[left], identifiers[right], OVERLAP_PROTOCOL
        )
        if overlap is None:
            overlap = full_spin_overlap(records[left], records[right])
            database.put_overlap(
                identifiers[left],
                identifiers[right],
                OVERLAP_PROTOCOL,
                overlap,
                metadata={"unitarized": False},
            )
        minima.append(float(np.min(np.linalg.svd(overlap, compute_uv=False))))
    return pairs, np.asarray(minima)


def plot_diagnostics(names, records, pair_minima, output, link_threshold=0.1):
    """Show candidate sectors and raw-link conditioning for the v6 migration."""

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.6, 3.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.2, 1.2, 1.0)},
    )
    colors = {-1: "#0072B2", 1: "#D55E00"}
    labels = {-1: "plane odd", 1: "plane even"}
    for axis, multiplicity in zip(axes[:2], ("singlet", "triplet")):
        for sample, record in enumerate(records):
            candidates = record["candidate_roots"]
            energies = np.asarray(candidates[f"{multiplicity}_energies"])
            parities = np.asarray(candidates[f"{multiplicity}_candidate_parities"])
            selected = set(
                map(int, candidates[f"{multiplicity}_selected_indices"])
            )
            energies = (energies - np.min(energies)) * au2ev
            for root, (energy, parity) in enumerate(zip(energies, parities)):
                x = sample + 0.045 * (root - 0.5 * (len(energies) - 1))
                axis.scatter(
                    x,
                    energy,
                    s=34 if root in selected else 13,
                    marker="o" if root in selected else "x",
                    facecolors=colors[int(parity)] if root in selected else None,
                    edgecolors=colors[int(parity)] if root in selected else None,
                    color=None if root in selected else colors[int(parity)],
                    linewidths=1.0,
                )
        axis.set(
            xlabel="saved geometry index",
            ylabel="Candidate energy (eV)",
            title=multiplicity.capitalize() + " candidates",
        )
        axis.spines[["top", "right"]].set_visible(False)
    for parity in (-1, 1):
        axes[0].scatter([], [], color=colors[parity], label=labels[parity])
    axes[0].scatter([], [], color="0.3", marker="o", s=34, label="selected")
    axes[0].scatter([], [], color="0.3", marker="x", s=18, label="candidate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    if len(pair_minima):
        order = np.arange(len(pair_minima))
        sorted_minima = np.sort(pair_minima)
        axes[2].plot(order, sorted_minima, color="0.45", lw=1.0)
        passed = sorted_minima >= float(link_threshold)
        axes[2].scatter(
            order[~passed], sorted_minima[~passed], s=22, color="#D55E00",
            label="below gate",
        )
        axes[2].scatter(
            order[passed], sorted_minima[passed], s=22, color="#0072B2",
            label="passed",
        )
        axes[2].axhline(link_threshold, color="0.45", ls="--", lw=1.0)
        axes[2].set(
            xlabel="local raw link (sorted)",
            ylabel=r"minimum singular value $\sigma_{\min}$",
            title="Selected-window links",
            ylim=(0.0, min(1.05, max(0.2, 1.08 * np.max(pair_minima)))),
        )
        axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    else:
        axes[2].text(0.5, 0.5, "one geometry\n(no links)", ha="center", va="center")
        axes[2].set(xticks=[], yticks=[], title="Selected-window links")
    axes[2].spines[["top", "right"]].set_visible(False)
    for label, axis in zip("abc", axes):
        axis.text(
            -0.12,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
        )
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_output() / "electronic.sqlite")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--singlet-candidates", type=int, default=6)
    parser.add_argument("--triplet-candidates", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=10)
    parser.add_argument("--spin-tol", type=float, default=1.0e-6)
    parser.add_argument("--verbose", type=int, default=1)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    database = ElectronicDatabase(arguments.database)
    entries = _source_entries(
        database, arguments.limit, source_summary=arguments.source_summary
    )
    if not entries:
        database.close()
        raise RuntimeError(f"no {SOURCE_SCHEMA} records found in {arguments.database}")
    settings = _settings(entries[0]["specification"]["protocol"], arguments)
    target_protocol = protocol(settings)
    run_id = "so2-cas88-somf-v6-augment-" + time.strftime("%Y%m%dT%H%M%S")
    database.start_run(
        run_id,
        metadata={
            "protocol": target_protocol,
            "source_schema": SOURCE_SCHEMA,
            "repeated_scf": False,
            "repeated_casscf": False,
        },
        status="augmenting",
    )

    jobs = []
    used_names = set()
    for index, entry in enumerate(entries):
        source = database.get(entry["specification"])
        name = _record_name(entry, index, used_names)
        target_key = specification(source["coordinate"], target_protocol)
        cached = database.get(target_key)
        jobs.append(
            {
                "index": index,
                "name": name,
                "source_id": entry["id"],
                "source": source,
                "target_key": target_key,
                "record": cached,
            }
        )

    missing = [job for job in jobs if job["record"] is None]
    try:
        if missing and arguments.workers == 1:
            for job in missing:
                print(f"[SO2 v6] augmenting {job['name']}", flush=True)
                job["record"] = _worker(job["source"], settings)
        elif missing:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(arguments.workers, len(missing)), mp_context=context
            ) as executor:
                futures = {
                    executor.submit(_worker, job["source"], settings): job
                    for job in missing
                }
                for future in as_completed(futures):
                    job = futures[future]
                    job["record"] = future.result()
                    print(
                        f"[SO2 v6] augmented {job['name']} in "
                        f"{job['record']['diagnostics']['seconds']:.1f} s",
                        flush=True,
                    )

        records = []
        names = []
        identifiers = []
        for job in jobs:
            identifier, inserted = database.put(
                job["target_key"],
                job["record"],
                metadata={
                    "name": job["name"],
                    "source_record_id": job["source_id"],
                    "repeated_scf": False,
                    "repeated_casscf": False,
                    "diagnostics": job["record"]["diagnostics"],
                },
            )
            source = "v5-orbital-augmentation" if inserted else "database"
            database.note_run_record(
                run_id,
                identifier,
                {
                    "index": [job["index"]],
                    "name": job["name"],
                    "coordinate": np.asarray(job["record"]["coordinate"]).tolist(),
                    "source_record_id": job["source_id"],
                },
                source,
            )
            names.append(job["name"])
            records.append(job["record"])
            identifiers.append(identifier)

        pairs, pair_minima = _overlap_diagnostics(
            database, records, identifiers
        )
        database.update_run(run_id, "complete")
        result_figure = arguments.output / f"{run_id}.png"
        plot(dict(zip(names, records)), result_figure)
        diagnostic_figure = arguments.output / f"{run_id}-sector-links.png"
        link_threshold = 0.1
        plot_diagnostics(
            names,
            records,
            pair_minima,
            diagnostic_figure,
            link_threshold=link_threshold,
        )
        link_rows = [
            {
                "left": names[int(left)],
                "right": names[int(right)],
                "minimum_singular_value": float(value),
                "passed": bool(value >= link_threshold),
            }
            for (left, right), value in zip(pairs, pair_minima)
        ]
        summary = {
            "run_id": run_id,
            "database": str(arguments.database),
            "source_schema": SOURCE_SCHEMA,
            "protocol": target_protocol,
            "records": len(records),
            "new_records": database.writes,
            "repeated_scf": False,
            "repeated_casscf": False,
            "minimum_raw_link_singular_value": (
                None if not len(pair_minima) else float(np.min(pair_minima))
            ),
            "raw_link_gate": link_threshold,
            "raw_links_passed": int(np.count_nonzero(pair_minima >= link_threshold)),
            "raw_links_total": int(len(pair_minima)),
            "dynamics_ready": bool(
                len(pair_minima) and np.all(pair_minima >= link_threshold)
            ),
            "raw_links": link_rows,
            "result_figure": str(result_figure),
            "diagnostic_figure": str(diagnostic_figure),
            "points": {
                name: {
                    "coordinate": np.asarray(record["coordinate"]).tolist(),
                    "source_record_id": job["source_id"],
                    "selected_singlet_candidates": np.asarray(
                        record["candidate_roots"]["singlet_selected_indices"]
                    ).tolist(),
                    "selected_triplet_candidates": np.asarray(
                        record["candidate_roots"]["triplet_selected_indices"]
                    ).tolist(),
                    "diagnostics": record["diagnostics"],
                }
                for name, record, job in zip(names, records, jobs)
            },
        }
        summary_path = arguments.output / f"{run_id}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        database.update_run(run_id, "failed")
        raise
    finally:
        database.close()


if __name__ == "__main__":
    main()
