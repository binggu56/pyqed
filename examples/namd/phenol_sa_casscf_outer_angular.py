#!/usr/bin/env python3
"""Sparse outer-region angular stars for the resolved phenol P1 channel."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from pyqed.ldr.overlap import positive_link_gauge, procrustes, track_states
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
RADII = np.asarray((1.95, 2.10))
TORSIONS = np.asarray((0.0, 0.10, 0.20, 0.30, 0.40))
BENDS = np.deg2rad(np.asarray((104.0, 108.8, 114.0)))
COLORS = ("#0072B2", "#D55E00", "#009E73")


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


def _coordinate(chart, radius, torsion=0.0, bend=None):
    value = np.array(chart.equilibrium, copy=True)
    value[0] = float(radius)
    value[1] = float(torsion)
    if bend is not None:
        value[2] = float(bend)
    return value


def _specification(chart, protocol, coordinate):
    geometry = np.asarray(chart.geometry(coordinate))
    return geometry, {"geometry": geometry, "protocol": protocol}


def _extract_diagnostic(record):
    return {
        name[len("diagnostic_") :]: np.asarray(value)
        for name, value in record.items()
        if name.startswith("diagnostic_")
    }


def _production_record(record):
    return {
        name: value
        for name, value in record.items()
        if not name.startswith("diagnostic_")
    }


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _diagnostic_path(root, radius, label):
    return root / f"r{radius:.5f}_{label}.npz"


def _angular_points(chart, radius):
    torsions = tuple(
        (
            f"phi{value:.3f}".replace(".", "p"),
            _coordinate(chart, radius, torsion=value),
        )
        for value in TORSIONS[1:]
    )
    return (*torsions,
        ("theta104p0", _coordinate(chart, radius, bend=np.deg2rad(104.0))),
        ("theta114p0", _coordinate(chart, radius, bend=np.deg2rad(114.0))),
    )


def _calculate_radius(database_path, diagnostic_dir, radius, diagnostic_roots,
                      ci_workers, quiet):
    chart = PhenolReactiveChart()
    protocol = phenol_sa6_protocol()
    database = ElectronicDatabase(database_path)
    provider = PhenolSACASSCFProvider(
        database,
        protocol,
        diagnostic_roots=diagnostic_roots,
        diagnostic_workers=ci_workers,
        verbose=0 if quiet else 1,
    )
    planar_coordinate = _coordinate(chart, radius)
    _planar_geometry, planar_specification = _specification(
        chart, protocol, planar_coordinate
    )
    planar = database.get(planar_specification)
    if planar is None:
        raise RuntimeError(f"missing planar bridge anchor at R={radius:.3f} A")
    planar_id = database.identifier(planar_specification)
    results = []
    previous_phi = planar
    previous_phi_id = planar_id
    for label, coordinate in _angular_points(chart, radius):
        geometry, specification = _specification(chart, protocol, coordinate)
        record = database.get(specification)
        source = "database"
        if label.startswith("phi"):
            initial = previous_phi
            initial_id = previous_phi_id
        else:
            initial = planar
            initial_id = planar_id
        cache = _diagnostic_path(Path(diagnostic_dir), radius, label)
        if record is None:
            sample = {
                "coordinates": coordinate[:3],
                "geometry": geometry,
            }
            calculated = provider.calculate(
                sample,
                initial_record=initial,
                initial_record_id=initial_id,
            )
            diagnostic = _extract_diagnostic(calculated)
            record = _production_record(calculated)
            record_id, _ = database.put(
                specification,
                record,
                metadata={"source": "sparse outer angular diagnostic"},
            )
            source = "calculated"
        else:
            record_id = database.identifier(specification)
            if cache.is_file():
                diagnostic = {
                    name: value
                    for name, value in _load_npz(cache).items()
                    if name not in {"geometry", "record_id", "coordinate"}
                }
            else:
                diagnostic = provider.diagnostic_casci(
                    record,
                    nroots=diagnostic_roots,
                    workers=ci_workers,
                )
        np.savez_compressed(
            cache,
            coordinate=coordinate,
            geometry=record["geometry"],
            record_id=np.asarray(record_id),
            **diagnostic,
        )
        if label.startswith("phi"):
            previous_phi = record
            previous_phi_id = record_id
        results.append(
            {
                "label": label,
                "record_id": record_id,
                "source": source,
                "macroiterations": len(np.asarray(record.get("macro_history", ()))),
                "sa_wall_seconds": float(record.get("wall_seconds", np.nan)),
                "diagnostic_wall_seconds": float(diagnostic["wall_seconds"]),
                "agreement": float(diagnostic["sa_energy_agreement"]),
            }
        )
        print(
            f"[angular R={radius:.2f}] {label} {source}; "
            f"macro={results[-1]['macroiterations']}, "
            f"diagnostic={results[-1]['diagnostic_wall_seconds']:.2f} s",
            flush=True,
        )
    provider.close()
    database.close()
    return radius, results


def _record_and_diagnostic(database, chart, protocol, diagnostic_dir, radius, label,
                           bridge_diagnostic_dir=None):
    if label == "planar":
        coordinate = _coordinate(chart, radius)
        path = Path(bridge_diagnostic_dir) / f"r{radius:.5f}.npz"
    else:
        lookup = dict(_angular_points(chart, radius))
        coordinate = lookup[label]
        path = _diagnostic_path(Path(diagnostic_dir), radius, label)
    geometry, specification = _specification(chart, protocol, coordinate)
    record = database.get(specification)
    if record is None or not path.is_file():
        raise RuntimeError(f"missing stored angular artifact {path}")
    diagnostic = {
        name: value
        for name, value in _load_npz(path).items()
        if name not in {"geometry", "record_id", "coordinate"}
    }
    frame = {
        "geometry": geometry,
        "mo_coeff": record["mo_coeff"],
        "ci": diagnostic["ci"],
    }
    return coordinate, record, diagnostic, frame


def _tracked_chain(overlap, frames, energies, anchor_states, anchor_gauge):
    links = np.asarray(
        [overlap(left, right) for left, right in zip(frames[:-1], frames[1:])]
    )
    roots, selected_links = track_states(links, anchor=0, states=anchor_states)
    selected_energies = np.asarray(
        [energy[index] for energy, index in zip(energies, roots)]
    )
    local_gauge, _ = positive_link_gauge(selected_links, anchor=0)
    gauge = np.einsum("rij,jk->rik", local_gauge, anchor_gauge, optimize=True)
    p_links = np.asarray(
        [
            gauge[edge].conj().T @ selected_links[edge] @ gauge[edge + 1]
            for edge in range(len(selected_links))
        ]
    )
    diagonal = np.asarray([np.diag(value) for value in selected_energies])
    hamiltonian = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauge.conj(), diagonal, gauge, optimize=True,
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.swapaxes(-1, -2).conj())
    return {
        "links": links,
        "roots": roots,
        "selected_links": selected_links,
        "selected_energies": selected_energies,
        "singular": np.linalg.svd(selected_links, compute_uv=False),
        "gauge": gauge,
        "p_links": p_links,
        "hamiltonian": hamiltonian,
    }


def _plot(output, analyses, reference, radii):
    figure, panels = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    markers = ("o", "s")
    for radial, radius in enumerate(radii):
        result = analyses[float(radius)]
        phi = np.rad2deg(TORSIONS)
        for channel, color in enumerate(COLORS):
            energy = (result["phi"]["selected_energies"][:, channel] - reference) * HARTREE_TO_EV
            panels[0, 0].plot(
                np.concatenate((-phi[:0:-1], phi)),
                np.concatenate((energy[:0:-1], energy)),
                marker=markers[radial], color=color, lw=1.25, ms=3.5,
                label=(f"P{channel}, R={radius:.2f} " + r"$\AA$"),
            )
        theta = np.rad2deg(result["bend_coordinates"])
        for channel, color in enumerate(COLORS):
            panels[0, 1].plot(
                theta,
                (result["bend_energies"][:, channel] - reference) * HARTREE_TO_EV,
                marker=markers[radial], color=color, lw=1.25, ms=3.5,
                label=(f"P{channel}, R={radius:.2f} " + r"$\AA$"),
            )
        values = np.concatenate(
            (result["phi"]["singular"][:, -1], result["bend_singular"][:, -1])
        )
        labels = tuple(
            rf"${left:g}\to{right:g}$"
            for left, right in zip(TORSIONS[:-1], TORSIONS[1:])
        ) + (r"$0\to104^\circ$", r"$0\to114^\circ$")
        offset = (-0.08, 0.08)[radial]
        panels[1, 0].plot(
            np.arange(len(values)) + offset,
            values,
            marker=markers[radial], lw=1.2, ms=4,
            label=rf"$R={radius:.2f}$ $\AA$",
        )
        panels[1, 0].set_xticks(range(len(labels)), labels)
        roots = np.vstack(
            (
                result["phi"]["roots"],
                result["bend_roots"][[0, 2]],
            )
        )
        for channel, color in enumerate(COLORS):
            panels[1, 1].plot(
                np.arange(len(roots)) + offset,
                roots[:, channel],
                marker=markers[radial], color=color, lw=1.1, ms=3.5,
                label=f"P{channel}, R={radius:.2f} " + r"$\AA$",
            )
    panels[0, 0].set(
        xlabel=r"torsion $\phi$ (degree)", ylabel="selected energy (eV)",
        title="Reflection-completed torsional stars",
    )
    panels[0, 1].set(
        xlabel=r"bend $\theta$ (degree)", ylabel="selected energy (eV)",
        title="Planar bending stars",
    )
    panels[1, 0].axhline(0.90, color="0.35", ls=":", lw=1)
    panels[1, 0].set(
        ylabel=r"minimum $\sigma(S^{(3)})$", ylim=(0.0, 1.03),
        title="Angular transport quality",
    )
    panels[1, 1].axhspan(5.5, 9.5, color="#E69F00", alpha=0.10)
    panels[1, 1].set(
        xlabel="planar, torsion scan, bend 104/114",
        ylabel="diagnostic CASCI root index", yticks=range(10),
        title="Outer physical-channel identities",
    )
    for label, panel in zip("abcd", panels.flat):
        panel.text(
            0.02, 0.96, label, transform=panel.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold",
        )
        panel.grid(alpha=0.18)
        panel.legend(fontsize=7.2, frameon=False, ncol=2)
    figure.suptitle("Phenol sparse outer angular diagnostic on unchanged SA(6) orbitals")
    png = output / "phenol_sa6_outer_angular.png"
    pdf = output / "phenol_sa6_outer_angular.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        default=DEFAULT_PHENOL_SA6_DATABASE,
    )
    parser.add_argument(
        "--bridge", type=Path,
        default=Path("/private/tmp/phenol_sa6_bridge_20260820/phenol_sa6_bridge_p_gauge.npz"),
    )
    parser.add_argument(
        "--bridge-diagnostics", type=Path,
        default=Path("/private/tmp/phenol_sa6_bridge_20260820/diagnostic_roots"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_outer_angular_20260820"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ci-workers", type=int, default=2)
    parser.add_argument("--diagnostic-roots", type=int, default=10)
    parser.add_argument("--continuity-threshold", type=float, default=0.90)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--radii", type=lambda value: tuple(float(item) for item in value.split(",")),
        default=tuple(map(float, RADII)),
        help="comma-separated outer R_OH values in angstrom",
    )
    args = parser.parse_args()
    radii = np.asarray(args.radii, dtype=float)
    if radii.ndim != 1 or not len(radii) or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be a nonempty increasing sequence")
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = args.output / "diagnostic_roots"
    diagnostic_dir.mkdir(exist_ok=True)

    worker_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _calculate_radius,
                args.database,
                diagnostic_dir,
                float(radius),
                args.diagnostic_roots,
                args.ci_workers,
                args.quiet,
            ): float(radius)
            for radius in radii
        }
        for future in as_completed(futures):
            radius, result = future.result()
            worker_results[float(radius)] = result

    bridge = _load_npz(args.bridge)
    chart = PhenolReactiveChart()
    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    database = ElectronicDatabase(args.database)
    run_id = "phenol-sa6-sparse-outer-angular-v1"
    database.start_run(
        run_id,
        status="analyzing",
        metadata={"radii": radii, "torsions": TORSIONS, "bends": BENDS},
    )
    analyses = {}
    all_singular = []
    all_p_defects = []
    all_spins = []
    all_agreement = []
    all_roots = []
    for radius in radii:
        radius = float(radius)
        bridge_index = int(np.argmin(np.abs(bridge["radii"] - radius)))
        if abs(float(bridge["radii"][bridge_index]) - radius) > 1.0e-7:
            raise RuntimeError(f"bridge gauge does not contain R={radius:.2f} A")
        anchor_states = np.asarray(bridge["root_indices"][bridge_index])
        anchor_gauge = np.asarray(bridge["p_gauge"][bridge_index])
        loaded = {}
        phi_labels = tuple(label for label, _coordinate in _angular_points(chart, radius) if label.startswith("phi"))
        for label in ("planar", *phi_labels, "theta104p0", "theta114p0"):
            loaded[label] = _record_and_diagnostic(
                database,
                chart,
                protocol,
                diagnostic_dir,
                radius,
                label,
                bridge_diagnostic_dir=args.bridge_diagnostics,
            )
            all_spins.append(np.max(np.abs(loaded[label][2]["spins"])))
            all_agreement.append(float(loaded[label][2]["sa_energy_agreement"]))

        phi_labels = ("planar", *phi_labels)
        phi = _tracked_chain(
            overlap,
            [loaded[label][3] for label in phi_labels],
            [loaded[label][2]["energies"] for label in phi_labels],
            anchor_states,
            anchor_gauge,
        )
        bend_entries = []
        for label in ("theta104p0", "theta114p0"):
            bend_entries.append(
                _tracked_chain(
                    overlap,
                    [loaded["planar"][3], loaded[label][3]],
                    [loaded["planar"][2]["energies"], loaded[label][2]["energies"]],
                    anchor_states,
                    anchor_gauge,
                )
            )
        bend_roots = np.vstack(
            (bend_entries[0]["roots"][1], anchor_states, bend_entries[1]["roots"][1])
        )
        bend_energies = np.vstack(
            (
                bend_entries[0]["selected_energies"][1],
                loaded["planar"][2]["energies"][anchor_states],
                bend_entries[1]["selected_energies"][1],
            )
        )
        bend_singular = np.vstack(
            (bend_entries[0]["singular"], bend_entries[1]["singular"])
        )
        bend_selected_links = np.asarray(
            (
                bend_entries[0]["selected_links"][0].conj().T,
                bend_entries[1]["selected_links"][0],
            )
        )
        bend_p_links = np.asarray(
            (
                bend_entries[0]["p_links"][0].conj().T,
                bend_entries[1]["p_links"][0],
            )
        )
        bend_gauge = np.asarray(
            (
                bend_entries[0]["gauge"][1],
                anchor_gauge,
                bend_entries[1]["gauge"][1],
            )
        )
        bend_hamiltonian = np.asarray(
            (
                bend_entries[0]["hamiltonian"][1],
                phi["hamiltonian"][0],
                bend_entries[1]["hamiltonian"][1],
            )
        )
        analyses[radius] = {
            "anchor_states": anchor_states,
            "phi": phi,
            "bend_coordinates": BENDS,
            "bend_roots": bend_roots,
            "bend_energies": bend_energies,
            "bend_singular": bend_singular,
            "bend_selected_links": bend_selected_links,
            "bend_p_links": bend_p_links,
            "bend_gauge": bend_gauge,
            "bend_hamiltonian": bend_hamiltonian,
        }
        all_singular.extend(phi["singular"][:, -1])
        all_singular.extend(bend_singular[:, -1])
        all_roots.extend(phi["roots"].reshape(-1))
        all_roots.extend(bend_roots.reshape(-1))
        for links in (phi["p_links"], *(entry["p_links"] for entry in bend_entries)):
            rotations = procrustes(links)[0]
            all_p_defects.extend(np.linalg.norm(rotations - np.eye(3), axis=(-2, -1)))

    minimum_singular = float(np.min(all_singular))
    maximum_p_defect = float(np.max(all_p_defects))
    gates = {
        "unchanged_sa6_protocol": True,
        "all_ten_diagnostic_roots_singlets": float(np.max(all_spins)) <= 1.0e-5,
        "diagnostic_first6_reproduce_sa6": float(np.max(all_agreement)) <= 1.0e-6,
        "all_three_channels_inside_ten_root_window": int(np.max(all_roots)) < 10,
        "angular_links_continuous": minimum_singular >= args.continuity_threshold,
        "angular_P_links_positive": maximum_p_defect <= 1.0e-9,
    }
    passed = all(gates.values())
    reference = float(analyses[float(radii[0])]["phi"]["selected_energies"][0, 0])
    png, pdf = _plot(args.output, analyses, reference, radii)
    data_path = args.output / "phenol_sa6_outer_angular_p_gauge.npz"
    np.savez_compressed(
        data_path,
        radii=radii,
        torsions=TORSIONS,
        bends=BENDS,
        phi_root_indices=np.asarray([analyses[float(r)]["phi"]["roots"] for r in radii]),
        phi_selected_energies=np.asarray(
            [analyses[float(r)]["phi"]["selected_energies"] for r in radii]
        ),
        phi_selected_links=np.asarray(
            [analyses[float(r)]["phi"]["selected_links"] for r in radii]
        ),
        phi_p_gauge=np.asarray([analyses[float(r)]["phi"]["gauge"] for r in radii]),
        phi_p_hamiltonian=np.asarray(
            [analyses[float(r)]["phi"]["hamiltonian"] for r in radii]
        ),
        bend_root_indices=np.asarray(
            [analyses[float(r)]["bend_roots"] for r in radii]
        ),
        bend_selected_energies=np.asarray(
            [analyses[float(r)]["bend_energies"] for r in radii]
        ),
        bend_singular_values=np.asarray(
            [analyses[float(r)]["bend_singular"] for r in radii]
        ),
        bend_selected_links=np.asarray(
            [analyses[float(r)]["bend_selected_links"] for r in radii]
        ),
        bend_p_links=np.asarray(
            [analyses[float(r)]["bend_p_links"] for r in radii]
        ),
        bend_p_gauge=np.asarray(
            [analyses[float(r)]["bend_gauge"] for r in radii]
        ),
        bend_p_hamiltonian=np.asarray(
            [analyses[float(r)]["bend_hamiltonian"] for r in radii]
        ),
    )
    database.update_run(run_id, "transported" if passed else "needs_refinement")
    summary = {
        "passed": passed,
        "gates": gates,
        "radii_angstrom": radii,
        "positive_torsions_radian": TORSIONS,
        "reflection_completed_torsions_radian": np.concatenate(
            (-TORSIONS[:0:-1], TORSIONS)
        ),
        "bends_degree": np.rad2deg(BENDS),
        "angular_sa6_records": int(len(radii) * (len(TORSIONS) + 1)),
        "new_sa6_records_this_execution": sum(
            item["source"] == "calculated"
            for values in worker_results.values() for item in values
        ),
        "reused_sa6_records_this_execution": sum(
            item["source"] == "database"
            for values in worker_results.values() for item in values
        ),
        "minimum_selected_angular_singular_value": minimum_singular,
        "maximum_P_link_rotation_defect": maximum_p_defect,
        "maximum_diagnostic_spin_square": float(np.max(all_spins)),
        "maximum_first6_energy_disagreement_hartree": float(np.max(all_agreement)),
        "root_indices": {
            str(radius): {
                "torsion": analyses[float(radius)]["phi"]["roots"],
                "bend": analyses[float(radius)]["bend_roots"],
            }
            for radius in radii
        },
        "workers": args.workers,
        "ci_workers_per_process": args.ci_workers,
        "database": args.database,
        "database_stats": database.stats,
        "data": data_path,
        "figure": png,
        "figure_pdf": pdf,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
