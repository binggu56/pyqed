#!/usr/bin/env python3
"""Resolve the phenol SA(6) planar bridge with a larger CASCI root window.

The orbital optimization remains the qualified equal-weight SA(6)-CASSCF
protocol.  Ten singlet CASCI roots are then solved on each converged orbital
frame solely for state tracking and positive-link gauge diagnostics.
"""

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
from pyqed.ldr.overlap import positive_link_gauge, procrustes, track_states
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev
CHANNEL_COLORS = ("#0072B2", "#D55E00", "#009E73")


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


def _geometry(chart, radius):
    coordinate = np.array(chart.equilibrium, copy=True)
    coordinate[0] = float(radius)
    return coordinate, np.asarray(chart.geometry(coordinate))


def _extract_diagnostic(record):
    prefix = "diagnostic_"
    return {
        name[len(prefix) :]: np.asarray(value)
        for name, value in record.items()
        if name.startswith(prefix)
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


def _diagnostic_frame(record, diagnostic):
    return {
        "geometry": np.asarray(record["geometry"]),
        "mo_coeff": np.asarray(record["mo_coeff"]),
        "ci": np.asarray(diagnostic["ci"]),
    }


def _neighbor_links(database, record_ids, records, overlap):
    links = []
    for left_id, right_id, left, right in zip(
        record_ids[:-1], record_ids[1:], records[:-1], records[1:]
    ):
        block = database.get_overlap(left_id, right_id, overlap.protocol)
        if block is None:
            block = overlap(left, right)
            database.put_overlap(
                left_id,
                right_id,
                overlap.protocol,
                block,
                metadata={"purpose": "fine planar bridge SA(6) link"},
            )
        links.append(block)
    return np.asarray(links)


def _extended_gauge(radii, selected_links, selected_energies, old_gauge_path):
    anchor = int(np.argmin(np.abs(radii - 1.85)))
    anchor_gauge = np.eye(3, dtype=complex)
    old = None
    if old_gauge_path.is_file():
        old = _load_npz(old_gauge_path)
        old_anchor = int(np.argmin(np.abs(old["radii"] - radii[anchor])))
        if abs(float(old["radii"][old_anchor]) - radii[anchor]) > 1.0e-7:
            raise RuntimeError("the existing P gauge does not contain the 1.85 A anchor")
        anchor_gauge = np.asarray(old["p_gauge"][old_anchor])

    local_gauge, _ = positive_link_gauge(selected_links, anchor)
    gauge = np.einsum("rij,jk->rik", local_gauge, anchor_gauge, optimize=True)
    p_links = np.asarray(
        [
            gauge[edge].conj().T @ selected_links[edge] @ gauge[edge + 1]
            for edge in range(len(selected_links))
        ]
    )
    diagonal = np.asarray([np.diag(energy) for energy in selected_energies])
    hamiltonian = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauge.conj(),
        diagonal,
        gauge,
        optimize=True,
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.swapaxes(-1, -2).conj())

    combined_radii = radii
    combined_hamiltonian = hamiltonian
    combined_gauge = gauge
    if old is not None:
        old_radii = np.asarray(old.get("combined_radii", old["radii"]))
        old_hamiltonian = np.asarray(
            old.get("combined_p_hamiltonian", old["p_hamiltonian"])
        )
        old_gauge = np.asarray(old.get("combined_p_gauge", old["p_gauge"]))
        keep = old_radii < radii[anchor] - 1.0e-9
        combined_radii = np.concatenate((old_radii[keep], radii[anchor:]))
        combined_hamiltonian = np.concatenate(
            (old_hamiltonian[keep], hamiltonian[anchor:]), axis=0
        )
        combined_gauge = np.concatenate(
            (old_gauge[keep], gauge[anchor:]), axis=0
        )
    return {
        "anchor": anchor,
        "gauge": gauge,
        "links": p_links,
        "hamiltonian": hamiltonian,
        "combined_radii": combined_radii,
        "combined_hamiltonian": combined_hamiltonian,
        "combined_gauge": combined_gauge,
    }


def _plot(output, radii, diagnostic_energies, root_indices, selected_energies,
          diagnostic_singular, sa_singular, p_hamiltonian, coarse_bridge_minimum,
          threshold):
    anchor = int(np.argmin(np.abs(radii - 1.85)))
    reference = float(diagnostic_energies[anchor, 0])
    midpoint = 0.5 * (radii[:-1] + radii[1:])
    figure, panels = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)

    for root in range(diagnostic_energies.shape[1]):
        panels[0, 0].plot(
            radii,
            (diagnostic_energies[:, root] - reference) * HARTREE_TO_EV,
            color="0.72",
            lw=0.8,
            zorder=1,
        )
    for channel, color in enumerate(CHANNEL_COLORS):
        panels[0, 0].plot(
            radii,
            (selected_energies[:, channel] - reference) * HARTREE_TO_EV,
            "o-",
            color=color,
            ms=3.2,
            lw=1.35,
            label=f"P{channel}",
            zorder=2,
        )
        panels[0, 1].step(
            radii,
            root_indices[:, channel],
            where="mid",
            color=color,
            lw=1.4,
            label=f"P{channel}",
        )
    panels[0, 0].set(
        ylabel=r"$E-E_0(1.85\,\AA)$ (eV)",
        title="Ten-root CASCI window on SA(6) orbitals",
    )
    panels[0, 1].axhspan(5.5, 9.5, color="#E69F00", alpha=0.10)
    panels[0, 1].set(
        ylabel="diagnostic CASCI root index",
        yticks=range(diagnostic_energies.shape[1]),
        title="Physical-channel transport",
    )

    panels[1, 0].plot(
        midpoint,
        diagnostic_singular[:, -1],
        "o-",
        color="#0072B2",
        lw=1.4,
        ms=3.5,
        label="10-root tracked channels",
    )
    panels[1, 0].plot(
        midpoint,
        sa_singular[:, -1],
        "s--",
        color="#D55E00",
        lw=1.1,
        ms=3.2,
        label="SA(6)-only tracking",
    )
    if np.isfinite(coarse_bridge_minimum):
        panels[1, 0].plot(
            1.90,
            coarse_bridge_minimum,
            "D",
            color="#CC79A7",
            ms=5,
            label=r"old 1.85--1.95 $\AA$ link",
        )
    panels[1, 0].axhline(threshold, color="0.35", ls=":", lw=1.0)
    panels[1, 0].set(
        ylabel=r"minimum $\sigma(S_{i,i+1}^{(3)})$",
        ylim=(0.0, 1.03),
        title="Selected-subspace continuity",
    )

    shifted = (
        p_hamiltonian - reference * np.eye(3)[None]
    ) * HARTREE_TO_EV
    for channel, color in enumerate(CHANNEL_COLORS):
        panels[1, 1].plot(
            radii,
            shifted[:, channel, channel].real,
            color=color,
            lw=1.4,
            label=rf"$\bar H_{{{channel}{channel}}}$",
        )
    for color, (left, right) in zip(CHANNEL_COLORS, ((0, 1), (0, 2), (1, 2))):
        panels[1, 1].plot(
            radii,
            shifted[:, left, right].real,
            "--",
            color=color,
            lw=1.0,
            label=rf"$\bar H_{{{left}{right}}}$",
        )
    panels[1, 1].set(
        ylabel=r"$P$-gauge Hamiltonian (eV)",
        title="Positive-link gauge across the bridge",
    )

    for label, panel in zip("abcd", panels.flat):
        panel.text(
            0.02, 0.96, label, transform=panel.transAxes, va="top",
            ha="left", fontsize=11, fontweight="bold"
        )
        panel.set_xlabel(r"$R_{OH}$ ($\AA$)")
        panel.grid(alpha=0.18)
        panel.legend(fontsize=8, frameon=False, ncol=2)
    figure.suptitle("Phenol planar bridge: unchanged SA(6), enlarged diagnostic window")
    png = output / "phenol_sa6_bridge_diagnostic.png"
    pdf = output / "phenol_sa6_bridge_diagnostic.pdf"
    figure.savefig(png, dpi=350)
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
        default=Path("/private/tmp/phenol_sa6_bridge_20260820"),
    )
    parser.add_argument(
        "--old-gauge",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_p_gauge_20260820/"
            "phenol_sa6_tracked3_p_gauge.npz"
        ),
    )
    parser.add_argument("--start", type=float, default=1.80)
    parser.add_argument("--stop", type=float, default=2.10)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--diagnostic-roots", type=int, default=10)
    parser.add_argument("--ci-workers", type=int, default=4)
    parser.add_argument("--continuity-threshold", type=float, default=0.90)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = args.output / "diagnostic_roots"
    diagnostic_dir.mkdir(exist_ok=True)

    count = int(round((args.stop - args.start) / args.step))
    radii = args.start + args.step * np.arange(count + 1)
    if abs(float(radii[-1]) - args.stop) > 1.0e-8:
        raise ValueError("start, stop, and step do not define an inclusive grid")
    chart = PhenolReactiveChart()
    protocol = phenol_sa6_protocol()
    database = ElectronicDatabase(args.database)
    provider = PhenolSACASSCFProvider(
        database,
        protocol,
        diagnostic_roots=args.diagnostic_roots,
        diagnostic_workers=args.ci_workers,
        verbose=0 if args.quiet else 1,
    )
    overlap = PhenolCASSCFOverlap()
    run_id = "phenol-sa6-planar-bridge-diagnostic-v1"
    database.start_run(
        run_id,
        status="running",
        metadata={
            "purpose": "locate and transport the missing planar physical channel",
            "radii_angstrom": radii,
            "sa_roots": 6,
            "diagnostic_roots": args.diagnostic_roots,
        },
    )

    records = []
    diagnostics = []
    record_ids = []
    sources = []
    previous = None
    previous_id = None
    for index, radius in enumerate(radii):
        coordinate, geometry = _geometry(chart, radius)
        sample = {
            "index": (index,),
            "coordinates": coordinate[:3],
            "geometry": geometry,
        }
        specification = {"geometry": geometry, "protocol": protocol}
        record = database.get(specification)
        source = "database"
        if record is None:
            record = provider.calculate(
                sample,
                initial_record=previous,
                initial_record_id=previous_id,
            )
            diagnostic = _extract_diagnostic(record)
            production = _production_record(record)
            record_id, _ = database.put(
                specification,
                production,
                metadata={"source": "fine planar bridge", "run_id": run_id},
            )
            record = production
            source = "calculated"
        else:
            record_id = database.identifier(specification)
            diagnostic_path = diagnostic_dir / f"r{radius:.5f}.npz"
            if diagnostic_path.is_file():
                diagnostic = {
                    name: value
                    for name, value in _load_npz(diagnostic_path).items()
                    if name not in {"geometry", "record_id"}
                }
            else:
                diagnostic = provider.diagnostic_casci(
                    record,
                    nroots=args.diagnostic_roots,
                    workers=args.ci_workers,
                )

        diagnostic_path = diagnostic_dir / f"r{radius:.5f}.npz"
        np.savez_compressed(
            diagnostic_path,
            geometry=np.asarray(record["geometry"]),
            record_id=np.asarray(record_id),
            **diagnostic,
        )
        database.note_run_record(run_id, record_id, sample, source)
        records.append(record)
        diagnostics.append(diagnostic)
        record_ids.append(record_id)
        sources.append(source)
        previous = record
        previous_id = record_id
        print(
            f"[bridge] R={radius:.3f} A {source}; "
            f"SA macro={len(np.asarray(record.get('macro_history', ())))}, "
            f"diagnostic={float(diagnostic['wall_seconds']):.2f} s, "
            f"agreement={float(diagnostic['sa_energy_agreement']):.2e} Eh",
            flush=True,
        )

    sa_links = _neighbor_links(database, record_ids, records, overlap)
    diagnostic_frames = [
        _diagnostic_frame(record, diagnostic)
        for record, diagnostic in zip(records, diagnostics)
    ]
    diagnostic_links = []
    for edge, (left, right) in enumerate(
        zip(diagnostic_frames[:-1], diagnostic_frames[1:])
    ):
        print(
            f"[bridge] diagnostic overlap {radii[edge]:.3f} -> "
            f"{radii[edge + 1]:.3f} A",
            flush=True,
        )
        diagnostic_links.append(overlap(left, right))
    diagnostic_links = np.asarray(diagnostic_links)

    anchor = int(np.argmin(np.abs(radii - 1.85)))
    anchor_states = np.asarray((0, 5, 1), dtype=int)
    if args.old_gauge.is_file():
        old = _load_npz(args.old_gauge)
        old_anchor = int(np.argmin(np.abs(old["radii"] - radii[anchor])))
        anchor_states = np.asarray(old["root_indices"][old_anchor], dtype=int)
    root_indices, selected_links = track_states(
        diagnostic_links,
        anchor=anchor,
        states=anchor_states,
    )
    sa_tracking_available = bool(
        np.all(anchor_states >= 0)
        and np.all(anchor_states < sa_links.shape[-1])
    )
    if sa_tracking_available:
        sa_root_indices, sa_selected_links = track_states(
            sa_links,
            anchor=anchor,
            states=anchor_states,
        )
        sa_singular = np.linalg.svd(sa_selected_links, compute_uv=False)
    else:
        sa_root_indices = np.full((len(radii), len(anchor_states)), -1, dtype=int)
        sa_selected_links = np.full(
            (len(radii) - 1, len(anchor_states), len(anchor_states)),
            np.nan,
            dtype=complex,
        )
        sa_singular = np.full((len(radii) - 1, len(anchor_states)), np.nan)
    all_diagnostic_energies = np.asarray(
        [diagnostic["energies"] for diagnostic in diagnostics]
    )
    selected_energies = np.asarray(
        [energy[roots] for energy, roots in zip(all_diagnostic_energies, root_indices)]
    )
    diagnostic_singular = np.linalg.svd(selected_links, compute_uv=False)
    gauge = _extended_gauge(
        radii,
        selected_links,
        selected_energies,
        args.old_gauge,
    )
    p_rotation = procrustes(gauge["links"])[0]
    p_rotation_defect = np.linalg.norm(
        p_rotation - np.eye(3), axis=(-2, -1)
    )

    outside = np.argwhere(root_indices >= 6)
    missing_channels = np.unique(outside[:, 1]) if len(outside) else np.empty(0, dtype=int)
    missing_points = [
        {
            "radius_angstrom": float(radii[point]),
            "channel": f"P{channel}",
            "diagnostic_root": int(root_indices[point, channel]),
        }
        for point, channel in outside
    ]
    max_spin = max(float(np.max(np.abs(item["spins"]))) for item in diagnostics)
    max_agreement = max(float(item["sa_energy_agreement"]) for item in diagnostics)
    gates = {
        "unchanged_sa6_protocol": all(
            np.asarray(record["energies"]).shape == (6,) for record in records
        ),
        "all_sa6_orbitals_relaxed": all(
            bool(record["orbital_relaxed"]) for record in records
        ),
        "all_diagnostic_roots_singlets": max_spin <= 1.0e-5,
        "diagnostic_first6_reproduce_sa6": max_agreement <= 1.0e-6,
        "tracked_channel_transport_to_outer_radius": bool(
            np.min(diagnostic_singular[:, -1]) >= args.continuity_threshold
        ),
        "P_links_positive": bool(np.max(p_rotation_defect) <= 1.0e-9),
    }
    transported = all(gates.values())

    coarse_bridge_minimum = np.nan
    if args.old_gauge.is_file():
        old = _load_npz(args.old_gauge)
        edge = int(np.argmin(np.abs(0.5 * (old["radii"][:-1] + old["radii"][1:]) - 1.90)))
        singular_key = (
            "tracked_singular_values"
            if "tracked_singular_values" in old
            else "diagnostic_singular_values"
        )
        coarse_bridge_minimum = float(old[singular_key][edge, -1])
    png, pdf = _plot(
        args.output,
        radii,
        all_diagnostic_energies,
        root_indices,
        selected_energies,
        diagnostic_singular,
        sa_singular,
        gauge["hamiltonian"],
        coarse_bridge_minimum,
        args.continuity_threshold,
    )
    data_path = args.output / "phenol_sa6_bridge_p_gauge.npz"
    np.savez_compressed(
        data_path,
        radii=radii,
        record_ids=np.asarray(record_ids),
        sources=np.asarray(sources),
        sa_energies=np.asarray([record["energies"] for record in records]),
        diagnostic_energies=all_diagnostic_energies,
        diagnostic_spins=np.asarray([item["spins"] for item in diagnostics]),
        sa_energy_agreement=np.asarray(
            [item["sa_energy_agreement"] for item in diagnostics]
        ),
        sa_links=sa_links,
        diagnostic_links=diagnostic_links,
        anchor=np.asarray(anchor),
        anchor_states=anchor_states,
        root_indices=root_indices,
        sa_root_indices=sa_root_indices,
        selected_energies=selected_energies,
        selected_links=selected_links,
        diagnostic_singular_values=diagnostic_singular,
        sa_singular_values=sa_singular,
        p_gauge=gauge["gauge"],
        p_links=gauge["links"],
        p_hamiltonian=gauge["hamiltonian"],
        combined_radii=gauge["combined_radii"],
        combined_p_gauge=gauge["combined_gauge"],
        combined_p_hamiltonian=gauge["combined_hamiltonian"],
    )
    database.update_run(
        run_id,
        "transported" if transported else "diagnosed",
        metadata={"gates": gates, "missing_points": missing_points},
    )
    summary = {
        "passed": transported,
        "gates": gates,
        "protocol": protocol,
        "grid_angstrom": radii,
        "step_angstrom": args.step,
        "sa_roots": 6,
        "diagnostic_roots": args.diagnostic_roots,
        "new_sa6_records": sources.count("calculated"),
        "reused_sa6_records": sources.count("database"),
        "maximum_diagnostic_spin_square": max_spin,
        "maximum_first6_energy_disagreement_hartree": max_agreement,
        "anchor_radius_angstrom": float(radii[anchor]),
        "anchor_states": anchor_states,
        "root_indices": root_indices,
        "missing_physical_channels": [f"P{channel}" for channel in missing_channels],
        "outside_sa6_window": missing_points,
        "minimum_diagnostic_selected_singular_value": float(
            np.min(diagnostic_singular[:, -1])
        ),
        "sa6_tracking_available": sa_tracking_available,
        "minimum_sa6_selected_singular_value": (
            float(np.min(sa_singular[:, -1]))
            if sa_tracking_available else None
        ),
        "old_1p85_to_1p95_selected_singular_value": coarse_bridge_minimum,
        "maximum_P_link_rotation_defect": float(np.max(p_rotation_defect)),
        "transported_to_radius_angstrom": float(radii[-1]) if transported else None,
        "next_step": (
            "sample sparse angular dependence for R > 1.85 A"
            if transported
            else "increase the diagnostic root window before angular sampling"
        ),
        "database": args.database,
        "database_stats": database.stats,
        "data": data_path,
        "figure": png,
        "figure_pdf": pdf,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    provider.close()
    database.close()


if __name__ == "__main__":
    main()
