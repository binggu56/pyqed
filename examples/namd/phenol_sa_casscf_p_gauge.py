#!/usr/bin/env python3
"""Test an overlap-tracked three-channel P gauge for phenol SA-CASSCF data."""

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
    PhenolReflectionSymmetry,
    SamplingSymmetryImage,
    phenol_sa6_protocol,
)
from pyqed.ldr.database import canonical_json
from pyqed.ldr.overlap import positive_link_gauge, procrustes, track_states
from pyqed.models.phenol_coordinates import PhenolReactiveChart


HARTREE_TO_EV = au2ev


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


def _planar_records(database, protocol):
    rows = []
    protocol_json = canonical_json(protocol)
    equilibrium_bend = float(PhenolReactiveChart().equilibrium[2])
    for entry in database.entries():
        specification = entry["specification"]
        if canonical_json(specification.get("protocol")) != protocol_json:
            continue
        geometry = np.asarray(specification["geometry"], dtype=float)
        oh = geometry[7] - geometry[6]
        radius = float(np.linalg.norm(oh))
        torsion = float(np.arctan2(oh[2], oh[1]))
        bend = float(np.arccos(np.clip(-oh[0] / radius, -1.0, 1.0)))
        if abs(torsion) <= 1.0e-10 and abs(bend - equilibrium_bend) <= 1.0e-8:
            rows.append((radius, entry, database.get(specification)))
    rows.sort(key=lambda item: item[0])
    return rows


def _neighbor_links(database, rows, overlap):
    blocks = []
    for left, right in zip(rows[:-1], rows[1:]):
        block = database.get_overlap(left[1]["id"], right[1]["id"], overlap.protocol)
        if block is None:
            block = overlap(left[2], right[2])
            database.put_overlap(
                left[1]["id"],
                right[1]["id"],
                overlap.protocol,
                block,
                metadata={"diagnostic": "planar P-gauge state tracking"},
            )
        blocks.append(block)
    return np.asarray(blocks)


def _reflection_representations(database, rows, overlap, protocol):
    symmetry = PhenolReflectionSymmetry()
    representation_protocol = {
        "base": overlap.protocol,
        "operator": "sigma_xy",
        "version": 1,
    }
    blocks = []
    for radius, entry, record in rows:
        block = database.get_overlap(
            entry["id"], entry["id"], representation_protocol
        )
        if block is None:
            reflected = symmetry.transform_record(
                record,
                SamplingSymmetryImage((radius, 0.0), symmetry.operation),
                representative_geometry=record["geometry"],
                requested_geometry=record["geometry"],
                protocol=protocol,
            )
            block = overlap(record, reflected)
            database.put_overlap(
                entry["id"],
                entry["id"],
                representation_protocol,
                block,
                metadata={"operator": "sigma_xy"},
            )
        blocks.append(block)
    return np.asarray(blocks)


def _continuous_component(radii, edge_minimum, anchor, threshold):
    lower = int(anchor)
    while lower > 0 and edge_minimum[lower - 1] >= threshold:
        lower -= 1
    upper = int(anchor)
    while upper < len(radii) - 1 and edge_minimum[upper] >= threshold:
        upper += 1
    return lower, upper


def _plot(
    output,
    radii,
    energies,
    root_indices,
    tracked_singular,
    ordered_singular,
    p_hamiltonian,
    valid,
    threshold,
):
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, 3))
    anchor = int(np.argmin(np.abs(radii - 0.96994)))
    reference = float(energies[anchor, 0])
    figure, panels = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    for channel, color in enumerate(colors):
        panels[0, 0].plot(
            radii,
            (energies[:, channel] - reference) * HARTREE_TO_EV,
            "o-",
            color=color,
            ms=3.8,
            lw=1.1,
            label=f"tracked channel {channel}",
        )
        panels[0, 1].step(
            radii,
            root_indices[:, channel],
            where="mid",
            color=color,
            lw=1.2,
            label=f"channel {channel}",
        )
    panels[0, 0].set(
        ylabel=r"$E-E_0(R_{eq})$ (eV)",
        title="Equilibrium-selected physical branches",
    )
    panels[0, 1].set(
        ylabel="energy-ordered SA(6) root index",
        yticks=range(6),
        title="Maximum-overlap root tracking",
    )
    panels[0, 0].legend(fontsize=7.5)
    panels[0, 1].legend(fontsize=7.5)

    midpoint = 0.5 * (radii[:-1] + radii[1:])
    panels[1, 0].semilogy(
        midpoint,
        tracked_singular[:, -1],
        "o-",
        color="#0072B2",
        label="tracked three-channel subspace",
    )
    panels[1, 0].semilogy(
        midpoint,
        ordered_singular[:, -1],
        "s--",
        color="#D55E00",
        label="energy-ordered lowest three",
    )
    panels[1, 0].axhline(threshold, color="0.35", ls=":", lw=1.0)
    panels[1, 0].set(
        ylabel=r"minimum $\sigma(S_{i,i+1})$",
        ylim=(1.0e-7, 1.1),
        title="Selected-projector continuity",
    )
    panels[1, 0].legend(fontsize=7.5)

    selected_radii = radii[valid]
    selected_hamiltonian = p_hamiltonian[valid]
    shifted = (
        selected_hamiltonian
        - reference * np.eye(selected_hamiltonian.shape[-1])[None]
    ) * HARTREE_TO_EV
    for channel, color in enumerate(colors):
        panels[1, 1].plot(
            selected_radii,
            shifted[:, channel, channel].real,
            "-",
            color=color,
            lw=1.3,
            label=rf"$\bar H_{{{channel}{channel}}}$",
        )
    for number, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        panels[1, 1].plot(
            selected_radii,
            shifted[:, left, right].real,
            "--",
            color=colors[number],
            lw=1.0,
            label=rf"$\bar H_{{{left}{right}}}$",
        )
    panels[1, 1].set(
        ylabel=r"$P$-gauge Hamiltonian (eV)",
        title="Continuous component containing equilibrium",
    )
    panels[1, 1].legend(fontsize=7.0, ncol=2)
    for panel in panels.flat:
        panel.set_xlabel(r"$R_{OH}$ ($\AA$)")
        panel.grid(alpha=0.2, which="both")
    figure.suptitle("Phenol three-channel overlap tracking and positive-link gauge")
    png = output / "phenol_sa6_tracked3_p_gauge.png"
    pdf = output / "phenol_sa6_tracked3_p_gauge.pdf"
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
        default=Path("/private/tmp/phenol_sa6_p_gauge_20260820"),
    )
    parser.add_argument("--continuity-threshold", type=float, default=0.90)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    database = ElectronicDatabase(args.database)
    rows = _planar_records(database, protocol)
    radii = np.asarray([item[0] for item in rows])
    all_energies = np.asarray([item[2]["energies"] for item in rows])
    full_links = _neighbor_links(database, rows, overlap)
    reflection = _reflection_representations(
        database, rows, overlap, protocol
    )
    anchor = int(np.argmin(np.abs(radii - 0.96994)))
    root_indices, selected_links = track_states(
        full_links, anchor=anchor, states=(0, 1, 2)
    )
    selected_energies = np.asarray(
        [all_energies[index, roots] for index, roots in enumerate(root_indices)]
    )
    gauges, p_links = positive_link_gauge(selected_links, anchor)
    diagonal = np.asarray([np.diag(values) for values in selected_energies])
    p_hamiltonian = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauges.conj(),
        diagonal,
        gauges,
        optimize=True,
    )
    p_hamiltonian = 0.5 * (
        p_hamiltonian + p_hamiltonian.swapaxes(-1, -2).conj()
    )
    tracked_singular = np.linalg.svd(selected_links, compute_uv=False)
    ordered_singular = np.linalg.svd(full_links[:, :3, :3], compute_uv=False)
    edge_minimum = tracked_singular[:, -1]
    lower, upper = _continuous_component(
        radii, edge_minimum, anchor, args.continuity_threshold
    )
    valid = slice(lower, upper + 1)
    parity_diagonal = np.real(
        np.diagonal(reflection, axis1=-2, axis2=-1)
    )
    tracked_parity = np.asarray(
        [parity_diagonal[index, roots] for index, roots in enumerate(root_indices)]
    )
    parity_labels = np.where(tracked_parity >= 0.0, "A'", "A''")
    p_rotation = procrustes(p_links)[0]
    p_rotation_defect = np.linalg.norm(
        p_rotation - np.eye(3), axis=(-2, -1)
    )
    failure_edges = np.flatnonzero(edge_minimum < args.continuity_threshold)
    gates = {
        "tracked_subspace_continuous_to_1_85_angstrom": bool(
            radii[upper] >= 1.85 - 1.0e-8
        ),
        "tracked_subspace_continuous_to_3_00_angstrom": bool(
            radii[upper] >= 3.00 - 1.0e-8
        ),
        "P_links_positive_on_continuous_component": bool(
            np.max(p_rotation_defect[lower:upper]) <= 1.0e-10
        ),
        "reflection_characters_pure": bool(
            np.min(np.abs(tracked_parity)) >= 0.99
        ),
    }
    png, pdf = _plot(
        args.output,
        radii,
        selected_energies,
        root_indices,
        tracked_singular,
        ordered_singular,
        p_hamiltonian,
        valid,
        args.continuity_threshold,
    )
    data_path = args.output / "phenol_sa6_tracked3_p_gauge.npz"
    np.savez_compressed(
        data_path,
        radii=radii,
        all_energies=all_energies,
        full_links=full_links,
        anchor=np.asarray(anchor),
        root_indices=root_indices,
        selected_energies=selected_energies,
        selected_links=selected_links,
        tracked_singular_values=tracked_singular,
        ordered_lowest3_singular_values=ordered_singular,
        reflection_representations=reflection,
        tracked_parity=tracked_parity,
        p_gauge=gauges,
        p_links=p_links,
        p_hamiltonian=p_hamiltonian,
    )
    summary = {
        "ready_for_full_domain_generation": all(gates.values()),
        "gates": gates,
        "anchor_radius_angstrom": float(radii[anchor]),
        "anchor_states": [0, 1, 2],
        "root_indices": root_indices,
        "tracked_reflection_labels": parity_labels,
        "continuous_component_angstrom": [
            float(radii[lower]),
            float(radii[upper]),
        ],
        "continuous_component_minimum_singular_value": float(
            np.min(edge_minimum[lower:upper])
        ),
        "full_domain_minimum_singular_value": float(np.min(edge_minimum)),
        "energy_ordered_lowest3_minimum_singular_value": float(
            np.min(ordered_singular[:, -1])
        ),
        "failure_edges": [
            {
                "left_radius": float(radii[edge]),
                "right_radius": float(radii[edge + 1]),
                "minimum_singular_value": float(edge_minimum[edge]),
            }
            for edge in failure_edges
        ],
        "maximum_P_link_rotation_defect_on_continuous_component": float(
            np.max(p_rotation_defect[lower:upper])
        ),
        "maximum_reflection_offdiagonal": float(
            np.max(
                np.abs(
                    reflection
                    - np.asarray([np.diag(np.diag(block)) for block in reflection])
                )
            )
        ),
        "figure": str(png),
        "figure_pdf": str(pdf),
        "data": str(data_path),
        "database": str(args.database),
        "database_stats": database.stats,
        "new_quantum_chemistry_calculations": 0,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    database.close()
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
