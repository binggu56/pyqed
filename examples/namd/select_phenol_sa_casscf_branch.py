#!/usr/bin/env python3
"""Select the chemically intended phenol SA-CASSCF O--H orbital branch."""

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
from pyscf import fci

from phenol_sa_casscf_sequential import HARTREE_TO_EV, NCAS, NCORE, load_record
from phenol_sa_casscf_validate import ao_subspaces, lowdin_coefficients, molecule


DEFAULT_DISTANCES = [
    0.90,
    0.94,
    0.96994,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
    1.55,
    1.70,
    1.85,
    1.95,
    2.05,
    2.20,
    2.50,
    3.00,
]


def record_path(root, backend, direction, distance):
    return root / backend / direction / f"r{distance:.5f}.npz"


def natural_orbital_diagnostics(record, basis):
    ci = np.asarray(record["ci"])
    dm1 = sum(
        fci.direct_spin1.make_rdm1(state, NCAS, (5, 5)) for state in ci
    ) / len(ci)
    occupations, rotation = np.linalg.eigh(dm1)
    order = np.argsort(occupations)[::-1]
    occupations = occupations[order]
    active = np.asarray(record["mo_coeff"])[:, NCORE : NCORE + NCAS]
    natural_orbitals = active @ rotation[:, order]

    mol = molecule(record["geometry"], basis)
    lowdin = lowdin_coefficients(mol, natural_orbitals)
    pi_indices, oh_indices = ao_subspaces(mol)
    labels = mol.ao_labels(fmt=False)
    atom_indices = np.asarray([label[0] for label in labels], dtype=int)
    weights = {
        "pi": np.sum(np.abs(lowdin[pi_indices]) ** 2, axis=0),
        "oh": np.sum(np.abs(lowdin[oh_indices]) ** 2, axis=0),
        "oxygen": np.sum(np.abs(lowdin[atom_indices == 6]) ** 2, axis=0),
        "hydrogen": np.sum(np.abs(lowdin[atom_indices == 7]) ** 2, axis=0),
    }
    pair = np.argsort(weights["oh"])[-2:]
    pair = pair[np.argsort(occupations[pair])[::-1]]
    return occupations, weights, pair


def load_diagnostics(root, backend, direction, distances, basis):
    records = [
        load_record(record_path(root, backend, direction, distance))
        for distance in distances
    ]
    diagnostics = [natural_orbital_diagnostics(record, basis) for record in records]
    return records, diagnostics


def pair_array(diagnostics, key):
    values = []
    for occupations, weights, pair in diagnostics:
        source = occupations if key == "occupations" else weights[key]
        values.append(source[pair])
    return np.asarray(values)


def plot_selection(
    distances,
    forward_records,
    forward_diagnostics,
    reverse_records,
    reverse_diagnostics,
    active_overlap,
    figure_output,
):
    reverse_occupations = pair_array(reverse_diagnostics, "occupations")
    reverse_oxygen = pair_array(reverse_diagnostics, "oxygen")
    reverse_hydrogen = pair_array(reverse_diagnostics, "hydrogen")
    forward_oh = np.sum(pair_array(forward_diagnostics, "oh"), axis=1)
    reverse_oh = np.sum(pair_array(reverse_diagnostics, "oh"), axis=1)
    energy_advantage = (
        np.asarray([np.mean(record["energies"]) for record in forward_records])
        - np.asarray([np.mean(record["energies"]) for record in reverse_records])
    ) * HARTREE_TO_EV

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    axes[0, 0].plot(
        distances, reverse_occupations[:, 0], "o-", color="#0072B2", label="higher occupation"
    )
    axes[0, 0].plot(
        distances, reverse_occupations[:, 1], "s--", color="#D55E00", label="lower occupation"
    )
    axes[0, 0].set(ylabel="state-averaged occupation", title="Selected O–H natural-orbital pair", ylim=(-0.05, 2.05))
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        distances, np.max(reverse_hydrogen, axis=1), "o-", color="#009E73", label="maximum H weight"
    )
    axes[0, 1].plot(
        distances, np.max(reverse_oxygen, axis=1), "s--", color="#CC79A7", label="maximum O weight"
    )
    axes[0, 1].set(ylabel="Löwdin atomic weight", title="Bond-to-radical localization", ylim=(-0.02, 1.02))
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(distances, forward_oh, "o-", color="#D55E00", label="forward")
    axes[1, 0].plot(distances, reverse_oh, "s--", color="#0072B2", label="reverse (selected)")
    axes[1, 0].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="summed O–H AO character",
        title="Retention of the intended O–H pair",
        ylim=(-0.02, max(1.6, 1.05 * np.max(reverse_oh))),
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    energy_axis = axes[1, 1]
    overlap_axis = energy_axis.twinx()
    energy_line = energy_axis.plot(
        distances, energy_advantage, "o-", color="#6A3D9A", label="SA energy advantage"
    )[0]
    overlap_line = overlap_axis.plot(
        distances, active_overlap, "s--", color="#009E73", label="forward/reverse active overlap"
    )[0]
    energy_axis.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    energy_axis.set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel=r"$\bar E_{\rm forward}-\bar E_{\rm reverse}$ (eV)",
        title="Branch preference and merger",
    )
    overlap_axis.set(ylabel="minimum active-space singular value", ylim=(-0.02, 1.02))
    energy_axis.legend(
        [energy_line, overlap_line],
        [energy_line.get_label(), overlap_line.get_label()],
        frameon=False,
        fontsize=7,
        loc="center right",
    )

    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    overlap_axis.spines["top"].set_visible(False)

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    png = figure_output.with_suffix(".png")
    pdf = figure_output.with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_production_qualification_20260820"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_production_qualification_20260820/branch_selection"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_branch_selection"),
    )
    parser.add_argument("--distances", type=float, nargs="*", default=DEFAULT_DISTANCES)
    args = parser.parse_args()

    distances = np.asarray(sorted(args.distances), dtype=float)
    forward_records, forward_diagnostics = load_diagnostics(
        args.qualification_root, "pyscf", "forward", distances, args.basis
    )
    reverse_records, reverse_diagnostics = load_diagnostics(
        args.qualification_root, "pyscf", "reverse", distances, args.basis
    )
    qualification = json.loads(
        (args.qualification_root / "qualification_summary.json").read_text()
    )
    active_overlap = np.asarray(
        qualification["direction_active_overlap"]["pyscf"], dtype=float
    )

    anchor_index = int(np.argmin(np.abs(distances - 0.96994)))
    endpoint_index = int(np.argmax(distances))
    forward_anchor_oh = pair_array(forward_diagnostics, "oh")[anchor_index]
    reverse_anchor_oh = pair_array(reverse_diagnostics, "oh")[anchor_index]
    reverse_anchor_occ = pair_array(reverse_diagnostics, "occupations")[anchor_index]
    reverse_endpoint_occ = pair_array(reverse_diagnostics, "occupations")[endpoint_index]
    reverse_endpoint_o = pair_array(reverse_diagnostics, "oxygen")[endpoint_index]
    reverse_endpoint_h = pair_array(reverse_diagnostics, "hydrogen")[endpoint_index]
    reverse_continuity = float(
        qualification["cuts"]["pyscf"]["reverse"][
            "minimum_adjacent_active_singular_value"
        ]
    )
    forward_continuity = float(
        qualification["cuts"]["pyscf"]["forward"][
            "minimum_adjacent_active_singular_value"
        ]
    )
    merge_candidates = np.flatnonzero(active_overlap > 0.99)
    merge_distance = float(distances[merge_candidates[0]])
    forward_mean = float(np.mean(forward_records[anchor_index]["energies"]))
    reverse_mean = float(np.mean(reverse_records[anchor_index]["energies"]))

    criteria = {
        "equilibrium_bonding_antibonding_occupations": bool(
            reverse_anchor_occ[0] > 1.7 and reverse_anchor_occ[1] < 0.3
        ),
        "equilibrium_oh_pair_character": bool(np.min(reverse_anchor_oh) > 0.5),
        "dissociation_h_radical": bool(
            np.max(reverse_endpoint_h) > 0.8
            and np.min(np.abs(reverse_endpoint_occ - 1.0)) < 0.25
        ),
        "dissociation_o_radical": bool(
            np.max(reverse_endpoint_o) > 0.8
            and np.any((reverse_endpoint_occ > 0.8) & (reverse_endpoint_occ < 1.8))
        ),
        "adjacent_active_continuity": bool(reverse_continuity > 0.8),
        "lower_state_averaged_anchor_energy": bool(reverse_mean < forward_mean),
    }
    selected = "reverse" if all(criteria.values()) else "undetermined"
    if selected == "undetermined":
        raise RuntimeError(f"chemical branch criteria did not agree: {criteria}")

    args.output.mkdir(parents=True, exist_ok=True)
    selected_records = {
        backend: [
            str(record_path(args.qualification_root, backend, selected, distance))
            for distance in distances
        ]
        for backend in ("pyscf", "pyqed")
    }
    np.savez_compressed(
        args.output / "phenol_sa6_selected_branch_diagnostics.npz",
        distances_angstrom=distances,
        natural_occupations=np.asarray([item[0] for item in reverse_diagnostics]),
        oh_pair_occupations=pair_array(reverse_diagnostics, "occupations"),
        oh_pair_character=pair_array(reverse_diagnostics, "oh"),
        oh_pair_oxygen_weight=pair_array(reverse_diagnostics, "oxygen"),
        oh_pair_hydrogen_weight=pair_array(reverse_diagnostics, "hydrogen"),
        energies_pyscf=np.asarray([record["energies"] for record in reverse_records]),
        energies_pyqed=np.asarray(
            [
                load_record(record_path(args.qualification_root, "pyqed", selected, distance))["energies"]
                for distance in distances
            ]
        ),
    )
    png, pdf = plot_selection(
        distances,
        forward_records,
        forward_diagnostics,
        reverse_records,
        reverse_diagnostics,
        active_overlap,
        args.figure_output,
    )
    summary = {
        "method": qualification["method"],
        "selected_branch": selected,
        "selection_applies_to": ["pyscf", "pyqed"],
        "criteria": criteria,
        "equilibrium_distance_angstrom": float(distances[anchor_index]),
        "equilibrium_oh_pair_occupations": reverse_anchor_occ.tolist(),
        "equilibrium_oh_pair_character": reverse_anchor_oh.tolist(),
        "forward_equilibrium_oh_pair_character": forward_anchor_oh.tolist(),
        "dissociation_distance_angstrom": float(distances[endpoint_index]),
        "dissociation_oh_pair_occupations": reverse_endpoint_occ.tolist(),
        "dissociation_oxygen_weights": reverse_endpoint_o.tolist(),
        "dissociation_hydrogen_weights": reverse_endpoint_h.tolist(),
        "forward_minimum_adjacent_active_overlap": forward_continuity,
        "reverse_minimum_adjacent_active_overlap": reverse_continuity,
        "orbital_branch_merge_distance_angstrom": merge_distance,
        "reverse_state_averaged_energy_advantage_at_equilibrium_ev": (
            forward_mean - reverse_mean
        )
        * HARTREE_TO_EV,
        "selected_records": selected_records,
        "diagnostics_npz": str(
            args.output / "phenol_sa6_selected_branch_diagnostics.npz"
        ),
        "figure": str(png),
        "figure_pdf": str(pdf),
        "remaining_requirement": (
            "Align the six CI roots by overlap before fitting a multistate Hamiltonian."
        ),
    }
    summary_path = args.output / "phenol_sa6_selected_branch.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
