#!/usr/bin/env python3
"""Locate gauge-invariant failures of the saved H3+ MACE Hamiltonian."""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr.database import _load_record
from pyqed.ml import MACE
from pyqed.units import au2ev

from examples.namd.plot_h3plus_hamiltonian_cuts import geometry_map


root = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=root / "h3plus_fci_state_leakage_curvilinear_expanded.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "physical_s3_quotient_mace_y_curvilinear_expanded_abinitio.pt",
    )
    parser.add_argument("--database", type=Path, default=root / "electronic.sqlite")
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path("/private/tmp/h3plus_fci_augccpvdz_physical")
        / "h3plus_fci_initial_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "h3plus_hamiltonian_failure_map",
    )
    parser.add_argument("--gate", type=float, default=1.5e-3, help="gate in hartree")
    return parser.parse_args()


def database_index(database):
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT object_path, metadata, specification FROM records"
    ).fetchall()
    connection.close()
    index = {}
    for object_path, metadata, specification in rows:
        coordinate = tuple(
            np.round(json.loads(metadata).get("coordinates", ()), 12)
        )
        if len(coordinate) != 3:
            continue
        geometry = np.asarray(json.loads(specification)["geometry"], dtype=float)
        index.setdefault(coordinate, []).append((object_path, geometry))
    return index


def load_energies(database, index, coordinate, geometry):
    key = tuple(np.round(coordinate, 12))
    candidates = index.get(key, ())
    if not candidates:
        raise KeyError(f"no cached electronic record at {coordinate}")
    target = geometry(coordinate)
    object_path, stored_geometry = min(
        candidates,
        key=lambda item: np.linalg.norm(item[1] - target),
    )
    mismatch = float(np.linalg.norm(stored_geometry - target))
    if mismatch > 2.0e-6:
        raise KeyError(
            f"cached geometry at {coordinate} belongs to a different chart "
            f"(mismatch {mismatch:.3e} bohr)"
        )
    record = _load_record(database.parent / object_path)
    return np.asarray(record[1], dtype=float)


def main():
    args = parse_args()
    preparation = json.loads(args.preparation.read_text())
    geometry = geometry_map(float(preparation["equilibrium_bond_bohr"]))
    coordinates = np.load(args.coordinates)["gap_coordinates"]
    model = MACE.load(args.checkpoint, geometry, distill=False)
    predicted = np.linalg.eigvalsh(model.neural_energy.predict(coordinates))

    index = database_index(args.database)
    reference = np.asarray(
        [
            load_energies(args.database, index, coordinate, geometry)[1:3]
            for coordinate in coordinates
        ]
    )
    origin_reference = load_energies(
        args.database, index, np.zeros(3), geometry
    )[1:3]
    origin_prediction = np.linalg.eigvalsh(
        model.neural_energy.predict(np.zeros((1, 3)))
    )[0]
    reference -= np.mean(origin_reference - origin_prediction)

    delta_levels = predicted - reference
    delta_h0 = np.mean(delta_levels, axis=1)
    delta_rho = 0.5 * np.diff(delta_levels, axis=1)[:, 0]
    error = np.linalg.norm(delta_levels, axis=1)
    error_mev = error * au2ev * 1000.0
    trace_mev = np.abs(delta_h0) * au2ev * 1000.0
    splitting_mev = np.abs(delta_rho) * au2ev * 1000.0
    gate_mev = float(args.gate) * au2ev * 1000.0
    widths = np.sqrt(
        np.diag(np.asarray(preparation["probability_covariance_bohr2"], dtype=float))
    )
    scaled_radius = np.linalg.norm(coordinates / widths, axis=1)
    branching_radius = np.linalg.norm(coordinates[:, 1:], axis=1)
    order = np.argsort(error_mev)[::-1]
    failing = error_mev > gate_mev

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, panels = plt.subplots(2, 2, figsize=(8.0, 6.6), constrained_layout=True)
    color_limit = max(gate_mev, float(np.max(error_mev)))
    scatter_options = {
        "c": error_mev,
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": color_limit,
        "s": 34,
        "edgecolor": "black",
        "linewidth": 0.45,
    }

    map_qs = panels[0, 0].scatter(
        coordinates[:, 0], branching_radius, **scatter_options
    )
    panels[0, 0].scatter(
        coordinates[failing, 0], branching_radius[failing],
        marker="x", s=58, color="#D55E00", linewidth=1.3,
    )
    panels[0, 0].set(
        xlabel=r"$Q_s$ (bohr)",
        ylabel=r"$Q_E=(Q_x^2+Q_y^2)^{1/2}$ (bohr)",
        title="(a) Breathing–branching plane",
    )

    panels[0, 1].scatter(
        coordinates[:, 1], coordinates[:, 2], **scatter_options
    )
    panels[0, 1].scatter(
        coordinates[failing, 1], coordinates[failing, 2],
        marker="x", s=58, color="#D55E00", linewidth=1.3,
        label="guaranteed gate failure",
    )
    panels[0, 1].set(
        xlabel=r"$Q_x$ (bohr)",
        ylabel=r"$Q_y$ (bohr)",
        title="(b) Branching plane (S3 quotient)",
    )
    panels[0, 1].legend(frameon=False, loc="upper left")

    panels[1, 0].scatter(
        scaled_radius,
        error_mev,
        c=error_mev,
        cmap="viridis",
        vmin=0.0,
        vmax=color_limit,
        s=34,
        edgecolor="black",
        linewidth=0.45,
    )
    panels[1, 0].axhline(
        gate_mev,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=fr"Hamiltonian gate: {gate_mev:.1f} meV",
    )
    panels[1, 0].set(
        xlabel=r"packet-scaled radius $|q/\sigma|$",
        ylabel=r"$\|\Delta\lambda\|_2$ (meV)",
        title="(c) Error moves to the outer chart",
    )
    panels[1, 0].legend(frameon=False, loc="upper left")

    top = order[:6]
    positions = np.arange(len(top))
    width = 0.34
    panels[1, 1].bar(
        positions - width / 2,
        np.sqrt(2.0) * trace_mev[top],
        width,
        color="#0072B2",
        label=r"trace: $\sqrt{2}|\Delta h_0|$",
    )
    panels[1, 1].bar(
        positions + width / 2,
        np.sqrt(2.0) * splitting_mev[top],
        width,
        color="#E69F00",
        label=r"splitting: $\sqrt{2}|\Delta\rho|$",
    )
    panels[1, 1].plot(
        positions,
        error_mev[top],
        "ko",
        ms=4.0,
        label=r"total $\|\Delta\lambda\|_2$",
    )
    panels[1, 1].axhline(
        gate_mev, color="#D55E00", linestyle="--", linewidth=1.2
    )
    panels[1, 1].set(
        xticks=positions,
        xticklabels=[str(number + 1) for number in range(len(top))],
        xlabel="worst validation point",
        ylabel="error contribution (meV)",
        title="(d) Trace versus level splitting",
    )
    panels[1, 1].legend(frameon=False, loc="upper right")

    for panel in panels.flat:
        panel.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(map_qs, ax=panels[0, :], shrink=0.88, pad=0.02)
    colorbar.set_label(r"gauge-invariant error $\|\Delta\lambda\|_2$ (meV)")
    figure.suptitle("Where the H3+ MACE Hamiltonian fails")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        coordinates=coordinates,
        packet_scaled_radius=scaled_radius,
        reference_levels_hartree=reference,
        fitted_levels_hartree=predicted,
        level_error_hartree=error,
        trace_error_hartree=np.abs(delta_h0),
        half_splitting_error_hartree=np.abs(delta_rho),
        guaranteed_failure=failing,
        gate_hartree=np.asarray(args.gate),
    )

    print(f"guaranteed failures: {np.count_nonzero(failing)}/{len(coordinates)}")
    print(
        "inside 5 sigma: "
        f"max={np.max(error_mev[scaled_radius <= 5.0]):.2f} meV; "
        f"RMS={np.sqrt(np.mean(error_mev[scaled_radius <= 5.0] ** 2)):.2f} meV"
    )
    print("rank  Qs       Qx       Qy       |q/sigma|  error/meV")
    for rank, index_value in enumerate(top, start=1):
        qs, qx, qy = coordinates[index_value]
        print(
            f"{rank:>4} {qs:>8.4f} {qx:>8.4f} {qy:>8.4f} "
            f"{scaled_radius[index_value]:>10.3f} {error_mev[index_value]:>10.2f}"
        )
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".npz"))


if __name__ == "__main__":
    main()
