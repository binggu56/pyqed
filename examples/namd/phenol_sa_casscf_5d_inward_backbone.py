#!/usr/bin/env python3
"""Add the existing inward planar P-gauge backbone to the phenol 5D data."""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HARTREE_TO_EV = au2ev


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _planar_point(coordinates, radius, theta, tolerance=1.0e-8):
    coordinates = np.asarray(coordinates, dtype=float)
    target = np.asarray((radius, 0.0, theta, 0.0, 0.0))
    matches = np.flatnonzero(
        np.max(np.abs(coordinates - target), axis=1) <= float(tolerance)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one planar 5D point at R_OH={float(radius):.8f} A"
        )
    return int(matches[0])


def _source_point(radii, radius, tolerance=1.0e-8):
    matches = np.flatnonzero(np.abs(np.asarray(radii) - float(radius)) <= tolerance)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one inward-backbone point at R_OH={float(radius):.8f} A"
        )
    return int(matches[0])


def _reference_theta(coordinates):
    values = np.unique(np.asarray(coordinates, dtype=float)[:, 2])
    return float(values[np.argmin(np.abs(values - np.deg2rad(108.8)))])


def augment_inward_backbone(
    base,
    inward,
    *,
    minimum_radius=0.75,
    anchor_radius=0.95,
    terminal_radius=1.15,
):
    """Return a minimal MACE-Y artifact with a fine inward planar chain.

    The inward P gauge is transformed once at the shared 0.95 A anchor.  Its
    missing planar points and consecutive links are then added through the
    existing 1.15 A 5D point.  The original 5D field and links are untouched.
    """

    coordinates = np.asarray(base["coordinates"], dtype=float)
    hamiltonians = np.asarray(base["p_hamiltonian"], dtype=complex)
    pairs = np.asarray(base["pairs"], dtype=int)
    links = np.asarray(base["p_links"], dtype=complex)
    radii = np.asarray(inward["radii"], dtype=float)
    source_h = np.asarray(inward["p_hamiltonian"], dtype=complex)
    source_links = np.asarray(inward["p_links"], dtype=complex)
    source_gauges = np.asarray(inward["p_gauge"], dtype=complex)
    if len(source_links) != len(radii) - 1:
        raise ValueError("the inward P-gauge links must join consecutive radii")

    theta = _reference_theta(coordinates)
    anchor = _planar_point(coordinates, anchor_radius, theta)
    terminal = _planar_point(coordinates, terminal_radius, theta)
    source_anchor = _source_point(radii, anchor_radius)
    source_terminal = _source_point(radii, terminal_radius)
    if source_terminal <= source_anchor:
        raise ValueError("terminal_radius must follow anchor_radius")
    source_to_base = (
        np.asarray(base["gauges"])[anchor].conj().T
        @ source_gauges[source_anchor]
    )
    unitary_defect = float(
        np.linalg.norm(source_to_base.conj().T @ source_to_base - np.eye(3))
    )
    if unitary_defect > 1.0e-8:
        raise RuntimeError("the anchor gauge transformation is not unitary")

    selected = np.flatnonzero(
        (radii >= float(minimum_radius) - 1.0e-10)
        & (radii <= float(terminal_radius) + 1.0e-10)
    )
    if not len(selected) or selected[0] > source_anchor or selected[-1] < source_terminal:
        raise ValueError("the inward artifact does not span the requested radial range")

    new_coordinates = list(coordinates)
    new_hamiltonians = list(hamiltonians)
    new_gauges = list(np.asarray(base["gauges"], dtype=complex))
    point_ids = {
        round(float(coordinate[0]), 10): index
        for index, coordinate in enumerate(coordinates)
        if np.allclose(coordinate[1:], (0.0, theta, 0.0, 0.0), atol=1.0e-8)
    }
    appended_radii = []
    for source in selected:
        radius = float(radii[source])
        key = round(radius, 10)
        if key in point_ids:
            continue
        point_ids[key] = len(new_coordinates)
        new_coordinates.append(np.asarray((radius, 0.0, theta, 0.0, 0.0)))
        new_hamiltonians.append(
            source_to_base @ source_h[source] @ source_to_base.conj().T
        )
        new_gauges.append(source_gauges[source] @ source_to_base.conj().T)
        appended_radii.append(radius)

    new_pairs = list(pairs)
    new_links = list(links)
    new_pair_axes = list(np.asarray(base["pair_axes"], dtype=int))
    added_link_singular_values = []
    for left_source, right_source in zip(selected[:-1], selected[1:]):
        left_radius = float(radii[left_source])
        right_radius = float(radii[right_source])
        left = point_ids[round(left_radius, 10)]
        right = point_ids[round(right_radius, 10)]
        link = (
            source_to_base
            @ source_links[left_source]
            @ source_to_base.conj().T
        )
        new_pairs.append((left, right))
        new_links.append(link)
        new_pair_axes.append(0)
        added_link_singular_values.extend(np.linalg.svd(link, compute_uv=False))

    added_count = len(appended_radii)
    added_links = len(selected) - 1
    energy_holdout = np.concatenate(
        (np.asarray(base["energy_holdout"], dtype=bool), np.zeros(added_count, dtype=bool))
    )
    link_holdout = np.concatenate(
        (np.asarray(base["link_holdout"], dtype=bool), np.zeros(added_links, dtype=bool))
    )
    reflection = np.asarray(base["reflection"], dtype=complex)
    appended_h = np.asarray(new_hamiltonians[len(coordinates) :])
    appended_l = np.asarray(new_links[len(links) :])
    reflection_h = float(
        np.max(
            np.linalg.norm(
                appended_h - reflection.conj().T @ appended_h @ reflection,
                axis=(1, 2),
            )
        )
    )
    reflection_l = float(
        np.max(
            np.linalg.norm(
                appended_l - reflection.conj().T @ appended_l @ reflection,
                axis=(1, 2),
            )
        )
    )
    anchor_target = (
        source_to_base @ source_h[source_anchor] @ source_to_base.conj().T
    )
    terminal_target = (
        source_to_base @ source_h[source_terminal] @ source_to_base.conj().T
    )
    anchor_defect = float(np.linalg.norm(hamiltonians[anchor] - anchor_target))
    terminal_spectral_defect = float(
        np.max(
            np.abs(
                np.linalg.eigvalsh(hamiltonians[terminal])
                - np.linalg.eigvalsh(terminal_target)
            )
        )
    )
    summary = {
        "passed": bool(
            np.isclose(min(appended_radii), minimum_radius)
            and anchor_defect <= 1.0e-6
            and terminal_spectral_defect <= 1.0e-6
            and min(added_link_singular_values) >= 0.90
            and reflection_h <= 2.0e-7
            and reflection_l <= 2.0e-7
        ),
        "base_points": int(len(coordinates)),
        "augmented_points": int(len(new_coordinates)),
        "appended_radii_angstrom": appended_radii,
        "base_links": int(len(links)),
        "augmented_links": int(len(new_links)),
        "added_radial_links": int(added_links),
        "anchor_radius_angstrom": float(anchor_radius),
        "terminal_radius_angstrom": float(terminal_radius),
        "anchor_hamiltonian_defect_hartree": anchor_defect,
        "terminal_spectral_defect_hartree": terminal_spectral_defect,
        "anchor_unitary_defect": unitary_defect,
        "minimum_added_link_singular_value": float(min(added_link_singular_values)),
        "maximum_added_link_singular_value": float(max(added_link_singular_values)),
        "maximum_appended_reflection_hamiltonian_defect": reflection_h,
        "maximum_appended_reflection_link_defect": reflection_l,
    }
    artifact = {
        "coordinates": np.asarray(new_coordinates),
        "p_hamiltonian": np.asarray(new_hamiltonians),
        "gauges": np.asarray(new_gauges),
        "pairs": np.asarray(new_pairs, dtype=int),
        "p_links": np.asarray(new_links),
        "pair_axes": np.asarray(new_pair_axes, dtype=int),
        "energy_holdout": energy_holdout,
        "link_holdout": link_holdout,
        "reflection": reflection,
        "coordinate_parities": np.asarray(base["coordinate_parities"]),
        "coordinate_scales": np.asarray(base["coordinate_scales"]),
        "modes": np.asarray(base["modes"]),
        "source_is_inward_backbone": np.concatenate(
            (np.zeros(len(coordinates), dtype=bool), np.ones(added_count, dtype=bool))
        ),
        "source_to_base_gauge": source_to_base,
    }
    return artifact, summary


def plot_diagnostics(output, artifact, summary):
    coordinates = artifact["coordinates"]
    theta = _reference_theta(coordinates)
    planar = np.all(
        np.isclose(
            coordinates[:, 1:],
            (0.0, theta, 0.0, 0.0),
            atol=1.0e-8,
        ),
        axis=1,
    )
    order = np.argsort(coordinates[planar, 0])
    radii = coordinates[planar, 0][order]
    energies = np.linalg.eigvalsh(artifact["p_hamiltonian"][planar][order])
    energies = (energies - np.min(energies)) * HARTREE_TO_EV
    inward = artifact["source_is_inward_backbone"][planar][order]

    figure, panels = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for state, color in enumerate(colors):
        panels[0].plot(radii, energies[:, state], "-", color=color, lw=1.1)
        panels[0].scatter(
            radii[~inward], energies[~inward, state], s=24,
            facecolors="none", edgecolors=color,
        )
        panels[0].scatter(
            radii[inward], energies[inward, state], s=25,
            color=color, marker="x", label=f"P{state}" if state == 0 else None,
        )
    panels[0].axvline(0.95, color="0.35", ls="--", lw=1.0)
    panels[0].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="relative energy (eV)",
        title="Fine inward planar backbone",
    )
    panels[0].text(
        0.03, 0.94, "x: reused inward records\no: original 5D points",
        transform=panels[0].transAxes, va="top", fontsize=8,
    )
    singular = np.asarray(
        [
            np.linalg.svd(link, compute_uv=False)
            for link in artifact["p_links"][-summary["added_radial_links"] :]
        ]
    )
    link_radii = np.linspace(0.75, 1.15, len(singular), endpoint=False)
    for state, color in enumerate(colors):
        panels[1].plot(
            link_radii, singular[:, state], "o-", color=color, ms=3,
            label=fr"$\sigma_{state + 1}$",
        )
    panels[1].set(
        xlabel=r"left $R_{OH}$ (angstrom)", ylabel="link singular value",
        ylim=(0.95, 1.005), title="Inward overlap quality",
    )
    panels[1].legend(frameon=False, fontsize=8)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    png = output / "phenol_sa6_5d_inward_backbone.png"
    figure.savefig(png, dpi=300)
    figure.savefig(output / "phenol_sa6_5d_inward_backbone.pdf")
    plt.close(figure)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_5d_pilot_20260822/"
            "phenol_sa6_5d_p_gauge.npz"
        ),
    )
    parser.add_argument(
        "--inward", type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_p_gauge_inward_20260821/"
            "phenol_sa6_tracked3_p_gauge.npz"
        ),
    )
    parser.add_argument("--minimum-radius", type=float, default=0.75)
    parser.add_argument("--anchor-radius", type=float, default=0.95)
    parser.add_argument("--terminal-radius", type=float, default=1.15)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_5d_inward_20260822"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact, summary = augment_inward_backbone(
        _load(args.base),
        _load(args.inward),
        minimum_radius=args.minimum_radius,
        anchor_radius=args.anchor_radius,
        terminal_radius=args.terminal_radius,
    )
    data_path = args.output / "phenol_sa6_5d_p_gauge_inward.npz"
    np.savez_compressed(data_path, **artifact)
    figure = plot_diagnostics(args.output, artifact, summary)
    summary.update(
        {
            "base": str(args.base),
            "inward": str(args.inward),
            "data": str(data_path),
            "figure": str(figure),
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("the augmented inward 5D artifact failed qualification")


if __name__ == "__main__":
    main()
