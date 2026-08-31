#!/usr/bin/env python3
"""Build and diagnose a three-patch four-state SO2 Procrustes atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.so2_procrustes_gauge import (
    direct_reference_overlaps,
    local_hamiltonian,
    reference_index,
    stitch,
    stitch_upper,
)
from pyqed.ldr.overlap import procrustes, unpack


def load_or_build(path, point_cache, shape, anchor, nstates, workers, reuse):
    if reuse and path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["overlaps"])
    overlaps = direct_reference_overlaps(
        point_cache,
        shape,
        anchor,
        nstates,
        workers=workers,
    )
    np.savez(path, overlaps=overlaps, center=np.asarray(anchor))
    return overlaps


def link_arrays(shape, links, gauge):
    output = []
    for axis, size in enumerate(shape):
        edge_shape = list(shape)
        edge_shape[axis] = size - 1
        values = np.empty((*edge_shape, gauge.shape[-1], gauge.shape[-1]), complex)
        for left in np.ndindex(tuple(edge_shape)):
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            values[left] = gauge[left].conj().T @ links[(axis, left)] @ gauge[right]
        output.append(values)
    return tuple(output)


def central_theta(values, center):
    return values[center[0], :, center[2]]


def step_norm(values, center):
    cut = central_theta(values, center)
    return np.linalg.norm(np.diff(cut, axis=0), axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        type=Path,
        nargs="?",
        default=Path(
            "/private/tmp/so2_cas4state_631gstar_9x9x9/electronic_reference.npz"
        ),
    )
    parser.add_argument("--point-cache", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--low-anchor", type=int, default=2)
    parser.add_argument("--low-boundary", type=int, default=2)
    parser.add_argument("--high-anchor", type=int, default=7)
    parser.add_argument("--high-boundary", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas4state_three_patch_9x9x9"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_cache = args.point_cache or args.reference.parent / "point_cache"

    with np.load(args.reference, allow_pickle=False) as archive:
        energies = np.asarray(archive["energies"])
        spin_square = np.asarray(archive["spin_square"])
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    if nstates != 4:
        raise ValueError(f"four electronic states required; found {nstates}")
    if np.max(np.abs(spin_square)) > 1.0e-7:
        raise RuntimeError("electronic reference is not spin-pure")

    middle = reference_index(grids)
    low = (middle[0], int(args.low_anchor), middle[2])
    high = (middle[0], int(args.high_anchor), middle[2])
    anchors = (middle, low, high)
    names = ("middle", "low", "high")
    started = time.perf_counter()
    overlap_fields = []
    for name, anchor in zip(names, anchors):
        overlap_fields.append(
            load_or_build(
                args.output_dir / f"direct_overlaps_{name}.npz",
                point_cache,
                shape,
                anchor,
                nstates,
                args.workers,
                args.reuse,
            )
        )
    decompositions = tuple(procrustes(values) for values in overlap_fields)
    (primary, primary_positive, primary_singular), (
        low_gauge,
        low_positive,
        low_singular,
    ), (high_gauge, high_positive, high_singular) = decompositions

    two, low_transition = stitch(
        shape,
        links,
        primary,
        low_gauge,
        axis=1,
        boundary=int(args.low_boundary),
    )
    gauge, high_transition = stitch_upper(
        shape,
        links,
        two,
        high_gauge,
        axis=1,
        boundary=int(args.high_boundary),
    )
    theta_indices = np.indices(shape)[1]
    positive = np.array(primary_positive, copy=True)
    singular = np.array(primary_singular, copy=True)
    low_mask = theta_indices <= int(args.low_boundary)
    high_mask = theta_indices > int(args.high_boundary)
    positive[low_mask] = low_positive[low_mask]
    singular[low_mask] = low_singular[low_mask]
    positive[high_mask] = high_positive[high_mask]
    singular[high_mask] = high_singular[high_mask]

    primary_energy = local_hamiltonian(
        energies,
        primary.reshape(-1, nstates, nstates),
    ).reshape(*shape, nstates, nstates)
    two_energy = local_hamiltonian(
        energies,
        two.reshape(-1, nstates, nstates),
    ).reshape(*shape, nstates, nstates)
    energy = local_hamiltonian(
        energies,
        gauge.reshape(-1, nstates, nstates),
    ).reshape(*shape, nstates, nstates)
    aligned_links = link_arrays(shape, links, gauge)

    link_singular = {}
    for axis in range(3):
        values = [
            block for (direction, _index), block in links.items() if direction == axis
        ]
        spectra = np.asarray(
            [np.linalg.svd(block, compute_uv=False) for block in values]
        )
        link_singular[str(axis)] = {
            "minimum": float(np.min(spectra)),
            "maximum": float(np.max(spectra)),
        }
    selected_min = singular[..., -1]
    summary = {
        "method": "spin-pure SO2 CASCI(6e,6o)/6-31G* four-state three-patch Procrustes atlas",
        "grid": list(shape),
        "nstates": nstates,
        "anchors": {name: list(anchor) for name, anchor in zip(names, anchors)},
        "anchor_theta_degree": {
            name: float(np.rad2deg(grids[1][anchor[1]]))
            for name, anchor in zip(names, anchors)
        },
        "boundaries": {
            "low_theta_index": int(args.low_boundary),
            "high_theta_index": int(args.high_boundary),
        },
        "max_abs_s2": float(np.max(np.abs(spin_square))),
        "selected_overlap_sigma_min": float(np.min(selected_min)),
        "selected_overlap_condition_ratio_min": float(
            np.min(selected_min / np.maximum(singular[..., 0], np.finfo(float).tiny))
        ),
        "nearest_link_singular_values": link_singular,
        "central_theta_max_step_Eh": {
            "one_patch": float(np.max(step_norm(primary_energy, middle))),
            "two_patch": float(np.max(step_norm(two_energy, middle))),
            "three_patch": float(np.max(step_norm(energy, middle))),
        },
        "atlas_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(
        args.output_dir / "procrustes_gauge.npz",
        gauge=gauge,
        positive=positive,
        singular_values=singular,
        aligned_local_hamiltonian=energy,
        center=np.asarray(middle),
        low_anchor=np.asarray(low),
        high_anchor=np.asarray(high),
        low_boundary_theta_index=int(args.low_boundary),
        high_boundary_theta_index=int(args.high_boundary),
        low_transition=low_transition,
        high_transition=high_transition,
        **{f"link_{axis}": values for axis, values in enumerate(aligned_links)},
    )

    theta = np.rad2deg(grids[1])
    edge_theta = 0.5 * (theta[:-1] + theta[1:])
    shifted = central_theta(energies - np.min(energies), middle)
    figure, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    for state, (color, marker) in enumerate(zip(colors, markers)):
        axes[0].plot(
            theta,
            shifted[:, state],
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=rf"$E_{state + 1}$",
        )
    for values, label, color, marker in (
        (primary_energy, "one patch", "#999999", "o"),
        (two_energy, "two patches", "#D55E00", "s"),
        (energy, "three patches", "#0072B2", "^"),
    ):
        axes[1].plot(
            edge_theta,
            step_norm(values, middle),
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=label,
        )
    central = (middle[0], slice(None), middle[2], -1)
    for values, label, color, marker in (
        (low_singular[central], "low anchor", "#009E73", "^"),
        (primary_singular[central], "middle anchor", "#0072B2", "o"),
        (high_singular[central], "high anchor", "#CC79A7", "D"),
        (singular[central], "selected", "#111111", "s"),
    ):
        axes[2].semilogy(
            theta,
            values,
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=label,
        )
    axes[0].set(xlabel=r"$\theta$ (degree)", ylabel=r"Adiabatic energy ($E_h$)")
    axes[1].set(
        xlabel=r"$\theta$ edge midpoint (degree)",
        ylabel=r"$\|\Delta\bar E\|_F$ ($E_h$)",
    )
    axes[2].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"Direct-overlap $\sigma_{\min}$",
    )
    for axis in axes:
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
        axis.legend(frameon=False, fontsize=7.2)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    figure_path = args.output_dir / "so2_four_state_atlas.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
