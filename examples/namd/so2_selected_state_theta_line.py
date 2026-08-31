#!/usr/bin/env python3
"""Transport a selected four-state SO2 subspace inside extra CASCI roots."""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
import json
import pickle
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.so2_casci_cgldr import casci_overlap_active
from examples.ldr.so2_casci_full_ldr import _electronic_point


def point_task(index, grids, basis, nstates, integral_workers):
    return (
        tuple(index),
        float(grids[0][index[0]]),
        float(grids[1][index[1]]),
        float(grids[2][index[2]]),
        basis,
        int(integral_workers),
        int(nstates),
        max(16, int(nstates) + 8),
        "eigsh",
        False,
    )


def selected_isometries(links, center, physical_states=4):
    npoints = len(links) + 1
    nstates = links[0].shape[0]
    isometries = np.empty((npoints, nstates, physical_states), dtype=complex)
    isometries[center] = np.eye(nstates, physical_states)
    for index in range(center, npoints - 1):
        overlap = isometries[index].conj().T @ links[index]
        left, _singular, right = np.linalg.svd(overlap, full_matrices=False)
        isometries[index + 1] = (left @ right).conj().T
    for index in range(center - 1, -1, -1):
        overlap = links[index] @ isometries[index + 1]
        left, _singular, right = np.linalg.svd(overlap, full_matrices=False)
        isometries[index] = left @ right
    return isometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        type=Path,
        default=Path("/private/tmp/so2_9x9x9_grids.npz"),
    )
    parser.add_argument(
        "--four-state",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas4state_three_patch_9x9x9/procrustes_gauge.npz"
        ),
    )
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--integral-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_selected_state_theta_line"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "point_cache"
    cache.mkdir(exist_ok=True)
    with np.load(args.grids, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    center = tuple(len(grid) // 2 for grid in grids)
    started = time.perf_counter()
    results = []
    for theta_index in range(len(grids[1])):
        path = cache / f"theta_{theta_index}.pkl"
        if path.is_file():
            with path.open("rb") as stream:
                result = pickle.load(stream)
        else:
            index = (center[0], theta_index, center[2])
            result = _electronic_point(
                point_task(index, grids, args.basis, args.nstates, args.integral_workers)
            )
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(result, stream, pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        results.append(result)
        print(f"[selected-state theta] {theta_index + 1}/{len(grids[1])}", flush=True)

    frames = tuple(result[1] for result in results)
    energies = np.asarray([result[2] for result in results])
    spin_square = np.asarray([result[3] for result in results])
    with np.errstate(divide="ignore", invalid="ignore"):
        links = np.asarray(
            [
                casci_overlap_active(
                    frames[index], frames[index + 1], range(args.nstates)
                )
                for index in range(len(frames) - 1)
            ]
        )
    isometries = selected_isometries(links, center[1], physical_states=4)
    shifted = energies - float(np.min(energies))
    hbar = np.asarray(
        [
            transform.conj().T @ np.diag(values) @ transform
            for values, transform in zip(shifted, isometries)
        ]
    )
    aligned_links = np.asarray(
        [
            isometries[index].conj().T @ links[index] @ isometries[index + 1]
            for index in range(len(links))
        ]
    )
    selected_singular = np.linalg.svd(aligned_links, compute_uv=False)

    with np.load(args.four_state, allow_pickle=False) as archive:
        old_energy = np.asarray(archive["aligned_local_hamiltonian"])[
            center[0], :, center[2]
        ]
        old_links = np.asarray(archive["link_1"])[center[0], :, center[2]]
    singular4 = np.linalg.svd(old_links, compute_uv=False)
    summary = {
        "method": f"four selected states in {args.nstates}-root RHF/CASCI(6e,6o)/{args.basis}",
        "theta_degree": np.rad2deg(grids[1]).tolist(),
        "candidate_states": int(args.nstates),
        "selected_states": 4,
        "max_abs_s2": float(np.max(np.abs(spin_square))),
        "four_state_min_link_singular": float(np.min(singular4)),
        "selected_min_link_singular": float(np.min(selected_singular)),
        "four_state_max_abs_e14_eh": float(np.max(np.abs(old_energy[:, 0, 3]))),
        "selected_max_abs_e14_eh": float(np.max(np.abs(hbar[:, 0, 3]))),
        "isometry_defect": float(
            np.max(
                np.abs(
                    np.einsum("...ji,...jk->...ik", isometries.conj(), isometries)
                    - np.eye(4)
                )
            )
        ),
        "seconds": time.perf_counter() - started,
    }
    np.savez(
        args.output_dir / "theta_line.npz",
        theta=grids[1],
        energies=energies,
        spin_square=spin_square,
        links=links,
        isometries=isometries,
        aligned_energy=hbar,
        aligned_links=aligned_links,
        old_aligned_energy=old_energy,
        old_aligned_links=old_links,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    theta = np.rad2deg(grids[1])
    edge_theta = 0.5 * (theta[:-1] + theta[1:])
    relative = (energies - np.min(energies)) * au2ev
    figure, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, args.nstates))
    for state, color in enumerate(colors):
        axes[0].plot(theta, relative[:, state], color=color, marker="o", ms=3, lw=1.1)
    axes[1].plot(
        theta,
        np.abs(old_energy[:, 0, 3]),
        color="#D55E00",
        marker="o",
        lw=1.3,
        label="4-state atlas",
    )
    axes[1].plot(
        theta,
        np.abs(hbar[:, 0, 3]),
        color="#0072B2",
        marker="s",
        lw=1.3,
        label=rf"{args.nstates}$\to$4 selected",
    )
    axes[2].plot(
        edge_theta,
        singular4[:, -1],
        color="#D55E00",
        marker="o",
        lw=1.3,
        label="4 states",
    )
    axes[2].plot(
        edge_theta,
        selected_singular[:, -1],
        color="#0072B2",
        marker="s",
        lw=1.3,
        label=rf"{args.nstates}$\to$4 selected",
    )
    axes[0].set(xlabel=r"$\theta$ (degree)", ylabel="Relative energy (eV)")
    axes[1].set(xlabel=r"$\theta$ (degree)", ylabel=r"$|\bar E_{14}|$ ($E_h$)")
    axes[2].set(
        xlabel=r"Link midpoint $\theta$ (degree)",
        ylabel="Minimum link singular value",
        yscale="log",
    )
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
        axis.grid(axis="y", color="0.9", lw=0.6)
    figure_path = args.output_dir / "so2_selected_state_theta_line.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
