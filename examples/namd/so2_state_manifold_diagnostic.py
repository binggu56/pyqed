#!/usr/bin/env python3
"""Diagnose SO2 state-manifold loss across one problematic theta link."""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.so2_casci_cgldr import casci_overlap_active
from examples.ldr.so2_casci_full_ldr import _electronic_point


def calculate(index, grids, basis, nstates, integral_workers):
    return _electronic_point(
        (
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
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        type=Path,
        default=Path("/private/tmp/so2_9x9x9_grids.npz"),
    )
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--left-theta-index", type=int, default=6)
    parser.add_argument("--right-theta-index", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--integral-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_state_manifold_diagnostic"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.grids, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    center = tuple(len(grid) // 2 for grid in grids)
    indices = (
        (center[0], int(args.left_theta_index), center[2]),
        (center[0], int(args.right_theta_index), center[2]),
    )

    started = time.perf_counter()
    if args.workers == 1:
        results = [
            calculate(index, grids, args.basis, args.nstates, args.integral_workers)
            for index in indices
        ]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, 2)) as executor:
            futures = [
                executor.submit(
                    calculate,
                    index,
                    grids,
                    args.basis,
                    args.nstates,
                    args.integral_workers,
                )
                for index in indices
            ]
            results = [future.result() for future in futures]

    frames = tuple(result[1] for result in results)
    energies = np.asarray([result[2] for result in results])
    spin_square = np.asarray([result[3] for result in results])
    overlap = np.asarray(
        casci_overlap_active(frames[0], frames[1], range(args.nstates)),
        dtype=complex,
    )
    retained = tuple(
        sorted({size for size in (4, 6, 8, args.nstates) if size <= args.nstates})
    )
    spectra = {
        size: np.linalg.svd(overlap[:size, :size], compute_uv=False)
        for size in retained
    }
    rectangular = {
        "left4_right_all": np.linalg.svd(overlap[:4, :], compute_uv=False),
        "left_all_right4": np.linalg.svd(overlap[:, :4], compute_uv=False),
    }
    summary = {
        "method": f"RHF/CASCI(6e,6o)/{args.basis}",
        "indices": [list(index) for index in indices],
        "theta_degree": [float(np.rad2deg(grids[1][index[1]])) for index in indices],
        "nstates": int(args.nstates),
        "max_abs_s2": float(np.max(np.abs(spin_square))),
        "square_singular_values": {
            str(size): spectra[size].tolist() for size in retained
        },
        "rectangular_singular_values": {
            name: values.tolist() for name, values in rectangular.items()
        },
        "left_state4_overlap_by_right_root": np.abs(overlap[3]).tolist(),
        "right_state4_overlap_by_left_root": np.abs(overlap[:, 3]).tolist(),
        "seconds": time.perf_counter() - started,
    }
    np.savez(
        args.output_dir / "diagnostic.npz",
        energies=energies,
        spin_square=spin_square,
        overlap=overlap,
        theta_degree=np.asarray(summary["theta_degree"]),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {4: "#D55E00", 6: "#009E73", 7: "#CC79A7", 8: "#0072B2"}
    figure, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    roots = np.arange(1, args.nstates + 1)
    relative = (energies - np.min(energies)) * au2ev
    for endpoint, marker in enumerate(("o", "s")):
        axes[0].plot(
            roots,
            relative[endpoint],
            color=("#0072B2", "#D55E00")[endpoint],
            marker=marker,
            lw=1.3,
            label=rf"$\theta={summary['theta_degree'][endpoint]:.1f}^\circ$",
        )
    for size in retained:
        axes[1].plot(
            np.arange(1, size + 1),
            spectra[size],
            color=colors[size],
            marker="o",
            lw=1.3,
            label=rf"{size} states",
        )
    image = axes[2].imshow(
        np.abs(overlap),
        origin="upper",
        cmap="magma",
        aspect="equal",
        vmin=0.0,
        vmax=float(np.max(np.abs(overlap))),
    )
    figure.colorbar(image, ax=axes[2], pad=0.02, label=r"$|S_{ij}|$")
    axes[0].set(xlabel="CASCI root", ylabel="Relative energy (eV)")
    axes[1].set(xlabel="Singular-value index", ylabel="Singular value")
    axes[2].set(
        xlabel="Right root",
        ylabel="Left root",
        xticks=np.arange(args.nstates),
        yticks=np.arange(args.nstates),
        xticklabels=roots,
        yticklabels=roots,
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    for label, axis in zip("abc", axes):
        axis.text(
            0.02,
            0.98,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
            color="white" if axis is axes[2] else "black",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    figure_path = args.output_dir / "so2_state_manifold_diagnostic.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
