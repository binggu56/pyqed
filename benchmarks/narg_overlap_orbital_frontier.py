"""Benchmark graph-frontier control of SU(2) orbital-circuit bond growth."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.nonabelian.mps import MPS
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
from pyqed.mps.nonabelian.states import (
    build_random_reduced_spatial_mps,
    spatial_target_sector,
)
from pyqed.narg.qchem.su2_overlap import _graph_block_orbital_map
from pyqed.qchem.dmrg.dmrg import _fully_reduced_spatial_mps_to_component_mps


def _component_vector(state):
    component = _fully_reduced_spatial_mps_to_component_mps(state)
    return np.asarray(tt_to_tensor(component.factors)).reshape(-1)


def run_benchmark(L=8, bond_multiplicity=2):
    state = MPS.from_sites(
        build_random_reduced_spatial_mps(
            L,
            target_sector=spatial_target_sector(L, 0),
            bond_multiplicity=bond_multiplicity,
            seed=19,
        )
    )
    rng = np.random.default_rng(8)
    orbital_map = np.eye(L)
    midpoint = L // 2
    for start, stop in ((0, midpoint), (midpoint, L)):
        size = stop - start
        orbital_map[start:stop, start:stop] += 0.04 * rng.normal(
            size=(size, size)
        )
    orbital_map[:midpoint, midpoint:] += 1.0e-7 * rng.normal(
        size=(midpoint, L - midpoint)
    )
    orbital_map[midpoint:, :midpoint] += 1.0e-7 * rng.normal(
        size=(L - midpoint, midpoint)
    )

    thresholds = (0.0, 1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6)
    rows = []
    vectors = []
    for threshold in thresholds:
        approximated, blocks, map_residual = _graph_block_orbital_map(
            orbital_map,
            threshold,
        )
        started = time.perf_counter()
        transformed, info = apply_spatial_orbital_transform(
            state,
            approximated,
            inverse=False,
            orbital_blocks=blocks,
            cutoff=0.0,
            max_bond=None,
            return_info=True,
        )
        elapsed = time.perf_counter() - started
        vectors.append(_component_vector(transformed))
        rows.append(
            {
                "threshold": threshold,
                "blocks": len(blocks),
                "map_residual": map_residual,
                "gates": info["adjacent_gate_count"],
                "peak_bond": info["peak_reduced_bond_dimension"],
                "runtime_s": elapsed,
            }
        )
    reference = vectors[0]
    reference_norm = np.linalg.norm(reference)
    for row, vector in zip(rows, vectors):
        row["relative_state_error"] = float(
            np.linalg.norm(vector - reference) / reference_norm
        )
    return rows


def plot(rows, output):
    labels = ["exact" if row["threshold"] == 0 else f'{row["threshold"]:.0e}' for row in rows]
    x = np.arange(len(rows))
    colors = ["#0072B2" if row["blocks"] == 1 else "#D55E00" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)

    axes[0].bar(x - 0.18, [row["gates"] for row in rows], 0.36, label="gates", color="#0072B2")
    axes[0].bar(x + 0.18, [row["peak_bond"] for row in rows], 0.36, label="peak bond", color="#E69F00")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, [1.0e7 * row["map_residual"] for row in rows], color=colors)
    axes[1].set_ylabel(r"Map residual ($10^{-7}$)")

    axes[2].bar(x, [1.0e6 * row["relative_state_error"] for row in rows], color=colors)
    axes[2].set_ylabel(r"Relative state error ($10^{-6}$)")

    for label, axis in zip("abc", axes):
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_xlabel(r"Graph threshold $\tau$")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.17, 1.03, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/narg_orbital_frontier_scaling.png"),
    )
    args = parser.parse_args()
    rows = run_benchmark(L=args.L)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)
    for row in rows:
        print(row)
    print(f"data: {csv_path}")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
