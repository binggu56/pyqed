"""Scale dense and graph-local SU(2) orbital circuits with active-space size."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.nonabelian.mps import MPS
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
from pyqed.mps.nonabelian.states import build_reduced_product_spatial_mps
from pyqed.narg.qchem.su2_overlap import _graph_block_orbital_map


def _problem(L, block_size=4):
    state = MPS.from_sites(
        build_reduced_product_spatial_mps(["double", "empty"] * (L // 2))
    )
    rng = np.random.default_rng(100 + L)
    orbital_map = np.eye(L)
    block_mask = np.zeros((L, L), dtype=bool)
    for start in range(0, L, block_size):
        stop = min(L, start + block_size)
        size = stop - start
        orbital_map[start:stop, start:stop] += 0.04 * rng.normal(
            size=(size, size)
        )
        block_mask[start:stop, start:stop] = True
    weak_edges = 1.0e-7 * rng.normal(size=(L, L))
    orbital_map[~block_mask] += weak_edges[~block_mask]
    return state, orbital_map


def _closed_shell_relative_error(exact_map, approximate_map):
    occupied = np.arange(0, exact_map.shape[0], 2)
    exact = np.asarray(exact_map)[:, occupied]
    approximate = np.asarray(approximate_map)[:, occupied]
    with np.errstate(divide="ignore", invalid="ignore"):
        exact_norm = np.linalg.det(exact.conj().T @ exact) ** 2
        approximate_norm = np.linalg.det(approximate.conj().T @ approximate) ** 2
        cross = np.linalg.det(exact.conj().T @ approximate) ** 2
    squared = (exact_norm + approximate_norm - 2.0 * np.real(cross)) / exact_norm
    return float(np.sqrt(max(0.0, np.real(squared))))


def _apply(state, orbital_map, *, blocks=None, cutoff=0.0, max_bond=None):
    started = time.perf_counter()
    transformed, info = apply_spatial_orbital_transform(
        state,
        orbital_map,
        inverse=False,
        orbital_blocks=blocks,
        cutoff=cutoff,
        max_bond=max_bond,
        return_info=True,
    )
    runtime = time.perf_counter() - started
    del transformed
    gc.collect()
    return runtime, info


def _row(L, method, runtime, info, *, blocks, map_residual, state_error):
    peak = int(info["peak_reduced_bond_dimension"])
    return {
        "L": L,
        "method": method,
        "blocks": blocks,
        "gates": int(info["adjacent_gate_count"]),
        "peak_bond": peak,
        "bond_squared": peak * peak,
        "runtime_s": runtime,
        "map_residual": map_residual,
        "state_error": state_error,
        "discarded_weight": float(info["sum_gate_discarded_weight"]),
    }


def run_scaling(sizes=(8, 12, 16, 20), block_size=4, threshold=1.0e-6):
    rows = []
    for L in sizes:
        state, orbital_map = _problem(L, block_size=block_size)

        runtime, info = _apply(
            state,
            orbital_map,
            cutoff=1.0e-10,
            max_bond=128,
        )
        rows.append(
            _row(
                L,
                "dense cap 128",
                runtime,
                info,
                blocks=1,
                map_residual=0.0,
                state_error=np.nan,
            )
        )

        local_map, blocks, residual = _graph_block_orbital_map(
            orbital_map,
            threshold,
        )
        runtime, info = _apply(
            state,
            local_map,
            blocks=blocks,
            cutoff=0.0,
            max_bond=None,
        )
        rows.append(
            _row(
                L,
                f"block exact (tau={threshold:.0e})",
                runtime,
                info,
                blocks=len(blocks),
                map_residual=residual,
                state_error=_closed_shell_relative_error(orbital_map, local_map),
            )
        )

        if L <= 12:
            runtime, info = _apply(
                state,
                orbital_map,
                cutoff=0.0,
                max_bond=None,
            )
            rows.append(
                _row(
                    L,
                    "dense exact",
                    runtime,
                    info,
                    blocks=1,
                    map_residual=0.0,
                    state_error=0.0,
                )
            )
    return rows


def plot(rows, output):
    styles = {
        "dense exact": ("#CC79A7", "o", "--"),
        "dense cap 128": ("#0072B2", "s", "-"),
        "block exact (tau=1e-06)": ("#D55E00", "^", "-"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    panels = (
        ("runtime_s", "Runtime (s)", True),
        ("peak_bond", r"Peak reduced bond $\chi$", True),
        ("gates", "Adjacent gates", False),
    )
    for axis, (key, ylabel, logy) in zip(axes.flat[:3], panels):
        for method, (color, marker, linestyle) in styles.items():
            selected = [row for row in rows if row["method"] == method]
            if not selected:
                continue
            axis.plot(
                [row["L"] for row in selected],
                [row[key] for row in selected],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=method,
            )
        if logy:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)

    local = [row for row in rows if row["method"].startswith("block exact")]
    axes[1, 1].plot(
        [row["L"] for row in local],
        [row["map_residual"] for row in local],
        color="#009E73",
        marker="o",
        label=r"map residual $\epsilon_{\rm map}$",
    )
    axes[1, 1].plot(
        [row["L"] for row in local],
        [row["state_error"] for row in local],
        color="#E69F00",
        marker="s",
        label="relative state error",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Approximation error")

    for label, axis in zip("abcd", axes.flat):
        axis.set_xlabel(r"Active orbitals $L$")
        axis.set_xticks((8, 12, 16, 20))
        axis.grid(color="0.88", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.17, 1.03, label, transform=axis.transAxes, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/narg_orbital_L_scaling.png"),
    )
    args = parser.parse_args()
    rows = run_scaling()
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
