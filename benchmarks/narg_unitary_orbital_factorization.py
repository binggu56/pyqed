"""Compare direct and SVD-mediated two-orbital factorizations of rotations."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.nonabelian.mps import MPS
from pyqed.mps.nonabelian.orbital_transform import (
    _adjacent_unitary_circuit,
    _apply_adjacent_gate,
    _apply_diagonal,
    _as_reduced_mps,
    apply_spatial_orbital_transform,
)
from pyqed.mps.nonabelian.states import build_reduced_product_spatial_mps
from pyqed.narg.qchem.su2_overlap import _batched_overlap_matrix


def _rotation(L, block_size=4):
    rotation = np.zeros((L, L))
    blocks = []
    rng = np.random.default_rng(200 + L)
    for start in range(0, L, block_size):
        stop = min(L, start + block_size)
        block = tuple(range(start, stop))
        blocks.append(block)
        rotation[np.ix_(block, block)] = np.linalg.qr(
            rng.normal(size=(len(block), len(block)))
        )[0]
    return rotation, blocks


def _legacy_svd_circuit(rotation, blocks):
    circuit = []
    L = rotation.shape[0]
    for block in blocks:
        left, singular, right_h = np.linalg.svd(
            rotation[np.ix_(block, block)],
            full_matrices=False,
        )
        local = (
            _adjacent_unitary_circuit(right_h)
            + [("diagonal", singular.astype(complex))]
            + _adjacent_unitary_circuit(left)
        )
        for kind, *payload in local:
            if kind == "diagonal":
                diagonal = np.ones(L, dtype=complex)
                diagonal[np.asarray(block)] = payload[0]
                circuit.append((kind, diagonal))
            else:
                bond, gate = payload
                circuit.append((kind, block[0] + bond, gate))
    return circuit


def _apply_circuit(state, circuit):
    transformed = _as_reduced_mps(state)
    peak = 1
    for kind, *payload in circuit:
        if kind == "diagonal":
            _apply_diagonal(transformed, payload[0])
        else:
            _apply_adjacent_gate(
                transformed,
                payload[0],
                payload[1],
                cutoff=0.0,
                max_bond=None,
            )
            peak = max(
                peak,
                *(len(site.qns[2]) for site in transformed.sites[:-1]),
            )
    return transformed, peak


def _relative_difference(left, right):
    ll = _batched_overlap_matrix(left, left, 1, 1)[0, 0]
    rr = _batched_overlap_matrix(right, right, 1, 1)[0, 0]
    lr = _batched_overlap_matrix(left, right, 1, 1)[0, 0]
    squared = np.real(ll + rr - 2.0 * lr) / np.real(ll)
    return float(np.sqrt(max(0.0, squared)))


def run(sizes=(8, 12, 16, 20), repeats=5):
    rows = []
    for L in sizes:
        state = MPS.from_sites(
            build_reduced_product_spatial_mps(["double", "empty"] * (L // 2))
        )
        rotation, blocks = _rotation(L)
        legacy_circuit = _legacy_svd_circuit(rotation, blocks)

        direct_times = []
        legacy_times = []
        direct = legacy = None
        direct_info = None
        legacy_peak = 1
        for _ in range(repeats):
            started = time.perf_counter()
            direct, direct_info = apply_spatial_orbital_transform(
                state,
                rotation,
                inverse=False,
                orbital_blocks=blocks,
                cutoff=0.0,
                max_bond=None,
                return_info=True,
            )
            direct_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            legacy, legacy_peak = _apply_circuit(state, legacy_circuit)
            legacy_times.append(time.perf_counter() - started)

        error = _relative_difference(direct, legacy)
        rows.extend(
            [
                {
                    "L": L,
                    "method": "direct Givens",
                    "gates": direct_info["adjacent_gate_count"],
                    "peak_bond": direct_info["peak_reduced_bond_dimension"],
                    "runtime_s": float(np.median(direct_times)),
                    "relative_difference": error,
                },
                {
                    "L": L,
                    "method": "SVD Givens",
                    "gates": sum(step[0] == "gate" for step in legacy_circuit),
                    "peak_bond": legacy_peak,
                    "runtime_s": float(np.median(legacy_times)),
                    "relative_difference": error,
                },
            ]
        )
    return rows


def plot(rows, output):
    styles = {
        "direct Givens": ("#D55E00", "o"),
        "SVD Givens": ("#0072B2", "s"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9), constrained_layout=True)
    for method_index, (method, (color, marker)) in enumerate(styles.items()):
        selected = [row for row in rows if row["method"] == method]
        x = [row["L"] for row in selected]
        axes[0].plot(x, [row["gates"] for row in selected], color=color, marker=marker, label=method)
        axes[1].plot(x, [row["runtime_s"] for row in selected], color=color, marker=marker)
        offset = -0.35 if method_index == 0 else 0.35
        axes[2].bar(
            np.asarray(x) + offset,
            [row["peak_bond"] for row in selected],
            width=0.7,
            color=color,
        )
    axes[0].set_ylabel("Two-orbital gates")
    axes[1].set_ylabel("Median runtime (s)")
    axes[2].set_ylabel(r"Peak reduced bond $\chi$")
    for label, axis in zip("abc", axes):
        axis.set_xlabel(r"Active orbitals $L$")
        axis.set_xticks((8, 12, 16, 20))
        axis.grid(color="0.88", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.17, 1.03, label, transform=axis.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/narg_unitary_orbital_factorization.png"),
    )
    args = parser.parse_args()
    rows = run()
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
