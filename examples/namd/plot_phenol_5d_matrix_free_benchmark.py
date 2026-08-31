#!/usr/bin/env python3
"""Compare streamed and matrix-free phenol 5D KEO dressing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(directory):
    directory = Path(directory)
    with (directory / "summary.json").open() as handle:
        summary = json.load(handle)
    arrays = np.load(
        directory / "phenol_sa_casscf_5d_ftt_ttldr.npz",
        allow_pickle=True,
    )
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("matrix_free", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference, old = _load(args.reference)
    matrix_free, new = _load(args.matrix_free)
    left = old["final_wavefunction"].ravel()
    right = new["final_wavefunction"].ravel()
    phase = np.vdot(right, left) / np.vdot(right, right)
    errors = {
        "wavefunction": float(
            np.linalg.norm(left - phase * right) / np.linalg.norm(left)
        ),
        "populations": float(
            np.max(np.abs(old["populations"] - new["populations"]))
        ),
        "absorbed probability": float(
            np.max(
                np.abs(
                    old["absorbed_probabilities"]
                    - new["absorbed_probabilities"]
                )
            )
        ),
    }
    summaries = (reference, matrix_free)
    labels = ("Streamed raw core", "Hybrid matrix-free")
    colors = ("#777777", "#009E73")
    memory = np.asarray([item["peak_rss_gib"] for item in summaries])
    setup = np.asarray(
        [item["timings_seconds"]["dressed_operator_setup"] for item in summaries]
    )
    propagation = np.asarray(
        [item["timings_seconds"]["propagation"] for item in summaries]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25))
    x = np.arange(2)
    bars = axes[0].bar(x, memory, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Peak RSS (GiB)")
    axes[0].set_ylim(0.0, 10.0)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    for bar, value in zip(bars, memory):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    axes[0].text(
        0.97,
        0.95,
        f"$-{100 * (1 - memory[1] / memory[0]):.1f}\\%$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=colors[1],
    )

    width = 0.34
    axes[1].bar(
        x - width / 2,
        setup,
        width,
        label="MPO setup",
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].bar(
        x + width / 2,
        propagation,
        width,
        label="0.05 fs step",
        color="#D55E00",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].legend(frameon=False, fontsize=8)

    error_labels = (r"$\psi$", "populations", r"$P_{\rm abs}$")
    error_values = np.asarray(list(errors.values()))
    axes[2].bar(
        np.arange(3),
        error_values,
        color=("#0072B2", "#E69F00", "#CC79A7"),
        edgecolor="black",
        linewidth=0.5,
    )
    axes[2].set_yscale("log")
    axes[2].set_ylim(1.0e-15, 1.0e-9)
    axes[2].set_ylabel("Matrix-free difference")
    axes[2].set_xticks(np.arange(3), error_labels)

    for panel, label in zip(axes, ("a", "b", "c")):
        panel.text(
            -0.17,
            1.04,
            label,
            transform=panel.transAxes,
            fontweight="bold",
        )
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", color="0.88", linewidth=0.7)
        panel.set_axisbelow(True)

    fig.suptitle("Phenol 5D matrix-free LDR-KEO dressing", fontsize=10)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "phenol_5d_matrix_free_benchmark"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    with stem.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "peak_rss_gib": memory.tolist(),
                "setup_seconds": setup.tolist(),
                "propagation_seconds": propagation.tolist(),
                "errors": errors,
            },
            handle,
            indent=2,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
