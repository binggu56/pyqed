#!/usr/bin/env python3
"""Plot the phenol 5D TTLDR memory-optimization benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(run: Path):
    with (run / "summary.json").open() as handle:
        summary = json.load(handle)
    arrays = np.load(run / "phenol_sa_casscf_5d_ftt_ttldr.npz")
    return summary, arrays


def _wavefunction_error(reference, candidate):
    left = np.asarray(reference["final_wavefunction"]).ravel()
    right = np.asarray(candidate["final_wavefunction"]).ravel()
    phase = np.vdot(right, left) / np.vdot(right, right)
    return float(np.linalg.norm(left - phase * right) / np.linalg.norm(left))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=4, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = ["Reference", "Streamed MPO", "+ packed RHS", "+ shared cores"]
    summaries = []
    arrays = []
    for run in args.runs:
        summary, data = _load(run)
        summaries.append(summary)
        arrays.append(data)

    peak = np.array([item["peak_rss_gib"] for item in summaries])
    setup = np.array(
        [item["timings_seconds"]["dressed_operator_setup"] for item in summaries]
    )
    step = np.array(
        [item["timings_seconds"]["propagation"] for item in summaries]
    )
    errors = np.array(
        [0.0, *[_wavefunction_error(arrays[0], item) for item in arrays[1:]]]
    )

    colors = ["#777777", "#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))

    x = np.arange(len(labels))
    bars = axes[0].bar(x, peak, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Peak RSS (GiB)")
    axes[0].set_ylim(0, 11.5)
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].grid(axis="y", color="0.88", linewidth=0.7)
    for bar, value in zip(bars, peak):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    reduction = 100.0 * (1.0 - peak[-1] / peak[0])
    axes[0].text(
        0.98,
        0.96,
        f"$-{reduction:.1f}\\%$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=colors[-1],
        fontsize=9,
    )

    width = 0.36
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
        step,
        width,
        label="0.05 fs step",
        color="#D55E00",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].grid(axis="y", color="0.88", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].text(
        0.98,
        0.72,
        f"final $\\epsilon_\\psi={errors[-1]:.1e}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)

    fig.suptitle("Phenol 5D TTLDR exact-path memory optimization", fontsize=10)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "phenol_5d_memory_benchmark"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    with stem.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "labels": labels,
                "peak_rss_gib": peak.tolist(),
                "setup_seconds": setup.tolist(),
                "propagation_seconds": step.tolist(),
                "phase_aligned_relative_wavefunction_error": errors.tolist(),
            },
            handle,
            indent=2,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
