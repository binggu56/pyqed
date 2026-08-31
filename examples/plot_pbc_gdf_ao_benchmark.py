#!/usr/bin/env python3
"""Plot native periodic GDF AO-build timing, work, and memory diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _study(path):
    payload = json.loads(Path(path).read_text())
    studies = payload.get("studies", [])
    if len(studies) != 1:
        raise ValueError(f"{path} must contain exactly one validation study.")
    return studies[0]


def _timings(study):
    blocks = study.get("q_blocks", [])
    if not blocks:
        raise ValueError("Validation study does not contain q-block timings.")
    return blocks[0]["native_build_timings"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unscreened", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_ao_architecture_benchmark.pdf"),
    )
    args = parser.parse_args()

    unscreened = _study(args.unscreened)
    optimized = _study(args.optimized)
    raw = _timings(unscreened)
    sparse = _timings(optimized)

    raw_seconds = np.asarray(
        [raw["three_center_sr_component_seconds"], unscreened["native_gdf_seconds"]]
    )
    sparse_seconds = np.asarray(
        [
            sparse["three_center_sr_component_seconds"],
            optimized["native_gdf_seconds"],
        ]
    )
    pyscf_seconds = float(optimized["pyscf_gdf_build_seconds"])
    candidates = int(sparse["three_center_sr_primitive_candidates"])
    skipped = int(sparse["three_center_sr_primitive_skips"])
    evaluated = candidates - skipped
    memory_mib = np.asarray(
        [
            sparse["three_center_sr_peak_image_tensor_bytes"],
            sparse["three_center_sr_group_output_bytes"],
            sparse["three_center_sr_group_workspace_bytes_upper_bound"],
        ],
        dtype=float,
    ) / (1024.0**2)

    colors = ("#4C78A8", "#E45756", "#59A14F")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.28,
        wspace=0.42,
    )

    ax = axs[0]
    x = np.arange(2)
    width = 0.34
    ax.bar(
        x - width / 2,
        raw_seconds,
        width=width,
        color=colors[0],
        edgecolor="black",
        linewidth=0.6,
        label="Unscreened native",
    )
    ax.bar(
        x + width / 2,
        sparse_seconds,
        width=width,
        color=colors[1],
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
        label="Sparse shell engine",
    )
    ax.axhline(
        pyscf_seconds,
        color=colors[2],
        linewidth=1.4,
        linestyle="--",
        label="PySCF DF build",
    )
    for index, speedup in enumerate(raw_seconds / sparse_seconds):
        ax.text(
            index,
            raw_seconds[index] * 1.12,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Wall time (s)")
    ax.set_xticks(x, ("Short-range AO", "Full native factors"))
    ax.set_ylim(1.0, 250.0)
    ax.set_title("Production precision")
    ax.grid(axis="y", which="major", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    ax = axs[1]
    work = np.asarray([candidates, evaluated], dtype=float) / 1.0e6
    bars = []
    for row, (value, color) in enumerate(zip(work, colors[:2])):
        bars.append(
            ax.barh(
                [row],
                [value],
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )[0]
        )
    for bar, value in zip(bars, work):
        ax.text(
            value * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} M",
            ha="left",
            va="center",
            fontsize=9,
        )
    removed = 100.0 * skipped / candidates
    ax.text(
        0.98,
        0.08,
        f"{removed:.1f}% removed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=colors[1],
    )
    ax.set_xlabel("Primitive image tasks (millions)")
    ax.set_yticks(np.arange(2), ("Candidate", "Evaluated"))
    ax.set_xlim(0.0, 620.0)
    ax.set_title("Compiled work")
    ax.grid(axis="x", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    ax = axs[2]
    y = np.arange(3)
    memory_bars = ax.barh(
        y,
        memory_mib,
        color=(colors[0], colors[2], colors[1]),
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, value in zip(memory_bars, memory_mib):
        label = "0 B" if value == 0.0 else f"{value:.2f} MiB"
        ax.text(
            max(value, 0.06) + 0.10,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Transient storage (MiB)")
    ax.set_yticks(
        y,
        ("Image tensor", "Bloch output", "Workspace bound"),
    )
    ax.set_xlim(0.0, max(8.5, memory_mib.max() * 1.18))
    ax.set_title("Bounded memory")
    ax.grid(axis="x", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for label, panel in zip(("a", "b", "c"), axs):
        panel.text(
            -0.14,
            1.05,
            label,
            transform=panel.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        ncols=3,
        fontsize=8.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
