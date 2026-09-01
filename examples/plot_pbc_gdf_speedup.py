#!/usr/bin/env python3
"""Plot periodic GDF timing changes from before/after benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path, section=None):
    payload = json.loads(Path(path).read_text())
    mesh = tuple(int(value) for value in payload["kmesh"])
    if len(set(mesh)) != 1:
        raise ValueError(f"Expected a cubic mesh; got {mesh} in {path}.")
    timing = payload.get(section, payload) if section else payload.get("pyqed", payload)
    return mesh[0], float(timing["gdf_seconds"])


def plot(before, after, output, reference=None):
    import matplotlib.pyplot as plt

    before = dict(_load(path) for path in before)
    after = dict(_load(path) for path in after)
    if before.keys() != after.keys():
        raise ValueError("Before and after inputs must contain the same meshes.")
    reference = None if reference is None else dict(
        _load(path, section="pyscf") for path in reference
    )
    if reference is not None and before.keys() != reference.keys():
        raise ValueError("Reference inputs must contain the same meshes.")
    n = np.asarray(sorted(before), dtype=int)
    old = np.asarray([before[value] for value in n])
    new = np.asarray([after[value] for value in n])
    speedup = old / new

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    if reference is None:
        width = 0.34
        axes[0].bar(
            n - width / 2,
            old,
            width,
            color="#999999",
            label="Original PyQED",
        )
        axes[0].bar(
            n + width / 2,
            new,
            width,
            color="#0072B2",
            label="Current PyQED",
        )
    else:
        width = 0.25
        reference_values = np.asarray([reference[value] for value in n])
        axes[0].bar(
            n - width,
            old,
            width,
            color="#999999",
            label="Original PyQED",
        )
        axes[0].bar(
            n,
            new,
            width,
            color="#0072B2",
            label="Current PyQED",
        )
        axes[0].bar(
            n + width,
            reference_values,
            width,
            color="#D55E00",
            label="PySCF",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GDF build time (s)")
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(n, speedup, color="#009E73", marker="o", linewidth=1.4)
    for x_value, value in zip(n, speedup):
        axes[1].annotate(
            f"{value:.2f}x",
            (x_value, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    axes[1].axhline(1.0, color="0.45", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Speedup")
    axes[1].set_ylim(0.9, max(3.0, 1.15 * float(np.max(speedup))))
    axes[1].set_xlim(float(np.min(n)) - 0.25, float(np.max(n)) + 0.15)

    for label, axis in zip(("a", "b"), axes):
        axis.set_xlabel(r"Cubic mesh size $n$ in $n^3$")
        axis.set_xticks(n)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.10,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.90, wspace=0.32)

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, speedup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs="+", type=Path, required=True)
    parser.add_argument("--after", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_scheduler_speedup"),
    )
    args = parser.parse_args()
    png, pdf, speedup = plot(
        args.before,
        args.after,
        args.output,
        reference=args.reference,
    )
    print(
        json.dumps(
            {
                "figure_png": str(png),
                "figure_pdf": str(pdf),
                "speedup": speedup.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
