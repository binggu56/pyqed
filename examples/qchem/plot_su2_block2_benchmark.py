#!/usr/bin/env python3
"""Plot matched PyQED/block2 SU(2) large-CAS benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    records = sorted(records, key=lambda row: int(row["ncas"]))
    labels = [f"CAS({row['nelecas']},{row['ncas']})" for row in records]
    py_time = np.asarray(
        [row["median"]["pyqed_sweep_seconds"] for row in records]
    )
    b2_time = np.asarray(
        [row["median"]["block2_sweep_seconds"] for row in records]
    )
    py_memory = np.asarray(
        [row["median"]["pyqed_peak_rss_bytes"] for row in records]
    ) / 2**20
    b2_memory = np.asarray(
        [row["median"]["block2_peak_rss_bytes"] for row in records]
    ) / 2**20

    colors = ("#0072B2", "#D55E00")
    x = np.arange(len(records), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, py_values, b2_values, ylabel in (
        (axes[0], py_time, b2_time, "Sweep time (s)"),
        (axes[1], py_memory, b2_memory, "Peak RSS (MiB)"),
    ):
        axis.bar(
            x - width / 2,
            py_values,
            width,
            color=colors[0],
            label="PyQED",
        )
        axis.bar(
            x + width / 2,
            b2_values,
            width,
            color=colors[1],
            hatch="//",
            label="block2",
        )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].set_title("a  Matched one-thread sweeps", loc="left", fontsize=10)
    axes[1].set_title("b  Process memory", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
