"""Plot before/after timings for the pyrazine conditional-gauge benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _letta_result(path):
    payload = json.loads(Path(path).read_text())
    return next(row for row in payload["results"] if row["method"] == "SU2-LETTA")


def plot(baseline_path, reuse_path, optimized_path, cpp_path, output):
    baseline = _letta_result(baseline_path)
    reuse = _letta_result(reuse_path)
    optimized = _letta_result(optimized_path)
    cpp = _letta_result(cpp_path)
    rows = (baseline, reuse, optimized, cpp)
    labels = (
        "rebuild",
        "moving env",
        "conditional\nmoving env",
        "matrix-free +\nroute basis",
    )
    colors = ("#9e9e9e", "#277da1", "#f9c74f", "#f3722c")
    total = np.array([row["setup_s"] + row["optimization_s"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(labels, total, color=colors)
    axes[0].set_ylabel("setup + optimization (s)")
    axes[0].tick_params(axis="x", labelsize=8)
    speedup = total[0] / total[-1]
    axes[0].text(
        0.67,
        0.95,
        f"{speedup:.2f}x faster",
        ha="center",
        va="top",
        transform=axes[0].transAxes,
    )

    for row, label, color in zip(rows, labels, colors):
        diagnostics = row["cycle_diagnostics"]
        axes[1].plot(
            [item["cycle"] for item in diagnostics],
            [item["elapsed_s"] for item in diagnostics],
            marker="o",
            label=label.replace("\n", " "),
            color=color,
        )
    axes[1].set(xlabel="complete cycle", ylabel="cycle time (s)")
    axes[1].set_xticks([1, 2, 3])
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    fig.suptitle("Pyrazine SU2-LETTA CAS(4,4), D=1")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reuse", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(plot(args.baseline, args.reuse, args.optimized, args.cpp, args.output))


if __name__ == "__main__":
    main()
