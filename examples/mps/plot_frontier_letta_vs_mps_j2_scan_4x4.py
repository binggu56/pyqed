#!/usr/bin/env python3
"""Plot the 4x4 J1-J2 scan for small/large MPS and graph-LETTA states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results" / "frontier_letta_vs_mps_j2_scan_4x4.json"
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_j2_scan_4x4"

METHODS = {
    "mps_d4": {
        "label": r"MPS $D=4$ (424)",
        "color": "#3b76af",
        "linestyle": "--",
        "marker": "o",
    },
    "letta_d2": {
        "label": r"LETTA $D=2$ (372)",
        "color": "#df6659",
        "linestyle": "--",
        "marker": "s",
    },
    "mps_d8": {
        "label": r"MPS $D=8$ (1,448)",
        "color": "#3b76af",
        "linestyle": "-",
        "marker": "o",
    },
    "letta_d4": {
        "label": r"LETTA $D=4$ (1,448)",
        "color": "#df6659",
        "linestyle": "-",
        "marker": "s",
    },
}


def _load_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records", ())
    present = {row["method"] for row in records}
    missing = set(METHODS) - present
    if missing:
        raise ValueError(f"scan is missing methods: {sorted(missing)}")
    return data


def _aggregate(records, method, field, *, transform=None):
    rows = [row for row in records if row["method"] == method]
    ratios = np.asarray(sorted({float(row["j2_ratio"]) for row in rows}))
    median = []
    low = []
    high = []
    for ratio in ratios:
        values = np.asarray(
            [
                float(row[field])
                for row in rows
                if np.isclose(float(row["j2_ratio"]), ratio, atol=1.0e-14, rtol=0.0)
            ]
        )
        if transform is not None:
            values = transform(values)
        values = np.maximum(values, np.finfo(float).tiny)
        median.append(np.median(values))
        low.append(np.quantile(values, 0.25))
        high.append(np.quantile(values, 0.75))
    return ratios, np.asarray(median), np.asarray(low), np.asarray(high)


def plot_scan(data_path, output_stem):
    data = _load_data(data_path)
    records = data["records"]
    model = data["model"]
    seeds = sorted({int(row["seed"]) for row in records})

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.55), sharex=True)
    fig.subplots_adjust(
        left=0.105,
        right=0.97,
        bottom=0.19,
        top=0.88,
        hspace=0.39,
        wspace=0.34,
    )

    panels = (
        (
            axes[0, 0],
            "energy_error_per_site",
            None,
            r"energy error per site, $(E-E_0)/N$",
        ),
        (
            axes[0, 1],
            "ground_state_fidelity",
            lambda values: 1.0 - values,
            r"ground-state infidelity, $1-\mathcal{F}$",
        ),
        (
            axes[1, 0],
            "variance",
            None,
            r"energy variance, $\langle(H-E)^2\rangle$",
        ),
        (
            axes[1, 1],
            "optimization_seconds",
            None,
            "optimizer wall time (s)",
        ),
    )

    for axis, field, transform, ylabel in panels:
        for method, style in METHODS.items():
            ratios, center, low, high = _aggregate(
                records,
                method,
                field,
                transform=transform,
            )
            axis.fill_between(
                ratios,
                low,
                high,
                color=style["color"],
                alpha=0.11,
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                ratios,
                center,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.8,
                markerfacecolor=(
                    "white" if "--" == style["linestyle"] else style["color"]
                ),
                markeredgewidth=0.8,
                linewidth=1.45,
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(color="#dddddd", linewidth=0.5, which="major")
        axis.grid(color="#eeeeee", linewidth=0.35, which="minor", alpha=0.7)
        axis.tick_params(which="both", direction="out")

    for axis in axes[1, :]:
        axis.set_xlabel(r"frustration, $J_2/J_1$")
    for axis in axes.ravel():
        axis.set_xlim(-0.025, 1.025)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(
            0.0,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor=("white" if style["linestyle"] == "--" else style["color"]),
            linewidth=1.5,
            label=style["label"],
        )
        for style in METHODS.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.035),
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.3,
    )
    fig.suptitle(
        rf"${model['nrows']}\times{model['ncols']}$ frustrated Heisenberg model: "
        rf"fixed 24-edge $J_1$ LETTA graph; median and IQR over {len(seeds)} seeds",
        fontsize=10.2,
        y=0.96,
    )
    fig.text(
        0.53,
        0.01,
        "Numbers in parentheses are raw parameters; dashed = small, solid = large.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#505050",
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_scan(args.data, args.output)


if __name__ == "__main__":
    main()
