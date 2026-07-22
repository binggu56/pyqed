#!/usr/bin/env python3
"""Plot the converged Heisenberg-chain LETTA/MPS error density."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
if (HERE / "data" / "heisenberg_1d_converged_summary.csv").exists():
    DEFAULT_INPUT = HERE / "data" / "heisenberg_1d_converged_summary.csv"
    DEFAULT_OUTPUT = HERE / "figures" / "heisenberg_letta_vs_mps.pdf"
else:
    DEFAULT_INPUT = HERE / "results" / "heisenberg_1d_converged_summary.csv"
    DEFAULT_OUTPUT = HERE / "results" / "heisenberg_1d_converged_per_site.pdf"


def load_summary(path: Path) -> dict[int, dict[str, np.ndarray]]:
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["bond_dim"])].append(row)

    result = {}
    for bond_dim, rows in grouped.items():
        rows.sort(key=lambda row: int(row["length"]))
        result[bond_dim] = {
            "L": np.asarray([int(row["length"]) for row in rows]),
            "mps": np.asarray([float(row["mps_error_per_site"]) for row in rows]),
            "letta": np.asarray([float(row["letta_error_per_site"]) for row in rows]),
        }
    return result


def make_figure(data: dict[int, dict[str, np.ndarray]], output: Path) -> None:
    uplt.rc.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    colors = {1: "#3b6ea8", 2: "#c84d36", 4: "#2d9465"}
    markers = {1: "o", 2: "s", 4: "^"}
    error_floor = 5.0e-6

    fig, axes = uplt.subplots(refwidth=3.75, refheight=2.65)
    ax = axes[0]

    for bond_dim in (1, 2, 4):
        series = data[bond_dim]
        color = colors[bond_dim]
        marker = markers[bond_dim]
        if bond_dim != 1:
            ax.plot(
                series["L"],
                series["mps"],
                linestyle="--",
                marker=marker,
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=5.0,
                linewidth=1.2,
            )
        visible = series["letta"] >= error_floor
        ax.plot(
            series["L"][visible],
            series["letta"][visible],
            linestyle="-",
            marker=marker,
            color=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=5.2,
            linewidth=1.6,
        )
    ax.format(
        yscale="log",
        xlabel=r"chain length $L$",
        ylabel=r"energy error per site $\Delta e$",
        xticks=[6, 12, 20, 32, 48],
        xlim=(4.8, 49.2),
        ylim=(error_floor, 6.0e-2),
        grid=True,
        gridminor=True,
    )
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    fig.format(
        toplabels=False,
        gridcolor="#d9d9d9",
        gridlinewidth=0.55,
        tickminor=False,
    )
    letta_handles = {
        bond_dim: Line2D(
            [0],
            [0],
            color=colors[bond_dim],
            marker=markers[bond_dim],
            linestyle="-",
            linewidth=1.6,
            markersize=4.8,
            label=fr"LETTA, $D={bond_dim}$",
        )
        for bond_dim in (1, 2, 4)
    }
    mps_handles = {
        bond_dim: Line2D(
            [0],
            [0],
            color=colors[bond_dim],
            marker=markers[bond_dim],
            markerfacecolor="white",
            markeredgewidth=1.1,
            linestyle="--",
            linewidth=1.2,
            markersize=4.8,
            label=fr"MPS, $D={bond_dim}$",
        )
        for bond_dim in (2, 4)
    }
    blank = Line2D([], [], linestyle="none", label="")
    handles = [
        letta_handles[1],
        blank,
        letta_handles[2],
        mps_handles[2],
        letta_handles[4],
        mps_handles[4],
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncols=2,
        frame=False,
        columnspacing=0.65,
        handlelength=1.45,
        handletextpad=0.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    uplt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(load_summary(args.input), args.output)
