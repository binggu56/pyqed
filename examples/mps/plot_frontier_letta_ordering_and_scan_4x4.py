#!/usr/bin/env python3
"""Combine the 4x4 LETTA ordering graph with scan accuracy panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_frontier_letta_vs_mps_j2_scan_4x4 import (
    METHODS,
    _aggregate,
    _load_data,
)
from plot_frustrated_j1j2_letta_4x4 import (
    COLORS,
    _draw_edges,
    _draw_nodes,
    _edge_set,
    _format_lattice_axis,
    _snake_site_order,
    _square_j1_j2_bonds,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results" / "frontier_letta_vs_mps_j2_scan_4x4.json"
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_ordering_and_scan_4x4"


def _draw_ordering_panel(axis):
    nrows = ncols = 4
    order = _snake_site_order(nrows, ncols)
    positions = {
        site: (column, nrows - 1 - row) for site, (row, column) in enumerate(order)
    }
    nearest, diagonals = _square_j1_j2_bonds(nrows, ncols)
    virtual_edges = {(site, site + 1) for site in range(nrows * ncols - 1)}

    _draw_edges(
        axis,
        _edge_set(diagonals),
        positions,
        color=COLORS["j2"],
        linewidth=0.8,
        linestyle=(0, (2.5, 2.0)),
        alpha=0.22,
        zorder=0,
    )
    _draw_edges(
        axis,
        _edge_set(nearest),
        positions,
        color=COLORS["tie"],
        linewidth=4.4,
        alpha=0.82,
        solid_capstyle="round",
        zorder=2,
    )
    _draw_edges(
        axis,
        virtual_edges,
        positions,
        color=COLORS["virtual"],
        linewidth=1.8,
        solid_capstyle="round",
        zorder=4,
    )
    _draw_nodes(axis, positions)
    _format_lattice_axis(axis, nrows, ncols)
    axis.set_title(
        "Fixed $J_1$ ties\n" "snake site ordering",
        pad=5.0,
    )


def _draw_scan_panel(axis, records, field, ylabel, *, transform=None):
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
            markersize=3.7,
            markerfacecolor=("white" if style["linestyle"] == "--" else style["color"]),
            markeredgewidth=0.8,
            linewidth=1.4,
            zorder=3,
        )
    axis.set_yscale("log")
    axis.set_xlim(-0.025, 1.025)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_ylabel(ylabel)
    axis.grid(color="#dddddd", linewidth=0.5, which="major")
    axis.grid(color="#eeeeee", linewidth=0.35, which="minor", alpha=0.7)
    axis.tick_params(which="both", direction="out")


def _panel_label(axis, label):
    axis.text(
        -0.08,
        1.045,
        label,
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=8,
    )


def plot_combined(data_path, output_stem):
    data = _load_data(data_path)
    records = data["records"]
    seeds = sorted({int(row["seed"]) for row in records})

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(7.4, 3.45))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.98,
        bottom=0.30,
        top=0.88,
        width_ratios=(0.90, 1.18, 1.18),
        wspace=0.48,
    )
    graph_axis = figure.add_subplot(grid[0, 0])
    energy_axis = figure.add_subplot(grid[0, 1])
    infidelity_axis = figure.add_subplot(grid[0, 2], sharex=energy_axis)

    _draw_ordering_panel(graph_axis)
    _draw_scan_panel(
        energy_axis,
        records,
        "energy_error_per_site",
        r"$(E-E_0)/N$",
    )
    _draw_scan_panel(
        infidelity_axis,
        records,
        "ground_state_fidelity",
        r"$1-\mathcal{F}$",
        transform=lambda values: 1.0 - values,
    )
    energy_axis.set_ylim(3.0e-3, 1.2e-1)
    infidelity_axis.set_ylim(2.0e-2, 1.25)
    energy_axis.set_title("Energy error", pad=7.0)
    infidelity_axis.set_title("Ground-state infidelity", pad=7.0)
    energy_axis.set_xlabel(r"$J_2/J_1$")
    infidelity_axis.set_xlabel(r"$J_2/J_1$")

    for axis, label in zip(
        (graph_axis, energy_axis, infidelity_axis),
        "abc",
    ):
        _panel_label(axis, label)

    graph_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["tie"],
            linewidth=4.2,
            label=r"tied $J_1$ legs",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["virtual"],
            linewidth=1.9,
            label="virtual backbone",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["j2"],
            linewidth=1.1,
            linestyle=(0, (2.5, 2.0)),
            alpha=0.45,
            label=r"untied $J_2$ interactions",
        ),
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.1,
            markerfacecolor=("white" if style["linestyle"] == "--" else style["color"]),
            linewidth=1.45,
            label=style["label"],
        )
        for style in METHODS.values()
    ]
    graph_legend = figure.legend(
        handles=graph_handles,
        loc="lower center",
        bbox_to_anchor=(0.175, 0.045),
        ncol=1,
        frameon=False,
        handlelength=2.5,
        labelspacing=0.48,
    )
    figure.add_artist(graph_legend)
    figure.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.695, 0.080),
        ncol=2,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.3,
        labelspacing=0.55,
    )
    figure.text(
        0.705,
        0.020,
        rf"Median/IQR: {len(seeds)} seeds; parentheses: parameters; "
        "dashed/solid: small/large.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#505050",
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=400)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_combined(args.data, args.output)


if __name__ == "__main__":
    main()
