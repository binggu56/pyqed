#!/usr/bin/env python3
"""Plot an adaptive CP-LETTA graph on its square-lattice embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = (
    HERE / "results" / "adaptive_cp_letta_j1j2_square_D4_joint.json"
)


def _edge_set(edges):
    return {tuple(sorted(map(int, edge))) for edge in edges}


def _parent_edges(parent_sets):
    return {
        (site, int(parent))
        for site, parents in enumerate(parent_sets)
        for parent in parents
    }


def _draw_edges(ax, edges, positions, **plot_kwargs):
    for left, right in sorted(edges):
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        ax.plot([x0, x1], [y0, y1], **plot_kwargs)


def plot_graph(data, output_stem: Path) -> tuple[Path, Path]:
    """Render the Hamiltonian and optimized LETTA graphs side by side."""
    nrows = int(data["model"]["nrows"])
    ncols = int(data["model"]["ncols"])
    site_order = tuple(tuple(map(int, coordinate)) for coordinate in data["site_order"])
    positions = {
        site: (column, nrows - 1 - row)
        for site, (row, column) in enumerate(site_order)
    }

    nearest = _edge_set(data["nearest_bonds"])
    diagonals = _edge_set(data["diagonal_bonds"])
    interactions = nearest | diagonals
    baseline = _edge_set(data["initial_tie_edges"])
    selected = _parent_edges(data["parent_sets"])
    added = selected - baseline
    physical_added = added & interactions
    nonphysical_added = added - interactions
    ranks = np.asarray(data["tie_ranks"], dtype=int)

    colors = {
        "j1": "#9aa0a6",
        "j2": "#d55e5e",
        "virtual": "#e69f00",
        "baseline": "#315b8a",
        "added": "#009e73",
        "nonphysical": "#aa4499",
        "node_edge": "#202124",
    }
    rank_palette = {
        1: "#d9edf7",
        2: "#78b7c5",
        3: "#f2c14e",
        4: "#e76f51",
    }
    node_colors = [rank_palette.get(int(rank), "#b8b8b8") for rank in ranks]

    uplt.rc.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = uplt.subplots(ncols=2, refwidth=3.1, refheight=2.55, wspace=0.7)
    model_ax, graph_ax = axes

    _draw_edges(
        model_ax,
        nearest,
        positions,
        color=colors["j1"],
        linewidth=1.5,
        solid_capstyle="round",
        zorder=1,
    )
    _draw_edges(
        model_ax,
        diagonals,
        positions,
        color=colors["j2"],
        linewidth=1.25,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )

    _draw_edges(
        graph_ax,
        nearest - selected,
        positions,
        color=colors["j1"],
        linewidth=0.7,
        alpha=0.28,
        zorder=0,
    )
    _draw_edges(
        graph_ax,
        diagonals - selected,
        positions,
        color=colors["j2"],
        linewidth=0.7,
        linestyle=(0, (2.5, 2.0)),
        alpha=0.23,
        zorder=0,
    )
    virtual_edges = {(site, site + 1) for site in range(len(site_order) - 1)}
    _draw_edges(
        graph_ax,
        virtual_edges,
        positions,
        color=colors["virtual"],
        linewidth=5.2,
        alpha=0.72,
        solid_capstyle="round",
        zorder=1,
    )
    _draw_edges(
        graph_ax,
        baseline & selected,
        positions,
        color=colors["baseline"],
        linewidth=1.6,
        solid_capstyle="round",
        zorder=2,
    )
    _draw_edges(
        graph_ax,
        physical_added,
        positions,
        color=colors["added"],
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3,
    )
    _draw_edges(
        graph_ax,
        nonphysical_added,
        positions,
        color=colors["nonphysical"],
        linewidth=2.8,
        linestyle=(0, (2.0, 1.4)),
        solid_capstyle="round",
        zorder=3,
    )

    coordinates = np.asarray([positions[site] for site in range(len(site_order))])
    for ax, facecolors in ((model_ax, ["white"] * len(ranks)), (graph_ax, node_colors)):
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=285,
            facecolor=facecolors,
            edgecolor=colors["node_edge"],
            linewidth=1.05,
            zorder=4,
        )
        for site, (x, y) in enumerate(coordinates):
            ax.text(
                x,
                y,
                str(site),
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="semibold",
                color="#151515",
                zorder=5,
            )
        ax.format(
            xlim=(-0.48, ncols - 0.52),
            ylim=(-0.48, nrows - 0.52),
            aspect="equal",
            xticks=[],
            yticks=[],
            xlabel="",
            ylabel="",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    model = data["model"]
    metrics = data["metrics"]
    model_ax.format(
        title=(
            rf"${nrows}\times{ncols}$ $J_1$–$J_2$ model"
            "\n"
            rf"$J_1={model['j1']:g},\ J_2={model['j2']:g}$"
        )
    )
    graph_ax.format(
        title=(
            rf"Final adaptive LETTA graph, $D={data['settings']['bond_dim']}$"
            "\n"
            rf"$\Delta E={metrics['energy_error']:.3f},\ "
            rf"\mathcal{{F}}={metrics['fidelity']:.3f}$"
        )
    )
    model_ax.text(-0.08, 1.06, "a", transform=model_ax.transAxes, fontweight="bold")
    graph_ax.text(-0.08, 1.06, "b", transform=graph_ax.transAxes, fontweight="bold")

    edge_handles = [
        Line2D([0], [0], color=colors["j1"], linewidth=1.6, label=r"$J_1$ bond"),
        Line2D(
            [0],
            [0],
            color=colors["j2"],
            linewidth=1.4,
            linestyle=(0, (3.0, 2.0)),
            label=r"$J_2$ bond",
        ),
        Line2D(
            [0],
            [0],
            color=colors["virtual"],
            linewidth=5.0,
            label="virtual backbone",
        ),
        Line2D(
            [0],
            [0],
            color=colors["baseline"],
            linewidth=1.8,
            label="inherited tie",
        ),
        Line2D(
            [0],
            [0],
            color=colors["added"],
            linewidth=3.0,
            label="selected physical tie",
        ),
    ]
    if nonphysical_added:
        edge_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["nonphysical"],
                linewidth=2.8,
                linestyle=(0, (2.0, 1.4)),
                label="selected non-H tie",
            )
        )
    rank_handles = [
        Patch(
            facecolor=rank_palette.get(int(rank), "#b8b8b8"),
            edgecolor=colors["node_edge"],
            label=rf"$r_i={int(rank)}$",
        )
        for rank in sorted(set(ranks))
    ]
    fig.legend(
        handles=edge_handles + rank_handles,
        loc="bottom",
        ncols=min(5, len(edge_handles + rank_handles)),
        frame=False,
        columnspacing=1.25,
        handlelength=2.2,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    uplt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        help="output stem; defaults to the JSON path without its suffix",
    )
    args = parser.parse_args()
    with args.data.open(encoding="utf-8") as handle:
        data = json.load(handle)
    output = args.output if args.output is not None else args.data.with_suffix("")
    pdf_path, png_path = plot_graph(data, output)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
