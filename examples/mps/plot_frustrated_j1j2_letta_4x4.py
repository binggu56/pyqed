#!/usr/bin/env python3
"""Plot the frustrated 4x4 J1-J2 model and its graph-LETTA representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frustrated_j1j2_letta_4x4"

COLORS = {
    "j1": "#7c8794",
    "j2": "#d55e5e",
    "tie": "#007f73",
    "virtual": "#e69f00",
    "node_edge": "#202124",
    "node_face": "#f8fafc",
    "frustrated": "#f3c969",
}


def _edge_set(edges):
    return {tuple(sorted(map(int, edge))) for edge in edges}


def _snake_site_order(nrows, ncols):
    order = []
    for row in range(nrows):
        columns = range(ncols) if row % 2 == 0 else range(ncols - 1, -1, -1)
        order.extend((row, column) for column in columns)
    return tuple(order)


def _square_j1_j2_bonds(nrows, ncols):
    order = _snake_site_order(nrows, ncols)
    chain_site = {coordinate: site for site, coordinate in enumerate(order)}

    def edge(left, right):
        return tuple(sorted((chain_site[left], chain_site[right])))

    nearest = []
    diagonals = []
    for row in range(nrows):
        for column in range(ncols):
            if column + 1 < ncols:
                nearest.append(edge((row, column), (row, column + 1)))
            if row + 1 < nrows:
                nearest.append(edge((row, column), (row + 1, column)))
            if row + 1 < nrows and column + 1 < ncols:
                diagonals.append(edge((row, column), (row + 1, column + 1)))
                diagonals.append(edge((row, column + 1), (row + 1, column)))
    return tuple(sorted(set(nearest))), tuple(sorted(set(diagonals)))


def _draw_edges(ax, edges, positions, **plot_kwargs):
    for left, right in sorted(edges):
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        ax.plot([x0, x1], [y0, y1], **plot_kwargs)


def _draw_nodes(ax, positions):
    coordinates = np.asarray([positions[site] for site in range(len(positions))])
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=245,
        facecolor=COLORS["node_face"],
        edgecolor=COLORS["node_edge"],
        linewidth=1.05,
        zorder=8,
    )
    for site, (x, y) in enumerate(coordinates):
        ax.text(
            x,
            y,
            str(site),
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="semibold",
            color="#151515",
            zorder=9,
        )


def _format_lattice_axis(ax, nrows, ncols):
    ax.set_xlim(-0.48, ncols - 0.52)
    ax.set_ylim(-0.48, nrows - 0.52)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_model(output_stem: Path) -> tuple[Path, Path]:
    """Render the Hamiltonian and LETTA graphs as PDF and high-resolution PNG."""
    nrows = ncols = 4
    j1 = 1.0
    j2 = 0.5
    site_order = _snake_site_order(nrows, ncols)
    positions = {
        site: (column, nrows - 1 - row) for site, (row, column) in enumerate(site_order)
    }
    nearest, diagonals = _square_j1_j2_bonds(nrows, ncols)
    nearest = _edge_set(nearest)
    diagonals = _edge_set(diagonals)
    virtual_edges = {(site, site + 1) for site in range(nrows * ncols - 1)}

    # The loop 0-1-6 contains two J1 bonds and one J2 bond.  For positive
    # couplings, its three pairwise antiferromagnetic preferences conflict.
    frustrated_triangle = (0, 1, 6)

    plt.rcParams.update(
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.15))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.24, wspace=0.30)
    model_ax, letta_ax = axes

    model_ax.add_patch(
        Polygon(
            [positions[site] for site in frustrated_triangle],
            closed=True,
            facecolor=COLORS["frustrated"],
            edgecolor="none",
            alpha=0.27,
            zorder=0,
        )
    )
    _draw_edges(
        model_ax,
        nearest,
        positions,
        color=COLORS["j1"],
        linewidth=1.65,
        solid_capstyle="round",
        zorder=2,
    )
    _draw_edges(
        model_ax,
        diagonals,
        positions,
        color=COLORS["j2"],
        linewidth=1.35,
        linestyle=(0, (3.0, 2.0)),
        zorder=3,
    )

    # In the benchmark all nearest-neighbor physical legs are tied.  The J2
    # interactions remain in the Hamiltonian but are not physical ties.
    _draw_edges(
        letta_ax,
        diagonals,
        positions,
        color=COLORS["j2"],
        linewidth=0.9,
        linestyle=(0, (2.5, 2.0)),
        alpha=0.25,
        zorder=0,
    )
    _draw_edges(
        letta_ax,
        nearest,
        positions,
        color=COLORS["tie"],
        linewidth=4.6,
        alpha=0.82,
        solid_capstyle="round",
        zorder=2,
    )
    _draw_edges(
        letta_ax,
        virtual_edges,
        positions,
        color=COLORS["virtual"],
        linewidth=1.9,
        solid_capstyle="round",
        zorder=4,
    )

    for ax in axes:
        _draw_nodes(ax, positions)
        _format_lattice_axis(ax, nrows, ncols)

    model_ax.set_title(
        rf"Antiferromagnetic ${nrows}\times{ncols}$ $J_1$--$J_2$ model"
        "\n"
        rf"$J_1={j1:g},\ J_2={j2:g}$, open boundaries"
    )
    letta_ax.set_title(
        r"Graph LETTA used in the benchmark, $D=4$"
        "\n"
        r"all $J_1$ legs tied; row-wise snake backbone"
    )
    model_ax.text(-0.02, 0.94, "a", transform=model_ax.transAxes, fontweight="bold")
    letta_ax.text(-0.02, 0.94, "b", transform=letta_ax.transAxes, fontweight="bold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["j1"],
            linewidth=1.7,
            label=r"$J_1$ nearest-neighbor interaction",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["j2"],
            linewidth=1.4,
            linestyle=(0, (3.0, 2.0)),
            label=r"$J_2$ diagonal interaction",
        ),
        Patch(
            facecolor=COLORS["frustrated"],
            edgecolor="none",
            alpha=0.5,
            label="frustrated AFM triangle",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["tie"],
            linewidth=4.5,
            label="tied physical legs",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["virtual"],
            linewidth=2.0,
            label=r"virtual backbone, $D=4$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncols=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.5,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output stem (both PDF and PNG are written)",
    )
    args = parser.parse_args()
    pdf_path, png_path = plot_model(args.output)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
