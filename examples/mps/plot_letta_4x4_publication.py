"""Draw a publication-ready 4x4 frontier-LETTA contraction diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#222222",
    "muted": "#6B7280",
    "tie": "#0072B2",
    "frontier": "#D55E00",
    "highlight": "#E69F00",
    "done": "#EAF3F8",
    "waiting": "#F6F6F4",
    "white": "#FFFFFF",
}


def site_index(row: int, col: int, ncols: int = 4) -> int:
    """Return the row-wise snake-order index."""
    return row * ncols + (col if row % 2 == 0 else ncols - 1 - col)


def site_position(index: int, ncols: int = 4) -> np.ndarray:
    """Map a snake-order index to its lattice coordinate."""
    row, offset = divmod(index, ncols)
    col = offset if row % 2 == 0 else ncols - 1 - offset
    return np.array([1.18 * col, 1.18 * (3 - row)], dtype=float)


def add_bond(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    curvature: float = 0.0,
    zorder: float = 1.0,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={curvature}",
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        capstyle="round",
        joinstyle="round",
        shrinkA=13.5,
        shrinkB=13.5,
        zorder=zorder,
    )
    ax.add_patch(patch)


def curved_midpoint(
    start: np.ndarray, end: np.ndarray, curvature: float
) -> np.ndarray:
    """Approximate the midpoint of Matplotlib's arc3 curve."""
    delta = end - start
    length = np.linalg.norm(delta)
    normal = np.array([-delta[1], delta[0]]) / length
    return 0.5 * (start + end) - 0.50 * curvature * length * normal


def draw_diagram(output_dir: Path) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(5.55, 4.75), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_facecolor(COLORS["white"])

    positions = {i: site_position(i) for i in range(16)}
    cut_y = 1.77

    # A restrained cut band makes the contraction state readable without
    # suggesting that the frontier itself is another tensor-network bond.
    ax.add_patch(
        Rectangle(
            (-0.62, cut_y - 0.105),
            4.72,
            0.21,
            facecolor=COLORS["frontier"],
            edgecolor="none",
            alpha=0.075,
            zorder=-3,
        )
    )
    horizontal_edges = [
        (site_index(row, col), site_index(row, col + 1))
        for row in range(4)
        for col in range(3)
    ]
    vertical_edges = [
        (site_index(row, col), site_index(row + 1, col))
        for row in range(3)
        for col in range(4)
    ]
    nearest_edges = horizontal_edges + vertical_edges
    backbone_edges = [(i, i + 1) for i in range(15)]
    nearest_sets = {frozenset(edge) for edge in nearest_edges}
    backbone_sets = {frozenset(edge) for edge in backbone_edges}

    tie_curvature: dict[frozenset[int], float] = {}
    tie_orientation: dict[frozenset[int], tuple[int, int]] = {}
    tie_endpoints: dict[frozenset[int], tuple[np.ndarray, np.ndarray]] = {}
    for a, b in nearest_edges:
        edge = frozenset((a, b))
        tie_orientation[edge] = (a, b)
        pa, pb = positions[a], positions[b]
        shared = edge in backbone_sets
        draw_pa, draw_pb = pa, pb
        if shared and abs(pa[0] - pb[0]) < 1e-9:
            port_offset = np.array([-0.060, 0.0])
            draw_pa, draw_pb = pa + port_offset, pb + port_offset
        tie_endpoints[edge] = (draw_pa, draw_pb)
        if abs(pa[1] - pb[1]) < 1e-9:
            rad = 0.13 if shared else 0.035
        else:
            col = int(round(pa[0] / 1.18))
            rad = (0.052 if col % 2 == 0 else -0.052)
            if shared:
                rad = 0.13
        tie_curvature[edge] = rad
        add_bond(
            ax,
            draw_pa,
            draw_pb,
            color=COLORS["tie"],
            linewidth=1.25,
            linestyle=(0, (3.1, 2.0)),
            curvature=rad,
            zorder=1,
        )

    # Draw the snake backbone separately. On shared lattice edges its opposite
    # curvature makes the tied and variational bonds visibly distinct.
    backbone_curvature: dict[frozenset[int], float] = {}
    backbone_endpoints: dict[
        frozenset[int], tuple[np.ndarray, np.ndarray]
    ] = {}
    for a, b in backbone_edges:
        edge = frozenset((a, b))
        pa, pb = positions[a], positions[b]
        draw_pa, draw_pb = pa, pb
        if abs(pa[0] - pb[0]) < 1e-9:
            port_offset = np.array([0.060, 0.0])
            draw_pa, draw_pb = pa + port_offset, pb + port_offset
        backbone_endpoints[edge] = (draw_pa, draw_pb)
        if edge in nearest_sets:
            tie_rad = tie_curvature[edge]
            same_direction = tie_orientation[edge] == (a, b)
            rad = (-0.82 if same_direction else 0.82) * tie_rad
        else:
            rad = 0.0
        backbone_curvature[edge] = rad
        add_bond(
            ax,
            draw_pa,
            draw_pb,
            color=COLORS["ink"],
            linewidth=1.65,
            curvature=rad,
            zorder=2,
        )

    # Frontier circles mark the four tied indices crossing the cut.
    crossing_ties = [
        (site_index(1, col), site_index(2, col)) for col in range(4)
    ]
    for a, b in crossing_ties:
        edge = frozenset((a, b))
        start, end = tie_endpoints[edge]
        point = curved_midpoint(start, end, tie_curvature[edge])
        ax.add_patch(
            Circle(
                point,
                radius=0.072,
                facecolor=COLORS["white"],
                edgecolor=COLORS["frontier"],
                linewidth=1.35,
                zorder=5,
            )
        )

    # The backbone has its own independent open index at the same cut.
    backbone_cut = frozenset((7, 8))
    backbone_start, backbone_end = backbone_endpoints[backbone_cut]
    backbone_point = curved_midpoint(
        backbone_start, backbone_end, backbone_curvature[backbone_cut]
    )
    ax.scatter(
        [backbone_point[0]],
        [backbone_point[1]],
        marker="D",
        s=36,
        facecolor=COLORS["white"],
        edgecolor=COLORS["frontier"],
        linewidth=1.35,
        zorder=6,
    )

    # Physical legs are deliberately short and diagonal so they cannot be
    # confused with lattice ties.
    leg_offset = np.array([0.30, 0.34])
    for i, pos in positions.items():
        endpoint = pos + leg_offset
        ax.plot(
            [pos[0], endpoint[0]],
            [pos[1], endpoint[1]],
            color=COLORS["muted"],
            linewidth=0.85,
            zorder=2.5,
        )
        ax.add_patch(
            Circle(
                endpoint,
                radius=0.035,
                facecolor=COLORS["white"],
                edgecolor=COLORS["muted"],
                linewidth=0.8,
                zorder=4,
            )
        )

    node_width = 0.48
    node_height = 0.39
    for i, pos in positions.items():
        processed = i < 8
        next_site = i == 8
        ax.add_patch(
            FancyBboxPatch(
                (pos[0] - node_width / 2, pos[1] - node_height / 2),
                node_width,
                node_height,
                boxstyle="round,pad=0.025,rounding_size=0.075",
                facecolor=COLORS["done"] if processed else COLORS["waiting"],
                edgecolor=COLORS["highlight"] if next_site else COLORS["ink"],
                linewidth=1.65 if next_site else 1.05,
                zorder=3,
            )
        )
        ax.text(
            pos[0],
            pos[1] - 0.005,
            rf"$A^{{[{i}]}}$",
            ha="center",
            va="center",
            fontsize=9.3,
            color=COLORS["ink"],
            zorder=4,
        )

    ax.text(
        -0.50,
        2.93,
        "contracted",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
        rotation=90,
    )
    ax.text(
        -0.50,
        0.61,
        "remaining",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
        rotation=90,
    )
    ax.text(
        3.94,
        cut_y,
        "frontier",
        ha="left",
        va="center",
        fontsize=8.2,
        color=COLORS["frontier"],
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.38, 2.12),
            (0.38, 1.41),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.15,
            color=COLORS["frontier"],
            shrinkA=0,
            shrinkB=0,
            zorder=7,
        )
    )
    frontier_circle = Line2D(
        [0],
        [0],
        marker="o",
        markersize=5.5,
        markerfacecolor=COLORS["white"],
        markeredgecolor=COLORS["frontier"],
        markeredgewidth=1.2,
        linestyle="none",
    )
    physical_leg = Line2D(
        [0],
        [0],
        color=COLORS["muted"],
        linewidth=0.85,
        marker="o",
        markerfacecolor=COLORS["white"],
        markeredgecolor=COLORS["muted"],
        markersize=4.2,
    )
    handles = [
        Line2D([0], [0], color=COLORS["ink"], linewidth=1.65),
        Line2D(
            [0],
            [0],
            color=COLORS["tie"],
            linewidth=1.25,
            linestyle=(0, (3.1, 2.0)),
        ),
        frontier_circle,
        physical_leg,
    ]
    labels = ["backbone", r"tied $J_1$ bond", "open frontier index", "physical leg"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.49, -0.010),
        ncol=4,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.2,
        handletextpad=0.55,
        fontsize=7.7,
    )

    ax.set_xlim(-0.82, 4.70)
    ax.set_ylim(-0.28, 4.00)
    ax.set_aspect("equal")
    ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "letta_4x4_contraction.pdf"
    png_path = output_dir / "letta_4x4_contraction.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf"),
        help="Directory for the PDF and review PNG (default: output/pdf).",
    )
    args = parser.parse_args()
    pdf_path, png_path = draw_diagram(args.output_dir)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
