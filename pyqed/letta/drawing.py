"""Small, dependency-light diagrams for frontier LETTA states."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def draw_frontier_letta(
    state,
    path=None,
    *,
    show_physical=True,
    show_bond_dims=True,
    show_tie_labels=False,
    figsize=None,
):
    """Draw the sequential backbone, physical legs, and tied physical edges."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MplPath

    nsites = len(state.sites)
    if figsize is None:
        figsize = (max(5.0, 0.92 * nsites + 1.2), 3.4)
    fig, ax = plt.subplots(figsize=figsize)
    positions = {
        site: np.array((float(site), 0.0))
        for site in range(nsites)
    }

    # Virtual backbone.
    for site in range(nsites - 1):
        left = positions[site]
        right = positions[site + 1]
        ax.plot(
            (left[0], right[0]),
            (left[1], right[1]),
            color="#222222",
            linewidth=1.8,
            solid_capstyle="round",
            zorder=1,
        )
        if show_bond_dims:
            ax.text(
                0.5 * (left[0] + right[0]),
                -0.24,
                rf"$D_{{{site + 1}}}={state.bond_dims[site + 1]}$",
                ha="center",
                va="top",
                fontsize=8,
                color="#4B5563",
            )

    # Tied copies of future physical indices.
    edges = tuple(
        getattr(
            state,
            "graph",
            tuple(
                (site, parent)
                for site, parents in enumerate(state.parent_sets)
                for parent in parents
            ),
        )
    )
    routed_edges = []
    side_counts = {1: 0, -1: 0}

    def edges_cross(first, second):
        a, b = sorted(first)
        c, d = sorted(second)
        return (a < c < b < d) or (c < a < d < b)

    for edge in edges:
        edge = tuple(sorted((int(edge[0]), int(edge[1]))))
        crossing_counts = {
            side: sum(
                prior_side == side and edges_cross(edge, prior)
                for prior, prior_side, _lane in routed_edges
            )
            for side in (1, -1)
        }
        side = min(
            (1, -1),
            key=lambda candidate: (
                crossing_counts[candidate],
                side_counts[candidate],
                candidate < 0,
            ),
        )
        lane = side_counts[side]
        side_counts[side] += 1
        routed_edges.append((edge, side, lane))

    route_heights = []
    tied_junctions = set()
    for (left_site, right_site), side, lane in routed_edges:
        left = positions[left_site]
        right = positions[right_site]
        span = right_site - left_site
        height = side * (0.66 + 0.18 * span + 0.24 * lane)
        route_heights.append(height)
        start = left + np.array((-0.08, 0.17 * side))
        # The tie terminates at the owner's physical-leg junction.  This
        # denotes one shared index rather than a second independent s_j.
        end = right + np.array((0.24, 0.12))
        tied_junctions.add(right_site)
        vertices = (
            tuple(start),
            (start[0] + 0.24, height),
            (end[0] + (-0.24 if side > 0 else 0.24), height),
            tuple(end),
        )
        patch = PathPatch(
            MplPath(
                vertices,
                (
                    MplPath.MOVETO,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                ),
            ),
            facecolor="none",
            edgecolor="#0072B2",
            linewidth=1.45,
            linestyle=(0, (3.0, 2.0)),
            capstyle="round",
            zorder=2,
        )
        ax.add_patch(patch)
        if show_tie_labels:
            ax.text(
                0.5 * (left[0] + right[0]),
                height + 0.06 * side,
                "tie",
                ha="center",
                va="bottom" if side > 0 else "top",
                fontsize=9,
                color="#0072B2",
            )

    # Physical legs point to the upper right and remain distinct from ties.
    if show_physical:
        for site, position in positions.items():
            junction = position + np.array((0.24, 0.12))
            endpoint = position + np.array((0.43, 0.47))
            ax.plot(
                (junction[0], endpoint[0]),
                (junction[1], endpoint[1]),
                color="#6B7280",
                linewidth=1.0,
                zorder=3,
            )
            if site in tied_junctions:
                ax.add_patch(
                    Circle(
                        junction,
                        radius=0.035,
                        facecolor="#FFFFFF",
                        edgecolor="#0072B2",
                        linewidth=1.1,
                        zorder=4,
                    )
                )
            ax.text(
                endpoint[0] + 0.02,
                endpoint[1] + 0.01,
                rf"$s_{{{site}}}$",
                ha="left",
                va="bottom",
                fontsize=9,
                color="#4B5563",
            )

    for site, position in positions.items():
        width, height = 0.46, 0.36
        ax.add_patch(
            FancyBboxPatch(
                (position[0] - width / 2, position[1] - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#F6F6F4",
                edgecolor="#222222",
                linewidth=1.15,
                zorder=5,
            )
        )
        ax.text(
            position[0],
            position[1],
            rf"$A^{{{site}}}$",
            ha="center",
            va="center",
            fontsize=10,
            color="#222222",
            zorder=6,
        )

    legend = (
        Line2D((0,), (0,), color="#222222", linewidth=1.8, label="backbone"),
        Line2D(
            (0,),
            (0,),
            color="#0072B2",
            linewidth=1.45,
            linestyle=(0, (3.0, 2.0)),
            label="physical tie",
        ),
    )
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    maximum_height = max((height for height in route_heights if height > 0), default=0.8)
    minimum_height = min((height for height in route_heights if height < 0), default=-0.5)
    ax.set_xlim(-0.55, max(float(nsites - 1), 0.0) + 0.75)
    ax.set_ylim(min(-0.72, minimum_height - 0.35), maximum_height + 0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()

    if path is not None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    return fig, ax


__all__ = ["draw_frontier_letta"]
