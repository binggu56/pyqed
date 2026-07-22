#!/usr/bin/env python3
"""Illustrate dimer-first site ordering for Shastry-Sutherland LETTA."""

from __future__ import annotations

from pathlib import Path

from matplotlib.lines import Line2D
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


SITES_2D = {
    1: (0.0, 0.0),
    2: (0.85, 0.85),
    3: (2.05, 0.0),
    4: (2.90, 0.85),
    5: (4.10, 0.0),
    6: (4.95, 0.85),
    7: (0.85, 2.05),
    8: (0.0, 2.90),
    9: (2.90, 2.05),
    10: (2.05, 2.90),
    11: (4.95, 2.05),
    12: (4.10, 2.90),
}

DIMER_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
INTERDIMER_EDGES = [
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (8, 10),
    (10, 12),
    (7, 9),
    (9, 11),
    (1, 8),
    (2, 7),
    (3, 10),
    (4, 9),
    (5, 12),
    (6, 11),
]


def draw_site(ax, xy, label, *, radius=170):
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=radius,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.25,
        zorder=4,
    )
    ax.text(xy[0], xy[1], str(label), ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=5)


def draw_bond(ax, coords, pair, *, color, linewidth, linestyle="-", zorder=1):
    (x0, y0), (x1, y1) = coords[pair[0]], coords[pair[1]]
    ax.plot(
        [x0, x1],
        [y0, y1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=zorder,
    )


def main() -> None:
    uplt.rc.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    dimer_color = "#2d9465"
    inter_color = "#9aa0a6"
    order_color = "#3b6ea8"

    fig, axes = uplt.subplots(ncols=2, refwidth=3.2, refheight=2.45, wspace=0.58, share=False)

    ax = axes[0]
    for edge in INTERDIMER_EDGES:
        draw_bond(ax, SITES_2D, edge, color=inter_color, linewidth=1.1, zorder=1)
    for pair in DIMER_PAIRS:
        draw_bond(ax, SITES_2D, pair, color=dimer_color, linewidth=3.0, zorder=2)
    for label, xy in SITES_2D.items():
        draw_site(ax, xy, label)
    ax.text(-0.08, 1.04, "a", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=11)
    ax.format(title="number sites by dimers", xlim=(-0.45, 5.40), ylim=(-0.35, 3.25), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    chain_coords = {site: (site - 1, 0.0) for site in range(1, 13)}
    for left, right in [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]:
        draw_bond(ax, chain_coords, (left, right), color=order_color, linewidth=1.35, linestyle=":", zorder=1)
    for pair in DIMER_PAIRS:
        draw_bond(ax, chain_coords, pair, color=dimer_color, linewidth=3.0, zorder=2)
    for site, xy in chain_coords.items():
        draw_site(ax, xy, site, radius=145)
    for dimer_index, (left, right) in enumerate(DIMER_PAIRS, start=1):
        mid = 0.5 * (chain_coords[left][0] + chain_coords[right][0])
        ax.text(mid, -0.55, rf"$d_{dimer_index}$", ha="center", va="top", fontsize=8.8)
    ax.text(-0.08, 1.04, "b", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=11)
    ax.format(title="1D LETTA order", xlim=(-0.55, 11.55), ylim=(-0.95, 0.55), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], color=dimer_color, linewidth=3.0, label=r"strong dimer $J$"),
        Line2D([0], [0], color=inter_color, linewidth=1.2, label=r"physical $J'$ link"),
        Line2D([0], [0], color=order_color, linewidth=1.4, linestyle=":", label="1D ordering link"),
    ]
    fig.legend(handles=handles, loc="bottom", ncols=3, frame=False, columnspacing=1.25)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shastry_sutherland_dimer_ordering.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
