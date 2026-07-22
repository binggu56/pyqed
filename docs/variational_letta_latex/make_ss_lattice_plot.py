#!/usr/bin/env python3
"""Generate a Shastry-Sutherland lattice schematic."""

from __future__ import annotations

from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def shastry_sutherland_dimers(nx: int, ny: int):
    """Return non-overlapping orthogonal dimer bonds on a square site grid."""
    dimers = []
    for x in range(nx - 1):
        for y in range(ny - 1):
            if (x + y) % 2 != 0:
                continue
            if x % 2 == 0:
                dimers.append(((x, y), (x + 1, y + 1)))
            else:
                dimers.append(((x + 1, y), (x, y + 1)))
    return dimers


def main() -> None:
    nx, ny = 7, 6
    sites = np.array([(x, y) for y in range(ny) for x in range(nx)])
    dimers = shastry_sutherland_dimers(nx, ny)

    uplt.rc.update(
        {
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = uplt.subplots(refwidth=4.3, refheight=3.2)

    square_color = "#8b8f97"
    dimer_color = "#2d9465"

    for x in range(nx):
        for y in range(ny):
            if x + 1 < nx:
                ax.plot([x, x + 1], [y, y], color=square_color, linewidth=1.05, zorder=1)
            if y + 1 < ny:
                ax.plot([x, x], [y, y + 1], color=square_color, linewidth=1.05, zorder=1)

    for (x0, y0), (x1, y1) in dimers:
        ax.plot([x0, x1], [y0, y1], color=dimer_color, linewidth=3.2, solid_capstyle="round", zorder=2)

    ax.scatter(
        sites[:, 0],
        sites[:, 1],
        s=42,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=3,
    )

    ax.format(
        title="Shastry-Sutherland lattice",
        xlim=(-0.45, nx - 0.55),
        ylim=(-0.45, ny - 0.55),
        aspect="equal",
        xticks=[],
        yticks=[],
        xlabel="",
        ylabel="",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], color=dimer_color, linewidth=3.2, label=r"strong dimer bond $J$"),
        Line2D([0], [0], color=square_color, linewidth=1.4, label=r"interdimer bond $J'$"),
    ]
    fig.legend(handles=handles, loc="bottom", ncols=2, frame=False, columnspacing=1.6)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shastry_sutherland_lattice.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
