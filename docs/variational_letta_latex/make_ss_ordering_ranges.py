#!/usr/bin/env python3
"""Plot Shastry-Sutherland 1D bond ranges for common cylinder sizes."""

from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.lines import Line2D
import ultraplot as uplt

from compare_ss_orderings import summarize


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

WIDTHS = (4, 6, 8, 10, 12, 14)


def collect_rows():
    rows = []
    for ly in WIDTHS:
        lx = 2 * ly
        for ordering, dimer, jprime in summarize(lx, ly):
            for bond_kind, stats in [
                ("strong dimer $J$", dimer),
                ("interdimer $J'$", jprime),
            ]:
                rows.append(
                    {
                        "Lx": lx,
                        "Ly": ly,
                        "N": lx * ly,
                        "ordering": ordering,
                        "bond_kind": bond_kind,
                        "mean": stats["mean"],
                        "max": stats["max"],
                        "adjacent_fraction": stats["adjacent_fraction"],
                    }
                )
    return rows


def write_csv(rows) -> None:
    fields = ["Lx", "Ly", "N", "ordering", "bond_kind", "mean", "max", "adjacent_fraction"]
    with (FIG_DIR / "shastry_sutherland_ordering_ranges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def series(rows, *, ordering: str, bond_kind: str, metric: str):
    values = []
    for ly in WIDTHS:
        match = [
            row
            for row in rows
            if row["Ly"] == ly and row["ordering"] == ordering and row["bond_kind"] == bond_kind
        ]
        if len(match) != 1:
            raise RuntimeError(f"missing row for {ordering}, {bond_kind}, Ly={ly}")
        values.append(match[0][metric])
    return values


def main() -> None:
    rows = collect_rows()
    write_csv(rows)

    uplt.rc.update(
        {
            "font.size": 11.2,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.3,
            "ytick.labelsize": 10.3,
            "legend.fontsize": 9.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
            "savefig.facecolor": "white",
        }
    )

    styles = {
        ("dimer-first", "strong dimer $J$"): {
            "color": "#2d9465",
            "marker": "o",
            "linestyle": "-",
            "label": r"dimer-first, $J$",
        },
        ("snake", "strong dimer $J$"): {
            "color": "#4f6d7a",
            "marker": "s",
            "linestyle": "-",
            "label": r"snake, $J$",
        },
        ("dimer-first", "interdimer $J'$"): {
            "color": "#c67d2d",
            "marker": "^",
            "linestyle": "--",
            "label": r"dimer-first, $J'$",
        },
        ("snake", "interdimer $J'$"): {
            "color": "#8a5a82",
            "marker": "D",
            "linestyle": "--",
            "label": r"snake, $J'$",
        },
    }

    fig, axes = uplt.subplots(ncols=2, refwidth=3.35, refheight=2.72, wspace=0.42, share=False)

    panel_specs = [
        ("mean", "1D bond distance", (0.0, 15.0), "average range"),
        ("max", "", (0.0, 31.0), "worst-case range"),
    ]

    for panel_index, (ax, (metric, ylabel, ylim, title)) in enumerate(zip(axes, panel_specs)):
        for key, style in styles.items():
            y = series(rows, ordering=key[0], bond_kind=key[1], metric=metric)
            ax.plot(
                WIDTHS,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.65,
                markersize=5.3,
                markeredgewidth=0.8,
                markeredgecolor="white",
                label=style["label"],
            )
        ax.axhline(1, color="#2f2f2f", linewidth=0.8, linestyle=":", alpha=0.55)
        ax.text(
            -0.12,
            1.03,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
        )
        ax.format(
            title=title,
            xlabel=r"cylinder circumference $L_y$",
            ylabel=ylabel,
            xlim=(3.5, 14.5),
            ylim=ylim,
            xticks=WIDTHS,
            grid=True,
            gridalpha=0.18,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.65,
            markersize=5.3,
            markeredgewidth=0.8,
            markeredgecolor="white",
            label=style["label"],
        )
        for style in styles.values()
    ]
    fig.legend(handles=handles, loc="bottom", ncols=4, frame=False, columnspacing=1.2)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shastry_sutherland_ordering_ranges.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
