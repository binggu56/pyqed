#!/usr/bin/env python3
"""Plot high-bond MPS/DMRG Shastry-Sutherland benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

import ultraplot as uplt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

ERROR_FLOOR = 1.0e-6


def read_rows(path: Path, *, size_label: str):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if not row.get("energy"):
                continue
            if row.get("converged", "").lower() in {"false", "0", "no"}:
                continue
            row = dict(row)
            row["size"] = size_label
            row["D"] = int(row["D"])
            row["jprime"] = float(row["jprime"])
            row["energy"] = float(row["energy"])
            row["exact"] = None if row["exact"] == "" else float(row["exact"])
            row["error"] = None if row["error"] == "" else float(row["error"])
            row["seconds"] = None if row["seconds"] == "" else float(row["seconds"])
            row["sweeps_completed"] = (
                None if row["sweeps_completed"] == "" else int(row["sweeps_completed"])
            )
            rows.append(row)
    return rows


def load_data():
    rows = []
    rows.extend(read_rows(RESULTS_DIR / "ss_energy_Lx4_Ly4_dense.csv", size_label=r"$4\times4$"))
    rows.extend(
        read_rows(RESULTS_DIR / "ss_energy_Lx4_Ly4_mps_highD.csv", size_label=r"$4\times4$")
    )
    return rows


def subset(rows, *, method: str, bond_dim: int):
    data = [
        row
        for row in rows
        if row["method"] == method and row["D"] == bond_dim and row["error"] is not None
    ]
    return sorted(data, key=lambda row: row["jprime"])


def error_values(data):
    return [max(float(row["error"]), ERROR_FLOOR) for row in data]


def main() -> None:
    rows = load_data()

    uplt.rc.update(
        {
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 13.8,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 10.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
            "savefig.facecolor": "white",
        }
    )

    fig, axes = uplt.subplots(ncols=2, refwidth=3.75, refheight=2.95, wspace=0.66, share=False)

    mps_styles = {
        1: ("#4f6d7a", "s", ":", r"MPS, $D=1$"),
        2: ("#8a5a82", "D", "--", r"MPS, $D=2$"),
        4: ("#2a9d8f", "v", "-.", r"MPS, $D=4$"),
        8: ("#e9a53f", "P", "-", r"MPS, $D=8$"),
        16: ("#2f2f2f", "o", "-", r"MPS, $D=16$"),
    }

    ax = axes[0]
    for bond_dim, (color, marker, linestyle, label) in mps_styles.items():
        data = subset(rows, method="MPS/DMRG snake", bond_dim=bond_dim)
        ax.plot(
            [row["jprime"] for row in data],
            error_values(data),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.85,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.axvspan(0.65, 0.75, color="#d7d7d7", alpha=0.38, zorder=0)
    ax.format(
        title=r"snake MPS improves with $D$",
        xlabel=r"$J'/J$",
        ylabel=r"$E-E_{\rm ED}$",
        yscale="log",
        ylim=(ERROR_FLOOR, 1.0e1),
        xlim=(-0.04, 1.04),
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
        gridalpha=0.18,
    )
    ax.set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0, 1.0e1])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{1}$"])
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=17, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=2, frame=False)

    ax = axes[1]
    comparison_styles = {
        ("LETTA dimer-first", 1): ("#2d9465", "o", "-", r"LETTA, $D=1$"),
        ("LETTA dimer-first", 2): ("#c67d2d", "^", "-", r"LETTA, $D=2$"),
        ("MPS/DMRG snake", 16): ("#2f2f2f", "o", "--", r"MPS, $D=16$"),
    }
    for (method, bond_dim), (color, marker, linestyle, label) in comparison_styles.items():
        data = subset(rows, method=method, bond_dim=bond_dim)
        ax.plot(
            [row["jprime"] for row in data],
            error_values(data),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.9,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.axvspan(0.65, 0.75, color="#d7d7d7", alpha=0.38, zorder=0)
    ax.format(
        title=r"high-$D$ MPS baseline",
        xlabel=r"$J'/J$",
        ylabel=r"$E-E_{\rm ED}$",
        yscale="log",
        ylim=(ERROR_FLOOR, 1.0e1),
        xlim=(-0.04, 1.04),
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
        gridalpha=0.18,
    )
    ax.set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0, 1.0e1])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{1}$"])
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes, ha="left", va="bottom", fontsize=17, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=1, frame=False)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shastry_sutherland_mps_highD.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
