#!/usr/bin/env python3
"""Plot Shastry-Sutherland LETTA/MPS energy benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
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
            rows.append(row)
    return rows


def load_data():
    rows = []
    rows.extend(read_rows(RESULTS_DIR / "ss_energy_Lx4_Ly4_dense.csv", size_label=r"$4\times4$"))
    rows.extend(read_rows(RESULTS_DIR / "ss_energy_Lx8_Ly4_D1_probe.csv", size_label=r"$8\times4$"))
    return rows


def subset(rows, *, size: str, method: str, bond_dim: int):
    out = [row for row in rows if row["size"] == size and row["method"] == method and row["D"] == bond_dim]
    return sorted(out, key=lambda row: row["jprime"])


def energy_gain(rows, *, size: str, bond_dim: int):
    mps = {row["jprime"]: row for row in subset(rows, size=size, method="MPS/DMRG snake", bond_dim=bond_dim)}
    letta = {row["jprime"]: row for row in subset(rows, size=size, method="LETTA dimer-first", bond_dim=bond_dim)}
    xs = sorted(set(mps) & set(letta))
    return np.asarray(xs), np.asarray([mps[x]["energy"] - letta[x]["energy"] for x in xs])


def main() -> None:
    rows = load_data()

    uplt.rc.update(
        {
            "font.size": 11.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
            "savefig.facecolor": "white",
        }
    )

    fig, axes = uplt.subplots(ncols=2, refwidth=3.35, refheight=2.62, wspace=0.78, share=False)

    ax = axes[0]
    styles = {
        ("MPS/DMRG snake", 1): ("#4f6d7a", "s", "--", r"MPS, $D=1$"),
        ("LETTA dimer-first", 1): ("#2d9465", "o", "-", r"LETTA, $D=1$"),
        ("MPS/DMRG snake", 2): ("#8a5a82", "D", "--", r"MPS, $D=2$"),
        ("LETTA dimer-first", 2): ("#c67d2d", "^", "-", r"LETTA, $D=2$"),
    }
    for (method, bond_dim), (color, marker, linestyle, label) in styles.items():
        data = subset(rows, size=r"$4\times4$", method=method, bond_dim=bond_dim)
        data = [row for row in data if row["error"] is not None]
        if not data:
            continue
        ax.plot(
            [row["jprime"] for row in data],
            [max(row["error"], ERROR_FLOOR) for row in data],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.65,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.format(
        title=r"$4\times4$ exact-reference error",
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
    ax.set_yticklabels(
        [
            r"$10^{-6}$",
            r"$10^{-4}$",
            r"$10^{-2}$",
            r"$10^{0}$",
            r"$10^{1}$",
        ]
    )
    ax.text(-0.13, 1.03, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5, fontweight="bold")

    ax = axes[1]
    gain_styles = {
        (r"$4\times4$", 1): ("#2d9465", "o", "-", r"$4\times4$, $D=1$"),
        (r"$4\times4$", 2): ("#c67d2d", "^", "-", r"$4\times4$, $D=2$"),
        (r"$8\times4$", 1): ("#4f6d7a", "s", "--", r"$8\times4$, $D=1$"),
    }
    for (size, bond_dim), (color, marker, linestyle, label) in gain_styles.items():
        x, y = energy_gain(rows, size=size, bond_dim=bond_dim)
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.65,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.axhline(0.0, color="#2f2f2f", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.format(
        title="same-D LETTA energy gain",
        xlabel=r"$J'/J$",
        ylabel=r"$E_{\rm MPS}-E_{\rm LETTA}$",
        xlim=(-0.04, 1.04),
        ylim=(-0.6, 8.8),
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
        gridalpha=0.18,
    )
    ax.text(-0.13, 1.03, "b", transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5, fontweight="bold")

    for ax in axes:
        ax.legend(loc="bottom", ncols=2, frame=False)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"shastry_sutherland_energy_benchmark.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
