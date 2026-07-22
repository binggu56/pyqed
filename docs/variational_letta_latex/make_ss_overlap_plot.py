#!/usr/bin/env python3
"""Plot LETTA adjacent-overlap diagnostics for the Shastry-Sutherland model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ultraplot as uplt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def read_rows(path: Path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "D": int(row["D"]),
                    "jprime_mid": float(row["jprime_mid"]),
                    "jprime_left": float(row["jprime_left"]),
                    "jprime_right": float(row["jprime_right"]),
                    "overlap_abs": float(row["overlap_abs"]),
                    "infidelity": float(row["infidelity"]),
                    "fidelity_susceptibility": float(row["fidelity_susceptibility"]),
                }
            )
    return rows


def subset(rows, bond_dim: int):
    return sorted([row for row in rows if row["D"] == int(bond_dim)], key=lambda row: row["jprime_mid"])


def parse_ints(text: str):
    return [int(item) for item in str(text).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="ss_letta_overlap_Lx4_Ly4.csv")
    parser.add_argument("--output-prefix", default="shastry_sutherland_letta_overlap")
    parser.add_argument("--size-label", default=r"$4\times4$")
    parser.add_argument("--bond-dims", default=None)
    parser.add_argument("--left-panel", choices=("infidelity", "overlap"), default="infidelity")
    parser.add_argument("--chi-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--auto-y", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = RESULTS_DIR / input_path
    rows = read_rows(input_path)
    bond_dims = sorted({row["D"] for row in rows})
    if args.bond_dims:
        requested = set(parse_ints(args.bond_dims))
        bond_dims = [bond_dim for bond_dim in bond_dims if bond_dim in requested]
    if not bond_dims:
        raise ValueError("No overlap rows remain after filtering bond dimensions.")

    uplt.rc.update(
        {
            "font.size": 11.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 420,
            "savefig.facecolor": "white",
        }
    )

    fig, axes = uplt.subplots(ncols=2, refwidth=3.35, refheight=2.45, wspace=0.72, share=False)
    base_styles = {
        1: ("#2d9465", "o", r"LETTA, $D=1$"),
        2: ("#c67d2d", "^", r"LETTA, $D=2$"),
        3: ("#4f6d7a", "s", r"LETTA, $D=3$"),
        4: ("#8a5a82", "D", r"LETTA, $D=4$"),
    }

    ax = axes[0]
    ax.axvspan(0.66, 0.77, color="#8a8a8a", alpha=0.12, zorder=0)
    left_key = "overlap_abs" if args.left_panel == "overlap" else "infidelity"
    left_values = []
    for bond_dim in bond_dims:
        color, marker, label = base_styles.get(bond_dim, ("#444444", "o", rf"LETTA, $D={bond_dim}$"))
        data = subset(rows, bond_dim)
        yvalues = [row[left_key] for row in data]
        if args.left_panel == "infidelity":
            yvalues = [max(value, 1.0e-5) for value in yvalues]
        left_values.extend(yvalues)
        ax.plot(
            [row["jprime_mid"] for row in data],
            yvalues,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.65,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    if args.left_panel == "overlap":
        left_title = "adjacent overlap"
        left_ylabel = r"$|\langle\Psi(\lambda)|\Psi(\lambda+\delta)\rangle|$"
        left_yscale = "linear"
        left_ylim = None
        if args.auto_y:
            span = max(left_values) - min(left_values)
            pad = max(0.08 * span, 5.0e-4)
            left_ylim = (max(0.0, min(left_values) - pad), min(1.005, max(left_values) + pad))
    else:
        left_title = "adjacent fidelity loss"
        left_ylabel = r"$1-F(\lambda,\lambda+\delta)$"
        left_yscale = "log"
        left_ylim = (3.0e-5, 1.2)
    ax.format(
        title=rf"{args.size_label} {left_title}",
        xlabel=r"midpoint $J'/J$",
        ylabel=left_ylabel,
        yscale=left_yscale,
        xlim=(-0.02, 1.02),
        ylim=left_ylim,
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
        gridalpha=0.18,
    )
    if args.left_panel == "infidelity":
        ax.set_yticks([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
        ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"])
    ax.text(-0.13, 1.03, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5, fontweight="bold")

    ax = axes[1]
    ax.axvspan(0.66, 0.77, color="#8a8a8a", alpha=0.12, zorder=0)
    chi_values = []
    for bond_dim in bond_dims:
        color, marker, label = base_styles.get(bond_dim, ("#444444", "o", rf"LETTA, $D={bond_dim}$"))
        data = subset(rows, bond_dim)
        chi_values.extend(row["fidelity_susceptibility"] for row in data)
        ax.plot(
            [row["jprime_mid"] for row in data],
            [row["fidelity_susceptibility"] for row in data],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.65,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    chi_ylim = None
    if args.auto_y and args.chi_scale == "linear":
        span = max(chi_values) - min(chi_values)
        pad = max(0.12 * span, 5.0e-4)
        chi_ylim = (max(0.0, min(chi_values) - pad), max(chi_values) + pad)
    elif args.chi_scale == "log":
        chi_ylim = (4.0e-3, 60.0)
    ax.format(
        title=rf"{args.size_label} finite-step susceptibility",
        xlabel=r"midpoint $J'/J$",
        ylabel=r"$\chi_F$",
        yscale=args.chi_scale,
        xlim=(-0.02, 1.02),
        ylim=chi_ylim,
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
        gridalpha=0.18,
    )
    if args.chi_scale == "log":
        ax.set_yticks([1.0e-2, 1.0e-1, 1.0e0, 1.0e1])
        ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$"])
    ax.text(-0.13, 1.03, "b", transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5, fontweight="bold")

    legend_cols = min(2, len(bond_dims))
    for ax in axes:
        ax.legend(loc="bottom", ncols=legend_cols, frame=False)

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{args.output_prefix}.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
