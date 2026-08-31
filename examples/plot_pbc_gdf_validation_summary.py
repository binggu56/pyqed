#!/usr/bin/env python3
"""Plot precision convergence and final LiH periodic-GDF validation errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#5B5B5B"
HARTREE_TO_MEV = 27211.386245988


def _row(path):
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return payload["studies"][0]


def _rows(path):
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return payload["studies"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision_scan", type=Path)
    parser.add_argument("full_validation", type=Path)
    parser.add_argument("unpruned_scan", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_validation_summary.png"),
    )
    args = parser.parse_args()

    precision_rows = _rows(args.precision_scan)
    final = _row(args.full_validation)
    cutoff_rows = _rows(args.unpruned_scan)
    unpruned = min(cutoff_rows, key=lambda row: row["aux_min_exponent"])

    precision = np.asarray([row["gdf_precision"] for row in precision_rows])
    j_precision = np.asarray(
        [row["max_abs_J_error_meV"] for row in precision_rows]
    )
    k_precision = np.asarray(
        [row["max_abs_K_error_meV"] for row in precision_rows]
    )

    implementation_labels = ["J", "K", "KRHF", "GW QP", "BSE matrix", "TDA"]
    implementation_errors = np.asarray(
        [
            final["max_abs_J_error_meV"],
            final["max_abs_K_error_meV"],
            abs(final["native_krhf"]["energy_error_vs_pyscf_gdf_Ha"])
            * HARTREE_TO_MEV,
            final["gw"]["max_abs_qp_error_meV"],
            final["bse"]["max_abs_A_error_Ha"] * HARTREE_TO_MEV,
            final["bse"]["max_abs_tda_eigenvalue_error_meV"],
        ]
    )
    fit_shift = abs(
        final["pyscf_gdf_krhf_energy_Ha"]
        - unpruned["pyscf_gdf_krhf_energy_Ha"]
    ) * HARTREE_TO_MEV

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)

    axes[0].loglog(precision, j_precision, "o-", color=BLUE, label=r"$J$")
    axes[0].loglog(precision, k_precision, "s--", color=ORANGE, label=r"$K$")
    axes[0].axhline(1.0e-3, color=GRAY, ls=":", lw=1.0, label="target")
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel="GDF precision",
        ylabel="Max difference (meV)",
        title=r"a  Precision at $\alpha_{\min}^{\rm aux}=0.075$",
    )
    axes[0].legend(frameon=False)

    positions = np.arange(len(implementation_labels) + 1)
    values = np.r_[implementation_errors, fit_shift]
    colors = [BLUE] * len(implementation_labels) + [GREEN]
    axes[1].bar(positions, values, color=colors, width=0.72)
    axes[1].axhline(1.0e-3, color=GRAY, ls=":", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        positions,
        implementation_labels + ["fit-space\nshift"],
        rotation=35,
        ha="right",
    )
    axes[1].set(
        ylabel="Energy scale (meV)",
        title="b  Final validation",
    )

    for axis in axes:
        axis.grid(alpha=0.2, lw=0.6, which="both", axis="y")
        axis.spines[["top", "right"]].set_visible(False)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure: {output}")
    print(f"pdf: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
