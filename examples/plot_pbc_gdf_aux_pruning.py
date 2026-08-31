#!/usr/bin/env python3
"""Plot LiH periodic-GDF auxiliary pruning and precision convergence."""

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


def _studies(path):
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return payload["studies"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cutoff_scan", type=Path)
    parser.add_argument("precision_scan", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_aux_pruning.png"),
    )
    args = parser.parse_args()

    cutoff_rows = _studies(args.cutoff_scan)
    precision_rows = _studies(args.precision_scan)
    cutoffs = np.asarray([row["aux_min_exponent"] for row in cutoff_rows])
    j_cutoff = np.asarray([row["max_abs_J_error_meV"] for row in cutoff_rows])
    k_cutoff = np.asarray([row["max_abs_K_error_meV"] for row in cutoff_rows])
    base_energy = float(cutoff_rows[0]["pyscf_gdf_krhf_energy_Ha"])
    energy_drift = HARTREE_TO_MEV * np.asarray(
        [row["pyscf_gdf_krhf_energy_Ha"] - base_energy for row in cutoff_rows]
    )

    precision = np.asarray([row["gdf_precision"] for row in precision_rows])
    j_precision = np.asarray(
        [row["max_abs_J_error_meV"] for row in precision_rows]
    )
    k_precision = np.asarray(
        [row["max_abs_K_error_meV"] for row in precision_rows]
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)

    axes[0].semilogy(cutoffs, j_cutoff, "o-", color=BLUE, label=r"$J$")
    axes[0].semilogy(cutoffs, k_cutoff, "s--", color=ORANGE, label=r"$K$")
    axes[0].axvline(0.075, color=GRAY, ls=":", lw=1.0)
    axes[0].set(
        xlabel=r"$\alpha_{\min}^{\mathrm{aux}}$ ($a_0^{-2}$)",
        ylabel="Max difference (meV)",
        title="a  Matched pruning",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(cutoffs, energy_drift, "D-", color=GREEN, ms=4.5)
    axes[1].axhline(0.0, color=GRAY, ls=":", lw=1.0)
    axes[1].axvline(0.075, color=GRAY, ls=":", lw=1.0)
    axes[1].set(
        xlabel=r"$\alpha_{\min}^{\mathrm{aux}}$ ($a_0^{-2}$)",
        ylabel=r"PySCF $\Delta E_{\mathrm{SCF}}$ (meV)",
        title="b  Fitting-space drift",
    )

    axes[2].loglog(precision, j_precision, "o-", color=BLUE, label=r"$J$")
    axes[2].loglog(precision, k_precision, "s--", color=ORANGE, label=r"$K$")
    axes[2].axhline(1.0e-3, color=GRAY, ls=":", lw=1.0)
    axes[2].invert_xaxis()
    axes[2].set(
        xlabel="GDF precision",
        ylabel="Max difference (meV)",
        title=r"c  Precision at $\alpha_{\min}=0.08$",
    )
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2, lw=0.6, which="both")
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
