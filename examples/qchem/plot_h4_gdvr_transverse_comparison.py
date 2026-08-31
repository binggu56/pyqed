#!/usr/bin/env python3
"""Compare two transverse-basis GDVR PES scans on the same coordinate grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, default=Path("h4_gdvr_transverse_comparison.png"))
    args = parser.parse_args()

    ref = np.load(args.reference)
    cmp = np.load(args.comparison)
    q_plus = ref["q_plus"]
    q_minus = ref["q_minus"]
    if not (
        np.array_equal(q_plus, cmp["q_plus"])
        and np.array_equal(q_minus, cmp["q_minus"])
    ):
        raise ValueError("The PES coordinate grids differ")

    e_ref = ref["gdvr_rhf_energy"]
    e_cmp = cmp["gdvr_rhf_energy"]
    rel_ref = 1000.0 * (e_ref - e_ref.min())
    rel_cmp = 1000.0 * (e_cmp - e_cmp.min())
    shape_delta = rel_cmp - rel_ref
    x, y = np.meshgrid(q_minus, q_plus)
    common_levels = np.linspace(0.0, max(rel_ref.max(), rel_cmp.max()), 25)
    delta_bound = max(abs(shape_delta.min()), abs(shape_delta.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for ax, values, title in (
        (axes[0], rel_ref, "d-aug-cc-pVDZ transverse"),
        (axes[1], rel_cmp, "6-31G transverse"),
    ):
        image = ax.contourf(x, y, values, levels=common_levels, cmap="viridis")
        ax.scatter(x, y, s=9, color="black", alpha=0.45)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Relative energy (mEh)")

    image = axes[2].contourf(
        x,
        y,
        shape_delta,
        levels=np.linspace(-delta_bound, delta_bound, 25),
        cmap="coolwarm",
    )
    axes[2].scatter(x, y, s=9, color="black", alpha=0.45)
    axes[2].set_title("6-31G − d-aug PES shape")
    fig.colorbar(image, ax=axes[2], label="Relative-energy difference (mEh)")
    for ax in axes:
        ax.set_xlabel(r"$q_-$ (bohr)")
        ax.set_ylabel(r"$q_+$ (bohr)")
    fig.suptitle(r"Newton-optimized GDVR-RHF PES, $N_z=41$, $L_z=6$ bohr")
    fig.savefig(args.output, dpi=240)
    plt.close(fig)

    absolute_shift = 1000.0 * (e_cmp - e_ref)
    print(f"Saved {args.output}")
    print(
        "6-31G - d-aug absolute shift (mEh): "
        f"min={absolute_shift.min():+.6f}, max={absolute_shift.max():+.6f}, "
        f"mean={absolute_shift.mean():+.6f}"
    )
    print(
        "Relative-PES shape difference (mEh): "
        f"RMSE={np.sqrt(np.mean(shape_delta**2)):.6f}, "
        f"maxabs={np.max(np.abs(shape_delta)):.6f}"
    )


if __name__ == "__main__":
    main()
