#!/usr/bin/env python3
"""Compute and plot a CPHF-relaxed Gamma-point KRHF Hessian."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem.pbc import Cell


def build_reference(recip_cut):
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.eye(3) * 6.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    return cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=recip_cut,
        one_body_nuclear_cut=1,
        jk_builder="reciprocal",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).run(max_cycle=60, conv_tol=1.0e-11, conv_tol_dm=1.0e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=2.0e-4)
    parser.add_argument("--recip-cut", type=int, default=2)
    parser.add_argument(
        "--second-derivative-backend",
        choices=("auto", "analytic", "finite_difference"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gamma_hessian.pdf"),
    )
    args = parser.parse_args()

    mean_field = build_reference(args.recip_cut)
    hessian = mean_field.Hessian()
    matrix = hessian.kernel(
        step=args.step,
        second_derivative_backend=args.second_derivative_backend,
    )
    frequencies = hessian.frequencies()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    limit = float(np.max(np.abs(matrix)))
    image = axes[0].imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    axes[0].set_title(r"Relaxed $\Gamma$ Hessian")
    axes[0].set_xlabel("Nuclear coordinate")
    axes[0].set_ylabel("Nuclear coordinate")
    axes[0].set_xticks(range(len(matrix)))
    axes[0].set_yticks(range(len(matrix)))
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label=r"$E_h/a_0^2$")

    mode_index = np.arange(len(frequencies))
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].vlines(mode_index, 0.0, frequencies, color="#28666E", linewidth=1.4)
    axes[1].scatter(mode_index, frequencies, color="#B24C63", s=25, zorder=3)
    axes[1].set_title(r"Mass-weighted $\Gamma$ modes")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel(r"Frequency (cm$^{-1}$)")
    axes[1].set_xticks(mode_index)
    axes[1].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=300)
    data = args.output.with_suffix(".npz")
    np.savez(
        data,
        hessian=matrix,
        raw_hessian=hessian.raw_hess,
        frequencies_cm1=frequencies,
        first_order_density=hessian.first_order_density,
        cphf_residual=hessian.response.residual_norm,
        acoustic_sum_rule_residual=hessian.acoustic_sum_rule_residual,
        second_derivative_backend=hessian.second_derivative_backend,
        step=args.step,
    )
    print(f"second derivatives: {hessian.second_derivative_backend}")
    print(f"CPHF residual: {hessian.response.residual_norm:.3e}")
    print(f"acoustic sum-rule residual: {hessian.acoustic_sum_rule_residual:.3e}")
    print("Gamma frequencies (cm^-1):", np.array2string(frequencies, precision=3))
    print(f"Hessian build: {hessian.seconds:.3f} s")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
