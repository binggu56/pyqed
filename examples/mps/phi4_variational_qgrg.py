"""Diagnose variational-Gaussian QGRG with interacting-shell Feshbach terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.geometric_rg import (
    Phi4FunctionalQGRG,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4VariationalQGRG,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_variational_qgrg"),
    )
    parser.add_argument("--log-width", type=float, default=0.8)
    parser.add_argument("--nfield", type=int, default=81)
    parser.add_argument("--quadrature-order", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    field = np.linspace(-0.7, 0.7, args.nfield)
    center = np.argmin(np.abs(field))
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    potential = shell.potential(field, couplings)
    gaussian = Phi4FunctionalQGRG(field, spatial_dimension=1)
    gaussian_rates = gaussian.rates(potential)
    variational = Phi4VariationalQGRG(
        field,
        log_width=args.log_width,
        quadrature_order=args.quadrature_order,
    )
    variational_rates = variational.rates(potential)
    frame = variational.frame
    feshbach = variational.feshbach
    potential_rate_without_feshbach = (
        2.0 * potential + frame["energy"] / args.log_width
    )

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.5), constrained_layout=True
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0, 0]
    axis.plot(field, frame["hartree"], color=blue, label=r"Hartree gap $\Delta_H$")
    axis.plot(
        field,
        frame["variance"],
        color=orange,
        linestyle="--",
        label=r"shell variance $\langle\varphi_>^2\rangle_G$",
    )
    axis.set(xlabel=r"background $\phi$", ylabel="optimized Gaussian frame")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(
        field,
        gaussian_rates[0] - gaussian_rates[0][center],
        color="0.45",
        linestyle=":",
        linewidth=1.7,
        label="infinitesimal Gaussian",
    )
    axis.plot(
        field,
        potential_rate_without_feshbach - potential_rate_without_feshbach[center],
        color=green,
        linestyle="--",
        label="variational Gaussian",
    )
    axis.plot(
        field,
        variational_rates[0] - variational_rates[0][center],
        color=blue,
        label="variational + Feshbach",
    )
    axis.set(xlabel=r"background $\phi$", ylabel=r"$\partial_\ell U(\phi)-\partial_\ell U(0)$")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(
        field,
        gaussian_rates[1],
        color=blue,
        linestyle=":",
        label=r"Gaussian $\partial_\ell Z_t$",
    )
    axis.plot(
        field,
        variational_rates[1],
        color=blue,
        label=r"variational $\partial_\ell Z_t$",
    )
    axis.plot(
        field,
        gaussian_rates[2],
        color=orange,
        linestyle=":",
        label=r"Gaussian $\partial_\ell Z_x$",
    )
    axis.plot(
        field,
        variational_rates[2],
        color=orange,
        label=r"variational $\partial_\ell Z_x$",
    )
    axis.axhline(0.0, color="0.4", linewidth=0.7)
    axis.set(xlabel=r"background $\phi$", ylabel="quantum-geometric response")
    axis.legend(frameon=False, ncols=2)

    axis = axes[1, 1]
    axis.plot(
        field,
        feshbach["three_boson"] / args.log_width,
        color=purple,
        label=r"three-boson $\Delta U_3/d\ell$",
    )
    axis.plot(
        field,
        feshbach["four_boson"] / args.log_width,
        color=green,
        label=r"four-boson $\Delta U_4/d\ell$",
    )
    axis.axhline(0.0, color="0.4", linewidth=0.7)
    axis.set(xlabel=r"background $\phi$", ylabel="Feshbach energy rate")
    axis.legend(frameon=False)

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(
            0.02,
            0.97,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
        axis.grid(color="0.9", linewidth=0.5)

    png = args.output_dir / "phi4_variational_qgrg.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    np.savez(
        args.output_dir / "diagnostics.npz",
        field=field,
        potential=potential,
        gaussian_rates=np.asarray(gaussian_rates),
        variational_rates=np.asarray(variational_rates),
        hartree=frame["hartree"],
        variance=frame["variance"],
        metric=frame["metric"],
        feshbach_three=feshbach["three_boson"],
        feshbach_four=feshbach["four_boson"],
    )
    summary = {
        "model": "1+1D finite-shell variational Gaussian QGRG",
        "log_width": args.log_width,
        "field_points": args.nfield,
        "momentum_points": int(2 * args.quadrature_order),
        "maximum_hartree_gap": float(np.max(frame["hartree"])),
        "maximum_shell_variance": float(np.max(frame["variance"])),
        "three_boson_energy_range": [
            float(np.min(feshbach["three_boson"])),
            float(np.max(feshbach["three_boson"])),
        ],
        "four_boson_energy_range": [
            float(np.min(feshbach["four_boson"])),
            float(np.max(feshbach["four_boson"])),
        ],
        "maximum_metric": float(np.max(frame["metric"])),
        "temporal_rate_range": [
            float(np.min(variational_rates[1])),
            float(np.max(variational_rates[1])),
        ],
        "spatial_rate_range": [
            float(np.min(variational_rates[2])),
            float(np.max(variational_rates[2])),
        ],
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
