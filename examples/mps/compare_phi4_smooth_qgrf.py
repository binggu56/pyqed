"""Compare smooth Hamiltonian QGRF with the standard 1+1D phi4 shell flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4SmoothQGRF,
)


def _local_potential_couplings(flow, couplings, radius=0.05):
    field = np.linspace(-radius, radius, 9)
    potential_rate = flow.rates(field, couplings)[0]
    scaled_field = field / radius
    coefficients = np.polynomial.polynomial.polyfit(
        scaled_field, potential_rate, 6
    )
    beta_mass2 = 2.0 * coefficients[2] / radius**2
    beta_quartic = 24.0 * coefficients[4] / radius**4
    return float(beta_mass2), float(beta_quartic)


def _normalized_qgrf_beta(flow, mass2, quartic):
    couplings = Phi4GaussianCouplings(mass2=mass2, quartic=quartic)
    beta_mass2, beta_quartic = _local_potential_couplings(flow, couplings)
    minimum = np.sqrt(max(0.0, -6.0 * mass2 / quartic))
    _, beta_zt, beta_zx = flow.rates(np.array([minimum]), couplings)
    eta_t = float(beta_zt[0])
    eta_x = float(beta_zx[0])
    dynamic_exponent = 1.0 + 0.5 * (eta_t - eta_x)
    beta_mass2 -= eta_x * mass2
    beta_quartic -= (dynamic_exponent - 1.0 + 2.0 * eta_x) * quartic
    return beta_mass2, beta_quartic, eta_t, eta_x, dynamic_exponent


def _bisect(function, lower, upper, iterations=24):
    lower_value = function(lower)
    upper_value = function(upper)
    if lower_value * upper_value > 0.0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = function(midpoint)
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
            upper_value = midpoint_value
        else:
            lower = midpoint
            lower_value = midpoint_value
    return 0.5 * (lower + upper)


def _mass_null_curves(standard, qgrf, quartics):
    standard_mass = np.empty_like(quartics)
    standard_quartic = np.empty_like(quartics)
    qgrf_mass = np.empty_like(quartics)
    qgrf_quartic = np.empty_like(quartics)
    qgrf_eta_t = np.empty_like(quartics)
    qgrf_eta_x = np.empty_like(quartics)
    for index, quartic in enumerate(quartics):
        def standard_mass_rate(mass2):
            return standard.beta(
                Phi4GaussianCouplings(mass2=mass2, quartic=quartic)
            ).mass2

        standard_mass[index] = _bisect(
            standard_mass_rate, -2.0 / 3.0, 0.1
        )
        standard_quartic[index] = standard.beta(
            Phi4GaussianCouplings(
                mass2=standard_mass[index], quartic=quartic
            )
        ).quartic

        def qgrf_mass_rate(mass2):
            return _normalized_qgrf_beta(qgrf, mass2, quartic)[0]

        qgrf_mass[index] = _bisect(qgrf_mass_rate, -0.5, 0.1)
        values = _normalized_qgrf_beta(
            qgrf, qgrf_mass[index], quartic
        )
        qgrf_quartic[index] = values[1]
        qgrf_eta_t[index] = values[2]
        qgrf_eta_x[index] = values[3]
    return {
        "standard_mass2": standard_mass,
        "standard_beta_quartic": standard_quartic,
        "qgrf_mass2": qgrf_mass,
        "qgrf_beta_quartic": qgrf_quartic,
        "qgrf_eta_t": qgrf_eta_t,
        "qgrf_eta_x": qgrf_eta_x,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_smooth_qgrf"),
    )
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--scan-quadrature-order", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    field = np.linspace(-0.65, 0.65, 101)
    center = field.size // 2
    standard = Phi4GaussianShell(spatial_dimension=1)
    smooth = Phi4SmoothQGRF(quadrature_order=args.quadrature_order)
    smooth_potential, smooth_zt, smooth_zx = smooth.rates(field, couplings)
    standard_potential = standard.beta_potential(field, couplings)
    components = smooth.components

    quartics = np.linspace(2.0, 9.5, 16)
    scan_flow = Phi4SmoothQGRF(
        quadrature_order=args.scan_quadrature_order
    )
    mass_null = _mass_null_curves(standard, scan_flow, quartics)
    standard_fixed_mass2 = -0.4
    standard_fixed_quartic = (
        16.0 * np.pi * (1.0 + standard_fixed_mass2) ** 1.5 / 3.0
    )
    standard_jacobian = np.array(
        [
            [4.0 / 3.0, 1.0 / (4.0 * np.pi * np.sqrt(0.6))],
            [
                9.0 * standard_fixed_quartic**2
                / (16.0 * np.pi * 0.6**2.5),
                -2.0,
            ],
        ]
    )
    relevant_eigenvalue = float(np.max(np.linalg.eigvals(standard_jacobian).real))
    standard_nu = 1.0 / relevant_eigenvalue

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
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=2,
        refwidth=3.0,
        refheight=2.3,
        share=False,
        wspace=4.2,
        hspace=6.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0]
    axis.plot(
        field,
        standard_potential - standard_potential[center],
        color="0.4",
        linestyle="--",
        label="standard sharp shell",
    )
    axis.plot(
        field,
        smooth_potential - smooth_potential[center],
        color=blue,
        label="smooth variational QGRF",
    )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$\beta_U(\phi)-\beta_U(0)$",
        title="a  Potential flow",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center")

    axis = axes[1]
    axis.plot(field, smooth_zt, color=orange, label=r"$\beta_{Z_t}$")
    axis.plot(field, smooth_zx, color=green, label=r"$\beta_{Z_x}$")
    axis.axhline(0.0, color="0.45", linewidth=0.7)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="kinetic flow",
        title="b  Dressed kinetic kernel",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower center", ncols=2)

    axis = axes[2]
    axis.plot(
        field,
        standard.curvature(field, couplings),
        color="0.45",
        linestyle="--",
        label=r"bare $U''$",
    )
    axis.plot(
        field,
        components["hartree_mass2"],
        color=purple,
        label=r"Hartree $\mu^2$",
    )
    axis.axhline(0.0, color="0.45", linewidth=0.7)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="mass squared",
        title="c  Variational frame",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower right")

    axis = axes[3]
    axis.plot(
        quartics,
        mass_null["standard_beta_quartic"],
        color="0.4",
        linestyle="--",
        marker="o",
        markersize=3.2,
        label=r"standard, $\beta_r=0$",
    )
    axis.plot(
        quartics,
        mass_null["qgrf_beta_quartic"],
        color=blue,
        marker="s",
        markersize=3.2,
        label=r"QGRF, $\beta_r=0$",
    )
    axis.scatter(
        [standard_fixed_quartic],
        [0.0],
        color=orange,
        marker="*",
        s=40,
        zorder=5,
        label="standard fixed point",
    )
    axis.axhline(0.0, color="0.45", linewidth=0.7)
    axis.format(
        xlabel=r"quartic coupling $\lambda$",
        ylabel=r"$\beta_\lambda$ on $\beta_r=0$",
        title="d  Fixed-point diagnostic",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left")

    png = args.output_dir / "phi4_smooth_qgrf_comparison.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    np.savez(
        args.output_dir / "comparison.npz",
        field=field,
        standard_potential=standard_potential,
        smooth_potential=smooth_potential,
        smooth_zt=smooth_zt,
        smooth_zx=smooth_zx,
        quartics=quartics,
        **mass_null,
        **components,
    )
    summary = {
        "model": "canonical 1+1D phi4",
        "benchmark_couplings": {
            "mass2": couplings.mass2,
            "quartic": couplings.quartic,
        },
        "smooth_qgrf_at_origin": {
            "beta_Zt": float(smooth_zt[center]),
            "beta_Zx": float(smooth_zx[center]),
        },
        "standard_quartic_fixed_point": {
            "mass2": standard_fixed_mass2,
            "quartic": standard_fixed_quartic,
            "nu": standard_nu,
            "eta": 0.0,
        },
        "smooth_qgrf_quartic_closure": {
            "fixed_point_found": False,
            "scanned_quartic_interval": [float(quartics[0]), float(quartics[-1])],
            "minimum_beta_quartic_on_mass_null": float(
                np.min(mass_null["qgrf_beta_quartic"])
            ),
            "exponents_extracted": False,
        },
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
