"""Benchmark regulator-based Hamiltonian QGRF for the 1+1D phi4 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt
from scipy.optimize import root

from pyqed.narg.geometric_rg import (
    Phi4CovariantFRG,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4RegulatedQGRF,
)


def _jacobian(function, point, step=2.0e-4):
    point = np.asarray(point, dtype=float)
    columns = []
    for axis in range(point.size):
        displacement = np.zeros_like(point)
        displacement[axis] = step
        columns.append(
            (function(point + displacement) - function(point - displacement))
            / (2.0 * step)
        )
    return np.column_stack(columns)


def _fixed_point(flow, guess):
    def beta(values):
        couplings = Phi4GaussianCouplings(
            mass2=values[0], quartic=values[1]
        )
        result = flow.beta(couplings)
        coupling_rate = result[0]
        return np.array([coupling_rate.mass2, coupling_rate.quartic])

    solution = root(beta, np.asarray(guess, dtype=float), tol=2.0e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    couplings = Phi4GaussianCouplings(
        mass2=solution.x[0], quartic=solution.x[1]
    )
    result = flow.beta(couplings)
    eigenvalues = np.linalg.eigvals(_jacobian(beta, solution.x)).real
    relevant = float(np.max(eigenvalues))
    return couplings, result[1:], eigenvalues, 1.0 / relevant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_regulated_qgrf"),
    )
    parser.add_argument("--quadrature-order", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sharp = Phi4GaussianShell(spatial_dimension=1)
    covariant = Phi4CovariantFRG(radial_order=100, angular_order=64)
    regulated_gaussian = Phi4RegulatedQGRF(
        quadrature_order=args.quadrature_order,
        include_feshbach=False,
    )
    regulated_feshbach = Phi4RegulatedQGRF(
        quadrature_order=args.quadrature_order,
        include_feshbach=True,
    )

    sharp_fixed = Phi4GaussianCouplings(
        mass2=-0.4,
        quartic=16.0 * np.pi * 0.6**1.5 / 3.0,
    )

    def sharp_beta(values):
        rate = sharp.beta(
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1])
        )
        return np.array([rate.mass2, rate.quartic])

    sharp_eigenvalues = np.linalg.eigvals(
        _jacobian(sharp_beta, [sharp_fixed.mass2, sharp_fixed.quartic])
    ).real
    sharp_nu = 1.0 / np.max(sharp_eigenvalues)
    covariant_fixed, covariant_data, covariant_eigenvalues, covariant_nu = (
        _fixed_point(covariant, [-0.19, 3.6])
    )
    gaussian_fixed, gaussian_data, gaussian_eigenvalues, gaussian_nu = (
        _fixed_point(regulated_gaussian, [-0.22, 3.97])
    )
    feshbach_fixed, feshbach_data, feshbach_eigenvalues, feshbach_nu = (
        _fixed_point(regulated_feshbach, [-0.61, 4.72])
    )
    covariant_eta = float(covariant_data[0])
    gaussian_eta_t, gaussian_eta_x, gaussian_z = map(float, gaussian_data)
    feshbach_eta_t, feshbach_eta_x, feshbach_z = map(float, feshbach_data)

    benchmark = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    field = np.linspace(-0.65, 0.65, 81)
    center = field.size // 2
    gaussian_potential = regulated_gaussian.potential_rate(field, benchmark)
    feshbach_potential = regulated_feshbach.potential_rate(field, benchmark)
    kinetic_field = np.linspace(-1.05, 1.05, 101)
    feshbach_kinetic = regulated_feshbach.rates(
        kinetic_field, feshbach_fixed
    )
    feshbach_minimum = np.sqrt(
        -6.0 * feshbach_fixed.mass2 / feshbach_fixed.quartic
    )

    labels = ["sharp", "covariant", "reg. G", "reg. G+F", "exact"]
    nu_values = [sharp_nu, covariant_nu, gaussian_nu, feshbach_nu, 1.0]
    eta_x_values = [0.0, covariant_eta, gaussian_eta_x, feshbach_eta_x, 0.25]
    eta_t_values = [0.0, covariant_eta, gaussian_eta_t, feshbach_eta_t, 0.25]
    positions = np.arange(len(labels))

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
        refheight=2.25,
        share=False,
        wspace=4.0,
        hspace=6.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0]
    axis.plot(
        field,
        gaussian_potential - gaussian_potential[center],
        color=blue,
        linestyle="--",
        label="regulated Gaussian",
    )
    axis.plot(
        field,
        feshbach_potential - feshbach_potential[center],
        color=orange,
        label="regulated Gaussian + Feshbach",
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
    axis.plot(
        kinetic_field,
        feshbach_kinetic[1],
        color=orange,
        label=r"$\beta_{Z_t}$",
    )
    axis.plot(
        kinetic_field,
        feshbach_kinetic[2],
        color=green,
        label=r"$\beta_{Z_x}$",
    )
    axis.axvline(-feshbach_minimum, color="0.55", linewidth=0.7, linestyle=":")
    axis.axvline(feshbach_minimum, color="0.55", linewidth=0.7, linestyle=":")
    axis.axhline(0.0, color="0.45", linewidth=0.7)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="kinetic flow",
        title="b  Feshbach fixed point",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center", ncols=2)

    axis = axes[2]
    colors = ["0.55", purple, blue, orange, green]
    axis.bar(positions, nu_values, color=colors, width=0.68)
    axis.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    axis.format(
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, 1.08),
        ylabel=r"correlation exponent $\nu$",
        title="c  Relevant exponent",
        grid=False,
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", color="0.9", linewidth=0.5)

    axis = axes[3]
    width = 0.32
    axis.bar(
        positions - width / 2,
        eta_x_values,
        width=width,
        color=blue,
        label=r"$\eta_x$",
    )
    axis.bar(
        positions + width / 2,
        eta_t_values,
        width=width,
        color=orange,
        label=r"$\eta_t$",
    )
    axis.axhline(0.25, color="0.35", linewidth=0.8, linestyle="--")
    axis.format(
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, 0.28),
        ylabel="anomalous dimension",
        title="d  Kinetic exponent",
        grid=False,
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left", ncols=2)

    png = args.output_dir / "phi4_regulated_qgrf_comparison.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    summary = {
        "model": "canonical 1+1D phi4",
        "exact_ising": {"nu": 1.0, "eta": 0.25},
        "sharp_shell": {
            "mass2": sharp_fixed.mass2,
            "quartic": sharp_fixed.quartic,
            "nu": sharp_nu,
            "eta": 0.0,
            "eigenvalues": sharp_eigenvalues.tolist(),
        },
        "covariant_frg": {
            "mass2": covariant_fixed.mass2,
            "quartic": covariant_fixed.quartic,
            "nu": covariant_nu,
            "eta": covariant_eta,
            "eigenvalues": covariant_eigenvalues.tolist(),
        },
        "regulated_gaussian_qgrf": {
            "mass2": gaussian_fixed.mass2,
            "quartic": gaussian_fixed.quartic,
            "nu": gaussian_nu,
            "eta_t": gaussian_eta_t,
            "eta_x": gaussian_eta_x,
            "z": gaussian_z,
            "eigenvalues": gaussian_eigenvalues.tolist(),
        },
        "regulated_feshbach_qgrf": {
            "mass2": feshbach_fixed.mass2,
            "quartic": feshbach_fixed.quartic,
            "nu": feshbach_nu,
            "eta_t": feshbach_eta_t,
            "eta_x": feshbach_eta_x,
            "z": feshbach_z,
            "eigenvalues": feshbach_eigenvalues.tolist(),
            "regulated_origin_gap2": 1.0 + feshbach_fixed.mass2,
        },
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez(
        args.output_dir / "comparison.npz",
        field=field,
        kinetic_field=kinetic_field,
        gaussian_potential=gaussian_potential,
        feshbach_potential=feshbach_potential,
        feshbach_beta_zt=feshbach_kinetic[1],
        feshbach_beta_zx=feshbach_kinetic[2],
        nu_values=nu_values,
        eta_t_values=eta_t_values,
        eta_x_values=eta_x_values,
    )
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
