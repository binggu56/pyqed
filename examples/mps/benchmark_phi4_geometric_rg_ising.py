"""Benchmark Gaussian geometric ``phi4`` RG against the 2D Ising CFT."""

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
)


def _jacobian(function, point, step=1.0e-6):
    point = np.asarray(point, dtype=float)
    jacobian = np.empty((point.size, point.size), dtype=float)
    for column in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jacobian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_geometric_rg_ising_benchmark"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shell = Phi4GaussianShell(spatial_dimension=1)
    prefactor = shell.fluctuation_prefactor

    lpa_mass2 = -2.0 / 5.0
    lpa_quartic = -4.0 * lpa_mass2 * np.sqrt(1.0 + lpa_mass2) / prefactor
    lpa_fixed = Phi4GaussianCouplings(
        mass2=lpa_mass2, quartic=lpa_quartic
    )

    def lpa_beta(values):
        beta = shell.beta(
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1])
        )
        return np.array([beta.mass2, beta.quartic])

    lpa_jacobian = _jacobian(lpa_beta, [lpa_mass2, lpa_quartic])
    lpa_eigenvalues = np.sort(np.linalg.eigvals(lpa_jacobian).real)
    relevant_eigenvalue = lpa_eigenvalues[-1]
    lpa_nu = 1.0 / relevant_eigenvalue

    def lpa_prime_beta(values):
        beta, _ = shell.beta_lpa_prime(
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1])
        )
        return np.array([beta.mass2, beta.quartic])

    lpa_prime_solution = root(
        lpa_prime_beta, np.array([lpa_mass2, lpa_quartic])
    )
    if not lpa_prime_solution.success:
        raise RuntimeError(lpa_prime_solution.message)
    lpa_prime_mass2, lpa_prime_quartic = lpa_prime_solution.x
    lpa_prime_fixed = Phi4GaussianCouplings(
        mass2=lpa_prime_mass2, quartic=lpa_prime_quartic
    )
    _, lpa_prime_eta = shell.beta_lpa_prime(lpa_prime_fixed)
    lpa_prime_jacobian = _jacobian(
        lpa_prime_beta, [lpa_prime_mass2, lpa_prime_quartic]
    )
    lpa_prime_eigenvalues = np.sort(
        np.linalg.eigvals(lpa_prime_jacobian).real
    )
    lpa_prime_nu = 1.0 / lpa_prime_eigenvalues[-1]

    def spatial_beta(values):
        beta, _ = shell.beta_lpa_prime(
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1]),
            projection="spatial",
        )
        return np.array([beta.mass2, beta.quartic])

    spatial_solution = root(spatial_beta, np.array([-0.41, 8.4]))
    if not spatial_solution.success:
        raise RuntimeError(spatial_solution.message)
    spatial_mass2, spatial_quartic = spatial_solution.x
    spatial_fixed = Phi4GaussianCouplings(
        mass2=spatial_mass2, quartic=spatial_quartic
    )
    _, spatial_eta = shell.beta_lpa_prime(
        spatial_fixed, projection="spatial"
    )
    spatial_jacobian = _jacobian(
        spatial_beta, [spatial_mass2, spatial_quartic]
    )
    spatial_eigenvalues = np.sort(np.linalg.eigvals(spatial_jacobian).real)
    spatial_nu = 1.0 / spatial_eigenvalues[-1]

    covariant = Phi4CovariantFRG()

    def covariant_beta(values):
        beta, _ = covariant.beta(
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1])
        )
        return np.array([beta.mass2, beta.quartic])

    covariant_solution = root(covariant_beta, np.array([-0.2, 3.6]))
    if not covariant_solution.success:
        raise RuntimeError(covariant_solution.message)
    covariant_mass2, covariant_quartic = covariant_solution.x
    covariant_fixed = Phi4GaussianCouplings(
        mass2=covariant_mass2, quartic=covariant_quartic
    )
    _, covariant_eta = covariant.beta(covariant_fixed)
    covariant_jacobian = _jacobian(
        covariant_beta, [covariant_mass2, covariant_quartic]
    )
    covariant_eigenvalues = np.sort(
        np.linalg.eigvals(covariant_jacobian).real
    )
    covariant_nu = 1.0 / covariant_eigenvalues[-1]

    minimum = np.sqrt(
        -6.0 * lpa_prime_mass2 / lpa_prime_quartic
    )
    exact_kinetic_rates = shell.external_kinetic_rates(
        minimum, lpa_prime_fixed
    )
    momentum_steps = np.geomspace(1.0e-3, 0.2, 18)
    finite_kinetic_rates = np.array(
        [
            shell.external_kinetic_rates(
                minimum, lpa_prime_fixed, momentum_step=step
            )
            for step in momentum_steps
        ]
    )
    covariant_minimum = np.sqrt(
        -6.0 * covariant_mass2 / covariant_quartic
    )
    covariant_rates0 = np.array(
        [
            covariant.kinetic_rate(
                covariant_minimum,
                covariant_fixed,
                eta=covariant_eta,
                axis=0,
                momentum_steps=[step],
            )
            for step in momentum_steps
        ]
    )
    covariant_rates1 = np.array(
        [
            covariant.kinetic_rate(
                covariant_minimum,
                covariant_fixed,
                eta=covariant_eta,
                axis=1,
                momentum_steps=[step],
            )
            for step in momentum_steps
        ]
    )

    # In D=2, beta_z2=0 requires z2=-lambda/(1+r).  The only
    # non-Gaussian stationary solution then lies at negative lambda.
    geometric_mass2 = 0.5
    geometric_w = 1.0 + geometric_mass2
    geometric_quartic = -2.0 * geometric_w**1.5 / (3.0 * prefactor)
    geometric_z2 = -geometric_quartic / geometric_w
    geometric_beta, geometric_beta_z2 = shell.beta_z2(
        Phi4GaussianCouplings(
            mass2=geometric_mass2, quartic=geometric_quartic
        ),
        geometric_z2,
    )
    geometric_residual = np.array(
        [geometric_beta.mass2, geometric_beta.quartic, geometric_beta_z2]
    )

    exact_nu = 1.0
    exact_eta = 0.25
    epsilon = 2.0
    one_loop_nu = 0.5 + epsilon / 12.0
    one_loop_eta = 0.0

    summary = {
        "dimension": "D=2 (d=1 Hamiltonian spatial dimension)",
        "exact_ising": {"nu": exact_nu, "eta": exact_eta},
        "strict_one_loop_epsilon_expansion": {
            "epsilon": epsilon,
            "nu": one_loop_nu,
            "eta": one_loop_eta,
        },
        "gaussian_local_potential": {
            "fixed_mass2": lpa_mass2,
            "fixed_quartic": lpa_quartic,
            "stability_eigenvalues": lpa_eigenvalues.tolist(),
            "nu": lpa_nu,
            "eta": 0.0,
            "relative_nu_error": abs(lpa_nu / exact_nu - 1.0),
            "absolute_eta_error": exact_eta,
        },
        "geometric_lpa_prime": {
            "kinetic_projection": "Z_t=Z_x normalized at the running minimum",
            "fixed_mass2": float(lpa_prime_mass2),
            "fixed_quartic": float(lpa_prime_quartic),
            "stability_eigenvalues": lpa_prime_eigenvalues.tolist(),
            "nu": float(lpa_prime_nu),
            "eta": float(lpa_prime_eta),
            "relative_nu_error": abs(lpa_prime_nu / exact_nu - 1.0),
            "absolute_eta_error": abs(lpa_prime_eta - exact_eta),
        },
        "sharp_shell_external_momentum": {
            "fixed_mass2": float(spatial_mass2),
            "fixed_quartic": float(spatial_quartic),
            "stability_eigenvalues": spatial_eigenvalues.tolist(),
            "nu": float(spatial_nu),
            "eta_x": float(spatial_eta),
            "eta_t_at_matched_fixed_point": float(exact_kinetic_rates[0]),
            "eta_x_at_matched_fixed_point": float(exact_kinetic_rates[1]),
            "interpretation": (
                "negative eta_x diagnoses anisotropy of the sharp spatial shell"
            ),
        },
        "covariant_frg": {
            "regulator": "exponential full-Euclidean-momentum regulator",
            "fixed_mass2": float(covariant_mass2),
            "fixed_quartic": float(covariant_quartic),
            "stability_eigenvalues": covariant_eigenvalues.tolist(),
            "nu": float(covariant_nu),
            "eta": float(covariant_eta),
            "axis_projection_difference": float(
                np.max(np.abs(covariant_rates0 - covariant_rates1))
            ),
            "relative_nu_error": abs(covariant_nu / exact_nu - 1.0),
            "absolute_eta_error": abs(covariant_eta - exact_eta),
        },
        "geometric_z2_closure": {
            "physical_interacting_fixed_point": False,
            "reason": (
                "beta_z2=0 and lambda>0 imply beta_lambda>0 in D=2"
            ),
            "unphysical_stationary_point": {
                "mass2": geometric_mass2,
                "quartic": geometric_quartic,
                "z2": geometric_z2,
                "beta_residual_norm": float(np.linalg.norm(geometric_residual)),
            },
            "nu": None,
            "eta": None,
        },
    }

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = uplt.subplots(
        ncols=3,
        refwidth=2.75,
        refheight=2.75,
        share=False,
        wspace=4.8,
    )

    axis = axes[0]
    methods = [
        "Exact Ising",
        r"1-loop $\epsilon$",
        "Gaussian LPA",
        r"GQD LPA$'$",
        "Covariant FRG",
    ]
    positions = np.arange(len(methods))
    width = 0.32
    nu_values = [exact_nu, one_loop_nu, lpa_nu, lpa_prime_nu, covariant_nu]
    eta_values = [
        exact_eta,
        one_loop_eta,
        0.0,
        lpa_prime_eta,
        covariant_eta,
    ]
    axis.bar(
        positions - width / 2.0,
        nu_values,
        width,
        color="#0072B2",
        label=r"$\nu$",
    )
    axis.bar(
        positions + width / 2.0,
        eta_values,
        width,
        color="#E69F00",
        label=r"$\eta$",
    )
    axis.format(
        ylabel="critical exponent",
        xticks=positions,
        xticklabels=methods,
        xrotation=18,
        xlim=(-0.6, 4.6),
        ylim=(0.0, 1.08),
        grid=False,
        title=r"$\mathbf{(a)}$ Exponent benchmark",
    )
    axis.grid(axis="y", color="0.88", linewidth=0.5)
    axis.legend(frame=False, loc="upper right", ncols=1)

    axis = axes[1]
    axis.scatter(
        [lpa_mass2],
        [lpa_quartic],
        s=58,
        marker="o",
        color="#0072B2",
        label="Gaussian LPA",
        zorder=3,
    )
    axis.scatter(
        [lpa_prime_mass2],
        [lpa_prime_quartic],
        s=70,
        marker="x",
        linewidth=2.0,
        color="#009E73",
        label=r"GQD LPA$'$",
        zorder=3,
    )
    axis.scatter(
        [spatial_mass2],
        [spatial_quartic],
        s=62,
        marker="^",
        facecolor="white",
        edgecolor="#D55E00",
        linewidth=1.4,
        label=r"external $q$ (sharp)",
        zorder=3,
    )
    axis.scatter(
        [covariant_mass2],
        [covariant_quartic],
        s=68,
        marker="D",
        color="#CC79A7",
        label="covariant FRG",
        zorder=3,
    )
    axis.format(
        xlabel=r"$r_*$",
        ylabel=r"$\lambda_*$",
        xlim=(-0.44, -0.17),
        ylim=(3.2, 8.8),
        grid=False,
        title=r"$\mathbf{(b)}$ Interacting fixed point",
    )
    axis.grid(color="0.90", linewidth=0.45)
    axis.legend(frame=False, loc="upper right")

    axis = axes[2]
    axis.semilogx(
        momentum_steps,
        finite_kinetic_rates[:, 0],
        color="#0072B2",
        marker="o",
        markersize=3.0,
        label=r"frequency: $\eta_t(q)$",
    )
    axis.semilogx(
        momentum_steps,
        finite_kinetic_rates[:, 1],
        color="#D55E00",
        marker="s",
        markersize=3.0,
        label=r"momentum: $\eta_x(q)$",
    )
    axis.semilogx(
        momentum_steps,
        covariant_rates0,
        color="#009E73",
        marker="^",
        markersize=3.0,
        label="covariant axis 1",
    )
    axis.semilogx(
        momentum_steps,
        covariant_rates1,
        color="#CC79A7",
        linestyle="--",
        marker="x",
        markersize=3.0,
        label="covariant axis 2",
    )
    axis.axhline(
        exact_kinetic_rates[0], color="#0072B2", linestyle="--", linewidth=0.8
    )
    axis.axhline(
        exact_kinetic_rates[1], color="#D55E00", linestyle="--", linewidth=0.8
    )
    axis.axhline(0.0, color="0.35", linewidth=0.7)
    axis.format(
        xlabel=r"external step $q/\Lambda$",
        ylabel="kinetic projection",
        xlim=(8.0e-4, 0.24),
        ylim=(-0.13, 0.065),
        grid=False,
        title=r"$\mathbf{(c)}$ External-$q$ projection",
    )
    axis.grid(color="0.90", linewidth=0.45)
    axis.legend(frame=False, loc="center right", ncols=1)

    png = args.output_dir / "phi4_geometric_rg_ising_benchmark.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    with open(args.output_dir / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
