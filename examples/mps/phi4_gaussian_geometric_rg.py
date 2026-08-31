"""Continuum Gaussian-frame geometric RG diagnostic for ``phi4`` theory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pyqed.narg.geometric_rg import Phi4GaussianCouplings, Phi4GaussianShell


def _jacobian(shell, couplings, indices=(1, 3), step=1.0e-6):
    point = couplings.asarray()
    selected = list(indices)
    jacobian = np.empty((len(indices), len(indices)))
    for column, index in enumerate(indices):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        beta_plus = shell.beta(Phi4GaussianCouplings.from_array(plus)).asarray()
        beta_minus = shell.beta(Phi4GaussianCouplings.from_array(minus)).asarray()
        jacobian[:, column] = (
            beta_plus[selected] - beta_minus[selected]
        ) / (2.0 * step)
    return jacobian


def _integrate(shell, initial, ell_max):
    def rhs(_, values):
        return shell.beta(Phi4GaussianCouplings.from_array(values)).asarray()

    return solve_ivp(
        rhs,
        (0.0, ell_max),
        initial.asarray(),
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.01,
        dense_output=True,
    )


def _integrate_inertia_feedback(shell, initial, ell_max):
    def rhs(_, values):
        couplings = Phi4GaussianCouplings(
            mass2=values[0], quartic=values[1]
        )
        beta, beta_z2 = shell.beta_z2(couplings, values[2])
        return np.array([beta.mass2, beta.quartic, beta_z2])

    return solve_ivp(
        rhs,
        (0.0, ell_max),
        np.array([initial.mass2, initial.quartic, 0.0]),
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.01,
        dense_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_gaussian_geometric_rg"),
    )
    parser.add_argument("--ell-max", type=float, default=1.2)
    parser.add_argument("--quadrature-order", type=int, default=36)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shell = Phi4GaussianShell(spatial_dimension=1)
    fixed_mass2 = -2.0 / 5.0
    fixed_quartic = -8.0 * np.pi * fixed_mass2 * np.sqrt(1.0 + fixed_mass2)
    fixed = Phi4GaussianCouplings(mass2=fixed_mass2, quartic=fixed_quartic)
    jacobian = _jacobian(shell, fixed)
    eigenvalues = np.linalg.eigvals(jacobian)

    initial = Phi4GaussianCouplings(mass2=-0.30, quartic=6.0)
    flow = _integrate(shell, initial, args.ell_max)
    feedback_flow = _integrate_inertia_feedback(shell, initial, args.ell_max)
    ell = np.linspace(0.0, args.ell_max, 241)
    trajectory = flow.sol(ell)
    feedback_trajectory = feedback_flow.sol(ell)

    final_feedback = Phi4GaussianCouplings(
        mass2=feedback_trajectory[0, -1],
        quartic=feedback_trajectory[1, -1],
    )
    final_z2 = feedback_trajectory[2, -1]
    field = np.linspace(-0.8, 0.8, 321)
    inertia = 1.0 + 0.5 * final_z2 * field**2
    inertia_derivative = final_z2 * field
    feedback_metric = shell.metric_rate(
        field,
        final_feedback,
        inertia=inertia,
        inertia_derivative=inertia_derivative,
    )
    inertia_beta = shell.inertia_beta(
        field,
        final_feedback,
        inertia=inertia,
        inertia_derivative=inertia_derivative,
    )

    background_couplings = Phi4GaussianCouplings(
        mass2=0.2, cubic=0.4, quartic=0.8
    )
    metric = shell.metric_rate(field, background_couplings)
    weighted = shell.weighted_metric_rate(
        field, background_couplings, energy=0.0
    )

    widths = np.geomspace(0.02, 0.35, 11)
    residual4 = np.empty_like(widths)
    residual3 = np.empty_like(widths)
    for index, width in enumerate(widths):
        correction = shell.residual_corrections(
            0.2,
            background_couplings,
            log_width=width,
            quadrature_order=args.quadrature_order,
        )
        residual3[index] = correction["three_boson"]
        residual4[index] = correction["four_boson"]
    residual_slope, residual_intercept = np.polyfit(
        np.log(widths), np.log(np.abs(residual4)), 1
    )

    mass_grid = np.linspace(-0.72, 0.32, 43)
    quartic_grid = np.linspace(0.05, 13.0, 45)
    rr, uu = np.meshgrid(mass_grid, quartic_grid)
    beta_r = np.empty_like(rr)
    beta_u = np.empty_like(uu)
    for index in np.ndindex(rr.shape):
        beta = shell.beta(
            Phi4GaussianCouplings(mass2=rr[index], quartic=uu[index])
        )
        beta_r[index] = beta.mass2
        beta_u[index] = beta.quartic
    speed = np.hypot(beta_r, beta_u)
    beta_r /= np.maximum(speed, 1.0e-14)
    beta_u /= np.maximum(speed, 1.0e-14)

    colors = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)

    axis = axes[0, 0]
    axis.streamplot(
        mass_grid,
        quartic_grid,
        beta_r,
        beta_u,
        color="#B0B0B0",
        density=0.9,
        linewidth=0.7,
        arrowsize=0.7,
    )
    axis.plot(
        trajectory[1],
        trajectory[3],
        color=colors["blue"],
        linestyle="--",
        linewidth=1.5,
        label="local potential",
    )
    axis.plot(
        feedback_trajectory[0],
        feedback_trajectory[1],
        color=colors["green"],
        linewidth=2.0,
        label=r"with $Z_\Lambda(\phi)$",
    )
    axis.scatter(
        [fixed.mass2],
        [fixed.quartic],
        color=colors["orange"],
        marker="*",
        s=95,
        zorder=3,
    )
    axis.annotate(
        "nontrivial fixed point",
        (fixed.mass2, fixed.quartic),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set(xlabel=r"$r_\Lambda$", ylabel=r"$\lambda_\Lambda$")
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    axis = axes[0, 1]
    axis.plot(
        ell,
        trajectory[1],
        color=colors["blue"],
        linestyle="--",
        linewidth=1.2,
        label=r"$r_\Lambda$ (LPA)",
    )
    axis.plot(
        ell,
        feedback_trajectory[0],
        color=colors["blue"],
        linewidth=2.0,
        label=r"$r_\Lambda$ (geometric)",
    )
    axis.set(
        xlabel=r"$\ell=\log(\Lambda_0/\Lambda)$",
        ylabel=r"$r_\Lambda$",
    )
    coupling_axis = axis.twinx()
    coupling_axis.plot(
        ell,
        feedback_trajectory[1],
        color=colors["orange"],
        linewidth=1.8,
        label=r"$\lambda_\Lambda$",
    )
    coupling_axis.plot(
        ell,
        feedback_trajectory[2],
        color=colors["green"],
        linewidth=1.8,
        label=r"$z_{2,\Lambda}$",
    )
    coupling_axis.set_ylabel(r"$\lambda_\Lambda,\ z_{2,\Lambda}$")
    lines = axis.get_lines() + coupling_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.plot(
        field,
        feedback_metric,
        color=colors["blue"],
        label=r"$\partial_\ell g_{\phi\phi}$",
    )
    axis.plot(
        field,
        inertia_beta,
        color=colors["green"],
        label=r"$\partial_\ell Z_\Lambda$",
    )
    axis.set(xlabel=r"background $\phi$", ylabel="geometric shell rate")
    inertia_axis = axis.twinx()
    inertia_axis.plot(
        field,
        inertia - 1.0,
        color=colors["orange"],
        linestyle="--",
        label=r"$Z_\Lambda(\phi)-1$",
    )
    inertia_axis.set_ylabel(r"$Z_\Lambda(\phi)-1$")
    lines = axis.get_lines() + inertia_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    axis = axes[1, 1]
    axis.loglog(
        widths,
        np.abs(residual4),
        "o-",
        color=colors["orange"],
        label=r"$|\Delta\mathcal{E}_4|$",
    )
    reference = np.exp(residual_intercept) * widths**3
    axis.loglog(widths, reference, "--", color="#555555", label=r"$(d\ell)^3$")
    axis.set(xlabel=r"finite shell width $d\ell$", ylabel="Feshbach correction")
    axis.legend(frameon=False, fontsize=8)

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(0.02, 0.97, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(alpha=0.18, linewidth=0.6)

    figure = args.output_dir / "phi4_gaussian_geometric_rg.png"
    figure_pdf = figure.with_suffix(".pdf")
    fig.savefig(figure, dpi=240)
    fig.savefig(figure_pdf)

    summary = {
        "model": "1+1D homogeneous phi4 Gaussian geometric RG",
        "flow_equation": "Hamiltonian spatial-shell local-potential flow",
        "fixed_point": {
            "mass2": fixed.mass2,
            "quartic": fixed.quartic,
            "jacobian_eigenvalues": np.sort(eigenvalues).tolist(),
        },
        "initial_couplings": initial.asarray().tolist(),
        "final_couplings": trajectory[:, -1].tolist(),
        "geometric_feedback": {
            "final_mass2": float(feedback_trajectory[0, -1]),
            "final_quartic": float(feedback_trajectory[1, -1]),
            "final_inertia2": float(final_z2),
            "maximum_inertia": float(np.max(inertia)),
        },
        "maximum_metric_rate": float(np.max(metric)),
        "minimum_weighted_metric_rate": float(np.min(weighted)),
        "three_boson_corrections": residual3.tolist(),
        "four_boson_shell_width_slope": float(residual_slope),
        "four_boson_corrections": residual4.tolist(),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez(
        args.output_dir / "flow_data.npz",
        ell=ell,
        trajectory=trajectory,
        feedback_trajectory=feedback_trajectory,
        field=field,
        metric=metric,
        weighted_metric=weighted,
        feedback_metric=feedback_metric,
        inertia=inertia,
        inertia_beta=inertia_beta,
        shell_widths=widths,
        residual3=residual3,
        residual4=residual4,
    )
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
