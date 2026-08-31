"""Evolve full ``U(phi)``, ``Z_t(phi)``, and ``Z_x(phi)`` with Hamiltonian QGRG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pyqed.narg.geometric_rg import (
    Phi4FunctionalQGRG,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_functional_qgrg"),
    )
    parser.add_argument("--ell-max", type=float, default=0.12)
    parser.add_argument("--nfield", type=int, default=121)
    parser.add_argument("--field-max", type=float, default=0.8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    field = np.linspace(-args.field_max, args.field_max, args.nfield)
    if args.nfield % 2 != 1:
        raise ValueError("nfield must be odd so that the grid contains phi=0.")
    center = args.nfield // 2
    qgrg = Phi4FunctionalQGRG(field, spatial_dimension=1, stencil_size=7)
    shell = Phi4GaussianShell(spatial_dimension=1)
    initial_couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    initial_potential = shell.potential(field, initial_couplings)
    initial = np.concatenate(
        [initial_potential, np.ones(args.nfield), np.ones(args.nfield)]
    )

    def rhs(_, values):
        potential, inertia, stiffness = values.reshape(3, args.nfield)
        rates = list(
            qgrg.rates(potential, inertia=inertia, stiffness=stiffness)
        )
        # The homogeneous vacuum energy is dynamically irrelevant.
        rates[0] = rates[0] - rates[0][center]
        return np.concatenate(rates)

    solution = solve_ivp(
        rhs,
        (0.0, args.ell_max),
        initial,
        method="DOP853",
        rtol=2.0e-8,
        atol=2.0e-10,
        max_step=0.002,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    ell = np.linspace(0.0, args.ell_max, 5)
    trajectory = solution.sol(ell).reshape(3, args.nfield, ell.size)
    potential = trajectory[0]
    inertia = trajectory[1]
    stiffness = trajectory[2]
    final_geometry = qgrg.shell_geometry(
        potential[:, -1],
        inertia=inertia[:, -1],
        stiffness=stiffness[:, -1],
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
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, ell.size))

    axis = axes[0, 0]
    for index, value in enumerate(ell):
        axis.plot(
            field,
            potential[:, index] - potential[center, index],
            color=colors[index],
            label=rf"$\ell={value:.2f}$",
        )
    axis.set(xlabel=r"background $\phi$", ylabel=r"$U_\ell(\phi)-U_\ell(0)$")
    axis.legend(frameon=False, ncols=2)

    axis = axes[0, 1]
    for index, value in enumerate(ell):
        axis.plot(field, inertia[:, index] - 1.0, color=colors[index])
    axis.set(xlabel=r"background $\phi$", ylabel=r"$Z_{t,\ell}(\phi)-1$")

    axis = axes[1, 0]
    for index, value in enumerate(ell):
        axis.plot(field, stiffness[:, index] - 1.0, color=colors[index])
    axis.axhline(0.0, color="0.4", linewidth=0.7)
    axis.set(xlabel=r"background $\phi$", ylabel=r"$Z_{x,\ell}(\phi)-1$")

    axis = axes[1, 1]
    axis.plot(
        field,
        final_geometry["temporal_rate"],
        color="#0072B2",
        label=r"$\partial_\ell Z_t$",
    )
    axis.plot(
        field,
        final_geometry["spatial_rate"],
        color="#D55E00",
        label=r"$\partial_\ell Z_x$",
    )
    axis.axhline(0.0, color="0.4", linewidth=0.7)
    axis.set(xlabel=r"background $\phi$", ylabel="final shell response")
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

    png = args.output_dir / "phi4_functional_qgrg.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    np.savez(
        args.output_dir / "flow.npz",
        ell=ell,
        field=field,
        potential=potential,
        inertia=inertia,
        stiffness=stiffness,
        temporal_rate=final_geometry["temporal_rate"],
        spatial_rate=final_geometry["spatial_rate"],
        metric_rate=final_geometry["metric_rate"],
    )
    summary = {
        "model": "1+1D Hamiltonian functional QGRG",
        "field_points": args.nfield,
        "ell_max": args.ell_max,
        "initial_mass2": initial_couplings.mass2,
        "initial_quartic": initial_couplings.quartic,
        "minimum_inertia": float(np.min(inertia[:, -1])),
        "minimum_stiffness": float(np.min(stiffness[:, -1])),
        "maximum_temporal_change": float(np.max(np.abs(inertia[:, -1] - 1.0))),
        "maximum_spatial_change": float(np.max(np.abs(stiffness[:, -1] - 1.0))),
        "maximum_metric_rate": float(np.max(final_geometry["metric_rate"])),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
