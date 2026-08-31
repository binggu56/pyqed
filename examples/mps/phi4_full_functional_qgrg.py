"""Diagnose full-grid fixed points and low-rank Feshbach self-energies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt
from matplotlib.ticker import LogFormatterMathtext

from pyqed.narg.geometric_rg import (
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def _branch(extent, points):
    field = np.linspace(-extent, extent, points)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=12,
        include_feshbach=False,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(
            mass2=-0.22057162344985815,
            quartic=4.121003313906062,
        ),
    )
    flow.solve_fixed_point(initial, tolerance=1.0e-9)
    return flow


def _spectrum(flow, step=2.0e-6):
    center = flow.field.size // 2
    fixed = flow.fixed_potential
    independent = fixed[center:]

    def expand(values):
        return np.concatenate((values[:0:-1], values))

    jacobian = np.empty((independent.size, independent.size))
    for index in range(independent.size):
        displacement = np.zeros_like(independent)
        displacement[index] = step
        upper = flow.potential_rate(expand(independent + displacement))[center:]
        lower = flow.potential_rate(expand(independent - displacement))[center:]
        jacobian[:, index] = (upper - lower) / (2.0 * step)
    return np.linalg.eigvals(jacobian)


def main():
    output = Path("/private/tmp/phi4_full_functional_qgrg")
    output.mkdir(parents=True, exist_ok=True)
    branches = [_branch(0.6, 17), _branch(0.8, 23), _branch(1.2, 33)]
    reference = branches[0]
    momenta = np.linspace(-0.25, 0.25, 31)
    self_energy = reference.fit_self_energy(
        reference.fixed_potential, momenta, tolerance=1.0e-6
    )
    eigenvalues = _spectrum(reference)

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=2,
        refwidth=3.0,
        refheight=2.2,
        share=False,
        wspace=4.5,
        hspace=5.0,
    )
    colors = ("#0072B2", "#D55E00", "#009E73")

    axis = axes[0]
    for flow, color in zip(branches, colors):
        center = flow.field.size // 2
        axis.plot(
            flow.field,
            flow.fixed_potential - flow.fixed_potential[center],
            color=color,
            label=rf"$|\phi|\leq {flow.field[-1]:.1f}$",
        )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$U_*(\phi)-U_*(0)$",
        title="a  Full-grid branches",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower center")

    axis = axes[1]
    for flow, color in zip(branches, colors):
        axis.plot(
            flow.field,
            flow.derivative(flow.fixed_potential, 2),
            color=color,
        )
    axis.axhline(-1.0, color="0.4", linestyle="--", linewidth=0.8)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$U_*''(\phi)$",
        title="b  Fixed-point curvature",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)

    axis = axes[2]
    for index, color in zip((15, 23, 30), colors):
        axis.plot(
            reference.field,
            self_energy["values"][:, index],
            color=color,
            label=rf"$q={momenta[index]:.3f}$",
        )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$\Sigma_F(\phi,q)$",
        title=r"c  Self-energy cuts",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="best")

    axis = axes[3]
    singular_values = np.linalg.svd(
        self_energy["values"], compute_uv=False
    )
    axis.semilogy(
        np.arange(1, singular_values.size + 1),
        singular_values / singular_values[0],
        color="#CC79A7",
        marker="o",
        markersize=3,
    )
    axis.format(
        xlabel="SVD rank",
        ylabel=r"$\sigma_r/\sigma_1$",
        title="d  Self-energy compression",
        grid=False,
    )
    axis.yaxis.set_major_formatter(LogFormatterMathtext())
    axis.grid(color="0.9", linewidth=0.5)

    png = output / "phi4_full_functional_qgrg.png"
    figure.savefig(png, dpi=400)
    figure.savefig(png.with_suffix(".pdf"))
    summary = {
        "branches": [
            {
                "extent": float(flow.field[-1]),
                "points": int(flow.field.size),
                "success": flow.success,
                "max_residual": float(np.max(np.abs(flow.fixed_beta))),
                "minimum_gap2": float(
                    np.min(1.0 + flow.derivative(flow.fixed_potential, 2))
                ),
                "potential_range": float(np.ptp(flow.fixed_potential)),
            }
            for flow in branches
        ],
        "self_energy_rank": int(self_energy["rank"]),
        "self_energy_relative_error": self_energy["relative_error"],
        "linearized_eigenvalues": [
            [float(value.real), float(value.imag)]
            for value in sorted(eigenvalues, key=lambda value: -value.real)
        ],
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
