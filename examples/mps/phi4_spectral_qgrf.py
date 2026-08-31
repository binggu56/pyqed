"""Solve and plot the coupled spectral functional QGRF fixed point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=17)
    parser.add_argument("--extent", type=float, default=0.8)
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--max-evaluations", type=int, default=250)
    parser.add_argument(
        "--output", type=Path, default=Path("/private/tmp/phi4_spectral_qgrf")
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    field = np.linspace(-args.extent, args.extent, args.points)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=args.quadrature_order,
        include_feshbach=True,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(mass2=-0.6259, quartic=4.3428),
    )
    started = time.perf_counter()
    flow.solve_spectral_fixed_point(
        initial,
        modes=args.modes,
        tolerance=args.tolerance,
        max_evaluations=args.max_evaluations,
    )
    elapsed = time.perf_counter() - started

    beta_u, beta_zt, beta_zx = flow.rates(
        flow.fixed_potential,
        inertia=flow.fixed_inertia,
        stiffness=flow.fixed_stiffness,
    )
    curvature = flow.derivative(flow.fixed_potential, 2)
    center = flow.normalize_index
    quartic = flow.derivative(flow.fixed_potential, 4)

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
        refwidth=2.8,
        refheight=2.0,
        share=False,
        wspace=4.5,
        hspace=8.0,
    )
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"

    axes[0].plot(
        field,
        flow.fixed_potential - flow.fixed_potential[center],
        color=blue,
        linewidth=1.5,
    )
    axes[0].format(
        xlabel=r"background $\phi$",
        ylabel=r"$U_*(\phi)-U_*(0)$",
        title="a  Fixed-point potential",
    )

    axes[1].plot(field, flow.fixed_inertia, color=orange, label=r"$Z_t$")
    axes[1].plot(field, flow.fixed_stiffness, color=green, label=r"$Z_x$")
    axes[1].axhline(1.0, color="0.5", linewidth=0.7, linestyle="--")
    axes[1].format(
        xlabel=r"background $\phi$",
        ylabel="kinetic function",
        title="b  Geometric kinetic flow",
    )
    axes[1].legend(frame=False, loc="best")

    axes[2].plot(field, curvature, color=blue, label=r"$U_*''$")
    axes[2].plot(field, quartic, color=orange, label=r"$U_*''''$")
    axes[2].format(
        xlabel=r"background $\phi$",
        ylabel="local vertex",
        title="c  Local vertices",
    )
    axes[2].legend(frame=False, loc="best")

    floor = 1.0e-14
    axes[3].plot(
        field,
        np.log10(np.maximum(np.abs(beta_u), floor)),
        color=blue,
        label=r"$\beta_U$",
    )
    axes[3].plot(
        field,
        np.log10(np.maximum(np.abs(beta_zt), floor)),
        color=orange,
        label=r"$\beta_{Z_t}$",
    )
    axes[3].plot(
        field,
        np.log10(np.maximum(np.abs(beta_zx), floor)),
        color=green,
        label=r"$\beta_{Z_x}$",
    )
    axes[3].format(
        xlabel=r"background $\phi$",
        ylabel=r"$\log_{10}|\beta|$",
        title="d  Fixed-point residual",
    )
    axes[3].legend(frame=False, loc="best")

    for axis in axes:
        axis.grid(color="0.9", linewidth=0.5)

    stem = args.output / "phi4_spectral_qgrf"
    figure.savefig(stem.with_suffix(".png"), dpi=400)
    figure.savefig(stem.with_suffix(".pdf"))
    np.savez(
        stem.with_suffix(".npz"),
        field=field,
        potential=flow.fixed_potential,
        inertia=flow.fixed_inertia,
        stiffness=flow.fixed_stiffness,
        beta_u=beta_u,
        beta_zt=beta_zt,
        beta_zx=beta_zx,
    )
    summary = {
        "success": flow.success,
        "message": flow.message,
        "elapsed_seconds": elapsed,
        "maximum_residual": float(
            max(np.max(np.abs(beta_u)), np.max(np.abs(beta_zt)), np.max(np.abs(beta_zx)))
        ),
        "eta_t": flow.geometry["eta_t"],
        "eta_x": flow.geometry["eta_x"],
        "dynamic_exponent": flow.geometry["dynamic_exponent"],
        "mass_vertex": float(curvature[center]),
        "quartic_vertex": float(quartic[center]),
        "inertia_range": float(np.ptp(flow.fixed_inertia)),
        "stiffness_range": float(np.ptp(flow.fixed_stiffness)),
    }
    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
