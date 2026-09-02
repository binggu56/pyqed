#!/usr/bin/env python3
"""Plot smooth total nuclear-density marginals from saved H3+ dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.interpolate import RectBivariateSpline


def interpolate_density(qx, qy, density, points=241):
    """Cubic display interpolation; the saved probability is not modified."""

    dense_qx = np.linspace(float(qx[0]), float(qx[-1]), int(points))
    dense_qy = np.linspace(float(qy[0]), float(qy[-1]), int(points))
    spline = RectBivariateSpline(qx, qy, density, kx=3, ky=3, s=0.0)
    smooth = np.clip(spline(dense_qx, dense_qy), 0.0, None)
    return dense_qx, dense_qy, smooth


def main():
    root = Path(__file__).resolve().parents[3]
    run = root / "data" / "h3plus_fci_augccpvdz" / "runs" / "mace_tnldr_17"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=run / "h3plus_mace_tnldr_dynamics.npz"
    )
    parser.add_argument(
        "--output", type=Path,
        default=run / "h3plus_mace_tnldr_total_density_smooth.png",
    )
    parser.add_argument("--interpolation-points", type=int, default=241)
    parser.add_argument("--gamma", type=float, default=0.45)
    args = parser.parse_args()

    data = np.load(args.input)
    times = data["snapshot_times_fs"]
    qx, qy = data["axes"][1:]
    densities = data["snapshot_densities"]
    marginals = np.sum(densities, axis=1)
    smooth_marginals = []
    for marginal in marginals:
        dense_qx, dense_qy, smooth = interpolate_density(
            qx, qy, marginal, args.interpolation_points
        )
        integral = np.trapezoid(
            np.trapezoid(smooth, dense_qy, axis=1), dense_qx
        )
        smooth *= float(np.sum(marginal)) / integral
        smooth_marginals.append(smooth)
    reference = max(
        float(np.max(smooth_marginals[0])), np.finfo(float).tiny
    )
    display_norm = PowerNorm(gamma=float(args.gamma), vmin=0.0, vmax=1.0)

    figure, panels = plt.subplots(
        2, 3, figsize=(8.0, 5.1), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    for panel, time, smooth in zip(panels.flat, times, smooth_marginals):
        image = panel.pcolormesh(
            dense_qx, dense_qy, (smooth / reference).T,
            shading="auto", cmap="magma", norm=display_norm, rasterized=True,
        )
        panel.set_title(fr"$t={time:.1f}$ fs")
        panel.set_aspect("equal")
    for panel in panels[-1]:
        panel.set_xlabel(r"$Q_x$ / bohr")
    for panel in panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ / bohr")
    figure.colorbar(
        image, ax=panels, shrink=0.84,
        label=(
            r"$\rho_{\rm tot}(Q_x,Q_y;t)/\rho_{\rm tot,max}(0)$"
            + fr" (display $\gamma={args.gamma:g}$)"
        ),
    )
    figure.suptitle(
        r"H$_3^+$ total nuclear density: $S_1+S_2$, marginalized over $Q_s$"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=320)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
