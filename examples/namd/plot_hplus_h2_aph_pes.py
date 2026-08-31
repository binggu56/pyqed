#!/usr/bin/env python3
"""Plot the cached ab initio H3+ APH potential-energy surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev


def periodic(values, phi):
    return np.concatenate((values, values[..., :1]), axis=-1), np.append(phi, 2.0 * np.pi)


def phi_ticks(axis):
    ticks = np.arange(0, 7) / 3.0
    labels = ("0", r"$\frac{1}{3}$", r"$\frac{2}{3}$", "1", r"$\frac{4}{3}$", r"$\frac{5}{3}$", "2")
    axis.set_xticks(ticks, labels)
    axis.set_xlim(0.0, 2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/private/tmp/hplus_h2_aph_scattering/h3plus_casci_aph_pes.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/hplus_h2_aph_scattering/hplus_h2_aph_pes"),
    )
    parser.add_argument("--emax", type=float, default=8.0)
    args = parser.parse_args()

    data = np.load(args.input)
    rho = np.asarray(data["rho"])
    theta = np.asarray(data["theta"])
    phi = np.asarray(data["phi"])
    potential = np.asarray(data["potential"])
    energy = (potential - np.min(potential)) * au2ev
    floor, phi_closed = periodic(np.min(energy, axis=1), phi)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), constrained_layout=True)
    cmap = "viridis"
    mesh = axes[0, 0].pcolormesh(
        phi_closed / np.pi,
        rho,
        np.clip(floor, 0.0, args.emax),
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=args.emax,
        rasterized=True,
    )
    axes[0, 0].set(
        xlabel=r"hyperangle $\phi/\pi$",
        ylabel=r"hyperradius $\rho$ / bohr",
        title=r"$min_{\theta} V(\rho,\theta,\phi)$",
    )
    phi_ticks(axes[0, 0])

    targets = (2.67, 4.52, 6.37)
    titles = ("H$_3^+$ well", "interaction region", "atom-diatom asymptote")
    arrangement_axes = (1.0 / 6.0 + np.arange(6) / 3.0)
    for axis, target, title in zip(axes.flat[1:], targets, titles):
        index = int(np.argmin(np.abs(rho - target)))
        angular, _ = periodic(energy[index], phi)
        axis.pcolormesh(
            phi_closed / np.pi,
            np.rad2deg(theta),
            np.clip(angular, 0.0, args.emax),
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=args.emax,
            rasterized=True,
        )
        for center in arrangement_axes:
            axis.axvline(center, color="white", lw=0.55, ls=":", alpha=0.65)
        axis.set(
            xlabel=r"hyperangle $\phi/\pi$",
            ylabel=r"hyperangle $\theta$ / deg",
            title=rf"{title}, $\rho={rho[index]:.2f}$ bohr",
        )
        phi_ticks(axis)

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            0.02,
            0.97,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
    colorbar = figure.colorbar(mesh, ax=axes, shrink=0.92, pad=0.025, extend="max")
    colorbar.set_label(r"$V-V_{\min}$ / eV")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=360)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
