#!/usr/bin/env python3
"""Plot direct periodic and local MACE C2H4 conical-intersection PES cuts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "data" / "pyqed"


def nearest(values, target=0.0):
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--periodic",
        type=Path,
        default=DATA_ROOT
        / "ethylene_ci_periodic_2d"
        / "runs"
        / "standard"
        / "ethylene_ci_2d_direct_periodic.npz",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=DATA_ROOT
        / "ethylene_ci_2d"
        / "runs"
        / "standard"
        / "ethylene_ci_2d_tnldr_benchmark.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manuscripts" / "TNLDR" / "figures" / "ethylene_ci_pes_cuts.png",
    )
    args = parser.parse_args()

    with np.load(args.periodic, allow_pickle=False) as archive:
        torsion = np.asarray(archive["torsion"])
        pyramid = np.asarray(archive["pyramidalization"])
        periodic_energy = np.asarray(archive["energies"])
    with np.load(args.local, allow_pickle=False) as archive:
        local_axes = np.asarray(archive["axes"])
        exact = np.asarray(archive["exact_energies"])
        fitted = np.asarray(archive["fitted_energies"])

    periodic_zero = float(
        np.mean(periodic_energy[nearest(torsion), nearest(pyramid)])
    )
    periodic_energy = (periodic_energy - periodic_zero) * au2ev
    local_zero = float(
        np.mean(exact[nearest(local_axes[0]), nearest(local_axes[1])])
    )
    exact = (exact - local_zero) * au2ev
    fitted = (fitted - local_zero) * au2ev
    colors = ("#1665d8", "#d94841")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    periodic_cuts = (
        (torsion, periodic_energy[:, nearest(pyramid)], r"torsion $\phi$ / rad"),
        (pyramid, periodic_energy[nearest(torsion)], r"pyramidalization $\chi$ / rad"),
    )
    for panel, (coordinate, energy, xlabel) in zip(axes[0], periodic_cuts):
        for state, color in enumerate(colors):
            panel.plot(coordinate, energy[:, state], "o-", color=color, label=rf"$S_{state}$")
        panel.axvline(0.0, color="0.65", linewidth=0.8)
        panel.set_xlabel(xlabel)
        panel.set_ylabel(r"$E-E_\mathrm{CI}$ / eV")
        panel.grid(alpha=0.2)
    axes[0, 0].set_title(r"Direct periodic grid, $\chi=0$")
    axes[0, 1].set_title(r"Direct periodic grid, $\phi=0$")
    axes[0, 0].legend(frameon=False, ncol=2)

    local_cuts = (
        (
            local_axes[0],
            exact[:, nearest(local_axes[1])],
            fitted[:, nearest(local_axes[1])],
            r"torsion $\phi$ / rad",
        ),
        (
            local_axes[1],
            exact[nearest(local_axes[0])],
            fitted[nearest(local_axes[0])],
            r"pyramidalization $\chi$ / rad",
        ),
    )
    for panel, (coordinate, reference, prediction, xlabel) in zip(axes[1], local_cuts):
        for state, color in enumerate(colors):
            panel.plot(
                coordinate,
                reference[:, state],
                "o-",
                color=color,
                label=rf"ab initio $S_{state}$",
            )
            panel.plot(
                coordinate,
                prediction[:, state],
                "--",
                color=color,
                linewidth=1.8,
                label=rf"MACE--FTT $S_{state}$",
            )
        panel.axvline(0.0, color="0.65", linewidth=0.8)
        panel.set_xlabel(xlabel)
        panel.set_ylabel(r"$E-E_\mathrm{CI}$ / eV")
        panel.grid(alpha=0.2)
    axes[1, 0].set_title(r"Accepted local fit, $\chi=0$")
    axes[1, 1].set_title(r"Accepted local fit, $\phi=0$")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8)

    fig.suptitle(
        r"C$_2$H$_4$ SA(2)-CASSCF(2,2)/6-31G* conical-intersection cuts",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
