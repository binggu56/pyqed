#!/usr/bin/env python3
"""Train native MACE LDR fields and build a one-coordinate H2 TTLDR model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ml import MACE
from pyqed.namd.ttldr import TTLDR


def geometry(coordinate):
    distance = float(coordinate[0])
    return np.asarray(
        [[0.0, 0.0, -0.5 * distance], [0.0, 0.0, 0.5 * distance]]
    )


def reference(grid):
    displacement = grid - 1.4
    ground = 0.12 * displacement**2
    excited = 0.32 + 0.08 * (grid - 1.7) ** 2
    coupling = 0.018 * np.exp(-10.0 * (grid - 1.65) ** 2)
    energy = np.zeros((len(grid), 2, 2))
    energy[:, 0, 0] = ground
    energy[:, 1, 1] = excited
    energy[:, 0, 1] = coupling
    energy[:, 1, 0] = coupling
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    angle = 0.08 * np.exp(-12.0 * (midpoint - 1.65) ** 2)
    links = np.zeros((len(midpoint), 2, 2))
    links[:, 0, 0] = np.cos(angle)
    links[:, 0, 1] = -np.sin(angle)
    links[:, 1, 0] = np.sin(angle)
    links[:, 1, 1] = np.cos(angle)
    return energy, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("/private/tmp/h2_mace_ttldr.png")
    )
    args = parser.parse_args()

    grid = np.linspace(1.0, 2.2, args.points)
    energy, links = reference(grid)
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        2,
        geometry_units="bohr",
        channels=8,
        max_ell=2,
        interactions=2,
        radial_basis=6,
        cutoff=4.0,
    ).fit_grid(
        energy,
        (links,),
        hidden=(32, 32),
        epochs=args.epochs,
        tt_rank=8,
        tt_degree=6,
        seed=7,
    )
    checkpoint = args.output.with_suffix(".pt")
    fit.save(checkpoint)

    spacing = grid[1] - grid[0]
    kinetic = (
        np.diag(np.full(args.points, 1.0 / spacing**2))
        + np.diag(np.full(args.points - 1, -0.5 / spacing**2), 1)
        + np.diag(np.full(args.points - 1, -0.5 / spacing**2), -1)
    )
    driver = TTLDR.from_fit(
        fit,
        keo=((1.0, (kinetic,)),),
        overlap_rank=8,
        operator_rank=None,
    )

    predicted_energy = fit.neural_energy.predict(grid[:, None])
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    predicted_links = fit.neural_links[0].predict(midpoint[:, None])
    figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.5), constrained_layout=True)
    axes[0].plot(fit.history)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].plot(grid, energy[:, 0, 0], "k-", label="reference")
    axes[1].plot(grid, predicted_energy[:, 0, 0].real, "o", ms=3, label="MACE")
    axes[1].plot(grid, energy[:, 1, 1], "k-")
    axes[1].plot(grid, predicted_energy[:, 1, 1].real, "o", ms=3)
    axes[1].set_xlabel(r"$R$ (bohr)")
    axes[1].set_ylabel("Energy (a.u.)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].plot(midpoint, links[:, 0, 1], "k-")
    axes[2].plot(midpoint, predicted_links[:, 0, 1].real, "o", ms=3)
    axes[2].set_xlabel(r"$R$ (bohr)")
    axes[2].set_ylabel(r"$L_{01}$")
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    print(f"TTLDR dimensions: {driver.dims}")
    print(f"final MACE loss: {fit.history[-1]:.6e}")
    print(f"figure: {args.output}")
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
