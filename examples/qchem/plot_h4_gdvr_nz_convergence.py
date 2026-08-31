#!/usr/bin/env python3
"""Plot longitudinal-grid convergence of an H4 GDVR collective-coordinate cut."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", nargs="+", type=Path, metavar="SCAN")
    parser.add_argument("--output", type=Path, default=Path("h4_gdvr_nz_convergence.png"))
    args = parser.parse_args()
    if len(args.scans) < 2:
        parser.error("provide at least two GDVR scan files")

    scans = [np.load(path) for path in args.scans]
    if any(len(scan["q_plus"]) != 1 for scan in scans):
        raise ValueError("Expected one-dimensional q_plus cuts")
    q = np.array(
        [
            value
            for value in scans[0]["q_minus"]
            if all(np.any(np.isclose(scan["q_minus"], value, atol=1.0e-12)) for scan in scans[1:])
        ]
    )
    if len(q) < 3:
        raise ValueError("The scans have fewer than three common q_minus points")
    indices = [
        np.array([int(np.argmin(abs(scan["q_minus"] - value))) for value in q])
        for scan in scans
    ]

    gto = scans[-1]["gto_rhf_energy"][0, indices[-1]]
    energies = [
        scan["gdvr_rhf_energy"][0, point_indices]
        for scan, point_indices in zip(scans, indices)
    ]
    nz_values = [int(scan["nz"]) for scan in scans]
    rel_gto = 1000.0 * (gto - gto.min())
    relative = [1000.0 * (energy - energy.min()) for energy in energies]
    force_gto = -np.gradient(gto, q, edge_order=2)
    forces = [-np.gradient(energy, q, edge_order=2) for energy in energies]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    axes[0].plot(q, rel_gto, "o-", label="GTO-RHF")
    for nz, values in zip(nz_values, relative):
        axes[0].plot(q, values, "o-", label=rf"GDVR $N_z={nz}$")
    axes[0].set_ylabel("Relative energy (mEh)")
    axes[0].legend(frameon=False)
    axes[0].set_title("Potential-energy cut")

    for previous, current, nz_previous, nz_current in zip(
        relative[:-1], relative[1:], nz_values[:-1], nz_values[1:]
    ):
        axes[1].plot(
            q,
            current - previous,
            "o-",
            label=rf"$N_z={nz_current}-{nz_previous}$",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Relative-PES change (mEh)")
    axes[1].set_title("Successive grid changes")
    axes[1].legend(frameon=False)

    axes[2].plot(q, force_gto, "o-", label="GTO-RHF")
    for nz, values in zip(nz_values, forces):
        axes[2].plot(q, values, "o-", label=rf"GDVR $N_z={nz}$")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel(r"$-dE/dq_-$ ($E_h$/bohr)")
    axes[2].set_title("Finite-difference force")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel(r"$q_-$ (bohr)")
        ax.grid(alpha=0.2)
    fig.suptitle(r"Linear H$_4$, $q_+=0$: GDVR longitudinal-grid convergence")
    fig.savefig(args.output, dpi=240)
    plt.close(fig)

    print(f"Saved {args.output}")
    for previous, current, force_previous, force_current, nz_previous, nz_current in zip(
        relative[:-1], relative[1:], forces[:-1], forces[1:], nz_values[:-1], nz_values[1:]
    ):
        shape_delta = current - previous
        force_delta = force_current[1:-1] - force_previous[1:-1]
        print(
            f"N{nz_current}-N{nz_previous} relative-PES change (mEh): "
            f"RMSE={np.sqrt(np.mean(shape_delta**2)):.6f}, "
            f"maxabs={np.max(np.abs(shape_delta)):.6f}"
        )
        print(
            f"N{nz_current}-N{nz_previous} interior-force change (Eh/bohr): "
            f"RMSE={np.sqrt(np.mean(force_delta**2)):.6f}, "
            f"maxabs={np.max(np.abs(force_delta)):.6f}"
        )


if __name__ == "__main__":
    main()
