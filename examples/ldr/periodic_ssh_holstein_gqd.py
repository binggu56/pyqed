#!/usr/bin/env python3
"""Validate periodic SSH-Holstein GQD against an exact diabatic reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import PeriodicSSHHolsteinGQD


def _plot(model, output):
    colors = ("#0072B2", "#D55E00")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3))
    coordinate = model.coordinates
    times = model.times

    for state, label in enumerate((r"$E_-$", r"$E_+$")):
        axes[0, 0].plot(
            coordinate,
            model.energies[:, state],
            color=colors[state],
            linewidth=1.5,
            label=label,
        )
    density = model.nuclear_density[0]
    density_scale = 0.32 * np.ptp(model.energies) / np.max(density)
    density_floor = float(np.min(model.energies))
    axes[0, 0].fill_between(
        coordinate,
        density_floor,
        density_floor + density_scale * density,
        color="#009E73",
        alpha=0.28,
        linewidth=0.0,
        label=r"$|\chi(Q,0)|^2$",
    )
    axes[0, 0].set_xlabel(r"Optical coordinate $Q$")
    axes[0, 0].set_ylabel("Energy (model units)")
    axes[0, 0].set_title("Periodic vibronic surfaces")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)

    for state, label in enumerate((r"$P_-$", r"$P_+$")):
        axes[0, 1].plot(
            times,
            model.adiabatic_populations[:, state],
            color=colors[state],
            linewidth=1.5,
            label=f"GQD {label}",
        )
        axes[0, 1].plot(
            times[::10],
            model.exact_adiabatic_populations[::10, state],
            color=colors[state],
            marker=("o", "s")[state],
            markersize=3.2,
            linestyle="none",
            markerfacecolor="white",
            label=f"exact {label}",
        )
    axes[0, 1].set_xlabel(r"Time $t$ ($\hbar=1$)")
    axes[0, 1].set_ylabel("Adiabatic population")
    axes[0, 1].set_ylim(-0.025, 1.025)
    axes[0, 1].set_title("Nonadiabatic transfer")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)

    crossing = int(np.argmin(np.abs(model.mean_coordinate)))
    snapshots = (0, crossing, len(times) - 1)
    styles = (
        ("#0072B2", "-", r"$t=0$"),
        ("#009E73", "--", rf"$t={times[crossing]:.1f}$"),
        ("#D55E00", "-.", rf"$t={times[-1]:.1f}$"),
    )
    for sample, (color, linestyle, label) in zip(snapshots, styles):
        axes[1, 0].plot(
            coordinate,
            model.nuclear_density[sample],
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=label,
        )
    axes[1, 0].set_xlabel(r"Optical coordinate $Q$")
    axes[1, 0].set_ylabel(r"Nuclear density $\rho(Q)$")
    axes[1, 0].set_title("Quantum phonon wavepacket")
    axes[1, 0].legend(frameon=False, fontsize=8)

    floor = 1.0e-16
    axes[1, 1].semilogy(
        times,
        np.maximum(model.state_error, floor),
        color="#0072B2",
        linewidth=1.4,
        label=r"$\|\Psi_{\rm GQD}-\Psi_{\rm exact}\|$",
    )
    axes[1, 1].semilogy(
        times,
        np.maximum(np.abs(model.norm_history - 1.0), floor),
        color="#009E73",
        linestyle="--",
        linewidth=1.4,
        label="norm drift",
    )
    axes[1, 1].semilogy(
        times,
        np.maximum(
            np.abs(model.energy_history - model.energy_history[0]),
            floor,
        ),
        color="#CC79A7",
        linestyle="-.",
        linewidth=1.4,
        label="energy drift",
    )
    axes[1, 1].set_xlabel(r"Time $t$ ($\hbar=1$)")
    axes[1, 1].set_ylabel("Absolute error")
    axes[1, 1].set_title("Exact-reference validation")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis.text(
            -0.12,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    png = output.with_suffix(".png")
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=111)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--nout", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/periodic_ssh_holstein_gqd.pdf"),
    )
    args = parser.parse_args()

    model = PeriodicSSHHolsteinGQD().build(
        domain=(-7.0, 7.0),
        npts=args.npts,
    ).run(
        dt=args.dt,
        nsteps=args.steps,
        nout=args.nout,
    )
    png = _plot(model, args.output)
    data = args.output.with_suffix(".npz")
    np.savez_compressed(
        data,
        coordinates=model.coordinates,
        energies=model.energies,
        times=model.times,
        adiabatic_populations=model.adiabatic_populations,
        exact_adiabatic_populations=model.exact_adiabatic_populations,
        diabatic_populations=model.diabatic_populations,
        nuclear_density=model.nuclear_density,
        mean_coordinate=model.mean_coordinate,
        state_error=model.state_error,
        norm_history=model.norm_history,
        energy_history=model.energy_history,
    )
    summary = {
        "model": "periodic two-sublattice SSH-Holstein chain",
        "parameters": {
            "hopping": model.hopping,
            "dimerization": model.dimerization,
            "ssh_coupling": model.ssh_coupling,
            "sublattice_bias": model.sublattice_bias,
            "holstein_coupling": model.holstein_coupling,
            "phonon_frequency": model.phonon_frequency,
            "kpoint": model.kpoint,
            "npts": args.npts,
            "dt": args.dt,
            "steps": args.steps,
            "nout": args.nout,
        },
        "validation": {
            "minimum_gap": model.minimum_gap,
            "hamiltonian_error": model.hamiltonian_error,
            "link_unitarity_error": model.link_unitarity_error,
            "max_state_error": model.max_state_error,
            "max_population_error": model.max_population_error,
            "max_norm_drift": model.max_norm_drift,
            "max_energy_drift": model.max_energy_drift,
            "max_excited_population": model.max_excited_population,
        },
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"minimum gap: {model.minimum_gap:.8e}")
    print(f"max excited population: {model.max_excited_population:.8f}")
    print(f"Hamiltonian error: {model.hamiltonian_error:.3e}")
    print(f"max state error: {model.max_state_error:.3e}")
    print(f"max population error: {model.max_population_error:.3e}")
    print(f"max norm drift: {model.max_norm_drift:.3e}")
    print(f"max energy drift: {model.max_energy_drift:.3e}")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()
