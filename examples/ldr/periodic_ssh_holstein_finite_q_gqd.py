#!/usr/bin/env python3
"""Validate finite-q periodic GQD against an exact coupled-k reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import (
    PeriodicSSHHolsteinGQD,
    PeriodicSSHHolsteinMomentumGQD,
)


def _momentum_label(index, size):
    numerator = 2 * index
    if numerator > size:
        numerator -= 2 * size
    if numerator == 0:
        return r"$0$"
    if abs(numerator) == size:
        return r"$\pi$"
    sign = "-" if numerator < 0 else ""
    numerator = abs(numerator)
    divisor = size
    common = np.gcd(numerator, divisor)
    numerator //= common
    divisor //= common
    top = "" if numerator == 1 else str(numerator)
    return rf"${sign}{top}\pi/{divisor}$"


def _plot(model, output):
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.4))

    dense_k = np.linspace(-np.pi, np.pi, 301)
    dense_bands = np.asarray(
        [
            np.linalg.eigvalsh(
                PeriodicSSHHolsteinGQD.electronic_hamiltonian(
                    model,
                    0.0,
                    kpoint=kpoint,
                )
            )
            for kpoint in dense_k
        ]
    )
    for band, color in enumerate(colors[:2]):
        axes[0, 0].plot(
            dense_k / np.pi,
            dense_bands[:, band],
            color=color,
            linewidth=1.5,
            label=("lower band", "upper band")[band],
        )
    initial = model.initial_k_index
    targets = {
        (initial + model.q_index) % model.ncells,
        (initial - model.q_index) % model.ncells,
    }
    initial_k = model.plot_kpoints[initial]
    initial_energy = model.zero_coordinate_band_energies[
        initial, model.initial_band
    ]
    axes[0, 0].scatter(
        [initial_k / np.pi],
        [initial_energy],
        s=34,
        color="#111111",
        zorder=4,
        label="initial sector",
    )
    for target in sorted(targets):
        target_k = model.plot_kpoints[target]
        target_energy = model.zero_coordinate_band_energies[
            target, model.initial_band
        ]
        axes[0, 0].annotate(
            "",
            xy=(target_k / np.pi, target_energy),
            xytext=(initial_k / np.pi, initial_energy),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
        )
        axes[0, 0].scatter(
            [target_k / np.pi],
            [target_energy],
            s=28,
            facecolor="white",
            edgecolor="#555555",
            zorder=4,
        )
    axes[0, 0].set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    axes[0, 0].set_xticklabels(
        (r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$")
    )
    axes[0, 0].set_xlabel(r"Crystal momentum $k$")
    axes[0, 0].set_ylabel("Electronic energy")
    axes[0, 0].set_title(rf"Bloch sectors coupled by $q={model.qpoint / np.pi:g}\pi$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")

    for state in range(model.nstates):
        edge_state = state in (0, model.nstates - 1)
        axes[0, 1].plot(
            model.coordinates,
            model.energies[:, state],
            color=colors[state > 0] if edge_state else "#666666",
            alpha=0.65,
            linewidth=1.2 if edge_state else 0.8,
        )
    density = model.nuclear_density[0]
    density_scale = 0.24 * np.ptp(model.energies) / np.max(density)
    density_floor = float(np.min(model.energies))
    axes[0, 1].fill_between(
        model.coordinates,
        density_floor,
        density_floor + density_scale * density,
        color="#009E73",
        alpha=0.3,
        linewidth=0.0,
        label=r"$|\chi(Q,0)|^2$",
    )
    axes[0, 1].set_xlabel(r"Standing-wave coordinate $Q_q$")
    axes[0, 1].set_ylabel("Vibronic energy")
    axes[0, 1].set_title(r"Coupled-$k$ adiabatic surfaces")
    axes[0, 1].legend(frameon=False, fontsize=8)

    marker_stride = max(1, len(model.times) // 11)
    for k_index in range(model.ncells):
        label = _momentum_label(k_index, model.ncells)
        axes[1, 0].plot(
            model.times,
            model.momentum_populations[:, k_index],
            color=colors[k_index % len(colors)],
            linewidth=1.5,
            label=rf"GQD $k={label[1:-1]}$",
        )
        axes[1, 0].plot(
            model.times[::marker_stride],
            model.exact_momentum_populations[::marker_stride, k_index],
            linestyle="none",
            marker=("o", "s", "^", "D")[k_index % 4],
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=colors[k_index % len(colors)],
        )
    axes[1, 0].set_xlabel(r"Time $t$ ($\hbar=1$)")
    axes[1, 0].set_ylabel(r"Momentum population $P_k$")
    axes[1, 0].set_ylim(-0.025, 1.025)
    axes[1, 0].set_title(r"Finite-$q$ electron-phonon scattering")
    axes[1, 0].legend(frameon=False, fontsize=7.5, ncol=2)

    floor = 1.0e-16
    diagnostics = (
        (model.state_error, colors[0], "-", r"$\|\Psi_{\rm GQD}-\Psi_{\rm exact}\|$"),
        (model.momentum_population_error, colors[1], "--", r"$\max_k|\Delta P_k|$"),
        (np.abs(model.norm_history - 1.0), colors[2], "-.", "norm drift"),
        (
            np.abs(model.energy_history - model.energy_history[0]),
            colors[3],
            ":",
            "energy drift",
        ),
    )
    for values, color, linestyle, label in diagnostics:
        axes[1, 1].semilogy(
            model.times,
            np.maximum(values, floor),
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=label,
        )
    axes[1, 1].set_xlabel(r"Time $t$ ($\hbar=1$)")
    axes[1, 1].set_ylabel("Absolute error")
    axes[1, 1].set_title("Exact coupled-sector validation")
    axes[1, 1].legend(frameon=False, fontsize=7.5)

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
    parser.add_argument("--ncells", type=int, default=4)
    parser.add_argument("--q-index", type=int, default=1)
    parser.add_argument("--npts", type=int, default=81)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--nout", type=int, default=5)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/periodic_ssh_holstein_finite_q_gqd.pdf"),
    )
    args = parser.parse_args()

    model = PeriodicSSHHolsteinMomentumGQD(
        ncells=args.ncells,
        q_index=args.q_index,
    ).build(
        domain=(-6.0, 6.0),
        npts=args.npts,
    ).run(
        state=args.state,
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
        kpoints=model.kpoints,
        qpoint=model.qpoint,
        mode_profile=model.mode_profile,
        coupling_block_norms=model.coupling_block_norms,
        times=model.times,
        momentum_populations=model.momentum_populations,
        exact_momentum_populations=model.exact_momentum_populations,
        band_momentum_populations=model.band_momentum_populations,
        nuclear_density=model.nuclear_density,
        mean_coordinate=model.mean_coordinate,
        state_error=model.state_error,
        momentum_population_error=model.momentum_population_error,
        norm_history=model.norm_history,
        energy_history=model.energy_history,
    )
    summary = {
        "model": "finite-q periodic SSH-Holstein GQD",
        "parameters": {
            "ncells": model.ncells,
            "q_index": model.q_index,
            "qpoint": model.qpoint,
            "hopping": model.hopping,
            "dimerization": model.dimerization,
            "ssh_coupling": model.ssh_coupling,
            "sublattice_bias": model.sublattice_bias,
            "holstein_coupling": model.holstein_coupling,
            "phonon_frequency": model.phonon_frequency,
            "npts": args.npts,
            "dt": args.dt,
            "steps": args.steps,
            "nout": args.nout,
            "initial_state": args.state,
        },
        "validation": {
            "selection_rule_error": model.selection_rule_error,
            "hamiltonian_error": model.hamiltonian_error,
            "link_unitarity_error": model.link_unitarity_error,
            "max_state_error": model.max_state_error,
            "max_momentum_population_error": (
                model.max_momentum_population_error
            ),
            "max_norm_drift": model.max_norm_drift,
            "max_energy_drift": model.max_energy_drift,
            "max_scattered_population": model.max_scattered_population,
        },
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"q point: {model.qpoint / np.pi:.8f} pi")
    print(f"max scattered population: {model.max_scattered_population:.8f}")
    print(f"selection-rule error: {model.selection_rule_error:.3e}")
    print(f"Hamiltonian error: {model.hamiltonian_error:.3e}")
    print(f"max state error: {model.max_state_error:.3e}")
    print(
        "max momentum-population error: "
        f"{model.max_momentum_population_error:.3e}"
    )
    print(f"max norm drift: {model.max_norm_drift:.3e}")
    print(f"max energy drift: {model.max_energy_drift:.3e}")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()
