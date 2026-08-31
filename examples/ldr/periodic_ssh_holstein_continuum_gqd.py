#!/usr/bin/env python3
"""Validate continuum-embedded periodic GQD by exact Feshbach projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import FeshbachEmbedding, PeriodicSSHHolsteinGQD


def _lorentzian_spectrum(energies, poles, weights, eta):
    delta = energies[:, None] - poles[None, :]
    return np.sum(
        weights[None, :] * eta / (np.pi * (delta**2 + eta**2)),
        axis=1,
    )


def _calculate(npts, nenergy, eta):
    model = PeriodicSSHHolsteinGQD().build(
        domain=(-6.0, 6.0),
        npts=npts,
    )
    embedded = FeshbachEmbedding.from_ldr(
        model.solver,
        active_states=1,
        diagonalize_continuum=True,
    )
    full_hamiltonian = model.solver.hamiltonian(sparse=True).toarray()
    full_energies, full_states = np.linalg.eigh(full_hamiltonian)
    active_weights = np.sum(
        np.abs(full_states[embedded.active_indices]) ** 2,
        axis=0,
    )
    upper = min(4.5, float(full_energies[-1]))
    energy_grid = np.linspace(float(full_energies[0]) - 0.2, upper, nenergy)
    embedded.run_spectrum(energy_grid, eta=eta)

    exact_spectrum = _lorentzian_spectrum(
        energy_grid,
        full_energies,
        active_weights,
        eta,
    )
    active_hamiltonian = embedded.active_hamiltonian.toarray()
    active_poles = np.linalg.eigvalsh(active_hamiltonian)
    truncated_spectrum = _lorentzian_spectrum(
        energy_grid,
        active_poles,
        np.ones(active_poles.size),
        eta,
    )
    embedded.exact_spectrum = exact_spectrum
    embedded.truncated_spectrum = truncated_spectrum
    embedded.full_energies = full_energies
    embedded.full_active_weights = active_weights
    embedded.maximum_spectrum_error = float(
        np.max(np.abs(embedded.spectral_density - exact_spectrum))
    )
    return model, embedded


def _plot(model, embedded, output):
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5))

    axes[0, 0].plot(
        model.coordinates,
        model.energies[:, 0],
        color=blue,
        linewidth=1.5,
        label=r"active $P$ surface",
    )
    axes[0, 0].plot(
        model.coordinates,
        model.energies[:, 1],
        color=orange,
        linewidth=1.5,
        label=r"continuum $Q$ surface",
    )
    axes[0, 0].set_xlabel(r"Optical coordinate $Q$")
    axes[0, 0].set_ylabel("Energy (model units)")
    axes[0, 0].set_title("Local projector partition")
    axes[0, 0].legend(frameon=False, fontsize=8)

    energies = embedded.energy_grid
    axes[0, 1].plot(
        energies,
        embedded.spectral_density,
        color=blue,
        linewidth=1.5,
        label="Feshbach embedding",
    )
    stride = max(1, energies.size // 45)
    axes[0, 1].plot(
        energies[::stride],
        embedded.exact_spectrum[::stride],
        color=green,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        linestyle="none",
        label="complete GQD",
    )
    axes[0, 1].plot(
        energies,
        embedded.truncated_spectrum,
        color=orange,
        linestyle="--",
        linewidth=1.2,
        label="drop continuum",
    )
    axes[0, 1].set_xlabel(r"Energy $E$")
    axes[0, 1].set_ylabel(r"Projected spectrum $A_P(E)$")
    axes[0, 1].set_title("Exact continuum elimination")
    axes[0, 1].legend(frameon=False, fontsize=8)

    scale = embedded.nactive
    axes[1, 0].plot(
        energies,
        embedded.self_energy_trace.real / scale,
        color=purple,
        linewidth=1.2,
        label=r"$\mathrm{Re}\,\mathrm{Tr}\,\Sigma^R/N_P$",
    )
    axes[1, 0].plot(
        energies,
        embedded.hybridization_trace / scale,
        color=orange,
        linewidth=1.2,
        label=r"$\mathrm{Tr}\,\Gamma/N_P$",
    )
    axes[1, 0].axhline(0.0, color="#666666", linewidth=0.7)
    axes[1, 0].set_xlabel(r"Energy $E$")
    axes[1, 0].set_ylabel("Average self-energy")
    axes[1, 0].set_title("Continuum shift and broadening")
    axes[1, 0].legend(frameon=False, fontsize=8)

    visible = embedded.full_energies <= energies[-1]
    marker_size = 16.0 + 42.0 * embedded.full_active_weights[visible]
    axes[1, 1].scatter(
        embedded.full_energies[visible],
        embedded.full_active_weights[visible],
        s=marker_size,
        facecolor=blue,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
    )
    axes[1, 1].set_xlabel(r"Exact vibronic energy $E_j$")
    axes[1, 1].set_ylabel(r"Active weight $\langle j|P|j\rangle$")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_title("Hybridized active-continuum poles")

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
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    png = output.with_suffix(".png")
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=41)
    parser.add_argument("--nenergy", type=int, default=600)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/periodic_ssh_holstein_continuum_gqd.pdf"
        ),
    )
    args = parser.parse_args()

    model, embedded = _calculate(args.npts, args.nenergy, args.eta)
    png = _plot(model, embedded, args.output)
    data = args.output.with_suffix(".npz")
    np.savez_compressed(
        data,
        coordinates=model.coordinates,
        local_energies=model.energies,
        energy_grid=embedded.energy_grid,
        embedded_spectrum=embedded.spectral_density,
        exact_spectrum=embedded.exact_spectrum,
        truncated_spectrum=embedded.truncated_spectrum,
        self_energy_trace=embedded.self_energy_trace,
        hybridization_trace=embedded.hybridization_trace,
        full_energies=embedded.full_energies,
        full_active_weights=embedded.full_active_weights,
    )
    summary = {
        "method": "projector GQD with a Feshbach electronic continuum",
        "npts": args.npts,
        "nactive": embedded.nactive,
        "ncontinuum": embedded.ncontinuum,
        "eta": args.eta,
        "continuum_coupling_norm": embedded.continuum_coupling_norm,
        "minimum_projector_overlap": embedded.minimum_projector_overlap,
        "maximum_projector_leakage": embedded.maximum_projector_leakage,
        "maximum_spectrum_error": embedded.maximum_spectrum_error,
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"active dimension: {embedded.nactive}")
    print(f"continuum dimension: {embedded.ncontinuum}")
    print(f"continuum coupling norm: {embedded.continuum_coupling_norm:.8e}")
    print(f"minimum projector overlap: {embedded.minimum_projector_overlap:.8f}")
    print(f"maximum spectrum error: {embedded.maximum_spectrum_error:.3e}")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()
