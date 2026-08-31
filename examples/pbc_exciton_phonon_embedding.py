"""Finite-q exciton-phonon Feshbach embedding reference calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import (
    ExcitonPhononChannel,
    ExcitonPhononContinuum,
    ProjectedTDAContinuum,
    bose_occupation,
)
from pyqed.units import au2ev, au2fs


def _continuum(energies, center, strength, width, phase):
    spacing = float(np.mean(np.diff(energies)))
    envelope = np.exp(-0.5 * ((energies - center) / width) ** 2)
    coupling = strength * np.sqrt(spacing) * envelope * np.exp(1.0j * phase)
    return ProjectedTDAContinuum(
        np.diag(energies).astype(np.complex128),
        np.zeros((energies.size, 0), dtype=np.complex128),
        coupling[None, :],
    )


def run(output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    active_energy = 0.205
    target_1 = np.linspace(0.155, 0.335, 80)
    target_2 = np.linspace(0.175, 0.355, 80)
    continuum_1 = _continuum(target_1, 0.205, 0.020, 0.050, 0.0)
    continuum_2 = _continuum(target_2, 0.235, 0.014, 0.045, 0.7)
    temperature = 300.0
    channels = (
        ExcitonPhononChannel(
            continuum_1,
            frequency=0.008,
            occupation=bose_occupation(0.008, temperature),
            phonon_q_index=1,
            branch=0,
        ),
        ExcitonPhononChannel(
            continuum_2,
            frequency=0.012,
            occupation=bose_occupation(0.012, temperature),
            phonon_q_index=2,
            branch=1,
        ),
    )
    continuum = ExcitonPhononContinuum(channels)
    energies = np.linspace(0.145, 0.34, 500)
    eta = 2.5e-3
    embedding = continuum.run_spectrum(
        np.asarray([[active_energy]]),
        energies,
        eta=eta,
    )
    times = np.linspace(0.0, 1500.0, 151)
    continuum.run_dynamics(
        np.asarray([[active_energy]]),
        np.asarray([1.0]),
        times,
    )

    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    energy_ev = energies * au2ev
    spectrum = embedding.spectral_density
    axes[0].plot(energy_ev, spectrum / np.max(spectrum), color=colors[0], lw=1.5)
    axes[0].axvline(active_energy * au2ev, color="0.35", lw=1.0, ls="--")
    axes[0].set(xlabel="Energy (eV)", ylabel="Normalized active spectrum")

    gamma_mev = embedding.hybridization_trace * au2ev * 1.0e3
    axes[1].plot(energy_ev, gamma_mev, color=colors[1], lw=1.5)
    axes[1].set(xlabel="Energy (eV)", ylabel=r"$\Gamma(E)$ (meV)")

    time_fs = times * au2fs
    axes[2].plot(
        time_fs,
        continuum.active_populations[:, 0],
        color=colors[0],
        lw=1.5,
        label="Bound exciton",
    )
    axes[2].plot(
        time_fs,
        continuum.continuum_population,
        color=colors[2],
        lw=1.5,
        ls="--",
        label="Continuum",
    )
    axes[2].set(xlabel="Time (fs)", ylabel="Population", ylim=(-0.02, 1.02))
    axes[2].legend(frameon=False, fontsize=8)
    for label, axis in zip("abc", axes):
        axis.text(
            0.02,
            0.97,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
        axis.grid(alpha=0.18, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.savefig(output, dpi=320)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    summary = {
        "figure": str(output),
        "pdf": str(pdf),
        "maximum_continuum_population": float(
            np.max(continuum.continuum_population)
        ),
        "maximum_norm_error": float(np.max(np.abs(continuum.total_norm - 1.0))),
        "channels": len(channels),
        "temperature_kelvin": temperature,
        "bose_occupations": [float(channel.occupation) for channel in channels],
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/private/tmp/pbc_exciton_phonon_embedding.png",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
