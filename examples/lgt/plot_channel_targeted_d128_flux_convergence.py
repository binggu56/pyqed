#!/usr/bin/env python3
"""Plot the matched N=11, D=128 vector flux-4/5 convergence check."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_d128_flux_convergence"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for flux_cutoff in (4, 5):
        path = (
            HERE
            / f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}_d128"
            / "n11_vector_excited_dmrg.json"
        )
        row = json.loads(path.read_text())
        rows.append(
            {
                "flux_cutoff": flux_cutoff,
                "vector_excitation": row["vector_excitation"],
                "vector_mass": row["vector_mass"],
                "vector_strength": row["vector_strengths"][row["active_root"]],
                "wall_seconds": row["wall_seconds"],
            }
        )
    cutoff = np.array([row["flux_cutoff"] for row in rows])
    excitation = np.array([row["vector_excitation"] for row in rows])
    mass = np.array([row["vector_mass"] for row in rows])
    continuum = 1.0 / np.sqrt(np.pi)
    continuum_excitation = np.sqrt(continuum**2 + (2.0 * np.pi / 10.0) ** 2)
    payload = {
        "description": "Matched N=11, D=128 flux-4/5 excited-DMRG convergence",
        "records": rows,
        "continuum_vector_mass": float(continuum),
        "flux_4_to_5_mass_change": float(mass[1] - mass[0]),
        "flux_4_to_5_relative_change": float((mass[1] - mass[0]) / mass[0]),
    }
    data_path = OUTPUT / "n11_d128_flux_convergence.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    axes[0].plot(cutoff, excitation, "o-")
    axes[0].axhline(continuum_excitation, color="C3", ls=":", label="continuum")
    axes[0].set(xlabel=r"flux cutoff $\ell_{\max}$", ylabel=r"$\omega_V/g$")
    axes[0].legend(frameon=False)
    axes[1].plot(cutoff, mass, "s-")
    axes[1].axhline(continuum, color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[1].set(xlabel=r"flux cutoff $\ell_{\max}$", ylabel=r"$M_V/g$")
    axes[1].legend(frameon=False)
    axes[2].bar(
        [str(value) for value in cutoff],
        [row["vector_strength"] for row in rows],
    )
    axes[2].set(
        xlabel=r"flux cutoff $\ell_{\max}$",
        ylabel=r"$|\langle V|O_V|0\rangle|^2$",
    )
    for axis in axes[:2]:
        axis.set_xticks(cutoff)
    for axis in axes:
        style(axis)
    figure_path = OUTPUT / "26_n11_d128_flux45_convergence.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
