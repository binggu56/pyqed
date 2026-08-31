#!/usr/bin/env python3
"""Compare the N=11, D=64 vector mass at two electric-flux cutoffs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_flux_convergence"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [
        HERE
        / "results/channel_targeted_vector_excited_dmrg_n11_d64"
        / "n11_vector_excited_dmrg.json",
    ]
    paths.extend(
        HERE
        / f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}_d64"
        / "n11_vector_excited_dmrg.json"
        for flux_cutoff in (2, 3, 4)
        if (
            HERE
            / f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}_d64"
            / "n11_vector_excited_dmrg.json"
        ).exists()
    )
    rows = [json.loads(path.read_text()) for path in paths]
    cutoff = np.array([row["parameters"]["flux_cutoff"] for row in rows])
    mass = np.array([row["vector_mass"] for row in rows])
    excitation = np.array([row["vector_excitation"] for row in rows])
    strength = np.array(
        [row["vector_strengths"][row["active_root"]] for row in rows]
    )
    continuum = 1.0 / np.sqrt(np.pi)
    continuum_excitation = np.sqrt(continuum**2 + (2.0 * np.pi / 10.0) ** 2)
    payload = {
        "description": "Matched N=11, D=64 excited-DMRG vector flux-cutoff comparison",
        "records": [
            {
                "flux_cutoff": int(c),
                "vector_excitation": float(e),
                "vector_mass": float(m),
                "vector_strength": float(s),
                "wall_seconds": float(row["wall_seconds"]),
            }
            for c, e, m, s, row in zip(cutoff, excitation, mass, strength, rows)
        ],
        "continuum_vector_mass": float(continuum),
        "continuum_finite_momentum_excitation": float(continuum_excitation),
        "successive_mass_changes": [
            {
                "from_flux_cutoff": int(cutoff[index - 1]),
                "to_flux_cutoff": int(cutoff[index]),
                "mass_change": float(mass[index] - mass[index - 1]),
                "relative_change": float(
                    (mass[index] - mass[index - 1]) / mass[index - 1]
                ),
            }
            for index in range(1, len(cutoff))
        ],
    }
    data_path = OUTPUT / "n11_vector_excited_dmrg_flux_convergence.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    axes[0].plot(cutoff, excitation, "o-")
    axes[0].axhline(continuum_excitation, color="C3", ls=":", label="continuum")
    axes[0].set(xlabel=r"flux cutoff $\ell_{\max}$", ylabel=r"$\omega_V/g$")
    axes[0].legend(frameon=False)

    axes[1].plot(cutoff, mass, "s-")
    axes[1].axhline(continuum, color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[1].set(xlabel=r"flux cutoff $\ell_{\max}$", ylabel=r"$M_V/g$")
    axes[1].legend(frameon=False)

    axes[2].plot(cutoff, abs(mass - continuum), "d-")
    axes[2].set(
        xlabel=r"flux cutoff $\ell_{\max}$",
        ylabel=r"$|M_V/g-1/\sqrt{\pi}|$",
        yscale="log",
    )
    for axis in axes:
        axis.set_xticks(cutoff)
        style(axis)
    figure_path = OUTPUT / "22_n11_vector_excited_dmrg_flux_convergence.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
