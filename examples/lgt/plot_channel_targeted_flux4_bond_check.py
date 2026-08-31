#!/usr/bin/env python3
"""Plot the N=11, flux-4 vector-mass bond-dimension check."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_flux4_bond_check"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for bond_dim in (64, 96, 128):
        path = (
            HERE
            / f"results/channel_targeted_vector_excited_dmrg_n11_flux4_d{bond_dim}"
            / "n11_vector_excited_dmrg.json"
        )
        row = json.loads(path.read_text())
        rows.append(
            {
                "bond_dim": bond_dim,
                "vector_excitation": row["vector_excitation"],
                "vector_mass": row["vector_mass"],
                "vector_strength": row["vector_strengths"][row["active_root"]],
                "wall_seconds": row["wall_seconds"],
            }
        )
    bond = np.array([row["bond_dim"] for row in rows])
    excitation = np.array([row["vector_excitation"] for row in rows])
    mass = np.array([row["vector_mass"] for row in rows])
    continuum = 1.0 / np.sqrt(np.pi)
    continuum_excitation = np.sqrt(continuum**2 + (2.0 * np.pi / 10.0) ** 2)
    payload = {
        "description": "N=11, flux-cutoff-4 excited-DMRG bond check",
        "records": rows,
        "continuum_vector_mass": float(continuum),
        "bond_64_to_96_mass_change": float(mass[1] - mass[0]),
        "bond_64_to_96_relative_change": float((mass[1] - mass[0]) / mass[0]),
        "bond_96_to_128_mass_change": float(mass[2] - mass[1]),
        "bond_96_to_128_relative_change": float((mass[2] - mass[1]) / mass[1]),
    }
    data_path = OUTPUT / "n11_flux4_bond_check.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    axes[0].plot(bond, excitation, "o-")
    axes[0].axhline(continuum_excitation, color="C3", ls=":", label="continuum")
    axes[0].set(xlabel="bond dimension $D$", ylabel=r"$\omega_V/g$")
    axes[0].legend(frameon=False)
    axes[1].plot(bond, mass, "s-")
    axes[1].axhline(continuum, color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[1].set(xlabel="bond dimension $D$", ylabel=r"$M_V/g$")
    axes[1].legend(frameon=False)
    axes[2].bar(
        [str(value) for value in bond],
        [row["vector_strength"] for row in rows],
    )
    axes[2].set(
        xlabel="bond dimension $D$",
        ylabel=r"$|\langle V|O_V|0\rangle|^2$",
    )
    for axis in axes[:2]:
        axis.set_xticks(bond)
    for axis in axes:
        style(axis)
    figure_path = OUTPUT / "24_n11_flux4_bond_check.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
