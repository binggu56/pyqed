#!/usr/bin/env python3
"""Plot the N=11 vector excited-DMRG bond convergence benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_convergence"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for bond_dim in (64, 96, 128):
        path = (
            HERE
            / f"results/channel_targeted_vector_excited_dmrg_n11_d{bond_dim}"
            / "n11_vector_excited_dmrg.json"
        )
        row = json.loads(path.read_text())
        records.append(
            {
                "bond_dim": bond_dim,
                "vector_mass": row["vector_mass"],
                "vector_excitation": row["vector_excitation"],
                "vector_strength": row["vector_strengths"][row["active_root"]],
                "wall_seconds": row["wall_seconds"],
                "tdvp_vector_mass": row.get(
                    "tdvp_same_bond_vector_mass", row.get("tdvp_d128_vector_mass")
                ),
            }
        )

    bond = np.array([row["bond_dim"] for row in records], dtype=float)
    mass = np.array([row["vector_mass"] for row in records])
    tdvp_mass = np.array([row["tdvp_vector_mass"] for row in records])
    dmrg_time = np.array([row["wall_seconds"] for row in records])
    tdvp_time = np.array([293.793, 1314.43, 3596.83])
    continuum = 1.0 / np.sqrt(np.pi)
    extrapolated_mass, slope = np.polyfit(1.0 / bond, mass, 1)[::-1]

    payload = {
        "description": "N=11 vector-channel excited-DMRG bond convergence",
        "records": records,
        "continuum_vector_mass": continuum,
        "linear_inverse_bond_extrapolation": {
            "model": "M(D) = M_inf + a/D",
            "M_inf": float(extrapolated_mass),
            "a": float(slope),
        },
    }
    data_path = OUTPUT / "n11_vector_excited_dmrg_convergence.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    axes[0].plot(bond, tdvp_mass, "o--", label="TDVP pole")
    axes[0].plot(bond, mass, "s-", label="excited DMRG")
    axes[0].axhline(continuum, color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[0].set(xlabel="bond dimension $D$", ylabel=r"$M_V/g$")
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        bond, abs(tdvp_mass - continuum), "o--", label="TDVP pole"
    )
    axes[1].semilogy(
        bond, abs(mass - continuum), "s-", label="excited DMRG"
    )
    axes[1].set(
        xlabel="bond dimension $D$",
        ylabel=r"$|M_V/g-1/\sqrt{\pi}|$",
    )
    axes[1].legend(frameon=False)

    axes[2].plot(dmrg_time, abs(mass - continuum), "s-", label="excited DMRG")
    axes[2].plot(tdvp_time, abs(tdvp_mass - continuum), "o--", label="TDVP pole")
    axes[2].set(
        xlabel="wall time (s)",
        ylabel=r"$|M_V/g-1/\sqrt{\pi}|$",
        xscale="log",
        yscale="log",
    )
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(bond) if axis is not axes[2] else None
        style(axis)
    figure_path = OUTPUT / "20_n11_vector_excited_dmrg_convergence.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
