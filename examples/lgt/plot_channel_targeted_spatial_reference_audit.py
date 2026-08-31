#!/usr/bin/env python3
"""Audit N=11/13 vector masses against two ground-energy references."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style
from pyqed.mps import DMRG


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_spatial_audit"
CASES = (
    (
        11,
        HERE
        / "results/channel_targeted_vector_excited_dmrg_n11_flux5_d128"
        / "n11_vector_excited_dmrg.json",
        HERE
        / "results/channel_targeted_vector_excited_dmrg_n11_flux5_d128"
        / "ground_state_checkpoint_lifted.pkl",
    ),
    (
        13,
        HERE
        / "results/channel_targeted_vector_excited_dmrg_n13_flux4_d128"
        / "n13_vector_excited_dmrg.json",
        HERE
        / "results/channel_targeted_vector_excited_dmrg_n13_flux4_d128"
        / "ground_cutoff4_checkpoint.pkl",
    ),
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    momentum = 2.0 * np.pi / 10.0
    records = []
    for npts, result_path, ground_path in CASES:
        result = json.loads(result_path.read_text())
        ground_energy = float(DMRG.load_checkpoint(ground_path)["energy"])
        state_average_ground = float(min(result["energies"][:2]))
        vector_energy = float(result["energies"][result["active_root"]])
        separate_excitation = vector_energy - ground_energy
        separate_mass = float(
            np.sqrt(max(separate_excitation**2 - momentum**2, 0.0))
        )
        records.append(
            {
                "npts": npts,
                "spacing_length_over_n": 10.0 / npts,
                "ground_energy_single_state": ground_energy,
                "ground_energy_state_average": state_average_ground,
                "ground_reference_mismatch": state_average_ground - ground_energy,
                "vector_energy": vector_energy,
                "vector_strength": result["vector_strengths"][result["active_root"]],
                "mass_state_average_reference": result["vector_mass"],
                "mass_single_state_ground_reference": separate_mass,
            }
        )
    payload = {
        "description": "N=11/13 vector-mass audit for state-averaged versus independently optimized ground references",
        "records": records,
        "continuum_vector_mass": float(1.0 / np.sqrt(np.pi)),
    }
    data_path = OUTPUT / "n11_n13_spatial_reference_audit.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    npts = np.array([row["npts"] for row in records])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    axes[0].plot(
        npts,
        [row["ground_reference_mismatch"] for row in records],
        "o-",
    )
    axes[0].set(
        xlabel="DVR points $N$",
        ylabel=r"$(E_0^{\rm SA}-E_0^{\rm GS})/g$",
    )
    axes[1].plot(
        npts,
        [row["mass_state_average_reference"] for row in records],
        "s-",
        label="state-averaged reference",
    )
    axes[1].plot(
        npts,
        [row["mass_single_state_ground_reference"] for row in records],
        "o--",
        label="single-state ground",
    )
    axes[1].axhline(1.0 / np.sqrt(np.pi), color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[1].set(xlabel="DVR points $N$", ylabel=r"$M_V/g$")
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].bar(
        [str(value) for value in npts],
        [row["vector_strength"] for row in records],
    )
    axes[2].set(
        xlabel="DVR points $N$",
        ylabel=r"$|\langle V|O_V|0\rangle|^2$",
    )
    for axis in axes[:2]:
        axis.set_xticks(npts)
    for axis in axes:
        style(axis)
    figure_path = OUTPUT / "28_n11_n13_spatial_reference_audit.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
