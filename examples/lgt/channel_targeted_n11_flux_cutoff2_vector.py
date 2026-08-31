#!/usr/bin/env python3
"""Compare N=11 vector masses at electric-flux cutoffs one and two."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import (
    channel_source,
    correlation,
    ground_state,
    matrix_pencil,
    plot_ground_convergence,
    rank_stable_pole,
    style,
)
from pyqed.lgt import AlternatingWilsonDVRMPO


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_mv_ms_mps_n11_flux2_d64"
CUTOFF1 = (
    HERE
    / "results/channel_targeted_mv_ms_mps_n11_bond_convergence"
    / "n11_vector_bond_convergence.json"
)


def analyze(values, dt, length):
    _frequency, _roots, singular = matrix_pencil(values, dt, rank=12)
    momentum = 2.0 * np.pi / length
    excitation, spread, audit = rank_stable_pole(
        values,
        dt,
        minimum=momentum,
    )
    return {
        "vector_excitation": float(excitation),
        "vector_mass": float(np.sqrt(max(excitation**2 - momentum**2, 0.0))),
        "vector_pole_rank_mad": float(spread),
        "singular_values": singular[:24].tolist(),
        "pole_rank_audit": audit,
    }


def plot_comparison(cutoff1, cutoff2, output, length):
    cutoffs = np.asarray([1, 2])
    mass = np.asarray([cutoff1["vector_mass"], cutoff2["vector_mass"]])
    excitation = np.asarray(
        [cutoff1["vector_excitation"], cutoff2["vector_excitation"]]
    )
    exact_mass = 1.0 / np.sqrt(np.pi)
    exact_excitation = np.sqrt(exact_mass**2 + (2.0 * np.pi / length) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.1), constrained_layout=True)
    axes[0].plot(cutoffs, mass, "o-")
    axes[0].axhline(exact_mass, color="C3", ls="--", label="continuum")
    axes[0].set(xlabel="electric-flux cutoff", ylabel=r"$M_V/g$")
    axes[0].legend(frameon=False)
    axes[1].plot(cutoffs, np.abs(excitation - exact_excitation), "o-")
    axes[1].set(
        xlabel="electric-flux cutoff",
        ylabel=r"$|E_V(k)-E_V^{\rm continuum}(k)|/g$",
    )
    for axis in axes:
        axis.set_xticks(cutoffs)
        style(axis)
    path = output / "18_n11_vector_flux_cutoff_comparison.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    npts = 11
    length = 10.0
    dt = 0.1
    steps = 80
    ground_bond = 128
    dynamics_bond = 64
    archive_path = OUTPUT / "vector_flux2_d64.npz"
    data_path = OUTPUT / "n11_vector_flux_cutoff_comparison.json"
    if archive_path.exists() and data_path.exists():
        with np.load(archive_path) as archive:
            values = np.asarray(archive["vector"])
        payload = json.loads(data_path.read_text())
        cutoff2 = analyze(values, dt, length)
        previous = json.loads(CUTOFF1.read_text())
        cutoff1 = next(
            row for row in previous["records"] if row["dynamics_bond_dim"] == 64
        )
        payload["cutoff1"] = {
            "vector_excitation": cutoff1["vector_excitation"],
            "vector_mass": cutoff1["vector_mass"],
        }
        payload["cutoff2"] = cutoff2
        data_path.write_text(json.dumps(payload, indent=2) + "\n")
        comparison_figure = plot_comparison(cutoff1, cutoff2, OUTPUT, length)
        print(json.dumps(payload, indent=2))
        print(data_path)
        print(comparison_figure)
        return payload
    builder = AlternatingWilsonDVRMPO(
        npts=npts,
        length=length,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=2,
    )
    maps, target, manager = builder.gauss_symmetry()
    sectors = [
        [site_map[state] for state in sorted(site_map)] for site_map in maps
    ]
    hamiltonian, ground, ground_seconds = ground_state(
        builder,
        maps,
        target,
        manager,
        ground_bond,
        sweeps=20,
        seed=7,
        checkpoint_path=OUTPUT / "ground_state_checkpoint.pkl",
    )
    source = channel_source(
        builder.build_vector_mpo(),
        ground.ground_state,
        maps,
        bond_dim=dynamics_bond,
    )
    values, vector_seconds = correlation(
        hamiltonian,
        source,
        sectors,
        target,
        ground.e_tot,
        dt=dt,
        steps=steps,
        bond_dim=dynamics_bond,
        label="flux cutoff 2, vector D=64",
        progress_interval=1,
    )
    np.savez(archive_path, dt=dt, vector=values)
    cutoff2 = analyze(values, dt, length)
    previous = json.loads(CUTOFF1.read_text())
    cutoff1 = next(
        row for row in previous["records"] if row["dynamics_bond_dim"] == 64
    )
    full_sweeps = [
        float(row["post_truncation_energy"])
        for row in ground.sweep_history
        if row.get("direction") == "rl"
        and row.get("post_truncation_energy") is not None
    ]
    payload = {
        "description": "Matched-D N=11 vector mass versus electric-flux cutoff",
        "parameters": {
            "npts": npts,
            "length": length,
            "ground_bond_dim": ground_bond,
            "dynamics_bond_dim": dynamics_bond,
            "dt": dt,
            "steps": steps,
        },
        "cutoff1": {
            "vector_excitation": cutoff1["vector_excitation"],
            "vector_mass": cutoff1["vector_mass"],
        },
        "cutoff2": cutoff2,
        "cutoff2_ground_energy": float(ground.e_tot),
        "cutoff2_ground_converged": bool(ground.converged),
        "cutoff2_ground_full_sweep_energy": full_sweeps,
        "timing_seconds": {
            "cutoff2_ground": float(ground_seconds),
            "cutoff2_vector": float(vector_seconds),
        },
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    ground_figure = plot_ground_convergence(ground.sweep_history, OUTPUT)
    comparison_figure = plot_comparison(cutoff1, cutoff2, OUTPUT, length)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(ground_figure)
    print(comparison_figure)
    return payload


if __name__ == "__main__":
    run()
