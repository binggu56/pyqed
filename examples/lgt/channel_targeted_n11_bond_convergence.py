#!/usr/bin/env python3
"""Converge the N=11 vector mass using one checkpointed DMRG vacuum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import (
    channel_source,
    correlation,
    matrix_pencil,
    rank_stable_pole,
    style,
)
from pyqed.lgt import AlternatingWilsonDVRMPO
from pyqed.mps import (
    DMRG,
    MPO,
    MPS,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE
    / "results/channel_targeted_mv_ms_mps_n11_d96/ground_state_checkpoint.pkl"
)
DEFAULT_EXISTING = (
    HERE / "results/channel_targeted_mv_ms_mps_n11_d96"
)
DEFAULT_OUTPUT = (
    HERE / "results/channel_targeted_mv_ms_mps_n11_bond_convergence"
)


def analyze(values, dt, length):
    _frequency, _roots, singular = matrix_pencil(values, dt, rank=12)
    momentum = 2.0 * np.pi / length
    excitation, spread, audit = rank_stable_pole(
        values,
        dt,
        minimum=momentum,
    )
    mass = np.sqrt(max(excitation**2 - momentum**2, 0.0))
    return {
        "vector_excitation": float(excitation),
        "vector_mass": float(mass),
        "vector_pole_rank_mad": float(spread),
        "singular_values": singular[:24].tolist(),
        "pole_rank_audit": audit,
    }


def plot(records, output, length):
    records = sorted(records, key=lambda row: row["dynamics_bond_dim"])
    bond = np.asarray([row["dynamics_bond_dim"] for row in records])
    mass = np.asarray([row["vector_mass"] for row in records])
    seconds = np.asarray([row["wall_seconds"] for row in records])
    exact_mass = 1.0 / np.sqrt(np.pi)
    exact_excitation = np.sqrt(exact_mass**2 + (2.0 * np.pi / length) ** 2)
    excitation = np.asarray([row["vector_excitation"] for row in records])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)
    axes[0].plot(bond, mass, "o-", label="MPS")
    axes[0].axhline(exact_mass, color="C3", ls="--", label="continuum")
    axes[0].set(xlabel=r"$D_{\rm dyn}$", ylabel=r"$M_V/g$")
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        bond,
        np.abs(excitation - exact_excitation),
        "o-",
    )
    axes[1].set(
        xlabel=r"$D_{\rm dyn}$",
        ylabel=r"$|E_V(k)-E_V^{\rm continuum}(k)|/g$",
    )

    axes[2].plot(bond, seconds, "o-")
    axes[2].set(xlabel=r"$D_{\rm dyn}$", ylabel="vector wall time (s)")
    for axis in axes:
        axis.set_xticks(bond)
        style(axis)
    path = output / "17_n11_vector_bond_convergence.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(output, checkpoint, existing, bond_dims):
    output.mkdir(parents=True, exist_ok=True)
    length = 10.0
    dt = 0.1
    steps = 80
    builder = AlternatingWilsonDVRMPO(
        npts=11,
        length=length,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=1,
    )
    maps, target, manager = builder.gauss_symmetry()
    sectors = [
        [site_map[state] for state in sorted(site_map)] for site_map in maps
    ]
    raw_hamiltonian = builder.build_mpo()
    hamiltonian = compress_symmetric_mpo(
        MPO(
            dense_to_symmetric_mpo(
                raw_hamiltonian.factors,
                maps,
                native_site_storage=True,
            )
        )
    )
    saved = DMRG.load_checkpoint(checkpoint)
    if not saved.get("final"):
        raise ValueError("the supplied ground-state checkpoint is incomplete")
    vacuum = MPS(saved["mps"], labels=["lv", "rv", "p"])
    ground_energy = float(saved["energy"])
    vector_operator = builder.build_vector_mpo()

    data_path = output / "n11_vector_bond_convergence.json"
    prior = {}
    if data_path.exists():
        prior = {
            row["dynamics_bond_dim"]: row
            for row in json.loads(data_path.read_text()).get("records", [])
        }
    records = []
    requested = sorted(set(map(int, bond_dims)) | {96})
    for bond_dim in requested:
        if bond_dim == 96:
            with np.load(existing / "channel_targeted_correlations.npz") as data:
                values = np.asarray(data["vector"])
            old = json.loads(
                (existing / "channel_targeted_mv_ms_data.json").read_text()
            )
            wall_seconds = float(old["timing_seconds"]["vector_correlation"])
            source = "existing N=11 production run"
        elif (output / f"vector_d{bond_dim}.npz").exists():
            with np.load(output / f"vector_d{bond_dim}.npz") as data:
                values = np.asarray(data["vector"])
            wall_seconds = float(prior[bond_dim]["wall_seconds"])
            source = "saved checkpoint-reused calculation"
        else:
            vector_source = channel_source(
                vector_operator,
                vacuum,
                maps,
                bond_dim=bond_dim,
            )
            values, wall_seconds = correlation(
                hamiltonian,
                vector_source,
                sectors,
                target,
                ground_energy,
                dt=dt,
                steps=steps,
                bond_dim=bond_dim,
                label=f"vector D={bond_dim}",
                progress_interval=1,
            )
            np.savez(
                output / f"vector_d{bond_dim}.npz",
                dt=dt,
                vector=values,
            )
            source = "checkpoint-reused calculation"
        record = {
            "dynamics_bond_dim": bond_dim,
            "wall_seconds": float(wall_seconds),
            "source": source,
            **analyze(values, dt, length),
        }
        records.append(record)
        payload = {
            "description": "N=11 vector-channel TDVP bond-dimension convergence",
            "parameters": {
                "npts": 11,
                "length": length,
                "flux_cutoff": 1,
                "ground_bond_dim": 128,
                "dt": dt,
                "steps": steps,
            },
            "ground_checkpoint": str(checkpoint),
            "ground_energy": ground_energy,
            "records": sorted(records, key=lambda row: row["dynamics_bond_dim"]),
        }
        data_path.write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    figure = plot(records, output, length)
    print(json.dumps(payload, indent=2))
    print(figure)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--bond-dims", type=int, nargs="+", default=(64, 128))
    args = parser.parse_args()
    run(args.output, args.checkpoint, args.existing, args.bond_dims)


if __name__ == "__main__":
    main()
