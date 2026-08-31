#!/usr/bin/env python3
"""Run the N=13, flux-4, D=128 vector excited-DMRG benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import channel_source, style
from lift_channel_targeted_flux_ground import lift_factors
from pyqed.lgt import AlternatingWilsonDVRMPO
from pyqed.mps import (
    DMRG,
    MPO,
    MPS,
    TDMPS,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_n13_flux4_d128"
N11_RESULT = (
    HERE
    / "results/channel_targeted_vector_excited_dmrg_n11_flux4_d128"
    / "n11_vector_excited_dmrg.json"
)


def build_system(flux_cutoff):
    builder = AlternatingWilsonDVRMPO(
        npts=13,
        length=10.0,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=flux_cutoff,
    )
    maps, target, manager = builder.gauss_symmetry()
    raw = builder.build_mpo()
    hamiltonian = compress_symmetric_mpo(
        MPO(
            dense_to_symmetric_mpo(
                raw.factors,
                maps,
                native_site_storage=True,
            )
        )
    )
    return builder, maps, target, manager, hamiltonian


def run_ground(
    hamiltonian,
    initial,
    maps,
    target,
    manager,
    checkpoint,
    label,
):
    history = []

    def progress(**info):
        energy = float(np.real(np.asarray(info.get("energy")).reshape(-1)[0]))
        history.append(
            {
                "sweep": int(info.get("sweep", -1)),
                "direction": str(info.get("direction")),
                "energy": energy,
            }
        )
        print(
            f"[{label}] half-sweep {len(history)} "
            f"({info.get('direction')}): {energy:.12f}",
            flush=True,
        )

    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=128,
        init_guess=initial,
        nsweeps=20,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        davidson_max_iter=300,
        noise=1.0e-6,
        performance="packed-compiled-fast",
        checkpoint_path=checkpoint,
        sweep_callback=progress,
    ).run()
    return solver, history, perf_counter() - started


def checkpoint_state(path, sites):
    saved = DMRG.load_checkpoint(path)
    if not saved.get("final"):
        raise ValueError(f"incomplete checkpoint: {path}")
    return MPS(saved["mps"], labels=["lv", "rv", "p"], sites=sites)


def plot_result(payload, output):
    n11 = json.loads(N11_RESULT.read_text())
    roots = np.arange(len(payload["energies"]))
    energies = np.asarray(payload["energies"])
    strengths = np.asarray(payload["vector_strengths"])
    continuum = payload["continuum_vector_mass"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    axes[0].plot(roots, energies - np.min(energies), "o-")
    axes[0].set(xlabel="DMRG root", ylabel=r"$(E_i-E_0)/g$")
    axes[0].set_xticks(roots)
    axes[1].bar(roots, strengths)
    axes[1].set(xlabel="DMRG root", ylabel=r"$|\langle i|O_V|0\rangle|^2$")
    axes[1].set_xticks(roots)
    axes[2].plot(
        [11, 13],
        [n11["vector_mass"], payload["vector_mass"]],
        "s-",
    )
    axes[2].axhline(continuum, color="C3", ls=":", label=r"$1/\sqrt{\pi}$")
    axes[2].set(xlabel="DVR points $N$", ylabel=r"$M_V/g$")
    axes[2].set_xticks([11, 13])
    axes[2].legend(frameon=False)
    for axis in axes:
        style(axis)
    figure_path = output / "27_n11_n13_spatial_convergence.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    return figure_path


def main(*, resume_from_active_root=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cutoff1_checkpoint = OUTPUT / "ground_cutoff1_checkpoint.pkl"
    cutoff4_checkpoint = OUTPUT / "ground_cutoff4_checkpoint.pkl"
    builder4, maps4, target4, manager4, hamiltonian4 = build_system(4)
    if cutoff4_checkpoint.exists():
        ground4 = checkpoint_state(cutoff4_checkpoint, hamiltonian4.input_sites)
        ground1_history = []
        ground1_seconds = 0.0
        ground4_history = []
        ground4_seconds = 0.0
    else:
        builder1, maps1, target1, manager1, hamiltonian1 = build_system(1)
        if cutoff1_checkpoint.exists():
            ground1 = checkpoint_state(cutoff1_checkpoint, hamiltonian1.input_sites)
            ground1_history = []
            ground1_seconds = 0.0
        else:
            initial1 = builder1.gauss_seed_mps(
                bond_dim=128,
                seed=7,
                native_site_storage=True,
            )
            solver1, ground1_history, ground1_seconds = run_ground(
                hamiltonian1,
                initial1,
                maps1,
                target1,
                manager1,
                cutoff1_checkpoint,
                "N=13 cutoff-1 ground",
            )
            ground1 = solver1.ground_state
        cutoff1_factors = [factor.copy() for factor in ground1.factors]
        del ground1, hamiltonian1, builder1, maps1, manager1
        gc.collect()
        initial4 = MPS(
            lift_factors(cutoff1_factors, maps4),
            labels=["lv", "rv", "p"],
            sites=hamiltonian4.input_sites,
        )
        solver4, ground4_history, ground4_seconds = run_ground(
            hamiltonian4,
            initial4,
            maps4,
            target4,
            manager4,
            cutoff4_checkpoint,
            "N=13 cutoff-4 lifted ground",
        )
        ground4 = solver4.ground_state

    source_raw = channel_source(
        builder4.build_vector_mpo(),
        ground4,
        maps4,
        bond_dim=128,
    )
    source = MPS(
        source_raw.factors,
        labels=["lv", "rv", "p"],
        sites=hamiltonian4.input_sites,
    )
    initial_excited = source
    if resume_from_active_root:
        state_path = OUTPUT / "n13_vector_excited_dmrg_states.pkl"
        with state_path.open("rb") as handle:
            previous = pickle.load(handle)
        active_guess = int(np.argmax(previous["strengths"]))
        initial_excited = MPS(
            previous["states"][active_guess],
            labels=["lv", "rv", "p"],
            sites=hamiltonian4.input_sites,
        )
        print(f"[N=13 vector D=128] continuing from root {active_guess}", flush=True)
    excited_history = []

    def excited_progress(**info):
        energies = np.real(np.asarray(info.get("energy", []))).reshape(-1)
        excited_history.append(
            {
                "sweep": int(info.get("sweep", -1)),
                "direction": str(info.get("direction")),
                "energies": energies.tolist(),
            }
        )
        values = ", ".join(f"{value:.9f}" for value in energies)
        print(
            f"[N=13 vector D=128] half-sweep {len(excited_history)} "
            f"({info.get('direction')}): {values}",
            flush=True,
        )

    started = perf_counter()
    excited = DMRG(
        hamiltonian4,
        D=128,
        init_guess=initial_excited,
        nsweeps=12,
        symmetry=True,
        target_qn=target4,
        sym_mgr=manager4,
        site_qn_maps=maps4,
        nstates=3,
        weights=[0.2, 0.2, 0.6],
        not_conv_err=False,
        sweep_tol=1.0e-9,
        davidson_tol=1.0e-10,
        davidson_max_iter=300,
        noise=1.0e-6,
        performance="packed-compiled-fast",
        sweep_callback=excited_progress,
    ).run()
    excited_seconds = perf_counter() - started
    energies = np.asarray(excited.e_tot, dtype=float)
    strengths = np.asarray(
        [abs(TDMPS.state_overlap(source, state)) ** 2 for state in excited.states]
    )
    active = int(2 + np.argmax(strengths[2:]))
    excitation = float(energies[active] - np.min(energies[:2]))
    momentum = 2.0 * np.pi / 10.0
    mass = float(np.sqrt(max(excitation**2 - momentum**2, 0.0)))
    payload = {
        "description": "N=13 flux-4 D=128 vector excited-DMRG spatial benchmark",
        "parameters": {
            "npts": 13,
            "length": 10.0,
            "spacing_length_over_n": 10.0 / 13.0,
            "flux_cutoff": 4,
            "bond_dim": 128,
            "half_sweeps": 12,
            "weights": [0.2, 0.2, 0.6],
            "performance": "packed-compiled-fast",
            "resume_from_active_root": bool(resume_from_active_root),
        },
        "energies": energies.tolist(),
        "vector_strengths": strengths.tolist(),
        "active_root": active,
        "vector_excitation": excitation,
        "vector_mass": mass,
        "continuum_vector_mass": float(1.0 / np.sqrt(np.pi)),
        "ground_cutoff1_history": ground1_history,
        "ground_cutoff4_history": ground4_history,
        "excited_history": excited_history,
        "timing_seconds": {
            "ground_cutoff1": ground1_seconds,
            "ground_cutoff4": ground4_seconds,
            "excited": excited_seconds,
        },
        "converged": bool(excited.converged),
    }
    data_path = OUTPUT / "n13_vector_excited_dmrg.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    with (OUTPUT / "n13_vector_excited_dmrg_states.pkl").open("wb") as handle:
        pickle.dump(
            {
                "energies": energies,
                "states": [state.factors for state in excited.states],
                "strengths": strengths,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    figure_path = plot_result(payload, OUTPUT)
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from-active-root", action="store_true")
    args = parser.parse_args()
    main(resume_from_active_root=args.resume_from_active_root)
