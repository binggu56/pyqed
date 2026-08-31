#!/usr/bin/env python3
"""Extract the N=11 Schwinger vector mass with three-root DMRG."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import channel_source, style
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
GROUND_CHECKPOINT_CUTOFF1 = (
    HERE / "results/channel_targeted_mv_ms_mps_n11_d96/ground_state_checkpoint.pkl"
)
GROUND_CHECKPOINT_CUTOFF2 = (
    HERE
    / "results/channel_targeted_mv_ms_mps_n11_flux2_d64"
    / "ground_state_checkpoint.pkl"
)
TDVP_DATA = (
    HERE
    / "results/channel_targeted_mv_ms_mps_n11_bond_convergence"
    / "n11_vector_bond_convergence.json"
)


def plot_result(
    energies,
    strengths,
    vector_mass,
    tdvp_mass,
    seconds,
    bond_dim,
    flux_cutoff,
    output,
):
    roots = np.arange(len(energies))
    gaps = np.asarray(energies) - np.min(energies)
    continuum = 1.0 / np.sqrt(np.pi)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    axes[0].plot(roots, gaps, "o-")
    axes[0].set(xlabel="DMRG root", ylabel=r"$(E_i-E_0)/g$")
    axes[0].set_xticks(roots)

    axes[1].bar(roots, strengths)
    axes[1].set(xlabel="DMRG root", ylabel=r"$|\langle i|O_V|0\rangle|^2$")
    axes[1].set_xticks(roots)

    labels = [f"excited DMRG\n$D={bond_dim}$", "continuum"]
    values = [vector_mass, continuum]
    colors = ["C1", "C3"]
    if tdvp_mass is not None:
        labels.insert(0, f"TDVP\n$D={bond_dim}$")
        values.insert(0, tdvp_mass)
        colors.insert(0, "C0")
    axes[2].bar(labels, values, color=colors)
    axes[2].set(
        ylabel=r"$M_V/g$",
        title=rf"$\ell_{{\max}}={flux_cutoff}$: {seconds:.0f} s",
    )
    for axis in axes:
        style(axis)
    path = output / (
        "19_n11_vector_excited_dmrg.png"
        if flux_cutoff == 1
        else f"21_n11_vector_excited_dmrg_flux{flux_cutoff}.png"
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(
    *,
    bond_dim=128,
    half_sweeps=12,
    flux_cutoff=1,
    resume_from_active_root=False,
    performance="symmetric",
):
    flux_cutoff = int(flux_cutoff)
    if flux_cutoff == 1:
        output = (
            HERE / f"results/channel_targeted_vector_excited_dmrg_n11_d{int(bond_dim)}"
        )
        ground_checkpoint = GROUND_CHECKPOINT_CUTOFF1
    else:
        output = HERE / (
            f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}"
            f"_d{int(bond_dim)}"
        )
        if flux_cutoff == 2:
            ground_checkpoint = GROUND_CHECKPOINT_CUTOFF2
        else:
            if flux_cutoff >= 5:
                lifted_checkpoint = output / "ground_state_checkpoint_lifted.pkl"
            else:
                lifted_checkpoint = Path("__missing_lifted_checkpoint__")
            if lifted_checkpoint.exists():
                ground_checkpoint = lifted_checkpoint
            else:
                ground_directory = HERE / (
                    f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}_d64"
                )
                repaired_checkpoint = (
                    ground_directory / "ground_state_checkpoint_repaired.pkl"
                )
                ground_checkpoint = (
                    repaired_checkpoint
                    if repaired_checkpoint.exists()
                    else ground_directory / "ground_state_checkpoint.pkl"
                )
    output.mkdir(parents=True, exist_ok=True)
    parameters = {
        "npts": 11,
        "length": 10.0,
        "coupling": 1.0,
        "mass": 0.0,
        "flux_cutoff": flux_cutoff,
    }
    bond_dim = int(bond_dim)
    half_sweeps = int(half_sweeps)
    builder = AlternatingWilsonDVRMPO(**parameters)
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
    if ground_checkpoint.exists():
        saved = DMRG.load_checkpoint(ground_checkpoint)
        if not saved.get("final"):
            raise ValueError("the N=11 ground-state checkpoint is incomplete")
        vacuum = MPS(
            saved["mps"],
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        )
    else:
        initial = builder.gauss_seed_mps(
            bond_dim=128,
            seed=7,
            native_site_storage=True,
        )
        print(
            f"[ground DMRG] generating flux cutoff {flux_cutoff} checkpoint",
            flush=True,
        )
        ground = DMRG(
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
            performance="symmetric",
            checkpoint_path=ground_checkpoint,
        ).run()
        vacuum = ground.ground_state
    source_raw = channel_source(
        builder.build_vector_mpo(),
        vacuum,
        maps,
        bond_dim=bond_dim,
    )
    source = MPS(
        source_raw.factors,
        labels=["lv", "rv", "p"],
        sites=hamiltonian.input_sites,
    )
    initial_guess = source
    if resume_from_active_root:
        state_path = output / "n11_vector_excited_dmrg_states.pkl"
        with state_path.open("rb") as handle:
            previous_states = pickle.load(handle)
        active_guess = int(np.argmax(previous_states["strengths"]))
        initial_guess = MPS(
            previous_states["states"][active_guess],
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        )
        print(
            f"[excited DMRG D={bond_dim}] continuing from saved root "
            f"{active_guess}",
            flush=True,
        )

    def progress(**info):
        energy = np.asarray(info.get("energy", []), dtype=float).reshape(-1)
        formatted = ", ".join(f"{value:.9f}" for value in energy)
        print(
            f"[excited DMRG D={bond_dim}] half-sweep "
            f"{int(info.get('sweep', -1)) + 1}/{half_sweeps} "
            f"({info.get('direction')}): {formatted}",
            flush=True,
        )

    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=bond_dim,
        init_guess=initial_guess,
        nsweeps=half_sweeps,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        nstates=3,
        weights=[0.2, 0.2, 0.6],
        not_conv_err=False,
        sweep_tol=1.0e-9,
        davidson_tol=1.0e-10,
        davidson_max_iter=300,
        noise=1.0e-6,
        performance=performance,
        sweep_callback=progress,
    ).run()
    seconds = perf_counter() - started
    energies = np.asarray(solver.e_tot, dtype=float)
    strengths = np.asarray(
        [abs(TDMPS.state_overlap(source, state)) ** 2 for state in solver.states]
    )
    vacuum_dimension = 2
    active = int(vacuum_dimension + np.argmax(strengths[vacuum_dimension:]))
    excitation = float(energies[active] - np.min(energies[:vacuum_dimension]))
    momentum = 2.0 * np.pi / parameters["length"]
    vector_mass = float(np.sqrt(max(excitation**2 - momentum**2, 0.0)))
    if flux_cutoff == 1:
        tdvp = json.loads(TDVP_DATA.read_text())
        tdvp_mass = float(
            next(
                row
                for row in tdvp["records"]
                if row["dynamics_bond_dim"] == bond_dim
            )["vector_mass"]
        )
    elif flux_cutoff == 2:
        comparison = json.loads(
            (
                HERE
                / "results/channel_targeted_mv_ms_mps_n11_flux2_d64"
                / "n11_vector_flux_cutoff_comparison.json"
            ).read_text()
        )
        tdvp_mass = float(comparison["cutoff2"]["vector_mass"])
    else:
        tdvp_mass = None
    payload = {
        "description": "Three-root vector-weighted Gauss-symmetric excited-state DMRG",
        "parameters": {
            **parameters,
            "bond_dim": bond_dim,
            "half_sweeps": half_sweeps,
            "nstates": 3,
            "weights": [0.2, 0.2, 0.6],
            "resume_from_active_root": bool(resume_from_active_root),
            "performance": str(performance),
        },
        "energies": energies.tolist(),
        "vector_strengths": strengths.tolist(),
        "active_root": active,
        "vector_excitation": excitation,
        "vector_mass": vector_mass,
        "continuum_vector_mass": float(1.0 / np.sqrt(np.pi)),
        "tdvp_same_bond_vector_mass": tdvp_mass,
        "wall_seconds": float(seconds),
        "converged": bool(solver.converged),
    }
    data_path = output / "n11_vector_excited_dmrg.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "n11_vector_excited_dmrg_states.pkl").open("wb") as handle:
        pickle.dump(
            {
                "energies": energies,
                "states": [state.factors for state in solver.states],
                "strengths": strengths,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    figure = plot_result(
        energies,
        strengths,
        vector_mass,
        tdvp_mass,
        seconds,
        bond_dim,
        flux_cutoff,
        output,
    )
    print(json.dumps(payload, indent=2))
    print(data_path)
    print(figure)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dim", type=int, default=128, choices=(64, 96, 128))
    parser.add_argument("--half-sweeps", type=int, default=12)
    parser.add_argument(
        "--flux-cutoff", type=int, default=1, choices=(1, 2, 3, 4, 5)
    )
    parser.add_argument("--resume-from-active-root", action="store_true")
    parser.add_argument(
        "--performance",
        default="symmetric",
        choices=("symmetric", "packed-compiled-fast", "reference"),
    )
    args = parser.parse_args()
    run(
        bond_dim=args.bond_dim,
        half_sweeps=args.half_sweeps,
        flux_cutoff=args.flux_cutoff,
        resume_from_active_root=args.resume_from_active_root,
        performance=args.performance,
    )
