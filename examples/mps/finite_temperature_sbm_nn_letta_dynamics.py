#!/usr/bin/env python3
"""NN-LETTA dynamics for the two-arm thermofield spin-boson chain."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.letta import (
    LETTAEvolution,
    nearest_neighbor_hamiltonian,
    site_reduced_density_matrix,
    window2_hamiltonian_from_mpo,
    window2_product_state,
)
from pyqed.models.impurity.spin_boson import (
    thermofield_spin_boson_bond_hamiltonians,
    thermofield_spin_boson_interleaved_mpo,
    thermofield_spin_boson_interleaved_product_factors,
    thermofield_spin_boson_product_factors,
    thermofield_spin_boson_wilson_chains,
)
from pyqed.narg.spin_boson import local_boson_operators


def _ranks(state):
    if hasattr(state, "ranks"):
        return tuple(int(value) for value in state.ranks)
    return tuple(int(tensor.shape[-1]) for tensor in state.tensors[:-1])


def _parameters(state):
    tensors = state.cores if hasattr(state, "cores") else state.tensors
    return int(sum(int(tensor.numel()) if hasattr(tensor, "numel") else tensor.size for tensor in tensors))


def _observe(state, spin_site):
    rho, info = site_reduced_density_matrix(state, spin_site, return_info=True)
    return {
        "sigma_z": float(np.real(rho[0, 0] - rho[1, 1])),
        "coherence": float(abs(rho[0, 1])),
        "norm": float(math.exp(0.5 * info["log_norm"])),
        "hermiticity_error": float(info["hermiticity_error"]),
        "minimum_eigenvalue": float(info["minimum_eigenvalue"]),
        "max_rank": max(_ranks(state), default=1),
    }


def _problem(args):
    positive, negative, _ = thermofield_spin_boson_wilson_chains(
        args.nmodes,
        temperature=args.temperature,
        alpha=args.alpha,
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    operators = local_boson_operators(args.local_dim, basis="fock")
    if args.ordering == "arms":
        bonds, dims = thermofield_spin_boson_bond_hamiltonians(
            positive, negative, *operators
        )
        factors = thermofield_spin_boson_product_factors(
            positive, negative, operators[-1], spin_state=args.spin_state
        )
        return nearest_neighbor_hamiltonian(bonds, dims), factors, args.nmodes

    mpo, _dims = thermofield_spin_boson_interleaved_mpo(
        positive, negative, *operators
    )
    factors = thermofield_spin_boson_interleaved_product_factors(
        positive, negative, operators[-1], spin_state=args.spin_state
    )
    return window2_hamiltonian_from_mpo(mpo), factors, 0


def _run(operator, factors, spin_site, rank, times, args):
    state = window2_product_state(factors, max_bond=rank)
    driver = LETTAEvolution(
        operator,
        max_bond=rank,
        cutoff=args.cutoff,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        saturation_steps=args.saturation_steps,
        force_switch_time=args.force_switch_time,
        backend=args.backend,
        torch_num_threads=args.torch_threads,
        channel_mode=args.channel_mode,
    )
    observations = [_observe(state, spin_site)]
    step_seconds = [0.0]
    truncation = [0.0]
    krylov_residual = [0.0]
    modes = ["tdvp2"]
    started = time.perf_counter()
    for step in range(1, len(times)):
        tic = time.perf_counter()
        state, info = driver.step(
            state, times[step] - times[step - 1], normalize=False
        )
        step_seconds.append(time.perf_counter() - tic)
        observations.append(_observe(state, spin_site))
        truncation.append(float(info.get("truncation_error", 0.0)))
        krylov_residual.append(float(info.get("krylov_residual_max", 0.0)))
        modes.append(str(info["mode_used"]))
        if step % args.report_every == 0 or step == len(times) - 1:
            print(
                f"{args.ordering} LETTA D={rank}: step {step}/{len(times)-1}, "
                f"t={times[step]:.3f}, mode={modes[-1]}, "
                f"rank={observations[-1]['max_rank']}, "
                f"seconds={step_seconds[-1]:.3f}",
                flush=True,
            )
    return {
        "rank": int(rank),
        "backend": driver.backend,
        "sigma_z": np.asarray([value["sigma_z"] for value in observations]),
        "coherence": np.asarray([value["coherence"] for value in observations]),
        "norm": np.asarray([value["norm"] for value in observations]),
        "hermiticity_error": np.asarray(
            [value["hermiticity_error"] for value in observations]
        ),
        "minimum_eigenvalue": np.asarray(
            [value["minimum_eigenvalue"] for value in observations]
        ),
        "max_rank": np.asarray([value["max_rank"] for value in observations]),
        "step_seconds": np.asarray(step_seconds),
        "truncation": np.asarray(truncation),
        "krylov_residual": np.asarray(krylov_residual),
        "modes": np.asarray(modes),
        "parameters": _parameters(state),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _load_reference(path, times):
    data = np.load(path)
    reference_times = np.asarray(data["time"])
    if times[-1] > reference_times[-1] + 1.0e-12:
        raise ValueError("reference trajectory is shorter than the requested run")
    indices = np.searchsorted(reference_times, times)
    if not np.allclose(reference_times[indices], times):
        raise ValueError("reference and LETTA time grids are incompatible")
    return {
        "sigma_z": np.asarray(data["case0_sigma_z"])[indices],
        "coherence": np.asarray(data["case0_coherence"])[indices],
    }


def _figure(times, reference, runs, output, args):
    colors = plt.cm.plasma(np.linspace(0.12, 0.78, len(runs)))
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 8.1), sharex=True)
    axes[0].plot(times, reference["sigma_z"], color="black", lw=2.0, label="MPS $D=32$")
    for color, run in zip(colors, runs):
        label = rf"LETTA $D={run['rank']}$ ({run['backend']})"
        axes[0].plot(times, run["sigma_z"], color=color, lw=1.6, label=label)
        axes[1].semilogy(
            times,
            np.maximum(np.abs(run["sigma_z"] - reference["sigma_z"]), 1.0e-15),
            color=color,
            lw=1.5,
            label=label,
        )
        axes[2].semilogy(
            times,
            np.maximum(np.abs(run["norm"] - 1.0), 1.0e-16),
            color=color,
            lw=1.5,
            label=rf"norm, $D={run['rank']}$",
        )
    axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[1].set_ylabel("absolute error vs MPS")
    axes[2].set_ylabel("norm error")
    axes[2].set_xlabel(r"time $t\,\omega_c$")
    axes[0].set_title(
        rf"High-$T$ thermofield SBM: {args.ordering} ordering, "
        rf"$N={args.nmodes}$, $d={args.local_dim}$"
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmodes", type=int, default=14)
    parser.add_argument("--local-dim", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--Lambda", type=float, default=2.0)
    parser.add_argument("--s", type=float, default=0.8)
    parser.add_argument("--omegac", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--spin-state", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ordering", choices=("arms", "interleaved"), default="arms")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--channel-mode", default="auto")
    parser.add_argument("--cutoff", type=float, default=1.0e-11)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--saturation-steps", type=int, default=4)
    parser.add_argument(
        "--force-switch-time",
        type=float,
        default=None,
        help="force the hybrid driver from two-site to one-site TDVP at this time",
    )
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    times = np.arange(int(round(args.tmax / args.dt)) + 1) * args.dt
    operator, factors, spin_site = _problem(args)
    reference = _load_reference(args.reference, times)
    runs = [_run(operator, factors, spin_site, rank, times, args) for rank in args.ranks]

    arrays = {
        "time": times,
        "reference_sigma_z": reference["sigma_z"],
        "reference_coherence": reference["coherence"],
    }
    records = []
    for index, run in enumerate(runs):
        record = {
            "rank": run["rank"],
            "backend": run["backend"],
            "parameters": run["parameters"],
            "wall_seconds": run["wall_seconds"],
            "mean_step_seconds": float(np.mean(run["step_seconds"][1:])),
            "max_sigma_z_error_vs_mps": float(
                np.max(np.abs(run["sigma_z"] - reference["sigma_z"]))
            ),
            "max_coherence_error_vs_mps": float(
                np.max(np.abs(run["coherence"] - reference["coherence"]))
            ),
            "max_norm_error": float(np.max(np.abs(run["norm"] - 1.0))),
            "max_krylov_residual": float(np.max(run["krylov_residual"])),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        for key, value in run.items():
            if isinstance(value, np.ndarray):
                arrays[f"case{index}_{key}"] = value
    summary = {
        "model": {
            "temperature": args.temperature,
            "ordering": args.ordering,
            "nmodes_per_branch": args.nmodes,
            "purified_sites": 2 * args.nmodes + 1,
            "local_dim": args.local_dim,
            "alpha": args.alpha,
            "Lambda": args.Lambda,
            "s": args.s,
            "delta": args.delta,
            "dt": args.dt,
            "tmax": args.tmax,
        },
        "runs": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    _figure(times, reference, runs, args.output / "nn_letta_vs_mps.png", args)


if __name__ == "__main__":
    main()
