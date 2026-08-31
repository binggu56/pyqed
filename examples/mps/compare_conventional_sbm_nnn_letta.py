#!/usr/bin/env python3
"""Compare real-time NNN-LETTA with an MPS reference for a Wilson-chain SBM."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.letta import (
    NNNLETTATDVPEngine,
    nnn_product_state,
    nnn_system_reduced_density_matrix,
)
from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_wilson_chain,
    spin_boson_bond_hamiltonians,
    spin_boson_product_factors,
)
from pyqed.mps.mpo import nearest_neighbor_mpo
from pyqed.narg.spin_boson import local_boson_operators


def build_problem(args):
    chain = log_discretized_spin_boson_wilson_chain(
        args.nmodes,
        alpha=args.alpha,
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    identity, annihilation, creation, oscillator = local_boson_operators(
        args.local_dim, basis="sine-dvr", dvr_qmax=args.qmax
    )
    bonds, dims = spin_boson_bond_hamiltonians(
        chain, identity, annihilation, creation, oscillator
    )
    factors = spin_boson_product_factors(chain, oscillator, spin_state=1)
    return nearest_neighbor_mpo(bonds, dims), dims, factors


def read_reference(path, tmax):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["time"]) for row in rows])
    selected = times <= float(tmax) + 1.0e-12
    return {
        "time": times[selected],
        "sigma_z": np.asarray([float(row["sigma_z"]) for row in rows])[selected],
        "rho01_abs": np.asarray([float(row["rho01_abs"]) for row in rows])[selected],
    }


def observe(state):
    rho = nnn_system_reduced_density_matrix(state)
    return {
        "sigma_z": float((rho[0, 0] - rho[1, 1]).real),
        "rho01_abs": float(abs(rho[0, 1])),
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_error": float(np.max(np.abs(rho - rho.conj().T))),
    }


def run_rank(mpo, factors, rank, times, args):
    state = nnn_product_state(factors, max_bond=rank)
    engine = NNNLETTATDVPEngine(
        mpo,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        canonicalize_first=True,
    )
    rows = [observe(state)]
    norms = [state.norm()]
    energies = [state.expectation_mpo(mpo)]
    residuals = [0.0]
    step_seconds = [0.0]
    started = time.perf_counter()
    for step in range(1, len(times)):
        tic = time.perf_counter()
        state, info = engine.step(state, times[step] - times[step - 1])
        step_seconds.append(time.perf_counter() - tic)
        rows.append(observe(state))
        norms.append(state.norm())
        energies.append(state.expectation_mpo(mpo))
        residuals.append(info["krylov_residual_max"])
        if step % args.report_every == 0 or step == len(times) - 1:
            print(
                f"NNN D={rank} step={step}/{len(times)-1} "
                f"t={times[step]:.3f} seconds={step_seconds[-1]:.3f}",
                flush=True,
            )
    return {
        "rank": int(rank),
        "ranks": [int(tensor.shape[-1]) for tensor in state.tensors[:-1]],
        "parameters": int(sum(tensor.size for tensor in state.tensors)),
        "sigma_z": np.asarray([row["sigma_z"] for row in rows]),
        "rho01_abs": np.asarray([row["rho01_abs"] for row in rows]),
        "trace_error": np.asarray([row["trace_error"] for row in rows]),
        "hermiticity_error": np.asarray(
            [row["hermiticity_error"] for row in rows]
        ),
        "norm": np.asarray(norms),
        "energy": np.asarray(energies),
        "krylov_residual": np.asarray(residuals),
        "step_seconds": np.asarray(step_seconds),
        "wall_seconds": float(time.perf_counter() - started),
    }


def make_figure(times, reference, runs, output):
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    axes[0].plot(
        reference["time"], reference["sigma_z"], color="black", lw=2.2,
        label="MPS $D=16$ reference",
    )
    for color, run in zip(colors, runs):
        label = rf"NNN-LETTA $D={run['rank']}$ ({run['parameters']:,} params)"
        axes[0].plot(times, run["sigma_z"], color=color, lw=1.5, label=label)
        axes[1].semilogy(
            times,
            np.maximum(np.abs(run["sigma_z"] - reference["sigma_z"]), 1.0e-15),
            color=color,
            lw=1.5,
            label=rf"$D={run['rank']}$",
        )
        axes[2].semilogy(
            times,
            np.maximum(np.abs(run["norm"] - 1.0), 1.0e-16),
            color=color,
            lw=1.5,
            label=rf"norm, $D={run['rank']}$",
        )
        axes[2].semilogy(
            times,
            np.maximum(np.abs(run["energy"] - run["energy"][0]), 1.0e-16),
            color=color,
            lw=1.0,
            ls="--",
            label=rf"energy, $D={run['rank']}$",
        )
    axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[1].set_ylabel("absolute population error")
    axes[2].set_ylabel("conservation error")
    axes[2].set_xlabel("time")
    axes[0].set_title("Conventional SBM on a Wilson chain: NNN-LETTA")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmodes", type=int, default=20)
    parser.add_argument("--local-dim", type=int, default=16)
    parser.add_argument("--qmax", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--Lambda", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=0.8)
    parser.add_argument("--omegac", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--krylov-dim", type=int, default=20)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-10)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mpo, dims, factors = build_problem(args)
    reference = read_reference(args.reference, args.tmax)
    times = reference["time"]
    if len(times) < 2 or not np.allclose(np.diff(times), args.dt):
        raise SystemExit("reference time grid does not match --dt")
    runs = [run_rank(mpo, factors, rank, times, args) for rank in args.ranks]
    summary = {
        "model": {
            "name": "conventional zero-temperature spin-boson Wilson chain",
            "nmodes": args.nmodes,
            "local_dim": args.local_dim,
            "alpha": args.alpha,
            "Lambda": args.Lambda,
            "s": args.s,
            "omegac": args.omegac,
            "delta": args.delta,
            "epsilon": args.epsilon,
            "dt": args.dt,
            "tmax": args.tmax,
            "dims": list(dims),
            "reference": str(args.reference),
        },
        "runs": [],
    }
    arrays = {
        "time": times,
        "reference_sigma_z": reference["sigma_z"],
        "reference_rho01_abs": reference["rho01_abs"],
    }
    for run in runs:
        record = {
            "rank": run["rank"],
            "ranks": run["ranks"],
            "parameters": run["parameters"],
            "wall_seconds": run["wall_seconds"],
            "mean_step_seconds": float(np.mean(run["step_seconds"][1:])),
            "max_sigma_z_error": float(
                np.max(np.abs(run["sigma_z"] - reference["sigma_z"]))
            ),
            "max_rho01_abs_error": float(
                np.max(np.abs(run["rho01_abs"] - reference["rho01_abs"]))
            ),
            "max_norm_error": float(np.max(np.abs(run["norm"] - 1.0))),
            "max_energy_drift": float(
                np.max(np.abs(run["energy"] - run["energy"][0]))
            ),
            "max_krylov_residual": float(np.max(run["krylov_residual"])),
        }
        summary["runs"].append(record)
        print(json.dumps(record), flush=True)
        prefix = f"nnn_d{run['rank']}"
        for key in (
            "sigma_z", "rho01_abs", "norm", "energy", "krylov_residual",
            "step_seconds",
        ):
            arrays[f"{prefix}_{key}"] = run[key]
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    make_figure(
        times,
        reference,
        runs,
        args.output / "conventional_sbm_nnn_letta.png",
    )


if __name__ == "__main__":
    main()
