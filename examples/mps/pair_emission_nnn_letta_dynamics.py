#!/usr/bin/env python3
"""Real-time NNN-LETTA benchmark for correlated boson-pair emission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from pyqed.letta import NNNLETTATDVPEngine, nnn_product_state
from pyqed.mps.mpo import sop_to_mpo


def _operators(dim):
    annihilation = np.diag(np.sqrt(np.arange(1, dim)), 1)
    creation = annihilation.T
    number = creation @ annihilation
    return annihilation, creation, number


def build_model(nmodes, local_dim, *, omega_system, frequencies, hopping, coupling):
    dims = (2,) + (local_dim,) * nmodes
    eye_spin = np.eye(2)
    eye_boson = np.eye(local_dim)
    excited = np.diag([0.0, 1.0])
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]])
    raising = lowering.T
    annihilation, creation, number = _operators(local_dim)
    identities = (eye_spin,) + (eye_boson,) * nmodes

    def product_term(coefficient, replacements):
        operators = list(identities)
        for site, operator in replacements.items():
            operators[site] = operator
        return coefficient, tuple(operators)

    terms = [product_term(omega_system, {0: excited})]
    terms.extend(
        product_term(frequency, {mode + 1: number})
        for mode, frequency in enumerate(frequencies)
    )
    for mode in range(1, nmodes):
        terms.append(
            product_term(hopping, {mode: creation, mode + 1: annihilation})
        )
        terms.append(
            product_term(hopping, {mode: annihilation, mode + 1: creation})
        )
    terms.append(
        product_term(
            coupling,
            {0: lowering, 1: creation, 2: creation},
        )
    )
    terms.append(
        product_term(
            coupling,
            {0: raising, 1: annihilation, 2: annihilation},
        )
    )
    return sop_to_mpo(dims, terms), excited, number


def exact_population(mpo, initial, times, excited):
    hamiltonian = csr_matrix(mpo.to_dense())
    states = expm_multiply(
        -1.0j * hamiltonian,
        initial,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    block = int(initial.size // 2)
    population = np.sum(np.abs(states[:, block:]) ** 2, axis=1).real
    norms = np.sum(np.abs(states) ** 2, axis=1).real
    energies = np.einsum(
        "ti,ij,tj->t", states.conj(), hamiltonian.toarray(), states, optimize=True
    ).real
    return population, norms, energies


def nnn_population(state, excited):
    operators = [np.eye(dim) for dim in state.dims]
    operators[0] = excited
    return float(state.expectation_product_operator(operators))


def run_rank(mpo, factors, excited, times, rank, *, krylov_dim, krylov_tol):
    state = nnn_product_state(factors, max_bond=rank)
    engine = NNNLETTATDVPEngine(
        mpo,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        canonicalize_first=True,
    )
    population = [nnn_population(state, excited)]
    norm = [state.norm()]
    energy = [state.expectation_mpo(mpo)]
    residual = [0.0]
    started = time.perf_counter()
    for _ in range(1, len(times)):
        state, info = engine.step(state, times[1] - times[0])
        population.append(nnn_population(state, excited))
        norm.append(state.norm())
        energy.append(state.expectation_mpo(mpo))
        residual.append(info["krylov_residual_max"])
    wall = time.perf_counter() - started
    return {
        "rank": int(rank),
        "parameters": int(sum(tensor.size for tensor in state.tensors)),
        "ranks": [int(tensor.shape[-1]) for tensor in state.tensors[:-1]],
        "population": np.asarray(population),
        "norm": np.asarray(norm),
        "energy": np.asarray(energy),
        "krylov_residual": np.asarray(residual),
        "wall_seconds": float(wall),
    }


def make_figure(times, exact, runs, output):
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    axes[0].plot(times, exact["population"], color="black", lw=2.2, label="exact")
    for color, run in zip(colors, runs):
        label = rf"NNN-LETTA $D={run['rank']}$ ({run['parameters']} params)"
        axes[0].plot(times, run["population"], color=color, lw=1.5, label=label)
        axes[1].semilogy(
            times,
            np.maximum(np.abs(run["population"] - exact["population"]), 1.0e-15),
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
    axes[0].set_ylabel(r"excited population $P_e$")
    axes[1].set_ylabel(r"$|P_e-P_e^{\rm exact}|$")
    axes[2].set_ylabel("conservation error")
    axes[2].set_xlabel("time")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[2].legend(frameon=False, ncol=2, fontsize=7)
    axes[0].set_title("Correlated pair emission: real-time NNN-LETTA")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmodes", type=int, default=4)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--omega-system", type=float, default=1.0)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[0.45, 0.55, 0.72, 0.88])
    parser.add_argument("--hopping", type=float, default=0.18)
    parser.add_argument("--coupling", type=float, default=0.28)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tmax", type=float, default=8.0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--krylov-dim", type=int, default=24)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.frequencies) != args.nmodes:
        raise SystemExit("--frequencies must contain one value per bath mode")
    args.output.mkdir(parents=True, exist_ok=True)

    mpo, excited, _number = build_model(
        args.nmodes,
        args.local_dim,
        omega_system=args.omega_system,
        frequencies=args.frequencies,
        hopping=args.hopping,
        coupling=args.coupling,
    )
    factors = [np.array([0.0, 1.0])] + [
        np.eye(args.local_dim)[:, 0] for _ in range(args.nmodes)
    ]
    initial = factors[0]
    for factor in factors[1:]:
        initial = np.kron(initial, factor)
    times = np.linspace(0.0, args.tmax, int(round(args.tmax / args.dt)) + 1)
    exact_population_values, exact_norm, exact_energy = exact_population(
        mpo, initial, times, excited
    )
    exact = {
        "population": exact_population_values,
        "norm": exact_norm,
        "energy": exact_energy,
    }
    runs = [
        run_rank(
            mpo,
            factors,
            excited,
            times,
            rank,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
        )
        for rank in args.ranks
    ]
    summary = {
        "model": {
            "Hamiltonian": (
                "Omega |e><e| + sum_j omega_j n_j + J sum_j(b_j^dagger b_{j+1}+h.c.) "
                "+ g(sigma_- b_1^dagger b_2^dagger+h.c.)"
            ),
            "nmodes": args.nmodes,
            "local_dim": args.local_dim,
            "omega_system": args.omega_system,
            "frequencies": list(args.frequencies),
            "hopping": args.hopping,
            "coupling": args.coupling,
            "dt": args.dt,
            "tmax": args.tmax,
        },
        "runs": [],
    }
    for run in runs:
        record = {
            "rank": run["rank"],
            "ranks": run["ranks"],
            "parameters": run["parameters"],
            "wall_seconds": run["wall_seconds"],
            "max_population_error": float(
                np.max(np.abs(run["population"] - exact["population"]))
            ),
            "max_norm_error": float(np.max(np.abs(run["norm"] - 1.0))),
            "max_energy_drift": float(
                np.max(np.abs(run["energy"] - run["energy"][0]))
            ),
            "max_krylov_residual": float(np.max(run["krylov_residual"])),
        }
        summary["runs"].append(record)
        print(json.dumps(record), flush=True)

    arrays = {"time": times, "exact_population": exact["population"]}
    for run in runs:
        prefix = f"nnn_d{run['rank']}"
        for key in ("population", "norm", "energy", "krylov_residual"):
            arrays[f"{prefix}_{key}"] = run[key]
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    make_figure(
        times,
        exact,
        runs,
        args.output / "pair_emission_nnn_letta_dynamics.png",
    )


if __name__ == "__main__":
    main()
