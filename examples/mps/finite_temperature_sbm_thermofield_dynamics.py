#!/usr/bin/env python3
"""Finite-temperature spin-boson dynamics by thermofield-chain TDVP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_wilson_chain,
    spin_boson_bond_hamiltonians,
    spin_boson_product_factors,
    thermofield_spin_boson_bond_hamiltonians,
    thermofield_spin_boson_product_factors,
    thermofield_spin_boson_wilson_chains,
)
from pyqed.mps.mpo import nearest_neighbor_mpo
from pyqed.mps.mps import MPS
from pyqed.mps.tdvp import TDVPEngine
from pyqed.narg.spin_boson import local_boson_operators


def _parse_case(value):
    try:
        fields = tuple(int(part) for part in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "cases must have the form local_dim:max_bond or nmodes:local_dim:max_bond"
        ) from error
    if len(fields) == 2:
        nmodes = None
        local_dim, max_bond = fields
    elif len(fields) == 3:
        nmodes, local_dim, max_bond = fields
    else:
        raise argparse.ArgumentTypeError(
            "cases must have the form local_dim:max_bond or nmodes:local_dim:max_bond"
        )
    if (nmodes is not None and nmodes < 1) or local_dim < 2 or max_bond < 1:
        raise argparse.ArgumentTypeError("local_dim must be >=2 and max_bond positive")
    return nmodes, local_dim, max_bond


def _observe(state, mpo, spin_site):
    norm2 = float(np.real(state.norm_squared()))
    local_rdms = state.make_local_site_rdm()
    rho = local_rdms[spin_site]
    bath_rdms = [value for site, value in local_rdms.items() if site != spin_site]
    fock_edge_population = max(float(np.real(value[-1, -1])) for value in bath_rdms)
    max_occupation = max(
        float(np.dot(np.arange(value.shape[0]), np.real(np.diag(value))))
        for value in bath_rdms
    )
    energy = float(np.real(state.expectation(mpo) / norm2))
    return {
        "sigma_z": float(np.real(rho[0, 0] - rho[1, 1])),
        "coherence": float(abs(rho[0, 1])),
        "norm": float(np.sqrt(norm2)),
        "energy": energy,
        "max_bond": max(state.bond_orders()),
        "fock_edge_population": fock_edge_population,
        "max_occupation": max_occupation,
    }


def _run(mpo, factors, spin_site, times, max_bond, args, label):
    state = MPS([factor.reshape(1, factor.size, 1) for factor in factors])
    engine = TDVPEngine(
        mpo,
        integrator="tdvp2",
        max_bond=max_bond,
        cutoff=args.cutoff,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        canonicalize_first=True,
        canonicalize_each_step=False,
    )
    observations = [_observe(state, mpo, spin_site)]
    truncation = [0.0]
    residual = [0.0]
    step_seconds = [0.0]
    started = time.perf_counter()
    for step in range(1, len(times)):
        tic = time.perf_counter()
        state, info = engine.step(
            state,
            times[step] - times[step - 1],
            normalize=False,
            return_info=True,
        )
        step_seconds.append(time.perf_counter() - tic)
        observations.append(_observe(state, mpo, spin_site))
        truncation.append(float(info.get("truncation_error", 0.0)))
        residual.append(float(info.get("krylov_residual_max", 0.0)))
        if step % args.report_every == 0 or step == len(times) - 1:
            print(
                f"{label}: step {step}/{len(times)-1}, t={times[step]:.3f}, "
                f"bond={observations[-1]['max_bond']}, "
                f"seconds={step_seconds[-1]:.3f}",
                flush=True,
            )
    engine.close()
    return {
        "sigma_z": np.asarray([value["sigma_z"] for value in observations]),
        "coherence": np.asarray([value["coherence"] for value in observations]),
        "norm": np.asarray([value["norm"] for value in observations]),
        "energy": np.asarray([value["energy"] for value in observations]),
        "max_bond": np.asarray([value["max_bond"] for value in observations]),
        "fock_edge_population": np.asarray(
            [value["fock_edge_population"] for value in observations]
        ),
        "max_occupation": np.asarray(
            [value["max_occupation"] for value in observations]
        ),
        "truncation": np.asarray(truncation),
        "krylov_residual": np.asarray(residual),
        "step_seconds": np.asarray(step_seconds),
        "parameters": int(sum(np.asarray(factor).size for factor in state.factors)),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _finite_temperature_problem(args, nmodes, local_dim):
    positive, negative, occupations = thermofield_spin_boson_wilson_chains(
        nmodes,
        temperature=args.temperature,
        alpha=args.alpha,
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    operators = local_boson_operators(local_dim, basis="fock")
    bonds, dims = thermofield_spin_boson_bond_hamiltonians(
        positive, negative, *operators
    )
    factors = thermofield_spin_boson_product_factors(
        positive, negative, operators[-1], spin_state=args.spin_state
    )
    return nearest_neighbor_mpo(bonds, dims), factors, nmodes, occupations


def _zero_temperature_problem(args, nmodes, local_dim):
    chain = log_discretized_spin_boson_wilson_chain(
        nmodes,
        alpha=args.alpha,
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    operators = local_boson_operators(local_dim, basis="fock")
    bonds, dims = spin_boson_bond_hamiltonians(chain, *operators)
    factors = spin_boson_product_factors(
        chain, operators[-1], spin_state=args.spin_state
    )
    return nearest_neighbor_mpo(bonds, dims), factors, 0


def _make_figure(times, cases, zero_temperature, output, temperature):
    reference = cases[-1]
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(cases)))
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 10.4), sharex=True)
    if zero_temperature is not None:
        axes[0].plot(
            times,
            zero_temperature["run"]["sigma_z"],
            color="0.25",
            lw=1.7,
            ls="--",
            label=r"$T=0$",
        )
    for color, case in zip(colors, cases):
        label = (
            rf"$T={temperature:g}$, $N={case['nmodes']}$, "
            rf"$d={case['local_dim']}$, $D={case['max_bond']}$"
        )
        axes[0].plot(times, case["run"]["sigma_z"], color=color, lw=1.55, label=label)
        error = np.abs(case["run"]["sigma_z"] - reference["run"]["sigma_z"])
        axes[1].semilogy(times, np.maximum(error, 1.0e-15), color=color, lw=1.5, label=label)
        axes[2].semilogy(
            times,
            np.maximum(case["run"]["fock_edge_population"], 1.0e-16),
            color=color,
            lw=1.35,
            label=label,
        )
        norm_error = np.abs(case["run"]["norm"] - 1.0)
        energy_error = np.abs(case["run"]["energy"] - case["run"]["energy"][0])
        axes[3].semilogy(times, np.maximum(norm_error, 1.0e-16), color=color, lw=1.35)
        axes[3].semilogy(
            times,
            np.maximum(energy_error, 1.0e-16),
            color=color,
            lw=1.1,
            ls="--",
        )
    axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[1].set_ylabel("error vs reference $(N,d,D)$")
    axes[2].set_ylabel("highest Fock-level probability")
    axes[3].set_ylabel("norm (solid), energy (dashed)")
    axes[3].set_xlabel(r"time $t\,\omega_c$")
    regime = "High-temperature" if temperature >= 1.0 else "Finite-temperature"
    axes[0].set_title(f"{regime} SBM: thermofield Wilson chains + MPS TDVP")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmodes", type=int, default=6, help="modes per thermofield branch")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--Lambda", type=float, default=2.0)
    parser.add_argument("--s", type=float, default=0.8)
    parser.add_argument("--omegac", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--spin-state", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument(
        "--cases",
        type=_parse_case,
        nargs="+",
        default=[(None, 6, 12), (None, 6, 24), (None, 8, 24)],
        help=(
            "local_dim:max_bond or nmodes:local_dim:max_bond convergence cases; "
            "the last is the reference"
        ),
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-11)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--no-zero-temperature", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    times = np.arange(int(round(args.tmax / args.dt)) + 1) * args.dt

    cases = []
    occupations = None
    for requested_nmodes, local_dim, max_bond in args.cases:
        nmodes = args.nmodes if requested_nmodes is None else requested_nmodes
        mpo, factors, spin_site, occupations = _finite_temperature_problem(
            args, nmodes, local_dim
        )
        label = f"finite T N={nmodes} d={local_dim} D={max_bond}"
        run = _run(mpo, factors, spin_site, times, max_bond, args, label)
        cases.append(
            {
                "nmodes": nmodes,
                "local_dim": local_dim,
                "max_bond": max_bond,
                "occupations": occupations,
                "run": run,
            }
        )

    zero_temperature = None
    if not args.no_zero_temperature:
        requested_nmodes, local_dim, max_bond = args.cases[-1]
        nmodes = args.nmodes if requested_nmodes is None else requested_nmodes
        mpo, factors, spin_site = _zero_temperature_problem(args, nmodes, local_dim)
        run = _run(mpo, factors, spin_site, times, max_bond, args, "zero T")
        zero_temperature = {
            "nmodes": nmodes,
            "local_dim": local_dim,
            "max_bond": max_bond,
            "run": run,
        }

    reference = cases[-1]["run"]
    records = []
    arrays = {"time": times}
    for index, case in enumerate(cases):
        run = case["run"]
        record = {
            "nmodes": case["nmodes"],
            "local_dim": case["local_dim"],
            "max_bond": case["max_bond"],
            "final_bond": int(run["max_bond"][-1]),
            "parameters": run["parameters"],
            "wall_seconds": run["wall_seconds"],
            "max_sigma_z_error_vs_reference": float(
                np.max(np.abs(run["sigma_z"] - reference["sigma_z"]))
            ),
            "max_norm_error": float(np.max(np.abs(run["norm"] - 1.0))),
            "max_energy_drift": float(
                np.max(np.abs(run["energy"] - run["energy"][0]))
            ),
            "max_krylov_residual": float(np.max(run["krylov_residual"])),
            "max_fock_edge_population": float(
                np.max(run["fock_edge_population"])
            ),
            "max_local_occupation": float(np.max(run["max_occupation"])),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        arrays[f"case{index}_thermal_occupations"] = case["occupations"]
        for key, value in run.items():
            if isinstance(value, np.ndarray):
                arrays[f"case{index}_{key}"] = value
    if zero_temperature is not None:
        for key, value in zero_temperature["run"].items():
            if isinstance(value, np.ndarray):
                arrays[f"zero_temperature_{key}"] = value

    summary = {
        "model": {
            "temperature": args.temperature,
            "nmodes_per_branch": [case["nmodes"] for case in cases],
            "purified_sites": [2 * case["nmodes"] + 1 for case in cases],
            "alpha": args.alpha,
            "Lambda": args.Lambda,
            "s": args.s,
            "omegac": args.omegac,
            "delta": args.delta,
            "epsilon": args.epsilon,
            "dt": args.dt,
            "tmax": args.tmax,
        },
        "cases": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    _make_figure(
        times,
        cases,
        zero_temperature,
        args.output / "finite_temperature_sbm_thermofield.png",
        args.temperature,
    )


if __name__ == "__main__":
    main()
