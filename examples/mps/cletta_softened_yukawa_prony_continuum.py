"""Softened-Yukawa continuum cLETTA benchmark using a Prony kernel fit.

The physical kernel is

    V(r) = g exp(-kappa r) / sqrt(r^2 + a^2),

sampled on a continuum distance grid and fitted as

    V(r) ~= sum_i strength_i exp(-decay_i r)

by the same Prony idea used by the GDVR spatial-density kernel path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ContinuousMPS,
    fit_exponential_kernel_prony,
    pack_canonical_parameters,
    softened_yukawa_kernel,
)


def product_state(density):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    return ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)


def state_row(label, state, *, cutoff=""):
    row = {
        "ansatz": label,
        "cutoff": cutoff,
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success) if state.success is not None else True,
        "nfev": int(state.nfev),
        "memory_rates": "",
        "memory_frequencies": "",
        "tie_norm": "",
    }
    if state.cletta_decay_rates is not None:
        row["memory_rates"] = ";".join(f"{value:.12g}" for value in state.cletta_decay_rates)
    if state.cletta_frequencies is not None:
        row["memory_frequencies"] = ";".join(f"{value:.12g}" for value in state.cletta_frequencies)
    if state.cletta_tie_matrices is not None:
        row["tie_norm"] = float(np.linalg.norm(state.cletta_tie_matrices))
    return row


def run(args):
    distances = args.fit_dx * np.arange(1, args.fit_points + 1, dtype=float)
    target = softened_yukawa_kernel(
        distances,
        strength=args.strength,
        screening=args.screening,
        softening=args.softening,
    )
    fit = fit_exponential_kernel_prony(distances, target, rank=args.prony_rank)
    rates = fit["decay_rates"]
    strengths = fit["strengths"]

    print(
        f"Prony rank={args.prony_rank} terms={len(rates)} "
        f"rel_error={fit['rel_error']:.3e} max_abs={fit['max_abs_error']:.3e}"
    )
    print("decay rates:", " ".join(f"{value:.8g}" for value in rates))
    print("strengths:  ", " ".join(f"{value:.8g}" for value in strengths))

    rows = []
    product = product_state(args.density)
    product_values = product.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=args.density,
    )
    product.energy = product_values["energy_density"]
    product.density = product_values["density"]
    product.kinetic = product_values["kinetic"]
    product.interaction = product_values["interaction"]
    product.raw_density = product_values["raw_density"]
    product.scale = product_values["scale"]
    rows.append(state_row("product", product))

    if args.cmps_bond_dim > 0:
        cmps = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
            bond_dim=args.cmps_bond_dim,
            decay_rates=rates,
            strengths=strengths,
            density=args.density,
            restarts=args.cmps_restarts,
            seed=args.seed + 100,
            maxiter=args.cmps_maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
        )
        rows.append(state_row(f"cMPS-D{args.cmps_bond_dim}", cmps))

    seed_parameters = None
    for cutoff_index, cutoff in enumerate(args.cutoffs):
        seeds = [] if seed_parameters is None else [seed_parameters]
        cletta = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=rates,
            strengths=strengths,
            density=args.density,
            num_modes=len(rates),
            depth=int(cutoff),
            memory_decay_rates=rates if args.initialize_memory_from_fit else None,
            optimize_memory_rates=not args.fixed_memory_rates,
            optimize_memory_frequencies=not args.fixed_memory_frequencies,
            seed_parameters=seeds,
            restarts=max(args.restarts, len(seeds) + 1),
            seed=args.seed + cutoff_index,
            maxiter=args.maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            tie_scale=args.tie_scale,
        )
        seed_parameters = cletta.cletta_parameters
        rows.append(state_row(f"auxL-cLETTA-D{args.bond_dim}-M{len(rates)}", cletta, cutoff=int(cutoff)))

    for row in rows:
        cutoff = f" Naux={row['cutoff']}" if row["cutoff"] != "" else ""
        print(
            f"{row['ansatz']:>20s}{cutoff:>8s} "
            f"E={row['energy']:.10f} T={row['kinetic']:.10f} "
            f"V={row['interaction']:.10f} success={row['success']}"
        )
        if row["memory_rates"]:
            print(f"  memory rates=[{row['memory_rates']}] tie_norm={row['tie_norm']}")
        if row["memory_frequencies"]:
            print(f"  memory frequencies=[{row['memory_frequencies']}]")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        fit_keys = [
            "fit_rel_error",
            "fit_max_abs_error",
            "fit_decays",
            "fit_strengths",
        ]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys + fit_keys)
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["fit_rel_error"] = fit["rel_error"]
                out["fit_max_abs_error"] = fit["max_abs_error"]
                out["fit_decays"] = ";".join(f"{value:.12g}" for value in rates)
                out["fit_strengths"] = ";".join(f"{value:.12g}" for value in strengths)
                writer.writerow(out)
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        labels = [
            row["ansatz"] if row["cutoff"] == "" else f"{row['ansatz']} N={row['cutoff']}"
            for row in rows
        ]
        kinetic = np.array([row["kinetic"] for row in rows], dtype=float)
        interaction = np.array([row["interaction"] for row in rows], dtype=float)
        x = np.arange(len(rows))

        fig, (ax_energy, ax_kernel) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        ax_energy.bar(x, interaction, label="interaction", color="#4477aa")
        ax_energy.bar(x, kinetic, bottom=interaction, label="kinetic", color="#cc6677")
        ax_energy.plot(x, kinetic + interaction, "ko-", lw=1, ms=4)
        ax_energy.set_xticks(x, labels, rotation=18, ha="right")
        ax_energy.set_ylabel("energy density")
        ax_energy.grid(axis="y", alpha=0.25)
        ax_energy.legend(frameon=False)

        rgrid = np.linspace(args.fit_dx, args.fit_dx * args.fit_points, 500)
        exact = softened_yukawa_kernel(
            rgrid,
            strength=args.strength,
            screening=args.screening,
            softening=args.softening,
        )
        fitted = np.exp(-rgrid[:, None] * rates[None, :]) @ strengths
        ax_kernel.plot(rgrid, exact, label="softened Yukawa", color="#228833")
        ax_kernel.plot(rgrid, fitted, "--", label="Prony fit", color="#aa3377")
        ax_kernel.set_xlabel("r")
        ax_kernel.set_ylabel("V(r)")
        ax_kernel.set_title(f"Prony rel. error {fit['rel_error']:.1e}")
        ax_kernel.grid(alpha=0.25)
        ax_kernel.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows, fit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--screening", type=float, default=0.3)
    parser.add_argument("--softening", type=float, default=0.5)
    parser.add_argument("--fit-dx", type=float, default=0.1)
    parser.add_argument("--fit-points", type=int, default=120)
    parser.add_argument("--prony-rank", type=int, default=6)
    parser.add_argument("--cmps-bond-dim", type=int, default=2)
    parser.add_argument("--cmps-restarts", type=int, default=3)
    parser.add_argument("--cmps-maxiter", type=int, default=180)
    parser.add_argument("--bond-dim", type=int, default=1)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1])
    parser.add_argument("--initialize-memory-from-fit", action="store_true")
    parser.add_argument("--fixed-memory-rates", action="store_true")
    parser.add_argument("--fixed-memory-frequencies", action="store_true")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-4)
    parser.add_argument("--tie-scale", type=float, default=0.03)
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_softened_yukawa_prony_continuum.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_softened_yukawa_prony_continuum.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
