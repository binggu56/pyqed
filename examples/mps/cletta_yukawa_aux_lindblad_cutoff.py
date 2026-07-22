"""Auxiliary-Lindblad-first continuum cLETTA benchmark.

This scans the explicit finite-Fock pseudomode cutoff before using the
HEOM-like hierarchy language.  The model is the same nonlocal continuum Bose
gas as ``cletta_yukawa_continuum.py``:

    V(r) = sum_i strength_i exp(-decay_i r).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import ContinuousMPS, pack_canonical_parameters


def product_observables(density, decay_rates, strengths):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)
    values = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=decay_rates,
        strengths=strengths,
        density=density,
    )
    return {
        "ansatz": "product",
        "cutoff": "",
        "energy": values["energy_density"],
        "kinetic": values["kinetic"],
        "interaction": values["interaction"],
        "raw_density": values["raw_density"],
        "scale": values["scale"],
        "memory_rates": "",
        "memory_frequencies": "",
        "tie_norm": "",
        "success": True,
        "nfev": 0,
    }


def row_from_state(label, cutoff, state):
    row = {
        "ansatz": label,
        "cutoff": cutoff,
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success),
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
    rates = np.asarray(args.decay_rates, dtype=float)
    strengths = np.asarray(args.strengths, dtype=float)
    if rates.shape != strengths.shape:
        raise ValueError("--decay-rates and --strengths must have the same length.")

    rows = [product_observables(args.density, rates, strengths)]

    cmps = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
        bond_dim=args.bond_dim,
        decay_rates=rates,
        strengths=strengths,
        density=args.density,
        restarts=args.cmps_restarts,
        seed=args.seed + 100,
        maxiter=args.cmps_maxiter,
        regularization=args.regularization,
        density_gauge_penalty=args.density_gauge_penalty,
    )
    rows.append(row_from_state(f"cMPS-D{args.bond_dim}", "", cmps))

    seed_parameters = None
    for cutoff_index, cutoff in enumerate(args.cutoffs):
        cutoff = int(cutoff)
        if cutoff < 1:
            continue
        seeds = [] if seed_parameters is None else [seed_parameters]
        state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=rates,
            strengths=strengths,
            density=args.density,
            num_modes=args.num_modes,
            depth=cutoff,
            memory_decay_rates=args.memory_decay_rates,
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
        seed_parameters = state.cletta_parameters
        rows.append(row_from_state(f"auxL-cLETTA-D{args.bond_dim}-M{args.num_modes}", cutoff, state))

    for row in rows:
        cutoff = f" Naux={row['cutoff']}" if row["cutoff"] != "" else ""
        print(
            f"{row['ansatz']:>18s}{cutoff:>8s}  "
            f"E={float(row['energy']):.10f}  "
            f"T={float(row['kinetic']):.10f}  "
            f"V={float(row['interaction']):.10f}  "
            f"success={row['success']}"
        )
        if row["memory_rates"]:
            print(f"  memory rates=[{row['memory_rates']}] tie_norm={row['tie_norm']}")
        if row["memory_frequencies"]:
            print(f"  memory frequencies=[{row['memory_frequencies']}]")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        labels = [
            row["ansatz"] if row["cutoff"] == "" else f"{row['ansatz']} N={row['cutoff']}"
            for row in rows
        ]
        kinetic = np.array([float(row["kinetic"]) for row in rows])
        interaction = np.array([float(row["interaction"]) for row in rows])
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(x, interaction, label="interaction", color="#4477aa")
        ax.bar(x, kinetic, bottom=interaction, label="kinetic", color="#cc6677")
        ax.plot(x, kinetic + interaction, "ko-", lw=1, ms=4)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel("energy density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--decay-rates", nargs="+", type=float, default=[0.5, 2.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 0.5])
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1])
    parser.add_argument("--memory-decay-rates", nargs="*", type=float, default=None)
    parser.add_argument("--fixed-memory-rates", action="store_true")
    parser.add_argument("--fixed-memory-frequencies", action="store_true")
    parser.add_argument("--cmps-restarts", type=int, default=4)
    parser.add_argument("--cmps-maxiter", type=int, default=220)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-4)
    parser.add_argument("--tie-scale", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_yukawa_aux_lindblad_cutoff.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_yukawa_aux_lindblad_cutoff.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
