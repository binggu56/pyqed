"""Infinite-continuum cLETTA benchmark for a nonlocal 1D Bose gas.

The model is

    H/L = <d psi^dag d psi> + int_0^inf dr V(r) < :n(r)n(0): >

with an exponential/Yukawa kernel

    V(r) = sum_i strength_i exp(-decay_i r).

The integral is evaluated by the continuum cMPS transfer resolvent; no
real-space grid, finite box, or finite particle-number calculation appears.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import ContinuousMPS, pack_canonical_parameters


def product_state(density):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    return ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)


def kernel_value(r, decay_rates, strengths):
    r = np.asarray(r, dtype=float)
    values = np.zeros_like(r)
    for decay, strength in zip(decay_rates, strengths):
        values += float(strength) * np.exp(-float(decay) * r)
    return values


def row_from_state(label, state):
    return {
        "ansatz": label,
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success) if state.success is not None else True,
        "nfev": int(state.nfev),
    }


def run(args):
    rates = np.asarray(args.decay_rates, dtype=float)
    strengths = np.asarray(args.strengths, dtype=float)
    if rates.shape != strengths.shape:
        raise ValueError("--decay-rates and --strengths must have the same length.")

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
    rows.append(row_from_state("product", product))

    cmps = None
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
        rows.append(row_from_state(f"cMPS-D{args.cmps_bond_dim}", cmps))

    cletta = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=args.bond_dim,
        interaction_decay_rates=rates,
        strengths=strengths,
        density=args.density,
        num_modes=args.num_modes,
        depth=args.depth,
        memory_decay_rates=args.memory_decay_rates,
        optimize_memory_rates=not args.fixed_memory_rates,
        optimize_memory_frequencies=not args.fixed_memory_frequencies,
        restarts=args.restarts,
        seed=args.seed,
        maxiter=args.maxiter,
        regularization=args.regularization,
        density_gauge_penalty=args.density_gauge_penalty,
        tie_scale=args.tie_scale,
    )
    cletta_row = row_from_state(
        f"cLETTA-D{args.bond_dim}-M{args.num_modes}-L{args.depth}",
        cletta,
    )
    cletta_row["memory_rates"] = ";".join(f"{value:.12g}" for value in cletta.cletta_decay_rates)
    cletta_row["memory_frequencies"] = ";".join(f"{value:.12g}" for value in cletta.cletta_frequencies)
    cletta_row["tie_norm"] = float(np.linalg.norm(cletta.cletta_tie_matrices))
    rows.append(cletta_row)

    for row in rows:
        print(
            f"{row['ansatz']:>18s}  E={row['energy']:.10f}  "
            f"T={row['kinetic']:.10f}  V={row['interaction']:.10f}  "
            f"success={row['success']}"
        )
    if "memory_rates" in cletta_row:
        print(f"cLETTA memory rates: [{cletta_row['memory_rates']}]")
    if "memory_frequencies" in cletta_row:
        print(f"cLETTA memory frequencies: [{cletta_row['memory_frequencies']}]")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        labels = [row["ansatz"] for row in rows]
        kinetic = np.array([row["kinetic"] for row in rows], dtype=float)
        interaction = np.array([row["interaction"] for row in rows], dtype=float)
        x = np.arange(len(rows))

        fig, (ax_energy, ax_kernel) = plt.subplots(1, 2, figsize=(8.5, 3.6))
        ax_energy.bar(x, interaction, label="interaction", color="#4477aa")
        ax_energy.bar(x, kinetic, bottom=interaction, label="kinetic", color="#cc6677")
        ax_energy.set_xticks(x, labels, rotation=18, ha="right")
        ax_energy.set_ylabel("energy density")
        ax_energy.legend(frameon=False)
        ax_energy.grid(axis="y", alpha=0.25)

        rmax = max(8.0 / float(np.min(rates)), 4.0)
        grid = np.linspace(0.0, rmax, 300)
        ax_kernel.plot(grid, kernel_value(grid, rates, strengths), color="#228833")
        ax_kernel.set_xlabel("r")
        ax_kernel.set_ylabel("V(r)")
        ax_kernel.set_title("exponential kernel")
        ax_kernel.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--decay-rates", nargs="+", type=float, default=[0.5, 2.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 0.5])
    parser.add_argument("--cmps-bond-dim", type=int, default=2)
    parser.add_argument("--cmps-restarts", type=int, default=4)
    parser.add_argument("--cmps-maxiter", type=int, default=220)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--memory-decay-rates", nargs="*", type=float, default=None)
    parser.add_argument("--fixed-memory-rates", action="store_true")
    parser.add_argument("--fixed-memory-frequencies", action="store_true")
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-4)
    parser.add_argument("--tie-scale", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_yukawa_continuum.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_yukawa_continuum.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
