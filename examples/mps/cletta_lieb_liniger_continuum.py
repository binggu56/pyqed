"""Infinite-continuum cLETTA benchmark for the Lieb-Liniger gas.

This example has no real-space lattice, box length, or particle-number
finite-size extrapolation.  The variational state is an infinite uniform
continuum cMPS with finite-depth exponential cLETTA memory modes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pyqed.mps import ContinuousMPS


def lieb_liniger_bethe_energy(coupling, *, density=1.0, n_grid=160):
    """Return the thermodynamic-limit Lieb-Liniger energy density."""
    coupling = float(coupling)
    density = float(density)
    if coupling <= 0.0:
        raise ValueError("coupling must be positive.")
    if density <= 0.0:
        raise ValueError("density must be positive.")

    gamma = coupling / density
    points, weights = np.polynomial.legendre.leggauss(int(n_grid))
    rhs = np.full(points.size, 1.0 / (2.0 * np.pi), dtype=float)

    def solve_lambda(lam):
        delta = points[:, None] - points[None, :]
        kernel = 2.0 * lam / (lam * lam + delta * delta)
        matrix = np.eye(points.size) - weights[None, :] * kernel / (2.0 * np.pi)
        rapidity_density = np.linalg.solve(matrix, rhs)
        norm = float(np.dot(weights, rapidity_density))
        kinetic = float(np.dot(weights, points * points * rapidity_density))
        return lam / norm, norm, kinetic

    def residual(log_lam):
        return solve_lambda(np.exp(log_lam))[0] - gamma

    log_lam = brentq(residual, -30.0, 30.0, xtol=1.0e-12, rtol=1.0e-12)
    _, norm, kinetic = solve_lambda(np.exp(log_lam))
    return density**3 * kinetic / norm**3


def product_energy(coupling, *, density=1.0):
    """Gross-Pitaevskii/product cMPS energy density."""
    return float(coupling) * float(density) ** 2


def run(args):
    rows = []
    previous_cletta_parameters = None
    previous_cmps_theta = None

    for index, coupling in enumerate(args.couplings):
        exact = lieb_liniger_bethe_energy(
            coupling,
            density=args.density,
            n_grid=args.bethe_grid,
        )
        product = product_energy(coupling, density=args.density)

        cmps_energy = np.nan
        cmps_error = np.nan
        cmps_state = None
        if args.cmps_bond_dim > 0:
            seed_thetas = [] if previous_cmps_theta is None else [previous_cmps_theta]
            cmps_state = ContinuousMPS.optimize_lieb_liniger_fixed_density(
                bond_dim=args.cmps_bond_dim,
                coupling=coupling,
                density=args.density,
                seed_thetas=seed_thetas,
                restarts=max(args.cmps_restarts, len(seed_thetas) + 1),
                seed=args.seed + 1000 + index,
                maxiter=args.cmps_maxiter,
                use_jax=not args.no_jax,
            )
            previous_cmps_theta = cmps_state.theta
            cmps_energy = cmps_state.energy
            cmps_error = cmps_energy - exact

        seed_parameters = [] if previous_cletta_parameters is None else [previous_cletta_parameters]
        cletta = ContinuousMPS.optimize_lieb_liniger_cletta_fixed_density(
            bond_dim=args.bond_dim,
            coupling=coupling,
            density=args.density,
            num_modes=args.num_modes,
            depth=args.depth,
            decay_rates=args.decay_rates,
            optimize_rates=not args.fixed_rates,
            seed_parameters=seed_parameters,
            restarts=max(args.restarts, len(seed_parameters) + 1),
            seed=args.seed + index,
            maxiter=args.maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            tie_scale=args.tie_scale,
        )
        previous_cletta_parameters = cletta.cletta_parameters
        cletta_error = cletta.energy - exact

        row = {
            "coupling": float(coupling),
            "density": float(args.density),
            "exact_energy": float(exact),
            "product_energy": float(product),
            "product_error": float(product - exact),
            "cmps_energy": float(cmps_energy),
            "cmps_error": float(cmps_error),
            "cletta_energy": float(cletta.energy),
            "cletta_error": float(cletta_error),
            "cletta_relative_error": float(cletta_error / abs(exact)),
            "cletta_kinetic": float(cletta.kinetic),
            "cletta_contact": float(cletta.contact),
            "cletta_raw_density": float(cletta.raw_density),
            "cletta_scale": float(cletta.scale),
            "cletta_rates": ";".join(f"{value:.12g}" for value in cletta.cletta_decay_rates),
            "cletta_tie_norm": float(np.linalg.norm(cletta.cletta_tie_matrices)),
            "cletta_success": bool(cletta.success),
            "cletta_nfev": int(cletta.nfev),
        }
        if cmps_state is not None:
            row["cmps_success"] = bool(cmps_state.success)
            row["cmps_nfev"] = int(cmps_state.nfev)
        rows.append(row)

        print(
            f"c={coupling:g} exact={exact:.10f} product={product:.10f} "
            f"cLETTA={cletta.energy:.10f} err={cletta_error:.3e} "
            f"rates=[{row['cletta_rates']}]"
        )

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
        couplings = np.array([row["coupling"] for row in rows], dtype=float)
        exact = np.array([row["exact_energy"] for row in rows], dtype=float)
        product = np.array([row["product_energy"] for row in rows], dtype=float)
        cletta = np.array([row["cletta_energy"] for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(couplings, exact, "o-", label="Bethe ansatz")
        ax.plot(couplings, product, "s--", label="product cMPS")
        if args.cmps_bond_dim > 0:
            cmps = np.array([row["cmps_energy"] for row in rows], dtype=float)
            ax.plot(couplings, cmps, "d-.", label=f"cMPS D={args.cmps_bond_dim}")
        ax.plot(
            couplings,
            cletta,
            "^-",
            label=f"cLETTA D={args.bond_dim}, M={args.num_modes}, L={args.depth}",
        )
        if np.max(couplings) / np.min(couplings) > 5.0:
            ax.set_xscale("log")
        ax.set_xlabel("Lieb-Liniger coupling c at fixed density")
        ax.set_ylabel("energy density")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--couplings", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--bond-dim", type=int, default=1)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--decay-rates", nargs="*", type=float, default=None)
    parser.add_argument("--fixed-rates", action="store_true")
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=220)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-4)
    parser.add_argument("--tie-scale", type=float, default=0.05)
    parser.add_argument("--bethe-grid", type=int, default=160)
    parser.add_argument("--cmps-bond-dim", type=int, default=0)
    parser.add_argument("--cmps-restarts", type=int, default=4)
    parser.add_argument("--cmps-maxiter", type=int, default=400)
    parser.add_argument("--no-jax", action="store_true")
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_lieb_liniger_continuum.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_lieb_liniger_continuum.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
