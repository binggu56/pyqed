"""Full matrix cLETTA benchmark for the nonlocal Luttinger liquid.

Unlike the Gaussian pole fit, this calculation has an explicit cMPS core bond
dimension D.  Each state contains D x D matrices Q and R plus M matrix-valued
exponential memory ties.  The energy is contracted from the normal and
anomalous momentum spectra of the doubled cLETTA transfer generator.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ExponentialLuttingerModel,
    cmps_luttinger_energy_shift_density,
    cmps_luttinger_parameter,
    cmps_luttinger_spectra,
    optimize_luttinger_cletta,
)


def run(args):
    model = ExponentialLuttingerModel(
        decay_rates=args.interaction_decay_rates,
        strengths=args.interaction_strengths,
        fermi_velocity=args.fermi_velocity,
    )
    exact_energy, exact_error = model.ground_state_energy_shift_density()
    states = []
    lower_dim_states = {}
    rows = []

    print(
        f"exact Delta E0/L={exact_energy:.12f} "
        f"quadrature_error={exact_error:.2e}"
    )
    for bond_dim in args.bond_dims:
        previous = []
        for num_modes in range(args.max_modes + 1):
            seeds = list(previous)
            if num_modes in lower_dim_states:
                seeds.append(lower_dim_states[num_modes])
            state = optimize_luttinger_cletta(
                model,
                bond_dim=bond_dim,
                num_modes=num_modes,
                depth=args.depth,
                seed_states=seeds,
                restarts=args.restarts,
                seed=args.seed + 10 * bond_dim + num_modes,
                maxiter=args.maxiter,
                quadrature_points=args.quadrature_points,
                regularization=args.regularization,
                initial_scale=args.initial_scale,
                tie_scale=args.tie_scale,
            )
            validated_energy = cmps_luttinger_energy_shift_density(
                model,
                state,
                quadrature_points=args.validation_quadrature_points,
            )
            state.energy = validated_energy
            state.luttinger_energy_shift_density = validated_energy
            error = validated_energy - exact_energy
            recovered = 1.0 - error / abs(exact_energy)
            row = {
                "bond_dim": bond_dim,
                "num_modes": num_modes,
                "depth": args.depth,
                "effective_bond_dim": state.bond_dim,
                "energy_shift_density": validated_energy,
                "exact_energy_shift_density": exact_energy,
                "energy_error": error,
                "fraction_recovered": recovered,
                "memory_decay_rates": ";".join(
                    f"{value:.12g}" for value in state.cletta_decay_rates
                ),
                "tie_norm": float(np.linalg.norm(state.cletta_tie_matrices)),
                "success": state.success,
                "nfev": state.nfev,
            }
            rows.append(row)
            states.append(state)
            previous = [state]
            lower_dim_states[num_modes] = state
            print(
                f"D={bond_dim} M={num_modes} Deff={state.bond_dim} "
                f"E={validated_energy:.12f} error={error:.3e} "
                f"recovered={100.0 * recovered:.6f}% "
                f"success={state.success}"
            )
            if num_modes:
                print(
                    "  rates=["
                    + ";".join(
                        f"{value:.9g}" for value in state.cletta_decay_rates
                    )
                    + f"] tie_norm={row['tie_norm']:.9g}"
                )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        momentum = np.linspace(0.0, args.momentum_max, args.momentum_points)
        exact_parameter = model.luttinger_parameter(momentum)
        exact_theta = model.exact_squeezing(momentum)
        exact_normal = np.sinh(exact_theta) ** 2
        exact_anomalous = -0.5 * np.sinh(2.0 * exact_theta)

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.5))
        ax_energy, ax_parameter, ax_normal, ax_anomalous = axes.ravel()

        for bond_dim in args.bond_dims:
            selected = [row for row in rows if row["bond_dim"] == bond_dim]
            modes = np.array([row["num_modes"] for row in selected])
            errors = np.maximum(
                [row["energy_error"] for row in selected],
                1.0e-15,
            )
            ax_energy.semilogy(modes, errors, "o-", label=f"D={bond_dim}")
        ax_energy.set_xticks(range(args.max_modes + 1))
        ax_energy.set_xlabel("memory channels M")
        ax_energy.set_ylabel("variational energy error")
        ax_energy.set_title("Full matrix cLETTA convergence")
        ax_energy.legend(frameon=False)

        ax_parameter.plot(momentum, exact_parameter, color="black", linewidth=2, label="exact")
        ax_normal.plot(momentum, exact_normal, color="black", linewidth=2, label="exact")
        ax_anomalous.plot(momentum, exact_anomalous, color="black", linewidth=2, label="exact")
        for state in states:
            if state.luttinger_num_modes != args.max_modes:
                continue
            label = f"D={state.luttinger_bond_dim}, M={state.luttinger_num_modes}"
            normal, anomalous = cmps_luttinger_spectra(state, momentum)
            ax_parameter.plot(
                momentum,
                cmps_luttinger_parameter(state, momentum),
                "--",
                label=label,
            )
            ax_normal.plot(momentum, normal, "--", label=label)
            ax_anomalous.plot(momentum, anomalous, "--", label=label)

        ax_parameter.set_xlabel("momentum k")
        ax_parameter.set_ylabel("K(k)")
        ax_parameter.set_title("Quadrature covariance")
        ax_normal.set_xlabel("momentum k")
        ax_normal.set_ylabel("n(k)")
        ax_normal.set_title("Normal spectrum")
        ax_anomalous.set_xlabel("momentum k")
        ax_anomalous.set_ylabel("m(k)")
        ax_anomalous.set_title("Anomalous pairing spectrum")
        for axis in (ax_parameter, ax_normal, ax_anomalous):
            axis.legend(frameon=False, fontsize=8)
        for axis in axes.ravel():
            axis.grid(alpha=0.22)
        fig.suptitle("Nonlocal Luttinger liquid: genuine D x M cLETTA")
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return {"model": model, "states": states, "rows": rows}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fermi-velocity", type=float, default=1.0)
    parser.add_argument(
        "--interaction-decay-rates", nargs="+", type=float, default=[1.0]
    )
    parser.add_argument(
        "--interaction-strengths", nargs="+", type=float, default=[2.0]
    )
    parser.add_argument("--bond-dims", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-modes", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--quadrature-points", type=int, default=160)
    parser.add_argument("--validation-quadrature-points", type=int, default=420)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--initial-scale", type=float, default=0.25)
    parser.add_argument("--tie-scale", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--momentum-max", type=float, default=8.0)
    parser.add_argument("--momentum-points", type=int, default=401)
    parser.add_argument(
        "--output",
        default="/private/tmp/nonlocal_luttinger_matrix_cletta.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/nonlocal_luttinger_matrix_cletta.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
