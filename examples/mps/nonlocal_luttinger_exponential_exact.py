"""Exact infinite-continuum benchmark for a nonlocal Luttinger liquid.

The real-space density interaction is represented by exponential terms,

    V(x) = sum_j g_j exp(-kappa_j |x|),

which are the same kernel terms consumed by the continuum cLETTA examples.
The linearized spinless-fermion Hamiltonian is exactly diagonalized by a
momentum-dependent Bogoliubov rotation; no spatial grid or finite box is used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import ExponentialLuttingerModel, GaussianLuttingerCLETTA


def run(args):
    model = ExponentialLuttingerModel(
        decay_rates=args.decay_rates,
        strengths=args.strengths,
        fermi_velocity=args.fermi_velocity,
    )
    momentum = np.linspace(0.0, args.momentum_max, args.momentum_points)
    distance = np.linspace(0.0, args.distance_max, args.distance_points)
    interaction_k = model.interaction_momentum(momentum)
    parameter = model.luttinger_parameter(momentum)
    velocity = model.mode_velocity(momentum)
    dispersion = model.dispersion(momentum)
    structure = model.static_structure_factor(momentum)
    correlation = model.density_correlation(
        distance,
        uv_cutoff=args.uv_cutoff,
        points=args.correlation_points,
    )
    energy, quadrature_error = model.ground_state_energy_shift_density()

    cletta_states = [GaussianLuttingerCLETTA.optimize(model, num_modes=0)]
    for num_modes in range(1, args.max_aux + 1):
        state = GaussianLuttingerCLETTA.optimize(
            model,
            num_modes=num_modes,
            seed_states=[cletta_states[-1]],
            restarts=args.restarts,
            seed=args.seed + num_modes,
            maxiter=args.maxiter,
            quadrature_points=args.quadrature_points,
        )
        cletta_states.append(state)

    print("exact infinite-continuum nonlocal Luttinger liquid")
    print(
        "kernel terms: "
        + ", ".join(
            f"g={strength:g}, kappa={decay:g}"
            for decay, strength in zip(model.decay_rates, model.strengths)
        )
    )
    print(
        f"V(k=0)={model.interaction_momentum(0.0):.10f}  "
        f"K(0)={model.luttinger_parameter(0.0):.10f}  "
        f"u(0)={model.mode_velocity(0.0):.10f}"
    )
    print(
        f"Bogoliubov Delta E0/L={energy:.10f}  "
        f"quadrature_error={quadrature_error:.2e}"
    )
    for state in cletta_states:
        error = state.energy_shift_density - energy
        recovered = 1.0 - error / abs(energy)
        print(
            f"Gaussian cLETTA M={state.num_modes}: "
            f"Delta E0/L={state.energy_shift_density:.10f}  "
            f"error={error:.3e}  recovered={100.0 * recovered:.6f}%  "
            f"success={state.success}"
        )
        if state.num_modes:
            print(
                "  amplitudes=["
                + ";".join(f"{value:.10g}" for value in state.amplitudes)
                + "] rates=["
                + ";".join(f"{value:.10g}" for value in state.decay_rates)
                + "]"
            )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "momentum",
                    "interaction_momentum",
                    "luttinger_parameter",
                    "mode_velocity",
                    "dispersion",
                    "structure_factor",
                ]
                + [
                    key
                    for state in cletta_states
                    for key in (
                        f"cletta_M{state.num_modes}_K",
                        f"cletta_M{state.num_modes}_structure_factor",
                    )
                ],
            )
            writer.writeheader()
            for index in range(momentum.size):
                row = {
                        "momentum": momentum[index],
                        "interaction_momentum": interaction_k[index],
                        "luttinger_parameter": parameter[index],
                        "mode_velocity": velocity[index],
                        "dispersion": dispersion[index],
                        "structure_factor": structure[index],
                    }
                for state in cletta_states:
                    row[f"cletta_M{state.num_modes}_K"] = state.luttinger_parameter(
                        momentum[index]
                    )
                    row[
                        f"cletta_M{state.num_modes}_structure_factor"
                    ] = state.static_structure_factor(momentum[index])
                writer.writerow(row)
        print(f"wrote {output}")

    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "num_modes",
                    "energy_shift_density",
                    "exact_energy_shift_density",
                    "energy_error",
                    "fraction_recovered",
                    "amplitudes",
                    "decay_rates",
                    "success",
                    "nfev",
                    "nit",
                ],
            )
            writer.writeheader()
            for state in cletta_states:
                error = state.energy_shift_density - energy
                writer.writerow(
                    {
                        "num_modes": state.num_modes,
                        "energy_shift_density": state.energy_shift_density,
                        "exact_energy_shift_density": energy,
                        "energy_error": error,
                        "fraction_recovered": 1.0 - error / abs(energy),
                        "amplitudes": ";".join(map(str, state.amplitudes)),
                        "decay_rates": ";".join(map(str, state.decay_rates)),
                        "success": state.success,
                        "nfev": state.nfev,
                        "nit": state.nit,
                    }
                )
        print(f"wrote {output}")

    if args.correlation_output:
        output = Path(args.correlation_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["distance", "connected_density_correlation"])
            writer.writerows(zip(distance, correlation))
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
        ax_kernel, ax_modes, ax_structure, ax_correlation = axes.ravel()

        real_grid = np.linspace(0.0, args.distance_max, args.distance_points)
        ax_kernel.plot(real_grid, model.interaction_real_space(real_grid), color="#0072B2")
        ax_kernel.set_xlabel("distance x")
        ax_kernel.set_ylabel("V(x)")
        ax_kernel.set_title("Nonlocal interaction")

        ax_modes.plot(momentum, parameter, label="exact K(k)", color="black", linewidth=2.0)
        colors = ["#999999", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        for state, color in zip(cletta_states, colors):
            ax_modes.plot(
                momentum,
                state.luttinger_parameter(momentum),
                "--",
                color=color,
                label=f"cLETTA M={state.num_modes}",
            )
        ax_modes.set_xlabel("momentum k")
        ax_modes.set_ylabel("K(k)")
        ax_modes.set_title("Exact and finite-memory states")
        ax_modes.legend(frameon=False)

        mode_counts = np.array([state.num_modes for state in cletta_states])
        energies = np.array([state.energy_shift_density for state in cletta_states])
        energy_errors = np.maximum(energies - energy, 1.0e-15)
        ax_structure.semilogy(
            mode_counts,
            energy_errors,
            "o-",
            color="#CC79A7",
        )
        ax_structure.set_xticks(mode_counts)
        ax_structure.set_xlabel("number of auxiliary memories M")
        ax_structure.set_ylabel("variational energy error")
        ax_structure.set_title("Convergence to exact energy")

        for state, color in zip(cletta_states[1:], colors[1:]):
            error = np.abs(state.luttinger_parameter(momentum) - parameter)
            ax_correlation.semilogy(
                momentum[1:],
                np.maximum(error[1:], 1.0e-14),
                color=color,
                label=f"M={state.num_modes}",
            )
        ax_correlation.set_xlabel("momentum k")
        ax_correlation.set_ylabel("absolute K(k) error")
        ax_correlation.set_title("Finite-memory error")
        ax_correlation.legend(frameon=False)

        for axis in axes.ravel():
            axis.grid(alpha=0.22)
        fig.suptitle(
            f"Exponential nonlocal Luttinger liquid: Delta E0/L = {energy:.6f}"
        )
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return {
        "model": model,
        "energy_shift_density": energy,
        "quadrature_error": quadrature_error,
        "momentum": momentum,
        "luttinger_parameter": parameter,
        "mode_velocity": velocity,
        "structure_factor": structure,
        "distance": distance,
        "density_correlation": correlation,
        "cletta_states": cletta_states,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fermi-velocity", type=float, default=1.0)
    parser.add_argument("--decay-rates", nargs="+", type=float, default=[1.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[2.0])
    parser.add_argument("--momentum-max", type=float, default=10.0)
    parser.add_argument("--momentum-points", type=int, default=501)
    parser.add_argument("--distance-max", type=float, default=10.0)
    parser.add_argument("--distance-points", type=int, default=301)
    parser.add_argument("--uv-cutoff", type=float, default=8.0)
    parser.add_argument("--correlation-points", type=int, default=12000)
    parser.add_argument("--max-aux", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--quadrature-points", type=int, default=700)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--output",
        default="/private/tmp/nonlocal_luttinger_exponential_exact.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="/private/tmp/nonlocal_luttinger_cletta_summary.csv",
    )
    parser.add_argument(
        "--correlation-output",
        default="/private/tmp/nonlocal_luttinger_exponential_correlation.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/nonlocal_luttinger_exponential_exact.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
