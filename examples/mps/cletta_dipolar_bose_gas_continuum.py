"""Uniform 1D dipolar Bose gas with continuum cMPS and cLETTA.

The fixed-density Hamiltonian is

    H/L = <d psi^dag d psi>
          + int_0^inf dr V_d(r) < :n(r)n(0): >,

with the shifted-core dipolar interaction

    V_d(r) = C_d / (r + a)^3.

This regularization preserves the repulsive 1/r^3 tail and has the exact
Laplace representation

    1 / (r + a)^3 = 1/2 int_0^inf ds s^2 exp[-s(r+a)].

The finite exponential fit therefore controls only the Hamiltonian
contraction.  The cLETTA wavefunction memory poles remain independent
variational parameters.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ContinuousMPS,
    canonical_parameter_size,
    fit_exponential_kernel_nonlinear,
    pack_canonical_parameters,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def dipolar_kernel(distance, strength, softening):
    """Return the shifted-core repulsive dipolar interaction."""
    distance = np.asarray(distance, dtype=float)
    return float(strength) / (np.abs(distance) + float(softening)) ** 3


def product_state(density):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    return ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)


def apply_observables(state, values):
    state.energy = float(values["energy_density"])
    state.density = float(values["density"])
    state.kinetic = float(values["kinetic"])
    state.interaction = float(values["interaction"])
    state.raw_density = float(values["raw_density"])
    state.scale = float(values["scale"])
    return state


def pair_correlation(state, distances):
    """Evaluate $g_2(r)$ using biorthogonal transfer fixed points."""
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    left, right, dominant = state.dominant_fixed_points()
    transfer = state.transfer_matrix()
    insertion = np.kron(state.r, state.r.conj())
    scale = float(state.scale)
    density = scale * np.vdot(left, insertion @ right)
    shifted = scale * (transfer - dominant * np.eye(transfer.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eig(shifted)
    coefficients = np.linalg.solve(eigenvectors, insertion @ right)
    values = []
    for distance in distances:
        evolved = eigenvectors @ (np.exp(eigenvalues * distance) * coefficients)
        values.append(scale**2 * np.vdot(left, insertion @ evolved) / density**2)
    values = np.asarray(values)
    if np.max(np.abs(np.imag(values))) > 1.0e-10:
        raise ValueError("pair correlation has a significant imaginary component.")
    return np.real(values)


def attenuated_memory_seeds(parameters, bond_dim, num_modes):
    """Continue a shallower solution with several memory amplitudes."""
    parameters = np.asarray(parameters, dtype=float)
    base_size = canonical_parameter_size(int(bond_dim))
    tie_size = int(num_modes) * int(bond_dim) ** 2
    tie_slice = slice(base_size, base_size + tie_size)
    seeds = []
    for factor in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
        seed = np.array(parameters, copy=True)
        seed[tie_slice] *= factor
        seeds.append(seed)
    return seeds


def result_row(label, state, *, cutoff="", backend=""):
    rates = getattr(state, "cletta_decay_rates", None)
    frequencies = getattr(state, "cletta_frequencies", None)
    ties = getattr(state, "cletta_tie_matrices", None)
    return {
        "ansatz": label,
        "cutoff_L": cutoff,
        "contraction_backend": backend,
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success) if state.success is not None else True,
        "nfev": int(state.nfev),
        "message": str(state.message) if state.message is not None else "",
        "memory_rates": (
            "" if rates is None else ";".join(f"{value:.12g}" for value in rates)
        ),
        "memory_frequencies": (
            ""
            if frequencies is None
            else ";".join(f"{value:.12g}" for value in frequencies)
        ),
        "tie_norm": "" if ties is None else float(np.linalg.norm(ties)),
    }


def fit_dipolar_kernel(args):
    distances = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, args.fit_near_max, args.fit_near_points),
                np.geomspace(
                    args.fit_near_max / (args.fit_near_points - 1),
                    args.fit_rmax,
                    args.fit_tail_points,
                ),
            ]
        )
    )
    target = dipolar_kernel(distances, args.strength, args.softening)
    fit = fit_exponential_kernel_nonlinear(
        distances,
        target,
        rank=args.fit_rank,
        starts=args.fit_starts,
        max_nfev=args.fit_max_nfev,
    )
    exact_integral = args.strength / (2.0 * args.softening**2)
    fitted_integral = float(np.sum(fit["strengths"] / fit["decay_rates"]))
    fit["exact_integral"] = exact_integral
    fit["fitted_integral"] = fitted_integral
    fit["integral_relative_error"] = abs(fitted_integral - exact_integral) / abs(
        exact_integral
    )
    return fit


def run(args):
    fit = fit_dipolar_kernel(args)
    interaction_rates = fit["decay_rates"]
    interaction_weights = fit["strengths"]
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    seed_checkpoint = {}
    if args.seed_checkpoint:
        with np.load(args.seed_checkpoint) as saved:
            seed_checkpoint = {key: np.array(saved[key], copy=True) for key in saved.files}
        print(f"loaded seeds from {args.seed_checkpoint}")
    checkpoint_data = dict(seed_checkpoint)
    checkpoint_data.update(
        {
            "interaction_rates": interaction_rates,
            "interaction_strengths": interaction_weights,
        }
    )

    def save_checkpoint():
        np.savez(checkpoint_path, **checkpoint_data)

    def print_state(label, state):
        status = "converged" if state.success else "NOT CONVERGED"
        print(
            f"{label:>22s} E={state.energy:.10f} "
            f"T={state.kinetic:.10f} V={state.interaction:.10f} "
            f"{status} nfev={state.nfev}",
            flush=True,
        )

    print(
        f"dipolar fit rank={args.fit_rank} rel={fit['rel_error']:.3e} "
        f"rms_rel={fit['relative_rms_error']:.3e} "
        f"max_rel={fit['max_rel_error']:.3e} "
        f"integral_rel={fit['integral_relative_error']:.3e}"
    )

    rows = []
    states = []
    product = product_state(args.density)
    apply_observables(
        product,
        product.exponential_bose_gas_fixed_density_observables(
            decay_rates=interaction_rates,
            strengths=interaction_weights,
            density=args.density,
            connected=False,
        ),
    )
    product.success = True
    product.nfev = 0
    product.message = "analytic product state"
    rows.append(result_row("product", product))
    states.append(("product", product))
    print_state("product", product)

    cmps_states = {}
    for index, bond_dim in enumerate(dict.fromkeys(args.cmps_bond_dims)):
        seed_key = f"cmps_D{bond_dim}_theta"
        state = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
            bond_dim=bond_dim,
            decay_rates=interaction_rates,
            strengths=interaction_weights,
            density=args.density,
            connected=False,
            seed_thetas=(
                [seed_checkpoint[seed_key]] if seed_key in seed_checkpoint else ()
            ),
            restarts=args.cmps_restarts,
            seed=args.seed + 101 + 19 * index,
            maxiter=args.cmps_maxiter,
            maxfun=args.cmps_maxfun,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
        )
        cmps_states[bond_dim] = state
        rows.append(result_row(f"cMPS-D{bond_dim}", state))
        states.append((f"cMPS $D={bond_dim}$", state))
        checkpoint_data[seed_key] = state.theta
        save_checkpoint()
        print_state(f"cMPS-D{bond_dim}", state)

    previous_parameters = []
    for cutoff in args.cutoffs:
        seed_key = f"cletta_D{args.bond_dim}_M{args.num_modes}_L{cutoff}_parameters"
        checkpoint_seeds = []
        if seed_key in seed_checkpoint:
            checkpoint_seeds.append(seed_checkpoint[seed_key])
        for previous_cutoff in range(cutoff - 1, 0, -1):
            previous_key = (
                f"cletta_D{args.bond_dim}_M{args.num_modes}_"
                f"L{previous_cutoff}_parameters"
            )
            if previous_key in seed_checkpoint:
                checkpoint_seeds.append(seed_checkpoint[previous_key])
                break
        direct_seeds = previous_parameters + checkpoint_seeds
        continuation_seeds = []
        for parameters in direct_seeds:
            continuation_seeds.extend(
                attenuated_memory_seeds(
                    parameters,
                    args.bond_dim,
                    args.num_modes,
                )
            )
        state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=interaction_rates,
            strengths=interaction_weights,
            density=args.density,
            connected=False,
            num_modes=args.num_modes,
            depth=cutoff,
            seed_parameters=continuation_seeds,
            seed_base_thetas=(
                [cmps_states[args.bond_dim].theta]
                if args.seed_from_cmps and args.bond_dim in cmps_states
                else ()
            ),
            restarts=args.restarts,
            seed=args.seed + 211 + 23 * cutoff,
            maxiter=args.maxiter,
            regularization=args.cletta_regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            rate_bounds=(args.min_memory_decay, args.max_memory_decay),
            frequency_bounds=(-args.max_memory_frequency, args.max_memory_frequency),
            tie_scale=args.tie_scale,
            contraction_backend=args.contraction_backend,
            eigensolver=args.eigensolver,
            eigen_iterations=args.eigen_iterations,
        )
        values = state.cletta_base.cletta_exponential_bose_gas_fixed_density_observables(
            state.cletta_tie_matrices,
            state.cletta_decay_rates,
            interaction_decay_rates=interaction_rates,
            strengths=interaction_weights,
            density=args.density,
            depth=cutoff,
            frequencies=state.cletta_frequencies,
            connected=False,
            contraction_backend=args.validation_backend,
            iterative_tolerance=args.iterative_tolerance,
            iterative_maxiter=args.iterative_maxiter,
        )
        apply_observables(state, values)
        previous_parameters = [state.cletta_parameters]
        label = f"cLETTA-D{args.bond_dim}-M{args.num_modes}-L{cutoff}"
        rows.append(
            result_row(
                label,
                state,
                cutoff=cutoff,
                backend=args.validation_backend,
            )
        )
        states.append((rf"cLETTA $D={args.bond_dim},M={args.num_modes},L={cutoff}$", state))
        checkpoint_data[seed_key] = state.cletta_parameters
        save_checkpoint()
        print_state(label, state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fit_fields = {
        "fit_relative_error": fit["rel_error"],
        "fit_relative_rms_error": fit["relative_rms_error"],
        "fit_max_relative_error": fit["max_rel_error"],
        "fit_integral_relative_error": fit["integral_relative_error"],
        "fit_rates": ";".join(f"{value:.12g}" for value in interaction_rates),
        "fit_strengths": ";".join(f"{value:.12g}" for value in interaction_weights),
        "dipolar_strength": args.strength,
        "softening": args.softening,
        "density": args.density,
    }
    fieldnames = list(rows[0]) + list(fit_fields)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **fit_fields})
    print(f"wrote {output}")

    distance = np.linspace(0.0, args.correlation_range, args.correlation_points)
    correlations = []
    for label, state in states[1:]:
        try:
            correlations.append((label, pair_correlation(state, distance)))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            print(f"correlation failed for {label}: {exc}")

    correlation_output = Path(args.correlation_output)
    correlation_output.parent.mkdir(parents=True, exist_ok=True)
    with correlation_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ansatz", "distance", "g2"))
        for (label, values) in correlations:
            writer.writerows(
                (label, float(r), float(value))
                for r, value in zip(distance, values)
            )
    print(f"wrote {correlation_output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

        grid = np.unique(
            np.concatenate(
                [np.linspace(0.0, 2.0, 300), np.geomspace(0.02, args.fit_rmax, 500)]
            )
        )
        exact = dipolar_kernel(grid, args.strength, args.softening)
        fitted = np.exp(-grid[:, None] * interaction_rates) @ interaction_weights
        axes[0].loglog(grid[1:], exact[1:], color="black", label="dipolar")
        axes[0].loglog(grid[1:], fitted[1:], "--", color="#0072B2", label="exponential fit")
        axes[0].set(xlabel=r"$r$", ylabel=r"$V_d(r)$", title="Hamiltonian kernel")
        axes[0].legend(frameon=False)

        labels = [row["ansatz"] for row in rows[1:]]
        energies = [row["energy"] for row in rows[1:]]
        colors = ["#0072B2" if label.startswith("cMPS") else "#D55E00" for label in labels]
        axes[1].bar(np.arange(len(labels)), energies, color=colors)
        axes[1].set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axes[1].set(ylabel="energy density", title="Variational comparison")

        for label, values in correlations:
            axes[2].plot(distance, values, label=label)
        axes[2].axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
        axes[2].set(
            xlabel=r"$r$",
            ylabel=r"$g_2(r)$",
            title="Pair correlations",
            xlim=(0.0, args.correlation_range),
        )
        axes[2].legend(frameon=False, fontsize=7)
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure, dpi=220)
        print(f"wrote {figure}")

    return rows, fit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--softening", type=float, default=0.5)
    parser.add_argument("--fit-near-max", type=float, default=2.0)
    parser.add_argument("--fit-near-points", type=int, default=121)
    parser.add_argument("--fit-rmax", type=float, default=32.0)
    parser.add_argument("--fit-tail-points", type=int, default=300)
    parser.add_argument("--fit-rank", type=int, default=10)
    parser.add_argument("--fit-starts", type=int, default=3)
    parser.add_argument("--fit-max-nfev", type=int, default=8000)
    parser.add_argument("--cmps-bond-dims", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--cmps-restarts", type=int, default=6)
    parser.add_argument("--cmps-maxiter", type=int, default=600)
    parser.add_argument("--cmps-maxfun", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="cLETTA hierarchy depths; pass the flag with no values for cMPS only",
    )
    parser.add_argument("--seed-from-cmps", action="store_true", default=True)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--regularization", type=float, default=1.0e-11)
    parser.add_argument("--cletta-regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-3)
    parser.add_argument("--tie-scale", type=float, default=0.02)
    parser.add_argument("--min-memory-decay", type=float, default=0.02)
    parser.add_argument("--max-memory-decay", type=float, default=20.0)
    parser.add_argument("--max-memory-frequency", type=float, default=12.0)
    parser.add_argument(
        "--contraction-backend",
        choices=("explicit", "hierarchy", "heom"),
        default="explicit",
    )
    parser.add_argument(
        "--validation-backend",
        choices=("explicit", "hierarchy", "heom", "hierarchy_iterative"),
        default="hierarchy",
    )
    parser.add_argument("--eigensolver", choices=("auto", "dense", "iterative"), default="auto")
    parser.add_argument("--eigen-iterations", type=int, default=512)
    parser.add_argument("--iterative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--iterative-maxiter", type=int, default=600)
    parser.add_argument("--correlation-range", type=float, default=8.0)
    parser.add_argument("--correlation-points", type=int, default=321)
    parser.add_argument(
        "--output",
        default=str(RESULTS / "cletta_dipolar_bose_gas.csv"),
    )
    parser.add_argument(
        "--correlation-output",
        default=str(RESULTS / "cletta_dipolar_bose_gas_correlations.csv"),
    )
    parser.add_argument(
        "--figure",
        default=str(RESULTS / "cletta_dipolar_bose_gas.png"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(RESULTS / "cletta_dipolar_bose_gas_states.npz"),
    )
    parser.add_argument("--seed-checkpoint", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
