"""Background-subtracted cMPS/cLETTA benchmark on a Fourier-truncated cylinder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    commuting_cylinder_parameter_size,
    fit_exponential_kernel_nonlinear,
    optimize_cylinder_cletta,
    optimize_cylinder_cmps,
    softened_yukawa_cylinder_fourier,
)


def _fit_cylinder_kernels(args, modes):
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
    transfers = np.arange(0, int(np.ptp(modes)) + 1)
    exact = softened_yukawa_cylinder_fourier(
        distances,
        transfers,
        circumference=args.circumference,
        strength=args.strength,
        screening=args.screening,
        softening=args.softening,
        quadrature_points=args.transverse_quadrature,
    )
    fits = {}
    kernels = {}
    for transfer in transfers:
        transverse_momentum = 2.0 * np.pi * transfer / args.circumference
        asymptotic_rate = np.sqrt(args.screening**2 + transverse_momentum**2)
        relative = asymptotic_rate * args.fit_rmax <= args.relative_fit_span
        fit_rank = args.fit_rank if relative else min(args.fit_rank, 6)
        fit = fit_exponential_kernel_nonlinear(
            distances,
            exact[int(transfer)],
            rank=fit_rank,
            relative=relative,
            rate_offset=asymptotic_rate,
            starts=args.fit_starts,
            max_nfev=args.fit_max_nfev,
            amplitude_regularization=(
                0.0 if relative else args.short_range_fit_regularization
            ),
        )
        fits[int(transfer)] = fit
        kernels[int(transfer)] = (fit["decay_rates"], fit["strengths"])
        print(
            f"q={transfer} k={transverse_momentum:.6f} "
            f"fit={'relative' if relative else 'absolute'} rank={fit_rank} "
            f"rel_L2={fit['rel_error']:.3e} max_abs={fit['max_abs_error']:.3e}"
        )
    return distances, exact, fits, kernels


def _row(label, state, parameter_count, *, method, memory_modes="", depth=""):
    return {
        "ansatz": label,
        "method": method,
        "memory_modes": memory_modes,
        "depth": depth,
        "parameter_count": int(parameter_count),
        "effective_bond_dim": int(state.bond_dim),
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "axial_kinetic": float(state.axial_kinetic),
        "transverse_kinetic": float(state.transverse_kinetic),
        "interaction": float(state.interaction),
        "field_densities": ";".join(f"{value:.12g}" for value in state.field_densities),
        "channel_interactions": ";".join(
            f"q{transfer}:{value:.12g}"
            for transfer, value in sorted(state.channel_interactions.items())
        ),
        "memory_rates": ""
        if state.cletta_decay_rates is None
        else ";".join(f"{value:.12g}" for value in state.cletta_decay_rates),
        "tie_norm": ""
        if state.cletta_tie_matrices is None
        else float(np.linalg.norm(state.cletta_tie_matrices)),
        "success": bool(state.success),
        "message": str(state.message),
        "iterations": int(state.nit),
        "jacobian_norm": float(state.jacobian_norm),
        "nfev": int(state.nfev),
    }


def run(args):
    modes = np.arange(-args.mode_cutoff, args.mode_cutoff + 1, dtype=int)
    transverse_momenta = 2.0 * np.pi * modes / args.circumference
    distances, exact_kernels, fits, kernels = _fit_cylinder_kernels(args, modes)

    print(
        f"infinite cylinder Ly={args.circumference:g} modes={modes.tolist()} "
        f"n1D={args.density:g} n2D={args.density / args.circumference:g} "
        f"g={args.strength:g} a={args.softening:g} kappa={args.screening:g} "
        "interaction=connected"
    )
    rows = []
    states = {}
    cmps_states = {}
    for index, bond_dim in enumerate(dict.fromkeys(args.cmps_bond_dims)):
        state = optimize_cylinder_cmps(
            bond_dim=bond_dim,
            mode_numbers=modes,
            transverse_momenta=transverse_momenta,
            interaction_kernels=kernels,
            circumference=args.circumference,
            density=args.density,
            restarts=args.cmps_restarts,
            seed=args.seed + 31 * index,
            maxiter=args.cmps_maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            workers=args.optimizer_workers,
        )
        cmps_states[int(bond_dim)] = state
        count = commuting_cylinder_parameter_size(bond_dim, modes.size)
        rows.append(_row(f"cMPS-D{bond_dim}", state, count, method="cMPS"))
        states[f"cMPS-D{bond_dim}"] = state

    base = cmps_states[args.cletta_bond_dim]
    previous_memory_state = None
    for memory_modes in args.memory_modes:
        seeds = []
        if (
            previous_memory_state is not None
            and memory_modes > previous_memory_state.cletta_tie_matrices.shape[0]
        ):
            old_ties = previous_memory_state.cletta_tie_matrices
            old_rates = previous_memory_state.cletta_decay_rates
            ties = np.zeros((memory_modes, args.cletta_bond_dim))
            rates = np.geomspace(
                args.min_memory_decay, args.max_memory_decay, memory_modes
            )
            ties[: old_ties.shape[0]] = previous_memory_state.cletta_tie_coefficients
            rates[: old_rates.size] = old_rates
            seeds.append(
                np.concatenate(
                    [
                        previous_memory_state.cletta_base.cylinder_parameters,
                        ties.reshape(-1),
                        np.log(rates),
                    ]
                )
            )
        for depth in args.depths:
            state = optimize_cylinder_cletta(
                bond_dim=args.cletta_bond_dim,
                mode_numbers=modes,
                transverse_momenta=transverse_momenta,
                interaction_kernels=kernels,
                circumference=args.circumference,
                density=args.density,
                num_memory_modes=memory_modes,
                depth=depth,
                seed_base_parameters=[base.cylinder_parameters],
                seed_parameters=seeds,
                restarts=args.cletta_restarts,
                seed=args.seed + 101 * memory_modes + 17 * depth,
                maxiter=(
                    args.cletta_maxiter
                    if depth == args.depths[0]
                    else args.deeper_cletta_maxiter
                ),
                regularization=args.cletta_regularization,
                density_gauge_penalty=args.density_gauge_penalty,
                rate_bounds=(args.min_memory_decay, args.max_memory_decay),
                tie_scale=args.tie_scale,
                workers=args.optimizer_workers,
                use_jax=args.cletta_gradient == "jax",
                eigensolver=args.cletta_eigensolver,
                eigen_iterations=args.cletta_eigen_iterations,
                linear_solver=args.cletta_linear_solver,
            )
            seeds = [state.cletta_parameters]
            base_count = commuting_cylinder_parameter_size(
                args.cletta_bond_dim, modes.size
            )
            count = base_count + memory_modes * args.cletta_bond_dim + memory_modes
            rows.append(
                _row(
                    f"cLETTA-D{args.cletta_bond_dim}-M{memory_modes}-L{depth}",
                    state,
                    count,
                    method="cLETTA",
                    memory_modes=memory_modes,
                    depth=depth,
                )
            )
            states[f"cLETTA-D{args.cletta_bond_dim}-M{memory_modes}-L{depth}"] = state
            previous_memory_state = state

    for row in rows:
        print(
            f"{row['ansatz']:>20s} p={row['parameter_count']:2d} "
            f"Deff={row['effective_bond_dim']:2d} E={row['energy']:.10f} "
            f"Tx={row['axial_kinetic']:.8f} Ty={row['transverse_kinetic']:.8f} "
            f"V={row['interaction']:.10f} success={row['success']} "
            f"nit={row['iterations']} |jac|={row['jacobian_norm']:.3e}"
        )
        print(f"  occupations=[{row['field_densities']}]")
        print(f"  sectors=[{row['channel_interactions']}]")
        if row["memory_rates"]:
            print(
                f"  memory rates=[{row['memory_rates']}] tie_norm={row['tie_norm']}"
            )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for label, state in states.items():
            key = label.replace("-", "_")
            arrays[f"{key}__q"] = np.asarray(state.q)
            arrays[f"{key}__r"] = np.asarray(state.r_ops)
            arrays[f"{key}__parameters"] = np.asarray(
                state.cletta_parameters
                if state.cletta_parameters is not None
                else state.cylinder_parameters
            )
            arrays[f"{key}__canonical"] = np.asarray(
                state.cletta_parameters is None
            )
        np.savez_compressed(checkpoint, **arrays)
        print(f"wrote {checkpoint}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_energy, ax_kernel) = plt.subplots(1, 2, figsize=(10.2, 4.0))
        labels = [row["ansatz"] for row in rows]
        energies = np.array([row["energy"] for row in rows])
        x = np.arange(len(rows))
        colors = ["#4477aa" if row["method"] == "cMPS" else "#aa3377" for row in rows]
        ax_energy.bar(x, energies, color=colors)
        ax_energy.plot(x, energies, "ko-", lw=1, ms=4)
        ax_energy.set_xticks(x, labels, rotation=22, ha="right")
        ax_energy.set_ylabel("energy per axial length")
        ax_energy.grid(axis="y", alpha=0.25)

        for transfer, values in exact_kernels.items():
            fitted = np.exp(
                -distances[:, None] * fits[transfer]["decay_rates"][None, :]
            ) @ fits[transfer]["strengths"]
            line = ax_kernel.semilogy(
                distances,
                np.maximum(values, 1.0e-15),
                label=fr"$V_{{{transfer}}}$",
            )[0]
            ax_kernel.semilogy(
                distances,
                np.maximum(fitted, 1.0e-15),
                "--",
                color=line.get_color(),
            )
        ax_kernel.set_xlabel("axial separation x")
        ax_kernel.set_ylabel(r"cylinder Fourier kernel $V_q(x)$")
        ax_kernel.set_xlim(0.0, args.fit_rmax)
        ax_kernel.grid(alpha=0.25)
        ax_kernel.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows, fits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circumference", type=float, default=8.0)
    parser.add_argument("--mode-cutoff", type=int, default=1)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=10.0)
    parser.add_argument("--screening", type=float, default=0.2)
    parser.add_argument("--softening", type=float, default=0.5)
    parser.add_argument("--transverse-quadrature", type=int, default=1000)
    parser.add_argument("--fit-near-max", type=float, default=2.0)
    parser.add_argument("--fit-near-points", type=int, default=81)
    parser.add_argument("--fit-rmax", type=float, default=24.0)
    parser.add_argument("--fit-tail-points", type=int, default=180)
    parser.add_argument("--fit-rank", type=int, default=8)
    parser.add_argument("--fit-starts", type=int, default=1)
    parser.add_argument("--fit-max-nfev", type=int, default=5000)
    parser.add_argument("--relative-fit-span", type=float, default=25.0)
    parser.add_argument("--short-range-fit-regularization", type=float, default=1.0e-10)
    parser.add_argument("--cmps-bond-dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--cmps-restarts", type=int, default=10)
    parser.add_argument("--cmps-maxiter", type=int, default=500)
    parser.add_argument("--cletta-bond-dim", type=int, default=2)
    parser.add_argument("--memory-modes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=[1])
    parser.add_argument("--cletta-restarts", type=int, default=4)
    parser.add_argument("--cletta-maxiter", type=int, default=500)
    parser.add_argument("--deeper-cletta-maxiter", type=int, default=240)
    parser.add_argument("--min-memory-decay", type=float, default=0.05)
    parser.add_argument("--max-memory-decay", type=float, default=5.0)
    parser.add_argument("--tie-scale", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--cletta-regularization", type=float, default=1.0e-7)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument(
        "--cletta-gradient",
        choices=("jax", "finite-difference"),
        default="jax",
    )
    parser.add_argument(
        "--cletta-eigensolver",
        choices=("auto", "dense", "iterative"),
        default="auto",
    )
    parser.add_argument("--cletta-eigen-iterations", type=int, default=256)
    parser.add_argument(
        "--cletta-linear-solver",
        choices=("auto", "dense", "iterative"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint",
        default="/private/tmp/cletta_softened_screened_coulomb_cylinder.npz",
    )
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_softened_screened_coulomb_cylinder.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_softened_screened_coulomb_cylinder.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
