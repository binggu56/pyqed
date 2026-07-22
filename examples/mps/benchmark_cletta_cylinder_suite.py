"""Convergence and resource study for cMPS/cLETTA on an infinite cylinder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    commuting_cylinder_parameter_size,
    cylinder_density_mode_correlation,
    cylinder_static_structure_factor,
    fit_exponential_kernel_nonlinear,
    optimize_cylinder_cletta,
    optimize_cylinder_cmps,
    softened_yukawa_cylinder_fourier,
)


def fit_kernels(mode_cutoff, args):
    modes = np.arange(-mode_cutoff, mode_cutoff + 1, dtype=int)
    distances = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 2.0, 41),
                np.geomspace(0.025, 24.0, 80),
            ]
        )
    )
    transfers = np.arange(2 * mode_cutoff + 1)
    exact = softened_yukawa_cylinder_fourier(
        distances,
        transfers,
        circumference=args.circumference,
        strength=1.0,
        screening=args.screening,
        softening=args.softening,
        quadrature_points=args.transverse_quadrature,
    )
    kernels = {}
    errors = {}
    for transfer in transfers:
        momentum = 2.0 * np.pi * transfer / args.circumference
        asymptotic_rate = np.sqrt(args.screening**2 + momentum**2)
        relative = asymptotic_rate * 24.0 <= 25.0
        fit = fit_exponential_kernel_nonlinear(
            distances,
            exact[int(transfer)],
            rank=args.fit_rank if relative else min(args.fit_rank, 6),
            relative=relative,
            rate_offset=asymptotic_rate,
            starts=2,
            max_nfev=3000,
            amplitude_regularization=0.0 if relative else 1.0e-10,
        )
        kernels[int(transfer)] = (
            fit["decay_rates"],
            args.strength * fit["strengths"],
        )
        errors[int(transfer)] = float(fit["rel_error"])
    return modes, 2.0 * np.pi * modes / args.circumference, kernels, errors


def parameter_count(state, modes, memory_modes=0):
    base = commuting_cylinder_parameter_size(
        state.cletta_base.bond_dim if memory_modes else state.bond_dim,
        len(modes),
    )
    if not memory_modes:
        return base
    return base + memory_modes * state.cletta_base.bond_dim + memory_modes


def result_row(study, coordinate, label, state, modes, memory_modes=0, depth=0):
    return {
        "study": study,
        "coordinate": float(coordinate),
        "label": label,
        "base_bond_dim": int(
            state.cletta_base.bond_dim if memory_modes else state.bond_dim
        ),
        "memory_modes": int(memory_modes),
        "depth": int(depth),
        "mode_cutoff": int(np.max(np.abs(modes))),
        "parameter_count": parameter_count(state, modes, memory_modes),
        "effective_bond_dim": int(state.bond_dim),
        "energy": float(state.energy),
        "axial_kinetic": float(state.axial_kinetic),
        "transverse_kinetic": float(state.transverse_kinetic),
        "interaction": float(state.interaction),
        "jacobian_norm": float(state.jacobian_norm),
        "iterations": int(state.nit),
        "success": bool(state.success),
    }


def expand_memory_seed(state, new_modes):
    old_coefficients = state.cletta_tie_coefficients
    old_rates = state.cletta_decay_rates
    bond_dim = state.cletta_base.bond_dim
    coefficients = np.zeros((new_modes, bond_dim))
    coefficients[: old_coefficients.shape[0]] = old_coefficients
    rates = np.geomspace(0.5, 2.0, new_modes)
    rates[: old_rates.size] = old_rates
    return np.concatenate(
        [
            state.cletta_base.cylinder_parameters,
            coefficients.reshape(-1),
            np.log(rates),
        ]
    )


def expand_mode_seed(state, old_modes, new_modes):
    old_modes = np.asarray(old_modes, dtype=int)
    new_modes = np.asarray(new_modes, dtype=int)
    bond_dim = state.bond_dim
    skew_size = bond_dim * (bond_dim - 1) // 2
    prefix_size = skew_size + bond_dim * bond_dim
    parameters = np.asarray(state.cylinder_parameters)
    old_coefficients = parameters[prefix_size:].reshape(len(old_modes) - 1, bond_dim)
    by_mode = {}
    index = 0
    for mode in old_modes:
        if mode == 0:
            continue
        by_mode[int(mode)] = old_coefficients[index]
        index += 1
    new_coefficients = [
        by_mode.get(int(mode), np.zeros(bond_dim))
        for mode in new_modes
        if mode != 0
    ]
    return np.concatenate([parameters[:prefix_size], np.asarray(new_coefficients).reshape(-1)])


def expand_cletta_mode_seed(state, old_modes, new_modes):
    base = expand_mode_seed(state.cletta_base, old_modes, new_modes)
    return np.concatenate(
        [
            base,
            state.cletta_tie_coefficients.reshape(-1),
            np.log(state.cletta_decay_rates),
        ]
    )


def optimize_cmps(args, bond_dim, modes, momenta, kernels, *, seeds=(), seed=0):
    return optimize_cylinder_cmps(
        bond_dim=bond_dim,
        mode_numbers=modes,
        transverse_momenta=momenta,
        interaction_kernels=kernels,
        circumference=args.circumference,
        density=args.density,
        seed_parameters=seeds,
        restarts=args.restarts,
        seed=args.seed + seed,
        maxiter=args.maxiter,
        workers=args.workers,
    )


def optimize_cletta(
    args,
    bond_dim,
    memory_modes,
    depth,
    modes,
    momenta,
    kernels,
    base,
    *,
    seeds=(),
    seed=0,
):
    return optimize_cylinder_cletta(
        bond_dim=bond_dim,
        mode_numbers=modes,
        transverse_momenta=momenta,
        interaction_kernels=kernels,
        circumference=args.circumference,
        density=args.density,
        num_memory_modes=memory_modes,
        depth=depth,
        seed_base_parameters=[base.cylinder_parameters],
        seed_parameters=seeds,
        restarts=args.cletta_restarts,
        seed=args.seed + seed,
        maxiter=args.maxiter,
        workers=args.workers,
    )


def run_cutoff_only(args):
    rows = []
    previous_modes = None
    previous_base = None
    previous_state = None
    checkpoints = {}
    for cutoff in range(0, args.max_mode_cutoff + 1):
        modes, momenta, kernels, _errors = fit_kernels(cutoff, args)
        base_seeds = (
            []
            if previous_base is None
            else [expand_mode_seed(previous_base, previous_modes, modes)]
        )
        base = optimize_cmps(
            args,
            2,
            modes,
            momenta,
            kernels,
            seeds=base_seeds,
            seed=1500 + cutoff,
        )
        cletta_seeds = (
            []
            if previous_state is None
            else [expand_cletta_mode_seed(previous_state, previous_modes, modes)]
        )
        state = optimize_cletta(
            args,
            2,
            2,
            1,
            modes,
            momenta,
            kernels,
            base,
            seeds=cletta_seeds,
            seed=1600 + cutoff,
        )
        if previous_base is not None and base.energy > previous_base.energy + 1.0e-7:
            raise RuntimeError("cMPS transverse-cutoff energy violates nested monotonicity.")
        if previous_state is not None and state.energy > previous_state.energy + 1.0e-7:
            raise RuntimeError("cLETTA transverse-cutoff energy violates nested monotonicity.")
        rows.append(result_row("cutoff-cmps", cutoff, f"cMPS-m{cutoff}", base, modes))
        rows.append(result_row("cutoff-cletta", cutoff, f"cLETTA-m{cutoff}", state, modes, 2, 1))
        checkpoints[f"cMPS-m{cutoff}"] = base
        checkpoints[f"cLETTA-m{cutoff}"] = state
        print("cutoff", cutoff, base.energy, state.energy, flush=True)
        previous_modes = modes
        previous_base = base
        previous_state = state

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {}
    for label, state in checkpoints.items():
        key = label.replace("-", "_")
        arrays[f"{key}__q"] = state.q
        arrays[f"{key}__r"] = np.asarray(state.r_ops)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    print("wrote", output, flush=True)
    return rows


def run(args):
    if args.cutoff_only:
        return run_cutoff_only(args)
    rows = []
    checkpoints = {}
    modes, momenta, kernels, fit_errors = fit_kernels(1, args)

    # Resource and parameter-matched comparisons.
    cmps = {}
    for bond_dim in (2, 3, 4):
        state = optimize_cmps(
            args, bond_dim, modes, momenta, kernels, seed=100 * bond_dim
        )
        cmps[bond_dim] = state
        label = f"cMPS-D{bond_dim}"
        rows.append(result_row("resources", bond_dim, label, state, modes))
        checkpoints[label] = state
        print(label, state.energy, state.jacobian_norm, flush=True)

    d2_m2 = optimize_cletta(
        args, 2, 2, 1, modes, momenta, kernels, cmps[2], seed=220
    )
    rows.append(result_row("resources", 2.2, "cLETTA-D2-M2-L1", d2_m2, modes, 2, 1))
    checkpoints["cLETTA-D2-M2-L1"] = d2_m2
    print("cLETTA-D2-M2-L1", d2_m2.energy, d2_m2.jacobian_norm, flush=True)

    d2_m3 = optimize_cletta(
        args,
        2,
        3,
        1,
        modes,
        momenta,
        kernels,
        cmps[2],
        seeds=[expand_memory_seed(d2_m2, 3)],
        seed=230,
    )
    rows.append(result_row("resources", 2.3, "cLETTA-D2-M3-L1", d2_m3, modes, 3, 1))
    checkpoints["cLETTA-D2-M3-L1"] = d2_m3
    print("cLETTA-D2-M3-L1", d2_m3.energy, d2_m3.jacobian_norm, flush=True)

    d3_m2 = optimize_cletta(
        args, 3, 2, 1, modes, momenta, kernels, cmps[3], seed=320
    )
    rows.append(result_row("resources", 3.2, "cLETTA-D3-M2-L1", d3_m2, modes, 2, 1))
    checkpoints["cLETTA-D3-M2-L1"] = d3_m2
    print("cLETTA-D3-M2-L1", d3_m2.energy, d3_m2.jacobian_norm, flush=True)

    # Hierarchy convergence for one memory channel.
    previous = None
    for depth in range(1, args.max_depth + 1):
        state = optimize_cletta(
            args,
            2,
            1,
            depth,
            modes,
            momenta,
            kernels,
            cmps[2],
            seeds=[] if previous is None else [previous.cletta_parameters],
            seed=400 + depth,
        )
        rows.append(result_row("depth", depth, f"cLETTA-L{depth}", state, modes, 1, depth))
        print("depth", depth, state.energy, state.jacobian_norm, flush=True)
        previous = state

    # Transverse-mode convergence at D=2, M=2, L=1.
    cutoff_states = {}
    for cutoff in range(0, args.max_mode_cutoff + 1):
        cutoff_modes, cutoff_momenta, cutoff_kernels, cutoff_errors = fit_kernels(
            cutoff, args
        )
        base = optimize_cmps(
            args,
            2,
            cutoff_modes,
            cutoff_momenta,
            cutoff_kernels,
            seed=500 + cutoff,
        )
        state = optimize_cletta(
            args,
            2,
            2,
            1,
            cutoff_modes,
            cutoff_momenta,
            cutoff_kernels,
            base,
            seed=600 + cutoff,
        )
        rows.append(result_row("cutoff-cmps", cutoff, f"cMPS-m{cutoff}", base, cutoff_modes))
        rows.append(result_row("cutoff-cletta", cutoff, f"cLETTA-m{cutoff}", state, cutoff_modes, 2, 1))
        cutoff_states[cutoff] = (base, state)
        fit_errors.update(
            {f"m{cutoff}_q{key}": value for key, value in cutoff_errors.items()}
        )
        print("cutoff", cutoff, base.energy, state.energy, flush=True)

    # Coupling scan with continuation in g.
    unit_kernels = {
        transfer: (rates, strengths / args.strength)
        for transfer, (rates, strengths) in kernels.items()
    }
    previous_cmps = None
    previous_cletta = None
    for index, coupling in enumerate(args.couplings):
        scan_kernels = {
            transfer: (rates, coupling * strengths)
            for transfer, (rates, strengths) in unit_kernels.items()
        }
        base = optimize_cmps(
            args,
            2,
            modes,
            momenta,
            scan_kernels,
            seeds=[] if previous_cmps is None else [previous_cmps.cylinder_parameters],
            seed=700 + index,
        )
        state = optimize_cletta(
            args,
            2,
            2,
            1,
            modes,
            momenta,
            scan_kernels,
            base,
            seeds=[] if previous_cletta is None else [previous_cletta.cletta_parameters],
            seed=800 + index,
        )
        rows.append(result_row("coupling-cmps", coupling, f"cMPS-g{coupling:g}", base, modes))
        rows.append(result_row("coupling-cletta", coupling, f"cLETTA-g{coupling:g}", state, modes, 2, 1))
        print("coupling", coupling, base.energy, state.energy, flush=True)
        previous_cmps = base
        previous_cletta = state

    # Correlations for the exactly parameter-matched pair: 18 parameters each.
    correlation_rows = []
    distances = np.linspace(0.0, args.correlation_rmax, args.correlation_points)
    momenta_x = np.linspace(0.0, args.structure_kmax, args.structure_points)
    selected = {
        "cMPS-D3": (cmps[3], True),
        "cLETTA-D2-M3-L1": (d2_m3, False),
        "cLETTA-D3-M2-L1": (d3_m2, False),
    }
    for label, (state, canonical) in selected.items():
        for transfer in (0, 1):
            correlation = cylinder_density_mode_correlation(
                state,
                distances,
                mode_numbers=modes,
                transfer=transfer,
                density=args.density,
                connected=True,
                canonical=canonical,
            )
            structure = cylinder_static_structure_factor(
                state,
                momenta_x,
                mode_numbers=modes,
                transfer=transfer,
                density=args.density,
                canonical=canonical,
            )
            for distance, value in zip(distances, correlation):
                correlation_rows.append(
                    {
                        "kind": "correlation",
                        "label": label,
                        "transfer": transfer,
                        "coordinate": distance,
                        "value": float(np.real(value)),
                    }
                )
            for momentum, value in zip(momenta_x, structure):
                correlation_rows.append(
                    {
                        "kind": "structure",
                        "label": label,
                        "transfer": transfer,
                        "coordinate": momentum,
                        "value": float(np.real(value)),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    correlation_output = output.with_name(output.stem + "_correlations.csv")
    with correlation_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)
    arrays = {}
    for label, state in checkpoints.items():
        key = label.replace("-", "_")
        arrays[f"{key}__q"] = state.q
        arrays[f"{key}__r"] = np.asarray(state.r_ops)
    arrays["fit_errors"] = np.asarray(list(fit_errors.items()), dtype=object)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    print("wrote", output, correlation_output, flush=True)
    return rows, correlation_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circumference", type=float, default=8.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=10.0)
    parser.add_argument("--screening", type=float, default=0.2)
    parser.add_argument("--softening", type=float, default=0.5)
    parser.add_argument("--transverse-quadrature", type=int, default=256)
    parser.add_argument("--fit-rank", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--cletta-restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-mode-cutoff", type=int, default=3)
    parser.add_argument("--cutoff-only", action="store_true")
    parser.add_argument("--couplings", type=float, nargs="+", default=[2, 5, 10, 20, 40])
    parser.add_argument("--correlation-rmax", type=float, default=8.0)
    parser.add_argument("--correlation-points", type=int, default=161)
    parser.add_argument("--structure-kmax", type=float, default=6.0)
    parser.add_argument("--structure-points", type=int, default=121)
    parser.add_argument(
        "--output", default="/private/tmp/cletta_cylinder_suite.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
