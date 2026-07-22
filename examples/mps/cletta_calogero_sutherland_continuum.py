"""Uniform cMPS/cLETTA benchmark for the Calogero-Sutherland gas.

The target finite-ring Hamiltonian is

    H = -1/2 sum_i d_i^2
        + (pi/L)^2 lambda (lambda - 1)
          sum_{i<j} sin^{-2}(pi(x_i-x_j)/L).

The uniform cMPS calculation takes its thermodynamic limit, where the periodic
kernel approaches 1/r^2.  The internal cMPS kinetic functional has coefficient
one, so the optimizer evaluates 2H and reported energies are divided by two.
At density rho the exact thermodynamic-limit energy density of H is

    e0 = pi^2 lambda^2 rho^3 / 6.

The inverse-square kernel is contracted through its positive Laplace
representation, 1/r^2 = int_0^inf da a exp(-a r).  A logarithmic quadrature
resolves a requested window r_min <= r <= r_max; tightening r_min is therefore
an explicit ultraviolet-convergence study rather than a hidden softening.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from pyqed.mps import ContinuousMPS, pack_canonical_parameters, unpack_canonical_parameters


def inverse_square_laplace_terms(*, r_min, r_max, rank, tail_factor=20.0):
    """Return positive exponential terms resolving ``1/r**2`` on a window."""
    r_min = float(r_min)
    r_max = float(r_max)
    rank = int(rank)
    tail_factor = float(tail_factor)
    if not (0.0 < r_min < r_max):
        raise ValueError("r_min and r_max must satisfy 0 < r_min < r_max.")
    if rank < 2:
        raise ValueError("rank must be at least two.")
    if tail_factor <= 0.0:
        raise ValueError("tail_factor must be positive.")

    alpha_min = 0.02 / r_max
    alpha_max = tail_factor / r_min
    log_alpha = np.linspace(np.log(alpha_min), np.log(alpha_max), rank)
    step = float(log_alpha[1] - log_alpha[0])
    rates = np.exp(log_alpha)
    strengths = step * rates**2
    return rates, strengths


def kernel_errors(rates, strengths, *, r_min, r_max, points=600):
    distances = np.geomspace(float(r_min), float(r_max), int(points))
    exact = 1.0 / distances**2
    fitted = np.exp(-distances[:, None] * rates[None, :]) @ strengths
    relative = fitted / exact - 1.0
    return {
        "distances": distances,
        "exact": exact,
        "fitted": fitted,
        "relative": relative,
        "relative_rms": float(np.sqrt(np.mean(relative**2))),
        "relative_max": float(np.max(np.abs(relative))),
    }


def calogero_sutherland_energy(lambda_value, *, density=1.0):
    lambda_value = float(lambda_value)
    density = float(density)
    if lambda_value <= 1.0:
        raise ValueError("this repulsive bosonic benchmark requires lambda > 1.")
    if density <= 0.0:
        raise ValueError("density must be positive.")
    return np.pi**2 * lambda_value**2 * density**3 / 6.0


def _stationary_g2(state, distances):
    """Return fixed-density g2 for canonical or enlarged cLETTA matrices."""
    distances = np.asarray(distances, dtype=float)
    left, right, eigenvalue = state.dominant_fixed_points()
    transfer = state.transfer_matrix()
    shifted = transfer - eigenvalue * np.eye(transfer.shape[0], dtype=complex)
    insertion = np.kron(state.r, state.r.conj())
    raw_density = float(np.real(np.vdot(left, insertion @ right)))
    if raw_density <= 0.0:
        raise FloatingPointError("stationary density must be positive.")
    scale = float(state.scale)
    initial = insertion @ right
    values = [
        np.vdot(left, insertion @ expm(shifted * (scale * distance)) @ initial)
        for distance in distances
    ]
    return np.real(np.asarray(values)) / raw_density**2


def _require_converged(state, label):
    if not bool(state.success):
        raise RuntimeError(f"{label} did not converge: {state.message}")
    if not np.isfinite(state.energy):
        raise FloatingPointError(f"{label} returned a non-finite energy.")


def _parse_config(value):
    try:
        modes, depth = value.lower().split("x", maxsplit=1)
        modes, depth = int(modes), int(depth)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("configs must have the form MxL, e.g. 2x1") from exc
    if modes < 1 or depth < 1:
        raise argparse.ArgumentTypeError("M and L must be positive")
    return modes, depth


def _conjugate_pair_seed(single_mode_state):
    """Embed an optimized real single mode into the reduced pair parameters."""
    base_theta = np.asarray(single_mode_state.cletta_base.theta, dtype=float)
    tie = np.asarray(single_mode_state.cletta_tie_matrices[0], dtype=float) / np.sqrt(2.0)
    rate = float(single_mode_state.cletta_decay_rates[0])
    frequency = float(single_mode_state.cletta_frequencies[0])
    return np.concatenate([base_theta, tie.reshape(-1), [np.log(rate), frequency]])


def _real_two_mode_seeds(single_mode_state, *, max_rate):
    """Embed a real single mode into two independent real memory channels."""
    base_theta = np.asarray(single_mode_state.cletta_base.theta, dtype=float)
    tie = np.asarray(single_mode_state.cletta_tie_matrices[0], dtype=float)
    zero = np.zeros_like(tie)
    rate = float(single_mode_state.cletta_decay_rates[0])
    second_rate = min(float(max_rate), 4.0 * rate)
    return [
        np.concatenate(
            [base_theta, tie.reshape(-1), zero.reshape(-1), np.log([rate, second_rate])]
        ),
        np.concatenate(
            [
                base_theta,
                (tie / np.sqrt(2.0)).reshape(-1),
                (tie / np.sqrt(2.0)).reshape(-1),
                np.log([rate, rate]),
            ]
        ),
    ]


def _embed_canonical_theta(theta, source_dim, target_dim, *, coupling=1.0e-3):
    """Embed a converged canonical cMPS into a weakly coupled larger chart."""
    source_dim = int(source_dim)
    target_dim = int(target_dim)
    if target_dim <= source_dim:
        raise ValueError("target_dim must exceed source_dim.")
    _q, (r_source,), a_source = unpack_canonical_parameters(theta, source_dim)
    a_target = np.zeros((target_dim, target_dim), dtype=float)
    r_target = np.zeros((target_dim, target_dim), dtype=float)
    a_target[:source_dim, :source_dim] = np.real(a_source)
    r_target[:source_dim, :source_dim] = np.real(r_source)
    for index in range(source_dim, target_dim):
        vector = float(coupling) * np.arange(1, source_dim + 1)
        a_target[:source_dim, index] = vector
        a_target[index, :source_dim] = -vector
        r_target[:source_dim, index] = vector
        r_target[index, :source_dim] = vector[::-1]
        r_target[index, index] = float(coupling)
    return pack_canonical_parameters(a_target, r_target)


def _row(label, state, exact_energy, *, modes=0, depth=0, conjugate_pair=False):
    energy = 0.5 * float(state.energy)
    return {
        "label": label,
        "bond_dim": state.cletta_base.bond_dim if modes else state.bond_dim,
        "num_modes": modes,
        "depth": depth,
        "conjugate_pair": bool(conjugate_pair),
        "energy": energy,
        "solver_energy_2H": float(state.energy),
        "energy_error": float(energy - exact_energy),
        "relative_error": float((energy - exact_energy) / exact_energy),
        "kinetic": 0.5 * float(state.kinetic),
        "interaction": 0.5 * float(state.interaction),
        "success": bool(state.success),
        "nfev": int(state.nfev),
        "message": str(state.message),
        "memory_rates": "" if not modes else ";".join(
            f"{value:.12g}" for value in state.cletta_decay_rates
        ),
        "memory_frequencies": "" if not modes else ";".join(
            f"{value:.12g}" for value in state.cletta_frequencies
        ),
        "tie_norm": "" if not modes else float(np.linalg.norm(state.cletta_tie_matrices)),
    }


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _saved_cletta_seed(archive, *, bond_dim, modes, depth):
    if archive is None:
        return None
    prefix = f"cletta_d{int(bond_dim)}_m{int(modes)}_l"
    suffix = "_parameters"
    candidates = []
    for key in archive.files:
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        depth_text = key[len(prefix) : -len(suffix)]
        try:
            saved_depth = int(depth_text)
        except ValueError:
            continue
        if saved_depth <= int(depth):
            candidates.append((saved_depth, np.asarray(archive[key], dtype=float)))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _plot(args, rows, correlations, kernel):
    import ultraplot as uplt
    from matplotlib.ticker import LogFormatterMathtext

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#5F6368")
    markers = ("o", "s", "^", "D", "v")
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "tick.labelsize": 10,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axs = uplt.subplots(ncols=3, refwidth=2.75, refheight=2.55, share=False, wspace=3.2)

    ax = axs[0]
    x = np.arange(len(rows))
    energies = np.asarray([row["energy"] for row in rows])
    ax.plot(x, energies, color=colors[0], marker="o", markerfacecolor="white")
    ax.axhline(args.exact_energy, color=colors[1], linestyle="--", label="exact")
    ax.format(
        xticks=x,
        xticklabels=[row["label"].replace("cLETTA-", "") for row in rows],
        xrotation=25,
        ylabel=r"energy density $e$",
        title="Variational energy",
        grid=False,
    )
    ax.legend(loc="upper right", frame=False)

    ax = axs[1]
    for index, row in enumerate(rows):
        label = row["label"]
        ax.plot(
            args.correlation_grid,
            correlations[label],
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            markerfacecolor="white",
            markevery=max(1, len(args.correlation_grid) // 10),
            markersize=4.2,
            label=label.replace("cLETTA-", ""),
        )
    ax.axhline(1.0, color=colors[-1], linewidth=0.9, linestyle=":")
    ax.format(xlabel=r"$\rho r$", ylabel=r"$g_2(r)$", title="Pair correlation", grid=False)
    ax.legend(loc="lower right", frame=False, ncols=1)

    ax = axs[2]
    ax.semilogx(
        kernel["distances"],
        np.abs(kernel["relative"]),
        color=colors[2],
    )
    ax.format(
        xlabel=r"$\rho r$",
        ylabel=r"relative kernel error",
        title=rf"$R={args.kernel_rank}$ Laplace terms",
        yscale="log",
        grid=False,
    )
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    for label, ax in zip(("a", "b", "c"), axs):
        ax.text(0.0, 1.045, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    output = Path(args.figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=350)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.png')}")


def run(args):
    rates, unit_strengths = inverse_square_laplace_terms(
        r_min=args.r_min,
        r_max=args.r_max,
        rank=args.kernel_rank,
        tail_factor=args.tail_factor,
    )
    kernel = kernel_errors(rates, unit_strengths, r_min=args.r_min, r_max=args.r_max)
    coupling = 2.0 * args.lambda_value * (args.lambda_value - 1.0)
    strengths = coupling * unit_strengths
    exact_energy = calogero_sutherland_energy(args.lambda_value, density=args.density)
    args.exact_energy = exact_energy
    args.correlation_grid = np.linspace(0.0, args.correlation_rmax, args.correlation_points)

    print(
        f"lambda={args.lambda_value:g} rho={args.density:g} exact={exact_energy:.10f} "
        f"kernel_R={args.kernel_rank} rms={kernel['relative_rms']:.3e} "
        f"max={kernel['relative_max']:.3e} window=[{args.r_min:g}, {args.r_max:g}]"
    )

    rows = []
    states = {}
    cmps_seeds = {}
    seed_archive = np.load(args.seed_states) if args.seed_states else None
    for index, bond_dim in enumerate(dict.fromkeys(args.cmps_bond_dims)):
        seed_thetas = []
        smaller_dims = [dim for dim in cmps_seeds if dim < bond_dim]
        if smaller_dims:
            source_dim = max(smaller_dims)
            seed_thetas.append(
                _embed_canonical_theta(
                    cmps_seeds[source_dim],
                    source_dim,
                    bond_dim,
                    coupling=args.cmps_embedding_scale,
                )
            )
        state = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
            bond_dim=bond_dim,
            decay_rates=rates,
            strengths=strengths,
            density=args.density,
            seed_thetas=seed_thetas,
            restarts=max(args.cmps_restarts, len(seed_thetas) + 1),
            seed=args.seed,
            maxiter=args.cmps_maxiter,
            maxfun=args.cmps_maxfun,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
        )
        label = f"cMPS-D{bond_dim}"
        _require_converged(state, label)
        rows.append(_row(label, state, exact_energy))
        states[label] = state
        cmps_seeds[bond_dim] = state.theta
        energy = 0.5 * float(state.energy)
        print(f"{label:18s} E={energy:.10f} error={energy-exact_energy:+.3e}")

    previous_by_modes = {}
    single_mode_state = None
    for index, (modes, depth) in enumerate(args.configs):
        conjugate_pair = bool(args.m2_mode_type == "conjugate" and modes == 2)
        real_two_mode = bool(args.m2_mode_type == "real" and modes == 2)
        seed_parameters = []
        saved_seed = _saved_cletta_seed(
            seed_archive,
            bond_dim=args.bond_dim,
            modes=modes,
            depth=depth,
        )
        if saved_seed is not None:
            seed_parameters.append(saved_seed)
        if modes in previous_by_modes:
            seed_parameters.append(previous_by_modes[modes])
        if conjugate_pair and single_mode_state is not None:
            seed_parameters.append(_conjugate_pair_seed(single_mode_state))
        if real_two_mode and single_mode_state is not None:
            seed_parameters.extend(
                _real_two_mode_seeds(
                    single_mode_state,
                    max_rate=args.max_memory_rate,
                )
            )
        seed_base = [cmps_seeds[args.bond_dim]] if args.bond_dim in cmps_seeds else []
        state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=rates,
            strengths=strengths,
            density=args.density,
            num_modes=modes,
            depth=depth,
            seed_parameters=seed_parameters,
            seed_base_thetas=seed_base,
            restarts=max(args.restarts, len(seed_parameters) + len(seed_base)),
            seed=args.seed + 1000 + index,
            maxiter=args.maxiter,
            regularization=args.cletta_regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            rate_bounds=(args.min_memory_rate, args.max_memory_rate),
            frequency_bounds=(-args.max_memory_frequency, args.max_memory_frequency),
            memory_frequencies=np.zeros(modes) if real_two_mode else None,
            optimize_memory_frequencies=not real_two_mode,
            tie_scale=args.tie_scale,
            conjugate_pair=conjugate_pair,
            eigensolver=args.eigensolver,
            eigen_iterations=args.eigen_iterations,
        )
        label = f"cLETTA-D{args.bond_dim}-M{modes}-L{depth}"
        _require_converged(state, label)
        previous_by_modes[modes] = state.cletta_parameters
        if modes == 1:
            single_mode_state = state
        rows.append(
            _row(
                label,
                state,
                exact_energy,
                modes=modes,
                depth=depth,
                conjugate_pair=conjugate_pair,
            )
        )
        states[label] = state
        print(
            f"{label:18s} E={0.5*state.energy:.10f} "
            f"error={0.5*state.energy-exact_energy:+.3e} "
            f"rates={np.array2string(state.cletta_decay_rates, precision=5)} "
            f"frequencies={np.array2string(state.cletta_frequencies, precision=5)}"
        )

    correlations = (
        {}
        if args.skip_correlations
        else {
            label: _stationary_g2(state, args.correlation_grid)
            for label, state in states.items()
        }
    )
    for row in rows:
        row.update(
            {
                "lambda": args.lambda_value,
                "density": args.density,
                "exact_energy": exact_energy,
                "inverse_square_coupling": args.lambda_value * (args.lambda_value - 1.0),
                "solver_inverse_square_coupling_2H": coupling,
                "kernel_rank": args.kernel_rank,
                "r_min": args.r_min,
                "r_max": args.r_max,
                "kernel_relative_rms": kernel["relative_rms"],
                "kernel_relative_max": kernel["relative_max"],
            }
        )
    _write_rows(args.output, rows)
    print(f"wrote {args.output}")

    if correlations:
        correlation_rows = [
            {"label": label, "distance": float(distance), "g2": float(value)}
            for label, values in correlations.items()
            for distance, value in zip(args.correlation_grid, values)
        ]
        _write_rows(args.correlations, correlation_rows)
        print(f"wrote {args.correlations}")

    arrays = {"kernel_rates": rates, "kernel_strengths": strengths}
    for label, state in states.items():
        key = label.lower().replace("-", "_")
        arrays[f"{key}_theta"] = (
            state.cletta_base.theta if state.cletta_parameters is not None else state.theta
        )
        if state.cletta_parameters is not None:
            arrays[f"{key}_parameters"] = state.cletta_parameters
    np.savez(args.states, **arrays)
    print(f"wrote {args.states}")

    if args.figure and correlations:
        _plot(args, rows, correlations, kernel)
    return rows, correlations, kernel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-value", type=float, default=2.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=20.0)
    parser.add_argument("--kernel-rank", type=int, default=16)
    parser.add_argument("--tail-factor", type=float, default=20.0)
    parser.add_argument("--cmps-bond-dims", type=int, nargs="*", default=[2, 3])
    parser.add_argument("--cmps-restarts", type=int, default=3)
    parser.add_argument("--cmps-maxiter", type=int, default=300)
    parser.add_argument("--cmps-maxfun", type=int, default=100000)
    parser.add_argument("--cmps-embedding-scale", type=float, default=1.0e-3)
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument(
        "--configs",
        type=_parse_config,
        nargs="*",
        default=[(1, 1), (2, 1)],
    )
    parser.add_argument(
        "--m2-mode-type",
        choices=("real", "conjugate", "independent"),
        default="real",
        help="use real poles, a z/z* pair, or unconstrained complex M=2 modes",
    )
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--cletta-regularization", type=float, default=1.0e-8)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-3)
    parser.add_argument("--tie-scale", type=float, default=0.01)
    parser.add_argument("--min-memory-rate", type=float, default=0.02)
    parser.add_argument("--max-memory-rate", type=float, default=20.0)
    parser.add_argument("--max-memory-frequency", type=float, default=20.0)
    parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "iterative"),
        default="auto",
    )
    parser.add_argument("--eigen-iterations", type=int, default=256)
    parser.add_argument("--correlation-rmax", type=float, default=4.0)
    parser.add_argument("--correlation-points", type=int, default=81)
    parser.add_argument("--skip-correlations", action="store_true")
    root = Path(__file__).with_name("results")
    parser.add_argument("--output", default=str(root / "cletta_calogero_sutherland.csv"))
    parser.add_argument("--correlations", default=str(root / "cletta_calogero_sutherland_g2.csv"))
    parser.add_argument("--states", default=str(root / "cletta_calogero_sutherland_states.npz"))
    parser.add_argument("--seed-states", default=None)
    parser.add_argument("--figure", default=str(root / "cletta_calogero_sutherland.pdf"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
