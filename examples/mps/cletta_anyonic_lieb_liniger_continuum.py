"""Anyonic Lieb-Liniger string-correlation benchmark for cMPS and cLETTA.

The local variational state is optimized at the effective bosonic
Lieb-Liniger coupling ``c_eff``.  Anyonic statistics enters observables through
the exact continuum string

    psi_A(x) = exp(-i theta int_{-inf}^x n(y)dy) psi_B(x).

Thus the energy benchmark is Bethe-ansatz controlled, while the complex
one-body function tests the nonlocal statistical string fairly for both
ansatzes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pyqed.mps import (
    ContinuousMPS,
    canonical_parameter_size,
    pack_canonical_parameters,
    unpack_canonical_parameters,
)


def lieb_liniger_bethe_energy(coupling, *, density=1.0, n_grid=180):
    """Return the thermodynamic Lieb-Liniger energy density."""
    coupling = float(coupling)
    density = float(density)
    points, weights = np.polynomial.legendre.leggauss(int(n_grid))
    rhs = np.full(points.size, 1.0 / (2.0 * np.pi))

    def solve(lam):
        delta = points[:, None] - points[None, :]
        kernel = 2.0 * lam / (lam * lam + delta * delta)
        matrix = np.eye(points.size) - weights[None, :] * kernel / (2.0 * np.pi)
        rapidity_density = np.linalg.solve(matrix, rhs)
        norm = float(np.dot(weights, rapidity_density))
        kinetic = float(np.dot(weights, points**2 * rapidity_density))
        return lam / norm, norm, kinetic

    gamma = coupling / density
    log_lam = brentq(
        lambda value: solve(np.exp(value))[0] - gamma,
        -30.0,
        30.0,
        xtol=1.0e-12,
        rtol=1.0e-12,
    )
    _gamma, norm, kinetic = solve(np.exp(log_lam))
    return density**3 * kinetic / norm**3


def embed_canonical_theta(theta, source_dim, target_dim, *, coupling=1.0e-3):
    """Embed a converged canonical cMPS in a weakly coupled larger chart."""
    _q, (r_source,), a_source = unpack_canonical_parameters(theta, source_dim)
    a_target = np.zeros((target_dim, target_dim), dtype=float)
    r_target = np.zeros((target_dim, target_dim), dtype=float)
    a_target[:source_dim, :source_dim] = np.real(a_source)
    r_target[:source_dim, :source_dim] = np.real(r_source)
    vector = float(coupling) * np.arange(1, source_dim + 1)
    a_target[:source_dim, source_dim] = vector
    a_target[source_dim, :source_dim] = -vector
    r_target[:source_dim, source_dim] = vector
    r_target[source_dim, :source_dim] = vector[::-1]
    r_target[source_dim, source_dim] = float(coupling)
    return pack_canonical_parameters(a_target, r_target)


def require_converged(state, label):
    if not bool(state.success):
        raise RuntimeError(f"{label} did not converge: {state.message}")
    if not np.isfinite(state.energy):
        raise FloatingPointError(f"{label} returned a non-finite energy")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(args, rows, correlations):
    import ultraplot as uplt

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "tick.labelsize": 10,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axs = uplt.subplots(ncols=3, refwidth=2.8, refheight=2.55, share=False, wspace=3.2)

    ax = axs[0]
    x = np.arange(len(rows))
    errors = np.abs([row["energy_error"] for row in rows])
    ax.semilogy(x, errors, color="#5F6368", linewidth=1.0)
    for index, row in enumerate(rows):
        ax.scatter(
            index,
            errors[index],
            edgecolor=colors[index],
            marker=markers[index],
            facecolor="white",
            linewidth=1.4,
            s=42,
        )
    ax.format(
        xticks=x,
        xticklabels=[row["label"] for row in rows],
        xrotation=25,
        ylabel=r"energy error $|e-e_{\rm BA}|$",
        title="Ground-state energy",
        grid=False,
    )

    legend_handles = []
    for panel, component, ylabel, title in (
        (1, np.real, r"$\operatorname{Re} g_1^A(x)$", "Real part"),
        (2, np.imag, r"$\operatorname{Im} g_1^A(x)$", "Imaginary part"),
    ):
        ax = axs[panel]
        for index, row in enumerate(rows):
            values = correlations[row["label"]]
            line = ax.plot(
                args.distances,
                component(values),
                color=colors[index],
                marker=markers[index],
                markerfacecolor="white",
                markevery=max(1, len(args.distances) // 10),
                markersize=4.2,
            )
            if panel == 1:
                legend_handles.append(line[0])
        ax.axhline(0.0, color="#5F6368", linewidth=0.8, linestyle=":")
        ax.format(xlabel=r"$\rho x$", ylabel=ylabel, title=title, grid=False)
    axs[1].legend(
        legend_handles,
        [row["label"] for row in rows],
        loc="upper right",
        ncols=1,
        frame=False,
    )

    for label, ax in zip(("a", "b", "c"), axs):
        ax.text(0.0, 1.045, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    output = Path(args.figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=350)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.png')}")


def run(args):
    exact_energy = lieb_liniger_bethe_energy(
        args.coupling,
        density=args.density,
        n_grid=args.bethe_grid,
    )
    args.distances = np.linspace(0.0, args.rmax, args.points)
    rows = []
    states = {}

    cmps_d3 = ContinuousMPS.optimize_lieb_liniger_fixed_density(
        bond_dim=3,
        coupling=args.coupling,
        density=args.density,
        restarts=args.cmps_restarts,
        seed=args.seed,
        maxiter=args.cmps_maxiter,
    )
    require_converged(cmps_d3, "cMPS-D3")
    states["cMPS-D3"] = cmps_d3

    d4_seed = embed_canonical_theta(cmps_d3.theta, 3, 4)
    cmps_d4 = ContinuousMPS.optimize_lieb_liniger_fixed_density(
        bond_dim=4,
        coupling=args.coupling,
        density=args.density,
        seed_thetas=[d4_seed],
        restarts=max(args.cmps_restarts, 2),
        seed=args.seed,
        maxiter=args.cmps_maxiter,
    )
    require_converged(cmps_d4, "cMPS-D4")
    states["cMPS-D4"] = cmps_d4

    cletta_l1 = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=3,
        interaction_decay_rates=[1.0],
        strengths=[0.0],
        contact_coupling=args.coupling,
        density=args.density,
        num_modes=1,
        depth=1,
        seed_base_thetas=[cmps_d3.theta],
        restarts=args.cletta_restarts,
        seed=args.seed + 100,
        maxiter=args.cletta_maxiter,
        regularization=args.regularization,
        rate_bounds=(args.min_memory_rate, args.max_memory_rate),
        frequency_bounds=(-args.max_memory_frequency, args.max_memory_frequency),
        tie_scale=args.tie_scale,
    )
    require_converged(cletta_l1, "cLETTA-D3-M1-L1")
    states["cLETTA-D3-M1-L1"] = cletta_l1

    if args.depth > 1:
        label = f"cLETTA-D3-M1-L{args.depth}"
        cletta_target = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=3,
            interaction_decay_rates=[1.0],
            strengths=[0.0],
            contact_coupling=args.coupling,
            density=args.density,
            num_modes=1,
            depth=args.depth,
            seed_parameters=[cletta_l1.cletta_parameters],
            seed_base_thetas=[cmps_d3.theta],
            restarts=args.cletta_restarts,
            seed=args.seed + 100 + args.depth,
            maxiter=args.cletta_maxiter,
            regularization=args.regularization,
            rate_bounds=(args.min_memory_rate, args.max_memory_rate),
            frequency_bounds=(-args.max_memory_frequency, args.max_memory_frequency),
            tie_scale=args.tie_scale,
        )
        require_converged(cletta_target, label)
        states[label] = cletta_target

    parameter_counts = {
        "cMPS-D3": canonical_parameter_size(3),
        "cMPS-D4": canonical_parameter_size(4),
    }
    for label in states:
        if label.startswith("cLETTA"):
            parameter_counts[label] = canonical_parameter_size(3) + 3 * 3 + 2
    for label, state in states.items():
        row = {
            "label": label,
            "parameters": parameter_counts[label],
            "energy": float(state.energy),
            "exact_energy": exact_energy,
            "energy_error": float(state.energy - exact_energy),
            "relative_energy_error": float((state.energy - exact_energy) / exact_energy),
            "success": bool(state.success),
            "nfev": int(state.nfev),
            "coupling": args.coupling,
            "density": args.density,
            "statistical_angle": args.statistical_angle,
            "memory_rate": "",
            "memory_frequency": "",
            "tie_norm": "",
        }
        if state.cletta_decay_rates is not None:
            row["memory_rate"] = float(state.cletta_decay_rates[0])
            row["memory_frequency"] = float(state.cletta_frequencies[0])
            row["tie_norm"] = float(np.linalg.norm(state.cletta_tie_matrices))
        rows.append(row)
        print(
            f"{label:20s} E={state.energy:.10f} "
            f"error={state.energy-exact_energy:+.3e} params={parameter_counts[label]}"
        )

    correlations = {
        label: state.anyonic_field_correlation(
            args.distances,
            statistical_angle=args.statistical_angle,
            density=args.density,
            normalized=True,
        )
        for label, state in states.items()
    }
    reference = correlations["cMPS-D4"]
    mask = args.distances > 0.0
    for row in rows:
        delta = correlations[row["label"]][mask] - reference[mask]
        row["correlation_error_vs_D4"] = float(
            np.linalg.norm(delta) / np.linalg.norm(reference[mask])
        )
        print(
            f"  {row['label']:18s} correlation error vs D4 = "
            f"{row['correlation_error_vs_D4']:.3e}"
        )

    write_rows(args.output, rows)
    correlation_rows = [
        {
            "label": label,
            "distance": float(distance),
            "real_g1": float(np.real(value)),
            "imag_g1": float(np.imag(value)),
            "abs_g1": float(np.abs(value)),
            "phase_g1": float(np.angle(value)),
        }
        for label, values in correlations.items()
        for distance, value in zip(args.distances, values)
    ]
    write_rows(args.correlations, correlation_rows)
    arrays = {"distances": args.distances}
    for label, state in states.items():
        key = label.lower().replace("-", "_")
        arrays[f"{key}_theta"] = state.cletta_base.theta if state.cletta_base else state.theta
        arrays[f"{key}_g1"] = correlations[label]
        if state.cletta_parameters is not None:
            arrays[f"{key}_parameters"] = state.cletta_parameters
    np.savez(args.states, **arrays)
    print(f"wrote {args.output}")
    print(f"wrote {args.correlations}")
    print(f"wrote {args.states}")
    plot_results(args, rows, correlations)
    return rows, correlations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=10.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--statistical-angle", type=float, default=0.6 * np.pi)
    parser.add_argument("--bethe-grid", type=int, default=180)
    parser.add_argument("--rmax", type=float, default=6.0)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--cmps-restarts", type=int, default=3)
    parser.add_argument("--cmps-maxiter", type=int, default=1500)
    parser.add_argument("--cletta-restarts", type=int, default=3)
    parser.add_argument("--cletta-maxiter", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--regularization", type=float, default=1.0e-8)
    parser.add_argument("--tie-scale", type=float, default=0.01)
    parser.add_argument("--min-memory-rate", type=float, default=0.02)
    parser.add_argument("--max-memory-rate", type=float, default=20.0)
    parser.add_argument("--max-memory-frequency", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=821)
    root = Path(__file__).with_name("results")
    parser.add_argument("--output", default=str(root / "cletta_anyonic_lieb_liniger.csv"))
    parser.add_argument(
        "--correlations",
        default=str(root / "cletta_anyonic_lieb_liniger_g1.csv"),
    )
    parser.add_argument("--states", default=str(root / "cletta_anyonic_lieb_liniger_states.npz"))
    parser.add_argument("--figure", default=str(root / "cletta_anyonic_lieb_liniger.pdf"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
