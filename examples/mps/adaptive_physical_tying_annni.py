"""Benchmark adaptive physical-parent exact-factorization states on ANNNI.

The benchmark first compresses the exact dense ground state into physical-only
conditional factors.  It can then relax the factor tables variationally and
replace the Gram-selected parent graph through local or screened-relaxed
Rayleigh--Ritz sweeps.  No summed virtual indices are introduced.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import eigh

from pyqed.letta.physical_tying import (
    VariationalPhysicalTie,
    compress_physical_ties,
    fixed_range_parent_sets,
)


def annni_dense(
    nsites: int,
    *,
    nearest: float = 1.0,
    next_nearest: float = 0.0,
    field: float = 0.8,
) -> np.ndarray:
    """Return the open-chain ANNNI Hamiltonian in the computational basis.

    The convention is

    ``H = -nearest * sum(Z_i Z_{i+1})``
    ``    + next_nearest * sum(Z_i Z_{i+2})``
    ``    - field * sum(X_i)``.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    parameters = np.asarray([nearest, next_nearest, field], dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("ANNNI couplings must be finite.")

    dimension = 1 << nsites
    basis = np.arange(dimension, dtype=np.int64)
    bits = np.empty((dimension, nsites), dtype=np.int8)
    for site in range(nsites):
        bits[:, site] = (basis >> (nsites - 1 - site)) & 1
    z = 1 - 2 * bits

    diagonal = np.zeros(dimension, dtype=float)
    if nsites > 1:
        diagonal -= nearest * np.sum(z[:, :-1] * z[:, 1:], axis=1)
    if nsites > 2:
        diagonal += next_nearest * np.sum(z[:, :-2] * z[:, 2:], axis=1)

    hamiltonian = np.diag(diagonal)
    for site in range(nsites):
        flipped = basis ^ (1 << (nsites - 1 - site))
        hamiltonian[basis, flipped] -= field
    return hamiltonian


def _ground_state(hamiltonian: np.ndarray) -> tuple[float, np.ndarray]:
    values, vectors = eigh(
        hamiltonian,
        subset_by_index=(0, 0),
        driver="evr",
        check_finite=False,
    )
    return float(values[0]), vectors[:, 0]


def _parent_distance_profile(parent_sets) -> str:
    fields = []
    for site, parents in enumerate(parent_sets):
        distances = tuple(parent - site for parent in parents)
        fields.append("+".join(str(distance) for distance in distances) or "-")
    return ",".join(fields)


def _raw_entries(state) -> int:
    return int(sum(factor.size for factor in state.factors) + state.terminal.size)


def _fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    candidate = np.asarray(candidate).reshape(-1)
    candidate = candidate / np.linalg.norm(candidate)
    return float(abs(np.vdot(reference, candidate)) ** 2)


def _copy_variational(ansatz: VariationalPhysicalTie) -> VariationalPhysicalTie:
    return ansatz.copy()


def benchmark_point(
    *,
    nsites: int,
    nearest: float,
    next_nearest: float,
    field: float,
    relative_tolerance: float,
    variational_sweeps: int = 0,
    variational_noise: float = 0.0,
    variational_seed: int = 7,
    rr_graph_sweeps: int = 0,
    rr_tensor_sweeps: int = 1,
    relaxed_graph_sweeps: int = 0,
    relaxed_candidate_budget: int = 6,
    relaxed_per_site_candidates: int = 2,
    relaxed_trial_sweeps: int = 2,
    relaxed_final_sweeps: int = 0,
    tie_budgets: tuple[int, ...] = (1, 2),
) -> list[dict[str, object]]:
    """Return Rq/Aq/RRq/LRRq diagnostics at one ANNNI point.

    ``Rq`` uses the next ``q`` sites as physical parents.  ``Aq`` chooses at
    most ``q`` parents greedily from the local discarded Gram weight.  ``RRq``
    and ``LRRq`` start from the variationally relaxed ``Aq`` state and optimize
    its graph by direct or screened-relaxed Rayleigh--Ritz comparisons.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    relative_tolerance = float(relative_tolerance)
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    variational_sweeps = int(variational_sweeps)
    rr_graph_sweeps = int(rr_graph_sweeps)
    rr_tensor_sweeps = int(rr_tensor_sweeps)
    relaxed_graph_sweeps = int(relaxed_graph_sweeps)
    relaxed_candidate_budget = int(relaxed_candidate_budget)
    relaxed_per_site_candidates = int(relaxed_per_site_candidates)
    relaxed_trial_sweeps = int(relaxed_trial_sweeps)
    relaxed_final_sweeps = int(relaxed_final_sweeps)
    if variational_sweeps < 0:
        raise ValueError("variational_sweeps must be nonnegative.")
    if rr_graph_sweeps < 0 or rr_tensor_sweeps < 0:
        raise ValueError("RR sweep counts must be nonnegative.")
    if relaxed_graph_sweeps < 0:
        raise ValueError("relaxed_graph_sweeps must be nonnegative.")
    if relaxed_candidate_budget < 1 or relaxed_per_site_candidates < 1:
        raise ValueError("relaxed graph candidate budgets must be positive.")
    if relaxed_trial_sweeps < 0 or relaxed_final_sweeps < 0:
        raise ValueError("relaxed tensor sweep counts must be nonnegative.")
    variational_noise = float(variational_noise)
    if not np.isfinite(variational_noise) or variational_noise < 0.0:
        raise ValueError("variational_noise must be finite and nonnegative.")

    tie_budgets = tuple(sorted({int(budget) for budget in tie_budgets}))
    if not tie_budgets or any(budget < 1 for budget in tie_budgets):
        raise ValueError("tie_budgets must contain positive integers.")

    hamiltonian = annni_dense(
        nsites,
        nearest=nearest,
        next_nearest=next_nearest,
        field=field,
    )
    exact_energy, exact_state = _ground_state(hamiltonian)
    dims = (2,) * nsites

    compressed = []
    for tie_range in (0, *tie_budgets):
        compressed.append(
            (
                f"R{tie_range}",
                compress_physical_ties(
                    exact_state,
                    dims,
                    parent_sets=fixed_range_parent_sets(nsites, tie_range),
                ),
            )
        )
    for max_parents in tie_budgets:
        compressed.append(
            (
                f"A{max_parents}",
                compress_physical_ties(
                    exact_state,
                    dims,
                    max_parents=max_parents,
                    relative_tolerance=relative_tolerance,
                ),
            )
        )

    results: list[dict[str, object]] = []
    for label, physical_state in compressed:
        approximate = physical_state.state_vector(normalize=True)
        fidelity = _fidelity(exact_state, approximate)
        energy = float(np.vdot(approximate, hamiltonian @ approximate).real)

        variational = physical_state.variational_ansatz(hamiltonian)
        variational.run(
            nsweeps=variational_sweeps,
            tol=1.0e-10,
            noise=variational_noise,
            seed=variational_seed,
        )
        variational_state = variational.state_vector(normalize=True)
        variational_fidelity = _fidelity(exact_state, variational_state)
        row = {
            "j2": float(next_nearest),
            "e0": exact_energy,
            "method": label,
            "fidelity": fidelity,
            "energy_error": energy - exact_energy,
            "variational_fidelity": variational_fidelity,
            "variational_energy_error": float(variational.energy - exact_energy),
            "control_energy_error": float(variational.energy - exact_energy),
            "rr_gain": 0.0,
            "rr_graph_changes": 0,
            "rr_graph_sweeps": 0,
            "variational_sweeps": len(variational.history),
            "discarded_weight": physical_state.discarded_weight,
            "entries": _raw_entries(physical_state),
            "distances": _parent_distance_profile(physical_state.parent_sets),
        }
        results.append(row)

        if not label.startswith("A"):
            continue
        max_parents = int(label[1:])
        search_seed = variational
        search_seed_sweeps = len(variational.history)

        if rr_graph_sweeps > 0:
            rr = _copy_variational(search_seed)
            control = _copy_variational(search_seed)
            rr.run_parent_search(
                max_parents=max_parents,
                nsweeps=rr_graph_sweeps,
                tensor_sweeps=rr_tensor_sweeps,
                tol=1.0e-10,
                metric_tol=1.0e-12,
                graph_tol=1.0e-10,
            )
            extra_tensor_sweeps = len(rr.graph_history) * rr_tensor_sweeps
            if extra_tensor_sweeps:
                control.run(
                    nsweeps=extra_tensor_sweeps,
                    tol=0.0,
                    noise=0.0,
                )
            rr_state = rr.state_vector(normalize=True)
            rr_error = float(rr.energy - exact_energy)
            control_error = float(control.energy - exact_energy)
            results.append(
                {
                    **row,
                    "method": f"RR{max_parents}",
                    "variational_fidelity": _fidelity(exact_state, rr_state),
                    "variational_energy_error": rr_error,
                    "control_energy_error": control_error,
                    "rr_gain": control_error - rr_error,
                    "rr_graph_changes": sum(
                        sweep["graph_changes"] for sweep in rr.graph_history
                    ),
                    "rr_graph_sweeps": len(rr.graph_history),
                    "variational_sweeps": search_seed_sweeps + extra_tensor_sweeps,
                    "entries": _raw_entries(rr),
                    "distances": _parent_distance_profile(rr.parent_sets),
                }
            )
            search_seed = rr
            search_seed_sweeps += extra_tensor_sweeps

        if relaxed_graph_sweeps > 0:
            relaxed = _copy_variational(search_seed)
            control = _copy_variational(search_seed)
            relaxed.run_relaxed_parent_search(
                max_parents=max_parents,
                nsweeps=relaxed_graph_sweeps,
                candidate_budget=relaxed_candidate_budget,
                per_site_candidates=relaxed_per_site_candidates,
                tensor_sweeps=relaxed_trial_sweeps,
                metric_tol=1.0e-12,
                graph_tol=1.0e-10,
            )
            accepted_moves = sum(
                update["accepted"] for update in relaxed.relaxed_graph_history
            )
            matched_sweeps = accepted_moves * relaxed_trial_sweeps + relaxed_final_sweeps
            if relaxed_final_sweeps:
                relaxed.run(
                    nsweeps=relaxed_final_sweeps,
                    tol=0.0,
                    noise=0.0,
                )
            if matched_sweeps:
                control.run(
                    nsweeps=matched_sweeps,
                    tol=0.0,
                    noise=0.0,
                )
            relaxed_state = relaxed.state_vector(normalize=True)
            relaxed_error = float(relaxed.energy - exact_energy)
            control_error = float(control.energy - exact_energy)
            results.append(
                {
                    **row,
                    "method": f"LRR{max_parents}",
                    "variational_fidelity": _fidelity(exact_state, relaxed_state),
                    "variational_energy_error": relaxed_error,
                    "control_energy_error": control_error,
                    "rr_gain": control_error - relaxed_error,
                    "rr_graph_changes": accepted_moves,
                    "rr_graph_sweeps": len(relaxed.relaxed_graph_history),
                    "variational_sweeps": search_seed_sweeps + matched_sweeps,
                    "entries": _raw_entries(relaxed),
                    "distances": _parent_distance_profile(relaxed.parent_sets),
                }
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=10)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument(
        "--j2",
        type=float,
        nargs="+",
        default=(0.0, 0.6, 0.8, 1.0),
        help="One or more frustrating next-nearest couplings.",
    )
    parser.add_argument("--field", type=float, default=0.8)
    parser.add_argument("--tie-budgets", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument("--variational-sweeps", type=int, default=20)
    parser.add_argument("--variational-noise", type=float, default=1.0e-6)
    parser.add_argument("--variational-seed", type=int, default=7)
    parser.add_argument("--rr-graph-sweeps", type=int, default=0)
    parser.add_argument("--rr-tensor-sweeps", type=int, default=1)
    parser.add_argument("--relaxed-graph-sweeps", type=int, default=0)
    parser.add_argument("--relaxed-candidate-budget", type=int, default=6)
    parser.add_argument("--relaxed-per-site-candidates", type=int, default=2)
    parser.add_argument("--relaxed-trial-sweeps", type=int, default=2)
    parser.add_argument("--relaxed-final-sweeps", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(f"# ANNNI L={args.nsites}, J1={args.j1:g}, h={args.field:g}")
    print("# Rq=fixed range; Aq=Gram selected; RRq=local graph RR; LRRq=relaxed graph RR")
    print(
        " J2  method       F(compress)      dE(compress)       "
        "F(variational)   dE(variational)     graph gain  moves  entries  dist"
    )
    for next_nearest in args.j2:
        rows = benchmark_point(
            nsites=args.nsites,
            nearest=args.j1,
            next_nearest=next_nearest,
            field=args.field,
            relative_tolerance=args.relative_tolerance,
            variational_sweeps=args.variational_sweeps,
            variational_noise=args.variational_noise,
            variational_seed=args.variational_seed,
            rr_graph_sweeps=args.rr_graph_sweeps,
            rr_tensor_sweeps=args.rr_tensor_sweeps,
            relaxed_graph_sweeps=args.relaxed_graph_sweeps,
            relaxed_candidate_budget=args.relaxed_candidate_budget,
            relaxed_per_site_candidates=args.relaxed_per_site_candidates,
            relaxed_trial_sweeps=args.relaxed_trial_sweeps,
            relaxed_final_sweeps=args.relaxed_final_sweeps,
            tie_budgets=tuple(args.tie_budgets),
        )
        for row in rows:
            print(
                f"{float(row['j2']):4.1f}  {str(row['method']):>5s}  "
                f"{float(row['fidelity']): .10f}  "
                f"{float(row['energy_error']): .8e}  "
                f"{float(row['variational_fidelity']): .10f}  "
                f"{float(row['variational_energy_error']): .8e}  "
                f"{float(row['rr_gain']): .8e}  "
                f"{int(row['rr_graph_changes']):5d}  "
                f"{int(row['entries']):7d}  {row['distances']}"
            )
        print()


if __name__ == "__main__":
    main()
