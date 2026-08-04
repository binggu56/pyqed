#!/usr/bin/env python3
"""Compare one-site U(1), unrestricted, and legacy LETTA on a 1D chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.benchmark_frontier_letta_u1_4x4 import (
    _neel_cores,
    _u1_mps_run,
)
from examples.mps.benchmark_heisenberg_1d_letta_vs_mps import (
    exact_ground_state,
)
from pyqed.letta import (
    LETTA,
    LocalHamiltonian,
    LocalTerm,
    TiedFrontierLayout,
    abelian_frontier_tied_letta_from_mps,
)
from pyqed.mps import MPO


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "u1_letta_vs_legacy_1d_L10.json"


def comma_separated_ints(value):
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def heisenberg_chain(length):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = (
        np.kron(sx, sx)
        + np.kron(sy, sy)
        + np.kron(sz, sz)
    )
    return LocalHamiltonian(
        (2,) * int(length),
        tuple(
            LocalTerm((site, site + 1), exchange)
            for site in range(int(length) - 1)
        ),
    )


def nearest_neighbor_parents(length):
    return tuple(
        (site + 1,) if site + 1 < int(length) else ()
        for site in range(int(length))
    )


def normalized_vector(state):
    if hasattr(state, "parent_sets"):
        vector = state.state_vector(normalize=True)
    else:
        vector = state.state_vector()
    vector = np.asarray(vector).reshape(-1)
    return vector / np.linalg.norm(vector)


def phase_aligned_error(reference, vector):
    overlap = np.vdot(reference, vector)
    if abs(overlap) == 0.0:
        return float(np.linalg.norm(reference - vector))
    return float(np.linalg.norm(reference - vector * overlap.conjugate() / abs(overlap)))


def sector_leakage(vector, length):
    probability = np.abs(np.asarray(vector).reshape(-1)) ** 2
    probability /= np.sum(probability)
    return float(
        sum(
            weight
            for configuration, weight in zip(
                np.ndindex(*((2,) * int(length))),
                probability,
            )
            if sum(configuration) != int(length) // 2
        )
    )


def graph_history(state):
    return [
        {
            "pass": int(row["sweep"]),
            "energy": float(row["energy"]),
            "delta_energy": float(row["delta_energy"]),
            "accepted_sites": int(row["accepted_sites"]),
            "solver_failures": int(row["solver_failures"]),
        }
        for row in state.history
    ]


def legacy_history(state):
    return [
        {
            "pass": int(row["sweep"]),
            "direction": str(row["direction"]),
            "energy": float(row["energy"]),
            "delta_energy": (
                None
                if row["delta_energy"] is None
                else float(row["delta_energy"])
            ),
        }
        for row in state.history
    ]


def state_record(
    state,
    *,
    method,
    energy,
    seconds,
    exact_energy,
    length,
    active_parameters,
    dense_parameters,
    history,
    dense_hamiltonian,
):
    vector = normalized_vector(state)
    h_vector = dense_hamiltonian @ vector
    dense_energy = float(np.real(np.vdot(vector, h_vector)))
    variance = float(
        max(
            0.0,
            np.real(np.vdot(h_vector, h_vector)) - dense_energy**2,
        )
    )
    return {
        "method": method,
        "energy": float(energy),
        "dense_vector_energy": dense_energy,
        "error": float(energy - exact_energy),
        "error_per_site": float((energy - exact_energy) / int(length)),
        "variance": variance,
        "seconds": float(seconds),
        "passes": len(history),
        "converged": bool(state.converged),
        "active_parameters": int(active_parameters),
        "dense_parameters": int(dense_parameters),
        "active_parameter_fraction": float(active_parameters / dense_parameters),
        "sector_leakage": sector_leakage(vector, length),
        "history": history,
    }


def unrestricted_graph_from_u1(state):
    unrestricted = LETTA(
        state.hamiltonian,
        parents=state.parent_sets,
        bond_dims=state.bond_dims,
        tensors=[tensor.copy() for tensor in state.tensors],
        frontier_backend="identity_block",
    )
    # The constructor may rebalance gauges.  Restore exactly the same local
    # coordinates so support-sliced local operators can be compared directly.
    unrestricted.tensors = [tensor.copy() for tensor in state.tensors]
    unrestricted.energy = unrestricted.expectation()
    return unrestricted


def legacy_from_graph(state):
    pair_tensors = [
        np.transpose(tensor, (0, 2, 3, 1)).copy()
        for tensor in state.tensors[:-1]
    ]
    terminal = state.tensors[-1][:, 0, :].T.copy()
    return LETTA(
        None,
        state.dims,
        bond_dim=max(state.bond_dims),
        tensors=pair_tensors + [terminal],
    )


def run_graph(state, *, passes, tolerance, canonicalize):
    started = perf_counter()
    state.run(
        nsweeps=int(passes),
        tol=float(tolerance),
        metric_tol=1.0e-12,
        solver="whitened",
        eig_tol=1.0e-11,
        maxiter=500,
        max_subspace=48,
        frontier_canonicalization=bool(canonicalize),
        frontier_gauge_weighting="probability",
        environment_cache="full",
    )
    return float(perf_counter() - started)


def run_legacy(state, mpo, *, passes, tolerance, gauge, local_solver="auto"):
    started = perf_counter()
    state.run(
        mpo,
        nsweeps=int(passes),
        tol=float(tolerance),
        local_solver=local_solver,
        gauge=gauge,
        identity_metric=None if gauge is not None else False,
        metric_tol=1.0e-12,
        adapt_bonds=False,
    )
    return float(perf_counter() - started)


def random_native_pair_state(dims, bond_dim, seed, *, abelian_layout=None):
    rng = np.random.default_rng(seed)
    bonds = (
        [1]
        + [int(bond_dim)] * max(0, len(dims) - 2)
        + [1]
    )
    tensors = [
        rng.normal(
            size=(
                bonds[site],
                int(dims[site]),
                int(dims[site + 1]),
                bonds[site + 1],
            )
        )
        for site in range(len(dims) - 1)
    ]
    return LETTA(
        None,
        dims,
        bond_dim=int(bond_dim),
        tensors=tensors,
        abelian_layout=abelian_layout,
        seed=int(seed),
    )


def local_operator_parity(hamiltonian, parents, *, bond_dim, seed):
    u1 = LETTA(
        hamiltonian,
        parents=parents,
        symmetry="u1",
        charges=((1, -1),) * len(hamiltonian.dims),
        target=0,
        bond_dim=int(bond_dim),
        charge_assignment="physical",
        frontier_backend="identity_block",
        seed=int(seed),
    )
    unrestricted = unrestricted_graph_from_u1(u1)
    metric_errors = []
    hamiltonian_errors = []
    support_sizes = []
    for site in range(len(hamiltonian.dims)):
        u1_metric, u1_hamiltonian = u1.local_operators(site)
        metric, effective = unrestricted.local_operators(site)
        support = u1._support_indices(site)
        metric_errors.append(
            float(
                np.max(
                    np.abs(
                        u1_metric[np.ix_(support, support)]
                        - metric[np.ix_(support, support)]
                    ),
                    initial=0.0,
                )
            )
        )
        hamiltonian_errors.append(
            float(
                np.max(
                    np.abs(
                        u1_hamiltonian[np.ix_(support, support)]
                        - effective[np.ix_(support, support)]
                    ),
                    initial=0.0,
                )
            )
        )
        support_sizes.append(int(support.size))
    return {
        "bond_dim": int(bond_dim),
        "seed": int(seed),
        "maximum_metric_error": max(metric_errors, default=0.0),
        "maximum_hamiltonian_error": max(hamiltonian_errors, default=0.0),
        "support_sizes": support_sizes,
    }


def paired_warm_runs(
    hamiltonian,
    mpo,
    parents,
    dense_hamiltonian,
    *,
    exact_energy,
    bond_dims,
    seeds,
    mps_sweeps,
    tie_noise,
    passes,
    tolerance,
):
    rows = []
    for bond_dim in bond_dims:
        _solver, dense_mps, layout, mps_record = _u1_mps_run(
            MPO(list(mpo)),
            _neel_cores(len(hamiltonian.dims)),
            bond_dim=int(bond_dim),
            sweeps=int(mps_sweeps),
            tolerance=1.0e-11,
        )
        for seed in seeds:
            u1 = abelian_frontier_tied_letta_from_mps(
                hamiltonian,
                parents,
                dense_mps.factors,
                abelian_layout=layout,
                tie_noise=float(tie_noise),
                charge_assignment="physical",
                seed=int(seed),
                frontier_backend="identity_block",
            )
            unrestricted = unrestricted_graph_from_u1(u1)
            legacy = legacy_from_graph(u1)
            initial_vectors = {
                "u1_frontier": normalized_vector(u1),
                "unrestricted_frontier": normalized_vector(unrestricted),
                "legacy_unrestricted": normalized_vector(legacy),
            }
            reference_vector = initial_vectors["u1_frontier"]
            initial_energies = {
                "u1_frontier": float(u1.expectation()),
                "unrestricted_frontier": float(unrestricted.expectation()),
                "legacy_unrestricted": float(legacy.expectation_mpo(mpo)),
            }
            states = {
                "u1_frontier": u1,
                "unrestricted_frontier": unrestricted,
                "legacy_unrestricted": legacy,
            }
            order = tuple(states)
            offset = int(seed) % len(order)
            order = order[offset:] + order[:offset]
            seconds = {}
            for method in order:
                if method == "legacy_unrestricted":
                    seconds[method] = run_legacy(
                        states[method],
                        mpo,
                        passes=passes,
                        tolerance=tolerance,
                        gauge=None,
                    )
                else:
                    seconds[method] = run_graph(
                        states[method],
                        passes=passes,
                        tolerance=tolerance,
                        canonicalize=False,
                    )
            methods = {}
            for method, state in states.items():
                if method == "legacy_unrestricted":
                    energy = float(state.expectation_mpo(mpo))
                    history = legacy_history(state)
                    parameters = int(sum(tensor.size for tensor in state.tensors))
                    dense_parameters = parameters
                else:
                    energy = float(state.expectation())
                    history = graph_history(state)
                    dense_parameters = int(sum(tensor.size for tensor in state.tensors))
                    parameters = (
                        int(state.nparameters)
                        if method == "u1_frontier"
                        else dense_parameters
                    )
                methods[method] = state_record(
                    state,
                    method=method,
                    energy=energy,
                    seconds=seconds[method],
                    exact_energy=exact_energy,
                    length=len(hamiltonian.dims),
                    active_parameters=parameters,
                    dense_parameters=dense_parameters,
                    history=history,
                    dense_hamiltonian=dense_hamiltonian,
                )
            rows.append(
                {
                    "bond_dim": int(bond_dim),
                    "seed": int(seed),
                    "mps_warm_start": mps_record,
                    "initial_energies": initial_energies,
                    "maximum_initial_energy_difference": float(
                        max(initial_energies.values())
                        - min(initial_energies.values())
                    ),
                    "initial_vector_errors_vs_u1": {
                        method: phase_aligned_error(reference_vector, vector)
                        for method, vector in initial_vectors.items()
                    },
                    "methods": methods,
                }
            )
    return rows


def native_random_runs(
    hamiltonian,
    mpo,
    parents,
    dense_hamiltonian,
    *,
    exact_energy,
    u1_bond_dims,
    legacy_bond_dims,
    seeds,
    passes,
    tolerance,
):
    rows = []
    for bond_dim in u1_bond_dims:
        for seed in seeds:
            state = LETTA(
                hamiltonian,
                parents=parents,
                symmetry="u1",
                charges=((1, -1),) * len(hamiltonian.dims),
                target=0,
                bond_dim=int(bond_dim),
                charge_assignment="physical",
                frontier_backend="identity_block",
                seed=int(seed),
            )
            seconds = run_graph(
                state,
                passes=passes,
                tolerance=tolerance,
                canonicalize=True,
            )
            rows.append(
                {
                    "initialization": "native random U1 support",
                    "bond_dim": int(bond_dim),
                    "seed": int(seed),
                    **state_record(
                        state,
                        method="u1_frontier",
                        energy=float(state.expectation()),
                        seconds=seconds,
                        exact_energy=exact_energy,
                        length=len(hamiltonian.dims),
                        active_parameters=state.nparameters,
                        dense_parameters=state.dense_nparameters,
                        history=graph_history(state),
                        dense_hamiltonian=dense_hamiltonian,
                    ),
                }
            )
    for bond_dim in legacy_bond_dims:
        for seed in seeds:
            state = random_native_pair_state(
                hamiltonian.dims,
                bond_dim,
                seed,
            )
            seconds = run_legacy(
                state,
                mpo,
                passes=passes,
                tolerance=tolerance,
                gauge="conditional",
            )
            parameters = int(sum(tensor.size for tensor in state.tensors))
            rows.append(
                {
                    "initialization": "native random unrestricted",
                    "bond_dim": int(bond_dim),
                    "seed": int(seed),
                    **state_record(
                        state,
                        method="legacy_unrestricted",
                        energy=float(state.expectation_mpo(mpo)),
                        seconds=seconds,
                        exact_energy=exact_energy,
                        length=len(hamiltonian.dims),
                        active_parameters=parameters,
                        dense_parameters=parameters,
                        history=legacy_history(state),
                        dense_hamiltonian=dense_hamiltonian,
                    ),
                }
            )
    return rows


def tied_frontier_native_runs(
    hamiltonian,
    mpo,
    dense_hamiltonian,
    *,
    exact_energy,
    bond_dims,
    seeds,
    passes,
    tolerance,
):
    """Run native pair LETTA with charge on the virtual--tied frontier."""
    rows = []
    length = len(hamiltonian.dims)
    local_qns = (((0,), (1,)),) * length
    target = (length // 2,)
    for bond_dim in bond_dims:
        pilot = random_native_pair_state(
            hamiltonian.dims,
            bond_dim,
            seeds[0],
        )
        run_legacy(
            pilot,
            mpo,
            passes=passes,
            tolerance=tolerance,
            gauge="conditional",
            local_solver="dense",
        )
        layout = TiedFrontierLayout.from_state_vector(
            normalized_vector(pilot),
            local_qns,
            target,
            bond_dims=int(bond_dim),
            project=True,
            truncate=True,
            sector_rtol=1.0e-10,
        )
        layout_record = [
            [
                [list(charge) for charge in conditional]
                for conditional in frontier
            ]
            for frontier in layout.frontier_qns
        ]
        for seed in seeds:
            state = random_native_pair_state(
                hamiltonian.dims,
                bond_dim,
                seed,
                abelian_layout=layout,
            )
            seconds = run_legacy(
                state,
                mpo,
                passes=passes,
                tolerance=tolerance,
                gauge="conditional",
                local_solver="dense",
            )
            parameters = int(
                sum(np.count_nonzero(mask) for mask in state.local_masks)
            )
            dense_parameters = int(sum(tensor.size for tensor in state.tensors))
            rows.append(
                {
                    "initialization": "native random tied-frontier U1 support",
                    "layout_source": (
                        "conditional charge ranks of a projected unrestricted "
                        "pilot state"
                    ),
                    "frontier_qns": layout_record,
                    "bond_dim": int(bond_dim),
                    "seed": int(seed),
                    **state_record(
                        state,
                        method="u1_tied_frontier",
                        energy=float(state.expectation_mpo(mpo)),
                        seconds=seconds,
                        exact_energy=exact_energy,
                        length=length,
                        active_parameters=parameters,
                        dense_parameters=dense_parameters,
                        history=legacy_history(state),
                        dense_hamiltonian=dense_hamiltonian,
                    ),
                }
            )
    return rows


def summarize(paired, native):
    paired_summary = []
    for bond_dim in sorted({row["bond_dim"] for row in paired}):
        subset = [row for row in paired if row["bond_dim"] == bond_dim]
        methods = tuple(subset[0]["methods"])
        paired_summary.append(
            {
                "bond_dim": int(bond_dim),
                "maximum_final_energy_spread": float(
                    max(
                        max(row["methods"][method]["energy"] for method in methods)
                        - min(row["methods"][method]["energy"] for method in methods)
                        for row in subset
                    )
                ),
                "methods": {
                    method: {
                        "mean_energy": float(
                            np.mean(
                                [row["methods"][method]["energy"] for row in subset]
                            )
                        ),
                        "mean_seconds": float(
                            np.mean(
                                [row["methods"][method]["seconds"] for row in subset]
                            )
                        ),
                        "active_parameters": int(
                            subset[0]["methods"][method]["active_parameters"]
                        ),
                    }
                    for method in methods
                },
            }
        )
    native_summary = []
    keys = sorted({(row["method"], row["bond_dim"]) for row in native})
    for method, bond_dim in keys:
        subset = [
            row
            for row in native
            if row["method"] == method and row["bond_dim"] == bond_dim
        ]
        best = min(subset, key=lambda row: row["energy"])
        native_summary.append(
            {
                "method": method,
                "bond_dim": int(bond_dim),
                "active_parameters": int(best["active_parameters"]),
                "best_seed": int(best["seed"]),
                "best_energy": float(best["energy"]),
                "best_error": float(best["error"]),
                "median_energy": float(
                    np.median([row["energy"] for row in subset])
                ),
                "mean_seconds": float(
                    np.mean([row["seconds"] for row in subset])
                ),
                "converged_runs": int(sum(row["converged"] for row in subset)),
                "runs": len(subset),
            }
        )
    return {
        "paired_warm": paired_summary,
        "native_random": native_summary,
    }


def benchmark(
    *,
    length=10,
    paired_bond_dims=(2, 4),
    native_u1_bond_dims=(2, 4, 7, 8),
    native_legacy_bond_dims=(2, 4),
    seeds=(1, 2, 3, 4, 5),
    mps_sweeps=8,
    tie_noise=0.02,
    paired_passes=40,
    native_passes=100,
    tolerance=1.0e-10,
    output=DEFAULT_OUTPUT,
):
    length = int(length)
    if length % 2:
        raise ValueError("the fixed-Sz=0 comparison requires even length")
    hamiltonian = heisenberg_chain(length)
    parents = nearest_neighbor_parents(length)
    mpo = tuple(hamiltonian.to_mpo().compress().tensors)
    dense_hamiltonian = hamiltonian.to_dense()
    exact = exact_ground_state(length, 1.0e-13)
    exact_energy = float(exact["energy"])
    parity = local_operator_parity(
        hamiltonian,
        parents,
        bond_dim=max(paired_bond_dims),
        seed=23,
    )
    paired = paired_warm_runs(
        hamiltonian,
        mpo,
        parents,
        dense_hamiltonian,
        exact_energy=exact_energy,
        bond_dims=paired_bond_dims,
        seeds=seeds,
        mps_sweeps=mps_sweeps,
        tie_noise=tie_noise,
        passes=paired_passes,
        tolerance=tolerance,
    )
    native = native_random_runs(
        hamiltonian,
        mpo,
        parents,
        dense_hamiltonian,
        exact_energy=exact_energy,
        u1_bond_dims=native_u1_bond_dims,
        legacy_bond_dims=native_legacy_bond_dims,
        seeds=seeds,
        passes=native_passes,
        tolerance=tolerance,
    )
    native.extend(
        tied_frontier_native_runs(
            hamiltonian,
            mpo,
            dense_hamiltonian,
            exact_energy=exact_energy,
            bond_dims=native_legacy_bond_dims,
            seeds=seeds,
            passes=native_passes,
            tolerance=tolerance,
        )
    )
    payload = {
        "model": {
            "name": "open spin-1/2 antiferromagnetic Heisenberg chain",
            "length": length,
            "hamiltonian": "sum_i S_i dot S_{i+1}",
            "target_two_sz": 0,
            "parent_sets": [list(parent) for parent in parents],
        },
        "protocol": {
            "paired_bond_dims": list(paired_bond_dims),
            "native_u1_bond_dims": list(native_u1_bond_dims),
            "native_legacy_bond_dims": list(native_legacy_bond_dims),
            "seeds": list(seeds),
            "mps_sweeps": int(mps_sweeps),
            "tie_noise": float(tie_noise),
            "paired_passes": int(paired_passes),
            "native_passes": int(native_passes),
            "tolerance": float(tolerance),
            "u1_charge_assignment": "physical",
            "tied_frontier_charge_assignment": (
                "conditional cumulative charge C_i(s_{i+1}, alpha_i)"
            ),
            "tied_frontier_layout_source": (
                "projected unrestricted pilot at the same native bond dimension"
            ),
            "graph_solver": "one-site whitened",
            "legacy_solver": "one-site auto with conditional gauge",
            "tied_frontier_solver": "one-site dense with conditional gauge",
            "small_masked_operator_backend": (
                "vectorized full local contraction followed by support slicing "
                "for raw local dimension <= 256"
            ),
            "alternating_canonical_center_reuse": True,
        },
        "exact": exact,
        "local_operator_parity": parity,
        "paired_warm_runs": paired,
        "native_random_runs": native,
        "summary": summarize(paired, native),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument(
        "--paired-bond-dims",
        type=comma_separated_ints,
        default=(2, 4),
    )
    parser.add_argument(
        "--native-u1-bond-dims",
        type=comma_separated_ints,
        default=(2, 4, 7, 8),
    )
    parser.add_argument(
        "--native-legacy-bond-dims",
        type=comma_separated_ints,
        default=(2, 4),
    )
    parser.add_argument("--seeds", type=comma_separated_ints, default=(1, 2, 3, 4, 5))
    parser.add_argument("--mps-sweeps", type=int, default=8)
    parser.add_argument("--tie-noise", type=float, default=0.02)
    parser.add_argument("--paired-passes", type=int, default=40)
    parser.add_argument("--native-passes", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = benchmark(
        length=args.length,
        paired_bond_dims=args.paired_bond_dims,
        native_u1_bond_dims=args.native_u1_bond_dims,
        native_legacy_bond_dims=args.native_legacy_bond_dims,
        seeds=args.seeds,
        mps_sweeps=args.mps_sweeps,
        tie_noise=args.tie_noise,
        paired_passes=args.paired_passes,
        native_passes=args.native_passes,
        tolerance=args.tolerance,
        output=args.output,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
