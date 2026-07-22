#!/usr/bin/env python3
"""Scan frustration for small/large MPS and graph-LETTA states on 4x4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA, frontier_tied_letta_from_mps


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_j2_scan_4x4.json"
DEFAULT_RATIOS = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
DEFAULT_SEEDS = (3, 7, 11)
REPORTED_MPS_DIMS = (4, 8)
LETTA_DIMS = (2, 4)
AUXILIARY_MPS_DIMS = (2,)
EXACT_V0_SEED = 271828


def _validated_ratios(ratios):
    ratios = tuple(float(value) for value in ratios)
    if not ratios:
        raise ValueError("at least one J2/J1 ratio is required.")
    if any(not np.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError("J2/J1 ratios must be finite and nonnegative.")
    if any(right <= left for left, right in zip(ratios, ratios[1:])):
        raise ValueError("J2/J1 ratios must be strictly increasing.")
    return ratios


def _validated_seeds(seeds):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required.")
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers.")
    return seeds


def _mps_ranks(nsites, bond_dim):
    return tuple(
        min(int(bond_dim), 2 ** min(cut, nsites - cut)) for cut in range(nsites + 1)
    )


def _mps_capacity(nsites, bond_dim):
    ranks = _mps_ranks(nsites, bond_dim)
    return int(sum(ranks[site] * 2 * ranks[site + 1] for site in range(nsites)))


def _random_mps(nsites, bond_dim, seed):
    from pyqed.mps import MPS

    ranks = _mps_ranks(nsites, bond_dim)
    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(size=(ranks[site], 2, ranks[site + 1]))
        / np.sqrt(ranks[site] * 2 * ranks[site + 1])
        for site in range(nsites)
    ]
    return MPS(factors, labels=["lv", "p", "rv"])


def _ordered_mps_factors(state):
    ordered = state.to_order(["lv", "p", "rv"])
    return tuple(np.asarray(factor).copy() for factor in ordered.factors)


def _mps_state_vector(state):
    factors = state.to_order(["lv", "p", "rv"]).factors
    environment = np.ones((1, 1), dtype=np.result_type(*factors))
    for factor in factors:
        environment = np.einsum(
            "ca,apb->cpb",
            environment,
            factor,
            optimize=True,
        ).reshape(-1, factor.shape[2])
    vector = environment[:, 0]
    return vector / np.linalg.norm(vector)


def _diagnostics(vector, sparse_hamiltonian, exact_state, exact_energy, nsites):
    vector = np.asarray(vector) / np.linalg.norm(vector)
    h_vector = sparse_hamiltonian @ vector
    energy = float(np.real(np.vdot(vector, h_vector)))
    residual = h_vector - energy * vector
    variance = float(np.real(np.vdot(residual, residual)))
    return {
        "energy": energy,
        "energy_error": float(energy - exact_energy),
        "energy_error_per_site": float((energy - exact_energy) / nsites),
        "variance": variance,
        "residual_norm": float(np.sqrt(max(variance, 0.0))),
        "ground_state_fidelity": float(abs(np.vdot(exact_state, vector)) ** 2),
    }


def _child_seed(base_seed, family, bond_dim, ratio_index=0):
    sequence = np.random.SeedSequence(
        [int(base_seed), int(family), int(bond_dim), int(ratio_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _optimize_mps(
    hamiltonian,
    sparse_hamiltonian,
    exact_state,
    exact_energy,
    *,
    bond_dim,
    seed,
    pass_limit,
    tolerance,
    initial_state=None,
):
    from pyqed.mps import DMRG, MPO

    nsites = len(hamiltonian.dims)
    if initial_state is None:
        initial_state = _random_mps(nsites, bond_dim, seed)
        initialization = "random"
    else:
        initial_state = initial_state.copy()
        initialization = "previous_ratio"
    initial_vector = _mps_state_vector(initial_state)
    initial_energy = float(
        np.real(np.vdot(initial_vector, sparse_hamiltonian @ initial_vector))
    )
    start = perf_counter()
    solver = DMRG(
        MPO(list(hamiltonian.to_mpo().compress().tensors)),
        D=int(bond_dim),
        init_guess=initial_state,
        nsweeps=int(pass_limit),
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=0,
        sweep_tol=float(tolerance),
        davidson_tol=min(float(tolerance), 1.0e-10),
        davidson_max_iter=100,
        noise=0.0,
        recenter_final=False,
        performance="generic",
    ).run()
    seconds = perf_counter() - start
    directional_history = [
        row for row in solver.sweep_history if row.get("direction") in {"lr", "rl"}
    ]
    vector = _mps_state_vector(solver.ground_state)
    diagnostics = _diagnostics(
        vector,
        sparse_hamiltonian,
        exact_state,
        exact_energy,
        nsites,
    )
    ranks = _mps_ranks(nsites, bond_dim)
    stored_parameters = int(
        sum(np.asarray(factor).size for factor in solver.ground_state.factors)
    )
    record = {
        "method": f"mps_d{int(bond_dim)}",
        "kind": "mps",
        "optimizer": "two_site_dmrg",
        "symmetry": "none",
        "bond_dim": int(bond_dim),
        "parameter_capacity": _mps_capacity(nsites, bond_dim),
        "stored_parameters": stored_parameters,
        "bond_ranks_capacity": list(ranks),
        "initialization": initialization,
        "initial_energy": initial_energy,
        **diagnostics,
        "optimization_seconds": float(seconds),
        "standalone_total_seconds": float(seconds),
        "directional_pass_limit": int(pass_limit),
        "directional_passes_completed": len(directional_history),
        "converged": bool(solver.converged),
        "solver_failures": 0,
        "final_delta_energy": (
            float(
                abs(
                    directional_history[-1]["energy"]
                    - directional_history[-2]["energy"]
                )
            )
            if len(directional_history) >= 2
            else 0.0
        ),
        "directional_pass_energies": [
            float(row["energy"]) for row in directional_history
        ],
    }
    return solver.ground_state.copy(), record


def _continued_letta_state(
    previous_state,
    hamiltonian,
    parent_sets,
    *,
    bond_dim,
):
    if previous_state is None:
        return None
    return FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dim=int(bond_dim),
        tensors=[tensor.copy() for tensor in previous_state.tensors],
        frontier_backend="compressed",
    )


def _optimize_letta(
    hamiltonian,
    sparse_hamiltonian,
    exact_state,
    exact_energy,
    parent_sets,
    warm_mps,
    previous_state,
    *,
    bond_dim,
    tie_seed,
    tie_noise,
    pass_limit,
    tolerance,
    frontier_gauge,
    frontier_gauge_weighting,
    warm_mps_seconds,
):
    nsites = len(hamiltonian.dims)
    setup_start = perf_counter()
    lifted = frontier_tied_letta_from_mps(
        hamiltonian,
        parent_sets,
        _ordered_mps_factors(warm_mps),
        bond_dim=int(bond_dim),
        tie_noise=float(tie_noise),
        seed=int(tie_seed),
        frontier_backend="compressed",
    )
    candidates = {"same_d_mps_lift": lifted}
    continued = _continued_letta_state(
        previous_state,
        hamiltonian,
        parent_sets,
        bond_dim=bond_dim,
    )
    if continued is not None:
        candidates["previous_ratio"] = continued
    candidate_energies = {
        name: float(candidate.expectation()) for name, candidate in candidates.items()
    }
    initialization = min(candidate_energies, key=candidate_energies.get)
    state = candidates[initialization]
    setup_seconds = perf_counter() - setup_start
    initial_energy = float(state.energy)

    start = perf_counter()
    state.run(
        nsweeps=int(pass_limit),
        tol=float(tolerance),
        solver="direct",
        frontier_canonicalization=bool(frontier_gauge),
        frontier_gauge_weighting=frontier_gauge_weighting,
    )
    seconds = perf_counter() - start
    diagnostics = _diagnostics(
        state.state_vector(normalize=True),
        sparse_hamiltonian,
        exact_state,
        exact_energy,
        nsites,
    )
    site_updates = [update for row in state.history for update in row["updates"]]
    gauge_updates = [
        update for row in state.history for update in (row.get("frontier_gauge") or ())
    ]
    applied_gauges = [update for update in gauge_updates if update.applied]
    metric_rank_fractions = np.asarray(
        [update.metric_rank / update.raw_dim for update in site_updates],
        dtype=float,
    )
    solver_failures = int(sum(row["solver_failures"] for row in state.history))
    record = {
        "method": f"letta_d{int(bond_dim)}",
        "kind": "letta",
        "optimizer": "one_site_generalized_eigensolve",
        "symmetry": "none",
        "bond_dim": int(bond_dim),
        "parameter_capacity": int(state.nparameters),
        "stored_parameters": int(state.nparameters),
        "tie_edges": int(sum(map(len, parent_sets))),
        "initialization": initialization,
        "initial_candidate_energies": candidate_energies,
        "tie_noise": float(tie_noise),
        "initial_energy": initial_energy,
        **diagnostics,
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": float(seconds),
        "warm_mps_seconds": float(warm_mps_seconds),
        "standalone_total_seconds": float(
            setup_seconds
            + seconds
            + (warm_mps_seconds if initialization == "same_d_mps_lift" else 0.0)
        ),
        "directional_pass_limit": int(pass_limit),
        "directional_passes_completed": len(state.history),
        "converged": bool(state.converged),
        "solver_failures": solver_failures,
        "final_delta_energy": (
            float(abs(state.history[-1]["delta_energy"])) if state.history else 0.0
        ),
        "directional_pass_energies": [float(row["energy"]) for row in state.history],
        "frontier_gauge": bool(frontier_gauge),
        "frontier_gauge_weighting": (
            frontier_gauge_weighting if frontier_gauge else None
        ),
        "frontier_gauge_bond_attempts": len(gauge_updates),
        "applied_frontier_gauges": len(applied_gauges),
        "maximum_frontier_imbalance_after": (
            max((update.imbalance_after for update in applied_gauges), default=None)
        ),
        "minimum_local_metric_rank_fraction": (
            float(np.min(metric_rank_fractions)) if metric_rank_fractions.size else None
        ),
        "median_local_metric_rank_fraction": (
            float(np.median(metric_rank_fractions))
            if metric_rank_fractions.size
            else None
        ),
        "maximum_local_residual_norm": max(
            (float(update.residual_norm) for update in site_updates),
            default=None,
        ),
    }
    return state, record


def _exact_reference(sparse_hamiltonian, previous_vector=None):
    dimension = sparse_hamiltonian.shape[0]
    random_vector = np.random.default_rng(EXACT_V0_SEED).normal(size=dimension)
    random_vector /= np.linalg.norm(random_vector)
    if previous_vector is None:
        initial_vector = random_vector
    else:
        previous_vector = np.asarray(previous_vector)
        previous_vector = previous_vector / np.linalg.norm(previous_vector)
        initial_vector = previous_vector + 1.0e-3 * random_vector
        initial_vector /= np.linalg.norm(initial_vector)
    start = perf_counter()
    values, vectors = eigsh(
        sparse_hamiltonian,
        k=2,
        which="SA",
        tol=1.0e-12,
        v0=initial_vector,
    )
    seconds = perf_counter() - start
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    return vectors[:, 0], {
        "energy": float(values[0]),
        "first_excited_energy": float(values[1]),
        "gap": float(values[1] - values[0]),
        "seconds": float(seconds),
        "used_during_optimization": False,
    }


def _exact_reference_scan(ratios):
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    previous_vector = None
    references = []
    for ratio in ratios:
        weighted_bonds = tuple((i, j, 1.0) for i, j in nearest)
        weighted_bonds += tuple((i, j, float(ratio)) for i, j in diagonals)
        previous_vector, reference = _exact_reference(
            sparse_heisenberg_hamiltonian(16, weighted_bonds),
            previous_vector,
        )
        references.append({"j2_ratio": float(ratio), **reference})
    return references


def _summaries(records):
    summaries = {}
    methods = sorted({record["method"] for record in records})
    ratios = sorted({float(record["j2_ratio"]) for record in records})
    for method in methods:
        method_rows = {}
        for ratio in ratios:
            rows = [
                record
                for record in records
                if record["method"] == method
                and np.isclose(record["j2_ratio"], ratio, atol=1.0e-14, rtol=0.0)
            ]
            if not rows:
                continue
            entry = {
                "runs": len(rows),
                "parameter_capacity": int(rows[0]["parameter_capacity"]),
                "converged_runs": int(sum(bool(row["converged"]) for row in rows)),
                "total_solver_failures": int(
                    sum(row["solver_failures"] for row in rows)
                ),
            }
            for field in (
                "energy_error_per_site",
                "variance",
                "ground_state_fidelity",
                "optimization_seconds",
                "standalone_total_seconds",
                "directional_passes_completed",
            ):
                values = np.asarray([float(row[field]) for row in rows])
                entry[f"median_{field}"] = float(np.median(values))
                entry[f"interquartile_{field}"] = [
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.75)),
                ]
            method_rows[f"{ratio:.12g}"] = entry
        summaries[method] = method_rows
    return summaries


def _result_payload(model, settings, exact_references, records):
    records = sorted(
        records,
        key=lambda row: (float(row["j2_ratio"]), int(row["seed"]), row["method"]),
    )
    return {
        "model": model,
        "settings": settings,
        "exact_references": sorted(
            exact_references,
            key=lambda row: float(row["j2_ratio"]),
        ),
        "records": records,
        "summary": _summaries(records),
    }


def run_scan(
    *,
    ratios=DEFAULT_RATIOS,
    seeds=DEFAULT_SEEDS,
    mps_passes=80,
    letta_passes=40,
    tolerance=1.0e-10,
    tie_noise=1.0e-3,
    frontier_gauge=True,
    frontier_gauge_weighting="probability",
):
    ratios = _validated_ratios(ratios)
    seeds = _validated_seeds(seeds)
    mps_passes = int(mps_passes)
    letta_passes = int(letta_passes)
    if mps_passes <= 0 or letta_passes <= 0:
        raise ValueError("pass limits must be positive.")
    tolerance = float(tolerance)
    tie_noise = float(tie_noise)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if not np.isfinite(tie_noise) or tie_noise < 0.0:
        raise ValueError("tie_noise must be finite and nonnegative.")
    frontier_gauge_weighting = str(frontier_gauge_weighting).lower().replace("-", "_")
    if frontier_gauge_weighting not in {"uniform", "probability"}:
        raise ValueError("frontier gauge weighting must be uniform or probability.")

    nrows = ncols = 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    model = {
        "nrows": nrows,
        "ncols": ncols,
        "j1": 1.0,
        "j2_ratios": list(ratios),
        "boundary": "open",
        "site_order": "row-wise-snake",
        "letta_tie_graph": "all-j1-nearest-neighbor-bonds",
        "letta_tie_edges": int(sum(map(len, parent_sets))),
    }
    settings = {
        "seeds": list(seeds),
        "scan_direction": "ascending-continuation",
        "reported_mps_bond_dims": list(REPORTED_MPS_DIMS),
        "auxiliary_mps_bond_dims": list(AUXILIARY_MPS_DIMS),
        "letta_bond_dims": list(LETTA_DIMS),
        "mps_directional_pass_limit": mps_passes,
        "letta_directional_pass_limit": letta_passes,
        "tolerance": tolerance,
        "tie_noise": tie_noise,
        "frontier_gauge": bool(frontier_gauge),
        "frontier_gauge_weighting": (
            frontier_gauge_weighting if frontier_gauge else None
        ),
        "letta_initialization": (
            "lower variational energy of same-D MPS lift and previous-ratio LETTA"
        ),
        "exact_reference_used_during_optimization": False,
    }

    records = []
    exact_references = []
    previous_exact = None
    previous_mps = {
        (seed, bond_dim): None
        for seed in seeds
        for bond_dim in (*AUXILIARY_MPS_DIMS, *REPORTED_MPS_DIMS)
    }
    previous_letta = {
        (seed, bond_dim): None for seed in seeds for bond_dim in LETTA_DIMS
    }
    for ratio_index, ratio in enumerate(ratios):
        weighted_bonds = tuple((i, j, 1.0) for i, j in nearest)
        weighted_bonds += tuple((i, j, ratio) for i, j in diagonals)
        hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
        sparse_hamiltonian = sparse_heisenberg_hamiltonian(nsites, weighted_bonds)
        exact_state, exact = _exact_reference(sparse_hamiltonian, previous_exact)
        previous_exact = exact_state
        exact_references.append({"j2_ratio": ratio, **exact})
        print(
            f"ratio={ratio:.3f} exact={exact['energy']:.10f} gap={exact['gap']:.6f}",
            flush=True,
        )

        for seed in seeds:
            mps_records = {}
            current_mps = {}
            for bond_dim in (*AUXILIARY_MPS_DIMS, *REPORTED_MPS_DIMS):
                mps_state, record = _optimize_mps(
                    hamiltonian,
                    sparse_hamiltonian,
                    exact_state,
                    exact["energy"],
                    bond_dim=bond_dim,
                    seed=_child_seed(seed, 101, bond_dim),
                    pass_limit=mps_passes,
                    tolerance=tolerance,
                    initial_state=previous_mps[(seed, bond_dim)],
                )
                previous_mps[(seed, bond_dim)] = mps_state.copy()
                current_mps[bond_dim] = mps_state
                mps_records[bond_dim] = record
                if bond_dim in REPORTED_MPS_DIMS:
                    records.append({"j2_ratio": ratio, "seed": seed, **record})
                print(
                    f"  seed={seed} mps_d{bond_dim} "
                    f"dE/N={record['energy_error_per_site']:.3e} "
                    f"passes={record['directional_passes_completed']} "
                    f"time={record['optimization_seconds']:.2f}s",
                    flush=True,
                )

            for bond_dim in LETTA_DIMS:
                state, record = _optimize_letta(
                    hamiltonian,
                    sparse_hamiltonian,
                    exact_state,
                    exact["energy"],
                    parent_sets,
                    current_mps[bond_dim],
                    previous_letta[(seed, bond_dim)],
                    bond_dim=bond_dim,
                    tie_seed=_child_seed(seed, 202, bond_dim, ratio_index),
                    tie_noise=tie_noise,
                    pass_limit=letta_passes,
                    tolerance=tolerance,
                    frontier_gauge=frontier_gauge,
                    frontier_gauge_weighting=frontier_gauge_weighting,
                    warm_mps_seconds=mps_records[bond_dim]["optimization_seconds"],
                )
                previous_letta[(seed, bond_dim)] = state
                records.append({"j2_ratio": ratio, "seed": seed, **record})
                print(
                    f"  seed={seed} letta_d{bond_dim} "
                    f"dE/N={record['energy_error_per_site']:.3e} "
                    f"init={record['initialization']} "
                    f"passes={record['directional_passes_completed']} "
                    f"time={record['optimization_seconds']:.2f}s",
                    flush=True,
                )

    return _result_payload(model, settings, exact_references, records)


def merge_results(paths):
    inputs = [_load_json(path) for path in paths]
    if not inputs:
        raise ValueError("at least one result is required for merging.")
    model = inputs[0]["model"]
    settings = dict(inputs[0]["settings"])
    exact_by_ratio = {}
    records = []
    seen = set()
    seeds = set()
    for result in inputs:
        if result["model"] != model:
            raise ValueError("cannot merge scans with different models.")
        for exact in result["exact_references"]:
            ratio = float(exact["j2_ratio"])
            if ratio in exact_by_ratio and not np.isclose(
                exact_by_ratio[ratio]["energy"],
                exact["energy"],
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError("cannot merge inconsistent exact references.")
            exact_by_ratio.setdefault(ratio, exact)
        for record in result["records"]:
            key = (float(record["j2_ratio"]), int(record["seed"]), record["method"])
            if key in seen:
                raise ValueError(f"duplicate scan record {key}.")
            seen.add(key)
            seeds.add(int(record["seed"]))
            records.append(record)
    settings["seeds"] = sorted(seeds)
    refreshed_references = _exact_reference_scan(model["j2_ratios"])
    for reference in refreshed_references:
        ratio = float(reference["j2_ratio"])
        if ratio not in exact_by_ratio or not np.isclose(
            exact_by_ratio[ratio]["energy"],
            reference["energy"],
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError(
                f"refreshed exact ground energy is inconsistent at J2/J1={ratio}."
            )
    settings["exact_references_refreshed_on_merge"] = True
    return _result_payload(
        model,
        settings,
        refreshed_references,
        records,
    )


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mps-passes", type=int, default=80)
    parser.add_argument("--letta-passes", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--no-frontier-gauge", action="store_true")
    parser.add_argument(
        "--frontier-gauge-weighting",
        choices=("uniform", "probability"),
        default="probability",
    )
    parser.add_argument("--merge-results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = (
        merge_results(args.merge_results)
        if args.merge_results
        else run_scan(
            ratios=args.ratios,
            seeds=args.seeds,
            mps_passes=args.mps_passes,
            letta_passes=args.letta_passes,
            tolerance=args.tolerance,
            tie_noise=args.tie_noise,
            frontier_gauge=not args.no_frontier_gauge,
            frontier_gauge_weighting=args.frontier_gauge_weighting,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
