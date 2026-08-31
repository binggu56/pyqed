"""Benchmark warm starts, block-metric relaxation, and frontier LETTA gauges."""

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
from examples.mps.benchmark_frontier_letta_vs_mps_4x4 import (
    _mps_state_vector,
    _summary,
    _vector_diagnostics,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA, frontier_tied_letta_from_mps


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_optimization_4x4.json"

VARIANTS = (
    "random_sweep",
    "random_block_metric",
    "random_frontier_gauge",
    "mps_warm_sweep",
    "mps_warm_block_metric",
    "mps_warm_frontier_gauge",
)
POSTHOC_CONVERGENCE_WINDOW = 6
POSTHOC_CONVERGENCE_TOL = 1.0e-9
EXACT_V0_SEED = 271828


def _validated_run_inputs(passes, seeds, mps_warm_passes, tie_noise, variants):
    passes = int(passes)
    mps_warm_passes = int(mps_warm_passes)
    seeds = tuple(int(seed) for seed in seeds)
    variants = tuple(str(variant) for variant in variants)
    tie_noise = float(tie_noise)
    if passes <= 0:
        raise ValueError("passes must be positive.")
    if not seeds:
        raise ValueError("at least one seed is required.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique.")
    if not variants:
        raise ValueError("at least one variant is required.")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique.")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}.")
    if mps_warm_passes <= 0:
        raise ValueError("mps_warm_passes must be positive.")
    if not np.isfinite(tie_noise) or tie_noise < 0.0:
        raise ValueError("tie_noise must be finite and nonnegative.")
    return passes, seeds, mps_warm_passes, tie_noise, variants


def _child_seeds(seed):
    children = np.random.SeedSequence(int(seed)).spawn(3)
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)


def _posthoc_converged(history):
    if len(history) < POSTHOC_CONVERGENCE_WINDOW:
        return False
    recent = history[-POSTHOC_CONVERGENCE_WINDOW:]
    return not any(row["solver_failures"] for row in history) and all(
        abs(float(row["delta_energy"])) <= POSTHOC_CONVERGENCE_TOL for row in recent
    )


def _optimized_mps_d4(
    hamiltonian,
    sparse_hamiltonian,
    *,
    seed,
    initialization_seed,
    passes,
):
    from pyqed.mps import DMRG, MPS, MPO

    start = perf_counter()
    nsites = len(hamiltonian.dims)
    ranks = tuple(min(4, 2 ** min(cut, nsites - cut)) for cut in range(nsites + 1))
    rng = np.random.default_rng(initialization_seed)
    factors = [
        rng.normal(size=(ranks[site], 2, ranks[site + 1]))
        / np.sqrt(ranks[site] * 2 * ranks[site + 1])
        for site in range(nsites)
    ]
    solver = DMRG(
        MPO(list(hamiltonian.to_mpo().compress().tensors)),
        D=4,
        init_guess=MPS(factors, labels=["lv", "p", "rv"]),
        nsweeps=passes,
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=0,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-10,
        davidson_max_iter=100,
        noise=0.0,
        recenter_final=False,
        performance="reference",
    ).run()
    ordered_state = solver.ground_state.to_order(["lv", "p", "rv"])
    ordered_factors = tuple(
        np.asarray(factor).copy() for factor in ordered_state.factors
    )
    seconds = perf_counter() - start
    vector = _mps_state_vector(ordered_state)
    h_vector = sparse_hamiltonian @ vector
    energy = float(np.real(np.vdot(vector, h_vector)))
    return ordered_factors, {
        "seed": int(seed),
        "initialization_seed": int(initialization_seed),
        "bond_dim": 4,
        "energy": energy,
        "seconds": float(seconds),
        "directional_pass_limit": int(passes),
        "directional_passes_completed": len(
            [
                row
                for row in solver.sweep_history
                if row.get("direction") in {"lr", "rl"}
            ]
        ),
        "converged": bool(solver.converged),
    }


def _run_variant(
    name,
    state,
    sparse_hamiltonian,
    exact_state,
    exact_energy,
    *,
    seed,
    initialization_seed,
    passes,
    warm_start,
    block_metric,
    frontier_gauge,
    frontier_gauge_weighting,
    setup_seconds,
    warm_start_seconds,
):
    initial_energy = float(state.energy)
    start = perf_counter()
    state.run(
        nsweeps=passes,
        tol=0.0,
        solver="direct",
        natural_gradient_every=2 if block_metric else 0,
        natural_gradient_damping=1.0e-6,
        natural_gradient_trust_radius=0.25,
        virtual_canonicalization=False,
        frontier_canonicalization=frontier_gauge,
        frontier_gauge_weighting=frontier_gauge_weighting,
    )
    seconds = perf_counter() - start
    diagnostics = _vector_diagnostics(
        state.state_vector(normalize=True),
        sparse_hamiltonian,
        exact_state,
        exact_energy,
    )
    relaxations = [
        row["natural_gradient"]
        for row in state.history
        if row["natural_gradient"] is not None
    ]
    gauge_updates = [
        update for row in state.history for update in (row["frontier_gauge"] or ())
    ]
    site_updates = [update for row in state.history for update in row["updates"]]
    metric_rank_fractions = np.asarray(
        [update.metric_rank / update.raw_dim for update in site_updates]
    )
    applied_gauges = [update for update in gauge_updates if update.applied]
    skipped_gauges = {}
    for update in gauge_updates:
        if not update.applied:
            skipped_gauges[update.message] = skipped_gauges.get(update.message, 0) + 1
    solver_failures = int(sum(row["solver_failures"] for row in state.history))
    posthoc_converged = _posthoc_converged(state.history)
    print(
        f"{name} seed={seed} "
        f"E={diagnostics['energy']:.10f} time={seconds:.2f}s "
        f"natural={sum(update.accepted for update in relaxations)}/"
        f"{len(relaxations)} gauge={len(applied_gauges)}/{len(gauge_updates)}",
        flush=True,
    )
    return {
        "name": name,
        "seed": int(seed),
        "initialization_seed": int(initialization_seed),
        "bond_dim": state.bond_dim,
        "parameters": state.nparameters,
        "tie_edges": int(sum(map(len, state.parent_sets))),
        "warm_start": bool(warm_start),
        "block_metric_relaxation": bool(block_metric),
        "frontier_gauge": bool(frontier_gauge),
        "frontier_gauge_weighting": (
            frontier_gauge_weighting if frontier_gauge else None
        ),
        "converged": bool(posthoc_converged),
        "solver_reported_converged": bool(state.converged),
        "fixed_budget_requested": True,
        "budget_exhausted": len(state.history) == int(passes),
        "directional_pass_limit": int(passes),
        "directional_passes_completed": len(state.history),
        "initial_energy": initial_energy,
        **diagnostics,
        "frontier_energy_discrepancy": diagnostics["energy"] - float(state.energy),
        "warm_start_seconds": float(warm_start_seconds),
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": float(seconds),
        "standalone_total_seconds": float(warm_start_seconds + setup_seconds + seconds),
        "seconds_per_directional_pass": float(seconds / max(len(state.history), 1)),
        "peak_frontier_elements": state.peak_frontier_elements,
        "cached_environment_elements": state.cached_environment_elements,
        "final_directional_pass_delta_energy": float(state.history[-1]["delta_energy"]),
        "solver_failures": solver_failures,
        "natural_gradient_steps": len(relaxations),
        "accepted_natural_gradient_steps": int(
            sum(update.accepted for update in relaxations)
        ),
        "natural_gradient_energy_gain": float(
            sum(update.energy_before - update.energy for update in relaxations)
        ),
        "frontier_gauge_bond_attempts": len(gauge_updates),
        "applied_frontier_gauges": len(applied_gauges),
        "skipped_frontier_gauges": skipped_gauges,
        "maximum_frontier_imbalance_before": max(
            (update.imbalance_before for update in applied_gauges),
            default=None,
        ),
        "maximum_frontier_imbalance_after": max(
            (update.imbalance_after for update in applied_gauges),
            default=None,
        ),
        "maximum_frontier_gauge_condition": max(
            (update.gauge_condition for update in applied_gauges),
            default=None,
        ),
        "minimum_local_metric_rank_fraction": min(
            metric_rank_fractions,
            default=0.0,
        ),
        "lower_quartile_local_metric_rank_fraction": float(
            np.quantile(metric_rank_fractions, 0.25)
        ),
        "median_local_metric_rank_fraction": float(np.median(metric_rank_fractions)),
        "local_metric_updates_below_90_percent_rank": int(
            np.count_nonzero(metric_rank_fractions < 0.9)
        ),
        "maximum_local_residual_norm": max(
            (update.residual_norm for update in site_updates),
            default=0.0,
        ),
        "directional_pass_energies": [float(row["energy"]) for row in state.history],
    }


def _optimization_summary(runs):
    summary = _summary(runs)
    setup_times = np.asarray([run["setup_seconds"] for run in runs])
    total_times = np.asarray([run["standalone_total_seconds"] for run in runs])
    natural_steps = int(sum(run["natural_gradient_steps"] for run in runs))
    accepted_steps = int(sum(run["accepted_natural_gradient_steps"] for run in runs))
    natural_gains = np.asarray([run["natural_gradient_energy_gain"] for run in runs])
    gauge_attempts = int(sum(run["frontier_gauge_bond_attempts"] for run in runs))
    applied_gauges = int(sum(run["applied_frontier_gauges"] for run in runs))
    summary.update(
        {
            "median_setup_seconds": float(np.median(setup_times)),
            "median_standalone_total_seconds": float(np.median(total_times)),
            "accepted_natural_gradient_steps": accepted_steps,
            "natural_gradient_steps": natural_steps,
            "natural_gradient_acceptance_rate": (
                float(accepted_steps / natural_steps) if natural_steps else None
            ),
            "median_natural_gradient_energy_gain": float(np.median(natural_gains)),
            "frontier_gauge_bond_attempts": gauge_attempts,
            "applied_frontier_gauges": applied_gauges,
            "frontier_gauge_application_rate": (
                float(applied_gauges / gauge_attempts) if gauge_attempts else None
            ),
            "median_lower_quartile_local_metric_rank_fraction": float(
                np.median(
                    [run["lower_quartile_local_metric_rank_fraction"] for run in runs]
                )
            ),
            "median_local_metric_rank_fraction": float(
                np.median([run["median_local_metric_rank_fraction"] for run in runs])
            ),
            "median_maximum_local_residual_norm": float(
                np.median([run["maximum_local_residual_norm"] for run in runs])
            ),
        }
    )
    return summary


def _paired_summaries(runs):
    comparisons = (
        (
            "random_block_metric_minus_random_sweep",
            "random_block_metric",
            "random_sweep",
        ),
        (
            "random_frontier_gauge_minus_random_sweep",
            "random_frontier_gauge",
            "random_sweep",
        ),
        (
            "mps_warm_block_metric_minus_mps_warm_sweep",
            "mps_warm_block_metric",
            "mps_warm_sweep",
        ),
        (
            "mps_warm_frontier_gauge_minus_mps_warm_sweep",
            "mps_warm_frontier_gauge",
            "mps_warm_sweep",
        ),
        (
            "mps_warm_sweep_minus_random_sweep",
            "mps_warm_sweep",
            "random_sweep",
        ),
        (
            "mps_warm_block_metric_minus_random_block_metric",
            "mps_warm_block_metric",
            "random_block_metric",
        ),
    )
    result = {}
    for name, candidate_name, baseline_name in comparisons:
        if candidate_name not in runs or baseline_name not in runs:
            continue
        candidate = {int(run["seed"]): run for run in runs[candidate_name]}
        baseline = {int(run["seed"]): run for run in runs[baseline_name]}
        seeds = sorted(set(candidate) & set(baseline))
        if not seeds:
            continue
        energy_deltas = {
            str(seed): float(candidate[seed]["energy"] - baseline[seed]["energy"])
            for seed in seeds
        }
        time_deltas = [
            candidate[seed]["standalone_total_seconds"]
            - baseline[seed]["standalone_total_seconds"]
            for seed in seeds
        ]
        tolerance = 1.0e-12
        deltas = np.asarray(list(energy_deltas.values()))
        result[name] = {
            "candidate": candidate_name,
            "baseline": baseline_name,
            "seeds": seeds,
            "energy_delta_by_seed": energy_deltas,
            "median_energy_delta": float(np.median(deltas)),
            "candidate_wins": int(np.count_nonzero(deltas < -tolerance)),
            "baseline_wins": int(np.count_nonzero(deltas > tolerance)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
            "median_standalone_seconds_delta": float(np.median(time_deltas)),
        }
    return result


def _benchmark_summary(runs):
    return {
        "by_variant": {
            name: _optimization_summary(values) for name, values in runs.items()
        },
        "paired": _paired_summaries(runs),
    }


def run_benchmark(
    *,
    passes=20,
    seeds=(3, 7, 11, 19, 23),
    mps_warm_passes=100,
    tie_noise=1.0e-3,
    variants=VARIANTS,
    frontier_gauge_weighting="uniform",
):
    passes, seeds, mps_warm_passes, tie_noise, variants = _validated_run_inputs(
        passes,
        seeds,
        mps_warm_passes,
        tie_noise,
        variants,
    )
    frontier_gauge_weighting = str(frontier_gauge_weighting).lower().replace("-", "_")
    if frontier_gauge_weighting not in {"uniform", "probability"}:
        raise ValueError("frontier_gauge_weighting must be 'uniform' or 'probability'.")
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    weighted_bonds = tuple((i, j, 1.0) for i, j in nearest)
    weighted_bonds += tuple((i, j, 0.5) for i, j in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(16, weighted_bonds)
    sparse_hamiltonian = sparse_heisenberg_hamiltonian(16, weighted_bonds)
    exact_values, exact_vectors = eigsh(
        sparse_hamiltonian,
        k=1,
        which="SA",
        tol=1.0e-12,
        v0=np.random.default_rng(EXACT_V0_SEED).normal(
            size=sparse_hamiltonian.shape[0]
        ),
    )
    exact_energy = float(exact_values[0])
    exact_state = exact_vectors[:, 0]
    parent_sets = parent_sets_from_edges(16, nearest)

    runs = {variant: [] for variant in variants}
    warm_starts = {}
    for seed in seeds:
        mps_seed, random_seed, tie_seed = _child_seeds(seed)
        needs_warm = any(variant.startswith("mps_warm") for variant in variants)
        mps_factors = None
        if needs_warm:
            mps_factors, warm_record = _optimized_mps_d4(
                hamiltonian,
                sparse_hamiltonian,
                seed=seed,
                initialization_seed=mps_seed,
                passes=mps_warm_passes,
            )
            warm_starts[str(seed)] = warm_record
        for variant in variants:
            warm = variant.startswith("mps_warm")
            block_metric = variant.endswith("block_metric")
            frontier_gauge = variant.endswith("frontier_gauge")
            initialization_seed = tie_seed if warm else random_seed
            setup_start = perf_counter()
            if warm:
                state = frontier_tied_letta_from_mps(
                    hamiltonian,
                    parent_sets,
                    mps_factors,
                    bond_dim=4,
                    tie_noise=tie_noise,
                    seed=initialization_seed,
                    frontier_backend="compressed",
                )
            else:
                state = FrontierTiedLETTA(
                    hamiltonian,
                    hamiltonian.dims,
                    parent_sets,
                    bond_dim=4,
                    seed=initialization_seed,
                    frontier_backend="compressed",
                )
            setup_seconds = perf_counter() - setup_start
            record = _run_variant(
                variant,
                state,
                sparse_hamiltonian,
                exact_state,
                exact_energy,
                seed=seed,
                initialization_seed=initialization_seed,
                passes=passes,
                warm_start=warm,
                block_metric=block_metric,
                frontier_gauge=frontier_gauge,
                frontier_gauge_weighting=frontier_gauge_weighting,
                setup_seconds=setup_seconds,
                warm_start_seconds=(warm_record["seconds"] if warm else 0.0),
            )
            runs[variant].append(record)

    return {
        "model": {
            "nrows": 4,
            "ncols": 4,
            "j1": 1.0,
            "j2": 0.5,
            "boundary": "open",
            "site_order": "row-wise-snake",
        },
        "settings": {
            "directional_pass_limit": passes,
            "seeds": list(seeds),
            "variant_seeds": {variant: list(seeds) for variant in variants},
            "mps_warm_directional_pass_limit": int(mps_warm_passes),
            "tie_noise": float(tie_noise),
            "variants": list(variants),
            "seed_derivation": "SeedSequence(base).spawn(mps,random_letta,tie_noise)",
            "natural_gradient_every": 2,
            "natural_gradient_damping": 1.0e-6,
            "natural_gradient_trust_radius": 0.25,
            "virtual_canonicalization": False,
            "frontier_gauge_weighting": frontier_gauge_weighting,
            "fixed_budget": True,
            "posthoc_convergence_window": POSTHOC_CONVERGENCE_WINDOW,
            "posthoc_convergence_tolerance": POSTHOC_CONVERGENCE_TOL,
        },
        "exact_reference": {
            "energy": exact_energy,
            "v0_seed": EXACT_V0_SEED,
            "used_during_optimization": False,
        },
        "mps_warm_starts": warm_starts,
        "runs": runs,
        "summary": _benchmark_summary(runs),
    }


def merge_results(paths):
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not inputs:
        raise ValueError("at least one result file is required.")
    model = inputs[0]["model"]
    exact_reference = inputs[0]["exact_reference"]
    variable_settings = {"seeds", "variants", "variant_seeds"}
    invariant_settings = {
        key: value
        for key, value in inputs[0]["settings"].items()
        if key not in variable_settings
    }
    runs = {}
    warm_starts = {}
    seen_runs = set()
    for result in inputs:
        if result["model"] != model:
            raise ValueError("cannot merge different models.")
        current_invariants = {
            key: value
            for key, value in result["settings"].items()
            if key not in variable_settings
        }
        if current_invariants != invariant_settings:
            raise ValueError("cannot merge results with different settings.")
        current_exact = result["exact_reference"]
        if set(current_exact) != set(exact_reference):
            raise ValueError("cannot merge different exact-reference metadata.")
        if not np.isclose(
            current_exact["energy"],
            exact_reference["energy"],
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("cannot merge inconsistent exact energies.")
        if any(
            current_exact[key] != exact_reference[key]
            for key in exact_reference
            if key != "energy"
        ):
            raise ValueError("cannot merge different exact-reference metadata.")

        declared_variants = tuple(result["settings"]["variants"])
        if len(set(declared_variants)) != len(declared_variants):
            raise ValueError("a result file declares duplicate variants.")
        if set(declared_variants) != set(result["runs"]):
            raise ValueError("declared variants do not match stored runs.")
        for variant, values in result["runs"].items():
            if variant not in VARIANTS:
                raise ValueError(f"unknown stored variant {variant!r}.")
            for value in values:
                seed = int(value["seed"])
                key = (variant, seed)
                if key in seen_runs:
                    raise ValueError(
                        f"duplicate result for variant={variant!r}, seed={seed}."
                    )
                if value.get("name") != variant:
                    raise ValueError("run name does not match its variant group.")
                seen_runs.add(key)
                runs.setdefault(variant, []).append(value)

        actual_variant_seeds = {
            variant: sorted(int(value["seed"]) for value in values)
            for variant, values in result["runs"].items()
        }
        declared_variant_seeds = {
            variant: sorted(int(seed) for seed in seeds)
            for variant, seeds in result["settings"]["variant_seeds"].items()
        }
        if declared_variant_seeds != actual_variant_seeds:
            raise ValueError("declared per-variant seeds do not match stored runs.")
        if sorted(
            {seed for seeds in actual_variant_seeds.values() for seed in seeds}
        ) != sorted(int(seed) for seed in result["settings"]["seeds"]):
            raise ValueError("declared seeds do not match stored runs.")

        for seed_key, record in result["mps_warm_starts"].items():
            seed_key = str(int(seed_key))
            if int(record["seed"]) != int(seed_key):
                raise ValueError("warm-start seed does not match its key.")
            if seed_key not in warm_starts:
                warm_starts[seed_key] = record
                continue
            previous = warm_starts[seed_key]
            comparable_keys = (set(previous) | set(record)) - {"seconds", "energy"}
            if any(previous.get(key) != record.get(key) for key in comparable_keys):
                raise ValueError(f"inconsistent warm starts for seed {seed_key}.")
            if not np.isclose(
                previous["energy"],
                record["energy"],
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise ValueError(f"inconsistent warm-start energy for seed {seed_key}.")
            previous["seconds"] = min(previous["seconds"], record["seconds"])

    for values in runs.values():
        values.sort(key=lambda value: value["seed"])
    ordered_variants = [variant for variant in VARIANTS if variant in runs]
    settings = dict(inputs[0]["settings"])
    settings["variants"] = ordered_variants
    settings["variant_seeds"] = {
        variant: [int(value["seed"]) for value in runs[variant]]
        for variant in ordered_variants
    }
    settings["seeds"] = sorted(
        {seed for seeds in settings["variant_seeds"].values() for seed in seeds}
    )
    runs = {variant: runs[variant] for variant in ordered_variants}
    warm_starts = {
        seed: warm_starts[seed]
        for seed in sorted(warm_starts, key=lambda value: int(value))
    }
    return {
        "model": model,
        "settings": settings,
        "exact_reference": exact_reference,
        "mps_warm_starts": warm_starts,
        "runs": runs,
        "summary": _benchmark_summary(runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passes",
        type=int,
        default=20,
        help="number of LETTA directional passes",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 7, 11, 19, 23))
    parser.add_argument(
        "--mps-warm-passes",
        type=int,
        default=100,
        help="maximum number of MPS directional warm-start passes",
    )
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument(
        "--frontier-gauge-weighting",
        choices=("uniform", "probability"),
        default="uniform",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
    )
    parser.add_argument("--merge-results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = (
        merge_results(args.merge_results)
        if args.merge_results
        else run_benchmark(
            passes=args.passes,
            seeds=args.seeds,
            mps_warm_passes=args.mps_warm_passes,
            tie_noise=args.tie_noise,
            variants=args.variants,
            frontier_gauge_weighting=args.frontier_gauge_weighting,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
