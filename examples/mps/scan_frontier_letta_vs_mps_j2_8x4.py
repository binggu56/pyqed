#!/usr/bin/env python3
"""Scalable 8x4 J1-J2 scan for parameter-matched MPS and graph-LETTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from examples.mps.scan_frontier_letta_vs_mps_j2_4x4 import (
    _child_seed,
    _continued_letta_state,
    _mps_capacity,
    _mps_ranks,
    _ordered_mps_factors,
    _random_mps,
    _validated_seeds,
)
from pyqed.letta import frontier_tied_letta_from_mps
from pyqed.mps import DMRG, MPO
from pyqed.mps.dmrg import _normalized_mps_mpo_expectation


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_j2_scan_8x4.json"
DEFAULT_RATIOS = (0.0, 0.5, 0.7, 0.8, 1.0)
DEFAULT_SEEDS = (3, 7, 11)
REPORTED_MPS_DIMS = (4, 8)
AUXILIARY_MPS_DIMS = (2,)
LETTA_DIMS = (2, 4)
REFERENCE_DIMS = (16, 32)


def _validated_scan_ratios(ratios):
    ratios = tuple(float(ratio) for ratio in ratios)
    if not ratios:
        raise ValueError("at least one J2/J1 ratio is required.")
    if any(not np.isfinite(ratio) or ratio < 0.0 for ratio in ratios):
        raise ValueError("J2/J1 ratios must be finite and nonnegative.")
    differences = np.diff(ratios)
    if differences.size and not (
        np.all(differences > 0.0) or np.all(differences < 0.0)
    ):
        raise ValueError("J2/J1 ratios must be strictly monotone.")
    return ratios


def _mps_energy(state, mpo):
    ordered = state.to_order(["lv", "p", "rv"])
    return float(_normalized_mps_mpo_expectation(ordered.factors, mpo.factors))


def _directional_history(solver):
    return [row for row in solver.sweep_history if row.get("direction") in {"lr", "rl"}]


def _truncation_history(rows):
    values = []
    for row in rows:
        value = row.get("truncation")
        try:
            value = float(np.real(np.asarray(value).reshape(-1)[0]))
        except (TypeError, ValueError, IndexError):
            value = None
        values.append(value)
    return values


def _optimize_mps(
    mpo,
    *,
    nsites,
    bond_dim,
    seed,
    pass_limit,
    tolerance,
    initial_state=None,
    initialization=None,
):
    if initial_state is None:
        # The entry-wise scaling used by the small-system benchmark can make
        # the contracted norm fall below the DMRG safety threshold at N=32.
        # Canonicalization restores an O(1) norm without constructing a state
        # vector.
        initial_state = _random_mps(nsites, bond_dim, seed).right_canonicalize()
        initialization = initialization or "random"
    else:
        initial_state = initial_state.copy()
        initialization = initialization or "continued"
    initial_energy = _mps_energy(initial_state, mpo)
    start = perf_counter()
    solver = DMRG(
        mpo,
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
        performance="auto",
    ).run()
    seconds = perf_counter() - start
    rows = _directional_history(solver)
    energy = float(solver.e_tot)
    truncations = _truncation_history(rows)
    stored_parameters = int(
        sum(np.asarray(factor).size for factor in solver.ground_state.factors)
    )
    record = {
        "optimizer": "two_site_dmrg",
        "symmetry": "none",
        "bond_dim": int(bond_dim),
        "parameter_capacity": _mps_capacity(nsites, bond_dim),
        "stored_parameters": stored_parameters,
        "bond_ranks_capacity": list(_mps_ranks(nsites, bond_dim)),
        "initialization": initialization,
        "initial_energy": initial_energy,
        "energy": energy,
        "energy_per_site": energy / nsites,
        "optimization_seconds": float(seconds),
        "directional_pass_limit": int(pass_limit),
        "directional_passes_completed": len(rows),
        "converged": bool(solver.converged),
        "final_delta_energy": (
            float(abs(rows[-1]["energy"] - rows[-2]["energy"]))
            if len(rows) >= 2
            else None
        ),
        "directional_pass_energies": [float(row["energy"]) for row in rows],
        "directional_pass_truncations": truncations,
        "maximum_reported_truncation": max(
            (value for value in truncations if value is not None),
            default=None,
        ),
    }
    return solver.ground_state.copy(), record


def _lower_energy_mps_candidate(mpo, candidates):
    energies = {name: _mps_energy(state, mpo) for name, state in candidates.items()}
    selected = min(energies, key=energies.get)
    return candidates[selected], selected, energies


def _optimize_references(
    mpo,
    current_mps,
    previous_references,
    *,
    nsites,
    seed,
    pass_limit,
    tolerance,
):
    references = {}
    records = []
    lower_state = current_mps
    for bond_dim in REFERENCE_DIMS:
        candidates = {"lower_d_reference": lower_state}
        previous = previous_references.get(bond_dim)
        if previous is not None:
            candidates["previous_ratio"] = previous
        initial_state, initialization, candidate_energies = _lower_energy_mps_candidate(
            mpo,
            candidates,
        )
        state, record = _optimize_mps(
            mpo,
            nsites=nsites,
            bond_dim=bond_dim,
            seed=_child_seed(seed, 303, bond_dim),
            pass_limit=pass_limit,
            tolerance=tolerance,
            initial_state=initial_state,
            initialization=initialization,
        )
        record.update(
            {
                "method": f"reference_mps_d{bond_dim}",
                "kind": "reference_mps",
                "candidate_energies": candidate_energies,
            }
        )
        references[bond_dim] = state
        records.append(record)
        lower_state = state
    return references, records


def _optimize_letta(
    hamiltonian,
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
    energy = float(state.energy)
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
        "energy": energy,
        "energy_per_site": energy / nsites,
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": float(seconds),
        "warm_mps_seconds": float(warm_mps_seconds),
        "directional_pass_limit": int(pass_limit),
        "directional_passes_completed": len(state.history),
        "converged": bool(state.converged),
        "solver_failures": solver_failures,
        "final_delta_energy": (
            float(abs(state.history[-1]["delta_energy"])) if state.history else None
        ),
        "directional_pass_energies": [float(row["energy"]) for row in state.history],
        "frontier_gauge": bool(frontier_gauge),
        "frontier_gauge_weighting": (
            frontier_gauge_weighting if frontier_gauge else None
        ),
        "frontier_peak_elements": int(state.peak_frontier_elements),
        "cached_environment_elements": int(state.cached_environment_elements),
        "compressed_hamiltonian_mpo_bond_dim": int(
            state.compressed_hamiltonian_mpo_bond_dim
        ),
        "frontier_gauge_bond_attempts": len(gauge_updates),
        "applied_frontier_gauges": len(applied_gauges),
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


def _annotate_energy_errors(records, references, nsites):
    reference_by_ratio = {}
    for row in references:
        if row["bond_dim"] != max(REFERENCE_DIMS):
            continue
        ratio = float(row["j2_ratio"])
        reference_by_ratio[ratio] = min(
            float(row["energy"]),
            reference_by_ratio.get(ratio, float("inf")),
        )
    best_by_ratio = dict(reference_by_ratio)
    for row in records:
        ratio = float(row["j2_ratio"])
        best_by_ratio[ratio] = min(
            float(row["energy"]),
            best_by_ratio.get(ratio, float("inf")),
        )
    for row in records:
        ratio = float(row["j2_ratio"])
        row["energy_above_reference_per_site"] = (
            float(row["energy"]) - reference_by_ratio[ratio]
        ) / nsites
        row["energy_above_best_per_site"] = (
            float(row["energy"]) - best_by_ratio[ratio]
        ) / nsites
    for row in references:
        ratio = float(row["j2_ratio"])
        row["energy_above_best_per_site"] = (
            float(row["energy"]) - best_by_ratio[ratio]
        ) / nsites


def _summaries(records):
    summaries = {}
    methods = sorted({row["method"] for row in records})
    ratios = sorted({float(row["j2_ratio"]) for row in records})
    for method in methods:
        per_ratio = {}
        for ratio in ratios:
            rows = [
                row
                for row in records
                if row["method"] == method
                and np.isclose(row["j2_ratio"], ratio, atol=1.0e-14, rtol=0.0)
            ]
            if not rows:
                continue
            entry = {
                "runs": len(rows),
                "parameter_capacity": int(rows[0]["parameter_capacity"]),
                "converged_runs": int(sum(bool(row["converged"]) for row in rows)),
                "total_solver_failures": int(
                    sum(int(row.get("solver_failures", 0)) for row in rows)
                ),
            }
            for field in (
                "energy",
                "energy_per_site",
                "energy_above_reference_per_site",
                "energy_above_best_per_site",
                "optimization_seconds",
                "directional_passes_completed",
                "final_delta_energy",
            ):
                values = np.asarray(
                    [float(row[field]) for row in rows if row.get(field) is not None]
                )
                if not values.size:
                    continue
                entry[f"median_{field}"] = float(np.median(values))
                entry[f"interquartile_{field}"] = [
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.75)),
                ]
            per_ratio[f"{ratio:.12g}"] = entry
        summaries[method] = per_ratio
    return summaries


def _payload(model, settings, records, references):
    nsites = int(model["nrows"]) * int(model["ncols"])
    records = [dict(row) for row in records]
    references = [dict(row) for row in references]
    _annotate_energy_errors(records, references, nsites)
    records.sort(key=lambda row: (row["j2_ratio"], row["seed"], row["method"]))
    references.sort(key=lambda row: (row["j2_ratio"], row["seed"], row["bond_dim"]))
    return {
        "model": model,
        "settings": settings,
        "reference_runs": references,
        "records": records,
        "summary": _summaries(records),
    }


def run_scan(
    *,
    ratios=DEFAULT_RATIOS,
    seeds=DEFAULT_SEEDS,
    mps_passes=80,
    letta_passes=80,
    reference_passes=50,
    tolerance=1.0e-9,
    tie_noise=1.0e-3,
    frontier_gauge=True,
    frontier_gauge_weighting="uniform",
    checkpoint_path=None,
    optimize_letta=True,
):
    ratios = _validated_scan_ratios(ratios)
    seeds = _validated_seeds(seeds)
    nrows, ncols = 8, 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    model = {
        "nrows": nrows,
        "ncols": ncols,
        "j1": 1.0,
        "j2_ratios": sorted(ratios),
        "boundary": "open",
        "site_order": "row-wise-snake-across-width-4",
        "letta_tie_graph": "all-j1-nearest-neighbor-bonds",
        "letta_tie_edges": len(nearest),
        "j2_diagonal_edges": len(diagonals),
    }
    settings = {
        "seeds": list(seeds),
        "scan_direction": (
            "descending-continuation-with-lower-d-mps-candidate"
            if len(ratios) > 1 and ratios[-1] < ratios[0]
            else "ascending-continuation-with-lower-d-mps-candidate"
        ),
        "scan_path": list(ratios),
        "reported_mps_bond_dims": list(REPORTED_MPS_DIMS),
        "auxiliary_mps_bond_dims": list(AUXILIARY_MPS_DIMS),
        "letta_bond_dims": list(LETTA_DIMS),
        "reference_mps_bond_dims": list(REFERENCE_DIMS),
        "mps_directional_pass_limit": int(mps_passes),
        "letta_directional_pass_limit": int(letta_passes),
        "reference_directional_pass_limit": int(reference_passes),
        "tolerance": float(tolerance),
        "tie_noise": float(tie_noise),
        "frontier_gauge": bool(frontier_gauge),
        "frontier_gauge_weighting": (
            frontier_gauge_weighting if frontier_gauge else None
        ),
        "full_state_vectors_constructed": False,
        "sparse_full_hamiltonian_constructed": False,
        "reference_is_exact": False,
        "reference_note": (
            "lowest observed MPS D32 energy; D16 is retained as a convergence proxy"
        ),
        "letta_optimization_enabled": bool(optimize_letta),
    }

    records = []
    reference_records = []
    previous_mps = {
        (seed, bond_dim): None
        for seed in seeds
        for bond_dim in (*AUXILIARY_MPS_DIMS, *REPORTED_MPS_DIMS)
    }
    previous_letta = {
        (seed, bond_dim): None for seed in seeds for bond_dim in LETTA_DIMS
    }
    previous_references = {
        (seed, bond_dim): None for seed in seeds for bond_dim in REFERENCE_DIMS
    }

    def write_checkpoint():
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = _payload(model, settings, records, reference_records)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for ratio_index, ratio in enumerate(ratios):
        weighted_bonds = tuple((i, j, 1.0) for i, j in nearest)
        weighted_bonds += tuple((i, j, ratio) for i, j in diagonals)
        hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
        local_mpo = hamiltonian.to_mpo().compress()
        mpo = MPO(list(local_mpo.tensors))
        print(
            f"ratio={ratio:.3f} mpo_D={max(local_mpo.bond_dims)}",
            flush=True,
        )

        for seed in seeds:
            current_mps = {}
            mps_records = {}
            for bond_dim in (*AUXILIARY_MPS_DIMS, *REPORTED_MPS_DIMS):
                previous = previous_mps[(seed, bond_dim)]
                candidates = {}
                if previous is not None:
                    candidates["previous_ratio"] = previous
                if current_mps:
                    lower_dim = max(current_mps)
                    candidates[f"current_mps_d{lower_dim}"] = current_mps[lower_dim]
                if candidates:
                    initial_state, initialization, candidate_energies = (
                        _lower_energy_mps_candidate(mpo, candidates)
                    )
                else:
                    initial_state = None
                    initialization = "random"
                    candidate_energies = None
                state, record = _optimize_mps(
                    mpo,
                    nsites=nsites,
                    bond_dim=bond_dim,
                    seed=_child_seed(seed, 101, bond_dim),
                    pass_limit=mps_passes,
                    tolerance=tolerance,
                    initial_state=initial_state,
                    initialization=initialization,
                )
                if candidate_energies is not None:
                    record["candidate_energies"] = candidate_energies
                previous_mps[(seed, bond_dim)] = state.copy()
                current_mps[bond_dim] = state
                mps_records[bond_dim] = record
                if bond_dim in REPORTED_MPS_DIMS:
                    record.update(
                        {
                            "method": f"mps_d{bond_dim}",
                            "kind": "mps",
                            "solver_failures": 0,
                        }
                    )
                    records.append({"j2_ratio": ratio, "seed": seed, **record})
                print(
                    f"  seed={seed} mps_d{bond_dim} E/N={record['energy_per_site']:.9f} "
                    f"passes={record['directional_passes_completed']} "
                    f"time={record['optimization_seconds']:.2f}s",
                    flush=True,
                )

            reference_state_map = {
                bond_dim: previous_references[(seed, bond_dim)]
                for bond_dim in REFERENCE_DIMS
            }
            reference_states, seed_reference_records = _optimize_references(
                mpo,
                current_mps[max(REPORTED_MPS_DIMS)],
                reference_state_map,
                nsites=nsites,
                seed=seed,
                pass_limit=reference_passes,
                tolerance=tolerance,
            )
            for bond_dim, state in reference_states.items():
                previous_references[(seed, bond_dim)] = state.copy()
            for record in seed_reference_records:
                record.update({"j2_ratio": ratio, "seed": seed})
                reference_records.append(record)
                print(
                    f"  seed={seed} ref_d{record['bond_dim']} "
                    f"E/N={record['energy_per_site']:.9f} "
                    f"passes={record['directional_passes_completed']} "
                    f"time={record['optimization_seconds']:.2f}s",
                    flush=True,
                )

            if not optimize_letta:
                write_checkpoint()
                continue

            for bond_dim in LETTA_DIMS:
                state, record = _optimize_letta(
                    hamiltonian,
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
                write_checkpoint()
                print(
                    f"  seed={seed} letta_d{bond_dim} E/N={record['energy_per_site']:.9f} "
                    f"init={record['initialization']} "
                    f"passes={record['directional_passes_completed']} "
                    f"time={record['optimization_seconds']:.2f}s",
                    flush=True,
                )

    return _payload(model, settings, records, reference_records)


def merge_results(paths):
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not inputs:
        raise ValueError("at least one result is required for merging.")
    model = inputs[0]["model"]
    settings = dict(inputs[0]["settings"])
    records = []
    references = []
    seen_records = set()
    seen_references = set()
    seeds = set()
    for result in inputs:
        if result["model"] != model:
            raise ValueError("cannot merge scans with different models.")
        comparable_settings = dict(result["settings"])
        comparable_settings.pop("seeds", None)
        baseline_settings = dict(settings)
        baseline_settings.pop("seeds", None)
        if comparable_settings != baseline_settings:
            raise ValueError("cannot merge scans with different settings.")
        for row in result["records"]:
            key = (float(row["j2_ratio"]), int(row["seed"]), row["method"])
            if key in seen_records:
                raise ValueError(f"duplicate result record {key}.")
            seen_records.add(key)
            seeds.add(int(row["seed"]))
            records.append(row)
        for row in result["reference_runs"]:
            key = (
                float(row["j2_ratio"]),
                int(row["seed"]),
                int(row["bond_dim"]),
            )
            if key in seen_references:
                raise ValueError(f"duplicate reference record {key}.")
            seen_references.add(key)
            references.append(row)
    settings["seeds"] = sorted(seeds)
    return _payload(model, settings, records, references)


def combine_branches(paths):
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(inputs) < 2:
        raise ValueError("at least two scans are required for branch selection.")

    def model_without_ratios(result):
        model = dict(result["model"])
        model.pop("j2_ratios", None)
        return model

    baseline_model = model_without_ratios(inputs[0])
    baseline_settings = dict(inputs[0]["settings"])
    for field in ("seeds", "scan_direction", "scan_path"):
        baseline_settings.pop(field, None)

    records_by_key = {}
    references_by_key = {}
    paths_by_branch = {}
    seeds = set()
    ratios = set()
    for index, result in enumerate(inputs):
        if model_without_ratios(result) != baseline_model:
            raise ValueError("cannot combine branches from different models.")
        comparable_settings = dict(result["settings"])
        for field in ("seeds", "scan_direction", "scan_path"):
            comparable_settings.pop(field, None)
        if comparable_settings != baseline_settings:
            raise ValueError("cannot combine branches with different settings.")
        direction = str(result["settings"].get("scan_direction", f"branch-{index}"))
        branch = direction.split("-", 1)[0]
        if branch in paths_by_branch:
            branch = f"{branch}-{index}"
        paths_by_branch[branch] = list(
            result["settings"].get("scan_path", result["model"]["j2_ratios"])
        )
        for row in result["records"]:
            key = (float(row["j2_ratio"]), int(row["seed"]), row["method"])
            records_by_key.setdefault(key, {})[branch] = row
            seeds.add(int(row["seed"]))
            ratios.add(float(row["j2_ratio"]))
        for row in result["reference_runs"]:
            key = (
                float(row["j2_ratio"]),
                int(row["seed"]),
                int(row["bond_dim"]),
            )
            references_by_key.setdefault(key, {})[branch] = row

    def select_rows(grouped):
        selected = []
        for candidates in grouped.values():
            branch = min(candidates, key=lambda name: float(candidates[name]["energy"]))
            row = dict(candidates[branch])
            row["selected_branch"] = branch
            row["branch_candidate_energies"] = {
                name: float(candidate["energy"])
                for name, candidate in sorted(candidates.items())
            }
            selected.append(row)
        return selected

    model = dict(baseline_model)
    model["j2_ratios"] = sorted(ratios)
    settings = dict(baseline_settings)
    settings.update(
        {
            "seeds": sorted(seeds),
            "scan_direction": "bidirectional-lower-energy-branch-selection",
            "scan_paths": paths_by_branch,
            "branch_selection": "lowest variational energy for each ratio/seed/method",
        }
    )
    return _payload(
        model,
        settings,
        select_rows(records_by_key),
        select_rows(references_by_key),
    )


def select_best_results(paths):
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(inputs) < 2:
        raise ValueError("at least two result files are required for selection.")

    def model_without_ratios(result):
        model = dict(result["model"])
        model.pop("j2_ratios", None)
        return model

    baseline_model = model_without_ratios(inputs[0])
    records_by_key = {}
    references_by_key = {}
    source_runs = {}
    seeds = set()
    ratios = set()
    for index, (path, result) in enumerate(zip(paths, inputs)):
        if model_without_ratios(result) != baseline_model:
            raise ValueError("cannot select runs from different models.")
        source = f"source_{index}"
        source_runs[source] = {
            "path": str(path),
            "j2_ratios": list(result["model"]["j2_ratios"]),
            "settings": result["settings"],
        }
        for row in result["records"]:
            key = (float(row["j2_ratio"]), int(row["seed"]), row["method"])
            records_by_key.setdefault(key, {})[source] = row
            seeds.add(int(row["seed"]))
            ratios.add(float(row["j2_ratio"]))
        for row in result["reference_runs"]:
            key = (
                float(row["j2_ratio"]),
                int(row["seed"]),
                int(row["bond_dim"]),
            )
            references_by_key.setdefault(key, {})[source] = row

    def select_rows(grouped):
        selected = []
        for candidates in grouped.values():
            source = min(candidates, key=lambda name: float(candidates[name]["energy"]))
            row = dict(candidates[source])
            row["selected_source"] = source
            row["source_candidate_energies"] = {
                name: float(candidate["energy"])
                for name, candidate in sorted(candidates.items())
            }
            selected.append(row)
        return selected

    common_settings = {}
    first_settings = inputs[0]["settings"]
    for key, value in first_settings.items():
        if all(result["settings"].get(key, object()) == value for result in inputs[1:]):
            common_settings[key] = value
    common_settings.update(
        {
            "seeds": sorted(seeds),
            "result_selection": "lowest variational energy for each ratio/seed/method",
            "source_runs": source_runs,
        }
    )
    model = dict(baseline_model)
    model["j2_ratios"] = sorted(ratios)
    return _payload(
        model,
        common_settings,
        select_rows(records_by_key),
        select_rows(references_by_key),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mps-passes", type=int, default=80)
    parser.add_argument("--letta-passes", type=int, default=80)
    parser.add_argument("--reference-passes", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--no-frontier-gauge", action="store_true")
    parser.add_argument("--skip-letta", action="store_true")
    parser.add_argument(
        "--frontier-gauge-weighting",
        choices=("uniform", "probability"),
        default="uniform",
    )
    parser.add_argument("--merge-results", type=Path, nargs="+")
    parser.add_argument("--combine-branches", type=Path, nargs="+")
    parser.add_argument("--select-best-results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection_modes = sum(
        option is not None
        for option in (
            args.merge_results,
            args.combine_branches,
            args.select_best_results,
        )
    )
    if selection_modes > 1:
        parser.error("result merge/selection options are mutually exclusive.")
    if args.merge_results:
        result = merge_results(args.merge_results)
    elif args.combine_branches:
        result = combine_branches(args.combine_branches)
    elif args.select_best_results:
        result = select_best_results(args.select_best_results)
    else:
        result = run_scan(
            ratios=args.ratios,
            seeds=args.seeds,
            mps_passes=args.mps_passes,
            letta_passes=args.letta_passes,
            reference_passes=args.reference_passes,
            tolerance=args.tolerance,
            tie_noise=args.tie_noise,
            frontier_gauge=not args.no_frontier_gauge,
            frontier_gauge_weighting=args.frontier_gauge_weighting,
            checkpoint_path=args.output,
            optimize_letta=not args.skip_letta,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
