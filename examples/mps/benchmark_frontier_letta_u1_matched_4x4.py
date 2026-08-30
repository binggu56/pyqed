#!/usr/bin/env python3
"""Convergence and matched-parameter U(1) graph-LETTA study on 4x4 J1-J2.

The exact ground-state energy is evaluated only after every variational run.
It is never used to initialize, expand, select, or relax a variational state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from examples.mps.benchmark_frontier_letta_u1_4x4 import (
    _dense_mps_run,
    _neel_cores,
    _sector_leakage,
    _u1_mps_run,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA, abelian_frontier_tied_letta_from_mps


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_u1_matched_4x4.json"


def _saturated_bond_dims(nsites, maximum):
    """Open-boundary ranks capped by the Hilbert space on either side."""
    return tuple(
        min(int(maximum), 2 ** min(cut, int(nsites) - cut))
        for cut in range(int(nsites) + 1)
    )


def _history_summary(state, *, seconds, initial_energy):
    energies = [float(row["energy"]) for row in state.history]
    updates = [update for row in state.history for update in row["updates"]]
    gauges = [
        update
        for row in state.history
        for update in (row.get("frontier_gauge") or ())
    ]
    energy = float(state.expectation())
    if len(energies) > 1:
        final_delta = abs(energies[-1] - energies[-2])
    elif energies:
        final_delta = abs(energies[-1] - float(initial_energy))
    else:
        final_delta = 0.0
    tail_start = max(0, len(energies) - 10)
    tail_reference = (
        float(initial_energy) if tail_start == 0 else energies[tail_start - 1]
    )
    finite_identity_errors = [
        float(update.solver_metric_identity_error)
        for update in updates
        if np.isfinite(update.solver_metric_identity_error)
    ]
    rank_fractions = [
        update.metric_rank / max(update.raw_dim, 1)
        for update in updates
        if not update.metric_rank_is_projected
    ]
    maximum_gauge_imbalance = max(
        (
            float(update.imbalance_after)
            for update in gauges
            if update.applied and np.isfinite(update.imbalance_after)
        ),
        default=None,
    )
    symmetry = "U1_fixed_Sz" if hasattr(state, "abelian_layout") else "none"
    gauge_name = "none"
    if gauges:
        gauge_name = "sector_probability" if symmetry != "none" else "probability"
    return {
        "symmetry": symmetry,
        "optimizer": "one_site_exact_local_S_equals_I",
        "frontier_gauge": gauge_name,
        "initial_energy": float(initial_energy),
        "energy": energy,
        "energy_per_site": float(energy / len(state.dims)),
        "seconds": float(seconds),
        "directional_passes": len(energies),
        "directional_pass_energies": energies,
        "final_delta_energy": float(final_delta),
        "last_ten_pass_energy_gain": float(tail_reference - energy),
        "converged_at_1e_8": bool(final_delta < 1.0e-8),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "solver_failures": int(sum(not update.solver_converged for update in updates)),
        "maximum_solver_identity_error": max(finite_identity_errors, default=float("nan")),
        "minimum_local_metric_rank_fraction": min(rank_fractions, default=float("nan")),
        "frontier_gauge_attempts": len(gauges),
        "frontier_gauges_applied": int(sum(update.applied for update in gauges)),
        "maximum_applied_gauge_imbalance_after": maximum_gauge_imbalance,
        "symmetry_parameters": int(state.nparameters),
        "dense_equivalent_parameters": int(
            getattr(state, "dense_nparameters", sum(tensor.size for tensor in state.tensors))
        ),
        "bond_dims": list(state.bond_dims),
        "sector_leakage": float(
            _sector_leakage(state.state_vector(normalize=True), len(state.dims))
        ),
    }


def _run_stage(state, passes, *, frontier_gauge=True):
    initial_energy = float(state.expectation())
    start = perf_counter()
    state.run(
        nsweeps=int(passes),
        tol=0.0,
        solver="whitened",
        gauge="frontier" if frontier_gauge else None,
        gauge_weight="probability",
    )
    return _history_summary(
        state,
        seconds=perf_counter() - start,
        initial_energy=initial_energy,
    )


def _expansion_summary(records):
    rows = [asdict(record) for record in records]
    return {
        "updates": rows,
        "maximum_relative_norm_error": max(
            (float(row["norm_error"]) for row in rows), default=0.0
        ),
        "maximum_absolute_energy_change": max(
            (abs(float(row["energy"] - row["energy_before"])) for row in rows),
            default=0.0,
        ),
        "seeded_directions": int(sum(row["seeded_directions"] for row in rows)),
    }


def _expand_to_parameter_target(state, target, *, maximum, scale, seed):
    """Greedily add symmetry sectors until the allowed count best matches target."""
    target = int(target)
    caps = _saturated_bond_dims(len(state.dims), maximum)
    records = []
    selected_cuts = []
    rng = np.random.default_rng(seed)
    while True:
        current = int(state.nparameters)
        current_error = abs(target - current)
        candidates = []
        for cut in range(1, len(state.dims)):
            if state.bond_dims[cut] >= caps[cut]:
                continue
            labels = state._automatic_expansion_labels(cut, 1)
            layout = state.abelian_layout.with_expanded_bond(cut, labels)
            count = int(
                sum(
                    np.count_nonzero(mask)
                    for mask in layout.local_masks(state.sites)
                )
            )
            error = abs(target - count)
            center_distance = abs(cut - len(state.dims) / 2)
            candidates.append((error, center_distance, cut, count))
        if not candidates:
            break
        error, _center_distance, cut, _count = min(candidates)
        if error >= current_error:
            break
        records.append(
            state.expand_bond(
                cut,
                state.bond_dims[cut] + 1,
                direction="right",
                strategy="residual",
                scale=scale,
                seed=int(rng.integers(np.iinfo(np.int64).max)),
            )
        )
        selected_cuts.append(cut)
    return tuple(records), tuple(selected_cuts)


def benchmark(
    *,
    mps_passes=20,
    dense_passes=40,
    u1_d4_passes=80,
    expansion_d4_passes=20,
    expansion_d6_passes=10,
    expansion_matched_passes=20,
    u1_d8_passes=20,
    tie_noise=1.0e-3,
    expansion_scale=1.0e-3,
    seed=7,
):
    nrows = ncols = 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, 0.5) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
    dense_mpo = hamiltonian.to_mpo().compress()
    parent_sets = parent_sets_from_edges(nsites, nearest)
    product = _neel_cores(nsites)

    _dense_solver, dense_mps = _dense_mps_run(
        dense_mpo,
        product,
        bond_dim=4,
        sweeps=mps_passes,
        tolerance=1.0e-10,
    )
    _u1_solver, u1_mps_dense, layout, u1_mps = _u1_mps_run(
        dense_mpo,
        product,
        bond_dim=4,
        sweeps=mps_passes,
        tolerance=1.0e-10,
    )
    warm_u1 = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parent_sets,
        u1_mps_dense.factors,
        abelian_layout=layout,
        tie_noise=tie_noise,
        seed=seed,
        frontier_backend="identity_block",
    )
    warm_energy = float(warm_u1.expectation())

    dense = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dims=warm_u1.bond_dims,
        tensors=[tensor.copy() for tensor in warm_u1.tensors],
        frontier_backend="identity_block",
    )
    # Construction applies a state-preserving scalar balance.  Restore the
    # shared tensor coordinates so dense and U(1) runs start from exactly the
    # same local parametrization as well as the same physical state.
    dense.tensors = [tensor.copy() for tensor in warm_u1.tensors]
    dense.energy = dense.expectation()
    dense_initial_difference = float(
        np.max(np.abs(dense.state_vector() - warm_u1.state_vector()))
    )
    # Keep the established dense D=4 reference ungauged.  The sector gauge is
    # the intervention under study and is applied to every U(1) branch below.
    dense_result = _run_stage(dense, dense_passes, frontier_gauge=False)

    fixed_d4 = warm_u1.copy()
    fixed_d4_result = _run_stage(fixed_d4, u1_d4_passes)
    if u1_d4_passes >= 40:
        fixed_d4_result["energy_at_pass_40"] = float(
            fixed_d4_result["directional_pass_energies"][39]
        )
        fixed_d4_result["gain_after_pass_40"] = float(
            fixed_d4_result["energy_at_pass_40"] - fixed_d4_result["energy"]
        )

    expanded = warm_u1.copy()
    d4_stage = _run_stage(expanded, expansion_d4_passes)
    d6_dims = _saturated_bond_dims(nsites, 6)
    d6_expansions = expanded.expand_bond_dims(
        d6_dims,
        direction="right",
        strategy="residual",
        scale=expansion_scale,
        seed=seed + 100,
    )
    d6_expansion = _expansion_summary(d6_expansions)
    d6_expansion["bond_dims"] = list(expanded.bond_dims)
    d6_expansion["symmetry_parameters"] = int(expanded.nparameters)
    d6_expansion["dense_equivalent_parameters"] = int(expanded.dense_nparameters)
    d6_stage = _run_stage(expanded, expansion_d6_passes)

    # Match the unrestricted D=4 variational count using only structural
    # support sizes.  Exact-state data and exact energies are unavailable here.
    matched_expansion_records, matched_cuts = _expand_to_parameter_target(
        expanded,
        dense_result["symmetry_parameters"],
        maximum=8,
        scale=expansion_scale,
        seed=seed + 200,
    )
    matched_expansion = _expansion_summary(matched_expansion_records)
    matched_expansion["selected_cuts"] = list(matched_cuts)
    matched_expansion["selection_rule"] = (
        "greedy closest symmetry-allowed count up to saturated D=8; "
        "central-cut tie break; no exact-state information"
    )
    matched_expansion["target_symmetry_parameters"] = int(
        dense_result["symmetry_parameters"]
    )
    matched_expansion["bond_dims"] = list(expanded.bond_dims)
    matched_expansion["symmetry_parameters"] = int(expanded.nparameters)
    matched_expansion["dense_equivalent_parameters"] = int(expanded.dense_nparameters)
    matched_result = _run_stage(expanded, expansion_matched_passes)

    # Isolate the larger MPS construction from the established D=4 warm-start
    # path by using a fresh product-state object after all D=4 branches.
    _u1_d8_solver, u1_d8_mps_dense, layout_d8, u1_d8_mps = _u1_mps_run(
        dense_mpo,
        _neel_cores(nsites),
        bond_dim=8,
        sweeps=mps_passes,
        tolerance=1.0e-10,
    )
    _dense_d8_solver, dense_d8_mps = _dense_mps_run(
        dense_mpo,
        _neel_cores(nsites),
        bond_dim=8,
        sweeps=mps_passes,
        tolerance=1.0e-10,
    )
    dense_d8_mps["symmetry_parameters"] = int(dense_d8_mps["parameters"])
    dense_d8_mps["dense_equivalent_parameters"] = int(
        dense_d8_mps["parameters"]
    )
    fresh_d8 = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parent_sets,
        u1_d8_mps_dense.factors,
        abelian_layout=layout_d8,
        tie_noise=tie_noise,
        seed=seed,
        frontier_backend="identity_block",
    )
    fresh_d8_result = _run_stage(fresh_d8, u1_d8_passes)

    sparse = sparse_heisenberg_hamiltonian(nsites, weighted)
    exact_start = perf_counter()
    exact_energy = float(
        eigsh(sparse, k=1, which="SA", return_eigenvectors=False, tol=1.0e-11)[0]
    )
    exact_seconds = perf_counter() - exact_start
    for record in (
        dense_result,
        fixed_d4_result,
        d4_stage,
        d6_stage,
        matched_result,
        fresh_d8_result,
    ):
        record["energy_error"] = float(record["energy"] - exact_energy)
        record["energy_error_per_site"] = float(
            (record["energy"] - exact_energy) / nsites
        )

    dense_capacity = dense_result["symmetry_parameters"]
    return {
        "model": {
            "shape": [nrows, ncols],
            "j1": 1.0,
            "j2": 0.5,
            "boundary": "open",
            "target_two_sz": 0,
            "site_order": "row-wise snake",
            "tie_graph": "all J1 nearest-neighbor bonds",
        },
        "settings": {
            "mps_directional_pass_limit": int(mps_passes),
            "dense_letta_directional_pass_limit": int(dense_passes),
            "u1_d4_directional_pass_limit": int(u1_d4_passes),
            "expansion_stage_directional_pass_limits": [
                int(expansion_d4_passes),
                int(expansion_d6_passes),
                int(expansion_matched_passes),
            ],
            "fresh_u1_d8_directional_pass_limit": int(u1_d8_passes),
            "tie_noise": float(tie_noise),
            "expansion_scale": float(expansion_scale),
            "seed": int(seed),
            "local_solver": "whitened_exact_S_equals_I",
            "dense_d4_frontier_gauge": "none",
            "u1_frontier_gauge": "sector_probability",
        },
        "exact": {
            "energy": exact_energy,
            "seconds": float(exact_seconds),
            "used_during_optimization": False,
            "exact_vector_computed": False,
        },
        "initialization": {
            "u1_mps_warm_energy": warm_energy,
            "dense_and_u1_max_state_difference": dense_initial_difference,
        },
        "mps": {
            "dense_d4": dense_mps,
            "u1_d4": u1_mps,
            "dense_d8": dense_d8_mps,
            "u1_d8": u1_d8_mps,
        },
        "letta": {
            "dense_d4": dense_result,
            "u1_fixed_d4_long": fixed_d4_result,
            "u1_variable_d6_to_d8_matched": {
                "d4_relaxation": d4_stage,
                "d4_to_d6_expansion": d6_expansion,
                "d6_relaxation": d6_stage,
                "selected_d6_to_d8_expansion": matched_expansion,
                "matched_relaxation": matched_result,
                "total_directional_passes": int(
                    expansion_d4_passes
                    + expansion_d6_passes
                    + expansion_matched_passes
                ),
                "total_relaxation_seconds": float(
                    d4_stage["seconds"] + d6_stage["seconds"] + matched_result["seconds"]
                ),
            },
            "u1_fresh_d8": fresh_d8_result,
        },
        "comparisons": {
            "u1_d4_allowed_parameter_fraction_of_dense_d4": float(
                fixed_d4_result["symmetry_parameters"] / dense_capacity
            ),
            "matched_u1_allowed_parameter_fraction_of_dense_d4": float(
                matched_result["symmetry_parameters"] / dense_capacity
            ),
            "matched_minus_dense_d4_energy": float(
                matched_result["energy"] - dense_result["energy"]
            ),
            "fixed_u1_d4_minus_dense_d4_energy": float(
                fixed_d4_result["energy"] - dense_result["energy"]
            ),
            "fresh_u1_d8_allowed_parameter_fraction_of_dense_d4": float(
                fresh_d8_result["symmetry_parameters"] / dense_capacity
            ),
            "fresh_u1_d8_minus_dense_d4_energy": float(
                fresh_d8_result["energy"] - dense_result["energy"]
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-passes", type=int, default=20)
    parser.add_argument("--dense-passes", type=int, default=40)
    parser.add_argument("--u1-d4-passes", type=int, default=80)
    parser.add_argument("--expansion-d4-passes", type=int, default=20)
    parser.add_argument("--expansion-d6-passes", type=int, default=10)
    parser.add_argument("--expansion-matched-passes", type=int, default=20)
    parser.add_argument("--u1-d8-passes", type=int, default=20)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--expansion-scale", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark(
        mps_passes=args.mps_passes,
        dense_passes=args.dense_passes,
        u1_d4_passes=args.u1_d4_passes,
        expansion_d4_passes=args.expansion_d4_passes,
        expansion_d6_passes=args.expansion_d6_passes,
        expansion_matched_passes=args.expansion_matched_passes,
        u1_d8_passes=args.u1_d8_passes,
        tie_noise=args.tie_noise,
        expansion_scale=args.expansion_scale,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
