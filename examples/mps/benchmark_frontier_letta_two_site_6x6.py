#!/usr/bin/env python3
"""Screen environment-retracted LETTA pair updates on the saved 6x6 state.

Every candidate starts from the same saved variational state.  Candidate
selection uses only a fresh post-retraction energy contraction; no exact state
or reference energy enters the selection.  The best strictly improving update
is retained and checkpointed.  The conditional SVD is retained only as an
initializer and truncation diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.continue_frontier_letta_bond_expansion_6x6 import (
    DEFAULT_SOURCE_RESULT,
    DEFAULT_SOURCE_SNAPSHOT as DEFAULT_SOURCE_STATE,
    _load_state,
)
from pyqed.letta import tie_frontier_cost


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_two_site_6x6.json"
DEFAULT_OUTPUT_STATE = RESULTS / "frontier_letta_two_site_6x6.npz"
# Three pairs around each enlarged cut.  The central pairs span D=6 cuts.
DEFAULT_PAIR_SITES = (13, 14, 15, 19, 20, 21)
DEFAULT_CONTROL_PAIR_SITES = (14, 20)


def _save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        tie_edges=np.asarray(
            [
                (site, parent)
                for site, parents in enumerate(state.parent_sets)
                for parent in parents
            ],
            dtype=np.int64,
        ),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )


def _frontier_cost(dims, parent_sets, bond_dims):
    cost = tie_frontier_cost(
        dims,
        parent_sets,
        bond_dims=bond_dims,
    )
    return {
        "peak_width": int(cost.peak_width),
        "peak_physical_states": int(cost.peak_physical_states),
        "peak_norm_elements": int(cost.peak_norm_elements),
        "total_norm_elements": int(cost.total_norm_elements),
        "log_objective": float(cost.log_objective),
    }


def _temporary_pair_frontier_cost(state, site):
    """Return the physical-frontier proxy used by the merged pair state."""
    _merged, union_sites = state._merged_pair_tensor(site)
    temporary_parents = list(state.parent_sets)
    temporary_parents[site] = tuple(
        physical for physical in union_sites if physical != site
    )
    temporary_bonds = list(state.bond_dims)
    temporary_bonds[site + 1] = temporary_bonds[site + 2]
    return _frontier_cost(
        state.dims,
        tuple(temporary_parents),
        tuple(temporary_bonds),
    )


def _candidate_record(
    source,
    site,
    *,
    solver,
    metric_tol,
    eig_tol,
    maxiter,
    max_subspace,
):
    total_start = perf_counter()
    copy_start = perf_counter()
    candidate = source.copy()
    copy_seconds = perf_counter() - copy_start
    temporary_cost = _temporary_pair_frontier_cost(candidate, site)
    update_start = perf_counter()
    update = candidate.optimize_two_sites(
        site,
        solver=solver,
        metric_tol=metric_tol,
        eig_tol=eig_tol,
        maxiter=maxiter,
        max_subspace=max_subspace,
    )
    update_seconds = perf_counter() - update_start
    accepted_energy = float(candidate.expectation())
    source_energy = float(source.expectation())
    attempted_energy = float(update.attempted_energy)
    attempted_gain = float(source_energy - attempted_energy)
    accepted_gain = float(source_energy - accepted_energy)
    full_ranks = tuple(int(value) for value in update.conditional_ranks)
    retained_ranks = tuple(
        min(rank, int(update.old_bond_dimension)) for rank in full_ranks
    )
    record = {
        "sites": [int(site), int(site + 1)],
        "spanned_cut": int(site + 1),
        "overlap_sites": [int(value) for value in update.overlap_sites],
        "raw_merged_dimension": int(update.raw_merged_dim),
        "old_bond_dimension": int(update.old_bond_dimension),
        "temporary_bond_dimension": int(update.temporary_bond_dimension),
        "full_conditional_ranks": list(full_ranks),
        "full_conditional_rank_min": int(min(full_ranks, default=0)),
        "full_conditional_rank_max": int(max(full_ranks, default=0)),
        "retained_conditional_ranks": list(retained_ranks),
        "maximum_discarded_rank_channels": int(
            max(
                (rank - retained for rank, retained in zip(full_ranks, retained_ranks)),
                default=0,
            )
        ),
        "relative_truncation_error": float(update.relative_truncation_error),
        "split_strategy": update.split_strategy,
        "selected_retraction_start": update.selected_start,
        "environment_metric_projection_error": float(update.metric_projection_error),
        "factor_sweeps": int(update.factor_sweeps),
        "factor_accepted_half_updates": int(update.factor_accepted_updates),
        "factor_random_starts": int(update.factor_random_starts),
        "energy_before": float(update.energy_before),
        "merged_energy_before_split": float(update.merged_energy),
        "attempted_post_retraction_energy": attempted_energy,
        "attempted_post_retraction_gain": attempted_gain,
        "attempted_post_split_energy": attempted_energy,
        "attempted_post_split_gain": attempted_gain,
        "accepted_energy": accepted_energy,
        "accepted_energy_gain": accepted_gain,
        "accepted": bool(update.accepted),
        "local_solver": update.local_update.solver,
        "local_solver_converged": bool(update.local_update.solver_converged),
        "local_solver_message": update.local_update.message,
        "local_metric_rank": int(update.local_update.metric_rank),
        "local_residual_norm": float(update.local_update.residual_norm),
        "local_coordinate_residual_norm": float(
            update.local_update.solver_coordinate_residual_norm
        ),
        "local_metric_is_identity": bool(update.local_update.solver_metric_is_identity),
        "local_metric_identity_error": float(
            update.local_update.solver_metric_identity_error
        ),
        "temporary_frontier_cost": temporary_cost,
        "timing_seconds": {
            "copy_and_replan": float(copy_seconds),
            "merge_optimize_retract": float(update_seconds),
            "total": float(perf_counter() - total_start),
        },
    }
    return candidate, record


def _one_site_pair_control(
    source,
    site,
    *,
    solver,
    metric_tol,
    eig_tol,
    maxiter,
    max_subspace,
):
    """Optimize the same two tensors separately from the common source."""
    total_start = perf_counter()
    control = source.copy()
    copy_seconds = perf_counter() - total_start
    update_start = perf_counter()
    updates = [
        control.optimize_site(
            current,
            solver=solver,
            metric_tol=metric_tol,
            eig_tol=eig_tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
        )
        for current in (site, site + 1)
    ]
    update_seconds = perf_counter() - update_start
    energy = float(control.expectation())
    source_energy = float(source.expectation())
    return {
        "sites": [int(site), int(site + 1)],
        "energy": energy,
        "energy_gain": float(source_energy - energy),
        "accepted_updates": [bool(update.accepted) for update in updates],
        "solver_converged": [bool(update.solver_converged) for update in updates],
        "raw_local_dimensions": [int(update.raw_dim) for update in updates],
        "timing_seconds": {
            "copy_and_replan": float(copy_seconds),
            "two_one_site_updates": float(update_seconds),
            "total": float(perf_counter() - total_start),
        },
    }


def run_benchmark(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_state=DEFAULT_SOURCE_STATE,
    output=DEFAULT_OUTPUT,
    output_state=DEFAULT_OUTPUT_STATE,
    pair_sites=DEFAULT_PAIR_SITES,
    control_pair_sites=DEFAULT_CONTROL_PAIR_SITES,
    solver="whitened",
    metric_tol=1.0e-12,
    eig_tol=1.0e-10,
    maxiter=1600,
    max_subspace=96,
    minimum_gain=1.0e-12,
):
    total_start = perf_counter()
    source, source_payload = _load_state(source_result, source_state)
    source_energy = float(source.expectation())
    pair_sites = tuple(dict.fromkeys(int(site) for site in pair_sites))
    control_pair_sites = tuple(dict.fromkeys(int(site) for site in control_pair_sites))
    if not pair_sites:
        raise ValueError("pair_sites must not be empty.")
    if any(site < 0 or site + 1 >= len(source.dims) for site in pair_sites):
        raise ValueError("pair_sites contains an invalid left site.")

    source_cost = _frontier_cost(
        source.dims,
        source.parent_sets,
        source.bond_dims,
    )
    source_network_cost = {
        "peak_frontier_elements": int(source.peak_frontier_elements),
        "norm_peak_frontier_elements": int(source.norm_peak_frontier_elements),
        "hamiltonian_peak_frontier_elements": int(
            source.hamiltonian_peak_frontier_elements
        ),
        "cached_environment_elements": int(source.cached_environment_elements),
    }
    candidates = []
    best_state = None
    best_record = None
    for site in pair_sites:
        candidate, record = _candidate_record(
            source,
            site,
            solver=solver,
            metric_tol=metric_tol,
            eig_tol=eig_tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
        )
        candidates.append(record)
        if (
            record["accepted"]
            and record["accepted_energy_gain"] > float(minimum_gain)
            and (
                best_record is None
                or record["accepted_energy"] < best_record["accepted_energy"]
            )
        ):
            best_state = candidate
            best_record = record
        print(
            "pair",
            tuple(record["sites"]),
            f"attempted_gain={record['attempted_post_split_gain']:.8e}",
            f"trunc={record['relative_truncation_error']:.3e}",
            f"accepted={record['accepted']}",
            f"seconds={record['timing_seconds']['total']:.2f}",
            flush=True,
        )
        if candidate is not best_state:
            del candidate
        gc.collect()

    controls = []
    for site in control_pair_sites:
        if site not in pair_sites:
            raise ValueError("each control pair must also be a screened pair.")
        record = _one_site_pair_control(
            source,
            site,
            solver=solver,
            metric_tol=metric_tol,
            eig_tol=eig_tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
        )
        controls.append(record)
        print(
            "one-site control",
            tuple(record["sites"]),
            f"gain={record['energy_gain']:.8e}",
            f"seconds={record['timing_seconds']['total']:.2f}",
            flush=True,
        )
        gc.collect()

    # One bounded rank-proposal test: pair (13, 14) had appreciable discarded
    # weight at D=4.  Open cut 14 to D=6 without changing the state, then rerun
    # exactly that merged solve.  This is diagnostic and is not mixed into the
    # fixed-ansatz candidate selection above.
    proposal = source.copy()
    proposal_energy_before = float(proposal.expectation())
    proposal_norm_before = float(
        np.real(proposal._norm_frontier.scalar(proposal.tensors))
    )
    expansion_start = perf_counter()
    expansion = proposal.expand_bond(
        14,
        6,
        strategy="zero",
        direction="right",
    )
    expansion_seconds = perf_counter() - expansion_start
    proposal_energy_after = float(proposal.expectation())
    proposal_norm_after = float(
        np.real(proposal._norm_frontier.scalar(proposal.tensors))
    )
    _proposal_state, proposal_pair = _candidate_record(
        proposal,
        13,
        solver=solver,
        metric_tol=metric_tol,
        eig_tol=eig_tol,
        maxiter=maxiter,
        max_subspace=max_subspace,
    )
    rank_proposal = {
        "trigger_pair": [13, 14],
        "trigger_cut": 14,
        "old_dimension": int(expansion.old_dimension),
        "proposed_dimension": int(expansion.new_dimension),
        "initialization": "zero, exactly state-preserving",
        "energy_before_expansion": proposal_energy_before,
        "energy_after_expansion": proposal_energy_after,
        "expansion_energy_error": float(proposal_energy_after - proposal_energy_before),
        "expansion_norm_error": float(proposal_norm_after - proposal_norm_before),
        "expansion_reported_norm_error": float(expansion.norm_error),
        "pair_update": proposal_pair,
        "attempted_gain_vs_unexpanded_source": float(
            source_energy - proposal_pair["attempted_post_split_energy"]
        ),
        "accepted_gain_vs_unexpanded_source": float(
            source_energy - proposal_pair["accepted_energy"]
        ),
        "timing_seconds": {
            "expansion": float(expansion_seconds),
            "pair_total": float(proposal_pair["timing_seconds"]["total"]),
        },
    }
    print(
        "rank proposal cut 14 D4->D6",
        (f"attempted_gain={rank_proposal['attempted_gain_vs_unexpanded_source']:.8e}"),
        f"trunc={proposal_pair['relative_truncation_error']:.3e}",
        f"accepted={proposal_pair['accepted']}",
        flush=True,
    )

    selected = best_state if best_state is not None else source.copy()
    final_energy = float(selected.expectation())
    tolerance = 512.0 * np.finfo(float).eps * max(1.0, abs(source_energy))
    if final_energy > source_energy + tolerance:
        raise RuntimeError("the selected two-site update increased the energy.")
    _save_state(output_state, selected)

    payload = {
        "status": "complete",
        "model": source_payload["model"],
        "protocol": {
            "source_result": str(Path(source_result).resolve()),
            "source_state": str(Path(source_state).resolve()),
            "optimizer": (
                "adjacent merge solve, exact-environment metric retraction, "
                "and guarded factor Rayleigh sweeps"
            ),
            "svd_role": "initializer and Euclidean truncation diagnostic only",
            "retraction": "deterministic; no random starts by default",
            "candidate_selection": (
                "lowest accepted energy after a fresh post-truncation "
                "contraction from a common source; no exact-state or "
                "reference-energy oracle"
            ),
            "applied_update": "best single strictly improving candidate",
            "symmetry": "none",
            "reference_mps_symmetry": "none",
            "solver": str(solver),
            "metric_tol": float(metric_tol),
            "eig_tol": float(eig_tol),
            "maxiter": int(maxiter),
            "max_subspace": int(max_subspace),
            "minimum_selection_gain": float(minimum_gain),
            "pair_left_sites": list(pair_sites),
            "one_site_control_pair_left_sites": list(control_pair_sites),
            "bounded_rank_proposals": 1,
        },
        "source": {
            "energy": source_energy,
            "energy_per_site": source_energy / len(source.dims),
            "parameters": int(source.nparameters),
            "bond_dims": list(source.bond_dims),
            "physical_frontier_cost": source_cost,
            "network_frontier_cost": source_network_cost,
        },
        "candidates": candidates,
        "matched_one_site_pair_controls": controls,
        "rank_proposal": rank_proposal,
        "selection": {
            "selected_sites": (
                None if best_record is None else list(best_record["sites"])
            ),
            "accepted_strict_improvement": best_record is not None,
            "energy": final_energy,
            "energy_per_site": final_energy / len(source.dims),
            "energy_gain": float(source_energy - final_energy),
            "energy_gain_per_site": float(
                (source_energy - final_energy) / len(source.dims)
            ),
            "parameters": int(selected.nparameters),
            "bond_dims": list(selected.bond_dims),
            "peak_frontier_elements": int(selected.peak_frontier_elements),
            "snapshot": str(Path(output_state).resolve()),
        },
        "timing_seconds": {"total": float(perf_counter() - total_start)},
    }
    _write_json(Path(output), payload)
    return payload


def _parse_pair_sites(value):
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-state", type=Path, default=DEFAULT_OUTPUT_STATE)
    parser.add_argument(
        "--pair-sites",
        type=_parse_pair_sites,
        default=DEFAULT_PAIR_SITES,
        help="comma-separated left sites of adjacent pairs",
    )
    parser.add_argument(
        "--control-pair-sites",
        type=_parse_pair_sites,
        default=DEFAULT_CONTROL_PAIR_SITES,
        help="screened pairs receiving a two-one-site-update control",
    )
    parser.add_argument(
        "--solver",
        choices=("whitened", "matrix_free", "block_sparse", "direct"),
        default="whitened",
    )
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--max-subspace", type=int, default=96)
    parser.add_argument("--minimum-gain", type=float, default=1.0e-12)
    args = parser.parse_args()
    payload = run_benchmark(
        source_result=args.source_result,
        source_state=args.source_state,
        output=args.output,
        output_state=args.output_state,
        pair_sites=args.pair_sites,
        control_pair_sites=args.control_pair_sites,
        solver=args.solver,
        metric_tol=args.metric_tol,
        eig_tol=args.eig_tol,
        maxiter=args.maxiter,
        max_subspace=args.max_subspace,
        minimum_gain=args.minimum_gain,
    )
    print(json.dumps(payload["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
