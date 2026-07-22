#!/usr/bin/env python3
"""Canonical-gauge 6x6 J1-J2 benchmark with J2-only graph-tied LETTA."""

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
    _mps_capacity,
    _ordered_mps_factors,
)
from examples.mps.scan_frontier_letta_vs_mps_j2_8x4 import _optimize_mps
from pyqed.letta import frontier_tied_letta_from_mps
from pyqed.mps import MPO


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_j2_tied_6x6.json"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_snapshot(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(tensors)
        },
    )


def _history_records(state):
    records = []
    for row in state.history:
        gauge_updates = tuple(row.get("frontier_gauge") or ())
        applied = [update for update in gauge_updates if update.applied]
        updates = tuple(row["updates"])
        records.append(
            {
                "sweep": int(row["sweep"]),
                "energy": float(row["energy"]),
                "delta_energy": float(row["delta_energy"]),
                "accepted_sites": int(row["accepted_sites"]),
                "solver_failures": int(row["solver_failures"]),
                "maximum_residual_norm": float(
                    max((update.residual_norm for update in updates), default=0.0)
                ),
                "frontier_gauge_bond_attempts": len(gauge_updates),
                "applied_frontier_gauges": len(applied),
                "maximum_frontier_imbalance_after": (
                    max((update.imbalance_after for update in applied), default=None)
                ),
            }
        )
    return records


def _append_sweep_diagnostics(result, state):
    records = _history_records(state)
    result["directional_pass_energies"].extend(
        float(record["energy"]) for record in records
    )
    result.setdefault("directional_pass_diagnostics", []).extend(records)
    result["directional_passes_completed"] += len(records)
    updates = [update for row in state.history for update in row["updates"]]
    result["solver_failures"] += int(
        sum(not update.solver_converged for update in updates)
    )
    result["accepted_updates"] += int(sum(update.accepted for update in updates))
    result["site_updates"] += len(updates)
    gauge_updates = [
        update
        for row in state.history
        for update in tuple(row.get("frontier_gauge") or ())
    ]
    applied_gauges = [update for update in gauge_updates if update.applied]
    result["frontier_gauge_bond_attempts"] += len(gauge_updates)
    result["applied_frontier_gauges"] += len(applied_gauges)
    if applied_gauges:
        previous = result["maximum_frontier_imbalance_after"]
        current = max(update.imbalance_after for update in applied_gauges)
        result["maximum_frontier_imbalance_after"] = (
            current if previous is None else max(previous, current)
        )
    if state.history:
        result["final_delta_energy"] = float(state.history[-1]["delta_energy"])


def run_benchmark(
    *,
    j2=0.5,
    bond_dim=4,
    reference_bond_dim=32,
    mps_passes=20,
    reference_passes=20,
    tolerance=1.0e-9,
    convergence_tolerance_per_site=1.0e-6,
    consecutive_cycles=2,
    max_cycles=100,
    passes_per_cycle=2,
    tie_noise=1.0e-3,
    seed=7,
    frontier_gauge_weighting="uniform",
    output=DEFAULT_OUTPUT,
    snapshot=None,
):
    nrows = ncols = 6
    nsites = nrows * ncols
    print(f"building 6x6 J1-J2={j2:g} Hamiltonian and J2 tie graph", flush=True)
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(j2)) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, diagonals)
    local_mpo = hamiltonian.to_mpo().compress()
    mpo = MPO(list(local_mpo.tensors))

    output = Path(output)
    snapshot = output.with_suffix(".npz") if snapshot is None else Path(snapshot)
    payload = {
        "status": "running",
        "model": {
            "nrows": nrows,
            "ncols": ncols,
            "nsites": nsites,
            "j1": 1.0,
            "j2": float(j2),
            "boundary": "open",
            "site_order": "row-wise snake",
            "tie_graph": "diagonal J2 bonds only",
            "tie_edges": len(diagonals),
            "j1_nearest_edges": len(nearest),
            "j2_diagonal_edges": len(diagonals),
            "full_hilbert_dimension": 2**nsites,
        },
        "settings": {
            "bond_dim": int(bond_dim),
            "reference_bond_dim": int(reference_bond_dim),
            "mps_directional_pass_limit": int(mps_passes),
            "reference_directional_pass_limit": int(reference_passes),
            "tolerance": float(tolerance),
            "convergence_tolerance_per_site": float(
                convergence_tolerance_per_site
            ),
            "convergence_consecutive_cycles": int(consecutive_cycles),
            "passes_per_cycle": int(passes_per_cycle),
            "max_cycles": int(max_cycles),
            "tie_noise": float(tie_noise),
            "seed": int(seed),
            "frontier_backend": "identity_block",
            "local_solver": "direct",
            "frontier_canonicalization": True,
            "frontier_gauge_weighting": frontier_gauge_weighting,
        },
        "results": {},
        "convergence": {},
    }
    _write_json(output, payload)

    print(f"6x6 J1-J2={j2:g}: optimizing MPS D={bond_dim}", flush=True)
    mps_state, mps_record = _optimize_mps(
        mpo,
        nsites=nsites,
        bond_dim=bond_dim,
        seed=seed,
        pass_limit=mps_passes,
        tolerance=tolerance,
    )
    mps_record["parameter_capacity"] = _mps_capacity(nsites, bond_dim)
    payload["results"][f"mps_d{bond_dim}"] = mps_record
    _write_json(output, payload)

    reference_record = None
    reference_state = mps_state
    previous_bond_dim = int(bond_dim)
    while previous_bond_dim < reference_bond_dim:
        current_bond_dim = min(reference_bond_dim, 2 * previous_bond_dim)
        print(f"optimizing MPS reference D={current_bond_dim}", flush=True)
        reference_state, reference_record = _optimize_mps(
            mpo,
            nsites=nsites,
            bond_dim=current_bond_dim,
            seed=seed + current_bond_dim,
            pass_limit=reference_passes,
            tolerance=tolerance,
            initial_state=reference_state,
            initialization=f"mps_d{previous_bond_dim}",
        )
        reference_record["parameter_capacity"] = _mps_capacity(
            nsites, current_bond_dim
        )
        payload["results"][f"mps_d{current_bond_dim}"] = reference_record
        _write_json(output, payload)
        previous_bond_dim = current_bond_dim

    print("lifting MPS into J2-only graph-tied LETTA", flush=True)
    setup_start = perf_counter()
    state = frontier_tied_letta_from_mps(
        hamiltonian,
        parent_sets,
        _ordered_mps_factors(mps_state),
        bond_dim=bond_dim,
        tie_noise=tie_noise,
        seed=seed + 2,
        frontier_backend="identity_block",
    )
    setup_seconds = perf_counter() - setup_start
    initial_energy = float(state.expectation())
    best_reference_energy = min(
        float(row["energy"])
        for name, row in payload["results"].items()
        if name.startswith("mps_d")
    )
    result = {
        "bond_dim": int(bond_dim),
        "parameters": int(state.nparameters),
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": 0.0,
        "initial_energy": initial_energy,
        "energy": initial_energy,
        "fresh_energy": initial_energy,
        "energy_per_site": initial_energy / nsites,
        "improvement_from_same_d_mps": (
            initial_energy - float(mps_record["energy"])
        ),
        "energy_above_mps_reference_per_site": (
            (initial_energy - float(reference_record["energy"])) / nsites
            if reference_record is not None
            else None
        ),
        "energy_above_best_mps_reference_per_site": (
            initial_energy - best_reference_energy
        )
        / nsites,
        "directional_passes_completed": 0,
        "directional_pass_energies": [],
        "directional_pass_diagnostics": [],
        "solver_failures": 0,
        "accepted_updates": 0,
        "site_updates": 0,
        "frontier_gauge": True,
        "frontier_gauge_weighting": frontier_gauge_weighting,
        "frontier_gauge_bond_attempts": 0,
        "applied_frontier_gauges": 0,
        "maximum_frontier_imbalance_after": None,
        "final_delta_energy": None,
        "converged": False,
        "convergence_tolerance_per_site": float(convergence_tolerance_per_site),
        "convergence_consecutive_cycles": 0,
        "peak_frontier_elements": int(state.peak_frontier_elements),
        "peak_compressed_frontier_elements": int(
            state.peak_compressed_frontier_elements
        ),
        "cached_environment_elements": int(state.cached_environment_elements),
        "snapshot": str(snapshot),
    }
    payload["results"][f"j2_letta_d{bond_dim}"] = result
    _write_json(output, payload)

    cycle_records = []
    streak = 0
    total_optimization_seconds = 0.0
    for cycle in range(int(max_cycles)):
        energy_before = float(state.expectation())
        start = perf_counter()
        state.run(
            nsweeps=int(passes_per_cycle),
            tol=0.0,
            solver="direct",
            eig_tol=tolerance,
            metric_tol=1.0e-12,
            frontier_canonicalization=True,
            frontier_gauge_weighting=frontier_gauge_weighting,
            verbose=True,
        )
        seconds = perf_counter() - start
        total_optimization_seconds += seconds
        energy = float(state.expectation())
        gain_per_site = (energy_before - energy) / nsites
        failures = int(sum(row["solver_failures"] for row in state.history))
        passed = gain_per_site < float(convergence_tolerance_per_site) and failures == 0
        streak = streak + 1 if passed else 0
        _append_sweep_diagnostics(result, state)
        result["optimization_seconds"] = float(total_optimization_seconds)
        result["energy"] = energy
        result["fresh_energy"] = energy
        result["energy_per_site"] = energy / nsites
        result["improvement_from_same_d_mps"] = energy - float(mps_record["energy"])
        result["energy_above_mps_reference_per_site"] = (
            (energy - float(reference_record["energy"])) / nsites
            if reference_record is not None
            else None
        )
        result["energy_above_best_mps_reference_per_site"] = (
            energy - best_reference_energy
        ) / nsites
        result["convergence_consecutive_cycles"] = streak
        record = {
            "cycle": cycle + 1,
            "energy": energy,
            "energy_per_site": energy / nsites,
            "gain_per_site": gain_per_site,
            "solver_failures": failures,
            "criterion_passed": passed,
            "consecutive_passes": streak,
            "seconds": seconds,
        }
        cycle_records.append(record)
        payload["convergence"] = {
            "converged": streak >= int(consecutive_cycles),
            "criterion": "full-cycle energy gain per site",
            "tolerance_per_site": float(convergence_tolerance_per_site),
            "required_consecutive_cycles": int(consecutive_cycles),
            "cycles_completed_in_run": len(cycle_records),
            "final_consecutive_cycles": streak,
            "cycle_records": cycle_records,
        }
        result["converged"] = bool(payload["convergence"]["converged"])
        _save_snapshot(snapshot, state.tensors)
        _write_json(output, payload)
        print(
            f"cycle={cycle + 1:3d} E/N={energy / nsites:.12f} "
            f"cycle_gain/N={gain_per_site:.3e} failures={failures} "
            f"streak={streak}/{consecutive_cycles}",
            flush=True,
        )
        if result["converged"]:
            break

    payload["status"] = "complete" if result["converged"] else "incomplete"
    _write_json(output, payload)
    if not result["converged"]:
        raise RuntimeError(
            f"J2-only LETTA did not converge within {max_cycles} cycles; "
            f"final gain/site={cycle_records[-1]['gain_per_site']:.3e}."
        )
    print(f"wrote {output}", flush=True)
    print(f"wrote {snapshot}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--reference-bond-dim", type=int, default=32)
    parser.add_argument("--mps-passes", type=int, default=20)
    parser.add_argument("--reference-passes", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--convergence-tolerance-per-site", type=float, default=1.0e-6)
    parser.add_argument("--consecutive-cycles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--passes-per-cycle", type=int, default=2)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--frontier-gauge-weighting",
        choices=("uniform", "probability"),
        default="uniform",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    run_benchmark(
        j2=args.j2,
        bond_dim=args.bond_dim,
        reference_bond_dim=args.reference_bond_dim,
        mps_passes=args.mps_passes,
        reference_passes=args.reference_passes,
        tolerance=args.tolerance,
        convergence_tolerance_per_site=args.convergence_tolerance_per_site,
        consecutive_cycles=args.consecutive_cycles,
        max_cycles=args.max_cycles,
        passes_per_cycle=args.passes_per_cycle,
        tie_noise=args.tie_noise,
        seed=args.seed,
        frontier_gauge_weighting=args.frontier_gauge_weighting,
        output=args.output,
        snapshot=args.snapshot,
    )


if __name__ == "__main__":
    main()
