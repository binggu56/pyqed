#!/usr/bin/env python3
"""Continue the saved 6x6 J2-only graph-tied LETTA benchmark."""

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
from examples.mps.benchmark_frontier_letta_j2_tied_6x6 import (
    DEFAULT_OUTPUT,
    _append_sweep_diagnostics,
    _save_snapshot,
    _write_json,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA


DEFAULT_SNAPSHOT = DEFAULT_OUTPUT.with_suffix(".npz")


def _load_snapshot(path, nsites):
    with np.load(path, allow_pickle=False) as archive:
        return [
            np.array(archive[f"tensor_{site:03d}"], copy=True)
            for site in range(nsites)
        ]


def continue_run(
    *,
    result_path=DEFAULT_OUTPUT,
    snapshot_path=DEFAULT_SNAPSHOT,
    max_cycles=100,
    natural_gradient_every=0,
    natural_gradient_damping=1.0e-6,
    natural_gradient_trust_radius=0.25,
):
    result_path = Path(result_path)
    snapshot_path = Path(snapshot_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    model = payload["model"]
    settings = payload["settings"]
    nsites = int(model["nsites"])
    nearest, diagonals = square_j1_j2_bonds(int(model["nrows"]), int(model["ncols"]))
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(model["j2"])) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, diagonals)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dim=int(settings["bond_dim"]),
        tensors=_load_snapshot(snapshot_path, nsites),
        frontier_backend=settings.get("frontier_backend", "identity_block"),
    )
    key = f"j2_letta_d{int(settings['bond_dim'])}"
    result = payload["results"][key]
    initial_energy = float(state.expectation())
    recorded_energy = float(result["fresh_energy"])
    if abs(initial_energy - recorded_energy) > 2.0e-8:
        raise RuntimeError(
            "snapshot energy does not match the latest recorded J2-LETTA energy."
        )

    convergence = payload.setdefault("convergence", {})
    cycle_records = list(convergence.get("cycle_records", ()))
    streak = int(convergence.get("final_consecutive_cycles", 0))
    required = int(settings["convergence_consecutive_cycles"])
    tolerance_per_site = float(settings["convergence_tolerance_per_site"])
    passes_per_cycle = int(settings["passes_per_cycle"])
    total_seconds = float(result.get("optimization_seconds", 0.0))
    start_cycle = len(cycle_records)

    for cycle_offset in range(int(max_cycles)):
        energy_before = float(state.expectation())
        start = perf_counter()
        state.run(
            nsweeps=passes_per_cycle,
            sweep_offset=int(result["directional_passes_completed"]),
            tol=0.0,
            solver="direct",
            eig_tol=float(settings["tolerance"]),
            metric_tol=1.0e-12,
            frontier_canonicalization=True,
            frontier_gauge_weighting=settings["frontier_gauge_weighting"],
            natural_gradient_every=int(natural_gradient_every),
            natural_gradient_damping=float(natural_gradient_damping),
            natural_gradient_trust_radius=float(natural_gradient_trust_radius),
            verbose=True,
        )
        seconds = perf_counter() - start
        total_seconds += seconds
        energy = float(state.expectation())
        gain_per_site = (energy_before - energy) / nsites
        failures = int(sum(row["solver_failures"] for row in state.history))
        passed = gain_per_site < tolerance_per_site and failures == 0
        streak = streak + 1 if passed else 0
        _append_sweep_diagnostics(result, state)
        result["optimization_seconds"] = float(total_seconds)
        result["energy"] = energy
        result["fresh_energy"] = energy
        result["energy_per_site"] = energy / nsites
        result["convergence_consecutive_cycles"] = streak
        mps_d4 = payload["results"].get("mps_d4")
        if mps_d4 is not None:
            result["improvement_from_same_d_mps"] = energy - float(mps_d4["energy"])
        mps_rows = [
            row
            for name, row in payload["results"].items()
            if name.startswith("mps_d")
        ]
        if mps_rows:
            best_reference = min(float(row["energy"]) for row in mps_rows)
            result["energy_above_best_mps_reference_per_site"] = (
                energy - best_reference
            ) / nsites
            reference_key = f"mps_d{int(settings['reference_bond_dim'])}"
            if reference_key in payload["results"]:
                result["energy_above_mps_reference_per_site"] = (
                    energy - float(payload["results"][reference_key]["energy"])
                ) / nsites
        record = {
            "cycle": start_cycle + cycle_offset + 1,
            "energy": energy,
            "energy_per_site": energy / nsites,
            "gain_per_site": gain_per_site,
            "solver_failures": failures,
            "natural_gradient_every": int(natural_gradient_every),
            "natural_gradient_damping": float(natural_gradient_damping),
            "natural_gradient_trust_radius": float(natural_gradient_trust_radius),
            "criterion_passed": passed,
            "consecutive_passes": streak,
            "seconds": seconds,
        }
        cycle_records.append(record)
        converged = streak >= required
        payload["convergence"] = {
            "converged": converged,
            "criterion": "full-cycle energy gain per site",
            "tolerance_per_site": tolerance_per_site,
            "required_consecutive_cycles": required,
            "cycles_completed_in_run": len(cycle_records),
            "final_consecutive_cycles": streak,
            "cycle_records": cycle_records,
        }
        result["converged"] = bool(converged)
        payload["status"] = "complete" if converged else "incomplete"
        _save_snapshot(snapshot_path, state.tensors)
        _write_json(result_path, payload)
        print(
            f"cycle={record['cycle']:3d} E/N={energy / nsites:.12f} "
            f"cycle_gain/N={gain_per_site:.3e} failures={failures} "
            f"streak={streak}/{required}",
            flush=True,
        )
        if converged:
            print(f"wrote {result_path}", flush=True)
            print(f"wrote {snapshot_path}", flush=True)
            return payload

    raise RuntimeError(
        f"J2-only LETTA did not converge within {max_cycles} resumed cycles; "
        f"final gain/site={cycle_records[-1]['gain_per_site']:.3e}."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--natural-gradient-every", type=int, default=0)
    parser.add_argument("--natural-gradient-damping", type=float, default=1.0e-6)
    parser.add_argument("--natural-gradient-trust-radius", type=float, default=0.25)
    args = parser.parse_args()
    continue_run(
        result_path=args.result,
        snapshot_path=args.snapshot,
        max_cycles=args.max_cycles,
        natural_gradient_every=args.natural_gradient_every,
        natural_gradient_damping=args.natural_gradient_damping,
        natural_gradient_trust_radius=args.natural_gradient_trust_radius,
    )


if __name__ == "__main__":
    main()
