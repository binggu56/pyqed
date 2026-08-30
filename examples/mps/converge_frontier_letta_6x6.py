#!/usr/bin/env python3
"""Checkpoint the 6x6 LETTA state after every full optimization cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.mps.continue_frontier_letta_block_sparse_6x6 import (
    DEFAULT_REFERENCE,
    DEFAULT_RESULT,
    DEFAULT_SNAPSHOT,
    _write_json,
    continue_run,
)


def converge(
    *,
    result_path=DEFAULT_RESULT,
    snapshot_path=DEFAULT_SNAPSHOT,
    reference_path=DEFAULT_REFERENCE,
    tolerance_per_site=1.0e-6,
    consecutive_cycles=2,
    max_cycles=100,
):
    result_path = Path(result_path)
    streak = 0
    cycle_records = []
    converged = False
    for cycle in range(int(max_cycles)):
        payload = continue_run(
            result_path=result_path,
            snapshot_path=snapshot_path,
            reference_path=reference_path,
            passes=2,
            solver="whitened",
            metric_tol=1.0e-12,
            gauge="frontier",
            gauge_weight="uniform",
            natural_gradient_every=0,
            natural_gradient_damping=1.0e-6,
            natural_gradient_trust_radius=0.25,
        )
        nsites = int(payload["model"]["nsites"])
        latest = payload["continuations"][-1]
        gain_per_site = float(latest["energy_lowering"]) / nsites
        failures = int(latest["solver_failures"])
        passed = gain_per_site < float(tolerance_per_site) and failures == 0
        streak = streak + 1 if passed else 0
        record = {
            "cycle": cycle + 1,
            "energy": float(latest["energy"]),
            "energy_per_site": float(latest["energy_per_site"]),
            "gain_per_site": gain_per_site,
            "solver_failures": failures,
            "criterion_passed": passed,
            "consecutive_passes": streak,
        }
        cycle_records.append(record)
        print(
            f"cycle={cycle + 1:3d} E/N={record['energy_per_site']:.12f} "
            f"cycle_gain/N={gain_per_site:.3e} failures={failures} "
            f"streak={streak}/{consecutive_cycles}",
            flush=True,
        )
        if streak >= int(consecutive_cycles):
            converged = True
            break

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    convergence = {
        "converged": converged,
        "criterion": "full-cycle energy gain per site",
        "tolerance_per_site": float(tolerance_per_site),
        "required_consecutive_cycles": int(consecutive_cycles),
        "cycles_completed_in_run": len(cycle_records),
        "final_consecutive_cycles": streak,
        "cycle_records": cycle_records,
    }
    payload["convergence"] = convergence
    base = payload["results"]["letta_d4"]
    base["converged"] = converged
    base["convergence_tolerance_per_site"] = float(tolerance_per_site)
    base["convergence_consecutive_cycles"] = streak
    if cycle_records:
        base["final_cycle_gain_per_site"] = cycle_records[-1]["gain_per_site"]
    _write_json(result_path, payload)
    if not converged:
        raise RuntimeError(
            f"LETTA did not converge within {max_cycles} full cycles; "
            f"final gain/site={cycle_records[-1]['gain_per_site']:.3e}."
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tolerance-per-site", type=float, default=1.0e-6)
    parser.add_argument("--consecutive-cycles", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=100)
    args = parser.parse_args()
    converge(
        result_path=args.result,
        snapshot_path=args.snapshot,
        reference_path=args.reference,
        tolerance_per_site=args.tolerance_per_site,
        consecutive_cycles=args.consecutive_cycles,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
