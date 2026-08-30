#!/usr/bin/env python3
"""Accelerate and certify the converged fixed-graph 6x6 two-site state.

Safeguarded block natural-gradient batches precondition the slow sweep tail.
Convergence is nevertheless certified only by two consecutive complete
two-site LR/RL cycles below the requested energy-gain threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.converge_frontier_letta_bond_expansion_6x6 import (
    DEFAULT_MPS_REFERENCE,
    _state_from_snapshot,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = RESULTS / "frontier_letta_two_site_converged_6x6.json"
DEFAULT_SOURCE_SNAPSHOT = RESULTS / "frontier_letta_two_site_converged_6x6.npz"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_two_site_hybrid_converged_6x6.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_two_site_hybrid_converged_6x6.npz"


def _fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _save_snapshot(path, state, *, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        tie_edges=np.asarray(
            [
                (site, parent)
                for site, parents in enumerate(state.parent_sets)
                for parent in parents
            ],
            dtype=np.int64,
        ),
        recorded_energy=np.asarray(float(payload["result"]["energy"])),
        natural_gradient_steps=np.asarray(
            len(payload["natural_gradient_steps"]), dtype=np.int64
        ),
        two_site_cycles=np.asarray(
            len(payload["two_site_cycles"]), dtype=np.int64
        ),
        protocol_fingerprint=np.asarray(payload["protocol_fingerprint"]),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    temporary.replace(path)


def _snapshot_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            "energy": float(archive["recorded_energy"]),
            "natural_gradient_steps": int(archive["natural_gradient_steps"]),
            "two_site_cycles": int(archive["two_site_cycles"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
        }


def _source_two_site_cycles(source):
    cycles = source.get("two_site_cycles", [])
    if cycles:
        return int(cycles[-1].get("absolute_cycle", cycles[-1]["cycle"]))
    result = source.get("result", {})
    return int(result.get("two_site_cycles", result.get("cycles_completed", 0)))


def _pair_diagnostics(updates):
    merged = [update.merged_solve for update in updates]
    unresolved = [
        int(update.sites[0])
        for update, diagnostics in zip(updates, merged)
        if diagnostics is None
        or not diagnostics.verified
        or not diagnostics.lowest_root_certified
    ]
    residuals = [
        float(item.metric_dual_relative_residual)
        for item in merged
        if item is not None and np.isfinite(item.metric_dual_relative_residual)
    ]
    discarded = [
        float(item.discarded_support_residual)
        for item in merged
        if item is not None and np.isfinite(item.discarded_support_residual)
    ]
    outer_counts = Counter(int(update.outer_cycles) for update in updates)
    return {
        "pair_updates": len(updates),
        "accepted_pair_updates": int(sum(update.accepted for update in updates)),
        "verified_merged_roots": int(
            sum(item is not None and item.verified for item in merged)
        ),
        "certified_lowest_roots": int(
            sum(item is not None and item.lowest_root_certified for item in merged)
        ),
        "unresolved_sites": unresolved,
        "maximum_metric_dual_relative_residual": float(
            max(residuals, default=0.0)
        ),
        "maximum_discarded_support_residual": float(
            max(discarded, default=0.0)
        ),
        "outer_cycle_histogram": {
            str(count): int(frequency)
            for count, frequency in sorted(outer_counts.items())
        },
    }


def _update_result(payload, state, *, reference_energy, snapshot, status):
    energy = float(state.expectation())
    source_energy = float(payload["source"]["energy"])
    payload["status"] = status
    payload["result"] = {
        "converged": status == "complete",
        "energy": energy,
        "energy_per_site": energy / 36,
        "energy_gain_from_source": source_energy - energy,
        "energy_gain_per_site_from_source": (source_energy - energy) / 36,
        "natural_gradient_steps": len(payload["natural_gradient_steps"]),
        "two_site_cycles": len(payload["two_site_cycles"]),
        "two_site_low_gain_streak": int(payload["state_machine"]["low_streak"]),
        "dense_mps_d32_energy": reference_energy,
        "energy_above_dense_mps_d32": energy - reference_energy,
        "energy_above_dense_mps_d32_per_site": (
            energy - reference_energy
        ) / 36,
        "parameters": int(state.nparameters),
        "bond_dims": list(state.bond_dims),
        "snapshot": str(Path(snapshot).resolve()),
    }


def converge(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    mps_reference=DEFAULT_MPS_REFERENCE,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    ng_steps_per_batch=40,
    ng_small_gain_per_site=1.0e-8,
    ng_required_small_steps=10,
    ng_damping=1.0e-6,
    maximum_ng_steps=400,
    maximum_two_site_cycles=20,
    stopping_gain_per_site=1.0e-7,
    required_two_site_cycles=2,
    outer_cycles=3,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    output = Path(output)
    snapshot = Path(snapshot)
    source = json.loads(source_result.read_text(encoding="utf-8"))
    model = source["model"]
    source_energy = float(source["result"]["energy"])
    reference = json.loads(Path(mps_reference).read_text(encoding="utf-8"))
    reference_energy = float(reference["results"]["mps_d32"]["energy"])
    protocol = {
        "source_result": str(source_result.resolve()),
        "source_snapshot": str(source_snapshot.resolve()),
        "graph": "fixed all-nearest-neighbour J1 ties",
        "ranks": "fixed",
        "symmetry": "none",
        "frontier_backend": "identity_block",
        "preconditioner": "safeguarded block-diagonal natural gradient",
        "ng_steps_per_batch": int(ng_steps_per_batch),
        "ng_small_gain_per_site": float(ng_small_gain_per_site),
        "ng_required_small_steps": int(ng_required_small_steps),
        "maximum_ng_steps": int(maximum_ng_steps),
        "ng_metric_tol": 1.0e-12,
        "ng_damping": float(ng_damping),
        "ng_trust_radius": 0.25,
        "ng_max_backtracks": 12,
        "certifier": "two consecutive complete two-site LR/RL cycles",
        "maximum_two_site_cycles": int(maximum_two_site_cycles),
        "stopping_gain_per_site": float(stopping_gain_per_site),
        "required_two_site_cycles": int(required_two_site_cycles),
        "pair_solver": "residual verified with dense certification",
        "pair_operator_backend": "dense",
        "factor_solver": "dense",
        "metric_support": "regularized",
        "metric_tol": 1.0e-12,
        "eig_tol": 1.0e-10,
        "outer_cycles": int(outer_cycles),
        "split_metric_sweeps": 6,
        "split_variational_sweeps": 8,
    }
    protocol_fingerprint = _fingerprint(protocol)
    resumed = bool(resume and output.is_file() and snapshot.is_file())
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _snapshot_metadata(snapshot)
        if payload["protocol_fingerprint"] != protocol_fingerprint:
            raise RuntimeError("result protocol does not match requested run")
        if metadata["protocol_fingerprint"] != protocol_fingerprint:
            raise RuntimeError("snapshot protocol does not match requested run")
        if metadata["natural_gradient_steps"] != len(
            payload["natural_gradient_steps"]
        ) or metadata["two_site_cycles"] != len(payload["two_site_cycles"]):
            raise RuntimeError("snapshot and JSON progress are inconsistent")
        state = _state_from_snapshot(model, snapshot)
        if abs(float(state.expectation()) - metadata["energy"]) > 2.0e-8:
            raise RuntimeError("snapshot energy is inconsistent")
        elapsed_before = float(payload["timing_seconds"]["total"])
        if payload["result"].get("converged", False):
            return payload
    else:
        state = _state_from_snapshot(model, source_snapshot)
        measured = float(state.expectation())
        if abs(measured - source_energy) > 2.0e-8:
            raise RuntimeError("source snapshot and result energies are inconsistent")
        elapsed_before = 0.0
        payload = {
            "status": "running",
            "model": model,
            "protocol": protocol,
            "protocol_fingerprint": protocol_fingerprint,
            "source": {
                "energy": measured,
                "energy_per_site": measured / 36,
                "source_two_site_cycles": int(
                    _source_two_site_cycles(source)
                ),
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
            },
            "natural_gradient_steps": [],
            "two_site_cycles": [],
            "state_machine": {
                "action": "natural_gradient",
                "batch_steps": 0,
                "small_gain_streak": 0,
                "low_streak": 0,
            },
            "result": {},
            "timing_seconds": {"total": 0.0},
        }
        _update_result(
            payload,
            state,
            reference_energy=reference_energy,
            snapshot=snapshot,
            status="running",
        )

    run_start = perf_counter()
    while True:
        machine = payload["state_machine"]
        if machine["action"] == "natural_gradient":
            if len(payload["natural_gradient_steps"]) >= int(maximum_ng_steps):
                machine["action"] = "two_site"
                machine["batch_steps"] = 0
                machine["small_gain_streak"] = 0
                continue
            start = perf_counter()
            update = state.natural_gradient_step(
                metric_tol=1.0e-12,
                damping=float(ng_damping),
                trust_radius=0.25,
                max_backtracks=12,
            )
            seconds = float(perf_counter() - start)
            gain = float(update.energy_before - update.energy)
            small = bool(
                update.accepted
                and gain >= 0.0
                and gain / 36 < float(ng_small_gain_per_site)
            )
            machine["small_gain_streak"] = (
                int(machine["small_gain_streak"]) + 1 if small else 0
            )
            machine["batch_steps"] = int(machine["batch_steps"]) + 1
            payload["natural_gradient_steps"].append(
                {
                    "step": len(payload["natural_gradient_steps"]) + 1,
                    "energy_before": float(update.energy_before),
                    "energy": float(update.energy),
                    "energy_gain": gain,
                    "energy_gain_per_site": gain / 36,
                    "seconds": seconds,
                    **asdict(update),
                }
            )
            if (
                not update.accepted
                or int(machine["batch_steps"]) >= int(ng_steps_per_batch)
                or int(machine["small_gain_streak"])
                >= int(ng_required_small_steps)
            ):
                machine["action"] = "two_site"
                machine["batch_steps"] = 0
                machine["small_gain_streak"] = 0
            print(
                f"NG {len(payload['natural_gradient_steps']):4d}  "
                f"energy={update.energy:.14f}  gain/site={gain/36:.3e}  "
                f"seconds={seconds:.1f}  accepted={update.accepted}",
                flush=True,
            )
        else:
            if len(payload["two_site_cycles"]) >= int(maximum_two_site_cycles):
                _update_result(
                    payload,
                    state,
                    reference_energy=reference_energy,
                    snapshot=snapshot,
                    status="capped",
                )
                break
            energy_before = float(state.expectation())
            cycle_start = perf_counter()
            rows = []
            absolute_cycle = int(payload["source"]["source_two_site_cycles"]) + len(
                payload["two_site_cycles"]
            )
            for direction in range(2):
                state.run_two_site(
                    nsweeps=1,
                    sweep_offset=2 * absolute_cycle + direction,
                    tol=0.0,
                    solver="verified",
                    verify_pair_energies=False,
                    verbose=True,
                    pair_operator_backend="dense",
                    factor_solver="dense",
                    outer_cycles=int(outer_cycles),
                    metric_support="regularized",
                    split_strategy="variational",
                    metric_tol=1.0e-12,
                    eig_tol=1.0e-10,
                    maxiter=1600,
                    max_subspace=96,
                    merged_dense_fallback_dim=2048,
                    split_metric_tol=1.0e-12,
                    split_metric_sweeps=6,
                    split_variational_sweeps=8,
                    split_random_starts=0,
                    split_energy_tol=1.0e-12,
                )
                rows.append(state.history[0])
            seconds = float(perf_counter() - cycle_start)
            energy = float(state.expectation())
            gain = energy_before - energy
            updates = tuple(update for row in rows for update in row["updates"])
            diagnostics = _pair_diagnostics(updates)
            verified = bool(
                all(row["accepted"] for row in rows)
                and not diagnostics["unresolved_sites"]
            )
            below = bool(
                verified
                and gain >= 0.0
                and gain / 36 < float(stopping_gain_per_site)
            )
            machine["low_streak"] = (
                int(machine["low_streak"]) + 1 if below else 0
            )
            payload["two_site_cycles"].append(
                {
                    "cycle": len(payload["two_site_cycles"]) + 1,
                    "absolute_cycle": absolute_cycle + 1,
                    "energy_before": energy_before,
                    "energy": energy,
                    "energy_gain": gain,
                    "energy_gain_per_site": gain / 36,
                    "seconds": seconds,
                    "endpoint_energies": [float(row["energy"]) for row in rows],
                    "endpoints_accepted": bool(all(row["accepted"] for row in rows)),
                    "low_gain_streak": int(machine["low_streak"]),
                    **diagnostics,
                }
            )
            if int(machine["low_streak"]) >= int(required_two_site_cycles):
                _update_result(
                    payload,
                    state,
                    reference_energy=reference_energy,
                    snapshot=snapshot,
                    status="complete",
                )
            elif below:
                machine["action"] = "two_site"
            else:
                machine["action"] = "natural_gradient"
            print(
                f"two-site certificate {len(payload['two_site_cycles']):3d}  "
                f"energy={energy:.14f}  gain/site={gain/36:.3e}  "
                f"seconds={seconds:.1f}  low-gain streak={machine['low_streak']}",
                flush=True,
            )

        if payload["status"] != "complete":
            _update_result(
                payload,
                state,
                reference_energy=reference_energy,
                snapshot=snapshot,
                status="running",
            )
        payload["timing_seconds"] = {
            "total": float(elapsed_before + perf_counter() - run_start)
        }
        _save_snapshot(snapshot, state, payload=payload)
        _write_json(output, payload)
        if payload["status"] == "complete":
            break

    payload["timing_seconds"] = {
        "total": float(elapsed_before + perf_counter() - run_start)
    }
    _save_snapshot(snapshot, state, payload=payload)
    _write_json(output, payload)
    result = payload["result"]
    print(
        f"final E={result['energy']:.14f}, converged={result['converged']}, "
        f"NG steps={result['natural_gradient_steps']}, "
        f"two-site cycles={result['two_site_cycles']}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument(
        "--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT
    )
    parser.add_argument("--mps-reference", type=Path, default=DEFAULT_MPS_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--ng-steps-per-batch", type=int, default=40)
    parser.add_argument("--ng-small-gain-per-site", type=float, default=1.0e-8)
    parser.add_argument("--ng-required-small-steps", type=int, default=10)
    parser.add_argument("--ng-damping", type=float, default=1.0e-6)
    parser.add_argument("--maximum-ng-steps", type=int, default=400)
    parser.add_argument("--maximum-two-site-cycles", type=int, default=20)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--required-two-site-cycles", type=int, default=2)
    parser.add_argument("--outer-cycles", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    converge(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        mps_reference=args.mps_reference,
        output=args.output,
        snapshot=args.snapshot,
        ng_steps_per_batch=args.ng_steps_per_batch,
        ng_small_gain_per_site=args.ng_small_gain_per_site,
        ng_required_small_steps=args.ng_required_small_steps,
        ng_damping=args.ng_damping,
        maximum_ng_steps=args.maximum_ng_steps,
        maximum_two_site_cycles=args.maximum_two_site_cycles,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_two_site_cycles=args.required_two_site_cycles,
        outer_cycles=args.outer_cycles,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
