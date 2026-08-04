#!/usr/bin/env python3
"""Converge the fixed-graph 6x6 J1-J2 LETTA state with two-site sweeps.

The calculation starts from the strongest existing checkpoint with the same
nearest-neighbour tie graph and fixed virtual ranks.  It runs complete
left-to-right/right-to-left cycles, verifies every merged generalized root,
and atomically checkpoints after each completed cycle.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT_SOURCE_RESULT = (
    RESULTS / "frontier_letta_bond_expansion_6x6_continued_30passes.json"
)
DEFAULT_SOURCE_SNAPSHOT = (
    RESULTS / "frontier_letta_bond_expansion_6x6_continued_30passes.npz"
)
DEFAULT_OUTPUT = RESULTS / "frontier_letta_two_site_converged_6x6.json"
DEFAULT_SNAPSHOT = RESULTS / "frontier_letta_two_site_converged_6x6.npz"
CHECKPOINT_ENERGY_TOL = 5.0e-8


def _protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _protocol_without_run_cap(protocol):
    comparable = dict(protocol)
    comparable.pop("maximum_cycles", None)
    return comparable


def _save_checkpoint(
    path,
    state,
    *,
    energy,
    completed_cycles,
    low_gain_streak,
    protocol_fingerprint,
):
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
        recorded_energy=np.asarray(float(energy)),
        completed_cycles=np.asarray(int(completed_cycles), dtype=np.int64),
        low_gain_streak=np.asarray(int(low_gain_streak), dtype=np.int64),
        protocol_fingerprint=np.asarray(str(protocol_fingerprint)),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    temporary.replace(path)


def _checkpoint_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        tensor_keys = sorted(key for key in archive if key.startswith("tensor_"))
        if tensor_keys != [f"tensor_{site:03d}" for site in range(36)]:
            raise RuntimeError("checkpoint does not contain all 36 tensors")
        return {
            "energy": float(archive["recorded_energy"]),
            "completed_cycles": int(archive["completed_cycles"]),
            "low_gain_streak": int(archive["low_gain_streak"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
        }


def _pair_diagnostics(updates):
    merged = [update.merged_solve for update in updates]
    verified = [item for item in merged if item is not None and item.verified]
    certified = [
        item for item in verified if item.lowest_root_certified
    ]
    residuals = [
        float(item.metric_dual_relative_residual)
        for item in merged
        if item is not None
        and np.isfinite(item.metric_dual_relative_residual)
    ]
    discarded = [
        float(item.discarded_support_residual)
        for item in merged
        if item is not None
        and np.isfinite(item.discarded_support_residual)
    ]
    outer_counts = Counter(int(update.outer_cycles) for update in updates)
    return {
        "pair_updates": len(updates),
        "accepted_pair_updates": int(sum(update.accepted for update in updates)),
        "verified_merged_roots": len(verified),
        "certified_lowest_roots": len(certified),
        "dense_certifications": int(
            sum(bool(item is not None and item.dense_fallback) for item in merged)
        ),
        "uncertified_sites": [
            int(update.sites[0])
            for update, item in zip(updates, merged)
            if item is None or not item.verified or not item.lowest_root_certified
        ],
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


def _source_absolute_passes(source):
    source_info = source.get("source", {})
    return int(source_info.get("source_directional_passes", 0)) + int(
        source.get("result", {}).get("passes_completed", 0)
    )


def converge(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    mps_reference=DEFAULT_MPS_REFERENCE,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    maximum_cycles=20,
    stopping_gain_per_site=1.0e-7,
    required_consecutive_cycles=2,
    outer_cycles=3,
    metric_support="regularized",
    metric_tol=1.0e-12,
    split_metric_tol=None,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    output = Path(output)
    snapshot = Path(snapshot)
    metric_support = str(metric_support)
    metric_tol = float(metric_tol)
    split_metric_tol = (
        metric_tol if split_metric_tol is None else float(split_metric_tol)
    )
    if metric_support not in {"regularized", "numerical"}:
        raise ValueError("metric_support must be 'regularized' or 'numerical'")
    if not np.isfinite(metric_tol) or metric_tol < 0.0:
        raise ValueError("metric_tol must be finite and nonnegative")
    if not np.isfinite(split_metric_tol) or split_metric_tol <= 0.0:
        raise ValueError("split_metric_tol must be finite and positive")
    source = json.loads(source_result.read_text(encoding="utf-8"))
    model = source["model"]
    source_energy = float(source["result"]["energy"])
    reference = json.loads(Path(mps_reference).read_text(encoding="utf-8"))
    mps_d32_energy = float(reference["results"]["mps_d32"]["energy"])
    protocol = {
        "source_result": str(source_result.resolve()),
        "source_snapshot": str(source_snapshot.resolve()),
        "source_absolute_one_site_directional_passes": _source_absolute_passes(
            source
        ),
        "graph": "fixed all-nearest-neighbour J1 ties",
        "ranks": "fixed",
        "symmetry": "none",
        "frontier_backend": "identity_block",
        "sweep": "complete left-to-right/right-to-left cycles",
        "solver": "residual-verified merged solve with dense certification",
        "pair_operator_backend": "dense",
        "factor_solver": "dense",
        "metric_support": metric_support,
        "metric_tol": metric_tol,
        "eig_tol": 1.0e-10,
        "maxiter": 1600,
        "max_subspace": 96,
        "merged_dense_fallback_dim": 2048,
        "outer_cycles": int(outer_cycles),
        "split_metric_tol": split_metric_tol,
        "split_metric_sweeps": 6,
        "split_variational_sweeps": 8,
        "split_energy_tol": 1.0e-12,
        "verify_pair_energies": False,
        "endpoint_verification": True,
        "checkpoint_energy_tolerance": CHECKPOINT_ENERGY_TOL,
        "maximum_cycles": int(maximum_cycles),
        "stopping_gain_per_site": float(stopping_gain_per_site),
        "required_consecutive_cycles": int(required_consecutive_cycles),
        "convergence_requires": (
            "accepted LR and RL endpoints, every merged root verified and "
            "lowest-root certified, and consecutive full-cycle gains below "
            "the threshold"
        ),
        "safe_resume": True,
    }
    fingerprint = _protocol_fingerprint(protocol)

    resumed = bool(resume and output.is_file() and snapshot.is_file())
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _checkpoint_metadata(snapshot)
        stored_fingerprint = payload["protocol_fingerprint"]
        if metadata["protocol_fingerprint"] != stored_fingerprint:
            raise RuntimeError("snapshot and JSON protocols are inconsistent")
        if _protocol_without_run_cap(payload["protocol"]) != (
            _protocol_without_run_cap(protocol)
        ):
            raise RuntimeError("checkpoint protocol does not match requested run")
        cycles = list(payload["cycles"])
        if int(maximum_cycles) < len(cycles):
            raise ValueError("maximum_cycles is below the completed cycle count")
        if metadata["completed_cycles"] != len(cycles):
            raise RuntimeError("snapshot and JSON cycle counts are inconsistent")
        state = _state_from_snapshot(model, snapshot)
        measured = float(state.expectation())
        if abs(measured - metadata["energy"]) > CHECKPOINT_ENERGY_TOL:
            raise RuntimeError(
                "resumed snapshot energy is inconsistent: "
                f"measured={measured:.17g}, recorded={metadata['energy']:.17g}, "
                f"delta={measured - metadata['energy']:.3e}"
            )
        low_gain_streak = int(metadata["low_gain_streak"])
        initial_energy = float(payload["source"]["energy"])
        elapsed_before = float(payload["timing_seconds"]["total"])
        if payload.get("result", {}).get("converged", False):
            return payload
        # The cycle cap is operational rather than physical.  Allow a
        # checkpointed run to be extended while requiring every numerical and
        # convergence setting to remain identical.
        payload["protocol"] = protocol
        payload["protocol_fingerprint"] = fingerprint
    else:
        state = _state_from_snapshot(model, source_snapshot)
        measured = float(state.expectation())
        if abs(measured - source_energy) > CHECKPOINT_ENERGY_TOL:
            raise RuntimeError(
                "source snapshot and JSON energies are inconsistent: "
                f"measured={measured:.17g}, recorded={source_energy:.17g}, "
                f"delta={measured - source_energy:.3e}"
            )
        cycles = []
        low_gain_streak = 0
        initial_energy = measured
        elapsed_before = 0.0
        payload = {
            "status": "running",
            "model": model,
            "protocol": protocol,
            "protocol_fingerprint": fingerprint,
            "source": {
                "energy": initial_energy,
                "energy_per_site": initial_energy / 36,
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
            },
            "cycles": cycles,
            "result": {},
            "timing_seconds": {"total": 0.0},
        }

    run_start = perf_counter()
    stop_reason = "maximum full cycles reached"
    for cycle_index in range(len(cycles), int(maximum_cycles)):
        energy_before = float(state.expectation())
        cycle_start = perf_counter()
        rows = []
        for direction in range(2):
            state.run_two_site(
                nsweeps=1,
                sweep_offset=2 * cycle_index + direction,
                tol=0.0,
                solver="verified",
                verify_pair_energies=False,
                verbose=True,
                pair_operator_backend="dense",
                factor_solver="dense",
                outer_cycles=int(outer_cycles),
                metric_support=metric_support,
                split_strategy="variational",
                metric_tol=metric_tol,
                eig_tol=1.0e-10,
                maxiter=1600,
                max_subspace=96,
                merged_dense_fallback_dim=2048,
                split_metric_tol=split_metric_tol,
                split_metric_sweeps=6,
                split_variational_sweeps=8,
                split_random_starts=0,
                split_energy_tol=1.0e-12,
            )
            if len(state.history) != 1:
                raise RuntimeError(
                    "a directional pass did not produce one history row"
                )
            rows.append(state.history[0])
        cycle_seconds = float(perf_counter() - cycle_start)
        rows = tuple(rows)
        energy = float(state.expectation())
        gain = float(energy_before - energy)
        updates = tuple(update for row in rows for update in row["updates"])
        diagnostics = _pair_diagnostics(updates)
        endpoints_accepted = bool(all(row["accepted"] for row in rows))
        cycle_verified = bool(
            endpoints_accepted and not diagnostics["uncertified_sites"]
        )
        below_threshold = bool(
            gain >= 0.0
            and gain / 36 < float(stopping_gain_per_site)
            and cycle_verified
        )
        low_gain_streak = low_gain_streak + 1 if below_threshold else 0
        cycles.append(
            {
                "cycle": cycle_index + 1,
                "directional_sweeps": [int(row["sweep"]) for row in rows],
                "energy_before": energy_before,
                "energy": energy,
                "energy_per_site": energy / 36,
                "energy_gain": gain,
                "energy_gain_per_site": gain / 36,
                "seconds": cycle_seconds,
                "endpoint_energies": [float(row["energy"]) for row in rows],
                "endpoint_gains": [
                    float(
                        (energy_before if index == 0 else rows[index - 1]["energy"])
                        - row["energy"]
                    )
                    for index, row in enumerate(rows)
                ],
                "endpoints_accepted": endpoints_accepted,
                "low_gain_streak": low_gain_streak,
                **diagnostics,
            }
        )
        converged = low_gain_streak >= int(required_consecutive_cycles)
        if converged:
            stop_reason = (
                f"{required_consecutive_cycles} consecutive verified full-cycle "
                f"gains/site below {stopping_gain_per_site:.3e}"
            )
        payload["cycles"] = cycles
        payload["status"] = (
            "complete"
            if converged
            else "capped" if len(cycles) >= int(maximum_cycles) else "running"
        )
        payload["result"] = {
            "converged": converged,
            "stop_reason": stop_reason,
            "cycles_completed": len(cycles),
            "directional_passes_completed": 2 * len(cycles),
            "energy": energy,
            "energy_per_site": energy / 36,
            "energy_gain_from_source": float(initial_energy - energy),
            "energy_gain_per_site_from_source": float(
                (initial_energy - energy) / 36
            ),
            "dense_mps_d32_energy": mps_d32_energy,
            "energy_above_dense_mps_d32": float(energy - mps_d32_energy),
            "energy_above_dense_mps_d32_per_site": float(
                (energy - mps_d32_energy) / 36
            ),
            "parameters": int(state.nparameters),
            "bond_dims": list(state.bond_dims),
            "snapshot": str(snapshot.resolve()),
        }
        payload["timing_seconds"] = {
            "total": float(elapsed_before + perf_counter() - run_start),
            "cycles": [float(record["seconds"]) for record in cycles],
        }
        _save_checkpoint(
            snapshot,
            state,
            energy=energy,
            completed_cycles=len(cycles),
            low_gain_streak=low_gain_streak,
            protocol_fingerprint=fingerprint,
        )
        _write_json(output, payload)
        print(
            f"cycle {cycle_index + 1:3d}  energy={energy:.14f}  "
            f"gain/site={gain/36:.3e}  seconds={cycle_seconds:.1f}  "
            f"low-gain streak={low_gain_streak}",
            flush=True,
        )
        if converged:
            break

    final = payload["result"]
    print(
        f"final E={final['energy']:.14f}, cycles={final['cycles_completed']}, "
        f"converged={final['converged']}; {final['stop_reason']}",
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
    parser.add_argument("--maximum-cycles", type=int, default=20)
    parser.add_argument("--stopping-gain-per-site", type=float, default=1.0e-7)
    parser.add_argument("--required-consecutive-cycles", type=int, default=2)
    parser.add_argument("--outer-cycles", type=int, default=3)
    parser.add_argument(
        "--metric-support",
        choices=("regularized", "numerical"),
        default="regularized",
    )
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--split-metric-tol", type=float)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    converge(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        mps_reference=args.mps_reference,
        output=args.output,
        snapshot=args.snapshot,
        maximum_cycles=args.maximum_cycles,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_consecutive_cycles=args.required_consecutive_cycles,
        outer_cycles=args.outer_cycles,
        metric_support=args.metric_support,
        metric_tol=args.metric_tol,
        split_metric_tol=args.split_metric_tol,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
