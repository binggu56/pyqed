#!/usr/bin/env python3
"""Precondition and strictly certify the fixed-graph 6x6 LETTA state.

Fast safeguarded one-site sweeps, with a natural-gradient correction after
each right-to-left pass, relax the state between complete two-site LR/RL
cycles.  Only the two-site cycles contribute to the convergence certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from examples.mps.converge_frontier_letta_bond_expansion_6x6 import (
    DEFAULT_MPS_REFERENCE,
    _state_from_snapshot,
)
from examples.mps.converge_frontier_letta_two_site_hybrid_6x6 import (
    _fingerprint,
    _pair_diagnostics,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = (
    RESULTS / "frontier_letta_two_site_hybrid_converged_6x6.json"
)
DEFAULT_SOURCE_SNAPSHOT = (
    RESULTS / "frontier_letta_two_site_hybrid_converged_6x6.npz"
)
DEFAULT_OUTPUT = (
    RESULTS / "frontier_letta_two_site_whitened_converged_6x6.json"
)
DEFAULT_SNAPSHOT = (
    RESULTS / "frontier_letta_two_site_whitened_converged_6x6.npz"
)


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
        one_site_passes=np.asarray(
            len(payload["one_site_passes"]), dtype=np.int64
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
            "one_site_passes": int(archive["one_site_passes"]),
            "two_site_cycles": int(archive["two_site_cycles"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
        }


def _source_two_site_cycles(source):
    cycles = source.get("two_site_cycles", [])
    if cycles:
        return int(cycles[-1].get("absolute_cycle", cycles[-1]["cycle"]))
    result = source["result"]
    return int(result.get("two_site_cycles", result.get("cycles_completed", 0)))


def _protocol_fingerprint(protocol):
    """Fingerprint numerical choices while allowing safe cap increases."""
    numerical_protocol = {
        key: value
        for key, value in protocol.items()
        if key not in {"maximum_one_site_passes", "maximum_two_site_cycles"}
    }
    return _fingerprint(numerical_protocol)


def _one_site_diagnostics(row):
    updates = row["updates"]
    failures = [update for update in updates if not update.solver_converged]
    natural = row["natural_gradient"]
    return {
        "directional_sweep": int(row["sweep"]),
        "direction": (
            "left_to_right" if int(row["sweep"]) % 2 == 0 else "right_to_left"
        ),
        "accepted_site_updates": int(sum(update.accepted for update in updates)),
        "site_updates": len(updates),
        "solver_failures": len(failures),
        "failure_sites": [int(update.site) for update in failures],
        "maximum_residual_norm": float(
            max((update.residual_norm for update in updates), default=0.0)
        ),
        "identity_metric_sites": int(
            sum(update.solver_metric_is_identity for update in updates)
        ),
        "maximum_identity_metric_error": float(
            max(
                (
                    update.solver_metric_identity_error
                    for update in updates
                    if update.solver_metric_is_identity
                ),
                default=0.0,
            )
        ),
        "natural_gradient": None if natural is None else asdict(natural),
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
        "one_site_passes": len(payload["one_site_passes"]),
        "two_site_cycles": len(payload["two_site_cycles"]),
        "two_site_low_gain_streak": int(payload["state_machine"]["low_streak"]),
        "dense_mps_d32_energy": reference_energy,
        "energy_above_dense_mps_d32": energy - reference_energy,
        "energy_above_dense_mps_d32_per_site": (energy - reference_energy) / 36,
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
    one_site_passes_per_batch=20,
    one_site_small_gain_per_site=1.0e-8,
    one_site_required_small_passes=2,
    maximum_one_site_passes=400,
    maximum_two_site_cycles=30,
    stopping_gain_per_site=1.0e-7,
    required_two_site_cycles=2,
    outer_cycles=3,
    resume=True,
):
    if int(one_site_passes_per_batch) <= 0:
        raise ValueError("one_site_passes_per_batch must be positive")
    if int(one_site_required_small_passes) <= 0:
        raise ValueError("one_site_required_small_passes must be positive")
    if int(required_two_site_cycles) <= 0:
        raise ValueError("required_two_site_cycles must be positive")
    if int(outer_cycles) <= 0:
        raise ValueError("outer_cycles must be positive")
    if int(maximum_one_site_passes) < 0 or int(maximum_two_site_cycles) < 0:
        raise ValueError("optimization caps must be nonnegative")
    if (
        not np.isfinite(one_site_small_gain_per_site)
        or float(one_site_small_gain_per_site) < 0.0
        or not np.isfinite(stopping_gain_per_site)
        or float(stopping_gain_per_site) < 0.0
    ):
        raise ValueError("energy-gain thresholds must be finite and nonnegative")
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
        "preconditioner": "whitened one-site sweep plus RL natural gradient",
        "one_site_solver": "whitened",
        "one_site_frontier_canonicalization": True,
        "one_site_frontier_gauge_weighting": "uniform",
        "one_site_passes_per_batch": int(one_site_passes_per_batch),
        "one_site_small_gain_per_site": float(one_site_small_gain_per_site),
        "one_site_required_small_passes": int(one_site_required_small_passes),
        "maximum_one_site_passes": int(maximum_one_site_passes),
        "one_site_metric_tol": 1.0e-12,
        "one_site_eig_tol": 1.0e-10,
        "natural_gradient_every": 2,
        "natural_gradient_damping": 1.0e-6,
        "natural_gradient_trust_radius": 0.25,
        "natural_gradient_max_backtracks": 12,
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
    protocol_fingerprint = _protocol_fingerprint(protocol)
    resumed = bool(resume and output.is_file() and snapshot.is_file())
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _snapshot_metadata(snapshot)
        if payload["protocol_fingerprint"] != protocol_fingerprint:
            raise RuntimeError("result protocol does not match requested run")
        if metadata["protocol_fingerprint"] != protocol_fingerprint:
            raise RuntimeError("snapshot protocol does not match requested run")
        if metadata["one_site_passes"] != len(payload["one_site_passes"]):
            raise RuntimeError("snapshot and JSON one-site progress disagree")
        if metadata["two_site_cycles"] != len(payload["two_site_cycles"]):
            raise RuntimeError("snapshot and JSON two-site progress disagree")
        if abs(float(payload["result"]["energy"]) - metadata["energy"]) > 1.0e-12:
            raise RuntimeError("snapshot and JSON energies disagree")
        payload["protocol"]["maximum_one_site_passes"] = int(
            maximum_one_site_passes
        )
        payload["protocol"]["maximum_two_site_cycles"] = int(
            maximum_two_site_cycles
        )
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
                "source_two_site_cycles": _source_two_site_cycles(source),
                "parameters": int(state.nparameters),
                "bond_dims": list(state.bond_dims),
            },
            "one_site_passes": [],
            "two_site_cycles": [],
            "state_machine": {
                "action": "one_site",
                "batch_passes": 0,
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
        if machine["action"] == "one_site":
            if len(payload["one_site_passes"]) >= int(maximum_one_site_passes):
                machine["action"] = "two_site"
                machine["batch_passes"] = 0
                machine["small_gain_streak"] = 0
                continue
            energy_before = float(state.expectation())
            directional_sweep = len(payload["one_site_passes"]) + 1
            start = perf_counter()
            state.run(
                nsweeps=1,
                sweep_offset=directional_sweep,
                tol=0.0,
                metric_tol=1.0e-12,
                solver="whitened",
                eig_tol=1.0e-10,
                frontier_canonicalization=True,
                frontier_gauge_weighting="uniform",
                natural_gradient_every=2,
                natural_gradient_damping=1.0e-6,
                natural_gradient_trust_radius=0.25,
                natural_gradient_max_backtracks=12,
                verbose=False,
            )
            seconds = float(perf_counter() - start)
            row = state.history[0]
            energy = float(state.expectation())
            gain = energy_before - energy
            diagnostics = _one_site_diagnostics(row)
            verified = bool(
                diagnostics["solver_failures"] == 0
                and np.isfinite(energy)
                and gain >= -1.0e-10
            )
            small = bool(
                verified
                and gain >= 0.0
                and gain / 36 < float(one_site_small_gain_per_site)
            )
            machine["small_gain_streak"] = (
                int(machine["small_gain_streak"]) + 1 if small else 0
            )
            machine["batch_passes"] = int(machine["batch_passes"]) + 1
            payload["one_site_passes"].append(
                {
                    "pass": len(payload["one_site_passes"]) + 1,
                    "energy_before": energy_before,
                    "energy": energy,
                    "energy_gain": gain,
                    "energy_gain_per_site": gain / 36,
                    "seconds": seconds,
                    "verified": verified,
                    **diagnostics,
                }
            )
            if (
                not verified
                or int(machine["batch_passes"]) >= int(one_site_passes_per_batch)
                or int(machine["small_gain_streak"])
                >= int(one_site_required_small_passes)
            ):
                machine["action"] = "two_site"
                machine["batch_passes"] = 0
                machine["small_gain_streak"] = 0
            natural = diagnostics["natural_gradient"]
            print(
                f"one-site pass {len(payload['one_site_passes']):4d}  "
                f"energy={energy:.14f}  gain/site={gain/36:.3e}  "
                f"seconds={seconds:.1f}  failures={diagnostics['solver_failures']}  "
                f"natural={None if natural is None else natural['accepted']}",
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
            elif len(payload["one_site_passes"]) < int(maximum_one_site_passes):
                machine["action"] = "one_site"
            else:
                machine["action"] = "two_site"
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
        f"one-site passes={result['one_site_passes']}, "
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
    parser.add_argument("--one-site-passes-per-batch", type=int, default=20)
    parser.add_argument("--one-site-small-gain-per-site", type=float, default=1.0e-8)
    parser.add_argument("--one-site-required-small-passes", type=int, default=2)
    parser.add_argument("--maximum-one-site-passes", type=int, default=400)
    parser.add_argument("--maximum-two-site-cycles", type=int, default=30)
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
        one_site_passes_per_batch=args.one_site_passes_per_batch,
        one_site_small_gain_per_site=args.one_site_small_gain_per_site,
        one_site_required_small_passes=args.one_site_required_small_passes,
        maximum_one_site_passes=args.maximum_one_site_passes,
        maximum_two_site_cycles=args.maximum_two_site_cycles,
        stopping_gain_per_site=args.stopping_gain_per_site,
        required_two_site_cycles=args.required_two_site_cycles,
        outer_cycles=args.outer_cycles,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
