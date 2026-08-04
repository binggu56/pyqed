#!/usr/bin/env python3
"""Run one resumable exact-U(1) two-site LETTA directional sweep on 6x6.

The variational state is ``P_{2Sz=0}|Psi(A)>`` and every coordinate of the
unrestricted LETTA tensors ``A`` remains active.  The fixed projector and the
Hamiltonian-projector product are contracted exactly.  Adjacent pairs use an
exact memory-gated local backend: resident dense pair pencils and dense factor
equations when they fit the configured budget, otherwise conditional blocks
and matrix-free factors.  This never constructs a dense many-body projector or
restricts a LETTA tensor.

The dense Hamiltonian assembly uses ordered Python workers for large pairs;
set BLAS/OpenMP thread counts to one to avoid nested thread oversubscription.

The JSON/NPZ checkpoint is replaced atomically after every pair.  Both the
current tensors and the tensors at the start of the directional sweep are
stored, so the exact endpoint check can still reject and roll back the whole
sweep after an interrupted/resumed run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.converge_frontier_letta_bond_expansion_6x6 import (
    _state_from_snapshot,
)
from pyqed.letta import SectorProjectedLETTA


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = RESULTS / "frontier_letta_sector_projected_u1_6x6_j2_0p5.json"
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_SOURCE_RESULT.with_suffix(".npz")
DEFAULT_OUTPUT = (
    RESULTS
    / "frontier_letta_sector_projected_u1_two_site_fast_frontier_6x6_j2_0p5.json"
)
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT.with_suffix(".npz")
NSITES = 36
NPAIRS = NSITES - 1
EXPECTED_PARAMETERS = 4008
CHECKPOINT_ENERGY_TOL = 3.0e-7


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _tensor_digest(tensors):
    digest = sha256()
    for tensor in tensors:
        array = np.ascontiguousarray(tensor)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _records_digest(records):
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _protocol_fingerprint(protocol):
    stable = dict(protocol)
    # Increasing this operational cap is the intended resume workflow.
    stable.pop("maximum_pairs", None)
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _checkpoint_id(protocol_fingerprint, next_pair_ordinal, energy, records_digest):
    encoded = (
        f"{protocol_fingerprint}:{int(next_pair_ordinal)}:"
        f"{float(energy).hex()}:{records_digest}"
    ).encode()
    return sha256(encoded).hexdigest()


def _source_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        tensor_keys = sorted(key for key in archive.files if key.startswith("tensor_"))
        expected = [f"tensor_{site:03d}" for site in range(NSITES)]
        if tensor_keys != expected:
            raise RuntimeError("source checkpoint does not contain all 36 tensors.")
        required = {
            "virtual_bond_dims",
            "tie_edges",
            "recorded_energy",
            "target_two_sz",
            "local_two_sz",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"source checkpoint is missing metadata: {', '.join(missing)}"
            )
        return {
            "energy": float(archive["recorded_energy"]),
            "bond_dims": tuple(int(value) for value in archive["virtual_bond_dims"]),
            "tie_edges": np.array(archive["tie_edges"], dtype=np.int64, copy=True),
            "target_two_sz": int(archive["target_two_sz"]),
            "local_two_sz": np.array(archive["local_two_sz"], copy=True),
        }


def _save_snapshot_atomic(
    path,
    state,
    *,
    sweep_start_tensors,
    tie_edges,
    recorded_energy,
    sweep_start_energy,
    directional_sweep,
    next_pair_ordinal,
    endpoint_verified,
    endpoint_accepted,
    protocol_fingerprint,
    checkpoint_id,
    records_digest,
    source_checkpoint_id,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        tie_edges=np.asarray(tie_edges, dtype=np.int64),
        recorded_energy=np.asarray(float(recorded_energy)),
        sweep_start_energy=np.asarray(float(sweep_start_energy)),
        directional_sweep=np.asarray(int(directional_sweep), dtype=np.int64),
        next_pair_ordinal=np.asarray(int(next_pair_ordinal), dtype=np.int64),
        endpoint_verified=np.asarray(bool(endpoint_verified)),
        endpoint_accepted=np.asarray(
            -1 if endpoint_accepted is None else int(bool(endpoint_accepted)),
            dtype=np.int8,
        ),
        protocol_fingerprint=np.asarray(str(protocol_fingerprint)),
        checkpoint_id=np.asarray(str(checkpoint_id)),
        records_digest=np.asarray(str(records_digest)),
        current_tensor_digest=np.asarray(_tensor_digest(state.tensors)),
        sweep_start_tensor_digest=np.asarray(_tensor_digest(sweep_start_tensors)),
        source_checkpoint_id=np.asarray(str(source_checkpoint_id)),
        target_two_sz=np.asarray(0, dtype=np.int64),
        local_two_sz=np.asarray(((1, -1),) * NSITES, dtype=np.int64),
        parameter_count=np.asarray(int(state.nparameters), dtype=np.int64),
        dense_parameter_count=np.asarray(int(state.dense_nparameters), dtype=np.int64),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
        **{
            f"sweep_start_tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(sweep_start_tensors)
        },
    )
    temporary.replace(path)


def _read_snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        expected_current = [f"tensor_{site:03d}" for site in range(NSITES)]
        expected_start = [f"sweep_start_tensor_{site:03d}" for site in range(NSITES)]
        if any(
            key not in archive.files for key in (*expected_current, *expected_start)
        ):
            raise RuntimeError(
                "resume checkpoint must contain current and sweep-start tensors."
            )
        required = {
            "virtual_bond_dims",
            "tie_edges",
            "recorded_energy",
            "sweep_start_energy",
            "directional_sweep",
            "next_pair_ordinal",
            "endpoint_verified",
            "endpoint_accepted",
            "protocol_fingerprint",
            "checkpoint_id",
            "records_digest",
            "current_tensor_digest",
            "sweep_start_tensor_digest",
            "source_checkpoint_id",
            "target_two_sz",
            "local_two_sz",
            "parameter_count",
            "dense_parameter_count",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"resume checkpoint is missing metadata: {', '.join(missing)}"
            )
        current_tensors = [
            np.array(archive[key], copy=True) for key in expected_current
        ]
        sweep_start_tensors = [
            np.array(archive[key], copy=True) for key in expected_start
        ]
        return {
            "bond_dims": tuple(int(value) for value in archive["virtual_bond_dims"]),
            "tie_edges": np.array(archive["tie_edges"], dtype=np.int64, copy=True),
            "recorded_energy": float(archive["recorded_energy"]),
            "sweep_start_energy": float(archive["sweep_start_energy"]),
            "directional_sweep": int(archive["directional_sweep"]),
            "next_pair_ordinal": int(archive["next_pair_ordinal"]),
            "endpoint_verified": bool(archive["endpoint_verified"]),
            "endpoint_accepted": int(archive["endpoint_accepted"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
            "checkpoint_id": str(archive["checkpoint_id"]),
            "records_digest": str(archive["records_digest"]),
            "current_tensor_digest": str(archive["current_tensor_digest"]),
            "sweep_start_tensor_digest": str(archive["sweep_start_tensor_digest"]),
            "source_checkpoint_id": str(archive["source_checkpoint_id"]),
            "target_two_sz": int(archive["target_two_sz"]),
            "local_two_sz": np.array(archive["local_two_sz"], copy=True),
            "parameter_count": int(archive["parameter_count"]),
            "dense_parameter_count": int(archive["dense_parameter_count"]),
            "current_tensors": current_tensors,
            "sweep_start_tensors": sweep_start_tensors,
        }


def _projected_state(model, snapshot):
    unrestricted = _state_from_snapshot(model, snapshot)
    local_charges = tuple(((1,), (-1,)) for _ in range(NSITES))
    return SectorProjectedLETTA.from_unrestricted(
        unrestricted,
        local_charges=local_charges,
        target=(0,),
        frontier_backend="identity_block",
        _balance_initial_gauges=False,
    )


def _assert_full_parameterization(state, tensor_shapes, bond_dims):
    if state.nparameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAMETERS} LETTA parameters, got {state.nparameters}."
        )
    if state.dense_nparameters != state.nparameters:
        raise RuntimeError("the projected state reduced the tensor parameterization.")
    if hasattr(state, "local_masks"):
        raise RuntimeError("the projected state unexpectedly contains local masks.")
    if any(active != dense for active, dense in state.local_support_sizes()):
        raise RuntimeError("not every unrestricted tensor coordinate is active.")
    if tuple(tensor.shape for tensor in state.tensors) != tuple(tensor_shapes):
        raise RuntimeError("a two-site update changed a LETTA tensor shape.")
    if tuple(state.bond_dims) != tuple(bond_dims):
        raise RuntimeError("a two-site update changed a fixed virtual bond.")


def _merged_solve_record(diagnostics):
    if diagnostics is None:
        return None
    return {
        "method": str(diagnostics.method),
        "attempts": list(diagnostics.attempts),
        "verified": bool(diagnostics.verified),
        "lowest_root_certified": bool(diagnostics.lowest_root_certified),
        "fallback_reason": str(diagnostics.fallback_reason),
        "dense_fallback": bool(diagnostics.dense_fallback),
        "metric_support": str(diagnostics.metric_support),
        "metric_requested_rank": int(diagnostics.metric_requested_rank),
        "metric_numerical_rank": int(diagnostics.metric_numerical_rank),
        "metric_rank_complete": bool(diagnostics.metric_rank_complete),
        "metric_condition": _finite_or_none(diagnostics.metric_condition),
        "backward_residual": _finite_or_none(diagnostics.backward_residual),
        "metric_dual_relative_residual": _finite_or_none(
            diagnostics.metric_dual_relative_residual
        ),
        "null_residual": _finite_or_none(diagnostics.null_residual),
        "discarded_support_residual": _finite_or_none(
            diagnostics.discarded_support_residual
        ),
        "hamiltonian_action_calls": int(diagnostics.hamiltonian_action_calls),
        "hamiltonian_vector_products": int(diagnostics.hamiltonian_vector_products),
        "hamiltonian_batch_calls": int(diagnostics.hamiltonian_batch_calls),
    }


def _pair_record(update, *, ordinal, site, seconds):
    diagnostics = update.merged_solve
    dense_elements = 2 * int(update.raw_merged_dim) ** 2
    intentional_dense = update.pair_operator_backend == "dense"
    unexpected_dense_fallback = bool(
        diagnostics is not None and diagnostics.dense_fallback and not intentional_dense
    )
    return {
        "ordinal": int(ordinal),
        "sites": [int(site), int(site + 1)],
        "seconds": float(seconds),
        "energy_before": float(update.energy_before),
        "merged_energy": float(update.merged_energy),
        "attempted_energy": float(update.attempted_energy),
        "energy": float(update.energy),
        "accepted": bool(update.accepted),
        "raw_merged_dim": int(update.raw_merged_dim),
        "old_bond_dimension": int(update.old_bond_dimension),
        "temporary_bond_dimension": int(update.temporary_bond_dimension),
        "conditional_ranks": [int(value) for value in update.conditional_ranks],
        "relative_truncation_error": _finite_or_none(update.relative_truncation_error),
        "metric_projection_error": _finite_or_none(update.metric_projection_error),
        "outer_cycles": int(update.outer_cycles),
        "factor_sweeps": int(update.factor_sweeps),
        "factor_accepted_updates": int(update.factor_accepted_updates),
        "pair_operator_backend": str(update.pair_operator_backend),
        "pair_operator_requested_backend": str(update.pair_operator_requested_backend),
        "pair_operator_selection_reason": str(update.pair_operator_selection_reason),
        "factor_solver": str(update.factor_solver),
        "pair_operator_workers": int(update.pair_operator_workers),
        "pair_operator_stored_elements": int(update.pair_operator_stored_elements),
        "pair_operator_stored_bytes": int(update.pair_operator_stored_bytes),
        "operator_assembly_seconds": float(update.operator_assembly_seconds),
        "merged_solve_seconds": float(update.merged_solve_seconds),
        "split_seconds": float(update.split_seconds),
        "intentional_dense_local_pair": bool(intentional_dense),
        "dense_local_pair_elements": int(dense_elements if intentional_dense else 0),
        "full_dense_pair_elements_avoided": int(
            0 if intentional_dense else dense_elements
        ),
        "full_dense_pair_fallback": unexpected_dense_fallback,
        "merged_solve": _merged_solve_record(diagnostics),
    }


def _result_record(
    state,
    *,
    source_energy,
    sweep_start_energy,
    pairs,
    next_pair_ordinal,
    pair_sites,
    endpoint_verified,
    endpoint_accepted,
    snapshot,
):
    energy = float(state.energy)
    complete = bool(endpoint_verified)
    next_site = (
        None if next_pair_ordinal >= NPAIRS else int(pair_sites[next_pair_ordinal])
    )
    dense_fallbacks = int(sum(record["full_dense_pair_fallback"] for record in pairs))
    intentional_dense_pairs = int(
        sum(record["intentional_dense_local_pair"] for record in pairs)
    )
    return {
        "complete": complete,
        "stop_reason": (
            "completed exact endpoint verification"
            if complete
            else "maximum completed-pair cap reached"
        ),
        "directional_sweeps_completed": int(complete),
        "pairs_completed": int(next_pair_ordinal),
        "pairs_total": NPAIRS,
        "next_pair_site": next_site,
        "endpoint_verified": complete,
        "endpoint_accepted": (
            None if endpoint_accepted is None else bool(endpoint_accepted)
        ),
        "energy": energy,
        "energy_per_site": energy / NSITES,
        "energy_gain_from_source": float(source_energy - energy),
        "energy_gain_from_sweep_start": float(sweep_start_energy - energy),
        "parameters": int(state.nparameters),
        "dense_parameters": int(state.dense_nparameters),
        "all_tensor_coordinates_variational": True,
        "local_tensor_masks": False,
        "serialized_tensors": len(state.tensors),
        "bond_dims": list(state.bond_dims),
        "pair_operator_backends": sorted(
            {record["pair_operator_backend"] for record in pairs}
        ),
        "intentional_dense_local_pairs": intentional_dense_pairs,
        "full_dense_pair_fallbacks": dense_fallbacks,
        "no_full_dense_pair_fallback": dense_fallbacks == 0,
        "maximum_pair_operator_stored_elements": max(
            (record["pair_operator_stored_elements"] for record in pairs),
            default=0,
        ),
        "maximum_full_dense_pair_elements_avoided": max(
            (record["full_dense_pair_elements_avoided"] for record in pairs),
            default=0,
        ),
        "snapshot": str(Path(snapshot).resolve()),
    }


def _checkpoint(
    *,
    output,
    snapshot,
    payload,
    state,
    sweep_start_tensors,
    tie_edges,
    source_energy,
    sweep_start_energy,
    directional_sweep,
    next_pair_ordinal,
    pair_sites,
    endpoint_verified,
    endpoint_accepted,
    source_checkpoint_id,
):
    records_digest = _records_digest(payload["pairs"])
    checkpoint_id = _checkpoint_id(
        payload["protocol_fingerprint"],
        next_pair_ordinal,
        state.energy,
        records_digest,
    )
    payload["result"] = _result_record(
        state,
        source_energy=source_energy,
        sweep_start_energy=sweep_start_energy,
        pairs=payload["pairs"],
        next_pair_ordinal=next_pair_ordinal,
        pair_sites=pair_sites,
        endpoint_verified=endpoint_verified,
        endpoint_accepted=endpoint_accepted,
        snapshot=snapshot,
    )
    payload["result"]["checkpoint_id"] = checkpoint_id
    payload["records_digest"] = records_digest
    _save_snapshot_atomic(
        snapshot,
        state,
        sweep_start_tensors=sweep_start_tensors,
        tie_edges=tie_edges,
        recorded_energy=state.energy,
        sweep_start_energy=sweep_start_energy,
        directional_sweep=directional_sweep,
        next_pair_ordinal=next_pair_ordinal,
        endpoint_verified=endpoint_verified,
        endpoint_accepted=endpoint_accepted,
        protocol_fingerprint=payload["protocol_fingerprint"],
        checkpoint_id=checkpoint_id,
        records_digest=records_digest,
        source_checkpoint_id=source_checkpoint_id,
    )
    _write_json_atomic(output, payload)


def _validate_resume(
    payload,
    metadata,
    *,
    protocol,
    protocol_fingerprint,
    maximum_pairs,
    source_checkpoint_id,
):
    next_pair_ordinal = int(metadata["next_pair_ordinal"])
    checks = (
        (
            payload.get("protocol_fingerprint"),
            protocol_fingerprint,
            "JSON protocol fingerprint",
        ),
        (
            metadata["protocol_fingerprint"],
            protocol_fingerprint,
            "snapshot protocol fingerprint",
        ),
        (
            payload.get("result", {}).get("checkpoint_id"),
            metadata["checkpoint_id"],
            "checkpoint ID",
        ),
        (
            payload.get("records_digest"),
            metadata["records_digest"],
            "records digest",
        ),
        (
            _records_digest(payload.get("pairs", [])),
            metadata["records_digest"],
            "pair-record contents",
        ),
        (
            len(payload.get("pairs", [])),
            next_pair_ordinal,
            "completed pair count",
        ),
        (
            metadata["source_checkpoint_id"],
            source_checkpoint_id,
            "source checkpoint ID",
        ),
        (metadata["target_two_sz"], 0, "target 2Sz"),
        (
            metadata["parameter_count"],
            EXPECTED_PARAMETERS,
            "parameter count",
        ),
        (
            metadata["dense_parameter_count"],
            EXPECTED_PARAMETERS,
            "dense parameter count",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RuntimeError(
                f"unsafe resume: {label} mismatch ({actual!r} != {expected!r})."
            )
    stable_payload = dict(payload["protocol"])
    stable_payload.pop("maximum_pairs", None)
    stable_requested = dict(protocol)
    stable_requested.pop("maximum_pairs", None)
    if stable_payload != stable_requested:
        raise RuntimeError("unsafe resume: numerical protocol changed.")
    if next_pair_ordinal > int(maximum_pairs):
        raise ValueError(
            f"maximum_pairs={maximum_pairs} is below the "
            f"{next_pair_ordinal} already completed pairs."
        )
    if metadata["directional_sweep"] != int(protocol["directional_sweep"]):
        raise RuntimeError("unsafe resume: directional sweep changed.")
    if metadata["endpoint_verified"] and maximum_pairs != NPAIRS:
        raise ValueError("a completed sweep already contains all 35 pairs.")
    expected_local = np.asarray(((1, -1),) * NSITES)
    if metadata["local_two_sz"].shape != expected_local.shape or not np.array_equal(
        metadata["local_two_sz"], expected_local
    ):
        raise RuntimeError("unsafe resume: local projected charges changed.")
    if (
        _tensor_digest(metadata["current_tensors"]) != metadata["current_tensor_digest"]
        or _tensor_digest(metadata["sweep_start_tensors"])
        != metadata["sweep_start_tensor_digest"]
    ):
        raise RuntimeError("unsafe resume: tensor digest mismatch.")
    if (
        abs(metadata["recorded_energy"] - float(payload["result"]["energy"]))
        > CHECKPOINT_ENERGY_TOL
    ):
        raise RuntimeError("unsafe resume: JSON and snapshot energies differ.")


def _pair_messages(
    state,
    *,
    directional_sweep,
    next_pair_ordinal,
    hamiltonian_workers=1,
    hamiltonian_executor=None,
):
    if directional_sweep % 2 == 0:
        fixed_norm = state._norm_frontier.build_right(state.tensors)
        fixed_hamiltonian = state._hamiltonian_frontier.build_right(
            state.tensors,
            max_workers=hamiltonian_workers,
            executor=hamiltonian_executor,
        )
        moving_norm = state._norm_frontier.left_boundary()
        moving_hamiltonian = state._hamiltonian_frontier.left_boundary()
        for site in range(next_pair_ordinal):
            moving_norm = state._norm_frontier.advance_left(
                moving_norm, state.tensors, site
            )
            moving_hamiltonian = state._hamiltonian_frontier.advance_left(
                moving_hamiltonian,
                state.tensors,
                site,
                max_workers=hamiltonian_workers,
                executor=hamiltonian_executor,
            )
    else:
        site = NPAIRS - 1 - next_pair_ordinal
        fixed_norm = state._norm_frontier.build_left(state.tensors)
        fixed_hamiltonian = state._hamiltonian_frontier.build_left(
            state.tensors,
            max_workers=hamiltonian_workers,
            executor=hamiltonian_executor,
        )
        moving_norm = state._norm_frontier.right_boundary()
        moving_hamiltonian = state._hamiltonian_frontier.right_boundary()
        for following in range(NSITES - 1, site + 1, -1):
            moving_norm = state._norm_frontier.advance_right(
                moving_norm, state.tensors, following
            )
            moving_hamiltonian = state._hamiltonian_frontier.advance_right(
                moving_hamiltonian,
                state.tensors,
                following,
                max_workers=hamiltonian_workers,
                executor=hamiltonian_executor,
            )
    return fixed_norm, fixed_hamiltonian, moving_norm, moving_hamiltonian


def _endpoint_energy(
    state,
    *,
    directional_sweep,
    moving_norm,
    moving_hamiltonian,
):
    if directional_sweep % 2 == 0:
        moving_norm = state._norm_frontier.advance_left(
            moving_norm, state.tensors, NSITES - 1
        )
        moving_hamiltonian = state._hamiltonian_frontier.advance_left(
            moving_hamiltonian, state.tensors, NSITES - 1
        )
        boundary_cut = NSITES
    else:
        moving_norm = state._norm_frontier.advance_right(moving_norm, state.tensors, 0)
        moving_hamiltonian = state._hamiltonian_frontier.advance_right(
            moving_hamiltonian, state.tensors, 0
        )
        boundary_cut = 0
    norm = float(
        np.real(
            state._completed_frontier_scalar(
                state._norm_frontier,
                moving_norm,
                boundary_cut,
            )
        )
    )
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("projected LETTA state is numerically zero.")
    numerator = state._completed_frontier_scalar(
        state._hamiltonian_frontier,
        moving_hamiltonian,
        boundary_cut,
    )
    return float(np.real(numerator / norm)), norm


def run(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    maximum_pairs=NPAIRS,
    directional_sweep=0,
    outer_cycles=1,
    metric_tol=1.0e-12,
    eig_tol=1.0e-10,
    maxiter=1600,
    max_subspace=96,
    block_dense_component_max_size=64,
    pair_operator_backend="auto",
    pair_dense_max_elements=2_000_000,
    pair_operator_workers=4,
    frontier_workers=2,
    factor_solver="auto",
    merged_dense_fallback_dim=2048,
    split_metric_tol=1.0e-12,
    split_metric_sweeps=2,
    split_variational_sweeps=2,
    checkpoint_interval=1,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    output = Path(output)
    snapshot = Path(snapshot)
    maximum_pairs = int(maximum_pairs)
    directional_sweep = int(directional_sweep)
    pair_operator_backend = str(pair_operator_backend).lower().replace("-", "_")
    factor_solver = str(factor_solver).lower().replace("-", "_")
    pair_dense_max_elements = int(pair_dense_max_elements)
    pair_operator_workers = int(pair_operator_workers)
    frontier_workers = int(frontier_workers)
    checkpoint_interval = int(checkpoint_interval)
    merged_dense_fallback_dim = int(merged_dense_fallback_dim)
    if maximum_pairs < 0 or maximum_pairs > NPAIRS:
        raise ValueError(f"maximum_pairs must lie in [0, {NPAIRS}].")
    if directional_sweep < 0:
        raise ValueError("directional_sweep must be nonnegative.")
    if pair_operator_backend not in {"auto", "dense", "block"}:
        raise ValueError("pair_operator_backend must be 'auto', 'dense', or 'block'.")
    if factor_solver not in {"auto", "dense", "matrix_free"}:
        raise ValueError("factor_solver must be 'auto', 'dense', or 'matrix_free'.")
    if pair_dense_max_elements < 1:
        raise ValueError("pair_dense_max_elements must be positive.")
    if pair_operator_workers < 1:
        raise ValueError("pair_operator_workers must be positive.")
    if frontier_workers < 1:
        raise ValueError("frontier_workers must be positive.")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    if merged_dense_fallback_dim < 1:
        raise ValueError("merged_dense_fallback_dim must be positive.")
    protected = {source_result.resolve(), source_snapshot.resolve()}
    if output.resolve() in protected or snapshot.resolve() in protected:
        raise ValueError("output checkpoints must not overwrite source artifacts.")

    source = json.loads(source_result.read_text(encoding="utf-8"))
    model = source["model"]
    if (
        int(model["nrows"]) != 6
        or int(model["ncols"]) != 6
        or int(model["nsites"]) != NSITES
    ):
        raise ValueError("the source checkpoint must be the 6x6 model.")
    source_checkpoint_id = str(source["result"]["checkpoint_id"])
    source_energy = float(source["result"]["energy"])
    protocol = {
        "ansatz": "P_Q |Psi(A)> with every unrestricted A coordinate retained",
        "symmetry": "exact variation-after-projection U(1)",
        "target_two_sz": 0,
        "local_two_sz": [1, -1],
        "local_tensor_masks": False,
        "expected_parameters": EXPECTED_PARAMETERS,
        "source_result": str(source_result.resolve()),
        "source_snapshot": str(source_snapshot.resolve()),
        "source_checkpoint_id": source_checkpoint_id,
        "frontier_backend": "identity_block",
        "objective_mpo": "sparse factorized H times P_Q",
        "materialize_objective_mpo": False,
        "contraction": "exact",
        "optimization": "one fixed-bond directional two-site sweep",
        "directional_sweep": directional_sweep,
        "direction": (
            "left_to_right" if directional_sweep % 2 == 0 else "right_to_left"
        ),
        "pair_operator_backend": pair_operator_backend,
        "pair_dense_max_elements": pair_dense_max_elements,
        "pair_operator_workers": pair_operator_workers,
        "frontier_workers": frontier_workers,
        "factor_solver": factor_solver,
        "dense_local_pair_is_many_body_projection": False,
        "merged_dense_fallback_dim": merged_dense_fallback_dim,
        "conditional_block_dense_component_max_size": int(
            block_dense_component_max_size
        ),
        "metric_support": "numerical",
        "metric_tol": float(metric_tol),
        "eig_tol": float(eig_tol),
        "maxiter": int(maxiter),
        "max_subspace": int(max_subspace),
        "outer_cycles": int(outer_cycles),
        "split_strategy": "variational",
        "split_metric_tol": float(split_metric_tol),
        "split_metric_sweeps": int(split_metric_sweeps),
        "split_variational_sweeps": int(split_variational_sweeps),
        "split_random_starts": 0,
        "split_energy_tol": 1.0e-12,
        "verify_pair_energies": False,
        "endpoint_verification": True,
        "checkpoint_granularity": (
            "every adjacent pair"
            if checkpoint_interval == 1
            else "periodic adjacent pairs plus directional endpoint"
        ),
        "checkpoint_interval_pairs": checkpoint_interval,
        "checkpoint_energy_tolerance": CHECKPOINT_ENERGY_TOL,
        "maximum_pairs": maximum_pairs,
        "safe_resume": True,
    }
    protocol_fingerprint = _protocol_fingerprint(protocol)
    pair_sites = tuple(
        range(NPAIRS) if directional_sweep % 2 == 0 else range(NPAIRS - 1, -1, -1)
    )
    invocation_start = perf_counter()

    output_exists = output.is_file()
    snapshot_exists = snapshot.is_file()
    if resume and output_exists != snapshot_exists:
        raise RuntimeError(
            "unsafe resume: JSON and NPZ checkpoints must both exist or both be absent."
        )
    resumed = bool(resume and output_exists and snapshot_exists)
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _read_snapshot(snapshot)
        _validate_resume(
            payload,
            metadata,
            protocol=protocol,
            protocol_fingerprint=protocol_fingerprint,
            maximum_pairs=maximum_pairs,
            source_checkpoint_id=source_checkpoint_id,
        )
        if metadata["endpoint_verified"]:
            print(
                "checkpoint already contains a completed two-site "
                f"directional sweep: E={payload['result']['energy']:.12f}",
                flush=True,
            )
            return payload
        state = _projected_state(model, snapshot)
        measured_energy = float(state.energy)
        if abs(measured_energy - metadata["recorded_energy"]) > CHECKPOINT_ENERGY_TOL:
            raise RuntimeError(
                "resume snapshot does not reproduce its projected energy."
            )
        sweep_start_tensors = metadata["sweep_start_tensors"]
        sweep_start_energy = float(metadata["sweep_start_energy"])
        tie_edges = metadata["tie_edges"]
        next_pair_ordinal = int(metadata["next_pair_ordinal"])
        payload["protocol"]["maximum_pairs"] = maximum_pairs
        elapsed_before = float(payload["timing_seconds"]["total"])
    else:
        source_metadata = _source_metadata(source_snapshot)
        if source_metadata["target_two_sz"] != 0:
            raise RuntimeError("source snapshot has the wrong target charge.")
        expected_local = np.asarray(((1, -1),) * NSITES)
        if not np.array_equal(source_metadata["local_two_sz"], expected_local):
            raise RuntimeError("source snapshot has different local charges.")
        state = _projected_state(model, source_snapshot)
        measured_energy = float(state.energy)
        if (
            abs(measured_energy - source_energy) > CHECKPOINT_ENERGY_TOL
            or abs(source_metadata["energy"] - source_energy) > CHECKPOINT_ENERGY_TOL
        ):
            raise RuntimeError(
                "source JSON, snapshot, and reconstructed projected energy "
                "are inconsistent."
            )
        sweep_start_tensors = [tensor.copy() for tensor in state.tensors]
        sweep_start_energy = measured_energy
        tie_edges = source_metadata["tie_edges"]
        next_pair_ordinal = 0
        elapsed_before = 0.0
        payload = {
            "status": "initialized",
            "model": model,
            "protocol": protocol,
            "protocol_fingerprint": protocol_fingerprint,
            "source": {
                "result": str(source_result.resolve()),
                "snapshot": str(source_snapshot.resolve()),
                "checkpoint_id": source_checkpoint_id,
                "energy": source_energy,
                "parameters": int(state.nparameters),
                "dense_parameters": int(state.dense_nparameters),
                "bond_dims": list(state.bond_dims),
            },
            "projection": {
                "definition": "P_{2Sz=0} |Psi(A)>",
                "fixed_projector": True,
                "all_tensor_coordinates_variational": True,
                "local_tensor_masks": False,
                "parameters": int(state.nparameters),
                "dense_parameters": int(state.dense_nparameters),
                "all_local_supports_dense": bool(
                    all(
                        active == dense for active, dense in state.local_support_sizes()
                    )
                ),
                "projector_bond_dims": list(state.projection.mpo_bond_dims),
                "maximum_projector_bond": int(state.projection.max_mpo_bond),
            },
            "contraction": {
                "frontier_backend": state.frontier_backend,
                "exact": bool(state.contraction_is_exact),
                "factorized_objective_mpo": bool(
                    state._hamiltonian_frontier.factorized_mpo
                ),
                "stored_objective_mpo_elements": int(
                    state._hamiltonian_frontier.stored_mpo_elements
                ),
                "materialized_objective_mpo_elements": int(
                    state._hamiltonian_frontier.dense_mpo_elements
                ),
                "objective_mpo_materialized": False,
            },
            "pairs": [],
            "result": {},
            "timing_seconds": {
                "initial_setup": float(perf_counter() - invocation_start),
                "pairs": [],
                "pair_total": 0.0,
                "total": 0.0,
            },
        }

    tensor_shapes = tuple(tensor.shape for tensor in sweep_start_tensors)
    bond_dims = tuple(state.bond_dims)
    _assert_full_parameterization(state, tensor_shapes, bond_dims)
    if next_pair_ordinal > maximum_pairs:
        raise ValueError("completed pair count exceeds maximum_pairs.")

    if next_pair_ordinal == maximum_pairs and maximum_pairs < NPAIRS:
        state.energy = measured_energy
        payload["status"] = "initialized" if next_pair_ordinal == 0 else "capped"
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - invocation_start
        )
        if next_pair_ordinal < NPAIRS and (
            next_pair_ordinal % checkpoint_interval == 0
            or next_pair_ordinal == maximum_pairs
        ):
            _checkpoint(
                output=output,
                snapshot=snapshot,
                payload=payload,
                state=state,
                sweep_start_tensors=sweep_start_tensors,
                tie_edges=tie_edges,
                source_energy=source_energy,
                sweep_start_energy=sweep_start_energy,
                directional_sweep=directional_sweep,
                next_pair_ordinal=next_pair_ordinal,
                pair_sites=pair_sites,
                endpoint_verified=False,
                endpoint_accepted=None,
                source_checkpoint_id=source_checkpoint_id,
            )
        print(
            f"exact projected two-site checkpoint: pairs={next_pair_ordinal}/"
            f"{NPAIRS}, E={state.energy:.12f}",
            flush=True,
        )
        return payload

    frontier_executor = (
        None
        if frontier_workers == 1
        else ThreadPoolExecutor(max_workers=frontier_workers)
    )
    (
        fixed_norm,
        fixed_hamiltonian,
        moving_norm,
        moving_hamiltonian,
    ) = _pair_messages(
        state,
        directional_sweep=directional_sweep,
        next_pair_ordinal=next_pair_ordinal,
        hamiltonian_workers=frontier_workers,
        hamiltonian_executor=frontier_executor,
    )
    for ordinal in range(next_pair_ordinal, maximum_pairs):
        site = pair_sites[ordinal]
        if directional_sweep % 2 == 0:
            environment = state._pair_environment_from_outer_messages(
                site,
                moving_norm,
                fixed_norm[site + 2],
                moving_hamiltonian,
                fixed_hamiltonian[site + 2],
                hamiltonian_workers=frontier_workers,
                hamiltonian_executor=frontier_executor,
            )
        else:
            environment = state._pair_environment_from_outer_messages(
                site,
                fixed_norm[site],
                moving_norm,
                fixed_hamiltonian[site],
                moving_hamiltonian,
                hamiltonian_workers=frontier_workers,
                hamiltonian_executor=frontier_executor,
            )
        pair_start = perf_counter()
        update = state.optimize_two_sites(
            site,
            solver="verified",
            environment=environment,
            verify_global=False,
            pair_operator_backend=pair_operator_backend,
            pair_dense_max_elements=pair_dense_max_elements,
            pair_operator_workers=pair_operator_workers,
            factor_solver=factor_solver,
            merged_dense_fallback_dim=merged_dense_fallback_dim,
            block_dense_component_max_size=int(block_dense_component_max_size),
            metric_support="numerical",
            outer_cycles=int(outer_cycles),
            split_strategy="variational",
            metric_tol=float(metric_tol),
            eig_tol=float(eig_tol),
            maxiter=int(maxiter),
            max_subspace=int(max_subspace),
            split_metric_tol=float(split_metric_tol),
            split_metric_sweeps=int(split_metric_sweeps),
            split_variational_sweeps=int(split_variational_sweeps),
            split_random_starts=0,
            split_energy_tol=1.0e-12,
        )
        pair_seconds = float(perf_counter() - pair_start)
        if pair_operator_backend != "auto":
            if update.pair_operator_backend != pair_operator_backend:
                raise RuntimeError(
                    "a pair did not use the explicitly requested backend."
                )
        if update.merged_solve is None:
            raise RuntimeError("a pair did not return merged-solve diagnostics.")
        if (
            update.pair_operator_backend != "dense"
            and update.merged_solve.dense_fallback
        ):
            raise RuntimeError(
                "a conditional-block pair unexpectedly used a dense fallback."
            )
        if (
            update.pair_operator_backend == "dense"
            and 2 * update.raw_merged_dim**2 > pair_dense_max_elements
            and pair_operator_backend != "dense"
        ):
            raise RuntimeError("automatic dense pair selection exceeded its budget.")
        _assert_full_parameterization(state, tensor_shapes, bond_dims)
        payload["pairs"].append(
            _pair_record(
                update,
                ordinal=ordinal,
                site=site,
                seconds=pair_seconds,
            )
        )
        if directional_sweep % 2 == 0:
            moving_norm = state._norm_frontier.advance_left(
                moving_norm, state.tensors, site
            )
            moving_hamiltonian = state._hamiltonian_frontier.advance_left(
                moving_hamiltonian,
                state.tensors,
                site,
                max_workers=frontier_workers,
                executor=frontier_executor,
            )
        else:
            moving_norm = state._norm_frontier.advance_right(
                moving_norm, state.tensors, site + 1
            )
            moving_hamiltonian = state._hamiltonian_frontier.advance_right(
                moving_hamiltonian,
                state.tensors,
                site + 1,
                max_workers=frontier_workers,
                executor=frontier_executor,
            )
        next_pair_ordinal = ordinal + 1
        payload["status"] = "running" if next_pair_ordinal < maximum_pairs else "capped"
        payload["timing_seconds"]["pairs"] = [
            float(record["seconds"]) for record in payload["pairs"]
        ]
        payload["timing_seconds"]["pair_total"] = float(
            sum(payload["timing_seconds"]["pairs"])
        )
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - invocation_start
        )
        state.energy = float(update.energy)
        _checkpoint(
            output=output,
            snapshot=snapshot,
            payload=payload,
            state=state,
            sweep_start_tensors=sweep_start_tensors,
            tie_edges=tie_edges,
            source_energy=source_energy,
            sweep_start_energy=sweep_start_energy,
            directional_sweep=directional_sweep,
            next_pair_ordinal=next_pair_ordinal,
            pair_sites=pair_sites,
            endpoint_verified=False,
            endpoint_accepted=None,
            source_checkpoint_id=source_checkpoint_id,
        )
        print(
            f"pair {next_pair_ordinal:2d}/{NPAIRS} sites=({site},{site + 1}) "
            f"E={state.energy:.12f} accepted={update.accepted} "
            f"seconds={pair_seconds:.1f}",
            flush=True,
        )

    if frontier_executor is not None:
        frontier_executor.shutdown()

    if next_pair_ordinal == NPAIRS:
        attempted_energy, norm = _endpoint_energy(
            state,
            directional_sweep=directional_sweep,
            moving_norm=moving_norm,
            moving_hamiltonian=moving_hamiltonian,
        )
        tolerance = 512.0 * np.finfo(float).eps * max(1.0, abs(sweep_start_energy))
        endpoint_accepted = bool(
            np.isfinite(attempted_energy)
            and attempted_energy <= sweep_start_energy + tolerance
        )
        if endpoint_accepted:
            state.energy = attempted_energy
        else:
            state.tensors = [
                np.asarray(tensor).copy() for tensor in sweep_start_tensors
            ]
            state.energy = sweep_start_energy
        _assert_full_parameterization(state, tensor_shapes, bond_dims)
        payload["status"] = "complete"
        payload["endpoint"] = {
            "attempted_energy": float(attempted_energy),
            "energy": float(state.energy),
            "accepted": endpoint_accepted,
            "energy_gain": float(sweep_start_energy - state.energy),
            "norm_before_balancing": float(norm),
        }
        payload["timing_seconds"]["total"] = float(
            elapsed_before + perf_counter() - invocation_start
        )
        _checkpoint(
            output=output,
            snapshot=snapshot,
            payload=payload,
            state=state,
            sweep_start_tensors=sweep_start_tensors,
            tie_edges=tie_edges,
            source_energy=source_energy,
            sweep_start_energy=sweep_start_energy,
            directional_sweep=directional_sweep,
            next_pair_ordinal=next_pair_ordinal,
            pair_sites=pair_sites,
            endpoint_verified=True,
            endpoint_accepted=endpoint_accepted,
            source_checkpoint_id=source_checkpoint_id,
        )
        print(
            f"two-site directional endpoint E={state.energy:.12f}, "
            f"accepted={endpoint_accepted}, pairs={NPAIRS}",
            flush=True,
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--maximum-pairs", type=int, default=NPAIRS)
    parser.add_argument("--directional-sweep", type=int, default=0)
    parser.add_argument("--outer-cycles", type=int, default=1)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--max-subspace", type=int, default=96)
    parser.add_argument("--block-dense-component-max-size", type=int, default=64)
    parser.add_argument(
        "--pair-operator-backend",
        choices=("auto", "dense", "block"),
        default="auto",
    )
    parser.add_argument(
        "--pair-dense-max-elements",
        type=int,
        default=2_000_000,
        help="maximum combined N_eff/H_eff elements for automatic dense pairs",
    )
    parser.add_argument(
        "--pair-operator-workers",
        type=int,
        default=4,
        help="ordered contraction workers for dense local Hamiltonians",
    )
    parser.add_argument(
        "--frontier-workers",
        type=int,
        default=2,
        help="ordered contraction workers for frontier-message updates",
    )
    parser.add_argument(
        "--factor-solver",
        choices=("auto", "dense", "matrix_free"),
        default="auto",
    )
    parser.add_argument(
        "--merged-dense-fallback-dim",
        type=int,
        default=2048,
        help="dense certification limit after a dense local pair is selected",
    )
    parser.add_argument("--split-metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--split-metric-sweeps", type=int, default=2)
    parser.add_argument("--split-variational-sweeps", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        output=args.output,
        snapshot=args.snapshot,
        maximum_pairs=args.maximum_pairs,
        directional_sweep=args.directional_sweep,
        outer_cycles=args.outer_cycles,
        metric_tol=args.metric_tol,
        eig_tol=args.eig_tol,
        maxiter=args.maxiter,
        max_subspace=args.max_subspace,
        block_dense_component_max_size=args.block_dense_component_max_size,
        pair_operator_backend=args.pair_operator_backend,
        pair_dense_max_elements=args.pair_dense_max_elements,
        pair_operator_workers=args.pair_operator_workers,
        frontier_workers=args.frontier_workers,
        factor_solver=args.factor_solver,
        merged_dense_fallback_dim=args.merged_dense_fallback_dim,
        split_metric_tol=args.split_metric_tol,
        split_metric_sweeps=args.split_metric_sweeps,
        split_variational_sweeps=args.split_variational_sweeps,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
