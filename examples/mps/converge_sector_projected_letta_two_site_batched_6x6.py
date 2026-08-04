#!/usr/bin/env python3
"""Converge exact projected 6x6 LETTA with cached alternating two-site passes.

The state is ``P_{2Sz=0}|Psi(A)>`` with all 4008 unrestricted LETTA
coordinates active.  One state and one frontier executor remain alive for the
whole invocation.  Messages generated on the moving side of an accepted
directional pass are the exact all-cut fixed messages for the reverse pass, so
they are reused without rebuilding the frontier.

Only accepted directional endpoints are checkpointed.  A crash inside a pass
therefore resumes from the previous verified endpoint.

The message flow is the DMRG ``SweepEnvironment`` pattern: one all-cut side
is fixed, the opposite side advances after each accepted pair, and the
completed moving messages become the exact fixed cache for the reverse pass.
While pair contraction plans are cold, one fixed-side pair binding is
prefetched ahead with bounded memory; warm sweeps remain serial to avoid CPU
contention.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps import converge_sector_projected_letta_two_site_6x6 as single


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE_RESULT = (
    RESULTS
    / "frontier_letta_sector_projected_u1_two_site_converge_sweep16_6x6_j2_0p5.json"
)
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_SOURCE_RESULT.with_suffix(".npz")
DEFAULT_OUTPUT = (
    RESULTS / "frontier_letta_sector_projected_u1_two_site_batched_6x6_j2_0p5.json"
)
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT.with_suffix(".npz")

NSITES = single.NSITES
NPAIRS = single.NPAIRS
EXPECTED_PARAMETERS = single.EXPECTED_PARAMETERS
CHECKPOINT_PAIR_ENERGY_TOL = 5.0e-10
RECONSTRUCTION_ENERGY_TOL_PER_SITE = 1.0e-7
RECONSTRUCTION_ENERGY_TOL = NSITES * RECONSTRUCTION_ENERGY_TOL_PER_SITE


def _stable_protocol(protocol):
    stable = dict(protocol)
    stable.pop("maximum_directional_passes", None)
    return stable


def _target_model(source_model, j2):
    model = dict(source_model)
    source_j2 = float(model["j2"])
    target_j2 = source_j2 if j2 is None else float(j2)
    if not np.isfinite(target_j2) or target_j2 < 0.0:
        raise ValueError("j2 must be finite and nonnegative.")
    model["j2"] = target_j2
    return model, source_j2, target_j2


def _protocol_fingerprint(protocol):
    encoded = json.dumps(
        _stable_protocol(protocol),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _checkpoint_id(protocol_fingerprint, next_directional_sweep, energy, digest):
    encoded = (
        f"{protocol_fingerprint}:{int(next_directional_sweep)}:"
        f"{float(energy).hex()}:{digest}"
    ).encode()
    return sha256(encoded).hexdigest()


def _message_cache(
    state,
    orientation,
    norm_messages,
    hamiltonian_messages,
):
    orientation = str(orientation)
    ncuts = len(state.dims) + 1
    norm_messages = tuple(norm_messages)
    hamiltonian_messages = tuple(hamiltonian_messages)
    if orientation not in {"left", "right"}:
        raise ValueError("message-cache orientation must be 'left' or 'right'.")
    if len(norm_messages) != ncuts or len(hamiltonian_messages) != ncuts:
        raise ValueError("message cache must contain every site cut.")
    if any(message is None for message in (*norm_messages, *hamiltonian_messages)):
        raise ValueError("message cache contains an unfilled cut.")
    for cut, message in enumerate(hamiltonian_messages):
        if getattr(message, "cut", cut) != cut:
            raise ValueError("Hamiltonian message belongs to the wrong cut.")
    return {
        "orientation": orientation,
        "norm": norm_messages,
        "hamiltonian": hamiltonian_messages,
        "tensor_digest": single._tensor_digest(state.tensors),
    }


def _start_directional_messages(
    state,
    directional_sweep,
    reusable,
    *,
    frontier_workers,
    frontier_executor,
):
    nsites = len(state.dims)
    left_to_right = int(directional_sweep) % 2 == 0
    required_orientation = "right" if left_to_right else "left"
    build_start = perf_counter()
    reused = reusable is not None
    if reused:
        if reusable["orientation"] != required_orientation:
            raise ValueError("reusable messages have the wrong orientation.")
        if reusable["tensor_digest"] != single._tensor_digest(state.tensors):
            raise ValueError("reusable messages are stale for the current tensors.")
        fixed_norm = reusable["norm"]
        fixed_hamiltonian = reusable["hamiltonian"]
    elif left_to_right:
        fixed_norm = tuple(state._norm_frontier.build_right(state.tensors))
        fixed_hamiltonian = tuple(
            state._hamiltonian_frontier.build_right(
                state.tensors,
                max_workers=frontier_workers,
                executor=frontier_executor,
            )
        )
    else:
        fixed_norm = tuple(state._norm_frontier.build_left(state.tensors))
        fixed_hamiltonian = tuple(
            state._hamiltonian_frontier.build_left(
                state.tensors,
                max_workers=frontier_workers,
                executor=frontier_executor,
            )
        )
    build_seconds = float(perf_counter() - build_start)

    generated_norm = [None] * (nsites + 1)
    generated_hamiltonian = [None] * (nsites + 1)
    if left_to_right:
        moving_norm = state._norm_frontier.left_boundary()
        moving_hamiltonian = state._hamiltonian_frontier.left_boundary()
        generated_norm[0] = moving_norm
        generated_hamiltonian[0] = moving_hamiltonian
    else:
        moving_norm = state._norm_frontier.right_boundary()
        moving_hamiltonian = state._hamiltonian_frontier.right_boundary()
        generated_norm[nsites] = moving_norm
        generated_hamiltonian[nsites] = moving_hamiltonian
    return {
        "fixed_norm": fixed_norm,
        "fixed_hamiltonian": fixed_hamiltonian,
        "moving_norm": moving_norm,
        "moving_hamiltonian": moving_hamiltonian,
        "generated_norm": generated_norm,
        "generated_hamiltonian": generated_hamiltonian,
        "fixed_reused": bool(reused),
        "fixed_build_seconds": build_seconds,
    }


def _advance_directional_messages(
    state,
    directional_sweep,
    site,
    messages,
    *,
    frontier_workers,
    frontier_executor,
):
    if int(directional_sweep) % 2 == 0:
        cut = int(site) + 1
        messages["moving_norm"] = state._norm_frontier.advance_left(
            messages["moving_norm"],
            state.tensors,
            site,
        )
        messages["moving_hamiltonian"] = state._hamiltonian_frontier.advance_left(
            messages["moving_hamiltonian"],
            state.tensors,
            site,
            max_workers=frontier_workers,
            executor=frontier_executor,
        )
    else:
        cut = int(site) + 1
        messages["moving_norm"] = state._norm_frontier.advance_right(
            messages["moving_norm"],
            state.tensors,
            cut,
        )
        messages["moving_hamiltonian"] = state._hamiltonian_frontier.advance_right(
            messages["moving_hamiltonian"],
            state.tensors,
            cut,
            max_workers=frontier_workers,
            executor=frontier_executor,
        )
    messages["generated_norm"][cut] = messages["moving_norm"]
    messages["generated_hamiltonian"][cut] = messages["moving_hamiltonian"]


def _complete_directional_messages(
    state,
    directional_sweep,
    messages,
    *,
    frontier_workers,
    frontier_executor,
):
    if int(directional_sweep) % 2 == 0:
        site = len(state.dims) - 1
        orientation = "left"
        boundary_cut = len(state.dims)
        messages["moving_norm"] = state._norm_frontier.advance_left(
            messages["moving_norm"],
            state.tensors,
            site,
        )
        messages["moving_hamiltonian"] = state._hamiltonian_frontier.advance_left(
            messages["moving_hamiltonian"],
            state.tensors,
            site,
            max_workers=frontier_workers,
            executor=frontier_executor,
        )
    else:
        site = 0
        orientation = "right"
        boundary_cut = 0
        messages["moving_norm"] = state._norm_frontier.advance_right(
            messages["moving_norm"],
            state.tensors,
            site,
        )
        messages["moving_hamiltonian"] = state._hamiltonian_frontier.advance_right(
            messages["moving_hamiltonian"],
            state.tensors,
            site,
            max_workers=frontier_workers,
            executor=frontier_executor,
        )
    messages["generated_norm"][boundary_cut] = messages["moving_norm"]
    messages["generated_hamiltonian"][boundary_cut] = messages["moving_hamiltonian"]
    norm = float(
        np.real(
            state._completed_frontier_scalar(
                state._norm_frontier,
                messages["moving_norm"],
                boundary_cut,
            )
        )
    )
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("projected LETTA state is numerically zero.")
    numerator = state._completed_frontier_scalar(
        state._hamiltonian_frontier,
        messages["moving_hamiltonian"],
        boundary_cut,
    )
    energy = float(np.real(numerator / norm))
    cache = _message_cache(
        state,
        orientation,
        messages["generated_norm"],
        messages["generated_hamiltonian"],
    )
    return energy, norm, cache


def _validate_pair_update(update, *, requested_backend, dense_budget):
    if (
        requested_backend != "auto"
        and update.pair_operator_backend != requested_backend
    ):
        raise RuntimeError("a pair did not use the explicitly requested backend.")
    if update.merged_solve is None:
        raise RuntimeError("a pair did not return merged-solve diagnostics.")
    if update.pair_operator_backend != "dense" and update.merged_solve.dense_fallback:
        raise RuntimeError("a conditional-block pair used a dense fallback.")
    if (
        requested_backend == "auto"
        and update.pair_operator_backend == "dense"
        and 2 * int(update.raw_merged_dim) ** 2 > int(dense_budget)
    ):
        raise RuntimeError("automatic dense pair selection exceeded its budget.")


def _run_directional_pass(
    state,
    directional_sweep,
    reusable,
    *,
    tensor_shapes,
    bond_dims,
    frontier_executor,
    pair_environment_executor,
    options,
):
    start = perf_counter()
    sweep_start_tensors = [tensor.copy() for tensor in state.tensors]
    sweep_start_energy = float(state.energy)
    messages = _start_directional_messages(
        state,
        directional_sweep,
        reusable,
        frontier_workers=options["frontier_workers"],
        frontier_executor=frontier_executor,
    )
    left_to_right = int(directional_sweep) % 2 == 0
    pair_sites = tuple(
        range(NPAIRS) if left_to_right else range(NPAIRS - 1, -1, -1)
    )
    pair_plan_fingerprint = (
        state._pair_plan_fingerprint() if left_to_right else None
    )
    pair_plans_cold = bool(
        left_to_right
        and any(
            site not in state._pair_plan_cache
            or state._pair_plan_cache[site].fingerprint
            != pair_plan_fingerprint
            for site in pair_sites
        )
    )
    prefetch_enabled = bool(
        pair_plans_cold and options["pair_environment_prefetch"]
    )
    bound_right_future = None
    if prefetch_enabled:
        first_site = pair_sites[0]
        bound_right_future = pair_environment_executor.submit(
            state._bind_pair_right_environment,
            first_site,
            messages["fixed_norm"][first_site + 2],
            messages["fixed_hamiltonian"][first_site + 2],
            hamiltonian_workers=1,
        )
    pairs = []
    environment_seconds = 0.0
    environment_prefetch_wait_seconds = 0.0
    moving_message_seconds = 0.0
    for ordinal, site in enumerate(pair_sites):
        environment_start = perf_counter()
        if prefetch_enabled:
            wait_start = perf_counter()
            bound_right = bound_right_future.result()
            environment_prefetch_wait_seconds += perf_counter() - wait_start
            next_ordinal = ordinal + 1
            bound_right_future = (
                None
                if next_ordinal == len(pair_sites)
                else pair_environment_executor.submit(
                    state._bind_pair_right_environment,
                    pair_sites[next_ordinal],
                    messages["fixed_norm"][pair_sites[next_ordinal] + 2],
                    messages["fixed_hamiltonian"][
                        pair_sites[next_ordinal] + 2
                    ],
                    hamiltonian_workers=1,
                )
            )
            environment = state._pair_environment_from_bound_right(
                site,
                messages["moving_norm"],
                messages["moving_hamiltonian"],
                bound_right,
            )
        elif left_to_right:
            environment = state._pair_environment_from_outer_messages(
                site,
                messages["moving_norm"],
                messages["fixed_norm"][site + 2],
                messages["moving_hamiltonian"],
                messages["fixed_hamiltonian"][site + 2],
                hamiltonian_workers=options["frontier_workers"],
                hamiltonian_executor=frontier_executor,
            )
        else:
            environment = state._pair_environment_from_outer_messages(
                site,
                messages["fixed_norm"][site],
                messages["moving_norm"],
                messages["fixed_hamiltonian"][site],
                messages["moving_hamiltonian"],
                hamiltonian_workers=options["frontier_workers"],
                hamiltonian_executor=frontier_executor,
            )
        environment_seconds += perf_counter() - environment_start
        pair_start = perf_counter()
        update = state.optimize_two_sites(
            site,
            solver="verified",
            environment=environment,
            verify_global=False,
            pair_operator_backend=options["pair_operator_backend"],
            pair_dense_max_elements=options["pair_dense_max_elements"],
            pair_operator_workers=options["pair_operator_workers"],
            factor_solver=options["factor_solver"],
            merged_dense_fallback_dim=options["merged_dense_fallback_dim"],
            block_dense_component_max_size=options["block_dense_component_max_size"],
            metric_support="numerical",
            outer_cycles=options["outer_cycles"],
            split_strategy="variational",
            metric_tol=options["metric_tol"],
            eig_tol=options["eig_tol"],
            maxiter=options["maxiter"],
            max_subspace=options["max_subspace"],
            skip_redundant_full_rank_davidson=options[
                "skip_redundant_full_rank_davidson"
            ],
            redundant_full_rank_davidson_min_dimension=options[
                "redundant_full_rank_davidson_min_dimension"
            ],
            split_metric_tol=options["split_metric_tol"],
            split_metric_sweeps=options["split_metric_sweeps"],
            split_variational_sweeps=options["split_variational_sweeps"],
            split_random_starts=0,
            split_energy_tol=1.0e-12,
        )
        pair_seconds = float(perf_counter() - pair_start)
        _validate_pair_update(
            update,
            requested_backend=options["pair_operator_backend"],
            dense_budget=options["pair_dense_max_elements"],
        )
        single._assert_full_parameterization(state, tensor_shapes, bond_dims)
        pairs.append(
            single._pair_record(
                update,
                ordinal=ordinal,
                site=site,
                seconds=pair_seconds,
            )
        )
        state.energy = float(update.energy)
        moving_message_start = perf_counter()
        _advance_directional_messages(
            state,
            directional_sweep,
            site,
            messages,
            frontier_workers=options["frontier_workers"],
            frontier_executor=frontier_executor,
        )
        moving_message_seconds += perf_counter() - moving_message_start

    endpoint_start = perf_counter()
    attempted_energy, norm, generated = _complete_directional_messages(
        state,
        directional_sweep,
        messages,
        frontier_workers=options["frontier_workers"],
        frontier_executor=frontier_executor,
    )
    endpoint_seconds = float(perf_counter() - endpoint_start)
    endpoint_tolerance = (
        512.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(sweep_start_energy),
        )
    )
    accepted = bool(
        np.isfinite(attempted_energy)
        and attempted_energy <= sweep_start_energy + endpoint_tolerance
    )
    if accepted:
        state.energy = attempted_energy
        reusable = generated
    else:
        state.tensors = sweep_start_tensors
        state.energy = sweep_start_energy
        reusable = None
    single._assert_full_parameterization(state, tensor_shapes, bond_dims)
    energy_gain = float(sweep_start_energy - state.energy)
    return (
        {
            "directional_sweep": int(directional_sweep),
            "direction": (
                "left_to_right" if int(directional_sweep) % 2 == 0 else "right_to_left"
            ),
            "energy_before": sweep_start_energy,
            "attempted_energy": attempted_energy,
            "energy": float(state.energy),
            "energy_gain": energy_gain,
            "energy_gain_per_site": energy_gain / NSITES,
            "accepted": accepted,
            "norm": norm,
            "fixed_messages_reused": messages["fixed_reused"],
            "fixed_message_build_seconds": messages["fixed_build_seconds"],
            "pair_environment_seconds": float(environment_seconds),
            "pair_environment_prefetch": prefetch_enabled,
            "pair_environment_plans_cold": pair_plans_cold,
            "pair_environment_prefetch_wait_seconds": float(
                environment_prefetch_wait_seconds
            ),
            "moving_message_seconds": float(moving_message_seconds),
            "endpoint_seconds": endpoint_seconds,
            "pair_total_seconds": float(sum(pair["seconds"] for pair in pairs)),
            "seconds": float(perf_counter() - start),
            "accepted_pairs": int(sum(pair["accepted"] for pair in pairs)),
            "pairs": pairs,
        },
        reusable,
    )


def _save_snapshot_atomic(
    path,
    state,
    *,
    tie_edges,
    protocol_fingerprint,
    checkpoint_id,
    directional_passes_digest,
    source_checkpoint_id,
    next_directional_sweep,
    converged,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        virtual_bond_dims=np.asarray(state.bond_dims, dtype=np.int64),
        tie_edges=np.asarray(tie_edges, dtype=np.int64),
        recorded_energy=np.asarray(float(state.energy)),
        target_two_sz=np.asarray(0, dtype=np.int64),
        local_two_sz=np.asarray(((1, -1),) * NSITES, dtype=np.int64),
        parameter_count=np.asarray(int(state.nparameters), dtype=np.int64),
        dense_parameter_count=np.asarray(
            int(state.dense_nparameters),
            dtype=np.int64,
        ),
        protocol_fingerprint=np.asarray(str(protocol_fingerprint)),
        checkpoint_id=np.asarray(str(checkpoint_id)),
        directional_passes_digest=np.asarray(str(directional_passes_digest)),
        current_tensor_digest=np.asarray(single._tensor_digest(state.tensors)),
        source_checkpoint_id=np.asarray(str(source_checkpoint_id)),
        next_directional_sweep=np.asarray(
            int(next_directional_sweep),
            dtype=np.int64,
        ),
        converged=np.asarray(bool(converged)),
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    temporary.replace(path)


def _read_snapshot_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "recorded_energy",
            "parameter_count",
            "dense_parameter_count",
            "protocol_fingerprint",
            "checkpoint_id",
            "directional_passes_digest",
            "current_tensor_digest",
            "source_checkpoint_id",
            "next_directional_sweep",
            "converged",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"batch snapshot is missing metadata: {', '.join(missing)}"
            )
        tensors = [
            np.array(archive[f"tensor_{site:03d}"], copy=True) for site in range(NSITES)
        ]
        return {
            "energy": float(archive["recorded_energy"]),
            "parameter_count": int(archive["parameter_count"]),
            "dense_parameter_count": int(archive["dense_parameter_count"]),
            "protocol_fingerprint": str(archive["protocol_fingerprint"]),
            "checkpoint_id": str(archive["checkpoint_id"]),
            "directional_passes_digest": str(archive["directional_passes_digest"]),
            "current_tensor_digest": str(archive["current_tensor_digest"]),
            "source_checkpoint_id": str(archive["source_checkpoint_id"]),
            "next_directional_sweep": int(archive["next_directional_sweep"]),
            "converged": bool(archive["converged"]),
            "tensors": tensors,
        }


def _checkpoint(
    *,
    output,
    snapshot,
    payload,
    state,
    tie_edges,
    source_checkpoint_id,
):
    passes_digest = single._records_digest(payload["directional_passes"])
    next_directional_sweep = int(payload["result"]["next_directional_sweep"])
    checkpoint_id = _checkpoint_id(
        payload["protocol_fingerprint"],
        next_directional_sweep,
        state.energy,
        passes_digest,
    )
    payload["result"]["checkpoint_id"] = checkpoint_id
    payload["directional_passes_digest"] = passes_digest
    _save_snapshot_atomic(
        snapshot,
        state,
        tie_edges=tie_edges,
        protocol_fingerprint=payload["protocol_fingerprint"],
        checkpoint_id=checkpoint_id,
        directional_passes_digest=passes_digest,
        source_checkpoint_id=source_checkpoint_id,
        next_directional_sweep=next_directional_sweep,
        converged=payload["result"]["converged"],
    )
    single._write_json_atomic(output, payload)


def _validate_resume(payload, metadata, protocol, source_checkpoint_id):
    passes = payload.get("directional_passes", [])
    expected_digest = single._records_digest(passes)
    checks = (
        (
            payload.get("protocol_fingerprint"),
            _protocol_fingerprint(protocol),
            "JSON protocol fingerprint",
        ),
        (
            metadata["protocol_fingerprint"],
            _protocol_fingerprint(protocol),
            "snapshot protocol fingerprint",
        ),
        (
            payload.get("result", {}).get("checkpoint_id"),
            metadata["checkpoint_id"],
            "checkpoint ID",
        ),
        (
            payload.get("directional_passes_digest"),
            expected_digest,
            "JSON directional-pass digest",
        ),
        (
            metadata["directional_passes_digest"],
            expected_digest,
            "snapshot directional-pass digest",
        ),
        (
            metadata["current_tensor_digest"],
            single._tensor_digest(metadata["tensors"]),
            "tensor digest",
        ),
        (
            metadata["source_checkpoint_id"],
            source_checkpoint_id,
            "source checkpoint ID",
        ),
        (metadata["parameter_count"], EXPECTED_PARAMETERS, "parameter count"),
        (
            metadata["dense_parameter_count"],
            EXPECTED_PARAMETERS,
            "dense parameter count",
        ),
        (
            metadata["next_directional_sweep"],
            payload.get("result", {}).get("next_directional_sweep"),
            "next directional sweep",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RuntimeError(
                f"unsafe batch resume: {label} mismatch ({actual!r} != {expected!r})."
            )
    if _stable_protocol(payload["protocol"]) != _stable_protocol(protocol):
        raise RuntimeError("unsafe batch resume: numerical protocol changed.")
    if (
        abs(metadata["energy"] - float(payload["result"]["energy"]))
        > CHECKPOINT_PAIR_ENERGY_TOL
    ):
        raise RuntimeError("unsafe batch resume: JSON and snapshot energies differ.")


def _result_record(
    state,
    *,
    source_energy,
    directional_passes,
    next_directional_sweep,
    gain_tolerance,
    maximum_directional_passes,
    snapshot,
):
    last = directional_passes[-1] if directional_passes else None
    last_cycle = directional_passes[-2:]
    cycle_gain = (
        None
        if len(last_cycle) < 2
        else max(abs(float(row["energy_gain"])) for row in last_cycle)
    )
    cycle_gain_per_site = (
        None if cycle_gain is None else float(cycle_gain / NSITES)
    )
    converged = bool(
        len(last_cycle) == 2
        and all(row["accepted"] for row in last_cycle)
        and cycle_gain_per_site < float(gain_tolerance)
    )
    if last is not None and not last["accepted"]:
        stop_reason = "directional endpoint rejected and rolled back"
    elif converged:
        stop_reason = (
            "both directional gains per site in the last cycle are below "
            f"{float(gain_tolerance):.6e}"
        )
    elif len(directional_passes) >= int(maximum_directional_passes):
        stop_reason = "maximum directional-pass cap reached"
    else:
        stop_reason = "more directional passes requested"
    return {
        "converged": converged,
        "stop_reason": stop_reason,
        "directional_passes_completed": len(directional_passes),
        "next_directional_sweep": int(next_directional_sweep),
        "energy": float(state.energy),
        "energy_per_site": float(state.energy) / NSITES,
        "energy_gain_from_source": float(source_energy - state.energy),
        "energy_gain_from_source_per_site": (
            float(source_energy - state.energy) / NSITES
        ),
        "last_directional_gain": (None if last is None else float(last["energy_gain"])),
        "last_directional_gain_per_site": (
            None if last is None else float(last["energy_gain"]) / NSITES
        ),
        "last_cycle_maximum_gain": cycle_gain,
        "last_cycle_maximum_gain_per_site": cycle_gain_per_site,
        "parameters": int(state.nparameters),
        "dense_parameters": int(state.dense_nparameters),
        "all_tensor_coordinates_variational": True,
        "local_tensor_masks": False,
        "bond_dims": list(state.bond_dims),
        "fixed_message_reuses": int(
            sum(row["fixed_messages_reused"] for row in directional_passes)
        ),
        "snapshot": str(Path(snapshot).resolve()),
    }


def run(
    *,
    source_result=DEFAULT_SOURCE_RESULT,
    source_snapshot=DEFAULT_SOURCE_SNAPSHOT,
    output=DEFAULT_OUTPUT,
    snapshot=DEFAULT_SNAPSHOT,
    j2=None,
    starting_directional_sweep=16,
    maximum_directional_passes=16,
    gain_tolerance=1.0e-4,
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
    pair_environment_prefetch=True,
    skip_redundant_full_rank_davidson=True,
    redundant_full_rank_davidson_min_dimension=256,
    resume=True,
):
    source_result = Path(source_result)
    source_snapshot = Path(source_snapshot)
    output = Path(output)
    snapshot = Path(snapshot)
    starting_directional_sweep = int(starting_directional_sweep)
    maximum_directional_passes = int(maximum_directional_passes)
    gain_tolerance = float(gain_tolerance)
    pair_operator_backend = str(pair_operator_backend).lower().replace("-", "_")
    factor_solver = str(factor_solver).lower().replace("-", "_")
    pair_environment_prefetch = bool(pair_environment_prefetch)
    skip_redundant_full_rank_davidson = bool(skip_redundant_full_rank_davidson)
    redundant_full_rank_davidson_min_dimension = int(
        redundant_full_rank_davidson_min_dimension
    )
    if starting_directional_sweep < 0:
        raise ValueError("starting_directional_sweep must be nonnegative.")
    if maximum_directional_passes < 1:
        raise ValueError("maximum_directional_passes must be positive.")
    if not np.isfinite(gain_tolerance) or gain_tolerance < 0.0:
        raise ValueError("gain_tolerance must be finite and nonnegative.")
    if pair_operator_backend not in {"auto", "dense", "block"}:
        raise ValueError("pair_operator_backend must be 'auto', 'dense', or 'block'.")
    if factor_solver not in {"auto", "dense", "matrix_free"}:
        raise ValueError("factor_solver must be 'auto', 'dense', or 'matrix_free'.")
    if int(pair_dense_max_elements) < 1:
        raise ValueError("pair_dense_max_elements must be positive.")
    if int(pair_operator_workers) < 1 or int(frontier_workers) < 1:
        raise ValueError("worker counts must be positive.")
    if int(merged_dense_fallback_dim) < 1:
        raise ValueError("merged_dense_fallback_dim must be positive.")
    if redundant_full_rank_davidson_min_dimension < 0:
        raise ValueError(
            "redundant_full_rank_davidson_min_dimension must be nonnegative."
        )
    protected = {source_result.resolve(), source_snapshot.resolve()}
    if output.resolve() in protected or snapshot.resolve() in protected:
        raise ValueError("batch checkpoints must not overwrite source artifacts.")

    source = json.loads(source_result.read_text(encoding="utf-8"))
    model, source_j2, target_j2 = _target_model(source["model"], j2)
    if (
        int(model["nrows"]) != 6
        or int(model["ncols"]) != 6
        or int(model["nsites"]) != NSITES
    ):
        raise ValueError("the source checkpoint must be the 6x6 model.")
    source_checkpoint_id = str(source["result"]["checkpoint_id"])
    source_checkpoint_energy = float(source["result"]["energy"])
    target_differs = target_j2 != source_j2
    if target_differs and (
        output.resolve() == DEFAULT_OUTPUT.resolve()
        or snapshot.resolve() == DEFAULT_SNAPSHOT.resolve()
    ):
        raise ValueError(
            "a target j2 different from the source requires explicit "
            "--output and --snapshot paths."
        )
    options = {
        "outer_cycles": int(outer_cycles),
        "metric_tol": float(metric_tol),
        "eig_tol": float(eig_tol),
        "maxiter": int(maxiter),
        "max_subspace": int(max_subspace),
        "block_dense_component_max_size": int(block_dense_component_max_size),
        "pair_operator_backend": pair_operator_backend,
        "pair_dense_max_elements": int(pair_dense_max_elements),
        "pair_operator_workers": int(pair_operator_workers),
        "frontier_workers": int(frontier_workers),
        "factor_solver": factor_solver,
        "merged_dense_fallback_dim": int(merged_dense_fallback_dim),
        "split_metric_tol": float(split_metric_tol),
        "split_metric_sweeps": int(split_metric_sweeps),
        "split_variational_sweeps": int(split_variational_sweeps),
        "pair_environment_prefetch": pair_environment_prefetch,
        "skip_redundant_full_rank_davidson": (skip_redundant_full_rank_davidson),
        "redundant_full_rank_davidson_min_dimension": (
            redundant_full_rank_davidson_min_dimension
        ),
    }
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
        "source_j2": source_j2,
        "target_j2": target_j2,
        "source_tensors_reused_at_new_hamiltonian": target_differs,
        "frontier_backend": "identity_block",
        "objective_mpo": "sparse factorized H times P_Q",
        "materialize_objective_mpo": False,
        "contraction": "exact",
        "optimization": "batched fixed-bond alternating two-site passes",
        "starting_directional_sweep": starting_directional_sweep,
        "maximum_directional_passes": maximum_directional_passes,
        "gain_tolerance": gain_tolerance,
        "gain_tolerance_units": "energy_per_site",
        "checkpoint_pair_energy_tolerance": CHECKPOINT_PAIR_ENERGY_TOL,
        "reconstruction_energy_tolerance_per_site": (
            RECONSTRUCTION_ENERGY_TOL_PER_SITE
        ),
        "convergence_rule": (
            "both directional gains per site in one complete alternating "
            "cycle are below gain_tolerance"
        ),
        "endpoint_verification": True,
        "checkpoint_granularity": "accepted or rolled-back directional endpoints",
        "cross_direction_fixed_message_reuse": True,
        "gauge_balancing_between_passes": False,
        **options,
    }
    protocol_fingerprint = _protocol_fingerprint(protocol)
    invocation_start = perf_counter()

    output_exists = output.is_file()
    snapshot_exists = snapshot.is_file()
    if resume and output_exists != snapshot_exists:
        raise RuntimeError(
            "unsafe batch resume: JSON and NPZ must both exist or both be absent."
        )
    resumed = bool(resume and output_exists and snapshot_exists)
    if resumed:
        payload = json.loads(output.read_text(encoding="utf-8"))
        metadata = _read_snapshot_metadata(snapshot)
        _validate_resume(payload, metadata, protocol, source_checkpoint_id)
        if payload["model"] != model:
            raise RuntimeError("batch checkpoint belongs to a different model.")
        state = single._projected_state(model, snapshot)
        if (
            abs(float(state.energy) - metadata["energy"])
            > RECONSTRUCTION_ENERGY_TOL
        ):
            raise RuntimeError(
                "batch snapshot does not reproduce its energy: "
                f"recorded={metadata['energy']:.16g}, "
                f"reconstructed={float(state.energy):.16g}, "
                f"difference={float(state.energy) - metadata['energy']:.6e}."
            )
        initial_target_energy = float(payload["source"]["target_initial_energy"])
        directional_passes = list(payload["directional_passes"])
        next_directional_sweep = int(metadata["next_directional_sweep"])
        elapsed_before = float(payload["timing_seconds"]["total"])
        payload["protocol"]["maximum_directional_passes"] = maximum_directional_passes
        payload["timing_seconds"].setdefault("resume_setup_seconds", []).append(
            float(perf_counter() - invocation_start)
        )
    else:
        source_metadata = single._source_metadata(source_snapshot)
        if source_metadata["target_two_sz"] != 0:
            raise RuntimeError("source snapshot has the wrong target charge.")
        expected_local = np.asarray(((1, -1),) * NSITES)
        if not np.array_equal(source_metadata["local_two_sz"], expected_local):
            raise RuntimeError("source snapshot has different local charges.")
        state = single._projected_state(model, source_snapshot)
        if (
            abs(source_metadata["energy"] - source_checkpoint_energy)
            > CHECKPOINT_PAIR_ENERGY_TOL
        ):
            raise RuntimeError(
                "source JSON and snapshot record inconsistent energies."
            )
        initial_target_energy = float(state.energy)
        if (
            not target_differs
            and abs(initial_target_energy - source_checkpoint_energy)
            > RECONSTRUCTION_ENERGY_TOL
        ):
            raise RuntimeError(
                "source JSON, snapshot, and reconstructed energy are inconsistent: "
                f"recorded={source_checkpoint_energy:.16g}, "
                f"reconstructed={initial_target_energy:.16g}, "
                f"difference={initial_target_energy - source_checkpoint_energy:.6e}."
            )
        directional_passes = []
        next_directional_sweep = starting_directional_sweep
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
                "checkpoint_j2": source_j2,
                "checkpoint_energy": source_checkpoint_energy,
                "target_j2": target_j2,
                "target_initial_energy": initial_target_energy,
                "energy": initial_target_energy,
                "tensors_reused_at_new_hamiltonian": target_differs,
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
                "projector_bond_dims": list(state.projection.mpo_bond_dims),
                "maximum_projector_bond": int(state.projection.max_mpo_bond),
            },
            "contraction": {
                "frontier_backend": state.frontier_backend,
                "exact": bool(state.contraction_is_exact),
                "factorized_objective_mpo": bool(
                    state._hamiltonian_frontier.factorized_mpo
                ),
                "objective_mpo_materialized": False,
            },
            "directional_passes": directional_passes,
            "result": {},
            "timing_seconds": {
                "initial_setup": float(perf_counter() - invocation_start),
                "directional_passes": [],
                "total": 0.0,
            },
        }
        tie_edges = source_metadata["tie_edges"]
    if resumed:
        tie_edges = single._source_metadata(snapshot)["tie_edges"]

    tensor_shapes = tuple(tensor.shape for tensor in state.tensors)
    bond_dims = tuple(state.bond_dims)
    single._assert_full_parameterization(state, tensor_shapes, bond_dims)
    if len(directional_passes) >= maximum_directional_passes:
        print(
            "batch checkpoint already reached the directional-pass cap: "
            f"E={state.energy:.12f}",
            flush=True,
        )
        return payload
    if payload.get("result", {}).get("converged"):
        print(
            f"batch checkpoint is already converged: E={state.energy:.12f}",
            flush=True,
        )
        return payload

    reusable = None
    with (
        ThreadPoolExecutor(
            max_workers=options["frontier_workers"]
        ) as executor,
        ThreadPoolExecutor(max_workers=1) as pair_environment_executor,
    ):
        while len(directional_passes) < maximum_directional_passes:
            row, reusable = _run_directional_pass(
                state,
                next_directional_sweep,
                reusable,
                tensor_shapes=tensor_shapes,
                bond_dims=bond_dims,
                frontier_executor=executor,
                pair_environment_executor=pair_environment_executor,
                options=options,
            )
            directional_passes.append(row)
            next_directional_sweep += 1
            payload["directional_passes"] = directional_passes
            payload["timing_seconds"]["directional_passes"] = [
                float(record["seconds"]) for record in directional_passes
            ]
            payload["timing_seconds"]["total"] = float(
                elapsed_before + perf_counter() - invocation_start
            )
            payload["result"] = _result_record(
                state,
                source_energy=initial_target_energy,
                directional_passes=directional_passes,
                next_directional_sweep=next_directional_sweep,
                gain_tolerance=gain_tolerance,
                maximum_directional_passes=maximum_directional_passes,
                snapshot=snapshot,
            )
            if not row["accepted"]:
                payload["status"] = "endpoint_rejected"
            elif payload["result"]["converged"]:
                payload["status"] = "converged"
            elif len(directional_passes) >= maximum_directional_passes:
                payload["status"] = "maximum_passes"
            else:
                payload["status"] = "running"
            _checkpoint(
                output=output,
                snapshot=snapshot,
                payload=payload,
                state=state,
                tie_edges=tie_edges,
                source_checkpoint_id=source_checkpoint_id,
            )
            print(
                f"directional sweep {row['directional_sweep']:2d} "
                f"{row['direction']} E={row['energy']:.12f} "
                f"gain={row['energy_gain']:.3e} "
                f"fixed_reused={row['fixed_messages_reused']} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
            if payload["status"] != "running":
                break
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=DEFAULT_SOURCE_SNAPSHOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--j2",
        type=float,
        default=None,
        help=(
            "target J2/J1; when it differs from the source checkpoint, reuse "
            "only its tensors and recompute the projected starting energy"
        ),
    )
    parser.add_argument("--starting-directional-sweep", type=int, default=16)
    parser.add_argument("--maximum-directional-passes", type=int, default=16)
    parser.add_argument(
        "--gain-tolerance",
        type=float,
        default=1.0e-4,
        help="maximum energy gain per site in each direction of the last cycle",
    )
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
    parser.add_argument("--pair-dense-max-elements", type=int, default=2_000_000)
    parser.add_argument("--pair-operator-workers", type=int, default=4)
    parser.add_argument("--frontier-workers", type=int, default=2)
    parser.add_argument(
        "--factor-solver",
        choices=("auto", "dense", "matrix_free"),
        default="auto",
    )
    parser.add_argument("--merged-dense-fallback-dim", type=int, default=2048)
    parser.add_argument("--split-metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--split-metric-sweeps", type=int, default=2)
    parser.add_argument("--split-variational-sweeps", type=int, default=2)
    parser.add_argument(
        "--no-pair-environment-prefetch",
        action="store_true",
        help="disable one-pair fixed-side environment look-ahead",
    )
    parser.add_argument(
        "--keep-redundant-full-rank-davidson",
        action="store_true",
    )
    parser.add_argument(
        "--full-rank-davidson-min-dimension",
        type=int,
        default=256,
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(
        source_result=args.source_result,
        source_snapshot=args.source_snapshot,
        output=args.output,
        snapshot=args.snapshot,
        j2=args.j2,
        starting_directional_sweep=args.starting_directional_sweep,
        maximum_directional_passes=args.maximum_directional_passes,
        gain_tolerance=args.gain_tolerance,
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
        pair_environment_prefetch=(
            not args.no_pair_environment_prefetch
        ),
        skip_redundant_full_rank_davidson=(not args.keep_redundant_full_rank_davidson),
        redundant_full_rank_davidson_min_dimension=(
            args.full_rank_davidson_min_dimension
        ),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
