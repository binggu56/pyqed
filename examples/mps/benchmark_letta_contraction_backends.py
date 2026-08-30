#!/usr/bin/env python3
"""Benchmark exact, block, TT, and VMC contractions of identical LETTA states.

The default ``smoke`` profile uses the saved 4x4, D=4, seed-7 state and small
TT/VMC settings.  The ``full`` profile additionally benchmarks the saved 8x4
state and a reproducibly generated 6x6, D=2 MPS-lifted state.  No dense state
vector, dense many-body Hamiltonian, or full configuration table is built.

Examples
--------
Fast integration check::

    PYTHONPATH=. python examples/mps/benchmark_letta_contraction_backends.py

All requested geometries::

    PYTHONPATH=. python examples/mps/benchmark_letta_contraction_backends.py \
        --profile full

Resume-friendly custom run (the JSON is rewritten after every backend)::

    PYTHONPATH=. python examples/mps/benchmark_letta_contraction_backends.py \
        --profile custom --geometries 4x4,8x4,6x6 --tt-ranks 4,8,16 \
        --tt-transfer-ranks 4,8 \
        --vmc-samples 256 --output /private/tmp/letta-contract.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA, frontier_tensors_from_mps
from pyqed.letta.tt_frontier import TTMPOFrontier
from pyqed.letta.vmc import VMC
from pyqed.tn import MPO


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SNAPSHOTS = {
    "4x4": RESULTS / "frontier_letta_j2_half_convergence_4x4_seed7.npz",
    "8x4": RESULTS / "frontier_letta_j2_half_convergence_8x4_seed7.npz",
}

PROFILE_DEFAULTS = {
    "smoke": {
        "geometries": ("4x4",),
        "tt_ranks": (2, 4),
        "tt_transfer_ranks": (4,),
        "vmc_samples": 24,
        "vmc_burn_in": 4,
        "vmc_sweeps_between": 1,
        "vmc_proposal": "mixed",
        "vmc_exchange_probability": 0.9,
        "repeats": 1,
    },
    "quick": {
        "geometries": ("4x4", "8x4", "6x6"),
        "tt_ranks": (4, 8),
        "tt_transfer_ranks": (4, 8),
        "vmc_samples": 64,
        "vmc_burn_in": 10,
        "vmc_sweeps_between": 1,
        "vmc_proposal": "mixed",
        "vmc_exchange_probability": 0.9,
        "repeats": 1,
    },
    "full": {
        "geometries": ("4x4", "8x4", "6x6"),
        "tt_ranks": (4, 8, 16, 32),
        "tt_transfer_ranks": (4, 8, 16),
        "vmc_samples": 256,
        "vmc_burn_in": 50,
        "vmc_sweeps_between": 2,
        "vmc_proposal": "mixed",
        "vmc_exchange_probability": 0.9,
        "repeats": 2,
    },
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    nrows: int
    ncols: int
    bond_dim: int
    hamiltonian: object
    parent_sets: tuple[tuple[int, ...], ...]
    tensors: tuple[np.ndarray, ...]
    tensor_source: dict

    @property
    def nsites(self):
        return self.nrows * self.ncols

    @property
    def dims(self):
        return self.hamiltonian.dims

    @property
    def physical_groups(self):
        return tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )


def _csv_strings(value):
    if value is None:
        return None
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    return result


def _csv_positive_ints(value):
    try:
        result = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("TT ranks must be positive")
    return result


def _portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(HERE.parents[1]))
    except ValueError:
        return str(path)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _timed(function, *, repeats=1):
    """Return the final value and repeat timing/allocation diagnostics."""

    seconds = []
    peaks = []
    value = None
    for _ in range(int(repeats)):
        gc.collect()
        tracemalloc.start()
        start = perf_counter()
        try:
            value = function()
        finally:
            elapsed = perf_counter() - start
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        seconds.append(float(elapsed))
        peaks.append(int(peak))
    return value, {
        "repeats": int(repeats),
        "seconds": seconds,
        "median_seconds": float(statistics.median(seconds)),
        "minimum_seconds": float(min(seconds)),
        "maximum_seconds": float(max(seconds)),
        "maximum_tracemalloc_bytes": int(max(peaks)),
        "memory_note": (
            "tracemalloc measures traced allocations during the call; use the "
            "structural element counts for backend-independent memory comparisons"
        ),
    }


def _scalar(value):
    value = np.asarray(value).reshape(()).item()
    return {
        "real": float(np.real(value)),
        "imaginary": float(np.imag(value)),
    }


def _real(value):
    value = np.asarray(value).reshape(()).item()
    if abs(np.imag(value)) > 5.0e-10 * max(1.0, abs(value)):
        raise ValueError(f"expected a real scalar, received {value!r}")
    return float(np.real(value))


def _random_left_canonical_mps(nsites, bond_dim, *, seed):
    """Generate a normalized real OBC MPS without a dense state vector."""

    nsites = int(nsites)
    bond_dim = int(bond_dim)
    rng = np.random.default_rng(seed)
    bonds = (1,) + (bond_dim,) * max(0, nsites - 1) + (1,)
    tensors = [
        rng.normal(size=(bonds[site], 2, bonds[site + 1]))
        / np.sqrt(2 * bonds[site])
        for site in range(nsites)
    ]
    for site in range(nsites - 1):
        left, physical, right = tensors[site].shape
        q_factor, transfer = np.linalg.qr(
            tensors[site].reshape(left * physical, right), mode="reduced"
        )
        rank = q_factor.shape[1]
        tensors[site] = q_factor.reshape(left, physical, rank)
        tensors[site + 1] = np.tensordot(
            transfer, tensors[site + 1], axes=(1, 0)
        )
    final_norm = float(np.linalg.norm(tensors[-1]))
    if not np.isfinite(final_norm) or final_norm <= 0.0:
        raise ValueError("generated MPS has invalid norm")
    tensors[-1] /= final_norm
    return tuple(tensors)


def _expected_tensor_shapes(dims, parent_sets, bond_dim):
    nsites = len(dims)
    bonds = (1,) + (int(bond_dim),) * max(0, nsites - 1) + (1,)
    return tuple(
        (bonds[site], bonds[site + 1], dims[site])
        + tuple(dims[parent] for parent in parents)
        for site, parents in enumerate(parent_sets)
    )


def _load_snapshot(path, expected_shapes):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        expected_keys = [f"tensor_{site:03d}" for site in range(len(expected_shapes))]
        if keys != expected_keys:
            raise ValueError(
                f"{path} has tensor keys inconsistent with {len(expected_shapes)} sites"
            )
        tensors = tuple(np.array(archive[key], copy=True) for key in expected_keys)
    for site, (tensor, shape) in enumerate(zip(tensors, expected_shapes)):
        if tensor.shape != shape:
            raise ValueError(
                f"snapshot tensor {site} has shape {tensor.shape}, expected {shape}"
            )
        if np.any(~np.isfinite(tensor)):
            raise ValueError(f"snapshot tensor {site} contains nonfinite values")
    return tensors


def _case(geometry, args):
    try:
        nrows, ncols = (int(value) for value in geometry.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid geometry {geometry!r}; expected ROWSxCOLS") from error
    if nrows < 1 or ncols < 1:
        raise ValueError("geometry dimensions must be positive")
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(args.j2)) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    snapshot = {
        "4x4": Path(args.snapshot_4x4),
        "8x4": Path(args.snapshot_8x4),
    }.get(geometry)
    use_snapshot = (
        snapshot is not None and snapshot.is_file() and not args.ignore_snapshots
    )
    bond_dim = 4 if use_snapshot else int(args.generated_bond_dim)
    expected_shapes = _expected_tensor_shapes(
        hamiltonian.dims, parent_sets, bond_dim
    )

    if use_snapshot:
        tensors = _load_snapshot(snapshot, expected_shapes)
        source = {
            "kind": "saved-optimized-letta",
            "path": _portable_path(snapshot),
            "seed": 7,
        }
    else:
        mps_tensors = _random_left_canonical_mps(
            nsites, bond_dim, seed=args.seed
        )
        tensors = frontier_tensors_from_mps(
            mps_tensors,
            parent_sets,
            bond_dim=bond_dim,
            tie_noise=args.tie_noise,
            seed=args.seed + 1,
        )
        source = {
            "kind": "generated-mps-lift-plus-tie-noise",
            "seed": int(args.seed),
            "tie_noise": float(args.tie_noise),
            "snapshot_requested": (
                _portable_path(snapshot) if snapshot is not None else None
            ),
            "snapshot_missing_or_ignored": snapshot is not None,
        }
    return BenchmarkCase(
        name=geometry,
        nrows=nrows,
        ncols=ncols,
        bond_dim=bond_dim,
        hamiltonian=hamiltonian,
        parent_sets=parent_sets,
        tensors=tuple(np.array(tensor, copy=True) for tensor in tensors),
        tensor_source=source,
    )


def _frontier_memory(engine):
    message_elements = [
        int(engine.message_elements(cut)) for cut in range(engine.nsites + 1)
    ]
    dense_message_elements = [
        int(
            engine.dense_message_elements(cut)
            if hasattr(engine, "dense_message_elements")
            else engine.message_elements(cut)
        )
        for cut in range(engine.nsites + 1)
    ]
    return {
        "representation": type(engine).__name__,
        "message_elements_by_cut": message_elements,
        "peak_message_elements": int(max(message_elements)),
        "total_message_elements": int(sum(message_elements)),
        "dense_message_elements_by_cut": dense_message_elements,
        "dense_peak_message_elements": int(max(dense_message_elements)),
        "dense_total_message_elements": int(sum(dense_message_elements)),
    }


def _benchmark_exact(case, backend, *, repeats):
    tensors = [np.array(tensor, copy=True) for tensor in case.tensors]

    def construct():
        return FrontierTiedLETTA(
            case.hamiltonian,
            case.dims,
            case.parent_sets,
            bond_dim=case.bond_dim,
            tensors=tensors,
            frontier_backend=backend,
            path_optimizer="greedy",
        )

    state, setup = _timed(construct)
    # Constructors balance gauges independently.  Reset the raw tensors so
    # every backend contracts exactly the same numerical arrays.
    state.tensors = [np.array(tensor, copy=True) for tensor in case.tensors]

    norm, norm_timing = _timed(state.norm, repeats=repeats)
    numerator, numerator_timing = _timed(
        lambda: state._hamiltonian_frontier.scalar(state.tensors),
        repeats=repeats,
    )
    energy, energy_timing = _timed(state.expectation, repeats=repeats)
    dtype = np.result_type(*(tensor.dtype for tensor in case.tensors))
    itemsize = int(np.dtype(dtype).itemsize)
    norm_memory = _frontier_memory(state._norm_frontier)
    hamiltonian_memory = _frontier_memory(state._hamiltonian_frontier)
    cached_elements = norm_memory["total_message_elements"] + hamiltonian_memory[
        "total_message_elements"
    ]
    peak_elements = max(
        norm_memory["peak_message_elements"],
        hamiltonian_memory["peak_message_elements"],
    )
    return state, {
        "backend": backend,
        "setup": setup,
        "norm": {"value": _scalar(norm), "timing": norm_timing},
        "hamiltonian_numerator": {
            "value": _scalar(numerator),
            "timing": numerator_timing,
        },
        "energy": {"value": float(energy), "timing": energy_timing},
        "ratio_energy": float(_real(numerator) / _real(norm)),
        "contraction_plans": int(state.contraction_plans),
        "hamiltonian_mpo_bond_dims": {
            "uncompressed_maximum": int(state.uncompressed_hamiltonian_mpo_bond_dim),
            "compressed_maximum": int(state.compressed_hamiltonian_mpo_bond_dim),
            "contracted_maximum": int(max(state.hamiltonian_mpo.bond_dims)),
        },
        "memory": {
            "dtype_itemsize": itemsize,
            "peak_frontier_elements": int(peak_elements),
            "peak_frontier_bytes": int(peak_elements * itemsize),
            "cached_environment_elements": int(cached_elements),
            "cached_environment_bytes": int(cached_elements * itemsize),
            "norm": norm_memory,
            "hamiltonian": hamiltonian_memory,
        },
    }


def _tt_diagnostics(diagnostics, *, itemsize):
    advances = tuple(diagnostics.advances)
    maximum_rank = max(
        (
            max(advance.target_ranks, default=1)
            for advance in advances
        ),
        default=1,
    )
    result = {
        "advances": len(advances),
        "total_discarded_weight": float(diagnostics.total_discarded_weight),
        "max_relative_discarded_weight": float(
            diagnostics.max_relative_discarded_weight
        ),
        "maximum_observed_rank": int(maximum_rank),
        "peak_message_storage_elements": int(
            diagnostics.peak_message_storage_elements
        ),
        "peak_message_storage_bytes": int(
            diagnostics.peak_message_storage_elements * itemsize
        ),
        "peak_dense_message_elements": int(
            diagnostics.peak_dense_message_elements
        ),
        "peak_dense_message_bytes": int(
            diagnostics.peak_dense_message_elements * itemsize
        ),
        "peak_product_storage_elements": int(
            diagnostics.peak_product_storage_elements
        ),
        "peak_product_storage_bytes": int(
            diagnostics.peak_product_storage_elements * itemsize
        ),
        "peak_local_factor_elements": int(diagnostics.peak_local_factor_elements),
        "dense_frontier_absorptions": int(
            diagnostics.dense_frontier_absorptions
        ),
        "total_local_factor_discarded_weight": float(
            sum(advance.local_factor_discarded_weight for advance in advances)
        ),
        "total_message_discarded_weight": float(
            sum(advance.message_discarded_weight for advance in advances)
        ),
        "peak_local_factor_storage_elements": int(
            max(
                (advance.local_factor_storage_elements for advance in advances),
                default=0,
            )
        ),
        "maximum_local_factor_rank": int(
            max(
                (
                    max(advance.local_factor_ranks, default=1)
                    for advance in advances
                ),
                default=1,
            )
        ),
    }
    # The transfer-compression branch adds these aggregate fields.  Keeping
    # this conditional makes archived benchmark JSON readable across that API
    # addition without changing the numerical protocol.
    for name in (
        "total_local_factor_discarded_weight",
        "max_local_factor_relative_discarded_weight",
        "peak_local_factor_storage_elements",
    ):
        if hasattr(diagnostics, name):
            result[name] = (
                int(getattr(diagnostics, name))
                if name.endswith("storage_elements")
                else float(getattr(diagnostics, name))
            )
    return result


def _tt_engine(case, mpo, *, paired_sites, max_rank, transfer_max_rank, args):
    return TTMPOFrontier(
        case.dims,
        case.sites,
        [tensor.shape for tensor in case.tensors],
        mpo.tensors,
        paired_sites=paired_sites,
        max_rank=int(max_rank),
        rtol=float(args.tt_rtol),
        atol=float(args.tt_atol),
        transfer_max_rank=int(transfer_max_rank),
        transfer_rtol=float(args.tt_transfer_rtol),
        transfer_atol=float(args.tt_transfer_atol),
        absorption="structured",
        optimize="greedy",
    )


def _benchmark_tt(
    case,
    _compressed_state,
    reference,
    *,
    ranks,
    transfer_ranks,
    repeats,
    args,
):
    identity = MPO(
        tuple(
            np.eye(dim, dtype=np.result_type(*[t.dtype for t in case.tensors]))[
                None, None, :, :
            ]
            for dim in case.dims
        ),
    )
    hamiltonian_mpo = (
        _compressed_state.hamiltonian_mpo
        if _compressed_state is not None
        else case.hamiltonian.to_mpo().compress()
    )
    itemsize = int(np.dtype(np.result_type(*[t.dtype for t in case.tensors])).itemsize)
    rows = []
    for transfer_rank, rank in (
        (transfer_rank, rank)
        for transfer_rank in transfer_ranks
        for rank in ranks
    ):
        print(
            f"    TT boundary rank {rank}, transfer rank {transfer_rank}",
            flush=True,
        )
        (norm_engine, hamiltonian_engine), setup = _timed(
            lambda: (
                _tt_engine(
                    case,
                    identity,
                    paired_sites=(),
                    max_rank=rank,
                    transfer_max_rank=transfer_rank,
                    args=args,
                ),
                _tt_engine(
                    case,
                    hamiltonian_mpo,
                    paired_sites=None,
                    max_rank=rank,
                    transfer_max_rank=transfer_rank,
                    args=args,
                ),
            )
        )
        norm, norm_timing = _timed(
            lambda: norm_engine.scalar(case.tensors), repeats=repeats
        )
        norm_diagnostics = _tt_diagnostics(
            norm_engine.diagnostics, itemsize=itemsize
        )
        numerator, numerator_timing = _timed(
            lambda: hamiltonian_engine.scalar(case.tensors), repeats=repeats
        )
        hamiltonian_diagnostics = _tt_diagnostics(
            hamiltonian_engine.diagnostics, itemsize=itemsize
        )
        approximate_energy = numerator / norm
        norm_scale = max(1.0, abs(norm))
        constructor_normalization_compatible = bool(
            np.real(norm) > 0.0 and abs(np.imag(norm)) <= 5.0e-10 * norm_scale
        )
        reference_norm = None if reference is None else reference["norm"]
        reference_energy = None if reference is None else reference["energy"]
        hybrid_ratio = (
            None if reference_norm is None else numerator / reference_norm
        )
        exact_norm_peak = (
            None
            if _compressed_state is None
            else int(_compressed_state._norm_frontier.peak_message_elements)
        )
        peak_storage = max(
            norm_diagnostics["peak_message_storage_elements"],
            hamiltonian_diagnostics["peak_message_storage_elements"],
        )
        peak_dense = max(
            norm_diagnostics["peak_dense_message_elements"],
            hamiltonian_diagnostics["peak_dense_message_elements"],
        )
        rows.append(
            {
                "maximum_rank": int(rank),
                "transfer_maximum_rank": int(transfer_rank),
                "setup": setup,
                "interface": "standalone TTMPOFrontier",
                "integration_note": (
                    "FrontierTiedLETTA(frontier_backend='tensor_train') uses "
                    "an exact norm frontier and this TT Hamiltonian engine by "
                    "default. The separately reported all-TT norm is an "
                    "experimental diagnostic."
                ),
                "contractor_types": {
                    "norm": type(norm_engine).__name__,
                    "hamiltonian": type(hamiltonian_engine).__name__,
                },
                "contraction_is_exact": False,
                "all_tt_norm_is_positive_real": (
                    constructor_normalization_compatible
                ),
                "norm": {
                    "value": _scalar(norm),
                    "timing": norm_timing,
                    "absolute_error": (
                        None
                        if reference_norm is None
                        else float(abs(norm - reference_norm))
                    ),
                    "relative_error": (
                        None
                        if reference_norm is None
                        else float(
                            abs(norm - reference_norm)
                            / max(abs(reference_norm), np.finfo(float).tiny)
                        )
                    ),
                    "diagnostics": norm_diagnostics,
                },
                "hamiltonian_numerator": {
                    "value": _scalar(numerator),
                    "timing": numerator_timing,
                    "diagnostics": hamiltonian_diagnostics,
                },
                "energy": _scalar(approximate_energy),
                "absolute_energy_error": (
                    None
                    if reference_energy is None
                    else float(abs(approximate_energy - reference_energy))
                ),
                "absolute_energy_error_per_site": (
                    None
                    if reference_energy is None
                    else float(
                        abs(approximate_energy - reference_energy) / case.nsites
                    )
                ),
                "hybrid_exact_norm_tt_hamiltonian": {
                    "energy": (
                        None if hybrid_ratio is None else float(np.real(hybrid_ratio))
                    ),
                    "imaginary_residue": (
                        None if hybrid_ratio is None else float(np.imag(hybrid_ratio))
                    ),
                    "absolute_energy_error": (
                        None
                        if hybrid_ratio is None or reference_energy is None
                        else float(abs(np.real(hybrid_ratio) - reference_energy))
                    ),
                    "exact_norm_peak_elements": exact_norm_peak,
                    "peak_stored_message_elements": (
                        None
                        if exact_norm_peak is None
                        else int(
                            max(
                                exact_norm_peak,
                                hamiltonian_diagnostics[
                                    "peak_message_storage_elements"
                                ],
                            )
                        )
                    ),
                    "is_default_integrated_architecture": True,
                },
                "memory": {
                    "dtype_itemsize": itemsize,
                    "peak_message_storage_elements": int(peak_storage),
                    "peak_message_storage_bytes": int(peak_storage * itemsize),
                    "peak_transient_product_elements": int(
                        max(
                            norm_diagnostics["peak_product_storage_elements"],
                            hamiltonian_diagnostics[
                                "peak_product_storage_elements"
                            ],
                        )
                    ),
                    "peak_transient_product_bytes": int(
                        max(
                            norm_diagnostics["peak_product_storage_elements"],
                            hamiltonian_diagnostics[
                                "peak_product_storage_elements"
                            ],
                        )
                        * itemsize
                    ),
                    "storage_note": (
                        "message storage is resident boundary-TT storage; the "
                        "transient product is reported separately and can be larger"
                    ),
                    "corresponding_dense_peak_elements": int(peak_dense),
                    "corresponding_dense_peak_bytes": int(peak_dense * itemsize),
                    "dense_to_tt_peak_ratio": float(
                        peak_dense / max(peak_storage, 1)
                    ),
                },
            }
        )
    return rows


def _benchmark_vmc(case, *, reference_energy, args):
    state_view = SimpleNamespace(
        tensors=case.tensors,
        physical_groups=case.physical_groups,
        dims=case.dims,
        hamiltonian=case.hamiltonian,
    )
    vmc, setup = _timed(
        lambda: VMC(
            state_view,
            seed=args.seed + 17,
            proposal=args.vmc_proposal,
            exchange_probability=args.vmc_exchange_probability,
        )
    )
    estimate, timing = _timed(
        lambda: vmc.estimate(
            args.vmc_samples,
            burn_in=args.vmc_burn_in,
            sweeps_between=args.vmc_sweeps_between,
        )
    )
    diagnostics = estimate.diagnostics
    warnings = []
    if diagnostics.acceptance_rate < 0.01:
        warnings.append(
            "overall Metropolis acceptance is below 1%; increase thinning or "
            "change the proposal mixture"
        )
    if args.vmc_proposal == "exchange":
        warnings.append(
            "exchange-only sampling conserves the physical-label histogram "
            "and is valid only when the target sector is intentional"
        )
    if (
        args.vmc_proposal == "mixed"
        and diagnostics.single_site_attempts
        and diagnostics.single_site_acceptance_rate < 0.01
    ):
        warnings.append(
            "single-site acceptance is below 1%, so mixing between conserved "
            "label sectors can remain slow even when exchanges mix within a sector"
        )
    if (
        args.vmc_proposal == "mixed"
        and args.vmc_exchange_probability >= 0.8
        and diagnostics.single_site_acceptance_rate >= 0.05
    ):
        warnings.append(
            "single-site moves are viable but rare in this proposal mixture; "
            "reduce the exchange probability or compare independent chains to "
            "check mixing between physical-label sectors"
        )
    return {
        "samples": int(estimate.nsamples),
        "proposal": args.vmc_proposal,
        "exchange_probability": float(args.vmc_exchange_probability),
        "burn_in_sweeps": int(args.vmc_burn_in),
        "sweeps_between_samples": int(args.vmc_sweeps_between),
        "setup": setup,
        "timing": timing,
        "energy": _scalar(estimate.energy),
        "variance": float(estimate.variance),
        "real_variance": float(estimate.real_variance),
        "naive_standard_error": float(estimate.standard_error),
        "autocorrelation_standard_error": float(
            estimate.autocorrelation_standard_error
        ),
        "integrated_autocorrelation_time": float(
            estimate.integrated_autocorrelation_time
        ),
        "effective_sample_size": float(estimate.effective_sample_size),
        "standard_error_note": (
            "both errors use the real local-energy variance; the corrected "
            "estimate uses Geyer's initial positive sequence"
        ),
        "absolute_energy_error": (
            None
            if reference_energy is None
            else float(abs(estimate.energy.real - reference_energy))
        ),
        "diagnostics": {
            "attempts": int(diagnostics.attempts),
            "accepted": int(diagnostics.accepted),
            "acceptance_rate": float(diagnostics.acceptance_rate),
            "zero_amplitude_rejections": int(
                diagnostics.zero_amplitude_rejections
            ),
            "initialization_attempts": int(diagnostics.initialization_attempts),
            "site_attempts": list(diagnostics.site_attempts),
            "site_accepts": list(diagnostics.site_accepts),
            "single_site_attempts": int(diagnostics.single_site_attempts),
            "single_site_accepts": int(diagnostics.single_site_accepts),
            "single_site_acceptance_rate": float(
                diagnostics.single_site_acceptance_rate
            ),
            "exchange_attempts": int(diagnostics.exchange_attempts),
            "exchange_accepts": int(diagnostics.exchange_accepts),
            "exchange_acceptance_rate": float(
                diagnostics.exchange_acceptance_rate
            ),
        },
        "warnings": warnings,
    }


def _case_metadata(case):
    dtype = np.dtype(np.result_type(*[tensor.dtype for tensor in case.tensors]))
    tensor_elements = int(sum(tensor.size for tensor in case.tensors))
    nearest, diagonals = square_j1_j2_bonds(case.nrows, case.ncols)
    return {
        "geometry": case.name,
        "nrows": int(case.nrows),
        "ncols": int(case.ncols),
        "nsites": int(case.nsites),
        "boundary": "open",
        "site_order": "row-wise snake",
        "j1": 1.0,
        "j2": None,
        "bond_dim": int(case.bond_dim),
        "physical_dim": 2,
        "tie_graph": "all nearest-neighbor J1 bonds",
        "tie_edges": len(nearest),
        "j2_diagonal_edges": len(diagonals),
        "parameters": tensor_elements,
        "tensor_storage_bytes": int(tensor_elements * dtype.itemsize),
        "tensor_dtype": str(dtype),
        "tensor_source": case.tensor_source,
        "full_hilbert_dimension": int(2**case.nsites),
        "full_hilbert_materialized": False,
    }


def _record_error(target, backend, error):
    target[backend] = {
        "status": "failed",
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def run(args):
    output = Path(args.output)
    payload = {
        "schema": "letta-contraction-backends-v1",
        "status": "running",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "profile": args.profile,
            "geometries": list(args.geometries),
            "j1": 1.0,
            "j2": float(args.j2),
            "seed": int(args.seed),
            "generated_bond_dim": int(args.generated_bond_dim),
            "generated_tie_noise": float(args.tie_noise),
            "repeats": int(args.repeats),
            "tt_ranks": list(args.tt_ranks),
            "tt_rtol": float(args.tt_rtol),
            "tt_atol": float(args.tt_atol),
            "tt_transfer_ranks": list(args.tt_transfer_ranks),
            "tt_transfer_rtol": float(args.tt_transfer_rtol),
            "tt_transfer_atol": float(args.tt_transfer_atol),
            "integrated_tt_norm_backend": "exact",
            "integrated_tt_hermitize_local_actions": True,
            "vmc_samples": int(args.vmc_samples),
            "vmc_burn_in": int(args.vmc_burn_in),
            "vmc_sweeps_between": int(args.vmc_sweeps_between),
            "vmc_proposal": args.vmc_proposal,
            "vmc_exchange_probability": float(args.vmc_exchange_probability),
            "skip_exact": bool(args.skip_exact),
            "skip_block": bool(args.skip_block),
            "skip_tt": bool(args.skip_tt),
            "skip_vmc": bool(args.skip_vmc),
            "no_dense_many_body_objects": True,
        },
        "cases": [],
    }
    _write_json(output, payload)

    for geometry in args.geometries:
        print(f"[{geometry}] preparing identical tensors", flush=True)
        case = _case(geometry, args)
        record = {
            "model": _case_metadata(case),
            "backends": {},
        }
        record["model"]["j2"] = float(args.j2)
        payload["cases"].append(record)
        _write_json(output, payload)

        compressed_state = None
        reference = None
        if not args.skip_exact:
            print(f"[{geometry}] exact compressed frontier", flush=True)
            try:
                compressed_state, compressed = _benchmark_exact(
                    case, "compressed", repeats=args.repeats
                )
                record["backends"]["exact_compressed"] = compressed
                reference = {
                    "norm": compressed["norm"]["value"]["real"],
                    "energy": compressed["energy"]["value"],
                }
            except Exception as error:
                _record_error(record["backends"], "exact_compressed", error)
                if args.fail_fast:
                    raise
            _write_json(output, payload)

        if not args.skip_block:
            print(f"[{geometry}] exact identity-block frontier", flush=True)
            try:
                _block_state, block = _benchmark_exact(
                    case, "identity_block", repeats=args.repeats
                )
                if reference is not None:
                    block["absolute_energy_difference_from_compressed"] = float(
                        abs(block["energy"]["value"] - reference["energy"])
                    )
                    block["absolute_norm_difference_from_compressed"] = float(
                        abs(block["norm"]["value"]["real"] - reference["norm"])
                    )
                record["backends"]["exact_identity_block"] = block
            except Exception as error:
                _record_error(record["backends"], "exact_identity_block", error)
                if args.fail_fast:
                    raise
            _write_json(output, payload)

        if not args.skip_tt:
            print(f"[{geometry}] TT frontier rank sweep", flush=True)
            try:
                record["backends"]["tt_frontier"] = {
                    "absorption": "structured",
                    "rows": _benchmark_tt(
                        case,
                        compressed_state,
                        reference,
                        ranks=args.tt_ranks,
                        transfer_ranks=args.tt_transfer_ranks,
                        repeats=args.repeats,
                        args=args,
                    ),
                }
            except Exception as error:
                _record_error(record["backends"], "tt_frontier", error)
                if args.fail_fast:
                    raise
            _write_json(output, payload)

        if not args.skip_vmc:
            print(f"[{geometry}] VMC energy estimate", flush=True)
            try:
                record["backends"]["vmc"] = _benchmark_vmc(
                    case,
                    reference_energy=(
                        None if reference is None else reference["energy"]
                    ),
                    args=args,
                )
            except Exception as error:
                _record_error(record["backends"], "vmc", error)
                if args.fail_fast:
                    raise
            _write_json(output, payload)

    payload["status"] = "complete"
    _write_json(output, payload)
    print(f"wrote {output}", flush=True)
    return payload


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("smoke", "quick", "full", "custom"), default="smoke"
    )
    parser.add_argument(
        "--geometries",
        type=_csv_strings,
        help="comma-separated ROWSxCOLS list; profile default when omitted",
    )
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--generated-bond-dim", type=int, default=2)
    parser.add_argument("--tie-noise", type=float, default=0.02)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--tt-ranks", type=_csv_positive_ints)
    parser.add_argument("--tt-rtol", type=float, default=0.0)
    parser.add_argument("--tt-atol", type=float, default=0.0)
    parser.add_argument(
        "--tt-transfer-ranks",
        type=_csv_positive_ints,
        help="comma-separated site-transfer TT ranks; profile default when omitted",
    )
    parser.add_argument("--tt-transfer-rtol", type=float, default=0.0)
    parser.add_argument("--tt-transfer-atol", type=float, default=0.0)
    parser.add_argument("--vmc-samples", type=int)
    parser.add_argument("--vmc-burn-in", type=int)
    parser.add_argument("--vmc-sweeps-between", type=int)
    parser.add_argument(
        "--vmc-proposal",
        choices=("single_site", "exchange", "heat_bath", "mixed"),
    )
    parser.add_argument("--vmc-exchange-probability", type=float)
    parser.add_argument("--snapshot-4x4", type=Path, default=DEFAULT_SNAPSHOTS["4x4"])
    parser.add_argument("--snapshot-8x4", type=Path, default=DEFAULT_SNAPSHOTS["8x4"])
    parser.add_argument("--ignore-snapshots", action="store_true")
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-block", action="store_true")
    parser.add_argument("--skip-tt", action="store_true")
    parser.add_argument("--skip-vmc", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output; defaults to results/letta_contraction_backends_PROFILE.json",
    )
    return parser


def _resolved_args(argv=None):
    args = _parser().parse_args(argv)
    if args.profile == "custom":
        defaults = PROFILE_DEFAULTS["smoke"]
    else:
        defaults = PROFILE_DEFAULTS[args.profile]
    for name in (
        "geometries",
        "tt_ranks",
        "tt_transfer_ranks",
        "vmc_samples",
        "vmc_burn_in",
        "vmc_sweeps_between",
        "vmc_proposal",
        "vmc_exchange_probability",
        "repeats",
    ):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    if args.output is None:
        args.output = RESULTS / f"letta_contraction_backends_{args.profile}.json"
    if args.generated_bond_dim < 1:
        raise SystemExit("--generated-bond-dim must be positive")
    if args.tie_noise < 0.0 or not np.isfinite(args.tie_noise):
        raise SystemExit("--tie-noise must be finite and nonnegative")
    if not np.isfinite(args.j2):
        raise SystemExit("--j2 must be finite")
    if args.repeats < 1 or args.vmc_samples < 1:
        raise SystemExit("--repeats and --vmc-samples must be positive")
    if args.vmc_burn_in < 0 or args.vmc_sweeps_between < 1:
        raise SystemExit("invalid VMC burn-in or sampling interval")
    if not 0.0 <= args.vmc_exchange_probability <= 1.0:
        raise SystemExit("--vmc-exchange-probability must lie in [0, 1]")
    if min(
        args.tt_rtol,
        args.tt_atol,
        args.tt_transfer_rtol,
        args.tt_transfer_atol,
    ) < 0.0:
        raise SystemExit("TT tolerances must be nonnegative")
    return args


def main(argv=None):
    return run(_resolved_args(argv))


if __name__ == "__main__":
    main()
