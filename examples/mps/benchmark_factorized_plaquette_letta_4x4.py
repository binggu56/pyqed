#!/usr/bin/env python3
"""Internally factorized 2x2-plaquette LETTA for the open 4x4 J1-J2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh

from examples.mps.benchmark_full_plaquette_letta_4x4 import (
    DEFAULT_OUTPUT as FULL_OUTPUT,
    plaquette_hamiltonian,
)
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from pyqed.letta import FrontierTiedLETTA


HERE = Path(__file__).resolve().parent
DEFAULT_INITIAL = FULL_OUTPUT.with_name(
    "full_plaquette_letta_4x4_whitened_hybrid_natural.npz"
)
DEFAULT_OUTPUT = HERE / "results" / "factorized_plaquette_letta_4x4.json"
PARENT_SETS = ((3,), (), (), ())


def _tensor_train_array(tensor, *, tied):
    tensor = np.asarray(tensor)
    left, right = tensor.shape[:2]
    if tied:
        if tensor.shape[2:] != (16, 16):
            raise ValueError("the tied plaquette tensor must have shape (*, *, 16, 16).")
        expanded = tensor.reshape(left, right, *(2,) * 8)
        expanded = expanded.transpose(
            0, 2, 6, 3, 7, 4, 8, 5, 9, 1
        )
        return expanded.reshape(left, 4, 4, 4, 4, right)
    if tensor.shape[2:] != (16,):
        raise ValueError("a plaquette tensor must have shape (*, *, 16).")
    return tensor.reshape(left, right, *(2,) * 4).transpose(0, 2, 3, 4, 5, 1)


def factorize_plaquette_tensor(tensor, *, rank, tied):
    """Return a four-core TT-SVD of one dense macro-plaquette tensor."""
    array = _tensor_train_array(tensor, tied=tied)
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be positive.")
    cores = []
    remainder = array
    left_rank = array.shape[0]
    for physical in array.shape[1:4]:
        matrix = remainder.reshape(left_rank * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        kept = min(rank, singular.size)
        cores.append(u[:, :kept].reshape(left_rank, physical, kept))
        remainder = singular[:kept, None] * vh[:kept]
        left_rank = kept
    cores.append(remainder.reshape(left_rank, array.shape[4], array.shape[5]))
    return cores


def materialize_plaquette_tensor(cores, *, tied):
    """Contract four internal cores back to the macro tensor convention."""
    if len(cores) != 4:
        raise ValueError("a factorized plaquette must contain four cores.")
    array = np.asarray(cores[0])
    for core in cores[1:]:
        array = np.tensordot(array, np.asarray(core), axes=([-1], [0]))
    left, *physical, right = array.shape
    if tied:
        if physical != [4, 4, 4, 4]:
            raise ValueError("tied plaquette cores must have local dimension four.")
        expanded = array.reshape(left, 2, 2, 2, 2, 2, 2, 2, 2, right)
        expanded = expanded.transpose(0, 9, 1, 3, 5, 7, 2, 4, 6, 8)
        return expanded.reshape(left, right, 16, 16)
    if physical != [2, 2, 2, 2]:
        raise ValueError("plaquette cores must have local dimension two.")
    return array.transpose(0, 5, 1, 2, 3, 4).reshape(left, right, 16)


def _core_map(cores, core_index, *, tied):
    """Return the linear map from one flattened core to the dense tensor."""
    core = np.asarray(cores[core_index])
    columns = []
    for element in range(core.size):
        basis = np.zeros(core.size, dtype=complex)
        basis[element] = 1.0
        factors = list(cores)
        factors[core_index] = basis.reshape(core.shape)
        columns.append(
            materialize_plaquette_tensor(factors, tied=tied).reshape(-1)
        )
    return np.column_stack(columns)


def _lowest_generalized_vector(hamiltonian, metric, *, metric_tol):
    metric = 0.5 * (metric + metric.conj().T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    values, vectors = np.linalg.eigh(metric)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    keep = values > float(metric_tol) * scale
    if not np.any(keep):
        raise ValueError("the internal-core metric has empty numerical support.")
    frame = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    reduced = frame.conj().T @ hamiltonian @ frame
    _, vector = eigh(
        0.5 * (reduced + reduced.conj().T),
        subset_by_index=(0, 0),
        driver="evr",
    )
    return frame @ vector[:, 0], int(np.count_nonzero(keep))


def _optimize_plaquette_cores(
    cores,
    metric,
    hamiltonian,
    *,
    tied,
    metric_tol,
    internal_sweeps,
):
    cores = [np.asarray(core).copy() for core in cores]
    ranks = []
    directions = (range(4), range(3, -1, -1))
    for internal_sweep in range(int(internal_sweeps)):
        for core_index in directions[internal_sweep % 2]:
            core_map = _core_map(cores, core_index, tied=tied)
            reduced_metric = core_map.conj().T @ (metric @ core_map)
            reduced_hamiltonian = core_map.conj().T @ (hamiltonian @ core_map)
            vector, support = _lowest_generalized_vector(
                reduced_hamiltonian,
                reduced_metric,
                metric_tol=metric_tol,
            )
            cores[core_index] = vector.reshape(cores[core_index].shape)
            ranks.append(support)
    return cores, tuple(ranks)


def _load_state(snapshot, *, ranks, j2):
    archive = np.load(snapshot)
    dense = [archive[f"tensor_{site:03d}"] for site in range(4)]
    cores = [
        factorize_plaquette_tensor(
            tensor,
            rank=ranks[site],
            tied=(site == 0),
        )
        for site, tensor in enumerate(dense)
    ]
    tensors = [
        materialize_plaquette_tensor(site_cores, tied=(site == 0))
        for site, site_cores in enumerate(cores)
    ]
    state = FrontierTiedLETTA(
        plaquette_hamiltonian(j2=j2),
        (16,) * 4,
        PARENT_SETS,
        bond_dim=4,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    return state, cores


def _load_factorized_state(snapshot, *, j2):
    archive = np.load(snapshot)
    cores = [
        [archive[f"core_{site:03d}_{core:03d}"] for core in range(4)]
        for site in range(4)
    ]
    tensors = [
        materialize_plaquette_tensor(site_cores, tied=(site == 0))
        for site, site_cores in enumerate(cores)
    ]
    state = FrontierTiedLETTA(
        plaquette_hamiltonian(j2=j2),
        (16,) * 4,
        PARENT_SETS,
        bond_dim=4,
        tensors=tensors,
        frontier_backend="identity_block",
    )
    return state, cores


def _parameter_count(cores):
    return int(sum(core.size for site in cores for core in site))


def run_benchmark(
    *,
    rank,
    tied_rank=None,
    plain_rank=None,
    j2=0.5,
    sweeps=80,
    internal_sweeps=2,
    tolerance=1.0e-8,
    metric_tol=1.0e-12,
    initial_snapshot=DEFAULT_INITIAL,
    resume_snapshot=None,
    output=DEFAULT_OUTPUT,
    verbose=True,
):
    tied_rank = int(rank if tied_rank is None else tied_rank)
    plain_rank = int(rank if plain_rank is None else plain_rank)
    ranks = (tied_rank, plain_rank, plain_rank, plain_rank)
    if resume_snapshot is None:
        state, cores = _load_state(initial_snapshot, ranks=ranks, j2=j2)
    else:
        state, cores = _load_factorized_state(resume_snapshot, j2=j2)
    initial_energy = float(state.expectation())
    history = []
    previous = initial_energy
    start = perf_counter()
    converged = False
    for sweep in range(int(sweeps)):
        order = range(4) if sweep % 2 == 0 else range(3, -1, -1)
        accepted = 0
        supports = []
        for site in order:
            energy_before = float(state.expectation())
            old_tensor = state.tensors[site].copy()
            old_cores = [core.copy() for core in cores[site]]
            metric, hamiltonian = state.local_operators(site)
            candidate, local_supports = _optimize_plaquette_cores(
                old_cores,
                metric,
                hamiltonian,
                tied=(site == 0),
                metric_tol=metric_tol,
                internal_sweeps=internal_sweeps,
            )
            state.tensors[site] = materialize_plaquette_tensor(
                candidate,
                tied=(site == 0),
            )
            energy_after = float(state.expectation())
            if np.isfinite(energy_after) and energy_after <= energy_before + 1.0e-11:
                cores[site] = candidate
                accepted += 1
                supports.extend(local_supports)
            else:
                state.tensors[site] = old_tensor
                cores[site] = old_cores
        energy = float(state.expectation())
        delta = abs(energy - previous)
        row = {
            "sweep": sweep,
            "energy": energy,
            "delta_energy": delta,
            "accepted_sites": accepted,
            "solver_failures": 0,
            "minimum_core_metric_rank": min(supports) if supports else 0,
        }
        history.append(row)
        if verbose:
            print(
                f"factorized-plaquette sweep={sweep} E={energy:.12f} "
                f"dE={delta:.3e} accepted={accepted}",
                flush=True,
            )
        if delta <= float(tolerance):
            converged = True
            break
        previous = energy

    snapshot = Path(output).with_suffix(".npz")
    np.savez_compressed(
        snapshot,
        **{
            f"core_{site:03d}_{core:03d}": value
            for site, site_cores in enumerate(cores)
            for core, value in enumerate(site_cores)
        },
    )
    exact = -7.505556950081064 if float(j2) == 0.5 else None
    payload = {
        "model": {
            "shape": [4, 4],
            "j1": 1.0,
            "j2": float(j2),
            "boundary": "open",
        },
        "ansatz": {
            "type": "four-core internally factorized plaquette LETTA",
            "macro_order": ["B", "A", "C", "D"],
            "macro_tree": [["B", "A"], ["A", "C"], ["C", "D"]],
            "factorized_physical_tie": ["B", "D"],
            "internal_ranks": {
                "B_tied": tied_rank,
                "A": plain_rank,
                "C": plain_rank,
                "D": plain_rank,
            },
            "macro_bond_dim": 4,
            "parameters": _parameter_count(cores),
        },
        "settings": {
            "sweeps": int(sweeps),
            "internal_sweeps": int(internal_sweeps),
            "tolerance": float(tolerance),
            "metric_tolerance": float(metric_tol),
            "initial_snapshot": str(initial_snapshot),
            "resume_snapshot": (
                None if resume_snapshot is None else str(resume_snapshot)
            ),
            "optimizer": "exact macro environments with variational core ALS",
        },
        "initial_energy": initial_energy,
        "energy": float(state.expectation()),
        "exact_energy": exact,
        "energy_error": None if exact is None else float(state.expectation()) - exact,
        "converged": converged,
        "seconds": perf_counter() - start,
        "history": history,
        "snapshot": str(snapshot),
    }
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--tied-rank", type=int)
    parser.add_argument("--plain-rank", type=int)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--sweeps", type=int, default=80)
    parser.add_argument("--internal-sweeps", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--initial-snapshot", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--resume-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        rank=args.rank,
        tied_rank=args.tied_rank,
        plain_rank=args.plain_rank,
        j2=args.j2,
        sweeps=args.sweeps,
        internal_sweeps=args.internal_sweeps,
        tolerance=args.tolerance,
        metric_tol=args.metric_tol,
        initial_snapshot=args.initial_snapshot,
        resume_snapshot=args.resume_snapshot,
        output=args.output,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "energy": result["energy"],
                "parameters": result["ansatz"]["parameters"],
                "converged": result["converged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
