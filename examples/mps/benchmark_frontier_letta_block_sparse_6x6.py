#!/usr/bin/env python3
"""Warm-started 6x6 J1-J2 benchmark for block-sparse frontier LETTA."""

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


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_block_sparse_6x6.json"


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


def run_benchmark(
    *,
    j2=0.5,
    bond_dim=4,
    reference_bond_dim=8,
    mps_passes=20,
    reference_passes=20,
    letta_passes=4,
    tolerance=1.0e-9,
    tie_noise=1.0e-3,
    seed=7,
    output=DEFAULT_OUTPUT,
    snapshot=None,
):
    nrows = ncols = 6
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(j2)) for left, right in diagonals
    )
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    mpo = hamiltonian.to_mpo().compress()

    output = Path(output)
    snapshot = (
        output.with_suffix(".npz") if snapshot is None else Path(snapshot)
    )
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
            "tie_graph": "all nearest-neighbor J1 bonds",
            "tie_edges": len(nearest),
            "j2_diagonal_edges": len(diagonals),
            "full_hilbert_dimension": 2**nsites,
        },
        "settings": {
            "bond_dim": int(bond_dim),
            "reference_bond_dim": int(reference_bond_dim),
            "mps_directional_pass_limit": int(mps_passes),
            "reference_directional_pass_limit": int(reference_passes),
            "letta_directional_pass_limit": int(letta_passes),
            "tolerance": float(tolerance),
            "tie_noise": float(tie_noise),
            "seed": int(seed),
            "frontier_backend": "identity_block",
            "local_solver": "block_sparse",
            "mps_symmetry": "none",
            "letta_symmetry": "none",
        },
        "results": {},
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
    print(
        f"  MPS D={bond_dim}: E={mps_record['energy']:.12f}, "
        f"E/N={mps_record['energy_per_site']:.12f}, "
        f"time={mps_record['optimization_seconds']:.2f}s",
        flush=True,
    )

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
        print(
            f"  MPS D={current_bond_dim}: E={reference_record['energy']:.12f}, "
            f"E/N={reference_record['energy_per_site']:.12f}, "
            f"time={reference_record['optimization_seconds']:.2f}s",
            flush=True,
        )
        previous_bond_dim = current_bond_dim

    print("lifting MPS into all-NN-tied LETTA", flush=True)
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
    initial_energy = float(state.energy)
    print(
        f"  LETTA initial: E={initial_energy:.12f}, "
        f"peak frontier={state.peak_compressed_frontier_elements:,} elements",
        flush=True,
    )

    optimization_start = perf_counter()
    state.run(
        nsweeps=letta_passes,
        tol=0.0,
        solver="block_sparse",
        eig_tol=tolerance,
        maxiter=800,
        max_subspace=64,
        verbose=True,
    )
    optimization_seconds = perf_counter() - optimization_start
    fresh_energy = float(state.expectation())
    updates = [update for row in state.history for update in row["updates"]]
    result = {
        "bond_dim": int(bond_dim),
        "optimizer": "one_site_block_sparse_generalized_eigensolve",
        "symmetry": "none",
        "parameters": int(state.nparameters),
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": float(optimization_seconds),
        "initial_energy": initial_energy,
        "energy": float(state.energy),
        "fresh_energy": fresh_energy,
        "energy_per_site": fresh_energy / nsites,
        "improvement_from_same_d_mps": fresh_energy - float(mps_record["energy"]),
        "energy_above_mps_reference_per_site": (
            (fresh_energy - float(reference_record["energy"])) / nsites
            if reference_record is not None
            else None
        ),
        "directional_passes_completed": len(state.history),
        "directional_pass_energies": [
            float(row["energy"]) for row in state.history
        ],
        "solver_failures": int(
            sum(row["solver_failures"] for row in state.history)
        ),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "site_updates": len(updates),
        "maximum_block_resident_elements": max(
            (update.stored_operator_elements for update in updates), default=0
        ),
        "physical_block_counts": sorted(
            {int(update.physical_blocks) for update in updates}
        ),
        "hamiltonian_block_counts": sorted(
            {int(update.hamiltonian_blocks) for update in updates}
        ),
        "component_size_patterns": sorted(
            {tuple(update.block_component_sizes) for update in updates}
        ),
        "peak_frontier_elements": int(state.peak_frontier_elements),
        "peak_compressed_frontier_elements": int(
            state.peak_compressed_frontier_elements
        ),
        "cached_environment_elements": int(state.cached_environment_elements),
        "snapshot": str(snapshot),
    }
    payload["results"][f"letta_d{bond_dim}"] = result
    payload["status"] = "complete"
    _save_snapshot(snapshot, state.tensors)
    _write_json(output, payload)
    print(
        f"  LETTA D={bond_dim}: E={fresh_energy:.12f}, "
        f"E/N={fresh_energy / nsites:.12f}, "
        f"time={optimization_seconds:.2f}s",
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    print(f"wrote {snapshot}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--reference-bond-dim", type=int, default=8)
    parser.add_argument("--mps-passes", type=int, default=20)
    parser.add_argument("--reference-passes", type=int, default=20)
    parser.add_argument("--letta-passes", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    run_benchmark(
        j2=args.j2,
        bond_dim=args.bond_dim,
        reference_bond_dim=args.reference_bond_dim,
        mps_passes=args.mps_passes,
        reference_passes=args.reference_passes,
        letta_passes=args.letta_passes,
        tolerance=args.tolerance,
        tie_noise=args.tie_noise,
        seed=args.seed,
        output=args.output,
        snapshot=args.snapshot,
    )


if __name__ == "__main__":
    main()
