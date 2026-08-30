"""Benchmark the no-tie 2x2 supersite MPS baseline for square J1-J2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

from examples.mps.adaptive_cp_letta_j1j2_square import (
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import AbelianFrontierTiedLETTA, FrontierAbelianLayout
from pyqed.letta.plaquette_blocks import (
    block_local_hamiltonian,
    block_state_vector,
    blocked_local_charges,
    square_plaquette_blocks,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "plaquette_block_letta_j1j2_4x4.json"


def _truncated_tt_state(vector, dims, max_rank):
    """Return a normalized TT-SVD reference and its internal ranks."""
    dims = tuple(int(dim) for dim in dims)
    max_rank = int(max_rank)
    work = vector.reshape(dims)
    cores = []
    left_rank = 1
    for site, dim in enumerate(dims[:-1]):
        matrix = work.reshape(left_rank * dim, -1)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, singular_values.size)
        cores.append(left[:, :rank].reshape(left_rank, dim, rank))
        work = singular_values[:rank, None] * right[:rank]
        left_rank = rank
    cores.append(work.reshape(left_rank, dims[-1], 1))
    state = cores[0][0]
    for core in cores[1:]:
        state = np.tensordot(state, core, axes=(-1, 0))
    state = state[..., 0].reshape(-1)
    state /= np.linalg.norm(state)
    return state, tuple(core.shape[2] for core in cores[:-1])


def benchmark(
    *,
    nrows=4,
    ncols=4,
    j1=1.0,
    j2=0.5,
    bond_dim=4,
    sweeps=2,
    optimizer="two_site",
    solver="matrix_free",
    pair_backend="block",
    frontier_backend="identity_block",
    seed=7,
    exact_reference=True,
):
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows % 2 or ncols % 2:
        raise ValueError("the plaquette trial requires even lattice dimensions.")
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, float(j1)) for left, right in nearest)
    weighted += tuple((left, right, float(j2)) for left, right in diagonals)
    microscopic = heisenberg_local_hamiltonian(nsites, weighted)
    blocks = square_plaquette_blocks(nrows, ncols)

    start = perf_counter()
    hamiltonian = block_local_hamiltonian(microscopic, blocks)
    blocking_seconds = perf_counter() - start
    local_qns = blocked_local_charges((((1,), (-1,)),) * nsites, blocks)
    layout = FrontierAbelianLayout.from_local_charges(
        local_qns,
        target=(0,),
        bond_dims=int(bond_dim),
    )
    parent_sets = ((),) * len(blocks)

    start = perf_counter()
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        abelian_layout=layout,
        seed=seed,
        frontier_backend=frontier_backend,
    )
    setup_seconds = perf_counter() - start
    initial_energy = float(state.energy)

    start = perf_counter()
    if optimizer == "two_site":
        state.run_two_site(
            nsweeps=int(sweeps),
            solver=solver,
            split_strategy="svd",
            pair_operator_backend=pair_backend,
            eig_tol=1.0e-10,
            maxiter=400,
            max_subspace=48,
        )
    elif optimizer == "one_site":
        state.run(
            nsweeps=int(sweeps),
            solver=solver,
            eig_tol=1.0e-10,
            maxiter=400,
        )
    else:
        raise ValueError("optimizer must be 'one_site' or 'two_site'.")
    sweep_seconds = perf_counter() - start

    exact_energy = None
    exact_seconds = None
    exact_block_tt = None
    if exact_reference and nsites <= 16:
        sparse = sparse_heisenberg_hamiltonian(nsites, weighted)
        start = perf_counter()
        values, vectors = eigsh(
            sparse,
            k=1,
            which="SA",
            return_eigenvectors=True,
            tol=1.0e-11,
        )
        exact_energy = float(values[0])
        exact_seconds = perf_counter() - start
        exact_block_vector = block_state_vector(
            vectors[:, 0],
            microscopic.dims,
            blocks,
        )
        tt_state, tt_ranks = _truncated_tt_state(
            exact_block_vector,
            hamiltonian.dims,
            bond_dim,
        )
        tt_energy = float(
            np.real(np.vdot(tt_state, hamiltonian.matvec(tt_state)))
        )
        exact_block_tt = {
            "bond_limit": int(bond_dim),
            "ranks": list(tt_ranks),
            "energy": tt_energy,
            "energy_error": float(tt_energy - exact_energy),
            "fidelity": float(abs(np.vdot(exact_block_vector, tt_state)) ** 2),
            "used_during_optimization": False,
        }

    return {
        "model": {
            "shape": [nrows, ncols],
            "j1": float(j1),
            "j2": float(j2),
            "boundary": "open",
            "target_two_sz": 0,
        },
        "blocking": {
            "ansatz_kind": "dimension-16 plaquette supersite MPS",
            "shape": [2, 2],
            "nblocks": len(blocks),
            "blocks": [list(block) for block in blocks],
            "block_dims": list(hamiltonian.dims),
            "unfused_physical_shape": [2, 2, 2, 2],
            "parent_sets": [list(parents) for parents in parent_sets],
            "intra_block_ties": False,
            "hamiltonian_mapping": "exact product-basis permutation",
            "blocking_seconds": float(blocking_seconds),
        },
        "solver": {
            "bond_dims": list(state.bond_dims),
            "optimizer": str(optimizer),
            "solver": str(solver),
            "pair_backend": str(pair_backend),
            "frontier_backend": str(frontier_backend),
            "sweeps": int(sweeps),
            "symmetry_parameters": int(state.nparameters),
            "dense_parameters": int(state.dense_nparameters),
            "setup_seconds": float(setup_seconds),
            "sweep_seconds": float(sweep_seconds),
            "initial_energy": initial_energy,
            "energy": float(state.energy),
            "history": [float(record["energy"]) for record in state.history],
            "peak_frontier_elements": int(state.peak_frontier_elements),
            "cached_environment_elements": int(state.cached_environment_elements),
        },
        "exact": {
            "energy": exact_energy,
            "seconds": exact_seconds,
            "block_tt_svd_reference": exact_block_tt,
            "energy_error": (
                None if exact_energy is None else float(state.energy - exact_energy)
            ),
            "energy_error_per_site": (
                None
                if exact_energy is None
                else float((state.energy - exact_energy) / nsites)
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=("one_site", "two_site"),
        default="two_site",
    )
    parser.add_argument(
        "--solver",
        choices=("direct", "whitened", "matrix_free", "block_sparse"),
        default="matrix_free",
    )
    parser.add_argument(
        "--frontier-backend",
        choices=("compressed", "identity_block"),
        default="identity_block",
    )
    parser.add_argument(
        "--pair-backend",
        choices=("block", "dense", "auto"),
        default="block",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-exact-reference", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = benchmark(
        nrows=args.rows,
        ncols=args.cols,
        j1=args.j1,
        j2=args.j2,
        bond_dim=args.bond_dim,
        sweeps=args.sweeps,
        optimizer=args.optimizer,
        solver=args.solver,
        pair_backend=args.pair_backend,
        frontier_backend=args.frontier_backend,
        seed=args.seed,
        exact_reference=not args.no_exact_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
