"""Benchmark a microscopic plaquette-ordered frontier LETTA with boundary ties."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

from scipy.sparse.linalg import eigsh

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierAbelianLayout,
    abelian_frontier_tied_letta_from_mps,
    exact_block_factor_layout,
)
from pyqed.letta.plaquette_blocks import (
    interplaquette_edges,
    plaquette_site_order,
    remap_site_edges,
    square_plaquette_blocks,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE / "results" / "plaquette_frontier_letta_j1j2_4x4_D4.json"
)


def _remap_weighted_edges(weighted_edges, site_order):
    position = {site: new_site for new_site, site in enumerate(site_order)}
    return tuple(
        (
            *sorted((position[int(left)], position[int(right)])),
            float(coupling),
        )
        for left, right, coupling in weighted_edges
    )


def plaquette_frontier_problem(
    nrows,
    ncols,
    *,
    j1=1.0,
    j2=0.5,
    tie_graph="nearest",
):
    """Build a plaquette-contiguous Hamiltonian and inter-plaquette tie graph."""
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows % 2 or ncols % 2:
        raise ValueError("the plaquette trial requires even lattice dimensions.")
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    blocks = square_plaquette_blocks(nrows, ncols)
    site_order = plaquette_site_order(blocks)

    weighted = tuple((left, right, float(j1)) for left, right in nearest)
    weighted += tuple((left, right, float(j2)) for left, right in diagonals)
    remapped_weighted = _remap_weighted_edges(weighted, site_order)

    tie_graph = str(tie_graph).lower().replace("-", "_")
    if tie_graph == "nearest":
        tie_candidates = nearest
    elif tie_graph in {"interactions", "all"}:
        tie_candidates = nearest + diagonals
        tie_graph = "interactions"
    elif tie_graph in {"none", "mps"}:
        tie_candidates = ()
        tie_graph = "none"
    else:
        raise ValueError("tie_graph must be 'nearest', 'interactions', or 'none'.")
    boundary_edges = interplaquette_edges(blocks, tie_candidates)
    remapped_ties = remap_site_edges(boundary_edges, site_order)
    parent_sets = parent_sets_from_edges(nsites, remapped_ties)
    hamiltonian = heisenberg_local_hamiltonian(nsites, remapped_weighted)

    remapped_blocks = tuple(
        tuple(range(4 * block, 4 * block + 4))
        for block in range(len(blocks))
    )
    return {
        "hamiltonian": hamiltonian,
        "weighted_edges": remapped_weighted,
        "parent_sets": parent_sets,
        "tie_edges": remapped_ties,
        "tie_graph": tie_graph,
        "site_order": site_order,
        "original_blocks": blocks,
        "blocks": remapped_blocks,
    }


def benchmark(
    *,
    nrows=4,
    ncols=4,
    j1=1.0,
    j2=0.5,
    bond_dim=4,
    sweeps=1,
    optimizer="two_site",
    solver="matrix_free",
    pair_backend="action",
    frontier_backend="identity_block",
    tie_graph="nearest",
    warm_sweeps=4,
    tie_noise=0.0,
    fused=False,
    environment_cache="checkpointed",
    checkpoint_interval=None,
    seed=7,
    exact_reference=True,
):
    nrows = int(nrows)
    ncols = int(ncols)
    nsites = nrows * ncols
    problem = plaquette_frontier_problem(
        nrows,
        ncols,
        j1=j1,
        j2=j2,
        tie_graph=tie_graph,
    )
    hamiltonian = problem["hamiltonian"]
    boundary_layout = FrontierAbelianLayout.spin_half(
        nsites,
        target_two_sz=0,
        bond_dims=int(bond_dim),
    )
    physical_sites = tuple(
        (site,) + parents
        for site, parents in enumerate(problem["parent_sets"])
    )
    layout = (
        exact_block_factor_layout(
            boundary_layout,
            problem["blocks"],
            physical_sites,
        )
        if fused
        else boundary_layout
    )

    warm_sweeps = int(warm_sweeps)
    if warm_sweeps < 0:
        raise ValueError("warm_sweeps must be nonnegative.")
    tie_noise = float(tie_noise)
    if tie_noise < 0.0:
        raise ValueError("tie_noise must be nonnegative.")

    start = perf_counter()
    baseline = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * nsites,
        abelian_layout=layout,
        seed=seed,
        frontier_backend=frontier_backend,
    )
    random_initial_energy = float(baseline.energy)
    if warm_sweeps:
        baseline.run_two_site(
            nsweeps=warm_sweeps,
            solver=solver,
            split_strategy="svd",
            pair_operator_backend=pair_backend,
            eig_tol=1.0e-10,
            maxiter=400,
            max_subspace=48,
        )
    baseline_energy = float(baseline.energy)
    mps_tensors = tuple(
        tensor.transpose(0, 2, 1) for tensor in baseline.tensors
    )
    state = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        problem["parent_sets"],
        mps_tensors,
        abelian_layout=layout,
        tie_noise=tie_noise,
        seed=seed,
        frontier_backend=frontier_backend,
    )
    setup_seconds = perf_counter() - start
    initial_energy = float(state.energy)
    del baseline, mps_tensors
    gc.collect()

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
    elif optimizer == "four_site":
        if not fused:
            raise ValueError("optimizer='four_site' requires fused=True.")
        state.run_blocks(
            problem["blocks"],
            nsweeps=int(sweeps),
            environment_cache=environment_cache,
            environment_checkpoint_interval=checkpoint_interval,
            operator_backend="action",
            eig_tol=1.0e-10,
            maxiter=400,
            max_subspace=48,
            merged_dense_fallback_dim=2048,
        )
    elif optimizer == "one_site":
        state.run(
            nsweeps=int(sweeps),
            solver=solver,
            eig_tol=1.0e-10,
            maxiter=400,
        )
    else:
        raise ValueError(
            "optimizer must be 'one_site', 'two_site', or 'four_site'."
        )
    sweep_seconds = perf_counter() - start

    exact_energy = None
    exact_seconds = None
    if exact_reference and nsites <= 16:
        sparse = sparse_heisenberg_hamiltonian(
            nsites,
            problem["weighted_edges"],
        )
        start = perf_counter()
        exact_energy = float(
            eigsh(
                sparse,
                k=1,
                which="SA",
                return_eigenvectors=False,
                tol=1.0e-11,
            )[0]
        )
        exact_seconds = perf_counter() - start
    used_checkpoint_interval = checkpoint_interval
    if optimizer == "four_site" and state.history:
        used_checkpoint_interval = state.history[-1].get(
            "environment_checkpoint_interval"
        )

    return {
        "model": {
            "shape": [nrows, ncols],
            "j1": float(j1),
            "j2": float(j2),
            "boundary": "open",
            "target_two_sz": 0,
        },
        "ansatz": {
            "kind": (
                "exact-factorized fused-plaquette frontier LETTA"
                if fused and problem["tie_edges"]
                else (
                    "exact-factorized fused-plaquette U(1) MPS baseline"
                    if fused
                    else (
                        "microscopic plaquette-ordered frontier LETTA"
                        if problem["tie_edges"]
                        else "microscopic plaquette-ordered U(1) MPS baseline"
                    )
                )
            ),
            "block_shape": [2, 2],
            "microscopic_sites": nsites,
            "microscopic_tensors": nsites,
            "site_order": list(problem["site_order"]),
            "blocks": [list(block) for block in problem["blocks"]],
            "original_blocks": [
                list(block) for block in problem["original_blocks"]
            ],
            "tie_graph": problem["tie_graph"],
            "tie_edges": [list(edge) for edge in problem["tie_edges"]],
            "tie_count": len(problem["tie_edges"]),
            "parent_sets": [
                list(parents) for parents in problem["parent_sets"]
            ],
            "intra_plaquette_ties": False,
            "boundary_tie_dimension": 2,
            "logical_fused_tensors": len(problem["blocks"]) if fused else 0,
            "exact_factorized_fusion": bool(fused),
        },
        "solver": {
            "bond_dims": list(state.bond_dims),
            "optimizer": str(optimizer),
            "solver": str(solver),
            "pair_backend": str(pair_backend),
            "frontier_backend": str(frontier_backend),
            "sweeps": int(sweeps),
            "initialization": "exact U(1) MPS lift",
            "warm_sweeps": warm_sweeps,
            "tie_noise": tie_noise,
            "symmetry_parameters": int(state.nparameters),
            "dense_parameters": int(state.dense_nparameters),
            "setup_seconds": float(setup_seconds),
            "sweep_seconds": float(sweep_seconds),
            "random_mps_energy": random_initial_energy,
            "warm_mps_energy": baseline_energy,
            "initial_energy": initial_energy,
            "lift_energy_error": float(initial_energy - baseline_energy),
            "energy": float(state.energy),
            "energy_per_site": float(state.energy / nsites),
            "history": [float(record["energy"]) for record in state.history],
            "peak_frontier_elements": int(state.peak_frontier_elements),
            "cached_environment_elements": int(
                state.cached_environment_elements
            ),
            "environment_cache": (
                str(environment_cache) if optimizer == "four_site" else None
            ),
            "checkpoint_interval": (
                used_checkpoint_interval if optimizer == "four_site" else None
            ),
            "fixed_environment_cache_elements": (
                int(
                    state.fixed_block_environment_cache_elements(
                        problem["blocks"],
                        mode=environment_cache,
                        interval=used_checkpoint_interval,
                    )
                )
                if optimizer == "four_site"
                else int(state.cached_environment_elements)
            ),
        },
        "exact": {
            "energy": exact_energy,
            "seconds": exact_seconds,
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
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=("one_site", "two_site", "four_site"),
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
        choices=("action", "block", "dense", "auto"),
        default="action",
    )
    parser.add_argument(
        "--tie-graph",
        choices=("nearest", "interactions", "none"),
        default="nearest",
    )
    parser.add_argument("--warm-sweeps", type=int, default=4)
    parser.add_argument("--tie-noise", type=float, default=0.0)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument(
        "--environment-cache",
        choices=("checkpointed", "full"),
        default="checkpointed",
    )
    parser.add_argument("--checkpoint-interval", type=int)
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
        tie_graph=args.tie_graph,
        warm_sweeps=args.warm_sweeps,
        tie_noise=args.tie_noise,
        fused=args.fused,
        environment_cache=args.environment_cache,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        exact_reference=not args.no_exact_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
