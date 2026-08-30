"""Projector-free frontier LETTA on the square-lattice J1-J2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from pyqed.letta import (
    DenseTiedLETTA,
    FrontierTiedLETTA,
    heisenberg_block_frontier_profile,
    heisenberg_frontier_profile,
    optimize_heisenberg_block_order,
    optimize_heisenberg_order,
)
from pyqed.tn import LocalHamiltonian, LocalTerm


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_j1j2_square_4x4_D4.json"


def heisenberg_local_hamiltonian(nsites, weighted_bonds):
    """Build a termwise spin-1/2 Heisenberg Hamiltonian."""
    exchange = np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    )
    terms = [
        LocalTerm((int(left), int(right)), float(coupling) * exchange)
        for left, right, coupling in weighted_bonds
    ]
    return LocalHamiltonian((2,) * int(nsites), terms)


def remap_edges(edges, site_order):
    """Map edges from the original snake indices to positions in ``site_order``."""
    position = {int(site): new_site for new_site, site in enumerate(site_order)}
    result = []
    for edge in edges:
        left, right, *payload = edge
        mapped = tuple(sorted((position[int(left)], position[int(right)])))
        result.append((*mapped, *payload))
    return tuple(sorted(result))


def benchmark_frontier(
    *,
    nrows=4,
    ncols=4,
    j1=1.0,
    j2=0.5,
    bond_dim=4,
    sweeps=1,
    seed=7,
    solver="direct",
    frontier_backend="compressed",
    optimize_order=False,
    path_optimizer="greedy",
    validate_projector=False,
    validation_site=None,
    validate_matrix_free=False,
    matrix_free_site=None,
    verbose=False,
):
    nsites = int(nrows) * int(ncols)
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((i, j, float(j1)) for i, j in nearest)
    weighted_bonds += tuple((i, j, float(j2)) for i, j in diagonals)
    identity_order = tuple(range(nsites))
    normalized_backend = str(frontier_backend).lower().replace("-", "_")
    if normalized_backend == "identity_block":
        profile_function = heisenberg_block_frontier_profile
        order_function = optimize_heisenberg_block_order
        ordering_objective = "identity-aware-block-elements-without-D-squared"
    else:
        profile_function = heisenberg_frontier_profile
        order_function = optimize_heisenberg_order
        ordering_objective = "compressed-MPO-elements-without-D-squared"
    start = perf_counter()
    site_order = (
        order_function(nsites, nearest, weighted_bonds)
        if optimize_order
        else identity_order
    )
    ordering_seconds = perf_counter() - start
    original_profile = profile_function(
        nsites,
        nearest,
        weighted_bonds,
        identity_order,
    )
    selected_profile = profile_function(
        nsites,
        nearest,
        weighted_bonds,
        site_order,
    )
    nearest = remap_edges(nearest, site_order)
    diagonals = remap_edges(diagonals, site_order)
    weighted_bonds = remap_edges(weighted_bonds, site_order)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)

    start = perf_counter()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dim=bond_dim,
        seed=seed,
        frontier_backend=frontier_backend,
        path_optimizer=path_optimizer,
    )
    setup_seconds = perf_counter() - start
    initial_energy = state.energy
    start = perf_counter()
    state.run(nsweeps=sweeps, solver=solver, verbose=verbose)
    sweep_seconds = perf_counter() - start

    validation = {}
    if validate_projector:
        site = nsites // 2 if validation_site is None else int(validation_site)
        sparse_hamiltonian = sparse_heisenberg_hamiltonian(
            nsites,
            weighted_bonds,
        )
        dense_reference = DenseTiedLETTA(
            sparse_hamiltonian,
            hamiltonian.dims,
            parent_sets,
            bond_dim=bond_dim,
            tensors=state.tensors,
        )
        dense_reference.tensors = [tensor.copy() for tensor in state.tensors]
        start = perf_counter()
        projector = dense_reference.local_projector(site, sparse=True)
        reference_metric = (projector.getH() @ projector).toarray()
        reference_effective = (
            projector.getH() @ (sparse_hamiltonian @ projector)
        ).toarray()
        projector_seconds = perf_counter() - start
        start = perf_counter()
        environment = state.site_environment(site)
        environment_seconds = perf_counter() - start
        start = perf_counter()
        metric, effective = state.local_operators(
            site,
            environment=environment,
        )
        cached_hole_seconds = perf_counter() - start
        probe = np.linspace(-0.5, 0.7, state.tensors[site].size)
        metric_error = float(np.max(np.abs(metric - reference_metric)))
        hamiltonian_error = float(np.max(np.abs(effective - reference_effective)))
        validation["explicit_projector"] = {
            "site": site,
            "metric_max_error": metric_error,
            "metric_relative_error": metric_error
            / max(float(np.max(np.abs(reference_metric))), np.finfo(float).tiny),
            "hamiltonian_max_error": hamiltonian_error,
            "hamiltonian_relative_error": hamiltonian_error
            / max(
                float(np.max(np.abs(reference_effective))),
                np.finfo(float).tiny,
            ),
            "metric_action_max_error": float(
                np.max(
                    np.abs(
                        state.metric_action(
                            site,
                            probe,
                            environment=environment,
                        )
                        - reference_metric @ probe
                    )
                )
            ),
            "hamiltonian_action_max_error": float(
                np.max(
                    np.abs(
                        state.hamiltonian_action(
                            site,
                            probe,
                            environment=environment,
                        )
                        - reference_effective @ probe
                    )
                )
            ),
            "environment_build_seconds": float(environment_seconds),
            "cached_hole_seconds": float(cached_hole_seconds),
            "explicit_projector_seconds": float(projector_seconds),
        }

    if validate_matrix_free:
        site = nsites - 1 if matrix_free_site is None else int(matrix_free_site)
        direct_state = state.copy()
        iterative_state = state.copy()
        start = perf_counter()
        direct_environment = direct_state.site_environment(site)
        direct_environment_seconds = perf_counter() - start
        start = perf_counter()
        direct_update = direct_state.optimize_site(
            site,
            solver="direct",
            environment=direct_environment,
        )
        direct_seconds = perf_counter() - start
        start = perf_counter()
        iterative_environment = iterative_state.site_environment(site)
        iterative_environment_seconds = perf_counter() - start
        start = perf_counter()
        iterative_update = iterative_state.optimize_site(
            site,
            solver="matrix_free",
            eig_tol=1.0e-11,
            environment=iterative_environment,
        )
        iterative_seconds = perf_counter() - start
        validation["matrix_free"] = {
            "site": site,
            "local_dimension": state.tensors[site].size,
            "metric_rank": iterative_update.metric_rank,
            "metric_rank_is_projected": (iterative_update.metric_rank_is_projected),
            "hamiltonian_matvecs": iterative_update.hamiltonian_matvecs,
            "metric_matvecs": iterative_update.metric_matvecs,
            "iterations": iterative_update.iterations,
            "residual_norm": iterative_update.residual_norm,
            "solver": iterative_update.solver,
            "solver_converged": iterative_update.solver_converged,
            "message": iterative_update.message,
            "direct_energy": float(direct_update.energy),
            "matrix_free_energy": float(iterative_update.energy),
            "energy_difference": float(iterative_update.energy - direct_update.energy),
            "direct_environment_seconds": float(direct_environment_seconds),
            "matrix_free_environment_seconds": float(iterative_environment_seconds),
            "direct_seconds": float(direct_seconds),
            "matrix_free_seconds": float(iterative_seconds),
        }

    return {
        "model": {
            "nrows": int(nrows),
            "ncols": int(ncols),
            "j1": float(j1),
            "j2": float(j2),
            "boundary": "open",
            "local_terms": hamiltonian.nterms,
        },
        "ansatz": {
            "kind": "frontier-dense-tied-letta",
            "tie_graph": "all-j1-nearest-neighbor-bonds",
            "bond_dim": int(bond_dim),
            "tie_edges": int(sum(map(len, parent_sets))),
            "parameters": state.nparameters,
            "hamiltonian_mpo_bond_dim": max(state.hamiltonian_mpo.bond_dims),
            "compressed_hamiltonian_mpo_bond_dim": (
                state.compressed_hamiltonian_mpo_bond_dim
            ),
            "uncompressed_hamiltonian_mpo_bond_dim": (
                state.uncompressed_hamiltonian_mpo_bond_dim
            ),
            "parent_sets": parent_sets,
            "site_order_in_original_snake_indices": site_order,
        },
        "settings": {
            "sweeps": int(sweeps),
            "solver": solver,
            "frontier_backend": frontier_backend,
            "optimize_order": bool(optimize_order),
            "path_optimizer": path_optimizer,
            "seed": int(seed),
        },
        "ordering": {
            "method": "exact-subset-dp" if optimize_order else "snake",
            "objective": ordering_objective,
            "seconds": float(ordering_seconds),
            "original_peak_score": max(
                (entry["score"] for entry in original_profile),
                default=0.0,
            ),
            "selected_peak_score": max(
                (entry["score"] for entry in selected_profile),
                default=0.0,
            ),
            "original_total_score": sum(entry["score"] for entry in original_profile),
            "selected_total_score": sum(entry["score"] for entry in selected_profile),
            "selected_max_frontier_width": max(
                (entry["frontier_width"] for entry in selected_profile),
                default=0,
            ),
            "selected_max_operator_rank": max(
                (
                    entry["operator_rank"]
                    for entry in selected_profile
                    if "operator_rank" in entry
                ),
                default=None,
            ),
            "selected_max_uncompressed_mpo_channels": max(
                (
                    entry["mpo_channels"]
                    for entry in selected_profile
                    if "mpo_channels" in entry
                ),
                default=None,
            ),
        },
        "explicit_basis_materialized_in_solver": False,
        "initial_energy": float(initial_energy),
        "energy": float(state.energy),
        "setup_seconds": float(setup_seconds),
        "sweep_seconds": float(sweep_seconds),
        "contraction_plans": state.contraction_plans,
        "peak_frontier_elements": state.peak_frontier_elements,
        "norm_peak_frontier_elements": state.norm_peak_frontier_elements,
        "hamiltonian_peak_frontier_elements": (
            state.hamiltonian_peak_frontier_elements
        ),
        "cached_environment_elements": state.cached_environment_elements,
        "sweep_energies": [float(record["energy"]) for record in state.history],
        "validation": validation or None,
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
        "--solver",
        choices=("direct", "matrix_free", "block_sparse", "auto"),
        default="direct",
    )
    parser.add_argument(
        "--frontier-backend",
        choices=("compressed", "identity_block"),
        default="compressed",
    )
    parser.add_argument(
        "--optimize-order",
        action="store_true",
        help="exactly minimize peak and then total compressed-frontier storage",
    )
    parser.add_argument("--path-optimizer", default="greedy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-projector", action="store_true")
    parser.add_argument("--validation-site", type=int)
    parser.add_argument("--validate-matrix-free", action="store_true")
    parser.add_argument("--matrix-free-site", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = benchmark_frontier(
        nrows=args.rows,
        ncols=args.cols,
        j1=args.j1,
        j2=args.j2,
        bond_dim=args.bond_dim,
        sweeps=args.sweeps,
        solver=args.solver,
        frontier_backend=args.frontier_backend,
        optimize_order=args.optimize_order,
        path_optimizer=args.path_optimizer,
        seed=args.seed,
        validate_projector=args.validate_projector,
        validation_site=args.validation_site,
        validate_matrix_free=args.validate_matrix_free,
        matrix_free_site=args.matrix_free_site,
        verbose=args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
