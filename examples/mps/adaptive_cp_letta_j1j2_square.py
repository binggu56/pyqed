"""Residual-adaptive graph LETTA for a frustrated square-lattice Heisenberg model.

The spin lattice is mapped to the LETTA chain in a row-wise snake order.  All
variational states are optimized before sparse exact diagonalization is run;
the exact state is used only for post-hoc benchmark errors and fidelities.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from pyqed.letta import CPTiedLETTA
from pyqed.letta.physical_tying import fixed_range_parent_sets


_MODES = (
    "fixed-chain-r2",
    "fixed-chain-r4",
    "warm-lattice-r2",
    "adaptive-rank",
    "adaptive-graph",
    "adaptive-staged",
    "adaptive-joint",
)


def snake_site_order(nrows: int, ncols: int) -> tuple[tuple[int, int], ...]:
    """Return ``(row, column)`` coordinates in row-wise snake order."""
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive.")
    order = []
    for row in range(nrows):
        columns = range(ncols) if row % 2 == 0 else range(ncols - 1, -1, -1)
        order.extend((row, column) for column in columns)
    return tuple(order)


def square_j1_j2_bonds(
    nrows: int,
    ncols: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return OBC nearest and plaquette-diagonal bonds in snake indices."""
    order = snake_site_order(nrows, ncols)
    chain_site = {coordinate: site for site, coordinate in enumerate(order)}

    def edge(left, right):
        return tuple(sorted((chain_site[left], chain_site[right])))

    nearest = []
    diagonals = []
    for row in range(int(nrows)):
        for column in range(int(ncols)):
            if column + 1 < int(ncols):
                nearest.append(edge((row, column), (row, column + 1)))
            if row + 1 < int(nrows):
                nearest.append(edge((row, column), (row + 1, column)))
            if row + 1 < int(nrows) and column + 1 < int(ncols):
                diagonals.append(edge((row, column), (row + 1, column + 1)))
                diagonals.append(edge((row, column + 1), (row + 1, column)))
    return tuple(sorted(set(nearest))), tuple(sorted(set(diagonals)))


def sparse_heisenberg_hamiltonian(
    nsites: int,
    weighted_bonds,
):
    r"""Build ``sum_(ij) J_ij S_i.S_j`` in the full spin basis.

    Site zero is the most-significant bit, matching ``CPTiedLETTA``'s
    ``np.ndindex`` configuration order.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    dimension = 1 << nsites
    states = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=float)
    row_blocks = []
    column_blocks = []
    data_blocks = []
    for left, right, coupling in weighted_bonds:
        left = int(left)
        right = int(right)
        coupling = float(coupling)
        if left == right or min(left, right) < 0 or max(left, right) >= nsites:
            raise ValueError("bond sites must be distinct valid site indices.")
        if not np.isfinite(coupling):
            raise ValueError("bond couplings must be finite.")
        left_mask = 1 << (nsites - 1 - left)
        right_mask = 1 << (nsites - 1 - right)
        different = ((states & left_mask) != 0) != ((states & right_mask) != 0)
        diagonal += coupling * np.where(different, -0.25, 0.25)
        source = states[different]
        row_blocks.append(source ^ left_mask ^ right_mask)
        column_blocks.append(source)
        data_blocks.append(np.full(source.size, 0.5 * coupling))
    row_blocks.append(states)
    column_blocks.append(states)
    data_blocks.append(diagonal)
    return coo_matrix(
        (
            np.concatenate(data_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(dimension, dimension),
    ).tocsr()


def parent_sets_from_edges(
    nsites: int,
    edges,
) -> tuple[tuple[int, ...], ...]:
    """Orient undirected edges toward future sites for physical tying."""
    parents = [set() for _ in range(int(nsites))]
    for left, right in edges:
        left, right = sorted((int(left), int(right)))
        if left == right or left < 0 or right >= int(nsites):
            raise ValueError("edges must contain distinct valid site indices.")
        parents[left].add(right)
    return tuple(tuple(sorted(site_parents)) for site_parents in parents)


def _parent_edges(parent_sets) -> set[tuple[int, int]]:
    return {
        (site, parent)
        for site, parents in enumerate(parent_sets)
        for parent in parents
    }


def _graph_diagnostics(
    parent_sets,
    interaction_edges,
    *,
    baseline_edges=(),
) -> dict[str, float | int]:
    selected = _parent_edges(parent_sets)
    physical = set(interaction_edges)
    baseline = set(baseline_edges)
    overlap = len(selected & physical)
    added = selected - baseline
    added_overlap = len(added & physical)
    missing_physical = physical - baseline
    return {
        "selected_edges": len(selected),
        "physical_edges": len(physical),
        "physical_overlap": overlap,
        "graph_precision": overlap / len(selected) if selected else 0.0,
        "graph_recall": overlap / len(physical) if physical else 0.0,
        "nonphysical_edges": len(selected - physical),
        "added_edges": len(added),
        "added_physical_overlap": added_overlap,
        "added_graph_precision": added_overlap / len(added) if added else 0.0,
        "added_graph_recall": (
            added_overlap / len(missing_physical) if missing_physical else 0.0
        ),
        "removed_baseline_edges": len(baseline - selected),
    }


def mps_cut_diagnostics(state, nsites: int) -> dict[str, object]:
    """Return exact snake-order Schmidt diagnostics for post-hoc comparison."""
    nsites = int(nsites)
    if nsites < 2:
        raise ValueError("MPS cut diagnostics require at least two spins.")
    vector = np.asarray(state).reshape(-1)
    if vector.size != 1 << nsites:
        raise ValueError("state size is inconsistent with nsites.")
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("state must be nonzero.")
    vector = vector / norm
    cuts = []
    for cut in range(1, nsites):
        singular_values = np.linalg.svd(
            vector.reshape(1 << cut, -1),
            compute_uv=False,
        )
        weights = np.square(np.abs(singular_values))
        weights /= np.sum(weights)
        nonzero = weights > 128.0 * np.finfo(float).eps
        entropy = -np.sum(weights[nonzero] * np.log(weights[nonzero]))
        cumulative = np.cumsum(weights)
        required = {
            tolerance: int(np.searchsorted(cumulative, 1.0 - tolerance) + 1)
            for tolerance in (1.0e-2, 1.0e-4, 1.0e-6)
        }
        cuts.append(
            {
                "cut": cut,
                "entropy": float(entropy),
                "exact_rank": int(np.count_nonzero(nonzero)),
                "required_bond_dims": required,
            }
        )
    return {
        "cuts": cuts,
        "max_entropy": max(cut["entropy"] for cut in cuts),
        "max_exact_rank": max(cut["exact_rank"] for cut in cuts),
        "max_bond_dims": {
            tolerance: max(
                cut["required_bond_dims"][tolerance] for cut in cuts
            )
            for tolerance in (1.0e-2, 1.0e-4, 1.0e-6)
        },
    }


def _new_ansatz(hamiltonian, nsites, parents, rank, bond_dim, seed):
    return CPTiedLETTA(
        hamiltonian,
        (2,) * int(nsites),
        parents,
        tie_ranks=int(rank),
        bond_dim=int(bond_dim),
        seed=seed,
    )


def benchmark_square_j1_j2(
    *,
    nrows: int = 3,
    ncols: int = 4,
    j1: float = 1.0,
    j2: float = 0.5,
    bond_dim: int = 2,
    fixed_sweeps: int = 8,
    adaptive_cycles: int = 6,
    initial_sweeps: int = 4,
    branch_sweeps: int = 2,
    candidate_budget: int = 1,
    per_site_graph_candidates: int = 1,
    max_parents: int = 4,
    max_tie_rank: int = 4,
    adaptive_initial_rank: int = 2,
    probe_noise: float = 1.0e-2,
    seed: int = 7,
    modes=_MODES,
    verbose: bool = False,
) -> dict[str, object]:
    """Compare fixed, rank-adaptive, graph-adaptive, and joint LETTA states."""
    nrows = int(nrows)
    ncols = int(ncols)
    nsites = nrows * ncols
    if nsites < 2:
        raise ValueError("the benchmark requires at least two spins.")
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    j1 = float(j1)
    j2 = float(j2)
    if not np.isfinite(j1) or not np.isfinite(j2):
        raise ValueError("j1 and j2 must be finite.")
    active_nearest = nearest if j1 != 0.0 else ()
    active_diagonals = diagonals if j2 != 0.0 else ()
    interactions = tuple(sorted(set(active_nearest) | set(active_diagonals)))
    weighted_bonds = (
        tuple((left, right, j1) for left, right in active_nearest)
        + tuple((left, right, j2) for left, right in active_diagonals)
    )
    hamiltonian = sparse_heisenberg_hamiltonian(nsites, weighted_bonds)
    chain_parents = fixed_range_parent_sets(nsites, tie_range=1) + ((),)
    lattice_parents = parent_sets_from_edges(nsites, interactions)
    if isinstance(modes, str):
        modes = (modes,)
    modes = tuple(dict.fromkeys(str(mode) for mode in modes))
    unknown = set(modes) - set(_MODES)
    if not modes or unknown:
        raise ValueError(f"modes must be a nonempty subset of {_MODES}.")
    adaptive_initial_rank = int(adaptive_initial_rank)
    if (
        any(mode.startswith("adaptive-") for mode in modes)
        and not 1 <= adaptive_initial_rank <= int(max_tie_rank)
    ):
        raise ValueError(
            "adaptive_initial_rank must lie between one and max_tie_rank."
        )

    solved = []
    for mode in modes:
        if verbose:
            print(f"optimizing {mode}", flush=True)
        start = perf_counter()
        if mode == "fixed-chain-r2":
            ansatz = _new_ansatz(
                hamiltonian, nsites, chain_parents, 2, bond_dim, seed
            )
            ansatz.run(nsweeps=fixed_sweeps, tol=0.0)
        elif mode == "fixed-chain-r4":
            ansatz = _new_ansatz(
                hamiltonian, nsites, chain_parents, 4, bond_dim, seed
            )
            ansatz.run(nsweeps=fixed_sweeps, tol=0.0)
        elif mode == "warm-lattice-r2":
            ansatz = _new_ansatz(
                hamiltonian, nsites, chain_parents, 2, bond_dim, seed
            )
            ansatz.run(nsweeps=initial_sweeps, tol=0.0)
            for site, parents in enumerate(lattice_parents[:-1]):
                ansatz.set_parent_set(site, parents)
            ansatz.run(nsweeps=fixed_sweeps, tol=0.0)
        elif mode == "adaptive-staged":
            initial_rank = adaptive_initial_rank
            ansatz = _new_ansatz(
                hamiltonian,
                nsites,
                chain_parents,
                initial_rank,
                bond_dim,
                seed,
            )
            graph_cycles = (int(adaptive_cycles) + 1) // 2
            rank_cycles = int(adaptive_cycles) // 2
            ansatz.run_residual_adaptive(
                max_parents=max_parents,
                max_tie_rank=initial_rank,
                ncycles=graph_cycles,
                initial_sweeps=initial_sweeps,
                branch_sweeps=branch_sweeps,
                candidate_budget=candidate_budget,
                per_site_graph_candidates=per_site_graph_candidates,
                probe_noise=probe_noise,
                proposal_kinds=("graph",),
                seed=seed,
                verbose=verbose,
            )
            graph_history = [
                {**record, "stage": "graph", "global_cycle": index}
                for index, record in enumerate(ansatz.adaptive_history)
            ]
            rank_history = []
            if rank_cycles:
                ansatz.run_residual_adaptive(
                    max_parents=max_parents,
                    max_tie_rank=max_tie_rank,
                    ncycles=rank_cycles,
                    initial_sweeps=0,
                    branch_sweeps=branch_sweeps,
                    candidate_budget=candidate_budget,
                    per_site_graph_candidates=per_site_graph_candidates,
                    probe_noise=probe_noise,
                    proposal_kinds=("rank",),
                    seed=seed + graph_cycles,
                    verbose=verbose,
                )
                rank_history = [
                    {
                        **record,
                        "stage": "rank",
                        "global_cycle": len(graph_history) + index,
                    }
                    for index, record in enumerate(ansatz.adaptive_history)
                ]
            ansatz.adaptive_history = graph_history + rank_history
        else:
            initial_rank = adaptive_initial_rank
            ansatz = _new_ansatz(
                hamiltonian,
                nsites,
                chain_parents,
                initial_rank,
                bond_dim,
                seed,
            )
            proposal_kinds = {
                "adaptive-rank": ("rank",),
                "adaptive-graph": ("graph",),
                "adaptive-joint": ("graph", "rank"),
            }[mode]
            ansatz.run_residual_adaptive(
                max_parents=max_parents if "graph" in proposal_kinds else 1,
                max_tie_rank=(
                    max_tie_rank if "rank" in proposal_kinds else initial_rank
                ),
                ncycles=adaptive_cycles,
                initial_sweeps=initial_sweeps,
                branch_sweeps=branch_sweeps,
                candidate_budget=candidate_budget,
                per_site_graph_candidates=per_site_graph_candidates,
                probe_noise=probe_noise,
                proposal_kinds=proposal_kinds,
                seed=seed,
                verbose=verbose,
            )
        solved.append((mode, ansatz, perf_counter() - start))

    # No exact eigenvector enters initialization, proposal scoring, or relaxation.
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=2,
        which="SA",
        tol=1.0e-11,
        maxiter=100_000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues)[order]
    eigenvectors = np.asarray(eigenvectors)[:, order]
    exact = eigenvectors[:, 0]
    mps_diagnostics = mps_cut_diagnostics(exact, nsites)
    rows = []
    for mode, ansatz, elapsed in solved:
        vector = ansatz.state_vector(normalize=True)
        row = {
            "mode": mode,
            "energy": float(ansatz.energy),
            "energy_error": float(ansatz.energy - eigenvalues[0]),
            "fidelity": float(abs(np.vdot(exact, vector)) ** 2),
            "residual_norm": ansatz.energy_residual()[2],
            "parameters": ansatz.nparameters,
            "tie_ranks": ansatz.tie_ranks,
            "parent_sets": ansatz.parent_sets,
            "elapsed": float(elapsed),
            "adaptive_history": list(ansatz.adaptive_history),
        }
        row.update(
            _graph_diagnostics(
                ansatz.parent_sets,
                interactions,
                baseline_edges=_parent_edges(chain_parents),
            )
        )
        rows.append(row)
    return {
        "nrows": nrows,
        "ncols": ncols,
        "nsites": nsites,
        "j1": float(j1),
        "j2": float(j2),
        "nearest_bonds": nearest,
        "diagonal_bonds": diagonals,
        "interaction_parent_sets": lattice_parents,
        "exact_energy": float(eigenvalues[0]),
        "gap": float(eigenvalues[1] - eigenvalues[0]),
        "mps_diagnostics": mps_diagnostics,
        "settings": {
            "bond_dim": int(bond_dim),
            "fixed_sweeps": int(fixed_sweeps),
            "adaptive_cycles": int(adaptive_cycles),
            "initial_sweeps": int(initial_sweeps),
            "branch_sweeps": int(branch_sweeps),
            "candidate_budget": int(candidate_budget),
            "per_site_graph_candidates": int(per_site_graph_candidates),
            "max_parents": int(max_parents),
            "max_tie_rank": int(max_tie_rank),
            "adaptive_initial_rank": adaptive_initial_rank,
            "probe_noise": float(probe_noise),
            "seed": int(seed),
            "modes": modes,
        },
        "rows": rows,
    }


def _print_benchmark(result) -> None:
    print(
        f"{result['nrows']}x{result['ncols']} OBC J1={result['j1']:g} "
        f"J2={result['j2']:g} E0={result['exact_energy']:.12f} "
        f"gap={result['gap']:.6e}"
    )
    print(
        "mode params dE fidelity residual edges hit add add-hit "
        "add-precision ranks time"
    )
    for row in result["rows"]:
        print(
            f"{row['mode']:>18s} {row['parameters']:5d} "
            f"{row['energy_error']:.6e} {row['fidelity']:.8f} "
            f"{row['residual_norm']:.3e} {row['selected_edges']:2d} "
            f"{row['physical_overlap']:2d} {row['added_edges']:2d} "
            f"{row['added_physical_overlap']:2d} "
            f"{row['added_graph_precision']:.3f} {row['tie_ranks']} "
            f"{row['elapsed']:.1f}s"
        )
        distances = tuple(
            tuple(parent - site for parent in parents)
            for site, parents in enumerate(row["parent_sets"])
        )
        print(f"  parent distances: {distances}")
    diagnostics = result["mps_diagnostics"]
    print(
        f"exact snake-MPS: max S={diagnostics['max_entropy']:.6f} "
        f"exact rank={diagnostics['max_exact_rank']} "
        f"D(1e-2,1e-4,1e-6)="
        f"{tuple(diagnostics['max_bond_dims'][tol] for tol in (1e-2, 1e-4, 1e-6))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--fixed-sweeps", type=int, default=8)
    parser.add_argument("--adaptive-cycles", type=int, default=6)
    parser.add_argument("--initial-sweeps", type=int, default=4)
    parser.add_argument("--branch-sweeps", type=int, default=2)
    parser.add_argument("--candidate-budget", type=int, default=1)
    parser.add_argument("--per-site-graph-candidates", type=int, default=1)
    parser.add_argument("--max-parents", type=int, default=4)
    parser.add_argument("--max-tie-rank", type=int, default=4)
    parser.add_argument("--adaptive-initial-rank", type=int, default=2)
    parser.add_argument("--probe-noise", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--modes", nargs="+", choices=_MODES, default=_MODES)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = benchmark_square_j1_j2(
        nrows=args.rows,
        ncols=args.cols,
        j1=args.j1,
        j2=args.j2,
        bond_dim=args.bond_dim,
        fixed_sweeps=args.fixed_sweeps,
        adaptive_cycles=args.adaptive_cycles,
        initial_sweeps=args.initial_sweeps,
        branch_sweeps=args.branch_sweeps,
        candidate_budget=args.candidate_budget,
        per_site_graph_candidates=args.per_site_graph_candidates,
        max_parents=args.max_parents,
        max_tie_rank=args.max_tie_rank,
        adaptive_initial_rank=args.adaptive_initial_rank,
        probe_noise=args.probe_noise,
        seed=args.seed,
        modes=args.modes,
        verbose=args.verbose,
    )
    _print_benchmark(result)


if __name__ == "__main__":
    main()
