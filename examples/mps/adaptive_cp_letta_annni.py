"""CP-rank and parent-graph benchmark for dense-reference LETTA on ANNNI."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from examples.mps.adaptive_physical_tying_annni import annni_dense
from pyqed.letta.cp import cp_als
from pyqed.letta.cp_tying import CPTiedLETTA
from pyqed.letta.physical_tying import compress_physical_ties, fixed_range_parent_sets


def _reduced_density(state, dims, sites) -> np.ndarray:
    sites = tuple(int(site) for site in sites)
    complement = tuple(site for site in range(len(dims)) if site not in sites)
    kept_dim = int(np.prod([dims[site] for site in sites]))
    matrix = np.asarray(state).reshape(dims).transpose(sites + complement).reshape(kept_dim, -1)
    return matrix @ matrix.T.conj()


def _entropy(density) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (density + density.T.conj())).real
    eigenvalues = eigenvalues[eigenvalues > 64.0 * np.finfo(float).eps]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def quantum_mutual_information_parent_sets(state, dims, max_parents: int):
    """Select future parents by pair quantum mutual information."""
    dims = tuple(int(dim) for dim in dims)
    max_parents = int(max_parents)
    if max_parents < 0:
        raise ValueError("max_parents must be nonnegative.")
    single_entropies = [
        _entropy(_reduced_density(state, dims, (site,)))
        for site in range(len(dims))
    ]
    parent_sets = []
    for site in range(len(dims) - 1):
        scores = []
        for parent in range(site + 1, len(dims)):
            pair_entropy = _entropy(_reduced_density(state, dims, (site, parent)))
            information = single_entropies[site] + single_entropies[parent] - pair_entropy
            scores.append((information, parent))
        scores.sort(key=lambda item: (-item[0], item[1]))
        selected = tuple(sorted(parent for _score, parent in scores[:max_parents]))
        parent_sets.append(selected)
    return tuple(parent_sets)


def _parent_profile(parent_sets) -> str:
    fields = []
    for site, parents in enumerate(parent_sets):
        distances = tuple(parent - site for parent in parents)
        fields.append("+".join(str(distance) for distance in distances) or "-")
    return ",".join(fields)


def _subspace_fidelity(state, eigenvectors) -> float:
    state = np.asarray(state).reshape(-1)
    state = state / np.linalg.norm(state)
    return float(np.sum(np.abs(eigenvectors.T.conj() @ state) ** 2).real)


def _full_entries(compressed) -> int:
    return int(sum(factor.size for factor in compressed.factors) + compressed.terminal.size)


def _full_virtual_entries(dims, parent_sets, bond_dim: int) -> int:
    parent_sets = tuple(parent_sets) + ((),)
    bonds = (1,) + (int(bond_dim),) * max(0, len(dims) - 1) + (1,)
    return int(
        sum(
            bonds[site]
            * bonds[site + 1]
            * dims[site]
            * int(np.prod([dims[parent] for parent in parents], dtype=int))
            for site, parents in enumerate(parent_sets)
        )
    )


def _cp_errors(compressed, rank: int, *, seed: int) -> tuple[float, float]:
    factors = (*compressed.factors, compressed.terminal)
    errors = [
        cp_als(factor, rank, max_iter=300, tol=1.0e-11, seed=seed + site).relative_error
        for site, factor in enumerate(factors)
    ]
    return float(max(errors)), float(np.sqrt(np.mean(np.square(errors))))


def select_adaptive_tie_ranks(
    compressed,
    hamiltonian,
    target_state,
    *,
    candidate_ranks=(1, 2, 4, 8),
    objective: str = "fidelity",
    gain_tolerance: float = 1.0e-8,
    max_steps: int | None = None,
    seed: int = 7,
):
    """Grow individual tie ranks until the selected seed objective stalls.

    Rank selection uses the CP-compressed physical-only seed (``D=1``), so it
    measures the benefit of physical-tie rank without mixing in virtual-bond
    improvements.  The selected profile can subsequently be relaxed at any
    virtual bond dimension.
    """
    rank_grid = tuple(sorted({int(rank) for rank in candidate_ranks}))
    if not rank_grid or rank_grid[0] < 1:
        raise ValueError("candidate_ranks must contain positive integers.")
    objective = str(objective).lower()
    if objective not in {"fidelity", "energy"}:
        raise ValueError("objective must be 'fidelity' or 'energy'.")
    gain_tolerance = float(gain_tolerance)
    if gain_tolerance < 0.0 or not np.isfinite(gain_tolerance):
        raise ValueError("gain_tolerance must be finite and nonnegative.")
    if max_steps is None:
        max_steps = len(compressed.dims) * max(0, len(rank_grid) - 1)
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative.")
    sparse_hamiltonian = csr_matrix(hamiltonian)
    ranks = [rank_grid[0]] * len(compressed.dims)

    def evaluate(profile):
        ansatz = CPTiedLETTA.from_physical_tie_state(
            compressed,
            sparse_hamiltonian,
            tie_ranks=profile,
            bond_dim=1,
            virtual_noise=0.0,
            seed=seed,
        )
        if objective == "fidelity":
            score = ansatz.fidelity(target_state)
        else:
            score = -ansatz.expectation()
        return float(score), ansatz.nparameters

    score, parameters = evaluate(ranks)
    history = [
        {
            "step": 0,
            "site": None,
            "ranks": tuple(ranks),
            "score": score,
            "parameters": parameters,
            "gain": 0.0,
        }
    ]
    for step in range(1, max_steps + 1):
        trials = []
        for site, rank in enumerate(ranks):
            larger = [candidate for candidate in rank_grid if candidate > rank]
            if not larger:
                continue
            trial_ranks = list(ranks)
            trial_ranks[site] = larger[0]
            trial_score, trial_parameters = evaluate(trial_ranks)
            gain = trial_score - score
            added = max(1, trial_parameters - parameters)
            trials.append(
                (
                    gain / added,
                    gain,
                    -site,
                    site,
                    tuple(trial_ranks),
                    trial_score,
                    trial_parameters,
                )
            )
        if not trials:
            break
        improving_trials = [trial for trial in trials if trial[1] > gain_tolerance]
        if not improving_trials:
            break
        (
            _gain_per_parameter,
            gain,
            _negative_site,
            site,
            profile,
            trial_score,
            trial_parameters,
        ) = max(improving_trials)
        ranks = list(profile)
        score = trial_score
        parameters = trial_parameters
        history.append(
            {
                "step": step,
                "site": site,
                "ranks": tuple(ranks),
                "score": score,
                "parameters": parameters,
                "gain": gain,
            }
        )
    return tuple(ranks), history


def benchmark_cp_rank_sweep(
    *,
    nsites: int = 10,
    nearest: float = 1.0,
    next_nearest: float = 2.0,
    field: float = 0.8,
    max_parents: int = 4,
    bond_dim: int = 2,
    ranks=(1, 2, 4, 8),
    selectors=("fixed", "gram", "qmi"),
    variational_sweeps: int = 10,
    virtual_noise: float = 1.0e-4,
    seed: int = 7,
) -> list[dict[str, object]]:
    """Compare physical-parent selectors and CP ranks at one ANNNI point."""
    nsites = int(nsites)
    dims = (2,) * nsites
    hamiltonian = annni_dense(
        nsites,
        nearest=nearest,
        next_nearest=next_nearest,
        field=field,
    )
    eigenvalues, eigenvectors = eigh(
        hamiltonian,
        subset_by_index=(0, min(1, hamiltonian.shape[0] - 1)),
        driver="evr",
        check_finite=False,
    )
    exact_energy = float(eigenvalues[0])
    exact_state = eigenvectors[:, 0]
    sparse_hamiltonian = csr_matrix(hamiltonian)

    graphs = {}
    for selector in selectors:
        selector = str(selector).lower()
        if selector == "fixed":
            graphs[selector] = fixed_range_parent_sets(nsites, max_parents)
        elif selector == "gram":
            graphs[selector] = compress_physical_ties(
                exact_state,
                dims,
                max_parents=max_parents,
                relative_tolerance=0.0,
            ).parent_sets
        elif selector == "qmi":
            graphs[selector] = quantum_mutual_information_parent_sets(
                exact_state,
                dims,
                max_parents,
            )
        else:
            raise ValueError("selectors must contain only 'fixed', 'gram', or 'qmi'.")

    rows = []
    for selector, parent_sets in graphs.items():
        compressed = compress_physical_ties(
            exact_state,
            dims,
            parent_sets=parent_sets,
        )
        compressed_vector = compressed.state_vector(normalize=True)
        compressed_energy = float(
            np.vdot(compressed_vector, hamiltonian @ compressed_vector).real
        )
        common = {
            "selector": selector,
            "j2": float(next_nearest),
            "e0": exact_energy,
            "gap": float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else np.nan,
            "bond_dim": int(bond_dim),
            "max_parents": int(max_parents),
            "parent_profile": _parent_profile(parent_sets),
            "full_entries": _full_entries(compressed),
            "full_virtual_entries": _full_virtual_entries(
                dims,
                parent_sets,
                bond_dim,
            ),
            "compression_energy_error": compressed_energy - exact_energy,
            "compression_fidelity": compressed.fidelity(exact_state),
        }
        for rank in ranks:
            rank = int(rank)
            max_cp_error, rms_cp_error = _cp_errors(compressed, rank, seed=seed)
            ansatz = CPTiedLETTA.from_physical_tie_state(
                compressed,
                sparse_hamiltonian,
                tie_ranks=rank,
                bond_dim=bond_dim,
                virtual_noise=virtual_noise,
                seed=seed,
            )
            initial_energy = float(ansatz.energy)
            initial_fidelity = ansatz.fidelity(exact_state)
            ansatz.run(nsweeps=variational_sweeps, tol=1.0e-10)
            vector = ansatz.state_vector(normalize=True)
            rows.append(
                {
                    **common,
                    "rank": rank,
                    "parameters": ansatz.nparameters,
                    "max_local_cp_error": max_cp_error,
                    "rms_local_cp_error": rms_cp_error,
                    "initial_energy_error": initial_energy - exact_energy,
                    "initial_fidelity": initial_fidelity,
                    "energy_error": float(ansatz.energy - exact_energy),
                    "fidelity": ansatz.fidelity(exact_state),
                    "doublet_fidelity": _subspace_fidelity(vector, eigenvectors),
                    "completed_sweeps": len(ansatz.history),
                }
            )
    return rows


def benchmark_adaptive_rank_profile(
    *,
    nsites: int = 10,
    nearest: float = 1.0,
    next_nearest: float = 2.0,
    field: float = 0.8,
    max_parents: int = 4,
    selector: str = "fixed",
    bond_dim: int = 2,
    candidate_ranks=(1, 2, 4, 8),
    rank_objective: str = "fidelity",
    rank_gain_tolerance: float = 1.0e-8,
    variational_sweeps: int = 30,
    virtual_noise: float = 1.0e-4,
    seed: int = 7,
) -> dict[str, object]:
    """Select site-dependent CP ranks, then jointly optimize CP and virtual blocks."""
    dims = (2,) * int(nsites)
    hamiltonian = annni_dense(
        nsites,
        nearest=nearest,
        next_nearest=next_nearest,
        field=field,
    )
    eigenvalues, eigenvectors = eigh(
        hamiltonian,
        subset_by_index=(0, 1),
        driver="evr",
        check_finite=False,
    )
    exact_state = eigenvectors[:, 0]
    selector = str(selector).lower()
    if selector == "fixed":
        parent_sets = fixed_range_parent_sets(nsites, max_parents)
    elif selector == "gram":
        parent_sets = compress_physical_ties(
            exact_state,
            dims,
            max_parents=max_parents,
            relative_tolerance=0.0,
        ).parent_sets
    elif selector == "qmi":
        parent_sets = quantum_mutual_information_parent_sets(
            exact_state,
            dims,
            max_parents,
        )
    else:
        raise ValueError("selector must be 'fixed', 'gram', or 'qmi'.")
    compressed = compress_physical_ties(
        exact_state,
        dims,
        parent_sets=parent_sets,
    )
    ranks, rank_history = select_adaptive_tie_ranks(
        compressed,
        hamiltonian,
        exact_state,
        candidate_ranks=candidate_ranks,
        objective=rank_objective,
        gain_tolerance=rank_gain_tolerance,
        seed=seed,
    )
    ansatz = CPTiedLETTA.from_physical_tie_state(
        compressed,
        csr_matrix(hamiltonian),
        tie_ranks=ranks,
        bond_dim=bond_dim,
        virtual_noise=virtual_noise,
        seed=seed,
    )
    initial_energy = float(ansatz.energy)
    initial_fidelity = ansatz.fidelity(exact_state)
    ansatz.run(nsweeps=variational_sweeps, tol=1.0e-10)
    vector = ansatz.state_vector(normalize=True)
    return {
        "selector": selector,
        "j2": float(next_nearest),
        "e0": float(eigenvalues[0]),
        "gap": float(eigenvalues[1] - eigenvalues[0]),
        "bond_dim": int(bond_dim),
        "ranks": ranks,
        "rank_steps": len(rank_history) - 1,
        "rank_selection_score": rank_history[-1]["score"],
        "rank_history": rank_history,
        "parameters": ansatz.nparameters,
        "full_entries": _full_entries(compressed),
        "full_virtual_entries": _full_virtual_entries(
            dims,
            parent_sets,
            bond_dim,
        ),
        "parent_profile": _parent_profile(parent_sets),
        "initial_energy_error": initial_energy - eigenvalues[0],
        "initial_fidelity": initial_fidelity,
        "energy_error": float(ansatz.energy - eigenvalues[0]),
        "fidelity": ansatz.fidelity(exact_state),
        "doublet_fidelity": _subspace_fidelity(vector, eigenvectors),
        "completed_sweeps": len(ansatz.history),
    }


def benchmark_residual_adaptive(
    *,
    nsites: int = 10,
    nearest: float = 1.0,
    next_nearest: float = 2.0,
    field: float = 0.8,
    initial_tie_range: int = 1,
    initial_tie_rank: int = 1,
    bond_dim: int = 2,
    max_parents: int = 4,
    max_tie_rank: int = 4,
    ncycles: int = 12,
    initial_sweeps: int = 4,
    branch_sweeps: int = 3,
    candidate_budget: int = 6,
    per_site_graph_candidates: int = 2,
    probe_noise: float = 1.0e-2,
    seed: int = 7,
) -> dict[str, object]:
    """Run residual adaptation without using an exact state for any proposal.

    Exact diagonalization is deliberately performed only after the adaptive
    solve, and is used solely to report benchmark errors and fidelities.
    """
    nsites = int(nsites)
    dims = (2,) * nsites
    hamiltonian = annni_dense(
        nsites,
        nearest=nearest,
        next_nearest=next_nearest,
        field=field,
    )
    ansatz = CPTiedLETTA(
        csr_matrix(hamiltonian),
        dims,
        fixed_range_parent_sets(nsites, initial_tie_range),
        tie_ranks=initial_tie_rank,
        bond_dim=bond_dim,
        seed=seed,
    )
    ansatz.run_residual_adaptive(
        max_parents=max_parents,
        max_tie_rank=max_tie_rank,
        ncycles=ncycles,
        initial_sweeps=initial_sweeps,
        branch_sweeps=branch_sweeps,
        candidate_budget=candidate_budget,
        per_site_graph_candidates=per_site_graph_candidates,
        probe_noise=probe_noise,
        seed=seed,
    )

    # Reference data enter only after the variational solve is finished.
    eigenvalues, eigenvectors = eigh(
        hamiltonian,
        subset_by_index=(0, 1),
        driver="evr",
        check_finite=False,
    )
    vector = ansatz.state_vector(normalize=True)
    return {
        "j2": float(next_nearest),
        "energy": float(ansatz.energy),
        "e0": float(eigenvalues[0]),
        "energy_error": float(ansatz.energy - eigenvalues[0]),
        "gap": float(eigenvalues[1] - eigenvalues[0]),
        "fidelity": ansatz.fidelity(eigenvectors[:, 0]),
        "doublet_fidelity": _subspace_fidelity(vector, eigenvectors),
        "residual_norm": ansatz.energy_residual()[2],
        "parameters": ansatz.nparameters,
        "parent_sets": ansatz.parent_sets,
        "parent_profile": _parent_profile(ansatz.parent_sets[:-1]),
        "tie_ranks": ansatz.tie_ranks,
        "adaptive_history": ansatz.adaptive_history,
    }


def _print_rows(rows) -> None:
    print(
        "selector rank params init_dE final_dE fidelity doublet_F "
        "max_CP_error sweeps"
    )
    for row in rows:
        print(
            f"{row['selector']:>8s} {row['rank']:4d} {row['parameters']:6d} "
            f"{row['initial_energy_error']: .6e} {row['energy_error']: .6e} "
            f"{row['fidelity']: .8f} {row['doublet_fidelity']: .8f} "
            f"{row['max_local_cp_error']: .3e} {row['completed_sweeps']:3d}"
        )
    for selector in dict.fromkeys(row["selector"] for row in rows):
        row = next(item for item in rows if item["selector"] == selector)
        print(f"{selector} parents: {row['parent_profile']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=10)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=2.0)
    parser.add_argument("--field", type=float, default=0.8)
    parser.add_argument("--max-parents", type=int, default=4)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--ranks", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--selectors", nargs="+", default=("fixed", "gram", "qmi"))
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--virtual-noise", type=float, default=1.0e-4)
    parser.add_argument("--probe-noise", type=float, default=1.0e-2)
    parser.add_argument("--candidate-budget", type=int, default=6)
    parser.add_argument("--per-site-graph-candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adaptive-ranks", action="store_true")
    parser.add_argument("--rank-objective", choices=("fidelity", "energy"), default="fidelity")
    parser.add_argument("--residual-adaptive", action="store_true")
    args = parser.parse_args()
    if args.residual_adaptive:
        adaptive = benchmark_residual_adaptive(
            nsites=args.nsites,
            nearest=args.j1,
            next_nearest=args.j2,
            field=args.field,
            initial_tie_range=1,
            initial_tie_rank=1,
            bond_dim=args.bond_dim,
            max_parents=args.max_parents,
            max_tie_rank=max(args.ranks),
            ncycles=args.sweeps,
            initial_sweeps=4,
            branch_sweeps=3,
            candidate_budget=args.candidate_budget,
            per_site_graph_candidates=args.per_site_graph_candidates,
            probe_noise=args.probe_noise,
            seed=args.seed,
        )
        print(
            "residual-adaptive "
            f"params={adaptive['parameters']} dE={adaptive['energy_error']:.6e} "
            f"F={adaptive['fidelity']:.8f} F2={adaptive['doublet_fidelity']:.8f} "
            f"||r||={adaptive['residual_norm']:.3e} "
            f"ranks={adaptive['tie_ranks']}"
        )
        print(f"parents: {adaptive['parent_profile']}")
        return
    rows = benchmark_cp_rank_sweep(
        nsites=args.nsites,
        nearest=args.j1,
        next_nearest=args.j2,
        field=args.field,
        max_parents=args.max_parents,
        bond_dim=args.bond_dim,
        ranks=args.ranks,
        selectors=args.selectors,
        variational_sweeps=args.sweeps,
        virtual_noise=args.virtual_noise,
        seed=args.seed,
    )
    _print_rows(rows)
    if args.adaptive_ranks:
        adaptive = benchmark_adaptive_rank_profile(
            nsites=args.nsites,
            nearest=args.j1,
            next_nearest=args.j2,
            field=args.field,
            max_parents=args.max_parents,
            selector=args.selectors[0],
            bond_dim=args.bond_dim,
            candidate_ranks=args.ranks,
            rank_objective=args.rank_objective,
            variational_sweeps=args.sweeps,
            virtual_noise=args.virtual_noise,
            seed=args.seed,
        )
        print(
            "adaptive "
            f"selector={adaptive['selector']} ranks={adaptive['ranks']} "
            f"params={adaptive['parameters']} dE={adaptive['energy_error']:.6e} "
            f"F={adaptive['fidelity']:.8f} F2={adaptive['doublet_fidelity']:.8f} "
            f"steps={adaptive['rank_steps']}"
        )


if __name__ == "__main__":
    main()
