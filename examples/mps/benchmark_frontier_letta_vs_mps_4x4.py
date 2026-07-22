"""Parameter-matched frontier LETTA versus MPS benchmark on a 4x4 lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    sparse_heisenberg_hamiltonian,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import FrontierTiedLETTA


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_4x4.json"


def _state_diagnostics(state, sparse_hamiltonian, exact_state, exact_energy):
    vector = state.state_vector(normalize=True)
    h_vector = sparse_hamiltonian @ vector
    energy = float(np.real(np.vdot(vector, h_vector)))
    residual = h_vector - energy * vector
    variance = float(np.real(np.vdot(residual, residual)))
    fidelity = float(abs(np.vdot(exact_state, vector)) ** 2)
    return {
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_error_per_site": (energy - exact_energy) / len(state.dims),
        "variance": variance,
        "residual_norm": float(np.sqrt(max(variance, 0.0))),
        "ground_state_fidelity": fidelity,
        "frontier_energy_discrepancy": energy - float(state.energy),
    }


def _mps_state_vector(state):
    factors = state.to_order(["lv", "p", "rv"]).factors
    environment = np.ones((1, 1), dtype=np.result_type(*factors))
    for factor in factors:
        environment = np.einsum(
            "ca,apb->cpb",
            environment,
            factor,
            optimize=True,
        ).reshape(-1, factor.shape[2])
    vector = environment[:, 0]
    return vector / np.linalg.norm(vector)


def _vector_diagnostics(vector, sparse_hamiltonian, exact_state, exact_energy):
    vector = np.asarray(vector) / np.linalg.norm(vector)
    h_vector = sparse_hamiltonian @ vector
    energy = float(np.real(np.vdot(vector, h_vector)))
    residual = h_vector - energy * vector
    variance = float(np.real(np.vdot(residual, residual)))
    fidelity = float(abs(np.vdot(exact_state, vector)) ** 2)
    return {
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_error_per_site": (energy - exact_energy) / 16,
        "variance": variance,
        "residual_norm": float(np.sqrt(max(variance, 0.0))),
        "ground_state_fidelity": fidelity,
    }


def _run_state(
    name,
    hamiltonian,
    sparse_hamiltonian,
    exact_state,
    exact_energy,
    parent_sets,
    *,
    bond_dim,
    seed,
    sweeps,
    tolerance,
    verbose,
):
    start = perf_counter()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        bond_dim=bond_dim,
        seed=seed,
        frontier_backend="compressed",
    )
    setup_seconds = perf_counter() - start
    initial_energy = float(state.energy)
    print(
        f"starting {name} seed={seed} parameters={state.nparameters} "
        f"E0={initial_energy:.10f}",
        flush=True,
    )
    start = perf_counter()
    state.run(
        nsweeps=sweeps,
        tol=tolerance,
        solver="direct",
        verbose=verbose,
    )
    optimization_seconds = perf_counter() - start
    diagnostics = _state_diagnostics(
        state,
        sparse_hamiltonian,
        exact_state,
        exact_energy,
    )
    print(
        f"finished {name} seed={seed} sweeps={len(state.history)} "
        f"E={diagnostics['energy']:.10f} "
        f"dE={diagnostics['energy_error']:.3e} "
        f"variance={diagnostics['variance']:.3e} "
        f"time={optimization_seconds:.2f}s",
        flush=True,
    )
    return {
        "name": name,
        "seed": int(seed),
        "bond_dim": int(bond_dim),
        "parameters": state.nparameters,
        "tie_edges": int(sum(map(len, parent_sets))),
        "converged": bool(state.converged),
        "sweeps_completed": len(state.history),
        "initial_energy": initial_energy,
        **diagnostics,
        "setup_seconds": float(setup_seconds),
        "optimization_seconds": float(optimization_seconds),
        "seconds_per_sweep": float(optimization_seconds / max(len(state.history), 1)),
        "peak_frontier_elements": state.peak_frontier_elements,
        "cached_environment_elements": state.cached_environment_elements,
        "final_delta_energy": (
            float(state.history[-1]["delta_energy"]) if state.history else 0.0
        ),
        "solver_failures": int(
            sum(record["solver_failures"] for record in state.history)
        ),
        "sweep_energies": [float(record["energy"]) for record in state.history],
    }


def _run_mps_d8(
    hamiltonian,
    sparse_hamiltonian,
    exact_state,
    exact_energy,
    *,
    seed,
    sweeps,
    tolerance,
    verbose,
):
    from pyqed.mps import DMRG, MPS, MPO

    nsites = len(hamiltonian.dims)
    max_bond_dim = 8
    ranks = tuple(
        min(max_bond_dim, 2 ** min(cut, nsites - cut)) for cut in range(nsites + 1)
    )
    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(size=(ranks[site], 2, ranks[site + 1]))
        / np.sqrt(ranks[site] * 2 * ranks[site + 1])
        for site in range(nsites)
    ]
    initial_state = MPS(factors, labels=["lv", "p", "rv"])
    parameters = int(sum(factor.size for factor in factors))
    initial_diagnostics = _vector_diagnostics(
        _mps_state_vector(initial_state),
        sparse_hamiltonian,
        exact_state,
        exact_energy,
    )
    mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    print(
        f"starting mps_d8 seed={seed} parameters={parameters} "
        f"E0={initial_diagnostics['energy']:.10f}",
        flush=True,
    )
    start = perf_counter()
    solver = DMRG(
        mpo,
        D=max_bond_dim,
        init_guess=initial_state,
        nsweeps=sweeps,
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=int(bool(verbose)),
        sweep_tol=tolerance,
        davidson_tol=min(tolerance, 1.0e-10),
        davidson_max_iter=100,
        noise=0.0,
        recenter_final=False,
        performance="generic",
    ).run()
    optimization_seconds = perf_counter() - start
    diagnostics = _vector_diagnostics(
        _mps_state_vector(solver.ground_state),
        sparse_hamiltonian,
        exact_state,
        exact_energy,
    )
    directional_history = [
        row for row in solver.sweep_history if row.get("direction") in {"lr", "rl"}
    ]
    energies = [float(row["energy"]) for row in directional_history]
    truncations = [
        float(row["truncation"])
        for row in directional_history
        if row.get("truncation") is not None
    ]
    print(
        f"finished mps_d8 seed={seed} passes={len(directional_history)} "
        f"E={diagnostics['energy']:.10f} "
        f"dE={diagnostics['energy_error']:.3e} "
        f"variance={diagnostics['variance']:.3e} "
        f"time={optimization_seconds:.2f}s",
        flush=True,
    )
    return {
        "name": "mps_d8",
        "seed": int(seed),
        "bond_dim": max_bond_dim,
        "bond_ranks": ranks,
        "parameters": parameters,
        "tie_edges": 0,
        "optimizer": "canonical-two-site-dmrg",
        "symmetry": "none",
        "converged": bool(solver.converged),
        "sweeps_completed": len(directional_history),
        "initial_energy": initial_diagnostics["energy"],
        **diagnostics,
        "frontier_energy_discrepancy": diagnostics["energy"] - float(solver.e_tot),
        "setup_seconds": 0.0,
        "optimization_seconds": float(optimization_seconds),
        "seconds_per_sweep": float(
            optimization_seconds / max(len(directional_history), 1)
        ),
        "peak_frontier_elements": None,
        "cached_environment_elements": None,
        "final_delta_energy": (
            abs(energies[-1] - energies[-2]) if len(energies) >= 2 else 0.0
        ),
        "solver_failures": 0,
        "maximum_truncation": max(truncations, default=0.0),
        "sweep_energies": energies,
    }


def _summary(runs):
    energies = np.asarray([run["energy"] for run in runs])
    variances = np.asarray([run["variance"] for run in runs])
    fidelities = np.asarray([run["ground_state_fidelity"] for run in runs])
    times = np.asarray([run["optimization_seconds"] for run in runs])
    best = int(np.argmin(energies))
    return {
        "runs": len(runs),
        "parameters": int(runs[0]["parameters"]),
        "best_seed": int(runs[best]["seed"]),
        "best_energy": float(energies[best]),
        "median_energy": float(np.median(energies)),
        "energy_range": [float(np.min(energies)), float(np.max(energies))],
        "energy_interquartile_range": [
            float(np.quantile(energies, 0.25)),
            float(np.quantile(energies, 0.75)),
        ],
        "energy_standard_deviation": float(np.std(energies)),
        "best_energy_error": float(runs[best]["energy_error"]),
        "median_variance": float(np.median(variances)),
        "best_ground_state_fidelity": float(np.max(fidelities)),
        "median_ground_state_fidelity": float(np.median(fidelities)),
        "median_optimization_seconds": float(np.median(times)),
        "converged_runs": int(sum(run["converged"] for run in runs)),
        "total_solver_failures": int(sum(run["solver_failures"] for run in runs)),
    }


def merge_results(paths):
    """Merge independently executed seed/ansatz result files."""
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not inputs:
        raise ValueError("at least one result file is required.")
    model = inputs[0]["model"]
    exact = inputs[0]["exact_reference"]
    runs = {}
    pass_limits = {}
    seeds = set()
    for result in inputs:
        if result["model"] != model:
            raise ValueError("cannot merge results for different models.")
        if not np.isclose(
            result["exact_reference"]["energy"],
            exact["energy"],
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("cannot merge inconsistent exact references.")
        for name, values in result["runs"].items():
            runs.setdefault(name, []).extend(values)
            pass_limits[name] = max(
                pass_limits.get(name, 0),
                int(result["settings"]["maximum_sweeps"]),
            )
            seeds.update(int(value["seed"]) for value in values)
    for values in runs.values():
        values.sort(key=lambda value: value["seed"])
    return {
        "model": model,
        "settings": {
            "directional_pass_limits": pass_limits,
            "tolerance": float(inputs[0]["settings"]["tolerance"]),
            "seeds": sorted(seeds),
            "solver": {
                "tied_letta_d4": "direct one-site generalized eigensolve",
                "mps_d8": "canonical two-site DMRG",
            },
            "frontier_backend": "compressed",
        },
        "exact_reference": exact,
        "runs": runs,
        "summary": {name: _summary(values) for name, values in runs.items()},
    }


def run_benchmark(
    *,
    sweeps=100,
    seeds=(3, 7, 11, 19, 23),
    tolerance=1.0e-10,
    j2=0.5,
    ansatz="both",
    verbose=False,
):
    nrows = ncols = 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((i, j, 1.0) for i, j in nearest)
    weighted_bonds += tuple((i, j, float(j2)) for i, j in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted_bonds)
    sparse_hamiltonian = sparse_heisenberg_hamiltonian(nsites, weighted_bonds)

    start = perf_counter()
    exact_values, exact_vectors = eigsh(
        sparse_hamiltonian,
        k=2,
        which="SA",
        tol=1.0e-12,
    )
    exact_seconds = perf_counter() - start
    exact_order = np.argsort(exact_values)
    exact_values = exact_values[exact_order]
    exact_vectors = exact_vectors[:, exact_order]
    exact_energy = float(exact_values[0])
    exact_state = exact_vectors[:, 0]

    candidates = []
    if ansatz in {"both", "tied_letta_d4"}:
        candidates.append(
            (
                "tied_letta_d4",
                parent_sets_from_edges(nsites, nearest),
                4,
            )
        )
    if ansatz in {"both", "mps_d8"}:
        candidates.append(("mps_d8", None, 8))

    runs = {name: [] for name, _parents, _bond_dim in candidates}
    for seed in tuple(int(seed) for seed in seeds):
        for name, parent_sets, bond_dim in candidates:
            if name == "mps_d8":
                result = _run_mps_d8(
                    hamiltonian,
                    sparse_hamiltonian,
                    exact_state,
                    exact_energy,
                    seed=seed,
                    sweeps=sweeps,
                    tolerance=tolerance,
                    verbose=verbose,
                )
            else:
                result = _run_state(
                    name,
                    hamiltonian,
                    sparse_hamiltonian,
                    exact_state,
                    exact_energy,
                    parent_sets,
                    bond_dim=bond_dim,
                    seed=seed,
                    sweeps=sweeps,
                    tolerance=tolerance,
                    verbose=verbose,
                )
            runs[name].append(result)

    return {
        "model": {
            "nrows": nrows,
            "ncols": ncols,
            "j1": 1.0,
            "j2": float(j2),
            "boundary": "open",
            "site_order": "row-wise-snake",
        },
        "settings": {
            "maximum_sweeps": int(sweeps),
            "tolerance": float(tolerance),
            "seeds": [int(seed) for seed in seeds],
            "solver": "direct",
            "frontier_backend": "compressed",
            "ansatz": ansatz,
        },
        "exact_reference": {
            "energy": exact_energy,
            "first_excited_energy": float(exact_values[1]),
            "gap": float(exact_values[1] - exact_values[0]),
            "seconds": float(exact_seconds),
            "used_during_optimization": False,
        },
        "runs": runs,
        "summary": {name: _summary(values) for name, values in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 7, 11, 19, 23))
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument(
        "--ansatz",
        choices=("both", "tied_letta_d4", "mps_d8"),
        default="both",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-results", type=Path, nargs="+")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.merge_results:
        result = merge_results(args.merge_results)
    else:
        result = run_benchmark(
            sweeps=args.sweeps,
            seeds=args.seeds,
            tolerance=args.tolerance,
            j2=args.j2,
            ansatz=args.ansatz,
            verbose=args.verbose,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
