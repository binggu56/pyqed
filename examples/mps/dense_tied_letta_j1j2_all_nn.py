"""Unrestricted LETTA with every square-lattice nearest neighbor tied."""

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
from pyqed.letta import DenseTiedLETTA


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "dense_letta_j1j2_square_D4_all_nn.json"


def benchmark_all_nearest_neighbors(
    *,
    nrows: int = 3,
    ncols: int = 4,
    j1: float = 1.0,
    j2: float = 0.5,
    bond_dim: int = 4,
    dense_sweeps: int = 80,
    tol: float = 1.0e-10,
    metric_tol: float = 1.0e-12,
    seed: int = 7,
    verbose: bool = False,
) -> dict[str, object]:
    """Optimize a dense tied ansatz on the complete J1 nearest-neighbor graph."""
    nrows = int(nrows)
    ncols = int(ncols)
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted_bonds = tuple((left, right, float(j1)) for left, right in nearest)
    weighted_bonds += tuple(
        (left, right, float(j2)) for left, right in diagonals
    )
    hamiltonian = sparse_heisenberg_hamiltonian(nsites, weighted_bonds)
    parent_sets = parent_sets_from_edges(nsites, nearest)

    state = DenseTiedLETTA(
        hamiltonian,
        (2,) * nsites,
        parent_sets,
        bond_dim=bond_dim,
        seed=seed,
    )
    initial_energy = state.energy
    start = perf_counter()
    state.run(
        nsweeps=dense_sweeps,
        tol=tol,
        metric_tol=metric_tol,
        verbose=verbose,
    )
    elapsed = perf_counter() - start
    vector = state.state_vector(normalize=True)
    energy, _residual, residual_norm = state.energy_residual()

    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=2,
        which="SA",
        tol=1.0e-11,
        maxiter=100_000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues)[order]
    exact = np.asarray(eigenvectors)[:, order[0]]
    sweep_energies = [float(record["energy"]) for record in state.history]
    return {
        "model": {
            "nrows": nrows,
            "ncols": ncols,
            "j1": float(j1),
            "j2": float(j2),
            "boundary": "open",
        },
        "ansatz": {
            "kind": "dense-tied-letta",
            "tie_graph": "all-j1-nearest-neighbor-bonds",
            "bond_dim": int(bond_dim),
            "parent_sets": parent_sets,
            "tie_edges": int(sum(map(len, parent_sets))),
            "parameters": state.nparameters,
        },
        "settings": {
            "dense_sweeps": int(dense_sweeps),
            "tol": float(tol),
            "metric_tol": float(metric_tol),
            "seed": int(seed),
        },
        "exact_energy": float(eigenvalues[0]),
        "gap": float(eigenvalues[1] - eigenvalues[0]),
        "initial_energy": float(initial_energy),
        "energy": float(energy),
        "energy_error": float(energy - eigenvalues[0]),
        "fidelity": float(abs(np.vdot(exact, vector)) ** 2),
        "residual_norm": float(residual_norm),
        "elapsed_seconds": float(elapsed),
        "converged": bool(state.converged),
        "last_sweep_delta": (
            float(state.history[-1]["delta_energy"]) if state.history else 0.0
        ),
        "sweep_energies": sweep_energies,
    }


def _print_result(result) -> None:
    model = result["model"]
    ansatz = result["ansatz"]
    print(
        f"{model['nrows']}x{model['ncols']} OBC J1={model['j1']:g} "
        f"J2={model['j2']:g} D={ansatz['bond_dim']} "
        f"ties={ansatz['tie_edges']} params={ansatz['parameters']}"
    )
    print(
        f"E={result['energy']:.12f} E0={result['exact_energy']:.12f} "
        f"dE={result['energy_error']:.6e} fidelity={result['fidelity']:.8f} "
        f"residual={result['residual_norm']:.3e} "
        f"last-dE={result['last_sweep_delta']:.3e} "
        f"time={result['elapsed_seconds']:.1f}s"
    )
    print(f"parent sets: {result['ansatz']['parent_sets']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--dense-sweeps", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = benchmark_all_nearest_neighbors(
        nrows=args.rows,
        ncols=args.cols,
        j1=args.j1,
        j2=args.j2,
        bond_dim=args.bond_dim,
        dense_sweeps=args.dense_sweeps,
        tol=args.tol,
        metric_tol=args.metric_tol,
        seed=args.seed,
        verbose=args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_result(result)
    print(args.output)


if __name__ == "__main__":
    main()
