"""Compare CP-compressed and unrestricted LETTA on the 3x4 J1-J2 graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

from examples.mps.adaptive_cp_letta_j1j2_square import (
    sparse_heisenberg_hamiltonian,
)
from pyqed.letta import CPTiedLETTA, DenseTiedLETTA


HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = (
    HERE / "results" / "adaptive_cp_letta_j1j2_square_D4_joint.json"
)
DEFAULT_OUTPUT = HERE / "results" / "dense_vs_cp_letta_j1j2_square_D4.json"


def compare_dense_to_cp(
    graph_data,
    *,
    cp_sweeps: int = 30,
    dense_sweeps: int = 12,
    tol: float = 1.0e-10,
    metric_tol: float = 1.0e-12,
    seed: int = 7,
    verbose: bool = False,
) -> dict[str, object]:
    """Optimize CP, expand it exactly, and relax unrestricted tensors."""
    model = graph_data["model"]
    nrows = int(model["nrows"])
    ncols = int(model["ncols"])
    nsites = nrows * ncols
    dims = (2,) * nsites
    j1 = float(model["j1"])
    j2 = float(model["j2"])
    nearest = tuple(tuple(map(int, edge)) for edge in graph_data["nearest_bonds"])
    diagonals = tuple(tuple(map(int, edge)) for edge in graph_data["diagonal_bonds"])
    weighted_bonds = tuple((left, right, j1) for left, right in nearest) + tuple(
        (left, right, j2) for left, right in diagonals
    )
    hamiltonian = sparse_heisenberg_hamiltonian(nsites, weighted_bonds)
    parents = tuple(tuple(map(int, item)) for item in graph_data["parent_sets"])
    tie_ranks = tuple(map(int, graph_data["tie_ranks"]))
    bond_dim = int(graph_data["settings"]["bond_dim"])

    cp_state = CPTiedLETTA(
        hamiltonian,
        dims,
        parents,
        bond_dim=bond_dim,
        tie_ranks=tie_ranks,
        seed=seed,
    )
    cp_start = perf_counter()
    cp_state.run(
        nsweeps=cp_sweeps,
        tol=tol,
        metric_tol=metric_tol,
        verbose=verbose,
    )
    cp_elapsed = perf_counter() - cp_start
    cp_vector = cp_state.state_vector(normalize=True)
    cp_energy, _cp_residual, cp_residual_norm = cp_state.energy_residual()

    dense_state = DenseTiedLETTA.from_cp(cp_state)
    dense_initial_vector = dense_state.state_vector(normalize=True)
    expansion_error = float(np.linalg.norm(dense_initial_vector - cp_vector))
    expansion_fidelity = float(abs(np.vdot(cp_vector, dense_initial_vector)) ** 2)
    dense_initial_energy = dense_state.energy
    dense_start = perf_counter()
    dense_state.run(
        nsweeps=dense_sweeps,
        tol=tol,
        metric_tol=metric_tol,
        verbose=verbose,
    )
    dense_elapsed = perf_counter() - dense_start
    dense_vector = dense_state.state_vector(normalize=True)
    dense_energy, _dense_residual, dense_residual_norm = dense_state.energy_residual()

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

    def summary(state, vector, energy, residual_norm, elapsed):
        return {
            "energy": float(energy),
            "energy_error": float(energy - eigenvalues[0]),
            "fidelity": float(abs(np.vdot(exact, vector)) ** 2),
            "residual_norm": float(residual_norm),
            "parameters": int(state.nparameters),
            "elapsed_seconds": float(elapsed),
            "sweep_energies": [
                float(record["energy"]) for record in state.history
            ],
        }

    return {
        "model": {
            "nrows": nrows,
            "ncols": ncols,
            "j1": j1,
            "j2": j2,
            "boundary": model.get("boundary", "open"),
        },
        "settings": {
            "bond_dim": bond_dim,
            "parent_sets": parents,
            "cp_tie_ranks": tie_ranks,
            "cp_sweeps": int(cp_sweeps),
            "dense_sweeps": int(dense_sweeps),
            "tol": float(tol),
            "metric_tol": float(metric_tol),
            "seed": int(seed),
        },
        "exact_energy": float(eigenvalues[0]),
        "gap": float(eigenvalues[1] - eigenvalues[0]),
        "cp": summary(
            cp_state,
            cp_vector,
            cp_energy,
            cp_residual_norm,
            cp_elapsed,
        ),
        "dense": summary(
            dense_state,
            dense_vector,
            dense_energy,
            dense_residual_norm,
            dense_elapsed,
        ),
        "cp_to_dense": {
            "initial_energy": float(dense_initial_energy),
            "state_vector_error": expansion_error,
            "fidelity": expansion_fidelity,
            "energy_improvement": float(cp_energy - dense_energy),
        },
    }


def _print_result(result) -> None:
    print(
        f"{result['model']['nrows']}x{result['model']['ncols']} "
        f"J1={result['model']['j1']:g} J2={result['model']['j2']:g} "
        f"E0={result['exact_energy']:.12f}"
    )
    print("ansatz params energy dE fidelity residual time")
    for name in ("cp", "dense"):
        row = result[name]
        print(
            f"{name:>6s} {row['parameters']:5d} {row['energy']:.12f} "
            f"{row['energy_error']:.6e} {row['fidelity']:.8f} "
            f"{row['residual_norm']:.3e} {row['elapsed_seconds']:.1f}s"
        )
    bridge = result["cp_to_dense"]
    print(
        f"CP->dense state error={bridge['state_vector_error']:.3e} "
        f"fidelity={bridge['fidelity']:.12f} "
        f"energy gain={bridge['energy_improvement']:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cp-sweeps", type=int, default=30)
    parser.add_argument("--dense-sweeps", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    with args.graph.open(encoding="utf-8") as handle:
        graph_data = json.load(handle)
    result = compare_dense_to_cp(
        graph_data,
        cp_sweeps=args.cp_sweeps,
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
