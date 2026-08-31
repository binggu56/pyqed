#!/usr/bin/env python3
"""Dense 2x2-plaquette LETTA benchmark for the open 4x4 J1-J2 model."""

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
from examples.mps.continue_frontier_letta_block_sparse_6x6 import _write_json
from pyqed.letta import (
    LocalHamiltonian,
    LocalTerm,
    frontier_tied_letta_from_mps,
)
from pyqed.mps import DMRG, MPS, MPO


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "full_plaquette_letta_4x4.json"
PLAQUETTES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # A
    ((0, 2), (0, 3), (1, 2), (1, 3)),  # B
    ((2, 0), (2, 1), (3, 0), (3, 1)),  # C
    ((2, 2), (2, 3), (3, 2), (3, 3)),  # D
)
# The macro tree is the path B-A-C-D.  Its missing square edge is B-D.
MACRO_ORDER = (1, 0, 2, 3)


def _embedded_spin(component, position):
    identity = np.eye(2)
    factors = [identity] * 4
    factors[position] = component
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def plaquette_hamiltonian(*, j2=0.5):
    """Return the exact spin Hamiltonian in four dense plaquette variables."""
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    components = (sx, sy, sz)
    coordinate_data = {
        coordinate: (plaquette, local)
        for plaquette, coordinates in enumerate(PLAQUETTES)
        for local, coordinate in enumerate(coordinates)
    }
    macro_position = {
        plaquette: position for position, plaquette in enumerate(MACRO_ORDER)
    }
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    snake = []
    for row in range(4):
        columns = range(4) if row % 2 == 0 else range(3, -1, -1)
        snake.extend((row, column) for column in columns)

    local = [np.zeros((16, 16), dtype=complex) for _ in range(4)]
    pair = {}
    for edges, coupling in ((nearest, 1.0), (diagonals, float(j2))):
        for left, right in edges:
            left_block, left_local = coordinate_data[snake[left]]
            right_block, right_local = coordinate_data[snake[right]]
            if left_block == right_block:
                target = local[macro_position[left_block]]
                for component in components:
                    target += coupling * (
                        _embedded_spin(component, left_local)
                        @ _embedded_spin(component, right_local)
                    )
                continue
            left_macro = macro_position[left_block]
            right_macro = macro_position[right_block]
            if left_macro > right_macro:
                left_macro, right_macro = right_macro, left_macro
                left_local, right_local = right_local, left_local
            target = pair.setdefault(
                (left_macro, right_macro),
                np.zeros((256, 256), dtype=complex),
            )
            for component in components:
                target += coupling * np.kron(
                    _embedded_spin(component, left_local),
                    _embedded_spin(component, right_local),
                )
    terms = [
        LocalTerm((site,), operator)
        for site, operator in enumerate(local)
        if np.any(operator)
    ]
    terms.extend(LocalTerm(sites, operator) for sites, operator in sorted(pair.items()))
    return LocalHamiltonian((16,) * 4, terms)


def _neel_plaquette_mps():
    factors = []
    for plaquette in MACRO_ORDER:
        bits = tuple((row + column) % 2 for row, column in PLAQUETTES[plaquette])
        physical = int(np.ravel_multi_index(bits, (2,) * 4))
        tensor = np.zeros((1, 16, 1), dtype=complex)
        tensor[0, physical, 0] = 1.0
        factors.append(tensor)
    return MPS(factors, labels=["lv", "p", "rv"])


def _directional_history(solver):
    return [
        row
        for row in solver.sweep_history
        if row.get("direction") in {"lr", "rl"}
    ]


def _run_block_mps(mpo, *, bond_dim, sweeps, tolerance):
    start = perf_counter()
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=_neel_plaquette_mps(),
        nsweeps=int(sweeps),
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=0,
        sweep_tol=float(tolerance),
        davidson_tol=min(float(tolerance), 1.0e-10),
        davidson_max_iter=200,
        noise=0.0,
        recenter_final=False,
        performance="reference",
    ).run()
    rows = _directional_history(solver)
    factors = solver.ground_state.to_order(["lv", "p", "rv"]).factors
    return solver, factors, {
        "energy": float(solver.e_tot),
        "seconds": perf_counter() - start,
        "parameters": int(sum(np.asarray(tensor).size for tensor in factors)),
        "directional_passes": len(rows),
        "final_delta_energy": (
            abs(float(rows[-1]["energy"] - rows[-2]["energy"]))
            if len(rows) > 1
            else None
        ),
        "converged": bool(solver.converged),
    }


def _run_letta(
    hamiltonian,
    factors,
    *,
    bond_dim,
    sweeps,
    tolerance,
    tie_noise,
    seed,
    solver,
    natural_gradient_every,
):
    # Site 0 is plaquette B and site 3 is D in the B-A-C-D macro path.
    parents = ((3,), (), (), ())
    state = frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        factors,
        bond_dim=bond_dim,
        tie_noise=tie_noise,
        seed=seed,
        frontier_backend="identity_block",
    )
    initial = float(state.expectation())
    start = perf_counter()
    state.run(
        nsweeps=int(sweeps),
        tol=float(tolerance),
        solver=solver,
        frontier_canonicalization=True,
        frontier_gauge_weighting="probability",
        eig_tol=float(tolerance),
        maxiter=800,
        max_subspace=64,
        natural_gradient_every=int(natural_gradient_every),
        verbose=True,
    )
    seconds = perf_counter() - start
    energies = [float(row["energy"]) for row in state.history]
    return state, {
        "energy": float(state.expectation()),
        "initial_energy": initial,
        "seconds": seconds,
        "parameters": int(state.nparameters),
        "directional_passes": len(energies),
        "directional_pass_energies": energies,
        "final_delta_energy": (
            abs(energies[-1] - energies[-2]) if len(energies) > 1 else None
        ),
        "converged": bool(state.converged),
        "peak_frontier_elements": int(state.peak_compressed_frontier_elements),
    }


def benchmark(
    *,
    j2=0.5,
    bond_dim=4,
    mps_sweeps=40,
    letta_sweeps=40,
    tolerance=1.0e-10,
    tie_noise=1.0e-3,
    seed=7,
    letta_solver="whitened",
    natural_gradient_every=0,
    output=DEFAULT_OUTPUT,
):
    hamiltonian = plaquette_hamiltonian(j2=j2)
    mpo = MPO(list(hamiltonian.to_mpo().compress().tensors))
    solver, factors, mps = _run_block_mps(
        mpo,
        bond_dim=bond_dim,
        sweeps=mps_sweeps,
        tolerance=tolerance,
    )
    state, letta = _run_letta(
        hamiltonian,
        factors,
        bond_dim=bond_dim,
        sweeps=letta_sweeps,
        tolerance=tolerance,
        tie_noise=tie_noise,
        seed=seed,
        solver=letta_solver,
        natural_gradient_every=natural_gradient_every,
    )
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    exact_start = perf_counter()
    exact = float(
        eigsh(
            sparse_heisenberg_hamiltonian(
                16,
                tuple((*edge, 1.0) for edge in nearest)
                + tuple((*edge, float(j2)) for edge in diagonals),
            ),
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1.0e-11,
        )[0]
    )
    payload = {
        "model": {
            "shape": [4, 4],
            "j1": 1.0,
            "j2": float(j2),
            "boundary": "open",
        },
        "ansatz": {
            "plaquettes": [[list(site) for site in block] for block in PLAQUETTES],
            "macro_order": ["B", "A", "C", "D"],
            "macro_tree": [["B", "A"], ["A", "C"], ["C", "D"]],
            "full_plaquette_dimension": 16,
            "physical_tie": ["B", "D"],
            "physical_tie_dimension": 16,
            "bond_dim": int(bond_dim),
        },
        "settings": {
            "mps_sweeps": int(mps_sweeps),
            "letta_sweeps": int(letta_sweeps),
            "tolerance": float(tolerance),
            "tie_noise": float(tie_noise),
            "seed": int(seed),
            "letta_solver": str(letta_solver),
            "natural_gradient_every": int(natural_gradient_every),
            "frontier_canonicalization": True,
            "frontier_gauge_weighting": "probability",
        },
        "exact": {
            "energy": exact,
            "seconds": perf_counter() - exact_start,
        },
        "plaquette_mps": mps,
        "plaquette_letta": letta,
        "comparison": {
            "letta_minus_plaquette_mps": letta["energy"] - mps["energy"],
            "letta_error": letta["energy"] - exact,
            "mps_error": mps["energy"] - exact,
        },
    }
    snapshot = Path(output).with_suffix(".npz")
    np.savez_compressed(
        snapshot,
        **{
            f"tensor_{site:03d}": np.asarray(tensor)
            for site, tensor in enumerate(state.tensors)
        },
    )
    payload["plaquette_letta"]["snapshot"] = str(snapshot)
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=40)
    parser.add_argument("--letta-sweeps", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--letta-solver",
        choices=("whitened", "block_sparse", "matrix_free", "direct"),
        default="whitened",
    )
    parser.add_argument("--natural-gradient-every", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark(
        j2=args.j2,
        bond_dim=args.bond_dim,
        mps_sweeps=args.mps_sweeps,
        letta_sweeps=args.letta_sweeps,
        tolerance=args.tolerance,
        tie_noise=args.tie_noise,
        seed=args.seed,
        letta_solver=args.letta_solver,
        natural_gradient_every=args.natural_gradient_every,
        output=args.output,
    )
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
