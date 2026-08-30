#!/usr/bin/env python3
"""Compare dense and fixed-Sz MPS/graph-LETTA on the 4x4 J1-J2 model."""

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
from pyqed.letta import (
    FrontierAbelianLayout,
    FrontierTiedLETTA,
    abelian_frontier_tied_letta_from_mps,
)
from pyqed.mps import (
    DMRG,
    MPS,
    dense_to_symmetric,
    dense_to_symmetric_mpo,
    symmetric_to_dense,
)
from pyqed.mps.symmetry import AbelianSector, SymmetryManager


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_u1_4x4.json"


def _neel_cores(nsites):
    cores = []
    for site in range(int(nsites)):
        core = np.zeros((1, 2, 1))
        core[0, site % 2, 0] = 1.0
        cores.append(core)
    return cores


def _directional_history(solver):
    return [
        row
        for row in solver.sweep_history
        if row.get("direction") in {"lr", "rl"}
    ]


def _dense_mps_run(mpo, initial_cores, *, bond_dim, sweeps, tolerance):
    initial = MPS(
        [np.asarray(core).copy() for core in initial_cores],
        labels=["lv", "p", "rv"],
    )
    start = perf_counter()
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=initial,
        nsweeps=int(sweeps),
        opt="2site",
        symmetry=False,
        not_conv_err=False,
        verbose=0,
        sweep_tol=float(tolerance),
        davidson_tol=min(float(tolerance), 1.0e-10),
        davidson_max_iter=100,
        noise=0.0,
        recenter_final=False,
        performance="generic",
    ).run()
    seconds = perf_counter() - start
    cores = solver.state.to_order(["lv", "p", "rv"]).factors
    return solver, {
        "symmetry": "none",
        "optimizer": "two_site_dmrg",
        "energy": float(solver.energy),
        "seconds": float(seconds),
        "parameters": int(sum(np.asarray(core).size for core in cores)),
        "directional_passes": len(_directional_history(solver)),
        "converged": bool(solver.converged),
    }


def _u1_mps_run(dense_mpo, initial_cores, *, bond_dim, sweeps, tolerance):
    nsites = len(initial_cores)
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1} for _ in range(nsites)]
    symmetric_mpo = dense_to_symmetric_mpo(
        list(dense_mpo.factors),
        site_qn_maps,
    )
    initial = dense_to_symmetric(initial_cores, phys_qns=[q0, q1])
    manager = SymmetryManager(["charge"])
    target = manager.get_target_qn(nsites // 2)
    start = perf_counter()
    solver = DMRG(
        symmetric_mpo,
        D=int(bond_dim),
        init_guess=initial,
        nsweeps=int(sweeps),
        opt="2site",
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=site_qn_maps,
        not_conv_err=False,
        verbose=0,
        sweep_tol=float(tolerance),
        davidson_tol=min(float(tolerance), 1.0e-10),
        davidson_max_iter=100,
        noise=0.0,
        recenter_final=False,
        performance="generic",
    ).run()
    seconds = perf_counter() - start
    block_parameters = int(
        sum(
            np.asarray(block).size
            for tensor in solver.state.factors
            for block in tensor.data.values()
        )
    )
    dense_state = symmetric_to_dense(
        solver.state,
        site_qn_maps=site_qn_maps,
    ).to_order(["lv", "p", "rv"])
    labels = solver.state.labels
    left_axis = labels.index("lv")
    right_axis = labels.index("rv")
    factors = solver.state.factors
    bond_qns = [
        tuple(tuple(int(value) for value in charge) for charge in factors[0].qns[left_axis])
    ]
    bond_qns.extend(
        tuple(tuple(int(value) for value in charge) for charge in tensor.qns[right_axis])
        for tensor in factors
    )
    local_qns = tuple((((0,), (1,))) for _ in range(nsites))
    layout = FrontierAbelianLayout(
        local_qns=local_qns,
        bond_qns=tuple(bond_qns),
        target=(nsites // 2,),
    )
    return solver, dense_state, layout, {
        "symmetry": "U1_fixed_Sz",
        "optimizer": "two_site_dmrg",
        "energy": float(solver.energy),
        "seconds": float(seconds),
        "symmetry_parameters": block_parameters,
        "dense_equivalent_parameters": int(
            sum(np.asarray(core).size for core in dense_state.factors)
        ),
        "directional_passes": len(_directional_history(solver)),
        "converged": bool(solver.converged),
    }


def _sector_leakage(vector, nsites):
    vector = np.asarray(vector).reshape(-1)
    probability = np.abs(vector) ** 2
    probability /= np.sum(probability)
    leakage = 0.0
    for configuration, weight in zip(np.ndindex(*((2,) * nsites)), probability):
        if sum(configuration) != nsites // 2:
            leakage += float(weight)
    return leakage


def _letta_record(state, *, initial_energy, seconds):
    updates = [update for row in state.history for update in row["updates"]]
    return {
        "symmetry": (
            "U1_fixed_Sz"
            if hasattr(state, "abelian_layout")
            else "none"
        ),
        "optimizer": "one_site_exact_local_S_equals_I",
        "initial_energy": float(initial_energy),
        "energy": float(state.expectation()),
        "energy_per_site": float(state.expectation() / len(state.dims)),
        "seconds": float(seconds),
        "symmetry_parameters": int(state.nparameters),
        "dense_equivalent_parameters": int(
            getattr(state, "dense_nparameters", sum(tensor.size for tensor in state.tensors))
        ),
        "bond_dims": list(state.bond_dims),
        "directional_passes": len(state.history),
        "accepted_updates": int(sum(update.accepted for update in updates)),
        "solver_failures": int(sum(not update.solver_converged for update in updates)),
        "sector_leakage": float(
            _sector_leakage(state.state_vector(normalize=True), len(state.dims))
        ),
    }


def benchmark(*, bond_dim=4, mps_sweeps=20, letta_sweeps=40, tie_noise=1.0e-3, seed=7):
    nrows = ncols = 4
    nsites = nrows * ncols
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, 0.5) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
    dense_mpo = hamiltonian.to_mpo().compress()
    parents = parent_sets_from_edges(nsites, nearest)
    product = _neel_cores(nsites)

    _dense_solver, dense_mps = _dense_mps_run(
        dense_mpo,
        product,
        bond_dim=bond_dim,
        sweeps=mps_sweeps,
        tolerance=1.0e-10,
    )
    _u1_solver, u1_mps_dense, layout, u1_mps = _u1_mps_run(
        dense_mpo,
        product,
        bond_dim=bond_dim,
        sweeps=mps_sweeps,
        tolerance=1.0e-10,
    )

    u1 = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        u1_mps_dense.factors,
        abelian_layout=layout,
        tie_noise=tie_noise,
        seed=seed,
        frontier_backend="identity_block",
    )
    dense = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dims=u1.bond_dims,
        tensors=[tensor.copy() for tensor in u1.tensors],
        frontier_backend="identity_block",
    )
    dense.tensors = [tensor.copy() for tensor in u1.tensors]
    dense.energy = dense.expectation()
    initial_energy = float(u1.expectation())
    initial_overlap_error = float(
        np.max(np.abs(dense.state_vector() - u1.state_vector()))
    )

    start = perf_counter()
    dense.run(
        nsweeps=letta_sweeps,
        tol=0.0,
        solver="whitened",
    )
    dense_seconds = perf_counter() - start
    start = perf_counter()
    u1.run(
        nsweeps=letta_sweeps,
        tol=0.0,
        solver="whitened",
    )
    u1_seconds = perf_counter() - start

    sparse = sparse_heisenberg_hamiltonian(nsites, weighted)
    exact_start = perf_counter()
    exact_energy = float(
        eigsh(sparse, k=1, which="SA", return_eigenvectors=False, tol=1.0e-11)[0]
    )
    exact_seconds = perf_counter() - exact_start
    dense_letta = _letta_record(
        dense,
        initial_energy=initial_energy,
        seconds=dense_seconds,
    )
    u1_letta = _letta_record(
        u1,
        initial_energy=initial_energy,
        seconds=u1_seconds,
    )
    for record in (dense_letta, u1_letta):
        record["energy_error"] = float(record["energy"] - exact_energy)
        record["energy_error_per_site"] = float(
            (record["energy"] - exact_energy) / nsites
        )
    return {
        "model": {
            "shape": [nrows, ncols],
            "j1": 1.0,
            "j2": 0.5,
            "boundary": "open",
            "target_two_sz": 0,
            "site_order": "row-wise snake",
            "tie_graph": "all J1 nearest-neighbor bonds",
        },
        "settings": {
            "bond_dim": int(bond_dim),
            "mps_directional_pass_limit": int(mps_sweeps),
            "letta_directional_pass_limit": int(letta_sweeps),
            "tie_noise": float(tie_noise),
            "seed": int(seed),
        },
        "exact": {
            "energy": exact_energy,
            "seconds": float(exact_seconds),
            "used_during_optimization": False,
        },
        "initial_letta_state_max_difference": initial_overlap_error,
        "results": {
            "mps_dense_d4": dense_mps,
            "mps_u1_d4": u1_mps,
            "letta_dense_d4": dense_letta,
            "letta_u1_d4": u1_letta,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=20)
    parser.add_argument("--letta-sweeps", type=int, default=40)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark(
        bond_dim=args.bond_dim,
        mps_sweeps=args.mps_sweeps,
        letta_sweeps=args.letta_sweeps,
        tie_noise=args.tie_noise,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
