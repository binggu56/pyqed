#!/usr/bin/env python3
"""Converged U(1) MPS-to-frontier-LETTA comparison on 6x6 J1-J2.

For every requested bond cap, the LETTA calculation starts from the resulting
U(1) MPS and then enables all nearest-neighbor Hamiltonian ties.  This makes
the comparison a direct test of the extra tied variational coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.benchmark_frontier_letta_u1_4x4 import (
    _directional_history,
    _neel_cores,
    _u1_mps_run,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta import abelian_frontier_tied_letta_from_mps


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_u1_6x6.json"


def _peak_rss_mib():
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024.0
    return peak / 1024.0**2


def _mps_energies(solver):
    return [
        float(np.real(row["energy"]))
        for row in _directional_history(solver)
        if "energy" in row
    ]


def _tail_delta(energies):
    return (
        float(abs(energies[-1] - energies[-2]))
        if len(energies) >= 2
        else float("inf")
    )


def benchmark(
    *,
    bond_dims=(8, 12, 16),
    mps_passes=30,
    letta_passes=16,
    two_site_cycles=1,
    polish_passes=4,
    two_site_pair_selection="residual",
    two_site_max_pairs=4,
    residual_fraction=0.9,
    min_pairs=2,
    coverage_every=0,
    block_size="auto",
    compute_dtype="auto",
    device="cpu",
    workers=1,
    tolerance=1.0e-7,
    eig_tolerance=1.0e-8,
    eig_tolerance_initial=1.0e-4,
    eig_maxiter=60,
    tie_noise=1.0e-3,
    seed=7,
    gauge="frontier",
    verbose=False,
):
    nrows = ncols = 6
    nsites = nrows * ncols
    effective_compute_dtype = (
        "float32"
        if compute_dtype == "auto" and eig_tolerance >= 1.0e-5
        else None
        if compute_dtype in {None, "same", "auto"}
        else compute_dtype
    )
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, 0.5) for left, right in diagonals)
    hamiltonian = heisenberg_local_hamiltonian(nsites, weighted)
    dense_mpo = hamiltonian.to_mpo().compress()
    parent_sets = parent_sets_from_edges(nsites, nearest)
    product = _neel_cores(nsites)

    results = []
    for bond_dim in bond_dims:
        mps_solver, mps_dense, layout, mps_record = _u1_mps_run(
            dense_mpo,
            product,
            bond_dim=bond_dim,
            sweeps=mps_passes,
            tolerance=tolerance,
        )
        mps_history = _mps_energies(mps_solver)
        mps_record.update(
            {
                "bond_dim": int(bond_dim),
                "directional_pass_energies": mps_history,
                "final_delta_energy": _tail_delta(mps_history),
                "peak_process_rss_mib": _peak_rss_mib(),
            }
        )

        letta = abelian_frontier_tied_letta_from_mps(
            hamiltonian,
            parent_sets,
            mps_dense.factors,
            abelian_layout=layout,
            tie_noise=tie_noise,
            seed=seed,
            frontier_backend="identity_block",
            route_memory=32,
            action_memory=32,
            compute_dtype=(
                effective_compute_dtype
            ),
            device=device,
            workers=workers,
        )
        initial_energy = float(letta.expectation())
        start = perf_counter()
        selected_gauge = None if gauge == "none" else gauge
        if letta_passes:
            letta.run(
                nsweeps=letta_passes,
                tol=0.0,
                solver="matrix_free",
                eig_tol=eig_tolerance,
                adaptive_solver=True,
                eig_tol_initial=eig_tolerance_initial,
                maxiter=eig_maxiter,
                gauge=selected_gauge,
                gauge_weight="probability",
                environment_cache="auto",
                environment_memory=64,
            )
        one_site_history = tuple(letta.history)
        if two_site_cycles:
            letta.run_two_site(
                nsweeps=2 * int(two_site_cycles),
                tol=tolerance,
                solver="matrix_free",
                verify_pair_energies=False,
                pair_operator_backend="auto",
                factor_solver="matrix_free",
                split_strategy="hybrid",
                split_metric_sweeps=1,
                split_variational_sweeps=1,
                outer_cycles=1,
                metric_tol=1.0e-8,
                eig_tol=max(eig_tolerance, tolerance * 0.1),
                adaptive_solver=True,
                eig_tol_initial=eig_tolerance_initial,
                pair_selection=two_site_pair_selection,
                max_pairs=two_site_max_pairs,
                residual_fraction=residual_fraction,
                min_pairs=min_pairs,
                coverage_every=coverage_every,
                reuse_residual_scores=False,
                certify_residual=False,
                block_size=block_size,
                preconditioner="auto",
                recycle=True,
                recycle_min_size=1,
                maxiter=eig_maxiter,
                max_subspace=32,
                verbose=verbose,
            )
        two_site_history = tuple(letta.history) if two_site_cycles else ()
        if polish_passes:
            letta.run(
                nsweeps=int(polish_passes),
                tol=tolerance,
                solver="matrix_free",
                eig_tol=eig_tolerance,
                adaptive_solver=True,
                eig_tol_initial=eig_tolerance_initial,
                maxiter=eig_maxiter,
                gauge=selected_gauge,
                gauge_weight="probability",
                environment_cache="checkpointed",
                environment_memory=32,
            )
        polish_history = tuple(letta.history) if polish_passes else ()
        polish_converged = bool(letta.converged)
        tie_reductions = letta.prune_ties()
        bond_reductions = letta.reduce_null_bonds()
        final_pair_residuals = letta.pair_residual_certificates()
        residual_certificate_tol = max(10.0 * eig_tolerance, 1.0e-7)
        final_residual_certified = bool(
            not final_pair_residuals
            or max(final_pair_residuals) <= residual_certificate_tol
        )
        final_converged = bool(polish_converged and final_residual_certified)
        letta_seconds = perf_counter() - start
        letta_history = [
            float(row["energy"])
            for row in (*one_site_history, *two_site_history, *polish_history)
        ]
        letta_energy = float(letta.expectation())
        results.append(
            {
                "D": int(bond_dim),
                "mps": mps_record,
                "letta": {
                    "symmetry": "U1_fixed_Sz",
                    "optimizer": (
                        "one_site_then_u1_two_site_then_one_site"
                        if letta_passes and two_site_cycles and polish_passes
                        else "u1_two_site_matrix_free"
                        if two_site_cycles
                        else "one_site_matrix_free"
                    ),
                    "ties": "all_J1_nearest_neighbor_bonds",
                    "gauge": (
                        "none"
                        if selected_gauge is None
                        else "conditional_frontier_probability"
                    ),
                    "initial_energy": initial_energy,
                    "energy": letta_energy,
                    "energy_gain_from_mps": float(
                        mps_record["energy"] - letta_energy
                    ),
                    "seconds": float(letta_seconds),
                    "one_site_directional_passes": len(one_site_history),
                    "two_site_cycles_requested": int(two_site_cycles),
                    "two_site_directional_passes": len(two_site_history),
                    "polish_directional_passes": len(polish_history),
                    "two_site_pair_selection": two_site_pair_selection,
                    "two_site_max_pairs": two_site_max_pairs,
                    "residual_fraction": residual_fraction,
                    "min_pairs": min_pairs,
                    "coverage_every": coverage_every,
                    "block_size": block_size,
                    "compute_dtype": compute_dtype,
                    "effective_compute_dtype": (
                        "same"
                        if effective_compute_dtype is None
                        else effective_compute_dtype
                    ),
                    "device": device,
                    "workers": workers,
                    "two_site_selected_pairs": [
                        list(row.get("selected_pairs", ()))
                        for row in two_site_history
                    ],
                    "residual_scores_reused": [
                        bool(row.get("residual_scores_reused", False))
                        for row in two_site_history
                    ],
                    "pair_operator_backends": [
                        [update.pair_operator_backend for update in row["updates"]]
                        for row in two_site_history
                    ],
                    "two_site_solvers": [
                        [update.local_update.solver for update in row["updates"]]
                        for row in two_site_history
                    ],
                    "two_site_solver_converged": [
                        [
                            bool(update.local_update.solver_converged)
                            for update in row["updates"]
                        ]
                        for row in two_site_history
                    ],
                    "two_site_solver_messages": [
                        [update.local_update.message for update in row["updates"]]
                        for row in two_site_history
                    ],
                    "directional_passes": len(letta_history),
                    "directional_pass_energies": letta_history,
                    "final_delta_energy": (
                        _tail_delta(letta_history)
                        if polish_history
                        else
                        float(two_site_history[-1]["cycle_delta"])
                        if two_site_history
                        and two_site_history[-1]["cycle_complete"]
                        else _tail_delta(letta_history)
                    ),
                    "converged": final_converged,
                    "final_pair_residuals": list(final_pair_residuals),
                    "maximum_pair_residual": max(
                        final_pair_residuals,
                        default=0.0,
                    ),
                    "residual_certificate_tol": residual_certificate_tol,
                    "residual_certified": final_residual_certified,
                    "tie_reductions": [
                        {
                            "edge": list(record.edge),
                            "relative_discarded_weight": record.relative_discarded_weight,
                            "norm_error": record.norm_error,
                        }
                        for record in tie_reductions
                    ],
                    "bond_reductions": [
                        {
                            "cut": record.cut,
                            "old_dimension": record.old_dimension,
                            "new_dimension": record.new_dimension,
                            "relative_discarded_weight": record.relative_discarded_weight,
                            "norm_error": record.norm_error,
                        }
                        for record in bond_reductions
                    ],
                    "two_site_cycles": [
                        {
                            "cycle": int(row["cycle"]),
                            "energy": float(row["energy"]),
                            "energy_gain": float(row["cycle_delta"]),
                            "endpoints_accepted": bool(
                                row["cycle_endpoints_accepted"]
                            ),
                        }
                        for row in two_site_history
                        if row["cycle_complete"]
                    ],
                    "symmetry_parameters": int(letta.nparameters),
                    "dense_equivalent_parameters": int(
                        letta.dense_nparameters
                    ),
                    "bond_dims": list(letta.bond_dims),
                    "hamiltonian_matvecs": int(
                        sum(row["hamiltonian_matvecs"] for row in one_site_history)
                        + sum(row["hamiltonian_matvecs"] for row in polish_history)
                        + sum(
                            update.local_update.hamiltonian_matvecs
                            for row in two_site_history
                            for update in row["updates"]
                        )
                    ),
                    "solver_failures": int(
                        sum(row["solver_failures"] for row in one_site_history)
                        + sum(row["solver_failures"] for row in polish_history)
                        + sum(
                            not update.local_update.solver_converged
                            for row in two_site_history
                            for update in row["updates"]
                        )
                    ),
                    "environment_cache": (
                        polish_history[-1]["environment_cache"]
                        if polish_history
                        else one_site_history[-1]["environment_cache"]
                        if one_site_history
                        else None
                    ),
                    "full_environment_mib": (
                        polish_history[-1]["full_environment_bytes"] / 1024.0**2
                        if polish_history
                        else one_site_history[-1]["full_environment_bytes"] / 1024.0**2
                        if one_site_history
                        else None
                    ),
                    "peak_process_rss_mib": _peak_rss_mib(),
                },
            }
        )

    return {
        "model": {
            "shape": [nrows, ncols],
            "j1": 1.0,
            "j2": 0.5,
            "boundary": "open",
            "target_two_sz": 0,
            "site_order": "row-wise snake",
        },
        "settings": {
            "bond_dims": [int(value) for value in bond_dims],
            "mps_directional_pass_limit": int(mps_passes),
            "letta_directional_pass_limit": int(letta_passes),
            "two_site_cycle_limit": int(two_site_cycles),
            "polish_directional_pass_limit": int(polish_passes),
            "two_site_pair_selection": str(two_site_pair_selection),
            "two_site_max_pairs": two_site_max_pairs,
            "residual_fraction": float(residual_fraction),
            "min_pairs": int(min_pairs),
            "coverage_every": int(coverage_every),
            "block_size": block_size,
            "compute_dtype": compute_dtype,
            "effective_compute_dtype": (
                "same"
                if effective_compute_dtype is None
                else effective_compute_dtype
            ),
            "device": device,
            "workers": int(workers),
            "energy_tolerance": float(tolerance),
            "local_eigensolver_tolerance": float(eig_tolerance),
            "initial_local_eigensolver_tolerance": float(eig_tolerance_initial),
            "local_eigensolver_maxiter": int(eig_maxiter),
            "tie_noise": float(tie_noise),
            "seed": int(seed),
            "gauge": str(gauge),
            "verbose": bool(verbose),
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dims", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--mps-passes", type=int, default=30)
    parser.add_argument("--letta-passes", type=int, default=16)
    parser.add_argument("--two-site-cycles", type=int, default=1)
    parser.add_argument("--polish-passes", type=int, default=4)
    parser.add_argument(
        "--two-site-pair-selection",
        choices=("all", "residual"),
        default="residual",
    )
    parser.add_argument("--two-site-max-pairs", type=int, default=4)
    parser.add_argument("--residual-fraction", type=float, default=0.9)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--coverage-every", type=int, default=0)
    parser.add_argument("--block-size", default="auto")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "same", "float32", "float64"),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--eig-tol", type=float, default=1.0e-8)
    parser.add_argument("--eig-tol-initial", type=float, default=1.0e-4)
    parser.add_argument("--eig-maxiter", type=int, default=60)
    parser.add_argument("--tie-noise", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gauge", choices=("frontier", "none"), default="frontier")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark(
        bond_dims=args.bond_dims,
        mps_passes=args.mps_passes,
        letta_passes=args.letta_passes,
        two_site_cycles=args.two_site_cycles,
        polish_passes=args.polish_passes,
        two_site_pair_selection=args.two_site_pair_selection,
        two_site_max_pairs=args.two_site_max_pairs,
        residual_fraction=args.residual_fraction,
        min_pairs=args.min_pairs,
        coverage_every=args.coverage_every,
        block_size=(
            args.block_size
            if args.block_size == "auto"
            else int(args.block_size)
        ),
        compute_dtype=args.compute_dtype,
        device=args.device,
        workers=args.workers,
        tolerance=args.tol,
        eig_tolerance=args.eig_tol,
        eig_tolerance_initial=args.eig_tol_initial,
        eig_maxiter=args.eig_maxiter,
        tie_noise=args.tie_noise,
        seed=args.seed,
        gauge=args.gauge,
        verbose=args.verbose,
    )
    text = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
