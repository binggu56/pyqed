#!/usr/bin/env python3
"""Compute adjacent LETTA wavefunction overlaps for the Shastry-Sutherland model."""

from __future__ import annotations

import argparse
import csv
import json
from time import perf_counter

import numpy as np

from run_ss_energy_benchmark import (
    RESULTS_DIR,
    dimer_product_letta,
    heisenberg_long_range_mpo,
    ordered_bonds,
    parse_floats,
    parse_ints,
)


def optimize_letta_state(
    nsites: int,
    mpo,
    *,
    bond_dim: int,
    max_sweeps: int,
    seed: int,
    tol: float,
):
    start = perf_counter()
    letta = dimer_product_letta(
        nsites,
        bond_dim=int(bond_dim),
        noise=0.0 if int(bond_dim) == 1 else 1.0e-3,
        seed=seed,
    )
    initial = letta.expectation(mpo)
    letta.run(
        mpo,
        nsweeps=int(max_sweeps),
        tol=float(tol),
        local_solver="auto",
        matrix_free_threshold=1024,
        matrix_free_tol=1.0e-8,
        matrix_free_maxiter=80,
        verbose=0,
    )
    energy = letta.expectation(mpo)
    final_delta = None if not letta.history else letta.history[-1]["delta_energy"]
    result = {
        "energy": float(energy),
        "initial": float(initial),
        "seconds": perf_counter() - start,
        "sweeps_completed": int(letta.ncompleted),
        "converged": bool(letta.converged),
        "final_delta_energy": None if final_delta is None else float(final_delta),
        "convergence_tol": float(tol),
        "max_sweeps": int(max_sweeps),
    }
    if not letta.converged:
        delta_text = "None" if final_delta is None else f"{float(final_delta):.3e}"
        raise RuntimeError(
            "LETTA did not converge within "
            f"{int(max_sweeps)} sweeps (final dE={delta_text}, tol={float(tol):.3e}, "
            f"E={float(energy):.12g})"
        )
    return letta, result


def normalized_overlap(left, right):
    raw = left.state_overlap(right)
    norm_left = float(np.real(left.state_overlap(left)))
    norm_right = float(np.real(right.state_overlap(right)))
    if norm_left <= 0.0 or norm_right <= 0.0:
        raise ValueError("Cannot normalize overlap with a zero-norm LETTA state.")
    return float(abs(raw) / np.sqrt(norm_left * norm_right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--jprime", default="0.0,0.125,0.25,0.5,0.6,0.65,0.675,0.7,0.725,0.75,0.8,0.85,0.9,1.0")
    parser.add_argument("--letta-d", default="1,2")
    parser.add_argument("--letta-max-sweeps", type=int, default=300)
    parser.add_argument("--letta-tol", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    lx = int(args.lx)
    ly = int(args.ly)
    nsites = lx * ly
    jprime_values = parse_floats(args.jprime)
    bond_dims = parse_ints(args.letta_d)
    output_prefix = args.output_prefix or f"ss_letta_overlap_Lx{lx}_Ly{ly}"
    csv_path = RESULTS_DIR / f"{output_prefix}.csv"
    json_path = RESULTS_DIR / f"{output_prefix}.json"

    rows = []
    state_metadata = {}
    for bond_dim in bond_dims:
        states = []
        summaries = []
        print(f"# LETTA D={bond_dim}")
        for jprime in jprime_values:
            bonds = ordered_bonds(lx, ly, "dimer-first", jprime)
            mpo = heisenberg_long_range_mpo(nsites, bonds)
            seed = int(args.seed) + 31 * int(bond_dim) + int(round(1000 * float(jprime)))
            state, summary = optimize_letta_state(
                nsites,
                mpo,
                bond_dim=bond_dim,
                max_sweeps=args.letta_max_sweeps,
                seed=seed,
                tol=args.letta_tol,
            )
            states.append(state)
            summaries.append(summary)
            print(
                f"  J'/J={jprime:g} E={summary['energy']:.12g} "
                f"sweeps={summary['sweeps_completed']} dE={summary['final_delta_energy']}"
            )

        state_metadata[str(bond_dim)] = summaries
        for index, (left_j, right_j) in enumerate(zip(jprime_values[:-1], jprime_values[1:])):
            left = states[index]
            right = states[index + 1]
            left_summary = summaries[index]
            right_summary = summaries[index + 1]
            delta = float(right_j) - float(left_j)
            overlap_abs = normalized_overlap(left, right)
            overlap_abs = min(max(overlap_abs, 0.0), 1.0)
            fidelity = overlap_abs**2
            fidelity_susceptibility = -2.0 * np.log(max(overlap_abs, 1.0e-300)) / (nsites * delta**2)
            rows.append(
                {
                    "D": int(bond_dim),
                    "jprime_left": float(left_j),
                    "jprime_right": float(right_j),
                    "jprime_mid": 0.5 * (float(left_j) + float(right_j)),
                    "delta_jprime": delta,
                    "overlap_abs": overlap_abs,
                    "fidelity": fidelity,
                    "infidelity": 1.0 - fidelity,
                    "fidelity_susceptibility": float(fidelity_susceptibility),
                    "energy_left": left_summary["energy"],
                    "energy_right": right_summary["energy"],
                    "sweeps_left": left_summary["sweeps_completed"],
                    "sweeps_right": right_summary["sweeps_completed"],
                    "final_delta_left": left_summary["final_delta_energy"],
                    "final_delta_right": right_summary["final_delta_energy"],
                    "convergence_tol": float(args.letta_tol),
                    "max_sweeps": int(args.letta_max_sweeps),
                }
            )

    fieldnames = [
        "D",
        "jprime_left",
        "jprime_right",
        "jprime_mid",
        "delta_jprime",
        "overlap_abs",
        "fidelity",
        "infidelity",
        "fidelity_susceptibility",
        "energy_left",
        "energy_right",
        "sweeps_left",
        "sweeps_right",
        "final_delta_left",
        "final_delta_right",
        "convergence_tol",
        "max_sweeps",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "lx": lx,
        "ly": ly,
        "nsites": nsites,
        "geometry": "open x, periodic y",
        "order": "dimer-first",
        "jprime_values": jprime_values,
        "bond_dims": bond_dims,
        "letta_max_sweeps": int(args.letta_max_sweeps),
        "letta_tol": float(args.letta_tol),
        "seed": int(args.seed),
        "state_runs": state_metadata,
    }
    with json_path.open("w") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)

    print(f"# wrote {csv_path}")
    print(f"# wrote {json_path}")


if __name__ == "__main__":
    main()
