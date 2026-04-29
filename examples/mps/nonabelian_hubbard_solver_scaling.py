#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time

import numpy as np

from pyqed.mps.nonabelian import (
    BlockSparseEnvironmentChain,
    LocalOperator,
    ReducedStateLayout,
    ReducedStateVector,
    SweepDriver,
    build_random_spatial_mps,
    build_spatial_hubbard_mpo,
    merge_mps_sites,
    pack_two_site_state,
    solve_local_two_site,
    unpack_two_site_state,
)


def _wrap_tensor_operator(base_op):
    counter = {"matvecs": 0}

    def tensor_matvec(tensor):
        counter["matvecs"] += 1
        return base_op.tensor_matvec(tensor)

    return (
        LocalOperator(
            tensor_matvec=tensor_matvec,
            diag=base_op.diag,
            name=f"{base_op.name or 'operator'}-tensor-benchmark",
        ),
        counter,
    )


def _wrap_reduced_operator(base_op, template):
    counter = {"matvecs": 0}
    _, layout = pack_two_site_state(template)
    state_layout = ReducedStateLayout(tuple(layout))

    def reduced_matvec(state):
        counter["matvecs"] += 1
        if not isinstance(state, ReducedStateVector):
            raise TypeError("Benchmark reduced_matvec expects a ReducedStateVector.")
        tensor = unpack_two_site_state(state.to_packed(dtype=complex), template, layout=layout)
        out = base_op.tensor_matvec(tensor)
        if out.rank != template.rank:
            raise TypeError("Benchmark local operator returned an incompatible tensor rank.")
        packed, _ = pack_two_site_state(out, layout=layout)
        return state_layout.from_packed(packed)

    return (
        LocalOperator(
            reduced_matvec=reduced_matvec,
            diag=base_op.diag,
            name=f"{base_op.name or 'operator'}-reduced-benchmark",
        ),
        counter,
    )


def _benchmark_local_solve(operator, merged, *, tol, itermax):
    start = time.perf_counter()
    try:
        _optimized, objective = solve_local_two_site(
            merged,
            operator,
            tol=tol,
            itermax=itermax,
            couple_physical=False,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return elapsed, {"error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - start
    return elapsed, objective


def _chain_energy_converged(driver, *, energy_tol):
    if driver.converged:
        return True
    if len(driver.history) < 2:
        return False
    curr_entry = driver.history[-1]
    prev_entry = None
    for candidate in reversed(driver.history[:-1]):
        if candidate.get("direction") == curr_entry.get("direction"):
            prev_entry = candidate
            break
    if prev_entry is None:
        prev_entry = driver.history[-2]
    prev = prev_entry.get("energy")
    curr = curr_entry.get("energy")
    if prev is not None and curr is not None:
        if abs(float(curr) - float(prev)) <= float(energy_tol):
            return True

    best_history = []
    best = None
    for entry in driver.history:
        energy = entry.get("energy")
        if energy is None:
            continue
        best = float(energy) if best is None else min(best, float(energy))
        best_history.append(best)
    if len(best_history) >= 3:
        return abs(best_history[-1] - best_history[-3]) <= float(energy_tol)
    return False


def _benchmark_reference_chain(sites, mpo, *, max_bond, cutoff, max_nsweeps, energy_tol):
    start = time.perf_counter()
    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=max_nsweeps,
        mpo_factors=mpo,
        max_bond=max_bond,
        cutoff=cutoff,
    )
    driver.run()
    elapsed = time.perf_counter() - start
    return elapsed, {
        "energy": driver.last_energy,
        "nsweeps": driver.ncompleted,
        "converged": _chain_energy_converged(driver, energy_tol=energy_tol),
        "local_solver_kwargs": [entry.get("local_solver_kwargs") for entry in driver.history],
    }


def run_case(
    length,
    *,
    seed,
    bond_multiplicity,
    hopping_t,
    onsite_u,
    chemical_potential,
    tol,
    itermax,
    chain_max_nsweeps,
    chain_max_bond,
    chain_cutoff,
    chain_energy_tol,
):
    sites = build_random_spatial_mps(length, seed=seed, bond_multiplicity=bond_multiplicity)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=hopping_t,
        onsite_u=onsite_u,
        chemical_potential=chemical_potential,
    )
    bond = length // 2 - 1
    merged = merge_mps_sites(sites[bond], sites[bond + 1])
    env = BlockSparseEnvironmentChain.build(sites, mpo)
    base_operator = env.bond_operator(bond, merged)

    tensor_operator, tensor_counter = _wrap_tensor_operator(base_operator)
    reduced_operator, reduced_counter = _wrap_reduced_operator(base_operator, merged)

    tensor_time, tensor_objective = _benchmark_local_solve(
        tensor_operator,
        merged,
        tol=tol,
        itermax=itermax,
    )
    reduced_time, reduced_objective = _benchmark_local_solve(
        reduced_operator,
        merged,
        tol=tol,
        itermax=itermax,
    )
    chain_time, chain_objective = _benchmark_reference_chain(
        sites,
        mpo,
        max_bond=chain_max_bond,
        cutoff=chain_cutoff,
        max_nsweeps=chain_max_nsweeps,
        energy_tol=chain_energy_tol,
    )

    return {
        "L": length,
        "bond": bond,
        "packed_energy": tensor_objective.get("energy"),
        "reduced_energy": reduced_objective.get("energy"),
        "energy_delta": (
            abs(tensor_objective["energy"] - reduced_objective["energy"])
            if "energy" in tensor_objective and "energy" in reduced_objective
            else None
        ),
        "packed_time_s": tensor_time,
        "reduced_time_s": reduced_time,
        "packed_subspace": tensor_objective.get("subspace_dim"),
        "reduced_subspace": reduced_objective.get("subspace_dim"),
        "packed_iterations": tensor_objective.get("davidson_iterations"),
        "reduced_iterations": reduced_objective.get("davidson_iterations"),
        "packed_matvecs": tensor_counter["matvecs"],
        "reduced_matvecs": reduced_counter["matvecs"],
        "packed_status": tensor_objective.get("error", "ok"),
        "reduced_status": reduced_objective.get("error", "ok"),
        "chain_energy": chain_objective.get("energy"),
        "chain_time_s": chain_time,
        "chain_nsweeps": chain_objective.get("nsweeps"),
        "chain_converged": chain_objective.get("converged"),
        "chain_local_solver_kwargs": chain_objective.get("local_solver_kwargs"),
    }


def _format_solver_kwargs(items):
    if not items:
        return "[]"
    chunks = []
    for idx, item in enumerate(items):
        if not item:
            chunks.append(f"s{idx}={{}}")
            continue
        joined = ", ".join(f"{key}={value!r}" for key, value in sorted(item.items()))
        chunks.append(f"s{idx}=" + "{" + joined + "}")
    return "[" + ", ".join(chunks) + "]"


def _format_table(rows):
    headers = [
        "L",
        "bond",
        "packed_time_s",
        "reduced_time_s",
        "packed_matvecs",
        "reduced_matvecs",
        "packed_subspace",
        "reduced_subspace",
        "packed_status",
        "reduced_status",
        "chain_time_s",
        "chain_nsweeps",
        "chain_converged",
        "chain_E",
        "packed_E",
        "reduced_E",
        "|dE|",
    ]
    formatted = []
    for row in rows:
        formatted.append(
            {
                "L": str(row["L"]),
                "bond": str(row["bond"]),
                "packed_time_s": f'{row["packed_time_s"]:.6f}',
                "reduced_time_s": f'{row["reduced_time_s"]:.6f}',
                "packed_matvecs": str(row["packed_matvecs"]),
                "reduced_matvecs": str(row["reduced_matvecs"]),
                "packed_subspace": str(row["packed_subspace"]),
                "reduced_subspace": str(row["reduced_subspace"]),
                "packed_status": row["packed_status"],
                "reduced_status": row["reduced_status"],
                "chain_time_s": f'{row["chain_time_s"]:.6f}',
                "chain_nsweeps": str(row["chain_nsweeps"]),
                "chain_converged": str(row["chain_converged"]),
                "chain_E": (
                    f'{row["chain_energy"]:.12f}'
                    if row["chain_energy"] is not None
                    else "n/a"
                ),
                "packed_E": (
                    f'{row["packed_energy"]:.12f}'
                    if row["packed_energy"] is not None
                    else "n/a"
                ),
                "reduced_E": (
                    f'{row["reduced_energy"]:.12f}'
                    if row["reduced_energy"] is not None
                    else "n/a"
                ),
                "|dE|": (
                    f'{row["energy_delta"]:.3e}'
                    if row["energy_delta"] is not None
                    else "n/a"
                ),
            }
        )
    widths = {
        header: max(len(header), *(len(item[header]) for item in formatted))
        for header in headers
    }
    lines = []
    lines.append("  ".join(header.ljust(widths[header]) for header in headers))
    lines.append("  ".join("-" * widths[header] for header in headers))
    for item in formatted:
        lines.append("  ".join(item[header].ljust(widths[header]) for header in headers))
    lines.append("")
    lines.extend(
        f"L={row['L']} chain_local_solver={_format_solver_kwargs(row.get('chain_local_solver_kwargs'))}"
        for row in rows
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark packed vs reduced local solves for non-Abelian Hubbard chains."
        " The chain reference uses the MPO sweep default adaptive local-solver schedule unless overridden in code.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4, 6, 8],
        help="Chain lengths to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random MPS seed.")
    parser.add_argument(
        "--bond-multiplicity",
        type=int,
        default=4,
        help="Initial random MPS bond multiplicity.",
    )
    parser.add_argument("--hopping-t", type=float, default=1.0, help="Hubbard hopping.")
    parser.add_argument("--onsite-u", type=float, default=4.0, help="Onsite Hubbard U.")
    parser.add_argument(
        "--chemical-potential",
        type=float,
        default=0.0,
        help="Chemical potential.",
    )
    parser.add_argument("--tol", type=float, default=1e-8, help="Local solver tolerance.")
    parser.add_argument("--itermax", type=int, default=40, help="Local solver iteration cap.")
    parser.add_argument(
        "--chain-max-nsweeps",
        type=int,
        default=8,
        help="Maximum sweep count used to reach converged chain energy.",
    )
    parser.add_argument(
        "--chain-max-bond",
        type=int,
        default=128,
        help="Reference sweep bond cap.",
    )
    parser.add_argument(
        "--chain-cutoff",
        type=float,
        default=0.0,
        help="Reference sweep truncation cutoff.",
    )
    parser.add_argument(
        "--chain-energy-tol",
        type=float,
        default=1e-8,
        help="Energy-difference tolerance used to mark chain convergence.",
    )
    args = parser.parse_args()

    rows = [
        run_case(
            length,
            seed=args.seed,
            bond_multiplicity=args.bond_multiplicity,
            hopping_t=args.hopping_t,
            onsite_u=args.onsite_u,
            chemical_potential=args.chemical_potential,
            tol=args.tol,
            itermax=args.itermax,
            chain_max_nsweeps=args.chain_max_nsweeps,
            chain_max_bond=args.chain_max_bond,
            chain_cutoff=args.chain_cutoff,
            chain_energy_tol=args.chain_energy_tol,
        )
        for length in args.lengths
    ]

    print(_format_table(rows))


if __name__ == "__main__":
    main()
