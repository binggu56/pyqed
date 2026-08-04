#!/usr/bin/env python3
"""Compare unrestricted and exact U(1)-projected LETTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed.letta import LETTA, LocalHamiltonian, LocalTerm, SectorProjectedLETTA


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "projected_u1_letta_1d_L10_D4.json"


def heisenberg_chain(length: int) -> LocalHamiltonian:
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return LocalHamiltonian(
        (2,) * int(length),
        tuple(LocalTerm((site, site + 1), exchange) for site in range(int(length) - 1)),
    )


def nearest_neighbor_parents(length: int):
    return tuple(
        (site + 1,) if site + 1 < int(length) else () for site in range(int(length))
    )


def sector_mask(length: int) -> np.ndarray:
    target = int(length) // 2
    return np.fromiter(
        (
            sum(configuration) == target
            for configuration in np.ndindex(*((2,) * length))
        ),
        dtype=bool,
        count=2**length,
    )


def raw_vector(state) -> np.ndarray:
    if isinstance(state, SectorProjectedLETTA):
        return np.asarray(state.raw_state_vector()).reshape(-1)
    return np.asarray(state.state_vector()).reshape(-1)


def sector_weight(state, mask: np.ndarray) -> float:
    vector = raw_vector(state)
    norm2 = float(np.vdot(vector, vector).real)
    if norm2 <= 0.0:
        return 0.0
    return float(np.vdot(vector[mask], vector[mask]).real / norm2)


def physical_vector(state, mask: np.ndarray) -> np.ndarray:
    vector = raw_vector(state)
    if isinstance(state, SectorProjectedLETTA):
        vector = np.where(mask, vector, 0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("benchmark state is numerically zero.")
    return vector / norm


def history_records(state):
    return [
        {
            "sweep": int(row["sweep"]),
            "energy": float(row["energy"]),
            "delta_energy": float(row["delta_energy"]),
            "accepted_sites": int(row["accepted_sites"]),
            "solver_failures": int(row["solver_failures"]),
        }
        for row in state.history
    ]


def run_one_site(state, *, sweeps: int) -> float:
    started = perf_counter()
    state.run(
        nsweeps=int(sweeps),
        tol=0.0,
        metric_tol=1.0e-12,
        solver="whitened",
        eig_tol=1.0e-11,
        maxiter=500,
        max_subspace=48,
        frontier_canonicalization=False,
        environment_cache="full",
    )
    return float(perf_counter() - started)


def state_record(
    state,
    *,
    initial_energy: float,
    initial_raw_sector_weight: float,
    seconds: float,
    dense_hamiltonian: np.ndarray,
    exact_energy: float,
    mask: np.ndarray,
    requested_sweeps: int,
    construction_seconds: float,
):
    vector = physical_vector(state, mask)
    dense_energy = float(np.real(np.vdot(vector, dense_hamiltonian @ vector)))
    dense_parameters = int(sum(tensor.size for tensor in state.tensors))
    active_parameters = int(getattr(state, "nparameters", dense_parameters))
    final_raw_sector_weight = sector_weight(state, mask)
    represented_sector_weight = (
        1.0 if isinstance(state, SectorProjectedLETTA) else final_raw_sector_weight
    )
    record = {
        "initial_energy": float(initial_energy),
        "energy": float(state.expectation()),
        "dense_vector_energy": dense_energy,
        "energy_error": float(dense_energy - exact_energy),
        "initial_raw_target_sector_weight": float(initial_raw_sector_weight),
        "final_raw_target_sector_weight": final_raw_sector_weight,
        "represented_target_sector_weight": represented_sector_weight,
        "active_parameters": active_parameters,
        "stored_parameters": dense_parameters,
        "active_parameter_fraction": float(active_parameters / dense_parameters),
        "seconds": float(seconds),
        "construction_seconds": float(construction_seconds),
        "requested_sweeps": int(requested_sweeps),
        "executed_sweeps": len(state.history),
        "converged": bool(state.converged),
        "history": history_records(state),
    }
    frontier = state._hamiltonian_frontier
    record["objective_mpo_factorized"] = bool(
        getattr(frontier, "factorized_mpo", False)
    )
    record["objective_mpo_stored_elements"] = int(
        getattr(frontier, "stored_mpo_elements", 0)
    )
    record["objective_mpo_dense_elements"] = int(
        getattr(frontier, "dense_mpo_elements", 0)
    )
    record["objective_contraction_plans"] = int(frontier.plan_count)
    return record


def benchmark(
    *,
    length: int = 10,
    bond_dim: int = 4,
    sweeps: int = 6,
    seed: int = 23,
    output: Path = DEFAULT_OUTPUT,
):
    length = int(length)
    bond_dim = int(bond_dim)
    sweeps = int(sweeps)
    if length < 2 or length % 2:
        raise ValueError("length must be an even integer of at least two.")
    if bond_dim < 1 or sweeps < 0:
        raise ValueError("bond_dim must be positive and sweeps nonnegative.")

    hamiltonian = heisenberg_chain(length)
    parents = nearest_neighbor_parents(length)
    dense_hamiltonian = hamiltonian.to_dense()
    exact_energy = float(np.linalg.eigvalsh(dense_hamiltonian)[0])
    charges = ((0, 1),) * length
    target = length // 2
    mask = sector_mask(length)

    construction_started = perf_counter()
    unrestricted = LETTA(
        hamiltonian,
        parents=parents,
        bond_dim=bond_dim,
        frontier_backend="identity_block",
        seed=seed,
    )
    unrestricted_construction_seconds = perf_counter() - construction_started
    construction_started = perf_counter()
    projected = SectorProjectedLETTA.from_unrestricted(
        unrestricted,
        local_charges=charges,
        target=target,
        frontier_backend="identity_block",
    )
    projected_construction_seconds = perf_counter() - construction_started
    initial_tensor_error = max(
        float(np.max(np.abs(left - right), initial=0.0))
        for left, right in zip(unrestricted.tensors, projected.tensors)
    )
    states = {
        "unrestricted": unrestricted,
        "sector_projected": projected,
    }
    construction_timings = {
        "unrestricted": unrestricted_construction_seconds,
        "sector_projected": projected_construction_seconds,
    }
    initial = {
        name: {
            "energy": float(state.expectation()),
            "sector_weight": sector_weight(state, mask),
        }
        for name, state in states.items()
    }

    timings = {}
    for name, state in states.items():
        timings[name] = run_one_site(state, sweeps=sweeps)
        print(
            f"{name:30s} E={state.expectation(): .12f} "
            f"raw_weight={sector_weight(state, mask):.9f} "
            f"time={timings[name]:.3f}s",
            flush=True,
        )

    payload = {
        "model": {
            "name": "open spin-1/2 antiferromagnetic Heisenberg chain",
            "length": length,
            "bond_dim": bond_dim,
            "target_particle_count": target,
            "exact_energy": exact_energy,
            "one_site_sweeps": sweeps,
            "seed": int(seed),
        },
        "initialization": {
            "description": (
                "The unrestricted and sector-projected states start from "
                "identical unrestricted LETTA tensors."
            ),
            "maximum_unrestricted_projected_tensor_error": initial_tensor_error,
        },
        "methods": {
            name: state_record(
                state,
                initial_energy=initial[name]["energy"],
                initial_raw_sector_weight=initial[name]["sector_weight"],
                seconds=timings[name],
                dense_hamiltonian=dense_hamiltonian,
                exact_energy=exact_energy,
                mask=mask,
                requested_sweeps=sweeps,
                construction_seconds=construction_timings[name],
            )
            for name, state in states.items()
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark(
        length=args.length,
        bond_dim=args.bond_dim,
        sweeps=args.sweeps,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
