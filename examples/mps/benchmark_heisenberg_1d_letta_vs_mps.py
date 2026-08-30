#!/usr/bin/env python3
"""Benchmark fixed-budget MPS and nearest-neighbor LETTA on a Heisenberg chain.

The benchmark intentionally reproduces the protocol used for the LETTA
manuscript's finite-size figure: an open spin-1/2 chain, a Neel-state MPS
start, two random LETTA starts, and a fixed number of directional passes.
Exact references are computed in the zero-magnetization sector, avoiding the
full ``2**L`` Hilbert space.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyqed.letta import LETTA  # noqa: E402
from pyqed.models.heisenberg import Heisenberg  # noqa: E402
from pyqed.mps.dmrg import DMRG  # noqa: E402


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "heisenberg_1d_letta_vs_mps_fixed8.json"
)


def comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def fixed_sz_heisenberg(length: int):
    """Return the open-chain Heisenberg Hamiltonian in the smallest-|Sz| sector."""
    if length < 2:
        raise ValueError("length must be at least two")

    n_up = length // 2
    basis = []
    for occupied in combinations(range(length), n_up):
        state = 0
        for site in occupied:
            state |= 1 << site
        basis.append(state)
    position = {state: index for index, state in enumerate(basis)}

    rows = []
    columns = []
    values = []
    for row, state in enumerate(basis):
        diagonal = 0.0
        for site in range(length - 1):
            different = ((state >> site) ^ (state >> (site + 1))) & 1
            if different:
                diagonal -= 0.25
                flipped = state ^ (1 << site) ^ (1 << (site + 1))
                rows.append(row)
                columns.append(position[flipped])
                values.append(0.5)
            else:
                diagonal += 0.25
        rows.append(row)
        columns.append(row)
        values.append(diagonal)

    dimension = len(basis)
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension),
        dtype=float,
    ).tocsr()
    return hamiltonian, n_up


def exact_ground_state(length: int, tolerance: float, seed: int = 1729) -> dict:
    started = perf_counter()
    hamiltonian, n_up = fixed_sz_heisenberg(length)
    build_seconds = perf_counter() - started

    rng = np.random.default_rng(seed + length)
    v0 = rng.normal(size=hamiltonian.shape[0])
    started = perf_counter()
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        v0=v0,
        tol=float(tolerance),
        maxiter=100_000,
    )
    solve_seconds = perf_counter() - started
    energy = float(eigenvalues[0])
    vector = eigenvectors[:, 0]
    residual = float(np.linalg.norm(hamiltonian @ vector - energy * vector))
    return {
        "energy": energy,
        "sector_n_up": int(n_up),
        "sector_dimension": int(hamiltonian.shape[0]),
        "nonzeros": int(hamiltonian.nnz),
        "residual_norm": residual,
        "tolerance": float(tolerance),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
    }


def scalar_history(history, keys: tuple[str, ...]) -> list[dict]:
    records = []
    for row in history:
        record = {}
        for key in keys:
            value = row.get(key)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, complex):
                value = float(np.real(value))
            if value is None or isinstance(value, (bool, int, float, str)):
                record[key] = value
        records.append(record)
    return records


def mps_run(
    mpo,
    neel,
    bond_dim: int,
    sweeps: int,
    tolerance: float,
    *,
    davidson_tolerance: float = 1.0e-5,
    davidson_max_iterations: int = 30,
) -> dict:
    started = perf_counter()
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=neel,
        nsweeps=int(sweeps),
        not_conv_err=False,
        sweep_tol=float(tolerance),
        davidson_tol=float(davidson_tolerance),
        davidson_max_iter=int(davidson_max_iterations),
        noise=1.0e-6,
        verbose=0,
        recenter_final=False,
        performance="legacy-auto",
    ).run()
    seconds = perf_counter() - started
    return {
        "energy": float(np.real(solver.energy)),
        "seconds": seconds,
        "converged": bool(solver.converged),
        "passes_completed": len(solver.sweep_history),
        "history": scalar_history(
            solver.sweep_history,
            ("sweep", "direction", "energy", "local_energy", "truncation", "gauge"),
        ),
    }


def letta_run(
    mpo,
    length: int,
    bond_dim: int,
    seed: int,
    sweeps: int,
    tolerance: float,
    gauge: str,
) -> dict:
    started = perf_counter()
    # LETTA's built-in random tensors divide every local tensor by a scalar.
    # That leaves the normalized state unchanged but can underflow the global
    # norm on long chains before the constructor normalizes it.  Generate the
    # identical random directions without those irrelevant local scalars.
    rng = np.random.default_rng(seed)
    bonds = [1] + [int(bond_dim)] * max(0, length - 2) + [1]
    tensors = [
        rng.normal(size=(bonds[site], 2, 2, bonds[site + 1]))
        for site in range(length - 1)
    ]
    state = LETTA(
        None,
        (2,) * length,
        bond_dim=int(bond_dim),
        seed=int(seed),
        tensors=tensors,
    )
    state.run(
        mpo,
        nsweeps=int(sweeps),
        tol=float(tolerance),
        local_solver="auto",
        gauge=gauge,
    )
    seconds = perf_counter() - started
    return {
        "seed": int(seed),
        "energy": float(state.expectation_mpo(mpo)),
        "seconds": seconds,
        "converged": bool(state.converged),
        "passes_completed": int(state.ncompleted),
        "history": scalar_history(
            state.history,
            ("sweep", "direction", "energy", "delta_energy", "gauge"),
        ),
    }


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_results(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)


def write_csv(payload: dict, output: Path) -> Path:
    csv_path = output.with_suffix(".csv")
    fieldnames = [
        "length",
        "bond_dim",
        "exact_energy",
        "mps_energy",
        "mps_error",
        "mps_converged",
        "mps_passes",
        "mps_seconds",
        "letta_energy",
        "letta_error",
        "letta_seed",
        "letta_converged",
        "letta_passes",
        "letta_seconds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for length_record in payload["results"]:
            exact = float(length_record["exact"]["energy"])
            for run in length_record["runs"]:
                mps = run["mps"]
                letta = run["letta_selected"]
                writer.writerow(
                    {
                        "length": length_record["length"],
                        "bond_dim": run["bond_dim"],
                        "exact_energy": f"{exact:.16g}",
                        "mps_energy": f"{mps['energy']:.16g}",
                        "mps_error": f"{mps['error']:.16g}",
                        "mps_converged": mps["converged"],
                        "mps_passes": mps["passes_completed"],
                        "mps_seconds": f"{mps['seconds']:.9g}",
                        "letta_energy": f"{letta['energy']:.16g}",
                        "letta_error": f"{letta['error']:.16g}",
                        "letta_seed": letta["seed"],
                        "letta_converged": letta["converged"],
                        "letta_passes": letta["passes_completed"],
                        "letta_seconds": f"{letta['seconds']:.9g}",
                    }
                )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=comma_separated_ints, default=(16, 18, 20))
    parser.add_argument("--bond-dims", type=comma_separated_ints, default=(1, 2, 4))
    parser.add_argument("--seeds", type=comma_separated_ints, default=(1, 2))
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--ed-tol", type=float, default=1.0e-11)
    parser.add_argument("--letta-gauge", choices=("virtual", "conditional"), default="virtual")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(length < 2 for length in args.lengths):
        raise ValueError("all lengths must be at least two")
    if any(dimension < 1 for dimension in args.bond_dims):
        raise ValueError("all bond dimensions must be positive")
    if args.sweeps < 1:
        raise ValueError("sweeps must be positive")

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "open spin-1/2 antiferromagnetic Heisenberg chain, J=1",
        "protocol": {
            "lengths": list(args.lengths),
            "bond_dimensions": list(args.bond_dims),
            "passes": int(args.sweeps),
            "energy_tolerance": float(args.tol),
            "mps_initial_state": "Neel product state",
            "mps_noise": 1.0e-6,
            "mps_performance": "legacy-auto",
            "mps_recenter_final": False,
            "letta_seeds": list(args.seeds),
            "letta_gauge": args.letta_gauge,
            "letta_selection": "lowest final energy",
            "exact_sector": "n_up=floor(L/2)",
            "exact_tolerance": float(args.ed_tol),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "git_revision": git_revision(),
        },
        "results": [],
    }

    for length in args.lengths:
        exact = exact_ground_state(length, args.ed_tol)
        exact_energy = float(exact["energy"])
        length_record = {"length": int(length), "exact": exact, "runs": []}
        payload["results"].append(length_record)
        print(
            f"L={length:2d} exact={exact_energy:.15f} "
            f"dim={exact['sector_dimension']:,} residual={exact['residual_norm']:.2e}",
            flush=True,
        )

        model = Heisenberg(L=length)
        mpo = model.build_H_mpo().factors
        for bond_dim in args.bond_dims:
            mps = mps_run(
                mpo,
                model.build_neel_state(),
                bond_dim,
                args.sweeps,
                args.tol,
            )
            mps["error"] = float(mps["energy"] - exact_energy)

            starts = [
                letta_run(
                    mpo,
                    length,
                    bond_dim,
                    seed,
                    args.sweeps,
                    args.tol,
                    args.letta_gauge,
                )
                for seed in args.seeds
            ]
            selected = dict(min(starts, key=lambda record: record["energy"]))
            for record in starts:
                record["error"] = float(record["energy"] - exact_energy)
            selected["error"] = float(selected["energy"] - exact_energy)

            run = {
                "bond_dim": int(bond_dim),
                "mps": mps,
                "letta_starts": starts,
                "letta_selected": selected,
            }
            length_record["runs"].append(run)
            write_results(payload, args.output)
            print(
                f"  D={bond_dim:<2d} "
                f"MPS err={mps['error']:.6e} conv={str(mps['converged']):5s} "
                f"| LETTA err={selected['error']:.6e} seed={selected['seed']} "
                f"conv={str(selected['converged']):5s}",
                flush=True,
            )

    write_results(payload, args.output)
    csv_path = write_csv(payload, args.output)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
