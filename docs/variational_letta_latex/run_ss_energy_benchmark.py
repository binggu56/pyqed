#!/usr/bin/env python3
"""Run small Shastry-Sutherland LETTA/MPS energy benchmarks.

The script uses a generic long-range spin MPO after choosing a 1D site order.
It is meant to produce reproducible manuscript data, not a highly optimized
production DMRG implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.letta import LETTA
from pyqed.mps.dmrg import DMRG
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.tn import MPO

from compare_ss_orderings import dimer_first_order, snake_order, square_edges, ss_dimers


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


IDENTITY = np.eye(2, dtype=float)
SX = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
SY_REAL_TERMS = (
    np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float),
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float),
)
SZ = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
SINGLET = (1.0 / np.sqrt(2.0)) * np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)


class LETTAConvergenceError(RuntimeError):
    """Raised when a LETTA calculation reaches the sweep cap without converging."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


class MPSConvergenceError(RuntimeError):
    """Raised when an MPS/DMRG calculation reaches the sweep cap without converging."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def unique_edges(edges):
    seen = set()
    out = []
    for a, b in edges:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def ss_bonds(lx: int, ly: int):
    """Return physical-coordinate SS bonds as ``(site_a, site_b, coupling, kind)``."""
    bonds = []
    for a, b in unique_edges(ss_dimers(lx, ly)):
        bonds.append((a, b, 1.0, "J"))
    for a, b in unique_edges(square_edges(lx, ly)):
        bonds.append((a, b, None, "Jp"))
    return bonds


def ordered_bonds(lx: int, ly: int, order_name: str, jprime: float):
    order = dimer_first_order(lx, ly) if order_name == "dimer-first" else snake_order(lx, ly)
    index = {site: i for i, site in enumerate(order)}
    out = []
    for a, b, coupling, kind in ss_bonds(lx, ly):
        ia = index[a]
        ib = index[b]
        if ia == ib:
            raise ValueError("invalid repeated site in bond")
        if ia > ib:
            ia, ib = ib, ia
        value = 1.0 if coupling is not None else float(jprime)
        if abs(value) > 1.0e-15:
            out.append((ia, ib, value, kind))
    return out


def sparse_spin_hamiltonian(nsites: int, bonds):
    """Sparse spin-1/2 Heisenberg Hamiltonian from 1D-ordered bonds."""
    dim = 1 << int(nsites)
    diag = np.zeros(dim, dtype=float)
    rows = []
    cols = []
    data = []
    for i, j, coupling, _kind in bonds:
        mask = (1 << i) | (1 << j)
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            zi = 0.5 if bi == 0 else -0.5
            zj = 0.5 if bj == 0 else -0.5
            diag[state] += coupling * zi * zj
            if bi != bj:
                flipped = state ^ mask
                rows.append(flipped)
                cols.append(state)
                data.append(0.5 * coupling)
    rows.extend(range(dim))
    cols.extend(range(dim))
    data.extend(diag)
    return coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()


def heisenberg_long_range_mpo(nsites: int, bonds) -> list[np.ndarray]:
    """Exact MPO for arbitrary two-site Heisenberg bonds in the current order."""
    components = []
    for i, j, coupling, _kind in bonds:
        if i > j:
            i, j = j, i
        components.append((i, j, SX, SX, coupling))
        components.append((i, j, SY_REAL_TERMS[0], SY_REAL_TERMS[1], 0.25 * coupling))
        components.append((i, j, SZ, SZ, coupling))

    full_dim = 2 + len(components)
    final = full_dim - 1
    factors = []
    for site in range(nsites):
        core = np.zeros((full_dim, full_dim, 2, 2), dtype=float)
        core[0, 0] = IDENTITY
        core[final, final] = IDENTITY
        for channel, (start, stop, left_op, right_op, coeff) in enumerate(components, start=1):
            if site == start:
                core[0, channel] += left_op
            elif start < site < stop:
                core[channel, channel] += IDENTITY
            elif site == stop:
                core[channel, final] += float(coeff) * right_op
        if site == 0:
            core = core[0:1]
        if site == nsites - 1:
            core = core[:, final : final + 1]
        factors.append(core)
    return factors


def validate_mpo(nsites: int, bonds, *, atol: float = 1.0e-12):
    if nsites > 10:
        return None
    sparse = sparse_spin_hamiltonian(nsites, bonds).toarray()
    dense_from_mpo = _mpo_to_dense_operator(MPO(heisenberg_long_range_mpo(nsites, bonds)))
    err = float(np.max(np.abs(sparse - dense_from_mpo)))
    if err > atol:
        raise AssertionError(f"long-range MPO validation failed: max error {err:.3e}")
    return err


def dimer_product_letta(nsites: int, *, bond_dim: int = 1, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    bonds = [1] + [int(bond_dim)] * (nsites - 2) + [1]
    tensors = []
    for pair in range(nsites - 1):
        tensor = np.zeros((bonds[pair], 2, 2, bonds[pair + 1]), dtype=float)
        base = SINGLET if pair % 2 == 0 else np.ones((2, 2), dtype=float)
        tensor[0, :, :, 0] = base
        if noise > 0.0 and tensor.size > 4:
            tensor += float(noise) * rng.normal(size=tensor.shape)
            tensor[0, :, :, 0] = base
        tensors.append(tensor)
    return LETTA(None, (2,) * nsites, bond_dim=bond_dim, tensors=tensors)


def random_mps(nsites: int, bond_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    bonds = [1] + [int(bond_dim)] * (nsites - 1) + [1]
    factors = []
    for site in range(nsites):
        shape = (bonds[site], 2, bonds[site + 1])
        tensor = rng.normal(size=shape)
        norm = np.linalg.norm(tensor.reshape(-1))
        factors.append(tensor / max(norm, 1.0e-14))
    return factors


def exact_energy(nsites: int, bonds, max_dim: int):
    dim = 1 << int(nsites)
    if dim > int(max_dim):
        return None
    hamiltonian = sparse_spin_hamiltonian(nsites, bonds)
    value = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False, tol=1.0e-10)[0]
    return float(value)


def final_energy_delta(history):
    energies = []
    for row in history:
        if row.get("energy") is None:
            continue
        try:
            energies.append(float(np.real(np.asarray(row["energy"]).reshape(-1)[0])))
        except Exception:
            continue
    if len(energies) < 2:
        return None
    return abs(energies[-1] - energies[-2])


def run_mps_dmrg(
    nsites: int,
    mpo,
    *,
    bond_dim: int,
    max_sweeps: int,
    seed: int,
    tol: float,
    require_converged: bool = True,
):
    start = perf_counter()
    max_sweeps = int(max_sweeps)
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive.")
    solver = DMRG(
        mpo,
        D=int(bond_dim),
        init_guess=random_mps(nsites, bond_dim, seed),
        nsweeps=max_sweeps,
        not_conv_err=False,
        verbose=0,
        sweep_tol=float(tol),
        davidson_tol=1.0e-8,
        davidson_max_iter=40,
        noise=0.0,
        recenter_final=False,
        performance="legacy-auto",
    )
    solver.run()
    final_delta = final_energy_delta(solver.sweep_history)
    result = {
        "energy": float(solver.e_tot),
        "seconds": perf_counter() - start,
        "converged": bool(solver.converged),
        "sweeps_completed": len(solver.sweep_history),
        "final_delta_energy": final_delta,
        "convergence_tol": float(tol),
        "max_sweeps": max_sweeps,
    }
    if require_converged and not solver.converged:
        delta_text = "None" if final_delta is None else f"{float(final_delta):.3e}"
        raise MPSConvergenceError(
            "MPS/DMRG did not converge within "
            f"{max_sweeps} sweeps (final dE={delta_text}, tol={float(tol):.3e}, "
            f"E={float(solver.e_tot):.12g})",
            result,
        )
    return result


def run_letta(
    nsites: int,
    mpo,
    *,
    bond_dim: int,
    max_sweeps: int,
    seed: int,
    tol: float,
    require_converged: bool = True,
):
    start = perf_counter()
    max_sweeps = int(max_sweeps)
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive.")
    letta = dimer_product_letta(
        nsites,
        bond_dim=int(bond_dim),
        noise=0.0 if int(bond_dim) == 1 else 1.0e-3,
        seed=seed,
    )
    initial = letta.expectation(mpo)
    letta.run(
        mpo,
        nsweeps=max_sweeps,
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
        "sweeps_completed": letta.ncompleted,
        "converged": bool(letta.converged),
        "final_delta_energy": None if final_delta is None else float(final_delta),
        "convergence_tol": float(tol),
        "max_sweeps": max_sweeps,
    }
    if require_converged and not letta.converged:
        delta_text = "None" if final_delta is None else f"{float(final_delta):.3e}"
        raise LETTAConvergenceError(
            "LETTA did not converge within "
            f"{max_sweeps} sweeps (final dE={delta_text}, tol={float(tol):.3e}, "
            f"E={float(energy):.12g})",
            result,
        )
    return result


def parse_ints(text: str):
    return [int(item) for item in str(text).split(",") if item.strip()]


def parse_floats(text: str):
    return [float(item) for item in str(text).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--jprime", default="0.0,0.3,0.5,0.7,1.0")
    parser.add_argument("--mps-d", default="1,2")
    parser.add_argument("--letta-d", default="1,2")
    parser.add_argument("--mps-sweeps", type=int, default=None, help="Deprecated alias for --mps-max-sweeps.")
    parser.add_argument(
        "--mps-max-sweeps",
        type=int,
        default=None,
        help="Maximum number of MPS/DMRG sweeps before reporting non-convergence.",
    )
    parser.add_argument(
        "--mps-tol",
        type=float,
        default=1.0e-8,
        help="MPS/DMRG sweep energy convergence tolerance.",
    )
    parser.add_argument(
        "--allow-unconverged-mps",
        action="store_true",
        help="Keep unconverged MPS/DMRG energies for diagnostic runs instead of writing a failure row.",
    )
    parser.add_argument(
        "--letta-sweeps",
        type=int,
        default=None,
        help="Deprecated alias for --letta-max-sweeps.",
    )
    parser.add_argument(
        "--letta-max-sweeps",
        type=int,
        default=None,
        help="Maximum number of LETTA sweeps before reporting non-convergence.",
    )
    parser.add_argument(
        "--letta-tol",
        type=float,
        default=1.0e-9,
        help="LETTA sweep energy convergence tolerance.",
    )
    parser.add_argument(
        "--allow-unconverged-letta",
        action="store_true",
        help="Keep unconverged LETTA energies for diagnostic runs instead of writing a failure row.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ed-max-dim", type=int, default=131072)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    lx = int(args.lx)
    ly = int(args.ly)
    nsites = lx * ly
    mps_max_sweeps = args.mps_max_sweeps
    if mps_max_sweeps is None:
        mps_max_sweeps = args.mps_sweeps if args.mps_sweeps is not None else 100
    letta_max_sweeps = args.letta_max_sweeps
    if letta_max_sweeps is None:
        letta_max_sweeps = args.letta_sweeps if args.letta_sweeps is not None else 50
    output_prefix = args.output_prefix or f"ss_energy_Lx{lx}_Ly{ly}"
    csv_path = RESULTS_DIR / f"{output_prefix}.csv"
    json_path = RESULTS_DIR / f"{output_prefix}.json"

    rows = []
    metadata = {
        "lx": lx,
        "ly": ly,
        "nsites": nsites,
        "geometry": "open x, periodic y",
        "mps_order": "snake",
        "letta_order": "dimer-first",
        "mps_max_sweeps": int(mps_max_sweeps),
        "mps_tol": float(args.mps_tol),
        "require_mps_convergence": not bool(args.allow_unconverged_mps),
        "letta_max_sweeps": int(letta_max_sweeps),
        "letta_tol": float(args.letta_tol),
        "require_letta_convergence": not bool(args.allow_unconverged_letta),
        "seed": int(args.seed),
    }

    for jprime in parse_floats(args.jprime):
        mps_bonds = ordered_bonds(lx, ly, "snake", jprime)
        letta_bonds = ordered_bonds(lx, ly, "dimer-first", jprime)
        validation_error = validate_mpo(nsites, letta_bonds)
        exact = exact_energy(nsites, letta_bonds, args.ed_max_dim)
        print(f"# J'/J={jprime:g} | exact={exact if exact is not None else 'skipped'}")

        mps_mpo = heisenberg_long_range_mpo(nsites, mps_bonds)
        letta_mpo = heisenberg_long_range_mpo(nsites, letta_bonds)

        for bond_dim in parse_ints(args.mps_d):
            try:
                result = run_mps_dmrg(
                    nsites,
                    mps_mpo,
                    bond_dim=bond_dim,
                    max_sweeps=mps_max_sweeps,
                    seed=args.seed + 17 * bond_dim + int(round(1000 * jprime)),
                    tol=args.mps_tol,
                    require_converged=not args.allow_unconverged_mps,
                )
                row = {
                    "method": "MPS/DMRG snake",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": result["energy"],
                    "initial": "",
                    "exact": "" if exact is None else exact,
                    "error": "" if exact is None else result["energy"] - exact,
                    "seconds": result["seconds"],
                    "sweeps_completed": result["sweeps_completed"],
                    "converged": result["converged"],
                    "final_delta_energy": ""
                    if result["final_delta_energy"] is None
                    else result["final_delta_energy"],
                    "convergence_tol": result["convergence_tol"],
                    "max_sweeps": result["max_sweeps"],
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                }
            except MPSConvergenceError as exc:
                result = exc.result
                row = {
                    "method": "MPS/DMRG snake",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": "",
                    "initial": "",
                    "exact": "" if exact is None else exact,
                    "error": "",
                    "seconds": result["seconds"],
                    "sweeps_completed": result["sweeps_completed"],
                    "converged": result["converged"],
                    "final_delta_energy": ""
                    if result["final_delta_energy"] is None
                    else result["final_delta_energy"],
                    "convergence_tol": result["convergence_tol"],
                    "max_sweeps": result["max_sweeps"],
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                    "failure": str(exc),
                }
            except Exception as exc:
                row = {
                    "method": "MPS/DMRG snake",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": "",
                    "initial": "",
                    "exact": "" if exact is None else exact,
                    "error": "",
                    "seconds": "",
                    "sweeps_completed": "",
                    "converged": False,
                    "final_delta_energy": "",
                    "convergence_tol": float(args.mps_tol),
                    "max_sweeps": int(mps_max_sweeps),
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if row.get("failure"):
                print(f"  {row['method']:16s} D={bond_dim:<2d} FAILED {row['failure']}")
            else:
                print(f"  {row['method']:16s} D={bond_dim:<2d} E={row.get('energy')} error={row.get('error')}")

        for bond_dim in parse_ints(args.letta_d):
            try:
                result = run_letta(
                    nsites,
                    letta_mpo,
                    bond_dim=bond_dim,
                    max_sweeps=letta_max_sweeps,
                    seed=args.seed + 31 * bond_dim + int(round(1000 * jprime)),
                    tol=args.letta_tol,
                    require_converged=not args.allow_unconverged_letta,
                )
                row = {
                    "method": "LETTA dimer-first",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": result["energy"],
                    "initial": result["initial"],
                    "exact": "" if exact is None else exact,
                    "error": "" if exact is None else result["energy"] - exact,
                    "seconds": result["seconds"],
                    "sweeps_completed": result["sweeps_completed"],
                    "converged": result["converged"],
                    "final_delta_energy": ""
                    if result["final_delta_energy"] is None
                    else result["final_delta_energy"],
                    "convergence_tol": result["convergence_tol"],
                    "max_sweeps": result["max_sweeps"],
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                }
            except LETTAConvergenceError as exc:
                result = exc.result
                row = {
                    "method": "LETTA dimer-first",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": "",
                    "initial": result["initial"],
                    "exact": "" if exact is None else exact,
                    "error": "",
                    "seconds": result["seconds"],
                    "sweeps_completed": result["sweeps_completed"],
                    "converged": result["converged"],
                    "final_delta_energy": ""
                    if result["final_delta_energy"] is None
                    else result["final_delta_energy"],
                    "convergence_tol": result["convergence_tol"],
                    "max_sweeps": result["max_sweeps"],
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                    "failure": str(exc),
                }
            except Exception as exc:
                row = {
                    "method": "LETTA dimer-first",
                    "D": bond_dim,
                    "jprime": jprime,
                    "energy": "",
                    "initial": "",
                    "exact": "" if exact is None else exact,
                    "error": "",
                    "seconds": "",
                    "sweeps_completed": "",
                    "converged": False,
                    "final_delta_energy": "",
                    "convergence_tol": float(args.letta_tol),
                    "max_sweeps": int(letta_max_sweeps),
                    "mpo_validation_error": "" if validation_error is None else validation_error,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if row.get("failure"):
                print(f"  {row['method']:16s} D={bond_dim:<2d} FAILED {row['failure']}")
            else:
                print(f"  {row['method']:16s} D={bond_dim:<2d} E={row.get('energy')} error={row.get('error')}")

    fieldnames = [
        "method",
        "D",
        "jprime",
        "energy",
        "initial",
        "exact",
        "error",
        "seconds",
        "sweeps_completed",
        "converged",
        "final_delta_energy",
        "convergence_tol",
        "max_sweeps",
        "mpo_validation_error",
        "failure",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)
    print(f"# wrote {csv_path}")
    print(f"# wrote {json_path}")


if __name__ == "__main__":
    main()
