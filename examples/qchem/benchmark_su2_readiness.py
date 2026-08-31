#!/usr/bin/env python3
"""Readiness benchmark for the qchem SU(2) DMRG backend.

This script compares small SU(2)-adapted DMRG calculations against exact
diagonalization in the same active space. It is intended as a practical gate
before promoting the backend to production use: a case is "ready" only if the
reported DMRG energy, the final MPS expectation value, and the exact target-spin
energy agree within the requested tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/private/tmp/pyqed-matplotlib-cache")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_home = Path("/private/tmp/pyqed-cache")
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.mps.nonabelian.canonical import mixed_canonical_errors
from pyqed.mps.nonabelian.environment import contract_chain_expectation
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.ed import _spin_adapted_dense_roots
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.hf import RHF


PRESETS = {
    "h2": {
        "atom": "H 0 0 0; H 0 0 1.4",
        "unit": "bohr",
        "ncas": 2,
        "nelecas": 2,
        "spin": 0,
    },
    "h4": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        "unit": "bohr",
        "ncas": 4,
        "nelecas": 4,
        "spin": 0,
    },
    "h6": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8; H 0 0 6.4; H 0 0 8.0",
        "unit": "bohr",
        "ncas": 6,
        "nelecas": 6,
        "spin": 0,
    },
}


@dataclass
class SU2ReadinessResult:
    """Compact result for one SU(2) DMRG readiness case."""

    system: str
    basis: str
    ncas: int
    nelecas: int
    spin: int
    bond_dim: int
    nsweeps: int
    exact_energy: float
    dmrg_energy: float
    mps_energy: float
    energy_error: float
    mps_energy_error: float
    elapsed: float
    converged: bool
    ncompleted: int
    target_charge: int
    target_two_s: int
    canonical_center: int
    canonical_left_error: float
    canonical_right_error: float
    final_metric: float | None
    final_truncation_error: float | None
    max_bond_operator_time: float
    max_local_matvec_time: float
    max_local_solve_time: float
    family_backend_counts: dict
    family_native_kernel_elements: int
    family_factor_kernel_elements: int
    family_dense_skipped_total_budget: int
    family_dense_skipped_threshold: int
    passed: bool


def _scalar(value):
    """Return the first scalar from a possibly array-like energy."""

    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _best_history_entry(history):
    """Return the finite-energy history entry with the lowest sweep energy."""

    finite = [
        entry
        for entry in history
        if entry.get("energy") is not None and np.isfinite(float(entry["energy"]))
    ]
    if not finite:
        return history[-1] if history else {}
    return min(finite, key=lambda entry: float(entry["energy"]))


def _infer_canonical_center(history, nsites):
    """Infer the mixed-canonical center from the direction of the best sweep."""

    if int(nsites) <= 1:
        return 0
    direction = str(_best_history_entry(history).get("direction", "lr")).lower()
    return int(nsites) - 1 if direction == "lr" else 0


def _final_truncation_error(history):
    """Return the maximum truncation error recorded in the last sweep."""

    if not history:
        return None
    values = [
        float(update["trunc_err"])
        for update in history[-1].get("updates", [])
        if update.get("trunc_err") is not None
    ]
    return max(values) if values else None


def _max_timing(history, key):
    """Return the maximum per-sweep timing value for one timing key."""

    return max(
        (
            float((entry.get("timing") or {}).get(key, 0.0))
            for entry in history
        ),
        default=0.0,
    )


def _family_kernel_diagnostics(history):
    """Collect complementary-family backend/storage diagnostics from history."""

    backend_counts = {}
    native_elements = 0
    factor_elements = 0
    skipped_total = 0
    skipped_threshold = 0
    for entry in history:
        for objective in entry.get("bond_objectives", []) or []:
            stats = objective.get("renormalized_operator_table_stats") or {}
            family_table = stats.get("complementary_family_table") or {}
            if not family_table:
                continue
            backend = family_table.get("backend")
            if backend is None:
                # Packed C++ factor routes expose their source metadata here,
                # but they are not a ComplementaryFamilyTensorTable backend.
                continue
            backend = str(backend)
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            native_elements += int(family_table.get("native_kernel_elements", 0))
            factor_elements += int(family_table.get("factor_kernel_elements", 0))
            skipped_total += int(family_table.get("dense_kernel_skipped_total_budget", 0))
            skipped_threshold += int(family_table.get("dense_kernel_skipped_threshold", 0))
    return {
        "backend_counts": dict(sorted(backend_counts.items())),
        "native_elements": int(native_elements),
        "factor_elements": int(factor_elements),
        "skipped_total": int(skipped_total),
        "skipped_threshold": int(skipped_threshold),
    }


def _mps_expectation_energy(qcdmrg):
    """Evaluate the final non-Abelian MPS energy from the Hamiltonian MPO."""

    state = qcdmrg.dmrg.ground_state
    numerator = contract_chain_expectation(state.sites, qcdmrg.H)
    denominator = contract_chain_expectation(
        state.sites,
        _identity_mpo_factors_for_sites_and_mpo(state.sites, qcdmrg.H),
    )
    denom = float(np.real(denominator))
    if abs(denom) < 1.0e-15:
        raise ValueError("Final SU(2) DMRG MPS has near-zero norm.")
    return float(np.real(numerator / denominator)) + float(qcdmrg.e_core)


def exact_target_energy(qcdmrg, *, max_dense_dim=4096, spin_tol=1.0e-7):
    """Return the lowest exact active-space root in the requested SU(2) sector."""

    active_roots, _s2_values, _states = _spin_adapted_dense_roots(
        qcdmrg,
        1,
        max_dense_dim=int(max_dense_dim),
        spin_tol=float(spin_tol),
    )
    return float(active_roots[0]) + float(qcdmrg.e_core)


def run_case(
    system,
    *,
    basis="sto-3g",
    bond_dim=16,
    nsweeps=4,
    energy_tol=1.0e-7,
    max_dense_dim=4096,
    local_basis_policy="block2_like",
    orthonormalized_operator_dim=512,
    family_kernel_backend=None,
    family_dense_threshold=None,
    family_dense_max_total_elements=None,
    conv_tol=-1.0,
    require_convergence=False,
):
    """Run one SU(2) readiness case and return a structured result."""

    if system not in PRESETS:
        raise ValueError(f"Unknown readiness preset {system!r}; choose one of {sorted(PRESETS)}.")
    case = PRESETS[system]
    mol = Molecule(atom=case["atom"], unit=case["unit"], basis=basis, spin=case["spin"])
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    build_info = mol._builtin_build_info
    if (
        build_info.get("eri_backend") != "cpp"
        or not str(build_info.get("dense_builder", "")).startswith("cpp-")
    ):
        raise RuntimeError("The SU(2) readiness benchmark requires the compiled C++ ERI backend.")
    mf = RHF(mol).run()

    qcdmrg = DMRG(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=int(bond_dim),
        init_guess="cid",
        symmetry="su2",
        verbose=0,
    )
    qcdmrg.build()
    exact_energy = exact_target_energy(qcdmrg, max_dense_dim=max_dense_dim)

    t0 = time.perf_counter()
    run_kwargs = {
        "nsweeps": int(nsweeps),
        "conv_tol": float(conv_tol),
        "require_convergence": bool(require_convergence),
        "local_basis_policy": local_basis_policy,
        "orthonormalized_operator_dim": int(orthonormalized_operator_dim),
        "max_bond_mode": "per_sector",
        "mixer_zero_block_noise_scale": 0.0,
        "profile": True,
    }
    if family_kernel_backend is not None:
        run_kwargs["family_kernel_backend"] = family_kernel_backend
    if family_dense_threshold is not None:
        run_kwargs["family_dense_threshold"] = int(family_dense_threshold)
    if family_dense_max_total_elements is not None:
        run_kwargs["family_dense_max_total_elements"] = int(
            family_dense_max_total_elements
        )
    qcdmrg.run(**run_kwargs)
    elapsed = time.perf_counter() - t0

    history = getattr(qcdmrg.dmrg, "history", []) or []
    dmrg_energy = _scalar(qcdmrg.e_tot)
    mps_energy = _mps_expectation_energy(qcdmrg)
    energy_error = abs(dmrg_energy - exact_energy)
    mps_energy_error = abs(mps_energy - exact_energy)

    state = qcdmrg.dmrg.ground_state
    center = _infer_canonical_center(history, len(state.sites))
    left_err, right_err = mixed_canonical_errors(state.sites, center)
    target = qcdmrg.dmrg.target_sector
    family_diagnostics = _family_kernel_diagnostics(history)

    result = SU2ReadinessResult(
        system=system,
        basis=basis,
        ncas=int(case["ncas"]),
        nelecas=int(case["nelecas"]),
        spin=int(case["spin"]),
        bond_dim=int(bond_dim),
        nsweeps=int(nsweeps),
        exact_energy=float(exact_energy),
        dmrg_energy=float(dmrg_energy),
        mps_energy=float(mps_energy),
        energy_error=float(energy_error),
        mps_energy_error=float(mps_energy_error),
        elapsed=float(elapsed),
        converged=bool(qcdmrg.dmrg.converged),
        ncompleted=int(qcdmrg.dmrg.ncompleted),
        target_charge=int(target.charge),
        target_two_s=int(target.irrep.two_j),
        canonical_center=int(center),
        canonical_left_error=float(left_err),
        canonical_right_error=float(right_err),
        final_metric=(
            None
            if not history or history[-1].get("metric") is None
            else float(history[-1]["metric"])
        ),
        final_truncation_error=_final_truncation_error(history),
        max_bond_operator_time=float(_max_timing(history, "bond_operator")),
        max_local_matvec_time=float(_max_timing(history, "local_matvec")),
        max_local_solve_time=float(_max_timing(history, "update_local_solve")),
        family_backend_counts=family_diagnostics["backend_counts"],
        family_native_kernel_elements=family_diagnostics["native_elements"],
        family_factor_kernel_elements=family_diagnostics["factor_elements"],
        family_dense_skipped_total_budget=family_diagnostics["skipped_total"],
        family_dense_skipped_threshold=family_diagnostics["skipped_threshold"],
        passed=bool(energy_error <= energy_tol and mps_energy_error <= energy_tol),
    )
    return result


def format_result(result):
    """Return a compact human-readable report for one result."""

    status = "PASS" if result.passed else "FAIL"
    return "\n".join(
        [
            (
                f"{status} {result.system} {result.basis}: "
                f"E_DMRG={result.dmrg_energy:.12f} "
                f"E_MPS={result.mps_energy:.12f} "
                f"E_ED={result.exact_energy:.12f}"
            ),
            (
                f"  errors: reported={result.energy_error:.3e} "
                f"mps={result.mps_energy_error:.3e}; "
                f"target=(N={result.target_charge}, 2S={result.target_two_s})"
            ),
            (
                f"  sweeps: completed={result.ncompleted}/{result.nsweeps} "
                f"converged={result.converged} metric={result.final_metric} "
                f"trunc={result.final_truncation_error}"
            ),
            (
                f"  kernels: backends={result.family_backend_counts} "
                f"dense_elems={result.family_native_kernel_elements} "
                f"factor_elems={result.family_factor_kernel_elements} "
                f"skipped_total={result.family_dense_skipped_total_budget} "
                f"skipped_threshold={result.family_dense_skipped_threshold}"
            ),
            (
                f"  timing: bond_operator={result.max_bond_operator_time:.3f}s "
                f"local_matvec={result.max_local_matvec_time:.3f}s "
                f"local_solve={result.max_local_solve_time:.3f}s"
            ),
            (
                f"  canonical: center={result.canonical_center} "
                f"left_err={result.canonical_left_error:.3e} "
                f"right_err={result.canonical_right_error:.3e}; "
                f"time={result.elapsed:.3f}s"
            ),
        ]
    )


def main():
    """Run the readiness benchmark from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(PRESETS), action="append")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--nsweeps", type=int, default=4)
    parser.add_argument("--energy-tol", type=float, default=1.0e-7)
    parser.add_argument("--max-dense-dim", type=int, default=4096)
    parser.add_argument("--local-basis-policy", default="block2_like")
    parser.add_argument("--orthonormalized-operator-dim", type=int, default=512)
    parser.add_argument("--family-kernel-backend", choices=["auto", "dense", "factor"], default=None)
    parser.add_argument("--family-dense-threshold", type=int, default=None)
    parser.add_argument("--family-dense-max-total-elements", type=int, default=None)
    parser.add_argument("--conv-tol", type=float, default=-1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    systems = args.system or ["h2", "h4"]
    results = [
        run_case(
            system,
            basis=args.basis,
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            energy_tol=args.energy_tol,
            max_dense_dim=args.max_dense_dim,
            local_basis_policy=args.local_basis_policy,
            orthonormalized_operator_dim=args.orthonormalized_operator_dim,
            family_kernel_backend=args.family_kernel_backend,
            family_dense_threshold=args.family_dense_threshold,
            family_dense_max_total_elements=args.family_dense_max_total_elements,
            conv_tol=args.conv_tol,
        )
        for system in systems
    ]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        for index, result in enumerate(results):
            if index:
                print()
            print(format_result(result))

    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
