#!/usr/bin/env python3
"""Benchmark native pyqed second-order CASSCF coupling modes."""

import argparse
import contextlib
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule, SecondOrderCASSCF
from pyqed.qchem.mcscf.direct_ci import CASCI


PRESETS = {
    "h2": {
        "atom": "H 0 0 0; H 0 0 1.4",
        "unit": "bohr",
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "rotation": (0, 1),
    },
    "lih": {
        "atom": "Li 0 0 0; H 0 0 1.6",
        "unit": "angstrom",
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "rotation": (1, 3),
    },
    "lih44": {
        "atom": "Li 0 0 0; H 0 0 1.6",
        "unit": "angstrom",
        "basis": "sto-3g",
        "ncas": 4,
        "nelecas": 4,
        "rotation": (1, 4),
    },
    "ethylene22": {
        "atom": [
            ["C", 0.00000000, 0.00000000, 0.66796400],
            ["H", 0.92288300, 0.00000000, 1.24294900],
            ["H", -0.92288300, 0.00000000, 1.24294900],
            ["C", 0.00000000, 0.00000000, -0.66796400],
            ["H", 0.54030916, 0.92288300, -0.86462045],
            ["H", 0.54030916, -0.92288300, -0.86462045],
        ],
        "unit": "angstrom",
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "rotation": (7, 10),
    },
    "ethylene44": {
        "atom": [
            ["C", 0.00000000, 0.00000000, 0.66796400],
            ["H", 0.92288300, 0.00000000, 1.24294900],
            ["H", -0.92288300, 0.00000000, 1.24294900],
            ["C", 0.00000000, 0.00000000, -0.66796400],
            ["H", 0.54030916, 0.92288300, -0.86462045],
            ["H", 0.54030916, -0.92288300, -0.86462045],
        ],
        "unit": "angstrom",
        "basis": "sto-3g",
        "ncas": 4,
        "nelecas": 4,
        "rotation": (6, 10),
    },
}


def _parse_modes(text):
    modes = [item.strip().lower().replace("-", "_") for item in text.split(",")]
    modes = [mode for mode in modes if mode]
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required.")
    allowed = {
        "qn",
        "uncoupled",
        "partial",
        "full",
        "simultaneous",
        "simultaneous_reduced",
        "relaxed_fd",
    }
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(
            "Unknown coupling mode(s): {}. Allowed: {}".format(
                ", ".join(invalid),
                ", ".join(sorted(allowed)),
            )
        )
    return modes


def _parse_active_orbitals(text):
    if text is None or str(text).strip().lower() in {"", "none", "default"}:
        return None
    try:
        active = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--active-orbitals must be a comma-separated list of zero-based MO indices."
        ) from exc
    if not active:
        return None
    return active


def _make_molecule(case, basis):
    mol = Molecule(atom=case["atom"], unit=case["unit"], basis=basis or case["basis"])
    mol.build(driver="gbasis")
    return mol


@contextlib.contextmanager
def _solver_output(enabled):
    if enabled:
        yield
        return

    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(logging.NOTSET)


def _rotated_mo(mo_coeff, pair, angle):
    mo_coeff = np.asarray(mo_coeff)
    kappa = np.zeros((mo_coeff.shape[1], mo_coeff.shape[1]))
    i, j = pair
    if i >= mo_coeff.shape[1] or j >= mo_coeff.shape[1]:
        raise ValueError(
            "Orbital rotation pair {} is incompatible with {} MOs.".format(
                pair,
                mo_coeff.shape[1],
            )
        )
    kappa[i, j] = angle
    kappa[j, i] = -angle
    return mo_coeff @ expm(kappa)


def _active_nelectron_count(nelecas):
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]) + int(nelecas[1])
    return int(nelecas)


def _reference_nelectron_count(mf):
    nelec = getattr(mf, "nelec", None)
    if nelec is not None:
        if isinstance(nelec, (tuple, list)):
            return int(nelec[0]) + int(nelec[1])
        return int(nelec)
    mol = getattr(mf, "mol", None)
    if mol is not None and hasattr(mol, "nelectron"):
        return int(mol.nelectron)
    return int(round(float(np.sum(mf.mo_occ))))


def _reorder_mo_for_active_orbitals(mo_coeff, mf, case, active_orbitals):
    if active_orbitals is None:
        return np.asarray(mo_coeff)
    active = [int(idx) for idx in active_orbitals]
    if len(active) != int(case["ncas"]):
        raise ValueError(
            "--active-orbitals must contain exactly {} entries.".format(case["ncas"])
        )
    if len(set(active)) != len(active):
        raise ValueError("--active-orbitals contains duplicate indices.")
    nmo = mo_coeff.shape[1]
    if min(active) < 0 or max(active) >= nmo:
        raise ValueError("--active-orbitals contains an out-of-range MO index.")
    ncore2 = _reference_nelectron_count(mf) - _active_nelectron_count(case["nelecas"])
    if ncore2 < 0 or ncore2 % 2:
        raise ValueError("Inconsistent reference/active electron counts.")
    ncore = ncore2 // 2
    active_set = set(active)
    rest = [idx for idx in range(nmo) if idx not in active_set]
    order = rest[:ncore] + active + rest[ncore:]
    return np.asarray(mo_coeff)[:, order]


def _last_history_value(history, key):
    if not history:
        return np.nan
    return float(history[-1].get(key, np.nan))


def _internal_preopt_stats(history):
    attempted = len(history)
    accepted = sum(1 for record in history if record.get("accepted", False))
    guard_rejected = sum(1 for record in history if record.get("rejected_by_guard", False))
    guarded = [record for record in history if "guard_after_energy" in record]
    if guarded:
        last = guarded[-1]
        guard_delta = float(last["guard_after_energy"] - last["guard_before_energy"])
    else:
        guard_delta = np.nan
    coupled = [record for record in history if record.get("coupled_ci_dim", 0) > 0]
    if coupled:
        last_coupled = coupled[-1]
        ci_dim = int(last_coupled.get("coupled_ci_dim", 0))
        q_cycles = int(last_coupled.get("coupled_q_cycles", 0))
        relaxed_min_eig = float(last_coupled.get("coupled_relaxed_min_eig", np.nan))
        fallback_count = sum(
            1 for record in coupled if record.get("coupled_fallback_diagonal", False)
        )
    else:
        ci_dim = 0
        q_cycles = 0
        relaxed_min_eig = np.nan
        fallback_count = 0
    post_gradients = [
        float(record["post_gradient_norm"])
        for record in history
        if "post_gradient_norm" in record
    ]
    last_post_grad = post_gradients[-1] if post_gradients else np.nan
    converged = any(record.get("internal_converged", False) for record in history)
    solver_records = [record for record in history if "solver_iterations" in record]
    if solver_records:
        last_solver = solver_records[-1]
        solver_iterations = int(last_solver.get("solver_iterations", 0))
        solver_residual = float(last_solver.get("solver_residual_norm", np.nan))
    else:
        solver_iterations = 0
        solver_residual = np.nan
    return (
        attempted,
        accepted,
        guard_rejected,
        guard_delta,
        ci_dim,
        q_cycles,
        relaxed_min_eig,
        fallback_count,
        last_post_grad,
        converged,
        solver_iterations,
        solver_residual,
    )


def _ah_ratio_stats(history):
    ratios = [
        float(record["ah_ratio"])
        for record in history
        if "ah_ratio" in record and np.isfinite(float(record["ah_ratio"]))
    ]
    if not ratios:
        return np.nan, np.nan, 0
    return float(np.mean(ratios)), float(ratios[-1]), sum(ratio < 0.25 for ratio in ratios)


def _coupled_micro_stats(history):
    records = [
        record for record in history if record.get("coupled_step_attempted", False)
    ]
    if not records:
        return 0, 0
    fallback = sum(1 for record in records if record.get("coupled_fallback_used", False))
    return len(records), fallback


def _run_mode(args, mf, mo_guess, case, mode):
    start = time.perf_counter()
    result = {
        "mode": mode,
        "ok": False,
        "error": "",
        "energy": np.nan,
        "wall_s": np.nan,
        "macro": 0,
        "micro": 0,
        "grad": np.nan,
        "step": np.nan,
        "ah_iter": np.nan,
        "ah_res": np.nan,
        "ah_ratio_avg": np.nan,
        "ah_ratio_last": np.nan,
        "ah_ratio_bad": 0,
        "coupled_attempt": 0,
        "coupled_fallback": 0,
        "ip_attempt": 0,
        "ip_accept": 0,
        "ip_guard_reject": 0,
        "ip_guard_delta": np.nan,
        "ip_ci_dim": 0,
        "ip_q_cycles": 0,
        "ip_relaxed_min_eig": np.nan,
        "ip_fallback": 0,
        "ip_post_grad": np.nan,
        "ip_converged": False,
        "ip_solver_iterations": 0,
        "ip_solver_residual": np.nan,
    }
    try:
        with _solver_output(args.verbose_solvers):
            mc = SecondOrderCASSCF(
                mf,
                ncas=case["ncas"],
                nelecas=case["nelecas"],
                max_cycle=args.max_cycle,
                max_micro_cycle=args.max_micro_cycle,
                conv_tol=args.conv_tol,
                conv_tol_grad=args.conv_tol_grad,
                conv_tol_grad_relaxed=args.conv_tol_grad_relaxed,
                conv_tol_step=args.conv_tol_step,
                max_step=args.max_step,
                level_shift=args.level_shift,
                coupling=mode,
                coupled_fd_step=args.coupled_fd_step,
                coupled_ci_roots=args.coupled_ci_roots,
                coupled_qspace_cycles=args.coupled_qspace_cycles,
                coupled_qspace_max_vectors=args.coupled_qspace_max_vectors,
                coupled_response_vectors=args.coupled_response_vectors,
                coupled_response_fd_step=args.coupled_response_fd_step,
                coupled_accept_min_ratio=args.coupled_accept_min_ratio,
                coupled_fallback=not args.no_coupled_fallback,
                coupled_reuse_subspace=args.coupled_reuse_subspace,
                orbital_parameterization=args.orbital_parameterization,
                internal_preopt_steps=args.internal_preopt_steps,
                internal_preopt_max_step=args.internal_preopt_max_step,
                internal_preopt_hessian=args.internal_preopt_hessian,
                internal_preopt_solver=args.internal_preopt_solver,
                internal_preopt_space=args.internal_preopt_space,
                internal_preopt_guard_cycles=args.internal_preopt_guard_cycles,
                internal_optimization=args.full_internal_optimization,
                internal_max_cycle=args.internal_max_cycle,
                internal_conv_tol_grad=args.internal_conv_tol_grad,
                internal_conv_tol_step=args.internal_conv_tol_step,
                internal_conv_tol_energy=args.internal_conv_tol_energy,
                ah_max_cycle=args.ah_max_cycle,
                ah_max_subspace=args.ah_max_subspace,
                ah_pspace_size=args.ah_pspace_size,
                ah_pspace_max_cycle=args.ah_pspace_max_cycle,
                ah_trust_metric=args.ah_trust_metric,
                ah_adaptive_trust=args.ah_adaptive_trust,
            ).run(mo_coeff=mo_guess, active_orbitals=args.active_orbitals)
    except Exception as exc:  # pragma: no cover - benchmark reporting path
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    else:
        ah_records = [
            record for record in mc.micro_history if "ah_residual_norm" in record
        ]
        ah_iter = (
            float(np.mean([record["ah_iterations"] for record in ah_records]))
            if ah_records
            else np.nan
        )
        ah_res = (
            float(ah_records[-1]["ah_residual_norm"]) if ah_records else np.nan
        )
        ah_ratio_avg, ah_ratio_last, ah_ratio_bad = _ah_ratio_stats(mc.micro_history)
        coupled_attempt, coupled_fallback = _coupled_micro_stats(mc.micro_history)
        (
            ip_attempt,
            ip_accept,
            ip_guard_reject,
            ip_guard_delta,
            ip_ci_dim,
            ip_q_cycles,
            ip_relaxed_min_eig,
            ip_fallback,
            ip_post_grad,
            ip_converged,
            ip_solver_iterations,
            ip_solver_residual,
        ) = _internal_preopt_stats(mc.internal_preopt_history)
        result.update(
            {
                "ok": True,
                "energy": float(np.ravel(mc.e_tot)[0]),
                "macro": len(mc.history),
                "micro": len(mc.micro_history),
                "grad": _last_history_value(mc.history, "gradient_norm"),
                "step": _last_history_value(mc.history, "step_norm"),
                "ah_iter": ah_iter,
                "ah_res": ah_res,
                "ah_ratio_avg": ah_ratio_avg,
                "ah_ratio_last": ah_ratio_last,
                "ah_ratio_bad": ah_ratio_bad,
                "coupled_attempt": coupled_attempt,
                "coupled_fallback": coupled_fallback,
                "ip_attempt": ip_attempt,
                "ip_accept": ip_accept,
                "ip_guard_reject": ip_guard_reject,
                "ip_guard_delta": ip_guard_delta,
                "ip_ci_dim": ip_ci_dim,
                "ip_q_cycles": ip_q_cycles,
                "ip_relaxed_min_eig": ip_relaxed_min_eig,
                "ip_fallback": ip_fallback,
                "ip_post_grad": ip_post_grad,
                "ip_converged": ip_converged,
                "ip_solver_iterations": ip_solver_iterations,
                "ip_solver_residual": ip_solver_residual,
            }
        )
    result["wall_s"] = time.perf_counter() - start
    return result


def _run_pyscf(args, mol, case):
    start = time.perf_counter()
    result = {
        "mode": "pyscf",
        "ok": False,
        "error": "",
        "energy": np.nan,
        "wall_s": np.nan,
        "macro": 0,
        "micro": 0,
        "grad": np.nan,
        "step": np.nan,
        "ah_iter": np.nan,
        "ah_res": np.nan,
        "ah_ratio_avg": np.nan,
        "ah_ratio_last": np.nan,
        "ah_ratio_bad": 0,
        "coupled_attempt": 0,
        "coupled_fallback": 0,
        "ip_attempt": 0,
        "ip_accept": 0,
        "ip_guard_reject": 0,
        "ip_guard_delta": np.nan,
        "ip_ci_dim": 0,
        "ip_q_cycles": 0,
        "ip_relaxed_min_eig": np.nan,
        "ip_fallback": 0,
        "ip_post_grad": np.nan,
        "ip_converged": False,
        "ip_solver_iterations": 0,
        "ip_solver_residual": np.nan,
    }
    try:
        from pyscf import mcscf, scf
    except Exception as exc:  # pragma: no cover - optional dependency path
        result["error"] = "PySCF unavailable: {}: {}".format(type(exc).__name__, exc)
        result["wall_s"] = time.perf_counter() - start
        return result

    history = []

    def callback(envs):
        entry = {
            "cycle": int(envs.get("imacro", len(history) + 1)),
            "energy": float(envs.get("e_tot", np.nan)),
            "gradient_norm": float(envs.get("norm_gorb", np.nan)),
        }
        if history and history[-1]["cycle"] == entry["cycle"]:
            history[-1] = entry
        else:
            history.append(entry)

    try:
        with _solver_output(args.verbose_solvers):
            pmol = mol.topyscf()
            pmol.verbose = 0
            pmf = scf.RHF(pmol)
            pmf.conv_tol = 1.0e-10
            pmf.verbose = 0
            pmf.kernel()
            pmo_guess = _rotated_mo(pmf.mo_coeff, case["rotation"], args.distort)
            pmo_guess = _reorder_mo_for_active_orbitals(
                pmo_guess,
                pmf,
                case,
                args.active_orbitals,
            )
            pmc = mcscf.CASSCF(pmf, case["ncas"], case["nelecas"])
            pmc.conv_tol = args.conv_tol
            pmc.max_cycle_macro = args.max_cycle
            pmc.max_cycle_micro = args.max_micro_cycle
            pmc.verbose = 0
            pmc.kernel(mo_coeff=pmo_guess, callback=callback)
    except Exception as exc:  # pragma: no cover - benchmark reporting path
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    else:
        result.update(
            {
                "ok": bool(getattr(pmc, "converged", True)),
                "energy": float(pmc.e_tot),
                "macro": len(history),
                "grad": _last_history_value(history, "gradient_norm"),
            }
        )
        if not result["ok"]:
            result["error"] = "PySCF CASSCF did not report convergence."
    result["wall_s"] = time.perf_counter() - start
    return result


def _print_table(rows, e_initial):
    e_ref = next((row["energy"] for row in rows if row["ok"] and row["mode"] == "qn"), np.nan)
    if not np.isfinite(e_ref):
        finite = [row["energy"] for row in rows if row["ok"] and np.isfinite(row["energy"])]
        e_ref = min(finite) if finite else np.nan
    e_pyscf = next(
        (row["energy"] for row in rows if row["ok"] and row["mode"] == "pyscf"),
        np.nan,
    )
    show_pyscf_delta = np.isfinite(e_pyscf)

    header = (
        "mode          status       energy / Eh       dE(init)      dE(ref)   "
        "wall/s  macro  micro   ah_it     ah_res   ah_ravg  ah_rlast "
        "ah_bad cpl_a cpl_fb  ip_a ip_ok ip_cv ip_it    ip_res ip_ci ip_q ip_fb   ip_heig  ip_pgrad  ip_grd  ip_dguard   grad        step"
    )
    if show_pyscf_delta:
        header = header.replace("wall/s", "dE(pyscf)   wall/s")
    print(header)
    print("-" * len(header))
    for row in rows:
        status = "ok" if row["ok"] else "failed"
        d_init = row["energy"] - e_initial if row["ok"] else np.nan
        d_ref = row["energy"] - e_ref if row["ok"] and np.isfinite(e_ref) else np.nan
        line = (
            f"{row['mode']:<13s} {status:<8s} "
            f"{row['energy']:16.10f} {d_init:12.3e} {d_ref:11.3e} "
        )
        if show_pyscf_delta:
            d_pyscf = row["energy"] - e_pyscf if row["ok"] else np.nan
            line += f"{d_pyscf:11.3e} "
        line += (
            f"{row['wall_s']:8.3f} {row['macro']:6d} {row['micro']:6d} "
            f"{row['ah_iter']:7.2f} {row['ah_res']:10.3e} "
            f"{row['ah_ratio_avg']:9.2f} {row['ah_ratio_last']:9.2f} "
            f"{row['ah_ratio_bad']:6d} "
            f"{row['coupled_attempt']:5d} {row['coupled_fallback']:6d} "
            f"{row['ip_attempt']:5d} {row['ip_accept']:5d} "
            f"{int(bool(row['ip_converged'])):5d} "
            f"{row['ip_solver_iterations']:5d} {row['ip_solver_residual']:9.2e} "
            f"{row['ip_ci_dim']:5d} {row['ip_q_cycles']:4d} "
            f"{row['ip_fallback']:5d} "
            f"{row['ip_relaxed_min_eig']:9.2e} {row['ip_post_grad']:9.2e} "
            f"{row['ip_guard_reject']:6d} {row['ip_guard_delta']:10.3e} "
            f"{row['grad']:10.3e} {row['step']:10.3e}"
        )
        print(line)
        if row["error"]:
            print(" " * 14 + row["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(PRESETS), default="lih")
    parser.add_argument("--basis", default=None, help="Override the preset basis.")
    parser.add_argument("--modes", type=_parse_modes, default=_parse_modes("qn,partial,relaxed_fd"))
    parser.add_argument(
        "--active-orbitals",
        type=_parse_active_orbitals,
        default=None,
        help="Comma-separated zero-based MO indices to place in the active block.",
    )
    parser.add_argument("--distort", type=float, default=0.12, help="Initial orbital rotation angle.")
    parser.add_argument("--max-cycle", type=int, default=10)
    parser.add_argument("--max-micro-cycle", type=int, default=4)
    parser.add_argument("--conv-tol", type=float, default=1.0e-2)
    parser.add_argument("--conv-tol-grad", type=float, default=1.0e-2)
    parser.add_argument("--conv-tol-grad-relaxed", type=float, default=1.0e-1)
    parser.add_argument("--conv-tol-step", type=float, default=1.0e-4)
    parser.add_argument("--max-step", type=float, default=0.10)
    parser.add_argument("--level-shift", type=float, default=5.0e-4)
    parser.add_argument("--ah-max-cycle", type=int, default=6)
    parser.add_argument("--ah-max-subspace", type=int, default=12)
    parser.add_argument("--ah-pspace-size", type=int, default=12)
    parser.add_argument("--ah-pspace-max-cycle", type=int, default=6)
    parser.add_argument(
        "--ah-trust-metric",
        choices=("component", "norm"),
        default="component",
    )
    parser.add_argument("--ah-adaptive-trust", action="store_true")
    parser.add_argument("--coupled-fd-step", type=float, default=1.0e-4)
    parser.add_argument("--coupled-ci-roots", type=int, default=0)
    parser.add_argument("--coupled-qspace-cycles", type=int, default=2)
    parser.add_argument("--coupled-qspace-max-vectors", type=int, default=None)
    parser.add_argument("--coupled-response-vectors", type=int, default=None)
    parser.add_argument("--coupled-response-fd-step", type=float, default=5.0e-4)
    parser.add_argument("--coupled-accept-min-ratio", type=float, default=0.05)
    parser.add_argument("--no-coupled-fallback", action="store_true")
    parser.add_argument(
        "--coupled-reuse-subspace",
        action="store_true",
        help="Reuse the previous full coupled AH vector as a Davidson seed.",
    )
    parser.add_argument(
        "--orbital-parameterization",
        choices=("exponential", "wmk"),
        default="exponential",
    )
    parser.add_argument("--internal-preopt-steps", type=int, default=0)
    parser.add_argument("--internal-preopt-max-step", type=float, default=None)
    parser.add_argument(
        "--internal-preopt-hessian",
        choices=("diagonal", "analytic", "finite_difference", "coupled", "coupled_fd"),
        default="finite_difference",
    )
    parser.add_argument(
        "--internal-preopt-solver",
        choices=("dense", "davidson"),
        default="dense",
        help="Solver for analytic internal orbital Hessian steps.",
    )
    parser.add_argument(
        "--internal-preopt-space",
        choices=("core_active", "nonredundant"),
        default="core_active",
    )
    parser.add_argument("--internal-preopt-guard-cycles", type=int, default=0)
    parser.add_argument(
        "--full-internal-optimization",
        action="store_true",
        help="Iterate the internal orbital subproblem to convergence before each macro step.",
    )
    parser.add_argument("--internal-max-cycle", type=int, default=None)
    parser.add_argument("--internal-conv-tol-grad", type=float, default=None)
    parser.add_argument("--internal-conv-tol-step", type=float, default=None)
    parser.add_argument("--internal-conv-tol-energy", type=float, default=None)
    parser.add_argument(
        "--verbose-solvers",
        action="store_true",
        help="Show RHF/CASCI/CASSCF solver output instead of only the benchmark table.",
    )
    parser.add_argument(
        "--compare-pyscf",
        action="store_true",
        help="Append a PySCF CASSCF reference row when PySCF is installed.",
    )
    args = parser.parse_args()

    case = PRESETS[args.system]
    with _solver_output(args.verbose_solvers):
        mol = _make_molecule(case, args.basis)
        mf = mol.RHF().run()
        mo_guess = _rotated_mo(mf.mo_coeff, case["rotation"], args.distort)
        mo_guess0 = _reorder_mo_for_active_orbitals(
            mo_guess,
            mf,
            case,
            args.active_orbitals,
        )
        mc0 = CASCI(mf, ncas=case["ncas"], nelecas=case["nelecas"]).run(
            nstates=1,
            mo_coeff=mo_guess0,
            method="direct_ci",
        )
    e_initial = float(np.ravel(mc0.e_tot)[0])

    print("System: {}  basis: {}  CAS({}, {})".format(
        args.system,
        args.basis or case["basis"],
        case["ncas"],
        case["nelecas"],
    ))
    print("Initial CASCI energy: {:.12f} Eh".format(e_initial))
    print("Initial orbital rotation: {} angle={:.6f}".format(case["rotation"], args.distort))
    if args.active_orbitals is not None:
        print("Initial active orbitals: {}".format(args.active_orbitals))
    print()

    rows = [_run_mode(args, mf, mo_guess, case, mode) for mode in args.modes]
    if args.compare_pyscf:
        rows.append(_run_pyscf(args, mol, case))
    _print_table(rows, e_initial)


if __name__ == "__main__":
    main()
