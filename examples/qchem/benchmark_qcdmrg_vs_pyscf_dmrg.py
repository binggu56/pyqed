#!/usr/bin/env python3
"""
Benchmark PyQED DMRG against a PySCF DMRG reference backend.

This script always runs the local PyQED DMRG benchmark. If `pyscf.dmrgscf`
is available and configured with an external DMRG solver (for example block2),
it also runs a PySCF CASCI+DMRG reference at matched bond dimensions.

If no PySCF DMRG backend is available, the script falls back to reporting a
PySCF exact CASCI/FCI reference when the active space is small enough.
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule, DMRG
from pyqed.qchem.hf import RHF

from pyscf import gto, mcscf, scf

try:
    from pyscf import dmrgscf as pyscf_dmrgscf
except Exception:
    pyscf_dmrgscf = None


PRESETS = {
    "h4_sto3g": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        "unit": "bohr",
        "basis": "sto-3g",
        "ncas": 4,
        "nelecas": 4,
    },
    "h6_sto3g": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8; H 0 0 6.4; H 0 0 8.0",
        "unit": "bohr",
        "basis": "sto-3g",
        "ncas": 6,
        "nelecas": 6,
    },
}


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _to_scalar_energy(value) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0])
    return float(value)


def run_pyqed_qcdmrg(case, D: int, nsweeps: int, d_schedule: list[int] | None, use_cholesky: bool):
    mol = Molecule(atom=case["atom"], unit=case["unit"], basis=case["basis"])
    t0 = time.perf_counter()
    mol.build(driver="gbasis")
    mf = RHF(mol).run(**({"cholesky_jk": True, "cholesky_tol": 1e-8} if use_cholesky else {}))
    t1 = time.perf_counter()

    dmrg = DMRG(mf, ncas=case["ncas"], nelecas=case["nelecas"], D=D, init_guess="cid")
    dmrg.build()
    t2 = time.perf_counter()
    dmrg.run(
        symmetry_list=["charge", "sz"],
        D_schedule=d_schedule,
        nsweeps=nsweeps,
        compute_s2=False,
    )
    t3 = time.perf_counter()
    return {
        "rhf_energy": float(mf.e_tot),
        "energy": _to_scalar_energy(dmrg.e_tot),
        "setup_s": t1 - t0,
        "build_s": t2 - t1,
        "solve_s": t3 - t2,
        "total_s": t3 - t0,
        "info": dict(dmrg._active_integral_build_info or {}),
    }


def run_pyscf_casci(case):
    mol = gto.M(
        atom=case["atom"],
        basis=case["basis"],
        unit="Bohr" if case["unit"].lower() in ("bohr", "b") else "Angstrom",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    mc = mcscf.CASCI(mf, case["ncas"], case["nelecas"])
    e, *_ = mc.kernel()
    return float(e)


def run_pyscf_dmrg(case, D: int):
    if pyscf_dmrgscf is None:
        raise RuntimeError("pyscf.dmrgscf is not available in this environment.")

    mol = gto.M(
        atom=case["atom"],
        basis=case["basis"],
        unit="Bohr" if case["unit"].lower() in ("bohr", "b") else "Angstrom",
        spin=0,
        verbose=0,
    )
    t0 = time.perf_counter()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    t1 = time.perf_counter()

    mc = mcscf.CASCI(mf, case["ncas"], case["nelecas"])
    solver = pyscf_dmrgscf.DMRGCI(mol, maxM=D, tol=1e-10)

    # The legacy pyscf.dmrgscf wrapper needs an explicit block2-style setup.
    # The defaults are Block-oriented and can produce incorrect results with
    # block2main on larger basis sets unless the full FCI space and cutoffs
    # are configured coherently.
    run_dir = tempfile.mkdtemp(prefix="pyscf_block2_")
    start_m = min(16, D)
    solver.runtimeDir = run_dir
    solver.scratchDirectory = run_dir
    solver.threads = 1
    solver.scheduleSweeps = [0, 4, 8, 12, 16]
    solver.scheduleMaxMs = [start_m, D, D, D, D]
    solver.scheduleTols = [1e-6, 1e-8, 1e-10, 1e-10, 1e-10]
    solver.scheduleNoises = [1e-4, 1e-5, 1e-6, 1e-7, 0.0]
    solver.maxIter = 24
    solver.twodot_to_onedot = 20
    solver.block_extra_keyword = [
        "full_fci_space",
        "cutoff 0",
        "fp_cps_cutoff 0",
    ]
    mc.fcisolver = solver
    t2 = time.perf_counter()
    e = mc.kernel()[0]
    t3 = time.perf_counter()
    return {
        "rhf_energy": float(mf.e_tot),
        "energy": float(e),
        "setup_s": t1 - t0,
        "build_s": t2 - t1,
        "solve_s": t3 - t2,
        "total_s": t3 - t0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(PRESETS), default="h4_sto3g")
    parser.add_argument("--D-values", default="8,16,24,32", help="Comma-separated bond dimensions.")
    parser.add_argument("--nsweeps", type=int, default=8, help="Total sweep budget for DMRG.")
    parser.add_argument(
        "--D-schedule",
        default="",
        help="Comma-separated DMRG D schedule. Default derives from each D as [8,16,...,D].",
    )
    parser.add_argument(
        "--use-cholesky",
        action="store_true",
        help="Run the PyQED DMRG side with cholesky-enabled RHF.",
    )
    parser.add_argument(
        "--skip-pyscf-ref",
        action="store_true",
        help="Skip the PySCF DMRG reference even if available.",
    )
    args = parser.parse_args()

    case = PRESETS[args.system]
    d_values = _parse_int_list(args.D_values)
    explicit_schedule = _parse_int_list(args.D_schedule) if args.D_schedule else None

    print(f"System      : {args.system}")
    print(f"Atom        : {case['atom']}")
    print(f"Basis       : {case['basis']}")
    print(f"Active space: CAS({case['ncas']},{case['nelecas']})")
    print(f"DMRG side   : {'cholesky' if args.use_cholesky else 'dense'} RHF")
    print(f"PySCF DMRG  : {'available' if pyscf_dmrgscf is not None and not args.skip_pyscf_ref else 'unavailable/skipped'}")
    print()

    exact_casci = None
    if case["ncas"] <= 8:
        exact_casci = run_pyscf_casci(case)
        print(f"PySCF CASCI : {exact_casci:.12f} Eh")
        print()

    header = (
        "D/maxM  "
        "qcdmrg_energy      qcdmrg_err       qcdmrg_build  qcdmrg_solve  "
        "pyscf_energy       pyscf_err        pyscf_total"
    )
    print(header)
    print("-" * len(header))

    for D in d_values:
        q_schedule = explicit_schedule
        if q_schedule is None:
            q_schedule = [x for x in (8, 16, 24, D) if x <= D]
            if not q_schedule or q_schedule[-1] != D:
                q_schedule.append(D)
            q_schedule = list(dict.fromkeys(q_schedule))

        qres = run_pyqed_qcdmrg(case, D=D, nsweeps=args.nsweeps, d_schedule=q_schedule, use_cholesky=args.use_cholesky)
        qerr = None if exact_casci is None else qres["energy"] - exact_casci

        pres = None
        perr = None
        if pyscf_dmrgscf is not None and not args.skip_pyscf_ref:
            try:
                pres = run_pyscf_dmrg(case, D=D)
                perr = None if exact_casci is None else pres["energy"] - exact_casci
            except Exception as exc:
                print(f"# PySCF DMRG failed at D={D}: {type(exc).__name__}: {exc}")

        def fmt_e(x):
            return f"{x: .12f}" if x is not None else " " * 15 + "n/a"

        def fmt_sci(x):
            return f"{x: .3e}" if x is not None else " " * 12 + "n/a"

        def fmt_t(x):
            return f"{x:8.3f}s" if x is not None else " " * 8 + "n/a"

        print(
            f"{D:6d}  "
            f"{fmt_e(qres['energy'])}  "
            f"{fmt_sci(qerr)}  "
            f"{fmt_t(qres['build_s'])}  "
            f"{fmt_t(qres['solve_s'])}  "
            f"{fmt_e(None if pres is None else pres['energy'])}  "
            f"{fmt_sci(perr)}  "
            f"{fmt_t(None if pres is None else pres['total_s'])}"
        )

    print()
    print("# Notes")
    print("# - qcdmrg_build = Molecule+RHF setup excluded; it is DMRG.build() only.")
    print("# - qcdmrg_solve = DMRG.run() only.")
    print("# - PySCF DMRG here uses pyscf.dmrgscf if available. In practice that may be backed by block2/Block depending on local configuration.")
    print("# - If pyscf.dmrgscf is unavailable, only the PyQED side and exact CASCI reference are reported.")


if __name__ == "__main__":
    main()
