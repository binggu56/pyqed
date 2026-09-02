#!/usr/bin/env python3
"""Benchmark native PyQED CASCI against PySCF for pyrazine/6-31G/CAS(8,8)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, mcscf, scf

from pyqed import Molecule
from pyqed.qchem.mcscf.casci import CASCI


GEOMETRY_BOHR = [
    ["N", 0.0, 0.0000046126, 2.9751681209],
    ["C", 0.0, 2.0213606485, 1.3447521663],
    ["C", 0.0, 2.0213594563, -1.3447637764],
    ["N", 0.0, -0.0000049244, -2.9751696399],
    ["C", 0.0, -2.0213693403, -1.3447570196],
    ["C", 0.0, -2.0213627060, 1.3447652675],
    ["H", 0.0, 3.8979353927, 2.1970440670],
    ["H", 0.0, 3.8979280273, -2.1970658170],
    ["H", 0.0, -3.8979425319, -2.1970514056],
    ["H", 0.0, -3.8979294535, 2.1970704549],
]


def _array(value):
    return np.asarray(value, dtype=float).reshape(-1)


def _time_repeated(function, repeats):
    function()  # Warm imports, JIT kernels, and library workspaces.
    values = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    return result, values


def _plot(report, output):
    pyqed_energies = np.asarray(report["pyqed_cholesky"]["energies_hartree"])
    pyqed_dense_energies = np.asarray(report["pyqed_dense"]["energies_hartree"])
    pyscf_energies = np.asarray(report["pyscf"]["energies_hartree"])
    roots = np.arange(1, pyqed_energies.size + 1)
    reference = pyscf_energies[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    axes[0].plot(roots, (pyscf_energies - reference) * 27.211386, "o-", label="PySCF")
    axes[0].plot(roots, (pyqed_dense_energies - reference) * 27.211386, "^--", label="PyQED dense")
    axes[0].plot(roots, (pyqed_energies - reference) * 27.211386, "s:", label="PyQED Cholesky")
    axes[0].set(
        xlabel="Singlet root",
        ylabel="Energy relative to PySCF S0 / eV",
        title="CASCI energies",
        xticks=roots,
    )
    axes[0].legend(frameon=False)

    timings = [
        report["pyscf"]["median_seconds"],
        report["pyqed_dense"]["median_seconds"],
        report["pyqed_cholesky"]["median_seconds"],
    ]
    bars = axes[1].bar(["PySCF", "PyQED\ndense", "PyQED\nCholesky"], timings, color=["#999999", "#4C78A8", "#F58518"])
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Median warmed CASCI time / s", title="CASCI wall time")
    axes[1].bar_label(bars, labels=[f"{value:.3f} s" for value in timings], padding=3)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--nroots", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/pyrazine_cas88_pyscf"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    qmol = Molecule(atom=GEOMETRY_BOHR, unit="bohr", basis=args.basis)
    qmol.build(eri="dense", options={"parallel": False})
    pyqed_integral_seconds = time.perf_counter() - started
    started = time.perf_counter()
    qmf = qmol.RHF().run(tol=1.0e-10, max_cycle=100, verbose=0)
    pyqed_rhf_seconds = time.perf_counter() - started

    def run_pyqed():
        solver = CASCI(qmf, ncas=8, nelecas=8, multiplicity=1, verbose=0)
        solver.tol = 1.0e-10
        solver.direct_ci_residual_tol = 1.0e-8
        solver.run(nstates=args.nroots, method="direct_ci", use_cholesky=False)
        return solver

    qsolver, pyqed_times = _time_repeated(run_pyqed, args.repeats)

    started = time.perf_counter()
    qmol_cd = Molecule(atom=GEOMETRY_BOHR, unit="bohr", basis=args.basis)
    qmol_cd.build(
        eri="cd",
        options={"parallel": False, "low_rank_tol": 1.0e-10, "eri_screen_tol": 0.0},
    )
    pyqed_cd_integral_seconds = time.perf_counter() - started
    started = time.perf_counter()
    qmf_cd = qmol_cd.RHF().run(tol=1.0e-10, max_cycle=100, verbose=0)
    pyqed_cd_rhf_seconds = time.perf_counter() - started

    def run_pyqed_cholesky():
        solver = CASCI(qmf_cd, ncas=8, nelecas=8, multiplicity=1, verbose=0)
        solver.tol = 1.0e-10
        solver.direct_ci_residual_tol = 1.0e-8
        solver.run(nstates=args.nroots, method="direct_ci", use_cholesky=True)
        return solver

    qsolver_cd, pyqed_cd_times = _time_repeated(run_pyqed_cholesky, args.repeats)

    started = time.perf_counter()
    pmol = gto.M(atom=GEOMETRY_BOHR, unit="Bohr", basis=args.basis, spin=0, verbose=0)
    pyscf_integral_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pmf = scf.RHF(pmol)
    pmf.conv_tol = 1.0e-10
    pmf.kernel()
    pyscf_rhf_seconds = time.perf_counter() - started

    def run_pyscf():
        solver = mcscf.CASCI(pmf, 8, 8)
        solver.verbose = 0
        solver.fcisolver = fci.direct_spin0.FCI(pmol)
        solver.fcisolver.nroots = args.nroots
        solver.fcisolver.conv_tol = 1.0e-10
        solver.kernel(mo_coeff=pmf.mo_coeff)
        return solver

    psolver, pyscf_times = _time_repeated(run_pyscf, args.repeats)
    pyqed_energies = _array(qsolver.e_tot)
    pyqed_cd_energies = _array(qsolver_cd.e_tot)
    pyscf_energies = _array(psolver.e_tot)
    errors = pyqed_energies - pyscf_energies
    cd_errors = pyqed_cd_energies - pyscf_energies
    pyscf_correlation = pyscf_energies - pmf.e_tot
    dense_correlation_errors = pyqed_energies - qmf.e_tot - pyscf_correlation
    cd_correlation_errors = pyqed_cd_energies - qmf_cd.e_tot - pyscf_correlation

    report = {
        "system": f"pyrazine/{args.basis}/CAS(8,8)",
        "nroots": args.nroots,
        "repeats": args.repeats,
        "comparison": "independent canonical RHF references; frozen-core singlet CASCI",
        "pyqed_dense": {
            "integral_seconds": pyqed_integral_seconds,
            "rhf_seconds": pyqed_rhf_seconds,
            "rhf_energy_hartree": float(qmf.e_tot),
            "times_seconds": pyqed_times,
            "median_seconds": float(np.median(pyqed_times)),
            "energies_hartree": pyqed_energies.tolist(),
            "solver_backend": qsolver.solver_backend,
            "native_diagnostics": qsolver.direct_ci_native_diagnostics,
        },
        "pyqed_cholesky": {
            "integral_seconds": pyqed_cd_integral_seconds,
            "rhf_seconds": pyqed_cd_rhf_seconds,
            "rhf_energy_hartree": float(qmf_cd.e_tot),
            "times_seconds": pyqed_cd_times,
            "median_seconds": float(np.median(pyqed_cd_times)),
            "energies_hartree": pyqed_cd_energies.tolist(),
            "solver_backend": qsolver_cd.solver_backend,
            "native_diagnostics": qsolver_cd.direct_ci_native_diagnostics,
        },
        "pyscf": {
            "integral_seconds": pyscf_integral_seconds,
            "rhf_seconds": pyscf_rhf_seconds,
            "rhf_energy_hartree": float(pmf.e_tot),
            "times_seconds": pyscf_times,
            "median_seconds": float(np.median(pyscf_times)),
            "energies_hartree": pyscf_energies.tolist(),
            "converged": np.asarray(psolver.fcisolver.converged).tolist(),
        },
        "energy_errors_hartree": errors.tolist(),
        "max_abs_energy_error_hartree": float(np.max(np.abs(errors))),
        "cholesky_energy_errors_hartree": cd_errors.tolist(),
        "cholesky_max_abs_energy_error_hartree": float(np.max(np.abs(cd_errors))),
        "dense_correlation_energy_errors_hartree": dense_correlation_errors.tolist(),
        "dense_max_abs_correlation_energy_error_hartree": float(np.max(np.abs(dense_correlation_errors))),
        "cholesky_correlation_energy_errors_hartree": cd_correlation_errors.tolist(),
        "cholesky_max_abs_correlation_energy_error_hartree": float(np.max(np.abs(cd_correlation_errors))),
        "rhf_energy_error_hartree": float(qmf.e_tot - pmf.e_tot),
        "cholesky_rhf_energy_error_hartree": float(qmf_cd.e_tot - pmf.e_tot),
        "pyscf_over_pyqed_dense": float(np.median(pyscf_times) / np.median(pyqed_times)),
        "pyscf_over_pyqed_cholesky": float(np.median(pyscf_times) / np.median(pyqed_cd_times)),
    }
    json_path = args.output_dir / "pyrazine_cas88_pyscf.json"
    figure_path = args.output_dir / "pyrazine_cas88_pyscf.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot(report, figure_path)
    print(json.dumps({"report": str(json_path), "figure": str(figure_path), **report}, indent=2))


if __name__ == "__main__":
    main()
