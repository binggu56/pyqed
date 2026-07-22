#!/usr/bin/env python3
"""Benchmark PyQED native CASPT2 and optionally compare with external MRPT codes.

The OpenMolcas and Psi4/Forte legs are intentionally optional: on machines
without ``pymolcas``/``molcas`` or ``psi4`` the script still writes input decks
and records skipped statuses.  The energy comparisons are diagnostic rather
than strict because OpenMolcas runs production internally contracted CASPT2,
Psi4/Forte runs DSRG-MRPT2, and PyQED's current CASPT2 path is still the native
diagonal/strong-contracted starter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time
from typing import Iterable

import numpy as np

from pyqed import au2angstrom
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf.caspt2 import CASPT2


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    atom: str
    unit: str
    basis: str
    ncas: int
    nelecas: int
    charge: int = 0
    spin: int = 0
    inactive: int | None = None
    openmolcas_basis: str | None = None


CASES = {
    "lih_cas22_sto3g": BenchmarkCase(
        name="lih_cas22_sto3g",
        atom="Li 0 0 0; H 0 0 1.6",
        unit="angstrom",
        basis="sto-3g",
        openmolcas_basis="STO-3G",
        ncas=2,
        nelecas=2,
    ),
    "h2o_cas44_sto3g": BenchmarkCase(
        name="h2o_cas44_sto3g",
        atom="O 0 0 0; H 0 0 0.958; H 0 0.926  -0.239",
        unit="angstrom",
        basis="sto-3g",
        openmolcas_basis="STO-3G",
        ncas=4,
        nelecas=4,
    ),
}


CASPT2_TOTAL_PATTERNS = (
    re.compile(
        r"(?:Total\s+CASPT2\s+energy|CASPT2\s+Root\s+\d+\s+Total\s+energy)"
        r"\s*[:=]\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"::\s+CASPT2.*?Total\s+energy\s*[:=]\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
        re.IGNORECASE,
    ),
)
RASSCF_TOTAL_PATTERNS = (
    re.compile(
        r"(?:RASSCF|CASSCF).*?(?:root\s+\d+.*?)?Total\s+energy\s*[:=]\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
        re.IGNORECASE,
    ),
)
PSI4_FORTE_SENTINELS = {
    "casscf_total_energy_hartree": re.compile(
        r"PYQED_BENCHMARK_PSI4_CASSCF_ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
    ),
    "forte_dsrg_mrpt2_total_energy_hartree": re.compile(
        r"PYQED_BENCHMARK_FORTE_DSRG_MRPT2_ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
    ),
}
FORTE_NATIVE_TOTAL_PATTERN = re.compile(
    r"DSRG-MRPT2\s+total\s+energy\s*=\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


def _expand_selection(value: str, available: Iterable[str]) -> list[str]:
    available = list(available)
    selected: list[str] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key == "all" or key == "both":
            selected.extend(available)
        elif key in available:
            selected.append(key)
        else:
            raise ValueError(f"Unknown selection {item!r}; choose from {available}.")
    return list(dict.fromkeys(selected))


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def build_pyqed_reference(case: BenchmarkCase):
    mol = Molecule(
        atom=case.atom,
        unit=case.unit,
        basis=case.basis,
        charge=case.charge,
        spin=case.spin,
    )
    t0 = time.perf_counter()
    mol.build(driver="gbasis")
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    mf = RHF(mol).run()
    t_scf = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc = direct_ci.CASCI(mf, ncas=case.ncas, nelecas=case.nelecas).run(
        nstates=1,
        method="direct_ci",
    )
    t_casci = time.perf_counter() - t0

    return mol, mf, mc, {
        "build_s": t_build,
        "rhf_s": t_scf,
        "casci_s": t_casci,
        "rhf_energy_hartree": float(mf.e_tot),
        "casci_energy_hartree": float(np.asarray(mc.e_tot).reshape(-1)[0]),
        "nmo": int(np.asarray(mc.mo_coeff).shape[1]),
        "ncore": int(mc.ncore),
        "ncas": int(mc.ncas),
        "nelecas": int(case.nelecas),
    }


def run_pyqed_caspt2(
    mc,
    zeroth_order: str,
    contraction: str,
    contracted_matrix: str,
    repeat: int,
    max_external_determinants: int | None,
):
    timings: list[float] = []
    last_pt: CASPT2 | None = None
    last_energy = 0.0
    for _ in range(repeat):
        pt = CASPT2(
            mc,
            zeroth_order=zeroth_order,
            contraction=contraction,
            contracted_matrix=contracted_matrix,
            max_external_determinants=max_external_determinants,
        )
        t0 = time.perf_counter()
        last_energy = pt.run()
        timings.append(time.perf_counter() - t0)
        last_pt = pt

    assert last_pt is not None
    components = {
        label: {
            "count": component.count,
            "energy_hartree": component.energy,
            "norm": component.norm,
            "denominator_hartree": component.denominator,
            "amplitude": component.amplitude,
        }
        for label, component in last_pt.components.items()
    }
    return {
        "zeroth_order": zeroth_order,
        "contraction": contraction,
        "contracted_matrix_request": contracted_matrix,
        "e_corr_hartree": float(last_energy),
        "e_tot_hartree": float(last_pt.e_tot),
        "timings_s": timings,
        "median_s": float(statistics.median(timings)),
        "min_s": float(min(timings)),
        "external_determinants": len(last_pt.external_determinants),
        "external_space_backend": last_pt.external_space_backend,
        "external_kernel_backend": last_pt.external_kernel_backend,
        "contraction_backend": last_pt.contraction_backend,
        "contracted_matrix_kind": last_pt.contracted_matrix_kind,
        "contracted_matrix_backend": last_pt.contracted_matrix_backend,
        "contracted_solver_backend": last_pt.contracted_solver_backend,
        "components": components,
    }


def write_openmolcas_input(
    case: BenchmarkCase,
    mol,
    work_dir: Path,
):
    case_dir = work_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = case_dir / f"{case.name}.xyz"
    input_path = case_dir / f"{case.name}.input"

    coords_angstrom = np.asarray(mol.atom_coords(), dtype=float) * au2angstrom
    with xyz_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{mol.natom}\n")
        handle.write(f"{case.name}\n")
        for symbol, coord in zip(mol.atom_symbols(), coords_angstrom):
            handle.write(
                f"{symbol:2s} {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}\n"
            )

    inactive = case.inactive
    if inactive is None:
        inactive = int((mol.nelec - case.nelecas) // 2)
    multiplicity = int(case.spin) + 1
    basis = case.openmolcas_basis or case.basis

    input_path.write_text(
        "\n".join(
            [
                "&GATEWAY",
                f"Coord = {xyz_path.name}",
                f"Basis = {basis}",
                "Group = C1",
                "",
                "&SEWARD",
                "",
                "&SCF",
                f"Charge = {case.charge}",
                f"Spin = {multiplicity}",
                "",
                "&RASSCF",
                f"Spin = {multiplicity}",
                f"Nactel = {case.nelecas} 0 0",
                f"Inactive = {inactive}",
                f"Ras2 = {case.ncas}",
                "CIRoot = 1 1 1",
                "",
                "&CASPT2",
                "IPEA = 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return input_path


def parse_openmolcas_output(text: str):
    result: dict[str, float | None] = {
        "caspt2_total_energy_hartree": None,
        "rasscf_total_energy_hartree": None,
    }
    for pattern in CASPT2_TOTAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result["caspt2_total_energy_hartree"] = float(matches[-1])
            break
    for pattern in RASSCF_TOTAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result["rasscf_total_energy_hartree"] = float(matches[-1])
            break
    caspt2 = result["caspt2_total_energy_hartree"]
    rasscf = result["rasscf_total_energy_hartree"]
    if caspt2 is not None and rasscf is not None:
        result["caspt2_correction_hartree"] = float(caspt2 - rasscf)
    return result


def find_openmolcas_command(requested: str | None):
    if requested:
        return requested
    return shutil.which("pymolcas") or shutil.which("molcas")


def run_openmolcas_case(
    input_path: Path,
    command: str | None,
    mode: str,
    timeout: float,
):
    resolved = find_openmolcas_command(command)
    if resolved is None:
        status = "skipped" if mode == "auto" else "missing"
        return {
            "status": status,
            "reason": "pymolcas/molcas executable was not found on PATH",
            "input_path": str(input_path),
        }
    if mode == "never":
        return {
            "status": "not_run",
            "reason": "OpenMolcas execution disabled",
            "command": resolved,
            "input_path": str(input_path),
        }

    argv = shlex.split(resolved) + [input_path.name]
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=input_path.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    output = proc.stdout + "\n" + proc.stderr
    log_path = input_path.with_suffix(".openmolcas.log")
    log_path.write_text(output, encoding="utf-8")
    parsed = parse_openmolcas_output(output)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "input_path": str(input_path),
        "log_path": str(log_path),
        **parsed,
    }


def write_psi4_forte_input(
    case: BenchmarkCase,
    mol,
    work_dir: Path,
):
    case_dir = work_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / f"{case.name}_psi4_forte.dat"

    coords_angstrom = np.asarray(mol.atom_coords(), dtype=float) * au2angstrom
    multiplicity = int(case.spin) + 1
    inactive = case.inactive
    if inactive is None:
        inactive = int((mol.nelec - case.nelecas) // 2)

    geometry_lines = [
        f"  {symbol:2s} {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}"
        for symbol, coord in zip(mol.atom_symbols(), coords_angstrom)
    ]
    geometry = "\n".join(geometry_lines)
    input_path.write_text(
        f"""import forte

memory 500 mb

molecule mol {{
  {case.charge} {multiplicity}
  units angstrom
  symmetry c1
{geometry}
}}

set globals {{
  basis {case.basis}
  reference rhf
  scf_type pk
  restricted_docc [{inactive}]
  active [{case.ncas}]
}}

set forte {{
  active_space_solver fci
  correlation_solver dsrg-mrpt2
  dsrg_s 0.5
  frozen_docc [0]
  restricted_docc [{inactive}]
  active [{case.ncas}]
}}

e_casscf, wfn = energy('casscf', return_wfn=True)
e_forte = energy('forte', ref_wfn=wfn)
print('PYQED_BENCHMARK_PSI4_CASSCF_ENERGY %.16f' % e_casscf)
print('PYQED_BENCHMARK_FORTE_DSRG_MRPT2_ENERGY %.16f' % e_forte)
""",
        encoding="utf-8",
    )
    return input_path


def parse_psi4_forte_output(text: str):
    result: dict[str, float | None] = {
        "casscf_total_energy_hartree": None,
        "forte_dsrg_mrpt2_total_energy_hartree": None,
    }
    for key, pattern in PSI4_FORTE_SENTINELS.items():
        matches = pattern.findall(text)
        if matches:
            result[key] = float(matches[-1])
    if result["forte_dsrg_mrpt2_total_energy_hartree"] is None:
        matches = FORTE_NATIVE_TOTAL_PATTERN.findall(text)
        if matches:
            result["forte_dsrg_mrpt2_total_energy_hartree"] = float(matches[-1])
    casscf = result["casscf_total_energy_hartree"]
    forte = result["forte_dsrg_mrpt2_total_energy_hartree"]
    if casscf is not None and forte is not None:
        result["forte_dsrg_mrpt2_correction_hartree"] = float(forte - casscf)
    return result


def find_psi4_command(requested: str | None):
    if requested:
        return requested
    return shutil.which("psi4")


def run_psi4_forte_case(
    input_path: Path,
    command: str | None,
    mode: str,
    timeout: float,
):
    resolved = find_psi4_command(command)
    if resolved is None:
        status = "skipped" if mode == "auto" else "missing"
        return {
            "status": status,
            "reason": "psi4 executable was not found on PATH",
            "input_path": str(input_path),
        }
    if mode == "never":
        return {
            "status": "not_run",
            "reason": "Psi4/Forte execution disabled",
            "command": resolved,
            "input_path": str(input_path),
        }

    output_path = input_path.with_suffix(".out")
    argv = shlex.split(resolved) + [input_path.name, output_path.name]
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=input_path.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    output = proc.stdout + "\n" + proc.stderr
    if output_path.exists():
        output += "\n" + output_path.read_text(encoding="utf-8", errors="replace")
    log_path = input_path.with_suffix(".psi4_forte.log")
    log_path.write_text(output, encoding="utf-8")
    parsed = parse_psi4_forte_output(output)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "log_path": str(log_path),
        **parsed,
    }


def benchmark_case(args, case: BenchmarkCase):
    mol, _mf, mc, reference = build_pyqed_reference(case)
    pyqed_runs = []
    for zeroth_order in _expand_selection(args.zeroth_order, ("fock", "en")):
        for contraction in _expand_selection(args.contraction, ("uncontracted", "strong")):
            pyqed_runs.append(
                run_pyqed_caspt2(
                    mc,
                    zeroth_order=zeroth_order,
                    contraction=contraction,
                    contracted_matrix=args.contracted_matrix,
                    repeat=args.repeat,
                    max_external_determinants=args.max_external_determinants,
                )
            )

    work_dir = Path(args.work_dir)
    input_path = write_openmolcas_input(case, mol, work_dir)
    openmolcas = run_openmolcas_case(
        input_path,
        command=args.openmolcas_command,
        mode=args.run_openmolcas,
        timeout=args.openmolcas_timeout,
    )
    psi4_forte_input = write_psi4_forte_input(case, mol, work_dir)
    psi4_forte = run_psi4_forte_case(
        psi4_forte_input,
        command=args.psi4_command,
        mode=args.run_psi4,
        timeout=args.psi4_timeout,
    )

    if openmolcas.get("caspt2_total_energy_hartree") is not None:
        om_total = float(openmolcas["caspt2_total_energy_hartree"])
        for run in pyqed_runs:
            run["delta_vs_openmolcas_total_hartree"] = run["e_tot_hartree"] - om_total
    if psi4_forte.get("forte_dsrg_mrpt2_total_energy_hartree") is not None:
        forte_total = float(psi4_forte["forte_dsrg_mrpt2_total_energy_hartree"])
        for run in pyqed_runs:
            run["delta_vs_forte_dsrg_mrpt2_total_hartree"] = run["e_tot_hartree"] - forte_total

    return {
        "case": asdict(case),
        "pyqed_reference": reference,
        "pyqed_caspt2": pyqed_runs,
        "openmolcas": openmolcas,
        "psi4_forte": psi4_forte,
        "notes": [
            "OpenMolcas CASPT2 and PyQED native CASPT2 are not yet the same zeroth-order/contraction model.",
            "Psi4/Forte DSRG-MRPT2 is an adjacent multireference PT2 benchmark, not standard CASPT2.",
            "Use energy deltas here as diagnostic trend data until PyQED implements the full production internally contracted CASPT2 path.",
        ],
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="lih_cas22_sto3g",
        help=f"Comma-separated case names or 'all'. Available: {', '.join(CASES)}",
    )
    parser.add_argument("--zeroth-order", default="fock,en", help="'fock', 'en', or comma-separated/both.")
    parser.add_argument("--contraction", default="strong", help="'uncontracted', 'strong', or comma-separated/both.")
    parser.add_argument("--contracted-matrix", default="auto", choices=CASPT2.supported_contracted_matrices)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-external-determinants", type=int, default=None)
    parser.add_argument("--work-dir", default="/private/tmp/pyqed-openmolcas-caspt2")
    parser.add_argument("--out", default="/private/tmp/pyqed_caspt2_openmolcas_benchmark.json")
    parser.add_argument("--openmolcas-command", default=None)
    parser.add_argument("--run-openmolcas", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--openmolcas-timeout", type=float, default=600.0)
    parser.add_argument("--psi4-command", default=None)
    parser.add_argument("--run-psi4", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--psi4-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    if args.repeat < 1:
        raise ValueError("--repeat must be positive.")

    selected_cases = _expand_selection(args.cases, CASES)
    results = {
        "benchmark": "pyqed_native_caspt2_external_mrpt_benchmark",
        "created_by": "benchmarks/caspt2_openmolcas.py",
        "cases": [benchmark_case(args, CASES[name]) for name in selected_cases],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonify(results), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonify(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
