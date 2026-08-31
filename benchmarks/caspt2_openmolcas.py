#!/usr/bin/env python3
"""Benchmark PyQED native CASPT2 and optionally compare with external MRPT codes.

The OpenMolcas and Psi4/Forte legs are intentionally optional: on machines
without ``pymolcas``/``molcas`` or ``psi4`` the script still writes input decks
and records skipped statuses. OpenMolcas and PyQED both use an IPEA shift of
zero here. Psi4/Forte DSRG-MRPT2 remains an adjacent, non-identical comparison.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import threading
import time
from typing import Iterable

import numpy as np

from pyqed import au2angstrom
from pyqed.qchem import CASCI, CASSCF, Molecule
from pyqed.qchem.hf import RHF
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
    reference: str = "casscf"
    basis_variant: str | None = None


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
    "lih_cas22_ccpvdz": BenchmarkCase(
        name="lih_cas22_ccpvdz",
        atom="Li 0 0 0; H 0 0 1.6",
        unit="angstrom",
        basis="cc-pvdz",
        openmolcas_basis="cc-pVDZ",
        ncas=2,
        nelecas=2,
        reference="casci",
        basis_variant="openmolcas_prascher_cc_pvdz",
    ),
}


CASPT2_TOTAL_PATTERNS = (
    re.compile(
        r"FINAL\s+CASPT2\s+RESULT:.*?Total\s+energy\s*:\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
        re.IGNORECASE | re.DOTALL,
    ),
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
CASPT2_REFERENCE_PATTERN = re.compile(
    r"FINAL\s+CASPT2\s+RESULT:.*?Reference\s+energy\s*:\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
CASPT2_E2_PATTERN = re.compile(
    r"FINAL\s+CASPT2\s+RESULT:.*?E2\s*\(Variational\)\s*:\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
CASPT2_RANK_PATTERN = re.compile(
    r"Total\s+nr\s+of\s+CASPT2\s+parameters:.*?After\s+reduction\s*:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
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


def _pyqed_basis(case: BenchmarkCase):
    if case.basis_variant is None:
        return case.basis
    if case.basis_variant != "openmolcas_prascher_cc_pvdz":
        raise ValueError(f"Unsupported basis variant {case.basis_variant!r}.")

    from pyscf import gto

    lithium = deepcopy(gto.basis.load("cc-pvdz", "Li"))
    d_shells = [shell for shell in lithium if shell[0] == 2]
    if len(d_shells) != 1 or len(d_shells[0]) != 2:
        raise RuntimeError("Unexpected PySCF Li cc-pVDZ representation.")
    d_shells[0][1][0] = 0.1144
    return {"Li": lithium, "H": "cc-pvdz"}


def build_pyqed_reference(case: BenchmarkCase):
    basis = _pyqed_basis(case)
    mol = Molecule(
        atom=case.atom,
        unit=case.unit,
        basis=basis,
        charge=case.charge,
        spin=case.spin,
    )
    t0 = time.perf_counter()
    mol.build()
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    mf = RHF(mol).run()
    t_scf = time.perf_counter() - t0

    t0 = time.perf_counter()
    if case.reference == "casscf":
        mc = CASSCF(
            mf,
            ncas=case.ncas,
            nelecas=case.nelecas,
            max_cycle=50,
            verbose=0,
        ).run(nstates=1)
    elif case.reference == "casci":
        mc = CASCI(mf, ncas=case.ncas, nelecas=case.nelecas).run(
            nstates=1, method="direct_ci"
        )
    else:
        raise ValueError(f"Unsupported reference method {case.reference!r}.")
    t_reference = time.perf_counter() - t0

    return mol, mf, mc, {
        "build_s": t_build,
        "rhf_s": t_scf,
        "reference_method": case.reference,
        "reference_s": t_reference,
        "rhf_energy_hartree": float(mf.e_tot),
        "reference_energy_hartree": float(np.asarray(mc.e_tot).reshape(-1)[0]),
        "reference_converged": bool(getattr(mc, "converged", True)),
        "basis_variant": case.basis_variant,
        "integral_engine": "native",
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
    ic_basis_backend: str,
    linear_solver: str,
    max_memory_mb: float | None,
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
            ic_basis_backend=ic_basis_backend,
            linear_solver=linear_solver,
            max_memory_mb=max_memory_mb,
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
        "e_corr_nonvariational_hartree": float(last_pt.e_corr_nonvariational),
        "shift_correction_hartree": float(last_pt.shift_correction),
        "e_tot_hartree": float(last_pt.e_tot),
        "timings_s": timings,
        "median_s": float(statistics.median(timings)),
        "min_s": float(min(timings)),
        "external_determinants": len(last_pt.external_determinants),
        "external_space_backend": last_pt.external_space_backend,
        "external_kernel_backend": last_pt.external_kernel_backend,
        "external_operator_nnz": last_pt.external_operator_nnz,
        "external_operator_backend": last_pt.external_operator_backend,
        "contraction_backend": last_pt.contraction_backend,
        "ic_basis_backend": last_pt.ic_basis_backend,
        "ic_metric_backend": last_pt.ic_metric_backend,
        "work_estimate": last_pt.work_estimate,
        "linear_solver": last_pt.linear_solver,
        "solver_iterations": last_pt.solver_iterations,
        "solver_history": last_pt.solver_history,
        "success": last_pt.success,
        "message": last_pt.message,
        "phase_timings_s": last_pt.timings,
        "contracted_matrix_kind": last_pt.contracted_matrix_kind,
        "contracted_matrix_backend": last_pt.contracted_matrix_backend,
        "contracted_solver_backend": last_pt.contracted_solver_backend,
        "contracted_basis_size": last_pt.contracted_basis_size,
        "contracted_basis_rank": last_pt.contracted_basis_rank,
        "contracted_residual_norm": last_pt.contracted_residual_norm,
        "contracted_relative_residual_norm": last_pt.contracted_relative_residual_norm,
        "reference_weight": last_pt.reference_weight,
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
    input_path = case_dir / f"{case.name}_{time.time_ns()}.input"

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
                *(["CIONLY"] if case.reference == "casci" else []),
                "",
                "&CASPT2",
                "Frozen = 0",
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
    reference_match = CASPT2_REFERENCE_PATTERN.search(text)
    if reference_match:
        result["rasscf_total_energy_hartree"] = float(reference_match.group(1))
    caspt2 = result["caspt2_total_energy_hartree"]
    rasscf = result["rasscf_total_energy_hartree"]
    e2_match = CASPT2_E2_PATTERN.search(text)
    if e2_match:
        result["caspt2_correction_hartree"] = float(e2_match.group(1))
    elif caspt2 is not None and rasscf is not None:
        result["caspt2_correction_hartree"] = float(caspt2 - rasscf)
    rank_match = CASPT2_RANK_PATTERN.search(text)
    if rank_match:
        result["contracted_basis_rank"] = int(rank_match.group(1))
    return result


def find_openmolcas_command(requested: str | None):
    if requested:
        return requested
    return shutil.which("pymolcas") or shutil.which("molcas")


def _process_ids():
    try:
        import psutil

        return set(psutil.pids())
    except Exception:
        return set()


def _monitor_module_processes(proc, stop, records, baseline_pids):
    """Sample OpenMolcas child lifetimes without modifying the external build."""
    try:
        import psutil

        root = psutil.Process(proc.pid)
    except Exception:
        return
    while not stop.is_set():
        now = time.perf_counter()
        try:
            candidates = root.children(recursive=True)
            if baseline_pids:
                candidates.extend(
                    psutil.Process(pid)
                    for pid in set(psutil.pids()).difference(baseline_pids)
                    if pid != proc.pid
                )
        except Exception:
            candidates = []
        for child in candidates:
            try:
                name = child.name().removesuffix(".exe").lower()
                cpu = child.cpu_times()
            except Exception:
                continue
            if name not in {"gateway", "seward", "scf", "rasscf", "caspt2"}:
                continue
            record = records.setdefault(
                child.pid,
                {
                    "module": name,
                    "first_seen_s": now,
                    "last_seen_s": now,
                    "user_cpu_s": 0.0,
                    "system_cpu_s": 0.0,
                },
            )
            record["last_seen_s"] = now
            record["user_cpu_s"] = float(cpu.user)
            record["system_cpu_s"] = float(cpu.system)
        stop.wait(0.001)


def _summarize_module_processes(records):
    summary = {}
    for record in records.values():
        module = record["module"]
        target = summary.setdefault(
            module,
            {"observed_wall_s": 0.0, "user_cpu_s": 0.0, "system_cpu_s": 0.0},
        )
        target["observed_wall_s"] += max(
            0.0, record["last_seen_s"] - record["first_seen_s"]
        )
        target["user_cpu_s"] += record["user_cpu_s"]
        target["system_cpu_s"] += record["system_cpu_s"]
    return summary


def run_openmolcas_case(
    input_path: Path,
    command: str | None,
    mode: str,
    timeout: float,
):
    if mode == "never":
        return {
            "status": "not_run",
            "reason": "OpenMolcas execution disabled",
            "command": command,
            "input_path": str(input_path),
        }
    resolved = find_openmolcas_command(command)
    if resolved is None:
        status = "skipped" if mode == "auto" else "missing"
        return {
            "status": status,
            "reason": "pymolcas/molcas executable was not found on PATH",
            "input_path": str(input_path),
        }
    argv = shlex.split(resolved) + [input_path.name]
    t0 = time.perf_counter()
    baseline_pids = _process_ids()
    proc = subprocess.Popen(
        argv,
        cwd=input_path.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    records = {}
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_module_processes,
        args=(proc, stop, records, baseline_pids),
        daemon=True,
    )
    monitor.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    finally:
        stop.set()
        monitor.join(timeout=1.0)
    elapsed = time.perf_counter() - t0
    output = stdout + "\n" + stderr
    log_path = input_path.with_suffix(".openmolcas.log")
    log_path.write_text(output, encoding="utf-8")
    parsed = parse_openmolcas_output(output)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "module_process_timings_s": _summarize_module_processes(records),
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
    if mode == "never":
        return {
            "status": "not_run",
            "reason": "Psi4/Forte execution disabled",
            "command": command,
            "input_path": str(input_path),
        }
    resolved = find_psi4_command(command)
    if resolved is None:
        status = "skipped" if mode == "auto" else "missing"
        return {
            "status": status,
            "reason": "psi4 executable was not found on PATH",
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
        for contraction in _expand_selection(args.contraction, ("full", "uncontracted", "strong")):
            pyqed_runs.append(
                run_pyqed_caspt2(
                    mc,
                    zeroth_order=zeroth_order,
                    contraction=contraction,
                    contracted_matrix=args.contracted_matrix,
                    repeat=args.repeat,
                    max_external_determinants=args.max_external_determinants,
                    ic_basis_backend=args.ic_basis_backend,
                    linear_solver=args.linear_solver,
                    max_memory_mb=args.max_memory_mb,
                )
            )

    work_dir = Path(args.work_dir)
    openmolcas_runs = []
    for _ in range(args.openmolcas_repeat):
        input_path = write_openmolcas_input(case, mol, work_dir)
        openmolcas_runs.append(
            run_openmolcas_case(
                input_path,
                command=args.openmolcas_command,
                mode=args.run_openmolcas,
                timeout=args.openmolcas_timeout,
            )
        )
    openmolcas = dict(openmolcas_runs[-1])
    if len(openmolcas_runs) > 1:
        elapsed_samples = [
            float(run["elapsed_s"])
            for run in openmolcas_runs
            if run.get("status") == "ok"
        ]
        module_samples = {}
        for run in openmolcas_runs:
            for module, timing in run.get("module_process_timings_s", {}).items():
                observed_wall = float(timing["observed_wall_s"])
                if observed_wall > 0.0:
                    module_samples.setdefault(module, []).append(observed_wall)
        openmolcas["elapsed_samples_s"] = elapsed_samples
        openmolcas["median_elapsed_s"] = (
            float(statistics.median(elapsed_samples)) if elapsed_samples else None
        )
        openmolcas["min_elapsed_s"] = min(elapsed_samples) if elapsed_samples else None
        openmolcas["module_wall_samples_s"] = module_samples
    psi4_forte_input = write_psi4_forte_input(case, mol, work_dir)
    psi4_forte = run_psi4_forte_case(
        psi4_forte_input,
        command=args.psi4_command,
        mode=args.run_psi4,
        timeout=args.psi4_timeout,
    )

    reference_matched = None
    om_reference = openmolcas.get("rasscf_total_energy_hartree")
    if om_reference is not None:
        reference_delta = reference["reference_energy_hartree"] - float(om_reference)
        reference_matched = abs(reference_delta) <= args.reference_match_tol
        openmolcas["delta_vs_pyqed_reference_hartree"] = reference_delta
        openmolcas["reference_match_tolerance_hartree"] = args.reference_match_tol
        openmolcas["reference_matched"] = reference_matched

    if openmolcas.get("caspt2_total_energy_hartree") is not None:
        om_total = float(openmolcas["caspt2_total_energy_hartree"])
        for run in pyqed_runs:
            run["delta_vs_openmolcas_total_hartree"] = run["e_tot_hartree"] - om_total
            run["openmolcas_comparison_valid"] = bool(reference_matched)
    if psi4_forte.get("forte_dsrg_mrpt2_total_energy_hartree") is not None:
        forte_total = float(psi4_forte["forte_dsrg_mrpt2_total_energy_hartree"])
        for run in pyqed_runs:
            run["delta_vs_forte_dsrg_mrpt2_total_hartree"] = run["e_tot_hartree"] - forte_total

    reference_note = (
        f"The OpenMolcas and PyQED {case.reference.upper()} references match within "
        f"{args.reference_match_tol:.1e} hartree."
        if reference_matched
        else "The external total-energy delta is not a release gate unless the reference_matched field is true."
    )
    return {
        "case": asdict(case),
        "pyqed_reference": reference,
        "pyqed_caspt2": pyqed_runs,
        "openmolcas": openmolcas,
        "psi4_forte": psi4_forte,
        "notes": [
            f"Both legs use {case.reference.upper()}, full internal contraction, and zero IPEA shift.",
            reference_note,
            "Psi4/Forte DSRG-MRPT2 is an adjacent multireference PT2 benchmark, not standard CASPT2.",
        ],
    }


def plot_benchmark(results, output_path):
    """Plot energy, contraction dimensions, and native CASPT2 runtime."""
    import matplotlib.pyplot as plt

    records = []
    for case in results["cases"]:
        for run in case["pyqed_caspt2"]:
            records.append((case["case"]["name"], run, case["openmolcas"]))
    if not records:
        return None

    labels = [
        f"{case}\n{run['zeroth_order']}/{run['contraction']}"
        for case, run, _openmolcas in records
    ]
    x = np.arange(len(records), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(max(12.0, 3.0 * len(records)), 4.2))

    bottom = np.zeros(len(records), dtype=float)
    component_labels = list(CASPT2.perturber_classes)
    for component in component_labels:
        values = np.array(
            [run["components"].get(component, {}).get("energy_hartree", 0.0) * 1000.0
             for _case, run, _openmolcas in records]
        )
        if np.any(np.abs(values) > 1.0e-12):
            axes[0].bar(x, values, bottom=bottom, label=component)
            bottom += values
    openmolcas = np.array(
        [
            external.get("caspt2_correction_hartree", np.nan) * 1000.0
            if external.get("status") == "ok"
            else np.nan
            for _case, _run, external in records
        ]
    )
    matched = np.isfinite(openmolcas)
    if np.any(matched):
        axes[0].scatter(
            x[matched], openmolcas[matched], marker="_", s=260,
            linewidths=2.0, color="black", label="OpenMolcas total", zorder=5,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"CASPT2 contribution (m$E_h$)")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title("Correlation-energy decomposition")

    width = 0.25
    external = np.array(
        [run["external_determinants"] for _case, run, _openmolcas in records]
    )
    raw = np.array(
        [run.get("contracted_basis_size", 0) for _case, run, _openmolcas in records]
    )
    rank = np.array(
        [run.get("contracted_basis_rank", 0) for _case, run, _openmolcas in records]
    )
    axes[1].bar(x - width, external, width, label="external determinants")
    axes[1].bar(x, raw, width, label="raw IC functions")
    axes[1].bar(x + width, rank, width, label="retained metric rank")
    axes[1].set_ylabel("dimension")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Internal-contraction compression")

    pyqed_medians = np.array(
        [run["median_s"] for _case, run, _openmolcas in records], dtype=float
    )
    pyqed_minima = np.array(
        [run["min_s"] for _case, run, _openmolcas in records], dtype=float
    )
    openmolcas_samples = []
    for _case, _run, external in records:
        samples = external.get("module_wall_samples_s", {}).get("caspt2", [])
        if not samples:
            observed = external.get("module_process_timings_s", {}).get("caspt2", {}).get(
                "observed_wall_s"
            )
            samples = [] if observed is None else [observed]
        samples = np.asarray(samples, dtype=float)
        openmolcas_samples.append(samples[samples > 0.0])
    openmolcas_medians = np.array(
        [statistics.median(samples) if samples.size else np.nan for samples in openmolcas_samples]
    )
    openmolcas_minima = np.array(
        [np.min(samples) if samples.size else np.nan for samples in openmolcas_samples]
    )
    runtime_width = 0.34
    axes[2].bar(
        x - runtime_width / 2,
        pyqed_medians,
        runtime_width,
        color="tab:blue",
        label="PyQED median",
    )
    axes[2].scatter(
        x - runtime_width / 2,
        pyqed_minima,
        color="black",
        marker="_",
        s=150,
        label="minimum",
    )
    available = np.isfinite(openmolcas_medians)
    if np.any(available):
        axes[2].bar(
            x[available] + runtime_width / 2,
            openmolcas_medians[available],
            runtime_width,
            color="tab:orange",
            label="OpenMolcas CASPT2 median",
        )
        axes[2].scatter(
            x[available] + runtime_width / 2,
            openmolcas_minima[available],
            color="black",
            marker="_",
            s=150,
        )
    axes[2].set_ylabel("CASPT2 wall time (s)")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Isolated CASPT2 kernel runtime")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="lih_cas22_sto3g",
        help=f"Comma-separated case names or 'all'. Available: {', '.join(CASES)}",
    )
    parser.add_argument("--zeroth-order", default="fock", help="'fock', 'en', or comma-separated/both.")
    parser.add_argument("--contraction", default="full", help="'full', 'uncontracted', 'strong', or comma-separated.")
    parser.add_argument("--contracted-matrix", default="auto", choices=CASPT2.supported_contracted_matrices)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-external-determinants", type=int, default=None)
    parser.add_argument(
        "--ic-basis-backend",
        default="auto",
        choices=CASPT2.supported_ic_basis_backends,
    )
    parser.add_argument(
        "--linear-solver",
        default="auto",
        choices=CASPT2.supported_linear_solvers,
    )
    parser.add_argument("--max-memory-mb", type=float, default=2048.0)
    parser.add_argument("--work-dir", default="/private/tmp/pyqed-openmolcas-caspt2")
    parser.add_argument("--out", default="/private/tmp/pyqed_caspt2_openmolcas_benchmark.json")
    parser.add_argument("--plot", default="/private/tmp/pyqed_caspt2_openmolcas_benchmark.png")
    parser.add_argument("--openmolcas-command", default=None)
    parser.add_argument("--run-openmolcas", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--openmolcas-repeat", type=int, default=1)
    parser.add_argument("--openmolcas-timeout", type=float, default=600.0)
    parser.add_argument("--reference-match-tol", type=float, default=1.0e-7)
    parser.add_argument("--psi4-command", default=None)
    parser.add_argument("--run-psi4", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--psi4-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    if args.repeat < 1 or args.openmolcas_repeat < 1:
        raise ValueError("--repeat and --openmolcas-repeat must be positive.")
    if args.reference_match_tol <= 0.0:
        raise ValueError("--reference-match-tol must be positive.")

    selected_cases = _expand_selection(args.cases, CASES)
    results = {
        "benchmark": "pyqed_native_caspt2_external_mrpt_benchmark",
        "created_by": "benchmarks/caspt2_openmolcas.py",
        "cases": [benchmark_case(args, CASES[name]) for name in selected_cases],
    }

    if args.plot:
        results["plot"] = plot_benchmark(results, args.plot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonify(results), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonify(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
