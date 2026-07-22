#!/usr/bin/env python3
"""Report qchem backend paths and timings for small RHF/AO2MO cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyqed.qchem import Molecule


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _build_case(name, atom, basis, *, eri="auto", unit="bohr", options=None, max_cycle=80):
    mol = Molecule(atom=atom, basis=basis, unit=unit)

    t0 = time.perf_counter()
    mol.build(driver="builtin", eri=eri, options=options)
    build_time = time.perf_counter() - t0

    mf = mol.RHF()
    t0 = time.perf_counter()
    mf.run(max_cycle=max_cycle)
    rhf_time = time.perf_counter() - t0

    ao2mo_time = None
    ao2mo_path = None
    if getattr(mf, "eri_factors", None) is not None or getattr(mol, "eri_factors", None) is not None:
        t0 = time.perf_counter()
        factors = mf.mo_factors(mf.mo_coeff)
        ao2mo_time = time.perf_counter() - t0
        ao2mo_path = {
            "kind": "mo-pair-factors",
            "shape": tuple(int(x) for x in factors.shape),
        }
    elif getattr(mol, "eri_s8", None) is not None or getattr(mf, "eri_s8", None) is not None:
        t0 = time.perf_counter()
        eri_mo = mf.get_eri_mo()
        ao2mo_time = time.perf_counter() - t0
        ao2mo_path = {
            "kind": "dense-from-s8",
            "shape": tuple(int(x) for x in eri_mo.shape),
        }

    direct_data = getattr(mol, "_builtin_direct_jk_data", None)
    return {
        "name": name,
        "basis": basis,
        "eri": eri,
        "nao": int(mol.nao),
        "build_time_s": build_time,
        "rhf_time_s": rhf_time,
        "rhf_energy": float(mf.e_tot),
        "rhf_converged": bool(mf.converged),
        "rhf_cholesky_jk": bool(getattr(mf, "cholesky_jk", False)),
        "build_info": _jsonable(getattr(mol, "_builtin_build_info", {})),
        "direct_jk_last": None if direct_data is None else {
            "kernel": direct_data.get("kernel"),
            "screening": direct_data.get("screening"),
            "task_cache": direct_data.get("task_cache"),
            "last_mode": direct_data.get("last_mode"),
            "last_computed": direct_data.get("last_computed"),
            "last_skipped": direct_data.get("last_skipped"),
        },
        "ao2mo": ao2mo_path,
        "ao2mo_time_s": ao2mo_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-medium", action="store_true", help="include a water/def2-svp auto-RI case")
    parser.add_argument("--out", type=Path, help="write JSON report to this path")
    args = parser.parse_args()

    cases = [
        ("h2_auto", "H 0 0 0; H 0 0 1.4", "sto-3g", "auto", None),
        ("h2_direct", "H 0 0 0; H 0 0 1.4", "sto-3g", "direct", None),
        ("h2_ri", "H 0 0 0; H 0 0 1.4", "sto-3g", "ri", None),
    ]
    if args.include_medium:
        cases.append((
            "h2o_auto",
            "O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
            "def2-svp",
            "auto",
            None,
        ))

    report = {
        "schema": "pyqed-qchem-path-report-v1",
        "cases": [
            _build_case(name, atom, basis, eri=eri, options=options)
            for name, atom, basis, eri, options in cases
        ],
    }
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
