#!/usr/bin/env python3
"""Validate native PyQED RHF against an independent PySCF calculation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import platform
from pathlib import Path


def _atom_spec(atoms: list[dict[str, object]]) -> str:
    rows = []
    for atom in atoms:
        x, y, z = atom["coordinates"]
        rows.append(f'{atom["element"]} {x:.17g} {y:.17g} {z:.17g}')
    return "; ".join(rows)


def _parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=directory / "input.json")
    parser.add_argument("--output", type=Path, default=directory / "raw-output.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(args.input.read_text(encoding="utf-8"))
    system = config["system"]
    method = config["method"]
    validation = config["validation"]
    atom = _atom_spec(system["atoms"])

    import numpy as np
    import pyqed
    import pyscf
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf import RHF
    from pyscf import gto, scf

    logging.disable(logging.CRITICAL)
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        molecule = Molecule(
            atom=atom,
            basis=method["basis"],
            unit=system["unit"],
            charge=system["charge"],
            spin=system["spin"],
        )
        molecule.build(
            driver=method["pyqed_integral_driver"],
            options={
                "eri_representation": method["pyqed_eri_representation"],
                "eri_backend": method["pyqed_eri_backend"],
                "aosym": method["pyqed_ao_symmetry"],
                "eri_screen_tol": method["pyqed_eri_screen_tolerance"],
                "parallel": False,
                "eri_workers": method["pyqed_integral_workers"],
                "one_electron_cache": method["pyqed_one_electron_cache"],
                "coord_type": method["coordinate_type"],
            },
        )
        pyqed_mf = RHF(molecule).run(
            tol=method["energy_convergence_tolerance_hartree"],
            conv_tol_dm=method["pyqed_density_convergence_tolerance"],
            conv_tol_grad=method["orbital_gradient_convergence_tolerance"],
            max_cycle=method["maximum_scf_cycles"],
            verbose=0,
        )

        reference_molecule = gto.M(
            atom=atom,
            basis=method["basis"],
            unit="Bohr",
            charge=system["charge"],
            spin=system["spin"],
            cart=False,
            verbose=0,
        )
        reference_mf = scf.RHF(reference_molecule)
        reference_mf.conv_tol = method["energy_convergence_tolerance_hartree"]
        reference_mf.conv_tol_grad = method["orbital_gradient_convergence_tolerance"]
        reference_mf.max_cycle = method["maximum_scf_cycles"]
        reference_mf.kernel()

    pyqed_energy = float(pyqed_mf.e_tot)
    reference_energy = float(reference_mf.e_tot)
    absolute_difference = abs(pyqed_energy - reference_energy)
    tolerance = float(validation["absolute_tolerance"])
    passed = bool(
        pyqed_mf.converged
        and reference_mf.converged
        and np.isfinite(absolute_difference)
        and absolute_difference <= tolerance
    )

    result = {
        "benchmark_id": "h2-sto3g-rhf-pyscf",
        "input": config,
        "runtime": {
            "python_version": platform.python_version(),
            "pyqed_module": str(Path(pyqed.__file__).resolve().relative_to(repository_root)),
            "pyscf_module": str(Path(pyscf.__file__).resolve()),
        },
        "implementations": {
            "pyqed": {
                "version": pyqed.__version__,
                "integral_driver": method["pyqed_integral_driver"],
                "eri_representation": method["pyqed_eri_representation"],
                "eri_backend": molecule._builtin_build_info["eri_backend"],
                "dense_builder": molecule._builtin_build_info["dense_builder"],
                "ao_symmetry": molecule._builtin_build_info["aosym"],
                "coordinate_type": molecule._builtin_build_info["coord_type"],
                "geometry_hash": molecule._builtin_build_info["geometry_hash"],
                "eri_quartets_computed": molecule._builtin_build_info["quartets_computed"],
                "eri_quartets_screened": molecule._builtin_build_info["quartets_screened"],
                "solver": "pyqed.qchem.hf.RHF",
                "converged": bool(pyqed_mf.converged),
                "scf_iterations": int(pyqed_mf.scf_info["iterations"]),
                "total_rhf_energy_hartree": pyqed_energy,
            },
            "pyscf": {
                "version": pyscf.__version__,
                "solver": "pyscf.scf.RHF",
                "converged": bool(reference_mf.converged),
                "total_rhf_energy_hartree": reference_energy,
            },
        },
        "validation": {
            "reference": "PySCF",
            "quantity": validation["quantity"],
            "unit": validation["unit"],
            "absolute_tolerance": tolerance,
            "absolute_difference": absolute_difference,
            "passed": passed,
        },
        "captured_output": {
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["validation"], indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
