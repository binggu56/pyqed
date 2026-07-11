#!/usr/bin/env python3
"""Compare native pyqed CASSCF macroiterations against pure PySCF CASSCF."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed import Molecule
from pyqed.qchem import CASSCF


PRESETS = {
    "ring": {
        "atom_list": [
            ["N", 0.0, 0.0, 2.9572191239],
            ["C", 0.0, 1.9852542304, 1.3474835864],
            ["C", 0.0, 1.9852542304, -1.3474835864],
            ["N", 0.0, 0.0, -2.9572191239],
            ["C", 0.0, -1.9852542304, -1.3474835864],
            ["C", 0.0, -1.9852542304, 1.3474835864],
            ["H", 0.0, 3.8965550440, 2.1759129688],
            ["H", 0.0, 3.8965550440, -2.1759129688],
            ["H", 0.0, -3.8965550440, -2.1759129688],
            ["H", 0.0, -3.8965550440, 2.1759129688],
        ],
        "unit": "b",
        "basis": "6-31g",
        "ncas": 6,
        "nelecas": 6,
        "nstates": 3,
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "spin_ss": 0.0,
        "spin_shift": 0.2,
        "pyqed_kwargs": {"max_cycles": 50},
        "pyscf_kwargs": {"max_cycle_macro": 50, "max_cycle_micro": 8, "max_stepsize": 0.02},
    },
    "lih": {
        "atom_list": "Li 0 0 0; H 0 0 1.6",
        "unit": "angstrom",
        "basis": "sto-3g",
        "ncas": 2,
        "nelecas": 2,
        "nstates": 1,
        "weights": None,
        "spin_ss": None,
        "spin_shift": 0.2,
        "pyqed_kwargs": {"max_cycle": 40},
        "pyscf_kwargs": {"max_cycle_macro": 40, "max_stepsize": 0.02},
    },
}


def _capture_pyscf_macro_history(mc, mo_coeff):
    history = []

    def callback(envs):
        cycle = int(envs["imacro"])
        entry = {
            "cycle": cycle,
            "energy": float(envs["e_tot"]),
            "gradient_norm": float(envs["norm_gorb"]),
            "ddm_norm": float(envs.get("norm_ddm", np.nan)),
        }
        if history and history[-1]["cycle"] == cycle:
            history[-1] = entry
        else:
            history.append(entry)

    mc.kernel(mo_coeff=np.array(mo_coeff, copy=True), callback=callback)
    return history


def _run_pyqed_case(case):
    mol = Molecule(atom=copy.deepcopy(case["atom_list"]), unit=case["unit"], basis=case["basis"])
    mol.build(driver="pyscf")
    mf = mol.RHF().run()

    mc = CASSCF(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        **case["pyqed_kwargs"],
    )
    if case["weights"] is not None:
        mc.state_average(case["weights"])
    if case["spin_ss"] is not None:
        mc.fix_spin(ss=case["spin_ss"], shift=case["spin_shift"])

    error = None
    try:
        mc.run(nstates=case["nstates"])
    except RuntimeError as exc:
        error = str(exc)

    return {
        "mf_energy": float(mf.e_tot),
        "history": list(mc.history),
        "converged": bool(mc.converged),
        "e_tot": None if mc.e_tot is None else np.array(mc.e_tot, copy=True),
        "error": error,
    }


def _run_pyscf_case(case):
    mol = gto.M(
        atom=copy.deepcopy(case["atom_list"]),
        basis=case["basis"],
        unit="Bohr" if case["unit"].lower() in ("b", "bohr") else "Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()

    mc = mcscf.CASSCF(mf, case["ncas"], case["nelecas"])
    mc.conv_tol = 1.0e-7
    for key, value in case["pyscf_kwargs"].items():
        setattr(mc, key, value)

    if case["weights"] is not None:
        mc.fcisolver.nroots = len(case["weights"])
        mc = mc.state_average_(case["weights"])
    elif case["nstates"] > 1:
        mc.fcisolver.nroots = case["nstates"]

    if case["spin_ss"] is not None:
        mc.fix_spin_(ss=case["spin_ss"], shift=case["spin_shift"])

    history = _capture_pyscf_macro_history(mc, mf.mo_coeff)
    return {
        "mf_energy": float(mf.e_tot),
        "history": history,
        "converged": bool(mc.converged),
        "e_tot": np.array(getattr(mc, "e_states", [mc.e_tot]), copy=True),
    }


def _first_divergence_cycle(pyqed_history, pyscf_history, energy_threshold):
    nshared = min(len(pyqed_history), len(pyscf_history))
    for idx in range(nshared):
        if abs(pyqed_history[idx]["energy"] - pyscf_history[idx]["energy"]) > energy_threshold:
            return idx + 1
    if len(pyqed_history) != len(pyscf_history):
        return nshared + 1
    return None


def _format_float(value, width=14):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return " " * (width - 3) + "n/a"
    return f"{value:{width}.8f}"


def _format_sci(value, width=11):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return " " * (width - 3) + "n/a"
    return f"{value:{width}.3e}"


def _print_history_table(pyqed_result, pyscf_result):
    pyqed_history = pyqed_result["history"]
    pyscf_history = pyscf_result["history"]
    nrows = max(len(pyqed_history), len(pyscf_history))

    header = (
        "cycle  "
        "pyqed_energy    pyqed_grad   pyqed_step   "
        "pyscf_energy    pyscf_grad    pyscf_ddm    "
        "delta_energy"
    )
    print(header)
    print("-" * len(header))

    for idx in range(nrows):
        py = pyqed_history[idx] if idx < len(pyqed_history) else None
        ps = pyscf_history[idx] if idx < len(pyscf_history) else None
        py_energy = None if py is None else py["energy"]
        ps_energy = None if ps is None else ps["energy"]
        delta_energy = None
        if py_energy is not None and ps_energy is not None:
            delta_energy = py_energy - ps_energy

        print(
            f"{idx + 1:5d}  "
            f"{_format_float(py_energy)}  "
            f"{_format_sci(None if py is None else py['gradient_norm'])}  "
            f"{_format_sci(None if py is None else py['step_norm'])}  "
            f"{_format_float(ps_energy)}  "
            f"{_format_sci(None if ps is None else ps['gradient_norm'])}  "
            f"{_format_sci(None if ps is None else ps['ddm_norm'])}  "
            f"{_format_sci(delta_energy)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=sorted(PRESETS),
        default="ring",
        help="Preset molecule/system to compare.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=1.0e-3,
        help="Energy difference threshold (Eh) for flagging the first divergence cycle.",
    )
    args = parser.parse_args()

    case = PRESETS[args.system]
    print(f"System: {args.system}")
    print(
        "Setup: basis={basis} ncas={ncas} nelecas={nelecas} nstates={nstates}".format(
            basis=case["basis"],
            ncas=case["ncas"],
            nelecas=case["nelecas"],
            nstates=case["nstates"],
        )
    )
    print()

    pyqed_result = _run_pyqed_case(case)
    pyscf_result = _run_pyscf_case(case)

    print("RHF energies")
    print(f"  pyqed: {pyqed_result['mf_energy']:.12f}")
    print(f"  PySCF: {pyscf_result['mf_energy']:.12f}")
    print()

    print("Macroiteration comparison")
    _print_history_table(pyqed_result, pyscf_result)
    print()

    divergence_cycle = _first_divergence_cycle(
        pyqed_result["history"],
        pyscf_result["history"],
        args.energy_threshold,
    )
    if divergence_cycle is None:
        print("No divergence detected within the shared macroiteration history.")
    else:
        print(
            "First divergence cycle (|delta_energy| > {:.1e} Eh or history length mismatch): {}".format(
                args.energy_threshold,
                divergence_cycle,
            )
        )

    print()
    print("Final status")
    print(f"  pyqed converged: {pyqed_result['converged']}")
    if pyqed_result["e_tot"] is not None:
        print(f"  pyqed final states: {pyqed_result['e_tot']}")
    if pyqed_result["error"] is not None:
        print("  pyqed error:")
        print(pyqed_result["error"])
    print(f"  PySCF converged: {pyscf_result['converged']}")
    print(f"  PySCF final states: {pyscf_result['e_tot']}")


if __name__ == "__main__":
    main()
