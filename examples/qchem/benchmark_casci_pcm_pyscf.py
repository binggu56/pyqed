"""Benchmark pyqed CASCI+PCM excitation energies against PySCF.

The benchmark uses the same methyl-lactate geometry, RHF/STO-3G orbitals,
CAS(4,4), and PySCF-style C-PCM defaults in both codes.  PySCF is the most
direct external reference for this code path because pyqed's PCM module follows
the same ddCOSMO/PCM conventions.

Example:
    python examples/qchem/benchmark_casci_pcm_pyscf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import CASCI, Molecule, RHF
from pyqed.units import au2ev

METHYL_LACTATE_ATOMS = (
    ("C", 0.000, 0.000, 0.000),
    ("H", 0.620, 0.620, 0.620),
    ("O", -0.950, 0.450, 0.850),
    ("H", -1.500, 1.000, 0.350),
    ("C", -0.500, -1.420, 0.200),
    ("H", -1.100, -1.680, -0.670),
    ("H", 0.350, -2.100, 0.270),
    ("H", -1.120, -1.550, 1.090),
    ("C", 1.180, 0.140, -0.980),
    ("O", 1.300, -0.180, -2.160),
    ("O", 2.120, 0.720, -0.250),
    ("C", 3.350, 0.910, -0.930),
    ("H", 3.250, 1.250, -1.960),
    ("H", 3.930, -0.010, -0.900),
    ("H", 3.890, 1.670, -0.370),
)


def atom_string():
    return "; ".join(f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, x, y, z in METHYL_LACTATE_ATOMS)


def run_pyqed(nstates, pcm_cycles):
    mol = Molecule(atom=atom_string(), unit="angstrom", basis="sto-3g")
    mol.build(driver="pyscf")
    mf = RHF(mol).run()

    gas = CASCI(mf, ncas=4, nelecas=4).run(nstates=nstates)
    pcm = CASCI(mf, ncas=4, nelecas=4).PCM(max_cycle=pcm_cycles).run(nstates=nstates)
    return np.asarray(gas.e_tot), np.asarray(pcm.e_tot)


def run_pyscf(nstates, pcm_cycles):
    from pyscf import gto, mcscf, scf, solvent

    mol = gto.M(
        atom=atom_string(),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)

    gas = mcscf.CASCI(mf, 4, 4)
    gas.fcisolver.nroots = nstates
    e_gas = np.asarray(gas.kernel(verbose=0)[0])

    pcm = mcscf.CASCI(mf, 4, 4)
    pcm.fcisolver.nroots = nstates
    pcm = solvent.PCM(pcm)
    pcm.with_solvent.max_cycle = pcm_cycles
    e_pcm = np.asarray(pcm.kernel(verbose=0)[0])
    return e_gas, e_pcm


def excitation_energies(energies):
    energies = np.asarray(energies, dtype=float)
    return (energies - energies[0]) * au2ev


def print_comparison(label, pyqed_energies, pyscf_energies):
    pyqed_exc = excitation_energies(pyqed_energies)
    pyscf_exc = excitation_energies(pyscf_energies)
    delta = pyqed_exc - pyscf_exc

    print(f"\n{label}")
    print("state      pyqed/eV      pyscf/eV     delta/meV")
    for state, (pyqed_e, pyscf_e, diff) in enumerate(zip(pyqed_exc, pyscf_exc, delta)):
        print(f"{state:5d} {pyqed_e:13.8f} {pyscf_e:13.8f} {1000.0 * diff:13.6f}")
    print(f"max |delta| = {np.max(np.abs(delta)) * 1000.0:.6f} meV")
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstates", type=int, default=10)
    parser.add_argument("--pcm-cycles", type=int, default=10)
    parser.add_argument(
        "--tolerance-mev",
        type=float,
        default=0.05,
        help="Fail if max excitation-energy difference exceeds this threshold.",
    )
    args = parser.parse_args()

    q_gas, q_pcm = run_pyqed(args.nstates, args.pcm_cycles)
    p_gas, p_pcm = run_pyscf(args.nstates, args.pcm_cycles)

    gas_delta = print_comparison("Gas-phase CASCI", q_gas, p_gas)
    pcm_delta = print_comparison("CASCI + PCM", q_pcm, p_pcm)

    max_delta_mev = 1000.0 * max(np.max(np.abs(gas_delta)), np.max(np.abs(pcm_delta)))
    if max_delta_mev > args.tolerance_mev:
        raise SystemExit(
            f"benchmark failed: max |delta| = {max_delta_mev:.6f} meV "
            f"> {args.tolerance_mev:.6f} meV"
        )


if __name__ == "__main__":
    main()
