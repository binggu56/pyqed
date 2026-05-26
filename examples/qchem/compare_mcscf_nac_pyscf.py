"""Benchmark PyQED SA-CASSCF NAC variants against PySCF.

The table compares four PyQED NAC estimates against PySCF full and ETFS
SA-CASSCF NACs:

* ``pyqed_mo``: default one-sided moving-MO analytical path.
* ``pyqed_full_ao``: PySCF-style full gauge, ``nac_gauge="full"``.
* ``pyqed_etfs``: PySCF-style ETFS gauge, ``nac_gauge="etfs"``.
* ``pyqed_overlap``: finite-difference state-overlap reference.

PySCF's ``NonAdiabaticCouplings(state=(ket, bra))`` returns ``<bra|d ket>``.
PyQED stores the same object as ``nac[bra, ket]``.  A global sign is optimized
when reporting errors because CI eigenvector phases are arbitrary.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule, SecondOrderCASSCF
from pyqed.qchem.nac.sacasscf import OverlapNACDriver, ResponseBackend, relaxed_nac
from pyqed.qchem.mcscf.zvector import MCSCFZVector


@dataclass(frozen=True)
class Case:
    name: str
    atom: str
    basis: str
    ncas: int
    nelecas: int | tuple[int, int]
    nroots: int
    pair: tuple[int, int]
    charge: int = 0
    spin: int = 0


CASES = {
    "h2o": Case(
        name="H2O asym / STO-3G / CAS(4,4) / SA3 / pair 0,2",
        atom="O 0 0 0; H 0 1.45 1.05; H 0 -1.25 1.2",
        basis="sto-3g",
        ncas=4,
        nelecas=4,
        nroots=3,
        pair=(0, 2),
    ),
    "ch2": Case(
        name="CH2 / STO-3G / CAS(4,4) / SA3 / pair 1,2",
        atom="C 0 0 0; H 1.6 0 1.1; H -1.35 0 1.25",
        basis="sto-3g",
        ncas=4,
        nelecas=4,
        nroots=3,
        pair=(1, 2),
    ),
}


def best_error(vec: np.ndarray, ref: np.ndarray) -> tuple[int, float, float, float]:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    sign = 1 if np.linalg.norm(vec - ref) <= np.linalg.norm(vec + ref) else -1
    diff = vec - sign * ref
    return sign, float(np.linalg.norm(diff)), float(np.max(np.abs(diff))), float(np.linalg.norm(vec))


def build_pyqed(case: Case, coords: np.ndarray | None = None, symbols=None):
    if coords is None:
        mol = Molecule(
            atom=case.atom,
            basis=case.basis,
            charge=case.charge,
            spin=case.spin,
            unit="bohr",
        )
    else:
        mol = Molecule(
            atom=[[symbol, tuple(coord)] for symbol, coord in zip(symbols, coords.reshape(-1, 3), strict=True)],
            basis=case.basis,
            charge=case.charge,
            spin=case.spin,
            unit="bohr",
        )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    driver = (
        SecondOrderCASSCF(
            mf,
            ncas=case.ncas,
            nelecas=case.nelecas,
            max_cycle=100,
            verbose=0,
            coupling="full",
            conv_tol=1.0e-9,
            conv_tol_grad=1.0e-7,
        )
        .state_average([1.0 / case.nroots] * case.nroots)
        .run(nstates=case.nroots)
    )
    return mol, driver


def pyqed_variants(case: Case, *, overlap: bool):
    mol, driver = build_pyqed(case)
    backend = ResponseBackend.from_driver(driver, driver.casci, nroots=case.nroots)
    zvector = MCSCFZVector.from_second_order_driver(
        driver,
        driver.casci,
        mo_coeff=driver.mo_coeff,
        nroots=case.nroots,
        symmetrize=False,
    )
    pair = case.pair
    variants = {
        "pyqed_mo": relaxed_nac(backend, zvector, state_pairs=[pair]).nac[pair],
        "pyqed_full_ao": relaxed_nac(backend, zvector, state_pairs=[pair], nac_gauge="full").nac[pair],
        "pyqed_etfs": relaxed_nac(backend, zvector, state_pairs=[pair], nac_gauge="etfs").nac[pair],
    }
    if overlap:
        symbols = mol.atom_symbols()

        def point_builder(coords):
            _, point = build_pyqed(case, np.asarray(coords, dtype=float), symbols=symbols)
            return point.casci

        variants["pyqed_overlap"] = OverlapNACDriver(
            mol,
            ncas=case.ncas,
            nelecas=case.nelecas,
            nstates=case.nroots,
            step=5.0e-4,
            point_builder=point_builder,
        ).nac(reference=driver.casci)[pair]

    h0, eri0 = driver._get_integrals(driver.mo_coeff)
    grad = sum(
        weight * driver._exact_orbital_gradient_vector(driver.casci, h0, eri0, ci)
        for weight, ci in zip(driver.weights, driver.casci.ci, strict=True)
    )
    return driver, variants, float(np.linalg.norm(grad)), float(np.max(np.abs(grad)))


def pyscf_variants(case: Case):
    from pyscf import gto, mcscf, scf
    from pyscf.nac import sacasscf

    mol = gto.M(
        atom=case.atom,
        basis=case.basis,
        charge=case.charge,
        spin=case.spin,
        unit="bohr",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, case.ncas, case.nelecas).state_average_([1.0 / case.nroots] * case.nroots)
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 1.0e-7
    mc.max_cycle_macro = 100
    mc.kernel()
    ket = case.pair[1]
    bra = case.pair[0]
    full = sacasscf.NonAdiabaticCouplings(mc, state=(ket, bra), use_etfs=False).kernel(verbose=0).reshape(-1)
    etfs = sacasscf.NonAdiabaticCouplings(mc, state=(ket, bra), use_etfs=True).kernel(verbose=0).reshape(-1)
    return mc, {"pyscf_full": full, "pyscf_etfs": etfs}


def print_case(case: Case, *, overlap: bool) -> None:
    pyqed_driver, pyqed, pyqed_grad_norm, pyqed_grad_max = pyqed_variants(case, overlap=overlap)
    pyscf_driver, pyscf = pyscf_variants(case)
    pair = case.pair
    print(f"\n{case.name}")
    print(f"  pair nac[{pair[0]},{pair[1]}], flattened Cartesian length {pyqed['pyqed_mo'].size}")
    print("  PyQED energies:", np.array2string(np.asarray(pyqed_driver.casci.e_tot[: case.nroots]), precision=10))
    print("  PySCF energies:", np.array2string(np.asarray(pyscf_driver.e_states[: case.nroots]), precision=10))
    print(f"  max |dE|: {np.max(np.abs(pyqed_driver.casci.e_tot[: case.nroots] - pyscf_driver.e_states[: case.nroots])):.3e} Eh")
    print(f"  PyQED SA orbital gradient norm/max: {pyqed_grad_norm:.3e} / {pyqed_grad_max:.3e}")

    for name, vec in pyqed.items():
        _, err_full, max_full, norm = best_error(vec, pyscf["pyscf_full"])
        _, err_etfs, max_etfs, _ = best_error(vec, pyscf["pyscf_etfs"])
        print(
            f"  {name:14s} norm={norm:.6e}  "
            f"vs PySCF full: {err_full:.3e} max {max_full:.3e}  "
            f"vs ETFS: {err_etfs:.3e} max {max_etfs:.3e}"
        )
        print("    ", np.array2string(np.asarray(vec).reshape(-1), precision=8, suppress_small=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    parser.add_argument("--skip-overlap", action="store_true", help="Skip finite-difference overlap NACs.")
    args = parser.parse_args()

    keys = CASES.keys() if args.case == "all" else [args.case]
    for key in keys:
        print_case(CASES[key], overlap=not args.skip_overlap)


if __name__ == "__main__":
    main()
