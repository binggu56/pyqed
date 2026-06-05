#!/usr/bin/env python3
"""Benchmark PyQED scalar-X2C RHF against PySCF spin-free X2C RHF."""

from dataclasses import dataclass

from pyscf import gto, scf

from pyqed.qchem import Molecule, RHF


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    atom: str
    basis: str
    unit: str = "bohr"


CASES = [
    BenchmarkCase(
        name="HF",
        atom="H 0 0 0; F 0 0 1.7",
        basis="sto-3g",
    ),
    BenchmarkCase(
        name="H2O",
        atom="O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11",
        basis="sto-3g",
    ),
    BenchmarkCase(
        name="HCl",
        atom="H 0 0 0; Cl 0 0 2.4",
        basis="sto-3g",
    ),
]


def pyqed_rhf(case, x2c=False):
    mol = Molecule(atom=case.atom, basis=case.basis, unit=case.unit)
    mol.build(driver="builtin", options={"coord_type": "spherical", "eri_representation": "dense", "aosym": "s1"})
    return RHF(mol).run(x2c=x2c, tol=1e-11, conv_tol_dm=1e-9, max_cycle=100)


def pyscf_rhf(case, x2c=False):
    mol = gto.M(atom=case.atom, basis=case.basis, unit=case.unit, cart=False, verbose=0)
    mf = scf.RHF(mol)
    if x2c:
        mf = mf.x2c()
        # Match PyQED's contracted-basis one-electron X2C Hamiltonian.
        mf.with_x2c.xuncontract = False
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    return mf


def main():
    header = (
        f"{'Molecule':<8} {'Basis':<10} {'Model':<8} "
        f"{'PySCF E_h':>18} {'PyQED E_h':>18} {'Delta E_h':>12}"
    )
    print(header)
    print("-" * len(header))
    for case in CASES:
        for model, use_x2c in (("NR", False), ("X2C", True)):
            ref = pyscf_rhf(case, x2c=use_x2c)
            got = pyqed_rhf(case, x2c=use_x2c)
            print(
                f"{case.name:<8} {case.basis:<10} {model:<8} "
                f"{ref.e_tot:18.10f} {got.e_tot:18.10f} {got.e_tot - ref.e_tot:12.3e}"
            )


if __name__ == "__main__":
    main()
