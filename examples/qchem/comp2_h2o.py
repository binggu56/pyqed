#!/usr/bin/env python3
"""Minimal COMP2 example on H2O / 6-31G."""

from pyqed.qchem import COMP2, MP2, Molecule


ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
BASIS = "6-31g"


def main():
    mol = Molecule(atom=ATOM, basis=BASIS, unit="angstrom")
    mol.build()

    mf = mol.RHF().run()
    mp2 = MP2(mf).run()
    comp2 = COMP2(
        mf,
        max_cycle=20,
        optimizer="RCG",
        optimizer_max_steps=60,
        optimizer_tol=1.0e-5,
    ).run()

    print("H2O / 6-31G")
    print(f"E(RHF)    = {mf.e_tot:.15f} Eh")
    print(f"E(MP2)    = {mp2.e_tot:.15f} Eh")
    print(f"E(COMP2)  = {comp2.e_tot:.15f} Eh")
    print(f"dE        = {comp2.e_tot - mp2.e_tot:+.6e} Eh")
    print(f"converged = {comp2.converged}")
    print(f"cycles    = {len(comp2.energy_history)}")
    print(f"history   = {comp2.energy_history}")
    print(f"steps     = {comp2.step_history}")
    print(f"rotations = {comp2.rotation_history}")


if __name__ == "__main__":
    main()
