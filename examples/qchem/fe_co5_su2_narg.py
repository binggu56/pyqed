#!/usr/bin/env python3
"""Fe(CO)5 CAS(30,30) SU(2)-NARG excitation benchmark."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.narg.qchem import NARG
from pyqed.qchem import Molecule


ATOM = """
Fe     0.0000033513     0.0000000000     0.0000000000
C      0.0000120362     0.0000000000     1.7875478540
C      0.0000120362     0.0000000000    -1.7875478540
C     -0.8972187835    -1.5540061679     0.0000000000
C     -0.8972187835     1.5540061679     0.0000000000
C      1.7944036865     0.0000000000     0.0000000000
O      0.0000196369     0.0000000000     2.9326301797
O      0.0000196369     0.0000000000    -2.9326301797
O     -1.4707073864    -2.5472855808     0.0000000000
O     -1.4707073864     2.5472855808     0.0000000000
O      2.9413819559     0.0000000000     0.0000000000
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=300)
    parser.add_argument("--nstates", type=int, default=20)
    parser.add_argument("--basis", default="ccpvdz")
    args = parser.parse_args()

    mol = Molecule(atom=ATOM, unit="angstrom", basis=args.basis)
    mol.build(driver="pyscf")
    mf = mol.RHF().run()

    solver = NARG(
        mf,
        symmetry="su2",
        D=args.D,
        nstates=args.nstates,
        ncas=30,
        nelecas=30,
        target_j2=0,
    ).run()
    energies = np.asarray(solver.e_tot, dtype=float)
    excitations = (energies - energies[0]) * 27.211386245988

    print(f"HF = {float(mf.e_tot):.12f} Eh")
    print("state  total energy (Eh)  excitation (eV)")
    for state, (energy, excitation) in enumerate(zip(energies, excitations)):
        print(f"{state:5d}  {energy:17.12f}  {excitation:15.9f}")


if __name__ == "__main__":
    main()
