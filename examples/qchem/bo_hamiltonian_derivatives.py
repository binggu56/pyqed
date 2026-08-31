#!/usr/bin/env python3
"""Example: analytic BO-Hamiltonian derivatives from an electronic-state model.

This script computes the first- and second-order nuclear derivatives of the
Born-Oppenheimer electronic Hamiltonian in a CASCI state basis, first in
Cartesian coordinates and then projected onto a simple diatomic stretch mode.

For real normal-mode work, replace ``mode_vectors`` with the normal-mode matrix
you want to project onto.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import CASCI, Molecule, bo_hamiltonian_derivatives


def main():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(options={"eri_representation": "factors"})

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    # Simple H-H stretch direction in Cartesian coordinates.
    # Replace this with your normal-mode matrix for real applications.
    mode_vectors = np.zeros((1, mol.natom, 3))
    mode_vectors[0, 0, 2] = -1.0
    mode_vectors[0, 1, 2] = 1.0

    terms = bo_hamiltonian_derivatives(
        mc,
        state_ids=[0, 1],
        mode_vectors=mode_vectors,
    )

    print("RHF energy:", mf.e_tot)
    print("CASCI energies:", mc.e_tot)
    print()
    print("Cartesian labels:")
    print(terms.cartesian_labels)
    print()
    print("First-order Cartesian derivatives F_cartesian[a, beta, alpha]:")
    print(terms.F_cartesian.real)
    print()
    print("Second-order Cartesian derivatives G_cartesian[a, b, beta, alpha]:")
    print(terms.G_cartesian.real)
    print()
    print("Projected first-order derivatives F_projected[k, beta, alpha]:")
    print(terms.F_projected.real)
    print()
    print("Projected second-order derivatives G_projected[k, l, beta, alpha]:")
    print(terms.G_projected.real)


if __name__ == "__main__":
    main()
