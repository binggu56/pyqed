#!/usr/bin/env python3
"""Compare LiH CASCI states at two nearby bond lengths.

The printed matrix contains overlaps between the three active-space states at
2.0 and 2.1 bohr and can be used to track state character across geometries.
"""

import numpy as np

from pyqed import Molecule
from pyqed.qchem.mcscf.casci import CASCI, overlap


def run_casci(bond_length):
    """Return a three-state CASCI calculation for LiH."""
    mol = Molecule(
        atom=f"H 0 0 0; Li 0 0 {bond_length}",
        unit="bohr",
        basis="6-31g",
    )
    mol.build()
    mf = mol.RHF().run()
    return CASCI(mf, ncas=4, nelecas=2).run(nstates=3)


if __name__ == "__main__":
    reference = run_casci(2.0)
    displaced = run_casci(2.1)
    state_overlap = overlap(reference, displaced)

    np.set_printoptions(precision=6, suppress=True)
    print("State-overlap matrix:")
    print(state_overlap)
