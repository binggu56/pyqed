#!/usr/bin/env python3
"""State-averaged factorized CASSCF example."""

from pyqed.qchem import CASSCF, Molecule


mol = Molecule(
    atom="H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7",
    unit="angstrom",
    basis="sto-3g",
)
mol.build(driver="builtin", eri="factors")

# A Cholesky RHF reference makes CASSCF use factorized integrals automatically.
mf = mol.RHF().run()

mc = CASSCF(mf, ncas=4, nelecas=4).state_average([0.5, 0.5]).run(nstates=2)

print("SA-CASSCF energies:", mc.e_tot)