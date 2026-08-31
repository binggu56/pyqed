#!/usr/bin/env python3
"""Minimal H2 CASSCF example using the current pyqed API."""

from pyqed.qchem import CASSCF, Molecule
from pyscf import mcscf


mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
mol.build()
mf = mol.RHF().run()

ncas, nelecas = 2, 2

mc = CASSCF(mf, ncas, nelecas)
mc.run()

print("pyqed RHF energy:", mf.e_tot)
print("pyqed CASSCF energy:", mc.e_tot[0])
print("pyqed converged:", mc.converged)
print("pyqed macro cycles:", len(mc.history))

####### PYSCF ########

pyscf_mol = mol.topyscf()
pyscf_mf = pyscf_mol.RHF().run()
pyscf_mc = mcscf.CASSCF(pyscf_mf, ncas, nelecas)
pyscf_mc.kernel()

print("PySCF RHF energy:", pyscf_mf.e_tot)
print("PySCF CASSCF energy:", pyscf_mc.e_tot)
