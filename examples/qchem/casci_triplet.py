#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:30:28 2026

@author: gugroup
"""

from pyqed.qchem import Molecule, CASCI

# H2 example
mol = Molecule(
    atom="H 0 0 0; F 0 0 1.4",
    unit="bohr",
    basis="sto-3g",
    spin=0,
)

# Fast builtin integral path
mol.build(driver="builtin", options={"eri_representation": "factors"})

mf = mol.RHF().run()

# Singlet CASCI
mc_s = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1)

# Triplet CASCI sector
# In pyqed, spin = N_alpha - N_beta, so triplet Ms=+1 means spin=2
mc_t = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1)

print("Singlet energy:")
print(mc_s.e_tot)

print("Triplet energies:")
for i, e in enumerate(mc_t.e_tot):
    print(f"T root {i}: {e:.12f} Ha")