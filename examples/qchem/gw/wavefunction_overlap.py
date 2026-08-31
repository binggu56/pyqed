#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:30:32 2026

@author: gugroup
"""
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.gw.bse import BSE, TDA
from pyqed.gw.gw import GW


def rhf(delta=0.0):
    atom = (
        f"O 0 0 0; "
        f"H 0 -0.757 {0.587 + delta}; "
        f"H 0  0.757 0.587"
    )
    mol = Molecule(atom=atom, basis="sto-3g", unit="angstrom")
    mol.build(eri="dense")
    return RHF(mol).run(verbose=0)


mf1 = rhf(0.0)
mf2 = rhf(0.05)



gw1 = GW(mf1).run()
gw2 = GW(mf2).run()

# e_corr = gw.rpa_correlation_energy()
# e_tot = gw1.total_energy(method="rpa")

# print(gw.e_corr)
# print(mf1.e_tot, gw1.e_tot)

nroots = 3
tda1 = TDA(gw1).run(nroots=nroots)
tda2 = TDA(gw2).run(nroots=nroots)

bse1 = BSE(gw1).run(nroots=nroots)
bse2 = BSE(gw2).run(nroots=nroots)

print("TDA energies")
print(np.array2string(tda1.e, precision=10))

print("\nBSE energies")
print(np.array2string(bse1.e, precision=10))

print("\nTDA displaced overlap")
print(np.array2string(tda1.wavefunction_overlap(tda2), precision=8))

print("\nBSE displaced overlap")
print(np.array2string(bse1.wavefunction_overlap(bse2), precision=8))