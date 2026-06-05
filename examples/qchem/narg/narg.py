#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:18:37 2026

@author: gugroup
"""

# import sys
#sys.path.insert(0, "/Users/gugroup/Documents/Codex/2026-05-15/check-manuscritps-narg-there-is-a/research")
# sys.path.insert(0, "/Users/gugroup/Library/CloudStorage/OneDrive-西湖大学/pyqed")


from pyqed.narg.qchem import NARG, LETTA
from pyqed.qchem import Molecule 

mol = Molecule(
    atom="H 0 0 0; F 0 0 0.74",
    unit="angstrom",
    basis="631g",
)
mol.build()

mf = mol.RHF().run()

# NARG
narg = NARG(mf, symmetry="abelian", D=8, nstates=10)
narg.run()

# Convention: narg.tensors[-1] is C with shape (4, D, nroots)
letta = LETTA.from_narg(narg, root=0, site="spatial")
res = letta.run(nsweeps=4)

print("HF     =", mf.e_tot)
print("NARG   =", narg.e_tot)
print("LETTA  =", res.energy)
