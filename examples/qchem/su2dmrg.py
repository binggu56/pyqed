#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:21:45 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import DMRG

mol = Molecule(
    atom="Li 0 0 0; F 0 0 1.6",
    unit="bohr",
    basis="631g",
)
mol.build(eri="dense",
    aosym="s1",
    options={"eri_backend": "cpp"},
)

mf = RHF(mol).run()

dmrg = DMRG(
    mf,
    ncas=8,
    nelecas=8,
    D=16,
    init_guess="cid",
    symmetry="su2",
    verbose=1,
)

dmrg.run(
    nstates=2,
    weights=[0.5, 0.5],
    su2_kernel_backend="cpp",
)

print("Root energies =", dmrg.e_tot)
print("State-average energy =", dmrg.state_average_energy)
print("Converged =", dmrg.converged)

for sweep in dmrg.history:
    print("sweep", sweep["sweep"], "E", sweep.get("energy"))
