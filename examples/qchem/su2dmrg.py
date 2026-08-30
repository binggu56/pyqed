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
mol.build(
    eri="dense",
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
    nsweeps=2,
    mixer_zero_block_noise_scale=0.0,
)

print("E =", dmrg.e_tot)

for sweep in dmrg.dmrg.history:
    print("sweep", sweep["sweep"], "E", sweep.get("energy"))
    print("backend:", sweep["backend_actual"])
