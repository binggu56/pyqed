#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:33:38 2026

@author: gugroup
"""

from pyqed import Molecule
from pyqed.qchem import COCAS
import numpy as np

mol = Molecule(atom="Li 0 0 0; F 0 0 1.4", unit="b", basis="631g")
mol.build(eri="dense")

mf = mol.RHF().run()
print("E(HF) =", mf.e_tot)

mc = COCAS(
    mf,
    ncas=8,
    nelecas=8,
    max_cycles=10,
    macro_tol=1e-6,
    optimizer="RCG",
    optimizer_tol=1e-4,
    optimizer_max_steps=200,
    macro_trust_radius=0.5,
    macro_trust_max=4.0,
    macro_trust_grow=2.0,
    reject_macro_energy=False,
    verbose=1,
)

mc.run(nstates=1, use_cholesky=False)

print("E(COCAS) =", mc.e_tot)
# print("Energy history =", [float(x) for x in mc.e_history])
# print("Macro diagnostics =", mc.macro_diagnostics)