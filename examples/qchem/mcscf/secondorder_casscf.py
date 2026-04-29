#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:27:27 2026

@author: gugroup
"""

from pyqed.qchem import Molecule, CASSCF

mol = Molecule(
    atom='Li 0 0 0; F 0 0 1.4',
    unit='bohr',
    basis='631g',
)
mol.build(driver='builtin')

mf = mol.RHF().run()
mc = CASSCF(mf, ncas=10, nelecas=10).run()

print(mc.e_tot)
# print(mc.e_history)
# print(mc.mo_coeff)

# -105.492896927289

# mol.topyscf().RHF().run().CASSCF(6,6).run()