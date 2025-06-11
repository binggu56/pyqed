#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:40:10 2024

@author: bingg
"""

from pyqed import RHF, FCI, Molecule
import pyqed

atom= 'H 0, 0, -3.6; \
    H 0, 0, -1.2; \
    H 0, 0, 1.2; \
    H 0, 0, 3.6'

mol = Molecule(atom, basis='631G')

# import os
# print(os.path.dirname(__file__))



mol.build()
print(mol.eri.shape)
mf = mol.RHF().run()
print(mf.mo_occ)

# FCI(mf).run(3)