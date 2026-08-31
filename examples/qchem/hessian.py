#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:36:03 2026

@author: gugroup
"""

from pyqed.qchem import Molecule
from pyqed.qchem.dft import RKS

mol = Molecule(atom='H 0 0 -0.8; H 0 0 0.8', unit='bohr', basis='sto-3g')
mol.build()

mf = RKS(mol, xc='svwn').run()

hobj = mf.Hessian()
H = hobj.run()
print(H.shape)
print(hobj.frequencies())