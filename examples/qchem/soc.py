#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:48:10 2026

@author: gugroup
"""

from pyqed.qchem import Molecule, st_soc
from pyqed import au2ev

mol = Molecule(atom='''
S  0.000000  0.000000  0.000000
H  0.000000  1.229000  0.958000
H  0.000000 -1.229000  0.958000
''', unit='angstrom', basis='sto-3g', spin=0)

mol.build()

mf = mol.RHF().run()

for model in ('1e', 'somf'):
    res = st_soc(
        mf,
        ncas=2,
        nelecas=2,
        model=model,
        method='direct_ci',
    )
    print(model, {ms: value * au2ev for ms, value in res.components.items()})
    print(model, 'norm', res.norm * au2ev)
