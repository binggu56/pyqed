#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:48:10 2026

@author: gugroup
"""

from pyqed.qchem import Molecule, soc_state_interaction
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed import au2ev

mol = Molecule(atom='''
S  0.000000  0.000000  0.000000
H  0.000000  1.229000  0.958000
H  0.000000 -1.229000  0.958000
''', unit='angstrom', basis='sto-3g', spin=0)

mol.build(driver='gbasis')

mf = mol.RHF().run()

mc_s = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
mc_t = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')

res = soc_state_interaction([(mc_s, 0), (mc_t, 0)])
print(res.h_soc[0, 1] * au2ev)
