#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:36:15 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF

mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='631g')
mol.build()
mf = mol.RHF().run()



ncas, nelecas = 4, 4

mc = CASSCF(mf, ncas, nelecas)
mc.run()

####### PYSCF ########

mol = mol.topyscf()
mf = mol.RHF().run()


from pyscf.mcscf import CASSCF
from pyscf import mcscf

# mc = CASSCF(mf, ncas=4, nelecas=4)
# mc.verbose = 4
# weights = [0.5, 0.5]
# mc.state_average(weights)


mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.kernel()

# mc = mcscf.CASSCF(mf, ncas, nelecas).state_average_(weights=(1/3,1/3,1/3))
# mc.fix_spin_(ss=0, shift=0.2)
# mc.kernel()

# mc.run()