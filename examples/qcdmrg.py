#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 00:07:50 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""



import numpy as np
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed import Molecule
from pyqed.qchem.mol import atomic_chain
from pyqed.qchem.dmrg.dmrg import DMRG

# np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


# mol = Molecule(atom = [
#     ['He' , (0. , 0. , 0.91)],
#     ['He' , (0. , 0. , -0.91)],
#     ['H' , (0. , 0. , 3.6)],
#     ['H' , (0. , 0. , -3.6)]])
natom = 4
z = np.linspace(-5, 5, natom)
print(z)
mol = atomic_chain(natom, z)
# mol.basis = 'ccpvdz'
mol.basis = '631g'
mol.build()

mf = mol.RHF().run()


dmrg = DMRG(mf, ncas=4, nelecas=4, D=20) #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
dmrg.build().run()

# dm1 = dmrg.make_rdm1(0)

# print(dm1)
