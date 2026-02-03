#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 00:07:50 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""




from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed import Molecule
from pyqed.qchem.dmrg.dmrg import QCDMRG

# np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


mol = Molecule(atom = [
    ['He' , (0. , 0. , 0.91)],
    ['He' , (0. , 0. , -0.91)],
    ['H' , (0. , 0. , 3.6)],
    ['H' , (0. , 0. , -3.6)]])

# mol.basis = 'ccpvdz'
mol.basis = '6311g'
mol.build(driver='pyscf')

mf = mol.RHF().run()


dmrg = QCDMRG(mf, ncas=12, nelecas=4, D=20, target_qn=None) #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
dmrg.build().run(U1=True)