#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:35:53 2023

@author: bing
"""


import numpy as np
from numpy import einsum

from pyqed import norm

from pyscf import gto, scf, dft, tddft
from pyqed.coordinates import Molecule

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)


class LVC_DFT:
    """
    Construct lnear vibronic coupling model by ab initio methods.

    Required parameters:
        1) normal modes
        2) excitation energy
        3) vibronic couplings
            - force
            - interstate coupling

    """
    def __init__(self, mol):
        self.mol = mol
        # self.mf = mf


        self.e = None




from pyscf import eph
mol = gto.M(
# atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
#         ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
#         ['H', [0.000000000000,   1.432564848792,   2.125164035930]]],
# basis = 'sto3g',
# unit = 'Bohr'
# )
atom = '''
 C                  0.000000    1.294496    0.000000
 N                  1.191532    0.687931   -0.000000
 C                  1.121066   -0.647248   -0.000000
 C                 -1.121066   -0.647248   -0.000000
 N                 -1.191532    0.687931   -0.000000
 H                  2.063953   -1.191624    0.000000
 H                  0.000000    2.383248    0.000000
 H                 -2.063953   -1.191624   -0.000000
 N                 -0.000000   -1.375863   -0.000000''',  # in Angstrom
    basis = 'sto3g',
    symmetry = True,
    verbose = 4,
)
mf = scf.RHF(mol).run()

grad = mf.nuc_grad_method().kernel()
assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

myeph = eph.EPH(mf)
ephmat, omega = myeph.kernel()
print('eph matrix', ephmat.shape)
print('phonon frequencies', omega)

# ephmat, omega = eph.eph_df.kernel(mf, disp=1e-4)
# print('eph matrix', ephmat)
# print('phonon frequencies', omega)