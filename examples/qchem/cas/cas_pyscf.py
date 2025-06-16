#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:25:53 2023

@author: bing
"""

#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CASSCF calculation.
'''

import pyscf
import numpy as np

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(6, 8).run()
#
# Note this mycas object can also be created using the APIs of mcscf module:
#
# from pyscf import mcscf
# mycas = mcscf.CASSCF(myhf, 6, 8).run()

# Natural occupancy in CAS space, Mulliken population etc.
# See also 00-simple_casci.py for the instruction of the output of analyze()
# method
# mycas.verbose = 4

#
# 1pdm in AO representation
#
dm1 = mycas.make_rdm1()

# mycas.nstates = 2
#
# alpha and beta 1-pdm in AO representation
#
dm1_alpha, dm1_beta = mycas.make_rdm1s()

print(np.allclose(dm1, dm1_alpha+dm1_beta))

mycas.analyze()