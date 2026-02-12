#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:26:26 2025

@author: bingg
"""

import pyscf

mol = pyscf.M(
    atom = 'Li 0 0 0; H 0 0 1.2',
    basis = '631g',
    unit = 'b',
    spin = 0)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(4, 4)
mycas.verbose = 4
mycas.run()
#
# Note this mycas object can also be created using the APIs of mcscf module:
#
# from pyscf import mcscf
# mycas = mcscf.CASSCF(myhf, 6, 8).run()

# Natural occupancy in CAS space, Mulliken population etc.
# See also 00-simple_casci.py for the instruction of the output of analyze()
# method

# mycas.analyze()