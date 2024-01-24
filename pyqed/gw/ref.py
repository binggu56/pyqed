#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:01:49 2023

@author: bingg
"""

#!/usr/bin/env python

'''
A simple example to run a GW calculation 
'''

from pyscf import gto, dft, gw, scf
from pyqed import au2ev 

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = '631g') 

mf = scf.RHF(mol)
# mf.xc = 'pbe'
mf.kernel()

nocc = mol.nelectron//2

# By default, GW is done with analytic continuation
gw = gw.GW(mf, freq_int='ac')
# same as gw = gw.GW(mf, freq_int='ac')
gw.kernel()
print(mf.mo_energy)
print(gw.mo_energy)