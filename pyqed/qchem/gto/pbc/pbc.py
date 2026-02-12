# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:06:32 2022

@author: Bing

DFT-GW-BSE first-principle solid state system computations
 
should be valid for not too strongly correlated system? 

The ground state needs to be described by DFT calculations.  

# step 1: SCF calculation -> band structure, Bloch states

# step 2: GW -> quasiparticle energies, screened Culomb interaction

# step 3: BSE calculation -> exciton energies and wavefunctions 

# step 4: biexciton calculation -> biexcitons

"""

from pyscf.pbc import gto
import numpy as np


cell = gto.Cell()
cell.atom = '''H  0 0 0; H 1 1 1'''
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 2
cell.build()


 