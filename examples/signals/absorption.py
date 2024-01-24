#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:04:54 2023

@author: bing
"""

import numpy as np
from pyqed import Mol, au2ev

H = np.diag([0., 0.5, 1.1, 1.3])/au2ev # define the model Hamiltonian

dip = np.zeros(H.shape) # define the transiton dipole moment
dip[0,1] = dip[1, 0] = 1.
dip[0, 3] = dip[3,0] = 1
dip[0, 2] = dip[2,0] = 1

mol = Mol(H, edip=dip) # initiate the model

mol.absorption(omegas=np.linspace(0., 2, 200)/au2ev)

# pump = np.linspace(0., 2, 100)/au2ev
# probe = np.linspace(0, 1, 100)/au2ev

# g_idx=[0]
# e_idx= [1, 2, 3]
# f_idx=[2, 3]
