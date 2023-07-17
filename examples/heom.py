#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:21:07 2023

@author: bing
"""

import numpy as np

import time

from pyqed import basis, ket2dm, mol, pauli
# from lime.superoperator import left
from pyqed.oqs import HEOMSolver, dm2vec
import proplot as plt

delta = 0.2 
eps0 = 1.0
gamma1 = 0.5

s0, sx, sy, sz = pauli()
H = - delta/2.0 * sx - eps0/2.0 * sz

def ohmic_spectrum(w):
  # if w == 0.0: # dephasing inducing noise
  #   return gamma1
  # else: # relaxation inducing noise
   return gamma1 / 2 * (w / (2 * np.pi))


# redfield = Redfield_solver(H, c_ops=[sx], spectra=[ohmic_spectrum])
# R, evecs = redfield.redfield_tensor()
rho0 = np.zeros((2,2))
rho0[0, 0] = 1
sol = HEOMSolver(H, c_ops=[sz])
nt = 100 
rho = sol.run(rho0=rho0, dt=0.001, nt=nt, temperature=300, cutoff=5, reorganization=0.2, nado=5)
print(rho)

# propagator approach
rho0 = dm2vec(rho0)

u = sol.propagator(dt=0.001, nt=nt, temperature=300, cutoff=5, reorganization=0.2, nado=5)

print(u @ rho0)
