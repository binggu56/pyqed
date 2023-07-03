#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:36:22 2023

@author: bing
"""

# from pyqed.lattice.chain import green_renormalization
import numpy as np
from numpy.linalg import inv


from pyqed import pauli, dagger

def green_renormalization(intra,inter,energy=0.0,nite=None,
                            info=False,delta=0.001,**kwargs):
    """ Calculates bulk and surface Green function by a renormalization
    algorithm, as described in I. Phys. F: Met. Phys. 15 (1985) 851-858 """
    # intra = algebra.todense(intra)
    # inter = algebra.todense(inter)
    error = np.abs(delta)*1e-6 # overwrite error

    e = np.matrix(np.identity(intra.shape[0])) * (energy + 1j*delta)
    ite = 0
    alpha = inter.copy()
    beta = dagger(inter).copy()
    epsilon = intra.copy()
    epsilon_s = intra.copy()

    print(' ite, alpha, beta')
    while True: # implementation of Eq 11
      einv = inv(e - epsilon) # inverse
      epsilon_s = epsilon_s + alpha @ einv @ beta
      epsilon = epsilon + alpha @ einv @ beta + beta @ einv @ alpha
      alpha = alpha @ einv @ alpha  # new alpha
      beta = beta @ einv @ beta  # new beta
      ite += 1
      print(ite, 'alpha = \n', alpha, '\n beta', beta)
      # stop conditions
      if not nite is None:
        if ite > nite:  break
      else:
        if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error: break

    if info:
      print("Converged in ",ite,"iterations")
    g_surf = inv(e - epsilon_s) # surface green function
    g_bulk = inv(e - epsilon)  # bulk green function
    return g_bulk, g_surf

s0, sx, sy, sz = pauli()

gb, gs = green_renormalization(intra=20*sx,inter=np.array([[0, 0], [1, 0]]),energy=0.0,nite=None,
                            info=True)

print(gs)