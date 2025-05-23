#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:27:05 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np
from pyqed import gwp, interval

nx = ny = nz = 32
nstates = 6
psi0 = np.zeros((nx, ny, nz, nstates), dtype=complex)

x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
z = np.linspace(-2, 2, nz)
ndim = 3
dx = dy = dz = interval(x)

a = 0
b = 0 
c = 0

# for n in range(nstates):
# psi0 = np.zeros((nx, ny), dtype=complex)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            psi0[i, j, k, 5] = gwp(np.array([x[i], y[j], z[k]]), a = 18 * np.eye(ndim), ndim=3) #a=18  x0=0.0, 1.78125 for x[31], y[50]
            # test_norm = np.sum(np.abs(psi0)**2) * dx * dy * dz
            # print("Test norm of gwp:", test_norm)
        
print("norm",np.einsum('ijka, ijka ->', psi0.conj(), psi0) * dx * dz * dy)  
      
  
print('psi0.shape', psi0.shape)

# psi0_new = np.einsum('ijkn,ijk->ijkn', sol.A[:, :, :, :, a, b, c, state], psi0[:,:,:,state])