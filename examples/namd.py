#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:48:06 2020

@author: bing
"""

import sys
sys.path.append(r'C:\Users\Bing\Google Drive\lime')
sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.namd import NAMD, gwp 
import numpy as np 

nx = 256
mass = 1. 
nstates = 2 

x = np.linspace(-12, 12, nx)
psi0 = np.zeros((len(x), nstates), dtype=complex) 
psi0[:, 1] = gwp(x, 1.0, 0.0, 0.0)



V_x = np.zeros((nx, nstates**2))
V_x[:,0] =  0.5*x**2 
V_x[:,2] = V_x[:,1] = 0.2
V_x[:,3] = 4*(x-1)**2 + 2.0  


mol = NAMD(x, nstates, psi0, mass, V_x)

dt = 0.001

 
import matplotlib.pyplot as plt 
fig, (ax1,ax2) = plt.subplots(nrows=2)

psi = psi0.copy()
for k in range(10):
    psi = mol.propagate(dt, psi, Nsteps=200)
    ax1.plot(x, psi[:,0].real)
    ax2.plot(x, psi[:,1].real)

