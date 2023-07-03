#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:10:07 2020

@author: bing

@Test two-photon absorption signal. 
"""



import numpy as np 
import proplot as plt
from scipy.special import erfc 
import yaml 
from matplotlib import cm 
from numpy import sqrt 
import sys
sys.path.append(r'C:\Users\Bing\Google Drive\lime')
sys.path.append(r'/Users/bing/Google Drive/lime')


from lime.phys import dag, lorentzian, pauli, norm, ket2dm, obs, transform 
from lime.style import matplot, set_style 
from lime.style import linecolors as lc  
from lime.units import au2ev, au2fs, au2kev, au2as, au2nm, au2wavenumber
from lime.cavity import Cavity, Polariton
from lime.optics import Pulse 
#from lime.oqs import Oqs 
from lime.mol import Mol

from lime.signal.sos import linear_absorption, TPA2D_time_order, TPA2D 
import lime.signal.liouville as so







N = 5 # number of states
E = [0, 0.6, 1.2, 1.6, 3.0] 
ham = np.diagflat(E)
dip = np.zeros((N,N)) 
for i in range(N):
    for j in range(i):
        dip[i,j] = dip[j,i] = 1. 

print('number of molecular states = {}'.format(N))


mol = Mol(ham, dip) # number of single-polariton states 
        
pump = np.linspace(0, 2)
probe = np.linspace(0, 2) 
omega_min = pump.min() 
omega_max = pump.max() 

gamma = np.zeros(N)

e_idx = [1, 2] 
f_idx = [3, 4] 


gamma[e_idx] = 0.1
gamma[f_idx] = 0.1
#gamma[0] = 0. # ground state has infinity lifetime 
 


omegaps = 2. * pump 
omega1s = probe 

signal = TPA2D(E, dip, omegaps=omegaps, omega1s=omega1s, g_idx=[0], e_idx=e_idx, f_idx = f_idx, \
              gamma=gamma)  


#np.savez('DQC23', R1=R1a, R2=R2b)  


from scipy import ndimage 

fig, ax = plt.subplots()


signal = ndimage.zoom(signal, 3)

scale = np.amax((signal))
signal /= scale 
print('Signal is scaled by {}'.format(scale))

extent = [2. * omega_min, 2. * omega_max, omega_min, omega_max]        

# im = ax.imshow(signal, interpolation='bilinear', cmap=cm.RdBu_r,\
#         extent=extent, origin='lower', norm=norm, 
#         vmax=0.2, vmin=-0.5, aspect=1) #-abs(SPE).max())
# levels = p.append(np.linspace(-1, -0.1, 15), np.linspace(0.1, 1, 15))
levels = np.linspace(0.01, 1, 20)

pump = np.linspace(0, 2, 150)
probe = np.linspace(0, 4, 150)

im = ax.contourf(pump, probe, signal, lw=0.5, cmap='viridis')

ax.set_xlabel(r'$\Omega_2$/eV')
ax.set_ylabel(r'$\Omega_3$/eV')

    