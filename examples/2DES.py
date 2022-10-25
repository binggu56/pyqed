#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:12:52 2020

@author: Bing Gu 

@description: Examples of the 2DES modules
"""


import numpy as np 
import matplotlib.pyplot as plt
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

from lime.signal.sos import linear_absorption, GSB, SE, ESA, DQC_R1, DQC_R2
import lime.signal.liouville as so







N = 3 # number of states
E = [0, 1, 1.5] 
ham = np.diagflat(E)
dip = np.zeros((N,N)) 
dip[0,1] = dip[1,0] = 1. 
dip[1,2] = dip[2,1] = 1. 

print('number of molecular states = {}'.format(N))


mol = Mol(ham, dip) # number of single-polariton states 
        
pump = np.linspace(0, 2)
probe = np.linspace(0, 2) 
omega_min = pump.min() 
omega_max = pump.max() 

gamma = np.array([0.1, ] * N)
gamma[0] = 0. # ground state has infinity lifetime 
 

e_idx = [1] 
f_idx = [2] 

R1 = DQC_R1(E, dip, omega1 = pump, omega2 = 2. * probe,\
              tau3=1e-6, g_idx=[0], e_idx=e_idx, f_idx = f_idx, \
              gamma=gamma)  

R2 = DQC_R2(E, dip, omega1 = pump, omega2 = 2. * probe,\
              tau3=1e-6, g_idx=[0], e_idx=e_idx, f_idx = f_idx, \
              gamma=gamma)  

#np.savez('DQC12', R1=R1, R2=R2)  

R1a = DQC_R1(E, dip, omega2 = 2. * pump, omega3 = probe, tau1=1e-6, \
               g_idx=[0], e_idx=e_idx, f_idx = f_idx, gamma=gamma)  
    
R2b = DQC_R2(E, dip, omega2 = 2. * pump, omega3 = probe, tau1=1e-6, \
               g_idx=[0], e_idx=e_idx, f_idx = f_idx, gamma=gamma)  

#np.savez('DQC23', R1=R1a, R2=R2b)  


from scipy import ndimage 

fig, ax = plt.subplots()

signal = np.abs(R2b) 
print(len(pump), len(probe))

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

pump = np.linspace(0, 4, 150)
probe = np.linspace(0, 2, 150)

im = ax.contour(signal.T, levels=levels, cmap=cm.Blues,\
            origin='lower', extent=extent)

ax.set_xlabel(r'$\Omega_2$/eV')
ax.set_ylabel(r'$\Omega_3$/eV')

    