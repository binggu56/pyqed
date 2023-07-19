#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:47:23 2019

@author: bing
"""

import sys 
sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.cavity import Cavity
#from lime.oqs import Oqs, lindblad
#from lime.correlation import correlation_2p_1t as corr

from lime.style import matplot

import numpy as np 
from scipy.sparse import csr_matrix 
import proplot as plt 


#t,corr = np.genfromtxt('cor.dat', unpack=True, dtype=complex)

fig, ax = plt.subplots() 
#ax.plot(t, corr.real)



x = np.linspace(-3,3, 60)
# y = np.linspace(-2,2, 60)

# #X, Y = np.meshgrid(x, y)

# #f = np.sin(-(X + Y)**2)
# f = np.zeros((len(x), len(y)))

# for i in range(len(x)):
#     for j in range(len(y)):
#         f[i,j] = np.sin(x[i]**2 + y[j]**2) 
        
# ax.matshow(x, y, f, diverge=True, vmin=-0.12, vmax=0.12)
ax.plot(x, np.sin(x))
ax.axvline(1, 0, 0.5)
ax.axvline(0.5, 0, 0.2)