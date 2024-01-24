#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:32:45 2023

Many-body theory of electron-phonon problems

Refs:
    
    Liu et al PRB 2021

@author: bing
"""

import numpy as np

from pyqed import pauli, interval
from numpy import pi
from numpy.linalg import inv

s0, s1, s2, s3 = pauli()

# 1D BZ, energy-momentum space for electrons
nk = 32
ne = 32
p = np.linspace(-pi, pi, nk)
e = np.linspace(-2, 2, ne)

g = np.zeros((ne, nk, 2, 2), dtype=complex)
g0 = np.zeros((ne, nk, 2, 2), dtype=complex)
g0inv = np.zeros((ne, nk, 2, 2), dtype=complex)
ginv = np.zeros((ne, nk, 2, 2), dtype=complex)



# wavevector and frequency for phonons
nq = 32
nomega = 32
q = np.linspace(-pi, pi, nq)
omega = np.linspace(-1, 1, nomega)
d0 = np.zeros((nomega, nq), dtype=complex)
domega = interval(omega)
dq = interval(q)





def gf0(e, p, mu=0, eta=1e-3):
    """
    bare electron time-ordered GF

    Parameters
    ----------
    e : TYPE
        DESCRIPTION.
    p : TYPE
        (quasi)momentum
    mu : TYPE, optional
        chemical potential. The default is 0.
    eta : TYPE, optional
        DESCRIPTION. The default is 1e-3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.linalg.inv( (e + 1j * eta * np.sign(e - mu)) * s0 - band(p)  * s3)
    
def gf0_ph(omega, q, eta=1e-3):
    """
    bare photon time-ordered GF

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    q : TYPE
        wavevector
    eta : TYPE, optional
        DESCRIPTION. The default is 1e-3.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    w = dispersion(q)
    d = (1./(omega - w + 1j*eta) - 1./(omega + w - 1j*eta))

    return d


def band(k):
    vf = 1.
    # return vf * np.abs(k)
    return - np.cos(k)

def dispersion(p):
    """
    electronic band structure

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # return abs(p)
    return 0.5

def gf(p):
    pass

def vertex(k, l, i, j, g, eta=1e-3):
    """
    electron-phonon vertex 

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    e : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    eta : TYPE, optional
        DESCRIPTION. The default is 1e-3.

    Returns
    -------
    v :  2 x 2 x np x nq
        vertex function.

    """
    g1 = g[i+k, j+l]
    g1inv = np.linalg.inv(g1)
    
    g2 = g[i,j]
    
    g2inv = np.linalg.inv(g2)
    
    e1 = band(p[j]+q[l])
    e2 = band(p[j])
    
    v = omega[k] * (g1inv @ s3 - s3 @ g2inv)/(omega[k]**2 - (e1 - e2)**2)
    
    v+= (e1 - e2) * (g1inv @ s0 - s0 @ g2inv)/((omega[k] + 1j * eta)**2 - (e1 - e2)**2)

    return v 
    


coupling_strength = 1

for k in range(nomega):
    for l in range(nq):
        d0[k, l] = gf0_ph(omega[k], q[l])

for i in range(ne):
    for j in range(nk):
        g0[i, j] = gf0(e[i], p[j])

g = g0.copy()

for i in range(ne):
    for j in range(nk):
        # g0inv[i, j] = inv(g0[i, j])
        
        tmp = 0
        for k in range(nomega-i):
            for l in range(nq-j):
                 tmp +=  g[i+k, j+l] @ vertex(k, l, i, j, g) * d0[k, l] 
                
        ginv[i, j] = inv(g[i, j]) - 1j * coupling_strength**2 * s3 @ tmp * domega * dq/(2*pi)**2


for i in range(ne):
    for j in range(nk):
        g[i, j] = inv(ginv[i, j])

import proplot as plt
# ax.plot(g[:, 0, 0, 0])
# for i in range(16):
fig, ax = plt.subplots()
ax.imshow(g[:, :, 0, 0].imag, cmap='spectral_r')

fig, ax = plt.subplots()
ax.imshow(g0[:, :, 0, 0].imag, cmap='spectral_r')
# ax.(np.imag(g[:, -1, 0, 0]))
