# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:59:19 2019

@author: Bing

@ G2 optical coherence
"""

import sys 
sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.cavity import Cavity
from lime.oqs import Oqs, lindblad
import numba 
from lime.correlation import correlation_3p_1t as corr
from lime.phys import dag

import numpy as np 
from scipy.sparse import csr_matrix, lil_matrix

def fock_dm(N, i):
    dm = np.zeros((N,N), dtype=complex)
    dm[i,i] = 1.0 
    return csr_matrix(dm)

def thermal(N, n_th):
    """
    thermal wavefunction 

    Parameters
    ----------
    N : int
        Size of the Hilbert space. 
    n_th : int 
        bath temperature in terms of excitation number. 

    Returns
    -------
    psi: thermal wavefunction 

    """
    psi = np.array([np.exp(-n/2./n_th) for n in range(N)]) 
    
    return psi 

def thermal_dm(N, n_th):
    """

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    a : float
        thermal energy in terms of excitation energies  

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dm = lil_matrix((N,N))
    w = np.array([np.exp(-n/n_th) for n in range(N)])
    w /= np.sum(w) 
    
    dm.setdiag(w)
    return dm.tocsr() 

N = 10
cav = Cavity(1.0, N)

H = cav.hamiltonian
H = csr_matrix(H)

kappa = 0.25

n_th = 2.0  # bath temperature in terms of excitation number
a =  cav.get_annihilate()
c_ops = [np.sqrt(kappa * (1 + n_th)) * a, np.sqrt(kappa * n_th) * dag(a)]



num_op = cav.get_num()

rho0 = thermal_dm(N, 2)

print(rho0)

tlist = np.linspace(0, 10, 200)

corr(H, rho0, [dag(a), num_op, a], c_ops, tlist, lindblad)

t,corr = np.genfromtxt('cor.dat', unpack=True, dtype=complex)


import matplotlib.pyplot as plt 

fig, ax = plt.subplots() 
ax.plot(t, corr.real)
plt.show()



