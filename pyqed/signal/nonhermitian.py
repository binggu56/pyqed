#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:47:45 2020

Generalized sum-over-states for non-Hermitian systems 
@author: bing
"""

import numpy as np 
import proplot as plt 
from numba import jit 
from scipy.linalg import eig

import sys
sys.path.append(r'C:\Users\Bing\Google Drive\lime')
sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.phys import lorentzian, dag 
from lime.units import au2ev


def linear_absorption(omegas, ham, dip):
    """
    Note that the eigenvectors of the liouvillian L and L^\dag have to 
    be ordered in parallel!

    Parameters
    ----------
    omegas : TYPE
        DESCRIPTION.
    ham : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    rho0 : 2d array
        initial density matrix 
    c_ops : TYPE, optional
        DESCRIPTION. The default is [].
    ntrans : int, optional
        Number of transitions to be computed. The default is 1.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    """    
    
    ntrans = np.size(ham, 0)
    
    eigvals1, U1 = eig(ham)
    eigvals1, U1 = sort(eigvals1, U1)

    eigvals2, U2 = eig(dag(ham))
    eigvals2, U2 = sort(eigvals2, U2)
    
    norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(ntrans)]

    signal = np.zeros(len(omegas), dtype=complex)  
    
    tmp = [np.vdot(dip, U1[:,n]) * np.vdot(U2[:,n], dip)/norm[n] \
           for n in range(ntrans)] 
    
    for j in range(len(omegas)):
        omega = omegas[j] 
        signal[j] += sum(tmp / (omega - eigvals1))   
    
    
    return  -2. * signal.imag, eigvals1, dag(U1).dot(dip)

def sort(eigvals, eigvecs):
    
    idx = np.argsort(eigvals)
    
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    
    return eigvals, eigvecs
