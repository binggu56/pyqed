#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:36:05 2023

@author: bing
"""
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, identity

def x2(size, mass=1, omega=1):
    """
    Matrix elements of the :math:`\langle n | x^2 | m\rangle` operator in the 
    eigenstates space

    Parameters
    ----------
    size : int
        size of the Fock space.
    mass : TYPE, optional
        DESCRIPTION. The default is 1.
    omega : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    
    xn = lil_matrix((size, size))

    # diagonals
    n = np.arange(size)  
    xn.setdiag(2 * n + 1, 0)

    
    n = np.linspace(2, size)
    xn.setdiag(np.sqrt(n * (n-1)), 2)
    
    # lower off-diagonal
    n = np.linspace(0, size-2)
    xn.setdiag(np.sqrt((n+1) * (n+2)), -2)
    
    return xn.tocsr() /2./mass/omega 

def p2(size, mass=1, omega=1):
    
    
    xn = lil_matrix((size, size))

    # diagonals
    n = np.arange(size)  
    xn.setdiag(2 * n + 1, 0)

    
    n = np.linspace(2, size)
    xn.setdiag(np.sqrt(n * (n-1)), 2)
    
    # lower off-diagonal
    n = np.linspace(0, size-2)
    xn.setdiag(np.sqrt((n+1) * (n+2)), -2)
    
    return - xn.tocsr() * mass * omega/2 


print(x2(4))