#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:10:08 2023

@author: bing
"""

import scipy
import numpy as np

from pyqed import dag, sort

def eig(a, norm=False):
    """
    compute eigenvalues and eigenvectors of a non-Hermitian matrix

    .. math::
        
        A | R_n \rangle = E_n |R_n \rangle
    
    under the normalization 
    
    .. math::
        
        \langle L_m | R_n \rangle = \delta_{mn}
    
    The left eigenvectors are computed by inverse the right eigenvectors. 
    
    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    norm : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    evals1 : TYPE
        DESCRIPTION.
    U1 : TYPE
        right eigenvectors :math:`U[a, n] = \langle  a | R_n \rangle`
        a is the primitive basis used to construct the a matrix. 
    U2 : TYPE
        left eigenvectors :math:`U[n, a] = \langle  L_n | a \rangle` 

    """
    
    evals1, U1 = scipy.linalg.eig(a)
    evals1, U1 = sort(evals1, U1)

    # U2 = U1.conj()
    # eigvals2, U2 = eig(dag(L).todense())
    # eigvals2, U2 = sort(eigvals2, U2)
    

    U2 = scipy.linalg.inv(U1)
    
    if norm:
        norm = np.array([np.vdot(U2[:,n], U1[:,n]) for n in range(len(evals1))])
    # norm = U2 @ U1 
    # print('normalization \n', norm)

    return evals1, U1, U2

def model(x):
    """
    1D model from Chem Sci 2020, 11, 9827-9835

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from pyqed import au2ev 
    
    nx = len(x)
    nstates = 3
    E1 = 1./au2ev 
    k0 = k1 = 3 / au2ev
    # Rc = 
    alpha = 0.8
    D = -1 / au2ev 
    v2 = 2 / au2ev 
    R1 = 0.25
    v12 = 0.1 / au2ev 
    
    v = np.zeros((nx, nstates, nstates), dtype=complex)
    v[:, 0, 0] = 0.5 * k0 * x**2
    v[:, 1, 1] = 0.5 * k1 * (x - R1)**2 + E1
    
    v[:, 2, 2] = D +  (v2 - D) * np.exp(- alpha * x)
    Rc = np.min(np.abs(v[:, 2, 2] - v[:, 1, 1]))
    
    
    v[:, 1, 2] = v12 * np.exp(-(x - Rc)**2)
    v[:, 2, 1] = v[:, 1, 2]

    
    gamma1 = (v[:, 1, 1] - v[:, 0, 0]) * 0.1
    gamma1[gamma1 < 0] = 0
    
    gamma2 = (v[:, 2, 2] - v[:, 0, 0]) * 0.2
    gamma2[gamma2 < 0] = 0
    
    v[:, 1, 1] -= 1j * gamma1 
    v[:, 2, 2] -= 1j * gamma2
    
    return v


def diabatic_to_adiabatic(v, ndim=1):
    if ndim == 1:
        
        nx, nstates, _ = v.shape
        va = np.zeros((nx, nstates), dtype=complex)
        for n in range(nx):
            w, ur, ul = eig(v[n])
            va[n, :] = w
    
        return va
    

if __name__ == '__main__':
    a = np.zeros((2,2),dtype=complex)
    a[1, 0] = 1 
    a[0, 1] = 1+1j
    
    eig(a)
    x = np.linspace(-1, 2, 100) 
    v = model(x)

    import proplot as plt
    
    fig, (ax, ax2) = plt.subplots(ncols=2)
    ax.plot(x, v[:, 1, 1].real)
    ax.plot(x, v[:, 0, 0].real)    
    ax.plot(x, v[:, 2, 2].real)
    
    va = diabatic_to_adiabatic(v)

    # fig, ax = plt.subplots()
    ax2.plot(x, va[:, 1].real)
    ax2.plot(x, va[:, 2].real)  
    ax2.plot(x, va[:, 0].real)    
    # ax2.format(ylim=(0.03, .045))

    # ax.plot(x, v[:, 2, 2])
    
    