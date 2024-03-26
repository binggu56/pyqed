#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:51:21 2024


https://peterwittek.com/the-jordan-wigner-transformation-in-python.html

@author: Bing Gu (gubing@westlake.edu.cn)


"""

import numpy as np
from scipy.sparse import kron, kronsum, csr_matrix, identity
from scipy.sparse.linalg import eigsh

sigma = csr_matrix(np.array([[0, 1], [0, 0]]))
sigma_z = np.array([[1, 0], [0, -1]])
I = identity(2)

def jordan_wigner(j, L):
    """
    Jordan Wigner transform 

    Parameters
    ----------
    j : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    operators = []
    for k in range(j):
        operators.append(sigma_z)
    
    operators.append(sigma)
    
    for k in range(L-j-1):
        operators.append(I)
    return -nested_kronecker_product(operators)


def nested_kronecker_product(a):
    if len(a) == 2:
        return np.kron(a[0],a[1])
    else:
        return np.kron(a[0], nested_kronecker_product(a[1:]))


def jordan_wigner_transform(j, lattice_length):
    sigma = np.array([[0, 1], [0, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    operators = []
    for k in range(j):
        operators.append(sigma_z)
    operators.append(sigma)
    for k in range(lattice_length-j-1):
        operators.append(I)
    return -nested_kronecker_product(operators)


lattice_length = 4
a = []
for i in range(lattice_length):
    a.append(jordan_wigner_transform(i, lattice_length))

class FermionChain:
    pass
        
"""    
exact diagonalization of 

spinless open fermion chain

by Jordan-Wigner transformation 

.. math::
    
    H = \sum_{<rs>} (c_r^\dag c_s + c†scr−γ(c†rc†s+cscr))−2λ∑rc†rcr,

where r and s indicate neighbors on the chain. 

Electron interactions can be included in the Hamiltonian easily.

"""

def hamiltonian(gam, lam, a, lattice_length):
    H = 0
    for i in range(lattice_length - 1):
        H += a[i].T.dot(a[i+1]) - a[i].dot(a[i+1].T)
        H -= gam*(a[i].T.dot(a[i+1].T) - a[i].dot(a[i+1]))
    for i in range(lattice_length):
        H -= 2*lam*(a[i].dot(a[i].T))
    return H

import time 

start = time.time()
gam, lam =1, 1
H = hamiltonian(gam, lam, a, lattice_length)
# eigenvalues = np.linalg.eig(H)[0]
eigenvalues = eigsh(H, k=6)[0]

print(sorted(eigenvalues))
print('time = ', time.time() - start) 
