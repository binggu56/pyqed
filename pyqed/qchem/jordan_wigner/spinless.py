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


from pyqed import dag, anticomm 

sigma = csr_matrix(np.array([[0, 1], [0, 0]]))
sigma_z = csr_matrix(np.array([[1, 0], [0, -1]]))
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
        return kron(a[0],a[1])
    else:
        return kron(a[0], nested_kronecker_product(a[1:]))


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

def bravyi_kitaev_transform():
    pass

lattice_length = L = 4
a = []
for i in range(lattice_length):
    a.append(jordan_wigner_transform(i, lattice_length))

ad = [dag(_a) for _a in a]

# print(a[0])
# assert anticomm(a[1], ad[1]) == identity(2**L)


class FermionChain:
        
    """    
    exact diagonalization of 
    
    spinless open fermion chain
    
    by Jordan-Wigner transformation 
    
    .. math::
        
        H = \sum_{<rs>} (c_r^\dag c_s + c†scr−γ(c†rc†s+cscr))−2λ∑rc†rcr,
    
    where r and s indicate neighbors on the chain. 
    
    Electron interactions can be included in the Hamiltonian easily.
    
    """
    def __init__(self, J, L=2, U=0):
        self.L = self.nsites = L
        self.tunneling = J
        self.coulomb = U
    
    def buildH(self):
        pass

class Hubbard:
    """
    exact diagonalization of 
    
    spinless open fermion chain
    
    by Jordan-Wigner transformation 
    
    .. math::
        
        H = \sum_{<r,s>} -J(c_r^\dag c_s + H.c.) + U \sum_r n_{r\alpha} n_{r\beta} −\mu \sum_r c^dag_r c_r
    """
    def __init__(self, J=1, U=0, nsites=2, mu=0):
        self.tunneling = self.J = J
        self.nsites = nsites
        self.coulomb = U
        self.chemical_potential = self.mu = 0
        
        self.H = None
        self.dim = 2**nsites
        
    def buildH(self):
        

        L = self.nsites 
        a = []
        for i in range(L):
            a.append(jordan_wigner_transform(i, L))

        ad = [dag(_a) for _a in a]
        
        H = 0
        N = 0 # number operator
        for i in range(L - 1):
            # H -= self.J * (a[i].T.dot(a[i+1]) + a[i].dot(a[i+1].T))
            H -= self.J * (ad[i] @ a[i+1] + ad[i+1] @ a[i])
        
        for i in range(L):
            N += ad[i] @ a[i]
            H -= self.mu * (ad[i] @ a[i])
        
        self.H = H
        self.number_operator = N
        return H    
    
    def electron_number(self, psi):
        return dag(psi) @ self.number_operator @ psi
        

    def exact_diagonalization(self, k=6):
        if k < self.dim:
            return eigsh(self.H, k=k, which='SA')
        else:
            return np.linalg.eigh(self.H.toarray())
        
def hamiltonian(gam, lam, a, lattice_length):
    H = 0
    for i in range(lattice_length - 1):
        H += a[i].T.dot(a[i+1]) - a[i].dot(a[i+1].T)
        H -= gam*(a[i].T.dot(a[i+1].T) - a[i].dot(a[i+1]))
        
    for i in range(lattice_length):
        H -= 2*lam*(a[i].dot(a[i].T))
    return H


if __name__=='__main__':
    
    import time 
    from pyqed import comm
    start = time.time()
    gam, lam =1, 1
    # H = hamiltonian(gam, lam, a, lattice_length)
    # eigenvalues = np.linalg.eig(H)[0]
    
    
    
    hubbard = Hubbard(J=1, nsites=31)
    H = hubbard.buildH()
    
    # print(comm(H, hubbard.number_operator))
    
    # eigenvalues, u = eigsh(H, k=6)
    eigenvalues, u = hubbard.exact_diagonalization(k=6)
    
    for i in range(4):
        print(hubbard.electron_number(u[:,i]))
        
    print(sorted(eigenvalues))
    
    import proplot as plt
    fig, ax = plt.subplots()
    ax.plot(eigenvalues)
    # from pyqed import level_scheme
    
    # level_scheme(eigenvalues)
    
    print('time = ', time.time() - start) 
