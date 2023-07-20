#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:05:24 2019

@author: binggu

@aims: superoperator algebra in Liouville space
"""

from scipy.sparse import identity, kron
import numpy as np

# def superoperator(A, sign=-1):
#     """
#     transform an ordinary operator to the corresponding superoperators
#     sign = 1 anticommutator
#           -1 commutator
#     output: sparse matrix of size n**2
#     """
#     I = identity(A.shape[0])
#     return kron(A, I) + sign * kron(I, A.T.conj())


# def time_convolutionless(initial_state, H0, D, n=2):
#     """
#     generalized time-convolutionless master equation
#     input:
#         H0: system Hamiltonian
#         D: dissipative superoperator
#         n: integer, perturbative order, default 2
#     """
# def dm2vec(rho):
#     """
#     transform a density matrix to a row vector
#     """
#     return np.ravel(rho)

# def inner_product(A, B):
#     """
#     inner product of two operators in Liouville space
#     """
#     return A.conj().dot(B)

def sort(eigvals, eigvecs):

    idx = np.argsort(np.abs(eigvals))

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return eigvals, eigvecs

if __name__ == "__main__":
    from phys import sz, sx, sy

    A = -1j*superoperator(sz, sign=1) + superoperator(sx, sign=1)
    print(A.toarray())

    from scipy.linalg import eig

    print(eigvals(A.toarray()))
