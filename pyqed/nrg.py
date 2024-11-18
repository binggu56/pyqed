#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:00:37 2024

NRG bosonic

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np
from scipy import integrate
from scipy.sparse import lil_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

from pyqed import Cavity, Mol, Composite, dag, SineDVR


class Boson(Cavity):
    def __init__(self, omega, n=None, ZPE=False):
        self.dim = n
        self.ZPE = ZPE
        self.omega = omega
        self.idm = eye(n)

        ###
        self.H = None

    def buildH(self):
        omega = self.omega
        n = self.dim
        if self.ZPE:
            h = lil_matrix((n,n))
            h.setdiag((np.arange(n) + 0.5) * omega)
        else:
            h = lil_matrix((n, n))
            h.setdiag(np.arange(n) * omega)

        self.H = h
        return h

    def annihilate(self):
        n_cav = self.dim
        a = lil_matrix((n_cav, n_cav))
        a.setdiag(np.sqrt(np.arange(1, n_cav)), 1)

        return a.tocsr()

def pauli():
    # spin-half matrices
    sz = np.array([[1.0,0.0],[0.0,-1.0]])

    sx = np.array([[0.0,1.0],[1.0,0.0]])

    sy = np.array([[0.0,-1j],[1j,0.0]])

    s0 = np.identity(2)

    for _ in [s0, sx, sy, sz]:
        _ = csr_matrix(_)

    return s0, sx, sy, sz


class SBM:
    """
    spin-boson model
    """
    def __init__(self, epsilon, Delta):

        I, X, Y, Z = pauli()

        self.H = 0.5 * (- epsilon * Z + X * Delta)

    def spectral_density(self, J):
        pass

    def discretize(self):
        pass

    def to_chain(self):
        pass

    def HEOM(self):
        pass

    def Redfield(self):
        pass


I, X, Y, Z = pauli()
epsilon = 1
Delta = 0.1
H = 0.5 * (- epsilon * Z + X * Delta)

omega = 1
mol = Mol(H, X)
site = Boson(omega, n=10)
site.buildH()
a = site.annihilate()

mol = Composite(mol, site)
H0  = mol.getH([X],  [a + dag(a)], g=[0.1])
# E, U = mol.eigenstates(k=6)
# a = mol.promote(a, subspace='B')
# a = mol.transform_basis(a)

print(a.shape)

t = 0.5
# add a boson site
nz = 16
dvr = SineDVR(-6, 6, nz)
z = dvr.x
for n in range(nz):
    H = H0 + t * kron(I, a + dag(a)) * z[n]
    E, U = eigsh(H, k=6)


# build the overlap matrix

# S = ...



def discretize(J, a, b, nmodes, mesh='log'):
    """
    Discretize a harmonic bath in the range (a, b) by the mean method in Ref. 1.


    Ref:
        [1] PRB 92, 155126 (2015)

    Parameters
    ----------
    J : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    domain : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    x : array
        mode frequecies
    g : array
        coupling strength

    """
    if mesh == 'linear':

        y = np.linspace(a, b, nmodes, endpoint=False)

    elif mesh == 'log':

        if a == 0: a += 1e-3
        y = np.logspace(a, np.log10(b), nmodes+1)

    x = np.zeros(nmodes)
    g = np.zeros(nmodes)


    for n in range(nmodes):
         g[n] = integrate.quad(J, y[n], y[n+1])[0]
         x[n] = integrate.quad(lambda x: x * J(x), y[n], y[n+1])[0]
         x[n] /= g[n]

    # last interval from y[-1] to b
    # g[-1] = integrate.quad(J, y[-1], b)[0]
    # x[-1] = integrate.quad(lambda x: x * J(x), y[-1], b)[0]/g[-1]

    return x, np.sqrt(g)

class NRG:
    """
    NRG bosonic for open quantum systems

    .. math::

        H = -\Delta X + \epsilon Z + \sum_i \omega_i a_i^\dager a_i + Z \sum_i \lambda_i (a_i + a_i^\dagger)

    is mapped to

    .. math::

        H = -\Delta X + \epsilon Z + \sqrt{\eta_0/\pi} Z(b_0+b_0^\dagger) + \sum_{n=0}^\infty \epsilon_n b_n^\dagger b_n + t_n(b_n b_{n+1}^\dagger + H.c.)


    """

    def __init__(self, Himp, onsite, hopping):
        self.nsites = len(onsite) + 1

        self.hopping = hopping

    def add_coupling(self):
        pass