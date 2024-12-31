#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:00:37 2024

NRG bosonic for chain model

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np
from scipy import integrate
from scipy.sparse import lil_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

from pyqed import Cavity, Mol, Composite, dag, SineDVR, pauli


class Boson(Cavity):
    def __init__(self, omega, n=None, ZPE=False):
        self.dim = n
        self.ZPE = ZPE
        self.omega = omega
        self.identity = eye(n)

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

# def pauli():
#     # spin-half matrices
#     sz = np.array([[1.0,0.0],[0.0,-1.0]])

#     sx = np.array([[0.0,1.0],[1.0,0.0]])

#     sy = np.array([[0.0,-1j],[1j,0.0]])

#     s0 = np.identity(2)

#     for _ in [s0, sx, sy, sz]:
#         _ = csr_matrix(_)

#     return s0, sx, sy, sz


class SBM:
    """
    spin-boson model
    """
    def __init__(self, epsilon, Delta, omegac=1):
        """


        Parameters
        ----------
        epsilon : TYPE
            DESCRIPTION.
        Delta : TYPE
            DESCRIPTION.
        omegac : TYPE, optional
            cutoff frequency. The default is 1.

        Returns
        -------
        None.

        """

        self.omegac = omegac

        I, X, Y, Z = pauli()

        self.H = 0.5 * (- epsilon * Z + X * Delta)

    def spectral_density(self, s=1, alpha=1):
        pass

    def discretize(self):
        pass

    def to_chain(self):
        pass

    def HEOM(self):
        pass

    def Redfield(self):
        pass





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
        y = np.logspace(a, 1, nmodes+1, base=2)
    
        print(y)

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

def J(omega, s=1, alpha=1, omegac=1):
    """
    

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    s : TYPE, optional
        DESCRIPTION. The default is 1.
        
        1: ohmic
        < 1: subohmic 
        > 1: superohmic 
        
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    omegac : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return 2 * np.pi * alpha * omegac**(1-s) * omega**s

class NRG:
    """
    NRG bosonic for open quantum systems

    .. math::

        H = -\Delta X + \epsilon Z + \sum_i \omega_i a_i^\dager a_i + Z \sum_i \lambda_i (a_i + a_i^\dagger)

    is mapped to

    .. math::

        H = -\Delta X + \epsilon Z + \sqrt{\eta_0/\pi} Z/2 (b_0+b_0^\dagger) + 
        \sum_{n=0}^\infty \epsilon_n b_n^\dagger b_n + t_n(b_n b_{n+1}^\dagger + H.c.)


    """

    def __init__(self, Himp, L=2.0):
        # self.nsites = len(onsite) + 1

        # self.hopping = hopping
        self.L = L # Lambda for log-discretization
        self.H = Himp
        
        self.nmodes = None 
        
    def add_coupling(self):
        pass
    
    def discretize(self, N, s=1.0, omegac=1, alpha=1):
        # H = -\Delta X + \epsilon Z + \sum_i \xi_i a_i^\dagger a_i + \frac{Z}{2\sqrt{\pi}} \sum_i  \gamma_i (a_i + a_i^\dagger)
        """
        
        H = H_imp + \sqrt{\eta0/\pi} Z/2 (b_0 + b_0^\dagger)

        Refs 
        
        PHYSICAL REVIEW B 71, 045122 s2005d
        
        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        s : TYPE, optional
            DESCRIPTION. The default is 1.
        omegac : TYPE, optional
            DESCRIPTION. The default is 1.
        alpha : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        xi : TYPE
            DESCRIPTION.
        g : TYPE
            DESCRIPTION.

        """
        
        n = np.arange(N)
        self.nmodes = N 
        
        L = self.L 
        
        xi = (s+1)/(s+2) * (1. - L**(-s-2))/(1. - L**(-s-1)) * omegac * L**(-n)
        
        g2 = 2 * np.pi * alpha/(s+1) * omegac**2 * (1 - L**(-s-1))* L**(-n * (s+1)) 
        
        
        # to chain 
        eta0 = np.sum(g2) # \int_0^\infty J(omega) \dif omega 
        
        self.eta0 = eta0
        
        U = np.zeros((N, N))
        
        U[0, :] = np.sqrt(g2)/np.sqrt(eta0)
        

        t = np.zeros(N) # hopping 
        epsilon = np.zeros(N) # onsite 

        epsilon[0] = np.sum(U[0]**2 * xi)
        
        t[0] = np.sum( (xi - epsilon[0])**2 * g2 )/eta0 
        t[0] = np.sqrt(t[0])
        
        U[1] = (xi - epsilon[0]) * U[0]/t[0]
        
        for m in range(1, N-1):
            
            epsilon[m] = np.sum(U[m]**2 * xi)
    
            t[m] = np.sqrt( np.sum( ((xi - epsilon[m])* U[m] -  t[m-1] * U[m-1] )**2) )
            
            U[m+1] = ((xi - epsilon[m]) * U[m] - t[m-1] * U[m-1])/t[m]
        
        return epsilon, t
    
    def run(self):
        
        eta0 = self.eta0 
        I, X, Y, Z = pauli()
        
        # impurity + the first boson site
        nz = 16
        site = Boson(epsilon[0], nz) # the 0th site 
        a = site.annihilate()
        
        # x = dvr.x
        # dvr.v = x**2/ 

        # for n in range(nz):
        H = kron(self.H, eye(nz)) + kron(I, site.buildH())  +  np.sqrt(eta0/np.pi) * kron(Z/2, a + dag(a))
        E, U = eigsh(H, k=6)
    

if __name__=='__main__':
    
    I, X, Y, Z = pauli()
    epsilon = 1
    Delta = 0.1
    H = 0.5 * (epsilon * Z + X * Delta)
    
    # omega = 1
    # mol = Mol(H, X)
    # site = Boson(omega, n=10)
    # site.buildH()
    # a = site.annihilate()
    
    # mol = Composite(mol, site)
    # H0  = mol.getH([X],  [a + dag(a)], g=[0.1])
    
    # E, U = mol.eigenstates(k=6)
    # a = mol.promote(a, subspace='B')
    # a = mol.transform_basis(a)
    
    nrg = NRG(H)
    
    x, g = nrg.discretize(10)

    
    print(x, g)
    

    
    
    # build the overlap matrix
    
    # S = ...