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

from pyqed import Cavity, Mol, Composite, dag, SineDVR, pauli, sort, householder

from pyqed.phys import eigh
from opt_einsum import contract

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
    
    def create(self):
        n_cav = self.dim
        c = lil_matrix((n_cav, n_cav))
        c.setdiag(np.sqrt(np.arange(1, n_cav)), -1)
        return c.tocsr()

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

    def to_wilson_chain(self):
        pass

    def HEOM(self):
        pass

    def Redfield(self):
        pass





# def discretize(J, a, b, nmodes, mesh='log'):
#     """
#     Discretize a harmonic bath in the range (a, b) by the mean method in Ref. 1.


#     Ref:
#         [1] PRB 92, 155126 (2015)

#     Parameters
#     ----------
#     J : TYPE
#         DESCRIPTION.
#     n : TYPE
#         DESCRIPTION.
#     domain : TYPE, optional
#         DESCRIPTION. The default is None.

#     Returns
#     -------
#     x : array
#         mode frequecies
#     g : array
#         coupling strength

#     """
#     if mesh == 'linear':

#         y = np.linspace(a, b, nmodes, endpoint=False)

#     elif mesh == 'log':

#         if a == 0: a += 1e-3
#         y = np.logspace(a, 1, nmodes+1, base=2)


#     x = np.zeros(nmodes)
#     g = np.zeros(nmodes)


#     for n in range(nmodes):
#          g[n] = integrate.quad(J, y[n], y[n+1])[0]
#          x[n] = integrate.quad(lambda x: x * J(x), y[n], y[n+1])[0]
#          x[n] /= g[n]

#     # last interval from y[-1] to b
#     # g[-1] = integrate.quad(J, y[-1], b)[0]
#     # x[-1] = integrate.quad(lambda x: x * J(x), y[-1], b)[0]/g[-1]

#     return x, np.sqrt(g)

# def ohmic(omega, s=1, alpha=1, omegac=1):
#     """
#     ohmic spectral density

#     .. math::

#         J(\omega) = 2\pi \alpha \omega^s e^{-\omega/\omega_c}

#     Parameters
#     ----------
#     omega : TYPE
#         DESCRIPTION.
#     s : TYPE, optional
#         DESCRIPTION. The default is 1.

#         1: ohmic
#         < 1: subohmic
#         > 1: superohmic

#     alpha : TYPE, optional
#         DESCRIPTION. The default is 1.
#     omegac : TYPE, optional
#         DESCRIPTION. The default is 1.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     return 2 * np.pi * alpha * omegac**(1-s) * omega**s

class NRGSpin:
    def __init__(self, L, sites):
        self.L = L

    def run(self):
        pass


class AIM:
    def __init__(self, energy=1, U=0):
        # Anderson impurity model
        pass

class NRG:
    """
    NRG bosonic for open quantum systems

    .. math::

        H = -\Delta X + \epsilon Z + \sum_i \omega_i a_i^\dager a_i + Z \sum_i \lambda_i (a_i + a_i^\dagger)

    is mapped to

    .. math::

        H = -\Delta X + \epsilon Z + \sqrt{\eta_0/\pi} Z/2 (b_0+b_0^\dagger) +
        \sum_{n=0}^\infty \epsilon_n b_n^\dagger b_n + t_n(b_n b_{n+1}^\dagger + H.c.)

    where X, Y, Z are spin-half operators.

    The spectral density is defined as

    .. math::
        J(\omega) = \pi \sum_i \lambda_i^2 \delta(\omega - \omega_i)

    for `math`: \omega \in [0, \omega_c]


    """

    def __init__(self, Himp, alpha, L=2.0, s=1, omegac=1):
        """


        Parameters
        ----------
        Himp : TYPE
            DESCRIPTION.
        L : TYPE, optional
            DESCRIPTION. The default is 2.0.
        s : TYPE, optional
            DESCRIPTION. The default is 1 corresponding to the Ohmic bath.
        omegac : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        # self.nsites = len(onsite) + 1

        # self.hopping = hopping
        self.L = L # Lambda for log-discretization
        self.H = Himp

        self.nmodes = None
        assert(s > -1) # s has to be larger than -1
        self.s = s
        self.omegac = omegac
        self.alpha = alpha


        self.xi = None
        self.g = None
        ### Wilson chain params
        self.onsite = None
        self.hopping = None
        
        self.t0 = None

    def add_coupling(self):
        pass

    def oscillator_energy(self, n):
        """
        n-th mode energy in the log-discretization (n = 0, 1, ...)

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        L = self.L
        s = self.s
        omegac = self.omegac

        return (s+1)/(s+2) * (1. - L**(-s-2))/(1. - L**(-s-1)) * omegac * L**(-n)


    def discretize(self, N):
        # H = -\Delta X + \epsilon Z + \sum_i \xi_i a_i^\dagger a_i + \frac{Z}{2\sqrt{\pi}} \sum_i  \gamma_i (a_i + a_i^\dagger)
        """
        logrithmic discretization

        .. math::

            H = H_imp + \sqrt{\eta0/\pi} Z (b_0 + b_0^\dagger)

        Refs:
            PHYSICAL REVIEW B 71, 045122 s2005d

        Parameters
        ----------
        N : TYPE
            number of modes.
        s : TYPE, optional
            exponent in spectral density. The default is 1.
        omegac : TYPE, optional
            cutoff frequency. The default is 1.
        alpha : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        xi : TYPE
            DESCRIPTION.
        g : TYPE
            DESCRIPTION.

        """
        
        nmax = N
        n = np.arange(nmax)

        self.nmodes = N

        L = self.L
        alpha = self.alpha
        s = self.s
        omegac = self.omegac


        # star configuration
        xi = (s+1)/(s+2) * (1. - L**(-s-2))/(1. - L**(-s-1)) * omegac * L**(-n)

        g2 = 2 * np.pi * alpha/(s+1) * omegac**2 * (1 - L**(-s-1))* L**(-n * (s+1))
        g = np.sqrt(g2)
        
        self.g = g
        self.xi = xi 
        
        d, c, U = star_to_chain(xi, g)
        
        epsilon = d[1:]
        t = c[1:]
            
            
        
        # to chain
        # eta0 = sum(g2) # \int_0^{\omega_c} J(omega) \dif omega
        
        # print(c[0], np.sqrt(eta0) )
        # eta0 = c[0]

        self.t0 = c[0]


        # U = np.zeros((N, nmax))

        # U[0] = g/np.sqrt(eta0)

        # t = np.zeros(N) # hopping
        # epsilon = np.zeros(N, dtype=float) # onsite

        # epsilon[0] = sum(U[0]**2 * xi)

        # t[0] = sum( (xi - epsilon[0])**2 * g2 )/eta0
        # t[0] = np.sqrt(t[0])

        # U[1] = (xi - epsilon[0]) * U[0]/t[0]

        # for m in range(1, N-1):

        #     epsilon[m] = sum(U[m]**2 * xi)


        #     t[m] = np.sqrt( sum( ((xi - epsilon[m]) * U[m] -  t[m-1] * U[m-1] )**2) )


        #     U[m+1] = ((xi - epsilon[m]) * U[m] - t[m-1] * U[m-1])/t[m]
            
        #     print(sum(U[m]**2))


        # epsilon[N-1] = sum(U[N-1]**2 * xi)
        # t[N-1] = np.sqrt( sum( ((xi - epsilon[N-1])* U[N-1] -  t[N-2] * U[N-2])**2) )

        self.onsite = epsilon
        self.hopping = t

        return epsilon, t

    # def to_wilson_chain(self):
    #     pass



    def run(self, N, nb=60, D=10, chain=True):
        """


        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        nb : TYPE, optional
            DESCRIPTION. The default is 60.
        D: retained eigenstates

        Returns
        -------
        None.

        """


        I, X, Y, Z = pauli()

        # if self.onsite is None:
        #     epsilon, t = self.discretize(N)

        # else:

        epsilon = self.onsite
        t = self.hopping

        # N = self.nmodes

        L = self.L
        # impurity + the first boson site

        # nz = 60 # truncation in the Fock space

        e_tot = np.zeros((N, D))

        site = Boson(1, nb) # the 0th site
        Hb = site.buildH()

        a = site.annihilate()
        # ad = site.create()
        
        Isite = site.identity


        # x = dvr.x
        # dvr.v = x**2/

        if chain:
            
            # for n in range(nz):
            H = kron(self.H, eye(nb)) + kron(I, epsilon[0] * Hb)  + \
                self.t0 * np.sqrt(1/np.pi) * kron(Z/2., a + dag(a))
    
            E, U = eigh(H, k=D)
            E = E - E[0]
            
            
            e_tot[0] = E
            
            I = eye(D)
    
    
            a_tilde = dag(U) @ kron(eye(2), a) @ U
            # ad_tilde = dag(U) @ kron(eye(2), ad) @ U
            num =  U[:,0].conj().T @ kron(eye(2), Hb) @ U[:,0]
            
            Z_tilde = dag(U) @ kron(Z, eye(nb)) @ U
            


            
            for n in range(N-1):
    
    
                H = L * kron(np.diag(E), eye(nb)) + L**(n+1) * (\
                    kron(I, epsilon[n+1] * Hb) + t[n] * (kron(a_tilde, dag(a)) + \
                                                         kron(dag(a_tilde), a)))
    
                E, U = eigh(H, k=D)
                E = E - E[0]
    
                a_tilde = dag(U) @ kron(I, a) @ U
                Z_tilde = dag(U) @ kron(Z_tilde, site.identity) @ U
                
                # ad_tilde = dag(U) @ kron(I, ad) @ U
                
                # num =  contract('i,ij,j->', U[:,0].conj(), kron(I, dag(a) @ a), U[:,0])
                
                num =  U[:,0].conj().T @ kron(I, Hb) @ U[:,0]
                Sz = U[:,0].conj().T @ kron(Z_tilde, eye(nb)) @ U[:,0]
                
                print(num, Sz)
                
    
                e_tot[n+1] = E
    
            return e_tot
        
        else:
            # star 
            
            xi = self.xi 
            g = self.g 
            
            # for n in range(nz):
            H = kron(self.H, eye(nb)) + kron(I, xi[0] * Hb)  + \
                g[0] * np.sqrt(1/np.pi) * kron(Z/2., a + dag(a))
    
            E, U = eigh(H, k=D)
            E = E - E[0]
            
            e_tot[0] = E
            
            I = eye(D)
    
    
            a_tilde = dag(U) @ kron(eye(2), a) @ U
            # ad_tilde = dag(U) @ kron(eye(2), ad) @ U
            num =  U[:,0].conj().T @ kron(eye(2), Hb) @ U[:,0]
            
            Z_tilde = dag(U) @ kron(Z, eye(nb)) @ U
            
            for n in range(N-1):
    
    
                H = L * kron(np.diag(E), eye(nb)) + L**(n+1) * (\
                    kron(I, xi[n+1] * Hb) + g[n+1]/2./np.sqrt(np.pi) * (kron(Z_tilde, dag(a) + a)))
    
                E, U = eigh(H, k=D)
                E = E - E[0]
    

                
                # ad_tilde = dag(U) @ kron(I, ad) @ U
                
                # num =  contract('i,ij,j->', U[:,0].conj(), kron(I, dag(a) @ a), U[:,0])
                
                num =  U[:,0].conj().T @ kron(I, Hb) @ U[:,0]
                Sz = U[:,0].conj().T @ kron(Z_tilde, eye(nb)) @ U[:,0]
                
                a_tilde = dag(U) @ kron(I, a) @ U
                Z_tilde = dag(U) @ kron(Z_tilde, site.identity) @ U
                
                print(num, Sz)
                
    
                e_tot[n+1] = E
    
            return e_tot
            


def star_to_chain(xi, g):
    """
    transform a star configuration to a Wilson chain model by Householder method
    
    .. math::
        
        H = \sum_i \omega_i a_i^\dagger a_i + S \otimes g_i (a_i+a_i^\dagger) 
    
    where S is an system operator. 
    
    Parameters
    ----------
    xi : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        mode frequencies.
    c : TYPE
        hopping. the first element is the impurity to the first site
    U : TYPE
        mode transformation matrix 

    """

    N = len(xi)
    
    A = np.zeros((N+1, N+1))
    for n in range(N): 
        A[n+1, n+1] = xi[n]
        
    A[0, 1:] = g
    A[1:, 0] = g 

    d, c, U = householder(A)
            
    U = U[1:, 1:]
    
    return d, c, U


# import numpy as np
# from hamiltonian_not_diagonal import form_not_diagonal_hamiltonian
class XYZ:
    def __init__(self, L=12, L0=4, Jx=1, Jy=1, Jz=1):
        """
        NRG for Heisenberg XYZ model

        .. math::
            H = \sum_{<i,j>} -J_z Z_i Z_j - J_x X_i X_j - J_y Y_i Y_j

        where X, Y, Z are spin-half operators.

        Ref:

        https://quantum-spin-systems.readthedocs.io/en/latest/chapter2_eng.html#numerical-renormalization-group-nrg-method

        Parameters
        ----------
        L : TYPE, optional
            chain length. The default is 12.
        L0 : int, optional
            initial block size. The default is 4.
        Jx : TYPE, optional
            DESCRIPTION. The default is 1.
        Jy : TYPE, optional
            DESCRIPTION. The default is 1.
        Jz : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.L = L
        self.L0 = L0
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

    def run(self):
        _XYZ_NRG(self.L, self.L0, self.Jx, self.Jy, self.Jz)



def _XYZ_NRG(L=12, L0=4, JX=1, JY=1, JZ=1):
    """
    NRG for Heisenberg XYZ model

    .. math::
        H = \sum_{<i,j>} -J_z Z_i Z_j - J_x X_i X_j - J_y Y_i Y_j

    where X, Y, Z are spin-half operators.

    Ref:

    https://quantum-spin-systems.readthedocs.io/en/latest/chapter2_eng.html#numerical-renormalization-group-nrg-method

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    NUMBER_OF_SPINS_APPROXIMATION = 4

    SX = 0.5 * np.array([(0, 1), (1, 0)])
    SY = 0.5 * np.array([(0, -1j), (1j, 0)])
    SZ = 0.5 * np.array([(1, 0), (0, -1)])


    # %% Create Hamiltonian m*m by the iterative way

    hamiltonian = -JZ * np.kron(SZ, SZ) - JX * np.kron(SX, SX) - JY * np.kron(SY, SY)

    for i in range(NUMBER_OF_SPINS_APPROXIMATION - 2):
        i = i + 1
        sx_tilde = np.kron(np.eye(2 ** i), SX)
        sy_tilde = np.kron(np.eye(2 ** i), SY)
        sz_tilde = np.kron(np.eye(2 ** i), SZ)

        hamiltonian = np.kron(hamiltonian, np.eye(2)) - JZ * np.kron(sz_tilde, SZ) - \
            JX * np.kron(sx_tilde, SX) - JY * np.kron(sy_tilde, SY)

    # % tilde operators
    sx_tilde = np.kron(np.eye(2), sx_tilde)
    sy_tilde = np.kron(np.eye(2), sy_tilde)
    sz_tilde = np.kron(np.eye(2), sz_tilde)

    # %% Normal Renormalization Group

    for i in range(NUMBER_OF_SPINS_APPROXIMATION+2, L+2):

        [eigen_values, eigen_vectors] = np.linalg.eigh(hamiltonian)

        hamiltonian = eigen_vectors.T @ hamiltonian @ eigen_vectors

        hamiltonian = hamiltonian[:2**NUMBER_OF_SPINS_APPROXIMATION,
                                  :2**NUMBER_OF_SPINS_APPROXIMATION]

        eigen_vectors = eigen_vectors[:2**NUMBER_OF_SPINS_APPROXIMATION,
                                      :2**NUMBER_OF_SPINS_APPROXIMATION]

        sx_tilde = eigen_vectors.T @ sx_tilde @ eigen_vectors
        sy_tilde = eigen_vectors.T @ sy_tilde @ eigen_vectors
        sz_tilde = eigen_vectors.T @ sz_tilde @ eigen_vectors

        hamiltonian = np.kron(hamiltonian, np.eye(2)) - JZ * np.kron(sz_tilde, SZ) - \
            JX * np.kron(sx_tilde, SX) - JY * np.kron(sy_tilde, SY)

    return np.linalg.eigh(hamiltonian)


class TDNRG(NRG):
    pass

# %% Compare with certain results

# [eigen_values_nrg_certain, eigen_vectors_nrg_certain] = \
#     np.linalg.eigh(form_not_diagonal_hamiltonian(1, NUMBER_OF_SPINS_FINAL)[0])

if __name__=='__main__':

    I, X, Y, Z = pauli()
    epsilon = 0.
    Delta = 0.01
    H = 0.5 * (epsilon * Z - X * Delta)

    nrg = NRG(H, s=0.8, L=2., alpha=0.05, omegac=1)

    eps, t = nrg.discretize(80)

    e_tot = nrg.run(40, nb=32, D=40, chain=False)

    # print(eps)
    # print(t)
    import ultraplot as plt

    fig, ax = plt.subplots()
    ax.plot(eps)
    ax.plot(t)

    fig, ax = plt.subplots()

    for j in range(6):
        ax.plot(e_tot[:-1,j], '-o')

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

    # nrg = NRG(H)
    # x, g = discretize(J, 0, 10, 10, mesh='log')

    # print(x, g)

    # e, u = XYZ()

    # print(e)


    # build the overlap matrix

    # S = ...