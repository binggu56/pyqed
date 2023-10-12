#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:07:34 2023


Using Gaussian basis/Fourier series set to propagate the nonadiabatic molecular dynamics

@author: Bing Gu

"""

import numpy as np
from numpy import exp, pi, sqrt, meshgrid
from pyqed import transform, dag, isunitary, rk4, isdiag

from pyqed.mol import Result

import scipy

from scipy.linalg import inv
from scipy.sparse import kron, eye
from scipy.linalg import eigh

import matplotlib.pyplot as plt
# import proplot as plt

# class ResultLDR(Result):
#     def nuclear_density(self):
        

class DHO2:
    def __init__(self, x=None, y=None, mass=[1, 1], nstates=2):
        # super().__init__(x, y, mass, nstates=nstates)

        # self.dpes()
        self.edip = sigmax()
        assert(self.edip.ndim == nstates)
        self.mass = mass
        self.nstates = nstates


    
    def apes(self, x):

        X, Y = x

        N = self.nstates

        v = np.zeros((N, N))

        v[0, 0] = (X+1)**2/2. + Y**2/2.
        v[1, 1] = (X-1)**2/2. + (Y)**2/2. + 2
        v[0, 1] = 0.2 * Y
        v[1, 0] = v[0, 1]

        # self.v = v
        return eigh(v)




def gram_schmidt():
    pass

class GWP:
    def __init__(self, q, p=0, a=1, phase=0, ndim=1):
        """
        normalized multidimensional Gaussian wavepackets
        .. math::
            g(x) ~ e^{- 1/2 (x-q)^T A (x-q) + ip(x-q) + i \theta}

        Parameters
        ----------
        q : TYPE
            DESCRIPTION.
        p : TYPE, optional
            DESCRIPTION. The default is 0.
        a : TYPE, optional
            DESCRIPTION. The default is 1.
        phase : TYPE, optional
            DESCRIPTION. The default is 0.
        ndim : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.q = self.x = np.array(q)
        self.p = np.array(p)

        self.phase = phase
        # self.coeff = coeff # electronic coefficients
        self.ndim = ndim
        # self.a = a
        
        if self.ndim == 1:
            self.var = 1./sqrt(a)
            self.fwhm = 2.*sqrt(2. * np.log(2)) / sqrt(a)
            self.a = a 
            

        else:
            
            if isinstance(a, (float, int)):
                a = [a] * ndim # homogenous width
            
            self.a = a 
            
            if isinstance(p, (float, int)):
                p = np.array([p] * ndim)
                self.p = p
            
            if isinstance(q, (float, int)):
                q = np.array([q] * ndim) 
                self.q = q
            
            # assert(self.a.shape == (ndim, ndim))
            assert(len(self.q) == ndim)
            assert(len(self.p) == ndim)
            # assert(isdiag(self.a))
        
        self.params = (a, q, p)


    def evaluate(self, x):
        a, q, p = self.params
        phase = self.phase 
        
        if self.ndim == 1:
            return (a/pi)**(1/4) * exp(-0.5 * a * (x-q)**2 + 1j * p * (x-q))

        else: 
            
            a = np.diag(a)
            if isinstance(x, list):
                x = np.array(x)
            # the following expression is valid for non-diagonal matrix A
            return (np.linalg.det(a)/pi)**(1/4) * exp(-0.5 * (x-q) @ a @ (x-q) \
                                                      + 1j * p @ (x-q) + 1j * phase)

    def __mult__(self, other):
        pass
        # return GWP(a, q)

def _overlap(aj, qj, ak, qk):
    """
    overlap for real GWPs

    Parameters
    ----------
    aj : TYPE
        DESCRIPTION.
    qj : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.
    qk : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    dq = qk - qj

    return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
    -0.5 * aj*ak/(aj+ak) * dq**2)

def _moment(aj, qj, ak, qk, n=1):
    """
    overlap for real GWPs

    Parameters
    ----------
    aj : TYPE
        DESCRIPTION.
    qj : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.
    qk : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    dq = qj - qk

    S = _overlap(aj, qj, ak, qk)

    if n == 0:
        return S
    elif n == 1:
        return (aj * dq/(aj + ak) + qk) * S
    # elif n == 2:
    #     return (1./(aj + ak) + aj**2 * dq**2/(aj + ak)**2) * S

    # return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
    # -0.5 * aj*ak/(aj+ak) * (dq**2))

def moment(gj, gk, d=0, order=1):
    """
    compute moments between two GWPs
    
    .. math::
        \mu_n = \braket{g_j| x^n |g_k}

    Parameters
    ----------
    g1 : TYPE
        DESCRIPTION.
    g2 : TYPE
        DESCRIPTION.
    d : int
        which DOF
    order : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    aj, qj, pj, sj = gj.a, gj.x, gj.p, gj.phase
    ak, qk, pk, sk = gk.a, gk.x, gk.p, gk.phase

    ndim = gj.ndim 
    
    # overlap for each dof 
    # S = [overlap_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk) \
         # for d in range(ndim)]
    S = [_overlap(aj[n], qj[n], ak[n], qk[n]) for n in range(ndim)]       
    
    M = _moment(aj[d], qj[d], ak[d], qk[d], order)

    
    where = [True] * ndim
    where[d] = False
    res = M * np.prod(S, where=where)
    
    return res
    

class WPD:
    """
    multi-dimensional wavepacket dynamics in a single PES with fixed REAL Gaussian basis set
    """
    def __init__(self, basis, mass=None, coeff=None, ndim=1):
        self.nbasis = len(basis)
        # self.nstates = len(basis[0].coeff)
        self.ndim = self.dim = ndim
        self.coeff = self.c = coeff
        self.basis = basis
        self.mass = mass

        self.v = None
        self.x = None
        self.x_evals = None
        self.x_evecs = None
        self.x2 = None
        self.p = None
        self._K = None # kinetic energy operator
        self._V = None # potential
        self.H = None
        self.S = None

    def set_potential(self, v):
        assert(v.ndim == self.ndim)
        self.v = v

    def buildH(self, representation='dvr'):

        if representation == 'gwp':
            U = self.x_evecs
            # self.H = (U) @ self.buildK() @ dag(U) + self.buildV()
            # self.H = self.buildK() + dag(U) @ self.buildV() @ (U)
            self.H = self.buildK() + self.buildV()

        elif representation == 'dvr':

            U = self.x_evecs
            self.H = dag(U) @ self.buildK() @ U + np.diag(self.v)

        return self.H

    def eigenstates(self, representation='dvr'):

        self.buildH(representation=representation)

        return scipy.linalg.eigh(self.H)


    def norm(self):
        pass

    def overlap(self):

        """
        construct overlap matrix from GWPs defined by {a,x,p}
        This calculation can be significantly reduced if a uniform GWP is used
        """
        N = self.nbasis
        S = np.identity(N)

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k):
                gj = self.basis[j]
                aj, qj = gj.a, gj.q

                S[j, k] = _overlap(aj, qj, ak, qk)

                S[k, j] = S[j, k]

        self.S = S
        return S

    def buildK(self):
        N = self.nbasis
        K = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.x

                dq = qj - qk

                K[j, k] = -1./(2.*self.mass) * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                                              - ak * _overlap(aj, qj, ak, qk))

                K[k, j] = K[j, k]

        self._K = K
        return K

    def buildV(self, method='dvr'):
        # potential energy operator
        # x = self.x
        # v = x @ inv(self.overlap()) @ x
        # # self._V = v

        # print('potential', v)


        # # print(dag(U) @ self.S @ U)

        # # print(dag(U) @ np.diag(self.x_evals**2 @ (U) == self.x)
        # # v = U @ v @ dag(U)
        # v *= 0.5


        if method == 'dvr':
            U = self.x_evecs
            U = inv(U)

            # print(dag(U) @ np.diag(self.x_evals**2 @ (U) == self.x)
            v = dag(U) @ np.diag(self.v) @ U

        elif method == 'lha': # local harmonic approximation

            N = self.nbasis
            v = np.zeros((N, N))

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    v[j, k] = _moment(aj, qj, ak, qk, n=2) + 2*qk*_moment(aj,qj,ak,qk)\
                                                  + qk**2 * _overlap(aj, qj, ak, qk)

                    v[k, j] = v[j, k]

            v *= 0.5


        self._V = v
        return v

    def position(self):
        """
        position matrix elements
        .. math::
            x_{jk} = \braket{\phi_j | x | \phi_k}

        Parameters
        ----------
        shift : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        N = self.nbasis
        x = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.q

                dq = qj - qk

                x[j, k] = (aj * dq/(aj + ak) + qk) * _overlap(aj, qj, ak, qk)

                x[k, j] = x[j, k]

        self.x = x

        return x

    def momentum(self):
        pass

    def moment(self, n=2):
        # return np.diag(self.x_evals**n)
        # self.central_moment(2) + 2.*
        pass

    def central_moment(self, n=1):
        if n == 1:

            N = self.nbasis
            x = np.identity(N)

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (aj * dq/(aj + ak)) * _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        elif n == 2:

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (1./(aj + ak) + aj**2 * dq**2/(aj + ak)**2) * \
                        _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        return x

    def diag_x(self):
        """
        diagonalize the x matrix

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        if self.x is None:
            self.position()
            
        x = self.x

        if self.S is None:
            self.overlap()

        w, u = scipy.linalg.eigh(x, self.S)

        self.x_evals = w
        self.x_evecs = u

        return w, u

    def gwp2dvr(self, a):
        """
        transform from Gaussian basis to the orthogonal basis
        """
        U = self.x_evecs # shape [old, new]
        return dag(U) @ a @ U

    def run(self):
        pass

    def plot_wavepacket(self, c, x):

        psi = 0
        for i, g in enumerate(self.basis):
            psi += c[i] * g.evaluate(x)

        fig, ax = plt.subplots()
        ax.plot(x, psi.real)
        return


class WPD2:
    """
    multi-dimensional wavepacket dynamics in a single PES with fixed REAL Gaussian basis set
    """
    def __init__(self, basis, mass, coeff=None, ndim=2):
        """
        

        Parameters
        ----------
        basis : TYPE
            DESCRIPTION.
        mass : TYPE
            use mass-scaled coordinates.
        coeff : TYPE, optional
            DESCRIPTION. The default is None.
        ndim : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        self.nbasis = len(basis)
        # self.nstates = len(basis[0].coeff)
        self.ndim = self.dim = ndim
        self.coeff = self.c = coeff
        self.basis = basis
        self.mass = mass

        self.v = None
        self.x = [None] * ndim
        self.x_evals = None
        self.x_evecs = None
        self.x2 = None
        self.p = None
        self._K = None # kinetic energy operator
        self._V = None # potential
        self.H = None
        self.S = None

    def set_potential(self, v):
        assert(v.ndim == self.ndim)
        self.v = v

    def buildH(self, representation='dvr'):

        if representation == 'gwp':
            
            U = self.x_evecs
            
            # U = kron(U, U)
            
            # self.H = (U) @ self.buildK() @ dag(U) + self.buildV()
            # self.H = self.buildK() + dag(U) @ self.buildV() @ (U)
            self.H = self.buildK() + self.buildV()

        elif representation == 'dvr':

            U = self.x_evecs
            self.H = dag(U) @ self.buildK() @ U + np.diag(self.v)

        return self.H

    def eigenstates(self, representation='dvr'):

        self.buildH(representation=representation)

        return scipy.linalg.eigh(self.H)


    def norm(self):
        pass

    def overlap(self):

        """
        construct overlap matrix from GWPs defined by {a,x,p}
        This calculation can be significantly reduced if a uniform GWP is used
        """
        N = self.nbasis
        S = np.identity(N)

        for k in range(N):
            gk = self.basis[k]
            # ak, qk = gk.a, gk.q

            for j in range(k):
                gj = self.basis[j]
                # aj, qj = gj.a, gj.q

                # S[j, k] = _overlap(aj, qj, ak, qk)
                S[j, k] = overlap(gj, gk)
                S[k, j] = S[j, k]

        self.S = S
        return S

    def buildK(self):
        N = self.nbasis
        K = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.x

                dq = qj - qk

                K[j, k] = -1./(2.*self.mass) * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                                              - ak * _overlap(aj, qj, ak, qk))

                K[k, j] = K[j, k]

        self._K = K
        return K

    def buildV(self, method='dvr'):
        # potential energy operator
        # x = self.x
        # v = x @ inv(self.overlap()) @ x
        # # self._V = v

        # print('potential', v)


        # # print(dag(U) @ self.S @ U)

        # # print(dag(U) @ np.diag(self.x_evals**2 @ (U) == self.x)
        # # v = U @ v @ dag(U)
        # v *= 0.5


        if method == 'dvr':
            U = self.x_evecs
            U = inv(U)

            # print(dag(U) @ np.diag(self.x_evals**2 @ (U) == self.x)
            v = dag(U) @ np.diag(self.v) @ U

        elif method == 'lha': # local harmonic approximation

            N = self.nbasis
            v = np.zeros((N, N))

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    v[j, k] = _moment(aj, qj, ak, qk, n=2) + 2*qk*_moment(aj,qj,ak,qk)\
                                                  + qk**2 * _overlap(aj, qj, ak, qk)

                    v[k, j] = v[j, k]

            v *= 0.5


        self._V = v
        return v

    def position(self, d=0):
        """
        position operator matrix elements
        
        .. math::
            x_{\mu, jk} = \braket{\phi_j | x_\mu | \phi_k}

        Parameters
        ----------
        d: int
            which DOF
        shift : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        N = self.nbasis
        x = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.q

                # dq = qj - qk

                # x[j, k] = (aj * dq/(aj + ak) + qk) * _overlap(aj, qj, ak, qk)
                x[j, k] = moment(gj, gk, d=d, order=1)

                x[k, j] = x[j, k]

        self.x[d] = x

        return x

    def xprod(self):
        N = self.nbasis
        x = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.q

                dq = qj - qk

                x[j, k] = np.prod([(aj[d] * dq[d]/(aj[d] + ak[d]) + qk[d]) * \
                                _overlap(aj[d], qj[d], ak[d], qk[d]) for d in \
                                    range(self.ndim)])
                # x[j, k] = moment(gj, gk, d=d, order=1)

                x[k, j] = x[j, k]

        w, u = eigh(x)
        return u
    
    def momentum(self):
        pass

    def moment(self, n=2):
        # return np.diag(self.x_evals**n)
        # self.central_moment(2) + 2.*
        pass

    def central_moment(self, n=1):
        if n == 1:

            N = self.nbasis
            x = np.identity(N)

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (aj * dq/(aj + ak)) * _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        elif n == 2:

            for k in range(N):
                gk = self.basis[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.basis[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (1./(aj + ak) + aj**2 * dq**2/(aj + ak)**2) * \
                        _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        return x

    def diag_x(self, d=0):
        """
        diagonalize the x matrix
        Parameters
        ----------
        m: int
            which DOF
            
        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        if self.x[d] is None:
            self.position(d)
            
        x = self.x[d]

        if self.S is None:
            self.overlap()

        w, u = scipy.linalg.eigh(x, self.S)

        self.x_evals = w
        self.x_evecs = u

        return w, u

    def gwp2dvr(self, a):
        """
        transform from Gaussian basis to the orthogonal basis
        
        Parameters
        ----------
        a: ndarray
            operator represented in GWPs
        """
        U = self.x_evecs # shape [old, new]
        # U = kron(U, U)
        return dag(U) @ a @ U

    def run(self):
        pass

    def plot_wavepacket(self, c, x):

        psi = 0
        for i, g in enumerate(self.basis):
            psi += c[i] * g.evaluate(x)

        fig, ax = plt.subplots()
        ax.plot(x, psi.real)
        return


def braket(ket, bra):
    return np.vdot(bra, ket)

class NAWPD(WPD):
    """
    multi-dimensional nonadiabatic wavepacket dynamics (NAWPD) with locally diabatic
    representation

    This method requires adiabatic electronic states at DVR grids, obtained by
    diagonalizing the x matrix.
    """
    def __init__(self, basis, mol=None, nstates=2, ndim=1):
        """


        Parameters
        ----------
        mol : object
            Vibronic coupling model.
        basis : TYPE
            DESCRIPTION.
        nstates : TYPE, optional
            DESCRIPTION. The default is 2.
        ndim : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.nbasis = len(basis)
        self.mol = mol
        # nb, nel = coeff.shape
        # assert(nb == self.nbasis)

        self.basis = basis
        self.nstates = nstates
        self.ndim = ndim


        self.v = None
        self.S = None # overlap matrix
        self.H = None
        self._K = None
        self._V = None

        self.adiabatic_states = None # electronic states


    def buildH(self):
        """
        The potential energy operator is straightforward as it only requires the
        adiabatic PES.

        The T_N construction requires the electronic overlap
        .. math::
            T_{m \beta, n\alpha} = K_{mn} \braket{\beta(R_m)| \alpha(R_n)}

        """
        self._H = self.buildK() + self.buildV()

        pass

    def buildK(self, dtype=float):
        # \braket{x_n|T_N| x_m}
        N = self.nbasis
        M = self.nstates
        mass = self.mol.mass

        K = np.zeros((N, N), dtype=dtype)

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.x

                K[j, k] = -1./(2.*mass) * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                                              - ak * _overlap(aj, qj, ak, qk))

                K[k, j] = K[j, k]

        K = self.gwp2dvr(K)
        # print(K.shape)

        # K = kron(K, eye(M)).toarray().reshape(N, M, N, M)

        # K = scipy.linalg.kron(K, eye(M))
        # K = np.reshape(K, (N, M, N, M))

        # overlap of electronic states
        A = np.zeros((N, N, M, M), dtype=dtype)

        for i in range(N):
            psi1 = self.adiabatic_states[i]

            A[i, i] = np.eye(M) * K[i, i] # identity matrix at the same geometry

            for j in range(i):
                psi2 = self.adiabatic_states[j]

                # for a in range(M):
                #     for b in range(M):
                #         A[i, j, a, b] = braket(psi1[:, a], psi2[:, b])
                #         A[j, i, b, a] = A[i, j, a, b].conj()
                A[i, j] = dag(psi1) @ psi2 * K[i, j]
                A[j, i] = dag(A[i, j])

        A = np.transpose(A, (0, 2, 1, 3))

        self._K = A
        # self._K = np.einsum('ba, mbna ->mbna', K, A)

        return A

    def buildV(self, v=None):
        """
        adiabatic potential energy surface

        Parameters
        ----------
        v : TYPE, size nb, nstates
            DESCRIPTION. The default is None.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        if v is None:
            mol = self.mol
            x = self.x_evals
            v, u = mol.apes(x)

            self._V = np.array(v)
            assert(self._V.shape == (self.nbasis, self.nstates))

            self.adiabatic_states = u # u should be a list of eigenvectors
        else:
            self._V = np.array(v)

        return self._V


    def run(self, psi0, dt, nt, nout=1, e_ops=[]):
        psi = psi0.copy()

        if self._V is None:
            self.buildV()
        V = self._V

        K = self.buildK()

        # assert(V.shape == , K.shape)

        # assert(e_op.shape = (self.nstates)

        result = np.zeros((len(e_ops), nt//nout), dtype=complex)

        for k in range(nt//nout):
            for j in range(nout):
                psi = rk4(psi, tdse, dt, K, V)

            # calculate observables
            result[:, k] = [self.obs_el(psi, e_op) for e_op in e_ops]

        return result



    def obs_el(self, psi, a, kind='electronic'):
        # assume Condon approximation O^n_{ba} = O_{ba}
        if a.ndim == 2:
            return np.trace(psi.conj() @ a @ psi.T)
        elif a.ndim == 3:
            return np.einsum('nb, nba, na', psi.conj(), a, psi)

    def obs_nuc(self):
        pass


class GWP2(WPD2):
    """
    2-dimensional nonadiabatic wavepacket dynamics (NAWPD) with locally diabatic
    representation

    This method requires adiabatic electronic states at DVR grids, obtained by
    diagonalizing the x matrix.
    """
    def __init__(self, basis, mol=None, nstates=2, ndim=2):
        """

        Use mass-scaled coordinates.
        
        Use direct product basis set first, the transformation to Wannier basis 
        is simply a tensor product of 1D transformation matrix. This should be generalized 
        to non-product basis set. 
        

        Parameters
        ----------
        mol : object
            Vibronic coupling model.
        basis : TYPE
            DESCRIPTION.
        nstates : TYPE, optional
            DESCRIPTION. The default is 2.
        ndim : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        self.nbasis = len(basis)
        self.mol = mol
        self.mass = mol.mass
        assert(len(self.mass) == ndim)
        # nb, nel = coeff.shape
        # assert(nb == self.nbasis)

        self.basis = basis
        self.nstates = nstates
        self.ndim = ndim

        self.x = [None] * ndim

        self.v = None
        self.S = None # nuclear overlap matrix
        self.H = None
        self._K = None
        self._V = None
        self.A = None # electronic state overlap matrix
        
        self.U = None # transformation matrix from GWP to DVR
        self.x_evals = None # eigenvalues of the position operators

        self.adiabatic_states = None # electronic states

    def position(self, d=0):
        """
        position operator matrix elements
        
        .. math::
            x_{\mu, jk} = \braket{\phi_j | x_\mu | \phi_k}

        Parameters
        ----------
        d: int
            which DOF
        shift : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        N = self.nbasis
        x = np.zeros((N, N))

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.q

                # dq = qj - qk

                # x[j, k] = (aj * dq/(aj + ak) + qk) * _overlap(aj, qj, ak, qk)
                x[j, k] = moment(gj, gk, d=d, order=1)

                x[k, j] = x[j, k].conj()

        self.x[d] = x

        return x

    # def buildH(self):
    #     """
    #     The potential energy operator is straightforward as it only requires the
    #     adiabatic PES.

    #     The T_N construction requires the electronic overlap
    #     .. math::
    #         T_{m \beta, n\alpha} = K_{mn} \braket{\beta(R_m)| \alpha(R_n)}

    #     """
    #     self._H = self.buildK() + self.buildV()

    #     pass
    def gwp2dvr(self, a):
        """
        transform from Gaussian basis to the orthogonal basis
        
        Parameters
        ----------
        a: ndarray
            operator represented in GWPs
        """
        if self.U is None:
            raise ValueError('Transformation matrix is None.')
        U = self.U # shape [old, new]
        # U = kron(U, U)
        return dag(U) @ a @ U
    
    def buildK(self, dtype=complex):
        # \braket{x_n|T_N| x_m}
        N = self.nbasis
        M = self.nstates
        mass = self.mol.mass

        K = np.zeros((N, N), dtype=dtype)

        for k in range(N):
            gk = self.basis[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.basis[j]
                aj, qj = gj.a, gj.x

                # K[j, k] = -0.5 * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                #                               - ak * _overlap(aj, qj, ak, qk))
                K[j, k] = kin_me(gj, gk, mass)
                
                K[k, j] = K[j, k]

        # transform to Wannier basis
        K = self.gwp2dvr(K)


        # overlap of electronic states
        A = np.zeros((N, N, M, M), dtype=dtype)
        self._K = np.zeros((N, N, M, M), dtype=dtype)


        for i in range(N):
            psi1 = self.adiabatic_states[i]

            A[i, i] = np.eye(M) #* K[i, i] # identity matrix at the same geometry

            for j in range(i):
                psi2 = self.adiabatic_states[j]

                # for a in range(M):
                #     for b in range(M):
                #         A[i, j, a, b] = braket(psi1[:, a], psi2[:, b])
                #         A[j, i, b, a] = A[i, j, a, b].conj()
                A[i, j] = dag(psi1) @ psi2 #* K[i, j]
                A[j, i] = dag(A[i, j])


        # self._K = A
        for a in range(ns):
            for b in range(ns):        
                self._K[:, :, a, b] = A[:, :, a, b] * K 

        self._K = np.transpose(self._K, (0, 2, 1, 3))

        A = np.transpose(A, (0, 2, 1, 3))
        self.A = A
        
        return self._K

    def buildV(self, v=None):
        """
        adiabatic potential energy surface

        Parameters
        ----------
        v : TYPE, size nb, nstates
            DESCRIPTION. The default is None.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        if v is None:
            mol = self.mol
            x = self.x_evals
            ns = self.nstates
            
            wlist = []
            ulist = []

            for i in range(nb):
                v = np.zeros((ns, ns))

                w, u = mol.apes(x[i])
                wlist.append(w.copy())
                ulist.append(u.copy())

            self._V = np.array(wlist)
            assert(self._V.shape == (self.nbasis, self.nstates))

            self.adiabatic_states = ulist # u should be a list of eigenvectors
        else:
            self._V = np.array(v)

        return self._V


    def run(self, psi0, dt, nt, nout=1, e_ops=[], nuc_ops=[]):
        """
        

        Parameters
        ----------
        psi0 : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        nt : TYPE
            DESCRIPTION.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        e_ops : TYPE, optional
            list of electronic operators. The default is [].
        nuc_ops : TYPE, optional
            list of nuclear operators. The default is [].

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        psi = psi0.copy()

        if self._V is None:
            self.buildV()
        V = self._V

        K = self.buildK()

        # assert(V.shape == , K.shape)

        # assert(e_op.shape = (self.nstates)
        result = Result(dt=dt, Nt=nt, nout=nout, t0=0.)
        result.psilist = [psi0]
        
        
        obs = np.zeros((len(e_ops) + len(nuc_ops), nt//nout), dtype=complex)
        
        # observables at t0
        obs[:, 0] = [self.obs_nuc(psi0, e_op) for e_op in nuc_ops] + \
            [self.obs_el(psi0, a) for a in e_ops]
                
        for k in range(1, nt//nout):
            for j in range(nout):
                psi = rk4(psi, tdse, dt, K, V)
            
            result.psilist.append(psi.copy())
            
            # calculate observables
            obs[:, k] = [self.obs_nuc(psi, e_op) for e_op in nuc_ops] + \
                [self.obs_el(psi, a) for a in e_ops]
        
        result.observables = obs
        
        return result



    def obs_el(self, psi, a, kind='electronic'):
        # assume Condon approximation A^n_{ba} = A_{ba}
        if a.ndim == 2:
            return np.trace(psi.conj() @ a @ psi.T)
        elif a.ndim == 3:
            return np.einsum('nb, nba, na', psi.conj(), a, psi)

    def obs_nuc(self, psi, a):
        """
        Compute the expectation value of an nuclear operator A.

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.einsum('mb, mn, mbna, na', psi.conj(), a, self.A, psi)
        
    def rdm_nuc(self, psi):
        return np.einsum('mb, mbna, na -> nm', psi.conj(), self.A, psi)
    
    def nuclear_density(self, psi, x, y):
        """
        compute the nuclear density 
        .. math::
            \rho(\bf R, t) = \sum_{n,m} \sum_{\beta, \alpha} \
                C_{m\beta}^* A_{m\beta, n\alpha} C_{n\alpha} \chi_n(\bf R) \chi^*_m(\bf R)

        Parameters
        ----------
        x : TYPE, optional
            DESCRIPTION. The default is None.
        y : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """ 
        # reduced 
        rdm = self.rdm_nuc(psi)
        
        # if x is None:
        #     x = self.x 
        # if y is None:
        #     y = self.y 
            
        # X, Y = meshgrid(x, y)
        nx = len(x)
        ny = len(y)
        
        g = np.zeros((nb, nx, ny))
        for n in range(nb):
            for i in range(nx):
                for j in range(ny):
                    g[n, i, j] = self.basis[n].evaluate([x[i], y[j]])
        
        
        chi = np.einsum('aij, an -> nij', g, U.toarray()) # nij 
        
        rho = np.einsum('nm, nij, mij -> ij', rdm, chi, chi.conj())
        
        return rho



    

class Smolyak:
    """
    Conical intersection dynamcis with Smolyak sparse grid
    """
    def __init__(self, lmax, ndim=2):
        self.lmax = lmax
    
            
        


def tdse(psi, K, V):
    # return -1j * (np.einsum('ijkl, kl -> ij', K, psi) + V * psi)
    return -1j * (np.tensordot(K, psi) + V * psi)

def overlap_1d(aj, x, px, sj, ak, y, py, sk):
    """
    overlap between two 1D GWPs
    """
    dp = py - px
    dq = y - x

    return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
            -0.5 * aj*ak/(aj+ak) * (dp**2/aj/ak + dq**2  \
            + 2.0*1j* (px/aj + py/ak) *dq) ) * np.exp(1j * (sk-sj))


def overlap(gj, gk):
    """
    overlap between two GWPs defined by {a,x,p}
    """

    aj, qj, pj, sj = gj.a, gj.x, gj.p, gj.phase
    ak, qk, pk, sk = gk.a, gk.x, gk.p, gk.phase

    assert(gj.ndim == gk.ndim)

    ndim = gj.ndim

    if ndim == 1:
        return overlap_1d(aj, qj, pj, sj, ak, qk, pk, sk)
    
    else:
        tmp = 1.0
        for d in range(ndim):
            tmp *= overlap_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk)

        return tmp

def kin_me(gj, gk, am):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj, pj, sj = gj.a, gj.x, gj.p, gj.phase
    ak, qk, pk, sk = gk.a, gk.x, gk.p, gk.phase

    ndim = gj.ndim 
    
    # overlap for each dof 
    S = [overlap_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk) \
         for d in range(ndim)]
        
    
    K = [kin_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk, am[d])\
         for d in range(ndim)]

    res = 0
    for d in range(ndim):
        where = [True] * ndim
        where[d] = False
        res += K[d] * np.prod(S, where=where)

    return res

# @numba.jit
def kin_1d(aj, qj, pj, sj, ak, qk, pk, sk, am):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    p0 = (aj*pk + ak*pj)/(aj+ak)
    d0 = 0.5/am * ( (p0 + 1j*aj*ak/(aj+ak)*(qj-qk))**2 + aj*ak/(aj+ak) )

    l = d0 * overlap_1d(aj, qj, pj, sj, ak, qk, pk, sk)

    return l

def kmat(bset):
    """
    kinetic energy matrix
    """

    nb = len(bset)

    kin = np.zeros((nb, nb), dtype=complex)

    for i in range(nb):
        gi = bset[i]

        for j in range(i+1):
            gj = bset[j]

            kin[i,j] = kin_me(gi, gj)
            kin[j,i] = np.conj(kin[i,j])

    return kin

def H():
    """
    construct the hamiltonian matrix
    Kinetic energy operator can be computed exactly.
    Potential energy operator - approximation
    Nonadiabatic coupling -
    """
    pass


if __name__ == '__main__':



    def testWPD():
        sol = WPD(basis = [GWP(y, a=4) for y in x], mass=1)

        sol.position()
        w, u = sol.diag_x()

        v = 0.5 * sol.x_evals**2 #+ 0.1*sol.x_evals ** 4

        sol.set_potential(v)

        # sol.buildH()
        # # print(sol.x)
        # # print('potential', np.diag(sol._V))
        # w, v = scipy.linalg.eigh(sol.H, sol.S)
        # w, v = scipy.linalg.eigh(sol.H)
        w, v = sol.eigenstates()

        print('eigvalues', w)


        # y = np.linspace(-8, 8, 100)
        # for j in range(10):
        #     sol.plot_wavepacket(u[:, j], y)

        fig, ax = plt.subplots()
        ax.plot(sol.x_evals**2, 'o')

    def testWPD2():
        x = np.linspace(-8, 8, 15)
        y = np.linspace(-10, 6, 15)
        nx = len(x)
        ny = len(y)
        
        nb = len(x) * len(y)
        print('dx = ', interval(x))
    
        pyrazine = Pyrazine()
        # dho = DHO(d=1, coupling=0.1)
        # ns = dho.nstates
        # for x in range():
        #     w, u = pyrazine.apes(x, y=0)
    
        basis_x = [GWP(_x, a=1) for _x in x]
        basis_y = [GWP(_y, a=1) for _y in y]
        
        solx = WPD(basis_x)
        wx, ux = solx.diag_x()
        xmat = solx.position()
        
        # print(isdiag(solx.gwp2dvr(xmat)))
        
        soly = WPD(basis_y)
        wy, uy = soly.diag_x()
        
    
        U = kron(ux, uy)  
    
        basis = []
        for i in range(nx):
            for j in range(ny):
                q = [x[i], y[j]]
                basis.append(GWP(q, a=1, ndim=2))
        
        nb = len(basis)
        # print('FWHM = ', basis[0].fwhm)
        print('# of basis =', nb)
        # print(basis[0].a)
        solver = WPD2(mol=pyrazine, basis = basis)
        

        v = 0.5 * sol.x_evals**2 #+ 0.1*sol.x_evals ** 4

        sol.set_potential(v)

        # sol.buildH()
        # # print(sol.x)
        # # print('potential', np.diag(sol._V))
        # w, v = scipy.linalg.eigh(sol.H, sol.S)
        # w, v = scipy.linalg.eigh(sol.H)
        w, v = sol.eigenstates()

        print('eigvalues', w)


        # y = np.linspace(-8, 8, 100)
        # for j in range(10):
        #     sol.plot_wavepacket(u[:, j], y)

        fig, ax = plt.subplots()
        ax.plot(sol.x_evals**2, 'o')
        
        
    def draw_pes():
        v = solver.buildV()
        
        fig, (ax, ax1) = plt.subplots(nrows=2)
        v0 = np.array([w[0] for w in v]).reshape((nx, ny))
        v1 = np.array([w[1] for w in v]).reshape((nx, ny))
    
        ax.matshow(v0, cmap='viridis')
        ax1.matshow(v1, cmap='viridis')

    from pyqed.models.pyrazine import DHO
    from pyqed import sigmaz, interval, sigmax, norm
    from pyqed.units import au2fs

    # from pyqed.models.dho import DHO

    x = np.linspace(-6, 6, 21)
    y = np.linspace(-6, 6, 21)
    nx = len(x)
    ny = len(y)
    
    nb = len(x) * len(y)
    print('dx = ', interval(x))

    # pyrazine = Pyrazine()
    mol = DHO2()
    # dho = DHO(d=1, coupling=0.1)
    # ns = dho.nstates
    # for x in range():
    #     w, u = pyrazine.apes(x, y=0)

    basis_x = [GWP(_x, a=1) for _x in x]
    basis_y = [GWP(_y, a=1) for _y in y]
    
    solx = WPD(basis_x)
    wx, ux = solx.diag_x()
    xmat = solx.position()
    
    # print(isdiag(solx.gwp2dvr(xmat)))
    
    soly = WPD(basis_y)
    wy, uy = soly.diag_x()
    

    U = kron(ux, uy)  

    # build DVR basis set
    basis = []
    k = 0
    for i in range(nx):
        for j in range(ny):
            k += 1
            q = [x[i], y[j]]
            print(k, q)
            basis.append(GWP(q, a=1, ndim=2))
    
    nb = len(basis)
    # print('FWHM = ', basis[0].fwhm)
    print('number of basis =', nb)
    print('width of Gaussian wavepacket = {}'.format(basis[0].a))
    # for i, b in enumerate(basis):
    #     print(i, b.x)

    solver = GWP2(mol=mol, basis=basis)

    # U = solver.xprod()
    # xmat = solver.position(d=0)

    # print('position', x)
    # print('overlap matrix', solver.overlap().shape)
    # w, u = solver.diag_x()
    
    solver.U = U
    
    Y = solver.position(d=0)
    Y = solver.gwp2dvr(Y)
 
    print(isdiag(Y))
    
    
    
    x = [] 
    for i in range(nx):
        for j in range(ny):
            x.append([wx[i], wy[j]])
    print('len x =', len(x))
    solver.x_evals =  x   
    # print('eigvals of x = ', w)
    v = solver.buildV()

    ns = mol.nstates
    
    # initial state in GWP
    psi0 = np.zeros((nb, ns), dtype=complex)
    # print(x[len(x)//2])
    # psi0[(len(x)//2), 0] = 1

    # psi0 = dag(solver.x_evecs) @ solver.S @ psi0
    x0 = (-1, 0)
    chi0 = GWP(x0, ndim=2)
    tmp = [overlap(chi0, g) for g in basis]

    psi0[:, 1] =   np.conj(np.array(tmp) @ solver.U)

    print('norm of initial state', norm(psi0[:,1]))
    # psi0[:, 0] = chi0.evaluate(solver.x_evals)

    p0 = np.zeros((2,2))
    p0[0, 0] = 1.

    # diabatic population
    u = solver.adiabatic_states

    o = np.zeros((nb, ns, ns))
    for i in range(nb):
        ui = u[i]
        # print(isunitary(ui))
        # o[i] = np.outer(ui[0,:].conj(), ui[0, :])
        # o[i] = dag(ui) @ p0 @ ui
        o[i] = ui @ p0 @ dag(ui)

    dt = 0.02
    nt = 2000
    nout = 1
    print('time', nt * dt * au2fs)
    result = solver.run(psi0, dt, nt, e_ops = [o], nuc_ops=[Y], nout=nout)
    
    # plot A
    # fig, ax = plt.subplots(figsize=(4,4))
    # ax.matshow(np.abs(solver.A[:, 0, 112, 0].reshape((nx, ny))))
    
    # # fig, ax = plt.subplots(figsize=(4,4))
    # # ax.imshow(np.abs(solver.A[:, 0, :, 1]))
    # fig, ax = plt.subplots(figsize=(4,4))
    # ax.matshow(np.abs(solver.A[:, 1, 112, 0].reshape((nx, ny))))
    
    # fig, ax = plt.subplots(figsize=(4,4))
    # ax.imshow(np.abs(solver.A[:, 1, :, 1]))

    
    result.save('gwp')

    fig, ax = plt.subplots()
    # ax.plot(result[1, :].real)
    # ax.plot(result[0, :].real)
    # ax.plot(result.times, result.observables[0, :].real)

    # fig, ax = plt.subplots()
    # ax.plot(result.times, result.observables[1, :].real)
    
    x = np.linspace(-6,6)
    y = np.linspace(-6,6)
    rho = solver.nuclear_density(result.psilist[-1], x, y)
    np.savez('nuclear_density_{}'.format(int(nt*dt)), x, y, rho)
    fig, ax = plt.subplots()
    ax.contour(rho.real, levels=40, origin='lower')
    
    # c = [result.psilist[k][112, :] for k in range(nt//nout)]
    # fig, ax = plt.subplots()
    # ax.plot(result.times, [np.prod(_c).real for _c in c])

