#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:22:56 2022

Using Gaussian basis set to propagate the nonadiabatic molecular dynamics

@author: Bing Gu

"""

import numpy as np
from numpy import exp, pi, sqrt
from pyqed import transform, dag, isunitary, rk4, isdiag


import scipy

from scipy.linalg import inv
from scipy.sparse import kron, eye

import matplotlib.pyplot as plt



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
        self.q = self.x = q
        self.p = p

        self.phase = phase
        # self.coeff = coeff # electronic coefficients
        self.ndim = ndim
        self.a = a
        
        if self.ndim == 1:
            self.var = 1./sqrt(a)
            self.fwhm = 2.*sqrt(2. * np.log(2)) / sqrt(a)

        if ndim > 1:

            if isinstance(a, float):
                self.a = a * eye(ndim) # homogenous width
            
            if isinstance(p, float):
                self.p = np.array([p] * ndim)
                
            if isinstance(q, float):
                self.q = np.array([q] * ndim) 
                
            assert(a.shape == (ndim, ndim))
            assert(len(q) == ndim)
            assert(len(p) == ndim)
            assert(isdiag(a))
        
        self.params = (a, q, p)


    def evaluate(self, x):
        a, q, p = self.params
        if self.ndim == 1:
            return (a/pi)**(1/4) * exp(-0.5 * a * (x-q)**2 + 1j * p * (x-q))
        else: 
            pass

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

    if n == 1:
        return (aj * dq/(aj + ak)) * S
    elif n == 2:
        return (1./(aj + ak) + aj**2 * dq**2/(aj + ak)**2) * S

    # return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
    # -0.5 * aj*ak/(aj+ak) * (dq**2))

class WPD:
    """
    multi-dimensional wavepacket dynamics in a single PES with fixed REAL Gaussian basis set
    """
    def __init__(self, basis, mass, coeff=None, ndim=1):
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


class NAWPD2(WPD):
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

                K[j, k] = -0.5 * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                                              - ak * _overlap(aj, qj, ak, qk))

                K[k, j] = K[j, k]

        # transform to Wannier basis
        K = self.gwp2dvr(K)


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
            tmp *= overlap_1d(aj[d, d], qj[d], pj[d], sj, ak[d,d], qk[d], pk[d], sk)

        return tmp

def kin_me(gj, gk, am):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj, pj, sj = gj.a, gj.x, gj.p, gj.phase
    ak, qk, pk, sk = gk.a, gk.x, gk.p, gk.phase

    ndim = gj.ndim 
    
    l = 0.0
    for d in range(ndim):
        l += kin_1d(aj[d,d], qj[d], pj[d], sj, ak[d,d], qk[d], pk[d], sk, am[d])

    return l

# @numba.jit
def kin_1d(aj, qj, pj, sj, ak, qk, pk, sk, am):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    p0 = (aj*pk + ak*pj)/(aj+ak)
    d0 = 0.5/am * ( (p0+1j*aj*ak/(aj+ak)*(qj-qk))**2 + aj*ak/(aj+ak) )

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


    from pyqed.models.pyrazine import Pyrazine, DHO
    from pyqed import sigmaz, interval, sigmax, norm

    # from pyqed.models.dho import DHO

    x = np.linspace(-6, 6, 21)
    nb = len(x)
    print('dx = ', interval(x))

    pyrazine = Pyrazine()
    dho = DHO(d=1, coupling=0.1)
    ns = dho.nstates
    # for x in range():
    #     w, u = pyrazine.apes(x, y=0)


    basis = [GWP(y, a=8) for y in x]
    
    print('FWHM = ', basis[0].fwhm)

    solver = NAWPD(mol=dho, basis = basis)
    solver.position()
    solver.diag_x()
    solver.buildV()


    # initial state in GWP
    psi0 = np.zeros((len(x), ns), dtype=complex)
    # print(x[len(x)//2])
    # psi0[(len(x)//2), 0] = 1

    # psi0 = dag(solver.x_evecs) @ solver.S @ psi0
    chi0 = GWP(0)
    tmp = [overlap(chi0, g) for g in basis]

    psi0[:, 0] =   np.conj(np.array(tmp) @ solver.x_evecs)

    print(norm(psi0[:,0]))
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
        o[i] = dag(ui) @ p0 @ ui

    result = solver.run(psi0, dt=0.05, nt=600, e_ops=[p0, sigmax()])

    fig, ax = plt.subplots()
    # ax.plot(result[1, :].real)
    ax.plot(result[0, :].real)
    # ax.format(ylim=(-0.5,1))


