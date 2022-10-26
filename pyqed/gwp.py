#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:22:56 2022

Using Gaussian basis set to propagate the nonadiabatic molecular dynamics

@author: Bing Gu

"""

import numpy as np
from numpy import exp, pi
from pyqed import transform, dag
import scipy

from scipy.linalg import inv

import proplot as plt

def isunitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))
    # return np.allclose(np.eye(len(m)), m.T.conj() @ m)

def gram_schmidt():
    pass

class GWP:
    def __init__(self, q, p=0, a=1, phase=0, ndim=1):
        self.q = self.x = q
        self.p = p
        self.a = a
        self.params = (a, q, p)
        self.phase = phase
        # self.coeff = coeff # electronic coefficients
        self.ndim = ndim

    def evaluate(self, x):
        a, q, p = self.params
        return (a/pi)**(1/4) * exp(-0.5 * a * (x-q)**2 + 1j * p * (x-q))

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
    multi-dimensional wavepacket dynamics with fixed REAL Gaussian basis set
    """
    def __init__(self, bases, mass, coeff=None, ndim=1):
        self.nbasis = len(bases)
        # self.nstates = len(bases[0].coeff)
        self.ndim = self.dim = ndim
        self.coeff = self.c = coeff
        self.bases = bases
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

    def buildH(self):
        U = self.x_evecs
        # self.H = (U) @ self.buildK() @ dag(U) + self.buildV()
        # self.H = self.buildK() + dag(U) @ self.buildV() @ (U)
        self.H = self.buildK() + self.buildV()

        return self.H

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
            gk = self.bases[k]
            ak, qk = gk.a, gk.q

            for j in range(k):
                gj = self.bases[j]
                aj, qj = gj.a, gj.q

                S[j, k] = _overlap(aj, qj, ak, qk)

                S[k, j] = S[j, k]

        self.S = S
        return S

    def buildK(self):
        N = self.nbasis
        K = np.zeros((N, N))

        for k in range(N):
            gk = self.bases[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.bases[j]
                aj, qj = gj.a, gj.x

                dq = qj - qk

                K[j, k] = -1./(2.*self.mass) * (ak**2 * _moment(aj, qj, ak, qk, n=2)\
                                              - ak * _overlap(aj, qj, ak, qk))

                K[k, j] = K[j, k]

        self._K = K
        return K

    def buildV(self):
        # potential energy operator
        # x = self.x
        # v = x @ inv(self.overlap()) @ x
        # self._V = v

        v = 0.5 * np.diag(self.x_evals**2)

        U = self.x_evecs

        print(dag(U) @ np.diag(self.x_evals) @ (U) == self.x)

        v = dag(U) @ v @ U

        # N = self.nbasis
        # v = np.zeros((N, N))

        # for k in range(N):
        #     gk = self.bases[k]
        #     ak, qk = gk.a, gk.q

        #     for j in range(k+1):
        #         gj = self.bases[j]
        #         aj, qj = gj.a, gj.x

        #         dq = qj - qk

        #         v[j, k] = _moment(aj, qj, ak, qk, n=2) + 2*qk*_moment(aj,qj,ak,qk)\
        #                                       + qk**2 * _overlap(aj, qj, ak, qk)

        #         v[k, j] = v[j, k]

        # v *= 0.5

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
            gk = self.bases[k]
            ak, qk = gk.a, gk.q

            for j in range(k+1):
                gj = self.bases[j]
                aj, qj = gj.a, gj.q

                dq = qj - qk

                x[j, k] = (aj * dq/(aj + ak) + qk) * _overlap(aj, qj, ak, qk)

                x[k, j] = x[j, k]

        self.x = x


        return x

    def moment(self, n=2):
        # return np.diag(self.x_evals**n)
        # self.central_moment(2) + 2.*
        pass

    def central_moment(self, n=1):
        if n == 1:

            N = self.nbasis
            x = np.identity(N)

            for k in range(N):
                gk = self.bases[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.bases[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (aj * dq/(aj + ak)) * _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        elif n == 2:

            for k in range(N):
                gk = self.bases[k]
                ak, qk = gk.a, gk.q

                for j in range(k+1):
                    gj = self.bases[j]
                    aj, qj = gj.a, gj.x

                    dq = qj - qk

                    x[j, k] = (1./(aj + ak) + aj**2 * dq**2/(aj + ak)**2) * \
                        _overlap(aj, qj, ak, qk)

                    x[k, j] = x[j, k]

        return x




    def orthogonalize(self):

        # orthogonalize the basis set
        x = self.x

        w, u = scipy.linalg.eigh(x)

        self.x_evals = w
        self.x_evecs = u

        return w, u

    def basis_transform(self, u):
        # transform from Gaussian bases to the orthogonal bases
        pass

    def run(self):
        pass

    def plot_wavepacket(self, c, x):

        psi = 0
        for i, g in enumerate(self.bases):
            psi += c[i] * g.evaluate(x)

        fig, ax = plt.subplots()
        ax.plot(x, psi.real)
        return



class NAMD:
    """
    multi-dimensional nonadiabatic wavepacket dynamics with locally diabatic
    representation
    """
    def __init__(self, bases, coeff, nstates=2, dim=1):
        self.nbasis = len(bases)
        # self.nstates = len(bases[0].coeff)
        self.dim = dim
        self.v = v

    def buildH(self):
        pass

    def overlap(self):

        """
        construct overlap matrix from GWPs defined by {a,x,p}
        """
        # N = self.nbasis
        # S = np.identity(N, dtype=np.complex128)

        # for j in range(N):
        #     gj = self.bases[j]
        #     aj, qj, pj = gj.a, gj.x, gj.p

        #     for k in range(j):
        #         gk = self.bases[k]
        #         ak, qk, pk = gk.a, gk.x, gk.p

        #         dq = qk - qj
        #         dp = pk - pj

        #         S[j,k] = (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
        #         -0.5 * aj*ak/(aj+ak) * (dp**2/aj/ak + dq**2  \
        #         + 2.0*1j* (pj/aj + pk/ak) *dq)   )

        #         S[k, j] = S[j, k].conj()

        # return S

    def kmat(self):
        pass

    def vmat(self):
        pass

    def orthogonalize(self):

        # orthogonalize the basis set
        pass

    def run(self):
        pass



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

    tmp = 1.0
    for d in range(ndim):
        tmp *= overlap_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk)

    return tmp

def kin_me(gj, gk):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj, pj, sj = gj.a, gj.x, gj.p, gj.phase
    ak, qk, pk, sk = gk.a, gk.x, gk.p, gk.phase

    l = 0.0

    for d in range(ndim):
        l += kin_1d(aj[d], qj[d], pj[d], sj, ak[d], qk[d], pk[d], sk, am[d])

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


if __name__ == '__main__':
    x = np.linspace(-4, 4, 4)
    sol = WPD(bases = [GWP(y, a=8) for y in x], mass=1)

    sol.set_potential(x**2/2.)

    sol.position()
    w, u = sol.orthogonalize()
    print('x eigvals', w)
    print(sol.overlap())
    print(isunitary(sol.x_evecs))
    sol.buildH()
    # print(sol.x)
    # print('potential', np.diag(sol._V))
    w, v = scipy.linalg.eigh(sol.H, sol.S)
    # w, v = scipy.linalg.eigh(sol.H)


    print('eigvalues', w)


    # y = np.linspace(-8, 8, 100)
    # for j in range(10):
    #     sol.plot_wavepacket(u[:, j], y)

    # fig, ax = plt.subplots()
    # ax.plot(w, 'o')
