#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:22:56 2022

Using Gaussian basis set to propagate the nonadiabatic molecular dynamics

@author: Bing Gu

"""

import numpy as np

class GWP:
    def __init__(self, x, p, a, phase, coeff):
        self.x = x
        self.p = p
        self.a = a
        self.phase = phase
        self.coeff = coeff # electronic coefficients


class NAMD:
    def __init__(self, bases, dim=1):
        self.nbasis = len(bases)
        self.nstates = len(bases[0].coeff)
        self.dim = dim

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