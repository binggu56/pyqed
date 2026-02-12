#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 22:14:36 2025

1D Gaussian integrals

@author: bingg
"""


import numpy as np
import pyqed



class Gaussian:
    """
    2D Real GWP
    """
    def __init__(self, alpha=1, center=0):

        if isinstance(alpha, (float, int)):
            alpha = np.array([alpha, ] * ndim)
        self.alpha = alpha

        if isinstance(center, (float, int)):
            center = np.array([center, ] * ndim)
        self.center = center

        self.ndim = ndim

class STO:
    """
    Contracted Gaussians for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return

class ContractedGaussian:
    """
    Contracted Gaussian basis set for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return


def sto_3g(center, zeta):

    scaling = zeta ** 2

    return ContractedGaussian(3, [0.444635, 0.535328, 0.154329],
               [Gaussian(scaling*0.109818, center),
                Gaussian(scaling*0.405771, center),
                Gaussian(scaling*2.22766, center)])

def sto_6g(center, zeta=1):

    c = [0.9163596281E-02, 0.4936149294E-01,  0.1685383049E+00, 0.3705627997E+00,\
         0.4164915298E+00, 0.1303340841E+00]

    a = [0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01, 0.6259552659E+00, \
      0.2430767471E+00, 0.1001124280E+00]

    g = [Gaussian(alpha=a[i], center=center) for i in range(6)]

    return ContractedGaussian(6, c, g)


def sliced_eigenstates(mol, basis, z, k=1, contract=True):
    """
    adiabatic states at z

    Parameters
    ----------
    basis : ContractedGaussian obj or list of Gaussians
        CGBF.

    z : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    basis : TYPE
        unnormalized contracted Gaussian basis set

    """

    # scaling = zeta ** 2

    # sto = STO(3, [0.444635, 0.535328, 0.154329],
    #            [Gaussian(scaling*0.109818, center),
    #             Gaussian(scaling*0.405771, center),
    #             Gaussian(scaling*2.22766, center)])

    assert isinstance(basis, ContractedGaussian)
    # if isinstance(basis, ContractedGaussian):
        # if the basis is a single CGBF, then we use the primitive Gaussians as
        # the basis set for the transversal SE

    # absorb the exp(-a *z**2) part into coefficients
    n = basis.n
    sliced_basis = ContractedGaussian(n)

    gs = []
    # c = np.zeros(n)
    for i in range(basis.n):

        # g = basis.g[i]
        a = basis.g[i].alpha
        r0 = basis.g[i].center

        # c[i] = basis.c[i] * exp(-a[2] * (z-r0[2])**2) * (2*a[2]/np.pi)**0.25

        # reduce the dimension to 2
        gs.append(Gaussian(center = r0[:2], alpha = a[:2], ndim=2))


    # build overlap matrix
    S = np.eye(n)
    for i in range(n):
        for j in range(i):
            S[i,j] = overlap_2d(gs[i], gs[j])
            S[j, i] = S[i, j]

    # build the kinetic energy
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = kin_2d(gs[i], gs[j])
            if i != j: K[j, i] = K[i, j]

    # build the potential energy
    V = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):

            for a in range(mol.natom):
                V[i, j] += electron_nuclear_attraction(gs[i], gs[j], (z - mol.atom_coord(a)[2]))

            if i != j: V[j, i] = V[i, j]

    H = K + V
    E, U = eigsh(H, k=k, M=S, which='SA')

    # print(E)
    # renormalize the sliced basis
    # sto.d *= normalize(z)
    # for n in range(k):
    if contract:
        sliced_basis = [ContractedGaussian(n, U[:,m], gs) for m in range(k)]

        return E, U, sliced_basis
    else:
        return E, U

    # elif isinstance(basis, list):
    #     # a list of CGBFs, we use the CGBF as the basis without unfolding
    #     # to the primitive GB
    #    pass




class Molecule(pyqed.qchem.Molecule):
    def __init__(self, atom, nz, zrange, norb=1, basis='sto-6g', sliced_basis='eigensates', **kwargs):
        """

        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.
        zrange : TYPE
            DESCRIPTION.
        norbs : TYPE, optional
            transversal orbs for each slice. The default is 1.
        basis : TYPE, optional
            DESCRIPTION. The default is 'sto-3g'.
        sliced_basis : TYPE, optional
            DESCRIPTION. The default is 'no'.

        Returns
        -------
        None.

        """
        super().__init__(atom, kwargs)

        self.zrange = zrange
        self.nz = nz
        self.basis = basis
        self.sliced_basis = sliced_basis
        self.norb = norb
        # self.mol = mol

        ###
        # self.hcore = None
        # self.eri = None


    def build(self):

        nz = self.nz

        dvr_z = SineDVR(npts=nz, *self.zrange)
        dz = dvr_z.dx

        # transversal orbs
        nstates = norb = self.norb

        dvr_z = SineDVR(npts=nz, xmin=-L, xmax=L)
        # dvr_z = SincDVR(20, nz)
        z = dvr_z.x
        Kz = dvr_z.t()

        # print("z = ",z)
        nstates = self.norb

        # T = np.zeros((nz, no, no))

        sto = sto_6g(0)
        nbas = sto.n
        # sto = []
        # for a in range(self.mol.natom):
        #     sto.append(sto_6g(self.mol.atom_coord(a)))


        # if self.sliced_basis == 'eigenstates':

        basis = []
        E = np.zeros((nz, nstates))
        C = np.zeros((nz, nbas, nstates))

        for i in range(nz):
            # for ao in sto:
            E[i, :], C[i], sliced_basis = sliced_eigenstates(self, sto, z[i], k=nstates)

            basis.append(sliced_basis)

            print(E[i])

        # # overlap between 2D Gaussians
        b = basis[0][0]
        nb = b.n
        s = np.eye(nb)
        for i in range(nb):
            for j in range(i):
                s[i, j] = overlap_2d(b.g[i], b.g[j])
                s[j, i] = s[i, j]



        # basis, s = sliced_contracted_gaussian(sto, z, ret_s=True)

        # basis = [normalize(b) for b in basis]

            # C = normalize(b)
            # # for g in basis.g:
            # b.c = np.array(b.c) * C

        # print(normalize(basis, s))


        # # transversal kinetic energy matrix
        # T = np.zeros(nz)
        # for n in range(nz):

        #     b = basis[n]
        #     T[n] = kin_sto(b, b)

        # T = np.diag(T)

        # # attraction energy matrix

        # v = np.zeros(nz)
        # # b1 = sto_3g_hydrogen(0)
        # # b2 = sto_3g_hydrogen(0)

        # for i in range(nz):
        #     b = basis[i]
        #     v[i] = nuclear_attraction_sto(b, b, z[i])
        # V = np.diag(v)


        # # construct H'

        # hcore = T + V




        #overlap matrix

        S = np.eye(nz)
        # basis = [0]*nz
        # for i in range(nz):
        #     basis[i] = sto_3g_hydrogen(0)
        # construct S

        # for i in range(nz):
        #     for j in range(i):
        #         S[i, j] = overlap_sto(basis[i], basis[j], s)
        #         S[j, i] = S[i, j]
        # Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S


        # transversal overlap matrix
        S = np.zeros((nz, nz, nstates, nstates))

        for n in range(nz):
            S[n, n] = np.eye(nstates)

        for i in range(nz):
            for j in range(i):

                for u in range(nstates):
                    for v in range(nstates):
                        S[i, j, u, v] = overlap_sto(basis[i][u], basis[j][v], s=s)


                S[j, i] = S[i, j].conj().T


        #print("Tz = ",Tz)
        V = np.diag(E.flatten())

        size = nz * nstates
        self.nao = size
        H = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size) + V

        #print("H = ",H)
        self.hcore = H


        # build ERI between 2D GTOs
        # TODO: exploit the 8-fold symmetry
        eri_gto = np.zeros((nz, nbas, nbas, nbas, nbas))

        for n in range(nz):

            for i in range(nbas):
                g1 = b.g[i]
                for j in range(nbas):
                    g2 = b.g[j]
                    for k in range(nbas):
                        g3 = b.g[k]
                        for l in range(nbas):
                            g4 = b.g[l]

                            eri_gto[n, i,j,k,l] = electron_repulsion_integral(g1, g2, g3, g4, n * dz)

                            # if i != j: eri_gto[n, j, i, k,l] = eri_gto[n, i,j,k,l]
                            # if k != l: eri_gto[n, i,j, l, k] = eri_gto[n, i,j,k,l]

        # print('eri_gto', eri_gto[0])
        # print( C[0].T @ s @ C[0])
        # print(C[0, :, 1])

        # from GTOs to sliced orbs
        eri = np.zeros((nz, nz, norb, norb, norb, norb))

        for m in range(nz):
            for n in range(m, nz):

                eri[m, n] = contract('ijkl, ia, jb, kc, ld -> abcd', eri_gto[n-m], C[m].conj(), C[m], C[n].conj(), C[n])
                if m != n:
                    eri[n, m] = np.transpose(eri[m,n], (2,3,0,1))

        # print('eri', eri[0,0][1,1,1,1])
        self.eri = eri

        return self

    def run(self):

        if self.hcore is None:
            self.build()
        H = self.hcore

        E, U = eigsh(H, k=1, which='SA')
        print("Ground state energy = ", E)

        return E, U

def kinetic_energy(aj, qj, pj, sj, ak, qk, pk, sk, mass=1):
    """
    kinetic energy matrix elements between two multidimensional GWPs

    .. math::

        T_{jk} = \langle g_j | - \frac{1}{2m} \grad^2 | g_k \rangle

    """

    p0 = (aj*pk + ak*pj)/(aj+ak)
    d0 = 0.5/mass * ( (p0+1j*aj*ak/(aj+ak)*(qj-qk))**2 + aj*ak/(aj+ak) )

    l = d0 * overlap(aj, qj, pj, sj, ak, qk, pk, sk)

    return l

def overlap(aj, x, px, sj, ak, y, py, sk):
    """
    overlap between two 1D GWPs <g_j|g_k>

    .. math::

        g(x) = (\alpha/\pi)^{1/4} e^{- \alpha/2 (x-q)^2 + ip(x-q) + i \theta}
    """
    dp = py - px
    dq = y - x

    return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
            -0.5 * aj*ak/(aj+ak) * (dp**2/aj/ak + dq**2  \
            + 2.0*1j* (px/aj + py/ak) *dq) ) * np.exp(1j * (sk-sj))


def nuclear_attraction():
    pass


def electron_repulsion():
    pass

class CGF:
    def __init__(self, coeff, alpha, center):
        """
        a contracted Gaussian functions for z in the HF/DVR method.

        Parameters
        ----------
        coeff : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        center : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.coeff = coeff
        self.alpha = alpha
        self.center = center

    def overlap(self, other):
        pass


def overlap(cg1, cg2):
    pass cg1.coeff.conj() * overlap() * cg2.coeff

def electron_nuclear_attraction():
    pass



from pyqed.dvr.dvr_2d import DVR2

nx=15
ny=15

dvr = DVR2([-6, 6], [-6, 6], nx, ny)

# kinetic energy matrix elements
t = dvr.t()
t = np.reshape(t, (nx, ny, nx, ny))


zbasis = []
for grid in DVR2.grid:
    # for each grid point, there is CGF





mol = Molecule('HHH')

mf = HF(mol, xlim, ylim, nx, ny, basis='sto6g')

mf.run()