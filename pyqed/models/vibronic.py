#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:01:33 2023

@author: bing
"""

import numpy as np
import numba
from scipy.sparse import identity, coo_matrix, lil_matrix, csr_matrix, kron
from numpy import meshgrid
from scipy.linalg import eigh
# from cmath import log

import sys
import proplot as plt


from pyqed import boson, interval, sigmax, sort, ket2dm, overlap,\
    polar2cartesian
from pyqed.style import set_style
# from pyqed.units import au2ev, wavenumber2hartree


class CI:
    """
    two-state two-mode Conical Intersection model

    .. math::
        V(x, y) = h x \sigma_z + l y \sigma_x
    """
    def __init__(self, x=None, y=None, h=1, l=1, delta=0, mass=[1, 1], nstates=2):
        # super().__init__(x, y, mass, nstates=nstates)

        assert(len(mass) == nstates)

        # self.dpes()
        self.edip = sigmax()
        assert(self.edip.ndim == nstates)
        self.mass = mass
        self.nstates = nstates

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)

        self.h = h
        self.l = l
        self.delta = delta

        self.va = None
        self.v = None


    def dpes_global(self):
        """
        compute the glocal DPES

        Returns
        -------
        None.

        """
        x, y = self.x, self.y
        nx = self.nx
        ny = self.ny
        N = self.nstates
        h, l = self.h, self.l
        delta = self.delta

        X, Y = meshgrid(x, y)

        v = np.zeros((nx, ny, N, N))

        v[:, :, 0, 0] = X**2/2. + (Y)**2/2. + h * X + delta
        v[:, :, 1, 1] = X**2/2. + (Y)**2/2. - h * X - delta



        self.v = v
        return

    def dpes(self, x):

        X, Y = x
        h, l = self.h, self.l
        delta = self.delta

        v = np.zeros((self.nstates, self.nstates), dtype=complex)

        v[0, 0] = (X)**2/2. + (Y)**2/2. + h * X +  delta
        v[1, 1] = (X)**2/2. + (Y)**2/2. - h * X -  delta
        # v[0, 0] =  + h * X + delta
        # v[1, 1] =  - h * X - delta

        v[0, 1] =   l * Y - 0.5j
        v[1, 0] = v[0, 1].conj()

        self.v = v
        return v

    def apes(self, x):


        v = self.dpes(x)

        # self.v = v
        w, u = eigh(v)
        return w, u

    def wilson_loop(self, n=0, r=1):


        l = identity(self.nstates)


        for theta in np.linspace(0, 2 * np.pi, 800):

            x, y = polar2cartesian(r, theta)


            w, u = self.apes([x, y])

            # print(u[:,0])
            # ground state projection operator
            p = ket2dm(u[:,n])

            l = l @ p

        return np.trace(l)


    def berry_curvature(self):
        """
        Compute the Berry curvature in a grid

        Ref
            Journal of the Physical Society of Japan, Vol. 74, No. 6, June,\
                2005, pp. 1674–1677

        Returns
        -------
        None.

        """
        x, y = self.x, self.y
        nx, ny = self.nx, self.ny

        v, u = self.apes_global()

        def link_x(i, j):

            bra = u[i, j, :, 0]
            ket = u[i+1, j, :, 0]
            z = overlap(bra, ket)
            return z/abs(z)

        def link_y(i, j):
            bra = u[i, j, :, 0]
            ket = u[i, j+1, :, 0]
            z = overlap(bra, ket)
            return z/abs(z)

        F = np.zeros((nx, ny), dtype=complex)
        for i in range(nx-1):
            for j in range(ny-1):

                tmp = link_x(i, j) * link_y((i+1), j)/link_x(i, (j+1))/link_y(i,j)
                F[i ,j] = np.log(tmp)


        return F


    def berry_phase(self, n=0, r=1):


        phase = 0
        z  = 1

        w, u = self.apes([r, 0])
        u0 = u[:, n]

        uold = u0
        loop = np.linspace(0, 2 * np.pi)
        for i in range(1, len(loop)):

            theta = loop[i]
            x, y = polar2cartesian(r, theta)

            w, u = self.apes([x, y])
            unew = u[:,n]
            z *= overlap(uold, unew)

            uold = unew

        z *= overlap(unew, u0)

        return z
        # return -np.angle(z)

    def apes_global(self):

        x = self.x
        y = self.y
        assert(x is not None)

        nstates = self.nstates

        nx = len(x)
        ny = len(y)

        v = np.zeros((nx, ny, nstates))
        u = np.zeros((nx, ny, nstates, nstates), dtype=complex)

        for i in range(nx):
            for j in range(ny):
                w, _u = self.apes([x[i], y[j]])
                v[i, j, :] = w
                u[i, j] = _u

        return v, u

    def plot_apes(self):

        v = self.apes_global()[0]
        mayavi([v[:,:,k] for k in range(self.nstates)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import rcParams

    rcParams['axes.labelpad'] = 6
    rcParams['xtick.major.pad']='2'
    rcParams['ytick.major.pad']='2'

    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    deltas = [-0.5, 0, 0.5, 1, 2]

    # loop_integral = np.zeros(len(deltas))
    # for i, delta in enumerate(deltas):
    #     mol = DHO2(x, y, delta=delta)
    #     # mol.plot_apes()

    #     loop_integral[i] = mol.berry_phase(n=1, r=3)

    # fig, ax = plt.subplots()
    # ax.plot(deltas, loop_integral)

    mol = CI(x, y, delta=0.)
    F = mol.berry_curvature()
    fig, ax = plt.subplots()
    ax.matshow(F.imag)