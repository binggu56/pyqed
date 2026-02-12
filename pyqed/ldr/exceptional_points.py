#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:22:03 2023

@author: bing
"""

from pyqed.polariton.cavity import Cavity, VibronicPolariton2
from pyqed.models.vibronic import Vibronic2
from pyqed.models.pyrazine import Pyrazine

import numpy as np
from pyqed import au2ev
import cmath
import scipy

from pyqed import sort, interval

from scipy.sparse import kron, identity

from pyqed.units import au2fs, au2k, au2ev
from pyqed import dag, coth, ket2dm, comm, anticomm, sigmax, sort, Mol, \
    polar2cartesian
from pyqed.optics import Pulse
from pyqed.wpd import SPO2, ResultSPO2
from pyqed.ldr.ldr import WPD2, LDR2
from pyqed.ldr.nonherm import NHLDR2

# from pyqed.ldr.gwp import WPD2
from pyqed.nonherm import eig

from pyqed.dvr.dvr_1d import SincDVR

import warnings

# import sys
# if sys.version_info[1] < 10:
#     import proplot as plt
# else:
#     import matplotlib.pyplot as plt


class VibronicPolariton(VibronicPolariton2):
    # def __init__(self, mol, cav):
    #     super(VibronicPolariton2(mol, cav))

    def dpes_global(self, rwa=False):
        """
        Compute the non-Hermitian diabatic potential energy matrix

        .. math::
            H = (\omega_\text{c} - \frac{\kappa}{2} ) a^\dag a

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        mol = self.mol
        cav = self.cav

        if cav.decay is None:
            raise ValueError('Please set cavity decay rate.')

        omegac = cav.omega - 0.5j * cav.decay

        nx, ny = self.nx, self.ny

        nel = mol.nstates
        ncav = cav.ncav

        nstates = self.nstates # polariton states

        v = np.zeros((nx, ny, nstates, nstates), dtype=complex)

        # build the global DPES
        for i in range(nx):
            for j in range(ny):
                v[i, j] = self.dpes(x[i], y[j])

        self.v = v


        return v



    def dpes(self, x, y):
        """
        Compute the diabatic potential energy matrix at geometry (x, y)

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        mol = self.mol
        cav = self.cav
        g = self.g

        assert(g is not None)


        v = kron(mol.dpes(x, y), cav.idm) + kron(mol.idm_el, cav.nonhermH())

        # cavity-molecule coupling
        a = cav.annihilate()

        v += g * kron(mol.edip.real, a + dag(a)).toarray()

        return v

    def ppes(self, return_transformation=True):
        """
        Compute the polaritonic potential energy surfaces by diagonalization

        Parameters
        ----------
        return_transformation : TYPE, optional
            Return transformation matrices. The default is False.

        Returns
        -------
        E : array [nx, ny, nstates]
            eigenvalues

        T : array [nx, ny, nstates, nstates]
            transformation matrix from diabatic states to polaritonic states

        """

        if self.cav.decay is None:
            raise ValueError('Please set the cavity decay rate.')

        nx = self.nx
        ny = self.ny
        nstates = self.nstates

        E = np.zeros((self.nx, self.ny, self.nstates), dtype=complex)

        if not return_transformation:

            for i in range(self.nx):
                for j in range(self.ny):
                    # V = self.v[i, j, :, :]
                    V = self.dpes(x[i], y[j])
                    w = np.linalg.eigvals(V)
                    # E, U = sort(E, U)
                    E[i, j, :] = w

            self.va = E
            return E

        else:

            # T = np.zeros((nx, ny, nstates, nstates), dtype=complex)
            # for i in range(self.nx):
            #     for j in range(self.ny):
            #         V = self.dpes(x[i], y[j])
            #         w, u = sort(*scipy.linalg.eig(V, right=True))
            #         E[i, j, :] = w
            #         T[i, j, :, :] = u
            # self._transformation = T
            # self.va = E

            vr = np.zeros((nx, ny, nstates, nstates), dtype=complex)
            vl = np.zeros((nx, ny, nstates, nstates), dtype=complex)

            for i in range(self.nx):
                for j in range(self.ny):

                    V = self.dpes(x[i], y[j])
                    w, ur, ul = eig(V)

                    E[i, j] = w

                    vr[i, j] = ur
                    vl[i, j] = dag(ul)

            # self._transformation = T_R
            self.apes = E

            return E, vr, vl



    def cut(self, x=None, d=0):

        if x is None:
            x = self.x

        ny = self.ny

        from pyqed import nlargest

        # find the index closest to 0
        ix = nlargest(-np.abs(self.x), with_index=True)[0][1]

        E = np.zeros((ny, self.nstates), dtype=complex)

        for i in range(self.ny):
            V = self.dpes(x=0, y=y[i])
            w, u = sort(*scipy.linalg.eig(V))
            # E, U = sort(E, U)
            E[i, :] = w

        self.va = E
        return E


    def berry_phase(self, state_id=None, loc=np.array([0, 0]), r=0.1):
        """
        Compute the Berry phase along a circle centered at loc with radius r

        Parameters
        ----------
        state_id : int, optional
            DESCRIPTION. The default is 0.
        loc : TYPE, optional
            DESCRIPTION. The default is (0, 0).
        r : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        z : TYPE
            DESCRIPTION.

        """

        z  = 1
        x0, y0 = loc
        if isinstance(loc, list):
            loc = np.array(loc)

        w, ur, ul = self.apes(x0 + r, y0)

        ur0 = ur[:, state_id]
        ul0 = ul[:, state_id]

        ul_old = ul0
        loop = np.linspace(0, 2 * np.pi, 200,endpoint=False)

        points = [[x0+r, y0]]

        for i in range(1, len(loop)):

            theta = loop[i]
            x, y = loc + polar2cartesian(r, theta)

            points.append([x, y])

            w, ur, ul = self.apes(x, y)

            ur_new = ur[:, state_id]
            ul_new = ul[:, state_id]

            z *= np.dot(ul_old, ur_new)

            ul_old = ul_new

        z *= np.dot(ul_new, ur0)

        return - cmath.phase(z)


    def apes(self, x, y):
        """
        Compute the non-Hermitian adiabatic potential energy surfaces

        .. math::
            H = (\omega_\text{c} - \frac{\kappa}{2} ) a^\dag a

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """

        v = self.dpes(x, y)

        w, ur, ul = scipy.linalg.eig(v, left=True, right=True)

        idx = np.argsort(w.real)

        w = w[idx]
        ur = ur[:, idx]
        ul = ul[:, idx]

        return w, ur, ul


    def wilson_loop(self, n=0, loc=[0, 0], r=1, npts=50):
        """
        Compute the Wilson loop along a circle with center at loc and radias r

        .. math::
            W_n = \mathcal{P}  Tr
                = Tr \prod_{i=0}^{N+1} | \phi^R_n(\mathbf R_n) \rangle \langle\
                                      \phi^L_n(\mathbf R_n) |)

        Parameters
        ----------
        n : TYPE, optional
            DESCRIPTION. The default is 0.
        loc : TYPE, optional
            DESCRIPTION. The default is [0, 0].
        r : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        l = identity(self.nstates)

        if isinstance(loc, list):
            loc = np.array(loc)

        for theta in np.linspace(0, 2 * np.pi, npts):

            x, y = loc + polar2cartesian(r, theta)

            w, ur, ul = self.apes(x, y)

            # print(u[:,0])
            # nth state biorthogonal projection operator |R><L|

            p = np.einsum('i, j ->ij', ur[:, n], ul[:, n])
            l = l @ p

        return np.trace(l)


    def SPO(self, indices):
        """
        Wavepacket dynamics in the single-polariton manifold.

        Params
        ======
        indices: list
            index for states to be included in the dynamics

        Returns
        -------
        spo : TYPE
            DESCRIPTION.

        """
        spo = SPO2(self.x, self.y, self.mass, nstates=self.nstates)

        # remove the ground state from the dynamics.
        v = self.v[:, :, [[index] for index in indices], indices]

        # set the DPES
        spo.set_dpes(v)
        # print(spo.v)

        return spo


    def LDR(self):
        """
        Wavepacket dynamics in the LDR manifold.

        Returns
        -------
        LDR : TYPE
            DESCRIPTION.

        """
        ldr = NHLDR2(self.x, self.y, nstates=self.nstates, ndim=2, mass=self.mass, dvr='sine')
        ldr.apes, ldr.right_eigenstates, ldr.left_eigenstates = self.ppes()

        return ldr


    def Lindblad(self):
        # NOT FINISHED. Using grids for the density matrix too expansive.
        # CHECK Markus paper
        from pyqed.oqs import LindbladSolver
        me = LindbladSolver(self.H.real, c_ops=[kron(self.mol.idm, self.cav.a)])

        return me





def kinetic(x, mass=1, dvr='sinc'):
    """
    kinetic enegy operator for the DVR set

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    mass : TYPE, optional
        DESCRIPTION. The default is 1.
    dvr : TYPE, optional
        DESCRIPTION. The default is 'sinc'.

    Returns
    -------
    Tx : TYPE
        DESCRIPTION.
    Refs:

        M.H. Beck et al. Physics Reports 324 (2000) 1-105
    """

    nx = len(x)
    L = x[-1] - x[0]
    dx = interval(x)
    n = np.arange(nx)
    nx = npts = len(x)


    if dvr == 'sinc':

        # Colbert-Miller DVR 1992

        _m = n[:, np.newaxis]
        _n = n[np.newaxis, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2

        T[n, n] = np.pi**2. / 3. / dx**2
        T *= 0.5/mass   # (pc)^2 / (2 mc^2)


    elif dvr == 'sine':
        npts = N = len(x)
        n = np.arange(1, npts + 1)
        _i = n[:, np.newaxis]
        _j = n[np.newaxis, :]
        m = npts + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2 * (-1.)**(_i-_j)/(N+1)**2 * \
                np.sin(np.pi * _i/(N+1)) * np.sin(np.pi * _j/(N+1))\
                /(np.cos(np.pi * _i /(N+1)) - np.cos(_j * np.pi/(N+1)))**2
        T[n - 1, n - 1] = 0.
        T += np.diag(-1/3 + 1/(6 * (N+1)**2) - 1/(2 * (N+1)**2 * np.sin(n * np.pi/(N+1))**2))
        T *= np.pi**2. / (2. * mass * dx**2) #prefactor common to all of T


    elif dvr == 'SincPeriodic':
        _m = n[:, np.newaxis]
        _n = n[np.newaxis, :]
        _arg = np.pi*(_m-_n)/nx
        if (0 == nx % 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.
            T[n, n] = (nx**2. + 2.)/3.
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.
            T[n, n] = (nx**2. - 1.)/3.
        T *= (np.pi/L)**2.
        T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    else:
        raise ValueError("DVR {} does not exist. Please use sinc, sinc_periodic, sine.")

    return T





if __name__ == '__main__':

    from pyqed import gwp, interval
    import ultraplot as plt

    x = np.linspace(-6, 6, 32, endpoint=False)[1:]
    y = np.linspace(-6, 6, 32, endpoint=False)[1:]

    # y = np.linspace(-3.514, -3.5136, 64)
    dx = interval(x)
    dy = interval(y)
    nx = len(x)
    ny = len(y)

    print(nx)

    mol = Pyrazine(x, y)
    mol.buildV()

    omegac = 4.3140605/au2ev
    coupling_strength = 0.1/au2ev
    kappa = 2 * coupling_strength

    cav = Cavity(n_cav=3, freq = omegac, decay = kappa)

    pol = VibronicPolariton(mol, cav, g = coupling_strength)
    print('number of polariton states', pol.nstates)

    print("omegac = ", omegac * au2ev)
    print("kappa = ", kappa * au2ev)
    print("coupling_strength = ", coupling_strength * au2ev)


    ldr = pol.LDR()

    # construct the initial psi. Note that the initial psi0 is diabatic representation.
    psi0 = np.zeros([nx, ny, 9], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            psi0[i, j, 6] = gwp([x[i], y[j]], ndim=2)


    # APES of polariton
    # va = pol.ppes()
    # adiabatic_polaritonic_states_right = va[1]
    # adiabatic_polaritonic_states_left = va[2]
    # np.savez('APES.npz', va = va[0])
    # print(va[0].shape, va[1].shape, va[2].shape)

    # transfrom the initial state to the adiabatic representation
    psi0 = np.einsum('ija, ijab -> ijb', psi0, ldr.left_eigenstates.conj())

    for j in range(9):
        fig, ax = plt.subplots()
        ax.imshow(abs(psi0[:, :, j])**2)


    norm = np.einsum('ijk, ijk', np.conj(psi0), psi0) * dx * dy
    print("----Population of inital time----", norm)


    # r.v = pol.dpes_global()
    results = ldr.run(psi0=psi0, dt=0.5/au2fs, nt=100, nout=10)

    population = results.get_population(plot=True)
    xAve, yAve = ldr.position(results.psilist)
    print(xAve, yAve)

    psilist = results.psilist

    # position1, position2 = r.position1(psilist)

    np.savez('population.npz', pAve=population)
    np.save('psilist', results.psilist) # Only (101,)
    np.savez('position_LDR_Hermitian.npz', xAve=xAve, yAve=yAve)
    # np.savez('position_LDR_NonHermitian.npz', xAve=position1, yAve=position2)
    np.savez('t.npz', times=results.times)

    norm = np.einsum('ijk, ijk', np.conj(psilist[-1]), psilist[-1]) * dx * dy
    print("----Population of last time----", norm)



# number of polariton states 9
# omegac =  4.3140605
# kappa =  0.4
# coupling_strength =  0.1
# (32, 32, 9) (32, 32, 9, 9) (32, 32, 9, 9)
# (32, 32, 9)
# ----Population of inital time---- (1.0143550614070316+0j)
# /home/xyj/.local/lib/python3.10/site-packages/numpy/ma/core.py:2820: ComplexWarning: Casting complex values to real discards the imaginary part
#   _data = np.array(data, dtype=dtype, copy=copy,
# building the adibatic potential energy surfaces ...
# building the kinetic and potential energy propagator
# building the electronic overlap matrix
# ----Population of last time---- (0.41711399807790167+0j)