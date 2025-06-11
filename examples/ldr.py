#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:32:45 2023

@author: bingg
"""

import numpy as np
from numpy import exp, pi, sqrt, meshgrid
from pyqed import transform, dag, isunitary, rk4, isdiag, sinc, sort
from pyqed.wpd import ResultSPO2
from pyqed.ldr.ldr import WPD2

import warnings

# from pyqed.wpd import Result

import scipy

from scipy.linalg import inv
from scipy.sparse import kron, eye
from scipy.linalg import eigh

try:
    import proplot as plt
except:
    import matplotlib.pyplot as plt

# def sincDVRKinetic(x, mass=1, dvr='sinc'):
#     """
#     kinetic enegy operator for the DVR set

#     Parameters
#     ----------
#     x : TYPE
#         DESCRIPTION.
#     mass : TYPE, optional
#         DESCRIPTION. The default is 1.
#     dvr : TYPE, optional
#         DESCRIPTION. The default is 'sinc'.

#     Returns
#     -------
#     Tx : TYPE
#         DESCRIPTION.

#     """

#     # L = xmax - xmin
#     # a = L / npts
#     nx = len(x)
#         # self.n = np.arange(npts)
#         # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
#         # self.w = np.ones(npts, dtype=np.float64) * self.a
#         # self.k_max = np.pi/self.a

#     L = x[-1] - x[0]
#     dx = interval(x)
#     n = np.arange(nx)

#         # Colbert-Miller DVR 1992

#     _m = n[:, np.newaxis]
#     _n = n[np.newaxis, :]

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2

#     T[n, n] = np.pi**2. / 3. / dx**2
#     T *= 0.5/mass   # (pc)^2 / (2 mc^2)

#     return T

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

    # L = xmax - xmin
    # a = L / npts
    nx = len(x)
        # self.n = np.arange(npts)
        # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
        # self.w = np.ones(npts, dtype=np.float64) * self.a
        # self.k_max = np.pi/self.a

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

        # Sine DVR (particle in-a-box)
        # n = np.arange(1, npts + 1)
        # dx = (xmax - xmin)/(npts + 1)
        # x = float(xmin) + self.a * self.n

        npts = N = len(x)
        n = np.arange(1, npts + 1)

        _i = n[:, np.newaxis]
        _j = n[np.newaxis, :]

        m = npts + 1

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     T = ((-1.)**(_i-_j)
        #         * (1./np.square(np.sin(np.pi / (2. * m) * (_i-_j)))
        #         - 1./np.square(np.sin(np.pi / (2. * m) * (_i+_j)))))

        # T[n - 1, n - 1] = 0.
        # T += np.diag((2. * m**2. + 1.) / 3.
        #              - 1./np.square(np.sin(np.pi * n / m)))
        # T *= np.pi**2. / 2. / L**2. #prefactor common to all of T
        # T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

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

def sigma(n):
    """
    zigzag function mapping integers to Fourier components

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if np.mod(n, 2):
        return -n//2
    else:
        return (n+1)//2

# class LDR2(WPD2):

#     def __init__(self, x, y, nstates=2, ndim=2, mass=None):
#         self.x = x
#         self.y = y
#         self.dx = interval(x)
#         self.dy = interval(y)
#         self.nstates = nstates

#         self.kx = 2 * np.pi * scipy.fft.fftfreq(len(x), self.dx)
#         self.ky = 2 * np.pi * scipy.fft.fftfreq(len(y), self.dy)

#         self.nx = len(x)
#         self.ny = len(y)

#         if mass is None:
#             mass = [1, ] * ndim
#         self.mass = mass

#         self.ngrid = [nx, ny]

#         self.kmax = np.pi/self.dx # energy range [-K, K]
#         self.shift = (self.nx - 1)//2

#         self.nbasis = self.nx * self.ny # for product DVR

#         self.X = None
#         self.K = None
#         self.V = None
#         self.H = None
#         self._v = None

#         self.geometies = None
#         self.adiabatic_states = []
#         self.apes = None


class SincDVR(WPD2):
    """
    N-state two-mode conical intersection dynamics with Fourier series

    LDR-SincDVR
    """
    def __init__(self, x, y, nstates=2, ndim=2, mass=None, dvr='sinc'):
        self.x = x
        self.y = y
        self.dx = interval(x)
        self.dy = interval(y)
        self.nstates = nstates

        self.kx = 2 * np.pi * scipy.fft.fftfreq(len(x), self.dx)
        self.ky = 2 * np.pi * scipy.fft.fftfreq(len(y), self.dy)

        self.nx = len(x)
        self.ny = len(y)

        self.dvr = dvr

        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass

        self.ngrid = [nx, ny]

        self.kmax = np.pi/self.dx # energy range [-K, K]
        self.shift = (self.nx - 1)//2

        self.nbasis = self.nx * self.ny # for product DVR

        self.X = None
        self.K = None
        self.V = None
        self.H = None
        self._v = None

        self.geometies = None
        self.adiabatic_states = []
        self.apes = None
        self.A = None # electronic overlap matrix

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        assert(v.shape == (self.nx, self.ny, self.nstates, self.nstates))

        self._v = v

    def primitive_basis(self, n):
        # the index n \in [-N, ... 0, 1, ... N]
        n = sigma(n)
        dx = self.dx
        x = self.x

        return
        # return 1/sqrt(dx) * sinc(np.pi * (x/dx - n))

    def ip(self):
        """
        (i) * momentum matrix elements
        """
        p = np.zeros((self.nx, self.nx))
        for n in range(nx):
            for m in range(nx):
                p[n, m] = 1/self.dx * (-1)**(n-m)/(n-m)
        return p



    def buildK(self, dt):

        dx = self.dx
        # Tx = np.zeros((self.nx, self.nx))
        mx, my = self.mass


        # for n in range(nx):
        #     Tx[n, n] = np.pi**2/3 * 1/(2 * mx * dx**2)

        #     for m in range(n):
        #         Tx[n, m] = 1/(2 * mx * dx**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Tx[m, n] = Tx[n, m]


        Tx = kinetic(self.x, mass=mx, dvr=self.dvr)
        # p2 = kron(p2, np.eye(self.ny))

        expKx = scipy.linalg.expm(-1j * Tx * dt)

        Ty = kinetic(self.y, my, dvr=self.dvr)
        # dy = self.dy
        # Ty = np.zeros((self.ny, self.ny))

        # for n in range(ny):
        #     Ty[n, n] = np.pi**2/3 * 1/(2 * my * dy**2)

        #     for m in range(n):
        #         Ty[n, m] = 1/(2 * my * dy**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Ty[m, n] = Ty[n,m]

        # p2_y = kron(np.eye(self.nx, p2_y))
        expKy = scipy.linalg.expm(-1j * Ty * dt)

        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y

        self.exp_K = [expKx, expKy]
        return

    # def build_apes(self, apes):
    #     nb = self.nbasis
    #     # compute electronic overlap matrix elements
    #     # self.A =

    #     # build the adiabatic PES
    #     v = np.zeros((self.nbasis, self.nstates))
    #     for i in range(nb):

    #         v[i, :], u = apes(self.geometries[i])

    #         self.adiabatic_states.append(u.copy()) # electronic states

    #     self.apes = v

    #     return

    def build_apes(self):
        nx, ny, nstates = self.nx, self.ny, self.nstates

        va = np.zeros((nx, ny, nstates))
        u = np.zeros((nx, ny, nstates, nstates))  # diabatic to adiabatic transformation

        for i in range(nx):
            for j in range(ny):

                vij = self.v[i, j]
                w, v = scipy.linalg.eigh(vij)

                va[i,j] = w
                u[i,j] = v
                #print(np.dot(U.conj().T, Vmat.dot(U)))

        # self.apes = va.reshape((self.nbasis, nstates))
        # self.adiabatic_states = u.reshape((self.nbasis, nstates, nstates))

        self.apes = va
        self.adiabatic_states = u



        return va





    def build(self, dt):
        """
        Setup the propagators appearing in the split-operator method.

        For the kinetic energy operator with Jacobi coordinates

            K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

        Since the two KEOs for each dof do not commute, it has to be factorized as

        e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}

        where $p_\theta = -i \pa_\theta$ is the momentum operator.


        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        intertia: func
            moment of inertia, only used for Jocabi coordinates.

        Returns
        -------
        None.

        """

        # setup kinetic energy operator
        # kx, ky = self.kx, self.ky

        nx, ny, nstates = self.nx, self.ny, self.nstates

        # if self.coords == 'linear':

        # mx, my = self.mass

        # Kx, Ky = meshgrid(self.kx, self.ky)

        # self.exp_K = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)


        # elif self.coords == 'jacobi':

        #     # self.exp_K = np.zeros((nx, ny, nx, ny))
        #     mx = self.masses[0]

        #     self.exp_Kx = np.exp(-1j * self.kx**2/2./mx * dt)

        #     Iinv = 1./self.masses[1](self.x) # y is the angle
        #     ky = self.ky

        #     self.exp_Ky = np.exp(-1j * np.outer(Iinv, ky**2/2.) * dt)

        #     # psik = fft(typsi, axis=0)
        #     # kpsi = np.einsum('i, ija -> ija', np.exp(-1j * kx**2/2./mx * dt), psik)

        #     # # for i in range(nx):
        #     # #     for j in range(ny):
        #     # #         my = self.masses[1](y[j])
        #     # #         self.exp_K[i, j, :, :] = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)


        # dt2 = 0.5 * dt

        # if self.V is None:
        #     raise ValueError('The diabatic PES is not specified.')

        # v = self.V

        # self.exp_V = np.zeros((nx, ny, nstates), dtype=complex)
        # self.exp_V_half = np.zeros((nx, ny, nstates), dtype=complex)
        # self.apes = np.zeros((nx, ny))

        # if self.abc:
        #     eig = scipy.linalg.eig
        # else:
        #     eig = scipy.linalg.eigh

        dt2 = 0.5 * dt
        self.exp_V = np.exp(-1j * dt * self.apes)

        self.exp_V_half = np.exp(-1j * dt2 * self.apes)

        # if np.iscomplexobj(v):

        #     # complex potential
        #     for i in range(nx):
        #         for j in range(ny):

        #             _v = v[i, j]

        #             w, ul, ur = scipy.linalg.eig(_v, left=True, right=True)


        #             V = np.diagflat(np.exp(- 1j * w * dt))
        #             V2 = np.diagflat(np.exp(- 1j * w * dt2))

        #             self.exp_V[i, j, :,:] = ur @ V @ dag(ul)
        #             self.exp_V_half[i, j, :,:] = ur @ V2 @ dag(ul)

        # else:

            # for i in range(nx):
            #     for j in range(ny):

            #         w, u = scipy.linalg.eigh(v[i, j, :, :])

            #         #print(np.dot(U.conj().T, Vmat.dot(U)))

            #         self.exp_V[i,j] = np.diagflat(np.exp(- 1j * w * dt))
            #         self.exp_V_half[i,j] = np.diagflat(np.exp(- 1j * w * dt2))

                    # self.exp_V[i, j, :,:] = u.dot(V.dot(dagger(u)))
                    # self.exp_V_half[i, j, :,:] = u.dot(V2.dot(dagger(u)))

        return

    def build_ovlp(self, dtype=float):


        N = self.nbasis
        M = nstates = self.nstates

        # K = self.buildK().reshape((N, N))

        # overlap of electronic states
        A = np.zeros((nx, ny, nx, ny, nstates, nstates), dtype=dtype)
        # self._K = np.zeros((N, N, M, M), dtype=dtype)


        for k in range(N):

            i, j = np.unravel_index(k, (nx, ny))
        # for i in range(nx):
        #     for j in range(ny):

            psi1 = self.adiabatic_states[i, j]

            A[i, j, i, j] = np.eye(nstates) #* K[i, i] # identity matrix at the same geometry

            for l in range(k):
                ii, jj = np.unravel_index(l, (nx, ny))
                # for ii in range(nx):
                #     for jj in range(ny):
                psi2 = self.adiabatic_states[ii, jj]

                # for a in range(M):
                #     for b in range(M):
                #         A[i, j, a, b] = braket(psi1[:, a], psi2[:, b])
                #         A[j, i, b, a] = A[i, j, a, b].conj()
                A[i, j, ii, jj] = dag(psi1) @ psi2 #* K[i, j]
                A[ii, jj, i, j] = dag(A[i, j, ii, jj])

        expKx, expKy = self.exp_K

        # trivializing the topology
        # A = np.abs(A)

        self.exp_T = np.einsum('ijklab, ik, jl -> ijaklb', A, expKx, expKy)

        # for a in range(nstates):
        #     for b in range(nstates):
        #         self._K[:, :, a, b] = A[:, :, a, b] * K

        # self._K = np.transpose(self._K, (0, 2, 1, 3))


        # A = np.transpose(A, (0, 1, 4, 2, 3, 5))
        self.A = A


        # return self._K


    def xmat(self, d=0):
        """
        position operator matrix elements

        .. math::
            e^{i x_\mu}_{jk} = \braket{\phi_j | e^{i x_\mu} | \phi_k}

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

        # for k in range(N):


        # self.x[d] = x

        # return x

    def run(self, psi0, dt, nt, nout=1, t0=0):

        assert(psi0.shape == (self.nx, self.ny, self.nstates))

        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()

        print('building the kinetic and potential energy propagator')
        self.buildK(dt)

        print('building the electronic overlap matrix')
        if self.A is None:
            self.build_ovlp()

        self.build(dt)


        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0]
        psi = psi0.copy()
        psi = self.exp_V_half * psi
        # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)

        for k in range(nt//nout):
            for kk in range(nout):
                # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)

                # psi = self._KEO_linear(psi)
                # psi = np.einsum('ijaklb, klb->ija', self.A, psi)
                psi = np.einsum('ijaklb, klb->ija', self.exp_T, psi)
                psi = self.exp_V * psi
                # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)

            r.psilist.append(psi.copy())

        psi = self.exp_V_half * psi

        return r

    def rdm_el(self, psi):
        """
        Compute the reduced electronic density matrix

        Parameters
        ----------
        psi : TYPE
            vibronic wavefunction.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """
        nstates = self.nstates

        if isinstance(psi, np.ndarray):

            rho = np.zeros((nstates, nstates), dtype=complex)

            for i in range(self.nstates):
                for j in range(i, self.nstates):
                    rho[i, j] = np.sum(np.multiply(np.conj(psi[:, :, i]), psi[:, :, j]))*self.dx*self.dy
                    if i != j:
                        rho[j, i] = rho[i, j].conj()

        elif isinstance(psi, list):

            rho = []
            for p in psi:
                tmp = np.einsum('ija, ijb -> ab', p.conj(), p) * self.dx * self.dy
                rho.append(tmp.copy())

        return rho

    def _KEO_linear(self, psi):
        # psik = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     psik[:,:,j] = fft2(psi[:,:,j])
        psik = scipy.fft.fft2(psi, axes=(0,1))
        kpsi = np.einsum('ij, ija -> ija', self.exp_K, psik)

        # out = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     out[:, :, j] = ifft2(kpsi[:, :, j])
        psi = scipy.fft.ifft2(kpsi, axes=(0,1))
        return psi

    def fbr2dvr(self):
        # this is simply the FFT.
        pass

    def dvr2fbr(self):
        pass


    def x_evolve(self, psi):
        return self.expV * psi

    def k_evolve(self, psi):
        return self.expK @ psi


class SineDVR(SincDVR):
    """
    https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
    """

    def buildK(self, dt):

        mx, my = self.mass


        # for n in range(nx):
        #     Tx[n, n] = np.pi**2/3 * 1/(2 * mx * dx**2)

        #     for m in range(n):
        #         Tx[n, m] = 1/(2 * mx * dx**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Tx[m, n] = Tx[n, m]

        Tx = kinetic(self.x, mx, dvr='sine')

        # p2 = kron(p2, np.eye(self.ny))

        expKx = scipy.linalg.expm(-1j * Tx * dt)

        Ty = kinetic(self.y, my, dvr='sine')
        # dy = self.dy
        # Ty = np.zeros((self.ny, self.ny))

        # for n in range(ny):
        #     Ty[n, n] = np.pi**2/3 * 1/(2 * my * dy**2)

        #     for m in range(n):
        #         Ty[n, m] = 1/(2 * my * dy**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Ty[m, n] = Ty[n,m]

        # p2_y = kron(np.eye(self.nx, p2_y))
        expKy = scipy.linalg.expm(-1j * Ty * dt)

        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y

        self.exp_K = [expKx, expKy]
        return

def position_eigenstate(x, n, npts, L):
    """
    nth position eigenstate for sine DVR, n= 1, \cdots, npts

    Both ends are not included in the grid.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    npts : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    """
    dx = L/(npts + 1)
    xn = grid[n]
    chi = np.sin(np.pi/2 *(2 * npts + 1) * (x - xn)/L)/np.sin(np.pi/2 * (x - xn)/L) -\
          np.sin(np.pi/2 *(2 * npts + 1) * (x + xn)/L)/np.sin(np.pi/2 * (x + xn)/L)
    chi *= 1/2/np.sqrt(L * (npts + 1))

    return chi



class SincDVRPeriodic(SincDVR):

    def buildK(self, dt):

        mx, my = self.mass


        # for n in range(nx):
        #     Tx[n, n] = np.pi**2/3 * 1/(2 * mx * dx**2)

        #     for m in range(n):
        #         Tx[n, m] = 1/(2 * mx * dx**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Tx[m, n] = Tx[n, m]

        Tx = kinetic(self.x, mx, dvr='sincperiodic')

        # p2 = kron(p2, np.eye(self.ny))

        expKx = scipy.linalg.expm(-1j * Tx * dt)

        Ty = kinetic(self.y, my, dvr='sinc_periodic')
        # dy = self.dy
        # Ty = np.zeros((self.ny, self.ny))

        # for n in range(ny):
        #     Ty[n, n] = np.pi**2/3 * 1/(2 * my * dy**2)

        #     for m in range(n):
        #         Ty[n, m] = 1/(2 * my * dy**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Ty[m, n] = Ty[n,m]

        # p2_y = kron(np.eye(self.nx, p2_y))
        expKy = scipy.linalg.expm(-1j * Ty * dt)

        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y

        self.exp_K = [expKx, expKy]
        return

def dpes(x, y):
    nx, ny = len(x), len(y)
    v = np.zeros(shape = (nx, ny, 2,2))

    X, Y = np.meshgrid(x, y, indexing='ij')

    v[:, :, 0, 0] = 0.5 * (X+1)**2 + 0.5 * Y**2
    v[:, :, 1, 1] = 0.5 * (X-1)**2 + 0.5 * Y**2 + 2
    v[:, :, 0, 1] = v[:, :, 1, 0] = 0.2 * Y

    return v

def dump(r, fname):
    """
    save results to disk

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(r, f)

if __name__ == '__main__':

    # from pyqed.models.pyrazine import DHO
    from pyqed import sigmaz, interval, sigmax, norm
    from pyqed.models.pyrazine import Pyrazine
    from pyqed import discretize
    from pyqed.phys import gwp
    from pyqed.units import au2fs

    # from pyqed.models.dho import DHO

    # x = np.linspace(-6, 6, 2**5)
    # y = np.linspace(-6, 6, 2**5)
    l = 5
    x = discretize(-6, 6, l)
    y = discretize(-6, 6, l)
    nx = len(x)
    ny = len(y)

    nb = len(x) * len(y)
    dx = interval(x)
    dy = interval(y)

    print('dx = ', dx, dy)

    # X, Y = np.meshgrid(x, y)

    v = dpes(x, y)
    print(v.shape)
    nstates = 2


    # mol = Pyrazine()
    # v = mol.dpes(x, y)
    # mol = DHO2()
    # # dho = DHO(d=1, coupling=0.1)
    # # ns = dho.nstates
    # # for x in range():
    # #     w, u = pyrazine.apes(x, y=0)

    # basis_x = [GWP(_x, a=1) for _x in x]
    # basis_y = [GWP(_y, a=1) for _y in y]

    # solx = WPD(basis_x)
    # wx, ux = solx.diag_x()
    # xmat = solx.position()

    # # print(isdiag(solx.gwp2dvr(xmat)))

    # soly = WPD(basis_y)
    # wy, uy = soly.diag_x()


    # U = kron(ux, uy)

    # # build DVR basis set
    # basis = []
    # k = 0
    # for i in range(nx):
    #     for j in range(ny):
    #         k += 1
    #         q = [x[i], y[j]]
    #         print(k, q)
    #         basis.append(GWP(q, a=1, ndim=2))

    # nb = len(basis)
    # # print('FWHM = ', basis[0].fwhm)
    # print('number of basis =', nb)
    # print('width of Gaussian wavepacket = {}'.format(basis[0].a))
    # # for i, b in enumerate(basis):
    # #     print(i, b.x)

    # exact dynamics

    solver = SincDVR(x, y)
    # solver = SineDVR(x, y)
    solver.v = v

    psi0 = np.zeros((nx, ny, 2), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            psi0[i,j,1] = gwp(np.array([x[i], y[j]]), x0=[-1., 0.], ndim=2) * np.sqrt(dx * dy)

    nt = 1


    result = solver.run(psi0, dt=0.5, nt=nt)


    # print(np.linalg.eigvals(np.reshape(np.transpose(solver.A, (0, 1, 4, 2, 3, 5)),\
    #                                    (nx * ny * nstates, nx*ny*nstates))))

    # result.get_population(fname='population', plot=True)
    # result.position(plot=True)

    # result.dump('ldr_l{}'.format(l))







    # x = []
    # for i in range(nx):
    #     for j in range(ny):
    #         x.append([wx[i], wy[j]])
    # print('len x =', len(x))
    # solver.x_evals =  x
    # print('eigvals of x = ', w)
    # v = solver.buildV()

    # ns = mol.nstates

    # # initial state in GWP
    # psi0 = np.zeros((nb, ns), dtype=complex)
    # # print(x[len(x)//2])
    # # psi0[(len(x)//2), 0] = 1

    # # psi0 = dag(solver.x_evecs) @ solver.S @ psi0
    # x0 = (-1, 0)
    # chi0 = GWP(x0, ndim=2)
    # tmp = [overlap(chi0, g) for g in basis]

    # psi0[:, 1] =   np.conj(np.array(tmp) @ solver.U)

    # print('norm of initial state', norm(psi0[:,1]))
    # # psi0[:, 0] = chi0.evaluate(solver.x_evals)

    # p0 = np.zeros((2,2))
    # p0[0, 0] = 1.

    # # diabatic population
    # u = solver.adiabatic_states

    # o = np.zeros((nb, ns, ns))
    # for i in range(nb):
    #     ui = u[i]
    #     # print(isunitary(ui))
    #     # o[i] = np.outer(ui[0,:].conj(), ui[0, :])
    #     # o[i] = dag(ui) @ p0 @ ui
    #     o[i] = ui @ p0 @ dag(ui)

    # dt = 0.02
    # nt = 2000
    # nout = 1
    # print('time', nt * dt * au2fs)
    # result = solver.run(psi0, dt, nt, e_ops = [o], nuc_ops=[Y], nout=nout)

    # # plot A
    # # fig, ax = plt.subplots(figsize=(4,4))
    # # ax.matshow(np.abs(solver.A[:, 0, 112, 0].reshape((nx, ny))))

    # # # fig, ax = plt.subplots(figsize=(4,4))
    # # # ax.imshow(np.abs(solver.A[:, 0, :, 1]))
    # # fig, ax = plt.subplots(figsize=(4,4))
    # # ax.matshow(np.abs(solver.A[:, 1, 112, 0].reshape((nx, ny))))

    # # fig, ax = plt.subplots(figsize=(4,4))
    # # ax.imshow(np.abs(solver.A[:, 1, :, 1]))


    # result.save('gwp')

    # fig, ax = plt.subplots()
    # # ax.plot(result[1, :].real)
    # # ax.plot(result[0, :].real)
    # # ax.plot(result.times, result.observables[0, :].real)

    # # fig, ax = plt.subplots()
    # # ax.plot(result.times, result.observables[1, :].real)

    # x = np.linspace(-6,6)
    # y = np.linspace(-6,6)
    # rho = solver.nuclear_density(result.psilist[-1], x, y)
    # np.savez('nuclear_density_{}'.format(int(nt*dt)), x, y, rho)
    # fig, ax = plt.subplots()
    # ax.contour(rho.real, levels=40, origin='lower')

    # c = [result.psilist[k][112, :] for k in range(nt//nout)]
    # fig, ax = plt.subplots()
    # ax.plot(result.times, [np.prod(_c).real for _c in c])

    #
    # Sparse Grid
    #
    # from pyqed.smolyak.sg import SparseGrid

    # level = 5
    # dim = 2
    # sg = SparseGrid(dim=dim, level=level, domain=[[-6, 6], ] * dim)

    # index_set, c = sg.combination_technique(4)

    # s = 0
    # for index in index_set:
    #     i, j = index
    #     s += 2**(i+j)
    # print(s)
    # print('index sets', index_set)
    # print('Combination coefficient', c)

    # result = []
    # points = []
    # xAve = 0
    # for l, index in enumerate(index_set):
    #     l1, l2 = index
    #     x = np.linspace(-6, 6, 2**l1, endpoint=False)[1:]
    #     y = np.linspace(-6, 6, 2**l2, endpoint=False)[1:]

    #     print(len(x), interval(x))

    #     # points = genpoints(x, y)
    #     # scatter(points)

    #     # call SPO solver
    #     v = dpes(x, y)

    #     sol = SincDVR(x, y)
    #     sol.v = v

    #     # X, Y = np.meshgrid(x, y)
    #     nx, ny = len(x), len(y)
    #     # ntot = nx * ny
    #     # grid = np.asarray([X.reshape(ntot), Y.reshape(ntot)]).T

    #     psi0 = np.zeros((nx, ny, 2),dtype=complex)
    #     for i in range(nx):
    #         for j in range(ny):
    #             psi0[i, j, 1] = gwp([x[i], y[j]], x0=[-1.0, 0], ndim=2)

    #     r = sol.run(psi0=psi0, dt=0.25, nt=80)
    #     x = r.position(plot=True)
    #     # P = r.population()
    #     # psilist += c[l] * r.psilist
    #     x = np.array(x)
    #     xAve += c[l] * x

    # np.save('x', xAve)
    # # sg.printGrid()
    # # xAve = 0
    # # for i in range(len(index_set)):
    # #     xAve += c[i] * result[i]

    # # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(xAve[0, :].real)
    # ax.plot(xAve[1, :].real)
    # ax.set_title('Sparse Grid')