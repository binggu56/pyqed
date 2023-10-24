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
import matplotlib.pyplot as plt
import math


from pyqed import boson, interval, sigmax, sort, ket2dm, overlap,\
    polar2cartesian, Mol, SESolver, dag, SPO, SPO2, SPO3
from pyqed.style import set_style
# from pyqed.units import au2ev, wavenumber2hartree

def hermite(x, nb):
    """
    hermite polynomials
    return:
        H: list of Hermite polynomials up to order Nb
    """
    cons = np.array([1. / np.sqrt(float(2**n) * float(math.factorial(n))) for n in range(nb)])

    H = []
    H.append(1.0)
    H.append( x * 2.0 )
    if nb > 2:
        for n in range(2,nb):
            Hn = 2.0 * x * H[n-1] - 2.0*(n-1) * H[n-2]
            H.append(Hn)

    for n in range(nb):
        H[n] = H[n]*cons[n]

    return H

def gauss_hermite2d(x,y,nx,ny):
    """
    composite basis for two harmonic oscillators
    """
    gx = gauss_hermite(x,nx)
    gy = gauss_hermite(y,ny)

    tmp = []
    for i in range(nx):
        for j in range(ny):
            tmp.append(np.einsum('i, j -> ij', gx[i], gy[j]))
    return tmp

def gauss_hermite(x, nb, alpha=1., xAve=0., pAve=0.0):
   """
   compute the value of Gauss-Hermite at x
   """
   a = alpha.real
   z = (x - xAve) * np.sqrt(a)

   gauss = (a/np.pi)**0.25 * np.exp( - alpha * (x-xAve)**2/2.0 \
            + 1j*pAve*(x-xAve))

   gh = [ gauss*h for h in hermite(z, nb)]
   return gh

def fock2grid(state, x, y, nx, ny):
    """
    transform a state or a density matrix in the Fock space to the coordinate space
    nuclear density at position R
    
    state: 1d or 2d array
        density matrix for the nuclear dofs
    
    Returns
    =======
    psi or den
    """
    # construct basis
    gh = gauss_hermite2d(x, y, nx, ny)
    
    nb = nx*ny
    
    if state.shape == (nb,):
        
        psi = np.zeros((len(x),len(y)), dtype=complex)

        for i in range(nb):
            psi += gh[i] * state[i] 

        return psi 
    
    elif state.shape == (nb, nb):
        
        den = np.zeros((len(x),len(y)), dtype=complex)
    
        for i in range(nb):
            for j in range(nb):
                den += gh[i] * state[i,j] * gh[j]

        return den
    else:
        raise ValueError('Not a valid state.')


class Vibronic2:
    """
    vibronic model in the diabatic representation with 2 nuclear coordinates

    """
    def __init__(self, x, y, mass=[1,1], nstates=2, nmodes=2, coords='linear'):
        self.x = x
        self.y = y
        self.X, self.Y = meshgrid(x, y)
        self.nstates = nstates

        self.nx = len(x)
        self.ny = len(y)
        self.dx = interval(x)
        self.dy = interval(y) # for uniform grids
        self.mass = mass
        self.kx = None
        self.ky = None
        self.dim = 2
        self.mass = mass


        self.v = None # diabatic PES
        self.apes = None

        # self.h = h
        # self.l = l

    def set_grid(self, x, y):
        self.x = x
        self.y = y

        return

    def set_mass(self, mass):
        self.mass = mass

    # def diabatic(self, x, y):
        
    #     nstates = self.nstates
    #     kappa = self.kappa 
    #     v = np.zeros((nstates, nstates))
    #     v[0, 0] = 
    #     return v
        
    def buildV(self, vd):
        
        nx = self.nx 
        ny = self.ny 
        x = self.x
        y = self.y 
        
        
        for i in range(nx):
            for j in range(ny):
                self.v[i, j, :, :] = vd[[x[i], y[j]]]
        
        return v
        
    
    def set_DPES(self, surfaces, diabatic_couplings, abc=False, eta=None):
        """
        set the diabatic PES and vibronic couplings

        Parameters
        ----------
        surfaces : TYPE
            DESCRIPTION.
        diabatic_couplings : TYPE
            DESCRIPTION.
        abc : boolean, optional
            indicator of whether using absorbing boundary condition. This is
            often used for dissociation.
            The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """

        nx = self.nx
        ny = self.ny
        ns = self.ns

        # DPES and diabatic couplings
        v = np.zeros([nx, ny, ns, ns])

        # assume we have analytical forms for the DPESs
        for a in range(self.ns):
            v[:, :, a, a] = surfaces[a]

        for dc in diabatic_couplings:
            a, b = dc[0][:]
            v[:, :, a, b] = v[:, :, b, a] = dc[1]


        if abc:
            for n in range(self.ns):
                v[:, :, n, n] = -1j * eta * (self.X - 9.)**2

        self.v = v
        return v

    def adiabats(self, x, y):
        """
        Compute the adiabatic potential energy surfaces from diabats

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        
        w, u = np.linalg.eigh(self.v)

        return w, u
    
    def get_apes(self):
        """
        Compute the adiabatic potential energy surfaces from diabats

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        x = self.x
        y = self.y
        assert(x is not None)
        
        nstates = self.nstates
        
        nx = len(x)
        ny = len(y)
        
        v = np.zeros((nx, ny, nstates))
        
        for i in range(nx):
            for j in range(ny):
                w, u = self.apes([x[i], y[j]])
                v[i, j, :] = w
        
        self.apes = v 
        return v

    def plot_apes(self):
        """
        plot the APES
        """
        if self.apes is None:
            self.get_apes()
        
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.contourf(self.X, self.Y, self.apes[:, :, 1], lw=0.7)
        ax1.contourf(self.X, self.Y, self.apes[:, :, 0], lw=0.7)
        return

    def plot_dpes(self, style='2D'):
        if style == '2D':
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.contourf(self.X, self.Y, self.v[:, :, 1, 1], lw=0.7)
            ax1.contourf(self.X, self.Y, self.v[:, :, 0, 0], lw=0.7)
            return

        else:
            from pyqed.style import plot_surface
            plot_surface(self.x, self.y, self.V[:,:,0,0])

            return

    def plt_wp(self, psilist, **kwargs):


        if not isinstance(psilist, list): psilist = [psilist]


        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(self.X, self.Y, np.abs(psi[:,:, 1])**2)
            ax1.contour(self.X, self.Y, np.abs(psi[:, :,0])**2)
            ax0.format(**kwargs)
            ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1

    def spo(self):
        return SPO2(nstates=self.nstates, mass=self.mass, x=self.x, y=self.y)


class LVC2:
    """
    2D linear vibronic coupling model
    """
    def __init__(self, x=None, y=None, h=1, l=1, delta=0, mass=[1, 1], nstates=2):
        # super().__init__(x, y, mass, nstates=nstates)

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

        v[:, :, 0, 0] = X**2/2. + (Y)**2/2. + h * X + delta/2
        v[:, :, 1, 1] = X**2/2. + (Y)**2/2. - h * X - delta/2
        v[:, :, 0, 1] = l * Y
        v[:, :, 1, 0] = l * Y


        self.v = v
        return

    def dpes(self, x):

        X, Y = x
        h, l = self.h, self.l
        delta = self.delta

        v = np.zeros((self.nstates, self.nstates))

        # v[0, 0] = (X)**2/2. + (Y)**2/2. + h * X + delta
        # v[1, 1] = (X)**2/2. + (Y)**2/2. - h * X - delta
        v[0, 0] =  + h * X + delta
        v[1, 1] =  - h * X - delta

        v[0, 1] =   l * Y
        v[1, 0] = v[0, 1]

        # self.v = v
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

        for i in range(nx):
            for j in range(ny):
                w, u = self.apes([x[i], y[j]])
                v[i, j, :] = w

        return v

    def plot_apes(self):

        v = self.apes_global()
        mayavi([v[:,:,k] for k in range(self.nstates)])


    def run(self):
        pass
    
    def buildH(self, representation='ho'):
        # compute the Hamiltonian using DVR or harmonic oscillator eigenstates
        pass
        # return H


class DHO:
    def __init__(self, x, mass=1, nstates=2, coupling=0):

        # self.dpes()
        self.edip = sigmax()
        self.nstates = nstates
        self.mass = mass
        self.x = x
        self.nx = len(x)
        # self.d = d # displacement
        self.coupling = coupling # vibronic coupling strength
        assert(self.edip.ndim == nstates)


    def dpes(self, d, e0):

        x = self.x
        nx = self.nx
        N = self.nstates

        v = np.zeros((nx, N, N))

        v[:, 0, 0] = x**2/2. 
        v[:, 1, 1] = (x-d)**2/2. + e0

        self.v = v
        return

    def apes(self, x):
        d = self.d
        c = self.coupling

        x = np.atleast_1d(x)
        nx = len(x)

        ns = self.nstates

        wlist = []
        ulist = []

        for i in range(nx):
            v = np.zeros((ns, ns))

            v[0, 0] = x[i]**2/2.
            v[1, 1] = (x[i]-d)**2/2.
            v[0, 1] = v[1, 0] = c

            w, u = np.linalg.eigh(v)
            wlist.append(w.copy())
            ulist.append(u.copy())

        return wlist, ulist
    
    
class VibronicAdiabatic(Mol):
    """
    1D vibronic model in the adiabatic representation 
    """
    def __init__(self, x=None, mass=1, nstates=2, edip=None, mdip=None, equad=None, \
                 nac=None):
        """

        Parameters
        ----------
        E : 1d array
            electronic energy at ground-state minimum
        modes : list of Mode objs
            vibrational modes

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.x = x
        self.nx = len(x)
        self.mass = mass
        self.nel = self.nstates = nstates
        # self.nmodes = len(modes)
        # self.modes = modes
        # self.truncate = None
        # self.fock_dims = [m.truncate for m in modes]
        # self.nvib = np.prod(self.fock_dims)

        # self.idm_vib = identity(self.nvib) # vibrational identity
        # self.idm_el = identity(self.nstates) # electronic identity
        # self.omegas = [mode.omega for mode in self.modes]


        # self.H = None
        self._v = None # adiabatic PES
        # self.dim = None
        # self._x = None # list of coordinate operators
        
        self.edip = edip 
        self.mdip = mdip 
        self.nac = nac
        
    # def buildH(self):
    #     """
    #     Calculate the vibronic Hamiltonian.

    #     Parameters
    #     ----------
    #     nums : list of integers
    #         size for the Fock space of each mode

    #     Returns
    #     -------
    #     2d array
    #         Hamiltonian

    #     """

    #     omegas = self.omegas
    #     nmodes = self.nmodes

    #     # identity matrices in each subspace
    #     nel = self.nstates
    #     I_el = identity(nel)

    #     h_el = np.diagflat(self.e_fc)

    #     # calculate the vibrational Hamiltonian
    #     # hv, xs = multimode(omegas, nmodes, truncate=self.fock_dims[0])

    #     # bare vibronic H in real e-states
    #     # H = kron(h_el, identity(hv.shape[0])) + kron(I_el, hv)



    #     # vibronic coupling, tuning + coupling
    #     for j, mode in enumerate(self.modes):
    #         # n = mode.truncate

    #         # # vibrational Hamiltonian
    #         # hv = boson(mode.omega, n, ZPE=False)

    #         # H = kron(H, Iv) + kron(identity(H.shape), hv)
    #         V = 0.
    #         for c in mode.couplings:
    #             a, b = c[0]
    #             strength = c[1]
    #             V += strength * jump(a, b, nel)

    #         H += kron(V, xs[j])

    #     self.H = H
    #     self.dim = H.shape[0]
    #     self._x = xs

        # return self.H

    # def APES(self, x):

    #     V = np.diag(self.e_fc)

    #     # for n in range(self.nmodes):
    #     V += 0.5 * np.sum(self.omegas * x**2) * self.idm_el

    #     # V += tmp * self.idm_el

    #     for j, mode in enumerate(self.modes):
    #         for c in mode.couplings:
    #             a, b = c[0]
    #             strength = c[1]
    #             V += strength * jump(a, b, self.nstates) * x[j]

    #     E = np.linalg.eigvals(V)
    #     return np.sort(E)

    # def calc_edip(self):
    #     pass

    # def promote(self, A, which='el'):

    #     if which in ['el', 'e', 'electronic']:
    #         A = kron(A, self.idm_vib)
    #     elif which in ['v', 'vib', 'vibrational']:
    #         A = kron(self.idm_el, A)

    #     return A

    # def vertical(self, n=1):
    #     """
    #     generate the initial state created by vertical excitation

    #     Parameters
    #     ----------
    #     n : int, optional
    #         initially excited state. The default is 1.

    #     Returns
    #     -------
    #     psi : TYPE
    #         DESCRIPTION.

    #     """
    #     psi = basis(self.nstates, n)

    #     dims = self.fock_dims

    #     chi = basis(dims[0], 0)

    #     for j in range(1, self.nmodes):
    #         chi = np.kron(chi, basis(dims[j], 0))

    #     psi = np.kron(psi, chi)

    #     return psi

    def vibrational_eigenstates(self, n=0, lha=False):
        """
        compute the vibrational eigenstates of n-th PES by DVR
        
        lha: bool
            local harmonic approximation. If true, the ground state will be a Gaussian.
            
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        pass

    # def buildop(self, i, f=None, isherm=True):
    #     """
    #     construct electronic operator

    #         \ket{f}\bra{i}

    #     if isherm:
    #         return \ket{f}\bra{i} + \ket{i}\bra{f}

    #     Parameters
    #     -------
    #     i: int
    #         initial state.
    #     f: int, optional
    #         final state. Default None. If None, set f = i.

    #     isherm: bool, optional
    #         indicator of whether the returned matrix is Hermitian or not
    #         Default: True

    #     Returns
    #     -------
    #     2darray
    #         DESCRIPTION.

    #     """
    #     if f is None:
    #         f = i

    #     p = jump(i=i, f=f, dim=self.nstates, isherm=isherm)

    #     return kron(p, self.idm_vib)
    
    # def coordinate(self, n):
    #     """
    #     build coordinate operators in the full space 

    #     Parameters
    #     ----------
    #     n : int
    #         mode id

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """
    #     return kron(self.idm_el, self._x[n])
        
    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, v):
        """
        set up the adiabatic PESs

        Parameters
        ----------
        v : ndarray [nx, nel, nel]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print(v.shape)
        assert(v.shape == (self.nx, self.nel, self.nel))
        self._v = v
        
        

    def run(self, psi0, dt, nt, method='SPO'):
        
        if self.nac is not None and method == 'SPO':
            return ValueError('Method SPO cannot be used for nonadiabatic couplings.')

        if method == 'RK4':

            sol = SESolver()

            if self.H is None:
                self.buildH()

            sol.H = self.H

            sol.ground_state = self.get_ground_state()

            return sol

        elif method == 'SPO':

            from pyqed.namd.diabatic import SPO

            sol = SPO(self.x, mass=self.mass, nstates=self.nstates, v=v)

            # sol.V = mol.v
            
            return sol.run(psi0=psi0, dt=dt, nt=nt) 

        #     elif self.nmodes == 2:
        #         from pyqed.wpd import SPO2

        #         sol = SPO2()

        else:
            raise ValueError('The number of modes {} is not \
                             supported.'.format(self.nmodes) )




            return sol

    def rdm_el(self, psi):
        """
        Compute the electronic reduced density matrix.

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        psi_reshaped = np.reshape(psi, (self.nel, self.nvib))

        return psi_reshaped.dot(dag(psi_reshaped))

    def add_coupling(self, coupling):
        """
        add additional coupling terms to the Hamiltonian such as Stark
        and Zeeman effects

        Parameters
        ----------
        coupling : list, [[a, b], strength]
            describe the coupling, a, b labels the electronic states

        Returns
        -------
        ndarray
            updated H.

        """
        a, b = coupling[0]
        strength = coupling[1]

        self.H += strength * kron(jump(a, b, self.nel),  self.idm_vib)

        return self.H

    def plot_PES(self, x, y):
        """
        plot the 3D adiabatic potential energy surfaces

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        from pyqed.style import plot_surfaces

        if self.nmodes != 2:
            raise ValueError('This function only works for nmodes=2.')

        nx, ny = len(x), len(y)
        E = np.zeros(nx, ny, self.nstates)

        for i in range(nx):
            for j in range(ny):
                E[i, j, :] = self.APES([x[i], y[j]])

        return plot_surfaces(x, y, [E[:,:, 1], E[:,:, 2]])
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

    def test_CI():
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
    
    from pyqed import gwp
    x = np.linspace(-2, 2)
    nx = len(x)
    mol = VibronicAdiabatic(x, nstates=2)
    
    v = np.zeros((nx, 2, 2))
    v[:, 0, 0] = x**2/2
    v[:, 1, 1] = x**2/2
    
    mol.v = v 
    
    psi0 = np.zeros((nx, 2), dtype=complex) 
    psi0[:, 0] = gwp(x, a=1)
    mol.run(psi0, dt=0.2, nt=1)