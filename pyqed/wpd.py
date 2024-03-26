#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:44:15 2021

Exact nonadiabatic wavepacket dynamics solver for vibronic models with N vibrational modes
(N = 1 ,2)

For linear coordinates, use SPO method
For curvilinear coordinates, use RK4 method

@author: Bing Gu
"""

import numpy as np

try:    
    import proplot as plt
except:
    import matplotlib.pyplot as plt

from numpy import cos, pi
# from numba import jit
import scipy
from scipy.fftpack import fft2, ifft2, fftfreq, fft, ifft
from numpy.linalg import inv, det

from pyqed import rk4, dagger, gwp, interval, meshgrid, norm2, dag, sort
from pyqed.units import au2fs
from pyqed.mol import Result

from pyqed.nonherm import eig





def plot_wavepacket(x, y, psilist, **kwargs):

    if not isinstance(psilist, list): psilist = [psilist]
    
    X, Y = np.meshgrid(x, y)

    for i, psi in enumerate(psilist):
        fig, ax0 = plt.subplots(nrows=1, sharey=True)
        # levels = np.linspace(0, 0.005, 20)
        ax0.contourf(X, Y, np.abs(psi)**2, colorbar='r', cmap='viridis')

        # levels = np.linspace(0, 0.0005, 20)
        ax0.format(ylim=(-0.5, 0.5), **kwargs)
        fig.savefig('vibrational_eigenstates_D0_'+str(i)+'.png')

    return ax0



class ResultSPO2(Result):
    def __init__(self, **args):
        super().__init__(**args)

        self.x = None
        self.y = None
        self.population = None
        self.xAve = None
        self.nstates = self.psi0.shape[-1]

    def plot_wavepacket(self, psilist, state_id=None, **kwargs):

        X, Y = np.meshgrid(self.x, self.y, indexing='ij')

        if not isinstance(psilist, list): psilist = [psilist]

        if isinstance(state_id, int):

            for i, psi in enumerate(psilist):
                fig, ax0 = plt.subplots(nrows=1)

                ax0.contour(X, Y, np.abs(psi[:,:, state_id])**2)
                ax0.format(**kwargs)

                fig.savefig('psi'+str(i)+'.pdf')

            return ax0

        else:
            
            nstates = self.psi0.shape[-1]

            for i, psi in enumerate(psilist):

                fig, axes = plt.subplots(nrows=nstates, sharey=True, sharex=True,\
                                         figsize=(3.5,4))
                
                for n in range(nstates):
                    axes[n].contourf(X, Y, np.abs(psi[:,:, n])**2)
                    
                # ax1.contour(X, Y, np.abs(psi[:, :,0])**2)
                    # axes[n].format(**kwargs)
                # ax1.format(**kwargs)
                fig.savefig('psi'+str(i)+'.pdf')

            return axes
        
    def get_population(self, fname=None, plot=False):
        dx = interval(self.x)
        dy = interval(self.y)
        p = np.zeros((len(self.psilist), self.nstates))
        for n in range(self.nstates):
            p[:, n] = [norm2(psi[:, :, n]).real * dx * dy for psi in self.psilist]
        # p1 = [norm2(psi[:, :, 1]) * dx * dy for psi in self.psilist]
        

            # ax.plot(self.times, p1)
        
        self.population = p
        if fname is not None:
            # fname = 'population'
            np.savez(fname, p)
            
        if plot:
            fig, ax = plt.subplots()
            for n in range(self.nstates):
                ax.plot(self.times, p[:,n])
            return fig, ax
        
        else:   
            return p
    
    def plot_population(self, p):
        
        fig, ax = plt.subplots()
        for n in range(self.nstates):
            ax.plot(self.times, p[:,n])
            
        return fig, ax
        
    
    def position(self, plot=False, fname=None):        
        # X, Y = np.meshgrid(self.x, self.y)
        x = self.x 
        y = self.y
        dx = interval(x)
        dy = interval(y)
        
        xAve = [np.einsum('ijn, i, ijn', psi.conj(), x, psi) * dx*dy for psi in self.psilist]
        yAve = [np.einsum('ijn, j, ijn', psi.conj(), y, psi) * dx*dy for psi in self.psilist]
        
        xAve = np.real_if_close(xAve)
        yAve = np.real_if_close(yAve)
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.times, xAve)
            ax.plot(self.times, yAve)
        
        self.xAve = [xAve, yAve]
        np.savez('xAve', xAve, yAve)
        
        return xAve, yAve
        
        
    def dump(self, fname):
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
            pickle.dump(self, f)
            
            
class Solver():
    def __init__(self):
        self.obs_ops = None
        self.grid = None

    def set_obs_ops(self, obs_ops):
        self.obs_ops = obs_ops
        return


class SPO:
    def __init__(self, x, mass=1, ns=2):
        self.x = x
        self.dx = interval(x)
        self.nx = len(x)
        self.k = 2. * pi * fftfreq(self.nx, self.dx)
        self.ns = ns

        self.V = None
        self.mass = mass
        self._exp_K = None
        self._exp_V = None
        self._exp_V_half = None


    def set_grid(self, xmin=-1, xmax=1, npts=32):
        self.x = np.linspace(xmin, xmax, npts)

    def set_potential(self, potential):
        self.V = potential(self.x)
        return

    def build(self, dt):
        self._exp_V = np.exp(- 1j * self.V * dt)
        self._exp_V_half = np.exp(-1j * self.V * dt/2.)
        m = self.mass
        k = self.k
        self._exp_K = np.exp(-0.5j / m * (k * k) * dt)
        return

    def run(self, psi0, dt, Nt=1, t0=0, nout=1):

        """
        Time-dependent Schrodinger Equation for wavepackets on a single PES.

        Parameters
        ----------
        psi0: 1d array, complex
            initial wavepacket
        t0: float
            initial time
        dt : float
            the small time interval over which to integrate
        nt : float, optional
            the number of intervals to compute.  The total change
            in time at the end of this method will be dt * Nsteps.
            default is N = 1
        """
        from pyqed import Result

        self.build(dt)

        t = t0
        psi_x = psi0.copy()

        r = Result(psi0=psi0, dt=dt, Nt=Nt, t0=t0, nout=nout)

        # SPO propagation
        psi_x = self.x_evolve_half(psi_x)

        for i in range(1, Nt//nout):
            for k in range(nout):
                t += dt

                psi_x = self.k_evolve(psi_x)
                psi_x = self.x_evolve(psi_x)

            r.psilist.append(psi_x.copy())

            # f.write('{} {} {} {} {} \n'.format(t, *rho))

        psi_x = self.k_evolve(psi_x)
        psi_x = self.x_evolve_half(psi_x)

        t += dt
        # f.close()
        r.psi = psi_x

        return r

    def x_evolve(self, psi):
        """
        one time step for exp(-i * V * dt)

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._exp_V * psi

    def x_evolve_half(self, psi):
        """
        one time step for exp(-i * V * dt)

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._exp_V_half * psi

    def k_evolve(self, psi_x):
        """
        one time step for exp(-i * K * dt)
        """

        psi_k = fft(psi_x)
        #psi_k = fftshift(psi_k)

        # psi_k *= np.exp(-0.5 * 1j / m * (k * k) * dt)
        psi_k *= self._exp_K

        return ifft(psi_k)




# def adiabatic_1d(x, v, psi0, dt, Nt=1, t0=0.):
#     """
#     Time-dependent Schrodinger Equation for wavepackets on a single PES.

#     Parameters
#     ----------
#     psi0: 1d array, complex
#         initial wavepacket
#     t0: float
#         initial time
#     dt : float
#         the small time interval over which to integrate
#     nt : float, optional
#         the number of intervals to compute.  The total change
#         in time at the end of this method will be dt * Nsteps.
#         default is N = 1
#     """

#     f = open('density_matrix.dat', 'w')
#     t = t0
#     psi_x = psi0.copy()
#     dt2 = 0.5 * dt

#     N = len(x)
#     dx = interval(x)

#     k = 2. * pi * scipy.fftpack.fftfreq(N, dx)
#     # k[:] = 2.0 * np.pi * k[:]


#     # SPO propagation
#     x_evolve(dt2, x, v, psi_x)

#     for i in range(nt - 1):

#         t += dt

#         psi_x = k_evolve(dt, k, psi_x)
#         psi_x = x_evolve(dt, x, v, psi_x)

#         # rho = density_matrix(psi_x, dx)
#         # f.write('{} {} {} {} {} \n'.format(t, *rho))

#     psi_x = k_evolve(dt, k, psi_x)
#     psi_x = x_evolve(dt2, x, v, psi_x)

#     t += dt
#     f.close()

#     return psi_x

class SPO2:
    """
    second-order split-operator method for nonadiabatic wavepacket dynamics
    in the diabatic representation with two-dimensional nuclear coordinate

    For time-independent Hamiltonian

        e^{-i H \Delta t} = e^{- i V \Delta t/2} e^{-i K \Delta t} e^{-iV\Delta t/2}

    For time-dependent H,
        TBI
    """
    def __init__(self, x, y, mass=None, nstates=2, coords='linear', G=None, abc=False):
        self.x = x
        self.y = y
        self.X, self.Y = meshgrid(x, y)

        self.nx = len(x)
        self.ny = len(y)
        self.dx = interval(x)
        self.dy = interval(y) # for uniform grids
        if mass is None:
            mass  = [1, 1]
        self.mass = self.masses = mass
        self.kx = None
        self.ky = None
        
        self.apes = None
        self.dim = 2
        self.exp_V = None
        self.exp_V_half = None
        self.exp_K = None
        self.v = self.V = None
        self.G = G
        self.nstates = self.ns = nstates
        self.coords =  coords
        self.abc = abc
        
        self.d2a = None # diabatic to adiabatic transformation
        self.a2d = None # adiabatic to diabatic transformation

    def set_grid(self, x, y):
        self.x = x
        self.y = y

        return

    def set_masses(self, mass):
        self.mass = mass


    def setG(self, G):
        self.G = G

    def set_DPES(self, surfaces, diabatic_couplings, eta=None):
        """
        set the potential energy operatpr from the diabatic PES and vibronic couplings

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

        # nx = len(x)
        # ny = len(y)
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
            v[:, :, a, b] = dc[1] 
            v[:, :, b, a] = v[:, :, a, b].conj()


        if self.abc:
            for n in range(self.ns):
                v[:, :, n, n] = -1j * eta * (self.X - 9.)**2

        self.V = v
        return v
    
    def set_dpes(self, v):
        self.V = self.v = v
        return 


    def build(self, dt, inertia=None):
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
        nx = self.nx
        ny = self.ny
        nstates = self.nstates

        dx = interval(self.x)
        dy = interval(self.y)

        self.kx = 2. * np.pi * fftfreq(nx, dx)
        self.ky = 2. * np.pi * fftfreq(ny, dy)


        if self.coords == 'linear':

            mx, my = self.masses

            Kx, Ky = meshgrid(self.kx, self.ky)

            self.exp_K = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

        elif self.coords == 'jacobi':

            # self.exp_K = np.zeros((nx, ny, nx, ny))
            mx = self.masses[0]

            self.exp_Kx = np.exp(-1j * self.kx**2/2./mx * dt)

            Iinv = 1./self.masses[1](self.x) # y is the angle
            ky = self.ky

            self.exp_Ky = np.exp(-1j * np.outer(Iinv, ky**2/2.) * dt)

            # psik = fft(typsi, axis=0)
            # kpsi = np.einsum('i, ija -> ija', np.exp(-1j * kx**2/2./mx * dt), psik)

            # # for i in range(nx):
            # #     for j in range(ny):
            # #         my = self.masses[1](y[j])
            # #         self.exp_K[i, j, :, :] = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)


        dt2 = 0.5 * dt

        if self.v is None:
            raise ValueError('The diabatic PES is not specified.')

        v = self.v

        self.exp_V = np.zeros(v.shape, dtype=complex)
        self.exp_V_half = np.zeros(v.shape, dtype=complex)
        # self.apes = np.zeros((nx, ny))
        
        # if self.abc:
        #     eig = scipy.linalg.eig
        # else:
        #     eig = scipy.linalg.eigh
        
        
        if np.iscomplexobj(v): # complex Hermitian H
            
            self.d2a = np.zeros((nx, ny, nstates, nstates), dtype=complex)

            # complex potential
            for i in range(nx):
                for j in range(ny):
                    
                    _v = v[i, j]
                    
                    # w, ul, ur = scipy.linalg.eig(_v, left=True, right=True)
                    w, u = sort(*scipy.linalg.eigh(_v))
    
                    V = np.diagflat(np.exp(- 1j * w * dt))
                    V2 = np.diagflat(np.exp(- 1j * w * dt2))
    
                    self.exp_V[i, j, :,:] = u.dot(V.dot(dagger(u)))
                    self.exp_V_half[i, j, :,:] = u.dot(V2.dot(dagger(u)))
                    
                    self.d2a[i, j] = u
                    
        else: 
            
            self.d2a = np.zeros((nx, ny, nstates, nstates))
            self.apes = np.zeros((nx, ny, nstates))
            
            for i in range(nx):
                for j in range(ny):
    
                    w, u = sort(*scipy.linalg.eigh(v[i, j, :, :]))
    
                    #print(np.dot(U.conj().T, Vmat.dot(U)))
                    self.apes[i, j] = w 
                    
                    V = np.diagflat(np.exp(- 1j * w * dt))
                    V2 = np.diagflat(np.exp(- 1j * w * dt2))
    
                    self.exp_V[i, j, :,:] = u.dot(V.dot(dagger(u)))
                    self.exp_V_half[i, j, :,:] = u.dot(V2.dot(dagger(u)))
                    
                    self.d2a[i, j] = u

        return

    def population(self, psi, representation='diabatic', plot=False):
        """
        return the electronic populations

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        2darray, nt, nstates

        """
        if representation == 'diabatic':
            
            if isinstance(psi, np.ndarray):
                P = np.zeros(self.ns)
                for j in range(self.ns):
                    P[j] = np.linalg.norm(psi[:, :, j])**2 * self.dx * self.dy
    
                assert(np.close(np.sum(P), 1))
    
            elif isinstance(psi, list):
    
                N = len(psi)
                P = np.zeros((N, self.nstates))
                for k in range(N):
                    P[k, :] = [np.linalg.norm(psi[k][:, :, j])**2 * self.dx * self.dy \
                               for j in range(self.nstates)]
        
        elif representation == 'adiabatic':
            
            assert(self.d2a is not None)
            
            if isinstance(psi, np.ndarray):
                
                psi = np.einsum('ijab, ijb -> ija', self.d2a, psi)
                
                P = np.zeros(self.ns)
                for j in range(self.ns):
                    P[j] = np.linalg.norm(psi[:, :, j])**2 * self.dx * self.dy
    
                assert(np.close(np.sum(P), 1))
    
            elif isinstance(psi, list):
                
                psi = [np.einsum('ijab, ijb -> ija', self.d2a, phi) \
                       for phi in psi]
    
                N = len(psi)
                P = np.zeros((N, self.nstates))
                for k in range(N):
                    P[k, :] = [np.linalg.norm(psi[k][:, :, j])**2 * self.dx * self.dy \
                               for j in range(self.nstates)]
            
        
        else:
            raise ValueError('Representation = {}, which can only be \
                             diabatic or adiabatic'.format(representation))
    
        
        return P


    def run(self, psi0, e_ops=[], dt=0.01, nt=1, t0=0., nout=1, return_states=True):

        print('Building the propagators ...')

        self.build(dt=dt)

        psi = psi0.copy()

        def _V_half(psi):

            return np.einsum('ijab, ijb -> ija', self.exp_V_half, psi) # evolve V half step

        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0]

        t = t0
        if self.coords == 'linear':

            KEO = self._KEO_linear

        elif self.coords == 'jacobi':

            KEO = self._KEO_jacobi

        # observables
        if return_states:

            for i in range(nt//nout):
                for n in range(nout):

                    t += dt

                    psi = _V_half(psi)
                    psi = KEO(psi)
                    psi = _V_half(psi)

                r.psilist.append(psi.copy())

        else:

            psi = _V_half(psi)

            for i in range(nt//nout):
                for n in range(nout):
                    t += dt

                    psi = KEO(psi)
                    psi = self._PEO(psi)


                r.psilist.append(psi.copy())
                # tlist.append(t)

                # rho = density_matrix(psi_x, dx)

                # # store the density matrix
                # f.write('{} {} {} {} {} \n'.format(t, *rho))

            psi = KEO(psi)
            psi = np.einsum('ijab, ijb -> ija', self.exp_V_half, psi) # evolve V half step

        r.psi = psi
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
    
    def current_density(self, psi, state_id=0):
        """
        Compute the velocity field of the vibrational flow

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.
        state_id : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        x, y = self.x, self.y
        
        chi  = psi[:, :, state_id]
        rx, ry = np.gradient(np.log(np.abs(chi)), self.dx, self.dy)
        px, py = np.gradient(np.angle(chi), self.dx, self.dy, edge_order=2)
        
        
        fig, ax = plt.subplots(ncols=1)
        ax.contourf(x, y, np.abs(chi))
        ax.quiver(self.x, self.y, rx, ry, transpose=True)
        # ax1.quiver(self.x, self.y, rx, ry)
        
        
        # divergence of current
        div = divergence([rx, ry], [self.dx, self.dy])
        
        fig, ax = plt.subplots()
        ax.imshow(div)
        return [px, py], [rx, ry]

    def _PEO(self, psi):

        vpsi = np.einsum('ijab, ijb -> ija', self.exp_V, psi)
        return vpsi

    def _KEO_linear(self, psi):
        # psik = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     psik[:,:,j] = fft2(psi[:,:,j])
        psik = fft2(psi, axes=(0,1))
        kpsi = np.einsum('ij, ija -> ija', self.exp_K, psik)

        # out = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     out[:, :, j] = ifft2(kpsi[:, :, j])
        psi = ifft2(kpsi, axes=(0,1))
        return psi

    def _KEO_jacobi(self, psi):
        """
        The current implementation is of order $\mc{O}(\Delta t^2)$. For improvements,
        we can use the splitting

        K_1(\Delta t/2)K_2(\Delta t/2) V(\Delta t) K_2(\Delta t/2) K_1(\Delta t/2)

        For time-independent H, this is of order $\mc{O}(\Delta t^3)$.

        see Hermann, M. R. & Fleck, J. A.  Phys. Rev. A 38, 6000–6012 (1988).

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        psi : TYPE
            DESCRIPTION.


        """

        psi_ky = fft(psi, axis=1)
        typsi = np.einsum('ij, ija -> ija', self.exp_Ky, psi_ky)
        # psik = ifft(typsi, axis=1)
        # psik = fft(psik, axis=0)
        # kpsi = np.einsum('i, ija -> ija', self.exp_Kx, psik)
        # psi = ifft(kpsi, axis=0)

        psik = fft(typsi, axis=0)
        kpsi = np.einsum('i, ija -> ija', self.exp_Kx, psik)

        psi = ifft2(kpsi, axes=(0,1))


        return psi

    def plot_surface(self, style='2D'):
        if style == '2D':
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.contourf(self.X, self.Y, self.V[:, :, 1, 1], lw=0.7, cmap='viridis')
            ax1.contourf(self.X, self.Y, self.V[:, :, 0, 0], lw=0.7, cmap='viridis')
            return

        else:
            from pyqed.style import plot_surface
            plot_surface(self.x, self.y, self.V[:,:,0,0])

            return

    def plt_wp(self, psilist, **kwargs):


        if not isinstance(psilist, list): psilist = [psilist]


        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(self.X, self.Y, np.abs(psi[:,:, 1])**2, cmap='viridis')
            ax1.contour(self.X, self.Y, np.abs(psi[:, :,0])**2, cmap='viridis')
            ax0.format(**kwargs)
            ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1




class SPO2NH(SPO2):
    
    def __init__(self, x, y, *args, **kwargs):
        self.right_eigenstates = None
        super().__init__(x, y, *args, **kwargs)


    
    def build(self, dt):
        nx = self.nx
        ny = self.ny
        nstates = self.nstates

        dx = interval(self.x)
        dy = interval(self.y)

        self.kx = 2. * np.pi * fftfreq(nx, dx)
        self.ky = 2. * np.pi * fftfreq(ny, dy)
        
        v = self.v 
        
        dt2 = 0.5 * dt
        
        if self.coords == 'linear':

            mx, my = self.masses

            Kx, Ky = meshgrid(self.kx, self.ky)

            self.exp_K = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

        elif self.coords == 'jacobi':

            # self.exp_K = np.zeros((nx, ny, nx, ny))
            mx = self.masses[0]

            self.exp_Kx = np.exp(-1j * self.kx**2/2./mx * dt)

            Iinv = 1./self.masses[1](self.x) # y is the angle
            ky = self.ky

            self.exp_Ky = np.exp(-1j * np.outer(Iinv, ky**2/2.) * dt)

        
        self.right_eigenstates = np.zeros((nx, ny, nstates, nstates), dtype=complex)
        self.ovlp_rr = np.zeros((nx, ny, nstates, nstates), dtype=complex)
        
        self.exp_V = np.zeros(v.shape, dtype=complex)
        self.exp_V_half = np.zeros(v.shape, dtype=complex)
        
        # complex potential
        for i in range(nx):
            for j in range(ny):
                
                _v = v[i, j]
                
                # w, ul, ur = scipy.linalg.eig(_v, left=True, right=True)
                w, ur, ul = eig(_v)
                
                self.right_eigenstates[i,j] = ur
                
                self.ovlp_rr[i,j] = dag(ur) @ ur
                
                V = np.diagflat(np.exp(- 1j * w * dt))
                V2 = np.diagflat(np.exp(- 1j * w * dt2))

                self.exp_V[i, j, :,:] = ur @ V @ ul
                self.exp_V_half[i, j, :,:] = ur @ V2 @ ul
                

        return
    
    def position(self, psilist, plot=False):
        
        x = self.x 
        y = self.y
        dx = interval(x)
        dy = interval(y)
        
        S = self.ovlp_rr
        
        xAve = [np.einsum('ijm, i, ijmn, ijn ->', psi.conj(), x, S, psi) * dx*dy for psi in psilist]
        yAve = [np.einsum('ijn, j, ijmn, ijn ->', psi.conj(), y, S, psi) * dx*dy for psi in psilist]

        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(xAve)
            ax.plot(yAve)
        
        self.xAve = [xAve, yAve]
        np.savez('xAve', xAve, yAve)
        
        return xAve, yAve
    
    def run(self, psi0, e_ops=[], dt=0.01, nt=1, t0=0., nout=1, return_states=True):

        print('Building the propagators ...')

        self.build(dt=dt)

        psi = psi0.copy()

        def _V_half(psi):

            return np.einsum('ijab, ijb -> ija', self.exp_V_half, psi) # evolve V half step

        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0]

        t = t0
        if self.coords == 'linear':

            KEO = self._KEO_linear

        elif self.coords == 'jacobi':

            KEO = self._KEO_jacobi

        # observables
        if return_states:

            for i in range(nt//nout):
                for n in range(nout):

                    t += dt

                    psi = _V_half(psi)
                    psi = KEO(psi)
                    psi = _V_half(psi)

                r.psilist.append(psi.copy())

        else:

            psi = _V_half(psi)

            for i in range(nt//nout):
                for n in range(nout):
                    t += dt

                    psi = KEO(psi)
                    psi = self._PEO(psi)


                r.psilist.append(psi.copy())
                # tlist.append(t)

                # rho = density_matrix(psi_x, dx)

                # # store the density matrix
                # f.write('{} {} {} {} {} \n'.format(t, *rho))

            psi = KEO(psi)
            psi = np.einsum('ijab, ijb -> ija', self.exp_V_half, psi) # evolve V half step

        r.psi = psi
        return r



def divergence(f,h):
    """
    div(F) = dFx/dx + dFy/dy + ...
    g = np.gradient(Fx,dx, axis=1)+ np.gradient(Fy,dy, axis=0) #2D
    g = np.gradient(Fx,dx, axis=2)+ np.gradient(Fy,dy, axis=1) +np.gradient(Fz,dz,axis=0) #3D
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i], axis=i) \
                                    for i in range(num_dims)])


def S0(x, y):
    pass

def S1(x, y):
    pass

def diabatic_coupling(x, y):
    pass

class SPO3():
    # def __init__(self, x, y, z, mass=[1, 1, 1]):
    #     self.x = x
    #     self.y = y
    #     self.z = z

    """
    second-order split-operator method for nonadiabatic wavepacket dynamics
    in the diabatic representation with three-dimensional nuclear space

    For time-independent Hamiltonian

    .. math::

        e^{-i H \Delta t} = e^{- i V \Delta t/2} e^{-i K \Delta t} e^{-iV\Delta t/2}

    For time-dependent H,
        # TODO
    """
    def __init__(self, x, y, z, masses, nstates=2, coords='linear', G=None, abc=False):
        self.x = x
        self.y = y
        self.z = z
        self.X, self.Y, self.Z = meshgrid(x, y, z)

        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        self.dx = interval(x)
        self.dy = interval(y) # for uniform grids
        self.dz = interval(z)

        self.masses = masses
        self.kx = None
        self.ky = None
        self.kz = None
        self.dim = 3
        self.exp_V = None
        self.exp_V_half = None
        self.exp_K = None
        self.V = None
        self.G = G
        self.nstates = nstates
        self.coords =  coords
        self.abc = abc

    def set_grid(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_masses(self, masses):
        self.masses = masses


    def setG(self, G):
        self.G = G

    def set_DPES(self, surfaces, diabatic_couplings, eta=None):
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
        nz = self.nz
        ns = self.nstates

        # DPES and diabatic couplings
        v = np.zeros([nx, ny, nz, ns, ns])

        # assume we have analytical forms for the DPESs
        for a in range(self.nstates):
            v[:, :, :, a, a] = surfaces[a]

        for dc in diabatic_couplings:
            a, b = dc[0][:]
            v[:, :, :, a, b] = v[:, :, :, b, a] = dc[1]


        self.V = v
        return v

    def set_abc(self):
        #set the absorbing boundary condition
        #     for n in range(self.ns):
        #         v[:, :, n, n] = -1j * eta * (self.X - 9.)**2
        pass
    
    def build(self, dt, inertia=None):
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
        nx = self.nx
        ny = self.ny
        nz = self.nz

        dx = interval(self.x)
        dy = interval(self.y)
        dz = self.dz

        self.kx = 2. * np.pi * fftfreq(nx, dx)
        self.ky = 2. * np.pi * fftfreq(ny, dy)
        self.kz = 2. * np.pi * fftfreq(nz, dz)


        if self.coords == 'linear':

            mx, my, mz = self.masses

            Kx, Ky, Kz = meshgrid(self.kx, self.ky, self.kz)

            self.exp_K = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my + Kz**2/2./mz) * dt)
            
            
        elif self.coords == 'jacobi':

            # self.exp_K = np.zeros((nx, ny, nx, ny))
            mx = self.masses[0]

            self.exp_Kx = np.exp(-1j * self.kx**2/2./mx * dt)

            Iinv = 1./self.masses[1](self.x) # y is the angle
            ky = self.ky

            self.exp_Ky = np.exp(-1j * np.outer(Iinv, ky**2/2.) * dt)

            # psik = fft(typsi, axis=0)
            # kpsi = np.einsum('i, ija -> ija', np.exp(-1j * kx**2/2./mx * dt), psik)

            # # for i in range(nx):
            # #     for j in range(ny):
            # #         my = self.masses[1](y[j])
            # #         self.exp_K[i, j, :, :] = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)


        dt2 = 0.5 * dt

        if self.V is None:
            raise ValueError('The diabatic PES is not specified.')

        v = self.V

        self.exp_V = np.zeros(v.shape, dtype=complex)
        self.exp_V_half = np.zeros(v.shape, dtype=complex)
        # self.apes = np.zeros((nx, ny))
        
        if self.abc:
            eig = scipy.linalg.eig
        else:
            eig = scipy.linalg.eigh

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):

                    w, u = eig(v[i, j, k, :, :])
        
                    V = np.diagflat(np.exp(- 1j * w * dt))
                    V2 = np.diagflat(np.exp(- 1j * w * dt2))
    
                    self.exp_V[i, j, k, :,:] = u.dot(V.dot(dagger(u)))
                    self.exp_V_half[i, j, k, :,:] = u.dot(V2.dot(dagger(u)))


        return

    def population(self, psi):
        """
        return the electronic populations

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(psi, np.ndarray):
            P = np.zeros(self.ns)
            for j in range(self.ns):
                P[j] = np.linalg.norm(psi[:, :, j])**2 * self.dx * self.dy

            assert(np.close(np.sum(P), 1))

        elif isinstance(psi, list):

            N = len(psi)
            P = np.zeros((N, self.nstates))
            for k in range(N):
                P[k, :] = [np.linalg.norm(psi[k][:, :, j])**2 * self.dx * self.dy \
                           for j in range(self.nstates)]


        return P


    def run(self, psi0, e_ops=[], dt=0.01, nt=1, t0=0., nout=1, return_states=True):

        print('Building the propagators ...')

        self.build(dt=dt)

        psi = psi0.copy()

        def _V_half(psi):

            return np.einsum('ijkab, ijkb -> ijka', self.exp_V_half, psi) # evolve V half step

        r = Result(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)

        t = t0
        if self.coords == 'linear':

            KEO = self._KEO_linear

        elif self.coords == 'jacobi':

            KEO = self._KEO_jacobi

        # observables
        if return_states:

            for i in range(nt//nout):
                for n in range(nout):

                    t += dt

                    psi = _V_half(psi)
                    psi = KEO(psi)
                    psi = _V_half(psi)

                r.psilist.append(psi.copy())

        else:

            psi = _V_half(psi)

            for i in range(nt//nout):
                for n in range(nout):
                    
                    t += dt

                    psi = KEO(psi)
                    psi = self._PEO(psi)


                r.psilist.append(psi.copy())
                # tlist.append(t)

                # rho = density_matrix(psi_x, dx)

                # # store the density matrix
                # f.write('{} {} {} {} {} \n'.format(t, *rho))

            psi = KEO(psi)
            psi = _V_half(psi) # evolve V half step

        r.psi = psi
        return r

    def _PEO(self, psi):

        vpsi = np.einsum('ijkab, ijkb -> ijka', self.exp_V, psi)
        return vpsi

    def _KEO_linear(self, psi):
        # psik = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     psik[:,:,j] = fft2(psi[:,:,j])
        
        psik = np.fft.fftn(psi, axes=(0,1, 2))

        kpsi = np.einsum('ijk, ijka -> ijka', self.exp_K, psik)

        # out = np.zeros(psi.shape, dtype=complex)
        # for j in range(ns):
        #     out[:, :, j] = ifft2(kpsi[:, :, j])
        psi = np.fft.ifftn(kpsi, axes=(0,1, 2))
        
        return psi

    def _KEO_jacobi(self, psi):
        """
        The current implementation is of order $\mc{O}(\Delta t^2)$. For improvements,
        we can use the splitting

        K_1(\Delta t/2)K_2(\Delta t/2) V(\Delta t) K_2(\Delta t/2) K_1(\Delta t/2)

        For time-independent H, this is of order $\mc{O}(\Delta t^3)$.

        see Hermann, M. R. & Fleck, J. A.  Phys. Rev. A 38, 6000–6012 (1988).

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        psi : TYPE
            DESCRIPTION.


        """

        psi_ky = fft(psi, axis=1)
        typsi = np.einsum('ij, ija -> ija', self.exp_Ky, psi_ky)
        # psik = ifft(typsi, axis=1)
        # psik = fft(psik, axis=0)
        # kpsi = np.einsum('i, ija -> ija', self.exp_Kx, psik)
        # psi = ifft(kpsi, axis=0)

        psik = fft(typsi, axis=0)
        kpsi = np.einsum('i, ija -> ija', self.exp_Kx, psik)

        psi = ifft2(kpsi, axes=(0,1))


        return psi

    def plot_surface(self, style='2D'):
        if style == '2D':
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.contourf(self.X, self.Y, self.V[:, :, 1, 1], lw=0.7)
            ax1.contourf(self.X, self.Y, self.V[:, :, 0, 0], lw=0.7)
            return

        else:
            from lime.style import plot_surface
            plot_surface(self.x, self.y, self.V[:,:,0,0])

            return

    def plt_wp(self, psilist, **kwargs):


        if not isinstance(psilist, list): psilist = [psilist]

        X, Y = np.meshgrid(self.x, self.y)
        x, y = self.x, self.y 
        
        
        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(x, y, np.abs(psi[:,:,0, 1])**2)
            ax1.contour(x, y, np.abs(psi[:, :,0, 0])**2)
            # ax0.format(**kwargs)
            # ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1
# @jit
# def gwp(x, sigma=1., x0=0., p0=0.):
#     """
#     a Gaussian wave packet centered at x0, with momentum k0
#     """

#     a = 1./sigma**2
#     return (a/np.sqrt(np.pi))**(-0.25)*\
#     np.exp(-0.5 * a * (x - x0)**2 + 1j * (x-x0) * p0)

# @jit
def gauss_x_2d(sigma, x0, y0, kx0, ky0):
    """
    generate the gaussian distribution in 2D grid
    :param x0: float, mean value of gaussian wavepacket along x
    :param y0: float, mean value of gaussian wavepacket along y
    :param sigma: float array, covariance matrix with 2X2 dimension
    :param kx0: float, initial momentum along x
    :param ky0: float, initial momentum along y
    :return: gauss_2d: float array, the gaussian distribution in 2D grid
    """
    gauss_2d = np.zeros((len(x), len(y)), dtype=np.complex128)

    for i in range(len(x)):
        for j in range(len(y)):
            delta = np.dot(np.array([x[i]-x0, y[j]-y0]), inv(sigma))\
                      .dot(np.array([x[i]-x0, y[j]-y0]))
            gauss_2d[i, j] = (np.sqrt(det(sigma))
                              * np.sqrt(np.pi) ** 2) ** (-0.5) \
                              * np.exp(-0.5 * delta + 1j
                                       * np.dot(np.array([x[i], y[j]]),
                                                  np.array([kx0, ky0])))

    return gauss_2d


# @jit
def potential_2d(x_range_half, y_range_half, couple_strength, couple_type):
    """
    generate two symmetric harmonic potentials wrt the origin point in 2D
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return: v_2d: float list, a list containing for matrices:
                               v_2d[0]: the first potential matrix
                               v_2d[1]: the potential coupling matrix
                                        between the first and second
                               v_2d[2]: the potential coupling matrix
                                        between the second and first
                               v_2d[3]: the second potential matrix
    """
    v_2d = [0, 0, 0, 0]
    v_2d[0] = (xv + x_range_half) ** 2 / 2.0 + (yv + y_range_half) ** 2 / 2.0
    v_2d[3] = (xv - x_range_half) ** 2 / 2.0 + (yv - y_range_half) ** 2 / 2.0

    # x_cross = sympy.Symbol('x_cross')
    # mu = sympy.solvers.solve(
    #     (x_cross - x_range_half) ** 2 / 2.0 -
    #     (x_cross + x_range_half) ** 2 / 2.0,
    #     x_cross)

    if couple_type == 0:
        v_2d[1] = np.zeros(np.shape(v_2d[0]))
        v_2d[2] = np.zeros(np.shape(v_2d[0]))
    elif couple_type == 1:
        v_2d[1] = np.full((np.shape(v_2d[0])), couple_strength)
        v_2d[2] = np.full((np.shape(v_2d[0])), couple_strength)
    elif couple_type == 2:
        v_2d[1] = couple_strength * (xv+yv)
        v_2d[2] = couple_strength * (xv+yv)
    # elif couple_type == 3:
    #     v_2d[1] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    #     v_2d[2] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    else:
        raise 'error: coupling type not existing'

    return v_2d


# @jit
def diabatic(x, y):
    """
    PESs in diabatic representation
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return:
        v:  float 2d array, matrix elements of the DPES and couplings
    """
    nstates = 2

    v = np.zeros((nstates, nstates))

    v[0,0] = (x + 4.) ** 2 / 2.0 + (y + 3.) ** 2 / 2.0
    v[1,1] = (x - 4.) ** 2 / 2.0 + (y - 3.) ** 2 / 2.0

    v[0, 1] = v[1, 0] = 0

    return v

# @jit
# def x_evolve_half_2d(dt, v_2d, psi_grid):
#     """
#     propagate the state in grid basis half time step forward with H = V
#     :param dt: float
#                 time step
#     :param v_2d: float array
#                 the two electronic states potential operator in grid basis
#     :param psi_grid: list
#                 the two-electronic-states vibrational state in grid basis
#     :return: psi_grid(update): list
#                 the two-electronic-states vibrational state in grid basis
#                 after being half time step forward
#     """

#     for i in range(len(x)):
#         for j in range(len(y)):
#             v_mat = np.array([[v_2d[0][i, j], v_2d[1][i, j]],
#                              [v_2d[2][i, j], v_2d[3][i, j]]])

#             w, u = scipy.linalg.eigh(v_mat)
#             v = np.diagflat(np.exp(-0.5 * 1j * w / hbar * dt))
#             array_tmp = np.array([psi_grid[0][i, j], psi_grid[1][i, j]])
#             array_tmp = np.dot(u.conj().T, v.dot(u)).dot(array_tmp)
#             psi_grid[0][i, j] = array_tmp[0]
#             psi_grid[1][i, j] = array_tmp[1]
#             #self.x_evolve = self.x_evolve_half * self.x_evolve_half
#             #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
#             #               (self.k * self.k) * dt)


# @jit
def x_evolve_2d(dt, psi, v):
    """
    propagate the state in grid basis half time step forward with H = V
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = np.einsum('ijab, ijb -> ija', np.exp(- 1j * v * dt), psi)


    return vpsi


def k_evolve_2d(dt, masses, kx, ky, psi):
    """
    propagate the state in grid basis a time step forward with H = K
    :param dt: float, time step
    :param kx: float, momentum corresponding to x
    :param ky: float, momentum corresponding to y
    :param psi_grid: list, the two-electronic-states vibrational states in
                           grid basis
    :return: psi_grid(update): list, the two-electronic-states vibrational
                                     states in grid basis
    """

    psi_k = fft2(psi)
    mx, my = masses

    Kx, Ky = np.meshgrid(kx, ky)

    kin = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

    psi_k = kin * psi_k
    psi = ifft2(psi_k)

    return psi


def dpsi(psi, kx, ky, ndim=2):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.
    ndim : int, default 2
        coordinates dimension
    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    kpsi = np.zeros((nx, ny, ndim), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

    return kpsi

def dxpsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi

def dypsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi


def adiabatic_2d(x, y, psi0, v, dt, Nt=0, coords='linear', mass=None, G=None):
    """
    propagate the adiabatic dynamics at a single surface

    :param dt: time step
    :param v: 2d array
                potential matrices in 2D
    :param psi: list
                the initial state
    mass: list of 2 elements
        reduced mass

    Nt: int
        the number of the time steps, Nt=0 indicates that no propagation has been done,
                   only the initial state and the initial purity would be
                   the output

    G: 4D array nx, ny, ndim, ndim
        G-matrix

    :return: psi_end: list
                      the final state

    G: 2d array
        G matrix only used for curvilinear coordinates
    """
    #f = open('density_matrix.dat', 'w')
    t = 0.0
    dt2 = dt * 0.5

    psi = psi0.copy()

    nx, ny = psi.shape

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    if coords == 'linear':
        # Split-operator method for linear coordinates

        psi = x_evolve_2d(dt2, psi,v)

        for i in range(Nt):
            t += dt
            psi = k_evolve_2d(dt, kx, ky, psi)
            psi = x_evolve_2d(dt, psi, v)

    elif coords == 'curvilinear':

        # kxpsi = np.einsum('i, ijn -> ijn', kx, psi_k)
        # kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

        # tpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dypsi = np.zeros((nx, ny, nstates), dtype=complex)

        # for i in range(nstates):

        #     dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
        #     dypsi[:,:,i] = ifft2(kypsi[:,:,i])

        for k in range(Nt):
            t += dt
            psi = rk4(psi, hpsi, dt, kx, ky, v, G)

        #f.write('{} {} {} {} {} \n'.format(t, *rho))
        #purity[i] = output_tmp[4]



    # t += dt
    #f.close()

    return psi

def KEO(psi, kx, ky, G):
    '''
    compute kinetic energy operator K * psi

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
#    kpsi = dpsi(psi, kx, ky)

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    nx, ny = len(kx), len(ky)
    kpsi = np.zeros((nx, ny, 2), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

#   ax.contour(x, y, np.abs(kpsi[:,:,1]))

    tmp = np.einsum('ijrs, ijs -> ijr', G, kpsi)
    #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

    # Fourier transform of the wavefunction
    phi_x = tmp[:,:,0]
    phi_y = tmp[:,:,1]

    phix_k = fft2(phi_x)
    phiy_k = fft2(phi_y)

    # momentum operator in the Fourier space
    kxphi = np.einsum('i, ij -> ij', kx, phix_k)
    kyphi = np.einsum('j, ij -> ij', ky, phiy_k)

    # transform back to coordinate space
    kxphi = ifft2(kxphi)
    kyphi = ifft2(kyphi)

    # psi += -1j * dt * 0.5 * (kxphi + kyphi)

    return 0.5 * (kxphi + kyphi)

def PEO(psi, v):
    """
    V |psi>
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = v * psi
    return vpsi

def hpsi(psi, kx, ky, v, G):

    kpsi = KEO(psi, kx, ky, G)
    vpsi = PEO(psi, v)

    return -1j * (kpsi + vpsi)

######################################################################
# Helper functions for gaussian wave-packets


def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

# @jit
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y


def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

# @jit
# def density_matrix(psi_grid):
#     """
#     compute electronic purity from the wavefunction
#     """
#     rho00 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[0]))*dx*dy
#     rho01 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[1]))*dx*dy
#     rho11 = np.sum(np.multiply(np.conj(psi_grid[1]), psi_grid[1]))*dx*dy

#     purity = rho00**2 + 2*rho01*rho01.conj() + rho11**2

#     return rho00, rho01, rho01.conj(), rho11, purity



if __name__ == '__main__':

    # specify time steps and duration
    ndim = 2 # 2D problem, DO NOT CHANGE!
    dt = 0.01
    print('time step = {} fs'.format(dt * au2fs))

    num_steps = 10


    nx = 2 ** 5
    ny = 2 ** 5
    nz = 2 ** 5
    xmin = -6
    xmax = -xmin
    ymin = -6
    ymax = -ymin
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(xmin, xmax, nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]


    # k-space grid
    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    v0 = 0.5 * ((X+1)**2 + Y**2)
    v1 = 0.5 * ((X-1)**2 + Y**2) + 1.0
    
    
    
    # for i in range(nx):
    #     for j in range(ny):
    #         v[i,j] = diabatic(x[i], y[j])[0,0]

    #ax.imshow(v)

    # specify constants
    mass = [1.0, 1.0]  # particle mass

    x0, y0, kx0, ky0 = -1, 0, 0.0, 0

    #coeff1, phase = np.sqrt(0.5), 0

    print('x range = ', x[0], x[-1])
    print('dx = {}'.format(dx))
    print('number of grid points along x = {}'.format(nx))
    print('y range = ', y[0], y[-1])
    print('dy = {}'.format(dy))
    print('number of grid points along y = {}'.format(ny))

    sigma = np.identity(2) * 1.
    ns = nstates = 2
    psi0 = np.zeros((nx, ny, ns), dtype=complex)
    psi0[:, :, 1] = gauss_x_2d(sigma, x0, y0, kx0, ky0)

    fig, ax = plt.subplots()
    ax.contour(x, y, np.abs(psi0[:, :, 1]).T, cmap='viridis')
    ax.set_title('Initial wavepacket')

    #psi = psi0

    # propagate

    # store the final wavefunction
    #f = open('wft.dat','w')
    #for i in range(N):
    #    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    #f.close()


    # G = np.zeros((nx, ny, ndim, ndim))
    # G[:,:,0, 0] = G[:,:,1, 1] = 1.

    
    extent=[xmin, xmax, ymin, ymax]

    # psi1 = adiabatic_2d(x, y, psi0, v, dt=dt, Nt=num_steps, coords='curvilinear',G=G)
    sol = SPO2(nstates=2, mass=[1, 1], x=x, y=y)

    sol.set_DPES(surfaces = [v0, v1], diabatic_couplings = [[[0, 1], X * 0.2]])

    r = sol.run(psi0=psi0, dt=0.5, Nt=2000)
    
    rho = np.zeros((nstates, nstates, len(r.times)))

    # sol.current_density(r.psilist[-1])
    # r.plot_wavepacket(r.psilist[-1])

    # for i in range(len(r.times)):
    #     rho[:, :, i] = sol.rdm_el(r.psilist[i])
    
    sol.rdm_el(r.psilist)
    P = sol.population(r.psilist, representation='adiabatic')

    
    # p0, p1 = r.get_population()
    fig, ax = plt.subplots()
    ax.plot(r.times, P[:, 0])
    ax.plot(r.times, P[:, 1], label=r'P$_1$')
    ax.legend()

    # r.position()
    
    # for j in range(4):
    #     ax.contourf(x, y, np.abs(r.psilist[j][:, :, 1]).T, cmap='viridis')
        
    # for psi in r.psilist:
    
    
    
        
    


    # psi2 = adiabatic_2d(x, y, psi0, v, mass=mass, dt=dt, Nt=num_steps)
    # ax.contour(x,y, np.abs(psi2).T)
