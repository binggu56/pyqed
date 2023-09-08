#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 10 11:14:55 2017

@author: Bing Gu

History:
2/12/18 : fix a bug with the FFT frequency

Several possible improvements:
    1. use pyFFTW to replace the Scipy


"""

import numpy as np

import sys
if sys.version_info[1] < 10:
    import proplot as plt
else:
    import matplotlib.pyplot as plt

# from matplotlib import animation

from scipy.linalg import expm, sinm, cosm
import scipy


#from lime.fft import fft, ifft
from pyqed import dagger, gwp, meshgrid
from numpy import cos, pi
from scipy.fftpack import fft2, ifft2, fftfreq, fft, ifft, fftshift
from numpy.linalg import inv, det

from pyqed import rk4, dagger, gwp, interval, meshgrid, norm2
from pyqed.units import au2fs
from pyqed.mol import Result

class SPO:
    def __init__(self, x, nstates, psi0=None, mass=1, v=None):
        """
        Non-adiabatic molecular dynamics (NAMD) simulations for one nuclear dof
            and many electronic states.

        Args:
            x: real array of size N
                grid points

            psi0: complex array [N, ns]
                initial wavefunction

            mass: float, nuclear mass

            nstates: integer, number of states, default 2

            V_x: real array [N, ns**2]
                potential energy surfaces and vibronic couplings

        """
        self.x = x
        self.psi0 = psi0
        self.mass = mass
        self.V_x = v
        self.nstates = nstates

    def x_evolve(self, psi, vpsi):
        """
        vpsi = exp(-i V dt)
        """

        # for i in range(len(x)):

        #     tmp = psi_x[i, :]
        #     utmp = U[i,:,:]
        #     psi_x[i,:] = np.dot(U,V.dot(dagger(U))).dot(tmp)

        psi = np.einsum('imn, in -> im', vpsi, psi)

        return psi


    def k_evolve(self, dt, k, psi_x):
        """
        one time step for exp(-i * K * dt)
        """
        mass = self.mass
        #x = self.x

        for n in range(2):

            psi_k = fft(psi_x[:,n])

            psi_k *= np.exp(-0.5 * 1j / mass * (k * k) * dt)

            psi_x[:,n] = ifft(psi_k)

        return psi_x

    def run(self, dt, psi0, nt = 1):

        """
        Perform a series of time-steps via the time-dependent
        Schrodinger Equation.

        Parameters
        ----------
        dt : float
            the small time interval over which to integrate

        Nsteps : float, optional
            the number of intervals to compute.  The total change
            in time at the end of this method will be dt * Nsteps.
            default is N = 1
        """
        if dt > 0.0:
            f = open('density_matrix.dat', 'w')
        else:
            f = open('density_matrix_backward.dat', 'w')

        x = self.x
        V_x = self.V_x

        nx = len(x)
        nstates = self.nstates

        dt2 = 0.5 * dt


        vpsi = np.zeros((nx, nstates, nstates), dtype=complex)
        vpsi2 = np.zeros((nx, nstates, nstates), dtype=complex)

        for i in range(nx):

            Vmat = np.reshape(V_x[i,:], (nstates, nstates))
            w, u = scipy.linalg.eigh(Vmat)

            #print(np.dot(U.conj().T, Vmat.dot(U)))

            v = np.diagflat(np.exp(- 1j * w * dt))
            v2 = np.diagflat(np.exp(- 1j * w * dt2))

            vpsi[i,:,:] = u.dot(v.dot(dagger(u)))
            vpsi2[i,:,:] = u.dot(v2.dot(dagger(u)))


        dx = x[1] - x[0]

        k = 2.0 * np.pi * scipy.fftpack.fftfreq(nx, dx)

        print('Propagating the wavefunction ...')

        t = 0.0
        psi_x = self.x_evolve(psi0, vpsi2) # evolve V half step

        for i in range(nt - 1):

            t += dt

            psi_x = self.k_evolve(dt, k, psi_x)
            psi_x = self.x_evolve(psi_x, vpsi)

            rho = density_matrix(psi_x, dx)

            # store the density matrix
            f.write('{} {} {} {} {} \n'.format(t, *rho))

        # psi_x = self.k_evolve(dt, psi_x)
        # psi_x = self.x_evolve(dt2, psi_x, vpsi2)


        f.close()

        return psi_x

def density_matrix(psi_x,dx):
    """
    compute purity from the vibronic wavefunction
    """
    rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
    rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
    rho11 = 1. - rho00
    return rho00, rho01, rho01.conj(), rho11


class SPO2:
    """
    second-order split-operator method for nonadiabatic wavepacket dynamics
    in the diabatic representation with two-dimensional nuclear coordinate

    For time-independent Hamiltonian

        e^{-i H \Delta t} = e^{- i V \Delta t/2} e^{-i K \Delta t} e^{-iV\Delta t/2}

    For time-dependent H,
        TBI
    """
    def __init__(self, x, y, mass, nstates=2, coords='linear', G=None, abc=False):
        self.x = x
        self.y = y
        self.X, self.Y = meshgrid(x, y)

        self.nx = len(x)
        self.ny = len(y)
        self.dx = interval(x)
        self.dy = interval(y) # for uniform grids
        self.mass = self.masses = mass
        self.kx = None
        self.ky = None
        self.dim = 2
        self.exp_V = None
        self.exp_V_half = None
        self.exp_K = None
        self.V = None
        self.G = G
        self.nstates = self.ns = nstates
        self.coords =  coords
        self.abc = abc

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
        self.V = v
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

                i
                w, u = eig(v[i, j, :, :])

                #print(np.dot(U.conj().T, Vmat.dot(U)))

                V = np.diagflat(np.exp(- 1j * w * dt))
                V2 = np.diagflat(np.exp(- 1j * w * dt2))

                self.exp_V[i, j, :,:] = u.dot(V.dot(dagger(u)))
                self.exp_V_half[i, j, :,:] = u.dot(V2.dot(dagger(u)))


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


    def run(self, psi0, e_ops=[], dt=0.01, Nt=1, t0=0., nout=1, return_states=True):

        print('Building the propagators ...')

        self.build(dt=dt)

        psi = psi0.copy()

        def _V_half(psi):

            return np.einsum('ijab, ijb -> ija', self.exp_V_half, psi) # evolve V half step

        r = ResultSPO2(dt=dt, psi0=psi0, Nt=Nt, t0=t0, nout=nout)
        r.x = self.x
        r.y = self.y

        t = t0
        if self.coords == 'linear':

            KEO = self._KEO_linear

        elif self.coords == 'jacobi':

            KEO = self._KEO_jacobi

        # observables
        if return_states:

            for i in range(Nt//nout):
                for n in range(nout):

                    t += dt

                    psi = _V_half(psi)
                    psi = KEO(psi)
                    psi = _V_half(psi)

                r.psilist.append(psi.copy())

        else:

            psi = _V_half(psi)

            for i in range(Nt//nout):
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
            from lime.style import plot_surface
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
    def __init__(self, x, y, z, mass, nstates=2, coords='linear', G=None, abc=False):
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

        self.masses = mass
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
    
    
def divergence(f,h):
    """
    div(F) = dFx/dx + dFy/dy + ...
    g = np.gradient(Fx,dx, axis=1)+ np.gradient(Fy,dy, axis=0) #2D
    g = np.gradient(Fx,dx, axis=2)+ np.gradient(Fy,dy, axis=1) +np.gradient(Fz,dz,axis=0) #3D
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i], axis=i) \
                                    for i in range(num_dims)])
######################################################################
# Helper functions for gaussian wave-packets

# def gwp(x, a, x0, k0):
#     """
#     a gaussian wave packet of width a, centered at x0, with momentum k0
#     """
#     return ((a * np.sqrt(np.pi)) ** (-0.5)
#             * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))


######################################################################
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


if __name__ == '__main__':
    # specify time steps and duration
    dt = 0.05
    # t_max = 100
    # frames = int(t_max / float(N_steps * dt))

    # specify constants
    hbar = 1.0   # planck's constant
    m = 1.0      # particle mass

    # specify range in x coordinate
    N = 2 ** 9
    xmin = -6
    xmax = -xmin
    #dx = 0.01
    #x = dx * (np.arange(N) - 0.5 * N)
    x = np.linspace(xmin,xmax,N)
    print('x range = ',x[0], x[-1])
    dx = x[1] - x[0]
    print('dx = {}'.format(dx))
    print('number of grid points = {}'.format(N))

    # specify potential
    #V0 = 1.5
    #L = hbar / np.sqrt(2 * m * V0)
    #a = 3 * L

    # diabatic surfaces with vibronic couplings
    V_x = np.zeros((N,4))
    V_x[:,0] = (x)**2/2.0
    V_x[:,3] = (x-1.)**2/2.0
    c = 0.1
    V_x[:,1] = c
    V_x[:,2] = c


    print('constant vibronic coupling  = ', c)

    # specify initial momentum and quantities derived from it
    #p0 = np.sqrt(2 * m * 0.2 * V0)
    p0 = 0.0
    x0 = 0.0
    #dp2 = p0 * p0 * 1./80
    #d = hbar / np.sqrt(2 * dp2)
    a = 1.

    k0 = p0 / hbar
    v0 = p0 / m
    angle = 0.0  # np.pi/4.0
    print('initial phase difference between c_g and c_e = {} Pi'.format(angle/np.pi))
    psi0 = np.zeros((N,2), dtype=complex)
    psi0[:,0] =  gwp(x, a, x0, k0) * np.exp(1j*angle)
    # psi_x0[:,1] = 1./np.sqrt(2.) * gauss_x(x, a, x0, k0)

    sol = NAMD(x, 2, psi0, mass=1, V_x =V_x)
    sol.propagate(dt, psi0, nt=1000)

    rho = np.genfromtxt('density_matrix.dat')

    import proplot as plt
    fig, ax = plt.subplots()
    # ax.plot(rho[:, 0], rho[:, 4])
    print(rho[:,2])
    ax.plot(rho[:, 0], rho[:, 1].real)
    # ax.format(ylim=(-1,1), xlim=(0,100))


######################################################################


