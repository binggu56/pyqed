#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:40:01 2023

@author: bing
"""


import numpy as np
from numpy import exp, pi, sqrt, meshgrid
from pyqed import transform, dag, isunitary, rk4, isdiag, sinc, sort, interval
from pyqed.wpd import ResultSPO2, SPO2
from pyqed.namd.ldr import WPD2
from pyqed.nonherm import eig
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

    elif dvr == 'periodic':
        
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




class NonHermitianLDR2(WPD2):
    """
    N-state two-mode non-Hermitian conical intersection dynamics 
    
    LDR-SincDVR
    """
    def __init__(self, x, y, nstates=2, ndim=2, mass=None, dvr='sinc'):
        self.x = x
        self.y = y
        self.dx = interval(x)
        self.dy = interval(y)
        self.nx = len(x)
        self.ny = len(y)
        self.nstates = nstates
        
        self.kx = 2 * np.pi * scipy.fft.fftfreq(len(x), self.dx)
        self.ky = 2 * np.pi * scipy.fft.fftfreq(len(y), self.dy)

        self.nx = len(x)
        self.ny = len(y)
        
        self.dvr = dvr
        
        if mass is None:
            mass = [1, ] * ndim 
        self.mass = mass
        
        self.ngrid = [self.nx, self.ny]

        self.kmax = np.pi/self.dx # energy range [-K, K]
        self.shift = (self.nx - 1)//2
        
        self.nbasis = self.nx * self.ny # for product DVR
        
        self.X = None
        self.K = None
        self.V = None
        self.H = None 
        self._v = None
        
        self.geometies = None
        self.right_eigenstates = None
        self.left_eigenstates = None
        self.apes = None
        self.norm_right = None
        
    @property
    def v(self):
        return self._v 
    
    @v.setter
    def v(self, v):
        assert(v.shape == (self.nx, self.ny, self.nstates, self.nstates))
        
        self._v = v
    
    # def primitive_basis(self, n):
    #     # the index n \in [-N, ... 0, 1, ... N]
    #     n = sigma(n)
    #     dx = self.dx
    #     x = self.x

    #     return 
    #     # return 1/sqrt(dx) * sinc(np.pi * (x/dx - n))
    
    def ip(self):
        """
        (i) * momentum matrix elements
        """
        nx = self.nx
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
    
    def buildV(self):
        nx, ny, nstates = self.nx, self.ny, self.nstates

        va = np.zeros((nx, ny, nstates), dtype=complex)
        
        self.right_eigenstates = np.zeros((nx, ny, nstates, nstates), dtype=complex)  # diabatic to adiabatic transformation 
        self.left_eigenstates = np.zeros((nx, ny, nstates, nstates), dtype=complex)  # diabatic to adiabatic transformation 
        self.norm_right = np.zeros((nx, ny, nstates, nstates), dtype=complex)
        
        for i in range(nx):
            for j in range(ny):
                
                vij = self.v[i, j]
                
                # w, ul, ur = scipy.linalg.eig(vij, left=True, right=True)
                
                
                # norm = np.diag(dag(ul) @ ur)

                # ur = np.einsum('ij, j -> ij', ur, 1./sqrt(norm))
                # ul = np.einsum('ij, j -> ij', ul, 1./np.sqrt(norm))
                
                w, ur, ul = eig(vij)
            
                
                # idx = np.argsort(w.real)
                # w = w[idx]     
                # ur = ur[:, idx]
                # # ul = ul[:, idx]
                
                self.right_eigenstates[i,j] = ur
                self.left_eigenstates[i, j] = dag(ul)
                
                va[i,j] = w 
                
                self.norm_right[i,j] = dag(ur) @ ur

                
                # ur[i,j] = ur
                #print(np.dot(U.conj().T, Vmat.dot(U)))

        # self.apes = va.reshape((self.nbasis, nstates))
        # self.adiabatic_states = u.reshape((self.nbasis, nstates, nstates))
        
        self.apes = va
        # self.adiabatic_states = u    
        
        

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
    
    def build_ovlp(self, dtype=complex):
        

        N = self.nbasis
        
        nstates = self.nstates
        
        nx, ny = self.nx, self.ny
        
        # K = self.buildK().reshape((N, N))
        
        # overlap of electronic states
        A = np.zeros((nx, ny, nx, ny, nstates, nstates), dtype=dtype)
        # self._K = np.zeros((N, N, M, M), dtype=dtype)


        for k in range(N):
            
            i, j = np.unravel_index(k, (nx, ny))
            
            psi1 = self.left_eigenstates[i, j]

            A[i, j, i, j] = np.eye(nstates) #* K[i, i] # identity matrix at the same geometry

            for l in range(k):
                    
                    ii, jj = np.unravel_index(l, (nx, ny))

                    psi2 = self.right_eigenstates[ii, jj]
    
                    # for a in range(M):
                    #     for b in range(M):
                    #         A[i, j, a, b] = braket(psi1[:, a], psi2[:, b])
                    #         A[j, i, b, a] = A[i, j, a, b].conj()
                    A[i, j, ii, jj] = dag(psi1) @ psi2 #* K[i, j]
                    
                    A[ii, jj, i, j] = (A[i, j, ii, jj].T)

        expKx, expKy = self.exp_K
        
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
        
        print('building the adibatic potential energy surfaces ...')
        self.build_apes()
        
        print('building the kinetic and potential energy propagator')
        self.buildK(dt)
        
        print('building the electronic overlap matrix')
        self.build_ovlp()
        
        self.build(dt)

        
        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0.copy()]
        
        # project the intial state onto the left eigenvectors
        # psi = np.einsum('ijal, ija -> ijl', self.left_eigenstates.conj(), psi0)
        psi = psi0.copy()
        
        # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)
        psi = self.exp_V_half * psi
        
        
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
    
    def position(self, psi, d=0):
        x = self.x
        if isinstance(psi, list):
            xAve = [np.einsum('ijb, i, ijba, ija ->', _psi.conj(), x, self.norm_right, _psi)\
                     for _psi in psi]
            # xAve = [np.einsum('ija, i, ija', _psi.conj(), x, _psi) for _psi in psi]
            yAve = [np.einsum('ijb, j, ijba, ija ->', _psi.conj(), self.y, self.norm_right, _psi)\
                     for _psi in psi]
                #[np.einsum('j, ija, ija', self.y, self.norm_right, np.abs(_psi)**2) for _psi in psi]
    
        
        return np.array(xAve)*self.dx*self.dy, np.array(yAve) * self.dx * self.dy 
    
        
        
        
        
    
if __name__ == '__main__':
    pass
    