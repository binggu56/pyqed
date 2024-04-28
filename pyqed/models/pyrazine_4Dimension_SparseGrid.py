#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:44:37 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import numpy as np
from pyqed import transform, dag, isunitary, rk4, isdiag, sinc, sort, isherm, interval,\
    cartesian_product
from pyqed.wpd import ResultSPO2
from pyqed.ldr.gwp import WPD2
from pyqed.mol import Result
from pyqed.units import au2ev, wavenum2au
from pyqed.models.vibronic import Vibronic2

import warnings

# from pyqed.wpd import Result

import scipy
import string

from scipy.linalg import inv
from scipy.sparse import kron, eye
from scipy.linalg import eigh
import logging

try:
    import proplot as plt
except ImportError:
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
    


def ip(nx, dx):
    """
    (i) * momentum matrix elements = :math:`\nabla`
    """

    p = np.zeros((nx, nx))
    for n in range(nx):
        for m in range(nx):
            p[n, m] = 1/dx * (-1)**(n-m)/(n-m)
    return p

def gen_enisum_string(D):
    alphabet = list(string.ascii_lowercase)
    if (D > 10):
        raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))
    
    ini = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
    einsum_string = ini  
    for n in range(D):
        einsum_string += ','
        einsum_string += alphabet[n] + alphabet[n+D]

    return (einsum_string + ' -> ' + ini)




class ResultLDR(Result):
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

                fig.savefig('psi'+str(i)+'.pdf')

            return axes
        
    def get_population(self, fname=None, plot=False):
        dx = interval(coord[0])
        dy = interval(coord[1])
        dz = interval(coord[2])
        dq = interval(coord[3])
        # p0 = [norm2(psi[:, :, 0]) * dx * dy for psi in self.psilist]
        # p1 = [norm2(psi[:, :, 1]) * dx * dy for psi in self.psilist]
        # p2 = [norm2(psi[:, :, 2]) * dx * dy for psi in self.psilist]

        # p0 = [np.einsum('ijkl, ijkl -> ', psi[:, :, :, :, 0], np.conj(psi[:, :, :, :, 0])) * dx * dy * dz * dq for psi in self.psilist]
        # p1 = [np.einsum('ijkl, ijkl -> ', psi[:, :, :, :, 1], np.conj(psi[:, :, :, :, 1])) * dx * dy * dz * dq for psi in self.psilist]
        # p2 = [np.einsum('ijkl, ijkl -> ', psi[:, :, :, :, 2], np.conj(psi[:, :, :, :, 2])) * dx * dy * dz * dq for psi in self.psilist]

        p0 = [(np.einsum('ijkl, ijkl', np.conj(psi[:, :, :, :, 0]), psi[:, :, :, :, 0]) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]
        p1 = [(np.einsum('ijkl, ijkl', np.conj(psi[:, :, :, :, 1]), psi[:, :, :, :, 1]) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]
        p2 = [(np.einsum('ijkl, ijkl', np.conj(psi[:, :, :, :, 2]), psi[:, :, :, :, 2]) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]

        # p0 = [np.trace(dag(psi[:, :, :, :, 0])@psi[:, :, :, :, 0]) * dx * dy * dz * dq for psi in self.psilist]
        # p1 = [np.trace(dag(psi[:, :, :, :, 1])@psi[:, :, :, :, 1]) * dx * dy * dz * dq for psi in self.psilist]
        # p2 = [np.trace(dag(psi[:, :, :, :, 2])@psi[:, :, :, :, 2]) * dx * dy * dz * dq for psi in self.psilist]

        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.times, p0)
            ax.plot(self.times, p1)
            ax.plot(self.times, p2)

        self.population = [p0, p1, p2]
        if fname is not None:
            # fname = 'population'
            np.savez(fname, p0, p1, p2)

        return p0, p1, p2

    
    def plot_population(self, p):
        
        fig, ax = plt.subplots()
        for n in range(self.nstates):
            ax.plot(self.times, p[:,n])
            
        return fig, ax
        
    
    def position(self, plot=False, fname=None):        
        # X, Y = np.meshgrid(self.x, self.y)
        dx = interval(coord[0])
        dy = interval(coord[1])
        dz = interval(coord[2])
        dq = interval(coord[3])        
        
        xAve = [(np.einsum('ijkln, i, ijkln', psi.conj(), coord[0], psi) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]
        yAve = [(np.einsum('ijkln, j, ijkln', psi.conj(), coord[1], psi) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]
        zAve = [(np.einsum('ijkln, k, ijkln', psi.conj(), coord[2], psi) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]
        qAve = [(np.einsum('ijkln, l, ijkln', psi.conj(), coord[3], psi) * dx * dy * dz * dq) / (np.einsum('ijkla, ijkla', psi.conj(), psi) * dx * dy * dz * dq) for psi in self.psilist]   

        xAve = np.real_if_close(xAve)
        yAve = np.real_if_close(yAve)
        zAve = np.real_if_close(zAve)
        qAve = np.real_if_close(qAve)        
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.times, xAve)
            ax.plot(self.times, yAve)
            ax.plot(self.times, zAve)
            ax.plot(self.times, qAve)            
        
        self.xAve = [xAve, yAve, zAve, qAve]
        np.savez('xAve', xAve, yAve, zAve, qAve)
        
        return xAve, yAve, zAve, qAve
        
        
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
            
            


class LDRN:
    """
    many-dimensional many-state nonadiabatic conical intersection dynamics in 
    DVR + LDR + SPO
    """
    def __init__(self, domains, levels, ndim=3, nstates=2, mass=None, dvr_type='sinc'): 

        assert(len(domains) == len(levels) == ndim)
        
        self.L = [domain[1] - domain[0] for domain in domains]

        
        x = []
        if dvr_type in ['sinc', 'sine']:
        
            for d in range(ndim):
                x.append(discretize(*domains[d], levels[d], endpoints=False))
        # elif dvr_type == 'sine':
        else:
            raise ValueError('DVR {} is not supported. Please use sinc.')
            
        
        self.x = x
        self.dx = [interval(_x) for _x in x]
        self.nx = [len(_x) for _x in x] 
        
        self.dvr_type = dvr_type
        self.ndim = ndim
        
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass
        
        self.nstates = nstates
        
        # all configurations in a vector
        self.points = np.fliplr(cartesian_product(x))
        self.npts = len(self.points)

        ###
        self.H = None
        self._K = None
        # self._V = None
        
        self._v = None
        self.exp_K = None
        self.exp_V = None
        self.wf_overlap = self.A = None
        
        self.apes = None 
        self.adiabatic_states = None
        
        
    @property
    def v(self):
        return self._v 
    
    @v.setter
    def v(self, v):
        assert(v.shape == (*self.nx, self.nstates, self.nstates))
        
        self._v = v



    def build_ovlp(self):
    
        N = self.npts
        nx = self.nx
        nstates = self.nstates
        x1, x2, x3, x4 = self.x
        d0 = 4
        
        # overlap of electronic states
        A = np.zeros((nx[0], nx[1], nx[2], nx[3], nx[0], nx[1], nx[2], nx[3], nstates, nstates), dtype=np.float16)

        # for k in range(N):            
        #     i0, i1, i2, i3 = np.unravel_index(k, (nx[0], nx[1], nx[2], nx[3]))
            
        #     psi1 = self.adiabatic_states[i0, i1, i2, i3]

        #     A[i0, i1, i2, i3, i0, i1, i2, i3] = np.eye(nstates) #* K[i, i] # identity matrix at the same geometry

        #     for l in range(k):
        #         j0, j1, j2, j3 = np.unravel_index(l, (nx[0], nx[1], nx[2], nx[3]))

        #         psi2 = self.adiabatic_states[j0, j1, j2, j3]

        #         A[i0, i1, i2, i3, j0, j1, j2, j3] = dag(psi1) @ psi2 #* K[i, j]
        #         A[j0, j1, j2, j3, i0, i1, i2, i3] = dag(A[i0, i1, i2, i3, j0, j1, j2, j3])

        #         # # short-range approximation
        #         # d = np.sqrt((x[i] - x[ii])**2 + (y[j] - y[jj])**2)
        #         # if d > d0:
        #         #     A[i, j, ii, jj] = np.zeros(nstates)
        #         #     A[ii, jj, i, j] = np.zeros(nstates)

        # A = np.transpose(A, (0, 1, 2, 3, 8, 4, 5, 6, 7, 9))  
    
        U = self.adiabatic_states        
        A = np.einsum('abcdmp, ijklmq -> abcdpijklq', U.conj(), U).astype(np.float16)

        return A



    def buildK(self, dt):
        """
        For the kinetic energy operator with Jacobi coordinates

            K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

        Since the two KEOs for each dof do not commute, it has to be factorized as

        e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}

        where $p_\theta = -i \pa_\theta$ is the momentum operator.


        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """


        self.exp_K = []
        
        for d in range(self.ndim):
                    
            Tx = kinetic(self.x[d], mass=self.mass[d], dvr=self.dvr_type)
        
            expKx = scipy.linalg.expm(-1j * Tx * dt)

            self.exp_K.append(expKx.copy())
            
        return self.exp_K
    


    def buildV(self, dt):
        """
        Setup the propagators appearing in the split-operator method.



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
        
        dt2 = 0.5 * dt
        self.exp_V = np.exp(-1j * dt * self.apes)

        self.exp_V_half = np.exp(-1j * dt2 * self.apes)

        return



    def run(self, psi0, dt, nt, nout=1, t0=0):
        
        print('nx =', self.nx, 'nstates =', self.nstates)
        
        assert(psi0.shape == (*self.nx, self.nstates))
        
        
        
        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()
        
        self.buildV(dt)
        
        print('building the kinetic energy propagator')
        self.buildK(dt)

        
        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()
        

        einsum_string = gen_enisum_string(self.ndim)
        exp_T = np.einsum(einsum_string, self.A, *self.exp_K)
        
            
        r = ResultLDR(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)

        r.psilist = [psi0]
        
        alphabet = list(string.ascii_lowercase)
        D = self.ndim
        _string = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y, ' + \
            "".join(alphabet[D:2*D])+'y -> ' + "".join(alphabet[:D]) + 'x'
            
        psi = psi0.copy()
        psi = self.exp_V_half * psi
        # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)
        
        for k in range(nt//nout):
            for kk in range(nout):
                psi = np.einsum(_string, exp_T, psi) 
                psi = self.exp_V * psi 
                
            r.psilist.append(psi.copy())
        
        psi = self.exp_V_half * psi
        
        return r
    


    def rdm_el(self, psi):
        """
        compute the reduced electronic density matrices 

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """        
        D = self.ndim 
        alphabet = list(string.ascii_lowercase)
        
        einsum_string = "".join(alphabet[:D]) + 'x, ' + "".join(alphabet[:D])+'y ->  xy'
        
        rho = np.einsum(einsum_string, psi.conj(), psi)
        return rho
    


    

class LDR2(WPD2):
    """
    N-state two-mode conical intersection dynamics with Fourier series 
    
    LDR-SPO-SincDVR
    
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
        
        self.ngrid = [self.nx, self.ny]

        self.kmax = np.pi/self.dx # energy range [-K, K]
        self.shift = (self.nx - 1)//2
        
        self.nbasis = self.nx * self.ny # for product DVR
        
        self.X = None
        self.K = None
        self.V = None
        self.H = None 
        self._v = None # diabatic potential matrix
        
        
        self.geometies = None
        self.adiabatic_states = []
        self._apes = None
        self.electronic_overlap = self.A = None
        
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


    
    def buildK(self, dt):

        mx, my = self.mass 
        
        Tx = kinetic(self.x, mass=mx, dvr=self.dvr)
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)
        
        Ty = kinetic(self.y, my, dvr=self.dvr)

        expKy = scipy.linalg.expm(-1j * Ty * dt)

        self.exp_K = [expKx, expKy]
        return 

    

    def build_apes(self):
        """
        build the APES by diagonalizing the diabatic potential matrix.
        
        This should only be used for diabatic models.

        Returns
        -------
        va : TYPE
            DESCRIPTION.

        """
        nx, ny, nstates = self.nx, self.ny, self.nstates

        va = np.zeros((nx, ny, nstates))
        
        if np.iscomplexobj(self.v):
            
            u = np.zeros((nx, ny, nstates, nstates), dtype=complex)  # diabatic to adiabatic transformation 
        else:
            u = np.zeros((nx, ny, nstates, nstates))  # diabatic to adiabatic transformation 
        

        for i in range(nx):
            for j in range(ny):
                
                vij = self.v[i, j]

                w, v = sort(*scipy.linalg.eigh(vij))
                
                va[i,j] = w 
                u[i,j] = v
        
        self.apes = va
        self.adiabatic_states = u    

        return va
    
    @property
    def apes(self):
        return self._apes 
    
    @apes.setter
    def apes(self, v):
        self._apes = v


    
    def buildV(self, dt):
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

        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        dt2 = 0.5 * dt
        self.exp_V = np.exp(-1j * dt * self.apes)

        self.exp_V_half = np.exp(-1j * dt2 * self.apes)

        return
    


    def build_ovlp(self):
        
        N = self.nbasis
        nstates = self.nstates
        nx, ny = self.nx, self.ny
        
        # overlap of electronic states
        A = np.zeros((nx, ny, nx, ny, nstates, nstates), dtype=complex)


        for k in range(N):
            
            i, j = np.unravel_index(k, (nx, ny))
            
            psi1 = self.adiabatic_states[i, j]

            A[i, j, i, j] = np.eye(nstates) #* K[i, i] # identity matrix at the same geometry

            for l in range(k):
                ii, jj = np.unravel_index(l, (nx, ny))
                psi2 = self.adiabatic_states[ii, jj]

                A[i, j, ii, jj] = dag(psi1) @ psi2 #* K[i, j]
                A[ii, jj, i, j] = dag(A[i, j, ii, jj])

        A = np.transpose(A, (0, 1, 4, 2, 3, 5))
        self.A = A
        return A


    
    def run(self, psi0, dt, nt, nout=1, t0=0):
        
        assert(psi0.shape == (self.nx, self.ny, self.nstates))
        
        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()
        
        self.buildV(dt)
        
        print('building the kinetic energy propagator')
        self.buildK(dt)

        
        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()
        
        expKx, expKy = self.exp_K

        # T_{mn} A_{mb, na} = kinetic energy operator in LDR
        self.exp_T = np.einsum('ijaklb, ik, jl -> ijaklb', self.A, expKx, expKy)
        
        
        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0]
        
        psi = psi0.copy()
        psi = self.exp_V_half * psi
        
        for k in range(nt//nout):
            for kk in range(nout):
                psi = np.einsum('ijaklb, klb->ija', self.exp_T, psi) 
                psi = self.exp_V * psi 
                
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
        psik = scipy.fft.fft2(psi, axes=(0,1))
        kpsi = np.einsum('ij, ija -> ija', self.exp_K, psik)
        psi = scipy.fft.ifft2(kpsi, axes=(0,1))
        return psi
    
    def fbr2dvr(self):
        pass
    
    def dvr2fbr(self):
        pass


    def x_evolve(self, psi):
        return self.expV * psi
    
    def k_evolve(self, psi):
        return self.expK @ psi
    


class SincDVR2(WPD2):
    """
    N-state two-mode conical intersection dynamics with Fourier series 
    
    LDR-SPO-SincDVR
    
    Deprecated. Use LDR2 instead.
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
        
        self.ngrid = [self.nx, self.ny]

        self.kmax = np.pi/self.dx # energy range [-K, K]
        self.shift = (self.nx - 1)//2
        
        self.nbasis = self.nx * self.ny # for product DVR
        
        self.X = None
        self.K = None
        self.V = None
        self.H = None 
        self._v = None # diabatic potential matrix
        
        
        self.geometies = None
        self.adiabatic_states = []
        self.apes = None
        self.electronic_overlap = self.A = None
        
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
 

    
    def buildK(self, dt):
        
        mx, my = self.mass 

        Tx = kinetic(self.x, mass=mx, dvr=self.dvr)
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)
        
        Ty = kinetic(self.y, my, dvr=self.dvr)

        expKy = scipy.linalg.expm(-1j * Ty * dt)

        self.exp_K = [expKx, expKy]
        return 
    
 
    
    def build_apes(self):
        """
        build the APES by diagonalizing the diabatic potential matrix.
        
        This should only be used for diabatic models.

        Returns
        -------
        va : TYPE
            DESCRIPTION.

        """
        nx, ny, nstates = self.nx, self.ny, self.nstates

        va = np.zeros((nx, ny, nstates))
        
        if np.iscomplexobj(self.v):
            
            u = np.zeros((nx, ny, nstates, nstates), dtype=complex)  # diabatic to adiabatic transformation 
        else:
            u = np.zeros((nx, ny, nstates, nstates))  # diabatic to adiabatic transformation 
        

        for i in range(nx):
            for j in range(ny):
                
                vij = self.v[i, j]

                w, v = sort(*scipy.linalg.eigh(vij))
                
                va[i,j] = w 
                u[i,j] = v
        
        self.apes = va
        self.adiabatic_states = u    

        return va
    
    @property
    def apes(self):
        return self.apes 
    
    @apes.setter
    def apes(self, v):
        self.apes = v


    
    def buildV(self, dt):
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

        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        dt2 = 0.5 * dt
        self.exp_V = np.exp(-1j * dt * self.apes)

        self.exp_V_half = np.exp(-1j * dt2 * self.apes)

        return
    
    def build_ovlp(self):
    
        N = self.nbasis
        nstates = self.nstates
        nx, ny = self.nx, self.ny
        x, y =self.x, self.y
        d0 = 0.01
        
        # overlap of electronic states
        A = np.zeros((nx, ny, nx, ny, nstates, nstates), dtype=complex)

        for k in range(N):
            
            i, j = np.unravel_index(k, (nx, ny))
            
            psi1 = self.adiabatic_states[i, j]

            A[i, j, i, j] = np.eye(nstates) #* K[i, i] # identity matrix at the same geometry

            for l in range(k):
                ii, jj = np.unravel_index(l, (nx, ny))

                psi2 = self.adiabatic_states[ii, jj]

                A[i, j, ii, jj] = dag(psi1) @ psi2 #* K[i, j]
                A[ii, jj, i, j] = dag(A[i, j, ii, jj])

                d = np.sqrt((x[i] - x[ii])**2 + (y[j] - y[jj])**2)

                if d > d0:
                    A[i, j, ii, jj] = np.zeros(nstates)
                    A[ii, jj, i, j] = np.zeros(nstates)
    
        A = np.transpose(A, (0, 1, 4, 2, 3, 5))
        self.A = A
        return A
        

    
    def run(self, psi0, dt, nt, nout=1, t0=0):
        
        assert(psi0.shape == (self.nx, self.ny, self.nstates))
        
        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()
        
        self.buildV(dt)
        
        print('building the kinetic energy propagator')
        self.buildK(dt)

        
        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()
        
        expKx, expKy = self.exp_K

        self.exp_T = np.einsum('ijaklb, ik, jl -> ijaklb', self.A, expKx, expKy)
        
        
        r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        r.x = self.x
        r.y = self.y
        r.psilist = [psi0]
        
        psi = psi0.copy()
        psi = self.exp_V_half * psi
        
        for k in range(nt//nout):
            for kk in range(nout):
                psi = np.einsum('ijaklb, klb->ija', self.exp_T, psi) 
                psi = self.exp_V * psi 
                
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
        psik = scipy.fft.fft2(psi, axes=(0,1))
        kpsi = np.einsum('ij, ija -> ija', self.exp_K, psik)

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


class SineDVR2(SincDVR2):
    """
    https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
    """
    
    def buildK(self, dt):
 
        mx, my = self.mass 
        
        Tx = kinetic(self.x, mx, dvr='sine')
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)
        
        Ty = kinetic(self.y, my, dvr='sine')

        expKy = scipy.linalg.expm(-1j * Ty * dt)

        self.exp_K = [expKx, expKy]
        return 

        

class SincDVR_PBC(SincDVR2):
    
    def buildK(self, dt):
 
        mx, my = self.mass 
        
        Tx = kinetic(self.x, mx, dvr='sinc_periodic')
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)
        
        Ty = kinetic(self.y, my, dvr='sinc_periodic')

        expKy = scipy.linalg.expm(-1j * Ty * dt)

        self.exp_K = [expKx, expKy]
        return 




class Pyrazine(Vibronic2):
    """
    vibronic coupling model for pyrazine S0/S1/S2 conical intersection
    """
    def __init__(self, x=None, y=None, z=None, q=None):
        self.x = x 
        self.y = y
        self.z = z
        self.q = q
        
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        self.nq = len(q)        
        
        self.nstates = 3
        
        self.idm_el = np.eye(self.nstates)
        
        self.edip = np.zeros((self.nstates, self.nstates))
        self.edip[0, 2] = self.edip[2, 0] = 1.
        
        self.mass = [1/(1015. * wavenum2au), 1./(596. * wavenum2au), 1/(1230. * wavenum2au), 1./(919. * wavenum2au)]
        
        self.v = None

    
    def buildV(self): #global
        """
        Build the global diabatic PES (global)

        Returns
        -------
        None.

        """
        nx, ny, nz, nq = self.nx, self.ny, self.nz, self.nq 
        nstates = self.nstates        
        x, y, z, q = self.x, self.y, self.z, self.q
        
        v = np.zeros((nx, ny, nz, nq, nstates, nstates))
        
        X, Y, Z, Q = np.meshgrid(x, y, z, q, indexing='ij')
        
        freq_1   = 1015. * wavenum2au
        freq_6a  = 596. * wavenum2au
        freq_9a  = 1230. * wavenum2au
        freq_10a = 919. * wavenum2au

        Eshift = np.array([3.94, 4.89]) / au2ev
        kappa_1 = np.array([-0.0470, -0.0964, 0.1594, 0.]) / au2ev
        kappa_2 = np.array([-0.2012,  0.1193, 0.0484, 0.]) / au2ev
        gamma = np.array([-0.018]) / au2ev

        # The Hamiltonian only includes first order
        # v1 = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2. + kappa_1[0] * X + kappa_1[1] * Y + kappa_1[2] * Z + Eshift[0]
        # v2 = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2. + kappa_2[0] * X + kappa_2[1] * Y + kappa_2[2] * Z + Eshift[1]
        # coup = 0.1825 * Q /au2ev # vibronic coupling between S1 and S2
        # vg = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2.

        # The Hamiltonian includes both first order and second order 
        v1 = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2. + kappa_1[0] * X + kappa_1[1] * Y + kappa_1[2] * Z + Eshift[0] + gamma * Q**2
        v2 = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2. + kappa_2[0] * X + kappa_2[1] * Y + kappa_2[2] * Z + Eshift[1] + gamma * Q**2
        coup = 0.1825 * Q /au2ev # vibronic coupling between S1 and S2
        vg = freq_1 * X**2/2. + freq_6a * Y**2/2. + freq_9a * Z**2/2. + freq_10a * Q**2/2.

        v[:, :, :, :, 0, 0] = vg 
        v[:, :, :, :, 1, 1] = v1
        v[:, :, :, :, 2, 2] = v2
        v[:, :, :, :, 2, 1] = coup 
        v[:, :, :, :, 1, 2] = coup 
        
        self.v = v
        
        return v


    def apes(self):
        """
        Abatic PES (single_point)
        """
    
        v = dpes(x, y, z, q)
        # v = self.buildV()       
        w, u = np.linalg.eigh(v)
        return w, u


    def apes_global(self):
        """
        Abatic PES (global)
        """        
    
        x, y, z, q = self.x, self.y, self.z, self.q
        assert(x is not None)
    
        nstates = self.nstates
    
        nx = len(x)
        ny = len(y)
        nz = len(z)
        nq = len(q)        
    
        adiabatic_states = np.zeros((nx, ny, nz, nq, nstates, nstates))
        va = np.zeros((nx, ny, nz, nq, nstates))
    
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for l in range(nq):
                        # vd[i, j, k, l, :, :] 
                        _v = dpes(x[i], y[j], z[k], q[l])
                        w, u = np.linalg.eigh(_v)
                        va[i, j, k, l, :] = w
                        adiabatic_states[i, j, k, l] = u
    
        return va, adiabatic_states



def dpes(x, y, z, q, nstates = 3): #single_point
    """
    Diabatic PES (single_point)

    Parameters
    ----------
    x : TYPE
        qc coupling mode coordinate
    y : TYPE
        qt tuning mode coordinate

    Returns
    -------
    2D array
        molecular Hamiltonian

    """
    
    freq_1   = 1015. * wavenum2au
    freq_6a  = 596. * wavenum2au
    freq_9a  = 1230. * wavenum2au
    freq_10a = 919. * wavenum2au

    Eshift = np.array([3.94, 4.89]) / au2ev
    kappa_1 = np.array([-0.0470, -0.0964, 0.1594, 0.]) / au2ev
    kappa_2 = np.array([-0.2012,  0.1193, 0.0484, 0.]) / au2ev
    gamma = np.array([-0.018]) / au2ev

    # The Hamiltonian only includes first order 
    # v1 = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2. + kappa_1[0] * x + kappa_1[1] * y + kappa_1[2] * z + Eshift[0]
    # v2 = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2. + kappa_2[0] * x + kappa_2[1] * y + kappa_2[2] * z + Eshift[1]
    # coup = 0.1825 * q / au2ev # vibronic coupling between S1 and S2
    # vg = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2.

    # The Hamiltonian includes both first order and second order 
    v1 = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2. + kappa_1[0] * x + kappa_1[1] * y + kappa_1[2] * z + Eshift[0] + gamma * q**2
    v2 = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2. + kappa_2[0] * x + kappa_2[1] * y + kappa_2[2] * z + Eshift[1] + gamma * q**2
    coup = 0.1825 * q / au2ev # vibronic coupling between S1 and S2
    vg = freq_1 * x**2/2. + freq_6a * y**2/2. + freq_9a * z**2/2. + freq_10a * q**2/2.

    hmol = np.zeros((nstates, nstates))
    hmol[0, 0] = vg 
    hmol[1, 1] = v1
    hmol[2, 2] = v2
    hmol[2, 1] = coup 
    hmol[1, 2] = coup                          

    return hmol







if __name__ == '__main__':

    # from pyqed.models.pyrazine import Pyrazine
    from pyqed.phys import gwp, discretize
    from pyqed.units import au2fs
    import time    

    nstates = 3  
    dt = 0.5/au2fs
    nt = 1
    nout = 1
    ndim = 4    
    mass = [1/(1015. * wavenum2au), 1./(596. * wavenum2au), 1/(1230. * wavenum2au), 1./(919. * wavenum2au)]
    domains = [[-6, 6], ] * ndim 
    levels = [2， 2， 2， 2] 
    start_time = time.time()
                
    sol = LDRN(domains, levels, ndim=ndim, nstates=nstates, dvr_type='sinc', mass = mass)
    
    coord = sol.x
    n1, n2, n3, n4 = len(coord[0]), len(coord[1]), len(coord[2]), len(coord[3])        
    d1, d2, d3, d4 = interval(coord[0]), interval(coord[1]), interval(coord[2]), interval(coord[3])
    print(n1, n2, n3, n4)
    print(d1, d2, d3, d4)
    
    mol = Pyrazine(coord[0], coord[1], coord[2], coord[3])
    v = mol.buildV()  
    PES = mol.apes_global()
    print('The shape of global dpes of Pyrazine is: ', v.shape)  
    print('The shape of APES is', PES[0].shape)
    print('The shape of adiabatic_states is', PES[1].shape)
    
    sol.v = v        
    sol.apes = PES[0]
    sol.adiabatic_states = PES[1] 
    sol.A = sol.build_ovlp()
    print(sol.A.shape)
    print(sol.A.dtype)
    
    psi0 = np.zeros((n1, n2, n3, n4, nstates), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for m in range (n4):
                        psi0[i,j,k,m,2] = gwp(np.array([coord[0][i], coord[1][j], coord[2][k], coord[3][m]]), x0=[0,] * ndim, ndim=4)                
    print('The shape of psi0 is', psi0.shape)            
    
    norm = np.einsum('ijkla, ijkla', np.conj(psi0), psi0) * d1 * d2 * d3 * d4
    print("----Population of inital state----", norm)
         
    r = sol.run(psi0=psi0, dt=dt, nt=nt, nout=nout)
    r.dump('Result')   

    x, y, z, q = r.position() # np.savez('xAve', xAve, yAve) 保存了文件xAve.npz
    p0, p1, p2 = r.get_population()
            
    np.savez('position.npz', xAve=x, yAve=y, zAve=z, qAve=q)
    np.savez('population.npz', p0Ave=p0, p1Ave=p1, p2Ave=p2)
    np.save('psilist', r.psilist)
    
    end_time = time.time()        
    execution_time = end_time - start_time
    print(f"----- The total time of Sparse Grid is：{execution_time} seconds. -----")  
