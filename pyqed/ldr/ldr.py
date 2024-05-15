#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:44:37 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import numpy as np
from numpy import exp, pi, sqrt, meshgrid
from pyqed import transform, dag, isunitary, rk4, isdiag, sinc, sort, isherm, interval,\
    cartesian_product, discretize, norm2
from pyqed.wpd import ResultSPO2
from pyqed.ldr.gwp import WPD2
from pyqed.dvr.dvr_1d import HermiteDVR, SincDVR


import warnings
from opt_einsum import contract
from tqdm import tqdm

# from pyqed.wpd import Result

import scipy
import string

from scipy.linalg import inv
from scipy.sparse import kron, eye
from scipy.linalg import eigh
import logging

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

    return T


# def kinetic_energy_matrix(N):
#     """
#     Collbert-Miller DVR for an azimuthal coordinate [0, 2\pi] and the boundary conditions 
#     in this case are that the wave function is periodic.
    
#     The basis functions are 
#     .. math::
#         \chi_i(\phi) = \frac{1}{\sqrt{2 \pi}} e^{i n \phi}, n = 0, \pm 1, ... \pm N

#     Parameters
#     ----------
#     n : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    

def free_propagator(x, t, mass=1):
    """
    propagator of a one-dimensional free particle

    Parameters
    ----------
    mass : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    U = np.sqrt(mass/(2j * np.pi * t)) * np.exp(-mass * x**2/(2j * t))
        
    return U




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

class ResultLDR(ResultSPO2):
    def __init__(self, dx=None, **kwargs):
        super().__init__(**kwargs)
        
        
        self.dx = dx
    

    
    def get_population(self, fname=None, plot=True):
        
        d = np.prod(self.dx)
        p = np.zeros((len(self.psilist), self.nstates))
        for n in range(self.nstates):
            p[:, n] = [norm2(psi[:, :, n]).real * d for psi in self.psilist]
        # p1 = [norm2(psi[:, :, 1]) * dx * dy for psi in self.psilist]
        
        self.population = p
        if fname is not None:
            # fname = 'population'
            np.savez(fname, p)
            
            
        if plot:
            fig, ax = plt.subplots()
            for n in range(self.nstates):
                ax.plot(self.times, p[:,n])
            # ax.plot(self.times, p1)
            return fig, ax
        else:
                
            return p

class LDRN:
    """
    many-dimensional many-state nonadiabatic conical intersection dynamics in 
    DVR + LDR + SPO
    
    
    The required input to run is APES and electronic overlap matrix.
    
    
    This is extremely expansive, the maximum dimension should be < 4. 
    
    """
    def __init__(self, domains, levels, ndim=3, nstates=2, x0=None, mass=None, \
                 dvr_type='sine'): 

        assert(len(domains) == len(levels) == ndim)
        
        self.L = [domain[1] - domain[0] for domain in domains]

        
        x = []
        w = [] 
        dvr = []
        if dvr_type in ['sinc', 'sine']:
            # uniform grid 
            for d in range(ndim):
                l = levels[d]
                x.append(discretize(*domains[d], levels[d], endpoints=False))
                _w = [1/(2**l-1), ] * (2**l-1)
                w.append(_w)
                
        elif dvr_type == 'gauss_hermite':
            
            assert x0 is not None
            
            for d in range(ndim):
                _dvr = HermiteDVR(x0[d], levels[d])
                x.append(_dvr.x)
                w.append(_dvr.w)
                dvr.append(_dvr.copy())
        
        else:
            raise ValueError('DVR {} is not supported. Please use sinc.')
            
        
        self.x = x
        self.w = w # weights
        self.dvr = dvr 
        self.dx = [interval(_x) for _x in x]
        self.nx = [len(_x) for _x in x] 
        
        self.dvr_type = [dvr_type, ] * ndim
        
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass
        
        self.nstates = nstates
        self.ndim = ndim
        
        # all configurations in a vector
        self.points = np.fliplr(cartesian_product(x))
        self.npts = len(self.points)

        ###
        self.H = None
        self.K = None
        # self._V = None
        
        self._v = None
        self.exp_K = None
        self.exp_V = None
        self.exp_T = None # KEO in LDR
        self.wf_overlap = self.A = None
        self.apes = None
        
        
    @property
    def v(self):
        return self._v 
    
    @v.setter
    def v(self, v):
        assert(v.shape == (*self.nx, self.nstates, self.nstates))
        
        if abs(np.min(v)) > 0.1:
            raise ValueError('The PES minimum is not 0. Shift the PES.')
        
        self._v = v

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
        self.K = []
        
        for d in range(self.ndim):
                    
            Tx = kinetic(self.x[d], mass=self.mass[d], dvr=self.dvr_type[d])
            
            # we can use free-particle propagator for the e^{-i K \Delta t}
            expKx = scipy.linalg.expm(-1j * Tx * dt)
            
            self.exp_K.append(expKx.copy())
            self.K.append(Tx.copy())
            
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
    
    
    def gen_enisum_string(self, D):
        """
        Generate einsum string for computing the short-time propagator
        
        ij...a, ij...a, kl...b, lk...b -> ija, klb

        Parameters
        ----------
        D : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        alphabet = list(string.ascii_lowercase)
        if (D > 10):
            raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))
        
        s1 = "".join(alphabet[:D]) + 'x' 
        # s2 = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
        s3 = "".join(alphabet[D:2*D])+'y'
        s2 = s1 + s3 
        
        einsum_string = s1 + ',' + s2 + ',' + s3 + '->' + s2 

        return einsum_string
    
    def short_time_propagator(self, dt):
        
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
        exp_T = contract(einsum_string, self.A, *self.exp_K)
        
        
        einsum_string = self.gen_enisum_string(self.ndim)
        U = contract(einsum_string, self.exp_V_half, exp_T, self.exp_V_half)
        return U
    
    def buildH(self, dt):
        
        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()
        
        print('building the potential energy propagator')
        self.buildV(dt)
        
        print('building the kinetic energy propagator')
        self.buildK(dt)
        
        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()

        size = np.prod(self.nx) * self.nstates        
        
        einsum_string = gen_enisum_string(self.ndim)
        H = np.diag(self.apes.flatten()) + \
            contract(einsum_string, self.A, *self.K).reshape(size, size)
        
        
        self.H = H
        
        return H
    
            
    def run(self, psi0, dt, nt, nout=1, t0=0):
        
        assert(psi0.shape == (*self.nx, self.nstates))
        
        if self.H is None:
            self.buildH(dt)

    
        # T_{mn} A_{mb, na} = kinetic energy operator in LDR
        # if self.ndim == 2:
        
            # expKx, expKy = self.exp_K
        einsum_string = gen_enisum_string(self.ndim)
        exp_T = contract(einsum_string, self.A, *self.exp_K)
        # self.exp_T = exp_T
            
        r = ResultLDR(dx=self.dx, dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        # r.x = self.x
        # r.y = self.y
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
                # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)

                # psi = self._KEO_linear(psi)
                # psi = np.einsum('ijaklb, klb->ija', self.A, psi)
                psi = contract(_string, exp_T, psi) 
                psi = self.exp_V * psi 
                # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)
                
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
        vol = np.prod(self.dx)
        
        alphabet = list(string.ascii_lowercase)
        
        einsum_string = "".join(alphabet[:D]) + 'x, ' + "".join(alphabet[:D])+'y ->  xy'
        
        rho = np.einsum(einsum_string, psi.conj(), psi) * vol
        return rho


class LDRN_LvN(LDRN):
    """
    Liouville von Newmann equation in the LDR
    """
    def run(self, rho0, dt, nt, nout=1, t0=0):
        
        nx = self.nx
        nstates = self.nstates

        U = self.short_time_propagator(dt)
        
        size = np.prod(self.nx) * self.nstates
        
        print('Size of vibronic space ', size)
        
        U = np.reshape(U, (size, size))
        
        results = {} 
        results['rho'] = [rho0]
        
        rho0 = rho0.reshape((size, size))
        
        # times = np.arange(nt) * dt
        times = np.power(2, range(1, nt+1)) * dt
        results['times'] = times 
        
        U = np.identity(size)
        # t = t0
        for k in tqdm(range(nt)):
            # for k1 in range(nout):
                # t += dt
            U = U @ U
            
            rho = U @ rho0 @ dag(U)
            
            rho = rho.reshape((*nx, self.nstates, *nx, self.nstates))
            results['rho'].append(rho.copy())
            
        return results 
    
    def fast_LvN(self, rho0, dt, nt, nout=1, t0=0):
        """
        solve Liouville von Newnman equation by exponential integrator

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        nt : TYPE
            DESCRIPTION.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0 : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        
        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        U = self.short_time_propagator(dt)
        U = np.reshape(U, (nx*ny*nstates, nx*ny*nstates))
        
        results = {} 
        results['psilist'] = []
        
        rho0 = rho0.reshape((nx * ny * nstates, nx * ny * nstates))
        
        times = np.power(2, range(1, nt+1)) * dt
        results['times'] = times 
        
        for k in range(nt):
            U = U @ U
            rho = U @ rho0 @ dag(U)
            
            rho = rho.reshape((nx, ny, nstates, nx, ny, nstates))
            results['rho'].append(rho.copy())
            
        return results 
    
    def rdm_el(self, rho):
        """
        reduced electronic density matrix

        Parameters
        ----------
        rho : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """
        D = self.ndim 
        # vol = np.prod(self.dx)
        
        alphabet = list(string.ascii_lowercase)
        
        einsum_string = "".join(alphabet[:D]) + 'x ' + "".join(alphabet[:D])+'y ->  xy'
        
        rho_el = contract(einsum_string, rho)
        
        return rho_el
        
        


        

def gauss_hermite_quadrature(npts, xmax=None, x0=0.):
        assert (npts < 269), \
            "Must make npts < 269 for numpy to find quadrature points."
        n = np.arange(npts)
        c = np.zeros(npts+1)
        c[-1] = 1.
        x = np.polynomial.hermite.hermroots(c)
        if xmax is None:
            gamma = 1.
        else:
            assert xmax is None, "Sorry, xmax is currently broken"
            gamma = x.max() / float(xmax)

        x = x0 + x / gamma
        w = np.exp(-np.square(x))
        L = x.max() - x.min()
        a = None
        k_max = None
        return x, w 
        
class GaussHermiteLDRN(LDRN):
    
    def __init__(self, x0, levels, ndim=3, nstates=2, mass=None): 

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
        
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass
        
        self.nstates = nstates
        self.ndim = ndim
        
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
        


    def t(self, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t(V)

        @returns T kinetic energy matrix
        """
        _i = self.n[:, np.newaxis]
        _j = self.n[np.newaxis, :]
        _xi = self.x[:, np.newaxis]
        _xj = self.x[np.newaxis, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2.*(-1.)**(_i-_j)/(_xi-_xj)**2.

        T[self.n, self.n] = 0.
        T += np.diag((2. * self.npts + 1.
                      - np.square(self.x)) / 3.)
        T *= self.gamma
        T *= 0.5 * hc**2. / mc2   # (pc)^2 / (2 mc^2)
        return T

# class LDRN:
#     """
#     many-dimensional many-state nonadiabatic conical intersection dynamics in 
#     DVR + LDR + SPO
#     """
#     def __init__(self, domains, levels, ndim=3, nstates=2, mass=None, dvr_type='sinc'): 

#         assert(len(domains) == len(levels) == ndim)
        
#         self.L = [domain[1] - domain[0] for domain in domains]

        
#         x = []
#         if dvr_type in ['sinc', 'sine']:
        
#             for d in range(ndim):
#                 x.append(discretize(*domains[d], levels[d], endpoints=False))
#         # elif dvr_type == 'sine':
#         else:
#             raise ValueError('DVR {} is not supported. Please use sinc.')
            
        
#         self.x = x
#         self.dx = [interval(_x) for _x in x]
#         self.nx = [len(_x) for _x in x] 
        
#         self.dvr_type = dvr_type
        
#         if mass is None:
#             mass = [1, ] * ndim
        
#         self.nstates = nstates
        
#         # all configurations in a vector
#         self.points = np.fliplr(cartesian_product(x))
#         self.npts = len(self.points)

#         ###
#         self.H = None
#         self._K = None
#         # self._V = None
        
#         self._v = None
#         self.exp_K = None
#         self.exp_V = None
#         self.wf_overlap = self.A = None
        
        
#     @property
#     def v(self):
#         return self._v 
    
#     @v.setter
#     def v(self, v):
#         assert(v.shape == (*self.nx, self.nstates, self.nstates))
        
#         self._v = v

#     def buildK(self, dt):
#         """
#         For the kinetic energy operator with Jacobi coordinates

#             K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

#         Since the two KEOs for each dof do not commute, it has to be factorized as

#         e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}

#         where $p_\theta = -i \pa_\theta$ is the momentum operator.


#         Parameters
#         ----------
#         dt : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         """


#         self.exp_K = []
        
#         for d in self.ndim:
                    
#             Tx = kinetic(self.x[d], mass=self.mass[d], dvr=self.dvr)
        
#             expKx = scipy.linalg.expm(-1j * Tx * dt)

#             self.exp_K.append(expKx.copy())
            
#         return self.exp_K
    
#     def buildV(self, dt):
#         """
#         Setup the propagators appearing in the split-operator method.



#         Parameters
#         ----------
#         dt : TYPE
#             DESCRIPTION.

#         intertia: func
#             moment of inertia, only used for Jocabi coordinates.

#         Returns
#         -------
#         None.

#         """
        
#         dt2 = 0.5 * dt
#         self.exp_V = np.exp(-1j * dt * self.apes)

#         self.exp_V_half = np.exp(-1j * dt2 * self.apes)

#         return
    
#     def run(self, psi0, dt, nt, nout=1, t0=0):
        
#         assert(psi0.shape == (*self.nx, self.nstates))
        
#         if self.apes is None:
#             print('building the adibatic potential energy surfaces ...')
#             self.build_apes()
        
#         self.buildV(dt)
        
#         print('building the kinetic energy propagator')
#         self.buildK(dt)

        
#         if self.A is None:
#             logging.info('building the electronic overlap matrix')
#             self.build_ovlp()
        


#         # T_{mn} A_{mb, na} = kinetic energy operator in LDR
#         # if self.ndim == 2:
        
#             # expKx, expKy = self.exp_K
#         einsum_string = gen_enisum_string(self.ndim)
#         exp_T = np.einsum(einsum_string, self.A, *self.exp_K)
        
            
#         r = ResultSPO2(dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
#         r.x = self.x
#         r.y = self.y
#         r.psilist = [psi0]
        
#         alphabet = list(string.ascii_lowercase)
#         D = self.ndim
#         _string = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y, ' + \
#             "".join(alphabet[D:2*D])+'y -> ' + "".join(alphabet[:D]) + 'x'
            
#         psi = psi0.copy()
#         psi = self.exp_V_half * psi
#         # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)
        
#         for k in range(nt//nout):
#             for kk in range(nout):
#                 # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)

#                 # psi = self._KEO_linear(psi)
#                 # psi = np.einsum('ijaklb, klb->ija', self.A, psi)
#                 psi = np.einsum(_string, exp_T, psi) 
#                 psi = self.exp_V * psi 
#                 # psi = np.einsum('ija, ija -> ija', self.exp_V_half, psi)
                
#             r.psilist.append(psi.copy())
        
#         psi = self.exp_V_half * psi
        
#         return r
    
#     def rdm_el(self, psi):
#         """
#         compute the reduced electronic density matrices 

#         Parameters
#         ----------
#         psi : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         rho : TYPE
#             DESCRIPTION.

#         """        
#         D = self.ndim 
#         alphabet = list(string.ascii_lowercase)
        
#         einsum_string = "".join(alphabet[:D]) + 'x, ' + "".join(alphabet[:D])+'y ->  xy'
        
#         rho = np.einsum(einsum_string, psi.conj(), psi)
#         return rho
    

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
        self.nx = len(x)
        self.ny = len(y)
        
        self.size = self.nx * self.ny * nstates # size of Hilbert space
        
        self.kx = 2 * np.pi * scipy.fft.fftfreq(len(x), self.dx)
        self.ky = 2 * np.pi * scipy.fft.fftfreq(len(y), self.dy)


        
        self.dvr = dvr
        
        if mass is None:
            mass = [1, ] * ndim 
        self.mass = mass
        
        self.ngrid = [self.nx, self.ny]

        self.kmax = np.pi/self.dx # energy range [-K, K]
        self.shift = (self.nx - 1)//2
        
        self.nbasis = self.nx * self.ny # for product DVR
        
        self.X = None
        self.T = None
        self.V = None
        self.H = None 
        self._v = None # diabatic potential matrix
        
        
        self.geometies = None
        self.adiabatic_states = []
        self._apes = None
        self.electronic_overlap = self.A = None
        self.exp_T = None
        
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
    

    
    # def kinetic(self, npts, L, x0=0.):
    #     L = float(L)
    #     # self.npts = npts
    #     # self.L = L
    #     # self.x0 = x0
    #     dx = L / npts
    #     n = np.arange(npts)
    #     x = x0 + n * dx - L / 2. + dx / 2.
    #     # self.w = np.ones(npts, dtype=np.float64) * self.a
    #     # self.k_max = np.pi/self.a
        
    # # L = xmax - xmin 
    # # a = L / npts
    #     n = np.arange(npts)

    #     # self.w = np.ones(npts, dtype=np.float64) * self.a
    #     # self.k_max = np.pi/self.a



        
    #     _m = n[:, np.newaxis]
    #     _n = n[np.newaxis, :]
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2
            
    #     T[n, n] = np.pi**2. / 3. / dx**2
    #     T *= 0.5/mass   # (pc)^2 / (2 mc^2)

    
    # def kinetic(self):
        
    #     x = self.x         
    #     nx = self.nx
    #         # self.n = np.arange(npts)
    #         # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
    #         # self.w = np.ones(npts, dtype=np.float64) * self.a
    #         # self.k_max = np.pi/self.a
        
    #     L = x[-1] - x[0]
    #     dx = interval(x)
    #     n = np.arange(nx)


            
    #     # Colbert-Miller DVR 1992
        
    #     _m = n[:, np.newaxis]
    #     _n = n[np.newaxis, :]
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2
            
    #     T[n, n] = np.pi**2. / 3. / dx**2
    #     T *= 0.5/mass   # (pc)^2 / (2 mc^2)
    
    #     return T
    
    def buildK(self, dt):
        
        # dx = self.dx
        # Tx = np.zeros((self.nx, self.nx))
        mx, my = self.mass 
        

        # for n in range(nx):
        #     Tx[n, n] = np.pi**2/3 * 1/(2 * mx * dx**2)
            
        #     for m in range(n):
        #         Tx[n, m] = 1/(2 * mx * dx**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Tx[m, n] = Tx[n, m]
        

        
        Tx = kinetic(self.x, mass=mx, dvr=self.dvr)
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)

        
        Ty = kinetic(self.y, my, dvr=self.dvr)

        expKy = scipy.linalg.expm(-1j * Ty * dt)

        # x, y = self.x, self.y 
        # X = x[:, None] - x[None, :]
        # expKx = free_propagator(X, dt, mass=mx)
        
        # Y = y[:, None] - y[None, :]
        # expKy = free_propagator(Y, dt, mass=my)
 


        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y

        self.exp_K = [expKx, expKy]
        self.K = [Tx, Ty]
        
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
                #print(np.dot(U.conj().T, Vmat.dot(U)))

        # self.apes = va.reshape((self.nbasis, nstates))
        # self.adiabatic_states = u.reshape((self.nbasis, nstates, nstates))
        
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
        nstates = self.nstates
        nx, ny = self.nx, self.ny
        
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


        
        # for a in range(nstates):
        #     for b in range(nstates):        
        #         self._K[:, :, a, b] = A[:, :, a, b] * K 

        # self._K = np.transpose(self._K, (0, 2, 1, 3))

    
        A = np.transpose(A, (0, 1, 4, 2, 3, 5))
        self.A = A
        return A
        
        # return self._K

    def buildH(self):
        H = np.diag(self.apes) + self.build
    # def xmat(self, d=0):
    #     """
    #     position operator matrix elements 
        
    #     .. math::
    #         e^{i x_\mu}_{jk} = \braket{\phi_j | e^{i x_\mu} | \phi_k}

    #     Parameters
    #     ----------
    #     d: int
    #         which DOF
    #     shift : TYPE, optional
    #         DESCRIPTION. The default is False.

    #     Returns
    #     -------
    #     None.

    #     """

    #     N = self.nbasis
    #     x = np.zeros((N, N))

    #     # for k in range(N):
            

    #     # self.x[d] = x

    #     # return x
    #     return 
    
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
    
    def fast_run(self, psi0, dt, nt, nout=1, t0=0):
        
        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        U = self.short_time_propagator(dt)
        U = np.reshape(U, (nx*ny*nstates, nx*ny*nstates))
        
        results = {} 
        results['psilist'] = []
        
        psi0 = psi0.reshape((nx * ny * nstates))
        
        times = np.power(2, range(1, nt+1)) * dt
        results['times'] = times 
        
        for k in range(nt):
            U = U @ U
            psi = U @ psi0
            
            psi = psi.reshape((nx, ny, nstates))
            results['psilist'].append(psi.copy())
            
        return results 
    
    def fast_LvN(self, rho0, dt, nt, nout=1, t0=0):
        """
        solve Liouville von Newnman equation by exponential integrator

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        nt : TYPE
            DESCRIPTION.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0 : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        
        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        U = self.short_time_propagator(dt)
        U = np.reshape(U, (nx*ny*nstates, nx*ny*nstates))
        
        results = {} 
        results['psilist'] = []
        
        rho0 = rho0.reshape((nx * ny * nstates, nx * ny * nstates))
        
        times = np.power(2, range(1, nt+1)) * dt
        results['times'] = times 
        
        for k in range(nt):
            U = U @ U
            rho = U @ rho0 @ dag(U)
            
            rho = rho.reshape((nx, ny, nstates, nx, ny, nstates))
            results['rho'].append(rho.copy())
            
        return results 
    
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
    
    def short_time_propagator(self, dt):
        
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

        U = np.einsum('ija, ijaklb, klb -> ijaklb', self.exp_V_half, self.exp_T, self.exp_V_half)
        return U
    
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
    
    
class LDR2_Jacobi(LDR2):
    """
    LDRN solver for curvilinear coordinates with metric g_{\mu \nu}
    The kinetic energy operator reads 
    
    .. math::
        T_N = -\frac{1}{2} g_{\mu \nu}(\bf q) \frac{\partial}{\partial q^\mu}\
            \frac{\partial}{\partial q^\nu}
    """
    def __init__(self, domains, levels, dvr_types, mass=None, ndim=2):
        
        assert(len(domains) == len(levels) == ndim)
        
        self.L = [domain[1] - domain[0] for domain in domains]
        
        # center of grids
        self.x0 = [0.5 * (domain[1] + domain[0]) for domain in domains]
        
        x = []
        w = [] 
        dvr = []
        for d in range(ndim):
            dvr_type = dvr_types[d]
            
            if dvr_type in ['sinc', 'sine']:
            # uniform grid 
                l = levels[d]
                x.append(discretize(*domains[d], levels[d], endpoints=False))
                _w = [1/(2**l-1), ] * (2**l-1)
                
                w.append(_w)
                
            elif dvr_type == 'gauss_hermite':
                
                for d in range(ndim):
                    _dvr = HermiteDVR(self.x0[d], levels[d])
                    x.append(_dvr.x)
                    w.append(_dvr.w)
                    dvr.append(_dvr.copy())
            
            elif dvr_type == 'fourier':
                # _dvr = FourierDVR()
                pass
                
            else:
                raise ValueError('DVR {} is not supported. Please use sinc.'.format(dvr_type))
            
        
        self.x = x
        self.w = w # weights
        self.dvr = dvr 
        # self.dx = [interval(_x) for _x in x]
        self.nx = [len(_x) for _x in x] 
        
        self.dvr_types = dvr_types
        
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass
        
        self.nstates = nstates
        self.ndim = ndim
        
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
    
    def metric(self, q):
        # Cartesian/normal coordinates
        g = np.diag(1/self.mass)
        return g        
    
    def buildK(self):
        """
        For the kinetic energy operator with Jacobi coordinates
        
        .. math::

            K = \frac{p_r^2}{2\mu} + \frac{1}{2I(r)} p_\theta^2

        Since the two KEOs for each dof do not commute, it has to be factorized as

        .. math::
            
            e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}
        
        Does the ordering matter here?

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

        x, y = self.x 
        mx, moment_of_inertia = self.mass 
        
        I = moment_of_inertia(x)
        
        self.exp_K = []
        
        for d in range(self.ndim):
                    
            Tx = kinetic(self.x[d], mass=self.mass[d], dvr=self.dvr_type[d])
            
            # we can use free-particle propagator for the e^{-i K \Delta t}
            expKx = scipy.linalg.expm(-1j * Tx * dt)
            
            

            self.exp_K.append(expKx.copy())
            
        return self.exp_K

class LDR2_IT(LDR2):
    """
    imaginary-time local diabatic representation in 2D
    """
    def buildV(self, dt):
        """
        Setup the propagators appearing in the split-operator method.

        For the kinetic energy operator with Jacobi coordinates

            K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

        Since the two KEOs for each dof do not commute, it has to be factorized as

        e^{- K \tau} = e{- K_1 \delta \tau} e^{- i K_2 \delta t}

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
        
        dt2 = 0.5 * dt
        self.exp_V = np.exp(-  dt * self.apes)

        self.exp_V_half = np.exp(- dt2 * self.apes)
        
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
    
    def buildK(self, dt):
        
        # dx = self.dx
        # Tx = np.zeros((self.nx, self.nx))
        mx, my = self.mass 
        
        Tx = kinetic(self.x, mass=mx, dvr=self.dvr)
        
        expKx = scipy.linalg.expm(- Tx * dt)

        
        Ty = kinetic(self.y, my, dvr=self.dvr)

        expKy = scipy.linalg.expm(- Ty * dt)

        # x, y = self.x, self.y 
        # X = x[:, None] - x[None, :]
        # expKx = free_propagator(X, dt, mass=mx)
        
        # Y = y[:, None] - y[None, :]
        # expKy = free_propagator(Y, dt, mass=my)
 


        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y
        self.T = [Tx, Ty]
        self.exp_K = [expKx, expKy]
        return 
    
    def run(self, dt, nt):
        
        nx, ny, nstates = self.nx, self.ny, self.nstates
        
        U = self.short_time_propagator(dt)

        U = np.reshape(U, (nx*ny*nstates, nx*ny*nstates))
        
        results = {} 
        results['rho'] = []
        
        # psi0 = psi0.reshape((nx * ny * nstates))
        
        beta = np.power(2, range(1, nt+1)) * dt # inverse temperature
        results['beta'] = beta 
        
        for k in range(nt):
            U = U @ U

            results['rho'].append(U.copy())

            # psi = U @ psi0
            
            # psi = psi.reshape((nx, ny, nstates))
            # results['psilist'].append(psi.copy())
            
        return results 
    
    
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
        # return 1/sqrt(dx) * sinc(np.pi * (x/dx - n))
    

    
    # def kinetic(self, npts, L, x0=0.):
    #     L = float(L)
    #     # self.npts = npts
    #     # self.L = L
    #     # self.x0 = x0
    #     dx = L / npts
    #     n = np.arange(npts)
    #     x = x0 + n * dx - L / 2. + dx / 2.
    #     # self.w = np.ones(npts, dtype=np.float64) * self.a
    #     # self.k_max = np.pi/self.a
        
    # # L = xmax - xmin 
    # # a = L / npts
    #     n = np.arange(npts)

    #     # self.w = np.ones(npts, dtype=np.float64) * self.a
    #     # self.k_max = np.pi/self.a



        
    #     _m = n[:, np.newaxis]
    #     _n = n[np.newaxis, :]
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2
            
    #     T[n, n] = np.pi**2. / 3. / dx**2
    #     T *= 0.5/mass   # (pc)^2 / (2 mc^2)

    
    # def kinetic(self):
        
    #     x = self.x         
    #     nx = self.nx
    #         # self.n = np.arange(npts)
    #         # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
    #         # self.w = np.ones(npts, dtype=np.float64) * self.a
    #         # self.k_max = np.pi/self.a
        
    #     L = x[-1] - x[0]
    #     dx = interval(x)
    #     n = np.arange(nx)


            
    #     # Colbert-Miller DVR 1992
        
    #     _m = n[:, np.newaxis]
    #     _n = n[np.newaxis, :]
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2
            
    #     T[n, n] = np.pi**2. / 3. / dx**2
    #     T *= 0.5/mass   # (pc)^2 / (2 mc^2)
    
    #     return T
    
    def buildK(self, dt):
        
        # dx = self.dx
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
                #print(np.dot(U.conj().T, Vmat.dot(U)))

        # self.apes = va.reshape((self.nbasis, nstates))
        # self.adiabatic_states = u.reshape((self.nbasis, nstates, nstates))
        
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
    
    def build_ovlp(self):
        

        N = self.nbasis
        nstates = self.nstates
        nx, ny = self.nx, self.ny
        
        # K = self.buildK().reshape((N, N))
        
        # overlap of electronic states
        A = np.zeros((nx, ny, nx, ny, nstates, nstates), dtype=complex)
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


        
        # for a in range(nstates):
        #     for b in range(nstates):        
        #         self._K[:, :, a, b] = A[:, :, a, b] * K 

        # self._K = np.transpose(self._K, (0, 2, 1, 3))

    
        A = np.transpose(A, (0, 1, 4, 2, 3, 5))
        self.A = A
        return A
        
        # return self._K

        
    # def xmat(self, d=0):
    #     """
    #     position operator matrix elements 
        
    #     .. math::
    #         e^{i x_\mu}_{jk} = \braket{\phi_j | e^{i x_\mu} | \phi_k}

    #     Parameters
    #     ----------
    #     d: int
    #         which DOF
    #     shift : TYPE, optional
    #         DESCRIPTION. The default is False.

    #     Returns
    #     -------
    #     None.

    #     """

    #     N = self.nbasis
    #     x = np.zeros((N, N))

    #     # for k in range(N):
            

    #     # self.x[d] = x

    #     # return x
    #     return 
    
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


class SineDVR2(SincDVR2):
    """
    https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
    """
    
    def buildK(self, dt):
 
        mx, my = self.mass 
        
        Tx = kinetic(self.x, mx, dvr='sine')
        
        # p2 = kron(p2, np.eye(self.ny))
        
        expKx = scipy.linalg.expm(-1j * Tx * dt)
        
        Ty = kinetic(self.y, my, dvr='sine')
        
        # p2_y = kron(np.eye(self.nx, p2_y))
        expKy = scipy.linalg.expm(-1j * Ty * dt)

        # return -0.5/self.mass[0] * p2 - 0.5/self.mass[1] * p2_y

        self.exp_K = [expKx, expKy]
        return 
    
# def position_eigenstate(x, n, npts, L):
#     """
#     nth position eigenstate for sine DVR, n= 1, \cdots, npts
    
#     Both ends are not included in the grid.

#     Parameters
#     ----------
#     x : TYPE
#         DESCRIPTION.
#     n : TYPE
#         DESCRIPTION.
#     npts : TYPE
#         DESCRIPTION.
#     L : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     chi : TYPE
#         DESCRIPTION.

#     """
#     dx = L/(npts + 1)
#     xn = grid[n]
#     chi = np.sin(np.pi/2 *(2 * npts + 1) * (x - xn)/L)/np.sin(np.pi/2 * (x - xn)/L) -\
#           np.sin(np.pi/2 *(2 * npts + 1) * (x + xn)/L)/np.sin(np.pi/2 * (x + xn)/L)
#     chi *= 1/2/np.sqrt(L * (npts + 1))
    
#     return chi
    
        

class SincDVR_PBC(SincDVR2):
    
    def buildK(self, dt):
 
        mx, my = self.mass 
        

        # for n in range(nx):
        #     Tx[n, n] = np.pi**2/3 * 1/(2 * mx * dx**2)
            
        #     for m in range(n):
        #         Tx[n, m] = 1/(2 * mx * dx**2) * (-1)**(n-m) * 2/(n-m)**2
        #         Tx[m, n] = Tx[n, m]
        
        Tx = kinetic(self.x, mx, dvr='sinc_periodic')
        
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


if __name__ == '__main__':

    
    # from pyqed.models.pyrazine import DHO
    from pyqed import sigmaz, interval, sigmax, norm, gwp
    from pyqed.models.pyrazine import Pyrazine
    from pyqed.phys import gwp, discretize
    from pyqed.units import au2fs, au2wavenumber
    from pyqed.wpd import SPO2
    import time    


    def diabatic_potential(x, y, e_so, omega, kappa, g):
        xp = x + 1j * y
        # xm = x - 1j * y        
        h = np.diag([e_so/2, -e_so/2, -e_so/2, e_so/2])+0j
        h[0, 1] = kappa * xp 
        h[0, 2] = g/2 * xp**2
        h[1, 3] = -g/2 * xp**2
        h[2, 3] = kappa * xp
        
        h[1, 0] = h[0, 1].conj()
        h[2, 0] = h[0, 2].conj()
        h[3, 1] = h[1, 3].conj()
        h[3, 2] = h[2, 3].conj()
        return h + np.eye(4) * omega/2 * (x**2 + y**2)


    x = discretize(-6, 6, 5, endpoints=False)
    y = discretize(-6, 6, 5, endpoints=False)
    
    nx = len(x)
    ny = len(y)
    
    nb = len(x) * len(y)
    dx = interval(x)
    dy = interval(y)



    
    nstates = 4
    #mass = mol.mass
        

 
    omega = 700 / au2wavenumber
    e_so = 0.2 * omega    
    kappa = 0.1 * omega
    g = 0.0 * omega 

    v = np.zeros((nx, ny, nstates, nstates), dtype=complex)
    
    for i in range(nx):
        for j in range(ny):
            v[i,j] = diabatic_potential(x[i], y[j], e_so, omega, kappa, g)
     
    print('The shape of global diabatic potential matrix is: ', v.shape)   
    
    
    # ---------------------------
    # ------ LDR dynamics -------
    # ---------------------------
    start_time = time.time()
    
    domains = [[-6,6], ]*2
    levels = [5, ] * 2 
    solver = LDR2(x, y, nstates = nstates, \
                  mass = [1/omega, ] * 2, ndim=2) # mol.mass = [230.5405791702069, 367.62919827476884]

    solver.v = v
    # solver.build_apes()
    
    psi0 = np.zeros((nx, ny, nstates), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            psi0[i,j,3] = gwp(np.array([x[i], y[j]]), x0=[0, 0], ndim=2)
    
    # psi0 = np.reshape(psi0, (nx*ny*nstates,1))
    
    # transfrom the initial state to the adiabatic representation
    # psi0 = np.einsum('ija, ijab -> ijb', psi0, solver.adiabatic_states)

    dt = 0.2/au2fs
    nt = 100
    nout = 1
    
    def rdm(psi):
        rho = np.einsum('ija, ijb -> ab', psi.conj(), psi) * dx * dy
        return rho 
                
    # U = solver.short_time_propagator(dt)
    # U = np.reshape(U, (nx*ny*nstates, nx*ny*nstates))
    
    # for k in range(6):
    #     U = U @ U
    #     psi = U @ psi0
    #     psi = psi.reshape((nx, ny, nstates))
    #     rho = rdm(psi)
    #     print(rho[0,0], rho[1,1], rho[2,2], rho[3,3])
    
    result = solver.run(psi0=psi0, dt=dt, nt=nt, nout=nout)

    # fig, ax = plt.subplots()
    # for n in range(nstates):
    #     ax.plot(x, solver.apes[:, ny//2, n], label='{}'.format(n))
    # ax.legend()
    
        
    end_time = time.time()    
    execution_time = end_time - start_time
    print(f"----- The total time of exact dynamics is： {execution_time} seconds. -----")    
             
    result.dump('ldr') # level = 6
            
    result.get_population(plot=True) 
    result.position(plot=True)
    result.plot_wavepacket(result.psilist[0])
    result.plot_wavepacket(result.psilist[-1])