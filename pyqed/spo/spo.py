#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Numerical Solver for 1D wavepacket dynamics on a single PES.

Created on Tue Oct 10 11:14:55 2017

@author: Bing Gu

Several possible improvements:
    1. use pyFFTW to replace the Scipy

"""

import numpy as np
from matplotlib import pyplot as pl
from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift
from scipy.linalg import expm, sinm, cosm
import scipy
from numba import jit
from numpy import pi 

from pyqed import dagger, interval


def x_evolve(dt, x, v, psi):

    return np.exp(- 1j * v * dt) * psi



def k_evolve(dt, k, psi_x):
    """
    one time step for exp(-i * K * dt)
    """
    
    psi_k = fft(psi_x)
    #psi_k = fftshift(psi_k)

    psi_k *= np.exp(-0.5 * 1j / m * (k * k) * dt)

    return ifft(psi_k)
    

    #psi_x = property(_get_psi_x, _set_psi_x)
    #psi_k = property(_get_psi_k, _set_psi_k)
    #dt = property(_get_dt, _set_dt)

#    def compute_k_from_x(self, psi_x):
#        psi_k = fft(psi_x)
#        return psi_k
#
#    def compute_x_from_k(self, psi_k):
#        psi_x = ifft(psi_k)
#        return psi_x

def adiabatic_1d(x, v, psi0, dt, Nt=1, t0=0.):
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

    f = open('density_matrix.dat', 'w')
    t = t0
    psi_x = psi0.copy()
    dt2 = 0.5 * dt
    
    N = len(x)
    dx = interval(x)
    
    k = 2. * pi * scipy.fftpack.fftfreq(N, dx)
    # k[:] = 2.0 * np.pi * k[:]


    # SPO propagation
    x_evolve(dt2, x, v, psi_x)

    for i in range(nt - 1):

        t += dt

        psi_x = k_evolve(dt, k, psi_x)
        psi_x = x_evolve(dt, x, v, psi_x)

        # rho = density_matrix(psi_x, dx)
        # f.write('{} {} {} {} {} \n'.format(t, *rho))

    psi_x = k_evolve(dt, k, psi_x)
    psi_x = x_evolve(dt2, x, v, psi_x)

    t += dt
    f.close()

    return psi_x


######################################################################
# Helper functions for gaussian wave-packets

def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return (a/np.sqrt(np.pi))**(-0.25)*\
        np.exp(-0.5 * a * (x - x0)**2 + 1j * (x-x0) * k0)

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


def density_matrix(psi_x,dx):
    """
    compute purity from the wavefunction
    """
    rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
    rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
    rho11 = 1. - rho00
    return rho00, rho01, rho01.conj(), rho11


class SPO:
    """
    1D split-operator method for adiabatic wave packet dynamics 
    """
    def __init__(self, x, v, mass=1):
        self.x = x 
        self.v = v 
        self.mass = mass
    
    def run(self, psi0, dt, nt):
        x = self.x 
        v = self.v 
        psi_x = adiabatic_1d(dt=dt, x=x, v=v, psi0=psi0, Nt=nt)
        return psi_x
    
######################################################################
if __name__ == '__main__':
    
    from pyqed.units import au2fs, au2ev, au2wavenumber
    import matplotlib.pyplot as plt 
    
    # specify time steps and duration
    dt = 0.02/au2fs
    nt = 400
    # frames = int(t_max / float(N_steps * dt))
    
    # specify constants
    hbar = 1.0   # planck's constant
    m = 1.0      # particle mass
    
    # specify range in x coordinate
    N = 256
    xmin = -6
    xmax = -xmin
    #dx = 0.01
    #x = dx * (np.arange(N) - 0.5 * N)
    x = np.linspace(-8,8,128)
    
    print('x range = ',x[0], x[-1])
    # print('dx = {} \n'.format(dx))
    print('number of grid points = {}\n'.format(N))
    
    # specify potential
    #V0 = 1.5
    #L = hbar / np.sqrt(2 * m * V0)
    #a = 3 * L
    
    # diabatic surfaces with vibronic couplings
    # V_x = np.zeros((N,4))
    # omega = 2000/au2wavenumber
    # print('period = {} fs'.format(2*pi/omega * au2fs))
    omega = 1
    v = x**2/2.0 * omega**2
    
    fig, ax = plt.subplots()
    ax.plot(x, v)
    
    # specify initial momentum and quantities derived from it
    #p0 = np.sqrt(2 * m * 0.2 * V0)

    #dp2 = p0 * p0 * 1./80
    #d = hbar / np.sqrt(2 * dp2)
    
    
    psi_x0 =  gauss_x(x, a=(omega), x0=-2, k0=0)
    # psi_x0[:,1] = 1./np.sqrt(2.) * gauss_x(x, a, x0, k0)
    
    
    
    
    fig, ax = plt.subplots()
    ax.plot(x, np.abs(psi_x0))
    
    
    # propagate
    psi_x = SPO(x, v).run(dt=dt, psi0=psi_x0, nt=nt)
    
    ax.plot(x, np.abs(psi_x))
    
    # store the final wavefunction
    # f = open('wft.dat','w')
    # for i in range(N):
    #     f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    # f.close()
    
    
    print('**********************')
    print('  Mission Complete!   ')
    print('**********************')
