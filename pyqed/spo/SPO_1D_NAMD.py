#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Numerical Solver for 1D Non-adiabatic dynamics with N electronic states

Created on Tue Oct 10 11:14:55 2017

@author: Bing Gu

History:
2/12/18 : fix a bug with the FFT frequency

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

from lime.phys import dagger, interval


def x_evolve(dt, x, V_x, psi_x):
    #if dt != self.dt_:
    #self.dt_ = dt

    for i in range(len(x)):

        Vmat = np.reshape(V_x[i,:], (2,2))
        w, U = scipy.linalg.eigh(Vmat)

        #print(np.dot(U.conj().T, Vmat.dot(U)))

        V = np.diagflat(np.exp(- 1j * w * dt))


        psi_x[i,:] = np.dot(U,V.dot(dagger(U))).dot(psi_x[i,:])
        #self.x_evolve = self.x_evolve_half * self.x_evolve_half
        #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
        #               (self.k * self.k) * dt)


def k_evolve(dt, k, psi_x):
    """
    one time step for exp(-i * K * dt)
    """

    for n in range(2):

        psi_k = fft(psi_x[:,n])
        #psi_k = fftshift(psi_k)

        psi_k *= np.exp(-0.5 * 1j / m * (k * k) * dt)

        psi_x[:,n] = ifft(psi_k)


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

def evolve(x, v, psi0, dt, nt=1, t0=0.):
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

    f = open('density_matrix.dat', 'w')
    t = t0
    psi_x = psi0
    dt2 = 0.5 * dt
    
    N = len(x)
    dx = interval(x)

    k = scipy.fftpack.fftfreq(N, dx)
    k[:] = 2.0 * np.pi * k[:]


    # SPO propagation
    x_evolve(dt2, x, v, psi_x)

    for i in range(nt - 1):

        t += dt

        k_evolve(dt, k, psi_x)
        x_evolve(dt, x, v, psi_x)

        # rho = density_matrix(psi_x, dx)
        # f.write('{} {} {} {} {} \n'.format(t, *rho))

    k_evolve(dt, k, psi_x)
    x_evolve(dt2, x, V_x, psi_x)

    t += dt
    f.close()

    return psi_x


######################################################################
# Helper functions for gaussian wave-packets

def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

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

######################################################################

# specify time steps and duration
dt = 0.01
N_steps = 1
t_max = 100
frames = int(t_max / float(N_steps * dt))

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
V_x[:,0] = (x-1.)**2/2.0
V_x[:,3] = (x+1.)**2/2.0
c = 0.5
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
psi_x0 = np.zeros((N,2), dtype=np.complex128)
psi_x0[:,0] = 1./np.sqrt(2.) * gauss_x(x, a, x0, k0) * np.exp(1j*angle)
psi_x0[:,1] = 1./np.sqrt(2.) * gauss_x(x, a, x0, k0)


# propagate
psi_x = evolve(dt=dt, x=x, v=V_x, psi0=psi_x0, nt=t_max)

import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
ax.plot(x, psi_x0)
ax.plot(x, psi_x)
plt.show()

# store the final wavefunction
f = open('wft.dat','w')
for i in range(N):
    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
f.close()


print('**********************')
print('  Mission Complete!   ')
print('**********************')
