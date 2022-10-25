#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:12:19 2020

@author: Bing Gu 
"""


import numpy as np
from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift

from scipy.linalg import expm, sinm, cosm
import scipy

import sys

def dagger(a):
    return a.conjugate().transpose()

class NAMD:
    def __init__(self, x, nstates, psi0, mass, V_x):
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
        self.V_x = V_x
        self.nstates = nstates 

    def x_evolve(self, dt, psi, vpsi):
        """
        vpsi = exp(-i V dt)
        """

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

    def propagate(self, dt, psi_x, Nsteps = 1):

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
        
        t = 0.0 
        self.x_evolve(dt2, psi_x, vpsi2)

        for i in range(Nsteps - 1):

            t += dt

            psi_x = self.k_evolve(dt, k, psi_x)
            psi_x = self.x_evolve(dt, psi_x, vpsi)

            rho = density_matrix(psi_x, dx)
            
            # store the density matrix 
            f.write('{} {} {} {} {} \n'.format(t, *rho))

        # psi_x = self.k_evolve(dt, psi_x)
        # psi_x = self.x_evolve(dt2, psi_x, vpsi2)


        f.close()

        return psi_x

def density_matrix(psi_x,dx):
    """
    compute purity from the wavefunction
    """
    rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
    rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
    rho11 = 1. - rho00
    return rho00, rho01, rho01.conj(), rho11

######################################################################
# Helper functions for gaussian wave-packets

def gwp(x, a, x0, k0):
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




######################### main program #############################################

nx = 256
mass = 1. 
nstates = 2 

x = np.linspace(-12, 12, nx) # grid 

# initial wavepackets on ALL surfaces 
psi0 = np.zeros((len(x), nstates), dtype=complex) 
psi0[:, 1] = gwp(x, 1.0, 0.0, 0.0)


# Diabatic surfaces 
V_x = np.zeros((nx, nstates**2))
V_x[:,0] =  0.5*x**2 
V_x[:,2] = V_x[:,1] = 0.
V_x[:,3] = .5*(x-1)**2 + 2.0  


mol = NAMD(x, nstates, psi0, mass, V_x)

dt = 0.002

 
import matplotlib.pyplot as plt 
fig, (ax1,ax2) = plt.subplots(nrows=2)

psi = psi0.copy()

for k in range(5):
    
    print('Propagating the wavefunction ...')

    psi = mol.propagate(dt, psi, Nsteps=600)
    ax1.plot(x, psi[:,0].real)
    ax2.plot(x, np.abs(psi[:,1]))

