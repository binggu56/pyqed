#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:09:59 2026


General Numerical Solver for  2D TDSE


@author: Bing Gu
"""

import numpy as np
import scipy

from numba import jit 
from scipy.fftpack import fft2, ifft2, fftfreq
from numpy.linalg import inv, det

from pyqed import interval 

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
    gauss_2d = np.zeros((len(x), len(y)), dtype=complex)

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


def k_evolve_2d(dt, mass, kx, ky, psi):
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
    mx, my = mass

    Kx, Ky = np.meshgrid(kx, ky)

    kin = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

    psi_k = kin * psi_k
    psi = ifft2(psi_k)

    return psi



class SPO2:
    """
    split-operator method for real-time dynamics of 2D systems
    """
    
    def __init__(self, x, y, v=None, mass=[1,1]):
        self.x = x 
        self.y = y 
        self.v = v
        
        self.nx = len(x)
        self.ny = len(y)
        self.dx = interval(x)
        self.dy = interval(y)
        
        self.mass = mass 
        
        
    
    def run(self, dt, psi0, nt=1, nout=10):
        """
        perform the propagation of the dynamics and calculate observables at
        every time step
        
        :param dt: time step
        :param v_2d: list
                    potential matrices in 2D
        :param psi_grid_0: list
                    the initial state
        :param num_steps: the number of the time steps
                       num_steps=0 indicates that no propagation has been done,
                       only the initial state and the initial purity would be
                       the output
        :return: psi_end: list
                          the final state
                 purity: float array
                          purity values at each time point
        """
        #f = open('density_matrix.dat', 'w')
        t = 0.0
        psi = psi0.copy()
    
        # purity = np.zeros(nt)
    
        # k-space grid
        kx = 2. * np.pi * fftfreq(nx, dx)
        ky = 2. * np.pi * fftfreq(ny, dy)
        
    
        dt2 = dt * 0.5 
        v = self.v 
            
        psi = np.exp(-1j * v * dt2) * psi
        
        for i in range(nt//nout):
            for k in range(nout):
                t += dt
                
                psi = k_evolve_2d(dt, self.mass, kx, ky, psi)
                psi =  np.exp(-1j * v * dt) * psi
            
            
            fig, ax = plt.subplots()
            ax.imshow(np.abs(psi)**2)
            
            # output_tmp = density_matrix(psi)
    
            #f.write('{} {} {} {} {} \n'.format(t, *rho))
            # purity[i] = output_tmp[4].real 
    
        return psi


######################################################################
# Helper functions for gaussian wave-packets


def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))


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



if __name__=='__main__':


    nx = 2 ** 6
    ny = 2 ** 6
    xmin = -6
    xmax = -xmin
    ymin = -6
    ymax = -ymin
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print('x range = ', x[0], x[-1])
    print('dx = {}'.format(dx))
    print('number of grid points along x = {}'.format(nx))
    print('y range = ', y[0], y[-1])
    print('dy = {}'.format(dy))
    print('number of grid points along y = {}'.format(ny))


    
    num_steps = 100
    dt = 0.05
    print('number of timesteps = ', num_steps)


    X, Y = np.meshgrid(x, y, indexing='ij')    
    # specify potential
    v_2d = X**2/2 + Y**2/2


    sigma = np.identity(2) 
    psi0 = gauss_x_2d(sigma, x0=-2, y0=-2, kx0=0, ky0=0) 
    
    spo = SPO2(x, y, v_2d, mass=[1,1])
    
    import ultraplot as plt 

    psi = spo.run(dt, psi0, num_steps)

    # store the final wavefunction
    #f = open('wft.dat','w')
    #for i in range(N):
    #    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    #f.close()

    
