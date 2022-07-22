#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Numerical Solver for non-adiabatic molecular dynamics with multi potential
energy surfaces in 2-dimensional nuclear space 


@author: Bing Gu
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from numba import autojit 
from scipy.fftpack import fft2, ifft2, fftfreq
from numpy.linalg import inv, det


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
    gauss_2d = np.zeros((len(x), len(y)))

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



def vpsi(dt, v_2d, psi_grid):
    """
    propagate the state in grid basis half time step forward with 
    Potential Energy Operator
    
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

    for i in range(len(x)):
        for j in range(len(y)):
            v_mat = np.array([[v_2d[0][i, j], v_2d[1][i, j]],
                             [v_2d[2][i, j], v_2d[3][i, j]]])

            w, u = scipy.linalg.eigh(v_mat)
            v = np.diagflat(np.exp(-1j * w * dt))
            
            array_tmp = np.array([psi_grid[0][i, j], psi_grid[1][i, j]])
            array_tmp = np.dot(u.conj().T, v.dot(u)).dot(array_tmp)
            
            psi_grid[0][i, j] = array_tmp[0]
            psi_grid[1][i, j] = array_tmp[1]

            #self.x_evolve = self.half * self.x_evolve_half
            #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
            #               (self.k * self.k) * dt)


def k_evolve_2d(dt, kx, ky, psi_grid):
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
    for i in range(2):
        psi_k_tmp = fft2(psi_grid[i])
        for j in range(len(kx)):
            for k in range(len(ky)):
                psi_k_tmp[j, k] *= np.exp(-0.5 * 1j / m *
                                          (kx[j]**2+ky[k]**2) * dt)
        psi_grid[i] = ifft2(psi_k_tmp)

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



def spo_dynamics(dt, v_2d, psi0, num_steps=0):
    """
    perform the propagation of the dynamics and calculate the purity at
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
    psi_grid = psi0

    purity = np.zeros(num_steps)
    #purity[0] = density_matrix(psi_grid)[4].real 

    kx = fftfreq(nx, dx)
    ky = fftfreq(ny, dy)

    dt2 = dt * 0.5 
    
    vpsi(dt2, v_2d, psi_grid)

    for i in range(num_steps):
        t += dt
        
        k_evolve_2d(dt, kx, ky, psi_grid)
        vpsi(dt, v_2d, psi_grid)
        
        output_tmp = density_matrix(psi_grid)

        #f.write('{} {} {} {} {} \n'.format(t, *rho))
        purity[i] = output_tmp[4].real 

    #k_evolve_2d(dt, kx, ky, psi_grid)
    #vpsi(dt, v_2d, psi_grid)

    # t += dt
    #f.close()

    return psi_grid, purity


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


def density_matrix(psi_grid):
    """
    compute purity from the wavefunction
    """
    rho00 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[0]))*dx*dy
    rho01 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[1]))*dx*dy
    rho11 = np.sum(np.multiply(np.conj(psi_grid[1]), psi_grid[1]))*dx*dy

    purity = rho00**2 + 2*rho01*rho01.conj() + rho11**2

    return rho00, rho01, rho01.conj(), rho11, purity



def main(dt, sigma, x0, y0, kx0, ky0, coeff1, x_range_half, y_range_half,
         phase, couple_strength, couple_type, time_end):
    """
    the main porgram for the 2D nonadiabatic quantum dynamics
    :param sigma:
    :param x0:
    :param y0:
    :param kx0:
    :param ky0:
    :param coeff1:
    :param x_range_half:
    :param y_range_half:
    :param phase:
    :param couple_strength:
    :param couple_type:
    :param time_end:
    :return:
    """
    # specify time steps and duration
    num_steps = 40
    print('number of timesteps = ', num_steps)
    dt = 0.01
    tlist = np.arange(num_steps) * dt 

    # specify potential
    v_2d = potential_2d(x_range_half, y_range_half, couple_strength,
                        couple_type)

    # setup the initial state
    coeff2 = np.sqrt(1-coeff1**2)

    psi0 = [coeff1 * gauss_x_2d(sigma, x0, y0, kx0, ky0) * np.exp(1j*phase),
                  coeff2 * gauss_x_2d(sigma, x0, y0, kx0, ky0)]

    # propagate
    psi, purity = spo_dynamics(dt, v_2d, psi0, num_steps)

    # store the final wavefunction
    #f = open('wft.dat','w')
    #for i in range(N):
    #    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    #f.close()

    return tlist, purity

# covergence test wrt the grid points
for i in range(5, 6):
    # the variables below are global variables for this module
    nx = 2 ** i
    ny = 2 ** i
    xmin = -11
    xmax = -xmin
    ymin = -11
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

    xv, yv = np.meshgrid(x, y)
    xv = xv.T
    yv = yv.T

    # specify constants
    m = 1.0  # particle mass

    # test the main
    sigma_tmp = np.identity(2) * 2.
    t_axis, purity = main(0.001, sigma_tmp, -3, -3, 0, 0, 1./2.,
                                             3, 3, 0, 2., 0,
                                             1)
    plt.plot(t_axis, purity)

plt.show()

# test the 2d gaussian distribution
# psigrid = ['', '']
# psigrid[0] = 1/np.sqrt(2) * gauss_x_2d(sigma_tmp, -3, -3, 0, 0)
# psigrid[1] = 1/np.sqrt(2) * gauss_x_2d(sigma_tmp, -3, -3, 0, 0)
# plt.imshow(psigrid)
# plt.show()

# test the 2D potential
# v_list = potential_2d(3, 3, 2, 1)
# plt.figure(1)
# plt.imshow(v_list[0])
# plt.figure(2)
# plt.imshow(v_list[3])
# plt.show()

# test the purity calculation
# print(density_matrix(psigrid))

# test the x_evolve
# vpsi(0.01, v_list, psigrid)


# test the k_evolve
# kx = fftfreq(nx, dx)
# ky = fftfreq(ny, dy)
# k_evolve_2d(0.01, kx, ky, psigrid)
# vpsi(0.01, v_list, psigrid)
# print(density_matrix(psigrid))

# test dynamics
# print(spo_dynamics(0.01, v_list, psigrid, num_steps=2)[1])

