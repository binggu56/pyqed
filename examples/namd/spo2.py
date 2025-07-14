#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:39:25 2023

@author: bingg
"""
import numpy as np
from pyqed import au2fs, gwp, meshgrid
from scipy.fftpack import fft2, ifft2, fftfreq, fft, ifft

from pyqed.wpd import SPO2

import ultraplot as plt

if __name__ == '__main__':
    
    # specify time steps and duration
    ndim = 2

    # define 3D grids
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


    mass = [1.0, 1.0]  # particle mass

    # initial wavepacket
    x0, y0, kx0, ky0 = -1, 0, 0.0, 0

    sigma = np.identity(ndim) * 2.
    ns = nstates = 2
    psi0 = np.zeros((nx, ny, ns), dtype=complex)
    
    for i in range(nx):
        for j in range(ny):
            psi0[i,j,1] = gwp(np.array([x[i], y[j]]), sigma, \
                                      np.array([x0, y0]), np.array([kx0, ky0]), ndim=2)


    
    fig, ax = plt.subplots()
    ax.imshow(np.abs(psi0[:, :, 1]).T, cmap='viridis')
    # ax.format(title='Initial wavepacket')
    
    # dynamics
    sol = SPO2(nstates=2, mass=mass, x=x, y=y)
    
    # set up DPEM
    X, Y = meshgrid(x, y)

    v0 = 0.5 * ((X+1)**2 + Y**2)
    v1 = 0.5 * ((X-1)**2 + Y**2)
    
    sol.set_DPES(surfaces = [v0, v1], diabatic_couplings = [[[0, 1],  0.2 * X]])

    dt = 0.2
    nt = 100
    print('time step = {} fs'.format(dt * au2fs))
    r = sol.run(psi0=psi0, dt=dt, nt=nt, nout=10)
    

    for n in range(0, 10):
        fig, ax = plt.subplots()
        ax.imshow(np.abs(r.psilist[n][:, :, 1]).T, cmap='viridis')
    
    plt.show()