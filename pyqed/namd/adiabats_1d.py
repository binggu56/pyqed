#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 10 11:14:55 2017

Solving the nuclear wavepacket dynamics on 1D adiabatic potential energy surface.

@author: Bing Gu

History:
2/12/18 : fix a bug with the FFT frequency

Possible improvements:
    1. use pyFFTW to replace the Scipy


"""

import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift
# from scipy.linalg import expm, sinm, cosm
import scipy

# import sys
# sys.path.append(r'C:\Users\Bing\Google Drive\lime')
# sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.phys import dagger, rk4


class NAMD:
    def __init__(self, x, nstates, mass, v, nac):
        """
        Non-adiabatic molecular dynamics (NAMD) simulations for one nuclear dof
            and many electronic states.

        Args:
            x: real array of size N
                grid points

            psi0: complex array [N, ns]
                initial wavefunction

            mass: float, nuclear mass

            nstates: integer, number of states

            v: ndarray [nx, nstates]
                adiabatic potential energy surfaces
            nac: ndarray (nx, nstates, nstates)
                non-adiabatic couplings. Diagonal elements vanish by definition.
        """
        self.x = x
        # self.psi0 = psi0
        self.mass = mass
        self.V_x = v
        self.v = v
        self.nstates = nstates
        self.nac = nac 

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

        for n in range(nstates):

            psi_k = fft(psi_x[:,n])

            psi_k *= np.exp(-0.5 * 1j / mass * (k * k) * dt)

            psi_x[:,n] = ifft(psi_k)

        return psi_x

    def spo(self, dt, psi_x, Nsteps = 1):

        """
        solve the time-dependent Schrodinger Equation with split-operator method.

        Parameters
        ----------
        dt : float
            time interval over which to integrate

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
        self.x_evolve(psi_x, vpsi2) # evolve V half step

        for i in range(Nsteps - 1):

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

    def evolve(self, psi0, dt=0.001, Nt=1,  t0=0., nout=1, coordinates='linear'):
        """
        Propagate the wavepacket dynamics

        Parameters
        ----------
        psi0 : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        t0 : TYPE, optional
            DESCRIPTION. The default is 0..
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        coordinates : TYPE, optional
            DESCRIPTION. The default is 'linear'.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        psi : TYPE
            DESCRIPTION.

        """
        
        psi = psi0
        t = t0
        x = self.x         
        
        nx = len(x)
        dx = x[1] - x[0]
        
        vmat = self.v 
        nac = self.nac 
        
        # momentum k-space 
        k = 2.0 * np.pi * scipy.fftpack.fftfreq(nx, dx)
        
        if coordinates == 'linear':
            print('The nuclear coordinate is linear.')
        
        elif coordinates == 'curvilinear':
            
            raise NotImplementedError('Kinetic energy operator for curvilinear\
                                      coordinates has not been implemented.')
        
        fig, ax = plt.subplots()
        
        for j in range(Nt//nout):
            for i in range(nout):
              
                t += dt
                psi = rk4(psi, hpsi, dt, x, k, vmat, nac)
                #output_tmp = density_matrix(psi)
        
                #f.write('{} {} {} {} {} \n'.format(t, *rho))
                #purity[i] = output_tmp
            
            
            # ax.plot(x, np.abs(psi[:,0]) + 0.1 * j)
            ax.plot(x, np.abs(psi[:,1]))
            
        return psi 
    
    
def density_matrix(psi_x,dx):
    """
    compute purity from the wavefunction
    """
    rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
    rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
    rho11 = 1. - rho00
    return rho00, rho01, rho01.conj(), rho11

def hpsi(psi, x, k, vmat, nac, coordinates='linear', use_nac2=False):
    """
    evaluate H \psi
    input:
        v: 1d array, adiabatic surfaces
        d: nonadiabatic couplings, matrix
        use_nac2: bool 
            indicator whether to include the second-order nonadiabatic couplings
    output:
        hpsi: H operators on psi
    """
    # v |psi>
#    for i in range(len(x)):
#        for j in range(len(y)):
#            v_tmp = np.diagflat(vmat[:][i,j])
#            array_tmp = np.array([psi[0][i, j], psi[1][i, j]])
#            vpsi = vmat.dot(array_tmp)
    # if nstates != len(vmat):
    #     sys.exit('Error: number of electronic states does not match
    #      the length of PPES matrix!')

    # APESs act on the wavefunction
    vpsi = np.einsum('in, in -> in', vmat, psi)
    #vpsi = [vmat[i] * psi[i] for i in range(nstates)]

    # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
    # D\grad |psi> = D(x) * F^{-1} F

    psi_k = np.zeros((nx, nstates), dtype=complex)
    dpsi = np.zeros((nx, nstates), dtype=complex)
    tpsi = np.zeros((nx, nstates), dtype=complex)
    kpsi = np.zeros((nx, nstates), dtype=complex)

    for n in range(nstates):
        psi_k[:,n] = fft(psi[:, n])

        # momentum operator operate on the WF
        kpsi[:,n] = -1j * k * psi_k[:, n]

        dpsi[:,n] = ifft(kpsi[:, n])


    # kinetic energy operator
    # if coordinates == 'linear':
    #     for a in range(nstates):
    #         tpsi[:,a] = ifft( k*k/2./mx * psi_k[:, a])

    # elif coordinates == 'curvilinear':
        
    #     raise NotImplementedError('Kinetic energy operator for the curvilinear\
    #                               coordinates has not been implemented.')
    for a in range(nstates):
        tpsi[:,a] = ifft( k*k/2./mx * psi_k[:, a])
            
    #     G = np.identity(2)

    #     for i in range(nx):
    #         for j in range(ny):
    #             #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

    #             for k in range(nstates):
    #                 tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))



    # NACs operate on the WF

    nacpsi = -np.einsum('imn, in -> im', nac, dpsi)/mx  # array with size nstates

    hpsi = tpsi + vpsi + nacpsi

    return -1j * hpsi

# def propagate_rk4(x, y, cav, dt, psi, num_steps=0):
#     """
#     perform the propagation of the dynamics using RK4 integrator
#     :param dt: time step
#     :param v_2d: list
#                 potential matrices in 2D
#     :param psi_grid_0: list
#                 the initial state
#     :param num_steps: the number of the time steps
#                    num_steps=0 indicates that no propagation has been done,
#                    only the initial state and the initial purity would be
#                    the output
#     :return: psi_end: list
#                       the final state
#              purity: float array
#                       purity values at each time point
#     """
#     #f = open('density_matrix.dat', 'w')
#     t = 0.

#     nstates = 2

#     # setup the adiabatic potential matrix
#     x = np.linspace(-8, 8)
#     vmat = apes(x) # list of APESs

#     # setup the dipole surface
#     # dip_mat = dipole(x)

#     # setup the polaritonic surfaces

#     nac = get_nac(x, nstates)

#     print('Propagation starts ...\n')

#     for i in range(num_steps):
#         t += dt
#         psi = rk4(psi, hpsi, dt, kx, ky, vmat)
#         #output_tmp = density_matrix(psi)

#         #f.write('{} {} {} {} {} \n'.format(t, *rho))
#         #purity[i] = output_tmp
#     return psi
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


def apes(x):
    v = np.zeros((nx, nstates))
    v[:, 0] = x**2/2.
    v[:, 1] = x**2 + 2

    return v

def get_nac(x):
    NAC = np.zeros((nx, nstates, nstates))
    NAC[:, 0, 1] = np.exp(-x**2/2.)
    NAC[:, 1, 0] = - NAC[:, 0 ,1]

    return NAC

######################################################################

if __name__ == '__main__':

    import time
    
    start_time = time.time()

    nstates = 2 # number of electronic states
    
    mx = 1.0 # mass
    
    dt = 0.001
    
    # setup the grid
    nx = 128
    x = np.linspace(-8, 8, nx)
    dx = x[1] - x[0]
    

    vmat = apes(x) # list of APESs
    
    # setup the nonadiabatic couplings
    nac = get_nac(x)
    
    # kx = 2.0 * np.pi * scipy.fftpack.fftfreq(nx, dx)
    
    # set initial state
    psi = np.zeros((nx, nstates), dtype=complex)
    psi[:, 0] = gwp(x, a=1.0, x0=1.0, k0=2.0)
        
    print('Propagation starts ...\n')
    # fig, ax = plt.subplots()
    
    # for j in range(Nt//nout):
    #     for i in range(nout):
    #         t += dt
    #         psi = rk4(psi, hpsi, dt, x, kx, vmat, nac)
    #         #output_tmp = density_matrix(psi)
    
    #         #f.write('{} {} {} {} {} \n'.format(t, *rho))
    #         #purity[i] = output_tmp
    
    #     ax.plot(x, np.abs(psi[:,0]) + 0.1 * j)
    #     ax.plot(x, psi[:,1].real)

    sol = NAMD(x, nstates=nstates, mass=mx, v=vmat, nac=nac)
    sol.evolve(psi0=psi, dt=dt, Nt=4000, nout=1000)

    print('Execution Time = {} s'.format(time.time() - start_time))









