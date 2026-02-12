#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 10 11:14:55 2017

Solving the nuclear wavepacket dynamics on 1D/2D adiabatic potential energy surface.

@author: Bing Gu

History:
2/12/18 : fix a bug with the FFT frequency

Possible improvements:
    1. use pyFFTW to replace the Scipy

General Numerical Solver for Non-adiabatic dynamics with many electronic
state and two vibrational coordinates in adiabatic representation

integrator: RK4  + FFT

@author: Bing Gu
@data: Sep 11, 2019

Todo
1. This code should be benchmarked with the diabatic code using e.g. the pyrazine model.

"""

import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift
# from scipy.linalg import expm, sinm, cosm
import scipy



from numpy import cos, pi, sqrt

from scipy import linalg
from scipy.sparse import kron


from numba import jit
from scipy.fftpack import fft2, ifft2, fftfreq

# the scipy fft2 program has to be modified to match normal definitions
# from numpy.linalg import inv, det


from pyqed.phys import rk4, tdse, heaviside, get_index, dag, interval, dagger
from pyqed import au2ev, au2fs

from pyqed.wpd import adiabatic_2d, KEO, PEO
from pyqed.fft import dft2


# def sinc(x):
#     return np.sinc(x/np.pi)


@jit
def gauss_x_2d(x, y, sigma, x0, y0, kx0, ky0):
    """
    generate the gaussian distribution in 2D grid
    :param x0: float, mean value of gaussian wavepacket along x
    :param y0: float, mean value of gaussian wavepacket along y
    :param sigma: float array, covariance matrix with 2X2 dimension
    :param kx0: float, initial momentum along x
    :param ky0: float, initial momentum along y
    :return: gauss_2d: float array, the gaussian distribution in 2D grid
    """
    sigmax, sigmay = sigma[0, 0], sigma[1,1]

    gauss_2d = (1./ np.sqrt(sigmax * sigmay) / np.sqrt(np.pi)) \
                              * np.exp(-0.5 * (x-x0)**2/sigmax**2 - 0.5 * (y-y0)**2/sigmay**2)\
                            * np.exp(- 1j * kx0 * (x-x0) - 1j * ky0 * (y-y0))

    return gauss_2d


# @jit
# def diabatic(x, y):
#     """
#     PESs in diabatic representation
#     :param x_range_half: float, the displacement of potential from the origin
#                                 in x
#     :param y_range_half: float, the displacement of potential from the origin
#                                 in y
#     :param couple_strength: the coupling strength between these two potentials
#     :param couple_type: int, the nonadiabatic coupling type. here, we used:
#                                 0) no coupling
#                                 1) constant coupling
#                                 2) linear coupling
#     :return:
#         v:  float 2d array, matrix elements of the DPES and couplings
#     """
#     nstates = 2

#     v = np.zeros((nstates, nstates))

#     v[0,0] = (x + 4.) ** 2 / 2.0 + (y + 3.) ** 2 / 2.0
#     v[1,1] = (x - 4.) ** 2 / 2.0 + (y - 3.) ** 2 / 2.0

#     v[0, 1] = v[1, 0] = 0

#     return v

# @jit
# def x_evolve_half_2d(dt, v_2d, psi_grid):
#     """
#     propagate the state in grid basis half time step forward with H = V
#     :param dt: float
#                 time step
#     :param v_2d: float array
#                 the two electronic states potential operator in grid basis
#     :param psi_grid: list
#                 the two-electronic-states vibrational state in grid basis
#     :return: psi_grid(update): list
#                 the two-electronic-states vibrational state in grid basis
#                 after being half time step forward
#     """

#     for i in range(len(x)):
#         for j in range(len(y)):
#             v_mat = np.array([[v_2d[0][i, j], v_2d[1][i, j]],
#                              [v_2d[2][i, j], v_2d[3][i, j]]])

#             w, u = scipy.linalg.eigh(v_mat)
#             v = np.diagflat(np.exp(-0.5 * 1j * w / hbar * dt))
#             array_tmp = np.array([psi_grid[0][i, j], psi_grid[1][i, j]])
#             array_tmp = np.dot(u.conj().T, v.dot(u)).dot(array_tmp)
#             psi_grid[0][i, j] = array_tmp[0]
#             psi_grid[1][i, j] = array_tmp[1]
#             #self.x_evolve = self.x_evolve_half * self.x_evolve_half
#             #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
#             #               (self.k * self.k) * dt)




def tdm_surf(x,y, N=2):
    """
    transition dipole moment surface
    N: int
        number of electronic states
    """

    # if Condon approximation
    tdm = np.zeros((nx, ny, N, N))
    tdm[:,:, 0, 1] = tdm[:, :, 1, 0] = 1.

    return tdm

# def nac_apes(x, y, N):
#     """
#     nonadiabatic couplings between APESs
#     """
#     nac = np.zeros((N, N))

#     return nac

# def nonadiabatic_coupling(x, y, nstates):
#     """
#     nonadiabatic couplings among PPES
#     nstates: int
#         number of states
#     """
#     #d0 = nac_apes(x, y, nstates)

#     nac_x = np.zeros((nx, ny, nstates, nstates))
#     nac_y = np.zeros((nx, ny, nstates, nstates))

#     nac_x[:,:,0,1] = 0.1 * np.exp(-X**2-Y**2)
#     nac_x[:,:,1,0] = - nac_x[:,:,0,1]

#     return nac_x, nac_y

# def hpsi(psi, kx, ky, vmat, coordinates='linear'):
#     """
#     evaluate H \psi
#     input:
#         v: 1d array, adiabatic surfaces
#         d: nonadiabatic couplings, matrix
#     output:
#         hpsi: H operators on psi
#     """
#     # v |psi>
# #    for i in range(len(x)):
# #        for j in range(len(y)):
# #            v_tmp = np.diagflat(vmat[:][i,j])
# #            array_tmp = np.array([psi[0][i, j], psi[1][i, j]])
# #            vpsi = vmat.dot(array_tmp)
#     # if nstates != len(vmat):
#     #     sys.exit('Error: number of electronic states does not match the length of PPES matrix!')

#     vpsi = np.einsum('ijk, ijk -> ijk', vmat, psi)

#     #vpsi = [vmat[i] * psi[i] for i in range(nstates)]

#     # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
#     # D\grad |psi> = D(x) * F^{-1} F

#     psi_k = np.zeros((nx, ny, nstates), dtype=complex)

#     for i in range(nstates):
#         psi_k[:,:,i] = fft2(psi[:,:,i])


#     # momentum operator operate on the WF
#     kxpsi = [np.einsum('i, ij -> ij', kx, psi_k[i]) for i in range(nstates)]
#     kypsi = [np.einsum('ij, j -> ij', ky, psi_k[i]) for i in range(nstates)]

#     dxpsi = [ifft2(kxpsi[i]) for i in range(nstates)]
#     dypsi = [ifft2(kypsi[i]) for i in range(nstates)]

#     G = np.identity(2) # metric tensor

#     # kinetic energy operator
#     if coordinates == 'linear':
#         T = np.einsum('i, j -> ij', kx**2/2./mx, ky**2/2./my)
#         tpsi = [ifft2( T * psi_k[i]) for i in range(nstates)]

#     elif coordinates == 'curvilinear':

#         for i in range(nx):
#             for j in range(ny):
#                 #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

#                 for k in range(nstates):
#                     tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))



#     # NACs operate on the WF

#     dpsi = [np.zeros(psi[0].shape), ] * nstates

#     for i in range(nx):
#         for j in range(ny):

#             nac_x, nac_y = nonadiabatic_couplings(x[i], y[j])

#             tmp1 = np.array([dxpsi[k][i,j] for k in range(nstates)])
#             tmp2 = np.array([dypsi[k][i,j] for k in range(nstates)])

#             tmp = nac_x.dot(tmp1) + nac_y.dot(tmp2) # array with size nstates

#             dpsi[:][i,j] = tmp[:]



#     hpsi = tpsi + vpsi + dpsi

#     return -1j * hpsi

@jit
def hpsi_full(psi, kx, ky, vmat, nac_x, nac_y, coordinates='linear', \
         mass=None, G=None):
    """
    evaluate H \psi with the full vibronic Hamiltonian
    input:
        v: 1d array, adiabatic surfaces
        d: nonadiabatic couplings, matrix
        G: 4d array N, N, Nx, Ny (N: # of states)
    output:
        hpsi: H operates on psi
    """
    # v |psi>
#    for i in range(len(x)):
#        for j in range(len(y)):
#            v_tmp = np.diagflat(vmat[:][i,j])
#            array_tmp = np.array([psi[0][i, j], psi[1][i, j]])
#            vpsi = vmat.dot(array_tmp)
    # if nstates != len(vmat):
    #     sys.exit('Error: number of electronic states does not match the length of PPES matrix!')

    nx, ny, nstates = psi.shape

    vpsi = np.einsum('ijn, ijn -> ijn', vmat, psi)

    #vpsi = [vmat[i] * psi[i] for i in range(nstates)]

    # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
    # D\grad |psi> = D(x) * F^{-1} F
    psi_k = np.zeros((nx, ny, nstates), dtype=complex)

    tpsi = np.zeros((nx, ny, nstates), dtype=complex)
    nacpsi = np.zeros((nx, ny, nstates), dtype=complex)

    # FFT
    for i in range(nstates):
        tmp = psi[:,:,i]
        psi_k[:,:,i] = fft2(tmp)

    # momentum operator operate on the WF
    kxpsi = np.einsum('i, ijn -> ijn', kx, psi_k)
    kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

    dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
    dypsi = np.zeros((nx, ny, nstates), dtype=complex)

    for i in range(nstates):
        dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
        dypsi[:,:,i] = ifft2(kypsi[:,:,i])

    # kinetic energy operator
    if coordinates == 'linear':

        mx, my = mass

        # T = np.einsum('i, j -> ij', kx**2/2./mx, ky**2/2./my)
        Kx, Ky = np.meshgrid(kx, ky)

        T = Kx**2/2./mx + Ky**2/2./my

        tpsi_k = np.einsum('ij, ijn -> ijn', T, psi_k)

        for i in range(nstates):
            tpsi[:,:,i] = ifft2(tpsi_k[:,:,i])

        nacpsi = np.einsum('ijmn, ijn -> ijm', nac_x, dxpsi) * 1j/mx + \
        np.einsum('ijmn, ijn -> ijm', nac_y, dypsi) * 1j/my # array with size nstates

    elif coordinates == 'curvilinear':


        # for i in range(nx):
        #     for j in range(ny):
        #         #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

        #         for k in range(nstates):
        #             tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))

        for n in range(nstates):
            tpsi[:, :, n] = KEO(psi[:, :, n], kx, ky, G)


    # NACs operate on the WF


    # nac_x, nac_y = nonadiabatic_couplings(X, Y, nstates)


    # for i in range(nx):
    #     for j in range(ny):
            # tmp1 = np.array([dxpsi[k][i,j] for k in range(nstates)])
            # tmp2 = np.array([dypsi[k][i,j] for k in range(nstates)])

    # kinetic energy operator for linear coordinates


    # ... NAC_x * G11 * P_x + NAC_y * G22 * P_y + cross terms
    hpsi = tpsi + vpsi + nacpsi

    return -1j * hpsi

#def hpsi(v, d, psi):
#    """
#    evaluate H \psi
#    input:
#        v: 1d array, adiabatic surfaces
#        d: nonadiabatic couplings, matrix
#        psi: 3D array, nuclear wavefunctions, last index denotes the polaritonic states
#    output:
#        hpsi: H operators on psi
#    """
#    # v |psi>
#    for i in range(len(x)):
#        for j in range(len(y)):
#
#            v, d[i,j] = ppes(x[i], y[j])
#
#            vmat = np.diagflat(v)
#
#            vpsi = vmat.dot(psi[i, j, :])
#
#    # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
#    # D\grad |psi> = D(x) * F^{-1} F
#    for i in range(nstates):
#
#        psi = [fft2(psi[i])
#
#        for j in range(len(kx)):
#            for k in range(len(ky)):
#                kpsi[0][j, k] = psi_k[j, k] * kx
#                kpsi[1][j, k] = psi_k[j, k] * ky
#                psi_k[j, k] *= (kx[j]**2/2/mx + ky[k]**2/2./my)
#
#        tpsi[i] = ifft2(psi_k)
#
#        psi[0][i,j] = tmp[0]
#        psi[1][i,j] = tmp[1]
#
#    return hpsi

# @jit
# def x_evolve(dt, psi):
#     """
#     propagate the state in grid basis half time step forward with H = V
#     :param dt: float
#                 time step
#     :param v_2d: float array
#                 the two electronic states potential operator in grid basis
#     :param psi_grid: list
#                 the two-electronic-states vibrational state in grid basis
#     :return: psi_grid(update): list
#                 the two-electronic-states vibrational state in grid basis
#                 after being half time step forward
#     """

#     for i in range(len(x)):
#         for j in range(len(y)):

#             v = diabatic(x[i], y[j])

#             w, u = scipy.linalg.eigh(v)

#             vmat = np.diagflat(np.exp(- 1j * w / hbar * dt))

#             array_tmp = np.array([psi[0][i, j], psi[1][i, j]])

#             tmp = np.dot(u.conj().T, vmat.dot(u)).dot(array_tmp)

#             psi[0][i,j] = tmp[0]
#             psi[1][i,j] = tmp[1]

#     return psi



# @jit
# def propogate_spo(dt, psi, num_steps=0):
#     """
#     perform the propagation of the dynamics using second-order SPO
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
#     """
#     #f = open('density_matrix.dat', 'w')
#     t = 0.0
#     dt2 = dt * 0.5
#     #purity = np.zeros((num_steps + 1, 1))
#     #purity[0] = density_matrix(psi)[4]

#     kx = 2. * np.pi * fftfreq(nx, dx)
#     ky = 2. * np.pi * fftfreq(ny, dy)

#     if num_steps > 0:
#         psi = x_evolve(dt2, psi)

#     for i in range(num_steps):
#         t += dt
#         psi = k_evolve_2d(dt, kx, ky, psi)
#         psi = x_evolve(dt, psi)
#         #output_tmp = density_matrix(psi)

#         #f.write('{} {} {} {} {} \n'.format(t, *rho))
#         #purity[i] = output_tmp[4]

#     #k_evolve_2d(dt, kx, ky, psi_grid)
#     #x_evolve_half_2d(dt, v_2d, psi_grid)

#     # t += dt
#     #f.close()
#     return psi

class BornHuang2:
    def __init__(self, x, y, mass=[1,1], nstates=2, ndim=2):
        """
        2D non-adiabatic wave packet dynamics in
        adiabatic representation using RK4 integrator

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        mass : TYPE, optional
            DESCRIPTION. The default is [1,1].
        nstates: int
            electronic states

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)

        self.mass = mass

        self.v = None # APES
        self.nac = None

    def set_apes(self, v):

        if isinstance(v, np.ndarray):
            assert(v.shape == (self.nx, self.ny, self.nstates))
            self.v = v
        elif isinstance(v, callable):
            X, Y = np.meshgrid(self.x, self.y)
            self.v = v(X, Y)
        else:
            raise TypeError('v can only be array or function.')


    def run(self, psi0, dt, nt=1, t0=0):
        """

        :param dt: time step

        :param psi0: 3D array, (nx, ny, nstates)
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
        t = t0

        # setup the dipole surface
        # dip_mat = dipole(X, Y)

        # nac = nonadiabatic_couplings(X, Y, nstates)

        v = self.v
        x, y = self.x, self.y
        nx, ny = self.nx, self.ny

        nac_x, nac_y = self.nac

        mass = self.mass

        dx = interval(x)
        dy = interval(y)

        kx = 2.0 * np.pi * scipy.fftpack.fftfreq(nx, dx)
        ky = 2.0 * np.pi * scipy.fftpack.fftfreq(ny, dy)

        print('Propagation starts ...\n')

        psi = psi0

        for i in range(nt):

            t += dt
            psi = rk4(psi, hpsi_full, dt, kx, ky, v, nac_x, nac_y, mass)
            #output_tmp = density_matrix(psi)

            # for obs_op in obs_ops:
            #     tmp = obs(psi, )

            #f.write('{} {} {} {} {} \n'.format(t, *rho))
            #purity[i] = output_tmp

        return psi

    def population(self):
        pass


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

# @jit
# def density_matrix(psi_grid):
#     """
#     compute electronic purity from the wavefunction
#     """
#     rho00 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[0]))*dx*dy
#     rho01 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[1]))*dx*dy
#     rho11 = np.sum(np.multiply(np.conj(psi_grid[1]), psi_grid[1]))*dx*dy

#     purity = rho00**2 + 2*rho01*rho01.conj() + rho11**2

#     return rho00, rho01, rho01.conj(), rho11, purity

def rdm_el(psi):
    '''
    compute the reduced electronic density matrix

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.

    '''
    rho = np.einsum('ijk, ijl -> kl', psi.conj(), psi) * dx * dy
    return rho

# def population(psi):
#     '''
#     Parameters
#     ----------
#     psi : 3d array, complex
#         the nuclear wavefunction

#     Returns
#     -------
#     P0, P1.

#     '''
#     psi0 = psi[:,:,0]
#     P0 = linalg.norm(psi0)**2 * dx*dy

#     psi1 = psi[:,:,1]
#     P1 = linalg.norm(psi1)**2 * dx*dy

#     return P0, P1



def hpsi(psi, kx, ky, v, G):

    kpsi = KEO(psi, kx, ky, G)
    vpsi = PEO(psi, v)

    return -1j * (kpsi + vpsi)




# nout = 10
# f = open('rho_el.dat', 'w')

# for i in range(num_steps//nout):

#     for k in range(nout):
#         t += dt
#         psi = rk4(psi, hpsi, dt, kx, ky, vmat)
#     #output_tmp = density_matrix(psi)

#     # store population and coherence dynamics
#     rho = rdm_el(psi)

#     f.write('{} {} {} {} {} \n'.format(t, *rho.flatten()))
#     #purity[i] = output_tmp

# f.close()



#psi = propagate_rk4(x, y, dt, psi, num_steps)

# store the final wavefunction
# np.save('psi_final', psi)


# plot initial wavefunction
# fig, (ax0, ax1) = plt.subplots(ncols=2)
# extent = [xmin, xmax, ymin, ymax]
# ax0.contour(X, Y, np.abs(psi0[:,:,0]),extent=extent, colors='white')
# ax0.matshow(vmat[:,:,0], extent=extent)
# ax1.matshow(vmat[:,:,1], extent=extent)

# #ax1.contour(X, Y, np.abs(psi0[:,:,1]))

# # plot the final wavefunction
# fig, (ax0, ax1) = plt.subplots(ncols=2)
# extent=[xmin, xmax, ymin, ymax]
# ax0.contour(X, Y, np.abs(psi[:,:,0]))
# #
# ax1.contour(X, Y, np.abs(psi[:,:,1]))

#ax.contour(X, Y, np.abs(psi[:,:,1]))



#
#
## covergence test wrt the grid points
#for i in range(5, 6):
#    # the variables below are global variables for this module
#    nx = 2 ** i
#    ny = 2 ** i
#    xmin = -11
#    xmax = -xmin
#    ymin = -11
#    ymax = -ymin
#    x = np.linspace(xmin, xmax, nx)
#    y = np.linspace(ymin, ymax, ny)
#    dx = x[1] - x[0]
#    dy = y[1] - y[0]
#
#    print('x range = ', x[0], x[-1])
#    print('dx = {}'.format(dx))
#    print('number of grid points along x = {}'.format(nx))
#    print('y range = ', y[0], y[-1])
#    print('dy = {}'.format(dy))
#    print('number of grid points along y = {}'.format(ny))
#
#    xv, yv = np.meshgrid(x, y)
#    xv = xv.T
#    yv = yv.T
#
#    # specify constants
#    hbar = 1.0  # planck's constant
#    m = 1.0  # particle mass
#
#    # test the main
#    sigma_tmp = np.identity(2) * 2.
#    t_axis, purity = main(0.001, sigma_tmp, -3, -3, 0, 0, 1./2.,
#                                             3, 3, 0, 2., 0,
#                                             0.1)
#    plt.plot(t_axis, purity)
#plt.show()

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
# x_evolve_half_2d(0.01, v_list, psigrid)


# test the k_evolve
# kx = fftfreq(nx, dx)
# ky = fftfreq(ny, dy)
# k_evolve_2d(0.01, kx, ky, psigrid)
# x_evolve_half_2d(0.01, v_list, psigrid)
# print(density_matrix(psigrid))

# test dynamics
# print(spo_dynamics(0.01, v_list, psigrid, num_steps=2)[1])


class BornHuang:
    def __init__(self, x, nstates, v=None, nac=None, mass=1):
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

        self.psilist = []

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

    def run(self, psi0, dt=0.001, nt=1,  t0=0., nout=1, coordinates='linear'):
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
        mass = self.mass

        # momentum k-space
        k = 2.0 * np.pi * scipy.fftpack.fftfreq(nx, dx)

        if coordinates == 'linear':
            print('The nuclear coordinate is linear.')

        elif coordinates == 'curvilinear':

            raise NotImplementedError('Kinetic energy operator for curvilinear\
                                      coordinates has not been implemented.')



        for j in range(nt//nout):
            for i in range(nout):

                t += dt
                psi = rk4(psi, hpsi, dt, x, k, vmat, nac, mass)
                #output_tmp = density_matrix(psi)

                #f.write('{} {} {} {} {} \n'.format(t, *rho))
                #purity[i] = output_tmp

            self.psilist.append(psi.copy())

            # ax.plot(x, np.abs(psi[:,0]) + 0.1 * j)


        return self


def density_matrix(psi_x,dx):
    """
    compute purity from the wavefunction
    """
    rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
    rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
    rho11 = 1. - rho00
    return rho00, rho01, rho01.conj(), rho11

def hpsi(psi, x, k, vmat, nac, mass=1, coordinates='linear', use_nac2=False):
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
        tpsi[:,a] = ifft( k*k/2./mass * psi_k[:, a])

    #     G = np.identity(2)

    #     for i in range(nx):
    #         for j in range(ny):
    #             #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

    #             for k in range(nstates):
    #                 tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))



    # NACs operate on the WF

    nacpsi = -np.einsum('imn, in -> im', nac, dpsi)/mass  # array with size nstates

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




######################################################################

if __name__ == '__main__':

    import time

    def apes(x):
        v = np.zeros((nx, nstates))
        v[:, 0] = x**2/2.
        v[:, 1] = (x-1)**2 + 2

        return v

    def get_nac(x):
        NAC = np.zeros((nx, nstates, nstates))
        NAC[:, 0, 1] = np.exp(-x**2/2.)
        NAC[:, 1, 0] = - NAC[:, 0 ,1]

        return NAC

    start_time = time.time()

    nstates = 2 # number of electronic states

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
    psi[:, 0] = gwp(x, a=1.0, x0=0.0, k0=2.0)

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

    sol = BornHuang(x, nstates=nstates, mass=1.0, v=vmat, nac=nac)
    sol.run(psi0=psi, dt=dt, nt=4000, nout=1000)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for psi in sol.psilist:
        ax.plot(x, np.abs(psi[:,1]))

    print('Execution Time = {} s'.format(time.time() - start_time))