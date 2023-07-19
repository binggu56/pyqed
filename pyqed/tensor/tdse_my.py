# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:09:06 2019

@author: Bing
"""

""" Comparison of the entanglement growth S(t) following a global quench using
the first order MPO based time evolution for the XX model with the TEBD algorithm.

The TEBD algorithm is also used to recompress the MPS after the MPO time evolution.
This is simpler to code but less efficient than a variational optimization.
See arXiv:1407.1832 for details and how to extend to higher orders.

Frank Pollmann, frankp@pks.mpg.de """

import numpy as np
import pylab as pl
from scipy.linalg import expm

from scipy.fftpack import fft, ifft, fftfreq



s0 = np.eye(2)
sp = np.array([[0.,1.],[0.,0.]])
sm = np.array([[0.,0.],[1.,0.]])

def initial_state(d, chi_max, L, dtype=complex):
    """
    Create an initial product state.
    input:
        L: number of sites
        chi_max: maximum bond dimension
        d: local dimension for each site

    return
    =======
    MPS in right canonical form B0-s0-B1-s1-....B_L
    """
    B_list = []
    s_list = []
    for i in range(L):
        B = np.zeros((d,1,1),dtype=dtype)
        B[np.mod(i,2),0,0] = 1.
        s = np.zeros(1)
        s[0] = 1.
        B_list.append(B)
        s_list.append(s)
    s_list.append(s)
    return B_list,s_list

def apply_mpo_svd(B_list,s_list,w_list,chi_max):
    " Apply the MPO to an MPS."
    d = B_list[0].shape[0] # size of local space
    D = w_list[0].shape[0]
    L = len(B_list) # nsites

    chi1 = B_list[0].shape[1]
    chi2 = B_list[0].shape[2] # left and right bond dims

    B = np.tensordot(B_list[0],w_list[0][0,:,:,:],axes=(0,1))
    B = np.reshape(np.transpose(B,(3,0,1,2)),(d,chi1,chi2*D))
    B_list[0] = B

    for i_site in range(1,L-1):
        chi1 = B_list[i_site].shape[1]
        chi2 = B_list[i_site].shape[2]
        B = np.tensordot(B_list[i_site],w_list[i_site][:,:,:,:],axes=(0,2))
        B = np.reshape(np.transpose(B,(4,0,2,1,3)),(d,chi1*D,chi2*D))
        B_list[i_site] = B
        s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)

    chi1 = B_list[L-1].shape[1]
    chi2 = B_list[L-1].shape[2]

    B = np.tensordot(B_list[L-1],w_list[L-1][:,0,:,:],axes=(0,1))
    B = np.reshape(np.transpose(B,(3,0,2,1)),(d,D*chi1,chi2))
    s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B

    tebd(B_list,s_list,(L-1)*[np.reshape(np.eye(d**2),[d,d,d,d])],chi_max)
    return


def make_U_xx_bond(L,delta):
    """
    Create the bond evolution operator used by the TEBD algorithm.
    ouput:
        u_list: single-step evolution operator

    """

    d = 2
    # Hamiltonian
    H = np.real(np.kron(sp,sm) + np.kron(sm,sp))

    u_list = (L-1)*[np.reshape(expm(-delta*H),(d,d,d,d))]
    return u_list, d

def make_U_xx_mpo(L,dt,dtype=float):
    " Create the MPO of the time evolution operator.  "

    w = np.zeros((3,3,2,2),dtype=type(dt))
    w[0,:] = [s0,sp,sm]
    w[1:,0] = [-dt*sm,-dt*sp]
    w_list = [w]*L
    return w_list

def tebd(B_list,s_list,U_list,chi_max):
    """
    Use TEBD to optmize the MPS and to rduce it back to the orginal size.
    """
    d = B_list[0].shape[0]
    L = len(B_list)

    for p in [0,1]:
        for i_bond in np.arange(p,L-1,2):
            i1=i_bond
            i2=i_bond+1

            chi1 = B_list[i1].shape[1]
            chi3 = B_list[i2].shape[2]

            # Construct theta matrix #
            C = np.tensordot(B_list[i1],B_list[i2],axes=(2,1))
            #C = np.einsum('aij, bjk -> aibk', B_list[i1], B_list[i2])
            C = np.tensordot(C,U_list[i_bond],axes=([0,2],[2,3]))
            print(np.shape(C))

            # ? Why not directly SVD the C tensor?

            theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))

            C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))
            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta)
            Z=Z.T

            W = np.dot(C,Z.conj())
            chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

            # Obtain the new values for B and l #
            invsq = np.sqrt(sum(Y[:chi2]**2))
            s_list[i2] = Y[:chi2]/invsq
            B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq
            B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))


def k_evolve_1d(k, psi):
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
    psi_k = fft(psi)
    psi_k = psi_k * np.exp(-0.5 * 1j * k**2 * dt)
    psi = ifft(psi_k)

    return psi

def kinetic(k, B_list):
    """
    kinetic energy (KE) component of the one-step evolution operator e^{-i * dt * K) on the MPS
    where K is the total KE operator
    """
    for i in range(L):
        chi1, chi2 = np.shape(B_list[i])[1:]
        for a in range(chi1):
            for b in range(chi2):
                B_list[i][:,a,b] = k_evolve_1d(k, B_list[i][:,a,b])

    return B_list

def make_V_list(X, Y):
    """
    return:
        V: 2d array e^{-1j * V * dt}
    """
    V = apes(X, Y)

    return V

def apes(x,y):
    """
    adiabatic PES
    """
    return x**2/2. + y**2/2. + x*y

def potential(B_list, s_list, V, chi_max):
    """
    potential energy component of the one-step evolution operator e^{-i * dt * V) on the MPS
    input:
        V: n-d array, PES
        L: int
            number of sites
    """
    U = np.exp(-1j * dt * V)

    for i in range(L-1):

            i1 = i; i2 = i+1

            chi1 = B_list[i1].shape[1]
            chi3 = B_list[i2].shape[2]

            C = np.tensordot(B_list[i1],B_list[i2],axes=(2,1))
            C = np.einsum('iajb, ij ->iajb', C, U)
            #C = np.tensordot(C, U, axes=([0,2],[2,3]))

            # ? Why not directly SVD the C tensor?

            theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))

            C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))
            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta)
            Z=Z.T

            W = np.dot(C,Z.conj())
            chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

            # Obtain the new values for B and l #
            invsq = np.sqrt(sum(Y[:chi2]**2))
            s_list[i2] = Y[:chi2]/invsq
            B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq
            B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))

    return B_list, s_list

def qd(B_list,s_list,U_list,chi_max):
    """
    Use TEBD to optmize the MPS and to project it back.
    """
    d = B_list[0].shape[0]
    L = len(B_list)

    make_V_list(X, Y)

    B_list = kinetic(B_list)
    B_list, s_list = potential(B_list, s_list, V, chi_max)

def expectation():
    """
    how to compute the observables?
    """

    return


if __name__ == "__main__":
    # Define Pararemeter here
    delta = dt = 0.02
    L = 2
    chi_max = 10
    N_steps = 10

    # grid
    d = 2**4 # local size of Hilbert space
    x = np.linspace(-2,2,d)
    y = np.linspace(-2,2,d)
    X, Y = np.meshgrid(x,y)

    V = make_V_list(X,Y)
    # frequency space
    k = 2. * np.pi * fftfreq(d)

    # TEBD algorithm
    B_list,s_list = initial_state(d, chi_max, L, dtype=complex)

    print(len(B_list), len(s_list))

    S = [0]
    for step in range(N_steps):

        B_list = kinetic(k, B_list)
        B_list, s_list = potential(B_list, s_list, V, chi_max)

        s2 = np.array(s_list[L//2])**2
        S.append(-np.sum(s2*np.log(s2)))

    pl.plot(delta*np.arange(N_steps+1),S)
    pl.xlabel('$t$')
    pl.ylabel('$S$')
    pl.legend(['MPO','TEBD'],loc='upper left')
    pl.show()