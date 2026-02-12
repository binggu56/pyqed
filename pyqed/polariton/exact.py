#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 23:35:06 2023


Open quantum system dynamics in optical cavities;

Basis:
    Here the polaritonic states are used as a basis instead of the direct product states

Created on Feb 10 11:14:55 2018

@author: Bing Gu

History:

"""

# import yaml
import numpy as np
from numpy import sqrt
import numba

from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, identity
from scipy.sparse import kron
import sys
from timeit import default_timer as timer

# import sys
# sys.path.append(r'C:\Users\Bing\Google Drive\lime')
# sys.path.append(r'/Users/bing/Google Drive/lime')

from pyqed.units import au2fs, au2ev, au2k
from pyqed import dag, dagger, commutator, comm, anticomm, \
    rk4, ket2dm, Mol, multi_spin, Cavity, Polariton, tensor_power, pauli, \
        driven_dynamics

from pyqed import obs_dm as obs

class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.gamma = cutoff
        self.reorg = reorg
        self.c_ops = None

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return


#sigmaz = np.array([[1.0,0.0],[0.0,-1.0]], dtype=np.complex128)
#sigmax = np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)
#sigmay = np.array([[0.0,-1j],[1j,0.0]], dtype=np.complex128)


# hbar = 0.65821220 # in units of eV*fs
# hbarinv = 1.0/hbar

wavenumber2hartree = 4.55633525277e-06 #

def spec_den(x):
    return  2. * reorg * x * cutfreq/(x**2 + cutfreq**2)

#def Hamiltonian():
#    g = 0.5 # cavity-molecule coupling strength
#    g /= au2ev
#
#    eps = 3.0 # excitation energy at Frank-Condo point
#    eps /= au2ev
#
#    omega_c = eps
#
#    H = np.zeros((ns, ns))
#    # the states are respectively, |000>, |100>, |010>, |001>, the position are ordered as
#    # cavity, molecule a, molecule b
#
#    H[1,1] = omega_c
#    H[1,2] = H[2,1] =  g
#    H[1,3] =  H[3,1] = g
#    H[2,2] = eps
#    H[3,3] = eps
#
#
#    return H

def corr(t):
    """
    bath correlation function C(t) = <x(t)x>. For the Drude spectral density,
    in the high-temperature limit, C(t) ~ pi * reorg * T * e^{-cutoff * t}
    """
    # numerical treatment
    #NP = 1000
    #maxfreq = 1.0
    #omega = np.linspace(1e-4, maxfreq, NP)
    #dfreq = omega[1] - omega[0]

#    cor = sum(spec_den(omega) * (coth(omega/2./T) * np.cos(omega * t) - \
#                       1j * np.sin(omega * t))) * dfreq

    # test correlation function
#    reorg = 500. # cm^{-1}
#    reorg *= wavenumber2hartree

#    T = 0.03 # eV
#    T /= au2ev

#    td = 10 # fs
#    td /= au2fs

    ### analytical

    # Drude spectral density at high-temperature approximation
    cor = (2.0 * T * reorg - 1j * reorg * cutfreq) * np.exp(- cutfreq * t)

    return cor


class Basis():
    def __init__(self,  n_el, n_cav, n_vc, n_vt):
        self.n_el = n_el
        self.n_cav = n_cav
        self.n_vt = n_vt
        self.n_vc = n_vc

@numba.jit
def ham_sys(n_el, n_cav, n_vc, n_vt):
    """
    contruct vibronic-cavity basis for polaritonic states, that is a
    direct product of electron-vibration-photon space
    """

    n_basis = n_el * n_cav * n_vc * n_vt

    freq_vc = 952. * wavenumber2hartree
    freq_vt = 597. * wavenumber2hartree

    Eshift = np.array([31800.0, 39000]) * wavenumber2hartree
    kappa = np.array([-847.0, 1202.]) * wavenumber2hartree
    coup = 2110.0 * wavenumber2hartree # inter-state coupling lambda

#   construct basis set
#    basis_set = []
#    for i in range(n_el):
#        for j in range(n_cav):
#            for k in range(n_vc):
#                for l in range(n_vt):
#                    basis_set.append(Basis(i,j,k,l))


    #    h_cav = np.zeros((n_cav, n_cav), dtype=np.complex128)
#    h_cav[0,0] = 2.0
#    h_cav[1,1] = 3.0
#    h_vc = np.zeros((n_vc, n_vc))
#    h_vc[0,1] = h_vc[1,0] = 1.2

    # indentity matrices in each subspace
    I_el = np.identity(n_el)
    I_cav = np.identity(n_cav)
    I_vc = np.identity(n_vc)
    I_vt = np.identity(n_vt)

    h_cav = ham_ho(freq_cav, n_cav)
    h_vt = ham_ho(freq_vt, n_vt)
    h_vc = ham_ho(freq_vc, n_vc)

    h_el = np.diagflat(Eshift)

    # the bare term in the system Hamiltonian
    h0 = kron(h_el, kron(I_cav, kron(I_vc, I_vt))) + \
        kron(I_el, kron(h_cav, kron(I_vc, I_vt))) \
        + kron(I_el, kron(I_cav, kron(h_vc, I_vt))) +\
        kron(I_el, kron(I_cav, kron(I_vc, h_vt))) \

    X = pos(n_vt)

    h1 = kron(np.diagflat(kappa), kron(I_cav, np.kron(I_vc, X)), format='csr')


    Xc = pos(n_vc)

    ex = np.zeros((n_el, n_el)) # electronic excitation operator
    deex = np.zeros((n_el, n_el)) # deexcitation
    deex[0, 1] = 1.0
    ex[1, 0] = 1.0
    trans_el = ex + deex

    #h_fake = kron(np.diagflat(kappa), kron(I_cav, kron(Xc, I_vt)))

    ###
#    h_m = np.zeros((n_basis, n_basis))
#
#    for m, b0 in enumerate(basis_set):
#        for n, b1 in enumerate(basis_set):
##            h_m[m,n] = h_el[b0.n_el, b1.n_el] * h_cav[b0.n_cav, b1.n_cav] * h_vc[b0.n_vc, b1.n_vc]
#            h_m[m,n] = trans_el[b0.n_el, b1.n_el] * I_cav[b0.n_cav, b1.n_cav] \
#                    * Xc[b0.n_vc, b1.n_vc] * I_vt[b0.n_vt, b1.n_vt]

    h2 = coup * kron(trans_el, kron(I_cav, kron(Xc, I_vt)), format='csr')

    # if n_cav = n _el
    deex_cav = deex
    ex_cav = ex

    h3 = g_cav * (kron(ex, kron(deex_cav, kron(I_vc, I_vt))) + \
                  kron(deex, kron(ex_cav, kron(I_vc, I_vt))))

    h_s = h0 + h1 + h2 + h3

    # polaritonic states can be obtained by diagonalizing H_S
    # v is the basis transformation matrix, v[i,j] = <old basis i| polaritonic state j>
    #eigvals, v = np.linalg.eigh(h_s)

    #h_s = csr_matrix(h_s)

    # collapse operators in dissipative dynamics
    Sc = kron(I_el, kron(I_cav, kron(Xc, I_vt)), format='csr')
    St = kron(I_el, kron(I_cav, kron(I_vc, X)), format='csr')

    return h_s, Sc, St


def single_polariton_H(onsite, nsites, omegac, g):
    '''
    single-polariton Hamiltonian for N two-level molecules coupled to
    a single cavity mode including the ground state

    input:
        g_cav: single cavity-molecule coupling strength
    '''

    nstates = nsites + 1 + 1 # number of states in the system

    dip = np.zeros((nstates, nstates))
    # the states are respectively, |000>, |100>, |010>, |001>,
    # the position are ordered as

    H = np.diagflat([0.] + onsite + [omegac])
    H[1:nsites+1, -1] = H[-1, 1:nsites+1] = g

    dip[0,1:nsites+1] = dip[1:nsites+1, 0] =  1

    return csr_matrix(H), csr_matrix(dip)

# @numba.jit
# def change_basis(A, v):
#     """
#     transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
#     input:
#         A: matrix of operator A in old basis
#         v: basis transformation matrix
#     output:
#         Anew: matrix A in the new basis
#     """
#     Anew = dagger(v).dot(A.dot(v))
#     Anew = csr_matrix(A)

#     return Anew


def nondissipative(n_el, n_cav, n_vc, n_vt, Nt, dt):
    """
    time propagation of the nonadiabatic dynamics without dissipation
    """
    t = 0.0

    ns = n_el * n_vt * n_vc * n_cav  # number of states in the system

    # initialize the density matrix
    rho_el = np.zeros((n_el, n_el), dtype=np.complex128)
    rho_el[1, 1] = 1.0

    rho_cav = np.zeros((n_cav, n_cav), dtype=np.complex128)
    rho_cav[0, 0] = 1.0
    rho_vc = np.zeros((n_vc, n_vc), dtype=np.complex128)
    rho_vc[0, 0] = 1.0
    rho_vt = np.zeros((n_vt, n_vt), dtype=np.complex128)
    rho_vt[0, 0] = 1.0

    rho0 = kron(rho_el, kron(rho_cav, kron(rho_vc, rho_vt)))

    rho = rho0

    #f = open(fname,'w')
    fmt = '{} '* 5 + '\n'

    # short time approximation
    # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + \
    #1./cutfreq * sigmaz)

    h0, S1, S2 = ham_sys(n_el, n_cav, n_vc, n_vt)

    # constuct the operators needed in Redfield equation
    #Lambda1 = getLambda(ns, h0, S1, T, cutfreq, reorg)
    #Lambda2 = getLambda(ns, h0, S2, T, cutfreq, reorg)

    f_dm = open('den_mat.dat', 'w')
    f_pos = open('position.dat', 'w')

    t = 0.0
    dt2 = dt/2.0


    Ag, Ae, A_coh = pop(n_el, n_cav, n_vc, n_vt)

    # first-step
    rho_half = rho - 1j * comm(h0, rho) * dt2
    rho1 = rho - 1j * comm(h0, rho) * dt

    rho_old = rho
    rho = rho1

    print('Time propagation start ... \n')

    for k in range(Nt):

        t += dt

        rho_new = rho_old - 1j * comm(h0, rho) * 2. * dt

        # update rho_old
        rho_old = rho
        rho = rho_new

        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        #f.write('{} {} \n'.format(t, cor))

        # take a partial trace to obtain the rho_el
        P1 = obs(Ag, rho)
        P2 = obs(Ae, rho)
        coh = obs(A_coh, rho)

        # position expectation
        qc = obs(S1, rho)
        qt = obs(S2, rho)

        f_dm.write('{} {} {} {} \n'.format(t * au2fs, P1, P2, coh))
        f_pos.write('{} {} {} \n'.format(t*au2fs, qc, qt))


    f_pos.close()
    f_dm.close()

    return rho


#def ave_qt(S2, v, rho):
#    S_tmp = dagger(v).dot(S2.dot(v))
#    return np.trace(S_tmp.dot(rho))


def ham_ho(freq, n):
    """
    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
    output:
        h: hamiltonian of the harmonic oscilator
    """
    energy = np.zeros(n)
    for i in range(n):
        energy[i] = (i + 0.5) * freq

    return np.diagflat(energy)

#def leapfrog(x0, Nt, dt, func, h0, S1, S2, Lambda1, Lambda2):
#    """
#    leapfrog integration for first-order ODE
#    dx/dt = f(x)
#    input:
#        x0: initial value
#        dt: timestep
#        Nt: time steps
#        func: right-hand side of ODE, f(x)
#    """
#    dt2 = dt/2.0
#
#    # first-step
#    x_half = x0 + func(x0, h0, S1, S2, Lambda1, Lambda2) * dt2
#    x1 = x0 + func(x_half)*dt
#
#
#
#    xold = x0
#    x = x1
#    for k in range(Nt):
#        xnew = xold + func(x, h0, S1, S2, Lambda1, Lambda2) * 2. * dt
#
#        # update xold
#        xold = x
#        x = xnew
#
#        obs(x)
#
#    return x

@numba.jit
def func(rho, h0, S1, S2, Lambda1, Lambda2):
    """
    right-hand side of the master equation
    """
    rhs = -1j * commutator(h0, rho) - ( commutator(S1, \
                Lambda1.dot(rho) - rho.dot(dagger(Lambda1))) + \
                commutator(S2, Lambda2.dot(rho) - rho.dot(dagger(Lambda2)))
                )
    return csr_matrix(rhs)

# def dipole_cor(Nt, dt):
#     """
#     compute the dipole auto-correlation function using quantum regression theorem

#     """
#     ns = n_el * n_vt * n_vc * n_cav  # number of states in the system

#     # initialize the density matrix
#     rho0 = np.zeros((ns, ns), dtype=np.complex128)
#     rho0[0,0] = 1.0

#     d = np.zeros((ns, ns), dtype = np.complex128)
#     d[0, 2] = d[2,0] = 1.0
#     d[3, 0] = d[3,0] = 1.0

#     rho0 = np.matmul(d, rho0)


#     # dissipative operators
#     S1 = np.zeros((ns,ns), dtype=np.complex128)
#     S2 = np.zeros((ns,ns), dtype = np.complex128)

#     S1[2,2] = 1.0
#     S1[0,0] = -1.0
#     S2[3,3] = 1.0
#     S2[0,0] = -1.0

#     Lambda1 = getLambda(S1)
#     Lambda2 = getLambda(S2)

#     # short time approximation
#     # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)

#     h0 = Hamiltonian()

#     f = open('cor.dat', 'w')
#     f_dm = open('den_mat.dat', 'w')

#     t = 0.0
#     dt2 = dt/2.0


#     # first-step
#     rho_half = rho0 + func(rho0, h0, S1, S2, Lambda1, Lambda2) * dt2
#     rho1 = rho0 + func(rho_half, h0, S1, S2, Lambda1, Lambda2) * dt

#     rho_old = rho0
#     rho = rho1


#     for k in range(Nt):

#         t += dt

#         rho_new = rho_old + func(rho, h0, S1, S2, Lambda1, Lambda2) * 2. * dt

#         # update rho_old
#         rho_old = rho
#         rho = rho_new

#         cor = np.trace(np.matmul(d, rho))
#         # store the reduced density matrix
#         f.write('{} {} \n'.format(t, cor))
#         f_dm.write('{} {} \n'.format(t, rho[2,0]))


#     f.close()
#     f_dm.close()

#     return


@numba.jit
def getLambda(ns, h0, S, T, cutfreq, reorg):

    tmax = 10000.0
    print('correlation function at {} fs = {} \n'.format(tmax * au2fs, np.exp(-tmax*cutfreq)))
    time = np.linspace(0, tmax, 1000)
    dt = time[1] - time[0]

    print('Timestep for computing correlation function = {}'.format(dt * au2fs))

    # back propagation

    Lambda = csr_matrix((ns, ns), dtype=np.complex128)

    #t = time[0]
    #phi = hop * t - Delta * np.sin(omegad * t)/omegad
    #Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2

    #h0 = Hamiltonian()

    for k in range(len(time)):

        t = time[k]

        S = rk4_step_op(S, liouville, -dt, h0)

        Lambda += dt * S * corr(t, T, cutfreq, reorg)


#    t = time[len(time)-1]
#    phi = hop * t + Delta * np.sin(omegad * t)/omegad
#    Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2
#    Lambda = cy * sigmay + cz * sigmaz

    return Lambda

def liouville(a, h):
    return 1j * comm(h, a)

def rk4_step_op(a, fun, dt, *args):

    dt2 = dt/2.0

    k1 = fun(a, *args)
    k2 = fun(a + k1*dt2, *args)
    k3 = fun(a + k2*dt2, *args)
    k4 = fun(a + k3*dt, *args)

    a += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

    return a

def Redfield(n_el, n_cav, n_vc, n_vt, Nt, dt, read_lambda):
    """
    time propagation of the Redfield equation
    """
    t = 0.0

    ns = n_el * n_vt * n_vc * n_cav  # number of states in the system

    # initialize the density matrix
    rho_el = np.zeros((n_el, n_el), dtype=np.complex128)
    rho_el[1, 1] = 1.0
    rho_el[0, 0] = 0.


    rho_cav = np.zeros((n_cav, n_cav), dtype=np.complex128)
    rho_cav[0, 0] = 1.0
    rho_vc = np.zeros((n_vc, n_vc), dtype=np.complex128)
    rho_vc[0, 0] = 1.0
    rho_vt = np.zeros((n_vt, n_vt), dtype=np.complex128)
    rho_vt[0, 0] = 1.0

    rho0 = kron(rho_el, np.kron(rho_cav, np.kron(rho_vc, rho_vt)))

    rho = rho0

    #f = open(fname,'w')
    #fmt = '{} '* 5 + '\n'

    # construct system-bath operators in H_SB

    # short time approximation
    # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)

    # constuct system Hamiltonian and system collapse operators
    h0, S1, S2 = ham_sys(n_el, n_cav, n_vc, n_vt)

#    file_lambda = 'lambda.npz'
#
#    if read_lambda == True:
#
#        data = np.load(file_lambda)
#        Lambda1 = data['arr_0']
#        Lambda2 = data['arr_1']
#
#    else:
#        # constuct the operators needed in Redfield equation
#        Lambda1 = getLambda(ns, h0, S1, T, cutfreq, reorg)
#        Lambda2 = getLambda(ns, h0, S2, T, cutfreq, reorg)
#
#        np.savez(file_lambda, Lambda1, Lambda2)


    #print(Lambda1, '\n', Lambda2)

    f_dm = open('den_mat.dat', 'w')
    f_pos = open('position.dat', 'w')
    f_ph = open('photon.dat', 'w')

    t = 0.0
    dt2 = dt/2.0

    # observables
    Ag, Ae, A_coh, A_ph = pop(n_el, n_cav, n_vc, n_vt)


    print('Time propagation start ... \n')

    Lambda1 = csr_matrix((ns, ns), dtype=np.complex128)
    Lambda2 = csr_matrix((ns, ns), dtype=np.complex128)
    s1_int = S1
    s2_int = S2

    for k in range(Nt):


        #k1 = func(rho, h0, S1, S2, Lambda1, Lambda2)
        #k2 = func(rho + k1*dt2, h0, S1, S2, Lambda1, Lambda2)
        #k3 = func(rho + k2*dt2, h0, S1, S2, Lambda1, Lambda2)
        #k4 = func(rho + k3*dt, h0, S1, S2, Lambda1, Lambda2)
        #rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

        rho = rk4(rho, func, dt, h0, S1, S2, Lambda1, Lambda2)

        Lambda1 += s1_int * corr(t) * dt2
        Lambda2 += s2_int * corr(t) * dt2

        s1_int = rk4_step(s1_int, liouville, -dt, h0)
        s2_int = rk4_step(s2_int, liouville, -dt, h0)

        Lambda1 += s1_int * corr(t) * dt2
        Lambda2 += s2_int * corr(t) * dt2


        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        #f.write('{} {} \n'.format(t, cor))

        # take a partial trace to obtain the rho_el
        P1 = obs(Ag, rho)
        P2 = obs(Ae, rho)
        coh = obs(A_coh, rho)
        n_ph = obs(A_ph, rho)

        # position expectation
        qc = obs(S1, rho)
        qt = obs(S2, rho)

        t += dt


        f_dm.write('{} {} {} {} \n'.format(t * au2fs, P1, P2, coh))
        f_pos.write('{} {} {} \n'.format(t*au2fs, qc, qt))
        f_ph.write('{} {} \n'.format(t * au2fs, n_ph))

    f_pos.close()
    f_dm.close()
    f_ph.close()

    return rho



def theta(env, i, rho):
    '''
    input:
        i: index for collpse opeartor
        rho: density matrix
    '''
    reorg = env.reorg[i]
    c_op = env.c_ops[i]
    temperature = env.temperature
    gamma = env.gamma[i]

    return 1j * (2*reorg * temperature * comm(c_op, rho) - 1j * reorg * gamma * anticomm(c_op, rho))


def heom(env, hs, rho, obs_ops, Nt, dt):
    """
    HEOM time propagation of open quantum systems
    input:
        env: object of class Env, contain information of the environment
        hs: system hamiltonian
        rho: initial density matrix
        obs:
            list of operators corresonding to disired observables
    """

    ns = hs.shape[0]

    #f = open(fname,'w')
    #fmt = '{} '* 5 + '\n'

    f_dm = open('den_mat.dat', 'w')
    f_pos = open('position.dat', 'w')
    f_ph = open('photon.dat', 'w')

    t = 0.0
    #dt2 = dt/2.0

    nmax = 5
    print('Time propagation start ... \n')

    # build a list of ADOs
    ado = []
    for lc in range(nmax):
        tmp = []
        for lt in range(nmax):
            tmp.append(csr_matrix((ns, ns), dtype=complex))
        ado.append(tmp)
    #Lambda2 = csr_matrix((ns, ns), dtype=np.complex128)

    #s1_int = S1
    #s2_int = S2

    Sc, St = env.c_ops
    gammac, gammat = env.gamma[0], env.gamma[1]


    ado[0][0] = rho

    Ac, Ad, Aa = obs_ops

    for k in range(Nt):


        #k1 = func(rho, h0, S1, S2, Lambda1, Lambda2)
        #k2 = func(rho + k1*dt2, h0, S1, S2, Lambda1, Lambda2)
        #k3 = func(rho + k2*dt2, h0, S1, S2, Lambda1, Lambda2)
        #k4 = func(rho + k3*dt, h0, S1, S2, Lambda1, Lambda2)
        #rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

        ado[0][0] += (-1j * comm(hs, ado[0][0]) - 1j*comm(Sc, ado[1][0]) -\
                    1j*comm(St, ado[0][1]) ) * dt

        ado[0][1] += (-1j * comm(hs, ado[0][1]) - gammat * ado[0][1]  \
                    -1j*comm(Sc, ado[1][1]) - 1j*comm(St, ado[0][2])-\
                    gammat * theta(env, 1, ado[0][0]) ) * dt

        ado[1][0] += ( -1j * comm(hs, ado[1][0]) - gammac * ado[1][0] - \
                    1j*comm(Sc, ado[2][0]) - 1j*comm(St, ado[1][1])-\
                    gammac * theta(env, 0, ado[0][0]) ) * dt


        for lc in range(1, nmax):
            for lt in range(1, nmax-lc):
                ado[lc][lt] += ( -1j * comm(hs, ado[lc][lt]) -\
                    (lc*gammac + lt*gammat) * ado[lc][lt]  \
                    -1j*comm(Sc, ado[lc+1][lt]) - 1j*comm(St, ado[lc][lt+1])-\
                    lc  * theta(env, 0, ado[lc-1][lt]) - \
                    lt  * theta(env, 1, ado[lc][lt-1])
                    ) * dt


        #rho = rk4_step(rho, func, dt, hs, S1, S2, Lambda1, Lambda2)

#        Lambda1 += s1_int * corr(t) * dt2
#        Lambda2 += s2_int * corr(t) * dt2
#
#        s1_int = rk4_step(s1_int, liouville, -dt, h0)
#        s2_int = rk4_step(s2_int, liouville, -dt, h0)
#
#        Lambda1 += s1_int * corr(t) * dt2
#        Lambda2 += s2_int * corr(t) * dt2


        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        #f.write('{} {} \n'.format(t, cor))

        # take a partial trace to obtain the rho_el
        rho = ado[0][0]
        #Ag, Ae, A_coh, A_ph = obs_ops
        Pc = obs(Ac, rho)
        Pd = obs(Ad, rho)
        Pa = obs(Aa, rho)
        #P2 = obs(Ae, rho)
        #coh = obs(A_coh, rho)
        #n_ph = obs(A_ph, rho)

        # position expectation
        #Pd = obs(, rho)
        #Pa = obs(St, rho)

        t += dt


        f_dm.write('{} {} {} {} \n'.format(t * au2fs, Pc, Pd, Pa))
        #f_pos.write('{} {} {} \n'.format(t*au2fs, Pd, Pa))
        #f_ph.write('{} {} \n'.format(t * au2fs, n_ph))

    f_pos.close()
    f_dm.close()
    f_ph.close()

    return rho

def pop(n_el, n_cav, n_vc, n_vt):
    """
    take a partial trace to obtain the reduced electronic density matrix
    """
    pop_g = np.zeros((n_el, n_el))
    pop_e = np.zeros((n_el, n_el))

    coh = np.zeros((n_el, n_el), dtype=complex)

    pop_g[0, 0] = 1.0
    pop_e[1, 1] = 1.0

    coh[0, 1] = 1.0
    coh[1, 0] = 1.0

    pop_g = csr_matrix(pop_g)
    pop_e = csr_matrix(pop_e)
    coh = csr_matrix(coh)

    I_cav = identity(n_cav)
    I_vc = identity(n_vc)
    I_vt = identity(n_vt)
    I_el = identity(n_el)

    # operator corresponding to observable
    Ag = kron(pop_g, kron(I_cav, kron(I_vc, I_vt)))
    #Ag = csr_matrix(Ag)

    Ae = kron(pop_e, kron(I_cav, kron(I_vc, I_vt)))
    #Ae = csr_matrix(Ae)

    A_coh = kron(coh, kron(I_cav, kron(I_vc, I_vt)), format='csr')
    #A_coh = csr_matrix(A_coh)

    n_ph = csr_matrix([[0.0, 0.0], [0.0, 1.0]])
    A_ph = kron(I_el, kron(n_ph, kron(I_vc, I_vt)))

    return Ag, Ae, A_coh, A_ph


def pulse(t, tau=2/au2fs):
    return 0.001 * np.exp(-(t-8/au2fs)**2/2/tau**2) * np.cos(omegac * t)


if __name__ == '__main__':

    # with open("input.yaml", 'rb') as stream:
    #     data = yaml.load(stream, Loader=yaml.Loader)

    # #data = data_loaded
    # dt = data['dt'] / au2fs
    # Nt = data['Nt']

    # # nmol = data['nmol']
    # nmol = 10
    # n_cav = data['n_cav']
    # n_vc = data['n_vc']
    # n_vt = data['n_vt']

    # # everything should be transformed in atomic untis
    # cutfreq = data['cutfreq'] * wavenumber2hartree  # cutoff frequency
    # reorg = data['reorg'] * wavenumber2hartree  # reorganization energy
    # T = data['T'] /au2k  # temperature
    # #hop = data['hop'] * wavenumber2hartree  # hopping parameter (positive)
    # #Delta = data['Delta'] * wavenumber2hartree  # drving amplitude
    # #omegad = data['omegad'] * wavenumber2hartree  # driving frequency
    # #theta = data['theta'] * np.pi # relative phase for initial state
    # # g_cav = data['g_cav'] * wavenumber2hartree # system-cavity coupling strength
    # # omegac = data['freq_cav'] /au2ev # cavity frequency
    g_cav = 0.1/au2ev/sqrt(2)
    omegac = 1/au2ev

    cav = Cavity(freq=omegac, ncav=4)
    nmol = 8


    # print('Number of molecules = {} \n'.format(nmol))

    # print('timestep = {} fs'.format(dt * au2fs))
    # print('total time steps = {} \n'.format(Nt))

    # print('energy units is in cm^-1 \n')
    # print('cutoff freq of environment = {} \n'.format(cutfreq/wavenumber2hartree))
    # print('temperature = {} K \n'.format(T * au2k))
    # print('reorg = {} cm^-1 \n'.format(reorg / wavenumber2hartree))
    # print('g_cav = {} cm-1 \n'.format(g_cav/wavenumber2hartree))
    # #print('driving amplitude = {} \n'.format(Delta/wavenumber2hartree))
    # print('Is high-temperature approximation valid? \n')
    # print('beta * cutfreq = {}, supposed to be < 1\n'.format(cutfreq/T))
    # #print('initial relative phase = {} \n'.format(theta))

    # initial density matrix
    #cL = 1./np.sqrt(2.0)
    #cR = np.sqrt(1. - cL**2) * np.exp(1.0j*theta)
    #rho0 = np.array([[np.abs(cL)**2, cL*np.conj(cR)], [np.conj(cL)*cR, np.abs(cR)**2]], \
    #                  dtype=np.complex128)
    #print('initial density matrix\n', rho0)


    start = timer()

    ###
    # nstates = nmol + 1 + 1  # number of states in the single-excitation manifold

    onsite = [1. / au2ev, ] * nmol

    print('Cavity resonance = {} eV '.format(omegac * au2ev))
    print('Cooperative coupling strength = {} eV\n'.format(g_cav * au2ev))

    g_cav = g_cav/sqrt(nmol)


    # Preparing open quantum dynamics simulations
    # Required quantities 1) H, 2) c_ops 3) obs_ops; (4) rho0

    print('Constucting the system Hamiltonian and dipole operator\n')
   # hs, dip = single_polariton_H(onsite, nmol, omegac, g_cav)

    hmol, lowering = multi_spin(onsite, nmol)


    raising = dag(lowering)

    edip = lowering + raising

    mol = Mol(hmol, edip=edip, lowering=lowering)

    pol = Polariton(mol, cav)
    H = pol.getH(g_cav, RWA=False)
    print(H.shape)


    evals, evecs = pol.eigenstates()
    print('Polariton energies = \n', evals[:nmol+2] * au2ev)

    # lower and upper polaritons
    LP = evecs[:,1]
    UP = evecs[:,-1]
    ground = evecs[:,0]

    # initialize the density matrix, we assume the system is excited to the LP state
    # rho0 = ket2dm(ground)
    # rho0 = csr_matrix(rho0)

    # print('Constructing collapse operators ... \n')
    # cavity_leak = False
    # if cavity_leak:
    #     kappa = 0.1 / au2ev
    #     a = np.zeros((nstates, nstates))
    #     a[0, -1] = 1.
    #     c_ops = [np.sqrt(kappa) *  csr_matrix(a)]
    # else:
    #     c_ops = []

    # # pure dephasing
    # gamma = 0.5/ au2ev
    # print('Single-molecule dephasing time = {} fs.\n'.format(1./gamma * au2fs))

    # for n in range(nmol):
    #     c_op = np.zeros((nstates, nstates))
    #     c_op[n+1, n+1] = 1.0
    #     c_op = sqrt(gamma) * csr_matrix(c_op)
    #     c_ops.append(c_op)

    print('Constructing observable operators ... \n')

    # obs_ops = [csr_matrix(ket2dm(ground)), rho0, csr_matrix(ket2dm(UP))]
    # obs_ops = [csr_matrix(ket2dm(ground))]
    s0, sx, sy, sz = pauli()

    obs_ops  = [kron(kron(0.5 * (s0 - sz), tensor_power(s0, nmol-1)), cav.idm)]

    # mol.lindblad(rho0=rho0, dt=0.02/au2fs, Nt=400, c_ops=c_ops, obs_ops=obs_ops)
    nt = 1500
    dt=0.04/au2fs

    H = [H, [pol.get_edip(), pulse]]

    result = driven_dynamics(H, psi0=ground, dt=dt,\
                             Nt=nt, e_ops=obs_ops)

    result.dump('exact.dat')
    
    #Redfield(n_el, n_cav, n_vc, n_vt, Nt, dt, read_lambda=False)
    #heom(env, hs, rho, obs_ops, Nt, dt)

    # import proplot as plt

    # ts = np.arange(nt) * dt * au2fs
    # # fig, ax = plt.subplots()
    # # ax.plot(ts, pulse(ts))

    # fig, ax = plt.subplots()
    # ax.plot(ts, result.observables[:,0].real)

    # #nondissipative(n_el, n_cav, n_vc, n_vt, Nt, dt)
    # end = timer()
    # print('Finished. Time spend = {} s '.format(end - start))
