#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 01:16:33 2020

@author: bing

From Qutip
"""

#
# quantum state number helper functions
#

import numpy as np
from pyqed import au2k, au2wavenumber, Mol, pauli
from pyqed.superoperator import left, right

# from numba import jit, njit


from scipy.linalg import eigh, eig
from scipy.sparse import csr_matrix, issparse, identity
import scipy.sparse.linalg as la
from scipy.integrate import solve_ivp
# from scipy.sparse.linalg import eigs
import opt_einsum as oe



from pyqed import commutator, anticommutator, comm, anticomm, dag, ket2dm, \
    obs_dm, destroy, rk4, basis, transform, isherm, expm, coth

# from lime.superoperator import lindblad_dissipator
from pyqed.superoperator import op2sop, dm2vec, obs, left, right, operator_to_superoperator
from pyqed.liouville import sort
from pyqed import Mol, Result


def state_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0  0 ]
        [ 0  1 ]
        [ 1  0 ]
        [ 1  1 ]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        Current state in the iteration. Used internally.

    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.

    idx : integer
        Current index in the iteration. Used internally.

    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if state is None:
        state = np.zeros(len(dims), dtype=int)

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


#
# Excitation-number restricted (enr) states
#
def enr_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.

    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.

    excitations : integer
        The maximum numbers of dimension

    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state

def _calc_matsubara_params(N_exp, coup_strength, cut_freq, temperature):
    """
    Calculate the Matsubara coefficents and frequencies

    Returns
    -------
    c, nu: both list(float)

    """
    c = []
    nu = []
    lam0 = coup_strength
    gam = cut_freq
    hbar = 1.
    beta = 1.0/temperature
    N_m = N_exp

    g = 2*np.pi / (beta*hbar)
    for k in range(N_m):
        if k == 0:
            nu.append(gam)
            c.append(lam0*gam*
                (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
        else:
            nu.append(k*g)
            c.append(4*lam0*gam*nu[k] /
                  ((nu[k]**2 - gam**2)*beta*hbar**2))

#    exp_coeff = c
#    exp_freq = nu
    return c, nu

class HEOMSolver():
    """
    HEOM solver with a single exponential function for the correlation functon.
    Valid for Lorentz-Drude spectral density at high-T

    """
    def __init__(self, H=None, c_ops=None, e_ops=None):
        self.c_ops = c_ops
        self.e_ops = e_ops
        self.e_ops = e_ops
        self.H = H
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def set_e_ops(self, e_ops):
        """
        set observable operators
        """
        self.e_ops = e_ops

        return

    def setH(self, H):
        self.H = H
        return

    def configure(self, c_ops, e_ops):
        self.c_ops = c_ops
        self.e_ops = e_ops
        return

    def run(self, rho0, dt, nt, temperature, cutoff, reorganization, nado):

        return _heom(self.H, rho0, self.c_ops, e_ops=self.e_ops, \
                  nt=nt, dt=dt, temperature=temperature, cutoff=cutoff, \
                  reorganization=reorganization, nado=nado)

    def propagator(self, dt, nt, temperature, cutoff, \
                            reorganization, nado):

        return _heom_propagator(self.H, self.c_ops, self.e_ops, temperature, cutoff, \
                                reorganization, nado, dt, nt)


    def correlation_2op_1t(self, rho0, a_op, b_op, dt, Nt, output='cor.dat'):
        '''
        two-point correlation function <A(t)B>

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        ops : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'lindblad'.
        output : TYPE, optional
            DESCRIPTION. The default is 'cor.dat'.

        Returns
        -------
        None.

        '''

        c_ops = self.c_ops
        H = self.H

        return _correlation_2p_1t(H, rho0, ops=[a_op, b_op], c_ops=c_ops, dt=dt,\
                          Nt=Nt, output=output)


    def correlation_3op_2t(self, rho0, ops, dt, Nt, Ntau):
        """
        Internal function for calculating the three-operator two-time
        correlation function:
        <A(t)B(t+tau)C(t)>
        using the Linblad master equation solver.
        """
        pass

        # # the solvers only work for positive time differences and the correlators
        # # require positive tau
        # # if state0 is None:
        # #     rho0 = steadystate(H, c_ops)
        # #     tlist = [0]
        # # elif isket(state0):
        # #     rho0 = ket2dm(state0)
        # # else:
        # #     rho0 = state0
        # H = self.H
        # c_ops = self.c_ops
        # rho_t = _lindblad(H, rho0, c_ops, dt=dt, Nt=Nt, return_result=True).rholist

        # a_op, b_op, c_op = ops

        # corr_mat = np.zeros([Nt, Ntau], dtype=complex)

        # for t_idx, rho in enumerate(rho_t):

        #     corr_mat[t_idx, :] = _lindblad(H, rho0=c_op @ rho @ a_op, \
        #                                    dt=dt, Nt=Ntau, c_ops=c_ops,\
        #         e_ops=[b_op], return_result=True).observables[:,0]

        # return corr_mat


def _heom(H, rho0, c_ops, e_ops, temperature, cutoff, reorganization,\
             nado, dt, nt, fname=None, return_result=True):
    '''

    terminator : ado[:,:,nado] = 0

    INPUT:
        T: in units of energy, kB * T, temperature of the bath
        reorg: reorganization energy
        nado : maximum depth of auxiliary density operators, truncation of the hierachy
        fname: file name for output

    '''
    nst = H.shape[0]
    ado = np.zeros((nst, nst, nado), dtype=np.complex128)     # auxiliary density operators
    ado[:,:,0] = rho0 # initial density matrix



    gamma = cutoff # cutoff frequency of the environment, larger gamma --> more Makovian
    # T = temperature/au2k
    T = temperature
    reorg = reorganization
    print('Temperature of the environment = {}'.format(T))
    print('Cutoff gamma/(kT) = {}'.format(gamma/T))

    if gamma/T > 0.8:
        print('WARNING: High-Temperature Approximation may fail.')

    print('Reorganization energy = {}'.format(reorg))

    # D(t) = (a + ib) * exp(- gamma * t)
    # a = np.pi * reorg * T  # initial value of the correlation function D(0) = pi * lambda * kB * T
    # b = 0.0

    # leading term of the Matsubara expansion
    # D0 = reorg * gamma * (coth(gamma/(2. * T)) - 1j)
    D0 = reorg * (2. * T - 1j * gamma)

    print('Amplitude of the fluctuations = {}'.format(D0))

    #sz = np.zeros((nstate, nstate), dtype=np.complex128)
    sz = c_ops[0] # collapse opeartor

    def L(ado):
        rhs = np.zeros_like(ado)

        rhs[:,:,0] = - 1j * comm(H, ado[:,:,0]) - \
                comm(sz, ado[:,:,1])

        for n in range(1, nado-1):
            rhs[:,:,n] += -1j * comm(H, ado[:,:,n]) + \
                            (- comm(sz, ado[:,:,n+1]) - n * gamma * ado[:,:,n] + n * \
                            (D0.real * commutator(sz, ado[:,:,n-1]) + \
                             1j * D0.imag * anticommutator(sz, ado[:,:,n-1])))
        return rhs

    # propagation time loop - HEOM
    observables = np.zeros((len(e_ops), nt), dtype=complex)
    t = 0.0
    for k in range(nt):

        t += dt # time increments

        ado = rk4(ado, L, dt)

    #     # store the reduced density matrix
    #     f.write(fmt.format(t, ado[0,0,0], ado[0,1,0], ado[1,0,0], ado[1,1,0]))

    #     #sz += -1j * commutator(sz, H) * dt
        observables[:, k] = [obs(ado[:, :, 0], e) for e in e_ops]
    # f.close()
    return observables

def _heom_propagator(H, c_ops, e_ops, temperature, cutoff, reorganization,\
             nado, dt, nt, fname=None):
    '''

    terminator : ado[:,:,nado] = 0

    INPUT:
        T: in units of energy, kB * T, temperature of the bath
        reorg: reorganization energy
        nado : auxiliary density operators, truncation of the hierachy
        fname: file name for output

    '''
    nst = H.shape[0]
    u = np.zeros((nado, nst**2, nst**2), dtype=np.complex128)     # auxiliary density operators
    u[0] = np.eye(nst**2) # propagator


    gamma = cutoff # cutoff frequency of the environment, larger gamma --> more Makovian
    T = temperature/au2k
    reorg = reorganization
    print('Temperature of the environment = {}'.format(T))
    print('High-Temperature check gamma/(kT) = {}'.format(gamma/T))

    if gamma/T > 0.8:
        print('WARNING: High-Temperature Approximation may fail.')

    print('Reorganization energy = {}'.format(reorg))

    # D(t) = (a + ib) * exp(- gamma * t)
    a = np.pi * reorg * T  # initial value of the correlation function D(0) = pi * lambda * kB * T
    b = 0.0
    print('Amplitude of the fluctuations = {}'.format(a))

    #sz = np.zeros((nstate, nstate), dtype=np.complex128)
    sz = c_ops[0] # collapse opeartor


    # f = open(fname,'w')
    # fmt = '{} '* 5 + '\n'

    # propagation time loop - HEOM
    L0 = operator_to_superoperator(H)
    Sm = operator_to_superoperator(sz)
    Sp = operator_to_superoperator(sz, kind='anticommutator')

    t = 0.0
    for k in range(nt):

        t += dt # time increments

        u[0] += -1j * L0 @ u[0] * dt - Sm @ u[1] * dt

        for n in range(1, nado-1):
            u[n] += -1j * L0 @ u[n] * dt + \
                        ((- Sm @ u[n+1]) - n * gamma * u[n] + n * \
                        (a *  Sm  + 1j * b * Sp) @ u[n-1] )*dt

        # store the reduced density matrix
        # f.write(fmt.format(t, u[0,0,0], u[0,1,0], u[1,0,0], u[1,1,0]))

        #sz += -1j * commutator(sz, H) * dt

    # f.close()
    return u


def heom_multiexp():
    # TODO

    N_c = 4
    N_m = 2
    N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*3 , N_c)
    print(N_he)
    print(he2idx)
    print(idx2he)

    s0, sx, sy, sz = pauli()
    mol = Mol(sz, dip=sx)

    coup_strength = 200/au2wavenumber
    cut_freq = 100/au2wavenumber
    temperature = 300/au2k

    for he_idx in range(N_he):
        he_state = list(idx2he[he_idx])
        print(he_state)

        n_excite = sum(he_state)

        c, nu = _calc_matsubara_params(N_m, coup_strength, cut_freq, temperature)
        print(c)
        print(nu)
        # The diagonal elements for the hierarchy operator
        # coeff for diagonal elements
        sum_n_m_freq = 0.0
        for k in range(N_m):
            sum_n_m_freq += he_state[k]*nu[k]

        unit_sup = mol.idm

        op = -sum_n_m_freq*unit_sup
        L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx)
        L_helems += L_he

        # Add the neighour interations
        he_state_neigh = copy(he_state)
        for k in range(N_m):

            n_k = he_state[k]
            if n_k >= 1:
                # find the hierarchy element index of the neighbour before
                # this element, for this Matsubara term
                he_state_neigh[k] = n_k - 1
                he_idx_neigh = he2idx[tuple(he_state_neigh)]

                op = c[k]*spreQ - np.conj(c[k])*spostQ
                if renorm:
                    op = -1j*norm_minus[n_k, k]*op
                else:
                    op = -1j*n_k*op

                L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                L_helems += L_he
                N_he_interact += 1

                he_state_neigh[k] = n_k

            if n_excite <= N_c - 1:
                # find the hierarchy element index of the neighbour after
                # this element, for this Matsubara term
                he_state_neigh[k] = n_k + 1
                he_idx_neigh = he2idx[tuple(he_state_neigh)]

                op = commQ
                if renorm:
                    op = -1j*norm_plus[n_k, k]*op
                else:
                    op = -1j*op

                L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                L_helems += L_he
                N_he_interact += 1

                he_state_neigh[k] = n_k


if __name__ == '__main__':
    pass