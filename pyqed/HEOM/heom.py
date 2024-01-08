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

if __name__ == '__main__':
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
