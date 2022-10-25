#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:01:00 2020

@author: Bing

Modules for computing signals with superoperator formalism in Liouville space

Instead of performing open quantum dynamics, the Liouvillian is directly diagonalized

Possible improvements:
    1. merge the Qobj class with QUTIP
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, issparse
from scipy.sparse.linalg import eigs
import scipy
import math
from numba import jit

from numpy import exp

from lime.phys import dag, pauli
from lime.mol import Result
# from qutip import Qobj as Basic


def liouvillian(H, c_ops):
    '''
    Construct the Liouvillian out of the Hamiltonian and collapse operators

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    c_ops : TYPE
        DESCRIPTION.

    Returns
    -------
    l : TYPE
        DESCRIPTION.

    '''
    # dissipator = 0.

    if c_ops is None:
        c_ops = []

    l = -1j * operator_to_superoperator(H)

    for c_op in c_ops:
        l = l + lindblad_dissipator(c_op)

    # l = operator_to_superoperator(H) + 1j * dissipator

    return l

class Qobj():
    def __init__(self, data=None, dims=None):
        """
        Class for quantum operators: is this useful?

        Parameters
        ----------
        n : int
            size of Hilbert space.

        Returns
        -------
        None.

        """
        # Basic.__init__(self, dims=dims, inpt=data)
        self.dims = dims
        self.data = data

        if data is None:
            self.data = np.random.randn(*dims)

        self.shape = self.data.shape
        return

    def dot(self, b):

        return Qobj(np.dot(self.data, b.data))

    def conjugate(self):
        return np.conjugate(self.data)

    def to_vector(self):
        return operator_to_vector(self.data)

    def to_super(self, type='commutator'):

        return operator_to_superoperator(self.data, type=type)

    def to_linblad(self, gamma=1.):
        l = self.data
        return gamma * (kron(l, l.conj()) - \
                operator_to_superoperator(dag(l).dot(l), type='anticommutator'))


def liouville_space(N):
    """
    constuct liouville space out of N Hilbert space basis |ij>
    """
    return

def operator_to_vector(rho):
    """
    transform an operator/density matrix to an superoperator in Liouville space

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(rho, np.ndarray):
        return rho.flatten()
    else:
        return rho.toarray().flatten()

def dm2vec(rho):
    """
    transform an operator/density matrix to a vector in Liouville space

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


    if issparse(rho):
        n, m = rho.shape
        return rho.tolil().reshape((n*m, 1))
    else:
        return rho.flatten()


# def vec2dm(rho):
#     """
#     transform an operator/density matrix to a vector in Liouville space

#     Parameters
#     ----------
#     A : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     if isinstance(rho, np.ndarray):
#         return np.reshape()
#     else:
#         return rho.toarray().flatten()

def op2sop(a, kind='commutator'):
    return operator_to_superoperator(a, kind=kind)

def to_super(a, kind='commutator'):
    return operator_to_superoperator(a, kind=kind)
  
def vec2mat_index(N, I):
    """
    Convert a vector index to a matrix index pair that is compatible with the
    vector to matrix rearrangement done by the vec2mat function.
    
    From Qutip.
    """
    j = int(I / N)
    i = I - N * j
    return i, j


def mat2vec_index(N, i, j):
    """
    Convert a matrix index pair to a vector index that is compatible with the
    matrix to vector rearrangement done by the mat2vec function.
    
    From Qutip.
    """
    
    return i + N * j

def operator_to_superoperator(a, kind='commutator'):
    """
    promote an operator/density matrix to an superoperator in
    Liouville space

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    N = a.shape[-1]

    idm = identity(N)

    if kind in ['commutator', 'c', '-']:

        return kron(a, idm) - kron(idm, a.T)

    elif kind in ['left', 'l']:

        # elementwise operator for defining the commutator

        # for n in range(N2):
        #     i, j = divmod(n, N)
        #     for m in range(N2):
        #         k, l = divmod(m, N)
        #         am[n, m] = a[i, k] * idm[j,l]

        return kron(a, idm)

    elif kind in ['right', 'r']:

        return kron(idm, a.T)

    elif kind in ['anticommutator', 'a', '+']:

        return kron(a, idm) + kron(idm, a.T)

    else:

        raise ValueError('Error: superoperator {} does not exist.'.format(kind))
        

def lindblad_dissipator(l):
    return kron(l, l.conj()) - 0.5 *\
                      operator_to_superoperator(dag(l).dot(l), kind='anticommutator')
    # return gamma * (left(l).dot(right(dag(l))) - 0.5 *\
    #                  operator_to_superoperator(dag(l).dot(l), type='anticommutator'))


def left(a):
    if issparse(a):
        idm = identity(a.toarray().shape[-1])
    else:
        idm = identity(a.shape[-1])
    return kron(a, idm)

def right(a):

    if issparse(a):
        idm = identity(a.toarray().shape[-1])
    else:
        idm = identity(a.shape[-1])

    return kron(idm, a.T)

def kraus(a):
    """
    Kraus superoperator a rho a^\dag = a^\dag_R a_L

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    al = left(a)
    ar = right(dag(a))

    return ar.dot(al)

# def obs(rho, a):
#     """
#     Return expectation value of a for rho in Liouville space.

#     Parameters
#     ----------
#     rho : TYPE
#         DESCRIPTION.
#     a : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """

#     idv = operator_to_vector(np.identity(a.shape[-1]))
#     return np.vdot(idv, left(a).dot(rho))

# @jit
def obs(rho, a):
    return np.vdot(operator_to_vector(dag(a)), rho)

def trace(rho):
    n = math.isqrt(len(rho))
    return np.vdot(operator_to_vector(np.identity(n)), rho)

def resolvent(omega, L):
    '''
    Resolvent of the Lindblad quantum master equation

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    L : 2d array
        full liouvillian
    Returns
    -------
    None.

    '''
    idm = np.identity(L.shape[0])

    return np.linalg.inv(omega * idm - L)

def _correlation_2p_1f(omegas, rho0, ops, L):
    a, b = ops
    out = np.zeros(len(omegas))

    for j in range(len(omegas)):
        omega = omegas[j]
        r = resolvent(omega, L)
        out[j] = operator_to_vector(a.T).dot(r.dot(operator_to_vector(b.rho0)))

    return out

def _correlation_2p_1t(omegas, rho0, ops, L):
    a, b = ops
    cor = np.zeros(len(omegas))

    for j in range(len(omegas)):
        omega = omegas[j]
        r = resolvent(omega, L)
        cor[j] = operator_to_vector(a.T).dot(r.dot(operator_to_vector(b.rho0)))

    return cor

def sort(eigvals, eigvecs):

    idx = np.argsort(eigvals)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return eigvals, eigvecs

def cdot(a, b):
    """
    matrix product of a.H.dot(b)

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return dag(a) @ b

def absorption(mol, omegas, c_ops):
    """
    superoperator formalism for absorption spectrum

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    omegas: vector
        detection window of the spectrum
    c_ops : TYPE
        list of collapse operators

    Returns
    -------
    None.

    """

    gamma = 0.02

    l = op2sop(H) + 1j * c_op.to_linblad(gamma=gamma)


    ntrans = 3 * nstates # number of transitions
    eigvals1, U1 = eigs(l, k=ntrans, which='LR')

    eigvals1, U1 = sort(eigvals1, U1)

    # print(eigvals1)

    omegas = np.linspace(0.1 , 10.5, 200)

    rho0 = Qobj(dims=[10,10])
    rho0.data[0,0] = 1.0

    ops = [sz, sz]

    # out = correlation_2p_1t(omegas, rho0, ops, L)
    # print(eigvecs)
    eigvals2, U2 = eigs(dag(l), k=ntrans, which='LR')


    eigvals2, U2 = sort(eigvals2, U2)

    #idx = np.where(eigvals2.real > 0.2)[0]
    #print(idx)


    norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(ntrans)]

    la = np.zeros(len(omegas), dtype=complex) # linear absorption
    for j, omega in enumerate(omegas):

        for n in range(ntrans):

            la[j] += np.vdot(dip.to_vector(), U1[:,n]) * \
                 np.vdot(U2[:,n], dip.dot(rho0).to_vector()) \
                 /(omega - eigvals1[n]) / norm[n]


    fig, ax = plt.subplots()
    # ax.scatter(eigvals1.real, eigvals1.imag)
    ax.plot(omegas, -2 * la.imag)

    return

class Lindblad_solver:
    def __init__(self, H, c_ops=None):
        """
        Liouville equation solver.

        Parameters
        ----------
        H : TYPE
            DESCRIPTION.
        c_ops : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.H = H
        self.c_ops = c_ops
        self.L = None
        self.dim = H.shape[-1]**2

        self.idv = operator_to_vector(np.identity(H.shape[-1]))
        self.left_eigvecs = None
        self.right_eigvecs = None
        self.eigvals = None
        self.norm = None
        # self.nstates = None # number of states used while diagonalizing L

    def liouvillian(self):
        L = liouvillian(self.H, self.c_ops)
        self.L = L
        return L

    def eigenstates(self, k=None):

        if self.L is None:
            # raise ValueError('L is None. Call liouvillian to construct L first.')
            L = self.liouvillian()
        else:
            L = self.L

        N = L.shape[-1]

        if k is None:

            w, vl, vr = scipy.linalg.eig(L.toarray(), left=True, \
                                              right=True)
            self.eigvals = w
            self.left_eigvecs = vl
            self.right_eigvecs = vr
            # self.norm = [np.vdot(vl[:,n], vr[:,n]) for n in range(self.dim)]

            self.norm = np.diagonal(cdot(vl, vr)).real

        elif k < N-1:
            # right eigenvectors of L
            evals1, U1 = eigs(L, k=k, which='LR')
            evals1, U1 = sort(evals1, U1)

            # left
            evals2, U2 = eigs(dag(L), k=k, which='LR')
            evals2, U2 = sort(evals2, U2)

        else:
            raise ValueError('k should be < the size of the matrix.')

        return w, vr, vl

    def evolve(self, rho0, tlist, e_ops):

        result = Result(times=tlist)
        # evals, evecs_r, evecs_l = self.eigvals, self.eigvecs_right,\
        #     self.eigvecs_left


        # if self.eigvals is None:
        #     evals, U1, U2 = self.eigenstates(k=k)
        # else:

        evals = self.eigvals
        U1 = self.right_eigvecs
        U2 = self.left_eigvecs
        norm = self.norm

        # print('norm', norm)

        rho0 = operator_to_vector(rho0)

        # if k is None:
        #     k = self.dim
        k = U1.shape[-1]

        observables = np.zeros((len(tlist), len(e_ops)), dtype=complex)

        coeff = [np.vdot(U2[:,n], rho0)/norm[n] for n in range(k)]

        for i, t in enumerate(tlist):

            rho = U1.dot(coeff * exp(evals * t))

            # print(trace(rho))

            observables[i, :] = [obs(rho, e_op) for e_op in e_ops]

        result.observables = observables

        return result


    def correlation_2op_1t(self, rho0, ops, tlist):
        """
        Compute <A(t)B> by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        a, b = ops

        # if self.eigvals is None:
        #     evals, U1, U2 = self.eigenstates(k=k)
        # else:

        evals, U1, U2 = self.eigvals, self.right_eigvecs, self.left_eigvecs

        # if k is None:
        #     k = self.dim
        k = U1.shape[-1]

        # norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(k)]
        norm = self.norm

        idv = self.idv

        cor = np.zeros(len(tlist), dtype=complex)

        coeff = np.array([np.vdot(idv, left(a).dot(U1[:,n])) * \
                    np.vdot(U2[:,n], operator_to_vector(b.dot(rho0)))/norm[n]\
                        for n in range(k)])

        for i, t in enumerate(tlist):
            cor[i] = np.sum(exp(evals * t) *  coeff)

        return cor

    def correlation_2op_1w(self, rho0, ops, w):
        """
        Compute S(w) = <A(w)B> = int_0^\infty <A(t)B> exp(iwt) dt
        by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        a, b = ops

        # if self.eigvals is None:
        #     evals, U1, U2 = self.eigenstates(k=k)
        # else:

        evals, U1, U2 = self.eigvals, self.right_eigvecs, self.left_eigvecs

        k = len(evals)
        norm = self.norm

        idv = self.idv

        S = np.zeros(len(w), dtype=complex)

        coeff = np.array([np.vdot(idv, left(a).dot(U1[:,n])) * \
                    np.vdot(U2[:,n], operator_to_vector(b.dot(rho0)))/norm[n]\
                        for n in range(k)])

        for i in range(len(w)):
            S[i] = np.sum( -1./(evals + 1j * w[i]) *  coeff)

        return S

    def correlation_3op_1t(self, rho0, ops, t):
        """
        Compute <A(t)B> by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        a, b, c = ops

        evals, U1, U2 = self.eigvals, self.right_eigvecs, self.left_eigvecs

        k = U1.shape[-1]

        norm = self.norm

        idv = self.idv

        cor = np.zeros(len(t), dtype=complex)

        coeff = np.array([np.vdot(idv, left(b).dot(U1[:,n])) * \
                    np.vdot(U2[:,n], operator_to_vector(c @ rho0 @ a))/norm[n]\
                        for n in range(k)])


        for i in range(len(t)):
            cor[i] = np.sum( exp(evals * t[i]) *  coeff)

        return cor

    def correlation_3op_1w(self, rho0, ops, w):
        """
        Compute <A(t)B> by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        a, b, c = ops

        evals, U1, U2 = self.eigvals, self.right_eigvecs, self.left_eigvecs

        k = U1.shape[-1]

        norm = self.norm

        idv = self.idv

        cor = np.zeros(len(w), dtype=complex)

        coeff = np.array([np.vdot(idv, left(b).dot(U1[:,n])) * \
                    np.vdot(U2[:,n], operator_to_vector(c @ rho0 @ a))/norm[n]\
                        for n in range(k)])


        for i in range(len(w)):
            cor[i] = np.sum( -1./(evals + 1j * w[i]) *  coeff)

        return cor

    def correlation_3op_2t(self, rho0, ops, tlist, taulist, k=None):
        """
        Compute <A(t)B(t+tau)C(t)> by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        a, b, c = ops
        rho0 = operator_to_vector(rho0)

        cor = np.zeros((len(tlist), len(taulist)), dtype=complex)

        # diagonalize the Liouvillian
        # evals, U1, U2 = self.eigenstates(k=k)
        evals = self.eigvals
        U1 = self.right_eigvecs
        U2 = self.left_eigvecs

        if k is None:
            k = self.dim

        # assert(np.allclose(evals1, evals2.conj()))
        # print(evals2)
        # assert(evals1.imag.all() > 0)
        # assert(np.allclose(evals1, evals2.conj(), atol=1e-4))

        #norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(k)]
        norm = self.norm

        idv = self.idv

        coeff = np.zeros((k,k), dtype=complex)

        for m in range(k):
            for n in range(k):
                coeff[m, n] = np.vdot(idv, left(b).dot(U1[:, m])) * \
                    np.vdot(U2[:,m], right(a).dot(left(c).dot(U1[:,n])))/norm[m]\
                    * np.vdot(U2[:, n], rho0)/norm[n]


        # for i, t in enumerate(tlist):
        #     for j, tau in enumerate(taulist):

        tmp1 = exp(np.outer(evals, taulist))
        tmp2 = exp(np.outer(evals, tlist))

        cor = tmp1.T @ coeff @ tmp2

        return cor

    def correlation_4op_2t(self, rho0, ops, tlist, taulist, k=None):
        """
        Compute <A(t)B(t+tau)C(t)> by diagonalizing the Liouvillian.

        Returns
        -------
        1D array.
            correlation function.

        """
        if len(ops) != 4:
            raise ValueError('Number of operators is not 4.')
        else:
            a, b, c, d = ops

        corr = self.correlation_3op_2t(rho0=rho0, ops=[a, b@c, d], tlist=tlist, \
                                       taulist=taulist, k=k)
        return corr



if __name__ == '__main__':

    from lime.units import au2fs, au2ev

    s0, sx, sy, sz = pauli()

    nstates = 2

    H = np.diagflat([0, 1])/au2ev + 0.5/au2ev*sx
    #h = np.diagflat(np.arange(10))

    dip = np.zeros(H.shape)
    dip[0,:] = dip[:,0] = np.random.rand(nstates)

    c_op = dip
    gamma = 0.05

    # l = h.to_super() + 1j * c_op.to_linblad(gamma=gamma)
    # l = liouvillian(H, c_ops=[gamma*c_op])

    # ntrans = 3 * nstates # number of transitions
    # eigvals1, U1 = eigs(l, k=ntrans, which='LR')

    # eigvals1, U1 = sort(eigvals1, U1)

    # print(eigvals1.real)

    # omegas = np.linspace(0.1 , 10.5, 200)

    from lime.phys import ket2dm
    from lime.style import matplot, subplots

    rho0  =  ket2dm(np.array([1.0, 0.0]))
    # rho0.data[0,0] = 1.0

    ops = [sx, sx]

    # out = correlation_2p_1t(omegas, rho0, ops, L)
    # print(eigvecs)
    # eigvals2, U2 = eigs(dag(l), k=ntrans, which='LR')


    # eigvals2, U2 = sort(eigvals2, U2)

    # #idx = np.where(eigvals2.real > 0.2)[0]
    # #print(idx)


    # norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(ntrans)]

    # la = np.zeros(len(omegas), dtype=complex) # linear absorption
    # for j, omega in enumerate(omegas):

    #     for n in range(ntrans):

    #         la[j] += np.vdot(operator_to_vector(dip), U1[:,n]) * \
    #              np.vdot(U2[:,n], operator_to_vector(dip.dot(rho0))) \
    #              /(omega - eigvals1[n]) / norm[n]
    # fig, ax = plt.subplots()
    # # ax.scatter(eigvals1.real, eigvals1.imag)
    # ax.plot(omegas, -2 * la.imag)
    # plt.show()



    solver = Lindblad_solver(H, c_ops=[0.02*sx])
    # solver.liouvillian()
    solver.eigenstates()
    # print(solver.right_eigvecs)


    times = np.linspace(0, 40)/au2fs

    result = solver.evolve(rho0, tlist=times, e_ops=[sx, sz])

    # cor = solver.correlation_2op_1t(rho0=rho0, ops=[sx, sx], tlist=times)
    w = np.linspace(0.4, 2., 100)/au2ev
    S = solver.correlation_3op_1w(rho0=rho0, ops=[sx, sx, sx], w=w)

    fig, ax = subplots()
    ax.plot(w * au2ev, S.real)
    # print(cor)
    # fig, ax = matplot(times, times, cor.real)

    # cor = solver.correlation_3op_2t(ops=[sx, sx, sz], taulist=times, tlist=times, rho0=rho0, k=4)

    # fig, ax = subplots()
    # # # ax.scatter(eigvals1.real, eigvals1.imag)
    # ax.plot(result.times, result.observables[:,1])
    # plt.show()
    # from lime.style import matplot
    # fig, ax = matplot(times, times, cor.real)

