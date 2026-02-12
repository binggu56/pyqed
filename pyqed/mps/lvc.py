# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:09:06 2019

TEBD for quantum dynamics of vibronic model systems

@author: Bing
"""

""" Comparison of the entanglement growth S(t) following a global quench using
the first order MPO based time evolution for the XX model with the TEBD algorithm.

The TEBD algorithm is also used to recompress the MPS after the MPO time evolution.
This is simpler to code but less efficient than a variational optimization.
See arXiv:1407.1832 for details and how to extend to higher orders.

Frank Pollmann, frankp@pks.mpg.de
"""

import numpy as np
import pylab as pl
from scipy.linalg import expm, block_diag
import logging
import copy
import warnings

from scipy.fftpack import fft, ifft, fftfreq, fftn, ifftn

from pyqed.mps.decompose import decompose, compress

from pyqed import gwp, discretize, pauli, sigmaz

from pyqed.mps.mps import Site





def make_block_from_site(site):
    """Makes a brand new block using a single site.

    You use this function at the beginning of the DMRG algorithm to
    upgrade a single site to a block.

    Parameters
    ----------
    site : a Site object.
        The site you want to upgrade.

    Returns
    -------
    result: a Block object.
        A brand new block with the same contents that the single site.

    Postcond
    --------
    The list for the operators in the site and the block are copied,
    meaning that the list are different and modifying the block won't
    modify the site.

    Examples
    --------
    >>> from dmrg101.core.block import Block
    >>> from dmrg101.core.block import make_block_from_site
    >>> from dmrg101.core.sites import SpinOneHalfSite
    >>> spin_one_half_site = SpinOneHalfSite()
    >>> brand_new_block = make_block_from_site(spin_one_half_site)
    >>> # check all it's what you expected
    >>> print brand_new_block.dim
    2
    >>> print brand_new_block.operators.keys()
    ['s_p', 's_z', 's_m', 'id']
    >>> print brand_new_block.operators['s_z']
    [[-0.5  0. ]
     [ 0.   0.5]]
    >>> print brand_new_block.operators['s_p']
    [[ 0.  0.]
     [ 1.  0.]]
    >>> print brand_new_block.operators['s_m']
    [[ 0.  1.]
     [ 0.  0.]]
    >>> # operators for site and block are different objects
    >>> print ( id(spin_one_half_site.operators['s_z']) ==
    ...		id(brand_new_block.operators['s_z']) )
    False
    """
    result = Block(site.dim)
    result.operators = copy.deepcopy(site.operators)
    return result



# def make_updated_block_for_site(transformation_matrix,
# 		                operators_to_add_to_block):
#     """Make a new block for a list of operators.

#     Takes a dictionary of operator names and matrices and makes a new
#     block inserting in the `operators` block dictionary the result of
#     transforming the matrices in the original dictionary accoring to the
#     transformation matrix.

#     You use this function everytime you want to create a new block by
#     transforming the current operators to a truncated basis.

#     Parameters
#     ----------
#     transformation_matrix : a numpy array of ndim = 2.
#         The transformation matrix coming from a (truncated) unitary
# 	transformation.
#     operators_to_add_to_block : a dict of strings and numpy arrays of ndim = 2.
#         The list of operators to transform.

#     Returns
#     -------
#     result : a Block.
#         A block with the new transformed operators.
#     """
#     cols_of_transformation_matrix = transformation_matrix.shape[1]
#     result = Block(cols_of_transformation_matrix)
#     for key in operators_to_add_to_block.keys():
#         ult.add_operator(key)
#         ult.operators[key] = transform_matrix(operators_to_add_to_block[key],
# 			                         transformation_matrix)
#     return result




def is_right_canonical(M):
    N = len(M)
    for l in range(N):
        Mdag = M[l].conj().T #right-top-left
        MMdag = np.einsum('ijk,kjl',M[l],Mdag) #top-bottom
        I = np.eye(np.shape(M[l])[0]) #(leg order is indiferent)
        print('l =', l, ': max(|M[l] · M[l]^† - I|) =', np.max(abs(MMdag-I)))

# parameters
N = 10
d = 3
D = 20

# random MPS
'''
    Order of legs: left-bottom-right.
    Note: this is the conventional order used for MPSs in the code.
'''
Mrand = []
Mrand.append(np.random.rand(1,d,D))
for l in range(1,N-1):
    Mrand.append(np.random.rand(D,d,D))
Mrand.append(np.random.rand(D,d,1))

Mleft = LeftCanonical(Mrand)

def is_left_canonical(M):
    L = len(M)
    for l in range(L):
        Mdag = Mleft[l].conj().T #right-top-left
        '''
            Note: as a consequence of the conventional leg order chosen for the MPSs, the corresponding hermitian
                conjugate versions are ordered as right-top-left.
        '''
        MdagM = np.einsum('ijk,kjl',Mdag,Mleft[l]) #bottom-top
        I = np.eye(np.shape(Mleft[l])[2]) #(leg order is indiferent)
        print('l =', l, ': max(|M[l]^† · M[l] - I|) =', np.max(abs(MdagM-I)))


def mps_to_tensor(mps):
    B0, B1, B2 = mps

    # obs[k] = np.einsum('ib, jk, kb->', B0[:,0, :].conj(), sp@sm, B0[:, 0, :])
    psi = np.einsum('ib, bjc, ck ->ijk', B0[0,:,:], B1, B2[:, :, 0])
    return psi

def tensor_to_vec(psi):
    return psi.flatten()





s0 = np.eye(2)
sp = np.array([[0.,1.],[0.,0.]])
sm = np.array([[0.,0.],[1.,0.]])

# def initial_state(d, chi_max, L, dtype=complex):
#     """
#     Create an initial product state.
#     input:
#         L: number of sites
#         chi_max: maximum bond dimension
#         d: local dimension for each site

#     return
#     =======
#     MPS in right canonical form S0-B0--B1-....B_L
#     """
#     B_list = []
#     s_list = []
#     for i in range(L):
#         B = np.zeros((d,1,1),dtype=dtype)
#         B[np.mod(i,2),0,0] = 1.
#         s = np.zeros(1)
#         s[0] = 1.
#         B_list.append(B)
#         s_list.append(s)
#     s_list.append(s)
#     return B_list,s_list





def apply_mpo_svd(B_list, s_list, w_list, chi_max):
    """
    Apply the MPO to an MPS.


    Parameters
    ----------
    B_list : TYPE
        DESCRIPTION.
    s_list : TYPE
        DESCRIPTION.
    w_list : TYPE
        DESCRIPTION.
    chi_max : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

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


# class TimeEvolvingBlockDecimation:

class TEBD:
    def __init__(self, D):
        self.D = D




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
    psi_k = psi_k * np.exp(-0.5j * k**2 * dt)
    psi = ifft(psi_k)

    return psi

# def kinetic(k, B_list):
#     """
#     kinetic energy (KE) component of the one-step evolution operator
#     :math:`e^{-i T \delta t )` on the MPS

#               where T is the total KE operator
#     """
#     L = len(B_list)

#     for i in range(L):
#         _, chi1, chi2 = np.shape(B_list[i])
#         # for a in range(chi1):
#         #     for b in range(chi2):
#                 # B_list[i][:,a,b] = k_evolve_1d(k, B_list[i][:,a,b])
#         B_list[i] = ifftn(np.einsum('i, iab -> iab', np.exp(-0.5j * k**2 * dt), \
#                                     fftn(B_list[i], axes=(0))), axes=(0))

#     return B_list

def kinetic(k, B_list):
    """
    kinetic energy (KE) component of the one-step evolution operator
    :math:`e^{-i T \delta t )` on the MPS

              where T is the total KE operator

    The factors are of shape [chi1, d, chi2]
    Returns
    -------

    """
    L = len(B_list)

    for i in range(L):
        _, chi1, chi2 = np.shape(B_list[i])
        # for a in range(chi1):
        #     for b in range(chi2):
                # B_list[i][:,a,b] = k_evolve_1d(k, B_list[i][:,a,b])
        B_list[i] = ifftn(np.einsum('i, aib -> aib', np.exp(-0.5j * k**2 * dt), \
                                    fftn(B_list[i], axes=(1))), axes=(1))

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
    potential energy component of the evolution operator e^{-i * dt * V) on the MPS
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

            C = np.tensordot(B_list[i1], B_list[i2], axes=(2,1))
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


def potential_svd(B_list, s_list, v_mps, chi_max):
    """
    potential energy component of the one-step evolution operator :math:`e^{-i V \delta t)` on the MPS
    input:
        V: n-d array, PES
        L: int
            number of sites
    """
    L = len(B_list)


    # U = np.exp(-1j * dt * V)

    # decompose the potential energy matrix

    # vf, vs = decompose(U, rank=chi_max)

    As = []
    for i in range(L):

        a1, d, a2 = v_mps[i].shape
        chi1, d, chi2 = B_list[i].shape

        A = np.einsum('aib, cid-> aci bd', v_mps[i], B_list[i])
        A = np.reshape(A, (a1 * chi1, d, a2 * chi2))

        As.append(A.copy())

    As, Ss = compress(As, chi_max=chi_max)

    # for i in range(L-1):

    #         i1 = i; i2 = i+1

    #         chi1 = B_list[i1].shape[1]
    #         chi3 = B_list[i2].shape[2]

    #         C = np.tensordot(B_list[i1], B_list[i2], axes=(2,1))
    #         C = np.einsum('iajb, ij ->iajb', C, U)
    #         #C = np.tensordot(C, U, axes=([0,2],[2,3]))

    #         # ? Why not directly SVD the C tensor?

    #         theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))

    #         C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))

    #         # Schmidt decomposition #
    #         X, Y, Z = np.linalg.svd(theta)
    #         Z=Z.T

    #         W = np.dot(C,Z.conj())
    #         chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

    #         # Obtain the new values for B and l #
    #         invsq = np.sqrt(sum(Y[:chi2]**2))
    #         s_list[i2] = Y[:chi2]/invsq
    #         B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq
    #         B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))

    return As, Ss





class SPO:
    """
    MPS representation for adiabatic wave packet dynamics
    """
    def __init__(self, domains, levels, chi_max, dvr_type='sinc'):
        """
        Use TEBD to optmize the MPS and to project it back.

        The first site is the electronic, while the rest represents the vibrational
        modes. :math:`| \alpha n_1 n_2 \cdots n_d\rangle`

        """
        self.nsites = self.L = len(levels)
        if dvr_type == 'sinc': # particle in a box eigenstates
            self.x = []
            for d in range(self.ndim):
                a, b = domains[d]
                self.x.append(discretize(a, b, levels[d]))

        self.dims = [len(_x) for _x in self.x] # [B.shape[1] for B in B_list]

        # self.nsites = self.L = self.ndim  # nuclear degrees of freedom

        self.chi_max = chi_max
        # make_V_list(X, Y)

        self.v = None

    def set_apes(self, v):


        assert(v.shape == tuple(self.dims))

        self.v = v




    def run(self, B_list, s_list, dt=0.001, nt=10, nout=1):

        chi_max = self.chi_max
        v = self.v
        V = np.exp(-1j * v * dt)

        # decompose the potential propagator
        vf, vs = decompose(V, chi_max)

        X = np.diag(x)
        Xs = []

        for n in range(nt):
            for k1 in range(nout):

                B_list = kinetic(kx, B_list)
                B_list, s_list = potential_svd(B_list, s_list, vf, chi_max)

            Xs.append(self.expect_one_site(B_list, X))

        return Xs


    def expect_one_site(self, mps, a, n=-1):
        """
        compute single-site observable
        """

        return expect_one_site(mps, a=a, n=n)

    def expect_two_sites(self, mps, e_ops, n):
        pass


def expect_one_site(mps, a, n=-1):
    """
    how to compute the observables
    """
    d = mps[n].shape[1]
    assert(a.shape == (d,d))

    if n == -1:
        M = mps[n]
        return np.einsum('aib, ij, ajb', M.conj(), a, M)
    else:
        raise NotImplementedError


def overlap(bra, ket):
    """

    Compute the overlap between two MPSs with the same sites

    .. math::

        S = \langle \phi | \chi \rangle

    Note
    ----
    It does not work for two MPS with different basis sets.

    Parameters
    ----------
    bra : TYPE
        DESCRIPTION.
    ket : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert(ket.dims == bra.dims)

    C = np.einsum('aib, aic -> bc', bra.factors[0].conj(), ket.factors[0])

    for i in range(1, bra.L):
        C = np.einsum('aib, ac, cid -> bd', bra.factors[i].conj(), C, ket.factors[i])


    return C[0, 0]



if __name__ == "__main__":

    from pyqed import interval

    def initial_state(d, chi_max, L, dtype=complex):
        """
        Create an initial product state.
        input:
            L: number of sites
            chi_max: maximum bond dimension
            d: local dimension for each site

        return
        =======
        MPS in right canonical form S0-B0--B1-....B_L
        """
        B_list = []
        s_list = []

        g = gwp(x, x0=-1)

        for i in range(L):
            B = np.zeros((1,d,1),dtype=dtype)
            B[0, :, 0] = g

            s = np.zeros(1)
            s[0] = 1.
            B_list.append(B)
            s_list.append(s)

        s_list.append(s)

        # return B_list,s_list
        return MPS(B_list)



    # Define Pararemeter here
    delta = dt = 0.02
    L = 2
    chi_max = 10
    N_steps = 10

    # # grid
    d = 2**4 # local size of Hilbert space
    # x = np.linspace(-2,2,d, endpoint=False)
    # y = np.linspace(-2,2,d, endpoint=False)


    # print(interval(x))

    # X, Y = np.meshgrid(x,y)

    # V = make_V_list(X,Y)
    def pes(x):
        dim = len(x)
        v = 0
        for d in range(dim):
            v += 0.5 *d* x[d]**2
        v += 0.3 * x[0] * x[2] #+ x[0]**2 * 0.2
        return v


    # a = np.random.randn(3, 3, 3)
    level = 4
    # n = 2**level - 1 # number of grid points for each dim
    x = np.linspace(-6, 6, 2**level, endpoint=False)[1:]
    n = len(x)

    dx = interval(x)


    v = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v[i, j, k] = pes([x[i], x[j], x[k]])
    # frequency space
    kx = 2. * np.pi * fftfreq(n, dx)



    # TEBD algorithm
    L = 3
    # B_list,s_list

    # initialize a vibronic state
    # mps = initial_state(n, chi_max, L, dtype=complex)
    mps = vibronic_state(x)



    # spo = SPO(L, dims=[n, ] * 3, chi_max=6)
    # spo.set_apes(v)

    # Xs = spo.run(B_list, s_list, dt=0.04, nt=500)

    # # print(len(B_list), len(s_list))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(Xs)

    # S = [0]
    # for step in range(N_steps):

    #     B_list = kinetic(k, B_list)
    #     B_list, s_list = potential(B_list, s_list, V, chi_max)

    #     s2 = np.array(s_list[L//2])**2
    #     S.append(-np.sum(s2*np.log(s2)))

    # pl.plot(delta*np.arange(N_steps+1),S)
    # pl.xlabel('$t$')
    # pl.ylabel('$S$')
    # pl.legend(['MPO','TEBD'],loc='upper left')
    # pl.show()