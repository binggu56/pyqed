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
from scipy.linalg import expm, block_diag
import logging
import copy
import warnings

from scipy.fftpack import fft, ifft, fftfreq, fftn, ifftn

from pyqed.mps.decompose import decompose, compress

from pyqed import gwp, discretize, pauli, sigmaz

class MPS:
    def __init__(self, Bs, Ss=None, homogenous=True, bc='open', form="B"):
        """
        class for matrix product states.

        Parameters
        ----------
        mps : list
            list of 3-tensors.

        Returns
        -------
        None.

        """
        assert bc in ['finite', 'infinite']
        self.Bs = self.factors = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'open' else self.L


        self.data = self.factors = Bs
        # self.nsites = self.L = len(mps)
        if homogenous:
            self.dim = Bs[0].shape[1]
        else:
            self.dims = [B.shape[1] for B in Bs] # physical dims of each site

        # self._mpo = None

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss], self.bc)

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.nbonds)]

    # def decompose(self, chi_max):
    #     pass

    def __add__(self, other):
        assert len(self.data) == len(other.data)
        # for different length, we should choose the maximum one
        C = []
        for j in range(self.sites):
            tmp = block_diag(self.data[j], other.data[j])
            C.append(tmp.copy())

        return MPS(C)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=(1, 1))  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        B = self.Bs[0]  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self.Bs[i]
            T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
            # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
        T = np.reshape(T, (chi**2, chi**2))
        # Obtain the 2nd largest eigenvalue
        eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi =  -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i) # vL i vR
        C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
        C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self.Bs[j]  # vL k vR
        C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
        C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
        C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
        return C

    def evolve_v(self, other):
        """
        apply the evolution operator due to V(R) to the wavefunction in the TT format

                    |   |
                ---V---V---
                    |   |
                    |   |
                ---A---A---
            =
                    |   |
                ===B===B===

        .. math::

            U_{\beta_i \beta_{i+1}}^{j_i} A_{\alpha_i \alpha_{i+1}}^{j_i} =
            A^{j_i}_{\beta_i \alpha_i, \beta_{i+1} \alpha_{i+1}}

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        MPS object.

        """
        assert(other.L == self.L)
        assert(other.dims == self.dims)

        As = []
        for n in range(self.L):

            al, d, ar = self.factors[n].shape
            bl, d, br = other.factors[n].shape

            c = np.einsum('aib, cid -> acibd', other.factors[n], self.factors[n])
            c.reshape((al * bl, d, ar * br))
            As.append(c.copy())

        return MPS(As)

    # def evolve_t(self):
    #     pass

    def left_canonicalize(self):
        pass

    def right_canonicalize(self):
        pass

    def left_to_vidal(self):
        pass

    def left_to_right(self):
        pass

    # def build_U_mpo(self):
    #     # build MPO representation of the short-time propagator
    #     pass

    # # def run(self, dt=0.1, Nt=10):
    # #     pass

    # # def obs_local(self, e_op, n):
    # #     pass

    # def apply_mpo(self):
    #     pass

    def compress(self, chi_max):
        return MPS(compress(self.factors, chi_max)[0])


# class MPS:
#     def __init__(self, factors, homogenous=False, form=None):
#         """
#         class for matrix product states.

#         Parameters
#         ----------
#         mps : list
#             list of 3-tensors. [chi1, d, chi2]
#         chi_max:
#             maximum bond order used in compress. Default None.

#         Returns
#         -------
#         None.

#         """
#         self.factors = self.data = factors
#         self.nsites = self.L = len(factors)
#         self.nbonds = self.nsites - 1
#         # self.chi_max = chi_max

#         self.form = form

#         if homogenous:
#             self.dims = [mps[0].shape[1], ] * self.nsites
#         else:
#             self.dims = [t.shape[1] for t in factors] # physical dims of each site

#         # self._mpo = None

#     def bond_orders(self):
#         return [t.shape[2] for t in self.factors] # bond orders


#     def compress(self, chi_max):
#         return MPS(compress(self.factors, chi_max)[0])

#     def __add__(self, other):
#         assert len(self.data) == len(other.data)
#         # for different length, we should choose the maximum one
#         C = []
#         for j in range(self.sites):
#             tmp = block_diag(self.data[j], other.data[j])
#             C.append(tmp.copy())

#         return MPS(C)

    # def build_mpo_list(self):
    #     # build MPO representation of the propagator
    #     pass

    # def copy(self):
    #     return copy.copy(self)

    # def run(self, dt=0.1, Nt=10):
    #     pass

    # def obs_single_site(self, e_op, n):
    #     pass

    # def two_sites(self):
    #     pass

    # # def to_tensor(self):
    # #     return mps_to_tensor(self.factors)

    # # def to_vec(self):
    # #     return mps_to_tensor(self.factors)

    # def left_canonicalize(self):
    #     pass

    # def right_canonicalize(self):
    #     pass

    # def left_to_right(self):
    #     pass

    # def site_canonicalize(self):
    #     pass


class MPO:
    def __init__(self, factors, homogenous=False):
        """
        class for matrix product operators.

        Parameters
        ----------
        mps : list
            list of 4-tensors. [chi1, d, chi2, d]
        chi_max:
            maximum bond order used in compress. Default None.

        Returns
        -------
        None.

        """
        self.factors = self.data = factors
        self.nsites = self.L = len(factors)
        self.nbonds = self.L - 1
        # self.chi_max = chi_max


        if homogenous:
            self.dims = [mps[0].shape[1], ] * self.nsites
        else:
            self.dims = [t.shape[1] for t in factors] # physical dims of each site

        # self._mpo = None

    def bond_orders(self):
        return [t.shape[0] for t in self.factors] # bond orders

    def dot(self, mps, rank):
        # apply MPO to MPS followed by a compression

        factors = apply_mpo(self.factors, mps.factors, rank)

        return MPS(factors)



def LeftCanonical(M):
    '''
        Function that takes an MPS 'M' as input (order of legs: left-bottom-right) and returns a copy of it that is
            transformed into left canonical form and normalized.

    Src:
        https://github.com/GCatarina/DMRG_MPS_didactic/blob/main/DMRG-MPS_implementation.ipynb
    '''
    Mcopy = M.copy() #create copy of M

    N = len(Mcopy) #nr of sites

    for l in range(N):
        # reshape
        Taux = Mcopy[l]
        Taux = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))

        # SVD
        U,S,Vdag = np.linalg.svd(Taux,full_matrices=False)
        '''
            Note: full_matrices=False leads to a trivial truncation of the matrices (thin SVD).
        '''

        # update M[l]
        Mcopy[l] = np.reshape(U,(np.shape(Mcopy[l])[0],np.shape(Mcopy[l])[1],np.shape(U)[1]))

        # update M[l+1]
        SVdag = np.matmul(np.diag(S),Vdag)
        if l < N-1:
            Mcopy[l+1] = np.einsum('ij,jkl',SVdag,Mcopy[l+1])
        else:
            '''
                Note: in the last site (l=N-1), S*Vdag is a number that determines the normalization of the MPS.
                    We discard this number, which corresponds to normalizing the MPS.
            '''

    return Mcopy


def RightCanonical(M):
    '''
        Function that takes an MPS 'M' as input (order of legs: left-bottom-right) and returns a copy of it that is
            transformed into right canonical form and normalized.
    '''
    Mcopy = M.copy() #create copy of M

    N = len(Mcopy) #nr of sites

    for l in range(N-1,-1,-1):
        # reshape
        Taux = Mcopy[l]
        Taux = np.reshape(Taux,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))

        # SVD
        U,S,Vdag = np.linalg.svd(Taux,full_matrices=False)

        # update M[l]
        Mcopy[l] = np.reshape(Vdag,(np.shape(Vdag)[0],np.shape(Mcopy[l])[1],np.shape(Mcopy[l])[2]))

        # update M[l-1]
        US = np.matmul(U,np.diag(S))
        if l > 0:
            Mcopy[l-1] = np.einsum('ijk,kl',Mcopy[l-1],US)
        else:
            '''
                Note: in the first site (l=0), U*S is a number that determines the normalization of the MPS. We
                    discard this number, which corresponds to normalizing the MPS.
            '''

    return Mcopy

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



'''
    Function that makes the following contractions (numbers denote leg order):

         /--3--**--1--Mt--3--
         |             |
         |             2
         |             |
         |             *
         |             *
         |             |
         |             4                 /--3--
         |             |                 |
        Tl--2--**--1---O--3--     =     Tf--2--
         |             |                 |
         |             2                 \--1--
         |             |
         |             *
         |             *
         |             |
         |             2
         |             |
         \--1--**--3--Mb--1--
'''
def ZipperLeft(Tl,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mb,Tl)
    Taux = np.einsum('ijkl,kjmn',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mt)

    return Tf

def expect(mpo, mps):
    # <GS| O |GS> , closing the zipper from the left
    Taux = np.ones((1,1,1))
    for l in range(N):
        Taux = ZipperLeft(Taux, mps[l].conj().T, mpo[l], mps[l])
    print('<GS| H |GS> = ', Taux[0,0,0])
    # print('analytical result = ', -2*(N-1)/3)
    return Taux[0, 0, 0]

'''
    Function that makes the following contractions (numbers denote leg order):

         --1--Mt--3--**--1--\
               |            |
               2            |
               |            |
               *            |
               *            |
               |            |
               4            |            --1--\
               |            |                 |
         --1---O--3--**--2--Tr     =     --2--Tf
               |            |                 |
               2            |            --3--/
               |            |
               *            |
               *            |
               |            |
               2            |
               |            |
         --3--Mb--1--**--3--/
'''
def ZipperRight(Tr,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mt,Tr)
    Taux = np.einsum('ijkl,mnkj',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mb)

    return Tf

def expect_zipper_right(mpo, mps):
    # <GS| H |GS> for AKLT model, closing the zipper from the right
    Taux = np.ones((1,1,1))
    for l in range(N-1,-1,-1):
        Taux = ZipperRight(Taux, mps[l].conj().T, mpo[l], mps[l])
    # print('<GS| H |GS> = ', Taux[0,0,0])
    # print('analytical result = ', -2*(N-1)/3)

    return Taux[0,0,0]

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


def apply_mpo(w_list, B_list, chi_max):
    """
    Apply the MPO to an MPS.

    MPS in :math:`[\alpha_l, d_l, \alpha_{l+1}]`

    MPO in :math:`[\alpha_l, d_l, \alpha_{l+1}, d_l]`

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

    # d = B_list[0].shape[1] # size of local space
    # D = w_list[0].shape[1]

    L = len(B_list) # nsites

    chi1, d, chi2 = B_list[0].shape # left and right bond dims
    b1, d, b2, d = w_list[0].shape # left and right bond dims

    B = np.tensordot(w_list[0], B_list[0], axes=(3,1))
    B = np.transpose(B,(3,0,1,4,2))

    B = np.reshape(B,(chi1*b1, d, chi2*b2))

    B_list[0] = B

    for i_site in range(1,L-1):
        chi1, d, chi2 = B_list[i_site].shape
        b1, _, b2, _ = w_list[i_site].shape # left and right bond dims

        B = np.tensordot(w_list[i_site], B_list[i_site], axes=(3,1))
        B = np.reshape(np.transpose(B,(3,0,1,4,2)),(chi1*b1, d, chi2*b2))

        B_list[i_site] = B
        # s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)

    # last site
    chi1, d, chi2 = B_list[L-1].shape
    b1, _, b2, _ = w_list[L-1].shape # left and right bond dims

    B = np.tensordot(w_list[L-1], B_list[L-1], axes=(3,1))
    B = np.reshape(np.transpose(B,(3,0,1,4,2)),(chi1*b1, d, chi2*b2))

    # s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B

    return compress(B_list, chi_max)


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


def tebd(B_list, s_list, U_list, chi_max):
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
    # compute the overlap between two factors
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