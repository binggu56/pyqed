#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:41:46 2024

#####################################################

#  main DMRG module using MPS/MPO representations

ground state optimization

time-evolving block decimation

# Ian McCulloch August 2017                         #
#####################################################


@author: Bing Gu
"""



import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.sparse as sparse
import math
from copy import deepcopy
from scipy.sparse.linalg import eigsh #Lanczos diagonalization for hermitian matrices

# from pyqed.mps.mps import LeftCanonical, RightCanonical, ZipperLeft, ZipperRight
from pyqed.mps.decompose import decompose, compress
from scipy.linalg import expm, block_diag
import warnings


class MPS:
    def __init__(self, Bs, Ss=None, homogenous=True, bc='finite', form="B"):
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

    def get_bond_dimensions(self):
        """
        Return bond dimensions.
        """
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

    def __add__(self, other):
        pass

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




class Site(object):
    """A general single site

    You use this class to create a single site. The site comes empty (i.e.
    with no operators included), but for th identity operator. You should
    add operators you need to make you site up.

    Parameters
    ----------
    dim : an int
	Size of the Hilbert space. The dimension must be at least 1. A site of
        dim = 1  represents the vaccum (or something strange like that, it's
        used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the site.

    Examples
    --------
    >>> from dmrg101.core.sites import Site
    >>> brand_new_site = Site(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_site.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_site.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
        """
        Creates an empty site of dimension dim.

        	Raises
        	------
        	DMRGException
        	    if `dim` < 1.

        	Notes
        	-----
        	Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
        	is added to the `self.operators` dictionary.
        """
        if dim < 1:
            raise DMRGException("Site dim must be at least 1")
        super(Site, self).__init__()
        self.dim = dim
        self.operators = { "id" : np.eye(self.dim, self.dim) }

    def add_operator(self, operator_name):
        #  """
        # Adds an operator to the site.

        #   Parameters
       	# ----------
        #    	operator_name : string
       	#     The operator name.

       	# Raises
       	# ------
       	# DMRGException
       	#     if `operator_name` is already in the dict.

       	# Notes
       	# -----
       	# Postcond:

        #       - `self.operators` has one item more, and
        #       - the newly created operator is a (`self.dim`, `self.dim`)
        #         matrix of full of zeros.

       	# Examples
       	# --------
       	# >>> from dmrg101.core.sites import Site
       	# >>> new_site = Site(2)
       	# >>> print new_site.operators.keys()
       	# ['id']
       	# >>> new_site.add_operator('s_z')
       	# >>> print new_site.operators.keys()
       	# ['s_z', 'id']
       	# >>> # note that the newly created op has all zeros
       	# >>> print new_site.operators['s_z']
       	# [[ 0.  0.]
        # 	 [ 0.  0.]]
        # """
        if str(operator_name) in self.operators.keys():

        # if str(operator_name) in self.operators.keys():
            raise DMRGException("Operator name exists already")
        else:
            self.operators[str(operator_name)] = np.zeros((self.dim, self.dim))

"""Exception class for the DMRG code
"""
class DMRGException(Exception):
    """A base exception for the DMRG code

    Parameters
    ----------
    msg : a string
        A message explaining the error
    """
    def __init__(self, msg):
        super(DMRGException, self).__init__()
        self.msg = msg

    def __srt__(self, msg):
        	return repr(self.msg)

class Block(Site):
    """A block.

    That is the representation of the Hilbert space and operators of a
    direct product of single site's Hilbert space and operators, that have
    been truncated.

    You use this class to create the two blocks (one for the left, one for
    the right) needed in the DMRG algorithm. The block comes empty.

    Parameters
    ----------
    dim : an int.
	Size of the Hilbert space. The dimension must be at least 1. A
	block of dim = 1  represents the vaccum (or something strange like
	that, it's used for demo purposes mostly.)
    operators : a dictionary of string and numpy array (with ndim = 2).
	Operators for the block.

    Examples
    --------
    >>> from dmrg101.core.block import Block
    >>> brand_new_block = Block(2)
    >>> # the Hilbert space has dimension 2
    >>> print brand_new_block.dim
    2
    >>> # the only operator is the identity
    >>> print brand_new_block.operators
    {'id': array([[ 1.,  0.],
           [ 0.,  1.]])}
    """
    def __init__(self, dim):
        	"""Creates an empty block of dimension dim.

        	Raises
        	------
        	DMRGException
        	     if `dim` < 1.

        	Notes
        	-----
        	Postcond : The identity operator (ones in the diagonal, zeros elsewhere)
        	is added to the `self.operators` dictionary. A full of zeros block
        	Hamiltonian operator is added to the list.
        	"""
        	super(Block, self).__init__(dim)

class PauliSite(Site):
    """
    A site for spin 1/2 models.

    You use this site for models where the single sites are spin
    one-half sites. The Hilbert space is ordered such as the first state
    is the spin down, and the second state is the spin up. Therefore e.g.
    you have the following relation between operator matrix elements:

    .. math::

        \langle \downarrow | A | uparrow \rangle = A_{0,1}

    Notes
    -----
    Postcond: The site has already built-in the spin operators for s_z, s_p, s_m.

    Examples
    --------
    >>> from dmrg101.core.sites import PauliSite
    >>> pauli_site = PauliSite()
    >>> # check all it's what you expected
    >>> print pauli_site.dim
    2
    >>> print pauli_site.operators.keys()
    ['s_p', 's_z', 's_m', 'id']
    >>> print pauli_site.operators['s_z']
    [[-1.  0.]
      [ 0.  1.]]
    >>> print pauli_site.operators['s_x']
    [[ 0.  1.]
      [ 1.  0.]]
    """
    def __init__(self):
        """
        Creates the spin one-half site with Pauli matrices.

 	  Notes
 	  -----
 	  Postcond : the dimension is set to 2, and the Pauli matrices
 	  are added as operators.

        """
        super(PauliSite, self).__init__(2)
	# add the operators
        self.add_operator("s_z")
        self.add_operator("s_x")
        self.add_operator("s_m")

	# for clarity
        s_z = self.operators["s_z"]
        s_x = self.operators["s_x"]
        s_m = self.operators["s_m"]

	# set the matrix elements different from zero to the right values
        s_z[0, 0] = -1.0
        s_z[1, 1] = 1.0
        s_x[0, 1] = 1.0
        s_x[1, 0] = 1.0
        s_m[0, 1] = 1.0





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
        factors : list
            list of 4-tensors of dimension. [chi1, d, chi2, d]
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

    def __matmul__(self, other):
        """
        define product of two MPOs

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(other, MPO):
            return product_MPO(self.factors, other.factors)

        elif isinstance(other, MPS):
            return apply_mpo(self.factors, other.factors)


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

    return B
    # return compress(B_list, chi_max)

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


# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual vonds

# MPO W-matrix is a 4-index tensor, W[s,t,i,j]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds

## initial E and F matrices for the left and right vacuum states
def initial_E(W):
    E = np.zeros((W.shape[0],1,1))
    E[0] = 1
    return E

def initial_F(W):
    F = np.zeros((W.shape[1],1,1))
    F[-1] = 1
    return F


def contract_from_right(W, A, F, B):
    """
    ## tensor contraction from the right hand side
    ##  -+     -A--+
    ##   |      |  |
    ##  -F' =  -W--F
    ##   |      |  |
    ##  -+     -B--+

    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    #return np.einsum("abst,sij,bjl,tkl->aik",W,A,F,B, optimize=True)

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    Temp = np.einsum("sij,bjl->sbil", A, F)
    Temp = np.einsum("sbil,abst->tail", Temp, W)
    return np.einsum("tail,tkl->aik", Temp, B)


def contract_from_left(W, A, E, B):
    """
    ## tensor contraction from the left hand side
    ## +-    +--A-
    ## |     |  |
    ## E' =  E--F-
    ## |     |  |
    ## +-    +--B-

    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    # return np.einsum("abst,sij,aik,tkl->bjl",W,A,E,B, optimize=True)

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    Temp = np.einsum("sij,aik->sajk", A, E)
    Temp = np.einsum("sajk,abst->tbjk", Temp, W)
    return np.einsum("tbjk,tkl->bjl", Temp, B)


def construct_F(Alist, MPO, Blist):
    """
    # construct the initial E and F matrices.
    # we choose to start from the left hand side, so the initial E matrix
    # is zero, the initial F matrices cover the complete chain

    Parameters
    ----------
    Alist : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    Blist : TYPE
        DESCRIPTION.

    Returns
    -------
    F : TYPE
        DESCRIPTION.

    """
    F = [initial_F(MPO[-1])]

    for i in range(len(MPO)-1, 0, -1):
        F.append(contract_from_right(MPO[i], Alist[i], F[-1], Blist[i]))
    return F

def construct_E(Alist, MPO, Blist):
    return [initial_E(MPO[0])]


def coarse_grain_MPO(W, X):
    """
    # 2-1 coarse-graining of two site MPO into one site
    #  |     |  |
    # -R- = -W--X-
    #  |     |  |

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("abst,bcuv->acsutv",W,X),
                      [W.shape[0], X.shape[1],
                       W.shape[2]*X.shape[2],
                       W.shape[3]*X.shape[3]])


def product_W(W, X):
    """
    # 'vertical' product of MPO W-matrices
    #        |
    #  |    -W-
    # -R- =  |
    #  |    -X-
    #        |

    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("abst,cdtu->acbdsu", W, X), [W.shape[0]*X.shape[0],
                                                             W.shape[1]*X.shape[1],
                                                             W.shape[2],X.shape[3]])


def product_MPO(M1, M2):
    assert len(M1) == len(M2)
    Result = []
    for i in range(0, len(M1)):
        Result.append(product_W(M1[i], M2[i]))
    return Result



def coarse_grain_MPS(A,B):
    """
    # 2-1 coarse-graining of two-site MPS into one site
    #   |     |  |
    #  -R- = -A--B-

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.reshape(np.einsum("sij,tjk->stik",A,B),
                      [A.shape[0]*B.shape[0], A.shape[1], B.shape[2]])

def fine_grain_MPS(A, dims):
    assert A.shape[0] == dims[0] * dims[1]
    Theta = np.transpose(np.reshape(A, dims + [A.shape[1], A.shape[2]]),
                         (0,2,1,3))
    M = np.reshape(Theta, (dims[0]*A.shape[1], dims[1]*A.shape[2]))
    U, S, V = np.linalg.svd(M, full_matrices=0)
    U = np.reshape(U, (dims[0], A.shape[1], -1))
    V = np.transpose(np.reshape(V, (-1, dims[1], A.shape[2])), (1,0,2))
    # assert U is left-orthogonal
    # assert V is right-orthogonal
    #print(np.dot(V[0],np.transpose(V[0])) + np.dot(V[1],np.transpose(V[1])))
    return U, S, V

def truncate_SVD(U, S, V, m):
    """
    # truncate the matrices from an SVD to at most m states

    Parameters
    ----------
    U : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    U : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    trunc : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    """
    m = min(len(S), m)
    trunc = np.sum(S[m:])
    S = S[0:m]
    U = U[:,:,0:m]
    V = V[:,0:m,:]
    return U,S,V,trunc,m

# Functor to evaluate the Hamiltonian matrix-vector multiply
#        +--A--+
#        |  |  |
# -R- =  E--W--F
#  |     |  |  |
#        +-   -+
class HamiltonianMultiply(sparse.linalg.LinearOperator):
    def __init__(self, E, W, F):
        self.E = E
        self.W = W
        self.F = F
        self.dtype = np.dtype('d')
        self.req_shape = [W.shape[2], E.shape[1], F.shape[2]]
        self.size = self.req_shape[0]*self.req_shape[1]*self.req_shape[2]
        self.shape = [self.size, self.size]

    def _matvec(self, A):
        # the einsum function doesn't appear to optimize the contractions properly,
        # so we split it into individual summations in the optimal order
        #R = np.einsum("aij,sik,abst,bkl->tjl",self.E,np.reshape(A, self.req_shape),
        #              self.W,self.F, optimize=True)
        R = np.einsum("aij,sik->ajsk", self.E, np.reshape(A, self.req_shape))
        R = np.einsum("ajsk,abst->bjtk", R, self.W)
        R = np.einsum("bjtk,bkl->tjl", R, self.F)
        return np.reshape(R, -1)

## optimize a single site given the MPO matrix W, and tensors E,F
def optimize_site(A, W, E, F, tol=1E-8):
    H = HamiltonianMultiply(E,W,F)
    # we choose tol=1E-8 here, which is OK for small calculations.
    # to bemore robust, we should take the tol -> 0 towards the end
    # of the calculation.
    E, V = sparse.linalg.eigsh(H,1,v0=A,which='SA', tol=tol)
    return (E[0],np.reshape(V[:,0], H.req_shape))


def optimize_two_sites(A, B, W1, W2, E, F, m, dir):
    """
    two-site optimization of MPS A,B with respect to MPO W1,W2 and
    environment tensors E,F
    dir = 'left' or 'right' for a left-moving or right-moving sweep

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    W1 : TYPE
        DESCRIPTION.
    W2 : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    dir : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    trunc : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    """
    W = coarse_grain_MPO(W1,W2)
    AA = coarse_grain_MPS(A,B)
    H = HamiltonianMultiply(E,W,F)
    E,V = sparse.linalg.eigsh(H,1,v0=AA,which='SA')
    AA = np.reshape(V[:,0], H.req_shape)
    A,S,B = fine_grain_MPS(AA, [A.shape[0], B.shape[0]])
    A,S,B,trunc,m = truncate_SVD(A,S,B,m)
    if (dir == 'right'):
        B = np.einsum("ij,sjk->sik", np.diag(S), B)
    else:
        assert dir == 'left'
        A = np.einsum("sij,jk->sik", A, np.diag(S))
    return E[0], A, B, trunc, m

def two_site_dmrg(MPS, MPO, m, sweeps=50, conv=1e-6):
    """
    Driver function to perform sweeps of 2-site DMRG


    Parameters
    ----------
    MPS : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    sweeps : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    MPS : TYPE
        DESCRIPTION.

    """

    E = construct_E(MPS, MPO, MPS)
    F = construct_F(MPS, MPO, MPS)
    F.pop()

    Eold = expect(MPS, MPO, MPS)

    converged = False

    for sweep in range(0, int(sweeps/2)):

        for i in range(0, len(MPS)-2): # forward
            Energy,MPS[i],MPS[i+1],trunc,states = optimize_two_sites(MPS[i],MPS[i+1],
                                                                     MPO[i],MPO[i+1],
                                                                     E[-1], F[-1], m, 'right')
            print("Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2,i,i+1, Energy, states, trunc))

            E.append(contract_from_left(MPO[i], MPS[i], E[-1], MPS[i]))
            F.pop();

        if abs(Energy - Eold) < conv:
            print("DMRG Converged at sweep {}. \n Total energy = {}".format(sweep, Energy))
            converged = True
            break
        else:
            Eold = Energy

        for i in range(len(MPS)-2, 0, -1): # backward

            Energy,MPS[i],MPS[i+1],trunc,states = optimize_two_sites(MPS[i],MPS[i+1],
                                                                     MPO[i],MPO[i+1],
                                                                     E[-1], F[-1], m, 'left')

            print("Sweep {} Sites {},{}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2+1,i,i+1, Energy, states, trunc))

            F.append(contract_from_right(MPO[i+1], MPS[i+1], F[-1], MPS[i+1]))
            E.pop();

        if abs(Energy - Eold) < conv:
            print("DMRG Converged at sweep {}. \n Total energy = {}".format(sweep, Energy))
            converged = True
            break
        else:
            Eold = Energy

    if not converged:
        print("DMRG not converged. Try increasing nsweep or a better initial guess.")

    return Energy, MPS


def expect(bra, MPO, ket=None):
    """
    Evaluate the expectation value of an MPO on a given MPS
    .. math::

         <A|MPO|B>

    Parameters
    ----------
    AList : TYPE
        DESCRIPTION.
    MPO : TYPE
        DESCRIPTION.
    BList : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    AList = bra
    BList = ket

    if ket is None:
        ket = bra

    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]






class DMRG:
    """
    ground state finite DMRG in MPO/MPS framework
    """
    def __init__(self, H, D, nsweeps=None, init_guess=None, opt='2site'):
        """


        Parameters
        ----------
        H : TYPE
            MPO of H.
        D : TYPE
            maximum bond dimension.
        nsweeps : TYPE, optional
            DESCRIPTION. The default is None.
        init_guess : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.H = H
        self.D = D
        self.nsweeps = nsweeps
        self.opt = opt

        self.init_guess = init_guess
        self.mps = None
        self.e_tot = None

        self.ground_state = None


    def run(self):

        if self.init_guess is None:
            raise ValueError('Invalid initial guess.')

        if self.opt == '1site':

            fDMRG_1site_GS_OBC(self.H, self.D, self.nsweeps)

        else:
            self.e_tot, self.ground_state = two_site_dmrg(self.init_guess, self.H, self.D, self.nsweeps)

        return self

    def expect(self, e_ops):
        """
        Compute expectation value of ground states

        Parameters
        ----------
        e_ops : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """

        psi = self.ground_state

        return [expect(psi, e_op) for e_op in e_ops]

    def make_rdm(self):
        # \gamma_{ij} = < 0| c_j^\dagger c_i | 0 >
        pass

    def make_rdm2(self):
        pass

def autoMPO(h1e, eri):
    """
    write the Hamiltonian into the MPO form

    .. math::

        H = \sum_{i,j} h_{ij} E_{ij} + \sum_{i < j} v_{ij} n_i n_j

        E_{ij} = \sum_\sigma c_{i\sigma}^\dagger c_{j\sigma}

    Parameters
    ----------
    h1e : TYPE
        one-electron core Hamiltonian.
    eri : TYPE
        electron-repulsion integral

    Returns
    -------
    None.

    """
    pass


class TEBD(DMRG):

    def run(self, psi0):
        return tebd(psi0, self.U, chi_max=self.D)


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


class TDVP(DMRG):
    pass



def fDMRG_1site_GS_OBC(H,D,Nsweeps):
    '''
    Function that implements finite-system DMRG (one-site update version) to obtain the ground state of an input
            Hamiltonian MPO (order of legs: left-bottom-right-top), 'H', that represents a system with open boundary
            conditions.

    Notes:
            - the outputs are the ground state energy at every step of the algorithm, 'E_list', and the ground state
                MPS (order of legs: left-bottom-right) at the final step, 'M'.
            - the maximum bond dimension allowed for the ground state MPS is an input, 'D'.
            - the number of sweeps is an input, 'Nsweeps'.
    '''
    N = len(H) #nr of sites

    # random MPS (left-bottom-right)
    M = []
    M.append(np.random.rand(1, np.shape(H[0])[3],D))

    for l in range(1,N-1):
        M.append(np.random.rand(D,np.shape(H[l])[3],D))
    M.append(np.random.rand(D,np.shape(H[N-1])[3],1))

    ## normalized MPS in right canonical form
    # M = LeftCanonical(M)
    M = RightCanonical(M)

    # Hzip
    '''
        Every step of the finite-system DMRG consists in optimizing a local tensor M[l] of an MPS in site
            canonical form. The value of l is sweeped back and forth between 0 and N-1.

        For a given l, we define Hzip as a list with N+2 elements where:

            - Hzip[0] = Hzip[N+1] = np.ones((1,1,1))

            - Hzip[it] =

                /--------------M[it-1]--3--
                |             \|
                |              |
                |              |
                Hzip[it-1]-----H[it-1]--2--          for it = 1, 2, ..., l
                |              |
                |              |
                |             /|
                \--------------M[it-1]^†--1--

            - Hzip[it] =

                --1--M[it-1]-----\
                     |/          |
                     |           |
                     |           |
                --2--H[it-1]-----Hzip[it+1]          for it = l+1, l+2, ..., N
                     |           |
                     |           |
                     |\          |
                --3--M[it-1]^†---/

        Here, we initialize Hzip considering l=0 (note that this is consistent with starting with a random MPS in
            right canonical form). Consistently, we will start the DMRG routine with a right sweep.
    '''
    Hzip = [np.ones((1,1,1)) for it in range(N+2)]
    for l in range(N-1,-1,-1):
        Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    # DMRG routine
    E_list = []
    for itsweeps in range(Nsweeps):
        ## right sweep
        for l in range(N):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                    np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            '''
                Note: for performance purposes, we initialize Lanczos with the previous version of the local
                    tensor M[l].
            '''
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the right sweep, the local tensor M[l] obtained from Lanczos has to be left normalized.
                    This is achieved by SVD. The remaining S*Vdag is contracted with M[l+1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(U,(np.shape(Taux)[0],np.shape(Taux)[1],np.shape(U)[1]))
            if l < N-1:
                M[l+1] = np.einsum('ij,jkl',np.matmul(np.diag(S),Vdag),M[l+1])

            ### update Hzip
            Hzip[l+1] = ZipperLeft(Hzip[l],M[l].conj().T,H[l],M[l])

        ## left sweep
        for l in range(N-1,-1,-1):
            ### H matrix
            Taux = np.einsum('ijk,jlmn',Hzip[l],H[l])
            Taux = np.einsum('ijklm,nlo',Taux,Hzip[l+2])
            Taux = np.transpose(Taux,(0,2,5,1,3,4))
            Hmat = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1]*np.shape(Taux)[2],
                                   np.shape(Taux)[3]*np.shape(Taux)[4]*np.shape(Taux)[5]))

            ### Lanczos diagonalization of H matrix (lowest energy eigenvalue)
            val,vec = eigsh(Hmat, k=1, which='SA', v0=M[l])
            E_list.append(val[0])

            ### update M[l]
            '''
                Note: in the left sweep, the local tensor M[l] obtained from Lanczos has to be right normalized.
                    This is achieved by SVD. The remaining U*S is contracted with M[l-1].
            '''
            Taux2 = np.reshape(vec,(np.shape(Taux)[0],np.shape(Taux)[1]*np.shape(Taux)[2]))
            U,S,Vdag = np.linalg.svd(Taux2,full_matrices=False)
            M[l] = np.reshape(Vdag,(np.shape(Vdag)[0],np.shape(Taux)[1],np.shape(Taux)[2]))
            if l > 0:
                M[l-1] = np.einsum('ijk,kl',M[l-1],np.matmul(U,np.diag(S)))

            ### update Hzip
            Hzip[l+1] = ZipperRight(Hzip[l+2],M[l].conj().T,H[l],M[l])

    return E_list,M


def SpinHalfFermionOperators(filling=1.):
    d = 4
    states = ['empty', 'up', 'down', 'full']
    # 0) Build the operators.
    Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
    Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)

    Nu = np.diag(Nu_diag)
    Nd = np.diag(Nd_diag)
    Ntot = np.diag(Nu_diag + Nd_diag)
    dN = np.diag(Nu_diag + Nd_diag - filling)
    NuNd = np.diag(Nu_diag * Nd_diag)
    JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
    JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
    JW = JWu * JWd  # (-1)^{Nu+Nd}


    Cu = np.zeros((d, d))
    Cu[0, 1] = Cu[2, 3] = 1
    Cdu = np.transpose(Cu)
    # For spin-down annihilation operator: include a Jordan-Wigner string JWu
    # this ensures that Cdu.Cd = - Cd.Cdu
    # c.f. the chapter on the Jordan-Wigner trafo in the userguide
    Cd_noJW = np.zeros((d, d))
    Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
    Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
    Cdd = np.transpose(Cd)

    # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
    # where S^gamma is the 2x2 matrix for spin-half
    Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
    Sp = np.dot(Cdu, Cd)
    Sm = np.dot(Cdd, Cu)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    ops = dict(JW=JW, JWu=JWu, JWd=JWd,
               Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
               Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
               Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable
    return ops



if __name__ == '__main__':

    ##
    ## Parameters for the DMRG simulation for spin-1/2 chain
    ## To apply to fermions, we only need to change the MPO if H
    ##

    d=2   # local bond dimension, 0=up, 1=down
    N=10 # number of sites

    ## initial state |+-+-+-+-+->

    InitialA1 = np.zeros((d,1,1))
    InitialA1[0,0,0] = 1
    InitialA2 = np.zeros((d,1,1))
    InitialA2[1,0,0] = 1

    MPS = [InitialA1, InitialA2] * int(N/2)

    ## Local operators
    I = np.identity(2)
    Z = np.zeros((2,2))
    Sz = np.array([[0.5,  0  ],
                 [0  , -0.5]])
    Sp = np.array([[0, 0],
                 [1, 0]])
    Sm = np.array([[0, 1],
                 [0, 0]])

    ## Hamiltonian MPO
    W = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z],
                  [Z,  Z,      Z,      Z,  Sz],
                  [Z,  Z,      Z,      Z,  Sm],
                  [Z,  Z,      Z,      Z,  Sp],
                  [Z,  Z,      Z,      Z,   I]])

    print(W.shape)

    # left-hand edge is 1x5 matrix
    Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z]])

    # right-hand edge is 5x1 matrix
    Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])

    # the complete MPO
    H = MPO = [Wfirst] + ([W] * (N-2)) + [Wlast]

    dmrg = DMRG(H, D=10, nsweeps=8)
    dmrg.init_guess = MPS
    dmrg.run()




    # # MPO for H^2, to calculate the variance
    # HamSquared = product_MPO(MPO, MPO)

    # 8 sweeps with m=10 states
    # two_site_dmrg(MPS, MPO, 10, 8)

# # energy and energy squared
# E_10 = Expectation(MPS, MPO, MPS);
# Esq_10 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=20 states
# two_site_dmrg(MPS, MPO, 20, 2)

# # energy and energy squared
# E_20 = Expectation(MPS, MPO, MPS);
# Esq_20 = Expectation(MPS, HamSquared, MPS);

# # 2 sweeps with m=30 states
# two_site_dmrg(MPS, MPO, 30, 2)

# # energy and energy squared
# E_30 = Expectation(MPS, MPO, MPS);
# Esq_30 = Expectation(MPS, HamSquared, MPS);

# Energy = Expectation(MPS, MPO, MPS)
# print("Final energy expectation value {}".format(Energy))

# # calculate the variance <(H-E)^2> = <H^2> - E^2

# print("m=10 variance = {:16.12f}".format(Esq_10 - E_10*E_10))
# print("m=20 variance = {:16.12f}".format(Esq_20 - E_20*E_20))
# print("m=30 variance = {:16.12f}".format(Esq_30 - E_30*E_30))