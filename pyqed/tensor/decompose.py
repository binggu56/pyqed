#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:00:26 2021

@author: bing
"""

import tensorly as tl
from tensorly import random, validate_tt_rank
from tensorly.tenalg import inner
from tensorly.decomposition import tensor_train

# tensor = random.random_tensor((10, 10, 10))
# # This will be a NumPy array by default

# from lime.phys import dag

import numpy as np
import warnings
from scipy.linalg import svd

import logging


def tensor_train(input_tensor, rank):
    """
    TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

        In left canonical form.

    Adapated from Tensorly package.
    
    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
              order-3 tensors of the TT decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape # list of phys dims
    
    n_dim = len(tensor_size)

    # this is not correct
    # if not isinstance(rank, list):
    #     rank = [rank] * n_dim

    unfolding = input_tensor
    factors = [None] * n_dim

    S0 = np.array([1.])
    Ss = [S0]
    
    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):

        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape

        current_rank = min(n_row, n_column, rank[k+1])
        U, S, V = truncated_svd(unfolding, current_rank)
        rank[k+1] = current_rank

        Ss.append(S.copy())
        # print('schmidt coeff', tl.norm(S))

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k+1]))


        logging.info("TT factor " + str(k) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        unfolding= tl.reshape(S, (-1, 1))*V
        # unfolding = V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = np.reshape(unfolding, (prev_rank, last_dim, 1))

    
    logging.info("TT factor " + str(n_dim-1) + " computed with shape " + str(factors[n_dim-1].shape))

    return factors, Ss

def truncated_svd(matrix, k=None, **kwargs):
    """Computes a truncated SVD on `matrix` using the backends's standard SVD
    Parameters

    Adapted from Tensorly.
    ----------
    matrix : 2D-array
    k : int, optional, default is None
        if specified, number of eigen[vectors-values] to return
    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    # Check that matrix is... a matrix!
    # if np.dims(matrix) != 2:
    #     raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
    #                      % np.dims(matrix))

    dim_1, dim_2 = np.shape(matrix)
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if k is None:
        k = max_dim

    if k > max_dim:
        warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                      'is larger than max(matrix.shape)={1}. Setting '
                      'n_eigenvecs to {1}'.format(k, max_dim))
        k = max_dim

    full_matrices = k > min_dim

    U, S, V = svd(matrix, full_matrices=full_matrices)
    U, S, V = U[:, :k], S[:k], V[:k, :]

    # VV^\dag = I, U^\dag U = I
    # print('U', dag(U).dot(U))
    # approx = U @ np.diag(S) @ V
    # print('svd norm', tl.norm(approx))

    return U, S, V

def tt_to_tensor(factors):
    """Returns the full tensor whose TT decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TT/Matrix-Product-State format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TT factors (TT-cores)

    Returns
    -------
    output_tensor : ndarray
                   tensor whose TT/MPS decomposition was given by 'factors'
    """
    if isinstance(factors, (float, int)): #0-order tensor
        return factors

    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)

def compress(B_list, chi_max):
    " Compress the MPS by reducing the bond dimension."
    # d = B_list[0].shape[0]
    L = len(B_list)
    s_list  = [None] * L
    # for p in [0,1]:

    for i_bond in np.arange(L-1):

        i1=i_bond
        i2=i_bond+1

        chi1, d1, _ = B_list[i1].shape
        _, d2, chi3 = B_list[i2].shape

        print(r'bond {}, dims, {} {} {} {}'.format(i_bond, chi1, d1, d2, chi3))

        # Construct theta matrix
        # C[chi1, i, j, chi3] = B1[chi1, i, chi2] B2[chi2, j, chi3]
        
        # C = np.tensordot(B_list[i1], B_list[i2],axes=1)
        C = np.einsum('aib, bjc -> aijc', B_list[i1], B_list[i2])
        
        theta = np.reshape(C, (chi1 * d1, d2*chi3))

        # theta = np.reshape(np.einsum('a, aijb->aijb', s_list[i1], C),\
        #                     (d1*chi1, d2*chi3))
        # C = np.reshape(C,(d1*chi1,d2*chi3))

        # C = theta.copy()

        # Schmidt decomposition X Y Z^T = theta
        X, Y, Z = svd(theta)
        # Z=Z.T # d2*chi3, chi2

        # W = np.dot(C,Z.T.conj())
        chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

        # Obtain the new values for B and l #
        invsq = np.sqrt(sum(Y[:chi2]**2))

        s_list[i2] = Y[:chi2]/invsq

        # B_list[i1] = np.reshape(W[:,:chi2],(chi1, d1, chi2))/invsq

        B_list[i1] = np.reshape(X[:,:chi2],(chi1, d1, chi2))

        B_list[i2] = np.reshape(np.diag(s_list[i2])@Z[:chi2,:],(chi2, d2, chi3))


    return B_list, s_list

# def mps_to_tensor(b):
#     return np.einsum('ib, bjc, ck->ijk', b[0][0,:,:], b[1], b[2][:,:,0])


if __name__ == '__main__':
    
    def pes(x):
        dim = len(x)
        v = 0 
        for d in range(dim):
            v += 0.5 * x[d]**2
        v += 0.1 * x[0] * x[1] + x[0]**4 * 0.2
        return v
    
    
    # a = np.random.randn(3, 3, 3)
    level = 4
    n = 2**level - 1 # number of grid points for each dim
    x = np.linspace(-6, 6, 2**level, endpoint=False)[1:]

    
    v = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v[i, j, k] = pes([x[i], x[j], x[k]])

    dt = 0.05
    v = np.exp(-1j * v * dt)                
    # print(np.einsum('ijk, ijk', a, a))
    
    # print(inner(a, a))
    # a = a/tl.norm(a)
    
    rank = 4
    
    As, Ss = tensor_train(v,rank=rank)
    b = As.copy()
    print('singular values', Ss[0],Ss[1], Ss[2])
    
    # b_to_tensor = np.einsum('ib, bjc, ck->ijk', b[0][0,:,:], b[1], b[2][:,:,0])
    b = tt_to_tensor(As)    
    print(b.shape)
    print(b[:, 0, 0]-v[:, 0, 0])
    
    # print('norm a', tl.norm(a))
    
    # print('norm a_tt', tl.norm(b_to_tensor))
    
    # b2 = my_tensor_train(a, rank=4)
    
    # print(validate_tt_rank(tl.shape(a), rank=rank))
    
    # tl.set_backend('numpy')
    # c = tensor_train(a, rank=rank)
    
    
    
    # print(b == c)
    # print(len(b))
    # for j in range(len(b)):
    #     print(b[j].shape)
    #     print(c[j].shape)
    
    
    b_compressed, slist = compress(As, chi_max=4)
    # a_compressed = mps_to_tensor(b_compressed)
    
    # tmp = np.tensordot(b_compressed[1],  b_compressed[2][:,:,0], axes=1) # aib, bj -> aij
    # tmp = b_compressed[1]
    # r = np.einsum('aib, cib ->ac', tmp, tmp)
    print(tt_to_tensor(b_compressed)[:, 0, 0] - v[:, 0, 0])

# print(b_compressed[0])
# print(As[0]
#       )