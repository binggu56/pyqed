#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:05:39 2023

@author: bing


"""

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Routines for simultaneous diagonalization
Arun Chaganty <arunchaganty@gmail.com>

NOT TESTED (ERROR EXIST)
"""

import numpy as np
from numpy import zeros, eye, diag
from numpy.linalg import norm

def givens_rotate( A, i, j, c, s ):
    """
    Rotate A along axis (i,j) by c and s
    """
    Ai, Aj = A[i,:], A[j,:]
    A[i,:], A[j,:] = c * Ai + s * Aj, c * Aj - s * Ai 

    return A

def givens_double_rotate( A, i, j, c, s ):
    """
    Rotate A along axis (i,j) by c and s
    """
    Ai, Aj = A[i,:], A[j,:]
    A[i,:], A[j,:] = c * Ai + s * Aj, c * Aj - s * Ai 
    A_i, A_j = A[:,i], A[:,j]
    A[:,i], A[:,j] = c * A_i + s * A_j, c * A_j - s * A_i 

    return A

def jacobi_angles( *Ms, **kwargs ):
    r"""
    Simultaneously diagonalize using Jacobi angles
    
    @article{SC-siam,
       HTML =   "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
       author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
       journal = "{SIAM} J. Mat. Anal. Appl.",
       title = "Jacobi angles for simultaneous diagonalization",
       pages = "161--164",
       volume = "17",
       number = "1",
       month = jan,
       year = {1995}}

    (a) Compute Givens rotations for every pair of indices (i,j) i < j 
            - from eigenvectors of G = gg'; g = A_ij - A_ji, A_ij + A_ji
            - Compute c, s as \sqrt{x+r/2r}, y/\sqrt{2r(x+r)}
    (b) Update matrices by multiplying by the givens rotation R(i,j,c,s)
    (c) Repeat (a) until stopping criterion: sin theta < threshold for all ij pairs
    """

    assert len(Ms) > 0
    m, n = Ms[0].shape
    assert m == n

    sweeps = kwargs.get('sweeps', 500)
    threshold = kwargs.get('eps', 1e-8)
    rank = kwargs.get('rank', m)

    R = eye(m)

    for _ in range(sweeps):
        done = True
        for i in range(rank):
            for j in range(i+1, m):
                G = zeros((2,2))
                for M in Ms:
                    g = np.array([ M[i,i] - M[j,j], M[i,j] + M[j,i] ])
                    G += np.outer(g,g) / len(Ms)
                # Compute the eigenvector directly
                t_on, t_off = G[0,0] - G[1,1], G[0,1] + G[1,0]
                theta = 0.5 * np.arctan2( t_off, t_on + np.sqrt( t_on*t_on + t_off * t_off) )
                c, s = np.cos(theta), np.sin(theta)

                if abs(s) > threshold:
                    done = False
                    # Update the matrices and V
                    for M in Ms:
                        givens_double_rotate(M, i, j, c, s)
                        #assert M[i,i] > M[j, j]
                    R = givens_rotate(R, i, j, c, s)

        if done:
            break
    R = R.T

    L = np.zeros((m, len(Ms)))
    err = 0
    for i, M in enumerate(Ms):
        # The off-diagonal elements of M should be 0
        L[:,i] = diag(M)
        err += norm(M - diag(diag(M)))

    return R, L, err