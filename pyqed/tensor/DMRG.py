#####################################################
# Simple DMRG code using MPS/MPO representations    #
# Ian McCulloch August 2017                         #
#####################################################

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.sparse as sparse
import math
from copy import deepcopy

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

## tensor contraction from the right hand side
##  -+     -A--+
##   |      |  |
##  -F' =  -W--F
##   |      |  |
##  -+     -B--+
def contract_from_right(W, A, F, B):
    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    #return np.einsum("abst,sij,bjl,tkl->aik",W,A,F,B, optimize=True)
    Temp = np.einsum("sij,bjl->sbil", A, F)
    Temp = np.einsum("sbil,abst->tail", Temp, W)
    return np.einsum("tail,tkl->aik", Temp, B)

## tensor contraction from the left hand side
## +-    +--A-
## |     |  |
## E' =  E--F-
## |     |  |
## +-    +--B-
def contract_from_left(W, A, E, B):
    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    # return np.einsum("abst,sij,aik,tkl->bjl",W,A,E,B, optimize=True)
    Temp = np.einsum("sij,aik->sajk", A, E)
    Temp = np.einsum("sajk,abst->tbjk", Temp, W)
    return np.einsum("tbjk,tkl->bjl", Temp, B)

# construct the initial E and F matrices.
# we choose to start from the left hand side, so the initial E matrix
# is zero, the initial F matrices cover the complete chain
def construct_F(Alist, MPO, Blist):
    F = [initial_F(MPO[-1])]

    for i in range(len(MPO)-1, 0, -1):
        F.append(contract_from_right(MPO[i], Alist[i], F[-1], Blist[i]))
    return F

def construct_E(Alist, MPO, Blist):
    return [initial_E(MPO[0])]

# 2-1 coarse-graining of two site MPO into one site
#  |     |  |
# -R- = -W--X-
#  |     |  |
def coarse_grain_MPO(W, X):
    return np.reshape(np.einsum("abst,bcuv->acsutv",W,X),
                      [W.shape[0], X.shape[1],
                       W.shape[2]*X.shape[2],
                       W.shape[3]*X.shape[3]])

# 'vertical' product of MPO W-matrices
#        |
#  |    -W-
# -R- =  |
#  |    -X-
#        |
def product_W(W, X):
    return np.reshape(np.einsum("abst,cdtu->acbdsu", W, X), [W.shape[0]*X.shape[0],
                                                             W.shape[1]*X.shape[1],
                                                             W.shape[2],X.shape[3]])


def product_MPO(M1, M2):
    assert len(M1) == len(M2)
    Result = []
    for i in range(0, len(M1)):
        Result.append(product_W(M1[i], M2[i]))
    return Result


# 2-1 coarse-graining of two-site MPS into one site
#   |     |  |
#  -R- = -A--B-
def coarse_grain_MPS(A,B):
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

# truncate the matrices from an SVD to at most m states
def truncate_SVD(U, S, V, m):
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
def optimize_site(A, W, E, F):
    H = HamiltonianMultiply(E,W,F)
    # we choose tol=1E-8 here, which is OK for small calculations.
    # to bemore robust, we should take the tol -> 0 towards the end
    # of the calculation.
    E,V = sparse.linalg.eigsh(H,1,v0=A,which='SA', tol=1E-8)
    return (E[0],np.reshape(V[:,0], H.req_shape))

## two-site optimization of MPS A,B with respect to MPO W1,W2 and
## environment tensors E,F
## dir = 'left' or 'right' for a left-moving or right-moving sweep
def optimize_two_sites(A, B, W1, W2, E, F, m, dir):
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

## Driver function to perform sweeps of 2-site DMRG
def two_site_dmrg(MPS, MPO, m, sweeps):
    E = construct_E(MPS, MPO, MPS)
    F = construct_F(MPS, MPO, MPS)
    F.pop()
    for sweep in range(0,int(sweeps/2)):
        for i in range(0, len(MPS)-2):
            Energy,MPS[i],MPS[i+1],trunc,states = optimize_two_sites(MPS[i],MPS[i+1],
                                                                     MPO[i],MPO[i+1],
                                                                     E[-1], F[-1], m, 'right')
            print("Sweep {:} Sites {:},{:}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2,i,i+1, Energy, states, trunc))
            E.append(contract_from_left(MPO[i], MPS[i], E[-1], MPS[i]))
            F.pop();
        for i in range(len(MPS)-2, 0, -1):
            Energy,MPS[i],MPS[i+1],trunc,states = optimize_two_sites(MPS[i],MPS[i+1],
                                                                     MPO[i],MPO[i+1],
                                                                     E[-1], F[-1], m, 'left')
            print("Sweep {} Sites {},{}    Energy {:16.12f}    States {:4} Truncation {:16.12f}"
                     .format(sweep*2+1,i,i+1, Energy, states, trunc))
            F.append(contract_from_right(MPO[i+1], MPS[i+1], F[-1], MPS[i+1]))
            E.pop();
    return MPS

## Function the evaluate the expectation value of an MPO on a given MPS
## <A|MPO|B>
def Expectation(AList, MPO, BList):
    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]

##
## Parameters for the DMRG simulation
##

d=2   # local bond dimension, 0=up, 1=down
N=100 # number of sites

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

# left-hand edge is 1x5 matrix
Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm,   Z]])

# right-hand edge is 5x1 matrix
Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])

# the complete MPO
MPO = [Wfirst] + ([W] * (N-2)) + [Wlast]

# MPO for H^2, to calculate the variance
HamSquared = product_MPO(MPO, MPO)

# 8 sweeps with m=10 states
two_site_dmrg(MPS, MPO, 10, 8)

# energy and energy squared
E_10 = Expectation(MPS, MPO, MPS);
Esq_10 = Expectation(MPS, HamSquared, MPS); 

# 2 sweeps with m=20 states
two_site_dmrg(MPS, MPO, 20, 2)

# energy and energy squared
E_20 = Expectation(MPS, MPO, MPS);
Esq_20 = Expectation(MPS, HamSquared, MPS); 

# 2 sweeps with m=30 states
two_site_dmrg(MPS, MPO, 30, 2)

# energy and energy squared
E_30 = Expectation(MPS, MPO, MPS);
Esq_30 = Expectation(MPS, HamSquared, MPS); 

Energy = Expectation(MPS, MPO, MPS)
print("Final energy expectation value {}".format(Energy))

# calculate the variance <(H-E)^2> = <H^2> - E^2

print("m=10 variance = {:16.12f}".format(Esq_10 - E_10*E_10))
print("m=20 variance = {:16.12f}".format(Esq_20 - E_20*E_20))
print("m=30 variance = {:16.12f}".format(Esq_30 - E_30*E_30))