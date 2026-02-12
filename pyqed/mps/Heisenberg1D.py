# -*- coding: utf-8 -*-

#####################################################
# Simple DMRG example for the 1D Heisenberg model   #
# Ian McCulloch, August 2017    
# Bing Gu, Nov 2018                     #
# After S.R.White, Phys. Rev. Lett. 69, 2863 (1992) #
#####################################################

import numpy as np
import scipy
import scipy.sparse.linalg
import math

##
## Initial parameters
##


# number of iterations.  Final lattice size is 2*NIter + 2
NIter = 100

# Exact energy per site, for comparison
ExactEnergy = -math.log(2) + 0.25

print("  Iter  Size     Energy        BondEnergy   EnergyError   Truncation")

##
## local operators
##

I = np.mat(np.identity(2))
Sz = np.mat([[0.5,  0  ],
             [0  , -0.5]])
Sp = np.mat([[0, 0],
               [1, 0]])
Sm = np.mat([[0, 1],
             [0, 0]])

##
## Initial block operators.  At the start
## these represent a single site.  We assume
## reflection symmetry so we only need one block
## we can use on both the left and the right
##

BlockSz = Sz
BlockSp = Sp
BlockSm = Sm
BlockI = I
BlockH = np.zeros((2,2))  # Hamiltonian for 1-site system

Energy = -0.75        # initial energy for 2 sites
                      # (we start iterations from 4 sites)

##
## Begin main iterations
##

def infinite_system_algorithm(L, m):
    
    for i in range(0,NIter):
    
        ## Add a site to the block
        BlockH = np.kron(BlockH, I) + np.kron(BlockSz, Sz) + \
                 0.5 * (np.kron(BlockSp, Sm) + np.kron(BlockSm, Sp))
        BlockSz = np.kron(BlockI, Sz)
        BlockSp = np.kron(BlockI, Sp)
        BlockSm = np.kron(BlockI, Sm)
        BlockI = np.kron(BlockI, I)
    
        ## 'Superblock' Hamiltonian
    
        H_super = np.kron(BlockH, BlockI) + np.kron(BlockI, BlockH) + \
                  np.kron(BlockSz, BlockSz) + 0.5 * (np.kron(BlockSp, BlockSm) + \
                                                     np.kron(BlockSm, BlockSp))
    
        ## Diagonalize the Hamiltonian
    
        LastEnergy = Energy
        E, Psi = scipy.sparse.linalg.eigsh(H_super, k=1, which='SA')
        Energy = E[0]
        EnergyPerBond = (Energy - LastEnergy) / 2;
    
        ## form the reduced density matrix by reshaping Psi into a matrix
    
        Dim = BlockH.shape[0]
        PsiMatrix = np.mat(np.reshape(Psi, [Dim, Dim]))
        Rho = PsiMatrix * PsiMatrix.H
        
        ## Diagonalize the density matrix
        ## The eigenvalues are arranged in ascending order
        D, V = np.linalg.eigh(Rho)
    
        ## Construct the truncation operator, which is the projector
        ## onto the m largest eigenvalues of Rho
    
        T = np.mat(V[:, max(0,Dim-m):Dim])
        TruncationError = 1 - np.sum(D[max(0,Dim-m):Dim])
    
        print("{:6} {:6} {:16.8f} {:12.8f} {:12.8f} {:12.8f}"
              .format(i, 4+i*2, Energy, EnergyPerBond,
                      ExactEnergy-EnergyPerBond, TruncationError))
    
        ## Truncate the block operators
    
        BlockH = T.H * BlockH * T
        BlockSz = T.H * BlockSz * T
        BlockSp = T.H * BlockSp * T
        BlockSm = T.H * BlockSm * T
        BlockI = T.H * BlockI * T

# finish
if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    infinite_system_algorithm(L=100, m=10)