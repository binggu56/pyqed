#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:21:30 2026

@author: gugroup
"""

import numpy as np
from pyqed.mps import DMRG

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