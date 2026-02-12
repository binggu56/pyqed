#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:06:50 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""


class LongRangeChain(FermiHubbard):
    def __init__(self, L):
        self.L = L # length

        self.H = None
        self.W = None

    def add_single_site(self, operators):
        """

        the MPO of the Hamiltonian for single-site operators

        .. math::

            H = \sum_i O_i

        Returns
        -------
        None.

        """

        ops = SpinHalfFermionOperators()
        # JWu = ops['JWu']
        # JWd = ops['JWd']
        JW = ops['JW']
        Cu = ops['Cu']
        Cdu = ops['Cdu']
        Cd = ops['Cd']
        Cdd = ops['Cdd']
        Sz = ops['Sz']
        Nu = ops['Nu']
        Nd = ops['Nd']
        Ntot = ops['Ntot']

        L = self.L

        U = self.U
        mu = self.mu

        self.I = I = np.eye(4)
        Z = np.zeros((4,4))

        W_first = np.array([[operators[0]], [I]])

        W = [W_first]
        for l in range(1, L-1):

            Wl = np.array([[I, Z],
                          [operators[l], I]])
            W.append(Wl)

        W_last = np.array([[I], [operators[-1]]])
        W += [W_last]

        self.W_first = W_first

        self.W = W
        self.W_last = W_last
        # print('w_first',np.shape(W_first))
        # print('w',np.shape(W))

        if self.L >= 3:
            mpo = [self.W_first] + ([self.W] * (self.L-2)) + [self.W_last]
            # result = mpo[0]
            # for i in range(1,self.L):
            #     result = coarse_gain_MPO(result,mpo[i])   # translate MPO form into exact form

        elif self.L == 2:
            mpo = [self.W_first] + [self.W_last]
            # result = coarse_gain_MPO(mpo[0],mpo[1])

        else:
            print("L should be more than 2")
            # result = -1
            mpo = -1

        self.h_mpo = MPO(mpo)
        return

    def add_exponentially_decaying_coupling(self, strength, lambda_decay, \
                                            op_i='Sz', op_j='Sz'):
        """

        an exponentially decaying matrix product operator (MPO) can be created

        .. math::

            H = \sum _{i,j}\text{strength} \times e^{-\lambda |i-j|} O_{i} \otimes O_{j}

        Parameters
        ----------
        strength : TYPE, optional
            DESCRIPTION. The default is strength_decay.
        lambda_decay : TYPE
            DESCRIPTION.
        op_i : TYPE, optional
            DESCRIPTION. The default is 'Sz'.
        op_j : TYPE, optional
            DESCRIPTION. The default is 'Sz'.

        Returns
        -------
        None.

        """
        pass
    
    def build_h_mpo(self):
        # exponential decomposition t, v
        
        # call self.add_expo...
        
        # call self.add_single_site...
        
        # call DMRG... 
        pass
        
    def transition_density_matrix(self):
        pass
