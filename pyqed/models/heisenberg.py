#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 22:30:16 2026

Heisenberg model 

@author: Shuoyi Hu, Bing Gu (gubing at westlake dot edu dot cn)
"""
from pyqed.mps import TDMPS, MPO, MPS
import numpy as np
import logging

class Heisenberg:
    def __init__(self, L, J=1):
        
        logging.info(f"Heisenberg Chain with {L} sites")

        self.L = L
        self.d = 2 # local dim
        self.J = J

        ####DO NOT MODIFY###
        self.H = None

    def build_H_mpo(self):
        """
        Constructs the Heisenberg Hamiltonian MPO for N sites.
        """
        N = self.L

        I = np.identity(2)
        Z = np.zeros((2, 2))
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 0], [1, 0]])
        Sm = np.array([[0, 1], [0, 0]])

        # Define the bulk MPO tensor W (5x5 matrix of operators)
        # Rows/Cols: I, Sz, Sp, Sm, Hamiltonian-accumulator
        W = np.array([[I, Sz, 0.5*Sp, 0.5*Sm, Z],
                    [Z, Z,  Z,      Z,      Sz],
                    [Z, Z,  Z,      Z,      Sm],
                    [Z, Z,  Z,      Z,      Sp],
                    [Z, Z,  Z,      Z,      I]])

        # Boundary vectors
        Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm, Z]]) # 1x5
        Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])  # 5x1

        # Construct full list
        H_factors = [Wfirst] + ([W] * (N - 2)) + [Wlast]
        self.H = MPO(H_factors)
        return self.H

    def build_neel_state(self):
        """
        Builds a Neel state |up down up down ...>
        Layout: (Left, Phys, Right) -> labels=['lv', 'p', 'rv']
        """
        N = self.L

        factors = []
        for i in range(N):
            # MPS Shape (L,P,R)
            B = np.zeros((1, 2, 1))
            if i % 2 == 0:
                B[0, 0, 0] = 1.0 # Up
            else:
                B[0, 1, 0] = 1.0 # Down
            factors.append(B)
        return MPS(factors, labels=['lv', 'p', 'rv'])

    def build_ferromagnetic_state(self):
        """
        Builds a ferromagnetic state |up up up up ...>
        Layout: (Left, Phys, Right) -> labels=['lv', 'p', 'rv']
        """
        N = self.L
        factors = []
        for i in range(N):
            # Shape (Left=1, Phys=2, Right=1)
            B = np.zeros((1, 2, 1))
            B[0, 0, 0] = 1.0 # Up
            factors.append(B)
        return MPS(factors, labels=['lv', 'p', 'rv'])

    def TDDMRG(self, D=20, **kwargs):
        """
        TDDMRG 

        Parameters
        ----------
        D : int, optional
            the maximum bond dimension for building the short-time propagator. 
            The default is 20.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.01.
        nt : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        if self.H is None: self.build_H_mpo()
        
        return TDMPS(self.H, D=D, **kwargs)
    
    TDMPS = TDDMRG

    def TEBD(self):
        pass

    def DMRG(self, D=None, **kwargs):
        """
        

        Parameters
        ----------
        D : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        from pyqed.mps import DMRG

        if self.H is None:
            self.build_H_mpo()

        if D is None: D = self.D

        return DMRG(H=self.H, D=D, **kwargs)


if __name__ == '__main__':

    mol = Heisenberg(L=10)
    neel = mol.build_neel_state()
    
    # ground state 
    # dmrg = mol.DMRG(D=40, init_guess = neel)
    # dmrg.run()

    # real-time evolution
    td = mol.TDMPS()
    td.run(psi0=neel, dt=0.01, steps=10, e_ops=[mol.H])
    
    # make plots
    import ultraplot as plt 
    fig, ax = plt.subplots()
    ax.plot(td.times, td.observables[:,0])


    # dmrg = mol.TEBD(D=40, init_guess=psi0)
