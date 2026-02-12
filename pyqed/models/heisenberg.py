#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 22:30:16 2026

Heisenberg model 

@author: Shuoyi Hu, Bing Gu (gubing at westlake dot edu dot cn)
"""
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.mps import MPO, MPS

# import ultraplot as plt

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

    def TDDMRG(self):
        """
        TDDMRG 

        Parameters
        ----------
        psi0 : TYPE, optional
            DESCRIPTION. The default is None.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.01.
        nt : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # if psi0 is None:
        #     print('Initial state not provided. Using Neel state.')
        #     psi0 = self.build_neel_state()
        
        if self.H is None: self.build_H_mpo()
        
        return TDMPS(self.H)

    def TEBD(self):
        pass

    def DMRG(self, D=None, **kwargs):
        from pyqed.mps import DMRG

        if self.H is None:
            self.build_H_mpo()

        if D is None: D = self.D

        return DMRG(H=self.H, D=D, **kwargs)


if __name__ == '__main__':

    mol = Heisenberg(L=10)
    neel = mol.build_neel_state()
    
    # ground state 
    dmrg = mol.DMRG(D=40, init_guess = neel)
    dmrg.run()

    # real-time evolution
    td = mol.TDDMRG()
    td.run(psi0=neel, dt=0.01, steps=10, e_ops=[mol.H])
    
    # make plots
    import ultraplot as plt 
    fig, ax = plt.subplots()
    ax.plot(td.times, td.observables[:,0])


    # dmrg = mol.TEBD(D=40, init_guess=psi0)

    
    # # Initialize TDMPS Solver
    # # solver = TDMPS(psi0, H_mpo, dt, bond_dim=bond_dim, order=4)
    
    
    # # Plot if you wish
    # times = results['time']
    # energy = np.real(results['obs'][0])
    # norms = results['norm_check']
    
    # print("\nSimulation Complete.")
    # print(f"Final Energy: {energy[-1]:.6f}")
    # print(f"Energy Conservation Error: {np.max(np.abs(energy - energy[0])):.2e}")
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(times, energy, 'b.-')
    # plt.title('Total Energy <H>(t)')
    # plt.xlabel('Time')
    # plt.ylabel('Energy')
    # plt.grid(True)
    
    # plt.subplot(1, 2, 2)
    # plt.plot(times, norms, 'r--')
    # plt.title('Norm <psi|psi>')
    # plt.xlabel('Time')
    # plt.ylim(0.99, 1.01)
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()