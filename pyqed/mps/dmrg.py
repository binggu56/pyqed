#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:15:58 2026

@author: Shuoyi Hu, Sha Mo, Bing Gu
"""

from pyqed.mps import MPS, MPO, fDMRG_1site_GS_OBC, two_site_dmrg, dense_to_symmetric,\
    expect_mps
import numpy as np

try:
    from pyqed.mps.symmetry import BlockTensor, tensordot, solve_davidson, QN, SymmetryManager
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None


class DMRG:
    """
    ground state finite DMRG in MPO/MPS framework
    """
    def __init__(self, H, D, init_guess=None, nsweeps=50, opt='2site',\
                 symmetry=None, charge=None, not_conv_err=True):
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
        # self.L = self.H.L
        self.D = D
        self.nsweeps = nsweeps
        self.opt = opt

        self.init_guess = init_guess
        self.mps = None
        self.e_tot = None
        self.U1 = self.symmetry = symmetry

        self.target_qn = self.charge = charge

        self.ground_state = None
        self.mps = None # to hold eigenstates
        
        # self.ground_state_raw = None

        self.not_conv_err = not_conv_err
        self.converged = False
        # self.sym_mgr = sym_mgr

    def run(self):

        if self.init_guess is None:
            raise ValueError('Please provide an initial guess.')

        # Standardize MPS to ['lv', 'p', 'rv']
        # but currently we are not using the initial guess as MPS objects a lot, but i do think that is the better option. so need to fix initial guess in dmrg.py. remve this TODO when fixed.
        if isinstance(self.init_guess, MPS):
            mps_list = self.init_guess.to_order(['lv', 'p', 'rv']).factors
            # mps_list = self.init_guess.factors
        else:
            # If it's a raw list, we assume it respects the convention. TODO: maybe add auto check and warning and raise error.
            mps_list = self.init_guess

        if isinstance(self.H, MPO):
            mpo_list = self.H.factors
        else:
            mpo_list = self.H

        if self.symmetry:

            if isinstance(mps_list, list) and not \
                isinstance(mps_list[0], BlockTensor):

                mps_list = dense_to_symmetric(mps_list, phys_qns=None)

            if self.charge is not None:

                qs = sorted({key[1] for key in mps_list[-1].data.keys()})
                if len(qs) != 1:
                    raise ValueError(f"Ambiguous total charge: {qs}.")
                self.charge = qs[0]

        if self.opt == '1site':

            fDMRG_1site_GS_OBC(mpo_list, self.D, self.nsweeps)

        elif self.opt == '2site':

            self.e_tot, ground_state, gauge, self.converged = two_site_dmrg(
                mps_list, mpo_list, self.D, self.nsweeps, \
                    U1=self.U1, target_qn=self.charge, not_conv_err=self.not_conv_err, sym_mgr=None)

            if self.U1:
                # U1 engine returns [Left, Right, Phys]
                labels = ['lv', 'rv', 'p']
            else:
                # Dense engine returns [Left, Phys, Right]
                labels = ['lv', 'p', 'rv']
            
            # self.ground_state = self.mps = MPS(ground_state, labels=labels, gauge=gauge)
            
            gauge = gauge.lower()
            if gauge == "left":

                self.ground_state = MPS(ground_state, labels=labels,\
                                        center=len(ground_state)-1)
                
                # TODO:THIS IS REDUNDENT, but retained here for computing Ss 
                # which is needed in make_rdm1()
                
                self.ground_state.left_canonicalize() 

            elif gauge == "right":

                self.ground_state = MPS(ground_state, labels=labels,
                                        center=0)
                self.ground_state.right_canonicalize()

        else:
            raise ValueError('Optimization algorithm {self.opt} does not exist. Use "1site" or "2site".')

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

        return [expect_mps(psi, e_op) for e_op in e_ops]

    def make_rdm1(self, idx=None):
        """
        Calculate 1-site reduced density matrix of the ground state.
        Wrapper for MPS.calc_1site_rdm
        \gamma_{ij} = < 0| c_j^\dagger c_i | 0 >
        """
        if self.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")

        return self.ground_state.calc_1site_rdm(idx)

    def make_rdm2(self, idx_pairs=None):
        """
        Calculate 2-site reduced density matrix of the ground state.
        Wrapper for MPS.calc_2site_rdm
        """
        if self.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")

        return self.ground_state.calc_2site_rdm(idx_pairs)

if __name__ == '__main__':

    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dmrg = DMRG(H, D=20, nsweeps=8)
    dmrg.init_guess = neel
    dmrg.run()
    