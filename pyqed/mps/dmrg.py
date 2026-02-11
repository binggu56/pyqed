#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:15:58 2026

@author: Shuoyi Hu, Sha Mo, Bing Gu
"""

from pyqed.mps import MPS, MPO, fDMRG_1site_GS_OBC, two_site_dmrg, dense_to_symmetric,\
    expect


from pyqed.mps.decompose import decompose, compress
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
        self.D = D
        self.nsweeps = nsweeps
        self.opt = opt

        self.init_guess = init_guess
        self.mps = None
        self.e_tot = None
        self.U1 = self.symmetry = symmetry
        
        self.target_qn = self.charge = charge
        
        self.ground_state = None
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
            mps_std = self.init_guess.to_order(['lv', 'p', 'rv'])
            self.mps_list = mps_std.factors
        else:
            # If it's a raw list, we assume it respects the convention. TODO: maybe add auto check and warning and raise error.
            self.mps_list = self.init_guess

        if isinstance(self.H, MPO):
            self.mpo_list = self.H.factors
        else:
            self.mpo_list = self.H

        if self.U1:
            
            if isinstance(self.mps_list, list) and not \
                isinstance(self.mps_list[0], BlockTensor):
                
                self.mps_list = dense_to_symmetric(self.mps_list, phys_qns=None)

            if self.charge is not None:
                
                qs = sorted({key[1] for key in self.mps_list[-1].data.keys()})
                if len(qs) != 1:
                    raise ValueError(f"Ambiguous total charge: {qs}.")
                self.charge = qs[0]

        if self.opt == '1site':
        
            fDMRG_1site_GS_OBC(self.mpo_list, self.D, self.nsweeps)
        
        elif self.opt == '2site':
            
            self.e_tot, self.ground_state_raw, self.gauge, self.converged = two_site_dmrg(
                self.mps_list, self.mpo_list, self.D, self.nsweeps, \
                    U1=self.U1, target_qn=self.charge, not_conv_err=self.not_conv_err, sym_mgr=None)
            
            if self.U1:
                # U1 engine returns [Left, Right, Phys]
                final_labels = ['lv', 'rv', 'p']
            else:
                # Dense engine returns [Left, Phys, Right]
                final_labels = ['lv', 'p', 'rv']
                
            if self.gauge == "left":
                
                self.ground_state = MPS(self.ground_state_raw, labels=final_labels, center=len(self.mps_list) -1)
                self.ground_state.left_canonicalize()
            
            elif self.gauge == "right":
                
                self.ground_state = MPS(self.ground_state_raw, labels=final_labels, center=0)
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

        return [expect(psi, e_op) for e_op in e_ops]

    def make_rdm(self, idx=None):
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


