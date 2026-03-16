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
                symmetry=False, charge=None, spin = None,\
                target_qn = None, sym_mgr = None, not_conv_err=True,
                nstates=1, weights=None): # [FIX] Added nstates and weights
        """
        Parameters
        ----------
        H : MPO
            MPO of the Hamiltonian.
        D : int
            maximum bond dimension.
        nsweeps : int
            Number of sweeps to perform.
        nstates : int
            Number of states for State-Averaged DMRG.
        weights : list
            Weights for state averaging.
        """

        self.H = H
        self.L = len(self.H)
        self.D = D
        self.nsweeps = nsweeps
        self.opt = opt

        self.init_guess = init_guess
        self.e_tot = None
        self.U1 = self.symmetry = symmetry
        

        self.nstates = nstates
        self.weights = weights if weights is not None else [1.0/nstates]*nstates

        # Symmetry Logic
        if target_qn is not None and (sym_mgr is None):
            raise ValueError("Symmetry manager must be provided when target quantum number is specified.")
        elif target_qn is None and sym_mgr is not None:
            raise ValueError("Target quantum number must be specified when sym_mgr is given.")
        elif (charge is not None) and (spin is not None):
            sym_mgr = SymmetryManager(['charge', 'sz'])
            target_qn = sym_mgr.get_target_qn(charge, 2*spin)
        elif (charge is not None) and (spin is None):
            sym_mgr = SymmetryManager(['charge'])
            target_qn = sym_mgr.get_target_qn(charge)
        elif (charge is None) and (spin is not None):
            sym_mgr = SymmetryManager(['sz'])
            target_qn = sym_mgr.get_target_qn(2*spin)
            
        self.charge = charge
        self.target_qn = target_qn 
        self.sym_mgr = sym_mgr

        self.ground_state = None # Holds Root 0
        self.states = None       # Holds list of all Roots
        self.not_conv_err = not_conv_err
        self.converged = False

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

        mpo_list = self.H.factors if isinstance(self.H, MPO) else self.H

        if self.symmetry and not isinstance(mps_list[0], BlockTensor):
            mps_list = dense_to_symmetric(mps_list, sym_mgr=self.sym_mgr)

        if self.opt == '1site':

            fDMRG_1site_GS_OBC(mpo_list, self.D, self.nsweeps)

        elif self.opt == '2site':
            res = two_site_dmrg(
                mps_list, mpo_list, self.D, self.nsweeps, 
                U1=self.U1, target_qn=self.target_qn, 
                not_conv_err=self.not_conv_err, sym_mgr=self.sym_mgr,
                nstates=self.nstates, weights=self.weights
            )
            e_elec, mps_out, self.gauge, self.converged = res

            shift = getattr(self.H, 'constant', 0.0)
            
            labels = ['lv', 'rv', 'p'] if self.U1 else ['lv', 'p', 'rv']
            center = (len(self.H) - 1) if self.gauge.lower() == "left" else 0

            if self.nstates == 1:
                self.e_tot = e_elec + shift
                self.ground_state = MPS(mps_out, labels=labels, center=center)
                self.states = [self.ground_state]
            else:
                self.e_tot = [e + shift for e in e_elec]
                self.states = [MPS(s, labels=labels, center=center) for s in mps_out]
                self.ground_state = self.states[0]

            for s in self.states:
                if self.gauge.lower() == "left": 
                    s.left_canonicalize()
                else: 
                    s.right_canonicalize()

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

    def make_rdm1(self):
        """
        Calculate the global 1-site reduced density matrix of the optimized ground state.
        
        Wrapper for `MPS.make_rdm1`. Computes the matrix $\\gamma_{ij} = \\langle 0 | c_i^\\dagger c_j | 0 \\rangle$.

        Parameters
        ----------
        idx : optional
            Placeholder parameter to maintain API compatibility. Currently ignored as 
            the function computes the full `(L, L)` global matrix. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L)` representing the global 1-RDM.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_rdm1(sym_mgr=self.sym_mgr)

    def make_local_site_rdm(self, idx=None):
        """
        Calculate the local reduced density matrices for individual, isolated sites.
        
        Wrapper for `MPS._calc_local_site_rdms`. Traces out the rest of the chain 
        to isolate the internal $d \\times d$ quantum state of specific sites.

        Parameters
        ----------
        idx : int or list of int, optional
            The specific site index (or indices) to evaluate. If None, evaluates 
            the local density matrices for all sites in the chain. By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices to their corresponding 
            $d \\times d$ local density matrices (as numpy arrays).
        """
        return self.ground_state._calc_local_site_rdms(idx=idx)

    def make_rdm2(self, idx_pairs=None):
        """
        Calculate the full global 2-site reduced density matrix of the ground state.
        
        Wrapper for `MPS.make_rdm2`. Computes the complete $\\mathcal{O}(L^4)$ tensor 
        $\\Gamma_{pqrs} = \\langle c_p^\\dagger c_r^\\dagger c_s c_q \\rangle$.

        Parameters
        ----------
        idx_pairs : optional
            Placeholder parameter to maintain API compatibility. Currently ignored as 
            the function computes the full `(L, L, L, L)` global tensor. By default None.

        Returns
        -------
        np.ndarray
            A dense complex numpy array of shape `(L, L, L, L)`.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_rdm2(sym_mgr=self.sym_mgr)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the diagonal blocks of the 2-site reduced density matrix.
        
        Wrapper for `MPS.make_diagonal_rdm2`. Extracts the two-site quantum state $\\rho_{ij}$ needed to compute density-density correlations like $\\langle n_i n_j \\rangle$ without evaluating the full $\\mathcal{O}(L^4)$ tensor.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the 2-site RDM for. 
            If None, computes RDMs for all possible unique pairs. By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding 
            dense reduced density matrix numpy array.
        """
        # if self.ground_state is None:
        #     raise ValueError("Run DMRG first to generate a ground state.")
            
        return self.ground_state.make_diagonal_rdm2(idx_pairs=idx_pairs)


if __name__ == '__main__':

    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dmrg = DMRG(H, D=20, nsweeps=8)
    dmrg.init_guess = neel
    dmrg.run()
    