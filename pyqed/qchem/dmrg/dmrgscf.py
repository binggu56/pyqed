#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:10:50 2026

DMRGSCF

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""
# TODO: so since we are sharing CASSCF optimization code, currently after the DMRGSCF, final print get E(CASSCF) = xxxxxxx, it might be better if we fix that.
from pyqed.qchem import QCDMRG, CASSCF
from pyqed.qchem.mcscf.cocas import kernel, kernel_state_average
import numpy as np


class DMRGSCF(QCDMRG):
    def __init__(self, mf, ncas, nelecas, D=20, max_cycles=30, **kwargs):
       
        super().__init__(mf, ncas, nelecas, D, **kwargs)

        self.max_cycles = max_cycles # macroiterations
        self.tol = 1e-6 # energy tol
        self.mo_coeff = None # opt orb


        self.weights = None
        self.nstates = 1


    def run(self, nstates=1, weights = None, **kwargs):
        mf = self.mf

        # canonical molecular orbs
        C0 = mf.mo_coeff

        # CASCI roots
        if nstates == None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if weights != None:
            self.weights = weights
            if nstates != len(self.weights):
                raise ValueError("the nstates you requires does not align with the nstates indicated by the weights. check input.")

        nmo = self.mf.nao
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        mc = QCDMRG(
            mf,
            ncas=ncas,
            nelecas=nelecas,
            D=self.D,
            site=getattr(self, "site", getattr(self, "site_basis", "spin_orbital")),
            verbose=getattr(self, "verbose", 0),
        )

        # spin
        mc.spin_purification = self.spin_purification
        mc.ss = self.ss
        mc.shift = self.shift

        mc.run(nstates=self.nstates, weights=self.weights, **kwargs)

        # matrix elements in CMOs
        h1e = mf.get_hcore_mo()
        eri = mf.get_eri_mo()

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.

        if nstates == 1: # ground state only
            C, mc = kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=self.max_cycles)

        elif nstates > 1:
            if self.weights is None:
                self.state_average(weights = np.ones(nstates)/nstates)
            if len(self.weights) != nstates: 
                self.state_average(weights = np.ones(nstates)/nstates)
            mc.nstates = self.nstates
            C, mc = kernel_state_average(mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                                            C0=C0, h1e=h1e, eri=eri)

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = mc.ci
        self.e_history = getattr(mc, 'e_history', [self.e_tot])

        return self

    def state_average(self, weights):
        self.nstates = len(weights)
        self.weights = weights
        return self

if __name__=='__main__':

    from pyqed import Molecule

    mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='6311g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = DMRGSCF(mf, ncas=6, nelecas=6, D=60, max_cycles=50)

    mc.fix_spin(ss=0, shift=0.2)
    mc.run(
        nstates=1,
        symmetry_list=['charge', 'sz'], 
        initial_guess='cid'
    )
