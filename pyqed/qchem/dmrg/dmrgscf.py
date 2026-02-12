#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:10:50 2026

DMRGSCF

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""
from pyqed.qchem import QCDMRG, CASSCF
import numpy as np


class DMRGSCF(QCDMRG):
   def __init__(self, mf, ncas, nelecas, D=20, max_cycles=30, **kwargs):
       super().__init__(mf, ncas, nelecas, D, **kwargs)

       self.max_cycles = max_cycles # macroiterations
       self.tol = 1e-6 # energy tol
       self.mo_coeff = None # opt orb


       self.weights = None
       self.nstates = 1


   def run(self, nstates=1):
       mf = self.mf

       # canonical molecular orbs
       C0 = mf.mo_coeff

       # CASCI roots
       nstates = self.nstates

       nmo = self.mf.nao
       ncas = self.ncas
       nelecas = self.nelecas
       ncore = self.ncore

       mc = QCDMRG(mf, ncas=ncas, nelecas=nelecas, D=self.D)

       # spin
       mc.spin_purification = self.spin_purification
       mc.ss = self.ss
       mc.shift = self.shift


       mc.run(nstates)


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
               raise ValueError('State weights not provided.')

           C, mc = kernel_state_average(mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                                        C0=C0, h1e=h1e, eri=eri)

       self.mo_coeff = C
       self.e_tot = mc.e_tot
       self.ci = mc.ci

       return self

   def state_average(self, weights):
       self.nstates = len(weights)
       self.weights = weights
       return self