#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:59:10 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np
import scipy.constants as const
import scipy.linalg as la
import scipy 
from scipy.sparse import identity, kron, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import kron, norm, eigh

from scipy.special import erf
from scipy.constants import e, epsilon_0, hbar
import logging
import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson
from pyqed.ldr.ldr import kinetic
from pyqed import au2ev, au2angstrom
from pyqed.dvr import SineDVR
# from pyqed import scf
from pyqed.qchem.gto.rhf import make_rdm1, energy_elec
# from pyqed.jordan_wigner import jordan_wigner_one_body, jordan_wigner_two_body

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB

from numba import vectorize, float64, jit
import sys
from opt_einsum import contract


class SpinHalfFermionChain:
        
    """    
    exact diagonalization of spin-half open fermion chain with long-range interactions
    
    by Jordan-Wigner transformation 
    
    .. math::
        
        H = \sum_{<rs>} (c_r^\dag c_s + c†scr−γ(c†rc†s+cscr))−2λ∑rc†rcr,
    
    where r and s indicate neighbors on the chain. 
    
    Electron interactions can be included in the Hamiltonian easily.
    
    """
    def __init__(self, h1e, eri, nelec):
        # if L is None:
        L = h1e.shape[-1]
        self.L = self.nsites = L
        
        self.h1e = h1e
        self.eri = eri
        self.d = 4 # local dimension of each site
        # self.filling = filling
        self.nelec = nelec
        
        self.H = None
        
    def run(self, nstates=6):
            
        # # single electron part
        # Ca = mf.mo_coeff[:, :self.ncas]
        # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)
        
                
        # eri = self.mf.eri
        # eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

        # self.hcore_mo = hcore_mo
        
        H = self.jordan_wigner(self.h1e, self.eri)
        self.H = H        
        E, X = eigsh(H, k=nstates, which='SA')

        return E, X
        
        
    
    def jordan_wigner(self, h1e, v):
        """
        MOs based on Restricted HF calculations 

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        # an inefficient implementation without consdiering any syemmetry 
        
        from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body
        
        nelec = self.nelec
        
        norb = h1e.shape[-1]
        nmo = L = norb # does not necesarrily have to MOs

        
        Cu = annihilate(norb, spin='up')
        Cd = annihilate(norb, spin='down')
        Cdu = create(norb, spin='up')
        Cdd = create(norb, spin='down')
        
        H = 0        
        # for p in range(nmo):
        #     for q in range(p+1):
                # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
        for p in range(nmo):
            for q in range(nmo):
                H += h1e[p, q] * (Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q])
        
        # build total number operator
        # number_operator = 0
        Na = 0
        Nb = 0
        for p in range(L):
            Na += Cdu[p] @ Cu[p] 
            Nb += Cdd[p] @ Cd[p]

        
        # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry 
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        H += 0.5 * v[p, q, r, s] * (\
                            Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q] +\
                            Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q] +\
                            Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q] +
                            Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q])
                        # H += jordan_wigner_two_body(p, q, s, r, )
    
        # digonal elements for p = q, r = s
        I = tensor(Is(L))
        
        return H + (Na - nelec/2 * I) @ (Na - self.nelec/2 * I) + \
            (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)
    
    
    def DMRG(self):
        pass
    
    def gen_mps(self):
        pass


class DMRG:
    """
    ab initio DRMG quantum chemistry calculation
    """
    def __init__(self, mf):
        self.mf = mf
        
    def run(self):
        pass
    
        
    
if __name__=='__main__':
    from pyscf import gto, scf, dft, tddft, ao2mo
    from pyqed.qchem.mol import get_hcore_mo, get_eri_mo
    from pyqed.qchem.gto.rhf import RHF
    
    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['Li' , (0. , 0. , 0.)], ]
    mol.basis = 'sto3g'
    mol.build()
    

    
    mf = scf.RHF(mol).run()
    
    # e, fcivec = pyscf.fci.FCI(mf).kernel(verbose=4)
    # print(e)
    # Ca = mf.mo_coeff[0ArithmeticError
    # n = Ca.shape[-1]
    
    # mo_coeff = mf.mo_coeff
    # get the two-electron integrals as a numpy array
    # eri = get_eri_mo(mol, mo_coeff)
    
    # n = mol.nao
    # Ca = mo_coeff
        
    h1e = get_hcore_mo(mf)
    eri = get_eri_mo(mf)
    
    E, X = SpinHalfFermionChain(h1e, eri, nelec=mol.nelectron).run()