#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 23:48:00 2025

@author: bingg
"""
from pyqed.qchem.ci import CASCI
import numpy as np



def newton_raphson():
    pass

def super_ci():
    pass

def perturbative_superci():
    pass

def orbital_gradient(fock_inactive, fock_active, dm1, dm2, eri):
    """
    compute the orbital gradients
    .. math::

        \langle 0 | H E_{pq} | 0 \rangle

    The non-redundent transitions are pq = ai, ti, at

    Refs
    ----


    Parameters
    ----------
    fock_inactive : TYPE
        DESCRIPTION.
    fock_active : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.
    eri : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

class CASPT2(CASSCF):
    pass


if __name__=='__main__':
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import ao2mo
    
    from opt_einsum import contract
    
    # J. Comput. Chem. 2019, 40, 1463-1470
    
    mol = Molecule(atom = [
        ['Li' , (0. , 0. , 0)],
        ['H' , (0. , 0. , 1)], ])

    mol.basis = 'sto6g'
    mol.charge = 0
    
    mol.build()
    
    
    mf = mol.RHF().run()
    C = mf.mo_coeff 
    
    # CASSCF 
    ncas, nelecas = 2, 2 
    
    mc = CASCI(mf, ncas, nelecas)
    
    # inactive AO Fock matrix
    ncore = mc.ncore
    C_core = C[:, :ncore]
    dm_core = 2 * contract('ui ,vi -> uv', C_core, C_core)
    
    Fcore = mf.get_fock(dm_core)
    
    Fcore_mo = ao2mo(Fcore, C)
    
    eri = mf.get_eri_mo(C)
    
    mc.run()
    dm1 = mc.make_rdm1()
    dm2 = mc.make_rdm2()
    
    # transform RDMs to AO
    
    
    # active AO Fock matrix
    C_active = C[:, ncore:ncore+ncas]
    D_active = C_active @ D @ C_active.T 

    G = mf.get_veff(D_active)
    
    ### pyscf 
    
    # mol = mol.topyscf()
    # mf = mol.RHF().run()
    # F = mf.get_fock()
    
    # print(F)
    