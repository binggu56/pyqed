#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, ao2mo, mcscf

'''
MO integrals needed by spin-orbit coupling
'''

mol = gto.M(
    atom = 'H 0 0 -1.8; Li 0 0 0; H 0. 0 1.8',
    basis = 'augccpvdz',
    spin = 1,
    charge = 4)


def soc(mol):
    """
    
    Ref:
        JCP, 122, 034107

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    h1 : TYPE
        DESCRIPTION.
    h2 : TYPE
        DESCRIPTION.

    """
    
    myhf = scf.RHF(mol)
    myhf.kernel()
    print(myhf.mo_energy[:10])
    # mycas = mcscf.CASSCF(myhf, 2, 1) # 6 orbital, 8 electron
    # mycas.kernel()

    # CAS space orbitals
    # cas_orb = mycas.mo_coeff[:,mycas.ncore:mycas.ncore+mycas.ncas]
    
    # 2-electron part spin-same-orbit coupling
    #       [ijkl] = <ik| p1 1/r12 cross p1 |jl>
    # JCP, 122, 034107 Eq (3) = h2 * (-i)
    # For simplicty, we didn't consider the permutation symmetry k >= l, therefore aosym='s1'
    
    # if mol.nelectron > 1:
    #     h2 = ao2mo.kernel(mol, cas_orb, intor='int2e_p1vxp1_sph', comp=3, aosym='s1')
    #     print('SSO 2e integrals shape %s' % str(h2.shape))
    
    # 1-electron part for atom A
    #       <i| p 1/|r-R_A| cross p |j>
    # JCP, 122, 034107 Eq (2) = h1 * (iZ_A)
    mo = myhf.mo_coeff
    
    # OAM of MOs
    # l = numpy.einsum('xpq,pi,qj->xij', mol.intor('int1e_giao_irjxp_sph')  , mo, mo)
    # for p in range(mol.nao):
    #     print(l[2, p, p])
              
    # print(mo.shape)
    mol.set_rinv_origin(mol.atom_coord(1))  # set the gauge origin on second atom
    

    h1 = numpy.einsum('xpq,pi,qj->xij', mol.intor('int1e_prinvxp'), mo, mo)
    print('1e integral shape %s' % str(h1.shape))
    for p in range(10):
        print(1j * h1[2, p, p])
    # print(1j * h1[2, 3, 3])
    
    # transition density matrix 
    
    
    return h1

soc(mol)