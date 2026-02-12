#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:28:03 2022

@author: bing

Dyson orbital from EOM-CCSD
"""

from pyscf import gto, scf, cc
import numpy as np
from pyqed.units import au2ev

def dyson_orb_R(a, l_ee, r_ip):
    pass

def dyson_orb_L(l_ip, r_ee):
    '''
    left Dyson amplitudes in MO

    .. math::
        \gamma^L_p = \braket{\Psi^{N-1} | p |\Psi^N}

    Ref:
        PhD thesis USC, 2007

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    r_ee : TYPE
        DESCRIPTION.
    l_ip : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    d = np.einsum('ikc, kc -> i', l_ip, r_ee)
    return d

def analyze_single(t1):
    """
    analyze single-electron transition amplitudes

    .. math::
        T_1 = t_{i}^a a^\dag i

    Parameters
    ----------
    t1 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nocc, nvir = t1.shape
    # nhomo = nocc - 1
    print("\n 1e transition amplitude analysis\n")
    # for i in range(len(w)):
        # print('Root {}    {:6.4f} Hartree    {:6.4f} eV'.format(i, w[i], w[i] * au2ev))
    print('nocc      nvir     coeff')
    print('------------------------')
    # get the indexes with coeff > 0.1
    idx_i, idx_a = np.where(np.abs(t1) > 0.05)

    for k in range(len(idx_i)):
        # coord = np.unravel_index(k, (nocc, nvir))
        i = idx_i[k]
        a = idx_a[k]
        print('{}  --->  {}     {:6.4f}'.format(i, a, t1[i, a]))
    print('------------------------')
    return

def analyze_double(t2):
    pass

# Singlet

# mol = gto.Mole()
# mol.verbose = 5
# mol.unit = 'A'
# mol.atom = 'H 0 0 0; H 0 0 0.7'
# mol.basis = '631g**'
# mol.build()

mol = gto.M(atom='''
  N                   0.000000    0.000000    1.117931
  C                  -0.000000    1.118530    0.333898
  C                  -0.000000    0.714879   -0.976964
  C                  -0.000000   -0.714879   -0.976964
  C                  -0.000000   -1.118530    0.333898
  H                  -0.000000    2.098533    0.763321
  H                  -0.000000    1.359624   -1.831944
  H                  -0.000000   -1.359624   -1.831944
  H                  -0.000000   -2.098533    0.763321
  H                   0.000000    0.000000    2.028535
  ''',
            basis='631g')

mf = scf.RHF(mol)
# mf.verbose = 5
mf.chkfile = 'pyrrole.chk'
mf.scf()

############
# CCSD
############
mycc = cc.RCCSD(mf)
# mycc.verbose = 5
mycc.ccsd()

# t1, t2 amplitudes
# mycc.t1, mycc.t2


print('CCSD total energy', mycc.e_tot)

print('number of MOs = ', mycc.nmo)
print('nocc = {}, nvir = {}'.format(mycc.nocc, mycc.nmo - mycc.nocc))

#############
# IP-EOM-CCSD
#############

# eip, cip = mycc.ipccsd(nroots=2)

# print('ionization energy = ', eip)
i = 1 # D0
# r1_ip, r2_ip = cc.eom_rccsd.vector_to_amplitudes_ip(np.array(cip[i]), mycc.nmo, mycc.nocc)
# print(r2_ip.shape) # nocc, nocc, nvir

eip, cip = mycc.ipccsd(nroots=2, left=True)

print('ionization energy = ', eip)

l1_ip, l2_ip = cc.eom_rccsd.vector_to_amplitudes_ip(np.array(cip[i]), mycc.nmo, mycc.nocc)
# print(l2_ip.shape) # nocc, nocc, nvir

# # eea,cea = mycc.eaccsd(nroots=1)
# # eee,cee = mycc.eeccsd(nroots=1)

#############
# EE-EOM-CCSD
#############
# S->S excitation
ee, cee = mycc.eomee_ccsd_singlet(nroots=2)
i = 1 # S2
print('Root {}, excitation energy  = {:4.2f} eV'.format(i, ee[i]*au2ev))
r1_ee, r2_ee = cc.eom_rccsd.vector_to_amplitudes_ee(np.array(cee[i]), mycc.nmo, mycc.nocc)

    # print(r2.shape) # ijab, nocc, nocc, nvir, nvir
# ee, cee = mycc.eomee_ccsd_singlet(nroots=2, left=True)

do = dyson_orb_L(l2_ip, r1_ee)
print(do)

# analyze_single(r1)

# S->T excitation
# eT = mycc.eomee_ccsd_triplet(nroots=1)[0]


