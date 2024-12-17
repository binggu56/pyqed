#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:40:40 2024

@author: bingg
"""

import numpy as np
from pyqed.qchem.rhf import RHF, get_veff

class UHF:
    def __init__(self, mol):
        self.mol = mol

    def run(self):
        pass


def energy_elec(mf, dm=None, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock

    Note this function has side effects which cause mf.scf_summary updated.

    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = get_veff(mf.mol, dm)
    if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
        h1e = (h1e, h1e)
    e1 = np.einsum('ij,ji->', h1e[0], dm[0])
    e1+= np.einsum('ij,ji->', h1e[1], dm[1])
    e_coul =(np.einsum('ij,ji->', vhf[0], dm[0]) +
             np.einsum('ij,ji->', vhf[1], dm[1])) * .5
    e_elec = (e1 + e_coul).real
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
    return e_elec, e_coul