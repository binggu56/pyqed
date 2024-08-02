#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:52:57 2023


Wfn analysis for PySCF calculations

@author: bing
"""

import os
os.environ['OMP_NUM_THREADS'] = "4"

import logging
import numpy as np
import numpy
from pyscf import gto, dft, scf, lib, ao2mo
from pyscf.lib import logger
import scipy
from pyqed.units import au2ev
from pyqed.phys import dag

def _tdm_ee(X1, X2, mo_coeff, reduced_transition_space=False, occidx=None, viridx=None):
    """
    Compute the transition density matrix in AO basis between TDA/CIS/TDHF excited
    states with the same excitation space.

    D_{ia} = \braket{\Phi_J| a^\dag i | \Phi_I}

    This only calculates the transition density matrix. For reduced density matrices,
    call density_matrix(state_id).

    The excited states are chosen as
    .. math::
        \Phi_I = X^I_{ia} a^\dag i \ket{\Phi_0}

    Parameters
    ----------
    X1 : ndarray [nocc, nvir, nstates], nocc, nvir are the size of the excitation space
        TDA coeff for ket state.
    X2 : TYPE
        TDA coeff for bra state.
    mo_coeff : TYPE
        DESCRIPTION.
    mo_occ : TYPE
        DESCRIPTION.
    reduced_transition_space: bool
        whether the transition space is reduced. If True, the occidx and viridx have to be provided.
    occidx : list, optional
        occupied orbitals index for the ket state. The default is None.
    viridx : list of integers, optional
        vir orbs index. The default is None.

    Returns
    -------
    tdm : TYPE
        DESCRIPTION.

    """

    
    # if occidx is None:
    #     occidx = list(numpy.where(mo_occ==2)[0])
    # if viridx is None:
    #     viridx = list(numpy.where(mo_occ==0)[0])

    # the two excited states use the same transition space
    assert(X1.shape == X2.shape)

    # occ and vir orbs in the reduced excitation ov space
    nocc, nvir = X1.shape
    nmo = nocc + nvir
    tdm = np.zeros((nmo, nmo))
    tdm_oo = -np.einsum('ia,ka->ik', X2.conj(), X1)
    tdm_vv = np.einsum('ia,ic->ac', X1, X2.conj())

    tdm[:nocc,:nocc] += tdm_oo * 2
    tdm[nocc:,nocc:] += tdm_vv * 2

    # Transform density matrix to AO basis, this is only valid
    # for non-restricted calculations
    if reduced_transition_space:
        mo = mo_coeff[:, [*occidx, *viridx]]
    else:
        mo = mo_coeff

    # tdm = np.einsum('up, pq, vq->uv', mo, tdm, mo.conj())
    tdm = mo @ tdm @ mo.conj().T
    return tdm


def build_tdm_ee(td, J, I):
    """
    Compute the transition density matrix between excited states in AO
    basis for TDA
    ..math::
        \Gamma_{\mu \nu} = \langle \Phi_J | \mu^\dag \nu |\Phi_I \rangle
         = C_{\mu p} \langle \Phi_J | p^\dag q |\Phi_I \rangle\
            C^*_{\nu q}
    :math:`\mu, \nu` labels the AOs, i,j ... occ.; a, b, ... virtual

    Parameters
    ----------
    td: tdscf obj
    J: int
        bra many-electron state id
    I: int, 0 is the first excited state.
        ket state id
    Returns
    -------

    """
    assert(I < J)
    # I, J has to be using the same active space
    X1 = td.xy[I][0] # [no, nv]
    X2 = td.xy[J][0]

    # nocc, nvir = X1.shape
    # nmo = nocc + nvir
    # tdm = np.zeros((nmo, nmo))
    # tdm_oo =-np.einsum('ia,ka->ik', X2.conj(), X1)
    # tdm_vv = np.einsum('ia,ic->ac', X1, X2.conj())

    # tdm[:nocc,:nocc] += tdm_oo * 2
    # tdm[nocc:,nocc:] += tdm_vv * 2

    # # Transform density matrix to AO basis, this is only valid
    # # for non-restricted calculations
    # mo = td._scf.mo_coeff[:, self.occidx + self.viridx]
    # tdm = np.einsum('pi,ij,qj->pq', tdm, mo.conj())
    mo_occ = td._scf.mo_occ
    mo_coeff = td._scf.mo_coeff
    # print(mo_coeff)
    
    # occidx = list(numpy.where(mo_occ==2)[0])
    # viridx = list(numpy.where(mo_occ==0)[0])
    # print(occidx)
    
    return _tdm_ee(X1, X2, mo_coeff)

def transition_dipole_from_tdm(mol, tdm, mo_coeff):

    # 1e operator in ao basis
    ints = mol.intor_symmetric('int1e_r', comp=3)

    # transform r to mo basis
    # r_pq = C_up^* r_uv C_vq, here in contrast to ground-to-excited state,
    # p, q run through all MOs
    ints = np.einsum('xuv, up, vq -> xpq', ints, \
                      mo_coeff.conj(), mo_coeff)

    # The transition dipole r_JI = r_pq \braket{\Phi_J| p^\dag q|\Phi_I}
    # = r_pq * D_{qp}
    # the factor of 2 from spin is included in the TDM
    if isinstance(tdm, list):

        dip = [np.einsum('xai,ia->x', ints, t) for t in tdm]

    else:
        dip = np.einsum('xai,ia->x', ints, tdm)

    return dip


def tda_denisty_matrix(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    
    state_id : int
        0 refers to the first-excited state
    
    From PySCF/examples/tddft/22
    
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

def dip_moment(td, state_id):
    """
    compute the permanent dipole moment of an excited state

    Parameters
    ----------
    td : TYPE
        DESCRIPTION.
    state_id :  int
        DESCRIPTION.

    Returns
    -------
    dip : TYPE
        DESCRIPTION.

    """
    dm = tda_denisty_matrix(td, state_id)
    dip = np.einsum('xij,ji->x', mol.intor('int1e_r'), dm)
    return dip 

def transition_dipole_moment(td, bra_id, ket_id):
    
    assert(bra_id > ket_id)

    # transition_density_matrix between excited states
    tdm = build_tdm_ee(td, bra_id, ket_id)
    dip = transition_dipole_from_tdm(td._scf.mol, tdm, td._scf.mo_coeff)
    
    return dip


if __name__ == '__main__':
    from pyscf import tddft 
    
    mol = gto.Mole(
        atom = 
        # '''
        # C                  0.00000000    1.29304800    0.00000000
        # C                 -1.11981300   -0.64652400    0.00000000
        # C                  1.11981300   -0.64652400    0.00000000
        # H                  0.00000000    2.37935800    0.00000000
        # H                 -2.06058400   -1.18967900    0.00000000
        # H                  2.06058400   -1.18967900    0.00000000
        # N                  0.00000000   -1.37201200    0.00000000
        # N                  1.18819800    0.68600600    0.00000000
        # N                 -1.18819800    0.68600600    0.00000000
        # ''',  # in Angstrom
        """
        H 0 0 0 
        F 0 0 1.1
        """,
        basis = '631g',
        # basis = 'ccpvdz',
        symmetry = True,
    )
    
    mf = dft.RKS(mol)
    # mf.xc = 'svwn' # shorthand for slater,vwn
    #mf.xc = 'bp86' # shorthand for b88,p86
    #mf.xc = 'blyp' # shorthand for b88,lyp
    # mf.xc = 'pbe' # shorthand for pbe,pbe
    #mf.xc = 'lda,vwn_rpa'
    #mf.xc = 'b97,pw91'
    # mf.xc = 'pbe0'
    # mf.xc = 'b3p86'
    # mf.xc = 'wb97x'
    #mf.xc = '' or mf.xc = None # Hartree term only, without exchange
    mf.xc = 'b3lyp'
    mf.kernel()
    
    mf.analyze()
    
    td = tddft.TDA(mf)
    td.nstates = 3
    td.kernel()
    # td.analyze(verbose=4)
    
    tdm = transition_dipole_moment(td, 1, 0)
    print(tdm)
    

        

        
