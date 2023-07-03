# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:46:47 2022

Compute core-excitations

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



def banded(a):
    """
    Construct banded form of a matrix for eig_banded()

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    b : TYPE
        DESCRIPTION.

    """
    b = np.zeros_like(a)
    n = a.shape[0]
    for i in range(n):
        b[i, :n-i] = np.diag(a, i)
    return b


def get_ab_ras(mf, occidx, viridx, mo_energy=None, mo_coeff=None, mo_occ=None, TDA=True):
    # construct the truncated A, B matrix for TDA/TDDFT calculations
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    if TDA, B = 0.
    '''

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    assert(mo_coeff.dtype == numpy.double)

    mol = mf.mol
    nao, nmo = mo_coeff.shape

    # if occidx is None:
    #     occidx = numpy.where(mo_occ==2)[0]
    # if viridx is None:
    #     viridx = numpy.where(mo_occ==0)[0]

    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))
    nmo = nocc + nvir

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb

        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, 'LDA')
                fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[2]
                frr = fxc[0]

                rho_o = lib.einsum('rp,pi->ri', ao, orbo)
                rho_v = lib.einsum('rp,pi->ri', ao, orbv)
                rho_ov = numpy.einsum('ri,ra->ria', rho_o, rho_v)
                w_ov = numpy.einsum('ria,r->ria', rho_ov, weight*frr)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov, w_ov) * 2
                a += iajb
                b += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, 'GGA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vgamma = vxc[1]
                frho, frhogamma, fgg = fxc[:3]

                rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = numpy.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += numpy.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
                sigma1 = numpy.einsum('xr,xria->ria', rho[1:4], rho_ov[1:4])

                w_ov = numpy.empty_like(rho_ov)
                w_ov[0]  = numpy.einsum('r,ria->ria', frho, rho_ov[0])
                w_ov[0] += numpy.einsum('r,ria->ria', 2*frhogamma, sigma1)
                f_ov = numpy.einsum('r,ria->ria', 4*fgg, sigma1)
                f_ov+= numpy.einsum('r,ria->ria', 2*frhogamma, rho_ov[0])
                w_ov[1:] = numpy.einsum('ria,xr->xria', f_ov, rho[1:4])
                w_ov[1:]+= numpy.einsum('r,xria->xria', 2*vgamma, rho_ov[1:4])
                w_ov *= weight[:,None,None]
                iajb = lib.einsum('xria,xrjb->iajb', rho_ov, w_ov) * 2
                a += iajb
                b += iajb

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        elif xctype == 'MGGA':
            raise NotImplementedError('meta-GGA')

    else:
        add_hf_(a, b)

    return a, b


def core_excitation(td, energy_range=None, analyze=True, occidx=None, \
                    viridx=None, REW=False, TDA=True, nstates=None):
    """
    Compute core-excitations by specifying an energy range for solving the TDA
    equation
    ..math::
        A X = \omega X

    Note that we are not truncating the H, but rather truncating the eigenstates.

    Restricted Energy Window (REW): truncate the Hamiltonian according to the
        given energy window :math:`E_{min} <E_a - E_i < E_{max}`.

    Parameters
    ----------
    td : TYPE
        DESCRIPTION.
    ew : list of two floats
        energy window for electronic excitations
    analyze : TYPE, optional
        DESCRIPTION. The default is True.
    rew: boolean
        indicator whether to use REW. This solves an truncated eigenvalue equation.
    Returns
    -------
    w : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    """
    # get TDSCF A, B matrices, this is very expansive, scales O(N^4)
    if occidx is None and viridx is None:
        A, B = td.get_ab()
    else:
        A, B = get_ab_ras(td._scf, occidx=occidx, viridx=viridx) # A, B are matrices of shape [nocc,nvir,nocc,nvir]


    if REW:
        raise NotImplementedError('REW is not implemented yet.')

    nocc, nvir = A.shape[0:2]
    A = A.reshape((nocc * nvir, nocc*nvir))
    B = B.reshape(nocc*nvir,nocc*nvir)

    if TDA:
        if energy_range is not None:

            Ab = banded(A)
            emin, emax = energy_range

            w, v = scipy.linalg.eig_banded(Ab, lower=True, eigvals_only=False,\
                                       select='v', select_range=[emin, emax])
        elif nstates is not None:
            # check if nstates is smaller than the full size
            from scipy.sparse.linalg import eigsh

            assert(nstates < nocc * nvir)

            w, v = eigsh(A, k=nstates, which='SM')

            # raise NotImplementedError('Davidson solver to get N lowest eigenvalues.')

        else:
            w, v = scipy.linalg.eigh(A)

        # Fill the td.xy variable, not necessary
        td.xy = []
        print('nroots = ', len(w))
        for i in range(len(w)):
            td.xy.append([v[:, i].copy().reshape(nocc, nvir), 0])

        if analyze:
            print("\nTDA Transition Amplitude Analysis\n")
            for i in range(len(w)):
                print('Root {}    {:6.4f} Hartree    {:6.4f} eV'.format(i, w[i], w[i] * au2ev))
                print('nocc      nvir     coeff')
                print('------------------------')
                # get the indexes with coeff > 0.1
                idx = np.where(np.abs(v[:,i]) > 0.1)[0]
                for k in idx:
                    coord = np.unravel_index(k, (nocc, nvir))

                    print('{}  --->  {}     {:6.4f}'.format(coord[0], coord[1], v[k, i]))


        return w, v

    else:
        # TDDFT
        e = np.linalg.eig(np.bmat([[A        , B       ],
                                         [-B.conj(),-A.conj()]]))[0]
        lowest_e = np.sort(e[e > 0])[:nstates]
        return lowest_e

# def transition_dipole(td, n):
#     """
#     Compute the transition dipole moment from the ground state to excited
#     states.

#     Parameters
#     ----------
#     td : TYPE
#         DESCRIPTION.
#     n : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     dip : TYPE
#         DESCRIPTION.

#     """

#     mo_coeff = td._scf.mo_coeff # <ao|mo>
#     # mo_occ = td._scf.mo_occ
#     # orbo = mo_coeff[:,mo_occ==2]
#     # orbv = mo_coeff[:,mo_occ==0]

#     cis_t1 = td.xy[n][0] # TDA X matrix in mo basis, shape [nocc, nvir]
#     # TDDFT (X,Y) has X^2-Y^2=1.
#     # Renormalizing X (X^2=1) to map it to CIS coefficients
#     cis_t1 *= 1. / np.linalg.norm(cis_t1)

#     nocc, nvir = cis_t1.shape
#     nmo = nocc + nvir

#     # 1e operator in ao basis
#     ints = td.mol.intor_symmetric('int1e_r', comp=3)
#     # transform to mo basis
#     ints = np.einsum('xpq,pi,qj->xij', ints.reshape(-1,nmo,nmo), \
#                      mo_coeff.conj(), mo_coeff)

#     ints = ints[:, :nocc, nocc:]
#     dip = np.einsum('xij,ij->x', ints, cis_t1)*2 # the factor of 2 comes from spin

#     return dip

def transition_dipole(td, n):
    """
    Compute the transition dipole moment from the ground state to excited
    states.

    We choose the excited state as
    .. maht::

        |\Phi^I \rangle = \sum_{i,a} X^I_{ia} a^\dag i|\Phi_0\rangle

        \mu = \braket{\Phi_I | \mu | \Phi_0}

    Parameters
    ----------
    td : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    dip : TYPE
        DESCRIPTION.

    """

    mo_coeff = td._scf.mo_coeff # <ao|mo>
    mo_occ = td._scf.mo_occ

    # mo_occ = td._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]

    cis_t1 = td.xy[n][0] # TDA X matrix in mo basis, shape [nocc, nvir]
    # TDDFT (X,Y) has X^2-Y^2=1.
    # Renormalizing X (X^2=1) to map it to CIS coefficients
    # cis_t1 *= 1. / np.linalg.norm(cis_t1)

    nocc, nvir = cis_t1.shape

    # 1e operator in ao basis
    ints = td.mol.intor_symmetric('int1e_r', comp=3)

    # transform r to mo basis
    # r_pq = C_up^* r_uv C_vq, we only need p = vir, q = occ
    ints = np.einsum('xuv, ua, vi -> xai', ints, \
                      orbv.conj(), orbo)

    dip = np.einsum('xai,ia->x', ints, cis_t1)*2 # the factor of 2 comes from spin
    # the minus sign comes from electron charge

    return dip


# def _build_tdm(td, J, I=0):
#     """
#     Compute the transition density matrix in MO basis for TDA
#     ..math::
#         T_{ia} = \rangle \Phi_n | i a^\dag |\Phi_0\rangle

#     Parameters
#     ----------
#     td : tdobj
#         e.g. CIS, TDHF, TDDFT
#     n : int/list
#         state id. E.g., n = 1 is the first-excited state.
#         If n is None, it will print out all excited states.

#     Returns
#     -------
#     None.

#     """

#     if I == 0:
#         cis_t1 = td.xy[J][0] # TDA X matrix, shape [nocc, nvir]
#         # TDDFT (X,Y) has X^2-Y^2=1.
#         # Renormalizing X (X^2=1) to map it to CIS coefficients
#         # cis_t1 *= 1. / np.linalg.norm(cis_t1)

#         nocc, nvir = cis_t1.shape


#         # mo_coeff = td._scf.mo_coeff

#         return np.einsum('ui, ia, vj ->uv', mo_coeff[:, :nocc], \
#                          cis_t1, mo_coeff[:, nocc:].conj())


# def _build_tdm_ee(td, J, I):
#     """
#     Compute the transition density matrix in MO basis for TDA
#     ..math::
#         D_{pq} = \langle \Phi_J | p^\dagger q | \Phi_I \rangle

#     Parameters
#     ----------
#     td: tdscf obj
#     J: int
#         bra state id
#     I: int, 0 is the first excited state.
#         ket state id
#     Returns
#     -------

#     """
#     assert(I < J)

#     X1 = td.xy[I][0] # [no, nv]
#     X2 = td.xy[J][0]

#     nocc, nvir = X1.shape
#     nmo = nocc + nvir
#     tdm = np.zeros((nmo, nmo))
#     tdm_oo =-np.einsum('ia,ka->ik', X2.conj(), X1)
#     tdm_vv = np.einsum('ia,ic->ac', X1, X2.conj())

#     tdm[:nocc,:nocc] += tdm_oo * 2
#     tdm[nocc:,nocc:] += tdm_vv * 2

#     # Transform density matrix to AO basis, this is only valid
#     # for non-restricted calculations
#     mo = td._scf.mo_coeff
#     tdm = np.einsum('pi,ij,qj->pq', mo, tdm, mo.conj())
#     return tdm


# class TDA:
#     def __init__(self, td):
#         """
#         TDA core-excitation computations with the FULL ov excitation space

#         Parameters
#         ----------
#         td : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """

#     def core(self, energy_range):
#         pass

class RXS:
    def __init__(self, td, occidx=None, viridx=None):
        """
        Reduced (Single) Excitation Space computations for TDDFT/TDA/TDHF

        Parameters
        ----------
        td : tdscf object
            DESCRIPTION.
        occidx : list of integers, optional
            reduced occupied orbitals taken. The default is None.
        viridx : TYPE, optional
            reduced virtual orbitals to consider. The default is None.

        Returns
        -------
        None.

        """
        self.td = td
        self.occidx = occidx
        self.viridx = viridx
        self.x = None # TDA X coeffs
        self.mo_coeff = td._scf.mo_coeff
        self.mo_occ = td._scf.mo_occ
        self.r_ao = None # position matrix element in AO

    @property
    def occidx(self):
        return self._occidx

    @occidx.setter
    def occidx(self, occidx):
        self._occidx = occidx

    @property
    def viridx(self):
        return self._viridx

    @viridx.setter
    def viridx(self, viridx):
        self._viridx = viridx

    def core_excitation(self, nstates=None, energy_range=None, analyze=True):
        # occidx = self.occidx
        # viridx = self.viridx

        # check if occidx and virdix are provided
        if self.occidx is None:
            self.occidx = (np.where(self.mo_occ==2)[0])
        if self.viridx is None:
            self.viridx = (np.where(self.mo_occ==0)[0])

        nocc = len(self.occidx)
        nvir = len(self.viridx)

        print('Computing core-excitations in reduced excitation space')
        print('occ. orbs = ', self.occidx)
        print('vir. orbs = ', self.viridx)

        w, v = core_excitation(self.td, energy_range=energy_range,\
                               occidx=self.occidx, viridx=self.viridx, \
                                   nstates=nstates, analyze=analyze)

        self.x = v.reshape(nocc, nvir, len(w))

        return w, v

    def get_ab(self):
        return get_ab_ras(self.td._scf, self.occidx, self.viridx)

    def tdm(self, n, representation='mo'):
        """
        Compute the transition density matrix in MO basis

        ..math::

            D_{i a} = \rangle \Phi_n | a^\dag i |\Phi_0\rangle = \
                \overline{X^n_{ia}}

            \Phi_n = X^n_{ia} a^\dag i \ket{|Phi_0}

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # for real X, Y, the 1RDM = X
        return self.x[:,:, n].conj()





    # def transition_dipole_from_tdm(self, tdm):
    #     """
    #     Compute the transition dipole moment from the ground state to excited
    #     states. Using a slightly different method than transition_dipole().

    #     The TDM is build first from x coeff, and then calculate transition
    #     dipole.

    #     Parameters
    #     ----------
    #     tdm : ndarray [nao, nao]
    #         TDM in AO representation.

    #     Returns
    #     -------
    #     dip : ndarray [3]
    #         DESCRIPTION.

    #     """

    #     td = self.td
    #     mo_coeff = self.mo_coeff # <ao|mo>

    #     occidx = self.occidx
    #     viridx = self.viridx

    #     nocc = len(occidx)
    #     # nvir = len(viridx)

    #     # 1e operator in ao basis
    #     ints = td.mol.intor_symmetric('int1e_r', comp=3)

    #     # transform to mo basis
    #     # mo_coeff = mo_coeff[:, [*occidx, *viridx]]

    #     # ints = np.einsum('xpq,pi,qj->xij', ints, \
    #     #                  mo_coeff.conj(), mo_coeff)

    #     # ints = ints[:, :nocc, nocc:]


    #     dip = -np.einsum('xij,ij -> x', ints, tdm)*2 # the factor of 2 comes from spin

    #     return dip

    def transition_dipole(self):
        """
        Compute the transition dipole moment from the ground state to excited
        states.

        Parameters
        ----------
        td : TYPE
            DESCRIPTION.
        n : int
            state id. 0 is the first excited state because the ground state is
            not included in the TDA calculations.

        Returns
        -------
        dip : ndarray [3, nstates]
            DESCRIPTION.

        """

        td = self.td
        mo_coeff = td._scf.mo_coeff # <ao|mo>
        # mo_occ = td._scf.mo_occ
        # orbo = mo_coeff[:,mo_occ==2]
        # orbv = mo_coeff[:,mo_occ==0]

        # TDDFT (X,Y) has X^2-Y^2=1.
        # Renormalizing X (X^2=1) to map it to CIS coefficients
        #cis_t1 *= 1. / np.linalg.norm(cis_t1)

        # nmo = td.mol.nelectron//2

        occidx = self.occidx
        viridx = self.viridx

        # if occidx is None:
        #     occidx = numpy.where(mo_occ==2)[0]
        # if viridx is None:
        #     viridx = numpy.where(mo_occ==0)[0]

        nocc = len(occidx)
        # nvir = len(viridx)

        # 1e operator in ao basis
        ints = td.mol.intor_symmetric('int1e_r', comp=3)
        # transform to mo basis
        mo_coeff = mo_coeff[:, [*occidx, *viridx]]

        ints = np.einsum('xpq,pi,qj->xij', ints, \
                         mo_coeff.conj(), mo_coeff)

        ints = ints[:, :nocc, nocc:]


        dip = np.einsum('xij,ijn->xn', ints, self.x)*2 # the factor of 2 comes from spin

        return dip

    def transition_density_matrix(self):
        pass

    def build_tdm_ee(self, td, J, I):
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
        return tdm_ee(X1, X2, self.td._scf.mo_coeff, self.occidx, self.viridx)


def _denisty_matrix(x, mo_coeff, mo_occ, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states

    adapted from PySCF/tddft/example 22

    '''
    cis_t1 = x
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    # mf = td._scf
    dm = np.diag(mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm


def tdm_ee(X1, X2, mo_coeff, reduced_transition_space=False, occidx=None, viridx=None):
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


def tdm_ee_diff_exc_space(X1, X2, mo_coeff, mo_occ, occidx=None, viridx=None,\
            occidx_bra=None, viridx_bra=None, representation='mo'):
    """
    Compute the transition density matrix in MO representation between two TDA/CIS/TDHF excited
    states with inequivalent excitation spaces.
    ..math::
        D_{qp} = \langle \Phi_2 | p^\dag q | \Phi_1 \rangle

    This only calculates the transition density matrix. For density matrices,
    call dm_ee().

    Parameters
    ----------
    X1 : ndarray [nocc, nvir, nstates], nocc, nvir are the size of the excitation space
        TDA coeff for ket state.
    X2 : TYPE
        TDA coeff for bra state. The shape can be different from X1 when the
        excitation space are different for ket and bra.
    mo_coeff : TYPE
        DESCRIPTION.
    mo_occ : TYPE
        DESCRIPTION.
    occidx : list, optional
        occupied orbitals index for the ket state. The default is None.
    viridx : list of integers, optional
        vir orbs index. The default is None.
    occidx_bra : TYPE, optional
        DESCRIPTION. The default is None.
    viridx_bra : TYPE, optional
        DESCRIPTION. The default is None.
    same_excitation_space : TYPE, optional
        DESCRIPTION. The default is True.
        """
    # occ and vir orbs in total
    occall = np.where(mo_occ==2)[0]
    virall = np.where(mo_occ==0)[0]

    nocc = len(occall)
    nvir = len(virall)

    # expand both reduced TDA amplitudes to the full ov space and then perform
    # calculations
    x1 = np.zeros((nocc, nvir))
    x2 = np.zeros((nocc, nvir))

    if occidx is None:
        occidx = occall
    if viridx is None:
        viridx = virall

    if occidx_bra is None:
        occidx_bra = occall
    if viridx_bra is None:
        viridx_bra = virall

    if X1.shape != (len(occidx), len(viridx)):
        raise ValueError('The X1 shape {} do not match the occ {} and vir\
                         {}.'.format(X1.shape, len(occidx), len(viridx)))

    if X2.shape != (len(occidx_bra), len(viridx_bra)):
        raise ValueError('The X2 shape {} do not match the occ {} and vir\
                         {}.'.format(X2.shape, len(occidx_bra), len(viridx_bra)))

    x1[np.ix_(occidx, viridx-nocc)] = X1
    x2[np.ix_(occidx_bra, viridx_bra - nocc)] = X2

    tdm = np.zeros(mo_coeff.shape) # full MO space

    tdm_oo = -np.einsum('ia,ka->ik', x2.conj(), x1)
    tdm_vv = np.einsum('ia,ic->ac', x1, x2.conj())

    tdm[:nocc,:nocc] += tdm_oo * 2
    tdm[nocc:,nocc:] += tdm_vv * 2

    # Transform density matrix to AO basis
    # return np.einsum('pi,ij,qj->pq', mo_coeff, tdm, mo_coeff.conj())

    if representation == 'mo':
        return tdm
    elif representation == 'ao':
        return mo_coeff @ tdm @ dag(mo_coeff)
    else:
        raise ValueError('There is no {} representation. Try ao or mo.'.format(\
            representation))


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


if __name__=='__main__':

    from pyscf import tddft

    # np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    np.set_printoptions(precision=6, suppress=True)

    mol = gto.Mole()
    mol.build(
        # atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
     """
     O     0.00000000     0.00000000     0.12982363
     H     0.75933475     0.00000000    -0.46621158
     H    -0.75933475     0.00000000    -0.46621158
     """,
        basis = '321g',
        symmetry = True,
    )

    mf = dft.RKS(mol)
    # mf.init_guess='HF.chk'
    mf.xc = 'b3lyp'
    # mf.chkfile = 'HF.chk'
    mf.kernel()
    # pickle.dump(mf, open('mf', 'w'))

    mytd = tddft.TDA(mf)
    mytd.nstates = 5
    mytd.kernel()
    dip = mytd.transition_dipole()
    print('electric dipole \n', dip)

    for j in range(5):
        print(transition_dipole(mytd, j))

    # mytd.analyze()
    # w = core_excitation(mytd, ew=[20, 30])[0]

    ras = RXS(mytd)
    ras.occidx = [0]
    w, v = ras.core_excitation(nstates=4)


    print(ras.transition_dipole())

    tdms = []
    for j in range(4):
        tdm = (tdm_ee(ras.x[:,:,0], ras.x[:,:,j], mf.mo_coeff,\
                  ras.occidx, ras.viridx))
        tdms.append(tdm.copy())

    # tdm2 = (tdm_ee_diff_exc_space(ras.x[:,:,2], ras.x[:,:,4], mf.mo_coeff, mf.mo_occ,\
    #               ras.occidx, ras.viridx, ras.occidx, ras.viridx))

    dip = transition_dipole_from_tdm(mol, tdms, mf.mo_coeff)
    print(dip)

    # wr = core_excitation(mytd, ew=[20, 30], occidx=[0])[0]
    # # print(mytd.xy[0][0].shape)
    # edip = mytd.transition_dipole()
    # print(edip)
    # edip = transition_dipole(mytd, 1)
    # print(edip)
    # print(w*au2ev, '\n', wr*au2ev)
    # with open('td.pkl', 'w') as f:
    # pickle.dump(mytd.xy, open('td.pkl', 'wb'))

    # PySCF-1.6.1 and newer supports the .TDDFT method to create a TDDFT
    # object after importing tdscf module.
    # from pyscf import tddft
    # mytd = mf.TDDFT().run()
    # mytd = mol.RHF().run().TDHF().run()
