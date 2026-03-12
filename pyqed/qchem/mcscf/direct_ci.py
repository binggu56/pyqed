#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:07 2024

complete active space configuration interaction

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import logging
from functools import reduce
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, LinearOperator

import sys
from opt_einsum import contract

from pyqed import tensor
from itertools import combinations
import warnings

from pyqed.qchem import get_veff
from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct, get_fci_combos, SlaterCondon, CI_H
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo

from pyqed.qchem.mcscf.casci import h1e_for_cas, size_of_cas, spin_square
from pyqed.qchem import mcscf

from numba import njit, prange

@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag(H1, H2, Binary):

    n_dets, _, n_mo = Binary.shape


    H1_diag_alpha = np.diag(H1[0])

    H1_diag_beta = np.diag(H1[1])

    # pre caculate H2[p,p,q,q]
    H2_aa_ppqq = np.zeros((n_mo, n_mo))
    H2_bb_ppqq = np.zeros((n_mo, n_mo))
    H2_ab_ppqq = np.zeros((n_mo, n_mo))
    H2_ba_ppqq = np.zeros((n_mo, n_mo))

    for p in range(n_mo):
        for q in range(n_mo):
            H2_aa_ppqq[p, q] = H2[0, 0, p, p, q, q]
            H2_bb_ppqq[p, q] = H2[1, 1, p, p, q, q]
            H2_ab_ppqq[p, q] = H2[0, 1, p, p, q, q]
            H2_ba_ppqq[p, q] = H2[1, 0, p, p, q, q]


    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        # H1 diagonal part
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += H1_diag_alpha[p]
            if Binary[i, 1, p]:
                H_diag[i] += H1_diag_beta[p]

        # H2 diagonal part
        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    if Binary[i, 0, q]:
                        H_diag[i] += H2_aa_ppqq[p, q]/2
                    if Binary[i, 1, q]:
                        H_diag[i] += H2_ab_ppqq[p, q]/2

            if Binary[i, 1, p]:
                for q in range(n_mo):
                    if Binary[i, 1, q]:
                        H_diag[i] += H2_bb_ppqq[p, q]/2
                    if Binary[i, 0, q]:
                        H_diag[i] += H2_ba_ppqq[p, q]/2



    return H_diag

@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_single_excitation(H1_spin, H2_same, H2_cross, a_t, a, ca, binary_complement):

    n_exc = a_t.shape[0]
    n_mo = H1_spin.shape[0]
    H_result = np.zeros(n_exc)

    # pre-calculate H1[p,q]
    H1_matrix = H1_spin

    for k in prange(n_exc):
        h1_term = 0.0
        h2_same_term = 0.0
        h2_cross_term = 0.0


        for p in range(n_mo):
            a_t_val = a_t[k, p]
            if a_t_val == 0:
                continue
            for q in range(n_mo):
                a_val = a[k, q]
                if a_val == 0:
                    continue

                # H1
                h1_term += H1_matrix[p, q] * a_t_val * a_val

                # H2
                for r in range(n_mo):
                    ca_val = ca[k, r]
                    bin_val = binary_complement[k, r]

                    if ca_val != 0:
                        h2_same_term += H2_same[p, q, r, r] * a_t_val * a_val * ca_val

                    if bin_val != 0:
                        h2_cross_term += H2_cross[p, q, r, r] * a_t_val * a_val * bin_val

        H_result[k] = -(h1_term + h2_same_term + h2_cross_term)

    return H_result

@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_double_excitation(H2_tensor, at1, a1, at2, a2):

    n_exc = at1.shape[0]
    n_mo = H2_tensor.shape[0]
    H_result = np.zeros(n_exc)


    for k in prange(n_exc):
        val = 0.0


        p_indices = np.where(at1[k] != 0)[0]
        q_indices = np.where(a1[k] != 0)[0]
        r_indices = np.where(at2[k] != 0)[0]
        s_indices = np.where(a2[k] != 0)[0]


        for p in p_indices:
            at1_val = at1[k, p]
            for q in q_indices:
                a1_val = a1[k, q]
                for r in r_indices:
                    at2_val = at2[k, r]
                    for s in s_indices:
                        a2_val = a2[k, s]
                        val += H2_tensor[p, q, r, s] * at1_val * a1_val * at2_val * a2_val

        H_result[k] = val

    return H_result



def hamiltonian_matrix_elements(Binary, H1, H2, SC1, SC2):


    # slater-condon
    I_A, J_A, a_t, a, I_B, J_B, b_t, b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    n_dets = Binary.shape[0]
    n_mo = Binary.shape[2]

    # diagonal matrix element

    H_diag = _compute_diag(H1, H2, Binary)


    # single excitation
    H_A = np.array([])
    H_B = np.array([])

    if len(I_A) > 0:

        Binary_I_A_complement = Binary[I_A, 1]


        if a_t.ndim == 2:
            H_A = _compute_single_excitation(
                H1[0], H2[0, 0], H2[0, 1],
                a_t, a, ca, Binary_I_A_complement
            )
        else:

            H_A = np.zeros(len(I_A))
            for i in range(len(I_A)):

                pass

        # single_alpha_time = time.time() - single_start
    # else:
    #     single_alpha_time = 0.0

    if len(I_B) > 0:
        # single_start = time.time()
        Binary_I_B_complement = Binary[I_B, 0]

        if b_t.ndim == 2:
            H_B = _compute_single_excitation(
                H1[1], H2[1, 1], H2[1, 0],
                b_t, b, cb, Binary_I_B_complement
            )

        # single_beta_time = time.time() - single_start
    # else:
    #     single_beta_time = 0.0

    # double excitation
    H_AA = np.array([])
    H_BB = np.array([])
    H_AB = np.array([])

    if len(I_AA) > 0:
        # double_start = time.time()

        if isinstance(aa_t, np.ndarray) and aa_t.ndim == 3:

            H_AA = _compute_double_excitation(
                H2[0, 0], aa_t[0], aa[0], aa_t[1], aa[1]
            )
        else:

            H_AA = _compute_double_excitation(H2[0, 0], aa_t, aa, aa_t, aa)

        # double_aa_time = time.time() - double_start
    # else:
    #     double_aa_time = 0.0

    if len(I_BB) > 0:
        # double_start = time.time()
        if isinstance(bb_t, np.ndarray) and bb_t.ndim == 3:
            H_BB = _compute_double_excitation(
                H2[1, 1], bb_t[0], bb[0], bb_t[1], bb[1]
            )
        else:
            H_BB = _compute_double_excitation(H2[1, 1], bb_t, bb, bb_t, bb)

    #     double_bb_time = time.time() - double_start
    # else:
    #     double_bb_time = 0.0

    if len(I_AB) > 0:
        # double_start = time.time()
        H_AB = _compute_double_excitation(H2[0, 1], ab_t, ab, ba_t, ba)
    #     double_ab_time = time.time() - double_start
    # else:
    #     double_ab_time = 0.0


    return H_diag, H_A, H_B, H_AA, H_BB, H_AB


class CASCI(mcscf.casci.CASCI):
    def __init__(self, mf, ncas, nelecas, ncore=None, spin=None, tol=0):
        """
        Exact diagonalization (FCI) on the complete active space (CAS) by FCI or
        Jordan-Wigner transformation

        .. math::
            H = h_{ij}c_i^\dagger c_j + \frac{1}{2} v_{pqrs} c_p^\dagger c_q^\dagger c_s c_r\
                -\mu \sum_\sigma c_{i\sigma}^\dag c_{i\sigma}


        From Pyscf: Hartree-Fock orbitals are often poor for systems with significant static correlation.
        In such cases, orbitals from density functional calculations often
        yield better starting points for CAS calculations.

        Parameters
        ----------
        mf : TYPE
            A DFT/HF object.
        nstates : TYPE, optional
            number of excited states. The default is 3.
        ncas : TYPE, optional
            DESCRIPTION. The default is None.
        nelecas : TYPE, optional
            DESCRIPTION. The default is None.
        etol: energy convergence for diagonalization

        mu: float
            chemical pontential. The default is None.

        Returns
        -------
        None.

        """
        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        assert(ncore >= 0)

        self.ncore = ncore

        if ncas > 10:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = None
        self.mo_cas = None

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin
        self.shift = None
        self.ss = None

        self.mf = mf
        # self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.Nu = None
        self.Nd = None
        self.binary = None
        self.SC1 = None # SlaterCondon rule 1
        self.SC2 = None # SlaterCondon rule 2
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space

        self.spin_purification = False

        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None
        
        self.tol = tol


    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip

        Returns
        -------
        H1: list of [h1e_a, h1e_b]
        H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
        """
        # from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = h1e_for_cas(mf, ncas=self.ncas, ncore=self.ncore, \
                                     mo_coeff=self.mo_coeff)

        self.e_core = energy_core


        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        # H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        # nmo = Ca.shape[1] # n

        eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

        ### compute SO ERIs (MO)
        eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # physicts notation <pq|rs>
        # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

        # eri_aa -= eri_aa.swapaxes(1,3)

        eri_bb = eri_aa.copy()

        eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
        eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




        # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
        #                         compact=False)).reshape((n,n,n,n), order="C")
        # eri_aa -= eri_aa.swapaxes(1,3)

        # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # eri_bb -= eri_bb.swapaxes(1,3)

        # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
        # compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

        # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
                         # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
        H1 = [H, H]

        if spin_flip:
            raise NotImplementedError('Spin-flip matrix elements not implemented yet')
        #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
        #     return H1, H2, H2_SF
        # else:
        #     return H1, H2
        return H1, H2


    def natural_orbitals(self, dm, nco=None):
        natural_orb_occ, natural_orb_coeff = np.linalg.eigh(dm)

        return natural_orb_occ, natural_orb_coeff

    def size(self, basis='sd', S=0):

        return size_of_cas(self.ncas, self.nelecas)

    def qubitization(self, orb='mo'):

        if orb == 'mo':

            # transform the Hamiltonian in DVR set to (truncated) MOs
            # nmo = self.ncas
            mf = self.mf

            # single electron part
            Ca = mf.mo_coeff[:, self.ncore:self.ncore + self.ncas]
            # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)

            h1eff, e_core = h1e_for_cas(self.mf, ncas=self.ncas, ncore=self.ncore)

            self.e_core = e_core

            eri = self.mf.eri
            eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca,
                              eri, Ca.conj(), Ca)

            # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

            # self.hcore_mo = hcore_mo

            return self.jordan_wigner(h1eff, eri_mo)


        elif orb == 'natural':
            raise NotImplementedError('Nartural orbitals qubitization not implemented.')


    def fix_nelec(self, shift=0.1):
        """
        fix the electron number by energy penalty.
        This is only needed for JW solver without symmetry.

        Parameters
        ----------
        shift : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        """

        Na = self.Nu
        Nb = self.Nd

        I = tensor(Is(self.ncas))

        self.H += shift * ((Na - self.nelecas/2 * I) @ (Na - self.nelecas/2 * I) + \
            (Nb - self.nelecas/2 * I) @ (Nb - self.nelecas/2 * I))

    def jordan_wigner(self, h1e, v):
        """
        MOs based on Restricted HF calculations

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        # an inefficient implementation without consdiering any syemmetry


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

        self.Nu = Na
        self.Nd = Nb


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


        self.H = H
        return H

    def fix_spin(self, s=None, ss=0, shift=0.2):
        """
        fix the spin by energy penalty

        .. math::

            H = H + \mu (\hat{S}^2 - S(S+1))

        Parameters
        ----------
        s : TYPE, optional
            DESCRIPTION. The default is None.
        ss : TYPE, optional
            DESCRIPTION. The default is None.
        shift : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        None.

        """
        if s is None:
            s = (np.sqrt(4*ss+1)-1)/2
            if not np.isclose(2*s, round(2*s)):
                raise Warning("s = {} inconsistant spin value".format(s))
        else:
            if ss is None:
                ss = s * (s+1)
            else:
                raise ValueError('s and ss cannot be specified simulaneously.')

        if ss == 0:
            # first-order spin penalty J. Phys. Chem. A 2022, 126, 12, 2050–2060
            # H' = H + J \hat{S}^2

            self.ss = ss
            self.shift = shift
            self.spin_purification = True

            return self


        else:
            # second-order spin penalty
            raise NotImplementedError('Second-order spin panelty not implemented.')


    def run(self, nstates=1, mo_coeff=None, method='direct_ci', ci0=None):
        """
        solve the full CI in the active space

        Parameters
        ----------
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.
        mo : CAS MOs
            Default is canonical MOs.
        method : TYPE, optional
            choose which solver to use.
            'ci' is the standard CI solver.
            'jw' is the exact diagonalizaion by Jordan-Wigner transformation.
            The default is 'ci'.

        TODO: spin

        Returns
        -------
        TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.

        """
        # print('------------------------------')
        # print("             CASCI              ")
        # print('------------------------------\n')
        self.nstates = nstates

        if method == 'ci':

            # define the core and active space orbitals
            if mo_coeff is None:
                self.mo_coeff = self.mf.mo_coeff # use HF MOs
            else:
                self.mo_coeff = mo_coeff

            ncore = self.ncore
            ncas = self.ncas

            self.mo_core = self.mo_coeff[:,:ncore]
            self.mo_cas = self.mo_coeff[:,ncore:ncore+ncas]

            # FCI solver, more efficient than the JW solver

            mo_occ = [self.mf.mo_occ[ncore: ncore+ncas]//2, ] * 2
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary


            # print('Number of determinants', binary.shape[0])

            h1e, h2e = self.get_SO_matrix()

            if self.spin_purification:

                logging.info('Purify spin by energy penalty')

                # if self.shift is not None:
                # H1, H2 = self.fix_spin(H1, H2, ss=ss, shift=shift)
                shift = self.shift

                norb = self.ncas
                h1e = [h + 3./4 * shift * np.eye(norb) for h in h1e]

                for p in range(norb):
                    for q in range(norb):
                        h2e[:, :, p, q, q, p] -=  0.5 * shift * 2
                        # h2e[1, 1, p, q, q, p] -=  0.5 * shift
                        # h2e[0, 1, p, q, q, p] -=  0.5 * shift
                        # h2e[1, 0, p, q, q, p] -=  0.5 * shift

                        # h2e[0, 0, p, p, q, q] -= 0.25 * shift
                        # h2e[1, 1, p, p, q, q] -= 0.25 * shift

                        # h2e[0, 1, p, p, q, q] -= 0.25 * shift
                        # h2e[1, 0, p, p, q, q] -= 0.25 * shift


                        h2e[:, :, p, p, q, q] -= 0.25 * shift * 2

            h2e[0,0] -= h2e[0,0].swapaxes(1,3)
            h2e[1,1] -= h2e[1,1].swapaxes(1,3)


            self.hcore = h1e

            SC1, SC2 = SlaterCondon(binary)

            self.SC1 = SC1
            self.SC2 = SC2
            self.eri_so = h2e

            I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

            H_CI = CI_H(binary, h1e, h2e, SC1, SC2)


            E, X = eigsh(H_CI, k=nstates, which='SA')
            
            # from pyqed.davidson import davidson
            
            # E, X = davidson(H_CI, nstates, tol=1e-13)

        elif method == 'direct_ci':

            # define the core and active space orbitals
            if mo_coeff is None:
                self.mo_coeff = self.mf.mo_coeff # use HF MOs
            else:
                self.mo_coeff = mo_coeff

            ncore = self.ncore
            ncas = self.ncas

            self.mo_core = self.mo_coeff[:,:ncore]
            self.mo_cas = self.mo_coeff[:,ncore:ncore+ncas]

            # FCI solver, more efficient than the JW solver
            if self.binary is None:
                mo_occ = [self.mf.mo_occ[ncore: ncore+ncas]//2, ] * 2
                binary = get_fci_combos(mo_occ = mo_occ)

                SC1, SC2 = SlaterCondon(binary)

                self.SC1 = SC1
                self.SC2 = SC2
                self.binary = binary

            else:
                binary = self.binary
                SC1 = self.SC1
                SC2 = self.SC2


            h1e, h2e = self.get_SO_matrix()

            if self.spin_purification:
                logging.info('Purify spin by energy penalty')

                # if self.shift is not None:
                # H1, H2 = self.fix_spin(H1, H2, ss=ss, shift=shift)
                shift = self.shift

                norb = self.ncas
                h1e = [h + 3./4 * shift * np.eye(norb) for h in h1e]

                for p in range(norb):
                    for q in range(norb):
                        h2e[:, :, p, q, q, p] -=  0.5 * shift * 2
                        h2e[:, :, p, p, q, q] -= 0.25 * shift * 2

            h2e[0,0] -= h2e[0,0].swapaxes(1,3)
            h2e[1,1] -= h2e[1,1].swapaxes(1,3)



            self.hcore = h1e
            self.eri_so = h2e

            H_diag, H_A, H_B, H_AA, H_BB, H_AB = hamiltonian_matrix_elements(binary, h1e, h2e, SC1, SC2)

            def mv(c):
                return sigma(SC1, SC2, H_diag, H_A, H_B, H_AA, H_BB, H_AB, c)

            H = LinearOperator((binary.shape[0], binary.shape[0]), matvec=mv)

            E, X = eigsh(H, k=nstates, which='SA', tol=self.tol)
            



        elif method == 'jw':


            # exact diagonalization by JW transform

            H = self.qubitization()
            E, X = eigsh(H, k=nstates, which='SA')

        else:
            raise ValueError("There is no {} solver for CASCI. Use 'ci' or 'jw'".format(method))

        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(nstates)]

        for i in range(nstates):
            ss = spin_square(*self.make_rdm12(i))
            print("CASCI Root {}  E = {:.10f}  S^2 = {:.6f}".format(i, self.e_tot[i], ss))

        return self

    def make_rdm1_contract(self, state_id, h1e=None, representation='ao'):
        """
        spin-traced 1e reduced density matrix
        .. math::

            \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>


        Returns
        -------
        None.

        """

        ci = self.ci[state_id]
        if representation.lower() == 'ao':
            C = self.mf.mo_coeff
            h1e = ao2mo(h1e, C)

        ncore = self.ncore
        ncas = self.ncas

        if ncore > 0:
            c_core = 2 * np.trace(h1e[:ncore,:ncore])
        else:
            c_core = 0

        h1e = h1e[ncore:ncas+ncore, ncore:ncas+ncore]

        c_cas = contract_with_rdm1(ci, self.binary, self.SC1, h1e=h1e)

        return c_core + c_cas

    def make_rdm1(self, state_id, with_core=False, with_vir=False, representation='mo'):
        """
        spin-traced 1e reduced density matrix
        .. math::

            \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

        Parameters
        ----------
        representation: str
            indicate which representation RDM is defined. Default 'mo'.

        Returns
        -------
        None.

        """

        ci = self.ci[state_id]
        # if representation.lower() == 'ao':
        #     C = self.mf.mo_coeff
        #     h1e = ao2mo(h1e, C)

        ncore = self.ncore
        ncas = self.ncas
        nmo = self.mf.nmo

        # if ncore > 0:
        #     c_core = 2 * np.trace(h1e[:ncore,:ncore])
        # else:
        #     c_core = 0
        if with_core and not with_vir:

            norb = ncas + ncore
            D = np.zeros((norb, norb), dtype=float)

            if ncore > 0:
                for i in range(ncore):
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            return D

        if with_core and with_vir:

            D = np.zeros((nmo, nmo), dtype=float)
            if ncore > 0:
                for i in range(ncore):
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            return D
        else:
            return make_rdm1(ci, self.binary, self.SC1)

    def make_rdm1s(self, state_id):
        """
        spin-polarized 1e reduced density matrix
        .. math::

            \gamma_s[p,q] = <q_s^\dagger p_s>


        Returns
        -------
        None.

        """

        raise NotImplementedError()

    def make_rdm2(self, state_id=0, with_core=False, with_vir=False):
        """
        2-e reduced density matrix

        The definition follows the PySCF convention.
        .. math::

            \Gamma[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

        with this convention, the energy is computed as

        E = einsum('pqrs,pqrs', eri, rdm2)/2

        Returns
        -------
        None.

        """
        ci = self.ci[state_id]


        if with_core: # we probably never need this!

            ncore = self.ncore
            ncas = self.ncas
            # nmo = self.mf.nmo
            nmo = ncore + ncas

            D = np.zeros((nmo, nmo, nmo, nmo))

            assert ncore > 0

            # cccc block
            I = np.eye(ncore)
            D[:ncore, :ncore, :ncore, :ncore] = 4 * contract('ij, kl -> ijkl', I, I) - 2 * contract('ps, rq -> pqrs', I, I)

            # ccaa block
            dm1 = self.make_rdm1(state_id)

            for i in range(ncore):
                D[i, i, ncore:ncore+ncas, ncore:ncore+ncas] = 2*dm1
                D[ncore:ncore+ncas, ncore:ncore+ncas, i, i] = 2*dm1
                D[i, ncore:ncore+ncas, i, ncore:ncore+ncas] = -dm1
                D[ncore:ncore+ncas, i, ncore:ncore+ncas, i] = -dm1

            D[ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas]=\
                make_rdm2(ci, self.binary, self.SC1, self.SC2)

            return D

        else: #active space DM

            return make_rdm2(ci, self.binary, self.SC1, self.SC2)


    def contract_with_rdm2(self, h2e, state_id=0):

        if h2e.ndim == 4: # spin-free operator
            h2e = np.einsum('IJ, pqrs -> IJpqrs', np.ones((2,2)), h2e)

        return contract_with_rdm2(self.ci[state_id], h2e, self.binary, self.SC1, self.SC2)



    def make_rdm12(self, state_id, with_core=False):
        dm1 = self.make_rdm1(state_id, with_core=with_core)
        dm2 = self.make_rdm2(state_id, with_core=with_core)

        return dm1, dm2

    def spin_square(self, state_id=0):
        pass



    def dump(self, fname):
        import pickle

        with open(fname, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return

    def overlap(self, other):
        return overlap(self, other)

    def contract_with_tdm1(self, bra_id, ket_id=0, h1e=None, representation='mo'):
        """
        spin-traced 1e transition density matrix

        .. math::

            \gamma_{pq}^{\beta \alpha} = <\Psi_\beta | \hat{E}_{qp} | \Psi_\alpha >

        E_{qp} = q_alpha^\dagger p_alpha + q_beta^\dagger p_beta

        Parameters
        ----------
        bra_id : TYPE
            DESCRIPTION.
        ket_id : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """

        if bra_id == ket_id:

            print("CI ket and bra are the same. Computing 1e RDM instead.")
            return self.make_rdm1(ket_id, h1e)

        else:

            if representation.lower() == 'ao':
                C = self.mf.mo_coeff
                h1e = ao2mo(h1e, C)

            ncore = self.ncore
            ncas = self.ncas

            if ncore > 0:
                c_core = 2 * np.trace(h1e[:ncore,:ncore])
            else:
                c_core = 0

            h1e = h1e[ncore:ncas+ncore, ncore:ncas+ncore]


            c_cas = make_tdm1(self.ci[bra_id], self.ci[ket_id], self.binary, self.SC1, h1e)

        return c_cas + c_core

    def make_tdm1(self, bra_id, ket_id=0):
        """
        TDM

        Parameters
        ----------
        bra_id : TYPE
            DESCRIPTION.
        ket_id : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]

        return make_tdm1(cibra, ciket, self.binary, self.SC1)

    def make_tdm2(self, bra_id, ket_id=0):
        """
        spin-traced 1e transition density matrix in MO

        .. math::

            \gamma_{pq}^{\beta \alpha} = <\Psi_\beta | \hat{E}_{qp} | \Psi_\alpha >

        E_{qp} = q_alpha^\dagger p_alpha + q_beta^\dagger p_beta
        """
        raise NotImplementedError('TDM not implemented')





def sigma(SC1, SC2, H_diag, H_A, H_B, H_AA, H_BB, H_AB, c):
    """
    Avoid explicitly construct the CI Hamiltonian Matrix

    math: Hc = sigma

    GIVEN: H1 (1-body Hamtilonian)
    H2 (2-body Hamtilonian)

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    # # sum of MO energies I: configuration index, S: spin index, p: MO index
    # H_diag = np.einsum("Spp, ISp -> I", H1, Binary, optimize=True)

    # # ERI
    # H_diag += np.einsum("STppqq, ISp, ITq -> I", H2, Binary, Binary, optimize=True)/2
    # # print('Hdiag',H_diag.shape)
    sigma_vec = H_diag * c
    # print('sigma_vec shape', sigma_vec.shape)

    ## Rule 1
    # H_A = -np.einsum("pq, Kp, Kq -> K", H1[0], a_t, a, optimize=True)
    # H_A -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,0], a_t, a, ca, optimize=True)
    # H_A -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,1], a_t, a, Binary[I_A,1],
    # optimize=True)

    # print('HA',H_A.shape)

    # for idx, (i,j) in enumerate(zip(I_A, J_A)):
    #     sigma_vec[i] += H_A[idx] * c[j]
    c_J_A = c[J_A]
    contributions_A = H_A * c_J_A
    if len(I_A) > 1000:
        unique_I = np.unique(I_A)
        bincount_result = np.bincount(I_A, weights=contributions_A, minlength=len(sigma_vec))
        sigma_vec += bincount_result
    else:
        np.add.at(sigma_vec, I_A, contributions_A)
    # H_B = -np.einsum("pq, Kp, Kq -> K", H1[1], b_t, b, optimize=True)
    # H_B -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,1], b_t, b, cb, optimize=True)
    # H_B -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,0], b_t, b, Binary[I_B,0],
    # optimize=True)

    # for idx, (i,j) in enumerate(zip(I_B, J_B)):
    #     sigma_vec[i] += H_B[idx] * c[j]
    c_J_B = c[J_B]
    contributions_B = H_B * c_J_B
    if len(I_B) > 1000:
        bincount_result = np.bincount(I_B, weights=contributions_B, minlength=len(sigma_vec))
        sigma_vec += bincount_result
    else:
        np.add.at(sigma_vec, I_B, contributions_B)

    if len(I_AA) > 0:
    ## Rule 2
        # H_AA = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,0], aa_t[0], aa[0],
        # aa_t[1], aa[1], optimize=True)
        # for idx, (i,j) in enumerate(zip(I_AA, J_AA)):
        #     sigma_vec[i] += H_AA[idx] * c[j]
        c_J_AA = c[J_AA]
        contributions_AA = H_AA * c_J_AA
        if len(I_AA) > 1000:
            bincount_result = np.bincount(I_AA, weights=contributions_AA, minlength=len(sigma_vec))
            sigma_vec += bincount_result
        else:
            np.add.at(sigma_vec, I_AA, contributions_AA)

    if len(I_BB) > 0:
        # H_BB = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[1,1], bb_t[0], bb[0],
        # bb_t[1], bb[1], optimize=True)
        # for idx, (i,j) in enumerate(zip(I_BB, J_BB)):
        #     sigma_vec[i] += H_BB[idx] * c[j]
        c_J_BB = c[J_BB]
        contributions_BB = H_BB * c_J_BB
        if len(I_BB) > 1000:
            bincount_result = np.bincount(I_BB, weights=contributions_BB, minlength=len(sigma_vec))
            sigma_vec += bincount_result
        else:
            np.add.at(sigma_vec, I_BB, contributions_BB)

    # H_AB = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,1], ab_t, ab, ba_t, ba,
    #     optimize=True)
    # for idx, (i,j) in enumerate(zip(I_AB, J_AB)):
    #     sigma_vec[i] += H_AB[idx] * c[j]
    if len(I_AB) > 0:
        c_J_AB = c[J_AB]
        contributions_AB = H_AB * c_J_AB
        if len(I_AB) > 1000:
            bincount_result = np.bincount(I_AB, weights=contributions_AB, minlength=len(sigma_vec))
            sigma_vec += bincount_result
        else:
            np.add.at(sigma_vec, I_AB, contributions_AB)

    # print('sigma_shape',sigma_vec.shape)

    return sigma_vec





# def fcisolver(mo_occ):
#     # mo_occ = [self.mf.mo_occ[ncore: ncore+ncas]//2, ] * 2
#     binary = get_fci_combos(mo_occ = mo_occ)
#     # self.binary = binary

#     print('Number of determinants', binary.shape[0])

#     H1, H2 = get_SO_matrix()

#     SC1, SC2 = SlaterCondon(binary)


#     I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

#     H_CI = CI_H(binary, H1, H2, SC1, SC2)

#     E, X = eigsh(H_CI, k=nstates, which='SA')

#     return E, X




def contract_with_tdm1(cibra, ciket, binary, SC1, h1e):
    """

    1e transition DM contracted with 1e operators

    .. math::

        \langle \Phi_I  O_{pq} p^\dagger q | \Phi_J \rangle = O_{pq} A^{IJ}_{qp}}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """

    if isinstance(h1e, np.ndarray): # spin-independent 1e operator
        h1e = [h1e, h1e]

    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

    # sum of MO energies
    H = np.einsum("Spp, ISp -> I", h1e, binary, optimize=True)
    H = np.diag(H)

    ## Rule 1
    H[I_A, J_A] -= np.einsum("pq, Kp, Kq -> K", h1e[0], a_t, a, optimize=True)
    H[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", h1e[1], b_t, b, optimize=True)


    return np.einsum('I, IJ, J -> ', cibra.conj(), H, ciket)

def contract_with_rdm1(ci, binary, SC1, h1e):
    """

    make 1e RDM contracted with 1e operators without returning RDM

    .. math::
        \Tr{ O D} = O_{pq} D_{qp}} = O_{pq} \hat{E}_{pq}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """

    if isinstance(h1e, np.ndarray): # spin-independent 1e operator
        h1e = [h1e, h1e]

    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    H = np.einsum("Spp, ISp -> I", h1e, binary, optimize=True)
    H = np.diag(H)

    ## Rule 1
    H[I_A, J_A] -= np.einsum("pq, Kp, Kq -> K", h1e[0], a_t, a, optimize=True)
    H[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", h1e[1], b_t, b, optimize=True)


    return np.einsum('I, IJ, J -> ', ci.conj(), H, ci)

def make_rdm1(ci, binary, SC1):
    """

    make spin-traced 1e RDM E_{pq}

    .. math::

        \hat{E}_{pq}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """


    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    # H = np.einsum("ISp -> Ip", binary, optimize=True)
    # H = binary[:, 0, :] + binary[:, 1, :]

    # H = np.diag(H)

    nsd, _, nmo = binary.shape
    H = np.zeros((nsd, nsd, nmo, nmo))
    for I in range(nsd):
        for p in range(nmo):
            H[I, I, p, p] = sum(binary[I, :, p])

    ## Rule 1
    H[I_A, J_A] -= np.einsum("Kp, Kq -> Kpq", a_t, a, optimize=True)
    H[I_B, J_B] -= np.einsum("Kp, Kq -> Kpq", b_t, b, optimize=True)


    return np.einsum('I, IJpq, J -> pq', ci.conj(), H, ci).T

def make_tdm1(cibra, ciket, binary, SC1):
    """

    make spin-traced 1e TDM E_{pq}

    .. math::

        \braket{I|\hat{E}_{pq}|J}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """


    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    # H = np.einsum("ISp -> Ip", binary, optimize=True)
    # H = binary[:, 0, :] + binary[:, 1, :]

    # H = np.diag(H)

    nsd, _, nmo = binary.shape
    H = np.zeros((nsd, nsd, nmo, nmo))
    for I in range(nsd):
        for p in range(nmo):
            H[I, I, p, p] = sum(binary[I, :, p])

    ## Rule 1
    H[I_A, J_A] -= np.einsum("Kp, Kq -> Kpq", a_t, a, optimize=True)
    H[I_B, J_B] -= np.einsum("Kp, Kq -> Kpq", b_t, b, optimize=True)


    return np.einsum('I, IJpq, J -> pq', cibra.conj(), H, ciket)





def contract_with_rdm2(ci, H2, Binary, SC1, SC2):
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    # # sum of MO energies I: configuration index, S: spin index, p: MO index
    # H_CI = np.einsum("Spp, ISp -> I", H1, Binary, optimize=True)

    # ERI
    H_CI = np.einsum("STppqq, ISp, ITq -> I", H2, Binary, Binary, optimize=True)/2
    H_CI = np.diag(H_CI)

    ## Rule 1
    # H_CI[I_A , J_A ] -= np.einsum("pq, Kp, Kq -> K", H1[0], a_t, a, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,0], a_t, a, ca, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,1], a_t, a, Binary[I_A,1],
    optimize=True)

    # H_CI[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", H1[1], b_t, b, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,1], b_t, b, cb, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,0], b_t, b, Binary[I_B,0],
    optimize=True)

    if len(I_AA) > 0:
    ## Rule 2
        H_CI[I_AA, J_AA] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,0], aa_t[0], aa[0],
        aa_t[1], aa[1], optimize=True)

    if len(I_BB) > 0:
        H_CI[I_BB, J_BB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[1,1], bb_t[0], bb[0],
        bb_t[1], bb[1], optimize=True)

    H_CI[I_AB, J_AB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,1], ab_t, ab, ba_t, ba,
        optimize=True)

    return np.einsum('I, IJ, J -> ', ci.conj(), H_CI, ci)

def make_rdm2(ci, Binary, SC1, SC2):
    """
    build the spin-traced 2-particle operator with the 2e RDM

    .. math::

        \Gamma_{pqrs} = \sum_{\sigma, \tau} p^\dagger_\sigma r^\dagger_\tau s_\tau q_\sigma

    TODO: fix it

    Params
    ------
    Binary: binary string (I, s, p)
        I: configuration index, S: spin index, p: MO index

    Refs
    ----
    J. Chem. Theory Comput. 2022, 18, 6690−6699

    """
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    nsd, _, nmo = Binary.shape
    I = np.eye(nmo)

    H_CI = np.zeros((nsd, nsd, nmo, nmo, nmo, nmo)) # slow implementation

    # diagonal elements
    D = np.einsum("I, ISp, ITr, pq, rs -> pqrs", np.abs(ci)**2, Binary, Binary, I, I, optimize=True)
    D -= np.einsum("I, ISp, ISr, ps, rq -> pqrs", np.abs(ci)**2, Binary, Binary, I, I, optimize=True)

    ## Rule 1
    H_CI[I_A , J_A ] = -2 * np.einsum("Kp, Kq, Kr, rs -> Kpqrs",  a_t, a, ca, I, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("Kp, Kq, Kr, rs -> Kpqrs", a_t, a, Binary[I_A,1], I, optimize=True)

    H_CI[I_B , J_B ] -= 2 * np.einsum("Kp, Kq, Kr, rs -> Kpqrs", b_t, b, cb, I, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("Kp, Kq, Kr, rs -> Kpqrs", b_t, b, Binary[I_B,0], I, optimize=True)

    ## Rule 2
    if len(I_AA) > 0:

        H_CI[I_AA, J_AA] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", aa_t[0], aa[0],
        aa_t[1], aa[1], optimize=True)

    if len(I_BB) > 0:
        H_CI[I_BB, J_BB] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", bb_t[0], bb[0],
        bb_t[1], bb[1], optimize=True)

    H_CI[I_AB, J_AB] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", ab_t, ab, ba_t, ba,
        optimize=True)

    D += contract('I, IJpqrs, J -> pqrs', ci.conj(), H_CI, ci)

    return D

def overlap(cibra, ciket, s=None):
    """
    CASCI electronic overlap matrix

    The MO overlap is a block matrix

    for Restricted calculation only! (spin unpolarized.)

    TODO: unrestricted HF.

    S = [S_CC, S_CA]
        [S_AC, S_AA]



    Compute the overlap between Slater determinants first
    and contract with CI coefficients

    Parameters
    ----------
    cibra : TYPE
        DESCRIPTION.
    binary1 : TYPE
        DESCRIPTION.
    ciket : TYPE
        DESCRIPTION.
    binary2 : TYPE
        DESCRIPTION.
    s : TYPE
        AO overlap.

    Returns
    -------
    None.

    """
    # nstates = len(cibra) + 1

    # overlap matrix between MOs at different geometries
    if s is None:

        from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

        s = overlap_integral_asymmetric(cibra.mol._bas, ciket.mol._bas)
        s = reduce(np.dot, (cibra.mf.mo_coeff.T, s, ciket.mf.mo_coeff))


    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    S = np.zeros((nsd_bra, nsd_ket)) # overlap between determinants

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore

    scc = s[:ncore_bra, :ncore_ket]
    sca = s[:ncore_bra, ncore_ket:]
    sac = s[ncore_bra:, :ncore_ket]
    saa = s[ncore_bra:, ncore_ket:]

    scc_det = np.linalg.det(scc)
    scc_inv = np.linalg.inv(scc)

    for I in range(nsd_bra):
        occidx1_a  = [i for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
        occidx1_b  = [i for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

        for J in range(nsd_ket):
            occidx2_a  =  [i for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
            occidx2_b  =  [i for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.
            saa_occ_a = saa[np.ix_(occidx1_a, occidx2_a)]
            sca_occ_a = sca[:, occidx2_a]
            sac_occ_a = sac[occidx1_a, :]

            saa_occ_b = saa[np.ix_(occidx1_b, occidx2_b)]
            sca_occ_b = sca[:, occidx2_b]
            sac_occ_b = sac[occidx1_b, :]


            S[I, J] = scc_det**2 * np.linalg.det(saa_occ_a - sac_occ_a @ scc_inv @ sca_occ_a)*\
                np.linalg.det(saa_occ_b - sac_occ_b @ scc_inv @ sca_occ_b)



    # core_bra = list(range(cibra.ncore))
    # core_ket = list(range(ciket.ncore))




    # for I in range(nsd_bra):
    #     occidx1_a  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
    #     occidx1_b  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

    #     for J in range(nsd_ket):
    #         occidx2_a  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
    #         occidx2_b  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

    #         # print('b', occidx2_a, occidx2_b)
    #         # print(ciket.binary[J])

    # # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.

    #         S[I, J] = np.linalg.det(s[np.ix_(occidx1_a, occidx2_a)]) * \
    #                   np.linalg.det(s[np.ix_(occidx1_b, occidx2_b)])


    return contract('BI, IJ, AJ -> BA', np.array(cibra.ci).conj(), S, np.array(ciket.ci))

if __name__ == "__main__":
    from pyqed import Molecule
    from pyqed.qchem.ci.cisd import overlap
    import time

    # mol = Molecule(atom = [
    # ['Li' , (0. , 0. , 0)],
    # ['F' , (0. , 0. , 1)], ])

    # mol.basis = '631g'
    # mol.charge = 0

    # mol.molecular_frame()
    # print(mol.atom_coords())

    # nstates = 3
    # Rs = np.linspace(1,4,4)
    # E = np.zeros((nstates, len(Rs)))

    # for R in Rs:

    #     atom = [
    #     ['Li' , (0. , 0. , 0)],
    #     ['F' , (0. , 0. , R)]]

    #     mol = Molecule(atom, basis='631g')

    #     mol.build()

    #     mf = mol.RHF()
    #     mf.run()

    #     ncas, nelecas = (4,2)
    #     casci = CASCI(mf, ncas, nelecas)

    #     casci.run(nstates)

    #     casci.e_tot

    #### test overlap

    start_time = time.time()

    # mol2 = Molecule(atom = [
    #     ['H' , (0. , 0. , 0)],
    #     ['H' , (0. , 0. , 1)],
    #     ['H' , (0. , 0. , 2)],
    #     ['H' , (0. , 0. , 3)],
    #     ['H' , (0. , 0. , 4)],
    #     ['H' , (0. , 0. , 5)]])

    mol2 = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1.4)]])

    mol2.basis = '631g'

    # mol.unit = 'b'
    mol2.build()

    mf2 = mol2.RHF().run()


    ncas, nelecas = (4,4)
    # from pyqed.qchem import mcscf
    mc = CASCI(mf2, ncas, nelecas)
    mc.fix_spin()

    mc.run(3, method='ci')

    # mc = CASCI(mf2, ncas, nelecas)
    # mc.run(3, mo_coeff=mf2.mo_coeff, purify_spin=True, shift=0.3)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time: {execution_time:.6f} seconds")

    # casci.run()
    # S = overlap(casci, casci2)
    # print(S)

    ### pyscf
    # from pyscf import gto, mp, mcscf
    # # mol = gto.M(
    # #     atom = 'O 0 0 0; O 0 0 1.2',
    # #     basis = 'ccpvdz',
    # #     spin = 2)

    # mol = gto.M(atom = [
    # ['H' , (0. , 0. , 0)],
    # ['Li' , (0. , 0. , 1)], ], unit='b')
    # mol.basis = 'sto3g'
    # myhf = mol.RHF().run()
    # # Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
    # # mymp = mp.UMP2(myhf).run()

    # # noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)

    # mycas = mcscf.CASCI(myhf, ncas, nelecas)
    # mycas.nroots = 4
    # mycas.run()