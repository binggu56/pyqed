#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:07 2024

complete active space configuration interaction

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import logging
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from pyscf.scf import _vhf
import sys
from opt_einsum import contract

from pyqed import dagger, dag, tensor
from itertools import combinations
import warnings

from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct
from pyqed.qchem.dvr import RHF, RHF1D, RHF2D



class CASCI:
    def __init__(self, mf, ncas, nelecas=None, mu=None):
        """
        Exact diagonalization (FCI) on the complete active space (CAS) by FCI or
        Jordan-Wigner transformation

        .. math::
            H = h_{ij}c_i^\dagger c_j + v_{pqrs} c_p^\dagger c_q^\dagger c_s c_r\
                -\mu \sum_\sigma c_{i\sigma}^\dag c_{i\sigma}


        From Pyscf: Hartree-Fock orbitals are often poor for systems with significant static correlation.
        In such cases, orbitals from density functional calculations often
        yield better starting points for CAS calculations.

        Parameters
        ----------
        scf : TYPE
            A DFT/HF object.
        nstates : TYPE, optional
            number of excited states. The default is 3.
        ncas : TYPE, optional
            DESCRIPTION. The default is None.
        nelecas : TYPE, optional
            DESCRIPTION. The default is None.

        mu: float
            chemical pontential. The default is None.

        Returns
        -------
        None.

        """
        self.ncas = ncas # number of MOs
        if self.ncas > 10:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(self.ncas))

        self.nstates = None
        if nelecas is None:
            nelecas = mf.mol.nelec

        if nelecas <= 2:
            print('Electrons < 2. Use CIS or CISD instead.')

        self.nelecas = nelecas


        self.mf = mf
        self.mol = mf.mol
        self.chemical_potential = mu

        self.mo_coeff = mf.mo_coeff

    def get_SO_matrix(self, SF=False, H1=None, H2=None):
        """
        Given a PySCF rhf/uhf object get Spin-Orbit one-electron and two-electron H Matrices

        SF: bool
            spin-flip
        """
        from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals

        # if isinstance(mf, (RHF1D, RHF2D, RHF)):
        #     Ca, Cb = [mf.mo_coeff, ] * 2

        Ca, Cb = [mf.mo_coeff, ] * 2


        # print(Ca.shape)
        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        nmo = Ca.shape[1] # n

        eri = mf.eri  # (pq|rs) 1^*12^*2
        eri_aa = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # physicts notation <pq|rs>
        # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

        eri_aa -= eri_aa.swapaxes(1,3) # (pq||rs) = (pq|rs) - (ps|rq)

        eri_bb = eri_aa.copy()

        eri_ab = contract('ip, iq, ij, jr, js->pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
        eri_ba = contract('ip, iq, ij, jr, js->pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




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
        H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca), np.einsum("AB, Ap, Bq -> pq",
        H, Cb, Cb)])

        # if SF:
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

    def qubitization(self, orb='mo'):

        if orb == 'mo':

            # transform the Hamiltonian in DVR set to (truncated) MOs
            # nmo = self.ncas
            mf = self.mf

            # single electron part
            Ca = mf.mo_coeff[:, :self.ncas]
            hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)


            eri = self.mf.eri
            eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

            # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

            self.hcore_mo = hcore_mo

            return self.jordan_wigner(hcore_mo, eri_mo)


        elif orb == 'natural':
            raise NotImplementedError('Nartural orbitals qubitization not implemented.')



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

        return H + (Na - self.nelecas/2 * I) @ (Na - self.nelecas/2 * I) + \
            (Nb - self.nelecas/2 * I) @ (Nb - self.nelecas/2 * I)


    def run(self, nstates=3):
        from pyqed.qchem.ci.fci import SlaterCondon, CI_H

        mf = self.mf
        ncas = self.ncas

        mo_occ = mf.mo_occ[:, :ncas]/2

        mf.mo_coeff = mf.mo_coeff[:, :ncas]

        Binary = get_fci_combos(mo_occ)
        print('Binary shape', Binary.shape)

        # build the 1e and 2e Hamiltonian in MOs

        H1, H2 = self.get_SO_matrix(mf)

        # build the CI Hamiltonain
        SC1, SC2 = SlaterCondon(Binary)
        H_CI = CI_H(Binary, H1, H2, SC1, SC2)

        # print('HCI', H_CI)

        # E, X = np.linalg.eigh(H_CI)
        E, X = eigsh(H_CI, k=nstates, which='SA')

        e_nuc = self.mol.energy_nuc()

        return E + e_nuc, X

    def make_rdm1(self, civec):
        """
        spin-traced 1RDM

        .. math::

            D_{pq} = \sum_{s = \alpha, \beta} \lange \Psi| q^\dag_s p_s | \Psi \rangle

        Returns
        -------
        None.

        """
        mo_coeff = self.mo_coeff

        pass

    def make_rdm2(self):
        pass

# class CAS_JWT(CASCI):

class CASSCF(CASCI):
    """

    Using the OptOrbFCI algorithm to optimize orbitals (better than conventional
                                                        CASSCF algorithm)

    Refs:
        Q. Sun et al. / Chemical Physics Letters 683 (2017) 291–299



    """
    def run(self):
        pass


def get_fci_combos(mo_occ):
    # print(mf.mo_occ.shape)
    O_sp = np.asarray(mo_occ, dtype=np.int8)

    # number of electrons for each spin
    N_s = np.einsum("sp -> s", O_sp)

    N = O_sp.shape[1]
    Λ_α = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[0] ) ) )
    Λ_β = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[1] ) ) )
    ΛA, ΛB = SpinOuterProduct(Λ_α, Λ_β)
    Binary = givenΛgetB(ΛA, ΛB, N)
    return Binary

if __name__ == "__main__":
    pass