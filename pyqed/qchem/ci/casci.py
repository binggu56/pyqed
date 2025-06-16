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
from scipy.sparse.linalg import eigsh
from pyscf.scf import _vhf
import sys
from opt_einsum import contract

from pyqed import dagger, dag, tensor
from itertools import combinations
import warnings

from pyqed.qchem import get_veff
from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct, get_fci_combos, SlaterCondon, CI_H
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''CAS space one-electron hamiltonian

    Args:
        casci : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    hcore = casci.get_hcore()
    energy_core = casci.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        corevhf = get_veff(casci.mol, core_dm)
        energy_core += np.einsum('ij,ji', core_dm, hcore).real
        energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5
    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    return h1eff, energy_core


class CASCI:
    def __init__(self, mf, ncas, nelecas, ncore=None, mu=None):
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

        mu: float
            chemical pontential. The default is None.

        Returns
        -------
        None.

        """
        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        self.ncore = ncore

        if ncas > 10:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = mf.mo_coeff[:,:ncore]
        self.mo_cas = mf.mo_coeff[:,ncore:ncore+ncas]

        self.mf = mf
        self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None
        self.Nu = None
        self.Nd = None
        self.binary = None

    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip
        """
        # from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = h1e_for_cas(mf, ncas=self.ncas, ncore=self.ncore)

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

        eri_aa -= eri_aa.swapaxes(1,3)

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


    def fix_nelec_by_energy_penalty(self, shift=0.1):

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


    def run(self, nstates=1, method='ci'):
        """
        solve the full CI in the active space

        Parameters
        ----------
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.
        method : TYPE, optional
            choose which solver to use. 
            'ci' is the standard CI solver. 
            'jw' is the exact diagonalizaion by Jordan-Wigner transformation.
            The default is 'ci'.

        Returns
        -------
        TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.

        """

        if method == 'ci':
            # FCI solver, more efficient than the JW solver

            mo_occ = [self.mf.mo_occ[self.ncore: self.ncore+self.ncas]//2, ] * 2

            binary = get_fci_combos(mo_occ = mo_occ)

            self.binary = binary

            print('CASCI number of determinants', binary.shape[0])

            H1, H2 = self.get_SO_matrix()

            SC1, SC2 = SlaterCondon(binary)
            H_CI = CI_H(binary, H1, H2, SC1, SC2)


            E, X = eigsh(H_CI, k=nstates, which='SA')

        elif method == 'jw':
            
            # exact diagonalization by JW transform

            H = self.qubitization()
            E, X = eigsh(H, k=nstates, which='SA')
        
        else:
            raise ValueError("There is no {} solver for CASCI. Use 'ci' or 'jw'".format(method))

        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = X

        for i in range(nstates):
            print("Root {} {}".format(i, self.e_tot[i]))

        return self



if __name__ == "__main__":
    from pyqed import Molecule
    from pyqed.qchem.ci.cisd import overlap

    mol = Molecule(atom = [
    ['H' , (0. , 0. , 0)],
    ['H' , (0. , 0. , 1)], ])
    mol.basis = '631g'
    mol.charge = 0
    # mol.unit = 'b'
    mol.build()

    mf = mol.RHF()
    mf.run()

    ncas, nelecas = (4,2)
    casci = CASCI(mf, ncas, nelecas)

    casci.run(2)

    #### test overlap
    mol2 = Molecule(atom = [
    ['H' , (0. , 0. , 0)],
    ['H' , (0. , 0. , 1.4)], ])
    mol2.basis = '631g'

    # mol.unit = 'b'
    mol2.build()

    mf2 = mol2.RHF().run()


    ncas, nelecas = (4,2)
    casci2 = CASCI(mf2, ncas, nelecas).run(2)

    # casci.run()
    S = overlap(casci, casci2)
    print(S)

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