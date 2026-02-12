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
import logging

from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct
from pyqed.qchem.mcscf.casci import make_rdm1
from pyqed.qchem.dvr import RHF1D
import pyqed

def direct_ci(ci0=None):
    # CI without constructing the full Hamiltonian

    # if max_memory < civec_size*6*8e-6:
    #     logging.warn('Not enough memory for FCI solver. '
    #              'The minimal requirement is %.0f MB', civec_size*60e-6)
    pass

class CASCI(pyqed.qchem.mcscf.casci.CASCI):
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
            self.nelecas = mf.mol.nelec
            self.ncore = 0
        else:
            if nelecas != mf.mol.nelec:
                raise ValueError('All electrons have to be active.')
            self.nelecas = nelecas
            self.ncore = mf.nelec//2 - nelecas//2 # core orbs


        if self.nelecas <= 2:
            print('Electrons < 2. Use CIS or CISD instead.')


        self.mf = mf
        self.mol = mf.mol
        self.chemical_potential = mu

        self.mo_coeff = mf.mo_coeff

        #####
        self.binary = None
        self.SC1 = None
        self.SC2 = None
        self.e_tot = None

    def get_SO_matrix(self, SF=False, H1=None, H2=None):
        """
        Given a PySCF rhf/uhf object get Spin-Orbit one-electron and two-electron H Matrices

        SF: bool
            spin-flip
        """
        # from pyscf import ao2mo

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

    def natural_orbitals(self, dm=None, nco=None):
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
        from pyqed.qchem.mcscf import spin_square

        mf = self.mf
        ncas = self.ncas
        ncore = self.ncore

        mo_occ = mf.mo_occ[:, :ncas]/2


        # mo_occ = self.mf.mo_occ[self.ncore: self.ncore+ncas]/2


        mf.mo_coeff = mf.mo_coeff[:, :ncas]

        Binary = get_fci_combos(mo_occ)

        # print('Binary shape', Binary.shape)

        self.binary = Binary

        # build the 1e and 2e Hamiltonian in MOs

        H1, H2 = self.get_SO_matrix(mf)

        # build the CI Hamiltonain
        SC1, SC2 = SlaterCondon(Binary)
        H_CI = CI_H(Binary, H1, H2, SC1, SC2)

        self.SC1 = SC1
        self.SC2 = SC2


        # E, X = np.linalg.eigh(H_CI)
        E, X = eigsh(H_CI, k=nstates, which='SA')

        e_nuc = self.mol.energy_nuc()

        self.e_tot = E + e_nuc
        self.ci = [X[:, n] for n in range(nstates)]

        for i in range(nstates):
            ss = spin_square(*self.make_rdm12(i))
            print("CASCI Root {}  E = {:.10f}  S^2 = {:.6f}".format(i, self.e_tot[i], ss))

        return self



    def make_rdm1(self, state_id, with_core=False, with_vir=False, representation='mo'):
        """
        spin-traced 1e reduced density matrix
        .. math::

            \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>


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
        if with_core and with_vir:

            D = np.zeros((nmo, nmo), dtype=float)
            if ncore > 0: D[:ncore, :ncore] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            return D
        else:
            return make_rdm1(ci, self.binary, self.SC1)

    def fix_spin(self,s=0):
        pass



# def nonorthogonal_transition_density_matrix(cibra, ciket, h1e=None, h2e=None):

def overlap(cibra, ciket, h1e=None, h2e=None, return_tdm1=False):
    '''s
    Overlap between two nonorthogonal determinant wavefunctions with
    a common set of orthonormal AOs.

    For non-orthogonal AOs, the code can be easily adapted.

    .. math::

        \gamma^{II'}_{qp} = \langle \Phi_I | p^\dag q | \Phi_{I'}\rangle

        \Gamma_{pqrs} = \langle \Phi_I | p^\dag r^\dag s q | \Phi_{I'}\rangle

        C_I^* Tr{O \gamma^{II'} } C_{I'}

    where Phi denotes Slater determinant and p,q,r,s refers to orthonormal AOs.

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
        MCSCF objects.
    binary1 : TYPE
        DESCRIPTION.
    ciket : TYPE
        DESCRIPTION.
    h1e: ndarray
        single-electron operators to be contracted with the 1e NO-TDM
    binary2 : TYPE
        DESCRIPTION.
    s : TYPE
        AO overlap.



    Args:
        s : 2D array
            The overlap matrix of non-orthogonal one-particle basis

    Returns
    -------
    None.

    Refs:
        Ulsuno, Comp Phys Comm 2013


    '''
    # nstates = len(cibra) + 1

    # overlap matrix between MOs at different geometries
    # if s is None:

    #     from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

    #     s = overlap_integral_asymmetric(cibra.mol._bas, ciket.mol._bas)
    #     s = reduce(np.dot, (cibra.mf.mo_coeff.T, s, ciket.mf.mo_coeff))

    try:
        assert(isinstance(cibra.mf, RHF1D))
    except:
        raise Warning('use RHF.')

    C1 = cibra.mf.mo_coeff
    C2 = ciket.mf.mo_coeff

    s = dag(C1) @ C2 # MO overlap
    sinv = np.linalg.inv(s)

    nao = C1.shape[0] # number of AOs

    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]

    S = np.zeros((nsd_bra, nsd_ket)) # overlap between determinants
    S1 = np.zeros((nsd_bra, nsd_ket))

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore

    # scc = s[:ncore_bra, :ncore_ket]
    # sca = s[:ncore_bra, ncore_ket:]
    # sac = s[ncore_bra:, :ncore_ket]
    # saa = s[ncore_bra:, ncore_ket:]

    # scc_det = np.linalg.det(scc)
    # scc_inv = np.linalg.inv(scc)

    # for I in range(nsd_bra):
    #     occidx1_a  = [i for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
    #     occidx1_b  = [i for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

    #     for J in range(nsd_ket):
    #         occidx2_a  =  [i for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
    #         occidx2_b  =  [i for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.
            # saa_occ_a = saa[np.ix_(occidx1_a, occidx2_a)]
            # sca_occ_a = sca[:, occidx2_a]
            # sac_occ_a = sac[occidx1_a, :]

            # saa_occ_b = saa[np.ix_(occidx1_b, occidx2_b)]
            # sca_occ_b = sca[:, occidx2_b]
            # sac_occ_b = sac[occidx1_b, :]

            # S[I, J] = scc_det**2 * np.linalg.det(saa_occ_a - sac_occ_a @ scc_inv @ sca_occ_a)*\
            #     np.linalg.det(saa_occ_b - sac_occ_b @ scc_inv @ sca_occ_b)



    core_bra = list(range(cibra.ncore))
    core_ket = list(range(ciket.ncore))

    for I in range(nsd_bra):

        occidx1_a  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
        occidx1_b  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

        for J in range(nsd_ket):
            occidx2_a  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
            occidx2_b  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.

            S[I, J] = np.linalg.det(s[np.ix_(occidx1_a, occidx2_a)]) * \
                      np.linalg.det(s[np.ix_(occidx1_b, occidx2_b)])

            D = (C1[:, occidx1_a] @ sinv @ dag(C2[:, occidx2_a]) + \
                C1[:, occidx1_b] @ sinv @ dag(C2[:, occidx2_b])) * S[I,J]

            S1[I, J] = np.trace(h1e @ D)


    overlap = contract('IB, IJ, JA', cibra.ci.conj(), S, ciket.ci)
    tdm1 = contract('IB, IJ, JA', cibra.ci.conj(), S1, ciket.ci)

    return overlap, tdm1



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