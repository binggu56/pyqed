#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:19:05 2024

CISD

@author: Bing Gu (gubing@westlake.edu.cn)

"""
import numpy as np

from itertools import combinations
from pyscf import ao2mo, gto, scf
from scipy.sparse.linalg import eigsh
from functools import reduce
from opt_einsum import contract

from pyscf.ao2mo.outcore import general_iofree as ao2mofn

from pyqed.qchem.hf.rhf import RHF
from pyqed import dag


def givenΛgetB(ΛA, ΛB, N_mo):
    "Given Λ(i occupied orbitals for each determinant) get B (binary rep.)"
    Binary = np.zeros((ΛA.shape[0], 2, N_mo), dtype=np.int8)
    for I in range(len(Binary)):
        Binary[I, 0, ΛA[I,:]] = 1
        Binary[I, 1, ΛB[I,:]] = 1

    return Binary

def SpinOuterProduct(A, B, stack=False):
    ΛA = np.einsum("Ii, J -> IJi", A, np.ones(B.shape[0], dtype=np.int8)).reshape(
    (A.shape[0]*B.shape[0], A.shape[1]) )
    ΛB = np.einsum("Ii, J -> JIi", B, np.ones(A.shape[0], dtype=np.int8)).reshape(
    (A.shape[0]*B.shape[0], B.shape[1]) )
    if stack:
        return np.array([ΛA,ΛB])
    else:
        return ΛA, ΛB


def get_fci_combos(mf):
    # print(mf.mo_occ.shape)
    O_sp = np.asarray(mf.mo_occ, dtype=np.int8)

    # number of electrons for each spin
    N_s = np.einsum("sp -> s", O_sp)

    N = O_sp.shape[1]

    Λ_α = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[0] ) ) )
    Λ_β = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[1] ) ) )
    ΛA, ΛB = SpinOuterProduct(Λ_α, Λ_β)
    Binary = givenΛgetB(ΛA, ΛB, N)
    return Binary


def determinantsign(Binary):

    sign = np.cumsum( Binary, axis=2)
    for I in range(len(Binary)):
        iia = np.where( Binary[I,0] == 1)[0]
        iib = np.where( Binary[I,1] == 1)[0]
        sign[I, 0, iia] = np.arange(0, len(iia), 1)
        sign[I, 1, iib] = np.arange(0, len(iib), 1)
    return ( (-1.)**(sign) ).astype(np.int8)

def get_excitation_op(i, j, binary, sign, spin=0):


    Difference = binary[i,spin] - binary[j, spin]


    a_t = (Difference + 0.5).astype(np.int8)
    a = -1*(Difference - 0.5).astype(np.int8)

    # print('a', a.shape)
    if len(a) > 0:
        if np.sum(a[0]) > 1: ### this is a double excitation
            å_t = 1*a_t ## make copy
            å_t[ np.arange(len(å_t)),(å_t!=0).argmax(axis=1) ] = 0 ## zero first 1
            a_t = np.abs(å_t - a_t) ## absolute difference from orginal
            a_t = np.asarray([sign[j, spin]*å_t,sign[j, spin]*a_t]) ## stack
            å = 1*a ## make copy
            å[ np.arange(len(å)),(å!=0).argmax(axis=1) ] = 0 ## zero first 1
            a = np.abs(å - a) ## absolute difference from orginal
            a = np.asarray([sign[i, spin]*å,sign[i, spin]*a]) ## stack

    # print(a.shape, a_t.shape)

        # return a_t, a

    return sign[j, spin]*a_t, sign[i, spin]*a


def SlaterCondon(Binary):
    sign = determinantsign(Binary)
    SpinDifference = np.sum( np.abs(Binary[:, None, :, :] - Binary[None, :, :, :]), axis=3)//2

    ## indices for 1-difference
    I_A, J_A = np.where( np.all(SpinDifference==np.array([1,0], dtype=np.int8), axis=2) )
    I_B, J_B = np.where( np.all(SpinDifference==np.array([0,1], dtype=np.int8), axis=2) )

    ## indices for 2-differences
    I_AA, J_AA = np.where( np.all(SpinDifference==np.array([2,0], dtype=np.int8), axis=2) )
    I_BB, J_BB = np.where( np.all(SpinDifference==np.array([0,2], dtype=np.int8), axis=2) )
    I_AB, J_AB = np.where( np.all(SpinDifference==np.array([1,1], dtype=np.int8), axis=2) )


    ### get excitation operators

    a_t , a = get_excitation_op(I_A , J_A , Binary, sign, spin=0)
    b_t , b = get_excitation_op(I_B , J_B , Binary, sign, spin=1)


    ca = ((Binary[I_A,0,:] + Binary[J_A,0,:])/2).astype(np.int8)
    cb = ((Binary[I_B,1,:] + Binary[J_B,1,:])/2).astype(np.int8)

    # if len(I_AA) >0:
    aa_t, aa = get_excitation_op(I_AA, J_AA, Binary, sign, spin=0)

    # if len(I_BB) > 0:
    bb_t, bb = get_excitation_op(I_BB, J_BB, Binary, sign, spin=1)

    ab_t, ab = get_excitation_op(I_AB, J_AB, Binary, sign, spin=0)
    ba_t, ba = get_excitation_op(I_AB, J_AB, Binary, sign, spin=1)

    SC1 = [I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb]
    SC2 = [I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba]

    return SC1, SC2

# def get_SO_matrix(mf, SF=False, H1=None, H2=None):
#     """
#     Given a PySCF uhf object get Spin-Orbit Matrices

#     Parameters
#     ==========

#     mf: uhf_pyscf object

#     SF: bool
#         spin-flip
#     """
#     if isinstance(mf, (scf.rhf.RHF, RHF)):
#         # molecular orbitals
#         Ca, Cb = [mf.mo_coeff, ] * 2
#     else:
#         Ca, Cb = mf.mo_coeff



#     # S = (uhf_pyscf.mol).intor("int1e_ovlp")
#     # eig, v = np.linalg.eigh(S)
#     # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

#     # H1e in AO
#     H = mf.get_hcore()
#     n = Ca.shape[1]

#     # print('eri', uhf_pyscf._eri.shape, Ca.shape)

#     eri_aa = (ao2mo.general( mf._eri , (Ca, Ca, Ca, Ca),
#                             compact=False)).reshape((n,n,n,n), order="C")
#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = (ao2mo.general(mf._eri , (Cb, Cb, Cb, Cb), \
#                             compact=False)).reshape((n,n,n,n), order="C")
#     eri_bb -= eri_bb.swapaxes(1,3)

#     eri_ab = (ao2mo.general(mf._eri , (Ca, Ca, Cb, Cb), \
#                             compact=False)).reshape((n,n,n,n), order="C")

#     # eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

#     eri_ba = (ao2mo.general(mf._eri , (Cb, Cb, Ca, Ca), \
#                             compact=False)).reshape((n,n,n,n), order="C")

#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

#     H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca.conj(), Ca),\
#                      np.einsum("AB, Ap, Bq -> pq", H, Cb.conj(), Cb)])

#     if SF:
#         eri_abab = (ao2mo.general( mf._eri , (Ca, Cb, Ca, Cb),
#         compact=False)).reshape((n,n,n,n), order="C")
#         eri_abba = (ao2mo.general( mf._eri , (Ca, Cb, Cb, Ca),
#         compact=False)).reshape((n,n,n,n), order="C")
#         eri_baab = (ao2mo.general( mf._eri , (Cb, Ca, Ca, Cb),
#         compact=False)).reshape((n,n,n,n), order="C")
#         eri_baba = (ao2mo.general( mf._eri , (Cb, Ca, Cb, Ca),
#         compact=False)).reshape((n,n,n,n), order="C")
#         H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
#         return H1, H2, H2_SF
#     else:
#         return H1, H2

def get_SO_matrix(mf, SF=False, H1=None, H2=None):
    """
    Given a PySCF rhf/uhf object get Spin-Orbit one-electron and two-electron H Matrices

    SF: bool
        spin-flip
    """
    # from pyscf import ao2mo


    # molecular orbitals
    # if isinstance(mf, RHF):

    Ca, Cb = [mf.mo_coeff, ] * 2

    eri = mf.mol.eri


    # elif isinstance(mf, scf.uhf.UHF):
    #     Ca, Cb = mf.mo_coeff

    #     eri = mf.mol.intor('int2e', aosym='s8')


    # else:
    #     raise TypeError('mf type not supported.', type(mf))



    # S = (uhf_pyscf.mol).intor("int1e_ovlp")
    # eig, v = np.linalg.eigh(S)
    # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

    # H1e in AO
    H = mf.get_hcore()
    # H = dag(Ca) @ H @ Ca

    n = nmo = Ca.shape[1] # n



    # eri = mf.get_eri()

    # eri_aa = eri  # (pq||rs) 1^*12^*2
    # eri_aa -= eri_aa.swapaxes(1,3) # change to physicts notation 1* 2* 2 1



    # # eri_ab = contract('ip, iq, ij, jr, js->pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
    # # eri_ba = contract('ip, iq, ij, jr, js->pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)
    # eri_ab = eri.copy()
    # eri_ba = eri.copy()

    # if isinstance(mf, scf.uhf.UHF):

    # eri = ao2mo.get_ao_eri(mf.mol, compact=False).reshape(nmo, nmo, nmo, nmo)
    # print(eri.shape)

    eri_aa = (ao2mo.general( eri , (Ca, Ca, Ca, Ca),
                            compact=False)).reshape((n,n,n,n), order="C")
    eri_aa -= eri_aa.swapaxes(1,3)


    eri_bb = (ao2mo.general( eri , (Cb, Cb, Cb, Cb),
    compact=False)).reshape((n,n,n,n), order="C")
    eri_bb -= eri_bb.swapaxes(1,3)

    # eri_bb = eri_aa.copy()


    eri_ab = (ao2mo.general( eri , (Ca, Ca, Cb, Cb),
    compact=False)).reshape((n,n,n,n), order="C")
    #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

    eri_ba = (ao2mo.general( eri , (Cb, Cb, Ca, Ca),
    compact=False)).reshape((n,n,n,n), order="C")

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

def CI_H(Binary, H1, H2, SC1, SC2):
    """
    Explicitly construct the CI Hamiltonian Matrix
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

    # sum of MO energies
    H_CI = np.einsum("Spp, ISp -> I", H1, Binary, optimize=True)

    # ERI
    H_CI += np.einsum("STppqq, ISp, ITq -> I", H2, Binary, Binary, optimize=True)/2
    H_CI = np.diag(H_CI)

    ## Rule 1
    H_CI[I_A , J_A ] -= np.einsum("pq, Kp, Kq -> K", H1[0], a_t, a, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,0], a_t, a, ca, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,1], a_t, a, Binary[I_A,1],
    optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", H1[1], b_t, b, optimize=True)
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

    return H_CI


class CI:
    def __init__(self, mf, frozen=None, max_cycle=50):

#        assert(isinstance(mf, (scf.rhf.RHF, RHF)))

        self.mf = mf
        self.mol = mf.mol
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff

        # self.nstates = nstates

        self.nao = mf.mol.nao
        self.nmo = self.nao

        self.nocc = mf.mol.nelec//2

        self.max_cycle = max_cycle

        self.nso = self.nmo * 2


        # self.mo_energy = np.zeros(self.nso)
        # self.mo_energy[0::2] = self.mo_energy[1::2] = mf.mo_energy


        self.binary = None
        self.H = None



class CISD(CI):
    """
    restricted CISD
    """
    def run(self):
        pass

    def vec_to_amplitudes(self, civec, copy=True):

        nmo = self.nmo
        nocc = self.nocc

        nvir = nmo - nocc
        c0 = civec[0]
        cp = lambda x: (x.copy() if copy else x)
        c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
        c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir))

        return c0, c1, c2


class UCISD(CI):
    """
    As all determinants have :math:`S_z = 0`, the degeneracy may arise
    between the singlet and one of the triplets.

    Refs:
        C.D.  Sherrill, H.F. Schaefer III, Advances in Quantum Chemistry, Volume 34 , 1999, Pages 143-269
    """

    def buildH(self):
        '''
        Return diagonal of CISD hamiltonian in Slater determinant basis.

        Note that a constant has been substracted of all elements.
        The first element is the HF energy (minus the
        constant),
        the next elements are the diagonal elements with singly
        excited determinants (<D_i^a|H|D_i^a> within the constant),
        then
        doubly excited determinants (<D_ij^ab|H|D_ij^ab> within the
        constant).

        Args:
            myci : CISD (inheriting) object
            eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
                Contains the various (pq|rs) integrals needed.

        Returns:
            numpy array (size: (1, 1 + #single excitations from HF det
                                   + #double excitations from HF det))
                Diagonal elements of hamiltonian matrix within a constant,

        '''

        # restricted HF
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc

        # # get ERI, spin alpha and beta are in alternating order [1a, 1b, 2a, 2b, ...]
        # b = np.zeros((nso//2, nso))
        # b[:,0::2] = b[:,1::2] = mf.mo_coeff

        # # h_{pq}
        # v_mf = mf.get_veff() - mf.get_j() # J - K
        # self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
        # self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0

        # # electron repulsion integral
        # eri = ao2mofn(mf.mol, (b,b,b,b),
        #               compact=False).reshape(nso,nso,nso,nso)

        # eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0
        # # Integrals are in "chemist's notation"
        # # eri[i,j,k,l] = (ij|kl) = \int i(1) j(1) 1/r12 k(r2) l(r2)
        # print("Imag part of ERIs =", np.linalg.norm(eri.imag))
        # self.eri = eri.real


        # assert(nvir > 0)


        # number of Slater determinants (without spin symmetry)
        nsd = 1 + 2*nocc * nvir + nocc*(nocc-1)*nvir*(nvir-1)//2 + nocc**2*nvir**2
        HCI = np.zeros((nsd, nsd))


        # nsd = 1 + 4*nocc * nvir + nocc*(2*nocc-1)*nvir*(2*nvir-1)

        print('number of determinants', nsd)

        # # Given Λ(i occupied orbitals for each determinant) get B (binary rep.)"
        Binary = np.zeros((nsd, 2, nmo), dtype=np.int8)
        # # for I in range(len(Binary)):

        ### Group spin-orbitals together also includes spin-flip transitions.

        # if isinstance(self.mf, RHF):
        #     for I in range(nsd):
        #         Binary[I, :2*nocc] = 1

        # I = 1
        # for i in range(2*nocc):
        #     for a in range(2*nocc, nso):
        #         Binary[I, i] -= 1
        #         Binary[I, a] += 1
        #         I += 1

        # for i in range(2*nocc):
        #     for j in range(i):
        #         for a in range(2*nocc, nso):
        #             for b in range(a+1, nso):
        #                 Binary[I, i] -= 1
        #                 Binary[I, j] -= 1
        #                 Binary[I, a] += 1
        #                 Binary[I, b] += 1

        #                 I += 1
        # Binary = Binary.reshape(nsd, 2, nmo)


        if isinstance(self.mf, (RHF,scf.rhf.RHF)):
            # for I in range(nsd):
            Binary[:] = [mf.mo_occ//2, ] * 2
        else:
            Binary[:] = mf.mo_occ


        # singles
        I = 1
        for i in range(nocc):
            for a in range(nocc, nmo):
                Binary[I, 0, i] -= 1
                Binary[I, 0, a] += 1

                Binary[I+1, 1, i] -= 1
                Binary[I+1, 1, a] += 1

                # HCI[I, I] = mo_energy[a] - mo_energy[i] # +

                # This equality is because the input is restricted.
                # HCI[I+1, I+1] = HCI[I, I]


                I += 2

        # doubles aa, bb excitation     a^b^ji
        for i in range(nocc):
            for j in range(i): # i > j

                for a in range(nocc, nmo):
                    for b in range(nocc, a): # a > b

                        Binary[I, 0, i] -= 1
                        Binary[I, 0, j] -= 1

                        Binary[I+1, 1, i] -= 1
                        Binary[I+1, 1, j] -= 1

                        Binary[I, 0, b] += 1
                        Binary[I, 0, a] += 1

                        Binary[I+1, 1, a] += 1
                        Binary[I+1, 1, b] += 1

                        I += 2

        # doubles with ab excitation
        for i in range(nocc):
            for a in range(nocc, nmo):
                for j in range(nocc):
                    for b in range(nocc, nmo):

                        Binary[I, 0, i] -= 1
                        Binary[I, 0, a] += 1

                        Binary[I, 1, j] -= 1
                        Binary[I, 1, b] += 1

                        I += 1

        self.binary = Binary

        assert(I == nsd)

        H1, H2 = get_SO_matrix(self.mf)


        SC1, SC2 = SlaterCondon(Binary)
        H_CI = CI_H(Binary, H1, H2, SC1, SC2)


        # self.mf.energy_elec()

        # e_hf = self.mf.e_tot - self.mf.e_nuc


        # print(e_hf)

        # E, X = np.linalg.eigh(H_CI)
        # E, X = eigsh(H_CI, k=3, which='SA')


        return H_CI

    def run(self, ci0=None, nstates=1, tol=1e-6):

        H_CI = self.buildH()

        E, X = eigsh(H_CI, k=nstates, maxiter=self.max_cycle, \
                      which='SA', tol=tol, v0=ci0)

        self.e_tot = E + self.mf.energy_nuc()

        self.ci = X

        for n in range(nstates):
            print('UCISD root {} E = {} '.format(n, self.e_tot[n]))

        # TODO: total spin

        return self

    def spin(self):
        pass


    def vec_to_amplitudes(self, civec, copy=True):

        nmo = self.nmo
        nocc = self.nocc

        nvir = nmo - nocc
        c0 = civec[0]
        cp = lambda x: (x.copy() if copy else x)
        c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
        c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir))

        return c0, c1, c2

    def make_rdm1(self):
        pass

    def make_rdm2(self):
        pass

    def make_natural_orbitals(self):
        pass


def overlap(cibra, ciket, s=None):
    """
    CI electronic overlap matrix (CISD, CASCI)

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

    nsd = cibra.binary.shape[0]
    S = np.zeros((nsd, nsd))


    for I in range(nsd):
        occidx1_a  = [i for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
        occidx1_b  = [i for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

        # print('a', occidx1_a, occidx1_b)

        for J in range(nsd):
            occidx2_a  = [i for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
            occidx2_b  = [i for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])


            S[I, J] = np.linalg.det(s[np.ix_(occidx1_a, occidx2_a)]) * \
                      np.linalg.det(s[np.ix_(occidx1_b, occidx2_b)])


    return contract('IB, IJ, JA', cibra.ci.conj(), S, ciket.ci)





if __name__=='__main__':
    from pyscf import gto, scf, dft, tddft, ao2mo, fci, ci

    from pyqed.qchem.mol import get_hcore_mo, get_eri_mo, Molecule
    from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionChain
    from pyqed.qchem.hf.rhf import RHF
    from pyqed.qchem import FCI

    # mol = gto.Mole()
    mol = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1)], ])
    mol.basis = 'sto3g'
    mol.charge = 0
    # mol.unit = 'b'
    mol.build()

    ### pyscf reference
    # mf = scf.rhf.RHF(mol).run()
    # myfci = fci.FCI(mf).run(nroots=4)
    # print(myfci.e_tot)
    # mf = scf.uhf.UHF(mol).run()
    # myci = ci.ucisd.UCISD(mf).run(nroots=5)



    # cisd = ci.cisd.CISD(mf).run()
    mf = RHF(mol).run()
    myci = UCISD(mf).run(nstates=4)
    # myfci = FCI(mf).run(4)



    mol2 = Molecule(atom=[
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1.1)]])
    mol2.basis = 'sto3g'
    mol2.charge = 0
    # mol.unit = 'b'
    mol2.build()

    mf2 = RHF(mol2).run()
    ci2 = UCISD(mf2).run(nstates=4)
    # for I in range(93):
    #     print(myci.binary[I])



    A = overlap(myci, ci2)
    print(A)




    # myci = fci.FCI(mf).run(nroots=5)

    # print(myci.e_tot)



    # print(mf.e_tot - mf.energy_nuc())

    # h1e = get_hcore_mo(mf)
    # h2e = get_eri_mo(mf)

    # print(fci.direct_spin0.pspace(h1e, h2e, norb=6, nelec=4))

    # print(myfci.)
    # Ca = mf.mo_coeff[0ArithmeticError
    # n = Ca.shape[-1]

    # mo_coeff = mf.mo_coeff
    # get the two-electron integrals as a numpy array
    # eri = get_eri_mo(mol, mo_coeff)

    # n = mol.nao
    # Ca = mo_coeff

    # h1e = get_hcore_mo(mf)
    # eri = get_eri_mo(mf)

    # E, X = SpinHalfFermionChain(h1e, eri, nelec=mol.nelectron).run(nstates=2)
    # print(E + mol.energy_nuc())

    # eri_aa = (ao2mo.general( mf._eri , (Ca, Ca, Ca, Ca),
    #                         compact=False)).reshape((n,n,n,n), order="C")
    # print(eri_aa.shape)

    # E, X = FCI(mf).run()
    # print(E + mol.energy_nuc())