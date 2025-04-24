#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:29:59 2024

@author: Bing Gu (gubing@westlake.edu.cn)

From
https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/651592aca69febde9ed8d6a6/original/notes-on-generalized-configuration-interaction-in-python.pdf

"""
import numpy as np

from itertools import combinations
from pyscf import ao2mo, scf
from scipy.sparse.linalg import eigsh
from functools import reduce

from pyscf.ao2mo.outcore import general_iofree as ao2mofn
from opt_einsum import contract

from pyqed.qchem.hf.rhf import RHF
from pyqed.qchem.cisd import get_SO_matrix, get_excitation_op,\
    SlaterCondon, CI_H


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
    # if isinstance(mf, (scf.rhf.RHF, RHF)):

    mo_occ = [mf.mo_occ//2, ] * 2
    # else:
    #     mo_occ = mf.mo_occ

    O_sp = np.asarray(mo_occ, dtype=np.int8)


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

# def get_excitation_op(i, j, binary, sign, spin=0):


#     Difference = binary[i,spin] - binary[j, spin]


#     a_t = (Difference + 0.5).astype(np.int8)
#     a = -1*(Difference - 0.5).astype(np.int8)

#     # print('a', a.shape)
#     if a is not None:
#         if np.sum(a[0]) > 1: ### this is a double excitation
#             å_t = 1*a_t ## make copy
#             å_t[ np.arange(len(å_t)),(å_t!=0).argmax(axis=1) ] = 0 ## zero first 1
#             a_t = np.abs(å_t - a_t) ## absolute difference from orginal
#             a_t = np.asarray([sign[j, spin]*å_t,sign[j, spin]*a_t]) ## stack
#             å = 1*a ## make copy
#             å[ np.arange(len(å)),(å!=0).argmax(axis=1) ] = 0 ## zero first 1
#             a = np.abs(å - a) ## absolute difference from orginal
#             a = np.asarray([sign[i, spin]*å,sign[i, spin]*a]) ## stack

#         # print(a.shape, a_t.shape)

#         # return a_t, a

#     return sign[j, spin]*a_t, sign[i, spin]*a


# def SlaterCondon(Binary):
#     sign = determinantsign(Binary)
#     SpinDifference = np.sum( np.abs(Binary[:, None, :, :] - Binary[None, :, :, :]), axis=3)//2

#     ## indices for 1-difference
#     I_A, J_A = np.where( np.all(SpinDifference==np.array([1,0], dtype=np.int8), axis=2) )
#     I_B, J_B = np.where( np.all(SpinDifference==np.array([0,1], dtype=np.int8), axis=2) )

#     ## indices for 2-differences
#     I_AA, J_AA = np.where( np.all(SpinDifference==np.array([2,0], dtype=np.int8), axis=2) )
#     I_BB, J_BB = np.where( np.all(SpinDifference==np.array([0,2], dtype=np.int8), axis=2) )
#     I_AB, J_AB = np.where( np.all(SpinDifference==np.array([1,1], dtype=np.int8), axis=2) )


#     ### get excitation operators

#     a_t , a = get_excitation_op(I_A , J_A , Binary, sign, spin=0)
#     b_t , b = get_excitation_op(I_B , J_B , Binary, sign, spin=1)


#     ca = ((Binary[I_A,0,:] + Binary[J_A,0,:])/2).astype(np.int8)
#     cb = ((Binary[I_B,1,:] + Binary[J_B,1,:])/2).astype(np.int8)


#     aa_t, aa = get_excitation_op(I_AA, J_AA, Binary, sign, spin=0)

#     bb_t, bb = get_excitation_op(I_BB, J_BB, Binary, sign, spin=1)


#     ab_t, ab = get_excitation_op(I_AB, J_AB, Binary, sign, spin=0)
#     ba_t, ba = get_excitation_op(I_AB, J_AB, Binary, sign, spin=1)

#     SC1 = [I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb]
#     SC2 = [I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba]

#     return SC1, SC2

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

# def get_SO_matrix(mf, SF=False, H1=None, H2=None):
#     """
#     Given a PySCF rhf/uhf object get Spin-Orbit one-electron and two-electron H Matrices

#     SF: bool
#         spin-flip
#     """
#     from pyscf import ao2mo

#     # molecular orbitals
#     Ca, Cb = [mf.mo_coeff, ] * 2

#     # S = (uhf_pyscf.mol).intor("int1e_ovlp")
#     # eig, v = np.linalg.eigh(S)
#     # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

#     # H1e in AO
#     H = mf.get_hcore()
#     # H = dag(Ca) @ H @ Ca

#     nmo = Ca.shape[1] # n

#     eri = mf.eri  # (pq||rs) 1^*12^*2
#     eri_aa = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#     # physicts notation <pq|rs>
#     # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = eri_aa.copy()

#     eri_ab = contract('ip, iq, ij, jr, js->pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
#     eri_ba = contract('ip, iq, ij, jr, js->pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




#     # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
#     #                         compact=False)).reshape((n,n,n,n), order="C")
#     # eri_aa -= eri_aa.swapaxes(1,3)

#     # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
#     # compact=False)).reshape((n,n,n,n), order="C")
#     # eri_bb -= eri_bb.swapaxes(1,3)

#     # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
#     # compact=False)).reshape((n,n,n,n), order="C")
#     # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

#     # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
#     # compact=False)).reshape((n,n,n,n), order="C")

#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))
#     H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca), np.einsum("AB, Ap, Bq -> pq",
#     H, Cb, Cb)])

#     # if SF:
#     #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
#     #     return H1, H2, H2_SF
#     # else:
#     #     return H1, H2
#     return H1, H2

# def CI_H(Binary, H1, H2, SC1, SC2):
#     """
#     Explicitly construct the CI Hamiltonian Matrix
#     GIVEN: H1 (1-body Hamtilonian)
#     H2 (2-body Hamtilonian)
#     SC1 (1-body Slater-Condon Rules)
#     SC2 (2-body Slater-Condon Rules)
#     GET: CI Hamiltonian
#     """
#     I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
#     I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

#     # sum of MO energies
#     H_CI = np.einsum("Spp, ISp -> I", H1, Binary, optimize=True)

#     # ERI
#     H_CI += np.einsum("STppqq, ISp, ITq -> I", H2, Binary, Binary, optimize=True)/2
#     H_CI = np.diag(H_CI)

#     ## Rule 1
#     H_CI[I_A , J_A ] -= np.einsum("pq, Kp, Kq -> K", H1[0], a_t, a, optimize=True)
#     H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,0], a_t, a, ca, optimize=True)
#     H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,1], a_t, a, Binary[I_A,1],
#     optimize=True)
#     H_CI[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", H1[1], b_t, b, optimize=True)
#     H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,1], b_t, b, cb, optimize=True)
#     H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,0], b_t, b, Binary[I_B,0],
#     optimize=True)

#     ## Rule 2
#     H_CI[I_AA, J_AA] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,0], aa_t[0], aa[0],
#     aa_t[1], aa[1], optimize=True)
#     H_CI[I_BB, J_BB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[1,1], bb_t[0], bb[0],
#     bb_t[1], bb[1], optimize=True)
#     H_CI[I_AB, J_AB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,1], ab_t, ab, ba_t, ba,
#     optimize=True)

#     return H_CI

def fcisolver(mf, nstates=1):
    """
    Calculate the FCI of a PySCF Mean Field Object

    Parameters
    ==========

    mf: (PySCF Mean Field Object)

    nstates: int
        number of desired states. Default 1.

    Returns
    =======
    E (Eigenvlues)
    X (Eigenstates)
    """

    # create all determinants labeled by a 3D array (I, \sigma, p)
    Binary = get_fci_combos(mf)

    print('Number of determinants', Binary.shape[0])

    H1, H2 = get_SO_matrix(mf)



    SC1, SC2 = SlaterCondon(Binary)
    H_CI = CI_H(Binary, H1, H2, SC1, SC2)




    # E, X = np.linalg.eigh(H_CI)
    E, X = eigsh(H_CI, k=nstates, which='SA')

    return E, X






class FCI:
    def __init__(self, mf):
        self.mf = mf

    def run(self, nstates=1):

        E, X = fcisolver(self.mf, nstates)


        self.ci = X
        self.e_tot = E + self.mf.energy_nuc()

        for n in range(nstates):
            print('FCI root {} E = {} '.format(n, self.e_tot[n]))

        return self



if __name__=='__main__':
    from pyscf import gto, scf, dft, tddft, ao2mo, ci
    # from pyscf.fci import FCI

    from pyqed.qchem.mol import get_hcore_mo, get_eri_mo, Molecule
    from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionChain
    # from pyqed.qchem.hf import RHF

    # mol = gto.Mole()
    mol = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1)]])
    
    mol.basis = 'sto3g'
    mol.charge = 0
    # mol.unit = 'b'
    mol.build()
    



    # # cisd = ci.cisd.CISD(mf).run()
    mf = RHF(mol)
    mf.run()
    
    print(mol.eri.shape)


    fci = FCI(mf).run()
    print(fci.e_tot)


    mol = gto.Mole(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1)]])
    
    mol.basis = 'sto3g'
    mol.charge = 0
    mol.unit = 'b'
    
    mf = scf.RHF(mol).run()
    
    # from pyscf.fci import FCI
    # FCI(mf).run()
    import pyscf 
    cisolver = pyscf.fci.FCI(mf).run()
    print(cisolver.e_tot)
    
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