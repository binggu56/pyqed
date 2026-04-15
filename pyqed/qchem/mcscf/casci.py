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

import sys
from opt_einsum import contract

from pyqed import tensor
from itertools import combinations
import itertools
import warnings

from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct, get_fci_combos, SlaterCondon, CI_H
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo
from pyqed.qchem.soc import get_soc_1e_spin_orbital, reorder_spin_orbital_matrix

def cistring(norb, nelec, sz=0):
    """
    Return all the possible :math:`n`-length binary
    where :math:`k` of :math:`n` digitals are set to 1.

    Parameters
    ----------
    norb: int
        number of orbs. Binary length :math:`n`.
    nelec: int or list of length 2
        number of electrons

    Returns
    -------
    res: list of int-lists
        A list of list containing the binary digitals.

    """
    n = norb
    if isinstance(nelec, (list, tuple)):
        na, nb = nelec
    else:
        if np.mod(nelec, 2):
            raise ValueError('odd number of electrons results in ambiguity. \
                             Please set alpha and beta electrons seperately.')
        na = nb = nelec//2

    if n == 0:
        return [[0]]

    res = []
    for bits in itertools.combinations(list(range(n)), na):
        sa = [0] * n
        for bit in bits:
            sa[bit] = 1
        res.append(sa)

    if na != nb:

        res_b = []
        for bits in itertools.combinations(list(range(n)), nb):

            sb = [0] * n
            for bit in bits:
                sb[bit] = 1

            res_b.append(sb)


    return list(itertools.product(res,res_b))


def get_combos(mo_occ, space='fci', ncore=None, ncas=None, nvir=None):

    space = space.replace(" ", "").lower()
    nmo = len(mo_occ)

    if space == 'fci': # FCI, CAS

        O_sp = np.asarray(mo_occ, dtype=np.int8)
        # number of electrons for each spin
        N_s = np.einsum("sp -> s", O_sp)



        N = O_sp.shape[1]

        Λ_α = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[0] ) ) )
        Λ_β = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[1] ) ) )
        ΛA, ΛB = SpinOuterProduct(Λ_α, Λ_β)
        Binary = givenΛgetB(ΛA, ΛB, N)

        return Binary

    elif space == 'cisd':
        pass
    elif space == 'cas+s': # MR-CIS

        if nvir is None:
            nvir = nmo - ncore - ncas


        # if isinstance(mf, (scf.rhf.RHF, RHF)):
        # if mo_occ is None:
        #     mo_occ = [mf.mo_occ//2, ] * 2
        # else:
        #     mo_occ = mf.mo_occ


def _is_uhf_reference(mo_coeff):
    return isinstance(mo_coeff, (tuple, list)) and len(mo_coeff) == 2


def _as_spin_tuple(values):
    if _is_uhf_reference(values):
        return values[0], values[1]
    return values, values


def _normalize_active_electrons(nelecas, spin):
    if isinstance(nelecas, (tuple, list)):
        if len(nelecas) != 2:
            raise ValueError('nelecas must be an int or a length-2 (nalpha, nbeta) pair.')
        na, nb = (int(nelecas[0]), int(nelecas[1]))
    else:
        nelecas = int(nelecas)
        spin = int(round(spin))
        if (nelecas + spin) % 2 != 0:
            raise ValueError(
                f'Incompatible active electron count/spin: nelecas={nelecas}, spin={spin}.'
            )
        na = (nelecas + spin) // 2
        nb = nelecas - na

    if na < 0 or nb < 0:
        raise ValueError(f'Invalid active electron count: {(na, nb)}')
    return na, nb


def _spin_occupations(mo_occ, ncore, ncas):
    if _is_uhf_reference(mo_occ):
        occ_a = np.asarray(mo_occ[0][ncore:ncore+ncas], dtype=np.int8)
        occ_b = np.asarray(mo_occ[1][ncore:ncore+ncas], dtype=np.int8)
    else:
        occ = np.asarray(mo_occ[ncore:ncore+ncas], dtype=np.int8)
        occ_a = occ_b = occ // 2
    return [occ_a, occ_b]


def _reference_active_occupations(nelecas_spin, ncas):
    na, nb = nelecas_spin
    occ_a = np.zeros(ncas, dtype=np.int8)
    occ_b = np.zeros(ncas, dtype=np.int8)
    occ_a[:na] = 1
    occ_b[:nb] = 1
    return [occ_a, occ_b]


def _slice_active_orbitals(mo_coeff, ncore, ncas):
    if _is_uhf_reference(mo_coeff):
        core = (
            mo_coeff[0][:, :ncore],
            mo_coeff[1][:, :ncore],
        )
        cas = (
            mo_coeff[0][:, ncore:ncore+ncas],
            mo_coeff[1][:, ncore:ncore+ncas],
        )
    else:
        core = mo_coeff[:, :ncore]
        cas = mo_coeff[:, ncore:ncore+ncas]
    return core, cas


def _normalize_spin_1e_operator(h1e):
    """
    Normalize a one-electron operator into explicit alpha/beta blocks.
    """
    if isinstance(h1e, (tuple, list)) and len(h1e) == 2:
        return np.asarray(h1e[0]), np.asarray(h1e[1])

    h1e = np.asarray(h1e)
    if h1e.ndim == 3 and h1e.shape[0] == 2:
        return h1e[0], h1e[1]
    return h1e, h1e


def _transform_1e_operator_ao_to_mo(h1e, mo_coeff):
    h1a, h1b = _normalize_spin_1e_operator(h1e)
    if _is_uhf_reference(mo_coeff):
        return (
            ao2mo(h1a, mo_coeff[0]),
            ao2mo(h1b, mo_coeff[1]),
        )
    out = ao2mo(h1a, mo_coeff)
    if h1b is h1a:
        return out
    return (out, ao2mo(h1b, mo_coeff))


def _validate_matching_active_spaces(cibra_obj, ciket_obj):
    if cibra_obj.ncas != ciket_obj.ncas:
        raise ValueError(
            f"CASCI objects must have the same ncas for spin-orbital TDMs: "
            f"{cibra_obj.ncas} != {ciket_obj.ncas}."
        )
    if cibra_obj.mo_cas is None or ciket_obj.mo_cas is None:
        raise ValueError("Run CASCI before requesting spin-orbital transition densities.")

    bra_mo = np.asarray(cibra_obj.mo_cas)
    ket_mo = np.asarray(ciket_obj.mo_cas)
    if bra_mo.shape != ket_mo.shape or not np.allclose(bra_mo, ket_mo):
        raise ValueError(
            "CASCI objects must share the same active orbitals for spin-orbital "
            "transition densities."
        )


def _binary_to_grouped_spin_orbital_occ(binary):
    binary = np.asarray(binary, dtype=np.int8)
    return np.concatenate((binary[0], binary[1])).astype(np.int8, copy=False)


def make_tdm1_spin_orbital(cibra, ciket, binary_bra, binary_ket, order='grouped'):
    """
    One-particle transition density matrix in a full spin-orbital basis.

    The returned matrix follows the convention

    ``D[u, v] = <Psi_bra | a_u^\dagger a_v | Psi_ket>``.

    Parameters
    ----------
    cibra, ciket : ndarray
        CI coefficient vectors in determinant bases ``binary_bra`` and
        ``binary_ket``.
    binary_bra, binary_ket : ndarray
        Determinant occupations with shape ``(ndet, 2, norb)``.
    order : {'grouped', 'interleaved'}
        Spin-orbital ordering of the returned matrix.
    """
    cibra = np.asarray(cibra)
    ciket = np.asarray(ciket)
    binary_bra = np.asarray(binary_bra, dtype=np.int8)
    binary_ket = np.asarray(binary_ket, dtype=np.int8)
    if binary_bra.ndim != 3 or binary_ket.ndim != 3 or binary_bra.shape[1] != 2 or binary_ket.shape[1] != 2:
        raise ValueError("binary_bra and binary_ket must have shape (ndet, 2, norb).")
    if binary_bra.shape[2] != binary_ket.shape[2]:
        raise ValueError("binary_bra and binary_ket must have the same number of spatial orbitals.")

    nso = 2 * binary_bra.shape[2]
    dtype = np.result_type(cibra, ciket, complex)
    tdm = np.zeros((nso, nso), dtype=dtype)

    bra_occ = [_binary_to_grouped_spin_orbital_occ(det) for det in binary_bra]
    ket_occ = [_binary_to_grouped_spin_orbital_occ(det) for det in binary_ket]
    bra_lookup = {occ.tobytes(): idx for idx, occ in enumerate(bra_occ)}

    for j, occ in enumerate(ket_occ):
        coeff_ket = ciket[j]
        if coeff_ket == 0:
            continue
        occupied = np.flatnonzero(occ)
        for v in occupied:
            sign_ann = -1 if int(np.sum(occ[:v])) % 2 else 1
            occ_after_ann = occ.copy()
            occ_after_ann[v] = 0
            unoccupied = np.flatnonzero(1 - occ_after_ann)
            for u in unoccupied:
                sign_cre = -1 if int(np.sum(occ_after_ann[:u])) % 2 else 1
                occ_final = occ_after_ann.copy()
                occ_final[u] = 1
                i = bra_lookup.get(occ_final.tobytes())
                if i is None:
                    continue
                tdm[u, v] += cibra[i].conj() * coeff_ket * (sign_ann * sign_cre)

    order = order.lower()
    if order == 'grouped':
        return tdm
    if order == 'interleaved':
        return reorder_spin_orbital_matrix(tdm, source='grouped', target='interleaved')
    raise ValueError("order must be 'grouped' or 'interleaved'.")



# def ao2mo(mf, mo_coeff=None, spin_flip=False, H1=None, H2=None):
#     """
#     Given a rhf object get Spin-Orbit Matrices

#     SF: bool
#         spin-flip

#     Returns
#     -------
#     H1: list of [h1e_a, h1e_b]
#     H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
#     """
#     # from pyscf import ao2mo

#     # mf = self.mf

#     if mo_coeff is None:
#         mo_coeff = mf.mo_coeff

#     # molecular orbitals
#     Ca, Cb = mo_coeff

#     H, energy_core = h1e_for_cas(mf, mo_coeff=mo_core)

#     self.e_core = energy_core


#     # S = (uhf_pyscf.mol).intor("int1e_ovlp")
#     # eig, v = np.linalg.eigh(S)
#     # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

#     # H1e in AO
#     # H = mf.get_hcore()
#     # H = dag(Ca) @ H @ Ca

#     # nmo = Ca.shape[1] # n

#     eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

#     ### compute SO ERIs (MO)
#     eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#     # physicts notation <pq|rs>
#     # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = eri_aa.copy()

#     eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
#     eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)


#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

#     # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
#                      # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
#     H1 = [H, H]

#     if spin_flip:
#         raise NotImplementedError('Spin-flip matrix elements not implemented yet')
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

def h1e_for_cas(mf, ncas, ncore, mo_coeff=None):
    '''CAS space effective one-electron hamiltonian

    Args:
        casci : a RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    mo_core, mo_cas = _slice_active_orbitals(mo_coeff, ncore, ncas)
    hcore = mf.get_hcore()

    if _is_uhf_reference(mo_coeff):
        hcore_a, hcore_b = _as_spin_tuple(hcore)
        mo_core_a, mo_core_b = mo_core
        mo_cas_a, mo_cas_b = mo_cas

        energy_core = mf.energy_nuc()
        if mo_core_a.size == 0:
            corevhf = (0, 0)
        else:
            core_dm = np.array((
                np.dot(mo_core_a, mo_core_a.conj().T),
                np.dot(mo_core_b, mo_core_b.conj().T),
            ))
            corevhf = mf.get_veff(core_dm)
            energy_core += np.einsum('ij,ji', core_dm[0], hcore_a).real
            energy_core += np.einsum('ij,ji', core_dm[1], hcore_b).real
            energy_core += 0.5 * np.einsum('ij,ji', core_dm[0], corevhf[0]).real
            energy_core += 0.5 * np.einsum('ij,ji', core_dm[1], corevhf[1]).real

        h1eff = np.array((
            reduce(np.dot, (mo_cas_a.conj().T, hcore_a + corevhf[0], mo_cas_a)),
            reduce(np.dot, (mo_cas_b.conj().T, hcore_b + corevhf[1], mo_cas_b)),
        ))
        return h1eff, energy_core

    mo_core = mo_core
    mo_cas = mo_cas
    energy_core = mf.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        corevhf = mf.get_veff(core_dm)
        energy_core += np.einsum('ij,ji', core_dm, hcore).real
        energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    return h1eff, energy_core


class CASCI:
    def __init__(self, mf, ncas, nelecas, ncore=None, spin=None):
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

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin

        self.nelecas = nelecas
        self.nelecas_spin = _normalize_active_electrons(nelecas, self.spin)
        self.nelecas_total = sum(self.nelecas_spin)

        ncore_electrons = mf.nelec - self.nelecas_total
        if ncore_electrons < 0 or ncore_electrons % 2 != 0:
            raise ValueError(
                'Frozen-core CASCI currently requires the inactive space to contain '
                'an even number of electrons.'
            )
        ncore = ncore_electrons // 2
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
        self.ss = None
        self.shift = None
        self.spin_purification = False


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
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space


        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None


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
        Ca, Cb = _as_spin_tuple(self.mo_cas)

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

        eri = mf.eri  # (pq||rs) = (pq|rs) - (ps|qr) 1^* 1 2^* 2

        ### compute SO antisymmetrized ERIs (MO)
        eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)
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
        if isinstance(H, np.ndarray) and H.ndim == 3:
            H1 = [H[0], H[1]]
        else:
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

        target = self.nelecas_total / 2
        self.H += shift * ((Na - target * I) @ (Na - target * I) + \
            (Nb - target * I) @ (Nb - target * I))

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
                # assert ss == s * (s+1)

                raise ValueError('s and ss cannot be specified simultaneously.')

        if ss == 0:
            # first-order spin penalty J. Phys. Chem. A 2022, 126, 12, 2050–2060
            # H' = H + J \hat{S}^2
            # norb = h1e[0].shape[0]

            # ncas = self.ncas

            # h1e = [h + 3./4 * shift * np.eye(ncas) for h in h1e]

            # for p in range(ncas):
            #     for q in range(ncas):
            #         h2e[:, :, p, q, q, p] -= 0.5 * shift
            #         h2e[:, :, p, p, q, q] -= 1./4 * shift

            self.spin_purification = True
            self.ss = 0
            self.shift = shift

            return self


        else:
            # second-order spin penalty
            raise NotImplementedError('Second-order spin panelty not implemented.')


    def run(self, nstates=1, mo_coeff=None, method='ci', ci0=None):
        """
        solve the full CI in the active space, more efficient than the JW solver

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

        # if method == 'ci':

        ncore = self.ncore
        ncas = self.ncas

        # define the core and active space orbitals
        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff # use HF MOs
            # self.mo_core = self.mo_coeff[:, :ncore]
            # self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]

        else:
            self.mo_coeff = mo_coeff

        self.mo_core, self.mo_cas = _slice_active_orbitals(self.mo_coeff, ncore, ncas)


        if self.binary is None:
            mo_occ = _reference_active_occupations(self.nelecas_spin, ncas)
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary
        else:
            binary = self.binary

        # print('Number of determinants', binary.shape[0])

        # effective hamiltonian in the CAS
        h1e, h2e = self.get_SO_matrix()

        if self.spin_purification:

            # logging.info('Purify spin by energy penalty')

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
        self.eri_so = h2e


        SC1, SC2 = SlaterCondon(binary)
        self.SC1 = SC1
        self.SC2 = SC2

        I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


        H_CI = CI_H(binary, h1e, h2e, SC1, SC2)
        E, X = eigsh(H_CI, k=nstates, which='SA')


        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(nstates)]

        for i in range(nstates):
            ss = spin_square(*self.make_rdm12(i))
            print("CASCI Root {}  E = {:.10f}  S^2 = {:.6f}".format(i, self.e_tot[i], ss))

        return self

    def exact_diag(self, nstates=1, mo_coeff=None, ci0=None):
        # elif method == 'jw':

        # exact diagonalization by JW transform

        H = self.qubitization()
        E, X = eigsh(H, k=nstates, which='SA')

    # else:
    #     raise ValueError("There is no {} solver for CASCI. Use 'ci' or 'jw'".format(method))
        return E, X

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
            h1e = _transform_1e_operator_ao_to_mo(h1e, self.mf.mo_coeff)

        ncore = self.ncore
        ncas = self.ncas

        h1a, h1b = _normalize_spin_1e_operator(h1e)
        if ncore > 0:
            c_core = np.trace(h1a[:ncore, :ncore]) + np.trace(h1b[:ncore, :ncore])
        else:
            c_core = 0

        h1e = (
            h1a[ncore:ncas+ncore, ncore:ncas+ncore],
            h1b[ncore:ncas+ncore, ncore:ncas+ncore],
        )

        c_cas = contract_with_rdm1(ci, self.binary, self.SC1, h1e=h1e)

        return c_core + c_cas

    def contract_with_rdm1(self, state_id, h1e=None, representation='ao'):
        return self.make_rdm1_contract(state_id, h1e=h1e, representation=representation)

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

        ci = self.ci[state_id]
        return make_rdm1s(ci, self.binary, self.SC1)

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

        return spin_square(*self.make_rdm12(state_id))



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
            return self.contract_with_rdm1(bra_id, h1e=h1e, representation=representation)

        if representation.lower() == 'ao':
            h1e = _transform_1e_operator_ao_to_mo(h1e, self.mf.mo_coeff)

        h1a, h1b = _normalize_spin_1e_operator(h1e)
        ncore = self.ncore
        ncas = self.ncas

        if ncore > 0:
            c_core = np.trace(h1a[:ncore, :ncore]) + np.trace(h1b[:ncore, :ncore])
        else:
            c_core = 0

        h1e_cas = (
            h1a[ncore:ncas+ncore, ncore:ncas+ncore],
            h1b[ncore:ncas+ncore, ncore:ncas+ncore],
        )
        c_cas = contract_with_tdm1(self.ci[bra_id], self.ci[ket_id], self.binary, self.SC1, h1e_cas)

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

    def make_tdm1s(self, bra_id, ket_id=0):
        """
        Spin-resolved one-particle transition density matrices in MO basis.
        """
        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]

        return make_tdm1s(cibra, ciket, self.binary, self.SC1)

    def make_tdm1_spin_orbital(self, bra_id, ket_id=0, other=None, order='grouped'):
        """
        One-particle transition density matrix in a full spin-orbital basis.

        Parameters
        ----------
        bra_id : int
            Bra-state index on ``self``.
        ket_id : int, optional
            Ket-state index on ``other``. Defaults to ``0``.
        other : CASCI, optional
            Ket-side CASCI object. Defaults to ``self``.
        order : {'grouped', 'interleaved'}
            Spin-orbital ordering of the returned matrix.
        """
        if other is None:
            other = self
        _validate_matching_active_spaces(self, other)
        return make_tdm1_spin_orbital(
            self.ci[bra_id],
            other.ci[ket_id],
            self.binary,
            other.binary,
            order=order,
        )

    def contract_with_tdm1_spin_orbital(self, bra_id, ket_id=0, h1e=None, other=None,
                                        order='grouped'):
        """
        Contract a spin-orbital one-body operator with a CASCI transition density.

        Parameters
        ----------
        bra_id : int
            Bra-state index on ``self``.
        ket_id : int, optional
            Ket-state index on ``other``. Defaults to ``0``.
        h1e : ndarray
            One-body operator in the active spin-orbital basis.
        other : CASCI, optional
            Ket-side CASCI object. Defaults to ``self``.
        order : {'grouped', 'interleaved'}
            Ordering shared by ``h1e`` and the returned TDM.
        """
        if h1e is None:
            raise ValueError("h1e is required for spin-orbital contractions.")
        tdm = self.make_tdm1_spin_orbital(bra_id, ket_id=ket_id, other=other, order=order)
        h1e = np.asarray(h1e)
        if h1e.shape != tdm.shape:
            raise ValueError(
                f"h1e shape {h1e.shape} is incompatible with spin-orbital TDM shape {tdm.shape}."
            )
        return np.einsum('uv,uv->', h1e, tdm, optimize=True)

    def soc_matrix_element(self, bra_id, ket_id=0, other=None, hso=None,
                           one_center=True, with_prefactor=True,
                           light_speed=None, order='grouped'):
        """
        One-electron SOC matrix element between CASCI states.

        If ``hso`` is not provided, the active-space SOC operator is built from
        the current active orbitals.
        """
        if other is None:
            other = self
        _validate_matching_active_spaces(self, other)
        if hso is None:
            hso = get_soc_1e_spin_orbital(
                self.mf,
                representation='mo',
                mo_coeff=self.mo_cas,
                one_center=one_center,
                with_prefactor=with_prefactor,
                light_speed=light_speed,
                order=order,
            )
        return self.contract_with_tdm1_spin_orbital(
            bra_id,
            ket_id=ket_id,
            h1e=hso,
            other=other,
            order=order,
        )

    def make_tdm2(self, bra_id, ket_id=0):
        """
        spin-traced 1e transition density matrix in MO

        .. math::

            \gamma_{pq}^{\beta \alpha} = <\Psi_\beta | \hat{E}_{qp} | \Psi_\alpha >

        E_{qp} = q_alpha^\dagger p_alpha + q_beta^\dagger p_beta
        """
        raise NotImplementedError('TDM not implemented')


# def get_SO_matrix(mo_coeff, eri, spin_flip=False, H1=None, H2=None):
#     """
#     Given a rhf object get Spin-Orbit Matrices

#     SF: bool
#         spin-flip
#     """

#     Ca, Cb = mo_coeff

#     H, energy_core = h1e_for_cas(mf, ncas=ncas, ncore=ncore, \
#                                  mo_coeff=mo_coeff)

#     # self.e_core = energy_core



#     eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

#     ### compute SO ERIs (MO)
#     eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#     # physicts notation <pq|rs>
#     # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = eri_aa.copy()

#     eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
#     eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)


#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

#     H1 = [H, H]

#     if spin_flip:
#         raise NotImplementedError('Spin-flip matrix elements not implemented yet')

#     return H1, H2

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

def size_of_cas(norb, nelec, basis='sd', S=0):
    """
    size of CAS

    Eq. 91, 92 Chem Rev 2012, 112, 108

    Parameters
    ----------
    norb : TYPE
        DESCRIPTION.
    nelec : TYPE
        DESCRIPTION.
    basis : TYPE, optional
        DESCRIPTION. The default is 'sd'.
    S : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from math import comb
    # if isinstance(norb, int): norb = [norb, ] * 2
    if isinstance(nelec, int): nelec = [nelec, ] * 2


    # norb_a, norb_b = norb
    na, nb = nelec
    if basis == 'sd':
        return comb(norb, na) * comb(norb, nb)
    elif basis == 'csf':
        N = na + nb
        return (2*S+1)/(norb + 1) * comb(norb+1, N//2 - S) * comb(norb+1, N//2+S+1)

def spin_square(dm1, dm2):
    """

    Compute the total spin S^2, require 2e RDM

    Ref:
        J. Chem. Theory Comput. 2021, 17, 5684−5703


    For a single SO,
    .. math::

            S_i^2 = \frac{3}{4} (E_ii - e_{ii,ii})
            S_i \cdot S_j = -frac{1}{2} (e_{ij,ji} + \frac{1}{2} e_{ii, jj}), j \ne i

    where E_{ij}, e_{ijkl} are 1 and 2e RDMs.

    Parameters
    ----------
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    spin_square = (0.75*np.einsum("ii", dm1)
               - 0.5*np.einsum("ijji", dm2)
               - 0.25*np.einsum("iijj", dm2))

    return spin_square

# def ss_matrix():

#     ss = CI_H(binary, H1, H2, SC1, SC2)

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

    h1e = _normalize_spin_1e_operator(h1e)

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

    h1e = _normalize_spin_1e_operator(h1e)

    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    H = np.einsum("Spp, ISp -> I", h1e, binary, optimize=True)
    H = np.diag(H)

    ## Rule 1
    H[I_A, J_A] -= np.einsum("pq, Kp, Kq -> K", h1e[0], a_t, a, optimize=True)
    H[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", h1e[1], b_t, b, optimize=True)


    return np.einsum('I, IJ, J -> ', ci.conj(), H, ci)

def _build_spin_rdm1_operators(binary, SC1):
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

    nsd, _, nmo = binary.shape
    H_a = np.zeros((nsd, nsd, nmo, nmo))
    H_b = np.zeros((nsd, nsd, nmo, nmo))

    for I in range(nsd):
        for p in range(nmo):
            H_a[I, I, p, p] = binary[I, 0, p]
            H_b[I, I, p, p] = binary[I, 1, p]

    H_a[I_A, J_A] -= np.einsum("Kp, Kq -> Kpq", a_t, a, optimize=True)
    H_b[I_B, J_B] -= np.einsum("Kp, Kq -> Kpq", b_t, b, optimize=True)

    return H_a, H_b


def make_rdm1s(ci, binary, SC1):
    """
    Make spin-resolved 1e RDMs in MO basis.
    """
    H_a, H_b = _build_spin_rdm1_operators(binary, SC1)
    dm1a = np.einsum('I, IJpq, J -> pq', ci.conj(), H_a, ci).T
    dm1b = np.einsum('I, IJpq, J -> pq', ci.conj(), H_b, ci).T
    return dm1a, dm1b


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
    dm1a, dm1b = make_rdm1s(ci, binary, SC1)
    return dm1a + dm1b


def make_tdm1s(cibra, ciket, binary, SC1):
    """
    Make spin-resolved 1e TDMs in MO basis.
    """
    H_a, H_b = _build_spin_rdm1_operators(binary, SC1)
    tdm1a = np.einsum('I, IJpq, J -> pq', cibra.conj(), H_a, ciket)
    tdm1b = np.einsum('I, IJpq, J -> pq', cibra.conj(), H_b, ciket)
    return tdm1a, tdm1b

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
    tdm1a, tdm1b = make_tdm1s(cibra, ciket, binary, SC1)
    return tdm1a + tdm1b





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
        try:
            from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric
            s = overlap_integral_asymmetric(cibra.mol._bas, ciket.mol._bas)
        except (ImportError, AttributeError, TypeError):
            from pyscf import gto
            mol_bra = cibra.mol.topyscf()
            mol_ket = ciket.mol.topyscf()
            mol_bra.build()
            mol_ket.build()
            s = gto.intor_cross('int1e_ovlp', mol_bra, mol_ket)
        s = reduce(np.dot, (cibra.mf.mo_coeff.T, s, ciket.mf.mo_coeff))


    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    dtype = np.result_type(s, np.asarray(cibra.ci), np.asarray(ciket.ci))
    S = np.zeros((nsd_bra, nsd_ket), dtype=dtype) # overlap between determinants

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore
    if ncore_bra != ncore_ket:
        raise ValueError(
            "Different numbers of core orbitals are not supported in overlap: "
            f"{ncore_bra} != {ncore_ket}."
        )

    scc = s[:ncore_bra, :ncore_ket]
    sca = s[:ncore_bra, ncore_ket:]
    sac = s[ncore_bra:, :ncore_ket]
    saa = s[ncore_bra:, ncore_ket:]

    if ncore_bra == 0:
        core_factor = dtype.type(1)
        saa_eff = saa
    else:
        scc_det = np.linalg.det(scc)
        core_factor = scc_det * scc_det
        saa_eff = saa - sac @ np.linalg.solve(scc, sca)

    occ_bra_a = [np.flatnonzero(cibra.binary[I, 0]) for I in range(nsd_bra)]
    occ_bra_b = [np.flatnonzero(cibra.binary[I, 1]) for I in range(nsd_bra)]
    occ_ket_a = [np.flatnonzero(ciket.binary[J, 0]) for J in range(nsd_ket)]
    occ_ket_b = [np.flatnonzero(ciket.binary[J, 1]) for J in range(nsd_ket)]

    for I in range(nsd_bra):
        occidx1_a = occ_bra_a[I]
        occidx1_b = occ_bra_b[I]

        for J in range(nsd_ket):
            occidx2_a = occ_ket_a[J]
            occidx2_b = occ_ket_b[J]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.
            S[I, J] = (
                core_factor
                * np.linalg.det(saa_eff[np.ix_(occidx1_a, occidx2_a)])
                * np.linalg.det(saa_eff[np.ix_(occidx1_b, occidx2_b)])
            )



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
    mol2 = Molecule(atom = [
    ['Li' , (0. , 0. , 0)],
    ['Li' , (0. , 0. , 1.4)], ])
    mol2.basis = '631g'

    # mol.unit = 'b'
    mol2.build()

    mf2 = mol2.RHF().run()


    ncas, nelecas = (4,2)
    mc = CASCI(mf2, ncas, nelecas)
    mc.run(5)

    print('Fix spin by penalty')

    # mc = CASCI(mf2, ncas, nelecas)
    mc.fix_spin(ss=0, shift=0.2)
    mc.run(5)

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
