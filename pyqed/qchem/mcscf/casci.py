#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:07 2024

complete active space configuration interaction

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import logging
from functools import lru_cache, reduce
import importlib
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

import sys
from opt_einsum import contract

from pyqed import tensor
from itertools import combinations
import itertools
import warnings
from dataclasses import dataclass
from numba import njit

from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct, get_fci_combos, SlaterCondon, CI_H
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo, get_or_build_low_rank_eri_factors
from pyqed.qchem.soc import (
    get_soc_1e_spin_orbital,
    get_soc_somf_spin_orbital,
    reorder_spin_orbital_matrix,
)

_CASSCF_CPP_UNINITIALIZED = object()
_casscf_cpp = _CASSCF_CPP_UNINITIALIZED


def _cpp_attr(*names):
    global _casscf_cpp
    if _casscf_cpp is _CASSCF_CPP_UNINITIALIZED:
        try:
            _casscf_cpp = importlib.import_module("pyqed.qchem._casscf_cpp")
        except Exception:  # pragma: no cover - optional accelerator
            _casscf_cpp = None
    if _casscf_cpp is None:
        return None
    for name in names:
        attr = getattr(_casscf_cpp, name, None)
        if attr is not None:
            return attr
    return None


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


def _validate_multiplicity(multiplicity):
    if multiplicity is None:
        return None
    multiplicity = int(round(multiplicity))
    if multiplicity < 1:
        raise ValueError("multiplicity must be a positive integer.")
    return multiplicity


def _s2_from_multiplicity(multiplicity):
    multiplicity = _validate_multiplicity(multiplicity)
    if multiplicity is None:
        return None
    spin_s = 0.5 * (multiplicity - 1)
    return spin_s * (spin_s + 1.0)


def _resolve_ms2(nelecas, mol_ms2, *, spin=None, ms2=None, multiplicity=None):
    if ms2 is None:
        ms2 = mol_ms2 if spin is None else spin
    elif spin is not None and int(round(spin)) != int(round(ms2)):
        raise ValueError("spin and ms2 both set but disagree; use ms2 for 2*M_S.")

    ms2 = int(round(ms2))
    multiplicity = _validate_multiplicity(multiplicity)
    if multiplicity is not None:
        spin2 = multiplicity - 1
        if abs(ms2) > spin2 or (spin2 - ms2) % 2:
            raise ValueError(
                f"multiplicity={multiplicity} is incompatible with ms2={ms2}."
            )
    return ms2


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


def _get_mf_cholesky_factors(mf):
    eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is not None:
        return eri_factors

    tol = getattr(mf, 'cholesky_tol', None)
    if tol is None:
        tol = getattr(mf, 'low_rank_tol', None)
    if tol is None:
        tol = 1e-8

    max_rank = getattr(mf, 'cholesky_max_rank', None)
    if max_rank is None:
        max_rank = getattr(mf, 'low_rank_max_rank', None)

    eri_factors = get_or_build_low_rank_eri_factors(mf.mol, tol=tol, max_rank=max_rank)
    mf.eri_factors = eri_factors
    return eri_factors


def _resolve_use_cholesky_integrals(mf, use_cholesky=None):
    """
    Enable the factor/Cholesky CASCI path automatically for factor-only RHF.
    """
    if use_cholesky is not None:
        return bool(use_cholesky)
    if bool(getattr(mf, 'cholesky_jk', False)):
        return True
    return getattr(mf, 'eri', None) is None and (
        getattr(mf, 'eri_factors', None) is not None
        or getattr(getattr(mf, 'mol', None), 'eri_factors', None) is not None
    )


def mo_pair_factors(eri_factors, mo_left, mo_right=None):
    """
    Transform AO Cholesky factors ``L_P[mu,nu]`` to an MO pair basis.
    """
    if mo_right is None:
        mo_right = mo_left
    from pyqed.qchem.basis import mo_pair_factors as _basis_mo_pair_factors
    return _basis_mo_pair_factors(eri_factors, mo_left, mo_right)


def transform_eri_factors_to_mo_pair(eri_factors, mo_left, mo_right=None):
    """
    Compatibility alias for :func:`mo_pair_factors`.
    """
    return mo_pair_factors(eri_factors, mo_left, mo_right)


def assemble_spatial_eri_from_factors(pair_factors_pq, pair_factors_rs=None):
    """
    Assemble a spatial ERI tensor from transformed Cholesky pair factors.
    """
    if pair_factors_rs is None:
        pair_factors_rs = pair_factors_pq
    return contract('Ppq,Prs->pqrs', pair_factors_pq, pair_factors_rs)


def _get_veff_for_dm(mf, dm):
    """Return the mean-field potential for ``dm`` across local and PySCF APIs."""
    try:
        return mf.get_veff(dm=dm)
    except TypeError:
        pass
    mol = getattr(mf, 'mol', None)
    if mol is not None:
        try:
            return mf.get_veff(mol, dm)
        except TypeError:
            pass
    return mf.get_veff(dm)


def transform_spatial_eri_to_mo(mf, mo_left, mo_right=None, mo_left_2=None, mo_right_2=None,
                                use_cholesky=False, eri_factors=None):
    """
    Transform AO spatial ERIs to MO form, optionally via AO Cholesky factors.
    """
    if mo_right is None:
        mo_right = mo_left
    if mo_left_2 is None:
        mo_left_2 = mo_left
    if mo_right_2 is None:
        mo_right_2 = mo_right

    if use_cholesky:
        if eri_factors is None:
            eri_factors = _get_mf_cholesky_factors(mf)
        pair_factors_pq = mo_pair_factors(eri_factors, mo_left, mo_right)
        if (
            mo_left_2 is mo_left and mo_right_2 is mo_right
        ) or (
            np.array_equal(mo_left_2, mo_left) and np.array_equal(mo_right_2, mo_right)
        ):
            pair_factors_rs = pair_factors_pq
        else:
            pair_factors_rs = mo_pair_factors(eri_factors, mo_left_2, mo_right_2)
        return assemble_spatial_eri_from_factors(pair_factors_pq, pair_factors_rs)

    eri_source = getattr(mf, 'eri', None)
    if eri_source is None and getattr(mf, 'eri_s4', None) is not None:
        from pyqed.qchem.basis import unpack_eri_s4
        eri_source = unpack_eri_s4(mf.eri_s4, mf.mol.nao)
    if eri_source is None and getattr(mf, 'eri_s8', None) is not None:
        from pyqed.qchem.basis import unpack_eri_s8
        eri_source = unpack_eri_s8(mf.eri_s8, mf.mol.nao)
    mol = getattr(mf, 'mol', None)
    if eri_source is None and mol is not None:
        eri_source = getattr(mol, 'eri', None)
    if eri_source is None and mol is not None and getattr(mol, 'eri_s4', None) is not None:
        from pyqed.qchem.basis import unpack_eri_s4
        eri_source = unpack_eri_s4(mol.eri_s4, mol.nao)
    if eri_source is None and mol is not None and getattr(mol, 'eri_s8', None) is not None:
        from pyqed.qchem.basis import unpack_eri_s8
        eri_source = unpack_eri_s8(mol.eri_s8, mol.nao)
    eri_ndim = None if eri_source is None else np.asarray(eri_source).ndim
    if eri_source is not None and eri_ndim == 4:
        return contract(
            'ip, jq, ijkl, kr, ls -> pqrs',
            mo_left.conj(),
            mo_right,
            eri_source,
            mo_left_2.conj(),
            mo_right_2,
        )

    if eri_factors is None:
        eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(getattr(mf, 'mol', None), 'eri_factors', None)
    if eri_factors is None and eri_source is not None and eri_ndim in (2, 3):
        eri_factors = eri_source
    if eri_factors is not None:
        pair_factors_pq = mo_pair_factors(eri_factors, mo_left, mo_right)
        if (
            mo_left_2 is mo_left and mo_right_2 is mo_right
        ) or (
            np.array_equal(mo_left_2, mo_left) and np.array_equal(mo_right_2, mo_right)
        ):
            pair_factors_rs = pair_factors_pq
        else:
            pair_factors_rs = mo_pair_factors(eri_factors, mo_left_2, mo_right_2)
        return assemble_spatial_eri_from_factors(pair_factors_pq, pair_factors_rs)

    if eri_source is None and mol is not None and type(mol).__module__.startswith('pyscf.'):
        try:
            from pyscf import ao2mo as pyscf_ao2mo
        except ImportError:
            pass
        else:
            dims = (
                mo_left.shape[1],
                mo_right.shape[1],
                mo_left_2.shape[1],
                mo_right_2.shape[1],
            )
            return np.asarray(
                pyscf_ao2mo.kernel(
                    mol,
                    (mo_left, mo_right, mo_left_2, mo_right_2),
                    compact=False,
                )
            ).reshape(dims)

    raise ValueError(
        "transform_spatial_eri_to_mo requires dense 4-index ERIs or RI/Cholesky "
        "factors. Got mf.eri with ndim={}.".format(eri_ndim)
    )


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
            corevhf = _get_veff_for_dm(mf, core_dm)
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
        corevhf = _get_veff_for_dm(mf, core_dm)
        energy_core += np.einsum('ij,ji', core_dm, hcore).real
        energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    return h1eff, energy_core


class CASCI:
    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        spin=None,
        ms2=None,
        multiplicity=None,
        verbose=0,
    ):
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
        self.verbose = int(verbose)

        self.ms2 = _resolve_ms2(
            nelecas,
            mf.mol.spin,
            spin=spin,
            ms2=ms2,
            multiplicity=multiplicity,
        )
        self.spin = self.ms2  # backward-compatible alias for 2*M_S
        self.multiplicity = _validate_multiplicity(multiplicity)
        self.target_s2 = _s2_from_multiplicity(self.multiplicity)
        self.spin_selection_tol = 1.0e-5
        self.spin_root_cushion = 8

        self.nelecas = nelecas
        self.nelecas_spin = _normalize_active_electrons(nelecas, self.ms2)
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
        self.use_cholesky_integrals = False
        self._property_operator_cache = {}
        self.active_symmetry = None
        self.active_orb_irrep_labels = None
        self.active_orb_sym = None
        self.active_irrep_counts = None
        self.det_irrep_labels = None
        self.det_irrep_ids = None
        self.det_irrep_counts = None
        self.wfnsym = None
        self.det_irrep_filter_indices = None
        self.spin0_symm_transform = None
        self.spin0_pair_indices = None


        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None

    def _clear_symmetry_metadata(self):
        self.active_symmetry = None
        self.active_orb_irrep_labels = None
        self.active_orb_sym = None
        self.active_irrep_counts = None
        self.det_irrep_labels = None
        self.det_irrep_ids = None
        self.det_irrep_counts = None
        self.wfnsym = None
        self.det_irrep_filter_indices = None
        self.spin0_symm_transform = None
        self.spin0_pair_indices = None

    def _update_symmetry_metadata(self, binary=None):
        if getattr(self.mol, "symmetry_info", None) is None:
            self._clear_symmetry_metadata()
            return None
        try:
            from pyqed.qchem.symmetry import build_active_space_symmetry

            info = build_active_space_symmetry(
                self.mf,
                self.ncore,
                self.ncas,
                mo_coeff=self.mo_coeff,
                binary=self.binary if binary is None else binary,
            )
        except NotImplementedError:
            self._clear_symmetry_metadata()
            return None

        self.active_symmetry = info
        self.active_orb_irrep_labels = info.orbital_labels
        self.active_orb_sym = info.orbital_ids
        self.active_irrep_counts = info.orbital_counts
        self.det_irrep_labels = info.determinant_labels
        self.det_irrep_ids = info.determinant_ids
        self.det_irrep_counts = info.determinant_counts
        return info

    def _filter_binary_by_irrep(self, binary, target_irrep=None, wfnsym=None):
        target = target_irrep if target_irrep is not None else wfnsym
        if target is None:
            self.det_irrep_filter_indices = None
            self.wfnsym = None
            self._update_symmetry_metadata(binary)
            return binary

        info = self._update_symmetry_metadata(binary)
        if info is None or info.determinant_ids is None:
            raise ValueError("Wavefunction symmetry filtering requires molecular symmetry metadata.")
        group = info.group
        target_id = group.irrep_id(target) if isinstance(target, str) else int(target)
        from pyqed.qchem.symmetry import irrep_id_to_name

        target_label = irrep_id_to_name(group, target_id)
        indices = np.asarray(
            [idx for idx, irrep_id in enumerate(info.determinant_ids) if irrep_id == target_id],
            dtype=int,
        )
        if indices.size == 0:
            raise ValueError(
                f"No determinants with irrep {target_label} "
                f"in the CAS({self.nelecas},{self.ncas}) determinant basis."
            )

        filtered = np.asarray(binary, dtype=np.int8)[indices]
        self.binary = filtered
        self.wfnsym = target_label
        self.det_irrep_filter_indices = indices
        self._update_symmetry_metadata(filtered)
        self.det_irrep_filter_indices = indices
        self.wfnsym = target_label
        return filtered

    def _spin0_symm_pairs(self, binary):
        """Build symmetric alpha/beta determinant-index pairs."""
        if tuple(self.nelecas_spin)[0] != tuple(self.nelecas_spin)[1]:
            raise ValueError("direct_spin0_symm requires N_alpha == N_beta.")

        dets = np.asarray(binary, dtype=np.int8)
        if dets.ndim != 3 or dets.shape[1] != 2:
            raise ValueError("binary must have shape (ndet, 2, ncas).")

        string_order = []
        string_seen = set()
        for det in dets:
            for spin in (0, 1):
                key = tuple(int(x) for x in det[spin])
                if key not in string_seen:
                    string_seen.add(key)
                    string_order.append(key)

        det_lookup = {}
        for idx, det in enumerate(dets):
            alpha = tuple(int(x) for x in det[0])
            beta = tuple(int(x) for x in det[1])
            det_lookup[(alpha, beta)] = idx

        pairs = []
        for ia, alpha in enumerate(string_order):
            for beta in string_order[ia:]:
                left = det_lookup.get((alpha, beta))
                if left is None:
                    continue
                right = det_lookup.get((beta, alpha))
                if right is None:
                    continue
                pairs.append((left, right))

        if not pairs:
            raise ValueError("No singlet spin-adapted configurations were found.")
        return tuple(pairs)

    def _spin0_symm_basis(self, binary):
        """Build symmetric alpha/beta string-pair columns in determinant space."""
        dets = np.asarray(binary, dtype=np.int8)
        pairs = self._spin0_symm_pairs(dets)
        transform = np.zeros((dets.shape[0], len(pairs)), dtype=float)
        for col, (left, right) in enumerate(pairs):
            if left == right:
                transform[left, col] = 1.0
            else:
                transform[left, col] = 2.0 ** -0.5
                transform[right, col] = 2.0 ** -0.5
        return transform, tuple(pairs)

    def _direct_spin0_symm_dense(
        self,
        binary,
        requested_nstates,
        *,
        use_cholesky=None,
    ):
        """Dense PySCF-like singlet spin-adapted CI with optional spatial symmetry."""
        transform, pairs = self._spin0_symm_basis(binary)
        if int(requested_nstates) > transform.shape[1]:
            raise ValueError(
                f"Requested {requested_nstates} spin0 roots but only "
                f"{transform.shape[1]} singlet configurations are available."
            )

        h1e, h2e = self.get_SO_matrix(use_cholesky=use_cholesky)
        if self.spin_purification:
            raise ValueError("direct_spin0_symm already targets singlets; do not combine it with fix_spin().")
        if h2e is not None:
            h2e[0, 0] -= h2e[0, 0].swapaxes(1, 3)
            h2e[1, 1] -= h2e[1, 1].swapaxes(1, 3)

        self.hcore = h1e
        self.eri_so = h2e
        self.h2e_cas = None
        self.spin0_symm_transform = transform
        self.spin0_pair_indices = pairs

        SC1, SC2 = SlaterCondon(binary)
        self.SC1 = SC1
        self.SC2 = SC2
        H_CI = CI_H(binary, h1e, h2e, SC1, SC2)
        H_spin0 = transform.T @ np.asarray(H_CI) @ transform
        energies, vecs_spin0 = self._lowest_dense_eigensystem(H_spin0, requested_nstates)
        return energies, transform @ vecs_spin0

    def as_scanner(
        self,
        nstates=None,
        method='direct_ci',
        build_driver=None,
        run_kwargs=None,
        reuse_ci=False,
        **kwargs,
    ):
        """Return a stateful CASCI scanner for nearby geometries.

        The scanner reuses the underlying mean-field scanner, then runs a fresh
        CASCI calculation at each geometry. It returns the computed CASCI object
        so downstream LDR code can use energies, overlaps, and density matrices
        from the same electronic calculation.
        """
        options = dict(run_kwargs or {})
        options.update(kwargs)
        return CASCIScanner(
            self,
            nstates=nstates,
            method=method,
            build_driver=build_driver,
            run_kwargs=options,
            reuse_ci=reuse_ci,
        )


    def PCM(self, solvent_obj=None, dm=None, **kwargs):
        """
        Attach a PCM solvent model and return a chainable CASCI+PCM object.

        Examples
        --------
        >>> mc = CASCI(mf, ncas=6, nelecas=6).PCM(eps=2.3653).run(nstates=4)

        Parameters
        ----------
        solvent_obj
            Optional preconfigured ``pyqed.qchem.solvent.pcm.PCM`` object.
            If omitted, a PCM object is created for ``self.mol``.
        dm
            Optional density matrix used to freeze the solvent potential.
        **kwargs
            PCM attributes to set on the solvent object, such as ``eps``,
            ``method``, ``state_id``, ``state_average``, ``state_weights``,
            ``max_cycle``, or ``conv_tol``.
        """
        from pyqed.qchem.solvent.pcm import PCM, pcm_for_casci

        if solvent_obj is None:
            solvent_obj = PCM(self.mol)

        for key, value in kwargs.items():
            if not hasattr(solvent_obj, key):
                raise ValueError(f"Unknown PCM option '{key}'.")
            setattr(solvent_obj, key, value)

        return pcm_for_casci(self, solvent_obj, dm)

    def _make_lr_pcm_fast_solvent(self, eps):
        if _is_uhf_reference(self.mo_coeff):
            raise NotImplementedError(
                "Determinant-space LR-PCM currently supports restricted CASCI references only."
            )
        from pyqed.qchem.solvent.pcm import PCM

        solvent = PCM(self.mol)
        reference = getattr(self, "with_solvent", None)
        if reference is not None:
            for key in (
                "method",
                "vdw_scale",
                "r_probe",
                "radii_table",
                "lebedev_order",
                "max_memory",
                "verbose",
            ):
                if hasattr(reference, key):
                    setattr(solvent, key, getattr(reference, key))
        solvent.eps = float(eps)
        solvent.equilibrium_solvation = False
        return solvent

    def _active_tdm_to_ao(self, active_tdm):
        full_mo = np.zeros((int(self.mf.nmo), int(self.mf.nmo)), dtype=float)
        ncore = int(self.ncore)
        ncas = int(self.ncas)
        full_mo[ncore:ncore + ncas, ncore:ncore + ncas] = active_tdm
        coeff = np.asarray(self.mo_coeff)
        return coeff @ full_mo @ coeff.conj().T

    def _lr_pcm_determinant_kernel(self, ground_ci, eps=1.78):
        """
        Build the LR-PCM response kernel in the CAS determinant basis.

        The kernel is constructed from transition densities between each
        determinant-basis vector and a fixed ground-state CI vector, then
        projected to leave the reference ground vector unchanged.
        """
        solvent = self._make_lr_pcm_fast_solvent(eps)
        ndet = len(ground_ci)
        tdms = []
        potentials = []
        for idx in range(ndet):
            unit = np.zeros(ndet, dtype=float)
            unit[idx] = 1.0
            active_tdm = make_tdm1(unit, ground_ci, self.binary, self.SC1)
            tdm_ao = self._active_tdm_to_ao(active_tdm)
            tdms.append(tdm_ao)
            potentials.append(solvent._B_dot_x(tdm_ao))

        kernel = np.empty((ndet, ndet), dtype=float)
        for i, tdm_i in enumerate(tdms):
            for j, v_j in enumerate(potentials):
                kernel[i, j] = np.einsum("ij,ji->", v_j, tdm_i, optimize=True).real
        kernel = 0.5 * (kernel + kernel.T)

        ground_ci = np.asarray(ground_ci, dtype=float)
        ground_ci = ground_ci / np.linalg.norm(ground_ci)
        projector = np.eye(ndet) - np.outer(ground_ci, ground_ci)
        return projector @ kernel @ projector

    @staticmethod
    def _lowest_dense_eigensystem(matrix, nstates):
        nstates = int(nstates)
        if nstates <= 0:
            raise ValueError("nstates must be positive.")
        evals, evecs = eigh(np.asarray(matrix))
        nout = min(nstates, evals.size)
        return evals[:nout], evecs[:, :nout]

    def _spin_selected_nstates(self, requested_nstates, ndet, spin_root_cushion=None):
        requested_nstates = int(requested_nstates)
        if self.multiplicity is None:
            return requested_nstates
        if spin_root_cushion is None:
            spin_root_cushion = self.spin_root_cushion
        nsolve = requested_nstates + max(0, int(spin_root_cushion))
        return min(nsolve, max(1, int(ndet) - 1))

    def _apply_multiplicity_selection(self, energies, vectors, requested_nstates,
                                      spin_selection_tol=None):
        energies = np.asarray(energies)
        vectors = np.asarray(vectors)
        requested_nstates = int(requested_nstates)
        if self.multiplicity is None:
            order = np.argsort(energies)[:requested_nstates]
            return energies[order], vectors[:, order]

        if spin_selection_tol is None:
            spin_selection_tol = self.spin_selection_tol
        spin_selection_tol = float(spin_selection_tol)

        old_e_tot = self.e_tot
        old_ci = self.ci
        self.e_tot = energies + self.e_core
        self.ci = [vectors[:, i] for i in range(vectors.shape[1])]
        s2 = np.asarray([self.spin_square(i) for i in range(vectors.shape[1])], dtype=float)
        self.e_tot = old_e_tot
        self.ci = old_ci

        target_s2 = self.target_s2
        selected = [
            i for i in np.argsort(energies)
            if abs(s2[i] - target_s2) <= spin_selection_tol
        ]
        if len(selected) < requested_nstates:
            ranked = sorted(
                range(len(energies)),
                key=lambda i: (abs(s2[i] - target_s2), energies[i]),
            )
            detail = ", ".join(
                f"root {i}: S^2={s2[i]:.6g}" for i in ranked[:min(6, len(ranked))]
            )
            raise ValueError(
                f"Found {len(selected)} roots with multiplicity={self.multiplicity} "
                f"(target S^2={target_s2:.6g}) among {len(energies)} solved roots; "
                f"need {requested_nstates}. Increase spin_root_cushion or nstates. "
                f"Closest roots: {detail}."
            )
        selected = selected[:requested_nstates]
        return energies[selected], vectors[:, selected]


    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None, use_cholesky=None):
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
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(mf, use_cholesky)

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

        same_spin_orbitals = Ca is Cb or np.array_equal(Ca, Cb)
        eri_factors = _get_mf_cholesky_factors(mf) if use_cholesky else None

        if same_spin_orbitals:
            # Restricted references use the same spatial active orbitals for both
            # spin channels, so all spin blocks share one spatial ERI transform.
            eri_spatial = transform_spatial_eri_to_mo(
                mf, Ca, Ca, Ca, Ca,
                use_cholesky=use_cholesky,
                eri_factors=eri_factors,
            )
            eri_aa = eri_spatial
            eri_ab = eri_spatial
            eri_ba = eri_spatial
            eri_bb = eri_spatial
        else:
            eri_aa = transform_spatial_eri_to_mo(
                mf, Ca, Ca, Ca, Ca,
                use_cholesky=use_cholesky,
                eri_factors=eri_factors,
            )
            eri_bb = transform_spatial_eri_to_mo(
                mf, Cb, Cb, Cb, Cb,
                use_cholesky=use_cholesky,
                eri_factors=eri_factors,
            )
            eri_ab = transform_spatial_eri_to_mo(
                mf, Ca, Ca, Cb, Cb,
                use_cholesky=use_cholesky,
                eri_factors=eri_factors,
            )
            eri_ba = transform_spatial_eri_to_mo(
                mf, Cb, Cb, Ca, Ca,
                use_cholesky=use_cholesky,
                eri_factors=eri_factors,
            )




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


    def run(
        self,
        nstates=1,
        mo_coeff=None,
        method='direct_ci',
        ci0=None,
        use_cholesky=None,
        solvent_response=None,
        solvent_response_eps=1.78,
        spin_root_cushion=None,
        spin_selection_tol=None,
        wfnsym=None,
        target_irrep=None,
    ):
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
            'direct_ci' is the default matrix-free direct-CI backend.
            'ci' is the standard dense CI solver.
            'jw' is the exact diagonalizaion by Jordan-Wigner transformation.
            The default is 'direct_ci'.

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
        solvent_response_model = None
        if solvent_response is not None:
            solvent_response_model = str(solvent_response).lower()
            if solvent_response_model in {"none", "false"}:
                solvent_response_model = None
            elif solvent_response_model not in {"lr", "lr_pcm", "lr-pcm"}:
                raise ValueError("solvent_response must be None or 'lr_pcm'.")
            else:
                solvent_response_model = "lr_pcm"
                method = "ci"

        self.nstates = nstates
        self.use_cholesky_integrals = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        method_key = str(method).lower().replace("-", "_")
        direct_spin0_methods = {"direct_spin0", "direct_spin0_symm", "spin0", "spin0_symm"}

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
        self._property_operator_cache = {}


        if self.binary is None:
            mo_occ = _reference_active_occupations(self.nelecas_spin, ncas)
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary
        else:
            binary = self.binary
        binary = self._filter_binary_by_irrep(binary, target_irrep=target_irrep, wfnsym=wfnsym)
        solve_nstates = self._spin_selected_nstates(
            nstates,
            binary.shape[0],
            spin_root_cushion=spin_root_cushion,
        )

        if solvent_response_model is None and (method_key == 'direct_ci' or method_key in direct_spin0_methods or (
            method_key == 'ci' and self.use_cholesky_integrals and not self.spin_purification
        )):
            from pyqed.qchem.mcscf.direct_ci import CASCI as DirectCASCI

            direct_solver = DirectCASCI(
                self.mf,
                ncas=self.ncas,
                nelecas=self.nelecas,
                ncore=self.ncore,
                ms2=self.ms2,
                multiplicity=self.multiplicity,
                tol=getattr(self, 'tol', 0),
                verbose=self.verbose,
            )
            direct_solver.spin_root_cushion = self.spin_root_cushion
            direct_solver.spin_selection_tol = self.spin_selection_tol
            if hasattr(self, 'direct_spin0_symm_dense_fallback_nconfigs'):
                direct_solver.direct_spin0_symm_dense_fallback_nconfigs = (
                    self.direct_spin0_symm_dense_fallback_nconfigs
                )
            if hasattr(self, 'direct_spin0_native_pair'):
                direct_solver.direct_spin0_native_pair = self.direct_spin0_native_pair
            if hasattr(self, 'direct_spin0_native_davidson'):
                direct_solver.direct_spin0_native_davidson = self.direct_spin0_native_davidson
            if hasattr(self, 'direct_ci_workers'):
                direct_solver.direct_ci_workers = self.direct_ci_workers
            direct_solver.binary = binary
            direct_solver.run(
                nstates=nstates,
                mo_coeff=self.mo_coeff,
                method=method_key,
                ci0=ci0,
                use_cholesky=use_cholesky,
                spin_root_cushion=spin_root_cushion,
                spin_selection_tol=spin_selection_tol,
                wfnsym=wfnsym,
                target_irrep=target_irrep,
            )

            self.mo_coeff = direct_solver.mo_coeff
            self.mo_core = direct_solver.mo_core
            self.mo_cas = direct_solver.mo_cas
            self.binary = direct_solver.binary
            self.e_core = direct_solver.e_core
            self.e_tot = direct_solver.e_tot
            self.ci = direct_solver.ci
            self.hcore = direct_solver.hcore
            self.eri_so = direct_solver.eri_so
            self.h2e_cas = getattr(direct_solver, 'h2e_cas', None)
            self.SC1 = getattr(direct_solver, 'SC1', None)
            self.SC2 = getattr(direct_solver, 'SC2', None)
            self.active_symmetry = getattr(direct_solver, 'active_symmetry', self.active_symmetry)
            self.active_orb_irrep_labels = getattr(
                direct_solver, 'active_orb_irrep_labels', self.active_orb_irrep_labels
            )
            self.active_orb_sym = getattr(direct_solver, 'active_orb_sym', self.active_orb_sym)
            self.active_irrep_counts = getattr(
                direct_solver, 'active_irrep_counts', self.active_irrep_counts
            )
            self.det_irrep_labels = getattr(direct_solver, 'det_irrep_labels', self.det_irrep_labels)
            self.det_irrep_ids = getattr(direct_solver, 'det_irrep_ids', self.det_irrep_ids)
            self.det_irrep_counts = getattr(direct_solver, 'det_irrep_counts', self.det_irrep_counts)
            self.wfnsym = getattr(direct_solver, 'wfnsym', self.wfnsym)
            self.det_irrep_filter_indices = getattr(
                direct_solver, 'det_irrep_filter_indices', self.det_irrep_filter_indices
            )
            self.spin0_symm_transform = getattr(
                direct_solver, 'spin0_symm_transform', self.spin0_symm_transform
            )
            self.spin0_pair_indices = getattr(
                direct_solver, 'spin0_pair_indices', self.spin0_pair_indices
            )
            self.solver_backend = getattr(direct_solver, 'solver_backend', 'direct_ci_factor_conn')
            return self

        # print('Number of determinants', binary.shape[0])

        # effective hamiltonian in the CAS
        h1e, h2e = self.get_SO_matrix(use_cholesky=use_cholesky)

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
        self.lr_pcm_response_matrix = None
        self.lr_pcm_response_eps = None
        self.lr_pcm_raw_e_tot = None
        if solvent_response_model == "lr_pcm":
            raw_E, raw_X = self._lowest_dense_eigensystem(H_CI, max(1, nstates))
            lr_kernel = self._lr_pcm_determinant_kernel(raw_X[:, 0], eps=solvent_response_eps)
            H_CI = H_CI + lr_kernel
            E, X = self._lowest_dense_eigensystem(H_CI, solve_nstates)
            self.lr_pcm_response_matrix = lr_kernel
            self.lr_pcm_response_eps = float(solvent_response_eps)
            self.lr_pcm_raw_e_tot = raw_E[:len(E)] + self.e_core
        else:
            if solve_nstates >= H_CI.shape[0]:
                E, X = self._lowest_dense_eigensystem(H_CI, solve_nstates)
            else:
                E, X = eigsh(H_CI, k=solve_nstates, which='SA')

        E, X = self._apply_multiplicity_selection(
            E,
            X,
            nstates,
            spin_selection_tol=spin_selection_tol,
        )


        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(nstates)]

        if self.verbose >= 1:
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

    def make_rdm1(
        self,
        state_id,
        with_core=False,
        with_vir=False,
        representation='mo',
        repr=None,
    ):
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
        if repr is not None:
            representation = repr
        representation = str(representation).lower()
        if representation not in ("mo", "ao"):
            raise ValueError("representation must be 'mo' or 'ao'.")

        if with_core and not with_vir:

            norb = ncas + ncore
            D = np.zeros((norb, norb), dtype=float)

            if ncore > 0: 
                for i in range(ncore): 
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            if representation == "ao":
                C = np.asarray(self.mf.mo_coeff)[:, :norb]
                return C @ D @ C.conj().T
            return D

        if with_core and with_vir:

            D = np.zeros((nmo, nmo), dtype=float)
            if ncore > 0: 
                for i in range(ncore): 
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            if representation == "ao":
                C = np.asarray(self.mf.mo_coeff)
                return C @ D @ C.conj().T
            return D
        else:
            D = make_rdm1(ci, self.binary, self.SC1)
            if representation == "ao":
                C = np.asarray(self.mf.mo_coeff)[:, ncore:ncore+ncas]
                return C @ D @ C.conj().T
            return D


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
            state_overlap = np.vdot(self.ci[bra_id], self.ci[ket_id])
            c_core = (
                np.trace(h1a[:ncore, :ncore])
                + np.trace(h1b[:ncore, :ncore])
            ) * state_overlap
        else:
            c_core = 0

        h1e_cas = (
            h1a[ncore:ncas+ncore, ncore:ncas+ncore],
            h1b[ncore:ncas+ncore, ncore:ncas+ncore],
        )
        c_cas = contract_with_tdm1(self.ci[bra_id], self.ci[ket_id], self.binary, self.SC1, h1e_cas)

        return c_cas + c_core

    def _electric_dipole_ao(self, center=None):
        if hasattr(self.mf, "dipole"):
            op = self.mf.dipole(center=center, basis="ao")
        else:
            if center is None:
                center = self.mol.center_of_mass()
            op = -np.asarray(
                self.mol.moment_integral(center=np.asarray(center, dtype=float)),
                dtype=float,
            )
        op = np.asarray(op)
        if op.ndim != 3:
            raise ValueError("Dipole operator must be a rank-3 array.")
        if op.shape[0] != 3:
            if op.shape[-1] == 3:
                op = np.moveaxis(op, -1, 0)
            else:
                raise ValueError("Dipole operator must have shape (3, nao, nao) or (nao, nao, 3).")
        return op

    def _electric_dipole_mo(self, center=None):
        if center is None:
            center_key = None
        else:
            center_key = tuple(np.asarray(center, dtype=float).ravel())
        cache_key = ("electric_dipole_mo", center_key, id(self.mo_coeff))
        cache = getattr(self, "_property_operator_cache", None)
        if cache is None:
            cache = self._property_operator_cache = {}
        if cache_key in cache:
            return cache[cache_key]

        dipole_ao = self._electric_dipole_ao(center=center)
        transformed = [
            _transform_1e_operator_ao_to_mo(dipole_ao[xyz], self.mo_coeff)
            for xyz in range(3)
        ]
        if _is_uhf_reference(self.mo_coeff):
            dipole_mo = (
                np.asarray([op[0] for op in transformed]),
                np.asarray([op[1] for op in transformed]),
            )
        else:
            dipole_mo = np.asarray(transformed)
        cache[cache_key] = dipole_mo
        return dipole_mo

    def transition_dipole_moment(self, bra_id=None, ket_id=0, center=None, state_ids=None):
        """
        Electronic transition dipole moments between CASCI roots.

        The operator is the electronic dipole ``mu = -r``. If ``bra_id`` is
        omitted, moments from ``ket_id`` to all other computed roots are
        returned with shape ``(nroots - 1, 3)``. Supplying ``bra_id`` returns a
        single ``(3,)`` vector.
        """
        if self.ci is None:
            raise ValueError("Run CASCI before requesting transition dipoles.")
        if bra_id is not None and state_ids is not None:
            raise ValueError("Specify either bra_id or state_ids, not both.")

        dipole_mo = self._electric_dipole_mo(center=center)
        if isinstance(dipole_mo, tuple):
            dipole_a, dipole_b = dipole_mo
        else:
            dipole_a = dipole_mo
            dipole_b = dipole_a

        ncore = int(self.ncore)
        ncas = int(self.ncas)
        active = slice(ncore, ncore + ncas)

        def contract_one(bra):
            bra = int(bra)
            ket = int(ket_id)
            tdm1a, tdm1b = make_tdm1s(
                self.ci[bra],
                self.ci[ket],
                self.binary,
                self.SC1,
            )
            value = (
                np.einsum("xpq,pq->x", dipole_a[:, active, active], tdm1a, optimize=True)
                + np.einsum("xpq,pq->x", dipole_b[:, active, active], tdm1b, optimize=True)
            )
            if ncore > 0:
                core_trace = (
                    np.trace(dipole_a[:, :ncore, :ncore], axis1=1, axis2=2)
                    + np.trace(dipole_b[:, :ncore, :ncore], axis1=1, axis2=2)
                )
                value = value + core_trace * np.vdot(self.ci[bra], self.ci[ket])
            return value

        if bra_id is not None:
            return contract_one(bra_id)

        if state_ids is None:
            state_ids = [idx for idx in range(len(self.ci)) if idx != int(ket_id)]
        return np.asarray([contract_one(idx) for idx in state_ids])

    def transition_dipole(self, *args, **kwargs):
        """Alias for :meth:`transition_dipole_moment`."""
        return self.transition_dipole_moment(*args, **kwargs)

    def make_tdm1(
        self,
        bra_id,
        ket_id=0,
        representation='mo',
        repr=None,
        with_core=False,
        with_vir=False,
    ):
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
        if repr is not None:
            representation = repr
        representation = str(representation).lower()
        if representation not in ("mo", "ao"):
            raise ValueError("representation must be 'mo' or 'ao'.")

        if bra_id == ket_id:
            return self.make_rdm1(
                bra_id,
                with_core=with_core,
                with_vir=with_vir,
                representation=representation,
            )

        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]
        D_active = make_tdm1(cibra, ciket, self.binary, self.SC1)

        if not with_core and not with_vir:
            if representation == "ao":
                C = np.asarray(self.mf.mo_coeff)[:, self.ncore:self.ncore + self.ncas]
                return C @ D_active @ C.conj().T
            return D_active

        ncore = self.ncore
        ncas = self.ncas
        nmo = self.mf.nmo if with_vir else ncore + ncas
        D = np.zeros((nmo, nmo), dtype=np.result_type(D_active, complex))
        D[ncore:ncore + ncas, ncore:ncore + ncas] = D_active

        if representation == "ao":
            C = np.asarray(self.mf.mo_coeff)[:, :nmo]
            return C @ D @ C.conj().T
        return D

    def vibronic_couplings(
        self,
        state_ids=None,
        modes=None,
        return_terms=False,
    ):
        """
        First- and second-order electronic Hamiltonian derivatives.

        Parameters
        ----------
        state_ids : sequence of int, optional
            Electronic states to include. If omitted, all available CASCI roots
            are used.
        modes : ndarray, optional
            Normal-mode or other collective-coordinate vectors with shape
            ``(nmodes, natom, 3)``, ``(nmodes, 3*natom)``, or
            ``(3*natom, nmodes)``.
        return_terms : bool, optional
            If true, also return the underlying ``BOHamiltonianDerivatives``
            object.

        Returns
        -------
        F, G : ndarray
            If ``modes`` is provided, the shapes are
            ``(nstates, nstates, nmodes)`` and
            ``(nstates, nstates, nmodes, nmodes)``.  Otherwise, Cartesian
            derivatives are returned with shapes ``(nstates, nstates, natom, 3)``
            and ``(nstates, nstates, natom, 3, natom, 3)``.
        """
        from pyqed.qchem.geometric import bo_hamiltonian_derivatives

        terms = bo_hamiltonian_derivatives(
            self,
            state_ids=state_ids,
            mode_vectors=modes,
        )
        if modes is not None:
            f = np.moveaxis(terms.F_projected, 0, -1)
            g = np.moveaxis(terms.G_projected, (0, 1), (-2, -1))
        else:
            natom = self.mol.natom
            f_cart = terms.F_cartesian.reshape(
                natom,
                3,
                *terms.F_cartesian.shape[1:],
            )
            f = np.moveaxis(f_cart, (0, 1), (-2, -1))
            g_cart = terms.G_cartesian.reshape(
                natom,
                3,
                natom,
                3,
                *terms.G_cartesian.shape[2:],
            )
            g = np.moveaxis(g_cart, (0, 1, 2, 3), (-4, -3, -2, -1))
        if return_terms:
            return f, g, terms
        return f, g

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
                           light_speed=None, order='grouped',
                           soc_model='1e', dm=None, states=None):
        """
        SOC matrix element between CASCI states.

        If ``hso`` is not provided, the active-space SOC operator is built from
        the current active orbitals.
        """
        if other is None:
            other = self
        _validate_matching_active_spaces(self, other)
        if hso is None:
            model = soc_model.lower()
            if model == '1e':
                hso = get_soc_1e_spin_orbital(
                    self.mf,
                    representation='mo',
                    mo_coeff=self.mo_cas,
                    one_center=one_center,
                    with_prefactor=with_prefactor,
                    light_speed=light_speed,
                    order=order,
                )
            elif model == 'somf':
                hso = get_soc_somf_spin_orbital(
                    self.mf,
                    representation='mo',
                    mo_coeff=self.mo_cas,
                    dm=dm,
                    states=states,
                    one_center=one_center,
                    with_prefactor=with_prefactor,
                    light_speed=light_speed,
                    order=order,
                )
            else:
                raise ValueError("soc_model must be '1e' or 'somf'.")
        return self.contract_with_tdm1_spin_orbital(
            bra_id,
            ket_id=ket_id,
            h1e=hso,
            other=other,
            order=order,
        )

    def make_tdm2(self, bra_id, ket_id=0):
        """
        Spin-traced two-particle transition density matrix in MO basis.

        .. math::

            \Gamma_{pqrs}^{\beta \alpha}
            = \sum_{\sigma\tau}<\Psi_\beta|
              p^\dagger_\sigma r^\dagger_\tau s_\tau q_\sigma
              |\Psi_\alpha>
        """
        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]
        return make_tdm2(cibra, ciket, self.binary, self.SC1, self.SC2)


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
    return _make_tdm1s_link_contractions(ci, ci, binary)


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
    return _make_tdm1s_link_contractions(cibra, ciket, binary)

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

def _annihilate_bit(bits, idx):
    if not ((bits >> idx) & 1):
        return None, 0
    phase = -1 if (bits & ((1 << idx) - 1)).bit_count() % 2 else 1
    return bits ^ (1 << idx), phase


def _create_bit(bits, idx):
    if (bits >> idx) & 1:
        return None, 0
    phase = -1 if (bits & ((1 << idx) - 1)).bit_count() % 2 else 1
    return bits | (1 << idx), phase


def _determinant_bits_from_binary(binary):
    """Encode PyQED alpha/beta occupation strings as ordered spin-orbitals."""
    bits = []
    _, _, nmo = binary.shape
    for occ in binary:
        det_bits = 0
        for spin in range(2):
            for orb in range(nmo):
                if occ[spin, orb]:
                    det_bits |= 1 << (spin * nmo + orb)
        bits.append(det_bits)
    return bits


def _spin_string_bits(strings):
    bits = []
    for occ in strings:
        det_bits = 0
        for orb, occupied in enumerate(occ):
            if occupied:
                det_bits |= 1 << orb
        bits.append(det_bits)
    return bits


def _unique_spin_strings_from_binary(binary, spin):
    strings = []
    seen = set()
    for occ in binary[:, spin, :]:
        key = tuple(int(x) for x in occ)
        if key not in seen:
            seen.add(key)
            strings.append(key)
    return np.asarray(strings, dtype=np.int8)


@lru_cache(maxsize=16)
def _cached_spin_string_basis(shape, data):
    binary = np.frombuffer(data, dtype=np.int8).reshape(shape)
    alpha = _unique_spin_strings_from_binary(binary, 0)
    beta = _unique_spin_strings_from_binary(binary, 1)
    alpha_index = {tuple(row): idx for idx, row in enumerate(alpha)}
    beta_index = {tuple(row): idx for idx, row in enumerate(beta)}
    alpha_det = np.empty(shape[0], dtype=np.int64)
    beta_det = np.empty(shape[0], dtype=np.int64)

    for det_id, occ in enumerate(binary):
        alpha_det[det_id] = alpha_index[tuple(int(x) for x in occ[0])]
        beta_det[det_id] = beta_index[tuple(int(x) for x in occ[1])]

    return alpha, beta, alpha_det, beta_det


def _ci_to_spin_string_matrix(ci, binary):
    alpha, beta, alpha_det, beta_det = _cached_spin_string_basis(
        tuple(binary.shape),
        np.ascontiguousarray(binary, dtype=np.int8).tobytes(),
    )
    coeff = np.zeros((len(alpha), len(beta)), dtype=np.asarray(ci).dtype)
    coeff[alpha_det, beta_det] = ci

    return alpha, beta, coeff


@lru_cache(maxsize=16)
def _cached_spin_string_ops(bit_tuple, nmo):
    bits = list(bit_tuple)
    nstr = len(bits)
    bit_index = {bits_i: idx for idx, bits_i in enumerate(bits)}

    one = np.zeros((nmo, nmo, nstr, nstr))
    two = np.zeros((nmo, nmo, nmo, nmo, nstr, nstr))

    for ket, bits0 in enumerate(bits):
        for q in range(nmo):
            bits1, phase1 = _annihilate_bit(bits0, q)
            if phase1 == 0:
                continue
            for p in range(nmo):
                bits2, phase2 = _create_bit(bits1, p)
                if phase2 == 0:
                    continue
                bra = bit_index.get(bits2)
                if bra is not None:
                    one[p, q, bra, ket] += phase1 * phase2

            for s in range(nmo):
                bits2, phase2 = _annihilate_bit(bits1, s)
                if phase2 == 0:
                    continue
                phase12 = phase1 * phase2
                for r in range(nmo):
                    bits3, phase3 = _create_bit(bits2, r)
                    if phase3 == 0:
                        continue
                    phase123 = phase12 * phase3
                    for p in range(nmo):
                        bits4, phase4 = _create_bit(bits3, p)
                        if phase4 == 0:
                            continue
                        bra = bit_index.get(bits4)
                        if bra is not None:
                            two[p, q, r, s, bra, ket] += phase123 * phase4

    return one, two


def _spin_string_ops(strings):
    _, nmo = strings.shape
    bits = tuple(_spin_string_bits(strings))
    return _cached_spin_string_ops(bits, nmo)


@lru_cache(maxsize=16)
def _cached_spin_string_links(bit_tuple, nmo):
    bits = list(bit_tuple)
    bit_index = {bits_i: idx for idx, bits_i in enumerate(bits)}
    one_links = []
    two_links = []

    for ket, bits0 in enumerate(bits):
        for q in range(nmo):
            bits1, phase1 = _annihilate_bit(bits0, q)
            if phase1 == 0:
                continue
            for p in range(nmo):
                bits2, phase2 = _create_bit(bits1, p)
                if phase2 == 0:
                    continue
                bra = bit_index.get(bits2)
                if bra is not None:
                    one_links.append((p, q, bra, ket, phase1 * phase2))

            for s in range(nmo):
                bits2, phase2 = _annihilate_bit(bits1, s)
                if phase2 == 0:
                    continue
                phase12 = phase1 * phase2
                for r in range(nmo):
                    bits3, phase3 = _create_bit(bits2, r)
                    if phase3 == 0:
                        continue
                    phase123 = phase12 * phase3
                    for p in range(nmo):
                        bits4, phase4 = _create_bit(bits3, p)
                        if phase4 == 0:
                            continue
                        bra = bit_index.get(bits4)
                        if bra is not None:
                            two_links.append((p, q, r, s, bra, ket, phase123 * phase4))

    if one_links:
        one = tuple(np.asarray(col, dtype=np.int64) for col in zip(*one_links))
    else:
        one = tuple(np.asarray([], dtype=np.int64) for _ in range(5))
    if two_links:
        two = tuple(np.asarray(col, dtype=np.int64) for col in zip(*two_links))
    else:
        two = tuple(np.asarray([], dtype=np.int64) for _ in range(7))
    return one, two


def _spin_string_links(strings):
    _, nmo = strings.shape
    bits = tuple(_spin_string_bits(strings))
    return _cached_spin_string_links(bits, nmo)


@njit
def _scatter_same_spin_rdm2_numba(dm2, p, q, r, s, bra, ket, phase, overlap):
    for link in range(p.shape[0]):
        dm2[p[link], q[link], r[link], s[link]] += (
            phase[link] * overlap[bra[link], ket[link]]
        )


@njit
def _scatter_spin_rdm1_numba(dm1, p, q, bra, ket, phase, overlap):
    for link in range(p.shape[0]):
        dm1[p[link], q[link]] += phase[link] * overlap[bra[link], ket[link]]


@njit
def _scatter_opposite_spin_rdm2_numba(
    dm2,
    pa,
    qa,
    bra_a,
    ket_a,
    phase_a,
    rb,
    sb,
    bra_b,
    ket_b,
    phase_b,
    cbra,
    cket,
):
    for la in range(pa.shape[0]):
        for lb in range(rb.shape[0]):
            dm2[pa[la], qa[la], rb[lb], sb[lb]] += (
                phase_a[la]
                * phase_b[lb]
                * cbra[bra_a[la], bra_b[lb]]
                * cket[ket_a[la], ket_b[lb]]
            )


def _scatter_same_spin_rdm2(dm2, links, overlap):
    p, q, r, s, bra, ket, phase = links
    if len(p) == 0:
        return
    _scatter_same_spin_rdm2_numba(dm2, p, q, r, s, bra, ket, phase, overlap)


def _scatter_spin_rdm1(dm1, links, overlap):
    p, q, bra, ket, phase = links
    if len(p) == 0:
        return
    _scatter_spin_rdm1_numba(dm1, p, q, bra, ket, phase, overlap)


def _scatter_opposite_spin_rdm2(dm2, alpha_links, beta_links, cbra, cket):
    pa, qa, bra_a, ket_a, phase_a = alpha_links
    rb, sb, bra_b, ket_b, phase_b = beta_links
    if len(pa) == 0 or len(rb) == 0:
        return
    scatter_cpp = _cpp_attr("scatter_opposite_spin_rdm2")
    if scatter_cpp is not None and not (np.iscomplexobj(dm2) or np.iscomplexobj(cbra) or np.iscomplexobj(cket)):
        try:
            scatter_cpp(
                dm2,
                np.ascontiguousarray(pa, dtype=np.intp),
                np.ascontiguousarray(qa, dtype=np.intp),
                np.ascontiguousarray(bra_a, dtype=np.intp),
                np.ascontiguousarray(ket_a, dtype=np.intp),
                np.ascontiguousarray(phase_a, dtype=np.intp),
                np.ascontiguousarray(rb, dtype=np.intp),
                np.ascontiguousarray(sb, dtype=np.intp),
                np.ascontiguousarray(bra_b, dtype=np.intp),
                np.ascontiguousarray(ket_b, dtype=np.intp),
                np.ascontiguousarray(phase_b, dtype=np.intp),
                np.ascontiguousarray(cbra, dtype=np.float64),
                np.ascontiguousarray(cket, dtype=np.float64),
            )
            return
        except Exception:
            pass
    _scatter_opposite_spin_rdm2_numba(
        dm2,
        pa,
        qa,
        bra_a,
        ket_a,
        phase_a,
        rb,
        sb,
        bra_b,
        ket_b,
        phase_b,
        cbra,
        cket,
    )


def _make_tdm1s_link_contractions(cibra, ciket, binary):
    if cibra is ciket:
        alpha_bra, beta_bra, cket = _ci_to_spin_string_matrix(ciket, binary)
        alpha_ket, beta_ket, cbra = alpha_bra, beta_bra, cket
    else:
        alpha_bra, beta_bra, cbra = _ci_to_spin_string_matrix(cibra, binary)
        alpha_ket, beta_ket, cket = _ci_to_spin_string_matrix(ciket, binary)

    if not np.array_equal(alpha_bra, alpha_ket) or not np.array_equal(beta_bra, beta_ket):
        raise ValueError("Bra and ket CI vectors must use the same determinant basis.")

    alpha_one, _ = _spin_string_links(alpha_bra)
    beta_one, _ = _spin_string_links(beta_bra)
    cbra = cbra.conj()
    dtype = np.result_type(cbra, cket, float)
    nmo = binary.shape[2]
    dm1a = np.zeros((nmo, nmo), dtype=dtype)
    dm1b = np.zeros((nmo, nmo), dtype=dtype)

    _scatter_spin_rdm1(dm1a, alpha_one, cbra @ cket.T)
    _scatter_spin_rdm1(dm1b, beta_one, cbra.T @ cket)

    return dm1a, dm1b


def _make_tdm2_link_contractions(cibra, ciket, binary):
    if cibra is ciket:
        alpha_bra, beta_bra, cket = _ci_to_spin_string_matrix(ciket, binary)
        alpha_ket, beta_ket, cbra = alpha_bra, beta_bra, cket
    else:
        alpha_bra, beta_bra, cbra = _ci_to_spin_string_matrix(cibra, binary)
        alpha_ket, beta_ket, cket = _ci_to_spin_string_matrix(ciket, binary)

    if not np.array_equal(alpha_bra, alpha_ket) or not np.array_equal(beta_bra, beta_ket):
        raise ValueError("Bra and ket CI vectors must use the same determinant basis.")

    alpha_one, alpha_two = _spin_string_links(alpha_bra)
    beta_one, beta_two = _spin_string_links(beta_bra)
    cbra = cbra.conj()
    dtype = np.result_type(cbra, cket, float)
    dm2 = np.zeros((binary.shape[2],) * 4, dtype=dtype)

    _scatter_same_spin_rdm2(dm2, alpha_two, cbra @ cket.T)
    _scatter_same_spin_rdm2(dm2, beta_two, cbra.T @ cket)

    _scatter_opposite_spin_rdm2(dm2, alpha_one, beta_one, cbra, cket)
    _scatter_opposite_spin_rdm2(dm2, beta_one, alpha_one, cbra.T, cket.T)

    return dm2


def _make_tdm2_string_contractions(cibra, ciket, binary):
    if cibra is ciket:
        alpha_bra, beta_bra, cket = _ci_to_spin_string_matrix(ciket, binary)
        alpha_ket, beta_ket, cbra = alpha_bra, beta_bra, cket
    else:
        alpha_bra, beta_bra, cbra = _ci_to_spin_string_matrix(cibra, binary)
        alpha_ket, beta_ket, cket = _ci_to_spin_string_matrix(ciket, binary)

    if not np.array_equal(alpha_bra, alpha_ket) or not np.array_equal(beta_bra, beta_ket):
        raise ValueError("Bra and ket CI vectors must use the same determinant basis.")

    alpha_one, alpha_two = _spin_string_ops(alpha_bra)
    beta_one, beta_two = _spin_string_ops(beta_bra)
    cbra = cbra.conj()
    dtype = np.result_type(cbra, cket, float)
    dm2 = np.zeros((binary.shape[2],) * 4, dtype=dtype)

    # Same-spin blocks: spectator spin is contracted out first.
    alpha_overlap = np.einsum('ai,bi->ab', cbra, cket, optimize=True)
    beta_overlap = np.einsum('ai,aj->ij', cbra, cket, optimize=True)
    dm2 += np.einsum('pqrsab,ab->pqrs', alpha_two, alpha_overlap, optimize=True)
    dm2 += np.einsum('pqrsij,ij->pqrs', beta_two, beta_overlap, optimize=True)

    # Opposite-spin blocks factor into alpha and beta one-body string operators.
    alpha_projected = np.einsum('ai,pqab,bj->pqij', cbra, alpha_one, cket, optimize=True)
    dm2 += np.einsum('pqij,rsij->pqrs', alpha_projected, beta_one, optimize=True)

    beta_projected = np.einsum('ai,pqij,bj->pqab', cbra, beta_one, cket, optimize=True)
    dm2 += np.einsum('pqab,rsab->pqrs', beta_projected, alpha_one, optimize=True)

    return dm2


def _make_tdm2_explicit(cibra, ciket, binary):
    cibra = np.asarray(cibra)
    ciket = np.asarray(ciket)
    _, _, nmo = binary.shape
    det_bits = _determinant_bits_from_binary(binary)
    det_index = {bits: idx for idx, bits in enumerate(det_bits)}
    dtype = np.result_type(cibra, ciket, float)
    dm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=dtype)

    for ket, bits0 in enumerate(det_bits):
        ket_coeff = ciket[ket]
        if ket_coeff == 0:
            continue

        for sigma in range(2):
            spin_offset_sigma = sigma * nmo
            for tau in range(2):
                spin_offset_tau = tau * nmo

                for q in range(nmo):
                    bits1, phase1 = _annihilate_bit(bits0, spin_offset_sigma + q)
                    if phase1 == 0:
                        continue
                    for s in range(nmo):
                        bits2, phase2 = _annihilate_bit(bits1, spin_offset_tau + s)
                        if phase2 == 0:
                            continue
                        phase12 = phase1 * phase2

                        for r in range(nmo):
                            bits3, phase3 = _create_bit(bits2, spin_offset_tau + r)
                            if phase3 == 0:
                                continue
                            phase123 = phase12 * phase3
                            for p in range(nmo):
                                bits4, phase4 = _create_bit(bits3, spin_offset_sigma + p)
                                if phase4 == 0:
                                    continue
                                bra = det_index.get(bits4)
                                if bra is None:
                                    continue
                                dm2[p, q, r, s] += (
                                    cibra[bra].conj()
                                    * ket_coeff
                                    * phase123
                                    * phase4
                                )

    return dm2


def make_rdm2(ci, Binary, SC1, SC2):
    """
    build the spin-traced 2-particle operator with the 2e RDM

    .. math::

        \Gamma_{pqrs} = \sum_{\sigma, \tau} p^\dagger_\sigma r^\dagger_\tau s_\tau q_\sigma

    Params
    ------
    Binary: binary string (I, s, p)
        I: configuration index, S: spin index, p: MO index

    Refs
    ----
    J. Chem. Theory Comput. 2022, 18, 6690−6699

    """
    return _make_tdm2_link_contractions(ci, ci, Binary)


def make_tdm2(cibra, ciket, Binary, SC1, SC2):
    """
    Build the spin-traced two-particle transition density matrix.

    The convention matches ``make_rdm2``:

    ``Gamma[p,q,r,s] = sum_{sigma,tau} <bra| p^+_sigma r^+_tau s_tau q_sigma |ket>``.
    """
    return _make_tdm2_link_contractions(cibra, ciket, Binary)


def _compute_ci_mo_overlap(cibra, ciket, s=None):
    if s is not None:
        return s

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
    return reduce(np.dot, (cibra.mf.mo_coeff.T, s, ciket.mf.mo_coeff))


def _as_state_ci_matrix(ci, ndet):
    ci_arr = np.asarray(ci)
    if ci_arr.ndim == 1:
        if ci_arr.shape[0] != ndet:
            raise ValueError(f"CI vector length {ci_arr.shape[0]} does not match ndet={ndet}.")
        return ci_arr.reshape(1, ndet)
    if ci_arr.ndim == 2 and ci_arr.shape[1] == ndet:
        return ci_arr
    raise ValueError(f"Unsupported CI coefficient shape {ci_arr.shape}; expected (*, {ndet}).")


def _unique_rows_first(rows):
    rows = np.asarray(rows, dtype=np.int8)
    if rows.shape[0] == 0:
        return rows
    _, first_idx = np.unique(rows, axis=0, return_index=True)
    return rows[np.sort(first_idx)]


def _occupation_lists(strings):
    return [np.flatnonzero(row) for row in strings]


def _string_overlap_matrix(saa_eff, bra_occ, ket_occ, dtype):
    out = np.empty((len(bra_occ), len(ket_occ)), dtype=dtype)
    for i, occ_i in enumerate(bra_occ):
        for j, occ_j in enumerate(ket_occ):
            out[i, j] = np.linalg.det(saa_eff[np.ix_(occ_i, occ_j)])
    return out


def _string_transform_matrix(orbital_transform, occ_strings, dtype):
    """Induced determinant-space transform for an active-orbital rotation."""
    occ_lists = _occupation_lists(occ_strings)
    nstr = len(occ_lists)
    out = np.empty((nstr, nstr), dtype=dtype)
    for i, occ_i in enumerate(occ_lists):
        for j, occ_j in enumerate(occ_lists):
            out[i, j] = np.linalg.det(orbital_transform[np.ix_(occ_i, occ_j)])
    return out


def _string_singular_weights(sigma, occ_strings):
    """Diagonal singular-value weights in the determinant/string representation."""
    occ_lists = _occupation_lists(occ_strings)
    if len(occ_lists) == 0:
        return np.empty((0,), dtype=sigma.dtype)
    out = np.empty((len(occ_lists),), dtype=sigma.dtype)
    for i, occ in enumerate(occ_lists):
        out[i] = np.prod(sigma[occ]) if len(occ) > 0 else 1.0
    return out


def _reconstruct_string_overlap_from_svd(u, sigma, right_vh, occ_strings, dtype):
    """Exact determinant-space reconstruction from the orbital-space SVD.

    For a fixed-electron string basis built from all combinations of occupied
    orbitals in a given one-particle space, Cauchy-Binet gives

    ``W(S) = W(U) @ diag(w(sigma)) @ W(Vh)``

    where ``S = U diag(sigma) Vh`` and ``W`` denotes the induced transform in
    determinant/string space. The right factor must be passed in ``Vh``
    orientation; using ``V`` instead gives the wrong determinant-space map.
    """
    left = _string_transform_matrix(u, occ_strings, dtype)
    right = _string_transform_matrix(right_vh, occ_strings, dtype)
    weights = np.diag(_string_singular_weights(sigma, occ_strings))
    return left @ weights @ right


def _biorthogonalize_active_overlap(saa_eff):
    """Balanced SVD biorthogonalization of the active-space overlap block."""
    u, sigma, vh = np.linalg.svd(saa_eff, full_matrices=False)
    if sigma.size == 0:
        return u, sigma, vh, saa_eff.copy(), saa_eff.copy()
    tol = np.finfo(sigma.dtype).eps * max(saa_eff.shape) * sigma.max()
    if np.min(sigma) <= tol:
        raise np.linalg.LinAlgError(
            "Active-space overlap is numerically singular and cannot be biorthogonalized "
            f"(min sigma={sigma.min():.3e}, tol={tol:.3e})."
        )
    sigma_inv_sqrt = sigma ** -0.5
    x_left = u * sigma_inv_sqrt[np.newaxis, :]
    x_right = vh.conj().T * sigma_inv_sqrt[np.newaxis, :]
    return u, sigma, vh, x_left, x_right


@dataclass
class _BiorthogonalOverlapPrep:
    s_mo: np.ndarray
    core_factor: complex
    saa_eff: np.ndarray
    scc: np.ndarray
    scc_u: np.ndarray
    scc_sigma: np.ndarray
    scc_vh: np.ndarray
    saa_u: np.ndarray
    saa_sigma: np.ndarray
    saa_vh: np.ndarray
    x_left: np.ndarray
    x_right: np.ndarray


def _svd_inverse(matrix, *, tol=None):
    """SVD-based inverse used as the first exact biorthogonalization step."""
    u, sigma, vh = np.linalg.svd(matrix, full_matrices=False)
    if sigma.size == 0:
        return u, sigma, vh, matrix.copy()
    if tol is None:
        tol = np.finfo(sigma.dtype).eps * max(matrix.shape) * sigma.max()
    if np.min(sigma) <= tol:
        raise np.linalg.LinAlgError(
            "Overlap block is numerically singular and cannot be biorthogonalized "
            f"(min sigma={sigma.min():.3e}, tol={tol:.3e})."
        )
    inv = (vh.conj().T / sigma) @ u.conj().T
    return u, sigma, vh, inv


def _active_orbital_slices(ncore_bra, ncore_ket, ncas_bra, ncas_ket):
    bra_active = slice(ncore_bra, ncore_bra + ncas_bra)
    ket_active = slice(ncore_ket, ncore_ket + ncas_ket)
    return bra_active, ket_active


def _prepare_biorthogonal_overlap(s, ncore_bra, ncore_ket, ncas_bra, ncas_ket, dtype):
    """Prepare exact overlap data with SVD-based core biorthogonalization."""
    if ncore_bra != ncore_ket:
        raise ValueError(
            "Different numbers of core orbitals are not supported in overlap: "
            f"{ncore_bra} != {ncore_ket}."
        )

    bra_active, ket_active = _active_orbital_slices(ncore_bra, ncore_ket, ncas_bra, ncas_ket)
    scc = np.asarray(s[:ncore_bra, :ncore_ket], dtype=dtype)
    sca = np.asarray(s[:ncore_bra, ket_active], dtype=dtype)
    sac = np.asarray(s[bra_active, :ncore_ket], dtype=dtype)
    saa = np.asarray(s[bra_active, ket_active], dtype=dtype)

    if ncore_bra == 0:
        saa_u, saa_sigma, saa_vh, x_left, x_right = _biorthogonalize_active_overlap(saa)
        return _BiorthogonalOverlapPrep(
            s_mo=np.asarray(s, dtype=dtype),
            core_factor=dtype.type(1),
            saa_eff=saa,
            scc=scc,
            scc_u=np.empty((0, 0), dtype=dtype),
            scc_sigma=np.empty((0,), dtype=float),
            scc_vh=np.empty((0, 0), dtype=dtype),
            saa_u=saa_u,
            saa_sigma=saa_sigma,
            saa_vh=saa_vh,
            x_left=x_left,
            x_right=x_right,
        )

    scc_u, scc_sigma, scc_vh, scc_inv = _svd_inverse(scc)
    saa_eff = saa - sac @ scc_inv @ sca
    saa_u, saa_sigma, saa_vh, x_left, x_right = _biorthogonalize_active_overlap(saa_eff)

    return _BiorthogonalOverlapPrep(
        s_mo=np.asarray(s, dtype=dtype),
        core_factor=np.linalg.det(scc) ** 2,
        saa_eff=saa_eff,
        scc=scc,
        scc_u=scc_u,
        scc_sigma=scc_sigma,
        scc_vh=scc_vh,
        saa_u=saa_u,
        saa_sigma=saa_sigma,
        saa_vh=saa_vh,
        x_left=x_left,
        x_right=x_right,
    )


def _effective_active_overlap(
    s,
    ncore_bra,
    ncore_ket,
    ncas_bra,
    ncas_ket,
    dtype,
):
    prep = _prepare_biorthogonal_overlap(s, ncore_bra, ncore_ket, ncas_bra, ncas_ket, dtype)
    return prep.core_factor, prep.saa_eff


def _overlap_slow_from_mo_overlap(
    cibra,
    ciket,
    s,
):
    s = _compute_ci_mo_overlap(cibra, ciket, s=s)
    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    dtype = np.result_type(s, np.asarray(cibra.ci), np.asarray(ciket.ci))
    S = np.zeros((nsd_bra, nsd_ket), dtype=dtype)

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore
    core_factor, saa_eff = _effective_active_overlap(
        s,
        ncore_bra,
        ncore_ket,
        cibra.ncas,
        ciket.ncas,
        dtype,
    )

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
            S[I, J] = (
                core_factor
                * np.linalg.det(saa_eff[np.ix_(occidx1_a, occidx2_a)])
                * np.linalg.det(saa_eff[np.ix_(occidx1_b, occidx2_b)])
            )

    return contract(
        'BI, IJ, AJ -> BA',
        _as_state_ci_matrix(cibra.ci, nsd_bra).conj(),
        S,
        _as_state_ci_matrix(ciket.ci, nsd_ket),
    )


def _factorized_ci_overlap(
    cibra,
    ciket,
    s=None,
):
    s = _compute_ci_mo_overlap(cibra, ciket, s=s)

    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    dtype = np.result_type(s, np.asarray(cibra.ci), np.asarray(ciket.ci))

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore
    prep = _prepare_biorthogonal_overlap(
        s,
        ncore_bra,
        ncore_ket,
        cibra.ncas,
        ciket.ncas,
        dtype,
    )

    bra_alpha = _unique_rows_first(cibra.binary[:, 0, :])
    bra_beta = _unique_rows_first(cibra.binary[:, 1, :])
    ket_alpha = _unique_rows_first(ciket.binary[:, 0, :])
    ket_beta = _unique_rows_first(ciket.binary[:, 1, :])

    nalpha_bra, nbeta_bra = len(bra_alpha), len(bra_beta)
    nalpha_ket, nbeta_ket = len(ket_alpha), len(ket_beta)

    if nalpha_bra * nbeta_bra != nsd_bra or nalpha_ket * nbeta_ket != nsd_ket:
        return _overlap_slow_from_mo_overlap(
            cibra,
            ciket,
            s,
        )

    if not np.array_equal(bra_alpha, ket_alpha) or not np.array_equal(bra_beta, ket_beta):
        return _overlap_slow_from_mo_overlap(
            cibra,
            ciket,
            s,
        )

    try:
        return _biorthogonal_ci_overlap_from_prep(cibra, ciket, prep, dtype)
    except (np.linalg.LinAlgError, ValueError):
        ci_bra = _as_state_ci_matrix(cibra.ci, nsd_bra).reshape((-1, nalpha_bra, nbeta_bra))
        ci_ket = _as_state_ci_matrix(ciket.ci, nsd_ket).reshape((-1, nalpha_ket, nbeta_ket))

        overlap_alpha = _string_overlap_matrix(
            prep.saa_eff, _occupation_lists(bra_alpha), _occupation_lists(ket_alpha), dtype
        )
        overlap_beta = _string_overlap_matrix(
            prep.saa_eff, _occupation_lists(bra_beta), _occupation_lists(ket_beta), dtype
        )

        return prep.core_factor * contract(
            'Xab,ac,bd,Ycd->XY',
            ci_bra.conj(),
            overlap_alpha,
            overlap_beta,
            ci_ket,
        )


def _transform_ci_tensors_to_biorthogonal_basis(ci_tensors, alpha_transform, beta_transform):
    """Apply inverse determinant-space transforms to CI tensors state by state."""
    out = np.empty(ci_tensors.shape, dtype=np.result_type(ci_tensors, alpha_transform, beta_transform))
    for i, ci in enumerate(ci_tensors):
        alpha_rot = np.linalg.solve(alpha_transform, ci)
        out[i] = np.linalg.solve(beta_transform, alpha_rot.T).T
    return out


def _biorthogonal_ci_overlap_from_prep(cibra, ciket, prep, dtype):
    """Biorthogonal CI overlap from precomputed active-space overlap prep."""
    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    bra_alpha = _unique_rows_first(cibra.binary[:, 0, :])
    bra_beta = _unique_rows_first(cibra.binary[:, 1, :])
    ket_alpha = _unique_rows_first(ciket.binary[:, 0, :])
    ket_beta = _unique_rows_first(ciket.binary[:, 1, :])

    nalpha_bra, nbeta_bra = len(bra_alpha), len(bra_beta)
    nalpha_ket, nbeta_ket = len(ket_alpha), len(ket_beta)

    if nalpha_bra * nbeta_bra != nsd_bra or nalpha_ket * nbeta_ket != nsd_ket:
        raise ValueError("Biorthogonal overlap candidate requires separable alpha/beta determinant grids.")
    if not np.array_equal(bra_alpha, ket_alpha) or not np.array_equal(bra_beta, ket_beta):
        raise ValueError("Biorthogonal overlap candidate requires matching alpha and beta string bases.")

    ci_bra = _as_state_ci_matrix(cibra.ci, nsd_bra).reshape((-1, nalpha_bra, nbeta_bra))
    ci_ket = _as_state_ci_matrix(ciket.ci, nsd_ket).reshape((-1, nalpha_ket, nbeta_ket))

    g_left_alpha = _string_transform_matrix(prep.x_left, bra_alpha, dtype)
    g_left_beta = _string_transform_matrix(prep.x_left, bra_beta, dtype)
    g_right_alpha = _string_transform_matrix(prep.x_right, ket_alpha, dtype)
    g_right_beta = _string_transform_matrix(prep.x_right, ket_beta, dtype)

    ci_bra_bio = _transform_ci_tensors_to_biorthogonal_basis(ci_bra, g_left_alpha, g_left_beta)
    ci_ket_bio = _transform_ci_tensors_to_biorthogonal_basis(ci_ket, g_right_alpha, g_right_beta)

    return prep.core_factor * contract('Xab,Yab->XY', ci_bra_bio.conj(), ci_ket_bio)


def _biorthogonal_ci_overlap_candidate(cibra, ciket, s=None):
    """Private candidate overlap using active-space biorthogonal CI transforms."""
    s = _compute_ci_mo_overlap(cibra, ciket, s=s)

    dtype = np.result_type(s, np.asarray(cibra.ci), np.asarray(ciket.ci))

    prep = _prepare_biorthogonal_overlap(
        s,
        cibra.ncore,
        ciket.ncore,
        cibra.ncas,
        ciket.ncas,
        dtype,
    )
    return _biorthogonal_ci_overlap_from_prep(cibra, ciket, prep, dtype)

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
    return _factorized_ci_overlap(
        cibra,
        ciket,
        s=s,
    )


class CASCIScanner:
    """Stateful multi-state CASCI scanner.

    Calling the scanner with Cartesian coordinates or a Molecule-like object
    updates the HF reference through ``mf.as_scanner()``, runs CASCI, stores the
    result as ``last_result``, and returns the CASCI object.
    """

    def __init__(
        self,
        mc,
        nstates=None,
        method='direct_ci',
        build_driver=None,
        run_kwargs=None,
        reuse_ci=False,
    ):
        self.template = mc
        self.mf = mc.mf
        self.mol = mc.mol
        self.nstates = int(nstates if nstates is not None else (mc.nstates or 1))
        self.method = method
        self.run_kwargs = dict(run_kwargs or {})
        self.reuse_ci = bool(reuse_ci)
        self.last_result = None
        self._mf_scanner = (
            self.mf.as_scanner(build_driver=build_driver)
            if hasattr(self.mf, "as_scanner")
            else None
        )

    def __call__(self, mol_or_geom):
        if self._mf_scanner is not None:
            self._mf_scanner(mol_or_geom)
            mf = self._mf_scanner.mf
        else:
            mf = self.mf
            if hasattr(mf, "mol") and isinstance(mol_or_geom, np.ndarray):
                mol = mf.mol
                mol.set_geom(np.asarray(mol_or_geom, dtype=float).reshape(mol.natom, 3))
                mol.build(driver=getattr(mol, "_build_driver", None) or "gbasis")
                mf.run()
            elif mol_or_geom is not None and mol_or_geom is not getattr(mf, "mol", None):
                mf.mol = mol_or_geom
                mf.run()

        scanner_mc = self.template.__class__(
            mf,
            ncas=self.template.ncas,
            nelecas=self.template.nelecas,
            spin=self.template.spin,
            verbose=self.template.verbose,
        )
        scanner_mc.binary = self.template.binary
        scanner_mc.spin_purification = self.template.spin_purification
        scanner_mc.ss = self.template.ss
        scanner_mc.shift = self.template.shift
        scanner_mc.use_cholesky_integrals = self.template.use_cholesky_integrals

        options = dict(self.run_kwargs)
        method = options.pop("method", self.method)
        ci0 = options.pop("ci0", None)
        if ci0 is None and self.reuse_ci and self.last_result is not None:
            ci0 = getattr(self.last_result, "ci", None)
        scanner_mc.run(
            nstates=self.nstates,
            method=method,
            ci0=ci0,
            **options,
        )

        self.mf = mf
        self.mol = mf.mol
        self.last_result = scanner_mc
        return scanner_mc

    def overlap(self, left, right):
        return overlap(left, right)

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
