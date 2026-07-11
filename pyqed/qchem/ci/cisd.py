#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:19:05 2024

CISD

@author: Bing Gu (gubing@westlake.edu.cn)

"""
import numpy as np

from itertools import combinations
from scipy.sparse.linalg import eigsh
from functools import reduce
from opt_einsum import contract
from pyqed import dag
from pyqed.qchem.ci.fci import SlaterCondon, CI_H, get_SO_matrix
from pyqed.qchem.mcscf.direct_ci import (
    _sigma_values_conn_numba,
    davidson_lowest,
    hamiltonian_matrix_elements,
)


from pyqed.qchem.ci.cis import CI

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


def _nuclear_energy(mf):
    if hasattr(mf, 'energy_nuc'):
        return mf.energy_nuc()
    if hasattr(mf, 'e_nuc') and mf.e_nuc is not None:
        return mf.e_nuc
    return mf.mol.energy_nuc()


def determinantsign(Binary):

    sign = np.cumsum( Binary, axis=2)
    for I in range(len(Binary)):
        iia = np.where( Binary[I,0] == 1)[0]
        iib = np.where( Binary[I,1] == 1)[0]
        sign[I, 0, iia] = np.arange(0, len(iia), 1)
        sign[I, 1, iib] = np.arange(0, len(iib), 1)
    return ( (-1.)**(sign) ).astype(np.int8)


def _spinorb_annihilate(bits, idx):
    if ((bits >> idx) & 1) == 0:
        return None, 0
    parity = (bits & ((1 << idx) - 1)).bit_count() & 1
    sign = -1 if parity else 1
    return bits ^ (1 << idx), sign


def _spinorb_create(bits, idx):
    if (bits >> idx) & 1:
        return None, 0
    parity = (bits & ((1 << idx) - 1)).bit_count() & 1
    sign = -1 if parity else 1
    return bits | (1 << idx), sign


def _build_cisd_det_cache(binary):
    nsd, _, nmo = binary.shape
    det_bits = [0] * nsd
    det_index = {}
    occ = [None] * nsd

    for i in range(nsd):
        bits = 0
        occ_a = tuple(np.flatnonzero(binary[i, 0]).tolist())
        occ_b = tuple(np.flatnonzero(binary[i, 1]).tolist())
        for p in occ_a:
            bits |= 1 << p
        for p in occ_b:
            bits |= 1 << (nmo + p)
        det_bits[i] = bits
        det_index[bits] = i
        occ[i] = (occ_a, occ_b)

    return det_bits, det_index, occ


def _apply_spin_excitation(bits, nmo, i, a, spin):
    bits_1, sign_1 = _spinorb_annihilate(bits, spin * nmo + i)
    if bits_1 is None:
        return None
    bits_2, sign_2 = _spinorb_create(bits_1, spin * nmo + a)
    if bits_2 is None:
        return None
    return bits_2, sign_1 * sign_2


def _build_restricted_cisd_transform(binary, nocc, nmo, det_bits, det_index):
    nvir = nmo - nocc
    n_amp = 1 + nocc * nvir + (nocc * nvir) ** 2
    transform = np.zeros((binary.shape[0], n_amp), dtype=float)

    ref_bits = det_bits[0]
    col = 0
    transform[0, col] = 1.0
    col += 1

    for i in range(nocc):
        for a in range(nocc, nmo):
            for spin in (0, 1):
                bits_out, sign = _apply_spin_excitation(ref_bits, nmo, i, a, spin)
                transform[det_index[bits_out], col] += sign
            col += 1

    for i in range(nocc):
        for a in range(nocc, nmo):
            for j in range(nocc):
                for b in range(nocc, nmo):
                    for spin_2 in (0, 1):
                        out_1 = _apply_spin_excitation(ref_bits, nmo, j, b, spin_2)
                        if out_1 is None:
                            continue
                        bits_1, sign_1 = out_1
                        for spin_1 in (0, 1):
                            out_2 = _apply_spin_excitation(bits_1, nmo, i, a, spin_1)
                            if out_2 is None:
                                continue
                            bits_2, sign_2 = out_2
                            transform[det_index[bits_2], col] += sign_1 * sign_2
                    col += 1

    if col != n_amp:
        raise RuntimeError("Restricted CISD amplitude transform size mismatch.")
    return transform


def _orthonormalize_restricted_cisd_basis(transform, tol=1e-10):
    overlap = transform.T @ transform
    evals, evecs = np.linalg.eigh(overlap)
    keep = evals > tol
    if not np.any(keep):
        raise RuntimeError("Restricted CISD basis orthogonalization failed.")
    orth = evecs[:, keep] / np.sqrt(evals[keep])
    return overlap, orth


def _build_restricted_mo_integrals(mf):
    coeff = np.asarray(mf.mo_coeff)
    if coeff.ndim != 2:
        raise ValueError("Restricted CISD requires a single RHF MO coefficient matrix.")
    hcore = np.asarray(mf.get_hcore())
    h1 = coeff.conj().T @ hcore @ coeff

    fock = None
    try:
        dm = mf.make_rdm1() if hasattr(mf, 'make_rdm1') else getattr(mf, 'dm', None)
        if dm is not None and hasattr(mf, 'get_veff'):
            fock_ao = hcore + np.asarray(mf.get_veff(dm=dm))
            fock = coeff.conj().T @ fock_ao @ coeff
    except Exception:
        fock = None
    if fock is None:
        fock = np.diag(np.asarray(mf.mo_energy))

    eri = None
    if hasattr(mf, 'get_eri'):
        try:
            eri_candidate = np.asarray(mf.get_eri(representation='mo'))
            if eri_candidate.ndim == 4:
                eri = eri_candidate
        except Exception:
            eri = None
    if eri is None:
        eri_ao = getattr(mf, '_eri', None)
        if eri_ao is None:
            eri_ao = getattr(mf.mol, 'eri', None)
        if eri_ao is None:
            raise ValueError("Restricted CISD requires AO ERIs or an MO ERI provider.")
        eri = contract('pqrs,pi,qj,rk,sl->ijkl', eri_ao, coeff.conj(), coeff, coeff.conj(), coeff)

    return np.asarray(fock), np.asarray(eri)


def _restricted_cisd_diagonal(fock, eri, nocc):
    mo_energy = np.asarray(np.diag(fock)).real
    nmo = mo_energy.size
    nvir = nmo - nocc
    jdiag = np.zeros((nmo, nmo), dtype=mo_energy.dtype)
    kdiag = np.zeros((nmo, nmo), dtype=mo_energy.dtype)
    oooo = eri[:nocc, :nocc, :nocc, :nocc]
    oovv = eri[:nocc, :nocc, nocc:, nocc:]
    ovvo = eri[:nocc, nocc:, nocc:, :nocc]
    vvvv = eri[nocc:, nocc:, nocc:, nocc:]

    jdiag[:nocc, :nocc] = np.einsum('iijj->ij', oooo, optimize=True)
    kdiag[:nocc, :nocc] = np.einsum('jiij->ij', oooo, optimize=True)
    jdiag[:nocc, nocc:] = np.einsum('iijj->ij', oovv, optimize=True)
    kdiag[:nocc, nocc:] = np.einsum('ijji->ij', ovvo, optimize=True)
    jdiag[nocc:, nocc:] = np.einsum('iijj->ij', vvvv, optimize=True)

    jksum = (jdiag[:nocc, :nocc] * 2 - kdiag[:nocc, :nocc]).sum()
    ehf = mo_energy[:nocc].sum() * 2 - jksum
    e_ia = mo_energy[nocc:][None, :] - mo_energy[:nocc][:, None]
    e_ia -= jdiag[:nocc, nocc:] - kdiag[:nocc, nocc:]
    e1diag = ehf + e_ia
    e2diag = (
        e_ia[:, None, :, None]
        + e_ia[None, :, None, :]
        + ehf
        + jdiag[:nocc, :nocc].reshape(nocc, nocc, 1, 1)
        - jdiag[:nocc, nocc:].reshape(nocc, 1, 1, nvir)
        - jdiag[:nocc, nocc:].reshape(1, nocc, nvir, 1)
        + jdiag[nocc:, nocc:].reshape(1, 1, nvir, nvir)
    )
    return np.hstack((ehf, e1diag.reshape(-1), e2diag.reshape(-1)))


def _restricted_cisd_init_guess(fock, eri, nocc):
    mo_e = np.asarray(np.diag(fock)).real
    nmo = mo_e.size
    nvir = nmo - nocc
    e_ia = mo_e[:nocc, None] - mo_e[nocc:][None, :]
    fov = fock[:nocc, nocc:]
    ovvo = eri[:nocc, nocc:, nocc:, :nocc]
    c0 = 1.0
    c1 = fov / e_ia
    c2 = 2 * ovvo.transpose(0, 3, 1, 2) - ovvo.transpose(0, 3, 2, 1)
    c2 /= e_ia[:, None, :, None] + e_ia[None, :, None, :]
    emp2 = np.einsum('ijab,iabj->', c2, ovvo, optimize=True)
    if abs(emp2) < 1e-3 and abs(c1).sum() < 1e-3:
        c1 = 1e-1 / e_ia
    return emp2, np.hstack((c0, c1.ravel(), c2.ravel()))


def _restricted_cisd_sigma(civec, nocc, nmo, fock, eri):
    nvir = nmo - nocc
    c0 = civec[0]
    c1 = civec[1:nocc * nvir + 1].reshape(nocc, nvir)
    c2 = civec[nocc * nvir + 1:].reshape(nocc, nocc, nvir, nvir)

    foo = fock[:nocc, :nocc].copy()
    fov = fock[:nocc, nocc:].copy()
    fvv = fock[nocc:, nocc:].copy()
    oooo = eri[:nocc, :nocc, :nocc, :nocc]
    oovv = eri[:nocc, :nocc, nocc:, nocc:]
    ovvo = eri[:nocc, nocc:, nocc:, :nocc]
    ovoo = eri[:nocc, nocc:, :nocc, :nocc]
    ovvv = eri[:nocc, nocc:, nocc:, nocc:]
    vvvv = eri[nocc:, nocc:, nocc:, nocc:]

    t2 = 0.5 * np.einsum('ijcd,acdb->ijab', c2, vvvv, optimize=True)

    t1 = fov * c0
    t1 += np.einsum('ib,ab->ia', c1, fvv, optimize=True)
    t1 -= np.einsum('ja,ji->ia', c1, foo, optimize=True)

    t2 += 0.5 * np.einsum('kilj,klab->ijab', oooo, c2, optimize=True)
    t2 += np.einsum('ijac,bc->ijab', c2, fvv, optimize=True)
    t2 -= np.einsum('kj,kiba->jiba', foo, c2, optimize=True)
    t2 += np.einsum('ia,jb->ijab', c1, fov, optimize=True)

    eris_oVoV = oovv.transpose(0, 2, 1, 3)
    tmp = np.einsum('kbjc,ikca->jiba', eris_oVoV, c2, optimize=True)
    t2 -= 0.5 * tmp
    t2 -= tmp.transpose(1, 0, 2, 3)

    t2 += ovvo.transpose(0, 3, 1, 2) * (0.5 * c0)
    t1 += 2 * np.einsum('ia,iabj->jb', c1, ovvo, optimize=True)
    t1 -= np.einsum('ib,iajb->ja', c1, eris_oVoV, optimize=True)

    ovov = -0.5 * eris_oVoV
    ovov += ovvo.transpose(3, 1, 0, 2)
    theta = 2 * c2.transpose(2, 0, 1, 3) - c2.transpose(2, 1, 0, 3)
    for j in range(nocc):
        t2[:, j] += np.einsum('ckb,ckia->iab', ovov[j], theta, optimize=True)

    t1 += np.einsum('aijb,ia->jb', theta, fov, optimize=True)
    t1 -= np.einsum('bjka,jbki->ia', theta, ovoo, optimize=True)
    t2 -= np.einsum('jbik,ka->jiba', ovoo.conj(), c1, optimize=True)

    t1 += np.einsum('cjib,jcba->ia', theta, ovvv.conj(), optimize=True)
    t2 += np.einsum('iacb,jc->ijab', ovvv.conj(), c1, optimize=True)

    for i in range(nocc):
        if i > 0:
            t2[i, :i] += t2[:i, i].transpose(0, 2, 1)
            t2[:i, i] = t2[i, :i].transpose(0, 2, 1)
        t2[i, i] = t2[i, i] + t2[i, i].T

    t0 = 2 * np.einsum('ia,ia->', fov, c1, optimize=True)
    t0 += 2 * np.einsum('iabj,ijab->', ovvo, c2, optimize=True)
    t0 -= np.einsum('iabj,jiab->', ovvo, c2, optimize=True)
    return np.hstack((t0, t1.ravel(), t2.ravel()))


def _restricted_cisd_dot(v1, v2, nocc, nmo):
    nvir = nmo - nocc
    hijab = v2[1 + nocc * nvir:].reshape(nocc, nocc, nvir, nvir)
    cijab = v1[1 + nocc * nvir:].reshape(nocc, nocc, nvir, nvir)
    val = np.dot(v1, v2) * 2 - v1[0] * v2[0]
    val -= np.einsum('jiab,ijab->', cijab, hijab, optimize=True)
    return val


def _cisd_rdm1_from_determinants(ci, binary, det_bits, det_index, occ):
    nsd, _, nmo = binary.shape
    dm1 = np.zeros((nmo, nmo), dtype=np.result_type(ci, np.complex128))

    for j in range(nsd):
        c_j = ci[j]
        bits_j = det_bits[j]
        for spin in range(2):
            offset = spin * nmo
            for p in occ[j][spin]:
                bits_1, sign_1 = _spinorb_annihilate(bits_j, offset + p)
                if bits_1 is None:
                    continue
                for q in range(nmo):
                    bits_2, sign_2 = _spinorb_create(bits_1, offset + q)
                    if bits_2 is None:
                        continue
                    i = det_index.get(bits_2)
                    if i is not None:
                        dm1[p, q] += ci[i].conjugate() * c_j * (sign_1 * sign_2)

    return dm1


def _cisd_rdm2_from_determinants(ci, binary, det_bits, det_index, occ):
    nsd, _, nmo = binary.shape
    dm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=np.result_type(ci, np.complex128))

    for j in range(nsd):
        c_j = ci[j]
        bits_j = det_bits[j]
        for sigma in range(2):
            offset_sigma = sigma * nmo
            occ_sigma = occ[j][sigma]
            for q in occ_sigma:
                bits_1, sign_1 = _spinorb_annihilate(bits_j, offset_sigma + q)
                if bits_1 is None:
                    continue
                for tau in range(2):
                    offset_tau = tau * nmo
                    if tau == sigma:
                        occ_tau = tuple(x for x in occ_sigma if x != q)
                    else:
                        occ_tau = occ[j][tau]
                    for s in occ_tau:
                        bits_2, sign_2 = _spinorb_annihilate(bits_1, offset_tau + s)
                        if bits_2 is None:
                            continue
                        for r in range(nmo):
                            bits_3, sign_3 = _spinorb_create(bits_2, offset_tau + r)
                            if bits_3 is None:
                                continue
                            for p in range(nmo):
                                bits_4, sign_4 = _spinorb_create(bits_3, offset_sigma + p)
                                if bits_4 is None:
                                    continue
                                i = det_index.get(bits_4)
                                if i is not None:
                                    dm2[p, q, r, s] += (
                                        ci[i].conjugate()
                                        * c_j
                                        * (sign_1 * sign_2 * sign_3 * sign_4)
                                    )

    return dm2




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






class CISD(CI):
    """
    restricted CISD
    """
    def buildH(self):
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc

        nsd = 1 + 2 * nocc * nvir + nocc * (nocc - 1) * nvir * (nvir - 1) // 2 + nocc**2 * nvir**2
        Binary = np.zeros((nsd, 2, nmo), dtype=np.int8)
        Binary[:] = [np.asarray(self.mo_occ, dtype=np.int8) // 2, ] * 2

        I = 1
        for i in range(nocc):
            for a in range(nocc, nmo):
                Binary[I, 0, i] -= 1
                Binary[I, 0, a] += 1
                Binary[I + 1, 1, i] -= 1
                Binary[I + 1, 1, a] += 1
                I += 2

        for i in range(nocc):
            for j in range(i):
                for a in range(nocc, nmo):
                    for b in range(nocc, a):
                        Binary[I, 0, i] -= 1
                        Binary[I, 0, j] -= 1
                        Binary[I, 0, b] += 1
                        Binary[I, 0, a] += 1

                        Binary[I + 1, 1, i] -= 1
                        Binary[I + 1, 1, j] -= 1
                        Binary[I + 1, 1, a] += 1
                        Binary[I + 1, 1, b] += 1
                        I += 2

        for i in range(nocc):
            for a in range(nocc, nmo):
                for j in range(nocc):
                    for b in range(nocc, nmo):
                        Binary[I, 0, i] -= 1
                        Binary[I, 0, a] += 1
                        Binary[I, 1, j] -= 1
                        Binary[I, 1, b] += 1
                        I += 1

        assert I == nsd

        self.binary = Binary
        self._rdm_det_bits = None
        self._rdm_det_index = None
        self._rdm_occ = None
        det_bits, det_index, occ = self._ensure_rdm_cache()
        self._amp_to_det = _build_restricted_cisd_transform(
            self.binary,
            self.nocc,
            self.nmo,
            det_bits,
            det_index,
        )
        self._restricted_fock, self._restricted_eri = _build_restricted_mo_integrals(self.mf)
        self._direct_diag = _restricted_cisd_diagonal(
            self._restricted_fock,
            self._restricted_eri,
            self.nocc,
        )
        self.emp2, self._direct_guess = _restricted_cisd_init_guess(
            self._restricted_fock,
            self._restricted_eri,
            self.nocc,
        )
        self.H1 = None
        self.H2 = None
        self.SC1 = None
        self.SC2 = None
        self.H_diag = None
        self.H_A = None
        self.H_B = None
        self.H_AA = None
        self.H_BB = None
        self.H_AB = None
        self._amp_overlap = None
        self._amp_orth = None
        self._orth_det_basis = None
        self._projected_diag = None
        self._projected_guess = None
        self._ci_representation = None
        self._orth_ci_from_direct = None
        self.H = None
        return self._direct_diag

    def _state_ci_vector(self, state_id=0):
        if isinstance(self.ci, list):
            return self.ci[state_id]
        return self.ci

    def _ensure_rdm_cache(self):
        if getattr(self, '_rdm_det_bits', None) is None:
            (
                self._rdm_det_bits,
                self._rdm_det_index,
                self._rdm_occ,
            ) = _build_cisd_det_cache(self.binary)
        return self._rdm_det_bits, self._rdm_det_index, self._rdm_occ

    def _det_sigma(self, vec):
        self._ensure_det_sigma_cache()
        return _sigma_values_conn_numba(
            self.H_diag,
            self.H_A,
            self.H_B,
            self.H_AA,
            self.H_BB,
            self.H_AB,
            vec,
            self.SC1[0],
            self.SC1[1],
            self.SC1[4],
            self.SC1[5],
            self.SC2[0],
            self.SC2[1],
            self.SC2[4],
            self.SC2[5],
            self.SC2[8],
            self.SC2[9],
        )

    def _ensure_det_sigma_cache(self):
        if self.H_diag is not None:
            return
        self.H1, self.H2 = get_SO_matrix(self.mf)
        self.SC1, self.SC2 = SlaterCondon(self.binary)
        (
            self.H_diag,
            self.H_A,
            self.H_B,
            self.H_AA,
            self.H_BB,
            self.H_AB,
        ) = hamiltonian_matrix_elements(self.binary, self.H1, self.H2, self.SC1, self.SC2)
        self._amp_overlap, self._amp_orth = _orthonormalize_restricted_cisd_basis(
            self._amp_to_det
        )
        self._orth_det_basis = self._amp_to_det @ self._amp_orth
        self._projected_diag = np.einsum(
            'pi,p,pi->i',
            self._orth_det_basis,
            self.H_diag,
            self._orth_det_basis,
            optimize=True,
        )
        self._projected_guess = self._orth_det_basis.T @ (self._amp_to_det @ self._direct_guess)

    def _restricted_sigma(self, vec):
        return _restricted_cisd_sigma(
            vec,
            self.nocc,
            self.nmo,
            self._restricted_fock,
            self._restricted_eri,
        )

    def _projected_sigma(self, vec):
        det_vec = self._orth_det_basis @ vec
        sigma_det = self._det_sigma(det_vec)
        return self._orth_det_basis.T @ sigma_det

    def _build_projected_hamiltonian(self):
        dim = self._orth_det_basis.shape[1]
        h = np.empty((dim, dim), dtype=float)
        for j in range(dim):
            h[:, j] = self._projected_sigma(np.eye(dim, dtype=float)[:, j])
        h = 0.5 * (h + h.T)
        return h

    def _solve_projected_states(self, nstates=1, tol=1e-8, guess=None):
        self._ensure_det_sigma_cache()
        dim = self._orth_det_basis.shape[1]
        guess = self._projected_guess if guess is None else guess
        if nstates >= dim:
            h_proj = self._build_projected_hamiltonian()
            evals, evecs = np.linalg.eigh(h_proj)
            evals = evals[:nstates]
            evecs = evecs[:, :nstates]
        else:
            try:
                evals, evecs = davidson_lowest(
                    self._projected_sigma,
                    self._projected_diag,
                    nroots=nstates,
                    tol=tol,
                    max_cycle=self.max_cycle,
                    max_subspace=min(dim, max(128, dim // 4)),
                    guess=guess,
                )
            except RuntimeError:
                h_proj = self._build_projected_hamiltonian()
                evals, evecs = np.linalg.eigh(h_proj)
                evals = evals[:nstates]
                evecs = evecs[:, :nstates]
        return evals, evecs

    def _ensure_projected_state_from_direct(self, tol=1e-8):
        if self._ci_representation != 'restricted':
            return
        if self._orth_ci_from_direct is not None:
            return
        nstates = len(self.ci) if isinstance(self.ci, list) else 1
        _, evecs = self._solve_projected_states(nstates=nstates, tol=tol)
        if nstates == 1:
            self._orth_ci_from_direct = evecs[:, 0]
        else:
            self._orth_ci_from_direct = [evecs[:, i] for i in range(nstates)]

    def _determinant_ci_vector(self, state_id=0):
        vec = self._state_ci_vector(state_id)
        if self._ci_representation == 'restricted':
            self._ensure_projected_state_from_direct()
            if isinstance(self._orth_ci_from_direct, list):
                return self._orth_det_basis @ self._orth_ci_from_direct[state_id]
            return self._orth_det_basis @ self._orth_ci_from_direct
        if self._ci_representation == 'orth':
            return self._orth_det_basis @ vec
        return vec

    def run(self, ci0=None, nstates=1, tol=1e-8, sigma_backend='projected'):
        self.buildH()
        if sigma_backend == 'direct':
            diag = self._direct_diag - self._direct_diag[0]
            level_shift = getattr(self, 'level_shift', 0.0)
            guess = self._direct_guess if ci0 is None else ci0
            if isinstance(guess, np.ndarray) and guess.ndim == 1:
                guess = [guess]
            elif not isinstance(guess, (list, tuple)):
                guess = [np.asarray(guess)]

            def op(xs):
                return [self._restricted_sigma(x) for x in xs]

            def precond(x, e, *args):
                denom = diag - (e - level_shift)
                denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
                return x / denom

            def direct_dot(x1, x2):
                return _restricted_cisd_dot(x1, x2, self.nocc, self.nmo)

            try:
                evals, evecs = davidson_lowest(
                    lambda x: self._restricted_sigma(x),
                    diag,
                    nroots=nstates,
                    tol=tol,
                    max_cycle=self.max_cycle,
                    max_subspace=max(20, min(self._direct_diag.size, 128)),
                    guess=guess,
                )
            except RuntimeError:
                sigma_backend = 'projected'
            else:
                if nstates == 1:
                    evals = np.asarray([evals[0]])
                    evecs = np.asarray([evecs[:, 0]]).T
                else:
                    evals = np.asarray(evals)
                    evecs = np.asarray(evecs)

        if sigma_backend == 'projected':
            guess = self._projected_guess if ci0 is None else ci0
            evals, evecs = self._solve_projected_states(nstates=nstates, tol=tol, guess=guess)

        self.converged = True
        if sigma_backend == 'direct':
            self.e_corr = np.asarray(evals)
            self.e_tot = self.e_corr + self.mf.e_tot
        else:
            self.e_tot = evals + _nuclear_energy(self.mf)
            self.e_corr = self.e_tot - self.mf.e_tot
        if nstates == 1:
            self.e_tot = self.e_tot[0]
            self.e_corr = self.e_corr[0]
            self.ci = evecs[:, 0]
        else:
            self.ci = [evecs[:, n] for n in range(nstates)]
        self._ci_representation = 'restricted' if sigma_backend == 'direct' else 'orth'

        return self

    def cisdvec_to_amplitudes(self, civec, nmo=None, nocc=None, copy=True):
        """
        extract the CI coefficients for each block (ground, singles, doubles)

        The number of states considered in a CISD C.I. involving n doubly-occupied M.O.s and m empty M.O.s is:

        No. of States = 1 + 2.n.m  + (n.m)2 + (n(n-1).m(m-1))/2

        This represents:

        Ground state
        + (one α electron excited + one β electron excited)
        + (one α electron and one β electron excited)
        + (two α electrons excited + two β electrons excited)

        The first term represents the ground state, the second term represents number of one-electron excitations, and the third and fourth terms represent the number of two-electron excitations.

        Parameters
        ----------
        civec : TYPE
            DESCRIPTION.
        nmo : TYPE, optional
            DESCRIPTION. The default is None.
        nocc : TYPE, optional
            DESCRIPTION. The default is None.
        copy : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        c0 : TYPE
            DESCRIPTION.
        c1 : TYPE
            DESCRIPTION.
        c2 : TYPE
            DESCRIPTION.

        """

        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc

        nvir = nmo - nocc
        c0 = civec[0]
        cp = lambda x: (x.copy() if copy else x)
        c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
        c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir))

        return c0, c1, c2

    def make_rdm1(self, state_id=0, ao_repr=False):
        if self._ci_representation == 'restricted':
            return make_rdm1(self, civec=self._state_ci_vector(state_id), ao_repr=ao_repr)
        ci = self._determinant_ci_vector(state_id)
        det_bits, det_index, occ = self._ensure_rdm_cache()
        dm1 = _cisd_rdm1_from_determinants(ci, self.binary, det_bits, det_index, occ)
        if ao_repr:
            return self.mo_coeff @ dm1 @ dag(self.mo_coeff)
        return dm1
    
    def make_rdm2(self, state_id=0, ao_repr=False):
        if self._ci_representation == 'restricted':
            return make_rdm2(self, civec=self._state_ci_vector(state_id), ao_repr=ao_repr)
        ci = self._determinant_ci_vector(state_id)
        det_bits, det_index, occ = self._ensure_rdm_cache()
        dm2 = _cisd_rdm2_from_determinants(ci, self.binary, det_bits, det_index, occ)
        if ao_repr:
            return contract(
                'pi,qj,ijkl,rk,sl->pqrs',
                self.mo_coeff,
                self.mo_coeff,
                dm2,
                self.mo_coeff,
                self.mo_coeff,
            )
        return dm2

    def make_rdm12(self, state_id=0, ao_repr=False):
        dm1 = self.make_rdm1(state_id=state_id, ao_repr=ao_repr)
        dm2 = self.make_rdm2(state_id=state_id, ao_repr=ao_repr)
        return dm1, dm2


def make_rdm1(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Spin-traced one-particle density matrix in MO basis (the occupied-virtual
    blocks from the orbital response contribution are not included).

    .. math::

        D[p,q] = <q_\alpha^\dagger p_\alpha> + <q_\beta^\dagger p_\beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Refs:
        PySCF

    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc

    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    return _make_rdm1(myci, d1, with_frozen=True, ao_repr=ao_repr)


def make_rdm2(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    The contraction with chemist ERIs is
    E = einsum('pqrs,pqrs', eri, dm2)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc

    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    d2 = _gamma2_intermediates(myci, civec, nmo, nocc)
    return _make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True, ao_repr=ao_repr)

def _gamma1_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    dvo = c0.conj() * c1.T
    dvo += np.einsum('jb,ijab->ai', c1.conj(), c2) * 2
    dvo -= np.einsum('jb,ijba->ai', c1.conj(), c2)
    dov = dvo.T.conj()

    theta = c2*2 - c2.transpose(0,1,3,2)
    doo  = -np.einsum('ia,ka->ik', c1.conj(), c1)
    doo -= contract('ijab,ikab->jk', c2.conj(), theta)
    dvv  = np.einsum('ia,ic->ac', c1, c1.conj())
    dvv += contract('ijab,ijac->bc', theta, c2.conj())
    return doo, dov, dvo, dvv


def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False, with_mf=True):
    r'''dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Refs:
        pyscf/cc/ccsd_rdm.py
    '''
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir
    dm1 = np.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T
    if with_mf:
        dm1[np.diag_indices(nocc)] += 2

    if with_frozen and mycc.frozen is not None:
        nmo = mycc.mo_occ.size
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        rdm1 = np.zeros((nmo,nmo), dtype=dm1.dtype)
        if with_mf:
            rdm1[np.diag_indices(nocc)] = 2
        moidx = np.where(mycc.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = contract('pi,ij,qj->pq', mo, dm1, mo.conj())
    return dm1


def _gamma2_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    nvir = nmo - nocc

    dovov = 2 * c0 * c2.conj().transpose(0, 2, 1, 3)
    dovov -= c0 * c2.conj().transpose(1, 2, 0, 3)

    doooo = contract('ijab,klab->ijkl', c2.conj(), c2)
    doooo = doooo.transpose(0, 2, 1, 3) - 0.5 * doooo.transpose(1, 2, 0, 3)

    dooov = -contract('ia,klac->klic', c1 * 2, c2.conj())
    dooov = dooov.transpose(0, 2, 1, 3) * 2 - dooov.transpose(1, 2, 0, 3)

    dvvvv = np.empty((nvir, nvir, nvir, nvir), dtype=np.result_type(civec))
    dovvv = np.empty((nocc, nvir, nvir, nvir), dtype=np.result_type(civec))
    for p in range(nvir):
        theta = c2[:, :, p] - 0.5 * c2[:, :, p].transpose(1, 0, 2)
        gvvvv = contract('ija,ijcd->acd', theta.conj(), c2)
        dvvvv[p] = gvvvv.conj().transpose(1, 0, 2)

        gvovv = np.einsum('ia,ikcd->akcd', (2 * c1[:, p:p+1]).conj(), c2, optimize=True)
        gvovv = gvovv.conj()
        dovvv[:, :, p:p+1] = gvovv.transpose(1, 3, 0, 2) * 2 - gvovv.transpose(1, 2, 0, 3)

    theta = 2 * c2 - c2.transpose(1, 0, 2, 3)
    doovv = np.einsum('ia,kc->ikca', c1.conj(), -c1, optimize=True)
    doovv -= contract('kjcb,kica->jiab', c2.conj(), theta)
    doovv -= contract('ikcb,jkca->ijab', c2.conj(), theta)

    dovvo = contract('ikac,jkbc->iabj', theta.conj(), theta)
    dovvo += np.einsum('ia,kc->iack', c1.conj(), c1, optimize=True) * 2

    return dovov, dvvvv, doooo, doovv, dovvo, None, dovvv, dooov


def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=False):
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = np.empty((nmo, nmo, nmo, nmo), dtype=doovv.dtype)

    dm2[:nocc, nocc:, :nocc, nocc:] = dovov
    dm2[:nocc, nocc:, :nocc, nocc:] += dovov.transpose(2, 3, 0, 1)
    dm2[nocc:, :nocc, nocc:, :nocc] = dm2[:nocc, nocc:, :nocc, nocc:].transpose(1, 0, 3, 2).conj()

    dm2[:nocc, :nocc, nocc:, nocc:] = doovv
    dm2[:nocc, :nocc, nocc:, nocc:] += doovv.transpose(1, 0, 3, 2).conj()
    dm2[nocc:, nocc:, :nocc, :nocc] = dm2[:nocc, :nocc, nocc:, nocc:].transpose(2, 3, 0, 1)

    dm2[:nocc, nocc:, nocc:, :nocc] = dovvo
    dm2[:nocc, nocc:, nocc:, :nocc] += dovvo.transpose(3, 2, 1, 0).conj()
    dm2[nocc:, :nocc, :nocc, nocc:] = dm2[:nocc, nocc:, nocc:, :nocc].transpose(1, 0, 3, 2).conj()

    dm2[nocc:, nocc:, nocc:, nocc:] = dvvvv
    dm2[nocc:, nocc:, nocc:, nocc:] += dvvvv.transpose(1, 0, 3, 2).conj()
    dm2[nocc:, nocc:, nocc:, nocc:] *= 2

    dm2[:nocc, :nocc, :nocc, :nocc] = doooo
    dm2[:nocc, :nocc, :nocc, :nocc] += doooo.transpose(1, 0, 3, 2).conj()
    dm2[:nocc, :nocc, :nocc, :nocc] *= 2

    dm2[:nocc, nocc:, nocc:, nocc:] = dovvv
    dm2[nocc:, nocc:, :nocc, nocc:] = dovvv.transpose(2, 3, 0, 1)
    dm2[nocc:, nocc:, nocc:, :nocc] = dovvv.transpose(3, 2, 1, 0).conj()
    dm2[nocc:, :nocc, nocc:, nocc:] = dovvv.transpose(1, 0, 3, 2).conj()

    dm2[:nocc, :nocc, :nocc, nocc:] = dooov
    dm2[:nocc, nocc:, :nocc, :nocc] = dooov.transpose(2, 3, 0, 1)
    dm2[:nocc, :nocc, nocc:, :nocc] = dooov.transpose(1, 0, 3, 2).conj()
    dm2[nocc:, :nocc, :nocc, :nocc] = dooov.transpose(3, 2, 1, 0).conj()

    if with_frozen and mycc.frozen is not None:
        nmo0 = nmo
        nmo = mycc.mo_occ.size
        nocc_frozen = np.count_nonzero(mycc.mo_occ > 0)
        rdm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=dm2.dtype)
        moidx = np.where(mycc.get_frozen_mask())[0]
        idx = (moidx.reshape(-1, 1) * nmo + moidx).ravel()
        rdm2.reshape(nmo ** 2, nmo ** 2)[np.ix_(idx, idx)] = dm2.reshape(nmo0 ** 2, nmo0 ** 2)
        dm2 = rdm2
        nocc = nocc_frozen

    if with_dm1:
        dm1 = _make_rdm1(mycc, d1, with_frozen=with_frozen)
        dm1[np.diag_indices(nocc)] -= 2

        for i in range(nocc):
            dm2[i, i, :, :] += dm1 * 2
            dm2[:, :, i, i] += dm1 * 2
            dm2[:, i, i, :] -= dm1
            dm2[i, :, :, i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i, i, j, j] += 4
                dm2[i, j, j, i] -= 2

    dm2 = dm2.transpose(1, 0, 3, 2)

    if ao_repr:
        mo = mycc.mo_coeff
        dm2 = contract('pi,qj,ijkl,rk,sl->pqrs', mo, mo, dm2, mo, mo)
    return dm2

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

        nocc = self.nocc
        nmo = self.nmo
        if isinstance(nocc, tuple):
            nocc_a, nocc_b = nocc
            nvir_a, nvir_b = self.nvir
        else:
            nocc_a = nocc_b = nocc
            nvir_a = nvir_b = nmo - nocc

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
        nsd = (
            1
            + nocc_a * nvir_a
            + nocc_b * nvir_b
            + nocc_a * (nocc_a - 1) * nvir_a * (nvir_a - 1) // 4
            + nocc_b * (nocc_b - 1) * nvir_b * (nvir_b - 1) // 4
            + nocc_a * nvir_a * nocc_b * nvir_b
        )

        # nsd = 1 + 4*nocc * nvir + nocc*(2*nocc-1)*nvir*(2*nvir-1)

        print('number of CISD determinants', nsd)

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


        mo_occ = np.asarray(self.mo_occ, dtype=np.int8)
        if mo_occ.ndim == 1:
            Binary[:] = [mo_occ // 2, ] * 2
        else:
            Binary[:] = mo_occ


        # singles
        I = 1
        for i in range(nocc_a):
            for a in range(nocc_a, nocc_a + nvir_a):
                Binary[I, 0, i] -= 1
                Binary[I, 0, a] += 1
                I += 1

        for i in range(nocc_b):
            for a in range(nocc_b, nocc_b + nvir_b):
                Binary[I, 1, i] -= 1
                Binary[I, 1, a] += 1
                I += 1

        # doubles aa, bb excitation     a^b^ji
        for i in range(nocc_a):
            for j in range(i): # i > j
                for a in range(nocc_a, nocc_a + nvir_a):
                    for b in range(nocc_a, a): # a > b
                        Binary[I, 0, i] -= 1
                        Binary[I, 0, j] -= 1
                        Binary[I, 0, b] += 1
                        Binary[I, 0, a] += 1
                        I += 1

        for i in range(nocc_b):
            for j in range(i): # i > j
                for a in range(nocc_b, nocc_b + nvir_b):
                    for b in range(nocc_b, a): # a > b
                        Binary[I, 1, i] -= 1
                        Binary[I, 1, j] -= 1
                        Binary[I, 1, a] += 1
                        Binary[I, 1, b] += 1
                        I += 1

        # doubles with ab -> ij excitation
        for i in range(nocc_a):
            for a in range(nocc_a, nocc_a + nvir_a):
                for j in range(nocc_b):
                    for b in range(nocc_b, nocc_b + nvir_b):

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

        # E, X = np.linalg.eigh(H_CI)
        # E, X = eigsh(H_CI, k=3, which='SA')


        return H_CI

    def run(self, ci0=None, nstates=1, tol=1e-6):

        H_CI = self.buildH()

        E, X = eigsh(H_CI, k=nstates, maxiter=self.max_cycle, \
                      which='SA', tol=tol, v0=ci0)

        self.e_tot = E + _nuclear_energy(self.mf)

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
    CI electronic overlap matrix (CIS, CISD, CASCI ...)

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
    S = np.zeros((nsd_bra, nsd_ket))


    for I in range(nsd_bra):
        occidx1_a  = [i for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
        occidx1_b  = [i for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

        # print('a', occidx1_a, occidx1_b)

        for J in range(nsd_ket):
            occidx2_a  = [i for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
            occidx2_b  = [i for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.

            S[I, J] = np.linalg.det(s[np.ix_(occidx1_a, occidx2_a)]) * \
                      np.linalg.det(s[np.ix_(occidx1_b, occidx2_b)])


    return contract('IB, IJ, JA', cibra.ci.conj(), S, ciket.ci)





if __name__=='__main__':
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
