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
from dataclasses import dataclass

from pyqed.qchem import get_veff
from pyqed.qchem.ci.fci import (
    givenΛgetB,
    SpinOuterProduct,
    get_fci_combos,
    SlaterCondon,
    CI_H,
    determinantsign,
    get_excitation_op,
)
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo

from pyqed.qchem.mcscf.casci import (
    h1e_for_cas,
    size_of_cas,
    spin_square as spin_square_from_rdm,
    transform_spatial_eri_to_mo,
    transform_eri_factors_to_mo_pair,
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
)
from pyqed.qchem import mcscf

from numba import njit, prange


DIRECT_CI_DENSE_FALLBACK_NDETS = 256


@dataclass
class DirectConnectivity:
    """
    Compact determinant connectivity cache for the matrix-free direct-CI solver.

    ``Binary`` is still stored on the CASCI object for overlap/RDM helpers.
    This structure only replaces the expensive all-pairs Slater-Condon setup in
    the direct-CI matvec path.
    """
    I_A: np.ndarray
    J_A: np.ndarray
    p_A: np.ndarray
    q_A: np.ndarray
    phase_A: np.ndarray
    I_B: np.ndarray
    J_B: np.ndarray
    p_B: np.ndarray
    q_B: np.ndarray
    phase_B: np.ndarray
    I_AA: np.ndarray
    J_AA: np.ndarray
    p_AA: np.ndarray
    q_AA: np.ndarray
    r_AA: np.ndarray
    s_AA: np.ndarray
    phase_AA: np.ndarray
    I_BB: np.ndarray
    J_BB: np.ndarray
    p_BB: np.ndarray
    q_BB: np.ndarray
    r_BB: np.ndarray
    s_BB: np.ndarray
    phase_BB: np.ndarray
    I_AB: np.ndarray
    J_AB: np.ndarray
    p_AB: np.ndarray
    q_AB: np.ndarray
    r_AB: np.ndarray
    s_AB: np.ndarray
    phase_AB: np.ndarray


def _orthonormalize_columns(V, tol=1e-12):
    """
    Return an orthonormal basis spanning the columns of ``V``.

    Davidson restarts and correction vectors can become nearly linearly
    dependent.  Using a small QR-based helper keeps the solver code readable and
    centralizes the rank filtering in one place.
    """
    if V.size == 0:
        return np.zeros((V.shape[0], 0), dtype=V.dtype)

    Q, R = np.linalg.qr(V, mode='reduced')
    keep = np.abs(np.diag(R)) > tol
    if not np.any(keep):
        return np.zeros((V.shape[0], 0), dtype=V.dtype)
    return Q[:, keep]


def _build_davidson_guess(diag, nroots, guess=None):
    """
    Build an initial Davidson subspace from the diagonal and optional user seeds.

    The default strategy mirrors standard CI Davidson practice: use unit vectors
    associated with the smallest diagonal Hamiltonian elements.  If the caller
    provides trial vectors via ``ci0`` we include them first, then fill any
    missing columns from the diagonal guess.
    """
    n = diag.size
    cols = []

    if guess is not None:
        if isinstance(guess, (list, tuple)):
            guess_cols = [np.asarray(v, dtype=float).reshape(n) for v in guess]
        else:
            arr = np.asarray(guess, dtype=float)
            if arr.ndim == 1:
                guess_cols = [arr.reshape(n)]
            else:
                guess_cols = [arr[:, i].reshape(n) for i in range(arr.shape[1])]
        cols.extend(guess_cols[:nroots])

    order = np.argsort(diag)
    for idx in order:
        e = np.zeros(n, dtype=float)
        e[idx] = 1.0
        cols.append(e)
        if len(cols) >= max(nroots, 2 * nroots):
            break

    return _orthonormalize_columns(np.column_stack(cols))


def _select_direct_ci_guess(casci, nstates, ci0=None):
    """
    Choose the starting vectors for the direct-CI eigensolver.

    Priority:
    1. Explicit user-provided ``ci0``
    2. Previously converged CI roots stored on the same CASCI object
    3. Fall back to the diagonal-based guess in ``_build_davidson_guess``

    Reusing previous CI roots is especially helpful for repeated runs on the
    same active space, which is exactly how we benchmark and scan solver
    settings.  It turns the Davidson solve into a true restart instead of
    rebuilding the subspace from unit vectors every time.
    """
    if ci0 is not None:
        return ci0
    if casci.ci is not None and len(casci.ci) > 0:
        return casci.ci[:nstates]
    return None


def davidson_lowest(matvec, diag, nroots=1, tol=1e-8, max_cycle=100,
                    max_subspace=None, guess=None):
    """
    Solve for the lowest ``nroots`` eigenpairs of a symmetric CI Hamiltonian.

    This Davidson implementation is tailored for the direct-CI backend:

    - ``matvec`` is the matrix-free CI sigma builder
    - ``diag`` is the precomputed Hamiltonian diagonal used for preconditioning
    - only a few lowest roots are needed, so block Davidson is a better fit
      than a generic sparse eigensolver

    The routine keeps the implementation compact and readable rather than trying
    to reproduce every feature of a production FCI Davidson solver.
    """
    n = diag.size
    if nroots < 1 or nroots > n:
        raise ValueError('nroots must be between 1 and the CI dimension.')

    if max_subspace is None:
        # A modest subspace is usually enough for low-root CASCI while keeping
        # the Rayleigh-Ritz diagonalization cheap.  This default works better
        # for the current direct-CI benchmarks than the larger initial choice.
        max_subspace = min(n, max(16, 6 * nroots))

    V = _build_davidson_guess(diag, nroots, guess=guess)
    AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])

    for _ in range(max_cycle):
        T = V.T @ AV
        theta_all, alpha_all = eigh(T)
        order = np.argsort(theta_all)[:nroots]
        theta = theta_all[order]
        alpha = alpha_all[:, order]

        ritz = V @ alpha
        Aritz = AV @ alpha
        resid = Aritz - ritz * theta
        resid_norm = np.linalg.norm(resid, axis=0)

        if np.all(resid_norm < tol):
            return theta, ritz

        new_vecs = []
        for root in range(nroots):
            if resid_norm[root] < tol:
                continue

            denom = theta[root] - diag
            safe = np.where(np.abs(denom) > 1e-12, denom, np.where(denom >= 0, 1e-12, -1e-12))
            corr = resid[:, root] / safe

            if V.shape[1] > 0:
                corr -= V @ (V.T @ corr)
            for prev in new_vecs:
                corr -= prev * np.dot(prev, corr)

            norm = np.linalg.norm(corr)
            if norm > 1e-12:
                new_vecs.append(corr / norm)

        if not new_vecs:
            return theta, ritz

        if V.shape[1] + len(new_vecs) > max_subspace:
            # Restart from the current Ritz vectors.  This keeps the subspace
            # size bounded without losing the best approximation accumulated so
            # far.
            V = _orthonormalize_columns(ritz)
            AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
        else:
            new_block = np.column_stack(new_vecs)
            AV_new = np.column_stack([matvec(new_block[:, i]) for i in range(new_block.shape[1])])
            V = np.column_stack((V, new_block))
            AV = np.column_stack((AV, AV_new))

    raise RuntimeError('Davidson solver did not converge within max_cycle iterations.')


def build_spin_square_operator(norb):
    """
    Return the active-space one- and two-electron pieces of the ``S^2`` operator.

    The tensor layout matches the direct-CI Hamiltonian convention:

    ``H1 = [h_alpha, h_beta]``
    ``H2 = [[eri_aa, eri_ab], [eri_ba, eri_bb]]``

    This is the same operator that is implicitly used by ``fix_spin()`` when the
    code adds a first-order ``J * S^2`` penalty. Keeping it as a standalone
    helper lets us evaluate ``<Psi|S^2|Psi>`` directly from CI vectors without
    building 1- and 2-RDMs.
    """
    h1 = np.asarray([0.75 * np.eye(norb), 0.75 * np.eye(norb)])
    h2 = np.zeros((2, 2, norb, norb, norb, norb))

    for p in range(norb):
        for q in range(norb):
            h2[:, :, p, q, q, p] -= 1.0
            h2[:, :, p, p, q, q] -= 0.5

    # Same-spin blocks need the antisymmetrized ``(pq||rs)`` form used by the
    # CI solver. Cross-spin blocks remain in Coulomb form.
    h2[0, 0] -= h2[0, 0].swapaxes(1, 3)
    h2[1, 1] -= h2[1, 1].swapaxes(1, 3)

    return h1, h2


def transform_active_space_spatial_integrals(mf, mo_coeff, ncas, ncore, use_cholesky=False):
    """
    Build the active-space spatial-orbital Hamiltonian for the direct-CI solver.

    Unlike ``get_SO_matrix()``, this helper keeps the two-electron piece as a
    single spatial ERI tensor ``(pq|rs)``. The matrix-free direct-CI kernel can
    then apply the spin cases logically:

    - same-spin terms use the antisymmetrized combination
      ``(pq|rs) - (ps|rq)``
    - opposite-spin terms use the Coulomb term ``(pq|rs)``

    This avoids storing a full ``2 x 2`` block tensor when the direct solver
    only needs the spatial integrals plus the spin-resolved occupation patterns.
    """
    h1, energy_core = h1e_for_cas(mf, ncas=ncas, ncore=ncore, mo_coeff=mo_coeff)
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    eri_spatial = transform_spatial_eri_to_mo(
        mf,
        mo_cas,
        mo_cas,
        mo_cas,
        mo_cas,
        use_cholesky=use_cholesky,
        eri_factors=_get_mf_cholesky_factors(mf) if use_cholesky else None,
    )
    return h1, eri_spatial, energy_core


def transform_active_space_pair_factors(mf, mo_coeff, ncas, ncore, eri_factors=None):
    """
    Build active-space MO-pair Cholesky factors for the direct-CI solver.
    """
    h1, energy_core = h1e_for_cas(mf, ncas=ncas, ncore=ncore, mo_coeff=mo_coeff)
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    if eri_factors is None:
        eri_factors = _get_mf_cholesky_factors(mf)
    pair_factors = transform_eri_factors_to_mo_pair(eri_factors, mo_cas, mo_cas)
    return h1, pair_factors, energy_core


def _binary_row_to_bits(occ):
    bits = 0
    for p, val in enumerate(occ):
        if val:
            bits |= 1 << p
    return bits


def _empty_int_array():
    return np.zeros(0, dtype=np.int32)


def _empty_phase_array():
    return np.zeros(0, dtype=np.int8)


def _extract_single_data(I, J, Binary, sign, spin):
    if len(I) == 0:
        return _empty_int_array(), _empty_int_array(), _empty_phase_array()

    a_t, a = get_excitation_op(I, J, Binary, sign, spin=spin)
    p = np.argmax(a_t != 0, axis=1).astype(np.int32)
    q = np.argmax(a != 0, axis=1).astype(np.int32)
    phase = (a_t[np.arange(len(I)), p] * a[np.arange(len(I)), q]).astype(np.int8)
    return p, q, phase


def _extract_double_same_data(I, J, Binary, sign, spin):
    if len(I) == 0:
        return (_empty_int_array(), _empty_int_array(), _empty_int_array(),
                _empty_int_array(), _empty_phase_array())

    a_t, a = get_excitation_op(I, J, Binary, sign, spin=spin)
    p = np.argmax(a_t[0] != 0, axis=1).astype(np.int32)
    q = np.argmax(a[0] != 0, axis=1).astype(np.int32)
    r = np.argmax(a_t[1] != 0, axis=1).astype(np.int32)
    s = np.argmax(a[1] != 0, axis=1).astype(np.int32)
    phase = (
        a_t[0, np.arange(len(I)), p]
        * a[0, np.arange(len(I)), q]
        * a_t[1, np.arange(len(I)), r]
        * a[1, np.arange(len(I)), s]
    ).astype(np.int8)
    return p, q, r, s, phase


def _extract_double_ab_data(I, J, Binary, sign):
    if len(I) == 0:
        return (_empty_int_array(), _empty_int_array(), _empty_int_array(),
                _empty_int_array(), _empty_phase_array())

    ab_t, ab = get_excitation_op(I, J, Binary, sign, spin=0)
    ba_t, ba = get_excitation_op(I, J, Binary, sign, spin=1)
    p = np.argmax(ab_t != 0, axis=1).astype(np.int32)
    q = np.argmax(ab != 0, axis=1).astype(np.int32)
    r = np.argmax(ba_t != 0, axis=1).astype(np.int32)
    s = np.argmax(ba != 0, axis=1).astype(np.int32)
    phase = (
        ab_t[np.arange(len(I)), p]
        * ab[np.arange(len(I)), q]
        * ba_t[np.arange(len(I)), r]
        * ba[np.arange(len(I)), s]
    ).astype(np.int8)
    return p, q, r, s, phase


def build_direct_connectivity(Binary):
    """
    Enumerate only the determinant pairs that are connected by the Hamiltonian.

    The legacy Slater-Condon builder compares every determinant to every other
    determinant. For the direct-CI solver we only need connected neighbors, so
    we enumerate those directly from each determinant's occupied/virtual orbital
    lists and store compact orbital-index records.
    """
    n_det, _, n_mo = Binary.shape
    alpha_bits = [_binary_row_to_bits(Binary[i, 0]) for i in range(n_det)]
    beta_bits = [_binary_row_to_bits(Binary[i, 1]) for i in range(n_det)]
    lookup = {(alpha_bits[i], beta_bits[i]): i for i in range(n_det)}

    singles_a = []
    singles_b = []
    doubles_aa = []
    doubles_bb = []
    doubles_ab = []

    for J in range(n_det):
        occ_a = np.where(Binary[J, 0] == 1)[0]
        vir_a = np.where(Binary[J, 0] == 0)[0]
        occ_b = np.where(Binary[J, 1] == 1)[0]
        vir_b = np.where(Binary[J, 1] == 0)[0]

        a_bits = alpha_bits[J]
        b_bits = beta_bits[J]

        for q in occ_a:
            removed = a_bits ^ (1 << q)
            for p in vir_a:
                I = lookup[(removed | (1 << p), b_bits)]
                singles_a.append((I, J))

        for q in occ_b:
            removed = b_bits ^ (1 << q)
            for p in vir_b:
                I = lookup[(a_bits, removed | (1 << p))]
                singles_b.append((I, J))

        for iq, q in enumerate(occ_a):
            for is_, s in enumerate(occ_a[iq + 1:], start=iq + 1):
                removed = a_bits ^ (1 << q) ^ (1 << s)
                for ip, p in enumerate(vir_a):
                    for ir, r in enumerate(vir_a[ip + 1:], start=ip + 1):
                        I = lookup[(removed | (1 << p) | (1 << r), b_bits)]
                        doubles_aa.append((I, J))

        for iq, q in enumerate(occ_b):
            for is_, s in enumerate(occ_b[iq + 1:], start=iq + 1):
                removed = b_bits ^ (1 << q) ^ (1 << s)
                for ip, p in enumerate(vir_b):
                    for ir, r in enumerate(vir_b[ip + 1:], start=ip + 1):
                        I = lookup[(a_bits, removed | (1 << p) | (1 << r))]
                        doubles_bb.append((I, J))

        for q in occ_a:
            a_removed = a_bits ^ (1 << q)
            for s in occ_b:
                b_removed = b_bits ^ (1 << s)
                for p in vir_a:
                    for r in vir_b:
                        I = lookup[(a_removed | (1 << p), b_removed | (1 << r))]
                        doubles_ab.append((I, J))

    sign = determinantsign(Binary)

    I_A = np.asarray([x[0] for x in singles_a], dtype=np.int32) if singles_a else _empty_int_array()
    J_A = np.asarray([x[1] for x in singles_a], dtype=np.int32) if singles_a else _empty_int_array()
    I_B = np.asarray([x[0] for x in singles_b], dtype=np.int32) if singles_b else _empty_int_array()
    J_B = np.asarray([x[1] for x in singles_b], dtype=np.int32) if singles_b else _empty_int_array()
    I_AA = np.asarray([x[0] for x in doubles_aa], dtype=np.int32) if doubles_aa else _empty_int_array()
    J_AA = np.asarray([x[1] for x in doubles_aa], dtype=np.int32) if doubles_aa else _empty_int_array()
    I_BB = np.asarray([x[0] for x in doubles_bb], dtype=np.int32) if doubles_bb else _empty_int_array()
    J_BB = np.asarray([x[1] for x in doubles_bb], dtype=np.int32) if doubles_bb else _empty_int_array()
    I_AB = np.asarray([x[0] for x in doubles_ab], dtype=np.int32) if doubles_ab else _empty_int_array()
    J_AB = np.asarray([x[1] for x in doubles_ab], dtype=np.int32) if doubles_ab else _empty_int_array()

    p_A, q_A, phase_A = _extract_single_data(I_A, J_A, Binary, sign, spin=0)
    p_B, q_B, phase_B = _extract_single_data(I_B, J_B, Binary, sign, spin=1)
    p_AA, q_AA, r_AA, s_AA, phase_AA = _extract_double_same_data(I_AA, J_AA, Binary, sign, spin=0)
    p_BB, q_BB, r_BB, s_BB, phase_BB = _extract_double_same_data(I_BB, J_BB, Binary, sign, spin=1)
    p_AB, q_AB, r_AB, s_AB, phase_AB = _extract_double_ab_data(I_AB, J_AB, Binary, sign)

    return DirectConnectivity(
        I_A, J_A, p_A, q_A, phase_A,
        I_B, J_B, p_B, q_B, phase_B,
        I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
        I_AB, J_AB, p_AB, q_AB, r_AB, s_AB, phase_AB,
    )

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
    """
    Evaluate all same-spin single-excitation Hamiltonian matrix elements.

    Each row ``k`` in ``a_t`` / ``a`` describes one excitation pair ``I <- J`` from
    the Slater-Condon tables. The returned vector contains the scalar matrix
    element for each pair, ready to be scattered into the CI sigma vector.
    """

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
    """
    Evaluate double-excitation matrix elements for a batch of determinant pairs.

    ``at1/a1`` and ``at2/a2`` encode the two creation/annihilation operators
    needed to connect a bra/ket determinant pair by a double excitation.
    """

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


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_spatial(h1, eri_spatial, Binary):
    """
    Diagonal CI matrix elements using spatial integrals plus spin occupations.

    Same-spin electron pairs use the antisymmetrized spatial combination
    ``(pp|qq) - (pq|qp)``, while opposite-spin pairs use the Coulomb term
    ``(pp|qq)``.
    """
    n_dets, _, n_mo = Binary.shape
    h1_diag = np.diag(h1)
    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += h1_diag[p]
            if Binary[i, 1, p]:
                H_diag[i] += h1_diag[p]

        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * (eri_spatial[p, p, q, q] - eri_spatial[p, q, q, p])
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * eri_spatial[p, p, q, q]

            if Binary[i, 1, p]:
                for q in range(n_mo):
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * (eri_spatial[p, p, q, q] - eri_spatial[p, q, q, p])
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * eri_spatial[p, p, q, q]

    return H_diag


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_compact(h1, eri_same, eri_cross, Binary):
    """
    Diagonal CI matrix elements using a compact two-tensor representation.

    ``eri_same`` stores the antisymmetrized same-spin tensor, while
    ``eri_cross`` stores the Coulomb tensor for opposite-spin electron pairs.
    This is the direct-CI representation used by the compact backend.
    """
    n_dets, _, n_mo = Binary.shape
    h1_diag = np.diag(h1)
    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += h1_diag[p]
            if Binary[i, 1, p]:
                H_diag[i] += h1_diag[p]

        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * eri_same[p, p, q, q]
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * eri_cross[p, p, q, q]

            if Binary[i, 1, p]:
                for q in range(n_mo):
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * eri_same[p, p, q, q]
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * eri_cross[p, p, q, q]

    return H_diag


@njit(nogil=True, cache=True, fastmath=True)
def _factor_coulomb(pair_factors, p, q, r, s):
    val = 0.0
    for t in range(pair_factors.shape[0]):
        val += pair_factors[t, p, q] * pair_factors[t, r, s]
    return val


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_compact_factors(h1, pair_factors, Binary):
    """
    Diagonal CI matrix elements using MO-pair factors directly.
    """
    n_dets, _, n_mo = Binary.shape
    h1_diag = np.diag(h1)
    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += h1_diag[p]
            if Binary[i, 1, p]:
                H_diag[i] += h1_diag[p]

        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    coul = _factor_coulomb(pair_factors, p, p, q, q)
                    if Binary[i, 0, q]:
                        exch = _factor_coulomb(pair_factors, p, q, q, p)
                        H_diag[i] += 0.5 * (coul - exch)
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * coul

            if Binary[i, 1, p]:
                for q in range(n_mo):
                    coul = _factor_coulomb(pair_factors, p, p, q, q)
                    if Binary[i, 1, q]:
                        exch = _factor_coulomb(pair_factors, p, q, q, p)
                        H_diag[i] += 0.5 * (coul - exch)
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * coul

    return H_diag



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


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_single_excitation(
    sigma_vec, c, I, J, H1_spin, H2_same, H2_cross, a_t, a, ca, binary_complement
):
    """
    Add all single-excitation contributions directly into ``sigma_vec``.

    This is the matrix-free version of ``_compute_single_excitation``: instead of
    forming a temporary vector of matrix elements and scattering it later, we
    contract the Hamiltonian contribution and immediately accumulate
    ``H[I, J] * c[J]`` into the output sigma vector.
    """
    n_exc = a_t.shape[0]
    n_mo = H1_spin.shape[0]

    for k in range(n_exc):
        val = 0.0
        for p in range(n_mo):
            a_t_val = a_t[k, p]
            if a_t_val == 0:
                continue
            for q in range(n_mo):
                a_val = a[k, q]
                if a_val == 0:
                    continue

                val -= H1_spin[p, q] * a_t_val * a_val
                for r in range(n_mo):
                    ca_val = ca[k, r]
                    if ca_val != 0:
                        val -= H2_same[p, q, r, r] * a_t_val * a_val * ca_val

                    comp_val = binary_complement[k, r]
                    if comp_val != 0:
                        val -= H2_cross[p, q, r, r] * a_t_val * a_val * comp_val

        sigma_vec[I[k]] += val * c[J[k]]


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_single_excitation_spatial(
    sigma_vec, c, I, J, h1, eri_spatial, a_t, a, ca, binary_complement
):
    """
    Add single-excitation contributions using spatial integrals only.

    The same-spin part uses the antisymmetrized contribution
    ``(pq|rr) - (pr|rq)``, while the opposite-spin part uses only the Coulomb
    term ``(pq|rr)``.
    """
    n_exc = a_t.shape[0]
    n_mo = h1.shape[0]

    for k in range(n_exc):
        val = 0.0
        for p in range(n_mo):
            a_t_val = a_t[k, p]
            if a_t_val == 0:
                continue
            for q in range(n_mo):
                a_val = a[k, q]
                if a_val == 0:
                    continue

                val -= h1[p, q] * a_t_val * a_val

                for r in range(n_mo):
                    ca_val = ca[k, r]
                    if ca_val != 0:
                        val -= (eri_spatial[p, q, r, r] - eri_spatial[p, r, r, q]) * a_t_val * a_val * ca_val

                    comp_val = binary_complement[k, r]
                    if comp_val != 0:
                        val -= eri_spatial[p, q, r, r] * a_t_val * a_val * comp_val

        sigma_vec[I[k]] += val * c[J[k]]


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_double_excitation(sigma_vec, c, I, J, H2_tensor, at1, a1, at2, a2):
    """
    Add all double-excitation contributions directly into ``sigma_vec``.

    This avoids allocating a temporary ``H_AA/H_BB/H_AB`` vector during each
    Lanczos matvec.
    """
    n_exc = at1.shape[0]
    n_mo = H2_tensor.shape[0]

    for k in range(n_exc):
        val = 0.0
        for p in range(n_mo):
            at1_val = at1[k, p]
            if at1_val == 0:
                continue
            for q in range(n_mo):
                a1_val = a1[k, q]
                if a1_val == 0:
                    continue
                for r in range(n_mo):
                    at2_val = at2[k, r]
                    if at2_val == 0:
                        continue
                    for s in range(n_mo):
                        a2_val = a2[k, s]
                        if a2_val == 0:
                            continue
                        val += H2_tensor[p, q, r, s] * at1_val * a1_val * at2_val * a2_val

        sigma_vec[I[k]] += val * c[J[k]]


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_double_excitation_spatial(
    sigma_vec, c, I, J, eri_spatial, at1, a1, at2, a2, antisymmetrize
):
    """
    Add double-excitation contributions from a spatial ERI tensor.

    ``antisymmetrize=True`` is used for same-spin double excitations, giving the
    usual ``(pq|rs) - (ps|rq)`` matrix element. For opposite-spin doubles we use
    the Coulomb term ``(pq|rs)`` directly.
    """
    n_exc = at1.shape[0]
    n_mo = eri_spatial.shape[0]

    for k in range(n_exc):
        val = 0.0
        for p in range(n_mo):
            at1_val = at1[k, p]
            if at1_val == 0:
                continue
            for q in range(n_mo):
                a1_val = a1[k, q]
                if a1_val == 0:
                    continue
                for r in range(n_mo):
                    at2_val = at2[k, r]
                    if at2_val == 0:
                        continue
                    for s in range(n_mo):
                        a2_val = a2[k, s]
                        if a2_val == 0:
                            continue
                        contrib = eri_spatial[p, q, r, s]
                        if antisymmetrize:
                            contrib -= eri_spatial[p, s, r, q]
                        val += contrib * at1_val * a1_val * at2_val * a2_val

        sigma_vec[I[k]] += val * c[J[k]]


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_on_the_fly_numba(
    H1, H2, H_diag, c,
    I_A, J_A, a_t, a, ca, binary_I_A_complement,
    I_B, J_B, b_t, b, cb, binary_I_B_complement,
    I_AA, J_AA, aa_t1, aa1, aa_t2, aa2,
    I_BB, J_BB, bb_t1, bb1, bb_t2, bb2,
    I_AB, J_AB, ab_t, ab, ba_t, ba,
):
    """
    Fully compiled sigma-vector builder for the direct-CI backend.

    The diagonal part is applied first. Singles and doubles are then accumulated
    directly from the Slater-Condon connectivity tables without ever forming the
    full CI Hamiltonian matrix.
    """
    sigma_vec = H_diag * c

    if len(I_A) > 0:
        _accumulate_single_excitation(
            sigma_vec, c, I_A, J_A,
            H1[0], H2[0, 0], H2[0, 1],
            a_t, a, ca, binary_I_A_complement
        )

    if len(I_B) > 0:
        _accumulate_single_excitation(
            sigma_vec, c, I_B, J_B,
            H1[1], H2[1, 1], H2[1, 0],
            b_t, b, cb, binary_I_B_complement
        )

    if len(I_AA) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_AA, J_AA,
            H2[0, 0], aa_t1, aa1, aa_t2, aa2
        )

    if len(I_BB) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_BB, J_BB,
            H2[1, 1], bb_t1, bb1, bb_t2, bb2
        )

    if len(I_AB) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_AB, J_AB,
            H2[0, 1], ab_t, ab, ba_t, ba
        )

    return sigma_vec


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_spatial_on_the_fly_numba(
    h1, eri_spatial, H_diag, c,
    I_A, J_A, a_t, a, ca, binary_I_A_complement,
    I_B, J_B, b_t, b, cb, binary_I_B_complement,
    I_AA, J_AA, aa_t1, aa1, aa_t2, aa2,
    I_BB, J_BB, bb_t1, bb1, bb_t2, bb2,
    I_AB, J_AB, ab_t, ab, ba_t, ba,
):
    """
    Matrix-free sigma-vector build using one spatial ERI tensor.

    This is the same direct-CI algorithm as ``_sigma_on_the_fly_numba``, but it
    derives the different spin cases from a single spatial integral tensor
    instead of receiving pre-expanded spin blocks.
    """
    sigma_vec = H_diag * c

    if len(I_A) > 0:
        _accumulate_single_excitation_spatial(
            sigma_vec, c, I_A, J_A, h1, eri_spatial, a_t, a, ca, binary_I_A_complement
        )

    if len(I_B) > 0:
        _accumulate_single_excitation_spatial(
            sigma_vec, c, I_B, J_B, h1, eri_spatial, b_t, b, cb, binary_I_B_complement
        )

    if len(I_AA) > 0:
        _accumulate_double_excitation_spatial(
            sigma_vec, c, I_AA, J_AA, eri_spatial, aa_t1, aa1, aa_t2, aa2, True
        )

    if len(I_BB) > 0:
        _accumulate_double_excitation_spatial(
            sigma_vec, c, I_BB, J_BB, eri_spatial, bb_t1, bb1, bb_t2, bb2, True
        )

    if len(I_AB) > 0:
        _accumulate_double_excitation_spatial(
            sigma_vec, c, I_AB, J_AB, eri_spatial, ab_t, ab, ba_t, ba, False
        )

    return sigma_vec


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_compact_on_the_fly_numba(
    h1, eri_same, eri_cross, H_diag, c,
    I_A, J_A, a_t, a, ca, binary_I_A_complement,
    I_B, J_B, b_t, b, cb, binary_I_B_complement,
    I_AA, J_AA, aa_t1, aa1, aa_t2, aa2,
    I_BB, J_BB, bb_t1, bb1, bb_t2, bb2,
    I_AB, J_AB, ab_t, ab, ba_t, ba,
):
    """
    Matrix-free sigma build with one same-spin tensor and one cross-spin tensor.

    This keeps only the tensors the direct-CI kernel actually needs:
    ``eri_same = (pq|rs) - (ps|rq)`` and ``eri_cross = (pq|rs)``.
    Relative to the pure spatial backend, this avoids doing the exchange
    subtraction inside every excitation loop.
    """
    sigma_vec = H_diag * c

    if len(I_A) > 0:
        _accumulate_single_excitation(
            sigma_vec, c, I_A, J_A, h1, eri_same, eri_cross, a_t, a, ca, binary_I_A_complement
        )

    if len(I_B) > 0:
        _accumulate_single_excitation(
            sigma_vec, c, I_B, J_B, h1, eri_same, eri_cross, b_t, b, cb, binary_I_B_complement
        )

    if len(I_AA) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_AA, J_AA, eri_same, aa_t1, aa1, aa_t2, aa2
        )

    if len(I_BB) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_BB, J_BB, eri_same, bb_t1, bb1, bb_t2, bb2
        )

    if len(I_AB) > 0:
        _accumulate_double_excitation(
            sigma_vec, c, I_AB, J_AB, eri_cross, ab_t, ab, ba_t, ba
        )

    return sigma_vec


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_single_from_conn(
    sigma_vec, c, I, J, p_idx, q_idx, phase, h1, eri_same, eri_cross, Binary, spin
):
    n_exc = len(I)
    n_mo = h1.shape[0]

    for k in range(n_exc):
        p = p_idx[k]
        q = q_idx[k]
        sign = phase[k]
        j = J[k]

        val = -sign * h1[p, q]
        for r in range(n_mo):
            if Binary[j, spin, r] and r != q:
                val -= sign * eri_same[p, q, r, r]
            if Binary[j, 1 - spin, r]:
                val -= sign * eri_cross[p, q, r, r]

        sigma_vec[I[k]] += val * c[j]


@njit(nogil=True, cache=True, fastmath=True)
def _accumulate_double_from_conn(
    sigma_vec, c, I, J, p_idx, q_idx, r_idx, s_idx, phase, eri_tensor
):
    n_exc = len(I)
    for k in range(n_exc):
        sigma_vec[I[k]] += phase[k] * eri_tensor[p_idx[k], q_idx[k], r_idx[k], s_idx[k]] * c[J[k]]


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_compact_conn_numba(
    h1, eri_same, eri_cross, H_diag, c, Binary,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    I_AB, J_AB, p_AB, q_AB, r_AB, s_AB, phase_AB,
):
    """
    Matrix-free sigma build from compact determinant connectivity lists.

    This is the setup-light direct-CI backend: determinant neighbors are
    enumerated once in Python, then the compiled matvec only walks the compact
    connection records.
    """
    sigma_vec = H_diag * c
    _accumulate_single_from_conn(sigma_vec, c, I_A, J_A, p_A, q_A, phase_A, h1, eri_same, eri_cross, Binary, 0)
    _accumulate_single_from_conn(sigma_vec, c, I_B, J_B, p_B, q_B, phase_B, h1, eri_same, eri_cross, Binary, 1)
    _accumulate_double_from_conn(sigma_vec, c, I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA, eri_same)
    _accumulate_double_from_conn(sigma_vec, c, I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB, eri_same)
    _accumulate_double_from_conn(sigma_vec, c, I_AB, J_AB, p_AB, q_AB, r_AB, s_AB, phase_AB, eri_cross)
    return sigma_vec


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_single_values_from_factors(J, p_idx, q_idx, phase, h1, pair_factors, Binary, spin):
    n_exc = len(J)
    n_mo = h1.shape[0]
    values = np.zeros(n_exc)

    for k in prange(n_exc):
        p = p_idx[k]
        q = q_idx[k]
        sign = phase[k]
        j = J[k]

        val = -sign * h1[p, q]
        for r in range(n_mo):
            coul = _factor_coulomb(pair_factors, p, q, r, r)
            if Binary[j, spin, r] and r != q:
                exch = _factor_coulomb(pair_factors, p, r, r, q)
                val -= sign * (coul - exch)
            if Binary[j, 1 - spin, r]:
                val -= sign * coul

        values[k] = val

    return values


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_double_same_values_from_factors(p_idx, q_idx, r_idx, s_idx, phase, pair_factors):
    n_exc = len(p_idx)
    values = np.zeros(n_exc)

    for k in prange(n_exc):
        values[k] = phase[k] * (
            _factor_coulomb(pair_factors, p_idx[k], q_idx[k], r_idx[k], s_idx[k]) -
            _factor_coulomb(pair_factors, p_idx[k], s_idx[k], r_idx[k], q_idx[k])
        )

    return values


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_double_cross_values_from_factors(p_idx, q_idx, r_idx, s_idx, phase, pair_factors):
    n_exc = len(p_idx)
    values = np.zeros(n_exc)

    for k in prange(n_exc):
        values[k] = phase[k] * _factor_coulomb(
            pair_factors, p_idx[k], q_idx[k], r_idx[k], s_idx[k]
        )

    return values


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_values_conn_numba(
    H_diag, H_A, H_B, H_AA, H_BB, H_AB, c,
    I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
):
    sigma_vec = H_diag * c

    for k in range(len(I_A)):
        sigma_vec[I_A[k]] += H_A[k] * c[J_A[k]]
    for k in range(len(I_B)):
        sigma_vec[I_B[k]] += H_B[k] * c[J_B[k]]
    for k in range(len(I_AA)):
        sigma_vec[I_AA[k]] += H_AA[k] * c[J_AA[k]]
    for k in range(len(I_BB)):
        sigma_vec[I_BB[k]] += H_BB[k] * c[J_BB[k]]
    for k in range(len(I_AB)):
        sigma_vec[I_AB[k]] += H_AB[k] * c[J_AB[k]]

    return sigma_vec


def sigma_on_the_fly(Binary, SC1, SC2, H1, H2, H_diag, c):
    """
    Matrix-free CI sigma-vector build without precomputing all excitation values.

    This keeps the Slater-Condon connectivity tables but evaluates the matrix
    elements inside each matvec, which avoids the large eager setup cost of the
    original direct_ci path.
    """
    I_A, J_A, a_t, a, I_B, J_B, b_t, b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    if len(I_A) > 0:
        binary_I_A_complement = Binary[I_A, 1]
    else:
        binary_I_A_complement = np.zeros((0, Binary.shape[2]), dtype=Binary.dtype)

    if len(I_B) > 0:
        binary_I_B_complement = Binary[I_B, 0]
    else:
        binary_I_B_complement = np.zeros((0, Binary.shape[2]), dtype=Binary.dtype)

    if getattr(aa_t, "ndim", 2) == 3:
        aa_t1, aa1 = aa_t[0], aa[0]
        aa_t2, aa2 = aa_t[1], aa[1]
    else:
        aa_t1 = aa_t2 = aa_t
        aa1 = aa2 = aa

    if getattr(bb_t, "ndim", 2) == 3:
        bb_t1, bb1 = bb_t[0], bb[0]
        bb_t2, bb2 = bb_t[1], bb[1]
    else:
        bb_t1 = bb_t2 = bb_t
        bb1 = bb2 = bb

    return _sigma_on_the_fly_numba(
        H1, H2, H_diag, c,
        I_A, J_A, a_t, a, ca, binary_I_A_complement,
        I_B, J_B, b_t, b, cb, binary_I_B_complement,
        I_AA, J_AA, aa_t1, aa1, aa_t2, aa2,
        I_BB, J_BB, bb_t1, bb1, bb_t2, bb2,
        I_AB, J_AB, ab_t, ab, ba_t, ba,
    )


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
        super().__init__(mf, ncas, nelecas, ncore=ncore, spin=spin)
        self.direct_connectivity = None
        
        self.tol = tol
        self.direct_ci_dense_fallback_ndets = DIRECT_CI_DENSE_FALLBACK_NDETS
        self.solver_backend = None
        self.direct_ci_eigensolver = 'davidson'
        self.direct_ci_max_cycle = 100
        self.direct_ci_max_subspace = None
        self.direct_ci_reuse_guess = True
        self._s2_operator = None
        self._s2_diag = None
        self._direct_spatial_h1 = None
        self._direct_spatial_eri = None
        self._direct_pair_factors = None
        self._direct_same_spin_eri = None
        self._direct_cross_spin_eri = None
        self._direct_factor_H_diag = None
        self._direct_factor_H_A = None
        self._direct_factor_H_B = None
        self._direct_factor_H_AA = None
        self._direct_factor_H_BB = None
        self._direct_factor_H_AB = None
        self._direct_integrals_mo_ref = None
        self._direct_integrals_ncore = None
        self._direct_integrals_ncas = None
        self._direct_integrals_use_cholesky = None


    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None, use_cholesky=None):
        """
        Build the active-space Hamiltonian in spin-orbital block form.

        The direct-CI code uses a spin-block representation instead of a single
        dense spin-orbital tensor:

        ``H1 = [h_alpha, h_beta]``
        ``H2 = [[eri_aa, eri_ab], [eri_ba, eri_bb]]``

        For the current CASCI path we use the same spatial active orbitals for
        alpha and beta channels. In the restricted case this lets us reuse one
        AO->MO ERI transformation for all four spin blocks before the
        same-spin antisymmetrization step in ``run()``.

        Parameters
        ----------
        spin_flip : bool
            spin-flip

        Returns
        -------
        H1 : list
            ``[h1e_alpha, h1e_beta]`` in the active MO basis.
        H2 : ndarray
            Spin-block ERIs with shape ``(2, 2, ncas, ncas, ncas, ncas)``.
        """
        # from pyscf import ao2mo

        mf = self.mf
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(mf, use_cholesky)

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

        same_spin_orbitals = Ca is Cb or np.array_equal(Ca, Cb)
        eri_factors = _get_mf_cholesky_factors(mf) if use_cholesky else None

        if same_spin_orbitals:
            # Restricted references use the same spatial active orbitals for both
            # spin channels, so all spin blocks start from the same spatial ERI.
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
            ### compute SO ERIs (MO)
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

    def get_direct_spatial_integrals(self, use_cholesky=None):
        """
        Return cached active-space spatial integrals for the direct-CI backend.

        The matrix-free solver may be called repeatedly on the same CASCI
        object, for example when benchmarking or requesting multiple roots after
        changing solver options. Reusing the transformed active-space integrals
        avoids paying the AO->MO contraction cost every time.
        """
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        if (
            self._direct_spatial_h1 is not None
            and self._direct_spatial_eri is not None
            and self._direct_integrals_mo_ref is self.mo_coeff
            and self._direct_integrals_ncore == self.ncore
            and self._direct_integrals_ncas == self.ncas
            and self._direct_integrals_use_cholesky == bool(use_cholesky)
        ):
            return self._direct_spatial_h1, self._direct_spatial_eri, self.e_core

        h1, eri_spatial, energy_core = transform_active_space_spatial_integrals(
            self.mf, self.mo_coeff, self.ncas, self.ncore, use_cholesky=use_cholesky
        )
        self._direct_spatial_h1 = h1
        self._direct_spatial_eri = eri_spatial
        self._direct_integrals_mo_ref = self.mo_coeff
        self._direct_integrals_ncore = self.ncore
        self._direct_integrals_ncas = self.ncas
        self._direct_integrals_use_cholesky = bool(use_cholesky)
        self.e_core = energy_core
        return h1, eri_spatial, energy_core

    def get_direct_pair_factors(self, use_cholesky=None):
        """
        Return cached active-space MO-pair factors for the direct-CI backend.
        """
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        if not use_cholesky:
            return None, None, self.e_core

        if (
            self._direct_spatial_h1 is not None
            and self._direct_pair_factors is not None
            and self._direct_integrals_mo_ref is self.mo_coeff
            and self._direct_integrals_ncore == self.ncore
            and self._direct_integrals_ncas == self.ncas
            and self._direct_integrals_use_cholesky == bool(use_cholesky)
        ):
            return self._direct_spatial_h1, self._direct_pair_factors, self.e_core

        h1, pair_factors, energy_core = transform_active_space_pair_factors(
            self.mf, self.mo_coeff, self.ncas, self.ncore
        )
        self._direct_spatial_h1 = h1
        self._direct_pair_factors = pair_factors
        self._direct_integrals_mo_ref = self.mo_coeff
        self._direct_integrals_ncore = self.ncore
        self._direct_integrals_ncas = self.ncas
        self._direct_integrals_use_cholesky = bool(use_cholesky)
        self.e_core = energy_core
        return h1, pair_factors, energy_core

    def get_direct_compact_integrals(self, use_cholesky=None):
        """
        Return the compact direct-CI two-electron representation.

        The direct solver only needs two versions of the active-space ERIs:
        the antisymmetrized same-spin tensor and the Coulomb cross-spin tensor.
        Caching them keeps the direct-CI setup light across repeated runs.
        """
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        h1, eri_spatial, energy_core = self.get_direct_spatial_integrals(use_cholesky=use_cholesky)
        if (
            self._direct_same_spin_eri is None
            or self._direct_cross_spin_eri is None
            or self._direct_integrals_mo_ref is not self.mo_coeff
            or self._direct_integrals_ncore != self.ncore
            or self._direct_integrals_ncas != self.ncas
            or self._direct_integrals_use_cholesky != bool(use_cholesky)
        ):
            self._direct_cross_spin_eri = eri_spatial
            self._direct_same_spin_eri = eri_spatial - eri_spatial.swapaxes(1, 3)

        return h1, self._direct_same_spin_eri, self._direct_cross_spin_eri, energy_core

    def get_direct_factor_hamiltonian(self, binary, use_cholesky=None):
        """
        Precompute direct-CI Hamiltonian connection values from MO-pair factors.
        """
        if use_cholesky is None:
            use_cholesky = self.use_cholesky_integrals
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        if not use_cholesky:
            return None

        h1, pair_factors, energy_core = self.get_direct_pair_factors(use_cholesky=use_cholesky)
        if pair_factors is None:
            return None

        if self.direct_connectivity is None:
            self.direct_connectivity = build_direct_connectivity(binary)
        conn = self.direct_connectivity

        cache_valid = (
            self._direct_factor_H_diag is not None
            and self._direct_integrals_mo_ref is self.mo_coeff
            and self._direct_integrals_ncore == self.ncore
            and self._direct_integrals_ncas == self.ncas
            and self._direct_integrals_use_cholesky == bool(use_cholesky)
        )
        if cache_valid:
            return (
                h1,
                pair_factors,
                self._direct_factor_H_diag,
                self._direct_factor_H_A,
                self._direct_factor_H_B,
                self._direct_factor_H_AA,
                self._direct_factor_H_BB,
                self._direct_factor_H_AB,
                energy_core,
            )

        self._direct_factor_H_diag = _compute_diag_compact_factors(h1, pair_factors, binary)
        self._direct_factor_H_A = _compute_single_values_from_factors(
            conn.J_A, conn.p_A, conn.q_A, conn.phase_A, h1, pair_factors, binary, 0
        )
        self._direct_factor_H_B = _compute_single_values_from_factors(
            conn.J_B, conn.p_B, conn.q_B, conn.phase_B, h1, pair_factors, binary, 1
        )
        self._direct_factor_H_AA = _compute_double_same_values_from_factors(
            conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA, pair_factors
        )
        self._direct_factor_H_BB = _compute_double_same_values_from_factors(
            conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB, pair_factors
        )
        self._direct_factor_H_AB = _compute_double_cross_values_from_factors(
            conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB, pair_factors
        )

        return (
            h1,
            pair_factors,
            self._direct_factor_H_diag,
            self._direct_factor_H_A,
            self._direct_factor_H_B,
            self._direct_factor_H_AA,
            self._direct_factor_H_BB,
            self._direct_factor_H_AB,
            energy_core,
        )

    def ensure_slater_condon_cache(self):
        """
        Lazily build the legacy Slater-Condon tables when helper routines need them.

        ``direct_ci`` itself no longer relies on ``SC1/SC2`` for setup or its
        fast matvec path, but overlap/RDM/TDM helpers still use the original
        representations. Keeping this lazy preserves those APIs without paying
        the all-pairs setup cost during every direct-CI solve.
        """
        if self.binary is None:
            raise ValueError('Build the determinant basis before requesting Slater-Condon tables.')
        if self.SC1 is None or self.SC2 is None:
            self.SC1, self.SC2 = SlaterCondon(self.binary)
        return self.SC1, self.SC2


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
            DESCRIPTION. The default is 0.
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


    def run(self, nstates=1, mo_coeff=None, method='direct_ci', ci0=None, use_cholesky=False):
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
        self.use_cholesky_integrals = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        use_cholesky = self.use_cholesky_integrals

        if method == 'ci':
            self.solver_backend = 'ci'

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

            mo_occ = mcscf.casci._reference_active_occupations(self.nelecas_spin, ncas)
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary


            # print('Number of determinants', binary.shape[0])

            h1e, h2e = self.get_SO_matrix(use_cholesky=use_cholesky)

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

            if h2e is not None:
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
                mo_occ = mcscf.casci._reference_active_occupations(self.nelecas_spin, ncas)
                binary = get_fci_combos(mo_occ = mo_occ)
                self.binary = binary
                self.direct_connectivity = build_direct_connectivity(binary)

            else:
                binary = self.binary
                if self.direct_connectivity is None:
                    self.direct_connectivity = build_direct_connectivity(binary)


            factor_data = None
            if self.spin_purification:
                # The first-order spin-penalty code is currently expressed in the
                # older spin-block Hamiltonian form, so we keep that path until
                # the penalty is rewritten in terms of spatial integrals.
                h1e, h2e = self.get_SO_matrix(use_cholesky=use_cholesky)
                h1e = np.asarray(h1e)
                spatial_h1 = None
                spatial_eri = None
            else:
                factor_data = self.get_direct_factor_hamiltonian(binary, use_cholesky=use_cholesky)
                if factor_data is not None:
                    (
                        spatial_h1,
                        pair_factors,
                        H_diag_factor,
                        H_A_factor,
                        H_B_factor,
                        H_AA_factor,
                        H_BB_factor,
                        H_AB_factor,
                        energy_core,
                    ) = factor_data
                    same_spin_eri = None
                    cross_spin_eri = None
                    spatial_eri = None
                    h1e = np.asarray([spatial_h1, spatial_h1])
                    h2e = None
                else:
                    spatial_h1, same_spin_eri, cross_spin_eri, energy_core = self.get_direct_compact_integrals(
                        use_cholesky=use_cholesky
                    )
                    spatial_eri = cross_spin_eri
                    h1e = np.asarray([spatial_h1, spatial_h1])
                    h2e = None

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

            if h2e is not None:
                h2e[0,0] -= h2e[0,0].swapaxes(1,3)
                h2e[1,1] -= h2e[1,1].swapaxes(1,3)

            self.hcore = h1e
            self.h2e_cas = spatial_eri
            self.eri_so = h2e

            if (
                factor_data is None
                and
                self.direct_ci_dense_fallback_ndets is not None
                and self.direct_ci_dense_fallback_ndets > 0
                and binary.shape[0] <= self.direct_ci_dense_fallback_ndets
            ):
                # For small determinant spaces the dense/vectorized CI builder is
                # faster than paying the Numba JIT + setup overhead of direct_ci.
                self.solver_backend = 'ci_dense_fallback'
                if spatial_eri is not None:
                    # The dense CI builder still expects the explicit spin-block
                    # Hamiltonian, so create it only for the small-space fallback.
                    h2e = np.stack((np.stack((spatial_eri.copy(), spatial_eri.copy())),
                                    np.stack((spatial_eri.copy(), spatial_eri.copy()))))
                    h2e[0,0] -= h2e[0,0].swapaxes(1,3)
                    h2e[1,1] -= h2e[1,1].swapaxes(1,3)
                    self.eri_so = h2e

                SC1, SC2 = self.ensure_slater_condon_cache()
                H_CI = CI_H(binary, h1e, h2e, SC1, SC2)
                E, X = eigsh(H_CI, k=nstates, which='SA')
            else:
                if factor_data is not None:
                    self.solver_backend = 'direct_ci_factor_conn'
                else:
                    self.solver_backend = 'direct_ci_compact_conn' if spatial_eri is not None else 'direct_ci'
                # The diagonal is reused in every matvec, so it is worth
                # computing once up front even in the matrix-free solver.
                if factor_data is not None:
                    H_diag = H_diag_factor
                else:
                    H_diag = _compute_diag_compact(spatial_h1, same_spin_eri, cross_spin_eri, binary) if spatial_eri is not None else _compute_diag(h1e, h2e, binary)
                conn = self.direct_connectivity

                def mv(c):
                    # Keep the Lanczos matvec almost entirely inside compiled
                    # code. The Python closure only forwards cached arrays.
                    if factor_data is not None:
                        return _sigma_values_conn_numba(
                            H_diag,
                            H_A_factor,
                            H_B_factor,
                            H_AA_factor,
                            H_BB_factor,
                            H_AB_factor,
                            c,
                            conn.I_A, conn.J_A,
                            conn.I_B, conn.J_B,
                            conn.I_AA, conn.J_AA,
                            conn.I_BB, conn.J_BB,
                            conn.I_AB, conn.J_AB,
                        )
                    if spatial_eri is not None:
                        return _sigma_compact_conn_numba(
                            spatial_h1, same_spin_eri, cross_spin_eri, H_diag, c, binary,
                            conn.I_A, conn.J_A, conn.p_A, conn.q_A, conn.phase_A,
                            conn.I_B, conn.J_B, conn.p_B, conn.q_B, conn.phase_B,
                            conn.I_AA, conn.J_AA, conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA,
                            conn.I_BB, conn.J_BB, conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB,
                            conn.I_AB, conn.J_AB, conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB,
                        )
                    SC1, SC2 = self.ensure_slater_condon_cache()
                    I_A, J_A, a_t, a, I_B, J_B, b_t, b, ca, cb = SC1
                    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2
                    binary_I_A_complement = binary[I_A, 1] if len(I_A) > 0 else np.zeros((0, binary.shape[2]), dtype=binary.dtype)
                    binary_I_B_complement = binary[I_B, 0] if len(I_B) > 0 else np.zeros((0, binary.shape[2]), dtype=binary.dtype)
                    if getattr(aa_t, "ndim", 2) == 3:
                        aa_t1, aa1 = aa_t[0], aa[0]
                        aa_t2, aa2 = aa_t[1], aa[1]
                    else:
                        aa_t1 = aa_t2 = aa_t
                        aa1 = aa2 = aa
                    if getattr(bb_t, "ndim", 2) == 3:
                        bb_t1, bb1 = bb_t[0], bb[0]
                        bb_t2, bb2 = bb_t[1], bb[1]
                    else:
                        bb_t1 = bb_t2 = bb_t
                        bb1 = bb2 = bb
                    return _sigma_on_the_fly_numba(h1e, h2e, H_diag, c,
                        I_A, J_A, a_t, a, ca, binary_I_A_complement,
                        I_B, J_B, b_t, b, cb, binary_I_B_complement,
                        I_AA, J_AA, aa_t1, aa1, aa_t2, aa2,
                        I_BB, J_BB, bb_t1, bb1, bb_t2, bb2,
                        I_AB, J_AB, ab_t, ab, ba_t, ba)

                if self.direct_ci_eigensolver == 'davidson':
                    # Use a CI-specific Davidson iteration by default.  The
                    # direct-CI backend already has a cheap diagonal
                    # preconditioner and only needs a few low roots, which is
                    # exactly the regime where Davidson is preferable to a
                    # generic sparse eigensolver.
                    guess = _select_direct_ci_guess(
                        self,
                        nstates,
                        ci0=ci0 if ci0 is not None or not self.direct_ci_reuse_guess else None,
                    )
                    E, X = davidson_lowest(
                        mv,
                        H_diag,
                        nroots=nstates,
                        tol=self.tol if self.tol > 0 else 1e-8,
                        max_cycle=self.direct_ci_max_cycle,
                        max_subspace=self.direct_ci_max_subspace,
                        guess=guess,
                    )
                elif self.direct_ci_eigensolver == 'eigsh':
                    H = LinearOperator((binary.shape[0], binary.shape[0]), matvec=mv)
                    E, X = eigsh(H, k=nstates, which='SA', tol=self.tol)
                else:
                    raise ValueError(
                        "Unknown direct_ci eigensolver '{}'. Use 'davidson' or 'eigsh'.".format(
                            self.direct_ci_eigensolver
                        )
                    )
            



        elif method == 'jw':
            self.solver_backend = 'jw'


            # exact diagonalization by JW transform

            H = self.qubitization()
            E, X = eigsh(H, k=nstates, which='SA')

        else:
            raise ValueError("There is no {} solver for CASCI. Use 'ci' or 'jw'".format(method))

        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(nstates)]

        for i in range(nstates):
            ss = self.spin_square(i)
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

        self.ensure_slater_condon_cache()
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

        self.ensure_slater_condon_cache()
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
        self.ensure_slater_condon_cache()
        return mcscf.casci.make_rdm1s(ci, self.binary, self.SC1)

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
        self.ensure_slater_condon_cache()
        ci = self.ci[state_id]


        if with_core: # we probably never need this!

            ncore = self.ncore
            ncas = self.ncas
            # nmo = self.mf.nmo
            nmo = ncore + ncas

            if ncore == 0:
                return make_rdm2(ci, self.binary, self.SC1, self.SC2)

            D = np.zeros((nmo, nmo, nmo, nmo))

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
        self.ensure_slater_condon_cache()

        if h2e.ndim == 4: # spin-free operator
            h2e = np.einsum('IJ, pqrs -> IJpqrs', np.ones((2,2)), h2e)

        return contract_with_rdm2(self.ci[state_id], h2e, self.binary, self.SC1, self.SC2)



    def make_rdm12(self, state_id, with_core=False):
        dm1 = self.make_rdm1(state_id, with_core=with_core)
        dm2 = self.make_rdm2(state_id, with_core=with_core)

        return dm1, dm2

    def spin_square(self, state_id=0):
        """
        Evaluate ``<Psi|S^2|Psi>`` directly from the CI vector.

        This avoids building the full 1- and 2-RDMs, which can dominate the
        post-processing cost after a fast direct-CI diagonalization.
        """
        if self.ci is None:
            raise ValueError('Run CASCI before requesting S^2.')

        ci = self.ci[state_id]

        if self.direct_connectivity is not None and self.binary is not None:
            if self._s2_operator is None:
                h1_s2, h2_s2 = build_spin_square_operator(self.ncas)
                self._s2_operator = (
                    h1_s2[0],
                    h2_s2[0, 0].copy(),
                    h2_s2[0, 1].copy(),
                )
                self._s2_diag = _compute_diag_compact(
                    self._s2_operator[0], self._s2_operator[1], self._s2_operator[2], self.binary
                )

            conn = self.direct_connectivity
            sigma_s2 = _sigma_compact_conn_numba(
                self._s2_operator[0], self._s2_operator[1], self._s2_operator[2], self._s2_diag, ci, self.binary,
                conn.I_A, conn.J_A, conn.p_A, conn.q_A, conn.phase_A,
                conn.I_B, conn.J_B, conn.p_B, conn.q_B, conn.phase_B,
                conn.I_AA, conn.J_AA, conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA,
                conn.I_BB, conn.J_BB, conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB,
                conn.I_AB, conn.J_AB, conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB,
            )
            return float(np.vdot(ci, sigma_s2).real)

        return spin_square_from_rdm(*self.make_rdm12(state_id))



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

        self.ensure_slater_condon_cache()
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
        self.ensure_slater_condon_cache()
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
