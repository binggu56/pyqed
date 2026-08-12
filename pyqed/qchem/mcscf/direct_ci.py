#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:07 2024

complete active space configuration interaction

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import logging
import os
from functools import reduce
import importlib
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
    FCIStringBasis,
    givenΛgetB,
    SpinOuterProduct,
    get_fci_combos,
    get_fci_string_basis,
    SlaterCondon,
    CI_H,
    determinantsign,
    get_excitation_op,
)
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo

from pyqed.qchem.mcscf.casci import (
    _as_spin_tuple,
    _is_uhf_reference,
    _normalize_spin_1e_operator,
    _reference_active_occupations,
    _slice_active_orbitals,
    _factorized_ci_overlap as overlap,
    h1e_for_cas,
    make_tdm2 as _make_tdm2_dense,
    size_of_cas,
    spin_square as spin_square_from_rdm,
    transform_spatial_eri_to_mo,
    mo_pair_factors,
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
)
from pyqed.qchem import mcscf

from numba import njit, prange

_CASSCF_CPP_UNINITIALIZED = object()
_casscf_cpp = _CASSCF_CPP_UNINITIALIZED
_casscf_cpp_import_error = None


def _cpp_attr(*names):
    global _casscf_cpp, _casscf_cpp_import_error
    if _casscf_cpp is _CASSCF_CPP_UNINITIALIZED:
        try:
            _casscf_cpp = importlib.import_module("pyqed.qchem._casscf_cpp")
        except Exception as exc:  # pragma: no cover - optional accelerator
            _casscf_cpp = None
            _casscf_cpp_import_error = f'{type(exc).__name__}: {exc}'
    if _casscf_cpp is None:
        return None
    for name in names:
        attr = getattr(_casscf_cpp, name, None)
        if attr is not None:
            return attr
    return None


DIRECT_CI_DENSE_FALLBACK_NDETS = 256
DIRECT_CI_AUTO_EIGSH_NDETS = 10000
DIRECT_CI_ROOT_CUSHION = 2
DIRECT_CI_PARALLEL_MIN_NDETS = 4096


def direct_ci_capabilities():
    """Return explicit native and fallback coverage for the direct-CASCI solver."""
    capabilities = {
        'native_extension': False,
        'cblas': False,
        'blas_provider': 'none',
        'rhf_davidson': False,
        'rhf_multiroot': False,
        'spin0_native_max_roots': 0,
        'python_rhf': True,
        'python_uhf': True,
        'python_complex': True,
        'python_symmetry': True,
        'root_homing': True,
    }
    query = _cpp_attr('native_capabilities')
    if query is not None:
        try:
            capabilities.update(dict(query()))
            capabilities['native_extension'] = True
        except Exception as exc:
            capabilities['native_error'] = f'{type(exc).__name__}: {exc}'
    elif _casscf_cpp_import_error is not None:
        capabilities['native_error'] = _casscf_cpp_import_error
    return capabilities


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


@dataclass
class SpinStringConnectivity:
    """
    Spin-string factored connectivity for RHF compact direct-CI.

    Determinants are ordered as ``alpha_index * n_beta + beta_index`` by
    ``get_fci_combos``.  Factoring the links by spin string avoids materializing
    the repeated determinant-product connectivity, which becomes the dominant
    setup cost for spaces such as CAS(10,10).
    """
    alpha_occ: np.ndarray
    beta_occ: np.ndarray
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
    alpha_offsets: np.ndarray | None = None
    beta_offsets: np.ndarray | None = None
    alpha_order: np.ndarray | None = None
    beta_order: np.ndarray | None = None
    alpha_ordered_I: np.ndarray | None = None
    alpha_ordered_J: np.ndarray | None = None
    alpha_ordered_phase: np.ndarray | None = None
    beta_ordered_I: np.ndarray | None = None
    beta_ordered_J: np.ndarray | None = None
    beta_ordered_phase: np.ndarray | None = None


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


def _build_davidson_guess(diag, nroots, guess=None, min_vectors=None):
    """
    Build an initial Davidson subspace from the diagonal and optional user seeds.

    The default strategy mirrors standard CI Davidson practice: use unit vectors
    associated with the smallest diagonal Hamiltonian elements.  If the caller
    provides trial vectors via ``ci0`` we include them first, then fill any
    missing columns from the diagonal guess.
    """
    n = diag.size
    target_cols = max(nroots, 2 * nroots)
    if min_vectors is not None:
        target_cols = max(target_cols, int(min_vectors))
    target_cols = min(n, target_cols)
    cols = []

    if guess is not None:
        if isinstance(guess, (list, tuple)):
            guess_cols = [np.asarray(v).reshape(n) for v in guess]
        else:
            arr = np.asarray(guess)
            if arr.ndim == 1:
                guess_cols = [arr.reshape(n)]
            else:
                guess_cols = [arr[:, i].reshape(n) for i in range(arr.shape[1])]
        cols.extend(guess_cols[:target_cols])

    dtype = np.result_type(np.asarray(diag).dtype, *(col.dtype for col in cols), float)
    order = np.argsort(np.real(diag))
    for idx in order:
        e = np.zeros(n, dtype=dtype)
        e[idx] = 1.0
        cols.append(e)
        if len(cols) >= target_cols:
            break

    return _orthonormalize_columns(np.column_stack(cols))

def _select_direct_ci_guess(casci, nstates, ci0=None, *, reuse=True):
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
    if reuse and casci.ci is not None and len(casci.ci) > 0:
        return casci.ci[:nstates]
    return None


def _resolve_direct_ci_workers(casci, n_det):
    value = getattr(casci, "direct_ci_workers", None)
    explicit = value is not None
    if value is None:
        value = os.environ.get("PYQED_DIRECT_CI_WORKERS")
        explicit = value is not None
    if value is None:
        value = os.environ.get("PYQED_CI_THREADS")
        explicit = value is not None

    if value is None:
        mol = getattr(getattr(casci, "mf", None), "mol", None)
        parallel_min = int(getattr(casci, "direct_ci_parallel_min_ndet", DIRECT_CI_PARALLEL_MIN_NDETS))
        if mol is not None and bool(getattr(mol, "builtin_parallel", False)) and n_det >= parallel_min:
            value = getattr(mol, "builtin_eri_workers", None)
            if value is None:
                value = min(os.cpu_count() or 1, 4)

    if value is None:
        parallel_min = int(getattr(casci, "direct_ci_parallel_min_ndet", DIRECT_CI_PARALLEL_MIN_NDETS))
        if n_det >= parallel_min:
            return max(1, min(os.cpu_count() or 1, 4))
        return 1
    if value is False:
        return 1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "false", "off", "none"}:
            return 1
        if text == "auto":
            return max(1, min(os.cpu_count() or 1, 4))
        value = text
    try:
        workers = max(1, int(value))
    except (TypeError, ValueError):
        return 1
    if not explicit and n_det >= 250_000:
        workers = max(workers, min(os.cpu_count() or workers, 8))
    return workers


def _resolve_direct_ci_tolerances(casci):
    """Return energy and residual thresholds for iterative CI solves."""
    energy_tol = float(getattr(casci, "tol", 0) or 1.0e-8)
    configured = getattr(casci, "direct_ci_residual_tol", None)
    if configured is None:
        residual_tol = max(energy_tol, np.sqrt(energy_tol))
    else:
        residual_tol = float(configured)
        if residual_tol <= 0.0:
            raise ValueError("direct_ci_residual_tol must be positive.")
    return energy_tol, residual_tol


def _spin0_pair_arrays(pairs):
    pair_array = np.asarray(pairs, dtype=np.intp)
    left = pair_array[:, 0]
    right = pair_array[:, 1]
    same = left == right
    return left, right, same


def _rectangular_spin0_pairs(binary):
    if isinstance(binary, FCIStringBasis):
        if binary.nalpha != binary.nbeta or not np.array_equal(
            binary.alpha_occ, binary.beta_occ
        ):
            return None
        alpha_idx, beta_idx = np.triu_indices(binary.nalpha)
        pairs = np.empty((alpha_idx.size, 2), dtype=np.intp)
        pairs[:, 0] = alpha_idx * binary.nbeta + beta_idx
        pairs[:, 1] = beta_idx * binary.nbeta + alpha_idx
        return pairs
    dets = np.asarray(binary, dtype=np.int8)
    if dets.ndim != 3 or dets.shape[1] != 2:
        raise ValueError("binary must have shape (ndet, 2, ncas).")

    n_det = dets.shape[0]
    n_beta = 0
    while n_beta < n_det and np.array_equal(dets[n_beta, 0, :], dets[0, 0, :]):
        n_beta += 1
    if n_beta == 0 or n_det % n_beta != 0:
        return None

    n_alpha = n_det // n_beta
    if n_alpha != n_beta:
        return None

    alpha_occ = dets[::n_beta, 0, :]
    beta_occ = dets[:n_beta, 1, :]
    if not np.array_equal(alpha_occ, beta_occ):
        return None
    if not (
        np.array_equal(dets[:, 0, :], np.repeat(alpha_occ, n_beta, axis=0))
        and np.array_equal(dets[:, 1, :], np.tile(beta_occ, (n_alpha, 1)))
    ):
        return None

    alpha_idx, beta_idx = np.triu_indices(n_alpha)
    pairs = np.empty((alpha_idx.size, 2), dtype=np.intp)
    pairs[:, 0] = alpha_idx * n_beta + beta_idx
    pairs[:, 1] = beta_idx * n_beta + alpha_idx
    return pairs


def _spin0_to_det_vector(c_spin0, left, right, same, ndet):
    c_spin0 = np.asarray(c_spin0)
    det_vec = np.zeros(ndet, dtype=c_spin0.dtype)
    if np.any(same):
        det_vec[left[same]] = c_spin0[same]
    offdiag = ~same
    if np.any(offdiag):
        scaled = c_spin0[offdiag] * (2.0 ** -0.5)
        det_vec[left[offdiag]] = scaled
        det_vec[right[offdiag]] = scaled
    return det_vec


def _det_to_spin0_vector(det_vec, left, right, same):
    det_vec = np.asarray(det_vec)
    c_spin0 = np.empty(left.size, dtype=det_vec.dtype)
    if np.any(same):
        c_spin0[same] = det_vec[left[same]]
    offdiag = ~same
    if np.any(offdiag):
        c_spin0[offdiag] = (det_vec[left[offdiag]] + det_vec[right[offdiag]]) * (2.0 ** -0.5)
    return c_spin0


def _spin0_pair_diagonal(det_diag, left, right, same):
    det_diag = np.asarray(det_diag)
    spin0_diag = np.empty(left.size, dtype=det_diag.dtype)
    if np.any(same):
        spin0_diag[same] = det_diag[left[same]]
    offdiag = ~same
    if np.any(offdiag):
        spin0_diag[offdiag] = 0.5 * (det_diag[left[offdiag]] + det_diag[right[offdiag]])
    return spin0_diag


def _project_spin0_guess(guess, left, right, same, ndet):
    if guess is None:
        return None

    nspin0 = left.size
    cols = []
    if isinstance(guess, (list, tuple)):
        raw_cols = [np.asarray(v, dtype=float).reshape(-1) for v in guess]
    else:
        arr = np.asarray(guess, dtype=float)
        if arr.ndim == 1:
            raw_cols = [arr.reshape(-1)]
        else:
            raw_cols = [arr[:, i].reshape(-1) for i in range(arr.shape[1])]

    for vec in raw_cols:
        if vec.size == nspin0:
            cols.append(vec)
        elif vec.size == ndet:
            cols.append(_det_to_spin0_vector(vec, left, right, same))
        else:
            raise ValueError(
                "Spin0 initial guess has length {}, expected {} or {}.".format(
                    vec.size,
                    nspin0,
                    ndet,
                )
            )

    if not cols:
        return None
    return np.column_stack(cols)


def davidson_lowest(matvec, diag, nroots=1, tol=1e-8, max_cycle=100,
                    max_subspace=None, guess=None, matvec_block=None,
                    trial_block_size=None, energy_tol=None, return_info=False):
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
        # Keep a meaningfully larger default subspace than the first lightweight
        # implementation. Medium-size CASCI problems often need more than a
        # handful of vectors before the diagonal preconditioner becomes useful.
        max_subspace = min(n, max(32, 20 * nroots))
    if trial_block_size is None or matvec_block is None:
        trial_block_size = nroots
    trial_block_size = max(nroots, int(trial_block_size))
    trial_block_size = min(n, max_subspace, trial_block_size)

    def apply_block(block):
        block = np.asarray(block)
        if block.ndim == 1:
            return matvec(block)
        if block.shape[1] == 0:
            return np.empty_like(block)
        if matvec_block is not None:
            result = np.asarray(matvec_block(block))
            if result.shape != block.shape:
                raise ValueError("Davidson block matvec returned an array with the wrong shape.")
            return result
        return np.column_stack([matvec(block[:, i]) for i in range(block.shape[1])])

    if energy_tol is None:
        energy_tol = tol

    V = _build_davidson_guess(diag, nroots, guess=guess, min_vectors=trial_block_size)
    AV = apply_block(V)
    previous_theta = None
    history = []

    for cycle in range(max_cycle):
        T = V.conj().T @ AV
        T = 0.5 * (T + T.conj().T)
        theta_all, alpha_all = eigh(T)
        order = np.argsort(theta_all)
        expand_count = min(trial_block_size, alpha_all.shape[1])
        expand_order = order[:expand_count]
        theta_expand = theta_all[expand_order]
        alpha_expand = alpha_all[:, expand_order]

        ritz_expand = V @ alpha_expand
        Aritz_expand = AV @ alpha_expand
        resid_expand = Aritz_expand - ritz_expand * theta_expand
        resid_norm_expand = np.linalg.norm(resid_expand, axis=0)

        theta = theta_expand[:nroots]
        ritz = ritz_expand[:, :nroots]
        resid_norm = resid_norm_expand[:nroots]
        energy_change = (
            np.full(nroots, np.inf)
            if previous_theta is None
            else np.abs(theta - previous_theta)
        )
        history.append({
            'iteration': cycle + 1,
            'energies': theta.copy(),
            'residual_norms': resid_norm.copy(),
            'energy_changes': energy_change.copy(),
            'subspace_dimension': int(V.shape[1]),
        })

        energy_converged = (
            previous_theta is not None
            and np.all(np.abs(theta - previous_theta) < energy_tol)
        )
        if energy_converged and np.all(resid_norm < tol):
            if return_info:
                return theta, ritz, {
                    'backend': 'python_davidson',
                    'converged': True,
                    'iterations': cycle + 1,
                    'subspace_dimension': int(V.shape[1]),
                    'residual_norms': resid_norm.copy(),
                    'energy_changes': energy_change.copy(),
                    'history': history,
                }
            return theta, ritz
        previous_theta = theta.copy()

        new_vecs = []
        for root in range(expand_count):
            if resid_norm_expand[root] < tol:
                continue

            denom = theta_expand[root] - diag
            safe = np.where(np.abs(denom) > 1e-12, denom, np.where(denom >= 0, 1e-12, -1e-12))
            corr = resid_expand[:, root] / safe

            if V.shape[1] > 0:
                corr -= V @ (V.conj().T @ corr)
            for prev in new_vecs:
                corr -= prev * np.vdot(prev, corr)

            norm = np.linalg.norm(corr)
            if norm > 1e-12:
                new_vecs.append(corr / norm)

        if not new_vecs:
            if np.all(resid_norm < tol):
                if return_info:
                    return theta, ritz, {
                        'backend': 'python_davidson',
                        'converged': True,
                        'iterations': cycle + 1,
                        'subspace_dimension': int(V.shape[1]),
                        'residual_norms': resid_norm.copy(),
                        'energy_changes': energy_change.copy(),
                        'history': history,
                    }
                return theta, ritz
            raise RuntimeError(
                'Davidson correction space collapsed before the requested roots converged.'
            )

        if V.shape[1] + len(new_vecs) > max_subspace:
            # Use a thick restart rather than collapsing all the way back to
            # the target roots. This preserves nearby Ritz information that is
            # often essential for correlated CI Hamiltonians.
            keep = min(alpha_all.shape[1], max(2 * nroots + 2, trial_block_size, nroots + 1))
            restart_block = V @ alpha_all[:, :keep]
            restart_av = AV @ alpha_all[:, :keep]
            restart_cols = [restart_block[:, i] for i in range(restart_block.shape[1])]
            restart_av_cols = [restart_av[:, i] for i in range(restart_av.shape[1])]
            extra_block = (
                new_vecs[0].reshape(-1, 1)
                if len(new_vecs) == 1
                else np.column_stack(new_vecs)
                if new_vecs
                else None
            )
            if extra_block is not None:
                restart_cols.extend(extra_block[:, i] for i in range(extra_block.shape[1]))
                extra_av = apply_block(extra_block)
                restart_av_cols.extend(extra_av[:, i] for i in range(extra_av.shape[1]))
            restart_basis = np.column_stack(restart_cols)
            restart_sigma = np.column_stack(restart_av_cols)
            gram = restart_basis.conj().T @ restart_basis
            if np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8, rtol=1e-8):
                V = restart_basis
                AV = restart_sigma
            else:
                V = _orthonormalize_columns(restart_basis)
                AV = apply_block(V)
        else:
            new_block = (
                new_vecs[0].reshape(-1, 1)
                if len(new_vecs) == 1
                else np.column_stack(new_vecs)
            )
            AV_new = apply_block(new_block)
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
    if _is_uhf_reference(mo_coeff):
        mo_cas_a = mo_coeff[0][:, ncore:ncore+ncas]
        mo_cas_b = mo_coeff[1][:, ncore:ncore+ncas]
        eri_factors = _get_mf_cholesky_factors(mf) if use_cholesky else None
        eri_aa = transform_spatial_eri_to_mo(
            mf, mo_cas_a, mo_cas_a, mo_cas_a, mo_cas_a,
            use_cholesky=use_cholesky, eri_factors=eri_factors,
        )
        eri_ab = transform_spatial_eri_to_mo(
            mf, mo_cas_a, mo_cas_a, mo_cas_b, mo_cas_b,
            use_cholesky=use_cholesky, eri_factors=eri_factors,
        )
        eri_ba = transform_spatial_eri_to_mo(
            mf, mo_cas_b, mo_cas_b, mo_cas_a, mo_cas_a,
            use_cholesky=use_cholesky, eri_factors=eri_factors,
        )
        eri_bb = transform_spatial_eri_to_mo(
            mf, mo_cas_b, mo_cas_b, mo_cas_b, mo_cas_b,
            use_cholesky=use_cholesky, eri_factors=eri_factors,
        )
        return h1, (eri_aa, eri_ab, eri_ba, eri_bb), energy_core

    if (
        not use_cholesky
        and type(mf).__name__ == "_FrozenIntegralRHF"
        and getattr(mf, "eri", None) is not None
    ):
        mo_arr = np.asarray(mo_coeff)
        if mo_arr.ndim == 2 and mo_arr.shape[0] == mo_arr.shape[1]:
            eye = np.eye(mo_arr.shape[0], dtype=mo_arr.dtype)
            if np.array_equal(mo_arr, eye):
                active = slice(ncore, ncore + ncas)
                eri_mo = np.asarray(mf.eri)
                if eri_mo.ndim == 4 and eri_mo.shape[0] >= ncore + ncas:
                    eri_active = np.array(
                        eri_mo[active, active, active, active],
                        copy=True,
                    )
                    return h1, eri_active, energy_core

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
    use_mf_factor_transform = eri_factors is None and hasattr(mf, "mo_factors")
    if eri_factors is None:
        eri_factors = _get_mf_cholesky_factors(mf)

    if _is_uhf_reference(mo_coeff):
        mo_cas_a = mo_coeff[0][:, ncore:ncore+ncas]
        mo_cas_b = mo_coeff[1][:, ncore:ncore+ncas]
        pair_factors = (
            mo_pair_factors(eri_factors, mo_cas_a, mo_cas_a),
            mo_pair_factors(eri_factors, mo_cas_b, mo_cas_b),
        )
        return h1, pair_factors, energy_core

    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    if use_mf_factor_transform:
        pair_factors = mf.mo_factors(mo_cas, mo_cas)
    else:
        pair_factors = mo_pair_factors(eri_factors, mo_cas, mo_cas)
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


def _string_orbital_phases(strings):
    occupied_before = np.cumsum(strings, axis=1) - strings
    return np.where(occupied_before % 2, -1, 1).astype(np.int8)


def _single_string_links(strings):
    single_string_links = _cpp_attr("single_string_links")
    if single_string_links is not None:
        try:
            return single_string_links(
                np.ascontiguousarray(strings, dtype=np.int8)
            )
        except Exception:
            pass

    n_string, n_mo = strings.shape
    bits = [_binary_row_to_bits(strings[i]) for i in range(n_string)]
    lookup = {bits[i]: i for i in range(n_string)}
    phases = _string_orbital_phases(strings)
    links = []

    for ket in range(n_string):
        occ = np.where(strings[ket] == 1)[0]
        vir = np.where(strings[ket] == 0)[0]
        ket_bits = bits[ket]
        for q in occ:
            removed = ket_bits ^ (1 << int(q))
            for p in vir:
                bra = lookup[removed | (1 << int(p))]
                phase = int(phases[ket, p]) * int(phases[bra, q])
                links.append((bra, ket, int(p), int(q), phase))

    if not links:
        return (
            _empty_int_array(), _empty_int_array(), _empty_int_array(),
            _empty_int_array(), _empty_phase_array(),
        )
    cols = list(zip(*links))
    return (
        np.asarray(cols[0], dtype=np.int32),
        np.asarray(cols[1], dtype=np.int32),
        np.asarray(cols[2], dtype=np.int32),
        np.asarray(cols[3], dtype=np.int32),
        np.asarray(cols[4], dtype=np.int8),
    )


def _double_string_links(strings):
    double_string_links = _cpp_attr("double_string_links")
    if double_string_links is not None:
        try:
            return double_string_links(
                np.ascontiguousarray(strings, dtype=np.int8)
            )
        except Exception:
            pass

    n_string, n_mo = strings.shape
    bits = [_binary_row_to_bits(strings[i]) for i in range(n_string)]
    lookup = {bits[i]: i for i in range(n_string)}
    phases = _string_orbital_phases(strings)
    links = []

    for ket in range(n_string):
        occ = np.where(strings[ket] == 1)[0]
        vir = np.where(strings[ket] == 0)[0]
        ket_bits = bits[ket]
        for iq, q in enumerate(occ):
            for s in occ[iq + 1:]:
                removed = ket_bits ^ (1 << int(q)) ^ (1 << int(s))
                for ip, p in enumerate(vir):
                    for r in vir[ip + 1:]:
                        bra = lookup[removed | (1 << int(p)) | (1 << int(r))]
                        # Match get_excitation_op's double-excitation ordering:
                        # first tensor leg uses the higher created/annihilated
                        # orbital, second leg uses the lower one.
                        phase = (
                            int(phases[ket, r])
                            * int(phases[bra, s])
                            * int(phases[ket, p])
                            * int(phases[bra, q])
                        )
                        links.append((bra, ket, int(r), int(s), int(p), int(q), phase))

    if not links:
        return (
            _empty_int_array(), _empty_int_array(), _empty_int_array(),
            _empty_int_array(), _empty_int_array(), _empty_int_array(),
            _empty_phase_array(),
        )
    cols = list(zip(*links))
    return (
        np.asarray(cols[0], dtype=np.int32),
        np.asarray(cols[1], dtype=np.int32),
        np.asarray(cols[2], dtype=np.int32),
        np.asarray(cols[3], dtype=np.int32),
        np.asarray(cols[4], dtype=np.int32),
        np.asarray(cols[5], dtype=np.int32),
        np.asarray(cols[6], dtype=np.int8),
    )


def _group_string_links_by_transition(p_idx, q_idx, n_mo):
    p_idx = np.asarray(p_idx, dtype=np.intp)
    q_idx = np.asarray(q_idx, dtype=np.intp)
    transitions = p_idx * int(n_mo) + q_idx
    n_transition = int(n_mo) * int(n_mo)
    counts = np.bincount(transitions, minlength=n_transition).astype(np.intp, copy=False)
    offsets = np.empty(n_transition + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.empty(transitions.size, dtype=np.intp)
    cursor = offsets[:-1].copy()
    for link, transition in enumerate(transitions):
        slot = cursor[transition]
        order[slot] = link
        cursor[transition] = slot + 1
    return np.ascontiguousarray(offsets), np.ascontiguousarray(order)


def _ordered_string_link_records(I, J, phase, order):
    order = np.asarray(order, dtype=np.intp)
    return (
        np.ascontiguousarray(np.asarray(I, dtype=np.int32)[order]),
        np.ascontiguousarray(np.asarray(J, dtype=np.int32)[order]),
        np.ascontiguousarray(np.asarray(phase, dtype=np.int8)[order]),
    )


def _spin_string_cross_diagonal(eri_cross, occupations):
    eri_cross = np.asarray(eri_cross, dtype=np.float64)
    occupations = np.asarray(occupations, dtype=np.float64)
    diag_terms = np.diagonal(eri_cross, axis1=2, axis2=3)
    contracted = np.einsum('pqr,ar->pqa', diag_terms, occupations, optimize=True)
    return np.ascontiguousarray(
        contracted.reshape(eri_cross.shape[0] * eri_cross.shape[1], occupations.shape[0]),
        dtype=np.float64,
    )


def build_spin_string_connectivity(Binary):
    """
    Build alpha/beta spin-string links without expanding to determinant pairs.
    """
    if isinstance(Binary, FCIStringBasis):
        alpha_occ = Binary.alpha_occ
        beta_occ = Binary.beta_occ
    else:
        n_det = Binary.shape[0]
        n_beta = 0
        while n_beta < n_det and np.array_equal(Binary[n_beta, 0, :], Binary[0, 0, :]):
            n_beta += 1
        if n_beta == 0 or n_det % n_beta != 0:
            raise ValueError("Determinant basis is not a rectangular alpha/beta product.")

        alpha_occ = np.ascontiguousarray(Binary[::n_beta, 0, :])
        beta_occ = np.ascontiguousarray(Binary[:n_beta, 1, :])
        n_alpha = alpha_occ.shape[0]
        if n_alpha * n_beta != Binary.shape[0]:
            raise ValueError("Determinant basis is not a rectangular alpha/beta product.")
        if not (
            np.array_equal(Binary[:, 0, :], np.repeat(alpha_occ, n_beta, axis=0))
            and np.array_equal(Binary[:, 1, :], np.tile(beta_occ, (n_alpha, 1)))
        ):
            raise ValueError("Spin-string direct-CI expects alpha-major determinant ordering.")

    I_A, J_A, p_A, q_A, phase_A = _single_string_links(alpha_occ)
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA = _double_string_links(alpha_occ)
    if np.array_equal(alpha_occ, beta_occ):
        I_B, J_B, p_B, q_B, phase_B = I_A, J_A, p_A, q_A, phase_A
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB = (
            I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA
        )
    else:
        I_B, J_B, p_B, q_B, phase_B = _single_string_links(beta_occ)
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB = _double_string_links(beta_occ)
    alpha_offsets, alpha_order = _group_string_links_by_transition(p_A, q_A, alpha_occ.shape[1])
    if p_B is p_A and q_B is q_A:
        beta_offsets, beta_order = alpha_offsets, alpha_order
    else:
        beta_offsets, beta_order = _group_string_links_by_transition(p_B, q_B, beta_occ.shape[1])
    alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase = _ordered_string_link_records(
        I_A, J_A, phase_A, alpha_order,
    )
    if beta_order is alpha_order and I_B is I_A and J_B is J_A and phase_B is phase_A:
        beta_ordered_I, beta_ordered_J, beta_ordered_phase = (
            alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase
        )
    else:
        beta_ordered_I, beta_ordered_J, beta_ordered_phase = _ordered_string_link_records(
            I_B, J_B, phase_B, beta_order,
        )
    return SpinStringConnectivity(
        alpha_occ, beta_occ,
        I_A, J_A, p_A, q_A, phase_A,
        I_B, J_B, p_B, q_B, phase_B,
        I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
        alpha_offsets, beta_offsets, alpha_order, beta_order,
        alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
        beta_ordered_I, beta_ordered_J, beta_ordered_phase,
    )


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


def _repeat_link_values(values, repeat):
    if len(values) == 0 or repeat == 0:
        return np.empty(0, dtype=values.dtype)
    return np.repeat(values, repeat).astype(values.dtype, copy=False)


def _tile_link_values(values, tile):
    if len(values) == 0 or tile == 0:
        return np.empty(0, dtype=values.dtype)
    return np.tile(values, tile).astype(values.dtype, copy=False)


def _expand_alpha_links_to_dets(I, J, p, q, phase, n_beta):
    n_link = len(I)
    if n_link == 0 or n_beta == 0:
        return (
            _empty_int_array(), _empty_int_array(),
            _empty_int_array(), _empty_int_array(), _empty_phase_array(),
        )
    beta = np.arange(n_beta, dtype=np.int32)
    I_det = (np.asarray(I, dtype=np.int32)[:, None] * np.int32(n_beta) + beta[None, :]).ravel()
    J_det = (np.asarray(J, dtype=np.int32)[:, None] * np.int32(n_beta) + beta[None, :]).ravel()
    return (
        np.ascontiguousarray(I_det, dtype=np.int32),
        np.ascontiguousarray(J_det, dtype=np.int32),
        np.ascontiguousarray(_repeat_link_values(np.asarray(p, dtype=np.int32), n_beta), dtype=np.int32),
        np.ascontiguousarray(_repeat_link_values(np.asarray(q, dtype=np.int32), n_beta), dtype=np.int32),
        np.ascontiguousarray(_repeat_link_values(np.asarray(phase, dtype=np.int8), n_beta), dtype=np.int8),
    )


def _expand_beta_links_to_dets(I, J, p, q, phase, n_alpha, n_beta):
    n_link = len(I)
    if n_link == 0 or n_alpha == 0:
        return (
            _empty_int_array(), _empty_int_array(),
            _empty_int_array(), _empty_int_array(), _empty_phase_array(),
        )
    alpha = np.arange(n_alpha, dtype=np.int32)
    I_det = (alpha[:, None] * np.int32(n_beta) + np.asarray(I, dtype=np.int32)[None, :]).ravel()
    J_det = (alpha[:, None] * np.int32(n_beta) + np.asarray(J, dtype=np.int32)[None, :]).ravel()
    return (
        np.ascontiguousarray(I_det, dtype=np.int32),
        np.ascontiguousarray(J_det, dtype=np.int32),
        np.ascontiguousarray(_tile_link_values(np.asarray(p, dtype=np.int32), n_alpha), dtype=np.int32),
        np.ascontiguousarray(_tile_link_values(np.asarray(q, dtype=np.int32), n_alpha), dtype=np.int32),
        np.ascontiguousarray(_tile_link_values(np.asarray(phase, dtype=np.int8), n_alpha), dtype=np.int8),
    )


def _build_direct_connectivity_from_spin_strings(Binary):
    spin_conn = build_spin_string_connectivity(Binary)
    n_alpha = spin_conn.alpha_occ.shape[0]
    n_beta = spin_conn.beta_occ.shape[0]

    I_A, J_A, p_A, q_A, phase_A = _expand_alpha_links_to_dets(
        spin_conn.I_A, spin_conn.J_A, spin_conn.p_A, spin_conn.q_A, spin_conn.phase_A, n_beta
    )
    I_B, J_B, p_B, q_B, phase_B = _expand_beta_links_to_dets(
        spin_conn.I_B, spin_conn.J_B, spin_conn.p_B, spin_conn.q_B, spin_conn.phase_B, n_alpha, n_beta
    )
    I_AA, J_AA, p_AA, q_AA, phase_AA_single = _expand_alpha_links_to_dets(
        spin_conn.I_AA, spin_conn.J_AA, spin_conn.p_AA, spin_conn.q_AA, spin_conn.phase_AA, n_beta
    )
    r_AA = np.ascontiguousarray(_repeat_link_values(np.asarray(spin_conn.r_AA, dtype=np.int32), n_beta), dtype=np.int32)
    s_AA = np.ascontiguousarray(_repeat_link_values(np.asarray(spin_conn.s_AA, dtype=np.int32), n_beta), dtype=np.int32)
    phase_AA = phase_AA_single
    I_BB, J_BB, p_BB, q_BB, phase_BB_single = _expand_beta_links_to_dets(
        spin_conn.I_BB, spin_conn.J_BB, spin_conn.p_BB, spin_conn.q_BB, spin_conn.phase_BB, n_alpha, n_beta
    )
    r_BB = np.ascontiguousarray(_tile_link_values(np.asarray(spin_conn.r_BB, dtype=np.int32), n_alpha), dtype=np.int32)
    s_BB = np.ascontiguousarray(_tile_link_values(np.asarray(spin_conn.s_BB, dtype=np.int32), n_alpha), dtype=np.int32)
    phase_BB = phase_BB_single

    n_a = len(spin_conn.I_A)
    n_b = len(spin_conn.I_B)
    if n_a == 0 or n_b == 0:
        I_AB = J_AB = p_AB = q_AB = r_AB = s_AB = _empty_int_array()
        phase_AB = _empty_phase_array()
    else:
        I_AB = np.ascontiguousarray(
            (np.asarray(spin_conn.I_A, dtype=np.int32)[:, None] * np.int32(n_beta)
             + np.asarray(spin_conn.I_B, dtype=np.int32)[None, :]).ravel(),
            dtype=np.int32,
        )
        J_AB = np.ascontiguousarray(
            (np.asarray(spin_conn.J_A, dtype=np.int32)[:, None] * np.int32(n_beta)
             + np.asarray(spin_conn.J_B, dtype=np.int32)[None, :]).ravel(),
            dtype=np.int32,
        )
        p_AB = np.ascontiguousarray(_repeat_link_values(np.asarray(spin_conn.p_A, dtype=np.int32), n_b), dtype=np.int32)
        q_AB = np.ascontiguousarray(_repeat_link_values(np.asarray(spin_conn.q_A, dtype=np.int32), n_b), dtype=np.int32)
        r_AB = np.ascontiguousarray(_tile_link_values(np.asarray(spin_conn.p_B, dtype=np.int32), n_a), dtype=np.int32)
        s_AB = np.ascontiguousarray(_tile_link_values(np.asarray(spin_conn.q_B, dtype=np.int32), n_a), dtype=np.int32)
        phase_AB = np.ascontiguousarray(
            (
                np.asarray(spin_conn.phase_A, dtype=np.int8)[:, None]
                * np.asarray(spin_conn.phase_B, dtype=np.int8)[None, :]
            ).ravel(),
            dtype=np.int8,
        )

    return DirectConnectivity(
        I_A, J_A, p_A, q_A, phase_A,
        I_B, J_B, p_B, q_B, phase_B,
        I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
        I_AB, J_AB, p_AB, q_AB, r_AB, s_AB, phase_AB,
    )


def _build_direct_connectivity_slow(Binary):
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
                I = lookup.get((removed | (1 << p), b_bits))
                if I is not None:
                    singles_a.append((I, J))

        for q in occ_b:
            removed = b_bits ^ (1 << q)
            for p in vir_b:
                I = lookup.get((a_bits, removed | (1 << p)))
                if I is not None:
                    singles_b.append((I, J))

        for iq, q in enumerate(occ_a):
            for is_, s in enumerate(occ_a[iq + 1:], start=iq + 1):
                removed = a_bits ^ (1 << q) ^ (1 << s)
                for ip, p in enumerate(vir_a):
                    for ir, r in enumerate(vir_a[ip + 1:], start=ip + 1):
                        I = lookup.get((removed | (1 << p) | (1 << r), b_bits))
                        if I is not None:
                            doubles_aa.append((I, J))

        for iq, q in enumerate(occ_b):
            for is_, s in enumerate(occ_b[iq + 1:], start=iq + 1):
                removed = b_bits ^ (1 << q) ^ (1 << s)
                for ip, p in enumerate(vir_b):
                    for ir, r in enumerate(vir_b[ip + 1:], start=ip + 1):
                        I = lookup.get((a_bits, removed | (1 << p) | (1 << r)))
                        if I is not None:
                            doubles_bb.append((I, J))

        for q in occ_a:
            a_removed = a_bits ^ (1 << q)
            for s in occ_b:
                b_removed = b_bits ^ (1 << s)
                for p in vir_a:
                    for r in vir_b:
                        I = lookup.get((a_removed | (1 << p), b_removed | (1 << r)))
                        if I is not None:
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


def build_direct_connectivity(Binary):
    try:
        return _build_direct_connectivity_from_spin_strings(Binary)
    except ValueError:
        return _build_direct_connectivity_slow(Binary)


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_expanded(H1, H2, Binary):

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


def _compute_diag(H1, H2, Binary):
    if isinstance(Binary, FCIStringBasis):
        return _compute_diag_compact_uhf(
            H1[0], H1[1],
            H2[0, 0], H2[0, 1], H2[1, 0], H2[1, 1],
            Binary,
        )
    return _compute_diag_expanded(H1, H2, Binary)

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
def _compute_diag_compact_expanded(h1, eri_same, eri_cross, Binary):
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


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_compact_uhf_expanded(h1a, h1b, eri_aa, eri_ab, eri_ba, eri_bb, Binary):
    """Diagonal CI elements for spin-dependent active orbitals."""
    n_dets, _, n_mo = Binary.shape
    h1a_diag = np.diag(h1a)
    h1b_diag = np.diag(h1b)
    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += h1a_diag[p]
            if Binary[i, 1, p]:
                H_diag[i] += h1b_diag[p]

        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * eri_aa[p, p, q, q]
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * eri_ab[p, p, q, q]
            if Binary[i, 1, p]:
                for q in range(n_mo):
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * eri_bb[p, p, q, q]
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * eri_ba[p, p, q, q]

    return H_diag


def _diagonal_pair_matrix(eri):
    norb = eri.shape[0]
    orbitals = np.arange(norb)
    return np.asarray(
        eri[
            orbitals[:, None],
            orbitals[:, None],
            orbitals[None, :],
            orbitals[None, :],
        ]
    )


def _compute_diag_compact(h1, eri_same, eri_cross, basis):
    if not isinstance(basis, FCIStringBasis):
        return _compute_diag_compact_expanded(h1, eri_same, eri_cross, basis)
    alpha = np.asarray(basis.alpha_occ, dtype=np.float64)
    beta = np.asarray(basis.beta_occ, dtype=np.float64)
    one_body = np.diag(h1)
    same = _diagonal_pair_matrix(eri_same)
    cross = _diagonal_pair_matrix(eri_cross)
    energy_alpha = alpha @ one_body + 0.5 * np.einsum(
        "ap,pq,aq->a", alpha, same, alpha, optimize=True
    )
    energy_beta = beta @ one_body + 0.5 * np.einsum(
        "ap,pq,aq->a", beta, same, beta, optimize=True
    )
    diagonal = 0.5 * alpha @ (cross + cross.T) @ beta.T
    diagonal += energy_alpha[:, None]
    diagonal += energy_beta[None, :]
    return np.ascontiguousarray(diagonal.reshape(-1))


def _compute_diag_compact_uhf(h1a, h1b, eri_aa, eri_ab, eri_ba, eri_bb, basis):
    if not isinstance(basis, FCIStringBasis):
        return _compute_diag_compact_uhf_expanded(
            h1a, h1b, eri_aa, eri_ab, eri_ba, eri_bb, basis
        )
    alpha = np.asarray(basis.alpha_occ, dtype=np.float64)
    beta = np.asarray(basis.beta_occ, dtype=np.float64)
    same_alpha = _diagonal_pair_matrix(eri_aa)
    same_beta = _diagonal_pair_matrix(eri_bb)
    cross_ab = _diagonal_pair_matrix(eri_ab)
    cross_ba = _diagonal_pair_matrix(eri_ba)
    energy_alpha = alpha @ np.diag(h1a) + 0.5 * np.einsum(
        "ap,pq,aq->a", alpha, same_alpha, alpha, optimize=True
    )
    energy_beta = beta @ np.diag(h1b) + 0.5 * np.einsum(
        "ap,pq,aq->a", beta, same_beta, beta, optimize=True
    )
    diagonal = 0.5 * alpha @ (cross_ab + cross_ba.T) @ beta.T
    diagonal += energy_alpha[:, None]
    diagonal += energy_beta[None, :]
    return np.ascontiguousarray(diagonal.reshape(-1))


@njit(nogil=True, cache=True, fastmath=True)
def _factor_coulomb(pair_factors, p, q, r, s):
    val = 0.0
    for t in range(pair_factors.shape[0]):
        val += pair_factors[t, p, q] * pair_factors[t, r, s]
    return val


@njit(nogil=True, cache=True, fastmath=True)
def _factor_coulomb_mixed(pair_factors_left, pair_factors_right, p, q, r, s):
    val = 0.0
    for t in range(pair_factors_left.shape[0]):
        val += pair_factors_left[t, p, q] * pair_factors_right[t, r, s]
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


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_diag_compact_factors_uhf(h1a, h1b, pair_factors_a, pair_factors_b, Binary):
    n_dets, _, n_mo = Binary.shape
    h1a_diag = np.diag(h1a)
    h1b_diag = np.diag(h1b)
    H_diag = np.zeros(n_dets)

    for i in prange(n_dets):
        for p in range(n_mo):
            if Binary[i, 0, p]:
                H_diag[i] += h1a_diag[p]
            if Binary[i, 1, p]:
                H_diag[i] += h1b_diag[p]

        for p in range(n_mo):
            if Binary[i, 0, p]:
                for q in range(n_mo):
                    if Binary[i, 0, q]:
                        coul = _factor_coulomb(pair_factors_a, p, p, q, q)
                        exch = _factor_coulomb(pair_factors_a, p, q, q, p)
                        H_diag[i] += 0.5 * (coul - exch)
                    if Binary[i, 1, q]:
                        H_diag[i] += 0.5 * _factor_coulomb_mixed(
                            pair_factors_a, pair_factors_b, p, p, q, q
                        )

            if Binary[i, 1, p]:
                for q in range(n_mo):
                    if Binary[i, 1, q]:
                        coul = _factor_coulomb(pair_factors_b, p, p, q, q)
                        exch = _factor_coulomb(pair_factors_b, p, q, q, p)
                        H_diag[i] += 0.5 * (coul - exch)
                    if Binary[i, 0, q]:
                        H_diag[i] += 0.5 * _factor_coulomb_mixed(
                            pair_factors_b, pair_factors_a, p, p, q, q
                        )

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


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_compact_spin_string_numba(
    h1, eri_same, eri_cross, H_diag, c,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
):
    """
    Compact RHF direct-CI sigma in a spin-string product basis.

    This performs the same work as the determinant connectivity kernel, but it
    loops over alpha and beta string links separately instead of walking a large
    pre-expanded determinant-pair table.
    """
    n_alpha = alpha_occ.shape[0]
    n_beta = beta_occ.shape[0]
    n_mo = h1.shape[0]
    sigma_vec = H_diag * c

    for link in range(I_A.shape[0]):
        ia = I_A[link]
        ja = J_A[link]
        p = p_A[link]
        q = q_A[link]
        sign = phase_A[link]
        same_part = -sign * h1[p, q]
        for r in range(n_mo):
            if alpha_occ[ja, r] and r != q:
                same_part -= sign * eri_same[p, q, r, r]
        for ib in range(n_beta):
            val = same_part
            for r in range(n_mo):
                if beta_occ[ib, r]:
                    val -= sign * eri_cross[p, q, r, r]
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + ib]

    for link in range(I_B.shape[0]):
        ib = I_B[link]
        jb = J_B[link]
        p = p_B[link]
        q = q_B[link]
        sign = phase_B[link]
        same_part = -sign * h1[p, q]
        for r in range(n_mo):
            if beta_occ[jb, r] and r != q:
                same_part -= sign * eri_same[p, q, r, r]
        for ia in range(n_alpha):
            val = same_part
            for r in range(n_mo):
                if alpha_occ[ia, r]:
                    val -= sign * eri_cross[p, q, r, r]
            sigma_vec[ia * n_beta + ib] += val * c[ia * n_beta + jb]

    for link in range(I_AA.shape[0]):
        ia = I_AA[link]
        ja = J_AA[link]
        val = phase_AA[link] * eri_same[p_AA[link], q_AA[link], r_AA[link], s_AA[link]]
        for ib in range(n_beta):
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + ib]

    for link in range(I_BB.shape[0]):
        ib = I_BB[link]
        jb = J_BB[link]
        val = phase_BB[link] * eri_same[p_BB[link], q_BB[link], r_BB[link], s_BB[link]]
        for ia in range(n_alpha):
            sigma_vec[ia * n_beta + ib] += val * c[ia * n_beta + jb]

    for la in range(I_A.shape[0]):
        ia = I_A[la]
        ja = J_A[la]
        pa = p_A[la]
        qa = q_A[la]
        phase_alpha = phase_A[la]
        for lb in range(I_B.shape[0]):
            ib = I_B[lb]
            jb = J_B[lb]
            val = phase_alpha * phase_B[lb] * eri_cross[pa, qa, p_B[lb], q_B[lb]]
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + jb]

    return sigma_vec


@njit(nogil=True, cache=True, fastmath=True)
def _sigma_compact_spin_string_uhf_numba(
    h1a, h1b, eri_aa, eri_ab, eri_ba, eri_bb, H_diag, c,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
):
    """Spin-string direct-CI sigma for distinct alpha/beta active orbitals."""
    n_alpha = alpha_occ.shape[0]
    n_beta = beta_occ.shape[0]
    n_mo = h1a.shape[0]
    sigma_vec = H_diag * c

    for link in range(I_A.shape[0]):
        ia = I_A[link]
        ja = J_A[link]
        p = p_A[link]
        q = q_A[link]
        sign = phase_A[link]
        same_part = -sign * h1a[p, q]
        for r in range(n_mo):
            if alpha_occ[ja, r] and r != q:
                same_part -= sign * eri_aa[p, q, r, r]
        for ib in range(n_beta):
            val = same_part
            for r in range(n_mo):
                if beta_occ[ib, r]:
                    val -= sign * eri_ab[p, q, r, r]
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + ib]

    for link in range(I_B.shape[0]):
        ib = I_B[link]
        jb = J_B[link]
        p = p_B[link]
        q = q_B[link]
        sign = phase_B[link]
        same_part = -sign * h1b[p, q]
        for r in range(n_mo):
            if beta_occ[jb, r] and r != q:
                same_part -= sign * eri_bb[p, q, r, r]
        for ia in range(n_alpha):
            val = same_part
            for r in range(n_mo):
                if alpha_occ[ia, r]:
                    val -= sign * eri_ba[p, q, r, r]
            sigma_vec[ia * n_beta + ib] += val * c[ia * n_beta + jb]

    for link in range(I_AA.shape[0]):
        ia = I_AA[link]
        ja = J_AA[link]
        val = phase_AA[link] * eri_aa[p_AA[link], q_AA[link], r_AA[link], s_AA[link]]
        for ib in range(n_beta):
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + ib]

    for link in range(I_BB.shape[0]):
        ib = I_BB[link]
        jb = J_BB[link]
        val = phase_BB[link] * eri_bb[p_BB[link], q_BB[link], r_BB[link], s_BB[link]]
        for ia in range(n_alpha):
            sigma_vec[ia * n_beta + ib] += val * c[ia * n_beta + jb]

    for la in range(I_A.shape[0]):
        ia = I_A[la]
        ja = J_A[la]
        pa = p_A[la]
        qa = q_A[la]
        phase_alpha = phase_A[la]
        for lb in range(I_B.shape[0]):
            ib = I_B[lb]
            jb = J_B[lb]
            val = phase_alpha * phase_B[lb] * eri_ab[pa, qa, p_B[lb], q_B[lb]]
            sigma_vec[ia * n_beta + ib] += val * c[ja * n_beta + jb]

    return sigma_vec


def _sigma_compact_spin_string(
    h1, eri_same, eri_cross, H_diag, c,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    alpha_offsets=None,
    beta_offsets=None,
    alpha_order=None,
    beta_order=None,
    alpha_cross_diag=None,
    beta_cross_diag=None,
    alpha_ordered_I=None,
    alpha_ordered_J=None,
    alpha_ordered_phase=None,
    beta_ordered_I=None,
    beta_ordered_J=None,
    beta_ordered_phase=None,
    workers=None,
):
    sigma_compact_spin_string = _cpp_attr("sigma_compact_spin_string")
    if (
        sigma_compact_spin_string is not None
        and not (
            np.iscomplexobj(h1)
            or np.iscomplexobj(eri_same)
            or np.iscomplexobj(eri_cross)
            or np.iscomplexobj(H_diag)
            or np.iscomplexobj(c)
        )
    ):
        try:
            args = (
                np.ascontiguousarray(h1, dtype=np.float64),
                np.ascontiguousarray(eri_same, dtype=np.float64),
                np.ascontiguousarray(eri_cross, dtype=np.float64),
                np.ascontiguousarray(H_diag, dtype=np.float64),
                np.ascontiguousarray(c, dtype=np.float64),
                np.ascontiguousarray(alpha_occ, dtype=np.int8),
                np.ascontiguousarray(beta_occ, dtype=np.int8),
                np.ascontiguousarray(I_A, dtype=np.int32),
                np.ascontiguousarray(J_A, dtype=np.int32),
                np.ascontiguousarray(p_A, dtype=np.int32),
                np.ascontiguousarray(q_A, dtype=np.int32),
                np.ascontiguousarray(phase_A, dtype=np.int8),
                np.ascontiguousarray(I_B, dtype=np.int32),
                np.ascontiguousarray(J_B, dtype=np.int32),
                np.ascontiguousarray(p_B, dtype=np.int32),
                np.ascontiguousarray(q_B, dtype=np.int32),
                np.ascontiguousarray(phase_B, dtype=np.int8),
                np.ascontiguousarray(I_AA, dtype=np.int32),
                np.ascontiguousarray(J_AA, dtype=np.int32),
                np.ascontiguousarray(p_AA, dtype=np.int32),
                np.ascontiguousarray(q_AA, dtype=np.int32),
                np.ascontiguousarray(r_AA, dtype=np.int32),
                np.ascontiguousarray(s_AA, dtype=np.int32),
                np.ascontiguousarray(phase_AA, dtype=np.int8),
                np.ascontiguousarray(I_BB, dtype=np.int32),
                np.ascontiguousarray(J_BB, dtype=np.int32),
                np.ascontiguousarray(p_BB, dtype=np.int32),
                np.ascontiguousarray(q_BB, dtype=np.int32),
                np.ascontiguousarray(r_BB, dtype=np.int32),
                np.ascontiguousarray(s_BB, dtype=np.int32),
                np.ascontiguousarray(phase_BB, dtype=np.int8),
            )
            if (
                alpha_offsets is not None
                and beta_offsets is not None
                and alpha_order is not None
                and beta_order is not None
                and alpha_cross_diag is not None
                and beta_cross_diag is not None
            ):
                args = args + (
                    np.ascontiguousarray(alpha_offsets, dtype=np.intp),
                    np.ascontiguousarray(beta_offsets, dtype=np.intp),
                    np.ascontiguousarray(alpha_order, dtype=np.intp),
                    np.ascontiguousarray(beta_order, dtype=np.intp),
                    np.ascontiguousarray(alpha_cross_diag, dtype=np.float64),
                    np.ascontiguousarray(beta_cross_diag, dtype=np.float64),
                )
                if (
                    alpha_ordered_I is not None
                    and alpha_ordered_J is not None
                    and alpha_ordered_phase is not None
                    and beta_ordered_I is not None
                    and beta_ordered_J is not None
                    and beta_ordered_phase is not None
                ):
                    args = args + (
                        np.ascontiguousarray(alpha_ordered_I, dtype=np.int32),
                        np.ascontiguousarray(alpha_ordered_J, dtype=np.int32),
                        np.ascontiguousarray(alpha_ordered_phase, dtype=np.int8),
                        np.ascontiguousarray(beta_ordered_I, dtype=np.int32),
                        np.ascontiguousarray(beta_ordered_J, dtype=np.int32),
                        np.ascontiguousarray(beta_ordered_phase, dtype=np.int8),
                    )
                    if workers is not None:
                        args = args + (int(workers),)
            return sigma_compact_spin_string(*args)
        except Exception:
            pass

    return _sigma_compact_spin_string_numba(
        h1, eri_same, eri_cross, H_diag, c,
        alpha_occ, beta_occ,
        I_A, J_A, p_A, q_A, phase_A,
        I_B, J_B, p_B, q_B, phase_B,
        I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    )


def _sigma_compact_spin0_pair(
    h1, eri_same, eri_cross, H_diag, c_pair, pair_left, pair_right,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    alpha_offsets, beta_offsets, alpha_order, beta_order,
    alpha_cross_diag, beta_cross_diag,
    alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
    beta_ordered_I, beta_ordered_J, beta_ordered_phase,
    workers=None,
):
    sigma_compact_spin0_pair = _cpp_attr("sigma_compact_spin0_pair")
    if (
        sigma_compact_spin0_pair is not None
        and not (
            np.iscomplexobj(h1)
            or np.iscomplexobj(eri_same)
            or np.iscomplexobj(eri_cross)
            or np.iscomplexobj(H_diag)
            or np.iscomplexobj(c_pair)
        )
    ):
        try:
            args = (
                np.ascontiguousarray(h1, dtype=np.float64),
                np.ascontiguousarray(eri_same, dtype=np.float64),
                np.ascontiguousarray(eri_cross, dtype=np.float64),
                np.ascontiguousarray(H_diag, dtype=np.float64),
                np.ascontiguousarray(c_pair, dtype=np.float64),
                np.ascontiguousarray(pair_left, dtype=np.intp),
                np.ascontiguousarray(pair_right, dtype=np.intp),
                np.ascontiguousarray(alpha_occ, dtype=np.int8),
                np.ascontiguousarray(beta_occ, dtype=np.int8),
                np.ascontiguousarray(I_A, dtype=np.int32),
                np.ascontiguousarray(J_A, dtype=np.int32),
                np.ascontiguousarray(p_A, dtype=np.int32),
                np.ascontiguousarray(q_A, dtype=np.int32),
                np.ascontiguousarray(phase_A, dtype=np.int8),
                np.ascontiguousarray(I_B, dtype=np.int32),
                np.ascontiguousarray(J_B, dtype=np.int32),
                np.ascontiguousarray(p_B, dtype=np.int32),
                np.ascontiguousarray(q_B, dtype=np.int32),
                np.ascontiguousarray(phase_B, dtype=np.int8),
                np.ascontiguousarray(I_AA, dtype=np.int32),
                np.ascontiguousarray(J_AA, dtype=np.int32),
                np.ascontiguousarray(p_AA, dtype=np.int32),
                np.ascontiguousarray(q_AA, dtype=np.int32),
                np.ascontiguousarray(r_AA, dtype=np.int32),
                np.ascontiguousarray(s_AA, dtype=np.int32),
                np.ascontiguousarray(phase_AA, dtype=np.int8),
                np.ascontiguousarray(I_BB, dtype=np.int32),
                np.ascontiguousarray(J_BB, dtype=np.int32),
                np.ascontiguousarray(p_BB, dtype=np.int32),
                np.ascontiguousarray(q_BB, dtype=np.int32),
                np.ascontiguousarray(r_BB, dtype=np.int32),
                np.ascontiguousarray(s_BB, dtype=np.int32),
                np.ascontiguousarray(phase_BB, dtype=np.int8),
                np.ascontiguousarray(alpha_offsets, dtype=np.intp),
                np.ascontiguousarray(beta_offsets, dtype=np.intp),
                np.ascontiguousarray(alpha_order, dtype=np.intp),
                np.ascontiguousarray(beta_order, dtype=np.intp),
                np.ascontiguousarray(alpha_cross_diag, dtype=np.float64),
                np.ascontiguousarray(beta_cross_diag, dtype=np.float64),
                np.ascontiguousarray(alpha_ordered_I, dtype=np.int32),
                np.ascontiguousarray(alpha_ordered_J, dtype=np.int32),
                np.ascontiguousarray(alpha_ordered_phase, dtype=np.int8),
                np.ascontiguousarray(beta_ordered_I, dtype=np.int32),
                np.ascontiguousarray(beta_ordered_J, dtype=np.int32),
                np.ascontiguousarray(beta_ordered_phase, dtype=np.int8),
            )
            if workers is not None:
                args = args + (int(workers),)
            return sigma_compact_spin0_pair(*args)
        except Exception:
            pass

    ndet = int(np.asarray(H_diag).shape[0])
    pair_left = np.asarray(pair_left, dtype=np.intp)
    pair_right = np.asarray(pair_right, dtype=np.intp)
    same = pair_left == pair_right
    c_det = _spin0_to_det_vector(c_pair, pair_left, pair_right, same, ndet)
    sigma_det = _sigma_compact_spin_string(
        h1, eri_same, eri_cross, H_diag, c_det,
        alpha_occ, beta_occ,
        I_A, J_A, p_A, q_A, phase_A,
        I_B, J_B, p_B, q_B, phase_B,
        I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
        I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
        alpha_offsets, beta_offsets, alpha_order, beta_order,
        alpha_cross_diag, beta_cross_diag,
        alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
        beta_ordered_I, beta_ordered_J, beta_ordered_phase,
        workers,
    )
    return _det_to_spin0_vector(sigma_det, pair_left, pair_right, same)


def _make_sigma_compact_spin0_pair_cpp_matvec(
    h1, eri_same, eri_cross, H_diag, pair_left, pair_right,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    alpha_offsets, beta_offsets, alpha_order, beta_order,
    alpha_cross_diag, beta_cross_diag,
    alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
    beta_ordered_I, beta_ordered_J, beta_ordered_phase,
    workers=None,
):
    sigma_compact_spin0_pair = _cpp_attr("sigma_compact_spin0_pair")
    if (
        sigma_compact_spin0_pair is None
        or np.iscomplexobj(h1)
        or np.iscomplexobj(eri_same)
        or np.iscomplexobj(eri_cross)
        or np.iscomplexobj(H_diag)
    ):
        return None
    try:
        head = (
            np.ascontiguousarray(h1, dtype=np.float64),
            np.ascontiguousarray(eri_same, dtype=np.float64),
            np.ascontiguousarray(eri_cross, dtype=np.float64),
            np.ascontiguousarray(H_diag, dtype=np.float64),
        )
        tail = (
            np.ascontiguousarray(pair_left, dtype=np.intp),
            np.ascontiguousarray(pair_right, dtype=np.intp),
            np.ascontiguousarray(alpha_occ, dtype=np.int8),
            np.ascontiguousarray(beta_occ, dtype=np.int8),
            np.ascontiguousarray(I_A, dtype=np.int32),
            np.ascontiguousarray(J_A, dtype=np.int32),
            np.ascontiguousarray(p_A, dtype=np.int32),
            np.ascontiguousarray(q_A, dtype=np.int32),
            np.ascontiguousarray(phase_A, dtype=np.int8),
            np.ascontiguousarray(I_B, dtype=np.int32),
            np.ascontiguousarray(J_B, dtype=np.int32),
            np.ascontiguousarray(p_B, dtype=np.int32),
            np.ascontiguousarray(q_B, dtype=np.int32),
            np.ascontiguousarray(phase_B, dtype=np.int8),
            np.ascontiguousarray(I_AA, dtype=np.int32),
            np.ascontiguousarray(J_AA, dtype=np.int32),
            np.ascontiguousarray(p_AA, dtype=np.int32),
            np.ascontiguousarray(q_AA, dtype=np.int32),
            np.ascontiguousarray(r_AA, dtype=np.int32),
            np.ascontiguousarray(s_AA, dtype=np.int32),
            np.ascontiguousarray(phase_AA, dtype=np.int8),
            np.ascontiguousarray(I_BB, dtype=np.int32),
            np.ascontiguousarray(J_BB, dtype=np.int32),
            np.ascontiguousarray(p_BB, dtype=np.int32),
            np.ascontiguousarray(q_BB, dtype=np.int32),
            np.ascontiguousarray(r_BB, dtype=np.int32),
            np.ascontiguousarray(s_BB, dtype=np.int32),
            np.ascontiguousarray(phase_BB, dtype=np.int8),
            np.ascontiguousarray(alpha_offsets, dtype=np.intp),
            np.ascontiguousarray(beta_offsets, dtype=np.intp),
            np.ascontiguousarray(alpha_order, dtype=np.intp),
            np.ascontiguousarray(beta_order, dtype=np.intp),
            np.ascontiguousarray(alpha_cross_diag, dtype=np.float64),
            np.ascontiguousarray(beta_cross_diag, dtype=np.float64),
            np.ascontiguousarray(alpha_ordered_I, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_J, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_phase, dtype=np.int8),
            np.ascontiguousarray(beta_ordered_I, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_J, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_phase, dtype=np.int8),
        )
        # The standalone native matvec owns temporary output buffers only for
        # one caller thread. Parallelism is provided by the fully native
        # Davidson workspace, which persists those buffers across iterations.
        worker_args = () if workers is None else (1,)

        def matvec(c_pair):
            return sigma_compact_spin0_pair(
                *head,
                np.ascontiguousarray(c_pair, dtype=np.float64),
                *tail,
                *worker_args,
            )

        return matvec
    except Exception:
        return None


def _make_sigma_compact_rhf_blas_cpp_matvec(
    h1, eri_same, eri_cross, H_diag, pair_left, pair_right,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    alpha_offsets, beta_offsets, alpha_order, beta_order,
    alpha_cross_diag, beta_cross_diag,
    alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
    beta_ordered_I, beta_ordered_J, beta_ordered_phase,
    workers=None,
):
    create_workspace = _cpp_attr("create_spin0_pair_workspace")
    apply_workspace = _cpp_attr("apply_spin0_pair_workspace_det")
    if (
        create_workspace is None
        or apply_workspace is None
        or np.iscomplexobj(h1)
        or np.iscomplexobj(eri_same)
        or np.iscomplexobj(eri_cross)
        or np.iscomplexobj(H_diag)
    ):
        return None
    try:
        workspace = create_workspace(
            np.ascontiguousarray(h1, dtype=np.float64),
            np.ascontiguousarray(eri_same, dtype=np.float64),
            np.ascontiguousarray(eri_cross, dtype=np.float64),
            np.ascontiguousarray(H_diag, dtype=np.float64),
            np.zeros(len(pair_left), dtype=np.float64),
            np.ascontiguousarray(pair_left, dtype=np.intp),
            np.ascontiguousarray(pair_right, dtype=np.intp),
            np.ascontiguousarray(alpha_occ, dtype=np.int8),
            np.ascontiguousarray(beta_occ, dtype=np.int8),
            np.ascontiguousarray(I_A, dtype=np.int32),
            np.ascontiguousarray(J_A, dtype=np.int32),
            np.ascontiguousarray(p_A, dtype=np.int32),
            np.ascontiguousarray(q_A, dtype=np.int32),
            np.ascontiguousarray(phase_A, dtype=np.int8),
            np.ascontiguousarray(I_B, dtype=np.int32),
            np.ascontiguousarray(J_B, dtype=np.int32),
            np.ascontiguousarray(p_B, dtype=np.int32),
            np.ascontiguousarray(q_B, dtype=np.int32),
            np.ascontiguousarray(phase_B, dtype=np.int8),
            np.ascontiguousarray(I_AA, dtype=np.int32),
            np.ascontiguousarray(J_AA, dtype=np.int32),
            np.ascontiguousarray(p_AA, dtype=np.int32),
            np.ascontiguousarray(q_AA, dtype=np.int32),
            np.ascontiguousarray(r_AA, dtype=np.int32),
            np.ascontiguousarray(s_AA, dtype=np.int32),
            np.ascontiguousarray(phase_AA, dtype=np.int8),
            np.ascontiguousarray(I_BB, dtype=np.int32),
            np.ascontiguousarray(J_BB, dtype=np.int32),
            np.ascontiguousarray(p_BB, dtype=np.int32),
            np.ascontiguousarray(q_BB, dtype=np.int32),
            np.ascontiguousarray(r_BB, dtype=np.int32),
            np.ascontiguousarray(s_BB, dtype=np.int32),
            np.ascontiguousarray(phase_BB, dtype=np.int8),
            np.ascontiguousarray(alpha_offsets, dtype=np.intp),
            np.ascontiguousarray(beta_offsets, dtype=np.intp),
            np.ascontiguousarray(alpha_order, dtype=np.intp),
            np.ascontiguousarray(beta_order, dtype=np.intp),
            np.ascontiguousarray(alpha_cross_diag, dtype=np.float64),
            np.ascontiguousarray(beta_cross_diag, dtype=np.float64),
            np.ascontiguousarray(alpha_ordered_I, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_J, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_phase, dtype=np.int8),
            np.ascontiguousarray(beta_ordered_I, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_J, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_phase, dtype=np.int8),
            int(workers or 1),
        )

        def matvec(c_det):
            return apply_workspace(
                workspace,
                np.ascontiguousarray(c_det, dtype=np.float64),
            )

        matvec._native_workspace = workspace
        return matvec
    except Exception:
        return None


def _davidson_rhf_blas_cpp(
    matvec,
    H_diag,
    *,
    guess=None,
    nroots=1,
    energy_tol=1.0e-8,
    residual_tol=1.0e-4,
    max_cycle=100,
    max_subspace=None,
):
    davidson_native = _cpp_attr("davidson_rhf_workspace")
    workspace = getattr(matvec, "_native_workspace", None)
    if int(nroots) < 1:
        return None, {
            'backend': 'native_rhf_davidson',
            'attempted': False,
            'used': False,
            'fallback_reason': 'invalid root count',
        }
    if davidson_native is None:
        detail = _casscf_cpp_import_error or 'extension not built'
        return None, {
            'backend': 'native_rhf_davidson',
            'attempted': False,
            'used': False,
            'fallback_reason': f'native _casscf_cpp extension is unavailable: {detail}',
        }
    if workspace is None:
        return None, {
            'backend': 'native_rhf_davidson',
            'attempted': False,
            'used': False,
            'fallback_reason': 'packed restricted-CI workspace is unavailable',
        }
    try:
        ndet = int(np.asarray(H_diag).shape[0])
        if max_subspace is None:
            base_subspace = 12 if ndet >= 250_000 else 32
            max_subspace = min(ndet, max(base_subspace, 4 * int(nroots) + 4))
        if guess is None:
            native_guess = np.empty((ndet, 0), dtype=np.float64)
        else:
            native_guess = _build_davidson_guess(
                np.asarray(H_diag),
                int(nroots),
                guess=guess,
                min_vectors=min(ndet, max(2, int(nroots))),
            )
            native_guess = np.ascontiguousarray(native_guess, dtype=np.float64)
        energies, vectors, diagnostics = davidson_native(
            workspace,
            np.ascontiguousarray(H_diag, dtype=np.float64),
            int(nroots),
            float(energy_tol),
            float(residual_tol),
            int(max_cycle),
            int(max_subspace),
            native_guess,
        )
        diagnostics = dict(diagnostics)
        diagnostics.update({
            'backend': 'native_rhf_davidson',
            'attempted': True,
            'used': True,
            'fallback_reason': None,
            'nroots': int(nroots),
            'ndeterminants': ndet,
            'max_subspace': int(max_subspace),
            'used_initial_guess': guess is not None,
        })
        return (
            np.asarray(energies, dtype=np.float64),
            np.asarray(vectors, dtype=np.float64),
        ), diagnostics
    except Exception as exc:
        return None, {
            'backend': 'native_rhf_davidson',
            'attempted': True,
            'used': False,
            'fallback_reason': f'{type(exc).__name__}: {exc}',
            'nroots': int(nroots),
        }


def _davidson_spin0_pair_cpp(
    h1, eri_same, eri_cross, H_diag, pair_left, pair_right,
    alpha_occ, beta_occ,
    I_A, J_A, p_A, q_A, phase_A,
    I_B, J_B, p_B, q_B, phase_B,
    I_AA, J_AA, p_AA, q_AA, r_AA, s_AA, phase_AA,
    I_BB, J_BB, p_BB, q_BB, r_BB, s_BB, phase_BB,
    alpha_offsets, beta_offsets, alpha_order, beta_order,
    alpha_cross_diag, beta_cross_diag,
    alpha_ordered_I, alpha_ordered_J, alpha_ordered_phase,
    beta_ordered_I, beta_ordered_J, beta_ordered_phase,
    *,
    workers=None,
    nroots=1,
    energy_tol=1e-8,
    residual_tol=1e-4,
    max_cycle=100,
    max_subspace=None,
):
    davidson_spin0_pair = _cpp_attr("davidson_spin0_pair")
    if (
        davidson_spin0_pair is None
        or int(nroots) != 1
        or np.iscomplexobj(h1)
        or np.iscomplexobj(eri_same)
        or np.iscomplexobj(eri_cross)
        or np.iscomplexobj(H_diag)
    ):
        return None
    try:
        n_pair = int(np.asarray(pair_left).shape[0])
        if n_pair <= 0:
            return None
        if max_subspace is None:
            max_subspace = min(n_pair, max(32, 20 * int(nroots)))
        c_dummy = np.empty(n_pair, dtype=np.float64)
        args = (
            np.ascontiguousarray(h1, dtype=np.float64),
            np.ascontiguousarray(eri_same, dtype=np.float64),
            np.ascontiguousarray(eri_cross, dtype=np.float64),
            np.ascontiguousarray(H_diag, dtype=np.float64),
            c_dummy,
            np.ascontiguousarray(pair_left, dtype=np.intp),
            np.ascontiguousarray(pair_right, dtype=np.intp),
            np.ascontiguousarray(alpha_occ, dtype=np.int8),
            np.ascontiguousarray(beta_occ, dtype=np.int8),
            np.ascontiguousarray(I_A, dtype=np.int32),
            np.ascontiguousarray(J_A, dtype=np.int32),
            np.ascontiguousarray(p_A, dtype=np.int32),
            np.ascontiguousarray(q_A, dtype=np.int32),
            np.ascontiguousarray(phase_A, dtype=np.int8),
            np.ascontiguousarray(I_B, dtype=np.int32),
            np.ascontiguousarray(J_B, dtype=np.int32),
            np.ascontiguousarray(p_B, dtype=np.int32),
            np.ascontiguousarray(q_B, dtype=np.int32),
            np.ascontiguousarray(phase_B, dtype=np.int8),
            np.ascontiguousarray(I_AA, dtype=np.int32),
            np.ascontiguousarray(J_AA, dtype=np.int32),
            np.ascontiguousarray(p_AA, dtype=np.int32),
            np.ascontiguousarray(q_AA, dtype=np.int32),
            np.ascontiguousarray(r_AA, dtype=np.int32),
            np.ascontiguousarray(s_AA, dtype=np.int32),
            np.ascontiguousarray(phase_AA, dtype=np.int8),
            np.ascontiguousarray(I_BB, dtype=np.int32),
            np.ascontiguousarray(J_BB, dtype=np.int32),
            np.ascontiguousarray(p_BB, dtype=np.int32),
            np.ascontiguousarray(q_BB, dtype=np.int32),
            np.ascontiguousarray(r_BB, dtype=np.int32),
            np.ascontiguousarray(s_BB, dtype=np.int32),
            np.ascontiguousarray(phase_BB, dtype=np.int8),
            np.ascontiguousarray(alpha_offsets, dtype=np.intp),
            np.ascontiguousarray(beta_offsets, dtype=np.intp),
            np.ascontiguousarray(alpha_order, dtype=np.intp),
            np.ascontiguousarray(beta_order, dtype=np.intp),
            np.ascontiguousarray(alpha_cross_diag, dtype=np.float64),
            np.ascontiguousarray(beta_cross_diag, dtype=np.float64),
            np.ascontiguousarray(alpha_ordered_I, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_J, dtype=np.int32),
            np.ascontiguousarray(alpha_ordered_phase, dtype=np.int8),
            np.ascontiguousarray(beta_ordered_I, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_J, dtype=np.int32),
            np.ascontiguousarray(beta_ordered_phase, dtype=np.int8),
            int(workers if workers is not None else 1),
            int(nroots),
            float(energy_tol),
            float(residual_tol),
            int(max_cycle),
            int(max_subspace),
        )
        energies, vecs = davidson_spin0_pair(*args)
        return np.asarray(energies, dtype=np.float64), np.asarray(vecs, dtype=np.float64)
    except Exception:
        return None


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _sigma_compact_derivative_batch_numba(
    h1_batch,
    eri_batch,
    c,
    Binary,
    I_A,
    J_A,
    p_A,
    q_A,
    phase_A,
    I_B,
    J_B,
    p_B,
    q_B,
    phase_B,
    I_AA,
    J_AA,
    p_AA,
    q_AA,
    r_AA,
    s_AA,
    phase_AA,
    I_BB,
    J_BB,
    p_BB,
    q_BB,
    r_BB,
    s_BB,
    phase_BB,
    I_AB,
    J_AB,
    p_AB,
    q_AB,
    r_AB,
    s_AB,
    phase_AB,
):
    """
    Batched compact direct-CI sigma for orbital-derivative active integrals.

    Each batch item is a spin-independent active-space Hamiltonian derivative.
    Keeping the orbital-variable loop inside Numba removes thousands of Python
    calls in the exact orbital-gradient path.
    """
    n_batch = h1_batch.shape[0]
    n_det, _, n_mo = Binary.shape
    sigma = np.zeros((n_batch, n_det), dtype=c.dtype)

    for b in prange(n_batch):
        h1 = h1_batch[b]
        eri = eri_batch[b]

        for det in range(n_det):
            val = 0.0
            for p in range(n_mo):
                if Binary[det, 0, p]:
                    val += h1[p, p]
                if Binary[det, 1, p]:
                    val += h1[p, p]

            for p in range(n_mo):
                if Binary[det, 0, p]:
                    for q in range(n_mo):
                        if Binary[det, 0, q]:
                            val += 0.5 * (eri[p, p, q, q] - eri[p, q, q, p])
                        if Binary[det, 1, q]:
                            val += 0.5 * eri[p, p, q, q]

                if Binary[det, 1, p]:
                    for q in range(n_mo):
                        if Binary[det, 1, q]:
                            val += 0.5 * (eri[p, p, q, q] - eri[p, q, q, p])
                        if Binary[det, 0, q]:
                            val += 0.5 * eri[p, p, q, q]

            sigma[b, det] = val * c[det]

        for k in range(I_A.shape[0]):
            p = p_A[k]
            q = q_A[k]
            sign = phase_A[k]
            j = J_A[k]
            val = -sign * h1[p, q]
            for r in range(n_mo):
                if Binary[j, 0, r] and r != q:
                    val -= sign * (eri[p, q, r, r] - eri[p, r, r, q])
                if Binary[j, 1, r]:
                    val -= sign * eri[p, q, r, r]
            sigma[b, I_A[k]] += val * c[j]

        for k in range(I_B.shape[0]):
            p = p_B[k]
            q = q_B[k]
            sign = phase_B[k]
            j = J_B[k]
            val = -sign * h1[p, q]
            for r in range(n_mo):
                if Binary[j, 1, r] and r != q:
                    val -= sign * (eri[p, q, r, r] - eri[p, r, r, q])
                if Binary[j, 0, r]:
                    val -= sign * eri[p, q, r, r]
            sigma[b, I_B[k]] += val * c[j]

        for k in range(I_AA.shape[0]):
            sigma[b, I_AA[k]] += (
                phase_AA[k]
                * (eri[p_AA[k], q_AA[k], r_AA[k], s_AA[k]]
                   - eri[p_AA[k], s_AA[k], r_AA[k], q_AA[k]])
                * c[J_AA[k]]
            )

        for k in range(I_BB.shape[0]):
            sigma[b, I_BB[k]] += (
                phase_BB[k]
                * (eri[p_BB[k], q_BB[k], r_BB[k], s_BB[k]]
                   - eri[p_BB[k], s_BB[k], r_BB[k], q_BB[k]])
                * c[J_BB[k]]
            )

        for k in range(I_AB.shape[0]):
            sigma[b, I_AB[k]] += (
                phase_AB[k]
                * eri[p_AB[k], q_AB[k], r_AB[k], s_AB[k]]
                * c[J_AB[k]]
            )

    return sigma


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
def _compute_single_values_from_factors_uhf(
    J, p_idx, q_idx, phase, h1_spin, pair_factors_spin, pair_factors_other, Binary, spin
):
    n_exc = len(J)
    n_mo = h1_spin.shape[0]
    values = np.zeros(n_exc)

    for k in prange(n_exc):
        p = p_idx[k]
        q = q_idx[k]
        sign = phase[k]
        j = J[k]

        val = -sign * h1_spin[p, q]
        for r in range(n_mo):
            coul = _factor_coulomb(pair_factors_spin, p, q, r, r)
            if Binary[j, spin, r] and r != q:
                exch = _factor_coulomb(pair_factors_spin, p, r, r, q)
                val -= sign * (coul - exch)
            if Binary[j, 1 - spin, r]:
                val -= sign * _factor_coulomb_mixed(
                    pair_factors_spin, pair_factors_other, p, q, r, r
                )

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


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _compute_double_cross_values_from_mixed_factors(
    p_idx, q_idx, r_idx, s_idx, phase, pair_factors_left, pair_factors_right
):
    n_exc = len(p_idx)
    values = np.zeros(n_exc)

    for k in prange(n_exc):
        values[k] = phase[k] * _factor_coulomb_mixed(
            pair_factors_left, pair_factors_right, p_idx[k], q_idx[k], r_idx[k], s_idx[k]
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


def _sigma_values_conn_fast(
    H_diag, H_A, H_B, H_AA, H_BB, H_AB, c,
    I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
):
    """
    Fast connection-value sigma for one vector or a Davidson trial block.
    """
    c_arr = np.asarray(c)
    if c_arr.ndim not in (1, 2):
        raise ValueError("CI sigma vector must be a 1D vector or 2D block.")
    sigma_values_conn = _cpp_attr("sigma_values_conn")
    if (
        sigma_values_conn is not None
        and not np.iscomplexobj(c_arr)
        and not np.iscomplexobj(H_diag)
        and not np.iscomplexobj(H_A)
        and not np.iscomplexobj(H_B)
        and not np.iscomplexobj(H_AA)
        and not np.iscomplexobj(H_BB)
        and not np.iscomplexobj(H_AB)
    ):
        try:
            return sigma_values_conn(
                np.ascontiguousarray(H_diag, dtype=np.float64),
                np.ascontiguousarray(H_A, dtype=np.float64),
                np.ascontiguousarray(H_B, dtype=np.float64),
                np.ascontiguousarray(H_AA, dtype=np.float64),
                np.ascontiguousarray(H_BB, dtype=np.float64),
                np.ascontiguousarray(H_AB, dtype=np.float64),
                np.ascontiguousarray(c_arr, dtype=np.float64),
                np.ascontiguousarray(I_A, dtype=np.int32),
                np.ascontiguousarray(J_A, dtype=np.int32),
                np.ascontiguousarray(I_B, dtype=np.int32),
                np.ascontiguousarray(J_B, dtype=np.int32),
                np.ascontiguousarray(I_AA, dtype=np.int32),
                np.ascontiguousarray(J_AA, dtype=np.int32),
                np.ascontiguousarray(I_BB, dtype=np.int32),
                np.ascontiguousarray(J_BB, dtype=np.int32),
                np.ascontiguousarray(I_AB, dtype=np.int32),
                np.ascontiguousarray(J_AB, dtype=np.int32),
            )
        except Exception:
            pass

    if c_arr.ndim == 1:
        return _sigma_values_conn_numba(
            H_diag, H_A, H_B, H_AA, H_BB, H_AB, c_arr,
            I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
        )
    if c_arr.shape[1] == 1:
        sigma = _sigma_values_conn_numba(
            H_diag, H_A, H_B, H_AA, H_BB, H_AB, c_arr[:, 0],
            I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
        )
        return sigma.reshape(-1, 1)

    return np.column_stack([
        _sigma_values_conn_numba(
            H_diag, H_A, H_B, H_AA, H_BB, H_AB, c_arr[:, col],
            I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
        )
        for col in range(c_arr.shape[1])
    ])


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
    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        spin=None,
        ms2=None,
        multiplicity=None,
        tol=0,
        verbose=0,
    ):
        r"""
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
        super().__init__(
            mf,
            ncas,
            nelecas,
            ncore=ncore,
            spin=spin,
            ms2=ms2,
            multiplicity=multiplicity,
            verbose=verbose,
        )
        self.direct_connectivity = None
        self.spin_string_connectivity = None

        self.tol = tol
        self.direct_ci_dense_fallback_ndets = DIRECT_CI_DENSE_FALLBACK_NDETS
        self.direct_spin0_symm_dense_fallback_nconfigs = DIRECT_CI_DENSE_FALLBACK_NDETS
        self.solver_backend = None
        self.direct_ci_eigensolver = 'davidson'
        self.direct_ci_auto_eigsh_ndets = DIRECT_CI_AUTO_EIGSH_NDETS
        self.direct_ci_root_cushion = DIRECT_CI_ROOT_CUSHION
        self.direct_ci_max_cycle = 100
        self.direct_ci_max_subspace = None
        self.direct_ci_residual_tol = None
        self.direct_ci_factor_davidson_block_size = 1
        self.direct_ci_reuse_guess = True
        self.direct_ci_auto_spin0 = True
        self.direct_ci_workers = None
        self.direct_ci_parallel_min_ndet = DIRECT_CI_PARALLEL_MIN_NDETS
        self.direct_ci_native_davidson = True
        self.direct_spin0_native_pair = True
        self.direct_spin0_native_davidson = True
        self._s2_operator = None
        self._s2_diag = None
        self._direct_spatial_h1 = None
        self._direct_spatial_eri = None
        self._direct_pair_factors = None
        self._direct_same_spin_eri = None
        self._direct_cross_spin_eri = None
        self._direct_uhf_compact_integrals = None
        self._direct_H_diag = None
        self._spin_string_alpha_cross_diag = None
        self._spin_string_beta_cross_diag = None
        self._direct_rhf_blas_matvec = None
        self._direct_ci_used_native_davidson = False
        self.direct_ci_native_diagnostics = {}
        self.direct_ci_diagnostics = {}
        self.direct_ci_fallback_reason = None
        self.converged = False
        self._direct_factor_H_diag = None
        self._direct_factor_H_A = None
        self._direct_factor_H_B = None
        self._direct_factor_H_AA = None
        self._direct_factor_H_BB = None
        self._direct_factor_H_AB = None
        self._direct_integrals_mo_ref = None
        self._direct_integrals_mo_snapshot = None
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
        h1a, h1b = _normalize_spin_1e_operator(H)
        H1 = [np.asarray(h1a), np.asarray(h1b)]

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
            and self._direct_integrals_mo_snapshot is not None
            and np.array_equal(self._direct_integrals_mo_snapshot, self.mo_coeff)
            and self._direct_integrals_ncore == self.ncore
            and self._direct_integrals_ncas == self.ncas
            and self._direct_integrals_use_cholesky == bool(use_cholesky)
        ):
            return self._direct_spatial_h1, self._direct_spatial_eri, self.e_core

        self._direct_same_spin_eri = None
        self._direct_cross_spin_eri = None
        self._direct_pair_factors = None
        self._direct_uhf_compact_integrals = None
        self._direct_H_diag = None
        self._spin_string_alpha_cross_diag = None
        self._spin_string_beta_cross_diag = None
        self._direct_rhf_blas_matvec = None
        self._direct_factor_H_diag = None
        self._direct_factor_H_A = None
        self._direct_factor_H_B = None
        self._direct_factor_H_AA = None
        self._direct_factor_H_BB = None
        self._direct_factor_H_AB = None
        h1, eri_spatial, energy_core = transform_active_space_spatial_integrals(
            self.mf, self.mo_coeff, self.ncas, self.ncore, use_cholesky=use_cholesky
        )
        self._direct_spatial_h1 = h1
        self._direct_spatial_eri = eri_spatial
        self._direct_integrals_mo_ref = self.mo_coeff
        self._direct_integrals_mo_snapshot = np.array(self.mo_coeff, copy=True)
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
            and self._direct_integrals_mo_snapshot is not None
            and np.array_equal(self._direct_integrals_mo_snapshot, self.mo_coeff)
            and self._direct_integrals_ncore == self.ncore
            and self._direct_integrals_ncas == self.ncas
            and self._direct_integrals_use_cholesky == bool(use_cholesky)
        ):
            return self._direct_spatial_h1, self._direct_pair_factors, self.e_core

        self._direct_spatial_eri = None
        self._direct_same_spin_eri = None
        self._direct_cross_spin_eri = None
        self._direct_uhf_compact_integrals = None
        self._direct_H_diag = None
        self._spin_string_alpha_cross_diag = None
        self._spin_string_beta_cross_diag = None
        self._direct_rhf_blas_matvec = None
        self._direct_factor_H_diag = None
        self._direct_factor_H_A = None
        self._direct_factor_H_B = None
        self._direct_factor_H_AA = None
        self._direct_factor_H_BB = None
        self._direct_factor_H_AB = None
        h1, pair_factors, energy_core = transform_active_space_pair_factors(
            self.mf, self.mo_coeff, self.ncas, self.ncore
        )
        self._direct_spatial_h1 = h1
        self._direct_pair_factors = pair_factors
        self._direct_integrals_mo_ref = self.mo_coeff
        self._direct_integrals_mo_snapshot = np.array(self.mo_coeff, copy=True)
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
            and self._direct_integrals_mo_snapshot is not None
            and np.array_equal(self._direct_integrals_mo_snapshot, self.mo_coeff)
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

        if _is_uhf_reference(pair_factors):
            pair_factors_a, pair_factors_b = pair_factors
            h1a, h1b = _normalize_spin_1e_operator(h1)
            self._direct_factor_H_diag = _compute_diag_compact_factors_uhf(
                h1a, h1b, pair_factors_a, pair_factors_b, binary
            )
            self._direct_factor_H_A = _compute_single_values_from_factors_uhf(
                conn.J_A, conn.p_A, conn.q_A, conn.phase_A,
                np.asarray(h1a), pair_factors_a, pair_factors_b, binary, 0
            )
            self._direct_factor_H_B = _compute_single_values_from_factors_uhf(
                conn.J_B, conn.p_B, conn.q_B, conn.phase_B,
                np.asarray(h1b), pair_factors_b, pair_factors_a, binary, 1
            )
            self._direct_factor_H_AA = _compute_double_same_values_from_factors(
                conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA, pair_factors_a
            )
            self._direct_factor_H_BB = _compute_double_same_values_from_factors(
                conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB, pair_factors_b
            )
            self._direct_factor_H_AB = _compute_double_cross_values_from_mixed_factors(
                conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB,
                pair_factors_a, pair_factors_b
            )
        else:
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
        if isinstance(self.binary, FCIStringBasis):
            self.binary = self.binary.materialize()
        if getattr(self, "SC1", None) is None or getattr(self, "SC2", None) is None:
            self.SC1, self.SC2 = SlaterCondon(self.binary)
        return self.SC1, self.SC2

    def ci_sigma(self, c):
        """
        Apply the active-space CI Hamiltonian to a determinant-space vector.

        The returned vector excludes the scalar core energy, matching the CI
        eigenvalues stored internally before ``e_core`` is added to ``e_tot``.
        This method exposes the matrix-free direct-CI action used by ``run()``
        so reduced CI subspace solvers do not need to build dense Hamiltonians.
        """
        if self.binary is None:
            raise ValueError("Run CASCI before requesting a CI sigma vector.")
        c = np.asarray(c)
        if c.ndim != 1 or c.shape[0] != self.binary.shape[0]:
            raise ValueError(
                "CI vector shape {} is incompatible with ndet={}.".format(
                    c.shape,
                    self.binary.shape[0],
                )
            )
        spin_conn = self.spin_string_connectivity
        if spin_conn is not None and self._direct_uhf_compact_integrals is not None:
            h_diag = self._direct_H_diag
            if h_diag is None:
                h_diag = _compute_diag_compact_uhf(
                    *self._direct_uhf_compact_integrals,
                    self.binary,
                )
                self._direct_H_diag = h_diag
            return _sigma_compact_spin_string_uhf_numba(
                *self._direct_uhf_compact_integrals, h_diag, c,
                spin_conn.alpha_occ, spin_conn.beta_occ,
                spin_conn.I_A, spin_conn.J_A, spin_conn.p_A, spin_conn.q_A, spin_conn.phase_A,
                spin_conn.I_B, spin_conn.J_B, spin_conn.p_B, spin_conn.q_B, spin_conn.phase_B,
                spin_conn.I_AA, spin_conn.J_AA,
                spin_conn.p_AA, spin_conn.q_AA, spin_conn.r_AA, spin_conn.s_AA, spin_conn.phase_AA,
                spin_conn.I_BB, spin_conn.J_BB,
                spin_conn.p_BB, spin_conn.q_BB, spin_conn.r_BB, spin_conn.s_BB, spin_conn.phase_BB,
            )

        if spin_conn is not None and self.h2e_cas is not None:
            rhf_blas_matvec = getattr(self, "_direct_rhf_blas_matvec", None)
            if rhf_blas_matvec is not None and not np.iscomplexobj(c):
                return rhf_blas_matvec(c)
            spatial_h1 = np.asarray(self.hcore)[0]
            spatial_eri = np.asarray(self.h2e_cas)
            same_spin_eri = (
                self._direct_same_spin_eri
                if self._direct_same_spin_eri is not None
                else spatial_eri - spatial_eri.swapaxes(1, 3)
            )
            cross_spin_eri = (
                self._direct_cross_spin_eri
                if self._direct_cross_spin_eri is not None
                else spatial_eri
            )
            h_diag = self._direct_H_diag
            if h_diag is None:
                h_diag = _compute_diag_compact(
                    spatial_h1, same_spin_eri, cross_spin_eri, self.binary
                )
                self._direct_H_diag = h_diag
            alpha_cross = self._spin_string_alpha_cross_diag
            beta_cross = self._spin_string_beta_cross_diag
            if alpha_cross is None:
                alpha_cross = _spin_string_cross_diagonal(cross_spin_eri, spin_conn.alpha_occ)
                beta_cross = _spin_string_cross_diagonal(cross_spin_eri, spin_conn.beta_occ)
                self._spin_string_alpha_cross_diag = alpha_cross
                self._spin_string_beta_cross_diag = beta_cross
            return _sigma_compact_spin_string(
                spatial_h1, same_spin_eri, cross_spin_eri, h_diag, c,
                spin_conn.alpha_occ, spin_conn.beta_occ,
                spin_conn.I_A, spin_conn.J_A, spin_conn.p_A, spin_conn.q_A, spin_conn.phase_A,
                spin_conn.I_B, spin_conn.J_B, spin_conn.p_B, spin_conn.q_B, spin_conn.phase_B,
                spin_conn.I_AA, spin_conn.J_AA,
                spin_conn.p_AA, spin_conn.q_AA, spin_conn.r_AA, spin_conn.s_AA, spin_conn.phase_AA,
                spin_conn.I_BB, spin_conn.J_BB,
                spin_conn.p_BB, spin_conn.q_BB, spin_conn.r_BB, spin_conn.s_BB, spin_conn.phase_BB,
                spin_conn.alpha_offsets, spin_conn.beta_offsets,
                spin_conn.alpha_order, spin_conn.beta_order,
                alpha_cross, beta_cross,
                spin_conn.alpha_ordered_I, spin_conn.alpha_ordered_J, spin_conn.alpha_ordered_phase,
                spin_conn.beta_ordered_I, spin_conn.beta_ordered_J, spin_conn.beta_ordered_phase,
                _resolve_direct_ci_workers(self, self.binary.shape[0]),
            )

        if self.direct_connectivity is None:
            self.direct_connectivity = build_direct_connectivity(self.binary)
        conn = self.direct_connectivity

        factor_ready = all(
            x is not None
            for x in (
                self._direct_factor_H_diag,
                self._direct_factor_H_A,
                self._direct_factor_H_B,
                self._direct_factor_H_AA,
                self._direct_factor_H_BB,
                self._direct_factor_H_AB,
            )
        )
        if factor_ready:
            return _sigma_values_conn_fast(
                self._direct_factor_H_diag,
                self._direct_factor_H_A,
                self._direct_factor_H_B,
                self._direct_factor_H_AA,
                self._direct_factor_H_BB,
                self._direct_factor_H_AB,
                c,
                conn.I_A,
                conn.J_A,
                conn.I_B,
                conn.J_B,
                conn.I_AA,
                conn.J_AA,
                conn.I_BB,
                conn.J_BB,
                conn.I_AB,
                conn.J_AB,
            )

        if self.h2e_cas is not None:
            hcore = np.asarray(self.hcore)
            if hcore.ndim == 3:
                if not np.allclose(hcore[0], hcore[1], atol=1.0e-12):
                    if self.eri_so is None:
                        raise NotImplementedError(
                            "Spin-dependent compact CI sigma needs eri_so fallback."
                        )
                spatial_h1 = hcore[0]
            else:
                spatial_h1 = hcore
            spatial_eri = np.asarray(self.h2e_cas)
            same_spin_eri = (
                self._direct_same_spin_eri
                if self._direct_same_spin_eri is not None
                else spatial_eri - spatial_eri.swapaxes(1, 3)
            )
            cross_spin_eri = (
                self._direct_cross_spin_eri
                if self._direct_cross_spin_eri is not None
                else spatial_eri
            )
            h_diag = _compute_diag_compact(
                spatial_h1,
                same_spin_eri,
                cross_spin_eri,
                self.binary,
            )
            return _sigma_compact_conn_numba(
                spatial_h1,
                same_spin_eri,
                cross_spin_eri,
                h_diag,
                c,
                self.binary,
                conn.I_A,
                conn.J_A,
                conn.p_A,
                conn.q_A,
                conn.phase_A,
                conn.I_B,
                conn.J_B,
                conn.p_B,
                conn.q_B,
                conn.phase_B,
                conn.I_AA,
                conn.J_AA,
                conn.p_AA,
                conn.q_AA,
                conn.r_AA,
                conn.s_AA,
                conn.phase_AA,
                conn.I_BB,
                conn.J_BB,
                conn.p_BB,
                conn.q_BB,
                conn.r_BB,
                conn.s_BB,
                conn.phase_BB,
                conn.I_AB,
                conn.J_AB,
                conn.p_AB,
                conn.q_AB,
                conn.r_AB,
                conn.s_AB,
                conn.phase_AB,
            )

        if self.eri_so is None or self.hcore is None:
            raise ValueError("CASCI Hamiltonian data are not available for sigma.")
        h1e = np.asarray(self.hcore)
        h2e = np.asarray(self.eri_so)
        h_diag = _compute_diag(h1e, h2e, self.binary)
        sc1, sc2 = self.ensure_slater_condon_cache()
        return sigma_on_the_fly(self.binary, sc1, sc2, h1e, h2e, h_diag, c)

    def ci_diagonal(self):
        """
        Return the determinant-space active CI Hamiltonian diagonal.

        The diagonal excludes ``e_core`` and is suitable for Davidson/Q-space
        residual preconditioning.
        """
        if self.binary is None:
            raise ValueError("Run CASCI before requesting a CI Hamiltonian diagonal.")

        factor_ready = self._direct_factor_H_diag is not None
        if factor_ready:
            return np.asarray(self._direct_factor_H_diag)
        if self._direct_H_diag is not None:
            return np.asarray(self._direct_H_diag)

        if self.h2e_cas is not None:
            hcore = np.asarray(self.hcore)
            if hcore.ndim == 3:
                if not np.allclose(hcore[0], hcore[1], atol=1.0e-12):
                    if self.eri_so is None:
                        raise NotImplementedError(
                            "Spin-dependent compact CI diagonal needs eri_so fallback."
                        )
                spatial_h1 = hcore[0]
            else:
                spatial_h1 = hcore
            spatial_eri = np.asarray(self.h2e_cas)
            same_spin_eri = (
                self._direct_same_spin_eri
                if self._direct_same_spin_eri is not None
                else spatial_eri - spatial_eri.swapaxes(1, 3)
            )
            cross_spin_eri = (
                self._direct_cross_spin_eri
                if self._direct_cross_spin_eri is not None
                else spatial_eri
            )
            return _compute_diag_compact(
                spatial_h1,
                same_spin_eri,
                cross_spin_eri,
                self.binary,
            )

        if self.eri_so is None or self.hcore is None:
            raise ValueError("CASCI Hamiltonian data are not available for diagonal.")
        return _compute_diag(np.asarray(self.hcore), np.asarray(self.eri_so), self.binary)


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
        r"""
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

    def _direct_spin0_symm_solve(
        self,
        binary,
        requested_nstates,
        *,
        ci0=None,
        use_cholesky=None,
    ):
        """
        Solve singlet, symmetry-filtered CI without materializing the spin0 map.

        The dense helper remains the small-space reference.  For larger spaces we
        use the spin-string sigma kernel behind a symmetric alpha/beta string-pair
        interface, with a native C++ pair-space route available when requested.
        """
        if tuple(self.nelecas_spin)[0] != tuple(self.nelecas_spin)[1]:
            raise ValueError("direct_spin0_symm requires N_alpha == N_beta.")
        pairs = _rectangular_spin0_pairs(binary)
        if pairs is None:
            pairs = self._spin0_symm_pairs(binary)
        nspin0 = len(pairs)
        requested_nstates = int(requested_nstates)
        if requested_nstates > nspin0:
            raise ValueError(
                f"Requested {requested_nstates} spin0 roots but only "
                f"{nspin0} singlet configurations are available."
            )

        dense_limit = self.direct_spin0_symm_dense_fallback_nconfigs
        if dense_limit is not None and dense_limit > 0 and nspin0 <= dense_limit:
            if isinstance(binary, FCIStringBasis):
                binary = binary.materialize()
                self.binary = binary
            self.solver_backend = 'direct_spin0_symm_dense'
            result = self._direct_spin0_symm_dense(
                binary,
                requested_nstates,
                use_cholesky=use_cholesky,
            )
            self.direct_ci_diagnostics = {
                'backend': 'direct_spin0_symm_dense',
                'converged': True,
            }
            self.converged = True
            return result

        if self.spin_purification:
            raise ValueError("direct_spin0_symm already targets singlets; do not combine it with fix_spin().")

        self.binary = binary
        self.spin0_pair_indices = pairs
        self.spin0_symm_transform = None
        self.direct_connectivity = None
        left, right, same = _spin0_pair_arrays(pairs)
        ndet = binary.shape[0]
        spin0_mv = None
        spin0_mv_block = None
        spin0_native_solver = None
        det_mv_block = None

        # A restricted active-space ERI tensor is tiny compared with the CI
        # vector and preserves the spin-string factorization.  Expanding
        # Cholesky factors into determinant-level connection/value arrays can
        # otherwise require gigabytes already at CAS(10,10).
        factor_data = None
        if factor_data is not None:
            (
                spatial_h1,
                pair_factors,
                H_diag,
                H_A_factor,
                H_B_factor,
                H_AA_factor,
                H_BB_factor,
                H_AB_factor,
                energy_core,
            ) = factor_data
            h1a, h1b = _normalize_spin_1e_operator(spatial_h1)
            self.hcore = np.asarray([h1a, h1b])
            self.h2e_cas = None
            self.eri_so = None
            conn = self.direct_connectivity
            backend = 'direct_spin0_symm_davidson_factor_conn'

            def det_mv(c_det):
                return _sigma_values_conn_fast(
                    H_diag,
                    H_A_factor,
                    H_B_factor,
                    H_AA_factor,
                    H_BB_factor,
                    H_AB_factor,
                    c_det,
                    conn.I_A, conn.J_A,
                    conn.I_B, conn.J_B,
                    conn.I_AA, conn.J_AA,
                    conn.I_BB, conn.J_BB,
                    conn.I_AB, conn.J_AB,
                )

            def det_mv_block(c_det_block):
                return _sigma_values_conn_fast(
                    H_diag,
                    H_A_factor,
                    H_B_factor,
                    H_AA_factor,
                    H_BB_factor,
                    H_AB_factor,
                    c_det_block,
                    conn.I_A, conn.J_A,
                    conn.I_B, conn.J_B,
                    conn.I_AA, conn.J_AA,
                    conn.I_BB, conn.J_BB,
                    conn.I_AB, conn.J_AB,
                )

        else:
            spatial_h1, same_spin_eri, cross_spin_eri, energy_core = self.get_direct_compact_integrals(
                use_cholesky=use_cholesky
            )
            h1a, h1b = _normalize_spin_1e_operator(spatial_h1)
            self.hcore = np.asarray([h1a, h1b])
            self.h2e_cas = cross_spin_eri
            self.eri_so = None
            H_diag = _compute_diag_compact(spatial_h1, same_spin_eri, cross_spin_eri, binary)
            spin_string_conn = None
            try:
                spin_string_conn = build_spin_string_connectivity(binary)
            except ValueError:
                spin_string_conn = None

            if spin_string_conn is not None:
                self.spin_string_connectivity = spin_string_conn
                use_native_spin0_pair = bool(getattr(self, "direct_spin0_native_pair", False))
                backend = (
                    'direct_spin0_symm_davidson_spin0_pair'
                    if use_native_spin0_pair
                    else 'direct_spin0_symm_davidson_spin_string'
                )
                spin_string_alpha_cross_diag = _spin_string_cross_diagonal(
                    cross_spin_eri,
                    spin_string_conn.alpha_occ,
                )
                spin_string_beta_cross_diag = _spin_string_cross_diagonal(
                    cross_spin_eri,
                    spin_string_conn.beta_occ,
                )
                spin_string_workers = _resolve_direct_ci_workers(self, binary.shape[0])

                if use_native_spin0_pair:
                    spin0_mv = _make_sigma_compact_spin0_pair_cpp_matvec(
                        spatial_h1, same_spin_eri, cross_spin_eri, H_diag,
                        left, right,
                        spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                        spin_string_conn.I_A, spin_string_conn.J_A,
                        spin_string_conn.p_A, spin_string_conn.q_A,
                        spin_string_conn.phase_A,
                        spin_string_conn.I_B, spin_string_conn.J_B,
                        spin_string_conn.p_B, spin_string_conn.q_B,
                        spin_string_conn.phase_B,
                        spin_string_conn.I_AA, spin_string_conn.J_AA,
                        spin_string_conn.p_AA, spin_string_conn.q_AA,
                        spin_string_conn.r_AA, spin_string_conn.s_AA,
                        spin_string_conn.phase_AA,
                        spin_string_conn.I_BB, spin_string_conn.J_BB,
                        spin_string_conn.p_BB, spin_string_conn.q_BB,
                        spin_string_conn.r_BB, spin_string_conn.s_BB,
                        spin_string_conn.phase_BB,
                        spin_string_conn.alpha_offsets,
                        spin_string_conn.beta_offsets,
                        spin_string_conn.alpha_order,
                        spin_string_conn.beta_order,
                        spin_string_alpha_cross_diag,
                        spin_string_beta_cross_diag,
                        spin_string_conn.alpha_ordered_I,
                        spin_string_conn.alpha_ordered_J,
                        spin_string_conn.alpha_ordered_phase,
                        spin_string_conn.beta_ordered_I,
                        spin_string_conn.beta_ordered_J,
                        spin_string_conn.beta_ordered_phase,
                        spin_string_workers,
                    )
                    if spin0_mv is None:
                        def spin0_mv(c_spin0):
                            return _sigma_compact_spin0_pair(
                                spatial_h1, same_spin_eri, cross_spin_eri, H_diag, c_spin0,
                                left, right,
                                spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                                spin_string_conn.I_A, spin_string_conn.J_A,
                                spin_string_conn.p_A, spin_string_conn.q_A,
                                spin_string_conn.phase_A,
                                spin_string_conn.I_B, spin_string_conn.J_B,
                                spin_string_conn.p_B, spin_string_conn.q_B,
                                spin_string_conn.phase_B,
                                spin_string_conn.I_AA, spin_string_conn.J_AA,
                                spin_string_conn.p_AA, spin_string_conn.q_AA,
                                spin_string_conn.r_AA, spin_string_conn.s_AA,
                                spin_string_conn.phase_AA,
                                spin_string_conn.I_BB, spin_string_conn.J_BB,
                                spin_string_conn.p_BB, spin_string_conn.q_BB,
                                spin_string_conn.r_BB, spin_string_conn.s_BB,
                                spin_string_conn.phase_BB,
                                spin_string_conn.alpha_offsets,
                                spin_string_conn.beta_offsets,
                                spin_string_conn.alpha_order,
                                spin_string_conn.beta_order,
                                spin_string_alpha_cross_diag,
                                spin_string_beta_cross_diag,
                                spin_string_conn.alpha_ordered_I,
                                spin_string_conn.alpha_ordered_J,
                                spin_string_conn.alpha_ordered_phase,
                                spin_string_conn.beta_ordered_I,
                                spin_string_conn.beta_ordered_J,
                                spin_string_conn.beta_ordered_phase,
                                spin_string_workers,
                            )

                    def spin0_native_solver(
                        nroots, energy_tol, residual_tol, max_cycle, max_subspace
                    ):
                        return _davidson_spin0_pair_cpp(
                            spatial_h1, same_spin_eri, cross_spin_eri, H_diag,
                            left, right,
                            spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                            spin_string_conn.I_A, spin_string_conn.J_A,
                            spin_string_conn.p_A, spin_string_conn.q_A,
                            spin_string_conn.phase_A,
                            spin_string_conn.I_B, spin_string_conn.J_B,
                            spin_string_conn.p_B, spin_string_conn.q_B,
                            spin_string_conn.phase_B,
                            spin_string_conn.I_AA, spin_string_conn.J_AA,
                            spin_string_conn.p_AA, spin_string_conn.q_AA,
                            spin_string_conn.r_AA, spin_string_conn.s_AA,
                            spin_string_conn.phase_AA,
                            spin_string_conn.I_BB, spin_string_conn.J_BB,
                            spin_string_conn.p_BB, spin_string_conn.q_BB,
                            spin_string_conn.r_BB, spin_string_conn.s_BB,
                            spin_string_conn.phase_BB,
                            spin_string_conn.alpha_offsets,
                            spin_string_conn.beta_offsets,
                            spin_string_conn.alpha_order,
                            spin_string_conn.beta_order,
                            spin_string_alpha_cross_diag,
                            spin_string_beta_cross_diag,
                            spin_string_conn.alpha_ordered_I,
                            spin_string_conn.alpha_ordered_J,
                            spin_string_conn.alpha_ordered_phase,
                            spin_string_conn.beta_ordered_I,
                            spin_string_conn.beta_ordered_J,
                            spin_string_conn.beta_ordered_phase,
                            workers=spin_string_workers,
                            nroots=nroots,
                            energy_tol=energy_tol,
                            residual_tol=residual_tol,
                            max_cycle=max_cycle,
                            max_subspace=max_subspace,
                        )
                else:
                    def det_mv(c_det):
                        return _sigma_compact_spin_string(
                            spatial_h1, same_spin_eri, cross_spin_eri, H_diag, c_det,
                            spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                            spin_string_conn.I_A, spin_string_conn.J_A,
                            spin_string_conn.p_A, spin_string_conn.q_A,
                            spin_string_conn.phase_A,
                            spin_string_conn.I_B, spin_string_conn.J_B,
                            spin_string_conn.p_B, spin_string_conn.q_B,
                            spin_string_conn.phase_B,
                            spin_string_conn.I_AA, spin_string_conn.J_AA,
                            spin_string_conn.p_AA, spin_string_conn.q_AA,
                            spin_string_conn.r_AA, spin_string_conn.s_AA,
                            spin_string_conn.phase_AA,
                            spin_string_conn.I_BB, spin_string_conn.J_BB,
                            spin_string_conn.p_BB, spin_string_conn.q_BB,
                            spin_string_conn.r_BB, spin_string_conn.s_BB,
                            spin_string_conn.phase_BB,
                            spin_string_conn.alpha_offsets,
                            spin_string_conn.beta_offsets,
                            spin_string_conn.alpha_order,
                            spin_string_conn.beta_order,
                            spin_string_alpha_cross_diag,
                            spin_string_beta_cross_diag,
                            spin_string_conn.alpha_ordered_I,
                            spin_string_conn.alpha_ordered_J,
                            spin_string_conn.alpha_ordered_phase,
                            spin_string_conn.beta_ordered_I,
                            spin_string_conn.beta_ordered_J,
                            spin_string_conn.beta_ordered_phase,
                            spin_string_workers,
                        )

            else:
                self.direct_connectivity = build_direct_connectivity(binary)
                conn = self.direct_connectivity
                backend = 'direct_spin0_symm_davidson_compact_conn'

                def det_mv(c_det):
                    return _sigma_compact_conn_numba(
                        spatial_h1,
                        same_spin_eri,
                        cross_spin_eri,
                        H_diag,
                        c_det,
                        binary,
                        conn.I_A, conn.J_A, conn.p_A, conn.q_A, conn.phase_A,
                        conn.I_B, conn.J_B, conn.p_B, conn.q_B, conn.phase_B,
                        conn.I_AA, conn.J_AA, conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA,
                        conn.I_BB, conn.J_BB, conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB,
                        conn.I_AB, conn.J_AB, conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB,
                    )

        spin0_diag = _spin0_pair_diagonal(H_diag, left, right, same)

        if spin0_mv is None:
            def spin0_mv(c_spin0):
                c_det = _spin0_to_det_vector(c_spin0, left, right, same, ndet)
                return _det_to_spin0_vector(det_mv(c_det), left, right, same)
            if det_mv_block is not None:
                def spin0_mv_block(block_spin0):
                    block_spin0 = np.asarray(block_spin0)
                    det_block = np.column_stack([
                        _spin0_to_det_vector(block_spin0[:, root], left, right, same, ndet)
                        for root in range(block_spin0.shape[1])
                    ])
                    sigma_det = det_mv_block(det_block)
                    return np.column_stack([
                        _det_to_spin0_vector(sigma_det[:, root], left, right, same)
                        for root in range(sigma_det.shape[1])
                    ])

        guess_source = _select_direct_ci_guess(
            self,
            requested_nstates,
            ci0=ci0,
            reuse=self.direct_ci_reuse_guess,
        )
        guess = _project_spin0_guess(guess_source, left, right, same, ndet)
        davidson_energy_tol, davidson_residual_tol = _resolve_direct_ci_tolerances(self)
        native_result = None
        native_diagnostics = {
            'backend': 'native_spin0_davidson',
            'attempted': False,
            'used': False,
            'fallback_reason': None,
        }
        if (
            spin0_native_solver is not None
            and bool(getattr(self, "direct_spin0_native_davidson", False))
            and requested_nstates == 1
            and guess is None
        ):
            native_diagnostics['attempted'] = True
            native_result = spin0_native_solver(
                requested_nstates,
                davidson_energy_tol,
                davidson_residual_tol,
                self.direct_ci_max_cycle,
                self.direct_ci_max_subspace,
            )
            if native_result is not None:
                native_diagnostics.update({
                    'used': True,
                    'converged': True,
                })
            else:
                native_diagnostics['fallback_reason'] = 'native spin0 Davidson failed; using Python Davidson'
        elif spin0_native_solver is None:
            native_diagnostics['fallback_reason'] = 'native spin0 workspace is unavailable'
        elif not bool(getattr(self, "direct_spin0_native_davidson", False)):
            native_diagnostics['fallback_reason'] = 'native spin0 Davidson disabled by solver setting'
        elif requested_nstates != 1:
            native_diagnostics['fallback_reason'] = 'native spin0 Davidson currently supports one root'
        else:
            native_diagnostics['fallback_reason'] = 'native spin0 Davidson does not accept an initial guess'
        self.direct_ci_native_diagnostics = native_diagnostics
        self.direct_ci_fallback_reason = native_diagnostics.get('fallback_reason')
        if native_result is None:
            energies, vecs_spin0, diagnostics = davidson_lowest(
                spin0_mv,
                spin0_diag,
                nroots=requested_nstates,
                tol=davidson_residual_tol,
                energy_tol=davidson_energy_tol,
                max_cycle=self.direct_ci_max_cycle,
                max_subspace=self.direct_ci_max_subspace,
                guess=guess,
                matvec_block=spin0_mv_block,
                trial_block_size=(
                    self.direct_ci_factor_davidson_block_size
                    if factor_data is not None
                    else requested_nstates
                ),
                return_info=True,
            )
            diagnostics['native'] = native_diagnostics
            self.direct_ci_diagnostics = diagnostics
        else:
            energies, vecs_spin0 = native_result
            self.direct_ci_diagnostics = native_diagnostics
        self.converged = True
        self.solver_backend = backend
        vecs_det = np.column_stack(
            [_spin0_to_det_vector(vecs_spin0[:, i], left, right, same, ndet) for i in range(vecs_spin0.shape[1])]
        )
        return energies, vecs_det


    def run(
        self,
        nstates=1,
        mo_coeff=None,
        method='direct_ci',
        ci0=None,
        use_cholesky=None,
        spin_root_cushion=None,
        spin_selection_tol=None,
        wfnsym=None,
        target_irrep=None,
    ):
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
        self._direct_rhf_blas_matvec = None
        self._direct_ci_used_native_davidson = False
        self.direct_ci_native_diagnostics = {}
        self.direct_ci_diagnostics = {}
        self.direct_ci_fallback_reason = None
        self.converged = False
        requested_nstates = nstates
        self.use_cholesky_integrals = _resolve_use_cholesky_integrals(self.mf, use_cholesky)
        use_cholesky = self.use_cholesky_integrals
        method_key = str(method).lower().replace("-", "_")
        direct_spin0_methods = {"direct_spin0", "direct_spin0_symm", "spin0", "spin0_symm"}

        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff  # use HF MOs
        else:
            self.mo_coeff = mo_coeff

        uhf_reference = _is_uhf_reference(self.mo_coeff)

        if (
            method_key == 'direct_ci'
            and bool(getattr(self, 'direct_ci_auto_spin0', True))
            and self.multiplicity == 1
            and not uhf_reference
            and self.nelecas_spin[0] == self.nelecas_spin[1]
        ):
            method_key = 'direct_spin0_symm'

        if method_key in direct_spin0_methods:
            if uhf_reference:
                raise NotImplementedError("direct_spin0_symm currently supports restricted references only.")

            ncore = self.ncore
            ncas = self.ncas
            self.mo_core, self.mo_cas = _slice_active_orbitals(self.mo_coeff, ncore, ncas)

            if self.binary is None:
                mo_occ = _reference_active_occupations(self.nelecas_spin, ncas)
                binary = get_fci_string_basis(mo_occ=mo_occ)
                self.binary = binary
            else:
                binary = self.binary
            binary = self._filter_binary_by_irrep(binary, target_irrep=target_irrep, wfnsym=wfnsym)
            requested_nstates = int(requested_nstates)
            E, X = self._direct_spin0_symm_solve(
                binary,
                requested_nstates,
                ci0=ci0,
                use_cholesky=use_cholesky,
            )

        elif method_key == 'ci':
            self.solver_backend = 'ci'

            # define the core and active space orbitals
            ncore = self.ncore
            ncas = self.ncas

            self.mo_core, self.mo_cas = _slice_active_orbitals(self.mo_coeff, ncore, ncas)

            # FCI solver, more efficient than the JW solver

            mo_occ = _reference_active_occupations(self.nelecas_spin, ncas)
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary
            binary = self._filter_binary_by_irrep(binary, target_irrep=target_irrep, wfnsym=wfnsym)
            solve_nstates = self._spin_selected_nstates(
                requested_nstates,
                binary.shape[0],
                spin_root_cushion=spin_root_cushion,
            )


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


            if solve_nstates >= H_CI.shape[0]:
                E, X = self._lowest_dense_eigensystem(H_CI, solve_nstates)
            else:
                E, X = eigsh(H_CI, k=solve_nstates, which='SA')

            # from pyqed.davidson import davidson

            # E, X = davidson(H_CI, nstates, tol=1e-13)

        elif method_key == 'direct_ci':

            ncore = self.ncore
            ncas = self.ncas

            self.mo_core, self.mo_cas = _slice_active_orbitals(self.mo_coeff, ncore, ncas)

            # FCI solver, more efficient than the JW solver
            if self.binary is None:
                mo_occ = _reference_active_occupations(self.nelecas_spin, ncas)
                binary = get_fci_string_basis(mo_occ=mo_occ)
                self.binary = binary

            else:
                binary = self.binary
            binary = self._filter_binary_by_irrep(binary, target_irrep=target_irrep, wfnsym=wfnsym)
            solve_nstates = self._spin_selected_nstates(
                requested_nstates,
                binary.shape[0],
                spin_root_cushion=spin_root_cushion,
            )
            if self.multiplicity is None and requested_nstates > 1:
                extra_roots = max(0, int(getattr(self, 'direct_ci_root_cushion', 0)))
                extra_roots = min(extra_roots, max(0, binary.shape[0] - requested_nstates))
                solve_nstates = requested_nstates + extra_roots


            factor_data = None
            if self.spin_purification:
                # The first-order spin-penalty code is currently expressed in the
                # older spin-block Hamiltonian form, so we keep that path until
                # the penalty is rewritten in terms of spatial integrals.
                h1e, h2e = self.get_SO_matrix(use_cholesky=use_cholesky)
                h1e = np.asarray(h1e)
                spatial_h1 = None
                same_spin_eri = None
                cross_spin_eri = None
                spatial_eri = None
                uhf_compact = None
            else:
                # Keep restricted Cholesky calculations in the compact
                # spin-string representation.  The active ERI tensor costs
                # only O(ncas**4), whereas determinant-expanded factor
                # connectivity scales with every single and double excitation.
                factor_data = None
                if uhf_reference:
                    h1_spin, eri_blocks, energy_core = self.get_direct_spatial_integrals(
                        use_cholesky=use_cholesky
                    )
                    h1a, h1b = _normalize_spin_1e_operator(h1_spin)
                    eri_aa, eri_ab, eri_ba, eri_bb = eri_blocks
                    uhf_compact = (
                        np.asarray(h1a),
                        np.asarray(h1b),
                        np.asarray(eri_aa) - np.asarray(eri_aa).swapaxes(1, 3),
                        np.asarray(eri_ab),
                        np.asarray(eri_ba),
                        np.asarray(eri_bb) - np.asarray(eri_bb).swapaxes(1, 3),
                    )
                    self._direct_uhf_compact_integrals = uhf_compact
                    same_spin_eri = None
                    cross_spin_eri = None
                    spatial_eri = None
                    spatial_h1 = None
                    h1e = np.asarray([h1a, h1b])
                    h2e = None
                else:
                    uhf_compact = None
                    spatial_h1, same_spin_eri, cross_spin_eri, energy_core = self.get_direct_compact_integrals(
                        use_cholesky=use_cholesky
                    )
                    spatial_eri = cross_spin_eri
                    h1a, h1b = _normalize_spin_1e_operator(spatial_h1)
                    h1e = np.asarray([h1a, h1b])
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
            spin_string_conn = None
            use_spin_string_backend = (
                factor_data is None
                and (spatial_eri is not None or uhf_compact is not None)
                and not self.spin_purification
                and getattr(self, 'det_irrep_filter_indices', None) is None
                and bool(getattr(self, 'direct_ci_spin_string_backend', True))
            )
            if use_spin_string_backend:
                if self.spin_string_connectivity is None:
                    self.spin_string_connectivity = build_spin_string_connectivity(binary)
                spin_string_conn = self.spin_string_connectivity

            if (
                factor_data is None
                and
                not uhf_reference
                and
                self.direct_ci_dense_fallback_ndets is not None
                and self.direct_ci_dense_fallback_ndets > 0
                and binary.shape[0] <= self.direct_ci_dense_fallback_ndets
            ):
                # For small determinant spaces the dense/vectorized CI builder is
                # faster than paying the Numba JIT + setup overhead of direct_ci.
                self.solver_backend = 'ci_dense_fallback'
                if isinstance(binary, FCIStringBasis):
                    binary = binary.materialize()
                    self.binary = binary
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
                if solve_nstates >= H_CI.shape[0]:
                    E, X = self._lowest_dense_eigensystem(H_CI, solve_nstates)
                else:
                    E, X = eigsh(H_CI, k=solve_nstates, which='SA')
                self.direct_ci_diagnostics = {
                    'backend': 'ci_dense_fallback',
                    'converged': True,
                }
            else:
                if spin_string_conn is not None:
                    self.solver_backend = (
                        'direct_ci_spin_string_uhf' if uhf_compact is not None
                        else 'direct_ci_spin_string'
                    )
                else:
                    self.solver_backend = 'direct_ci_compact_conn' if spatial_eri is not None else 'direct_ci'
                # The diagonal is reused in every matvec, so it is worth
                # computing once up front even in the matrix-free solver.
                if uhf_compact is not None:
                    H_diag = _compute_diag_compact_uhf(*uhf_compact, binary)
                else:
                    H_diag = _compute_diag_compact(spatial_h1, same_spin_eri, cross_spin_eri, binary) if spatial_eri is not None else _compute_diag(h1e, h2e, binary)
                self._direct_H_diag = H_diag
                conn = None if spin_string_conn is not None else self.direct_connectivity
                if conn is None and spin_string_conn is None:
                    self.direct_connectivity = build_direct_connectivity(binary)
                    conn = self.direct_connectivity
                spin_string_alpha_cross_diag = None
                spin_string_beta_cross_diag = None
                spin_string_workers = 1
                if (
                    spin_string_conn is not None
                    and cross_spin_eri is not None
                    and not np.iscomplexobj(cross_spin_eri)
                ):
                    spin_string_workers = _resolve_direct_ci_workers(self, binary.shape[0])
                    spin_string_alpha_cross_diag = _spin_string_cross_diagonal(
                        cross_spin_eri,
                        spin_string_conn.alpha_occ,
                    )
                    spin_string_beta_cross_diag = _spin_string_cross_diagonal(
                        cross_spin_eri,
                        spin_string_conn.beta_occ,
                    )
                    self._spin_string_alpha_cross_diag = spin_string_alpha_cross_diag
                    self._spin_string_beta_cross_diag = spin_string_beta_cross_diag

                rhf_blas_mv = None
                if (
                    spin_string_conn is not None
                    and cross_spin_eri is not None
                    and not np.iscomplexobj(spatial_h1)
                ):
                    spin0_pairs = _rectangular_spin0_pairs(binary)
                    if spin0_pairs is not None:
                        pair_left, pair_right, _ = _spin0_pair_arrays(spin0_pairs)
                        rhf_blas_mv = _make_sigma_compact_rhf_blas_cpp_matvec(
                            spatial_h1, same_spin_eri, cross_spin_eri, H_diag,
                            pair_left, pair_right,
                            spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                            spin_string_conn.I_A, spin_string_conn.J_A,
                            spin_string_conn.p_A, spin_string_conn.q_A,
                            spin_string_conn.phase_A,
                            spin_string_conn.I_B, spin_string_conn.J_B,
                            spin_string_conn.p_B, spin_string_conn.q_B,
                            spin_string_conn.phase_B,
                            spin_string_conn.I_AA, spin_string_conn.J_AA,
                            spin_string_conn.p_AA, spin_string_conn.q_AA,
                            spin_string_conn.r_AA, spin_string_conn.s_AA,
                            spin_string_conn.phase_AA,
                            spin_string_conn.I_BB, spin_string_conn.J_BB,
                            spin_string_conn.p_BB, spin_string_conn.q_BB,
                            spin_string_conn.r_BB, spin_string_conn.s_BB,
                            spin_string_conn.phase_BB,
                            spin_string_conn.alpha_offsets,
                            spin_string_conn.beta_offsets,
                            spin_string_conn.alpha_order,
                            spin_string_conn.beta_order,
                            spin_string_alpha_cross_diag,
                            spin_string_beta_cross_diag,
                            spin_string_conn.alpha_ordered_I,
                            spin_string_conn.alpha_ordered_J,
                            spin_string_conn.alpha_ordered_phase,
                            spin_string_conn.beta_ordered_I,
                            spin_string_conn.beta_ordered_J,
                            spin_string_conn.beta_ordered_phase,
                            spin_string_workers,
                        )
                self._direct_rhf_blas_matvec = rhf_blas_mv

                mv_block = None

                def mv(c):
                    # Keep the Lanczos matvec almost entirely inside compiled
                    # code. The Python closure only forwards cached arrays.
                    if spin_string_conn is not None:
                        if uhf_compact is not None:
                            return _sigma_compact_spin_string_uhf_numba(
                                *uhf_compact, H_diag, c,
                                spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                                spin_string_conn.I_A, spin_string_conn.J_A,
                                spin_string_conn.p_A, spin_string_conn.q_A,
                                spin_string_conn.phase_A,
                                spin_string_conn.I_B, spin_string_conn.J_B,
                                spin_string_conn.p_B, spin_string_conn.q_B,
                                spin_string_conn.phase_B,
                                spin_string_conn.I_AA, spin_string_conn.J_AA,
                                spin_string_conn.p_AA, spin_string_conn.q_AA,
                                spin_string_conn.r_AA, spin_string_conn.s_AA,
                                spin_string_conn.phase_AA,
                                spin_string_conn.I_BB, spin_string_conn.J_BB,
                                spin_string_conn.p_BB, spin_string_conn.q_BB,
                                spin_string_conn.r_BB, spin_string_conn.s_BB,
                                spin_string_conn.phase_BB,
                            )
                        if rhf_blas_mv is not None and not np.iscomplexobj(c):
                            return rhf_blas_mv(c)
                        return _sigma_compact_spin_string(
                            spatial_h1, same_spin_eri, cross_spin_eri, H_diag, c,
                            spin_string_conn.alpha_occ, spin_string_conn.beta_occ,
                            spin_string_conn.I_A, spin_string_conn.J_A,
                            spin_string_conn.p_A, spin_string_conn.q_A,
                            spin_string_conn.phase_A,
                            spin_string_conn.I_B, spin_string_conn.J_B,
                            spin_string_conn.p_B, spin_string_conn.q_B,
                            spin_string_conn.phase_B,
                            spin_string_conn.I_AA, spin_string_conn.J_AA,
                            spin_string_conn.p_AA, spin_string_conn.q_AA,
                            spin_string_conn.r_AA, spin_string_conn.s_AA,
                            spin_string_conn.phase_AA,
                            spin_string_conn.I_BB, spin_string_conn.J_BB,
                            spin_string_conn.p_BB, spin_string_conn.q_BB,
                            spin_string_conn.r_BB, spin_string_conn.s_BB,
                            spin_string_conn.phase_BB,
                            spin_string_conn.alpha_offsets,
                            spin_string_conn.beta_offsets,
                            spin_string_conn.alpha_order,
                            spin_string_conn.beta_order,
                            spin_string_alpha_cross_diag,
                            spin_string_beta_cross_diag,
                            spin_string_conn.alpha_ordered_I,
                            spin_string_conn.alpha_ordered_J,
                            spin_string_conn.alpha_ordered_phase,
                            spin_string_conn.beta_ordered_I,
                            spin_string_conn.beta_ordered_J,
                            spin_string_conn.beta_ordered_phase,
                            spin_string_workers,
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

                eigensolver = self.direct_ci_eigensolver
                if eigensolver == 'auto':
                    auto_eigsh_ndets = self.direct_ci_auto_eigsh_ndets
                    if (
                        auto_eigsh_ndets is not None
                        and auto_eigsh_ndets > 0
                        and binary.shape[0] <= auto_eigsh_ndets
                    ):
                        eigensolver = 'eigsh'
                    else:
                        eigensolver = 'davidson'

                if eigensolver == 'davidson':
                    # Use a CI-specific Davidson iteration by default.  The
                    # direct-CI backend already has a cheap diagonal
                    # preconditioner and only needs a few low roots, which is
                    # exactly the regime where Davidson is preferable to a
                    # generic sparse eigensolver.
                    guess = _select_direct_ci_guess(
                        self,
                        solve_nstates,
                        ci0=ci0,
                        reuse=self.direct_ci_reuse_guess,
                    )
                    davidson_energy_tol, davidson_residual_tol = (
                        _resolve_direct_ci_tolerances(self)
                    )
                    native_result = None
                    native_diagnostics = {
                        'backend': 'native_rhf_davidson',
                        'attempted': False,
                        'used': False,
                        'fallback_reason': None,
                    }
                    if not bool(getattr(self, "direct_ci_native_davidson", True)):
                        native_diagnostics['fallback_reason'] = 'native Davidson disabled by solver setting'
                    elif guess is not None and np.iscomplexobj(np.asarray(guess)):
                        native_diagnostics['fallback_reason'] = 'real native Davidson does not accept a complex CI guess'
                    elif rhf_blas_mv is None:
                        native_diagnostics['fallback_reason'] = (
                            'native packed workspace requires real balanced restricted spin strings'
                        )
                    else:
                        native_result, native_diagnostics = _davidson_rhf_blas_cpp(
                            rhf_blas_mv,
                            H_diag,
                            guess=guess,
                            nroots=solve_nstates,
                            energy_tol=davidson_energy_tol,
                            residual_tol=davidson_residual_tol,
                            max_cycle=self.direct_ci_max_cycle,
                            max_subspace=self.direct_ci_max_subspace,
                        )
                    self.direct_ci_native_diagnostics = native_diagnostics
                    self.direct_ci_fallback_reason = native_diagnostics.get('fallback_reason')
                    if native_result is None:
                        E, X, python_diagnostics = davidson_lowest(
                            mv,
                            H_diag,
                            nroots=solve_nstates,
                            tol=davidson_residual_tol,
                            energy_tol=davidson_energy_tol,
                            max_cycle=self.direct_ci_max_cycle,
                            max_subspace=self.direct_ci_max_subspace,
                            guess=guess,
                            matvec_block=mv_block,
                            trial_block_size=(
                                self.direct_ci_factor_davidson_block_size
                                if factor_data is not None
                                else solve_nstates
                            ),
                            return_info=True,
                        )
                        self.direct_ci_diagnostics = python_diagnostics
                        self.direct_ci_diagnostics['native'] = native_diagnostics
                    else:
                        E, X = native_result
                        self._direct_ci_used_native_davidson = True
                        self.direct_ci_diagnostics = native_diagnostics
                    self.converged = True
                elif eigensolver == 'eigsh':
                    H = LinearOperator(
                        (binary.shape[0], binary.shape[0]),
                        matvec=mv,
                        dtype=np.complex128,
                    )
                    E, X = eigsh(H, k=solve_nstates, which='SA', tol=self.tol)
                    self.direct_ci_diagnostics = {
                        'backend': 'eigsh',
                        'converged': True,
                    }
                    self.converged = True
                else:
                    raise ValueError(
                        "Unknown direct_ci eigensolver '{}'. Use 'auto', 'davidson' or 'eigsh'.".format(
                            eigensolver
                        )
                    )




        elif method_key == 'jw':
            self.solver_backend = 'jw'


            # exact diagonalization by JW transform

            H = self.qubitization()
            E, X = eigsh(H, k=nstates, which='SA')

        else:
            raise ValueError(
                "There is no {} solver for CASCI. Use 'ci', 'direct_ci', "
                "'direct_spin0_symm', or 'jw'.".format(method)
            )

        E, X = self._apply_multiplicity_selection(
            E,
            X,
            requested_nstates,
            spin_selection_tol=spin_selection_tol,
        )

        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(requested_nstates)]
        self.converged = True
        if not self.direct_ci_diagnostics:
            self.direct_ci_diagnostics = {
                'backend': self.solver_backend,
                'converged': True,
            }

        if self.verbose >= 1:
            for i in range(requested_nstates):
                ss = self.spin_square(i)
                print("CASCI Root {}  E = {:.10f}  S^2 = {:.6f}".format(i, self.e_tot[i], ss))

        return self

    def make_rdm1_contract(self, state_id, h1e=None, representation='ao'):
        r"""
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
        r"""
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
        sc1 = getattr(self, "SC1", None)
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
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, sc1)

            return D

        if with_core and with_vir:

            D = np.zeros((nmo, nmo), dtype=float)
            if ncore > 0:
                for i in range(ncore):
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, sc1)

            return D
        else:
            return make_rdm1(ci, self.binary, sc1)

    def make_rdm1s(self, state_id):
        r"""
        spin-polarized 1e reduced density matrix
        .. math::

            \gamma_s[p,q] = <q_s^\dagger p_s>


        Returns
        -------
        None.

        """
        ci = self.ci[state_id]
        return mcscf.casci.make_rdm1s(ci, self.binary, getattr(self, "SC1", None))

    def make_rdm2(self, state_id=0, with_core=False, with_vir=False):
        r"""
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
        sc1 = getattr(self, "SC1", None)
        sc2 = getattr(self, "SC2", None)


        if with_core: # we probably never need this!

            ncore = self.ncore
            ncas = self.ncas
            # nmo = self.mf.nmo
            nmo = ncore + ncas

            if ncore == 0:
                return make_rdm2(ci, self.binary, sc1, sc2)

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
                make_rdm2(ci, self.binary, sc1, sc2)

            return D

        else: #active space DM

            return make_rdm2(ci, self.binary, sc1, sc2)


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

        spin_conn = self.spin_string_connectivity
        if spin_conn is not None and isinstance(self.binary, FCIStringBasis):
            if self._s2_operator is None:
                h1_s2, h2_s2 = build_spin_square_operator(self.ncas)
                self._s2_operator = (
                    h1_s2[0],
                    h2_s2[0, 0].copy(),
                    h2_s2[0, 1].copy(),
                )
                self._s2_diag = _compute_diag_compact(
                    self._s2_operator[0],
                    self._s2_operator[1],
                    self._s2_operator[2],
                    self.binary,
                )
            alpha_cross = _spin_string_cross_diagonal(
                self._s2_operator[2], spin_conn.alpha_occ
            )
            beta_cross = _spin_string_cross_diagonal(
                self._s2_operator[2], spin_conn.beta_occ
            )
            sigma_s2 = _sigma_compact_spin_string(
                self._s2_operator[0], self._s2_operator[1], self._s2_operator[2],
                self._s2_diag, ci,
                spin_conn.alpha_occ, spin_conn.beta_occ,
                spin_conn.I_A, spin_conn.J_A, spin_conn.p_A, spin_conn.q_A, spin_conn.phase_A,
                spin_conn.I_B, spin_conn.J_B, spin_conn.p_B, spin_conn.q_B, spin_conn.phase_B,
                spin_conn.I_AA, spin_conn.J_AA,
                spin_conn.p_AA, spin_conn.q_AA, spin_conn.r_AA, spin_conn.s_AA, spin_conn.phase_AA,
                spin_conn.I_BB, spin_conn.J_BB,
                spin_conn.p_BB, spin_conn.q_BB, spin_conn.r_BB, spin_conn.s_BB, spin_conn.phase_BB,
                spin_conn.alpha_offsets, spin_conn.beta_offsets,
                spin_conn.alpha_order, spin_conn.beta_order,
                alpha_cross, beta_cross,
                spin_conn.alpha_ordered_I, spin_conn.alpha_ordered_J, spin_conn.alpha_ordered_phase,
                spin_conn.beta_ordered_I, spin_conn.beta_ordered_J, spin_conn.beta_ordered_phase,
                _resolve_direct_ci_workers(self, self.binary.shape[0]),
            )
            return float(np.vdot(ci, sigma_s2).real)

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
        r"""
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

            if self.verbose >= 1:
                print("CI ket and bra are the same. Computing 1e RDM instead.")
            return self.make_rdm1(ket_id, h1e)

        else:

            if representation.lower() == 'ao':
                C = self.mf.mo_coeff
                h1e = ao2mo(h1e, C)

            ncore = self.ncore
            ncas = self.ncas

            if ncore > 0:
                c_core = 2 * np.trace(h1e[:ncore,:ncore]) * np.vdot(
                    self.ci[bra_id],
                    self.ci[ket_id],
                )
            else:
                c_core = 0

            h1e = h1e[ncore:ncas+ncore, ncore:ncas+ncore]


            c_cas = contract_with_tdm1(
                self.ci[bra_id],
                self.ci[ket_id],
                self.binary,
                self.SC1,
                h1e,
            )

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

        return make_tdm1(cibra, ciket, self.binary, getattr(self, "SC1", None))

    def make_tdm2(self, bra_id, ket_id=0):
        r"""
        Spin-traced two-particle transition density matrix in MO basis.

        .. math::

            \Gamma_{pqrs}^{\beta \alpha}
            = \sum_{\sigma\tau}<\Psi_\beta|
              p^\dagger_\sigma r^\dagger_\tau s_\tau q_\sigma
              |\Psi_\alpha>
        """
        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]
        return _make_tdm2_dense(
            cibra,
            ciket,
            self.binary,
            getattr(self, "SC1", None),
            getattr(self, "SC2", None),
        )





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
    r"""

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
    r"""

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
    r"""

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


    return mcscf.casci.make_rdm1(ci, binary, SC1)

def make_tdm1(cibra, ciket, binary, SC1):
    r"""

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


    return mcscf.casci.make_tdm1(cibra, ciket, binary, SC1)





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
    r"""
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
    return _make_tdm2_dense(ci, ci, Binary, SC1, SC2)

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
