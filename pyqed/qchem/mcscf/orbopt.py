#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight orbital-optimization helpers for native first-order CASSCF.
"""

import numpy as np
from scipy.linalg import expm


def embed_rdm2(dm2, nmo):
    """
    Embed a core+active 2-RDM into the full MO space with zero virtual blocks.
    """
    dm2 = np.asarray(dm2)
    if dm2.ndim != 4 or len(set(dm2.shape)) != 1:
        raise ValueError("dm2 must have shape (n, n, n, n).")
    if dm2.shape[0] > nmo:
        raise ValueError("dm2 cannot be larger than the full MO space.")
    full = np.zeros((nmo, nmo, nmo, nmo), dtype=dm2.dtype)
    nocc_like = dm2.shape[0]
    full[:nocc_like, :nocc_like, :nocc_like, :nocc_like] = dm2
    return full


def generalized_fock(h1_mo, eri_mo, dm1, dm2):
    """
    Build a simple generalized Fock matrix from full-space MO integrals and RDMs.
    """
    h1_mo = np.asarray(h1_mo)
    eri_mo = np.asarray(eri_mo)
    dm1 = np.asarray(dm1)
    dm2 = np.asarray(dm2)
    f1 = np.einsum("pr,rq->pq", h1_mo, dm1, optimize=True)
    f2 = np.einsum("prst,rqst->pq", eri_mo, dm2, optimize=True)
    return f1 + f2


def orbital_gradient(fock):
    """
    Anti-Hermitian orbital gradient from the generalized Fock matrix.
    """
    fock = np.asarray(fock)
    return 2.0 * (fock - fock.conj().T)


def nonredundant_pairs(ncore, ncas, nmo):
    """
    Return the core-active, core-virtual, and active-virtual rotation pairs.
    """
    core = np.arange(0, ncore, dtype=int)
    act = np.arange(ncore, ncore + ncas, dtype=int)
    vir = np.arange(ncore + ncas, nmo, dtype=int)
    pairs = []
    for pset, qset in ((core, act), (core, vir), (act, vir)):
        for p in pset:
            for q in qset:
                pairs.append((int(p), int(q)))
    return pairs


def gradient_norm(gradient, ncore, ncas, nmo):
    """
    Infinity norm over the nonredundant orbital-rotation blocks.
    """
    pairs = nonredundant_pairs(ncore, ncas, nmo)
    if not pairs:
        return 0.0
    vals = [abs(gradient[p, q]) for p, q in pairs]
    return float(np.max(vals))


def pack_nonredundant(matrix, ncore, ncas, nmo):
    """Pack the independent orbital-rotation entries into a vector."""
    pairs = nonredundant_pairs(ncore, ncas, nmo)
    if not pairs:
        return np.zeros(0, dtype=float)
    return np.array([np.real(matrix[p, q]) for p, q in pairs], dtype=float)


def unpack_nonredundant(vec, ncore, ncas, nmo, max_step=None):
    """Unpack a vector of orbital-rotation parameters into an anti-Hermitian matrix."""
    pairs = nonredundant_pairs(ncore, ncas, nmo)
    kappa = np.zeros((nmo, nmo), dtype=float)
    if len(pairs) == 0:
        return kappa

    vec = np.asarray(vec, dtype=float)
    if max_step is not None:
        vec = np.clip(vec, -max_step, max_step)

    for value, (p, q) in zip(vec, pairs):
        kappa[p, q] = value
        kappa[q, p] = -value
    return kappa


def diagonal_denominator(fock, ncore, ncas, level_shift=1.0e-3):
    """
    Diagonal orbital Hessian approximation from generalized-Fock diagonals.
    """
    fock = np.asarray(fock)
    nmo = fock.shape[0]
    denom = np.zeros((nmo, nmo), dtype=float)
    diag = np.real(np.diag(0.5 * (fock + fock.conj().T)))
    for p, q in nonredundant_pairs(ncore, ncas, nmo):
        val = 2.0 * (diag[q] - diag[p])
        if abs(val) < level_shift:
            val = np.copysign(level_shift, val if val != 0.0 else 1.0)
        denom[p, q] = val
        denom[q, p] = -val
    return denom


def diagonal_preconditioned_vector(gradient, fock, ncore, ncas, level_shift=1.0e-3):
    """Return the diagonal-preconditioned nonredundant orbital step vector."""
    denom = diagonal_denominator(fock, ncore, ncas, level_shift=level_shift)
    pairs = nonredundant_pairs(ncore, ncas, fock.shape[0])
    if not pairs:
        return np.zeros(0, dtype=float)
    return np.array(
        [(-np.real(gradient[p, q]) / denom[p, q]) for p, q in pairs],
        dtype=float,
    )


def diagonal_inverse_hessian(fock, ncore, ncas, level_shift=1.0e-3):
    """Return the diagonal inverse-Hessian approximation in packed coordinates."""
    denom = diagonal_denominator(fock, ncore, ncas, level_shift=level_shift)
    pairs = nonredundant_pairs(ncore, ncas, fock.shape[0])
    if not pairs:
        return np.zeros(0, dtype=float)
    vals = []
    for p, q in pairs:
        d = np.real(denom[p, q])
        if abs(d) < level_shift:
            d = np.copysign(level_shift, d if d != 0.0 else 1.0)
        vals.append(1.0 / abs(d))
    return np.array(vals, dtype=float)


def limit_step_norm(step_vec, max_step):
    """Uniformly rescale a step so no packed component exceeds ``max_step``."""
    step_vec = np.asarray(step_vec, dtype=float)
    if step_vec.size == 0:
        return step_vec
    peak = np.max(np.abs(step_vec))
    if peak <= max_step or peak == 0.0:
        return step_vec
    return step_vec * (max_step / peak)


def lbfgs_direction(grad_vec, s_history, y_history, h0_diag=None):
    """Standard two-loop recursion for limited-memory BFGS in Euclidean coordinates."""
    grad_vec = np.asarray(grad_vec, dtype=float)
    if len(s_history) == 0:
        if h0_diag is None:
            return grad_vec.copy()
        return np.asarray(h0_diag, dtype=float) * grad_vec

    q = grad_vec.copy()
    alpha = []
    rho = []

    for s_vec, y_vec in zip(reversed(s_history), reversed(y_history)):
        sy = float(np.dot(s_vec, y_vec))
        if abs(sy) < 1.0e-16:
            alpha.append(0.0)
            rho.append(0.0)
            continue
        rho_i = 1.0 / sy
        alpha_i = rho_i * float(np.dot(s_vec, q))
        q = q - alpha_i * y_vec
        alpha.append(alpha_i)
        rho.append(rho_i)

    s_last = s_history[-1]
    y_last = y_history[-1]
    yy = float(np.dot(y_last, y_last))
    sy = float(np.dot(s_last, y_last))
    if h0_diag is not None:
        r = np.asarray(h0_diag, dtype=float) * q
    else:
        gamma = sy / yy if yy > 1.0e-16 else 1.0
        r = gamma * q

    for idx, (s_vec, y_vec) in enumerate(zip(s_history, y_history)):
        rho_i = rho[-1 - idx]
        alpha_i = alpha[-1 - idx]
        if rho_i == 0.0:
            continue
        beta = rho_i * float(np.dot(y_vec, r))
        r = r + s_vec * (alpha_i - beta)

    return r


def update_lbfgs_history(s_history, y_history, step_vec, grad_diff, history_size):
    """Append one secant pair if it satisfies the curvature condition."""
    step_vec = np.asarray(step_vec, dtype=float)
    grad_diff = np.asarray(grad_diff, dtype=float)
    curvature = float(np.dot(step_vec, grad_diff))
    if curvature <= 1.0e-12:
        return

    s_history.append(step_vec.copy())
    y_history.append(grad_diff.copy())
    if len(s_history) > history_size:
        del s_history[0]
        del y_history[0]


def orbital_step(
    fock,
    ncore,
    ncas,
    step_size=1.0,
    level_shift=1.0e-3,
    max_step=0.25,
):
    """
    Build a diagonal-preconditioned anti-Hermitian orbital-rotation step.
    """
    grad = orbital_gradient(fock)
    denom = diagonal_denominator(fock, ncore, ncas, level_shift=level_shift)
    kappa = np.zeros_like(fock, dtype=complex)
    for p, q in nonredundant_pairs(ncore, ncas, fock.shape[0]):
        step = -step_size * grad[p, q] / denom[p, q]
        step = np.clip(step.real, -max_step, max_step)
        kappa[p, q] = step
        kappa[q, p] = -step
    return kappa, grad


def rotate_orbitals(mo_coeff, kappa):
    """
    Rotate MO coefficients with an anti-Hermitian generator.
    """
    mo_coeff = np.asarray(mo_coeff)
    kappa = np.asarray(kappa)
    return np.real_if_close(mo_coeff @ expm(kappa))
