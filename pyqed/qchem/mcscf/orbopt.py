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


def generalized_fock_from_factors(h1_mo, pair_factors_full_occ, dm1_occ, dm2_occ):
    """
    Build the generalized Fock matrix from MO Cholesky pair factors.

    Parameters
    ----------
    h1_mo : ndarray, shape (nmo, nmo)
        One-electron Hamiltonian in the current MO basis.
    pair_factors_full_occ : ndarray, shape (naux, nmo, nocc_like)
        Cholesky pair factors ``B[P,p,r]`` transformed with the full MO basis on
        the left and the core+active orbitals on the right.
    dm1_occ : ndarray, shape (nocc_like, nocc_like)
        Spin-traced 1-RDM including core occupations.
    dm2_occ : ndarray, shape (nocc_like, nocc_like, nocc_like, nocc_like)
        Spin-traced 2-RDM in the same core+active space.

    Returns
    -------
    ndarray
        Generalized Fock matrix in the full MO basis. Only the core+active
        columns are non-zero, which matches the dense full-space contraction for
        an embedded CAS RDM with zero virtual blocks.
    """
    h1_mo = np.asarray(h1_mo)
    pair_factors_full_occ = np.asarray(pair_factors_full_occ)
    dm1_occ = np.asarray(dm1_occ)
    dm2_occ = np.asarray(dm2_occ)

    nmo = h1_mo.shape[0]
    nocc_like = dm1_occ.shape[0]
    if pair_factors_full_occ.ndim != 3:
        raise ValueError("pair_factors_full_occ must have shape (naux, nmo, nocc_like).")
    if pair_factors_full_occ.shape[1] != nmo or pair_factors_full_occ.shape[2] != nocc_like:
        raise ValueError("pair_factors_full_occ is inconsistent with h1_mo/dm1_occ.")
    if dm2_occ.shape != (nocc_like, nocc_like, nocc_like, nocc_like):
        raise ValueError("dm2_occ must have shape (nocc_like, nocc_like, nocc_like, nocc_like).")

    # Only the core+active columns contribute because the CAS density has no
    # virtual blocks in this spin-free formulation.
    fock_occ = np.einsum("pr,rq->pq", h1_mo[:, :nocc_like], dm1_occ, optimize=True)

    pair_factors_occ_occ = pair_factors_full_occ[:, :nocc_like, :]
    contracted_dm2 = np.einsum(
        "Pst,rqst->Prq", pair_factors_occ_occ, dm2_occ, optimize=True
    )
    fock_occ += np.einsum(
        "Ppr,Prq->pq", pair_factors_full_occ, contracted_dm2, optimize=True
    )

    fock = np.zeros_like(h1_mo, dtype=np.result_type(fock_occ, h1_mo))
    fock[:, :nocc_like] = fock_occ
    return fock


def orbital_h1_response(h1_mo, kappa):
    """First-order one-electron integral response to an orbital rotation."""
    h1_mo = np.asarray(h1_mo)
    kappa = np.asarray(kappa)
    return h1_mo @ kappa - kappa @ h1_mo


def orbital_eri_response(eri_mo, kappa):
    """First-order two-electron MO-integral response to an orbital rotation."""
    eri_mo = np.asarray(eri_mo)
    kappa = np.asarray(kappa)
    deri = np.einsum("ap,aqrs->pqrs", kappa, eri_mo, optimize=True)
    deri += np.einsum("bq,pbrs->pqrs", kappa, eri_mo, optimize=True)
    deri += np.einsum("cr,pqcs->pqrs", kappa, eri_mo, optimize=True)
    deri += np.einsum("ds,pqrd->pqrs", kappa, eri_mo, optimize=True)
    return deri


def orbital_hessian_action_from_integrals(h1_mo, eri_mo, dm1, dm2, kappa):
    """
    Analytic orbital-only Hessian action for the generalized-Fock gradient.

    This freezes the 1- and 2-RDMs in the current MO representation and
    differentiates the MO one- and two-electron integrals with respect to the
    orbital-rotation generator ``kappa``.
    """
    dh1 = orbital_h1_response(h1_mo, kappa)
    deri = orbital_eri_response(eri_mo, kappa)
    dfock = generalized_fock(dh1, deri, dm1, dm2)
    return orbital_gradient(dfock)


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


def _orthonormalize_columns(V, tol=1.0e-12):
    """Return a numerically stable orthonormal basis for the columns of ``V``."""
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        return np.zeros((V.shape[0], 0), dtype=float)
    Q, R = np.linalg.qr(V, mode="reduced")
    keep = np.abs(np.diag(R)) > tol
    if not np.any(keep):
        return np.zeros((V.shape[0], 0), dtype=float)
    return Q[:, keep]


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


def diagonal_hessian(fock, ncore, ncas, level_shift=1.0e-3, absolute=False):
    """Return the packed diagonal orbital-Hessian approximation."""
    denom = diagonal_denominator(fock, ncore, ncas, level_shift=level_shift)
    pairs = nonredundant_pairs(ncore, ncas, fock.shape[0])
    if not pairs:
        return np.zeros(0, dtype=float)
    vals = []
    for p, q in pairs:
        d = np.real(denom[p, q])
        if abs(d) < level_shift:
            d = np.copysign(level_shift, d if d != 0.0 else 1.0)
        if absolute:
            d = abs(d)
        vals.append(d)
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


def quadratic_model_change(step_vec, grad_vec, hess_diag):
    """Quadratic orbital model value for a packed step."""
    step_vec = np.asarray(step_vec, dtype=float)
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_diag = np.asarray(hess_diag, dtype=float)
    return float(
        np.dot(grad_vec, step_vec)
        + 0.5 * np.dot(step_vec, hess_diag * step_vec)
    )


def augmented_hessian_direction(
    grad_vec,
    hess_diag,
    max_step=None,
    regularization=1.0e-10,
    fallback_step=None,
):
    """
    Solve a diagonal augmented-Hessian model in packed orbital coordinates.

    The model uses the lowest eigenpair of

        [[0, g^T],
         [g, H ]]

    where ``g`` is the orbital gradient and ``H`` is a diagonal Hessian
    approximation. The resulting step is ``x / alpha`` from the normalized
    eigenvector ``[alpha, x]``.
    """
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_diag = np.asarray(hess_diag, dtype=float)
    if grad_vec.ndim != 1 or hess_diag.ndim != 1:
        raise ValueError("grad_vec and hess_diag must be 1D arrays.")
    if grad_vec.shape != hess_diag.shape:
        raise ValueError("grad_vec and hess_diag must have the same shape.")
    if grad_vec.size == 0:
        return np.zeros(0, dtype=float)

    hess_model = np.maximum(np.abs(hess_diag), float(regularization))
    ah = np.zeros((grad_vec.size + 1, grad_vec.size + 1), dtype=float)
    ah[0, 1:] = grad_vec
    ah[1:, 0] = grad_vec
    ah[1:, 1:] = np.diag(hess_model)

    eigvals, eigvecs = np.linalg.eigh(ah)
    candidates = []

    for root in range(eigvecs.shape[1]):
        vec = eigvecs[:, root]
        alpha = float(vec[0])
        if alpha < 0.0:
            vec = -vec
            alpha = -alpha
        if abs(alpha) < 1.0e-10:
            continue

        step = np.asarray(vec[1:] / alpha, dtype=float)
        if max_step is not None:
            step = limit_step_norm(step, max_step)

        directional_derivative = float(np.dot(step, grad_vec))
        if directional_derivative >= -1.0e-12:
            continue

        candidates.append(
            (
                quadratic_model_change(step, grad_vec, hess_model),
                np.linalg.norm(step),
                step,
            )
        )

    if fallback_step is not None:
        fallback_step = np.asarray(fallback_step, dtype=float)
        if max_step is not None:
            fallback_step = limit_step_norm(fallback_step, max_step)
        if np.dot(fallback_step, grad_vec) < -1.0e-12:
            candidates.append(
                (
                    quadratic_model_change(fallback_step, grad_vec, hess_model),
                    np.linalg.norm(fallback_step),
                    fallback_step,
                )
            )

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return np.asarray(candidates[0][2], dtype=float)

    step = -grad_vec / hess_model
    if max_step is not None:
        step = limit_step_norm(step, max_step)
    return np.asarray(step, dtype=float)


def davidson_augmented_hessian_direction(
    grad_vec,
    hess_diag,
    matvec,
    max_step=None,
    regularization=1.0e-10,
    max_cycle=4,
    max_subspace=8,
    tol=1.0e-4,
    guess=None,
    fallback_step=None,
):
    """
    Solve the orbital augmented-Hessian eigenproblem with a Davidson microiteration.

    Parameters
    ----------
    grad_vec : ndarray
        Packed orbital gradient.
    hess_diag : ndarray
        Diagonal Hessian approximation used for preconditioning.
    matvec : callable
        Returns the Hessian action ``H @ x`` for a packed step ``x``.
    """
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_diag = np.asarray(hess_diag, dtype=float)
    if grad_vec.ndim != 1 or hess_diag.ndim != 1:
        raise ValueError("grad_vec and hess_diag must be 1D arrays.")
    if grad_vec.shape != hess_diag.shape:
        raise ValueError("grad_vec and hess_diag must have the same shape.")
    if grad_vec.size == 0:
        return np.zeros(0, dtype=float)

    hess_model = np.maximum(np.abs(hess_diag), float(regularization))
    guess_cols = []

    if guess is not None:
        arr = np.asarray(guess, dtype=float)
        if arr.ndim == 1:
            guess_cols.append(arr.copy())
        else:
            guess_cols.extend(arr[:, i].copy() for i in range(arr.shape[1]))

    diag_step = -grad_vec / hess_model
    if fallback_step is not None:
        fallback_step = np.asarray(fallback_step, dtype=float)
        guess_cols.insert(0, fallback_step.copy())
    guess_cols.append(diag_step)

    seed_cols = []
    for col in guess_cols:
        if col.size != grad_vec.size:
            continue
        norm = np.linalg.norm(col)
        if norm > 1.0e-12:
            seed_cols.append(col / norm)
    if not seed_cols:
        seed_cols = [(-grad_vec / np.linalg.norm(grad_vec))]

    V = _orthonormalize_columns(np.column_stack(seed_cols))
    W = np.column_stack([np.asarray(matvec(V[:, i]), dtype=float) for i in range(V.shape[1])])

    best = None

    for _ in range(max_cycle):
        h_proj = 0.5 * (V.T @ W + (V.T @ W).T)
        g_proj = V.T @ grad_vec
        ah_proj = np.zeros((V.shape[1] + 1, V.shape[1] + 1), dtype=float)
        ah_proj[0, 1:] = g_proj
        ah_proj[1:, 0] = g_proj
        ah_proj[1:, 1:] = h_proj

        eigvals, eigvecs = np.linalg.eigh(ah_proj)
        candidate = None

        for root in np.argsort(eigvals):
            coeff = eigvecs[1:, root]
            alpha = float(eigvecs[0, root])
            if alpha < 0.0:
                coeff = -coeff
                alpha = -alpha
            if abs(alpha) < 1.0e-10:
                continue

            step_raw = V @ (coeff / alpha)
            hv_raw = W @ (coeff / alpha)
            scale = 1.0
            if max_step is not None:
                peak = np.max(np.abs(step_raw))
                if peak > max_step and peak > 0.0:
                    scale = max_step / peak
            step = scale * step_raw
            hv = scale * hv_raw

            directional_derivative = float(np.dot(step, grad_vec))
            if directional_derivative >= -1.0e-12:
                continue

            model = float(np.dot(grad_vec, step) + 0.5 * np.dot(step, hv))
            orbital_residual = alpha * grad_vec + hv_raw - float(eigvals[root]) * step_raw
            scalar_residual = float(np.dot(grad_vec, step_raw) - float(eigvals[root]) * alpha)
            residual_norm = float(
                np.sqrt(np.dot(orbital_residual, orbital_residual) + scalar_residual ** 2)
            )
            candidate = {
                "model": model,
                "residual_norm": residual_norm,
                "eigenvalue": float(eigvals[root]),
                "step": step,
                "orbital_residual": orbital_residual,
            }
            break

        if candidate is None:
            break

        if (
            best is None
            or candidate["model"] < best["model"]
            or (
                abs(candidate["model"] - best["model"]) < 1.0e-12
                and candidate["residual_norm"] < best["residual_norm"]
            )
        ):
            best = candidate

        if candidate["residual_norm"] < tol:
            return np.asarray(candidate["step"], dtype=float)

        denom = hess_model - candidate["eigenvalue"]
        safe = np.where(
            np.abs(denom) > 1.0e-8,
            denom,
            np.where(denom >= 0.0, 1.0e-8, -1.0e-8),
        )
        correction = -candidate["orbital_residual"] / safe
        correction -= V @ (V.T @ correction)
        corr_norm = np.linalg.norm(correction)
        if corr_norm <= 1.0e-12:
            break
        correction /= corr_norm

        if V.shape[1] + 1 > max_subspace:
            keep = []
            keep.append(candidate["step"])
            keep.append(-grad_vec / hess_model)
            V = _orthonormalize_columns(np.column_stack(keep))
            W = np.column_stack([np.asarray(matvec(V[:, i]), dtype=float) for i in range(V.shape[1])])
            continue

        corr_block = correction.reshape(-1, 1)
        V = np.column_stack((V, corr_block))
        W = np.column_stack((W, np.asarray(matvec(correction), dtype=float).reshape(-1, 1)))

    if best is not None:
        return np.asarray(best["step"], dtype=float)

    return augmented_hessian_direction(
        grad_vec,
        hess_diag,
        max_step=max_step,
        regularization=regularization,
        fallback_step=fallback_step,
    )


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
