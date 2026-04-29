#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Davidson eigensolvers used across pyqed.

The original module bundled a dense-matrix demo implementation.  This version
keeps the same public ``davidson(A, neigen, ...)`` entry point, but upgrades it
to a proper block Davidson routine with thick restart, explicit residual-based
convergence checks, optional matrix-free matvec support, and a compatibility
wrapper ``davidson_solver`` used by some older model code.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np


LOGGER = logging.getLogger(__name__)


def digaonal_dominant(n, sparsity=1e-4):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1e3 * np.random.rand()
    A = A + sparsity * np.random.randn(n, n)
    A = (A.T + A) / 2
    return A


def diag_non_tda(n, sparsity=1e-4):
    A = digaonal_dominant(n, sparsity=sparsity)
    C = sparsity * np.random.rand(n, n)
    return np.block([[A, C], [-C.T, -A.T]])


def jacobi_correction(uj, A, thetaj):
    I = np.eye(A.shape[0])
    Pj = I - np.outer(uj, uj)
    rj = (A - thetaj * I) @ uj
    w = Pj @ ((A - thetaj * I) @ Pj)
    return np.linalg.solve(w, rj)


def get_initial_guess(A, neigen):
    A = np.asarray(A)
    d = np.diag(A)
    index = np.argsort(d)
    guess = np.zeros((A.shape[0], neigen), dtype=A.dtype)
    for i in range(min(neigen, A.shape[0])):
        guess[index[i], i] = 1
    return guess


def reorder_matrix(A):
    A = np.asarray(A)
    n = A.shape[0]
    tmp = np.zeros((n, n), dtype=A.dtype)
    index = np.argsort(np.diagonal(A))
    for i in range(n):
        for j in range(i, n):
            tmp[i, j] = A[index[i], index[j]]
            tmp[j, i] = tmp[i, j]
    return tmp


def _orthonormalize_columns(V, tol=1e-12):
    if V.size == 0:
        return np.zeros((V.shape[0], 0), dtype=V.dtype)
    Q, R = np.linalg.qr(V, mode="reduced")
    keep = np.abs(np.diag(R)) > tol
    if not np.any(keep):
        return np.zeros((V.shape[0], 0), dtype=V.dtype)
    return Q[:, keep]


def _infer_dimension(A, diag=None, guess=None):
    if diag is not None:
        return np.asarray(diag).size
    if guess is not None:
        guess_arr = np.asarray(guess)
        return guess_arr.shape[0]
    if hasattr(A, "shape") and A.shape is not None:
        return int(A.shape[0])
    raise ValueError("Cannot infer Davidson dimension; provide diag or guess.")


def _resolve_matvec(A):
    if callable(A):
        return A
    if hasattr(A, "dot"):
        return lambda x: np.asarray(A.dot(x))
    A_arr = np.asarray(A)
    return lambda x: A_arr @ x


def _resolve_diag(A, diag, n):
    if diag is not None:
        diag_arr = np.asarray(diag, dtype=float).reshape(n)
        return diag_arr
    if callable(A):
        raise ValueError("Matrix-free Davidson requires a diagonal preconditioner.")
    A_arr = np.asarray(A)
    if A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("Davidson expects a square matrix.")
    return np.asarray(np.diag(A_arr), dtype=float)


def _build_guess(diag, neigen, guess=None):
    n = diag.size
    cols = []
    dtype = complex if guess is not None and np.iscomplexobj(guess) else float
    if guess is not None:
        guess_arr = np.asarray(guess, dtype=dtype)
        if guess_arr.ndim == 1:
            cols.append(guess_arr.reshape(n))
        else:
            cols.extend(guess_arr[:, i].reshape(n) for i in range(guess_arr.shape[1]))
    for idx in np.argsort(diag):
        e = np.zeros(n, dtype=dtype)
        e[idx] = 1.0
        cols.append(e)
        if len(cols) >= max(2 * neigen, neigen):
            break
    return _orthonormalize_columns(np.column_stack(cols))


def _projected_residual_norms(ritz, aritz, theta):
    resid = aritz - ritz * theta
    return resid, np.linalg.norm(resid, axis=0)


def _resolve_preconditioner(A, diag, jacobi=False, precond=None):
    if precond is not None:
        if callable(precond):
            return precond
        precond_arr = np.asarray(precond, dtype=float)

        def _diag_precond(resid, theta, vec):
            denom = theta - precond_arr
            safe = np.where(
                np.abs(denom) > 1e-12,
                denom,
                np.where(denom >= 0, 1e-12, -1e-12),
            )
            return resid / safe

        return _diag_precond

    if jacobi:
        if callable(A):
            raise ValueError("jacobi=True requires an explicit matrix.")
        explicit_matrix = np.asarray(A)

        def _jacobi_precond(resid, theta, vec):
            return jacobi_correction(vec, explicit_matrix, theta)

        return _jacobi_precond

    diag_arr = np.asarray(diag, dtype=float)

    def _default_precond(resid, theta, vec):
        denom = theta - diag_arr
        safe = np.where(
            np.abs(denom) > 1e-12,
            denom,
            np.where(denom >= 0, 1e-12, -1e-12),
        )
        return resid / safe

    return _default_precond


def _build_projected_matrix(V, AV):
    return V.conj().T @ AV


def _expand_projected_matrix(T, V, AV, new_block, AV_new):
    if T.size == 0:
        return new_block.conj().T @ AV_new
    cross = V.conj().T @ AV_new
    lower = new_block.conj().T @ AV
    diag_block = new_block.conj().T @ AV_new
    top = np.hstack((T, cross))
    bottom = np.hstack((lower, diag_block))
    return np.vstack((top, bottom))


def davidson(
    A,
    neigen,
    tol=1e-6,
    itermax=100,
    jacobi=False,
    diag=None,
    precond=None,
    guess=None,
    max_space=None,
    tol_residual=None,
    lindep=1e-12,
    return_info=False,
):
    """
    Compute the lowest ``neigen`` eigenpairs of a Hermitian problem.

    Parameters
    ----------
    A
        Dense matrix, sparse matrix-like object with ``dot``, or a callable
        matvec ``A(x)``.
    neigen : int
        Number of lowest eigenpairs to compute.
    tol : float, optional
        Energy convergence threshold.  The solver also checks residual norms.
    itermax : int, optional
        Maximum Davidson macro-iterations.
    jacobi : bool, optional
        Use the dense Jacobi correction when ``A`` is an explicit matrix.
    diag : array_like, optional
        Diagonal preconditioner. Required for matrix-free use.
    precond : callable or array_like, optional
        Custom preconditioner ``precond(resid, theta, vec)``. If an array is
        given, it is treated as a diagonal preconditioner.
    guess : array_like, optional
        Initial guess vector(s), shape ``(n,)`` or ``(n, nguess)``.
    max_space : int, optional
        Maximum subspace size before thick restart.
    tol_residual : float, optional
        Residual threshold. Defaults to ``sqrt(tol)``.
    lindep : float, optional
        Linear-dependence threshold for orthogonalized correction vectors.
    return_info : bool, optional
        When true, also return a diagnostics dict.
    """
    if neigen < 1:
        raise ValueError("neigen must be positive.")

    n = _infer_dimension(A, diag=diag, guess=guess)
    if neigen > n:
        raise ValueError("neigen cannot exceed the problem dimension.")

    matvec = _resolve_matvec(A)
    diag_arr = _resolve_diag(A, diag, n)
    if max_space is None:
        max_space = min(n, max(24, 12 * neigen))
    tol_res = np.sqrt(tol) if tol_residual is None else tol_residual

    V = _build_guess(diag_arr, neigen, guess=guess)
    AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
    T = _build_projected_matrix(V, AV)
    precondition = _resolve_preconditioner(A, diag_arr, jacobi=jacobi, precond=precond)

    prev_theta = None
    locked = np.zeros(neigen, dtype=bool)
    info = {
        "converged": False,
        "iterations": 0,
        "residual_norms": None,
        "energy_change": None,
        "subspace_dim": V.shape[1],
        "locked_roots": 0,
        "restarts": 0,
    }

    for iteration in range(1, itermax + 1):
        theta_all, alpha_all = np.linalg.eigh(T)
        order = np.argsort(theta_all)
        theta = theta_all[order][:neigen]
        alpha = alpha_all[:, order[:neigen]]

        ritz = V @ alpha
        aritz = AV @ alpha
        resid, resid_norms = _projected_residual_norms(ritz, aritz, theta)
        de = theta if prev_theta is None else theta - prev_theta
        max_de = np.max(np.abs(de))

        root_conv = resid_norms < tol_res
        locked |= root_conv

        info.update(
            iterations=iteration,
            residual_norms=resid_norms.copy(),
            energy_change=de.copy(),
            subspace_dim=V.shape[1],
            locked_roots=int(np.count_nonzero(locked)),
        )

        LOGGER.debug(
            "Davidson iter=%d space=%d max|de|=%.3e max|r|=%.3e",
            iteration,
            V.shape[1],
            max_de,
            np.max(resid_norms),
        )

        if np.all(locked):
            info["converged"] = True
            if return_info:
                return theta, ritz, info
            return theta, ritz

        new_vecs = []
        for root in range(neigen):
            if locked[root]:
                continue

            corr_raw = np.asarray(precondition(resid[:, root], theta[root], ritz[:, root]))
            corr = np.asarray(corr_raw, dtype=np.result_type(V.dtype, corr_raw.dtype, resid.dtype))
            corr -= V @ (V.conj().T @ corr)
            for prev in new_vecs:
                corr -= prev * np.vdot(prev, corr)

            norm = np.linalg.norm(corr)
            if norm > lindep:
                new_vecs.append(corr / norm)

        if not new_vecs:
            info["converged"] = np.all(locked)
            if return_info:
                return theta, ritz, info
            return theta, ritz

        new_block = np.column_stack(new_vecs)
        if V.shape[1] + new_block.shape[1] > max_space:
            keep = min(theta_all.size, max(2 * neigen + 2, neigen + 1))
            restart_cols = [V @ alpha_all[:, order[i]] for i in range(keep)]
            restart_cols.extend(new_block[:, i] for i in range(new_block.shape[1]))
            V = _orthonormalize_columns(np.column_stack(restart_cols))
            AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
            T = _build_projected_matrix(V, AV)
            info["restarts"] += 1
        else:
            AV_new = np.column_stack([matvec(new_block[:, i]) for i in range(new_block.shape[1])])
            T = _expand_projected_matrix(T, V, AV, new_block, AV_new)
            V = np.column_stack((V, new_block))
            AV = np.column_stack((AV, AV_new))

        prev_theta = theta.copy()

    raise RuntimeError("Davidson solver did not converge within itermax iterations.")


def davidson_solver(A, neigen, **kwargs):
    """Backward-compatible wrapper used by older model code."""
    return davidson(A, neigen=neigen, **kwargs)


def block_davidson(A, neig=3, max_iterations=20, tol=1e-9):
    """Compatibility alias to the upgraded block Davidson implementation."""
    return davidson(A, neigen=neig, tol=tol, itermax=max_iterations)
