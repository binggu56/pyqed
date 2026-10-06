#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified real symmetric and complex Hermitian Davidson eigensolver."""

from __future__ import annotations

import logging

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


def _apply_columns(matvec, vectors):
    """Apply a matrix-free operator to one or more column vectors."""

    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError("Davidson block input must be a rank-2 array.")
    matmat = getattr(matvec, "matmat", None)
    if callable(matmat):
        result = np.asarray(matmat(vectors))
        if result.shape != vectors.shape:
            raise ValueError(
                "Matrix-free block action returned shape "
                f"{result.shape}, expected {vectors.shape}."
            )
        return result
    return np.column_stack([matvec(vectors[:, i]) for i in range(vectors.shape[1])])


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


def _iterate(
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
    return_partial=False,
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
    return_partial : bool, optional
        When true, return the best Ritz pairs found after ``itermax`` instead
        of raising. The returned diagnostics keep ``converged=False``.
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

    V = _build_guess(diag_arr, neigen, guess=guess)[:, :max_space]
    AV = _apply_columns(matvec, V)
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
        locked = root_conv.copy()

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
            keep = min(theta_all.size, max_space - new_block.shape[1],
                       max(2 * neigen + 2, neigen + 1))
            restart_cols = [V @ alpha_all[:, order[i]] for i in range(keep)]
            restart_cols.extend(new_block[:, i] for i in range(new_block.shape[1]))
            V = _orthonormalize_columns(np.column_stack(restart_cols))
            AV = _apply_columns(matvec, V)
            T = _build_projected_matrix(V, AV)
            info["restarts"] += 1
        else:
            AV_new = _apply_columns(matvec, new_block)
            T = _expand_projected_matrix(T, V, AV, new_block, AV_new)
            V = np.column_stack((V, new_block))
            AV = np.column_stack((AV, AV_new))

        prev_theta = theta.copy()

    if return_partial:
        info["converged"] = False
        info["max_iterations_reached"] = True
        if return_info:
            return theta, ritz, info
        return theta, ritz
    raise RuntimeError("Davidson solver did not converge within itermax iterations.")


def davidson(matrix, roots, tolerance=1e-10, iterations=200, space=None,
             memory_limit=512*1024**2, guess=None, *, diag=None, precond=None,
             lindep=1e-12, backend="auto", return_info=True, return_partial=False,
             matmat=None, dtype=None):
    """Compute the lowest eigenpairs of a symmetric or Hermitian operator.

    Dense matrices, callable operators, sparse matrices, and LinearOperator
    objects use the compiled block solver when available. Custom preconditioners
    or nondefault ``lindep`` use the Python implementation. ``backend='python'`` or ``'compiled'`` forces a path;
    unsupported compiled inputs raise ValueError, and an unavailable required
    extension raises ImportError. Auto falls back only on extension absence,
    never on a failed solve. Callable/LinearOperator inputs require ``diag``;
    sparse diagonals are extracted without densifying. Matrix-free actions
    must be Hermitian; global Hermiticity cannot be checked without assembling
    the operator. ``matmat(X)`` (or the operator's matmat method) accepts (n,k)
    blocks; otherwise the solver calls matvec on each column. Compiled callbacks
    receive owned input arrays, so retaining or modifying them is safe. The
    GIL is released for compiled solver work and acquired for each callback.
    ``dtype`` selects real or complex arithmetic; it defaults to the operator's
    dtype, or complex128 for an untyped callable. Use dtype=float for an untyped
    real operator. Incompatible callback shapes/dtypes or nonfinite values raise.
    Python callback overhead remains; block actions reduce callback frequency.

    Returns ``(values, vectors, info)`` by default, or just the eigenpairs with
    ``return_info=False``. ``tolerance`` is an absolute residual norm on both
    paths. Failure raises RuntimeError unless ``return_partial=True``; inspect
    ``info['converged']`` when accepting partial Ritz pairs. ``info['backend']``
    identifies the implementation used. Values/vectors are NumPy arrays.
    Guesses have shape (n,) or (n, nguess); ``precond(residual, value, vector)``
    may be callable or a diagonal array. ``space`` caps the search space and
    defaults to min(n, max(24, 4*roots)). ``memory_limit`` limits a conservative
    scratch-space estimate, excluding input storage and operator callbacks.

    Both implementations adapt E. R. Davidson, J. Comput. Phys. 17, 87-94
    (1975), doi:10.1016/0021-9991(75)90065-0, using block expansion, diagonal
    preconditioning, thick restarts and explicit residual tests. The compiled
    path additionally retains bounded GD+k-inspired history (Stathopoulos and
    McCombs, ACM TOMS 37(2), Article 21, 2010,
    doi:10.1145/1731022.1731031). The Python path omits this history; neither
    reproduces PRIMME or implements inner Jacobi-Davidson solves. The compiled
    path temporarily freezes a converged lowest prefix at restarts (residual
    below 0.9*tolerance), solves in its orthogonal complement, and restores
    all couplings for a final Rayleigh-Ritz residual check. A newly discovered
    lower active root disables freezing. This is a heuristic deflation policy,
    not PRIMME locking or a guarantee of lowest-root completeness. Neither
    path provides universal convergence/performance guarantees. Accelerate real
    blocks use adapted guarded CholeskyQR2 (Fukaya et al., ScalA 2014,
    doi:10.1109/ScalA.2014.11) with pivoted QR fallback; complex blocks use
    pivoted QR directly. Portable builds use scalar contractions and Jacobi
    projected solves. Both public modules expose this same function.
    """
    if backend not in ("auto", "python", "compiled"):
        raise ValueError("backend must be auto, python, or compiled")
    if isinstance(roots, bool) or int(roots) != roots or roots < 1:
        raise ValueError("roots must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0 or iterations < 1:
        raise ValueError("Require positive finite tolerance and iteration count")
    roots = int(roots)
    operator_dtype = getattr(matrix, "dtype", None)
    sparse = hasattr(matrix, "tocsr")
    operator = callable(matrix) or sparse or hasattr(matrix, "matvec")
    if matmat is None:
        matmat = getattr(matrix, "matmat", None)
    if matmat is not None and not callable(matmat):
        raise ValueError("matmat must be callable")
    if sparse or hasattr(matrix, "matvec"):
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Expected a square operator")
        if sparse:
            if diag is None:
                diag = matrix.diagonal()
            if matmat is None:
                matmat = matrix.dot
        if diag is not None and np.asarray(diag).size != matrix.shape[0]:
            raise ValueError("Operator and diagonal dimensions differ")
        matrix = matrix.dot if sparse else matrix.matvec
    if not operator:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Expected a square matrix")
        if not np.all(np.isfinite(matrix)) or not np.allclose(
                matrix, matrix.conj().T, rtol=1e-12, atol=1e-13):
            raise ValueError("Expected a finite Hermitian matrix")
        n = matrix.shape[0]
    else:
        if diag is None:
            raise ValueError("Matrix-free Davidson requires diag")
        n = np.asarray(diag).size
    if roots > n:
        raise ValueError("roots cannot exceed the dimension")
    space = min(n, max(24, 4*roots) if space is None else int(space))
    if space < roots or (space == roots and roots < n):
        raise ValueError("Davidson space must allow expansion beyond roots")
    if diag is not None:
        diag = np.asarray(diag).reshape(n)
        if not np.all(np.isfinite(diag)) or np.any(np.abs(np.imag(diag)) > 1e-12):
            raise ValueError("Hermitian diagonal must be finite and real")
        diag = np.real(diag)
    if guess is not None:
        guess = np.asarray(guess)
        if guess.ndim == 1:
            guess = guess[:, None]
        if (guess.ndim != 2 or guess.shape[0] != n or not 1 <= guess.shape[1] <= min(space, 2*roots)
                or not np.all(np.isfinite(guess))):
            raise ValueError("Invalid Davidson guess shape or values")
    use_callback = operator or diag is not None or matmat is not None
    eligible = precond is None and lindep == 1e-12
    if backend == "compiled" and not eligible:
        raise ValueError("Compiled Davidson requires default preconditioner/lindep")
    if dtype is None:
        complex_data = np.iscomplexobj(guess) or (
            (operator_dtype is None or np.dtype(operator_dtype).kind == "c")
            if operator else np.iscomplexobj(matrix)
        )
    else:
        dtype = np.dtype(dtype)
        if dtype.kind not in "fc":
            raise ValueError("dtype must be a real or complex floating type")
        complex_data = dtype.kind == "c"
        if not complex_data and (np.iscomplexobj(guess) or
                                 (not operator and np.iscomplexobj(matrix))):
            raise ValueError("Real dtype cannot represent complex matrix or guess")
    scalar_dtype = np.complex128 if complex_data else np.float64
    estimate = np.dtype(scalar_dtype).itemsize*(
        2*n*space + (18 if use_callback else 14)*n*roots + 8*space*space)
    if estimate > memory_limit:
        raise MemoryError("Davidson estimated workspace exceeds memory_limit")
    kernel = None
    if eligible and backend != "python":
        from pyqed.linalg import davidson_kernels
        kernel = davidson_kernels.davidson_operator if use_callback else davidson_kernels.davidson
        if kernel is None and backend == "compiled":
            raise ImportError(f"Compiled Davidson unavailable: {davidson_kernels.build_error}")
    if use_callback:
        action = matrix if operator else matrix.dot
        if diag is None:
            diag = matrix.diagonal().real
        if matmat is None and not operator:
            matmat = matrix.dot
    if kernel is not None:
        if guess is not None:
            guess = np.ascontiguousarray(guess, dtype=scalar_dtype)
        if use_callback:
            values, vectors, info = kernel(
                action, np.ascontiguousarray(diag, dtype=float), roots, tolerance,
                int(iterations), space, int(memory_limit), guess, matmat, complex_data)
        else:
            values, vectors, info = kernel(
                np.ascontiguousarray(matrix, dtype=scalar_dtype), roots, tolerance,
                int(iterations), space, int(memory_limit), guess)
        info = dict(info, backend="compiled")
    else:
        if diag is None:
            diag = matrix.diagonal().real
        if use_callback and matmat is not None:
            def block_action(x):
                return action(x)
            block_action.matmat = matmat
            matrix = block_action
        values, vectors, info = _iterate(
            matrix, roots, itermax=int(iterations), diag=diag, precond=precond,
            guess=guess, max_space=space, tol_residual=tolerance, lindep=lindep,
            return_info=True, return_partial=True,
        )
        info = dict(info, backend="python")
    values = np.asarray(values)
    info["residual_norms"] = np.asarray(info["residual_norms"])
    info["converged"] = bool(info["converged"])
    info["max_iterations_reached"] = bool(
        not info["converged"] and info["iterations"] >= iterations
    )
    if not info["converged"] and not return_partial:
        raise RuntimeError("Davidson solver did not converge within the iteration/subspace limits")
    return (values, vectors, info) if return_info else (values, vectors)
