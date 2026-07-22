"""Small ODE integration helpers used across :mod:`pyqed`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import linalg as la
from scipy.integrate import DOP853 as _ScipyDOP853
from scipy.integrate import solve_ivp as _scipy_solve_ivp


def normalize_method(method):
    """Normalize common ODE method aliases to the names used internally."""
    method_key = str(method).lower().replace("_", "-")
    if method_key in ("adaptive", "dopri853"):
        return "dop853"
    if method_key in ("exp-krylov", "expm-krylov", "exponential-krylov"):
        return "krylov"
    return method_key


def _validate_t_eval(t_eval):
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.ndim != 1 or t_eval.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("t_eval must contain only finite values")
    if np.any(np.diff(t_eval) <= 0.0):
        raise ValueError("t_eval must be strictly increasing")
    return t_eval


def dop853(rhs, y0, t_eval, rtol=1.0e-8, atol=1.0e-10,
           observer=None, save=True, **options):
    """Integrate with adaptive DOP853 and stream values onto ``t_eval``.

    ``save=False`` retains only the final state and optional ``observer``
    values.  This is the memory-efficient generic path for Python right-hand
    sides; compiled solvers can implement the same result contract internally.
    """
    t_eval = _validate_t_eval(t_eval)
    initial = np.asarray(y0)
    dtype = np.complex128 if np.iscomplexobj(initial) else np.float64
    y = np.ascontiguousarray(initial, dtype=dtype).reshape(-1)
    if y.size == 0:
        raise ValueError("y0 must contain at least one value")

    states = [y.copy()] if save else None
    observations = [observer(t_eval[0], y)] if observer is not None else None
    if t_eval.size == 1:
        return SimpleNamespace(
            success=True,
            message="success",
            nfev=0,
            n_steps=0,
            t=t_eval.copy(),
            y=np.asarray(states).T if save else None,
            y_final=y.copy(),
            observations=observations,
        )

    solver = _ScipyDOP853(
        rhs,
        float(t_eval[0]),
        y,
        float(t_eval[-1]),
        rtol=rtol,
        atol=atol,
        **options,
    )
    output_index = 1
    n_steps = 0

    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            return SimpleNamespace(
                success=False,
                message=message or "DOP853 failed",
                nfev=solver.nfev,
                n_steps=n_steps,
                t=t_eval[:output_index].copy(),
                y=np.asarray(states).T if save else None,
                y_final=solver.y.copy(),
                observations=observations,
            )
        n_steps += 1
        tolerance = (
            32.0 * np.finfo(float).eps * max(1.0, abs(float(solver.t)))
        )
        due_end = output_index
        while due_end < t_eval.size and t_eval[due_end] <= solver.t + tolerance:
            due_end += 1

        dense_output = None
        for index in range(output_index, due_end):
            sample_t = t_eval[index]
            if abs(sample_t - solver.t) <= tolerance:
                sample = solver.y.copy()
            else:
                if dense_output is None:
                    dense_output = solver.dense_output()
                sample = np.asarray(dense_output(sample_t)).copy()
            if save:
                states.append(sample)
            if observer is not None:
                observations.append(observer(sample_t, sample))
        output_index = due_end

    if output_index != t_eval.size:
        raise RuntimeError("DOP853 did not reach every requested output time")

    return SimpleNamespace(
        success=True,
        message="success",
        nfev=solver.nfev,
        n_steps=n_steps,
        t=t_eval.copy(),
        y=np.asarray(states).T if save else None,
        y_final=solver.y.copy(),
        observations=observations,
    )


def solve_ivp(rhs, y0, t_eval, method="DOP853", rtol=1.0e-8,
              atol=1.0e-10, observer=None, save=True, **options):
    """Integrate ``dy/dt = rhs(t, y)`` on a requested output grid."""
    t_eval = _validate_t_eval(t_eval)
    method_key = normalize_method(method)
    if method_key == "dop853":
        return dop853(
            rhs,
            y0,
            t_eval,
            rtol=rtol,
            atol=atol,
            observer=observer,
            save=save,
            **options,
        )

    y0 = np.asarray(y0)
    if t_eval.size == 1:
        y = y0.reshape(-1)
        return SimpleNamespace(
            success=True,
            message="success",
            nfev=0,
            t=t_eval.copy(),
            y=y.reshape(-1, 1).copy() if save else None,
            y_final=y.copy(),
            observations=[observer(t_eval[0], y)] if observer is not None else None,
        )

    solution = _scipy_solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **options,
    )
    solution.y_final = solution.y[:, -1].copy()
    solution.observations = (
        [observer(t, state) for t, state in zip(solution.t, solution.y.T)]
        if observer is not None
        else None
    )
    if not save:
        solution.y = None
    return solution


def expm_krylov(matvec, y, dt, krylov_dim=30, tol=1.0e-12):
    """Approximate ``exp(dt * A) @ y`` by Arnoldi using only ``A @ y``."""
    y = np.asarray(y, dtype=np.complex128)
    beta = np.linalg.norm(y)
    if beta == 0:
        return y.copy()

    size = y.size
    mmax = min(int(krylov_dim), size)
    if mmax < 1:
        raise ValueError("krylov_dim must be at least 1")

    basis = np.zeros((mmax + 1, size), dtype=np.complex128)
    hessenberg = np.zeros((mmax + 1, mmax), dtype=np.complex128)
    basis[0] = y / beta
    used_dim = mmax

    for col in range(mmax):
        work = matvec(basis[col])
        for row in range(col + 1):
            hessenberg[row, col] = np.vdot(basis[row], work)
            work -= hessenberg[row, col] * basis[row]
        next_norm = np.linalg.norm(work)
        hessenberg[col + 1, col] = next_norm
        if next_norm <= tol:
            used_dim = col + 1
            break
        if col + 1 < mmax:
            basis[col + 1] = work / next_norm

    projected = hessenberg[:used_dim, :used_dim]
    e1 = np.zeros(used_dim, dtype=np.complex128)
    e1[0] = 1.0
    coeff = la.expm(dt * projected) @ e1
    return beta * (coeff @ basis[:used_dim])


def krylov(matvec, y0, t_eval, krylov_dim=30, tol=1.0e-12,
           observer=None, save=True, progress=None):
    """Integrate a time-independent linear ODE with Krylov exponential steps."""
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.ndim != 1 or t_eval.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional array")

    y = np.ascontiguousarray(np.asarray(y0, dtype=np.complex128).reshape(-1))
    states = [y.copy()] if save else None
    observations = [observer(t_eval[0], y)] if observer is not None else None

    nfev = 0
    steps = range(t_eval.size - 1)
    if progress is not None:
        steps = progress(steps)

    for i in steps:
        dt = t_eval[i + 1] - t_eval[i]
        y = expm_krylov(matvec, y, dt, krylov_dim=krylov_dim, tol=tol)
        nfev += min(int(krylov_dim), y.size)
        if save:
            states.append(y.copy())
        if observer is not None:
            observations.append(observer(t_eval[i + 1], y))

    if save:
        y_series = np.asarray(states, dtype=np.complex128).T
    else:
        y_series = None

    return SimpleNamespace(
        success=True,
        message="success",
        nfev=nfev,
        t=t_eval.copy(),
        y=y_series,
        y_final=y.copy(),
        observations=observations,
    )


__all__ = [
    "dop853",
    "expm_krylov",
    "krylov",
    "normalize_method",
    "solve_ivp",
]
