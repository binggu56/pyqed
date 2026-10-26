"""Projected isolated-state response using the existing SciPy GMRES backend.

Solves Q(L-e)Q dz = -Q(dL)z with the left/right eigenvector projector.
This is an exact differentiated eigenproblem, not an excitation-space truncation.
Uses the GMRES/diagonal-preconditioning approach of qchem.orbital_response;
Saad and Schultz, SIAM J. Sci. Stat. Comput. 7, 856 (1986),
doi:10.1137/0907058. Real isolated roots only. The caller checks spectral
isolation and supplies the complete operator action, including negative BSE roots.
"""
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def projected_response(action, diagonal, energy, state, left, derivative, *,
                       tolerance=1e-11, maxiter=200):
    """Return the gauge-fixed tangent and independently checked diagnostics."""
    start = perf_counter()
    if tolerance <= 0 or not np.isfinite(tolerance) or maxiter < 1:
        raise ValueError("Positive tolerance and iteration limit required")
    state, left, derivative = map(np.asarray, (state, left, derivative))
    norm = np.dot(left,state)
    if not np.isfinite(norm) or abs(norm) < 1e-12:
        raise ValueError("Nonzero left/right state overlap required")
    left = left / norm
    project = lambda v: v-state*np.dot(left, v)
    shift = float(np.dot(left, derivative))
    rhs = -project(derivative)
    def apply(v):
        q = project(v)
        return project(action(q)-energy*q)+state*np.dot(left, v)
    operator = LinearOperator((len(state),)*2, matvec=apply, dtype=float)
    diagonal = np.asarray(diagonal)-energy
    scale = np.where(abs(diagonal)>1e-8, diagonal, np.where(diagonal<0, -1e-8, 1e-8))
    preconditioner = LinearOperator(operator.shape,
        matvec=lambda v: project(project(v)/scale)+state*np.dot(left,v), dtype=float)
    history = []
    if np.linalg.norm(rhs) == 0:
        solution, status = np.zeros_like(rhs), 0
    else:
        solution, status = gmres(operator, rhs, M=preconditioner,
            rtol=tolerance, atol=1e-14, restart=min(80,len(state)), maxiter=maxiter,
            callback=history.append, callback_type='pr_norm')
    solution = project(solution)
    residual = float(np.linalg.norm(apply(solution)-rhs))
    reference_residual = float(np.linalg.norm(action(state)-energy*state))
    physical_residual = float(np.linalg.norm(action(solution)-energy*solution+derivative-shift*state))
    orthogonality = float(abs(np.dot(left,solution)))
    if (status or not np.all(np.isfinite(solution))
            or residual > max(1e-13, 10*tolerance*np.linalg.norm(rhs))
            or physical_residual > max(1e-10, 10*tolerance*np.linalg.norm(rhs),
                                       10*reference_residual*(1+np.linalg.norm(solution)))
            or orthogonality > 1e-10*max(1.,np.linalg.norm(solution))):
        raise RuntimeError(f'Projected BSE response failed: status={status}, residual={residual:.3e}')
    return solution, shift, dict(iterations=len(history), residual=residual,
        relative_residual=residual/max(np.linalg.norm(rhs),1e-14),
        orthogonality=orthogonality, physical_residual=physical_residual,
        reference_residual=reference_residual, seconds=perf_counter()-start)
