"""Compiled restarted Davidson for general right-eigenvector problems."""
from pathlib import Path
import sys
import numpy as np
from ._extension import load_extension

_module=None
build_error=None


def davidson_nonsymmetric(matrix, roots=1, *, diag=None, matmat=None, guess=None,
                         selection='smallest_real', target=0., tolerance=1e-9,
                         iterations=200, space=None, imaginary_tolerance=1e-8,
                         dependence_tolerance=1e-12, memory_limit=512*1024**2,
                         return_partial=False):
    """Return eigenvalues, right eigenvectors, and convergence diagnostics.

    Adaptation of E. R. Davidson, J. Comput. Phys. 17, 87–94 (1975),
    doi:10.1016/0021-9991(75)90065-0: diagonal residual corrections with a
    nonsymmetric Rayleigh–Ritz solve and thick restart. This is not an exact
    reproduction of the original Hermitian algorithm, PySCF, or a BSE-specific
    structure-preserving solver. Two-pass orthogonalization uses a relative
    rank test independent of the absolute residual tolerance. No left vectors,
    locking, biorthogonalization, or guarantees for defective matrices or
    completeness of interior/positive roots are provided.

    Interior selections use harmonic Ritz extraction, following R. B. Morgan,
    Linear Algebra Appl. 154–156, 289–309 (1991),
    doi:10.1016/0024-3795(91)90381-6. This adaptation uses the small generalized
    pencil F.H F y = (theta-shift) F.H V y, F=(A-shift)V. It factors F=QR
    and solves the equivalent reciprocal ordinary problem
    R^{-1} Q.H V y = mu y, theta=shift+1/mu, using a triangular solve,
    without forming an inverse or normal equations. This avoids generalized
    QZ failures observed on repeated molecular roots. The displaced target
    reduces exact-target singularities; rank-deficient shifted actions raise.
    A full-dimensional
    search uses ordinary Ritz extraction. Reference convergence bounds are
    not claimed for this restarted, preconditioned implementation.

    ``selection``: smallest_real, smallest_magnitude, positive_real, or target
    (closest to ``target``). Positive-real selection filters projected roots;
    it excludes real parts <= tolerance and does not assert physical stability.
    Always uses complex arithmetic.
    Callables require a diagonal and must accept complex trial vectors.
    ``matmat`` receives (dimension, columns); otherwise vector calls are used.
    macOS uses Accelerate; Linux requires a linkable LAPACK (-llapack).
    Workspace limit covers estimated solver arrays, not callback allocations.
    Failure raises unless return_partial=True; fresh action residuals are
    checked before convergence is reported.
    """
    global _module,build_error
    choices={'smallest_real':0,'smallest_magnitude':1,'positive_real':2,'target':3}
    if selection not in choices:raise ValueError('Invalid eigenvalue selection')
    for name,value in [('roots',roots),('iterations',iterations),('memory_limit',memory_limit)]:
        if isinstance(value,bool) or int(value)!=value or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid tolerance')
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance<0:raise ValueError('Invalid imaginary tolerance')
    if not np.isfinite(dependence_tolerance) or not 0<dependence_tolerance<1:raise ValueError('Invalid dependence tolerance')
    if not np.isfinite(target):raise ValueError('Nonfinite target')
    if hasattr(matrix,'matvec'):
        action=matrix.matvec
        if matmat is None:matmat=getattr(matrix,'matmat',None)
    elif hasattr(matrix,'tocsr'):
        action=matrix.dot
        if matmat is None:matmat=matrix.dot
        if diag is None:diag=matrix.diagonal()
    elif callable(matrix):action=matrix
    else:
        matrix=np.asarray(matrix)
        if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not np.all(np.isfinite(matrix)):
            raise ValueError('Expected a finite square matrix')
        action=matrix.dot
        if matmat is None:matmat=matrix.dot
        if diag is None:diag=matrix.diagonal()
    if diag is None:raise ValueError('Matrix-free Davidson requires diag')
    diag=np.asarray(diag,dtype=complex)
    if diag.ndim!=1 or not np.all(np.isfinite(diag)):raise ValueError('Invalid diagonal')
    n=len(diag)
    if hasattr(matrix,'shape') and matrix.shape!=(n,n):raise ValueError('Operator/diagonal shape mismatch')
    if space is not None and (isinstance(space,bool) or int(space)!=space or space<1):raise ValueError('Invalid space')
    space=min(n,max(24,6*int(roots)) if space is None else int(space))
    if roots>n or space<roots or (space==roots and space<n):raise ValueError('Invalid root/subspace dimensions')
    estimate=16*(6*n*space+12*n*roots+12*space*space+10*n)
    if estimate>memory_limit:raise MemoryError('Nonsymmetric Davidson workspace exceeds memory limit')
    if matmat is not None and not callable(matmat):raise ValueError('matmat must be callable')
    counts={'matvec_callback_calls':0,'matmat_callback_calls':0}
    def block(x):
        if matmat is None:
            counts['matvec_callback_calls']+=x.shape[1]
            y=np.column_stack([action(v) for v in x.T])
        else:
            counts['matmat_callback_calls']+=1
            y=matmat(x)
        y=np.asarray(y,dtype=complex)
        if y.shape!=x.shape or not np.all(np.isfinite(y)):raise ValueError('Invalid operator action shape or values')
        return y
    if _module is None:
        directory=Path(__file__).parent
        _module,build_error=load_extension('pyqed.linalg._nonsymmetric',directory/'nonsymmetric_module.cpp',
            [directory/'nonsymmetric_davidson.hpp',directory/'davidson.hpp'],
            link_flags=('-llapack',) if sys.platform.startswith('linux') else ())
    if _module is None:raise ImportError(f'Non-Hermitian Davidson unavailable: {build_error}')
    values,vectors,info=_module.solve(block,np.asfortranarray(diag),int(roots),float(tolerance),
        int(iterations),space,int(memory_limit),guess,choices[selection],complex(target),
        float(imaginary_tolerance),float(dependence_tolerance))
    values=np.asarray(values,dtype=complex)
    residuals=np.linalg.norm(block(vectors)-vectors*values,axis=0) if len(values) else np.empty(0)
    info['projected_residual_norms']=info['residual_norms']
    info.update(counts,backend='compiled',selection=selection,residual_norms=residuals,
                converged=bool(info['converged'] and len(values)==roots and np.all(residuals<=tolerance)))
    if not info['converged'] and not return_partial:
        raise RuntimeError(f"Non-Hermitian Davidson did not converge ({info['iterations']} iterations; residuals {residuals})")
    return values,vectors,info
