# Non-Hermitian Davidson

`pyqed.linalg.davidson_nonsymmetric` computes right eigenvectors using a
separate compiled solver; importing it does not load the MPS extension.

```python
from pyqed.linalg import davidson_nonsymmetric

energy, state, info = davidson_nonsymmetric(
    action, roots=8, diag=diagonal, selection="positive_real",
    tolerance=1e-9, space=106,
)
```

Dense and sparse matrices, LinearOperator objects, and vector callbacks are
supported. Callbacks require `diag` and must accept complex vectors. Optional
`matmat` accepts a `(dimension, columns)` block. Eigenvectors are columns.
Failure raises unless `return_partial=True`; inspect `info['converged']` and
`info['residual_norms']` before using partial results. Residuals are checked
with fresh operator applications. `matvecs` counts internal applied vectors;
callback counts additionally include final verification.

## Algorithm and fidelity

This is an adaptation of E. R. Davidson, *J. Comput. Phys.* **17**, 87–94
(1975), [doi:10.1016/0021-9991(75)90065-0](https://doi.org/10.1016/0021-9991(75)90065-0),
with complex nonsymmetric projected eigensolves, diagonal residual corrections,
two-pass orthogonalization, and thick restart. Correction denominators have
a magnitude floor of 1e-8, preserving complex phase. Basis dependence is
tested relative to each candidate's original norm, independently of the
absolute eigenvector residual tolerance.

`smallest_real` uses ordinary Ritz extraction. `smallest_magnitude` and
`positive_real` use harmonic extraction around zero; `target` uses the
specified complex target. Following R. B. Morgan, *Linear Algebra Appl.*
**154–156**, 289–309 (1991),
[doi:10.1016/0024-3795(91)90381-6](https://doi.org/10.1016/0024-3795(91)90381-6),
the small generalized eigenproblem is

$$F^\dagger F y=(\theta-\sigma)F^\dagger V y,
\qquad F=(A-\sigma I)V.$$

The implementation factors $F=QR$ with Householder QR and solves the
algebraically equivalent reciprocal problem

$$C y=\mu y,\qquad C=R^{-1}Q^\dagger V,\qquad
\theta=\sigma+\mu^{-1}.$$

$C$ is formed by a triangular solve, without explicitly forming $R^{-1}$.
This avoids both the normal equations $F^\dagger F$ and a generalized QZ
solve. Only the small projected operator is diagonalized; the physical
operator remains matrix-free. An exactly singular triangular factor raises
instead of silently changing eigensolvers.

The shift is displaced by $10^{-6}(1+|\sigma|)$ along the real axis to reduce
exact-target singularities; this does not guarantee a nonsingular shifted
action for every input. Full-dimensional searches use ordinary
Ritz extraction. `positive_real` retains real parts above `tolerance` with
imaginary magnitude at most `imaginary_tolerance`.

This is not an exact reproduction of either reference or PySCF, nor a
structure-preserving BSE solver. Reference convergence guarantees do not
carry over to this restarted, preconditioned adaptation. There are no left
eigenvectors, locking, biorthogonalization, or guarantees for defective
matrices or completeness of interior roots. Positive selection alone does
not establish physical stability.

The first call compiles and caches the independent extension. macOS uses
Accelerate; Linux requires linkable LAPACK (`-llapack`). Validation here was
on macOS, not Linux. There is no silent Python fallback. `memory_limit`
bounds estimated solver arrays, not allocations made inside user callbacks.
The Hermitian solver is unchanged. Molecular matrix-free full BSE now
defaults to this nonsymmetric solver after the molecular qualification below.

See [the reproducible full-BSE benchmark](bse_nonsymmetric_benchmark.md).


## Repeated-root QZ failure and QR repair

The pre-QR molecular LiH/cc-pVDZ test failed at iteration 12 with a
16-dimensional harmonic pencil after restarting a 30-vector search.
Accelerate `ZGGEV` returned `INFO=12`, indicating QZ nonconvergence
([LAPACK return-code documentation](https://netlib.org/lapack/lapack-3.1.1/html/zggev.f.html)).
The saved pencil was finite, its two matrix condition numbers were about
81 and 5.5, and the physical basis orthogonality error was 1.3e-15.
SciPy solved the same pencil with a residual below 4e-15. Thus this observed
failure was in the small generalized eigensolve, not nonfinite operator
output, loss of basis orthogonality, or BSE physical instability.

Analytic regression matrices formed by orthogonally mixing two identical
stable BSE blocks reproduce the old failure on this macOS host. The QR
reciprocal extraction passes these repeated-root cases, including complex
unitary rotations, with restarts and unchanged absolute residual checks.
It uses the same harmonic Ritz condition cited above; it is not a switch to
ordinary Ritz selection, a dense physical solve, or a looser tolerance.
The full-dimensional and smallest-real branches still use ordinary Ritz.
The original projected-error diagnostic now includes LAPACK status,
iteration, and subspace size.

Reproduce molecular qualification using
`benchmarks/benchmark_bse_molecules.py`; reports and figures for the repair
are in `/private/tmp/bse_real_molecules_repaired` (the QR-only intermediate
run is in `/private/tmp/bse_real_molecules_qr_fix`). Diagnostic pencil data and
the pre-fix reproduction are temporary artifacts in
`/private/tmp/bse_projected_fix`. No molecular wavefunction fixtures are
stored in the repository. The physical BSE default is unchanged by this
numerical repair.


With BSE real-basis recovery for degenerate complex mixtures, both scalar and
batched Davidson now pass all six molecular cases, with 24/24 accepted attempts
each. Maximum lowest-spectrum error is 2.59e-10 Hartree and maximum normalized
residual is 7.53e-10 Hartree at the unchanged 1e-9 target. All 37 focused tests
pass. See the molecular GW/BSE documentation for timings and qualification
limits. Following a further pyrazine/STO-3G dense-reference check,
Davidson is now the molecular matrix-free full-BSE default; ARPACK is
explicitly selectable. See `docs/source/gw_bse.rst` for qualification details.
