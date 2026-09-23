# Non-Hermitian Davidson for full molecular BSE

This compares PySCF's existing `davidson_nosym1` and, with `--compiled`,
PyQED's compiled `davidson_nonsymmetric`. It does not change `solve_bse`
dispatch. PySCF implementation/API reference:
https://pyscf.org/pyscf_api_docs/pyscf.lib.html#pyscf.lib.linalg_helper.davidson_nosym1

`benchmarks/benchmark_bse_nonsymmetric.py` reuses the water/cc-pVTZ factors
from `benchmark_bse_batched_action.py`. These use density-fitted RHF orbital
gaps and static TDH screening, **not GW quasiparticle corrections**. The full
BSE operator has dimension 530; eight lowest positive-real roots are sought.
A small dense expansion validates the spectrum and Euclidean-normalized
right-eigenvector residuals. It is not used by either iterative solve.

All solvers call exactly the same scalar BSE action. PySCF's list callback
loops over it; no batched contraction or compiled action advantage is included.
The Davidson initial space consists of positive-branch low-gap coordinate
vectors, with the signed orbital-gap diagonal as preconditioner. The custom
projected-root picker retains positive eigenvalues with imaginary part below
1e-8. It is restricted to this stable real BSE benchmark; it does not establish
robust handling of unstable/complex spectra, defective matrices, or root
completeness for general non-Hermitian problems. Left eigenvectors and
biorthogonality are not tested. No universal speedup is claimed.

## Measured results (2026-09-23)

Single BLAS/OpenMP thread, median of three accepted solves:

| Solver | Time | Applied vectors | Maximum residual (Hartree) |
|---|---:|---:|---:|
| ARPACK, default space (41) | 23.222 s | 3792 | 3.14e-14 |
| ARPACK, space 106 | 16.192 s | 2753 | 4.95e-14 |
| PySCF Davidson, adjusted dependence threshold | 2.146 s | 339 | 5.12e-10 |
| PyQED compiled Davidson, harmonic Ritz | 1.591 s | 97 | 5.40e-10 |

The compiled solver's maximum root error is 4.64e-12 Hartree. It is 1.35x
faster than tuned PySCF and 10.18x faster than larger-space ARPACK here.
Its 97 applied vectors include fresh final residual verification. It uses
complex arithmetic, harmonic Ritz extraction around zero, a relative basis
dependence threshold of 1e-12, and space 106. Algorithm and initialization
differences mean this is not a measurement of C++ translation alone.
See [solver details and limitations](nonsymmetric_davidson.md).

All eight roots match the dense reference: Davidson's maximum root error is
3.12e-13 Hartree. Davidson is 7.55x faster than larger-space ARPACK on this case.
ARPACK requests 20 smallest-magnitude roots before selecting the positive
eight (`which='SM'`, tolerance 1e-9, fixed random initial vector, maxiter 1000).
Both reported ARPACK variants converge all 20 search roots. A preliminary
200-iteration run converged only 17/20, although its desired eight roots were
already accurate; that limited run is not used for the timing comparison.
Davidson uses `max_space=64` (PySCF expands this to 106 for eight roots),
max_cycle 200, energy tolerance 1e-12, and residual tolerance 1e-9.
The stopping rules differ: ARPACK substantially overconverges the selected
roots. These timings compare validated workloads, not identical stopping
algorithms or equal target-root counts.

### Dependence threshold matters

PySCF rejects correction vectors when their squared residual is below
`lindep`. Its default 1e-14 stalled here at a 6.84e-8 residual: the 1.277 s
trial **failed** the requested accuracy and is not counted as a speedup.
An exploratory 1e-22 setting became numerically unstable. Setting `lindep`
to 1e-18 converged in all three measured runs. This is empirical tuning for
this case, not a generally safe formula; any new solver needs orthogonality
guards, residual checks, and difficult-spectrum tests before default dispatch.

Artifacts are in the sibling molecular database:
`database/h2o/full_bse_nonsymmetric_20260923/` (`report.json` and
`water_full_bse_nonsymmetric.png`). The script saves failed checks explicitly
and hatches failed cases in the plot. `--resume` retains completed methods
after verifying the input path/dimension and recomputed reference roots.
`solve_bse` continues to use ARPACK; no production solver was switched.
