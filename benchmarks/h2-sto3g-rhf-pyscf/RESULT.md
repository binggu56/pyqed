# H2/STO-3G native RHF validation against PySCF

## Claim scope

This record validates only the total restricted Hartree–Fock energy for an H2
molecule with a 1.4 bohr bond length in the STO-3G basis.  PyQED uses its
`builtin` integral driver, dense spherical integrals, the Rys backend, and its
native RHF solver.  PySCF supplies independent molecular integrals and an
independent RHF implementation.

This is a correctness comparison, not a timing result or a general accuracy or
performance claim.

## Exact command

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python benchmarks/h2-sto3g-rhf-pyscf/run.py --input benchmarks/h2-sto3g-rhf-pyscf/input.json --output benchmarks/h2-sto3g-rhf-pyscf/raw-output.json
```

## Recorded result

| Quantity | Value |
| --- | ---: |
| PyQED total RHF energy | -1.1167143251757676 hartree |
| PySCF total RHF energy | -1.116714325062552 hartree |
| Absolute difference | 1.1321565906996511e-10 hartree |
| Acceptance tolerance | 1e-9 hartree |
| Validation | PASS |

Both calculations converged.  PyQED reported four SCF iterations and evaluated
all six symmetry-unique ERI quartets without screening.

An exploratory run used a provisional `1e-10`-hartree cutoff and missed that
cutoff by `1.321565906996511e-11` hartree.  The final published cutoff was set
to `1e-9` hartree, matching the repository's existing external RHF energy
comparison scale, before the recorded run was produced.  The observed
difference is preserved at full precision above and in `raw-output.json`.

## Recorded environment

- PyQED 0.2.0 from the working tree at Git commit
  `0d3b6203f577ca42e6cd3f9f71f734ffc6bff6b2`; the tree was dirty.
- PySCF 2.12.1, NumPy 2.1.3, SciPy 1.15.3, Python 3.13.5.
- macOS/Darwin 25.2.0 on an Apple M2 Pro (`arm64`), 12 logical cores, and
  34359738368 bytes of memory.
- OpenBLAS 0.3.30; OpenBLAS, OpenMP, Accelerate, and NumExpr thread settings
  were each fixed to one.

Because the source tree was dirty, this is a transparent development-snapshot
record rather than release-level evidence.  Re-run it after the release tree is
clean before quoting the result in a release announcement.
