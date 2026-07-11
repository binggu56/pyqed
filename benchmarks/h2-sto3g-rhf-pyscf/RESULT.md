# H2/STO-3G native RHF validation against PySCF

## Claim scope

This record validates only the total restricted Hartree–Fock energy of H2 at
a 1.4 bohr bond length in the STO-3G basis. PyQED used its `builtin` integral
driver, dense spherical integrals, the requested Rys backend, the observed
`rys-cython-blocked` integral builder, and its native RHF solver. PySCF supplied
independent molecular integrals and an independent RHF implementation.

This is one correctness comparison. It is not a timing result and does not
support a general performance or accuracy claim for other systems, bases,
methods, platforms, or backend configurations.

## Source provenance

Immediately before execution, the checkout was clean and
`git rev-parse HEAD` returned:

```text
7dbb9bcc6625d9e4030627140dd14738c60a0e67
```

The result therefore records `git_dirty: false`. Generating the result then
created the benchmark output files described by the manifest.

## Exact command

Run from the checkout root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore MPLCONFIGDIR=/private/tmp/pyqed-benchmark-mpl-7dbb9bc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python benchmarks/h2-sto3g-rhf-pyscf/run.py --input benchmarks/h2-sto3g-rhf-pyscf/input.json --output benchmarks/h2-sto3g-rhf-pyscf/raw-output.json
```

## Recorded result

| Quantity | Value |
| --- | ---: |
| PyQED total RHF energy | -1.1167143251757676 hartree |
| PySCF total RHF energy | -1.116714325062552 hartree |
| Absolute difference | 1.1321565906996511e-10 hartree |
| Acceptance tolerance | 1e-9 hartree |
| Validation | PASS |

Both calculations converged. PyQED reported four SCF iterations and evaluated
all six symmetry-unique ERI quartets without screening.

## Recorded environment

- PyQED 0.2.0 from clean source commit
  `7dbb9bcc6625d9e4030627140dd14738c60a0e67`.
- PySCF 2.12.1, NumPy 2.1.3, SciPy 1.15.3, and Python 3.13.5.
- macOS 26.2 / Darwin 25.2.0 on an Apple M2 Pro (`arm64`), 12 logical
  cores, and 34359738368 bytes of memory.
- OpenBLAS 0.3.30; OpenBLAS, OpenMP, Accelerate, and NumExpr thread settings
  were each fixed to one.
- The checkout supplied tracked CPython 3.13 macOS arm64 `_rys_cy` and
  `_basis_cy` extension modules. Their exact SHA-256 hashes are recorded as
  execution-input artifacts in `manifest.json`.
