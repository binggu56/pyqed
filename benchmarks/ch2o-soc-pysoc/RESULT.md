# Formaldehyde one-electron SOC validation against PySOC/MolSOC

## Claim scope

This record validates PyQED's full multicenter one-electron Breit--Pauli
`p V x p` AO operator for one formaldehyde geometry against the MolSOC kernel
distributed with PySOC 2.3.0.  Both calculations use PySOC's bundled
`mio-1-1` fitted Gaussian basis and the unscreened `ONE` nuclear-charge model.
The three Cartesian AO matrices are compared after accounting for the two
programs' opposite global `p V x p` sign convention and their documented
normalized-versus-raw fitted-basis conventions.  No fitted numerical scale or
matrix rotation is applied.

This is an operator-level correctness comparison.  It does **not** compare
PyQED CASCI state couplings with PySOC LR-TDDFT or TD-DFTB state couplings;
those wavefunction models are different and their state labels are not
interchangeable.

PySOC reference: X. Gao, S. Bai, D. Fazzi, T. Niehaus, M. Barbatti, and
W. Thiel, *J. Chem. Theory Comput.* **13**, 515--524 (2017),
[DOI: 10.1021/acs.jctc.6b00915](https://doi.org/10.1021/acs.jctc.6b00915).
The reference source is [gaox-qd/pysoc](https://github.com/gaox-qd/pysoc),
commit `1a520c682f4851a8ec6e551a04ec5cc89ebc2894`.

## Exact command

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/matplotlib-codex \
PYTHONPATH=. python benchmarks/ch2o-soc-pysoc/run.py \
  --input benchmarks/ch2o-soc-pysoc/input.json \
  --reference benchmarks/ch2o-soc-pysoc/pysoc-reference.json \
  --output benchmarks/ch2o-soc-pysoc/raw-output.json \
  --figure-prefix benchmarks/ch2o-soc-pysoc/soc-integral-comparison
```

## Recorded result

| Quantity | Value |
| --- | ---: |
| Number of AOs | 10 |
| Compared AO tensor elements | 300 |
| Elements above the $10^{-6}$ reporting floor | 202 |
| Maximum PySOC $|p V x p|$ integral | 207.215256463 |
| Maximum absolute residual | 0.0308944251 |
| RMS absolute residual | 0.00450317806 |
| Relative Frobenius residual | $1.47107145\times 10^{-4}$ |
| Acceptance tolerance | $2.0\times 10^{-4}$ |
| Normalized-overlap mapping error | $5.88806805\times 10^{-8}$ |
| Validation | **PASS** |

The component-wise relative Frobenius residuals are
$8.0193\times 10^{-5}$, $1.5441\times 10^{-4}$, and
$1.8740\times 10^{-4}$ for $x$, $y$, and $z$, respectively.  The residual is
consistent with the limited decimal precision written by the PySOC/MolSOC
text interface and with small implementation-constant differences; it is not
removed by fitting.

![PyQED versus PySOC/MolSOC AO SOC integrals](soc-integral-comparison.png)

## Reference-generation notes

The official PySOC Gaussian path could not be used because it requires the
proprietary Gaussian `rwfdump` executable.  Instead, the open-source MolSOC
kernel was compiled with GNU Fortran and run on PySOC's self-contained bundled
formaldehyde TD-DFTB example.  MolSOC's AO stage completed normally and wrote
the overlap and one-electron SOC matrices stored in `pysoc-reference.json`.

The optional downstream `soc_td` executable reproduced its SOC table but then
segfaulted in the unrelated transition-dipole stage on macOS.  No `soc_td`
values are used by this benchmark.  This distinction matters: the passing
claim concerns the AO one-electron operator only, not PySOC's complete
excited-state workflow.

## Recorded environment

- PyQED 0.2.0 from Git commit
  `38eb0f639294ba5f05a839808e916590e756126a`; the source tree was dirty.
- PySOC 2.3.0 / MolSOC 0.1 at commit
  `1a520c682f4851a8ec6e551a04ec5cc89ebc2894`.
- Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, PySCF 2.12.1,
  Matplotlib 3.10.0, and UltraPlot 1.65.1.
- macOS 26.2 / Darwin 25.2.0 on Apple M2 Pro (`arm64`), 12 logical cores,
  and 34359738368 bytes of memory.
- The numerical thread settings were fixed to one.

Because the source tree was dirty, this is development-snapshot evidence.  It
should be regenerated from a clean release tree before use in a release claim.
