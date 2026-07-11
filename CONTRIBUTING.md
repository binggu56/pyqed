# Contributing to PyQED

PyQED welcomes focused bug fixes, tests, documentation, examples, and
scientific-method contributions. The project is research software, so a change
is easiest to review when its scientific scope and numerical evidence are
explicit.

## Before starting

- Search the [issue tracker](https://github.com/binggu56/pyqed/issues) for
  related work.
- Open an issue before a large API change, new dependency, or broad refactor.
- Do not include private datasets, credentials, generated caches, compiled
  artifacts, or machine-specific files.
- Keep a pull request focused. Unrelated cleanup makes scientific review harder.

## Development setup

From a source checkout:

```bash
python -m pip install -e .
python -m pip install pytest
```

Use `PYTHONPATH=.` when running scripts or tests from the repository root. For
numerical tests, avoid accidental thread oversubscription:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. python -m pytest tests/test_rhf.py -q
```

Documentation contributors can build locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs/source /tmp/pyqed-docs
```

## What a scientific change needs

A method, solver, or numerical-backend change should document:

1. the physical quantity and mathematical convention being implemented;
2. input and output units;
3. supported spin, symmetry, boundary-condition, and representation cases;
4. a focused regression test against an analytic result, an independent
   implementation, or a deliberately small dense reference;
5. tolerances and why they are appropriate;
6. the primary method references; and
7. limits that remain untested or experimental.

New capabilities begin as **Experimental** unless maintainers explicitly
promote them according to the criteria in `docs/source/capabilities.rst`.

## Examples and benchmarks

- Keep a small smoke example separate from an expensive research calculation.
- Make random seeds, input geometry, basis, grid, and solver tolerances explicit.
- Record the PyQED version or Git commit and whether the tree is dirty.
- Record hardware, operating system, Python, dependencies, BLAS, and thread
  settings for performance results.
- Store benchmark metadata using the templates in `benchmarks/`.
- Never describe a result as a general speedup or accuracy guarantee when it
  covers only one workload.

## Pull-request checklist

- [ ] The change is focused and contains no unrelated generated files.
- [ ] Relevant focused tests pass.
- [ ] New behavior is documented.
- [ ] Public API changes have an upgrade note in `HISTORY.rst`.
- [ ] Scientific conventions, units, and limitations are stated.
- [ ] New performance or accuracy claims include reproducible provenance.

By participating, contributors agree to follow `CODE_OF_CONDUCT.md`.
