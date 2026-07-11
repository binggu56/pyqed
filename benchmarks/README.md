# PyQED benchmark records

This directory defines how PyQED validation and performance results are
recorded. It does not contain a general performance claim.

Files:

- `catalog.json` lists tracked benchmark entry points and their narrow claim
  scope.
- `benchmark-manifest.schema.json` validates completed machine-readable result
  manifests.
- `manifest.template.json` is intentionally invalid until every `REPLACE_ME`
  value is replaced with observed metadata.
- `RESULT_TEMPLATE.md` is the human review summary stored beside raw output and
  a completed manifest.

## Workflow

1. Start from a clean checkout and record `git rev-parse HEAD` plus
   `git status --short`.
2. Copy `manifest.template.json` into a result-specific directory outside the
   source tree while running exploratory work.
3. Record inputs, versions, hardware, BLAS, threads, tolerances, warm-up, and
   repeat policy before interpreting results.
4. Preserve raw output and hash every published input/output artifact.
5. Replace every placeholder, validate against
   `benchmark-manifest.schema.json`, and complete `RESULT_TEMPLATE.md`.
6. Review the stated claim scope. One workload cannot support a general claim
   about all molecules, models, hardware, or methods.

Recommended local environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. python path/to/benchmark.py
```

Use different thread settings only when parallel scaling is itself part of the
benchmark, and record every setting.

## Candidate validation records

- [`h2-sto3g-rhf-pyscf`](h2-sto3g-rhf-pyscf/run.py) compares one native
  PyQED RHF total energy with PySCF. The script and exact input are tracked;
  a completed result, manifest, and hashes are added only after running from a
  clean, identified source commit.
