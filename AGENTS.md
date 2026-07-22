# Agent Guidance

## Scope
- This file applies to the whole repository.
- Prefer project-local patterns and focused changes over broad refactors.
- The worktree may contain user changes. Do not revert or overwrite unrelated edits.

## Compatibility
- Don't worry about compatibility unless the user explicitly asks for it.
- Prefer clean current behavior over preserving legacy APIs, old Python versions, or historical package metadata.

## Local Development
- Use `PYTHONPATH=.` when running examples, scripts, or tests from the repository root.
- Keep generated caches and heavy outputs outside the repo when practical, such as under `/private/tmp`.
- For numerical or chemistry runs, limit BLAS/OpenMP thread fanout when running locally:
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

## Testing
- Prefer focused `pytest` runs for the files or behavior touched.
- Avoid running expensive qchem, NAMD, or large benchmark examples unless they are directly relevant or the user requests them.
- If a full-suite or long-running validation is needed, say what will be run before starting it.

## SU2 / Non-Abelian NARG Guardrail
- Do not fix SU2 or non-Abelian NARG energy/recoupling discrepancies by adding final dense variational projection, projected-growth Hamiltonians, primitive branch-basis growth, or `4^n` determinant-space operator projection to the active solver path.
- Such projections are allowed only as explicit reference/validation helpers in tests or benchmarks.
- The active solver should carry reduced-sector operators, composites, environments, or recoupling data through truncation. If projected growth and recursive reduced growth disagree, debug the reduced representation instead of patching with projection.
- Add or update reduced-recursive-vs-projected/dense reference tests before removing any reference path.

## Code Style
- Keep changes readable and direct. Add abstractions only when they reduce real duplication or complexity.
- Do not casually create separate `Result` classes for solver/driver workflows. Prefer populating the solver/driver object with fields such as `energy`, `state`, `history`, `success`, and `message`; add a result object only when it clearly improves ownership, immutability, or composition.
- When optimizing code, prioritize architecture and data flow first; leave micro-optimization until the very end.
- Use structured numerical APIs and existing helper functions instead of ad hoc parsing or manual array manipulation.
- Add comments only where the intent would otherwise be hard to recover.

## Math Formatting
- When showing equations or derivations to the user, always use Markdown math format with inline `$...$` or display `$$...$$` blocks instead of plain-text equations.
