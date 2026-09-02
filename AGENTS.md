# Agent Guidance

## Scope
- This file applies to the whole repository.
- Prefer project-local patterns and focused changes over broad refactors.
- The worktree may contain user changes. Do not revert or overwrite unrelated edits.

## Compatibility (No compatibility)
- No compatibility: don't preserve old APIs/behavior unless the user explicitly asks.
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

## Calculation Figures
- After completing a numerical calculation, always generate the relevant diagnostic or result figures from its outputs and display them to the user before reporting completion.
- Save the figures with clear, calculation-specific filenames and state where they were written.
- Include plotting code in the calculation script or provide a companion plotting script so the figures are reproducible.

## SU2 / Non-Abelian NARG Guardrail
- Do not fix SU2 or non-Abelian NARG energy/recoupling discrepancies by adding final dense variational projection, projected-growth Hamiltonians, primitive branch-basis growth, or `4^n` determinant-space operator projection to the active solver path.
- Such projections are allowed only as explicit reference/validation helpers in tests or benchmarks.
- The active solver should carry reduced-sector operators, composites, environments, or recoupling data through truncation. If projected growth and recursive reduced growth disagree, debug the reduced representation instead of patching with projection.
- Add or update reduced-recursive-vs-projected/dense reference tests before removing any reference path.

## Code Style
- Keep changes readable and direct. Add abstractions only when they reduce real duplication or complexity.
- When adding new code, make it compact.
- For new implementations, do not use backend-oriented `cpp_*` or `native_*` names for files, modules, symbols, capability flags, profiling fields, or configuration keys. Name components after the algorithm or capability they provide, such as `davidson`, `moving_environment`, or `tdvp_kernels`.
- Do not casually create separate `Result` classes for solver/driver workflows. Prefer populating the solver/driver object with fields such as `energy`, `state`, `history`, `success`, and `message`; add a result object only when it clearly improves ownership, immutability, or composition.
- When optimizing code, prioritize architecture and data flow first; leave micro-optimization until the very end.
- Use structured numerical APIs and existing helper functions instead of ad hoc parsing or manual array manipulation.
- Add comments only where the intent would otherwise be hard to recover.

## Method References and Implementation Fidelity
- When implementing a method from the literature, add the primary references to the relevant user documentation and to the main public class or function docstring. Include enough bibliographic information to identify the work unambiguously, preferably with a DOI or stable URL.
- In the same documentation, state whether the implementation is an exact reproduction, an adaptation, or a simplified/inspired variant. Describe material approximations, omitted couplings or response terms, restricted cases, unsupported features, and any reference convergence or accuracy guarantees that therefore do not carry over.
- For hybrid methods, identify the reference or established formulation used for each major algorithmic component. Do not describe a method solely by a literature name when the implemented algorithm differs materially from that reference.
- Treat reference and fidelity documentation as part of the implementation: add or update it in the same change as the code, and keep it synchronized when the algorithm changes.

## Math Formatting
- When showing equations or derivations to the user, always use Markdown math format with inline `$...$` or display `$$...$$` blocks instead of plain-text equations.
