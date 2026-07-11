# PyQED roadmap

This roadmap communicates priorities, not delivery dates or guarantees. Work is
accepted according to available maintainers, review capacity, and scientific
evidence.

## Now: reproducible release foundation

- Establish one authoritative package version and modern build metadata.
- Make a clean wheel install reproduce the documented five-minute calculation.
- Test supported Python environments and documentation links in automation.
- Separate generated, machine-specific, and research-output files from release
  artifacts.
- Publish explicit capability maturity and optional-dependency boundaries.

## Next: validation and learning paths

- Turn representative workflows into small, deterministic, tested examples.
- Add structured benchmark manifests with commit, dependency, hardware, thread,
  input, tolerance, and artifact provenance.
- Expand task-oriented tutorials for quantum chemistry, grid dynamics,
  open-system dynamics, spectroscopy, and geometric/nonadiabatic dynamics.
- Connect API pages to examples and tests that demonstrate each supported path.
- Archive releases and add a DOI only when the archive exists.

## Later: community and interface stability

- Define stable public interfaces and deprecation policy from observed usage.
- Broaden platform testing and optional compiled-wheel coverage.
- Maintain method-specific validation suites and versioned reference datasets.
- Grow the maintainer roster and document ownership of scientific subsystems.
- Publish release notes and research notes on a predictable, sustainable cadence.

Requests and proposals belong in the
[issue tracker](https://github.com/binggu56/pyqed/issues). A roadmap item is not
considered complete until its tests, documentation, and provenance are merged.
