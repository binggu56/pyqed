# PyQED governance

PyQED is an open-source research-software project. This document records how
technical decisions are made; it does not imply an institutional endorsement.

## Roles

- **Contributors** propose changes through issues and pull requests.
- **Reviewers** evaluate code, tests, documentation, numerical evidence, and
  scientific scope.
- **Maintainers** merge changes, manage releases, assign capability maturity,
  and handle conduct and security reports.

The repository currently identifies Bing Gu and Zihao Chen (DEOM) as project
developers. Until a broader maintainer roster is recorded here, the repository
owner has final merge and release responsibility.

## Decision process

Routine, reversible changes use review and lazy consensus: maintainers allow a
reasonable opportunity for relevant contributors to object, then merge when
the evidence and scope are clear. The following changes require an issue or
proposal before implementation:

- incompatible public-API or file-format changes;
- a new required dependency or supported platform policy;
- changes to numerical conventions or default physical models;
- promotion of a capability from Experimental to Beta or Stable;
- release, licensing, governance, or security-policy changes.

When consensus is not possible, the responsible maintainer records the
decision, alternatives considered, and supporting tests or references. A
decision can be revisited when new evidence appears.

## Scientific review

Code review is not peer review of an underlying scientific method. Method
implementations must state their conventions and validation domain. A
maintainer may request an analytic limit, independent reference calculation,
or small dense comparison before merging. Maturity labels describe software
evidence and maintenance, not universal scientific validity.

## Releases

A release should have a single version, a reproducible source archive and
wheel, release notes, clean-install tests, and a documented capability status.
Publication to a package index occurs only after the release gates pass.

## Changes to governance

Governance changes are proposed through a pull request. Material changes should
be announced in the issue tracker and retained in repository history.
