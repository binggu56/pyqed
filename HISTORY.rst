=======
History
=======

Unreleased
----------

0.3.0 (2026-09-02)
------------------

* Ship compiled qchem wheels for CPython 3.10--3.13 on Linux, macOS, and
  Windows.
* Build the production qchem accelerators by default for source installs while
  retaining an explicit Python reference-kernel mode for debugging.
* Reject silent fallback from production integral paths to slow Python kernels
  and add ``python -m pyqed.qchem.check_install`` for installation checks.
* Build, test, and publish the platform wheels and source distribution through
  the tagged-release workflow.

0.2.0 (2026-07-11)
------------------

* Reorganized the documentation around installation, a tested quickstart,
  task-oriented guides, examples, API entry points, benchmarks, citation, and
  development.
* Added project governance, support, security, contribution, citation, and
  benchmark-reporting guidance.
* Modernized packaging for Python 3.10--3.13 with a single version source,
  declared dependency groups, portable pure-Python wheels, and opt-in native
  accelerators.
* Added clean-install, quickstart, documentation, native-extension, and tagged
  PyPI release workflows.
* Added a reproducible H2/STO-3G RHF comparison against PySCF, with a
  machine-readable manifest and narrow validation claim.

0.1.0 (2018-08-11)
------------------

* First release on PyPI.
