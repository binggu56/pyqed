=======
History
=======

Unreleased
----------

* Corrected ``LegendreDVR.t()`` to use the positive-semidefinite derivative
  quadratic form, rather than symmetrizing the squared cardinal derivative.
  Its flat-coordinate convention is now explicitly documented as natural
  Neumann boundaries. This changes previously incorrect kinetic energies;
  it does not supply the angular rotor metric or hard-wall boundaries.
  The grid, basis and momentum operator (including triatomic callers) are
  unchanged.
* Removed a stale ``LegendreDVR`` potential placeholder that shadowed the
  inherited ``v(V)`` method and broke ``h(V)`` and ``run(V)``.

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

0.1.5 (2022-10-25)
------------------

* Maintenance wheel published to PyPI; detailed change notes were not
  recorded in the historical project changelog.

0.1.4 (2022-10-25)
------------------

* Maintenance wheel published to PyPI; detailed change notes were not
  recorded in the historical project changelog.

0.1.3 (2022-10-25)
------------------

* Maintenance wheel published to PyPI; detailed change notes were not
  recorded in the historical project changelog.

0.1.2 (2022-10-14)
------------------

* Maintenance wheel published to PyPI; detailed change notes were not
  recorded in the historical project changelog.

0.1.1 (2022-07-21)
------------------

* Maintenance wheel published to PyPI; detailed change notes were not
  recorded in the historical project changelog.

0.1.0 (2018-08-11)
------------------

* First release on PyPI.
