OM2/MRCI
========

PyQED exposes an ``OM2``/``MRCI`` API for semiempirical excited-state
workflows.  The API is intentionally separated into two layers:

* ``OM2`` builds the semiempirical reference Hamiltonian and orbitals.
* ``MRCI`` diagonalizes the selected multireference CI Hamiltonian.

Current Status
--------------

The native API now includes a runnable closed-shell OM2-style reference and
MRCI driver:

* Built-in H/C/N/O/F OM2 parameter table from Dral et al.,
  J. Chem. Theory Comput. 12, 1082 (2016), Table 2, including the X-H
  resonance rows, orthogonalization factors, and semiempirical ECP rows.
* Orthogonal valence AO Hamiltonian data with one-electron terms, compact
  NDDO-like electron repulsion terms, core attraction, semiempirical ECP
  corrections, additive OM2 orthogonalization corrections, and core repulsion.
* Closed-shell SCF reference energies and orbitals.
* Full or selected determinant-space MRCI using the PyQED Slater-Condon CI
  builder.
* Scanner and CI-vector pseudo-overlap helpers for PES/LDR workflows.

The current integral kernel is intentionally modular.  It now uses the
published X-H resonance rows, semiempirical ECP rows, and explicit additive
F1/F2 and G1/G2 orthogonalization-correction contractions.  It is still not
bit-for-bit MNDO2005 OM2 because the native overlap/local-core primitives are
compact analytic approximations.  The next physics step is replacing those
primitive overlap/local-core kernels by the exact OM2 two-center primitives
while keeping the same user API.

API Sketch
----------

.. code-block:: python

   from pyqed.qchem.semiempirical import OM2

   om2 = OM2(
       atom="C 0 0 0; H 0 0 1.09",
       charge=0,
       spin=0,
       unit="angstrom",
   ).run()

   mrci = om2.MRCI(nstates=5).run()
   print(mrci.e)

``MRCI`` is intentionally both the driver and the result object.  After
``run()``, energies and CI vectors are available as ``mrci.e`` and ``mrci.ci``;
there is no separate ``MRCIResult`` wrapper.

For PES scans, the same interface will be usable as a callable scanner:

.. code-block:: python

   scanner = OM2(atom="H 0 0 0; H 0 0 0.74").as_scanner(
       nstates=2,
       full=True,
   )
   result = scanner(atom="H 0 0 0; H 0 0 0.80")
   print(result.e)

Wavefunction pseudo-overlaps can be computed directly from two completed MRCI
objects:

.. code-block:: python

   s = mrci1.wavefunction_overlap(mrci2)

Published Benchmarks
--------------------

PyQED ships a small registry of published OM2 benchmark targets from the OMx
papers.  It includes aggregate MAEs plus selected molecule-level
supporting-information values from the G2-CHNOF and S22 tables.  These values
are heats of formation or interaction energies rather than total electronic
energies, so they are useful for tracking the expected accuracy envelope while
a direct MNDO executable benchmark is unavailable.

.. code-block:: python

   from pyqed.qchem.semiempirical import (
       format_published_om2_benchmarks,
       format_published_om2_molecule_benchmarks,
   )

   print(format_published_om2_benchmarks())
   print(format_published_om2_molecule_benchmarks())

The same registry is used by ``examples/qchem/benchmark_om2_published.py``.

Implementation Roadmap
----------------------

1. Replace the compact approximate integral kernel with the exact OM2
   orthogonalization/ECP Hamiltonian.
2. Add open-shell/UHF OM2 references.
3. Add production MRCI reference selection and perturbative selection
   thresholds.
4. Add transition dipoles and oscillator strengths.
5. Add analytic or robust finite-difference gradients after energies/overlaps
   are stable.
