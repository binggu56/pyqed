Backends and Integral Representations
=====================================

PyQED supports multiple backend and integral-storage choices. The right choice
depends on whether a calculation needs small-molecule simplicity, dense tensor
access, or factorized contractions for larger systems.

Molecule Build Drivers
----------------------

The molecular build step selects the integral backend:

.. code-block:: python

   mol.build(driver="builtin", eri="auto")

Common driver choices:

* ``driver="builtin"`` uses PyQED's native molecular-integral path.
* ``driver="gbasis"`` uses the gbasis-based path when that dependency is
  installed.
* External packages such as PySCF are optional and are mainly useful for
  validation, comparison, or features that are not yet native.
* ``eri="auto"`` uses compact eight-fold exact storage for small native builds
  and switches to native RI/factorized storage for larger AO spaces when an
  auxiliary basis is available.

Electron-Repulsion Storage
--------------------------

The ``eri`` keyword controls the ERI representation family, and ``aosym``
controls AO permutation symmetry for dense-like storage:

* ``eri="dense", aosym="s1"`` stores the dense four-index tensor.
* ``eri="dense", aosym="s4"`` stores unique AO-pair rows and columns.
* ``eri="dense", aosym="s8"`` stores only unique AO-pair-pair values,
  exploiting the full eight-fold ERI permutation symmetry for memory.
* ``eri="direct"`` avoids dense AO ERI construction and uses compact ``s8``
  storage for cartesian J/K builds.
* ``eri="factors"`` stores a Cholesky/factorized representation.
* ``eri="ri"`` stores native density-fitting factors. The default auxiliary
  basis policy prefers JKFIT sets for SCF when available; use
  ``options={"ri_purpose": "ri"}`` to prefer RIFIT sets.
* ``eri="dense+factors"`` stores both dense and factorized representations.

Legacy shortcuts such as ``eri="s8"`` and ``eri="s8+factors"`` are still
accepted and normalize to ``eri="dense", aosym="s8"`` and
``eri="dense+factors", aosym="s8"`` respectively.

When to Use Dense Integrals
---------------------------

Full dense integrals are simplest and useful for:

* very small molecules
* debugging new methods
* algorithms that explicitly require ``(pq|rs)`` tensor access
* reference comparisons against dense implementations

The drawback is memory scaling. Dense four-index tensors become expensive as
the number of orbitals grows. For exact RHF calculations that do not need direct
``mol.eri`` tensor access, prefer ``eri="auto"`` or ``eri="dense",
aosym="s8"`` so J/K contractions use the compact packed path.

When to Use Factorized Integrals
--------------------------------

Factorized integrals are preferred for:

* larger basis sets
* RHF with Cholesky/factorized JK builds
* CASCI/CASSCF paths that can contract directly with factors
* workflows where avoiding transformed dense MO ERIs matters

Example:

.. code-block:: python

   mol.build(driver="builtin", eri="factors")
   mf = mol.RHF().run()

   # Factor-aware solvers can reuse mf.eri_factors instead of dense ERIs.
   mc = mol.CASSCF(mf, ncas=4, nelecas=4).run()

Native RI builds the three-center tensor in compact AO-pair form, then stores
SCF factors in full tensor form by default because that is currently the faster
RHF contraction path. Use ``ri_storage="packed"`` for memory-sensitive runs.
The metric solver uses a Cholesky factorization when the auxiliary Coulomb
metric is positive definite and falls back to an eigenvalue solver for
near-singular metrics. Useful RI options include ``ri_metric_solver="eigh"``
for a forced spectral solve, ``ri_screen_tol`` for three-center screening, and
``ri_block_size`` for the metric solve block size.

Optional Dependencies
---------------------

Some modules use optional compiled or third-party backends:

* ``libxc`` is used by parts of the native DFT stack.
* ``gbasis`` is used by the gbasis molecular-integral path.
* ``pyscf`` is useful for benchmarking and cross-validation.
* plotting and visualization examples may require packages such as PyVista.

Read the Docs does not need these optional dependencies for the static guide
pages. API pages that would import heavy optional backends are intentionally
kept static or excluded from the RTD build.

Recommended Defaults
--------------------

For native quantum chemistry examples:

.. code-block:: python

   mol.build(driver="builtin", eri="auto")

For debugging a new tensor formula:

.. code-block:: python

   mol.build(driver="builtin", eri="dense", aosym="s1")

For comparing factorized and dense algorithms:

.. code-block:: python

   mol.build(driver="builtin", eri="dense+factors")

For a compact dense reference without keeping the four-index tensor:

.. code-block:: python

   mol.build(driver="builtin", eri="dense", aosym="s8")

Related Pages
-------------

* :doc:`qchem`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`
* :doc:`qchem_architecture`
