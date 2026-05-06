Backends and Integral Representations
=====================================

PyQED supports multiple backend and integral-storage choices. The right choice
depends on whether a calculation needs small-molecule simplicity, dense tensor
access, or factorized contractions for larger systems.

Molecule Build Drivers
----------------------

The molecular build step selects the integral backend:

.. code-block:: python

   mol.build(driver="builtin", eri="factors")

Common driver choices:

* ``driver="builtin"`` uses PyQED's native molecular-integral path.
* ``driver="gbasis"`` uses the gbasis-based path when that dependency is
  installed.
* External packages such as PySCF are optional and are mainly useful for
  validation, comparison, or features that are not yet native.

Electron-Repulsion Storage
--------------------------

The ``eri`` keyword controls how two-electron integrals are stored:

* ``eri="dense"`` stores the dense four-index tensor.
* ``eri="factors"`` stores a Cholesky/factorized representation.
* ``eri="dense+factors"`` stores both dense and factorized representations.

The shorter ``eri`` keyword is equivalent to setting
``options={"eri_representation": ...}`` for the native build path.

When to Use Dense Integrals
---------------------------

Dense integrals are simplest and useful for:

* very small molecules
* debugging new methods
* algorithms that explicitly require ``(pq|rs)`` tensor access
* reference comparisons against dense implementations

The drawback is memory scaling. Dense four-index tensors become expensive as
the number of orbitals grows.

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

   mol.build(driver="builtin", eri="factors")

For debugging a new tensor formula:

.. code-block:: python

   mol.build(driver="builtin", eri="dense")

For comparing factorized and dense algorithms:

.. code-block:: python

   mol.build(driver="builtin", eri="dense+factors")

Related Pages
-------------

* :doc:`qchem`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`
* :doc:`qchem_architecture`
