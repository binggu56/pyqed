Quantum Chemistry Architecture
==============================

This page documents the structure and conventions of the native
``pyqed.qchem`` code. It is intended for contributors adding methods or
debugging numerical differences against external packages.

Package Layout
--------------

Important modules:

* ``pyqed.qchem.mol``: molecule object, geometry, basis setup, and build
  options.
* ``pyqed.qchem.basis``: native Gaussian basis and integral construction.
* ``pyqed.qchem.hf``: RHF/UHF drivers, low-rank JK, and HF analysis helpers.
* ``pyqed.qchem.dft``: native RKS, grids, XC wrappers, gradients, and Hessians.
* ``pyqed.qchem.ci``: CISD/FCI-style CI solvers and overlaps.
* ``pyqed.qchem.mcscf``: CASCI, CASSCF, orbital optimization, and active-space
  CI helpers.
* ``pyqed.qchem.mp``: MP2, UMP2, and COMP2.
* ``pyqed.qchem.tddft``: native linear-response TDA/TDDFT.
* ``pyqed.qchem.dmrg``: quantum-chemistry DMRG and DMRG-SCF-facing code.
* ``pyqed.qchem.tools``: density/cube/viewing helpers.

Backend Boundaries
------------------

The preferred pattern is:

1. Keep the public method class small.
2. Pull integral/data access from the mean-field or molecule object.
3. Put tensor kernels in small helper functions.
4. Add tests that compare dense and factorized paths where both exist.
5. Avoid importing optional heavy backends at module import time.

Optional dependencies should be imported inside the function that needs them,
not at top level. This keeps Read the Docs and minimal installations usable.

Integral Conventions
--------------------

PyQED uses chemist's two-electron integral notation in most quantum chemistry
code:

.. math::

   (pq|rs) =
   \int\int
   \phi_p(1)\phi_q(1)
   r_{12}^{-1}
   \phi_r(2)\phi_s(2)
   \,d1\,d2.

Dense ERIs are typically stored as ``eri[p, q, r, s]`` in this convention.
Factorized ERIs use

.. math::

   (pq|rs) \approx \sum_L L^L_{pq} L^L_{rs},

with factors stored as arrays shaped like ``(naux, nao, nao)`` or transformed
MO-pair analogues.

The native molecule build options are documented in :doc:`backends`.

RDM Conventions
---------------

RDM conventions should be stated explicitly in every new method. The common
active-space convention is spin-traced spatial-orbital RDMs:

.. math::

   \gamma_{pq} =
   \langle E_{pq} \rangle,
   \qquad
   E_{pq} = \sum_\sigma a^\dagger_{p\sigma} a_{q\sigma}.

The corresponding two-particle RDM is stored with four spatial indices and is
contracted with spatial-orbital integrals in the active-space energy
expressions. Before comparing against external packages, verify index order and
spin summation. Do not rely on matching array shape alone.

Practical comparison checklist:

* Confirm whether the RDM is spin-orbital or spin-traced spatial-orbital.
* Confirm whether the two-RDM order is ``pqrs`` or a transposed convention.
* Confirm whether inactive-core contributions are included.
* Confirm whether the density is in AO, MO, active-only, or full orbital space.
* Confirm whether the package uses physicist's or chemist's ERI notation.

Adding a New Method
-------------------

Recommended workflow:

1. Start from a small dense implementation with clear tensor contractions.
2. Add tests on H2 or H4 where exact reference values are cheap.
3. Add a factorized path only after the dense path is correct.
4. Compare dense and factorized results at tight tolerances.
5. Add a PySCF comparison test when PySCF has the same method and convention.
6. Keep optional PySCF imports inside tests or optional backend functions.
7. Add a docs page or example that does not import the method during RTD build.

Testing Strategy
----------------

Useful test categories:

* shape and symmetry tests for integral and RDM tensors
* energy comparisons against exact diagonalization for tiny systems
* dense-vs-factorized equivalence tests
* scanner/rebuild tests across nearby geometries
* finite-difference checks for gradients
* regression tests for public examples

When a method supports both dense and factorized integrals, tests should verify
that the factorized path does not silently build dense MO ERIs unless the
algorithm explicitly requires them.

Read the Docs Safety
--------------------

RTD builds import Sphinx configuration and any autodoc-targeted modules. To
keep documentation builds stable:

* Do not run calculations, plotting, file loading, or data downloads at module
  import time.
* Put demos under ``if __name__ == "__main__":``.
* Import optional packages inside functions.
* Prefer static guide pages for modules that still have import-time side
  effects.
* Run ``python -m sphinx -E -W -b html docs/source /tmp/pyqed-docs`` before
  pushing docs changes.

Related Pages
-------------

* :doc:`qchem`
* :doc:`backends`
* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`
