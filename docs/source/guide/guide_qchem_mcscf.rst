CASCI and CASSCF
================

PyQED provides native active-space methods for multireference quantum
chemistry. CASCI optimizes CI coefficients in a fixed active orbital space.
CASSCF additionally optimizes the orbitals.

Basic CASSCF
------------

.. code-block:: python

   from pyqed.qchem import CASSCF, Molecule

   mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
   mol.build(driver="builtin", eri="factors")

   mf = mol.RHF().run()
   mc = CASSCF(mf, ncas=2, nelecas=2).run()

   print(mc.e_tot)

``ncas`` is the number of active spatial orbitals. ``nelecas`` is the number
of active electrons, either as an integer for closed-shell active spaces or as
``(nalpha, nbeta)`` for explicit spin sectors.

Active Space Selection
----------------------

By default, PyQED places the active block after the inactive core orbitals. You
can provide explicit zero-based MO indices:

.. code-block:: python

   mc = CASSCF(mf, ncas=4, nelecas=4).run(
       active_orbitals=[2, 3, 4, 5],
   )

Use active orbital selection when:

* the chemically important orbitals are not contiguous,
* the active space changes character along a scan,
* you need to match a PySCF or external active-space reference,
* you want to preserve a manually localized or reordered active block.

The helper checks that the number of active orbital indices equals ``ncas`` and
that there are no duplicates.

First-Order CASSCF
------------------

``CASSCF`` is the lightweight native orbital optimizer. It:

* solves the active-space CI problem with the native CASCI solver,
* builds spin-traced active-space RDMs,
* forms a generalized Fock matrix,
* computes the nonredundant orbital gradient,
* updates orbitals with diagonal preconditioning and line search.

This path is useful for small and medium active spaces, quick scans, and
testing dense/factorized integral code.

Second-Order CASSCF
-------------------

``SecondOrderCASSCF`` adds a stronger orbital-optimization path with
microiterations:

.. code-block:: python

   from pyqed.qchem import SecondOrderCASSCF

   mc = SecondOrderCASSCF(
       mf,
       ncas=4,
       nelecas=4,
       coupling="full",
       max_micro_cycle=8,
   ).run()

The second-order implementation supports several coupling modes:

* ``coupling="full"`` uses the production full coupling path by default.
* ``coupling="qn"`` uses a quasi-Newton-like orbital path.
* ``coupling="simultaneous"`` performs joint CI-orbital microiterations.
* ``coupling="simultaneous_reduced"`` or ``"simultaneous_partial"`` uses a
  reduced simultaneous coupling.

For routine work, start with ``coupling="full"``. Use simultaneous coupling for
experiments where CI and orbital variables must relax together in the
microiteration.

State Averaging
---------------

State-averaged CASSCF optimizes orbitals for a weighted average of multiple
CASCI roots:

.. code-block:: python

   weights = [0.5, 0.5]
   mc = CASSCF(mf, ncas=4, nelecas=4)
   mc.state_average(weights).run(nstates=2)

State averaging is useful near avoided crossings or when multiple electronic
states must share a consistent orbital basis.

Factorized Integrals
--------------------

CASSCF can use factorized AO ERIs from the RHF reference:

.. code-block:: python

   mol.build(driver="builtin", eri="factors")
   mf = mol.RHF().run()
   mc = SecondOrderCASSCF(mf, ncas=4, nelecas=4).run()

In factorized mode, the CASSCF code avoids constructing dense transformed MO
ERI tensors when the active-space contraction can be performed directly with
pair factors:

.. math::

   (pq|rs) \approx \sum_L L^L_{pq} L^L_{rs}.

This is the recommended path for larger basis sets and larger active spaces.

Convergence Controls
--------------------

Important options:

* ``max_cycle`` controls macroiterations.
* ``max_micro_cycle`` controls second-order microiterations.
* ``conv_tol`` controls energy convergence.
* ``conv_tol_grad`` controls strict orbital-gradient convergence.
* ``conv_tol_grad_relaxed`` allows convergence when the energy is stable and
  the gradient is small enough for practical scans.
* ``conv_tol_step`` controls orbital-step convergence.

For production calculations, prefer a slightly larger ``max_cycle`` and monitor
both energy and gradient norms. If the active space changes character along a
scan, use explicit ``active_orbitals`` or orbital-overlap analysis.

QN vs Simultaneous Microiterations
----------------------------------

The quasi-Newton-style path treats the CI response and orbital response in a
more decoupled way. It is usually faster and often robust enough.

The simultaneous path updates CI and orbital variables together inside the
microiteration. It is closer to a fully coupled second-order formulation, but
it is more expensive and more sensitive to trust-region/acceptance settings.

Use this practical rule:

* use ``full`` or ``qn`` for normal calculations,
* use ``simultaneous`` when testing paper-faithful coupled CI-orbital behavior,
* fall back to ``full`` if simultaneous microiterations become too slow or
  reject too many steps.

Examples
--------

Relevant examples:

* ``examples/qchem/casscf.py``
* ``examples/qchem/sa_casscf_factor.py``
* ``examples/qchem/casscf_factor_vs_dense.py``
* ``examples/qchem/mcscf/secondorder_casscf.py``
* ``examples/qchem/benchmark_second_order_casscf.py``
* ``examples/qchem/lif_casscf_scan.py``

Related Pages
-------------

* :doc:`../qchem`
* :doc:`../backends`
* :doc:`../hf_analysis`
* :doc:`../qchem_architecture`
