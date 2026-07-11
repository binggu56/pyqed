Five-minute quickstart
======================

This page runs a small, native restricted Hartree--Fock (RHF) calculation from
geometry to a converged result.  It uses PyQED's built-in integral path and does
not require PySCF.

Install
-------

Install a release or development checkout as described in
:doc:`installation`.  For a source checkout, the minimal command is:

.. code-block:: bash

   python -m pip install -e .

Run one complete script
-----------------------

Save the following as ``quickstart.py``:

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       unit="angstrom",
       basis="sto-3g",
   )

   # Build one- and two-electron data before constructing a solver.
   mol.build(driver="builtin", eri="auto")

   mf = mol.RHF().run()
   print("converged:", mf.converged)
   print("RHF energy (Hartree):", mf.e_tot)

Then run:

.. code-block:: bash

   python quickstart.py

The calculation should report ``converged: True`` and a finite total energy.
The exact printed precision can vary by platform.  Do not copy an energy from
this page as a reference value; preserve the installed version, input, and
output together.

What each step means
--------------------

``Molecule`` records the geometry, units, charge/spin defaults, and basis.
``mol.build(...)`` constructs the data required by the selected backend.
``driver="builtin"`` requests the native path, while ``eri="auto"`` lets the
builder select an appropriate supported electron-repulsion representation for
the problem.  ``mol.RHF().run()`` constructs and executes the closed-shell
mean-field solver.

Continue to correlated and active-space methods
-----------------------------------------------

Once RHF converges, the mean-field object can seed other solvers.  The current
CASSCF implementation uses the optional Numba accelerator stack, so install it
before running the active-space example:

.. code-block:: bash

   python -m pip install "pyqed[accelerators]"

Then, for example:

.. code-block:: python

   from pyqed.qchem import CASSCF
   from pyqed.qchem.mp.mp2 import MP2

   mp = MP2(mf).run()
   print("MP2 correlation energy:", mp.e_corr)

   mc = CASSCF(mf, ncas=2, nelecas=2).run()
   print("CASSCF energy:", mc.e_tot)

These methods introduce additional assumptions and convergence behavior.  Read
:doc:`mp2_comp2`, :doc:`guide/guide_qchem_mcscf`, and :doc:`backends` before
changing the molecule or integral representation.

Trace the example to evidence
-----------------------------

The repository keeps implementation examples and focused tests separate:

* ``tests/test_rhf.py`` exercises RHF behavior and native representations.
* ``examples/qchem/sa_casscf_factor.py`` is a native factorized,
  state-averaged CASSCF workflow.
* ``examples/qchem/gw_qsgw.py`` extends the same molecule/RHF pattern into a
  small GW-family example.

Run tracked examples from the repository root with ``PYTHONPATH=.`` so that
relative data paths and the development checkout resolve consistently.

Next steps
----------

* Choose a learning path in :doc:`tutorials`.
* Browse runnable entry points in :doc:`examples`.
* Check method maturity in :doc:`capabilities`.
* Learn how validation evidence is reported in :doc:`benchmarks`.
