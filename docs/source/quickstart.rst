.. meta::
   :description: Install PyQED and run a native H2 restricted Hartree--Fock calculation from molecular geometry to a converged energy.

Five-minute quickstart
======================

This page runs a small, native restricted Hartree--Fock (RHF) calculation from
geometry to a converged result.  It uses PyQED's built-in integral path and does
not require PySCF.

Install
-------

Install the latest published release (currently PyQED 0.3.0):

.. code-block:: bash

   pip install pyqed

If PyQED is already installed, use ``pip install --upgrade pyqed`` to update it.
PyQED 0.3.0 provides prebuilt platform wheels for Python 3.10–3.13. Check the
compiled installation before continuing:

.. code-block:: bash

   python -m pyqed.qchem.check_install

For development checkouts and optional dependencies, see :doc:`installation`.

Run one complete script
-----------------------

Save the following as ``quickstart.py``:

.. literalinclude:: ../../examples/quickstart.py
   :language: python
   :start-at: from pyqed.qchem import Molecule

Then run:

.. code-block:: bash

   python quickstart.py

The script requires convergence and should print approximately:

.. code-block:: text

   RHF energy: -1.116759307480 Eh

``Eh`` denotes Hartree. The last digits can vary by platform; preserve the
installed version, input, and output together when recording a calculation.

What each step means
--------------------

``Molecule`` records the geometry, units, charge/spin defaults, and basis.
``mol.build(...)`` constructs native AO integrals. ``eri="auto"`` lets the
builder select an appropriate electron-repulsion representation for the
problem. ``mol.RHF().run()`` constructs and executes the closed-shell
mean-field solver.

Continue to correlated and active-space methods
-----------------------------------------------

Once RHF converges, the mean-field object can seed other solvers. For example:

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
