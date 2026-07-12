.. _core-workflow:

.. meta::
   :description: Learn the common PyQED workflow from physical model and numerical representation to solver diagnostics, validation, and reproducible results.

How PyQED calculations work
===========================

PyQED spans several method families, but a reliable calculation usually has
the same lifecycle:

#. define the physical question and units;
#. choose a model and numerical representation;
#. build the numerical data required by that representation;
#. configure and run a solver;
#. inspect convergence and physical diagnostics; and
#. repeat at tighter numerical settings and validate against an independent
   result before drawing a scientific conclusion.

The exact object names differ between quantum chemistry, grid dynamics, open
systems, spectroscopy, and tensor networks.  This page describes the shared
workflow; each method guide documents its own inputs and limitations.

A minimal calculation
---------------------

This native H2 restricted Hartree--Fock calculation shows the common
model--build--solve pattern:

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build(driver="builtin", eri="auto")

   mf = mol.RHF().run()
   print("converged:", mf.converged)
   print("RHF energy (Hartree):", f"{mf.e_tot:.8f}")

Run it in an installed environment, or from a source checkout with
``PYTHONPATH=.``.  The expected output is approximately:

.. code-block:: text

   converged: True
   RHF energy (Hartree): -1.11675931

Small last-digit differences can occur across numerical libraries.  The
important first checks are that the solver reports convergence and that the
energy is finite.  The :doc:`complete quickstart </quickstart>` explains each
line and shows how the converged reference feeds MP2 and CASSCF calculations.

Model, build, solve, inspect
----------------------------

**Model**
   Record the scientific inputs explicitly: geometry or Hamiltonian,
   coordinate units, basis or grid, charge and spin, bath or pulse parameters,
   initial state, and boundary conditions.  Defaults are convenient for
   exploration but should not be implicit in a published calculation.

**Build**
   Convert the physical description into a numerical representation.  In the
   example, ``mol.build(...)`` constructs molecular integrals and
   ``eri="auto"`` selects a supported electron-repulsion representation.  In
   other areas this step may construct a DVR kinetic operator, a Liouvillian,
   an HEOM hierarchy, or an MPO.

**Solve**
   Create the method object, set convergence controls deliberately, and run
   it.  PyQED drivers commonly retain their outputs on the solver object after
   ``run()``; field names differ by method, so follow the method page rather
   than assuming that every solver exposes ``e_tot`` or ``converged``.

**Inspect**
   Do not treat the main scalar result as sufficient evidence.  Check the
   method-specific convergence flag or residual, normalization and conserved
   quantities, state ordering, symmetry labels, and any warning messages.

Choose a representation before a method
---------------------------------------

The representation often controls both the physical approximation and the
computational cost.

.. list-table:: Starting points by task
   :header-rows: 1
   :widths: 25 38 37

   * - Task
     - Representation choices
     - Read next
   * - Molecular electronic structure
     - Atomic-orbital basis; dense, packed, RI, or factorized integrals
     - :doc:`Quantum chemistry </qchem>` and :doc:`backends </backends>`
   * - Grid quantum dynamics
     - Coordinate domain, DVR family, grid size, and boundary conditions
     - :doc:`Discrete variable representations </dvr>`
   * - Nonadiabatic dynamics
     - Adiabatic, diabatic, locally diabatic, or geometric representation
     - :doc:`Geometric dynamics </geometric_quantum_dynamics>` and
       :doc:`NAMD API </pyqed.namd>`
   * - Open-system dynamics
     - Density matrix, Liouville space, bath decomposition, and hierarchy
     - :doc:`Open dynamics <guide_open_dynamics>` and :doc:`HEOM </heom>`
   * - Tensor-network calculations
     - Site ordering, local basis, MPO, bond dimension, and symmetry sectors
     - :doc:`Matrix product states </mps>`

Convergence and validation
--------------------------

A completed call to ``run()`` is not by itself a validated calculation.  Vary
the controls that define the numerical approximation:

* basis size, active space, integral threshold, or auxiliary basis;
* grid spacing, coordinate range, and time step;
* hierarchy depth and bath decomposition;
* bond dimension, truncation tolerance, and sweep count; or
* number of states, propagation time, and frequency resolution.

Converge the observable of interest, not only the total energy or norm.  Then
compare a small case with an analytic limit, an independently implemented
algorithm, or a documented external reference.  :doc:`Benchmarks and
validation </benchmarks>` defines the distinction between a regression test,
independent validation, and a performance result.

Know the limits of the selected path
------------------------------------

PyQED combines Beta workflows with rapidly changing research implementations.
Optional dependencies, supported state or spin sectors, storage
representations, and available diagnostics vary by method.  Before scaling a
calculation:

* check the :doc:`capability matrix </capabilities>`;
* run the smallest tracked example for the exact solver path;
* read the method page for its assumptions and unsupported combinations; and
* confirm that the chosen backend exposes the data required by downstream
  methods.

Record a reproducible result
----------------------------

Preserve enough information to reconstruct both the physical problem and the
software environment:

* PyQED release or full Git commit, including whether the tree was modified;
* Python and dependency versions, platform, and numerical libraries;
* complete input, units, random seeds, backend, and solver tolerances;
* thread settings and hardware for performance results; and
* raw output plus the convergence and validation evidence used to accept it.

For a source checkout, a controlled serial baseline can be run as:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   PYTHONPATH=. python examples/quickstart.py

Use the :doc:`benchmark manifest </benchmarks>` for quantitative validation
or performance claims, and :doc:`cite the exact software artifact and method
</citing>` in published work.

Where to go next
----------------

Return to the :doc:`user-guide topic map <guide>`, follow a structured path in
:doc:`tutorials </tutorials>`, or start from a maintained script in the
:doc:`examples index </examples>`.  If a documented example fails, reduce the
problem to that smallest case and follow the :doc:`support checklist
</support>`.
