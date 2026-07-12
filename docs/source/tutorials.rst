Tutorials and learning paths
============================

PyQED is broad, so the safest route is to start with one small, inspectable
workflow and follow its linked guide, example, and test.  The paths below do
not imply that every combination of options is supported.

Before choosing a method, read :doc:`guide/core_workflow` for the shared
model--build--solve--inspect--validate lifecycle.  Each path below begins with
a small calculation and then points toward the more specialized material.

Quantum chemistry
-----------------

1. Run the native RHF :doc:`quickstart`.
2. Read :doc:`qchem` for molecule construction and solver families.
3. Read :doc:`backends` before selecting dense, packed, RI, or factorized
   electron-repulsion data.
4. Continue to :doc:`mp2_comp2` or :doc:`guide/guide_qchem_mcscf`.
5. Inspect ``examples/qchem/sa_casscf_factor.py`` and its corresponding
   focused tests before scaling the problem.

GW and response calculations are separate advanced paths; begin with
:doc:`gw_bse` or :doc:`tddft_ehrenfest` and use their documented conventions.

Grid and wavepacket dynamics
----------------------------

1. Read :doc:`dvr`, including its boundary-condition and convergence advice.
2. Run the maintained Sine-DVR smoke case:

   .. code-block:: bash

      PYTHONPATH=. python examples/dvr/sine_harmonic_oscillator.py

   Its first four energies should be ``[0.5 1.5 2.5 3.5]``.
3. Vary both the box and grid size until the states of interest are stable.
4. Compare representations with ``examples/dvr/fedvr_vs_sine_quartic.py``
   only after the one-dimensional Sine-DVR calculation is understood.
5. Treat multidimensional and Shin--Metiu scripts as research workflows: read
   the code and verify their model assumptions before execution.

Nonadiabatic and geometric dynamics
-----------------------------------

1. Read :doc:`geometric_quantum_dynamics` and :doc:`pyqed.namd`.
2. Start with ``examples/namd/ehrenfest.py`` or
   ``examples/namd/ldrfg_avoided_crossing.py``.
3. Treat ab initio and sparse-grid scripts as research workflows: inspect their
   optional dependencies, cached inputs, grid convergence, and output paths
   before execution.

Open systems and spectroscopy
-----------------------------

1. Read :doc:`guide/guide_open_dynamics` to choose between time-local master
   equations and hierarchy-based dynamics.
2. Run the compact, documented :doc:`HEOM <heom>` calculation:

   .. code-block:: bash

      PYTHONPATH=. python examples/heom_compact.py

   It should finish with ``Final <sigma_z>: -0.96907844``.  Then vary the time
   step and hierarchy depth before changing the physical model.
3. Use ``examples/heom.py`` only when you need the longer exploratory script;
   it contains historical comparison material and plotting imports.
4. For spectroscopy, read :doc:`guide/guide_spectroscopy` and inspect
   ``examples/signals/absorption.py`` before the larger ``examples/2DES.py``
   workflow.

Floquet and light--matter models
--------------------------------

Read :doc:`pyqed.floquet` and :doc:`pyqed.polariton`, then inspect
``examples/floquet/two_level_system.py`` and
``examples/floquet/RiceMele.py``.  These are model-specific entry points;
verify units and gauge conventions before adapting them.

How to turn an example into research evidence
---------------------------------------------

For every path:

* pin the PyQED release or Git commit;
* keep the exact input and random seed;
* record units, basis/grid, solver tolerances, dependencies, and threads;
* establish convergence with respect to the relevant numerical controls; and
* use the :doc:`benchmarks` manifest for results presented as validation or
  performance evidence.
