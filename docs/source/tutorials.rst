Tutorials and learning paths
============================

PyQED is broad, so the safest route is to start with one small, inspectable
workflow and follow its linked guide, example, and test.  The paths below do
not imply that every combination of options is supported.

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

1. Read the representation overview in :doc:`dvr`.
2. Run ``examples/dvr/fedvr_harmonic_oscillator.py``.  It prints computed
   eigenvalues beside the analytic harmonic-oscillator sequence.
3. Compare grid families with ``examples/dvr/fedvr_vs_sine_quartic.py``.
4. Move to ``examples/dvr/gwp_sddvr_2d_independent_ho.py`` only after the
   one-dimensional smoke case is understood.

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

1. Read :doc:`guide/guide_open_dynamics` and :doc:`heom` for open-system
   conventions.
2. Use ``examples/heom.py`` as an entry point, then inspect focused tests for
   the exact solver path being used.
3. For spectroscopy, read :doc:`guide/guide_spectroscopy` and inspect
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
