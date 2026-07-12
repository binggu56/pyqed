Capability maturity
===================

Maturity labels communicate the software evidence available on the current
development branch.  They do not certify physical accuracy outside the tested
domain.

Labels
------

Stable
   A versioned public workflow has documented compatibility expectations,
   clean-install coverage, representative regression tests, and release-level
   validation.  No workflow is declared Stable merely because it has existed
   for a long time.

Beta
   The documented path has focused tests and runnable examples, but interfaces,
   defaults, platform coverage, or validation breadth may still change.

Experimental
   A research or prototype path with narrower evidence, active interface
   changes, substantial optional dependencies, or incomplete end-to-end
   release coverage.

Current development-branch matrix
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 13 34 31

   * - Area
     - Maturity
     - Entry documentation
     - Evidence path
   * - Native molecular RHF and integral representations
     - Beta
     - :doc:`quickstart`, :doc:`backends`, :doc:`qchem`
     - ``tests/test_rhf.py``, ``tests/test_native_integrals.py``
   * - MP2 and selected active-space workflows
     - Beta
     - :doc:`mp2_comp2`, :doc:`guide/guide_qchem_mcscf`
     - ``tests/test_mp2.py``, ``tests/test_casci.py``,
       ``tests/test_casscf.py``; CASSCF currently requires the
       ``accelerators`` extra
   * - DVR and small grid-dynamics paths
     - Beta
     - :doc:`dvr`
     - ``tests/test_fedvr.py``, ``tests/test_sddvr.py``,
       ``examples/dvr/sine_harmonic_oscillator.py``
   * - Basic superoperators and Lindblad utilities
     - Experimental
     - :doc:`guide/guide_open_dynamics`
     - ``examples/liouville_space.py``, ``examples/redfield.py``
   * - Geometric, LDR/LDRFG, and nonadiabatic workflows
     - Experimental
     - :doc:`geometric_quantum_dynamics`, :doc:`pyqed.namd`
     - ``tests/test_ldrfg.py``, ``tests/test_ehrenfest.py`` and model-specific
       examples
   * - HEOM and structured-bath workflows
     - Experimental
     - :doc:`heom`
     - ``examples/heom_compact.py``, ``examples/heom.py``
   * - Molecular GW/BSE
     - Experimental
     - :doc:`gw_bse`
     - ``tests/test_gw_smoke.py`` and comparison scripts
   * - Real-time TDHF/TDDFT and coupled dynamics
     - Experimental
     - :doc:`tddft_ehrenfest`
     - ``tests/test_rttdhf.py``, ``tests/test_rttddft.py``
   * - MPS, DMRG, NARG, LETTA, and non-Abelian paths
     - Experimental
     - :doc:`mps`, :doc:`nonabelian_dmrg_design`
     - Method-specific tests under ``tests/``; support varies by solver
   * - Spectroscopy, Floquet, polariton, UED, MD, and ML research workflows
     - Experimental
     - :doc:`guide/guide_spectroscopy`, :doc:`pyqed.floquet`,
       :doc:`pyqed.polariton`, :doc:`examples`
     - Model-specific examples and focused tests where listed

Promotion criteria
------------------

Experimental to Beta requires a documented public path, deterministic smoke
coverage, stated units and limitations, and validation against an analytic or
independent reference for a representative case.  Beta to Stable additionally
requires versioned API expectations, clean-install and supported-platform
coverage, release artifacts, upgrade guidance, and sustained maintenance.

Report a stale label through the issue tracker with links to the relevant
documentation, tests, and validation evidence.
