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
       ``tests/test_casscf.py``
   * - Fully internally contracted zero-IPEA SS-CASPT2 energies
     - Beta
     - :doc:`caspt2`
     - ``tests/test_caspt2.py``, ``benchmarks/caspt2_openmolcas.py``;
       matched LiH and H2O zero-IPEA OpenMolcas energies, including a
       915-dimensional LiH/cc-pVDZ contracted-space gate
   * - Zero-IPEA MS-CASPT2 and XMS-CASPT2 energies
     - Beta
     - :doc:`caspt2`
     - algebraic/invariance checks and matched LiH OpenMolcas effective
       Hamiltonians in ``tests/test_caspt2.py``
   * - DVR and small grid-dynamics paths
     - Beta
     - :doc:`dvr`
     - ``tests/test_fedvr.py``, ``tests/test_sddvr.py``,
       ``examples/dvr/fedvr_harmonic_oscillator.py``
   * - Basic superoperators and Lindblad utilities
     - Beta
     - :doc:`guide/guide_open_dynamics`
     - ``tests/test_superoperator_lindblad.py``
   * - Geometric, LDR/LDRFG, and nonadiabatic workflows
     - Experimental
     - :doc:`geometric_quantum_dynamics`, :doc:`pyqed.namd`
     - ``tests/test_ldrfg.py``, ``tests/test_ehrenfest.py`` and model-specific
       examples
   * - HEOM and structured-bath workflows
     - Experimental
     - :doc:`heom`
     - ``tests/test_heom_smolyak_truncation.py`` and repository examples
   * - Molecular and periodic GW/BSE
     - Experimental
     - :doc:`gw_bse`, :doc:`periodic_gw_bse`
     - ``tests/test_gw_smoke.py``, ``tests/test_pbc_gw.py`` and comparison
       scripts
   * - Periodic finite-displacement phonons
     - Experimental
     - :doc:`periodic_phonons`
     - ``tests/test_pbc_phonon.py`` and ``examples/pbc_phonon.py``
   * - Static periodic q-resolved CPHF response
     - Experimental
     - :doc:`periodic_cphf`
     - ``tests/test_pbc_cphf.py`` and ``examples/pbc_cphf.py``
   * - CPHF-relaxed Gamma-point KRHF Hessian
     - Experimental
     - :doc:`periodic_phonons`
     - ``tests/test_pbc_rhf.py`` and ``examples/pbc_gamma_hessian.py``
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
