Examples Gallery
================

The repository includes small runnable examples under ``examples/``. This page
groups the most useful entry points by topic.

Quantum Chemistry
-----------------

* ``examples/qchem/h2.py``: minimal H2 quantum chemistry calculation.
* ``examples/qchem/mol.py``: molecule construction and basic setup.
* ``examples/qchem/casscf.py``: native CASSCF workflow.
* ``examples/qchem/sa_casscf_factor.py``: state-averaged CASSCF with factorized
  integrals.
* ``examples/qchem/casscf_factor_vs_dense.py``: compare dense and factorized
  CASSCF paths.
* ``examples/qchem/comp2_h2o.py``: COMP2 example on water.
* ``examples/qchem/gw_qsgw.py``: G0W0, eigenvalue-self-consistent GW, and qsGW
  on H2.
* ``examples/qchem/rttdhf_h2_kick_spectrum.py``: real-time TDHF kick spectrum
  for H2.
* ``examples/qchem/tddmrg_h2_threeway_compare.py``: time-dependent DMRG
  comparison example.

Multiconfigurational Methods
----------------------------

* ``examples/qchem/mcscf/secondorder_casscf.py``: second-order CASSCF example.
* ``examples/qchem/mcscf/cas_pyscf.py``: CAS comparison workflow.
* ``examples/qchem/mcscf/wfn_overlap.py``: wavefunction overlap example.
* ``examples/qchem/lif_casscf_scan.py``: LiF CASSCF scan.
* ``examples/qchem/casscf_compare_vs_pyscf.py``: CASSCF comparison against
  PySCF.

DVR and Grid Dynamics
---------------------

* ``examples/dvr/sddvr.py``: simultaneous-diagonalization DVR.
* ``examples/dvr/gwp_sddvr_2d_independent_ho.py``: Gaussian-wavepacket SD-DVR
  on a two-dimensional harmonic oscillator.
* ``examples/dvr/shin_metiu.py``: Shin-Metiu DVR model.
* ``examples/qchem/gdvr_h2_rhf.py``: grid-DVR RHF example for H2.
* ``examples/qchem/gdvr_h4_rhf.py``: grid-DVR RHF example for H4.

Nonadiabatic Dynamics
---------------------

* ``examples/namd/ehrenfest.py``: Ehrenfest dynamics.
* ``examples/namd/ehrenfest_histories.py``: trajectory/history handling.
* ``examples/namd/abinitio_ehrenfest_pyscf.py``: ab initio Ehrenfest workflow
  using PySCF as a backend.
* ``examples/namd/lif_population_dynamics.py``: LiF population dynamics.

Geometric Quantum Dynamics
--------------------------

* ``examples/ldr/ldr.py``: locally diabatic representation dynamics.
* ``examples/ldr/abinitio.py``: ab initio LDR workflow.
* ``examples/ldr/abinitio_pyscf.py``: PySCF-backed ab initio LDR workflow.
* ``examples/ldr/overlap_matrix_approximation_2D.py``: approximate electronic
  overlap matrices on a two-dimensional grid.
* ``examples/ldr/h3/1scan_PES_H3+.py``: H3+ adiabatic potential scan.
* ``examples/ldr/h3/2calculate_overlap_nearest_neighbor.py``: nearest-neighbor
  electronic-state overlaps.
* ``examples/qchem/bo_hamiltonian_derivatives.py``: BO Hamiltonian derivative
  construction.
* ``examples/qchem/bo_hamiltonian_derivatives_normal_modes.py``: BO derivative
  projection onto normal modes.

Floquet and Light-Matter Models
-------------------------------

* ``examples/floquet/two_level_system.py``: driven two-level model.
* ``examples/floquet/RiceMele.py``: Rice-Mele model.
* ``examples/floquet/Floquet_topological_phase_diagram.py``: Floquet phase
  diagram example.
* ``examples/test_cavity.py``: cavity-QED smoke example.

Open Quantum Systems and Spectroscopy
-------------------------------------

* ``examples/heom.py``: hierarchical equations of motion example.
* ``examples/deom.py``: dissipaton equation of motion example.
* ``examples/redfield.py``: Redfield dynamics example.
* ``examples/signals/absorption.py``: absorption signal example.
* ``examples/2DES.py``: two-dimensional electronic spectroscopy example.
* ``examples/TPA.py``: two-photon absorption example.

Running Examples
----------------

Run examples from the repository root so relative data files resolve correctly:

.. code-block:: bash

   PYTHONPATH=. python examples/qchem/h2.py
   PYTHONPATH=. python examples/qchem/sa_casscf_factor.py
   PYTHONPATH=. python examples/dvr/sddvr.py

Some examples require optional dependencies or compiled backends. If an example
imports PySCF, PyVista, libxc, or plotting packages, install those dependencies
separately before running it.

Related Pages
-------------

* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`gw_bse`
* :doc:`tddft_ehrenfest`
* :doc:`geometric_quantum_dynamics`
* :doc:`guide/guide_qchem_mcscf`
