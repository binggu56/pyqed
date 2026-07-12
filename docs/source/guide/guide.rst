.. _guide:

.. meta::
   :description: Task-oriented PyQED user guide for quantum chemistry, quantum dynamics, open systems, spectroscopy, and tensor networks.

PyQED user guide
================

The user guide organizes PyQED by scientific task.  It reuses the canonical
method, example, and API pages rather than repeating their content.  Choose a
topic below, run its smallest example first, and check the stated maturity and
limitations before adapting it to a research calculation.

.. important::

   PyQED is active research software.  A module being importable does not mean
   every option is supported or validated.  Consult the :doc:`capability
   matrix </capabilities>` and the limitations on the relevant method page.

New to PyQED?
-------------

#. :doc:`Install PyQED </installation>` in an isolated environment.
#. Run the :doc:`five-minute H2 quickstart </quickstart>`.
#. Read :doc:`How PyQED calculations work <core_workflow>` for the common
   model--build--solve--validate workflow.
#. Pick a task-oriented path in :doc:`Tutorials and learning paths
   </tutorials>` or browse the :doc:`runnable examples index </examples>`.

Foundations and common workflow
-------------------------------

* :doc:`How PyQED calculations work <core_workflow>` explains how inputs,
  numerical representations, solver objects, diagnostics, and validation fit
  together.
* :doc:`Theory overview </theory>` introduces the Hamiltonian, wavefunction,
  density-matrix, response, and representation conventions shared across
  method areas.
* :doc:`Backends and integral representations </backends>` describes native,
  dense, packed, RI, and factorized electronic-structure paths.
* :doc:`API entry points </api>` maps supported workflows to their modules;
  it is a navigation aid rather than a blanket stability promise.

Electronic structure
--------------------

Start with the native RHF workflow, then add correlation or response only
after the reference calculation is converged.

* :doc:`Quantum chemistry overview </qchem>` -- molecule construction,
  solver families, and integral choices.
* :doc:`Hartree--Fock analysis </hf_analysis>` -- orbitals, populations, and
  diagnostics after SCF.
* :doc:`MP2 and COMP2 </mp2_comp2>` -- perturbative correlation workflows.
* :doc:`CASCI and CASSCF <guide_qchem_mcscf>` -- active spaces, orbital
  optimization, state averaging, and convergence controls.
* :doc:`OM2/MRCI <guide_qchem_om2_mrci>` -- current semiempirical excited-state
  interface and its validation limits.
* :doc:`GW and BSE </gw_bse>` and :doc:`TDDFT/Ehrenfest
  </tddft_ehrenfest>` -- advanced response and excited-state paths.

Grid, quantum, and nonadiabatic dynamics
----------------------------------------

* :doc:`Discrete variable representations </dvr>` -- grid construction,
  kinetic energy, Hamiltonian assembly, and diagonalization.
* :doc:`Geometric quantum dynamics </geometric_quantum_dynamics>` --
  geometric and locally diabatic representations.
* :doc:`Nonadiabatic dynamics API </pyqed.namd>` -- available NAMD objects and
  implementation entry points.
* :doc:`TDDFT and Ehrenfest dynamics </tddft_ehrenfest>` -- coupled
  electronic--nuclear workflows and current backend restrictions.

Open systems and spectroscopy
-----------------------------

* :doc:`Open quantum dynamics <guide_open_dynamics>` -- Lindblad, Redfield,
  time-convolutionless, and hierarchy-based concepts.
* :doc:`HEOM and structured baths </heom>` -- solver imports, optional
  dependencies, hierarchy controls, and reproducibility requirements.
* :doc:`Nonlinear molecular spectroscopy <guide_spectroscopy>` --
  sum-over-states, correlation-function, and nonperturbative viewpoints.

Light--matter and periodically driven systems
---------------------------------------------

* :doc:`Floquet methods </pyqed.floquet>` -- periodically driven model
  workflows.
* :doc:`Polariton methods </pyqed.polariton>` -- coupled light--matter model
  entry points.
* :doc:`Model Hamiltonians </pyqed.models>` -- reusable model-building
  components.

Tensor networks and many-body methods
-------------------------------------

* :doc:`Matrix product states </mps>` -- MPS/MPO concepts, DMRG, package map,
  and example entry points.
* :doc:`Non-Abelian DMRG design </nonabelian_dmrg_design>` -- reduced-sector
  conventions and the status of spin-adapted research paths.

Reliability, reproduction, and help
-----------------------------------

* :doc:`Capability maturity </capabilities>` states which workflows are Beta
  or Experimental and points to their evidence.
* :doc:`Benchmarks and validation </benchmarks>` separates regression tests,
  independent validation, and performance claims.
* :doc:`Citing PyQED </citing>` explains software, method, version, and input
  citation.
* :doc:`Support and problem reports </support>` lists the information needed
  for a useful scientific bug report.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Foundations

   core_workflow

   ../theory
   ../backends

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Electronic structure

   ../qchem
   ../hf_analysis
   ../mp2_comp2
   ../gw_bse
   ../tddft_ehrenfest
   ../qchem_architecture

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Quantum dynamics

   ../dvr
   ../geometric_quantum_dynamics
   ../pyqed.namd

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Open systems and spectra

   guide_spectroscopy
   guide_open_dynamics
   ../heom

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Light–matter systems

   ../pyqed.floquet
   ../pyqed.polariton
   ../pyqed.models

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tensor networks

   ../mps
   ../nonabelian_dmrg_design
