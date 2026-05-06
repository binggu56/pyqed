.. lime documentation master file, created by
   sphinx-quickstart on Fri May  6 14:23:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyQED's documentation!
=================================

The goal is to provide a simple-to-use package to study ``how light interacts with matter``.  

Check docs/manual.pdf for theoretical details.

Main modules
------------

* Nonlinear molecular spectroscopy
* Molecular quantum dynamics

  * Adiabatic wavepacket dynamics
  * Split-operator method
  * Discrete variable representation
  * Nonadiabatic wavepacket dynamics

* Quantum chemistry

  * TDDFT core-level excitation
  * Reduced excitation space
  * Restricted energy window with full/reduced excitation space

* Open quantum systems

  * Lindblad quantum master equation
  * Redfield theory
  * Second-order time-convolutionless master equation
  * Hierarchical equation of motion

* Semiclassical quantum trajectory methods
* Quantum transport with Landauer theory
* Solid-state tight-binding band structures
* Periodically driven matter and Floquet spectra

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   quickstart
   installation
   guide/guide
   examples
   theory
   backends
   qchem
   hf_analysis
   mp2_comp2
   gw_bse
   tddft_ehrenfest
   qchem_architecture
   mps
   dvr
   geometric_quantum_dynamics
   pyqed.floquet
   pyqed.models
   pyqed.namd
   pyqed.polariton
   nonabelian_dmrg_design
   heom
   developers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
