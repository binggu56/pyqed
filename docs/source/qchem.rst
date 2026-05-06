Quantum Chemistry
=================

The :mod:`pyqed.qchem` package provides native quantum chemistry tools for
molecular integrals, Hartree-Fock references, post-HF correlation methods,
multiconfigurational methods, and excited-state calculations.

This page is a static overview. It avoids importing :mod:`pyqed.qchem` during
the documentation build because some optional compiled backends, such as
``libxc`` and integral libraries, may not be available on Read the Docs.

Core Workflow
-------------

A typical native calculation follows this structure:

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build(driver="builtin", eri="ri")

   mf = mol.RHF().run()
   print(mf.e_tot)

Main Components
---------------

* ``pyqed.qchem.mol`` defines the molecular object, basis setup, and integral
  construction paths.
* ``pyqed.qchem.hf`` contains restricted and unrestricted Hartree-Fock drivers.
* ``pyqed.qchem.hf.analysis`` contains orbital, charge, and bond-order analysis
  helpers.
* ``pyqed.qchem.ci`` contains CI methods, including CISD and FCI utilities.
* ``pyqed.qchem.mcscf`` contains CASCI, CASSCF, state-averaged CASSCF, and
  second-order orbital optimization paths.
* ``pyqed.qchem.mp`` contains MP2 and orbital-optimized MP2 utilities.
* ``pyqed.qchem.tddft`` contains linear-response TDDFT functionality.
* ``pyqed.qchem.dft`` contains native DFT and grid functionality.
* ``pyqed.qchem.dmrg`` contains DMRG and spin-adapted/non-Abelian development
  paths.
* ``pyqed.gw`` contains dense GW, eigenvalue-self-consistent GW, qsGW, and BSE
  reference implementations.

Integral Backends
-----------------

The molecular build step can use different integral representations depending
on the calculation:

* ``driver="builtin"`` uses the native integral path.
* ``eri="dense"`` stores the four-index electron repulsion tensor explicitly.
* ``eri="factors"`` stores a Cholesky/factorized representation when available.
* ``eri="dense+factors"`` keeps both representations for algorithms that need
  dense tensors and factorized contractions.
* ``eri="ri"`` builds native density-fitting factors from bundled auxiliary
  basis sets, without using PySCF. For example, ``cc-pVDZ`` automatically uses
  ``cc-pVDZ-RIFIT`` when available.
* ``eri="dense+ri"`` keeps the dense tensor and the native RI factors.

Auxiliary bases can be selected explicitly:

.. code-block:: python

   mol.build(
       driver="builtin",
       eri="ri",
       auxbasis="cc-pvdz-rifit",
   )

The Cholesky and RI paths are useful for larger active-space and CASSCF
workflows because they avoid materializing dense transformed electron-repulsion
tensors when the solver can contract directly with factors. Cholesky factors
are an exact low-rank decomposition of the AO ERI tensor up to the selected
tolerance; RI factors are an auxiliary-basis approximation.

Multiconfigurational Methods
----------------------------

PyQED includes native CASCI/CASSCF implementations, including factorized
integral support and second-order orbital optimization paths. See the
dedicated guide for examples:

.. toctree::
   :maxdepth: 1

   guide/guide_qchem_mcscf

Related Topics
--------------

* :doc:`dvr`
* :doc:`backends`
* :doc:`examples`
* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`gw_bse`
* :doc:`tddft_ehrenfest`
* :doc:`qchem_architecture`
* :doc:`mps`
* :doc:`nonabelian_dmrg_design`
