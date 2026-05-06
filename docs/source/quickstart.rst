Quickstart
==========

This page shows a minimal path from installation to a small quantum chemistry
calculation. The examples use the native PyQED API and avoid optional PySCF
dependencies.

Install
-------

From a local checkout:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install -r docs/requirements.txt

Build a Molecule
----------------

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build(driver="builtin", eri="factors")

Run RHF
-------

.. code-block:: python

   mf = mol.RHF().run()
   print("RHF energy:", mf.e_tot)

The ``eri="factors"`` option asks the native integral path to keep a
factorized electron-repulsion representation when available. RHF automatically
uses the factorized JK path for this representation.

Run MP2
-------

.. code-block:: python

   from pyqed.qchem.mp.mp2 import MP2

   mp = MP2(mf).run()
   print("MP2 correlation energy:", mp.e_corr)
   print("MP2 total energy:", mp.e_tot)

Run CASSCF
----------

.. code-block:: python

   from pyqed.qchem import CASSCF

   mc = CASSCF(mf, ncas=2, nelecas=2).run()
   print("CASSCF energy:", mc.e_tot)

For larger active spaces, prefer factorized integrals when supported by the
solver:

.. code-block:: python

   mol.build(driver="builtin", eri="factors")
   mf = mol.RHF().run()
   mc = CASSCF(mf, ncas=4, nelecas=4).run()

Next Steps
----------

* :doc:`backends`
* :doc:`qchem`
* :doc:`examples`
* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`
