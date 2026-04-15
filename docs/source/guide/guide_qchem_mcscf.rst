Multiconfigurational Quantum Chemistry
=====================================

PyQED now exposes two multiconfigurational orbital optimizers:

- :class:`pyqed.qchem.CASSCF` for the lightweight first-order native path
- :class:`pyqed.qchem.COCASCI` for the original CASCI + constrained-optimization implementation

.. code-block:: python

   from pyqed.qchem import CASSCF, Molecule

   mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
   mol.build(driver='gbasis')

   mf = mol.RHF().run()
   mc = CASSCF(mf, ncas=2, nelecas=2).run()

   print(mc.e_tot)

The first-order ``CASSCF`` implementation is intentionally simple:

- it reuses the existing native CASCI solver
- it builds a generalized Fock matrix from CASCI RDMs
- it updates orbitals with a diagonal-preconditioned first-order step
- it keeps :class:`pyqed.qchem.COCASCI` available for the original workflow

You can also pass an explicit starting orbital guess when you want to relax a
perturbed or localized active-space reference:

.. code-block:: python

   mc = CASSCF(mf, ncas=2, nelecas=2).run(mo_coeff=mo_guess)

Current scope
-------------

- restricted references
- state-specific optimization
- diagonal-preconditioned orbital rotations with a short backtracking search

The first-order implementation lives in
``pyqed.qchem.mcscf.native_casscf.CASSCF`` and the original constrained
optimizer lives in ``pyqed.qchem.mcscf.casscf.COCASCI``. The orbital helper
functions live in ``pyqed.qchem.mcscf.orbopt``.
