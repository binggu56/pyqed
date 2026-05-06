GW and BSE
==========

PyQED includes a native dense molecular GW/BSE path under :mod:`pyqed.gw`.
The current implementation is intended as a small-to-medium reference backend:
it is useful for validating algorithms, comparing against MOLGW/PySCF, and
building excited-state workflows.

GW Flavors
----------

The canonical entry point is ``pyqed.gw.gw.GW``.  It supports restricted
closed-shell references and the following modes:

* ``method="g0w0"``: one-shot GW from the input RHF orbitals and energies.
* ``evgw(update_screening=False)``: eigenvalue-only ``GnW0``; quasiparticle
  energies in ``G`` are updated while the screened interaction ``W0`` is kept
  fixed.
* ``evgw(update_screening=True)``: eigenvalue-only ``GnWn``; both ``G`` and
  ``W`` are updated through the quasiparticle energies.
* ``qsgw()``: dense quasiparticle self-consistent GW; PyQED builds a static
  Hermitian quasiparticle self-energy, transforms it back to AO form, solves
  the generalized AO eigenproblem, and rebuilds MO integrals each cycle.

Example
-------

.. code-block:: python

   from pyqed.gw.gw import GW
   from pyqed.qchem import Molecule
   from pyqed.qchem.hf import RHF

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       basis="sto-3g",
       unit="angstrom",
   )
   mol.build(driver="builtin", eri="dense")
   mf = RHF(mol).run()

   g0w0 = GW(mf, screening="TDH", eta=1e-3).run(method="g0w0")
   gnwn = GW(mf, screening="TDH", eta=1e-3).evgw(
       max_cycle=50,
       conv_tol=1e-8,
       damping=0.7,
   )
   qsgw = GW(mf, screening="TDH", eta=1e-2).qsgw(
       max_cycle=50,
       conv_tol=1e-8,
       damping=0.5,
   )

``qsgw`` is more expensive than ``evgw`` because it updates orbitals and
rebuilds transformed two-electron intermediates.  When the RHF reference has
AO Cholesky or native RI factors, GW transforms those factors to MO pair
factors and avoids materializing the full four-index MO/spin-orbital ERI
tensor.  The RPA matrices, self-energy coupling tensor, and static ``qsGW``
potential are still dense reference intermediates.

BSE
---

``pyqed.gw.bse.BSE`` builds Bethe-Salpeter excitation energies on top of the
native GW/RPA intermediates.  The dense BSE path has been checked against
MOLGW no-RI references for small molecules.

Validation Notes
----------------

The smoke tests compare:

* ``G0W0`` against PySCF exact-frequency GW.
* ``GnW0``, ``GnWn``, and ``qsGW`` against MOLGW no-RI references.
* dense and factorized RHF inputs for the ``qsGW`` path.

Related Examples
----------------

* ``examples/qchem/gw_qsgw.py``
* :doc:`qchem`
* :doc:`examples`
