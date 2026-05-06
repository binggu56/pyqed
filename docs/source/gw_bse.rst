GW and BSE
==========

PyQED provides a native dense molecular GW/BSE workflow in :mod:`pyqed.gw`.
The implementation is designed as a transparent reference backend for small
and medium molecules: it is useful for comparing against PySCF and MOLGW,
testing new approximations, and building neutral excited-state potential
energy surfaces.

The recommended workflow is:

.. code-block:: python

   mf = RHF(mol).run()
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

Mean-field, GW, and BSE have distinct roles:

* ``RHF`` builds the closed-shell reference, orbitals, and SCF total energy.
* ``GW`` computes quasiparticle energies and screening information.
* ``BSE`` computes neutral excitation energies from the GW/RPA reference.

Basic Example
-------------

.. code-block:: python

   from pyqed.qchem import Molecule
   from pyqed.qchem.hf import RHF
   from pyqed.gw.gw import GW
   from pyqed.gw.bse import BSE, TDA

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       basis="sto-3g",
       unit="angstrom",
   )
   mol.build(driver="builtin", eri="dense")

   mf = RHF(mol).run()
   gw = GW(mf, screening="TDH", eta=1e-3).run()

   bse = BSE(gw).run(nroots=3)
   tda = TDA(gw).run(nroots=3)

   print("SCF total energy:", mf.e_tot)
   print("GW quasiparticle energies:", gw.e_qp)
   print("BSE excitation energies:", bse.e)
   print("TDA excitation energies:", tda.e)

The :class:`~pyqed.gw.gw.GW` object stores quasiparticle energies in
``gw.e_qp``.  The older name ``gw.egw`` is kept as a compatibility alias.
For GW only, ``gw.e`` mirrors ``gw.e_qp``.  For BSE and TDA, ``bse.e`` and
``tda.e`` are excitation energies, so quasiparticle input energies live in
``bse.e_qp`` and ``tda.e_qp``.

GW Flavors
----------

The main entry point is :class:`pyqed.gw.gw.GW`.  It currently supports
restricted closed-shell references and dense/factorized molecular integrals.

Available methods include:

* ``GW(mf).run(method="g0w0")`` or ``GW(mf).g0w0()`` for one-shot GW.
* ``GW(mf).evgw(update_screening=False)`` for eigenvalue-only ``GnW0``.
* ``GW(mf).evgw(update_screening=True)`` for eigenvalue-only ``GnWn``.
* ``GW(mf).qsgw()`` for a dense quasiparticle self-consistent reference path.

``GW.run()`` returns the GW object so that downstream code can pass it directly
to BSE.  It still behaves like the quasiparticle-energy array in common NumPy
contexts:

.. code-block:: python

   gw = GW(mf).run()

   qp = gw.e_qp
   qp_array = np.asarray(gw)
   homo = gw[nocc - 1]
   qp_ev = gw * 27.211386245988

BSE and TDA
-----------

The preferred BSE API takes a completed GW object:

.. code-block:: python

   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)
   tda = TDA(gw).run(nroots=5)

``BSE`` solves the full Bethe-Salpeter eigenproblem and stores stacked
``X/Y`` amplitudes in ``bse.xy``.  The views ``bse.x`` and ``bse.y`` return
the excitation and de-excitation blocks.  ``TDA`` solves the Tamm-Dancoff
approximation and stores amplitudes in ``tda.x`` only.

For direct BSE calculations that follow the common MOLGW convention of using
HF/gKS orbital-energy differences instead of prior GW quasiparticle energies,
set ``use_qp=False``:

.. code-block:: python

   bse = BSE(gw).run(nroots=5, use_qp=False)
   tda = TDA(gw).run(nroots=5, use_qp=False)

Potential Energy Surfaces
-------------------------

For a neutral excited-state PES from BSE, use a consistent ground-state
reference at every geometry.  The practical default is:

.. math::

   E_0(R) = E_\mathrm{SCF}(R)

.. math::

   E_n(R) = E_\mathrm{SCF}(R) + \Omega_n^\mathrm{BSE}(R)

where ``mf.e_tot`` is the SCF total energy and ``bse.e[n]`` is the neutral BSE
excitation energy.

If an RPA-correlated ground-state reference is desired, use the same offset
for all excited states:

.. math::

   E_0(R) = E_\mathrm{SCF}(R) + E_c^\mathrm{RPA}(R)

.. math::

   E_n(R) = E_0(R) + \Omega_n^\mathrm{BSE}(R)

PyQED exposes this as:

.. code-block:: python

   gw = GW(mf).run()
   e0_rpa = gw.total_energy(method="rpa")
   bse = BSE(gw).run(nroots=3)
   excited_pes = e0_rpa + bse.e

The quasiparticle energies ``gw.e_qp`` should not be used directly as neutral
ground-state or excited-state PES energies; they correspond to charged
addition/removal quasiparticle levels.

Wavefunction Overlaps
---------------------

BSE and TDA objects can compute overlaps between excitation vectors at
different geometries:

.. code-block:: python

   gw1 = GW(mf1).run()
   gw2 = GW(mf2).run()

   tda1 = TDA(gw1).run(nroots=3, return_vectors=True)
   tda2 = TDA(gw2).run(nroots=3, return_vectors=True)
   overlap_tda = tda1.wavefunction_overlap(tda2)

   bse1 = BSE(gw1).run(nroots=3, return_vectors=True)
   bse2 = BSE(gw2).run(nroots=3, return_vectors=True)
   overlap_bse = bse1.wavefunction_overlap(bse2)

This is useful for following states along a PES and diagnosing state flips.

Integral Backends
-----------------

GW/BSE can use native dense integrals or factorized/RI inputs from the
mean-field reference:

.. code-block:: python

   mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
   mf = RHF(mol).run(cholesky_jk=True)
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

When available, AO Cholesky or RI factors are transformed to MO pair factors.
This avoids storing the full four-index MO tensor in the GW self-energy and
low-rank BSE/TDA paths.  The dense reference solvers are still intended for
small and medium molecules.

Validation Notes
----------------

The GW/BSE smoke tests currently cover:

* ``G0W0`` against PySCF exact-frequency GW.
* ``GnW0``, ``GnWn``, and ``qsGW`` against MOLGW reference data.
* dense and factorized integral consistency.
* dense and low-rank BSE/TDA consistency.
* same-geometry BSE/TDA overlap identities.

Related Examples
----------------

* ``examples/qchem/gw_qsgw.py``
* ``examples/qchem/gw_bse_ri.py``
* ``examples/qchem/gw/wavefunction_overlap.py``
* :doc:`qchem`
* :doc:`examples`
