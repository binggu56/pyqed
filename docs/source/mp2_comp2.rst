MP2 and COMP2
=============

PyQED includes canonical MP2, unrestricted MP2, and a constrained
orbital-relaxed MP2 variant called ``COMP2``. The implementations support dense
and factorized electron-repulsion integral paths when available.

Canonical MP2
-------------

For a closed-shell RHF reference, MP2 estimates the second-order correlation
energy from occupied-virtual double excitations:

.. math::

   E_\mathrm{MP2}
   =
   \sum_{ijab}
   \frac{(ia|jb)\left[2(ia|jb)-(ib|ja)\right]}
        {\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.

Basic usage:

.. code-block:: python

   from pyqed.qchem import Molecule
   from pyqed.qchem.mp.mp2 import MP2

   mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
   mol.build(eri="factors")

   mf = mol.RHF().run()
   mp = MP2(mf).run()

   print(mp.e_corr)
   print(mp.e_tot)

``MP2`` also provides relaxed-density helpers:

.. code-block:: python

   dm1 = mp.make_rdm1(ao_repr=True)
   dm2 = mp.make_rdm2(ao_repr=True)
   dm1, dm2 = mp.make_rdm12(ao_repr=True)

Factorized MP2
--------------

When the RHF object has ``eri_factors``, MP2 transforms AO pair factors into the
occupied-virtual MO block and avoids building the full dense MO ERI tensor:

.. math::

   (ia|jb) \approx \sum_L L^L_{ia} L^L_{jb}.

Recommended setup:

.. code-block:: python

   mol.build(eri="factors")
   mf = mol.RHF().run()
   mp = MP2(mf).run()
   print(mp.eri_backend)

The backend is reported as ``"factors"`` or ``"dense"`` through
``mp.eri_backend``.

UMP2
----

``UMP2`` is available for unrestricted references:

.. code-block:: python

   from pyqed.qchem.mp.mp2 import UMP2

   mf = mol.UHF().run()
   mp = UMP2(mf).run()

The unrestricted implementation tracks alpha-alpha, alpha-beta, and beta-beta
amplitudes and also uses factorized ERIs when present.

COMP2
-----

``COMP2`` is a constrained orbital-relaxed MP2 approximation. It alternates
between:

1. building MP2 amplitudes and RDMs,
2. minimizing an orbital objective with fixed RDMs,
3. semicanonicalizing the occupied and virtual subspaces,
4. rebuilding MP2 amplitudes in the updated basis.

Example:

.. code-block:: python

   from pyqed.qchem.mp.mp2 import COMP2

   comp2 = COMP2(
       mf,
       max_cycle=20,
       optimizer="RCG",
       optimizer_tol=1.0e-5,
   ).run()

   print(comp2.e_tot)
   print(comp2.converged)

``COMP2`` is not a full variational OOMP2 implementation. It is a pragmatic
macro-iterative orbital-relaxation path built from MP2 densities and PyQED's
orbital minimizer.

Semicanonicalization
--------------------

After each orbital update, COMP2 diagonalizes the Fock matrix separately in the
occupied and virtual subspaces:

.. math::

   F_{oo} U_o = U_o \epsilon_o,
   \qquad
   F_{vv} U_v = U_v \epsilon_v.

This preserves the occupied/virtual partition while producing orbital energies
that can be used in the MP2 denominator. The transform is stored in
``comp2.semicanonical_transform``.

When to Use Each Method
-----------------------

* Use canonical ``MP2`` for fast dynamic correlation from a good RHF reference.
* Use ``UMP2`` for open-shell unrestricted references.
* Use factorized MP2 when the dense MO ERI tensor is too large.
* Use ``COMP2`` when orbital relaxation is important but a full OOMP2 method is
  not required.
* Avoid MP2-like methods for strongly multireference systems; use CASCI/CASSCF
  instead.

Related Pages
-------------

* :doc:`quickstart`
* :doc:`backends`
* :doc:`qchem`
* :doc:`guide/guide_qchem_mcscf`
