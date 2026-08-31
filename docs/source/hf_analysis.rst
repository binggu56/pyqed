Hartree-Fock Analysis
=====================

PyQED provides analysis helpers for restricted Hartree-Fock results through
``RHFAnalysis`` and convenience methods on the RHF object. These tools are
intended for inspecting molecular orbitals, charges, bond orders, and orbital
similarity across geometries.

Interactive Molecular Fields
----------------------------

The top-level :func:`pyqed.view` function opens the PyQED molecular viewer.
The same interface handles molecular geometry, every molecular orbital,
density fields, electrostatic potential, natural transition orbitals (NTOs),
and Gaussian cube files:

.. code-block:: python

   from pyqed import view

   view(mol)                              # molecular geometry
   view(mf, orbital=["homo", "lumo"])    # selected orbitals
   view(mf, orbital="all")                # every MO in one state selector
   view(mf, density="electron")           # total electron density
   view(mf, esp=True)                     # ESP on the density surface

For an unrestricted calculation, ``density="all"`` includes the total,
alpha, beta, and spin densities. A custom AO density matrix can be supplied
directly with ``density=dm``. Excited-state calculations expose every NTO
pair through the same selector:

.. code-block:: python

   from pyqed.qchem.tddft import TDA

   td = TDA(mf).run(nstates=5)
   view(td, nto="all")

Automatic NTO extraction currently supports restricted references. Arbitrary
state-resolved AO density matrices can be named explicitly; names containing
``transition`` are rendered as transition densities and other signed matrices
as difference densities:

.. code-block:: python

   view(mf, density={
       "S1 transition density": transition_dm,
       "S1 - S0 difference density": excited_dm - ground_dm,
   })

Cube files are accepted without an optional chemistry viewer dependency:

.. code-block:: python

   view("density.cube")
   view("orbitals.cube", dataset="all")  # every dataset in a multi-orbital cube

The geometry-only view uses a compact link. Volumetric arrays never enter the
page URL: PyQED writes a local launcher and transfers validated float32 field
data directly to the viewer. The website renders it locally in the browser
and offers a field/state menu plus adjustable isovalues.

Basic Usage
-----------

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="O 0 0 0; H 0 0 0.96; H 0.92 0 -0.24",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build(eri="factors")

   mf = mol.RHF().run()
   analysis = mf.analyze()

   analysis.print_mo_composition(mo_indices=[0, 1, 2])
   analysis.print_mulliken_charges()
   analysis.print_lowdin_charges()
   analysis.print_mayer_bond_orders(min_bond_order=0.05)
   analysis.print_wiberg_bond_orders(min_bond_order=0.05)

MO Components
-------------

``mo_components()`` reports atomic-orbital contributions to selected molecular
orbitals:

.. code-block:: python

   components = analysis.mo_components(
       mo_indices=[0, 1, 2],
       metric="mulliken",
       min_contribution=0.01,
   )

Supported metrics:

* ``metric="mulliken"`` or ``"population"`` uses ``C_ao,mo * (S C)_ao,mo``.
* ``metric="coeff"`` or ``"coeff2"`` uses squared MO coefficients.

The Mulliken metric is overlap-aware and is usually the better default for
nonorthogonal atomic-orbital bases. The coefficient metric is useful for quick
debugging but is not a population analysis in a nonorthogonal AO basis.

MO Composition
--------------

``mo_composition()`` groups MO contributions by atom, shell, or atom+shell:

.. code-block:: python

   analysis.print_mo_composition(
       mo_indices=[4, 5],
       metric="mulliken",
       group_by="atom+shell",
       min_contribution=0.02,
   )

Supported grouping choices:

* ``group_by="atom"``
* ``group_by="shell"``
* ``group_by="atom+shell"``

This is the preferred user-facing orbital-composition tool. It summarizes the
AO-level data from ``mo_components()`` into chemically readable groups.

Mulliken and Lowdin Charges
---------------------------

Mulliken populations are computed from the AO density matrix ``D`` and overlap
matrix ``S``:

.. math::

   P_A = \sum_{\mu \in A} (D S)_{\mu\mu},
   \qquad
   q_A = Z_A - P_A.

Lowdin populations first symmetrically orthogonalize the AO basis:

.. math::

   P_A^\mathrm{Lowdin}
   = \sum_{\mu \in A} (S^{1/2} D S^{1/2})_{\mu\mu}.

Use Mulliken charges for continuity with many semiempirical and qualitative
analyses. Use Lowdin charges when you want a more symmetric orthogonalized AO
population.

Bond Orders
-----------

``mayer_bond_orders()`` computes Mayer bond orders from the density-overlap
matrix ``P = D S``:

.. math::

   B_{AB}^\mathrm{Mayer}
   = \sum_{\mu \in A}\sum_{\nu \in B} P_{\mu\nu} P_{\nu\mu}.

``wiberg_bond_orders()`` computes a Wiberg-style index in the Lowdin
orthogonalized AO basis:

.. math::

   B_{AB}^\mathrm{Wiberg}
   = \sum_{\mu \in A}\sum_{\nu \in B}
     \left(D^\mathrm{orth}_{\mu\nu}\right)^2.

Mayer bond orders are usually more directly tied to the AO overlap metric.
Wiberg bond orders are often easier to interpret in an orthogonalized basis.
Both are qualitative bonding indicators, not observables.

MO Overlap Between Geometries
-----------------------------

``mo_overlap()`` compares MO subspaces from two RHF calculations:

.. code-block:: python

   mf1 = mol1.RHF().run()
   mf2 = mol2.RHF().run()

   s_mo = mf1.analyze().mo_overlap(
       mf2,
       mo_indices=[0, 1, 2],
       other_mo_indices=[0, 1, 2],
   )

For different geometries, PyQED builds a cross-AO overlap matrix and evaluates

.. math::

   S^\mathrm{MO}_{ij} = C_i^\dagger S^{12}_\mathrm{AO} C'_j.

This is useful for tracking orbital character along scans and for diagnosing
active-space continuity in CASSCF workflows.

Related Pages
-------------

* :doc:`qchem`
* :doc:`backends`
* :doc:`guide/guide_qchem_mcscf`
