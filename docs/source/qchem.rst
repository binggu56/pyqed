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
   mol.build(eri="auto")

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
* ``pyqed.qchem.semiempirical`` contains semiempirical method interfaces,
  including the in-progress OM2/MRCI API.
* ``pyqed.qchem.dmrg`` contains DMRG and spin-adapted/non-Abelian development
  paths.
* ``pyqed.gw`` contains dense molecular GW, eigenvalue-self-consistent GW,
  qsGW, and BSE reference implementations.
* ``pyqed.pbc.gw`` contains dense small-cell periodic G0W0, evGW/GnW0, TDA,
  and full BSE development paths.

Integral Backends
-----------------

The molecular build step can use different integral representations depending
on the calculation:

* ``eri="auto"`` uses compact eight-fold exact storage for small systems and
  prefers native RI factors for larger systems when an auxiliary basis is
  available.
* ``eri="dense", aosym="s1"`` stores the four-index electron repulsion tensor
  explicitly.
* ``eri="dense", aosym="s4"`` stores the tensor in unique AO-pair form.
* ``eri="dense", aosym="s8"`` stores only unique AO-pair-pair values,
  exploiting the full eight-fold ERI permutation symmetry for memory.
* ``eri="direct"`` avoids dense AO ERI construction and uses compact ``s8``
  storage for cartesian J/K builds.
* ``eri="factors"`` stores a Cholesky/factorized representation when available.
* ``eri="dense+factors"`` keeps both representations for algorithms that need
  dense tensors and factorized contractions.
* ``eri="ri"`` builds native density-fitting factors from bundled auxiliary
  basis sets, without using PySCF. The default ``ri_purpose="jk"`` prefers
  JKFIT sets for SCF when available; use ``options={"ri_purpose": "ri"}`` to
  prefer RIFIT sets for correlation-style fitting.
* ``eri="dense+ri"`` keeps the dense tensor and the native RI factors.

Auxiliary bases can be selected explicitly:

.. code-block:: python

   mol.build(eri="ri",
       auxbasis="cc-pvdz-rifit",
   )

The Cholesky and RI paths are useful for larger active-space and CASSCF
workflows because they avoid materializing dense transformed electron-repulsion
tensors when the solver can contract directly with factors. Cholesky factors
are an exact low-rank decomposition of the AO ERI tensor up to the selected
tolerance; RI factors are an auxiliary-basis approximation. Native RI builds
three-center tensors in packed AO-pair form, uses full factor storage by default
for the current faster RHF contraction path, supports ``ri_storage="packed"``
for memory-sensitive runs, applies a Cholesky-first metric solve with an
eigenvalue fallback, and supports optional three-center screening via
``ri_screen_tol``.

Multiconfigurational Methods
----------------------------

PyQED includes native CASCI/CASSCF implementations, including factorized
integral support and second-order orbital optimization paths. See the
dedicated guide for examples:

.. toctree::
   :maxdepth: 1

   guide/guide_qchem_mcscf
   guide/guide_qchem_om2_mrci

SU(2)-NARG State Overlaps
-------------------------

Two completed direct-reduced SU(2)-NARG calculations can be connected without
recovering determinant amplitudes:

.. code-block:: python

   overlap = narg_bra.overlap(narg_ket)

   overlap_exact, info = narg_bra.overlap(
       narg_ket,
       cutoff=0.0,
       max_bond=None,
       return_info=True,
   )

When the calculations carry molecular orbitals, the method builds the
cross-geometry AO overlap automatically. ``ao_overlap`` or a full core plus
active ``mo_overlap`` can instead be supplied explicitly. Frozen cores are
eliminated with their exact Schur complement and determinant prefactor. The
conditional NARG tensors are converted to a fully reduced SU(2) MPS, including
the NARG local-state phase convention, and the nonorthogonal orbital map is
applied as a sector-preserving Gaussian circuit.

Here ``circuit`` means a tensor-network factorization, not quantum hardware or
physical time evolution. A one-particle map :math:`G` induces the Fock-space
operator

.. math::

   \widehat G\,a_p^\dagger\widehat G^{-1}
   = \sum_q G_{qp}a_q^\dagger.

Constructing :math:`\widehat G` as a dense operator would require the full
Fock space. Instead, an SVD and adjacent Givens factorizations express
:math:`G` as diagonal scalings and two-orbital maps. Their second-quantized
actions are applied directly to neighboring reduced-MPS tensors, followed by
an SU(2)-resolved SVD. This is why the implementation has a circuit even though
the overlap itself is a static scalar.

When :math:`G^\dagger G=I` within numerical tolerance, the implementation
recognizes a true orbital rotation and factors :math:`G` directly with one
adjacent-Givens sweep. A generic unitary map therefore needs at most
:math:`L(L-1)/2` two-orbital gates. Only a genuinely nonunitary map uses the
two-sweep :math:`U\Sigma V^\dagger` construction. The selected route and the
unitarity residual are available as ``orbital_factorization`` and
``unitarity_residual`` in the overlap diagnostics.

Selected roots are carried together as an open terminal boundary. Consequently
the orbital circuit is applied at most once to each root bundle, and a single
reduced environment contraction produces the full root-overlap matrix. The
expensive orbital transformation therefore does not repeat for every root or
root pair. ``return_info=True`` reports
``batched_roots``, the two root-batch sizes, the orbital-transform call count,
and the overlap-contraction count.

The default ``orbital_split="auto"`` estimates the circuit cost on each side
and chooses among balanced, bra-only, and ket-only factorizations of the exact
active-space relation

.. math::

   G_L^\dagger G_R = S_{\mathrm{eff}}.

For a geometry sequence, aligning the next orbital gauge before solving keeps
this map local:

.. code-block:: python

   next_narg = NARG.from_parallel_transport(
       previous_narg,
       next_mf,
       transport_method="polar",
       ncas=ncas,
       nelecas=nelecas,
       D=128,
   ).run()

``transport_method="match"`` restricts the gauge to permutation and phase
changes when preserving localized orbitals matters more than the optimal polar
alignment. Core and active spaces are aligned separately and never mixed.
``overlap_orbital_order`` can also suggest a common chain order by minimizing
the cumulative overlap-graph boundary cost.

An exact block-diagonal map is automatically factored into independent
contiguous Gaussian circuits. For a nearly local map,
``orbital_map_threshold=tau`` additionally drops off-diagonal edges with
magnitude at most :math:`\tau`. This is an explicit approximation: diagnostics
report the block count and

.. math::

   \epsilon_{\mathrm{map}}
   = \left\|G_L^\dagger G_R-S_{\mathrm{eff}}\right\|_2,

and ``exact`` is false whenever this residual is nonzero. The threshold is zero
by default. If the resulting contiguous block sizes are :math:`b_\alpha`, the
number of generic adjacent gates is reduced from
:math:`L(L-1)` to

.. math::

   N_{\mathrm{gate}}
   = \sum_\alpha b_\alpha(b_\alpha-1).

Thus the circuit frontier is controlled by the largest connected orbital block
rather than by the full active-orbital count :math:`L`.

The defaults ``cutoff=1e-10`` and ``max_bond="auto"`` compress intermediate
bonds and are approximate. Setting ``cutoff=0`` and ``max_bond=None`` removes
that compression, although an exact general orbital map can still produce
exponential intermediate bond growth. This implementation is an adaptation of
the biorthogonal transformations of P.-A. Malmqvist, *Int. J. Quantum Chem.*
**30**, 479 (1986), DOI ``10.1002/qua.560300404``, and the nonorthogonal MPS
state-interaction construction of S. Knecht et al., *J. Chem. Theory Comput.*
**12**, 5881 (2016), DOI ``10.1021/acs.jctc.6b00889``. It is not a direct
reproduction of either reference implementation.

Related Topics
--------------

* :doc:`dvr`
* :doc:`backends`
* :doc:`examples`
* :doc:`hf_analysis`
* :doc:`mp2_comp2`
* :doc:`gw_bse`
* :doc:`periodic_gw_bse`
* :doc:`tddft_ehrenfest`
* :doc:`qchem_architecture`
* :doc:`mps`
* :doc:`nonabelian_dmrg_design`
