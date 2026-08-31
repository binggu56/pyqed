CASPT2
======

Status
------

The restricted-real, zero-IPEA, fully internally contracted SS, MS, and XMS
energy drivers are **Beta**. They have analytic-limit and invariance tests plus
matched OpenMolcas validation. Broader molecular, active-space, intruder-state,
and platform coverage is still needed before any Stable claim.

The implementation currently supports restricted real-orbital CASCI and
CASSCF references. It does not yet provide IPEA shifts, analytic gradients,
RAS references, or shifted MS/XMS effective-Hamiltonian couplings. The current
reference algorithm explicitly enumerates the external determinant space, so
``max_external_determinants`` and ``max_ic_operators`` should be used as safety
limits for larger calculations. ``frozen_core`` excludes the requested number
of lowest doubly occupied spatial orbitals from that external space while
retaining their mean-field contribution. The complete-CAS external-space size
and all eight class counts are evaluated combinatorially before integral
transformation. They are exposed as ``estimated_external_determinants`` and
``estimated_external_class_counts`` in ``work_estimate``. A violated
``max_external_determinants`` limit therefore fails without building integrals
or determinants. The Python builder also excludes frozen-core holes while it
generates the space instead of generating and later discarding them.
``CASPT2(mc, frozen_core=...).estimate_external_space()`` exposes the same
counts as a read-only preflight operation, so orbital-space planning does not
require trial calculations.

For large single-state Fock-CASPT2 calculations,
``ic_basis_backend="direct"`` uses the non-enumerating production path. It
groups spin-adapted internally contracted functions in compact
external-signature x active-state tensors and solves the semicanonical
signature blocks independently. The two-phase tensor builder is the default;
``direct_build_backend="online"`` retains the older online-MGS implementation
as a numerical reference. Inactive and virtual orbitals are semicanonicalized
before the integral transformation. ``direct_workers="auto"`` parallelizes
independent tensor components and signature solves for large jobs; an integer
sets an explicit worker limit, and ``PYQED_CASPT2_WORKERS`` provides the same
process-level default.

The connected Hamiltonian RHS uses a native three-word determinant kernel, so
spaces with more than 31 spatial orbitals do not fall back to the old 64-bit
builder. The enumerated dense backend remains the small-system numerical
reference. ``use_cholesky=True`` keeps transformed pair factors instead of
assembling the four-index MO tensor. Its native RHS caches symmetry-equivalent
pair-factor dot products within each block, avoiding repeated auxiliary-index
contractions. Excitation-connected CAS references are stored once in globally
shared candidate groups rather than repeated for every direct row, and the
same representation is reused for SS/MS/XMS contractions. Local signature
Fock matrices and three-word Slater--Condon contractions are built by native
kernels, avoiding determinant-sized Python loops.

When the molecular orbitals carry native Abelian point-group labels, the
direct backend semicanonicalizes each irrep separately, rejects forbidden
one- and two-body excitation operators before determinant generation, and
retains only first-order rows with the reference-state irrep. Use a molecular
orientation consistent with the native point-group convention. This early
selection is important for production performance; merely filtering the final
rows would not reduce tensor-build cost.

``ic_basis_backend="auto"`` selects this direct path before FOIS generation
when the exact preflight count is at least 250,000 determinants and the request
is compatible with fully internally contracted Fock CASPT2. Smaller jobs keep
the dense/streaming reference selection. Thus a large calculation does not
need an empirical ``max_external_determinants`` or external-virtual cutoff to
discover the safe backend; all requested virtual orbitals can be retained and
the exact combinatorial count remains available from
``estimate_external_space()``.
Direct MS and XMS transition corrections are supported. Each term
:math:`\langle\Psi_I|H|\Psi_J^{(1)}\rangle` is evaluated in root :math:`J`'s
own semicanonical inactive/virtual basis. This is exact because rotations
within the fully occupied inactive and empty virtual subspaces leave the CAS
reference invariant. XMS roots naturally share the state-average Fock basis;
MS roots retain their state-specific Fock bases.

``max_memory_mb`` is a hard pre-allocation guard. ``ic_basis_backend="auto"`` uses dense canonical metric reduction for
small jobs and switches to streaming rank-revealing orthogonalization when the
raw metric would exceed that memory budget. The selected backend and byte
estimates are retained in ``ic_basis_backend`` and ``work_estimate``;
``success`` and ``message`` always reflect the final run state.
``linear_solver="auto"`` similarly keeps dense canonical diagonalization for
small retained spaces and switches to a matrix-free MINRES solve for larger
real-shift problems. Imaginary shifts use matrix-free GMRES. Convergence is
reported through ``solver_iterations``, ``solver_history``, and
``contracted_residual_norm``/``contracted_relative_residual_norm``; forcing
``linear_solver="direct"`` remains the reference path for cross-checks.
The Fock operator graph is assembled by the native C++ kernel, converted once
to sparse CSR form, and reused by every projected Krylov product. The raw IC
functions are built directly as compact, disjoint support components inside
their eight perturber classes, so the dense path never materializes the mostly
zero global raw basis or metric. Component-local canonical reduction removes
metric null modes. ``external_operator_nnz`` and the byte estimates in
``work_estimate`` expose these costs. ``solver_history`` retains the final true
MINRES residual and the inexpensive per-iteration GMRES residuals for
imaginary-shift solves.

Single-state calculation
------------------------

``CASPT2`` defaults to full internal contraction. The first-order space is
generated by applying spin-free one- and two-body excitation operators to the
whole CAS reference. Metric null modes are removed before solving the complete
projected generalized-Fock amplitude matrix.

.. code-block:: python

   from pyqed.qchem import CASPT2, CASSCF, Molecule

   mol = Molecule(
       atom="Li 0 0 0; H 0 0 1.6",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build()
   mf = mol.RHF().run()
   mc = CASSCF(mf, ncas=2, nelecas=2).run()

   pt = CASPT2(
       mc,
       ic_basis_backend="direct",
       direct_workers="auto",
       use_cholesky=True,
       real_shift=0.0,
       imaginary_shift=0.0,
       max_memory_mb=2048,
   )
   e_corr = pt.run()
   print(pt.e_tot, e_corr, pt.reference_weight)

For a symmetry-enabled production calculation, attach orbital irreps at the
integral/SCF stage, for example ``mol.build(symmetry="c2v")`` for a molecule in the standard C2v orientation. The selected
tensor, integral, worker, candidate-group, and symmetry paths are reported in
``ic_metric_backend``, ``direct_integral_backend``, and ``work_estimate``.

The nonorthogonal contracted functions are canonically orthogonalized. In the
retained basis, amplitudes satisfy

.. math::

   (K-sI)t=b,

where :math:`K` is the projected :math:`E_0-F` operator and :math:`s` is the
real level shift in PyQED's negative-denominator convention. For shifted
calculations, ``e_corr`` is the variational Hylleraas value

.. math::

   E^{(2)} = 2 b^Tt - t^T K t.

``e_corr_nonvariational`` and ``shift_correction`` expose the two reported
parts. ``contracted_residual_norm``, ``contracted_basis_size``,
``contracted_basis_rank``, ``first_order_norm``, and ``reference_weight`` are
available for reliability checks.

Multi-state calculations
------------------------

``MSCASPT2`` builds and diagonalizes the symmetrized second-order effective
Hamiltonian from state-specific first-order solutions:

.. code-block:: python

   from pyqed.qchem import MSCASPT2, XMSCASPT2

   ms = MSCASPT2(mc, roots=(0, 1))
   ms_energies = ms.run()

   xms = XMSCASPT2(mc, roots=(0, 1))
   xms_energies = xms.run()

For larger XMS calculations, all rotated roots can share the direct,
factorized backend:

.. code-block:: python

   xms = XMSCASPT2(
       mc,
       roots=(0, 1),
       ic_basis_backend="direct",
       use_cholesky=True,
       frozen_core=nfrozen,
   )
   xms_energies = xms.run()

XMS first diagonalizes the state-average generalized Fock operator in the
selected model space, rotates the references, and uses the same Fock operator
for all first-order equations. ``effective_hamiltonian``,
``reference_rotation``, and ``mixing`` retain the state-interaction data.
For determinant-based CI references, ``roots`` indexes the full fixed-
:math:`M_S` spectrum. Select spin-pure roots deliberately: in the LiH CAS(2,2)
validation the two OpenMolcas singlets correspond to PyQED roots ``(0, 2)``
because the triplet :math:`M_S=0` component lies between them.

Diagnostic approximations
-------------------------

``DiagonalCASPT2`` is an explicit determinant-diagonal diagnostic. The legacy
``contraction="strong"`` option uses only eight class vectors and must not be
reported as fully internally contracted CASPT2.

Validation
----------

Run the focused tests and external comparison harness from the repository
root:

.. code-block:: bash

   PYTHONPATH=. pytest -q tests/test_caspt2.py

   PYTHONPATH=. python benchmarks/caspt2_openmolcas.py \
       --cases all --contraction full --zeroth-order fock

The benchmark writes matched zero-IPEA OpenMolcas input, JSON diagnostics, and
a component/rank figure. With every inactive orbital correlated, the matched
LiH/STO-3G CAS(2,2) total energies differed by
:math:`2.34\times10^{-7}\ E_h`; the H\ :sub:`2`\ O/STO-3G CAS(4,4) totals
differed by :math:`1.89\times10^{-8}\ E_h`. The corresponding PyQED/OpenMolcas
contracted ranks were 57/58 and 138/138; the one-vector LiH difference reflects
the programs' different metric-reduction conventions and has negligible energy
effect. A larger LiH/cc-pVDZ CAS(2,2) CIONLY gate used 3,077 external
determinants and retained 915 of 1,525 nonzero raw contracted functions. The
automatic compact-basis, matrix-free calculation converged in 3 preconditioned
MINRES iterations with a relative residual of :math:`9.34\times10^{-11}` and
differed from OpenMolcas by :math:`5.41\times10^{-9}\ E_h`. The benchmark explicitly selects the Prascher
Li cc-pVDZ variant used by OpenMolcas; its CASCI references agree within
:math:`6.31\times10^{-10}\ E_h`, and unmatched references are flagged rather
than accepted as release comparisons. These runs used OpenMolcas commit
``cd52dbe08cf9611a376e0a434ef72d71659627ff``. The raw artifacts are written
under ``/private/tmp`` by the command above; this is one representative
validation set, not a broad accuracy claim.

On the same single-threaded machine, ten finalized repetitions gave isolated
PyQED/OpenMolcas CASPT2-kernel medians of 0.00486/0.03469 seconds for
LiH/STO-3G, 0.01202/0.03311 seconds for H\ :sub:`2`\ O/STO-3G, and
0.04511/0.04715 seconds for LiH/cc-pVDZ. Thus the representative larger gate is
at timing parity (PyQED/OpenMolcas ratio 0.957), while the two small gates are
faster in PyQED. The cc-pVDZ path uses a 125,649-nonzero native sparse Fock
operator, 105,984 bytes of compact raw IC blocks instead of the 56,832,192-byte
global dense-basis upper bound, component-local metric reduction, and a
projected-diagonal MINRES preconditioner. Timings are machine-specific; the
benchmark JSON records all repetitions and its generated figure compares the
current medians and minima. Very short external module lifetimes are sampled,
so sub-millisecond observations are omitted rather than reported as zero.

A larger native-builtin phenol CAS(6,6)/6-31G gate retained all 47 external
virtual orbitals while freezing 21 core orbitals. Its exact preflight FOIS
count was 3,730,040 determinants. C2v screening, compact tensor construction,
four direct workers, and cached Cholesky pair contractions reduced this to
736,738 active rows and rank 18,108. The correction was
:math:`-0.0536909642312\ E_h` with relative residual
:math:`9.89\times10^{-16}`; the CASPT2 step took 11.80 seconds on the local
test machine (10.21 seconds build, 1.37 seconds solve). The preceding C1
implementation with the same energy took 352.8 seconds, so these architecture
changes provide a 29.9x end-to-end CASPT2 speedup for this gate. This is a
PyQED before/after measurement, not an OpenMolcas parity claim; a matched
OpenMolcas phenol executable was not available on that machine.

For the same LiH fixed-orbital CIONLY reference, the two-state zero-IPEA
MS-CASPT2 energies agree with OpenMolcas within
:math:`3\times10^{-8}\ E_h`, and XMS-CASPT2 within
:math:`2\times10^{-8}\ E_h`. The effective-Hamiltonian off-diagonal elements
are included in ``tests/test_caspt2.py`` as independent regression values.

OpenMolcas describes production CASPT2 as an
iterative solution in a fully internally contracted first-order space; see the
`OpenMolcas CASPT2 manual
<https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html>`_.
