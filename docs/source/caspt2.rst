CASPT2
======

Status and scope
----------------

``CASPT2``, ``MSCASPT2`` and ``XMSCASPT2`` provide restricted-real, zero-IPEA,
fully internally contracted energy calculations. They are **Beta**: focused
analytic-limit, invariance and matched OpenMolcas tests pass, but broader
molecular, active-space, intruder-state and platform coverage is still needed.
Analytic gradients, IPEA shifts, RMS-CASPT2 and correlated transition
properties are not implemented. MS/XMS supports imaginary shifts; real-shift
effective-Hamiltonian couplings are not implemented.

The reference must contain the important near-degenerate configurations.
Perturbation theory does not repair an inadequate active space. Inspect the
reference weight, retained metric rank and convergence diagnostics, and check
sensitivity to the active space, basis, metric thresholds and level shift.

Single-state calculation
------------------------

.. code-block:: python

   from pyqed.qchem import CASPT2, CASSCF, Molecule

   mol = Molecule(
       atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g"
   ).build(eri="s8")
   mc = CASSCF(mol.RHF().run(), ncas=2, nelecas=2).run()

   pt = CASPT2(
       mc,
       ic_basis_backend="direct",
       direct_build_backend="shared",
       ic_convention="molcas",
       imaginary_shift=0.25,
       max_memory_mb=2048,
   )
   e_corr = pt.run()
   print(pt.e_tot, e_corr, pt.reference_weight)

``run()`` returns the correlation correction and populates the driver.
``e_tot`` includes the reference energy. ``success`` and ``message`` report
whether the run completed; failures raise an exception.
``frozen_core`` excludes the requested number of lowest doubly occupied
orbitals from the first-order space while retaining their mean-field effect.

Contraction conventions
-----------------------

``contraction="full"`` is the default. Spin-free single and double excitation
operators act on the entire CAS reference, and metric null modes are removed
before solving the coupled amplitude equation. In an orthonormal retained
basis, PyQED uses the negative-denominator convention

.. math::

   (K-sI)t=b, \qquad K=P(E_0^{(0)}-F)P.

Here :math:`b` is the projected Hamiltonian coupling and :math:`s` is the real
level shift. The reported correction is the Hylleraas value

.. math::

   E^{(2)}=2b^Tt-t^TKt.

``e_corr_nonvariational`` retains :math:`b^Tt`, and ``shift_correction`` is
the difference between that value and the reported variational-functional
correction. This terminology does not make the total CASPT2 energy a
variational upper bound.

Two explicit conventions are available:

* ``ic_convention="projected"`` is the default exact Galerkin projection of
  the generalized Fock operator onto the retained contracted space.
* ``ic_convention="molcas"`` reproduces OpenMolcas's ordered generators,
  metric reduction and A/E (Si/Sijr) coupling convention for the supported
  zero-IPEA equations. It requires ``ic_basis_backend="direct"`` and
  ``direct_build_backend="shared"``. The A/E coupling is :math:`\sqrt{2}`
  times the determinant-projected coupling, following ``SGM`` cases 4/5 and
  ``MKLIST`` lists 14/15. The convention is explicit, rather than an energy fit.

In the Molcas convention, CI coefficients are counterrotated while the active
Fock operator is diagonalized within each RAS partition. Redundant singles
are folded into number-operator traces. ``metric_norm_tol`` screens squared
raw-column norms, whereas ``lindep_tol`` screens diagonally scaled metric
eigenvalues. Their defaults are respectively ``1e-10`` and ``1e-8`` for
Molcas, and zero and ``1e-10`` for projected calculations. Near a cutoff,
small integral or orbital-frame changes can alter the retained rank.

``DiagonalCASPT2`` is an explicit determinant-diagonal diagnostic.
``contraction="strong"`` retains only eight class vectors. Neither should be
reported as fully internally contracted CASPT2. ``zeroth_order="en"`` is an
Epstein--Nesbet diagnostic rather than the production Fock-CASPT2 formulation.

Choosing the first-order representation
---------------------------------------

``ic_basis_backend="auto"`` uses dense or streaming canonical reduction for
small spaces. Compatible Fock-CASPT2 requests with at least 250,000 estimated
external determinants select the direct representation before enumeration.

``ic_basis_backend="direct"`` supports these builders:

* ``direct_build_backend="tensor"`` is the default compact
  external-signature by active-state representation. It retains inter-block
  Fock couplings and uses block eigensystems to precondition GMRES.
* ``direct_build_backend="shared"`` reuses active templates across external
  orbital labels. Hamiltonian coefficients and a sparse Fock graph replace
  repeated determinant-to-reference loops. It supports the projected and
  Molcas conventions, with no extra external-virtual cutoff.
* ``direct_build_backend="online"`` is the incremental orthogonalization
  reference for numerical comparisons.

For the shared builder, ``fock_contraction="operator"`` uses sparse operator
projection and ``"density"`` uses exact annihilation-factor contractions.
``"rdm"`` evaluates metrics, Fock moments and inter-class couplings using
RAS-safe active 1--3 RDMs and energy-weighted RDMs, without a four-RDM or
external-determinant templates. It requires a pseudocanonical active Fock
operator and ``metric_norm_tol >= 1e-12``. The default ``"auto"`` selects RDM
contractions for the direct shared Molcas convention, and operator
contractions otherwise.

``max_memory_mb`` guards planned basis, matrix and solver allocations.
``max_ic_operators`` caps raw excitation operators before metric reduction.
``max_external_determinants`` caps the estimated enumerated space, and
``estimate_external_space()`` provides a read-only preflight without integral
transformation. Counts are upper bounds when a CVS projector is requested.
The direct representation avoids global external-space enumeration, but still
has active-space and contracted-space costs.

``use_cholesky=True`` retains available transformed RI/Cholesky pair factors.
For packed AO integrals, the shared solver transforms only
:math:`(pi|qj)` with :math:`i,j` in the inactive plus active space; it does not
build three- or four-virtual integrals. This is an exact data-layout change,
subject to any integral approximation already present in the reference.

``direct_workers="auto"`` uses the central ``pyqed.num_workers()`` budget.
An explicit integer or ``PYQED_CASPT2_WORKERS`` sets the solver limit.
Independent build, coupling and solve stages reuse that budget without nested
pools; keep BLAS threads at one. Small components run serially. Root-package
builds honor ``PYQED_QCHEM_OPENMP=required`` when OpenMP is needed; automatic
builds retain the serial coupling path if OpenMP is unavailable.

Multistate calculations
-----------------------

.. code-block:: python

   from pyqed.qchem import MSCASPT2, XMSCASPT2

   options = dict(
       roots=(0, 2),
       ic_basis_backend="direct",
       direct_build_backend="shared",
       ic_convention="molcas",
       imaginary_shift=0.25,
   )
   ms = MSCASPT2(mc, **options)
   ms_energies = ms.run()
   xms = XMSCASPT2(mc, **options)
   xms_energies = xms.run()

The reference must have been solved for all requested roots. ``roots`` indexes
the reference solver's spectrum: determinant-based fixed-:math:`M_S` spectra
can contain both singlets and triplets. Select spin-pure states deliberately;
the LiH CAS(2,2) singlet validation uses roots ``(0, 2)``.

MS solves each selected root with its state-specific generalized Fock operator.
XMS diagonalizes the state-average Fock operator in the selected model space,
rotates the references and uses that common operator for the first-order
problems. Each ket retains its own internally contracted space; this is not
the union-space implementation in Shiozaki et al. (2011). ``weights`` specifies the XMS state-average weights.
``effective_hamiltonian``, ``reference_rotation``, ``mixing`` and
``state_specific`` retain the assembly and individual solves.

Shared multistate calculations require RDM contractions. Ordered transition
1--3 RDMs retain both transition orientations and intermediate determinants
outside a RAS reference. Each transition reference is counterrotated into the
ket's active Fock frame. The other direct builders evaluate transitions in
each ket's semicanonical inactive/virtual frame.

The imaginary shift follows the diagonalized signature-block prescription of
Forsberg and Malmqvist. Effective-Hamiltonian diagonals contain the
shift-corrected Hylleraas energies; off-diagonal terms use shifted real
amplitudes and are symmetrized, following OpenMolcas ``MCCTL``, ``HEFVAL`` and
``MLTCTL``. This reproduces that energy convention, rather than a full
multistate Hylleraas functional. The selected state window still requires
convergence checks.

RAS references and CVS
----------------------

The direct single-state path accepts restricted-real RASCI/RASSCF references.
Active Fock rotations preserve the RAS partitions, and density contractions
retain the excitation-reachable intermediate space outside the reference RAS.
Ordinary RASPT2 includes first-order occupations outside a one-core-hole
reference sector.

``cvs_core_orbitals`` optionally applies an additional fixed one-hole
core-valence projector to both reference and first-order space. It requires a
complete active CI within that sector and the tensor or online direct builder.
The shared builder and multistate CVS mixing do not support this extra
projector. This projected Fock-PT2 adaptation is not RMS-RASPT2.

Tiled elimination of external doubles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared RDM solver can eliminate the class with two holes in correlated
inactive orbitals and two particles in external virtual orbitals.

.. code-block:: python

    xms = XMSCASPT2(
        mc, roots=roots, frozen_core=nfrozen, use_cholesky=True,
        ic_basis_backend='direct', direct_build_backend='shared',
        fock_contraction='rdm', ic_convention='molcas',
        eliminate_external_doubles=True, max_ic_operators=None,
        max_memory_mb=workspace_mib,
    )
    xms.run()

In the semicanonical orbital basis this class has a diagonal within-class
operator. The solver takes its exact Schur complement, retaining the
off-diagonal Fock couplings to the other classes. It constructs direct and
exchange integral tiles from rectangular RI/Cholesky factors and recovers
external-double amplitudes only within a tile. The same replay accumulates
the unshifted Hylleraas energy, the XMS state-interaction terms, and the
full-equation residual. The amplitude equations use the same level shifts
as the original shared solver. A singular coupled diagonal block requires
an imaginary shift. This adapts the storage and linear algebra
of zero-IPEA XMS-CASPT2 (Shiozaki et al., J. Chem. Phys. **135**, 081106
(2011), `doi:10.1063/1.3633329 <https://doi.org/10.1063/1.3633329>`_).
The blocked RI integral evaluation follows the pair-factor representation
used for CASPT2 by Aquilante et al., J. Chem. Phys. **131**, 034113 (2009),
`doi:10.1063/1.3157463 <https://doi.org/10.1063/1.3157463>`_.
Fock couplings that share a virtual index use batched matrix products within
each integral tile. During the iterative solve, a compiled loop applies the Schur feedback
using shared Fock blocks and implicit pair indices. Large applications use
the ``direct_workers`` budget. Each round contains disjoint occupied pairs,
so workers accumulate into the output without copying the Krylov vector.
``external_doubles_matvec_s`` measures this step over the solve. The transition
corrections reuse each integral batch over
all bra roots. XMS roots in the same pseudocanonical orbital frame share the
rectangular RI transform. Its occupied-side indices include the correlated
inactive and active orbitals.

The shared Schur equations are real symmetric for both the unshifted and
blockwise imaginary-shift forms. The reduced solve uses a buffer-reusing adaptation of MINRES with a
positive diagonal preconditioner and checks the actual residual. The short
recurrence of Paige and Saunders, SIAM J. Numer. Anal. **12**, 617--629
(1975), `doi:10.1137/0712047 <https://doi.org/10.1137/0712047>`_, avoids a
full stored Krylov basis. PyQED reserves the equivalent of 10 full-length
vectors for the reduced solve, including the RHS, recurrence buffers, and
temporary vectors. The unshifted and shifted action diagonals are reconstructed in
bounded blocks from separate hole, virtual, and active energy arrays, rather
than stored as full-length vectors. The positive diagonal preconditioner
uses the same block generator, so it also needs no full-length storage.
The Hamiltonian RHS is built directly into one packed buffer. During MINRES,
this buffer holds the Schur RHS; afterward the original Hamiltonian RHS is
regenerated into it for energies and full-equation checks. This trades an
additional retained-RHS construction for one fewer live vector. Final residual
norms are accumulated in blocks from the unshifted action and the exact
shifted-minus-unshifted diagonal correction, without full residual arrays.
MINRES scales its oldest Lanczos residual and accumulates the next Schur
operator product directly into that buffer. Two rotating buffers hold the
previous residual and current preconditioned Lanczos vector. After the solution
update, the latter is converted back to its unpreconditioned residual using
the diagonal and Lanczos normalization. This removes both a separate
matrix-vector output and a separate preconditioned-residual allocation. The
true residual is still recomputed independently before accepting convergence.
The initial guess retains the signed denominators. Recurrence updates reuse the initial guess and two direction buffers,
and array temporaries are limited to 65,536 entries. Residual restarts use the
original equation, and inconsistent singular systems are rejected rather than
accepted as least-squares solutions. Orbital groups store separate compact hole and virtual axes. The
builder derives product rows by indexing those axes.

Every retained two-body excitation has an active orbital index. For the RI
path, matrix products assemble only the Gram block between transition pairs
with an active index and all allowed transition pairs. Ket and bra projections
look up the same integrals. XMS roots in the same orbital frame share this
block. Before MINRES, the block is released because the Schur action uses RI
factors directly. It is rebuilt after the solve for RHS restoration and
transition couplings, preventing overlap with the Krylov workspace. The memory
estimate takes the maximum of the solve and projection stages instead of
adding their allocations. This adds a Gram-block reconstruction per solve. When the block and its construction workspace exceed the solver
budget, the solver evaluates the requested pair contractions on demand.
The eliminated external-double class still evaluates its integral tiles
directly from the RI factors.

This option retains the requested virtual space and avoids a class-sized
label list, integral array, amplitude vector and Fock graph. The retained
Fock graph skips insertions into absent classes before allocating candidate
edge arrays. It generates the remaining insertions in bounded batches and
computes row indices directly from the orbital axes. The reduced solver
stores the active transition matrices per pair of templates and insertion
pattern. During a bounded discovery pass it caches external row, column, and
insertion labels only while they fit a 16 MiB cap. The cap is reduced to at
most one quarter of the workspace remaining after graph-generation and
transition-construction reserves. If the cap is exceeded, cached batches are
discarded and external labels are regenerated in batches of at most 16,384
candidates at each action. The cache is also discarded if active matrices
need that memory. Thus small graphs avoid repeated edge generation while
large graphs retain bounded edge storage. Active orbitals share the indices
and read Fock weights from the orbital matrix. The per-solve cache does not
persist across roots; the assembled reference path retains its separate cache.
The action shares projections onto the smaller active rank when they fit
two fixed scratch panels. Larger blocks use batches of edges, gathering and
projecting only the current batch in each direction. Both paths retain all
couplings, including repeated edge indices. Each panel holds at most 65,536
values (or one active row if larger); a third panel-equivalent is reserved
for gather/product temporaries. Scratch no longer scales with orbital groups.
Batching may repeat projections and increase runtime. ``fock_topology_bytes`` reports the
persistent edge indices (zero when the adaptive Schur path selects streaming), while
``fock_graph_bytes`` includes active matrices and the bounded edge-generation
and contraction workspace. The assembled reference path still stores edges.
Evaluating
the tiles still requires quartic pair work. The other classes retain their
existing storage and can dominate the workspace. The raw-operator cap,
when set, applies to those remaining classes. The workspace budget covers
the contracted solver and tile workspace and excludes input integrals and
the AO-to-MO transformation. Each root reports
``eliminated_external_double_groups`` and ``iterative_ic_rank`` in
``work_estimate``. ``retained_ic_rank`` still counts the full contracted
rank. Shifted and unshifted coupled comparisons and a two-root molecular RI
test cover this option. TDI-scale runtime and peak memory are unmeasured.

A matched two-root naphthalene/cc-pVDZ CAS(8,8) benchmark, with four workers
and single-threaded BLAS, measured 15.65 seconds and 1.09 GiB peak RSS with
the adaptive edge cache. Unconditional streaming took 17.44 seconds and
1.08 GiB. The effective Hamiltonian was identical at stored precision and
full relative residuals stayed below 9e-13. The cache selected 4.78 MiB of
edge indices; graphs exceeding its memory cap are streamed. These are
single-run comparisons, not TDI-scale measurements. Process RSS includes
allocations outside the solver workspace budget.

Diagnostics and validation
--------------------------

Inspect ``contracted_basis_size``, ``contracted_basis_rank``,
``first_order_norm``, ``reference_weight``, ``solver_iterations`` and
``solver_history``. ``work_estimate`` records representation, integral,
workspace and shared-template details; ``timings`` records calculation stages.
For imaginary shifts, the unregularized residual can remain nonzero:
``work_estimate["regularized_relative_residual"]`` reports the shared solver's
actual equation residual.

Run focused checks from the repository root::

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q \
     tests/test_caspt2.py tests/test_perturber_templates.py \
     tests/test_cvs_pt2.py tests/test_ras_density_contractions.py \
     tests/test_fic_contractions.py

These cover the MP2 limit, dense/streaming/direct/shared agreement, shifts,
frozen core, orbital and phase invariance, MS/XMS effective Hamiltonians,
RAS intermediate closure, memory guards, and compiled/Python contractions.
The nonstationary H6/STO-3G CAS(2,2) regression exercises the A/E convention
against fixed-orbital OpenMolcas corrections at imaginary shifts zero and
0.25 hartree. Its reproducible decks are in
``benchmarks/data/caspt2_h6_nonstationary/``. The tests do not require OpenMolcas.
``benchmarks/caspt2_openmolcas.py`` provides the external comparison harness.

Method references and fidelity
-------------------------------

* K. Andersson, P.-A. Malmqvist and B. O. Roos, *J. Chem. Phys.* **96**,
  1218--1226 (1992), doi:10.1063/1.462209: complete-CAS second-order theory.
* P.-A. Malmqvist et al., *J. Chem. Phys.* **128**, 204109 (2008),
  doi:10.1063/1.2920188: the RASPT2 formulation used by the direct RAS path.
* J. Finley et al., *Chem. Phys. Lett.* **288**, 299--306 (1998),
  doi:10.1016/S0009-2614(98)00252-8: multistate state interaction.
* T. Shiozaki et al., *J. Chem. Phys.* **135**, 081106 (2011),
  doi:10.1063/1.3633329: the extended multistate reference rotation; the
  contracted-space distinction above limits implementation fidelity.
* N. Forsberg and P.-A. Malmqvist, *Chem. Phys. Lett.* **274**, 196 (1997),
  doi:10.1016/S0009-2614(97)00669-6: imaginary level shifts.
* `OpenMolcas CASPT2 sources
  <https://github.com/Molcas/OpenMolcas/tree/master/src/caspt2>`_ and
  `manual <https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html>`_:
  ordered metrics, A/E couplings and shifted multistate energy assembly.
* L. S. Cederbaum et al., *Phys. Rev. A* **22**, 206 (1980),
  doi:10.1103/PhysRevA.22.206: the core-valence separation underlying the
  optional projected single-state adaptation.

The sparse, template and RDM implementations reorganize the supported equations
without cumulant approximations. The explicit projected/Molcas distinction,
zero-IPEA restriction, unsupported properties and CVS restrictions above are
material limits; general accuracy claims for other CASPT2 variants do not
carry over automatically.
