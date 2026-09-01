Native SU(2)-LETTA
===================

``SU2LETTA`` stores charge and spin multiplets rather than magnetic
projections.  ``D`` is therefore the number of reduced multiplets retained per
reachable virtual sector.

Construction
------------

By default, SU(2)-LETTA ties only nearest-neighbor orbitals in their current
order.  This keeps the frontier width bounded even though the quantum-chemistry
Hamiltonian MPO retains every one- and two-electron coupling.  Pass ``graph``
explicitly to use a different variational tie graph; the graph does not screen
or modify the Hamiltonian.  ``graph="nn"`` (or ``"nearest_neighbor"``) selects
the default chain explicitly.

Build directly from a completed mean-field calculation through the qchem
driver:

.. code-block:: python

   from pyqed.qchem.letta import LETTA

   state = LETTA(
       mf,
       symmetry="su2",
       D=32,
       n_threads=4,
   )

For an active-space calculation, additionally pass ``ncas``, ``nelecas``, and
optionally ``ncore`` or ``mo_coeff``.  ``LETTA.from_integrals`` remains
available for model Hamiltonians with precomputed spatial-orbital integrals.
The generic tensor ansatz remains ``pyqed.letta.LETTA``.

For a pre-existing fully reduced spatial-orbital MPS, use the native class:

.. code-block:: python

   from pyqed.letta import SU2LETTA

   state = SU2LETTA.from_mps(
       reduced_sites,
       rank_coupled_mpo,
       target_sector=target,
       graph=graph,
   )

``tie="auto"`` uses physical-sector labels when a site is a direct sum of
several local irreps, as for empty/single/double spatial orbitals.  When every
site carries one fixed local irrep, it instead ties the future incoming fusion
sector.  Thus a spin-half model conditions on singlet/triplet fusion channels
rather than copying a magnetic projection.  ``tie="physical"`` and
``tie="fusion"`` select either representation explicitly.

Restartable run
---------------

.. code-block:: python

   state.run(
       nsweeps=8,
       algorithm="projected",
       tol=1e-9,
       residual_tol=1e-8,
       truncation_tol=1e-7,
       consecutive_cycles=2,
       gauge="conditional",
       reuse_environments=True,
       checkpoint="su2_letta.chk",
   )

The convergence decision is made only after complete LR/RL cycles.  Inspect
``state.convergence_summary`` for the energy change, maximum local residual,
maximum pair-retraction error, rejected updates, memory owned by the state,
and the number of consecutive qualifying cycles.

For one-site updates, ``solver="auto"`` selects the native reduced
Wigner--Eckart path.  ``solver="polarization"`` is retained only as an explicit
small-system reference calculation; it is not selected automatically.

``algorithm="projected"`` is the preferred fixed-D optimizer.  It embeds the
tied one-site parameters into the adjacent channel-resolved DMRG pair space,
projects the effective Hamiltonian and norm actions with that embedding, and
removes null or redundant tied directions by diagonalizing the projected norm.
The retained basis is norm-orthonormal before ordinary Hermitian Davidson is
started.  This avoids an ill-conditioned generalized Krylov problem and lets
the local solve reuse the incrementally advanced reduced DMRG environments.
An update is committed only when Davidson converges and its recomputed local
Rayleigh quotient does not increase.  No magnetic or determinant-space
projection is used.

On states that retain the compiled SU(2) owner, the projected optimizer
installs ``E† H_eff E`` as indexed projections of only the active reduced
blocks.  Davidson vectors therefore remain in the tied orthonormal space
without Python lift/apply/project callbacks or dense zero-padded projection
matrices.  ``E† N_eff E`` is likewise contracted directly from factorized
reduced metric blocks.  Projection topology, batched one-site
Wigner--Eckart embedding bases, metric whiteners, and small Davidson/Ritz
spaces are cached across cycles; numerical transforms are refreshed whenever
their array revision changes.  Norm whitening is split into independent
connected blocks and uses direct diagonal/identity scaling whenever the
conditional gauge permits it.  The compiled Davidson solve is accepted only
after its true residual passes the requested tolerance; otherwise the solver
continues with the recycled Python Davidson space.  A projected
parent-Hamiltonian diagonal controls seed ordering, while a robust constant
shift is used for correction because omitted transformed off-diagonal terms
make that inexpensive diagonal unsafe as a direct denominator.

The tied-to-reduced coordinate map is compiled directly from frontier-sector
slices and parameter offsets.  It no longer constructs a materialized MPS
tensor for every unit parameter direction.  Unit-direction ``IrrepTensor``
objects are generated lazily only for explicit reference solvers.  Numerical
reduced MPS sites are retained by the state and invalidated precisely when a
site parameter or crossing conditional gauge changes.  Thus the production
path carries the tie constraint as a persistent reduced-coordinate scatter
map while retaining the rank-filtered metric basis required to remove null
directions.

For a physical nearest-neighbor tie chain, ``gauge="conditional"`` applies an
exact reduced QR gauge independently for every crossing physical SU(2) sector.
The adjacent tensor absorbs the transfer matrix, so the graph-tied state is
unchanged.  Projected and two-site sweeps then reuse incrementally advanced
Hamiltonian and norm environments within and across half-sweeps whenever the
tracked canonical center already has the required position.  A real gauge
coordinate change invalidates the affected side and triggers a rebuild.  The
projected cycle energy and norm are taken from its terminal exact local
Rayleigh quotient, avoiding redundant full-chain contractions; validation
tests compare these values to explicit chain contractions.  General graphs
that do not expose one next-site condition
per internal frontier skip the conditional gauge; ``gauge=None`` and
``reuse_environments=False`` select the rebuild-every-bond reference path.

States constructed by ``SU2LETTA.from_integrals`` retain the transient C++
SU(2) system owner.  Their two-site Hamiltonian actions install exact
channel-resolved contextual routes in that owner.  For chains of at least six
sites, Hamiltonian environments are advanced by the same C++ owner using exact
reduced boundary route batches.  Shorter chains retain the faster exact Python
boundary recursion because route packing cannot be amortized.  The factorized
norm stack remains explicit in Python because the LETTA retraction consumes its
metric blocks directly.  LETTA disables the
DMRG-specific direct-complementary shortcut because its unfolded conditional
virtual basis requires the full contextual route topology.  Checkpoints and
deep copies drop the non-pickleable owner and use the same exact reduced Python
route path.  Immutable left/right reduced MPO recoupling blocks are cached per
core and reused by environment construction, local actions, and expectation
values.

``n_threads`` configures the OpenMP team owned by the compiled reduced engine.
It parallelizes independent dependency waves and sufficiently large route or
block batches; small batches deliberately remain serial because thread-launch
overhead exceeds their work.  This is separate from ``workers``, which controls
Python-side independent local setup.  Avoid combining large values for both
with a multithreaded BLAS.

For the pyrazine STO-3G CAS(6,6), ``D=4`` nearest-neighbor benchmark, the first
cycle takes about 3.0 seconds and steady complete LR/RL cycles take about
1.9--2.0 seconds on the development machine.  The one-thread three-cycle total
is about 7.1 seconds.  This small ``D`` does not contain enough parallel
reduced-block work for material OpenMP scaling.  The run retained the CASCI
energy within ``1e-13`` Hartree.  This is the fast
LETTA path, but its tied metric and embedding work still make it slower than an
unconstrained SU(2)-DMRG sweep.  Timings are machine-dependent.

Pair updates
------------

Each adjacent pair is represented as a direct sum over its intermediate
charge-spin multiplet.  The default is a matrix-free, factorized rank-coupled
Hamiltonian action with an exact reduced block metric; null metric directions
are removed by a conditional canonical whitening before Davidson.  Set a
positive ``dense_dim`` on ``optimize_two_sites`` only to retain explicit small
transition matrices as a reference path.  The optimized pair is retracted into
the tied fixed-D manifold.  Retraction embeddings reuse the cached one-site
Wigner--Eckart route basis instead of rematerializing every unit parameter.
``truncation_error`` is the relative error in the
reduced pair metric and ``discarded_weight`` is its square; the separate
``coefficient_retraction_error`` is diagnostic only because it includes gauge
and metric-null directions.  Empty contextual operators, which can occur when
a Hamiltonian vanishes in a target-spin sector, are represented as valid zero
route tables.  The active path never expands magnetic projections or projects
the wavefunction into a dense determinant basis.

The older ``algorithm="two_site"`` route optimizes the unrestricted reduced
pair first and retracts it back to the tied manifold.  It remains useful as a
reference, but pair-space growth and nonlinear retraction make it less suitable
than the projected optimizer at larger ``D``.

Checkpoint and restart
----------------------

``save_checkpoint(path)`` writes atomically and excludes transient contraction
caches.  Restore with ``SU2LETTA.load_checkpoint(path, workers=...)`` and pass
``reset_history=False`` to append new complete cycles.  Checkpoints use Python
pickle and should therefore be loaded only from trusted sources.

Validation and scaling
----------------------

Run ``examples/qchem/validate_su2_letta.py`` for exact dimer checks and a
reproducible convergence figure.  The restored implementation is presently a
validated reduced-space development path, not a claim of block2-equivalent
production performance for large active spaces.

The local metric support can be much smaller than the raw pair space near a
boundary or when tied parameters are redundant.  ``solver_info`` reports both
``parent_dimension`` and ``metric_rank`` so this reduction is visible in
benchmarks.

Method provenance and scope
---------------------------

The charge-spin reduced tensors and recoupling follow the Wigner--Eckart
non-Abelian tensor-network formulation of A. Weichselbaum, *Phys. Rev. B* 86,
245124 (2012), `doi:10.1103/PhysRevB.86.245124
<https://doi.org/10.1103/PhysRevB.86.245124>`_.  SU(2)-LETTA is a PyQED
adaptation of that reduced-tensor machinery to the project-specific
graph-tied LETTA ansatz; it is not an exact reproduction of the paper's MPS or
NRG algorithm.  Only fully reduced spatial-orbital SU(2) states and scalar
rank-coupled qchem Hamiltonians are supported.  General non-Abelian groups,
arbitrary local irreps, and block2-level large-active-space performance are
not implied.
