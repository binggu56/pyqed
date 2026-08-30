Native SU(2)-LETTA
===================

``SU2LETTA`` stores charge and spin multiplets rather than magnetic
projections.  ``D`` is therefore the number of reduced multiplets retained per
reachable virtual sector.

Unified frontier construction
-----------------------------

The public ``FrontierLETTA`` constructor recognizes a rank-coupled
non-Abelian MPO.  A qchem Hamiltonian carrying ``nelec`` and ``spin`` metadata
needs no separate solver class at the call site:

.. code-block:: python

   state = FrontierLETTA(
       H,
       graph=graph,
       D=32,
       adaptive_bond=True,
   )

With ``adaptive_bond=True``, ``D`` is a per-sector cap.  The state begins with
at most two copies of each reachable reduced multiplet and grows only the
``(N,S)`` sectors required by a two-site target.

The cap is not the realized bond dimension.  Inspect
``reduced_bond_multiplicities(bond)`` when comparing calculations: two runs
with different caps are identical if adaptive growth selects the same active
multiplets.  Growth is nested and state preserving.  The zero-padded tensor is
optimized first so newly allocated multiplets become live rather than being
removed as unsupported directions by the first ALS step.

For a generic reduced model, pass its reduced MPS site tensors and target
sector explicitly:

.. code-block:: python

   state = FrontierLETTA(
       H,
       sites=reduced_sites,
       target_sector=target,
       graph=graph,
       D=32,
   )

``tie="auto"`` uses physical-sector labels when a site is a direct sum of
several local irreps, as for empty/single/double spatial orbitals.  When every
site carries one fixed local irrep, it instead ties the future incoming fusion
sector.  Thus a spin-half model conditions on singlet/triplet fusion channels
rather than copying a magnetic projection.  ``tie="physical"`` and
``tie="fusion"`` select either representation explicitly.

Production-oriented run
-----------------------

.. code-block:: python

   state.run(
       nsweeps=8,
       algorithm="two_site",
       tol=1e-9,
       residual_tol=1e-8,
       truncation_tol=1e-7,
       consecutive_cycles=2,
       checkpoint="su2_letta.chk",
   )

The convergence decision is made only after complete LR/RL cycles.  Inspect
``state.convergence_summary`` for the energy change, maximum local residual,
maximum pair-retraction error, rejected updates, memory owned by the state,
and the number of consecutive qualifying cycles.

Before the first sequential cycle, the two-site solver optimizes the widest
pair space once.  This avoids an edge-first LR update selecting a poorer
nonlinear basin.  Set ``widest_pair_warmup=False`` only for controlled
schedule comparisons.

Pair updates
------------

Each adjacent pair is represented as a direct sum over its intermediate
charge-spin multiplet.  Small pair spaces use exact reduced transition
matrices.  Larger spaces use a matrix-free, factorized rank-coupled action and
an exact block metric; null metric directions are removed by a conditional
canonical whitening before Davidson.  The optimized pair is retracted into
the tied manifold. ``truncation_error`` is the scale-invariant physical
fidelity loss in that exact metric, while ``parameter_retraction_error``
reports the coordinate-space residual for debugging.  Only the former can
trigger adaptive multiplet growth.  The active path never expands magnetic
projections or projects the wavefunction into a dense determinant basis.

Checkpoint and restart
----------------------

``save_checkpoint(path)`` writes atomically and excludes transient contraction
caches.  Restore with ``SU2LETTA.load_checkpoint(path, workers=...)`` and pass
``reset_history=False`` to append new complete cycles.  Checkpoints use Python
pickle and should therefore be loaded only from trusted sources.

Validation and scaling
----------------------

Run ``examples/qchem/benchmark_su2_letta_production.py`` to compare Hubbard
chains with an independent exact fixed-N/fixed-Sz determinant calculation.
The JSON output includes timing, Python peak memory, persistent state storage,
parameter count, frontier size, residuals, and truncation diagnostics.  Use
multiple values after ``--D`` and ``--workers`` for scaling scans.

The local metric support can be much smaller than the raw pair space near a
boundary or when tied parameters are redundant.  ``solver_info`` reports both
``parent_dimension`` and ``metric_rank`` so this reduction is visible in
benchmarks.
