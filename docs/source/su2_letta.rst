Native SU(2)-LETTA
===================

``SU2LETTA`` stores charge and spin multiplets rather than magnetic
projections.  ``D`` is therefore the number of reduced multiplets retained per
reachable virtual sector.

Closed SU(2) virtual multiplicities
----------------------------------

``SU2LETTA.from_integrals(..., virtual_boundary='periodic')`` closes the
last-to-first **multiplicity index**, independently of the physical tie graph.
The seam has spin zero and dimension ``D``. Charge is carried by a fixed
boundary twist connecting the total-particle sector to the vacuum sector;
only singlet targets are supported. This is a restricted periodic SU(2)
ansatz, not a ring containing arbitrary spin-carrying seam sectors.

For each allowed reduced fusion path the amplitude is the multiplicity trace

.. math::

   c_{p_1\ldots p_n}=\sum_{a=1}^{D}
     (A_1^{p_1} A_2^{p_2}\cdots A_n^{p_n})_{aa}.

An exact spectator-index lift carries ``a`` through reduced open environments.
It adds no independent variational parameters and introduces no truncation,
magnetic-state expansion or determinant-space projection. Interior contraction
multiplicities grow by ``D`` (up to ``D**2`` per sector before tie frontiers);
this implementation does not compress periodic transfer matrices.

Closed-loop states default to ``compact_frontier=True``. Forward and backward
reachability through structural parameter routes removes virtual coordinates
that occur on no complete path. This preserves the full tied ansatz, its raw
parameter count and all contributing tangent directions, even when their
current coefficients are zero. It uses no numerical tolerance, SVD, or
variational projection. The existing reduced SU(2) recoupling is unchanged.
``frontier_compaction`` reports original and retained multiplicities per cut;
``compact_frontier=False`` selects the unfolded validation reference. The policy
is retained by checkpoint restart and graph additions.

Route compilation constructs scalar-backed shape templates and maps only
structural entries through the seam, so it does not first allocate the dense
unfolded site tensors or full Cartesian seam-index tables. Numerical site
tensors are allocated after structural compaction. Retained frontier matrices
and environments can still grow rapidly with the number of ties crossing a
cut; this is not a graph-independent memory bound or a convergence guarantee.

``examples/qchem/benchmark_su2_frontier_layout.py`` compares the two layouts in
separate single-thread processes and writes reproducible PNG/PDF diagnostics.
The six-orbital closed D=2 reference verifies energies, norms, and one-site
Hamiltonian/norm actions. An optional ``--checkpoint`` measures layout
construction only and does not evaluate the molecular Hamiltonian. In the
saved pyrene CAS(16,16), D=2 layout, stored site arrays decrease from 130.5 to
80.5 MiB and the largest reduced cut dimension from 4860 to 3256, retaining
all 15960 parameters. These are layout measurements, not a full molecular
sweep or convergence benchmark.

Use ``run(algorithm='projected')`` for fixed-D, one-site reduced tensor updates.
Here ``projected`` means the existing tied-parameter local design map, not a
final many-body projection. Auto selects the fully reduced Hamiltonian MPO;
direct complementary Hamiltonians, rank-changing two-site updates, orbital
transformations and conditional periodic canonical gauges are unsupported.
Checkpoint restart and additions of physical ties preserve the closure.

``local_tangent_residual()`` freshly evaluates every one-site residual with all
other tensors fixed at the current state, whitened in the retained reduced norm
metric. Its maximum diagnoses retained-metric one-site stationarity. Residuals
recorded during local eigensolves do not certify stationarity after the other
tensors change. Neither criterion guarantees the global variational minimum.

The periodic trace is adapted from F. Verstraete, D. Porras and J. I. Cirac,
"DMRG and periodic boundary conditions: a quantum information perspective",
Phys. Rev. Lett. **93**, 227205 (2004),
https://doi.org/10.1103/PhysRevLett.93.227205. The reduced algebra follows the
Weichselbaum formulation cited in the public class docstring. This combination
is a LETTA adaptation, not a reproduction of either paper's optimization
algorithm; their convergence claims do not carry over.

``examples/qchem/benzene_su2_closed_virtual.py`` compares the saved localized
cc-pVDZ CAS(6,6) open D=2 ring-tied state with a closed D=2 state warm-started
from it plus a small second seam component. It uses saved CASCI energy only
as an independent reference, audits stationarity and writes reproducible
energy-error/residual plots. This compares variational refinement, not speed
from identical initial tensors.

The 12-cycle refinement (seed 11) gives CASCI errors of 0.310609 mEh
(open) and 0.310460 mEh (closed). Fresh maximum local tangent residuals are
1.833e-3 and 1.836e-3 Eh, respectively, far above the 1e-7 Eh target;
neither run is converged. Cumulative local times including audits are 36.2
and 112.2 seconds. This restricted seam shows no useful convergence benefit
in this comparison. Both calculations use reduced SU(2) operators throughout.

``examples/qchem/benzene_su2_closed_converge.py`` resumes a closed checkpoint
and jointly refines its reduced tensor parameters in restarted local-norm
charts. Exact sector-wise QR transfers balance the conditional coefficients
before each restart, preserving the trace and tensor shapes; this is not a
periodic canonical norm gauge or a Schmidt truncation. With
``--conditional-balance``, a transfer can depend on the next orbital's physical
sector when a neighbor tie shares that label. Other conditions are stacked,
so the inverse transfer is representable in the next tensor even with a
long ring tie. This is exact coefficient balancing, without a periodic
norm-orthogonality guarantee.

``state.balance_coefficients(conditional=True, rcond=1e-12)`` exposes coefficient
balancing without constructing Hamiltonian or norm environments. It adapts QR
gauge absorption from U. Schollwoeck, Ann. Phys. 326, 96--192 (2011),
https://doi.org/10.1016/j.aop.2010.09.012, to shared physical labels in reduced
LETTA tensors. Only internal bonds are transformed, not the periodic closing
bond. Rectangular, null, and relatively ill-conditioned transfers are skipped;
no singular values are truncated. Per-transfer diagnostics report these skips.
Updates are staged before the site's normal setter, so revision-dependent
stationarity certificates and materialized tensors cannot survive a changed
coordinate system. This remains opt-in and does not construct a canonical
periodic metric or certify floating-point state preservation. The benzene
helper uses this method while retaining its independent energy check.

On the pyrene diagnostic checkpoint the guarded method applies 417 transfers
and skips 212, yielding a norm spread of 2.42e-13. Its coordinates differ from
the earlier unguarded QR diagnostic, so the Hamiltonian audit of that earlier
checkpoint is not a direct validation of this API's output.

A direct one-site
adjoint contracts the reduced left/right boundaries, Wigner--Eckart operator
blocks and SU(2) cut norms, then pulls them back through the tied route map.
Tests compare it against independently built channel-resolved pair actions,
and check complex derivatives by energy finite differences.
One-site parameter actions advance one left boundary and consume exact right
boundary checkpoints. For eight or more sites the segment length is
``ceil(sqrt(nsites))``: only segment endpoints are retained initially, and
interior boundaries are recomputed when their segment is visited. C++ owner
buffers are released along with Python references. This retains
``O(sqrt(nsites))`` right blocks and requires at most one extra right pass.
It is an exact storage/recomputation adaptation of the reduced contractions
described above, without truncation. This bounds block count, not bytes;
individual frontier blocks and contraction temporaries can still be large.
This policy currently applies to parameter actions and stationarity audits,
not the optimization sweep or initialization. Closed-loop
stationarity audits contract the identity directly into conditional parameter
blocks: different invariant physical/fusion paths and future tie labels are
orthogonal, while virtual multiplicities inside each block remain coupled.
Structural scatter routes retain zero-valued directions. The audit applies the
existing support and metric eigenvalue thresholds component by component,
without building pair Hamiltonians or a global whitening matrix. This changes
the evaluation of the reduced norm, not the variational ansatz or convergence
criterion.

``examples/qchem/benchmark_su2_streamed_audit.py`` compares frozen six-orbital
closed D=2 states against an explicitly supplied solver snapshot. A one-thread
run reduced the warmed stationarity audit from 0.629 to 0.195 seconds and
worker peak RSS from 311.7 to 265.5 MiB, with matching energies, actions, and
residuals. These are small-reference audit measurements, not molecular sweep
or convergence timings.

An eight-orbital checkpoint comparison against the streamed full-right-stack
implementation retained 98.5 versus 58.1 MiB of right-boundary numerical data
at local actions (eight versus five blocks). Energies, actions and residuals
matched. Worker peak RSS did not improve (684.7 versus 689.2 MiB); the warmed
audit took 2.67 versus 2.79 seconds. These single-run measurements establish
reduced retained payload, not lower overall peak memory or a speedup.

Checkpoint restoration skips the disposable initial-state expectation value.
It installs the saved tensors and restores the saved energy diagnostic; call
``expectation()`` for a fresh value. If energy is missing from the checkpoint,
restoration evaluates it once after installing the saved state. The
``benchmark_su2_checkpoint_restore.py`` eight-orbital reference measured cold
restoration at 13.38 versus 0.097 seconds, with peak RSS through restoration
393.5 versus 232.2 MiB. Saved tensors matched exactly and a subsequent fresh
energy evaluation matched. This improves resume startup, not sweep runtime
or peak memory during a fresh contraction.

The Python streaming local Hamiltonian action retains up to 8 MiB of exact
boundary/operator factors per compiled local operator. These factors depend
on fixed environments, not the Davidson trial vector, so reusing them avoids
repeating their contraction on every matrix-vector product. Selection favors
route-use count per byte, so small frequently used factors take priority.
Once the budget is full, other factors remain streamed without cache eviction or numerical
compression. This is an exact data-flow optimization of the existing reduced
operator, not a change to the local eigensolver. The budget covers cached
factor payload only, not all local-operator storage or contraction workspaces.
Channel-resolved local actions also omit the unused transition-plan build;
their actual block-sparse operator environments supply the contraction.

``examples/qchem/benchmark_su2_factor_cache.py`` compares one complete projected
left/right cycle with factor reuse enabled and disabled, using identical
six-orbital closed D=2 initial tensors and Davidson settings. One unprofiled,
single-thread run took 57.97 versus 35.76 seconds (1.62x); whole-worker peak RSS
was 613.3 versus 554.3 MiB. Final tensors, energy and fresh tangent residual
matched. The residual was 0.596, so this is a throughput reference, not a
converged molecular benchmark. Both workers use ``reuse_environments=False``
to isolate factor reuse in the existing rebuild path.

Projected sweeps now select moving reduced environments by default
(``reuse_environments=None``). They retain and advance environments across
local updates and reuse a valid opposite stack at the direction change.
``reuse_environments=False`` remains an explicit rebuild reference for MPO
states. This selects the existing exact reduced sweep machinery; it does not
change the LETTA ansatz, metric thresholds, or convergence criteria.
Use ``benchmark_su2_factor_cache.py --comparison environments`` to compare
the default against rebuilding with identical factor-cache and solver settings.
An unprofiled six-orbital run measured 35.64 versus 2.78 seconds per first
complete cycle, and peak RSS 569.5 versus 356.1 MiB. Energy and fresh residual
comparisons pass. The strict coefficient comparison fails on three sites,
with maximum absolute difference 4.4e-9; the benchmark records this failure
and exits unsuccessfully. An independent reduced-space overlap check gives
absolute deviation of normalized fidelity from one of 9.2e-14. This supports
physical equivalence to contraction roundoff, not bitwise parameter identity.
Continuing the moving-environment state through cycle 13 reduced its fresh
residual from 0.596 to 0.00866 in 12.8 additional seconds; it remains
unconverged. These results concern the small reference, not pyrene scaling.

Bounded molecular boundary diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/qchem/benchmark_su2_boundary_prefix.py`` loads a saved real state
and advances only a requested prefix of right boundaries. It guards each
output arena before allocation; the guard is not a total process RAM limit.
``--direct --compaction-mib 0`` explicitly probes the existing direct
complementary boundary kernel using the checkpoint's own runtime. This is
an experimental diagnostic, not an enabled solver backend.

For the pyrene CAS(16,16) closed D=2 checkpoint, cuts 14 through 11 took
56.37 seconds on the generic path and 0.69 seconds on the direct candidate,
with process peaks 857 and 742 MiB. Exact-zero channel removal was enabled
earlier in the candidate. Strict block comparisons passed cuts 14, 13 and
12, but failed two blocks at cut 11 (global relative Frobenius difference
7.4e-15). The failed gate is preserved; the candidate is not enabled by default.
A separate guarded probe stopped before cut 10: its predicted raw arena was
711 MiB for the generic path and 960 MiB for the direct candidate. Output
topology needs further pruning before allocation, and the entrywise discrepancy
needs diagnosis. These partial-boundary results do not establish full-sweep
speed, bounded total memory, or molecular convergence.

The direct route packer now removes spin-forbidden output sectors before
constructing the numerical arena. For doubled spins ``j_b``, ``j_k`` and
operator rank ``k``, retained outputs require
``abs(j_b-j_k) <= k <= j_b+j_k`` and even ``j_b+j_k+k``. This is an exact
SU(2) selection rule, with no numerical cutoff or variational truncation.
The pyrene cut-10 planned arena consequently drops from 960 to 711 MiB;
the guard still prevents allocating it. Four completed boundaries match the
unpruned candidate at the existing ``rtol=1e-11, atol=1e-12`` gate, though
not bitwise (maximum difference 1.5e-12). Both-direction reduced-reference
tests verify the output selection and numerical contractions.

For the two scalar entries failing the generic/direct comparison at cut 11,
70-digit sums of the same rounded Python reduced cores and each path's own
rounded parent give local errors between 5.5e-14 and 9.1e-13. Their absolute
term sums are roughly 5,164 and 21,313, versus outputs 0.242 and 1.204.
Most of the path difference is therefore inherited through the parent
boundary and amplified by cancellation. This is a local precision diagnostic,
not a high-precision recomputation of the complete wavefunction or recoupling
algebra. The two molecular entrywise failures remain recorded, and the
candidate direct boundary path remains experimental for closed LETTA.

``examples/qchem/benchmark_su2_boundary_storage.py`` measures exact tiled
storage and independent batched actions on saved reduced boundary NPZ files.
For the same pyrene cut 11, packing 18,066 component matrices into one
block-diagonal 2-by-2 BSR matrix reduces matrix/index arrays from 62.42 to
31.55 MiB. Exact reconstruction passes, and four-column actions have zero
observed difference. Eight alternating-order repetitions give warm median
times of 21.55 ms dense and 12.50 ms batched sparse (1.72 times faster).
Packing alone costs 0.933 seconds, requiring about 103 repeated actions to
amortize at these timings. Separate sparse calls per component were slower
than dense calls in the preceding diagnostic. Thus batching and direct sparse
construction matter; post-conversion alone does not establish a speedup.
The script saves input provenance, timings and PNG/PDF diagnostics. Array
storage excludes Python metadata, construction temporaries, and other solver
data; this is not a process peak or full-sweep measurement. The active solver
still uses dense boundary arenas and this prototype is not integrated.

The active generic boundary planner now reuses buffers by source-array identity
before dtype conversion; its real/complex dispatch checks no longer construct
throwaway real buffers. This prevents repeated copies when real tensors occupy
complex storage. In the saved pyrene four-boundary prefix, conversion-copy
traffic falls from 4,196,288 to 250,816 bytes, but this is a minor part of the
workload: elapsed times remain 56.75 versus 56.95 seconds, with process peaks
878 versus 855 MiB in single runs. All four saved boundaries are bitwise
identical before and after the change. These measurements do not demonstrate a
speedup or resolve the dense-boundary scaling problem.

Profiling the active generic path identified reduced-core construction as
the dominant cost (53.76 of 54.55 seconds inside three profiled boundary
updates). Both contraction directions now resolve local operator components
for the physical-sector pair before enumerating virtual-channel routes.
Terms with exactly zero physical support are skipped, and supported components
are reused within the route loop. This changes neither recoupling coefficients
nor numerical tolerances. The unprofiled four-boundary pyrene prefix drops
from 56.95 to 19.25 seconds including saves (2.96 times faster), with all four
saved boundaries bitwise identical. Peak process memory remains approximately
unchanged, 855 versus 859 MiB; dense output storage is still unresolved.
The measurements cover a boundary prefix, not a complete optimization sweep.

The reduced-core cache now keys the four virtual legs by their spin irreps,
while retaining the complete physical sectors, direction, dual convention,
and Hamiltonian plan identity. Virtual particle counts constrain boundary
routes but do not enter these spin recoupling coefficients; local physical
charges do enter the fermionic operator blocks and are not merged. For the
same pyrene prefix, this reduces elapsed time further from 19.25 to 6.13
seconds and process peak from 859 to 728 MiB in single runs. All four
boundaries remain bitwise identical. Tests compare independently constructed
blocks for shifted virtual charges and verify physical charges remain distinct.
This removes duplicate reduced-core work and storage; it does not compress the
dense boundary output arenas or establish full-sweep convergence.

The generic planner also skips exactly zero parent channels and site blocks
before forming numerical routes. Support is recomputed on every update, so
a block that becomes nonzero during optimization is reintroduced. There is
no magnitude threshold. In the saved pyrene state this removes 1.25 MiB of
zero channels at cut 12 and avoids creating the 30.08 MiB of zero output
previously allocated and then compacted at cut 11. The next cut-10 planned
arena drops from 711.22 to 496.63 MiB (30.2 percent); a 128 MiB guard still
stops before allocating it. Across four completed boundaries, omitted arrays
are exactly zero and retained arrays pass ``rtol=1e-11, atol=1e-12``. The
largest retained-entry difference is 2.33e-10; bitwise equality fails at cuts
12 and 11. This reduces wasted allocation but is not a sparse representation
of the surviving blocks. Raw factor-route installation also clears the previous
reduced-contextual mode flag; otherwise changing boundary support can leave
diagonal evaluation using stale contextual storage during a moving sweep.

A subsequent diagnostic propagates boolean support through the same reduced
routes without a dense numerical child. At cut 11 it contains every saved
nonzero; at cut 10 it predicts 12,416,752 supported entries out of 65,094,256.
These are supports of the current saved tensors, not state-independent masks.
A Python/SciPy sparse numerical prototype constructs both cuts directly with
an 8 MiB intermediate-product cache. All 7,794 cut-11 blocks pass the existing
``rtol=1e-11, atol=1e-12`` reference check (maximum absolute difference 7.45e-9).
Cut 10 is constructed without its dense 496.63 MiB child arena, and conversion
to 2-by-2 BSR requires 113.61 MiB of matrix/index arrays. That later cut has no
independent numerical reference yet. Sparse construction takes 9.95 and 14.18
seconds, excluding final BSR conversion and reference checks; the whole
diagnostic peaks at 974 MiB while retaining dense parents and sparse workspaces.
This prototype is not integrated into boundary recursion or local actions and
does not establish a full-sweep speedup or convergence. Its next requirement
is efficient sparse construction and consumption without repeated conversion.

Further cut-10 checks reject that SciPy sparse arithmetic: 467 of 10,723
blocks fail against independent dense block contractions, and 289 fail against
the active compiled contraction evaluated one output block at a time, at the
unchanged tolerance. The latter comparison has relative Frobenius difference
8.15e-13; that small aggregate error does not override the entrywise failures.
The sparse arithmetic remains experimental.

An alternative prototype streams the existing compiled contraction one output
block at a time and stores its results as BSR arrays. This avoids the full
dense output arena while retaining the existing contraction kernel. Cuts 11
and 10 take 2.70 and 4.91 seconds including exact storage round-trip checks;
their largest dense output temporaries are 24 and 234.4 KiB. Stored matrix/index
arrays occupy 31.61 and 113.61 MiB, and the diagnostic process peaks at 828 MiB.
All 7,794 cut-11 blocks pass the saved full-boundary reference gate (maximum
difference 7.28e-12, not bitwise identical). Cut 10 uses the compiled block
reference directly and storage round-trips exactly; it is not independently
compared to a full-arena cut-10 calculation. Sparse snapshots are saved outside
the repository. Recursion and local actions do not yet consume this representation,
so it is not an active solver path or a full-sweep memory guarantee.

A follow-on diagnostic consumes BSR parents by gathering only the parent
blocks needed for one compiled output block. Starting from the exact saved
cut-11 values, all cut-10 components match the previous compiled reference
exactly. It then completes cut 9 without a full dense parent or child arena:
the child has 1,140,751,360 dense-equivalent bytes versus 204,539,808 stored
matrix/index bytes. Largest real parent gatherings are 821,632 and 5,820,032
bytes at cuts 10 and 9, within a 16 MiB per-arena guard. The retained sparse
child has a separate 256 MiB guard; neither guard is a whole-process limit.
Planning and construction take 17.57 and 33.59 seconds, and the process peaks
near 1.01 GiB including reference validation. This is still Python diagnostic
orchestration, not an enabled solver backend; local-action consumption and
efficient orchestration remain outstanding. An initial run using the slightly
different streamed cut-11 snapshot fails 13 component comparisons at cut 10;
that failure is preserved separately rather than attributed to compression.

``pyqed.mps.nonabelian.sparse_boundary.SparseBoundary`` now provides this
experimental recursion as a reusable component. The dense and sparse executors
share the same reduced route planner. Sparse parents have no dense ``data``
arena; access must gather specific blocks. An 8 MiB decode cache reuses those
blocks, and scratch slots beyond the physical chain are released on success
or failure. Metadata does not retain a reference cycle back to the boundary.
Both directions are tested against independent reduced boundary contractions,
including zero-block reactivation, budgets, and preservation of existing
physical boundary buffers. The component currently supports real Hamiltonian
boundaries; complex tensors and identity-metric-specific transport are rejected.

The one-site reduced adjoint can now consume these sparse boundaries through
lazy sector/channel views, without reconstructing a full dense boundary arena.
Its weighted-right and ket-absorption caches each retain at most 8 MiB by
default (``cache_bytes`` controls each cache). Each boundary separately retains
up to its configured decode budget; lowering that budget immediately evicts
excess cached blocks. These limits exclude output arrays, active contraction
temporaries, sparse storage, and Python metadata. Tests compare all one-site
actions of a small closed SU(2) model against dense reduced boundaries with
zero, tiny, and default intermediate-cache budgets. This enables sparse local
action consumption but does not yet replace the optimizer's dense boundary
construction or implement sparse two-site actions.

``state.reduced_parameter_actions(sparse_hamiltonian=True)`` now connects
sparse checkpoint/recomputation directly to all one-site Hamiltonian actions.
Only endpoint boundaries are initialized through the dense chain; interior
Hamiltonian boundaries are constructed and consumed as sparse blocks. This
explicit experimental option supports real states and retains the existing
dense norm transport. It uses the component's 16 MiB gathering/output and
256 MiB per-boundary storage guards, raising on overflow rather than silently
falling back to a dense stack. Checkpoint count is O(sqrt(n)) for eight or more
sites; this is still not a total-memory guarantee. Full optimization and
stationarity audits retain their existing default dispatch. Small closed-chain
tests cover all parameter actions and incomplete checkpoint segments.

Sparse channel membership checks inspect metadata only. Decoding is deferred
until the corresponding weighted-right or ket-absorption cache misses, so a
cached intermediate does not trigger another sparse-to-dense conversion.
``examples/qchem/benchmark_su2_sparse_actions.py`` compares complete frozen-state
parameter actions in separate processes and emits arrays, JSON, and PNG/PDF.
On the saved six-orbital closed-state diagnostic, one cold action evaluation
takes 0.912 s dense and 1.461 s sparse, with peak RSS 264.9 and 263.9 MiB.
All actions pass the entrywise comparison (rtol 1e-11, atol 1e-10); the largest
absolute Hamiltonian-action difference is 1.50e-11 and norm actions agree
exactly. This small case shows sparse construction overhead without a useful
memory benefit, not a speedup. Larger molecular action performance and complete
optimization remain unverified; the sparse option is not the default.

Boundary construction now batches output blocks while bounding both the union
of gathered parent arrays and the output arena by the configured arena budget.
Each output retains its original route order. A focused test forces multiple
small batches and checks bitwise equality with a larger batch, as well as both
arena limits. The six-orbital action rerun takes 1.20 s sparse versus 0.94 s
dense, with the same action errors as above; these are single-run measurements.
Batching reduces repeated metadata construction and parent gathering but does
not remove BSR conversion overhead or make sparse storage beneficial for every
small system.
The six-boundary pyrene prefix takes 37.05 s after batching versus the earlier
43.84 s. Cut 9 uses 152 batches for 13,740 output blocks and retains the same
195.06 MiB of sparse arrays. Peak process RSS rises from 1.17 to 1.33 GiB;
batching trades additional live workspace for fewer calls and repeated gathers.
All individual gathering/output arenas remain below 16 MiB. These molecular
timings do not validate a full action, sweep, or converged energy, and unchanged
storage sizes are not an entrywise molecular accuracy check.

A subsequent cut-10 entrywise check uses the exact saved dense cut-11 parent
values and the earlier compiled cut-10 reference. All 10,723 output blocks
pass rtol 1e-11 and atol 1e-12; the maximum absolute difference is 1.42e-14.
This validates that boundary update, not accumulated full-chain accuracy.
Completed batch arenas are now released from both the compiled owner and
Python references before the next batch is gathered. The same molecular
entrywise check passes after this lifetime change.
The six-boundary rerun peaks at 1.21 GiB versus 1.33 GiB before early release,
with unchanged sparse payload sizes. It takes 39.78 s while other validation
processes are active; this timing is not a controlled speed comparison.

Extending the guarded pyrene traversal completes cut 8, which retains
264,717,920 bytes (252.46 MiB) versus 1,571,713,536 bytes dense-equivalent
storage. Cut 7 then exceeds the 256 MiB sparse-array storage guard. The run
stops after 97.01 s with peak RSS 1.45 GiB; it does not complete all boundaries
or compute molecular parameter actions. Fully resident sparse blocks therefore
remain a scaling limit. The benchmark now records the failed update's partial
statistics and cut, and memory errors report required versus allowed bytes.

``SparseBoundary.advance(..., spill_directory=existing_scratch_directory)``
now writes child BSR arrays into one temporary binary file per boundary.
Only array offsets, shapes, and dtypes stay resident; channel access reads one sparse block
before decoding it through the existing bounded cache. No numerical truncation
is introduced. Files use raw NumPy array storage without pickle and are removed when
their boundaries are released; failed writes explicitly close the file.
``disk_bytes`` caps each file at 8 GiB by default. In this mode,
``storage_bytes`` limits an individual sparse block instead of the accumulated
child payload; arena and decode budgets retain their meanings. File metadata,
filesystem caching, and other solver arrays remain outside these limits.
The public experimental action option accepts the same scratch directory:
``state.reduced_parameter_actions(sparse_hamiltonian=True, spill_directory=...)``.
The boundary benchmark exposes ``--spill-directory`` and distinguishes disk
payload from resident sparse-array storage. Tests check recursive values,
complete parameter actions, file lifetimes, and the disk-budget failure path.
The temporary format stores array descriptors in memory and uses NumPy
``tofile``/``fromfile`` rather than repeatedly parsing per-array headers. The
exact byte requirement is checked before writing each block. These files are
internal scratch storage, not portable or persistent checkpoints.
The disk-backed pyrene run completes eight right-boundary updates through cut
7, passing the previous in-memory stopping point. Cut 7 contains 374,912,200
bytes of sparse arrays in a 390,799,816-byte temporary file, versus
2,126,163,200 bytes dense-equivalent. The run takes 157.83 s and peaks at
0.996 GiB RSS, with every gather/output arena below 16 MiB. This is a memory
scaling improvement with additional I/O cost, not a speedup or a completed
molecular action/sweep validation. Full-chain traversal remains to be tested.

A full right-traversal attempt with the initial disk format completes cut 6
but stops during cut 5: one output needs 16,889,472 bytes of gathered parents,
above the 16,777,216-byte arena cap, although its output is only 93,312 bytes.
Thus disk storage removes the retained-payload barrier but does not eliminate
the need to split parent contributions to an output. The guarded attempt takes
254.27 s and peaks at 1.16 GiB; full traversal remains incomplete.
The raw temporary format subsequently passes 193 tests. On the six-orbital
profile, identical read/write counts take 0.424/0.162 s cumulative time versus
0.700/0.252 s with array headers. Profiling and concurrent work affect timings;
these storage-function measurements are not an end-to-end molecular speedup.

Oversized parent gatherings are now split into contiguous reduced-route chunks.
The existing compiled ``advance_boundary`` accumulation option is exposed
through the Cython wrapper: the first chunk initializes the output arena and
later chunks add directly into it. Each output retains its original route
order, avoiding a Python sum of separately rounded partial boundaries. Parent
arenas are released between chunks while the bounded output stays resident.
An individual parent block or output block larger than the arena cap still
raises; spatial tiling of an individual matrix is not implemented. Focused
left/right tests force route splitting and require bitwise agreement with
unsplit accumulation. The rebuilt compiled extension passes 195 tests.
An identical-parent molecular cut-10 check with a 256 KiB arena forces
4,180 gatherings across 3,075 output batches. All 10,723 blocks pass the
saved compiled reference at rtol 1e-11 and atol 1e-12; maximum absolute
difference is 1.42e-14. The maximum parent gathering is 261,888 bytes and
maximum output arena is 262,144 bytes. This verifies split accumulation
locally, without certifying the accumulated full-chain state.

With route splitting and raw disk storage, the pyrene checkpoint now completes
all 15 right-boundary updates through cut 0. Elapsed time is 287.34 s and peak
RSS is 1.11 GiB; all individual gathering/output arenas stay within 16 MiB.
The largest boundary payload is 390,679,880 bytes on disk, versus
2,196,912,128 bytes dense-equivalent at cut 6. Cuts 5, 4, 3, and 2 use more
gatherings than output batches, confirming that split accumulation is exercised
in the molecular traversal. This verifies construction under the memory guards;
it does not validate all molecular local actions, left traversal, optimization,
or energy convergence.

For real periodic-virtual-bond states with explicit reduced MPOs,
``state.local_tangent_residual(sparse_hamiltonian=True, spill_directory=...)``
now uses this sparse Hamiltonian action path in the stationarity audit.
The scratch directory is optional; norm transport and conditional metrics
remain dense. The option changes storage, not metric cutoffs, whitening, or
the retained-support stationarity criterion. Tests compare all site residuals
with the dense audit and verify that coefficients are unchanged and temporary
files are released. This does not yet change optimization sweep storage.
The focused combined suite passes 197 tests after adding the audit option.

``examples/qchem/audit_su2_sparse_actions.py`` runs all frozen-state Hamiltonian
and norm actions from a checkpoint, saving each action, per-site timings,
boundary progress, checkpoint hash, and diagnostic figures. It reports the
spread of site Rayleigh quotients as an internal consistency diagnostic, not
an independent energy reference or a convergence certificate. The six-orbital
checkpoint completes in 1.62 s with a 5.56e-13 Eh quotient spread. Large-system
action validation is separate from the completed right-boundary traversal.

The local ket-absorption cache now spans a whole frozen action rather than
being reset for each bra/ket block pair. Its key includes the bra left sector,
complete ket block sector key, and incoming channel; no cached product survives
a local action or tensor update. The 8 MiB budget is unchanged. In the
six-orbital audit, 8,487 product requests require 2,756 actual products instead
of 5,035, reducing cumulative product-array construction from 32,546,880 to
14,552,000 bytes. Instrumented elapsed times are 1.64 and 1.61 s, so the reduced
work count should not be interpreted as a comparable end-to-end speedup.
The combined 197-test suite passes after this change.

Sparse parameter actions and stationarity audits accept ``checkpoint_stride``
to control Hamiltonian checkpoint spacing. The default remains the square-root
schedule; ``checkpoint_stride=1`` retains every right boundary and avoids
reconstruction. With ``spill_directory``, payloads are on disk but O(n)
boundary metadata remains resident. Construction decode caches are explicitly
cleared on retained disk parents, avoiding an extra dense cache per checkpoint.
Norm checkpoint spacing is unchanged. The action audit exposes this choice as
``--checkpoint-stride`` for measuring the speed/metadata-storage tradeoff.

Sparse decoding now converts BSR blocks through CSR before producing dense
arrays. In the installed SciPy version this avoids the direct BSR decoder's
COO coordinate expansion. All 18,066 components of the saved pyrene cut-11
boundary produce identical arrays. Three-repeat median decoding times are
0.598 s directly versus 0.243 s through CSR (2.46 times faster). This is a
decoding microbenchmark, not a full-action speedup; the 201-test suite passes
with the new decode path and unchanged numerical/cache budgets.
Each CSR component now decodes directly into its slice of the final dense
array. This removes the list of dense components and subsequent stacking
copy; mixed component dtypes retain NumPy promotion semantics. A fresh-process
synthetic test of three 2048-by-2048 float64 components (96 MiB output) measures
peak RSS of 399,147,008 bytes for stacking versus 293,126,144 bytes for direct
decoding. This isolates a transient allocation reduction, not a measured
full molecular peak-memory improvement. Focused tests require destination
buffer reuse and unchanged decoded values.

Boundary transport fills its packed parent arena one decoded channel at a
time instead of retaining a list of all decoded channels while allocating
the packed copy. Outside the bounded decode cache, only the current decoded
channel overlaps the packed gather buffer. A lifetime regression checks that
uncached decoded arrays are released before the next decode, while comparing
small-batch and large-batch contractions exactly. This removes up to one
gather's duplicate dense storage; whole-process molecular RSS and runtime
improvements have not yet been measured for this change.

Parameter assembly also reuses the active left and right channel arrays
across consecutive terms, preserving contraction order and existing decode
cache budgets. This avoids rereading an active channel that cannot fit in the
cache; it does not retain all channels. A four-repeat alternating six-orbital
site-3 comparison reduces array requests from 8064 to 5729, with bitwise-equal
probe actions. With decoding uncached, median assembly time is 0.3512 versus
0.2933 s. With the normal 8 MiB cache reset before each measurement, medians
are 0.1637 versus 0.1639 s: no speedup is established in that cache-resident
case. On the frozen pyrene site-10 checkpoint, requests decrease from 719,894
to 429,800, but profiled assembly only changes from 78.14 to 76.59 s in a
single comparison. Operator values and independently verified energy and
residual are bitwise unchanged. Whole-process peak RSS increases from
1,263,108,096 to 1,291,026,432 bytes, so this comparison establishes no molecular
memory reduction or substantial end-to-end speedup. The candidate remains
rejected with independent residual 1.58e-8; reuse does not fix convergence.
The operator benchmark exposes
``--profile-assembly`` to identify decoding and contraction costs separately;
its profiled timings include instrumentation overhead.

The frozen site-10 action diagnostic separates Hamiltonian and norm errors.
The raw Hamiltonian difference is 7.14e-11, amplified to 1.78e-8 in retained
metric coordinates; the energy-weighted norm difference is 1.88e-9 and their
combined difference is 1.59e-8. The largest transform entry is 861.5.
The appended incumbent direction contributes only 3.03e-14 to the Hamiltonian
discrepancy, so removing that direction would not resolve this mismatch.
The benchmark saves the raw and projected action differences in
``action_discrepancy.npz`` and generates a diagnostic PNG/PDF. This identifies
conditioning-amplified contraction disagreement, not which contraction is
more accurate; independent verification still rejects the candidate.
A 40-decimal-digit streamed product of the saved site-10 reduced-parameter
operator differs from the ordinary stored action by 2.58e-12 in raw norm,
while the independent action differs from that reference by 7.02e-11.
Thus ordinary stored matrix-vector rounding explains only a small part of
the 7.14e-11 discrepancy. This reference retains the original float64
operator coefficients and cannot certify their assembly accuracy.
The compiled route accumulator now accepts an optional persistent correction
array for compensated additions across routes and channel calls. A cancellation
regression recovers the two unit contributions in the sequence
``[1e16, 1, 1, -1e16]``. The production builder does not yet enable this
experimental option by default. ``iter_parameter_blocks(compensated=True)`` and the benchmark's
``--compensated`` option enable it for real scalar-physical compiled terms,
with one additional correction block bounded by ``block_bytes``. Unsupported
terms raise explicitly, and the benchmark forbids combining this option with
a saved-operator snapshot, since that would bypass the requested assembly.
On the exact frozen pyrene site-10 checkpoint, compensation reduces the
independently checked residual from 1.58e-8 to 2.06e-9, but still fails the
1e-10 tolerance. The projected Hamiltonian action discrepancy decreases from
1.78e-8 to 4.38e-10; the energy-weighted norm discrepancy is now 9.20e-10.
The stored residual is 7.07e-10. Profiled assembly increases from 76.59 to
84.85 seconds in this single comparison. These results support improved
Hamiltonian accumulation accuracy, not local or molecular convergence.
An identity-only diagnostic using the same compensated candidate compares
norm actions in the same retained metric coordinates. Ordinary route assembly,
its symmetrized action, compensated route assembly, its symmetrized action,
and the existing norm components differ from the independent action by
7.46e-12, 1.06e-11, 3.38e-11, 3.28e-11, and 3.26e-11, respectively.
Compensation reduces raw norm-action disagreement but not its transformed
value; symmetrization alone does not explain the discrepancy. These are
cross-contraction comparisons, not high-precision norm references, and do
not justify replacing the component metric or weakening verification.
A subsequent 40-digit local identity contraction, including the route lift
and adjoint in high precision, resolves this comparison for the frozen
candidate. Projected errors are 6.07e-13 for the independent action, 6.87e-12
for ordinary route assembly, 3.33e-11 for compensated route assembly, and
3.20e-11 for the stored components. Differences are formed before rounding
the high-precision reference. Boundary and recoupling coefficients remain
fixed float64 inputs, so this reference diagnoses local contraction accuracy,
not preceding environment transport. It supports testing the factored norm
action inside the local solve while retaining component-based coordinates.
``solve_parameter_operator(norm_matvec=...)`` now permits that separation:
component blocks choose coordinates, while the supplied action determines
the generalized solve, incumbent energy, candidate norm, and residual.
On the frozen compensated site-10 operator, the residual evaluated with
stored H and factored N improves from 1.62e-9 to 5.48e-10. Local solve plus
norm check increases from 0.30 to 1.61 seconds in this comparison. Neither
meets 1e-10. A subsequent independent Hamiltonian check gives residual
3.21e-10, versus 2.06e-9 with compensation alone and 1.58e-8 originally.
The projected Hamiltonian-action disagreement is 2.31e-10; the norm action
is shared and agrees exactly. The candidate remains rejected at 1e-10.
This option remains experimental and is not enabled in
production sweeps by default.
Both ``sweep_parameter_blocks`` and ``run_parameter_blocks`` accept
``compensated=True, factored_norm=True`` for full-cycle experiments.
Overrelaxation uses the same selected norm action as the local solve.
Per-update records retain both settings, and independent acceptance plus
the final whole-state stationarity audit remain mandatory for certification.
Small closed-SU(2) cycle tests cover successful final audits and deliberately
failed audits with both options enabled; this does not establish molecular
convergence or scaling beyond the tested examples.
The benchmark exposes this trial through ``--factored-norm``. For its saved
candidate, the projected action of ``(H-H†)/2`` has norm 1.38e-10, smaller
than the 5.48e-10 residual. Stored-operator Hermiticity loss alone therefore
does not explain the full residual; no symmetrization is applied to conceal
the remaining discrepancy.
A 40-digit product of the compensated stored operator at this candidate
gives projected differences of 1.12e-10 for ordinary stored multiplication
and 3.40e-10 for the independently contracted Hamiltonian action. Combining
the high-precision stored-H product with the factored float64 norm gives
residual 6.58e-10. Thus improving stored matrix-vector multiplication alone
does not close the remaining gap. This remains a fixed-coefficient local
reference, not a high-precision environment or molecular calculation.
Five successive factored-norm solves, each initialized from the previous
candidate and rebuilding its incumbent-augmented coordinates, produce fresh
stored-H residuals from 5.48e-10 to 1.67e-9. None meets 1e-10. The initial
solve stops after its cached Ritz residual passes, then correctly fails the
fresh check; subsequent restarts can also exhaust correction directions.
Automatic external retries are therefore not enabled by this experiment.

A subsequent full pyrene LR/RL continuation with compensated assembly and
factored norm actions lowers the energy from -28.281989155072214 to
-28.31971664527424 Eh. Thirty of 32 updates pass strict independent local
checks; none uses descent-only acceptance. Site 13 is rejected in both
directions with residuals 2.85e-9 and 5.93e-9. Site 10 passes in both directions.
The independent final-state maximum tangent residual decreases from
0.1016241 to 0.0666469 Eh, still far above the 1e-8 target. The cycle takes
3213.99 seconds, or 4275.67 seconds including the final audit, with peak RSS
1,709,047,808 bytes (1.592 GiB). Source hashes remain unchanged during the run.
This is a continuation from a newer state, not a controlled timing comparison
against the preceding cycle; neither convergence nor a speedup is established.

Local actions now prepare contiguous real ket buffers when a complex-stored
block has exactly zero imaginary entries. Any nonzero imaginary entry retains
the complex path; no tolerance-based projection is used. This avoids needless
complex ket absorption for real molecular states, at the cost of temporary
real ket copies outside the cache budgets. The combined suite passes 202
tests, including preservation of a 1e-20 imaginary perturbation.
An alternating eight-repeat comparison of the same six-orbital site-3 action
gives median times of 66.5 ms with complex-stored kets and 62.8 ms with real
copies. This isolates a small local improvement; it is not a molecular speed
claim. Complete small-audit timings vary with other work and are not used as
the controlled comparison.

The first complete disk-backed pyrene action audit takes 2,897.83 s (48.30 min)
with 1,278,066,688 bytes peak RSS (1.19 GiB). All 16 Hamiltonian and norm actions
finish, but the site Rayleigh quotients span 1.035e-5 Eh. Sites 0--10 agree
within 2.33e-10 Eh around -27.731170272 Eh; the discrepancy appears at sites
11--15. Norm contractions span 1.83e-7. Thus the memory-bounded traversal
completes, while molecular contraction accuracy and convergence remain
unverified. This baseline predates the later local-cache, decode, and read
optimizations; it is not a timing of the latest implementation.

A norm-only diagnosis of that checkpoint finds strong cancellation in the
local reduced metric: at site 12 the absolute quadratic terms sum to
2.93e9 while their signed sum is approximately one. Compensating only the
final parameter dot product does not remove the norm discrepancy. Applying
one pass of the existing conditional coefficient QR to a separate copy reduces
the cross-site norm spread from 1.83e-7 to 2.11e-13. The resulting norm differs
from the original site-0 contraction by 3.35e-9, so this is evidence of improved
conditioning, not proof of exact floating-point state preservation. A second
pass gives no further benefit. Unconditional QR leaves a 1.18e-9 norm spread
and concentrates coefficients as large as 2.52e33 in the endpoint; it is not
selected for the next molecular Hamiltonian audit. No gauge change is enabled
in production on the strength of these norm-only results.

The subsequent full audit of the unguarded one-pass conditional QR checkpoint
completes in 1,085.68 s (18.09 min), with 1,291,321,344 bytes peak RSS (1.20 GiB).
All 16 site energy quotients agree within 1.15e-11 Eh around -27.731170762758 Eh.
This improves internal consistency, but the common energy is 4.91e-7 Eh below
the original site-0 quotient. The original state's inconsistent contractions
prevent treating that comparison as an independent preservation certificate.
Convergence and absolute molecular accuracy remain unverified. The timing also
includes the newer I/O/local-action code and checkpoint stride 1; it does not
isolate a gauge-only speedup.

Each local action now omits exactly zero ket blocks before constructing real
copies or traversing operator routes. No threshold is applied and the check is
repeated on every call, so blocks can reactivate. Bra output blocks are always
retained, including directions with zero current coefficients. Pyrene sites
4--7 contain respectively 19/55, 24/78, 24/105, and 35/136 zero ket blocks.
The 203-test suite covers zero-ket short-circuiting, reactivation, and nonzero
adjoints on zero bra blocks. The small six-orbital timing comparison shows no
meaningful speedup (64.2 versus 64.8 ms median); molecular benefit is unmeasured.

Disk reads now fetch one contiguous channel payload and reconstruct immutable
array views, instead of issuing separate reads for each BSR array. Views retain
their own payload across subsequent reads; a truncated payload raises an error.
All 18,066 components in the saved pyrene cut-11 boundary are bitwise identical
to separate-array reads. Alternating four-repeat medians are 0.696 s and
0.170 s (4.10 times faster for reading). The 33,149,976-byte disk payload and
decode budgets are unchanged. This does not measure a full molecular action
speedup or alter the numerical contraction.

An experimental ``pyqed.letta.parameter_operator`` path contracts channel-resolved
boundaries directly into tied one-site parameter blocks. This uses the reduced
Wigner--Eckart formulation of Weichselbaum, Phys. Rev. B 86, 245124 (2012),
https://doi.org/10.1103/PhysRevB.86.245124, with a local tied-coordinate
projection. It forms neither determinant states nor an adjacent-pair embedding.
It does not replace the default optimizer's local action path; the explicit
bidirectional-pass integration is described below.

``iter_parameter_blocks`` tiles output parameters and, for the NumPy path,
structural route pairs. The compiled kernel streams all route pairs for an
output block without allocating a pair matrix. Its route tables are built
once per output block and reused across channel terms, avoiding repeated
tile-level calls and index-table construction. Route-table storage is linear
in the number of routes for that block and remains outside the block-value
budget. ``TiedParameterOperator`` retains the blocks or writes their values to
a temporary file for repeated actions. Defaults bound each output block to
1 MiB, retained numerical arrays (including indices) to 32 MiB, and disk payload
to 8 GiB. Boundary decoding, Python metadata, vectors, and one decoded block
are outside these budgets; they are not whole-process memory limits. No
numerical entries are dropped. Closing the operator releases its temporary
file, and construction failures do the same.

On the six-orbital 84-parameter local case, six alternating warm measurements
compare the former tiled compiled calls with streamed compiled calls. Median
assembly time falls from 72.69 to 47.08 ms (35.2 percent less time); all 305
blocks agree at rtol=1e-11, atol=1e-12. This isolates the Python dispatch/data
flow change with the same compiled kernel. It is not a molecular or whole-sweep
speedup measurement; the machine was also running the molecular calculation.

``examples/qchem/benchmark_su2_parameter_operator.py`` builds only the two
boundaries required at a selected site, then compares the operator against
independent reduced adjoints for the incumbent and a random complex vector.
The six-orbital 84-parameter reference uses 305 blocks, 19,520 resident numerical
bytes and 78,080 disk bytes. Building takes 0.428 s; subsequent actions take
0.70--0.73 ms versus 36--39 ms for fresh adjoints. Maximum discrepancies are
below 1.7e-14. These are small local measurements, not molecular performance or
optimization convergence claims. The molecular action audit is reported below.
Focused regressions also compare random complex actions against the independent
adjacent-pair embedding formulation and check Hermiticity, in addition to the
direct-adjoint, forced-tiling, storage-budget, and cleanup checks.
``TiedParameterOperator.diagonal()`` streams an exact raw-coordinate diagonal,
including contributions from overlapping blocks, without assembling a full
matrix. The returned vector is not retained in its storage budget.

A small frozen local solve with 84 parameters and 71 retained norm directions
uses these block actions in the existing generalized Davidson implementation.
A constant diagonal converges in 21 iterations to a true local residual of
9.75e-12, with an energy difference of 1.78e-15 Eh from an independently
assembled adjacent-pair reference. Simply transforming the raw diagonal while
omitting off-diagonal contributions instead stalls at residual 2.09e-4 after
160 iterations; that preconditioner is not selected. This demonstrates one
small local solve, not an integrated molecular sweep. Dense whitening in this
reference experiment is not a solution to large-metric storage.

Real scalar-physical operator terms now use a compiled route accumulator that
gathers and scatters directly, avoiding NumPy route-pair value arrays. It checks
route bounds before entering the numerical loop. ``compiled=False`` retains
the NumPy construction as an independent reference; nonzero imaginary payloads
always use that path without tolerance-based projection. Tests force parameter
and route tiling, duplicated routes, and 1e-20 imaginary data. The compiled
small-checkpoint build takes 0.297 s and retains the validated local solve.
On pyrene site 6, compiled assembly plus block-by-block reference checking
takes 83.05 s versus 887.90 s for the preceding NumPy run. All 2,060 blocks
pass rtol 1e-11/atol 1e-12; the largest tolerance-scaled error is 0.0245.
Independent candidate verification retains the 8.00 mEh energy lowering and
1.83e-11 restricted-space residual. Whole-run time is 389.84 s with 1.31 GiB
peak RSS. These sequential measurements establish a local assembly improvement,
not a full-sweep speed ratio. The memory high-water mark is reached during
boundary construction, before local operator assembly.

Component whitening additionally supports ``sparse=True`` to assemble CSR
coordinates directly from each retained local block, without full-height zero
panels. On the balanced pyrene site-6 metric this stores 75,204 bytes instead of
a 45,420,480-byte dense-equivalent array, with 751 retained directions in 3,780
parameters. Small regressions require exact equality of dense and sparse
coordinates, including component-cache reuse; the local Davidson regression
also uses the sparse coordinates. Default optimizer whitening remains dense.

The molecular check exposed an accuracy defect in the previous diagonal
shortcut: off-diagonal terms below an absolute threshold can still be large
relative to weak retained directions. Dropping them gave 4--7 percent errors
in norm-orthogonality. Only exactly diagonal blocks now use that shortcut;
all other components use Hermitian eigendecomposition with the existing
eigenvalue support cutoff. Three complex pyrene probes then give relative
errors between 7.5e-12 and 1.2e-11. Retained rank changes from 748 to 751 because
support is determined from the full component eigenvalues. This corrects the
metric construction; it does not by itself establish molecular convergence.

``TiedParameterOperator.transformed_diagonal(T)`` computes the exact diagonal
of ``T.conj().T @ H @ T`` by streaming blocks and each sparse direction's local
support. It includes off-diagonal raw-coordinate couplings and never constructs
the full transformed matrix. ``--preconditioner diagonal`` selects this in the
local benchmark; the default remains constant preconditioning. Complex,
overlapping-index tests compare against explicit small reference matrices.
The six-orbital trial converges in 20 iterations with independent residual
8.73e-12, versus 21 with constant preconditioning; this small iteration change
does not establish a speedup after preconditioner construction costs.

The pyrene site-6 operator audit now completes for 3,780 parameters. Its
2,060 blocks require 1,165,376 resident index bytes and 41,228,800 disk bytes;
assembly takes 777.52 s after 193.40 s of boundary construction. The incumbent
action takes 29.9 ms versus 91.36 s for a direct contraction; a random complex
action takes 16.0 ms versus 327.98 s. Both pass elementwise rtol 1e-11 and
atol 1e-12. Maximum absolute discrepancies are respectively 2.24e-9 and
4.48e-4 in raw parameter coordinates; the latter probe has very large action
values and is not an absolute-energy accuracy measurement. Peak RSS for the
whole audit is 1.04 GiB. The build cost must be amortized across repeated
actions; the action-only speed ratios are not sweep speedups.

The benchmark's optional ``--solve`` uses sparse component whitening and
the existing Davidson solver, then independently recomputes the candidate's
Hamiltonian and norm adjoints before accepting its local energy and true
residual. It saves a candidate vector without updating the checkpoint.
``--solve-only`` omits repeated incumbent/random probes but retains this fresh
candidate check. Operator arrays and checkpoint/site provenance are saved
alongside the result to permit further diagnostics without another assembly.
``--operator-snapshot DIRECTORY`` streams these saved blocks after checking
the exact checkpoint hash, site, dimension, index layout, and payload length.
It still constructs the boundaries needed for independent candidate validation;
snapshot reuse avoids operator assembly, not that validation.
The six-orbital trial lowers the local active energy from -8.428765748128 to
-8.428776524745 Eh, with independent residual 9.78e-12 after 21 iterations.
The first molecular trial reached a projected residual of 3.85e-11 in 50
iterations, but raised the energy by 0.03601 Eh and was correctly rejected.
Its 751-direction metric cutoff excluded 4.817 percent of the incumbent's
physical norm squared. Thus a small residual within retained coordinates was
not evidence of a valid variational update. Lowering the raw cutoff to 1e-17
still loses 1.38e-7 of the norm, while random-probe norm-orthogonality error
reaches 8.44e-8. Diagonal equilibration recovers the incumbent much better but
also exposes inaccurate weak directions; it is not enabled as a fix.

The benchmark now augments sparse metric coordinates with the incumbent's
missing direction, using two metric reorthogonalization passes. This is a
single reduced tensor-parameter direction, not a determinant-space projection.
It preserves the variational reference without admitting every weak direction
discarded by the cutoff. The resulting residual still certifies only this
restricted local space, not all local directions or a full molecular sweep.
Remainders below a floating-point cancellation estimate are not normalized;
otherwise near-null roundoff can become a spurious extra direction.
The updated pyrene site-6 trial passes independent validation: active energy
decreases from -27.731170762760 to -27.739174263584 Eh and the checked
retained-space residual is 1.83e-11. It uses 752 directions (751 from the cutoff
plus the incumbent remainder) and 130,844 whitener bytes. Exact-diagonal
construction takes 0.211 s and Davidson takes 0.422 s for 25 iterations.
Reusing the frozen operator takes 0.081 s, while boundary rebuilding takes
191.28 s and independent candidate verification takes 105.79 s. Total time
is 298.26 s; peak process RSS is 1.41 GiB. The larger peak than the rejected
trial means small local-operator storage does not establish a whole-process
memory improvement. The benchmark saves the accepted candidate separately;
the original checkpoint and production sweep implementation remain unchanged.

``SU2LETTA.sweep_parameter_blocks(spill_directory=..., direction='lr')`` now
performs an explicit pass with these operators and sparse metric coordinates.
``direction='rl'`` traverses the same physical sites in reverse without
reversing their orbital or sector ordering. It consumes opposite-side
checkpoints while advancing the updated side once per site. Hamiltonian checkpoints can be disk-backed;
identity environments use the existing reduced representation. The default
independently checks every candidate with reduced Hamiltonian and norm
adjoints. Channel contributions in the independent adjoint use compensated
summation to reduce cancellation, keeping one correction array per current
bra block and transient block-sized update arrays. This does not change the
acceptance tolerance or remove roundoff inside individual contractions.
Six alternating warm measurements of the six-orbital independent adjoint
give median times 45.70 ms for ordinary accumulation and 53.27 ms for
compensated accumulation (16.6 percent overhead). The normal-case action
differs by at most 1.07e-14; a cancellation regression preserves contributions
lost by ordinary summation. Improved molecular acceptance has not yet been
demonstrated, so this is an accuracy tradeoff rather than a speedup claim.
Optional ``accept_verified_descent=True`` separates update acceptance from
local stationarity. It requires independent per-update verification and a
converged stored-operator root. If the independent residual misses tolerance,
the candidate can still be accepted when its independently contracted energy
decreases by more than max(1e-10 Eh, 64 times machine epsilon times the energy
scale). Diagnostics mark ``root_converged=False`` and
``accepted_by_descent=True``. This is an inexact variational descent policy,
not a local eigenproblem convergence certificate, and adds no global
convergence guarantee. The default remains strict. The final-state tangent
and energy convergence requirements remain unchanged in either mode.
Rejected sites remain unchanged, and exceptions restore the whole
pass's starting tensors and energy. This API returns per-site diagnostics;
it does not replace ``run`` or declare global convergence. Real Hamiltonian
boundaries remain required. Standalone passes rebuild opposite checkpoints.
``run_parameter_blocks(spill_directory=..., nsweeps=...)`` performs complete
LR/RL cycles and reuses the updated checkpoints across direction changes when
the checkpoint stride is 1. Each old checkpoint is consumed as its replacement
is constructed. Reuse checks state revision, operator identity and direction;
separate tensor updates invalidate it. Larger strides retain the independent
checkpoint/recomputation path. A four-site LR/RL regression needs nine
Hamiltonian boundary advances instead of twelve and agrees with a fresh full
contraction. Temporary checkpoint ownership ends with the cycle driver.

The cycle driver populates the solver's energy, history and convergence fields.
Energy stability alone is insufficient: before declaring convergence it runs
an independent final-state ``local_tangent_residual`` audit with the same
relative metric cutoff and no additional diagonal support filter. Convergence
certifies retained-metric one-site stationarity, not a global optimum. Tests
explicitly reject convergence when that audit fails despite stable local roots.
``verify=False`` skips independent per-update adjoints, using the assembled
operator's local checks instead; it does not disable the cycle driver's final
stationarity gate. Tests cover rejection by that gate in both modes. A small
two-cycle comparison takes 6.28 s with per-update verification and 5.61 s
without it, excluding final audits. Fresh energy and final tangent residual
agree between modes; the residual is 0.00743, so neither run converges.
This single small-case comparison does not justify changing the default or
claim molecular accuracy without independent validation.
Six six-orbital cycles take 11.65 s with final energy -8.428941718229 Eh, but
their independent final tangent residual remains 0.00554; they are correctly
reported as unconverged.
Extending the small case to 100 cached cycles takes 177.52 s and reaches
-8.429338408808 Eh, agreeing with a fresh energy contraction within 1.3e-13 Eh.
Its final tangent residual remains 6.08e-4, so accurate local roots and boundary
reuse have not resolved slow convergence across cycles.

The packed generalized Davidson solver also retains its measured Krylov
overlap matrix after metric orthogonalization, including restarts and the
final Ritz extraction. Treating that overlap as exactly identity can produce
incorrect roots when numerical orthogonalization drifts. A regression with
deliberately misnormalized Krylov vectors checks the generalized eigenvalue
and physical residual against a small dense reference. Storage is quadratic
in the bounded Davidson subspace size, not in the full parameter dimension.
This correction alone does not establish molecular convergence.
Metric orthogonalization now reapplies the metric after each subtraction
pass, preventing cancellation errors in cached metric images from being
normalized into new search directions. This adds two metric actions per
orthogonalization, without retaining another basis. The final residual is
recomputed on the actual returned vector and gates the convergence flag.
On the saved pyrene site-13 problem (252 parameters, 105 retained coordinates),
the constant-preconditioned solve changes from 234 iterations and a spurious
-25.9620 Eh root to 38 iterations and -28.142113076170 Eh, matching the small
reduced-coordinate reference energy. Its stored-operator residual is still
5.80e-10, above the 1e-10 target; the diagonal variant gives 3.86e-10.
Neither is reported converged. This local diagnostic does not measure a
full-sweep speedup or memory reduction, or certify molecular convergence.
Incremental projected matrices now measure both triangles from cached actions,
matching the full projection used at restarts before Hermitian averaging.
Previously, copying one triangle could bias the projected root when contraction
roundoff broke exact symmetry. With this correction, the same site-13 constant
variant takes 28 iterations with residual 3.85e-10; the diagonal variant takes
19 iterations with residual 3.64e-10. Both remain unconverged at 1e-10.
An explicit small reduced-coordinate reference using 40-digit projection
finds that the stored raw Hamiltonian skew is 1.82e-10 versus matrix norm
1.32e6. The coordinate transformation amplifies this to 3.13e-7, so increasing
projection precision alone does not remove the inherited asymmetry. This
reference does not replace the reduced iterative solver.
Reusing the previously verified pyrene site-6 operator snapshot with a
freshly contracted norm, the corrected solver reaches -27.739174263583 Eh
in 25 Davidson iterations with stored-operator retained residual 1.67e-11.
The full snapshot/norm/solve check takes 1.93 s, including 0.60 s for
Davidson. It agrees with the earlier independently contracted local root;
this check does not rebuild the Hamiltonian boundaries or verify previously
rejected sites of a later sweep.

Optional ``relaxation`` in ``[1, 2)`` applies normalized, phase-aligned
overrelaxation after solving each local problem; the default remains 1.
The proposal backs off toward the local root until its contracted energy
passes a decrease guard. With the default independent verification, both
the stationary root and the extrapolated proposal are checked with fresh
reduced adjoints. ``root_residual`` reports the former; ``residual`` reports
the actual update, which need not be stationary. Final-state convergence
still requires the independent tangent audit. This is an adaptation inspired
by Ivan V. Oseledets, Maxim V. Rakhuba and André Uschmajew, "Local convergence
of alternating low-rank optimization methods with overrelaxation",
https://doi.org/10.1002/nla.2459 (arXiv:2111.14758). The closed-LETTA
Rayleigh quotient, normalization and acceptance guard differ from the
reference setting; its convergence and optimal-relaxation guarantees do
not carry over.

On the same six-orbital checkpoint, 100 cycles with relaxation 1.8 take
196.32 s and finish at -8.429210198440 Eh, with a fresh tangent residual
0.00410. Ordinary sweeps took 177.52 s and reached a lower energy and
smaller residual (6.08e-4). Neither converged. This trial does not support
enabling overrelaxation by default; timings were not collected on an idle
machine. Fresh energy agrees within 1.3e-12 Eh for the relaxed run.

Sparse boundary planning now reads stored numerical-support flags instead
of decoding each BSR block merely to test for zero. Disk-backed blocks
record support when written. This avoids those planning reads and dense
temporaries without introducing a numerical cutoff. Explicit sparse zeros
are excluded; duplicate entries that cancel can retain an unnecessary
route. This does not bound total RSS: norm checkpoints, route metadata,
materialized tensors and active contraction buffers remain separate owners.
For the saved pyrene D=2 checkpoint, array payloads total 255,360 bytes
for variational parameters, 84,459,904 bytes for materialized reduced
tensors and 1,842,304 bytes for structural route indices. Each complete
left or right norm-boundary set contains 26,220,808 bytes of array payload.
These counts exclude Python metadata, packed copies, Hamiltonian boundaries
and allocator overhead; they are not process RSS. Each parameter-sweep
report now records ``boundary_peaks`` for Hamiltonian route count, packed
route-pool bytes, output-block count and contraction gather/output bytes.
The peaks cover advances performed so far in that pass, including initial
opposite-boundary construction; they are component maxima, not a sum or
a process-wide memory bound.
The disk-backed sweep now clears each Hamiltonian core's recoupled-block
cache after route planning, before packing/transport, and clears remaining core caches on pass
completion or failure. Local assembly and its independent checks share the
cache until transport begins. Each original coefficient list is released
immediately after its packed pool is built. This prevents cache accumulation across
orbitals but may require recomputation on subsequent visits. Boundary
diagnostics also include ``reduced_cache_array_bytes`` before release;
this excludes dictionary and ndarray-object overhead. Cache-lifetime tests
check successful and interrupted sweeps and fresh contracted energy.
A two-cycle six-orbital comparison finishes at the same energy with and
without cache release (-8.428847249732 Eh). Retention leaves 695 cache
entries and 501,464 array bytes; release leaves none. Observed peak RSS
is 300.03 versus 288.67 MiB, with times 5.25 versus 5.58 s. These are
single-process samples on a machine also running the molecular benchmark,
not a statistical timing claim or evidence for the pyrene memory reduction.

Sparse final-state audits also release recoupling caches after route planning
and local actions, including on exceptions. A six-orbital audit leaves zero
cache entries instead of 695 (501,464 array bytes), with the identical tangent
residual 0.008660269427905357. Separate process peaks were 274,694,144 bytes
with retention and 286,031,872 bytes with release; this experiment establishes
cleanup and numerical agreement, not a reduction in peak RSS.

Explicit-MPO checkpoints exclude disposable contraction caches during deep
copy, rather than copying their payload and clearing the copy afterwards.
The live solver's caches remain intact. A regression uses a payload that
cannot be copied and verifies checkpoint round-trip tensors and contracted
energy. This removes a transient allocation during saving; no molecular
peak-RSS reduction has yet been measured for this change.

Disk boundary metadata stores one BSR component descriptor, deriving data
and index lengths on read, rather than three nested array descriptors.
Mixed block sizes, empty components, float32/float64 payloads, and wide
indices round-trip without numerical changes. A synthetic 18,000-component
Python object-size measurement decreases descriptor storage from 9,082,132
to 2,746,076 bytes; this excludes channel keys, topology, numerical payload,
and allocator overhead and is not a molecular peak-memory measurement.

Tied-parameter actions consume reduced adjoint blocks as an iterator and
scatter each block immediately into the parameter vector. This avoids
retaining the complete expanded adjoint alongside the input state. The
general block-returning adjoint remains a collecting wrapper over the same
contraction, including compensated accumulation. Tests compare streamed
actions against independent reduced contractions and verify that the tied
pullback releases consumed blocks. This changes storage lifetime without
truncation; the molecular comparison below measures its combined effect
with compact disk descriptors.

A subsequent same-checkpoint pyrene audit compares compact disk descriptors
plus streamed adjoints with the preceding audit implementation. Peak RSS
decreases from 1,426,833,408 to 1,278,410,752 bytes (10.4%); elapsed time
increases from 1055.03 to 1068.86 s (1.3%). All site residuals agree at
rtol=1e-9, atol=1e-10, with identical maximum 0.2042576570494711.
These are sequential single-process measurements, not a statistical speed
claim or a converged state. This comparison covers one checkpoint only;
arbitrary molecular scaling remains open.

``examples/qchem/benchmark_su2_parameter_sweep.py`` runs actual passes and saves
an updated checkpoint separately. Two six-orbital forward passes accept all
12 updates, lower energy from -8.428765748128 to -8.428827564780 Eh, and agree
with a fresh conventional contraction after each pass. The run takes 3.97 s
with 293.8 MiB peak RSS. Tests cover closed virtual seams, consumed/recomputed
suffixes, independent final energy, rejected sites, and interrupted-pass
rollback, without calling the adjacent-pair solver. Complete molecular
convergence remains to be verified.

The first full pyrene forward baseline stopped at site 15 after 1651.77 s,
with a 1.47 GiB peak, and rolled the pass back. The metric cutoff contained an
absolute floor of 1: for the saved endpoint, the largest metric entry is only
1.27e-16 even though the state norm is about 1. The old 1e-11 cutoff therefore
retained zero directions. Component whitening and tangent-norm evaluation now
scale the cutoff by the actual metric magnitude without that floor. The same
endpoint retains 12 directions, with 1.60e-13 norm-orthogonality error and
1.30e-26 lost physical norm squared. Uniform-scale regressions span 1e-24 to
1e24. This fixes the false null support; it does not establish a completed or
converged molecular pass.

The subsequent relative-cutoff pyrene forward pass completes all 16 sites
in 1749.35 s with 1,339,097,088 bytes peak RSS (1.25 GiB). Eleven updates
are accepted, lowering active energy from -27.731170762755 to
-27.810476416737 Eh. Sites 0, 1, 10, 11 and 13 are rejected by the local
checks. The final endpoint retains 12 directions and passes its independent
residual check. The updated checkpoint is saved, but this single pass does
not certify final-state stationarity. This process predates the measured
Krylov-overlap correction and recoupling-cache lifetime changes above.

A subsequent corrected LR/RL cycle accepts 28 of 32 updates and lowers the
energy to -28.032755427485473 Eh. The cycle takes 3171.79 s; including an
independent final-state audit, total time is 4216.10 s with 2,035,122,176 bytes
peak RSS. The audited tangent residual is 0.2042576570494525, far above the
1e-8 target: this state is not converged. That process predates sparse-audit
cache cleanup, route-table streaming, and compensated adjoint accumulation.
Its peak covers both optimization and audit and cannot be attributed to a
particular phase from the recorded data.

Continuing that checkpoint with verified-descent acceptance completes one
LR/RL cycle in 2650.39 s, accepting 30/32 updates. Active energy decreases
by 0.186820669048 Eh to -28.219576096534 Eh. A fresh final-state audit gives
maximum residual 0.106766778503, reduced from 0.204257657049 but still far
above 1e-8. Total cycle plus audit time is 3734.20 s, peak RSS
1,671,806,976 bytes. Reverse site 13 produces an unconverged, higher-energy
candidate and is correctly rejected; reverse site 14 is also rejected.
This continuation demonstrates energy descent, not convergence or an
isolated speed comparison with the preceding cycle at a different state.

A further continuation with refreshed Davidson metric images and measured
projected matrix triangles accepts 27/32 updates (four by verified descent).
The cycle takes 2953.07 s and lowers active energy by 0.062413058539 Eh to
-28.281989155072 Eh. The independent final-state residual is 0.101624059832,
only a modest reduction from 0.106766778503; convergence is not established.
The complete run including audit takes 4202.03 s and peaks at 1,670,840,320
bytes RSS. Across all 32 local attempts, operator assembly takes 979.08 s,
Davidson 4.11 s, and the measured local-update regions 2166.65 s. These
regions exclude boundary transport and initial environment construction.
This identifies contractions and environment work as the runtime priorities;
different starting states prevent interpreting cycle timings as an isolated
solver speed comparison. Forward and reverse sites 10 and 13 are rejected
for local residual failures, as is reverse site 14. The exact pre-update
forward-site-10 checkpoint was retained for reproducing the discrepancy
between stored-operator and independent adjoint residuals.

The local solve now supplies ``T.conj().T @ N @ T`` as a generalized Davidson
action rather than treating numerical whitening as an exact identity. The
first pyrene pass exposed this issue at sites 0--3: standard-coordinate solves
stopped while independent metric residuals remained between 5.96e-8 and
8.15e-6, so those candidates were rejected. A regression with an intentionally
imperfect whitener requires agreement with the independent generalized root.
A six-orbital LR/RL cycle with the corrected norm action accepts all 12 updates
in 3.78 s, reaches -8.428811828588 Eh, and agrees with a fresh final contraction
within 3.2e-12 Eh. Its maximum checked local residual is 3.33e-11. This verifies
direction handling and the small cycle, not molecular convergence. The
benchmark defaults to alternating directions and saves partial progress even
if a later pass fails and is rolled back. Each independently accepted update
also writes ``latest_local_update.chk`` atomically, marked unconverged; completed
passes write ``updated.chk``. The former can retain useful local progress even
when a subsequent pass failure restores the in-memory starting state. Pass
rollback restores the original convergence flags and message as well as tensors
and energy.

``examples/qchem/benchmark_su2_sparse_boundary.py`` runs reproducible sparse
recursion from a saved checkpoint and emits JSON plus PNG/PDF figures. Six
pyrene right-boundary updates through cut 9 complete in 43.84 seconds, with
cuts 10 and 9 taking 10.24 and 20.30 seconds. The last boundary retains
195.06 MiB versus 1,087.91 MiB dense-equivalent storage. Whole-process peak is
1.17 GiB with all sites materialized, demonstrating why the per-arena budgets
must not be described as process-memory limits. This component is exercised
by the standalone benchmark and the explicit parameter-block passes;
it does not replace the default optimizer's environment path.

``examples/qchem/validate_su2_checkpoint_norm.py`` checks selected saved-state
norm metrics without constructing a molecular Hamiltonian. For the pyrene
D=2 checkpoint, cuts 0, 7 and 15 retain the same support and metric ranks.
Their relative metric differences are at most 5.8e-14. The strict entrywise
comparison still fails for one diagonal element at site 7 (difference
1.37e-12). A 60-digit scalar contraction of the same rounded reduced boundaries
gives a direct-contraction error of 6.8e-21, versus 1.37e-12 for the pair metric.
This diagnoses reference-path roundoff; the script preserves the failed gate
and exits unsuccessfully rather than weakening its tolerance. This check does
not certify the full molecular Hamiltonian audit or convergence.
``reduced_parameter_actions()`` exposes the direct ``H*c`` and ``N*c`` actions
for explicit reduced MPOs. Closed-loop stationarity audits use these adjoints
and one common total Rayleigh quotient; this avoids cancellation introduced
by adjacent-pair embeddings near an eigenstate. The local metric and residual
tolerance are retained, and paired actions remain independent test references. For singleton reduced
physical multiplicities, boundary and parameter actions contract virtual
matrices before combining operator components, avoiding quartic virtual-index
loops. General physical multiplicities retain the existing contraction path.
Large boundary blocks bypass small-block accumulation kernels; this changes
contraction order only, without truncation or an altered metric. Boundary
updates absorb the site tensors once per incoming channel and share those
virtual products across outgoing channels. One-site adjoints similarly share
ket absorption across terms with the same incoming channel. The cache lasts
only for that block update, so every new state uses fresh numerical data.

Reduced boundary storage drops exactly zero channel blocks when an arena is
at least 64 MiB and omission saves at least 25 percent of its values. Small
and wholly zero operators keep their existing representation. Python-owned
boundaries retain surviving arrays without copying; the real C++ Hamiltonian
boundary store is reinstalled with the compact topology and values. Identical
bra and ket site layouts share one packed site pool. All nonzero values,
including tiny coefficients, are retained. This is numerical operator
sparsity, not virtual-basis truncation or a change to the reduced algebra.
Each update reevaluates the zeros, allowing previously zero channels to return.
C++ metric and complex boundary stores retain their existing storage paths.
Compacting a C++ arena temporarily holds both the original and retained values;
this does not bound peak memory or compress nonzero frontier matrices. Runtime
and convergence improvements are not guaranteed.

The nonlinear optimization adapts SciPy's unconstrained L-BFGS-B implementation
of R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, SIAM J. Sci. Comput. **16**,
1190--1208 (1995), https://doi.org/10.1137/0916069, or full BFGS as described
by J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. (2006),
https://doi.org/10.1007/978-0-387-40065-5. The reduced formulation
follows Weichselbaum as cited above. This is a hybrid LETTA refinement, not
either paper's complete algorithm: each chart excludes norm-null anchor
directions and is restarted, and no global-minimum guarantee is provided.
The frozen-core constant is held outside the electronic Hamiltonian to
reduce cancellation in ``H-E*N``. Checkpoints retain this external shift.
Fresh full one-site stationarity and energy stability under verification
sweeps are required; optimizer termination alone is not convergence.
Accepted coordinates are checkpointed at each step; fresh one-site audits
and the main checkpoint are saved every 20 iterations.
With ``--global-metric``, overlaps between different site tangents are included
in a joint parameter Gram matrix. Mixed reduced norm boundaries propagate a
single ket variation; their adjoints give each cross-site overlap. The
Hamiltonian remains a reduced MPO and is not projected onto a many-body basis.
Whitening retains Gram eigenvalues above ``--joint-metric-rtol`` (default
``1e-8``) relative to its largest eigenvalue. The joint Gram uses this looser
chart cutoff to suppress roundoff amplified by cross-site cancellations;
fresh one-site audits still use ``1e-11``. Whitening therefore
removes redundant simultaneous parameter directions. This is a coordinate
adaptation of metric whitening, not a full TDVP or periodic DMRG method;
nonlinear curvature of discarded anchor directions is omitted until a restart.
Tests check the Gram matrix against independent reduced transition overlaps
and check the resulting complex chart derivatives by finite differences.
When this joint chart is enabled, convergence also requires a fresh joint
tangent residual below the tolerance, both before and after verification
sweeps. Separate one-site residuals can underweight nearly cancelling
combinations of site tangents.
``--seam-perturbation`` optionally changes the starting state by seeding the
weak D=2 seam channel. Complementary endpoint singular vectors suppress
first-order mixing with the dominant loop branch. The random reduced blocks
retain charge, spin and tensor shapes. This is an initialization heuristic,
without an energy-decrease or convergence guarantee; both pre-seed and
post-seed energies are recorded.
``--relative-energy`` evaluates changes from a fixed chart reference using
an exact telescoping reduced MPS difference. A neutral clock permits one
changed tensor between a reference prefix and current suffix; no physical
basis, operator projection or rank truncation is introduced. After fixing
the reference overlap by a scalar gauge, the objective uses mixed and
quadratic difference contractions of ``H-E_reference*N``. This avoids
subtracting two complete Rayleigh quotients in energy line searches near
stationarity. The gradients remain direct reduced Rayleigh derivatives,
including the scalar-gauge pullback. Tests compare this arithmetic against
independent transition contractions and a tiny one-site quadratic reference.
Verification uses fresh two-vector one-site Rayleigh--Ritz sweeps, adapting
the locally optimal preconditioned-gradient trial space of A. V. Knyazev,
*Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block
Preconditioned Conjugate Gradient Method*, SIAM J. Sci. Comput. **23**,
517--541 (2001), https://doi.org/10.1137/S1064827500366124. Each trial space
contains the current tensor and its metric-natural gradient; the previous
direction and block extension of LOBPCG are omitted. A shifted 2-by-2
Rayleigh matrix resolves tiny local energy changes, and a fresh global
energy check guards acceptance. This simplified adaptation has no global
convergence guarantee. Both local actions use direct reduced one-site
contractions, avoiding the pair-embedding cancellation observed at the end
of this closed-loop calculation. Verification reuses fresh H/N boundaries
while the state revision is unchanged, and rebuilds both after any accepted
or rolled-back tensor update. It still visits every site in both directions
for every requested cycle.

For the saved localized benzene cc-pVDZ CAS(6,6) benchmark, the seeded closed
``D=2`` run with joint charts and relative-energy line searches converges to
``-230.79124813361153`` Eh. The CASCI reference is
``-230.79124813361145`` Eh; their signed difference is ``-8.53e-14`` Eh,
at the numerical roundoff scale. Fresh maximum one-site and joint tangent
residuals are ``1.18e-9`` and ``1.45e-9`` Eh, respectively. Two fresh
verification cycles leave the energy unchanged, and checkpoint reload
reproduces these diagnostics. This result concerns the six-orbital active
space and restricted spin-zero multiplicity seam; it is not a full-basis
FCI calculation or a general spin-carrying periodic solver.

Passing ``graph=[]`` removes all physical ties and gives an ordinary reduced
MPS ansatz in the same contraction engine. With ``virtual_boundary='periodic'``
it uses the same restricted spin-zero, fixed-charge multiplicity trace as
closed LETTA. The molecular Hamiltonian and orbital ordering remain unchanged;
only the virtual boundary is periodic. ``D`` counts multiplicities per reachable
charge/spin sector, including the closing seam, rather than the total virtual
dimension summed over sectors. For an ordinary periodic MPS calculation use
``SU2DMRG.run(..., virtual_boundary='periodic')``: its constrained one-site
Davidson sweeps live in the non-Abelian MPS backend and do not invoke LETTA.
See the periodic SU(2) section in :doc:`mps` for method restrictions.

With virtual matrix contractions and verification boundary reuse, resuming
the same final-refinement checkpoint takes 150.7 seconds through seven BFGS
iterations and 171.2 seconds through verification, versus the previous
recorded 371.5 and 428.6 seconds. These single-threaded stage times include
audits, and exclude the earlier warm-up and final post-record joint audit.
Fresh one-site and joint residuals remain 1.19e-9 and 1.45e-9 Eh. The helper
``examples/qchem/benchmark_su2_closed_refinement.py`` alternates old/new
contractions on identical parameters, checks action agreement, and saves
individual timings for comparisons with archived pre-optimization sources.
Sharing virtual products across operator channels further reduces the same
seven-step refinement to 93.3 seconds and the recorded verification stage to
111.1 seconds total. This is 1.54 times faster than the preceding matrix
contraction version and 3.86 times faster than the original recorded stage.
The energy and both fresh residuals reproduce the preceding result exactly;
the raw bond dimensions, metric cutoffs and convergence tests are unchanged.

Runtime requirement
-------------------

The production molecular route requires a loadable SU(2) extension. An
installed extension with unresolved symbols now raises an explicit error,
rather than silently selecting the restricted Python Hamiltonian builder.
On macOS, the build pins the selected OpenMP runtime's absolute library path:
an earlier Conda RPATH must not substitute an incompatible ``libomp``.
Rebuild extensions after changing the compiler or OpenMP installation.
This is a runtime correction, not a change to the reduced tensor algebra;
it does not establish convergence or performance for large active spaces.

State overlap
-------------

``bra.overlap(ket)`` contracts LETTA conditional tensors directly.
Pass ``s=s`` for a nonorthogonal spatial-orbital basis; this
second-quantizes the matrix, including fermionic signs. LETTA graph ties
remain explicit copy indices and no MPS is materialized. States may have
different tie graphs and bond dimensions. ``SU2LETTA.from_narg(solver)``
imports sequential NARG roots as a LETTA batch, whose overlaps are matrices.

MPS and LETTA overlap entry points share
``pyqed.mps.nonabelian.overlap``. Untied LETTA is a reduced MPS and uses the
same two-site transformation and direct chain contraction as
``MPS.overlap(ket, s=s, tol=1e-8, max_bond=None)``. NN/NNN physical conditions
use three/four-site conditional updates. NNN fusion conditions use at most
three sites while retaining the future fusion selector explicitly. Dense MPSs currently
support same-basis overlaps only; unsupported orbital representations raise
explicitly. DMRG cross-geometry overlap uses this shared reduced engine and
transforms each selected root once.

Cross-basis overlaps explicitly biorthogonalize both reduced states.
``bra.biorthogonalize(ket, s, return_info=True)`` returns the two transformed
states and orbital maps satisfying the identity cross metric. Local reduced
state compression adapts ranks to ``tol=1e-8`` (absolute overlap-matrix
Frobenius truncation error). ``max_bond=None`` is the default; an optional cap
that prevents certification raises ``OverlapConvergenceError``. Use
``tol=None`` for cap-only compression. Bounds exclude roundoff and orbital
factorization error; no many-body
overlap operator is built. This transformation supports states without ties
and NN/NNN physical or fusion ties
(``graph="nn"`` or ``graph="nnn"``). Conditions remain explicit through
local updates; they are not folded into ordinary virtual bonds. Ties beyond
range two
remain supported for same-basis overlaps only. Singular cross metrics are
rejected. See :ref:`NARG overlaps <narg-letta-overlaps>` for equations,
references, compression error diagnostics, and limitations.
Generic ``pyqed.letta.LETTA.overlap`` also accepts ``site_overlaps``,
one bra-by-ket local metric per site. Its stored basis metric is
``overlap_matrix``. Local product metrics and a fermionic
``s`` are distinct, mutually exclusive inputs.

Construction
------------

By default, SU(2)-LETTA ties only nearest-neighbor orbitals in their current
order.  This keeps the frontier width bounded even though the quantum-chemistry
Hamiltonian retains every one- and two-electron coupling.  The production
optimizer currently materializes the fully reduced Hamiltonian MPO.  Pass
``graph`` explicitly to use a different variational tie graph; the graph does
not screen or modify the Hamiltonian.
``graph="nn"`` (or ``"nearest_neighbor"``) selects the default chain explicitly.
``graph="nnn"`` (or ``"next_nearest_neighbor"``) includes distance-one and
distance-two edges. ``graph="nnnn"`` (or ``"third_neighbor"``) includes all
edges at distances one, two, and three. This uses the general reduced frontier
optimizer. The specialized conditional canonical gauge and compact NN scatter
path apply only to ``"nn"``. For spatial-orbital physical ties, an NNNN cut
can carry three future multiplet labels, up to 27 frontier assignments instead
of 3 for NN; unfolded multiplicity spaces and contraction costs grow accordingly.
The NN timing benchmarks do not establish NNNN performance.

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

``hamiltonian_representation="complementary"`` is the default for
``from_integrals`` and the qchem driver.  It carries the reduced
$S/R/A/P/B/Q$ operators directly and does not store a Hamiltonian MPO.
``hamiltonian_representation="mpo"`` remains available as an explicit
reference.  Projected direct optimization automatically uses the conditional
canonical gauge.  After each accepted coordinate update it exactly
rematerializes the active pair, invalidates the corresponding reduced boundary,
and advances the retained half-sweep environments.  The D>1 direct action and
post-update energy are tested against the explicit reduced MPO.  Checkpoints
store the active-space integrals and rebuild the selected representation on
restart.
Two-orbital models use the explicit reduced MPO automatically because there is
no interior complementary-operator cut to amortize.

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
       checkpoint_every="final",
   )

Use ``checkpoint_every="final"`` (or ``None``) for short benchmarks to write
one restart file after the run.  A positive integer retains periodic restart
checkpoints and the final checkpoint for longer calculations.

The convergence decision is made only after complete LR/RL cycles.  Inspect
``state.convergence_summary`` for the energy change, maximum local residual,
maximum pair-retraction error, rejected updates, memory owned by the state,
and the number of consecutive qualifying cycles.

For explicit-MPO one-site updates, ``solver="auto"`` selects the native
reduced Wigner--Eckart path.  ``solver="polarization"`` is retained only as an
explicit small-system reference calculation; it is not selected automatically.

``algorithm="projected"`` is the preferred fixed-D optimizer.  It embeds the
tied one-site parameters into the adjacent channel-resolved DMRG pair space,
projects the effective Hamiltonian and norm actions with that embedding, and
removes null or redundant tied directions by diagonalizing the projected norm.
The retained basis is norm-orthonormal before ordinary Hermitian Davidson is
started.  This avoids an ill-conditioned generalized Krylov problem.
Both explicit-MPO and direct complementary optimization incrementally advance
DMRG-style environments.  The direct path discards the incremental
materialization delta for the changed active pair and rematerializes that pair
exactly before advancing its reduced boundary; the untouched side remains
cached for the rest of the half sweep.
An update is committed when its recomputed exact local Rayleigh quotient does
not increase.  A Davidson vector that reaches its iteration cap may therefore
remain a useful monotone variational step; its true residual is retained and
must still pass the requested threshold before the complete LR/RL cycle can be
declared converged.  No magnetic or determinant-space projection is used.

On states that retain the compiled SU(2) owner, the projected optimizer
installs ``E† H_eff E`` as indexed projections of only the active reduced
blocks.  Davidson vectors therefore remain in the tied orthonormal space
without Python lift/apply/project callbacks or dense zero-padded projection
matrices.  ``E† N_eff E`` is likewise contracted directly from factorized
reduced metric blocks. Production projected solves contract each left/right
boundary-factor pair with the active embedding panels before assembling the
small connected tied-metric components. They do not expand the parent pair
metric into dense Kronecker products. The explicit dense block construction
remains available for references. All factorized norm routes are retained;
connected-component filtering and eigenvalue-based rank whitening still act
on the projected metric. ``metric_blocks_factorized`` and ``metric_nbytes``
report this local representation and its factor storage.
Projection topology, batched one-site
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

Projected Davidson convergence is adaptive.  LETTA replaces the old
unconditional 16-vector exploration floor with a minimum of two vectors and
two consecutive checks of the Ritz residual, energy change, and root overlap.
The general packed solver retains its conservative default for other callers.
Recycled vector panels cross into the compiled projected action in one
``matmat`` call.  ``compiled_davidson="auto"`` keeps the complete solve in C++
for retained dimensions up to 16; larger spaces use the adaptive Python
orchestration because the all-C++ pyrazine CAS(10,10), ``D=4`` solve was
slower despite matching energy and residual.  ``True`` and ``False`` force
either path for measurement.

``projected_diagonal_weight`` controls a clipped blend of the projected
parent diagonal into the correction denominator and defaults to zero.  This
quantity omits off-diagonal parent-Hamiltonian contributions to
``diag(W^\dagger H W)`` and is therefore not advertised as an exact diagonal.
On pyrazine, weights 0.25 and 0.5 increased the iteration count and missed the
requested local residual, so the robust constant shift remains the production
default.

For large matrix-free norm problems, ``conditional_preconditioner=True``
selects six CG steps on the diagonally scaled norm with a regularizer of
1e-3. This approximates an inverse metric, not the inverse shifted
Hamiltonian, and may increase cost; benchmark before enabling it. It is a
metric-informed adaptation of Davidson (J. Comput. Phys. 17, 87, 1975;
doi:10.1016/0021-9991(75)90065-0), not an exact reproduction.
``davidson_workspace_mib`` controls the large-problem basis budget (default
32); increasing ``davidson_max_space`` alone can otherwise hit the memory cap.
Checkpoints retain up to four search vectors per local problem, including
the latest Ritz vector. They are reorthonormalized in the new metric and
their operator actions are recomputed. They are warm-start candidates, not
cached Hamiltonian results or guaranteed gauge-transported vectors.

The large matrix-free path applies ``metric_rtol`` to each orthogonalized
Krylov direction and the final candidate: its norm Rayleigh quotient must
exceed ``metric_rtol * max(abs(norm_diagonal))``. This prevents normalization
of almost-null directions that amplify operator roundoff into spurious
energies. This directional filter is a matrix-free stabilization, not the
exact eigenvalue-based whitening used by the smaller connected-block path.
Residual checks and finite energies remain necessary; an energy decrease
alone is not a certificate of a reliable local solve.

Complementary-Hamiltonian energy checks first canonicalize a temporary
unfolded reduced MPS with sectorwise single-site QR and no truncation. The
LETTA parameters are not changed. This avoids catastrophic cancellation
in the global acceptance check when conditional gauges leave large,
nearly cancelling coefficients. It adds reduced-space factorization cost;
it cannot grow bond dimensions and is neither a determinant-space projection
nor variational relaxation.
The first-bond energy check builds only the right environment stack and the
initial left boundary. Direct complementary boundary contractions retain
their structural route arrays up to 4 MiB per MPO core, avoiding repeated
topology packing while refreshing numerical tensors on every use. Larger
route arrays are discarded and rebuilt when needed. The cache is included
in ``storage_nbytes``; the bound applies to retained Python route arrays,
not the compiled owner or transient contraction workspace.

For example, an explicitly budgeted larger search space is::

    state.run(algorithm="projected", nsweeps=2,
              davidson_max_space=96, davidson_workspace_mib=128,
              conditional_preconditioner=False)

The budget covers the generalized Davidson basis estimate, not all state,
environment, and local-action storage. The frozen-checkpoint diagnostic
``examples/qchem/benchmark_cr2_local_solver.py`` compares cold and recycled
solves with 32 vectors, 96 vectors, or 96 vectors plus metric CG. Its
120-iteration runs measure residual quality and time, not convergence or
full-sweep speedup. The companion ``plot_cr2_local_solver.py`` renders its
JSON outputs.

For explicitly whitened smaller problems, an experimental hybrid correction
is available through ``conditional_preconditioner=True``. It retains the exact global Davidson
matvec and residual, partitions the retained coordinates by connected
conditional norm component, and solves a small block-Jacobi correction in
each component.  The block Hamiltonians are inexpensive projections of the
parent diagonal, so the correction is blended with the robust global shift;
``conditional_preconditioner_weight`` controls that blend and defaults to
``0.1``.  Blocks larger than ``conditional_preconditioner_max_block`` disable
the hybrid for that local problem.  This changes convergence acceleration,
not the variational problem.  It is opt-in because small fixed-``D`` tests can
spend more time constructing the blocks than they recover in Davidson steps.
Exact conditional self-Hamiltonians and their lowest modes were also tested as
Davidson seeds.  They are not exposed as a solver option: constructing them
requires one full projected-Hamiltonian action per retained conditional
coordinate, while the electronic Hamiltonian still couples different
components.  On the pyrazine CAS(10,10), ``D=4`` benchmark this increased both
the total matvec count and wall time and missed the requested local residual.

Pair embeddings cache only the parameter directions that contribute to each
reduced block, together with their global parameter indices. Contractions
operate on these compact panels instead of multiplying zero directions for
every site parameter. Sparse complementary-family diagnostics use the exported
index-array shape without scanning the integral entries. Both changes preserve
the reduced variational space, norm filtering, and Davidson stopping criteria.
``examples/qchem/benchmark_su2_letta_embeddings.py`` measures complete projected
cycles on deterministic small dense model integrals and compares before/after
JSON outputs with ``--compare``. These models measure overhead; they do not
establish molecular convergence or performance for large active spaces.
On a local single-thread run, median times for two cycles over three repeats
changed from 1.18 to 0.91 s (6 orbitals, ``D=3``), 5.12 to 3.99 s
(8 orbitals, ``D=3``), and 28.27 to 21.85 s (10 orbitals, ``D=4``).
Energies and final local residuals were identical in these runs. The
10-orbital embedding-basis cache shrank from 1.08 GB to 27.1 MB; these are
owned-array counts, not process peak-memory measurements.

The tied-to-reduced coordinate map is compiled directly from frontier-sector
slices and parameter offsets.  It no longer constructs a materialized MPS
tensor for every unit parameter direction.  Unit-direction ``IrrepTensor``
objects are generated lazily only for explicit reference solvers.  Numerical
reduced MPS sites are retained by the state and invalidated precisely when a
site parameter or crossing conditional gauge changes.  Thus the production
path carries the tie constraint as a persistent reduced-coordinate scatter
map while retaining the rank-filtered metric basis required to remove null
directions.

At fixed ``D``, the adjacent channel-resolved pair topology and the
parameter-to-pair contraction plan are invariant.  The projected optimizer
therefore compiles them once per bond.  On later visits it compares the fixed
neighbor blocks, reuses unchanged contractions, and applies only their numeric
deltas to a block-scatter embedding.  The global pair-by-parameter matrix
``E`` and whitened matrix ``E W`` are never materialized in the production
path.  Forward and adjoint actions, support restriction, projected diagonals,
metric contractions, and compiled factor-route projections consume the same
reduced blocks directly.  Connected norm-component topology is derived from
the structural parameter incidence, so its union-find decomposition is also
reused while the component values remain freshly contracted.  The
``constraint_cache_stats`` cycle diagnostic reports pair-coordinate hits,
numeric embedding hits and delta updates, reused/updated embedding blocks, and
metric-component topology hits.  These caches never reuse Hamiltonian or norm
values across a tensor change.

The moving Hamiltonian and norm sweeps are the sole local boundary owners in
the projected path; redundant standalone adjacent-pair anchors are not built.
The norm projection is accumulated directly into connected tied-coordinate
components.  Their topology is cached, numerical blocks are refreshed
component-wise, and each component is whitened immediately, so no full
``E† N_eff E`` temporary is required.  The compiled owner still supplies the
local factorized norm action.  Its C++-boundary-only metric handle is not used
for whitening because that handle exposes matvecs and a diagonal but not the
connected blocks needed to remove LETTA null directions; using raw generalized
coordinates was tested and was slower for strongly rank-deficient sites.

Accepted parameter changes update cached unfolded MPS blocks through the
persistent scatter map.  Ritz vectors are transported through the new metric
whitener on the next visit.  Updates below the metric step tolerance preserve
their tensor, gauge, and environment revisions.  Before Davidson, the current
vector is tested against the exact metric-orthonormal projected residual.  A
passing site receives a dependency-scoped stationary certificate containing
the per-site tensor revisions required by that local effective problem.  Later
visits reuse it without reconstructing the local pair problem; a changed
dependency invalidates the certificate.  Metric whiteners are cached one
connected component at a time, so unchanged blocks retain their eigensystems
when another block changes.  Stationary projected updates also skip redundant
tied-site rematerialization before advancing the exact sweep cursor.  A
complete cycle in which every update
is stationary is a deterministic fixed point and satisfies the requested
consecutive-cycle check without executing an identical extra cycle.  All graphs use a compact ``frontier-block-scatter`` constraint plan.
Frontier-label compatibility is compiled once per site and reused across
sector degeneracies; integer parameter/target maps replace mostly zero
route matrices for wider frontiers. Physical and fusion ties share this plan.

On a six-orbital, ``D=2`` synthetic reference, compiling all site routes
(seven-repeat medians, one thread) took 4.77 → 3.18 ms for NN,
38.35 → 8.29 ms for NNN, and 377.51 → 17.30 ms for NNNN.
NNN route storage fell from 4,980,384 to 32,832 bytes; NNNN storage fell
from 84,189,888 to 70,272 bytes. These are route setup/storage improvements,
not proportional full-sweep speedups. Five interleaved two-cycle runs gave
NN 3.533 → 3.666 s (eight orbitals, ``D=3``), NNN 1.399 → 1.395 s
(six orbitals, ``D=2``), and NNNN 1.848 → 1.704 s (six orbitals, ``D=1``).
All final energies agreed within ``1e-10``. The wider-frontier runtime gain
was about 8%; this change established no NN/NNN sweep speedup.
The benchmark script accepts ``--graph nn|nnn|nnnn`` to reproduce the
individual cases with fixed integrals, seeds, sweep counts, and tolerance.

For projected one-site problems larger than
``max_dense_metric_parameters`` (default 4096), the frontier path also
keeps the tied embedding and norm operator matrix-free.  Compact integer
scatter routes apply the embedding and its adjoint, while the reduced norm is
contracted as left/right factors rather than assembled as Kronecker blocks.
This removes the quadratic tied-coordinate metric allocation; generalized
Davidson still uses the exact norm action.  Its support and preconditioner
diagonal are accumulated from the pair diagonal and fixed neighboring tensor
without materializing the embedding.

Large direct-complementary plans also use an adaptive execution cache and a
wider output-fusion threshold.  Once a local plan contains at least 4096
reduced actions, immutable packed boundary panels are retained across Davidson
vectors and repeated output products are aggregated before the final scatter.
Smaller plans keep the conservative cache and copy thresholds.  The resulting
extra workspace is bounded and stores reduced boundary data only; it does not
materialize an MPO or determinant-space Hamiltonian.  Advanced profiling can
override the defaults with ``PYQED_SU2_EXECUTION_CACHE_ELEMENTS`` and
``PYQED_SU2_OUTPUT_FUSION_COPY_BUDGET``.
The nearest-neighbor tangent embedding groups equal reduced block shapes into
batched matrix products in both directions.  This preserves its matrix-free
storage while avoiding one NumPy contraction dispatch per sector block and
Davidson vector.

For a physical nearest-neighbor tie chain, ``gauge="conditional"`` applies an
exact reduced QR gauge independently for every crossing physical SU(2) sector.
The adjacent tensor absorbs the transfer matrix, so the graph-tied state is
unchanged.  Projected and two-site sweeps then reuse incrementally advanced
Hamiltonian and norm environments within and across half-sweeps.  When a
projected half-sweep already owns the required boundary, it retains that
boundary and lets the exact local norm handle a noncanonical center instead of
performing a gauge-only rebuild.  A real local gauge or parameter change is
carried into the moving boundary as it advances.  The
projected cycle energy and norm are taken from its terminal exact local
Rayleigh quotient, avoiding redundant full-chain contractions; validation
tests compare these values to explicit chain contractions.  General graphs
that do not expose one next-site condition
per internal frontier skip the conditional gauge; ``gauge=None`` and
``reuse_environments=False`` select the rebuild-every-bond reference path.
The default ``gauge="auto"`` enables the conditional gauge only for projected
normal-complementary sweeps.  Explicit ``None`` or ``"none"`` disables it.

Projected normal-complementary sweeps also use an adaptive global variational
guard by default.  Local energy changes advance a cheap running estimate; an
exact chain contraction is reserved for coordinate-inconsistent or previously
learned bonds, followed by one exact end-of-cycle check.  A true uphill local
move is rejected, and an uphill complete cycle is restored atomically.  Use
``global_variational_guard="all"`` as the slower every-update validation mode,
or ``False`` to disable the guard.

``hamiltonian_representation="auto"`` selects direct normal-complementary
operators for ordinary runs because their cold start is much cheaper.  For an
unusually long run, ``representation_break_even_cycles`` (default 40) permits
a one-time switch to the exact reduced MPO, whose warmed pyrazine ``D=4``
cycle was modestly faster.  Explicit ``"complementary"`` and ``"mpo"``
requests override the policy.  Every cycle records the selected representation
and the decision reason.

States constructed by ``SU2LETTA.from_integrals`` retain the transient C++
SU(2) system owner.  Their two-site Hamiltonian actions install exact
channel-resolved contextual routes in that owner.  For chains of at least six
sites, Hamiltonian environments are advanced by the same C++ owner using exact
reduced boundary route batches.  The owner retains the two center-motion
topologies for each bond across complete cycles.  A repeated topology reuses
its recoupling and execution schedule while refreshing borrowed boundary
matrices and numerical scale recipes in place; topology-changing truncations
still rebuild the affected entry.  Shorter chains retain the faster exact Python
boundary recursion because route packing cannot be amortized.  The factorized
norm stack remains explicit in Python.  Small projected solves and pair
retraction consume its metric blocks directly; large nearest-neighbor projected
solves stream the same factors without building those blocks.  LETTA disables the
DMRG-specific direct-complementary shortcut because its unfolded conditional
virtual basis requires the full contextual route topology.  Checkpoints and
deep copies drop the non-pickleable owner and use the same exact reduced Python
route path.  Immutable left/right reduced MPO recoupling blocks are cached per
core and reused by environment construction, local actions, and expectation
values.

``n_threads`` configures the OpenMP team owned by the compiled reduced engine.
Its default, ``None``, resolves to ``pyqed.num_workers()`` at construction.
It parallelizes independent dependency waves and sufficiently large route or
block batches; small batches deliberately remain serial because thread-launch
overhead exceeds their work.  This is separate from ``workers``, which controls
Python-side independent local setup and defaults to 1. Avoid combining large values for both
with a multithreaded BLAS.

``benchmarks/su2_dmrg_sweep_threads.py`` measures complete production
SU(2)-DMRG LR/RL sweeps on an H10 STO-3G reference at reduced bond limits
64, 128 and 256, using one, two and four native threads. It rotates thread
order across three repetitions, fixes the initial seed and local tolerances,
and checks every half-sweep energy against the serial run. Run with
single-threaded BLAS and ``--output`` outside the repository. The script saves
raw timing/energy records and a reproducible scaling figure. Its two forced
sweeps measure execution scaling, not ground-state or bond-limit convergence.

Within each numeric route revision, the compiled engine reuses channel
workspace offsets and scheduling work estimates across Davidson vectors.
The next route refresh or released workspace rebuilds this metadata; operator
values and vector-dependent contractions are not cached by this optimization.
Thread selection is evaluated on every application. Diagnostics expose
``channel_workspace_builds`` and ``channel_workspace_reuses``. This changes
preparation work only, with no altered truncation or solver tolerances.

Normal/complementary boundary preparation shares the existing bounded angular
recoupling cache with contextual core construction (at most 250,000 entries).
The cache key includes physical charges, all angular labels, direction, and
the dual-right convention. Only immutable angular coefficients are reused;
primitive operator values and current boundary/site tensors remain live.
Exactly zero primitive contributions are skipped before recoupling. This
preserves the existing reduced representation and contraction order.

The per-bond execution cache also retains persistent output groups, task and
bundle assignments, pack-task indices, and slab offsets. Reuse requires the
existing route topology/shape checks and an exact match of the thread count
and SU(2) execution settings. Shape remapping rebuilds the output schedule.
Operator panels and right-action caches are repacked from the current
boundaries on every activation; no stale numerical environments are reused.
``output_schedule_reuses`` counts these numeric-only activations.
``PYQED_SU2_DISABLE_SCHEDULE_REUSE=1`` forces schedule reconstruction for
validation. This extends PyQED's existing reduced execution-plan cache, rather
than reproducing block2's delayed-contraction algorithm.

Parallel persistent output groups retain assembled right-hand operator panels
within ``PYQED_SU2_OUTPUT_PANEL_CACHE_BYTES`` (64 MiB by default, zero disables
this additional cache). These panels are independent of the Davidson vector:
they are packed once per numeric route activation and reused by all subsequent
Hamiltonian applications at that bond. Vector-dependent left products are
still recomputed. Groups that exceed the budget use the existing dynamic
packing path. Panel preparation copies existing right-action cache values
where available and fills disjoint panels in parallel. This extends the
existing output-fusion slab; it does not change
contraction order, truncation, or eigensolver tolerances.
``peak_output_panel_cache_bytes`` reports the peak additional panel storage.

Route preparation uses a two-stage OpenMP path when multiple threads are
requested. Mutable angular/core caches are resolved serially first. Workers
then join matching operator channels and look up boundary blocks using
read-only maps and separate result buffers. Factor IDs and route arrays are
committed serially in the original input/output/term order. The optional
action-fragment path retains its existing serial preparation. Shared-cache
construction and factor-ID assignment are not parallelized.

Compiled real environment updates group routes by output block and assign
whole blocks to workers. Each worker has private contraction scratch space;
contributions within a block retain their serial order, avoiding atomics or
floating-point reductions across workers. Grouping is reused with the compiled
boundary topology. Small updates (fewer than 128 routes), uncompiled routes,
and complex updates retain the serial path. These are PyQED scheduling
adaptations, not reproductions of block2's environment algorithm. No operator,
recoupling, truncation, or convergence approximation is added. Diagnostics
``route_prepare_parallel_calls`` and ``boundary_update_parallel_calls`` verify
which paths executed. BLAS must remain single-threaded to avoid nested fanout.

For non-overlapping sweep profiling, ``owned_prepare_seconds`` measures
local preparation (including routes and diagonals), ``owned_solve_seconds``
measures the actual Davidson calls, ``owned_split_seconds`` measures splitting
and truncation, and ``owned_environment_seconds`` measures moving-environment
updates. These counters cover the compiled half-sweep loop only; initialization
is excluded. Their sum leaves a small bookkeeping residual in
``owned_half_sweep_seconds``. Nested route and matvec counters must not be
added to these phases.

``benchmarks/block2_sweep_threads.py --profile`` separates initial environment
setup and reports block2's per-direction phase timers. The companion
``benchmarks/su2_dmrg_phase_report.py`` compares those logs with before/after
PyQED JSON results and checks every directional-sweep energy. It uses the
median-wall-time repetition's phase values, with all repetitions shown as
dots. The block2 mapping follows the `v0.5.3 sweep implementation
<https://github.com/block-hczhai/block2-preview/blob/v0.5.3/src/dmrg/sweep_algorithm.hpp>`_:
``Teff`` (effective-Hamiltonian preparation), ``Teig`` (eigensolve), ``Tmve``
(moving environment), and ``Tdm + Tsplt`` (density matrix/splitting). Inclusive
``Tblk`` and nested environment timers are excluded from the sum. PyQED's
preparation scope is broader than ``Teff``. These are matched-Hamiltonian
implementation benchmarks, not exact reproductions of block2's algorithm;
initial MPSs and truncation details differ between solvers. Validate numerical
invariance within each solver, not equality of finite-sweep cross-package
energies. See H. Zhai et al., J. Chem. Phys. 159, 234801 (2023),
DOI: 10.1063/5.0180424 for block2.

``boundary_prepare_seconds`` measures preparation of boundary operator blocks
and schedules; ``boundary_update_seconds`` separately measures the boundary
contractions. ``boundary_recoupling_cache_hits`` counts angular cache hits
in boundary preparation. These are cumulative owner diagnostics, including
initialization; route activation/refresh timers are nested and should not be
added together. Fusion and pivot-refresh configuration is read once per
application/batch or activation, avoiding repeated environment lookups in
inner loops while honoring changes before the next operation.

The optional ``PYQED_SU2_TASKED_OUTPUT=1`` schedule splits persistent output
contractions into finer tasks when there are more than twice as many tasks
as output-affinity groups or bundles. Each worker accumulates into a private
output vector; the final reduction partitions output elements across the
whole team instead of reducing pairs of entire vectors with a barrier at
each tree level. The existing ``PYQED_SU2_PRIVATE_OUTPUT_BYTES`` limit
(default 128 MiB) still bounds the additional output buffers and falls back
to output affinity if exceeded. Serial execution is unaffected.

This is an adaptation of the Tasked scheduling concept described by block2,
not a reproduction of its MPO construction or scheduling implementation:
H. Zhai et al., *J. Chem. Phys.* **159**, 234801 (2023),
https://doi.org/10.1063/5.0180424; see also the
`block2 threading definitions <https://github.com/block-hczhai/block2-preview/blob/master/src/core/threading.hpp>`_.
No integrals, contraction terms, truncation thresholds, or Davidson tolerances
are approximated by this option. Summation order can change roundoff.
Block2's scaling results do not transfer to this adaptation. Output affinity
remains the default unless private buffers permit more threads: the H10
benchmark showed only about a one-percent benefit from broader tasking and
worse performance under CPU contention.

``benchmarks/block2_sweep_threads.py --output /private/tmp/block2-h10``
benchmarks block2 at D=256 with one and four threads, three alternating
repetitions, four directional sweeps, zero noise, and fixed Davidson squared
residual thresholds of 1e-16 on every sweep. It prepares exactly the same
PyQED H10 orbitals/integrals in a separate process to isolate OpenMP runtimes.
It saves timings, half-sweep energies, runtime configuration, and a figure.
Its timer includes initial environment construction; PyQED's sweep timer
excludes solver startup, so compare within-package speedups. Initial MPS and
truncation algorithms differ between packages; equal bond limits and a fixed
sweep count do not imply identical cross-package energies or convergence.

For the pyrazine STO-3G CAS(6,6), ``D=4`` nearest-neighbor benchmark, the first
cycle takes about 2.4 seconds and its certified stationary LR/RL cycle takes
about 0.07--0.08 seconds on the development machine.  With a final-only
checkpoint, the one-thread run reaches its stationary fixed point in two
cycles and about 2.5 seconds, compared with 4.45 seconds before stationary
certification and 9.22 seconds for the earlier three-cycle projected
implementation.  This small
``D`` does not contain enough parallel reduced-block work for material OpenMP
scaling.  The run retained the CASCI energy within ``1e-13`` Hartree.  This is the fast
LETTA path, but its tied metric and embedding work still make it slower than an
unconstrained SU(2)-DMRG sweep.  Timings are machine-dependent.

For a fixed-canonical-RHF pyrazine/6-31G CAS(10,10), ``D=4`` benchmark, the
direct block-scatter coordinates and incremental caches reduced warm projected
cycles from roughly 24 seconds to 5.7--5.9 seconds on the same development
machine.  After cache population, pair topology and norm-component
connectivity are reused across complete LR/RL cycles; numeric embedding blocks
are still updated whenever the fixed neighboring tensor changes.  This is an
architectural reduction in setup work, not a change to the variational
manifold or convergence criterion.

For the corresponding ``D=16`` one-thread cold sweep, adaptive complementary
panel caching, output fusion, and batched tangent contractions reduced wall
time from 182.4 to 131.7 seconds.
Time in the compiled Hamiltonian application fell from 128.8 to 85.5 seconds,
and its generic direct-complementary component fell from 114.5 to 36.9 seconds.
The sweep still used 1238 Davidson iterations, so this measures cheaper exact
Hamiltonian applications rather than altered convergence or orbital
optimization.

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
caches. Restore with ``SU2LETTA.load_checkpoint(path, workers=1, n_threads=None)``;
the current central thread budget is used unless explicitly overridden.
Pass
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
Inexact local solves
-------------------

The two-site path applies the reduced pair norm through factorized
left/right metric routes. When each channel has one diagonal product block,
it diagonalizes the small left/right factors and retains eigenvalue products
above the relative cutoff. Whitening and its adjoint are applied as tensor
contractions, yielding a standard Davidson problem on the retained support.
This is spectral whitening, not just diagonal rescaling. More general metric
route connectivity falls back to generalized Davidson. Neither path assembles
a pair-dimension-squared norm matrix or a dense whitener.
Krylov storage is bounded by 128 MiB (excluding operators and contraction
workspaces). For arbitrary ties, scatter-based embedding and adjoint actions avoid
pair-dimension-by-local-parameter design matrices in both retraction and
factor relaxation. Wider frontiers can still require larger embeddings;
this is not a graph-independent memory bound.

Retraction uses up to 200 LSMR iterations per factor to fit coefficients,
selecting the best iterate in the pair norm, then relaxes the energy with
matrix-free factor H/N actions. This adapts Fong and Saunders, *LSMR: An
Iterative Algorithm for Sparse Least-Squares Problems*, SIAM J. Sci. Comput.
33, 2950–2971 (2011), doi:10.1137/10079687X, and Davidson, J. Comput. Phys.
17, 87–94 (1975), doi:10.1016/0021-9991(75)90065-0. The coefficient fit is
not an exact metric-optimal truncation. Directional null-space filtering,
finite iteration limits and diagonal-only preconditioners do not guarantee
a converged pair root. The full factor H/N actions retain off-diagonal
couplings. Accepted pair updates additionally pass a global energy check.
The reported pair residual is the relative unwhitened generalized residual.
``verified_residual`` independently checks the root in whitened coordinates
when available (raw coordinates otherwise); unresolved roots are rejected
before retraction. The factorized path reports its retained spectral rank;
the generalized fallback does not claim an exact metric rank.
If a preconditioned Davidson direction fails metric support checks, the
solver tries the raw residual and then its metric image before reporting
subspace exhaustion. These fallbacks use the same support tests, are counted
in ``correction_fallbacks``, and do not relax the convergence tolerance.
This is an implementation safeguard, not a guarantee against ill conditioning.

Pair memory profiling
~~~~~~~~~~~~~~~~~~~~~

``examples/qchem/audit_cr2_two_site_memory.py`` records current RSS, peak RSS,
partially accounted state arrays, and compiled-engine byte counters at each
stage. Engine categories overlap and must not be added blindly. Peak RSS is
a process high-water mark, not live pair storage. The optional
``--execution-cache-mb 64`` caps execution panels through the existing
``PYQED_SU2_EXECUTION_CACHE_ELEMENTS`` setting (8,000,000 double elements).
On the bounded Cr2 D=8 central-pair audit this reduced peak RSS from 4.03 to
3.33 GiB, while increasing pair time from 11.02 to 14.82 s; accepted energies
agreed within 5e-12 hartree. These are single-run measurements, not a universal
speed/memory guarantee. Library defaults remain unchanged because the cap
trades repeated packing work for memory. Clearing factor routes after the
audit did not reliably return heap pages to the OS.

The audit's ``--compact`` option additionally disables compiled output
fusion while retaining a 64 MB execution-panel cap. In the same bounded
central-pair experiment, default / cap-only / no-fusion / combined peak RSS
was 4.03 / 3.33 / 3.55 / 3.15 GiB, and pair times were
11.02 / 14.82 / 19.23 / 20.84 s. Energies agreed within 1e-11 hartree.
This uses existing execution policies without changing tensor dimensions or
truncation. It is an optional memory/time tradeoff, not a new contraction
algorithm or a recommended speed default. Process RSS varies with allocator
and cache history; these single-run data do not establish universal bounds.

Materialized site updates overwrite entries from the current tied coefficients
using the cached reduced scatter plan. They do not accumulate parameter
deltas: a large ill-conditioned trial followed by rollback can otherwise
destroy the incumbent through floating-point cancellation. The materialized
object identity is preserved for sweep references, and compiled site metadata
is invalidated. This is an exact representation update, not a truncation.

``run(adaptive_davidson=True, davidson_tol=1e-8)`` optionally uses
``max(davidson_tol, min(1e-4, 0.1 * previous_energy_delta))`` until the
previous round-trip energy change reaches the requested energy tolerance.
It then uses the requested Davidson tolerance. This is a heuristic
inexact-solve adaptation of the existing Davidson optimizer, not a new
convergence guarantee. Energy and true-residual stopping tests are unchanged.
The selected tolerance is recorded in each history entry. Use
``reset_history=False`` to retain this schedule across single-sweep calls.
Rejected site proposals clear recycled Krylov vectors for that site; a
rolled-back cycle clears all recycled vectors. Global energy guards remain
necessary for ill-conditioned local metrics.

Adaptive invariant ties
-----------------------

Enable graph growth during optimization with:

.. code-block:: python

   state.run(
       nsweeps=8,
       adaptive_ties=True,
       tie_options={
           "every": 2,
           "shortlist": 2,
           "relaxation_sweeps": 1,
           "max_parents": 2,
           "max_frontier_width": 2,
           "max_frontier_states": 81,
           "max_parameters": 100000,
           "minimum_energy_gain": 1e-8,  # Hartree
       },
   )
   print(state.graph, state.tie_history)

The graph is reconsidered after each batch of ``every`` ordinary cycles,
including the final batch. ``nsweeps`` budgets ordinary fixed-graph cycles;
each graph step additionally runs one fixed-graph control and up to
``shortlist`` candidate relaxations, each of ``relaxation_sweeps`` cycles.
Accepted relaxation cycles remain in ``state.history``. A newly accepted
graph clears convergence: another fixed-graph run is needed to certify its
stationarity. Adaptation runs to the requested budget; convergence of one
fixed graph does not stop graph exploration.

Missing edges are filtered by maximum parent count, frontier width, invariant
frontier assignments, and total reduced tensor entries before any candidate
allocation. ``candidate_edges`` can restrict screening to chemically motivated
pairs. Otherwise every missing pair is screened; this costs up to quadratically
many pair queries, each requiring several reduced norm contractions. The
assignment cap excludes virtual dimensions and Hamiltonian channels, so it is
not a complete byte-memory bound.

Candidates are ranked by classical mutual information of the current invariant
labels, minus ``cost_weight`` times the logarithmic increase in tensor entries
and accumulated frontier assignments. Joint probabilities come from exact
reduced sector-projector contractions, with no determinant expansion. Physical
ties measure local empty/single/double sectors; fusion ties measure incoming
coupling sectors. These scores omit quantum coherences within a sector and
are **not** full orbital quantum mutual information or a local-energy residual
score. They shortlist proposals, rather than proving an edge will help.

``state.with_tie_graph(graph)`` constructs an independent state containing the
existing edges plus additions. For every reduced block it repeats the old
conditional coefficients along the added invariant label axes. This preserves
the full wavefunction, its norm, and the virtual-sector layout before relaxation;
magnetic projections are never copied. Removal is unsupported. The current
Hamiltonian representation and molecular metadata are retained. Real direct
complementary states and complex explicit-MPO states are covered by migration
tests; complex direct-complementary contraction is not qualified here.

``state.adapt_ties(...)`` performs one graph step and returns the same solver.
It compares fresh candidate energies with both the original energy and a
fixed-graph control optimized for the same extra cycle budget. A tie must beat
both by ``minimum_energy_gain`` (and a roundoff floor). At most one candidate
is adopted; failed candidates leave the original graph, tensors, and cycle
history unchanged. A candidate that exceeds a runtime memory cap is recorded
as ``status="memory_limit"`` and skipped; remaining shortlisted candidates
are still tried. Other solver errors propagate. ``tie_history`` stores scores,
costs, control and candidate energies, and the chosen edge. Checkpoints retain this history, the graph,
and its frontier-assignment cap. During adaptive runs checkpoints are written
at graph-step boundaries, after acceptance; ``checkpoint_every`` is superseded
by this schedule. Adaptation settings must be supplied again on restart.

This is a greedy information-guided adaptation, inspired by Arseny Kovyrshin
and Markus Reiher, "Self-adaptive tensor network states with multi-site
correlators," *Journal of Chemical Physics* **147**, 214111 (2017),
https://doi.org/10.1063/1.5004693. It does not reproduce their SATNS ansatz,
quantum entropy selection, or correlator growth rule. Tensor relaxation uses
the existing Wigner--Eckart reduced solver (Weichselbaum, *Physical Review B*
**86**, 245124 (2012), https://doi.org/10.1103/PhysRevB.86.245124). There is
no global graph-optimality or convergence guarantee. The unchanged-graph
control reduces, but does not eliminate, finite-budget optimizer bias.

Cr2 adaptive-tie pilot
~~~~~~~~~~~~~~~~~~~~~

``examples/qchem/cr2_adaptive_ties.py`` reproduces a bounded graph-growth
comparison using a checkpoint and stored reference JSON. It also regenerates
``cr2_adaptive_ties.png`` and ``cr2_adaptive_ties.pdf`` from ``results.json``
with ``--plot-only``. All outputs are written to a fresh directory.

The 2026-10-01 run used the locally converged bond-paired D=8 Cr2 checkpoint:
CAS(12,12), 1.68 Angstrom, cc-pVDZ-DK, spin-free one-electron X2C, fixed
AVAS valence orbitals. Its integrals exactly match the stored bond-paired
benchmark. This remains an active-space calculation without external dynamic
correlation, spin-orbit coupling, or basis-limit corrections. The matched
DMRG reference is -2099.1769535957574 Hartree.

Screening 55 missing edges shortlisted orbital positions 10 and 12
(one-based). This would increase reduced tensor entries from 76,896 to 84,960
and the maximum frontier from one to two invariant labels. The starting
energy was -2099.1755411579566 Hartree, 1.41244 mHartree above DMRG. One extra
fixed-graph cycle reached -2099.175541202431 Hartree; one equally budgeted
enlarged-graph cycle reached -2099.1755412024763 Hartree. The approximately
4.5e-11 Hartree difference is below the 1e-7 Hartree acceptance threshold;
the tie was rejected and the original state retained. Screening and both
trials took about 246 seconds on the local single-thread run.

This one-candidate result does not establish that every other edge is useless
or that a globally optimized adaptive graph cannot improve the Cr2 result.
It demonstrates why current-state correlations must be followed by actual
variational gain checks. Outputs are in
``/private/tmp/cr2-adaptive-ties-D8-20261001/``.

Localized Cr2 pilot
~~~~~~~~~~~~~~~~~~

``examples/qchem/cr2_localized_letta.py`` provides ``prepare``, ``optimize``,
``adapt``, and ``plot`` stages for the same Cr2 active space. The preparation
jointly localizes all twelve active orbitals, including occupied/virtual
mixing, while leaving frozen-core and external subspaces fixed. It uses
Pipek--Mezey localization (Pipek and Mezey, *Journal of Chemical Physics*
**90**, 4916 (1989), https://doi.org/10.1063/1.456588) with PySCF's
meta-Lowdin population adaptation (https://doi.org/10.1021/ct401016x).
This is an adaptation of the original population criterion. Atom-local
partners are matched by Lowdin AO character and interleaved; this ordering
is a heuristic, without an optimality guarantee.

The active Hamiltonian is rotated exactly, with inverse-integral and
active-subspace checks; no AO or full-MO ERI tensor is allocated. Finite-D
LETTA is reoptimized from a fresh seed, rather than assuming its truncated
state is invariant under an orbital rotation. The same active-space DMRG
reference therefore remains applicable.

The 2026-10-01 run localized six orbitals on each Cr, with minimum dominant
atomic population 98.48 percent. Four D=8 nearest-neighbor round trips
reached -2099.1763659206176 Hartree, 0.587675 mHartree above the reference,
using the same 76,896 reduced tensor entries as the previous delocalized
trial. This is a lower variational energy, but the localized optimization
is not converged: the final cycle changed the energy by 2.63e-5 Hartree
and had maximum local residual 2.30e-5. Outputs and reproducible PNG/PDF
diagnostics are in ``/private/tmp/cr2-localized-D8-20261001/``.

Screening 55 missing edges ranked positions 5--7 highest. Its trial exceeded
the 256 MiB packed Wigner--Eckart route cap; this is a resource rejection,
not evidence of an unfavorable energy. The next-ranked pair, positions
1--3 on the same Cr atom, adds only 144 entries (77,040 total). One trial
cycle reached -2099.176371991047 Hartree, versus -2099.1763719996907 Hartree
for the equal-budget unchanged-graph control. It was rejected: it did not
beat the control by the required 1e-7 Hartree. The original four-cycle
localized checkpoint was retained. This second comparison took 446 seconds,
with outputs in ``/private/tmp/cr2-localized-D8-20261001/adaptive-low-cost/``.
It can be reproduced by running ``cr2_adaptive_ties.py`` on ``localized.chk``
with ``--candidate-edges 0 2`` and a fresh output directory. Neither the
finite-budget localization comparison nor these two trials establish the
optimal orbital ordering or tie graph.

Matched localized MPS comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/qchem/cr2_localized_comparison.py`` benchmarks Block2 SU(2)
two-site DMRG and nearest-neighbor D=8 LETTA against exactly the same
localized CAS integrals and orbital ordering. It verifies identical input
hashes before plotting. Run its ``mps`` and ``letta`` stages sequentially
in separate single-thread processes, then run ``plot``. All outputs default
to ``/private/tmp/cr2-localized-comparison-20261001/``; a fresh ``--out``
directory is required for another run.

The MPS dimension scan warm-starts increasing bond dimensions, with up to
eight round trips per dimension. Retained-state energies are measured by
fresh expectation contractions: two-site Ritz energies before truncation
are recorded separately. Stationarity requires two consecutive energy
changes below 1e-8 Hartree. A stationary finite-dimension state is not an
exact reference. LETTA starts from seed 7 and requires both energy-change
and local-residual criteria on consecutive cycles. This is a benchmark of
existing implementations of the MPS variational formulation (White,
*Physical Review Letters* **69**, 2863 (1992),
https://doi.org/10.1103/PhysRevLett.69.2863; Schollwoeck,
*Annals of Physics* **326**, 96 (2011),
https://doi.org/10.1016/j.aop.2010.09.012), not a reproduction of a published
Cr2 benchmark or a guarantee of a global variational minimum.

Plots compare energy error with stored reduced coefficients, cumulative
optimization time, and whole-process peak resident memory. Counts include
gauge-dependent stored entries rather than independent degrees of freedom.
MPS time includes lower-dimension warm starts; resident memory includes
libraries and the high-water mark of earlier dimensions. Timing comparisons
also reflect Block2's compiled implementation versus the current LETTA
implementation. The error plotting floor is 1e-5 mHartree; reference-level
points at that floor must not be interpreted as a measured nonzero error.

The 2026-10-01 MPS scan reached 0.063909 mHartree error at M=128 with
38,547 stored coefficients, 32.40 seconds cumulative time, and 761.06 MiB
peak process RSS. At M=256 it reached 0.00121142 mHartree with 84,585
coefficients, 56.43 seconds, and 1434.88 MiB peak RSS. Both points passed
the energy-stationarity check. M=512 matched the stored reference energy
to printed precision in 79.29 seconds cumulative time; its discarded
weight was about 7e-19. M=16 and M=64 exhausted their cycle budgets without
stationarity and appear as open circles in the coefficient-count plot.

The matched fresh D=8 LETTA run reached -2099.1763930756533 Hartree after
12 round trips: 0.560520 mHartree error with 76,896 stored coefficients,
1377.64 seconds total time, and 4274.45 MiB peak process RSS. It exhausted
the cap without convergence (last energy change 8.31e-7 Hartree, maximum
local residual 4.06e-6). Thus this finite-budget comparison demonstrates
no LETTA advantage: M=128 MPS is more accurate with fewer stored entries,
less runtime, and lower process memory. M=256 has approximately ten percent
more entries than LETTA but about 463 times smaller error. These results
apply to this Hamiltonian, ordering, ansatz, and implementations; they
do not prove a general optimality statement about MPS or tied networks.

To continue the same LETTA checkpoint with larger local Krylov solves:

.. code-block:: bash

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   python examples/qchem/cr2_localized_letta.py optimize \
       --out /private/tmp/cr2-localized-comparison-20261001/letta \
       --resume --sweeps 32 --davidson-maxiter 640 --davidson-max-space 128

The continuation appends history, records local-solver settings by segment,
and stops on two consecutive sweeps satisfying the unchanged energy and
residual criteria. Peak RSS is the maximum across restart processes, and
cumulative time includes the previous optimization. The original 12-sweep
comparison is retained in ``before-letta-continuation/``.

The continuation converged at cycle 16 with energy
-2099.1763936928464 Hartree, 0.559903 mHartree above the matched DMRG
reference. Cycles 15 and 16 changed the energy by 7.35e-8 and 4.23e-8
Hartree, with maximum local residuals 3.62e-7 and 1.21e-7. Reloading the
final checkpoint reproduced its energy exactly at printed precision and
gave norm 0.9999999999999958. The D=8 graph and 76,896 tensor entries
were unchanged. Total time including the first twelve cycles was
2816.22 seconds, with peak process RSS 4756.33 MiB. These checks establish
local fixed-ansatz convergence under the stated tolerances, not global
optimality over initializations, orbitals, or graphs. The converged result
still shows no advantage over the matched MPS points. Verification is
saved in ``letta/convergence_verification.json`` and the comparison figures
have been regenerated from the final histories.

Benzene molecular ring ties
--------------------------

``examples/qchem/benzene_su2_letta_ring.py`` compares open-chain ties with a
hexagonal carbon-ring graph, adding ``(0,5)`` to the five chain edges.
The fixed RHF/STO-3G pi CAS(6,6) uses symmetric-orthogonalized carbon 2pz
orbitals in geometric ring order and freezes 18 occupied sigma orbitals.
All Hamiltonian integrals are retained. This is a heuristic molecular graph
pilot, with no orbital optimization, external-space correlation, or general
graph-optimality guarantee. Exact CASCI is an independent small reference;
the active solver uses reduced SU(2) projected sweeps without determinant
projection.

Eight one-thread cycles from the same graph-neutral initial MPS gave:

* ``D=2``: chain error 3.567 mHartree in 2.35 s; ring error 0.427 mHartree
  in 6.10 s. The ring energy was still changing at the cycle cap.
* ``D=3``: chain error 0.324 mHartree in 2.50 s; ring agrees with exact
  CASCI (-227.997024016875 Hartree) to sub-nanohartree accuracy in 14.35 s.
  Stored coefficient counts were 1710 and 1764, respectively. The ring
  increases frontier width from one to two and assignments from three to nine.

Times exclude construction and are single-run pilot measurements. A separate
explicit-MPO expectation of the final ring checkpoint agrees with the direct
complementary expectation within 4.1e-10 Hartree. The plotted error floor is
1e-8 mHartree. Reproduce all results, validation, and figures with:

.. code-block:: bash

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   python examples/qchem/benzene_su2_letta_ring.py --validate

The same script also runs untied reduced MPS states via ``graph=[]``, using
the identical projected one-site solver, integral files, seeds, orbital order,
and eight-cycle cap. This isolates ansatz effects; these are not timings for
a separate optimized two-site DMRG implementation. At ``D=3`` the MPS has
576 stored coefficients and 3.280 mHartree error in 1.96 s, versus ring
LETTA's 1764 coefficients and reference-level energy in 14.35 s. Increasing
MPS to ``D=8`` gives 4016 coefficients and reference-level energy in 1.32 s
for eight cycles; its first cycle already matches CASCI in 0.53 s. Ring
LETTA first falls below 1e-9 Hartree error at cycle five (9.32 s).

Thus ring LETTA uses 56% fewer stored entries than this reference-accurate
MPS, but is much slower. Stored entries include gauge-dependent coefficients,
not independent variational degrees of freedom. The scan is a finite-budget,
single-initialization pilot: MPS dimensions 2, 3, 5, and 6 exhausted the cycle
cap, and the non-monotone MPS error versus dimension illustrates optimization
limitations. It establishes neither global MPS/LETTA minima nor general
compression or timing guarantees. MPS ``D=4`` was stationary at 0.433
mHartree error with 1016 stored entries in 2.31 s.

Add the MPS scan to the existing reference/output folder with:

.. code-block:: bash

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   python examples/qchem/benzene_su2_letta_ring.py --methods MPS --D 2 3 4 5 6 8

The comparison figure ``benzene_mps_vs_ring_letta.png`` plots energy error
against stored coefficient counts and optimization time, with open final
markers for runs that exhausted the cycle cap. JSON/checkpoint files and the
CASCI Hamiltonian are reused rather than regenerated on subsequent runs.

Larger-basis benzene pilot (cc-pVDZ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--basis cc-pvdz`` repeats the fixed pi CAS(6,6) benchmark in 114 AO
functions at the same geometry. Using one selected cc-pVDZ carbon AO per
site would omit much of the valence radial flexibility. Instead, PySCF AVAS
selects the six-orbital space from MINAO carbon 2pz references at threshold
0.2 (Sayfutyarova et al., JCTC **13**, 4063 (2017),
https://doi.org/10.1021/acs.jctc.7b00128). The script adds a symmetric
projection/orthogonalization within that selected space, mapping the orbitals
onto the six carbon references in geometric ring order. This is AVAS followed
by an additional localization rotation, not CASSCF or Boys/Pipek--Mezey.
Dominant reference-overlap weights are 0.948 for each carbon; these weights
are localization diagnostics, not atomic electron populations.

AO integrals remain packed; only the six-orbital CAS reference restores a
dense active ERI tensor. RHF energy is -230.721903074067 Hartree and exact
fixed-space CASCI is -230.791248133611 Hartree. Eight projected one-site
cycles, excluding construction, gave:

* MPS ``D=3``: 576 entries, 3.051 mHartree error, 2.77 s; cycle cap reached.
* Ring LETTA ``D=3``: 1764 entries, CASCI accuracy within 1e-9 Hartree,
  13.70 s. The error first falls below 1e-9 Hartree at cycle four (7.10 s).
* MPS ``D=8``: 4016 entries, CASCI accuracy within 1e-9 Hartree, 1.31 s;
  its first cycle is already accurate in 0.56 s.
* Ring LETTA ``D=8``: 12144 entries, CASCI accuracy, 82.88 s. Enlarging
  the ring ansatz adds cost without improving this reference-level energy.

The separate explicit-MPO expectation of ring ``D=3`` differs from direct
complementary expectation by 2.3e-11 Hartree. All trajectories satisfy the
active-space variational bound and decrease monotonically within tolerance.
These remain single-start, single-thread pilot timings in a six-orbital
active space: increasing the AO basis does not add sigma or external-space
correlation to CAS(6,6), nor establish basis convergence.

Reproduce the run and plots in a separate folder so STO-3G outputs remain:

.. code-block:: bash

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   python examples/qchem/benzene_su2_letta_ring.py --basis cc-pvdz \
       --output /private/tmp/benzene_su2_letta_ccpvdz \
       --methods MPS Ring --D 3 8 --validate

A matched eight-cycle cc-pVDZ ring LETTA ``D=2`` run used 804 stored entries
and reached -230.790730558680 Hartree, 0.517575 mHartree above CASCI, in
5.62 s excluding setup. Its final cycle changed the energy by 4.64e-5
Hartree, so it was not converged. Relative to ring ``D=3``, it uses 54%
fewer entries and 2.4 times less optimization time at the same cycle cap,
but does not reach reference accuracy. Add this point with ``--methods Ring
--D 2`` using the cc-pVDZ output folder above.

Ring sweep optimization
~~~~~~~~~~~~~~~~~~~~~~~

Projected factor-route transforms with exactly zero imaginary entries are
installed as real arrays. Real input vectors use the existing real-valued
factor-route action; genuinely complex vectors are applied through their
real and imaginary parts, and complex transforms retain the complex action.
No imaginary terms are dropped using a numerical tolerance.

``davidson_max_space=None`` now selects 96 vectors when frontier width exceeds
one and ``D >= 3``, otherwise 32. Explicit values override this policy, and
local support still bounds the subspace dimension. This is a restart-capacity
heuristic, not a new eigensolver or a relaxation of the residual tolerance.
The larger recycled search space can require more workspace. It does not
introduce a determinant projection or a dense global Hamiltonian.

Three interleaved, one-thread eight-cycle runs on the cc-pVDZ ring reference
(seed 11, projected sweeps, reused environments, ``tol=0``,
``davidson_tol=1e-9``) gave median D=3 time 13.765 → 10.563 s, about 1.30x
faster (23% less runtime). Davidson iteration counts fell from 2037 to 977,
and the last maximum local residual improved from approximately 1.3e-8 to
6.0e-10. The optimized energy first reaches 1e-9 Hartree CASCI accuracy at
cycle three, versus cycle four previously. The first cycle need not be more
accurate; the larger restart capacity changes the optimization trajectory.

D=2 remains on 32-vector capacity: medians 4.787 → 4.863 s show no reliable
runtime gain. Its final energy agrees with the previous path within 5.3e-12
Hartree; D=3 energies agree within 2e-13 Hartree. Real arithmetic alone and
forcing compiled Davidson did not establish a reliable full-sweep gain, so
the compiled-Davidson policy remains unchanged. Diagonal/conditional
preconditioner trials were not adopted. These timings are pilot measurements,
not graph-independent speed guarantees; explicit ``davidson_max_space=32``
restores the earlier restart policy for comparisons.

Periodic virtual-bond benzene reference
---------------------------------------

``examples/qchem/benzene_virtual_pbc.py`` compares open virtual endpoints
against a genuine trace closure, independently switching physical ring ties.
It reuses the localized cc-pVDZ CAS(6,6) integrals from the benzene example.
This is an explicit small-system dense reference, not a periodic extension
of ``SU2LETTA``. All amplitudes and local projectors are restricted to the
400 configurations with three alpha and three beta electrons. Number
projection adds expressivity; total spin is not constrained, and the script
records its expectation. Ordinary virtual dimensions cannot be equated
with the reduced solver's SU(2) multiplicity parameter.

The local generalized-metric updates use ``DenseTiedLETTA``'s reference
adaptation of Verstraete, Porras and Cirac, Phys. Rev. Lett. 93, 227205
(2004), https://doi.org/10.1103/PhysRevLett.93.227205. They enumerate exact
configuration projectors rather than compressed periodic environments.
There is no global convergence guarantee or large-system scaling claim.
Both boundaries use alternating passes, two initialization seeds (11, 23),
a 32-pass cap, and one numerical thread. Zero floating-point energy change
can terminate a run earlier; actual pass counts are recorded. Timings
exclude construction of the shared Hamiltonian and initialization.

At ordinary D=3, median errors in mEh were 38.164 (open, untied),
5.478 (periodic, untied), 13.028 (open, ring ties), and below 1e-9
(periodic, ring ties). The latter reached CASCI in the first pass for both seeds (stopping at
passes 4 and 7), but its
site-zero local projector already has rank 400: that update spans the
entire fixed-number reference sector. Thus exactness here is a saturation
of this tiny reference space, not evidence of efficient scalable periodic
SU(2) optimization. At D=2 the ring-tied results depend strongly on the
seed, with periodic bonds failing to improve the final energy in one pair.
Other runs remain unconverged at the pass cap. Raw parameters, histories,
metric ranks, residuals, spin expectations, checkpoints and reproducible
PNG/PDF plots are written to the requested output directory.

For example, with the earlier benzene integral file::

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python examples/qchem/benzene_virtual_pbc.py \
      --integrals /private/tmp/benzene_su2_letta_ccpvdz/benzene_pi_integrals.npz \
      --output /private/tmp/benzene_virtual_pbc

D=2 continuation and stationarity diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/qchem/converge_benzene_virtual_pbc.py`` resumes the eight D=2
checkpoints, retaining their tensor shapes, number sector, seeds and physical
ties. It subtracts the common reference energy from the Hamiltonian for
numerical conditioning, without using the CASCI eigenvector in optimization.
Alternating updates continue in blocks of 20 passes. Convergence requires
both a block energy change below 1e-10 Eh and a maximum local tangent
residual below 1e-7 Eh. The tangent residual projects the normalized
Schrodinger residual onto each local projector's metric-supported range
(cutoff 1e-12), rather than requiring a finite-D ansatz to be an exact
eigenstate. The full eigenstate residual is recorded separately.

The two untied open runs reach the same stationary error, 65.551776 mEh,
at 92 and 132 total passes. Untied periodic runs reach 48.356272 mEh at
1112 and 892 passes. The initial ring-tied continuation reached 2012 passes
without satisfying both criteria; this is an unfinished optimization, not
converged finite-D accuracy. The continuation driver now enforces its
maximum pass count without exceeding the final block budget.

The optional ``--polish-only`` mode applies joint-tensor Rayleigh-energy
minimization using SciPy's full BFGS optimizer and exact analytic derivatives
from the dense reference projectors, then performs 20 validation passes.
See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html.
This adaptation has no global convergence guarantee and does not target or
fit the known CASCI eigenvector. Optimizer success alone is not sufficient:
the same energy and tangent criteria determine the stored convergence flag.
Iteration histories, numerical failure messages and checkpoint tensors are
preserved, including unconverged cases. Refinement times should not be
compared to the earlier fixed-budget benchmark as equal-cost timings.

After joint refinement and validation, the ring-tied errors (seed 11, seed
23) are 28.933419 and 2.344288 mEh for open virtual bonds, and 0.023099
and 0.329134 mEh for periodic virtual bonds. All four still fail the
stationarity criterion. Open refinements report precision loss; periodic
refinements reach the 5000-iteration limit. These values must remain
labelled unconverged and cannot replace converged accuracy claims.
All eight final checkpoints were reloaded independently to verify their
stored energies, tangent residuals and variational bounds.

To continue and then refine::

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python examples/qchem/converge_benzene_virtual_pbc.py \
      --resume /private/tmp/benzene_virtual_pbc_final \
      --output /private/tmp/benzene_virtual_pbc_D2_convergence

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      MPLCONFIGDIR=/private/tmp/matplotlib-codex \
      python examples/qchem/converge_benzene_virtual_pbc.py \
      --resume /private/tmp/benzene_virtual_pbc_final \
      --output /private/tmp/benzene_virtual_pbc_D2_convergence --polish-only

Stable dense-reference projector updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PBC benzene reference now uses a direct local projector SVD instead of
forming its Gram overlap matrix before whitening. This avoids squaring
the projector condition number. ``projector_tol=1e-8`` is a relative
singular-value cutoff for dense projectors; ``metric_tol=1e-12`` remains
the relative metric cutoff in the large sparse fallback. These thresholds
refer to different quantities and should not be compared numerically as
identical truncations. Dense tensor components outside the retained
projector range are preserved instead of reset to zero.

The lifted local optimum is mixed with the previous tensor in a two-vector
Ritz minimization. Acceptance checks the energy of the actual contracted
state, rather than only the effective overlap and Hamiltonian matrices.
This is a numerical stabilization of the dense-reference adaptation of
Verstraete, Porras and Cirac cited above, not a canonical ring gauge or a
compressed-environment implementation. It does not change D, fit the CASCI
eigenvector, or modify the active reduced ``SU2LETTA`` solver.

``DenseTiedLETTA.run`` now declares convergence only when both the energy
change is below ``tol`` and the supported local tangent residual is below
``tangent_tol`` (default 1e-7 Eh). ``local_tangent_residual`` and
``max_tangent_residual`` expose the diagnostic. A finite-D stationary state
can have a nonzero full eigenstate residual and a nonzero CASCI error.
The continuation companion now uses this direct-SVD diagnostic; the earlier
stored continuation histories retain their original Gram-supported checks.

``examples/qchem/benchmark_letta_projector_convergence.py`` compares the
old metric update (an explicit validation helper) and the stabilized update
from identical saved, refined ring-tied D=2 checkpoints, for both open and
periodic bonds and both seeds. It uses the same direct-SVD tangent diagnostic
for both algorithms, a fixed extra 400-pass budget, and one numerical thread.
The old Gram threshold could obscure weak physical directions, so the
reported direct-SVD residuals can be larger than the previous continuation
residuals. Results remain subject to nonconvex optimization and numerical
support truncation; lower residuals are not a global-optimum guarantee.

After 400 additional passes, periodic seed 11 has a direct-SVD local tangent
residual of 3.89445e-5 Eh with the old update and 4.22855e-6 Eh with the
new update (9.2-fold reduction). Seed 23 changes from 1.33357e-4 to
1.42843e-5 Eh (9.3-fold reduction). Final CASCI errors are respectively
0.023077 and 0.328805 mEh with the new update: energy gains are small.
Open seed 23's residual improves about threefold, but open seed 11's
residual is slightly worse and remains stalled. None of these four
checkpoints satisfies the 1e-7 Eh stationarity threshold. The change
improves numerical convergence for some runs, rather than resolving
all tied-tensor local minima or conditioning problems.

Focused regressions cover a weak real/complex virtual channel that would
be discarded by the old Gram metric, actual variational update energies,
the complex sparse fallback, and rejection of false energy-plateau
convergence. Existing open/periodic cached sweep and trace tests also pass.

Conditional virtual-bond QR on the closed ring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DenseTiedLETTA.condition_virtual_bonds`` applies invertible QR absorption
separately for physical labels shared by neighboring tensors. It includes
the last-to-first bond when ``virtual_boundary='periodic'``. Forward passes
factor the left block and absorb its transfer matrix into the right block;
backward passes do the reverse. The optional scalar rebalance has unit
product, preserving unnormalized amplitudes. Rectangular, singular or
ill-conditioned transfers are skipped without truncation. Full-pass
amplitude and energy checks roll back all transformations if preservation
fails. Counts, closing-bond applications and preservation errors are stored
in ``gauge_info`` and sweep history.

This adapts the QR gauge absorption described by U. Schollwoeck, *The
density-matrix renormalization group in the age of matrix product states*,
Ann. Phys. 326, 96--192 (2011),
https://doi.org/10.1016/j.aop.2010.09.012, to shared-label blocks and finite
virtual rings. It conditions local tensor coordinates, not a globally
canonical periodic norm metric. The final seam absorption can change
blocks orthogonalized earlier in the pass. Generalized/projector metric
handling remains necessary. Enable it with
``run(conditional_gauge=True)``; conditioning occurs before constructing
cached environments for each pass. It remains opt-in.

``examples/qchem/benchmark_letta_conditional_pbc.py`` compares scalar-only
and conditional QR conditioning with the same stabilized local optimizer,
identical initial PBC ring-tied D=2 checkpoints, two seeds, 400 additional
alternating passes, and one numerical thread. Each seed applied 9600
conditional blocks, including 1600 closing-bond blocks, with no skips or
rolled-back passes. Maximum relative raw-amplitude changes were 4.6e-12
and 3.1e-11; gauge-only energy changes were below 1.5e-16 Eh in the shifted
Hamiltonian. Supported local ranks remained [220,64,64,64,64,16].

Final errors for scalar versus conditional conditioning were respectively
0.0230770155 versus 0.0230770157 mEh (seed 11), and 0.328805250 versus
0.328805223 mEh (seed 23). Residual trajectories were essentially identical.
The tensor Frobenius-norm product decreased by about 33% and 41%, but
conditional QR added about 8% runtime in this benchmark. Neither seed
reached the 1e-7 Eh supported tangent threshold. Thus this gauge is valid
and improves tensor balancing, but it did not improve convergence in these
already SVD-conditioned runs. These are number-projected dense references,
not a periodic extension of the reduced SU(2) solver.

Tests verify complex forward/backward seam transformations, raw amplitude
and energy preservation, multiple shared labels, singular-transfer skipping,
failed-check rollback, and agreement of cached conditioned passes with
explicit uncached updates.

Fixed-bond two-site periodic reference updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DenseTiedLETTA.run(update='two-site')`` visits adjacent virtual bonds,
including the last-to-first pair on a periodic ring. ``pair_tensor`` merges
the two factors with every shared physical label present only once, and
``pair_projector`` contracts the remaining ring into an exact reference
map. A direct SVD extracts its supported subspace for the merged Ritz solve.
The merged tensor is then split separately for each shared physical context,
retaining the original internal bond dimension. Twelve backtracking strengths
are compared using the actual recompressed energy. Finally, both retained
factors undergo a joint analytic-gradient L-BFGS-B energy refinement (40
iterations per pair, no bounds). The old pair is retained if no candidate
lowers its energy. Tensor shapes and the physical tie graph remain fixed.

This is an adaptation of the merged optimization and SVD splitting in
U. Schollwoeck, *The density-matrix renormalization group in the age of matrix
product states*, Ann. Phys. 326, 96--192 (2011),
https://doi.org/10.1016/j.aop.2010.09.012. The joint nonlinear factor refinement
uses SciPy's implementation of C. Zhu, R. H. Byrd, P. Lu and J. Nocedal,
*Algorithm 778: L-BFGS-B*, ACM Trans. Math. Softw. 23, 550--560 (1997),
https://doi.org/10.1145/279232.279236. Conditional tensor-Frobenius compression
is not a physical Schmidt truncation on the closed ring; ``discarded_weight``
records a tensor-Frobenius fraction, not discarded wavefunction probability.
The method carries no standard DMRG truncation or global convergence guarantee.
Convergence still requires the energy-change and supported one-site tangent
criteria together. This enumerated small-system reference does not implement
two-site growth in the active reduced SU(2) solver.

``examples/qchem/benchmark_letta_two_site.py`` resumes both stalled PBC D=2
benzene checkpoints with identical localized cc-pVDZ CAS(6,6) integrals and
400 number-sector configurations. Both methods use forward circular passes
and one numerical thread. The script records true errors, supported tangent
residuals, optimization time, fixed tensor shapes, and accepted pair updates;
it saves checkpoints and reproducible convergence figures.

After 100 additional circular passes, seed 11 has CASCI errors of
0.0230841 mEh (one-site) and 0.0230915 mEh (two-site), with supported
tangent residuals 6.05e-6 and 1.66e-5 Eh. Seed 23 has errors
0.328919 and 0.329094 mEh, with residuals 1.83e-5 and 6.04e-5 Eh.
Two-site accepted 541/600 and 600/600 pair updates respectively, but made
less progress toward stationarity. Optimization times were 5.25 versus
25.56 seconds and 5.25 versus 28.85 seconds, excluding diagnostics and
file output. Neither method reached the 1e-7 Eh supported tangent threshold.
Thus fixed-D conditional two-site compression and factor refinement did not
resolve these stalls and cost about five times more in this comparison.

Regressions check complex merged maps and conditional splits with multiple
shared labels, periodic closing pairs, a representable three-qubit exact
ground state, the rank-one Bell-state variational bound, monotone accepted
energies, and the joint complex factor gradient against finite differences.

Safeguarded Anderson acceleration of one-site sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DenseTiedLETTA.run(anderson_memory=6)`` keeps the one-site local optimizer
and fixed tensor shapes, then extrapolates from recent complete sweep maps.
Zero memory (the default) disables acceleration. Each sweep input and output
is normalized, phase-aligned, and brought toward the starting virtual frames
by shared-label unitary Procrustes absorption, including the periodic seam.
These transformations preserve amplitudes up to the chosen overall phase;
they do not create a canonical periodic norm metric. Separate histories are
used for forward and backward passes and reset on each ``run`` call.

The method adapts H. F. Walker and P. Ni, *Anderson acceleration for
fixed-point iterations*, SIAM J. Numer. Anal. 49, 1715--1735 (2011),
https://doi.org/10.1137/10078356X. A regularized direct least-squares solve
minimizes the affine combination of tensor-coordinate sweep residuals.
Complex residuals use real coefficients with stacked real/imaginary rows.
Extrapolation weights are damped to bound their affine coefficient norm by
``anderson_max_coefficient`` (default 20). Twelve backtracking strengths
are checked with the actual contracted
energy; the completed ordinary sweep is retained if no extrapolation lowers
it. ``history[*]['anderson']`` records acceptance, memory, coefficient norms,
step strength, and energies. This is an energy-safeguarded adaptation of
Anderson mixing, not a reproduction of its unconstrained fixed-point
algorithm. Neither a global optimum nor the reference convergence guarantees
are established for tied rings. Energy and supported tangent convergence
criteria remain required together.

``examples/qchem/benchmark_letta_anderson.py`` compares plain and accelerated
one-site circular sweeps from identical stalled benzene PBC ring-tied D=2
checkpoints, using the localized cc-pVDZ CAS(6,6) reference integrals and one
numerical thread. It restarts both drivers in blocks of 20 passes, recording
errors, tangent residuals, actual optimization time, and acceptance counts.
Saved checkpoints and PNG/PDF figures are generated by the script. This
remains a dense number-projected reference; active reduced SU(2) growth is
unmodified.

Tests verify raw complex amplitude preservation under shared-label unitary
absorption, scalar fixed-point acceleration, rejection of lower-coordinate-
residual proposals that raise the physical energy, and fixed shapes with
monotone complex one-site sweeps and separate direction histories.

After 200 additional passes, plain versus accelerated errors were
0.0230774 versus 0.0230565 mEh (seed 11) and 0.328839 versus 0.328789 mEh
(seed 23). Seed 11's supported tangent residual improved from 5.06e-6 to
3.20e-6 Eh; seed 23's worsened from 1.56e-5 to 1.71e-5 Eh. Acceleration
accepted 179 and 137 of the 200 proposals. Optimization times were
6.48 versus 7.51 seconds and 6.45 versus 7.47 seconds, about 16% overhead,
excluding output and diagnostics. Neither checkpoint reached the 1e-7 Eh
stationarity criterion. The energy improvement is modest and does not
establish consistent improvement of stationarity or time to convergence.

For longer continuation, ``anderson_residual='wave'`` fits normalized,
phase-aligned wavefunction sweep differences while mixing the same aligned
fixed-size tensors. This is a reference-only variant of the tensor-coordinate
Anderson fit, with the same physical-energy safeguard and no convergence
guarantee. ``retain_null=False`` instead uses the minimum-norm dense SVD lift
as the local candidate. Discarded components can affect later variation
spaces even when they do not affect the current supported wave; removing
them is therefore an optimization choice, not a universal gauge fix. The
old state remains in the two-vector Ritz panel, and final acceptance still
checks contracted energy. Sparse Gram fallback does not implement this option.

``examples/qchem/converge_letta_anderson.py`` continues saved accelerated
checkpoints with uninterrupted histories, saves diagnostics and tensors every
100 passes, and stops only when energy change and supported tangent residual
meet their tolerances. It exposes history length, coefficient allowance,
tensor/wave residual fitting, minimum-norm candidates, and circular/alternating
orders. These experiments retain ordinary D=2, number projection and localized
orbitals; they do not change active reduced SU(2) growth.

Continuation from the 200-pass accelerated checkpoints tested uninterrupted
tensor mixing for 2000 passes per seed, wave-residual mixing with a coefficient
allowance of 2000 for 1500 passes per seed, minimum-norm candidates for 1000
passes per seed, and alternating wave-mixed minimum-norm updates for 1000
passes per seed. A separate 500-pass aggressive tensor-mixing pilot used seed
11. These are independent continuations from the same starting checkpoints,
not a single cumulative trajectory. None met the 1e-7 Eh supported tangent
criterion. The smallest final residuals were 2.24e-6 and 9.33e-6 Eh after the
conservative 2000-pass runs. The lowest final errors were 0.0228189 mEh for
seed 11 (alternating minimum-norm run, residual 4.35e-6 Eh), and 0.327896 mEh
for seed 23 (conservative long run, residual 9.33e-6 Eh). Energies decreased,
but no convergence certification follows from these small gains.

``examples/qchem/plot_letta_continuation.py`` plots the saved JSON trajectories
and identifies final checkpoints with the lowest energy and supported local
residual separately. Saved checkpoint energies and residuals were independently
recomputed and their periodic D=2 tensor shapes verified.

Final-audit boundary reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~

At checkpoint stride 1, the parameter-block cycle driver reuses the final
reverse sweep's right Hamiltonian boundaries in its stationarity audit. This
eliminates one right-boundary construction pass; the left boundaries, local
reduced adjoint actions, and conditional norm metrics are still recomputed.
Reuse requires matching state identity, tensor revision, MPO identities, sweep
direction, and a complete boundary set. Otherwise the audit rebuilds boundaries.
The cached payload remains available for the next sweep, but decoded channel
buffers are released after audit use. This changes neither the retained metric
criterion nor the independent local-action verification. It is an implementation
optimization of the reduced formulation above, not a new convergence guarantee.

The closed-state audit also consumes conditional norm metrics one site at a
time: it computes that site's whitened residual, then releases the metric before
constructing the next. Previously all site metrics remained live until the norm
pass completed. This bounds retained metric storage by the largest site's metric
rather than their sum; it does not bound an individual metric, boundary metadata,
or contraction workspace. Hamiltonian action vectors are released as their
corresponding residuals are evaluated. The total Rayleigh quotient still comes
from site zero and is shared by every residual.

A norm-only check on the saved pyrene CAS(16e,16o), D=2 state measured
970,560 bytes summed conditional-metric payload versus 262,944 bytes for the
largest site (0.93 versus 0.25 MiB). Norm actions and whitened norms agreed to
1e-13 tolerance. Separate retained/streamed processes took 3.11/3.19 seconds
and peaked at 492.4/467.2 MiB RSS. These single-run process measurements include
loading and allocator effects; the small metric payload reduction does not
explain the full molecular run's approximately 1.6 GiB peak. Hamiltonian
boundary and contraction storage remain the primary optimization targets.

Tracing allocations during right-boundary planning on that same saved pyrene
state (sites 15 through 10, zero-based) identified substantial route-object
storage. At site 10, 343,049 routes reached 171.0 MiB traced planning peak and
108.0 MiB live allocations at planner return. Packed tensor pools occupied
16.0 MiB. Route tuples and repeated output-key tuples accounted for 26.0 and
20.9 MiB, respectively, in the return snapshot. Recoupling arrays and dictionaries
were also substantial. These measurements motivate compact route storage;
they do not establish a full-process memory bound or a production timing result,
since allocation tracing adds overhead.

Sparse boundary transport now asks the reduced route planner for groups of
four signed 64-bit indices per route, keyed once per output block. This replaces
per-route Python tuples and repeated sector-key tuples; it does not change
recoupling coefficients, numerical support, or the summation order within each
output. Dense transport retains its ordered route representation. The sparse
transport diagnostic ``route_index_bytes`` counts compact integer payload only,
excluding dictionary/array headers and allocator capacity. Whole-process savings
must be measured separately.

Repeating the six-boundary pyrene allocation profile with compact routes reduced
site-10 planner peak from 171.0 to 135.5 MiB (20.8%) and live allocations at
planner return from 108.0 to 68.0 MiB (37.0%). Route counts, output counts,
packed tensor-pool bytes, and stored sparse-boundary bytes matched at all six
sites. This checks structural consistency; independent reduced-contraction tests
provide the numerical validation. Whole-process peak RSS in these traced runs
was 904.8 versus 749.7 MiB, including tracer overhead. Production speed and
full-sweep peak memory still require untraced measurement.

An untraced comparison rebuilt the tuple-route sparse transport as an isolated
reference module and ran each representation in a fresh process on the same
checkpoint. All six disk boundary payloads (sites 15 through 10) had identical
SHA-256 hashes. Summed advance time was 47.25 s for tuple routes and 46.99 s for
compact routes: no meaningful speed gain is established by this single pair.
Peak process RSS was 834.0 versus 665.7 MiB, a 20.2% reduction in this boundary
construction check. This includes allocator effects and does not measure a full
optimization sweep or convergence. The reference module and comparison script
are archived with the diagnostic outputs.

The parameter sweep rejects candidates with a nonfinite or above-tolerance
stored-operator residual before performing the costly independent Hamiltonian
adjoint. Such candidates cannot pass either strict acceptance or verified-descent
acceptance. Their reports have ``independently_verified=False`` and
``verification_skipped='operator_residual'``; residual/energy diagnostics in
those reports are stored-operator values, not fresh independent measurements.
Every potentially acceptable candidate still receives independent verification
when ``verify=True``. Site tensors remain unchanged on rejection, and final
stationarity audits are unaffected.

Set ``audit_each_cycle=True`` on ``run_parameter_blocks`` to measure the final
state's tangent residual even before energy stability is reached. This uses
valid reverse-pass boundary checkpoints and records ``audit_elapsed_s`` in
history. It does not bypass the energy-stability or accepted-update gates for
convergence. The default continues to audit only potential convergence cycles.

Prepared MPO local actions
~~~~~~~~~~~~~~~~~~~~~~~~~~

``run_parameter_blocks(..., operator_mode='mpo')`` uses the experimental
``MPOParameterOperator`` instead of assembling the local parameter-space
Hamiltonian. It computes MPO/right-environment factors on demand per sector
pair and left channel, retaining only a bounded subset for subsequent actions.
Davidson applies these factors and the left environment to the lifted tied
tensor, then pulls the result back to parameters. Terms sharing a left channel
are combined before contraction. Prepared factors never spill to disk: the
first implementation exceeded 8 GiB on pyrene and was replaced. The default
factor cache is 32 MiB; a full cache causes recomputation, not allocation growth.
A single active factor is limited to 16 MiB; compensated construction uses up
to four such arrays. Recoupling data are released after each sector pair.
Metadata, boundaries, lifted tensors, and contraction outputs remain additional;
these budgets are not a bound on whole-process RSS.

This is an algebraically exact regrouping of the reduced Wigner--Eckart
contractions of Weichselbaum, Physical Review B 86, 245124 (2012),
https://doi.org/10.1103/PhysRevB.86.245124, adapted to LETTA tied coordinates.
It supports scalar physical multiplicities, including complex parameter
vectors; unsupported physical blocks raise explicitly. Regrouping changes
floating-point sums, so independent verification and final stationarity audits
remain required. The initial implementation uses a constant Davidson
preconditioner, not an explicitly assembled transformed diagonal. It is opt-in:
factor preparation, repeated action cost, and conditioning must be compared
with the assembled path for the actual molecule. No speedup or convergence
guarantee follows from avoiding matrix assembly alone.

The preceding full pyrene cycle with compact routes and audit reuse preserved
all 32 update energies, acceptance decisions, and final tangent residuals
exactly. Peak RSS fell from 1.592 to 1.058 GiB. Total cycle/audit runtime rose
from 71.26 to 80.13 minutes in this comparison, so that run establishes a memory
improvement but no speedup. The maximum final tangent residual remained
0.0666469072; convergence was not established.

Reusing complementary terms in ordinary runs
--------------------------------------------

The ordinary ``state.run()`` local pair-action path retains one set of
materialized reduced complementary terms and their spin-recoupling tables.
The cache belongs to the state and is keyed by the Hamiltonian factor identity
and integral revision. Changing variational tensors rebuilds the numerical
environments, not the unchanged operator terms. Changing integrals replaces
the cached terms. ``close()`` releases the cache, and deep copies omit it.
This preserves the compiled boundary owner and the original dual-channel
conventions. It changes neither the optimizer nor its convergence criteria.
The retained operator tables consume memory until replacement or close;
this optimization does not claim a lower process-memory peak.

On a four-orbital Hubbard ring with a nontrivial ring tie graph, D=2,
seed 9, complementary Hamiltonian, four engine threads and one outer worker,
three fresh runs per version with two requested cycles give the following
median timings (BLAS threads fixed at one):

======================= ============ ===========
Optimizer               Before       After
======================= ============ ===========
Default one-site        0.566 s       0.251 s
Two-site                2.965 s       3.040 s
======================= ============ ===========

The gain is in the default one-site path; the two-site comparison shows no
speedup. The one-site energy remains 3.5505102572168217, above the ground
state: this is a fixed-work comparison, not a ground-state accuracy claim.
Both two-site versions reach -2.102748483462075, agreeing with the independent
Hubbard reference within 1e-13. Final variational parameters and energies
match between versions. Profiles identify repeated operator-term construction
and local-action compilation as the avoidable one-site costs. These results
are distinct from the explicitly selected sparse-frontier backend.

Saved sources, profiles, matched runs, and reproducible benchmark/plot scripts
are in ``/private/tmp/letta-run`` (``compare.py``, ``plot.py``, and
``state-run-comparison.png``). Regression tests cover cached-versus-rebuilt
one-site and two-site updates, integral-revision invalidation, and release
on close.

Boundary support pruning for longer one-site runs
------------------------------------------------

The contextual local-action compiler now excludes bra/ket boundary-sector
pairs absent from either boundary before constructing spin-recoupling cores.
This is exact structural pruning: no numerical cutoff is introduced, no
existing sector is removed, and state-dependent environments are still
rebuilt. The full-chain Wigner--Eckart action reference test covers four- and
six-site tied chains, including complex probe vectors.

Open half-filled Hubbard chains with nearest-neighbor ties, U/t=4, D=2,
seed 9, complementary Hamiltonians, two one-site cycles, four engine threads,
and one BLAS thread give these median run times over three fresh processes
per version. Timings exclude state construction and profiling:

================= ============ ============ ===========
Sites             Before       After        Speedup
================= ============ ============ ===========
8                 2.983 s      2.329 s      1.28x
12                19.264 s     12.240 s     1.57x
================= ============ ============ ===========

Both versions give energies 4.508369918110783 and 5.242666036405986,
respectively. Final parameter vectors, cycle energies, local residuals, and
accepted/rejected update counts match exactly in all three repeats.
These fixed-D, fixed-work runs are not ground-state converged;
the optimization accelerates preparation without fixing one-site trapping.
The eight-site profile reduces angular-core construction calls from 5556 to
1547 and contextual compilation time from 4.664 to 2.129 seconds. Numerical
environment construction and remaining local recoupling still dominate.

Saved baseline sources, profiles, matched histories and parameter vectors,
and reproducible benchmark and plotting scripts are in
``/private/tmp/letta-one-site-chain`` (``bench.py``, ``matched.py``,
``plot.py``, ``matched.json``, and ``hubbard-one-site-comparison.png``).

Constructing only the active bond's environments
------------------------------------------------

Standalone one-site pair actions now request only the boundary prefixes
needed by the active bond. For an N-site chain, each Hamiltonian or metric
build advances N-2 boundaries instead of 2(N-1). Entries outside those
prefixes are unavailable, so this partial chain is restricted to standalone
local solves; full-chain and moving-sweep construction retain their existing
behavior. Every prefix is rebuilt from the current tensors. No numerical
boundary is reused across tensor updates.

Tests compare complex local actions with full-stack construction at every
bond, for explicit MPO and complementary representations, before and after
a tensor update. This reduces construction work and retained Python boundary
objects; it is not a measured process-memory reduction.

With the same Hubbard settings as above, three fresh runs per version give
median times of 2.328 to 1.975 seconds for eight sites (1.18x), and 12.136 to
9.835 seconds for twelve sites (1.23x). Parameters, energy histories,
accepted/rejected update counts, and local residual histories match exactly.
The fixed-D convergence limitation remains. Local spin recoupling and action
compilation still dominate the remaining time.

Saved sources, profiles, matched results, and reproducible benchmark/plot
scripts are in ``/private/tmp/letta-one-site-boundaries``. The diagnostic
figure is ``hubbard-one-site-comparison.png``.
