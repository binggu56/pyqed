# Non-Abelian DMRG Design Notes

This note records the implementation direction for the SU(2)-adapted DMRG
prototype in `pyqed.mps.nonabelian` and `pyqed.qchem.dmrg`.  The immediate goal
is a reduced spatial-orbital quantum-chemistry DMRG path that keeps SU(2)
Wigner-Eckart structure in the tensors and MPO virtual channels instead of
expanding every spin component.

## Reference Implementations

### TensorKit.jl and MPSKit.jl

TensorKit.jl is the closest architectural reference for the tensor core.  It
represents non-Abelian vector spaces, fusion trees, reduced tensor blocks, and
symmetry-aware tensor factorizations as first-class objects.  MPSKit.jl builds
MPS algorithms on top of those tensors.

Implementation lessons for pyqed:

- Keep reduced tensor data separate from Clebsch-Gordan/fusion metadata.
- Track fusion trees explicitly instead of relying on implicit dense axes.
- Make SVD/canonicalization operate on reduced block sectors and preserve
  multiplet degeneracies.
- Treat recoupling as a tensor-layout operation, not as ad-hoc reshaping.

Key references:

- J. Haegeman, J.utho et al., TensorKit.jl documentation,
  https://quantumkithub.github.io/TensorKit.jl/stable/
- MPSKit.jl documentation,
  https://quantumkithub.github.io/MPSKit.jl/stable/

### block2

block2 is the closest reference for the quantum-chemistry Hamiltonian MPO.  It
does not build the SU(2) quantum-chemistry MPO by enumerating all spin-orbital
strings.  It uses spin-adapted irreducible operators and complementary
operators on the MPO virtual bonds.

The relevant operator families are:

- one-index doublet channels such as `S_i` and `R_i`
- pair channels such as `A_ij` and `P_ij` in rank-0 and rank-1 sectors
- particle-hole channels such as `B_ij`, `Q_ij`, and related rank-0/rank-1
  channels
- scalar channels for `H` and `I`

Implementation lessons for pyqed:

- The current `AutoMPO.add_reduced_string(...)` primitive is useful for testing
  Wigner-Eckart recoupling, but a production QC MPO should group ERI
  contractions into complementary operators.
- MPO virtual channels should carry operator families and SU(2) ranks, not one
  independent channel per four-fermion integral.
- Spin-summed one-body terms should be represented as scalar-coupled
  `a^\dagger x a` with the correct `sqrt(2)` normalization.
- Two-body terms should use rank-0/rank-1 pair and particle-hole intermediates
  with the same normalization conventions as the documented block2 formulas.

Key references:

- block2 SU(2) Hamiltonian theory,
  https://block2.readthedocs.io/en/latest/theory/su2.html
- block2 quantum-chemistry Hamiltonian tutorial,
  https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html

### CheMPS2

CheMPS2 is an older spin-adapted quantum-chemistry DMRG code.  Its structure is
less general than TensorKit/MPSKit, but it is useful as a second reference for
spin-adapted complementary-operator Hamiltonian construction, CAS integration,
and reduced density matrix interfaces.

Implementation lessons for pyqed:

- Keep a clear quantum-chemistry layer above the tensor algebra layer.
- Reuse complementary operators across sweeps instead of rebuilding full local
  Hamiltonians from raw ERIs.
- Design RDM and CASCI/CASSCF integration around spatial orbitals and target
  total spin.

Key references:

- S. Wouters et al., CheMPS2: a free open-source spin-adapted implementation of
  the density matrix renormalization group for ab initio quantum chemistry,
  Computer Physics Communications 185, 1501-1514 (2014).
- CheMPS2 repository, https://github.com/SebWouters/CheMPS2

### TeNPy

TeNPy is a useful reference for symbolic MPO graph construction and automatic
Jordan-Wigner string handling, but not for SU(2)-reduced Wigner-Eckart tensors.
Its public tensor machinery is based on Abelian charge sectors rather than full
non-Abelian SU(2) fusion trees.

Implementation lessons for pyqed:

- A graph-based MPO builder is useful for compressing symbolic product terms.
- Automatic Jordan-Wigner handling should live in the model/builder layer.
- TeNPy is not the model to follow for non-Abelian reduced tensor storage.

Key references:

- TeNPy model documentation,
  https://tenpy.readthedocs.io/en/stable/reference/tenpy.models.model.CouplingModel.html
- TeNPy MPO model documentation,
  https://tenpy.readthedocs.io/en/v1.0.3/reference/tenpy.models.model.CouplingMPOModel.html

## Current pyqed Status

### Shared storage architecture

The repository now has one tensor vocabulary. `pyqed.symmetry.Leg` owns axis
orientation, dimensions, and optional irreducible sectors, and
`pyqed.symmetry.IrrepTensor` owns dense, Abelian-block, or reduced non-Abelian
data. Dense finite MPS/MPO cores are one-block tensors under this interface;
the Abelian direct implementation is an optimized subclass; SU(2) tensors add
fusion and recoupling metadata without changing the owner API. NARG operator
spaces and `UniformMPS` use these same objects.

`pyqed.mps.MPS` and `pyqed.mps.MPO` are the only public finite-chain owners.
The symmetry namespace re-exports them. Specialized non-Abelian `MPOCore`,
`IrreducibleMPO`, and `RankCoupledMPO` objects describe individual site cores
and virtual operator channels; they are not competing chain owners. Physical
site descriptors remain in `state.sites`, and numerical cores are in
`state.tensors` (also available as `state.factors`).

Implemented pieces:

- `ReducedTensorOperator` for local irreducible spatial-orbital fermion tensors.
- `RankCoupledMPO` virtual channels with optional Clebsch-Gordan recoupling.
- `AutoMPO.add_reduced_string(...)` for explicit Wigner-Eckart strings used by
  model Hamiltonians, LETTA, and reference tests.
- A production quantum-chemistry Hamiltonian carrier with persistent
  `S/R/A/P/B/Q` normal/complementary families built from the active-space
  integrals.
- A compiled moving environment that owns boundary propagation, complementary
  local actions, reduced Davidson solves, and symmetry-preserving bond splits
  for the complete half-sweep.
- Fully reduced ground-state and state-averaged SU(2) sweeps with reduced
  multiplet bond limits.
- A strict public SU(2)-QCDMRG route: the compiled extension and compiled
  integral builder are required, obsolete Python-backend controls are rejected,
  and native-readiness failures raise instead of falling back to Python bond
  callbacks.

The generic `pyqed.mps.nonabelian` Python contractions remain available for
model systems, LETTA, and exact validation. They are not a fallback for the
production quantum-chemistry driver.

Known limitations:

- SU(2)-QCDMRG has no pure-Python execution mode; a source-only installation
  cannot run this backend.
- The production route currently requires fully reduced spatial-orbital sites
  and interprets `D` as a reduced multiplet limit.
- The general reduced tensor layer is not yet a TensorKit-style fusion-tree
  object model throughout every decomposition and canonicalization routine.

## Recommended Next Steps

1. Benchmark larger active spaces and thread counts against block2, including
   peak memory and convergence by complete sweep cycles.
2. Extend the compiled normal/complementary owner to additional non-Abelian
   site models without weakening the strict production contract.
3. Move the remaining general reduced tensor factorizations toward explicit
   fusion-tree metadata.

### Direct compiled NC with continuous orbital ranks

Rebuild the extension with `PYQED_EXTENSION_GROUPS=mps` when installing from
source. The direct route requires the compiled `_su2_kernel` and has no
Hamiltonian fallback.

Supported finite point groups, Coov, Dooh, and SO3 now default to direct normal/complementary
(NC) execution on complete orbital-multiplet sites:

```python
solver = DMRG(mf, ncas, nelec, D=16, symmetry="su2").run(
    wfnsym=target)
```

`MultipletDMRG.from_integrals(...)`, `ReducedMultipletDMRG.from_integrals(...)`,
`SpatialSymmetryDMRG.from_integrals(...)`, and `NCMultipletDMRG.from_integrals(...)`
all default to compiled NC two-site sweeps for C1, Cs, Ci, C2, C2v, C2h, D2,
D2h, D3h, D6h, Coov, Dooh, and SO3. The qchem DMRG driver retains its
optimized direct compiled NC path with packed one-dimensional labels for
finite Abelian groups; D3h/D6h use the complete-shell extension. For supplied integrals, `contractions="reduced"` or
`contractions="components"` explicitly selects an MPO reference. The explicit
`spatial_method="multiplet"` selects the reduced-MPO reference. No Hamiltonian
MPO, operator-train fit, or molecular determinant-space projection occurs in
NC construction or sweeps. The earlier family-to-MPO adapter was removed.

The existing compiled `System` owns the molecular integral/family arenas.
`configure_spatial_shells` builds canonical NC schedules from the one-body
arena and the screened P coefficients. Normal channels carry at most two
consumed fermion indices; complementary channels carry at most one outstanding
index. The equivalent R/Q delta correction is included exactly once by normal
ordering the P operators. Integral coefficients enter at a single normal-to-
complementary crossing; structural growth and closure routes are shared.
Completing a channel includes all its spin/orbital partners and their
structural recursions, including channels with zero screened coefficients.

`orbital_rank.hpp` supplies exact integer-L / cylindrical fusion, with both
reflection channels at zero cylindrical angular momentum and g/u parity for
Dooh. No angular cutoff or partner selection is introduced. One-/two-fermion
channel maps use analytic product-group CG isometries, including spin duality
for complementary channels. Bounded shell primitives use the existing local
fermion algebra and the shared Wigner--Eckart operator basis. C++ accumulates
the integral-weighted, multiplicity-only rank transitions. `MovingEnvironment`
owns complex reduced boundary arenas, executes both update directions, and
applies sparse local Hamiltonians and their exact diagonals. Local actions
retain their boundary snapshots and reject stale operator representations.
The compiled sweep driver controls both sweep directions, reduced norm and
excited-state overlap contractions, and LAPACK sector QR/RQ gauge moves.
Local solves use the existing C++ Davidson implementation directly with sparse
NC actions, without assembling local Hamiltonian matrices. Checked diagonal
norms use scaling; noncanonical metrics use bounded local LAPACK whitening.
Excited-state orthogonality uses local constraint SVD. Dense metric and
constraint workspaces are limited to 256 MiB. Only rank-one/two local Ritz
problems and Davidson subspaces use dense Hamiltonian matrices. These are
local reduced variational spaces, never molecular determinant projections.
Python prepares symmetry topology and CG maps and exports reduced states;
there is no Python numerical sweep or eigensolver fallback.

This is an adaptation of the complementary-operator formulation of
G. K.-L. Chan and M. Head-Gordon, *Highly correlated calculations with a
polynomial cost algorithm: A study of the density matrix renormalization
group*, J. Chem. Phys. **116**, 4462–4476 (2002),
https://doi.org/10.1063/1.1449459, to complete shell sites. Spatial/spin rank
factorization adapts S. Singh and G. Vidal, *Tensor network states and algorithms in the presence of a global SU(2)
symmetry*, Phys. Rev. B **86**, 195114 (2012),
https://doi.org/10.1103/PhysRevB.86.195114. The Hamiltonian representation is
exact after integral screening and structural roundoff thresholds; the
finite-D one-site variational solver has no guarantee of finding the global
minimum. This is not an exact reproduction of the original implementation or
block2's optimized channel layout.

Supported inputs are real invariant integrals and complex reduced state
tensors under the supported finite groups and Coov/Dooh/SO3. Boundaries are open. Spin-orbit coupling
and complex integrals are unsupported by this NC path. Physical-shell construction retains the existing 512 MiB guard; NC
local primitive and operator-basis workspaces have 128 MiB guards. Large
angular shells can exceed these resource limits even though rank fusion has
no angular cutoff. Omitted threads use `pyqed.num_workers()` with serial outer
Python calculations and BLAS threads limited to one.

Validation covers exact small fermionic Hamiltonians, all local CG partner
rows, both environment directions, complex state tensors, local actions and
diagonals versus reduced-MPO references, one-/two-/three-electron states,
singlets and triplets, multiple excited multiplets, and actual returned partner
expectation values. Solver tests forbid Hamiltonian MPO construction and
component/global projection during direct NC sweeps.

Run the matched molecular benchmark with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=. \
python examples/qchem/benchmark_continuous_dmrg.py --molecules Be LiH Li2 --D 16
```

It uses packed s8 AO integrals and CAS(2e,8o), CAS(2e,10o), and CAS(2e,16o) in
6-31G. Each case compares NC with the MPO solver and an explicitly identified
CASCI validation reference, for scalar targets and complete L1/E1/E1u partner
multiplets. JSON provenance, energies, partner diagnostics, and reproducible
figures are saved outside the repository. Setup-plus-sweep timings use a
shared process in MPO-then-NC order with structural caches reused; these are
not isolated cold-start timings or evidence for many-electron scaling.

### Direct NC controlled bond expansion

`run(algorithm="one_site_cbe", cbe_expansion=4, cbe_preselection=16)`
selects compiled shrewd CBE. The integral-owned NC transitions contract
straight into left/right operator halves with a shared open rank channel;
there is no MPO conversion or full two-site Hamiltonian. The existing
four-stage selection is shared with the reduced reference solver and now
executes in the compiled extension, using LAPACK thin QR/SVD. Both sweep
directions expand complete multiplicities, rebuild the changed adjacent
environment, solve the expanded one-site problem with direct NC Davidson,
and truncate site factors with the existing compiled sector decomposition.
`D` caps multiplicities per sector; `cbe_expansion` is a global count of added
multiplets, never a count of individual partner rows. An initial solver with
smaller D may be grown by setting its `D` before `run`.

This adapts A. Gleis, J.-W. Li and J. von Delft, *Controlled bond expansion
for DMRG ground state search at single-site costs*, Phys. Rev. Lett. **130**,
246402 (2023), https://doi.org/10.1103/PhysRevLett.130.246402, supplement
Fig. S-2/Table I. It is not an exact reproduction: preselection budgets count
multiplicities, truncation is per sector, and reduced parent layouts and
bounded open-channel half matrices are retained. The publication's asymptotic
single-site-cost claim therefore does not carry over. Pair and operator-half
workspaces have explicit 256 MiB guards. CBE requires canonical external
metrics and currently optimizes one complete spatial multiplet. Excited
multiple-multiplet runs remain available with fixed-D one-site sweeps.
Finite-D convergence does not guarantee the global ground state.

`min_sweeps` on NC prevents energy-based early termination for sustained
validation; its default is one. `examples/qchem/benchmark_nc_stability.py`
compares four-electron 6-311G active spaces against explicit reduced-MPO
references, seeds NC at smaller multiplicities, and records forced sweeps,
local residuals, covariance, orthogonality, energy drift and resident memory.
It writes provenance hashes, JSON records, and a diagnostic figure outside
the repository. Run with the repository's BLAS limits and central NC thread
budget; use `--sweeps 50` for sustained validation.

Invariant CG route plans are cached by site and bra/ket irrep keys, independent
of multiplicity dimensions and state values. The cache is bounded to 64 plans
and 200,000 routes, and is released by `close()` along with NC boundary arenas
and CBE decomposition workspaces. Numerical environments are never reused as
structural data across a changed site update.

The stability benchmark records resident memory separately from in-use
allocations: macOS can compress and reactivate allocator pages without a
change in live storage. On macOS 26, `malloc_zone_statistics(NULL, ...)` can
rise during a loop that frees every buffer, so it is recorded only as allocator
accounting, not as live storage. The benchmark-only `_allocation_probe.c`
enumerates `MALLOC_PTR_IN_USE_RANGE_TYPE` at quiescent checkpoints. Its recorder
does not allocate or call Python, and the Python binding keeps the GIL. This
includes C++ and NumPy malloc allocations, but excludes non-malloc mappings,
including Python's mmap-backed small-object arenas; RSS and Python allocation
tracing provide separate checks for those. The benchmark checks late in-use
growth/range, the RSS envelope, stable boundary counts, and the bounded route
cache. It also evaluates every returned NC
partner with the independent MPO Hamiltonian, adding the same frozen-core
and nuclear energy offset to both total-energy comparisons.

For larger independent MPO references, the blocked suffix-QR constructor
stores oversized range bases and suffix-coordinate arrays in anonymous,
column-major mapped files using the existing RI workspace lifetime pattern.
Projections and reconstruction checks stream through bounded column blocks;
suffix-QR numerical blocks retain their 128 MiB guard, with a separate
2 GiB disk limit per mapped array. This changes storage only and is checked against
explicit real/complex unfoldings and a subsequent streamed contraction.
The direct NC solver does not call this reference constructor.

The benchmark's `--resume` option preserves completed verified jobs and
archives the prior provenance before running unfinished targets. The companion
`examples/qchem/summarize_nc_stability.py OUTPUT_DIRECTORY` checks all six
partner counts, source/extension hashes and independent expectations, and
plots allocator accounting separately from resident memory. Historical runs
using allocator statistics require a matching `benchmark_nc_memory.py` audit
of both largest Li2 targets before the summary accepts their memory validation.
That audit verifies the original computational-source and extension hashes,
runs 50 further cycles per target, and checks actual in-use ranges against a
64 KiB/cycle late-slope and 4 MiB late-range limit. Original numerical results
and archived instrumentation sources remain unchanged.

The 2026-10-24 macOS ARM64 validation used packed s8 AO integrals, 6-311G,
four active electrons, seed multiplicity 2, final per-sector D=16, expansion
4 and preselection 16. Every target completed 50 forced full sweep cycles
(300 total), using the central one-thread budget reported on that machine
and one BLAS thread:

| Molecule / group | Active space | Targets (retained partners) | Largest NC–MPO energy difference (hartree) |
| --- | --- | --- | --- |
| Be / SO3 | CAS(4e,13o) | L0 (1), L1 (3) | 2.67e-13 |
| LiH / Coov | CAS(4e,16o) | A1 (1), E1 (2) | 5.18e-10 |
| Li2 / Dooh | CAS(4e,25o) | A1g (1), E1u (2) | 6.76e-11 |

All ten returned partners were evaluated with the independent MPO; the
largest expectation discrepancy was 2.53e-13 hartree. Covariance and
orthogonality errors were below 3e-15, local Davidson residuals below 1e-11,
and late energy spreads below 2.14e-14 hartree. These are same-D reference
comparisons, not a claim of finite-D exact-CAS convergence. The numerical
records and source snapshots are written by the benchmark scripts above.

An additional 50-cycle in-use-allocation audit of each Li2 target (100 more
cycles, with the same compiled solver) found a final-20-cycle range of zero
bytes for A1g and 66,048 bytes for E1u, with no positive late trend. The relevant
regression set passed 373 tests with three skips, including explicit CBE
dispatch through qchem and all four continuous-symmetry factories, plus tests
that the memory probe counts live buffers and excludes freed-buffer churn.

### Direct NC two-site sweeps and CBE timing comparison

`NCMultipletDMRG.run(algorithm="two_site")` performs a finite-system two-site
optimization for one complete supported finite or continuous target multiplet. It enumerates
all allowed intermediate product-group irreps, contracts the existing direct
NC channel halves, and solves the metric-scaled pair problem with compiled
Davidson. A compiled sector SVD retains at most `D` multiplicities per irrep;
all rows of each retained irrep survive together. Both sweep directions rebuild
and advance direct NC environments. No Hamiltonian MPO, determinant projection,
or full two-site Hamiltonian matrix is constructed.

This adapts the two-site finite-system formulation reviewed by U. Schollwoeck,
*Annals of Physics* **326**, 96–192 (2011),
[doi:10.1016/j.aop.2010.09.012](https://doi.org/10.1016/j.aop.2010.09.012),
using the same NC partition and Davidson references as the one-site solver.
The truncation budget is per sector, not a global discarded-weight guarantee.
Pair layouts and channel halves are explicitly stored within the existing
256 MiB workspace guards; matrix-free refers to the Hamiltonian action, not
zero pair storage. Real molecular integrals and complex state coefficients are
supported. Multiple excited target multiplets and spin-orbit coupling are not
supported by this two-site option. The local solver requires canonical diagonal
norm boundaries and rejects invalid metrics.

`examples/qchem/benchmark_nc_two_site.py` compares direct NC CBE and two-site
sweeps on Li2 CAS(4e,25o), 6-311G, 4 bohr, with packed-s8 AO integrals and one
frozen core orbital. It retains A1g and both E1u partners, uses the same seed,
central thread budget, BLAS=1, per-sector D, and convergence controls, and
checks a common absolute energy tolerance against independently validated MPO
reference energies. Each repetition runs in a fresh process. Total measured
time includes imports, RHF, active-integral transformation, NC setup and sweeps;
partner diagnostics and plotting follow the time/peak-RSS snapshot. Peak RSS
is the OS process high-water mark, including setup allocations and allocator
retention, not just live tensor bytes. Runs stop on convergence rather than a
forced cycle count; failure to meet reference accuracy fails the benchmark.
The reference is finite-D, so this comparison does not establish exact-CAS
accuracy. The script writes JSON provenance, per-run logs and a reproducible
wall-time/memory/convergence figure outside the repository.

The integrated two-site validation run passed 402 tests (3 skipped), including
complex pair actions/diagonals/eigenpairs against independent MPO references,
Coov/Dooh/SO3 bond growth, all returned singlet and triplet orbital partners,
and qchem/factory dispatch with MPO construction forbidden. The portable
pair-action header also passes a non-Accelerate syntax check.

### Li2 matched-accuracy comparison (2026-10-24)

Three fresh-process repetitions per target/method, D=16 multiplicities per
sector (initial D=2), seed 11, one solver thread and one BLAS thread gave:

| Complete Dooh target | CBE total time (s) | Two-site total time (s) | CBE peak RSS (MiB) | Two-site peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| A1g (one partner) | 73.91 | 48.04 | 538.50 | 541.75 |
| E1u (both partners) | 97.85 | 50.33 | 564.88 | 605.34 |

Values are medians; all runs met the common 1e-8 hartree criterion against
the independent finite-D MPO reference (largest energy difference 6.95e-11
hartree). Two-site was 1.54x faster for A1g and 1.94x faster for E1u.
Peak RSS was similar for A1g and about 7.2% higher for two-site E1u; process
allocator variation makes these memory measurements workload-specific.
CBE required seven complete cycles; two-site required five for A1g and four
for E1u, under identical stopping tolerances.

The accepted data are `/private/tmp/li2-nc-cbe-two-site-final/results.json`.
Sources, extension hashes, and reference data are archived alongside them.
The runs used an isolated source copy and private compiled runtime cache after
external workspace/cache changes invalidated earlier attempts. Those aborted
attempts are excluded. The benchmark script generates the comparison figure
and verifies all saved partners with a separate, untimed reference MPO.

All 18 saved partner states across the twelve runs passed independent MPO
expectation checks, with a maximum NC/MPO discrepancy of 1.69e-13 hartree.
After restoring the tested files to the workspace, all 39 continuous qchem
and factory dispatch checks passed; delivered source hashes were unchanged.

Direct compiled NC now defaults to two-site sweeps when `algorithm` is omitted.
Explicit `one_site` and `one_site_cbe` remain available. Two-site targets one
complete spatial multiplet; multiple multiplet roots require explicit
`algorithm="one_site"`. All symmetry partners of the selected multiplet remain.

### Shared sweep control and specialized NC executors

Spin-only SU(2), spin/orbital NC, and discrete-group multiplet reference sweeps now use `SweepControl` for transaction
scheduling, convergence decisions, algorithm labels, and bond-cap reporting.
Both retain the shared C++ Davidson implementation and their specialized
compiled local actions, recoupling, SVD, and environment kernels. The control
callback occurs once per half sweep (SU2) or complete cycle (spatial NC), not
per bond. It does not construct an MPO or project a global Hamiltonian.

The common policy requires every supplied stopping criterion to pass at an
allowed convergence boundary. Spatial NC supplies the full-cycle energy
change and local Davidson convergence status. SU2 supplies its existing
energy/metric or state-averaged-energy criterion. These deliberately distinct
criteria are explicit inputs to one policy; their tolerances are not silently
reinterpreted. A failed spatial two-site local solve now prevents convergence.

`max_bond_mode` uses the same vocabulary in both paths: `reduced` limits total
retained multiplets, `states` counts physical states, and `per_sector` limits
multiplicity in each sector. Spatial NC currently implements only `per_sector`
and rejects other modes. Its D is therefore not numerically interchangeable
with a total reduced-state cap. Existing SU2 truncation modes remain available.

This shares control rather than replacing symmetry-specific numerical
representations. Discrete orbital groups are not newly enabled for spatial NC.
The formulation remains an adaptation of finite-system DMRG reviewed by
Schollwoeck, Ann. Phys. 326, 96 (2011), doi:10.1016/j.aop.2010.09.012;
energy-based stopping does not certify the global variational minimum.

The clean-build regression passed 285 tests; a final cross-solver run passed
119 tests including discrete multiplet partners. Eight failures found during
broader testing were reproduced in the saved pre-change tree: four tiny-workspace
SU2 cases and four legacy local-operator cases. These were subsequently fixed
by the corrections described below.

`examples/qchem/benchmark_sweep_control.py` compares saved and current trees
using one serialized H4 RHF dataset, deterministic four-shell orbital models,
identical loaded Davidson libraries, one thread, and three fresh processes per
case. Common RHF preparation and independent ED verification are outside timing.
The accepted run is `/private/tmp/nc-shared-control-verified/`; earlier attempts
with independently prepared data or mismatched builds are excluded. All six
energies for each case are identical, and H4 agrees with ED within 6e-15 hartree.
Median total worker time / peak RSS before and after were:

| Case | Before (s) | After (s) | Before (MiB) | After (MiB) |
| --- | ---: | ---: | ---: | ---: |
| SU2 H4 | 1.194 | 1.238 | 243.50 | 244.52 |
| Coov | 1.280 | 1.296 | 244.73 | 245.06 |
| Dooh | 1.290 | 1.313 | 246.44 | 246.56 |
| SO3 | 1.788 | 1.749 | 275.63 | 275.81 |

These small workloads test control overhead, not large-system scaling. Timing
changes range from -2.2% to +3.7%; peak RSS changes are under 0.5%.

### Small-workspace convergence and missing physical sectors

The single-root real and complex Davidson implementations now test correction
linear dependence relative to the pre-orthogonalization correction norm and
machine precision. The former absolute 1e-10 cutoff could terminate a valid
near-converged iteration before meeting a tighter requested residual tolerance.
Residual acceptance and workspace limits are unchanged.

When expanding sparse spatial-orbital MPS support, an absent expanded physical
spin sector retains its magnetic dimension. Fully reduced physical sectors
remain one-dimensional in multiplicity space. Missing blocks no longer create
a size-one physical layout against size-two spin operator factors. Existing
virtual-leg multiplicity handling is preserved. No dense/global projection or
Hamiltonian adapter is introduced.

Validation of these corrections: all eight original failures and three added
regression cases pass. The broader tensor/NC/CBE/linear-algebra run passed
327 tests, and the accelerated/portable Davidson binding tests passed all 128
cases (455 tests total). The latter require the local macOS SDK C++ include
path when invoked with the standalone test compiler. No energy tolerance or
workspace limit was relaxed.


### Finite point-group direct NC extension

Finite molecular irreps use exact indexed dimensions and fusion tables in
`FiniteOrbitalAlgebra`; their IDs are not interpreted as angular momenta.
The C++ schedule validates channel rows, fusion selection rules, and dimension
conservation against this table. Python prepares the same exact structural
CG maps used by the independent MPO reference. The compiled NC integral
routes, reduced moving boundaries, matrix-free pair actions, Davidson solve,
and complete-sector SVD are shared with the continuous-group implementation.
No MPO adapter or molecular determinant projection is involved.

The finite CG convention follows the matrix-unit construction in F. A.
Cotton, *Chemical Applications of Group Theory*, 3rd ed., Wiley (1990).
This is an adaptation of the Chan–Head-Gordon NC partition and the
Singh–Vidal product-irrep reduction cited above. Supported groups have real,
self-dual irreps and multiplicity-free binary products. General complex irreps,
outer fusion multiplicities, and spin-orbit coupling are not implemented.
Two-site sweeps target one complete spatial multiplet, including every partner;
this is distinct from requesting multiple excited multiplets. The existing
per-sector D cap and finite-D convergence limitations still apply.

`spatial_method="multiplet"` explicitly selects the independent molecular MPO
reference. `SpatialSymmetryDMRG.from_integrals(..., contractions="group_action")`
selects the earlier group-action reference; its default is now NC.
RDM dispatch uses the complete-multiplet state contractions for both NC and
MPO multiplet states.

`tests/test_nc_finite.py` checks interacting D3h and D6h E1/E2 fusion,
complex pair actions and diagonals, bond growth, singlet/triplet targets,
every returned partner, and default factory/molecular dispatch with fallback
construction forbidden. `examples/qchem/validate_finite_nc.py` reproduces the
small-model numerical comparison and writes JSON results, compiled-extension
provenance, and a diagnostic figure outside the repository.


The finite extension validation passed a 355-test regression suite, 30
finite-specific tests (including every D3h/D6h E irrep and finite-only
invariant ERIs), and 18 additional Abelian regressions. The reproducible
10-case comparison includes H3+ CAS(2e,3o) and benzene CAS(6e,6o); its maximum
all-partner MPO energy difference was 5.0e-16 Hartree with the central
one-thread budget. These are small-space correctness checks, not a claim
of convergence for arbitrary active spaces or bond dimensions.

### NC restart and threaded pair actions

`NCMultipletDMRG.save_checkpoint(path)` writes an atomic, pickle-free NumPy
archive after a completed run. It contains reduced coefficients, complete
irrep multiplicities, the saved per-sector D, the last run's sweep history and completed
sweep count. Reconstruct the identical integral/basis/target problem with
`from_integrals`, then call `load_checkpoint(path).run(...)`; `nsweeps` counts
additional sweeps. Thread count can change. A hash of the numerical problem
rejects incompatible inputs, and the restored state energy is checked before
acceptance. Compiled environments and pointers are rebuilt, not serialized.
Checkpoints currently represent one complete target multiplet at a completed
sweep boundary, not an interrupted Davidson iteration.

Compiled NC pair actions assign disjoint output sectors to OpenMP workers.
Each sector keeps the same channel summation order for every thread count;
BLAS remains limited to one thread. Per-local-solve `pair_threads_used` and
`pair_parallel_calls` record actual execution, rather than just configuration.
The parallel partition changes neither the NC Hamiltonian nor the reduced
variational space. The existing method references and finite-D limitations
apply unchanged.

`examples/qchem/validate_nc_endurance.py` prepares borazine (D3h) and benzene
(D6h) from packed-s8 STO-3G AO integrals. The frozen-core boundary is moved
outward when necessary to keep complete multiplets. Its workers operate on
identical saved active-space integrals in fresh processes, with one BLAS
thread, and record sweep energies, live heap, RSS, retained boundary storage,
route-cache size, and actual pair-action threading. Independent MPO
contractions of every returned partner are separate validation operations.

The larger-space validation exposed the dense storage of open-channel halves
as a two-site bottleneck: a CAS(8e,19o) D=4 run exceeded its 256 MiB guard.
Two-site actions now retain those halves in CSR storage and perform both
sparse contractions in C++, parallel over disjoint output sectors. This is
an exact storage/contraction change with no operator fitting or additional
truncation. The CBE path retains its separate dense-half workspace guard.

Finite-space endurance profiling also exposed repeated construction of
forbidden channel candidates and redundant boundary rebuilds. NC route
planning now applies the exact charge, spin and real-irrep fusion rules
before evaluating recoupling coefficients. Two-site sweeps reuse the valid
prefixes/suffixes built during the preceding half sweep, following the same
moving-environment scheme already used by the one-site multiplet solver.
Tests compare reused boundaries with rebuilding after every full cycle,
including complex tensors and actual bond growth.

The complete finite-group endurance matrix is reproducible with
`PYTHONPATH=. python examples/qchem/validate_nc_endurance.py --suite /private/tmp/nc-endurance-repeat --D 4 --sweeps 32`
under the BLAS thread limits above. Each molecular case runs uninterrupted,
then as two cycles followed by a fresh-process restart for the remaining
cycles. Additional fresh processes continue the same converged checkpoint
with one, two and four compiled threads. The one-thread continuation also
contracts both target partners against an independent MPO. The companion
`summarize_nc_endurance.py` checks complete partner covariance, restart
trajectories and overlaps, actual parallel execution, energy stability,
late live-allocation slope/range and bounded retained environments. It writes
JSON and reproducible energy, memory and thread-reproducibility figures.
These tests qualify the stated finite-D workloads; they do not establish
exact-CAS convergence or unrestricted active-space/bond-dimension scaling.

The endurance validator contracts finite-group rotations using sparse local
component slices and retains all final boundary rows in one contraction.
This avoids cubic dense virtual-bond contractions during validation, while
evaluating the same full partner Gram and rotation matrices. It uses the
existing structural CG threshold and introduces no additional truncation.
Small D3h/D6h tests compare every local slice, partner overlap and symmetry
diagnostic with the dense reference, including complex states. These
component contractions are validation-only; the active NC sweep stays reduced.
The report records a separately converged fixed-D MPO optimization as well as
MPO expectations on NC states. Their optimizer energy difference is reported
separately from Hamiltonian agreement: repeated two-site truncation and
fixed-space one-site optimization need not reach identical finite-D states.

The Linux validation on Westlake used the same source and saved integrals,
with fresh four-cycle CAS(8e,19o) runs at D=4 and one, two and four OpenMP
threads. Both D3h E' and D6h E1u retained their two partners. Thread counts
gave identical reported energies, with maximum partner-overlap error
2.3e-15. Independent MPO expectations on those NC states agreed within
8.6e-14 Hartree. These are within-platform reproducibility checks; different
platforms can follow different finite-D optimization trajectories. Slurm jobs
13276449–13276454 ran the matrix, and 13276455 checked and plotted it.
The 193 selected checkpoint, finite-group, two-site, orbital-rank, multiplet,
CBE, sparse-diagnostic and reference-initialization tests also passed on macOS and Linux. The Linux
suite required a separate rerun of the native benzene fixture after removing
an unnecessary import-time dependency on PySCF.

Compiled NC does not mean an end-to-end C++ execution path. The numerical
environment contractions, pair actions, Davidson solve and sector splitting
are compiled, while structural route planning and boundary assembly still
perform substantial Python work. A short finite-space profile exposed many
sector-hash lookups in that planning; it included initial boundary construction
and is not a steady-state timing breakdown. Endurance validation establishes
correctness and stability for its stated workload, not production throughput.
Here D=4 caps each charge/spin/spatial sector separately, rather than limiting
the total virtual bond to four states.

The 2026-10-25 spin-singlet CAS(8e,19o) endurance runs each completed 32 full
two-site cycles at D=4. Fresh-process restarts after cycle two reproduced all
remaining cycle energies within 2.9e-14 Hartree. Final partner-overlap errors
were 1.6e-15 for borazine D3h E' and 6.7e-16 for benzene D6h E1u. Their
last-eight-cycle energy spreads were 1.64e-9 and 4.42e-10 Hartree respectively;
all local Davidson residuals were below 1.0e-10. The largest retained bonds
contained 149 and 259 reduced multiplets, corresponding to 507 and 928
spin/spatial components. The final-eight-cycle maximum discarded weights
were 5.87e-4 and 7.94e-4, so these are stability tests at finite truncation,
not exact-CAS accuracy claims.

Both runs had zero range in their sampled live malloc allocations over the
last 16 cycles. This measurement excludes Python's mmap-backed small-object
arenas. RSS fluctuated and did not demonstrate a flat plateau; it must not be
described as constant total process memory. Separate Python allocation
tracing complements the malloc and retained-environment measurements.
`examples/qchem/trace_nc_allocations.py` forwards the endurance worker's
arguments and records Python-traced current/peak bytes and allocated-block
counts in an adjacent `.allocations.json` file. It starts tracing after
construction/restart and collects garbage at each measurement. Its instrumented
wall time is not a solver-throughput benchmark.

`examples/qchem/validate_mpo_polish.py` optionally initializes the independent
MPO reference from an NC checkpoint. It copies only reduced coefficients and
bond layouts, verifies the initial energy with the separately constructed MPO,
and runs the reference's one-site optimizer. It is a validation-only workflow;
the active NC solver never calls it. Complex D3h/D6h tests verify all initial
partner overlaps and that the coefficient arrays do not alias the NC state.
