Backends and Integral Representations
=====================================

PyQED supports multiple backend and integral-storage choices. The right choice
depends on whether a calculation needs small-molecule simplicity, dense tensor
access, or factorized contractions for larger systems.

Molecule Integral Build
-----------------------

The molecular build step always uses PyQED's native integral engine:

.. code-block:: python

   mol.build(eri="auto")

Common build choices:

* External packages such as PySCF are optional and are mainly useful for
  validation, comparison, or features that are not yet native; use
  ``mol.topyscf()`` when interoperability is needed.
* ``eri="auto"`` uses compact eight-fold exact storage for small native builds
  and switches to native RI/factorized storage for larger AO spaces when an
  auxiliary basis is available.

Molecular calculations use real-spherical AOs by default:

.. code-block:: python

   mol.build(options={"coord_type": "spherical"})

Use ``options={"coord_type": "cartesian"}`` to request Cartesian AOs
explicitly. Periodic calculations retain their Cartesian default. For a dense
spherical build, the native
integral engine computes one Cartesian shell quartet at a time, transforms that
small block to real-spherical AOs, and writes directly to the spherical output.
It does not allocate an intermediate Cartesian four-index tensor. The same
shell convention and ordering as libcint/PySCF is used through angular momentum
``l = 6`` (i functions). The native Obara–Saika recurrence handles shell-pair
angular momentum through 6, so all quartets composed of s through f shells stay
on the batched recurrence path; rarer higher-pair-momentum g quartets use the
scalar fallback.

The recurrence uses a compact Cartesian-state layout and generates HRR,
primitive-contraction, and sparse spherical target plans once per shell
quartet. Those plans and their worker scratch buffers are reused by later J/K
builds. Supported quartets contract recurrence targets directly into unique
spherical integrals, without materializing a Cartesian shell-quartet tensor;
the scalar high-angular-momentum fallback still uses a shell-local tensor. The
same kernels feed the direct-J/K path, so spherical density matrices are
contracted without a Cartesian density or potential round-trip. Schwarz bounds
are combined with shell-block density bounds before recurrence work, and
``parallel=True`` with ``eri_workers`` partitions shell quartets across native
workers with private J/K accumulators.
Direct builds use a separate ``direct_scf_tol`` (default ``1e-13``), matching
PySCF's direct-SCF control rather than reusing the dense-integral
``eri_screen_tol``. RHF contracts density increments, records aggregate
computed/skipped quartet counts in ``scf_info``, and screens whole s8 shell
quartets before recurrence. Accepted quartets retain the generated fixed-shape
tile contraction; enabling screening does not switch them to a slower
per-output scatter.

Contracted d/f quartets use a two-primitive recurrence batch when the target
block is large enough to amortize the second scratch table. Small blocks retain
the scalar loop. Boys values use a cached small-argument Taylor table and an
asymptotic large-argument branch, while intermediate arguments use the stable
upward recurrence.

``eri_backend="rys"`` selects the native Rys direct-J/K path for spherical
AOs. The C++ kernel constructs one to seven Rys roots and weights,
evaluates the coupled one-dimensional recurrence, applies horizontal recurrence
over the four centers, and uses fixed-shape s/p kernels plus all 81 fixed
s/p/d shell-shape recurrence kernels. The shell-shape and cache-mode dispatch
happens once outside the primitive loops, so the generated cached and uncached
variants contain neither runtime angular-shape tables nor inner cache branches.
The packed Cartesian output contraction is unrolled in four-value groups, and
the s/p specialization uses compile-time axis products. Generated s/p/d kernels
accumulate in fixed stack storage and scatter directly into J/K. Their axis
codes come from fixed Cartesian component tables, bypassing both the packed
output plan and an intermediate quartet vector. The five symmetry-reduced
shell-quartet shapes use generated s8 AO output lists; repeated-shell kernels
therefore evaluate only canonical components and scatter pre-grouped
compile-time symmetry classes without runtime AO equality tests. The sixth,
fully distinct shape already has one canonical AO output per component and
retains its shell-local tiled scatter. The general Cartesian and sparse
spherical f-shell plans carry a precomputed s8 symmetry class for each output.
Fixed-shape HRR arrays use their exact angular dimensions
instead of the general 7-by-4 bounds. Intermediate one-root Boys values use a
piecewise Chebyshev kernel, root loops are generated/unrolled, and ssss
contractions use two independent primitive lanes. Fully distinct quartets
accumulate into shell-local J/K tiles before flushing global matrix updates.
The one-worker driver uses a plain serial cursor instead of an atomic task
counter. Persistent
symmetry-shaped output plans
cover favorable d/f quartets. Same-pair kernels evaluate only
the unique eightfold-symmetric AO outputs needed by direct J/K. Obara--Saika is
retained only for quartets containing g or higher shells. Both kernels share
the planned quartet list, density screening, reusable native worker pool,
thread-private J/K matrices, and deterministic reduction; there is no Python
or Cython orchestration in the production path.

For repeated J/K builds at fixed geometry, the persistent native basis plan
caches the two exact Rys quadrature invariants needed for each retained
primitive-pair root, :math:`u^2` and its geometry-only root prefactor. Shell
contraction weights are applied in the hot loop, allowing contractions with
identical primitive exponents and centers to share one quadrature record. The
small recurrence shifts and coupling coefficients are rebuilt from those two
values, so each cache entry is 16 bytes rather than the former 80-byte expanded
recurrence record. Unique cache records are materialized while the immutable
native plan is built, before parallel workers can read them; a cold parallel
J/K call therefore cannot race on lazy cache population. This
does not cache AO ERIs or a four-index tensor, and the density-dependent J/K
contraction is still evaluated on every call. It avoids repeating Boys/root
solves and invariant divisions during SCF iterations, which is particularly
important for diffuse bases with many s/p shell quartets. The cache has a
256-MiB plan-wide budget. Set ``rys_cache_mib`` in the builtin build options
to trade additional persistent-plan memory for faster repeated J/K builds;
large serial diffuse-basis calculations can benefit from 1024 MiB. Tasks with
more Rys roots are assigned first because
their uncached root solve is more expensive; remaining tasks compute the same
invariants in-place without retaining them. Unplanned one-shot calls do not
populate this persistent cache. Serial and parallel execution share one
cost-sorted shell-quartet list instead of retaining duplicate multi-million
entry schedules. Plan construction groups tasks by root count in one pass,
uses linear-time logarithmic cost buckets, and stores retained quadrature points
in one exact-size contiguous pool instead of separately allocated vectors.
This reduces construction and allocator overhead without increasing
``rys_cache_mib``.

The d/f Rys output plan is a generated per-shape axis DAG: horizontal
recurrence and sparse spherical coefficients are expanded once, and the kernel
evaluates only the requested axis states before direct J/K scatter. Dispatch
uses this DAG only when it contains no more terms than the Cartesian contraction;
dense shapes use a Cartesian fallback packed by angular shape and s8 symmetry
class. A dense-to-packed lookup lets the spherical transform reuse the one
canonical Cartesian value for all equivalent terms, so the fallback does not
evaluate duplicate pair or bra--ket permutations. The Cartesian-to-spherical
direct-J/K fallback transforms the density into Cartesian shell space once,
contracts all fallback quartets directly into Cartesian J/K matrices, and
transforms those two matrices back once. It therefore avoids a four-index
spherical transform for every shell quartet. Dense-ERI and unplanned paths
retain the compiled per-shape coefficient/index transform.
Symmetric-density J/K scatter
is selected once per AO symmetry class and compiled into separate branch-free
specializations. Per-quartet direct-spherical plans use sparse slot tables and a
separate 256-MiB plan budget; quartets beyond that budget use the shared packed
Cartesian fallback rather than allocating an unbounded plan object. Per-task
caches are reused only when the runtime screening threshold matches the task
list from which the persistent plan was built.

``eri_backend="auto"`` uses fixed-shape Rys kernels for s/p quartets through
total angular rank 3 and the C++ Obara--Saika recurrence for higher angular
momentum. This measured hybrid is faster than forcing either recurrence for
the whole shell-quartet set. Primitive-pair geometry and contraction weights
are cached, and the identity s/p spherical blocks feed their completed unique
integrals directly into J/K scatter. The Cython Rys recurrence and older
derivative expansion remain available only as explicit compiled reference
helpers for validation.

Native RI builds transform primary and auxiliary three-center
tensors into spherical pair space before metric factorization, so their stored
factors are already in the requested AO basis.

Electron-Repulsion Storage
--------------------------

The ``eri`` keyword controls the ERI representation family, and ``aosym``
controls AO permutation symmetry for dense-like storage:

* ``eri="dense", aosym="s1"`` stores the dense four-index tensor.
* ``eri="dense", aosym="s4"`` stores unique AO-pair rows and columns.
* ``eri="dense", aosym="s8"`` stores only unique AO-pair-pair values,
  exploiting the full eight-fold ERI permutation symmetry for memory.
* ``eri="direct"`` avoids dense AO ERI construction and uses compact ``s8``
  storage for cartesian J/K builds.
* ``eri="factors"`` stores a Cholesky/factorized representation.
* ``eri="ri"`` stores native density-fitting factors. The default auxiliary
  basis policy prefers JKFIT sets for SCF when available; use
  ``options={"ri_purpose": "ri"}`` to prefer RIFIT sets.
* ``eri="dense+factors"`` stores both dense and factorized representations.

Legacy shortcuts such as ``eri="s8"`` and ``eri="s8+factors"`` are still
accepted and normalize to ``eri="dense", aosym="s8"`` and
``eri="dense+factors", aosym="s8"`` respectively.

When to Use Dense Integrals
---------------------------

Full dense integrals are simplest and useful for:

* very small molecules
* debugging new methods
* algorithms that explicitly require ``(pq|rs)`` tensor access
* reference comparisons against dense implementations

The drawback is memory scaling. Dense four-index tensors become expensive as
the number of orbitals grows. For exact RHF calculations that do not need direct
``mol.eri`` tensor access, prefer ``eri="auto"`` or ``eri="dense",
aosym="s8"`` so J/K contractions use the compact packed path.

When to Use Factorized Integrals
--------------------------------

Factorized integrals are preferred for:

* larger basis sets
* RHF with Cholesky/factorized JK builds
* CASCI/CASSCF paths that can contract directly with factors
* workflows where avoiding transformed dense MO ERIs matters

Example:

.. code-block:: python

   mol.build(eri="factors")
   mf = mol.RHF().run()

   # Factor-aware solvers can reuse mf.eri_factors instead of dense ERIs.
   mc = mol.CASSCF(mf, ncas=4, nelecas=4).run()

Native RI builds the three-center tensor in compact AO-pair form, then stores
SCF factors in full tensor form by default because that is currently the faster
RHF contraction path. Use ``ri_storage="packed"`` for memory-sensitive runs.
The metric solver uses a Cholesky factorization when the auxiliary Coulomb
metric is positive definite and falls back to an eigenvalue solver for
near-singular metrics. Useful RI options include ``ri_metric_solver="eigh"``
for a forced spectral solve, ``ri_screen_tol`` for three-center screening, and
``ri_block_size`` for the metric solve block size.

Optional Dependencies
---------------------

Some modules use optional compiled or third-party backends:

* ``libxc`` is used by parts of the native DFT stack.
* ``pyscf`` is useful for benchmarking and cross-validation.
* plotting and visualization examples may require packages such as PyVista.

Read the Docs does not need these optional dependencies for the static guide
pages. API pages that would import heavy optional backends are intentionally
kept static or excluded from the RTD build.

Recommended Defaults
--------------------

For native quantum chemistry examples:

.. code-block:: python

   mol.build(eri="auto")

For debugging a new tensor formula:

.. code-block:: python

   mol.build(eri="dense", aosym="s1")

For comparing factorized and dense algorithms:

.. code-block:: python

   mol.build(eri="dense+factors")

For a compact dense reference without keeping the four-index tensor:

.. code-block:: python

   mol.build(eri="dense", aosym="s8")

Related Pages
-------------

* :doc:`qchem`
* :doc:`mp2_comp2`
* :doc:`guide/guide_qchem_mcscf`
* :doc:`qchem_architecture`
