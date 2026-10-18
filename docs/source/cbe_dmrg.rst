Controlled bond expansion in reduced DMRG
=========================================

Enable CBE in a qchem DMRG driver with::

    dmrg.run(algorithm="one_site_cbe", cbe_expansion=8)

The default CBE selector is ``cbe_selection="shrewd"``. The ordinary SU(2)
solver still defaults to two-site DMRG. The reduced D3h/D6h orbital-multiplet
solver accepts the same CBE options and preserves both E partners.

``cbe_expansion`` bounds additional reduced bond multiplicities per update.
``cbe_preselection`` optionally sets the intermediate MPS-bond budget.
Its automatic value is the ceiling of the larger of current bond rank and
expansion size divided by the number of active open operator-factor channels,
with a minimum of one. This adapts the paper's D'=D/w estimate to the reduced
route representation. A larger explicit preselection budget can improve
selection at additional cost; truncations never select individual E rows.

The usual ``max_bond``/``D`` cap applies after optimization. ``D`` remains a
total reduced bond budget for SU(2) orbital sweeps with
``max_bond_mode="reduced"``, and a per-sector multiplicity cap for the spatial
multiplet driver. Qchem ``nsweeps`` counts complete forward/backward cycles;
lower-level ``run_sweeps`` counts directional passes.

``cbe_selection="exact"`` explicitly selects the full DD-residual reference
for validation and comparisons. Unsupported shrewd operator representations
raise an error; there is no silent exact-residual fallback. A nonzero noise
mixer cannot be combined with CBE.

Method and fidelity
-------------------

The implementation adapts A. Gleis, J.-W. Li and J. von Delft,
*Controlled Bond Expansion for Density Matrix Renormalization Group Ground
State Search at Single-Site Costs*, Physical Review Letters **130**, 246402
(2023), https://doi.org/10.1103/PhysRevLett.130.246402. Shrewd selection follows
the four SVD stages in supplement Fig. S-2 and Table I:

* Canonicalize the opposite discarded operator half, transferring its weight
  to the MPS bond while leaving the MPO bond open.
* Truncate the weighted MPS bond to the preselection budget.
* Redirect the open MPO bond to obtain the preselected complement; remove
  numerical null directions and reorthogonalize against kept directions.
* Close the MPO bond after projection into this smaller space, then retain
  the strongest DD directions across complete symmetry multiplets.

The kept basis is recovered from the site tensors with thin QRs and a small
bond SVD. Selection does not construct the full two-site Hamiltonian residual
or explicit discarded-space projectors. Existing contextual SU(2) operator
factors and product-irrep CG routes provide its one-sided contractions.
The old state is preserved to numerical rank tolerance. With untruncated
preselection, tests recover the exact DD selected subspace.

This is a reduced-route adaptation, not a claim of identical implementation
or timings to the publication. Intermediate budgets count reduced sector
multiplicities and open factor channels; non-Abelian fusion makes D'=D/w
approximate. Operator halves currently use bounded dense matrices within
individual reduced sector pairs. The orbital-site path contracts the fixed
neighbor into each operator half once, then uses the compiled factor-family
executor on one-site vectors throughout Davidson. Its preconditioner is the
exact diagonal of that one-site action, including off-diagonal contributions
from the fixed neighboring site.

Both orbital and spatial-multiplet CBE updates use a shared compiled sector
decomposition workspace. Thin site QR factorizations reduce truncation to a
small bond-core SVD; the engine selects whole multiplets and rotates the site
factors back. It never forms their pair product for truncation. The workspace
uses SciPy's compiled LAPACK API for QR/SVD across platforms, with BLAS site
rotations on macOS and a compiled matrix-product fallback elsewhere. Tensor metadata and selection remain
Python-managed. Pair layout/state packing and contextual operator setup still
exist once per bond, so **full-sweep single-site cost scaling is not claimed**.
The full spatial-multiplet path retains its genuine one-site contraction kernel.

Normal/complementary environment transport reuses compiled contextual spin
recoupling, while retaining the existing environment metric convention. The
right-core connection weight is removed before transport; contexts with a
singular connection retain the explicit recoupling path. Hamiltonian refresh
invalidates these numeric environment blocks. Local-operator assembly caches
each half's boundary/core routes before joining the shared MPO channel, so
the same half is not rebuilt for every opposite-site sector. These changes
do not alter the shrewd-selection budgets or expand individual spatial partners.

The eigensolver reuses the existing Davidson adaptation from
E. R. Davidson, J. Comput. Phys. **17**, 87--94 (1975),
https://doi.org/10.1016/0021-9991(75)90065-0. Neither path builds a determinant
Hamiltonian or performs a global variational projection.

Scope and validation
--------------------

Supported: open boundaries, one target ground-state multiplet, canonical
external blocks, and positive factorized diagonal reduced metrics. The
orbital-site shrewd path requires contextual normal/complementary operator
factors; the spatial path requires the fully reduced multiplet backend.
Excited-root constraints, state averaging, and the spatial component-reference
backend are unsupported. Estimated pair and one-sided-operator workspaces
have separate 256 MiB guards, not a total process-memory bound.

Final truncation can increase energy, and approximate selection can miss
useful directions. Check energy histories, local residuals, and bond-dimension
convergence; no global ground-state guarantee is implied.

``tests/test_cbe_dmrg.py`` checks the untruncated shrewd/exact DD limit,
one-sided factor actions against the original reduced operators, complex
thin canonicalization, compiled one-site actions against complex pair references,
factor updates against explicit pair SVDs in all three bond-budget modes,
independent Hubbard and molecular references,
Abelian targets, and interacting D3h/D6h E multiplets. Tests forbid full
residual selection and partner-component expansion in the default path.

Benchmarking
------------

Run the reproducible comparison and diagnostic plot with::

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python examples/mps/benchmark_cbe_hubbard.py \
      --sites 8 16 24 --algorithm one_site_cbe --selection exact shrewd \
      --D 32 --passes 12 --expansion 8 \
      --output /private/tmp/cbe-shrewd/final

Add ``two_site`` to ``--algorithm`` for a fresh two-site control. The script
uses the central engine thread budget, identical seeded initial states,
open half-filled Hubbard chains with U/t=4, and identical local tolerances.
JSON records fresh normalized energies of the returned best-pass states,
local residuals, histories, and runtime. Times exclude plotting and setup.
The figure compares convergence and sweep time; finite-D agreement is not
an independent exact-energy certificate.

The earlier projected-pair implementation gave this local single-run comparison
(D=32, expansion=8, 12 directional passes, four engine threads, one BLAS thread):

.. list-table:: Explicit DD versus shrewd selection
   :header-rows: 1

   * - Sites
     - Explicit DD (s)
     - Shrewd (s)
     - Absolute energy difference (t)
   * - 8
     - 6.30
     - 6.99
     - 9.07e-11
   * - 16
     - 57.16
     - 60.72
     - 1.80e-7
   * - 24
     - 197.08
     - 211.59
     - 8.12e-8

All local solves converged, and returned norms differed from unity by less
than 2e-15. These are fixed-pass finite-D comparisons, not timings to a
certified global energy tolerance. Shrewd selection avoids the explicit DD
residual but did not reduce total sweep time in those runs. These timings
predate the one-site contractions and compiled factor updates described above.

The one-site contraction and compiled factor-update comparison uses the same
settings, with a fresh two-site control::

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python examples/mps/benchmark_cbe_hubbard.py \
      --sites 8 16 24 --algorithm one_site_cbe two_site --selection shrewd \
      --D 32 --passes 12 --expansion 8 \
      --output /private/tmp/cbe-one-site/final-lapack

.. list-table:: One-site actions and compiled factor updates (single runs)
   :header-rows: 1

   * - Sites
     - Previous shrewd CBE (s)
     - Updated CBE (s)
     - Two-site (s)
     - CBE energy (t)
     - Two-site energy (t)
   * - 8
     - 6.99
     - 5.70
     - 1.61
     - -4.235806965853
     - -4.235806965643
   * - 16
     - 60.72
     - 50.94
     - 21.67
     - -8.818763052243
     - -8.818762530239
   * - 24
     - 211.59
     - 190.07
     - 95.72
     - -13.406047561570
     - -13.406047340663

CBE sweep times decreased by 10--18% relative to the earlier shrewd runs;
two-site DMRG remains faster. All local solves converged. The largest
CBE/two-site energy difference was 5.23e-7 t, and the largest difference
from the previous CBE energies was 2.52e-8 t. All returned norms were within
5e-15 of unity. The previous CBE column is the recorded earlier baseline;
updated CBE and two-site timings are fresh sequential runs with no concurrent
compilation or test jobs. No timing medians or global energy certificate are
implied. Selection, contextual operator setup, and tensor metadata still
involve Python; these changes specifically remove pair Hamiltonian matvecs
from Davidson and pair products/SVDs from truncation.

Compiled recoupling and cached half-route assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A subsequent profile of six 16-site CBE passes identified environment
transport and contextual operator assembly as the main costs. Reusing the
compiled recoupling kernels and resolving each half-route once gives the
following 12-pass, D=32, expansion=8 comparison with the same settings:

.. list-table:: Environment and route optimization (single runs)
   :header-rows: 1

   * - Sites
     - Recorded previous CBE (s)
     - Updated CBE (s)
     - Fresh two-site (s)
     - CBE energy (t)
     - Two-site energy (t)
   * - 16
     - 50.94
     - 41.87
     - 13.45
     - -8.818763052243
     - -8.818762530239
   * - 24
     - 190.07
     - 152.70
     - 55.45
     - -13.406047561570
     - -13.406047340663

The CBE times decrease by 18--20% relative to the recorded baseline. The
shared recoupling change also benefits two-site DMRG, which remains faster.
Updated runs were sequential, without concurrent tests or compilation. This
is a fixed-pass comparison, not time to a certified ground-state accuracy.
All local solves converged; CBE energies agree with the prior implementation
within 2e-14 t. The regression coverage includes independent Python
recoupling in both directions and virtual-bond conventions, Hamiltonian
refresh, and both D3h/D6h E partners (165 tests passed, 5 excluded).

Reproduce with the command above, changing ``--sites`` to ``16 24`` and
``--output`` to ``/private/tmp/cbe-speed/final``. The script writes
``results.json`` and ``cbe-hubbard-comparison.png`` there.
