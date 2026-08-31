Periodic GW and BSE
===================

The periodic Gaussian GW/BSE entry points live in ``pyqed.pbc.gw``.  They
are designed for small native periodic Hartree-Fock references, especially
compact validation cells where dense k/q-resolved matrices are still useful.
The older development namespace ``pyqed.gw.pbc`` remains a compatibility
alias, but new code should import from ``pyqed.pbc.gw``.

The public driver names mirror the molecular workflow:

* ``pyqed.pbc.gw.KGW`` computes periodic quasiparticle energies.
* ``pyqed.pbc.gw.KTDA`` solves q-resolved TDA-BSE excitations.
* ``pyqed.pbc.gw.KBSE`` solves q-resolved full BSE/Casida excitations.

Basic Example
-------------

.. code-block:: python

   import numpy as np

   from pyqed.qchem.pbc import Cell
   from pyqed.pbc.gw import KGW, KTDA, KBSE

   cell = Cell(
       atom="H 0 0 0; H 1.4 0 0",
       a=np.diag([5.0, 5.0, 5.0]),
       basis="sto-3g",
       unit="bohr",
       dimension=3,
       spin=0,
   ).build()

   mf = cell.KRHF(
       nk=(2, 1, 1),
       eta=0.5,
       real_cut=2,
       pair_cut=2,
       recip_cut=5,
   ).density_fit(
       auxbasis="def2-svp-jkfit",
       precision=1e-8,
       storage="auto",
       stream_pairs=True,
   )
   mf.with_df.build()
   mf.run()

   gw = KGW(mf, eta=1e-3).g0w0(
       backend="periodic",
       coulomb_component="gdf",
       direct_scale=1.0,
       prebuild_gdf=True,
   )
   tda = KTDA(gw).run(
       backend="periodic", q_index=0, direct_scale=1.0, nroots=2
   )
   bse = KBSE(gw).run(
       backend="periodic", q_index=0, direct_scale=1.0, nroots=2
   )

   print(gw.e_qp)  # shape: (nkpts, nband)
   print(tda.e)
   print(bse.e)

The runnable repository example is:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_gw_bse.py

Native GTH Pseudopotentials
---------------------------

Three-dimensional RHF/KRHF cells accept named GTH/HGH pseudopotentials and
use valence ionic charges consistently in the electron count and ion-ion
Ewald energy.  The local Fourier kernel and nonlocal Gaussian projectors are
evaluated by PyQED; PySCF is used only to load a named table when installed.
An explicit PySCF-format pseudopotential dictionary removes that dependency.

.. code-block:: python

   cell = Cell(
       atom="C 0 0 0; C 1.7 1.7 1.7",
       a=np.eye(3) * 6.8,
       basis="gth-szv",
       pseudo="gth-pade",
       dimension=3,
       integral_options={"eri_representation": "direct"},
   ).build()

   mf = cell.KRHF(
       nk=(2, 2, 2),
       eta=0.5,
       real_cut=2,
       pair_cut=2,
       pseudo_cut=1,
       recip_cut=7,
   ).run()

Pseudopotential cells select native GDF J/K and a valence-configuration SAD
guess by default.  The local potential uses an exact range-separated form:
point-charge Ewald plus short-range erfc and analytic Gaussian-polynomial
corrections.  ``real_cut`` controls the one-body AO image sum and
``pseudo_cut`` controls localized pseudopotential corrections and nonlocal
projectors.  Native GTH calculations require ``pair_cut >= 2``; smaller AO-pair
domains can shift total energies by tenths of a millihartree even after the
reciprocal sum is converged.  The current native path supports
scalar-relativistic GTH/HGH data in 3D.  Molecular ECP formats, spin-orbit
projectors, low-dimensional pseudopotentials, and the nonlocal pseudopotential
commutator in optical velocity matrix elements remain outside this path.

For an eigenvalue-only GW smoke path with quasiparticle gaps also used in the
BSE screening poles:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_gw_bse.py \
       --gw-method evgw --gw-max-cycle 1 --bse-screening-energy qp --nroots 1

For the dense small-cell full-Ewald diagnostic path:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_h2_gw_bse.py \
       --coulomb-component full_ewald --nroots 1

PySCF Benchmark Caveat
----------------------

The representation-matched benchmark in ``examples/pbc_gdf_validation.py``
compares native PyQED GDF factors, J/K matrices, and GW quasiparticle energies
against PySCF GDF on the same cell, basis, auxiliary basis, and k mesh:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gdf_validation.py \
       --case h2-3k --precision 1e-8 \
       --output /private/tmp/pbc_h2_gdf_validation.json

Add ``--bse`` to compare the dense BSE matrices, ``--native-krhf`` to run the
native self-consistent reference, or ``--finite-size-ladder`` for the
finite-size study:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gdf_validation.py \
       --case h2-3k --bse --native-krhf

General three-dimensional meshes are repeatable command-line inputs.  PySCF's
reference cell precision is controlled independently, so a native precision
ladder can be compared with one fixed, tighter reference:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gdf_validation.py \
       --case lih-rocksalt-2k-svp-solid \
       --kmesh 2,2,2 --kmesh 4,2,2 \
       --precision 1e-8 --reference-precision 1e-12

The ``pyscf_gdf`` component mirrors the native PyQED cell into PySCF, builds
GDF factors on the same k mesh, transforms them with the PyQED Bloch orbitals,
and uses ``direct_scale=1.0`` by default.  Its finite-size correction follows
PySCF's spin-summed response convention for the GDF body and applies the
one-sided diagonal correction with the corresponding half-residue sign.
For a dependency-free PyQED route, use ``coulomb_component="gdf"``.
This builds native auxiliary-basis GDF tensors with PyQED's Gaussian integral
primitives, transforms them to the Bloch MO pair basis, and uses the same
``direct_scale=1.0`` and finite-size response convention.  The auxiliary basis
defaults to the bundled RI/J-fit partner selected by the native molecular RI
helper; set ``mf.gdf_auxbasis`` or pass ``auxbasis=...`` to
``gdf_transition_factors`` for explicit control.

For native range-separated builds, ``mf.gdf_precision`` can drive the full
automatic setup:

.. code-block:: python

   mf.gdf_precision = 1e-6
   mf.gdf_mesh = "auto"
   mf.gdf_omega = "auto"
   mf.gdf_pair_cut = "auto"
   mf.gdf_reciprocal_kernel = "range_separated"
   mf.gdf_g_block_max_mb = 256

When no GDF convergence controls are set, native GDF uses this
range-separated policy automatically with ``gdf_precision=1e-8``.  The
reciprocal mesh is estimated with a finite range-separation seed, avoiding the
prohibitively large full-Coulomb mesh that would otherwise be selected for
tight core Gaussians.  The automatic mesh adds three reciprocal shells
(``gdf_mesh_safety_pad=6``) around the PySCF-style estimate to cover the
discretization margin of the independent native Fourier kernels; explicit
``gdf_mesh`` values are used unchanged.  The same precision derives the
short-range radial image domain.
The manual function-level ``gdf_short_range_screen_tol`` and relative metric
truncation both default to zero because heuristic thresholds can remove
physically important periodic terms.  The absolute pseudoinverse floor is
precision-aware by default,

.. math::

   \epsilon_{J_{2c}}^{\rm auto}
   = \max\!\left(10^{-14},\,0.1\,\epsilon_{\rm GDF}\right).

Set ``gdf_metric_tol`` explicitly to override this automatic value, and set
``gdf_metric_relative_tol`` explicitly for a deliberate rank-convergence
study.  Diffuse auxiliary primitives can likewise be removed explicitly with
``aux_min_exponent``.  This is equivalent to PySCF GDF's
``exp_to_discard`` control and retains primitive :math:`p` exactly when

.. math::

   \alpha_p \geq \alpha_{\min}^{\rm aux}.

Both controls are available from the persistent SCF backend without setting
attributes by hand:

.. code-block:: python

   mf = cell.KRHF(kpts).density_fit(
       auxbasis="def2-svp-jkfit",
       precision=1e-12,
       aux_min_exponent=0.075,
   )

Auxiliary pruning is deliberately not automatic.  It changes the fitting
space, and a cutoff suitable for one cell and auxiliary basis can be damaging
for another.  A production calculation should compare the unpruned and pruned
SCF energies and target observables, then tighten ``gdf_precision`` at the
chosen rank.  A reference-code comparison must apply the identical exponent
floor on both sides.  For rocksalt LiH with the solid-pruned def2-SVP orbital
basis and def2-SVP-JKFIT auxiliary basis, a floor just above
:math:`0.0747033\,a_0^{-2}` removes only the two most diffuse Li fitting
primitives.  It is a system-specific convergence point, not a universal
default.  The :math:`4\times2\times2` validation retains all 65 remaining
metric modes with ``gdf_metric_tol=1e-14``.  Relative or stronger absolute
metric truncation is not needed for this case.  Compared with the unpruned
fit, the matched PySCF SCF energy changes by
:math:`8.27\times10^{-3}\,\mathrm{meV}`; report this fitting-space convergence
separately from the PyQED--PySCF implementation error.

The validation driver can force PySCF's normal GDF builder to use an
eigendecomposed metric for explicit rank studies with
``--pyscf-metric-eig --metric-tol ...``.  This affects only the optional
benchmark reference and does not introduce a PySCF dependency into PyQED's
periodic GDF, KRHF, GW, or BSE implementations.  Internally generated BSE
screening spaces inherit the attached reference GDF context so direct,
exchange, and screened-exchange terms always use the same auxiliary basis,
primitive floor, metric rank, and stored factors.

For metric eigenpairs :math:`J\mathbf u_a=\lambda_a\mathbf u_a`, the retained
space satisfies

.. math::

   \lambda_a > \max\!\left(
       \epsilon_{\rm abs},
       \epsilon_{\rm rel}\lambda_{\max}
   \right),
   \qquad
   J^{-1/2}_{Pa}
   = \frac{(\mathbf u_a)_P}{\sqrt{\lambda_a}}.

Build diagnostics report the retained metric condition number and the
whitening amplification,

.. math::

   \kappa(J_{\rm ret})
   = \frac{\lambda_{\max}}{\lambda_{\min}^{\rm ret}},
   \qquad
   \kappa(J_{\rm ret}^{-1/2})
   = \sqrt{\kappa(J_{\rm ret})}.

This distinction matters when comparing implementations.  A positive but
nearly singular metric may let a Cholesky factorization retain modes at the
floating-point floor, whereas an eigensolver may deliberately remove them.
The resulting whitened factors need not agree even when the unwhitened
:math:`J_{2c}` and :math:`J_{3c}` tensors do.  Such a comparison must use the
same explicit retained subspace in both codes; otherwise it tests a numerical
factorization convention rather than the periodic GDF integrals.

Before the metric eigendecomposition, every auxiliary shell is rescaled to a
common radial multipole convention.  For angular momentum :math:`l`, primitive
exponents :math:`\alpha_p`, contraction coefficients :math:`c_p`, and radial
normalization :math:`N_l(\alpha_p)`, native GDF uses

.. math::

   M_l = \sum_p c_p N_l(\alpha_p)
         \int_0^\infty r^{2l+2}e^{-\alpha_p r^2}\,dr,
   \qquad
   c'_p = \frac{c_p}{\sqrt{4\pi}\,M_l}.

The GDF projection is invariant to this nonsingular shell rescaling before
truncation.  It makes the absolute metric cutoff well conditioned and gives
the same retained auxiliary rank as the standard periodic GDF convention.

An explicit ``gdf_reciprocal_kernel="full"`` selects the full reciprocal
algorithm; without explicit cutoffs it uses the conservative
``gdf_recip_cut=15`` and ``gdf_pair_cut=3`` fallback.  The range-separated
reciprocal metric and three-center tensors are accumulated in one streamed
G-vector pass, and repeated q blocks reuse the resulting AO factor stores.

For bounded-memory SCF and subsequent GW/BSE calculations, attach the
persistent periodic DF backend directly to KRHF:

.. code-block:: python

   cell = Cell(
       atom=atoms,
       a=lattice,
       basis=basis,
       integral_options={"eri_representation": "direct"},
   ).build()
   mf = cell.KRHF(kpts=cell.make_kpts((4, 4, 4))).density_fit(
       auxbasis=auxbasis,
       precision=1e-8,
       storage="auto",
       max_memory_mb=2048,
       cache_dir="/scratch/job/gdf",
       stream_pairs=True,
       stream_pair_batch_mb=128,
   )
   mf.with_df.build(workers=8)
   mf.run()

``eri_representation="direct"`` avoids constructing an unused molecular
four-center ERI tensor.  ``storage="auto"`` retains whitened cderi blocks up
to ``max_memory_mb`` and spills subsequent blocks to NumPy memmaps;
``storage="disk"`` forces all blocks out of core.  ``stream_pairs=True``
groups raw AO three-center k-pair blocks into bounded batches so each batch
shares one expensive short-range shell traversal.  ``stream_pair_batch_mb``
sets the approximate workspace budget, including live output buffers for the
configured inner short-range workers; ``stream_pair_batch_size`` can set an
explicit pair count for benchmarking or tightly controlled runs.  When at
least two complete q blocks fit that budget, their three-center AO Bloch sums
and two-center auxiliary metrics are built in shared multi-q traversals.  The
expensive image integrals are evaluated once, while vectorized Bloch phases
produce the q-resolved outputs.  Both caches are consumed immediately by the
q-resolved stores and are included in the workspace bound.  The scheduler uses
one outer q worker while compiled pair-FT or short-range kernels are internally
parallel, avoiding nested worker oversubscription.  Same-k ``q=0`` blocks use
Hermitian packed storage,
opposite q blocks are generated by conjugation, and GW/BSE reuse the persisted
SCF factors.  Call
``mf.with_df.close()`` to remove temporary cache files explicitly.

For a self-opposite mesh transfer,
:math:`\mathbf q=-\mathbf q+\mathbf G`, the ordered AO three-center blocks obey

.. math::

   B^P_{\mu\nu}(\mathbf k+\mathbf q,\mathbf k;\mathbf q)
   = \left[
       B^P_{\nu\mu}(\mathbf k,\mathbf k+\mathbf q;\mathbf q)
     \right]^*.

Native GDF therefore evaluates one canonical source for each unordered
:math:`\{\mathbf k,\mathbf k+\mathbf q\}` pair and reconstructs the reverse AO
block before applying the same q-resolved metric whitening,

.. math::

   L^a_{\mu\nu}
   = \sum_P \left[J^{-1/2}(\mathbf q)\right]^*_{Pa}
     B^P_{\mu\nu}.

This reduction also applies inside explicit one-pair streaming batches, where
both whitened targets are emitted while the canonical AO source is resident.
Set ``gdf_self_opposite_pair_reuse=False`` only to run the fully ordered
reference construction.  The setting is part of both persistent and
transition-factor cache identities.

The KRHF one-electron build evaluates independent real-space image blocks in
parallel through the compiled integral kernel.  AO-pair images are screened by
``one_body_screen_tol``.  For each retained primitive pair, the short-range
nuclear lattice is centered on its Gaussian product center and screened there,
so increasing ``real_cut`` does not create a quadratic nuclear-image loop.
Its reciprocal nuclear attraction shares one AO-pair Fourier traversal across
the complete k mesh, with the Bloch phases accumulated as a batch.

``pair_cut`` must be at least ``real_cut`` because the same translated AO
pairs enter the reciprocal electron-ion matrix.  Truncating that reciprocal
domain can leave the real-space overlap apparently converged while shifting
the core Hamiltonian.  Set both controls to ``"auto"`` to derive a sparse,
inversion-symmetric lattice domain from the contracted Gaussian decay bound;
the strict rocksalt LiH benchmark resolves both envelopes to eight cells.
Explicit cutoffs should be increased together until the target observable is
stable, and the smaller values in smoke examples are not production
convergence recommendations.  Repeated integral builds reuse
the resulting k-resolved overlap and core-Hamiltonian cache when the cell,
basis, cutoffs, mesh, and pseudopotential configuration are unchanged.

For primitive exponents :math:`a,b`, weights :math:`w_a,w_b`, total angular
momentum :math:`l_{\mu\nu}`, and translated center distance
:math:`d_{\mu\nu}(\mathbf L)`, the retained-image bound is

.. math::

   B_{\mu\nu}(\mathbf L)
   = \sum_{ab} |w_a w_b|
     \left(\frac{\pi}{a+b}\right)^{3/2}
     \exp\!\left[-\frac{ab}{a+b}d_{\mu\nu}(\mathbf L)^2\right]
     \left[d_{\mu\nu}(\mathbf L)+(a+b)^{-1/2}+1\right]^{l_{\mu\nu}}.

The automatic envelope is the first monotone radial root of
:math:`B_{\mu\nu}=\epsilon_{\mathrm{1e}}` over all contracted AO pairs, and an
individual image-pair block is stored only when
:math:`B_{\mu\nu}(\mathbf L)>\epsilon_{\mathrm{1e}}`, where
``one_body_screen_tol`` supplies :math:`\epsilon_{\mathrm{1e}}`.

When ``coulomb_component="gdf"`` is used after this SCF calculation, GW
prebuilds transition factors directly from ``mf.with_df`` instead of creating
a second q-resolved AO store.  The diagonal self-energy cache retains one
transition-factor object per q point, so all orbital-pair and screened-mode
couplings reuse the same metric resolution throughout the frequency and target
loops.

The default ``gdf_rs_aux_partition="smooth"`` uses a compensated
range-separated construction analogous to periodic RS-GDF builders: all
auxiliary functions receive the full reciprocal contribution, while only
compact auxiliary shells receive the analytic short-range correction minus
its reciprocal short-range representation.  A compact auxiliary view is sent
to the integral kernel and transformed back into the original auxiliary
space, reducing both kernel time and temporary storage.  Set the partition to
``"off"`` for the unpartitioned reference algorithm or ``"all"`` for a
reciprocal-only diagnostic.

``gdf_g_block_max_mb`` limits the combined reciprocal workspace for the
auxiliary Fourier values, weighted auxiliary values, and AO-pair Fourier
values.  Range-separated reciprocal terms with identical G-vector rows and AO
pair masks reuse one raw pair-FT subblock; their weighted auxiliary
contractions remain separate, preserving the accumulation order.  The raw
cache is released after each reciprocal block and never exceeds one configured
pair-FT subblock for the standard full-plus-compensation pair.  Short-range
image-pair tensors are streamed directly into all requested Bloch blocks with
bounded in-flight worker tasks and are never stored in the image-component
cache.  The final q-resolved AO/MO factors are still retained, so their
system-size-dependent storage is separate from this workspace limit.  An
explicit ``gdf_short_range_cut`` overrides the automatic image box.  Setting a
nonzero manual ``gdf_short_range_screen_tol`` remains a diagnostic opt-in and
also requires
``gdf_allow_heuristic_short_range_screening=True``.

The AO-pair Fourier plan applies two independent precision-derived screens.
For requested GDF precision :math:`\epsilon_{\mathrm{GDF}}`, the default shell
image envelope and primitive-coefficient threshold are

.. math::

   \epsilon_{\mathrm{image}}
   = 10^{-2}\epsilon_{\mathrm{GDF}}, \qquad
   \epsilon_{\mathrm{coeff}}
   = \max\!\left(\epsilon_{\mathrm{pair}},
                  10^{-3}\epsilon_{\mathrm{GDF}}\right).

The image test uses a contracted shell-overlap bound, while the coefficient
test removes individual primitive-product terms after the retained image
domain is known.  These separate safety margins avoid evaluating lattice tails
many decimal orders below the requested precision.  Override them with
``gdf_pair_image_tol_factor`` and ``gdf_pair_ft_coeff_tol_factor``; set either
resolved tolerance to zero for its unscreened reference construction.  Build
timings report the resolved tolerances, retained image pairs, primitive terms,
and Gaussian-product factors.

The compiled reciprocal kernel applies a second, G-dependent bound to the
retained Gaussian-product factors.  For factor :math:`f`, let
:math:`C_f=\sum_{t\in f}|c_t|` include every grouped primitive coefficient and
let :math:`m_{f\alpha}` be the largest Cartesian polynomial order on axis
:math:`\alpha`.  At reciprocal vector :math:`\mathbf G`, the kernel evaluates

.. math::

   B_f(\mathbf G)
   = C_f\exp\!\left(-\frac{|\mathbf G|^2}{4p_f}\right)
     \prod_{\alpha\in\{x,y,z\}}
     \max(1,|G_\alpha|)^{m_{f\alpha}}.

With :math:`N_f` retained factors, factor :math:`f` is omitted only when

.. math::

   B_f(\mathbf G)
   \leq \frac{\epsilon_{\mathrm{factor}}}{N_f}.

Therefore the sum of all omitted factor bounds is no larger than
:math:`\epsilon_{\mathrm{factor}}`.  By default,
:math:`\epsilon_{\mathrm{factor}}=\epsilon_{\mathrm{coeff}}`.  Set
``gdf_pair_ft_factor_screen_tol`` explicitly to change this global budget or
to zero for the unscreened reference kernel.  The resolved value is included
in the GDF build timings.

Automatic short-range builds use a precision-derived radial image domain with
a conservative ``gdf_short_range_radius_factor=1.25`` safety margin.  Inside
that domain, the compiled shell engine screens primitive image tasks with the
same Gaussian exponential envelope that controls the image radius.  For AO
primitive exponents :math:`a,b`, auxiliary exponent :math:`c`, and
range-separation parameter :math:`\omega`, define

.. math::

   \mu_{ab} = \frac{ab}{a+b}, \qquad
   \mathbf P = \frac{a\mathbf A+b\mathbf B}{a+b}, \qquad
   \theta_{abc}^{\mathrm{SR}} =
   \left(\frac{1}{a+b}+\frac{1}{c}+\frac{1}{\omega^2}\right)^{-1}.

The primitive is retained when

.. math::

   \mu_{ab}\lvert\mathbf A-\mathbf B\rvert^2
   + \theta_{abc}^{\mathrm{SR}}\lvert\mathbf P-\mathbf C\rvert^2
   \leq \Lambda_{\mathrm{SR}}, \qquad
   \Lambda_{\mathrm{SR}} = -\ln\epsilon_{\mathrm{GDF}} + 4\ln 10.

The default adds four decimal orders of margin beyond the requested GDF
precision; tighten that margin when validating unusually diffuse or large
lattice sums with ``gdf_short_range_primitive_safety_digits``.  Set
``gdf_short_range_primitive_exp_cutoff=0`` for an unscreened reference or set
an explicit non-negative exponent cutoff for a convergence study.  Build
timings report the resolved cutoff and candidate/skipped primitive counts.

Set ``gdf_short_range_image_domain="box"`` to retain the full enclosing image
box; explicit numerical ``gdf_short_range_cut`` values use the box convention
by default.  ``gdf_short_range_image_domain="radial"`` can also be requested
with an explicit cutoff, in which case the cutoff is only the enclosing
enumeration box.

Image pairs are grouped by their relative AO translation.  The compiled
shell-major kernel reuses AO-pair primitive geometry across each group,
combines the full and long-range vertical-recurrence tables before applying a
single horizontal recurrence, and accumulates forward and mirror Bloch phases
directly into the output.  Its internal phase tables use ``(task, Bloch)``
layout and the Bloch axis is the contiguous output dimension; one axis move at
the Python boundary restores the public ``(Bloch, auxiliary, AO, AO)`` layout.
This avoids a cache-line-strided inner Bloch loop without changing the
accumulation order.  The kernel does not materialize or rescan an image-sized
AO tensor.  ``gdf_short_range_workers`` controls the worker count; the default
uses up to 24 local CPUs.  Build timings report
``three_center_sr_grouped_bloch``, the number of relative-translation groups,
the primitive screening counts, and the reduced compiled-call count.

Before entering the primitive recurrence, the kernel also applies a
PySCF/libpbc-style shell-distance pre-screen.  For an AO shell pair, let
:math:`\mathcal B_{AB}` be the axis-aligned box containing all primitive
product centers.  For auxiliary image center :math:`\mathbf C_t`, the lower
bound

.. math::

   \underline E_{AB,C_t}
   = \min_{ab}\left(\mu_{ab}\lvert\mathbf A-\mathbf B\rvert^2\right)
   + \min_{abc}\left(\theta_{abc}^{\mathrm{SR}}\right)
     d\!\left(\mathbf C_t,\mathcal B_{AB}\right)^2

cannot exceed any primitive exponent in that shell task.  The whole task is
therefore skipped only when
:math:`\underline E_{AB,C_t}>\Lambda_{\mathrm{SR}}`; survivors still pass
through the exact primitive test above.  This changes no cutoff semantics.
Build timings expose ``three_center_sr_shell_task_skips`` and per-worker load
and elapsed-time diagnostics.

Implemented Periodic Pieces
---------------------------

The current multi-k implementation is intentionally direct and dense:

* k-point RHF/KRHF references are adapted through
  ``pyqed.pbc.gw.KPointSCFAdapter``.
* q blocks use the transition basis ``(v, k) -> (c, k + q)``.
* Reciprocal-space transition and orbital-pair factors are built from the
  native Ewald pair Fourier transform.  Their ``coulomb_component`` label is
  currently ``"reciprocal_ewald_lr"``.  They record the ``g2_tol`` used to
  define the reciprocal basis, reject negative tolerances, and only contract
  factors built on compatible q blocks and G bases.
* Screening is direct-RPA/TDH in a dense transition-space Casida problem.
  ``QBlockResponse`` and ``ScreenedInteractionPoles`` record the q block,
  canonical Coulomb component, kernel scale, and numerical tolerances used to
  build the reusable response layer.
* Dense small-cell response diagnostics can also use
  ``coulomb_component="full_ewald"`` in ``direct_tdh_matrices``,
  ``direct_rpa``, and ``KPointTransitionSpace.screened_interaction`` to build
  the direct kernel from native full Ewald pair blocks.
* Dense small-cell GW/BSE diagnostics can use the same ``coulomb_component`` option
  in ``diagonal_correlation_self_energy``, ``diagonal_g0w0``,
  ``periodic_bse_matrices``, ``periodic_tda``, and ``periodic_bse`` to build
  dense full-Ewald orbital-pair couplings.
* Optional PySCF-backed GW/BSE diagnostics can use
  ``coulomb_component="pyscf_gdf"`` to build the transition metric and
  orbital-pair couplings from PySCF Gaussian density-fitting factors.  This is
  intended for PySCF benchmark comparisons and requires PySCF at runtime.
* Dependency-free native factorized GW/BSE runs can use
  ``coulomb_component="gdf"``.  This builds an auxiliary-basis GDF vector basis
  from a native auxiliary Coulomb metric and periodic three-center AO tensors,
  then exposes the same transition/pair coupling interface as the PySCF GDF
  backend.
* ``gdf_mo_jk(space, coulomb_component="gdf")`` contracts those same
  q-resolved factors into closed-shell k-point MO Coulomb and exchange
  matrices.  It also accepts AO density matrices through ``dm=...`` for SCF
  iterations.  Passing ``"pyscf_gdf"`` provides a like-for-like validation
  diagnostic without changing the native GW/BSE implementation.
* Native ``Cell.KRHF(..., jk_builder="gdf")`` uses the density-driven GDF J/K
  contraction and applies the periodic Madelung exchange correction in the SCF
  layer.  Self-consistent mesh orbital energies are available with
  ``band_structure(exchange="mesh")``; off-mesh plots use
  ``exchange="mesh_interpolate"`` until direct finite-q GDF Fock builds are
  implemented.
* Periodic diagonal GW supports a PySCF-style small-sphere q->0 finite-size
  head/wing correction for ``coulomb_component="reciprocal_ewald_lr"`` and
  the vector-basis components ``"gdf"`` and ``"pyscf_gdf"`` via
  ``finite_size_correction=True``.
  Result metadata records the separate ``finite_size_head``,
  ``finite_size_wing``, and ``finite_size_sigma`` arrays.
* Coulomb-component aliases are canonicalized through
  ``normalize_coulomb_component``: ``"reciprocal"``, ``"long_range"``, and
  ``"lr"`` map to ``"reciprocal_ewald_lr"``, ``"full"`` maps to
  ``"full_ewald"``, ``"gdf"``/``"density_fit"`` map to the dependency-free
  factor backend, and ``"pyscf_gdf"``/``"pyscf_df"`` map to PySCF GDF.
  Result metadata records the canonical name, and periodic BSE metadata also
  records the kernel scales and numerical tolerances used to build each q
  block.
* Active transition windows are available through ``occ_bands`` and
  ``vir_bands`` in ``KPointTransitionSpace`` and the high-level ``KGW``,
  ``KTDA``, and ``KBSE`` wrappers.  Lists apply to every k-point; dictionaries
  select bands per k-point and unspecified k-points remain unrestricted.
  Band selectors must contain integer indices; fractional values are rejected
  instead of truncated.
* Diagonal GW corrections can be restricted with ``qp_bands`` in
  ``diagonal_g0w0``, ``diagonal_evgw``, and ``KGW``.  Lists target those bands
  at every k-point; dictionaries target explicit k-point/band pairs.  Bands
  outside the target set keep their input energies in ``e_qp`` and have
  ``nan`` entries in ``sigma_c``.
  GW result metadata records the normalized q-block selection, Coulomb
  component, kernel scale, broadening, and numerical tolerances used for the
  correction.
* ``DiagonalSelfEnergyCache`` stores q-resolved screening poles, reciprocal
  factors, and mode couplings for repeated diagonal self-energy evaluations.
  ``diagonal_g0w0`` and ``diagonal_evgw`` create one automatically, and an
  explicit cache can be passed when reusing intermediates across calls.
* The self-energy band sum can be truncated with ``intermediate_bands`` in
  ``diagonal_correlation_self_energy``, ``diagonal_g0w0``, ``diagonal_evgw``,
  and ``KGW``.  Lists apply at every intermediate k-point; dictionaries
  override individual k-points while unspecified k-points remain unrestricted.
  ``qp_bands`` and ``intermediate_bands`` follow the same integer-index
  validation as the transition-window selectors.
* Periodic TDA/BSE solvers validate ``nroots``: non-integer and negative
  requests are rejected, requesting more roots than a q block contains raises
  an error, and result metadata records ``nroots_requested`` and
  ``nroots_returned``.
* q-block requests use explicit ``q_index``/``q_indices`` validation across
  response, GW, and BSE helpers, so negative Python-style indices are rejected
  instead of silently selecting the last q block.
* Adapter-level k-point band queries validate ``k_index`` and require
  ``occupation_tol`` in ``[0, 1)`` so occupied/virtual band classification
  remains unambiguous.
* Orbital-pair integral helpers and diagonal self-energy calls validate
  ``k_index``/``kq_index`` and band indices explicitly; fractional values are
  rejected instead of silently truncated.
* GW iteration controls such as ``max_cycle`` and root-solver ``maxiter`` are
  validated as positive integer counts.
* Passing ``coulomb_component="full_ewald"`` or ``backend="periodic"`` through
  ``pyqed.pbc.gw.KGW``, ``KTDA``, or ``KBSE`` routes Gamma-point references
  through the periodic implementation instead of the molecular bridge.  A
  ``q_index`` request also selects the periodic ``KTDA``/``KBSE`` route.
  Conversely, ``backend="molecular"`` rejects periodic-only options rather
  than forwarding them into the molecular bridge, and true multi-k references
  require the periodic backend.
* ``pyqed.pbc.gw.KGW.g0w0`` computes diagonal one-shot G0W0 corrections.
* ``pyqed.pbc.gw.KGW.evgw`` runs diagonal eigenvalue-only GW with updated
  transition energies when ``update_screening=True``.
* ``pyqed.pbc.gw.KGW.gnw0`` runs the same diagonal eigenvalue loop while
  keeping the initial screened interaction fixed.
* ``pyqed.pbc.gw.KGW.spectral_function`` evaluates exact-pole diagonal GW
  spectral functions for selected k points and bands.
* ``pyqed.pbc.gw.KTDA`` and ``pyqed.pbc.gw.KBSE`` consume the quasiparticle
  energies from ``pyqed.pbc.gw.KGW`` by default.

Optical BSE Absorption
----------------------

The periodic TDA and full-BSE results can be converted into a bulk optical
spectrum from the vertical :math:`q=0` transition block.  The BSE kernel uses
the symmetrized Brillouin-zone quadrature

.. math::

   \widetilde K_{t t'}
   = \sqrt{w_t}\,K_{t t'}\,\sqrt{w_{t'}},
   \qquad
   w_t = \frac{1}{N_k},

where :math:`t=(v,c,k)`.  This normalization makes the exciton interaction
converge with the k-point mesh instead of scaling with :math:`N_k`.

For the native all-electron Gaussian backend, the independent-particle
velocity and length-gauge transition dipole are

.. math::

   \mathbf v_t
   = \langle v k | -i\boldsymbol{\nabla} | c k\rangle,
   \qquad
   \mathbf d_t
   = \frac{i\mathbf v_t}{E_{c k}-E_{v k}}.

The exciton transition dipole is

.. math::

   \mathbf D_S
   = \sqrt{2}\sum_t \sqrt{w_t}
     \left(X_t^S+Y_t^S\right)\mathbf d_t,

with :math:`Y_t^S=0` in TDA.  The factor :math:`\sqrt{2}` is the closed-shell
spin-singlet factor.  For polarization :math:`\mathbf e`, PyQED reports

.. math::

   f_S^{(\mathbf e)}
   = 2\Omega_S\left|\mathbf e^\dagger\mathbf D_S\right|^2

and

.. math::

   \operatorname{Im}\epsilon_{\mathbf e}(\omega)
   = \frac{4\pi^2}{\Omega_{\mathrm{cell}}}
     \sum_S
     \left|\mathbf e^\dagger\mathbf D_S\right|^2
     L_\eta(\omega-\Omega_S).

``polarization=None`` returns the Cartesian isotropic average.  Real vectors
select linear polarization and complex vectors can select circular
polarization.  For example:

.. code-block:: python

   import numpy as np

   from pyqed.pbc.gw import KGW, KTDA

   gw = KGW(mf, eta=1e-3).g0w0(
       backend="periodic",
       coulomb_component="gdf",
       direct_scale=1.0,
   )
   tda = KTDA(gw).run(
       backend="periodic",
       qpts="optical",
       q_index=0,
       nroots=8,
       return_vectors=True,
       coulomb_component="gdf",
       direct_scale=1.0,
   )
   optical = tda.absorption(
       energy_grid=np.linspace(0.0, 8.0, 1601),
       polarization="x",
       broadening=0.10,
       units="ev",
   )

``optical.dielectric_imag`` contains the polarization-resolved spectrum,
``optical.dielectric_tensor_imag`` contains the Cartesian tensor, and
``optical.oscillator_strengths`` contains one value per exciton root.

Matrix-Free TDA and Haydock Recursion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For fine k-point meshes, ``KTDA.haydock`` evaluates the optical response
without constructing or diagonalizing the global transition-space matrix.
For a transition-factor matrix :math:`Z_q`, transition quadrature
:math:`W_q`, and positive independent-particle gaps :math:`D_q`, define

.. math::

   F_q = W_q^{1/2} Z_q,
   \qquad
   P_q = F_q^\dagger D_q^{-1} F_q.

With the direct-kernel scale :math:`s`, the implemented static RPA-induced
interaction is evaluated exactly in the auxiliary space:

.. math::

   C_q
   = s^2 P_q\left(I+2sP_q\right)^{-1}.

If :math:`P_q=V_q\Lambda_qV_q^\dagger`, its low-rank factor is

.. math::

   U_q
   = V_q\operatorname{diag}\left(
       \sqrt{\frac{s^2\lambda_{q\mu}}
                    {1+2s\lambda_{q\mu}}}
     \right),
   \qquad
   C_q=U_qU_q^\dagger.

The default ``storage="transition_blocks"`` contracts these factors once and
stores only the Hermitian upper triangle as occupied-virtual k-point blocks.
For :math:`N_o` occupied and :math:`N_v` virtual bands at each of
:math:`N_k` k points, the stored interaction contains

.. math::

   \frac{N_k(N_k+1)}{2}
   \left(N_oN_v\right)^2

complex numbers.  ``storage="factorized"`` instead retains the auxiliary
pair factors and contracts them at each matrix-vector product.  It uses less
work to build but is normally slower over a Haydock recursion.

Starting from the weighted optical vector :math:`|d\rangle`, Lanczos recursion
produces diagonal coefficients :math:`\alpha_j` and off-diagonal coefficients
:math:`\beta_j`.  The broadened spectral density is the continued fraction

.. math::

   \rho_d(\omega)
   = -\frac{1}{\pi}\operatorname{Im}
     \frac{\langle d|d\rangle}
     {z-\alpha_0-
      \dfrac{\beta_0^2}{z-\alpha_1-
      \dfrac{\beta_1^2}{\ddots}}},
   \qquad
   z=\omega+i\eta.

The corresponding dielectric loss is

.. math::

   \operatorname{Im}\epsilon(\omega)
   = \frac{4\pi^2}{\Omega_{\mathrm{cell}}}\rho_d(\omega).

Unless the Krylov space closes by residual breakdown or reaches the full
transition dimension, spectral convergence must be checked by increasing
``niter`` at the chosen broadening.  The result metadata reports this exact
closure as ``krylov_complete``; it does not infer convergence from reaching a
user-requested truncated iteration count.

For example:

.. code-block:: python

   spectrum = KTDA(gw).haydock(
       qpts="mesh",
       energy_grid=np.linspace(0.0, 8.0, 1601),
       broadening=0.10,
       niter=120,
       coulomb_component="gdf",
       storage="transition_blocks",
   )

The same operator can return selected low-energy excitons without dense
diagonalization:

.. code-block:: python

   from pyqed.pbc.gw import periodic_tda_operator

   operator = periodic_tda_operator(
       space,
       coulomb_component="gdf",
       storage="transition_blocks",
   )
   roots = operator.eigensolve(nroots=8, tol=1e-9)

The high-level driver exposes the same sparse solver:

.. code-block:: python

   tda = KTDA(gw).eigensolve(
       nroots=8,
       tol=1e-9,
       coulomb_component="gdf",
       storage="transition_blocks",
   )
   excitation_energies = tda.e

For a silicon k-mesh study, record operator-build time, recursion time,
storage, roots, and spectra for each mesh explicitly.  Keep the mesh family
consistent: mixing odd Gamma-containing and even shifted Monkhorst--Pack grids
can produce a large parity oscillation in the apparent optical edge.

When a PySCF mean-field calculation already owns compatible periodic GDF
tensors, attach them to the transition space instead of rebuilding them:

.. code-block:: python

   from pyqed.pbc.gw import attach_pyscf_gdf_context

   attach_pyscf_gdf_context(space, pyscf_mf)
   operator = periodic_tda_operator(
       space,
       coulomb_component="pyscf_gdf",
   )

This remains an optional interoperability path: importing and running the
native ``"gdf"`` backend does not require PySCF.  For heavy all-electron
solids, PySCF GDF is currently the recommended production tensor engine;
native range-separated GDF remains substantially more expensive for tight
core Gaussian functions.

The builtin velocity backend is an all-electron canonical-momentum
implementation.  Calculations with nonlocal pseudopotentials must supply
velocity matrix elements that include the corresponding commutator
correction through ``transition_velocity=...``.  Full BSE remains dense; the
matrix-free TDA operator supports every commensurate momentum block, while
its optical absorption helper is restricted to :math:`q=0`.  Ordinary
:math:`q=0` BSE by itself does not include the phonon-assisted indirect
absorption edge of silicon.

Exciton-Phonon Feshbach Embedding
---------------------------------

``periodic_tda_operator`` exposes a finite-momentum matrix-free TDA block
:math:`H_{\mathrm{TDA}}(K)`.  For a mass-weighted phonon normal coordinate
:math:`Q_{q\nu}`, ``ExcitonPhononCoupling`` represents

.. math::

   M_{q\nu}
   = \frac{1}{\sqrt{2\omega_{q\nu}}}
     \frac{\partial H_{\mathrm{TDA}}}{\partial Q_{q\nu}},
   \qquad K\longrightarrow K+q.

``ExcitonPhononCoupling.from_finite_difference`` forms the derivative from
central differences of two displaced operators.  Analytic derivatives can be
passed directly through the same rectangular matrix-free interface.  Given
retained source excitons :math:`A^P_K`,
``ExcitonPhononChannel.from_coupling`` constructs the source-to-target
coupling

.. math::

   V^{q\nu}_{S,t}
   = \left\langle A^P_{S,K}\middle|M_{q\nu}^\dagger
     \middle|t,K+q\right\rangle .

Analytic one-body vertex
~~~~~~~~~~~~~~~~~~~~~~~~

``electron_phonon_mo_couplings`` transforms a self-consistent first-order AO
Fock matrix into the Bloch-band electron--phonon vertex.  For atom-centred
basis functions the implemented symmetric Pulay convention is

.. math::

   g_{mn}(k,q)
   = C_{m,k+q}^{\dagger}F_q^{[1]}(k)C_{n,k}
   -\frac{\epsilon_{m,k+q}+\epsilon_{n,k}}{2}
    C_{m,k+q}^{\dagger}S_q^{[1]}(k)C_{n,k}.

``PeriodicTDAElectronPhononDerivative`` lifts this one-electron vertex into
the transition basis.  With source exciton momentum :math:`Q` and target
momentum :math:`Q+q`, its matrix elements are

.. math::

   \left\langle v'c'k';Q+q\middle|H_{q\nu}^{[1]}
   \middle|vck;Q\right\rangle
   = \delta_{k'k}\delta_{v'v}g_{c'c}(k+Q,q)
   - \delta_{c'c}\delta_{k'+q,k}g_{vv'}(k-q,q)
   + K_{t't}^{[1]}.

The sparse implementation applies the electron and hole terms without
forming a dense transition-space matrix.  An analytic or externally computed
``kernel_derivative`` can supply :math:`K^{[1]}`.  If it is absent, metadata
records ``approximation="frozen_screening_one_body_fan"``.

For a Gamma-only Ewald, reciprocal, or native GDF KRHF reference,
``gamma_tda_electron_phonon_coupling`` constructs :math:`F^{[1]}` directly.
It combines analytic overlap, core-Hamiltonian, and fixed-density J/K nuclear
derivatives with the induced potential from periodic CPHF.  A normalized
mass-weighted mode :math:`e_{A\alpha}^{\nu}` is contracted as

.. math::

   \frac{\partial F}{\partial Q_{\nu}}
   =\sum_{A\alpha}
    \frac{e_{A\alpha}^{\nu}}{\sqrt{M_A}}
    \frac{\partial F}{\partial R_{A\alpha}},

before ``ExcitonPhononCoupling`` applies the zero-point factor
:math:`(2\omega_\nu)^{-1/2}`.  The native-GDF LiH electronic-derivative
validation uses a reciprocal-KRHF normal mode and is reproducible with

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gamma_electron_phonon.py \
       --output /private/tmp/pbc_gamma_gdf_electron_phonon.png

For native GDF, the AO electron-repulsion tensor is represented before
metric whitening as

.. math::

   (\mu\nu|\lambda\sigma)
   = \sum_{PQ} B_{P\mu\nu}^{*}(M^{-1})_{PQ}B_{Q\lambda\sigma}.

``gdf_derivative_factors`` exposes :math:`B`, :math:`B^{[1]}`,
:math:`M^{-1}`, and :math:`(M^{-1})^{[1]}` from the same periodic auxiliary
basis, reciprocal mesh, image domains, and screening controls as the SCF
builder.  The implemented derivative is

.. math::

   (\mu\nu|\lambda\sigma)^{[1]}
   = \sum_{PQ}\left[
       B_{P\mu\nu}^{[1]*}(M^{-1})_{PQ}B_{Q\lambda\sigma}
       +B_{P\mu\nu}^{*}(M^{-1})_{PQ}^{[1]}B_{Q\lambda\sigma}
       +B_{P\mu\nu}^{*}(M^{-1})_{PQ}B_{Q\lambda\sigma}^{[1]}
     \right],

and the fixed-density Hartree--Fock response follows from

.. math::

   J_{\mu\nu}^{[1]}
   = \sum_{\lambda\sigma}
     (\mu\nu|\lambda\sigma)^{[1]}D_{\sigma\lambda},
   \qquad
   K_{\mu\nu}^{[1]}
   = \sum_{\lambda\sigma}
     (\mu\lambda|\nu\sigma)^{[1]}D_{\sigma\lambda},

.. math::

   F^{[1]}_{\mathrm{explicit}}
   = h^{[1]}+J^{[1]}-\frac{1}{2}K^{[1]}.

The CPHF density response adds :math:`G[D^{[1]}]`.  Thus the one-body part
contains both explicit integral and orbital-relaxation response.

For a GDF TDA operator, ``kernel_derivative="bare_gdf"`` additionally
transforms the same differentiated four-index interaction to the fixed
reference MO basis and assembles

.. math::

   K_{t't}^{[1],\mathrm{bare}}
   = w_{t'}^{1/2}w_t^{1/2}\left[
       a_{\mathrm d}(v'c'|vc)^{[1]}
       -a_{\mathrm x}(c'c|v'v)^{[1]}
     \right].

This kernel term is explicitly a frozen-orbital bare-interaction derivative.
The MO-coefficient response is already represented in the one-body Fan term
and is not inserted a second time into this kernel contraction.

Analytic direct-RPA screening derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``kernel_derivative="screened_gdf"`` adds an analytic derivative of the
direct-RPA static screened-exchange term.  Define the diagonal transition-
energy matrix :math:`D`, the GDF transition Coulomb metric :math:`V`, and the
quadrature-weighted direct kernel

.. math::

   D_{tt'}=\Delta\epsilon_t\delta_{tt'},
   \qquad
   K_{tt'}=a_{\mathrm d}\sqrt{w_t}\,V_{tt'}\sqrt{w_{t'}}.

The direct Casida problem is

.. math::

   C Z_L=\Omega_L^2 Z_L,
   \qquad
   C=D^{1/2}(D+2K)D^{1/2}.

The implementation differentiates it as

.. math::

   C^{[1]}
   =(D^{1/2})^{[1]}(D+2K)D^{1/2}
   +D^{1/2}(D^{[1]}+2K^{[1]})D^{1/2}
   +D^{1/2}(D+2K)(D^{1/2})^{[1]}.

For nondegenerate poles,

.. math::

   \Omega_L^{[1]}
   =\frac{Z_L^\dagger C^{[1]}Z_L}{2\Omega_L},
   \qquad
   Z_L^{[1]}
   =\sum_{M\ne L}Z_M
    \frac{Z_M^\dagger C^{[1]}Z_L}
         {\Omega_L^2-\Omega_M^2}.

Both :math:`D^{[1]}` and :math:`K^{[1]}` are retained.  The former uses the
self-consistent KRHF orbital-energy derivative and the latter uses the
differentiated GDF interaction above.  With

.. math::

   P_{tL}=\frac{\sqrt{\Delta\epsilon_t}}{\sqrt{\Omega_L}}Z_{tL},
   \qquad
   M_{pL}=\left(a_{\mathrm d}\sqrt{w}\,V_p\right)^\dagger P_L,

the screened-exchange derivative is assembled without finite differences:

.. math::

   S_{pq}^{[1]}
   =\sum_L\left[
      \frac{M_{pL}^{[1]}M_{qL}^{*}
            +M_{pL}M_{qL}^{[1]*}}{\Omega_L}
      -\frac{M_{pL}M_{qL}^{*}\Omega_L^{[1]}}{\Omega_L^2}
     \right].

Analytic diagonal GW derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gamma_gdf_diagonal_self_energy_derivative`` differentiates the same
direct-RPA pole representation used by ``diagonal_correlation_self_energy``:

.. math::

   \Sigma_n^c(\omega)
   =\sum_{mL}\frac{|M_{nmL}|^2}{d_{mL}(\omega)},

.. math::

   (\Sigma_n^c)^{[1]}
   =\sum_{mL}\left[
      \frac{2\operatorname{Re}(M_{nmL}^*M_{nmL}^{[1]})}{d_{mL}}
      -\frac{|M_{nmL}|^2d_{mL}^{[1]}}{d_{mL}^2}
     \right].

For occupied and virtual intermediate states, respectively,

.. math::

   d_{mL}^{[1]}
   =\omega^{[1]}-\epsilon_m^{[1]}+\Omega_L^{[1]},
   \qquad
   d_{mL}^{[1]}
   =\omega^{[1]}-\epsilon_m^{[1]}-\Omega_L^{[1]}.

``gamma_gdf_g0w0_energy_derivative`` applies the chain rule to the default
Hartree--Fock-reference on-shell convention,

.. math::

   (E_n^{G_0W_0})^{[1]}
   =\epsilon_n^{[1]}
    +\operatorname{Re}\left[
       \left.\frac{\partial\Sigma_n^c}{\partial Q_\nu}\right|_{\omega}
       +\left.\frac{\partial\Sigma_n^c}{\partial\omega}\right|_{Q_\nu}
        \epsilon_n^{[1]}
     \right].

This corresponds to ``diagonal_g0w0`` with its default on-shell,
non-linearized, non-root-solved pole integration.

Primitive finite-q response and commensurate validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gdf_q_derivative`` is the production analytic finite-q KRHF driver for an
all-electron three-dimensional cell with the full reciprocal GDF kernel.  It
builds :math:`k\rightarrow k+q` overlap, core-Hamiltonian, and fixed-density
Fock blocks directly in the primitive cell and passes them to primitive
finite-q CPHF.  No supercell AO or auxiliary tensor is allocated.  For a
translated right AO at :math:`R`, the overlap and kinetic derivatives use

.. math::

   X_q^{[1]}(k)_{mu\nu}
   =\sum_R e^{ik\cdot R}\left[
      u_\mu\!\cdot\!\nabla_\mu
      +e^{iq\cdot R}u_\nu\!\cdot\!\nabla_{\nu R}
    \right]X(\mu_0,\nu_R),
   \qquad X\in\{S,T\}.

The short-range nuclear attraction adds the motion of every nuclear image,

.. math::

   V_{\mathrm{sr},q}^{[1]}(k)
   =\sum_R e^{ik\cdot R}\left[
      u_\mu\!\cdot\!\nabla_\mu
      +e^{iq\cdot R}u_\nu\!\cdot\!\nabla_{\nu R}
      +\sum_{LA}e^{iq\cdot L}u_A\!\cdot\!\nabla_{A L}
    \right]V_{\mathrm{sr}}(\mu_0,\nu_R),

and the reciprocal nuclear term uses :math:`K=G+q`,

.. math::

   \delta\rho_{\mathrm{nuc},q}(K)
   =-i\sum_A Z_A(K\!\cdot\!u_A)e^{-iK\cdot R_A},

.. math::

   V_{\mathrm{lr},q}^{[1]}(k)
   =-\frac{4\pi}{\Omega}\sum_{G:\,K=G+q\ne0}
      \frac{e^{-K^2/(4\eta^2)}}{K^2}
      \delta\rho_{\mathrm{nuc},q}(K)
      \rho_{\mu\nu,k}(-K)
     +V_{\mathrm{basis},q}^{[1]}(k).

The fixed-density GDF derivative contracts primitive momentum blocks.  Terms
in which a differentiated factor is conjugated use the opposite star,

.. math::

   D_q(B^*)=(D_{-q}B)^*,

and the Hartree auxiliary metric is transformed covariantly,

.. math::

   \widetilde M^{-1}_{QQ'}
   =E_Q^T M^{-1}E_{Q'}^*
   =E_{-Q}^{\dagger}M^{-1}E_{-Q'}.

These two details are required on a general non-self-opposite k mesh.  The
periodic Madelung exchange derivative is included as

.. math::

   V_{M,q}^{[1]}(k)
   =-\frac{v_M}{2}\left[
      S_q^{[1]}(k)D(k)S(k)
      +S(k+q)D(k+q)S_q^{[1]}(k)
    \right].

``commensurate_gdf_q_derivative`` remains the independent reference and the
fallback for range-separated or GTH-pseudopotential derivatives.  For a
Born--von Karman
supercell with :math:`N=N_1N_2N_3` primitive translations :math:`R`, define

.. math::

   U_k(R\mu,\nu)
   =\frac{1}{\sqrt{N}}e^{i k\cdot R}\delta_{\mu\nu}.

All primitive k points fold to one common supercell twist :math:`\kappa`,

.. math::

   k=\kappa+G_k^{\mathrm{SC}},

where :math:`G_k^{\mathrm{SC}}` is a supercell reciprocal vector.  The twist is
Gamma for an odd Gamma-centred mesh, but an even Monkhorst--Pack mesh generally
folds to an antiperiodic boundary condition instead.  The primitive k-point
density is embedded in this one-twist supercell AO basis as

.. math::

   D^{\mathrm{SC}}=\sum_k U_kD(k)U_k^\dagger.

One-k GDF derivatives are evaluated at this fixed density.  For a
primitive-cell mass-weighted phonon eigenvector :math:`e_{A\alpha,\nu}(q)`,
the traveling-wave supercell perturbation is

.. math::

   X_{q\nu}^{[1],\mathrm{SC}}
   =\sum_{R A\alpha}e^{i q\cdot R}
    \frac{e_{A\alpha,\nu}(q)}{\sqrt{M_A}}
    \frac{\partial X^{\mathrm{SC}}}{\partial R_{A\alpha}},
   \qquad X\in\{S,F_{\mathrm{explicit}}\}.

The primitive AO blocks passed to finite-q CPHF are

.. math::

   X_{q\nu}^{[1]}(k)
   =U_{k+q}^\dagger X_{q\nu}^{[1],\mathrm{SC}}U_k.

CPHF then adds the induced potential,

.. math::

   F_{q\nu}^{[1]}(k)
   =F_{q\nu}^{[1],\mathrm{explicit}}(k)
    +G_q[D_{q\nu}^{[1]}](k),

while retaining the finite-q Pulay term
:math:`-\epsilon_{i k}S_{q\nu}^{[1]}(k)` in the orbital-response equation.
At self-opposite zone-boundary momenta, the implementation projects the
blocks onto

.. math::

   X_q^{[1]}(k+q)=X_q^{[1]}(k)^\dagger

and records the pre-projection residual.

``commensurate_tda_electron_phonon_coupling`` passes these blocks to the TDA
exciton--phonon assembly.  With ``kernel_derivative="bare_gdf"``, the cached
primitive-cell factor response supplies the rectangular bare-kernel map from
the source exciton sector :math:`Q` to :math:`Q+q`.  For supercell orbital-pair
factors :math:`B_{ab}`, the implementation evaluates

.. math::

   D_q(ab|cd)
   =(D_qB_{ab})^T M^{-1}B_{cd}^{*}
    +B_{ab}^{T}(D_qM^{-1})B_{cd}^{*}
    +B_{ab}^{T}M^{-1}(D_{-q}B_{cd})^{*}.

The last term uses

.. math::

   D_q(B^*)=(D_{-q}B)^*,

which is essential away from self-opposite zone-boundary momenta.  Replacing
:math:`D_{-q}` by :math:`D_q` happens to pass a two-point mesh, where
:math:`q\equiv -q`, but gives the wrong derivative on a general mesh.

The bare kernel is then

.. math::

   K_{t't}^{[1],\mathrm{bare}}(Q+q,Q)
   =\sqrt{w_{t'}w_t}\left[
      a_{\mathrm d}(v'c'|vc)^{[1]}_q
      -a_{\mathrm x}(c'c|v'v)^{[1]}_q
    \right].

Only requested Bloch-orbital pair factors are transformed; no four-index AO
tensor is formed.  With ``kernel_derivative="bare_gdf"``, the MO coefficients
and BSE screening remain frozen.

``GDFQDerivativeFactors`` is the shared q-resolved consumer for these
contractions.  For every requested primitive Bloch-orbital pair it caches

.. math::

   \mathcal B_{ab}
   =\left(B_{ab},D_qB_{ab},D_{-q}B_{ab}\right),

and applies :math:`D_qM^{-1}` only when an ERI derivative is requested.  Bare,
screened, and continuum contractions for one nuclear perturbation reuse the
same cache; ``q_factor_info`` reports its pair count and retained bytes.  This
is a direct primitive-cell producer when ``reciprocal_kernel="full"``.  For
source transfer :math:`Q`, it evaluates

.. math::

   B^{(q)}_{Q+q,Q}
   =\langle\chi_{Q+q}|v|\rho_Q^{(q)}\rangle
    +\langle\chi_{Q+q}^{(-q)}|v|\rho_Q\rangle,

and the off-diagonal inverse-metric response

.. math::

   (M^{-1})^{(q)}_{Q+q,Q}
   =-M^{-1}_{Q+q}M^{(q)}_{Q+q,Q}M^{-1}_Q.

The AO-pair derivative carries the phonon phase on translated centers before
the reciprocal contraction.  Consequently, its largest factor arrays scale
as :math:`O(n_{\mathrm{aux}}n_{\mathrm{AO}}^2)`, independent of the number of
Born--von Karman cells.  ``temporary_supercell_factor_bytes`` is zero on this
path.  Range-separated short-range factor derivatives are not yet direct and
fall back explicitly to the commensurate reference producer.

``kernel_derivative="screened_gdf"`` additionally evaluates the static
off-diagonal direct-RPA response.  The zero-order screening blocks are built
from the primitive-cell q-resolved GDF factors, which preserves exactly the
same auxiliary gauges and Coulomb convention as the source BSE operator.  In
each transfer sector :math:`s`, define

.. math::

   \widetilde V_s
   =\sqrt{w_s}V_s\sqrt{w_s},\qquad
   Z_s=(D_s+2a_{\mathrm d}\widetilde V_s)^{-1},

where :math:`D_s` contains independent electron--hole transition energies.
For a phonon connecting transfer sectors :math:`a` and :math:`b`, the
rectangular RPA matrix derivative is

.. math::

   C_{ba}^{[1]}
   =H_{ba}^{[1]}
    +2a_{\mathrm d}\sqrt{w_b}V_{ba}^{[1]}\sqrt{w_a},

and differentiation of the inverse gives

.. math::

   Z_{ba}^{[1]}=-Z_b C_{ba}^{[1]}Z_a.

For external electron and hole pair couplings
:math:`\widetilde c_e` and :math:`\widetilde c_h`, the implemented induced
interaction derivative is

.. math::

   W_{\mathrm{ind},ba}^{[1]}(e,h)
   =a_{\mathrm d}^{2}\left[
      (\widetilde c_e^\dagger)^{[1]}Z_a\widetilde c_h
      +\widetilde c_e^\dagger Z_b\widetilde c_h^{[1]}
      -\widetilde c_e^\dagger Z_b C_{ba}^{[1]}Z_a\widetilde c_h
   \right].

The one-body particle--hole response obeys the independent momentum rule

.. math::

   D_qH^{(0)}_{p'p}\ne0\quad\Longrightarrow\quad p'=p+q.

At a self-opposite mesh momentum, :math:`q\equiv-q`, the two screened-kernel
orientations are contracted independently and projected onto the exact star
relation,

.. math::

   K_q^{[1]}(Q+q,Q)\leftarrow\frac{1}{2}\left[
   K_{q,\mathrm{raw}}^{[1]}(Q+q,Q)
   +K_{q,\mathrm{raw}}^{[1]}(Q,Q+q)^\dagger\right].

The maximum absolute difference before this projection is retained as
``raw_star_residual`` in the derivative metadata.

The screened-kernel contraction above requests :math:`b=a-q`.  Consequently,

.. math::

   H_{a-q,a}^{[1]}=0
   \quad\text{unless}\quad q\equiv -q,

so the central one-body term survives at Gamma and self-opposite zone-boundary
momenta but is absent for a general traveling wave.  The Coulomb response
:math:`V_{ba}^{[1]}` and both external vertex derivatives use analytic
primitive-cell three-center and auxiliary-metric derivatives for the full
reciprocal kernel.  This
transition-space form is gauge independent and avoids differentiating
individual RPA poles, so degenerate screening modes do not require a special
eigenvector convention.

``validate_commensurate_gdf_screened_tda_kernel_derivative`` checks the full
rectangular kernel with independently displaced cosine and sine supercells,

.. math::

   K_q^{[1]}=K_{\cos}^{[1]}+iK_{\sin}^{[1]}.

The zero-order RPA blocks remain in the primitive q-resolved representation.
The raw displaced one-body derivative enters the resolvent contraction with
no momentum-sector projection.  The expected :math:`p\to p+q` mask is used
only to report ``one_body_leakage_norm``.  Before any displaced calculation,
the validator requires representation equality of the primitive and
commensurate-supercell references.  With
:math:`\mathcal U=(U_{k_1},\ldots,U_{k_N})`, it monitors

.. math::

   r_X=\frac{\lVert X^{\mathrm{SC}}-
       \mathcal U\,\operatorname{diag}_k[X(k)]\mathcal U^\dagger\rVert_F}
       {\lVert\mathcal U\,\operatorname{diag}_k[X(k)]
       \mathcal U^\dagger\rVert_F},
   \qquad X\in\{S,h,F\},

and the density stationarity residual

.. math::

   r_D=\frac{\lVert D[F^{\mathrm{SC}}[D_0]]-D_0\rVert_F}
       {\lVert D_0\rVert_F}.

The largest residual must not exceed ``representation_tol``; the default is
:math:`10^{-7}`.  This turns insufficient real-space or reciprocal cutoffs
into an explicit failure instead of silently folding the mismatch into the
physical q-resolved derivative.  ``zero_density_residual`` separately records
the converged zero-supercell SCF drift.  A non-self-opposite three-k regression
checks the bare, screened, and total components separately.  The reproducible
diagnostic is

.. code-block:: bash

   PYTHONPATH=. python examples/pbc_general_q_gdf_derivative_validation.py

The one-electron reciprocal nuclear domain and the GDF FFT grid have different
folding rules.  For primitive one-electron index bound :math:`n_{c,i}` and
supercell multiplier :math:`L_i`, KRHF uses the anisotropic bound

.. math::

   n_{c,i}^{\mathrm{SC}}=L_i n_{c,i}.

This preserves the primitive nuclear reciprocal vectors without extending
unchanged transverse directions.  GDF additionally contains every folded
:math:`G+q` channel and keeps an inversion-symmetric grid.  For primitive GDF
half-width :math:`n_c`, the supercell half-width is

.. math::

   n_{c,i}^{\mathrm{SC}}
   =L_i n_c+\left\lfloor\frac{L_i}{2}\right\rfloor,
   \qquad M_i^{\mathrm{SC}}=2n_{c,i}^{\mathrm{SC}}+1.

For even :math:`L_i`, inversion symmetry retains both Nyquist endpoints.  This
is required for exact adjoint symmetry at self-opposite momenta, although the
extra finite-cutoff endpoint means primitive and supercell exchange agree only
after reciprocal convergence.  The representation residual gate detects when
that endpoint remains material.

Both the reciprocal and real-space image domains must be converged for a
quantitative finite-:math:`q` derivative.  In particular, ``pair_cut=0`` keeps
only same-cell AO pairs and is intended for inexpensive tests; it breaks
primitive/supercell representation equality and can artificially suppress the
screened-kernel derivative.  The general-:math:`q` diagnostic therefore
defaults to ``recip_cut=2`` and ``pair_cut=2`` and writes both controls and the
largest reference residual to its JSON output.  Even self-opposite meshes can
require a larger reciprocal cutoff because of the Nyquist endpoint; the
two-k regression uses ``recip_cut=8``.  Production calculations should verify
the result against at least one larger value of each cutoff.

The primitive driver contracts :math:`S^{[1]}`, :math:`h^{[1]}`, and
:math:`V_{\mathrm{HF}}^{[1]}` directly into one traveling-wave direction.
Its retained GDF arrays scale with primitive AO-pair and auxiliary blocks;
``temporary_supercell_nao`` and ``temporary_supercell_naux`` are zero.  The
commensurate gradient API remains available for validation and for unsupported
kernels.

Current fidelity boundary
~~~~~~~~~~~~~~~~~~~~~~~~~

The pole-resolved screened-interaction and GW self-energy derivatives remain
restricted to a native-GDF, Gamma-only reference and nondegenerate Casida
poles.  They differentiate transition energies, bare Coulomb vertices, RPA
poles, residues, and GW denominators.  The finite-q BSE derivative instead
uses the static transition-space resolvent and has no nondegenerate-pole
restriction.  Derivatives of finite-size head/wing corrections,
eigenvalue-self-consistent GW, linearized :math:`Z` factors, and root-solved
quasiparticle equations are not included.

The automatic ``"bare_gdf"`` option leaves BSE screening frozen;
``"screened_gdf"`` adds the static direct-RPA response above, and an external
``kernel_derivative`` may supply a different response model.  Full-reciprocal
all-electron nonzero-q one-electron integrals, moving-basis overlap,
fixed-density GDF Fock derivatives, finite-q CPHF, bare-GDF kernel derivatives,
and off-diagonal static screening use the primitive producer.  Range-separated
short-range and GTH-pseudopotential derivatives retain the commensurate
fallback.  External Bloch MO
coefficients in Coulomb vertices remain fixed; orbital mixing in the
independent transition resolvent is included through :math:`H^{[1]}`.
Dynamical screening, frequency-dependent BSE kernels, primitive finite-q GTH
derivatives, and a range-separated primitive response remain future work.

The electron--phonon convention follows F. Giustino, *Rev. Mod. Phys.*
**89**, 015003 (2017), doi:10.1103/RevModPhys.89.015003.  The exciton basis
vertex follows H.-Y. Chen, D. Sangalli, and M. Bernardi,
*Phys. Rev. Lett.* **125**, 107401 (2020),
doi:10.1103/PhysRevLett.125.107401.  This implementation is an adaptation to
the PyQED TDA transition ordering and does not claim the omitted dynamical
quasiparticle, one-electron primitive-DFPT, or short-range response terms of a
complete first-principles calculation.

The periodic GDF factorization is adapted from Q. Sun *et al.*,
*J. Chem. Phys.* **147**, 164119 (2017), doi:10.1063/1.4998644.  The analytic
derivative is a PyQED implementation of the differentiated factorization,
not a reproduction of a complete GDF--GW--BSE force formalism.
The GW pole convention follows L. Hedin, *Phys. Rev.* **139**, A796--A823
(1965), doi:10.1103/PhysRev.139.A796.  The analytic derivatives above are an
adaptation to PyQED's direct-RPA pole model, not a reproduction of a published
periodic DFPT--GW implementation.

Selected discrete excitons :math:`A^P_{K+q}` in the target block are removed
from the continuum with

.. math::

   Q_{K+q}=I-A^P_{K+q}(A^P_{K+q})^\dagger .

``ProjectedTDAContinuum`` then evaluates the pole-free Feshbach self-energy
without diagonalizing the target transition space:

.. math::

   \Sigma^R_{q\nu}(E)
   = V^{q\nu}Q_{K+q}
     \left[E+i\eta-Q_{K+q}H_{\mathrm{TDA}}(K+q)Q_{K+q}\right]^{-1}
     Q_{K+q}(V^{q\nu})^\dagger .

The projected resolvent is applied with GMRES.  ``ExcitonPhononContinuum``
sums emission and absorption channels using the one-phonon Fan convention,

.. math::

   \Sigma^R(E)
   = \sum_{q\nu}\left[
       (N_{q\nu}+1)\Sigma^R_{q\nu}(E-\omega_{q\nu})
       +N_{q\nu}\Sigma^R_{q\nu}(E+\omega_{q\nu})
     \right],

where ``bose_occupation`` evaluates

.. math::

   N_{q\nu}(T)
   =\frac{1}{\exp[\omega_{q\nu}/(k_{\mathrm B}T)]-1}.

``ExcitonPhononChannel.thermal_from_coupling`` evaluates this factor from a
temperature in kelvin.  ``ExcitonPhononContinuum.run_spectrum`` constructs the
``FeshbachEmbedding`` and evaluates the active spectrum in one call.

For retained source and target TDA eigenvectors :math:`A^S_K` and
:math:`A^{S'}_{K+q}`, ``ExcitonPhononCoupling.between`` returns the full complex
exciton-state matrix

.. math::

   g_{S'S\nu}(K,q)
   =\left(A^{S'}_{K+q}\right)^\dagger
    M_{q\nu}A^S_K,

rather than only a diagonal or lowest-root coupling.

and provides the corresponding finite-space memory kernel

.. math::

   \mathcal K(t)
   = \sum_{q\nu}\left[
       (N_{q\nu}+1)e^{-i\omega_{q\nu}t}
       +N_{q\nu}e^{+i\omega_{q\nu}t}
     \right]
     V^{q\nu}Qe^{-iQH_{\mathrm{TDA}}Qt}Q(V^{q\nu})^\dagger .

``TotalMomentumSector`` enforces the fixed-sector bookkeeping

.. math::

   P_{\mathrm{tot}}
   =K+\sum_{q\nu}n_{q\nu}q\pmod G.

For the supplied finite TDA operators and couplings, the projected resolvent,
memory kernel, and one-phonon auxiliary Hamiltonian are exact up to iterative
solver tolerance.  The current implementation is an adaptation of Feshbach
projection and the exciton-phonon Fan self-energy, not a complete
first-principles vibronic solver.  It uses TDA states, requires the displaced
or analytic coupling operators to be supplied, and omits Debye--Waller,
multiphonon, self-consistent vertex, and non-TDA response terms.  Long-time
production propagation will also require memory-kernel compression; the
current Krylov propagation is the finite-space validation path.

The projection follows H. Feshbach, *Ann. Phys.* **5**, 357--390 (1958),
doi:10.1016/0003-4916(58)90007-1.  The BSE basis follows M. Rohlfing and
S. G. Louie, *Phys. Rev. B* **62**, 4927--4944 (2000),
doi:10.1103/PhysRevB.62.4927.  The one-phonon self-energy convention is adapted
from H.-Y. Chen, D. Sangalli, and M. Bernardi, *Phys. Rev. Lett.* **125**,
107401 (2020), doi:10.1103/PhysRevLett.125.107401.

LiH convergence and embedded-spectrum benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/pbc_lih_exciton_phonon_convergence.py`` runs the complete compact
rocksalt-LiH reference workflow for
:math:`N_k\times1\times1` meshes with :math:`N_k=2,4,6` and reciprocal cutoffs
2, 3, and 4.  Every mesh uses the same self-opposite zone-boundary momentum;
the older first-nonzero-momentum choice compared different physical
:math:`q` points as :math:`N_k` changed.  The script builds GDF-KRHF, source
and target TDA operators, the analytic finite-q screened derivative, the
complex :math:`g_{S'S\nu}` matrix, and a 300 K Feshbach spectrum.  The default
localized three-function all-electron basis is a dependency-free engine
benchmark, not a quantitative basis for LiH;
``--basis sto-3g`` selects the larger molecular basis at substantially greater
real-space cost.  The phonon eigenvector and frequency are supplied benchmark
inputs, so this example does not claim an ab initio phonon calculation.  A
one-dimensional k-line also measures an engine trend rather than a converged
three-dimensional Brillouin-zone integral.

.. code-block:: console

   PYTHONPATH=. python examples/pbc_lih_exciton_phonon_convergence.py

For the recorded default run, the derivative-plus-kernel wall times at cutoff
4 were 1.30, 2.17, and 3.50 s for :math:`N_k=2,4,6`; the corresponding
GDF-KRHF times were 0.25, 1.29, and 3.92 s.  The retained MO-pair vectors used
7.88, 31.50, and 70.88 KiB, and the primitive derivative-engine caches used
194.5, 409.4, and 644.5 KiB.  No temporary supercell AO, auxiliary, or
derivative-factor arrays were constructed in these production cases.  The
maximum zone-boundary couplings at cutoff 4 were 266.58, 287.39, and
299.82 meV, respectively.

The displaced-supercell validator is a separate :math:`N_k=4`, cutoff-9 run;
cutoff 4 is intentionally not accepted because its finite-grid Nyquist
endpoint leaves a :math:`1.45\times10^{-4}` representation mismatch.  At
cutoff 9 the largest primitive/supercell reference residual is
:math:`8.51\times10^{-8}`.  With displacement :math:`h=10^{-3}` bohr, the
independent unprojected-supercell validation gives relative errors

.. math::

   \epsilon_{\mathrm{total}}=2.28\times10^{-5},\qquad
   \epsilon_{\mathrm{bare}}=2.20\times10^{-5},\qquad
   \epsilon_{\mathrm{screened}}=8.14\times10^{-4}.

The screened component is much smaller than the bare component, so its
relative finite-difference error is the least stable diagnostic.  The
calculation writes JSON data plus publication-ready PNG and PDF figures under
``/private/tmp`` by default.  ``--skip-validation`` omits the expensive
commensurate reference; ``--validation-recip-cut`` controls its independent
reciprocal cutoff.

Three-dimensional LiH qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The case ``lih-rocksalt-2k-svp-solid`` is the realistic three-dimensional GDF,
KRHF, GW, and BSE qualification.  It uses the rocksalt primitive cell at
:math:`a=7.72` bohr, an uncontracted PyQED/PySCF representation match, and the
bundled def2-SV(P)-JKFIT auxiliary basis.  Molecular def2-SVP is severely
linearly dependent in this dense solid: the unmodified :math:`2\times2\times2`
overlap has a minimum eigenvalue near :math:`7.1\times10^{-10}` and condition
numbers up to :math:`6.0\times10^9`.  The named ``-solid`` case therefore
removes primary primitives with exponent below 0.1.  This leaves nine AOs,
raises the minimum overlap eigenvalue to 0.0936, and must be described as a
solid-adapted def2-SVP benchmark rather than standard def2-SVP.

At native ``gdf_precision=1e-8`` against a fixed PySCF cell precision of
:math:`10^{-12}`, the :math:`2\times2\times2` result gives

.. math::

   \max|\Delta J|=3.34\times10^{-4}\ {\rm meV},\qquad
   \max|\Delta K|=1.81\times10^{-4}\ {\rm meV},

.. math::

   |\Delta E_{\rm KRHF}|=7.55\times10^{-10}\ E_h,\qquad
   \max|\Delta E^{\rm QP}|=5.81\times10^{-6}\ {\rm meV},

with maximum BSE :math:`A/B` matrix difference
:math:`4.92\times10^{-9}\ E_h`.  Def2-SV(P)-JKFIT and the bundled universal
JK-fit alias give the same Li/H auxiliary set.  Def2-SVP-RIFIT is not a
suitable replacement for this JK/GW path; its measured :math:`K` mismatch is
2.17 meV.

The larger-mesh results expose the current accuracy and scaling boundary.  At
:math:`4\times2\times2`, the worst auxiliary-projected pair metric differs by
:math:`1.15\times10^{-3}`, but the propagated errors remain

.. math::

   |\Delta E_{\rm KRHF}|=1.15\times10^{-8}\ E_h,\qquad
   \max|\Delta E^{\rm QP}|=3.02\times10^{-3}\ {\rm meV},

and the maximum BSE matrix difference is :math:`1.07\times10^{-6}\ E_h`.
Raw AO overlap and AO-pair Fourier tensors agree with PySCF to approximately
:math:`10^{-9}`; tightening pair-image, pair-factor, and reciprocal-grid
controls does not remove the residual.  It is localized to the periodic
auxiliary metric/three-center projection at general off-special :math:`k`
points, not the AO-pair Fourier engine.

The materialized factor footprint follows :math:`N_k^2`: 5.72, 22.89, and
366.28 MB for 8, 16, and 64 k points.  Native/PySCF GDF build times are
4.08/2.89, 9.51/9.84, and 124.22/21.81 s, respectively.  Thus the native
builder reaches PySCF parity at 16 k points but is :math:`5.7\times` slower at
64 k points.  The :math:`4\times4\times4` :math:`K` mismatch is 0.00862 meV.
These results qualify small and intermediate meshes while identifying
multi-q auxiliary construction and storage as the next production bottleneck.

``examples/pbc_lih_3d_derivative_validation.py`` is a separate finite-q
qualification with a compact three-function all-electron basis and the exact
full-reciprocal derivative kernel at cutoff 9.  On a
:math:`2\times2\times2` mesh, the primitive-cell and commensurate-supercell
total Fock derivatives differ by :math:`8.00\times10^{-6}` in relative
Frobenius norm, and their screened BSE derivatives differ by
:math:`1.27\times10^{-6}`.  An independent displaced-SCF check gives

.. math::

   \epsilon_{\rm total}=1.70\times10^{-4},\qquad
   \epsilon_{\rm bare}=1.74\times10^{-4},\qquad
   \epsilon_{\rm screened}=1.40\times10^{-5}.

The measured primitive/supercell zero-order representation residual is
:math:`5.60\times10^{-7}`.  The production primitive derivative takes 37.6 s
and caches 7.20 MB; the commensurate derivative takes 257.5 s, and one
finite-displacement validation point takes 502.1 s.  The supercell route is
therefore a release-qualification reference, not the production engine.
``examples/plot_pbc_lih_3d_qualification.py`` regenerates the combined PNG and
PDF convergence figure from the JSON outputs.

Photoemission Spectral Functions
--------------------------------

After an exact-pole or analytic-continuation quasiparticle calculation,
``KGW.spectral_function`` evaluates the frequency-dependent correlation
self-energy again with the exact RPA-pole representation.  For the current
periodic Hartree-Fock reference, the time-ordered diagonal Green function is

.. math::

   G_{n k}(\omega)
   =
   \left[
   \omega-\epsilon_{n k}-i s_{n k}\eta
   -\Sigma^c_{n k}(\omega)
   \right]^{-1},

where :math:`s_{n k}=+1` for occupied bands and :math:`s_{n k}=-1` for
virtual bands.  The positive spectral branch is

.. math::

   A_{n k}(\omega)
   = \frac{s_{n k}}{\pi}\operatorname{Im}G_{n k}(\omega).

For example, an occupied spectrum referenced to the valence-band maximum is

.. code-block:: python

   gw = KGW(mf, eta=0.01).g0w0(
       backend="periodic",
       frequency_integration="poles",
       coulomb_component="gdf",
       direct_scale=1.0,
   )
   spectrum = gw.spectral_function(
       binding_grid=np.linspace(0.0, 80.0, 1601),
       units="ev",
       bands=[0, 1],
       energy_reference="vbm",
   )

``spectrum.spectral_function`` has shape ``(ntarget, nenergy)`` and retains
the selected ``(k_index, band_index)`` pairs in ``spectrum.targets``.  The
``signal`` field is the occupied target sum with spin degeneracy two and
uniform :math:`1/N_k` weights.  ``energy_reference`` may be ``"vbm"``,
``"fermi"``, ``"zero"``, or an explicit value.  Screened interactions and
orbital-pair couplings from an exact-pole ``KGW`` run are reused by the
spectral calculation.

PySCF 2.12 exposes periodic ``KRGWAC`` quasiparticle energies and the
imaginary-axis correlation self-energy, but not a real-axis PES driver.  The
independent benchmark therefore compares :math:`\Sigma^c(i\omega)`,
linearized quasiparticle poles, and spectra reconstructed with the same
two-pole analytic-continuation model.

Independent PySCF KGW benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/pbc_gw_pyscf_benchmark.py`` performs an end-to-end comparison in
which PyQED and PySCF independently build GDF, converge KRHF, and evaluate
linearized analytic-continuation G0W0.  The primary reported residual is

.. math::

   \Delta^{\mathrm{e2e}}_{n\mathbf{k}}
   = E^{\mathrm{QP,PyQED(native)}}_{n\mathbf{k}}
   - E^{\mathrm{QP,PySCF}}_{n\mathbf{k}}.

Two benchmark-only controls identify its origin.  The head-aligned residual
uses native PyQED KRHF, GDF, and GW but substitutes the PySCF grid-gradient
:math:`\mathbf q\rightarrow 0` head,

.. math::

   \Delta^{\mathrm{head}}_{n\mathbf{k}}
   = E^{\mathrm{QP,PyQED(native;PySCF\ head)}}_{n\mathbf{k}}
   - E^{\mathrm{QP,PySCF}}_{n\mathbf{k}},

while the solver-aligned residual also supplies the PySCF KRHF orbitals and
GDF factors to the PyQED self-energy solver.  These controls are diagnostics;
the production ``"gdf"`` path and ``"builtin_gradient"`` head do not depend
on PySCF.

For rocksalt LiH with the solid-pruned def2-SVP orbital basis,
def2-SVP-JKFIT with ``aux_min_exponent=0.075``, a :math:`4\times2\times2`
mesh, precision :math:`10^{-12}`, 24 imaginary-frequency points, and finite
size corrections enabled, the maximum residuals are

.. math::

   \begin{aligned}
   \max_{n\mathbf{k}} |\Delta^{\mathrm{e2e}}_{n\mathbf{k}}|
   &= 2.1602\times10^{-2}\ \mathrm{meV},\\
   \max_{n\mathbf{k}} |\Delta^{\mathrm{head}}_{n\mathbf{k}}|
   &= 1.6711\times10^{-4}\ \mathrm{meV},\\
   \max_{n\mathbf{k}} |\Delta^{\mathrm{solver}}_{n\mathbf{k}}|
   &= 4.9987\times10^{-8}\ \mathrm{meV}.
   \end{aligned}

Thus the remaining end-to-end difference is dominated by the independent
native finite-size gradient head, rather than GDF or the bulk GW self-energy.
Run the qualification calculation and write JSON plus PDF/PNG diagnostics
with

.. code-block:: console

   PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     python examples/pbc_gw_pyscf_benchmark.py \
       --kmesh 4,2,2 --ac-nw 24 --workers 8

Diamond k-mesh stress test
~~~~~~~~~~~~~~~~~~~~~~~~~~

The covalent-solid stress test uses primitive-cell diamond at
:math:`a=6.74` bohr, STO-3G orbitals, the def2-SV(P)-JKFIT auxiliary basis,
24 imaginary-frequency points, and matched PyQED/PySCF finite-size
conventions.  It is an implementation and scaling test, not a predictive
diamond band-gap calculation.  The unpruned periodic auxiliary metric is
not the source of the initial discrepancy.  A raw q-resolved comparison gives

.. math::

   \max_{\mathbf q}
   \frac{\|J_{2c}^{\rm PyQED}(\mathbf q)
   -J_{2c}^{\rm PySCF}(\mathbf q)\|_F}
   {\|J_{2c}^{\rm PySCF}(\mathbf q)\|_F}
   =2.30\times10^{-13}.

The earlier :math:`1.23\times10^{-5}` factor-metric and :math:`0.217` meV
exchange residuals came from comparing PyQED's deterministic eigensolver with
PySCF's q-dependent mixture of Cholesky and eigendecomposition in a nearly
null auxiliary subspace.  At precision :math:`10^{-12}`, the automatic
:math:`10^{-13}` floor gives identical retained ranks in a matched-eigensolver
comparison.  The factor-metric residual is :math:`8.15\times10^{-8}`, the
exchange residual is :math:`5.37\times10^{-4}` meV, and the native KRHF total
energy residual is :math:`7.98\times10^{-4}` meV without auxiliary pruning.

At :math:`2\times2\times2`, the unpruned end-to-end GW residual is
:math:`9.53\times10^{-3}` meV.  The head-aligned and solver-aligned controls
are :math:`8.86\times10^{-4}` meV and :math:`9.26\times10^{-5}` meV,
respectively.  Use ``--pyscf-metric-eig`` for this rank-matched benchmark;
this option affects only the external validation reference.  The raw metric
diagnostic is reproducible with

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gdf_j2c_diagnostic.py \
     --case diamond --kmesh 2,2,2 --precision 1e-12

and the unpruned end-to-end comparison with

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gw_pyscf_benchmark.py \
     --case diamond --kmesh 2,2,2 --precision 1e-12 \
     --aux-min-exponent 0 --pyscf-metric-eig

The sampled quasiparticle gap is defined as

.. math::

   E_g^{\mathrm{QP}}(n)
   = \min_{\mathbf{k}} E_{c\mathbf{k}}^{\mathrm{QP}}(n)
   - \max_{\mathbf{k}} E_{v\mathbf{k}}^{\mathrm{QP}}(n).

For cubic :math:`n^3` Monkhorst--Pack meshes, the PySCF values are
:math:`19.2999`, :math:`20.8977`, and :math:`13.7553` eV for
:math:`n=1,2,3`; the corresponding maximum PyQED--PySCF residuals are
:math:`0.0902`, :math:`0.00941`, and :math:`0.01021` meV.  Thus cross-code
agreement is stable on the multi-k meshes, but the physical gap is not
k-mesh converged and shows a strong odd/even sampling effect.  A production
gap requires a larger, symmetry-reduced mesh and a converged orbital basis.

The convergence figure is reproducible from the saved benchmark JSON files:

.. code-block:: console

   PYTHONPATH=. python examples/plot_pbc_gw_kmesh_convergence.py \
     /private/tmp/pbc_diamond_111_pruned_pyqed_pyscf_kgw.json \
     /private/tmp/pbc_diamond_222_pruned_pyqed_pyscf_kgw.json \
     /private/tmp/pbc_diamond_333_pruned_pyqed_pyscf_kgw.json \
     --output /private/tmp/pbc_diamond_gw_kmesh_convergence

The runnable native rocksalt LiH workflow builds GDF-KRHF, exact-pole G0W0,
the intrinsic spectral function, and the matrix-element-weighted
photoemission signal over the 0--10 eV valence window:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_lih_gw_pes.py

The no-argument calculation is a Gamma-point smoke run.  The first
dispersion-resolving checkpoint uses the full primitive-cell k mesh:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_lih_gw_pes.py \
       --kmesh 2,2,2 --workers 2 \
       --output /private/tmp/pbc_lih_gw_pes.json

Use ``--stream-pair-batch-size 1`` to reproduce the conservative one-pair
reference build for timing comparisons.  The default automatic batching uses
``--stream-pair-batch-mb 128`` and preserves the same whitened GDF factors.
For an :math:`N_k`-point mesh, ``--stream-pair-batch-size`` :math:`N_k`
retains q-local pair batching while disabling multi-q grouping.

Three timing checkpoints can be compared reproducibly with:

.. code-block:: console

   PYTHONPATH=. python examples/plot_pbc_gdf_batch_benchmark.py \
       /path/to/one_pair.json /path/to/multi_q.json \
       --intermediate /path/to/pair_batch.json \
       --output /private/tmp/pbc_gdf_batch_benchmark.pdf

The driver writes a JSON provenance summary, compressed NPZ arrays, a CSV
integrated spectrum, and PDF/PNG figures.  The NPZ archive retains target-
resolved spectral functions, correlation self-energies, Green functions,
matrix elements, momentum weights, Fermi factors, and detector-broadened
intensities.  Increase ``--kmesh`` and compare the integrated spectrum and
peak positions before treating a result as k-point converged.

The two-pole analytic continuation remains suitable near quasiparticle roots,
but is deliberately not used to reconstruct satellites over a wide spectrum.

Experimental Photoemission Layer
--------------------------------

``KGW.experimental_pes`` applies a first experimental forward model to a
Fermi-referenced GW spectral function.  Energy conservation uses

.. math::

   E_{\mathrm{kin}} = h\nu - \Phi - E_B,

and the current free-electron final-state approximation evaluates the
velocity-gauge matrix element

.. math::

   M_{n k}(K,\mathbf e)
   =
   \mathbf e\cdot\mathbf K\,
   \widetilde{\psi}_{n k}(\mathbf K).

The Bloch-orbital Fourier amplitude is built directly from the native
Gaussian AO Fourier transform.  A Gaussian surface-parallel momentum factor
approximates finite momentum resolution around
:math:`\mathbf K_\parallel=\mathbf k_\parallel+\mathbf G_\parallel`.  The
reported signal is

.. math::

   I(E_B)
   =
   R_{\Delta E} *
   \sum_{n k}
   \frac{2}{N_k}
   |M_{n k}|^2
   P_\parallel
   A_{n k}(E_B)
   f(E_B,T),

where :math:`R_{\Delta E}` is a Gaussian detector-resolution kernel.

.. code-block:: python

   measured = gw.experimental_pes(
       spectral_kwargs={
           "binding_grid": np.linspace(0.0, 80.0, 1601),
           "units": "ev",
           "bands": [0, 1],
       },
       photon_energy=80.0,
       work_function=4.5,
       inner_potential=10.0,
       temperature=300.0,
       energy_resolution=0.2,
       direction=(0.5, 0.0, 0.8660254),
       polarization=(1.0, 0.0, 0.0),
       surface_normal=(0.0, 0.0, 1.0),
       momentum_broadening=0.2,
       units="ev",
   )

The result retains the intrinsic spectrum, matrix elements, momentum weights,
Fermi factors, raw signal, detector-broadened signal, kinetic energies, and
final-state momenta.  It is therefore a replaceable forward-model layer, not
yet a one-step photoemission calculation.  Remaining production components
include surface-matched multiple-scattering final states, inelastic mean free
paths and extrinsic losses, detector angular acceptance, and absolute
cross-section calibration.

Finite-Size Head/Wing Correction
--------------------------------

For a 3D cell volume ``Omega`` and ``N_k`` sampled k points, the correction
approximates the missing small sphere around q=0 with radius

.. math::

   q_c = \left(\frac{6\pi^2}{\Omega N_k}\right)^{1/3}.

For ``coulomb_component="reciprocal_ewald_lr"``, the q=0 body basis is the
reciprocal long-range Coulomb basis already used by the GW response kernel,

.. math::

   L_{tG} = \sqrt{v_G}\rho_t(G),
   \qquad
   v_G = \frac{4\pi}{\Omega |G|^2}.

For a small probe vector ``q_s`` in scaled reciprocal coordinates, the head
transition density is estimated as

.. math::

   q_{ia}(k) =
   \frac{\langle \psi_{ik}|e^{i q_s r}|\psi_{ak}\rangle}{\sqrt{\Omega}}.

At frequency ``u = |omega - epsilon_{nk}|`` the direct-RPA density responses for
``reciprocal_ewald_lr`` are written in PyQED's spin-adapted transition-basis
convention as

.. math::

   \Pi_{GG'}(u) =
   \frac{1}{N_k}\sum_{k,i,a}
   \frac{-\Delta_{ia}(k)}
        {u^2 + \Delta_{ia}(k)^2}
   L_{ia,k,G} L^*_{ia,k,G'},

.. math::

   \Pi_{00}(u) =
   \frac{1}{N_k}\sum_{k,i,a}
   \frac{-\Delta_{ia}(k)}
        {u^2 + \Delta_{ia}(k)^2}
   q^*_{ia}(k) q_{ia}(k),

.. math::

   \Pi_{G0}(u) =
   \frac{1}{N_k}\sum_{k,i,a}
   \frac{-\Delta_{ia}(k)}
        {u^2 + \Delta_{ia}(k)^2}
   L_{ia,k,G} q^*_{ia}(k),

where ``Delta_ia = epsilon_a - epsilon_i``.  The block dielectric pieces are

For ``coulomb_component="gdf"`` or ``"pyscf_gdf"``, the same equations
are evaluated with vector-basis body factors ``L^{X}_{ia,k,P}``, where
``X = \mathrm{GDF}`` for the PyQED factor backend and
``X = \mathrm{PySCF}`` for PySCF GDF.  These vector-basis components use PySCF's
spin-summed response prefactor:

.. math::

   J_{PQ} = (P|Q),
   \qquad
   B^P_{\mu\nu}(\mathbf k,\mathbf k+\mathbf q)
   =
   \sum_{\mathbf R}
   e^{i(\mathbf k+\mathbf q)\cdot\mathbf R}
   (\mu_{\mathbf 0}\nu_{\mathbf R}|P_{\mathbf 0}),

.. math::

   L^a_{\mu\nu}(\mathbf k,\mathbf k+\mathbf q)
   =
   \sum_P B^P_{\mu\nu}(\mathbf k,\mathbf k+\mathbf q)
   (J^{-1/2})_{Pa}.

.. math::

   \Pi^{X}_{PQ}(u) =
   \frac{4}{N_k}\sum_{k,i,a}
   \frac{-\Delta_{ia}(k)}
        {u^2 + \Delta_{ia}(k)^2}
   L^{X}_{ia,k,P}
   L^{X*}_{ia,k,Q}.

The head and wing responses use the analogous ``4/N_k`` prefactor.  This is
the convention used by PySCF PBC KGW and by PyQED's native vector backend.

The block dielectric pieces are

.. math::

   \epsilon^{-1}_{GG'} = [I - \Pi(u)]^{-1}_{GG'},

.. math::

   \epsilon_{00} =
   1 - \frac{4\pi}{|q_s|^2}\Pi_{00},
   \qquad
   \epsilon_{G0} =
   -\frac{\sqrt{4\pi}}{|q_s|}\Pi_{G0},

.. math::

   \epsilon^{-1}_{00} =
   \left(\epsilon_{00}
   - \epsilon^\dagger_{G0}\epsilon^{-1}_{GG'}\epsilon_{G'0}\right)^{-1},

.. math::

   \epsilon^{-1}_{G0} =
   -\epsilon^{-1}_{00}\epsilon^{-1}_{GG'}\epsilon_{G'0}.

The implemented head and wing increments are

.. math::

   \Delta_{00}(u) =
   \frac{2}{\pi} q_c \left(\epsilon^{-1}_{00}(u)-1\right),

.. math::

   \Delta_{G0,nk}(u) =
   \sqrt{\frac{\Omega}{4\pi^3}} q_c^2
   2\,\mathrm{Re}\left[
   L_{nk,nk,G}\epsilon^{-1}_{G0}(u)
   \right].

The diagonal self-energy correction added by PyQED is

.. math::

   \Sigma^{\mathrm{FS}}_{nk}(\omega)
   =
   s_{nk}\left[\Delta_{00}(u) + \Delta_{G0,nk}(u)\right],
   \qquad
   s_{nk} =
   \begin{cases}
   +1, & n k\ \mathrm{occupied},\\
   -1, & n k\ \mathrm{virtual}.
   \end{cases}

For ``gdf`` and ``pyscf_gdf`` the self-energy correction follows the
GDF/vector-basis convention with the half-residue one-sided sign

.. math::

   \Sigma^{\mathrm{FS},X}_{nk}(\omega)
   =
   -\frac{s_{nk}}{2}
   \left[\Delta^{X}_{00}(u)
   + \Delta^{X}_{P0,nk}(u)\right].

This is a small-cell diagnostic correction.  It currently uses the native
finite-q pair Fourier transform or the PySCF k.p AO-gradient expression for
``q_ia``; small-cell quasiparticle energies should still be benchmarked against
PySCF PBC KGW with ``fc=True``.

For BSE screening, the default is to keep the mean-field RPA screening poles.
Pass ``screening_from_qp=True`` to ``KTDA.run``, ``KBSE.run``, or
``q_spectrum`` to rebuild the BSE screening poles from the quasiparticle band
table:

.. code-block:: python

   gw = KGW(mf).evgw(max_cycle=3, direct_scale=1.0)
   bse = KBSE(gw).run(
       q_index=0,
       direct_scale=1.0,
       nroots=2,
       screening_from_qp=True,
   )

q-Resolved Spectra
------------------

Use ``q_spectrum`` to solve all q blocks in the SCF k mesh, or pass
``q_indices`` to select a subset.  Because q-resolved spectra are periodic
objects, ``KTDA.q_spectrum`` and ``KBSE.q_spectrum`` use the periodic route by
default even for Gamma-point periodic references:

.. code-block:: python

   spectrum = KTDA(gw).q_spectrum(
       direct_scale=1.0,
       nroots=1,
       return_vectors=False,
   )

   for q_index, qvec, energy in zip(
       spectrum.q_indices,
       spectrum.qpts,
       spectrum.lowest_roots(),
   ):
       print(q_index, qvec, energy)

The spectrum ``info`` dictionary records provenance such as ``q_indices``,
``uses_qp_energy``, ``uses_screening_energy``, ``coulomb_components``,
kernel scales, and numerical tolerances.
After ``q_spectrum`` returns, the ``KTDA``/``KBSE`` wrapper stores the same
metadata in ``.info`` and exposes q-block energies through
``excitation_energies``.

Current Scope and Limitations
-----------------------------

This is not yet a production-scale periodic GW/BSE implementation.  Important
limitations are explicit:

* closed-shell integer occupations only; metals and fractional occupations are
  rejected;
* native Ewald periodic references only;
* diagonal GW self-energy only for multi-k references;
* dense transition-space full BSE, suitable for small cells; TDA has a
  matrix-free operator at every commensurate momentum and a :math:`q=0`
  optical Haydock path;
* no spin-orbit, unrestricted, finite-temperature, analytic-continuation, or
  force support;
* the default factorized Coulomb kernels use the reciprocal Ewald long-range
  component, not the full short-range plus reciprocal dense Ewald ERI;
* ``coulomb_component="full_ewald"`` is currently a dense small-cell native
  Ewald diagnostic for response kernels, diagonal GW self-energy, and BSE pair
  couplings, not a production large-cell algorithm;
* ``coulomb_component="gdf"`` is a native auxiliary-basis GDF backend with
  streamed reciprocal blocks and compiled shell-blocked short-range image
  sums.  It removes the PySCF runtime dependency and has been validated against
  representation-matched PySCF GDF factors for H2, LiH, He, diamond, and BN
  small cells.  At general off-special k points, the periodic auxiliary
  metric/three-center projection still has an approximately
  :math:`10^{-3}` factor-metric residual, and the 64-k multi-q build remains
  slower than PySCF;
* screened-exchange conventions still need broader reference validation.

For small Gamma-point cells with dense Ewald ERIs, use
``dense_gamma_transition_metric`` to compare the factorized reciprocal metric
against dense ``"reciprocal_ewald_lr"``, ``"short_range_ewald"``,
``"background"``, or ``"full_ewald"`` transition-space Coulomb blocks.
Use ``dense_gamma_orbital_pair_coupling`` and
``dense_gamma_orbital_pair_metric`` for the corresponding transition-to-pair
and pair-to-pair dense Gamma diagnostics.
Use ``full_ewald_transition_metric``, ``full_ewald_orbital_pair_coupling``, and
``full_ewald_orbital_pair_metric`` for the k-compatible native Ewald pair-block
diagnostics used by the periodic kernels.  Use
``gdf_transition_factors`` to build native auxiliary-basis GDF transition and
orbital-pair vectors without importing PySCF.

Focused validation currently lives in ``tests/test_pbc_gw.py`` and exercises
Gamma bridging, multi-k transition spaces, reciprocal factors, direct-RPA
screening, diagonal G0W0/evGW/GnW0, TDA/full BSE, q spectra, and the compact
two-k H2 KRHF example path.  ``examples/pbc_h2_gw_bse.py`` is the runnable
native end-to-end workflow and writes JSON plus PDF/PNG diagnostics.  The
representation-matched PySCF comparison and broader native-GDF convergence
and timing harness is
``examples/pbc_gdf_validation.py``.  Its quick profile covers H2/LiH and a
polarized H2 basis, while the extended profile adds He, diamond, and BN.  It
can scan precision, auxiliary basis, k mesh, GW frequency quadrature, and the
finite-size correction, and optionally run native GDF-KRHF:

.. code-block:: console

   PYTHONPATH=. python examples/pbc_gdf_validation.py \
       --profile quick --native-krhf --strict

PySCF validation cells use the exact bundled PyQED Gaussian contractions and
auxiliary contractions.  This avoids comparing coefficient matrices expressed
in different, mathematically equivalent generalized-contraction rotations.
